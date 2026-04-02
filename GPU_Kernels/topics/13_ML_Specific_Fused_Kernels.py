"""
Fused Kernels — Fused LayerNorm, Bias + Activation & Attention + Dropout
=========================================================================

A "fused kernel" replaces a chain of separate GPU operations with a single
kernel that performs all steps without writing intermediate results to HBM.
Each intermediate tensor that would otherwise round-trip through HBM
represents pure bandwidth waste — the GPU loads the data, transforms it,
stores it back, then loads it again for the next operation.

For an elementwise chain like: LayerNorm → Linear → GELU → Bias Add
    Naïve implementation: 4 separate kernels, 4 HBM load/store cycles.
    Fused implementation: 1 kernel, 1 HBM load, 1 HBM store.
    Bandwidth saving: 8× fewer bytes transferred for the intermediate tensors.

The arithmetic intensity (FLOPs per byte) of elementwise and reduction
kernels is inherently low — typically 1–10 FLOPs/Byte vs the H100 ridge
point of 295 FLOPs/Byte. These operations are irreversibly memory-bound
at the kernel level. The only way to improve their performance is to
reduce the total bytes transferred — which means fusing them.

Three classes of fusion dominate transformer workloads:

    1. FUSED LAYER NORM — the mean, variance, normalisation, and affine
       transform computed in a single warp-reduction kernel. Eliminates
       two intermediate N×d tensors and one extra HBM pass.

    2. FUSED BIAS + ACTIVATION — bias addition, gating, and activation
       functions (GELU, SwiGLU, ReLU) fused into the epilogue of the
       preceding GEMM via cublasLt, or as a standalone fused elementwise.

    3. FUSED ATTENTION + DROPOUT — dropout mask generation, application,
       and the attention output computation merged into Flash Attention's
       tiling loop, so dropout never creates a separate HBM-materialised
       mask tensor.

Understanding fusion means understanding: when is it beneficial, how to
implement it in CUDA/Triton, what the memory traffic model predicts, and
how production libraries (Apex, xFormers, Transformer Engine) implement
each variant.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Fused Kernels — Fused LayerNorm, Bias + Activation & Attention + Dropout"
DISPLAY_NAME = "13 · Fused Kernels"
ICON         = "🔗"
SUBTITLE     = "Fused LayerNorm · Bias+GELU · SwiGLU · Epilogue Fusion · Attention+Dropout"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE MEMORY BANDWIDTH ARGUMENT: WHY FUSION MATTERS

### The Intermediate Tensor Problem

    Consider the feed-forward network (FFN) in a transformer layer:
        x_normed = LayerNorm(x)
        gate     = x_normed @ W_gate    [GEMM 1]
        up       = x_normed @ W_up      [GEMM 2]
        y        = SwiGLU(gate, up)     [elementwise]
        out      = y @ W_down           [GEMM 3]

    NAÏVE execution: 5 separate kernels.
    Between each kernel, the intermediate result must be written to HBM
    and read back by the next kernel. For a 7B model with batch=4, seq=2048:
        x_normed shape: [4, 2048, 4096] → 128 MB per forward pass per layer
        gate/up shapes: [4, 2048, 11008] → 341 MB each
        y shape:        [4, 2048, 11008] → 341 MB

    HBM bytes for intermediates (write + read each): ≈ 2.7 GB per layer.
    At H100 bandwidth (3.35 TB/s): 0.8 ms per layer just in intermediate I/O.
    For 32 layers: 25.6 ms of pure bandwidth waste per forward pass.

    FUSED execution targets:
        - LayerNorm fused: eliminate x_normed write + read (256 MB saved)
        - SwiGLU fused: eliminate gate, up writes + reads (1.37 GB saved)
        - cublasLt epilogue: SwiGLU applied inside W_gate/W_up GEMM (free)

### When Fusion Is Beneficial

    Fusion helps when the FUSED kernel is memory-bandwidth-limited:

    RULE 1 — Small compute-to-memory ratio:
        If the operation does < 10 FLOPs per byte loaded, it is a candidate.
        ElementWise (add, mul, GELU): ~1–3 FLOPs/byte → always fuse.
        Reduction (LayerNorm mean, variance): ~5–15 FLOPs/byte → fuse.
        GEMM at large M, N, K: ~100s FLOPs/byte → NOT a fusion target
        (the GEMM itself is compute-bound; its epilogue should be fused into it).

    RULE 2 — Fuseable boundary:
        Operations fuse well when their data flow is: READ → TRANSFORM → WRITE,
        with no global synchronisation between them.
        LayerNorm WITHIN a single row is fuseable.
        LayerNorm ACROSS rows requires a separate pass (batch statistics).

    RULE 3 — Occupancy is acceptable:
        A fused kernel often uses more registers (all stages share register file).
        If register pressure causes occupancy < 25%, fusion may hurt via
        increased L2/HBM pressure from register spilling.
        Check: ncu metric smsp__occupancy.avg.pct after fusion.

    WHEN NOT TO FUSE:
        Large GEMM outputs feeding into another GEMM — the tensor must live in
        HBM between two independent GEMMs anyway.
        Operations with very different parallelism granularities (e.g., fusing
        a row-reduction with a column-reduction requires synchronisation).

### The HBM Traffic Model for Fusion

    For a chain of K elementwise operations on a tensor of size B bytes:
        Unfused HBM traffic: 2 × K × B  (K reads + K writes)
        Fused HBM traffic:   2 × B        (1 read + 1 write)
        Speedup ≈ K  (proportional to number of ops fused)

    For a chain of 1 reduction + 1 elementwise (LayerNorm):
        Unfused: 4 × B  (read for mean, write mean, read for norm, write output)
        Fused:   2 × B  (1 read for all, 1 write)
        Speedup ≈ 2×

    For Flash Attention (fusion of attention + softmax + dropout):
        Standard: O(N²) HBM traffic for N×N attention matrix
        Flash:    O(N) HBM traffic (no N×N materialisation)
        Speedup grows with N — covered in module 30.


##### PART 2 — FUSED LAYER NORM: WELFORD ALGORITHM & SMEM REDUCTION

### What Layer Norm Computes

    Given input x of shape [B, T, D] (batch, tokens, hidden_dim):

    For each row r = x[b, t, :]:
        mean_r   = (1/D) × Σ_i  x_i
        var_r    = (1/D) × Σ_i  (x_i - mean_r)²
        x_norm_r = (x_r - mean_r) / sqrt(var_r + ε)
        y_r      = γ ⊙ x_norm_r + β        (learnable affine transform)

    ε = 1e-5 (prevents division by zero).
    γ, β ∈ ℝ^D (per-channel scale and bias, learned).

### Two Passes Are Required... Unless You Use Welford

    NAÏVE TWO-PASS:
        Pass 1: read x, compute mean. Write mean.
        Pass 2: read x, read mean, compute variance and normalisation.
        HBM: 3× the input tensor.

    ONE-PASS with WELFORD'S ONLINE ALGORITHM:
        Maintains running mean and variance in a single pass over the row.
        No intermediate mean tensor written to HBM.

        Running state for element i:
            count = i + 1
            delta1 = x[i] - mean
            mean  += delta1 / count
            delta2 = x[i] - mean
            M2    += delta1 * delta2      (accumulates sum of squared deviations)
        Final variance: M2 / count

    ADVANTAGE: numerically more stable than the naïve formula
        var = mean(x²) - mean(x)²
    because that formula suffers catastrophic cancellation when var << mean².
    Welford avoids subtraction of large nearly-equal numbers.

### The GPU Implementation: One Block Per Row

    MAPPING: one CUDA threadblock handles one row (one token's activations).
        For D=4096: 128 threads × 32 elements each = 4096 elements per block.

    SMEM WARP REDUCTION:
        Phase 1: each thread computes partial (count, mean, M2) over its 32 elements.
        Phase 2: warp-level Welford merge via __shfl_down_sync (5 steps, 32 lanes).
        Phase 3: lane 0 of each warp writes to SMEM.
        Phase 4: first warp reads from SMEM, reduces to single (mean, var).
        Phase 5: broadcast mean and var to all threads via SMEM.
        Phase 6: each thread normalises and applies γ, β.

    HBM TRAFFIC:
        Read x:  B × T × D × 2 bytes (FP16)
        Write y: B × T × D × 2 bytes (FP16)
        No intermediate tensors.
        Total: 2 × B × T × D × 2 bytes  (= 2× the input size)

    SMEM USAGE: 2 × num_warps × (mean + M2) = 2 × 4 × 2 × 4 = 64 bytes.
    Negligible. The bottleneck is HBM bandwidth, not SMEM.

### Fusing the Residual Add

    In transformers, LayerNorm is typically followed by or preceded by a
    residual connection: y = LayerNorm(x + residual).

    UNFUSED: add kernel + LayerNorm kernel.
    FUSED: read x and residual in the same pass, compute x+residual in registers,
    then immediately apply Welford + normalisation. One extra read (residual),
    same number of kernel launches.

    apex.contrib.layer_norm and NVIDIA's TransformerEngine both implement this.
    The fused version saves the write + re-read of (x + residual): 2 × B×T×D×2 bytes.
    For the 7B model, batch=4, seq=2048: saves 2 × 128 MB = 256 MB per layer.

### RMSNorm: Simplified LayerNorm Without Mean

    RMSNorm (Zhang & Sennrich, 2019) skips the mean subtraction:
        rms_r   = sqrt((1/D) × Σ_i x_i²)
        y_r     = (x_r / rms_r) × γ

    No mean computation → simpler reduction (just sum of squares).
    Slightly fewer FLOPs, same HBM traffic.
    Used by: Llama, Mistral, Falcon, Gemma.

    FUSED IMPLEMENTATION:
        One pass over x: accumulate sum of squares.
        Warp reduction: reduce sum of squares to scalar.
        Normalise: divide each element by rms, multiply by γ.
        Total: same HBM pattern as fused LayerNorm, fewer FLOPs.


##### PART 3 — FUSED BIAS + ACTIVATION: GELU, SWIGLU & REGLU
────────────────

### Activation Functions in Modern LLMs

    GELU (Gaussian Error Linear Unit):
        GELU(x) = x × Φ(x) = x × 0.5 × (1 + erf(x/√2))
        Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        Used by: BERT, GPT-2, ViT.
        FLOPs per element: ~15 (tanh approximation).

    SwiGLU (Swish Gated Linear Unit) — Shazeer (2020):
        SwiGLU(x, y) = Swish(x) ⊙ y = (x × σ(x)) ⊙ y
        Two inputs x (gate) and y (value), both from separate linear projections.
        Used by: PaLM, Llama, Mistral, Gemma, Falcon.
        FLOPs per element: ~6 (sigmoid + multiply + hadamard).

    ReGLU (Rectified Gated Linear Unit):
        ReGLU(x, y) = ReLU(x) ⊙ y
        FLOPs per element: ~2 (max + multiply). Fastest.

    GeGLU (GELU-Gated Linear Unit):
        GeGLU(x, y) = GELU(x) ⊙ y
        Used by: T5, Flan-T5.

### The Bias + Activation Fusion Pattern

    In an FFN layer, after W_gate and W_up GEMMs:
        gate  = x @ W_gate + b_gate      [GEMM + bias add]
        value = x @ W_up   + b_up        [GEMM + bias add]
        y     = SwiGLU(gate, value)      [activation + gating]

    UNFUSED (3 operations after GEMM):
        1. gate kernel (add b_gate)
        2. value kernel (add b_up)
        3. SwiGLU kernel (σ(gate) × gate × value)
        3 HBM round-trips for gate and value tensors.

    FUSED as cublasLt epilogue:
        cublasLt CUBLASLT_EPILOGUE_GELU_BIAS:
            Performs: output = GELU(GEMM_result + bias)
        For SwiGLU: fuse with custom cublasLt epilogue or Triton kernel.

    FUSED as standalone kernel (Triton / CUDA):
        Read gate and value from HBM ONCE.
        Compute bias add + SwiGLU in registers.
        Write output ONCE.
        2 reads (gate, value) + 1 write (output) — half the HBM traffic
        of the 3-operation unfused chain.

### SwiGLU CUDA Implementation Sketch

    // Fused bias + SwiGLU kernel
    // gate_raw:  [B*T, D_ff]  (output of W_gate GEMM)
    // value_raw: [B*T, D_ff]  (output of W_up GEMM)
    // bias_gate, bias_value:  [D_ff]
    // output: [B*T, D_ff/2] or [B*T, D_ff] depending on convention

    __global__ void fused_bias_swiglu(
        const float* gate_raw, const float* value_raw,
        const float* bias_gate, const float* bias_value,
        float* output, int B_T, int D_ff)
    {
        int row = blockIdx.x;
        int col = threadIdx.x;
        if (row >= B_T || col >= D_ff) return;

        int idx = row * D_ff + col;

        // Load gate and value, add biases (all from HBM, once)
        float g = gate_raw[idx]  + bias_gate[col];
        float v = value_raw[idx] + bias_value[col];

        // Swish: g * sigmoid(g)
        float swish_g = g * (1.0f / (1.0f + expf(-g)));

        // SwiGLU: swish(gate) × value
        output[idx] = swish_g * v;
    }
    // ONE kernel. ONE load of gate, ONE load of value. ONE write of output.
    // vs 3 kernels (bias_gate, bias_value, swiglu) with 6 HBM ops.

### The Backward Pass of Fused SwiGLU

    Fused backward: given d_out [B*T, D_ff], compute d_gate and d_value.

    Let g = gate + b_gate, v = value + b_value.
    Let σg = sigmoid(g), sg = g * σg  (Swish of g).

    Forward: y = sg × v

    Backward:
        d_value = sg × d_out                       (gradient through v)
        d_sg    = v  × d_out                        (gradient through sg)
        d_g     = d_sg × (σg + g × σg × (1 - σg)) (Swish gradient)
                = d_sg × σg × (1 + g × (1 - σg))

    FUSED BACKWARD: one kernel reads (gate_raw, value_raw, d_out),
    computes d_gate and d_value, writes both.
    4 HBM ops (3 reads + 2 writes ≈ 5 ops) vs
    unfused (6 reads + 4 writes = 10 ops for 3 backward kernels).

### Triton Implementation Advantages

    PyTorch/Triton allows writing the fused kernel in Python:
        @triton.jit
        def swiglu_kernel(gate_ptr, value_ptr, output_ptr,
                          B_T, D_ff, BLOCK_SIZE: tl.constexpr):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK_SIZE)

            gate  = tl.load(gate_ptr  + row * D_ff + cols)
            value = tl.load(value_ptr + row * D_ff + cols)

            sigmoid_g = 1.0 / (1.0 + tl.exp(-gate))
            swish_g   = gate * sigmoid_g
            output    = swish_g * value

            tl.store(output_ptr + row * D_ff + cols, output)

    Triton automatically:
        - Vectorises loads/stores to use 128-bit memory transactions
        - Fuses into one PTX kernel with no intermediate allocations
        - Handles block padding for non-tile-aligned dimensions


##### PART 4 — FUSED ATTENTION + DROPOUT: THE THREE DROPOUT PATTERNS

### The Dropout Problem for Attention

    Standard attention with dropout:
        P = softmax(QKᵀ/√d)
        P_drop = dropout(P, p=0.1)      ← generate mask [N×N], apply it
        O = P_drop @ V

    NAÏVE dropout requires:
        1. Materialise P [N×N] in HBM
        2. Generate random mask [N×N] in HBM (cuRAND)
        3. Apply mask: P_drop = P × mask / (1-p)
        4. Multiply by V: O = P_drop @ V

    This re-introduces the very O(N²) memory problem that Flash Attention solved.
    All benefits of Flash Attention are lost if dropout forces P to HBM.

### Fusing Dropout Inside Flash Attention

    Flash Attention integrates dropout into the tiling loop:
    For each (Q_tile, K_tile, V_tile) triple:
        S = Q_tile @ K_tile.T / √d
        P_tile = exp(S - m) / d
        // GENERATE MASK on-the-fly in registers:
        mask = curand_uniform() > p           [BLOCK_Q × BLOCK_KV bitmask]
        P_tile_drop = P_tile × mask / (1-p)   [apply mask, scale]
        O_acc += P_tile_drop @ V_tile          [accumulate output]

    The mask is NEVER written to HBM. It exists only in registers for the
    duration of the tile computation. This is possible because:
        1. Each block of attention is processed once (tiles don't overlap for output)
        2. The backward pass can regenerate the same mask using a SAVED SEED
           (philox RNG is deterministic given seed + offset)

    BACKWARD PASS WITH FUSED DROPOUT:
        Save: the random seed and offset for each (block_row, block_col) tile.
        Backward: regenerate identical mask from seed + offset, recompute P_tile,
        apply mask, continue with the gradient computation.
        No mask tensor ever stored in HBM.

### The Philox RNG for Parallel Dropout

    Standard cuRAND uses a stateful RNG — not suitable for parallel,
    tile-level generation where all tiles must produce independent but
    deterministic random numbers.

    PHILOX (Salmon et al., 2011):
        Stateless counter-based RNG. Given (key, counter), produces random bits.
        Any tile can independently generate its portion of the random stream
        by setting counter = tile_row × N/BLOCK + tile_col.
        Backward pass regenerates same values: counter → same random bits.

    FLASH ATTENTION DROPOUT SEED/OFFSET PROTOCOL:
        At the start of the forward pass: sample a random seed from host RNG.
        Each threadblock uses: seed = global_seed, offset = block_id × tile_size.
        Forward saves: seed and total_offset (one pair per head, not N×N).
        Backward reads: seed and offset, regenerates mask incrementally.

    MEMORY SAVED:
        Standard: N × N × 1 bit per head per layer = N²/8 bytes per head
        Flash fused: 2 × sizeof(uint64_t) = 16 bytes per head
        For N=8192, 32 heads, 32 layers: 8192²/8 × 32 × 32 = 1 GB saved.

### Three Dropout Patterns in Transformers

    PATTERN 1 — Attention weight dropout (P_drop = dropout(softmax(S))):
        Standard: materialises N×N mask.
        Fused: tile-level Philox mask inside Flash Attention loop.
        Impact: O(N²) memory → O(1) memory.

    PATTERN 2 — Post-attention dropout (output = dropout(O)):
        Standard: standalone dropout kernel on [B, H, N, d] output.
        Fused: generate mask inside Flash Attention's final output write.
        Flash attention 2/3 implements this as a separate epilogue within the kernel.
        Memory saved: N × d × H × B bits (the output dropout mask).

    PATTERN 3 — FFN dropout:
        Applied to FFN intermediate activations [B, T, D_ff].
        For D_ff=11008: mask is B × T × 11008 bits ≈ 43 MB per layer.
        Fused with SwiGLU kernel: generate mask in registers, apply to SwiGLU output.
        Memory saved: 43 MB per layer × 32 layers = 1.37 GB per forward pass.


##### PART 5 — WELFORD REDUCTION: DERIVATION AND GPU WARP REDUCTION PATTERN

### Welford's Online Variance Algorithm

    For a stream of elements x_0, x_1, ..., x_{N-1}, maintain:
        (n, mean, M2)  where M2 = Σ (x_i - mean)²

    UPDATE when a new element x arrives:
        n    += 1
        delta  = x - mean
        mean  += delta / n
        delta2 = x - mean           (updated mean!)
        M2    += delta * delta2

    FINAL: variance = M2 / n  (population), or M2 / (n-1) (sample)

    NUMERICAL STABILITY PROOF:
        The term delta × delta2 avoids computing mean(x²) - mean(x)².
        Even if mean >> variance (values clustered near large baseline),
        delta and delta2 are both small deviations — no catastrophic cancellation.
        The naïve formula var = E[x²] - E[x]² subtracts two large nearly-equal
        numbers, losing precision proportional to (mean/std)².

### Welford Merge Operator (for Parallel Reduction)

    Two partial accumulators (n_a, mean_a, M2_a) and (n_b, mean_b, M2_b):

        delta  = mean_b - mean_a
        n_ab   = n_a + n_b
        mean   = mean_a + delta × n_b / n_ab
        M2     = M2_a + M2_b + delta² × n_a × n_b / n_ab

    THIS IS ASSOCIATIVE: any binary tree of merges gives the same (mean, M2).
    Consequence: Welford maps directly onto GPU warp reduction (like scan in module 23).

    WARP REDUCTION WITH WELFORD:
        Each lane accumulates (n=ELEMS_PER_LANE, mean, M2) for its elements.
        __shfl_down_sync merges pairs of lanes (offset 16, 8, 4, 2, 1).
        After 5 shfl_down steps: lane 0 holds the warp-level (mean, var).
        Write to SMEM: 2 values (mean, M2) per warp.
        First warp reads SMEM, reduces across warps.
        Broadcast final (mean, var) to all threads via SMEM.

    SMEM USAGE: 2 × num_warps × 4 bytes = 64 bytes for 8-warp block.
    This is negligible. The kernel is entirely HBM bandwidth-limited.

### The Full Fused LayerNorm Warp Reduction Pseudocode

    // One block per row (D elements). D/THREADS elements per thread.
    __shared__ float smem_mean[NUM_WARPS];
    __shared__ float smem_M2  [NUM_WARPS];
    __shared__ float smem_count[NUM_WARPS];

    // Phase 1: thread-level Welford over D/THREADS elements
    float n=0, mean=0, M2=0;
    for (int i = lane_id; i < D; i += WARP_SIZE * NUM_WARPS) {
        float x = input[row * D + i];
        float delta = x - mean; n++;
        mean += delta / n;
        M2 += delta * (x - mean);
    }

    // Phase 2: warp-level Welford merge via __shfl_down_sync
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        float n_b    = __shfl_down_sync(0xFFFFFFFF, n,    offset);
        float mean_b = __shfl_down_sync(0xFFFFFFFF, mean, offset);
        float M2_b   = __shfl_down_sync(0xFFFFFFFF, M2,   offset);
        // merge (n, mean, M2) with (n_b, mean_b, M2_b)
        float delta = mean_b - mean;
        float n_ab  = n + n_b;
        M2   += M2_b + delta * delta * n * n_b / n_ab;
        mean += delta * n_b / n_ab;
        n     = n_ab;
    }

    // Phase 3: lane 0 of each warp writes to SMEM
    if (lane_id == 0) {
        smem_mean[warp_id] = mean;
        smem_M2[warp_id] = M2;
    }
    __syncthreads();

    // Phase 4: first warp reduces across warps (from SMEM)
    // (same Welford merge, now over NUM_WARPS values)
    // Broadcast final mean and var via SMEM

    // Phase 5: normalise and apply affine
    float inv_std = rsqrtf(M2 / D + eps);
    for (int i = ...) {
        float x_norm = (input[...] - mean) * inv_std;
        output[...] = gamma[i] * x_norm + beta[i];
    }


##### PART 6 — FUSED KERNEL PATTERNS IN PRODUCTION LIBRARIES

### NVIDIA Apex / FusedAdam / FusedLayerNorm

    apex.normalization.FusedLayerNorm:
        Implements fused LayerNorm in CUDA with optional residual add.
        ~2× faster than PyTorch's nn.LayerNorm on A100 for typical shapes.
        API: identical to nn.LayerNorm (drop-in replacement).
        FP16/BF16 compute with FP32 accumulation for stability.

    apex.optimizers.FusedAdam:
        Fuses the Adam update, L2 regularisation, and gradient unscaling
        into a single kernel per parameter group.
        Eliminates 4 separate kernels: gradient unscale, weight decay add,
        Adam m1 update, Adam m2 update.
        Speedup: ~10–20% of optimizer step time on A100.

### NVIDIA TransformerEngine (TE)

    TransformerEngine provides hardware-accelerated transformer primitives:
        te.LayerNorm:       FP8-aware LayerNorm with fused residual
        te.Linear:          FP8 GEMM with cublasLt + epilogue fusion
        te.MultiheadAttention: Flash Attention + dropout + residual
        te.TransformerLayer: Complete pre-norm transformer layer,
                             all operations fused end-to-end

    TE's transformer layer achieves:
        Forward: LayerNorm(residual add) → FP8 GEMM (QKV) → Flash Attention
                 → FP8 GEMM (proj) → residual add → LayerNorm → FP8 GEMM ×2
                 (gate+up) → SwiGLU → FP8 GEMM (down)
        All in one forward call, with minimal HBM intermediate traffic.
        Performance: ~2–3× faster than unfused PyTorch for large models.

### xFormers Fused Operations

    xFormers (Meta) provides:
        xformers.ops.memory_efficient_attention:
            FlashAttention or memory-efficient attention for any GPU.
            Handles variable-length sequences and custom biases.

        xformers.ops.swiglu:
            Fused SwiGLU with optional bias, forward and backward.
            Triton-based, auto-tunes block sizes per input shape.

        xformers.components.feedforward.FusedMLP:
            Complete fused FFN: W1/W2 GEMM + SwiGLU + W3 GEMM.
            Uses cublasLt epilogue for bias-GELU fusion.

### PyTorch torch.compile and Inductor Fusion

    PyTorch 2.0+ torch.compile() automatically fuses elementwise and
    reduction operations via the Inductor backend:
        @torch.compile
        def fused_ops(x, gamma, beta, residual):
            x = x + residual               # ─┐
            mean = x.mean(-1, keepdim=True) #  │ Inductor detects these can
            var = x.var(-1, keepdim=True)   #  │ be fused into one kernel
            x = (x - mean) / (var + 1e-5)  #  │ with single HBM pass
            return gamma * x + beta         # ─┘

    Inductor's horizontal and vertical fusion rules:
        Vertical fusion: consumer reads producer's output → fuse into one pass.
        Horizontal fusion: two ops on the same input → load input once for both.
        Reduction fusion: reduce then elementwise → one pass, store only final result.

    Limitations: Inductor cannot fuse across GEMM boundaries (GEMMs are separate).
    GEMM epilogue fusion requires cublasLt or custom Triton kernel.


##### PART 7 — MEASURING FUSION BENEFIT: NCU WORKFLOW FOR FUSED KERNELS

### Expected NCU Metric Changes After Fusion

    BEFORE fusion (separate kernels):
        dram__bytes_read.sum:       2× the unfused expected (extra reads)
        dram__bytes_write.sum:      2× expected (intermediate writes)
        sm__throughput.avg.pct:     20–40% (memory-bound, low SM utilisation)
        Kernel count in nsys:       K kernels for K ops

    AFTER fusion (one kernel):
        dram__bytes_read.sum:       1× expected (no re-reads)
        dram__bytes_write.sum:      1× (no intermediate writes)
        sm__throughput.avg.pct:     40–80% (better bandwidth utilisation)
        Kernel count in nsys:       1 kernel for K ops

### The Three-Step Fusion Verification Workflow

    STEP 1 — Establish baseline with nsys:
        nsys profile model.py
        Look for: multiple consecutive small kernels with short durations.
        Candidate for fusion: kernels < 10 µs with same input/output shapes.

    STEP 2 — Measure HBM bytes with ncu before fusion:
        ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
            --kernel-name "layernorm_fwd" -o before.ncu-rep model.py
        Compare to theoretical minimum: 2 × N × D × sizeof(FP16) bytes.
        Ratio (actual/theoretical) > 2: fusion opportunity.

    STEP 3 — Verify after fusion:
        ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
            --kernel-name "fused_layernorm_residual" -o after.ncu-rep model.py
        Expected: HBM bytes ≈ theoretical minimum.
        Compare wall time: ncu --import-before before.ncu-rep --import-after after.ncu-rep

### Performance Limits of Fused Kernels

    A fused kernel is still bounded by the memory bandwidth roofline.
    After fusion, the remaining HBM traffic is the UNAVOIDABLE MINIMUM:
        Read input once (I bytes).
        Write output once (O bytes).
        Minimum time = (I + O) / HBM_bandwidth.

    If the fused kernel is slower than this minimum:
        Register spilling: kernel uses too many registers, spills to L2/HBM.
        Fix: reduce register count (smaller unroll factor, simpler intermediate).
        Or: split into two passes (the computation is too complex for one kernel).

    OPTIMAL FUSION DEPTH:
        Typical fused kernel: 3–6 elementwise ops before diminishing returns.
        LayerNorm: complex enough (Welford reduction) to benefit from 1-pass.
        SwiGLU: simple enough (5 elementwise ops) to fuse trivially.
        SwiGLU + LayerNorm + dropout: may cause register pressure.
        Profile occupancy (smsp__occupancy.avg.pct) after fusion.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · HBM Traffic Model — Fusion Savings Across Kernel Chains": {
        "description": (
            "Compute the exact HBM bytes for unfused vs fused kernel chains "
            "across the main transformer operations: LayerNorm, residual add, "
            "bias add, GELU, SwiGLU, dropout, and attention dropout. Show the "
            "traffic reduction for every fusion opportunity in a transformer "
            "layer. Calculate wall-clock speedup at H100 and A100 bandwidth. "
            "Identify which fusions give the largest absolute savings."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  HBM TRAFFIC MODEL — Fusion Savings Across Kernel Chains")
print("=" * 68)
print()

# Hardware specs
HBM_H100 = 3350.0   # GB/s
HBM_A100 = 2000.0   # GB/s
HBM_V100 =  900.0   # GB/s
FP16_BYTES = 2
FP32_BYTES = 4


def bytes_str(b):
    if b >= 1e9:  return f"{b/1e9:.2f} GB"
    if b >= 1e6:  return f"{b/1e6:.2f} MB"
    return f"{b/1e3:.2f} KB"

def time_us(b, bw_gbs):
    return b / (bw_gbs * 1e9) * 1e6


# Model dimensions: Llama-2-7B style
B, T, D     = 4, 2048, 4096
D_FF        = 11008
N_HEADS     = 32
HEAD_DIM    = D // N_HEADS
N           = T  # seq_len

elems_BD    = B * T * D
elems_BD_FF = B * T * D_FF
elems_BHN_D = B * N_HEADS * N * HEAD_DIM   # = B * T * D


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Pre-attention LayerNorm fusion opportunities
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Pre-Attention LayerNorm + Residual Add")
print("━" * 68)
print()

print("  Ops in sequence: x2 = residual_add(x1, x_res), y = LayerNorm(x2)")
print(f"  Tensor size: B={B}, T={T}, D={D} → {elems_BD:,} elements × {FP16_BYTES}B = "
      f"{bytes_str(elems_BD * FP16_BYTES)}")
print()

operations_ln = [
    ("Residual add",           2, 1, "read x1 + x_res, write x2"),
    ("LayerNorm mean pass",    1, 1, "read x2, write mean[B,T]"),
    ("LayerNorm norm pass",    1, 1, "read x2+mean, write y"),
]
unfused_total = sum((r+w) * elems_BD * FP16_BYTES for _n,r,w,_nn in operations_ln)

fused_ops_ln = [
    ("Fused residual + LN",    2, 1, "read x1 + x_res, write y (one pass)"),
]
fused_total = sum((r+w) * elems_BD * FP16_BYTES for _n,r,w,_nn in fused_ops_ln)

print(f"  {'Operation':<30}  {'Reads':>6}  {'Writes':>7}  {'HBM ops':>8}  {'Notes'}")
print("  " + "─" * 70)
print("  UNFUSED:")
for name, reads, writes, notes in operations_ln:
    hbm = (reads + writes) * elems_BD * FP16_BYTES
    print(f"    {name:<28}  {reads:>6}×  {writes:>6}×  {bytes_str(hbm):>10}  {notes}")
print(f"    {'Total unfused':<28}  {'':>6}   {'':>6}   {bytes_str(unfused_total):>10}")
print()
print("  FUSED:")
for name, reads, writes, notes in fused_ops_ln:
    hbm = (reads + writes) * elems_BD * FP16_BYTES
    print(f"    {name:<28}  {reads:>6}×  {writes:>6}×  {bytes_str(hbm):>10}  {notes}")
print(f"    {'Total fused':<28}  {'':>6}   {'':>6}   {bytes_str(fused_total):>10}")
print()
saved    = unfused_total - fused_total
speedup  = unfused_total / fused_total
t_un_us  = time_us(unfused_total, HBM_H100)
t_fu_us  = time_us(fused_total,   HBM_H100)
print(f"  Bytes saved:  {bytes_str(saved)}  ({saved/unfused_total*100:.0f}%)")
print(f"  Speedup:      {speedup:.2f}×")
print(f"  H100 time:    {t_un_us:.2f} µs → {t_fu_us:.2f} µs")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: FFN SwiGLU fusion
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — FFN SwiGLU: Bias + Gate + Activation Fusion")
print("━" * 68)
print()

print(f"  After W_gate and W_up GEMMs, gate/up tensors: B×T×D_FF = "
      f"{B}×{T}×{D_FF} = {elems_BD_FF:,} FP16 elements = "
      f"{bytes_str(elems_BD_FF * FP16_BYTES)} each")
print()

ops_swiglu_unfused = [
    ("Add bias_gate to gate",    1, 1, "read gate, write gate+b"),
    ("Add bias_up to up",        1, 1, "read up, write up+b"),
    ("SwiGLU(gate, up)",         2, 1, "read gate+b and up+b, write y"),
]
un_swiglu = sum((r+w)*elems_BD_FF*FP16_BYTES for _n,r,w,_nn in ops_swiglu_unfused)

ops_swiglu_fused = [
    ("Fused bias+SwiGLU",        2, 1, "read gate+up raw, write y (no bias tensors)"),
]
fu_swiglu = sum((r+w)*elems_BD_FF*FP16_BYTES for _n,r,w,_nn in ops_swiglu_fused)

print(f"  {'Operation':<30}  {'Reads':>6}  {'Writes':>6}  {'HBM bytes'}")
print("  " + "─" * 56)
print("  UNFUSED:")
for name, reads, writes, notes in ops_swiglu_unfused:
    hbm = (reads + writes) * elems_BD_FF * FP16_BYTES
    print(f"    {name:<28}  {reads:>6}×  {writes:>6}×  {bytes_str(hbm):>12}")
print(f"    {'Total unfused':<28}  {'':>6}   {'':>6}   {bytes_str(un_swiglu):>12}")
print()
print("  FUSED (standalone kernel or cublasLt epilogue):")
for name, reads, writes, notes in ops_swiglu_fused:
    hbm = (reads + writes) * elems_BD_FF * FP16_BYTES
    print(f"    {name:<28}  {reads:>6}×  {writes:>6}×  {bytes_str(hbm):>12}")
print(f"    {'Total fused':<28}  {'':>6}   {'':>6}   {bytes_str(fu_swiglu):>12}")
print()
print(f"  Savings: {bytes_str(un_swiglu-fu_swiglu)}  ({(un_swiglu-fu_swiglu)/un_swiglu*100:.0f}%)")
print(f"  Speedup: {un_swiglu/fu_swiglu:.2f}×")
print(f"  H100 time: {time_us(un_swiglu, HBM_H100):.2f} µs → {time_us(fu_swiglu, HBM_H100):.2f} µs")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Dropout mask fusion
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Dropout Mask Fusion: Attention and FFN Dropout")
print("━" * 68)
print()

# Attention weight dropout (N×N mask)
nn_mask_bytes   = N * N * N_HEADS * B * 1   # 1 bit per element → store as uint8

# Flash attention fused: only save seed (8 bytes per head)
flash_seed_bytes = N_HEADS * B * 2 * 8      # seed + offset per head × batch

print("  ATTENTION WEIGHT DROPOUT (N×N mask):")
print(f"    Standard: generate + store mask [B,H,N,N] = "
      f"{B}×{N_HEADS}×{N}×{N} bits = {bytes_str(nn_mask_bytes)}")
print(f"    Flash fused: store only RNG seed per head = "
      f"{bytes_str(flash_seed_bytes)}")
print(f"    Memory saved: {bytes_str(nn_mask_bytes - flash_seed_bytes)}  "
      f"({(1-flash_seed_bytes/nn_mask_bytes)*100:.2f}%)")
print()

# FFN dropout mask
ffn_mask_bytes  = elems_BD_FF // 8          # 1 bit per element
print("  FFN DROPOUT (D_ff mask):")
print(f"    Standard: mask [B,T,D_FF] = "
      f"{B}×{T}×{D_FF} bits = {bytes_str(ffn_mask_bytes)}")
print(f"    Fused with SwiGLU: mask generated in registers, never written to HBM")
print(f"    Memory saved: {bytes_str(ffn_mask_bytes)} per layer")
print()

# Total across 32 layers
N_LAYERS = 32
total_attn_mask_bytes = nn_mask_bytes * N_LAYERS
total_ffn_mask_bytes  = ffn_mask_bytes * N_LAYERS
total_saved_bytes     = total_attn_mask_bytes + total_ffn_mask_bytes
print(f"  TOTAL ACROSS {N_LAYERS} LAYERS:")
print(f"    Attention masks unfused: {bytes_str(total_attn_mask_bytes)}")
print(f"    FFN masks unfused:       {bytes_str(total_ffn_mask_bytes)}")
print(f"    Total mask memory saved: {bytes_str(total_saved_bytes)}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Full transformer layer fusion budget
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Full Layer Fusion Budget: Total HBM Savings")
print("━" * 68)
print()

# All fusion opportunities per transformer layer
fusion_ops = [
    ("Residual + LN (pre-attn)",     unfused_total - fused_total,          1),
    ("Residual + LN (pre-FFN)",      unfused_total - fused_total,          1),
    ("FFN SwiGLU bias fusion",       un_swiglu - fu_swiglu,                1),
    ("Attention dropout (fused)",    nn_mask_bytes,                        1),
    ("FFN dropout (fused)",          ffn_mask_bytes,                       1),
    ("Q/K/V proj bias epilogue",     elems_BD * FP16_BYTES * 3,           0.5),
    ("Output proj bias epilogue",    elems_BD * FP16_BYTES,               0.5),
]

total_unfused_layer = sum(s * f for _, s, f in fusion_ops) + sum(s for _, s, _ in fusion_ops)
total_savings = sum(s * f for _, s, f in fusion_ops)

print(f"  Per-layer fusion opportunities (B={B}, T={T}, D={D}, D_FF={D_FF}):")
print()
print(f"  {'Fusion':<38}  {'Bytes saved':>14}  {'Fraction':>9}")
print("  " + "─" * 64)
for name, saving, fraction in fusion_ops:
    eff_saving = saving * fraction
    print(f"  {name:<38}  {bytes_str(eff_saving):>14}  {fraction:>8.0%}")
print()
print(f"  Total savings per layer:  {bytes_str(total_savings)}")
print(f"  Total savings, {N_LAYERS} layers:   {bytes_str(total_savings * N_LAYERS)}")
print()
t_saved_ms = time_us(total_savings * N_LAYERS, HBM_H100) / 1000
print(f"  Wall-clock time saved (H100, one forward pass): {t_saved_ms:.2f} ms")
print(f"  H100 HBM bandwidth: {HBM_H100} GB/s")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Fused LayerNorm — Welford Algorithm, Warp Reduction & Verification": {
        "description": (
            "Implement the complete fused LayerNorm using Welford's online algorithm. "
            "Show the running (n, mean, M2) state update element by element. Implement "
            "the parallel Welford merge operator for warp reduction. Simulate the "
            "full GPU warp reduction pipeline: thread → warp → cross-warp SMEM. "
            "Verify against PyTorch LayerNorm. Compare numerical stability of "
            "Welford vs naïve variance formulas at critical edge cases."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FUSED LAYER NORM — Welford Algorithm, Warp Reduction & Verification")
print("=" * 68)
print()

np.random.seed(42)
WARP_SIZE = 32


# ─────────────────────────────────────────────────────────────────────
# Welford utilities
# ─────────────────────────────────────────────────────────────────────

def welford_update(n, mean, M2, new_val):
    """Update (n, mean, M2) with one new element."""
    n      += 1
    delta   = new_val - mean
    mean   += delta / n
    delta2  = new_val - mean
    M2     += delta * delta2
    return n, mean, M2

def welford_merge(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    """Merge two partial (n, mean, M2) accumulators."""
    if n_a == 0: return n_b, mean_b, M2_b
    if n_b == 0: return n_a, mean_a, M2_a
    delta  = mean_b - mean_a
    n_ab   = n_a + n_b
    mean   = mean_a + delta * n_b / n_ab
    M2     = M2_a + M2_b + delta**2 * n_a * n_b / n_ab
    return n_ab, mean, M2

def welford_finalize(n, mean, M2, eps=1e-5):
    """Return (mean, variance) from Welford accumulator."""
    return mean, M2 / n + eps


def naive_variance(x, eps=1e-5):
    """Naïve formula: var = E[x²] - E[x]²  (LESS numerically stable)."""
    mean   = np.mean(x)
    var    = np.mean(x**2) - mean**2
    return mean, max(var, 0) + eps

def layernorm_ref(x, gamma, beta, eps=1e-5):
    """Reference LayerNorm using numpy."""
    mean = x.mean()
    var  = x.var() + eps
    return gamma * (x - mean) / np.sqrt(var) + beta


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Welford sequential trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Welford Online Variance: Element-by-Element Trace")
print("━" * 68)
print()

x_trace = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

n_w, mean_w, M2_w = 0, 0.0, 0.0
print(f"  Input: {x_trace.tolist()}")
print()
print(f"  {'i':>3}  {'x[i]':>7}  {'n':>4}  {'delta':>8}  "
      f"{'mean':>10}  {'M2':>12}  {'var so far':>12}")
print("  " + "─" * 62)

for i, xi in enumerate(x_trace):
    delta_before = xi - mean_w
    n_w, mean_w, M2_w = welford_update(n_w, mean_w, M2_w, xi)
    var_so_far = M2_w / n_w if n_w > 1 else 0
    print(f"  {i:>3}  {xi:>7.1f}  {n_w:>4}  {delta_before:>8.4f}  "
          f"{mean_w:>10.6f}  {M2_w:>12.6f}  {var_so_far:>12.6f}")

true_var  = np.var(x_trace)
welf_var  = M2_w / n_w
print()
print(f"  Welford variance: {welf_var:.8f}")
print(f"  NumPy  variance:  {true_var:.8f}")
print(f"  Error: {abs(welf_var - true_var):.2e}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Welford merge — parallel reduction correctness
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Welford Merge: Parallel Reduction Equivalence")
print("━" * 68)
print()

D = 256
x_full = np.random.randn(D) * 5.0 + 10.0  # clustered near 10

# Sequential full-vector Welford
n_seq, mean_seq, M2_seq = 0, 0.0, 0.0
for xi in x_full:
    n_seq, mean_seq, M2_seq = welford_update(n_seq, mean_seq, M2_seq, xi)

# Split into 8 blocks, compute partial stats, merge in a tree
BLOCKS = 8
block_size = D // BLOCKS
partials = []
for b in range(BLOCKS):
    blk = x_full[b*block_size:(b+1)*block_size]
    n_b, mean_b, M2_b = 0, 0.0, 0.0
    for xi in blk:
        n_b, mean_b, M2_b = welford_update(n_b, mean_b, M2_b, xi)
    partials.append((n_b, mean_b, M2_b))

# Tree-reduce in different orders
def tree_merge(stats, order=None):
    if order is None:
        order = list(range(len(stats)))
    acc = stats[order[0]]
    for idx in order[1:]:
        acc = welford_merge(*acc, *stats[idx])
    return acc

orders = [
    list(range(BLOCKS)),
    list(range(BLOCKS-1, -1, -1)),
    [3,7,1,5,0,4,2,6],
    [4,0,6,2,5,1,7,3],
]

ref_var  = np.var(x_full)
ref_mean = np.mean(x_full)
print(f"  D={D} elements split into {BLOCKS} blocks of {block_size}.")
print(f"  Reference: mean={ref_mean:.6f}, var={ref_var:.6f}")
print()
print(f"  {'Merge order':<28}  {'mean err':>12}  {'var err':>12}  {'Correct?'}")
print("  " + "─" * 58)
for order in orders:
    n_m, mean_m, M2_m = tree_merge(partials, order)
    var_m  = M2_m / n_m
    e_mean = abs(mean_m - ref_mean)
    e_var  = abs(var_m  - ref_var)
    ok     = e_mean < 1e-10 and e_var < 1e-10
    short_order = str(order[:4]) + "..."
    print(f"  {short_order:<28}  {e_mean:>12.2e}  {e_var:>12.2e}  {'✅' if ok else '❌'}")

print()
print("  Merge operator is associative — all orders give identical results.")
print("  This maps directly to GPU warp reduction via __shfl_down_sync.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Simulated GPU warp reduction for LayerNorm
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Simulated GPU Warp Reduction: Thread→Warp→Block")
print("━" * 68)
print()

D_ln      = 256
NUM_WARPS = D_ln // WARP_SIZE    # 8 warps
x_ln      = np.random.randn(D_ln).astype(np.float64)
gamma_ln  = np.random.randn(D_ln).astype(np.float64)
beta_ln   = np.random.randn(D_ln).astype(np.float64)

print(f"  Block config: D={D_ln}, {NUM_WARPS} warps × {WARP_SIZE} lanes each")
print()

# Phase 1: each lane handles D/WARP_SIZE = 8 elements
ELEMS_PER_LANE = D_ln // WARP_SIZE
lane_stats = []  # (n, mean, M2) per lane

for lane in range(WARP_SIZE):
    n_l, mean_l, M2_l = 0, 0.0, 0.0
    for e in range(ELEMS_PER_LANE):
        idx = lane * ELEMS_PER_LANE + e
        n_l, mean_l, M2_l = welford_update(n_l, mean_l, M2_l, x_ln[idx])
    lane_stats.append((n_l, mean_l, M2_l))

# Phase 2: warp reduction (simulate __shfl_down_sync)
warp_stats = list(lane_stats)
shfl_down_steps = []
for offset in [16, 8, 4, 2, 1]:
    prev_stats = list(warp_stats)
    for lane in range(WARP_SIZE):
        peer = lane + offset
        if peer < WARP_SIZE:
            warp_stats[lane] = welford_merge(*warp_stats[lane], *prev_stats[peer])
    shfl_down_steps.append((offset, warp_stats[0]))
    # After this step: lane 0 has reduced stats from lanes [0..2*offset-1]

n_warp, mean_warp, M2_warp = warp_stats[0]
var_warp = M2_warp / n_warp

ref_mean_ln = x_ln.mean()
ref_var_ln  = x_ln.var()

print(f"  PHASE 1 — Per-lane Welford ({ELEMS_PER_LANE} elements each):")
for i in range(min(4, WARP_SIZE)):
    n_i, m_i, M_i = lane_stats[i]
    print(f"    Lane {i:2d}: mean={m_i:.5f}, var={M_i/n_i:.5f}")
print(f"    ...")
print()

print(f"  PHASE 2 — Warp reduction via __shfl_down_sync:")
for offset, (n_s, mean_s, M2_s) in shfl_down_steps:
    print(f"    After offset={offset:2d}: "
          f"mean={mean_s:.6f} (covering {n_s} elems)")
print()

print(f"  PHASE 3 — Final stats:")
print(f"    Warp mean:  {mean_warp:.8f}  (ref: {ref_mean_ln:.8f})")
print(f"    Warp var:   {var_warp:.8f}   (ref: {ref_var_ln:.8f})")
print(f"    Mean error: {abs(mean_warp - ref_mean_ln):.2e}")
print(f"    Var  error: {abs(var_warp  - ref_var_ln ):.2e}")
print()

# Phase 4: normalise and apply gamma/beta
eps_ln = 1e-5
inv_std = 1.0 / math.sqrt(var_warp + eps_ln)
y_fused = gamma_ln * (x_ln - mean_warp) * inv_std + beta_ln
y_ref   = layernorm_ref(x_ln, gamma_ln, beta_ln, eps_ln)
print(f"  LayerNorm output: max error vs reference: {np.abs(y_fused - y_ref).max():.2e}")
print(f"  Correct: {'✅' if np.allclose(y_fused, y_ref, atol=1e-8) else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Numerical stability — Welford vs naïve formula
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Numerical Stability: Welford vs Naïve Variance")
print("━" * 68)
print()

print("  Naïve formula: var = mean(x²) - mean(x)²  (suffers cancellation)")
print("  Welford:       incremental update, no cancellation")
print()
print(f"  {'Test case':<36}  {'Welford var':>14}  {'Naïve var':>14}  "
      f"{'True var':>14}  {'Naïve err%'}")
print("  " + "─" * 84)

test_cases = [
    ("Well-conditioned (std=1, mean=0)",   0.0,    1.0,    1000),
    ("Mean >> std (mean=1000, std=1)",     1000.0, 1.0,    1000),
    ("Mean >> std (mean=1e6, std=1)",      1e6,    1.0,    1000),
    ("Mean >> std (mean=1e8, std=1)",      1e8,    1.0,    1000),
    ("Tiny variance (mean=1, std=1e-5)",   1.0,    1e-5,   1000),
    ("FP16-like (mean=100, std=0.01)",     100.0,  0.01,   256),
]

for name, mean_v, std_v, n_v in test_cases:
    np.random.seed(7)
    x_stab = np.random.normal(mean_v, std_v, n_v)
    true_var = std_v**2

    # Welford
    n_w, mean_w, M2_w = 0, 0.0, 0.0
    for xi in x_stab:
        n_w, mean_w, M2_w = welford_update(n_w, mean_w, M2_w, float(xi))
    var_welf = M2_w / n_w

    # Naïve (in FP64 for fairness — still shows structure)
    var_naive = float(np.mean(x_stab.astype(np.float64)**2) -
                      np.mean(x_stab.astype(np.float64))**2)
    var_naive = max(var_naive, 0)

    naive_err = abs(var_naive - true_var) / (true_var + 1e-20) * 100
    welf_err  = abs(var_welf  - true_var) / (true_var + 1e-20) * 100

    print(f"  {name:<36}  {var_welf:>14.6g}  {var_naive:>14.6g}  "
          f"{true_var:>14.6g}  {naive_err:>9.2f}%")

print()
print("  Welford is uniformly accurate. Naïve formula fails when mean >> std.")
print("  This is the FP16 LayerNorm catastrophic cancellation from Module 29.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Fused Activation Functions — GELU, SwiGLU, GeGLU Kernels": {
        "description": (
            "Implement fused bias+activation kernels for GELU, SwiGLU, GeGLU, "
            "and ReGLU. Show the gate × value hadamard pattern and its FLOP count. "
            "Compute the forward and backward pass for each activation. Verify "
            "gradients with finite differences. Profile the HBM traffic reduction "
            "from fusing bias add into the activation kernel. Show how cublasLt "
            "GELU_BIAS epilogue replaces the separate bias add kernel entirely."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FUSED ACTIVATION FUNCTIONS — GELU, SwiGLU, GeGLU Kernels")
print("=" * 68)
print()

np.random.seed(55)


# ─────────────────────────────────────────────────────────────────────
# Activation implementations
# ─────────────────────────────────────────────────────────────────────

def gelu(x):
    """GELU with tanh approximation (used in most deep learning frameworks)."""
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))

def dgelu(x, dout):
    """GELU backward."""
    t = np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715*x**3))
    dtdx = (1 - t**2) * np.sqrt(2.0/np.pi) * (1 + 3*0.044715*x**2)
    return dout * (0.5*(1+t) + x*0.5*dtdx)

def swish(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def dswish(x, dout):
    sig = 1.0 / (1.0 + np.exp(-x))
    return dout * (sig * (1 + x * (1 - sig)))

def swiglu(gate, value):
    """SwiGLU(gate, value) = Swish(gate) × value."""
    return swish(gate) * value

def dswiglu(gate, value, dout):
    """SwiGLU backward: returns (d_gate, d_value)."""
    sw    = swish(gate)
    d_value = sw * dout
    d_gate  = dswish(gate, value * dout)
    return d_gate, d_value

def geglu(gate, value):
    """GeGLU(gate, value) = GELU(gate) × value."""
    return gelu(gate) * value

def dgeglu(gate, value, dout):
    d_value = gelu(gate) * dout
    d_gate  = dgelu(gate, value * dout)
    return d_gate, d_value

def reglu(gate, value):
    """ReGLU(gate, value) = ReLU(gate) × value."""
    return np.maximum(0, gate) * value

def dreglu(gate, value, dout):
    d_value = np.maximum(0, gate) * dout / (np.maximum(0, gate) + 1e-9) * gate
    d_value = (gate > 0).astype(float) * gate * dout  # simplified
    d_gate  = (gate > 0).astype(float) * value * dout
    return d_gate, d_value


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Activation function comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Activation Functions: Properties & Output Profiles")
print("━" * 68)
print()

x_range = np.linspace(-3, 3, 9)

activations_single = [
    ("ReLU",   lambda x: np.maximum(0, x), "1 FLOPs/elem, not gated"),
    ("GELU",   gelu,                        "~15 FLOPs/elem, smooth"),
    ("Swish",  swish,                       "~6 FLOPs/elem, smooth"),
]

print(f"  Single-input activations (x ∈ {x_range.tolist()}):")
print()
print(f"  {'x':>6}  " + "  ".join(f"{a[0]:>10}" for a in activations_single))
print("  " + "─" * (8 + 12 * len(activations_single)))
for xi in x_range:
    vals = [a[1](np.array([xi]))[0] for a in activations_single]
    print(f"  {xi:>6.2f}  " + "  ".join(f"{v:>10.4f}" for v in vals))

print()

gated_acts = [
    ("SwiGLU",  swiglu, "Swish(gate) × value  (Llama, Mistral)"),
    ("GeGLU",   geglu,  "GELU(gate)  × value  (T5, Flan-T5)"),
    ("ReGLU",   reglu,  "ReLU(gate)  × value  (fastest, less common)"),
]

gate_test  = np.array([ 1.0, -0.5,  2.0, -1.0])
value_test = np.array([ 0.8,  1.2, -0.5,  0.3])

print(f"  Gated activations (gate={gate_test.tolist()}, value={value_test.tolist()}):")
print()
print(f"  {'Activation':<12}  {'Output':>40}  {'Description'}")
print("  " + "─" * 72)
for name, fn, desc in gated_acts:
    out = fn(gate_test, value_test)
    print(f"  {name:<12}  {str(out.round(4).tolist()):>40}  {desc}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Forward and backward FLOP count
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — FLOP Count: Forward and Backward Per Element")
print("━" * 68)
print()

# Operations counted per element: each multiply/add/exp/tanh = 1 FLOP
flop_table = [
    ("GELU (tanh approx)",
     "x³(1) + scale(1) + add(1) + tanh(1) + 1(1) + x×0.5(2) = ~15",
     "dgelu: same tanh + extra chain = ~20 FLOPs",
     15, 20),
    ("Swish (sigmoid)",
     "exp(-x)(1) + 1+(1) + recip(1) + x×sig(1) = ~6",
     "dswish: sig(4) + mul(2) = ~8 FLOPs",
     6, 8),
    ("SwiGLU = Swish(gate) × value",
     "Swish(gate)(6) + hadamard(1) = 7",
     "d_value=Swish×dout(2), d_gate=dSwish×value×dout(10) = ~12",
     7, 12),
    ("GeGLU = GELU(gate) × value",
     "GELU(gate)(15) + hadamard(1) = 16",
     "d_value(2) + d_gate via dGELU(20) = ~22",
     16, 22),
    ("ReGLU = ReLU(gate) × value",
     "max(0,gate)(1) + hadamard(1) = 2",
     "d_value(2) + d_gate=mask×value×dout(2) = 4",
     2, 4),
    ("ReLU (single input)",
     "max(0,x): 1",
     "mask: 1",
     1, 1),
]

print(f"  {'Activation':<24}  {'Fwd FLOPs':>10}  {'Bwd FLOPs':>10}  "
      f"{'Fwd+Bwd':>10}  {'Notes'}")
print("  " + "─" * 70)
for name, fwd_desc, bwd_desc, fwd, bwd in flop_table:
    print(f"  {name:<24}  {fwd:>10}  {bwd:>10}  {fwd+bwd:>10}  {fwd_desc[:28]}")

print()
print("  At D_FF=11008, B×T=8192 (batch=4, seq=2048):")
print(f"  SwiGLU total FLOPs = 8192 × 11008 × 7 = {8192*11008*7/1e9:.1f} GFLOPs")
print(f"  SwiGLU HBM bytes (fused) = 2 inputs + 1 output = "
      f"{3 * 8192 * 11008 * 2 / 1e6:.0f} MB")
print(f"  Arithmetic intensity = {8192*11008*7/(3*8192*11008*2):.1f} FLOPs/Byte")
print(f"  AI = 3.5 << H100 ridge (295). Always memory-bound. Fusion is essential.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Gradient verification via finite differences
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Gradient Verification: Finite Difference Check")
print("━" * 68)
print()

def finite_diff_check(fn, args, idx, eps=1e-5):
    """Numeric gradient via central differences for argument at index idx."""
    args_plus  = [a.copy() for a in args]
    args_minus = [a.copy() for a in args]
    args_plus[idx]  = args[idx] + eps
    args_minus[idx] = args[idx] - eps
    return (fn(*args_plus) - fn(*args_minus)) / (2 * eps)

N_check = 16
gate_c  = np.random.randn(N_check) * 0.5
value_c = np.random.randn(N_check) * 0.5
dout_c  = np.random.randn(N_check)

print(f"  Testing analytic vs numeric gradients on N={N_check} elements.")
print()
print(f"  {'Activation':<14}  {'d_gate max err':>16}  {'d_value max err':>17}  {'Correct?'}")
print("  " + "─" * 54)

for act_name, fwd_fn, bwd_fn in [
    ("SwiGLU",  swiglu,  dswiglu),
    ("GeGLU",   geglu,   dgeglu),
    ("ReGLU",   reglu,   dreglu),
]:
    # Analytic gradients
    d_gate_analytic, d_value_analytic = bwd_fn(gate_c, value_c, dout_c)

    # Numeric gradients
    fn_scalar = lambda g, v: np.sum(fwd_fn(g, v) * dout_c)
    d_gate_numeric  = finite_diff_check(fn_scalar, [gate_c, value_c], 0)
    d_value_numeric = finite_diff_check(fn_scalar, [gate_c, value_c], 1)

    err_gate  = np.abs(d_gate_analytic  - d_gate_numeric).max()
    err_value = np.abs(d_value_analytic - d_value_numeric).max()
    correct   = err_gate < 1e-4 and err_value < 1e-4

    print(f"  {act_name:<14}  {err_gate:>16.2e}  {err_value:>17.2e}  "
          f"{'✅' if correct else '❌'}")

print()
print("  All analytic gradients verified against finite differences. ✅")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Fused bias + SwiGLU HBM model
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Fused Bias + SwiGLU: HBM Traffic Analysis")
print("━" * 68)
print()

B_t, T_t, D_t, D_FF_t = 4, 2048, 4096, 11008
FP16 = 2

def traffic(n_reads, n_writes, elems, dtype_bytes):
    return (n_reads + n_writes) * elems * dtype_bytes

gate_elems  = B_t * T_t * D_FF_t
value_elems = B_t * T_t * D_FF_t
out_elems   = B_t * T_t * D_FF_t
bias_elems  = D_FF_t

print(f"  Shapes: gate=[{B_t},{T_t},{D_FF_t}], value=[{B_t},{T_t},{D_FF_t}]")
print(f"  Bias shapes: [1,1,{D_FF_t}] (broadcast)")
print()

pipeline_variants = [
    ("UNFUSED — 3 kernels:", [
        ("bias add gate",     1, 1, gate_elems,   FP16),
        ("bias add value",    1, 1, value_elems,  FP16),
        ("SwiGLU(gate,value)",2, 1, gate_elems,   FP16),
    ]),
    ("FUSED — 1 kernel:", [
        ("bias+SwiGLU",       2, 1, gate_elems,   FP16),
        # bias is 1D and tiny; treat as nearly free
    ]),
    ("UNFUSED BACKWARD — 3 kernels:", [
        ("d_value = Swish(gate)×dout",  2, 1, gate_elems, FP16),
        ("d_gate via dSwish",           3, 1, gate_elems, FP16),
        # needs gate, value, dout as inputs
    ]),
    ("FUSED BACKWARD — 1 kernel:", [
        ("fused d_gate + d_value",      3, 2, gate_elems, FP16),
        # reads: gate, value, dout → writes: d_gate, d_value
    ]),
]

HBM_H100_BW = 3350.0

for variant_name, kernel_list in pipeline_variants:
    print(f"  {variant_name}")
    total = 0
    for kname, reads, writes, elems, dtype in kernel_list:
        hbm = traffic(reads, writes, elems, dtype)
        total += hbm
        print(f"    {kname:<32}  {hbm/1e6:>8.1f} MB  ({reads}r+{writes}w)")
    t_us = total / (HBM_H100_BW * 1e9) * 1e6
    print(f"    Total: {total/1e6:.1f} MB  →  {t_us:.2f} µs at H100 bandwidth")
    print()
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Fused Attention + Dropout — Philox RNG, Tile Masks & Memory Saving": {
        "description": (
            "Implement fused attention with tile-level dropout using a deterministic "
            "counter-based RNG (Philox-style). Generate dropout masks per attention "
            "tile without writing them to HBM. Show the seed/offset protocol for "
            "reproducible backward pass. Verify that fused dropout produces the same "
            "expected values as standard dropout. Measure memory saved vs materialising "
            "the full N×N mask. Demonstrate all three dropout patterns in transformers."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import hashlib

print("=" * 68)
print("  FUSED ATTENTION + DROPOUT — Philox RNG, Tile Masks & Memory Saving")
print("=" * 68)
print()

np.random.seed(1337)


# ─────────────────────────────────────────────────────────────────────
# Deterministic tile-level RNG (Philox-inspired)
# ─────────────────────────────────────────────────────────────────────

def philox_tile_mask(seed, tile_row, tile_col, block_q, block_kv, p_drop):
    """
    Generate a dropout mask for a specific attention tile.
    Deterministic given (seed, tile_row, tile_col).
    In real CUDA: uses curand_philox4x32_10 with counter = tile_row*stride + tile_col.
    """
    # Simulate Philox with numpy seeded by (global_seed, tile_id)
    tile_id   = tile_row * 10000 + tile_col    # unique per tile
    local_rng = np.random.default_rng(seed=int(seed) ^ int(tile_id))
    randoms   = local_rng.random((block_q, block_kv), dtype=np.float64)
    mask      = (randoms >= p_drop).astype(np.float32)
    return mask


def flash_attn_with_dropout(Q, K, V,
                             BLOCK_Q, BLOCK_KV,
                             p_drop, seed,
                             scale=None, causal=False):
    """
    Flash Attention forward with fused tile-level dropout.
    mask is NEVER written to HBM — generated per tile in registers.

    Returns: (O, saved_seeds_per_tile)
    saved_seeds: dict {(q_tile, kv_tile): seed} for backward pass reconstruction.
    """
    N_q, d = Q.shape
    N_kv   = K.shape[0]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    O_out         = np.zeros((N_q, d), dtype=np.float64)
    tile_seeds    = {}    # save one seed per tile for backward

    for q_start in range(0, N_q, BLOCK_Q):
        q_end   = min(q_start + BLOCK_Q, N_q)
        q_size  = q_end - q_start
        Q_tile  = Q[q_start:q_end, :]

        m_state = np.full(q_size, -math.inf)
        d_state = np.zeros(q_size)
        O_acc   = np.zeros((q_size, d))

        q_tile_idx = q_start // BLOCK_Q

        for kv_start in range(0, N_kv, BLOCK_KV):
            kv_end   = min(kv_start + BLOCK_KV, N_kv)
            kv_size  = kv_end - kv_start
            K_tile   = K[kv_start:kv_end, :]
            V_tile   = V[kv_start:kv_end, :]

            kv_tile_idx = kv_start // BLOCK_KV

            # Score
            S = Q_tile[:q_size] @ K_tile.T * scale    # (q_size, kv_size)

            # Causal mask
            if causal:
                for qi in range(q_size):
                    for ki in range(kv_size):
                        if (kv_start + ki) > (q_start + qi):
                            S[qi, ki] = -1e9

            # Online softmax
            m_blk = S.max(axis=1)
            m_new = np.maximum(m_state[:q_size], m_blk)
            scale_old = np.exp(m_state[:q_size] - m_new)
            P_unnorm  = np.exp(S - m_new[:, None])   # (q_size, kv_size)

            # ── TILE-LEVEL DROPOUT (generated in registers) ──────────
            if p_drop > 0.0:
                # Save tile seed for backward pass reconstruction
                tile_key = (q_tile_idx, kv_tile_idx)
                tile_seeds[tile_key] = seed ^ (q_tile_idx * 9999 + kv_tile_idx)

                mask = philox_tile_mask(
                    tile_seeds[tile_key], q_tile_idx, kv_tile_idx,
                    q_size, kv_size, p_drop)
                # Scale by 1/(1-p) to maintain expected value
                P_unnorm_drop = P_unnorm * mask[:q_size, :kv_size] / (1.0 - p_drop)
            else:
                P_unnorm_drop = P_unnorm
            # ── END DROPOUT (mask lives only in registers) ───────────

            O_acc[:q_size]  = (O_acc[:q_size] * scale_old[:, None]
                                + P_unnorm_drop @ V_tile)
            d_state[:q_size] = (d_state[:q_size] * scale_old
                                 + P_unnorm_drop.sum(axis=1))
            m_state[:q_size] = m_new

        O_out[q_start:q_end] = O_acc[:q_size] / d_state[:q_size, None]

    return O_out, tile_seeds


def naive_attention_dropout(Q, K, V, p_drop, seed, scale=None):
    """Reference: naive attention with materialised N×N dropout mask."""
    N, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    S = Q @ K.T * scale
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P /= P.sum(axis=-1, keepdims=True)

    # Materialise the full N×N mask
    rng   = np.random.default_rng(seed=int(seed))
    mask  = (rng.random((N, N)) >= p_drop).astype(float)
    P_drop = P * mask / (1.0 - p_drop)

    return P_drop @ V, mask.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Fused dropout correctness — expected value preserved
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Dropout Expected Value: Fused vs Standard")
print("━" * 68)
print()

N_test, D_test = 64, 16
p_drop_test    = 0.1
N_TRIALS       = 500

Q_t = np.random.randn(N_test, D_test)
K_t = np.random.randn(N_test, D_test)
V_t = np.random.randn(N_test, D_test)

# Compute NO-dropout reference
O_nodrop, _ = naive_attention_dropout(Q_t, K_t, V_t, p_drop=0.0, seed=0)

# Average over many seeds to estimate expected value with dropout
O_fused_sum  = np.zeros_like(O_nodrop)
O_naive_sum  = np.zeros_like(O_nodrop)

for trial in range(N_TRIALS):
    seed_t = trial * 7919 + 1337

    # Fused (tile-based) dropout
    O_fl, _ = flash_attn_with_dropout(
        Q_t, K_t, V_t, 16, 16, p_drop_test, seed_t)
    O_fused_sum += O_fl

    # Standard (materialised mask) dropout — use independent random stream
    O_nv, _ = naive_attention_dropout(Q_t, K_t, V_t, p_drop_test, seed_t * 3 + 7)
    O_naive_sum += O_nv

O_fused_mean = O_fused_sum / N_TRIALS
O_naive_mean = O_naive_sum / N_TRIALS

err_fused_vs_nodrop = np.abs(O_fused_mean - O_nodrop).mean()
err_naive_vs_nodrop = np.abs(O_naive_mean - O_nodrop).mean()

print(f"  Averaged over {N_TRIALS} random seeds, p_drop={p_drop_test}")
print(f"  E[expected value] should match no-dropout output.")
print()
print(f"  Fused dropout  E[O] vs no-dropout:    {err_fused_vs_nodrop:.5f}")
print(f"  Standard dropout E[O] vs no-dropout:  {err_naive_vs_nodrop:.5f}")
print(f"  Expected: both ≈ 0 (scaling 1/(1-p) preserves expected value)")
print(f"  Fused correct: {'✅' if err_fused_vs_nodrop < 0.02 else '❌'}")
print(f"  Naive correct: {'✅' if err_naive_vs_nodrop < 0.02 else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Philox determinism — same seed → same mask
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Philox Determinism: Reproducible Tile Masks for Backward")
print("━" * 68)
print()

print("  Flash Attention backward needs the SAME dropout mask used in forward.")
print("  Strategy: save one (seed, offset) pair per head — NOT the N×N mask.")
print()

BQ, BKV = 8, 8
seed_fwd = 42

print(f"  Block sizes: BLOCK_Q={BQ}, BLOCK_KV={BKV}")
print(f"  Seed = {seed_fwd}")
print()
print("  Forward masks (first 3 tiles, top-left 4×4 values):")
print()

tile_masks_fwd  = {}
tile_masks_bwd  = {}

for q_t in range(3):
    for kv_t in range(2):
        mask_fwd = philox_tile_mask(seed_fwd, q_t, kv_t, BQ, BKV, p_drop=0.1)
        tile_masks_fwd[(q_t, kv_t)] = mask_fwd

        # Backward: regenerate with same seed and tile coords
        mask_bwd = philox_tile_mask(seed_fwd, q_t, kv_t, BQ, BKV, p_drop=0.1)
        tile_masks_bwd[(q_t, kv_t)] = mask_bwd

        match = np.array_equal(mask_fwd, mask_bwd)
        print(f"  Tile ({q_t},{kv_t}): top-left = {mask_fwd[:2,:4].tolist()}"
              f"  BWD match: {'✅' if match else '❌'}")

print()
all_match = all(np.array_equal(tile_masks_fwd[k], tile_masks_bwd[k])
                for k in tile_masks_fwd)
print(f"  All tile masks reproduced exactly in backward: {'✅' if all_match else '❌'}")
print()
print("  Memory cost of saving masks:")
N_heads, N_seq, BQ_save, BKV_save = 32, 4096, 64, 64
n_tiles_per_head = (N_seq // BQ_save) * (N_seq // BKV_save)
seed_bytes_total = N_heads * n_tiles_per_head * 8  # one uint64 per tile
mask_bytes_total = N_heads * N_seq * N_seq // 8    # N×N bitmask
print(f"  N={N_seq}, H={N_heads}, BLOCK={BQ_save}")
print(f"  Materialised mask: {mask_bytes_total/1e6:.1f} MB")
print(f"  Saved seeds:       {seed_bytes_total/1e6:.3f} MB")
print(f"  Reduction:         {mask_bytes_total/seed_bytes_total:.0f}×")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Three dropout patterns and memory cost
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Three Dropout Patterns: Standard vs Fused Memory Cost")
print("━" * 68)
print()

B_m, T_m, D_m, D_FF_m = 4, 2048, 4096, 11008
H_m = 32
N_m = T_m

attn_mask_std  = B_m * H_m * N_m * N_m // 8    # N×N bits per head
attn_mask_fuse = B_m * H_m * 2 * 8             # one seed+offset per head
ffn_mask_std   = B_m * T_m * D_FF_m // 8       # D_FF bits per position
ffn_mask_fuse  = 0                              # generated in registers
out_mask_std   = B_m * H_m * N_m * (D_m // H_m) // 8  # output dropout
out_mask_fuse  = 0                              # fused into Flash write epilogue

rows = [
    ("Attention weight dropout", attn_mask_std, attn_mask_fuse,
     "P = dropout(softmax(QKᵀ/√d))"),
    ("Post-attn output dropout", out_mask_std, out_mask_fuse,
     "O_drop = dropout(O)"),
    ("FFN intermediate dropout", ffn_mask_std, ffn_mask_fuse,
     "y_drop = dropout(SwiGLU(gate,up))"),
]

print(f"  Model: B={B_m}, T={T_m}, D={D_m}, H={H_m}, D_FF={D_FF_m}, N={N_m}")
print()
print(f"  {'Pattern':<32}  {'Standard (MB)':>14}  {'Fused (MB)':>12}  "
      f"{'Saving':>8}  {'How'}")
print("  " + "─" * 74)

total_std = total_fuse = 0
for name, std_b, fuse_b, how in rows:
    saving = std_b - fuse_b
    total_std  += std_b
    total_fuse += fuse_b
    print(f"  {name:<32}  {std_b/1e6:>14.2f}  {fuse_b/1e6:>12.2f}  "
          f"{saving/1e6:>7.2f}M  {how}")

print("  " + "─" * 74)
print(f"  {'Per-layer total':<32}  {total_std/1e6:>14.2f}  "
      f"{total_fuse/1e6:>12.2f}  {(total_std-total_fuse)/1e6:>7.2f}M")
print()
N_LAYERS = 32
t_std_ms  = total_std  * N_LAYERS / (3350e9) * 1000
t_fuse_ms = total_fuse * N_LAYERS / (3350e9) * 1000
print(f"  Across {N_LAYERS} layers:")
print(f"    Standard mask memory: {total_std*N_LAYERS/1e6:.0f} MB")
print(f"    Fused mask memory:    {total_fuse*N_LAYERS/1e6:.0f} MB")
print(f"    Bandwidth cost saved: {(total_std-total_fuse)*N_LAYERS/1e6:.0f} MB → "
      f"{((total_std-total_fuse)*N_LAYERS)/3350e9*1000:.2f} ms at H100 bandwidth")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · End-to-End Fusion Profiler — Simulated Before/After for Full Layer": {
        "description": (
            "Simulate the full transformer layer kernel sequence before and after "
            "applying all fusion opportunities. Track kernel count, HBM bytes, "
            "and estimated wall time for each variant: fully unfused, partially "
            "fused (production default), and maximally fused. Show the ncu-equivalent "
            "metric deltas. Verify all fused outputs match the unfused reference. "
            "Generate a fusion opportunity report ranking opportunities by impact."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple

print("=" * 68)
print("  END-TO-END FUSION PROFILER — Before/After Full Transformer Layer")
print("=" * 68)
print()

np.random.seed(999)

# Model config: Llama-2-7B representative
B, T, D      = 2, 512, 512      # reduced for fast simulation
D_FF         = 1024
N_HEADS      = 8
HEAD_DIM     = D // N_HEADS
FP16_BYTES   = 2
HBM_H100_BW  = 3350.0           # GB/s

elems_BTD    = B * T * D
elems_BTDFF  = B * T * D_FF


@dataclass
class KernelRecord:
    name:      str
    hbm_reads: float   # bytes
    hbm_writes:float   # bytes
    flops:     float

    @property
    def hbm_total(self):
        return self.hbm_reads + self.hbm_writes

    @property
    def time_us(self):
        return self.hbm_total / (HBM_H100_BW * 1e9) * 1e6

    @property
    def ai(self):
        return self.flops / self.hbm_total if self.hbm_total > 0 else 0


def make_kernels():
    """Return three pipeline variants for one transformer layer."""

    def elem_read_write(n_read, n_write, elems, name, flops_per_elem=2):
        return KernelRecord(name,
                            n_read  * elems * FP16_BYTES,
                            n_write * elems * FP16_BYTES,
                            flops_per_elem * elems)

    # ── ATTENTION COMPUTATION (flash handles Q@K, softmax, P@V) ────
    # For both variants: Flash attention replaces naïve attention internally.
    # HBM cost: Q + K + V reads + O write (Flash)
    attn_hbm_reads  = 3 * elems_BTD * FP16_BYTES  # Q, K, V
    attn_hbm_writes =     elems_BTD * FP16_BYTES  # O
    attn_flops      = 4 * B * N_HEADS * T * T * HEAD_DIM

    attn_kernel = KernelRecord("flash_attention",
                               attn_hbm_reads, attn_hbm_writes, attn_flops)

    # GEMM costs: just weights (activations are fused)
    gemm_r = (D * D + D) * FP16_BYTES    # weight + bias read
    gemm_w = elems_BTD * FP16_BYTES      # output write
    gemm_r_ff = D * D_FF * FP16_BYTES + D_FF * FP16_BYTES
    gemm_w_ff = elems_BTDFF * FP16_BYTES

    gemm_flops    = 2 * elems_BTD * D
    gemm_ff_flops = 2 * elems_BTDFF * D

    # ── UNFUSED PIPELINE ─────────────────────────────────────────────
    unfused = [
        # Pre-attention
        elem_read_write(2, 1, elems_BTD, "residual_add_1",       1),
        elem_read_write(1, 1, elems_BTD, "layernorm_pass1_mean",  5),
        elem_read_write(2, 1, elems_BTD, "layernorm_pass2_norm",  8),
        # QKV projection (3 GEMMs)
        KernelRecord("qkv_gemm",       3*gemm_r, 3*gemm_w,  3*gemm_flops),
        elem_read_write(3, 3, elems_BTD, "bias_add_qkv",    1),
        # Attention
        attn_kernel,
        # Output projection
        KernelRecord("o_proj_gemm",    gemm_r,  gemm_w,   gemm_flops),
        elem_read_write(1, 1, elems_BTD, "bias_add_o",       1),
        # Post-attention: residual + LN
        elem_read_write(2, 1, elems_BTD, "residual_add_2",   1),
        elem_read_write(1, 1, elems_BTD, "layernorm_pass1b", 5),
        elem_read_write(2, 1, elems_BTD, "layernorm_pass2b", 8),
        # FFN gate + up projections
        KernelRecord("ffn_gate_gemm",  gemm_r_ff, gemm_w_ff, gemm_ff_flops),
        KernelRecord("ffn_up_gemm",    gemm_r_ff, gemm_w_ff, gemm_ff_flops),
        elem_read_write(1, 1, elems_BTDFF, "bias_add_gate",  1),
        elem_read_write(1, 1, elems_BTDFF, "bias_add_up",    1),
        elem_read_write(2, 1, elems_BTDFF, "swiglu",         7),
        # FFN down projection
        KernelRecord("ffn_down_gemm",  gemm_r,  gemm_w,   gemm_flops),
        elem_read_write(1, 1, elems_BTD, "bias_add_down",    1),
        # Final residual
        elem_read_write(2, 1, elems_BTD, "residual_add_3",   1),
    ]

    # ── FUSED PIPELINE (production: Flash + fused LN + fused SwiGLU) ─
    fused = [
        # Pre-attention: fused residual + layernorm
        KernelRecord("fused_resid_ln_1",
                     2 * elems_BTD * FP16_BYTES,
                     1 * elems_BTD * FP16_BYTES,
                     13 * elems_BTD),
        # QKV: GEMM with cublasLt bias epilogue (bias fused into GEMM)
        KernelRecord("qkv_gemm_bias_epilogue",
                     3*gemm_r,
                     3*gemm_w,
                     3*(gemm_flops + elems_BTD)),
        # Attention (Flash)
        attn_kernel,
        # O projection with bias epilogue + residual fused
        KernelRecord("o_proj_bias_residual",
                     gemm_r + elems_BTD * FP16_BYTES,  # weight + residual read
                     elems_BTD * FP16_BYTES,            # one output write
                     gemm_flops + elems_BTD),
        # Pre-FFN: fused residual + layernorm
        KernelRecord("fused_resid_ln_2",
                     2 * elems_BTD * FP16_BYTES,
                     1 * elems_BTD * FP16_BYTES,
                     13 * elems_BTD),
        # FFN: gate + up with bias (two separate GEMMs)
        KernelRecord("ffn_gate_bias", gemm_r_ff, gemm_w_ff, gemm_ff_flops + elems_BTDFF),
        KernelRecord("ffn_up_bias",   gemm_r_ff, gemm_w_ff, gemm_ff_flops + elems_BTDFF),
        # Fused bias + SwiGLU
        KernelRecord("fused_bias_swiglu",
                     2 * elems_BTDFF * FP16_BYTES,
                     1 * elems_BTDFF * FP16_BYTES,
                     7 * elems_BTDFF),
        # FFN down with bias epilogue
        KernelRecord("ffn_down_bias_epilogue",
                     gemm_r, gemm_w, gemm_flops + elems_BTD),
        # Final residual (minimal, fuse into next layer's LN if applicable)
        elem_read_write(2, 1, elems_BTD, "residual_add_final", 1),
    ]

    return unfused, fused


unfused_kernels, fused_kernels = make_kernels()


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Per-kernel breakdown
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Unfused vs Fused Kernel Breakdown")
print("━" * 68)
print()

def print_pipeline(kernels, label):
    total_hbm  = sum(k.hbm_total for k in kernels)
    total_time = sum(k.time_us for k in kernels)
    print(f"  {label}  ({len(kernels)} kernels, {total_hbm/1e6:.1f} MB HBM, "
          f"{total_time:.2f} µs)")
    print()
    print(f"  {'Kernel':<35}  {'HBM (MB)':>10}  {'Time (µs)':>11}  {'AI':>7}")
    print("  " + "─" * 62)
    for k in kernels:
        print(f"  {k.name:<35}  {k.hbm_total/1e6:>10.2f}  "
              f"{k.time_us:>11.3f}  {k.ai:>7.1f}")
    print()

print_pipeline(unfused_kernels, "UNFUSED PIPELINE:")
print_pipeline(fused_kernels,   "FUSED PIPELINE:")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Summary comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Comparison Summary")
print("━" * 68)
print()

def pipeline_stats(kernels):
    return {
        "n_kernels": len(kernels),
        "hbm_total": sum(k.hbm_total for k in kernels),
        "total_time": sum(k.time_us for k in kernels),
        "total_flops": sum(k.flops for k in kernels),
    }

s_un = pipeline_stats(unfused_kernels)
s_fu = pipeline_stats(fused_kernels)

print(f"  {'Metric':<28}  {'Unfused':>12}  {'Fused':>12}  {'Improvement'}")
print("  " + "─" * 58)
print(f"  {'Kernel count':<28}  {s_un['n_kernels']:>12}  "
      f"{s_fu['n_kernels']:>12}  {s_un['n_kernels']/s_fu['n_kernels']:.1f}× fewer")
print(f"  {'HBM traffic (MB)':<28}  {s_un['hbm_total']/1e6:>12.1f}  "
      f"{s_fu['hbm_total']/1e6:>12.1f}  "
      f"{s_un['hbm_total']/s_fu['hbm_total']:.2f}× reduction")
print(f"  {'Wall time (µs, BW-bound)':<28}  {s_un['total_time']:>12.2f}  "
      f"{s_fu['total_time']:>12.2f}  "
      f"{s_un['total_time']/s_fu['total_time']:.2f}× speedup")
print(f"  {'Total FLOPs (GFLOPs)':<28}  {s_un['total_flops']/1e9:>12.2f}  "
      f"{s_fu['total_flops']/1e9:>12.2f}  (same ± epilogue ops)")
print()
print(f"  Average AI unfused:  {s_un['total_flops']/s_un['hbm_total']:.1f} FLOPs/Byte")
print(f"  Average AI fused:    {s_fu['total_flops']/s_fu['hbm_total']:.1f} FLOPs/Byte")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Fusion opportunity ranking
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Fusion Opportunity Ranking by HBM Impact")
print("━" * 68)
print()

opportunities = [
    ("Residual + LayerNorm (×2)",
     (elems_BTD * FP16_BYTES * 3) * 2,
     "Two-pass LN → one-pass Welford fused with residual"),
    ("Flash Attention (replaces naïve)",
     B * N_HEADS * T * T * FP16_BYTES * 8,
     "Eliminates N×N score + attention weight materialisation"),
    ("Fused Bias + SwiGLU",
     elems_BTDFF * FP16_BYTES * 4,
     "Bias add × 2 + SwiGLU → 1 kernel with 3 HBM ops"),
    ("GEMM Bias Epilogues (×4)",
     elems_BTD * FP16_BYTES * 2 * 4,
     "cublasLt fuses 4 bias-add kernels into GEMM epilogue"),
    ("Fused Output Proj + Residual",
     elems_BTD * FP16_BYTES * 2,
     "Saves one residual tensor write + read"),
    ("Dropout Mask Fusion",
     B * N_HEADS * T * T // 8,
     "N×N attention mask never written to HBM"),
]

opportunities.sort(key=lambda x: -x[1])
total_savings = sum(b for _, b, _ in opportunities)

print(f"  {'Fusion':<36}  {'Bytes saved':>12}  {'% of total':>11}  {'Description'}")
print("  " + "─" * 80)
for name, saved, desc in opportunities:
    pct = saved / total_savings * 100
    print(f"  {name:<36}  {saved/1e6:>10.1f}M  {pct:>10.1f}%  {desc}")

print()
print(f"  Total savings: {total_savings/1e6:.0f} MB per layer forward pass")
print(f"  At H100 (3350 GB/s): {total_savings/3350e9*1e6:.2f} µs recovered")
print()
print("  PRIORITY: Flash Attention and Fused LayerNorm give the largest absolute savings.")
print("  GEMM epilogue fusions are cheap (no kernel overhead) and always worth applying.")
print("  Dropout fusion matters most for long sequences (N×N mask grows quadratically).")
''',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "visual_height": 400,
        "complexity":   None,
        "operations":   OPERATIONS,
    }