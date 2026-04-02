"""
Quantization Kernels — INT8 Symmetric/Asymmetric, FP8 & GPTQ/AWQ Dequant
=========================================================================

Quantization maps high-precision floating-point weights and activations to
lower-precision integer or reduced-float representations. The payoff is
substantial: a 7B parameter model in FP16 requires 14 GB of VRAM; in INT8,
7 GB; in INT4 (via GPTQ/AWQ), 3.5 GB. More importantly, quantized GEMMs
can use faster hardware paths — INT8 tensor cores on A100 deliver 624 TOPS,
2× the FP16 rate. FP8 on H100 delivers 1979 TFLOP/s, 2× FP16.

The discipline has two distinct concerns that are often conflated:

    1. CALIBRATION / QUANTIZATION ALGORITHM — how to choose the scale
       factors that minimise accuracy loss when mapping FP16 → INT8.
       GPTQ, AWQ, SmoothQuant, and LLM.int8() are calibration methods.
       This is a model-compression research problem.

    2. DEQUANTIZATION KERNELS — the GPU code that, at inference time,
       converts quantized weights back to a usable format before or
       during the GEMM. This is the systems problem this module covers.

The dequant kernel sits on the critical path of every token generation step:
    - INT8 GEMM: multiply INT8 weights × INT8 activations → INT32, then
      rescale to FP16. The rescaling (dequant) happens in the GEMM epilogue.
    - FP8 GEMM: similar, with per-tensor or per-tile scaling.
    - GPTQ/AWQ INT4: weights are stored as packed 4-bit integers.
      At runtime, the dequant kernel unpacks 8 INT4 values per INT32,
      converts to FP16, and either feeds a FP16 GEMM or uses a
      specialised W4A16 (4-bit weight, 16-bit activation) kernel.

Understanding quantization kernels means understanding: the number formats
(INT8, INT4 packed, FP8 E4M3/E5M2), the scale-and-zero-point arithmetic,
the memory layout of packed weights, and the GPU kernel patterns that make
dequantization fast enough to not be the bottleneck.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Quantization Kernels — INT8 Symmetric/Asymmetric, FP8 & GPTQ/AWQ Dequant"
DISPLAY_NAME = "14 · Quantization Kernels"
ICON         = "🗜️"
SUBTITLE     = "INT8 · FP8 E4M3/E5M2 · GPTQ · AWQ · Dequant · W4A16 · Per-Channel Scaling"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — QUANTIZATION FUNDAMENTALS: SCALE, ZERO-POINT & CLIPPING

### The Linear Quantization Model

    Linear quantization maps a floating-point value x to an integer q via:

    QUANTIZE:
        q = clamp(round(x / scale + zero_point), q_min, q_max)

    DEQUANTIZE:
        x_approx = (q - zero_point) × scale

    where:
        scale      = (x_max - x_min) / (q_max - q_min)  [step size in FP space]
        zero_point = round(q_min - x_min / scale)        [integer offset]
        q_min, q_max  define the integer range (e.g., [-128, 127] for INT8)

    The QUANTIZATION ERROR for element x_i is:
        error_i = x_i - x̂_i = x_i - (round(x_i/scale + zp) - zp) × scale
        |error_i| ≤ scale/2   (bounded by half the step size)

    The maximum error over all elements ≤ scale/2 = (x_max-x_min) / (2*(q_max-q_min))
    For INT8: q_max - q_min = 255, so max_error = (x_max-x_min) / 510.

### Symmetric vs Asymmetric Quantization

    SYMMETRIC quantization: zero_point = 0, range is [-Xmax, +Xmax].
        q = clamp(round(x / scale), -q_sat, q_sat)
        x̂ = q × scale
        scale = max(|x|) / q_sat

        For INT8 signed: q_sat = 127. Range uses [-127, 127] (not -128 to 127).
        Why not -128? Symmetric preserves q=0 for x=0, which is crucial for
        sparse tensors and avoids asymmetric bias in matrix multiply accumulation.

        ADVANTAGE: no zero-point to track or add in the GEMM.
        INT8 GEMM epilogue: output = INT32_result × scale_A × scale_B
        (scalar multiply — much cheaper than zero-point correction)

        DISADVANTAGE: wastes the negative half if distribution is skewed.
        E.g., ReLU outputs ∈ [0, ∞) — symmetric wastes the [-Xmax, 0] range.
        For ReLU activations: only 128 levels used out of 256.

    ASYMMETRIC quantization: zero_point ≠ 0, full [q_min, q_max] range used.
        q = clamp(round(x / scale + zero_point), 0, 255)  [for UINT8]
        x̂ = (q - zero_point) × scale

        ADVANTAGE: uses the full 256 levels for any distribution.
        Essential for: activations after ReLU (non-negative), biases.

        DISADVANTAGE: zero-point correction in GEMM is expensive.
        For Y = X_quant @ W_quant where both have zero-points (zp_x, zp_w):
            Y_fp = (X_quant - zp_x) @ (W_quant - zp_w) × scale
               = X_quant @ W_quant × scale
                 - zp_x × (sum of W rows) × scale
                 - zp_w × (sum of X cols) × scale
                 + zp_x × zp_w × K × scale
        These correction terms require extra memory reads and FP operations.
        In practice: weights use symmetric, activations may use asymmetric.

### Granularity: Per-Tensor, Per-Channel, Per-Group

    QUANTIZATION GRANULARITY determines the scale and zero-point coverage:

    PER-TENSOR: one scale for the entire tensor.
        scale = max(|W|) / 127  (for symmetric INT8)
        Cheapest to store. Worst accuracy (outliers in one channel dominate).
        Used for: simple INT8 inference on well-behaved models.

    PER-CHANNEL (per-row for weights): one scale per output channel.
        scale_i = max(|W[i, :]|) / 127  for row i
        During GEMM: each output row is scaled by its channel's scale.
        Accuracy: much better than per-tensor (≈ FP16 baseline for most tasks).
        Used by: LLM.int8(), standard INT8 post-training quantization.
        Memory overhead: N_out × sizeof(FP32) for scales ≈ negligible.

    PER-GROUP (group quantization): one scale per group of g weights.
        For a weight matrix W [N_out, N_in], group_size = g:
        Number of scale groups = N_out × (N_in / g)
        Each group of g consecutive input weights shares one scale.
        scale_i_j = max(|W[i, j*g:(j+1)*g]|) / (2^(bits-1) - 1)

        GPTQ/AWQ use group_size = 128 (default).
        Memory for scales: N_out × (N_in/128) × sizeof(FP16) = 0.8% of weight size.
        Accuracy: near-lossless for INT4 on most LLMs.
        Used by: GPTQ, AWQ, AutoRound — the dominant INT4 quantization methods.

### The Clipping Tradeoff

    A fundamental tension: use a wide quantization range or a narrow one?

    WIDE RANGE (scale = max(|x|) / q_sat):
        All values preserved — no clipping.
        But outliers force a large scale → fine-grained values are quantized coarsely.
        For a weight distribution with 99% of values in [-0.1, 0.1] but one outlier at 10:
            scale = 10/127 ≈ 0.079
            Values in [-0.1, 0.1] map to integers in [-1, 1] → only 3 levels!
            99% of the weight information is lost to 2 bits of resolution.

    NARROW RANGE (clip outliers, use scale = percentile(|x|, 99.9%) / q_sat):
        Some values are clipped (bounded at q_sat after quantization).
        But most values get finer resolution.
        Optimal clip level minimises the expected squared quantization error:
            E[(x - x̂)²] = E[(clip error)²] + E[(round error)²]
        The optimal clip can be found by searching (SmoothQuant, GPTQ).


##### PART 2 — INT8 QUANTIZATION: SYMMETRIC, ASYMMETRIC & GEMM INTEGRATION

### INT8 Number Format

    Signed INT8:  range [-128, 127], 256 levels.
        Used for: weights (symmetric, range [-127, 127]).
        Note: -128 is avoided in symmetric quantization to keep the range symmetric.

    Unsigned UINT8: range [0, 255], 256 levels.
        Used for: activations after ReLU (naturally non-negative).
        PyTorch quantization uses UINT8 for activations by default.

    INT8 GEMM hardware:
        NVIDIA A100 and H100: INT8 tensor cores via IMMA instruction.
        A100 peak INT8: 624 TOPS (2× FP16 rate, same as FP8 on A100 isn't available).
        H100 peak INT8: 1979 TOPS (same peak as FP8).
        cublas API: cublasGemmEx with CUBLAS_COMPUTE_32I and CUDA_R_8I inputs.

### INT8 Symmetric GEMM: The Scale Factor Protocol

    For Y = X @ W where X and W are both INT8 (symmetric, zero_point=0):

    QUANTIZATION (offline for W, online for X):
        W_int8 = round(W / scale_w)              scale_w = max(|W|) / 127
        X_int8 = round(X / scale_x)              scale_x = max(|X|) / 127

    GEMM (INT8 × INT8 → INT32):
        Y_int32 = X_int8 @ W_int8                (integer accumulation in INT32)

    DEQUANTIZATION (GEMM epilogue):
        Y_fp16 = Y_int32 × (scale_x × scale_w)   (single FP32 multiply per element)

    Why accumulate in INT32?
        For K=4096: max value of an INT8 dot product = 127 × 127 × 4096 ≈ 66M.
        log2(66M) ≈ 26 bits — fits in INT32 (31 bits + sign). Does NOT fit in INT16.
        INT32 accumulation ensures no overflow during the K-dimension summation.

    PER-CHANNEL scale:
        scale_w is a vector of shape [N_out].
        The dequant step in the epilogue multiplies each output row by its channel scale:
            Y_fp16[i, :] = Y_int32[i, :] × scale_x × scale_w[i]
        This is a row-wise multiply — handled in the cublasLt epilogue or a
        separate elementwise kernel.

### INT8 Asymmetric GEMM: Zero-Point Correction

    For Y = (X - zp_x) @ (W - zp_w) with asymmetric quantization:

    Expanded form:
        Y = X@W - zp_x×(sum over K of W) - zp_w×(sum over K of X) + K×zp_x×zp_w

    CORRECTION TERMS:
        Term 1: -zp_x × row_sum(W)   [N_out values, precomputed offline for W]
        Term 2: -zp_w × col_sum(X)   [N_batch values, computed online for X]
        Term 3: K × zp_x × zp_w      [scalar]

    In practice: only weights need asymmetric quantization if activations use
    symmetric. Setting zp_w=0 (symmetric weights) eliminates term 2 and 3,
    leaving only term 1 as a simple bias correction.

### LLM.int8() — Mixed-Precision Decomposition (Dettmers et al., 2022)

    Challenge: large language models have OUTLIER activations — a small fraction
    of activation channels consistently have very large values (10–100× the norm).
    These outliers force a large scale, degrading quantization quality for all others.

    LLM.int8() solution:
        1. Identify OUTLIER COLUMNS in the activation matrix X (by magnitude threshold).
        2. Decompose X = X_outlier (FP16) + X_regular (INT8).
        3. Compute Y = X_outlier @ W_outlier (FP16 GEMM) +
                      X_regular @ W_regular (INT8 GEMM).
        4. Accumulate results in FP16.

    Typically 1–3% of columns are outliers. INT8 for 97–99% of compute.
    FP16 for the remaining 1–3% (but this is the bottleneck for very large models).
    Memory: weights stored in INT8 → 2× memory reduction vs FP16.
    Compute: ~1.5–1.8× speedup vs FP16 on A100 (theoretical 2× partially offset by
    overhead of decomposition and correction).


##### PART 3 — FP8 QUANTIZATION: E4M3 VS E5M2 AND H100 HARDWARE SUPPORT

### FP8 Number Formats

    Both FP8 formats are defined in the OCP (Open Compute Project) MX specification
    and supported natively by H100 wgmma instructions.

    FP8 E4M3 (4 exponent bits, 3 mantissa bits):
        Bit layout: S | EEEE | MMM
        Bias:       7
        Max normal: 448.0  (when exponent = 1111, mantissa = 110)
        Special:    NaN represented as 0 11111111 (no +/-inf, only NaN)
        Machine epsilon: 2^(-3) = 0.125
        Used for:   FORWARD PASS (activations, weights in inference)
                    Better precision for values near 1.0 (typical activations).

    FP8 E5M2 (5 exponent bits, 2 mantissa bits):
        Bit layout: S | EEEEE | MM
        Bias:       15
        Max normal: 57344.0  (large range, needed for gradients)
        Special:    Inf and NaN both supported
        Machine epsilon: 2^(-2) = 0.25
        Used for:   BACKWARD PASS (gradients can be large or small)
                    Better dynamic range for gradient values.

    COMPARISON TABLE:
        ┌─────────┬──────────┬──────────┬──────────┬────────────┬────────────┐
        │ Format  │ Exp bits │ Mant bits│ Max val  │  Epsilon   │ Use case   │
        ├─────────┼──────────┼──────────┼──────────┼────────────┼────────────┤
        │ E4M3    │    4     │    3     │    448   │  0.125     │ fwd (acts) │
        │ E5M2    │    5     │    2     │  57344   │  0.250     │ bwd (grad) │
        │ FP16    │    5     │   10     │  65504   │  0.001     │ general    │
        │ BF16    │    8     │    7     │  3.4e38  │  0.0078    │ training   │
        └─────────┴──────────┴──────────┴──────────┴────────────┴────────────┘

    Why E4M3 for activations?
        Activations (post-LayerNorm, post-GELU) typically lie in [-5, 5].
        E4M3 can represent this range with decent precision (8 levels per binade).
        E5M2 would waste the wider range on activations that rarely exceed ±10.

    Why E5M2 for gradients?
        Gradients span many orders of magnitude (from ~10^-4 to ~10^2 in training).
        E5M2's 5-bit exponent matches the gradient distribution better.
        E4M3 would clip or underflow too many gradient values.

### H100 FP8 Hardware Support

    H100 (Hopper) supports FP8 GEMM via the wgmma instruction:
        Inputs: FP8 E4M3 or E5M2
        Accumulation: FP32 (always, same as FP16 WMMA)
        Throughput: 1979 TFLOP/s (FP8, 2× FP16 rate of 989 TFLOP/s)
        Memory: 1 byte per weight (vs 2 bytes for FP16) → 2× more weights in HBM/SMEM

    FP8 GEMM via cuBLAS (cublasGemmEx):
        cublasGemmEx(handle, transa, transb, m, n, k,
                     &alpha, A, CUDA_R_8F_E4M3, lda,
                             B, CUDA_R_8F_E4M3, ldb,
                     &beta,  C, CUDA_R_32F, ldc,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

### FP8 Scaling: Per-Tensor Quantization

    Unlike INT8 which commonly uses per-channel scaling, FP8 typically uses
    PER-TENSOR scaling (one scale per matrix):

        scale = amax(X) / max_fp8   where max_fp8 = 448 for E4M3

        X_fp8 = X_fp16 / scale
        (clamp to [-448, 448], then round to nearest FP8 representable value)

    WHY per-tensor for FP8?
        FP8 has 8 bits of exponent-encoded range — the format itself handles
        a wider dynamic range within a tensor. Per-tensor scaling is sufficient
        for most activations.

        But: outliers in a tensor can still cause all other values to be quantized
        to a few coarse levels. Per-tensor FP8 is less robust to outliers than
        per-channel INT8.

    PER-TENSOR SCALING IN PRACTICE:
        Delayed scaling: compute scale from the previous iteration's amax.
        Faster than computing amax every step (no extra reduction).
        NVIDIA TransformerEngine and H100-specific Flash Attention 3 use this.

    FP8 dequantization:
        Y_fp16 = Y_fp32_accum * (scale_A * scale_B)  [one multiply per output]
        Same as INT8 symmetric — scalar scale applied in GEMM epilogue.


##### PART 4 — INT4 PACKED WEIGHT STORAGE: THE MEMORY LAYOUT

### Why INT4 Is Attractive

    For large language models at inference time, the bottleneck is loading
    weights from HBM for each generated token. At batch_size=1, seq generation:
        - Each linear layer: load the full weight matrix once per token.
        - Weight memory bandwidth = (N_out × N_in × dtype_bytes) / time_step
        - For 7B model, FP16: 14 GB × 1 / time_step  ← bottleneck

    INT4 weights: 4 bits per parameter → 3.5 GB total.
    Loading 3.5 GB instead of 14 GB → 4× more tokens per second (bandwidth-bound).
    With group quantization (group_size=128): near-FP16 accuracy at INT4 size.

### Packed INT4 Storage Format

    Four-bit values are packed into 32-bit integers (8 values per INT32):

    PACKING (row-major, 8 weights per uint32):
        For weights W[0..7] each in [0, 15] (4-bit unsigned UINT4):
            packed = W[0] | (W[1] << 4) | (W[2] << 8) | ... | (W[7] << 28)

    Memory layout for a weight matrix [N_out, N_in] with group_size=128:
        Packed weights:  [N_out, N_in/8]   uint32 (each uint32 holds 8 weights)
        Scales:          [N_out, N_in/128] fp16   (one scale per group)
        Zero-points:     [N_out, N_in/128] uint8  (packed as 4-bit; 2 per byte)
                         OR stored as fp16 half_zp = -scale × (zp - 8)

    AWQSIZE:
        Weight matrix:  N_out × N_in/8 × 4 bytes  =  N_out × N_in / 2 bytes
        Scales:         N_out × N_in/128 × 2 bytes =  N_out × N_in / 64 bytes
        Zero-points:    N_out × N_in/128 × 0.5 bytes ≈ negligible
        Total ≈ N_out × N_in × 0.5 bytes  (vs FP16: N_out × N_in × 2 bytes)
        4× compression

### GPTQ Weight Format vs AWQ Weight Format

    Both GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2023) produce
    INT4 quantized weights, but with different column orderings:

    GPTQ: weights in the ORDER THEY WERE QUANTIZED (column-by-column, with
    permutation if reorder=True). The quantization order matters for GPTQ's
    Hessian-based update — later columns are quantized with the correction
    applied from earlier ones.

    AWQ: weights reordered by ACTIVATION MAGNITUDE for grouping.
    Salient weights (those multiplied by large activations) get higher precision
    (put in the high-bits group or scaled differently to reduce error).

    FROM THE DEQUANT KERNEL'S PERSPECTIVE: both are just [N_out, N_in/8] uint32
    with associated scales and zero-points. The packing format is identical.
    The difference is in how the scales were chosen, not in the kernel itself.


##### PART 5 — DEQUANTIZATION KERNELS: UNPACKING, SCALING & W4A16 GEMM

### The Dequantization Operation

    Given a packed weight tile and its scales/zero-points, dequantization
    reconstructs FP16 weights:

        For each group g with scale_g and zero_point_g:
            w_fp16 = (w_int4 - zero_point_g) × scale_g

    A CUDA kernel for this processes one int32 at a time:
        uint32 packed = W_packed[row][col/8];
        for (int i = 0; i < 8; i++) {
            uint8 w4 = (packed >> (4*i)) & 0xF;    // extract 4-bit weight
            int   w4s = (int)w4 - (int)zero_point; // subtract zero-point
            float wf  = w4s * scale;               // multiply by scale
            W_fp16[row][col + i] = __float2half(wf);
        }

    VECTORISED VERSION (4 weights at once using byte manipulation):
        Process two uint32 values (8 weights) using byte shuffle instructions.
        __dp4a for 4-element INT8 dot product (faster than scalar extraction).

### The W4A16 GEMM Pattern

    For inference, the dominant pattern is W4A16:
        W: INT4 packed weights (read from HBM once per token)
        A: FP16 activations (the input, relatively small)
        Output: FP16

    Two implementation strategies:

    STRATEGY 1 — Dequantize-then-GEMM:
        1. Dequant kernel: unpack W_int4 → W_fp16 (one pass over weights)
        2. FP16 GEMM: A_fp16 @ W_fp16
        Disadvantage: double HBM reads (dequant reads W_int4, GEMM reads W_fp16).
        Only viable if W_fp16 fits in L2/SMEM for reuse.

    STRATEGY 2 — Fused dequant + GEMM (production):
        SMEM tile of W_int4 is loaded (small: 4× less than FP16 tile).
        Dequantization happens in SMEM: unpack INT4 → FP16 (a few cycles).
        FP16 tensor core MMA on the dequantized tile.
        No separate dequant kernel — everything fused into the GEMM kernel.

        MEMORY BENEFIT:
            INT4 tile size = BLOCK_M × BLOCK_K / 2 bytes (vs BLOCK_M × BLOCK_K × 2 for FP16)
            4× smaller tile → can use 4× larger BLOCK_K for same SMEM → better reuse.

    EXLLAMAV2 and MARLIN (key production kernels for W4A16):
        EXLlamaV2: optimised W4A16 GEMM with packed INT4 → SMEM unpack → FP16 MMA.
        MARLIN: optimal static batching for W4A16 with persistent thread blocks.
        MARLIN achieves near-theoretical roofline for small batch sizes (batch≤32).

### The Dequant Kernel's Role in the Memory Bandwidth Roofline

    For batch_size=1 (single-token generation):
        Operation: y = x @ W  where x: [1, D], W: [D, D_out]
        FLOPs: 2 × D × D_out
        Bytes (INT4 W): D × D_out / 2 bytes  (4 bits per weight)
        Bytes (FP16 A): D × 2 bytes          (activation, tiny)

        Arithmetic intensity = 2 × D × D_out / (D × D_out / 2) = 4 FLOPs/Byte
        H100 ridge: 989 TFLOP/s / 3.35 TB/s = 295 FLOPs/Byte

        AI (4) << ridge (295) → severely memory-bound.

    The memory-bound kernel can only go faster by:
        a) Reducing bytes per weight (INT4 vs FP16: 4× fewer bytes → 4× faster)
        b) Increasing batch size (more FLOPs per byte loaded)
        c) Caching weights in faster memory (L2 reuse for longer sequences)

    For batch_size=32 (continuous batching in vLLM):
        FLOPs: 2 × 32 × D × D_out
        Bytes: same D × D_out / 2 bytes
        AI = 64 FLOPs/Byte — still well below ridge, but 16× better.


##### PART 6 — GPTQ AND AWQ: HOW QUANTIZATION METHODS AFFECT THE KERNEL

### GPTQ: Hessian-Guided Column Quantization

    GPTQ quantizes weight columns one at a time, using the Fisher information
    matrix (Hessian of the squared error) to minimise the impact on model output.

    ALGORITHM (per layer):
        1. Compute H = (X^T @ X) / N  [Hessian of the quantization error]
        2. For each column j (in quantization order):
            a. Quantize W[:, j] → W_quant[:, j] (round to nearest INT4)
            b. Compute the quantization error: δ = W[:, j] - W_quant[:, j]
            c. Update remaining columns to compensate:
               W[:, j+1:] -= (δ / H[j, j]) * H[j, j+1:]  [block update]
        3. Optionally: reorder columns by (H diagonal) for best grouping.

    KEY PROPERTY: GPTQ produces INT4 weights with GROUP-WISE scales/zero-points.
    The group_size (default 128) trades off between accuracy and scale overhead.

    DEQUANT KERNEL VIEW: GPTQ weights are [N_out, N_in/8] packed INT4.
    Scales and zero-points: [N_out, N_in/group_size] FP16.
    The dequant kernel is identical to any INT4 kernel — GPTQ's novelty is
    in the calibration algorithm, not the inference kernel.

### AWQ: Activation-Aware Quantization

    AWQ (Lin et al., 2023) observes that not all weights are equally important:
    weights corresponding to HIGH-MAGNITUDE input channels cause more error
    when quantized because their contribution to the output is larger.

    AWQ SOLUTION: per-channel scaling before quantization.
        For each input channel c with activation scale s_c:
            Scale UP the weight: W'[:, c] = W[:, c] × s_c
            Scale DOWN the activation: X'[:, c] = X[:, c] / s_c
        Now quantize W' → the important weights (large s_c) get finer resolution.

    The per-channel scaling factors are folded into the group scales:
        effective_scale_g = scale_g / s_c    (absorbed into the quantization scale)
    No runtime overhead: the dequant kernel sees normal INT4 + FP16 scales.

    RESULT: AWQ typically achieves ~0.5 perplexity points better than GPTQ
    on LLaMA at 4-bit, because important weights are quantized more carefully.

### SmoothQuant: Migrating Outliers from Activations to Weights

    SmoothQuant (Xiao et al., 2022) addresses the INT8 activation quantization
    problem: activations have large outliers per channel that ruin per-tensor scale.

    TRICK: migrate the outlier variance from activations to weights using a
    per-channel smooth factor α:
        X_smooth = X / diag(s)     (divide each activation channel by s_c)
        W_smooth = W × diag(s)     (multiply corresponding weight row by s_c)
        Y = X_smooth @ W_smooth = X @ W  (same output, exactly)

    Choose s_c to balance the activation and weight magnitudes:
        s_c = max(|X[:, c]|)^α / max(|W[:, c]|)^(1-α)   with α ≈ 0.5

    After smoothing: both X_smooth and W_smooth have smaller per-channel variance.
    Per-tensor INT8 quantization of both works much better.
    No runtime overhead: the smooth factor is folded into W at calibration time.


##### PART 7 — PRODUCTION QUANTIZATION TOOLKIT: APIS AND CALIBRATION WORKFLOW

### PyTorch Native Quantization (torch.ao.quantization)

    PyTorch provides a full quantization stack:
        from torch.ao.quantization import quantize_dynamic, get_default_qconfig
        from torch.ao.quantization import prepare, convert

    POST-TRAINING QUANTIZATION (PTQ):
        model_fp32 = ...
        model_fp32.qconfig = get_default_qconfig('x86')  # or 'qnnpack', 'fbgemm'
        prepare(model_fp32, inplace=True)
        # Run calibration data through model (updates activation statistics)
        for batch in calibration_loader:
            model_fp32(batch)
        convert(model_fp32, inplace=True)  # replaces fp32 ops with INT8

    Dynamic quantization (weights only, activations at runtime):
        model_int8 = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

### GPTQ via AutoGPTQ / llm-awq

    Install: pip install auto-gptq
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    quantize_config = BaseQuantizeConfig(
        bits=4,                # INT4 weights
        group_size=128,        # 128-weight groups for per-group scale
        desc_act=True,         # reorder by Hessian diagonal for better grouping
        sym=False,             # asymmetric quantization
    )

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
    examples = [...]     # small calibration dataset
    model.quantize(examples)   # runs GPTQ algorithm
    model.save_quantized(output_path)

    Loading for inference:
        model = AutoGPTQForCausalLM.from_quantized(output_path, use_triton=True)
        # use_triton=True: uses Triton-based W4A16 kernel (EXLlamaV2 backend)

### AWQ via llm-awq / AutoAWQ

    Install: pip install autoawq
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_pretrained(model_path)
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4}
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_path, safetensors=True)

    Inference backend: GEMM kernel from llm-awq (CUDA) or autoawq_kernels.
    Supports: LLaMA, Mistral, Qwen, Falcon, Phi, MPT.

### bitsandbytes (LLM.int8() and NF4)

    Install: pip install bitsandbytes
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # INT8 inference:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)

    # INT4 with NF4 (Normal Float 4 — non-linear INT4 format for QLoRA):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # nested quantization of scales
        bnb_4bit_quant_type="nf4",        # NF4 vs fp4
    )

    NF4 (NormalFloat4): optimises the quantization levels for normally-distributed
    weights (which most neural network weights are). Places more levels near 0
    and fewer at extremes, minimising expected quantization error for Gaussian data.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Quantization Arithmetic — Scale, Zero-Point & Error Analysis": {
        "description": (
            "Implement symmetric and asymmetric quantization from scratch for "
            "INT8 and INT4. Show the scale and zero-point computation. Quantize "
            "sample weight distributions and compute the per-element quantization "
            "error. Compare per-tensor vs per-channel vs per-group granularity. "
            "Demonstrate the outlier problem: one large value degrades quantization "
            "quality for all others. Show how clipping the range reduces overall MSE."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  QUANTIZATION ARITHMETIC — Scale, Zero-Point & Error Analysis")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Core quantization functions
# ─────────────────────────────────────────────────────────────────────

def quantize_symmetric(x, bits=8):
    """Symmetric INT quantization: zero_point = 0."""
    q_max  = (1 << (bits - 1)) - 1            # 127 for INT8, 7 for INT4
    scale  = float(np.abs(x).max()) / q_max if np.abs(x).max() > 0 else 1.0
    q      = np.clip(np.round(x / scale), -q_max, q_max).astype(np.int8 if bits==8 else np.int8)
    return q, scale, 0

def quantize_asymmetric(x, bits=8):
    """Asymmetric UINT quantization: zero_point ≠ 0."""
    q_max  = (1 << bits) - 1                  # 255 for UINT8, 15 for UINT4
    x_min, x_max = float(x.min()), float(x.max())
    scale  = (x_max - x_min) / q_max if x_max != x_min else 1.0
    zp     = int(np.round(-x_min / scale))
    zp     = np.clip(zp, 0, q_max)
    q      = np.clip(np.round(x / scale + zp), 0, q_max).astype(np.uint8)
    return q, scale, zp

def dequantize(q, scale, zp):
    """Reconstruct float values from quantized integers."""
    return (q.astype(np.float32) - zp) * scale

def quantize_error(x, x_hat):
    mse    = np.mean((x - x_hat)**2)
    max_e  = np.abs(x - x_hat).max()
    rel_e  = mse / (np.var(x) + 1e-10)
    return mse, max_e, rel_e


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Symmetric vs asymmetric on example distributions
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Symmetric vs Asymmetric: Distribution Fit")
print("━" * 68)
print()

distributions = [
    ("Symmetric Gaussian (μ=0, σ=1)", np.random.randn(256)),
    ("Positive only, ReLU output",    np.abs(np.random.randn(256))),
    ("Skewed positive (μ=0.5, σ=0.2)",np.random.normal(0.5, 0.2, 256)),
    ("With outlier (σ=1, one ×50)",   np.concatenate([np.random.randn(255), [50.0]])),
]

print(f"  {'Distribution':<36}  {'Sym MSE':>10}  {'Asym MSE':>10}  "
      f"{'Sym MaxErr':>11}  {'Asym MaxErr':>12}  {'Better'}")
print("  " + "─" * 82)

for name, x in distributions:
    x = x.astype(np.float32)
    q_sym,  s_sym,  zp_sym  = quantize_symmetric(x,  bits=8)
    q_asym, s_asym, zp_asym = quantize_asymmetric(x, bits=8)

    x_hat_sym  = dequantize(q_sym,  s_sym,  zp_sym)
    x_hat_asym = dequantize(q_asym, s_asym, zp_asym)

    mse_sym,  maxe_sym,  _ = quantize_error(x, x_hat_sym)
    mse_asym, maxe_asym, _ = quantize_error(x, x_hat_asym)

    better = "Asym" if mse_asym < mse_sym * 0.95 else \
             "Sym " if mse_sym  < mse_asym * 0.95 else "Tie "
    print(f"  {name:<36}  {mse_sym:>10.6f}  {mse_asym:>10.6f}  "
          f"{maxe_sym:>11.5f}  {maxe_asym:>12.5f}  {better}")

print()
print("  Asymmetric is better for one-sided (ReLU) distributions.")
print("  Both fail for outlier case: one extreme value forces large scale.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Granularity comparison — per-tensor vs per-channel vs per-group
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Granularity: Per-Tensor vs Per-Channel vs Per-Group (INT8)")
print("━" * 68)
print()

N_OUT, N_IN = 32, 256
W = np.random.randn(N_OUT, N_IN).astype(np.float32)
# Simulate per-channel variance: each output channel has different scale
for i in range(N_OUT):
    W[i] *= np.random.uniform(0.1, 5.0)  # different magnitude per channel

# PER-TENSOR
q_pt, s_pt, zp_pt = quantize_symmetric(W, bits=8)
W_hat_pt = dequantize(q_pt, s_pt, zp_pt)
mse_pt = np.mean((W - W_hat_pt)**2)

# PER-CHANNEL (one scale per output row)
W_hat_pc = np.zeros_like(W)
for i in range(N_OUT):
    q_i, s_i, zp_i = quantize_symmetric(W[i], bits=8)
    W_hat_pc[i] = dequantize(q_i, s_i, zp_i)
mse_pc = np.mean((W - W_hat_pc)**2)

# PER-GROUP (group_size=128, one scale per 128 consecutive input weights)
GROUP_SIZE = 128
W_hat_pg = np.zeros_like(W)
for i in range(N_OUT):
    for g_start in range(0, N_IN, GROUP_SIZE):
        g_end  = min(g_start + GROUP_SIZE, N_IN)
        group  = W[i, g_start:g_end]
        q_g, s_g, zp_g = quantize_symmetric(group, bits=8)
        W_hat_pg[i, g_start:g_end] = dequantize(q_g, s_g, zp_g)
mse_pg = np.mean((W - W_hat_pg)**2)

scale_mem_bytes = {
    "per-tensor":  1 * 4,                        # one FP32
    "per-channel": N_OUT * 4,                     # N_OUT FP32
    "per-group":   N_OUT * (N_IN//GROUP_SIZE) * 4,# many FP32
}

print(f"  Weight matrix: {N_OUT}×{N_IN}, INT8 symmetric, group_size={GROUP_SIZE}")
print()
print(f"  {'Granularity':<16}  {'MSE':>14}  {'vs per-tensor':>14}  "
      f"{'Scale mem':>12}  {'Scale overhead'}")
print("  " + "─" * 68)
for (gran, mse, mem) in [("per-tensor",  mse_pt, scale_mem_bytes["per-tensor"]),
                           ("per-channel", mse_pc, scale_mem_bytes["per-channel"]),
                           ("per-group",   mse_pg, scale_mem_bytes["per-group"])]:
    weight_bytes = N_OUT * N_IN   # INT8 = 1 byte each
    overhead_pct = mem / weight_bytes * 100
    ratio = mse / mse_pt
    print(f"  {gran:<16}  {mse:>14.8f}  {ratio:>13.4f}×  "
          f"{mem:>10} B  {overhead_pct:>6.2f}% of weight size")

print()
print(f"  Per-group reduces MSE by {mse_pt/mse_pg:.0f}× vs per-tensor "
      f"with only {scale_mem_bytes['per-group']/scale_mem_bytes['per-tensor']:.0f}× "
      f"more scale storage.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: The outlier problem and clipping solution
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Outlier Problem: One Extreme Value Ruins All Others")
print("━" * 68)
print()

N_elem = 512
x_base = np.random.randn(N_elem).astype(np.float32)

print(f"  Activations: N={N_elem}, mostly in [-3, 3].")
print(f"  Adding one outlier at value V forces large scale → all others quantized coarsely.")
print()
print(f"  {'Outlier V':>12}  {'Scale':>10}  {'MSE (no outlier)':>18}  "
      f"{'MaxErr':>10}  {'Levels used (of 256)'}")
print("  " + "─" * 70)

for outlier_v in [0, 5, 10, 20, 50, 100, 500]:
    x = x_base.copy()
    if outlier_v > 0:
        x[0] = outlier_v    # inject one outlier

    q, s, zp = quantize_symmetric(x, bits=8)
    x_hat    = dequantize(q, s, zp)

    # Error only for the non-outlier elements
    mask     = np.abs(x) <= 5.0
    mse_norm = np.mean((x[mask] - x_hat[mask])**2) if mask.sum() > 0 else 0
    max_e    = np.abs(x - x_hat).max()
    levels   = len(np.unique(q))

    print(f"  {outlier_v:>12}  {s:>10.5f}  {mse_norm:>18.8f}  "
          f"{max_e:>10.5f}  {levels:>20}")

print()
print("  FIX: clip outliers before quantization.")
print()
print(f"  {'Clip %ile':>12}  {'Clipped range':>16}  {'MSE (no outlier)':>18}  {'Outlier clipped?'}")
print("  " + "─" * 64)

x_outlier = x_base.copy(); x_outlier[0] = 100.0
for pct in [100, 99.9, 99, 95]:
    clip_val  = float(np.percentile(np.abs(x_outlier), pct))
    x_clipped = np.clip(x_outlier, -clip_val, clip_val)
    q_c, s_c, zp_c = quantize_symmetric(x_clipped, bits=8)
    x_hat_c  = dequantize(q_c, s_c, zp_c)
    mask     = np.abs(x_outlier) <= 5.0
    mse_norm = np.mean((x_outlier[mask] - x_hat_c[mask])**2)
    outlier_clipped = "yes ✅" if clip_val < 100.0 else "no  ❌"
    print(f"  {pct:>11.1f}%  {clip_val:>14.2f}  {mse_norm:>18.8f}  {outlier_clipped}")

print()
print("  Clipping at 99th percentile: outlier contained, MSE drops 100×.")
print("  This is the intuition behind SmoothQuant and GPTQ's calibration.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · FP8 Format Deep Dive — E4M3 vs E5M2, Encoding & Scaling": {
        "description": (
            "Implement FP8 E4M3 and E5M2 encoding and decoding from bit manipulation. "
            "Show the representable values for each format. Compare dynamic range "
            "and precision of E4M3 vs E5M2 at typical activation and gradient "
            "magnitudes. Demonstrate per-tensor FP8 scaling with delayed scaling. "
            "Verify round-trip E4M3 quantization error against FP16 baseline. "
            "Show why E4M3 suits forward activations and E5M2 suits gradients."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import struct

print("=" * 68)
print("  FP8 FORMAT DEEP DIVE — E4M3 vs E5M2 Encoding & Scaling")
print("=" * 68)
print()

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# FP8 encoding / decoding
# ─────────────────────────────────────────────────────────────────────

def fp8_e4m3_max():
    """Max representable finite value in FP8 E4M3."""
    # E4M3: exponent=1111, mantissa=110 → value = 2^(15-7) * (1+0.75) = 448
    return 448.0

def fp8_e5m2_max():
    """Max representable finite value in FP8 E5M2 (non-inf)."""
    # E5M2: largest finite exponent=11110, mantissa=11 → 57344
    return 57344.0

def encode_fp8_e4m3(x):
    """Encode a float to 8-bit E4M3 representation, return as uint8."""
    # Clamp to representable range
    x_clamped = float(np.clip(x, -fp8_e4m3_max(), fp8_e4m3_max()))
    if x_clamped == 0:
        return 0

    sign     = 1 if x_clamped < 0 else 0
    abs_val  = abs(x_clamped)

    # Find exponent
    exp_raw  = math.floor(math.log2(abs_val)) if abs_val > 0 else -14
    bias     = 7
    exp_biased = exp_raw + bias

    # Handle subnormals (exp_biased <= 0)
    if exp_biased <= 0:
        # subnormal: mantissa_val = abs_val / 2^(1-bias) / 8
        mantissa = round(abs_val / (2**(1-bias)) * 8)
        mantissa = min(mantissa, 7)
        exp_biased = 0
    else:
        exp_biased = min(exp_biased, 15)
        # Normal: mantissa from fractional part
        significand = abs_val / (2**exp_raw)  # in [1.0, 2.0)
        mantissa    = round((significand - 1.0) * 8)
        if mantissa >= 8:
            mantissa = 7   # avoid overflow, special NaN case
        # E4M3 special: 0b11111111 is NaN, avoid it
        if exp_biased == 15 and mantissa == 7:
            mantissa = 6

    return (sign << 7) | (exp_biased << 3) | mantissa

def decode_fp8_e4m3(bits):
    """Decode uint8 E4M3 bits to float."""
    bits = int(bits) & 0xFF
    sign_bit = (bits >> 7) & 1
    exp_bits = (bits >> 3) & 0xF
    mant_bits = bits & 0x7

    if exp_bits == 0b1111 and mant_bits == 0b111:
        return float('nan')   # NaN

    bias = 7
    if exp_bits == 0:
        # Subnormal
        val = mant_bits * (2**(1-bias)) / 8
    else:
        val = (1.0 + mant_bits/8.0) * (2**(exp_bits - bias))

    return -val if sign_bit else val

def encode_fp8_e5m2(x):
    """Encode a float to 8-bit E5M2 representation."""
    x_clamped = float(np.clip(x, -fp8_e5m2_max(), fp8_e5m2_max()))
    if x_clamped == 0: return 0
    sign    = 1 if x_clamped < 0 else 0
    abs_val = abs(x_clamped)
    exp_raw = math.floor(math.log2(abs_val)) if abs_val > 0 else -14
    bias    = 15
    exp_biased = max(0, min(exp_raw + bias, 30))  # 0-30, 31=inf/nan
    if exp_biased == 0:
        mantissa = round(abs_val / 2**(1-bias) * 4)
    else:
        significand = abs_val / (2**exp_raw)
        mantissa    = round((significand - 1.0) * 4)
    mantissa = min(3, max(0, mantissa))
    return (sign << 7) | (exp_biased << 2) | mantissa

def decode_fp8_e5m2(bits):
    bits = int(bits) & 0xFF
    sign_bit  = (bits >> 7) & 1
    exp_bits  = (bits >> 2) & 0x1F
    mant_bits = bits & 0x3
    if exp_bits == 31:
        return float('inf') if mant_bits == 0 else float('nan')
    bias = 15
    if exp_bits == 0:
        val = mant_bits * (2**(1-bias)) / 4
    else:
        val = (1.0 + mant_bits/4.0) * (2**(exp_bits - bias))
    return -val if sign_bit else val


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Representable values and format comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Format Properties: E4M3 vs E5M2 vs FP16")
print("━" * 68)
print()

formats = [
    ("FP8 E4M3", 4, 3, 7,   fp8_e4m3_max(), 2**-10, "forward (activations)"),
    ("FP8 E5M2", 5, 2, 15,  fp8_e5m2_max(), 2**-7,  "backward (gradients)"),
    ("FP16",     5,10, 15,  65504.0,        2**-10,  "general compute"),
    ("BF16",     8, 7, 127, 3.39e38,        2**-7,   "training"),
]

print(f"  {'Format':<12}  {'Exp':>4}  {'Mant':>5}  {'Bias':>5}  "
      f"{'Max val':>10}  {'Epsilon':>10}  {'Levels':>8}  {'Use case'}")
print("  " + "─" * 78)
for name, exp, mant, bias, maxv, eps, use in formats:
    levels = 2 ** (exp + mant + 1)
    print(f"  {name:<12}  {exp:>4}  {mant:>5}  {bias:>5}  "
          f"{maxv:>10.1f}  {eps:>10.6f}  {levels:>8}  {use}")

print()
print("  E4M3: 8 mantissa levels per binade (3 bits). Good precision near 1.0.")
print("  E5M2: 4 mantissa levels per binade (2 bits). Wide range for gradients.")
print("  Both accumulate in FP32 inside tensor core wgmma on H100.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Encoding round-trip at key values
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Encoding Round-Trip: Quantization Error at Key Values")
print("━" * 68)
print()

test_values = [-2.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.5, 10.0, 100.0, 400.0]

print(f"  {'Value':>8}  {'E4M3 encoded':>12}  {'E4M3 err%':>12}  "
      f"{'E5M2 encoded':>12}  {'E5M2 err%':>12}  {'FP16 err%'}")
print("  " + "─" * 74)

for v in test_values:
    # E4M3
    bits_e4m3 = encode_fp8_e4m3(v)
    v_e4m3    = decode_fp8_e4m3(bits_e4m3)
    err_e4m3  = abs(v_e4m3 - v) / (abs(v) + 1e-10) * 100

    # E5M2
    bits_e5m2 = encode_fp8_e5m2(v)
    v_e5m2    = decode_fp8_e5m2(bits_e5m2)
    err_e5m2  = abs(v_e5m2 - v) / (abs(v) + 1e-10) * 100

    # FP16 reference
    v_fp16    = float(np.float16(v))
    err_fp16  = abs(v_fp16 - v) / (abs(v) + 1e-10) * 100

    clip_e4m3 = " (clipped)" if abs(v) > fp8_e4m3_max() else ""
    print(f"  {v:>8.3f}  {v_e4m3:>12.5f}  {err_e4m3:>11.3f}%  "
          f"{v_e5m2:>12.5f}  {err_e5m2:>11.3f}%  {err_fp16:.3f}%{clip_e4m3}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Per-tensor FP8 scaling
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Per-Tensor FP8 Scaling: Forward and Delayed Scaling")
print("━" * 68)
print()

print("  FP8 quantization requires a scale factor to map FP16 range → FP8 range.")
print("  scale = amax(|X|) / max_fp8   (max_fp8 = 448 for E4M3)")
print()

N_t, D_t = 4, 16
activations_fp16 = np.random.randn(N_t, D_t).astype(np.float32)

# Add a simulated outlier column
activations_fp16[:, 0] *= 5.0    # one channel with larger magnitude

amax_curr = float(np.abs(activations_fp16).max())
scale_e4m3 = amax_curr / fp8_e4m3_max()

print(f"  Activation matrix: {N_t}×{D_t}, amax={amax_curr:.4f}")
print(f"  scale = {amax_curr:.4f} / {fp8_e4m3_max():.0f} = {scale_e4m3:.6f}")
print()

# Quantize to FP8 E4M3
act_scaled = activations_fp16 / scale_e4m3
act_fp8    = np.array([[encode_fp8_e4m3(v) for v in row]
                        for row in act_scaled])
act_decoded = np.array([[decode_fp8_e4m3(b) for b in row]
                         for row in act_fp8]) * scale_e4m3

mse_fp8    = np.mean((activations_fp16 - act_decoded)**2)
mse_fp16   = np.mean((activations_fp16 - activations_fp16.astype(np.float16).astype(np.float32))**2)

print(f"  FP8 E4M3 quantization MSE:  {mse_fp8:.6f}")
print(f"  FP16 quantization MSE:      {mse_fp16:.8f}")
print(f"  FP8 / FP16 error ratio:     {mse_fp8/max(mse_fp16,1e-10):.1f}×")
print()

# Delayed scaling simulation
print("  DELAYED SCALING (production approach):")
print("  Use previous iteration's amax to set this iteration's scale.")
print()
print(f"  {'Iteration':>10}  {'True amax':>12}  {'Delayed scale':>14}  "
      f"{'Effective max':>14}  {'Overflow?'}")
print("  " + "─" * 62)

amax_prev = 1.0   # initial scale
for it in range(6):
    # Simulate varying activation magnitude
    true_amax = float(np.abs(np.random.randn(16) * (1.0 + it * 0.3)).max())
    scale_used = amax_prev / fp8_e4m3_max()
    effective_max = true_amax / scale_used    # in FP8 units
    overflow = "⚠ YES" if effective_max > fp8_e4m3_max() else "✅ no"
    print(f"  {it:>10}  {true_amax:>12.4f}  {scale_used:>14.6f}  "
          f"{effective_max:>14.3f}  {overflow}")
    amax_prev = true_amax   # update for next iteration

print()
print("  Delayed scaling can overflow if amax grows rapidly between iterations.")
print("  Production solution: use exponential moving average of amax:")
print("    amax_ema = 0.9 * amax_ema + 0.1 * current_amax  (smooth the scale)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: E4M3 vs E5M2 for activations vs gradients
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — E4M3 for Activations, E5M2 for Gradients: Why?")
print("━" * 68)
print()

np.random.seed(77)
# Typical activation distribution (post-LayerNorm, mostly in [-3, 3])
activations = np.random.normal(0, 1, 1000).astype(np.float32)
# Typical gradient distribution (wide range, much sparser near zero)
gradients   = np.random.normal(0, 0.01, 1000).astype(np.float32)
# Add large gradient spikes
gradients[np.random.choice(1000, 10)] *= 100

distributions_fp8 = [
    ("Activations (post-LN)", activations, "E4M3 preferred"),
    ("Gradients (backward)",  gradients,   "E5M2 preferred"),
]

for dist_name, data, expected in distributions_fp8:
    amax = float(np.abs(data).max())
    print(f"  {dist_name}:  amax={amax:.4f}, std={data.std():.4f}")
    print(f"    Expected: {expected}")

    for fmt_name, enc_fn, dec_fn, max_fp8 in [
        ("E4M3", encode_fp8_e4m3, decode_fp8_e4m3, fp8_e4m3_max()),
        ("E5M2", encode_fp8_e5m2, decode_fp8_e5m2, fp8_e5m2_max()),
    ]:
        scale    = amax / max_fp8
        scaled   = (data / scale).clip(-max_fp8, max_fp8)
        encoded  = np.array([enc_fn(v) for v in scaled])
        decoded  = np.array([dec_fn(b) for b in encoded]) * scale
        mse      = np.mean((data - decoded)**2)
        n_clipped = np.sum(np.abs(data) > max_fp8 * scale)
        print(f"    {fmt_name}: MSE={mse:.6f}  clipped={n_clipped}/{len(data)}")

    print()
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · INT4 Packed Storage — Packing, Unpacking & Group Dequantization": {
        "description": (
            "Implement INT4 weight packing (8 values per uint32) and unpacking. "
            "Show the exact bit manipulation for GPTQ/AWQ weight layout. Implement "
            "group-wise dequantization: unpack INT4 → subtract zero-point → "
            "multiply by scale → FP16. Verify packed round-trip. Show the memory "
            "layout of packed weights plus scales and zero-points. Compute the "
            "effective memory reduction vs FP16."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  INT4 PACKED STORAGE — Packing, Unpacking & Group Dequantization")
print("=" * 68)
print()

np.random.seed(11)


# ─────────────────────────────────────────────────────────────────────
# INT4 packing utilities
# ─────────────────────────────────────────────────────────────────────

def pack_int4_row(weights_int4):
    """
    Pack a row of INT4 values (each in [0, 15]) into uint32 words.
    8 values per uint32. Input length must be multiple of 8.
    weights_int4: numpy array of uint8 values in [0, 15].
    Returns: numpy array of uint32.
    """
    assert len(weights_int4) % 8 == 0, "Length must be divisible by 8"
    n_words = len(weights_int4) // 8
    packed  = np.zeros(n_words, dtype=np.uint32)
    for i in range(n_words):
        word = np.uint32(0)
        for j in range(8):
            w4   = np.uint32(int(weights_int4[i*8 + j]) & 0xF)
            word |= (w4 << np.uint32(j * 4))
        packed[i] = word
    return packed

def unpack_int4_row(packed):
    """
    Unpack uint32 words back to individual INT4 values.
    Returns: numpy array of uint8.
    """
    unpacked = np.zeros(len(packed) * 8, dtype=np.uint8)
    for i, word in enumerate(packed):
        for j in range(8):
            unpacked[i*8 + j] = np.uint8((int(word) >> (j * 4)) & 0xF)
    return unpacked

def quantize_int4_group(weights, group_size=128):
    """
    Quantize a 2D weight matrix [N_out, N_in] to INT4 with per-group scales.
    Returns:
        packed_weights: [N_out, N_in/8] uint32
        scales:         [N_out, N_in/group_size] float32
        zero_points:    [N_out, N_in/group_size] uint8 (INT4 zero-point, packed as UINT4)
    """
    N_out, N_in = weights.shape
    assert N_in % group_size == 0
    n_groups    = N_in // group_size

    scales_out  = np.zeros((N_out, n_groups), dtype=np.float32)
    zp_out      = np.zeros((N_out, n_groups), dtype=np.uint8)
    q_weights   = np.zeros((N_out, N_in),     dtype=np.uint8)

    for i in range(N_out):
        for g in range(n_groups):
            g_start = g * group_size
            g_end   = g_start + group_size
            group   = weights[i, g_start:g_end].astype(np.float32)

            w_min, w_max = float(group.min()), float(group.max())
            scale    = (w_max - w_min) / 15.0 if w_max != w_min else 1.0
            zp       = int(round(-w_min / scale))
            zp       = max(0, min(15, zp))

            q_group  = np.clip(np.round(group / scale + zp), 0, 15).astype(np.uint8)
            q_weights[i, g_start:g_end] = q_group
            scales_out[i, g]  = scale
            zp_out[i, g]      = zp

    # Pack weights
    packed_weights = np.zeros((N_out, N_in // 8), dtype=np.uint32)
    for i in range(N_out):
        packed_weights[i] = pack_int4_row(q_weights[i])

    return packed_weights, scales_out, zp_out

def dequantize_int4_group(packed_weights, scales, zero_points, group_size=128):
    """
    Dequantize INT4 packed weights back to FP16.
    This is the core dequant kernel operation.
    """
    N_out = packed_weights.shape[0]
    N_in  = packed_weights.shape[1] * 8
    n_groups = N_in // group_size

    weights_fp16 = np.zeros((N_out, N_in), dtype=np.float32)

    for i in range(N_out):
        # Unpack INT4 values
        q_row = unpack_int4_row(packed_weights[i])   # (N_in,) uint8

        # Group-wise dequantization
        for g in range(n_groups):
            g_start = g * group_size
            g_end   = g_start + group_size
            scale   = float(scales[i, g])
            zp      = int(zero_points[i, g])
            q_group = q_row[g_start:g_end].astype(np.int32)
            weights_fp16[i, g_start:g_end] = (q_group - zp) * scale

    return weights_fp16.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Bit-level packing trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Bit-Level Packing: 8 INT4 Values → 1 uint32")
print("━" * 68)
print()

sample_weights = np.array([3, 12, 0, 15, 7, 1, 9, 5], dtype=np.uint8)
packed_word    = pack_int4_row(sample_weights)
unpacked_back  = unpack_int4_row(packed_word)

print(f"  Input INT4 values (8 weights): {sample_weights.tolist()}")
print()
packed_int  = int(packed_word[0])
print(f"  Packed as uint32: {packed_int} = 0x{packed_int:08X} = 0b{packed_int:032b}")
print()
print("  Bit layout (each 4-bit nibble holds one weight):")
for j in range(8):
    nibble = (packed_int >> (j * 4)) & 0xF
    bits   = f"{nibble:04b}"
    print(f"    Bits [{j*4+3:2d}:{j*4:2d}] = {bits} → weight[{j}] = {nibble} "
          f"{'✅' if nibble == int(sample_weights[j]) else '❌'}")
print()
print(f"  Unpacked back: {unpacked_back.tolist()}")
print(f"  Round-trip correct: {'✅' if np.array_equal(sample_weights, unpacked_back) else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Full quantize-pack-dequantize pipeline
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Full Pipeline: FP32 → INT4 Packed → FP32 Dequant")
print("━" * 68)
print()

N_OUT, N_IN, GS = 16, 256, 128
W_orig = np.random.randn(N_OUT, N_IN).astype(np.float32)

# Quantize
packed_W, scales_W, zp_W = quantize_int4_group(W_orig, group_size=GS)

# Dequantize
W_deq = dequantize_int4_group(packed_W, scales_W, zp_W, group_size=GS)

mse   = np.mean((W_orig - W_deq)**2)
max_e = np.abs(W_orig - W_deq).max()

print(f"  Weight matrix: {N_OUT}×{N_IN}, group_size={GS}")
print()
print(f"  Memory layout:")
print(f"    FP32 original:    {N_OUT*N_IN*4:>8} bytes  ({N_OUT*N_IN*4/1e3:.1f} KB)")
print(f"    INT4 packed:      {packed_W.nbytes:>8} bytes  ({packed_W.nbytes/1e3:.1f} KB)")
print(f"    Scales (FP32):    {scales_W.nbytes:>8} bytes")
print(f"    Zero-points:      {zp_W.nbytes:>8} bytes")
print(f"    Total quantized:  {packed_W.nbytes+scales_W.nbytes+zp_W.nbytes:>8} bytes")
print(f"    Compression:      {N_OUT*N_IN*4/(packed_W.nbytes+scales_W.nbytes+zp_W.nbytes):.2f}×")
print()
print(f"  Quantization error:")
print(f"    MSE:      {mse:.6f}")
print(f"    Max err:  {max_e:.6f}")
print(f"    Rel MSE:  {mse/np.var(W_orig)*100:.3f}% of weight variance")
print()

# Show first row dequant in detail
print("  First row dequant trace (first 16 weights):")
print(f"  {'i':>4}  {'Original':>10}  {'INT4':>6}  {'Scale':>8}  "
      f"{'ZP':>4}  {'Dequant':>10}  {'Error':>10}")
print("  " + "─" * 56)
q_row0 = unpack_int4_row(packed_W[0])
for i in range(16):
    g = i // GS
    scale = scales_W[0, g]
    zp    = zp_W[0, g]
    dq    = (int(q_row0[i]) - int(zp)) * scale
    err   = W_orig[0, i] - dq
    print(f"  {i:>4}  {W_orig[0,i]:>10.5f}  {q_row0[i]:>6}  "
          f"{scale:>8.5f}  {int(zp):>4}  {dq:>10.5f}  {err:>10.6f}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Memory comparison — FP16 vs INT8 vs INT4
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Memory Comparison: 7B Model Across Precisions")
print("━" * 68)
print()

LLAMA7B_PARAMS = 6_738_415_616   # approximate Llama-2-7B parameter count

formats_mem = [
    ("FP32",  4, 0, 0,     "baseline"),
    ("FP16",  2, 0, 0,     "standard inference"),
    ("BF16",  2, 0, 0,     "standard training"),
    ("INT8",  1, 2, 1,     "LLM.int8() (weight-only)"),
    ("INT4",  0.5, 2, 128, "GPTQ/AWQ, group=128"),
    ("NF4",   0.5, 4, 64,  "bitsandbytes NF4, group=64"),
]

print(f"  Model: Llama-2-7B ({LLAMA7B_PARAMS/1e9:.2f}B parameters)")
print()
print(f"  {'Format':<12}  {'Weight MB':>10}  {'Scale MB':>10}  "
      f"{'Total MB':>10}  {'vs FP16':>9}  {'Notes'}")
print("  " + "─" * 62)

fp16_size = LLAMA7B_PARAMS * 2 / 1e6

for fmt, w_bytes, scale_bytes_per_group, group_size, notes in formats_mem:
    weight_mb = LLAMA7B_PARAMS * w_bytes / 1e6
    if group_size > 0:
        n_groups  = LLAMA7B_PARAMS // group_size
        scale_mb  = n_groups * scale_bytes_per_group / 1e6
    else:
        scale_mb  = 0
    total_mb  = weight_mb + scale_mb
    ratio     = fp16_size / total_mb
    print(f"  {fmt:<12}  {weight_mb:>10.0f}  {scale_mb:>10.1f}  "
          f"{total_mb:>10.0f}  {ratio:>8.2f}×  {notes}")

print()
print("  For H100 80GB VRAM:")
fp16_mb  = LLAMA7B_PARAMS * 2 / 1e6
int4_mb  = LLAMA7B_PARAMS * 0.5 / 1e6 + (LLAMA7B_PARAMS // 128) * 2 / 1e6
batch_fp16 = int(80000 / fp16_mb * 1000) // 1000 * 1  # rough
print(f"  FP16:  {fp16_mb:.0f} MB weights → limited context/batch headroom")
print(f"  INT4:  {int4_mb:.0f} MB weights → {fp16_mb/int4_mb:.1f}× more VRAM for KV cache or larger batch")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Dequant Kernel — W4A16 GEMM, HBM Roofline & Throughput Model": {
        "description": (
            "Implement the W4A16 dequantization and GEMM computation. Show the "
            "fused dequant-inside-GEMM pattern vs separate dequant-then-GEMM. "
            "Compute HBM traffic and arithmetic intensity at batch sizes 1, 8, "
            "32, and 128. Show where each configuration sits on the H100 roofline. "
            "Model the token generation throughput (tokens/second) for INT4 vs "
            "FP16 inference across increasing batch sizes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  DEQUANT KERNEL — W4A16 GEMM, HBM Roofline & Throughput Model")
print("=" * 68)
print()

# Hardware constants
H100_PEAK_FP16_TFLOPS = 989.0    # TFLOP/s
H100_HBM_BW_GBS       = 3350.0   # GB/s
H100_RIDGE            = H100_PEAK_FP16_TFLOPS * 1e12 / (H100_HBM_BW_GBS * 1e9)

A100_PEAK_FP16_TFLOPS = 312.0
A100_HBM_BW_GBS       = 2000.0
A100_RIDGE            = A100_PEAK_FP16_TFLOPS * 1e12 / (A100_HBM_BW_GBS * 1e9)

# Llama-2-7B representative layer
D      = 4096
D_OUT  = 4096
D_FF   = 11008


# ─────────────────────────────────────────────────────────────────────
# W4A16 GEMM simulation
# ─────────────────────────────────────────────────────────────────────

def w4a16_gemm(activations_fp16, packed_weights, scales, zero_points,
               group_size=128):
    """
    Simulate W4A16 GEMM:
    activations: [B, D] float32
    packed_weights: [D_out, D//8] uint32
    Returns: [B, D_out] float32
    """
    B    = activations_fp16.shape[0]
    D    = activations_fp16.shape[1]
    D_out = packed_weights.shape[0]
    n_groups = D // group_size

    # Dequantize weights to FP32
    W_fp32 = np.zeros((D_out, D), dtype=np.float32)
    for i in range(D_out):
        for g in range(n_groups):
            g_start = g * group_size
            g_end   = g_start + group_size
            scale   = float(scales[i, g])
            zp      = int(zero_points[i, g])
            for k in range(g_start, g_end):
                word_idx = k // 8
                bit_pos  = (k % 8) * 4
                q4       = (int(packed_weights[i, word_idx]) >> bit_pos) & 0xF
                W_fp32[i, k] = (q4 - zp) * scale

    # FP16 GEMM
    return activations_fp16 @ W_fp32.T


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness verification
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — W4A16 GEMM Correctness")
print("━" * 68)
print()

np.random.seed(99)
D_s, D_out_s, GS = 128, 64, 64
B_test = 4

W_fp32 = np.random.randn(D_out_s, D_s).astype(np.float32)
X_fp16 = np.random.randn(B_test, D_s).astype(np.float32)

# Pack weights (reuse from Op3 functions)
def pack_row(w_int4):
    n_words = len(w_int4) // 8
    packed  = np.zeros(n_words, dtype=np.uint32)
    for i in range(n_words):
        word = np.uint32(0)
        for j in range(8):
            word |= (np.uint32(int(w_int4[i*8+j]) & 0xF) << np.uint32(j*4))
        packed[i] = word
    return packed

scales_s = np.zeros((D_out_s, D_s//GS), dtype=np.float32)
zp_s     = np.zeros((D_out_s, D_s//GS), dtype=np.uint8)
packed_s = np.zeros((D_out_s, D_s//8), dtype=np.uint32)

for i in range(D_out_s):
    for g in range(D_s // GS):
        gs, ge = g*GS, (g+1)*GS
        grp    = W_fp32[i, gs:ge]
        s      = (grp.max() - grp.min()) / 15.0
        z      = int(np.clip(np.round(-grp.min() / s), 0, 15))
        q_g    = np.clip(np.round(grp / s + z), 0, 15).astype(np.uint8)
        scales_s[i, g] = s
        zp_s[i, g]     = z
        packed_s[i, gs//8:ge//8] = pack_row(q_g)

Y_w4a16 = w4a16_gemm(X_fp16, packed_s, scales_s, zp_s, GS)
Y_fp16   = X_fp16 @ W_fp32.T  # reference (FP16 GEMM with original weights)

# Compute dequantization error separately
W_deq = np.zeros_like(W_fp32)
for i in range(D_out_s):
    for g in range(D_s//GS):
        gs, ge = g*GS, (g+1)*GS
        s, z   = scales_s[i,g], int(zp_s[i,g])
        for k in range(gs, ge):
            wi = k // 8; bi = (k % 8)*4
            q4 = (int(packed_s[i,wi]) >> bi) & 0xF
            W_deq[i,k] = (q4 - z) * s

quant_err = np.mean((W_fp32 - W_deq)**2)
gemm_err  = np.abs(Y_w4a16 - Y_fp16).max()
print(f"  Weight quantization MSE: {quant_err:.6f}")
print(f"  W4A16 GEMM max error vs FP16 reference: {gemm_err:.6f}")
print(f"  Correct (error < 0.5): {'✅' if gemm_err < 0.5 else '❌'}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: HBM traffic and arithmetic intensity
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — HBM Traffic & Arithmetic Intensity: W4A16 vs FP16")
print("━" * 68)
print()

def compute_traffic(B, D_in, D_out, precision, group_size=128):
    """HBM bytes for one GEMM layer."""
    flops = 2 * B * D_in * D_out

    if precision == "FP16":
        w_bytes  = D_in * D_out * 2        # FP16 weights
        a_bytes  = B * D_in * 2            # FP16 activations
        o_bytes  = B * D_out * 2           # FP16 output
        total    = w_bytes + a_bytes + o_bytes

    elif precision == "INT8":
        w_bytes  = D_in * D_out * 1        # INT8 weights
        s_bytes  = D_out * 4               # per-channel FP32 scales
        a_bytes  = B * D_in * 2            # FP16 activations (INT8-quant online)
        o_bytes  = B * D_out * 2           # FP16 output
        total    = w_bytes + s_bytes + a_bytes + o_bytes

    elif precision == "INT4":
        w_bytes  = D_in * D_out // 2       # INT4 packed (0.5 byte each)
        s_bytes  = D_out * (D_in//group_size) * 2  # FP16 per-group scales
        z_bytes  = D_out * (D_in//group_size) // 2  # INT4 zero-points
        a_bytes  = B * D_in * 2            # FP16 activations
        o_bytes  = B * D_out * 2           # FP16 output
        total    = w_bytes + s_bytes + z_bytes + a_bytes + o_bytes

    return flops, total

print(f"  Layer: D_in={D}, D_out={D_OUT}")
print(f"  H100 ridge: {H100_RIDGE:.0f} FLOPs/Byte  (above = compute-bound)")
print()
print(f"  {'Precision':<10}  {'Batch':>6}  {'FLOPs':>12}  {'HBM bytes':>12}  "
      f"{'AI':>8}  {'Bound':>8}  {'Time µs'}")
print("  " + "─" * 72)

for prec in ["FP16", "INT8", "INT4"]:
    for B_val in [1, 4, 16, 64]:
        flops, hbm = compute_traffic(B_val, D, D_OUT, prec)
        ai         = flops / hbm
        bound      = "cmp" if ai > H100_RIDGE else "mem"
        bnd_sym    = "🟢 cmp" if bound == "cmp" else "🔴 mem"
        t_us       = max(flops/(H100_PEAK_FP16_TFLOPS*1e12),
                         hbm/(H100_HBM_BW_GBS*1e9)) * 1e6
        print(f"  {prec:<10}  {B_val:>6}  {flops/1e9:>10.2f}G  "
              f"{hbm/1e6:>10.2f}M  {ai:>8.1f}  {bnd_sym:>8}  {t_us:>7.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Token throughput model — tokens/second vs batch size
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Token Throughput: tokens/second vs Batch Size")
print("━" * 68)
print()

print("  Model: Llama-2-7B (32 layers, 3 GEMMs per FFN, 4 GEMMs per attn)")
print("  H100 SXM5, decode step (generating one token per request)")
print()

N_LAYERS     = 32
N_GEMM_FFN   = 3     # gate, up, down
N_GEMM_ATTN  = 4     # Q, K, V, output proj
N_GEMMS      = N_LAYERS * (N_GEMM_FFN + N_GEMM_ATTN)

print(f"  Total GEMMs per token: {N_GEMMS}")
print()
print(f"  {'Precision':<10}  {'Batch':>6}  {'Total HBM (GB)':>16}  "
      f"{'Time (ms)':>11}  {'Tokens/sec':>12}  {'vs FP16 B=1'}")
print("  " + "─" * 68)

fp16_base_tps = None

for prec in ["FP16", "INT8", "INT4"]:
    for B_val in [1, 4, 16, 64, 256]:
        total_hbm_bytes = 0
        total_flops     = 0
        for _ in range(N_LAYERS):
            for d_in, d_out in [(D, D), (D, D_FF), (D_FF, D), (D, D), (D, D)]:
                f, h = compute_traffic(B_val, d_in, d_out, prec)
                total_hbm_bytes += h
                total_flops     += f

        t_mem_s = total_hbm_bytes / (H100_HBM_BW_GBS * 1e9)
        t_cmp_s = total_flops / (H100_PEAK_FP16_TFLOPS * 1e12)
        t_s     = max(t_mem_s, t_cmp_s)
        tps     = B_val / t_s    # tokens per second = batch / time_for_one_step

        if fp16_base_tps is None and prec == "FP16" and B_val == 1:
            fp16_base_tps = tps

        ratio_str = f"{tps/fp16_base_tps:.1f}×" if fp16_base_tps else "-"
        print(f"  {prec:<10}  {B_val:>6}  {total_hbm_bytes/1e9:>16.2f}  "
              f"{t_s*1000:>11.2f}  {tps:>12.0f}  {ratio_str:>10}")
    print()

print("  INT4 at batch=1 gives ~4× more tokens/sec vs FP16 (bandwidth-bound).")
print("  At large batch: compute-bound, precision matters less for throughput.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · GPTQ vs AWQ Accuracy Simulation — Calibration Impact & Group Size": {
        "description": (
            "Simulate GPTQ-style Hessian-guided weight quantization and compare "
            "against naive round-to-nearest on a small weight matrix. Show how "
            "the Hessian correction reduces quantization error in sensitive columns. "
            "Demonstrate the group size tradeoff: larger groups → worse accuracy, "
            "smaller groups → more overhead. Simulate AWQ per-channel smoothing "
            "and verify it reduces activation-induced quantization error. Produce "
            "a perplexity proxy comparison across quantization methods."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  GPTQ vs AWQ ACCURACY SIMULATION — Calibration Impact & Group Size")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Quantization helpers (INT4 asymmetric per-group)
# ─────────────────────────────────────────────────────────────────────

def quant_group_int4(group, zp_offset=8):
    """Quantize a group of weights to INT4 asymmetric."""
    g_min, g_max = float(group.min()), float(group.max())
    scale = (g_max - g_min) / 15.0 if g_max != g_min else 1e-6
    zp    = int(np.clip(np.round(-g_min / scale), 0, 15))
    q     = np.clip(np.round(group / scale + zp), 0, 15).astype(np.int32)
    deq   = (q - zp) * scale
    return deq.astype(np.float32), scale, zp

def naive_quantize_int4(W, group_size=128):
    """Round-to-nearest INT4 quantization, per-group."""
    N_out, N_in = W.shape
    W_hat = np.zeros_like(W)
    for i in range(N_out):
        for g_start in range(0, N_in, group_size):
            g_end = min(g_start + group_size, N_in)
            deq, _, _ = quant_group_int4(W[i, g_start:g_end])
            W_hat[i, g_start:g_end] = deq
    return W_hat


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: GPTQ column-wise correction simulation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — GPTQ vs Naive INT4: Hessian-Guided Correction")
print("━" * 68)
print()

print("  GPTQ key insight: when column j is quantized, the error propagates to")
print("  output. GPTQ compensates by adjusting the remaining columns using the")
print("  inverse Hessian, minimising the total output reconstruction error.")
print()

N_OUT, N_IN = 4, 64
GS = 32

# Simulate input activations for Hessian computation
N_CALIB = 128
X_calib = np.random.randn(N_CALIB, N_IN).astype(np.float32)
X_calib[:, 0] *= 10.0   # outlier column 0 has large activations
X_calib[:, 1] *= 8.0    # column 1 also large

# Hessian approximation: H ≈ X^T X / N
H = (X_calib.T @ X_calib) / N_CALIB    # (N_IN, N_IN)
H_diag = np.diag(H)    # column importance (diagonal of Hessian)

W_true = np.random.randn(N_OUT, N_IN).astype(np.float32)

# Compute reference output
Y_ref = X_calib @ W_true.T   # (N_CALIB, N_OUT)

# Naive quantization
W_naive = naive_quantize_int4(W_true, group_size=GS)
Y_naive = X_calib @ W_naive.T
err_naive = np.mean((Y_naive - Y_ref)**2)

# GPTQ-style: quantize columns in order of decreasing H diagonal
# (simple simulation: columns with small H[j,j] are less sensitive)
col_order = np.argsort(H_diag)[::-1]  # most important first

W_gptq = W_true.copy()
W_gptq_quant = np.zeros_like(W_true)

for idx, j in enumerate(col_order):
    # Quantize column j
    col_orig = W_gptq[:, j].copy()
    deq_col, s, z = quant_group_int4(col_orig)
    W_gptq_quant[:, j] = deq_col

    # Hessian-based correction: propagate quantization error to remaining columns
    delta = col_orig - deq_col       # quantization error in this column
    h_jj  = max(H[j, j], 1e-8)      # diagonal Hessian element

    remaining = col_order[idx+1:]   # columns not yet quantized
    for k in remaining:
        correction = delta * (H[j, k] / h_jj)
        W_gptq[:, k] -= correction  # compensate remaining columns

Y_gptq = X_calib @ W_gptq_quant.T
err_gptq = np.mean((Y_gptq - Y_ref)**2)

print(f"  Matrix: {N_OUT}×{N_IN}, group_size={GS}")
print(f"  Calibration: {N_CALIB} samples, outlier activations in columns 0 and 1")
print()
print(f"  Naive INT4 output MSE:  {err_naive:.6f}")
print(f"  GPTQ  INT4 output MSE:  {err_gptq:.6f}")
print(f"  GPTQ improvement:       {err_naive/err_gptq:.2f}× lower MSE")
print()

# Show per-column error comparison
print(f"  Per-column weight quantization error (first 8 columns):")
print(f"  {'Col':>5}  {'H[j,j]':>10}  {'Naive err':>12}  {'GPTQ err':>12}  "
      f"{'GPTQ better?':>13}  {'Importance'}")
print("  " + "─" * 62)
sorted_cols = np.argsort(H_diag)[::-1]
for j in range(min(8, N_IN)):
    e_naive = np.mean((W_true[:, j] - W_naive[:, j])**2)
    e_gptq  = np.mean((W_true[:, j] - W_gptq_quant[:, j])**2)
    better  = "✅" if e_gptq < e_naive * 0.95 else "~"
    importance = "HIGH" if H_diag[j] > np.median(H_diag) else "low"
    print(f"  {j:>5}  {H_diag[j]:>10.4f}  {e_naive:>12.6f}  "
          f"{e_gptq:>12.6f}  {better:>13}  {importance}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Group size tradeoff
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Group Size Tradeoff: Accuracy vs Scale Overhead")
print("━" * 68)
print()

N_OUT_g, N_IN_g = 64, 4096
W_g = np.random.randn(N_OUT_g, N_IN_g).astype(np.float32)
# Add per-column variance variation (realistic weight distribution)
for c in range(N_IN_g):
    W_g[:, c] *= np.random.uniform(0.2, 3.0)

X_test = np.random.randn(32, N_IN_g).astype(np.float32)
Y_true_g = X_test @ W_g.T

print(f"  Weight matrix: {N_OUT_g}×{N_IN_g}, activation: 32×{N_IN_g}")
print()
print(f"  {'Group size':>12}  {'N scales':>10}  {'Scale overhead':>16}  "
      f"{'Output MSE':>12}  {'Perplexity proxy'}")
print("  " + "─" * 64)

for gs in [16, 32, 64, 128, 256, 512, N_IN_g]:
    if N_IN_g % gs != 0:
        continue
    n_groups    = N_OUT_g * (N_IN_g // gs)
    scale_bytes = n_groups * 2    # FP16 scales
    w_bytes     = N_OUT_g * N_IN_g // 2   # INT4 packed
    overhead    = scale_bytes / w_bytes * 100

    W_hat_g = np.zeros_like(W_g)
    for i in range(N_OUT_g):
        for g_start in range(0, N_IN_g, gs):
            g_end = min(g_start + gs, N_IN_g)
            deq, _, _ = quant_group_int4(W_g[i, g_start:g_end])
            W_hat_g[i, g_start:g_end] = deq

    Y_hat_g = X_test @ W_hat_g.T
    mse_g   = np.mean((Y_hat_g - Y_true_g)**2)
    ppl_proxy = math.log(1 + mse_g)   # proxy for perplexity increase

    print(f"  {gs:>12}  {n_groups:>10}  {overhead:>14.2f}%  "
          f"{mse_g:>12.6f}  {ppl_proxy:>12.6f}")

print()
print("  Group size 128 is the standard default: good accuracy / overhead balance.")
print("  Smaller groups: better accuracy but more scale memory to load at inference.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: AWQ smoothing — activation-aware quantization
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — AWQ Smoothing: Migrating Activation Scale to Weights")
print("━" * 68)
print()

print("  AWQ: for each input channel c, compute activation scale s_c = max|X[:,c]|.")
print("  Smooth: W'[:,c] = W[:,c] * s_c  (weight absorbs activation scale).")
print("  At inference: X' = X / s_c (activations divided by same factor).")
print("  Net effect: Y = X @ W = X' @ W' (mathematically equivalent).")
print("  Benefit: W' is better conditioned for quantization (no outlier channels).")
print()

N_rows, N_cols = 32, 256
GS_awq = 128

W_awq  = np.random.randn(N_cols, N_rows).astype(np.float32)   # weight: [D_out, D_in]
X_awq  = np.random.randn(64, N_rows).astype(np.float32)        # activation: [B, D_in]

# Simulate outlier channels in activations
for c in [0, 3, 7, 15]:
    X_awq[:, c] *= 20.0   # large activation channels

# Channel-wise activation scale
act_scales = np.abs(X_awq).max(axis=0)   # (N_rows,)

# Naive quantization (no smoothing)
def quantize_layer_int4(W, gs):
    D_out, D_in = W.shape
    W_hat = np.zeros_like(W)
    for i in range(D_out):
        for g0 in range(0, D_in, gs):
            g1 = min(g0 + gs, D_in)
            deq, _, _ = quant_group_int4(W[i, g0:g1])
            W_hat[i, g0:g1] = deq
    return W_hat

W_naive_awq = quantize_layer_int4(W_awq, GS_awq)

# AWQ smoothing
alpha = 0.5   # smoothing strength
smooth_factors = act_scales ** alpha    # s_c^alpha per channel

W_smooth = W_awq * smooth_factors[np.newaxis, :]   # W' = W * diag(s^alpha)
W_smooth_quant = quantize_layer_int4(W_smooth, GS_awq)

# At inference: X_smooth = X / s^alpha, Y = X_smooth @ W_smooth_quant.T
X_smooth = X_awq / smooth_factors[np.newaxis, :]
Y_ref   = X_awq @ W_awq.T

Y_naive = X_awq   @ W_naive_awq.T
Y_awq   = X_smooth @ W_smooth_quant.T

mse_naive = np.mean((Y_naive - Y_ref)**2)
mse_awq   = np.mean((Y_awq   - Y_ref)**2)

print(f"  Matrix: W={W_awq.shape}, X={X_awq.shape}, group_size={GS_awq}, alpha={alpha}")
print()
print(f"  Outlier columns (20× activation magnitude): {[0, 3, 7, 15]}")
print()
print(f"  Naive INT4 output MSE:         {mse_naive:.6f}")
print(f"  AWQ-smoothed INT4 output MSE:  {mse_awq:.6f}")
print(f"  AWQ improvement:               {mse_naive/mse_awq:.2f}× lower MSE")
print()

# Show per-column analysis
print("  Channel smoothing factors (first 16 channels):")
print(f"  {'Channel':>8}  {'Act scale':>12}  {'Smooth factor':>14}  "
      f"{'W_col MSE naive':>17}  {'W_col MSE AWQ':>14}  {'Outlier?'}")
print("  " + "─" * 76)
for c in range(16):
    w_col_naive = W_awq[:, c]
    w_col_hat_naive = W_naive_awq[:, c]
    w_col_awq   = W_smooth[:, c]   # W' = W * s_c^alpha
    # Find which quantized group this column belongs to
    g = c // GS_awq
    deq_naive, _, _ = quant_group_int4(W_awq[:, c])
    deq_awq,   _, _ = quant_group_int4(W_smooth[:, c])
    mse_c_naive = np.mean((W_awq[:, c] - deq_naive)**2)
    mse_c_awq   = np.mean((W_awq[:, c] - deq_awq / smooth_factors[c])**2)
    outlier = "⚠ OUTLIER" if act_scales[c] > 5 else ""
    print(f"  {c:>8}  {act_scales[c]:>12.4f}  {smooth_factors[c]:>14.4f}  "
          f"{mse_c_naive:>17.6f}  {mse_c_awq:>14.6f}  {outlier}")

print()
print("  AWQ reduces quantization error most on outlier channels.")
print("  After smoothing, W' column magnitudes are more uniform → better per-group scale.")
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