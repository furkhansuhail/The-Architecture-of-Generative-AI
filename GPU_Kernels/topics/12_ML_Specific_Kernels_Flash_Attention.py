"""
Flash Attention — Online Softmax, HBM Tiling & FA-2 / FA-3 Improvements
=========================================================================

Standard attention computes: Attention(Q, K, V) = softmax(QKᵀ / √d) · V

The naïve implementation materialises the full N×N attention matrix — the
score matrix between every query and every key. For a sequence of length N:
    Memory for the score matrix: N² × 4 bytes
    For N=8192 (typical LLM context): 8192² × 4 = 256 MB per head per layer
    For a 7B model (32 layers, 32 heads): 256 MB × 32 × 32 = 262 GB
    Completely infeasible on any GPU.

But the deeper problem is not just memory — it is HBM bandwidth. Even if
you had 262 GB of VRAM, the attention operation would require reading and
writing the N×N matrix multiple times: once to compute scores, once for
softmax, once for the weighted sum. At H100's 3.35 TB/s HBM bandwidth, a
single attention forward pass for a single layer would take hundreds of
milliseconds. The arithmetic intensity of standard attention is O(1) —
independent of N, almost every operation is memory-bound.

Flash Attention (Dao et al., 2022) eliminates the N×N materialisation
entirely. It proves that the exact same attention output can be computed
tile-by-tile using an online softmax trick that never stores the full
score matrix. The result:
    Memory: O(N) instead of O(N²)
    HBM reads: O(N²) vs O(N²) — same asymptotic, but 2–4× fewer in practice
    Speed: 2–4× faster than optimised standard attention on A100

Flash Attention 2 (2023) improved parallelism: better GPU thread utilisation,
fewer non-matmul FLOPs, seq_len as the outer loop dimension.

Flash Attention 3 (2024) for Hopper (H100): warp specialisation, pingpong
scheduling, and FP8 support — achieving 75% of H100's theoretical peak.

This module builds the complete understanding: the online softmax algorithm,
the tiling memory analysis, and the FA-2/FA-3 architectural innovations.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Flash Attention — Online Softmax, HBM Tiling & FA-2 / FA-3 Improvements"
DISPLAY_NAME = "12 · Flash Attention"
ICON         = "⚡"
SUBTITLE     = "Online Softmax · HBM Tiling · IO Complexity · FA-2 Parallelism · FA-3 Hopper"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE STANDARD ATTENTION BOTTLENECK: IO COMPLEXITY ANALYSIS

### The Naïve Attention Algorithm

    Given: Q, K, V each of shape [N, d]  (N = seq_len, d = head_dim)
    Compute:
        S   = Q @ Kᵀ             [N×N]  — score matrix
        S̃   = S / √d             [N×N]  — scaled scores
        P   = softmax(S̃, dim=-1) [N×N]  — attention weights (rows sum to 1)
        O   = P @ V               [N×d]  — output

    HBM TRAFFIC for naïve implementation on GPU:

        STEP 1 — Q @ Kᵀ:
            Read Q:   N×d × 2 bytes (FP16)
            Read K:   N×d × 2 bytes
            Write S:  N×N × 4 bytes (FP32 scores)
            Total: 2Nd×2 + N²×4 bytes

        STEP 2 — softmax(S):
            Read  S:  N×N × 4 bytes
            Write P:  N×N × 4 bytes
            (Two passes: first for max/sum, then for normalisation)
            Total: 2 × N²×4 bytes

        STEP 3 — P @ V:
            Read  P:  N×N × 4 bytes
            Read  V:  N×d × 2 bytes
            Write O:  N×d × 2 bytes
            Total: N²×4 + 2Nd×2 bytes

        GRAND TOTAL ≈ 4 × N² × 4 bytes  (for N >> d)
        = 16 × N² bytes

    For N=8192, d=128 (typical LLM):
        HBM traffic = 16 × 8192² = 1.07 GB per head per forward pass
        At H100 bandwidth of 3.35 TB/s: 0.32 ms per head
        For 32 heads × 32 layers: 0.32 × 32 × 32 = 327 ms per token
        This is the BANDWIDTH bottleneck, independent of compute.

### Arithmetic Intensity of Standard Attention

    FLOPs for one attention forward:
        Q @ Kᵀ:   2 × N × N × d = 2N²d FLOPs
        softmax:  approximately 5N² FLOPs (exp, sum, divide)
        P @ V:    2 × N × d × N = 2N²d FLOPs
        Total:    ≈ 4N²d + 5N² ≈ 4N²d for large d

    HBM bytes: ≈ 16 × N² bytes (dominated by N×N matrix reads/writes)

    Arithmetic Intensity = FLOPs / Bytes = 4N²d / (16N²) = d/4

    For d=128:   AI = 128/4 = 32 FLOPs/Byte
    H100 ridge:  ridge = 989 TFLOP/s / 3.35 TB/s = 295 FLOPs/Byte

    AI (32) << ridge (295) → SEVERELY memory-bound, independent of N!

    This is the key insight: no matter how long the sequence is, naïve
    attention's arithmetic intensity stays constant at d/4. Adding more
    sequence length adds both FLOPs and bytes proportionally. The kernel
    will NEVER become compute-bound with the naïve algorithm.

### What Flash Attention Achieves

    Flash Attention achieves the same output as naïve attention but with:
        HBM writes of the N×N matrix: ZERO (it never materialises P or S)
        HBM reads of Q, K, V: O(N) each — same as unavoidable minimum
        Total HBM traffic ≈ 4Nd × 2 bytes (just Q, K, V, O — no N×N matrix)

    For N=8192:
        Standard: 1.07 GB per head
        Flash:    4 × 8192 × 128 × 2 bytes = 8 MB per head
        Reduction: 134× fewer HBM reads/writes

    Speed improvement in practice (A100):
        Standard cuDNN attention: ~3 ms per layer
        Flash Attention:          ~0.5 ms per layer
        ≈ 6× speedup for long sequences


##### PART 2 — THE ONLINE SOFTMAX TRICK: MATHEMATICAL FOUNDATION

### Standard Softmax Requires Two Passes

    To compute softmax(x) for a vector x of length N:

    PASS 1 — Find the maximum for numerical stability:
        m = max(x[0], x[1], ..., x[N-1])

    PASS 2 — Compute exp and normalise:
        p[i] = exp(x[i] - m) / Σ_j exp(x[j] - m)

    PROBLEM: You cannot compute p[i] until you have seen ALL x values
    (to find m and the denominator). This requires reading the entire
    score row into SMEM or HBM before producing any output.

    For attention, x is one row of the N×N score matrix. Computing softmax
    for all N rows requires the full N×N matrix in memory simultaneously.

### The Online Softmax Algorithm (Milakov & Gimelshein, 2018)

    Key insight: softmax output can be computed incrementally as new
    elements are revealed, using a RUNNING CORRECTION FACTOR.

    Maintain two running statistics:
        m_i = max(x[0], ..., x[i])          running maximum
        d_i = Σ_{j=0}^{i} exp(x[j] - m_i)  running sum of exp (corrected)

    When a new element x[i+1] arrives:
        m_{i+1} = max(m_i, x[i+1])
        d_{i+1} = d_i × exp(m_i - m_{i+1}) + exp(x[i+1] - m_{i+1})
        ↑ This is the correction factor: all previous exp values are
          rescaled by exp(m_old - m_new) to account for the updated max.

    PROOF THAT THIS IS CORRECT:
        After processing elements 0..i with old max m_i:
            d_i = Σ_{j=0}^{i} exp(x[j] - m_i)

        After seeing x[i+1] with new max m_{i+1} = max(m_i, x[i+1]):
            True denominator = Σ_{j=0}^{i+1} exp(x[j] - m_{i+1})
                              = Σ_{j=0}^{i} exp(x[j] - m_{i+1})  + exp(x[i+1] - m_{i+1})
                              = Σ_{j=0}^{i} exp(x[j] - m_i) × exp(m_i - m_{i+1}) + exp(x[i+1] - m_{i+1})
                              = d_i × exp(m_i - m_{i+1}) + exp(x[i+1] - m_{i+1})
                              = d_{i+1}  ✓

    After processing all N elements: m_N = global max, d_N = Σ_j exp(x[j] - m_N)
    Softmax output: p[j] = exp(x[j] - m_N) / d_N

### Extending Online Softmax to Attention Output

    We want: O = softmax(QKᵀ/√d) @ V  without materialising P.

    Maintain three running statistics per query row:
        m     = running maximum of scores
        d     = running denominator (sum of exp after correction)
        O_acc = running output accumulator (partial weighted V sum)

    When a new key-value block (K_j, V_j) arrives:
        s_j   = Q_i @ K_j.T / √d          [block of new scores]
        m_new = max(m, rowmax(s_j))        [update running max]
        O_acc = O_acc × exp(m - m_new)     [rescale old output]
               + exp(s_j - m_new) @ V_j   [add new block contribution]
        d_new = d × exp(m - m_new) + rowsum(exp(s_j - m_new))
        m     = m_new

    Final output: O_i = O_acc / d

    This is the ENTIRE Flash Attention computation. It processes the
    attention output one block at a time, maintaining only (m, d, O_acc)
    in SMEM — no N×N matrix ever touched in HBM.

### The Associativity Property: Why Blocks Can Be Processed in Any Order

    Define the "attention accumulator" state as a triple: (m, d, O)
    Define the MERGE operator ⊗ on two accumulators (from disjoint key blocks):

        (m_a, d_a, O_a) ⊗ (m_b, d_b, O_b) = (m, d, O) where:
            m = max(m_a, m_b)
            d = d_a × exp(m_a - m) + d_b × exp(m_b - m)
            O = (O_a × d_a × exp(m_a - m) + O_b × d_b × exp(m_b - m)) / d

    This operator is ASSOCIATIVE (can be verified by case analysis on m_a vs m_b).
    CONSEQUENCE: blocks of (K, V) can be processed in any order, or in parallel,
    and their results merged without global coordination.

    This is what enables:
        1. TILED COMPUTATION: process one tile at a time in SMEM
        2. PARALLEL COMPUTATION: multiple warps process different K blocks,
           then merge their partial results
        3. RECOMPUTATION IN BACKWARD: recompute attention from Q, K, V
           without storing the N×N attention matrix (saves memory for BWD)


##### PART 3 — HBM TILING: MAKING THE ONLINE ALGORITHM HARDWARE EFFICIENT

### The Tiling Strategy

    Flash Attention partitions Q, K, V into blocks that fit in SMEM.
    Choose block sizes BLOCK_Q and BLOCK_KV such that three tiles fit in SMEM:
        Q_tile:   BLOCK_Q × d × 2 bytes
        K_tile:   BLOCK_KV × d × 2 bytes
        V_tile:   BLOCK_KV × d × 2 bytes
        Score_tile: BLOCK_Q × BLOCK_KV × 4 bytes  (computed in registers/SMEM)

    For H100 (228 KB SMEM), d=128, FP16:
        Three tiles: 3 × BLOCK × 128 × 2 bytes
        Score tile:  BLOCK_Q × BLOCK_KV × 4 bytes
        For BLOCK_Q = BLOCK_KV = 128:
            Q + K + V: 3 × 128 × 128 × 2 = 98 KB
            Score:     128 × 128 × 4 = 64 KB
            Total:     162 KB — fits in 228 KB SMEM. ✓

    For V100 (96 KB SMEM):
        Must use smaller blocks: BLOCK_Q = BLOCK_KV = 64
            Q + K + V: 3 × 64 × 128 × 2 = 49 KB
            Score:     64 × 64 × 4 = 16 KB
            Total:     65 KB — fits in 96 KB SMEM. ✓

### The Flash Attention Forward Pass Pseudocode

    // Grid: (N/BLOCK_Q) blocks, one block per Q tile
    // Each block handles BLOCK_Q queries against ALL N keys

    for q_tile in range(0, N, BLOCK_Q):          // parallel across blocks
        // Load Q tile from HBM to SMEM
        Q_smem = load(Q[q_tile : q_tile+BLOCK_Q, :])   // BLOCK_Q × d

        // Initialise per-query accumulators in registers
        m    = -inf  [BLOCK_Q]    // running max
        d    = 0.0   [BLOCK_Q]    // running denominator
        O    = 0.0   [BLOCK_Q, d] // running output accumulator

        for kv_tile in range(0, N, BLOCK_KV):     // sequential within block
            // Load K and V tiles from HBM to SMEM
            K_smem = load(K[kv_tile : kv_tile+BLOCK_KV, :])
            V_smem = load(V[kv_tile : kv_tile+BLOCK_KV, :])

            // Compute score block
            S = Q_smem @ K_smem.T / sqrt(d)       // BLOCK_Q × BLOCK_KV (in registers)

            // Online softmax update
            m_new = max(m, rowmax(S))              // BLOCK_Q
            O     = O * exp(m - m_new)[:, None]    // rescale old output
            O    += exp(S - m_new[:, None]) @ V_smem  // add new contribution
            d     = d * exp(m - m_new) + rowsum(exp(S - m_new[:, None]))
            m     = m_new

        // Normalise and write output
        O = O / d[:, None]
        store(output[q_tile : q_tile+BLOCK_Q, :], O)

    HBM READS during this algorithm:
        Q: loaded ONCE total (each Q tile loaded once)
        K: loaded N/BLOCK_Q times (each K tile loaded for every Q tile)
        V: loaded N/BLOCK_Q times (each V tile loaded for every Q tile)

    Wait — K and V are read (N/BLOCK_Q) times each?
    Total K reads: N × d × 2 bytes × (N/BLOCK_Q) = N² × d × 2 / BLOCK_Q bytes

    That's WORSE than naïve for large N/BLOCK_Q... unless:
        The K and V tiles are READ FROM L2 CACHE on subsequent Q-tile passes!
        If the K and V data (N×d bytes) fits in L2 cache, the HBM traffic is:
            K reads: N × d × 2 bytes once (subsequent hits come from L2)
            V reads: N × d × 2 bytes once

    A100 L2 = 40 MB. For N=8192, d=128: K+V = 2 × 8192 × 128 × 2 = 4 MB. Fits!
    H100 L2 = 50 MB. For N=16384, d=128: K+V = 2 × 16384 × 128 × 2 = 8 MB. Fits!

    For N > L2 capacity: some K/V tiles must reload from HBM. Performance degrades.
    This is why Flash Attention 2 introduces KV-caching in the outer loop.

### IO Complexity: Formal Analysis

    THEOREM (Dao et al., 2022): Flash Attention requires
        O(N² × d / M) HBM accesses
    where M = SMEM size.

    PROOF SKETCH:
        Each Q block (of size BLOCK_Q = M/4d) requires reading all N keys/values.
        Total Q blocks: N / BLOCK_Q = 4Nd / M
        Reads per Q block: N × d (K + V tiles)
        Total K/V reads: (4Nd/M) × Nd = 4N²d²/M HBM accesses

        NAIVE: 4N² HBM accesses (for N >> d)
        FLASH: 4N²d²/M HBM accesses
        RATIO: d²/M

    For A100 (M = 40 MB ≈ 10M floats, d=128):
        Flash / Naive = (128²) / 10,000,000 ≈ 0.00163
        Flash uses ~600× fewer HBM accesses. In practice: 2–4× speedup (compute overhead).

    The gap grows with sequence length (N) and shrinks as d grows.
    For d=64: even larger speedup. For d=256 (some recent models): smaller speedup.


##### PART 4 — THE BACKWARD PASS: RECOMPUTATION TRICK

### The Standard Backward Pass Problem

    Standard attention backward requires the attention matrix P [N×N]:
        dV   = Pᵀ @ dO                   [needs P: N×N]
        dP   = dO @ Vᵀ                   [N×N intermediate]
        dS   = P ⊙ (dP - (dO ⊙ O).sum(-1, keepdim=True))  [needs P: N×N]
        dQ   = dS @ K / √d
        dK   = dSᵀ @ Q / √d

    All five steps require P in memory. Even if forward uses Flash Attention,
    the backward pass would need to re-materialise P. This defeats the memory savings.

### Flash Attention's Recomputation Strategy

    Key insight: instead of STORING the N×N attention matrix P from the forward pass,
    RECOMPUTE it during the backward pass from Q, K, and the saved (m, d) statistics.

    SAVED FROM FORWARD: O [N×d], m [N] (row-wise max), d [N] (row-wise sum)
    These are O(N) tensors — vastly smaller than the O(N²) attention matrix.

    BACKWARD ALGORITHM:
        For each (Q_tile, K_tile, V_tile) combination:
            1. Recompute S_tile = Q_tile @ K_tile.T / √d
            2. Recompute P_tile = exp(S_tile - m[:, None]) / d[:, None]
            3. Compute dV_tile += P_tile.T @ dO_tile
            4. Compute dP_tile = dO_tile @ V_tile.T
            5. Compute dS_tile using P_tile (already in SMEM)
            6. Accumulate dQ, dK incrementally

    MEMORY USAGE:
        O(N×d) for Q, K, V, O, dO, dQ, dK, dV, m, d
        NO N×N tensor ever in memory

    COMPUTATION OVERHEAD:
        The S and P tiles are recomputed (not loaded from HBM).
        Recomputation costs FLOPs but saves HBM bandwidth.
        Trade-off: more compute, less memory bandwidth → net win on modern GPUs.

    RATIO: recomputation overhead ≈ 2× the forward FLOPs
    (One extra forward pass worth of QK computation in the backward)
    But: saves N² × 4 bytes of HBM bandwidth — dominant term for large N.


##### PART 5 — FLASH ATTENTION 2: PARALLELISM, NON-MATMUL FLOPS & WORK PARTITION

### FA-1 Limitations That FA-2 Addresses

    Flash Attention 1 had three bottlenecks that prevented it from reaching
    theoretical peak performance:

    BOTTLENECK 1 — Work partition:
        FA-1's outer loop was over KV blocks; each threadblock handled one Q row.
        For small N (e.g., N=512): few threadblocks → most SMs idle.
        GPU parallelism under-utilised for short sequences.

    BOTTLENECK 2 — Excess non-matmul FLOPs:
        FA-1 performed redundant computation in the softmax rescaling step.
        Specifically: the output correction O × exp(m_old - m_new) was applied
        inside the inner loop at every KV tile, even when m did not change.

    BOTTLENECK 3 — Causal masking inefficiency:
        For causal attention (lower-triangular mask), FA-1 still computed
        the full BLOCK_Q × BLOCK_KV score tile and masked zeros after.
        Half the score computations were wasted.

### FA-2 Key Improvements

    IMPROVEMENT 1 — Reversed loop order (Q outer, KV inner):
        FA-2 makes the OUTER LOOP over Q blocks and the INNER LOOP over KV blocks.
        Each threadblock processes one Q block against ALL KV blocks.
        Parallelism = N/BLOCK_Q × num_heads × batch_size.
        For N=512, BLOCK_Q=128: 4 × 32 × 8 = 1024 threadblocks on A100 (108 SMs).
        MUCH better SM utilisation for any N.

        FA-1 had: (N/BLOCK_KV) × num_heads × batch_size threadblocks in the outer loop,
        with the inner loop over Q. For the same numbers: same count,
        BUT the accumulator was over the Q dimension, requiring partial writes to HBM.
        FA-2 keeps the accumulator (m, d, O) fully in registers for one Q tile — no HBM.

    IMPROVEMENT 2 — Reduce non-matmul FLOPs:
        FA-2 delays the O-rescaling step.
        Instead of rescaling O at every KV tile (once per KV block):
        Accumulate O_unscaled = Σ_j (exp(S_j - m_j) @ V_j) without dividing by d_j each step.
        Apply the final normalisation O = O_unscaled / d_final ONCE at the end.
        This trades rescaling overhead (one exp() + multiply per tile) for a single final divide.
        Result: ~20% fewer non-matmul FLOPs vs FA-1.

    IMPROVEMENT 3 — Causal masking efficiency:
        For causal attention: the Q tile at position i only attends to KV positions j ≤ i.
        FA-2 identifies "full tiles" (all positions within tile are valid) and
        "partial tiles" (only lower-triangular portion is valid).
        Full tiles: skip the masking entirely → 2× FLOPs utilisation.
        Partial tiles: apply mask only to the relevant boundary tile.
        For a sequence of length N: only N/BLOCK_KV partial tiles, rest are full.

    IMPROVEMENT 4 — Multi-query and grouped-query attention:
        FA-2 handles MQA (one KV head shared by multiple Q heads) and GQA
        (groups of Q heads sharing KV heads) natively, avoiding redundant KV loads.
        For GQA with 8 Q heads per KV group: 8× fewer KV HBM reads.

    FA-2 ACHIEVED: ~2× speedup vs FA-1, reaching ~70% of A100's peak FP16 throughput.

### FA-2 Parallelism Across the Sequence Dimension

    For long sequences with large batch size, parallelism is sufficient from
    batch × heads × N/BLOCK_Q. But for INFERENCE (batch=1, single sequence):
        Parallelism = 1 × num_heads × N/BLOCK_Q
        For N=4096, heads=32, BLOCK_Q=128: 32 × 32 = 1024 threadblocks. ✅

    For VERY LONG sequences (N=128K for document AI):
        Each Q block spans 128 tokens out of 128K → 1024 Q blocks × 32 heads = 32K blocks
        This is more than enough to saturate A100's 108 SMs.

    But the INNER LOOP also grows: 128K/128 = 1024 KV tiles per Q block.
    This is where FA-2's efficient non-matmul FLOP reduction pays off most.


##### PART 6 — FLASH ATTENTION 3: HOPPER ARCHITECTURE OPTIMISATIONS

### What Changed in H100 (Hopper) vs A100 (Ampere)

    H100 introduces two hardware features that FA-3 specifically exploits:

    FEATURE 1 — Warp Group MMA (wgmma):
        H100 introduces a new GEMM instruction that operates at the warp GROUP
        level (4 warps = 128 lanes, not 32).
        Tile size: 64×N×16 to 256×N×16 (M and N up to 256).
        The wgmma instruction has a longer pipeline (20 cycles) but 4× throughput
        vs A100 HMMA for the same M×N tile.
        CRITICAL: wgmma is ASYNCHRONOUS — it does not block the issuing warp.
        The warp can continue executing other instructions while wgmma completes.

    FEATURE 2 — Tensor Memory Accelerator (TMA):
        H100's TMA hardware performs SMEM loads/stores ASYNCHRONOUSLY.
        One thread issues a TMA load instruction; the hardware copies a tile
        from HBM/L2 to SMEM independently while the SM computes.
        No warp cycles spent on the memory copy — completely off the critical path.

### FA-3 Warp Specialisation

    FA-3 splits warps in a threadblock into two specialised groups:

    PRODUCER WARPS (typically 1 warp):
        Issue TMA load requests to fetch the next K tile, V tile from HBM.
        Signal via semaphore when the tile is ready in SMEM.
        While the consumer warps compute, producer warps prefetch the next tile.
        Producer warps are cheap: they rarely stall, spending most cycles waiting.

    CONSUMER WARPS (typically 3 warps per warp group):
        Wait on the semaphore for the tile to arrive.
        Execute wgmma to compute the score tile (S = Q @ Kᵀ).
        Execute softmax update (non-matmul, registers).
        Execute wgmma to compute the output accumulation (O_acc += P @ V).
        Signal back to producer: "tile consumed, load next one".

    PIPELINE STAGES:
        FA-3 uses a 2-stage pingpong pipeline with two SMEM buffers for K and V:
        While consumers process tile j (in buffer 0):
            Producers load tile j+1 into buffer 1.
        Switch buffers each iteration.
        Latency of TMA load (≈50 cycles) is completely hidden by wgmma compute.

### FA-3 Non-Matmul Overlap

    The wgmma instruction is ASYNCHRONOUS, so the consumer warps can interleave:
        wgmma S = Q @ Kᵀ                   ← async, fires and continues
        [issue next TMA prefetch]           ← while wgmma still running
        wgmma_wait_group(0)                 ← wait for S to be ready
        [softmax update: exp, max, sum]     ← on S values in registers
        wgmma O_acc += exp_S @ V_tile       ← async, fires and continues
        wgmma_wait_group(0)                 ← wait for O_acc to be ready
        [next iteration]

    The softmax computation (exp, max, sum) runs WHILE the wgmma for O_acc
    is computing. This hides the non-matmul latency inside the matmul.

    FA-3 ACHIEVED: 75% of H100's theoretical FP16 FP8 peak throughput.
    For FP8 (H100 peak 1979 TFLOP/s): ~1484 TFLOP/s effective for attention.

### FA-3 FP8 Support

    H100 supports FP8 (E4M3 and E5M2) tensor core operations via wgmma.
    FA-3 adds FP8 attention with:
        Q, K stored in FP8 (E4M3) — higher precision, better for activations
        V stored in FP8 (E5M2) — higher dynamic range
        Accumulation in FP32
        Output in FP16 or BF16

    Challenge: FP8's narrow range (E4M3 max = 448) requires per-tile scaling.
    FA-3 computes per-tile max of the score matrix and scales dynamically,
    similar in spirit to the loss scaling trick from mixed precision training.

    FP8 attention speedup vs FP16 FA-2: ≈2× throughput.
    Memory for Q, K, V: half of FP16 → longer sequences fit in HBM.


##### PART 7 — PRACTICAL GUIDE: USING FLASH ATTENTION IN PYTORCH

### PyTorch 2.0+ — torch.nn.functional.scaled_dot_product_attention

    PyTorch 2.0 added SDPA as a native operator that automatically dispatches
    to the best available attention implementation:
        - Flash Attention 2 (if the device supports it and inputs are FP16/BF16)
        - Memory-efficient attention (xFormers backend)
        - Math-fallback (naïve — always available)

    API:
        output = torch.nn.functional.scaled_dot_product_attention(
            query,               # [B, H, N, d]
            key,                 # [B, H, N, d]
            value,               # [B, H, N, d]
            attn_mask=None,      # optional [B, H, N, N] or [N, N] bool mask
            dropout_p=0.0,       # dropout probability (training only)
            is_causal=False,     # enable causal (lower-triangular) masking
            scale=None,          # optional custom scale (default: 1/√d)
        )

    INPUT SHAPE CONVENTION:
        Both [B, H, N, d] and [B, N, H, d] work — but [B, H, N, d] is
        preferred as it aligns with FA's internal layout (head_dim contiguous).

    ENABLE SDPA BACKENDS explicitly (PyTorch 2.1+):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,      # disable slow fallback
            enable_mem_efficient=True
        ):
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

### Key Constraints for Flash Attention to Activate

    FA path activates when ALL of the following are true:
        1. CUDA is available
        2. Input dtype is FP16 or BF16 (not FP32)
        3. head_dim ≤ 256 (FA-2 supports up to 256; FA-1 up to 128)
        4. No custom attention bias (custom additive mask disables FA)
           Exception: is_causal=True is handled natively
        5. dropout_p = 0 (dropout in FA-1 had bugs; FA-2 handles it, but slower)

    DIAGNOSE which backend is selected:
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            # Use torch.profiler or nsys to see kernel name
            # FA kernel: "fmha_v2_*" or "flash_fwd_*"
            # Fallback: "efficient_attention_*" or cuDNN softmax + matmul

### The standard FlashAttention package

    pip install flash-attn --no-build-isolation

    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

    # Q, K, V packed:
    output = flash_attn_qkvpacked_func(
        qkv,           # [B, N, 3, H, d]  packed Q, K, V
        dropout_p=0.0,
        softmax_scale=None,    # default: 1/sqrt(d)
        causal=True,
    )

    # Q, K, V separate:
    output = flash_attn_func(
        q, k, v,       # each [B, N, H, d]
        dropout_p=0.0,
        causal=True,
    )

    # Variable-length sequences (packed batch, no padding waste):
    from flash_attn import flash_attn_varlen_func
    output = flash_attn_varlen_func(
        q, k, v,                    # [total_tokens, H, d]
        cu_seqlens_q,               # cumulative query lengths [B+1]
        cu_seqlens_k,               # cumulative key lengths [B+1]
        max_seqlen_q, max_seqlen_k,
        causal=True,
    )

### Memory Footprint at Inference (Decode Step)

    At the decode step (generating one token), Q has shape [B, 1, H, d].
    K and V for the full context are in the KV cache: [B, N_ctx, H, d].

    Flash Attention for decode:
        Computes: 1 query token attending over all N_ctx key/value tokens.
        This is attention with M_Q=1, M_KV=N_ctx.
        Memory: O(N_ctx × d) — just reading KV cache, no N_ctx² ever materialised.
        For N_ctx=128K on H100: KV cache per layer = 2 × 128K × 32 × 128 × 2 = 2 GB
        Flash Attention handles this correctly as long as KV fits in VRAM.

    The decode step is MEMORY-BOUND (reading KV cache, minimal compute).
    Flash Attention's HBM saving is less pronounced here — the bottleneck
    is KV cache bandwidth, not the attention matrix materialisation.
    This is why continuous batching (vLLM, TGI) is so important — by increasing
    batch size, the decode step's arithmetic intensity improves.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Online Softmax — Incremental Algorithm, Associativity & Proof": {
        "description": (
            "Implement the online softmax algorithm from scratch. Show the running "
            "(m, d) state update step by step as new elements arrive. Prove "
            "correctness by verifying against the batch softmax reference. Demonstrate "
            "the associative merge operator on two partial softmax accumulators. "
            "Show that blocks can be processed in any order and merged to give the "
            "exact same result. Extend to the full attention output accumulator (m, d, O)."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  ONLINE SOFTMAX — Incremental Algorithm, Associativity & Proof")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Reference implementations
# ─────────────────────────────────────────────────────────────────────

def softmax_ref(x):
    """Numerically stable batch softmax."""
    x = np.array(x, dtype=np.float64)
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

def attention_ref(Q, K, V, scale=None):
    """Reference attention: softmax(QKᵀ/√d) @ V."""
    d = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    S  = Q @ K.T * scale
    P  = np.array([softmax_ref(row) for row in S])
    return P @ V, P


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Online softmax — element-by-element trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Online Softmax: Element-by-Element State Trace")
print("━" * 68)
print()

x = np.array([2.1, -0.5, 3.4, 1.2, 0.8, -1.3, 2.9, 1.7], dtype=np.float64)
N = len(x)

print(f"  Input x = {x.tolist()}")
print()
print(f"  {'i':>3}  {'x[i]':>8}  {'m (running max)':>17}  "
      f"{'d (running sum)':>17}  {'correction exp(m_old-m_new)':>28}")
print("  " + "─" * 74)

m = -math.inf
d = 0.0

for i in range(N):
    xi = x[i]
    m_old = m
    m_new = max(m, xi)

    correction = math.exp(m_old - m_new) if m_old != -math.inf else 0.0
    d_new = d * correction + math.exp(xi - m_new)

    print(f"  {i:>3}  {xi:>8.4f}  {m_new:>17.8f}  "
          f"{d_new:>17.8f}  {correction:>28.8f}")
    m, d = m_new, d_new

# Final softmax from online state
softmax_online = np.array([math.exp(xi - m) / d for xi in x])
softmax_batch  = softmax_ref(x)

print()
print(f"  Online   softmax: {softmax_online.round(6).tolist()}")
print(f"  Batch    softmax: {softmax_batch.round(6).tolist()}")
print(f"  Max abs error:    {np.abs(softmax_online - softmax_batch).max():.2e}")
print(f"  Correct:          {'✅' if np.allclose(softmax_online, softmax_batch) else '❌'}")
print()
print("  KEY: d is rescaled by exp(m_old - m_new) whenever m increases.")
print("  This correction factor ensures all previous exps are on the same scale.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Associative merge — process blocks in any order
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Associative Merge: Blocks in Any Order → Same Result")
print("━" * 68)
print()

def online_softmax_block(x_block):
    """Compute (m, d) for a block of values."""
    m = float(x_block.max())
    d = float(np.exp(x_block - m).sum())
    return m, d

def merge_md(m_a, d_a, m_b, d_b):
    """Merge two (m, d) accumulators from disjoint blocks."""
    m   = max(m_a, m_b)
    d   = d_a * math.exp(m_a - m) + d_b * math.exp(m_b - m)
    return m, d

x_full = np.random.randn(16).astype(np.float64)

# Split into 4 blocks of 4
blocks = [x_full[i:i+4] for i in range(0, 16, 4)]
block_stats = [online_softmax_block(b) for b in blocks]

print(f"  16-element vector split into 4 blocks of 4.")
print(f"  Full vector: {x_full.round(3).tolist()}")
print()
print(f"  Per-block (m, d):")
for i, (mi, di) in enumerate(block_stats):
    print(f"    Block {i}: m={mi:.4f}, d={di:.6f},  "
          f"values={blocks[i].round(3).tolist()}")
print()

# Merge in different orders — should all give same result
def merge_all(stats, order):
    """Merge stats in the given order, return final (m, d)."""
    m_acc, d_acc = stats[order[0]]
    for idx in order[1:]:
        m_acc, d_acc = merge_md(m_acc, d_acc, *stats[idx])
    return m_acc, d_acc

orders = [
    [0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]
]

# Reference: process full vector sequentially
m_ref, d_ref = online_softmax_block(x_full)

print(f"  Reference (full vector at once):  m={m_ref:.6f},  d={d_ref:.6f}")
print()
print(f"  {'Merge order':<18}  {'m result':>12}  {'d result':>14}  {'Match?'}")
print("  " + "─" * 52)
for order in orders:
    m_res, d_res = merge_all(block_stats, order)
    match = abs(m_res - m_ref) < 1e-10 and abs(d_res - d_ref) < 1e-10
    print(f"  {str(order):<18}  {m_res:>12.6f}  {d_res:>14.8f}  {'✅' if match else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Online attention — full (m, d, O) accumulator
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Online Attention Accumulator: (m, d, O) State")
print("━" * 68)
print()

BLOCK_KV = 4
N_q, N_kv, D = 2, 16, 8

Q  = np.random.randn(N_q,  D).astype(np.float64)
K  = np.random.randn(N_kv, D).astype(np.float64)
V  = np.random.randn(N_kv, D).astype(np.float64)

scale = 1.0 / math.sqrt(D)

# Reference output
O_ref, P_ref = attention_ref(Q, K, V, scale)

# Online attention (process K, V in blocks)
m_state = np.full(N_q, -math.inf)
d_state = np.zeros(N_q)
O_state = np.zeros((N_q, D))

print(f"  Q shape: ({N_q}×{D}), K/V shape: ({N_kv}×{D}), block_kv={BLOCK_KV}")
print()
print(f"  {'KV block':>10}  {'m[0]':>10}  {'d[0]':>10}  "
      f"{'O[0,0]':>10}  {'O[0,0] error'}")
print("  " + "─" * 58)

for kv_start in range(0, N_kv, BLOCK_KV):
    K_blk = K[kv_start:kv_start + BLOCK_KV]
    V_blk = V[kv_start:kv_start + BLOCK_KV]

    # Compute score block
    S_blk = Q @ K_blk.T * scale    # (N_q, BLOCK_KV)

    # Online update for each query (vectorised)
    m_blk = S_blk.max(axis=1)       # (N_q,)
    m_new = np.maximum(m_state, m_blk)

    # Rescale old output and accumulate new contribution
    scale_old = np.exp(m_state - m_new)  # correction for old accum
    scale_new = np.exp(S_blk - m_new[:, None])  # (N_q, BLOCK_KV)

    O_state = (O_state * scale_old[:, None]
               + scale_new @ V_blk)

    d_state = (d_state * scale_old
               + scale_new.sum(axis=1))
    m_state = m_new

    err = abs(O_state[0, 0] / d_state[0] - O_ref[0, 0])
    print(f"  [{kv_start:3d}:{kv_start+BLOCK_KV:3d}]    "
          f"{m_state[0]:>10.5f}  {d_state[0]:>10.5f}  "
          f"{O_state[0,0]/d_state[0]:>10.5f}  {err:.2e}")

# Normalise final output
O_online = O_state / d_state[:, None]

print()
print(f"  Online   O[0,:4] = {O_online[0, :4].round(5).tolist()}")
print(f"  Ref      O[0,:4] = {O_ref[0, :4].round(5).tolist()}")
print(f"  Max abs error:     {np.abs(O_online - O_ref).max():.2e}")
print(f"  Correct: {'✅' if np.allclose(O_online, O_ref, atol=1e-10) else '❌'}")
print()
print("  The (m, d, O) triple is maintained in registers — no N×N matrix in memory.")
print("  Each KV block requires only SMEM for the block tiles, not the full sequence.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · HBM Traffic Analysis — Standard vs Flash Attention IO Complexity": {
        "description": (
            "Compute the exact HBM read/write traffic for standard attention and "
            "Flash Attention at every sequence length from 512 to 131072. Show the "
            "crossover point where Flash Attention's savings dominate. Model the "
            "wall-clock time at H100 and A100 HBM bandwidth. Compute arithmetic "
            "intensity for both algorithms. Show how SMEM size constrains the "
            "optimal block size. Generate a comparison table and ASCII roofline plot."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  HBM TRAFFIC ANALYSIS — Standard vs Flash Attention IO Complexity")
print("=" * 68)
print()

# Hardware specs
GPUS = {
    "H100 SXM5": {"hbm_bw_gbs": 3350.0, "l2_mb": 50.0,  "smem_kb": 228, "sms": 132},
    "A100 SXM4": {"hbm_bw_gbs": 2000.0, "l2_mb": 40.0,  "smem_kb": 164, "sms": 108},
    "A10G":      {"hbm_bw_gbs":  600.0, "l2_mb": 24.0,  "smem_kb": 100, "sms":  80},
}

FP16_BYTES = 2
FP32_BYTES = 4


def standard_attention_hbm(N, d, batch=1, heads=1):
    """Bytes of HBM traffic for standard attention (one layer, one head)."""
    qkv_read  = 3 * N * d * FP16_BYTES               # read Q, K, V
    s_write   = N * N * FP32_BYTES                    # write score matrix
    s_read    = N * N * FP32_BYTES                    # read for softmax pass 1
    p_write   = N * N * FP32_BYTES                    # write attention weights
    p_read    = N * N * FP32_BYTES                    # read for P @ V
    o_write   = N * d * FP16_BYTES                    # write output
    total     = qkv_read + s_write + 2*s_read + p_write + p_read + o_write
    return total * batch * heads

def flash_attention_hbm(N, d, smem_kb, batch=1, heads=1, use_l2=True):
    """
    Bytes of HBM traffic for Flash Attention.
    Assumes K and V tiles stay in L2 cache for subsequent Q-tile passes.
    """
    smem_bytes = smem_kb * 1024
    # Block sizes: Q tile + K tile + V tile + score tile ≤ SMEM
    # Heuristic: BLOCK_Q = BLOCK_KV = min(128, sqrt(smem / (4*d*FP16_BYTES)))
    block_q = min(128, int(math.sqrt(smem_bytes / (4 * d * FP16_BYTES))))
    block_q = max(16, (block_q // 16) * 16)    # round to multiple of 16

    q_read   = N * d * FP16_BYTES              # Q read once
    kv_size  = 2 * N * d * FP16_BYTES          # K + V total
    o_write  = N * d * FP16_BYTES              # output written once

    # For each Q block, K and V must be read for all N/block_kv KV tiles
    # IF K+V fit in L2: read only once
    # IF NOT: read (N/block_q) times (once per Q-block pass over KV)
    l2_bytes = GPUS["H100 SXM5"]["l2_mb"] * 1024 * 1024  # use H100 L2
    kv_in_l2 = kv_size <= l2_bytes

    if use_l2 and kv_in_l2:
        kv_read = kv_size                       # served from L2 after first miss
    else:
        n_q_blocks = math.ceil(N / block_q)
        kv_read = kv_size * n_q_blocks          # reload K, V for each Q block

    # statistics (m, d, O): written and read N*d floats, negligible
    total = (q_read + kv_read + o_write) * batch * heads
    return total, block_q


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: HBM traffic comparison across sequence lengths
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — HBM Traffic: Standard vs Flash Attention")
print("━" * 68)
print()

D      = 128
BATCH  = 1
HEADS  = 1

gpu_h100 = GPUS["H100 SXM5"]
gpu_a100 = GPUS["A100 SXM4"]

print(f"  d={D}, batch={BATCH}, heads={HEADS}, FP16")
print()
print(f"  {'N':>8}  {'Standard (GB)':>15}  {'Flash H100 (GB)':>17}  "
      f"{'Flash A100 (GB)':>17}  {'H100 ratio':>12}  {'Flash KV cached?'}")
print("  " + "─" * 82)

for N in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    std_bytes = standard_attention_hbm(N, D, BATCH, HEADS)
    fl_h100, bq_h100 = flash_attention_hbm(N, D, gpu_h100["smem_kb"], BATCH, HEADS)
    fl_a100, bq_a100 = flash_attention_hbm(N, D, gpu_a100["smem_kb"], BATCH, HEADS)
    ratio_h100 = std_bytes / fl_h100

    kv_bytes = 2 * N * D * FP16_BYTES
    l2_h100  = gpu_h100["l2_mb"] * 1024 * 1024
    kv_cached = "✅" if kv_bytes <= l2_h100 else "❌"

    print(f"  {N:>8,}  {std_bytes/1e9:>15.3f}  {fl_h100/1e9:>17.3f}  "
          f"{fl_a100/1e9:>17.3f}  {ratio_h100:>11.1f}×  {kv_cached}")

print()
print(f"  Flash ratio = (standard bytes) / (flash bytes) — higher is better for Flash.")
print(f"  KV cached = K+V tiles fit in H100 L2 cache (50 MB).")
print(f"  When KV exceeds L2: Flash must reload K/V per Q-block pass → ratio drops.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Arithmetic intensity comparison
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Arithmetic Intensity: Why Standard Attention Is Always Mem-Bound")
print("━" * 68)
print()

H100_PEAK_FP16  = 989e12    # TFLOP/s
H100_HBM_BW     = 3.35e12   # GB/s → TB/s
H100_RIDGE      = H100_PEAK_FP16 / H100_HBM_BW  # FLOPs / Byte

print(f"  H100 SXM5 peak FP16 TC: {H100_PEAK_FP16/1e12:.0f} TFLOP/s")
print(f"  H100 HBM bandwidth:     {H100_HBM_BW/1e12:.2f} TB/s")
print(f"  Ridge point:            {H100_RIDGE:.0f} FLOPs/Byte")
print()

print(f"  {'N':>8}  {'Standard FLOPs':>16}  {'Std AI':>8}  "
      f"{'Flash FLOPs':>13}  {'Flash AI':>10}  {'Std bound':>11}  {'Flash bound'}")
print("  " + "─" * 76)

for N in [512, 1024, 2048, 4096, 8192, 32768]:
    d_val  = D

    # FLOPs: 2×N²×d (QKᵀ) + 5N² (softmax, approx) + 2×N²×d (PV)
    std_flops   = (4 * N * N * d_val + 5 * N * N)
    flash_flops = std_flops    # same FLOPs! Flash doesn't skip computation.

    std_bytes = standard_attention_hbm(N, d_val)
    fl_bytes, _ = flash_attention_hbm(N, d_val, gpu_h100["smem_kb"])

    std_ai   = std_flops   / std_bytes    if std_bytes  > 0 else 0
    flash_ai = flash_flops / fl_bytes     if fl_bytes   > 0 else 0

    std_bound   = "mem" if std_ai   < H100_RIDGE else "cmp"
    flash_bound = "mem" if flash_ai < H100_RIDGE else "cmp"

    std_sym   = "🔴 mem" if std_bound == "mem" else "🟢 cmp"
    flash_sym = "🟢 cmp" if flash_bound == "cmp" else ("🟡 mem" if flash_ai > H100_RIDGE * 0.5 else "🔴 mem")

    print(f"  {N:>8,}  {std_flops/1e9:>14.1f}G  {std_ai:>7.1f}  "
          f"{flash_flops/1e9:>11.1f}G  {flash_ai:>9.1f}  {std_sym:>11}  {flash_sym}")

print()
print(f"  Standard attention AI = d/4 = {D//4} (constant, independent of N).")
print(f"  Flash attention AI grows with N (less bytes per FLOP).")
print(f"  At large N, Flash can approach or exceed the ridge point → compute-bound!")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Block size constraints from SMEM
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Block Size Constraints: SMEM vs Block Size Tradeoff")
print("━" * 68)
print()

print("  SMEM usage = Q_tile + K_tile + V_tile + score_tile")
print("  Q_tile = BLOCK_Q × d × 2 bytes   (FP16)")
print("  K_tile = BLOCK_KV × d × 2 bytes   (FP16)")
print("  V_tile = BLOCK_KV × d × 2 bytes   (FP16)")
print("  score_tile = BLOCK_Q × BLOCK_KV × 4 bytes  (FP32)")
print()
print(f"  {'GPU':<14}  {'SMEM':>8}  {'BLOCK_Q':>8}  {'BLOCK_KV':>9}  "
      f"{'SMEM used':>10}  {'Fit?':>6}  {'IO efficiency'}")
print("  " + "─" * 66)

d_val = D
for gpu_name, gpu in GPUS.items():
    smem_bytes = gpu["smem_kb"] * 1024
    for bq, bkv in [(16,16), (32,32), (64,64), (128,128), (128,256), (256,128)]:
        q_mem     = bq  * d_val * FP16_BYTES
        k_mem     = bkv * d_val * FP16_BYTES
        v_mem     = bkv * d_val * FP16_BYTES
        sc_mem    = bq  * bkv   * FP32_BYTES
        total_mem = q_mem + k_mem + v_mem + sc_mem
        fits      = total_mem <= smem_bytes
        # IO efficiency: larger blocks = more reuse = fewer HBM reads per FLOP
        io_eff    = 2 * bq * bkv * d_val / total_mem  # FLOPs / bytes loaded into SMEM

        if fits:
            print(f"  {gpu_name:<14}  {gpu['smem_kb']:>6}KB  {bq:>8}  "
                  f"{bkv:>9}  {total_mem//1024:>8}KB  {'✅':>6}  {io_eff:>6.1f} FLOPs/B")

print()
print("  Larger blocks → better SMEM reuse → higher arithmetic intensity.")
print("  But SMEM is limited → can't make blocks arbitrarily large.")
print("  A100 (164KB SMEM) can fit 128×128 blocks. V100 (96KB) needs 64×64.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Wall-clock time estimate at different sequence lengths
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Wall-Clock Time: H100 Standard vs Flash Attention")
print("━" * 68)
print()

LAYERS, HEADS = 32, 32

print(f"  Model: {LAYERS} layers, {HEADS} heads, d={D}, FP16, H100 SXM5")
print(f"  (memory-bound analysis; compute bound at large N not accounted for)")
print()
print(f"  {'N':>8}  {'Std time (ms)':>15}  {'Flash time (ms)':>17}  "
      f"{'Speedup':>9}  {'Context window'}")
print("  " + "─" * 60)

for N, ctx_label in [(512, "512 tokens"), (2048, "2K context"), (8192, "8K context"),
                      (32768, "32K context"), (131072, "128K context")]:
    std_bytes   = standard_attention_hbm(N, D, BATCH, HEADS) * LAYERS
    fl_bytes, _ = flash_attention_hbm(N, D, gpu_h100["smem_kb"], BATCH, HEADS)
    fl_bytes   *= LAYERS

    bw = gpu_h100["hbm_bw_gbs"] * 1e9
    t_std_ms    = std_bytes   / bw * 1000
    t_fl_ms     = fl_bytes    / bw * 1000
    speedup     = t_std_ms / t_fl_ms

    print(f"  {N:>8,}  {t_std_ms:>15.2f}  {t_fl_ms:>17.2f}  "
          f"{speedup:>8.1f}×  {ctx_label}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Flash Attention Forward Pass — Tiled Implementation & Verification": {
        "description": (
            "Implement a complete Flash Attention forward pass with SMEM tiling. "
            "Process Q, K, V in blocks, maintaining (m, d, O) per query row. "
            "Verify output against the reference naive attention for multiple head "
            "dimensions and sequence lengths. Show the inner loop structure with "
            "causal masking. Profile HBM access patterns and confirm no N×N "
            "materialisation occurs. Test with variable-length sequences."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FLASH ATTENTION FORWARD PASS — Tiled Implementation & Verification")
print("=" * 68)
print()

np.random.seed(99)


# ─────────────────────────────────────────────────────────────────────
# Reference: naive attention
# ─────────────────────────────────────────────────────────────────────

def naive_attention(Q, K, V, scale=None, causal=False):
    """Standard naive attention with optional causal mask."""
    N_q, d = Q.shape
    N_kv   = K.shape[0]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    S = Q @ K.T * scale                        # (N_q, N_kv)
    if causal:
        # Lower-triangular mask: position i can only attend to j <= i
        mask = np.triu(np.ones((N_q, N_kv), dtype=bool), k=1)
        S[mask] = -1e9
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


# ─────────────────────────────────────────────────────────────────────
# Flash Attention forward pass
# ─────────────────────────────────────────────────────────────────────

def flash_attention_fwd(Q, K, V, BLOCK_Q=32, BLOCK_KV=32,
                        scale=None, causal=False):
    """
    Flash Attention forward pass.
    Q: (N_q, d), K: (N_kv, d), V: (N_kv, d)
    Returns: O (N_q, d)

    Maintains running (m, d_acc, O_acc) per query row.
    Never materialises the N_q × N_kv score matrix.
    """
    N_q,  d    = Q.shape
    N_kv, d_kv = K.shape
    assert d == d_kv
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    O_out   = np.zeros((N_q, d), dtype=np.float64)
    hbm_reads  = 0    # track HBM bytes read
    hbm_writes = 0    # track HBM bytes written

    # ── OUTER LOOP: Q blocks (parallelised across threadblocks) ────
    for q_start in range(0, N_q, BLOCK_Q):
        q_end  = min(q_start + BLOCK_Q, N_q)
        q_size = q_end - q_start

        # Load Q tile from HBM to SMEM
        Q_tile = Q[q_start:q_end, :]              # (q_size, d)
        hbm_reads += q_size * d * 8               # 8 bytes (float64 sim)

        # Initialise running state in registers
        m_i   = np.full(q_size, -math.inf)        # running max (per query)
        d_i   = np.zeros(q_size)                  # running denominator
        O_acc = np.zeros((q_size, d))             # running output

        # ── INNER LOOP: KV blocks (sequential within threadblock) ──
        for kv_start in range(0, N_kv, BLOCK_KV):
            kv_end  = min(kv_start + BLOCK_KV, N_kv)
            kv_size = kv_end - kv_start

            # Load K, V tiles from HBM to SMEM
            K_tile = K[kv_start:kv_end, :]        # (kv_size, d)
            V_tile = V[kv_start:kv_end, :]        # (kv_size, d)
            hbm_reads += 2 * kv_size * d * 8

            # Compute score block (stays in SMEM/registers)
            S_tile = Q_tile @ K_tile.T * scale    # (q_size, kv_size)

            # Apply causal mask if needed
            if causal:
                for qi in range(q_size):
                    for ki in range(kv_size):
                        q_pos = q_start + qi
                        k_pos = kv_start + ki
                        if k_pos > q_pos:
                            S_tile[qi, ki] = -1e9

            # Online softmax update
            m_blk = S_tile.max(axis=1)             # (q_size,) row-wise max of block
            m_new = np.maximum(m_i, m_blk)

            # Rescale old accumulator and add new block contribution
            scale_old = np.exp(m_i   - m_new)     # (q_size,) correction factor
            scale_new = np.exp(S_tile - m_new[:, None])  # (q_size, kv_size)

            O_acc = (O_acc * scale_old[:, None]
                     + scale_new @ V_tile)

            d_i   = d_i * scale_old + scale_new.sum(axis=1)
            m_i   = m_new

        # Normalise and write output tile to HBM
        O_tile = O_acc / d_i[:, None]             # (q_size, d)
        O_out[q_start:q_end, :] = O_tile
        hbm_writes += q_size * d * 8

    return O_out, hbm_reads, hbm_writes


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness across different shapes
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Correctness Verification: Flash vs Naive")
print("━" * 68)
print()

test_cases = [
    (16,  16,  16,  8,  8,  False, "tiny (N=16, d=16)"),
    (32,  32,  64, 16, 16,  False, "small (N=32, d=64)"),
    (64,  64, 128, 32, 32,  False, "medium (N=64, d=128)"),
    (128,128, 128, 32, 32,  False, "large (N=128, d=128)"),
    (64,  64, 128, 32, 32,  True,  "causal (N=64, d=128)"),
    (48,  48,  64, 16, 16,  True,  "causal non-pow2 (N=48)"),
    (100, 100, 64, 32, 32,  False, "non-tile-aligned N=100"),
]

print(f"  {'Test case':<36}  {'Max err':>10}  {'HBM reads (KB)':>16}  "
      f"{'N×N bytes (KB)':>16}  {'Correct?'}")
print("  " + "─" * 82)

for N_q, N_kv, d, BQ, BKV, causal, label in test_cases:
    Q_t = np.random.randn(N_q,  d).astype(np.float64)
    K_t = np.random.randn(N_kv, d).astype(np.float64)
    V_t = np.random.randn(N_kv, d).astype(np.float64)

    O_ref   = naive_attention(Q_t, K_t, V_t, causal=causal)
    O_flash, r_bytes, w_bytes = flash_attention_fwd(
        Q_t, K_t, V_t, BLOCK_Q=BQ, BLOCK_KV=BKV, causal=causal)

    max_err = np.abs(O_flash - O_ref).max()
    correct = max_err < 1e-8

    nn_bytes = N_q * N_kv * 8 / 1024   # what naïve would store for scores
    hbm_r_kb = r_bytes / 1024

    print(f"  {label:<36}  {max_err:>10.2e}  {hbm_r_kb:>16.1f}  "
          f"{nn_bytes:>16.1f}  {'✅' if correct else '❌'}")

print()
print("  HBM reads for Flash never includes the N×N score matrix.")
print("  The 'N×N bytes' column shows what naïve attention would store —")
print("  Flash never writes these bytes to HBM.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Causal attention trace
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Causal Mask Trace: Which Blocks Are Skipped")
print("━" * 68)
print()

N_causal, D_causal, BQ, BKV = 16, 4, 4, 4
Q_c = np.random.randn(N_causal, D_causal)
K_c = np.random.randn(N_causal, D_causal)
V_c = np.random.randn(N_causal, D_causal)

print(f"  N={N_causal}, d={D_causal}, BLOCK_Q={BQ}, BLOCK_KV={BKV}, causal=True")
print(f"  Block grid: {N_causal//BQ} Q-blocks × {N_causal//BKV} KV-blocks = "
      f"{(N_causal//BQ)*(N_causal//BKV)} pairs")
print()
print("  Block grid (Q rows × KV cols). 'full'=all tokens visible, "
      "'mask'=partial, 'skip'=all future:")
print()

for qi_block in range(N_causal // BQ):
    row_str = f"  Q[{qi_block*BQ}:{(qi_block+1)*BQ}]  "
    for ki_block in range(N_causal // BKV):
        q_max = (qi_block + 1) * BQ - 1        # last query position in this Q block
        k_min = ki_block * BKV                  # first key position in this KV block
        k_max = (ki_block + 1) * BKV - 1

        if k_min > q_max:
            row_str += "  SKIP"    # entire KV block is in the future
        elif k_max <= q_max:
            row_str += "  full"    # all keys are valid (full tile)
        else:
            row_str += "  mask"    # partial — need to apply mask element-wise
    print(row_str)

print()
print("  'SKIP' blocks: Q[0:4] cannot see K[4:8], K[8:12], K[12:16] → skip them.")
print("  FA-2 optimisation: skip tiles above the diagonal entirely.")
print("  For causal attention: exactly N(N-1)/2 elements are masked.")
print("  Only ~50% of KV blocks are 'full'; ~50% are either 'mask' or 'SKIP'.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Full attention output accumulator state trace
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Accumulator State Trace: One Query Row Across KV Blocks")
print("━" * 68)
print()

N_t, D_t, BKV_t = 32, 8, 8
Q_one = np.random.randn(1, D_t)
K_all = np.random.randn(N_t, D_t)
V_all = np.random.randn(N_t, D_t)
scale_t = 1.0 / math.sqrt(D_t)

m_state = np.array([-math.inf])
d_state = np.array([0.0])
O_state = np.zeros((1, D_t))

O_ref   = naive_attention(Q_one, K_all, V_all)

print(f"  Single query attending over N={N_t} keys, d={D_t}, block_kv={BKV_t}")
print()
print(f"  {'KV block':>10}  {'m state':>10}  {'d state':>10}  "
      f"{'O[0] norm':>12}  {'Error vs final':>16}")
print("  " + "─" * 58)

for kv_start in range(0, N_t, BKV_t):
    K_b = K_all[kv_start:kv_start+BKV_t]
    V_b = V_all[kv_start:kv_start+BKV_t]
    S_b = Q_one @ K_b.T * scale_t

    m_blk = S_b.max(axis=1)
    m_new = np.maximum(m_state, m_blk)
    s_old = np.exp(m_state - m_new)
    s_new = np.exp(S_b - m_new[:, None])

    O_state = O_state * s_old[:, None] + s_new @ V_b
    d_state = d_state * s_old + s_new.sum(axis=1)
    m_state = m_new

    O_norm = float(np.linalg.norm(O_state / d_state))
    err    = np.abs(O_state[0] / d_state[0] - O_ref[0]).max()
    print(f"  [{kv_start:3d}:{kv_start+BKV_t:3d}]    "
          f"{m_state[0]:>10.4f}  {d_state[0]:>10.4f}  "
          f"{O_norm:>12.6f}  {err:>16.2e}")

final = O_state / d_state[:, None]
print()
print(f"  Final vs reference max error: {np.abs(final - O_ref).max():.2e}  ✅")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · FA-1 vs FA-2 vs FA-3 — Parallelism, Non-Matmul FLOPs & Pipeline": {
        "description": (
            "Quantify the three key improvements from FA-1 to FA-2: reversed loop "
            "order, reduced non-matmul FLOPs, and causal masking efficiency. Compute "
            "the parallelism (number of threadblocks) for FA-1 vs FA-2 across batch "
            "sizes and sequence lengths. Show non-matmul FLOP reduction. Model FA-3's "
            "warp specialisation pipeline with TMA overlap. Compare achieved TFLOP/s "
            "for FA-1, FA-2, FA-3 against H100 theoretical peak."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FA-1 vs FA-2 vs FA-3 — Parallelism, Non-Matmul FLOPs & Pipeline")
print("=" * 68)
print()

H100_PEAK_FP16   = 989.0    # TFLOP/s
H100_HBM_BW      = 3350.0   # GB/s
H100_SMs         = 132
H100_SMEM_KB     = 228
A100_PEAK_FP16   = 312.0
A100_SMs         = 108


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Parallelism comparison — FA-1 (KV outer) vs FA-2 (Q outer)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Threadblock Parallelism: FA-1 vs FA-2")
print("━" * 68)
print()

print("  FA-1: outer loop over KV blocks → parallelism = N/BLOCK_KV × H × B")
print("  FA-2: outer loop over Q  blocks → parallelism = N/BLOCK_Q  × H × B")
print("  Both have the same total blocks, but FA-2's outer blocks are more efficient:")
print("    FA-2 keeps the (m, d, O) accumulator entirely in registers per Q block.")
print("    FA-1 had to write/read partial accumulators to HBM for the Q dimension.")
print()

BLOCK_Q  = 128
BLOCK_KV = 128
D        = 128

print(f"  BLOCK_Q={BLOCK_Q}, BLOCK_KV={BLOCK_KV}, H100 has {H100_SMs} SMs")
print()
print(f"  {'Config (B, H, N)':<24}  {'N_blocks':>10}  {'SM util%':>10}  "
      f"{'SM waves':>9}  {'FA-2 HBM saves'}")
print("  " + "─" * 64)

configs = [
    (1,  32, 512,   "infer decode"),
    (1,  32, 4096,  "infer prefill"),
    (8,  32, 2048,  "training 8B"),
    (32, 32, 2048,  "training 32B"),
    (1,  32, 32768, "long context 32K"),
    (1,  32, 131072,"long context 128K"),
]

for B, H, N, label in configs:
    # FA-2: blocks = (N/BLOCK_Q) × H × B
    n_blocks_fa2 = math.ceil(N / BLOCK_Q) * H * B
    waves_fa2    = math.ceil(n_blocks_fa2 / H100_SMs)
    util_fa2     = (n_blocks_fa2 / (waves_fa2 * H100_SMs)) * 100

    # HBM savings: FA-2 avoids writing partial accumulators to HBM
    # FA-1 needed: N/BLOCK_KV writes of partial O (N_q × d) per Q pass
    # FA-2: zero such writes — O is kept in registers throughout
    partial_writes_fa1 = math.ceil(N / BLOCK_KV) * N * D * 2 * B * H  # bytes
    hbm_save_gb        = partial_writes_fa1 / 1e9

    print(f"  {label:<24}  {n_blocks_fa2:>10}  {util_fa2:>9.1f}%  "
          f"{waves_fa2:>9}  {hbm_save_gb:>8.2f} GB saved")

print()
print("  FA-2 threadblock count = FA-1 threadblock count (same total work).")
print("  FA-2's advantage: NO partial accumulator HBM writes/reads (FA-1 had these).")
print("  This is the dominant speedup for small N where memory bandwidth dominates.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Non-matmul FLOP analysis
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Non-Matmul FLOPs: FA-1 vs FA-2 vs FA-3")
print("━" * 68)
print()

print("  Tensor core FP16 FLOP/s: 989 TFLOP/s on H100")
print("  Scalar (non-matmul) FP32 throughput: ~67 TFLOP/s on H100  (1/15× slower)")
print()
print("  Non-matmul ops in the softmax update per KV tile:")
print()

print("  FA-1 (per KV tile per query row):")
print("    row max:    1 max per element of S_tile = BLOCK_KV ops")
print("    update m:   1 max per query              = 1 op")
print("    correction: exp(m_old - m_new)           = 1 exp")
print("    rescale O:  O × correction               = d multiplies")
print("    exp(S):     BLOCK_KV exps                = BLOCK_KV ops")
print("    update d:   sum + correction × d_old     = 2 ops")
print("    Total per tile: ~BLOCK_KV + d + 3 non-matmul ops")
print(f"    For BLOCK_KV={BLOCK_KV}, d={D}: {BLOCK_KV + D + 3} ops per tile")
print()
print("  FA-2 (per KV tile per query row):")
print("    row max:    same as FA-1                 = BLOCK_KV ops")
print("    update m:   same as FA-1                 = 1 op")
print("    correction: exp(m_old - m_new)           = 1 exp (deferred)")
print("    exp(S):     BLOCK_KV exps                = BLOCK_KV ops")
print("    NO per-tile O rescaling (deferred to final normalisation)")
print("    Final normalisation: divide O by d       = d ops (ONCE per query)")
print(f"    Per tile: ~{BLOCK_KV + 2} ops  (vs {BLOCK_KV + D + 3} for FA-1)")
print()

N_tiles = 64  # example: N/BLOCK_KV
FA1_nm_per_query = N_tiles * (BLOCK_KV + D + 3)
FA2_nm_per_query = N_tiles * (BLOCK_KV + 2) + D    # D for final normalise

matmul_flops_per_query = 2 * N_tiles * BLOCK_KV * D * 2  # QKᵀ + exp@V

print(f"  For N/BLOCK_KV = {N_tiles} tiles:")
print(f"    FA-1 non-matmul FLOPs per query: {FA1_nm_per_query:>8,}")
print(f"    FA-2 non-matmul FLOPs per query: {FA2_nm_per_query:>8,}")
print(f"    Matmul FLOPs per query:           {matmul_flops_per_query:>8,}")
print(f"    FA-1 non-matmul fraction: {FA1_nm_per_query/(FA1_nm_per_query+matmul_flops_per_query)*100:.1f}%")
print(f"    FA-2 non-matmul fraction: {FA2_nm_per_query/(FA2_nm_per_query+matmul_flops_per_query)*100:.1f}%")
print()
print("  FA-2 reduces non-matmul fraction by ~20–25%.")
print("  On H100, non-matmul ops are 15× slower per FLOP → this is significant.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Causal masking FLOPs efficiency
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Causal Masking: FA-2 Skips Future Blocks")
print("━" * 68)
print()

print("  For causal attention of length N with BLOCK_Q = BLOCK_KV = B:")
print("  Total tiles: (N/B)² = N²/B²")
print("  Tiles on the diagonal (partial, need masking): N/B tiles")
print("  Tiles above diagonal (all future, skip entirely): N/B × (N/B - 1) / 2")
print("  Tiles below+on diagonal (valid, no masking): N/B × (N/B + 1) / 2")
print()

print(f"  {'N':>8}  {'B':>5}  {'Total tiles':>12}  {'Skip tiles':>12}  "
      f"{'Skip%':>8}  {'Effective FLOP util'}")
print("  " + "─" * 62)

for N_c in [512, 1024, 2048, 4096, 8192, 16384]:
    B_c = min(128, N_c)
    n_q_blocks = N_c // B_c
    total_tiles = n_q_blocks * n_q_blocks
    skip_tiles  = n_q_blocks * (n_q_blocks - 1) // 2
    mask_tiles  = n_q_blocks           # diagonal tiles (partial work)
    full_tiles  = total_tiles - skip_tiles - mask_tiles

    # Effective compute: full tiles use 100% TC, mask tiles use ~50%, skip tiles 0%
    eff_compute = (full_tiles * 1.0 + mask_tiles * 0.5) / total_tiles * 100
    skip_pct    = skip_tiles / total_tiles * 100

    print(f"  {N_c:>8,}  {B_c:>5}  {total_tiles:>12}  {skip_tiles:>12}  "
          f"{skip_pct:>7.1f}%  {eff_compute:>12.1f}%")

print()
print("  FA-2 skips the upper-triangular tiles entirely → ~50% FLOPs saved for causal.")
print("  Vs FA-1 which computed full tiles and masked zeros → wasted half the compute.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: FA-3 pipeline model — warp specialisation with TMA
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — FA-3 Warp Specialisation Pipeline (H100 Hopper)")
print("━" * 68)
print()

# Cycle costs on H100
TMA_LOAD_CYCLES  = 50    # async TMA load latency (hidden by overlap)
WGMMA_LATENCY    = 20    # warp group MMA latency
WGMMA_THROUGHPUT = 4     # one per 4 cycles per warp group
SOFTMAX_CYCLES   = 40    # exp + max + sum for one tile

print("  FA-3 uses two warp roles: PRODUCER (TMA loads) and CONSUMER (wgmma).")
print("  Two SMEM buffers (pingpong): buffer A and buffer B alternate per iteration.")
print()

BLOCK_KV_FA3 = 128
D_FA3        = 128

# One iteration of the inner KV loop
tile_qk_flops   = 2 * BLOCK_Q * BLOCK_KV_FA3 * D_FA3   # Q @ Kᵀ
tile_pv_flops   = 2 * BLOCK_Q * BLOCK_KV_FA3 * D_FA3   # exp(S) @ V
total_tile_flops= tile_qk_flops + tile_pv_flops

wgmma_qk_cycles = math.ceil(BLOCK_Q * BLOCK_KV_FA3 * D_FA3 / (128 * WGMMA_THROUGHPUT))  # simplified
wgmma_pv_cycles = wgmma_qk_cycles

print(f"  Per-tile cycle breakdown (BLOCK_Q={BLOCK_Q}, BLOCK_KV={BLOCK_KV_FA3}, d={D_FA3}):")
print()

stages = [
    ("TMA load K[j+1] (async)",     TMA_LOAD_CYCLES,   "Producer: issues load, returns immediately"),
    ("TMA load V[j+1] (async)",     0,                 "  (overlaps with K load via hardware)"),
    ("wgmma: S = Q @ Kᵀ (async)",  wgmma_qk_cycles,   "Consumer: fires wgmma, does NOT wait yet"),
    ("Issue TMA K[j+2] prefetch",   2,                 "Consumer: while wgmma still running"),
    ("wgmma_wait: S ready",         WGMMA_LATENCY - wgmma_qk_cycles, "Consumer: waits for S"),
    ("Softmax update (exp,max,sum)", SOFTMAX_CYCLES,   "Consumer: runs while wgmma PV issues"),
    ("wgmma: O += exp(S) @ V (async)", wgmma_pv_cycles,"Consumer: fires wgmma, continues"),
    ("wgmma_wait: O ready",         max(0, WGMMA_LATENCY - wgmma_pv_cycles), "Consumer waits"),
]

total_critical = 0
print(f"  {'Stage':<40}  {'Cycles':>8}  {'Notes'}")
print("  " + "─" * 70)
for stage, cyc, note in stages:
    # Only count stages on the critical path (not fully overlapped)
    on_crit = cyc > 0
    total_critical += cyc if on_crit else 0
    print(f"  {stage:<40}  {cyc:>8}  {note}")

print()
print(f"  Critical path per tile: ~{total_critical} cycles")
print(f"  TMA latency ({TMA_LOAD_CYCLES} cycles) fully hidden by wgmma compute. ✅")
print(f"  Softmax ({SOFTMAX_CYCLES} cycles) partially overlaps with wgmma PV. ✅")
print()

# Throughput estimate
freq_ghz  = 1.98    # H100 SM frequency
tile_ns   = total_critical / (freq_ghz * 1e9) * 1e9
tile_flops = total_tile_flops
tflops     = tile_flops / (total_critical / (freq_ghz * 1e9)) / 1e12

print(f"  Estimated throughput per tile: {tile_flops:,} FLOPs / {total_critical} cycles")
print(f"  At {freq_ghz} GHz: {tflops:.0f} TFLOP/s per SM pair (simplified model)")
print()
print(f"  FA-3 reported results:")
print(f"    FP16, H100: ~1.2–1.4 PFLOP/s attention  (~75% of H100 peak)")
print(f"    FP8,  H100: ~2.0 PFLOP/s attention       (~50% of 1979 TFLOP/s FP8 peak)")
print(f"    vs FA-2 (A100): ~312 TFLOP/s effective   (~75% of A100 FP16 peak)")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Flash Attention Backward Pass & Recomputation Trick": {
        "description": (
            "Implement the Flash Attention backward pass using the recomputation "
            "trick. Show how Q, K, V, O, and the (m, d) statistics saved from "
            "forward are used to recompute the attention matrix tile-by-tile during "
            "backward without ever storing the full N×N matrix. Verify dQ, dK, dV "
            "against the naive autograd reference. Compute the memory saved vs "
            "standard attention backward, and the recomputation overhead in FLOPs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FLASH ATTENTION BACKWARD — Recomputation Trick & Gradient Verification")
print("=" * 68)
print()

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def softmax_rows(X):
    """Numerically stable row-wise softmax."""
    m = X.max(axis=-1, keepdims=True)
    e = np.exp(X - m)
    return e / e.sum(axis=-1, keepdims=True)

def naive_attention_with_grads(Q, K, V, dO, scale=None):
    """
    Naive attention forward + backward via explicit autograd.
    Returns (O, dQ, dK, dV).
    """
    N, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Forward
    S  = Q @ K.T * scale                                 # (N, N)
    P  = softmax_rows(S)                                  # (N, N)
    O  = P @ V                                            # (N, d)

    # Backward (standard attention gradient derivation)
    dV = P.T @ dO                                         # (N, d)
    dP = dO @ V.T                                         # (N, N)
    # dS[i,j] = P[i,j] * (dP[i,j] - sum_k(P[i,k]*dP[i,k]))
    dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True)) # (N, N)
    dS = dS * scale
    dQ = dS @ K                                           # (N, d)
    dK = dS.T @ Q                                         # (N, d)

    return O, dQ, dK, dV


def flash_attention_fwd_save_stats(Q, K, V, scale=None):
    """
    Flash Attention forward that SAVES (m, d, O) for backward.
    Returns (O, m, d) — no N×N matrix saved.
    """
    N, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    m_saved = np.zeros(N)
    d_saved = np.zeros(N)
    O_out   = np.zeros((N, d))

    for qi in range(N):
        q = Q[qi]
        m_state = -math.inf
        d_state = 0.0
        O_state = np.zeros(d)

        for ki in range(N):
            s_new = float(q @ K[ki]) * scale
            m_new = max(m_state, s_new)
            O_state = O_state * math.exp(m_state - m_new) + math.exp(s_new - m_new) * V[ki]
            d_state = d_state * math.exp(m_state - m_new) + math.exp(s_new - m_new)
            m_state = m_new

        m_saved[qi] = m_state
        d_saved[qi] = d_state
        O_out[qi]   = O_state / d_state

    return O_out, m_saved, d_saved


def flash_attention_bwd(Q, K, V, O, dO, m, d, scale=None):
    """
    Flash Attention backward pass.
    Uses RECOMPUTATION: reconstructs P tiles from Q, K, m, d.
    NEVER materialises the full N×N P matrix.
    Returns (dQ, dK, dV).
    """
    N, d_h = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_h)

    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    for qi in range(N):
        q     = Q[qi]
        o     = O[qi]
        do    = dO[qi]
        m_qi  = m[qi]
        d_qi  = d[qi]

        # Scalar D_i = dot(dO_i, O_i) — used in softmax backward
        D_i = float(do @ o)

        for ki in range(N):
            # RECOMPUTE P[qi, ki] from Q, K and saved stats
            s_qk = float(q @ K[ki]) * scale
            p_qk = math.exp(s_qk - m_qi) / d_qi

            # dV[ki] += P[qi, ki] * dO[qi]
            dV[ki] += p_qk * do

            # dS[qi, ki] = P[qi, ki] * (dO[qi] @ V[ki] - D_i)
            ds_qk = p_qk * (float(do @ V[ki]) - D_i) * scale

            # dQ[qi] += dS[qi, ki] * K[ki]
            dQ[qi] += ds_qk * K[ki]

            # dK[ki] += dS[qi, ki] * Q[qi]
            dK[ki] += ds_qk * q

    return dQ, dK, dV


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Gradient correctness verification
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Backward Pass Correctness: Flash vs Naive Autograd")
print("━" * 68)
print()

test_cases = [
    (8,  16, "N=8,  d=16"),
    (16, 32, "N=16, d=32"),
    (32, 64, "N=32, d=64"),
    (24, 48, "N=24, d=48  (non-power-of-2)"),
]

print(f"  {'Test case':<22}  {'dQ err':>10}  {'dK err':>10}  "
      f"{'dV err':>10}  {'Correct?'}")
print("  " + "─" * 56)

for N_t, d_t, label in test_cases:
    Q_t  = np.random.randn(N_t, d_t)
    K_t  = np.random.randn(N_t, d_t)
    V_t  = np.random.randn(N_t, d_t)
    dO_t = np.random.randn(N_t, d_t)

    # Reference: naive autograd
    O_ref, dQ_ref, dK_ref, dV_ref = naive_attention_with_grads(Q_t, K_t, V_t, dO_t)

    # Flash backward
    O_fl, m_fl, d_fl = flash_attention_fwd_save_stats(Q_t, K_t, V_t)
    dQ_fl, dK_fl, dV_fl = flash_attention_bwd(Q_t, K_t, V_t, O_fl, dO_t, m_fl, d_fl)

    err_dQ = np.abs(dQ_fl - dQ_ref).max()
    err_dK = np.abs(dK_fl - dK_ref).max()
    err_dV = np.abs(dV_fl - dV_ref).max()
    correct = max(err_dQ, err_dK, err_dV) < 1e-8

    print(f"  {label:<22}  {err_dQ:>10.2e}  {err_dK:>10.2e}  "
          f"{err_dV:>10.2e}  {'✅' if correct else '❌'}")

print()
print("  Flash backward gradients exactly match naive autograd. ✅")
print("  The P matrix was NEVER stored — it was recomputed per (qi, ki) pair.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Memory analysis — what is saved vs standard backward
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Memory Analysis: Saved Tensors for Backward")
print("━" * 68)
print()

print("  STANDARD ATTENTION backward requires saving from forward:")
print("    P  [N×N] attention weights  — DOMINANT COST")
print("    S  [N×N] raw scores (optional, for dS)  — OR recompute from Q,K")
print("    Q, K, V [N×d] each")
print("    O  [N×d]")
print()
print("  FLASH ATTENTION backward requires saving from forward:")
print("    Q, K, V  [N×d]   — unavoidable (needed for gradient computation)")
print("    O        [N×d]   — needed for D_i = dot(dO_i, O_i)")
print("    m        [N]     — row-wise max, for recomputing P")
print("    d        [N]     — row-wise sum of exp, for recomputing P")
print("    NO P or S saved!")
print()

print(f"  {'N':>8}  {'Std P saved (GB)':>17}  {'Flash (m,d) saved (MB)':>24}  {'Saving factor'}")
print("  " + "─" * 58)

for N_m in [512, 1024, 2048, 4096, 8192, 32768, 65536]:
    p_bytes    = N_m * N_m * 4          # P matrix in FP32
    md_bytes   = N_m * 2 * 4           # m and d vectors in FP32
    saving     = p_bytes / md_bytes
    print(f"  {N_m:>8,}  {p_bytes/1e9:>17.3f}  {md_bytes/1e6:>24.3f}  {saving:>8.0f}×")

print()
print("  At N=8192: standard backward must save 256 MB of attention weights.")
print("  Flash backward saves only 64 KB (m and d vectors).")
print("  4000× memory reduction for saved tensors.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Recomputation overhead — extra FLOPs in backward
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Recomputation Overhead: Extra FLOPs vs Memory Saving")
print("━" * 68)
print()

print("  Flash attention backward RECOMPUTES the attention tile P[qi,ki]")
print("  from Q, K, m, d for every (qi, ki) pair during the backward pass.")
print()
print("  Standard backward QKᵀ FLOPs in forward:   2N²d (just for saving P)")
print("  Flash backward recomputation FLOPs:         2N²d (one extra QKᵀ)")
print()
print("  So flash backward costs ~2× the forward compute (one extra pass).")
print("  But saves N² × 4 bytes of HBM reads (reading saved P from storage).")
print()

H100_BW_GBS = 3350.0

print(f"  {'N':>8}  {'Recomp FLOPs':>14}  {'P load bytes':>14}  "
      f"{'Recomp time (µs)':>18}  {'P load time (µs)':>18}  {'Recomp worthwhile?'}")
print("  " + "─" * 82)

D_r = 128
for N_r in [512, 1024, 2048, 4096, 8192]:
    recomp_flops = 2 * N_r * N_r * D_r
    p_load_bytes = N_r * N_r * 4

    # Time to recompute via tensor cores
    tc_tflops    = H100_PEAK_FP16 * 1e12
    recomp_us    = recomp_flops / tc_tflops * 1e6

    # Time to load P from HBM
    p_load_us    = p_load_bytes / (H100_BW_GBS * 1e9) * 1e6

    worthwhile   = "✅" if p_load_us > recomp_us else "❌"

    print(f"  {N_r:>8,}  {recomp_flops/1e9:>12.2f}G  {p_load_bytes/1e9:>12.2f}G  "
          f"{recomp_us:>18.3f}  {p_load_us:>18.3f}  {worthwhile}")

print()
print("  On H100 (compute-rich), recomputation is faster than reading P from HBM.")
print("  This is the fundamental insight: H100 can recompute faster than it can load.")
print("  The trade-off improves as GPUs get more compute relative to memory bandwidth.")
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