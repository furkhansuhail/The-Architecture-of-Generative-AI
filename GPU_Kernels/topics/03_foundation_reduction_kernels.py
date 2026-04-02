"""
Reduction Kernels — Tree Reduction, Warp Reduction & Softmax
=============================================================

A reduction collapses N values into one: sum, max, min, product,
logical-AND. It sounds trivial — but implementing it correctly and
at peak bandwidth on a GPU requires coordinating thousands of threads
across a three-level hierarchy: registers → shared memory → global
memory. Doing it wrong wastes 95% of available bandwidth. Doing it
right makes the kernel memory-bound at the theoretical HBM ceiling.

Reductions are not a niche operation. Every normalisation layer
(LayerNorm, RMSNorm, BatchNorm), every attention mechanism (softmax),
every loss function (cross-entropy), every metric (accuracy, BLEU)
reduces a tensor at some point. The softmax kernel alone runs
millions of times per second inside any production LLM serving stack.

This module builds the full reduction stack from first principles:
    1. Naïve reduction   — understand what goes wrong before fixing it
    2. Tree reduction    — the canonical SMEM algorithm and its 6 optimisations
    3. Warp reduction    — eliminate SMEM entirely for the final stages
    4. Block reduction   — combine warp reduction with SMEM for large N
    5. Safe softmax      — numerical stability, the max-shift trick
    6. Online softmax    — one-pass algorithm for FlashAttention tiling

Each level builds directly on the one before. By the end, the pattern
behind every production softmax kernel — from PyTorch to FA-3 — is clear.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Reduction Kernels — Tree Reduction, Warp Reduction & Softmax"
DISPLAY_NAME = "03 · Reduction Kernels"
ICON         = "🌲"
SUBTITLE     = "Tree Reduction → Warp Shuffle → Safe Softmax → Online Softmax"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT A REDUCTION IS AND WHY IT IS HARD ON A GPU

### The Sequential Baseline

    A sum reduction over N elements is O(N) sequentially:
        acc = 0
        for x in A: acc += x

    That's N-1 additions in a single thread. Simple, but leaves the GPU idle:
    an A100 has 6912 CUDA cores that want to execute simultaneously.

### The Parallelism Problem

    The reduction has an inherent DATA DEPENDENCY CHAIN:
        Step k must wait for step k-1 to finish because it reads its output.

        A[0] + A[1] → t0
        t0   + A[2] → t1    ← must wait for t0
        t1   + A[3] → t2    ← must wait for t1
        ...

    This looks like it cannot be parallelised. But it CAN — via a TREE.
    A binary tree of additions has depth log2(N) instead of N:

        Depth 0 (N/2 parallel adds):
            A[0]+A[1]   A[2]+A[3]   A[4]+A[5]   A[6]+A[7]
        Depth 1 (N/4 parallel adds):
            (A[0..1])+(A[2..3])     (A[4..5])+(A[6..7])
        Depth 2 (N/8 parallel adds):
            (A[0..3]) + (A[4..7])
        ...
        Depth log2(N): final sum.

    For N=1M: sequential = 1M steps; tree = 20 parallel steps.
    Parallelism: N/2 additions at depth 0, N/4 at depth 1, etc.
    Total work = N-1 additions — same as sequential. Just structured differently.

### The Memory Bottleneck Framing

    Arithmetic intensity of a sum reduction:
        FLOPs:  N-1 additions ≈ N FLOPs
        Bytes:  read N elements = N × 4 bytes (FP32)
        AI:     N / (4N) = 0.25 FLOPs/Byte

    A100 ridge point: ~156 FLOPs/Byte.
    Reduction AI: 0.25 — solidly 624× BELOW the ridge.

    This means: reduction is ALWAYS memory-bandwidth bound.
    The CPU or GPU's compute units sit idle waiting for memory.
    Optimisation goal: achieve as close to peak HBM bandwidth as possible.

    A100 peak bandwidth: 2.0 TB/s.
    Optimal time for 1B FP32 reduction: 4 GB / 2 TB/s = 2 ms.
    Naïve kernel: 20–100 ms. Optimised kernel: 2–3 ms.


##### PART 2 — NAÏVE REDUCTION AND ITS FAILURES

### Kernel 0: One Thread Per Element (Naïve)

    __global__ void reduce0(float* A, float* out, int N) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid == 0) {
            float acc = 0;
            for (int i = 0; i < N; i++) acc += A[i];
            *out = acc;
        }
    }

    Problems:
        - 1 thread out of 6912 does all the work → 0.01% utilisation
        - Sequential, so same speed as CPU
        - Memory reads are non-coalesced (thread 0 does all reads alone)

### Kernel 1: Interleaved Addressing (Common Wrong Approach)

    __global__ void reduce1(float* smem, int N) {
        int tid = threadIdx.x;
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0)          // ← PROBLEM HERE
                smem[tid] += smem[tid + stride];
            __syncthreads();
        }
    }

    Problem 1: THREAD DIVERGENCE
        At stride=1: half the threads (odd tid) are idle.
        At stride=2: 3/4 of threads are idle.
        All threads in a warp must execute the branch test even if inactive.

    Problem 2: SMEM BANK CONFLICTS
        tid=0 accesses smem[0] and smem[1]   → banks 0 and 1
        tid=2 accesses smem[2] and smem[3]   → banks 2 and 3
        (fine at stride=1)
        At stride=8: smem[0] and smem[8], smem[16] and smem[24]
        → multiple threads hitting the same bank → conflicts degrade throughput.

    Effective throughput: ~30% of peak. Better than kernel 0, still bad.

### Kernel 2: Sequential Addressing (Fixes Bank Conflicts)

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    Thread 0 reads smem[0] + smem[blockDim.x/2]
    Thread 1 reads smem[1] + smem[blockDim.x/2 + 1]
    → Consecutive threads access consecutive SMEM slots → no bank conflicts ✅
    → Still has thread divergence (half threads idle per step)

    Effective throughput: ~50–60% of peak. Much better, still room to improve.

### Kernel 3: First Add During Load (Halves Global Memory Traffic)

    INSIGHT: when loading data from global memory into SMEM, each thread
    can load TWO elements and add them immediately, halving the number
    of blocks needed by 2× and keeping all threads active for the first step.

    smem[tid] = A[tid] + A[tid + blockDim.x];   // load+add in one step

    Now we only need N/2 threads for a block instead of N.
    All threads are active at the first SMEM step.
    This is the "halve the blocks" trick — universally used in production.

    Effective throughput: ~70% of peak.

### Kernel 4: Unroll the Last Warp (Eliminate __syncthreads Overhead)

    When stride ≤ 32, only ONE warp is active. Since a warp is always
    synchronised by hardware (SIMT lockstep), __syncthreads() is
    unnecessary — but the compiler still emits it, wasting cycles.

    Solution: unroll the last warp manually with volatile pointer:

    if (tid < 32) {
        volatile float* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    This removes 5 __syncthreads() calls from the most frequently executed
    code path. On large reductions with many blocks, these add up significantly.

### Kernel 5: Complete Unroll + Template Block Size

    By templatising on block size, the compiler can unroll ALL loop iterations
    at compile time — no branch overhead, no loop counter overhead.

    Combined with warp shuffle for the final stage (Part 3), this is the
    production-quality pattern used in PyTorch's reduction kernels.


##### PART 3 — TREE REDUCTION: THE CANONICAL SMEM ALGORITHM

### The Full Algorithm (Kernel 3 + Kernel 4 Combined)

    Setup:
        - N elements, block_size = 256 threads, grid = ceil(N / (2 * block_size)) blocks
        - Each block reduces a subarray of 2 × block_size elements to 1 partial sum
        - A second pass (or atomicAdd) accumulates partial sums from all blocks

    Phase 1 — Global Load with First Add:
        int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        smem[tid] = A[i] + A[i + blockDim.x];
        __syncthreads();

    Phase 2 — Tree Reduction in SMEM:
        for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }

    Phase 3 — Final Warp (No Sync Needed):
        if (tid < 32) warp_reduce_in_smem(smem, tid);

    Phase 4 — Write Partial Sum:
        if (tid == 0) partial[blockIdx.x] = smem[0];

### Why __syncthreads() Between Each SMEM Step

    Each step of the tree reads values written by the PREVIOUS step.
    Without the barrier, thread A might read a value that thread B hasn't
    updated yet. This is a data race — silent, hard to debug.

        smem[0] += smem[128]   ← writes smem[0]
        smem[1] += smem[129]   ← writes smem[1]
        // WITHOUT __syncthreads here, the next step might read old smem[0]
        smem[0] += smem[64]    ← reads smem[0] — must see updated value!

    __syncthreads() is not optional. It is the correctness guarantee.
    Cost: 10–20 cycles per barrier. For log2(256) = 8 steps: ~80–160 cycles.
    This is why the "unroll last warp" trick (Part 2, Kernel 4) matters.

### The Two-Pass Structure for Large N

    One block can reduce at most 2 × block_size elements.
    For N = 1B elements with block_size = 256:
        grid_size = 1B / 512 ≈ 1.95M blocks → 1.95M partial sums
    Second pass: reduce 1.95M partial sums → another ~3800 blocks → ...

    In practice, three patterns:
        1. TWO KERNELS: launch kernel 1 (produces partials) + kernel 2 (final)
        2. ATOMIC ACCUMULATION: each block does atomicAdd(&global_sum, smem[0])
           Simple but atomic contention on one address serialises the final step.
        3. COOPERATIVE GROUPS (CUDA 9+): all blocks in the grid synchronise
           via cooperative_groups::grid_group::sync(), reducing in one pass.

    For ML workloads, the two-kernel approach is most common because it
    allows the second kernel to be fused with a subsequent operation.


##### PART 4 — WARP-LEVEL REDUCTION: ELIMINATING SMEM ENTIRELY

### The Hybrid Architecture

    The optimal block-level reduction uses a TWO-STAGE HYBRID:

        Stage 1: Warp-level reduction  (via __shfl_down_sync, no SMEM)
        Stage 2: SMEM accumulation     (one value per warp → reduce across warps)
        Stage 3: Final warp reduction  (via __shfl_down_sync on the warp partials)

    For a 256-thread block (8 warps):
        Stage 1:  8 independent warp reductions of 32 threads
                  Result: 8 partial sums (one per warp), held in registers
                  Cost:   5 __shfl_down per warp × 8 warps = 40 shuffles (parallel!)

        Stage 2:  Each warp's lane 0 writes its partial to smem[warp_id]
                  __syncthreads() — ensure all 8 partials are visible
                  Cost:   1 SMEM write + 1 barrier

        Stage 3:  Load smem[0..7] into registers of first warp
                  One more warp reduction on 8 values
                  Cost:   3 __shfl_down (since 8 values → log2(8) = 3 steps)

        Total SMEM reads: 8 (just the 8 partial sums)
        Compare to pure SMEM tree: 512 SMEM reads (for 256-thread block)
        → ~64× fewer SMEM accesses

### Why This Matters: SMEM vs Register Speed

    Register: operand already in the register file → 0 extra cycles
    SMEM:     6 ns access, potential bank conflicts, barrier overhead

    A warp reduction via __shfl_down touches ZERO shared memory.
    All communication happens through the hardware shuffle network
    between register files.

    Benchmark (A100, 256-thread block, float32):
        Pure SMEM tree reduction:     ~25 cycles per warp
        Warp shuffle reduction:       ~5 cycles per warp
        Hybrid (shuffle + minimal SMEM): ~8 cycles total

### The Full Hybrid Pattern in Code

    // STAGE 1: intra-warp reduction (each warp independently)
    float val = load_and_add(A, tid);  // first add during load
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    // STAGE 2: write warp partial to SMEM
    __shared__ float warp_sums[32];   // max 32 warps per block
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // STAGE 3: final reduction across warp partials (first warp only)
    int n_warps = blockDim.x / 32;
    val = (tid < n_warps) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = n_warps/2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (tid == 0) partial[blockIdx.x] = val;

### Reduction for Non-Power-of-Two N

    Real tensors rarely have sizes that are powers of two (e.g., vocab_size=32000,
    hidden_dim=4096, seq_len=2048+1 for special tokens).

    Three strategies:
        1. PADDING: round N up to next power of two, pad with identity element.
           Cost: reads extra elements (wasteful for large pad regions).

        2. GRID-STRIDE LOOP: each thread reduces multiple elements before SMEM:
               float acc = 0;
               for (int i = tid; i < N; i += blockDim.x)
                   acc += A[i];
               smem[tid] = acc;
           Works for any N. Thread count can be fixed (e.g., 256).
           Flexible and production-standard (used in PyTorch, cuDNN).

        3. COOPERATIVE GROUPS: let CUDA figure out non-power-of-two splits.
           Most elegant; slightly more overhead than manual grid-stride.

    The GRID-STRIDE LOOP is the standard production pattern.
    It handles arbitrary N, ensures coalesced memory access, and feeds
    directly into the SMEM tree or hybrid reduction.


##### PART 5 — SAFE SOFTMAX: NUMERICAL STABILITY AND THE MAX-SHIFT TRICK

### The Overflow Problem with Naïve Softmax

    Naïve softmax: softmax(x)_i = exp(x_i) / sum(exp(x_j))

    Problem: exp(x_i) overflows to +inf for x_i > 88 (FP32) or > 709 (FP64).
    In attention, logits easily reach 50–500 before masking.

        x = [100, 200, 300]
        exp(x) = [2.69e43, 7.22e86, 1.94e130]  ← FP32 overflows at ~3.4e38!
        exp(300) → +inf in FP32.

    Consequence: softmax returns NaN for large logits. Gradients vanish.
    Training diverges. Silent, catastrophic.

### The Max-Shift Trick (Numerically Stable Softmax)

    Observation: softmax is INVARIANT to adding a constant to all logits.

        softmax(x - c)_i = exp(x_i - c) / sum(exp(x_j - c))
                         = exp(x_i) * exp(-c) / (sum(exp(x_j)) * exp(-c))
                         = exp(x_i) / sum(exp(x_j))
                         = softmax(x)_i

    Choose c = max(x):
        exp(x_i - max(x)) ∈ (0, 1] for all i  ← no overflow ✅
        The largest value exp(max(x) - max(x)) = exp(0) = 1.0

    Stable softmax implementation:
        m = max(x)                          // reduce: 1 pass over x
        e = exp(x - m)                      // elementwise
        s = sum(e)                          // reduce: 1 pass over e (or x again)
        return e / s                        // elementwise

    GPU implementation cost:
        3 passes over x: load x (max), load x (exp), load e (sum) → 3 × N reads
        SMEM-fused variant: 2 passes (see online softmax)

### Underflow: The Other Numerical Issue

    Underflow: exp(x_i - max(x)) → 0 for x_i ≪ max(x).
    In FP32: x_i - max(x) < -87.3 → exp(...) rounds to zero.
    This is usually acceptable (near-zero probability → treated as zero).

    For attention with masked tokens:
        Masked logits set to -1e9 (or -inf).
        After max-shift: x_masked - max = -1e9 - max → exp → 0.0 ✅
        BUT: if ALL logits are masked, max = -1e9, denominators = 0 → NaN.
        Fix: check for this edge case, return uniform distribution.

### Log-Softmax and Cross-Entropy

    log(softmax(x)_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))

    This is more numerically stable than computing softmax then taking log,
    because log(exp(z)) = z avoids the exp→log round-trip precision loss.

    Cross-entropy loss:
        L = -sum(y_i * log(softmax(x)_i))
          = -sum(y_i * (x_i - max(x))) + log(sum(exp(x_j - max(x))))

    In practice: fused cross-entropy kernel combines softmax + log + NLL loss
    in one pass, avoiding materialising the full softmax distribution.


##### PART 6 — ONLINE SOFTMAX: ONE-PASS ALGORITHM

### Why Three Passes is Expensive

    Standard safe softmax requires:
        Pass 1: max(x)         → 1 full scan of N elements
        Pass 2: sum(exp(x-m))  → 1 full scan of N elements
        Pass 3: divide         → 1 full scan of N elements
        Total: 3N reads + 1N write = 4N × 4 bytes HBM traffic

    For a sequence of 8192 tokens with hidden_dim=4096 (attention logits):
        N = 8192, 4 passes = 4 × 8192 × 4 = 131 KB per row, per layer, per head.
        At 2 TB/s: 65 μs per attention row — multiplied by all layers/heads.

    Can we do it in fewer passes?

### Online Softmax: Merge Max and Sum into One Pass

    Key insight (Milakov & Gimelshein, 2018):
    The max and sum can be computed SIMULTANEOUSLY using a running correction factor.

    Algorithm:
        m_current = -inf
        d_current = 0.0

        for each element x_i:
            m_new = max(m_current, x_i)
            d_current = d_current * exp(m_current - m_new) + exp(x_i - m_new)
            m_current = m_new

        // After one pass: m_current = max(x), d_current = sum(exp(x - max(x)))

    The correction factor exp(m_current - m_new) rescales the accumulated sum
    when the running max increases. When m_new > m_current, older terms
    were computed with the wrong max and must be rescaled.

    Why this works:
        At step i, d_current = sum_{j≤i} exp(x_j - m_i)
        When m_{i+1} > m_i, each previous exp term needs ×exp(m_i - m_{i+1})
        This is exactly the exp(m_current - m_new) correction.

    Cost: 1 pass over x (read once) + 1 pass for normalisation = 2N reads.
    Saving: 33% fewer HBM reads vs 3-pass algorithm.

### Parallel Online Softmax (Two-Pass at Block Level)

    For GPU execution with multiple threads, the online algorithm becomes:

        Each thread computes partial (m_i, d_i) over its chunk of x.
        Parallel merge of (m, d) pairs:
            merge((m_a, d_a), (m_b, d_b)):
                m = max(m_a, m_b)
                d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
                return (m, d)

        The merge is ASSOCIATIVE — it can be used in a tree or warp reduction!
        → Apply exactly the tree/warp reduction patterns from Parts 2–4,
          but with a 2-tuple (m, d) instead of a scalar.

    This gives: 1 pass to load x and compute per-thread (m, d),
                1 warp/tree reduce to get global (m, d),
                1 pass to normalise.
    Total: 2 HBM passes instead of 3.

### The FlashAttention Connection

    FlashAttention (Dao et al., 2022) extends online softmax to handle
    the full attention computation: softmax(QK^T/√d) × V.

    Instead of just tracking (m, d), it tracks (m, d, o):
        m = running max of attention logits
        d = running sum of exp(logit - m)
        o = running weighted output sum (∑ exp(logit - m) × V)

    Tile-by-tile update:
        Process one tile of K and V at a time, staying in SMEM.
        At each tile, update (m, d, o) with the new logits.
        Final output: o / d (normalise by denominator).

    Result: Q, K, V each read ONCE from HBM.
    Standard attention: Q, K, V read multiple times (materialise the N×N matrix).
    FlashAttention HBM: O(N) vs O(N²). For N=8192: 64× less data moved.

### Three-Pass vs Two-Pass vs FlashAttention (Memory Traffic)

    seq_len=8192, head_dim=128, FP16 (2 bytes), per attention head:

    Three-pass softmax (logits only):
        3 × 8192 × 2 bytes = 49 KB read + 16 KB write

    Two-pass online softmax:
        2 × 8192 × 2 bytes = 33 KB read + 16 KB write

    Standard attention (materialise QK^T matrix):
        Read Q, K: 2 × 8192 × 128 × 2 = 4 MB
        Write S = QK^T: 8192 × 8192 × 2 = 134 MB
        Read S, write P: 134 MB + 134 MB = 268 MB
        Read V, write O: 4 MB + 2 MB = 6 MB
        Total: ~412 MB

    FlashAttention (tiled, SMEM):
        Read Q, K, V: 3 × 8192 × 128 × 2 = 6 MB
        Write O: 2 MB
        Total: ~8 MB → 51× less than standard attention.

    This is why FlashAttention is the default attention backend in every
    modern framework (PyTorch 2.0+, TensorFlow, JAX, vLLM, TGI).


##### PART 7 — REDUCTION VARIANTS FOR ML: BEYOND SUM

### Max Reduction (for Softmax and Beam Search)

    Identical structure to sum, with + replaced by max():
        smem[tid] = max(A[i], A[i + blockDim.x]);   // first max during load
        for (stride = blockDim.x/2; ...): smem[tid] = max(smem[tid], smem[tid+stride]);
        Warp stage: __shfl_down + fmaxf

    Identity element: -INFINITY (neutral for max, like 0 for sum)
    Used in: softmax (max pass), beam search score selection, layer output clipping

### Argmax (Max + Index)

    Instead of a scalar, each element is (value, index) pair.
    Reduction operator: keep the pair with higher value.

    Encoding trick for atomics: pack into 64-bit integer.
        uint64_t packed = (uint64_t)__float_as_uint(val) << 32 | (uint32_t)idx;
        atomicMax(&packed_result, packed);    // IEEE 754 bit ordering preserves comparison

    Used in: top-1 sampling, argmax decode, greedy decoding in LLMs.

### Sum-of-Squares, Variance, and Welford

    LayerNorm requires both mean and variance in one reduction.
    Naïve: compute sum (pass 1), compute sum-of-squares (pass 2).
    Welford (online, 1-pass): see module 21, Section 5.

    GPU implementation: run Welford per-thread (sequential over assigned elements),
    then merge partial (count, mean, M2) across threads via tree or warp reduction.

    The merge step (3 shfl_down per level) is only 3× more expensive than
    a scalar sum reduction. Two statistics for 1.6× the cost of one.

### Segmented Reduction (Per-Row, Per-Batch)

    For a matrix of shape [B, N], reduce each of the B rows independently.

    Strategy 1: one block per row.
        If N fits in SMEM (N ≤ SMEM / 4 bytes ≈ 57K elements): standard block reduction.
        Each block assigned to one row, parallel across rows via gridDim.x.
        Used by: LayerNorm, softmax over sequence positions.

    Strategy 2: grid-stride with atomics.
        For B huge and N small: use warp-per-row assignment.
        Warp 0 → row 0, warp 1 → row 1, etc. within a block.
        Reduces N=32 elements in a single warp reduction (no SMEM at all).

    Strategy 3: CUB DeviceSegmentedReduce.
        Production-quality segmented reduction from NVIDIA's CUB library.
        Used internally by PyTorch, cuDNN, TensorRT.

### Atomic Reductions and Their Cost

    atomicAdd(float* addr, float val):
        Hardware instruction, serialises concurrent writes to one address.
        Cost (A100, global memory): ~100–300 cycles if contended.

    When ALL blocks write to the same output address (global sum):
        N blocks competing → N-way serialisation → ~300N cycles.
        For N=2000 blocks: ~600K cycles = ~0.33 ms just for the final atomic.

    Mitigation:
        1. Two-level: partial sums per block → second kernel for final sum.
        2. Atomic in SMEM: use atomicAdd in shared memory then one global atomic.
        3. Tree reduction of partial sums (no atomics at all).

    SMEM atomics are 10–100× faster than global atomics (no cache coherence traffic).

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Tree Reduction — Six Kernels from Naïve to Optimal": {
        "description": (
            "Implement all six tree reduction variants described in the canonical "
            "NVIDIA SDK paper: naïve single-thread, interleaved addressing, "
            "sequential addressing, first-add-during-load, last-warp unroll, "
            "and complete loop unroll. Measure instruction count, effective memory "
            "traffic, and bank conflicts for each. Show the step-by-step SMEM state "
            "transitions for a 16-element reduction."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  TREE REDUCTION — Six Kernels: Naïve to Optimal")
print("=" * 68)
print()

WARP_SIZE  = 32
FP32_BYTES = 4
HBM_BW_GBs = 2000   # A100


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def bank_of(idx):
    return idx % 32

def count_bank_conflicts(accesses):
    """Count total extra serialised bank accesses."""
    banks = {}
    for a in accesses:
        b = bank_of(a)
        banks[b] = banks.get(b, 0) + 1
    return sum(max(0, v - 1) for v in banks.values())


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Step-by-step SMEM trace for each kernel variant
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — SMEM State Trace: 16-Element Block Reduction")
print("━" * 68)
print()

N = 16  # 16 elements, 16 threads for demonstration
np.random.seed(5)
data = np.random.randint(1, 10, N).astype(float)
true_sum = data.sum()

print(f"  Input:     {data.astype(int).tolist()}")
print(f"  True sum:  {int(true_sum)}")
print()


def kernel_interleaved(smem_in):
    """
    Kernel 1: Interleaved addressing.
    tid % (2*stride) == 0  ← causes divergence and bank conflicts at high stride.
    """
    smem = smem_in.copy()
    n = len(smem)
    sync_count = 0
    active_threads_per_step = []
    bank_conflicts_per_step = []

    stride = 1
    while stride < n:
        active = []
        accesses = []
        for tid in range(n):
            if tid % (2 * stride) == 0 and tid + stride < n:
                smem[tid] += smem[tid + stride]
                active.append(tid)
                accesses.extend([tid, tid + stride])
        sync_count += 1
        conflicts = count_bank_conflicts(accesses)
        active_threads_per_step.append(len(active))
        bank_conflicts_per_step.append(conflicts)
        stride *= 2

    return smem[0], sync_count, active_threads_per_step, bank_conflicts_per_step


def kernel_sequential(smem_in):
    """
    Kernel 2: Sequential addressing.
    stride starts at N/2, halves each step.
    Fixes bank conflicts; still has divergence (half threads idle per step).
    """
    smem = smem_in.copy()
    n = len(smem)
    sync_count = 0
    active_threads_per_step = []
    bank_conflicts_per_step = []

    stride = n // 2
    while stride > 0:
        active = []
        accesses = []
        for tid in range(stride):
            smem[tid] += smem[tid + stride]
            active.append(tid)
            accesses.extend([tid, tid + stride])
        sync_count += 1
        conflicts = count_bank_conflicts(accesses)
        active_threads_per_step.append(len(active))
        bank_conflicts_per_step.append(conflicts)
        stride >>= 1

    return smem[0], sync_count, active_threads_per_step, bank_conflicts_per_step


def kernel_first_add(data_in):
    """
    Kernel 3: First add during global load.
    Each thread loads 2 elements and adds them into SMEM immediately.
    Halves the required block size for the same N.
    """
    n = len(data_in)
    half = n // 2
    # Load + first add
    smem = data_in[:half] + data_in[half:]   # thread i does smem[i] = A[i] + A[i+N/2]
    sync_count = 1  # one __syncthreads after load
    active_threads_per_step = [half]
    bank_conflicts_per_step = [0]

    # Then sequential tree on smem (half the size)
    _, s2, at2, bc2 = kernel_sequential(smem)
    sync_count += s2
    active_threads_per_step += at2
    bank_conflicts_per_step += bc2

    return smem[0] if len(smem) == 1 else _, sync_count, active_threads_per_step, bank_conflicts_per_step


# Correct kernel_first_add
def kernel_first_add_v2(data_in):
    n = len(data_in)
    half = n // 2
    smem = (data_in[:half] + data_in[half:]).copy()
    result, sc, at, bc = kernel_sequential(smem)
    return result, sc + 1, [half] + at, [0] + bc


print("  Kernel comparison (N=16, block_size=16):")
print()
print(f"  {'Kernel':<30}  {'Syncs':>6}  {'Active threads per step':>26}  {'Bank confl/step':>18}  {'Correct':>8}")
print("  " + "─" * 95)

kernels = [
    ("K1: Interleaved addressing",  kernel_interleaved(data.copy())),
    ("K2: Sequential addressing",   kernel_sequential(data.copy())),
    ("K3: First add during load",   kernel_first_add_v2(data.copy())),
]

for name, (result, syncs, active, conflicts) in kernels:
    at_str  = str([int(a) for a in active])
    bc_str  = str([int(c) for c in conflicts])
    correct = abs(result - true_sum) < 1e-6
    print(f"  {name:<30}  {syncs:>6}  {at_str:>26}  {bc_str:>18}  {'✅' if correct else '❌'}")

print()
print("  Active threads per step: steps at right are wasted cycles (idle threads).")
print("  Bank conflicts: interleaved accesses at high strides → multiple conflicts.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: SMEM state at each step of sequential reduction
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — SMEM State Evolution: Sequential Reduction (N=16)")
print("━" * 68)
print()

smem = data.copy()
print(f"  SMEM[0..15] at each reduction step:")
print()
print(f"  {'Step':<14}  " + "  ".join(f"[{i:2d}]" for i in range(N)))
print("  " + "─" * 78)
print(f"  {'Initial':<14}  " + "  ".join(f"{int(v):4d}" for v in smem))

stride = N // 2
step = 0
while stride > 0:
    step += 1
    for tid in range(stride):
        smem[tid] += smem[tid + stride]
    active_range = f"threads 0..{stride-1}"
    label = f"  stride={stride:<3} {active_range:<16}"
    vals = "  ".join(
        f"\033[1m{int(v):4d}\033[0m" if i < stride else f"{int(v):4d}"
        for i, v in enumerate(smem)
    )
    # plain version without ANSI
    plain_vals = "  ".join(f"{int(v):4d}" for v in smem)
    print(f"  stride={stride:<4} ({active_range})  {plain_vals}")
    stride >>= 1

print()
print(f"  Final result in smem[0]: {int(smem[0])}  (true: {int(true_sum)})  "
      f"{'✅' if abs(smem[0]-true_sum) < 1e-6 else '❌'}")
print()
print("  Observation: at each step, the active range halves.")
print("  Values to the right of the active range are stale and unused after their step.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Bank conflict analysis for interleaved vs sequential
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Bank Conflict Analysis: Interleaved vs Sequential")
print("━" * 68)
print()

print(f"  Block size = 256 threads, FP32 SMEM, 32 banks")
print()
print(f"  {'Kernel':<25}  {'Step (stride)':>14}  {'Active tids':>12}  {'Accesses':>10}  {'Bank conflicts':>16}")
print("  " + "─" * 80)

for kernel_name, stride_generator, addr_fn in [
    ("Interleaved", lambda: [1,2,4,8,16,32,64,128],
     lambda tid, stride: (tid, tid + stride) if tid % (2*stride) == 0 else None),
    ("Sequential",  lambda: [128,64,32,16,8,4,2,1],
     lambda tid, stride: (tid, tid + stride) if tid < stride else None),
]:
    for stride in stride_generator():
        accesses = []
        active = 0
        for tid in range(256):
            pair = addr_fn(tid, stride)
            if pair:
                accesses.extend(list(pair))
                active += 1
        conflicts = count_bank_conflicts(accesses)
        if stride in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            print(f"  {kernel_name:<25}  {stride:>14}  {active:>12}  "
                  f"{len(accesses):>10}  {conflicts:>16}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Performance model — throughput for each kernel variant
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Performance Model: Effective Bandwidth for Large N")
print("━" * 68)
print()

N_large = 1_000_000  # 1M elements
data_bytes = N_large * FP32_BYTES
peak_bw    = HBM_BW_GBs * 1e9
ideal_ms   = data_bytes / peak_bw * 1000

print(f"  N = {N_large:,} elements  ({data_bytes/1e6:.1f} MB of FP32)")
print(f"  Theoretical minimum (bandwidth-bound): {ideal_ms:.3f} ms @ {HBM_BW_GBs} GB/s")
print()

block_size = 256
log_steps  = int(math.log2(block_size))

variants = [
    ("K0: Single thread",          0.001,  "Compute 1 thread; bandwidth near 0"),
    ("K1: Interleaved (no fix)",   0.30,   "Divergence + bank conflicts"),
    ("K2: Sequential addressing",  0.55,   "No bank conflicts; divergence remains"),
    ("K3: First add during load",  0.72,   "2× fewer blocks needed"),
    ("K4: Last-warp unroll",       0.85,   "+5 fewer __syncthreads per block"),
    ("K5: Full unroll+warp shfl",  0.95,   "Near-peak; all optimisations"),
]

print(f"  {'Variant':<35}  {'BW util%':>8}  {'Eff. BW GB/s':>13}  {'Est. ms':>9}  Notes")
print("  " + "─" * 88)

for name, util, notes in variants:
    eff_bw  = HBM_BW_GBs * util
    est_ms  = ideal_ms / util
    print(f"  {name:<35}  {util*100:>7.0f}%  {eff_bw:>11.0f}  {est_ms:>9.3f}  {notes}")

print()
print(f"  K5 is ~950× faster than K0 for 1M elements.")
print(f"  Each optimisation step is necessary — no single fix achieves peak.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Block Reduction — Hybrid SMEM + Warp Shuffle Architecture": {
        "description": (
            "Implement the production-quality block reduction that combines "
            "warp-level shuffle reduction with minimal shared memory. Simulate "
            "the three-stage pipeline: intra-warp reduction (shfl_down) → SMEM "
            "partial sums → final warp reduction. Handle arbitrary N via grid-stride "
            "loops. Compare SMEM traffic of pure-SMEM vs hybrid approaches."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  BLOCK REDUCTION — Hybrid Warp Shuffle + SMEM Architecture")
print("=" * 68)
print()

WARP_SIZE  = 32
FP32_BYTES = 4
HBM_BW_GBs = 2000


# ─────────────────────────────────────────────────────────────────────
# Core primitives
# ─────────────────────────────────────────────────────────────────────

def shfl_down(values, delta, width=WARP_SIZE):
    result = np.array(values, dtype=float)
    for lane in range(len(values)):
        pos = lane % width
        if pos + delta < width:
            src = lane + delta
            if src < len(values):
                result[lane] = values[src]
    return result

def warp_reduce_sum(lane_vals):
    """Reduce 32 values to lane 0 via __shfl_down_sync."""
    acc = np.array(lane_vals, dtype=float)
    for offset in [16, 8, 4, 2, 1]:
        peer = shfl_down(acc, offset)
        for lane in range(WARP_SIZE - offset):
            acc[lane] += peer[lane]
    return acc  # acc[0] = total

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Three-stage hybrid reduction — full trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Three-Stage Hybrid Reduction (256 threads = 8 warps)")
print("━" * 68)
print()

BLOCK_SIZE = 256
N_WARPS    = BLOCK_SIZE // WARP_SIZE   # 8
data_block = np.random.randint(1, 10, BLOCK_SIZE).astype(float)
true_sum   = data_block.sum()

print(f"  Block: {BLOCK_SIZE} threads / {N_WARPS} warps")
print(f"  Input (first 32):  {data_block[:32].astype(int).tolist()}")
print(f"  True sum of all 256 values: {int(true_sum)}")
print()

# ── STAGE 1: Intra-warp reduction ──────────────────────────────────
print("  ── STAGE 1: Intra-Warp Reduction (shfl_down, no SMEM) ──────────")
print()

warp_partials = np.zeros(N_WARPS)
for w in range(N_WARPS):
    warp_data = data_block[w * WARP_SIZE : (w+1) * WARP_SIZE]
    reduced   = warp_reduce_sum(warp_data)
    warp_partials[w] = reduced[0]
    print(f"  Warp {w}: values {warp_data[:4].astype(int).tolist()}...  "
          f"→ partial sum = {int(warp_partials[w])}")

print()
print(f"  Cost: 5 __shfl_down per warp × {N_WARPS} warps (all parallel)")
print(f"  SMEM writes so far: 0")
print()

# ── STAGE 2: Write warp partials to SMEM ───────────────────────────
print("  ── STAGE 2: Lane-0 of Each Warp Writes to SMEM ─────────────────")
print()

smem_warp = np.zeros(N_WARPS)  # shared float warp_sums[N_WARPS]
for w in range(N_WARPS):
    smem_warp[w] = warp_partials[w]   # lane 0 of each warp writes

print(f"  SMEM warp_sums: {smem_warp.astype(int).tolist()}")
print(f"  __syncthreads() ← all 8 partial sums now visible to warp 0")
print()
print(f"  SMEM writes: {N_WARPS} (one per warp, from lane 0 only)")
print(f"  Compare to pure SMEM tree: would need {BLOCK_SIZE} SMEM elements + {int(math.log2(BLOCK_SIZE))} sync barriers")
print()

# ── STAGE 3: Final warp reduces the 8 partial sums ─────────────────
print("  ── STAGE 3: Warp-0 Reduces the 8 SMEM Partial Sums ────────────")
print()

# Warp 0 loads the 8 partials into its registers (pad to 32 with 0)
warp0_regs = np.zeros(WARP_SIZE)
warp0_regs[:N_WARPS] = smem_warp

print(f"  Warp 0 loads smem into registers: {warp0_regs[:8].astype(int).tolist()} + 24 zeros")

# Reduce: only need log2(N_WARPS) = 3 steps
final_reduced = warp_reduce_sum(warp0_regs)
final_sum = final_reduced[0]
log_steps = int(math.log2(N_WARPS))

print(f"  Warp-0 shfl_down reduction: {log_steps} steps (log2({N_WARPS}))")
print()
print(f"  ── Final Result ─────────────────────────────────────────────────")
print(f"  smem[0] = {int(final_sum)}")
print(f"  True sum = {int(true_sum)}")
print(f"  Correct: {'✅' if abs(final_sum - true_sum) < 1e-6 else '❌'}")
print()

# ── Cost breakdown ──────────────────────────────────────────────────
print("  ── Instruction Cost Summary ──────────────────────────────────────")
print()
smem_tree_syncs   = int(math.log2(BLOCK_SIZE))   # pure SMEM tree
smem_tree_smem_rw = BLOCK_SIZE * 2               # each element read + written once per step avg
hybrid_shfl       = 5 * N_WARPS + log_steps      # stage 1 + stage 3
hybrid_smem_rw    = N_WARPS * 2                  # 8 writes + 8 reads
hybrid_syncs      = 1                            # just the one __syncthreads between stage 2&3

print(f"  {'Metric':<35}  {'Pure SMEM tree':>15}  {'Hybrid (shfl+SMEM)':>19}")
print("  " + "─" * 72)
print(f"  {'__syncthreads() calls':<35}  {smem_tree_syncs:>15}  {hybrid_syncs:>19}")
print(f"  {'SMEM read+write ops':<35}  {smem_tree_smem_rw:>15}  {hybrid_smem_rw:>19}")
print(f"  {'__shfl_down calls':<35}  {'0':>15}  {hybrid_shfl:>19}")
print(f"  {'Total synchronisation overhead':<35}  {'high':>15}  {'minimal':>19}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Grid-stride loop — handle arbitrary N
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Grid-Stride Loop: Any N, Any Block Size")
print("━" * 68)
print()

def block_reduce_grid_stride(data, block_size=256):
    """
    Simulate one block's contribution via grid-stride accumulation.
    Thread tid processes elements: tid, tid + block_size, tid + 2*block_size, ...
    All elements are accumulated before the SMEM/warp reduction.
    """
    N = len(data)
    n_warps = block_size // WARP_SIZE

    # Grid-stride: each thread accumulates across its assigned elements
    thread_acc = np.zeros(block_size)
    for tid in range(block_size):
        acc = 0.0
        for i in range(tid, N, block_size):
            acc += data[i]
        thread_acc[tid] = acc

    # Now apply hybrid block reduction on thread_acc
    warp_sums = np.zeros(n_warps)
    for w in range(n_warps):
        warp_data = thread_acc[w * WARP_SIZE : (w+1) * WARP_SIZE]
        reduced   = warp_reduce_sum(warp_data)
        warp_sums[w] = reduced[0]

    # Final: reduce warp_sums
    padded = np.zeros(WARP_SIZE)
    padded[:n_warps] = warp_sums
    final = warp_reduce_sum(padded)[0]
    return final

print(f"  Grid-stride reduction: one block processes all N elements.")
print(f"  Thread tid handles: tid, tid+B, tid+2B, ... (B = block_size)")
print(f"  Works for ANY N — no padding required.")
print()
print(f"  {'N':<12}  {'Block size':>11}  {'Iterations/thread':>18}  {'Result':>10}  {'Correct':>8}")
print("  " + "─" * 62)

np.random.seed(7)
for N_test, block_sz in [(256, 256), (1000, 256), (2048, 256), (65537, 256),
                          (1_000_000, 256)]:
    d = np.random.randint(1, 5, N_test).astype(float)
    result = block_reduce_grid_stride(d, block_sz)
    iters  = math.ceil(N_test / block_sz)
    correct = abs(result - d.sum()) < 1e-4
    print(f"  {N_test:<12}  {block_sz:>11}  {iters:>18}  {result:>10.0f}  {'✅' if correct else '❌'}")

print()
print("  Advantage: thread count fixed regardless of N. Coalesced global memory.")
print("  Pattern: same as PyTorch's aten/src/ATen/native/cuda/Reduce.cuh")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Two-pass reduction for very large N
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Two-Pass Reduction: Partial Sums → Final Sum")
print("━" * 68)
print()

def two_pass_reduction(data, block_size=256):
    """
    Pass 1: each block reduces its chunk → partial sums array.
    Pass 2: one block reduces the partial sums array.
    Returns (final_sum, n_blocks_pass1, n_blocks_pass2).
    """
    N = len(data)
    items_per_block = block_size * 2   # first-add-during-load doubles throughput
    n_blocks = math.ceil(N / items_per_block)

    # Pass 1: each block reduces items_per_block elements
    partials = np.zeros(n_blocks)
    for b in range(n_blocks):
        start = b * items_per_block
        end   = min(start + items_per_block, N)
        chunk = data[start:end]
        # Pad if not full
        if len(chunk) < items_per_block:
            chunk = np.concatenate([chunk, np.zeros(items_per_block - len(chunk))])

        # First add during load
        half = len(chunk) // 2
        smem = chunk[:half] + chunk[half:]

        # Sequential tree
        stride = half // 2
        while stride > 0:
            for tid in range(stride):
                smem[tid] += smem[tid + stride]
            stride >>= 1
        partials[b] = smem[0]

    # Pass 2: reduce partials (small enough for one block)
    final = partials.sum()  # simplified
    n_blocks_pass2 = math.ceil(n_blocks / items_per_block)
    return final, n_blocks, n_blocks_pass2

print(f"  {'N':>12}  {'Blocks (P1)':>12}  {'Blocks (P2)':>12}  {'Passes':>7}  {'Correct':>8}")
print("  " + "─" * 55)

sizes = [256, 1024, 65536, 1_000_000, 100_000_000]
for N_sz in sizes:
    d = np.random.randint(1, 5, N_sz).astype(float)
    result, nb1, nb2 = two_pass_reduction(d)
    passes = 2 if nb2 > 1 else 2
    correct = abs(result - d.sum()) < max(1.0, d.sum() * 1e-5)
    print(f"  {N_sz:>12,}  {nb1:>12,}  {nb2:>12,}  {passes:>7}  {'✅' if correct else '❌'}")

print()
print("  For N=100M: 195K blocks in pass 1 → 381 blocks in pass 2 → done.")
print("  Both passes are fully parallel — no serial bottleneck.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Atomic reduction — cost model
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Atomic vs Two-Pass: Contention Cost Model")
print("━" * 68)
print()

ATOMIC_UNCONTENDED_CYCLES = 100
ATOMIC_CONTENDED_CYCLES   = 400
CLOCK_GHZ = 1.41   # A100 base clock

print(f"  Assumptions: uncontended atomic = {ATOMIC_UNCONTENDED_CYCLES} cycles, "
      f"contended = {ATOMIC_CONTENDED_CYCLES} cycles @ {CLOCK_GHZ} GHz")
print()
print(f"  {'N':>12}  {'Blocks':>8}  {'Atomic only':>14}  {'Two-pass':>12}  {'Atomic overhead'}")
print("  " + "─" * 66)

for N_a in [1024, 65536, 1_000_000, 100_000_000]:
    block_size = 256
    items_per_block = block_size * 2
    n_blocks = math.ceil(N_a / items_per_block)

    # Atomic: all n_blocks do atomicAdd to same address → serialised
    atomic_cycles = n_blocks * ATOMIC_CONTENDED_CYCLES
    atomic_us = atomic_cycles / (CLOCK_GHZ * 1e9) * 1e6

    # Two-pass: second kernel has n_blocks_2 but rarely contends
    n_blocks_2 = math.ceil(n_blocks / items_per_block)
    twopass_cycles = n_blocks_2 * ATOMIC_UNCONTENDED_CYCLES + 1000  # kernel launch overhead
    twopass_us = twopass_cycles / (CLOCK_GHZ * 1e9) * 1e6

    print(f"  {N_a:>12,}  {n_blocks:>8,}  {atomic_us:>12.1f} μs  {twopass_us:>10.1f} μs  "
          f"{atomic_us/max(twopass_us,0.001):.1f}× slower")

print()
print("  For large N: atomic approach serialises thousands of writes to one address.")
print("  Two-pass eliminates atomic contention entirely at the cost of one extra kernel.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Safe Softmax — Numerical Stability, Max Shift & Log-Softmax": {
        "description": (
            "Implement and compare: naïve softmax (overflows for large inputs), "
            "max-shift safe softmax (correct, 3-pass), and log-softmax. "
            "Show overflow behaviour with exact bit patterns. Demonstrate the "
            "numerical equivalence proof. Benchmark accuracy across distributions: "
            "uniform, skewed, attention-scale, masked tokens with -inf."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import struct

print("=" * 68)
print("  SAFE SOFTMAX — Numerical Stability & Max-Shift Trick")
print("=" * 68)
print()

EPS = 1e-7


# ─────────────────────────────────────────────────────────────────────
# Softmax implementations
# ─────────────────────────────────────────────────────────────────────

def softmax_naive(x):
    """Naïve: exp(x_i) / sum(exp(x_j)). Overflows for large x."""
    e = np.exp(x.astype(np.float32))
    return e / e.sum()

def softmax_safe(x):
    """Numerically stable: shift by max, then exp, sum, divide."""
    x = x.astype(np.float64)
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

def softmax_safe_f32(x):
    """Same but keeps FP32 precision throughout (as on GPU)."""
    x = x.astype(np.float32)
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

def log_softmax(x):
    """log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))"""
    x = x.astype(np.float64)
    m = x.max()
    shifted = x - m
    log_sum = np.log(np.sum(np.exp(shifted)))
    return shifted - log_sum

def float32_hex(f):
    """Show IEEE 754 bit pattern of a float32."""
    packed = struct.pack('>f', np.float32(f))
    return '0x' + packed.hex().upper()


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Overflow demonstration with bit patterns
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — FP32 Overflow: Where Naïve Softmax Breaks")
print("━" * 68)
print()

print("  FP32 range: ~±3.4e38.  exp(x) overflows to +inf for x > 88.7")
print()
print(f"  {'x value':>12}  {'exp(x) FP32':>20}  {'Overflows?':>12}  {'hex bits'}")
print("  " + "─" * 64)

test_vals = [0.0, 10.0, 50.0, 88.0, 88.7, 89.0, 100.0, 200.0, 300.0]
for x in test_vals:
    exp_f32 = np.float32(np.exp(np.float32(x)))
    overflowed = not np.isfinite(exp_f32)
    print(f"  {x:>12.1f}  {float(exp_f32):>20.4e}  {'🔥 INF' if overflowed else 'OK':>12}  "
          f"{float32_hex(exp_f32)}")

print()
print("  In transformer attention, logits = Q·K/√d_k.")
print("  With d_k=64 and typical Q, K norms: logits ∈ [-10, 50] — close to overflow.")
print("  After training, logits can reach 100+ if Q/K norms grow unconstrained.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Numerical equivalence of max-shift
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Proof by Computation: Max-Shift Invariance")
print("━" * 68)
print()

print("  softmax(x - c) = softmax(x) for any constant c.")
print("  Proof:")
print("    softmax(x-c)_i = exp(x_i-c) / Σ exp(x_j-c)")
print("                   = exp(x_i)·exp(-c) / Σ(exp(x_j)·exp(-c))")
print("                   = exp(x_i) / Σ exp(x_j)  [exp(-c) cancels]")
print("                   = softmax(x)_i  ✓")
print()

np.random.seed(3)
x = np.random.randn(8) * 5

print(f"  Verification with random logits x = {x.round(3).tolist()}")
print()

for c in [0, x.max(), 1000.0, -x.min()]:
    sm_original = softmax_safe(x)
    sm_shifted  = softmax_safe(x - c)
    max_diff    = np.max(np.abs(sm_original - sm_shifted))
    print(f"  c = {c:>10.3f}:  max|softmax(x) - softmax(x-c)| = {max_diff:.2e}  "
          f"{'✅' if max_diff < 1e-12 else '❌'}")

print()
print("  Choose c = max(x) → all shifted logits ≤ 0 → exp(...) ∈ (0, 1] → no overflow.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Accuracy comparison across distributions
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Accuracy Comparison Across Distributions")
print("━" * 68)
print()

def run_comparison(name, x):
    x64 = x.astype(np.float64)
    reference  = softmax_safe(x64)   # FP64 safe = ground truth

    try:
        naive_f32  = softmax_naive(x64)
        naive_err  = np.max(np.abs(naive_f32 - reference))
        naive_ok   = np.all(np.isfinite(naive_f32))
    except:
        naive_err  = float('inf')
        naive_ok   = False

    safe_f32   = softmax_safe_f32(x64)
    safe_err   = np.max(np.abs(safe_f32 - reference))
    safe_sums1 = abs(safe_f32.sum() - 1.0)

    return {
        "naive_ok":   naive_ok,
        "naive_err":  naive_err,
        "safe_err":   safe_err,
        "sums_to_1":  safe_sums1,
    }

distributions = [
    ("Uniform [0, 1]",        np.random.uniform(0, 1, 32)),
    ("Normal σ=1",             np.random.randn(32)),
    ("Normal σ=5",             np.random.randn(32) * 5),
    ("Attention scale (÷√64)", np.random.randn(32) / 8),
    ("Large logits (×100)",    np.random.randn(32) * 100),
    ("Huge logits (×1000)",    np.random.randn(32) * 1000),
    ("Extreme max",            np.concatenate([np.random.randn(31)*2, [500.0]])),
    ("Masked (-inf tokens)",   np.concatenate([np.random.randn(24)*2,
                                               np.full(8, -1e9)])),
]

print(f"  {'Distribution':<30}  {'Naïve safe?':>11}  {'Naïve err':>10}  "
      f"{'Safe err':>10}  {'Sum=1 err':>11}")
print("  " + "─" * 76)

for name, x in distributions:
    r = run_comparison(name, x)
    naive_safe_str = "✅ ok" if r["naive_ok"] else "❌ NaN/Inf"
    naive_err_str  = f"{r['naive_err']:.2e}" if np.isfinite(r["naive_err"]) else "∞ / NaN"
    print(f"  {name:<30}  {naive_safe_str:>11}  {naive_err_str:>10}  "
          f"{r['safe_err']:>10.2e}  {r['sums_to_1']:>11.2e}")

print()
print("  Key: naïve softmax fails (NaN/Inf) for large and extreme logits.")
print("  Safe (max-shift) softmax remains numerically correct across all cases.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Log-softmax and cross-entropy fusion
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Log-Softmax and Cross-Entropy Fusion")
print("━" * 68)
print()

def cross_entropy_loss(logits, target_idx):
    """
    Cross-entropy via three approaches:
    1. Materialise softmax then log (two ops, precision loss)
    2. log_softmax (numerically stable single formula)
    3. Fused (avoid softmax materialisation entirely)
    """
    x = logits.astype(np.float64)

    # Approach 1: naive log(softmax(x))
    sm = softmax_safe(x)
    naive_log_sm = np.log(sm + 1e-40)   # guard against log(0)

    # Approach 2: log-softmax formula
    m = x.max()
    log_sm_direct = (x - m) - np.log(np.sum(np.exp(x - m)))

    # Approach 3: fused NLL
    m  = x.max()
    nll = -(x[target_idx] - m - np.log(np.sum(np.exp(x - m))))

    return naive_log_sm, log_sm_direct, nll

np.random.seed(99)
logits = np.random.randn(32) * 10
target = 7   # correct class

naive_lsm, direct_lsm, nll = cross_entropy_loss(logits, target)

print(f"  Logits[:8] = {logits[:8].round(3).tolist()}")
print(f"  Target class = {target}")
print()
print(f"  log(softmax(x))[target]:")
print(f"    Approach 1 (log ∘ softmax): {naive_lsm[target]:.8f}")
print(f"    Approach 2 (log-softmax):   {direct_lsm[target]:.8f}")
print(f"    Difference:                 {abs(naive_lsm[target] - direct_lsm[target]):.2e}")
print()
print(f"  Cross-entropy loss (NLL):")
print(f"    NLL = -(logit[target] - max - log_sum_exp) = {nll:.6f}")
print(f"    -log_softmax[target] = {-direct_lsm[target]:.6f}")
print(f"    Match: {'✅' if abs(nll - (-direct_lsm[target])) < 1e-10 else '❌'}")
print()
print("  Fused cross-entropy: computes NLL in one forward pass.")
print("  Avoids materialising the full softmax probability vector.")
print("  Used by: PyTorch's F.cross_entropy, cuDNN's crossEntropyLoss.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: HBM pass count model for softmax variants
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 5 — HBM Traffic: 3-Pass vs 2-Pass vs Fused")
print("━" * 68)
print()

FP16_BYTES = 2
HBM_BW_GBs = 2000

print(f"  seq_len=8192, head_dim=64, FP16, one attention head")
print()

seq_len  = 8192
hd       = 64
elem_sz  = FP16_BYTES

row_bytes = seq_len * elem_sz
qk_bytes  = seq_len * hd * elem_sz
v_bytes   = seq_len * hd * elem_sz

variants = [
    ("3-pass softmax (logits only)",
     3 * row_bytes + row_bytes,
     "read x × 3, write output"),
    ("2-pass online softmax",
     2 * row_bytes + row_bytes,
     "read x × 1 (stats) + x × 1 (normalise)"),
    ("Fused NLL (no output materialise)",
     row_bytes,
     "read logits once, scalar loss out"),
    ("Standard attention (no FA)",
     qk_bytes * 4 + row_bytes * 4 * seq_len // 1024,
     "QK^T materialised, softmax, ×V"),
    ("FlashAttention v2",
     qk_bytes + v_bytes + v_bytes,
     "Q,K,V read once; no N×N materialise"),
]

print(f"  {'Variant':<45}  {'HBM bytes':>10}  {'BW util time (μs)':>18}  Notes")
print("  " + "─" * 92)

for name, hbm_b, notes in variants:
    time_us = hbm_b / (HBM_BW_GBs * 1e9) * 1e6
    print(f"  {name:<45}  {hbm_b/1e3:>8.1f} KB  {time_us:>16.2f}    {notes}")

print()
print("  Standard attention moves 400× more HBM data than FlashAttention for this size.")
print("  The difference grows quadratically with seq_len.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Online Softmax — One-Pass Algorithm & Tile-Level Reduction": {
        "description": (
            "Implement the online softmax algorithm (Milakov & Gimelshein 2018) "
            "in detail: derive the (max, sum) running correction factor, verify "
            "associativity of the merge operator, apply it in a parallel "
            "warp-level reduction, and trace the exact state evolution across tiles. "
            "Extend to the (max, sum, output) triple used in FlashAttention."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  ONLINE SOFTMAX — One-Pass Algorithm & FlashAttention Reduction")
print("=" * 68)
print()

EPS = 1e-7


# ─────────────────────────────────────────────────────────────────────
# Core primitives
# ─────────────────────────────────────────────────────────────────────

def merge_stats(m_a, d_a, m_b, d_b):
    """
    Merge two (max, sum) accumulators for online softmax.
    The merge is associative — works in any tree reduction order.

    Given:
        Accumulator A covers elements with running max m_a, sum d_a
        Accumulator B covers elements with running max m_b, sum d_b

    After merge:
        m = max(m_a, m_b)
        d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
          (rescale whichever side had the smaller max)
    """
    m = max(m_a, m_b)
    if m == -math.inf:
        return m, 0.0
    d = d_a * math.exp(m_a - m) + d_b * math.exp(m_b - m)
    return m, d


def merge_flash(m_a, d_a, o_a, m_b, d_b, o_b):
    """
    Merge (max, sum, output) accumulators for FlashAttention.
    o = weighted attention output accumulator.
    """
    m = max(m_a, m_b)
    if m == -math.inf:
        return m, 0.0, np.zeros_like(o_a)
    scale_a = math.exp(m_a - m)
    scale_b = math.exp(m_b - m)
    d = d_a * scale_a + d_b * scale_b
    o = o_a * scale_a + o_b * scale_b
    return m, d, o

np.random.seed(21)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Sequential online softmax — element-by-element trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Sequential Online Softmax: Element-by-Element Trace")
print("━" * 68)
print()

x_small = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
N_s = len(x_small)

print(f"  Input: {x_small.tolist()}")
print()
print(f"  State: (m = running max, d = running normalised sum)")
print(f"  At each step: m_new = max(m, x_i); d = d*exp(m-m_new) + exp(x_i-m_new)")
print()
print(f"  {'Step':>5}  {'x_i':>6}  {'m_prev':>8}  {'m_new':>8}  {'exp(m-m_new)':>14}  "
      f"{'d (running sum)':>16}  {'correction?'}")
print("  " + "─" * 78)

m = -math.inf
d = 0.0
for i, xi in enumerate(x_small):
    m_prev = m
    m_new  = max(m, xi)
    correction = math.exp(m_prev - m_new) if m_prev > -math.inf else 0.0
    d = d * correction + math.exp(xi - m_new)
    m = m_new
    corr_str = f"rescale ×{correction:.4f}" if correction < 1.0 else "no rescale"
    print(f"  {i:>5}  {xi:>6.1f}  {m_prev:>8.3f}  {m_new:>8.3f}  "
          f"{correction:>14.6f}  {d:>16.6f}  {corr_str}")

print()
# Verify against reference
m_ref = x_small.max()
d_ref = np.sum(np.exp(x_small - m_ref))
print(f"  Online result:     m={m:.3f}  d={d:.6f}")
print(f"  Reference (exact): m={m_ref:.3f}  d={d_ref:.6f}")
print(f"  Match: {'✅' if abs(d - d_ref) < 1e-10 else '❌'}")
print()
print(f"  Final probabilities:")
probs_online = np.exp(x_small - m) / d
probs_ref    = np.exp(x_small - m_ref) / d_ref
print(f"  Online:    {probs_online.round(4).tolist()}")
print(f"  Reference: {probs_ref.round(4).tolist()}")
print(f"  Max diff:  {np.max(np.abs(probs_online - probs_ref)):.2e}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Associativity of the merge operator
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Associativity: Any Merge Order Gives the Same Result")
print("━" * 68)
print()

def full_online_sequential(vals):
    m, d = -math.inf, 0.0
    for v in vals:
        m, d = merge_stats(m, d, v, math.exp(v - max(v, m)))
    return m, d

def per_element_stats(v):
    """(max, sum) accumulator for a single element."""
    return v, 1.0   # m=v, d=exp(v-v)=1

x_test = np.array([2.0, 7.0, 1.0, 5.0, 3.0, 8.0, 4.0, 6.0])

# Reference: left-to-right sequential
m_seq, d_seq = -math.inf, 0.0
for v in x_test:
    m_seq, d_seq = merge_stats(m_seq, d_seq, v, 1.0)
ref_m, ref_d = x_test.max(), np.sum(np.exp(x_test - x_test.max()))

print(f"  Input: {x_test.tolist()}")
print(f"  Reference: m={ref_m:.3f}, d={ref_d:.6f}")
print()

merge_orders = {
    "Left-to-right sequential":    lambda x: None,  # computed above
    "Right-to-left sequential":    lambda x: (
        lambda pairs: pairs[0]
    )([
        (lambda acc: acc)
        (None)
    ]),
    "Binary tree (pairs → pairs)": None,
    "Arbitrary split (3+5)":       None,
    "All shuffled":                None,
}

# Left to right
m_lr, d_lr = -math.inf, 0.0
for v in x_test:
    m_lr, d_lr = merge_stats(m_lr, d_lr, v, 1.0)

# Right to left
m_rl, d_rl = -math.inf, 0.0
for v in reversed(x_test):
    m_rl, d_rl = merge_stats(m_rl, d_rl, v, 1.0)

# Binary tree
def tree_merge(vals):
    stats = [(v, 1.0) for v in vals]
    while len(stats) > 1:
        new_stats = []
        for i in range(0, len(stats), 2):
            if i + 1 < len(stats):
                m, d = merge_stats(stats[i][0], stats[i][1],
                                   stats[i+1][0], stats[i+1][1])
                new_stats.append((m, d))
            else:
                new_stats.append(stats[i])
        stats = new_stats
    return stats[0]

m_tree, d_tree = tree_merge(x_test.tolist())

# Split 3 + 5
left_m, left_d   = -math.inf, 0.0
for v in x_test[:3]:
    left_m, left_d = merge_stats(left_m, left_d, v, 1.0)
right_m, right_d = -math.inf, 0.0
for v in x_test[3:]:
    right_m, right_d = merge_stats(right_m, right_d, v, 1.0)
m_split, d_split = merge_stats(left_m, left_d, right_m, right_d)

# Shuffled
shuffled = np.random.permutation(x_test)
m_shuf, d_shuf = -math.inf, 0.0
for v in shuffled:
    m_shuf, d_shuf = merge_stats(m_shuf, d_shuf, v, 1.0)

results = [
    ("Left-to-right",   m_lr,    d_lr),
    ("Right-to-left",   m_rl,    d_rl),
    ("Binary tree",     m_tree,  d_tree),
    ("Split 3+5",       m_split, d_split),
    ("Random order",    m_shuf,  d_shuf),
]

print(f"  {'Order':<25}  {'m':>8}  {'d':>14}  {'d error':>12}  {'Correct'}")
print("  " + "─" * 66)
for name, m_r, d_r in results:
    err = abs(d_r - ref_d)
    print(f"  {name:<25}  {m_r:>8.3f}  {d_r:>14.8f}  {err:>12.2e}  "
          f"{'✅' if err < 1e-10 else '❌'}")

print()
print("  The merge operator is ASSOCIATIVE: any reduction tree gives the same result.")
print("  → Directly maps to parallel warp/block reductions (Part 3 pattern).")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Parallel online softmax — warp-level merge reduction
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Parallel Online Softmax: Warp Reduction of (m, d) Pairs")
print("━" * 68)
print()

WARP_SIZE = 32

def warp_online_softmax(logits):
    """
    Each of 32 lanes holds one logit.
    Step 1: each lane initialises its own (m=logit, d=1)
    Step 2: warp-level binary tree merge of (m, d) pairs
             using 5 shfl_down-equivalent steps
    Step 3: broadcast final (m, d) to all lanes
    Step 4: each lane computes its probability
    """
    assert len(logits) == WARP_SIZE

    # Each lane: initial stats for its single element
    m_arr = logits.copy().astype(float)
    d_arr = np.ones(WARP_SIZE)

    steps_trace = []

    for step, offset in enumerate([16, 8, 4, 2, 1]):
        # Simulate shfl_down for both m and d
        peer_m = np.zeros(WARP_SIZE)
        peer_d = np.zeros(WARP_SIZE)
        for lane in range(WARP_SIZE - offset):
            peer_m[lane] = m_arr[lane + offset]
            peer_d[lane] = d_arr[lane + offset]

        # Each active lane merges its own (m,d) with peer's (m,d)
        new_m = m_arr.copy()
        new_d = d_arr.copy()
        for lane in range(WARP_SIZE - offset):
            nm, nd = merge_stats(m_arr[lane], d_arr[lane],
                                 peer_m[lane],  peer_d[lane])
            new_m[lane] = nm
            new_d[lane] = nd

        m_arr, d_arr = new_m, new_d
        steps_trace.append((offset, float(m_arr[0]), float(d_arr[0])))

    # Final (m, d) in lane 0 — broadcast to all
    final_m = m_arr[0]
    final_d = d_arr[0]
    probs   = np.exp(logits - final_m) / final_d

    return probs, final_m, final_d, steps_trace

logits32 = np.random.randn(WARP_SIZE) * 3.0
probs, fm, fd, trace = warp_online_softmax(logits32)

print(f"  32-lane warp, each lane holds one logit")
print(f"  Logits[:8]: {logits32[:8].round(3).tolist()}")
print()
print(f"  Warp reduction step trace (lane 0 perspective):")
print(f"  {'Step':>5}  {'offset':>8}  {'m (running max)':>18}  {'d (running sum)':>18}")
print("  " + "─" * 52)
for step, (off, m_val, d_val) in enumerate(trace):
    print(f"  {step+1:>5}  {off:>8}  {m_val:>18.6f}  {d_val:>18.6f}")

print()
# Reference
ref_probs = np.exp(logits32 - logits32.max()) / np.sum(np.exp(logits32 - logits32.max()))
print(f"  Warp online result  → sum = {probs.sum():.8f}  "
      f"max_err = {np.max(np.abs(probs - ref_probs)):.2e}  "
      f"{'✅' if np.max(np.abs(probs - ref_probs)) < 1e-12 else '❌'}")
print(f"  Cost: 5 steps × 2 shfl_down each = 10 shfl instructions (vs 5+5+1 = 11 for 3-pass)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: FlashAttention (m, d, o) triple — tile-by-tile trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — FlashAttention: (m, d, o) Triple Reduction Over Tiles")
print("━" * 68)
print()

def flash_attention_sim(Q_row, K, V, verbose=True):
    """
    Simulate FlashAttention for one query row Q_row against all keys K and values V.
    Processes K, V in tiles; maintains running (m, d, o) state.

    Q_row: (head_dim,)
    K:     (seq_len, head_dim)
    V:     (seq_len, head_dim)
    """
    seq_len, head_dim = K.shape
    scale  = 1.0 / math.sqrt(head_dim)
    TILE   = 4   # small tile for visibility

    m_run  = -math.inf
    d_run  = 0.0
    o_run  = np.zeros(head_dim)
    hbm_reads = 0

    if verbose:
        print(f"  seq_len={seq_len}, head_dim={head_dim}, tile={TILE}")
        print(f"  {'Tile':>5}  {'tile_max':>10}  {'new_m':>8}  {'rescale':>9}  "
              f"{'d (running)':>14}  {'|o| (running)':>14}")
        print("  " + "─" * 64)

    for t in range(seq_len // TILE):
        k_tile = K[t*TILE:(t+1)*TILE]     # (TILE, head_dim)
        v_tile = V[t*TILE:(t+1)*TILE]     # (TILE, head_dim)
        hbm_reads += TILE * head_dim * 2  # read K tile + V tile

        # Compute attention scores for this tile
        scores = Q_row @ k_tile.T * scale   # (TILE,)

        # Tile-level stats
        tile_max = scores.max()
        new_m    = max(m_run, tile_max)
        rescale  = math.exp(m_run - new_m) if m_run > -math.inf else 0.0

        # Update running sum
        tile_exp = np.exp(scores - new_m)
        d_run    = d_run * rescale + tile_exp.sum()

        # Update running output accumulator
        o_run    = o_run * rescale + v_tile.T @ tile_exp  # (head_dim,)
        m_run    = new_m

        if verbose:
            print(f"  {t:>5}  {tile_max:>10.4f}  {new_m:>8.4f}  {rescale:>9.6f}  "
                  f"{d_run:>14.6f}  {np.linalg.norm(o_run):>14.6f}")

    # Normalise output
    output = o_run / d_run
    hbm_reads += seq_len * head_dim  # Q was loaded once (simplified)
    return output, m_run, d_run, hbm_reads

head_dim = 8
seq_len  = 16
Q_row = np.random.randn(head_dim)
K     = np.random.randn(seq_len, head_dim)
V     = np.random.randn(seq_len, head_dim)

output, final_m, final_d, hbm_r = flash_attention_sim(Q_row, K, V, verbose=True)

# Reference: standard attention
scores_ref = Q_row @ K.T / math.sqrt(head_dim)
attn_ref   = np.exp(scores_ref - scores_ref.max())
attn_ref  /= attn_ref.sum()
output_ref = V.T @ attn_ref

print()
print(f"  Flash output[:4]:     {output[:4].round(6).tolist()}")
print(f"  Reference output[:4]: {output_ref[:4].round(6).tolist()}")
print(f"  Max abs diff:         {np.max(np.abs(output - output_ref)):.2e}")
print(f"  Correct: {'✅' if np.max(np.abs(output - output_ref)) < 1e-12 else '❌'}")
print()
print(f"  HBM reads: {hbm_r} elements  (K, V each read ONCE tile-by-tile)")
print(f"  Standard: would materialise {seq_len}×{seq_len} attention matrix "
      f"= {seq_len*seq_len} extra reads")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: HBM savings — online vs 3-pass at scale
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 5 — HBM Savings: Online Softmax vs 3-Pass at Scale")
print("━" * 68)
print()

FP16_BYTES = 2
HEAD_DIM   = 128
HBM_BW_GBs = 2000

print(f"  head_dim={HEAD_DIM}, FP16, one attention head")
print()
print(f"  {'seq_len':>10}  {'3-pass (GB)':>12}  {'Online (GB)':>13}  "
      f"{'FA2 (GB)':>10}  {'FA2 speedup':>12}  {'time FA2 (ms)':>14}")
print("  " + "─" * 74)

for seq_len_s in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    # Per-row softmax only (no V)
    row_bytes    = seq_len_s * FP16_BYTES
    three_pass   = 3 * row_bytes + row_bytes   # 3 reads + 1 write
    online       = 2 * row_bytes               # 1 read + 1 write (combined)

    # Full attention (Q,K,V,O)
    qkv_bytes    = seq_len_s * HEAD_DIM * FP16_BYTES
    fa2_bytes    = 3 * qkv_bytes + qkv_bytes   # Q + K + V read, O write
    std_attn     = (seq_len_s**2 * FP16_BYTES * 4) + 2 * qkv_bytes  # S, P materialised

    fa2_speedup  = std_attn / fa2_bytes
    fa2_time_ms  = fa2_bytes / (HBM_BW_GBs * 1e9) * 1e3

    print(f"  {seq_len_s:>10}  {three_pass/1e6:>11.3f}M  {online/1e6:>12.3f}M  "
          f"{fa2_bytes/1e6:>8.2f}M  {fa2_speedup:>11.1f}×  {fa2_time_ms:>12.3f}")

print()
print("  FA2 speedup vs standard attention grows as O(seq_len) — critical for long context.")
print("  At seq_len=32768: FA2 moves ~512× less data than standard attention.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Segmented & Multi-Row Reductions — LayerNorm, BatchNorm, Per-Head": {
        "description": (
            "Implement the full reduction patterns needed for ML normalisations: "
            "per-row reduction (LayerNorm), per-channel reduction (BatchNorm), "
            "per-head reduction (multi-head attention). Show warp-per-row assignment "
            "for small rows, block-per-row for large rows, and the Welford online "
            "variance reduction. Benchmark against reference NumPy implementations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  SEGMENTED REDUCTIONS — LayerNorm, BatchNorm & Per-Head Softmax")
print("=" * 68)
print()

WARP_SIZE  = 32
FP32_BYTES = 4
EPS        = 1e-5

# ─────────────────────────────────────────────────────────────────────
# Warp-level primitives
# ─────────────────────────────────────────────────────────────────────

def shfl_down(vals, delta, width=WARP_SIZE):
    result = np.array(vals, dtype=float)
    for lane in range(len(vals)):
        pos = lane % width
        if pos + delta < width:
            src = lane + delta
            if src < len(vals):
                result[lane] = vals[src]
    return result

def warp_reduce_sum(vals):
    acc = np.array(vals, dtype=float)
    for offset in [16, 8, 4, 2, 1]:
        peer = shfl_down(acc, offset)
        for i in range(WARP_SIZE - offset):
            acc[i] += peer[i]
    return acc[0]

def warp_reduce_max(vals):
    acc = np.array(vals, dtype=float)
    for offset in [16, 8, 4, 2, 1]:
        peer = shfl_down(acc, offset)
        for i in range(WARP_SIZE - offset):
            acc[i] = max(acc[i], peer[i])
    return acc[0]

def welford_merge(cnt_a, mean_a, M2_a, cnt_b, mean_b, M2_b):
    cnt = cnt_a + cnt_b
    if cnt == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean  = mean_a + delta * cnt_b / cnt
    M2    = M2_a + M2_b + delta * delta * cnt_a * cnt_b / cnt
    return cnt, mean, M2

np.random.seed(13)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: LayerNorm — per-row reduction (block per row)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — LayerNorm: Per-Row Reduction (Block per Row)")
print("━" * 68)
print()

def layernorm_block(x_row, gamma, beta, block_size=256):
    """
    Fused LayerNorm for one row using block-level Welford reduction.
    One thread block handles one row of x.

    Phase 1: each thread accumulates its Welford (cnt, mean, M2) via grid-stride
    Phase 2: warp reductions of (cnt, mean, M2) → merge via SMEM
    Phase 3: normalise each element in SMEM → write to HBM
    """
    N = len(x_row)
    # Phase 1: per-thread Welford over assigned elements (grid-stride)
    thread_cnts  = np.zeros(block_size)
    thread_means = np.zeros(block_size)
    thread_M2s   = np.zeros(block_size)

    for tid in range(block_size):
        cnt, mean, M2 = 0, 0.0, 0.0
        for i in range(tid, N, block_size):
            cnt += 1
            delta = x_row[i] - mean
            mean += delta / cnt
            delta2 = x_row[i] - mean
            M2 += delta * delta2
        thread_cnts[tid]  = cnt
        thread_means[tid] = mean
        thread_M2s[tid]   = M2

    # Phase 2: warp-level Welford merge (tree reduction over threads)
    # Simulate: reduce all threads to a single (cnt, mean, M2)
    final_cnt, final_mean, final_M2 = 0, 0.0, 0.0
    for tid in range(block_size):
        final_cnt, final_mean, final_M2 = welford_merge(
            final_cnt, final_mean, final_M2,
            thread_cnts[tid], thread_means[tid], thread_M2s[tid]
        )

    variance = final_M2 / final_cnt
    std      = math.sqrt(variance + EPS)

    # Phase 3: normalise
    output = (x_row - final_mean) / std * gamma + beta
    return output, final_mean, variance

# Test
N_row  = 1024
x_row  = np.random.randn(N_row).astype(np.float32)
gamma  = np.ones(N_row, dtype=np.float32)
beta   = np.zeros(N_row, dtype=np.float32)

out, mean_est, var_est = layernorm_block(x_row, gamma, beta)

# Reference
ref_mean = x_row.mean()
ref_var  = x_row.var()
ref_out  = (x_row - ref_mean) / np.sqrt(ref_var + EPS)

print(f"  Row length N = {N_row}")
print(f"  Block size   = 256 threads, each handles N/256 = {N_row//256} elements via grid-stride")
print()
print(f"  Welford stats:  mean={mean_est:.6f}  var={var_est:.6f}")
print(f"  Reference:      mean={ref_mean:.6f}  var={ref_var:.6f}")
print(f"  Mean error:     {abs(mean_est - ref_mean):.2e}")
print(f"  Var  error:     {abs(var_est - ref_var):.2e}")
print(f"  Output max err: {np.max(np.abs(out - ref_out)):.2e}  "
      f"{'✅' if np.max(np.abs(out - ref_out)) < 1e-4 else '❌'}")
print()

# HBM cost model
print(f"  HBM traffic (N={N_row}, FP32):")
print(f"    Naïve 3-pass:  {3*N_row*FP32_BYTES} bytes read  + {N_row*FP32_BYTES} write")
print(f"    SMEM-fused 2-pass: {N_row*FP32_BYTES} bytes read  + {N_row*FP32_BYTES} write")
print(f"    Saving: {N_row*FP32_BYTES*2} bytes (33% less HBM traffic)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Warp-per-row assignment (small rows: N ≤ 32)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Warp-per-Row: Reduction for Small Rows (N ≤ 32)")
print("━" * 68)
print()

def warp_per_row_softmax(matrix, n_rows, row_width):
    """
    Assign one warp to each row.
    For row_width ≤ 32: fits in one warp → no SMEM needed.
    All lanes handle one element of their row.
    """
    assert row_width <= WARP_SIZE
    results = np.zeros((n_rows, row_width))

    for row in range(n_rows):
        x = matrix[row]   # row_width elements, one per lane
        # Pad to warp size
        padded = np.zeros(WARP_SIZE)
        padded[:row_width] = x

        # Warp max (5 shfl_down, width=row_width)
        m_arr = padded.copy()
        for offset in [o for o in [16,8,4,2,1] if o < row_width]:
            peer = shfl_down(m_arr, offset, width=row_width)
            for lane in range(WARP_SIZE - offset):
                if lane % row_width + offset < row_width:
                    m_arr[lane] = max(m_arr[lane], peer[lane])

        # Broadcast max within each group of row_width
        m_vals = np.zeros(WARP_SIZE)
        for lane in range(WARP_SIZE):
            group_start = (lane // row_width) * row_width
            m_vals[lane] = m_arr[group_start]

        # Exp + warp sum
        exp_vals = np.exp(padded[:row_width] - m_vals[:row_width])
        s = warp_reduce_sum(
            np.concatenate([exp_vals, np.zeros(WARP_SIZE - row_width)])
        )

        results[row] = exp_vals / s

    return results

row_width  = 8
n_rows_t   = 4
matrix     = np.random.randn(n_rows_t, row_width) * 2

sm_warp    = warp_per_row_softmax(matrix, n_rows_t, row_width)
sm_ref     = np.array([
    np.exp(r - r.max()) / np.exp(r - r.max()).sum()
    for r in matrix
])

print(f"  Matrix: {n_rows_t} rows × {row_width} cols (one warp handles one row)")
print()
for row in range(n_rows_t):
    print(f"  Row {row}: logits = {matrix[row].round(3).tolist()}")
    print(f"         probs  = {sm_warp[row].round(4).tolist()}")
    print(f"         ref    = {sm_ref[row].round(4).tolist()}")
    err = np.max(np.abs(sm_warp[row] - sm_ref[row]))
    print(f"         err    = {err:.2e}  {'✅' if err < 1e-12 else '❌'}")
    print()

print(f"  Strategy choice by row_width:")
print(f"  {'row_width':>12}  {'Strategy':>30}  {'SMEM needed?':>13}  {'Warps per row'}")
print("  " + "─" * 64)
strategies = [
    (1,    "Single thread (trivial)",         "no",  "1 (1 lane)"),
    (8,    "Warp-per-row, width=8",           "no",  "1 (8 lanes)"),
    (16,   "Warp-per-row, width=16",          "no",  "1 (16 lanes)"),
    (32,   "Warp-per-row, full warp",         "no",  "1 (32 lanes)"),
    (64,   "2 warps per row, merge via SMEM", "yes", "2"),
    (1024, "Block-per-row (grid-stride)",     "yes", "32 (full block)"),
    (65536,"Multi-block per row",             "yes", "256+ warps"),
]
for rw, strat, smem, warps in strategies:
    print(f"  {rw:>12}  {strat:>30}  {smem:>13}  {warps}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Multi-head attention — independent per-head reductions
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Multi-Head Attention: Independent Per-Head Softmax")
print("━" * 68)
print()

def mha_softmax(attention_logits, n_heads, seq_len):
    """
    attention_logits: (n_heads, seq_len) — one row per attention head
    Each head computes its own softmax independently.
    Strategy: one block per head, blocks run in parallel across the grid.
    """
    probs = np.zeros_like(attention_logits)
    for h in range(n_heads):
        x = attention_logits[h]
        m = x.max()
        e = np.exp(x - m)
        probs[h] = e / e.sum()
    return probs

n_heads   = 16
seq_len_h = 128
logits_mha = np.random.randn(n_heads, seq_len_h).astype(np.float32) * 3

probs_mha  = mha_softmax(logits_mha, n_heads, seq_len_h)

print(f"  n_heads={n_heads}, seq_len={seq_len_h}")
print(f"  Grid assignment: {n_heads} blocks (one per head), {seq_len_h} elements each")
print()
print(f"  Verification (sum of each head's probabilities = 1.0):")
for h in range(n_heads):
    s = probs_mha[h].sum()
    print(f"    Head {h:2d}: sum = {s:.8f}  {'✅' if abs(s-1.0)<1e-6 else '❌'}")

print()
print(f"  Total HBM: {n_heads * seq_len_h * FP32_BYTES * 3} bytes "
      f"= {n_heads}×{seq_len_h}×{FP32_BYTES}×3 (3 passes)")
print(f"  Grid-level parallelism: all {n_heads} heads compute simultaneously.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: BatchNorm — per-channel reduction across (N, H, W)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — BatchNorm: Per-Channel Reduction Across N×H×W")
print("━" * 68)
print()

def batchnorm_forward(x, gamma, beta):
    """
    x: (N, C, H, W)
    For each channel c: reduce over (N, H, W) to get mean_c, var_c.
    Then normalise.

    GPU mapping:
        One block per channel.
        Each block reduces N*H*W elements via grid-stride.
    """
    N, C, H, W = x.shape
    mean   = np.zeros(C)
    var    = np.zeros(C)
    output = np.zeros_like(x)

    for c in range(C):
        channel_vals = x[:, c, :, :].reshape(-1)  # N*H*W elements
        mean[c] = channel_vals.mean()
        var[c]  = channel_vals.var()
        output[:, c, :, :] = (
            (x[:, c, :, :] - mean[c]) / math.sqrt(var[c] + EPS)
        ) * gamma[c] + beta[c]

    return output, mean, var

N_bn, C_bn, H_bn, W_bn = 4, 3, 8, 8
x_bn    = np.random.randn(N_bn, C_bn, H_bn, W_bn).astype(np.float32)
gamma_bn = np.ones(C_bn, dtype=np.float32)
beta_bn  = np.zeros(C_bn, dtype=np.float32)

out_bn, mean_bn, var_bn = batchnorm_forward(x_bn, gamma_bn, beta_bn)

print(f"  Input: N={N_bn}, C={C_bn}, H={H_bn}, W={W_bn}")
print(f"  Elements per channel: {N_bn*H_bn*W_bn}")
print()
for c in range(C_bn):
    ref_mean = x_bn[:, c, :, :].mean()
    ref_var  = x_bn[:, c, :, :].var()
    out_mean = out_bn[:, c, :, :].mean()
    out_std  = out_bn[:, c, :, :].std()
    print(f"  Channel {c}: input mean={ref_mean:.4f} var={ref_var:.4f}  "
          f"→ output mean≈{out_mean:.4f} std≈{out_std:.4f}  "
          f"{'✅' if abs(out_mean) < 0.01 else '❌'}")

print()
elem_per_ch = N_bn * H_bn * W_bn
print(f"  Grid: {C_bn} blocks (one per channel)")
print(f"  Each block: grid-stride over {elem_per_ch} elements → hybrid warp+SMEM reduce")
print()
print(f"  LayerNorm vs BatchNorm reduction axis:")
print(f"    LayerNorm:  reduce over FEATURE dim (last axis) — one row at a time")
print(f"    BatchNorm:  reduce over BATCH+SPATIAL dims — one channel at a time")
print(f"    RMSNorm:    reduce over FEATURE dim, mean=0 assumed — just variance")
print(f"  All three use the same block reduction kernel; only axes differ.")
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