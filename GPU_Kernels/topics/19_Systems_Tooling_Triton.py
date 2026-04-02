"""
Triton — Language Basics, Blocked MatMul, Fused Softmax & Autotuning
=====================================================================

Triton is an open-source DSL (domain-specific language) and compiler for
writing high-performance GPU kernels in Python. Where CUDA requires the
programmer to reason about individual threads and warps, Triton raises the
abstraction level to BLOCKS — contiguous tiles of data processed together.
The Triton compiler then handles thread layout, shared memory allocation,
vectorisation, and instruction selection automatically.

Three fundamental observations motivate Triton:

    BLOCKS MATCH THE HARDWARE: GPU SMs execute instructions on warps (32
    threads), and the optimal computation granularity for tensor cores is
    a 16×16 or larger tile. Triton's block pointer abstraction expresses
    computation at this natural granularity without manual thread indexing.

    SMEM IS IMPLICIT: the most complex part of hand-written CUDA kernels is
    staging data through shared memory (SMEM) to avoid redundant HBM reads.
    Triton's compiler analyses data reuse and inserts SMEM automatically,
    guided by the programmer's block tiling structure.

    JIT COMPILATION WITH AUTOTUNING: Triton kernels are compiled at call time
    with tl.constexpr parameters that control tile sizes. Autotuning exhaustively
    searches over tile configurations and caches the best result per input shape.
    This is fundamentally different from hand-tuned CUDA — optimal tile sizes
    are discovered at runtime, not hard-coded.

The result: Triton kernels for matrix multiplication, attention, and layer
norm routinely achieve 80–100% of cuBLAS/cuDNN throughput with 10× less code,
and for novel fused operations (custom attention variants, sparse activations)
they outperform CUDA by eliminating HBM round-trips that would be required
by separate CUDA kernel calls.

"""

import textwrap, re, math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Triton — Language Basics, Blocked MatMul, Fused Softmax & Autotuning"
DISPLAY_NAME = "19 · Triton"
ICON         = "⚡"
SUBTITLE     = "Block Pointers · JIT · tl.load/store · Masked Ops · Autotuning · tl.constexpr"

THEORY = """

##### PART 1 — THE TRITON PROGRAMMING MODEL: BLOCKS, PROGRAMS, AND TENSORS

### The Program and Block Abstraction

    In CUDA: the programmer writes code for ONE THREAD. Thread index
    calculations map that thread to its data element. The grid of threads
    executes the same code in parallel — SIMT (Single Instruction, Multiple Threads).

    In Triton: the programmer writes code for ONE PROGRAM INSTANCE — a block
    of threads working cooperatively on a TILE of data. The "thread index"
    is replaced by a "program ID" identifying which tile this block handles.

    TRITON EXECUTION MODEL:
        Grid  = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))   — one program per tile
        Block = BLOCK_M × BLOCK_N threads (managed by Triton/compiler)
        Each program handles tile (pid_m, pid_n) of the output matrix.

    KEY DIFFERENCE from CUDA:
        CUDA: "I am thread (tx, ty, tz). My data is A[tx, ty]."
        Triton: "I am program pid. My block of data is A[pid*BLOCK:pid*BLOCK+BLOCK, :]."

### The tl.arange and Block Tensor

    Within a Triton program, all computation operates on BLOCK TENSORS —
    multi-dimensional arrays of fixed compile-time shape [BLOCK_M, BLOCK_N].
    These are NOT Python arrays; they represent the simultaneous operation
    of all threads in the block.

    FUNDAMENTAL PRIMITIVES:
        tl.arange(0, N):   Returns a block tensor [0, 1, 2, ..., N-1].
                           N must be a power of 2 and a tl.constexpr.
                           Used to compute per-element indices within a block.

        tl.load(ptr + offsets, mask, other):
                           Loads a block tensor from HBM.
                           ptr: base pointer, offsets: block of int indices.
                           mask: bool block — masked elements load 'other' instead.
                           Returns: a block tensor of the loaded values.

        tl.store(ptr + offsets, value, mask):
                           Stores a block tensor to HBM.
                           Elements where mask is False are NOT written.

        tl.dot(A, B):      Block matrix multiplication of two block tensors.
                           Calls the GPU's tensor core WMMA instruction.
                           Both A and B must be 2D block tensors.
                           Accumulates in FP32 even for FP16 inputs (like TC MMA).

        tl.sum(x, axis):   Reduces a block tensor along one axis.
        tl.max(x, axis):   Max reduction along one axis.
        tl.exp(x):         Element-wise exp, applied to all elements in block.
        tl.zeros(shape):   Returns a zero-filled block tensor.

### The @triton.jit Decorator and Kernel Launch

    KERNEL DEFINITION:
        @triton.jit
        def my_kernel(input_ptr, output_ptr, N,
                      BLOCK_SIZE: tl.constexpr):
            pid   = tl.program_id(0)
            offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask  = offs < N
            x     = tl.load(input_ptr + offs, mask=mask, other=0.0)
            y     = x * x
            tl.store(output_ptr + offs, y, mask=mask)

    KERNEL LAUNCH:
        BLOCK_SIZE = 1024
        grid = (cdiv(N, BLOCK_SIZE),)   # number of program instances
        my_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

    tl.constexpr parameters are COMPILE-TIME CONSTANTS — the JIT compiler
    specialises the code for each distinct (BLOCK_SIZE, dtype, ...) combination.
    Different BLOCK_SIZE values produce different compiled kernels.

### How Triton Compiles to GPU Code

    COMPILATION PIPELINE:
        Python + @triton.jit → Triton IR (SSA form)
        → Triton GPU dialect (MLIR) → LLVM IR → PTX → SASS

    TRITON IR: Triton's intermediate representation captures block semantics.
        tl.load becomes a block load instruction in IR.
        tl.dot becomes a block matrix multiply instruction.
        The IR preserves the tile structure for downstream optimisation.

    GPU DIALECT LOWERING:
        Block tensors are lowered to warp-level operations.
        tl.load is lowered to async loads with cp.async (Ampere+).
        tl.dot is lowered to wgmma (H100) or mma.sync.aligned (older GPUs).
        SMEM tiling is inserted automatically by the lowering pass.

    JIT CACHE: compiled kernels are cached in ~/.triton/cache/.
        First call: compile (~100ms for complex kernels).
        Subsequent calls: load from disk cache (< 1ms).
        Cache keyed by: kernel source, input dtypes, constexpr values, GPU arch.


##### PART 2 — MEMORY ACCESS PATTERNS: LOADS, STORES, AND MASKING

### tl.load and the Offset Pattern

    tl.load is the heart of every Triton kernel. Its usage follows a consistent
    pattern: compute offsets from a base pointer, load with masking.

    1D LOAD:
        offs = pid * BLOCK + tl.arange(0, BLOCK)   # 1D offset tensor [BLOCK]
        x    = tl.load(ptr + offs, mask=offs < N, other=0.0)
        # Loads BLOCK consecutive elements starting at pid * BLOCK.
        # Elements where offs >= N are filled with 0.0 (not read from memory).

    2D LOAD (for matrix tiles):
        row_offs = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]  # [BM, 1]
        col_offs = tl.arange(0, BLOCK_K)[None, :]                       # [1,  BK]
        offs_2d  = row_offs * stride_a + col_offs * stride_b            # [BM, BK]
        mask_2d  = (row_offs < M) & (col_offs < K)                      # [BM, BK]
        tile_a   = tl.load(ptr_a + offs_2d, mask=mask_2d, other=0.0)

    STRIDE PARAMETERS: for non-contiguous matrices (e.g., after a transpose),
    strides are passed as kernel arguments.
    ptr_a[i*stride_a + j*stride_b] = A[i, j] for any layout.
    Contiguous row-major: stride_a = K, stride_b = 1.

### Why Masking Is Critical

    In practice, M and N are rarely divisible by BLOCK_M and BLOCK_N.
    Without masking: the last tile would read/write out of bounds → SEGFAULT.
    With masking: out-of-bounds accesses are suppressed by the hardware.

    MASK CONSTRUCTION for a tile at (pid_m, pid_n):
        rows_valid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) < M
        cols_valid = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) < N
        # For 2D tile: broadcast to get [BLOCK_M, BLOCK_N] mask
        mask = rows_valid[:, None] & cols_valid[None, :]

    The GPU hardware converts masked loads to predicated instructions.
    No actual memory access occurs for masked-out elements.

### Block Pointer API (Triton ≥ 2.1): tl.make_block_ptr

    Modern Triton introduces block pointers — a cleaner API for 2D tile access:

        A_ptr = tl.make_block_ptr(
            base=a_ptr,                    # base pointer
            shape=(M, K),                  # full tensor shape
            strides=(stride_am, stride_ak),# strides per dimension
            offsets=(pid_m*BLOCK_M, 0),    # offset to the start of this tile
            block_shape=(BLOCK_M, BLOCK_K),# size of the block
            order=(1, 0),                  # inner dimension first (row-major)
        )
        tile_a = tl.load(A_ptr)  # no explicit offset arithmetic needed!

    ADVANCING the block pointer across the K loop:
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))  # move right by BLOCK_K

    Block pointers enable the compiler to emit more optimal async load sequences
    and better model the access pattern for prefetching.


##### PART 3 — BLOCKED MATRIX MULTIPLICATION: THE CANONICAL TRITON KERNEL

### Why Tiling Is Necessary

    A naïve matrix multiply reads each element of A once per column of B,
    and each element of B once per row of A — O(M × K × N) total reads,
    much larger than the matrix sizes themselves.

    ARITHMETIC INTENSITY of naïve matmul:
        FLOPs: 2MKN
        Bytes: (MK + KN + MN) × 4 = O(MKN) for large equal dims
        AI ≈ 2MKN / (3MKN × 4) = 0.16 FLOPs/Byte — memory-bound!

    TILED matmul: load a BLOCK_M × BLOCK_K tile of A and BLOCK_K × BLOCK_N
    tile of B into SMEM. Compute the BLOCK_M × BLOCK_N partial result.
    Each A tile is reused BLOCK_N times; each B tile is reused BLOCK_M times.

    ARITHMETIC INTENSITY of tiled matmul:
        FLOPs: 2 × BLOCK_M × BLOCK_K × BLOCK_N per tile load
        Bytes: (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × 4 per tile load
        AI ≈ 2 × BLOCK_K / (BLOCK_M^-1 + BLOCK_N^-1) / 4
        For BLOCK_M = BLOCK_N = BLOCK_K = 128: AI ≈ 64 FLOPs/Byte → compute-bound!

### The Blocked MatMul Algorithm in Triton

    C = A @ B  where A: [M, K], B: [K, N], C: [M, N].

    OUTER GRID: one program per (BLOCK_M, BLOCK_N) tile of C.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        GRID = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))

    ACCUMULATOR: each program accumulates into a [BLOCK_M, BLOCK_N] FP32 block.
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    INNER LOOP over K (stride BLOCK_K):
        for k in range(0, K, BLOCK_K):
            a_tile = tl.load(A_ptr + ...)   # [BLOCK_M, BLOCK_K]
            b_tile = tl.load(B_ptr + ...)   # [BLOCK_K, BLOCK_N]
            acc   += tl.dot(a_tile, b_tile) # tensor core matmul, += to accumulator

    WRITE OUTPUT:
        c_tile = acc.to(tl.float16)        # downcast from FP32 accumulator
        tl.store(C_ptr + ..., c_tile)

    KEY DETAILS:
        tl.dot ALWAYS accumulates in FP32 (like hardware tensor cores).
        The FP16 inputs are upcasted to FP32 during the MMA instruction.
        The accumulator in FP32 avoids precision loss across K iterations.

### SW PIPELINE: Hiding HBM Latency with Async Prefetch

    Without pipelining: load A tile, load B tile, wait, compute, repeat.
    With SW pipeline (num_stages=3):
        Iteration i: compute on tiles (i)
        Iteration i: prefetch tiles (i+1) asynchronously via cp.async

    Triton supports this via the num_stages parameter to @triton.jit:
        @triton.jit(num_warps=8, num_stages=4)
        def matmul_kernel(...):  # compiler inserts cp.async + barriers

    EFFECT: HBM latency (~200 cycles) is hidden by compute on the previous tile.
    The GPU's memory subsystem and compute engines overlap.
    For H100 with WGMMA: software pipeline is critical — wgmma has long latency.

### Swizzle and SMEM Bank Conflicts

    When Triton loads a tile into SMEM, it must arrange the data to avoid
    SMEM bank conflicts during the tl.dot operation.

    SMEM has 32 banks. If 32 threads in a warp all access the same bank
    (e.g., all reading column 0 of a column-major tile), they serialise.

    Triton's compiler inserts SWIZZLE instructions automatically for block
    pointer loads. For manual offset loads, the user can enable swizzling:
        A_ptr = tl.make_block_ptr(..., order=(1, 0))  # row-major → natural swizzle
    The order parameter tells the compiler the memory layout, enabling
    optimal bank-conflict-free SMEM organisation.


##### PART 4 — FUSED SOFTMAX: ONLINE ALGORITHM IN TRITON

### Why Fuse Softmax

    Naïve softmax (3 separate kernel passes):
        Pass 1: read X, compute row_max. Write row_max. [1 HBM read + write]
        Pass 2: read X, read row_max, compute exp(X-max). Write exp. [2 reads + write]
        Pass 3: read exp, compute sum. Write sum. Divide. Write output. [3 reads + write]
        TOTAL: 5× the input size in HBM reads + writes.

    Fused softmax (1 pass):
        Read X ONCE per row. Compute max, exp, sum, division in SMEM. Write output.
        TOTAL: 2× (one read + one write) — 2.5× less HBM traffic.

    Triton makes this fusion trivial: one kernel, one pass over each row.

### Online Softmax Algorithm

    The challenge: computing row_max requires a full pass before computing exp.
    SOLUTION: the online algorithm (same as online softmax from Module 30):

        m = -inf  (running maximum)
        d = 0.0   (running denominator, scaled)
        o = zeros (running output)

        for block in tl.range(0, N, BLOCK_N):
            x_block   = tl.load(x_ptr + ...)
            m_new     = max(m, tl.max(x_block, axis=0))
            d         = d * exp(m - m_new) + sum(exp(x_block - m_new))
            m         = m_new

        # Second pass: normalise (or integrate into first pass for fusion)
        for block in tl.range(0, N, BLOCK_N):
            x_block  = tl.load(...)
            y_block  = exp(x_block - m) / d
            tl.store(...)

    For rows that fit in SMEM in one block (N ≤ BLOCK_N):
        Load the entire row into a block tensor.
        Compute max with tl.max(x, axis=0).
        Compute exp of shifted values.
        Compute sum with tl.sum(exp_x, axis=0).
        Divide and store.
        ONE PASS. Perfect fusion.

### Triton Fused Softmax Kernel Structure

    @triton.jit
    def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                       n_cols, BLOCK_SIZE: tl.constexpr):
        row_idx  = tl.program_id(0)   # one program per row
        row_start = row_idx * input_row_stride
        col_offs  = tl.arange(0, BLOCK_SIZE)
        mask      = col_offs < n_cols

        # LOAD: read entire row (or tile) into registers
        x = tl.load(input_ptr + row_start + col_offs, mask=mask, other=-float('inf'))

        # COMPUTE: all in registers, zero HBM reads
        x_max   = tl.max(x, axis=0)   # scalar: max over the block
        x_shift = x - x_max           # broadcast: subtract scalar from block
        exp_x   = tl.exp(x_shift)
        exp_sum = tl.sum(exp_x, axis=0)
        result  = exp_x / exp_sum

        # STORE: write output
        tl.store(output_ptr + row_idx * output_row_stride + col_offs,
                 result, mask=mask)

    GRID: (n_rows,) — one program per row.

### When BLOCK_SIZE Must Be Larger Than n_cols

    If n_cols is not a power of 2, we must pad BLOCK_SIZE up to the next power.
    The mask prevents reading/writing the padded elements.
    The padding with -inf ensures masked elements don't affect the max.
    The padding with 0 in exp_x ensures masked elements don't affect the sum.

    In practice: BLOCK_SIZE is the next power-of-2 ≥ n_cols.
    Triton's autotuning can search over BLOCK_SIZE options.


##### PART 5 — AUTOTUNING: tl.constexpr AND THE TUNING SPACE

### Why Autotuning Is Necessary

    The optimal BLOCK_M, BLOCK_N, BLOCK_K, num_warps, and num_stages values
    depend on:
        - The GPU architecture (A100 vs H100 vs V100).
        - The matrix dimensions M, N, K.
        - The dtype (FP16 vs BF16 vs FP8).
        - The memory layout (contiguous vs strided).

    For example, on H100:
        Small matrices (M=N=K=64): BLOCK_M=BLOCK_N=32, num_warps=2 is optimal.
        Large matrices (M=N=K=8192): BLOCK_M=BLOCK_N=128, num_warps=8 is optimal.
    The same kernel structure with different tl.constexpr values produces
    entirely different compiled code and entirely different performance.

### The @triton.autotune Decorator

    DEFINING THE TUNING SPACE:
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,
                               'num_stages': 3, 'num_warps': 8}, pre_hook=...),
                triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,
                               'num_stages': 4, 'num_warps': 4}),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
                               'num_stages': 4, 'num_warps': 4}),
                # ... many more configurations
            ],
            key=['M', 'N', 'K'],   # retune when these change
        )
        @triton.jit
        def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
            ...

    HOW IT WORKS:
        1. On the FIRST call with a new (M, N, K) combination:
           Triton compiles and benchmarks ALL configurations.
           Picks the fastest (measured with CUDA events, averaged over 100 reps).
           Caches the winner for this (M, N, K, dtype, device) combination.

        2. On subsequent calls with the same key:
           Loads winner from cache. Instant.

    TUNING OVERHEAD: up to ~5-30 seconds for large config spaces.
    TYPICAL PRODUCTION STRATEGY: pre-warm the tuning cache at server startup.

### tl.constexpr: Compile-Time Specialisation

    Parameters annotated as tl.constexpr are "known at compile time":
        - They become constants in the JIT-compiled PTX.
        - Loop trip counts (range(0, K, BLOCK_K)) are unrolled if BLOCK_K is known.
        - Tiling factors become array shapes (allowed by GPU compiler).
        - Conditional branches on tl.constexpr are eliminated at compile time.

    NON-constexpr PARAMETERS: passed as kernel arguments at launch time.
        - Can vary between calls without recompilation.
        - Loop bounds (M, N, K) are typically non-constexpr.

    WHY THIS MATTERS:
        @triton.jit
        def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
            offs = tl.arange(0, BLOCK_SIZE)  # MUST be constexpr; shape is compile-time

    tl.arange(0, N) would fail if N is not tl.constexpr — the block tensor shape
    must be known at compile time (like CUDA's shared memory size).

### num_warps and num_stages

    num_warps: the number of warps per thread block.
        Controls register pressure (more warps = less registers per warp = lower occupancy).
        More warps = more latency hiding.
        For memory-bound kernels: 4 warps often optimal.
        For compute-bound (matmul): 8-16 warps.

    num_stages: the depth of the software pipeline.
        0 or 1: no pipeline (load → compute → load → compute).
        2: one stage of prefetch (load i+1 while computing i).
        3-4: deeper pipeline, more SMEM usage (holding multiple tiles).
        Higher stages: require more SMEM per block → fewer blocks per SM.
        Optimal balance depends on HBM latency and compute throughput.

### Diagnostic: Triton's auto-tuning Output

    Triton can dump the tuning process:
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        Best config for GEMM (M=8192, N=8192, K=8192):
            BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=3
            Time: 2.34 ms  (vs cuBLAS: 2.41 ms — 97% of cuBLAS performance)


##### PART 6 — ADVANCED TRITON PATTERNS

### Atomic Operations

    Triton supports atomic updates for reduction into global memory:
        tl.atomic_add(ptr + offset, value)
        tl.atomic_max(ptr + offset, value)
        tl.atomic_min(ptr + offset, value)
        tl.atomic_cas(ptr, cmp, val)  # compare-and-swap

    USE CASE: computing row-wise statistics when rows don't fit in one block.
    Multiple programs accumulate into the same output location atomically.

### Reduction Across the Block: tl.reduce

    For reductions more complex than max/sum:
        result = tl.reduce(x, axis=0, combine_fn=combine_fn)
    where combine_fn is a user-defined Python function called with pairs.
    This is used for: custom gated reductions, ArgMax (value + index).

### Persistent Kernels in Triton

    Triton supports a "persistent kernel" pattern via triton.language.core._persistent:
        while True:
            task_id = atomic_claim_task()
            if task_id >= total_tasks: break
            process_task(task_id)

    The @persistent decorator (Triton 3.0+) handles the occupancy-saturating
    launch and work queue automatically — equivalent to the persistent kernel
    model from Module 33, but in Python.

### Flash Attention in Triton

    Flash Attention 2's inner loop is a canonical Triton kernel:
        @triton.jit
        def _fwd_kernel(Q, K, V, sm_scale, L, Out, ...):
            q_block = tl.load(Q + ...)
            m_i     = -float('inf')
            l_i     = 0.0
            acc     = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            for kv_start in tl.range(0, SEQ_LEN, BLOCK_N):
                k_block = tl.load(K + ...)
                v_block = tl.load(V + ...)
                qk      = tl.dot(q_block, tl.trans(k_block)) * sm_scale
                m_ij    = tl.max(qk, axis=1)
                p       = tl.exp(qk - m_ij[:, None])
                alpha   = tl.exp(m_i - m_ij)
                acc     = acc * alpha[:, None] + tl.dot(p, v_block)
                l_i     = l_i * alpha + tl.sum(p, axis=1)
                m_i     = m_ij
            acc /= l_i[:, None]
            tl.store(Out + ..., acc)

    This is EXACTLY the Flash Attention online softmax algorithm (Module 30)
    expressed in Triton — no intermediate N×N materialisation, all in SMEM.


##### PART 7 — TRITON vs CUDA vs TORCH.COMPILE: WHEN TO USE EACH

### Decision Matrix

    PYTORCH EAGER (no kernel):
        Pro: zero engineering cost, full flexibility.
        Con: multiple kernel launches, intermediate tensors, no fusion.
        Use for: prototyping, operations with irregular shapes.

    TORCH.COMPILE (torch.inductor):
        Pro: automatic fusion of elementwise chains, minimal code change.
        Con: limited to vertical fusion (no reduction → elementwise fusion
             across a GEMM boundary). Cannot fuse custom attention variants.
        Use for: training loops where standard ops dominate.

    TRITON KERNEL:
        Pro: full control over tiling, fusion, memory layout, precision.
              Can express any computation in terms of blocks.
              Autotuning matches or exceeds cuBLAS for many shapes.
        Con: requires understanding the block model and Triton's semantics.
             JIT compilation latency on first call.
             Debugging is harder (no printf in device code without tricks).
        Use for: custom attention, novel activations, fused ops across GEMM.

    CUDA KERNEL (C++):
        Pro: maximum control over every GPU instruction.
             Access to hardware-specific intrinsics (wgmma, TMA, etc.)
        Con: ~10× more code than Triton for equivalent operations.
             Architecture-specific (must rewrite for H100 vs A100 TC).
             No built-in autotuning.
        Use for: latency-critical operations where every cycle matters,
                 hardware features not yet exposed by Triton.

### Triton Performance Relative to cuBLAS

    Triton matmul vs cuBLAS (A100, FP16):
        Large square (M=N=K=8192): Triton ≈ 98% of cuBLAS.
        Tall/thin (M=8192, N=128, K=8192): Triton ≈ 85% of cuBLAS.
        Very small (M=N=K=64): Triton ≈ 70% (cuBLAS has hardcoded small-N kernels).

    Triton fused softmax vs PyTorch:
        Sequence length 2048: Triton 2.3× faster (avoids 2 HBM round-trips).
        Sequence length 128: Triton 1.4× faster (launch overhead less significant).

    Triton Flash Attention vs PyTorch's scaled_dot_product_attention:
        H100 BF16, context 4096: Triton ≈ 95% of SDPA (which uses cuDNN Flash-Attn).
        H100 BF16, context 1024: Triton ≈ 90% (cuDNN has more static optimisations).

"""

OPERATIONS = {

    "1 · Triton Language Basics — Blocks, Loads, Stores & Masking": {
        "description": (
            "Implement core Triton primitives in pure Python/NumPy to show exactly "
            "what each operation does at the data level. Simulate tl.arange, tl.load "
            "with masking, tl.store, tl.sum, and tl.max. Build a 1D elementwise "
            "kernel (squared values) and a row-reduction kernel (row sum). Show "
            "the program_id decomposition for a 2D grid. Demonstrate why masks are "
            "critical for non-power-of-2 tensor sizes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  TRITON LANGUAGE BASICS — Blocks, Loads, Stores & Masking")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Simulate Triton primitives in NumPy
# ─────────────────────────────────────────────────────────────────────

def tl_arange(start, stop):
    """tl.arange: returns a block tensor [start, start+1, ..., stop-1]."""
    return np.arange(start, stop)

def tl_load(ptr_array, offs, mask, other=0.0):
    """tl.load: load elements at ptr_array[offs] with masking."""
    result = np.full(len(offs), other, dtype=ptr_array.dtype)
    valid  = mask & (offs >= 0) & (offs < len(ptr_array))
    result[valid] = ptr_array[offs[valid]]
    return result

def tl_store(ptr_array, offs, values, mask):
    """tl.store: store values at ptr_array[offs] where mask is True."""
    valid = mask & (offs >= 0) & (offs < len(ptr_array))
    ptr_array[offs[valid]] = values[valid]

def cdiv(a, b):
    return (a + b - 1) // b


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: 1D kernel — elementwise square
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — 1D Elementwise Kernel: x² with Masking")
print("━" * 68)
print()

def square_kernel(x, BLOCK_SIZE=8):
    """
    Simulate: @triton.jit square_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr)
      pid  = tl.program_id(0)
      offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offs < N
      x    = tl.load(x_ptr + offs, mask=mask, other=0.)
      tl.store(y_ptr + offs, x * x, mask=mask)
    """
    N    = len(x)
    y    = np.zeros(N, dtype=x.dtype)
    n_programs = cdiv(N, BLOCK_SIZE)

    print(f"  N={N}, BLOCK_SIZE={BLOCK_SIZE} → {n_programs} program instances (grid size)")
    print()
    print(f"  {'pid':>5}  {'offs':>30}  {'mask':>20}  {'loaded x':>30}")
    print("  " + "─" * 86)

    for pid in range(n_programs):
        offs = pid * BLOCK_SIZE + tl_arange(0, BLOCK_SIZE)
        mask = offs < N
        x_blk = tl_load(x, offs, mask, other=0.0)
        y_blk = x_blk * x_blk
        tl_store(y, offs, y_blk, mask)

        print(f"  {pid:>5}  {str(offs.tolist()):>30}  {str(mask.tolist()):>20}  "
              f"{str(x_blk.round(2).tolist()):>30}")

    return y

x_in  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
y_out = square_kernel(x_in, BLOCK_SIZE=4)

print()
print(f"  Input:    {x_in.tolist()}")
print(f"  Output:   {y_out.tolist()}")
print(f"  Expected: {(x_in**2).tolist()}")
print(f"  Correct:  {'✅' if np.allclose(y_out, x_in**2) else '❌'}")
print()
print("  Notice: pid=2 has offs=[8,9,10,11], mask=[T,T,F,F] → indices 10,11 NOT loaded.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: 2D program grid — tile assignment
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — 2D Grid Decomposition: Program ID to Tile Mapping")
print("━" * 68)
print()

M, N         = 10, 12
BLOCK_M      = 4
BLOCK_N      = 4
grid_m       = cdiv(M, BLOCK_M)
grid_n       = cdiv(N, BLOCK_N)

print(f"  Output matrix: {M}×{N}")
print(f"  Block size: {BLOCK_M}×{BLOCK_N}")
print(f"  Grid: {grid_m}×{grid_n} = {grid_m*grid_n} program instances")
print()
print(f"  {'pid_m':>6}  {'pid_n':>6}  {'row range':>12}  "
      f"{'col range':>12}  {'rows valid':>12}  {'cols valid'}")
print("  " + "─" * 62)

for pid_m in range(grid_m):
    for pid_n in range(grid_n):
        row_start = pid_m * BLOCK_M
        row_end   = min(row_start + BLOCK_M, M)
        col_start = pid_n * BLOCK_N
        col_end   = min(col_start + BLOCK_N, N)
        n_rows_valid = row_end - row_start
        n_cols_valid = col_end - col_start
        print(f"  {pid_m:>6}  {pid_n:>6}  [{row_start:>2},{row_end:>2})     "
              f"  [{col_start:>2},{col_end:>2})      "
              f"{n_rows_valid:>12}  {n_cols_valid}")

print()
print("  Masked tiles (on the boundary): load returns 0.0 for out-of-bounds elements.")
print("  This allows a single kernel to handle any (M, N) — no special-case code.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Row reduction kernel
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Row Sum Kernel: 2D Load + tl.sum Reduction")
print("━" * 68)
print()

def row_sum_kernel(matrix, BLOCK_N=8):
    """
    Simulate row-sum kernel: one program per row.
    pid = tl.program_id(0)  [row index]
    col_offs = tl.arange(0, BLOCK_N)
    mask     = col_offs < N_cols
    row      = tl.load(ptr + pid * stride + col_offs, mask, other=0.)
    result   = tl.sum(row, axis=0)
    tl.store(out_ptr + pid, result)
    """
    n_rows, n_cols = matrix.shape
    output         = np.zeros(n_rows, dtype=matrix.dtype)

    print(f"  Matrix: {n_rows}×{n_cols}, BLOCK_N={BLOCK_N}")
    print(f"  Grid: ({n_rows},) — one program per row")
    print()

    # Single-block version (BLOCK_N >= n_cols, one load per row)
    BLOCK_N_actual = 1 << math.ceil(math.log2(n_cols))   # next power of 2
    print(f"  Actual BLOCK_N (next pow2 ≥ {n_cols}): {BLOCK_N_actual}")
    print()

    for pid in range(n_rows):
        col_offs = tl_arange(0, BLOCK_N_actual)
        mask     = col_offs < n_cols
        row_data = tl_load(matrix[pid], col_offs, mask, other=0.0)
        output[pid] = row_data.sum()

    return output

M_r, N_r  = 5, 7
mat_r      = np.random.randint(1, 10, (M_r, N_r)).astype(np.float32)
row_sums   = row_sum_kernel(mat_r)
expected_r = mat_r.sum(axis=1)

print(f"  Matrix:")
for row in mat_r:
    print(f"    {row.astype(int).tolist()}")
print()
print(f"  Computed row sums: {row_sums.astype(int).tolist()}")
print(f"  Expected:          {expected_r.astype(int).tolist()}")
print(f"  Correct: {'✅' if np.allclose(row_sums, expected_r) else '❌'}")
print()
print("  Key: BLOCK_N padded to next power of 2. Masked load gives 0 for extra cols.")
print("  This is exactly how fused softmax handles non-power-of-2 sequence lengths.")
''',
    },

    "2 · Blocked MatMul — Tiling, Accumulator & Arithmetic Intensity": {
        "description": (
            "Implement the blocked matrix multiply algorithm in pure Python, "
            "simulating Triton's tiling strategy. Show how tiles of A and B "
            "are loaded and multiplied to produce tiles of C. Track HBM reads "
            "and compute the arithmetic intensity for different block sizes. "
            "Verify output against numpy reference. Show how the K-loop structure "
            "maps to SMEM reuse. Compare AI of tiled vs naïve matmul."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  BLOCKED MATMUL — Tiling, Accumulator & Arithmetic Intensity")
print("=" * 68)
print()

np.random.seed(7)


def cdiv(a, b):
    return (a + b - 1) // b


def blocked_matmul(A, B, BLOCK_M, BLOCK_N, BLOCK_K, verbose=False):
    """
    Simulate Triton's blocked matrix multiply.
    Returns C = A @ B.
    Tracks HBM reads to demonstrate bandwidth saving from tiling.
    """
    M, K_a = A.shape
    K_b, N = B.shape
    assert K_a == K_b, "Inner dimensions must match"
    K = K_a

    C = np.zeros((M, N), dtype=np.float64)
    hbm_reads = 0  # number of scalar values read from HBM

    n_blocks_m = cdiv(M, BLOCK_M)
    n_blocks_n = cdiv(N, BLOCK_N)
    n_blocks_k = cdiv(K, BLOCK_K)

    if verbose:
        print(f"  Grid: ({n_blocks_m}, {n_blocks_n}) programs")
        print(f"  K loop: {n_blocks_k} iterations of BLOCK_K={BLOCK_K}")
        print()

    for pid_m in range(n_blocks_m):
        for pid_n in range(n_blocks_n):
            # Output tile boundaries
            m_start = pid_m * BLOCK_M
            m_end   = min(m_start + BLOCK_M, M)
            n_start = pid_n * BLOCK_N
            n_end   = min(n_start + BLOCK_N, N)

            # FP32 accumulator (zero-initialised, never written to HBM until end)
            acc = np.zeros((m_end - m_start, n_end - n_start), dtype=np.float64)

            # K loop: load tiles of A and B, accumulate
            for k_idx in range(n_blocks_k):
                k_start = k_idx * BLOCK_K
                k_end   = min(k_start + BLOCK_K, K)

                # Load A tile from HBM [BLOCK_M, BLOCK_K]
                a_tile = A[m_start:m_end, k_start:k_end]
                hbm_reads += a_tile.size

                # Load B tile from HBM [BLOCK_K, BLOCK_N]
                b_tile = B[k_start:k_end, n_start:n_end]
                hbm_reads += b_tile.size

                # tl.dot: tensor core matmul, accumulated in FP32
                acc += a_tile @ b_tile   # stays in registers/SMEM

            # Write output tile ONCE to HBM
            C[m_start:m_end, n_start:n_end] = acc

    return C, hbm_reads


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness and tiling trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Blocked MatMul: Tiling Trace and Correctness")
print("━" * 68)
print()

M, K, N = 8, 12, 8
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

print(f"  A: {M}×{K}, B: {K}×{N}, C: {M}×{N}")
print()

for BLOCK_M, BLOCK_N, BLOCK_K in [(4, 4, 4), (8, 8, 4), (4, 4, 12)]:
    C_blocked, reads = blocked_matmul(A.astype(np.float64),
                                       B.astype(np.float64),
                                       BLOCK_M, BLOCK_N, BLOCK_K)
    C_ref = A.astype(np.float64) @ B.astype(np.float64)
    err   = np.abs(C_blocked - C_ref).max()
    naive_reads = M * K + K * N   # what naïve matmul would read (A+B once each)
    # Blocked reads A tiles: n_blocks_n times each A tile
    # (each A tile read once per output-N block iteration)
    blocks_n = cdiv(N, BLOCK_N)
    theory_reads = M * K * blocks_n + K * N * cdiv(M, BLOCK_M)
    print(f"  BLOCK={BLOCK_M}×{BLOCK_N}×{BLOCK_K}: "
          f"HBM reads={reads:,}, error={err:.2e}  "
          f"{'✅' if err < 1e-5 else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Arithmetic intensity vs block size
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Arithmetic Intensity: Tiled vs Naïve MatMul")
print("━" * 68)
print()

M_ai, K_ai, N_ai = 4096, 4096, 4096
FLOPS_TOTAL = 2 * M_ai * K_ai * N_ai

print(f"  Matrix: {M_ai}×{K_ai}×{N_ai}, FP16 (2 bytes)")
print(f"  Total FLOPs: {FLOPS_TOTAL/1e12:.2f} TFLOPs")
print()

# Naïve: each element of A read N times, each element of B read M times
naive_reads = M_ai * K_ai * N_ai + K_ai * N_ai * M_ai
naive_bytes = naive_reads * 2   # FP16
naive_ai    = FLOPS_TOTAL / naive_bytes

print(f"  Naïve matmul:")
print(f"    HBM reads: A read {N_ai}× + B read {M_ai}× = {naive_reads/1e9:.1f}B elements")
print(f"    HBM bytes: {naive_bytes/1e9:.1f} GB")
print(f"    AI:        {naive_ai:.2f} FLOPs/Byte  ← memory-bound!")
print()

print(f"  {'BLOCK':>18}  {'A reads':>12}  {'B reads':>12}  "
      f"{'Total GB':>10}  {'AI':>8}  {'Compute-bound?'}")
print("  " + "─" * 72)

H100_RIDGE = 989e12 / (3350e9)   # ~295 FLOPs/Byte

for BM, BN, BK in [(16,16,16), (32,32,16), (64,64,32),
                    (128,128,64), (256,128,64), (128,256,64)]:
    # Each A tile [BM × BK] is read N/BN times (once per output N-block)
    # Each B tile [BK × BN] is read M/BM times (once per output M-block)
    a_reads_total = M_ai * K_ai * cdiv(N_ai, BN)   # A tile reused across N blocks
    b_reads_total = K_ai * N_ai * cdiv(M_ai, BM)   # B tile reused across M blocks
    total_reads   = a_reads_total + b_reads_total
    total_bytes   = total_reads * 2   # FP16
    ai            = FLOPS_TOTAL / total_bytes
    bound         = "✅ compute" if ai > H100_RIDGE else "🔴 memory"
    print(f"  {BM}×{BN}×{BK}           {a_reads_total/1e9:>10.2f}G  "
          f"{b_reads_total/1e9:>10.2f}G  {total_bytes/1e9:>8.1f}  "
          f"{ai:>8.1f}  {bound}")

print()
print(f"  H100 ridge: {H100_RIDGE:.0f} FLOPs/Byte (FP16 TC)")
print("  Larger tiles = more reuse = higher AI → compute-bound territory.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: K-loop SMEM reuse analysis
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — K-Loop SMEM Reuse: Accumulator in Registers")
print("━" * 68)
print()

print("  For one output tile C[pid_m, pid_n] of shape BLOCK_M × BLOCK_N:")
print()

BLOCK_M_ex = 128
BLOCK_N_ex = 128
BLOCK_K_ex = 64
K_ex       = 4096

print(f"  BLOCK_M={BLOCK_M_ex}, BLOCK_N={BLOCK_N_ex}, BLOCK_K={BLOCK_K_ex}, K={K_ex}")
print()

n_k_iters   = K_ex // BLOCK_K_ex
smem_a_kb   = BLOCK_M_ex * BLOCK_K_ex * 2 / 1024   # FP16 KB
smem_b_kb   = BLOCK_K_ex * BLOCK_N_ex * 2 / 1024
acc_regs    = BLOCK_M_ex * BLOCK_N_ex              # FP32 values in registers

print(f"  K-loop iterations: {n_k_iters}")
print(f"  SMEM per iteration: A tile = {smem_a_kb:.0f} KB, B tile = {smem_b_kb:.0f} KB")
print(f"  Accumulator: {acc_regs} FP32 values in registers (never written to HBM)")
print()

steps = [
    ("SMEM load A tile",   f"[{BLOCK_M_ex}×{BLOCK_K_ex}] from HBM",      f"{smem_a_kb:.0f} KB"),
    ("SMEM load B tile",   f"[{BLOCK_K_ex}×{BLOCK_N_ex}] from HBM",      f"{smem_b_kb:.0f} KB"),
    ("tl.dot accumulate",  "A_smem @ B_smem → acc (registers)",           f"{acc_regs} FP32"),
    ("next K iteration",   "overwrite SMEM, acc persists in registers",   "(acc kept)"),
    ("...after K loops...",f"acc holds sum over all {n_k_iters} K tiles", "no HBM writes!"),
    ("tl.store",           f"acc → C[pid_m, pid_n] in HBM",               f"{BLOCK_M_ex*BLOCK_N_ex*2//1024} KB"),
]

print(f"  {'Step':<20}  {'Operation':<44}  {'Memory'}")
print("  " + "─" * 74)
for step, op, mem in steps:
    print(f"  {step:<20}  {op:<44}  {mem}")

print()
print(f"  Total HBM writes for this tile: {BLOCK_M_ex*BLOCK_N_ex*2//1024} KB")
print(f"  (vs {n_k_iters*smem_a_kb:.0f} KB of intermediate results if accumulated in HBM)")
print("  Accumulator staying in registers is the key to matmul performance.")
''',
    },

    "3 · Fused Softmax — Online Algorithm, Masking & HBM Savings": {
        "description": (
            "Implement fused softmax step by step: first the naïve 3-pass version "
            "tracking every HBM read/write, then the single-pass online algorithm "
            "that computes max, exp, sum, and normalisation in one kernel. Verify "
            "against scipy/numpy reference. Show how masking handles variable "
            "sequence lengths. Compute the exact HBM traffic reduction. Profile "
            "arithmetic intensity and show when fusion transitions from IO-bound to "
            "compute-bound."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FUSED SOFTMAX — Online Algorithm, Masking & HBM Savings")
print("=" * 68)
print()

np.random.seed(42)


def softmax_ref(x):
    """Standard numerically stable softmax."""
    x = x.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    e  = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cdiv(a, b):
    return (a + b - 1) // b


# ─────────────────────────────────────────────────────────────────────
# Simulate kernels with HBM accounting
# ─────────────────────────────────────────────────────────────────────

def naive_softmax_3pass(x, track=True):
    """
    3-pass naïve softmax — simulates 3 separate CUDA kernels.
    Tracks HBM reads and writes.
    """
    n_rows, n_cols = x.shape
    reads = writes = 0

    # Pass 1: find row max
    reads += x.size   # read x
    row_max = x.max(axis=1)
    writes += n_rows  # write row_max

    # Pass 2: compute exp(x - row_max)
    reads += x.size + n_rows   # read x + row_max
    exp_x = np.exp(x - row_max[:, None])
    writes += x.size  # write exp_x

    # Pass 3: sum and divide
    reads += x.size   # read exp_x
    row_sum = exp_x.sum(axis=1)
    writes += n_rows  # write row_sum
    reads += x.size + n_rows   # read exp_x + row_sum for division
    result = exp_x / row_sum[:, None]
    writes += x.size  # write result

    return result, reads + writes


def fused_softmax_1pass(x):
    """
    Single-pass fused softmax — simulates one Triton kernel.
    For each row: load once, compute max + exp + sum + divide, store once.
    """
    n_rows, n_cols = x.shape
    reads = writes = 0

    result = np.zeros_like(x, dtype=np.float64)

    for pid in range(n_rows):
        # ONE LOAD per row
        row   = x[pid].astype(np.float64)
        reads += n_cols

        # All computation in "registers" (SMEM equivalent)
        row_max  = row.max()
        shifted  = row - row_max
        exp_row  = np.exp(shifted)
        row_sum  = exp_row.sum()
        norm_row = exp_row / row_sum

        # ONE STORE per row
        result[pid] = norm_row
        writes += n_cols

    return result, reads + writes


def fused_softmax_with_mask(x, seq_lens):
    """
    Fused softmax with per-row variable-length masking.
    seq_lens[i] = number of valid tokens in row i.
    """
    n_rows, max_len = x.shape
    BLOCK_N = 1 << math.ceil(math.log2(max_len))  # next power of 2

    result = np.zeros((n_rows, max_len), dtype=np.float64)

    for pid in range(n_rows):
        L = seq_lens[pid]   # valid sequence length for this row
        col_offs = np.arange(BLOCK_N)
        mask     = col_offs < L

        # Load row (masked: out-of-range = -inf to not affect max)
        row_vals = np.full(BLOCK_N, -np.inf)
        row_vals[:L] = x[pid, :L]

        # Online algorithm (one pass)
        row_max  = row_vals[mask].max() if mask.sum() > 0 else 0.0
        exp_row  = np.where(mask, np.exp(row_vals - row_max), 0.0)
        row_sum  = exp_row.sum()
        norm_row = np.where(mask, exp_row / row_sum, 0.0)

        result[pid, :L] = norm_row[:L]

    return result


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness and HBM traffic comparison
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — 3-Pass Naïve vs 1-Pass Fused: HBM Traffic & Correctness")
print("━" * 68)
print()

for n_rows, n_cols, label in [
    (8, 16, "small"),
    (32, 512, "medium"),
    (128, 2048, "attention (seq=2048)"),
    (32, 8192, "long context (seq=8K)"),
]:
    x = np.random.randn(n_rows, n_cols).astype(np.float32)

    y_naive, traffic_naive = naive_softmax_3pass(x)
    y_fused, traffic_fused = fused_softmax_1pass(x)
    y_ref                  = softmax_ref(x)

    err_naive = np.abs(y_naive - y_ref).max()
    err_fused = np.abs(y_fused - y_ref).max()
    reduction  = traffic_naive / traffic_fused

    print(f"  {label} ({n_rows}×{n_cols}):")
    print(f"    3-pass naive: {traffic_naive:>10,} element accesses  "
          f"error={err_naive:.2e} {'✅' if err_naive < 1e-5 else '❌'}")
    print(f"    1-pass fused: {traffic_fused:>10,} element accesses  "
          f"error={err_fused:.2e} {'✅' if err_fused < 1e-5 else '❌'}")
    print(f"    HBM reduction: {reduction:.2f}× fewer accesses")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Online algorithm trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Online Algorithm: Max + Exp + Sum in One Pass (N=8)")
print("━" * 68)
print()

x_trace = np.array([2.0, -1.0, 3.5, 0.5, -2.0, 1.5, 4.0, -0.5], dtype=np.float64)
N_t     = len(x_trace)

print(f"  Input: {x_trace.tolist()}")
print()
print("  ONE PASS through the block tensor (loaded entirely into registers):")
print()

# Load (in Triton: one tl.load call)
row = x_trace.copy()
print(f"  tl.load → x = {row.tolist()}")

# tl.max
row_max = row.max()
print(f"  tl.max(x) → {row_max}")

# x - max (broadcast)
shifted = row - row_max
print(f"  x - max = {shifted.round(4).tolist()}")

# tl.exp
exp_x = np.exp(shifted)
print(f"  tl.exp(x-max) = {exp_x.round(6).tolist()}")

# tl.sum
exp_sum = exp_x.sum()
print(f"  tl.sum(exp) = {exp_sum:.6f}")

# divide
result = exp_x / exp_sum
print(f"  exp / sum = {result.round(6).tolist()}")
print()

ref = softmax_ref(x_trace)
print(f"  Reference: {ref.round(6).tolist()}")
print(f"  Max error: {np.abs(result - ref).max():.2e}  ✅")
print()
print("  ENTIRE COMPUTATION done in registers — ZERO intermediate HBM writes.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Variable-length sequence masking
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Variable-Length Masking: Attention Padding")
print("━" * 68)
print()

n_rows, max_len = 4, 8
x_var  = np.random.randn(n_rows, max_len).astype(np.float32)
seq_lens = [3, 8, 5, 1]   # variable sequence lengths

print(f"  Batch of {n_rows} sequences, max_len={max_len}")
print(f"  Actual lengths: {seq_lens}")
print()

y_var = fused_softmax_with_mask(x_var, seq_lens)

for i, L in enumerate(seq_lens):
    ref_i  = softmax_ref(x_var[i:i+1, :L])
    out_i  = y_var[i, :L]
    err_i  = np.abs(out_i - ref_i).max()
    print(f"  Row {i} (L={L}): softmax over {L} valid tokens, {max_len-L} masked")
    print(f"    Output: {out_i.round(4).tolist()}")
    print(f"    Sums to: {out_i.sum():.6f}  error: {err_i:.2e}  {'✅' if err_i < 1e-5 else '❌'}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Arithmetic intensity of fused softmax
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Arithmetic Intensity: Fused Softmax Roofline")
print("━" * 68)
print()

H100_BW_GBS   = 3350.0
H100_FP32_TFLOPS = 67.0  # scalar FP32 (softmax uses scalar, not TC)
RIDGE_SCALAR  = H100_FP32_TFLOPS * 1e12 / (H100_BW_GBS * 1e9)

print(f"  H100 scalar FP32: {H100_FP32_TFLOPS} TFLOP/s, BW: {H100_BW_GBS} GB/s")
print(f"  Scalar ridge: {RIDGE_SCALAR:.1f} FLOPs/Byte")
print()
print(f"  {'Shape':>16}  {'FLOPs':>14}  {'HBM bytes':>12}  "
      f"{'AI':>8}  {'Bound'}")
print("  " + "─" * 58)

for n_rows, n_cols in [(1, 128), (32, 512), (128, 2048),
                        (32, 8192), (128, 32768)]:
    # FLOPs: 1 max + N exp + 1 sum + N div + N sub ≈ 4N per row
    flops_per_row = 4 * n_cols   # sub, exp, sum, div
    total_flops   = flops_per_row * n_rows
    # HBM: 1 read + 1 write per element (fused)
    hbm_bytes     = 2 * n_rows * n_cols * 4   # FP32
    ai            = total_flops / hbm_bytes
    bound         = "compute" if ai > RIDGE_SCALAR else "🔴 memory"
    print(f"  {n_rows:>5}×{n_cols:<10}  {total_flops/1e6:>12.2f}M  "
          f"{hbm_bytes/1e6:>10.2f}M  {ai:>8.2f}  {bound}")

print()
print(f"  Softmax AI ≈ 4N / (2N×4) = 0.5 FLOPs/Byte — always memory-bound.")
print("  Fusion's value: 2.5× less HBM traffic → 2.5× faster despite same AI.")
''',
    },

    "4 · Autotuning — Config Space, tl.constexpr & Cache Mechanics": {
        "description": (
            "Implement a Triton-style autotuner in Python: define a configuration "
            "space over BLOCK_M, BLOCK_N, BLOCK_K, num_warps, and num_stages. "
            "Simulate kernel compilation and benchmarking. Show how tl.constexpr "
            "enables per-config specialisation. Demonstrate the config cache keyed "
            "by (M, N, K, dtype). Show how autotuning finds the optimal config for "
            "different matrix sizes. Model the expected speedup from optimal tuning."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from itertools import product
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

print("=" * 68)
print("  AUTOTUNING — Config Space, tl.constexpr & Cache Mechanics")
print("=" * 68)
print()

np.random.seed(42)

# H100 hardware constants
H100_HBM_BW_GBS  = 3350.0
H100_TC_TFLOPS   = 989.0    # FP16 tensor core
H100_SMs         = 132
H100_SMEM_KB     = 228


@dataclass
class TritonConfig:
    BLOCK_M:    int
    BLOCK_N:    int
    BLOCK_K:    int
    num_warps:  int
    num_stages: int

    def smem_usage_kb(self, dtype_bytes=2):
        """SMEM needed to hold A tile + B tile (per pipeline stage)."""
        a_kb = self.BLOCK_M * self.BLOCK_K * dtype_bytes / 1024
        b_kb = self.BLOCK_K * self.BLOCK_N * dtype_bytes / 1024
        return (a_kb + b_kb) * self.num_stages

    def register_usage_per_warp(self, dtype_bytes=2):
        """Approximate registers per thread for this config."""
        # Accumulator: BLOCK_M * BLOCK_N / (num_warps * 32) FP32 values per thread
        acc_regs    = self.BLOCK_M * self.BLOCK_N // (self.num_warps * 32) * 1
        layout_regs = 16  # overhead: indices, pointers, temporaries
        return acc_regs + layout_regs

    def occupancy_fraction(self):
        """
        Approximate SM occupancy based on SMEM and register limits.
        A100/H100: max 232 KB SMEM per SM, 65536 registers per SM.
        """
        smem_ok      = self.smem_usage_kb() <= H100_SMEM_KB
        if not smem_ok:
            return 0.0
        # Registers per block: num_warps * 32 threads * regs_per_thread
        regs_per_block = self.num_warps * 32 * self.register_usage_per_warp()
        # H100 has 65536 registers per SM → max blocks per SM
        max_blocks_smem = H100_SMEM_KB // max(self.smem_usage_kb(), 1e-3)
        max_blocks_regs = 65536 // max(regs_per_block, 1)
        max_blocks_sm   = min(max_blocks_smem, max_blocks_regs, 32)  # hardware limit
        # Active warps: max_blocks * num_warps, capped at 64
        active_warps    = min(max_blocks_sm * self.num_warps, 64)
        return active_warps / 64.0

    def is_valid(self):
        """Check if this config is physically feasible."""
        return (self.smem_usage_kb() <= H100_SMEM_KB and
                self.occupancy_fraction() > 0.1 and
                self.BLOCK_K >= 16)   # minimum for TC alignment

    def estimated_tflops(self, M, N, K, dtype_bytes=2):
        """
        Estimate effective TFLOP/s for this config on a given matrix size.
        Based on: occupancy, arithmetic intensity, pipeline efficiency.
        """
        if not self.is_valid():
            return 0.0

        # Arithmetic intensity of tiled matmul
        flops = 2 * M * N * K
        # A reads: M*K * (N/BLOCK_N) times = M*K per program * N/BLOCK_N programs
        a_reads = M * K * math.ceil(N / self.BLOCK_N)
        b_reads = K * N * math.ceil(M / self.BLOCK_M)
        bytes_io = (a_reads + b_reads) * dtype_bytes
        ai = flops / bytes_io

        # Pipeline efficiency: with num_stages pipelining, HBM latency is hidden
        # Assume latency hidden if num_stages * TC_time >= HBM_latency
        pipeline_eff = min(0.95, 0.5 + 0.15 * self.num_stages)

        # Occupancy factor: higher occupancy = better latency hiding
        occ_factor = 0.5 + 0.5 * self.occupancy_fraction()

        # Roofline: min of compute and bandwidth bound
        compute_bound_tflops = H100_TC_TFLOPS * pipeline_eff * occ_factor
        bw_bound_tflops      = H100_HBM_BW_GBS * ai

        return min(compute_bound_tflops, bw_bound_tflops)


class TritonAutotuner:
    """Simulates Triton's @triton.autotune decorator."""

    def __init__(self, configs: List[TritonConfig], key_args: List[str]):
        self.configs    = [c for c in configs if c.is_valid()]
        self.key_args   = key_args
        self.best_cache: Dict = {}   # (key_values) → best TritonConfig

    def tune(self, M, N, K, verbose=False):
        """Run autotuning for given matrix dimensions. Returns best config."""
        cache_key = (M, N, K)
        if cache_key in self.best_cache:
            if verbose:
                print(f"  Cache HIT for ({M}, {N}, {K})")
            return self.best_cache[cache_key]

        # Compile and benchmark all valid configs
        results = []
        for cfg in self.configs:
            tflops = cfg.estimated_tflops(M, N, K)
            results.append((tflops, cfg))

        results.sort(key=lambda x: -x[0])
        best_tflops, best_cfg = results[0]

        self.best_cache[cache_key] = best_cfg

        if verbose:
            print(f"  Autotuning for M={M}, N={N}, K={K}:")
            print(f"  Top 5 configs:")
            for tfl, cfg in results[:5]:
                print(f"    BM={cfg.BLOCK_M:>4} BN={cfg.BLOCK_N:>4} BK={cfg.BLOCK_K:>3} "
                      f"nw={cfg.num_warps} ns={cfg.num_stages} "
                      f"smem={cfg.smem_usage_kb():.0f}KB occ={cfg.occupancy_fraction():.0%} "
                      f"→ {tfl:.0f} GFLOP/s")

        return best_cfg


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Config space definition
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Triton Config Space: All Combinations")
print("━" * 68)
print()

configs = []
for BM, BN, BK, nw, ns in product(
    [16, 32, 64, 128, 256],  # BLOCK_M
    [16, 32, 64, 128, 256],  # BLOCK_N
    [16, 32, 64, 128],       # BLOCK_K
    [2, 4, 8, 16],           # num_warps
    [1, 2, 3, 4],            # num_stages
):
    configs.append(TritonConfig(BM, BN, BK, nw, ns))

valid_configs = [c for c in configs if c.is_valid()]
print(f"  Total configurations: {len(configs):,}")
print(f"  Valid (SMEM + occupancy): {len(valid_configs):,}")
print(f"  Invalid (SMEM overflow or low occupancy): {len(configs)-len(valid_configs):,}")
print()

# Show SMEM breakdown for some configs
print(f"  {'Config':<32}  {'SMEM (KB)':>10}  {'Occupancy':>12}  {'Valid?'}")
print("  " + "─" * 60)
sample_configs = [
    TritonConfig(128, 256, 64, 8, 3),
    TritonConfig(128, 128, 32, 4, 4),
    TritonConfig(256, 256, 64, 8, 3),   # likely too much SMEM
    TritonConfig(64,  64,  32, 4, 2),
    TritonConfig(32,  32,  16, 2, 1),
]
for cfg in sample_configs:
    valid = cfg.is_valid()
    smem  = cfg.smem_usage_kb()
    occ   = cfg.occupancy_fraction()
    print(f"  BM={cfg.BLOCK_M:>3} BN={cfg.BLOCK_N:>3} BK={cfg.BLOCK_K:>3} "
          f"nw={cfg.num_warps} ns={cfg.num_stages}  "
          f"{smem:>10.1f}  {occ:>11.0%}  {'✅' if valid else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Autotuning for different matrix sizes
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Best Config per Matrix Size: Autotuner Recommendations")
print("━" * 68)
print()

tuner = TritonAutotuner(valid_configs, key_args=["M", "N", "K"])

test_shapes = [
    (64,   64,   64,  "tiny (LLM decode small)"),
    (128,  128,  128, "small"),
    (512,  512,  512, "medium"),
    (1024, 1024, 4096,"FFN gate proj (batch=8)"),
    (4096, 4096, 4096,"large square"),
    (8192, 8192, 8192,"very large square"),
    (1,    4096, 4096,"decode (batch=1, N=1)"),
    (4096, 128,  4096,"tall-thin (decode batch=32)"),
]

print(f"  {'Shape (M×N×K)':<28}  {'Best Config':<36}  "
      f"{'Est. TFLOP/s':>13}  {'vs cuBLAS'}")
print("  " + "─" * 86)

CUBLAS_PEAK = H100_TC_TFLOPS * 0.92  # cuBLAS achieves ~92% of peak for large shapes

for M, N, K, label in test_shapes:
    best = tuner.tune(M, N, K)
    tflops = best.estimated_tflops(M, N, K)
    # cuBLAS degrades for small shapes
    cublas_tflops = CUBLAS_PEAK * min(1.0, math.sqrt(min(M,N,K) / 512))
    pct   = tflops / cublas_tflops * 100 if cublas_tflops > 0 else 0
    config_str = (f"BM={best.BLOCK_M} BN={best.BLOCK_N} BK={best.BLOCK_K} "
                  f"nw={best.num_warps} ns={best.num_stages}")
    print(f"  {label:<28}  {config_str:<36}  "
          f"{tflops:>10.0f}G  {pct:>8.0f}%")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: tl.constexpr — what changes between configs
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — tl.constexpr: How Configs Specialise the Compiled Kernel")
print("━" * 68)
print()

print("  Each (BLOCK_M, BLOCK_N, BLOCK_K) combination produces a DIFFERENT kernel.")
print("  tl.constexpr values are constants in the compiled PTX code.")
print()

cfg_a = TritonConfig(128, 256, 64, 8, 3)
cfg_b = TritonConfig(64,  64,  32, 4, 2)

print(f"  Config A: BM={cfg_a.BLOCK_M}, BN={cfg_a.BLOCK_N}, BK={cfg_a.BLOCK_K}, "
      f"nw={cfg_a.num_warps}, ns={cfg_a.num_stages}")
print(f"  Config B: BM={cfg_b.BLOCK_M}, BN={cfg_b.BLOCK_N}, BK={cfg_b.BLOCK_K}, "
      f"nw={cfg_b.num_warps}, ns={cfg_b.num_stages}")
print()

M_ex, N_ex, K_ex = 4096, 4096, 4096
diffs = [
    ("Loop trip count",
     f"K // BLOCK_K = {K_ex // cfg_a.BLOCK_K}",
     f"K // BLOCK_K = {K_ex // cfg_b.BLOCK_K}",
     "Unrolled differently if BLOCK_K known"),
    ("Block tensor shapes",
     f"arange(0, {cfg_a.BLOCK_M}), arange(0, {cfg_a.BLOCK_K})",
     f"arange(0, {cfg_b.BLOCK_M}), arange(0, {cfg_b.BLOCK_K})",
     "Different size register tiles"),
    ("tl.dot shape",
     f"[{cfg_a.BLOCK_M}×{cfg_a.BLOCK_K}] @ [{cfg_a.BLOCK_K}×{cfg_a.BLOCK_N}]",
     f"[{cfg_b.BLOCK_M}×{cfg_b.BLOCK_K}] @ [{cfg_b.BLOCK_K}×{cfg_b.BLOCK_N}]",
     "Different WMMA instruction size"),
    ("SMEM size",
     f"{cfg_a.smem_usage_kb():.0f} KB",
     f"{cfg_b.smem_usage_kb():.0f} KB",
     "Static allocation at compile time"),
    ("cp.async pipelines",
     f"{cfg_a.num_stages} stages",
     f"{cfg_b.num_stages} stages",
     "Different async barriers inserted"),
    ("Threads per block",
     f"{cfg_a.num_warps * 32}",
     f"{cfg_b.num_warps * 32}",
     "Different grid/block launch params"),
    ("Grid size",
     f"({math.ceil(M_ex/cfg_a.BLOCK_M)}×{math.ceil(N_ex/cfg_a.BLOCK_N)})",
     f"({math.ceil(M_ex/cfg_b.BLOCK_M)}×{math.ceil(N_ex/cfg_b.BLOCK_N)})",
     "Number of program instances"),
]

print(f"  {'Aspect':<22}  {'Config A':>32}  {'Config B':>32}  {'Impact'}")
print("  " + "─" * 88)
for aspect, va, vb, note in diffs:
    print(f"  {aspect:<22}  {va:>32}  {vb:>32}  {note}")

print()
print("  Each config → entirely different PTX. Triton caches all PTX variants.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Autotuning cache mechanics
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Autotuning Cache: Key Design and Cache Hits")
print("━" * 68)
print()

print("  Triton caches autotuning results keyed by:")
print("    (M, N, K, dtype, device_arch)")
print("  Cache stored in: ~/.triton/cache/<hash>/")
print()

tuner2 = TritonAutotuner(valid_configs, key_args=["M", "N", "K"])

# Simulate repeated calls
calls = [
    (4096, 4096, 4096, "first call — must autotune"),
    (4096, 4096, 4096, "same shape — cache HIT"),
    (8192, 4096, 4096, "different M — must autotune"),
    (4096, 4096, 4096, "same shape as first — cache HIT"),
    (8192, 4096, 4096, "same as third — cache HIT"),
]

TUNE_MS    = 5000   # ms to autotune (compile + benchmark all configs)
CACHE_MS   = 0.1    # ms to retrieve from cache

total_time = 0.0
print(f"  {'Call':>5}  {'Shape':>18}  {'Cache':>8}  "
      f"{'Time (ms)':>12}  {'Action'}")
print("  " + "─" * 64)

for i, (M, N, K, desc) in enumerate(calls):
    key = (M, N, K)
    is_hit = key in tuner2.best_cache
    if is_hit:
        t_ms = CACHE_MS
        action = "load from disk"
    else:
        tuner2.tune(M, N, K)
        t_ms  = TUNE_MS
        action = "compile all configs + benchmark"
    total_time += t_ms
    hit_str = "HIT ✅" if is_hit else "MISS ❌"
    print(f"  {i+1:>5}  {M}×{N}×{K}  {hit_str:>8}  {t_ms:>12.1f}  {action}")

print()
print(f"  Total time: {total_time:.0f} ms  ({total_time/1000:.1f} seconds)")
print(f"  Cache reduces repeated-shape overhead from {TUNE_MS} ms to {CACHE_MS} ms.")
print(f"  Production tip: warm the cache at server startup with representative shapes.")
''',
    },

    "5 · Triton vs CUDA vs PyTorch — Performance Comparison & When to Use": {
        "description": (
            "Build a comprehensive performance model comparing PyTorch eager, "
            "torch.compile, hand-written Triton, and CUDA for four representative "
            "operations: elementwise kernel, row reduction (softmax), matrix "
            "multiplication, and fused attention. Show HBM traffic, kernel launch "
            "count, and expected throughput for each. Identify when Triton "
            "outperforms PyTorch and when CUDA is still needed."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  TRITON vs CUDA vs PYTORCH — Performance Comparison & When to Use")
print("=" * 68)
print()

np.random.seed(77)

H100_HBM_BW_GBS   = 3350.0
H100_TC_TFLOPS    = 989.0     # FP16 tensor core
H100_SCALAR_TFLOPS= 67.0      # FP32 scalar (elementwise)
LAUNCH_US         = 5.0       # µs per kernel launch
FP16_BYTES        = 2


def bw_time_ms(bytes_, bw_gbs=H100_HBM_BW_GBS):
    return bytes_ / (bw_gbs * 1e9) * 1000

def compute_time_ms(flops, peak_tflops=H100_TC_TFLOPS):
    return flops / (peak_tflops * 1e12) * 1000

def kernel_time_ms(bytes_, flops, bw_gbs, peak_tflops, n_launches=1):
    t_bw   = bw_time_ms(bytes_, bw_gbs)
    t_cmp  = compute_time_ms(flops, peak_tflops)
    t_gpu  = max(t_bw, t_cmp)
    t_oh   = n_launches * LAUNCH_US / 1000
    return t_gpu + t_oh


def perf_table(op_name, configs):
    print(f"  ── {op_name} ──")
    print(f"  {'Method':<24}  {'HBM (MB)':>10}  {'FLOPs (M)':>11}  "
          f"{'Kernels':>9}  {'Time (ms)':>11}  {'vs PyTorch'}")
    print("  " + "─" * 70)
    baseline = None
    for name, hbm_mb, flops_m, n_kernels, t_ms in configs:
        if baseline is None:
            baseline = t_ms
        ratio = baseline / t_ms if t_ms > 0 else 0
        faster = f"{ratio:.2f}×" if ratio > 1 else f"1.00×"
        print(f"  {name:<24}  {hbm_mb:>10.1f}  {flops_m:>11.2f}  "
              f"{n_kernels:>9}  {t_ms:>11.4f}  {faster:>10}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Elementwise kernel — GELU activation
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Elementwise GELU: N=4096×4096 (67 MB)")
print("━" * 68)
print()

N_EL    = 4096 * 4096
hbm_el  = N_EL * FP16_BYTES   # one tensor
flops_el= N_EL * 15            # GELU: ~15 scalar FLOPs

# PyTorch eager: 1 kernel but intermediate alloc if combined ops
# (assume fused by pytorch: 1 kernel for single op)
t_pt_eager   = bw_time_ms(hbm_el * 2)              # read + write
t_triton     = bw_time_ms(hbm_el * 2)              # same, but with autotuned tile
t_cuda_gelu  = bw_time_ms(hbm_el * 2) * 0.95       # ~5% faster (hand-tuned)

print(f"  Shape: N = {N_EL:,}, dtype=FP16")
print(f"  HBM bandwidth limited: AI = {flops_el/(hbm_el*2):.1f} FLOPs/Byte << ridge")
print()
perf_table("GELU activation", [
    ("PyTorch eager",           hbm_el*2/1e6, flops_el/1e6, 1, t_pt_eager),
    ("torch.compile",           hbm_el*2/1e6, flops_el/1e6, 1, t_pt_eager*0.97),
    ("Triton (autotuned)",      hbm_el*2/1e6, flops_el/1e6, 1, t_triton*0.97),
    ("CUDA (hand-tuned)",       hbm_el*2/1e6, flops_el/1e6, 1, t_cuda_gelu),
])
print("  For single elementwise ops: all methods similar (bandwidth-bound). Triton = CUDA.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Fused GELU + add + layer norm chain
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Fused Chain: GELU + Add + LayerNorm (same N)")
print("━" * 68)
print()

N_chain    = N_EL
hbm_chain  = N_chain * FP16_BYTES  # input tensor size

# PyTorch eager: 3 kernels, 2 intermediate writes + 2 reads
t_eager_3k = (3 * bw_time_ms(hbm_chain * 2) + 3 * LAUNCH_US/1000)

# torch.compile: can fuse these 3 into 1 kernel
t_compile  = bw_time_ms(hbm_chain * 2) + LAUNCH_US/1000

# Triton: same as compile (both fuse), but Triton is explicit about it
t_triton_c = bw_time_ms(hbm_chain * 2) + LAUNCH_US/1000

flops_chain = N_chain * (15 + 2 + 15)  # gelu + add + layernorm approx

print(f"  Chain: GELU(x) + residual_add + LayerNorm")
print(f"  Fusion eliminates 2 intermediate tensor writes to HBM")
print()
perf_table("Fused GELU+Add+LN", [
    ("PyTorch eager (3 kernels)", hbm_chain*6/1e6, flops_chain/1e6, 3, t_eager_3k),
    ("torch.compile (fused)",     hbm_chain*2/1e6, flops_chain/1e6, 1, t_compile),
    ("Triton (explicit fusion)",  hbm_chain*2/1e6, flops_chain/1e6, 1, t_triton_c),
])
print("  torch.compile and Triton both fuse the chain. For standard ops: compile wins.")
print("  Use Triton when you need custom fusion not expressible in PyTorch ops.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Matrix multiplication
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Matrix Multiplication: M=N=K=4096 (FP16)")
print("━" * 68)
print()

M_mm = N_mm = K_mm = 4096
flops_mm = 2 * M_mm * N_mm * K_mm
hbm_mm   = (M_mm*K_mm + K_mm*N_mm + M_mm*N_mm) * FP16_BYTES
ai_mm    = flops_mm / hbm_mm

t_pytorch_mm  = max(bw_time_ms(hbm_mm), compute_time_ms(flops_mm)) * 1.12
t_cublas      = max(bw_time_ms(hbm_mm), compute_time_ms(flops_mm))  # ~100% of peak
t_triton_mm   = max(bw_time_ms(hbm_mm), compute_time_ms(flops_mm)) * 1.02   # ~98% cuBLAS

print(f"  M=N=K={M_mm}, FP16. AI={ai_mm:.1f} FLOPs/Byte >> ridge → compute-bound")
print()
perf_table("MatMul 4K×4K×4K", [
    ("torch.mm (calls cuBLAS)",  hbm_mm/1e6, flops_mm/1e6, 1, t_pytorch_mm),
    ("cuBLAS GEMM (FP16)",       hbm_mm/1e6, flops_mm/1e6, 1, t_cublas),
    ("Triton (autotuned)",       hbm_mm/1e6, flops_mm/1e6, 1, t_triton_mm),
    ("Triton (manual,large)",    hbm_mm/1e6, flops_mm/1e6, 1, t_triton_mm*0.99),
])
print("  Large square GEMM: Triton ≈ 98% of cuBLAS. cuBLAS wins for tiny shapes.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: When to use each approach
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Decision Guide: When to Use Each Approach")
print("━" * 68)
print()

decision_matrix = [
    ("Standard op (ReLU, add, mm)", "torch.compile",
     "Automatic fusion handles most cases. Zero engineering cost.",
     "Standard transformer training/inference with no custom ops."),
    ("Novel fused op (SwiGLU+dropout)",  "Triton",
     "Fuse any sequence without materialising intermediates.",
     "Custom activation, fused bias+norm+activation chains."),
    ("Custom attention variant",         "Triton",
     "Flash Attention pattern: online softmax + KV tiling. Python DSL.",
     "Sliding-window, alibi, RoPE-fused attention, GQA custom kernels."),
    ("Large standard GEMM",              "torch.mm/cublas",
     "cuBLAS beats Triton for perfectly-sized large GEMMs.",
     "Weight-compute GEMM in transformer FFN at large batch."),
    ("Tiny GEMM (batch=1, decode)",      "Triton or cuBLAS",
     "cuBLAS has hardcoded tiny-N kernels. Triton with custom tuning.",
     "Decode-phase projection GEMMs (single-token generation)."),
    ("Hardware intrinsics (TMA, WGMMA)", "CUDA C++",
     "Triton 3.0 exposes some; full H100 features need CUDA.",
     "Production FlashAttention3, CUTLASS custom ops."),
    ("Iterative solver, graph traverse",  "Triton",
     "Persistent kernel + work queue pattern in Python.",
     "Sparse attention, A* search, irregular parallel workloads."),
    ("Prototype/research kernel",         "Triton",
     "10× less code than CUDA, autotuning, Python debugging.",
     "New attention mechanisms, novel normalisation, sparse ops."),
]

print(f"  {'Use case':<36}  {'Approach':>20}  {'Rationale'}")
print("  " + "─" * 90)
for use_case, approach, rationale, example in decision_matrix:
    print(f"  {use_case:<36}  {approach:>20}  {rationale}")

print()
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ TRITON IS NOT A CUDA REPLACEMENT. It is the optimal tool when:  │")
print("  │   1. You need FUSION that torch.compile cannot express.         │")
print("  │   2. The computation maps naturally to 2D tile blocks.          │")
print("  │   3. Autotuning replaces manual CUDA kernel optimisation.       │")
print("  │   4. You want GPU perf from a Python workflow without C++.      │")
print("  └─────────────────────────────────────────────────────────────────┘")
''',
    },
}

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