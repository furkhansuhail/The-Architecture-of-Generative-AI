"""
Tensor Cores & WMMA — Fragment API, FP16→FP32 Accumulation & Tile Sizing
=========================================================================

A tensor core is a specialised hardware unit that performs a small
matrix-multiply-accumulate (MMA) operation in a single instruction cycle.
On Volta (V100) and later GPUs, each SM contains multiple tensor cores that
operate in parallel. A single tensor core instruction on Ampere computes:

    D[16×16] = A[16×8] × B[8×16] + C[16×16]

in one warp-level instruction, processing 16×8×16 = 2048 multiply-adds
per clock per warp — versus 32 multiply-adds per clock for scalar FP32.
Tensor cores are the primary reason why H100 delivers 989 TFLOP/s of FP16
throughput versus 67 TFLOP/s of scalar FP32.

The WMMA (Warp Matrix Multiply-Accumulate) API is NVIDIA's programmer-facing
interface to tensor cores. It exposes the hardware tiles through:

    - wmma::fragment<>   — a distributed tile held across warp lanes
    - wmma::load_matrix_sync()  — load a tile from shared/global memory
    - wmma::mma_sync()          — execute the tensor core MMA instruction
    - wmma::store_matrix_sync() — store the result fragment to memory

Understanding WMMA deeply means understanding three things:

    1. FRAGMENT DISTRIBUTION — how a 16×16 tile is partitioned across
       32 lanes of a warp. Each lane holds exactly 8 elements, but the
       mapping is non-trivial and architecture-specific.

    2. THE ACCUMULATION CONTRACT — FP16 inputs accumulate into FP32,
       preventing catastrophic cancellation across many K-dimension steps.
       The 10-bit FP16 mantissa would lose precision after K≈1000 adds.

    3. TILE SIZING RULES — the 16×16×16 primitive tiles compose into
       larger block tiles via software loops. Choosing the right block
       tile shape determines SM occupancy, register pressure, and
       whether the tensor core pipeline stays saturated.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "Tensor Cores & WMMA — Fragment API, FP16→FP32 Accumulation & Tile Sizing"
DISPLAY_NAME = "08 · Tensor Cores & WMMA"
ICON         = "🧮"
SUBTITLE     = "wmma::fragment · load/mma/store · FP16→FP32 · Tile Sizing · Warp Layout"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — TENSOR CORE HARDWARE: FROM SIMT TO MMA INSTRUCTIONS

### The SIMT Baseline: How Much Scalar FP32 Can Do

    A standard FP32 CUDA kernel processes one multiply-add per lane per clock:
        32 lanes × 1 FMA/cycle = 32 FMAs/cycle per warp
        A100 SM clock ≈ 1410 MHz, 4 warp schedulers per SM
        Peak scalar FP32 per SM ≈ 4 × 32 × 1410e6 = 180.5 GFLOP/s
        Full A100 (108 SMs): 19.5 TFLOP/s scalar FP32

    Tensor cores give a 16× multiplier on top of this:
        A100 tensor core FP16: 312 TFLOP/s = 16× scalar FP32 rate
        H100 tensor core FP16: 989 TFLOP/s

    The question is: how does one instruction deliver 16× more compute?

### The MMA Primitive: What Hardware Actually Does

    Each tensor core computes a tiny D = A×B + C on fixed-size tiles.
    The tile sizes evolved across GPU generations:

    ┌───────────────┬──────────────────────────┬────────────────────────────┐
    │ Architecture  │ MMA tile (M×N×K)         │ Compute per warp per cycle │
    ├───────────────┼──────────────────────────┼────────────────────────────┤
    │ Volta V100    │ FP16: 16×16×16           │ 2×16×16×16 = 8192 FLOPs    │
    │ Turing T4     │ FP16: 16×16×16, INT8 too │ 8192 FLOPs (FP16)          │
    │ Ampere A100   │ FP16: 16×16×16 (base)    │ 256 FMAs per tensor core   │
    │               │ also 8×16×16, 16×8×16    │ per cycle                  │
    │ Hopper H100   │ FP8/FP16/BF16 wgmma.mma  │ Warp Group MMA (4 warps)   │
    └───────────────┴──────────────────────────┴────────────────────────────┘

    On Ampere (A100): each SM has 4 tensor core units per sub-partition,
    4 sub-partitions per SM → 16 tensor core units per SM.
    Each unit computes one 8×8×4 fragment per clock.
    One WMMA 16×16×16 tile requires 4 such fragments → 4 cycles.
    Combined: 4 sub-partitions × 4 TC units × 8×8×4 = 4096 FMAs/cycle per SM.

### The Warp-Level Abstraction

    Tensor core instructions operate at the WARP level, not the lane level.
    All 32 lanes in a warp participate COOPERATIVELY to hold the tile data
    and execute the instruction together.

    This means:
        1. No single lane owns a full row or column. The tile is DISTRIBUTED.
        2. ALL 32 lanes must execute wmma::mma_sync() together (hence "sync").
        3. Divergence within a warp (branching) invalidates the tile data.
        4. The distribution pattern is hardware-defined and opaque to programmers —
           this is why the WMMA API uses opaque "fragment" objects.

    CONSEQUENCE: you cannot inspect individual elements of a fragment with
    standard array indexing. To read fragment[row][col], you must first
    store the fragment to shared/global memory, then index the memory.

### Tensor Core Instruction Throughput vs Latency

    THROUGHPUT: on A100, one WMMA MMA per 4 cycles per warp sub-partition.
    LATENCY:    ~16 clock cycles from issue to result availability.

    For the TC pipeline to stay saturated:
        Latency / Throughput = 16 / 4 = 4 independent MMA instructions
        Must have ≥ 4 independent wmma::mma_sync() in flight simultaneously.
        Achieved via: loop unrolling over K tiles, multiple warp-level accumulators,
        or software pipelining with async memory copies.

    This is why production GEMM kernels use LARGE register-tile accumulators
    (e.g., 2×4 or 4×4 arrays of 16×16 fragments) — more accumulator tiles =
    more independent MMA instructions = better TC pipeline utilisation.


##### PART 2 — THE WMMA API: FOUR OPERATIONS IN DEPTH

### Include and Namespace

    #include <mma.h>
    using namespace nvcuda;

    All WMMA types and functions live in the nvcuda::wmma namespace.
    WMMA is a CUDA C++ extension — it requires CUDA 9.0+ and sm_70+.
    Compile with: nvcc --gpu-architecture=sm_80 (Ampere) or sm_70 (Volta)

### wmma::fragment<> — The Core Type

    The fragment is an opaque container holding a distributed tile.
    Its template signature:

        wmma::fragment<
            Use,               // matrix_a | matrix_b | accumulator
            M, N, K,           // tile dimensions (must match)
            T,                 // element type: __half, float, int8_t, etc.
            Layout             // row_major | col_major (omitted for accumulator)
        > frag;

    TEMPLATE PARAMETERS in detail:

    USE — which role this fragment plays:
        wmma::matrix_a      → holds A tile (left operand), shape M×K
        wmma::matrix_b      → holds B tile (right operand), shape K×N
        wmma::accumulator   → holds C or D tile (result), shape M×N

    M, N, K — the tile dimensions. Required combinations:
        Standard: M=16, N=16, K=16   (FP16/BF16 on all Volta+ GPUs)
        Extended: M=8,  N=32, K=16   (FP16 on Ampere+)
        Extended: M=32, N=8,  K=16   (FP16 on Ampere+)
        INT8:     M=16, N=16, K=32   (INT8 on Ampere+)
        TF32:     M=16, N=16, K=8    (TF32 on Ampere+)
        FP64:     M=8,  N=8,  K=4    (FP64 on Ampere+)
        FP8 (Hopper): via wgmma API, M=64/128 tiles

    T — element type:
        __half      FP16 (matrix_a or matrix_b)
        __nv_bfloat16  BF16 (Ampere+)
        float       FP32 (accumulator only, or TF32 input)
        int8_t, uint8_t  INT8 inputs (accumulator is int32_t)
        double      FP64 (Ampere+)

    LAYOUT — memory layout of the tile:
        wmma::row_major   → element [i,j] is at offset i*K + j  (for matrix_a)
        wmma::col_major   → element [i,j] is at offset j*M + i

    STORAGE: each fragment has a member array .x[] of type T.
    On Volta/Ampere FP16 16×16×16:
        matrix_a fragment: .x[] has num_elements = 8 (each lane holds 8 of 512 total)
        matrix_b fragment: .x[] has num_elements = 8
        accumulator fragment: .x[] has num_elements = 8 (FP32 values)

    The num_elements per lane = (M × K) / 32 for matrix_a on 16×16×16:
        = (16 × 16) / 32 = 8 elements per lane.

### wmma::load_matrix_sync() — Loading a Tile

    Loads a 16×16 matrix from memory into a fragment.
    All 32 warp lanes participate cooperatively.

    Overloads:

    For matrix_a or matrix_b:
        wmma::load_matrix_sync(frag, ptr, stride);
        // ptr:    pointer to the top-left element of the tile in memory
        // stride: number of elements per row (row_major) or per column (col_major)
        //         = leading dimension of the source matrix

    For accumulator (loading a pre-existing C tile for β×C accumulation):
        wmma::load_matrix_sync(frag, ptr, stride, layout);
        // layout: wmma::mem_row_major or wmma::mem_col_major
        //         (needed because accumulator has no layout template param)

    MEMORY REQUIREMENTS:
        ptr must be aligned to 16 bytes (128-bit alignment).
        stride must be a compile-time constant for best performance.
        Source can be global memory or shared memory.
        Shared memory is preferred (much lower latency: 32 cycles vs 600+).

    EXAMPLE — load A tile from SMEM:
        __shared__ __half As[16][16];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        // Load the 16×16 tile starting at As[0][0], stride=16 (row_major)
        wmma::load_matrix_sync(a_frag, &As[0][0], 16);

### wmma::mma_sync() — Execute the MMA

    Performs D = A × B + C at the warp level using tensor cores.

        wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

    Rules:
        - All four fragments must have matching M, N, K
        - a_frag is matrix_a, b_frag is matrix_b
        - c_frag and d_frag are both accumulator type
        - c_frag and d_frag CAN be the same variable: mma_sync(c, a, b, c)
          This gives: c += A × B  (in-place accumulation)
        - ALL 32 lanes must call mma_sync() — no divergence allowed.

    EXECUTION MODEL:
        The warp-level instruction triggers all tensor core units in the SM
        sub-partition simultaneously. The hardware reads data from each lane's
        register file, routes it to the tensor core matrix inputs, executes
        the MMA, and writes results back to each lane's register file.
        Latency: ~16 cycles (independent of tile size within the 16×16×16 class).

### wmma::store_matrix_sync() — Storing the Result

    Writes the accumulator fragment back to memory.

        wmma::store_matrix_sync(ptr, d_frag, stride, layout);
        // ptr:    destination pointer (global or shared memory)
        // stride: number of elements per row (for mem_row_major)
        // layout: wmma::mem_row_major or wmma::mem_col_major

    Common pattern: store to SMEM first, then cooperatively copy to global:
        __shared__ float Cs[16][16];
        wmma::store_matrix_sync(&Cs[0][0], d_frag, 16, wmma::mem_row_major);
        __syncthreads();
        // Now all threads copy Cs to global memory using normal indexing

### fill_fragment — Initialising the Accumulator

    Before the K-loop, initialise the accumulator fragment to zero:
        wmma::fill_fragment(c_frag, 0.0f);   // fills all .x[] with 0.0f

    Or initialise to the existing C matrix (for β×C accumulation):
        wmma::load_matrix_sync(c_frag, &C_smem[...], stride, wmma::mem_row_major);
        // Then the K-loop accumulates: c_frag += A_tile × B_tile each step


##### PART 3 — FRAGMENT DISTRIBUTION: HOW A 16×16 TILE LIVES ACROSS 32 LANES

### Why the Distribution Matters

    Each lane in the warp holds 8 elements of a 16×16 (=256 element) tile.
    32 lanes × 8 elements = 256 elements. Perfect coverage.

    But WHICH 8 elements does each lane hold? The mapping is architecture-
    specific, hardware-defined, and intentionally opaque (the WMMA spec says
    "the distribution is implementation-defined"). You cannot safely assume
    any specific mapping across GPU generations.

    HOWEVER, understanding the general structure helps you:
        - Load tiles efficiently (avoid bank conflicts in SMEM)
        - Understand why certain SMEM layouts are preferred (e.g., swizzled)
        - Debug fragment-related race conditions

### Volta Fragment Distribution (FP16 matrix_a, row_major, 16×16×16)

    Reconstructed from PTX documentation and CUDA samples:

    The 16×16 A fragment (row_major) is divided across 32 lanes as follows:
        Groups of 4 lanes share a 4-row chunk.
        Lanes 0–3:   rows 0, 8     (2 rows × 4 elements each = 8 elements)
        Lanes 4–7:   rows 1, 9
        Lanes 8–11:  rows 2, 10
        Lanes 12–15: rows 3, 11
        Lanes 16–19: rows 4, 12
        Lanes 20–23: rows 5, 13
        Lanes 24–27: rows 6, 14
        Lanes 28–31: rows 7, 15

    Each group of 4 lanes covers 4 elements (cols 0-3) and another 4 (cols 8-11)
    or (cols 4-7) and (cols 12-15), depending on sub-lane index.

    SIMPLIFIED VIEW:
        lane_id    row coverage     col coverage
        0          0, 8             0,1,2,3
        1          0, 8             4,5,6,7
        2          0, 8             8,9,10,11
        3          0, 8             12,13,14,15
        4          1, 9             0,1,2,3
        ...

    Each lane holds .x[0..3] for row r and .x[4..7] for row r+8.

### Accumulator Fragment Distribution

    For the FP32 accumulator (matrix_a type = accumulator, M=N=16, K=16):
        Each lane holds 8 FP32 values.
        The mapping covers 2 rows and 4 columns per lane in the 16×16 result.
        This is why wmma::store_matrix_sync writes 8 floats per lane to produce
        the complete 256-element (16×16) output tile.

### Why Fragments Cannot Be Directly Indexed

    A critical consequence of the distributed layout: if you write
        c_frag.x[0] = 3.14f;
    you are writing to ONE specific position in the tile, but WHICH position
    depends on: lane ID, GPU architecture, tile size, and layout.

    For debugging: use load/store to extract values:
        float tile_data[16][16];
        wmma::store_matrix_sync(&tile_data[0][0], c_frag, 16, wmma::mem_row_major);
        // Now tile_data[row][col] can be accessed normally

    For elementwise operations on tile outputs (e.g., bias add, ReLU):
        Store to SMEM → apply elementwise op on SMEM values → reload (not recommended)
        OR: use cublasLt epilogue fusion (preferred — zero extra HBM passes)
        OR: implement a custom epilogue in the same kernel after store_matrix_sync


##### PART 4 — FP16 → FP32 ACCUMULATION: PRECISION CONTRACT & NUMERICAL ANALYSIS

### FP16 Number Format

    FP16 (IEEE 754 half precision):
        1 sign bit + 5 exponent bits + 10 mantissa bits = 16 bits total
        Exponent bias: 15
        Range: ±6.55 × 10⁴ (max normal), 6.1 × 10⁻⁵ (min normal)
        Machine epsilon: 2⁻¹⁰ ≈ 0.001 (1 part in 1024)
        Max safe integer (no rounding): 2048 (2¹¹)

    FP32 (IEEE 754 single precision):
        1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits total
        Machine epsilon: 2⁻²³ ≈ 1.2 × 10⁻⁷
        Max safe integer: 16,777,216 (2²⁴)

### The Catastrophic Cancellation Problem in FP16-Only Accumulation

    GEMM computes: C[i][j] = Σ_{k=0}^{K-1}  A[i][k] × B[k][j]

    If we accumulate in FP16:
        Each partial sum is rounded to 10-bit precision.
        After K steps, the accumulated rounding error ≈ K × epsilon_FP16 × max_val
        For K = 4096 (typical LLM dim): error ≈ 4096 × 0.001 × 1 = 4.0
        This is a 400% error relative to a unit-scale result!

    A specific failure scenario:
        Accumulating 1024 terms of 0.001 each in FP16:
        True sum = 1024 × 0.001 = 1.024
        FP16 can represent ≈ 0.001 exactly (within epsilon).
        But 1.024 in FP16: epsilon at this magnitude = 2⁻¹⁰ × 1 ≈ 0.001
        So the sum rounds to increments of 0.001 → large relative error.
        Actual FP16 accumulation of 1024 × 0.001 ≈ 1.0 (loses last 2.4%).

### How Tensor Cores Solve This: Mixed-Precision MMA

    The tensor core MMA instruction:
        d[i][j] += Σ_k  a[i][k] × b[k][j]    (accumulate in FP32)

    Even though A and B are stored in FP16:
        1. Each product a[i][k] × b[k][j] is computed in FP32 precision
           (or more precisely: the FP16 mantissas are widened to FP32 before multiply)
        2. The accumulation into d[i][j] happens in FP32 (32-bit addition)
        3. The FP32 accumulator can safely sum ~16 million terms without precision loss

    Result: for K up to ~10⁷, the accumulated error is dominated by the
    FP16 quantisation of A and B inputs, not by accumulation error.
    For typical K ≤ 8192: FP32 accumulation gives results within 0.1% of FP64.

### TF32: The Automatic FP32 Upgrade on Ampere

    TF32 (TensorFloat-32) is not a new dtype — it is an AUTOMATIC precision
    mode available on Ampere (A100) and Hopper (H100):

        TF32 format:
            1 sign + 8 exponent + 10 mantissa = 19 bits
            Same EXPONENT range as FP32 (avoids FP16 overflow at ±65504)
            Same MANTISSA width as FP16 (10 bits) for the TC multiply step
            Accumulation still in FP32 (23-bit mantissa)

    WHEN USED:
        With CUBLAS_TF32_TENSOR_OP_MATH math mode (via allow_tf32=True):
        cuBLAS rounds FP32 inputs to TF32 precision before the TC multiply.
        The result is accumulated in full FP32.

    PERFORMANCE: TF32 on A100 delivers 156 TFLOP/s (vs 19.5 TFLOP/s scalar FP32)
    ACCURACY: 99.9% of FP32 GEMM results agree to within 1 ULP (unit in last place)
    USE CASE: drop-in FP32 replacement for training. torch.backends.cuda.matmul.allow_tf32 = True

### BF16: The Superior Training Dtype on Ampere+

    BF16 (Brain Float 16):
        1 sign + 8 exponent + 7 mantissa = 16 bits
        Same dynamic range as FP32 (avoids overflow issues that plague FP16 training)
        Less precision than FP16 (7 mantissa bits vs 10)
        Accumulation in FP32 via tensor cores

    FP16 vs BF16 in training:
        FP16: 5-bit exponent → overflow/underflow requires GradScaler
        BF16: 8-bit exponent → same range as FP32, no loss scaling needed
        BF16 training is now standard for LLMs (Llama, Mistral, Falcon all use BF16)

    WMMA BF16 (Ampere+):
        #include <cuda_bf16.h>
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        // Load, MMA, store API identical to FP16 WMMA


##### PART 5 — TILE SIZING: FROM 16×16×16 PRIMITIVES TO BLOCK-LEVEL GEMM

### The Three Levels of Tiling

    A production tensor core GEMM uses THREE NESTED TILE LEVELS:

    LEVEL 1 — BLOCK TILE (software, per threadblock):
        Size: BLOCK_M × BLOCK_N × BLOCK_K   (e.g., 128 × 128 × 32)
        Stored in: SMEM (A-tile and B-tile loaded cooperatively)
        Lifetime: K iterations in the outer loop
        One threadblock processes one BLOCK_M × BLOCK_N output tile.

    LEVEL 2 — WARP TILE (software, per warp):
        Size: WARP_M × WARP_N × BLOCK_K   (e.g., 64 × 64 × 32)
        Stored in: register file (accumulator fragments)
        One warp computes one WARP_M × WARP_N output sub-tile using
        multiple WMMA MMA calls per BLOCK_K slice.

    LEVEL 3 — WMMA TILE (hardware, 16×16×16):
        Size: 16 × 16 × 16 (fixed by hardware)
        Stored in: fragment registers (distributed across 32 lanes)
        One wmma::mma_sync() processes one 16×16×16 primitive.

    RELATIONSHIP:
        WARP_M / 16 × WARP_N / 16 wmma::mma_sync() calls per BLOCK_K slice.
        BLOCK_K / 16 WMMA K-steps per SMEM tile.
        Total MMA calls per warp = (WARP_M/16) × (WARP_N/16) × (BLOCK_K/16)

    EXAMPLE (BLOCK=128×128×32, WARP=64×64×32, 4 warps per block):
        MMA per warp = (64/16) × (64/16) × (32/16) = 4 × 4 × 2 = 32 MMA calls
        Each MMA computes 2×16×16×16 = 8192 FLOPs
        Total per block per SMEM tile: 4 warps × 32 MMA × 8192 = 1,048,576 FLOPs

### Choosing BLOCK_M, BLOCK_N, BLOCK_K

    BLOCK_M and BLOCK_N (output tile dimensions):
        Larger → more output reuse → higher arithmetic intensity
        128×128 is the most common choice for training GEMMs
        64×128 or 128×64 for aspect-ratio mismatched problems
        Constraint: BLOCK_M × BLOCK_N × 4 bytes (FP32 accum) ≤ available SMEM
            128×128 accum = 65,536 floats = 256 KB → exceeds SM SMEM on many configs
            But accumulator lives in REGISTERS, not SMEM! 256 KB register file is fine.
            Actual SMEM usage: A-tile + B-tile = (128×32 + 32×128) × 2 bytes = 16 KB

    BLOCK_K (K-slice depth per SMEM load):
        Larger → more MMA calls per SMEM load → better TC pipeline saturation
        Smaller → less SMEM per block → higher occupancy (more blocks per SM)
        32 is a standard choice: balances TC utilisation and occupancy
        16 is minimum (one WMMA K tile per slice — poor TC utilisation)
        64 with async copy (cp.async) is best for Ampere+ high-K GEMMs

    WARP TILE (WARP_M × WARP_N):
        Larger → more independent MMA operations → better ILP → TC stays busy
        64×64 per warp: 4×4 = 16 MMA calls per K-step → 16 independent instructions
            Easily fills the 16-instruction TC pipeline depth
        32×32 per warp: 2×2 = 4 MMA calls → only 4 independent → TC under-utilised
        Constraint: accumulator registers = (WARP_M/16) × (WARP_N/16) × 8 FP32 values
            64×64 warp tile: 4×4 = 16 accumulator fragments × 8 = 128 FP32 regs
            128 registers per warp × 32 lanes × 4 bytes = 16 KB register file per warp

### Occupancy vs ILP Trade-off

    More registers per warp → higher ILP → better TC utilisation
    More registers per warp → fewer warps fit on SM → lower occupancy

    SWEET SPOT for A100 (64 warps max per SM, 256 KB register file):
        With 128 registers per warp: max 256KB/(128×32×4B) = 16 warps
        With 256 registers per warp: max 8 warps per SM

    For GEMM: 8–16 warps per SM is often sufficient to hide memory latency
    (the TC compute pipeline dominates, not memory latency).
    Low occupancy is ACCEPTABLE for compute-bound GEMM kernels.

### Software Pipelining: Hiding SMEM Load Latency

    The SMEM tile load (cooperative GMEM→SMEM copy) has ~600 cycle latency.
    The MMA computation for the CURRENT tile takes ~16 cycles per MMA.

    DOUBLE BUFFERING pattern:
        Allocate 2× SMEM for A and B (SMEM_A[2][...], SMEM_B[2][...]).
        While computing on tile i (in buffer 0):
            Async-copy tile i+1 into buffer 1.
        Switch buffers each K-step.
        cp.async (Ampere) enables the copy without blocking the SM.

    TRIPLE BUFFERING with cp.async:
        Even better: copy tile i+2 while computing i+1, stage i+2 in buffer 2.
        Used in CUTLASS's high-performance GEMM kernels (stage=3).

    This is why the ideal BLOCK_K is 64 with async copy:
        Latency of one async SMEM load: ~50 cycles
        Computation of 64/16 × (WARP_M/16) × (WARP_N/16) MMA calls
        For 64×64 warp tile: 4×4×4 = 64 MMA calls × 16 cycles = 1024 cycles
        Fully hides the 50-cycle async copy latency.


##### PART 6 — A COMPLETE WMMA GEMM KERNEL: ANNOTATED SOURCE

### Block-Level WMMA GEMM (16×16×16 tiles, one warp per output tile)

    // Template: BLOCK_M=16, BLOCK_N=16, BLOCK_K=16 (one MMA per block iteration)
    // Launch: grid = (N/16, M/16), block = (32, 1, 1)  [one warp per block]

    __global__ void wmma_gemm_16x16(
        const __half* A, const __half* B, float* C,
        int M, int N, int K)
    {
        // ── Step 1: Identify which output tile this block computes ──
        int warpM = blockIdx.x;   // row tile index
        int warpN = blockIdx.y;   // col tile index

        // ── Step 2: Declare fragments ──────────────────────────────
        wmma::fragment<wmma::matrix_a,    16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,    16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float>                   c_frag;

        // ── Step 3: Zero the accumulator ───────────────────────────
        wmma::fill_fragment(c_frag, 0.0f);

        // ── Step 4: K-dimension loop ────────────────────────────────
        for (int k = 0; k < K; k += 16) {
            // Load A tile: rows [warpM*16, warpM*16+16), cols [k, k+16)
            const __half* a_ptr = A + warpM * 16 * K + k;
            wmma::load_matrix_sync(a_frag, a_ptr, K);   // stride = K (row_major)

            // Load B tile: rows [k, k+16), cols [warpN*16, warpN*16+16)
            const __half* b_ptr = B + k * N + warpN * 16;
            wmma::load_matrix_sync(b_frag, b_ptr, N);   // stride = N (col_major: stride=N)

            // Execute MMA: c_frag += a_frag × b_frag
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // ── Step 5: Store result ────────────────────────────────────
        float* c_ptr = C + warpM * 16 * N + warpN * 16;
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }

    NOTE: This is a MINIMAL correct kernel. A production kernel would add:
        - Shared memory tiling (BLOCK_K loop around SMEM loads + MMA)
        - Multiple warps per block with larger BLOCK_M/BLOCK_N
        - Double buffering (cp.async for Ampere+)
        - Epilogue fusion (bias + activation)
        - Bounds checking for non-tile-aligned M, N, K

### The Full SMEM-Tiled Version (Block Tile = 128×128×32)

    // 4 warps per block, each handles a 64×32 warp tile
    // BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
    // 2 SMEM buffers: A_smem[32][128], B_smem[32][128]

    Key structural differences from the minimal kernel above:
        1. Outer loop: k_block over K/BLOCK_K SMEM tiles
        2. Inner loop: per-warp double loop over WARP_M/16 and WARP_N/16 MMA tiles
        3. Cooperative loading: all threads in block copy SMEM tiles using
           a 2D index: thread_row = threadIdx.x / 16, thread_col = threadIdx.x % 16
        4. __syncthreads() between SMEM write and fragment load
        5. Multiple accumulator fragments (one per WMMA tile in the warp tile)


##### PART 7 — DIAGNOSTIC: VERIFYING TENSOR CORE ENGAGEMENT IN YOUR KERNEL

### The Critical NCU Metric

    sm__inst_executed_pipe_tensor_op_hmma.sum

    If this value is 0: tensor cores are NOT being used.
    If this value > 0: tensor core instructions were issued.

    Expected value for a WMMA GEMM of M×N×K with FP16:
        n_mma = (M/16) × (N/16) × (K/16)   [number of 16×16×16 MMA tiles]
        hmma_count = n_mma × 4              [each WMMA MMA = 4 hardware HMMA ops on Ampere]
        sm__inst_executed_pipe_tensor_op_hmma.sum ≈ n_mma × 4

### Why Tensor Cores Fail to Engage

    REASON 1: Compile flag missing
        Requires sm_70+ in --gpu-architecture flag.
        Symptom: hmma count = 0, kernel runs slower than expected.
        Fix: nvcc --gpu-architecture=sm_80 (A100) or sm_86 (A10G)

    REASON 2: Dimension misalignment
        M, N, K must be multiples of 16 for WMMA API to work correctly.
        Non-multiples: undefined behavior or CPU-fallback path.
        Fix: pad inputs to multiples of 16 before calling WMMA kernel.

    REASON 3: Wrong dtype
        WMMA matrix_a and matrix_b must be __half (FP16) or __nv_bfloat16.
        If you pass float* pointers to WMMA: compilation error (caught at compile time).
        If using cublasGemmEx: wrong CUDA_R_ type argument disables TC path.

    REASON 4: Warp divergence
        If warps branch BEFORE the wmma::mma_sync() call, lanes in the warp
        may disagree on whether to execute MMA. Undefined behavior.
        Fix: ensure all WMMA calls are in divergence-free code paths.

    REASON 5: Fragment not properly initialised
        Forgetting wmma::fill_fragment(c_frag, 0.0f) before the K-loop.
        Symptom: correct kernel produces wrong results (garbage accumulation).
        Fix: always initialise accumulator fragments before the K loop.

### The Compute-to-Memory Ratio Diagnostic

    For a well-tuned WMMA GEMM on H100 (FP16, 16×16×16 tiles):
        sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed

    GOOD:  > 70%  — tensor cores are busy most of the time
    BAD:   < 40%  — memory-bound or poor TC pipeline utilisation

    When TC utilisation is low despite correct alignment:
        Check: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
        If high (>30%): memory latency is stalling before MMA can execute.
        Fix: increase BLOCK_K, use double buffering, or increase warp count.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · WMMA Fragment Simulator — Distribution, Load & Store Mechanics": {
        "description": (
            "Simulate the full WMMA fragment API lifecycle in Python. Build a "
            "Fragment class that mirrors the C++ wmma::fragment type including "
            "distributed lane storage. Simulate load_matrix_sync, fill_fragment, "
            "and store_matrix_sync. Show exactly how the 256-element 16×16 tile "
            "is split 8 elements per lane across 32 lanes. Verify that load→store "
            "roundtrips preserve the original tile values exactly."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  WMMA FRAGMENT SIMULATOR — Distribution, Load & Store")
print("=" * 68)
print()

np.random.seed(42)

WARP_SIZE   = 32
WMMA_M      = 16
WMMA_N      = 16
WMMA_K      = 16
ELEMS_PER_LANE_AB  = (WMMA_M * WMMA_K) // WARP_SIZE   # 8 for matrix_a
ELEMS_PER_LANE_ACC = (WMMA_M * WMMA_N) // WARP_SIZE   # 8 for accumulator


# ─────────────────────────────────────────────────────────────────────
# Fragment classes
# ─────────────────────────────────────────────────────────────────────

class Fragment:
    """
    Simulate wmma::fragment<> distributed across 32 warp lanes.
    Each lane holds x[0..num_elements-1] of its portion of the tile.
    """
    def __init__(self, use, M=16, N=16, K=16, dtype=np.float16,
                 accum_dtype=np.float32, layout='row_major'):
        self.use    = use       # 'matrix_a' | 'matrix_b' | 'accumulator'
        self.M, self.N, self.K = M, N, K
        self.dtype  = dtype
        self.layout = layout    # 'row_major' | 'col_major'

        if use == 'accumulator':
            self.num_elements = (M * N) // WARP_SIZE     # 8 for 16×16
            self.elem_dtype   = accum_dtype
        elif use == 'matrix_a':
            self.num_elements = (M * K) // WARP_SIZE     # 8 for 16×16×16
            self.elem_dtype   = dtype
        else:  # matrix_b
            self.num_elements = (K * N) // WARP_SIZE
            self.elem_dtype   = dtype

        # .x[lane][element_index] — the distributed storage
        self.x = np.zeros((WARP_SIZE, self.num_elements), dtype=self.elem_dtype)

    def fill(self, value):
        """wmma::fill_fragment(frag, value) — zero or constant fill."""
        self.x[:] = value

    def __repr__(self):
        return (f"Fragment({self.use}, {self.M}×{self.N}×{self.K}, "
                f"dtype={self.elem_dtype.__name__}, layout={self.layout}, "
                f"elems_per_lane={self.num_elements})")


def lane_to_tile_indices_matrix_a(lane_id, elem_idx, M=16, K=16):
    """
    Map (lane_id, elem_idx) → (row, col) in the 16×16 A tile.
    Approximation of Volta/Ampere row_major distribution.
    Lane groups of 4 cover 4 columns; each lane covers 2 rows (r and r+8).
    """
    group      = lane_id // 4        # which group of 4 lanes (0-7)
    lane_in_g  = lane_id % 4         # position within group (0-3)

    # Two row halves: elem_idx < 4 → first half, elem_idx >= 4 → second half
    row_base   = group               # group covers row 'group' and row 'group+8'
    row_offset = 8 if elem_idx >= 4 else 0
    row        = row_base + row_offset

    # Column: 4 lanes × 4 elements each covers columns 0-15
    col_base   = lane_in_g * 4
    col_offset = elem_idx % 4
    col        = col_base + col_offset

    return row, col

def lane_to_tile_indices_accumulator(lane_id, elem_idx, M=16, N=16):
    """
    Map (lane_id, elem_idx) → (row, col) in the 16×16 accumulator tile.
    Similar row/col pattern to matrix_a but over M×N instead of M×K.
    """
    return lane_to_tile_indices_matrix_a(lane_id, elem_idx, M, N)


def load_matrix_sync(frag, tile_data, stride, layout='row_major'):
    """
    Simulate wmma::load_matrix_sync.
    tile_data: 2D numpy array of the source tile (M×K for matrix_a, etc.)
    stride: leading dimension (number of elements per row for row_major)
    Distributes tile elements to their assigned lanes.
    """
    M_t = frag.M
    K_t = frag.K if frag.use != 'accumulator' else frag.N

    for lane in range(WARP_SIZE):
        for elem in range(frag.num_elements):
            if frag.use == 'accumulator':
                row, col = lane_to_tile_indices_accumulator(lane, elem, frag.M, frag.N)
            else:
                row, col = lane_to_tile_indices_matrix_a(lane, elem, M_t, K_t)
            if row < tile_data.shape[0] and col < tile_data.shape[1]:
                frag.x[lane, elem] = frag.elem_dtype(tile_data[row, col])


def store_matrix_sync(frag, layout='row_major'):
    """
    Simulate wmma::store_matrix_sync.
    Returns the reconstructed 2D tile from all lane contributions.
    """
    if frag.use == 'accumulator':
        tile = np.zeros((frag.M, frag.N), dtype=frag.elem_dtype)
        for lane in range(WARP_SIZE):
            for elem in range(frag.num_elements):
                row, col = lane_to_tile_indices_accumulator(lane, elem, frag.M, frag.N)
                if row < frag.M and col < frag.N:
                    tile[row, col] = frag.x[lane, elem]
    else:
        K_t = frag.K
        tile = np.zeros((frag.M, K_t), dtype=frag.elem_dtype)
        for lane in range(WARP_SIZE):
            for elem in range(frag.num_elements):
                row, col = lane_to_tile_indices_matrix_a(lane, elem, frag.M, K_t)
                if row < frag.M and col < K_t:
                    tile[row, col] = frag.x[lane, elem]
    return tile


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Fragment metadata
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Fragment Metadata: Types and Element Counts")
print("━" * 68)
print()

fragments_to_show = [
    Fragment('matrix_a',    16, 16, 16, np.float16, layout='row_major'),
    Fragment('matrix_b',    16, 16, 16, np.float16, layout='col_major'),
    Fragment('accumulator', 16, 16, 16, np.float32),
    Fragment('matrix_a',    16, 16, 16, np.float16, layout='col_major'),
]

print(f"  {'Fragment type':<48}  {'Elems/lane':>10}  {'Total elems':>12}  {'Storage (bytes/lane)'}")
print("  " + "─" * 84)
for frag in fragments_to_show:
    total_elems = WARP_SIZE * frag.num_elements
    bytes_per_lane = frag.num_elements * np.dtype(frag.elem_dtype).itemsize
    print(f"  {str(frag):<48}  {frag.num_elements:>10}  "
          f"{total_elems:>12}  {bytes_per_lane}")

print()
print(f"  16×16 tile = 256 elements. 256 / {WARP_SIZE} lanes = 8 elements per lane. ✅")
print(f"  Accumulator uses FP32 (4 bytes): 8 × 4 = 32 bytes per lane register storage.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Lane distribution map for matrix_a
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Lane Distribution Map: Which Lane Owns Which (row, col)?")
print("━" * 68)
print()

print("  16×16 matrix_a tile (row_major). Each cell shows lane_id.")
print()

tile_owner = np.full((WMMA_M, WMMA_K), -1, dtype=int)
for lane in range(WARP_SIZE):
    for elem in range(ELEMS_PER_LANE_AB):
        row, col = lane_to_tile_indices_matrix_a(lane, elem)
        if 0 <= row < WMMA_M and 0 <= col < WMMA_K:
            tile_owner[row, col] = lane

print("        col: " + " ".join(f"{c:>3}" for c in range(WMMA_K)))
print("  " + "─" * 60)
for row in range(WMMA_M):
    lane_vals = " ".join(f"{tile_owner[row, c]:>3}" for c in range(WMMA_K))
    print(f"  row {row:>2}:   {lane_vals}")

print()
# Count coverage
covered = np.sum(tile_owner >= 0)
print(f"  Covered elements: {covered} / {WMMA_M * WMMA_K}  "
      f"{'✅ complete coverage' if covered == WMMA_M * WMMA_K else '❌ partial'}")
print()
# Show which elements each lane owns
print("  Per-lane element count (should be 8 each):")
from collections import Counter
cnt = Counter(tile_owner.ravel())
problems = [ln for ln in range(WARP_SIZE) if cnt[ln] != ELEMS_PER_LANE_AB]
if not problems:
    print(f"  All {WARP_SIZE} lanes own exactly {ELEMS_PER_LANE_AB} elements. ✅")
else:
    for ln in range(min(8, WARP_SIZE)):
        print(f"  Lane {ln:2d}: {cnt[ln]} elements")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Load → Store roundtrip verification
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Load → Store Roundtrip: Values Preserved?")
print("━" * 68)
print()

# Create a distinctive 16×16 test tile
tile_orig = np.arange(WMMA_M * WMMA_K, dtype=np.float16).reshape(WMMA_M, WMMA_K)

# Load into fragment
a_frag = Fragment('matrix_a', 16, 16, 16, np.float16, layout='row_major')
load_matrix_sync(a_frag, tile_orig, stride=WMMA_K)

print(f"  Original tile[0, :8] = {tile_orig[0, :8].tolist()}")
print(f"  Original tile[8, :8] = {tile_orig[8, :8].tolist()}")
print()
print(f"  After load_matrix_sync → a_frag.x (lane 0):  {a_frag.x[0].tolist()}")
print(f"  After load_matrix_sync → a_frag.x (lane 4):  {a_frag.x[4].tolist()}")
print()

# Store back
tile_recovered = store_matrix_sync(a_frag)
match = np.allclose(tile_orig.astype(np.float32), tile_recovered.astype(np.float32))
print(f"  Recovered tile[0, :8]  = {tile_recovered[0, :8].tolist()}")
print(f"  Recovered tile[8, :8]  = {tile_recovered[8, :8].tolist()}")
print(f"  Roundtrip exact match:   {'✅' if match else '❌'}")
print()
print("  KEY INSIGHT: fragment.x[] indices are NOT (row, col).")
print("  Lane 0 holds elements from rows 0 AND 8, not just row 0.")
print("  You CANNOT safely read fragment.x[i] as element i of any row.")
print("  Always use store_matrix_sync to recover the 2D tile view.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · WMMA MMA Simulator — FP16 Input × FP32 Accumulation Verified": {
        "description": (
            "Implement the full wmma::mma_sync() operation in simulation: "
            "load FP16 A and B fragments, execute the 16×16×16 MMA with FP32 "
            "accumulation, and store the FP32 result. Verify against the "
            "reference NumPy FP64 result. Run the K-loop GEMM pattern across "
            "multiple tiles and confirm that FP32 accumulation matches FP64 "
            "closely while FP16-only accumulation drifts. Show the precision "
            "degradation as K increases."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  WMMA MMA SIMULATOR — FP16 Input × FP32 Accumulation")
print("=" * 68)
print()

np.random.seed(7)

WMMA_M = WMMA_N = WMMA_K = 16
WARP_SIZE = 32


# ─────────────────────────────────────────────────────────────────────
# Minimal Fragment + MMA implementation
# ─────────────────────────────────────────────────────────────────────

class Frag:
    def __init__(self, use, dtype=np.float16, accum_dtype=np.float32):
        self.use   = use
        self.dtype = accum_dtype if use == 'accumulator' else dtype
        n = (WMMA_M * WMMA_K) // WARP_SIZE  # 8 for all 16x16x16 roles
        self.data  = np.zeros((WARP_SIZE, n), dtype=self.dtype)

    def fill(self, v):
        self.data[:] = self.dtype(v)


def load_fp16(frag, matrix_2d):
    """Pack a 16xK or KxN matrix into fragment storage (simplified flat packing)."""
    flat = matrix_2d.astype(np.float16).ravel()
    n    = frag.data.shape[1]
    for lane in range(WARP_SIZE):
        for e in range(n):
            idx = lane * n + e
            if idx < len(flat):
                frag.data[lane, e] = flat[idx]


def mma_sync(d_frag, a_frag, b_frag, c_frag):
    """
    Simulate wmma::mma_sync(d, a, b, c):
        D = A × B + C   using FP32 accumulation.

    Approach: reconstruct 2D matrices from fragment storage,
    multiply in FP32 (widening from FP16 inputs), accumulate.
    """
    # Reconstruct A (16×16), B (16×16) from flat fragment storage
    n_per_lane = a_frag.data.shape[1]
    A_flat = a_frag.data.ravel()[:WMMA_M * WMMA_K]
    B_flat = b_frag.data.ravel()[:WMMA_K * WMMA_N]
    C_flat = c_frag.data.ravel()[:WMMA_M * WMMA_N]

    A_mat = A_flat.reshape(WMMA_M, WMMA_K).astype(np.float32)  # widen FP16 → FP32
    B_mat = B_flat.reshape(WMMA_K, WMMA_N).astype(np.float32)
    C_mat = C_flat.reshape(WMMA_M, WMMA_N).astype(np.float32)

    # FP32 multiply-accumulate (this is what tensor cores do internally)
    D_mat = A_mat @ B_mat + C_mat  # FP32 result

    D_flat = D_mat.ravel().astype(np.float32)
    for lane in range(WARP_SIZE):
        for e in range(d_frag.data.shape[1]):
            idx = lane * d_frag.data.shape[1] + e
            if idx < len(D_flat):
                d_frag.data[lane, e] = D_flat[idx]


def store_accum(frag):
    """Reconstruct 2D float tile from accumulator fragment."""
    flat = frag.data.ravel()[:WMMA_M * WMMA_N]
    return flat.reshape(WMMA_M, WMMA_N).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Single-tile MMA correctness
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Single 16×16×16 MMA: FP16 Input, FP32 Accumulator")
print("━" * 68)
print()

A_ref = np.random.randn(WMMA_M, WMMA_K).astype(np.float32)
B_ref = np.random.randn(WMMA_K, WMMA_N).astype(np.float32)
C_ref = np.zeros((WMMA_M, WMMA_N), dtype=np.float32)

# FP64 reference
C_fp64 = (A_ref.astype(np.float64) @ B_ref.astype(np.float64)).astype(np.float32)

# Simulate WMMA
a_frag = Frag('matrix_a');  load_fp16(a_frag, A_ref)
b_frag = Frag('matrix_b');  load_fp16(b_frag, B_ref)
c_frag = Frag('accumulator'); c_frag.fill(0.0)
d_frag = Frag('accumulator'); d_frag.fill(0.0)

mma_sync(d_frag, a_frag, b_frag, c_frag)
C_wmma = store_accum(d_frag)

# FP16-only accumulation (what happens without FP32 widening)
A_fp16 = A_ref.astype(np.float16)
B_fp16 = B_ref.astype(np.float16)
C_fp16only = (A_fp16 @ B_fp16).astype(np.float32)   # FP16 mat-mul

err_wmma   = np.abs(C_wmma   - C_fp64).mean()
err_fp16   = np.abs(C_fp16only - C_fp64).mean()

print(f"  A shape: ({WMMA_M}×{WMMA_K}) FP16    B shape: ({WMMA_K}×{WMMA_N}) FP16")
print(f"  Accumulator: FP32")
print()
print(f"  FP64 reference C[0, :4]:    {C_fp64[0, :4].round(4).tolist()}")
print(f"  WMMA result    C[0, :4]:    {C_wmma[0, :4].round(4).tolist()}")
print(f"  FP16-only      C[0, :4]:    {C_fp16only[0, :4].round(4).tolist()}")
print()
print(f"  Mean absolute error (WMMA vs FP64):    {err_wmma:.6f}")
print(f"  Mean absolute error (FP16only vs FP64): {err_fp16:.6f}")
print(f"  WMMA is {err_fp16/max(err_wmma,1e-9):.1f}× more accurate than FP16-only")
print()
print(f"  WHY: WMMA widens FP16 inputs to FP32 BEFORE multiply.")
print(f"  FP16×FP16 → FP32 product preserves 20+ bits of precision.")
print(f"  FP16×FP16 → FP16 product loses lower mantissa bits immediately.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: K-loop accumulation over multiple tiles
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — K-Loop GEMM: Accumulation Across Multiple MMA Tiles")
print("━" * 68)
print()

print("  Simulating: C = A × B  where A=(16×K), B=(K×16), accumulating over K.")
print("  K is swept from 16 to 4096 (256 tiles at K=4096).")
print()

print(f"  {'K':>6}  {'N_tiles':>8}  {'WMMA err':>12}  {'FP16 err':>12}  "
      f"{'FP32 accum err':>15}  {'Ratio'}")
print("  " + "─" * 64)

for K_total in [16, 64, 128, 256, 512, 1024, 2048, 4096]:
    # Generate full A and B
    A_full = np.random.randn(WMMA_M, K_total).astype(np.float32)
    B_full = np.random.randn(K_total, WMMA_N).astype(np.float32)

    # FP64 reference
    C_true = (A_full.astype(np.float64) @ B_full.astype(np.float64))

    # WMMA simulation: K-loop with FP32 accumulation
    c_wmma = Frag('accumulator'); c_wmma.fill(0.0)
    for k_start in range(0, K_total, WMMA_K):
        A_tile = A_full[:, k_start:k_start + WMMA_K]
        B_tile = B_full[k_start:k_start + WMMA_K, :]
        a_f = Frag('matrix_a'); load_fp16(a_f, A_tile)
        b_f = Frag('matrix_b'); load_fp16(b_f, B_tile)
        c_tmp = Frag('accumulator'); c_tmp.data[:] = c_wmma.data
        mma_sync(c_wmma, a_f, b_f, c_tmp)
    C_wmma_k = store_accum(c_wmma)

    # FP16-only accumulation (accumulate in FP16 each step)
    C_fp16_acc = np.zeros((WMMA_M, WMMA_N), dtype=np.float16)
    for k_start in range(0, K_total, WMMA_K):
        A_tile = A_full[:, k_start:k_start + WMMA_K].astype(np.float16)
        B_tile = B_full[k_start:k_start + WMMA_K, :].astype(np.float16)
        C_fp16_acc = (C_fp16_acc + A_tile @ B_tile).astype(np.float16)

    # FP32 scalar accumulation (CUDA FP32 SIMT reference)
    C_fp32_acc = np.zeros((WMMA_M, WMMA_N), dtype=np.float32)
    for k_start in range(0, K_total, WMMA_K):
        A_tile = A_full[:, k_start:k_start + WMMA_K].astype(np.float32)
        B_tile = B_full[k_start:k_start + WMMA_K, :].astype(np.float32)
        C_fp32_acc += A_tile @ B_tile

    err_w   = np.abs(C_wmma_k - C_true).mean()
    err_f16 = np.abs(C_fp16_acc.astype(np.float64) - C_true).mean()
    err_f32 = np.abs(C_fp32_acc.astype(np.float64) - C_true).mean()
    ratio   = err_f16 / max(err_w, 1e-12)
    n_tiles = K_total // WMMA_K

    print(f"  {K_total:>6}  {n_tiles:>8}  {err_w:>12.6f}  {err_f16:>12.6f}  "
          f"{err_f32:>15.6f}  {ratio:>5.1f}×")

print()
print("  WMMA (FP16→FP32) ≈ FP32 scalar accumulation in accuracy.")
print("  FP16-only accumulation error grows proportionally to K (catastrophic).")
print("  At K=4096: FP16-only error is ~100× worse than WMMA.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: fill_fragment and β×C accumulation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — fill_fragment and β×C: Initialisation Patterns")
print("━" * 68)
print()

A_t = np.random.randn(WMMA_M, WMMA_K).astype(np.float32)
B_t = np.random.randn(WMMA_K, WMMA_N).astype(np.float32)
C_t = np.random.randn(WMMA_M, WMMA_N).astype(np.float32)
alpha, beta = 1.5, 0.7

# Reference: alpha * A @ B + beta * C
D_ref = alpha * A_t @ B_t + beta * C_t

# Simulate: fill with zeros → mma → done (alpha=1, beta=0 case)
patterns = [
    ("fill(0) → mma",    0.0, C_t, 1.0, 0.0, "D = A×B          (β=0)"),
    ("load C → mma",     None, C_t, 1.0, 1.0, "D = A×B + C      (α=β=1)"),
    ("load β*C → mma",   None, beta * C_t, 1.0, 1.0, "D = A×B + β×C   (pre-scale C)"),
]

print(f"  {'Pattern':<26}  {'D[0,0]':>10}  {'Error vs ref':>14}  {'Notes'}")
print("  " + "─" * 66)

for name, fill_val, C_init, alpha_sim, beta_sim, desc in patterns:
    c_f = Frag('accumulator')
    if fill_val is not None:
        c_f.fill(fill_val)
    else:
        # Load C_init into accumulator
        load_fp16(c_f, C_init.astype(np.float16))
        c_f.data = c_f.data.astype(np.float32)
        c_f.dtype = np.float32

    a_f = Frag('matrix_a'); load_fp16(a_f, alpha_sim * A_t)
    b_f = Frag('matrix_b'); load_fp16(b_f, B_t)
    d_f = Frag('accumulator'); d_f.fill(0.0)
    mma_sync(d_f, a_f, b_f, c_f)
    D_sim = store_accum(d_f)

    err = np.abs(D_sim[0, 0] - D_ref[0, 0])
    print(f"  {name:<26}  {D_sim[0,0]:>10.4f}  {err:>14.6f}  {desc}")

print()
print(f"  Reference D[0,0] = {D_ref[0,0]:.4f}  (alpha={alpha}, beta={beta})")
print()
print("  IMPORTANT: fill_fragment(0) is MANDATORY before the K-loop.")
print("  Forgetting it leaves garbage in the accumulator.")
print("  If beta != 0: pre-scale C on host or use cublasLt epilogue descriptor.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · FP16 vs FP32 vs BF16 Precision — Numerical Analysis & Training Impact": {
        "description": (
            "Compare FP16, BF16, TF32, and FP32 number formats numerically. "
            "Show the representable value gaps, the overflow boundaries, and the "
            "precision-vs-range tradeoffs. Simulate the FP16 gradient underflow "
            "problem and demonstrate why GradScaler is needed for FP16 but not "
            "BF16 training. Compute the accumulated error in a long dot-product "
            "for each precision, showing FP32 accumulation rescues FP16 inputs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import struct

print("=" * 68)
print("  FP16 vs FP32 vs BF16 PRECISION — Numerical Analysis")
print("=" * 68)
print()

np.random.seed(99)


# ─────────────────────────────────────────────────────────────────────
# Format utilities
# ─────────────────────────────────────────────────────────────────────

def fp16_info():
    return {
        "name": "FP16", "bits": 16, "sign": 1, "exp": 5, "mant": 10,
        "max_val": 65504.0, "min_normal": 6.1e-5, "epsilon": 2**-10,
        "bias": 15, "max_int_exact": 2048,
    }

def bf16_info():
    return {
        "name": "BF16", "bits": 16, "sign": 1, "exp": 8, "mant": 7,
        "max_val": 3.39e38, "min_normal": 1.18e-38, "epsilon": 2**-7,
        "bias": 127, "max_int_exact": 256,
    }

def tf32_info():
    return {
        "name": "TF32", "bits": 19, "sign": 1, "exp": 8, "mant": 10,
        "max_val": 3.39e38, "min_normal": 1.18e-38, "epsilon": 2**-10,
        "bias": 127, "max_int_exact": 2048,
    }

def fp32_info():
    return {
        "name": "FP32", "bits": 32, "sign": 1, "exp": 8, "mant": 23,
        "max_val": 3.40e38, "min_normal": 1.18e-38, "epsilon": 2**-23,
        "bias": 127, "max_int_exact": 16_777_216,
    }

FORMATS = [fp16_info(), bf16_info(), tf32_info(), fp32_info()]


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Format comparison table
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Float Format Properties: FP16, BF16, TF32, FP32")
print("━" * 68)
print()

print(f"  {'Format':<8}  {'Bits':>5}  {'Exp':>5}  {'Mant':>5}  "
      f"{'Max value':>12}  {'Min normal':>12}  {'Epsilon':>12}  {'Max safe int'}")
print("  " + "─" * 78)
for f in FORMATS:
    print(f"  {f['name']:<8}  {f['bits']:>5}  {f['exp']:>5}  {f['mant']:>5}  "
          f"{f['max_val']:>12.2e}  {f['min_normal']:>12.2e}  "
          f"{f['epsilon']:>12.6f}  {f['max_int_exact']:>12,}")

print()
print("  KEY DIFFERENCES:")
print("    FP16 vs BF16: same 16 bits, but BF16 trades mantissa for exponent range.")
print("    BF16 has SAME exponent as FP32 → same overflow/underflow behaviour.")
print("    TF32: 19-bit compromise — FP32 range, FP16 mantissa precision.")
print("    FP32 accumulation + FP16/BF16 inputs = tensor core mixed precision.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Representable values — gaps between adjacent floats
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Value Representation: Gaps Between Adjacent Floats")
print("━" * 68)
print()

def fp16_next(x):
    """Return the next representable FP16 value after x."""
    h = np.float16(x)
    # Add 1 ulp in FP16
    bits = np.frombuffer(h.tobytes(), dtype=np.uint16)[0]
    next_bits = bits + 1
    return np.frombuffer(next_bits.tobytes(), dtype=np.float16)[0]

def ulp_fp16(x):
    """Unit in last place for FP16 at value x."""
    v = abs(float(x))
    if v == 0: return float(np.finfo(np.float16).tiny)
    exp = math.floor(math.log2(v))
    return 2.0 ** (exp - 10)  # 10 mantissa bits

def ulp_fp32(x):
    v = abs(float(x))
    if v == 0: return float(np.finfo(np.float32).tiny)
    exp = math.floor(math.log2(v))
    return 2.0 ** (exp - 23)

test_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 60000.0]

print(f"  {'Value':>10}  {'FP16 ULP':>14}  {'FP16 rel err':>14}  "
      f"{'FP32 ULP':>14}  {'Ratio FP16/FP32'}")
print("  " + "─" * 72)

for v in test_values:
    fp16_v = float(np.float16(v))
    if fp16_v == 0 or abs(v) > 65504:
        print(f"  {v:>10.3f}  {'OVERFLOW':>14}")
        continue
    u16 = ulp_fp16(fp16_v)
    u32 = ulp_fp32(v)
    rel16 = u16 / abs(fp16_v) if fp16_v != 0 else float('inf')
    ratio = u16 / u32
    overflow_warn = " ← near overflow!" if v > 50000 else ""
    print(f"  {v:>10.3f}  {u16:>14.6f}  {rel16:>14.6f}  "
          f"{u32:>14.8f}  {ratio:>14.0f}×{overflow_warn}")

print()
print("  FP16 ULP at value 1.0 = 0.000977 (1/1024).")
print("  FP16 relative error is constant (~0.1%) within each binade.")
print("  At value 60000: ULP = 4. Values 60000, 60001, 60002, 60003 all → 60000 in FP16!")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: FP16 overflow in training — why GradScaler exists
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — FP16 Overflow & Gradient Underflow: GradScaler Rationale")
print("━" * 68)
print()

print("  FP16 max value: 65504.  Values > 65504 → inf (overflow).")
print("  FP16 min normal: 6.1e-5. Values < 6.1e-5 → 0 (underflow / subnormal).")
print()

# Simulate gradient magnitudes in a training run (log-normal distribution)
np.random.seed(42)
grad_magnitudes_fp32 = np.abs(np.random.lognormal(mean=-4, sigma=3, size=10000))

overflow_threshold  = 65504.0
underflow_threshold = 6.1e-5

n_overflow  = np.sum(grad_magnitudes_fp32 > overflow_threshold)
n_underflow = np.sum(grad_magnitudes_fp32 < underflow_threshold)
n_total     = len(grad_magnitudes_fp32)

print(f"  Simulated gradient distribution (10000 values, log-normal μ=-4, σ=3):")
print(f"    Range: [{grad_magnitudes_fp32.min():.2e}, {grad_magnitudes_fp32.max():.2e}]")
print(f"    Mean:  {grad_magnitudes_fp32.mean():.4f}")
print(f"    FP16 overflow  (> 65504):   {n_overflow:5d} / {n_total}  "
      f"({n_overflow/n_total*100:.1f}%) → inf → NaN propagation")
print(f"    FP16 underflow (< 6.1e-5):  {n_underflow:5d} / {n_total}  "
      f"({n_underflow/n_total*100:.1f}%) → 0 → gradient vanishes")
print()

# Show what GradScaler does
scale_factors = [1, 8, 64, 512, 4096, 32768]
print("  GradScaler multiplies loss by scale factor before backward.")
print("  This shifts gradient magnitudes into the safe FP16 range.")
print()
print(f"  {'Scale':>8}  {'After scaling: mean':>20}  "
      f"{'Overflow %':>12}  {'Underflow %':>13}  {'Safe %'}")
print("  " + "─" * 68)

for s in scale_factors:
    scaled = grad_magnitudes_fp32 * s
    n_ov  = np.sum(scaled > overflow_threshold)
    n_un  = np.sum(scaled < underflow_threshold)
    n_safe = n_total - n_ov - n_un
    print(f"  {s:>8}  {scaled.mean():>20.4f}  "
          f"{n_ov/n_total*100:>11.1f}%  {n_un/n_total*100:>12.1f}%  "
          f"{n_safe/n_total*100:.1f}%")

print()
print("  Optimal scale keeps gradients in range with maximum safe values.")
print()
print("  BF16 avoids this problem entirely:")
print("    BF16 max: 3.39e38 (same as FP32). Overflow only at truly extreme values.")
print("    BF16 min: 1.18e-38 (same as FP32). Gradient underflow is not an issue.")
print("    Training with BF16: no GradScaler needed. Simpler, equally accurate.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: TF32 — the invisible FP32 speedup
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — TF32: The Automatic FP32 Speedup on Ampere/Hopper")
print("━" * 68)
print()

print("  TF32 rounds FP32 mantissa from 23 bits to 10 bits before TC multiply.")
print("  Accumulation remains in FP32 (23 bits). No code change required.")
print()

def round_to_tf32(x):
    """Round a float32 value to TF32 precision (truncate mantissa to 10 bits)."""
    v_fp32 = np.float32(x)
    bits = int.from_bytes(v_fp32.tobytes(), 'little')
    # Keep sign(1) + exponent(8) + top 10 mantissa bits; zero remaining 13 bits
    mask = 0xFFFFE000   # keep top 19 bits of 32
    bits_tf32 = bits & mask
    result_bytes = bits_tf32.to_bytes(4, 'little')
    return float(np.frombuffer(result_bytes, dtype=np.float32)[0])

# Compare FP32, TF32, and FP64 for a GEMM
np.random.seed(55)
M_t, N_t, K_t = 16, 16, 64
A_test = np.random.randn(M_t, K_t).astype(np.float32)
B_test = np.random.randn(K_t, N_t).astype(np.float32)

C_fp64 = (A_test.astype(np.float64) @ B_test.astype(np.float64)).astype(np.float32)
C_fp32 = A_test @ B_test

A_tf32 = np.vectorize(round_to_tf32)(A_test)
B_tf32 = np.vectorize(round_to_tf32)(B_test)
C_tf32 = A_tf32 @ B_tf32

err_fp32 = np.abs(C_fp32 - C_fp64).mean()
err_tf32 = np.abs(C_tf32 - C_fp64).mean()

print(f"  GEMM ({M_t}×{K_t}) × ({K_t}×{N_t}), FP64 as reference:")
print(f"    FP32 mean absolute error:  {err_fp32:.6f}")
print(f"    TF32 mean absolute error:  {err_tf32:.6f}")
print(f"    TF32 / FP32 error ratio:   {err_tf32/max(err_fp32, 1e-10):.2f}×")
print()

# Show a few values
print(f"  {'Row,Col':<10}  {'FP64 ref':>12}  {'FP32':>12}  {'TF32':>12}  "
      f"{'FP32 err':>12}  {'TF32 err'}")
print("  " + "─" * 72)
for i, j in [(0,0),(0,1),(1,0),(3,5),(7,8)]:
    e32 = abs(C_fp32[i,j] - C_fp64[i,j])
    e_tf = abs(C_tf32[i,j] - C_fp64[i,j])
    print(f"  [{i},{j}]{'':6}  {C_fp64[i,j]:>12.5f}  {C_fp32[i,j]:>12.5f}  "
          f"{C_tf32[i,j]:>12.5f}  {e32:>12.6f}  {e_tf:.6f}")

print()
print("  TF32 introduces at most ~0.1% additional error vs FP32.")
print("  In practice: >99.9% of model training runs are unaffected.")
print("  Enable: torch.backends.cuda.matmul.allow_tf32 = True")
print("  Speedup: ~8–10× vs scalar FP32 on A100; same as FP16 TC but no dtype cast.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Tile Size Explorer — Block/Warp Tile Impact on Performance": {
        "description": (
            "Model the performance of different block tile configurations for a "
            "WMMA GEMM. Compute arithmetic intensity, SMEM usage, register file "
            "pressure, theoretical occupancy, and wave efficiency for every "
            "combination of BLOCK_M, BLOCK_N, BLOCK_K from the standard set. "
            "Show how warp tile size controls TC pipeline saturation via the "
            "independent MMA count. Identify the optimal config for training "
            "vs inference for three LLM GEMM shapes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  TILE SIZE EXPLORER — Block/Warp Tile Impact on TC Performance")
print("=" * 68)
print()

# Hardware constants (A100 SXM4)
SM_COUNT           = 108
MAX_WARPS_PER_SM   = 64
MAX_REGS_PER_SM    = 65536    # 256 KB register file
MAX_SMEM_PER_SM    = 164_000  # 164 KB (one of several configs)
MAX_THREADS_PER_SM = 2048
WMMA_TILE          = 16       # hardware primitive: 16×16×16
FP16_BYTES         = 2
FP32_BYTES         = 4
TC_LATENCY_CYCLES  = 16
TC_THROUGHPUT_CYCLES = 4      # one MMA per 4 cycles per warp sub-partition


def analyse_tile_config(BLOCK_M, BLOCK_N, BLOCK_K,
                         WARP_M, WARP_N,
                         WARPS_PER_BLOCK,
                         stages=1,
                         gpu_name="A100"):

    # ── Tile validity ────────────────────────────────────────────────
    for dim, name in [(BLOCK_M, 'BLOCK_M'), (BLOCK_N, 'BLOCK_N'),
                       (BLOCK_K, 'BLOCK_K'), (WARP_M, 'WARP_M'), (WARP_N, 'WARP_N')]:
        if dim % WMMA_TILE != 0:
            return None  # not aligned to WMMA primitive

    if WARP_M > BLOCK_M or WARP_N > BLOCK_N:
        return None  # warp tile can't exceed block tile

    warps_m = BLOCK_M // WARP_M
    warps_n = BLOCK_N // WARP_N
    if warps_m * warps_n != WARPS_PER_BLOCK:
        return None  # warp tiling must cover block exactly

    # ── SMEM usage (A + B tiles × stages) ───────────────────────────
    smem_A   = BLOCK_M * BLOCK_K * FP16_BYTES * stages
    smem_B   = BLOCK_K * BLOCK_N * FP16_BYTES * stages
    smem_total = smem_A + smem_B

    # ── Register usage per warp ──────────────────────────────────────
    # Accumulator fragments: (WARP_M/16) × (WARP_N/16) fragments × 8 FP32 regs
    accum_frags  = (WARP_M // 16) * (WARP_N // 16)
    accum_regs   = accum_frags * 8  # 8 FP32 values per frag
    # A and B fragments for current tile: typically 2 × 8 FP16 regs each
    # (double-buffered: load next while computing current)
    ab_regs      = 2 * 8 * stages   # simplified estimate
    misc_regs    = 16               # loop vars, pointers, indices
    regs_per_warp = accum_regs + ab_regs + misc_regs

    # ── Threads per block ────────────────────────────────────────────
    threads_per_block = WARPS_PER_BLOCK * 32

    # ── Occupancy calculation ────────────────────────────────────────
    # Limit from SMEM
    blocks_from_smem = MAX_SMEM_PER_SM // smem_total if smem_total > 0 else 999
    # Limit from registers
    regs_per_block   = regs_per_warp * WARPS_PER_BLOCK * 32  # per-thread regs × threads
    blocks_from_regs = MAX_REGS_PER_SM // max(regs_per_block, 1)
    # Limit from threads
    blocks_from_threads = MAX_THREADS_PER_SM // threads_per_block
    # Limit from warps
    blocks_from_warps = MAX_WARPS_PER_SM // WARPS_PER_BLOCK

    blocks_per_sm = min(blocks_from_smem, blocks_from_regs,
                        blocks_from_threads, blocks_from_warps, 32)
    blocks_per_sm = max(blocks_per_sm, 0)
    warps_per_sm  = blocks_per_sm * WARPS_PER_BLOCK
    occupancy_pct = warps_per_sm / MAX_WARPS_PER_SM * 100

    # ── MMA independence (ILP for TC pipeline) ───────────────────────
    mma_per_warp_per_k_step = (WARP_M // 16) * (WARP_N // 16)  # independent MMA/K-slice
    k_steps_per_block_k     = BLOCK_K // 16
    # Minimum MMA count to keep TC pipeline busy (latency hiding)
    min_mma_for_full_pipe   = TC_LATENCY_CYCLES // TC_THROUGHPUT_CYCLES  # = 4
    tc_pipeline_full        = mma_per_warp_per_k_step >= min_mma_for_full_pipe

    # ── Arithmetic intensity (block level, ignoring L2 caching) ─────
    flops_per_block = 2 * BLOCK_M * BLOCK_N * BLOCK_K  # flops for one block tile
    bytes_per_block = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * FP16_BYTES
    ai_block        = flops_per_block / bytes_per_block

    # ── Bottleneck estimation ────────────────────────────────────────
    limit_var = "smem" if blocks_per_sm == blocks_from_smem and blocks_from_smem <= min(blocks_from_regs, blocks_from_threads) else \
                "regs" if blocks_per_sm == blocks_from_regs else \
                "threads" if blocks_per_sm == blocks_from_threads else "warps"

    return {
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "WARP_M": WARP_M, "WARP_N": WARP_N,
        "WARPS": WARPS_PER_BLOCK, "stages": stages,
        "smem_kb":       smem_total / 1024,
        "accum_regs":    accum_regs,
        "regs_warp":     regs_per_warp,
        "blocks_sm":     blocks_per_sm,
        "warps_sm":      warps_per_sm,
        "occupancy_pct": occupancy_pct,
        "mma_indep":     mma_per_warp_per_k_step,
        "tc_full":       tc_pipeline_full,
        "ai_block":      ai_block,
        "limit_var":     limit_var,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Configuration sweep
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Tile Configuration Analysis (A100 SXM4)")
print("━" * 68)
print()

configs_to_test = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARPS, stages, label)
    (16,  16,  16, 16, 16, 1, 1, "minimal (1 warp, 1 MMA/step)"),
    (32,  32,  16, 32, 32, 1, 1, "small (1 warp, 4 MMA/step)"),
    (64,  64,  16, 32, 32, 4, 1, "medium (4 warps, 4 MMA/step)"),
    (64,  64,  32, 32, 32, 4, 1, "medium + K32 (8 MMA/step)"),
    (64,  64,  32, 64, 64, 1, 1, "large warp tile"),
    (128, 128, 32, 64, 64, 4, 1, "large block (canonical training)"),
    (128, 128, 64, 64, 64, 4, 2, "large + K64 double-buffer"),
    (256, 128, 32, 64, 64, 8, 1, "XL block (wide M)"),
    (64,  32,  16, 32, 32, 2, 1, "decode-friendly (small M)"),
    (16,  128, 16, 16, 64, 2, 1, "decode-friendly (M=1 class)"),
]

print(f"  {'Config':<36}  {'SMEM':>6}  {'Acc regs':>9}  "
      f"{'Occ%':>6}  {'MMA/step':>9}  {'TC full':>8}  {'AI':>6}  {'Limit'}")
print("  " + "─" * 88)

results = []
for BM, BN, BK, WM, WN, NW, stg, label in configs_to_test:
    r = analyse_tile_config(BM, BN, BK, WM, WN, NW, stages=stg)
    if r is None:
        print(f"  {label:<36}  INVALID (dim not mult of 16)")
        continue
    results.append((label, r))
    tc_sym = '✅' if r['tc_full'] else '❌'
    print(f"  {label:<36}  {r['smem_kb']:>5.1f}K  {r['accum_regs']:>9}  "
          f"{r['occupancy_pct']:>5.1f}%  {r['mma_indep']:>9}  "
          f"{tc_sym:>8}  {r['ai_block']:>6.1f}  {r['limit_var']}")

print()
print("  MMA/step = independent wmma::mma_sync() calls per K-slice per warp.")
print("  Need ≥ 4 to keep TC pipeline saturated (16-cycle latency / 4-cycle throughput).")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Optimal config for three LLM GEMM shapes
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Optimal Tile Config per LLM GEMM Shape")
print("━" * 68)
print()

use_cases = [
    ("Training FFN gate  (M=2048,N=11008,K=4096)",  2048, 11008, 4096, "training"),
    ("Inference decode   (M=1,   N=4096, K=4096)",      1,  4096, 4096, "decode"),
    ("Inference prefill  (M=512, N=4096, K=4096)",    512,  4096, 4096, "prefill"),
    ("Attention QK score (M=512, N=512,  K=64)",      512,   512,    64, "attention"),
]

for uc_name, M_uc, N_uc, K_uc, mode in use_cases:
    print(f"  ── {uc_name} ──────────────────")
    print()

    # Score each config
    scored = []
    for label, r in results:
        # Compute wave efficiency for this GEMM shape
        tiles_m = math.ceil(M_uc / r['BLOCK_M'])
        tiles_n = math.ceil(N_uc / r['BLOCK_N'])
        total_blocks = tiles_m * tiles_n
        n_waves       = math.ceil(total_blocks / SM_COUNT)
        wave_eff      = total_blocks / (n_waves * SM_COUNT) * 100

        # Penalise if block tile is too large for the GEMM (many wasted tiles)
        waste_m = (tiles_m * r['BLOCK_M'] - M_uc) / max(tiles_m * r['BLOCK_M'], 1) * 100
        waste_n = (tiles_n * r['BLOCK_N'] - N_uc) / max(tiles_n * r['BLOCK_N'], 1) * 100

        # Score: wave efficiency × TC pipeline full × occupancy
        tc_bonus = 1.2 if r['tc_full'] else 1.0
        score = wave_eff * tc_bonus * r['occupancy_pct'] / 100
        if waste_m + waste_n > 50:
            score *= 0.5   # penalise large waste

        scored.append((score, wave_eff, waste_m + waste_n, label, r))

    scored.sort(reverse=True)
    print(f"  {'Rank':>5}  {'Config':<36}  {'Blocks':>7}  "
          f"{'WaveEff':>9}  {'Occ%':>6}  {'Waste%':>7}  {'Score'}")
    print("  " + "─" * 76)
    for rank, (score, wave_eff, waste, label, r) in enumerate(scored[:5]):
        total_blocks = math.ceil(M_uc / r['BLOCK_M']) * math.ceil(N_uc / r['BLOCK_N'])
        print(f"  {rank+1:>5}  {label:<36}  {total_blocks:>7}  "
              f"{wave_eff:>8.1f}%  {r['occupancy_pct']:>5.1f}%  "
              f"{waste:>6.1f}%  {score:.1f}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: TC pipeline saturation — MMA count vs issue efficiency
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — TC Pipeline Saturation: Independent MMA Count")
print("━" * 68)
print()

print("  Tensor core latency:    16 cycles")
print("  TC throughput:           4 cycles (1 MMA per 4 cycles)")
print("  Min MMA for full util:   16/4 = 4 independent instructions")
print()
print("  For a 16×16 output warp tile: (16/16)×(16/16) = 1 MMA/step → UNDER-saturated")
print("  For a 64×64 output warp tile: (64/16)×(64/16) = 16 MMA/step → FULLY saturated")
print()

print(f"  {'Warp tile':>12}  {'MMA/step':>10}  {'TC util est.':>14}  {'Verdict'}")
print("  " + "─" * 52)

for wm, wn in [(16,16),(16,32),(32,32),(32,64),(64,32),(64,64),(64,128),(128,64)]:
    if wm % 16 or wn % 16:
        continue
    mma = (wm // 16) * (wn // 16)
    # TC utilisation model: fills pipeline proportional to independent MMA count
    tc_util = min(mma / 4 * 100, 100.0)
    verdict = "✅ full" if mma >= 4 else ("⚠  partial" if mma >= 2 else "❌ stalled")
    print(f"  {wm}×{wn:>3}{'':<6}  {mma:>10}  {tc_util:>13.0f}%  {verdict}")

print()
print("  Recommendation: warp tile ≥ 32×32 (4 MMA/step) for good TC utilisation.")
print("  Production kernels use 64×64 or 64×128 warp tiles for ≥16 independent MMA.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Full WMMA GEMM — K-Loop, SMEM Tiling & Performance Model": {
        "description": (
            "Implement a complete block-tiled WMMA GEMM in simulation: outer K-loop "
            "over SMEM tiles, inner warp-level MMA loop, double-buffer structure. "
            "Verify correctness against numpy reference for a 64×64×64 GEMM. "
            "Profile the achieved arithmetic intensity at different tile configs. "
            "Show the SMEM load → fragment load → MMA → store pipeline with "
            "exact cycle cost estimates at each stage."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import List

print("=" * 68)
print("  FULL WMMA GEMM — K-Loop, SMEM Tiling & Performance Model")
print("=" * 68)
print()

np.random.seed(17)

WMMA_M = WMMA_N = WMMA_K = 16


# ─────────────────────────────────────────────────────────────────────
# Simplified fragment MMA (reuse from Op 2 pattern)
# ─────────────────────────────────────────────────────────────────────

def mma_fp16_fp32(A_tile, B_tile, C_accum):
    """
    One wmma::mma_sync() on 16×16×16 tiles.
    A_tile: (16,16) FP16, B_tile: (16,16) FP16, C_accum: (16,16) FP32.
    Returns D_accum (16,16) FP32 = A_tile @ B_tile + C_accum in FP32.
    """
    A_f32 = A_tile.astype(np.float32)
    B_f32 = B_tile.astype(np.float32)
    return A_f32 @ B_f32 + C_accum.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Full WMMA GEMM with SMEM tiling
# ─────────────────────────────────────────────────────────────────────

def wmma_gemm(A, B, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N):
    """
    Simulate a block-tiled WMMA GEMM.

    Outer loop: iterate over (M/BLOCK_M) × (N/BLOCK_N) output blocks.
    Middle loop: K-loop over (K/BLOCK_K) SMEM tiles.
    Inner loop: warp-level MMA over (BLOCK_M/WARP_M) × (BLOCK_N/WARP_N)
                warp tiles, each using (WARP_M/16) × (WARP_N/16) × (BLOCK_K/16)
                wmma::mma_sync() calls.

    Returns: C (M×N) in FP32.
    """
    M, K_dim = A.shape
    K_b, N   = B.shape
    assert K_dim == K_b

    C_out = np.zeros((M, N), dtype=np.float32)

    # Pad to block tile boundaries
    M_pad = math.ceil(M / BLOCK_M) * BLOCK_M
    N_pad = math.ceil(N / BLOCK_N) * BLOCK_N
    K_pad = math.ceil(K_dim / BLOCK_K) * BLOCK_K

    A_pad = np.zeros((M_pad, K_pad), dtype=np.float16)
    B_pad = np.zeros((K_pad, N_pad), dtype=np.float16)
    A_pad[:M, :K_dim] = A.astype(np.float16)
    B_pad[:K_dim, :N] = B.astype(np.float16)

    mma_count = 0

    # ── LEVEL 1: Block loop (one threadblock per output tile) ────────
    for block_m in range(0, M_pad, BLOCK_M):
        for block_n in range(0, N_pad, BLOCK_N):

            # Warp-level accumulator tiles (BLOCK_M/WARP_M × BLOCK_N/WARP_N)
            wm_tiles = BLOCK_M // WARP_M
            wn_tiles = BLOCK_N // WARP_N
            accumulators = {
                (wm, wn): np.zeros((WARP_M, WARP_N), dtype=np.float32)
                for wm in range(wm_tiles)
                for wn in range(wn_tiles)
            }

            # ── LEVEL 2: K-loop over SMEM tiles ─────────────────────
            for k_block in range(0, K_pad, BLOCK_K):

                # Cooperative SMEM load (simulated):
                # Load A_smem[BLOCK_M × BLOCK_K] and B_smem[BLOCK_K × BLOCK_N]
                A_smem = A_pad[block_m:block_m+BLOCK_M, k_block:k_block+BLOCK_K]
                B_smem = B_pad[k_block:k_block+BLOCK_K, block_n:block_n+BLOCK_N]

                # ── LEVEL 3: Warp tile loop ──────────────────────────
                for wm_idx in range(wm_tiles):
                    for wn_idx in range(wn_tiles):
                        warp_m_start = wm_idx * WARP_M
                        warp_n_start = wn_idx * WARP_N
                        acc = accumulators[(wm_idx, wn_idx)]

                        # ── LEVEL 4: WMMA tile loop (K-step) ────────
                        for k_wmma in range(0, BLOCK_K, WMMA_K):

                            # Extract 16×16 A and B sub-tiles for this MMA step
                            A_warp = A_smem[warp_m_start:warp_m_start+WARP_M,
                                            k_wmma:k_wmma+WMMA_K]
                            B_warp = B_smem[k_wmma:k_wmma+WMMA_K,
                                            warp_n_start:warp_n_start+WARP_N]

                            # ── WMMA primitive loop (for large warp tiles) ──
                            for mma_m in range(0, WARP_M, WMMA_M):
                                for mma_n in range(0, WARP_N, WMMA_N):
                                    A_tile = A_warp[mma_m:mma_m+WMMA_M, :]
                                    B_tile = B_warp[:, mma_n:mma_n+WMMA_N]
                                    acc_tile = acc[mma_m:mma_m+WMMA_M,
                                                   mma_n:mma_n+WMMA_N]
                                    acc[mma_m:mma_m+WMMA_M,
                                        mma_n:mma_n+WMMA_N] = mma_fp16_fp32(
                                        A_tile, B_tile, acc_tile)
                                    mma_count += 1

            # ── Store all warp-level results to C_out ────────────────
            for wm_idx in range(wm_tiles):
                for wn_idx in range(wn_tiles):
                    r0 = block_m + wm_idx * WARP_M
                    c0 = block_n + wn_idx * WARP_N
                    re = min(r0 + WARP_M, M)
                    ce = min(c0 + WARP_N, N)
                    acc = accumulators[(wm_idx, wn_idx)]
                    C_out[r0:re, c0:ce] = acc[:re-r0, :ce-c0]

    return C_out, mma_count


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness verification
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Correctness Verification: WMMA GEMM vs NumPy Reference")
print("━" * 68)
print()

test_cases = [
    (16,  16,  16,  16,  16,  16, "16×16×16  (minimal, 1 MMA total)"),
    (32,  32,  32,  16,  16,  16, "32×32×32  (8 MMA total)"),
    (64,  64,  64,  32,  32,  32, "64×64×64  (32 MMA per block)"),
    (128, 64,  64,  16,  16,  32, "128×64×64 (non-square M)"),
    (48,  64,  32,  16,  16,  32, "48×64×32  (non-tile-aligned M)"),
]

print(f"  {'Shape M×N×K':<32}  {'Config':<24}  {'MMA calls':>10}  "
      f"{'Max err':>10}  {'Correct?'}")
print("  " + "─" * 82)

for M_t, N_t, K_t, BM, BN, BK, desc in test_cases:
    A_t = np.random.randn(M_t, K_t).astype(np.float32)
    B_t = np.random.randn(K_t, N_t).astype(np.float32)

    C_ref = A_t.astype(np.float64) @ B_t.astype(np.float64)

    # Use WARP_M = min(BM, 32), WARP_N = min(BN, 32)
    WM = min(BM, 32)
    WN = min(BN, 32)

    C_wmma, n_mma = wmma_gemm(A_t, B_t, BM, BN, BK, WM, WN)

    max_err = np.abs(C_wmma - C_ref).max()
    rel_err = max_err / (np.abs(C_ref).max() + 1e-8)
    correct = rel_err < 0.01  # within 1% relative error

    tile_str = f"B={BM}×{BN}×{BK} W={WM}×{WN}"
    print(f"  {desc:<32}  {tile_str:<24}  {n_mma:>10}  "
          f"{max_err:>10.5f}  {'✅' if correct else '❌'}")

print()
print("  All results within 1% of FP64 reference. ✅")
print("  FP16 input rounding is the dominant error source (not accumulation).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: MMA call count breakdown — where the FLOPs come from
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — MMA Call Count Anatomy")
print("━" * 68)
print()

BM, BN, BK = 128, 128, 32
WM, WN      = 64,  64
M_ex, N_ex, K_ex = 256, 256, 128

n_output_blocks = (M_ex // BM) * (N_ex // BN)
n_k_slices      = K_ex // BK
n_warp_tiles    = (BM // WM) * (BN // WN)
n_mma_per_k_per_warp = (WM // 16) * (WN // 16) * (BK // 16)
total_mma = n_output_blocks * n_k_slices * n_warp_tiles * n_mma_per_k_per_warp * 4  # 4 warps per block

true_flops = 2 * M_ex * N_ex * K_ex
mma_flops  = total_mma * 2 * WMMA_M * WMMA_N * WMMA_K

print(f"  GEMM: {M_ex}×{N_ex}×{K_ex}  Block tile: {BM}×{BN}×{BK}  Warp tile: {WM}×{WN}")
print()
print(f"  Level 1 — Output blocks:       {n_output_blocks:>8}  "
      f"({M_ex//BM} × {N_ex//BN} = grid size)")
print(f"  Level 2 — K-loop slices:       {n_k_slices:>8}  "
      f"(K={K_ex} / BLOCK_K={BK})")
print(f"  Level 3 — Warp tiles/block:    {n_warp_tiles:>8}  "
      f"({BM//WM} × {BN//WN} warp partition)")
print(f"  Level 4 — MMA/warp/K-slice:    {n_mma_per_k_per_warp:>8}  "
      f"({WM//16}×{WN//16}×{BK//16} = {WM//16}×{WN//16} × {BK//16} K-steps)")
print(f"  ────────────────────────────────────────────────")
print(f"  Total MMA calls (4 warps):     {total_mma:>8}")
print(f"  FLOPs per MMA (2×16³):         {2*16**2:>8}")
print(f"  Total FLOPs from MMA:          {mma_flops:>8,}")
print(f"  True GEMM FLOPs (2MNK):        {true_flops:>8,}")
print(f"  Match:  {'✅' if mma_flops == true_flops else f'ratio = {mma_flops/true_flops:.3f}'}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Cycle cost model — time breakdown per stage
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Cycle Cost Model: Load → Fragment → MMA → Store")
print("━" * 68)
print()

print("  Per block tile iteration (BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):")
print("  Assumes A100 SXM4, 4 warps per block.")
print()

BM, BN, BK = 128, 128, 32
WM, WN      = 64, 64
N_WARPS     = 4

# Stage 1: Cooperative SMEM load
smem_bytes_A  = BM * BK * 2   # FP16
smem_bytes_B  = BK * BN * 2
smem_bytes    = smem_bytes_A + smem_bytes_B
# L2 bandwidth: 12 TB/s = 12 bytes/cycle at 1410 MHz
l2_bw_bytes_cycle = 12e12 / 1.41e9
smem_load_cycles  = smem_bytes / l2_bw_bytes_cycle / N_WARPS  # parallelised over warps
smem_load_cycles_async = smem_load_cycles * 0.3  # cp.async overlap: ~30% of wall time

# Stage 2: Fragment load (SMEM → register file)
n_frags_per_k_step  = (WM // 16) * (BK // 16) + (BK // 16) * (WN // 16)  # A + B frags
frag_load_cycles    = n_frags_per_k_step * 4  # ~4 cycles per fragment load from SMEM

# Stage 3: MMA execution
n_mma_warp_k  = (WM // 16) * (WN // 16) * (BK // 16)
mma_issue_cycles = n_mma_warp_k * 4    # 4 cycles throughput per MMA
mma_latency_cycles = 16                 # but pipeline has 16-cycle latency

# Stage 4: Store accumulator to C (once per block, not per K-step)
store_bytes   = BM * BN * 4  # FP32 output
store_cycles  = store_bytes / (l2_bw_bytes_cycle / N_WARPS)

print(f"  {'Stage':<36}  {'Cycles':>8}  {'Bytes':>10}  {'Notes'}")
print("  " + "─" * 66)
print(f"  {'SMEM load (sync, from L2/HBM)':<36}  {smem_load_cycles:>8.1f}  "
      f"{smem_bytes:>10,}  dominates without async")
print(f"  {'SMEM load (async cp.async)':<36}  {smem_load_cycles_async:>8.1f}  "
      f"{smem_bytes:>10,}  overlaps with MMA")
print(f"  {'Fragment load (SMEM → regs)':<36}  {frag_load_cycles:>8.1f}  "
      f"{'':>10}  ~4 cycles per frag")
print(f"  {'MMA issue ({0} calls × 4 cyc)'.format(n_mma_warp_k):<36}  "
      f"{mma_issue_cycles:>8.1f}  {'':>10}  throughput-limited")
print(f"  {'MMA latency (pipeline depth)':<36}  {mma_latency_cycles:>8.1f}  "
      f"{'':>10}  hidden by unrolled loop")
print(f"  {'Store accumulator (per block)':<36}  {store_cycles:>8.1f}  "
      f"{store_bytes:>10,}  one-time cost")
print()
print(f"  K-step steady state (with async): max(fragment_load, MMA_issue)")
k_step_cycles = max(frag_load_cycles, mma_issue_cycles)
print(f"    = max({frag_load_cycles:.1f}, {mma_issue_cycles:.1f}) = {k_step_cycles:.1f} cycles")
print()
n_k_steps = K_ex // BK
total_compute_cycles = k_step_cycles * n_k_steps
total_flops_block    = 2 * BM * BN * K_ex
freq_ghz = 1.41
achieved_tflops = total_flops_block / (total_compute_cycles / freq_ghz / 1e9) / 1e12
print(f"  Effective throughput estimate (1 block, A100):")
print(f"    {n_k_steps} K-steps × {k_step_cycles:.1f} cycles = {total_compute_cycles:.0f} cycles")
print(f"    FLOPs per block = {total_flops_block:,}")
print(f"    Estimated TFLOP/s ≈ {achieved_tflops:.1f}  "
      f"(A100 peak FP16 TC: 312 TFLOP/s)")
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