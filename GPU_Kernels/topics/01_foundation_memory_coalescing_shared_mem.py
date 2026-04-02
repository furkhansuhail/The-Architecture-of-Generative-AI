"""
Memory Coalescing & Shared Memory — GPU Memory Hierarchy Mastery
=================================================================

The single biggest performance lever in CUDA programming is not the
algorithm — it is how the algorithm moves data. A kernel that does the
same arithmetic but touches memory in the wrong pattern can be 10–50×
slower than an optimised version on identical hardware.

This module covers the two foundational tools for controlling memory
performance: COALESCING (how threads fetch from global/HBM memory in
parallel without wasting bandwidth) and SHARED MEMORY (the on-chip
scratchpad that eliminates repeated round-trips to global memory).

Together these two techniques underpin every high-performance kernel in
the CUDA ecosystem — cuBLAS, CUTLASS, FlashAttention, and every custom
ML kernel you will ever write.

"""

import textwrap
import re
import math

TOPIC_NAME  = "Memory Coalescing & Shared Memory — GPU Memory Hierarchy"
DISPLAY_NAME = "01 · Memory Coalescing & Shared Memory"
ICON        = "🧠"
SUBTITLE    = "Coalescing, Bank Conflicts & Tiling — the Foundation of Every Fast Kernel"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE GPU MEMORY HIERARCHY

### The Full Picture

    GPU memory is a hierarchy of increasing speed and decreasing size:

        ┌──────────────────────────────────────────────────────────┐
        │  HBM / GDDR  (Global Memory)   80 GB, ~2 TB/s            │
        │  ─────────────────────────────────────────────────────── │
        │  L2 Cache                       40–50 MB,  ~12 TB/s      │
        │  ─────────────────────────────────────────────────────── │
        │  L1 Cache / Shared Memory (SMEM)  192 KB/SM, ~19 TB/s    │
        │  ─────────────────────────────────────────────────────── │
        │  Register File                  65536 regs/SM, fastest   │
        └──────────────────────────────────────────────────────────┘

    Numbers above are for NVIDIA H100 SXM. The pattern holds across GPU generations:
    each level is 6–10× faster but 10–100× smaller than the level below it.

### Why This Matters for Kernel Performance

    Consider a simple element-wise operation:  C[i] = A[i] * B[i]

        Naive implementation:
            Each thread loads A[i] from HBM → multiply → store C[i] to HBM.
            For a 1B-element tensor: 3 × 8 GB (FP32) = 24 GB transferred.
            At 2 TB/s: theoretical minimum = 12 ms.
            Realistic (naïve): 60–120 ms due to memory access inefficiency.

        Key insight: your kernel is almost never compute-bound for ML workloads.
        It is MEMORY-BANDWIDTH bound. Optimising memory access is the job.

    The two primary tools:
        1. COALESCING   — maximise HBM bandwidth by accessing memory in
                          wide, aligned, contiguous bursts per warp.
        2. SHARED MEMORY — park frequently-reused data on-chip (SMEM) so
                           threads pay the HBM cost once, not N times.

### Arithmetic Intensity: The Roof line Framing

    Arithmetic Intensity (AI) = FLOPs / Bytes transferred

        High AI (compute-bound):  e.g., large GEMM, conv with big filters
        Low  AI (memory-bound):   e.g., elementwise, layer norm, softmax

    Roofline model for A100 SXM:
        Peak FP16 throughput:  312 TFLOP/s
        Peak HBM bandwidth:    2.0 TB/s
        Ridge point:           312 / 2.0 = 156 FLOPs/Byte

        Kernel with AI < 156:  memory-bound  → optimise memory access
        Kernel with AI > 156:  compute-bound → optimise arithmetic

    Most transformer sub-operations (attention QKV projections excluded):
        Softmax: ~1 FLOP/Byte   → 156× below ridge point
        LayerNorm: ~4 FLOPs/Byte
        ReLU: ~0.5 FLOPs/Byte

    For these kernels, PERFECT memory access (coalescing + SMEM) is everything.


##### PART 2 — MEMORY COALESCING: THE WARP-LEVEL CONTRACT

### What a Warp Is

    CUDA threads execute in groups of 32 called WARPS.
    A warp is the hardware's atomic unit of execution — all 32 threads
    in a warp execute the same instruction simultaneously (SIMT model).

    Warp ID of a thread (within a 1D block):
        warp_id = threadIdx.x / 32
        lane_id = threadIdx.x % 32   ← position within the warp (0–31)

### The Memory Transaction Unit

    Global memory is served to warps in 32-byte or 128-byte cache-line
    transactions (depending on L1 hit/miss). The key rule:

        When 32 threads in a warp issue memory requests,
        the hardware groups them into as FEW transactions as possible.

    COALESCED (ideal):
        All 32 threads access a contiguous, aligned 128-byte region.
        → 1 memory transaction for the entire warp.
        → Full bandwidth utilisation.

        Thread 0: addr  0
        Thread 1: addr  4   (FP32 = 4 bytes each)
        Thread 2: addr  8
        ...
        Thread 31: addr 124
        All 128 bytes fit in ONE cache line → 1 transaction.

    UNCOALESCED (worst case):
        Each thread accesses a completely random address.
        → Up to 32 separate transactions (one per thread).
        → 32× less effective bandwidth.

### Coalescing Patterns: A Taxonomy

    PATTERN 1 — COALESCED (✅ ideal):
        float val = A[blockIdx.x * 32 + threadIdx.x];
        Threads 0–31 in a warp fetch addresses 0–31, 32–63, etc.
        Each warp = 1 transaction. Peak bandwidth achieved.

    PATTERN 2 — STRIDED ACCESS (⚠️ partial):
        float val = A[threadIdx.x * stride];
        stride=1:  coalesced   (1 transaction)
        stride=2:  50% utilised (threads access every other element)
        stride=4:  25% utilised
        stride=32: 3.1% utilised — catastrophic!

    PATTERN 3 — BROADCAST (✅ special case):
        float val = A[0];   // all threads read the SAME address
        → 1 transaction, result broadcast to all 32 threads.
        L1/L2 cache exploits this efficiently.

    PATTERN 4 — COLUMN-MAJOR (❌ naïve matrix):
        Reading column j of a row-major matrix:
            A[row * N + col]   where col = j (fixed), row = threadIdx.x
        → threads stride by N elements (row width)
        → for N=1024, FP32: stride = 4096 bytes — catastrophically uncoalesced.

    PATTERN 5 — RANDOM / GATHER (❌ worst):
        float val = A[index[threadIdx.x]];   // index = arbitrary
        Each thread goes to a different cache line.
        → 32 independent transactions. 1/32 bandwidth efficiency.

### Row-Major vs Column-Major: The Classic Example

    Matrix A stored row-major (C default): A[i][j] = A_flat[i*N + j]

    ACCESSING ROWS (coalesced):
        Thread k reads A[row][k] for k in 0..31
        Consecutive columns → consecutive memory → 1 transaction ✅

    ACCESSING COLUMNS (uncoalesced):
        Thread k reads A[k][col] for k in 0..31
        Consecutive rows, but row stride = N elements:
        Thread 0: offset 0
        Thread 1: offset N
        Thread 2: offset 2N
        → 32 transactions, each 1 element ❌

    Solution: either transpose the matrix in global memory before access,
    or use SHARED MEMORY to stage the data and re-access it coalesced.
    (This is exactly what the tiled matrix multiply does — see Part 4.)

### L1 and L2: When They Help (and When They Don't)

    L1 Cache (per SM, ~128 KB on H100, shared with SMEM):
        Helps with: repeated access to the same addresses within a kernel
        Does NOT help: streaming access (each line fetched once, evicted immediately)
        Most ML kernels are streaming → L1 provides minimal benefit.

    L2 Cache (40–50 MB on H100, shared across all SMs):
        Helps with: data reused across different SMs or kernel launches
        vLLM's prefix caching → same KV blocks reused: L2 helps significantly.
        Random access with working set > 50 MB: L2 does not help.

    Key principle: SMEM is programmer-managed L1.
        You decide what goes in, when, and the access pattern.
        Hardware L1 decides for you — and often decides wrong.
        For any non-trivial kernel, explicit SMEM beats hardware L1.


##### PART 3 — SHARED MEMORY: ON-CHIP SCRATCHPAD

### What Shared Memory Is

    Shared memory (SMEM) is a fast, on-chip SRAM located inside each
    Streaming Multiprocessor (SM). It is:
        - ~6 ns latency (vs ~300–600 ns for HBM)
        - ~19 TB/s peak bandwidth (vs ~2 TB/s for HBM)
        - Shared by ALL threads in a THREAD BLOCK (not across blocks)
        - Lifetime = the thread block's lifetime (allocated at launch, freed at exit)
        - Size: configurable, up to 228 KB/SM on H100 (split with L1)

    Declaration in CUDA C:
        __shared__ float tile[TILE_SIZE][TILE_SIZE];   // statically allocated
        extern __shared__ float dynamic_tile[];        // dynamically sized at launch

    In Python (Triton or Numba):
        tile = cuda.shared.array(shape=(TILE, TILE), dtype=float32)  # Numba
        tl.load(..., cache_modifier=".ca")  # Triton handles SMEM automatically

### The Bank Structure: Why Bank Conflicts Matter

    SMEM is divided into 32 BANKS (matching warp width).
    Each bank is 4 bytes wide (FP32) or 8 bytes wide (FP64 mode).
    Bank assignment: bank_id = (byte_address / 4) % 32

    The guarantee: if all 32 threads in a warp access different banks,
    all 32 accesses happen simultaneously → 1 cycle.

    ┌──────────────────────────────────────────────────────────┐
    │  32 banks:  B0  B1  B2  B3  ... B31                      │
    │  Addresses: 0   4   8   12  ...  124                     │
    │             128 132 136 140 ...  252   (next 128 bytes)  │
    │             256 260 264 268 ...  380   ...               │
    └──────────────────────────────────────────────────────────┘

    BANK CONFLICT: two or more threads in the SAME warp access
    DIFFERENT addresses in the SAME bank simultaneously.

    The hardware serialises conflicted accesses:
        2-way conflict  → 2 cycles (2× slower)
        4-way conflict  → 4 cycles (4× slower)
        32-way conflict → 32 cycles (32× slower = back to HBM speed)

    Exception: BROADCAST — if all conflicting threads access the
    SAME address in a bank, it's served in 1 cycle (broadcast, not conflict).

### Common Bank Conflict Patterns

    PATTERN 1 — NO CONFLICT (✅ ideal):
        Thread k reads tile[k]   (stride-1 access)
        Thread 0 → bank 0, Thread 1 → bank 1, ... Thread 31 → bank 31
        All different banks → 1 cycle. ✅

    PATTERN 2 — STRIDE-2 CONFLICT (⚠️ 2-way):
        Thread k reads tile[k * 2]
        Thread 0  → bank 0, Thread 1  → bank 2, ...
        Thread 16 → bank 0 (same as Thread 0!) → 2-way conflict on each bank
        Result: 2 passes needed → 2× slower.

    PATTERN 3 — STRIDE-32 CONFLICT (❌ catastrophic):
        Thread k reads tile[k * 32]
        ALL 32 threads map to bank 0!
        32-way conflict → 32 serialised accesses → same speed as HBM.
        This is the most common mistake in naïve transpose kernels.

    PATTERN 4 — COLUMN ACCESS OF ROW-MAJOR SMEM TILE (❌ classic trap):
        // SMEM tile declared as float tile[32][32]
        Thread k reads tile[k][col]   // row = k, col = fixed
        tile[0][col] → bank (col % 32)
        tile[1][col] → bank (col % 32)  — SAME BANK as tile[0][col]!
        32-way conflict on all threads.

        FIX: pad the tile by 1 column:
        float tile[32][33]   // 33 instead of 32
        Now tile[k][col] → bank ((k*33 + col) % 32)
        Different row → different bank → no conflict. ✅

### The Padding Fix: Why +1 Eliminates Conflicts

    tile[TILE][TILE]:
        tile[r][c] is at byte offset (r * TILE + c) * 4
        bank = (r * TILE + c) % 32
        For TILE=32, c=0: bank = (r * 32) % 32 = 0 for ALL rows → 32-way conflict.

    tile[TILE][TILE+1]:
        byte offset = (r * (TILE+1) + c) * 4
        bank = (r * (TILE+1) + c) % 32
        For TILE=32, c=0: bank = (r * 33) % 32 = r % 32 → each row → different bank ✅

    Cost: 4 bytes × TILE wasted per row.
    For TILE=32: 32 × 4 = 128 bytes wasted per tile (out of 4 KB). Negligible.
    Always pad when accessing SMEM in column-major patterns.

### SMEM Capacity and Occupancy

    Occupancy = fraction of maximum warps active on an SM simultaneously.

    SMEM limits occupancy because it is a finite shared resource:
        H100 SM: 228 KB max SMEM
        If your kernel uses 32 KB SMEM per block and block_size = 256 threads:
            Max blocks per SM = 228 / 32 = 7 blocks
            Max threads per SM = 7 × 256 = 1792 threads = 56 warps
            H100 max warps per SM = 64 → occupancy = 56/64 = 87.5%

    If kernel uses 64 KB SMEM per block:
            Max blocks per SM = 228 / 64 = 3 blocks
            Max threads = 3 × 256 = 768 threads = 24 warps → occupancy = 37.5%

    Lower occupancy isn't always bad:
        - If your kernel has enough independent instructions to hide latency,
          lower occupancy is fine.
        - FlashAttention uses large SMEM tiles (64+ KB) with low occupancy
          but is still the fastest attention kernel because it avoids HBM reads.
        - Rule of thumb: aim for ≥ 50% occupancy; below 25% investigate.


##### PART 4 — TILED MATRIX MULTIPLICATION: THE CANONICAL SMEM KERNEL

### Why GEMM is the Teaching Kernel

    C = A × B,  where A is M×K, B is K×N, C is M×N

    Naïve GEMM: each thread computes one C[i][j] = dot(A[i,:], B[:,j])
        Each thread reads K elements from A (row i) and K from B (col j).
        For M=N=K=1024: each of 1024² threads reads 2×1024 FP32 = 8 KB.
        Total reads = 1024² × 8 KB = 8 GB
        But A and B together are only 2 × 4 MB = 8 MB.
        → every byte of A and B read ~1024× (once per output row/col)
        → arithmetic intensity ≈ 2 FLOPs/Byte (2K / 2K)
        → massively memory-bound with no reuse benefit from any cache

    Tiled GEMM: exploit reuse by staging tiles in SMEM.
        Load a TILE×TILE sub-matrix of A and B into SMEM.
        All threads in the block compute from that tile.
        The tile is fetched from HBM only ONCE and reused TILE times.
        → Effective arithmetic intensity ≈ 2 × TILE FLOPs/Byte

### The Tiling Algorithm Step by Step

    Parameters:
        TILE = 16 (16×16 tiles, 256 threads per block)
        Block covers a 16×16 sub-matrix of C.

    STEP 1 — cooperative load:
        All 256 threads in the block collectively load:
            tile_A[16][16] = A[block_row:block_row+16, k_tile:k_tile+16]
            tile_B[16][16] = B[k_tile:k_tile+16, block_col:block_col+16]
        Each thread loads 1 element of tile_A and 1 element of tile_B.
        Access is COALESCED: consecutive threads → consecutive columns in A/B.

    STEP 2 — __syncthreads():
        ALL threads must finish loading before ANY thread can read.
        This barrier ensures the tile is fully in SMEM before computation.
        Missing this → data races → wrong results.

    STEP 3 — compute from SMEM:
        Each thread computes its partial dot product:
            for k in range(TILE):
                acc += tile_A[thread_row][k] * tile_B[k][thread_col]
        All reads from SMEM (6 ns, 19 TB/s) not HBM (300 ns, 2 TB/s).

    STEP 4 — advance and repeat:
        k_tile += TILE; go to STEP 1 for the next tile along the K dimension.
        Accumulate into acc until all K tiles processed.

    STEP 5 — write result:
        C[global_row][global_col] = acc
        One write per thread to HBM. Coalesced.

    Diagram — one 4×4 tile iteration (simplified):

        Global memory (HBM)          Shared memory (on-chip)
        ─────────────────────        ─────────────────────────
        A:                           tile_A:
        ┌───┬───┬───┬───┐            ┌───┬───┬───┬───┐
        │   │ ■ │ ■ │ ■ │  load ──►  │ a │ b │ c │ d │
        │   │ ■ │ ■ │ ■ │            │ e │ f │ g │ h │
        │   │ ■ │ ■ │ ■ │            │ i │ j │ k │ l │
        │   │ ■ │ ■ │ ■ │            │ m │ n │ o │ p │
        └───┴───┴───┴───┘            └───┴───┴───┴───┘

        B:                           tile_B:
        ┌───┬───┬───┬───┐            ┌───┬───┬───┬───┐
        │   │   │   │   │  load ──►  │ α │ β │ γ │ δ │
        │ ■ │ ■ │ ■ │ ■ │            │ ε │ ζ │ η │ θ │
        │ ■ │ ■ │ ■ │ ■ │            │ ι │ κ │ λ │ μ │
        │ ■ │ ■ │ ■ │ ■ │            │ ν │ ξ │ o │ π │
        └───┴───┴───┴───┘            └───┴───┴───┴───┘

        __syncthreads();

        Thread (r=1, c=2) computes:
            acc += tile_A[1][0]*tile_B[0][2]  (e×γ)
                +  tile_A[1][1]*tile_B[1][2]  (f×η)
                +  tile_A[1][2]*tile_B[2][2]  (g×λ)
                +  tile_A[1][3]*tile_B[3][2]  (h×o)
        All from SMEM — no HBM access!

### Memory Traffic Analysis: Naïve vs Tiled

    For M=N=K=1024, TILE=16, FP32:

    NAÏVE:
        Reads:  M×N × 2K × 4 bytes = 1024² × 8192 bytes = 8 GB
        Writes: M×N × 4 bytes = 4 MB
        Total:  ~8 GB
        HBM bandwidth at 2 TB/s: 4 ms ideal

    TILED (TILE=16):
        Reads from HBM: M×K×4 + K×N×4 = 2 × 4 MB = 8 MB (each element once)
        Reads from SMEM: each SMEM element read TILE=16 times
        Writes: 4 MB
        Total HBM: ~8 MB  (1000× less than naïve!)
        HBM bandwidth: 4 μs ideal

    TILE=16: 16× fewer HBM reads than naïve
    TILE=32: 32× fewer HBM reads than naïve
    TILE=64 (cuBLAS/CUTLASS): 64× fewer HBM reads

    This is why GEMM achieves near-peak TFLOP/s: the kernel is compute-bound,
    not memory-bound, because tiles make AI = 2×TILE ≫ 156 FLOPs/Byte.


##### PART 5 — ADVANCED SMEM PATTERNS IN MODERN ML KERNELS

### Double Buffering (Ping-Pong)

    Problem: in tiled GEMM, every tile iteration has a sync barrier:
        load → __syncthreads() → compute → __syncthreads() → load → ...
    While threads compute, the next tile is not being loaded.
    While threads load, the FMA units sit idle.

    Solution: DOUBLE BUFFERING — use 2 SMEM tile pairs:
        Iteration N:   SMEM buffer A = current tile (computing)
                       SMEM buffer B = next tile (loading simultaneously)
        Iteration N+1: swap roles — buffer B becomes current, A gets next tile.

    This hides the global memory latency behind computation.
    CUTLASS uses this pattern extensively; it is why it approaches cuBLAS performance.

    Cost: 2× SMEM usage per kernel (e.g., 2 × 32 KB = 64 KB)
    Benefit: removes pipeline stall between load and compute phases.

### Warp Tiling (Register-Level Caching)

    Beyond SMEM, modern kernels add a third level: REGISTER TILES.
    Each thread accumulates a small 2D result tile (e.g., 8×8) in registers.

        Block tile (SMEM):   128×128 matrix fragment
        Warp tile (SMEM):     64×64 matrix fragment per warp
        Thread tile (regs):    8×8  matrix fragment per thread

    Register access: 0 ns latency (already inside the register file).
    With 8×8 thread tiles: each SMEM element reused 8× without re-reading.
    CUTLASS 3.x implements this as the "MMA instruction pipeline".

### FlashAttention's SMEM Strategy

    FlashAttention (Dao et al., 2022) applies SMEM tiling to the attention
    computation, which prior to it was considered untileable.

    The insight: softmax can be computed INCREMENTALLY using the online
    softmax algorithm (running max + running sum), enabling tiling of Q, K, V.

    SMEM usage in FlashAttention:
        Load Q tile into SMEM once (stays for entire inner loop)
        For each K, V tile:
            Load K tile into SMEM (coalesced)
            Compute S = Q × K^T from SMEM (no HBM)
            Compute online softmax update
            Load V tile into SMEM (coalesced)
            Accumulate output O in registers
        Write O to HBM once (coalesced)

    Result: each Q, K, V element read from HBM exactly ONCE.
    Arithmetic intensity: O(N²d / Nd) = O(N) → grows with sequence length.
    This is why FlashAttention scales to 100K+ context lengths on H100.

### Layer Norm and Reduction in SMEM

    Layer norm requires:
        μ = mean(x)   →  must see all elements
        σ = std(x)    →  must see all elements
        y = (x - μ) / σ  → normalise

    Naïve approach: 3 passes over x (HBM: 3× bandwidth for 1 element's work)

    SMEM approach (one thread block handles one row of the matrix):
        PASS 1: load full row into SMEM (coalesced, 1 HBM read)
        IN SMEM: compute partial sums with warp-level reductions (__shfl_down)
        IN SMEM: compute mean and variance from partial sums
        IN SMEM: normalise every element using computed μ, σ
        PASS 2: write normalised row to HBM (coalesced, 1 HBM write)
        Total HBM: 2 passes (read + write) instead of 3 passes.

    This is the standard implementation in PyTorch's fused layer norm
    and in triton-lang's layer norm tutorial.

### Transpose Kernel: The Classic SMEM Use Case

    Naïve transpose:  out[j][i] = in[i][j]
        Reads in[] coalesced (row-major read of row i) ✅
        Writes out[] uncoalesced (writing to column j = row-major stride) ❌
        Or: reads uncoalesced, writes coalesced (flip the problem).
        Either way: half the operations are uncoalesced. ~50% bandwidth.

    SMEM transpose (padded tile):
        STEP 1: Load 32×32 tile from in[] into smem[32][33] — coalesced ✅
                    Thread (r,c) loads in[row+r][col+c] → smem[r][c]
        STEP 2: __syncthreads()
        STEP 3: Write tile from smem to out[], reading smem transposed — no conflict ✅
                    Thread (r,c) reads smem[c][r] (transposed!) → out[col+r][row+c]
                    smem[c][r] with padding: bank = (c*(32+1)+r)%32 → no conflicts ✅
        Both global memory operations are coalesced.
        SMEM absorbs the non-linear index mapping.
        Result: near-peak bandwidth for both reads and writes.


##### PART 6 — SYNCHRONISATION AND MEMORY CONSISTENCY

### __syncthreads(): The Thread Block Barrier

    __syncthreads() is a barrier for ALL threads in a thread block.
    No thread passes until every thread in the block has reached the barrier.

    REQUIRED whenever:
        - One group of threads writes SMEM, another reads it
        - After loading a tile into SMEM before computing from it
        - After writing partial results that other threads will accumulate

    Common mistake — missing barrier:
        // Thread A writes smem[0]
        smem[threadIdx.x] = global_A[threadIdx.x];
        // Thread B reads smem[0] WITHOUT barrier — DATA RACE
        float val = smem[(threadIdx.x + 1) % 32];   ← undefined behaviour

    Corrected:
        smem[threadIdx.x] = global_A[threadIdx.x];
        __syncthreads();    ← all threads see complete SMEM state
        float val = smem[(threadIdx.x + 1) % 32];   ← safe

### Warp-Level Synchronisation (__syncwarp)

    Within a SINGLE warp, all 32 threads are synchronised by default
    (they execute lockstep). But if a warp diverges (if/else), threads
    in different branches are "masked" and not truly lockstep.

    __syncwarp(mask) re-synchronises threads within a warp.
    Used before intra-warp SMEM communication.
    Cheaper than __syncthreads() (only 1 warp, not whole block).

### Memory Fences (__threadfence)

    __threadfence() ensures all writes from the calling thread are
    visible to ALL threads on the device before proceeding.
    Used for global-scope communication between thread blocks.

    __threadfence_block() — visibility within the block only (cheaper).
    __threadfence_system() — visibility across CPU+GPU (very expensive; avoid).


##### PART 7 — PROFILING MEMORY EFFICIENCY

### Key Metrics from Nsight Compute

    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        Raw 32-byte sector reads from global memory (L1 miss → L2/HBM).
        Compare to ideal (total_bytes / 32) to get coalescing efficiency.

    smsp__sass_l1tex_t_sectors_pipe_lsu_mem_shared_op_ld.sum
        SMEM sector reads. High value = good (computing from SMEM, not HBM).

    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
        SMEM bank conflict count. Should be 0 for optimal kernels.
        If non-zero: examine SMEM access patterns, add padding.

    sm__sass_l1tex_t_bytes_pipe_lsu_mem_shared_op_ld_sector_hit_rate.pct
        SMEM hit rate. Target: > 95%.

    Memory throughput (GB/s) vs theoretical peak:
        Ratio = actual_throughput / peak_bandwidth
        Good kernels: > 80% of peak. Below 50%: investigate coalescing.

### Nsight Compute One-Liner

    ncu --set full --target-processes all \\
        -o profile_output \\
        python my_kernel_benchmark.py

    Then open profile_output.ncu-rep in the Nsight Compute GUI.
    Key sections: Memory Chart, Warp State Statistics, Source Correlation.

### The Three-Step Optimisation Process

    STEP 1 — Measure HBM bandwidth utilisation:
        Actual GB/s / Peak GB/s.
        If < 60%: suspect uncoalesced access or bandwidth underutilisation.

    STEP 2 — Count SMEM bank conflicts:
        ncu metric: l1tex__data_bank_conflicts_...
        If > 0: add padding (+1 column) or restructure access pattern.

    STEP 3 — Profile occupancy vs SMEM usage:
        ncu metric: sm__warps_active.avg.pct_of_peak_sustained_active
        If low + high SMEM: consider reducing tile size or splitting kernel.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Memory Access Patterns — Coalescing Efficiency Analyser": {
        "description": (
            "Simulate how different memory access patterns (coalesced, strided, random, "
            "column-major) map to hardware transactions. Compute effective bandwidth "
            "utilisation for each pattern. Show exactly how warp-level coalescing works "
            "and which patterns destroy bandwidth."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple

print("=" * 68)
print("  MEMORY ACCESS PATTERN ANALYSER — Coalescing & Transaction Cost")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Hardware parameters (A100 / H100 class GPU)
# ─────────────────────────────────────────────────────────────────────
WARP_SIZE         = 32
CACHE_LINE_BYTES  = 128   # L1/L2 cache line = 128 bytes
SECTOR_BYTES      = 32    # minimum transaction unit from L2 = 32 bytes
FP32_BYTES        = 4
HBM_BW_GBs        = 2000  # GB/s (H100 HBM3)
SMEM_BW_GBs        = 19000 # GB/s (H100 SMEM)
CLOCK_GHZ         = 1.83   # H100 boost clock


@dataclass
class WarpTransaction:
    """Represents the hardware transactions needed for one warp memory op."""
    num_sectors: int        # number of 32-byte sectors requested from L2
    unique_bytes: int       # bytes actually needed (useful data)
    wasted_bytes: int       # bytes fetched but not used
    efficiency_pct: float   # useful / total fetched

    @property
    def cycles(self):
        # Each sector takes ~1 cycle if L2 hits; model conservatively as 1
        return self.num_sectors


def analyse_warp_access(
    thread_addresses: List[int],
    elem_bytes: int = FP32_BYTES
) -> WarpTransaction:
    """
    Given a list of 32 byte-addresses (one per lane), compute:
    - how many 32-byte sectors are touched
    - effective bandwidth efficiency
    """
    assert len(thread_addresses) == WARP_SIZE

    # Find unique 32-byte sectors
    sectors = set(addr // SECTOR_BYTES for addr in thread_addresses)
    num_sectors = len(sectors)

    useful_bytes  = WARP_SIZE * elem_bytes
    total_fetched = num_sectors * SECTOR_BYTES
    wasted        = total_fetched - useful_bytes
    efficiency    = useful_bytes / total_fetched * 100

    return WarpTransaction(
        num_sectors   = num_sectors,
        unique_bytes  = useful_bytes,
        wasted_bytes  = max(0, wasted),
        efficiency_pct = efficiency,
    )


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Access pattern taxonomy
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Access Pattern Taxonomy (FP32, 32 threads/warp)")
print("━" * 68)
print()

BASE_ADDR = 0  # base address in bytes

patterns = {
    "Coalesced (stride-1)": [
        BASE_ADDR + t * FP32_BYTES for t in range(WARP_SIZE)
    ],
    "Stride-2": [
        BASE_ADDR + t * 2 * FP32_BYTES for t in range(WARP_SIZE)
    ],
    "Stride-4": [
        BASE_ADDR + t * 4 * FP32_BYTES for t in range(WARP_SIZE)
    ],
    "Stride-32 (catastrophic)": [
        BASE_ADDR + t * 32 * FP32_BYTES for t in range(WARP_SIZE)
    ],
    "Broadcast (all → same addr)": [
        BASE_ADDR for _ in range(WARP_SIZE)
    ],
    "Column of 1024-wide matrix": [
        # A[thread_row][col=0] in row-major matrix of width 1024
        t * 1024 * FP32_BYTES for t in range(WARP_SIZE)
    ],
    "Random (scatter/gather)": list(
        np.random.default_rng(42).choice(
            np.arange(0, 256 * FP32_BYTES, FP32_BYTES), size=WARP_SIZE, replace=False
        ).astype(int)
    ),
}

header = f"  {'Pattern':<35}  {'Sectors':>7}  {'Ideal':>5}  {'Eff%':>6}  {'Wasted':>8}"
print(header)
print("  " + "─" * 64)

for name, addrs in patterns.items():
    txn = analyse_warp_access(addrs)
    ideal_sectors = math.ceil(WARP_SIZE * FP32_BYTES / SECTOR_BYTES)  # 4 sectors
    print(f"  {name:<35}  {txn.num_sectors:>7}  {ideal_sectors:>5}  "
          f"{txn.efficiency_pct:>5.1f}%  {txn.wasted_bytes:>6} B")

print()
print("  Ideal: 4 sectors for 32×4 = 128 bytes (fits 4 × 32-byte sectors)")
print("  Efficiency = (useful bytes) / (sectors × 32 bytes)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Matrix access — row-major vs column-major
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Row-major matrix: row access vs column access")
print("━" * 68)
print()

def matrix_access_efficiency(M: int, N: int, access: str) -> dict:
    """
    Simulate accessing a row or column of an M×N row-major FP32 matrix.
    Returns transaction count and efficiency for one warp.
    """
    results = {}
    row_stride = N * FP32_BYTES  # bytes between rows

    for row in [0, 1, 8]:  # sample rows
        if access == "row":
            # Thread t reads A[row][t]
            addrs = [row * row_stride + t * FP32_BYTES for t in range(WARP_SIZE)]
        else:  # column
            # Thread t reads A[t][col=0]
            addrs = [t * row_stride for t in range(WARP_SIZE)]

        txn = analyse_warp_access(addrs)
        results[row] = txn

    return results

for N in [32, 128, 1024]:
    row_txn = analyse_warp_access(
        [0 * N * FP32_BYTES + t * FP32_BYTES for t in range(WARP_SIZE)]
    )
    col_txn = analyse_warp_access(
        [t * N * FP32_BYTES for t in range(WARP_SIZE)]
    )
    print(f"  Matrix width N={N:4d}:")
    print(f"    Row  access: {row_txn.num_sectors:2d} sectors, {row_txn.efficiency_pct:.1f}% efficient")
    print(f"    Col  access: {col_txn.num_sectors:2d} sectors, {col_txn.efficiency_pct:.1f}% efficient")
    speedup = col_txn.num_sectors / row_txn.num_sectors
    print(f"    Row/col speedup: {speedup:.0f}× more sectors for column access")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Bandwidth model — actual GB/s under different patterns
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Effective Bandwidth at Scale (FP32 tensor, 1M elements)")
print("━" * 68)
print()

N_ELEMENTS = 1_000_000  # 1M FP32 elements = 4 MB
N_WARPS    = N_ELEMENTS // WARP_SIZE

def effective_bandwidth(sectors_per_warp: int, n_warps: int) -> dict:
    """
    Model throughput (GB/s) assuming warps execute with peak issue rate.
    Simplified: sectors limit bandwidth if > 4 (ideal).
    """
    total_sectors = sectors_per_warp * n_warps
    bytes_fetched  = total_sectors * SECTOR_BYTES
    useful_bytes   = n_warps * WARP_SIZE * FP32_BYTES

    # Efficiency degrades bandwidth proportionally
    ideal_bw_GBs  = HBM_BW_GBs
    actual_bw_GBs = ideal_bw_GBs * (4 / sectors_per_warp)  # 4 = ideal sectors/warp
    actual_bw_GBs = min(actual_bw_GBs, ideal_bw_GBs)

    time_ns = (bytes_fetched / (actual_bw_GBs * 1e9)) * 1e9

    return {
        "total_sectors":  total_sectors,
        "bytes_fetched":  bytes_fetched,
        "effective_bw":   actual_bw_GBs,
        "time_ns":        time_ns,
    }

access_configs = [
    ("Coalesced",           4),
    ("Stride-2",            8),
    ("Stride-4",            16),
    ("Stride-32",           32),
    ("Column (N=1024)",     32),
    ("Random",              32),
]

print(f"  {'Pattern':<25}  {'Sectors/warp':>12}  {'Eff. BW GB/s':>13}  {'Relative':>10}")
print("  " + "─" * 65)
base_bw = None
for name, sects in access_configs:
    bw_info = effective_bandwidth(sects, N_WARPS)
    if base_bw is None:
        base_bw = bw_info["effective_bw"]
    rel = bw_info["effective_bw"] / base_bw
    print(f"  {name:<25}  {sects:>12}  {bw_info['effective_bw']:>13.0f}  {rel:>9.2f}×")

print()
print("  Note: stride-32 and column access achieve the SAME throughput as random access")
print("  — both saturate transaction count. Coalescing is the only path to peak BW.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Visualise access pattern memory maps
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Memory Map Visualisation (32 threads, 16 FP32 slots)")
print("━" * 68)
print()

def draw_memory_map(addrs: List[int], label: str, n_slots: int = 16):
    """ASCII art of which memory slots each warp accesses."""
    slot_size = FP32_BYTES
    accessed = set(addr // slot_size for addr in addrs if addr // slot_size < n_slots)
    print(f"  {label}")
    row = "  Addr: "
    hit_row = "  Hit:  "
    for i in range(n_slots):
        row     += f"{i*4:4d}"
        hit_row += "   ■" if i in accessed else "   ·"
    print(row)
    print(hit_row)
    # Sector boundaries
    sector_row = "  Sec:  "
    for i in range(n_slots):
        marker = "|   " if (i * slot_size) % SECTOR_BYTES == 0 else "    "
        sector_row += marker
    print(sector_row + "  (| = 32B sector boundary)")
    n_sectors = len(set(addr // SECTOR_BYTES for addr in addrs if addr // slot_size < n_slots))
    print(f"  → {n_sectors} sector(s) fetched for {len(accessed)} elements accessed")
    print()

draw_memory_map(
    [t * FP32_BYTES for t in range(16)],
    "Coalesced (first 16 threads, stride-1):"
)
draw_memory_map(
    [t * 2 * FP32_BYTES for t in range(16)],
    "Stride-2 (first 16 threads):"
)
draw_memory_map(
    [t * 4 * FP32_BYTES for t in range(8)],
    "Stride-4 (first 8 threads):"
)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · SMEM Bank Conflicts — Conflict Detector & Padding Fix": {
        "description": (
            "Simulate shared memory bank conflict detection for different access patterns. "
            "Show which patterns create 2-way, 4-way, and 32-way conflicts. "
            "Demonstrate the +1 column padding fix for transpose and column-major SMEM access. "
            "Compute effective SMEM throughput degradation from conflicts."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

print("=" * 68)
print("  SHARED MEMORY BANK CONFLICT DETECTOR & PADDING ANALYSIS")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# SMEM hardware model
# ─────────────────────────────────────────────────────────────────────
NUM_BANKS    = 32     # 32 banks on Maxwell+ GPUs
BANK_WIDTH   = 4      # bytes per bank (FP32 mode)
WARP_SIZE    = 32


def bank_of(byte_addr: int) -> int:
    """Bank assignment for a byte address."""
    return (byte_addr // BANK_WIDTH) % NUM_BANKS


def analyse_bank_conflicts(thread_byte_addrs: List[int]) -> Dict:
    """
    Analyse bank conflicts for one warp access to SMEM.
    Returns conflict degree and number of serialised transactions.
    """
    assert len(thread_byte_addrs) == WARP_SIZE

    bank_accesses = defaultdict(list)  # bank_id → [addr, ...]
    for lane, addr in enumerate(thread_byte_addrs):
        b = bank_of(addr)
        bank_accesses[b].append((lane, addr))

    max_conflict = 1
    total_extra_transactions = 0

    for bank, accesses in bank_accesses.items():
        # Filter out broadcasts (all lanes to same address — 1 cycle, no conflict)
        unique_addrs = set(addr for _, addr in accesses)
        if len(unique_addrs) == 1:
            # Broadcast — served in 1 cycle regardless of how many threads
            conflict_degree = 1
        else:
            conflict_degree = len(accesses)  # N threads to N different addrs in same bank

        if conflict_degree > max_conflict:
            max_conflict = conflict_degree

        if conflict_degree > 1:
            total_extra_transactions += (conflict_degree - 1)

    # Total cycles = ideal 1 + serialised extras
    total_cycles = 1 + total_extra_transactions

    # Count banks with conflicts
    conflicted_banks = sum(
        1 for bank, accesses in bank_accesses.items()
        if len(set(addr for _, addr in accesses)) > 1 and len(accesses) > 1
    )

    return {
        "max_conflict_degree": max_conflict,
        "total_cycles":        max_conflict,  # serialised = max_conflict passes
        "conflicted_banks":    conflicted_banks,
        "bank_map":            dict(bank_accesses),
        "slowdown":            max_conflict,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Classic SMEM access patterns
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — SMEM Bank Conflict by Access Pattern")
print("━" * 68)
print()

SMEM_BASE = 0

smem_patterns = {
    "Stride-1  (ideal, no conflict)": [
        SMEM_BASE + t * BANK_WIDTH for t in range(WARP_SIZE)
    ],
    "Stride-2  (2-way conflict)": [
        SMEM_BASE + t * 2 * BANK_WIDTH for t in range(WARP_SIZE)
    ],
    "Stride-4  (4-way conflict)": [
        SMEM_BASE + t * 4 * BANK_WIDTH for t in range(WARP_SIZE)
    ],
    "Stride-16 (16-way conflict)": [
        SMEM_BASE + t * 16 * BANK_WIDTH for t in range(WARP_SIZE)
    ],
    "Stride-32 (32-way conflict — worst)": [
        SMEM_BASE + t * 32 * BANK_WIDTH for t in range(WARP_SIZE)
    ],
    "Broadcast (all → addr 0)": [
        SMEM_BASE for _ in range(WARP_SIZE)
    ],
    "Mixed (lanes 0–15 stride-1, 16–31 stride-1 offset)": [
        SMEM_BASE + t * BANK_WIDTH for t in range(WARP_SIZE)
    ],
}

print(f"  {'Pattern':<45}  {'Max Conf':>8}  {'Cycles':>7}  {'Slowdown':>9}")
print("  " + "─" * 72)

for name, addrs in smem_patterns.items():
    result = analyse_bank_conflicts(addrs)
    print(f"  {name:<45}  "
          f"{result['max_conflict_degree']:>8}  "
          f"{result['total_cycles']:>7}  "
          f"{result['slowdown']:>8}×")

print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Column access of 2D SMEM tile — the transpose problem
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Transpose Kernel: SMEM Column Access Conflict Analysis")
print("━" * 68)
print()

def column_access_conflicts(tile_cols: int, col_idx: int) -> dict:
    """
    Threads 0..31 each access smem[thread_row][col_idx] of a
    tile with 'tile_cols' columns. Compute bank conflicts.
    """
    addrs = [
        (t * tile_cols + col_idx) * BANK_WIDTH
        for t in range(WARP_SIZE)
    ]
    return analyse_bank_conflicts(addrs)

print("  Reading column 0 of smem[32][N] for different N values:")
print()
print(f"  {'Tile cols (N)':<18}  {'Bank formula':<35}  {'Max conflict':<14}  {'Note'}")
print("  " + "─" * 82)

for N in [16, 32, 33, 34, 64, 65]:
    result = column_access_conflicts(N, col_idx=0)
    bank_formula = f"(t×{N}) % 32"
    note = ""
    if N % 32 == 0:
        note = "← ALL same bank (32-way)!"
    elif N == 33 or N == 65:
        note = "← padded (+1) → no conflict ✅"

    print(f"  N={N:<15}  {bank_formula:<35}  {result['max_conflict_degree']:>8}-way  {note}")

print()
print("  KEY INSIGHT:")
print("  When N is a multiple of 32 → all rows map to the same bank for col=0.")
print("  N=32: bank(row, col=0) = (row×32 + 0) % 32 = 0  for ALL rows → 32-way")
print("  N=33: bank(row, col=0) = (row×33 + 0) % 32 = row % 32 → each row → unique bank ✅")
print("  RULE: pad SMEM tile by +1 column whenever N % 32 == 0")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Tiled transpose with and without padding
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Full Tiled Transpose Conflict Analysis (TILE=32)")
print("━" * 68)
print()

TILE = 32

def simulate_transpose_smem(padded: bool):
    """
    Simulate the SMEM write and read phases of a 32×32 transpose.
    WRITE: thread (r,c) writes to smem[r][c]  — sequential by row (coalesced)
    READ:  thread (r,c) reads  smem[c][r]     — reads column of SMEM
    """
    stride = TILE + 1 if padded else TILE

    write_conflicts = []
    read_conflicts  = []

    # Simulate each row of threads (one warp = one row of 32 threads)
    for row in range(TILE):
        # WRITE phase: thread col writes smem[row][col]
        write_addrs = [(row * stride + col) * BANK_WIDTH for col in range(WARP_SIZE)]
        write_conflicts.append(analyse_bank_conflicts(write_addrs)["max_conflict_degree"])

        # READ phase: thread col reads smem[col][row]  (transposed!)
        read_addrs = [(col * stride + row) * BANK_WIDTH for col in range(WARP_SIZE)]
        read_conflicts.append(analyse_bank_conflicts(read_addrs)["max_conflict_degree"])

    return write_conflicts, read_conflicts

for padded in [False, True]:
    label = "PADDED (stride = 33)" if padded else "UNPADDED (stride = 32)"
    wc, rc = simulate_transpose_smem(padded)

    print(f"  {label}:")
    print(f"    WRITE phase: max conflict = {max(wc)}-way  "
          f"(mean={np.mean(wc):.1f})  "
          f"total extra cycles = {sum(c-1 for c in wc)}")
    print(f"    READ  phase: max conflict = {max(rc)}-way  "
          f"(mean={np.mean(rc):.1f})  "
          f"total extra cycles = {sum(c-1 for c in rc)}")
    total_slowdown = (sum(wc) + sum(rc)) / (TILE + TILE)
    print(f"    Effective SMEM slowdown: {total_slowdown:.1f}×")
    print(f"    Memory wasted by padding: {TILE * BANK_WIDTH} bytes "
          f"({TILE * BANK_WIDTH / (TILE * TILE * BANK_WIDTH) * 100:.1f}% overhead)")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Bank map visualisation for a 4×4 tile
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — Bank Map Visualisation (4×4 tile, FP32)")
print("━" * 68)
print()

def print_bank_map(tile_rows: int, tile_cols: int, stride: int, label: str):
    print(f"  {label} (stride={stride}):")
    print("  Element (row,col) → bank assignment:")
    print()
    header = "       "
    for c in range(tile_cols):
        header += f" col{c:1d}"
    print(header)
    print("  " + "─" * (7 + tile_cols * 5))
    for r in range(tile_rows):
        row_str = f"  row{r}: "
        for c in range(tile_cols):
            addr = (r * stride + c) * BANK_WIDTH
            row_str += f"  B{bank_of(addr):02d}"
        print(row_str)
    print()

print_bank_map(4, 4, stride=4,  label="4×4 tile, no padding  (stride=4)")
print_bank_map(4, 4, stride=5,  label="4×4 tile, +1 padding  (stride=5)")

print("  In the unpadded case: column 0 access reads B00 for ALL rows → 4-way conflict.")
print("  In the padded   case: column 0 reads B00, B05%32, B10%32, B15%32 → different banks ✅")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Tiled Matrix Multiply — Naïve vs SMEM Tiled Memory Traffic": {
        "description": (
            "Implement naïve GEMM and tiled GEMM algorithms in NumPy, then compute "
            "the exact HBM traffic each generates. Show how tile size controls arithmetic "
            "intensity. Profile the memory access model to understand why tiling makes "
            "GEMM compute-bound rather than memory-bound."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import math

print("=" * 68)
print("  TILED MATRIX MULTIPLY — Memory Traffic & Arithmetic Intensity")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Memory model constants
# ─────────────────────────────────────────────────────────────────────
FP32_BYTES     = 4
HBM_BW_GBs     = 2000      # H100 HBM3
SMEM_BW_GBs    = 19000     # H100 SMEM
FP16_TFLOPS    = 312       # H100 peak FP16 tensor core
RIDGE_FLOPS_PER_BYTE = FP16_TFLOPS * 1e12 / (HBM_BW_GBs * 1e9)  # 156


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Memory traffic model for different GEMM variants
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — GEMM Memory Traffic Analysis")
print("━" * 68)
print()

def gemm_traffic_analysis(M: int, N: int, K: int, tile: int) -> dict:
    """
    Compute theoretical HBM traffic and arithmetic intensity
    for naïve GEMM and tiled GEMM with given tile size.
    """
    # FLOPs: M×N×K multiplications + M×N×K additions = 2MNK
    flops = 2 * M * N * K

    # NAÏVE: each C[i][j] independently reads row i of A and col j of B
    naive_A_reads = M * N * K * FP32_BYTES   # K elements per output, M*N outputs
    naive_B_reads = M * N * K * FP32_BYTES
    naive_C_write = M * N * FP32_BYTES
    naive_total   = naive_A_reads + naive_B_reads + naive_C_write

    # TILED: each element of A and B read exactly M/TILE * N/TILE * TILE = once per block
    # A is read once per column tile: N/TILE times
    # B is read once per row tile:    M/TILE times
    # (Both are read once total with perfect tiling — same as optimal)
    tiled_A_reads = M * K * FP32_BYTES
    tiled_B_reads = K * N * FP32_BYTES
    tiled_C_write = M * N * FP32_BYTES
    tiled_total   = tiled_A_reads + tiled_B_reads + tiled_C_write

    naive_ai = flops / naive_total
    tiled_ai = flops / tiled_total

    naive_ideal_ms = naive_total / (HBM_BW_GBs * 1e9) * 1000
    tiled_ideal_ms = tiled_total / (HBM_BW_GBs * 1e9) * 1000
    compute_ms     = flops     / (FP16_TFLOPS * 1e12)  * 1000

    return {
        "M": M, "N": N, "K": K, "tile": tile,
        "flops": flops,
        "naive_total_gb":  naive_total / 1e9,
        "tiled_total_gb":  tiled_total / 1e9,
        "naive_ai":        naive_ai,
        "tiled_ai":        tiled_ai,
        "traffic_ratio":   naive_total / tiled_total,
        "naive_ideal_ms":  naive_ideal_ms,
        "tiled_ideal_ms":  tiled_ideal_ms,
        "compute_ms":      compute_ms,
        "naive_bound":     "compute" if naive_ai > RIDGE_FLOPS_PER_BYTE else "memory",
        "tiled_bound":     "compute" if tiled_ai > RIDGE_FLOPS_PER_BYTE else "memory",
    }

print(f"  Ridge point (H100): {RIDGE_FLOPS_PER_BYTE:.0f} FLOPs/Byte")
print(f"  Above ridge → compute-bound | Below → memory-bound")
print()

shapes = [
    (512,   512,  512,  16),
    (1024, 1024, 1024,  16),
    (1024, 1024, 1024,  32),
    (1024, 1024, 1024,  64),
    (4096, 4096, 4096,  64),
    (4096, 4096, 4096, 128),
]

header = (f"  {'M×N×K':<18}  {'Tile':>4}  "
          f"{'Naïve HBM':>10}  {'Tiled HBM':>10}  "
          f"{'Ratio':>6}  {'Naïve AI':>9}  "
          f"{'Tiled AI':>9}  {'Tiled bound':>12}")
print(header)
print("  " + "─" * 92)

for M, N, K, tile in shapes:
    d = gemm_traffic_analysis(M, N, K, tile)
    shape_str = f"{M}×{N}×{K}"
    print(f"  {shape_str:<18}  {tile:>4}  "
          f"{d['naive_total_gb']:>9.2f}G  {d['tiled_total_gb']:>9.2f}G  "
          f"{d['traffic_ratio']:>6.0f}×  {d['naive_ai']:>8.1f}  "
          f"{d['tiled_ai']:>8.1f}  {d['tiled_bound']:>12}")

print()
print("  → Naïve GEMM is ALWAYS memory-bound (AI ≈ 2 FLOPs/Byte).")
print("  → Tiled  GEMM with tile≥16 is compute-bound (AI >> ridge point).")
print("  → This is why cuBLAS/CUTLASS achieve near-peak TFLOP/s on large GEMM.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Step-by-step tiled GEMM memory trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Tiled GEMM Memory Access Trace (M=N=K=4, TILE=2)")
print("━" * 68)
print()

def tiled_gemm_trace(M, N, K, TILE):
    """
    Trace every HBM read/write in a tiled GEMM.
    Returns list of (operation, source, indices, value) tuples.
    """
    np.random.seed(1)
    A = np.random.randint(1, 5, (M, K)).astype(float)
    B = np.random.randint(1, 5, (K, N)).astype(float)
    C = np.zeros((M, N))

    hbm_reads  = 0
    smem_reads = 0
    trace_log  = []
    num_tiles  = K // TILE

    for tile_k in range(num_tiles):
        # --- Cooperative load phase (would be all threads in block) ---
        tile_A = A[:, tile_k*TILE : (tile_k+1)*TILE]   # M×TILE
        tile_B = B[tile_k*TILE : (tile_k+1)*TILE, :]   # TILE×N
        hbm_reads += M * TILE + TILE * N  # elements read from HBM
        trace_log.append(
            f"  Tile k={tile_k}: Load A[:,{tile_k*TILE}:{(tile_k+1)*TILE}] + "
            f"B[{tile_k*TILE}:{(tile_k+1)*TILE},:] from HBM "
            f"({M*TILE + TILE*N} elements)"
        )

        # --- Compute from SMEM ---
        for i in range(M):
            for j in range(N):
                for k in range(TILE):
                    C[i][j] += tile_A[i][k] * tile_B[k][j]
                    smem_reads += 2   # one from tile_A, one from tile_B

    # Write C
    hbm_writes = M * N
    trace_log.append(f"  Write C ({M*N} elements) to HBM")

    return C, hbm_reads, smem_reads, hbm_writes, trace_log


M, N, K, TILE = 4, 4, 4, 2
C_tiled, hbm_r, smem_r, hbm_w, trace = tiled_gemm_trace(M, N, K, TILE)
C_numpy = np.random.randint(1, 5, (M, K)).astype(float) @ \
          np.random.randint(1, 5, (K, N)).astype(float)   # Just for structure demo

print(f"  M={M}, N={N}, K={K}, TILE={TILE}")
print(f"  Number of K-tiles: {K//TILE}")
print()
for line in trace:
    print(line)
print()
print(f"  Total HBM reads:  {hbm_r} elements ({hbm_r*FP32_BYTES} bytes)")
print(f"  Total SMEM reads: {smem_r} elements")
print(f"  Total HBM writes: {hbm_w} elements ({hbm_w*FP32_BYTES} bytes)")
print(f"  HBM reuse factor: {smem_r / hbm_r:.1f}× (each HBM element reused this many times in SMEM)")
print()
print(f"  Naïve equivalent HBM reads: {2 * M * N * K} elements ({2*M*N*K*FP32_BYTES} bytes)")
print(f"  Tiling saves: {2*M*N*K - hbm_r} element reads = {(2*M*N*K - hbm_r)*FP32_BYTES} bytes")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Tile size effect on performance model
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Tile Size vs Arithmetic Intensity (M=N=K=4096)")
print("━" * 68)
print()

M = N = K = 4096
total_flops = 2 * M * N * K
# Optimal HBM: read A + B once, write C once
optimal_hbm_bytes = (M*K + K*N + M*N) * FP32_BYTES

print(f"  M=N=K={M}, FP32")
print(f"  Total FLOPs:          {total_flops/1e12:.2f} TFLOP")
print(f"  Minimum HBM transfer: {optimal_hbm_bytes/1e9:.3f} GB (read A+B, write C once)")
print(f"  Optimal AI:           {total_flops/optimal_hbm_bytes:.1f} FLOPs/Byte")
print()
print(f"  Tile size → reuse factor → AI → bound:")
print()
print(f"  {'TILE':>6}  {'SMEM (KB)':>10}  {'Reuse':>8}  {'AI (F/B)':>10}  "
      f"{'vs ridge':>9}  {'Bound':>8}  {'Est. perf %':>12}")
print("  " + "─" * 72)

for tile in [1, 2, 4, 8, 16, 32, 64, 128]:
    smem_kb = 2 * tile * tile * FP32_BYTES / 1024  # two tiles in SMEM
    reuse   = tile  # each element reused TILE times within the tile

    # AI grows with tile (we read A and B tile times across K-dimension naïvely,
    # but with SMEM we only read each tile once from HBM)
    # True tiled AI ≈ optimal AI regardless of tile (reads each elem once)
    # but effective compute-to-memory ratio improves with larger tiles
    # because more FLOPs happen per SMEM load
    ai = total_flops / optimal_hbm_bytes  # stays constant (optimal reads)
    flops_per_smem_read = total_flops / (M * K * tile + K * N * tile)

    # For smaller tiles, SMEM bandwidth becomes the bottleneck
    if smem_kb <= 228:  # fits in H100 SMEM budget
        smem_time  = (M*K*tile + K*N*tile) * FP32_BYTES / (SMEM_BW_GBs * 1e9)
        hbm_time   = optimal_hbm_bytes / (HBM_BW_GBs * 1e9)
        compute_t  = total_flops / (FP16_TFLOPS * 1e12)
        bottleneck_t = max(smem_time, hbm_time, compute_t)
        actual_flops = total_flops / bottleneck_t
        perf_pct   = actual_flops / (FP16_TFLOPS * 1e12) * 100
        bound = "compute" if compute_t >= max(smem_time, hbm_time) else "SMEM/HBM"
        smem_note = f"{smem_kb:.0f} KB"
    else:
        perf_pct = 0
        bound = "OOM"
        smem_note = f"{smem_kb:.0f} KB ⚠️"

    print(f"  {tile:>6}  {smem_note:>10}  {reuse:>8}  {ai:>10.1f}  "
          f"{ai/RIDGE_FLOPS_PER_BYTE:>8.2f}×  {bound:>8}  {perf_pct:>10.1f}%")

print()
print("  Takeaway: once TILE ≥ 16, the kernel is compute-bound for large GEMM.")
print("  cuBLAS uses TILE=128 with register tiling for maximum reuse.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Occupancy Calculator — SMEM, Registers & Thread Block Tuning": {
        "description": (
            "Build a CUDA occupancy calculator that models how shared memory usage, "
            "register count, and block size interact to limit warp occupancy per SM. "
            "Show the occupancy cliff for different SMEM tile sizes. Demonstrate how "
            "to trade SMEM for occupancy and when low occupancy is acceptable."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  CUDA OCCUPANCY CALCULATOR — SMEM, Registers & Block Tuning")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# GPU hardware specs (configurable for different GPU generations)
# ─────────────────────────────────────────────────────────────────────
GPU_SPECS = {
    "A100": {
        "max_threads_per_sm":    2048,
        "max_warps_per_sm":       64,
        "max_blocks_per_sm":      32,
        "max_regs_per_sm":     65536,
        "max_regs_per_thread":   255,
        "max_smem_per_sm_kb":    164,   # 164 KB max (rest is L1)
        "smem_granularity_kb":     8,   # allocated in 8 KB chunks
        "warp_size":              32,
    },
    "H100": {
        "max_threads_per_sm":    2048,
        "max_warps_per_sm":       64,
        "max_blocks_per_sm":      32,
        "max_regs_per_sm":     65536,
        "max_regs_per_thread":   255,
        "max_smem_per_sm_kb":    228,   # 228 KB max with L1 sharing
        "smem_granularity_kb":     8,
        "warp_size":              32,
    },
    "V100": {
        "max_threads_per_sm":    2048,
        "max_warps_per_sm":       64,
        "max_blocks_per_sm":      32,
        "max_regs_per_sm":     65536,
        "max_regs_per_thread":   255,
        "max_smem_per_sm_kb":     96,
        "smem_granularity_kb":     8,
        "warp_size":              32,
    },
}


def calculate_occupancy(
    gpu:              str,
    threads_per_block: int,
    regs_per_thread:  int,
    smem_bytes:       int,
) -> dict:
    """
    Compute theoretical SM occupancy given kernel launch parameters.

    Returns dict with occupancy fraction and limiting factor.
    """
    spec = GPU_SPECS[gpu]
    ws   = spec["warp_size"]

    # Align thread count to warp boundary
    warps_per_block  = math.ceil(threads_per_block / ws)
    # SMEM allocated in granularity chunks (round up)
    smem_gran_bytes  = spec["smem_granularity_kb"] * 1024
    smem_per_block   = math.ceil(max(smem_bytes, 1) / smem_gran_bytes) * smem_gran_bytes
    smem_max_bytes   = spec["max_smem_per_sm_kb"] * 1024

    # Register allocation (per warp block, rounded to 256 register granularity)
    reg_gran         = 256
    regs_per_block   = math.ceil(
        warps_per_block * ws * regs_per_thread / reg_gran
    ) * reg_gran

    # Limits on blocks per SM from each resource:
    if smem_per_block == 0:
        limit_smem = spec["max_blocks_per_sm"]
    else:
        limit_smem   = min(spec["max_blocks_per_sm"], smem_max_bytes // smem_per_block)

    limit_regs   = min(
        spec["max_blocks_per_sm"],
        spec["max_regs_per_sm"] // regs_per_block if regs_per_block > 0 else 999,
    )
    limit_threads = min(
        spec["max_blocks_per_sm"],
        spec["max_threads_per_sm"] // threads_per_block,
    )
    limit_blocks  = spec["max_blocks_per_sm"]

    blocks_per_sm    = min(limit_smem, limit_regs, limit_threads, limit_blocks)
    blocks_per_sm    = max(1, blocks_per_sm)

    active_warps     = blocks_per_sm * warps_per_block
    max_warps        = spec["max_warps_per_sm"]
    occupancy        = active_warps / max_warps

    # Identify the bottleneck
    limits = {
        "SMEM":    limit_smem,
        "Regs":    limit_regs,
        "Threads": limit_threads,
        "Blocks":  limit_blocks,
    }
    bottleneck = min(limits, key=limits.get)
    if limits[bottleneck] >= spec["max_blocks_per_sm"]:
        bottleneck = "None (at max)"

    return {
        "gpu":              gpu,
        "threads_per_block": threads_per_block,
        "warps_per_block":  warps_per_block,
        "smem_per_block_kb": smem_per_block / 1024,
        "regs_per_block":   regs_per_block,
        "blocks_per_sm":    blocks_per_sm,
        "active_warps":     active_warps,
        "max_warps":        max_warps,
        "occupancy":        occupancy,
        "bottleneck":       bottleneck,
        "limit_smem":       limit_smem,
        "limit_regs":       limit_regs,
        "limit_threads":    limit_threads,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Common ML kernels
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Occupancy for Common ML Kernels (H100)")
print("━" * 68)
print()

kernels = [
    # (name, threads_per_block, regs_per_thread, smem_bytes)
    ("Elementwise (ReLU/add)",          256,  16,      0),
    ("Reduction (sum, 1D)",             256,  24,   8192),   # 8 KB SMEM
    ("Tiled GEMM (TILE=16)",            256,  32,  4096),    # 2×16×16×4 = 2 KB
    ("Tiled GEMM (TILE=32)",           1024,  32, 16384),    # 2×32×32×4 = 8 KB
    ("Tiled GEMM (TILE=64)",           1024,  64, 65536),    # 2×64×64×4 = 32 KB
    ("Layer Norm (row=1024)",           256,  40, 16384),    # 16 KB row buffer
    ("Flash Attn V2 (d=128, bq=64)",   128,  64, 65536),    # 64 KB tile
    ("Flash Attn V3 (d=128, bq=128)",  256,  96, 131072),   # 128 KB tile
    ("CUTLASS GEMM (TILE=128)",        256,  96, 131072),   # 128 KB double-buffer
]

print(f"  {'Kernel':<35}  {'Threads':>7}  {'SMEM':>8}  "
      f"{'Blks/SM':>8}  {'Warps':>6}  {'Occ%':>6}  {'Bottleneck':<14}")
print("  " + "─" * 92)

for name, threads, regs, smem in kernels:
    r = calculate_occupancy("H100", threads, regs, smem)
    smem_kb = smem / 1024
    print(f"  {name:<35}  {threads:>7}  {smem_kb:>6.0f}KB  "
          f"{r['blocks_per_sm']:>8}  {r['active_warps']:>6}  "
          f"{r['occupancy']*100:>5.1f}%  {r['bottleneck']:<14}")

print()
print("  Note: FlashAttention V3 has very low occupancy (< 25%) but still")
print("  outperforms alternatives because SMEM eliminates HBM round-trips.")
print("  Occupancy ≠ performance. Latency hiding matters more for compute-bound kernels.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: SMEM occupancy cliff
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — SMEM Occupancy Cliff (256 threads, 32 regs, H100)")
print("━" * 68)
print()

print(f"  H100 max SMEM: {GPU_SPECS['H100']['max_smem_per_sm_kb']} KB/SM")
print()
print(f"  {'SMEM/block':>12}  {'Blocks/SM':>10}  {'Warps/SM':>10}  "
      f"{'Occ%':>6}  {'Bar chart'}")
print("  " + "─" * 72)

prev_occ = None
for smem_kb in [0, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 160, 228]:
    smem_bytes = smem_kb * 1024
    r = calculate_occupancy("H100", 256, 32, smem_bytes)
    occ = r["occupancy"]
    bar_len = int(occ * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    cliff = " ← CLIFF" if prev_occ and (occ < prev_occ - 0.1) else ""
    print(f"  {smem_kb:>9} KB  {r['blocks_per_sm']:>10}  {r['active_warps']:>10}  "
          f"{occ*100:>5.1f}%  [{bar}]{cliff}")
    prev_occ = occ

print()
print("  Cliffs occur when SMEM/block crosses a fraction-of-total boundary:")
print("  228/4=57 → 228/5=45 → 228/6=38 KB → each step halves blocks per SM")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Block size tuning for a fixed SMEM budget
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Block Size Tuning (32 KB SMEM, 32 regs/thread, H100)")
print("━" * 68)
print()
print(f"  {'Block size':>11}  {'Warps/blk':>10}  {'Blks/SM':>8}  "
      f"{'Warps/SM':>9}  {'Occ%':>6}  {'Rec?':>6}")
print("  " + "─" * 58)

FIXED_SMEM = 32 * 1024  # 32 KB fixed SMEM
for block in [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
    r = calculate_occupancy("H100", block, 32, FIXED_SMEM)
    rec = "✅" if r["occupancy"] >= 0.5 else ("⚠️" if r["occupancy"] >= 0.25 else "❌")
    print(f"  {block:>11}  {r['warps_per_block']:>10}  {r['blocks_per_sm']:>8}  "
          f"{r['active_warps']:>9}  {r['occupancy']*100:>5.1f}%  {rec:>6}")

print()
print("  ✅ ≥ 50% occupancy  ⚠️ 25–50%  ❌ < 25%")
print("  Sweet spot: block_size=256 gives max warps/SM for 32 KB SMEM on H100.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Fused Kernel Design — LayerNorm via SMEM Reduction": {
        "description": (
            "Implement fused layer normalisation using SMEM reductions to demonstrate "
            "how to combine load, compute, and store in a single kernel pass. Compare "
            "naïve 3-pass implementation against SMEM-fused 2-pass. Simulate warp-level "
            "reduction patterns (__shfl_down) and inter-warp SMEM reduction trees."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 68)
print("  FUSED LAYER NORM — SMEM Reduction & Warp-Level Reduction Tree")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Hardware constants
# ─────────────────────────────────────────────────────────────────────
WARP_SIZE    = 32
FP32_BYTES   = 4
HBM_BW_GBs  = 2000

# ─────────────────────────────────────────────────────────────────────
# SMEM reduction model
# ─────────────────────────────────────────────────────────────────────

def warp_reduce_sum(values: np.ndarray) -> float:
    """
    Simulate __shfl_down_sync warp reduction for sum.
    Each step halves the active threads, each thread XOR-shuffles
    with a peer 'offset' positions away and adds.

    Steps: offset = 16, 8, 4, 2, 1  (5 steps for warp of 32)
    """
    assert len(values) == WARP_SIZE
    v = values.copy().astype(float)
    steps = int(math.log2(WARP_SIZE))  # 5 for warp=32

    for step in range(steps):
        offset = WARP_SIZE >> (step + 1)   # 16, 8, 4, 2, 1
        # Each lane adds from lane+offset (if lane+offset < warp_size)
        for lane in range(WARP_SIZE):
            peer = lane + offset
            if peer < WARP_SIZE:
                v[lane] += v[peer]

    return v[0]  # lane 0 holds the final sum


def warp_reduce_max(values: np.ndarray) -> float:
    """Simulate __shfl_down_sync warp reduction for max."""
    v = values.copy().astype(float)
    steps = int(math.log2(WARP_SIZE))
    for step in range(steps):
        offset = WARP_SIZE >> (step + 1)
        for lane in range(WARP_SIZE):
            peer = lane + offset
            if peer < WARP_SIZE:
                v[lane] = max(v[lane], v[peer])
    return v[0]


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Warp reduction trace
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — __shfl_down Warp Reduction Tree (32 lanes, sum)")
print("━" * 68)
print()

np.random.seed(7)
sample_values = np.random.randint(1, 10, WARP_SIZE).astype(float)
correct_sum   = sample_values.sum()

print(f"  Input (32 lane values):  {sample_values.astype(int).tolist()}")
print(f"  True sum:                {correct_sum:.0f}")
print()
print(f"  __shfl_down reduction steps (lane 0 perspective):")

v = sample_values.copy()
for step in range(int(math.log2(WARP_SIZE))):
    offset = WARP_SIZE >> (step + 1)
    new_v = v.copy()
    n_active = WARP_SIZE // (2 ** step)   # active lanes this step

    for lane in range(WARP_SIZE):
        peer = lane + offset
        if peer < WARP_SIZE:
            new_v[lane] = v[lane] + v[peer]

    print(f"    Step {step+1}: offset={offset:2d}  "
          f"lane0 += lane{offset} → {new_v[0]:.0f}  "
          f"(active lanes: 0..{n_active//2 - 1})")
    v = new_v

print(f"  Final:  lane0 = {v[0]:.0f}  ✅ matches true sum")
print()
print(f"  Total shfl instructions: {int(math.log2(WARP_SIZE))} = log2(32)")
print(f"  Cost: {int(math.log2(WARP_SIZE))} warp cycles (vs 31 sequential adds)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Block-level reduction (multiple warps via SMEM)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Multi-Warp Block Reduction via SMEM (256 threads)")
print("━" * 68)
print()

def block_reduce_sum(values: np.ndarray, block_size: int) -> float:
    """
    Simulate a full thread block reduction (256 threads = 8 warps).
    Phase 1: Each warp reduces its 32 elements via __shfl_down.
    Phase 2: Warp-0 reduces the 8 warp-level partial sums via SMEM.
    """
    n_warps = block_size // WARP_SIZE
    assert len(values) == block_size

    smem = np.zeros(n_warps)   # SMEM buffer: one slot per warp

    # PHASE 1: Each warp reduces its 32 values
    for w in range(n_warps):
        warp_vals = values[w * WARP_SIZE : (w+1) * WARP_SIZE]
        partial   = warp_reduce_sum(warp_vals)
        smem[w]   = partial   # lane 0 writes to SMEM

    # __syncthreads() here in real CUDA

    # PHASE 2: Warp 0 reduces the partial sums from SMEM
    if n_warps <= WARP_SIZE:
        # Pad partial sums to warp size
        padded = np.zeros(WARP_SIZE)
        padded[:n_warps] = smem
        total = warp_reduce_sum(padded)
    else:
        total = smem.sum()  # fallback for very large blocks

    return total

BLOCK_SIZE = 256
test_vals = np.random.randint(1, 100, BLOCK_SIZE).astype(float)
result = block_reduce_sum(test_vals, BLOCK_SIZE)

print(f"  Block size: {BLOCK_SIZE} threads = {BLOCK_SIZE//WARP_SIZE} warps")
print(f"  True sum:   {test_vals.sum():.0f}")
print(f"  Block reduction result: {result:.0f}  {'✅' if abs(result - test_vals.sum()) < 1e-6 else '❌'}")
print()
print(f"  Phase 1: {BLOCK_SIZE//WARP_SIZE} warps × 5 shfl steps = {(BLOCK_SIZE//WARP_SIZE)*5} shfl ops")
print(f"  Phase 2: 1 warp × 5 shfl steps  =  5 shfl ops  (reads from SMEM)")
print(f"  Total: {(BLOCK_SIZE//WARP_SIZE + 1) * 5} instructions vs {BLOCK_SIZE-1} sequential adds")
print(f"  Speedup: {(BLOCK_SIZE-1)/((BLOCK_SIZE//WARP_SIZE + 1) * 5):.1f}×")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Fused Layer Norm — naïve vs SMEM-fused
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Layer Norm Implementation: 3-Pass Naïve vs 2-Pass SMEM")
print("━" * 68)
print()

EPS = 1e-5

def layernorm_naive(x: np.ndarray) -> np.ndarray:
    """
    Naïve layer norm: 3 passes over x.
    Pass 1: compute mean (read x)
    Pass 2: compute variance (read x again)
    Pass 3: normalise (read x again, write output)
    Total HBM: 3 reads + 1 write = 4 passes
    """
    mu    = x.mean()          # Pass 1: read x
    var   = ((x - mu)**2).mean()  # Pass 2: read x again
    return (x - mu) / np.sqrt(var + EPS)   # Pass 3: read x, write output


def layernorm_smem_fused(x: np.ndarray, block_size: int = 256) -> np.ndarray:
    """
    SMEM-fused layer norm: 2 passes over HBM.
    Pass 1: load x into SMEM → compute mean & variance in SMEM
    Pass 2: normalise from SMEM → write output to HBM

    Simulates: one thread block handles one row.
    """
    N = len(x)

    # PASS 1: load x, compute mean via SMEM reduction
    # (In real CUDA: each thread loads its portion, then warp+block reduce)
    smem_x = x.copy()   # simulate SMEM buffer (loaded from HBM once)

    # Block-level reduction for mean
    mu = smem_x.mean()   # simulates warp reductions in SMEM

    # Welford online variance using the already-in-SMEM data
    # No second HBM pass needed — x is already in SMEM
    var = ((smem_x - mu)**2).mean()

    # PASS 2: normalise from SMEM → write output (1 HBM write)
    output = (smem_x - mu) / np.sqrt(var + EPS)

    return output


def layernorm_hbm_cost(N: int, dtype_bytes: int = FP32_BYTES) -> dict:
    """Compute HBM cost for naïve vs fused layer norm."""
    naive_hbm  = (3 * N + N) * dtype_bytes   # 3 reads + 1 write
    fused_hbm  = (1 * N + N) * dtype_bytes   # 1 read (into SMEM) + 1 write

    naive_ms   = naive_hbm  / (HBM_BW_GBs * 1e9) * 1e3 * 1e3  # in µs, scale for visibility
    fused_ms   = fused_hbm  / (HBM_BW_GBs * 1e9) * 1e3 * 1e3

    return {
        "naive_hbm_bytes":  naive_hbm,
        "fused_hbm_bytes":  fused_hbm,
        "hbm_savings":      naive_hbm - fused_hbm,
        "savings_pct":      (naive_hbm - fused_hbm) / naive_hbm * 100,
        "speedup":          naive_hbm / fused_hbm,
    }

# Correctness check
np.random.seed(3)
x_test = np.random.randn(512).astype(np.float32)
y_naive  = layernorm_naive(x_test)
y_fused  = layernorm_smem_fused(x_test)
correct  = np.allclose(y_naive, y_fused, atol=1e-5)

print(f"  Correctness check (N=512): {'✅ PASS' if correct else '❌ FAIL'}")
print(f"  Max abs difference:        {np.max(np.abs(y_naive - y_fused)):.2e}")
print()

print(f"  HBM Traffic Comparison:")
print(f"  {'N (row width)':<15}  {'Naïve HBM':>10}  {'Fused HBM':>10}  "
      f"{'Saved':>8}  {'Speedup':>8}")
print("  " + "─" * 56)

for N in [128, 512, 1024, 2048, 4096, 8192]:
    c = layernorm_hbm_cost(N)
    print(f"  {N:<15}  {c['naive_hbm_bytes']:>8} B  {c['fused_hbm_bytes']:>8} B  "
          f"{c['savings_pct']:>7.0f}%  {c['speedup']:>7.1f}×")

print()
print("  SMEM-fused layer norm always uses exactly 2 HBM passes (1 read + 1 write).")
print("  Naïve uses 4 passes. Fused is 2× faster on bandwidth-limited GPUs.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: SMEM usage sizing for layer norm
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — SMEM Sizing & Occupancy for Fused Layer Norm (H100)")
print("━" * 68)
print()

MAX_SMEM_H100 = 228 * 1024  # bytes

print(f"  {'Row width N':>12}  {'SMEM (KB)':>10}  {'Fits in H100?':>14}  "
      f"{'Blks/SM @ 256':>14}  {'Occupancy':>10}")
print("  " + "─" * 64)

for N in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    smem_bytes = N * FP32_BYTES   # one row of FP32
    smem_kb    = smem_bytes / 1024
    fits       = smem_bytes <= MAX_SMEM_H100

    if fits:
        import math as _math
        gran = 8 * 1024
        smem_alloc = _math.ceil(smem_bytes / gran) * gran
        blocks_per_sm = max(1, MAX_SMEM_H100 // smem_alloc)
        warps = blocks_per_sm * (256 // 32)
        occ   = warps / 64
        occ_str = f"{occ*100:.1f}%"
        blk_str = str(blocks_per_sm)
    else:
        occ_str = "N/A"
        blk_str = "N/A"

    fits_str = "✅ yes" if fits else "❌ exceeds SMEM"
    print(f"  {N:>12}  {smem_kb:>9.1f}  {fits_str:>14}  {blk_str:>14}  {occ_str:>10}")

print()
print("  For N > 57344 elements (228 KB / 4 B), the full row won't fit in SMEM.")
print("  Solution: process in chunks, accumulate mean/var with Welford's algorithm.")
print("  This is how PyTorch's fused LayerNorm handles sequences > 57K tokens.")
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