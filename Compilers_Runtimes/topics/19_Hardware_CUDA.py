"""
CUDA — GPU Architecture, Thread Hierarchy, and Parallel Computing
==================================================================

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing
platform and programming model. Released in 2006, CUDA transformed the GPU
from a fixed-function graphics pipeline into a massively parallel processor
that can accelerate any compute-intensive workload.

The central idea: a modern GPU contains thousands of simple, in-order cores
grouped into streaming multiprocessors (SMs). Instead of one powerful core
doing one thing quickly, you have thousands of modest cores doing thousands
of things simultaneously. For problems with massive parallelism — matrix math,
image processing, deep learning — this architecture wins by orders of magnitude.

This module covers the full CUDA programming model: GPU hardware architecture,
the thread/warp/block/grid hierarchy, the memory hierarchy from registers to
VRAM, warp execution and divergence, occupancy and latency hiding, and the
two most important optimization techniques — memory coalescing and shared
memory tiling. Every concept is backed by a runnable simulation that produces
quantitative output.

"""

import textwrap
import re

TOPIC_NAME   = "CUDA — GPU Architecture & Parallel Computing"
DISPLAY_NAME = "19 · CUDA"
ICON         = "⚡"
SUBTITLE     = "Threads, Warps, and the Memory Hierarchy — Mastering GPU Parallelism"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY GPUs EXIST: THE CPU vs GPU PHILOSOPHY

### The Fundamental Trade-Off

    A CPU is designed for LATENCY: minimize the time to complete ONE task.
    A GPU is designed for THROUGHPUT: maximize the number of tasks per second.

    These goals lead to completely different silicon allocation strategies:

    CPU (e.g., Intel Core i9-13900K):
        24 cores, each with:
            ● Out-of-order execution engine (4+ GHz, 200+ instruction window)
            ● Branch predictor (99%+ accuracy, prevents pipeline stalls)
            ● 3–5 levels of cache (L1: 32KB, L2: 2MB, L3: 36MB shared)
            ● Hardware prefetcher (anticipates memory access patterns)
        Die area budget: ~70% spent on caches, branch prediction, OOO logic.
        Goal: 1 thread runs as fast as physically possible.

    GPU (e.g., NVIDIA A100):
        6912 CUDA cores across 108 SMs, each SM with:
            ● Simple in-order pipelines (no OOO, minimal branch prediction)
            ● 4 warp schedulers (each issues to 32 threads simultaneously)
            ● 192 KB shared memory / L1 cache per SM
        Die area budget: ~80% spent on compute units (ALUs, FMAs, tensor cores).
        Goal: 6912 threads run simultaneously, even if each is individually slow.

    Analogy:
        CPU = 24 Formula 1 cars (each blazingly fast, expensive, singular)
        GPU = 6912 motorcycles (each modest, cheap, parallel)

    For a 1-mile race: the F1 car wins.
    For delivering 6912 pizzas: the motorcycles win by an enormous margin.

### When GPUs Win and Why

    Deep learning training involves:
        ● Matrix multiply: C[i,j] = Σ_k A[i,k] × B[k,j]  for millions of (i,j)
        ● Each output element is INDEPENDENT of all others
        ● All elements can be computed simultaneously with no synchronization

    This is exactly what GPUs are built for:
        ● No data dependencies between output elements
        ● All threads execute the same operation (SIMT: Single Instruction, Multiple Threads)
        ● Memory access is predictable (structured, coalesceable)

    GPT-3 (175B parameters) requires ~350 TFLOPS of compute per forward pass.
    NVIDIA A100: 312 TFLOPS (FP16 tensor core) → ~1.1ms per forward pass.
    A single Intel Xeon: ~1 TFLOP → ~350ms per forward pass.
    GPU advantage: 300-500× on well-optimized matrix workloads.


##### PART 2 — GPU HARDWARE ARCHITECTURE

### The Streaming Multiprocessor (SM)

    The SM is the fundamental execution unit of an NVIDIA GPU.
    Think of it as a self-contained mini-CPU with many execution lanes.

    A100 SM breakdown (Ampere architecture):
    ┌─────────────────────────────────────────────────────────────────┐
    │  STREAMING MULTIPROCESSOR (SM)                                  │
    │                                                                 │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Instruction Cache   │  Warp Scheduler × 4               │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
    │  │  FP32 × 64   │  │  INT32 × 64  │  │  FP64 × 32   │           │
    │  │  (CUDA cores)│  │  (CUDA cores)│  │  (DP cores)  │           │
    │  └──────────────┘  └──────────────┘  └──────────────┘           │
    │                                                                 │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Tensor Cores × 4  (matrix operations: FP16/BF16/TF32)   │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Register File: 65536 × 32-bit registers (256 KB total)  │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  L1 Cache / Shared Memory: 192 KB (configurable split)   │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  Warp Slots: up to 64 warps (2048 threads) resident      │   │
    │  └──────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

    Key numbers for A100 (Ampere, 2020):
        SMs:                   108
        CUDA cores (FP32):     6,912  (64 per SM × 108 SMs)
        Tensor cores:          432    (4 per SM × 108 SMs)
        Max threads (total):   221,184 (2048 per SM × 108 SMs)
        Register file:         27.6 MB (256 KB × 108 SMs)
        Shared memory:         20.7 MB (192 KB × 108 SMs)
        L2 cache:              40 MB   (unified)
        HBM2e bandwidth:       2 TB/s
        FP16 Tensor TFLOPS:    312 TFLOPS (with sparsity: 624 TFLOPS)

### The Tensor Core: NVIDIA's Deep Learning Accelerator

    Standard CUDA cores: one multiply-add per clock (FMA: A × B + C)
    Tensor Core:         a 4×4 matrix multiply-accumulate per clock

    Tensor Core operation (D = A × B + C):
        A: 4×4 FP16 matrix
        B: 4×4 FP16 matrix
        C: 4×4 FP32 accumulator
        D: 4×4 FP32 result
        = 64 FMAs per Tensor Core per clock (vs 1 for CUDA core)

    In CUDA code, Tensor Cores are accessed through:
        ● WMMA (Warp Matrix Multiply Accumulate) API
        ● cuBLAS / cuDNN (uses Tensor Cores automatically)
        ● TF32 mode: PyTorch uses Tensor Cores by default when dtype=float32

    Why this matters for transformers:
        All attention, FFN, and projection layers are GEMMs.
        Tensor Cores provide 64× speedup vs FP32 CUDA cores.
        This is why a A100 achieves 312 TFLOPS vs the 19.5 TFLOPS FP32 rate.

### GPU Architecture Generations (NVIDIA)

    Volta  (V100, 2017):  First Tensor Cores. 640 TC × FP16/FP32.
    Turing (T4,  2018):   INT8/INT4 Tensor Cores. RT cores. 576 TC.
    Ampere (A100, 2020):  3rd gen TC. BF16, TF32, sparse. 432 TC × 4 per SM.
    Hopper (H100, 2022):  4th gen TC. FP8, Transformer Engine. NVLink4. 528 TC.
    Blackwell (B100, 2024): 5th gen TC. FP4/FP6. Second-gen Transformer Engine.

    Each generation roughly doubles relevant AI performance, primarily via:
        1. More Tensor Cores per SM
        2. Lower precision formats (FP16 → BF16 → TF32 → FP8 → FP4)
        3. Higher memory bandwidth (HBM2 → HBM2e → HBM3 → HBM3e)
        4. Better interconnects (NVLink 2 → 3 → 4 → 5)


##### PART 3 — THE CUDA PROGRAMMING MODEL: THREAD HIERARCHY

### The Three Levels of Hierarchy

    CUDA exposes three nested levels of parallelism to the programmer:

        GRID     → entire kernel launch (all work)
          └── BLOCK    → group of cooperating threads (share memory)
                └── THREAD  → one execution unit (one "worker")

    Example: Process a 1024×1024 image, one pixel per thread.

        Grid:    (64, 64) blocks  → 4,096 blocks total
        Block:   (16, 16) threads → 256 threads per block
        Threads: 4,096 × 256 = 1,048,576 threads total (= 1024 × 1024)

    CUDA kernel code (C++):
        __global__ void process_pixel(float* img, int W, int H) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;   // pixel x
            int y = blockIdx.y * blockDim.y + threadIdx.y;   // pixel y
            if (x < W && y < H) {
                int idx = y * W + x;
                img[idx] = img[idx] * 2.0f;   // each thread processes one pixel
            }
        }

        // Launch:
        dim3 block(16, 16);              // 256 threads per block
        dim3 grid(W/16, H/16);           // enough blocks to cover image
        process_pixel<<<grid, block>>>(img_ptr, W, H);

### Built-in Variables Every CUDA Thread Has

    threadIdx.{x,y,z}:   thread's position WITHIN its block (0-based)
    blockIdx.{x,y,z}:    block's position WITHIN the grid (0-based)
    blockDim.{x,y,z}:    size of each block (constant for all threads)
    gridDim.{x,y,z}:     size of the grid in blocks (constant)

    1D global thread index (most common pattern):
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

    2D global thread index:
        int gx = blockIdx.x * blockDim.x + threadIdx.x;
        int gy = blockIdx.y * blockDim.y + threadIdx.y;

### The Warp: The REAL Unit of Execution

    Although the programmer thinks in threads, the GPU executes in WARPS.

    WARP: 32 consecutive threads from the same block, executed in lockstep.
        ● All 32 threads execute the SAME instruction simultaneously (SIMT)
        ● If 32 threads each need different data → 32 memory loads happen simultaneously
        ● If threads take different if/else branches → WARP DIVERGENCE (performance loss)

    Thread → Warp mapping (within a block):
        Warp 0:  threads  0–31
        Warp 1:  threads 32–63
        Warp 2:  threads 64–95
        ...
        Warp N:  threads 32N – 32N+31

    Block of 256 threads = 8 warps = 8 groups of 32 executing in lockstep.

    This is why block sizes should always be multiples of 32:
        blockDim.x = 256 → 8 warps (no waste)
        blockDim.x = 100 → 3 full warps + 1 partial warp (8 threads idle)
        blockDim.x = 32  → 1 warp (fine, but limits occupancy options)

### Blocks and SMs: The Scheduling Relationship

    Blocks are assigned to SMs by the hardware scheduler:
        ● One block runs on exactly ONE SM (never split)
        ● One SM can run MULTIPLE blocks simultaneously (if resources allow)
        ● The programmer cannot control which SM receives which block

    SM resource limits (A100):
        Max threads per SM:       2048
        Max blocks per SM:        32
        Max shared memory per SM: 164 KB (usable for a single kernel)
        Max registers per SM:     65536

    Example: Block of 256 threads with 32 registers/thread:
        Register demand: 256 × 32 = 8192 registers per block
        Max blocks from registers: 65536 / 8192 = 8 blocks
        Max blocks from threads:   2048 / 256 = 8 blocks
        Actual blocks per SM: min(8, 8, 32) = 8 blocks
        Threads per SM: 8 × 256 = 2048 (100% thread occupancy!)

    This scheduling logic is called OCCUPANCY — one of the most important
    GPU performance concepts.


##### PART 4 — THE MEMORY HIERARCHY

### Memory Levels (fastest to slowest, smallest to largest)

    ┌──────────────────────────────────────────────────────────────────────┐
    │  LEVEL         │  LATENCY   │  BANDWIDTH  │  SIZE      │  SCOPE      │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Registers     │  1 cycle   │  ~50 TB/s   │  256 KB/SM │  1 thread   │
    │  Shared Mem/L1 │  1-5 cy    │  ~20 TB/s   │  192 KB/SM │  1 block    │
    │  L2 Cache      │  ~200 cy   │  ~12 TB/s   │  40 MB     │  entire GPU │
    │  Global (HBM)  │  ~700 cy   │  2 TB/s     │  40-80 GB  │  entire GPU │
    │  CPU RAM (PCIe)│  ms        │  ~64 GB/s   │  TB        │  host       │
    └──────────────────────────────────────────────────────────────────────┘

    The gap between registers/shared memory and global memory is enormous:
        Register bandwidth:     ~50 TB/s
        Global memory (HBM):    ~2 TB/s
        Ratio:                  25× faster at register level

    The #1 rule of GPU optimization:
        MINIMIZE GLOBAL MEMORY ACCESS. MAXIMIZE REGISTER/SHARED MEMORY REUSE.

### Registers: Thread-Private, Zero-Overhead

    Each thread has its own private register file.
    Local variables in a CUDA kernel live in registers.

        __global__ void example(float* A, float* B, float* C) {
            float a = A[threadIdx.x];   // loaded from global → register
            float b = B[threadIdx.x];   // loaded from global → register
            float c = a * b + a;        // pure register operation: ~1 cycle
            C[threadIdx.x] = c;         // stored from register → global
        }

    Register pressure: if a kernel uses too many registers per thread,
    the hardware spills to "local memory" (actually a region of global memory).
    Register spills: catastrophic for performance (~700 cycle latency).

    Control with __launch_bounds__ or compiler flag --maxrregcount.

### Shared Memory: The Programmable L1 Cache

    Shared memory is declared with the __shared__ keyword.
    All threads in a BLOCK share the same allocation.
    Lifetime: from block start to block end.
    Latency: ~5 cycles (vs ~700 for global memory).

    __global__ void use_shared(float* A, float* C) {
        __shared__ float smem[256];         // 256 × 4 bytes = 1 KB shared

        int tid = threadIdx.x;
        smem[tid] = A[blockIdx.x * 256 + tid];   // load from global to shared
        __syncthreads();                          // barrier: wait for all loads

        // Now use smem with ~5-cycle latency instead of 700-cycle global
        C[blockIdx.x * 256 + tid] = smem[tid] + smem[(tid+1) % 256];
    }

    __syncthreads(): a barrier synchronization instruction.
        ALL threads in the block must reach this point before any continue.
        Essential when shared memory written by one thread is read by another.
        Cost: ~few cycles for same-warp sync, potentially ~dozens for cross-warp.

### Shared Memory Bank Conflicts

    Shared memory is divided into 32 BANKS (one per thread in a warp).
    Optimal access: each thread in a warp accesses a DIFFERENT bank → 1 cycle.
    Bank conflict: two threads in a warp access the SAME bank → serialized.

    Bank assignment: element_index % 32 determines the bank number.

    Access pattern → performance:
        Thread i reads smem[i]:           no conflict  (1 cycle)
        Thread i reads smem[i*2]:         2-way conflict on half the banks
        Thread i reads smem[i*32]:        32-way conflict (worst case: 32 cycles)
        All threads read smem[0]:         broadcast (1 cycle — special case)

    Fix: add a padding element to shift stride patterns:
        __shared__ float smem[32][32 + 1];  // +1 column breaks stride-32 conflict

### Global Memory and Coalescing

    Global memory transactions occur in 128-byte cache lines.
    When a warp accesses global memory, the hardware combines adjacent accesses
    into the FEWEST possible cache line transactions.

    COALESCED access (ideal): 32 threads, each accesses consecutive floats.
        Thread 0: addr 0, Thread 1: addr 4, ..., Thread 31: addr 124
        → ONE 128-byte transaction for all 32 threads.  EFFICIENT.

    STRIDED access: 32 threads, each accesses every Nth float.
        Stride 2: Thread 0 → addr 0, Thread 1 → addr 8, ..., Thread 31 → addr 248
        → TWO 128-byte transactions (elements span 2 cache lines). 2× slower.

    RANDOM access: each thread accesses a completely different cache line.
        → 32 separate 128-byte transactions.  32× slower.

    This is the MOST COMMON performance bottleneck in GPU kernels.
    A naive matrix transpose is a classic example:
        Row read: coalesced → fast.
        Column write: strided with stride = matrix width → catastrophic.
    Fix: use shared memory as a coalescing staging area.


##### PART 5 — WARP EXECUTION, OCCUPANCY, AND LATENCY HIDING

### Warp Scheduling: Hiding Latency with Many Warps

    A GPU SM has 4 warp schedulers, each capable of issuing one instruction
    per clock to a different warp.

    When a warp stalls (e.g., waiting 700 cycles for global memory):
        → The warp scheduler INSTANTLY switches to a READY warp.
        → No context switch cost: all warps' registers are always live.
        → During the 700-cycle wait, OTHER warps make progress.

    This technique is called LATENCY HIDING.
    It's the primary reason GPUs can tolerate high memory latency.

    Analogy: a restaurant kitchen.
        One cook (CPU): stirs the soup, waits 10 minutes, then chops vegetables.
        Many cooks (GPU): 32 cooks, each stirring a different soup. While
        cook 1 waits for their soup to boil, cooks 2-32 are working on
        their tasks. The kitchen's throughput is 32× higher.

    Requirements for effective latency hiding:
        ● Enough warps per SM (high occupancy)
        ● Memory-independent instructions between memory ops (ILP)
        ● Low register pressure (more warps can fit)

### Warp Divergence: When SIMT Breaks Down

    SIMT: Single Instruction, Multiple Threads.
    All 32 threads in a warp execute the SAME instruction.

    What happens when threads take different branches?

    __global__ void divergent_kernel(float* data, float* out) {
        int tid = threadIdx.x;
        if (tid % 2 == 0) {            // even threads
            out[tid] = data[tid] * 2;
        } else {                       // odd threads
            out[tid] = data[tid] + 1;
        }
    }

    Execution on the hardware:
        Pass 1: Execute IF branch with even threads active, odd threads MASKED.
        Pass 2: Execute ELSE branch with odd threads active, even threads MASKED.
        Total time: 2× longer than a uniform (non-divergent) kernel.

    Rules for avoiding divergence:
        1. Ensure all threads in a warp take the same branch.
           → Predicate on warp-level properties (warpId, not threadId).
        2. Use warp shuffle intrinsics for warp-level communication.
        3. Sort input data so similar-behavior elements are in the same warp.

    Divergence analysis:
        if (threadIdx.x < 16):     divergent — threads 0-15 and 16-31 split
        if (threadIdx.x / 32 < 1): NOT divergent — whole warp takes same path
        if (condition_from_data):  potentially divergent — depends on data

### Occupancy: Filling the SM

    Occupancy = (active warps per SM) / (max warps per SM)

    High occupancy ≠ guaranteed high performance, BUT:
    Low occupancy (<50%) usually indicates a performance problem.

    Three resource limits determine occupancy:
        1. REGISTERS PER THREAD: Each thread uses N registers.
           Max warps = floor(65536 / (N × 32))
        2. SHARED MEMORY PER BLOCK: Each block uses S bytes.
           Max blocks = floor(192*1024 / S)
           Max warps = max_blocks × (threads_per_block / 32)
        3. THREADS PER BLOCK: Fixed by programmer.
           Max warps = threads_per_block / 32 × max_blocks

    Practical example (A100, kernel uses 40 registers/thread, 16KB smem/block):
        Register limit:   floor(65536 / (40 × 32)) = 51 warps (max 64)
        Shared mem limit: floor(192*1024 / 16384) = 12 blocks
                          12 × (256/32) = 96 warps (not binding)
        Thread limit:     2048/256 = 8 blocks × 8 warps = 64 warps (max)
        Binding limit:    threads → 64 warps vs registers → 51 warps
        Occupancy:        51 / 64 = 79.7%

    Use NVIDIA's Occupancy Calculator or nsight-compute to optimize.


##### PART 6 — THE ROOFLINE MODEL AND OPTIMIZATION STRATEGY

### The Roofline Model

    Every kernel is bounded by one of two limits:
        ● COMPUTE BOUND: the ALUs are the bottleneck. More FLOPs → slower.
        ● MEMORY BOUND:  DRAM bandwidth is the bottleneck. More bytes → slower.

    The roofline model formalizes this with one key metric:

        ARITHMETIC INTENSITY (AI) = FLOPs / bytes_of_global_memory_access

    Then:
        If AI > (peak_FLOPs / peak_bandwidth): COMPUTE BOUND
        If AI < (peak_FLOPs / peak_bandwidth): MEMORY BOUND

    For A100:
        Peak FP16 TFLOPS: 312e12 FLOPS
        Peak HBM bandwidth: 2e12 bytes/sec
        Ridge point: 312e12 / 2e12 = 156 FLOPs/byte

    Kernels with AI < 156 FLOP/byte are memory-bound on A100.

    Common AI values:
        Vector add C = A + B:            1/3 FLOP/byte (heavily memory-bound)
        Layer norm:                      ~2 FLOP/byte  (memory-bound)
        Dense matrix multiply (large):   ~50-200 FLOP/byte (near compute-bound)
        Convolution (small kernels):     ~5-20 FLOP/byte (memory-bound)
        Flash Attention:                 ~30-80 FLOP/byte (often memory-bound)

    Optimization implications:
        Memory-bound:  reduce global memory traffic (fuse ops, use shared mem)
        Compute-bound: improve instruction throughput (use Tensor Cores, FP8)

### Key Optimization Techniques

    1. MEMORY COALESCING
       Ensure 32 threads in a warp access consecutive memory addresses.
       Can improve bandwidth utilization from 1/32 to 1/1 (32× speedup).

    2. SHARED MEMORY TILING
       Load a tile from global memory into shared memory ONCE.
       Reuse across all threads in the block (amortizes global memory cost).
       Classic use: matrix multiplication (each element read O(N) → O(1) times).

    3. LOOP UNROLLING (#pragma unroll)
       Unroll inner loops to reduce branch overhead and increase ILP.
       The compiler can pipeline more instructions without branch stalls.

    4. WARP-LEVEL PRIMITIVES
       __shfl_down_sync, __shfl_xor_sync: communicate within a warp.
       No shared memory needed. ~4 cycle latency (vs ~5 for shared memory).
       Essential for reductions and prefix sums.

    5. PREFETCHING
       Load the NEXT tile while computing the CURRENT tile.
       Hides global memory latency behind computation.
       Implemented via double-buffered shared memory (ping-pong buffers).

    6. TENSOR CORE UTILIZATION
       Use FP16/BF16 data types. Use cuBLAS or WMMA API.
       Achieves 16× the throughput of FP32 CUDA cores for matrix ops.
       Requires M, N, K dimensions to be multiples of 16 (or 8 for BF16).

    7. KERNEL FUSION
       Instead of: kernel_A → global memory → kernel_B → global memory → kernel_C
       Fuse into: one kernel reads once, computes A+B+C, writes once.
       Saves (N-1) global memory round trips. Critical for memory-bound chains.

    8. ASYNCHRONOUS EXECUTION
       cudaMemcpyAsync + streams: overlap data transfer with kernel execution.
       Multiple streams run kernels concurrently on different SMs.
       cudaGraph: capture + replay kernel sequences with near-zero launch overhead.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Thread Hierarchy Simulator — Grid, Block, Warp, Thread Mapping": {
        "description": (
            "Simulate the full CUDA thread hierarchy: map threads to warps, "
            "blocks to SMs, and compute global thread IDs. "
            "Show how 1D, 2D, and 3D grid/block configurations are flattened. "
            "Demonstrate block-to-SM assignment and the occupancy calculation "
            "that determines how many blocks fit on each SM."
        ),
        "language": "python",
        "code": '''
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math

print("=" * 65)
print("  CUDA THREAD HIERARCHY SIMULATOR")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# GPU Hardware Spec
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class GPUSpec:
    """Hardware limits for one GPU model."""
    name:                str
    num_sms:             int
    max_threads_per_sm:  int   # hardware max resident threads per SM
    max_blocks_per_sm:   int   # hardware max resident blocks per SM
    max_warps_per_sm:    int   # = max_threads_per_sm // 32
    registers_per_sm:    int   # total 32-bit registers in the SM
    shared_mem_per_sm:   int   # bytes of shared mem / L1 per SM
    warp_size:           int = 32

A100 = GPUSpec(
    name="NVIDIA A100",
    num_sms=108,
    max_threads_per_sm=2048,
    max_blocks_per_sm=32,
    max_warps_per_sm=64,
    registers_per_sm=65536,
    shared_mem_per_sm=192 * 1024,   # 192 KB
)

H100 = GPUSpec(
    name="NVIDIA H100",
    num_sms=132,
    max_threads_per_sm=2048,
    max_blocks_per_sm=32,
    max_warps_per_sm=64,
    registers_per_sm=65536,
    shared_mem_per_sm=228 * 1024,   # 228 KB
)

# ─────────────────────────────────────────────────────────────────────────
# Kernel Configuration
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class KernelConfig:
    """Describes one kernel launch configuration."""
    name:                   str
    grid_dim:               Tuple[int, int, int]    # (gx, gy, gz)
    block_dim:              Tuple[int, int, int]    # (bx, by, bz)
    registers_per_thread:   int
    shared_mem_per_block:   int                     # bytes

    @property
    def threads_per_block(self) -> int:
        return self.block_dim[0] * self.block_dim[1] * self.block_dim[2]

    @property
    def warps_per_block(self) -> int:
        return math.ceil(self.threads_per_block / 32)

    @property
    def total_blocks(self) -> int:
        return self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]

    @property
    def total_threads(self) -> int:
        return self.total_blocks * self.threads_per_block


# ─────────────────────────────────────────────────────────────────────────
# Occupancy Calculator
# ─────────────────────────────────────────────────────────────────────────

def compute_occupancy(gpu: GPUSpec, kernel: KernelConfig) -> Dict:
    """
    Compute theoretical occupancy using the same logic as NVIDIA\'s
    Occupancy Calculator.

    Three resources limit how many blocks can co-reside on an SM:
      1. Max threads per SM
      2. Max blocks per SM
      3. Register file size
      4. Shared memory size
    The most restrictive (fewest blocks) wins.
    """
    tpb = kernel.threads_per_block    # threads per block
    wpb = kernel.warps_per_block      # warps per block

    # 1. Thread limit
    blocks_from_threads = gpu.max_threads_per_sm // tpb

    # 2. Block limit (hardware cap)
    blocks_from_block_limit = gpu.max_blocks_per_sm

    # 3. Register limit
    # Registers are allocated in granules of 256 per warp on Ampere/Hopper
    regs_granule = 256
    regs_per_warp = (
        math.ceil(kernel.registers_per_thread * 32 / regs_granule)
        * regs_granule
    )
    regs_per_block = regs_per_warp * wpb
    if regs_per_block > 0:
        blocks_from_registers = gpu.registers_per_sm // regs_per_block
    else:
        blocks_from_registers = gpu.max_blocks_per_sm

    # 4. Shared memory limit
    # Shared memory is allocated per block; minimum granule is 256 bytes
    smem_granule = 256
    smem_per_block = (
        math.ceil(kernel.shared_mem_per_block / smem_granule)
        * smem_granule
    ) if kernel.shared_mem_per_block > 0 else 256  # always alloc at least 256 B
    blocks_from_smem = gpu.shared_mem_per_sm // smem_per_block

    # Actual blocks per SM = most restrictive limit
    active_blocks = min(
        blocks_from_threads,
        blocks_from_block_limit,
        blocks_from_registers,
        blocks_from_smem,
    )
    active_blocks = max(0, active_blocks)   # can\'t be negative

    active_warps   = active_blocks * wpb
    active_threads = active_blocks * tpb
    occupancy      = active_warps / gpu.max_warps_per_sm if gpu.max_warps_per_sm > 0 else 0

    # Determine the binding (most restrictive) resource
    limits = {
        "threads":    blocks_from_threads,
        "block_cap":  blocks_from_block_limit,
        "registers":  blocks_from_registers,
        "smem":       blocks_from_smem,
    }
    binding_resource = min(limits, key=limits.get)

    return {
        "active_blocks_per_sm":  active_blocks,
        "active_warps_per_sm":   active_warps,
        "active_threads_per_sm": active_threads,
        "occupancy_pct":         occupancy * 100,
        "binding_resource":      binding_resource,
        "limits":                limits,
        "threads_per_block":     tpb,
        "warps_per_block":       wpb,
        "total_blocks":          kernel.total_blocks,
        "total_threads":         kernel.total_threads,
    }


# ─────────────────────────────────────────────────────────────────────────
# Thread ID computation utilities
# ─────────────────────────────────────────────────────────────────────────

def global_thread_id_1d(block_idx_x: int, thread_idx_x: int,
                         block_dim_x: int) -> int:
    """CUDA: blockIdx.x * blockDim.x + threadIdx.x"""
    return block_idx_x * block_dim_x + thread_idx_x

def global_thread_id_2d(block_idx: Tuple[int,int],
                         thread_idx: Tuple[int,int],
                         block_dim: Tuple[int,int],
                         grid_dim:  Tuple[int,int]) -> Tuple[int,int,int]:
    """Returns (gx, gy, linear_id) for a 2D grid/block configuration."""
    gx = block_idx[0] * block_dim[0] + thread_idx[0]
    gy = block_idx[1] * block_dim[1] + thread_idx[1]
    linear = gy * (grid_dim[0] * block_dim[0]) + gx
    return gx, gy, linear

def warp_id(thread_idx_x: int) -> int:
    """Thread\'s warp within its block: threadIdx.x / 32"""
    return thread_idx_x // 32

def lane_id(thread_idx_x: int) -> int:
    """Thread\'s lane within its warp: threadIdx.x % 32"""
    return thread_idx_x % 32


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Thread ID mapping in 1D and 2D grids
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Thread ID mapping: 1D and 2D grid configurations")
print("━" * 65)
print()

print("  1D Grid: gridDim=(4,), blockDim=(8,)")
print("  Each box = [blockIdx.x | threadIdx.x | globalId | warpId:laneId]")
print()
gx, bx = 4, 8
for b in range(gx):
    row = []
    for t in range(bx):
        gid = global_thread_id_1d(b, t, bx)
        w   = warp_id(t)
        l   = lane_id(t)
        row.append(f"[B{b}|T{t}|G{gid:02d}|W{w}:L{l}]")
    print("  " + " ".join(row))
print()
print("  Warps span block boundaries? NO — each block\'s threads form their own warps.")
print()

print()
print("  2D Grid: gridDim=(2,2), blockDim=(4,4) — 16 threads per block")
print("  Sample: block (1,0) — showing (gx, gy, linear_global_id)")
print()
G = (2,2)
B = (4,4)
print(f"  {'threadIdx.x →':>15}", end="")
for tx in range(B[0]):
    print(f"  tx={tx}     ", end="")
print()
for ty in range(B[1]):
    print(f"  threadIdx.y={ty}:  ", end="")
    for tx in range(B[0]):
        gx_val, gy_val, lin = global_thread_id_2d((1,0), (tx,ty), B, G)
        print(f"  ({gx_val},{gy_val})/{lin:02d} ", end="")
    print()
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Warp structure inside a block
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Warp structure: how threads map to warps")
print("━" * 65)
print()

BLOCK_SIZE = 128
print(f"  Block of {BLOCK_SIZE} threads → {BLOCK_SIZE // 32} warps:")
print()
print(f"  {'Warp':>6} | {'Thread range':>14} | {'Lane IDs'}")
print(f"  {'─'*50}")
for w in range(BLOCK_SIZE // 32):
    t_start = w * 32
    t_end   = t_start + 31
    print(f"  {w:6d} | {t_start:6d} – {t_end:6d}  | lanes 0–31 "
          f"{'(active)' if w < 3 else '(all must execute same instr)'}")
print()
print("  The warp is the HARDWARE scheduling unit.")
print("  All 32 threads in a warp execute the same instruction each cycle.")
print("  If block_size is not a multiple of 32, the last warp has idle lanes:")

for bs in [32, 64, 96, 128, 192, 256, 100, 48]:
    warps      = math.ceil(bs / 32)
    full_warps = bs // 32
    partial    = bs % 32
    idle_lanes = (32 - partial) % 32
    waste_pct  = idle_lanes / (warps * 32) * 100
    flag = " ⚠️  wasted lanes" if idle_lanes > 0 else " ✅"
    print(f"  blockDim={bs:3d}: {warps} warp(s), {idle_lanes:2d} idle lanes "
          f"({waste_pct:4.1f}% waste){flag}")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Occupancy analysis for real kernel configurations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Occupancy analysis (A100 GPU)")
print("━" * 65)
print()

KERNELS = [
    KernelConfig("VecAdd (simple)",
                 grid_dim=(1024,1,1), block_dim=(256,1,1),
                 registers_per_thread=8,  shared_mem_per_block=0),
    KernelConfig("MatMul tiled (moderate reg)",
                 grid_dim=(64,64,1),  block_dim=(16,16,1),
                 registers_per_thread=32, shared_mem_per_block=8*1024),
    KernelConfig("Attention kernel (high reg)",
                 grid_dim=(128,1,1),  block_dim=(128,1,1),
                 registers_per_thread=64, shared_mem_per_block=32*1024),
    KernelConfig("Reduction (high smem)",
                 grid_dim=(512,1,1),  block_dim=(512,1,1),
                 registers_per_thread=16, shared_mem_per_block=64*1024),
    KernelConfig("Small block (suboptimal)",
                 grid_dim=(4096,1,1), block_dim=(32,1,1),
                 registers_per_thread=20, shared_mem_per_block=0),
    KernelConfig("Oversubscribed registers",
                 grid_dim=(256,1,1),  block_dim=(256,1,1),
                 registers_per_thread=128, shared_mem_per_block=4*1024),
]

gpu = A100
print(f"  GPU: {gpu.name} — {gpu.num_sms} SMs, "
      f"{gpu.max_warps_per_sm} max warps/SM, "
      f"{gpu.max_threads_per_sm} max threads/SM")
print()

print(f"  {'Kernel':<35} | {'Threads/Blk':>11} | {'Blks/SM':>7} | "
      f"{'Warps/SM':>8} | {'Occupancy':>10} | {'Binding':>12}")
print(f"  {'─'*95}")

for kern in KERNELS:
    occ = compute_occupancy(gpu, kern)
    occ_bar_filled = int(occ["occupancy_pct"] / 100 * 10)
    occ_bar = "█" * occ_filled + "░" * (10 - occ_bar_filled) if (occ_filled := int(occ["occupancy_pct"] / 100 * 10)) >= 0 else ""
    print(f"  {kern.name:<35} | {occ['threads_per_block']:11d} | "
          f"{occ['active_blocks_per_sm']:7d} | "
          f"{occ['active_warps_per_sm']:8d} | "
          f"{occ['occupancy_pct']:9.1f}% | "
          f"{occ['binding_resource']:>12}")

print()
print("  Occupancy interpretation:")
print("    >75%: Good — enough warps to hide most memory latency")
print("    50-75%: Acceptable — likely some latency exposure")
print("    <50%: Poor — kernel likely register or shared memory bound")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Block-to-SM assignment simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Block scheduling: how blocks are assigned to SMs")
print("━" * 65)
print()

kern = KERNELS[0]   # VecAdd: simple, high occupancy
occ  = compute_occupancy(gpu, kern)
blocks_per_sm = occ["active_blocks_per_sm"]
total_blocks  = kern.total_blocks

print(f"  Kernel: {kern.name}")
print(f"  Grid: {kern.total_blocks} blocks, Block: {kern.threads_per_block} threads")
print(f"  Blocks per SM: {blocks_per_sm}")
print()

sm_loads  = [0] * gpu.num_sms
remaining = total_blocks

# Round-robin block assignment (simplified — real HW uses a wave-based scheduler)
block_id = 0
wave = 0
while remaining > 0:
    wave += 1
    blocks_this_wave = min(remaining, blocks_per_sm * gpu.num_sms)
    for sm in range(gpu.num_sms):
        take = min(blocks_per_sm, remaining)
        if take <= 0:
            break
        sm_loads[sm] += take
        remaining -= take
    if remaining <= 0:
        break

total_waves = math.ceil(total_blocks / (blocks_per_sm * gpu.num_sms))
print(f"  Total blocks:        {total_blocks:,}")
print(f"  SMs × blocks/SM:     {gpu.num_sms} × {blocks_per_sm} = "
      f"{gpu.num_sms * blocks_per_sm:,} blocks per wave")
print(f"  Waves to complete:   {total_waves}")
print()
blocks_last_wave = total_blocks % (blocks_per_sm * gpu.num_sms)
if blocks_last_wave == 0:
    blocks_last_wave = blocks_per_sm * gpu.num_sms
sms_busy_last   = math.ceil(blocks_last_wave / blocks_per_sm)
print(f"  Last wave: {blocks_last_wave} blocks → "
      f"{sms_busy_last}/{gpu.num_sms} SMs active "
      f"({'⚠️  SM tail effect' if sms_busy_last < gpu.num_sms else '✅ all SMs busy'})")
print()
print("  KEY INSIGHT — The tail effect:")
print("    If total_blocks is not a multiple of (SMs × blocks/SM),")
print("    the final wave leaves some SMs idle.")
print("    Solution: choose grid_dim to be a multiple of the SM count.")
print(f"    For A100: use multiples of {gpu.num_sms} blocks (or {gpu.num_sms * blocks_per_sm}).")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Memory Hierarchy — Bandwidth, Coalescing & Bank Conflicts": {
        "description": (
            "Simulate and quantify every level of the CUDA memory hierarchy. "
            "Model coalesced vs strided vs random global memory access, showing "
            "effective bandwidth degradation. Simulate shared memory bank conflicts "
            "and demonstrate the padding fix. Compute the arithmetic intensity for "
            "common operations and classify them on the roofline model."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import defaultdict
import math

print("=" * 65)
print("  CUDA MEMORY HIERARCHY — BANDWIDTH & ACCESS PATTERNS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# Memory level specifications
# ─────────────────────────────────────────────────────────────────────────

MEMORY_LEVELS = {
    "Registers":         {"latency_cycles": 1,   "bandwidth_TBs": 50,  "size_MB": 0.25},
    "Shared Mem / L1":   {"latency_cycles": 5,   "bandwidth_TBs": 20,  "size_MB": 0.19},
    "L2 Cache":          {"latency_cycles": 200, "bandwidth_TBs": 12,  "size_MB": 40},
    "Global Mem (HBM2e)":{"latency_cycles": 700, "bandwidth_TBs": 2,   "size_MB": 80*1024},
    "CPU DRAM (PCIe4)":  {"latency_cycles": 100000,"bandwidth_TBs":0.064,"size_MB":None},
}

GPU_CLOCK_GHZ = 1.41   # A100 boost clock

print("━" * 65)
print("  SECTION 1 — Memory hierarchy specifications (A100)")
print("━" * 65)
print()
print(f"  {'Memory Level':<25} | {'Latency (cyc)':>14} | {'Latency (ns)':>13} | "
      f"{'Bandwidth (TB/s)':>17} | {'Size':>12}")
print(f"  {'─'*88}")

for name, spec in MEMORY_LEVELS.items():
    lat_ns = spec["latency_cycles"] / (GPU_CLOCK_GHZ * 1e9) * 1e9
    size_s = f"{spec['size_MB']} MB" if spec["size_MB"] else "TBs"
    if spec["size_MB"] and spec["size_MB"] > 1024:
        size_s = f"{spec['size_MB']//1024} GB"
    print(f"  {name:<25} | {spec['latency_cycles']:14,} | {lat_ns:12.1f}  | "
          f"{spec['bandwidth_TBs']:17.3f} | {size_s:>12}")

print()
print("  Register ↔ Global memory bandwidth ratio: ~25×")
print("  Register ↔ Global memory latency ratio:  ~700×")
print()


# ─────────────────────────────────────────────────────────────────────────
# Coalescing model
# ─────────────────────────────────────────────────────────────────────────

print("━" * 65)
print("  SECTION 2 — Memory coalescing: the #1 global memory optimization")
print("━" * 65)
print()

CACHE_LINE_BYTES = 128   # L1 cache line = 128 bytes
FLOAT_BYTES      = 4
WARP_SIZE        = 32

def coalescing_analysis(stride_elements: int, dtype_bytes: int = 4) -> dict:
    """
    Model how many 128-byte cache line transactions a warp needs
    for a strided access pattern.

    Each thread accesses: base_addr + thread_id * stride * dtype_bytes

    stride_elements=1: perfectly coalesced (consecutive floats)
    stride_elements=2: every other float → spans more cache lines
    stride_elements=N: one access per cache line in the worst case
    """
    warp_size = 32
    transactions = set()  # unique cache lines touched

    for thread_id in range(warp_size):
        byte_addr = thread_id * stride_elements * dtype_bytes
        cache_line = byte_addr // CACHE_LINE_BYTES
        transactions.add(cache_line)

    n_transactions = len(transactions)
    efficiency     = 1.0 / n_transactions   # relative to fully coalesced
    useful_bytes   = warp_size * dtype_bytes
    transferred_bytes = n_transactions * CACHE_LINE_BYTES
    waste_pct      = (1 - useful_bytes / transferred_bytes) * 100

    return {
        "transactions":      n_transactions,
        "efficiency":        efficiency,
        "useful_bytes":      useful_bytes,
        "transferred_bytes": transferred_bytes,
        "waste_pct":         waste_pct,
    }

print("  One warp (32 threads), each loading one float (4 bytes)")
print("  Accessing: base + thread_id × stride × sizeof(float)")
print()
print(f"  {'Stride':>8} | {'Cache-line Transactions':>24} | "
      f"{'Bandwidth Efficiency':>22} | {'Bytes Wasted':>13} | {'Pattern'}")
print(f"  {'─'*85}")

STRIDES = [1, 2, 4, 8, 16, 32, 64]
STRIDE_LABELS = {
    1: "Coalesced (ideal) ✅",
    2: "Stride-2",
    4: "Stride-4",
    8: "Stride-8",
    16: "Stride-16",
    32: "One cache line per thread",
    64: "Scattered (worst case)",
}

for s in STRIDES:
    r = coalescing_analysis(s)
    label = STRIDE_LABELS.get(s, f"Stride-{s}")
    eff_bar = "█" * int(r["efficiency"] * 20) + "░" * (20 - int(r["efficiency"] * 20))
    print(f"  {s:8d} | {r['transactions']:24d} | {eff_bar} {r['efficiency']*100:4.1f}% | "
          f"{r['waste_pct']:12.1f}% | {label}")

print()
print("  Performance implication (A100 at 2 TB/s peak HBM bandwidth):")
print()
data_size_GB = 1.0   # 1 GB of data
for s in [1, 4, 32]:
    r = coalescing_analysis(s)
    effective_bw = 2e12 * r["efficiency"]   # TB/s
    time_ms = (data_size_GB * 1e9) / effective_bw * 1e3
    print(f"  Stride={s:2d}: effective BW = {effective_bw/1e12:.2f} TB/s, "
          f"time for 1 GB = {time_ms:.2f} ms")
print()
print("  Stride=1 vs Stride=32: 32× throughput difference on the same GPU.")
print("  The kernel appears to 'use' the GPU but most memory bandwidth is wasted.")
print()


# ─────────────────────────────────────────────────────────────────────────
# Shared memory bank conflict simulation
# ─────────────────────────────────────────────────────────────────────────

print("━" * 65)
print("  SECTION 3 — Shared memory bank conflicts")
print("━" * 65)
print()
print("  Shared memory is divided into 32 banks.")
print("  Bank(addr) = (addr / 4) % 32   (4 bytes per bank word, 32 banks)")
print("  One warp issues shared memory accesses in ONE cycle IF no conflicts.")
print("  N-way conflict: N threads hit the same bank → serialized into N cycles.")
print()

NUM_BANKS = 32

def shared_mem_bank_analysis(access_pattern: list, label: str) -> dict:
    """
    Analyse bank conflicts for a given 32-thread access pattern.
    access_pattern[i] = shared memory word index accessed by thread i.
    Returns number of serialized transactions required.
    """
    bank_access_counts = defaultdict(list)
    for thread_id, word_idx in enumerate(access_pattern):
        bank = word_idx % NUM_BANKS
        bank_access_counts[bank].append(thread_id)

    # Maximum accesses to any one bank = number of serialized transactions
    max_conflict = max(len(threads) for threads in bank_access_counts.values())

    # Special case: broadcast (all threads hit the same address) = 1 transaction
    all_same_addr = len(set(access_pattern)) == 1
    if all_same_addr:
        max_conflict = 1

    conflict_banks = {
        bank: threads for bank, threads in bank_access_counts.items()
        if len(threads) > 1
    }

    return {
        "label":            label,
        "max_conflict":     max_conflict,
        "conflict_banks":   len(conflict_banks),
        "is_broadcast":     all_same_addr,
        "is_conflict_free": max_conflict == 1,
        "bank_usage":       dict(bank_access_counts),
    }

# Various access patterns to test
patterns = [
    (list(range(32)),             "Sequential: smem[tid]"),
    ([i * 2 for i in range(32)],  "Stride-2:  smem[tid*2]"),
    ([i * 4 for i in range(32)],  "Stride-4:  smem[tid*4]"),
    ([i * 32 for i in range(32)], "Stride-32: smem[tid*32] (all bank 0!)"),
    ([0] * 32,                    "Broadcast: smem[0] (all threads same addr)"),
    ([i * 33 for i in range(32)], "Stride-33: smem[tid*33] (bank rotation)"),
    ([i + (i // 16) * 1 for i in range(32)],
                                  "Padded: smem[tid + tid/16] (approx +1 pad)"),
]

print(f"  {'Pattern':<42} | {'Conflict':>10} | {'Perf':>8} | {'Note'}")
print(f"  {'─'*82}")

for pat, label in patterns:
    r = shared_mem_bank_analysis(pat, label)
    conflict_str = f"{r['max_conflict']}-way" if r['max_conflict'] > 1 else "None"
    perf_ratio = f"{1/r['max_conflict']*100:.0f}%"
    note = ""
    if r["is_broadcast"]:      note = "broadcast ← special fast path"
    elif r["is_conflict_free"]: note = "ideal"
    elif r["max_conflict"] == 32: note = "worst case: 32 serial cycles"
    print(f"  {label:<42} | {conflict_str:>10} | {perf_ratio:>8} | {note}")

print()
print("  FIX FOR STRIDE-32 CONFLICT — Padding:")
print("    Instead of: __shared__ float smem[32][32];")
print("    Use:        __shared__ float smem[32][33];   // +1 padding column")
print()
print("  Why padding works:")
print("    Without pad: smem[row][col], bank = (row*32 + col) % 32 = col % 32")
print("    With pad:    smem[row][col], bank = (row*33 + col) % 32")
print("    Column-major access now hits different banks per row.")
print()

# Demonstrate the fix with a matrix transpose access
print("  Matrix transpose example (32×32 tile):")
# Without padding: writing columns creates stride-32 conflict
no_pad  = [(i * 32) % (32 * 32) for i in range(32)]   # column access, stride-32
with_pad= [(i * 33) % (32 * 33) for i in range(32)]   # column access with +1 pad

r_no  = shared_mem_bank_analysis(no_pad,  "Write column, no padding")
r_with= shared_mem_bank_analysis(with_pad,"Write column, +1 padding")
print(f"  No padding:  {r_no['max_conflict']}-way conflict "
      f"→ {1/r_no['max_conflict']*100:.0f}% bandwidth")
print(f"  +1 padding:  {r_with['max_conflict']}-way conflict "
      f"→ {1/r_with['max_conflict']*100:.0f}% bandwidth  ✅")
print()


# ─────────────────────────────────────────────────────────────────────────
# Roofline model
# ─────────────────────────────────────────────────────────────────────────

print("━" * 65)
print("  SECTION 4 — The Roofline Model: classify your kernel")
print("━" * 65)
print()

# A100 hardware peaks
PEAK_FLOPS_FP16_TC = 312e12    # 312 TFLOPS FP16 Tensor Core
PEAK_FLOPS_FP32    =  19.5e12  # 19.5 TFLOPS FP32 CUDA cores
PEAK_BW_HBM        = 2.0e12    # 2 TB/s HBM2e
RIDGE_POINT_FP16   = PEAK_FLOPS_FP16_TC / PEAK_BW_HBM   # FLOPs/byte
RIDGE_POINT_FP32   = PEAK_FLOPS_FP32    / PEAK_BW_HBM

print(f"  A100 hardware limits:")
print(f"    Peak FP16 (Tensor Core):  {PEAK_FLOPS_FP16_TC/1e12:.0f} TFLOPS")
print(f"    Peak FP32 (CUDA cores):   {PEAK_FLOPS_FP32/1e12:.1f} TFLOPS")
print(f"    Peak HBM bandwidth:       {PEAK_BW_HBM/1e12:.1f} TB/s")
print(f"    Ridge point (FP16 TC):    {RIDGE_POINT_FP16:.0f} FLOPs/byte")
print(f"    Ridge point (FP32):       {RIDGE_POINT_FP32:.1f} FLOPs/byte")
print()

def roofline(name: str, flops: float, bytes_accessed: float,
             dtype_peak: float = PEAK_FLOPS_FP32,
             ridge_point: float = RIDGE_POINT_FP32) -> dict:
    """Compute roofline position for one kernel."""
    ai = flops / bytes_accessed if bytes_accessed > 0 else 0
    memory_bound_limit  = ai * PEAK_BW_HBM
    compute_bound_limit = dtype_peak
    achievable_flops    = min(memory_bound_limit, compute_bound_limit)
    is_compute_bound    = ai > ridge_point
    bottleneck = "Compute" if is_compute_bound else "Memory BW"
    utilization = achievable_flops / dtype_peak * 100

    return {
        "name":              name,
        "ai":                ai,
        "achievable_gflops": achievable_flops / 1e9,
        "peak_gflops":       dtype_peak / 1e9,
        "utilization_pct":   utilization,
        "bottleneck":        bottleneck,
        "is_compute_bound":  is_compute_bound,
    }

# Common GPU workloads
N = 1024  # dimension

kernels_roofline = [
    roofline("Vector add C=A+B (FP32)",
             flops=N*N,                    bytes_accessed=3*N*N*4),
    roofline("Layer Norm (FP32)",
             flops=5*N*N,                  bytes_accessed=2*N*N*4),
    roofline("Elementwise GELU (FP32)",
             flops=8*N*N,                  bytes_accessed=2*N*N*4),
    roofline("GEMM 1024×1024 (FP32 CUDA)",
             flops=2*N**3,                 bytes_accessed=3*N*N*4),
    roofline("GEMM 1024×1024 (FP16 TC)",
             flops=2*N**3,                 bytes_accessed=3*N*N*2,
             dtype_peak=PEAK_FLOPS_FP16_TC, ridge_point=RIDGE_POINT_FP16),
    roofline("Flash Attention (FP16, seq=512)",
             flops=4*512*512*64,           bytes_accessed=3*512*64*2,
             dtype_peak=PEAK_FLOPS_FP16_TC, ridge_point=RIDGE_POINT_FP16),
    roofline("Conv 3×3, 256ch, 64×64 image",
             flops=2*3*3*256*256*64*64,    bytes_accessed=(9*256*256 + 64*64*256*2)*4),
]

print(f"  {'Kernel':<40} | {'AI (F/B)':>10} | {'Bottleneck':>12} | {'Roofline util':>14}")
print(f"  {'─'*82}")
for k in kernels_roofline:
    bar_len = min(20, int(k["utilization_pct"] / 5))
    bar = "█" * bar_len + "░" * (20 - bar_len)
    bound_icon = "🖥️ " if k["is_compute_bound"] else "💾 "
    print(f"  {k['name']:<40} | {k['ai']:10.1f} | "
          f"{bound_icon}{k['bottleneck']:>9} | "
          f"{bar} {k['utilization_pct']:5.1f}%")

print()
print("  Interpretation:")
print("    AI < 10 F/byte → nearly always memory-bound on A100")
print("    Optimization: reduce bytes read/written (kernel fusion, compression)")
print("    AI > 156 F/byte (FP16 TC) → compute-bound")
print("    Optimization: use Tensor Cores, lower precision (FP8), increase batch size")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Warp Divergence — Simulation and Quantification": {
        "description": (
            "Simulate CUDA warp divergence in detail: show how the hardware serializes "
            "divergent branches by executing multiple passes with thread masking. "
            "Quantify the performance penalty for common divergence patterns. "
            "Demonstrate data-dependent divergence from real workloads (softmax "
            "with masking, sparse access, conditional computation). "
            "Show techniques to eliminate or reduce divergence."
        ),
        "language": "python",
        "code": '''
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import math

print("=" * 65)
print("  WARP DIVERGENCE — SIMULATION AND QUANTIFICATION")
print("=" * 65)
print()

WARP_SIZE = 32

# ─────────────────────────────────────────────────────────────────────────
# Warp divergence simulator
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class WarpExecution:
    """
    Models one warp executing through a divergent if-else construct.

    In CUDA\'s SIMT model, divergence is handled by executing each unique
    execution path separately, masking (disabling) threads that don\'t
    take that path. The total work = sum of instructions on ALL taken paths.
    """
    warp_id:         int
    thread_data:     np.ndarray    # value seen by each of 32 threads
    cycles_uniform:  int = 10      # cycles for one path taken uniformly

    def simulate_branch(self, condition: Callable[[np.ndarray], np.ndarray],
                         if_cycles: int, else_cycles: int,
                         label: str = "") -> dict:
        """
        Simulate: if (condition) { /* if_cycles work */ } else { /* else_cycles */ }

        Returns timing analysis.
        """
        mask_if   = condition(self.thread_data)   # True for threads taking IF
        mask_else = ~mask_if

        n_if   = mask_if.sum()
        n_else = mask_else.sum()
        n_total = len(self.thread_data)

        # Without divergence: all threads take the same path
        # (hypothetical uniform execution)
        uniform_cycles = max(if_cycles, else_cycles)

        # With divergence: execute IF with else-threads masked,
        # then ELSE with if-threads masked (two separate passes)
        if n_if == 0:       # everyone takes else
            divergent_cycles = else_cycles
            passes = 1
        elif n_else == 0:   # everyone takes if
            divergent_cycles = if_cycles
            passes = 1
        else:               # actual divergence — two passes
            divergent_cycles = if_cycles + else_cycles
            passes = 2

        # Efficiency: what fraction of thread-cycles do useful work?
        total_thread_cycles   = divergent_cycles * n_total
        useful_thread_cycles  = (n_if * if_cycles) + (n_else * else_cycles)
        simd_efficiency       = useful_thread_cycles / total_thread_cycles

        slowdown_vs_uniform = divergent_cycles / uniform_cycles

        return {
            "label":                label,
            "n_if":                 int(n_if),
            "n_else":               int(n_else),
            "passes":               passes,
            "if_cycles":            if_cycles,
            "else_cycles":          else_cycles,
            "divergent_cycles":     divergent_cycles,
            "uniform_cycles":       uniform_cycles,
            "slowdown":             slowdown_vs_uniform,
            "simd_efficiency":      simd_efficiency * 100,
            "is_divergent":         passes > 1,
        }


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Basic divergence patterns
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Basic divergence patterns and penalties")
print("━" * 65)
print()

# Create a warp with varying data
thread_data = np.arange(WARP_SIZE, dtype=np.float32)
warp = WarpExecution(warp_id=0, thread_data=thread_data)

# Define different branch conditions
BRANCH_CONFIGS = [
    # (label, condition_fn, if_cycles, else_cycles)
    ("No divergence (all if)",
     lambda d: np.ones(len(d), dtype=bool),  10, 5),
    ("No divergence (all else)",
     lambda d: np.zeros(len(d), dtype=bool), 10, 5),
    ("1 thread diverges (thread 0)",
     lambda d: d != 0,                        10, 5),
    ("50/50 split (even/odd threads)",
     lambda d: (d % 2 == 0),                  10, 5),
    ("25/75 split",
     lambda d: d < 8,                          10, 5),
    ("1/31 split (extreme)",
     lambda d: d < 1,                          10, 5),
    ("Heavy if-path (both branches same weight)",
     lambda d: d < 16,                         20, 20),
    ("Heavy else-path (50/50, else is 3× if)",
     lambda d: d < 16,                         5,  15),
]

print(f"  {'Branch Condition':<40} | {'N_if':>6} | {'N_else':>7} | "
      f"{'Passes':>7} | {'Slowdown':>10} | {'SIMD Eff':>10}")
print(f"  {'─'*90}")

for label, cond, if_cyc, else_cyc in BRANCH_CONFIGS:
    r = warp.simulate_branch(cond, if_cyc, else_cyc, label)
    div_flag = "🔴" if r["is_divergent"] else "✅"
    print(f"  {div_flag} {label:<38} | {r['n_if']:6d} | {r['n_else']:7d} | "
          f"{r['passes']:7d} | {r['slowdown']:9.2f}× | {r['simd_efficiency']:8.1f}%")

print()
print("  KEY: Any branch where not ALL 32 threads take the same path = divergence.")
print("  Even 1 divergent thread causes a second execution pass for the whole warp.")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Real workload divergence — masked softmax
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Masked softmax (transformer attention masking)")
print("━" * 65)
print()
print("  In self-attention, causal masking sets some logits to -∞:")
print("    if (col > row): logit = -inf  // mask future tokens")
print("    else:           logit = dot_product")
print()
print("  This creates HEAVY DIVERGENCE when the mask boundary cuts through a warp.")
print()

SEQ_LEN = 32
np.random.seed(7)

def simulate_masked_softmax_divergence(seq_len: int, n_warps_simulated: int = 4):
    """
    Simulate a causal attention mask applied across warps.
    Each warp processes 32 consecutive columns (key positions).
    For each query row, some columns are masked (above diagonal).
    """
    results = []
    logits = np.random.randn(seq_len, seq_len).astype(np.float32)

    # Each warp covers columns [warp*32 : warp*32+32]
    n_warps = math.ceil(seq_len / WARP_SIZE)

    for query_row in range(min(n_warps_simulated, seq_len)):
        for warp_col_start in range(0, seq_len, WARP_SIZE):
            col_indices = np.arange(warp_col_start,
                                    min(warp_col_start + WARP_SIZE, seq_len))
            # Pad to full warp if needed
            n_active = len(col_indices)
            all_cols = np.zeros(WARP_SIZE, dtype=np.int32)
            all_cols[:n_active] = col_indices

            # Causal mask: thread sees True (compute) if col <= row
            mask = (all_cols <= query_row)
            mask[n_active:] = False   # padding threads are masked

            n_compute = mask.sum()
            n_skip    = (~mask[:n_active]).sum()

            # Work per thread:
            #   Compute path: exp(logit) - 3 FP ops
            #   Skip path:    load -inf, write -inf - 2 FP ops
            compute_cycles = 5
            skip_cycles    = 2

            warp_obj = WarpExecution(
                warp_id=warp_col_start // WARP_SIZE,
                thread_data=all_cols.astype(np.float32)
            )
            r = warp_obj.simulate_branch(
                lambda d, row=query_row: d <= row,
                compute_cycles, skip_cycles,
                f"Q={query_row}, cols {warp_col_start}-{warp_col_start+31}"
            )
            results.append({**r, "query_row": query_row,
                            "warp_col_start": warp_col_start,
                            "n_active": n_active})
    return results

results = simulate_masked_softmax_divergence(SEQ_LEN)

print(f"  Sequence length: {SEQ_LEN}, processing first 4 query rows")
print()
print(f"  {'Warp (Q=row, K=cols)':<35} | {'Compute':>9} | {'Skip':>5} | "
      f"{'Passes':>7} | {'Efficiency':>11}")
print(f"  {'─'*75}")

for r in results[:8]:
    label = f"Q={r['query_row']}, K={r['warp_col_start']}-{r['warp_col_start']+31}"
    print(f"  {label:<35} | {r['n_if']:9d} | {r['n_else']:5d} | "
          f"{r['passes']:7d} | {r['simd_efficiency']:10.1f}%")

# Summary statistics
avg_eff = np.mean([r["simd_efficiency"] for r in results])
pct_divergent = np.mean([r["is_divergent"] for r in results]) * 100
print()
print(f"  Summary across all warps:")
print(f"    Average SIMD efficiency: {avg_eff:.1f}%")
print(f"    Fraction of warps that diverge: {pct_divergent:.1f}%")
print()
print("  REAL IMPACT: In FlashAttention, causal masking adds ~10–15% overhead")
print("  for short sequences, but becomes negligible for long sequences (seq > 512)")
print("  because the diagonal \'boundary\' warps are a small fraction of total work.")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Techniques to eliminate divergence
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Techniques to eliminate warp divergence")
print("━" * 65)
print()

print("  TECHNIQUE 1: Predication (cmov instead of branch)")
print("  ──────────────────────────────────────────────────")
print("  Divergent code:")
print("    if (x > 0) { y = x * 2; } else { y = -x; }")
print()
print("  Predicated equivalent (compiler may auto-generate):")
print("    float pos = x * 2.0f;")
print("    float neg = -x;")
print("    y = (x > 0) ? pos : neg;   // select instruction — no branch!")
print()
print("  Both paths are computed by ALL threads, then one is selected.")
print("  Cost: 2× compute, 0× divergence penalty.")
print("  Best when: branches are short and divergence is frequent.")
print()

print("  TECHNIQUE 2: Sort data before processing")
print("  ─────────────────────────────────────────")
print("  Instead of: randomly distributed branches across warps")
print("  Sort input by which branch it takes → all warps are uniform:")
print()

np.random.seed(42)
random_data   = np.random.randn(WARP_SIZE).astype(np.float32)
sorted_data   = np.sort(random_data)

threshold = 0.0
rand_mask   = random_data > threshold
sorted_mask = sorted_data > threshold

print(f"  Unsorted warp: {rand_mask.sum()} positive, {(~rand_mask).sum()} negative → DIVERGENT")
print(f"  Sorted warp:   The first {(~sorted_mask).sum()} threads are negative, "
      f"last {sorted_mask.sum()} are positive")
print(f"  After sort:    If threshold is near median, warp may still diverge once,")
print(f"  but adjacent warps are UNIFORM — overall divergence is minimized.")
print()

print("  TECHNIQUE 3: Warp-uniform condition (predicate on warpId, not threadId)")
print("  ──────────────────────────────────────────────────────────────────────────")
print("  Divergent (threadIdx.x based):")
print("    if (threadIdx.x < 16) { ... }   // threads 0-15 vs 16-31 → DIVERGENT")
print()
print("  Uniform (blockIdx.x based):")
print("    if (blockIdx.x % 2 == 0) { ... }  // entire block takes same path → OK")
print()
print("  Key: any condition that evaluates identically for all 32 threads in a")
print("  warp never causes divergence, regardless of how complex the computation.")
print()

# Quantify impact across a typical kernel
print("━" * 65)
print("  SECTION 4 — Divergence impact summary")
print("━" * 65)
print()

patterns = [
    ("No divergence",               1.00, "Ideal"),
    ("1 thread diverges",           1.00, "Near-ideal (serializes 1 pass)"),
    ("50/50 split, equal work",     2.00, "2× slowdown"),
    ("50/50, one branch 4× heavier",1.80, "1.8× slowdown"),
    ("Random data-dependent branch",1.75, "Typical data-dependent code"),
    ("Sorted + predicated",         1.05, "After optimization"),
]

print(f"  {'Divergence Pattern':<45} | {'Slowdown':>10} | {'SIMD Eff':>10} | {'Notes'}")
print(f"  {'─'*80}")
for label, slowdown, note in patterns:
    eff = min(100.0, 100.0 / slowdown)
    bar = "█" * int(eff / 5) + "░" * (20 - int(eff / 5))
    print(f"  {label:<45} | {slowdown:9.2f}× | {eff:9.1f}% | {note}")
print()
print("  Rule of thumb: divergence is costly mainly when:")
print("    (a) the divergent branch contains many instructions (> ~10 cycles)")
print("    (b) divergence occurs frequently (every warp, every iteration)")
print("    (c) you are already compute-bound (no memory latency to hide it)")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Parallel Reduction — Naive to Warp-Shuffle Optimized": {
        "description": (
            "Implement the classic GPU parallel reduction from scratch in four "
            "progressively optimized versions: naive divergent, interleaved, "
            "shared memory optimized, and warp-shuffle based. "
            "Measure and compare the performance characteristics of each version. "
            "Show how each optimization addresses a specific GPU bottleneck. "
            "Reduction is the foundation of softmax, layer norm, and attention."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 65)
print("  PARALLEL REDUCTION — NAIVE TO WARP-SHUFFLE OPTIMIZED")
print("=" * 65)
print()
print("  Parallel reduction computes: result = sum(array[0..N-1])")
print("  Used in: softmax (sum exp), layer norm (mean/var), attention (rowsum)")
print()

WARP_SIZE = 32
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────
# Simulate CUDA kernel execution: model compute + memory cost
# ─────────────────────────────────────────────────────────────────────────

class KernelMetrics:
    """Track simulated kernel performance metrics."""
    def __init__(self, name: str):
        self.name          = name
        self.global_loads  = 0   # 4-byte words read from global mem
        self.global_stores = 0   # 4-byte words written to global mem
        self.shared_loads  = 0   # shared memory reads
        self.shared_stores = 0   # shared memory writes
        self.sync_barriers = 0   # __syncthreads() calls
        self.add_ops       = 0   # floating point additions
        self.divergent_warps = 0 # warps that diverge
        self.warp_ops      = 0   # total warp-level operations

    @property
    def bytes_global(self):
        return (self.global_loads + self.global_stores) * 4

    @property
    def arithmetic_intensity(self):
        return self.add_ops / self.bytes_global if self.bytes_global > 0 else 0

    def report(self, n_elements: int, block_size: int):
        n_blocks = math.ceil(n_elements / block_size)
        print(f"  {self.name}")
        print(f"    Elements: {n_elements:,}   Block size: {block_size}   "
              f"Blocks launched: {n_blocks}")
        print(f"    Global loads:     {self.global_loads:8,} reads  "
              f"({self.global_loads*4/1024:.1f} KB)")
        print(f"    Global stores:    {self.global_stores:8,} writes "
              f"({self.global_stores*4/1024:.1f} KB)")
        print(f"    Shared mem ops:   {self.shared_loads + self.shared_stores:8,}")
        print(f"    __syncthreads:    {self.sync_barriers:8,}")
        print(f"    FP additions:     {self.add_ops:8,}")
        print(f"    Divergent warps:  {self.divergent_warps:8,}")
        print(f"    Arith. intensity: {self.arithmetic_intensity:8.2f} FLOPs/byte")
        print()


# ─────────────────────────────────────────────────────────────────────────
# Version 1: Naive divergent reduction
# ─────────────────────────────────────────────────────────────────────────

def reduction_naive(data: np.ndarray, block_size: int = 256) -> tuple:
    """
    CUDA equivalent:
        __global__ void reduce_naive(float* g_in, float* g_out, int n) {
            extern __shared__ float sdata[];
            int tid = threadIdx.x;
            int gid = blockIdx.x * blockDim.x + threadIdx.x;
            sdata[tid] = (gid < n) ? g_in[gid] : 0.0f;
            __syncthreads();

            for (int s = 1; s < blockDim.x; s *= 2) {
                if (tid % (2*s) == 0)            // DIVERGENT: tid check
                    sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            if (tid == 0) g_out[blockIdx.x] = sdata[0];
        }

    Problems:
        1. tid % (2*s) == 0 → divergence: lower-half warps work, upper idle
        2. Access pattern: sdata[0], sdata[2], sdata[4]... → stride increases → bank conflicts
        3. Active thread count halves each round → most threads idle quickly
    """
    n = len(data)
    m = KernelMetrics("V1: Naive (divergent branches, stride access)")
    n_blocks = math.ceil(n / block_size)

    # Simulate across all blocks
    block_sums = []
    for block_id in range(n_blocks):
        start = block_id * block_size
        end   = min(start + block_size, n)
        sdata = np.zeros(block_size, dtype=np.float64)
        sdata[:end - start] = data[start:end]

        # Load from global to shared
        m.global_loads  += block_size
        m.shared_stores += block_size
        m.sync_barriers += 1

        # Reduction loop: s = 1, 2, 4, ..., block_size/2
        s = 1
        while s < block_size:
            for tid in range(block_size):
                if tid % (2 * s) == 0 and (tid + s) < block_size:
                    sdata[tid] += sdata[tid + s]
                    m.add_ops += 1
                    m.shared_loads  += 2
                    m.shared_stores += 1
            # Divergence: only threads where tid%(2s)==0 are active
            # In the first round (s=1): threads 0,2,4,...31 active in each warp
            # → all 32 warps diverge (odd threads idle)
            warps_this_round = block_size // WARP_SIZE
            m.divergent_warps += warps_this_round
            m.sync_barriers += 1
            s *= 2

        block_sums.append(sdata[0])
        m.shared_loads  += 1
        m.global_stores += 1

    return np.sum(block_sums), m


# ─────────────────────────────────────────────────────────────────────────
# Version 2: Interleaved addressing (no divergence within warp)
# ─────────────────────────────────────────────────────────────────────────

def reduction_interleaved(data: np.ndarray, block_size: int = 256) -> tuple:
    """
    CUDA equivalent:
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s)                         // NOT divergent within most warps
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

    Improvement over naive:
        tid < s  instead of  tid % (2s) == 0
        → upper half threads idle, lower half active, but NO DIVERGENCE
          within a warp (warp either all-active or all-idle)
        → fixes divergence problem at the cost of accessing sdata with stride s

    Remaining issue: bank conflicts when s is a multiple of 32.
    """
    n = len(data)
    m = KernelMetrics("V2: Interleaved (no divergence, some bank conflicts)")
    n_blocks = math.ceil(n / block_size)

    block_sums = []
    for block_id in range(n_blocks):
        start = block_id * block_size
        end   = min(start + block_size, n)
        sdata = np.zeros(block_size, dtype=np.float64)
        sdata[:end - start] = data[start:end]

        m.global_loads  += block_size
        m.shared_stores += block_size
        m.sync_barriers += 1

        s = block_size // 2
        while s > 0:
            for tid in range(s):   # only lower half threads active
                sdata[tid] += sdata[tid + s]
                m.add_ops += 1
                m.shared_loads  += 2
                m.shared_stores += 1
            # No divergence: warps are either entirely in [0..s) or entirely above
            # Bank conflict only when s is multiple of 32 (sdata[tid] and sdata[tid+s]
            # land on same bank when s%32==0)
            if s % 32 == 0:
                m.divergent_warps += 0   # no divergence
            m.sync_barriers += 1
            s >>= 1

        block_sums.append(sdata[0])
        m.shared_loads  += 1
        m.global_stores += 1

    return np.sum(block_sums), m


# ─────────────────────────────────────────────────────────────────────────
# Version 3: Bank-conflict-free + loop unrolling
# ─────────────────────────────────────────────────────────────────────────

def reduction_optimized(data: np.ndarray, block_size: int = 256) -> tuple:
    """
    CUDA best practices:
        1. Interleaved addressing (no divergence) ← already done in V2
        2. Load two elements per thread on first step (halve blocks needed)
        3. Unroll last warp: when s <= 32, warp is running synchronously
           → no __syncthreads needed, use volatile pointer
        4. Full unroll with #pragma unroll

    Additional optimization: first reduction step done during global load:
        sdata[tid] = g_in[gid] + g_in[gid + blockDim.x];
        → processes 2N elements per block, cuts total blocks in half
    """
    n = len(data)
    m = KernelMetrics("V3: Optimized (2-elem load, unroll, conflict-free)")
    # 2 elements per thread on load → block_size effective = block_size * 2
    effective_n_per_block = block_size * 2
    n_blocks = math.ceil(n / effective_n_per_block)

    block_sums = []
    for block_id in range(n_blocks):
        start  = block_id * effective_n_per_block
        mid    = start + block_size
        sdata  = np.zeros(block_size, dtype=np.float64)

        for tid in range(block_size):
            g0 = start + tid
            g1 = mid   + tid
            v0 = data[g0] if g0 < n else 0.0
            v1 = data[g1] if g1 < n else 0.0
            sdata[tid] = v0 + v1
            m.global_loads += 2
            m.shared_stores += 1
            m.add_ops += 1

        m.sync_barriers += 1

        s = block_size // 2
        while s > WARP_SIZE:
            for tid in range(s):
                sdata[tid] += sdata[tid + s]
                m.add_ops += 1
                m.shared_loads  += 2
                m.shared_stores += 1
            m.sync_barriers += 1
            s >>= 1

        # Last warp: unrolled (no __syncthreads needed)
        # volatile float* smem = sdata; — implicit synchronization within warp
        for unroll_s in [32, 16, 8, 4, 2, 1]:
            if unroll_s <= block_size // 2:
                for tid in range(min(unroll_s, block_size)):
                    if tid + unroll_s < block_size:
                        sdata[tid] += sdata[tid + unroll_s]
                        m.add_ops += 1
                        m.shared_loads  += 2
                        m.shared_stores += 1

        block_sums.append(sdata[0])
        m.shared_loads  += 1
        m.global_stores += 1

    return np.sum(block_sums), m


# ─────────────────────────────────────────────────────────────────────────
# Version 4: Warp-shuffle reduction (no shared memory needed)
# ─────────────────────────────────────────────────────────────────────────

def reduction_warp_shuffle(data: np.ndarray, block_size: int = 256) -> tuple:
    """
    Modern CUDA (compute capability >= 3.0) uses warp shuffle intrinsics:

        __shfl_down_sync(mask, val, offset):
            Thread i receives the value of thread (i + offset) in the same warp.
            If (i + offset) >= 32: thread i keeps its own value.
            No shared memory access: communication happens through the register file!

    Warp-level reduction:
        val += __shfl_down_sync(0xffffffff, val, 16);  // add from lane+16
        val += __shfl_down_sync(0xffffffff, val, 8);   // add from lane+8
        val += __shfl_down_sync(0xffffffff, val, 4);   // add from lane+4
        val += __shfl_down_sync(0xffffffff, val, 2);   // add from lane+2
        val += __shfl_down_sync(0xffffffff, val, 1);   // add from lane+1
        // Lane 0 now holds the warp sum

    Then one shared memory write per warp for cross-warp reduction.
    Benefits:
        1. No shared memory bank conflicts (register-to-register transfer)
        2. 5 cycles vs ~5 cycles smem, but eliminates __syncthreads overhead
        3. Only blockDim/32 shared mem accesses (vs blockDim for traditional)
    """
    n = len(data)
    m = KernelMetrics("V4: Warp-shuffle (register-based, no bank conflicts)")
    n_blocks = math.ceil(n / block_size)
    n_warps  = block_size // WARP_SIZE

    block_sums = []
    for block_id in range(n_blocks):
        start = block_id * block_size
        end   = min(start + block_size, n)

        # Each thread holds its value in a register
        thread_vals = np.zeros(block_size, dtype=np.float64)
        for tid in range(block_size):
            gid = start + tid
            thread_vals[tid] = data[gid] if gid < n else 0.0
            m.global_loads += 1

        # Warp-level shuffle reduction (5 steps per warp)
        warp_sums = []
        for warp_id in range(n_warps):
            lane_vals = thread_vals[warp_id*WARP_SIZE : (warp_id+1)*WARP_SIZE].copy()

            for offset in [16, 8, 4, 2, 1]:
                shifted = np.zeros(WARP_SIZE, dtype=np.float64)
                for lane in range(WARP_SIZE):
                    src = lane + offset
                    shifted[lane] = lane_vals[src] if src < WARP_SIZE else lane_vals[lane]
                lane_vals += shifted
                m.add_ops += WARP_SIZE
                # shfl_down counts as register operation — no shared memory

            warp_sums.append(lane_vals[0])
            m.shared_stores += 1   # only 1 store per warp (warp sum → smem)

        # Cross-warp reduction: n_warps sums in shared memory (small reduction)
        m.shared_loads += n_warps
        m.sync_barriers += 1   # one syncthreads to ensure warp sums are visible

        warp_arr = np.array(warp_sums)
        for offset in [n_warps // 2]:
            while offset > 0:
                for i in range(offset):
                    if i + offset < len(warp_arr):
                        warp_arr[i] += warp_arr[i + offset]
                        m.add_ops += 1
                        m.shared_loads  += 2
                        m.shared_stores += 1
                offset //= 2

        block_sums.append(warp_arr[0])
        m.global_stores += 1

    return np.sum(block_sums), m


# ─────────────────────────────────────────────────────────────────────────
# Run all versions and compare
# ─────────────────────────────────────────────────────────────────────────

N          = 1024 * 1024   # 1M elements
BLOCK_SIZE = 256
data       = np.random.rand(N).astype(np.float32)
reference  = float(data.sum())   # numpy reference

print("━" * 65)
print(f"  REDUCTION BENCHMARKS  —  N={N:,}  block_size={BLOCK_SIZE}")
print("━" * 65)
print()

versions = [
    reduction_naive,
    reduction_interleaved,
    reduction_optimized,
    reduction_warp_shuffle,
]

metrics_list = []
for fn in versions:
    result, m = fn(data, BLOCK_SIZE)
    err = abs(result - reference) / (abs(reference) + 1e-9)
    metrics_list.append((m, err, result))

print(f"  Reference sum (NumPy): {reference:.6f}")
print()

for m, err, result in metrics_list:
    m.report(N, BLOCK_SIZE)

# Summary table
print("━" * 65)
print("  COMPARISON SUMMARY")
print("━" * 65)
print()
print(f"  {'Version':<45} | {'Glb Loads':>10} | {'Barriers':>9} | "
      f"{'AI':>8} | {'Div Warps':>10} | {'Error':>8}")
print(f"  {'─'*97}")
for m, err, result in metrics_list:
    print(f"  {m.name:<45} | {m.global_loads:10,} | {m.sync_barriers:9,} | "
          f"{m.arithmetic_intensity:7.3f} | {m.divergent_warps:10,} | {err:.2e}")

print()
print("  TAKEAWAYS:")
print("    V1 → V2: Eliminate divergence. Active threads form contiguous ranges.")
print("    V2 → V3: 2-elem load halves blocks. Unroll last warp → fewer barriers.")
print("    V3 → V4: Warp shuffle eliminates 31/32 of shared memory accesses.")
print("    V4 barrier count ≈ log2(block/32) + 1  (vs log2(block) for V1-V3)")
print()
print("  In practice, cuDNN/cuBLAS use warp-shuffle reduction internally.")
print("  PyTorch\'s torch.sum() calls these optimized reduction kernels.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Matrix Multiplication — Naive vs Shared Memory Tiled GEMM": {
        "description": (
            "Implement matrix multiplication (GEMM) in three versions: "
            "naive (one thread per output element, global memory only), "
            "shared memory tiled (load tiles into L1 to reuse data), "
            "and a double-buffered prefetch version that hides memory latency. "
            "Compute arithmetic intensity, effective FLOP/s, and bandwidth for each. "
            "GEMM is the core operation in every dense neural network layer."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time

print("=" * 65)
print("  MATRIX MULTIPLICATION — NAIVE TO TILED GEMM")
print("=" * 65)
print()
print("  GEMM: C = A × B where A is M×K, B is K×N, C is M×N")
print("  Every dense layer, attention projection, and FFN is a GEMM.")
print()

np.random.seed(1)

# ─────────────────────────────────────────────────────────────────────────
# Performance model: count memory transactions and FLOPs
# ─────────────────────────────────────────────────────────────────────────

def gemm_naive_analysis(M: int, K: int, N: int,
                         block_size: int = 16) -> dict:
    """
    Naive GEMM: one thread per output element C[i,j].
    Each thread reads full row i of A and full column j of B.
    No data reuse: every element loaded independently.

    CUDA kernel structure:
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row*K + k] * B[k*N + col];   // global memory reads each iter
        }
        C[row*N + col] = sum;
    """
    total_output_elements = M * N

    # FLOPs: each output element needs K multiply-adds = 2K FLOPs
    flops = 2.0 * M * N * K

    # Global memory reads:
    # Each thread reads K elements from A and K elements from B (no reuse!)
    # Because threads in the same block load OVERLAPPING rows of A
    # and overlapping columns of B.
    reads_A = M * N * K   # every C[i,j] reads row i of A: M*N threads each read K
    reads_B = M * N * K   # every C[i,j] reads col j of B: M*N threads each read K
    reads_C = 0            # C is write-only

    # Each A[row][k] is read N times (once per column in the same row)
    # → reuse factor in theory is N, but naive kernel has no caching
    # In practice L1/L2 caches catch SOME, but not systematically
    total_reads_bytes = (reads_A + reads_B) * 4   # float32 = 4 bytes

    # Writes: each output element written once
    writes_C_bytes = M * N * 4

    total_bytes = total_reads_bytes + writes_C_bytes

    arithmetic_intensity = flops / total_bytes

    return {
        "version":              "Naive (global memory only)",
        "flops":                flops,
        "reads_A_bytes":        reads_A * 4,
        "reads_B_bytes":        reads_B * 4,
        "writes_C_bytes":       writes_C_bytes,
        "total_bytes":          total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "global_loads_per_output": 2 * K,   # K from A + K from B per thread
        "memory_bound":         arithmetic_intensity < 156,  # A100 ridge point
    }


def gemm_tiled_analysis(M: int, K: int, N: int,
                         tile_size: int = 16) -> dict:
    """
    Tiled GEMM with shared memory.

    Key idea: instead of each thread loading K elements independently,
    a block of tile_size × tile_size threads COOPERATIVELY loads a
    tile_size × tile_size tile from A and B into shared memory.
    Then each thread uses the shared tile (fast, ~5 cycle latency).

    CUDA kernel structure:
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];
        for (int t = 0; t < K/TILE; t++) {
            // COALESCED load of tile from A and B (all threads cooperate)
            As[ty][tx] = A[row][t*TILE + tx];
            Bs[ty][tx] = B[t*TILE + ty][col];
            __syncthreads();
            // Compute: each thread does TILE multiply-adds using shared mem
            for (int k = 0; k < TILE; k++)
                sum += As[ty][k] * Bs[k][tx];
            __syncthreads();
        }

    Reuse factor: each element loaded into shared memory is used TILE times
    → global memory traffic reduced by factor TILE vs naive.
    """
    T = tile_size
    n_tiles = math.ceil(K / T)

    flops = 2.0 * M * N * K

    # Global reads of A and B: each tile loaded ONCE from global memory
    # tile_size * tile_size elements per tile
    # Number of tiles of A: (M/T) * (K/T)  → total elements = M*K
    # But each element is loaded exactly once per thread block that uses it.
    # In tiled GEMM: each element of A is loaded by (N/T) blocks → N/T times
    # Wait — let\'s count differently:
    # Total A reads: each row of A is read by all column-blocks = (N/T) blocks
    # → each element of A read N/T times = (N/T) × M×K element reads
    # This equals M*N*K/T total reads → T× reduction vs naive
    reads_A_elements = (M * K) * math.ceil(N / T)  # each A row read by ceil(N/T) col-blocks
    reads_B_elements = (K * N) * math.ceil(M / T)  # each B col read by ceil(M/T) row-blocks

    # But each tile is read as a coalesced block → cache line efficiency = 1
    reads_A_bytes = reads_A_elements * 4
    reads_B_bytes = reads_B_elements * 4
    writes_C_bytes = M * N * 4
    total_bytes = reads_A_bytes + reads_B_bytes + writes_C_bytes

    arithmetic_intensity = flops / total_bytes

    # Shared memory accesses: each tile element is accessed T times
    smem_reads_per_block = 2 * T * T * T   # T² loads per tile × T iterations
    total_blocks = (M // T) * (N // T)
    total_smem_reads = smem_reads_per_block * total_blocks

    return {
        "version":              f"Tiled GEMM (tile={T}×{T} shared mem)",
        "tile_size":            T,
        "flops":                flops,
        "reads_A_bytes":        reads_A_bytes,
        "reads_B_bytes":        reads_B_bytes,
        "writes_C_bytes":       writes_C_bytes,
        "total_bytes":          total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "global_loads_per_output": 2 * K / T,   # T× reduction per tile
        "smem_reuse_factor":    T,
        "memory_bound":         arithmetic_intensity < 156,
    }


def gemm_tensor_core_analysis(M: int, K: int, N: int) -> dict:
    """
    cuBLAS-style GEMM with Tensor Cores (FP16).
    Tile size = 16×16 minimum per Tensor Core operation.
    Register tiling: 8×8 to 16×16 output per thread.
    Double buffering: prefetch next tile while computing current.

    Achieves near-peak Tensor Core utilization with:
        - 16×16×16 WMMA operations (one Tensor Core operation)
        - Prefetch: async global→shared load overlaps with compute
        - Very high arithmetic intensity → compute-bound
    """
    T = 16
    flops = 2.0 * M * N * K

    # With double buffering and careful layout, data reuse is maximized.
    # Practical implementations achieve >85% memory bandwidth efficiency.
    # Global reads reduced to theoretical minimum: one pass over A and B.
    reads_A_bytes = M * K * 2   # FP16 = 2 bytes
    reads_B_bytes = K * N * 2
    writes_C_bytes = M * N * 4  # FP32 accumulation

    total_bytes = reads_A_bytes + reads_B_bytes + writes_C_bytes
    arithmetic_intensity = flops / total_bytes

    return {
        "version":              "Tensor Core GEMM (cuBLAS, FP16 in/FP32 acc)",
        "tile_size":            T,
        "flops":                flops,
        "reads_A_bytes":        reads_A_bytes,
        "reads_B_bytes":        reads_B_bytes,
        "writes_C_bytes":       writes_C_bytes,
        "total_bytes":          total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "global_loads_per_output": 2 * K / T,
        "memory_bound":         arithmetic_intensity < 156,
    }


# ─────────────────────────────────────────────────────────────────────────
# Performance model: predict throughput
# ─────────────────────────────────────────────────────────────────────────

A100_HBM_BW    = 2.0e12     # bytes/sec
A100_FP32_TFLOPS = 19.5e12  # FLOP/sec
A100_FP16_TC   = 312.0e12   # FLOP/sec (with Tensor Cores)

def predict_throughput(analysis: dict, fp16_tc: bool = False) -> dict:
    """Predict achieved FLOP/s using the roofline model."""
    peak_flops = A100_FP16_TC if fp16_tc else A100_FP32_TFLOPS
    ai = analysis["arithmetic_intensity"]

    # Roofline: min of memory-limited and compute-limited performance
    memory_limited_flops = ai * A100_HBM_BW
    achievable_flops     = min(memory_limited_flops, peak_flops)

    # Time to execute
    time_s = analysis["flops"] / achievable_flops

    # Utilization of peak
    peak_util_pct = achievable_flops / peak_flops * 100

    return {
        **analysis,
        "achievable_gflops": achievable_flops / 1e9,
        "peak_gflops":       peak_flops / 1e9,
        "peak_util_pct":     peak_util_pct,
        "predicted_time_ms": time_s * 1e3,
        "bottleneck":        "Compute" if not analysis["memory_bound"] else "Memory BW",
    }


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Analysis for a single matrix size
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — GEMM analysis: M=N=K=1024 on A100")
print("━" * 65)
print()

M = N = K = 1024

analyses = [
    (gemm_naive_analysis(M, K, N),               False),
    (gemm_tiled_analysis(M, K, N, tile_size=16), False),
    (gemm_tiled_analysis(M, K, N, tile_size=32), False),
    (gemm_tensor_core_analysis(M, K, N),         True),
]

for analysis, is_fp16 in analyses:
    r = predict_throughput(analysis, fp16_tc=is_fp16)
    print(f"  ── {r['version']}")
    print(f"     FLOPs:              {r['flops']/1e9:.2f} GFLOP")
    print(f"     Global mem read A:  {r['reads_A_bytes']/1e6:.1f} MB")
    print(f"     Global mem read B:  {r['reads_B_bytes']/1e6:.1f} MB")
    print(f"     Arith. intensity:   {r['arithmetic_intensity']:.1f} FLOP/byte  "
          f"→ {'⚠️  Memory-bound' if r['memory_bound'] else '✅ Compute-bound'}")
    print(f"     Achievable perf:    {r['achievable_gflops']:.1f} GFLOPS "
          f"({r['peak_util_pct']:.1f}% of peak)")
    print(f"     Predicted time:     {r['predicted_time_ms']:.3f} ms")
    print(f"     Bottleneck:         {r['bottleneck']}")
    print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Tiling — data reuse explained with numbers
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Data reuse: why tiling is transformative")
print("━" * 65)
print()

print("  For M=N=K=N (square GEMM):")
print()
print(f"  {'N':>8} | {'Naive reads (GB)':>17} | {'Tiled T=16 (GB)':>17} | "
      f"{'Reuse factor':>14} | {'AI (tiled)':>11}")
print(f"  {'─'*72}")

for N_val in [128, 256, 512, 1024, 2048, 4096]:
    naive  = gemm_naive_analysis(N_val, N_val, N_val)
    tiled  = gemm_tiled_analysis(N_val, N_val, N_val, tile_size=16)
    reuse  = naive["total_bytes"] / tiled["total_bytes"]
    print(f"  {N_val:8d} | {naive['total_bytes']/1e9:17.2f} | "
          f"{tiled['total_bytes']/1e9:17.2f} | {reuse:14.1f}× | "
          f"{tiled['arithmetic_intensity']:10.1f}")

print()
print("  Key insight: as N grows, tiled GEMM arithmetic intensity → N/2.")
print("  For large matrices (N=4096), tiled GEMM becomes COMPUTE-BOUND even on A100.")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Shared memory layout for the tile
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Shared memory tile layout (16×16 tile)")
print("━" * 65)
print()

TILE = 4   # small for display purposes

print(f"  Tile = {TILE}×{TILE} = {TILE*TILE} floats = {TILE*TILE*4} bytes")
print()
print("  Step 0: Each of the 16 threads loads ONE element from global A and ONE from B.")
print()

# Show the tile coordinates each thread loads
print(f"  {'Thread (ty,tx)':<20} → A_tile[ty][tx] = A[row_base + ty][col_tile + tx]")
print(f"  {'Thread (ty,tx)':<20} → B_tile[ty][tx] = B[col_tile + ty][col_base + tx]")
print()

A_tile = np.array([[f"A[{ty}][{tx}]" for tx in range(TILE)] for ty in range(TILE)])
B_tile = np.array([[f"B[{ty}][{tx}]" for tx in range(TILE)] for ty in range(TILE)])

print("  Shared memory A tile (As[ty][tx]):")
for ty in range(TILE):
    print("    " + "  ".join(f"{A_tile[ty][tx]:>10}" for tx in range(TILE)))
print()
print("  Shared memory B tile (Bs[ty][tx]):")
for ty in range(TILE):
    print("    " + "  ".join(f"{B_tile[ty][tx]:>10}" for tx in range(TILE)))
print()
print("  After loading, each thread computes:")
print("    for k in range(TILE):")
print("      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]")
print()
print(f"  TILE={TILE}: each global memory element is reused {TILE} times from shared memory.")
print(f"  TILE=16:  each global memory element is reused 16 times → 16× fewer HBM accesses.")
print(f"  TILE=32:  32× reduction, but shared mem usage = 2 × 32×32×4 = 8 KB per block")
print()
print("  Access coalescing check for the tile load:")
print("  A load:  thread (ty, tx) reads A[row+ty][t*TILE+tx]")
print("           Within a row: tx=0,1,...,TILE-1 → consecutive addresses → ✅ coalesced")
print("  B load:  thread (ty, tx) reads B[t*TILE+ty][col+tx]")
print("           Within a row: tx=0,...,TILE-1 → consecutive in col direction → ✅ coalesced")
print()
print("  Both A and B tiles are loaded with perfectly coalesced accesses.")
print("  This is why tiled GEMM achieves near-peak bandwidth utilization on reads.")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Correctness check — compare numpy vs tiled simulation
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 4 — Correctness check: tiled GEMM simulation vs NumPy")
print("━" * 65)
print()

def tiled_gemm_sim(A: np.ndarray, B: np.ndarray, tile: int = 4) -> np.ndarray:
    """
    Pure Python simulation of the shared-memory tiled GEMM algorithm.
    Demonstrates the exact tile loop structure of the CUDA kernel.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "K dimensions must match"
    C = np.zeros((M, N), dtype=np.float64)

    for block_row in range(0, M, tile):
        for block_col in range(0, N, tile):
            # This block computes tile×tile submatrix of C
            c_sub = np.zeros((tile, tile), dtype=np.float64)

            for t in range(0, K, tile):
                # Load tile from A and B into shared memory
                A_tile = A[block_row:block_row+tile, t:t+tile].astype(np.float64)
                B_tile = B[t:t+tile, block_col:block_col+tile].astype(np.float64)

                # Pad if at boundary
                a_rows, a_cols = A_tile.shape
                b_rows, b_cols = B_tile.shape
                if a_rows < tile or a_cols < tile:
                    tmp = np.zeros((tile, tile), dtype=np.float64)
                    tmp[:a_rows, :a_cols] = A_tile
                    A_tile = tmp
                if b_rows < tile or b_cols < tile:
                    tmp = np.zeros((tile, tile), dtype=np.float64)
                    tmp[:b_rows, :b_cols] = B_tile
                    B_tile = tmp

                # Compute: all threads execute this inner loop from shared memory
                c_sub += A_tile @ B_tile

            # Write tile back to C
            r_end = min(block_row + tile, M)
            c_end = min(block_col + tile, N)
            C[block_row:r_end, block_col:c_end] = c_sub[:r_end-block_row, :c_end-block_col]

    return C

M_s, K_s, N_s = 16, 16, 16
A_mat = np.random.rand(M_s, K_s).astype(np.float32)
B_mat = np.random.rand(K_s, N_s).astype(np.float32)

C_numpy = A_mat @ B_mat
C_tiled = tiled_gemm_sim(A_mat, B_mat, tile=4).astype(np.float32)

max_err = np.max(np.abs(C_numpy - C_tiled))
rel_err = max_err / (np.max(np.abs(C_numpy)) + 1e-9)

print(f"  Matrix size: {M_s}×{K_s} × {K_s}×{N_s}")
print(f"  Max absolute error: {max_err:.2e}")
print(f"  Max relative error: {rel_err:.2e}")
print(f"  Correctness: {'✅ PASS' if rel_err < 1e-4 else '❌ FAIL'}")
print()
print("  The tiled algorithm produces identical results to naive GEMM.")
print("  The only difference is HOW memory is accessed — not WHAT is computed.")
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