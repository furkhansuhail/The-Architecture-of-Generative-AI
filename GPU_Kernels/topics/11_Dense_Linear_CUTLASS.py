"""
CUTLASS — CUDA Templates for Linear Algebra Subroutines
=========================================================

CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is
NVIDIA's open-source C++ template library for implementing high-performance
matrix multiplication and related operations directly in CUDA. It is the
building block beneath TensorRT's custom GEMM kernels, the reference
implementation used to validate cuBLAS, and increasingly the tool of choice
for ML researchers and compiler engineers who need GPU kernels that are
faster, more flexible, or more specialised than what cuDNN or cuBLAS provide.

The key distinction from cuDNN: cuDNN is a black box — you call a function,
NVIDIA's pre-compiled code runs. CUTLASS is a white box — you write the
kernel using CUTLASS's template abstractions, and the compiler generates
optimised PTX for your specific shapes, data types, and hardware.

This white-box approach means:
    - You can implement operations that cuDNN doesn't support
    - You can fuse custom operations into the GEMM epilogue
    - You can target Tensor Cores with custom data types
    - You control the exact algorithm, tile sizes, and pipeline depth
    - Your kernel can be compiled for the exact GPU architecture you target

cuDNN uses CUTLASS-like primitives internally. TensorRT generates CUTLASS
kernels for non-standard shapes. Flash Attention (FlashAttention-2/3) uses
CUTLASS for its inner loops. Understanding CUTLASS means understanding how
GPU kernels actually achieve peak performance.

"""

import textwrap
import re

TOPIC_NAME   = "CUTLASS — CUDA Templates for Linear Algebra"
DISPLAY_NAME = "11 · CUTLASS"
ICON         = "⚙️"
SUBTITLE     = "Custom CUDA GEMM Kernels, Tensor Cores, and Epilogue Fusion"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT CUTLASS IS AND WHY IT EXISTS

### The Gap Between cuBLAS and Custom Kernels

    cuBLAS's SGEMM (FP32 matrix multiply) delivers near-peak performance
    for standard shapes. But many ML workloads have constraints that
    cuBLAS cannot accommodate:

    1. CUSTOM EPILOGUES:
        cuBLAS computes: D = α × A × B + β × C
        ML needs:        D = ReLU(A × B + bias)  in ONE kernel
        Without CUTLASS: two kernel calls, one unnecessary HBM round-trip.
        With CUTLASS:    epilogue can be any fused operation.

    2. NON-STANDARD DATA TYPES:
        cuBLAS supports: FP32, FP64, FP16, BF16, INT8.
        Research needs:  INT4, E3M4, binary, custom quantised formats.
        CUTLASS:         extensible to arbitrary element types via templates.

    3. SPECIFIC TILE SIZES:
        cuBLAS auto-selects tile sizes based on heuristics.
        For a specific (M=128, N=7, K=4096) shape: cuBLAS may not be optimal.
        CUTLASS: you specify exact tile sizes (ThreadblockShape, WarpShape).

    4. TENSOR CORE PROGRAMMING:
        Tensor Cores accept matrix fragments in specific register layouts.
        Programming Tensor Cores in raw CUDA is highly complex.
        CUTLASS abstracts Tensor Core operations behind ergonomic C++ templates.

    5. NOVEL ALGORITHMS:
        FlashAttention's inner kernel (QK^T, softmax, PV) doesn't fit
        standard GEMM APIs but is implemented using CUTLASS primitives.
        Researchers writing new attention variants use CUTLASS.

### CUTLASS's Position in the Ecosystem

    ┌───────────────────────────────────────────────────────────────┐
    │  ML Frameworks: PyTorch, TensorFlow, JAX                      │
    └───────────────────────────────────────────────────────────────┘
                        ↓  calls
    ┌───────────────────────────────────────────────────────────────┐
    │  cuDNN / cuBLAS  (pre-compiled, opaque)                       │
    └───────────────────────────────────────────────────────────────┘
                        ↓  some operations compiled from
    ┌───────────────────────────────────────────────────────────────┐
    │  CUTLASS  (C++ template library — white box)                  │
    │  Also used directly by: TensorRT, Triton, FlashAttention,     │
    │  custom research kernels                                      │
    └───────────────────────────────────────────────────────────────┘
                        ↓  compiles to
    ┌───────────────────────────────────────────────────────────────┐
    │  CUDA PTX / SASS  (GPU assembly)                              │
    └───────────────────────────────────────────────────────────────┘
                        ↓  executes on
    ┌───────────────────────────────────────────────────────────────┐
    │  NVIDIA GPU: Tensor Cores, CUDA Cores, HBM                    │
    └───────────────────────────────────────────────────────────────┘

### Who Uses CUTLASS

    TensorRT:      Generates CUTLASS-based kernels for shapes where its
                   profiling engine finds them faster than cuBLAS.

    FlashAttention: The inner compute kernels of FA2 and FA3 are written
                    using CUTLASS's MMA (Matrix Multiply-Accumulate) primitives.

    Triton:        OpenAI's Triton compiler uses CUTLASS's tiling strategy
                   as a design reference (but generates its own kernels).

    XFormers:      Memory-efficient attention uses CUTLASS for its kernels.

    Research:      Quantisation research (INT4 GEMM, FP8 GEMM), sparse
                   attention (block-sparse patterns), LoRA fine-tuning
                   (low-rank GEMM) — all implemented in CUTLASS.


##### PART 2 — THE GEMM HIERARCHY: HOW MATRIX MULTIPLY MAPS TO GPU

### The Tiling Hierarchy

    Matrix multiplication D = A × B + C has three nested loops:
        for m in range(M):
            for n in range(N):
                for k in range(K):
                    D[m,n] += A[m,k] × B[k,n]

    On a GPU, this maps to a THREE-LEVEL tiling hierarchy:

    LEVEL 1: Threadblock (CTA) tile
        The entire (M × N × K) problem is split into tiles.
        Each threadblock (CTA) computes one tile of D.
        Tile size: ThreadblockShape = (128, 128, 32) for FP16 Tensor Cores.
        The A tile and B tile are loaded from HBM → Shared Memory.
        128 × 128 × 32 × 2 = 1 MB shared memory usage per block.

    LEVEL 2: Warp tile
        Each threadblock has multiple warps.
        Each warp computes its portion of the block's output tile.
        Warp tile: WarpShape = (32, 64, 32) for FP16.
        Warps cooperate via Shared Memory (no direct communication).

    LEVEL 3: Instruction tile (MMA — Matrix Multiply-Accumulate)
        Each warp computes using Tensor Core MMA instructions.
        MMA shape: InstructionShape = (16, 8, 16) for FP16 (wmma::m16n8k16).
        One MMA instruction: 16×8 output = 128 elements from one warp.

    The full computation:
        Grid of threadblocks → each block handles one 128×128 output tile
        Each block: 4 warps → each warp handles one 32×64 tile
        Each warp: 8 MMA instructions → each instruction: 16×8 output

### The Software Pipeline: Hiding Memory Latency

    HBM access latency: ~500 cycles (memory stall if not hidden)
    Tensor Core throughput: complete 16×8×16 MMA in ~10 cycles

    Without pipelining: Compute → Wait for memory → Compute → Wait...
    With pipelining:    Compute Tile_i while loading Tile_{i+1}

    CUTLASS implements a software pipeline with N stages:
        Stage 1: Load A tile K[0:32] and B tile K[0:32] → Shared Memory
        Stage 2: Load A tile K[32:64] and B tile K[32:64] → Shared Memory
        While Stage 2 is loading, compute Stage 1 using Tensor Cores

    Pipeline depth N (number of stages):
        N=2: one tile loading while one is being computed
        N=3: two tiles in flight while one is being computed
        N=4: three tiles pipelining (requires 3× shared memory)

    Larger N = better latency hiding = higher throughput
    But: shared memory is limited (~192 KB), so N is bounded.

    CUTLASS's multistage pipeline is a key innovation that gets
    GPU Tensor Core utilisation close to theoretical maximum.

### Threadblock Swizzling

    Which threadblocks execute on which SMs? The order matters for cache.

    Default: blocks execute in row-major order of tile indices.
        (0,0), (0,1), (0,2), ... (0,N/128), (1,0), (1,1), ...

    Problem: consecutive blocks in the same row share NO rows of matrix A.
             Consecutive blocks in the same column share no columns of B.
             L2 cache reuse is poor.

    Threadblock swizzle: reorder blocks so nearby blocks share L2 data.
        Hilbert curve ordering: space-filling curve preserves 2D locality.
        Identity + L-shaped: groups blocks that share an A panel or B panel.

    CUTLASS provides multiple swizzle patterns as template parameters.
    The right swizzle can improve performance 5–15% on large matrices.


##### PART 3 — TENSOR CORES: THE WMMA AND MMA INSTRUCTION SETS

### WMMA (Warp Matrix Multiply-Accumulate)

    WMMA is a CUDA C++ API for programming Tensor Cores:
        #include <mma.h>
        using namespace nvcuda::wmma;

        // Fragment types (describe the matrix pieces held in registers)
        fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;

        // Load data into fragments (from shared memory)
        load_matrix_sync(a_frag, a_ptr, A_stride);   // loads 16×16 half matrix
        load_matrix_sync(b_frag, b_ptr, B_stride);   // loads 16×16 half matrix
        fill_fragment(c_frag, 0.0f);                  // initialise accumulator

        // Execute one MMA instruction: C += A × B
        mma_sync(c_frag, a_frag, b_frag, c_frag);   // uses ALL 32 threads of the warp

        // Store result to memory
        store_matrix_sync(c_ptr, c_frag, C_stride, mem_row_major);

    This is WARP-LEVEL programming: all 32 threads in the warp participate.
    One mma_sync() call does 16×16×16 = 4096 multiply-accumulate operations
    using Tensor Core hardware in ~10 clock cycles.

### MMA Instructions (PTX Level, what CUTLASS uses internally)

    WMMA is high-level. CUTLASS uses lower-level PTX MMA instructions:
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
            {d0, d1, d2, d3},       // 8 FP32 output registers (4 per thread)
            {a0, a1, a2, a3},       // 8 FP16 input A registers
            {b0, b1},               // 4 FP16 input B registers
            {c0, c1, c2, c3};       // 8 FP32 accumulator registers

    The instruction shape m16n8k16: 16 rows of A, 8 cols of B, 16 depth
    Each warp (32 threads) computes: 16×8 = 128 output elements
    Thread ownership: each thread holds 4 output elements (128/32=4)

    CUTLASS abstracts this into MmaPolicy template:
        using MmaPolicy = MmaTensorOp<GemmShape<16, 8, 16>, ...>;

### Fragment Layout — The Register Puzzle

    The trickiest part of Tensor Core programming: WHERE in registers does
    each thread hold which matrix elements?

    For wmma::m16n8k16 FP16:
        The 16×8 output C is distributed across 32 threads.
        Thread i holds 4 FP32 elements at positions that follow a specific
        pattern defined by the Tensor Core hardware.

    CUTLASS handles this distribution transparently:
        - IteratorA/IteratorB: load from Shared Memory into register fragments
          in the exact layout that MMA instructions expect
        - This avoids explicit register shuffle (SWIZZLE) operations
        - Getting this wrong = incorrect results or severe performance loss

### Supported Tensor Core Shapes by Architecture

    Volta (V100): FP16 input, FP32 accumulate
        wmma: m16n16k16, m32n8k16, m8n32k16
        ptx mma: m8n8k4 (fine-grained)

    Turing/Ampere: FP16, BF16, INT8, INT4, BIN (binary)
        FP16: m16n8k16 (primary), m16n8k8
        BF16: m16n8k16
        INT8: m16n8k32
        INT4: m16n8k64
        BIN:  m8n8k128

    Hopper (H100): FP8, BF16, FP16, INT8
        FP8: m64n128k32 (through WGMMA — warpgroup MMA, 4-warp groups)
        BF16: m64n128k16 via WGMMA
        The WGMMA instruction is H100-specific and dramatically larger:
          4 warps collaborate on one 64×128 output tile
          5× more output per warp than Ampere's m16n8k16

    CUTLASS 3.x specifically targets WGMMA (H100 GEMM):
        "CuTe" (CUDA Tensor Engine) — a new tensor layout library
        Gemm_Universal<...> with SM90 architecture tag


##### PART 4 — THE CUTLASS GEMM TEMPLATE HIERARCHY

### CUTLASS 2.x Template Structure

    A CUTLASS GEMM kernel is assembled from nested templates:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Gemm<>  (outer driver — problem dimensions, algorithm)             │
    │  └── GemmKernel<>  (CUDA kernel function)                           │
    │      ├── ThreadblockMma<>  (block-level accumulation loop)          │
    │      │   ├── MmaPipelined<>  (multi-stage pipeline)                 │
    │      │   │   ├── IteratorA<>  (load A tile from HBM → Shared Mem)   │
    │      │   │   ├── IteratorB<>  (load B tile from HBM → Shared Mem)   │
    │      │   │   └── MmaWarp<>    (warp-level MMA accumulation)         │
    │      │   │       └── MmaTensorOp<>  (Tensor Core instruction wrap)  │
    │      └── EpilogueWithBias<>  (post-GEMM computation)                │
    │          ├── EpilogueFunctor<>  (elementwise: bias, ReLU, ...)      │
    │          └── Iterator<>  (store output to HBM)                      │
    └─────────────────────────────────────────────────────────────────────┘

### Key Template Parameters

    In CUTLASS, every performance-critical parameter is a compile-time
    template argument (not a runtime argument). This enables the compiler
    to generate optimal code for each configuration.

    using GemmKernel = cutlass::gemm::kernel::DefaultGemm<
        cutlass::half_t,            // ElementA — FP16 input
        cutlass::layout::RowMajor,  // LayoutA  — row-major A
        8,                          // AlignmentA — 8 elements = 16 bytes
        cutlass::half_t,            // ElementB
        cutlass::layout::ColumnMajor,// LayoutB — col-major B (transposed)
        8,                          // AlignmentB
        float,                      // ElementC — FP32 accumulator
        cutlass::layout::RowMajor,  // LayoutC
        float,                      // ElementAccumulator
        cutlass::arch::OpClassTensorOp,  // Use Tensor Cores
        cutlass::arch::Sm80,        // Target: Ampere GPU
        cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape
        cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape (MMA tile)
        cutlass::epilogue::thread::LinearCombination<  // epilogue: D = α×C + β
            float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        3  // pipeline stages (N=3 for double buffering + prefetch)
    >;

    using Gemm = cutlass::gemm::device::Gemm<GemmKernel>;

### Tile Sizes and Their Trade-offs

    ThreadblockShape (M_block × N_block × K_block):

        128 × 128 × 32 (large tile, FP16 Ampere):
            Pros: High arithmetic intensity per data loaded
            Cons: Needs 128×128×2 + 128×128×2 + overhead = ~64 KB Shared Mem
                  Only ~24 threadblocks can be resident on an A100 (limited by SMEM)
                  Good for M=N=K≥1024

        64 × 128 × 32 (medium tile):
            Balance between occupancy and data reuse.
            ~48 threadblocks per A100. Good for M or N ≈ 256–1024.

        64 × 64 × 32 (small tile):
            High occupancy (~96 blocks per A100).
            Less data reuse per block.
            Good for small M or N (< 128), like decode-stage LLM matmuls.

        Choosing the right tile size is THE key to performance.
        CUTLASS provides a profiler (tools/profiler/) to auto-select.

### WarpShape and the Warp-Level Partition

    WarpShape (M_warp × N_warp × K_warp) divides the block tile among warps:

    If ThreadblockShape = 128×128×32 and WarpShape = 64×64×32:
        128×128 / 64×64 = 4 warps per block (2×2 warp grid)
        Each warp handles 64×64 = 4096 output elements of the block's 16384

    Warp shape constraints:
        M_warp must be divisible by InstructionShape.M (16)
        N_warp must be divisible by InstructionShape.N (8)
        WarpShape.M × WarpShape.N ≤ available Tensor Core throughput per warp


##### PART 5 — THE EPILOGUE: FUSING POST-GEMM OPERATIONS

### What the Epilogue Is

    After the Tensor Cores finish accumulating A×B:
        - Accumulated result is in FP32 registers (each thread holds its portion)
        - Must be written to HBM (global memory)
        - Before writing: apply linear combination, activations, bias, etc.

    The EPILOGUE is the post-GEMM computation that runs on this data
    BEFORE it is written to HBM. Since the data is already in registers,
    any epilogue computation is essentially FREE (no extra memory reads).

    Without epilogue fusion:
        1. Write FP32 result to HBM (slow)
        2. Read FP32 result back from HBM (slow)
        3. Apply activation function (fast)
        4. Write output to HBM (slow)
        → 3 HBM accesses for the output tensor

    With CUTLASS epilogue fusion:
        1. Apply activation in registers (fast — data already there)
        2. Write final FP16/BF16 output to HBM (once)
        → 1 HBM write — 3× less memory traffic

### Built-in Epilogue Functors

    CUTLASS provides these epilogue operations out of the box:

    LinearCombination:
        D = α × AccumC + β × C
        Standard GEMM output with scaling. Used by cuBLAS equivalents.

    LinearCombinationRelu:
        D = max(0, α × AccumC + β × C + bias)
        GEMM + bias + ReLU — classic linear layer with activation.

    LinearCombinationGelu:
        D = GELU(α × AccumC + β × C + bias)
        GEMM + bias + GELU — standard transformer MLP.

    LinearCombinationClamp:
        D = clamp(α × AccumC + β × C, min_val, max_val)
        Used for quantisation (clamp to INT8 range before round).

    LinearCombinationResidual:
        D = activation(α × AccumC + β × C) + skip
        GEMM + skip connection — residual block in one kernel.

    LinearCombinationWithAbsMax:
        D = α × AccumC + β × C, also computes max(|D|) for FP8 scaling.
        Used for dynamic FP8 quantisation.

### Custom Epilogue Functors

    For operations not in the built-in list, you write a custom functor:

        struct MyCustomEpilogue {
            using ElementOutput = cutlass::half_t;  // output dtype
            using ElementAccumulator = float;        // accumulator dtype
            using ElementCompute = float;            // compute dtype

            static const int kCount = 1;  // elements per thread per iteration

            CUTLASS_HOST_DEVICE
            ElementOutput operator()(ElementAccumulator accum,
                                     ElementOutput      source,
                                     ElementCompute     scale) {
                // Example: SwiGLU activation (from Llama FFN)
                // Assumes accum is gate, source is linear output
                float gate   = static_cast<float>(accum);
                float linear = static_cast<float>(source);
                float swish  = gate / (1.0f + expf(-gate));   // SiLU on gate
                return static_cast<ElementOutput>(swish * linear);
            }
        };

    This functor runs in registers at the end of the GEMM computation.
    The SwiGLU/SiLU activation used in LLaMA's FFN can be fused directly.

### Epilogue Memory Layout

    After the Tensor Core accumulation, the result lives in registers
    in the MMA fragment layout — a specific pattern where each thread
    holds certain rows/columns of the output tile.

    Before writing to HBM, CUTLASS transposes/rearranges data through
    Shared Memory to achieve coalesced writes:

    1. Each thread: write its FP32 accumulators to Shared Memory (SMEM)
       in a layout that will allow coalesced global store.
    2. After __syncthreads(): each thread: read from SMEM in a coalesced
       pattern and write to HBM.

    This double-buffering through SMEM is standard in all efficient GEMM
    epilogues — naive direct-from-register writes would be uncoalesced.


##### PART 6 — CUTLASS 3.x: CuTe AND WARPGROUP MMA (H100)

### CuTe (CUDA Tensor Engine)

    CUTLASS 3.x introduced CuTe — a complete rewrite of the tensor layout
    system using a more algebraic approach.

    CuTe represents tensor layouts as (shape, stride) pairs with hierarchical
    composition:
        auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
        // Describes M×K row-major matrix

        // Hierarchical composition:
        auto thread_layout = make_layout(make_shape(4, 8));     // 4×8 threads
        auto val_layout    = make_layout(make_shape(2, 4));     // 2×4 values per thread
        auto total_layout  = make_layout(thread_layout, val_layout);  // composed

    Benefits of CuTe's layout algebra:
        - Generate optimal code for any combination of shapes and strides
        - Automatic handling of banking, padding, and vectorised loads
        - More composable than CUTLASS 2.x's ad-hoc layout specialisations

### WGMMA (WarpGroup MMA) for H100

    H100's Hopper architecture introduces a new instruction: WGMMA
    (WarpGroup Matrix Multiply-Accumulate).

    WGMMA properties:
        - Operates on 4 warps simultaneously (a "warpgroup" of 128 threads)
        - Input matrix A: read from Shared Memory via hardware tensor map (TMA)
        - Input matrix B: read from Shared Memory via hardware tensor map (TMA)
        - Output: 64×128 = 8192 elements across 128 threads (64 per thread)
        - Supported dtypes: BF16, FP16, FP8-E4M3, FP8-E5M2

    TMA (Tensor Memory Accelerator):
        H100's dedicated hardware for Shared Memory ↔ HBM transfers.
        Decoupled from SM's computation units — no SM cycles spent on copy.
        Supports strided access, multicast (one HBM load → multiple SMs).
        CUTLASS 3.x uses TMA for all A/B tile loads in H100 kernels.

    The H100 GEMM pipeline (CUTLASS 3.x):
        1. TMA hardware asynchronously loads A, B tiles into SMEM
        2. SM computes WGMMA on tiles already in SMEM
        3. While WGMMA runs: TMA preloads NEXT A, B tiles
        4. Epilogue: apply activation, quantise, write to HBM

    This overlapping of TMA load with WGMMA compute is what pushes H100
    close to its 989 TFLOPS BF16 peak — computation almost never waits for data.

### CUTLASS Persistent Kernels (Hopper)

    Standard GEMM: each threadblock processes ONE tile then terminates.
    Overhead: CUDA kernel launch overhead per tile.

    Persistent kernel: threadblocks stay alive and process MULTIPLE tiles.
        Thread block 0: processes tile (0,0), then (4,0), then (8,0), ...
        Thread block 1: processes tile (0,1), then (4,1), then (8,1), ...
        All blocks process tiles from a shared work queue.

    Benefits:
        - Eliminates kernel launch overhead per tile
        - Enables load balancing (fast blocks take more tiles)
        - Allows global reduction patterns (e.g., layer norm after GEMM)

    CUTLASS 3.x implements persistent kernels via the "CollectiveMma" and
    "PersistentTileScheduler" components.


##### PART 7 — CUTLASS IN ML: FLASHATTENTION, TRITON, AND TENSORRT

### FlashAttention Implementation

    FlashAttention-2 uses CUTLASS for its core kernel because:
        1. The QK^T matmul and PV matmul must be fused with softmax in registers
        2. No existing library provided this fusion — custom kernel needed
        3. CUTLASS provides the Tensor Core infrastructure without reinventing it

    FlashAttention's CUTLASS usage:
        - MmaTensorOp<m16n8k16>: the inner Tensor Core instruction
        - Shared Memory: tiles of Q, K, V blocks loaded iteratively
        - Custom epilogue: online softmax computed in registers

    The online softmax algorithm (the key innovation):
        Standard: compute all QK^T scores, then softmax(QK^T), then × V
                  Requires materialising N×N matrix
        Online:   for each K tile, update running max and sum in registers
                  D[partial] += exp(QK_tile^T - running_max) × V_tile
                  Final: D = D[partial] / running_sum
                  Never materialises N×N matrix — only O(block_size) SMEM!

    FlashAttention-3 (H100) adds:
        - WGMMA for the QK^T and PV matmuls
        - TMA for loading Q, K, V tiles asynchronously
        - Pipelining TMA load with WGMMA compute (3 stages)
        - FP8 input support

### Triton and CUTLASS: Different Approaches to the Same Goal

    Triton (OpenAI):
        Write GPU kernels in a Python-like DSL.
        Triton compiler handles tiling, shared memory, vectorisation.
        MORE ERGONOMIC: no C++ templates, easier for ML researchers.
        Less flexible on fine-grained control (e.g., specific register layout).

    CUTLASS:
        Write kernels in C++ using template abstractions.
        Full control over every aspect of the kernel.
        MORE PERFORMANT: can squeeze the last few percent from the hardware.
        Harder to use: requires deep understanding of GPU architecture.

    In practice:
        Triton: favoured for research kernels, custom ops in PyTorch
        CUTLASS: favoured for production inference kernels, TensorRT

    Many operations in TensorRT that are not in cuDNN are implemented using
    auto-generated CUTLASS kernels — TensorRT's profiling engine tries
    multiple CUTLASS tile configurations and selects the fastest.

### CUTLASS Profiler and Auto-Tuning

    CUTLASS ships with a command-line profiler:
        ./tools/profiler/cutlass_profiler \
            --operation=Gemm \
            --A=f16:row --B=f16:column --C=f32:row \
            --m=4096 --n=4096 --k=4096

    This benchmarks ALL CUTLASS GEMM implementations for the given shape:
        - All ThreadblockShape combinations: 64×64, 64×128, 128×128, 256×128
        - All pipeline depths: 2, 3, 4
        - All warp partitions
        - Reports latency, GFLOP/s, efficiency vs theoretical peak

    Output example:
        Op: cutlass_simt_sgemm_128x128_8x2_nn
            Runtime: 0.452ms, 149.8 GFLOP/s

        Op: cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nn
            Runtime: 0.127ms, 532.6 GFLOP/s  ← best for this shape

    TensorRT uses this profiling approach internally to generate the fastest
    kernel for each operation in the network, at each possible batch size.


##### PART 8 — WRITING AND USING CUTLASS KERNELS

### Installing CUTLASS

    CUTLASS is header-only — no compilation of the library itself:
        git clone https://github.com/NVIDIA/cutlass.git
        # headers are in: cutlass/include/

    To use in your project:
        nvcc your_kernel.cu \
             -I /path/to/cutlass/include \
             -I /path/to/cutlass/tools/util/include \
             -arch=sm_80 \              # target Ampere (A100)
             -std=c++17 \
             -O3

    CUTLASS Python bindings (for prototyping):
        pip install nvidia-cutlass
        import cutlass
        # Python API wraps C++ kernels — easier interface for experimentation

### A Complete CUTLASS GEMM Example (C++ pseudocode)

    #include <cutlass/gemm/device/gemm.h>

    // Define the GEMM kernel type (all parameters at compile time)
    using ElementA  = cutlass::half_t;        // FP16 input A
    using ElementB  = cutlass::half_t;        // FP16 input B
    using ElementC  = float;                   // FP32 accumulator
    using ElementD  = cutlass::bfloat16_t;    // BF16 output

    using GemmOp = cutlass::gemm::device::Gemm<
        ElementA,   cutlass::layout::RowMajor,
        ElementB,   cutlass::layout::ColumnMajor,
        ElementC,   cutlass::layout::RowMajor,
        ElementC,   cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape
        cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape
        cutlass::epilogue::thread::LinearCombinationRelu<ElementD, 4, ElementC, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        3     // pipeline stages
    >;

    // Set up operation arguments
    GemmOp::Arguments args(
        {M, N, K},           // problem size
        {A_ptr, ldA},        // A matrix (device pointer + leading dimension)
        {B_ptr, ldB},        // B matrix
        {C_ptr, ldC},        // C matrix (bias / source for β)
        {D_ptr, ldD},        // D matrix (output)
        {alpha, beta}        // linear combination scalars
    );

    GemmOp gemm_op;
    cutlass::Status status;

    // Optional: query workspace size (some variants need scratch space)
    size_t workspace_size = GemmOp::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Initialize (validates arguments, computes grid size, etc.)
    status = gemm_op.initialize(args, workspace.get());
    CUTLASS_CHECK(status);

    // Run the kernel
    status = gemm_op.run();    // launches CUDA kernel
    CUTLASS_CHECK(status);

    cudaDeviceSynchronize();   // wait for completion

### Common CUTLASS Pitfalls

    1. ARCHITECTURE MISMATCH:
        Compiling with -arch=sm_75 (Turing) but using Sm80 in template → crash.
        Always match template architecture to compile flag.

    2. ALIGNMENT REQUIREMENTS:
        AlignmentA=8 means A pointer must be 16-byte aligned (8 × 2 bytes).
        Misaligned pointer → undefined behaviour or significant slowdown.
        Check: (ptr % 16) == 0 before calling CUTLASS.

    3. TILE SIZE vs PROBLEM SIZE:
        If M < ThreadblockShape.M: only one block row, poor occupancy.
        CUTLASS handles partial tiles correctly but performance degrades.
        Solution: pad M,N to multiples of ThreadblockShape for best performance.

    4. PIPELINE DEPTH vs SHARED MEMORY:
        stages=4 needs ~4× more shared memory than stages=2.
        If shared memory exceeds per-SM limit (~192 KB on A100):
        kernel launch will silently fail (cudaErrorLaunchFailure).
        Always check: CUTLASS_CHECK(gemm_op.initialize(args, workspace));

    5. ACCUMULATOR OVERFLOW:
        With FP16 accumulation and large K: values may overflow FP16 range.
        Use ElementAccumulator=float for correct results with large matrices.
        FP32 accumulators use slightly more registers but avoid overflow.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · GEMM Tiling — Understanding the Compute-Memory Hierarchy": {
        "description": (
            "Build intuition for CUTLASS's tiling strategy from first principles. "
            "Implement a simplified matmul with increasing tile sizes and measure "
            "the effect on cache efficiency. Compute arithmetic intensity for each "
            "tile configuration. Show how shared memory reuse reduces HBM traffic. "
            "Demonstrate the relationship between tile size, occupancy, and throughput. "
            "Profile with PyTorch CUDA events to validate the theory."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import math

print("=" * 65)
print("  GEMM TILING — COMPUTE-MEMORY HIERARCHY")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Tiling theory — shared memory reuse
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Tiling: how shared memory reduces HBM traffic")
print("━" * 65)
print()

def analyse_tiling(M, N, K, tile_m, tile_n, tile_k, bytes_per_elem=2):
    """
    Compute HBM traffic for tiled matrix multiply D = A × B.
    A: M×K  B: K×N  D: M×N

    Without tiling (naive):
        Each element of A is read N times (once per output column block)
        Each element of B is read M times (once per output row block)

    With tiling (ThreadblockShape = tile_m × tile_n × tile_k):
        Each threadblock reads:
            A tile: tile_m × tile_k × (N / tile_n) times  (once per K partition)
            B tile: tile_k × tile_n × (M / tile_m) times  (once per K partition)
    """
    # Without tiling: naive
    hbm_A_naive = M * K * bytes_per_elem * (N / 1)   # read A for every output col
    hbm_B_naive = K * N * bytes_per_elem * (M / 1)   # read B for every output row
    hbm_naive   = hbm_A_naive + hbm_B_naive + M * N * bytes_per_elem

    # With tiling
    n_blocks_m   = math.ceil(M / tile_m)
    n_blocks_n   = math.ceil(N / tile_n)
    n_blocks_k   = math.ceil(K / tile_k)

    # Each A tile loaded once per (block_m, block_k) combination
    hbm_A_tiled  = n_blocks_m * n_blocks_k * (tile_m * tile_k) * bytes_per_elem
    # Actually each A tile is shared across N/tile_n output blocks → reuse!
    # Correct: A tile loaded n_blocks_n times for each (m, k) pair (once per n_block)
    # But the threadblock model: each (m,n) block loads its own A[:,k] slices
    # → A is read n_blocks_n times (once per n_block, for each m_block, k_block)
    hbm_A_tiled  = n_blocks_m * n_blocks_n * n_blocks_k * tile_m * tile_k * bytes_per_elem
    hbm_B_tiled  = n_blocks_m * n_blocks_n * n_blocks_k * tile_k * tile_n * bytes_per_elem
    hbm_D_tiled  = M * N * bytes_per_elem
    hbm_tiled    = hbm_A_tiled + hbm_B_tiled + hbm_D_tiled

    # Shared memory: tiles fit in SMEM → reused over k-loop within block
    smem_A = tile_m * tile_k * bytes_per_elem
    smem_B = tile_k * tile_n * bytes_per_elem
    smem_total = smem_A + smem_B

    # Arithmetic intensity (FLOPs / bytes from HBM)
    flops = 2 * M * N * K
    ai_naive = flops / hbm_naive
    ai_tiled = flops / hbm_tiled

    # Within-block reuse: each A element reused tile_n times, B element tile_m times
    reuse_A = tile_n   # each A element used once per output column in the block
    reuse_B = tile_m   # each B element used once per output row in the block

    return {
        "hbm_naive_MB":  hbm_naive / 1e6,
        "hbm_tiled_MB":  hbm_tiled / 1e6,
        "ai_naive":      ai_naive,
        "ai_tiled":      ai_tiled,
        "smem_KB":       smem_total / 1024,
        "reuse_A":       reuse_A,
        "reuse_B":       reuse_B,
    }


M, N, K = 4096, 4096, 4096
print(f"  Matrix multiply: A({M}×{K}) @ B({K}×{N}) = D({M}×{N}), BF16")
print(f"  A100 ridge point: 156 FLOP/byte (BF16)")
print()
print(f"  {'Tile (M×N×K)':>16} | {'HBM (MB)':>10} | {'AI (F/B)':>10} | "
      f"{'SMEM (KB)':>11} | {'Regime'}")
print(f"  {'─'*68}")

tile_configs = [
    (1,   1,   1,   "No tiling (naive)"),
    (16,  16,  16,  "Small tile (16³)"),
    (32,  32,  32,  "Medium tile (32³)"),
    (64,  64,  32,  "Good tile (64×64×32)"),
    (128, 128, 32,  "Standard (128×128×32)"),
    (256, 128, 32,  "Large (256×128×32)"),
]

for tm, tn, tk, label in tile_configs:
    r = analyse_tiling(M, N, K, tm, tn, tk)
    regime = "COMPUTE ✅" if r["ai_tiled"] >= 156 else "MEMORY  ⚠️ "
    print(f"  {str(tm)+'×'+str(tn)+'×'+str(tk):>16} | "
          f"{r['hbm_tiled_MB']:>10.1f} | {r['ai_tiled']:>10.2f} | "
          f"{r['smem_KB']:>11.1f} | {regime}")

print()
print("  Key observations:")
print("  1. Larger tiles → fewer HBM accesses (each element reused more)")
print("  2. Larger tiles → more shared memory needed (limited to ~192 KB)")
print("  3. 128×128×32 tile crosses the ridge point → compute bound!")
print("  4. Very large tiles (256×128) may exceed shared memory capacity")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Occupancy analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Occupancy: tile size vs SM residency")
print("━" * 65)
print()

print("  A100 SM resources:")
print("    Shared memory per SM:    192 KB (configurable)")
print("    Max threadblocks per SM: 32 (hardware limit)")
print("    Max threads per SM:      2048")
print()

A100_SMEM_PER_SM = 192 * 1024   # bytes
A100_MAX_BLOCKS  = 32
A100_MAX_THREADS = 2048
THREADS_PER_BLOCK = 128  # standard for 128×128 tile (4 warps × 32 threads)

print(f"  {'Tile (M×N×K)':>18} | {'SMEM/block (KB)':>17} | {'Blocks/SM':>11} | "
      f"{'Threads/SM':>12} | {'Occupancy':>11}")
print(f"  {'─'*80}")

for tm, tn, tk, label in tile_configs[1:]:   # skip no-tiling
    smem_a = tm * tk * 2   # BF16 = 2 bytes
    smem_b = tk * tn * 2
    smem_total = smem_a + smem_b

    # How many blocks fit based on shared memory?
    blocks_by_smem    = A100_SMEM_PER_SM // smem_total
    blocks_by_threads = A100_MAX_THREADS // THREADS_PER_BLOCK
    blocks_actual     = min(blocks_by_smem, A100_MAX_BLOCKS, blocks_by_threads)

    threads_total = blocks_actual * THREADS_PER_BLOCK
    occupancy_pct = threads_total / A100_MAX_THREADS * 100

    status = "✅ Good" if occupancy_pct >= 50 else "⚠️  Low"
    print(f"  {label:>18} | {smem_total/1024:>17.1f} | {blocks_actual:>11d} | "
          f"{threads_total:>12d} | {occupancy_pct:>9.0f}% {status}")

print()
print("  High occupancy (>50%) hides memory latency via warp switching.")
print("  Low occupancy (< 25%): GPU stalls when all resident warps wait for memory.")
print("  Best: choose the LARGEST tile that still achieves >50% occupancy.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Live matmul benchmark — tile effect via PyTorch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Live GEMM benchmark across problem sizes")
print("━" * 65)
print()

try:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print()

    if device == "cuda":
        torch.backends.cudnn.benchmark    = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Benchmark matrix multiply across different sizes
    # This exercises cuBLAS (which uses CUTLASS-like kernels internally)
    DTYPE = torch.bfloat16

    problem_sizes = [
        (128,  128,  4096,  "Decode-style (small M×N)"),
        (512,  512,  4096,  "Medium batch"),
        (1024, 1024, 4096,  "Moderate"),
        (2048, 2048, 4096,  "LLM training batch"),
        (4096, 4096, 4096,  "Large square"),
    ]

    REPS = 100

    if device == "cuda":
        print(f"  {'Problem (M×N×K)':>25} | {'Time (ms)':>12} | {'TFLOPS':>10} | "
              f"{'% of peak':>11} | {'Notes'}")
        print(f"  {'─'*75}")
        A100_PEAK_BF16 = 312.0   # TFLOPS

        for M, N, K, label in problem_sizes:
            A = torch.randn(M, K, device='cuda', dtype=DTYPE)
            B = torch.randn(K, N, device='cuda', dtype=DTYPE)

            # Warmup
            for _ in range(10): _ = torch.matmul(A, B)
            torch.cuda.synchronize()

            # Benchmark with CUDA events
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(REPS): torch.matmul(A, B)
            end_ev.record()
            torch.cuda.synchronize()
            t_ms = start_ev.elapsed_time(end_ev) / REPS

            flops    = 2 * M * N * K
            tflops   = flops / (t_ms / 1000) / 1e12
            pct_peak = tflops / A100_PEAK_BF16 * 100
            print(f"  {str(M)+'×'+str(N)+'×'+str(K):>25} | {t_ms:>12.4f} | "
                  f"{tflops:>10.2f} | {pct_peak:>10.1f}% | {label}")

        print()
        print("  % of peak tells you how close to Tensor Core maximum.")
        print("  Small M×N (decode-style): low % — memory bound, not compute bound.")
        print("  Large M×N×K: high % — cuBLAS selects 128×128×32 tiles → compute bound.")

    else:
        # CPU demonstration
        print("  GEMM scaling with problem size (CPU):")
        print(f"  {'Problem (M×N×K)':>25} | {'Time (ms)':>12} | {'GFLOPS':>10}")
        print(f"  {'─'*52}")

        for M, N, K, label in problem_sizes[:3]:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(5): np.dot(A, B)
            t_ms = (time.perf_counter() - t0) / 5 * 1000
            gflops = 2 * M * N * K / (t_ms / 1000) / 1e9
            print(f"  {str(M)+'×'+str(N)+'×'+str(K):>25} | {t_ms:>12.3f} | {gflops:>10.2f}")

except ImportError:
    print("  PyTorch not available for live benchmark")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: CUTLASS tile size selection guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — CUTLASS tile selection guide")
print("━" * 65)
print()

TILE_GUIDE = """
  CUTLASS TILE SIZE SELECTION RULES:

  Problem characteristic → Recommended ThreadblockShape

  ┌────────────────────────────────────────────────────────────────────┐
  │ Scenario                      │ Tile (M×N×K)  │ Reason            │
  ├────────────────────────────────────────────────────────────────────┤
  │ Large square (M=N=K≥2048)    │ 128×128×32    │ Max reuse, compute │
  │ Large non-square (M>>N)      │ 256×64×32     │ Balance M-heavy    │
  │ Large non-square (N>>M)      │ 64×256×32     │ Balance N-heavy    │
  │ LLM decode (M=batch×1 small) │ 64×64×32      │ High occupancy     │
  │ LLM prefill (M=seq_len)      │ 128×128×32    │ Long sequences     │
  │ INT8 inference               │ 128×256×64    │ INT8 tiles wider K │
  │ FP8 (H100)                   │ 128×256×64    │ WGMMA is large     │
  │ Small batch (M<32)           │ 32×128×32     │ Avoid tile waste   │
  └────────────────────────────────────────────────────────────────────┘

  PIPELINE STAGES (depth):
  stages=2: minimal shared memory, limited latency hiding
  stages=3: standard for Ampere (A100), good balance
  stages=4: more latency hiding, needs 4/3× more shared memory
  stages=5+: rare, only if k-loop is very long and compute is fast

  SWIZZLE:
  GemmIdentityThreadblockSwizzle<1>: default, no swizzle
  GemmIdentityThreadblockSwizzle<8>: 8 adjacent blocks form an L-shape
  Use swizzle=8 for large M×N (better L2 cache reuse between threadblocks)

  ALIGNMENT:
  AlignmentA=8, AlignmentB=8: 128-bit loads (FP16, 8 elements × 2 bytes)
  AlignmentA=16, AlignmentB=16: for INT8 (16 elements × 1 byte)
  Misalignment → falls back to scalar loads (8× slower)

  PROFILING WORKFLOW:
  1. Use CUTLASS profiler to find the best tile for your shape:
     ./cutlass_profiler --operation=Gemm --m=M --n=N --k=K --A=f16 --B=f16 --C=f32

  2. Hard-code that tile in your CUTLASS template (compile-time parameter)

  3. Re-profile if your problem size distribution changes

  RULE OF THUMB:
  Start with 128×128×32, pipeline=3.
  If M or N < 128: reduce that dimension's tile size.
  If occupancy < 25%: reduce both M and N tile dimensions.
  Run CUTLASS profiler to confirm — theory is a starting point.
"""
print(TILE_GUIDE)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Epilogue Fusion — Custom Post-GEMM Operations in CUTLASS": {
        "description": (
            "The CUTLASS epilogue: fusing post-GEMM operations into the kernel. "
            "Implement bias + activation fusion manually and show the memory savings. "
            "Show how CUTLASS epilogue functors enable custom operations. "
            "Demonstrate common ML epilogues: bias+ReLU, bias+GELU, residual. "
            "Compare performance of fused vs unfused approaches. "
            "CUTLASS Python API example for epilogue experimentation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  EPILOGUE FUSION — CUSTOM POST-GEMM OPERATIONS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Epilogue theory and memory savings
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — What the epilogue is and why it matters")
print("━" * 65)
print()

EPILOGUE_THEORY = """
  THE CUTLASS EPILOGUE:

  After Tensor Cores finish computing A×B, the result lives in registers.
  The epilogue processes this result BEFORE writing to HBM.

  ┌────────────────────────────────────────────────────────────────────┐
  │  Standard GEMM + separate bias + ReLU (3 memory operations):       │
  │                                                                    │
  │  [GEMM kernel] → write FP32 to HBM (slow!)                         │
  │  [Bias kernel] → read FP32 from HBM, add bias, write (slow!)       │
  │  [ReLU kernel] → read FP32 from HBM, apply ReLU, write (slow!)     │
  │                                                                    │
  │  HBM traffic for M×N output matrix (FP32):                         │
  │    Write from GEMM:  M×N×4 bytes                                   │
  │    Read for bias:    M×N×4 bytes                                   │
  │    Write after bias: M×N×4 bytes                                   │
  │    Read for ReLU:    M×N×4 bytes                                   │
  │    Write after ReLU: M×N×2 bytes (FP16 output)                     │
  │    TOTAL: ~18 × M × N bytes of HBM traffic                         │
  └────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │  CUTLASS fused GEMM + bias + ReLU (1 memory operation):            │
  │                                                                    │
  │  [Fused GEMM+Bias+ReLU kernel]:                                    │
  │    In registers: accumulate A×B (Tensor Cores)                     │
  │    In registers: add bias[n] (no HBM read — bias in L2)            │
  │    In registers: apply ReLU(x) = max(0, x)                         │
  │    In SMEM: rearrange to coalesced layout                          │
  │    Write FP16 output to HBM: M×N×2 bytes                           │
  │                                                                    │
  │  HBM traffic: ~2 × M × N bytes  (9× reduction!)                    │
  └────────────────────────────────────────────────────────────────────┘
"""
print(EPILOGUE_THEORY)

# Compute the savings
def epilogue_traffic(M, N, bpe_in=4, bpe_out=2):
    """HBM traffic for unfused GEMM + Bias + ReLU."""
    write_gemm   = M * N * bpe_in     # write GEMM output
    read_bias    = M * N * bpe_in     # read for bias kernel
    write_bias   = M * N * bpe_in     # write after bias
    read_relu    = M * N * bpe_in     # read for relu kernel
    write_relu   = M * N * bpe_out    # write final (FP16)
    total        = write_gemm + read_bias + write_bias + read_relu + write_relu
    return total

def fused_traffic(M, N, bpe_out=2):
    """HBM traffic for fused GEMM + Bias + ReLU (CUTLASS epilogue)."""
    # Only writes output + reads bias (once, cached in L2)
    write_output = M * N * bpe_out
    read_bias    = N * 2   # just the bias vector — tiny (fits in L2)
    return write_output + read_bias

print("  Memory traffic comparison (BF16 output):")
print()
print(f"  {'M × N':>12} | {'Unfused (MB)':>14} | {'Fused (MB)':>12} | {'Reduction':>10} | "
      f"{'A100 BW time':>14}")
print(f"  {'─'*72}")
A100_BW = 2000e9   # bytes/second

for M, N in [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]:
    unfused = epilogue_traffic(M, N)
    fused   = fused_traffic(M, N)
    reduction = unfused / fused
    time_saved_us = (unfused - fused) / A100_BW * 1e6
    print(f"  {str(M)+'×'+str(N):>12} | {unfused/1e6:>14.1f} | {fused/1e6:>12.1f} | "
          f"{reduction:>9.1f}× | {time_saved_us:>12.1f} μs saved")

print()
print("  At 4096×4096: unfused wastes ~540 ms of A100 bandwidth per second.")
print("  CUTLASS epilogue fusion recovers this computation time for free.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Epilogue functors — what cuDNN/TensorRT implement in CUTLASS
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Epilogue functors: the ML operations in CUTLASS")
print("━" * 65)
print()

EPILOGUE_FUNCTORS = """
  CUTLASS BUILT-IN EPILOGUE FUNCTORS:

  LinearCombination (standard GEMM):
    D = α × A×B + β × C
    template:
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,   // output element type (e.g., half_t)
        kCount,          // elements per thread (alignment)
        ElementAccumulator,  // FP32 accumulator
        ElementCompute   // compute type for α, β
    >

  LinearCombinationRelu (most common ML epilogue):
    D = max(0, α × A×B + β × bias)
    Code:
    cutlass::epilogue::thread::LinearCombinationRelu<
        cutlass::half_t, 8, float, float,
        cutlass::epilogue::thread::ScaleType::Default
    >
    Use: linear layers in CNNs (Conv + bias + ReLU in one kernel)

  LinearCombinationGelu (transformer MLP epilogue):
    D = GELU(α × A×B + β × bias)
    Use: transformer FFN first linear layer
    Note: GELU is more expensive than ReLU (exp), but still
    done in registers — no extra HBM traffic.

  LinearCombinationSigmoid:
    D = sigmoid(α × A×B + β × C)
    Use: gating mechanisms (Gated Linear Units)

  LinearCombinationClamp (quantisation preparation):
    D = clamp(α × A×B + β × C, min, max)
    Use: GEMM result clamped to INT8 range [-128, 127] before
    type conversion to INT8 for the next quantised layer.

  LinearCombinationWithAbsMax (dynamic FP8 scaling):
    D = α × A×B + β × C
    Also computes: max_abs = max(|D|)  [used to compute FP8 scale]
    Use: H100 dynamic FP8 quantisation between layers

  LinearCombinationResidual (skip connections):
    D = activation(α × A×B + β × C) + skip
    Use: transformer residual blocks — add skip connection in epilogue

  CUSTOM EPILOGUE (write your own):
  struct SwiGLUEpilogue {
    // SwiGLU: output = SiLU(gate) × linear
    // Requires two inputs: gate (from A×B) and linear (from another GEMM)
    CUTLASS_HOST_DEVICE
    ElementOutput operator()(
        ElementAccumulator gate,     // from current GEMM
        ElementOutput      linear,   // from second GEMM (passed as C input)
        ElementCompute     scale
    ) {
        float g = (float)gate;
        float l = (float)linear;
        float silu_g = g * (1.0f / (1.0f + expf(-g)));  // SiLU
        return (ElementOutput)(silu_g * l);
    }
  };
  Use: LLaMA, Mistral FFN: gate_proj × up_proj (SwiGLU)
       Gate and up computed as two GEMMs, merged in SwiGLU epilogue
"""
print(EPILOGUE_FUNCTORS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Live fusion benchmark
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Live fusion benchmark: fused vs unfused ops")
print("━" * 65)
print()

try:
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print()

    M, N, K = 2048, 2048, 4096
    DTYPE   = torch.bfloat16
    REPS    = 200

    A    = torch.randn(M, K, device=device, dtype=DTYPE)
    W    = torch.randn(N, K, device=device, dtype=DTYPE)
    bias = torch.randn(N,   device=device, dtype=DTYPE)

    def bench(fn, warmup=20, reps=REPS):
        for _ in range(warmup): fn()
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        if device == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    # Unfused: separate operations
    def unfused_linear_relu():
        x = torch.nn.functional.linear(A, W, bias)   # GEMM + bias
        return torch.relu(x)                           # ReLU

    def unfused_linear_gelu():
        x = torch.nn.functional.linear(A, W, bias)
        return torch.nn.functional.gelu(x)

    # Fused: using nn.Linear which has a more optimised path
    linear_layer = nn.Linear(K, N, bias=True).to(device).to(DTYPE)
    linear_layer.weight.data = W
    linear_layer.bias.data   = bias

    # torch.compile fuses these into CUTLASS-style kernels
    @torch.compile
    def compiled_linear_relu(x):
        return torch.relu(torch.nn.functional.linear(x, W, bias))

    @torch.compile
    def compiled_linear_gelu(x):
        return torch.nn.functional.gelu(torch.nn.functional.linear(x, W, bias))

    # Warmup compile
    if device == "cuda":
        _ = compiled_linear_relu(A)
        _ = compiled_linear_gelu(A)
        torch.cuda.synchronize()

    t_unfused_relu = bench(unfused_linear_relu)
    t_unfused_gelu = bench(unfused_linear_gelu)

    if device == "cuda":
        t_compiled_relu = bench(lambda: compiled_linear_relu(A))
        t_compiled_gelu = bench(lambda: compiled_linear_gelu(A))

    print(f"  Linear({K}→{N}) + activation, M={M}, BF16:")
    print()
    print(f"  {'Operation':<30} | {'Unfused (ms)':>14} | "
          f"{'Compiled (ms)':>15} | {'Speedup':>9}")
    print(f"  {'─'*73}")

    if device == "cuda":
        pairs = [
            ("Linear + ReLU",  t_unfused_relu, t_compiled_relu),
            ("Linear + GELU",  t_unfused_gelu, t_compiled_gelu),
        ]
        for name, t_u, t_c in pairs:
            print(f"  {name:<30} | {t_u:>14.4f} | {t_c:>15.4f} | "
                  f"{t_u/t_c:>9.2f}×")
    else:
        for name, t_u in [("Linear + ReLU", t_unfused_relu),
                           ("Linear + GELU", t_unfused_gelu)]:
            print(f"  {name:<30} | {t_u:>14.4f} | {'N/A (CPU)':>15} | {'N/A':>9}")

    print()
    print("  torch.compile generates TorchInductor kernels that use")
    print("  CUTLASS-style epilogue fusion internally.")
    print("  On GPU: fusion typically saves 15–40% vs separate ops.")
    print()

except ImportError:
    print("  PyTorch not available for live benchmark")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: CUTLASS Python API reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — CUTLASS Python API: experimentation interface")
print("━" * 65)
print()

PYTHON_API = """
  CUTLASS 3.x Python API (nvidia-cutlass package):
  Install: pip install nvidia-cutlass

  ── Basic GEMM ──────────────────────────────────────────────────────────

  import cutlass, numpy as np

  # Define element types
  plan = cutlass.op.Gemm(
      element_A=np.float16,
      element_B=np.float16,
      element_C=np.float16,
      element_D=np.float16,
      element_accumulator=np.float32,
  )

  # Optionally request specific tile size
  plan.tile_description = plan.tile_descriptions()[0]  # auto-select
  # or:
  plan.tile_description = cutlass.TileDescription(
      threadblock_shape=[128, 128, 32],
      warp_count=[2, 2, 1],  # 4 warps
      stages=3,
  )

  # Create and run GEMM
  A = cutlass.emit_device_tensor(np.random.randn(M, K).astype(np.float16))
  B = cutlass.emit_device_tensor(np.random.randn(K, N).astype(np.float16))
  C = cutlass.emit_device_tensor(np.zeros((M, N), dtype=np.float16))
  D = cutlass.emit_device_tensor(np.zeros((M, N), dtype=np.float16))

  plan.run(A, B, C, D, alpha=1.0, beta=0.0)

  ── Custom epilogue ──────────────────────────────────────────────────────

  import cutlass

  # CUTLASS Python generates C++ via Jinja templates
  plan = cutlass.op.Gemm(
      element=np.float16,
      element_accumulator=np.float32,
  )

  # Set epilogue to ReLU
  plan.activation = cutlass.epilogue.relu

  # Set epilogue to GELU
  plan.activation = cutlass.epilogue.gelu

  # Custom epilogue via Python:
  class SiLUActivation(cutlass.swizzle.EpilogueFunctor):
      @staticmethod
      def tag():
          return "SiLU"
      @staticmethod
      def emit():
          # Emits C++ code for the epilogue functor
          return "x * (1.0f / (1.0f + expf(-x)))"

  plan.activation = SiLUActivation

  ── Profiler ────────────────────────────────────────────────────────────
  # Find best tile for your problem:
  candidates = plan.tile_descriptions()
  for desc in candidates[:5]:
      plan.tile_description = desc
      result = plan.profile(
          m=4096, n=4096, k=4096,
          warmup_iterations=10, profiling_iterations=100,
      )
      print(f"  {desc.threadblock_shape}: {result.runtime_ms:.3f}ms, "
            f"{result.gflops:.1f} GFLOPS")

  ── C++ Kernel Generation ────────────────────────────────────────────────
  # CUTLASS Python can emit the C++ kernel for inspection:
  kernel_code = plan.to_cpp()
  print(kernel_code)   # Shows the full CUTLASS template instantiation

  # This is how TensorRT uses CUTLASS:
  # It generates C++ kernel templates, compiles with nvcc, profiles them,
  # and uses the fastest one for production inference.
"""
print(PYTHON_API)

try:
    import cutlass as cutlass_py
    print(f"  CUTLASS Python package available: v{cutlass_py.__version__} ✅")
    print()
    print("  Listing available tile configurations for FP16 GEMM:")
    plan = cutlass_py.op.Gemm(element_A=np.float16, element_B=np.float16,
                               element_C=np.float16, element_D=np.float16)
    descriptions = plan.tile_descriptions()
    for i, desc in enumerate(descriptions[:5]):
        print(f"    [{i}] Tile: {desc.threadblock_shape}, "
              f"Warp: {desc.warp_count}, Stages: {desc.stages}")
except (ImportError, Exception):
    print("  CUTLASS Python: pip install nvidia-cutlass  (requires CUDA GPU)")
    print()

print()
print("━" * 65)
print("  CUTLASS vs cuDNN vs cuBLAS: DECISION GUIDE")
print("━" * 65)
print()
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ Use cuBLAS when:                                             │")
print("  │   • Standard GEMM: D = α×A×B + β×C                           │")
print("  │   • Shape is large and standard                              │")
print("  │   • No custom epilogue needed                                │")
print("  │   • Maximum compatibility required                           │")
print("  │                                                              │")
print("  │ Use cuDNN when:                                              │")
print("  │   • Convolution, BatchNorm, Attention, RNN operations        │")
print("  │   • Using PyTorch/TF/JAX (they call cuDNN for you)           │")
print("  │   • Black-box approach is acceptable                         │")
print("  │                                                              │")
print("  │ Use CUTLASS when:                                            │")
print("  │   • Custom epilogue (SwiGLU, quantisation, residual fusion)  │")
print("  │   • Non-standard data types (INT4, FP8, custom)              │")
print("  │   • Specific tile sizes (batch decode, small M/N)            │")
print("  │   • Custom attention variants (sparse, sliding window)       │")
print("  │   • Need maximum performance beyond what cuDNN/cuBLAS gives  │")
print("  │   • Writing TensorRT custom plugins                          │")
print("  └──────────────────────────────────────────────────────────────┘")
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
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }