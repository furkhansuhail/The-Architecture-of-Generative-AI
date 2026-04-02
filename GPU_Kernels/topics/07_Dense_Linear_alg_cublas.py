"""
cuBLAS — SGEMM / DGEMM, Batched GEMM & cublasLt Tensor Core Path
==================================================================

Matrix multiplication is the engine of deep learning. Every linear
layer, every attention projection, every FFN weight application is a
GEMM — General Matrix-Matrix Multiplication. In a 7B-parameter LLM
inference run, more than 95% of FLOPs are GEMM operations. Optimising
GEMM is not a micro-optimisation; it is the primary performance lever
for the entire field.

cuBLAS (CUDA Basic Linear Algebra Subroutines) is NVIDIA's hand-tuned
BLAS library. It ships with every CUDA toolkit and provides:
    - SGEMM/DGEMM: FP32 and FP64 matrix multiplication
    - HGEMM: FP16 matrix multiplication (Volta+)
    - Batched GEMM: multiple independent matrix products in one call
    - Strided Batched GEMM: matrices stored as a single 3D tensor
    - cublasLt: a lower-level API exposing tensor core paths, epilogue
      fusion, workspace control, and algorithm search

Understanding cuBLAS deeply means understanding three things:

    1. THE GEMM CONTRACT — the mathematical operation, the leading
       dimension abstraction, and how transposes and strides map to
       memory layouts.

    2. THE BATCHED GEMM VARIANTS — pointer arrays vs strided batch,
       when to use each, and how attention's Q@K and scores@V operations
       map exactly to strided batched GEMM calls.

    3. cublasLt — the tensor core API: epilogue fusions (bias + ReLU
       in one pass), algorithm search (heuristic vs exhaustive), the
       matmul descriptor system, and the workspace budget that unlocks
       faster algorithms.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "cuBLAS — SGEMM / DGEMM, Batched GEMM & cublasLt Tensor Core Path"
DISPLAY_NAME = "07 · cuBLAS"
ICON         = "⚡"
SUBTITLE     = "SGEMM · DGEMM · Batched GEMM · cublasLt · Tensor Cores · Epilogue Fusion"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE GEMM CONTRACT: MATHEMATICS, MEMORY, AND THE LEADING DIMENSION

### The Mathematical Operation

    GEMM computes:
        C = α × op(A) × op(B) + β × C

    where:
        A   is an M×K matrix   (or K×M, depending on op(A))
        B   is a K×N matrix   (or N×K, depending on op(B))
        C   is an M×N matrix
        α, β are scalar multipliers
        op(X) is X (CUBLAS_OP_N) or Xᵀ (CUBLAS_OP_T) or X* (CUBLAS_OP_C)

    The dimensions M, N, K are called the problem dimensions:
        M = number of rows in op(A) and C
        N = number of columns in op(B) and C
        K = inner (contraction) dimension

    FLOP count:   2 × M × N × K  (K multiply-adds per output element)
    Memory:       (M×K + K×N + M×N) × sizeof(dtype) bytes read/written
    Arithmetic intensity: 2MNK / (MK + KN + MN) × sizeof(dtype)
        → For large square matrices: AI ≈ M/2  (FLOPs/Byte)
        → This grows with M, which is why large batch sizes are efficient.

### The Leading Dimension (lda, ldb, ldc)

    cuBLAS stores matrices in COLUMN-MAJOR order (Fortran convention).
    The "leading dimension" (ld) is the stride between consecutive columns.

    For a matrix with M rows stored in column-major layout:
        Element A[row, col] is at memory address: A_ptr + col * lda + row

    Why does lda matter? Because matrices can be SUBMATRICES of larger
    allocations. If you have a 1024×1024 buffer but only want to GEMM on
    the top-left 512×512 block:
        lda = 1024  (the stride of the full allocation)
        M   = 512   (only 512 rows participate)
        The cuBLAS call reads the correct 512×512 block using lda to skip
        the unused columns.

    MINIMUM valid value: lda ≥ M (for CUBLAS_OP_N).
    MOST COMMON value: lda = M (tightly packed, no padding).

    PyTorch tensors: .stride(0) gives the leading dimension (for a 2D tensor).
    A contiguous FP32 matrix of shape [M, K] in row-major has:
        .stride() = (K, 1)
        cuBLAS column-major lda = K   (treat row-major as transposed column-major)
        → Pass CUBLAS_OP_T to cuBLAS for row-major input from PyTorch.

### Column-Major vs Row-Major: The Transpose Trick

    PyTorch stores tensors in ROW-MAJOR order.
    cuBLAS expects COLUMN-MAJOR order.
    The standard idiom for calling cuBLAS from row-major code:

    WANT: C_rowmajor = A_rowmajor × B_rowmajor   (M×N = M×K × K×N)
    KNOW: Cᵀ = Bᵀ × Aᵀ   (since (AB)ᵀ = BᵀAᵀ)
    cuBLAS column-major sees: op(A)=Aᵀ, op(B)=Bᵀ → output is Cᵀ

    CALL:
        cublasSgemm(handle,
            CUBLAS_OP_N,   // op(A_colmajor) = Bᵀ (no additional transpose)
            CUBLAS_OP_N,   // op(B_colmajor) = Aᵀ
            N, M, K,       // note: N and M are SWAPPED vs the row-major signature
            &alpha,
            B_rowmajor, N, // B in row-major = Bᵀ in col-major, ldb=N
            A_rowmajor, K, // A in row-major = Aᵀ in col-major, lda=K
            &beta,
            C_rowmajor, N  // output in row-major = Cᵀ in col-major, ldc=N
        );

    RESULT: C_rowmajor contains A × B correctly.
    This transpose trick is used internally by every deep learning framework.

### Full cuBLAS SGEMM Signature

    cublasStatus_t cublasSgemm(
        cublasHandle_t handle,
        cublasOperation_t transa,   // CUBLAS_OP_N, _T, or _C for A
        cublasOperation_t transb,   // CUBLAS_OP_N, _T, or _C for B
        int m,                      // rows of op(A) and C
        int n,                      // cols of op(B) and C
        int k,                      // inner dimension
        const float *alpha,         // pointer to α (can be on GPU)
        const float *A, int lda,    // A matrix and leading dimension
        const float *B, int ldb,    // B matrix and leading dimension
        const float *beta,          // pointer to β (can be on GPU)
        float *C, int ldc           // C matrix and leading dimension
    );

    Error codes:
        CUBLAS_STATUS_SUCCESS          (0): success
        CUBLAS_STATUS_NOT_INITIALIZED  (1): handle not created
        CUBLAS_STATUS_INVALID_VALUE    (7): bad m/n/k or lda < max(1,m)
        CUBLAS_STATUS_EXECUTION_FAILED (13): GPU kernel execution failed

    Always check return codes in production code. cuBLAS errors are silent
    without checking — the GPU will just write garbage to C.


##### PART 2 — SGEMM / DGEMM: PRECISION, ARITHMETIC INTENSITY & PERFORMANCE MODEL

### Precision Options and Their Rooflines

    cuBLAS supports multiple precisions, each with a different peak rate
    and a different hardware unit:

    SGEMM (FP32):
        Hardware: FP32 CUDA cores (non-tensor-core path)
        H100 SXM5 peak:    67 TFLOP/s (FP32)
        A100 SXM4 peak:    19.5 TFLOP/s (FP32)
        Used for: inference where FP32 is required, legacy code
        lda/ldb dtype: float*

    DGEMM (FP64):
        Hardware: FP64 CUDA cores
        H100 SXM5 peak:    34 TFLOP/s (FP64)
        A100 SXM4 peak:    9.7 TFLOP/s (FP64)
        Used for: scientific computing, simulation, numerical stability
        lda/ldb dtype: double*

    HGEMM (FP16, cublasSgemmEx or cublasHgemm):
        Hardware: Tensor cores (Volta+)
        H100 SXM5 peak:   989 TFLOP/s (FP16 tensor core)
        A100 SXM4 peak:   312 TFLOP/s (FP16 tensor core)
        Used for: deep learning training and inference
        Requires: m, n, k all multiples of 8 for tensor core alignment

    TF32 (via cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32):
        Hardware: Tensor cores with FP32 accumulation
        H100 SXM5 peak:   989 TFLOP/s (TF32 tensor core)
        A100 SXM4 peak:   312 TFLOP/s (TF32 tensor core)
        Used for: drop-in FP32 replacement with TC speed, minimal accuracy loss
        Note: TF32 rounds the mantissa to 10 bits. Exponent kept at 8 bits.

    BF16 (via cublasGemmEx with BFLOAT16 compute type):
        H100 peak:  1979 TFLOP/s
        A100 peak:   312 TFLOP/s
        Better dynamic range than FP16 (8 exp bits vs 5), same mantissa as TF32.

    INT8 (IMMA tensor cores):
        H100 peak:  3958 TOPS (INT8)
        Used for: quantised inference (post-training quantisation)

### The GEMM Performance Model

    For a well-tuned GEMM on modern hardware, performance is bounded by
    the LARGER of two ceilings:

    COMPUTE CEILING:
        time_compute = 2 × M × N × K / peak_flops

    MEMORY CEILING (data must arrive from HBM):
        bytes = (M×K + K×N + M×N) × sizeof(dtype)
        time_memory = bytes / peak_hbm_bw

    Achieved time ≈ max(time_compute, time_memory)
    → Compute-bound: K is large relative to M,N (AI > ridge point)
    → Memory-bound:  K is small (AI < ridge point)

    Ridge point for H100 SXM5 (FP16 tensor core):
        ridge_point = peak_flops / peak_hbm_bw
                    = 989e12 / 3.35e12  ≈ 295 FLOPs/Byte

    For a square GEMM of side M:
        AI = 2M³ / (3M² × 2 bytes) = M / 3 FLOPs/Byte
        Compute-bound when M > 3 × 295 = 885

    CONSEQUENCE: LLM training (batch=512, seq=2048, dim=4096) → large M,
    easily compute-bound. LLM inference (batch=1, seq=1) → tiny M, often
    memory-bound. This is the fundamental tradeoff in serving.

### Alignment Requirements for Tensor Core Paths

    Tensor cores operate on FIXED TILE SIZES. For FP16 on Ampere:
        Warp-level tile: 16×16×16 (m×n×k per warp)
        Block-level tile: multiples of 16 in all dimensions

    For cuBLAS to use the tensor core path automatically:
        M, N, K must be multiples of 8 (minimum alignment)
        M, N, K multiples of 16 give best performance
        M, N, K multiples of 128 are ideal for largest tiles

    Non-aligned dimensions: cuBLAS pads internally, wasting cycles.
    Example: M=127, N=64, K=64 → padded to M=128, fills extra row with zeros.
    For critical hot paths: ensure all GEMM dimensions are multiples of 128.


##### PART 3 — BATCHED GEMM: POINTER ARRAY VS STRIDED BATCH

### The Two Batched GEMM Variants

    When you need to compute many independent GEMMs simultaneously,
    cuBLAS offers two APIs:

    POINTER ARRAY BATCH (cublasSgemmBatched):
        Takes arrays of pointers: Aarray[batchCount], Barray[batchCount], Carray[batchCount]
        Each pointer can point to ANY location in GPU memory.
        Matrices in the batch can have DIFFERENT sizes? NO — all same M, N, K, lda, ldb, ldc.
        But they can be NON-CONTIGUOUS (e.g., from a ragged tensor).
        Cost: pointer array itself must live on the GPU (small extra allocation).
        Good for: jagged batches, attention over variable-length sequences.

    STRIDED BATCH (cublasSgemmStridedBatched):
        Matrices are equally spaced in memory:
            A_i starts at: A_ptr + i * strideA
            B_i starts at: B_ptr + i * strideB
            C_i starts at: C_ptr + i * strideC
        One contiguous allocation for all batch elements.
        Lower overhead than pointer array (no pointer dereference per batch).
        Good for: attention Q@K and scores@V (batch dimension is num_heads).
        Used by every production LLM for multi-head attention GEMM calls.

    FULL SIGNATURES:

    Pointer array:
        cublasStatus_t cublasSgemmBatched(
            cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k,
            const float *alpha,
            const float *const Aarray[], int lda,
            const float *const Barray[], int ldb,
            const float *beta,
            float *const Carray[], int ldc,
            int batchCount
        );

    Strided batch:
        cublasStatus_t cublasSgemmStridedBatched(
            cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k,
            const float *alpha,
            const float *A, int lda, long long strideA,
            const float *B, int ldb, long long strideB,
            const float *beta,
            float *C, int ldc, long long strideC,
            int batchCount
        );

### Attention's Exact Mapping to Strided Batched GEMM

    Multi-head attention with:
        B = batch_size, H = num_heads, S = seq_len, D = head_dim

    QUERY shape:  [B, H, S, D]  → Q tensor
    KEY shape:    [B, H, S, D]  → K tensor
    VALUE shape:  [B, H, S, D]  → V tensor

    STEP 1 — Attention scores: S_scores = Q @ Kᵀ / sqrt(D)
        Each head: (S×D) @ (D×S) = (S×S)
        Batched: batchCount = B × H
        M = S, N = S, K = D
        strideA = S × D   (next Q head starts S*D elements later)
        strideB = S × D   (next K head starts S*D elements later)
        strideC = S × S   (next score matrix is S*S elements later)
        lda = D, ldb = D, ldc = S

    STEP 2 — Weighted sum: output = softmax(S_scores) @ V
        Each head: (S×S) @ (S×D) = (S×D)
        Batched: batchCount = B × H
        M = S, N = D, K = S
        strideA = S × S   (softmax matrix stride)
        strideB = S × D   (next V head stride)
        strideC = S × D   (next output head stride)

    These two strided batched GEMM calls are the dominant operations in
    every transformer attention layer. PyTorch's torch.bmm() calls
    cublasSgemmStridedBatched (or the cublasGemmStridedBatchedEx equivalent
    for FP16) under the hood.

### When Batched GEMM Outperforms Sequential GEMMs

    For SMALL per-batch GEMMs: batched is much better.
    Reason: a single small GEMM cannot fill all SMs. The GPU is underutilised.
    With batch_size=32, all 32 GEMMs can run concurrently → full SM utilisation.

    Break-even point (H100):
        A single GEMM saturates all SMs when M×N×K is large enough.
        Approximate threshold: M×N×K > 1e9 (depending on dtype).
        Below this: batching provides near-linear speedup with batch_size.

    LLM inference (batch=1, decode step):
        K = 4096 (model dim), M = 1 (single token), N = 4096 (output dim)
        Single GEMM FLOPs = 2 × 1 × 4096 × 4096 = 33.5 MFLOP
        At 989 TFLOP/s peak: 34 µs to keep all SMs busy.
        Actual time: ~80 µs (memory-bound — need to load 4096×4096 weights).
        Batching 32 requests: M=32, still memory-bound but 32× more output/second.


##### PART 4 — cublasLt: THE LOW-LEVEL TENSOR CORE API

### What cublasLt Is

    cublasLt (cuBLAS Lightweight) is a lower-level API introduced in CUDA 10.1.
    It exposes features that cublasSgemm/cublasSgemmEx hide behind heuristics:

        1. EPILOGUE FUSION — fuse bias add, ReLU, GELU, or other elementwise
           ops into the GEMM kernel. One kernel instead of two.

        2. ALGORITHM SEARCH — enumerate algorithms explicitly, benchmark them
           for your specific M, N, K, and save the winner. Beats heuristics.

        3. WORKSPACE CONTROL — allocate a larger workspace to enable faster,
           memory-intensive algorithms.

        4. DATA LAYOUT CONTROL — column-major, row-major, or
           IMMA_INTERLEAVED (needed for INT8 tensor cores).

        5. SCALE TYPE CONTROL — accumulate in FP32 while computing in FP16.
           Essential for mixed-precision training correctness.

### The cublasLt Descriptor System

    Unlike cublasSgemm (one function call), cublasLt uses DESCRIPTORS —
    opaque objects that carry configuration state:

    MATMUL DESCRIPTOR (cublasLtMatmulDesc_t):
        Holds: compute type, scale type, pointer mode, epilogue mode,
               transpose modes, bias pointer.
        Created per GEMM "type" (not per GEMM call).

    MATRIX LAYOUT (cublasLtMatrixLayout_t):
        Holds: dtype, rows, cols, leading dimension, batch_count, batch_stride.
        Created for each matrix (A, B, C, D).
        Note: D is the output matrix (can differ from C, which is bias input).

    MATMUL PREFERENCE (cublasLtMatmulPreference_t):
        Holds: workspace size budget, alignment constraints, algorithm IDs.
        Controls algorithm selection.

    ALGORITHM HANDLE (cublasLtMatmulAlgo_t):
        Identifies a specific GEMM algorithm (tile size, split-K strategy, etc.)
        Obtained from: cublasLtMatmulAlgoGetHeuristic (best guess) or
                       cublasLtMatmulAlgoCheck (enumerate and benchmark).

### Minimal cublasLt FP16 GEMM with Bias + ReLU

    // 1. Create handle and descriptors
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // 2. Set epilogue: GEMM + bias + ReLU in one kernel
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

    // 3. Set bias pointer (same device pointer each call if shape unchanged)
    cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));

    // 4. Matrix layout descriptors (FP16 inputs, FP32 output)
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, M, K, lda);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16F, K, N, ldb);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, ldc);  // bias input
    cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_32F, M, N, ldd);  // output

    // 5. Preference: give cuBLAS 256 MB workspace for faster algorithms
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t workspaceSz = 256 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSz, sizeof(workspaceSz));

    // 6. Get best algorithm heuristic
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedAlgoCount = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc,
        layoutA, layoutB, layoutC, layoutD,
        pref, 1, &heuristicResult, &returnedAlgoCount);

    // 7. Execute
    cublasLtMatmul(ltHandle, matmulDesc,
        &alpha, A, layoutA, B, layoutB,
        &beta,  C, layoutC, D, layoutD,
        &heuristicResult.algo,
        workspace, workspaceSz,
        stream);

### Epilogue Fusion Options

    CUBLASLT_EPILOGUE_DEFAULT           no epilogue (D = alpha*A*B + beta*C)
    CUBLASLT_EPILOGUE_RELU              D = max(0, A*B + bias)
    CUBLASLT_EPILOGUE_BIAS              D = A*B + bias     (bias broadcast over M)
    CUBLASLT_EPILOGUE_RELU_BIAS         D = max(0, A*B + bias)
    CUBLASLT_EPILOGUE_GELU              D = GELU(A*B)      (Gaussian Error Linear Unit)
    CUBLASLT_EPILOGUE_GELU_BIAS         D = GELU(A*B + bias)
    CUBLASLT_EPILOGUE_GELU_AUX         D = GELU(A*B), also writes aux tensor for BWD
    CUBLASLT_EPILOGUE_DRELU             D = dReLU (backward pass epilogue)
    CUBLASLT_EPILOGUE_DGELU             D = dGELU (backward pass epilogue)

    PERFORMANCE IMPACT of epilogue fusion:
        Without fusion: GEMM kernel → store output → ReLU kernel → store output
        Two HBM round-trips for the output tensor (MN × dtype × 2).
        With CUBLASLT_EPILOGUE_RELU_BIAS:
        Single kernel, single HBM write.
        For large M×N: 1.8–2.0× speedup just from eliminating the second pass.

### Algorithm Search: Heuristic vs Exhaustive

    HEURISTIC (production default):
        cublasLtMatmulAlgoGetHeuristic returns the BEST GUESS algorithm
        based on problem dimensions and GPU architecture.
        Fast to call (~1 µs). Returns top-N candidates (usually N=1 in practice).
        May not be optimal — NVIDIA's heuristic is trained offline.

    EXHAUSTIVE SEARCH (offline tuning):
        Enumerate all available algorithms:
        cublasLtMatmulAlgoGetHeuristic(..., returnedResults=8, ...);
        // Get 8 candidates, benchmark each:
        for (int i = 0; i < returnedResults; i++) {
            cudaEvent_t start, stop;
            cudaEventRecord(start);
            for (int iter = 0; iter < 100; iter++)
                cublasLtMatmul(..., &results[i].algo, ...);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            // save winner
        }
        Typical speedup vs heuristic: 10–40% for the specific M, N, K, dtype.
        Used in: Flash attention library, Transformer Engine, xFormers.

### Workspace and Split-K

    WORKSPACE SIZE affects which algorithms are available:
        0 bytes: only basic tile algorithms
        32 MB:   enables split-K reduction algorithms
        256 MB:  enables larger tile/persistent algorithms (best for large GEMM)
        512 MB:  maximum useful size on H100

    SPLIT-K STRATEGY:
        The K dimension is split into K/R chunks, each computed by a subset of SMs.
        A secondary reduction step combines the partial sums.
        Useful when K is very large relative to M and N.
        Trade-off: extra atomic reduction overhead.
        cublasLt workspace stores the partial sums for the reduction.

    WORKSPACE ALLOCATION PATTERN (PyTorch convention):
        torch.cuda.cublaslt allocates a shared workspace buffer once at startup.
        Default: 32 MB. Can be increased:
            torch.backends.cuda.preferred_linalg_library("cublaslt")
            # workspace size is set via environment or cuBLAS handle


##### PART 5 — cuBLAS HANDLE, STREAM, AND WORKSPACE MANAGEMENT

### The Handle

    Every cuBLAS function takes a cublasHandle_t handle.
    The handle encapsulates:
        - Associated CUDA stream (default: stream 0)
        - Math mode (CUBLAS_DEFAULT_MATH vs CUBLAS_TENSOR_OP_MATH)
        - Pointer mode (host vs device)
        - Internal workspace pointer and size

    CREATION / DESTRUCTION:
        cublasHandle_t handle;
        cublasCreate(&handle);
        // ... use handle ...
        cublasDestroy(handle);

    ONE HANDLE PER STREAM (for multi-stream pipelines):
        cublasSetStream(handle, stream);
        // All subsequent calls on this handle execute on 'stream'
        // For simultaneous multi-stream GEMM: create separate handles.
        // Handles share the GPU but execute asynchronously.

### Math Mode: Enabling Tensor Cores

    CUBLAS_DEFAULT_MATH:
        Uses FP32 CUDA cores for SGEMM. Does NOT use tensor cores.
        Fully IEEE 754 compliant.

    CUBLAS_TF32_TENSOR_OP_MATH (A100+):
        Rounds FP32 inputs to TF32 (10-bit mantissa) before sending to TC.
        Uses tensor cores. ~10× faster than FP32 CUDA cores.
        Accuracy: 99.9% of FP32 GEMM outputs within 1 ULP for well-conditioned A,B.
        Enable: cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)
        OR:     set environment CUBLAS_WORKSPACE_CONFIG=:16:8

    CUBLAS_TENSOR_OP_MATH:
        Full TC path. For FP16: uses FP16 TC with FP32 accumulation.
        Enable: cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)

    PyTorch: torch.backends.cuda.matmul.allow_tf32 = True
        → Sets CUBLAS_TF32_TENSOR_OP_MATH on all internal handles.

### Pointer Mode: Scalars on Host vs Device

    CUBLAS_POINTER_MODE_HOST (default):
        α and β are pointers to host memory. cuBLAS reads them before launching.
        Simple. But: breaks asynchrony if α is computed on GPU first.

    CUBLAS_POINTER_MODE_DEVICE:
        α and β are pointers to DEVICE memory. cuBLAS reads them at kernel time.
        Enables: GEMM whose scale factor is itself computed by a preceding kernel.
        Required for: chained operations without CPU synchronisation.
        Enable: cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE)

    EXAMPLE — fused scale computed on GPU:
        // Kernel 1: compute scale_d on GPU (e.g., 1/norm)
        compute_scale<<<...>>>(input, scale_d);
        // Kernel 2: GEMM scaled by scale_d (no CPU sync needed)
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSgemm(handle, ..., scale_d, A, ..., zero_d, C, ...);
        // GPU executes both kernels, CPU never blocks

### Workspace for cuBLAS (CUBLAS_WORKSPACE_CONFIG)

    cuBLAS uses an internal scratch buffer for:
        - Split-K partial sums
        - Tiling algorithm temporaries
        - Algorithm selection cache

    Default workspace: 8 MB (CUDA < 11.x), variable (CUDA 11+).
    To increase workspace and enable better algorithms:

        CUBLAS_WORKSPACE_CONFIG=:4096:8
        (4096 MB × 8 streams = 32 GB total — only for HPC applications)

        CUBLAS_WORKSPACE_CONFIG=:256:8
        (256 MB per stream — standard production setting for large GEMM)

    Per-handle workspace (preferred in production):
        cublasSetWorkspace(handle, workspace_ptr, workspace_bytes);


##### PART 6 — GEMM PERFORMANCE TUNING: DIMENSION ALIGNMENT, PADDING & ALGORITHM SELECTION

### Dimension Alignment Tiers

    TIER 1 — Multiples of 128 (ideal):
        Enables the largest tile algorithms (128×128, 256×128).
        All warp tiles fill completely. Zero padding overhead.
        EXAMPLE: M=4096, N=4096, K=4096 → ideal.

    TIER 2 — Multiples of 64:
        Enables most tile algorithms. Small loss vs Tier 1.
        Typical for many transformer layers (head_dim=64 is common).

    TIER 3 — Multiples of 16 (minimum for tensor cores):
        Tensor core warp tile = 16×16×16. Must be aligned here.
        Below this: cuBLAS pads internally → wasted cycles.

    TIER 4 — Non-power-of-2, non-multiples of 16:
        cuBLAS falls back to slower SIMT path (no tensor cores).
        M=127, K=33: expect 5–20× slowdown vs aligned equivalent.

    DIAGNOSIS: if a GEMM runs slower than expected, check alignment.
        ncu metric: sm__inst_executed_pipe_tensor_op_hmma.sum == 0
        → Tensor cores not used → dimension alignment problem.

### Padding Strategy for Non-Aligned Dimensions

    If model dimensions are fixed at non-aligned values (e.g., vocab_size=50257):
        PAD AT THE FRAMEWORK LEVEL: extend to next multiple of 128.
        torch.nn.Linear(50257, dim) → pad weight to [50304, dim].
        Cost: 47 extra rows × 4096 cols × 2 bytes = 386 KB. Negligible.
        Benefit: tensor core path enabled. Up to 3× speedup.

    PyTorch auto-padding for attention:
        For MHA with head_dim=64: already aligned to 64.
        For head_dim=80 (Llama 3): multiples of 16 but not 64.
        Custom kernel or cublasLt with explicit padding needed.

### The GEMM Kernel Algorithm Zoo

    cuBLAS (and cublasLt) maintains a library of O(1000) GEMM algorithms.
    Key differentiators:

    TILE SIZE (M_tile × N_tile × K_tile):
        128×128×32: best for balanced M, N. Uses 2KB registers + 32KB SMEM.
        256×128×32: better for large M. More output reuse.
        64×64×16: better for small M (inference batch=1).

    PIPELINE STAGES:
        1 stage: classic tiled GEMM (load tile, compute, load next).
        2 stage: double buffering (load N+1 while computing N).
        3+ stage: software pipelining with async memory copies (cp.async).

    SPLIT-K FACTOR:
        1: no split (standard). Best when K is moderate.
        2, 4, 8: split K dimension, reduce partial sums.
        Best when K >> M and K >> N (e.g., decode GEMM: M=batch_size, K=4096).

    SWIZZLE:
        Different SMs assigned to different output tile quadrants.
        Reduces L2 bank conflicts for large N.

    PERSISTENT KERNELS (H100 warp specialisation):
        Warps specialised as either producers (load) or consumers (compute).
        Warp specialisation hides memory latency at the warp scheduler level.
        H100 tensor memory accelerator (TMA) used for async tile loads.
        Enables near-theoretical tensor core utilisation (>90% on H100).


##### PART 7 — PYTORCH INTEGRATION: HOW torch.mm / torch.matmul CALL cuBLAS

### The Call Chain

    torch.mm(A, B) → at::mm → ATen CUDA dispatch →
    at::native::mm_cuda → cublas::gemm<...>() →
    cublasGemmEx(handle, ...) or cublasLtMatmul(...)

    Key decisions made in ATen:
        1. dtype determines which cuBLAS function is called
        2. allow_tf32 flag sets math mode
        3. If shape is batched: torch.bmm → StridedBatched variant
        4. If cublasLt preferred: cublasLtMatmul with heuristic

### Tensor Contiguity and the .contiguous() Problem

    torch.matmul checks if tensors are contiguous. If NOT contiguous:
        PyTorch inserts a .contiguous() call (allocates new tensor, copies).
        This can be the dominant cost for small GEMMs on non-contiguous views.

    DIAGNOSIS: profiling shows extra memcpy before GEMM kernel.
    FIX: ensure inputs are contiguous before hot loop:
        A = A.contiguous()
        # OR
        A = A.view(M, K)  # view doesn't copy, but only works if stride allows

    CONTIGUOUS CHECK: A.is_contiguous()  # True if stride[-1] == 1

### Mixed Precision with torch.cuda.amp

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(input)

    This context manager:
        1. Casts GEMM inputs to FP16 (linear layers, matmul, bmm)
        2. Keeps loss, softmax, layer norm in FP32 (numerical stability)
        3. cuBLAS then uses HGEMM with FP32 accumulation (tensor core path)
        4. GradScaler handles loss scaling to prevent FP16 underflow

    EFFECTIVE TYPE HIERARCHY:
        GEMM kernel: FP16 inputs, FP32 accumulation per warp tile.
        Final output: depends on output_dtype specified in the GEMM.
        Gradient accumulation: FP32 (master weights).

### cublasGemmEx: The Universal Precision API

    For mixed input/output types, use cublasGemmEx:

        cublasGemmEx(handle,
            transa, transb,
            m, n, k,
            &alpha,
            A, CUDA_R_16F, lda,   // FP16 input A
            B, CUDA_R_16F, ldb,   // FP16 input B
            &beta,
            C, CUDA_R_32F, ldc,   // FP32 output C
            CUBLAS_COMPUTE_32F,   // accumulate in FP32
            CUBLAS_GEMM_DEFAULT_TENSOR_OP  // use tensor cores
        );

    Compute types (cublasComputeType_t):
        CUBLAS_COMPUTE_16F         FP16 in, FP16 accum, FP16 out
        CUBLAS_COMPUTE_32F         FP32 in, FP32 accum, FP32 out
        CUBLAS_COMPUTE_32F_FAST_16F  FP32 in rounded to FP16, TC path
        CUBLAS_COMPUTE_32F_FAST_TF32 FP32 in rounded to TF32, TC path
        CUBLAS_COMPUTE_64F         FP64 in, FP64 accum, FP64 out
        CUBLAS_COMPUTE_32I         INT32 in, INT32 accum
        CUBLAS_COMPUTE_32I_PEDANTIC  strict INT32 (no tensor cores)

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · GEMM Performance Model — Roofline, Arithmetic Intensity & Alignment": {
        "description": (
            "Build the complete GEMM performance model for H100/A100/A10G. "
            "Compute arithmetic intensity for every GEMM shape in a 7B LLM "
            "(Q/K/V projections, FFN gate/up/down, vocab head). Place each on "
            "the roofline, determine compute-bound vs memory-bound status. Show "
            "how batch size and sequence length change the balance. Demonstrate "
            "dimension alignment impact on achievable performance."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  GEMM PERFORMANCE MODEL — Roofline, AI & Alignment")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────
# Hardware specs
# ─────────────────────────────────────────────────────────────────────

GPUS = {
    "H100 SXM5": {
        "fp16_tc_tflops":  989.0,
        "fp32_tflops":      67.0,
        "fp64_tflops":      34.0,
        "hbm_bw_gbs":     3350.0,
        "l2_bw_gbs":      12000.0,
        "sms":              132,
    },
    "A100 SXM4": {
        "fp16_tc_tflops":  312.0,
        "fp32_tflops":      19.5,
        "fp64_tflops":       9.7,
        "hbm_bw_gbs":     2000.0,
        "l2_bw_gbs":      8000.0,
        "sms":               108,
    },
    "A10G": {
        "fp16_tc_tflops":   31.2,
        "fp32_tflops":       7.8,
        "fp64_tflops":       0.24,
        "hbm_bw_gbs":      600.0,
        "l2_bw_gbs":      3600.0,
        "sms":              80,
    },
}


def gemm_perf(m, n, k, dtype_bytes, gpu, batch=1):
    """
    Compute arithmetic intensity, time lower bound, bottleneck.
    Returns dict with all metrics.
    """
    flops    = 2.0 * m * n * k * batch
    bytes_io = (m * k + k * n + m * n) * dtype_bytes * batch
    ai       = flops / bytes_io

    peak_flops = gpu["fp16_tc_tflops"] * 1e12   # using TC path
    peak_bw    = gpu["hbm_bw_gbs"] * 1e9
    ridge      = peak_flops / peak_bw            # FLOPs/Byte

    t_compute_us = flops / peak_flops * 1e6
    t_memory_us  = bytes_io / peak_bw * 1e6
    t_bound_us   = max(t_compute_us, t_memory_us)

    bottleneck   = "compute" if ai >= ridge else "memory"
    efficiency   = min(ai / ridge, 1.0)   # fraction of peak achievable

    return {
        "m": m, "n": n, "k": k, "batch": batch,
        "flops_gflop":    flops / 1e9,
        "bytes_gb":       bytes_io / 1e9,
        "ai":             ai,
        "ridge":          ridge,
        "t_compute_us":   t_compute_us,
        "t_memory_us":    t_memory_us,
        "t_bound_us":     t_bound_us,
        "bottleneck":     bottleneck,
        "peak_efficiency":efficiency * 100,
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Ridge point for each GPU
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Ridge Points: FP16 Tensor Core Roofline")
print("━" * 68)
print()

print(f"  {'GPU':<14}  {'FP16 TC TFLOP/s':>17}  {'HBM GB/s':>10}  "
      f"{'Ridge (FLOPs/B)':>17}  {'Square GEMM compute-bound M>'}")
print("  " + "─" * 72)

for gname, gpu in GPUS.items():
    ridge = gpu["fp16_tc_tflops"] * 1e12 / (gpu["hbm_bw_gbs"] * 1e9)
    # For square GEMM: AI = M/3 bytes (FP16, 2 bytes each)
    # Compute-bound when M/3 > ridge → M > 3 * ridge
    m_threshold = 3 * ridge * 2  # 2 bytes for FP16
    print(f"  {gname:<14}  {gpu['fp16_tc_tflops']:>17.1f}  "
          f"{gpu['hbm_bw_gbs']:>10.0f}  {ridge:>17.1f}  M > {m_threshold:.0f}")

print()
print("  LLM training (M=batch*seq=2048): H100 is compute-bound. ✅")
print("  LLM inference decode (M=batch=1): ALL GPUs are memory-bound. ⚠")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Every GEMM in a 7B LLM (Llama-2-7B style)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Roofline for Every GEMM in Llama-2-7B")
print("━" * 68)
print()

# Llama-2-7B architecture
D      = 4096   # hidden dim
D_FF   = 11008  # FFN intermediate (SwiGLU: 2/3 * 4 * D)
N_HEAD = 32
D_HEAD = D // N_HEAD  # 128
VOCAB  = 32000

# Batch size and sequence length variants
scenarios = [
    ("Inference decode (B=1,  S=1)",    1,    1),
    ("Inference prefill (B=1,  S=512)", 1,  512),
    ("Training (B=4, S=2048)",          4, 2048),
]

gemm_ops = [
    ("Q projection",     D,    D,    D),
    ("K projection",     D,    D,    D),
    ("V projection",     D,    D,    D),
    ("O projection",     D,    D,    D),
    ("FFN gate (SwiGLU)",D_FF, D,    D),
    ("FFN up   (SwiGLU)",D_FF, D,    D),
    ("FFN down",         D,    D_FF, D),
    ("Vocab head",       VOCAB,D,    D),
]

gpu = GPUS["H100 SXM5"]
DTYPE_BYTES = 2   # FP16

for sc_name, B, S in scenarios:
    M_tokens = B * S
    print(f"  Scenario: {sc_name}  (M_tokens = {M_tokens})")
    print()
    print(f"  {'Operation':<22}  {'M':>6}  {'N':>6}  {'K':>6}  "
          f"{'AI':>7}  {'Ridge':>7}  {'Bound':>8}  {'Time µs':>9}  {'Eff%'}")
    print("  " + "─" * 76)

    ridge = gpu["fp16_tc_tflops"] * 1e12 / (gpu["hbm_bw_gbs"] * 1e9)
    for op_name, N, K, _ in gemm_ops:
        r = gemm_perf(M_tokens, N, K, DTYPE_BYTES, gpu)
        bound_sym = "🔴 mem" if r["bottleneck"] == "memory" else "🟢 cmp"
        print(f"  {op_name:<22}  {M_tokens:>6}  {N:>6}  {K:>6}  "
              f"{r['ai']:>7.1f}  {r['ridge']:>7.1f}  {bound_sym:>8}  "
              f"{r['t_bound_us']:>9.3f}  {r['peak_efficiency']:.0f}%")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Alignment impact — performance vs dimension alignment
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Dimension Alignment: Performance vs Alignment Tier")
print("━" * 68)
print()

print("  Fixed problem: M=1024, N=1024, varying K alignment.")
print("  FP16, H100 SXM5. Model: aligned K achieves TC path; misaligned falls back.")
print()
print(f"  {'K':>8}  {'Alignment':>12}  {'TC path':>9}  {'AI':>7}  "
      f"{'Ideal µs':>10}  {'Actual µs est.':>16}  {'Penalty'}")
print("  " + "─" * 70)

M_align, N_align = 1024, 1024

for k_val in [32, 64, 128, 256, 512, 1023, 1024, 1025, 4096, 4097]:
    # Alignment tier
    if k_val % 128 == 0:
        tier, tc = "mult-128", True
    elif k_val % 64 == 0:
        tier, tc = "mult-64", True
    elif k_val % 16 == 0:
        tier, tc = "mult-16", True
    else:
        tier, tc = "non-aligned", False

    r = gemm_perf(M_align, N_align, k_val, DTYPE_BYTES, GPUS["H100 SXM5"])
    ideal_us = r["t_bound_us"]

    # Non-TC fallback: use FP32 CUDA cores (~15× slower than TC for FP16)
    if not tc:
        # Must pad K to next multiple of 16 for SMEM tiling, then run on CUDA cores
        k_padded = ((k_val + 15) // 16) * 16
        wasted_pct = (k_padded - k_val) / k_padded * 100
        actual_us = ideal_us * 12.0  # approximate fallback penalty
        penalty = "~12× (no TC)"
    else:
        actual_us = ideal_us
        penalty = "none"

    print(f"  {k_val:>8}  {tier:>12}  {'✅' if tc else '❌':>9}  "
          f"{r['ai']:>7.1f}  {ideal_us:>10.4f}  {actual_us:>16.4f}  {penalty}")

print()
print("  KEY: K=1023 is non-aligned → 12× slower than K=1024 for FP16 GEMM.")
print("  Always pad vocab_size, FFN dim, etc. to nearest multiple of 128.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Leading Dimension & Transpose Mechanics — Memory Layout Deep Dive": {
        "description": (
            "Simulate column-major vs row-major GEMM memory layouts. Show exactly "
            "how lda, ldb, ldc map to physical memory addresses. Derive the "
            "row-major → column-major transpose trick used by every framework. "
            "Verify correctness by computing A@B both ways and confirming identical "
            "numerical results. Demonstrate submatrix extraction via lda padding "
            "and its role in mixed-precision fused operations."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 68)
print("  LEADING DIMENSION & TRANSPOSE MECHANICS — Memory Layout")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Column-major element addressing
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Column-Major Addressing: A[row, col] in Memory")
print("━" * 68)
print()

print("  cuBLAS stores matrices in COLUMN-MAJOR order (Fortran convention).")
print("  Element A[row, col] lives at offset: col * lda + row")
print()

# 4x4 example
M_ex, N_ex = 4, 4
lda_tight = M_ex       # tightly packed: lda == M
lda_padded = M_ex + 2  # padded allocation: lda == M+2

A_colmaj = np.arange(1, M_ex * N_ex + 1, dtype=float).reshape(N_ex, M_ex).T
print(f"  Matrix A ({M_ex}×{N_ex}) — values:")
for row in range(M_ex):
    vals = "  ".join(f"{A_colmaj[row, col]:>3.0f}" for col in range(N_ex))
    print(f"    row {row}: [{vals}]")
print()

print(f"  Memory layout (column-major, lda={lda_tight}):")
print(f"  Offset:  " + " ".join(f"{col*lda_tight + row:>4}"
      for col in range(N_ex) for row in range(M_ex)))
print(f"  Value:   " + " ".join(f"{A_colmaj[row, col]:>4.0f}"
      for col in range(N_ex) for row in range(M_ex)))
print()

print(f"  With lda={lda_padded} (padded allocation):")
print(f"  A[row={0}, col={0}] at offset {0 * lda_padded + 0}")
print(f"  A[row={1}, col={0}] at offset {0 * lda_padded + 1}")
print(f"  A[row={0}, col={1}] at offset {1 * lda_padded + 0}")
print(f"  A[row={3}, col={3}] at offset {3 * lda_padded + 3}")
print()
print(f"  When lda > M: rows M..lda-1 in each column are UNUSED padding.")
print(f"  cuBLAS skips them correctly because it uses lda to stride between columns.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: The row-major → column-major transpose trick
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — The Transpose Trick: cuBLAS from Row-Major (PyTorch)")
print("━" * 68)
print()

# Build test matrices (row-major, as PyTorch stores them)
M, N, K = 3, 4, 5
A_row = np.random.randn(M, K).astype(np.float32)
B_row = np.random.randn(K, N).astype(np.float32)

# Reference result (NumPy, row-major)
C_ref = A_row @ B_row
print(f"  Problem: C = A @ B   (M={M}, N={N}, K={K})")
print(f"  Reference C[0,:] = {C_ref[0].round(4).tolist()}")
print()

print("  MATHEMATICAL DERIVATION OF THE TRICK:")
print("  ─────────────────────────────────────")
print("  cuBLAS expects column-major. A row-major matrix A of shape (M, K)")
print("  looks like Aᵀ to cuBLAS (column-major with lda=K sees transposed view).")
print()
print("  We WANT:   C     = A × B")
print("  cuBLAS sees: Cᵀ = Bᵀ × Aᵀ      (using identity: (AB)ᵀ = BᵀAᵀ)")
print()
print("  So we call cuBLAS with:")
print("    CUBLAS_OP_N on 'B_rowmajor' (it reads it as Bᵀ, no extra T needed)")
print("    CUBLAS_OP_N on 'A_rowmajor' (it reads it as Aᵀ, no extra T needed)")
print("    m_arg = N, n_arg = M, k_arg = K   ← M and N SWAPPED")
print("    ldb = N (width of B row-major = cols)")
print("    lda = K (width of A row-major = cols)")
print("    ldc = N (width of C row-major = cols)")
print("  cuBLAS computes: Cᵀ_colmaj = Bᵀ × Aᵀ")
print("  Reading Cᵀ_colmaj as row-major gives C correctly. ✅")
print()

# Simulate: interpret row-major A as col-major Aᵀ
# NumPy analog: cuBLAS call with transposed arguments
# cublas(B_row.T, A_row.T) in col-major = B @ A in row-major ordering
# But using the actual trick: compute Bᵀ × Aᵀ (numpy matmul of transposed)
C_trick = (B_row.T @ A_row.T).T   # = A @ B
print(f"  Result via transpose trick C[0,:] = {C_trick[0].round(4).tolist()}")
print(f"  Match reference: {'✅' if np.allclose(C_ref, C_trick, atol=1e-5) else '❌'}")
print()

# Show each variant and verify
variants = [
    ("C = A @ B",           lambda: A_row @ B_row,          "direct (row-major reference)"),
    ("C = (Bᵀ×Aᵀ)ᵀ",      lambda: (B_row.T @ A_row.T).T,  "transpose trick (cuBLAS calls)"),
    ("C = (B.T@A.T).T",    lambda: (B_row.T @ A_row.T).T,  "same as above, numpy notation"),
]
print(f"  All variants produce identical output:")
for name, fn, note in variants:
    result = fn()
    ok = np.allclose(result, C_ref, atol=1e-5)
    print(f"    {name:<20}  match={str(ok):<5}  {note}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Submatrix GEMM via lda
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Submatrix GEMM: lda > M (Strided Allocation)")
print("━" * 68)
print()

print("  Use case: only compute GEMM on the top-left M×K block of a larger buffer.")
print("  The full buffer has shape (M_full × K_full); we use M rows, K cols.")
print()

M_full, K_full = 8, 8
M_sub,  K_sub  = 4, 4
N_sub          = 4

# Create a full buffer with identifiable values
full_buffer = np.arange(1, M_full * K_full + 1, dtype=float).reshape(K_full, M_full).T
# This is column-major with lda=M_full

print(f"  Full buffer ({M_full}×{K_full}), column-major, lda={M_full}:")
for row in range(M_full):
    print(f"    row {row}: {full_buffer[row, :K_full].astype(int).tolist()}")
print()

A_sub = full_buffer[:M_sub, :K_sub]
B_sub = np.random.randn(K_sub, N_sub).astype(float)
C_sub = A_sub @ B_sub

print(f"  Sub-matrix A[0:{M_sub}, 0:{K_sub}]:")
for row in range(M_sub):
    print(f"    {A_sub[row, :].astype(int).tolist()}")
print()
print(f"  cuBLAS call with M={M_sub}, N={N_sub}, K={K_sub}, lda={M_full}:")
print(f"    A_ptr = &full_buffer[0,0]  (same base pointer as full buffer)")
print(f"    lda   = {M_full}           (stride of the FULL buffer, not the sub-matrix)")
print(f"    cuBLAS reads A_ptr + col * {M_full} + row for column 0..{K_sub-1}")
print(f"    Rows {M_sub}..{M_full-1} of each column are SKIPPED automatically.")
print()

# Verify: simulate cuBLAS column-major access with lda=M_full
reconstructed_A = np.array([[full_buffer[row, col]
                               for col in range(K_sub)]
                              for row in range(M_sub)])
print(f"  Reconstructed sub-matrix from full buffer with lda={M_full}:")
for row in range(M_sub):
    print(f"    {reconstructed_A[row].astype(int).tolist()}")
print(f"  Matches A_sub: {'✅' if np.allclose(reconstructed_A, A_sub) else '❌'}")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Transposed GEMM variants and their memory access patterns
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 4 — CUBLAS_OP_N vs CUBLAS_OP_T: Memory Access Patterns")
print("━" * 68)
print()

M_t, N_t, K_t = 4, 4, 4
A_t = np.random.randn(M_t, K_t).astype(np.float32)
B_t = np.random.randn(K_t, N_t).astype(np.float32)
B_T = B_t.T  # shape (N, K)

print(f"  A shape: ({M_t}×{K_t})   B shape: ({K_t}×{N_t})")
print()

ops_table = [
    ("OP_N, OP_N", A_t,    B_t,   False, False, "A × B"),
    ("OP_T, OP_N", A_t.T,  B_t,   True,  False, "Aᵀ × B"),
    ("OP_N, OP_T", A_t,    B_T.T, False, True,  "A × Bᵀ  (B stored as Bᵀ)"),
    ("OP_T, OP_T", A_t.T,  B_T.T, True,  True,  "Aᵀ × Bᵀ"),
]

print(f"  {'Op config':<14}  {'M':>4}  {'N':>4}  {'K':>4}  {'lda':>5}  "
      f"{'ldb':>5}  {'Operation':>16}  {'C[0,0] result'}")
print("  " + "─" * 72)

for opname, A_arg, B_arg, ta, tb, desc in ops_table:
    # Actual dimensions as cuBLAS sees them
    m_arg = M_t if not ta else K_t
    n_arg = N_t if not tb else K_t
    k_arg = K_t if not ta else M_t
    # NumPy equivalent
    A_eff = A_t.T if ta else A_t
    B_eff = B_t.T if tb else B_t
    # Compute
    try:
        C_res = A_eff @ B_eff
        c00   = C_res[0, 0]
        print(f"  {opname:<14}  {m_arg:>4}  {n_arg:>4}  {k_arg:>4}  "
              f"{A_t.shape[0]:>5}  {B_t.shape[0]:>5}  {desc:>16}  {c00:.4f}")
    except Exception as e:
        print(f"  {opname:<14}  shape mismatch: {e}")

print()
print("  All four op combinations are valid cuBLAS calls.")
print("  OP_T means: 'transpose this matrix before using it'.")
print("  lda always refers to the SIZE of the STORED matrix (before transposing).")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Batched GEMM Simulator — Pointer Array vs Strided Batch & Attention": {
        "description": (
            "Implement both batched GEMM variants (pointer array and strided batch) "
            "from scratch. Verify they produce identical results. Map multi-head "
            "attention Q@K and scores@V exactly to strided batched GEMM calls with "
            "the correct stride, lda, and transpose configuration. Show throughput "
            "scaling as batch_count increases and explain the SM utilisation model."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  BATCHED GEMM SIMULATOR — Pointer Array, Strided & Attention Mapping")
print("=" * 68)
print()

np.random.seed(99)


# ─────────────────────────────────────────────────────────────────────
# Reference implementations
# ─────────────────────────────────────────────────────────────────────

def batched_gemm_pointer_array(A_list, B_list, alpha=1.0, beta=0.0, C_list=None):
    """
    Simulate cublasSgemmBatched: list of independent (A_i, B_i) → C_i.
    Each A_i: (M, K), B_i: (K, N).
    """
    batch = len(A_list)
    M, K = A_list[0].shape
    K2, N = B_list[0].shape
    assert K == K2, f"Inner dim mismatch: {K} vs {K2}"
    C_out = []
    for i in range(batch):
        c_in = C_list[i] if C_list is not None else np.zeros((M, N))
        C_out.append(alpha * (A_list[i] @ B_list[i]) + beta * c_in)
    return C_out

def batched_gemm_strided(A, B, strideA, strideB, strideC,
                         M, N, K, batch_count,
                         alpha=1.0, beta=0.0, C=None):
    """
    Simulate cublasSgemmStridedBatched.
    A: flat array of length strideA * batch_count
    B: flat array of length strideB * batch_count
    C: flat array of length strideC * batch_count (output)
    """
    C_out = np.zeros(strideC * batch_count)
    if C is not None:
        C_out[:] = C

    for i in range(batch_count):
        a_start = i * strideA
        b_start = i * strideB
        c_start = i * strideC

        A_i = A[a_start: a_start + M * K].reshape(M, K)
        B_i = B[b_start: b_start + K * N].reshape(K, N)

        c_block = C_out[c_start: c_start + M * N].reshape(M, N)
        result  = alpha * (A_i @ B_i) + beta * c_block
        C_out[c_start: c_start + M * N] = result.ravel()

    return C_out


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Verify both variants produce identical results
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Pointer Array vs Strided Batch: Numerical Equivalence")
print("━" * 68)
print()

M, N, K    = 16, 16, 32
batch_count = 8

# Generate batch
A_list = [np.random.randn(M, K).astype(np.float32) for _ in range(batch_count)]
B_list = [np.random.randn(K, N).astype(np.float32) for _ in range(batch_count)]

# Pack into strided arrays (row-major flat)
A_strided = np.concatenate([a.ravel() for a in A_list])
B_strided = np.concatenate([b.ravel() for b in B_list])
strideA, strideB, strideC = M * K, K * N, M * N

# Run both
C_ptr    = batched_gemm_pointer_array(A_list, B_list)
C_stride = batched_gemm_strided(A_strided, B_strided,
                                 strideA, strideB, strideC,
                                 M, N, K, batch_count)

print(f"  M={M}, N={N}, K={K}, batch={batch_count}")
print(f"  strideA={strideA}  strideB={strideB}  strideC={strideC}")
print()

all_match = True
for i in range(batch_count):
    ptr_result    = C_ptr[i]
    stride_result = C_stride[i * strideC: (i+1) * strideC].reshape(M, N)
    match = np.allclose(ptr_result, stride_result, atol=1e-5)
    if not match:
        all_match = False
    if i < 3 or i == batch_count - 1:
        print(f"  batch[{i}]: pointer[0,0]={ptr_result[0,0]:.4f}  "
              f"strided[0,0]={stride_result[0,0]:.4f}  "
              f"match={'✅' if match else '❌'}")
    elif i == 3:
        print(f"  ...")

ok_str = '✅' if all_match else '❌'
print(f"  All {batch_count} batches match: {ok_str}")
print()
print("  WHEN TO PREFER EACH:")
print("    Pointer array:   ragged batches, non-contiguous inputs (e.g., sampled KV cache)")
print("    Strided batch:   contiguous 3D tensors (attention, transformer blocks)")
print("    Strided is faster: no pointer dereference overhead, better memory access pattern")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Multi-head attention exact GEMM mapping
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Multi-Head Attention → Strided Batched GEMM")
print("━" * 68)
print()

# Attention parameters
BATCH_SIZE = 2
N_HEADS    = 4
SEQ_LEN    = 8
HEAD_DIM   = 16

print(f"  Attention config: B={BATCH_SIZE}, H={N_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}")
print(f"  Q/K/V shape: [{BATCH_SIZE}, {N_HEADS}, {SEQ_LEN}, {HEAD_DIM}]")
print()

# Generate Q, K, V tensors (B, H, S, D)
Q = np.random.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)
K_attn = np.random.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)
V_attn = np.random.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)

# Reference: full attention (loop over batches and heads)
scale = 1.0 / math.sqrt(HEAD_DIM)
scores_ref = np.einsum('bhsd,bhtd->bhst', Q, K_attn) * scale
# Softmax
scores_sm  = np.exp(scores_ref - scores_ref.max(axis=-1, keepdims=True))
scores_sm /= scores_sm.sum(axis=-1, keepdims=True)
out_ref    = np.einsum('bhst,bhtd->bhsd', scores_sm, V_attn)

# ── STEP 1: Q @ Kᵀ as strided batched GEMM ──────────────────────────
print("  STEP 1: Scores = Q @ Kᵀ / sqrt(D)")
print(f"  Each head: (S×D) @ (D×S) = (S×S)")
print(f"  batchCount = B × H = {BATCH_SIZE * N_HEADS}")
print()
print(f"  cuBLAS strided batched GEMM parameters:")
print(f"    transa = CUBLAS_OP_N  (Q stored as S×D, treated as-is)")
print(f"    transb = CUBLAS_OP_T  (K stored as S×D, transposed to D×S)")
print(f"    m = {SEQ_LEN}  (rows of output = rows of Q)")
print(f"    n = {SEQ_LEN}  (cols of output = rows of K before T)")
print(f"    k = {HEAD_DIM}  (inner dim = cols of Q = cols of K)")
print(f"    lda = {HEAD_DIM}  (leading dim of Q: contiguous along D)")
print(f"    ldb = {HEAD_DIM}  (leading dim of K: contiguous along D)")
print(f"    ldc = {SEQ_LEN}  (leading dim of scores output)")
print(f"    strideA = S×D = {SEQ_LEN*HEAD_DIM}  (next head in Q)")
print(f"    strideB = S×D = {SEQ_LEN*HEAD_DIM}  (next head in K)")
print(f"    strideC = S×S = {SEQ_LEN*SEQ_LEN}  (next head's score matrix)")
print()

# Simulate via strided batched GEMM
batch_total = BATCH_SIZE * N_HEADS
Q_flat = Q.reshape(batch_total, SEQ_LEN, HEAD_DIM)
K_flat = K_attn.reshape(batch_total, SEQ_LEN, HEAD_DIM)
V_flat = V_attn.reshape(batch_total, SEQ_LEN, HEAD_DIM)

# Q @ Kᵀ  for all heads
scores = np.einsum('bsd,btd->bst', Q_flat, K_flat) * scale  # (batch, S, S)

# Compare with reference
scores_ref_flat = scores_ref.reshape(batch_total, SEQ_LEN, SEQ_LEN)
scores_match = np.allclose(scores, scores_ref_flat, atol=1e-5)
print(f"  Scores match reference: {'✅' if scores_match else '❌'}")
print(f"  scores[0, :3, :3] =")
for row in range(3):
    print(f"    {scores[0, row, :3].round(4).tolist()}")
print()

# ── STEP 2: softmax(scores) @ V ──────────────────────────────────────
print("  STEP 2: Output = softmax(Scores) @ V")
print(f"  Each head: (S×S) @ (S×D) = (S×D)")
print()
print(f"  cuBLAS strided batched GEMM parameters:")
print(f"    transa = CUBLAS_OP_N  (scores S×S)")
print(f"    transb = CUBLAS_OP_N  (V stored as S×D)")
print(f"    m = {SEQ_LEN}  n = {HEAD_DIM}  k = {SEQ_LEN}")
print(f"    lda = {SEQ_LEN}  ldb = {HEAD_DIM}  ldc = {HEAD_DIM}")
print(f"    strideA = {SEQ_LEN*SEQ_LEN}  strideB = {SEQ_LEN*HEAD_DIM}  "
      f"strideC = {SEQ_LEN*HEAD_DIM}")
print()

# Apply softmax
scores_sm_flat = np.exp(scores - scores.max(axis=-1, keepdims=True))
scores_sm_flat /= scores_sm_flat.sum(axis=-1, keepdims=True)
out = np.einsum('bst,btd->bsd', scores_sm_flat, V_flat)  # (batch, S, D)
out_ref_flat = out_ref.reshape(batch_total, SEQ_LEN, HEAD_DIM)
out_match = np.allclose(out, out_ref_flat, atol=1e-5)
print(f"  Output matches reference: {'✅' if out_match else '❌'}")
print(f"  out[0, :2, :4] =")
for row in range(2):
    print(f"    {out[0, row, :4].round(4).tolist()}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Throughput model — SM utilisation vs batch_count
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — SM Utilisation vs Batch Count (Small GEMM Analysis)")
print("━" * 68)
print()

print("  How well can cuBLAS utilise H100's 132 SMs for a single small GEMM?")
print()

SM_COUNT         = 132
WARPS_PER_SM     = 64
TILE_M, TILE_N   = 128, 128  # typical tile size per threadblock
BLOCK_WARPS      = 4         # warps per threadblock (128×128 tile)

def sm_utilisation(m, n, k, batch=1, tile_m=TILE_M, tile_n=TILE_N):
    """
    Estimate SM utilisation for a batched GEMM.
    n_blocks = ceil(M/tile_M) * ceil(N/tile_N) * batch
    SM utilisation = min(n_blocks / SM_COUNT, 1.0)
    """
    blocks_per_head = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    total_blocks    = blocks_per_head * batch
    sm_util         = min(total_blocks / SM_COUNT, 1.0) * 100
    return blocks_per_head, total_blocks, sm_util

print(f"  H100: {SM_COUNT} SMs, tile {TILE_M}×{TILE_N}, {BLOCK_WARPS} warps/block")
print()
print(f"  Attention score GEMM: S×D @ D×S = S×S   (per head, varying batch×heads)")
SEQ, DIM = 512, 64
print(f"  S={SEQ}, D={DIM}")
print()
print(f"  {'batchCount (B×H)':>20}  {'Blocks/head':>12}  {'Total blocks':>14}  "
      f"{'SM util':>8}  {'Verdict'}")
print("  " + "─" * 64)

for bc in [1, 4, 8, 16, 32, 64, 128, 256]:
    bph, tot, sm_util = sm_utilisation(SEQ, SEQ, DIM, batch=bc)
    if sm_util >= 95:
        verdict = "✅ fully saturated"
    elif sm_util >= 60:
        verdict = "⚠  mostly saturated"
    elif sm_util >= 30:
        verdict = "⚠  partial utilisation"
    else:
        verdict = "❌ underutilised"
    print(f"  {bc:>20}  {bph:>12}  {tot:>14}  {sm_util:>7.1f}%  {verdict}")

print()
print(f"  With S=512, D=64: need batch×heads ≥ 33 to saturate all {SM_COUNT} SMs.")
print(f"  Llama-2-7B (32 heads, batch=2): B×H = 64 → 97% SM utilisation. ✅")
print(f"  Single-request inference (B=1, H=32): only 32 blocks → 24% SM util. ❌")
print(f"  Fix: continuous batching (vLLM) → aggregate many requests → high B×H.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · cublasLt Epilogue Fusion — Bias, ReLU, GELU & Backward Passes": {
        "description": (
            "Simulate the cublasLt epilogue fusion mechanism for GEMM + bias + ReLU "
            "and GEMM + bias + GELU. Show the unfused (two-kernel) vs fused "
            "(one-kernel) HBM traffic model. Compute the memory savings and speedup "
            "for typical transformer FFN shapes. Implement the backward epilogue "
            "(dReLU, dGELU) fusion pattern. Show how the bias pointer is embedded "
            "in the matmul descriptor."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  cublasLt EPILOGUE FUSION — Bias, ReLU, GELU & Backward Passes")
print("=" * 68)
print()

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# Epilogue implementations
# ─────────────────────────────────────────────────────────────────────

def gelu(x):
    """Exact GELU (CUBLASLT_EPILOGUE_GELU uses tanh approximation)."""
    return x * 0.5 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi)
                                     * (x + 0.044715 * x**3)))

def dgelu(x, dout):
    """dGELU: gradient of GELU w.r.t. x, times upstream gradient."""
    t = np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
    dtdx = (1 - t**2) * math.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * x**2)
    return dout * (0.5 * (1 + t) + x * 0.5 * dtdx)

EPILOGUES = {
    "DEFAULT":          {"desc": "D = alpha*A*B + beta*C",
                         "fn": lambda ab, bias: ab},
    "BIAS":             {"desc": "D = A*B + bias",
                         "fn": lambda ab, bias: ab + bias},
    "RELU_BIAS":        {"desc": "D = max(0, A*B + bias)",
                         "fn": lambda ab, bias: np.maximum(0, ab + bias)},
    "GELU_BIAS":        {"desc": "D = GELU(A*B + bias)",
                         "fn": lambda ab, bias: gelu(ab + bias)},
    "GELU":             {"desc": "D = GELU(A*B)",
                         "fn": lambda ab, bias: gelu(ab)},
}


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness of each epilogue
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Epilogue Outputs: All Five Variants")
print("━" * 68)
print()

M_ep, N_ep, K_ep = 4, 8, 16
A_ep = np.random.randn(M_ep, K_ep).astype(np.float32)
B_ep = np.random.randn(K_ep, N_ep).astype(np.float32)
bias = np.random.randn(N_ep).astype(np.float32)  # bias broadcast over rows (M)

AB = A_ep @ B_ep

print(f"  M={M_ep}, N={N_ep}, K={K_ep}   bias shape: ({N_ep},) broadcast over M rows")
print()
print(f"  {'Epilogue':<16}  {'D[0, :4]':<40}  {'Description'}")
print("  " + "─" * 72)

for name, ep in EPILOGUES.items():
    D = ep["fn"](AB, bias)
    print(f"  {name:<16}  {str(D[0, :4].round(3).tolist()):<40}  {ep['desc']}")

print()
print("  The BIAS vector has shape (N,) and is broadcast across all M rows.")
print("  cuBLAS reads it once from HBM and accumulates into each row of D.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: HBM traffic model — fused vs unfused
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — HBM Traffic: Fused vs Unfused (Two-Kernel) Pipeline")
print("━" * 68)
print()

H100_HBM_BW_GBS = 3350.0

def hbm_traffic_unfused(M, N, K, dtype_bytes):
    """
    UNFUSED: GEMM kernel + elementwise kernel (bias+ReLU).
    GEMM writes output D (M×N) to HBM.
    Elementwise kernel reads D, reads bias, writes final D.
    """
    gemm_read   = (M * K + K * N) * dtype_bytes   # A + B
    gemm_write  = M * N * dtype_bytes              # GEMM output (intermediate)
    elt_read    = M * N * dtype_bytes + N * dtype_bytes  # D_intermediate + bias
    elt_write   = M * N * dtype_bytes              # final output
    total = gemm_read + gemm_write + elt_read + elt_write
    return total, gemm_write + elt_read  # extra traffic = intermediate roundtrip

def hbm_traffic_fused(M, N, K, dtype_bytes):
    """
    FUSED: GEMM + epilogue in one kernel.
    Writes final D directly. No intermediate tensor.
    """
    gemm_read  = (M * K + K * N) * dtype_bytes
    bias_read  = N * dtype_bytes
    out_write  = M * N * dtype_bytes
    total = gemm_read + bias_read + out_write
    return total, 0

print("  Scenario: FFN gate projection (D_FF × D_model → activation output)")
print()

# Typical Llama-2-7B FFN gate projection
ffn_shapes = [
    ("FFN gate (1-token inf)",  1,      11008, 4096),
    ("FFN gate (batch=32)",     32,     11008, 4096),
    ("FFN gate (train, M=512)", 512,    11008, 4096),
    ("FFN gate (train, M=2048)",2048,   11008, 4096),
    ("Large GEMM M=N=K=4096",   4096,   4096,  4096),
]

DTYPE_BYTES = 2  # FP16

print(f"  {'Shape (M, N, K)':<30}  {'Unfused (GB)':>13}  "
      f"{'Fused (GB)':>11}  {'Saved (GB)':>11}  {'Speedup est.':>13}")
print("  " + "─" * 78)

for name, M_f, N_f, K_f in ffn_shapes:
    uf_total, uf_extra = hbm_traffic_unfused(M_f, N_f, K_f, DTYPE_BYTES)
    f_total,  _        = hbm_traffic_fused(M_f, N_f, K_f, DTYPE_BYTES)
    saved     = (uf_total - f_total) / 1e9
    sx_est    = uf_total / f_total

    # Time estimate at H100 HBM peak
    t_uf_us = uf_total / (H100_HBM_BW_GBS * 1e9) * 1e6
    t_f_us  = f_total  / (H100_HBM_BW_GBS * 1e9) * 1e6

    print(f"  {name:<30}  {uf_total/1e9:>13.3f}  "
          f"{f_total/1e9:>11.3f}  {saved:>11.3f}  {sx_est:>12.2f}×")

print()
print("  The saved GB = two roundtrips of the M×N intermediate tensor.")
print("  For large M×N (training): fusion saves 2× the output size in HBM reads/writes.")
print("  Speedup is most pronounced when the kernel is memory-bound (small M).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Backward epilogue — dReLU and dGELU fusion
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Backward Epilogue: DRELU / DGELU Fusion")
print("━" * 68)
print()

print("  Forward: D = GELU(A@B + bias)")
print("  Backward: need dA = dD_effective @ Bᵀ and dB = Aᵀ @ dD_effective")
print("  where dD_effective = dGELU(preactivation) * upstream_dD")
print()
print("  UNFUSED backward:")
print("    Step 1: elementwise dGELU(preact) → writes dD_effective to HBM")
print("    Step 2: GEMM dA = dD_effective @ Bᵀ          ← reads dD_effective")
print("    Step 3: GEMM dB = Aᵀ @ dD_effective          ← reads dD_effective AGAIN")
print("    Total: dD_effective written once, read twice. 3× M×N HBM traffic.")
print()
print("  FUSED backward (CUBLASLT_EPILOGUE_DGELU):")
print("    cublasLt GEMM computes dD_effective on-the-fly from preactivation aux tensor.")
print("    The aux tensor (preactivation values) was saved during forward pass.")
print("    CUBLASLT_EPILOGUE_GELU_AUX: forward epilogue saves preact to aux.")
print("    CUBLASLT_EPILOGUE_DGELU:    backward epilogue reads aux, fuses dGELU.")
print("    Result: no explicit dD_effective tensor. 1× M×N HBM traffic saved.")
print()

# Numerical verification of backward
M_bk, N_bk, K_bk = 6, 8, 12
A_bk  = np.random.randn(M_bk, K_bk).astype(np.float32)
B_bk  = np.random.randn(K_bk, N_bk).astype(np.float32)
bias_bk = np.random.randn(N_bk).astype(np.float32)
dout_bk = np.random.randn(M_bk, N_bk).astype(np.float32)

# Forward
preact_bk = A_bk @ B_bk + bias_bk
D_bk      = gelu(preact_bk)

# Backward (reference)
dpreact = np.vectorize(dgelu)(preact_bk, dout_bk)
dA_ref  = dpreact @ B_bk.T
dB_ref  = A_bk.T @ dpreact
dbias   = dpreact.sum(axis=0)

print(f"  Numerical check (M={M_bk}, N={N_bk}, K={K_bk}):")
print(f"  dpreact (first row, dGELU × upstream): {dpreact[0, :4].round(4).tolist()}")
print(f"  dA[0,:3] = {dA_ref[0, :3].round(4).tolist()}")
print(f"  dB[0,:3] = {dB_ref[0, :3].round(4).tolist()}")
print(f"  dbias[:4] = {dbias[:4].round(4).tolist()}")
print()

# HBM savings from backward fusion
dout_bytes = M_bk * N_bk * 4   # FP32 upstream gradient
deff_bytes = M_bk * N_bk * 4   # dD_effective intermediate

print(f"  HBM traffic for backward on M={M_bk}, N={N_bk}, K={K_bk}:")
print(f"    Unfused: write dD_eff ({deff_bytes} B) + read twice ({2*deff_bytes} B)")
print(f"             = {3*deff_bytes} B intermediate traffic")
print(f"    Fused:   preact aux already in registers / SMEM")
print(f"             = 0 B intermediate traffic")
print(f"    Saving:  {3*deff_bytes} B  ({3*deff_bytes/(M_bk*N_bk*4):.1f}× output size)")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Descriptor configuration summary
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — cublasLt Descriptor Configuration Guide")
print("━" * 68)
print()

configs = [
    ("Standard FP32 GEMM",
     "CUBLAS_COMPUTE_32F", "CUDA_R_32F", "CUDA_R_32F", "DEFAULT",
     "classic cublasSgemm equivalent"),
    ("FP16 TC GEMM, FP32 accum",
     "CUBLAS_COMPUTE_32F", "CUDA_R_16F", "CUDA_R_32F", "DEFAULT",
     "training: FP16 in, FP32 out"),
    ("FP16 + Bias + ReLU fusion",
     "CUBLAS_COMPUTE_32F", "CUDA_R_16F", "CUDA_R_16F", "RELU_BIAS",
     "FFN activation layer"),
    ("FP16 + GELU forward + aux",
     "CUBLAS_COMPUTE_32F", "CUDA_R_16F", "CUDA_R_16F", "GELU_AUX",
     "saves preact for backward"),
    ("FP16 + dGELU backward",
     "CUBLAS_COMPUTE_32F", "CUDA_R_16F", "CUDA_R_16F", "DGELU",
     "reads preact aux, fuses grad"),
    ("TF32 tensor core path",
     "CUBLAS_COMPUTE_32F_FAST_TF32", "CUDA_R_32F", "CUDA_R_32F", "DEFAULT",
     "FP32 input, ~10× faster"),
    ("INT8 IMMA, INT32 accum",
     "CUBLAS_COMPUTE_32I", "CUDA_R_8I", "CUDA_R_32I", "DEFAULT",
     "quantised inference"),
    ("BF16 tensor core",
     "CUBLAS_COMPUTE_32F", "CUDA_R_16BF", "CUDA_R_32F", "DEFAULT",
     "training: better range than FP16"),
]

print(f"  {'Config name':<30}  {'Compute type':<28}  {'In':>8}  "
      f"{'Out':>10}  {'Epilogue':<12}  {'Use case'}")
print("  " + "─" * 96)

for cname, ctype, in_dtype, out_dtype, epil, note in configs:
    print(f"  {cname:<30}  {ctype:<28}  {in_dtype:>8}  "
          f"{out_dtype:>10}  {epil:<12}  {note}")

print()
print("  All configurations use the same cublasLtMatmul() function.")
print("  Only the descriptor attributes change — the call site is identical.")
print("  This is the key advantage of the descriptor pattern vs separate APIs.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Algorithm Search & Workspace Tuning — Heuristic vs Exhaustive": {
        "description": (
            "Simulate the cublasLt algorithm search process: enumerate tile sizes, "
            "split-K factors, and pipeline stages for a specific GEMM shape. "
            "Model the performance of each algorithm using the roofline and "
            "compute a ranking. Show how workspace budget unlocks faster algorithms. "
            "Demonstrate the offline tuning workflow: search → benchmark → save. "
            "Quantify the typical speedup vs heuristic baseline across LLM GEMM shapes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass
from typing import List

print("=" * 68)
print("  ALGORITHM SEARCH & WORKSPACE TUNING — Heuristic vs Exhaustive")
print("=" * 68)
print()

np.random.seed(55)


# ─────────────────────────────────────────────────────────────────────
# Hardware and algorithm models
# ─────────────────────────────────────────────────────────────────────

H100_FP16_TC   = 989e12     # FLOPs/s
H100_HBM_BW    = 3.35e12    # Bytes/s
H100_L2_BW     = 12e12      # Bytes/s (L2 bandwidth estimate)
H100_SMs       = 132
H100_SMEM_PER_SM = 228_000  # bytes (228 KB)


@dataclass
class GemmAlgorithm:
    algo_id:       int
    tile_m:        int
    tile_n:        int
    tile_k:        int
    stages:        int      # software pipeline stages (1=basic, 3=async)
    splitk:        int      # split-K factor
    workspace_mb:  float    # workspace required (MB)
    description:   str

    def smem_bytes(self):
        """SMEM per threadblock for this tile configuration."""
        return (self.tile_m * self.tile_k + self.tile_k * self.tile_n) * 2 * self.stages

    def is_feasible(self, workspace_budget_mb):
        return (self.workspace_mb <= workspace_budget_mb and
                self.smem_bytes() <= H100_SMEM_PER_SM)


# Algorithm library (representative cuBLAS algorithms)
ALGO_LIBRARY = [
    GemmAlgorithm(0,   64,  64, 16, 1,  1, 0,    "small tile, single-stage"),
    GemmAlgorithm(1,  128,  64, 16, 1,  1, 0,    "medium tile, single-stage"),
    GemmAlgorithm(2,   64, 128, 16, 1,  1, 0,    "medium tile, single-stage"),
    GemmAlgorithm(3,  128, 128, 16, 1,  1, 0,    "large tile, single-stage"),
    GemmAlgorithm(4,  128, 128, 32, 2,  1, 0,    "large tile, double-buffer"),
    GemmAlgorithm(5,  256, 128, 32, 2,  1, 4,    "XL tile, double-buffer"),
    GemmAlgorithm(6,  128, 128, 64, 3,  1, 8,    "large tile, async pipeline"),
    GemmAlgorithm(7,  256, 128, 64, 3,  1, 16,   "XL tile, async pipeline"),
    GemmAlgorithm(8,  128,  64, 32, 2,  2, 8,    "split-K=2, medium tile"),
    GemmAlgorithm(9,  128,  64, 32, 2,  4, 16,   "split-K=4, medium tile"),
    GemmAlgorithm(10,  64,  64, 32, 1,  8, 32,   "split-K=8, small tile"),
    GemmAlgorithm(11, 128, 128, 32, 3,  1, 32,   "large tile, 3-stage pipeline"),
    GemmAlgorithm(12, 256, 256, 32, 3,  1, 64,   "XXL tile, async (Hopper style)"),
]


def algo_perf_model(algo, M, N, K, workspace_budget_mb):
    """
    Model the effective performance of a GEMM algorithm.
    Returns: estimated time in µs, efficiency %, feasible flag.
    """
    if not algo.is_feasible(workspace_budget_mb):
        return float('inf'), 0.0, False

    flops      = 2.0 * M * N * K
    dtype_bytes = 2  # FP16

    # Compute time lower bound
    t_compute_us = flops / H100_FP16_TC * 1e6

    # Memory traffic: depends on tile size and stages
    # Tiles fill the register file → effective reuse = tile_k / element_size
    reuse_factor = max(algo.tile_k / (dtype_bytes * 8), 1.0)
    L2_traffic   = (M * K + K * N) * dtype_bytes / reuse_factor
    HBM_traffic  = (M * K + K * N) * dtype_bytes / (reuse_factor * algo.stages)
    C_traffic    = M * N * dtype_bytes  # output always written once
    total_hbm    = HBM_traffic + C_traffic

    t_memory_us  = total_hbm / H100_HBM_BW * 1e6

    # Split-K overhead: extra reduction pass
    splitk_overhead = 0.0
    if algo.splitk > 1:
        # Reduction: M*N partial sums, one atomic pass
        splitk_overhead = M * N * 4 / H100_HBM_BW * 1e6 * 0.5

    # Tile efficiency: blocks that don't fill perfectly waste cycles
    tiles_m = math.ceil(M / algo.tile_m)
    tiles_n = math.ceil(N / algo.tile_n)
    tiles_total = tiles_m * tiles_n * algo.splitk

    # Wasted work from non-divisible dimensions
    waste_m = (tiles_m * algo.tile_m - M) / (tiles_m * algo.tile_m)
    waste_n = (tiles_n * algo.tile_n - N) / (tiles_n * algo.tile_n)
    efficiency_factor = (1 - waste_m * waste_n)

    # SM wave occupancy: cost of partial SM waves
    n_waves = math.ceil(tiles_total / H100_SMs)
    wave_efficiency = tiles_total / (n_waves * H100_SMs)

    t_bound = (max(t_compute_us, t_memory_us) + splitk_overhead)
    t_actual = t_bound / (efficiency_factor * wave_efficiency + 1e-9)

    peak_eff = min(100.0, flops / (t_actual * 1e-6) / H100_FP16_TC * 100)
    return t_actual, peak_eff, True


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Algorithm ranking for a specific GEMM shape
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Algorithm Ranking for FFN Gate Projection (Training)")
print("━" * 68)
print()

M_search, N_search, K_search = 2048, 11008, 4096
WS_BUDGET = 32.0  # MB — standard production workspace

print(f"  GEMM shape: M={M_search}, N={N_search}, K={K_search}  (FP16, H100)")
print(f"  Workspace budget: {WS_BUDGET} MB")
print()

results = []
for algo in ALGO_LIBRARY:
    t_us, eff, feasible = algo_perf_model(algo, M_search, N_search, K_search, WS_BUDGET)
    if feasible:
        results.append((algo, t_us, eff))

results.sort(key=lambda x: x[1])
heuristic_t = results[3][1] if len(results) > 3 else results[0][1]  # simulate "4th best = heuristic"

print(f"  {'Rank':>5}  {'Algo ID':>8}  {'Tile M×N×K':>14}  {'Stages':>7}  "
      f"{'Split-K':>8}  {'WS MB':>6}  {'Time µs':>9}  {'Eff%':>6}  {'vs heuristic'}")
print("  " + "─" * 84)

for rank, (algo, t_us, eff) in enumerate(results):
    tile_str = f"{algo.tile_m}×{algo.tile_n}×{algo.tile_k}"
    vs_h = heuristic_t / t_us
    h_flag = "← HEURISTIC" if rank == 3 else ("← BEST" if rank == 0 else "")
    print(f"  {rank+1:>5}  {algo.algo_id:>8}  {tile_str:>14}  {algo.stages:>7}  "
          f"{algo.splitk:>8}  {algo.workspace_mb:>6.0f}  {t_us:>9.3f}  "
          f"{eff:>6.1f}%  {vs_h:.2f}×  {h_flag}")

best_t   = results[0][1]
print()
print(f"  Exhaustive search winner vs heuristic: {heuristic_t/best_t:.2f}×")
print(f"  (Exhaustive search time: ~100 iterations × {len(results)} algos × kernel time)")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Workspace budget — which algorithms unlock at each tier
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Workspace Budget: Algorithms Unlocked per Tier")
print("━" * 68)
print()

ws_budgets = [0, 4, 8, 16, 32, 64, 128, 256]

print(f"  {'WS budget (MB)':>16}  {'Algos available':>16}  "
      f"{'Best time µs':>14}  {'Best algo tile':>16}  {'vs 0MB'}")
print("  " + "─" * 68)

baseline_t = None
for ws in ws_budgets:
    avail = []
    for a in ALGO_LIBRARY:
        t_us, eff, feasible = algo_perf_model(a, M_search, N_search, K_search, ws)
        if feasible:
            avail.append((a, t_us, eff))
    avail.sort(key=lambda x: x[1])
    if not avail:
        print(f"  {ws:>16}  {'none':>16}")
        continue
    best_algo, best_t_ws, best_eff = avail[0]
    if baseline_t is None:
        baseline_t = best_t_ws
    tile_str = f"{best_algo.tile_m}×{best_algo.tile_n}×{best_algo.tile_k}"
    sx = baseline_t / best_t_ws
    print(f"  {ws:>16}  {len(avail):>16}  {best_t_ws:>14.3f}  "
          f"{tile_str:>16}  {sx:.2f}×")

print()
print("  Key insight: 32 MB workspace is the production sweet spot.")
print("  Beyond 128 MB: diminishing returns for most GEMM shapes.")
print("  0 MB: falls back to basic algorithms; 3–5× slower for large GEMM.")
print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Heuristic vs exhaustive search across all LLM GEMM shapes
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — Speedup from Exhaustive Search: All Llama-2-7B GEMMs")
print("━" * 68)
print()

llm_shapes = [
    ("Q proj  (training)",    2048,  4096, 4096),
    ("Q proj  (decode B=1)",     1,  4096, 4096),
    ("Q proj  (decode B=32)",   32,  4096, 4096),
    ("FFN gate (training)",  2048, 11008, 4096),
    ("FFN gate (decode B=1)",    1, 11008, 4096),
    ("FFN down (training)",  2048,  4096,11008),
    ("Vocab head (training)",2048, 32000, 4096),
    ("Attention QK (train)", 2048,  2048,  128),
]

HEURISTIC_RANK = 3   # assume heuristic selects 4th-best algorithm
WS_EXHAUS = 64.0     # exhaustive search uses larger workspace

print(f"  {'Shape name':<28}  {'M':>5}  {'N':>6}  {'K':>6}  "
      f"{'Heuristic µs':>14}  {'Best µs':>9}  {'Speedup':>9}")
print("  " + "─" * 76)

for name, M_l, N_l, K_l in llm_shapes:
    all_algos = []
    for algo in ALGO_LIBRARY:
        t_us, eff, feasible = algo_perf_model(algo, M_l, N_l, K_l, WS_EXHAUS)
        if feasible:
            all_algos.append((algo, t_us, eff))
    all_algos.sort(key=lambda x: x[1])

    if len(all_algos) == 0:
        continue
    best_t_l   = all_algos[0][1]
    heur_t_l   = all_algos[min(HEURISTIC_RANK, len(all_algos)-1)][1]
    sx_l       = heur_t_l / best_t_l

    print(f"  {name:<28}  {M_l:>5}  {N_l:>6}  {K_l:>6}  "
          f"{heur_t_l:>14.3f}  {best_t_l:>9.3f}  {sx_l:>9.2f}×")

print()
print("  Exhaustive search gives the largest gains for:")
print("    • Small-M decode shapes (heuristic often picks wrong split-K)")
print("    • Very wide N (large vocab) — tile shape matters more")
print("    • Attention score GEMM (square S×S shapes benefit from special tiles)")
print()
print("  Production practice: run exhaustive search ONCE per GPU type, cache results.")
print("  Tools: torch._dynamo compile with max-autotune, NVIDIA Transformer Engine,")
print("         xFormers, or hand-rolled cublasLtMatmulAlgoGetHeuristic + benchmark loop.")
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