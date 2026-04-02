"""
Sparse GEMM and GEMV — Exploiting Sparsity for Faster Matrix Computation
=========================================================================

Sparsity — the presence of zeros in a matrix — is one of the most powerful
levers for accelerating matrix computation in machine learning. When a weight
matrix is 90% zeros, a naive dense GEMM wastes 90% of its compute on
multiplications that produce zero. Sparse operations exploit this structure
to skip those zero multiplications entirely, delivering proportional speedups
in compute and memory bandwidth.

Sparsity arises naturally throughout ML:
    Pruned neural networks:  magnitude pruning zeros out small weights
    Attention masks:          causal masks, sliding-window, block-sparse
    Embedding lookups:        one-hot or sparse categorical inputs
    Graph neural networks:    adjacency matrices are overwhelmingly sparse
    Recommender systems:      user-item interaction matrices (< 0.1% dense)
    Natural language:         bag-of-words, TF-IDF, document-term matrices
    Scientific computing:     finite-element, PDE discretisation matrices

The challenge is that realising speedup from sparsity is non-trivial. Modern
hardware is optimised for dense, regular memory access. Sparse operations
introduce irregular memory access patterns, variable work per thread, and
load imbalance — all of which destroy the conditions that make dense GEMM
fast. Getting a 10× sparser matrix to run 10× faster requires careful choice
of sparse format, algorithm, and hardware target.

This module covers: sparse formats (CSR, CSC, COO, BSR, BCSR, ELL, hybrid),
the arithmetic intensity analysis of sparse operations, hardware-accelerated
sparse kernels (cuSPARSE, torch.sparse, NVIDIA Ampere 2:4 sparsity), pruning
strategies to create exploitable sparsity, and the full performance analysis
needed to decide when sparse is actually faster than dense.

"""

import textwrap
import re

TOPIC_NAME   = "Sparse GEMM and GEMV — Exploiting Sparsity"
DISPLAY_NAME = "15b · Sparse GEMM/GEMV"
ICON         = "🕸️"
SUBTITLE     = "Sparse Formats, cuSPARSE, 2:4 Sparsity, Pruning, and Performance Analysis"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE SPARSITY OPPORTUNITY AND THE DENSITY CHALLENGE

### Defining Sparsity

    Sparsity (s) = fraction of elements that are zero:
        s = |{(i,j) : A[i,j] = 0}| / (M × N)

    Density (d) = 1 - s = fraction of non-zero elements
    NNZ = number of non-zeros = d × M × N

    Common sparsity levels in practice:
        Neural network weights after pruning:   50–99% sparse (typical 80–90%)
        Attention masks (causal LLM):           50% sparse for seq=512
        Sliding-window attention (Longformer):  > 99% sparse for long sequences
        Graph adjacency matrices:               > 99.9% sparse
        Recommendation sparse features:         > 99.99% sparse
        Unstructured neural network (lottery):  typically 80–95%

### The Theoretical Speedup

    If a matrix has density d and you can perfectly skip zeros:
        Compute time:    d × (dense time)          → 1/d speedup
        Memory traffic:  d × (dense memory) + index_overhead

    The critical question: can you ACTUALLY skip zeros?

    Dense GEMM: all elements are loaded and computed regardless of value.
        Hardware doesn't know which elements are zero at runtime.
        The only way to skip is to use a sparse FORMAT that explicitly
        tracks which elements are non-zero.

    Sparse GEMM: only accesses and computes on NNZ elements.
        Index overhead: must store ROW/COLUMN indices for each NNZ.
        This overhead can dominate when density is high (> 10–20%).

### When Sparse Is Actually Faster Than Dense

    The crossover point depends on format overhead and hardware:

    For GPU (cuSPARSE SpMM):
        Sparse beats dense when: d < ~10–20% (s > 80–90%)
        Below this threshold: index overhead + irregular access dominates.

    For CPU (MKL SpMM):
        Similar crossover: d < ~10–15%

    For structured sparsity (NVIDIA 2:4):
        Sparse beats dense always (2× by hardware design), even at 50% density.
        But sparsity pattern is constrained to exactly 2 non-zeros per 4.

    The fundamental trade-off:
        DENSE:  regular memory access, no index overhead, Tensor Core eligible
        SPARSE: irregular access, index overhead, but skips zero computation

### The Irregular Memory Problem

    Dense GEMM: thread i reads A[i, k] for k=0,1,...,K sequentially.
        → sequential, predictable, coalesced, cache-friendly ✅

    Sparse GEMV (CSR format): thread i reads A[row_ptr[i]:row_ptr[i+1]]
        → each thread reads different length of data (variable work)
        → column indices are arbitrary (random access into x vector)
        → threads in a warp diverge in work amount (load imbalance)
        → non-coalesced x accesses → cache misses ❌

    This irregular access is the fundamental reason sparse is hard to
    accelerate on hardware designed for regular (dense) access patterns.


##### PART 2 — SPARSE MATRIX FORMATS

### COO — Coordinate Format

    The simplest sparse format. Stores three arrays:
        row_indices:  [i_0, i_1, ..., i_{NNZ-1}]     (row of each NNZ)
        col_indices:  [j_0, j_1, ..., j_{NNZ-1}]     (col of each NNZ)
        values:       [v_0, v_1, ..., v_{NNZ-1}]     (value of each NNZ)

    Example:
        Dense:   [[5, 0, 0],     COO row_indices: [0, 1, 1, 2]
                  [0, 8, 3],         col_indices: [0, 1, 2, 0]
                  [6, 0, 0]]         values:      [5, 8, 3, 6]

    Storage: 3 × NNZ × bpe (2 indices as INT32 + 1 value as FP32 = 12 bytes/NNZ)
    vs dense: M × N × bpe (1 value as FP32 = 4 bytes/element)
    COO wins when: 12×NNZ < 4×M×N → NNZ/MN < 1/3 → density < 33%

    Pros:  Simple. Easy to build (just append triplets).
    Cons:  Inefficient for SpMV/SpMM — row is repeated (wasted storage).
           Requires sorting for efficient access patterns.
           Not supported in most fast kernels (CSR is preferred).
    Use:   Intermediate format for assembly, before converting to CSR/CSC.

### CSR — Compressed Sparse Row

    The most widely used sparse format for ML and scientific computing.

    Three arrays:
        row_ptr:     [r_0, r_1, ..., r_M]        length M+1, int32
        col_indices: [j_0, j_1, ..., j_{NNZ-1}] length NNZ, int32
        values:      [v_0, v_1, ..., v_{NNZ-1}] length NNZ, float

    Semantics:
        Row i's non-zeros are at positions row_ptr[i] to row_ptr[i+1]-1
        in the col_indices and values arrays.
        row_ptr[0] = 0, row_ptr[M] = NNZ.

    Example (same matrix as COO):
        Dense: [[5,0,0],[0,8,3],[6,0,0]]
        row_ptr:     [0, 1, 3, 4]
        col_indices: [0, 1, 2, 0]
        values:      [5, 8, 3, 6]

    Reading row 1: positions row_ptr[1]=1 to row_ptr[2]-1=2
        → col_indices[1:3] = [1, 2]  → columns 1 and 2 have non-zeros
        → values[1:3]      = [8, 3]  → values 8 and 3

    Storage: (M+1)×4 + 2×NNZ×4 bytes (row_ptr + indices + values)
    vs COO: saves row_ptr vs repeated row index → (M+1)/NNZ fewer bytes for rows.

    Pros:  Efficient row access → ideal for SpMV (y = A·x)
           Row i accesses contiguous memory slice.
           Standard format for cuSPARSE, MKL, scipy.sparse.
    Cons:  Column access is O(NNZ) — slow for column-oriented operations.
           Not ideal for SpMM (matrix-matrix) — rows have variable length.
    Use:   SpMV, SpMM (row-wise operations), general-purpose sparse.

### CSC — Compressed Sparse Column

    Transpose of CSR: compress by column instead of row.

        col_ptr:     [c_0, c_1, ..., c_N]        length N+1
        row_indices: [i_0, i_1, ..., i_{NNZ-1}] length NNZ
        values:      [v_0, v_1, ..., v_{NNZ-1}] length NNZ

    Column j's non-zeros: positions col_ptr[j] to col_ptr[j+1]-1.

    Pros:  Efficient column access.
           Efficient for A^T · x (transpose SpMV).
    Cons:  Row access is O(NNZ) — slow.
    Use:   Transpose operations, column-oriented factorisation (LU, Cholesky).

### BSR — Block Sparse Row

    Like CSR but non-zeros are stored in BLOCKS of size (r × c):
        row_ptr:   same as CSR (over block rows, not element rows)
        col_indices: block column index for each non-zero block
        values:    3D array (NNZ_blocks × r × c)

    Example: r=c=2, 2×2 dense blocks at matrix positions:
        block_row_ptr:   [0, 1, 3]
        block_col_indices: [0, 0, 1]    ← which 2×2 block column
        values: [[[A00,A01],[A10,A11]], ...]   ← each is a 2×2 block

    Why BSR is important for ML:
        Dense matrix computation within each block → Tensor Core eligible!
        Block size 16×16 or 32×32: fully leverages Tensor Core instruction.
        If the matrix has block-sparse structure (many zero blocks),
        BSR preserves dense computation within non-zero blocks.

    Pros:  Regular computation (dense GEMM per block) → Tensor Core friendly.
           Better memory coalescing (block of values accessed together).
           Better index efficiency (one index per block, not per NNZ).
    Cons:  Requires matrix to have block-sparse structure.
           If blocks are partially non-zero: padding waste.
    Use:   Attention patterns (strided blocks), pruned layers with block pruning.

### ELL — ELLPACK Format

    Pads all rows to have the same number of non-zeros (max NNZ per row).

        col_indices: (M × max_nnz_per_row)  padded with -1 or 0
        values:      (M × max_nnz_per_row)  padded with 0.0

    For row i: elements are at col_indices[i, :] with values[i, :].

    Pros:  Regular memory access → no load imbalance!
           Natural for GPU (all threads do the same amount of work).
           Ideal for SIMD and vector hardware.
    Cons:  Padding waste when row nnz varies significantly.
           If one row has 100 NNZ and others have 5: 20× storage waste.
    Use:   GPU SpMV when NNZ per row is nearly uniform.

### HYB — Hybrid ELL + COO

    Combine ELL (for regular part) + COO (for outlier rows):
        ELL stores rows with ≤ k NNZ (the majority).
        COO stores the few rows with > k NNZ (the outliers).
        k is chosen to minimise total storage + padding waste.

    Pros:  Balances ELL's regularity with COO's flexibility.
           Better than pure ELL for matrices with heavy-tail row distributions.
    Cons:  More complex implementation (two sub-formats).
    Use:   cuSPARSE's default SpMV format (CUSPARSE_HYB_PARTITION_AUTO).

### BCSR / N:M Structured Sparsity (NVIDIA 2:4)

    BCSR (Block Compressed Sparse Row) with fixed block size.
    NVIDIA's 2:4 sparsity pattern: exactly 2 out of every 4 consecutive
    weights are non-zero (50% sparse by construction).

    Storage: compress 4 values to 2 values + 4 bits of mask:
        4 FP16 weights (64 bits) → 2 FP16 values (32 bits) + 4-bit mask
        Effective: 50% weight reduction + fast decode via hardware support.

    Ampere Sparse Tensor Core:
        Accepts the compressed format directly.
        Decompresses on-the-fly during MMA.
        Throughput: 2× dense Tensor Core throughput at 50% sparsity.
        Constraint: must satisfy exactly 2 NNZ per 4 consecutive elements.

    Pros:  2× guaranteed speedup (hardware path, not algorithmic).
           No format conversion overhead at inference time.
           Compatible with Tensor Core path (full precision operations).
    Cons:  Restricted sparsity pattern — must be 2:4.
           Accuracy loss for aggressive pruning beyond what 2:4 allows.
           Only available on NVIDIA Ampere+ GPUs.
    Use:   Inference speedup for pruned neural networks on A100/H100.


##### PART 3 — SPARSE ARITHMETIC INTENSITY

### Sparse GEMV Arithmetic Intensity

    Sparse GEMV (CSR): y = A · x  where A is M×N with NNZ non-zeros.

    FLOPs: 2 × NNZ  (one multiply + one add per non-zero)

    Bytes accessed (CSR):
        row_ptr:     (M+1) × 4 bytes  (integer pointers)
        col_indices: NNZ × 4 bytes    (integer column indices)
        values:      NNZ × bpe bytes  (floating-point values)
        x vector:    N × bpe bytes    (input — may be cached)
        y vector:    M × bpe bytes    (output)

    For large sparse matrix (M, N large):
        Dominant: values (NNZ × bpe) + col_indices (NNZ × 4) + x (N × bpe)

    Arithmetic intensity:
        AI_SpMV = 2×NNZ / (NNZ×(bpe+4) + N×bpe + M×bpe)
        For FP32 (bpe=4): AI ≈ 2 / (4+4) = 0.25 FLOP/byte  (values + indices)
        For FP16 (bpe=2): AI ≈ 2 / (2+4) = 0.33 FLOP/byte

    Compare to dense GEMV: AI_dense ≈ 2/bpe ≈ 0.5–1.0 FLOP/byte
    SPARSE is LOWER arithmetic intensity than dense GEMV!

    Why? The index overhead (4 bytes per NNZ for col_index) lowers AI.
    At FP32: sparse is 2× more bandwidth-hungry than dense per computation.
    But sparse does 1/d × fewer FLOPs and accesses 1/d × fewer values.

    Net effect (sparse vs dense GEMV):
        Sparse accesses: NNZ × (4+4) bytes for (values + indices) ≈ NNZ × 8
        Dense accesses:  M×N × 4 bytes
        For density d: sparse/dense ratio ≈ d × 8/4 = 2d (bandwidth ratio)
        Sparse wins bandwidth when: 2d < 1 → d < 50%

### Sparse GEMM Arithmetic Intensity

    For A sparse (M×K, density d_A), B dense (K×N):
        SpMM: Y = A · B

        FLOPs: 2 × NNZ_A × N  (each NNZ multiplied by a row of B)
        Bytes: NNZ_A × (bpe + 4) + K×N×bpe + M×N×bpe

        AI_SpMM = 2×NNZ_A×N / (NNZ_A×(bpe+4) + K×N×bpe + M×N×bpe)

    For large N (wide B matrix):
        B dominates: AI_SpMM ≈ 2×NNZ_A×N / (K×N×bpe)
                              = 2×d_A×M×K×N / (K×N×bpe)
                              = 2×d_A×M / bpe

    Compare to dense GEMM: AI_dense ≈ 2×M / (3×bpe) (for square matrices)
    SpMM AI is proportional to density: sparse is (3×d_A)× less intense than dense.
    But sparse does d_A fewer FLOPs → net: same time if AI is proportional.

    Actual crossover: sparse SpMM beats dense when d_A < 10–20%
    (index overhead + irregular access reduces this threshold).

### The Index Overhead Tax

    Every NNZ in a sparse format pays an INDEX TAX:
        CSR: 4 bytes per NNZ (col_index as int32) + amortised row_ptr
        COO: 8 bytes per NNZ (row + col as int32)
        BSR: 4 bytes per block / (r×c NNZ per block) → decreases with block size

    This tax means sparse is never truly as efficient as the FLOP reduction suggests.
    For CSR FP32: each NNZ costs 8 bytes (4 value + 4 index)
                  dense costs 4 bytes (4 value, no index)
                  → sparse needs at least 2× sparsity to break even on bandwidth.

    Practical threshold:
        Sparse wins on memory bandwidth when d × (bpe + index_bytes) < bpe
        For CSR FP32: d × 8 < 4 → d < 50%   (correct for bandwidth)
        But accounting for irregular access penalty: practical threshold d < 10–20%


##### PART 4 — NVIDIA 2:4 STRUCTURED SPARSITY

### The 2:4 Sparsity Pattern

    NVIDIA Ampere (A100) introduced hardware support for 2:4 sparsity:
        In every group of 4 consecutive elements in a row:
        EXACTLY 2 elements are non-zero, EXACTLY 2 are zero.

    Example 8-element row:  [w0, 0, w2, 0,   w4, 0, 0, w7]
    Grouped in 4s:          [w0, 0, w2, 0] | [w4, 0, 0, w7]
    Each group has exactly 2 NNZ: ✅ valid 2:4 pattern.

    Compressed storage:
        Values:   [w0, w2, w4, w7]         (4 values instead of 8)
        Metadata: [b00, b10, b00, b11]     (2-bit index per NNZ in group of 4)
                  → 4 bits per group of 4 = 50% compression of indices

    Total compression: 8 values → 4 values + 8 bits = ≈ 50% size reduction

### 2:4 Sparse Tensor Core Operation

    The A100 Sparse Tensor Core:
        Input:  A (compressed), B (dense), metadata (2-bit indices)
        Process:
            1. Decompress A using metadata: decode which 2 of 4 are non-zero
            2. Select corresponding 2 rows of B
            3. Compute 2×B_rows instead of 4×B_rows
            4. Accumulate into output C
        Result: 2× throughput vs dense Tensor Core!

    Formally:
        Dense:  C += A (full) × B         (4 multiplies per group)
        Sparse: C += A (2 values) × B[selected rows]  (2 multiplies per group)
        Speedup: 4 → 2 multiplies = 2× faster

    Hardware specs:
        Dense BF16 A100:  312 TFLOPS
        2:4 Sparse BF16:  624 TFLOPS (2× throughput)
        Dense BF16 H100:  989 TFLOPS
        2:4 Sparse BF16: 1978 TFLOPS (2× throughput on H100)

### Creating 2:4 Sparse Weights

    Pruning to 2:4 pattern:
        For each group of 4 consecutive weights in each row:
        Keep the 2 with the LARGEST absolute values.
        Set the other 2 to zero.

    This is a CONSTRAINED pruning: the pattern is fixed (2:4 per group).

    torch.ao.pruning (PyTorch sparse):
        from torch.ao.pruning import WeightNormSparsifier
        sparsifier = WeightNormSparsifier(sparsity_level=0.5,
                                          sparse_block_shape=(1, 4),
                                          zeros_per_block=2)
        sparsifier.prepare(model, config=[...])
        # Fine-tune to recover accuracy...
        sparsifier.squash_mask()  # apply the mask permanently

    Or direct numpy:
        def prune_2_4(W):
            # W: (out, in)
            W_4 = W.reshape(-1, 4)      # group into blocks of 4
            top2 = np.argsort(np.abs(W_4), axis=1)[:, -2:]  # top 2 per group
            mask = np.zeros_like(W_4); mask[np.arange(len(W_4))[:, None], top2] = 1
            return (W_4 * mask).reshape(W.shape)

### 2:4 Accuracy Impact

    Accuracy on ImageNet (ResNet-50):
        Dense FP16:     76.2% top-1
        2:4 sparse:     76.1% top-1  (0.1% loss with fine-tuning)
        Without fine-tune: 73.5% (2.7% loss — fine-tuning is essential)

    For Transformer/BERT:
        2:4 sparse with fine-tuning: < 0.5% accuracy loss for most tasks.
        Requires Sparse-aware fine-tuning (GMP or AST: Adaptive Sparse Training).

    The fine-tuning recipe:
        1. Train dense model to convergence.
        2. Apply 2:4 pruning (zero out the 2 smallest per group of 4).
        3. Fine-tune for 10–20% more epochs with 2:4 mask frozen.
        4. Convert to compressed format for inference.
        5. Result: 2× inference speedup with < 0.5% accuracy loss.


##### PART 5 — SPARSE GEMV ALGORITHMS ON GPU

### CSR SpMV: Vector and Warp Approaches

    The key challenge: rows have variable length → load imbalance.

    SCALAR CSR KERNEL (naive):
        One thread per row.
        Thread i: iterate over col_indices[row_ptr[i]:row_ptr[i+1]].
        Problem: thread 0 may have 100 NNZ, thread 1 may have 5 NNZ.
        Threads in a warp execute the same instruction → slowest thread paces all.
        Warp efficiency: 5/100 = 5% for this warp. Terrible.

    VECTOR CSR KERNEL (cuSPARSE style):
        One WARP (32 threads) per row.
        All 32 threads cooperate on one row's dot product.
        Each thread handles (row_nnz / 32) elements, then warp reduction.
        Better balance when NNZ per row > 32.
        Problem: short rows (NNZ < 32) waste threads.

    ADAPTIVE CSR (cuSPARSE csrmv_mp):
        Choose threads-per-row adaptively: 2, 4, 8, 16, or 32.
        Short rows (NNZ < 8): 2–4 threads per row.
        Long rows (NNZ > 64): 32+ threads (multiple warps, segmented reduction).
        Best practical approach for general sparse matrices.

    WARP SHUFFLE REDUCTION:
        After each thread computes its partial sum:
        __shfl_down_sync: tree reduction within the warp.
        32 threads → 16 → 8 → 4 → 2 → 1 final sum.
        No shared memory needed; very fast (1 cycle per level).

### cuSPARSE SpMV and SpMM

    cuSPARSE is NVIDIA's sparse linear algebra library.

    Key routines:
        cusparseSpMV:    sparse matrix × dense vector (SpMV, CSR format)
        cusparseSpMM:    sparse matrix × dense matrix (SpMM, CSR/BSR/COO)
        cusparseScsrgeam: sparse-sparse matrix addition
        cusparseScsrgemm: sparse-sparse matrix multiply (structural non-zeros)

    Generic API (cuSPARSE 11.0+):
        // 1. Create sparse matrix descriptor
        cusparseSpMatDescr_t A_desc;
        cusparseCreateCsr(&A_desc, M, N, NNZ,
                          row_ptr, col_indices, values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        // 2. Create dense vector descriptors
        cusparseDnVecDescr_t x_desc, y_desc;
        cusparseCreateDnVec(&x_desc, N, x_ptr, CUDA_R_32F);
        cusparseCreateDnVec(&y_desc, M, y_ptr, CUDA_R_32F);

        // 3. Query workspace size
        size_t ws_size;
        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A_desc, x_desc, &beta, y_desc,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &ws_size);
        cudaMalloc(&workspace, ws_size);

        // 4. Execute SpMV
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, A_desc, x_desc, &beta, y_desc,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, workspace);

    Algorithm choices:
        CUSPARSE_SPMV_ALG_DEFAULT:     auto-selects best algorithm
        CUSPARSE_SPMV_CSR_ALG1:        merges sort-based (good for irregular)
        CUSPARSE_SPMV_CSR_ALG2:        row-based (good for regular)

### Load Balancing: Merge-based SpMV

    The merge-based SpMV algorithm (Merrill & Garland, 2016):
        Model SpMV as a merge of two sorted sequences:
            Sequence 1: row intervals [row_ptr[0], row_ptr[1], ...]
            Sequence 2: NNZ indices [0, 1, 2, ..., NNZ-1]
        Partition work evenly by merge path position.
        Each thread processes the same total amount of "merge work."
        No load imbalance regardless of row length distribution.

    This is the state-of-the-art SpMV algorithm on GPU and is implemented
    in cuSPARSE's default algorithm for SpMV.

    Performance: typically achieves 50–80% of peak memory bandwidth
    for well-structured sparse matrices.


##### PART 6 — SPARSE IN ML: PRUNING STRATEGIES AND ATTENTION SPARSITY

### Pruning to Create Sparse Weights

    MAGNITUDE PRUNING (simplest):
        Zero out the |w| smallest-magnitude weights globally.
        Apply a binary mask M: W_sparse = W ⊙ M.
        Typically done iteratively: prune 10% at a time, fine-tune, repeat.
        Also: one-shot pruning (prune everything at once) — simpler but worse.

    GRADUAL MAGNITUDE PRUNING (GMP):
        Start with dense model. Prune to target sparsity over T steps.
        At each step t: sparsity = s_f × (1 - (1 - t/T)³)
        Cubic schedule ramps up sparsity slowly then fast.
        Re-train between pruning steps to recover accuracy.
        Standard in industry (used by NVIDIA for 2:4 sparse models).

    LOTTERY TICKET HYPOTHESIS (Frankle & Carlin, 2018):
        There exist small sparse subnetworks ("winning tickets") within
        a dense network that can train to the same accuracy as the dense network.
        Finding them: train dense, prune, reset weights to INITIAL values, retrain.
        Computationally expensive but produces highly sparse models (90%+).

    STRUCTURED PRUNING:
        Remove entire NEURONS, CHANNELS, or ATTENTION HEADS (not individual weights).
        Result: smaller dense matrix, not a sparse matrix.
        Use: when sparse hardware acceleration is unavailable.
        No sparse format needed — just smaller dense GEMM.

    UNSTRUCTURED PRUNING:
        Remove arbitrary individual weights.
        Result: irregular sparse matrix.
        Use: when sparse hardware acceleration IS available (cuSPARSE, 2:4).

### Attention Sparsity

    Attention mechanism: attention_weights = softmax(Q·K^T / √d_k) · V
    The attention matrix is (seq_len × seq_len) — quadratic in sequence length.

    CAUSAL (AUTOREGRESSIVE) ATTENTION:
        Attention mask: lower triangular — future tokens masked.
        Sparsity: 50% for long sequences (upper half zeroed).
        Standard for GPT, LLaMA, Mistral.
        Efficient implementation: FlashAttention (never materialises the matrix).
        NOT a sparse SpMM — handled by custom tiling.

    LOCAL WINDOW ATTENTION (Longformer, BigBird):
        Each token attends to only a window of w nearby tokens.
        Sparsity: 1 - w/seq_len → for seq=4096, w=128: 97% sparse.
        Implementation: custom CUDA kernels (not standard cuSPARSE).
        Speedup: O(seq × w) vs O(seq²) for full attention.

    BLOCK-SPARSE ATTENTION:
        Divide sequence into blocks of size B.
        Allow attention only between specific block pairs.
        Implementation: BSR format for attention matrix.
        Speedup: #active_blocks/total_blocks × full attention cost.

    LEARNED SPARSE ATTENTION (Reformer, Sparse Transformer):
        Train the model to produce sparse attention patterns.
        Techniques: locality-sensitive hashing (Reformer), top-k gating.
        Most effective for very long sequences (8K+).

### The Sparse Attention Hardware Problem

    Irregular sparse attention (arbitrary pattern per token):
        cuSPARSE SpMM is designed for FIXED sparsity patterns.
        If attention pattern changes per token, can't pre-compile the sparse format.
        Solution: custom CUDA kernels with dynamic pattern computation.
        Tools: Triton can write efficient block-sparse attention kernels.

    Block-sparse attention (fixed block pattern):
        If the sparsity pattern is known ahead of time (causal, local window):
        Can precompute BSR structure and use cuSPARSE SpMM.
        This is how Longformer efficiently handles long sequences.


##### PART 7 — SCIPY.SPARSE AND TORCH.SPARSE: PRACTICAL APIS

### scipy.sparse (CPU)

    scipy.sparse provides all major sparse formats for CPU computation:
        from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, bsr_matrix

    Creating CSR from dense:
        from scipy.sparse import csr_matrix
        A_sparse = csr_matrix(A_dense)          # auto-detect NNZ
        A_sparse = csr_matrix(A_dense, dtype=np.float32)

    Creating from COO triplets:
        from scipy.sparse import coo_matrix
        A = coo_matrix((values, (row_indices, col_indices)), shape=(M, N))
        A_csr = A.tocsr()   # convert to CSR for efficient SpMV

    SpMV and SpMM:
        y = A_sparse @ x    # SpMV: matrix-vector
        Y = A_sparse @ B    # SpMM: sparse-dense matrix multiply
        # Both use scipy's optimised routines (wraps MKL or OpenBLAS)

    Format conversion:
        A.tocsr(), A.tocsc(), A.tocoo(), A.tobsr(blocksize=(16,16))
        A.toarray()  → back to dense numpy array

    Statistics:
        A.nnz               → number of non-zeros
        A.density           → NNZ / (M×N)
        A.shape             → (M, N)

### torch.sparse (GPU)

    PyTorch supports sparse tensors on GPU (CUDA):

    Creating sparse tensors:
        # From indices and values (COO format)
        indices = torch.tensor([[0, 1, 2], [1, 0, 2]])  # (2, NNZ)
        values  = torch.tensor([3.0, 4.0, 5.0])
        A = torch.sparse_coo_tensor(indices, values, (3, 3))

        # From dense tensor
        A_sparse = A_dense.to_sparse()          # COO format
        A_csr    = A_dense.to_sparse_csr()      # CSR format (PyTorch 1.11+)
        A_bsr    = A_dense.to_sparse_bsr((16, 16))  # BSR with 16×16 blocks

    Sparse operations:
        y = torch.mv(A_csr, x)       # SpMV via cuSPARSE
        Y = torch.mm(A_csr, B)       # SpMM via cuSPARSE
        Y = torch.matmul(A_sparse, B)# auto-dispatch

    2:4 Structured Sparsity (Ampere+):
        from torch.ao.pruning import WeightNormSparsifier
        # After pruning to 2:4 pattern:
        A_24 = torch.to_sparse_semi_structured(A_masked)
        # A_24 uses Ampere's Sparse Tensor Core path automatically

### When to Use Which Library

    CPU dense (small/medium):   numpy (A @ B, np.dot) → OpenBLAS/MKL
    CPU sparse:                 scipy.sparse → wraps MKL sparse BLAS
    GPU dense:                  torch.matmul → cuBLAS Tensor Cores
    GPU sparse (general):       torch.sparse CSR → cuSPARSE
    GPU sparse (structured):    torch.ao.pruning 2:4 → Sparse Tensor Core
    GPU sparse (research):      Triton custom kernels


##### PART 8 — PERFORMANCE ENGINEERING: WHEN TO USE SPARSE AND HOW

### Decision Framework: Dense vs Sparse

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Scenario                              │ Recommendation              │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Density > 20%                         │ Dense GEMM (cuBLAS)         │
    │ Density 10–20%                        │ Profile both, sparse likely │
    │ Density < 10%                         │ Sparse (cuSPARSE/scipy)     │
    │ Exactly 50% sparse, Ampere GPU        │ 2:4 structured sparsity     │
    │ Block-sparse known pattern            │ BSR format                  │
    │ Many small independent GEMMs          │ Batched dense GEMM          │
    │ Causal attention                      │ FlashAttention (not sparse) │
    │ Local window attention                │ Block-sparse or Triton      │
    │ Embedding lookup (one-hot)            │ torch.nn.Embedding (scatter)│
    │ Graph adjacency matrix multiply       │ Sparse always               │
    └──────────────────────────────────────────────────────────────────────┘

### Format Selection Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Use case                    │ Best format │ Reason                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ General SpMV (variable nnz) │ CSR         │ Efficient row access    │
    │ Repeated SpMM (fixed A)     │ CSR or BSR  │ Can pack A once         │
    │ Both SpMV and SpMM          │ CSR         │ Universal format        │
    │ Block-structured matrix     │ BSR (r×c)   │ Dense blocks = TC eligible│
    │ Uniform NNZ per row         │ ELL         │ Regular = no imbalance  │
    │ Heavy-tail row distribution │ HYB (ELL+COO)│ Best of both          │
    │ Matrix assembly/modification│ COO/LIL     │ Easy to insert NNZ      │
    │ Ampere GPU inference        │ 2:4 semi-struct│ 2× hardware speedup  │
    │ Attention patterns          │ Custom/Triton│ Pattern-specific       │
    └──────────────────────────────────────────────────────────────────────┘

### Sparse Matrix-Specific Optimisations

    REORDERING: permute rows/columns to improve cache locality.
        RCM (Reverse Cuthill-McKee): reduces matrix bandwidth.
        Nested dissection: ideal for direct solvers.
        For ML: rarely needed (random sparsity doesn't have geometric structure).

    SORTING WITHIN ROWS: sort col_indices by value within each row.
        Improves cache hit rate for x[col] accesses.
        Most sparse libraries do this automatically.

    PREPROCESSING FOR REPEATED SpMV:
        If SpMV is called with the SAME matrix many times (different x):
        Analyse sparsity pattern once, build optimised execution plan.
        cuSPARSE: use cusparseSpMV with CUSPARSE_SPMV_ALG_DEFAULT
                  (auto-analyses and caches execution plan internally).

    MIXED PRECISION SpMV:
        For memory-bound SpMV: use FP16 values to halve bandwidth.
        cuSPARSE supports: FP16 values + FP32 accumulation.
        2× faster for large sparse matrices.

    PADDING AND ALIGNMENT:
        For BSR: choose block size = Tensor Core instruction size (16×16 or 32×32).
        For ELL: pad NNZ to multiple of 32 (warp size) per row.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Sparse Formats from Scratch — COO, CSR, BSR Implementation": {
        "description": (
            "Implement all major sparse formats from scratch in NumPy. "
            "COO: coordinate triplets. CSR: compressed row pointers. "
            "BSR: block compressed sparse row with dense blocks. "
            "Show storage requirements vs density and choose the best format. "
            "Implement CSR SpMV and verify against dense reference. "
            "Analyse arithmetic intensity for each format and operation."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  SPARSE FORMATS FROM SCRATCH — COO, CSR, BSR")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Building sparse formats
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — COO, CSR, BSR: building from first principles")
print("━" * 65)
print()


class SparseMatrix:
    """Base class for sparse matrix formats."""
    def __init__(self, shape):
        self.shape = shape

    @staticmethod
    def from_dense(A, tol=0.0):
        rows, cols = np.where(np.abs(A) > tol)
        vals = A[rows, cols]
        return rows.astype(np.int32), cols.astype(np.int32), vals.astype(np.float32)


class COOMatrix(SparseMatrix):
    """
    COO (Coordinate) Format.
    Stores (row, col, value) triplets for each non-zero.
    """
    def __init__(self, rows, cols, vals, shape):
        super().__init__(shape)
        order = np.lexsort((cols, rows))   # sort by (row, col)
        self.rows = rows[order].astype(np.int32)
        self.cols = cols[order].astype(np.int32)
        self.vals = vals[order].astype(np.float32)
        self.nnz  = len(vals)

    @classmethod
    def from_dense(cls, A):
        rows, cols, vals = SparseMatrix.from_dense(A)
        return cls(rows, cols, vals, A.shape)

    def to_dense(self):
        A = np.zeros(self.shape, dtype=np.float32)
        A[self.rows, self.cols] = self.vals
        return A

    def storage_bytes(self):
        return (2 * self.nnz * 4 +   # rows + cols (int32)
                self.nnz * 4)         # vals (float32)

    def spmv(self, x):
        """SpMV: y = A · x using COO (scatter-add)."""
        y = np.zeros(self.shape[0], dtype=np.float32)
        for i in range(self.nnz):
            y[self.rows[i]] += self.vals[i] * x[self.cols[i]]
        return y


class CSRMatrix(SparseMatrix):
    """
    CSR (Compressed Sparse Row) Format.
    row_ptr[i]:row_ptr[i+1] are the NNZ elements of row i.
    """
    def __init__(self, row_ptr, col_idx, vals, shape):
        super().__init__(shape)
        self.row_ptr = row_ptr.astype(np.int32)  # length M+1
        self.col_idx = col_idx.astype(np.int32)  # length NNZ
        self.vals    = vals.astype(np.float32)   # length NNZ
        self.nnz     = len(vals)

    @classmethod
    def from_dense(cls, A):
        rows, cols, vals = SparseMatrix.from_dense(A)
        M = A.shape[0]
        # Build row_ptr via histogram
        row_counts = np.bincount(rows, minlength=M)
        row_ptr    = np.zeros(M + 1, dtype=np.int32)
        row_ptr[1:] = np.cumsum(row_counts)
        return cls(row_ptr, cols, vals, A.shape)

    @classmethod
    def from_coo(cls, coo):
        M = coo.shape[0]
        row_counts = np.bincount(coo.rows, minlength=M)
        row_ptr    = np.zeros(M + 1, dtype=np.int32)
        row_ptr[1:] = np.cumsum(row_counts)
        return cls(row_ptr, coo.cols, coo.vals, coo.shape)

    def to_dense(self):
        A = np.zeros(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            for k in range(self.row_ptr[i], self.row_ptr[i+1]):
                A[i, self.col_idx[k]] = self.vals[k]
        return A

    def to_coo(self):
        rows = np.repeat(np.arange(self.shape[0]),
                          np.diff(self.row_ptr))
        return COOMatrix(rows, self.col_idx, self.vals, self.shape)

    def storage_bytes(self):
        return ((self.shape[0] + 1) * 4 +   # row_ptr (int32)
                self.nnz * 4 +               # col_idx (int32)
                self.nnz * 4)                # vals (float32)

    def nnz_per_row(self):
        return np.diff(self.row_ptr)

    def spmv(self, x):
        """Vectorised CSR SpMV: y = A · x."""
        y = np.zeros(self.shape[0], dtype=np.float32)
        for i in range(self.shape[0]):
            start, end = self.row_ptr[i], self.row_ptr[i+1]
            if start < end:
                y[i] = np.dot(self.vals[start:end], x[self.col_idx[start:end]])
        return y

    def spmv_numpy(self, x):
        """Fully vectorised SpMV using numpy.add.at (no Python loop)."""
        y = np.zeros(self.shape[0], dtype=np.float32)
        products = self.vals * x[self.col_idx]
        np.add.at(y, self.to_coo().rows, products)
        return y


class BSRMatrix(SparseMatrix):
    """
    BSR (Block Sparse Row) Format.
    Non-zeros stored as dense r×c blocks.
    """
    def __init__(self, block_row_ptr, block_col_idx, block_vals, shape, blocksize):
        super().__init__(shape)
        self.block_row_ptr = block_row_ptr.astype(np.int32)
        self.block_col_idx = block_col_idx.astype(np.int32)
        self.block_vals    = block_vals.astype(np.float32)  # (NNZ_blocks, r, c)
        self.blocksize     = blocksize
        self.r, self.c     = blocksize
        self.nnz_blocks    = len(block_col_idx)
        self.nnz           = self.nnz_blocks * self.r * self.c

    @classmethod
    def from_dense(cls, A, blocksize=(2, 2)):
        r, c  = blocksize
        M, N  = A.shape
        assert M % r == 0 and N % c == 0, "Matrix dims must be divisible by block size"
        M_b, N_b = M // r, N // c
        block_rows, block_cols, block_list = [], [], []
        for ib in range(M_b):
            for jb in range(N_b):
                block = A[ib*r:(ib+1)*r, jb*c:(jb+1)*c]
                if np.any(block != 0):   # non-zero block
                    block_rows.append(ib)
                    block_cols.append(jb)
                    block_list.append(block)
        if not block_list:
            return cls(np.zeros(M_b+1, np.int32),
                        np.array([], np.int32),
                        np.zeros((0, r, c), np.float32), A.shape, blocksize)
        block_rows = np.array(block_rows, np.int32)
        block_cols = np.array(block_cols, np.int32)
        block_vals = np.stack(block_list).astype(np.float32)
        # Compress row pointers
        row_counts = np.bincount(block_rows, minlength=M_b)
        block_row_ptr = np.zeros(M_b+1, np.int32)
        block_row_ptr[1:] = np.cumsum(row_counts)
        return cls(block_row_ptr, block_cols, block_vals, A.shape, blocksize)

    def storage_bytes(self):
        M_b = self.shape[0] // self.r
        return ((M_b + 1) * 4 +           # block_row_ptr
                self.nnz_blocks * 4 +      # block_col_idx
                self.nnz_blocks * self.r * self.c * 4)  # block values

    def spmv(self, x):
        """BSR SpMV using dense GEMV within each block."""
        M, N  = self.shape
        r, c  = self.blocksize
        M_b   = M // r
        y     = np.zeros(M, np.float32)
        for ib in range(M_b):
            for k in range(self.block_row_ptr[ib], self.block_row_ptr[ib+1]):
                jb    = self.block_col_idx[k]
                block = self.block_vals[k]         # r × c dense block
                x_seg = x[jb*c:(jb+1)*c]
                y[ib*r:(ib+1)*r] += block @ x_seg  # dense GEMV
        return y


# ─────────────────────────────────────────────────────────────────────────
# Demo: create a sparse matrix and show all formats
# ─────────────────────────────────────────────────────────────────────────
print("  Building sparse formats for a 6×6 matrix at 30% density:")
print()
np.random.seed(42)
M_, N_ = 6, 6
A_dense = np.random.randn(M_, N_).astype(np.float32)
# Create 70% sparsity
mask = np.random.rand(M_, N_) < 0.70
A_dense[mask] = 0.0
nnz_total = np.count_nonzero(A_dense)
density = nnz_total / (M_ * N_)

print(f"  Dense matrix ({M_}×{N_}), NNZ={nnz_total}, density={density:.2f}")
print()
print("  Dense A:")
for row in A_dense:
    print("   ", " ".join([f"{v:6.2f}" if v != 0 else "  0.00" for v in row]))
print()

# Build formats
coo = COOMatrix.from_dense(A_dense)
csr = CSRMatrix.from_dense(A_dense)
bsr = BSRMatrix.from_dense(A_dense, blocksize=(2, 2))

print(f"  COO format:")
print(f"    rows:    {coo.rows.tolist()}")
print(f"    cols:    {coo.cols.tolist()}")
print(f"    vals:    {[f'{v:.2f}' for v in coo.vals]}")
print(f"    Storage: {coo.storage_bytes()} bytes  (vs {M_*N_*4} dense)")
print()

print(f"  CSR format:")
print(f"    row_ptr: {csr.row_ptr.tolist()}")
print(f"    col_idx: {csr.col_idx.tolist()}")
print(f"    vals:    {[f'{v:.2f}' for v in csr.vals]}")
print(f"    Storage: {csr.storage_bytes()} bytes")
print(f"    NNZ per row: {csr.nnz_per_row().tolist()}")
print()

print(f"  BSR format (2×2 blocks):")
print(f"    block_row_ptr: {bsr.block_row_ptr.tolist()}")
print(f"    block_col_idx: {bsr.block_col_idx.tolist()}")
print(f"    NNZ blocks: {bsr.nnz_blocks}  "
      f"(some blocks have partial zeros — BSR padding)")
print(f"    Storage: {bsr.storage_bytes()} bytes")
print()

# Verify correctness: SpMV
x_test = np.random.randn(N_).astype(np.float32)
y_dense = A_dense @ x_test
y_coo   = coo.spmv(x_test)
y_csr   = csr.spmv(x_test)
y_bsr   = bsr.spmv(x_test)

print(f"  SpMV verification (y = A · x):")
print(f"    Dense reference: {y_dense.round(4)}")
print(f"    COO SpMV:        {y_coo.round(4)}  err={np.max(np.abs(y_coo-y_dense)):.2e}")
print(f"    CSR SpMV:        {y_csr.round(4)}  err={np.max(np.abs(y_csr-y_dense)):.2e}")
print(f"    BSR SpMV:        {y_bsr.round(4)}  err={np.max(np.abs(y_bsr-y_dense)):.2e}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Storage analysis across densities
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Storage requirements across density levels")
print("━" * 65)
print()

M_s, N_s = 1000, 1000
dense_bytes = M_s * N_s * 4   # float32

print(f"  Matrix size: {M_s}×{N_s} = {M_s*N_s:,} elements, "
      f"dense storage = {dense_bytes/1024:.1f} KB")
print()
print(f"  {'Density':>8} | {'NNZ':>8} | {'COO (KB)':>10} | "
      f"{'CSR (KB)':>10} | {'BSR 8×8 (KB)':>14} | {'Sparse beats dense?'}")
print(f"  {'─'*78}")

for density in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
    nnz = int(density * M_s * N_s)
    coo_b = (2 * nnz * 4 + nnz * 4) / 1024
    csr_b = ((M_s+1)*4 + nnz*4 + nnz*4) / 1024
    # BSR 8×8: each block = 64 elements, NNZ_blocks ≈ nnz/16 (avg half full)
    nnz_blocks_8 = max(1, nnz // 32)   # rough estimate
    bsr_b = ((M_s//8+1)*4 + nnz_blocks_8*4 + nnz_blocks_8*8*8*4) / 1024
    dense_b = dense_bytes / 1024

    winner = "✅" if csr_b < dense_b else "❌"
    print(f"  {density:>8.3f} | {nnz:>8,} | {coo_b:>10.1f} | "
          f"{csr_b:>10.1f} | {bsr_b:>14.1f} | "
          f"{winner} CSR {'saves' if csr_b < dense_b else 'costs'} "
          f"{abs(100*(1-csr_b/dense_b)):.0f}%")
print()
print("  CSR saves storage when density < ~33% (2 index bytes overhead per NNZ).")
print("  BSR can be LARGER than dense at low density (padding within blocks).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Arithmetic intensity analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Arithmetic intensity: sparse vs dense")
print("━" * 65)
print()

def sparse_spmv_ai(M, N, density, bpe=4, idx_bytes=4):
    """CSR SpMV arithmetic intensity."""
    nnz    = int(density * M * N)
    flops  = 2 * nnz
    # read: row_ptr (small), col_idx, vals, x (cached if small), write y
    bytes_ = (nnz * (bpe + idx_bytes)   # vals + col_idx
               + N * bpe                 # x (may be cached)
               + M * bpe)               # y write
    return flops / bytes_

def dense_spmv_ai(M, N, bpe=4):
    """Dense GEMV arithmetic intensity."""
    return 2 * M * N / ((M*N + N + M) * bpe)

A100_ridge = 156.0  # BF16
print("  Comparing CSR SpMV vs Dense GEMV arithmetic intensity:")
print(f"  A100 BF16 ridge: {A100_ridge} FLOP/byte")
print()
print(f"  {'Density':>8} | {'CSR AI':>9} | {'Dense AI':>10} | "
      f"{'CSR wins BW?':>14} | {'Speedup potential'}")
print(f"  {'─'*62}")

N_ai = 4096
for d in [0.001, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00]:
    ai_sparse = sparse_spmv_ai(N_ai, N_ai, d)
    ai_dense  = dense_spmv_ai(N_ai, N_ai)
    # CSR wins if it uses less bandwidth:
    # sparse_bytes = d * (bpe + idx) * M * N; dense_bytes = M * N * bpe
    sparse_bw_ratio = d * (4 + 4) / 4   # how much of dense bandwidth sparse uses
    csr_wins  = sparse_bw_ratio < 1.0
    speedup   = 1 / sparse_bw_ratio if csr_wins else 1 / sparse_bw_ratio
    print(f"  {d:>8.3f} | {ai_sparse:>9.3f} | {ai_dense:>10.3f} | "
          f"{'✅ YES' if csr_wins else '❌ NO':>14} | "
          f"{1/sparse_bw_ratio:.2f}× {'faster' if csr_wins else 'slower'}")
print()
print("  CSR SpMV wins bandwidth when density < ~50% (index overhead crosses over).")
print("  But irregular access penalty means practical crossover is 10–20%.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · scipy.sparse and cuSPARSE — Benchmarking Real Sparse Operations": {
        "description": (
            "Use scipy.sparse for CPU sparse operations and benchmark. "
            "Build CSR matrices at various densities and measure SpMV/SpMM. "
            "Compare sparse vs dense crossover experimentally. "
            "Show GPU sparse with torch.sparse (cuSPARSE under the hood). "
            "Demonstrate 2:4 structured sparsity with PyTorch. "
            "Build the density vs speedup curve for your hardware."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  SCIPY.SPARSE AND TORCH.SPARSE — BENCHMARKING")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: scipy.sparse CPU benchmark
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — scipy.sparse: CSR SpMV and SpMM on CPU")
print("━" * 65)
print()

try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, random as sparse_random
    import scipy
    print(f"  scipy {scipy.__version__}")
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  scipy not installed: pip install scipy")
print()

def make_sparse_csr(M, N, density, seed=42):
    """Create a random CSR matrix at given density."""
    rng = np.random.RandomState(seed)
    A_sp = sparse_random(M, N, density=density, format='csr',
                          random_state=rng, dtype=np.float32)
    return A_sp

if HAS_SCIPY:
    M_bench, N_bench = 4096, 4096
    x_bench = np.random.randn(N_bench).astype(np.float32)
    B_bench = np.random.randn(N_bench, 128).astype(np.float32)  # for SpMM
    REPS    = 20

    print(f"  Matrix: {M_bench}×{N_bench} = {M_bench*N_bench:,} elements")
    print(f"  x: ({N_bench},)    B: ({N_bench}×128)")
    print()

    # Dense baseline
    A_dense = np.random.randn(M_bench, N_bench).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(REPS): A_dense @ x_bench
    t_dense_mv = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS): A_dense @ B_bench
    t_dense_mm = (time.perf_counter() - t0) / REPS * 1000

    print(f"  Dense GEMV: {t_dense_mv:.3f}ms    Dense GEMM (K=128): {t_dense_mm:.3f}ms")
    print()
    print(f"  {'Density':>8} | {'NNZ':>9} | {'SpMV (ms)':>11} | "
          f"{'vs dense MV':>13} | {'SpMM(ms)':>10} | {'vs dense MM':>13}")
    print(f"  {'─'*74}")

    crossover_mv = None
    for density in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 0.80, 1.00]:
        if density == 1.00:
            A_test = csr_matrix(A_dense)
        else:
            A_test = make_sparse_csr(M_bench, N_bench, density)

        nnz = A_test.nnz

        # SpMV
        for _ in range(5): A_test @ x_bench
        t0 = time.perf_counter()
        for _ in range(REPS): y_sp = A_test @ x_bench
        t_mv = (time.perf_counter() - t0) / REPS * 1000

        # SpMM
        for _ in range(5): A_test @ B_bench
        t0 = time.perf_counter()
        for _ in range(REPS): Y_sp = A_test @ B_bench
        t_mm = (time.perf_counter() - t0) / REPS * 1000

        spd_mv = t_dense_mv / t_mv
        spd_mm = t_dense_mm / t_mm
        if spd_mv > 1.0 and crossover_mv is None:
            crossover_mv = density

        win_mv = "✅" if spd_mv > 1.0 else "❌"
        win_mm = "✅" if spd_mm > 1.0 else "❌"
        print(f"  {density:>8.3f} | {nnz:>9,} | {t_mv:>11.3f} | "
              f"{win_mv} {spd_mv:6.2f}× | {t_mm:>10.3f} | "
              f"{win_mm} {spd_mm:6.2f}×")

    print()
    if crossover_mv:
        print(f"  ✅ SpMV crossover (sparse faster): density ≈ {crossover_mv:.3f}")
    print("  Sparse wins at low density due to fewer FLOPs + less bandwidth.")
    print("  Dense wins at high density: regular memory access + BLAS pipelining.")
    print()

    # Show format conversion timings
    print("  Format conversion overhead:")
    A_dense_small = np.random.randn(512, 512).astype(np.float32)
    A_dense_small[np.random.rand(512, 512) > 0.05] = 0.0  # 5% dense

    t0 = time.perf_counter()
    for _ in range(100): A_csr = csr_matrix(A_dense_small)
    t_conv = (time.perf_counter() - t0) / 100 * 1000
    print(f"  Dense→CSR (512×512, 5% dense): {t_conv:.3f}ms")
    print("  Moral: pre-convert to CSR once; don't convert inside a loop!")

else:
    SCIPY_REF = """
  SCIPY.SPARSE API REFERENCE:
  from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

  # Create from dense
  A_sp = csr_matrix(A_dense)          # auto-detect NNZ
  A_sp = csr_matrix(A_dense, dtype=np.float32)

  # Create from triplets (COO-style)
  A_sp = csr_matrix((vals, (rows, cols)), shape=(M, N))

  # Create random sparse matrix
  from scipy.sparse import random as sparse_random
  A_sp = sparse_random(M, N, density=0.01, format='csr', dtype=np.float32)

  # Operations
  y = A_sp @ x           # SpMV: returns dense array
  Y = A_sp @ B           # SpMM: sparse @ dense = dense
  At = A_sp.T            # Transpose (returns CSC usually)
  A2 = A_sp + A_sp       # Sparse addition
  C_sp = A_sp @ B_sp     # Sparse @ Sparse = Sparse

  # Statistics
  A_sp.nnz               # number of non-zeros
  A_sp.density           # nnz / (M × N)

  # Format conversion
  A_csc = A_sp.tocsc()
  A_bsr = A_sp.tobsr(blocksize=(8,8))
  A_dense = A_sp.toarray()
"""
    print(SCIPY_REF)

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: torch.sparse on GPU
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — torch.sparse: GPU sparse operations (cuSPARSE)")
print("━" * 65)
print()

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    device    = "cpu"
    print("  PyTorch not installed")

if HAS_TORCH:
    M_t, N_t = 4096, 4096
    x_t      = torch.randn(N_t, device=device)
    B_t      = torch.randn(N_t, 256, device=device)  # for SpMM

    # Dense baseline
    A_dense_t = torch.randn(M_t, N_t, device=device)
    REPS_T    = 50

    def bench_t(fn, warmup=10, reps=REPS_T):
        for _ in range(warmup): fn()
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        if device == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    t_dense_mv_t = bench_t(lambda: torch.mv(A_dense_t, x_t))
    t_dense_mm_t = bench_t(lambda: torch.mm(A_dense_t, B_t))
    print(f"  Dense GEMV: {t_dense_mv_t:.3f}ms    Dense GEMM (K=256): {t_dense_mm_t:.3f}ms")
    print()

    print(f"  {'Density':>8} | {'NNZ':>9} | {'SpMV (ms)':>11} | "
          f"{'vs dense':>10} | {'SpMM (ms)':>11} | {'vs dense':>10}")
    print(f"  {'─'*65}")

    for density in [0.001, 0.01, 0.05, 0.10, 0.20, 0.50]:
        # Create sparse mask
        mask = torch.rand(M_t, N_t, device=device) < density
        A_sp = A_dense_t * mask.float()
        A_csr = A_sp.to_sparse_csr()

        nnz = int(density * M_t * N_t)

        try:
            t_mv = bench_t(lambda: torch.mv(A_csr, x_t))
            t_mm = bench_t(lambda: torch.mm(A_csr, B_t))
            spd_mv = t_dense_mv_t / t_mv
            spd_mm = t_dense_mm_t / t_mm
            w_mv   = "✅" if spd_mv > 1.0 else "❌"
            w_mm   = "✅" if spd_mm > 1.0 else "❌"
            print(f"  {density:>8.3f} | {nnz:>9,} | {t_mv:>11.3f} | "
                  f"{w_mv} {spd_mv:6.2f}× | {t_mm:>11.3f} | "
                  f"{w_mm} {spd_mm:6.2f}×")
        except Exception as e:
            print(f"  {density:>8.3f} | {nnz:>9,} | Error: {str(e)[:40]}")

    print()
    print("  Note: torch.sparse CSR → cuSPARSE internally.")
    print("  GPU cuSPARSE crossover typically needs density < 5% for SpMV speedup.")
    print("  (GPU bandwidth penalty for irregular access is larger than CPU.)")

TORCH_SPARSE_REF = """
  TORCH.SPARSE API REFERENCE:

  # Convert dense to sparse (various formats)
  A_coo = A_dense.to_sparse()                     # COO format (default)
  A_csr = A_dense.to_sparse_csr()                 # CSR format
  A_bsr = A_dense.to_sparse_bsr(blocksize=(16,16))# BSR format

  # Create from indices/values
  indices = torch.stack([rows, cols])              # (2, NNZ)
  A_coo   = torch.sparse_coo_tensor(indices, vals, (M, N))
  A_csr   = torch.sparse_csr_tensor(crow_indices, col_indices, values, (M, N))

  # Operations (dispatches to cuSPARSE on GPU)
  y = torch.mv(A_csr, x)      # SpMV: (M×N sparse) × (N,) → (M,)
  Y = torch.mm(A_csr, B)      # SpMM: (M×N sparse) × (N×K) → (M×K)
  Y = torch.matmul(A_sparse, B)  # auto-dispatch

  # 2:4 Semi-structured sparsity (Ampere+ GPU)
  from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
  # Requires: matrix has exactly 2 NNZ per 4 consecutive elements
  A_24 = to_sparse_semi_structured(A_masked)   # converts to 2:4 compressed
  # A_24 is used transparently in nn.Linear.forward (2× Tensor Core throughput)

  # Checking sparsity
  A_csr.crow_indices()    # row pointers
  A_csr.col_indices()     # column indices
  A_csr.values()          # non-zero values
"""
print(TORCH_SPARSE_REF)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · 2:4 Structured Sparsity, Pruning, and the Sparse Decision Guide": {
        "description": (
            "Implement 2:4 structured pruning from scratch. "
            "Verify the 2:4 pattern and compression format. "
            "Show how pruning accuracy compares to magnitude threshold pruning. "
            "Implement gradual magnitude pruning (GMP) schedule. "
            "Build the complete density vs speedup curve comparing all methods. "
            "Production decision guide: when to use each sparse approach."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  2:4 STRUCTURED SPARSITY, PRUNING, AND DECISION GUIDE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Implementing 2:4 pruning from scratch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — 2:4 structured sparsity: implementation and analysis")
print("━" * 65)
print()

print("  2:4 rule: in every group of 4 consecutive weights (per row),")
print("  keep exactly the 2 with the LARGEST absolute value.")
print()


def prune_24_structured(W):
    """
    Apply NVIDIA 2:4 structured sparsity to weight matrix W.
    Returns (W_sparse, mask) where mask is True for kept weights.
    W shape: (out_features, in_features)
    """
    W_orig  = W.copy()
    out_f, in_f = W.shape
    assert in_f % 4 == 0, "in_features must be divisible by 4"

    W_sparse = np.zeros_like(W)
    mask      = np.zeros_like(W, dtype=bool)

    # Reshape into groups of 4
    W_4 = W.reshape(out_f, in_f // 4, 4)   # (out, groups, 4)

    # For each group of 4, keep top-2 by magnitude
    abs_W4 = np.abs(W_4)
    top2_idx = np.argsort(abs_W4, axis=-1)[:, :, -2:]  # (out, groups, 2)

    # Build mask in grouped form
    mask_4 = np.zeros_like(W_4, dtype=bool)
    out_idx   = np.arange(out_f)[:, None, None]
    grp_idx   = np.arange(in_f // 4)[None, :, None]
    mask_4[out_idx, grp_idx, top2_idx] = True

    mask_flat    = mask_4.reshape(out_f, in_f)
    W_sparse     = W_orig * mask_flat
    return W_sparse, mask_flat


def compress_24(W_sparse, mask):
    """
    Compress a 2:4 sparse matrix to NVIDIA's format:
    - Values: only the 2 non-zeros per group of 4
    - Metadata: 2-bit position for each non-zero (which of 4 positions)
    Returns compressed_values (50% size), metadata (bits)
    """
    out_f, in_f = W_sparse.shape
    n_groups     = in_f // 4

    W_4    = W_sparse.reshape(out_f, n_groups, 4)
    mask_4 = mask.reshape(out_f, n_groups, 4)

    # Compressed values: (out_f, n_groups, 2)
    comp_vals = np.zeros((out_f, n_groups, 2), dtype=W_sparse.dtype)
    metadata  = np.zeros((out_f, n_groups, 2), dtype=np.uint8)  # 2-bit positions

    for o in range(out_f):
        for g in range(n_groups):
            positions  = np.where(mask_4[o, g])[0]  # which 2 positions are non-zero
            if len(positions) == 2:
                comp_vals[o, g, 0] = W_4[o, g, positions[0]]
                comp_vals[o, g, 1] = W_4[o, g, positions[1]]
                metadata[o, g, 0]  = positions[0]   # 2-bit (0–3)
                metadata[o, g, 1]  = positions[1]   # 2-bit (0–3)

    return comp_vals, metadata


# Demo on a small weight matrix
np.random.seed(42)
W_demo = np.random.randn(4, 8).astype(np.float32)  # 4 output, 8 input features

print(f"  Original weight matrix (4×8):")
for row in W_demo:
    print("   ", " ".join([f"{v:+6.3f}" for v in row]))
print()

W_24, mask_24 = prune_24_structured(W_demo)
print(f"  After 2:4 pruning (mask shows kept=1, pruned=0):")
for i, (row, m) in enumerate(zip(W_24, mask_24)):
    mask_str = " ".join(["█" if k else "░" for k in m])
    vals_str = " ".join([f"{v:+6.3f}" if m[j] else " 0.000" for j, v in enumerate(row)])
    print(f"  Row {i}: [{mask_str}]  {vals_str}")
print()

# Verify 2:4 pattern
print("  Verifying 2:4 pattern (exactly 2 NNZ per group of 4):")
groups_ok = True
for i, row in enumerate(mask_24):
    for g in range(len(row) // 4):
        group_nnz = mask_24[i, g*4:(g+1)*4].sum()
        if group_nnz != 2:
            print(f"  ❌ Row {i}, group {g}: has {group_nnz} NNZ (expected 2)")
            groups_ok = False
if groups_ok:
    print("  ✅ All groups have exactly 2 non-zeros")
print()

# Compression analysis
comp_vals, metadata = compress_24(W_24, mask_24)
orig_bytes  = W_demo.size * 4              # float32
comp_bytes  = comp_vals.size * 4           # compressed values
meta_bytes  = (metadata.size * 2) // 8 + 1  # 2 bits per entry → bits/8
total_comp  = comp_bytes + meta_bytes

print(f"  Compression analysis:")
print(f"    Original:    {orig_bytes:4d} bytes  ({W_demo.size} float32 values)")
print(f"    Values:      {comp_bytes:4d} bytes  ({comp_vals.size} float32, 50% fewer)")
print(f"    Metadata:    {meta_bytes:4d} bytes  ({metadata.size} 2-bit indices)")
print(f"    Total:       {total_comp:4d} bytes  ({total_comp/orig_bytes:.1%} of original)")
print(f"    Compression: {orig_bytes/total_comp:.2f}× smaller than dense")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Comparing pruning strategies
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Pruning strategies: magnitude vs structured")
print("━" * 65)
print()


def prune_magnitude_global(W, target_sparsity):
    """Remove the target_sparsity fraction of smallest |weight| globally."""
    threshold = np.percentile(np.abs(W), target_sparsity * 100)
    mask      = np.abs(W) >= threshold
    return W * mask, mask


def prune_magnitude_local(W, target_sparsity):
    """Remove the target_sparsity fraction per ROW."""
    mask = np.zeros_like(W, dtype=bool)
    for i in range(len(W)):
        thresh     = np.percentile(np.abs(W[i]), target_sparsity * 100)
        mask[i]    = np.abs(W[i]) >= thresh
    return W * mask, mask


def reconstruction_error(W_orig, W_pruned):
    """Frobenius norm relative error."""
    return np.linalg.norm(W_orig - W_pruned, 'fro') / np.linalg.norm(W_orig, 'fro')


np.random.seed(0)
W_large = np.random.randn(64, 256).astype(np.float32)

print(f"  Weight matrix: 64×256 = {64*256:,} params")
print()
print(f"  {'Method':<30} | {'Sparsity':>9} | {'Rel error':>11} | "
      f"{'Pattern':>12} | {'HW speedup'}")
print(f"  {'─'*75}")

# Magnitude global at various levels
for target_s in [0.50, 0.75, 0.90]:
    W_g, m_g   = prune_magnitude_global(W_large, target_s)
    err_g      = reconstruction_error(W_large, W_g)
    actual_s   = 1 - m_g.mean()
    print(f"  {'Magnitude (global)':30} | {actual_s:>9.2f} | {err_g:>11.4f} | "
          f"{'Unstructured':>12} | {'cuSPARSE'}")

# 2:4 structured (always 50%)
W_24l, m_24l = prune_24_structured(W_large)
err_24       = reconstruction_error(W_large, W_24l)
actual_24    = 1 - m_24l.mean()
print(f"  {'2:4 structured (top-2 per 4)':30} | {actual_24:>9.2f} | {err_24:>11.4f} | "
      f"{'2:4 pattern':>12} | {'2× TC (Ampere)'}")

# Local magnitude at 50%
W_l50, m_l50 = prune_magnitude_local(W_large, 0.50)
err_l50      = reconstruction_error(W_large, W_l50)
print(f"  {'Magnitude (per-row, 50%)':30} | {0.50:>9.2f} | {err_l50:>11.4f} | "
      f"{'Unstructured':>12} | {'cuSPARSE'}")

print()
print("  Key insights:")
print("  • 2:4 at 50% sparsity has LOWER reconstruction error than random 50%")
print("    (selects locally best 2 per group, not globally worst 50%).")
print("  • 2:4 gets 2× hardware speedup on Ampere regardless of quality.")
print("  • Random 50% sparse requires density < ~10% for real speedup on GPU.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Gradual Magnitude Pruning (GMP) schedule
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Gradual Magnitude Pruning (GMP) schedule")
print("━" * 65)
print()

def gmp_sparsity(t, T, s_final, s_initial=0.0):
    """
    GMP cubic schedule: sparsity at step t of T total steps.
    s(t) = s_f × (1 - (1 - t/T)^3)
    """
    return s_final + (s_initial - s_final) * (1 - t / T) ** 3

T_steps  = 100     # total pruning steps
s_target = 0.90   # target 90% sparsity

print(f"  GMP schedule: 0% → {s_target:.0%} sparsity over {T_steps} steps")
print(f"  Cubic schedule: s(t) = {s_target:.2f} × (1 - (1 - t/{T_steps})³)")
print()
print(f"  {'Step':>6} | {'Sparsity':>10} | {'NNZ remaining':>15} | "
      f"{'Pruning bar'}")
print(f"  {'─'*55}")

W_total_params = 64 * 256
for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    s  = gmp_sparsity(t, T_steps, s_target)
    remaining = int((1 - s) * W_total_params)
    bar = "█" * int(s * 30)
    print(f"  {t:>6} | {s:>10.4f} | {remaining:>15,} | [{bar}]")

print()
print("  GMP best practices:")
print("  1. Prune with cubic schedule (slow ramp → faster → plateau)")
print("  2. Re-train between pruning steps (fine-tune with mask frozen)")
print("  3. Use 'momentum masking': zero gradients of pruned weights")
print("  4. For 2:4: apply GMP up to 50% sparsity, then convert to 2:4 pattern")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: 2:4 with PyTorch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — 2:4 sparsity with PyTorch (Ampere GPU)")
print("━" * 65)
print()

PYTORCH_24 = """
  PyTorch 2:4 Structured Sparsity Workflow:

  # ── Step 1: Train dense model ─────────────────────────────────────────
  model = MyModel()
  train(model, train_loader, epochs=90)    # full training

  # ── Step 2: Apply 2:4 pruning ─────────────────────────────────────────
  from torch.ao.pruning import WeightNormSparsifier

  sparsifier = WeightNormSparsifier(
      sparsity_level=0.5,          # 50% sparsity
      sparse_block_shape=(1, 4),   # 1 row × 4 cols = groups of 4
      zeros_per_block=2,           # exactly 2 zeros per group → 2:4
  )

  # Specify which layers to prune
  config = [{"tensor_fqn": "layer1.weight"},
             {"tensor_fqn": "layer2.weight"}]
  sparsifier.prepare(model, config)

  # ── Step 3: Fine-tune with mask ────────────────────────────────────────
  # The mask is applied at each forward pass during fine-tuning
  train(model, train_loader, epochs=20)    # recover accuracy

  # ── Step 4: Freeze mask ────────────────────────────────────────────────
  sparsifier.squash_mask()     # makes sparsity permanent (removes mask)

  # ── Step 5: Convert to inference format ────────────────────────────────
  from torch.sparse import to_sparse_semi_structured
  for name, module in model.named_modules():
      if isinstance(module, nn.Linear):
          module.weight = nn.Parameter(
              to_sparse_semi_structured(module.weight)
          )

  # ── Step 6: Inference with 2× speedup ─────────────────────────────────
  output = model(input)  # automatically uses Sparse Tensor Cores on Ampere!

  # ── Direct benchmark ───────────────────────────────────────────────────
  import torch
  from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

  # Create a 2:4 weight matrix directly
  W = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
  # Apply 2:4 pruning: for each group of 4, zero the 2 smallest by magnitude
  W_reshaped = W.view(-1, 4)
  idx = torch.argsort(torch.abs(W_reshaped), dim=1)[:, :2]  # smallest 2
  W_reshaped.scatter_(1, idx, 0.0)

  # Convert to compressed format
  W_24 = to_sparse_semi_structured(W)   # W now stored in Ampere's format

  x = torch.randn(256, 1024, device='cuda', dtype=torch.float16)

  # Dense benchmark
  t_dense = timeit(lambda: x @ W.T, reps=100)

  # 2:4 sparse benchmark (uses Sparse Tensor Cores)
  t_sparse24 = timeit(lambda: x @ W_24.T, reps=100)

  # Expected: t_dense / t_sparse24 ≈ 2.0 (Ampere gives 2× for 2:4)
"""
print(PYTORCH_24)

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        try:
            from torch.sparse import to_sparse_semi_structured

            M_24, K_24, N_24 = 1024, 1024, 256
            W = torch.randn(M_24, K_24, device='cuda', dtype=torch.float16)

            # Create 2:4 mask
            W_4 = W.view(-1, 4)
            idx = torch.argsort(torch.abs(W_4), dim=1)[:, :2]
            W_4.scatter_(1, idx, 0.0)

            W_24_sp = to_sparse_semi_structured(W)
            x_24    = torch.randn(N_24, K_24, device='cuda', dtype=torch.float16)

            def bench(fn, warmup=20, reps=100):
                for _ in range(warmup): fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps): fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / reps * 1000

            t_d  = bench(lambda: torch.mm(x_24, W.T))
            t_24 = bench(lambda: torch.mm(x_24, W_24_sp.T))

            print(f"  Live 2:4 benchmark ({N_24}×{K_24} input, {M_24}×{K_24} weight):")
            print(f"    Dense:       {t_d:.4f}ms")
            print(f"    2:4 sparse:  {t_24:.4f}ms")
            print(f"    Speedup:     {t_d/t_24:.2f}×  "
                  f"(expected ≈ 2.0× on Ampere)")
        except Exception as e:
            print(f"  2:4 sparse not available: {e}")
            print("  (Requires PyTorch ≥ 2.0 and NVIDIA Ampere GPU)")
    else:
        print("  (GPU required for 2:4 benchmark)")
except ImportError:
    print("  (PyTorch required)")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Complete decision guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Complete sparse GEMM/GEMV decision guide")
print("━" * 65)
print()

DECISION_GUIDE = """
  SPARSE vs DENSE DECISION FLOWCHART:

  START: Do you have zeros in your matrix?
    No  → Use dense GEMM (cuBLAS). No benefit from sparse.
    Yes → What is the density (fraction of non-zeros)?

  DENSITY > 20%:
    → Dense GEMM is almost certainly faster.
    → Sparse format overhead + irregular access dominates.
    → Exception: if using NVIDIA Ampere GPU and 50% density → consider 2:4.

  DENSITY 10–20%:
    → Profile both. Sparse MIGHT be faster.
    → GPU: dense likely wins (irregular access penalty is severe).
    → CPU: sparse likely wins with MKL sparse BLAS.
    → Use scipy.sparse CSR for CPU, torch.sparse CSR for GPU.

  DENSITY < 10%:
    → Sparse wins on memory bandwidth.
    → Choose format based on structure:
        Unstructured: CSR (scipy / torch.sparse / cuSPARSE)
        Block-sparse:  BSR (block size = 16×16 for Tensor Core alignment)
        Uniform rows:  ELL (best GPU load balance)
        Heavily varying row lengths: HYB (ELL + COO)

  EXACTLY 50% DENSITY on AMPERE GPU:
    → 2:4 structured sparsity gives guaranteed 2× speedup.
    → Apply GMP + fine-tuning to get 2:4 pattern with < 1% accuracy loss.
    → Recommended for inference of pruned neural networks.

  ATTENTION PATTERNS:
    → Causal (lower triangular): use FlashAttention, NOT sparse SpMM.
    → Sliding window (Longformer): custom Triton kernel or BSR.
    → Block-sparse fixed: BSR + cuSPARSE SpMM.
    → Dynamic learned: custom CUDA kernel (pattern changes per input).

  EMBEDDINGS (one-hot lookup):
    → Never sparse GEMM. Always torch.nn.Embedding (scatter gather).
    → Or: FBGEMM / TorchRec for recommendation model embeddings.

  GRAPH NEURAL NETWORKS:
    → Always sparse (adjacency matrices are > 99% sparse).
    → Use torch_sparse or PyG (PyTorch Geometric) with COO/CSR.
    → SpMM for neighbourhood aggregation.

  IMPLEMENTATION CHECKLIST:
  □ Profile density first: nnz / (M × N)
  □ Check format: CSR for general, BSR for block, 2:4 for Ampere inference
  □ Ensure M, N, K alignment (multiples of 4–16 for BSR Tensor Core path)
  □ Pre-convert to sparse once; don't rebuild format inside a loop
  □ For cuSPARSE: use CUSPARSE_SPMV_ALG_DEFAULT (auto-selects merge-based)
  □ Benchmark: compare sparse vs dense on YOUR actual hardware
  □ Monitor: density can change during training (weight dropping)

  QUICK REFERENCE:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Library / API       │ Best for                │ Backend              │
  ├──────────────────────────────────────────────────────────────────────┤
  │ scipy.sparse        │ CPU sparse ops          │ MKL SpBLAS           │
  │ torch.sparse CSR    │ GPU sparse (cuSPARSE)   │ cuSPARSE             │
  │ torch.ao.pruning    │ 2:4 structured pruning  │ Sparse Tensor Core   │
  │ PyTorch Geometric   │ GNN sparse ops          │ torch_sparse + CUDA  │
  │ Triton kernel       │ Custom sparse patterns  │ Custom CUDA          │
  │ cuSPARSE direct C++ │ Maximum GPU performance │ cuSPARSE             │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(DECISION_GUIDE)
''',
    },
}

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