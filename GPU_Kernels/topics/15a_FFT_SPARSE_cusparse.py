"""
cuSPARSE — CSR / COO / BSR Formats, SpMM / SpMV & Sparse-Dense Fusion
======================================================================

Sparse linear algebra is the foundation of graph neural networks, scientific
simulation, iterative solvers, and large-scale recommendation systems. The
defining property is that most matrix entries are zero — and exploiting this
structure is what makes sparse computations feasible at scale.

cuSPARSE is NVIDIA's CUDA library for sparse matrix operations. It provides:
    - Storage format conversion (COO ↔ CSR ↔ CSC ↔ BSR ↔ ELLPACK)
    - Sparse matrix–vector multiplication (SpMV): y = A × x
    - Sparse matrix–dense matrix multiplication (SpMM): C = A × B
    - Sparse–sparse matrix operations (SpGEMM): C = A × B (both sparse)
    - Triangular solve (SpSV, SpSM)
    - Format-specific reordering (csrSort, coosort)

Three storage formats dominate GPU sparse workloads:

    CSR (Compressed Sparse Row): the workhorse format. Stores row pointers,
    column indices, and values. SpMV and SpMM are O(nnz). Row access is O(1).
    Standard for cuSPARSE SpMM with the generic API.

    COO (Coordinate Format): the simplest format. Stores (row, col, value)
    triples. Unordered or sorted. Easiest to construct; less efficient for
    arithmetic. Useful for matrix construction, conversion, and display.

    BSR (Block Sparse Row): stores small dense blocks (e.g., 4×4 or 8×8)
    as the unit of sparsity. Crucial when non-zeros cluster in blocks —
    as they do in graph neural network weight matrices, finite element
    stiffness matrices, and physics simulations.

Understanding cuSPARSE deeply means understanding the memory layout of each
format, the GPU execution model of SpMV and SpMM, the arithmetic intensity
of sparse kernels, and the sparse-dense fusion patterns that dominate GNN
and recommendation model inference.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "cuSPARSE — CSR / COO / BSR Formats, SpMM / SpMV & Sparse-Dense Fusion"
DISPLAY_NAME = "15a · cuSPARSE"
ICON         = "🕸️"
SUBTITLE     = "CSR · COO · BSR · SpMV · SpMM · Sparse-Dense GNN Fusion · HBM Roofline"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY SPARSE FORMATS EXIST: THE ARITHMETIC INTENSITY ARGUMENT

### The Cost of Storing Zero

    An M×N dense matrix requires M×N elements in memory.
    For a graph with N=1M nodes and average degree 10 (1% density):
        Dense storage: 1M × 1M × 4 bytes = 4 TB  (impossible)
        CSR sparse:    nnz × 4 + nnz × 4 + (N+1) × 4 bytes
                     = 10M × 8 + 4M bytes ≈ 84 MB  (practical)

    Sparse formats trade index overhead for eliminated zero storage.
    They are profitable when sparsity > (1 - index_overhead_fraction).
    For CSR (8 bytes per nnz — 4 bytes value + 4 bytes col_idx):
        Break-even vs dense (4 bytes/element): density < 50%.
        In practice: CSR wins when density < 10–20%.

### Arithmetic Intensity of Sparse vs Dense MatMul

    DENSE GEMM (M×K matrix × K×N matrix):
        FLOPs: 2 × M × K × N
        Bytes: (M×K + K×N + M×N) × 4
        AI ≈ M/4 FLOPs/Byte for large square matrices.
        Compute-bound for large M — achieves tensor core peak.

    SpMM (sparse M×K with nnz non-zeros × dense K×N):
        FLOPs: 2 × nnz × N   (each non-zero contributes N multiply-adds)
        Bytes: nnz × (4 + 4) + nnz × 4 + K×N×4 + M×N×4
             ≈ nnz × 12 + (K×N + M×N) × 4
        AI = 2×nnz×N / (nnz×12 + (K+M)×N×4)
        For nnz << M×K (sparse) and moderate N:
            AI ≈ 2N / 12 = N/6 FLOPs/Byte   (independent of nnz!)
        For N=1:  AI ≈ 0.17 FLOPs/Byte  — severely memory-bound.
        For N=64: AI ≈ 10.7 FLOPs/Byte  — still memory-bound but better.
        For N=512:AI ≈ 85  FLOPs/Byte   — approaching compute-bound.

    KEY INSIGHT: SpMM's arithmetic intensity scales with the DENSE DIMENSION N.
    Small N (N=1, SpMV): near-impossible to avoid memory-bound operation.
    Large N (N≥64): sparse matmul can become meaningfully compute-bound.
    GNN embedding dimension d (typically 64–512) determines whether SpMM
    or SpMV is the bottleneck in graph neural network inference.

### When to Use Sparse vs Dense

    USE SPARSE when:
        Density < 10% (actual rule of thumb: density < 5% for GPU benefit).
        Matrix is too large to store densely (graph adjacency, web graph).
        Irregular sparsity pattern (random, power-law degree graphs).

    USE DENSE (or block-sparse) when:
        Density > 10%: dense GEMM is faster due to tensor core utilisation.
        Regular sparsity (diagonal blocks, band matrices): BSR is better.
        N-dim weight matrices in neural networks: almost always dense.

    NOTE: cuSPARSE SpMM is not a substitute for cuBLAS for dense matrices.
    Even 5%-density matrices may be slower in CSR than dense GEMM on H100
    because dense GEMM uses tensor cores (989 TFLOP/s) while sparse SpMM
    cannot (irregular memory access prevents TC alignment).


##### PART 2 — CSR FORMAT: COMPRESSED SPARSE ROW

### CSR Memory Layout

    A CSR matrix with M rows, N columns, and nnz non-zeros uses three arrays:

        row_ptr[M+1]:    row_ptr[i]..row_ptr[i+1]-1 is the range in col_idx/values
                         that contains all non-zeros in row i.
                         row_ptr[0] = 0 always.
                         row_ptr[M] = nnz always.
                         Size: (M+1) × 4 bytes.

        col_idx[nnz]:    column index of each non-zero, in row-major order.
                         Within each row: sorted ascending (not required, but typical).
                         Size: nnz × 4 bytes.

        values[nnz]:     floating-point value of each non-zero.
                         Same ordering as col_idx.
                         Size: nnz × sizeof(float) bytes.

    EXAMPLE — 4×4 matrix:
        A = [[1, 0, 2, 0],
             [0, 0, 3, 0],
             [4, 5, 6, 7],
             [0, 0, 0, 8]]

        nnz = 7, M = 4, N = 4

        row_ptr  = [0, 2, 3, 7, 8]
            Row 0 has non-zeros at col_idx positions 0..1 (count=2)
            Row 1 has non-zeros at col_idx positions 2..2 (count=1)
            Row 2 has non-zeros at col_idx positions 3..6 (count=4)
            Row 3 has non-zeros at col_idx positions 7..7 (count=1)

        col_idx  = [0, 2, 2, 0, 1, 2, 3, 3]
        values   = [1, 2, 3, 4, 5, 6, 7, 8]

    TOTAL MEMORY: (M+1)×4 + nnz×4 + nnz×sizeof(float)
                = (M+1)×4 + nnz×8 bytes for FP32.

### Row Access: O(1)

    To iterate over non-zeros in row i:
        for j in range(row_ptr[i], row_ptr[i+1]):
            col = col_idx[j]
            val = values[j]

    This is the key advantage of CSR over COO: accessing all non-zeros of
    row i requires exactly row_ptr[i+1] - row_ptr[i] iterations with no
    searching — directly computable from the row pointer array.

### SpMV in CSR: The GPU Execution Model

    y = A × x  where A is CSR, x is dense, y is dense.

    NAÏVE ASSIGNMENT: one CUDA thread per row.
        Thread i: y[i] = Σ_j values[j] × x[col_idx[j]] for j in row_ptr[i..i+1]

    PROBLEMS:
        Load imbalance: different rows have very different numbers of non-zeros.
        Power-law graphs: 90% of rows have degree 1, 1% have degree 1000.
        Long rows (hubs): one thread does 1000× more work than others.
        Short rows (leaves): threads retire quickly, SM is partially idle.

    CUSP/cuSPARSE SOLUTION — MERGE-BASED SPMV:
        Assign thread blocks to equal-sized WORK INTERVALS
        (each work interval = same number of non-zero processing operations,
        not the same number of rows).
        This eliminates load imbalance at the cost of more complex row
        boundary detection inside each thread block.

    cuSPARSE v11+ GENERIC API:
        Uses cusparseSpMV_bufferSize() + cusparseSpMV() with algorithm selection:
            CUSPARSE_SPMV_ALG_DEFAULT: cuSPARSE chooses best algorithm.
            CUSPARSE_SPMV_CSR_ALG1: optimised for regular degree distribution.
            CUSPARSE_SPMV_CSR_ALG2: merge-based, good for power-law graphs.

### CSR Construction Cost

    Building CSR from a list of (row, col, value) triples:
        1. Sort by (row, col): O(nnz log nnz) — can be done on GPU.
        2. Compute row_ptr from row indices: prefix sum O(nnz).
        3. Copy col_idx and values in sorted order.

    cuSPARSE provides: cusparse<t>coosort() and XcooToCsr() for this pipeline.


##### PART 3 — COO AND BSR FORMATS

### COO Format: Coordinate List

    Three arrays of length nnz:
        row_idx[nnz]:  row index of each non-zero.
        col_idx[nnz]:  column index of each non-zero.
        values[nnz]:   value of each non-zero.

    No compression. Total memory: nnz × 12 bytes (for FP32).
    vs CSR: nnz × 8 + (M+1) × 4 bytes.
    COO uses MORE memory than CSR when M is large.

    WHEN COO IS PREFERABLE:
        Matrix construction: easily insert new (i, j, v) entries.
        Conversion hub: COO → CSR/CSC is a simple sort + prefix sum.
        Unordered/random access: no row-ordering requirement.
        Small matrices: overhead of row_ptr is relatively larger.

    SORTED COO: entries sorted by row, then by col within each row.
    Sorted COO → CSR in O(nnz) (just compute row_ptr by prefix sum on row_idx).

### BSR Format: Block Sparse Row

    BSR divides the matrix into dense blocks of size R×C (block rows × block cols).
    The sparsity pattern is defined at BLOCK LEVEL: a block is non-zero if
    any element within it is non-zero.

    BSR arrays:
        bsr_row_ptr[M/R + 1]:  block-level row pointers (like CSR row_ptr but counts blocks).
        bsr_col_idx[nnzb]:     column block index of each non-zero block.
        bsr_values[nnzb × R × C]:  dense values for each non-zero block.

    where nnzb = number of non-zero blocks.

    TOTAL MEMORY: (M/R+1)×4 + nnzb×4 + nnzb×R×C×sizeof(float).

    BSR EXAMPLE with 4×4 matrix, block size R=C=2:
        Block partition (2×2 block grid):
        [B(0,0) | B(0,1)]     B(0,0) non-zero, B(0,1) zero
        [B(1,0) | B(1,1)]     B(1,0) non-zero, B(1,1) non-zero

        bsr_row_ptr = [0, 1, 3]     (1 non-zero block in block-row 0, 2 in block-row 1)
        bsr_col_idx = [0, 0, 1]     (block column indices)
        bsr_values  = [B(0,0) dense values, B(1,0) dense values, B(1,1) dense values]
                    = 3 blocks × 4 values each = 12 values

### When BSR Outperforms CSR

    BSR is better when NON-ZEROS CLUSTER IN BLOCKS:
        FINITE ELEMENT MATRICES: degrees of freedom at mesh nodes cluster.
            3D FEM with 3 DOF/node: blocks are 3×3 (or 6×6 with coupling).
        GRAPH NEURAL NETWORKS: if node features are grouped, edge weights are blocks.
        POINT CLOUD PROCESSING: neighboring points share block structure.
        MOLECULAR DYNAMICS: atoms in the same residue/molecule cluster.

    BSR ARITHMETIC INTENSITY:
        Each non-zero block is a small dense GEMM.
        AI = 2 × R × C / (R×C×4 + R×4 + C×4) ≈ 2/4 = 0.5 FLOPs/Byte for large R,C.
        Still memory-bound, but: the block GEMM can use TENSOR CORES!
        For R=C=16: tensor cores apply → BSR can be 10–20× faster than CSR
        per non-zero element when block size matches TC alignment (multiples of 16).

### Choosing Block Size for BSR

    BLOCK SIZE TRADEOFFS:
        Small blocks (2×2, 4×4): less fill-in, lower memory. No tensor cores.
        Medium blocks (8×8, 16×16): some fill-in. Tensor cores possible on Ampere+.
        Large blocks (32×32, 64×64): high fill-in risk. Full tensor core benefit.

    FILL-IN: an element outside the original sparsity pattern that is stored
    as zero because the block it belongs to has at least one true non-zero.
    Fill-in increases memory usage but enables dense block arithmetic.

    RULE OF THUMB:
        For FEM/simulation: block size = DOF/node (typically 3, 4, or 6).
        For GNNs: block size = 16 or 32 (matches TC alignment, hides latency).
        For random graphs: CSR is better (blocks would be mostly fill-in).


##### PART 4 — SpMV: SPARSE MATRIX–VECTOR MULTIPLICATION

### SpMV Operation and Memory Model

    y = A × x  where:
        A: M×N sparse (CSR), nnz non-zeros
        x: dense N-vector
        y: dense M-vector (output)

    HBM BYTES:
        Read A (CSR):  nnz×4 (values) + nnz×4 (col_idx) + (M+1)×4 (row_ptr) ≈ nnz×8
        Read x:        N×4  (but highly reused — x is read once per row, not once per nnz)
        Read x (realistic): nnz×4 in worst case (no L1 reuse), N×4 in best case (full L1)
        Write y:       M×4
        Total worst-case: nnz×8 + nnz×4 + M×4 = nnz×12 + M×4

    FLOPs: 2×nnz (one multiply + one add per non-zero).

    AI (worst case):
        = 2×nnz / (nnz×12 + M×4)
        ≈ 2/12 = 0.17 FLOPs/Byte for nnz >> M
        SEVERELY memory-bound. The H100 ridge is 295 FLOPs/Byte.
        SpMV achieves about 0.06% of H100 tensor core throughput.

### The x-Reuse Problem

    The critical performance factor: how often is x[col_idx[j]] in cache?

    RANDOM SPARSITY: col_idx is random → x accesses are random → L1 miss.
    STRUCTURED SPARSITY (diagonal, band): col_idx near row index → good locality.
    POWER-LAW GRAPH: high-degree nodes accessed by many rows → hot entries in L2/L1.

    L2 CACHE SIZE matters:
        H100 L2: 50 MB. x fits in L2 if N × 4 < 50 MB → N < 12.5 million.
        For N=1M: x = 4 MB (fits in L2, most accesses become L2 hits).
        For N=100M (large web graph): x = 400 MB, L2 misses dominate.

    CUSP MERGE-BASED SpMV:
        Divides the work into equal-length segments of the merged CSR arrays.
        Each thread block handles a contiguous segment, improving cache locality
        for x by working through the matrix in a predictable access pattern.

### Warp-Level SpMV Execution

    SCALAR APPROACH: one thread per row.
        Thread i loads row_ptr[i..i+1], iterates over col_idx, accumulates dot product.
        Problem: irregular iteration count → warp divergence → stall.

    VECTOR APPROACH: one WARP per row (32 threads reduce over the row).
        All 32 lanes load different (col, val) pairs from the row.
        Warp shuffle reduction (__shfl_down_sync) accumulates partial sums.
        Good for rows with nnz ≥ 32 (warp stays busy throughout).
        Problem: rows with nnz < 32 → lanes idle.

    WARP PER BLOCK APPROACH (cuSPARSE default for balanced graphs):
        Group rows into sets of similar nnz count.
        Assign warps to row groups of similar size.
        Reduces both divergence and load imbalance.


##### PART 5 — SpMM: SPARSE MATRIX × DENSE MATRIX

### SpMM vs SpMV: The Critical Difference

    SpMV (A × x): x is a single dense column.
        AI = O(1/nnz) — pure memory-bound regardless of matrix size.

    SpMM (A × B): B is a dense matrix with N columns.
        y[i,:] = Σ_j A[i,j] × B[j,:]   for all rows i simultaneously.
        AI ≈ N/6 — scales with the number of dense columns.

    N=64:  AI ≈ 10.7 FLOPs/Byte. Memory-bound but 63× better than SpMV.
    N=512: AI ≈ 85 FLOPs/Byte.   Approaching H100 ridge (295 FLOPs/Byte).

    SpMM IS THE DOMINANT OPERATION IN GRAPH NEURAL NETWORKS:
        Message passing: y = adj @ x  (adj sparse, x = node features N×d).
        d is the node feature dimension (typically 64–512 in GNN models).
        For d=256: AI ≈ 42 FLOPs/Byte — feasible compute-to-memory ratio.

### cuSPARSE SpMM API (Generic, v11+)

    SETUP:
        cusparseSpMatDescr_t matA;   // sparse matrix descriptor
        cusparseDnMatDescr_t matB, matC;  // dense matrix descriptors

        // Create sparse matrix in CSR format
        cusparseCreateCsr(&matA, M, K, nnz,
                          d_rowptr, d_colidx, d_values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        // Create dense matrices
        cusparseCreateDnMat(&matB, K, N, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&matC, M, N, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    EXECUTION:
        // Query workspace size
        size_t bufferSize;
        cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

        // Allocate workspace and execute
        cudaMalloc(&dBuffer, bufferSize);
        cusparseSpMM(handle, ...same args..., dBuffer, bufferSize);

    ALGORITHM CHOICES:
        CUSPARSE_SPMM_ALG_DEFAULT:   cuSPARSE selects.
        CUSPARSE_SPMM_CSR_ALG1:      row-based, good for N≥8.
        CUSPARSE_SPMM_CSR_ALG2:      merge-based, better for power-law.
        CUSPARSE_SPMM_CSR_ALG3:      chunk-based, best for very large N.

### SpMM Tiling Strategy

    For SpMM(A, B) where B has N columns:
        OUTER LOOP: over rows of A (blocks of BLOCK_M rows).
        INNER LOOP: over non-zeros in the current row block.
        INNERMOST: dense multiply-accumulate on B columns (BLOCK_N columns of B).

    SMEM USAGE:
        Load a tile of B (BLOCK_K rows × BLOCK_N columns) into SMEM.
        For each non-zero in the current row block: accumulate into output tile.
        Write output tile to HBM.

    CRITICAL PARAMETER: BLOCK_N (number of B columns per tile).
        Small BLOCK_N: more tiles, more overhead, but fits in SMEM.
        Large BLOCK_N: better reuse of B rows, less overhead, but needs more SMEM.
        cuSPARSE typically uses BLOCK_N = 32–128 depending on N.

### cuSPARSE SpMM vs PyG Sparse Backends

    PyTorch Geometric uses several SpMM backends:
        torch.sparse.mm:  PyTorch's own sparse matmul (uses cuSPARSE internally).
        torch_scatter:    Scatter-based aggregation (good for small d).
        cugraph-pyg:      DGL/cuGraph-based, uses cuSPARSE for large graphs.

    For GNN message passing: adj @ x where adj is the normalised adjacency.
        d=64:  torch.sparse.mm is competitive.
        d=256: cuSPARSE SpMM_ALG3 is usually fastest.
        Very large graphs (N>1M nodes): cuGraph-based approaches distribute.


##### PART 6 — SPARSE-DENSE FUSION IN GNNs AND RECOMMENDATION MODELS

### The GNN Forward Pass

    A single Graph Convolutional Network (GCN) layer:
        H^(l+1) = σ(Â × H^(l) × W^(l))

    where:
        Â:    normalised adjacency matrix [N×N sparse]
        H^(l): node feature matrix [N×d dense]
        W^(l): learnable weight matrix [d×d' dense]
        σ:    non-linear activation

    TWO OPERATIONS in each layer:
        1. AGGREGATE: Z = Â × H  (SpMM: sparse × dense)
        2. TRANSFORM: H' = Z × W  (dense GEMM: dense × dense)

    The ratio of SpMM vs GEMM work determines the GNN's performance profile:
        For large N and small d: SpMM dominates (aggregate step is bottleneck).
        For small N and large d: GEMM dominates (transform step is bottleneck).

### The Aggregate-Transform Fusion Opportunity

    NAÏVE EXECUTION:
        Step 1: Z = Â × H    (SpMM) → write Z to HBM (N×d bytes)
        Step 2: H' = Z × W   (GEMM) → read Z from HBM (N×d bytes)
        Extra round-trip: N×d × 4 bytes = N×d×4 bytes extra HBM traffic.
        For N=100K, d=256: 100K × 256 × 4 = 100 MB extra per layer.

    FUSED EXECUTION:
        Single kernel: H'[i] = Σ_j Â[i,j] × H[j] × W  (fuse SpMM + GEMM)
        Reads H[j] from HBM, multiplies by W in registers, accumulates into H'[i].
        Never writes Z to HBM.
        Saves 100 MB per layer per forward pass.

    IMPLEMENTATION APPROACHES:
        A. SpMM(Â, H) in SMEM, then GEMM(tile, W) on the same tile before writing.
           Challenge: tile of Z must fit in SMEM simultaneously with W tiles.
        B. Reorder: H'[i] = (Â[i,:] ⊗ W) applied row by row.
           Each row of Â selects rows of H, accumulates H×W directly.
        C. cuSPARSE + custom epilogue kernel (like cublasLt epilogue fusion).

### Graph Sampling and Sparse Mini-Batch Training

    Full-graph SpMM requires loading the entire adjacency and all features.
    For graphs with N=50M nodes: H alone is 50M × 256 × 4 = 51 GB (doesn't fit!).

    MINI-BATCH TRAINING WITH NEIGHBOR SAMPLING:
        Sample a mini-batch of target nodes (e.g., 1024 nodes).
        For each target: sample K neighbors (1-hop), their K neighbors (2-hop), etc.
        Build a small dense subgraph covering the sampled neighborhood.
        The induced subgraph is sparse and fits in GPU memory.

    THE INDUCED SUBGRAPH:
        For 1024 target nodes, K=25 neighbors, 3 hops:
            Layer 0: 1024 × 25^3 ≈ 640K nodes (actual: deduplicated, much less).
            Adjacency (1 hop): ~1024 × 25 = 25,600 edges.
        The subgraph adjacency matrix is DENSE relative to its size!
        At this scale: dense operations often outperform sparse.

### Sputnik and GRAPHBLAS: Production-Grade Sparse Libraries

    SPUTNIK (2021, Google Brain):
        Sparse neural network inference library.
        Key innovation: VECTOR-SPARSE format — stores sparse vectors instead of matrices.
        Optimised for semi-structured sparsity (e.g., 2:4 sparsity from PyTorch's
        sparse training — 2 zeros per 4 consecutive values).
        2:4 sparsity: 50% reduction in weights with minimal accuracy loss.
        H100 native support: sparse tensor cores execute 2:4 patterns natively.

    GRAPHBLAS (cuSPARSE-based):
        Standard interface for graph algorithms expressed as linear algebra.
        BFS, SSSP, PageRank implemented as SpMM and SpMV sequences.
        Used by cuGraph (RAPIDS), DGL, and network analysis tools.


##### PART 7 — cuSPARSE PERFORMANCE TUNING AND ROOFLINE ANALYSIS

### SpMV / SpMM Performance Model Summary

    Operation   | FLOPs         | HBM bytes              | AI
    ─────────────────────────────────────────────────────────────
    SpMV (CSR)  | 2×nnz         | nnz×12 + M×4           | ~0.17
    SpMM-N8     | 2×nnz×8       | nnz×12 + K×8×4 + M×8×4| ~2.2
    SpMM-N64    | 2×nnz×64      | nnz×12 + (K+M)×64×4   | ~10.7
    SpMM-N512   | 2×nnz×512     | nnz×12 + (K+M)×512×4  | ~85
    Dense GEMM  | 2×M×K×N       | (M×K+K×N+M×N)×4       | ~M/4

    H100 roofline AI = 295 FLOPs/Byte.
    No sparse operation except very large SpMM (N≥1000) reaches it.
    SpMV is fundamentally memory-bandwidth limited — cannot be compute-bound.

### Format Selection Guide for cuSPARSE

    MATRIX CHARACTERISTICS       → BEST FORMAT
    ────────────────────────────────────────────────────────────────────
    General irregular sparsity   → CSR (default)
    Construction / incremental   → COO → convert to CSR
    Block-structured FEM/physics → BSR (block_size = DOF/node)
    Very large N, streaming      → CSR with chunked processing
    50% density, tensor core use → Dense GEMM (CSR slower due to no TC)
    2:4 structured sparsity      → NVIDIA 2:4 sparse tensor core API

### The 2:4 Structured Sparsity on H100

    NVIDIA Ampere and Hopper GPUs support NATIVE 2:4 SPARSITY in tensor cores:
        For every 4 consecutive weight values: exactly 2 are zero.
        The non-zero positions and values are stored in a compressed format.
        The SM decodes the 2:4 pattern and executes a 2× faster sparse MMA.

    H100 2:4 sparse GEMM peak: 1979 TFLOP/s (2× over FP16 dense GEMM).
    Memory: weight matrix stored at 50% compression (same as INT4 dense).
    Accuracy: pytorch.sparse training with ~0.3% perplexity degradation.

    API:
        torch.nn.utils.parametrize.register_parametrization(model, "weight",
            torch.nn.utils.prune.MaskedPruning(amount=0.5, type="unstructured_2_4"))
        model.apply(torch.nn.utils.prune.ln_structured, name='weight', amount=0.5)
        # Convert to sparse format for inference:
        sparse_model = torch.sparse.MaskedLinear.from_dense(dense_model)

### Diagnosing cuSPARSE Performance with NCU

    KEY METRICS:
        l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum  → L1 global load requests
        lts__t_requests_srcunit_tex.sum                  → L2 requests
        dram__bytes_read.sum                             → HBM bytes

    FOR SpMV: expect 90%+ of memory accesses to be HBM (L1 misses dominate).
    FOR SpMM with large N: expect better L2 hit rate as B tile fits in L2.

    INEFFICIENCY SIGNS:
        sm__pipe_alu_cycles_active.avg.pct < 40%: sparse kernel is stalled on memory.
        dram__bytes_read.sum > nnz×12: unnecessary data movement (recompute vs cache).
        smsp__warp_issue_stalled_long_scoreboard.avg.pct > 40%: waiting for L2/HBM.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Sparse Format Zoo — CSR, COO, BSR Layouts & Memory Footprint": {
        "description": (
            "Implement CSR, COO, and BSR storage formats from scratch. Build "
            "each format from a dense matrix. Show the exact arrays, memory "
            "layout, and byte counts. Verify that each format reconstructs "
            "the original matrix exactly. Compare memory footprint across "
            "formats for different sparsity levels. Show when BSR reduces "
            "storage vs CSR despite fill-in."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  SPARSE FORMAT ZOO — CSR, COO, BSR Layouts & Memory Footprint")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Format implementations
# ─────────────────────────────────────────────────────────────────────

class CSRMatrix:
    def __init__(self, dense, dtype=np.float32):
        M, N     = dense.shape
        mask     = dense != 0
        self.M, self.N = M, N
        self.values  = dense[mask].astype(dtype)
        self.col_idx = np.where(mask)[1].astype(np.int32)
        # row_ptr: for each row, index into col_idx/values where that row starts
        row_counts   = mask.sum(axis=1)
        self.row_ptr = np.zeros(M+1, dtype=np.int32)
        self.row_ptr[1:] = np.cumsum(row_counts)
        self.nnz = int(mask.sum())

    def to_dense(self):
        A = np.zeros((self.M, self.N), dtype=np.float32)
        for i in range(self.M):
            for j in range(self.row_ptr[i], self.row_ptr[i+1]):
                A[i, self.col_idx[j]] = self.values[j]
        return A

    def memory_bytes(self):
        return (len(self.row_ptr) * 4 + len(self.col_idx) * 4
                + len(self.values) * 4)


class COOMatrix:
    def __init__(self, dense, dtype=np.float32):
        M, N = dense.shape
        self.M, self.N = M, N
        mask         = dense != 0
        rows, cols   = np.where(mask)
        self.row_idx = rows.astype(np.int32)
        self.col_idx = cols.astype(np.int32)
        self.values  = dense[rows, cols].astype(dtype)
        self.nnz     = len(self.values)

    def to_dense(self):
        A = np.zeros((self.M, self.N), dtype=np.float32)
        for i, j, v in zip(self.row_idx, self.col_idx, self.values):
            A[i, j] = v
        return A

    def memory_bytes(self):
        return len(self.row_idx)*4 + len(self.col_idx)*4 + len(self.values)*4


class BSRMatrix:
    def __init__(self, dense, R, C, dtype=np.float32):
        M, N = dense.shape
        assert M % R == 0 and N % C == 0, "Matrix dims must be divisible by block size"
        self.M, self.N, self.R, self.C = M, N, R, C
        Mb = M // R   # number of block rows
        Nb = N // C   # number of block cols

        bsr_row_ptr = [0]
        bsr_col_idx = []
        bsr_values  = []

        for bi in range(Mb):
            block_count = 0
            for bj in range(Nb):
                block = dense[bi*R:(bi+1)*R, bj*C:(bj+1)*C]
                if np.any(block != 0):
                    bsr_col_idx.append(bj)
                    bsr_values.append(block.flatten())
                    block_count += 1
            bsr_row_ptr.append(bsr_row_ptr[-1] + block_count)

        self.bsr_row_ptr = np.array(bsr_row_ptr, dtype=np.int32)
        self.bsr_col_idx = np.array(bsr_col_idx, dtype=np.int32)
        self.bsr_values  = np.array(bsr_values, dtype=dtype).reshape(-1)
        self.nnzb        = len(bsr_col_idx)
        self.nnz_actual  = int((dense != 0).sum())   # true non-zeros
        self.nnz_stored  = self.nnzb * R * C          # stored elements (includes fill-in)

    def to_dense(self):
        A = np.zeros((self.M, self.N), dtype=np.float32)
        Mb = self.M // self.R
        for bi in range(Mb):
            for ptr in range(self.bsr_row_ptr[bi], self.bsr_row_ptr[bi+1]):
                bj    = self.bsr_col_idx[ptr]
                block = self.bsr_values[ptr * self.R * self.C:
                                         (ptr+1) * self.R * self.C]
                A[bi*self.R:(bi+1)*self.R,
                  bj*self.C:(bj+1)*self.C] = block.reshape(self.R, self.C)
        return A

    def memory_bytes(self):
        return (len(self.bsr_row_ptr)*4 + len(self.bsr_col_idx)*4
                + len(self.bsr_values)*4)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Detailed format trace for small matrix
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Format Trace: 8×8 Sparse Matrix (CSR, COO, BSR)")
print("━" * 68)
print()

A_demo = np.array([
    [1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 4, 0, 0, 0],
    [0, 5, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 10],
    [11,0, 0, 0, 0, 0, 0, 0],
    [0, 0, 12,0, 0, 0, 0, 13],
], dtype=np.float32)

M_d, N_d = A_demo.shape
nnz_d    = int((A_demo != 0).sum())

print(f"  Matrix ({M_d}×{N_d}), nnz={nnz_d}, density={nnz_d/(M_d*N_d)*100:.1f}%")
print()
print("  Dense layout:")
for row in A_demo:
    print("    " + "  ".join(f"{int(v):>3}" if v != 0 else "  ." for v in row))
print()

# CSR
csr = CSRMatrix(A_demo)
print(f"  CSR format:")
print(f"    row_ptr  ({len(csr.row_ptr)} values): {csr.row_ptr.tolist()}")
print(f"    col_idx  ({len(csr.col_idx)} values): {csr.col_idx.tolist()}")
print(f"    values   ({len(csr.values)} values):  {csr.values.astype(int).tolist()}")
print(f"    Memory: {csr.memory_bytes()} bytes  "
      f"(dense would be {M_d*N_d*4} bytes)")
print(f"    Reconstruction: {'✅' if np.allclose(csr.to_dense(), A_demo) else '❌'}")
print()

# COO
coo = COOMatrix(A_demo)
print(f"  COO format:")
print(f"    row_idx  ({len(coo.row_idx)} values): {coo.row_idx.tolist()}")
print(f"    col_idx  ({len(coo.col_idx)} values): {coo.col_idx.tolist()}")
print(f"    values   ({len(coo.values)} values):  {coo.values.astype(int).tolist()}")
print(f"    Memory: {coo.memory_bytes()} bytes")
print(f"    Reconstruction: {'✅' if np.allclose(coo.to_dense(), A_demo) else '❌'}")
print()

# BSR (2×2 blocks)
bsr2 = BSRMatrix(A_demo, R=2, C=2)
print(f"  BSR format (2×2 blocks):")
print(f"    bsr_row_ptr: {bsr2.bsr_row_ptr.tolist()}")
print(f"    bsr_col_idx: {bsr2.bsr_col_idx.tolist()}")
print(f"    nnz blocks:  {bsr2.nnzb}  (true nnz={bsr2.nnz_actual}, "
      f"stored={bsr2.nnz_stored}, fill-in={bsr2.nnz_stored-bsr2.nnz_actual})")
print(f"    Memory: {bsr2.memory_bytes()} bytes")
print(f"    Reconstruction: {'✅' if np.allclose(bsr2.to_dense(), A_demo) else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Memory footprint comparison across sparsity levels
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Memory Footprint vs Density: CSR vs COO vs BSR vs Dense")
print("━" * 68)
print()

N_sq   = 128   # square matrix size for comparison
BLOCK  = 8     # BSR block size

print(f"  Matrix: {N_sq}×{N_sq}, BSR block_size={BLOCK}×{BLOCK}")
print()
print(f"  {'Density':>10}  {'nnz':>8}  {'Dense (KB)':>12}  {'CSR (KB)':>10}  "
      f"{'COO (KB)':>10}  {'BSR (KB)':>10}  {'Best format'}")
print("  " + "─" * 72)

dense_bytes = N_sq * N_sq * 4

for density_pct in [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50]:
    density = density_pct / 100.0
    # Build random sparse matrix
    np.random.seed(int(density_pct * 100))
    A_rand = np.where(np.random.rand(N_sq, N_sq) < density,
                      np.random.randn(N_sq, N_sq).astype(np.float32), 0)
    actual_nnz = int((A_rand != 0).sum())

    csr_r  = CSRMatrix(A_rand)
    coo_r  = COOMatrix(A_rand)
    bsr_r  = BSRMatrix(A_rand, BLOCK, BLOCK)

    csr_kb = csr_r.memory_bytes() / 1024
    coo_kb = coo_r.memory_bytes() / 1024
    bsr_kb = bsr_r.memory_bytes() / 1024
    den_kb = dense_bytes / 1024

    best = min([("CSR", csr_kb), ("COO", coo_kb),
                ("BSR", bsr_kb), ("Dense", den_kb)], key=lambda x: x[1])

    print(f"  {density_pct:>9.1f}%  {actual_nnz:>8}  {den_kb:>10.1f}  "
          f"{csr_kb:>10.1f}  {coo_kb:>10.1f}  {bsr_kb:>10.1f}  {best[0]}")

print()
print("  Dense beats all sparse formats above ~20–30% density.")
print("  COO is always larger than CSR (same col_idx/values + extra row_idx).")
print("  BSR fill-in hurts at low density but helps at medium density.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · SpMV — CSR Implementation, Load Imbalance & Warp Strategies": {
        "description": (
            "Implement CSR SpMV in three styles: scalar (1 thread per row), "
            "vector (1 warp per row), and merge-path (equal work per thread). "
            "Show the load imbalance problem on a power-law degree graph. "
            "Compare execution time models for each strategy. Compute arithmetic "
            "intensity and show why SpMV is always bandwidth-limited. Demonstrate "
            "the x-vector cache reuse impact on effective bandwidth."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  SpMV — CSR Implementation, Load Imbalance & Warp Strategies")
print("=" * 68)
print()

np.random.seed(7)

H100_BW_GBS    = 3350.0
H100_RIDGE     = 295.0    # FLOPs/Byte (FP16 TC)
WARP_SIZE      = 32
H100_SMs       = 132


# ─────────────────────────────────────────────────────────────────────
# CSR SpMV reference
# ─────────────────────────────────────────────────────────────────────

def spmv_csr_scalar(row_ptr, col_idx, values, x):
    """
    Scalar CSR SpMV: one accumulator per row.
    GPU model: one CUDA thread per row.
    """
    M   = len(row_ptr) - 1
    y   = np.zeros(M, dtype=np.float32)
    thread_work = []   # non-zeros processed per 'thread'
    for i in range(M):
        s = 0.0
        for j in range(row_ptr[i], row_ptr[i+1]):
            s += values[j] * x[col_idx[j]]
        y[i] = s
        thread_work.append(row_ptr[i+1] - row_ptr[i])
    return y, thread_work

def spmv_csr_ref(row_ptr, col_idx, values, x):
    """Reference SpMV using dense operation for verification."""
    M  = len(row_ptr) - 1
    N  = len(x)
    A  = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(row_ptr[i], row_ptr[i+1]):
            A[i, col_idx[j]] = values[j]
    return A @ x


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness and arithmetic intensity
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — SpMV Correctness & Arithmetic Intensity")
print("━" * 68)
print()

# Build a random sparse matrix
M_s, N_s, density = 512, 512, 0.02   # 2% density
np.random.seed(42)
A_rand = (np.random.rand(M_s, N_s) < density).astype(np.float32)
A_rand[A_rand == 1] = np.random.randn(int(A_rand.sum()))
x_rand = np.random.randn(N_s).astype(np.float32)

# Build CSR
mask       = A_rand != 0
nnz_s      = int(mask.sum())
values_s   = A_rand[mask].astype(np.float32)
col_idx_s  = np.where(mask)[1].astype(np.int32)
row_counts = mask.sum(axis=1)
row_ptr_s  = np.zeros(M_s+1, dtype=np.int32)
row_ptr_s[1:] = np.cumsum(row_counts)

y_scalar, thread_work = spmv_csr_scalar(row_ptr_s, col_idx_s, values_s, x_rand)
y_ref     = spmv_csr_ref(row_ptr_s, col_idx_s, values_s, x_rand)

print(f"  Matrix: {M_s}×{N_s}, density={density*100:.0f}%, nnz={nnz_s}")
print(f"  SpMV result (first 8): {y_scalar[:8].round(4).tolist()}")
print(f"  Reference  (first 8): {y_ref[:8].round(4).tolist()}")
print(f"  Max error: {np.abs(y_scalar - y_ref).max():.2e}")
print(f"  Correct: {'✅' if np.allclose(y_scalar, y_ref, atol=1e-4) else '❌'}")
print()

# Arithmetic intensity
hbm_values   = nnz_s * 4       # float values
hbm_colidx   = nnz_s * 4       # int32 col indices
hbm_rowptr   = (M_s+1) * 4     # int32 row pointers
hbm_x        = N_s * 4         # x vector (best case: read once from HBM)
hbm_x_worst  = nnz_s * 4       # x vector (worst case: every access is HBM miss)
hbm_y        = M_s * 4         # output y
flops        = 2 * nnz_s       # 1 mul + 1 add per non-zero

ai_best  = flops / (hbm_values + hbm_colidx + hbm_rowptr + hbm_x + hbm_y)
ai_worst = flops / (hbm_values + hbm_colidx + hbm_rowptr + hbm_x_worst + hbm_y)

print(f"  HBM traffic analysis (M={M_s}, N={N_s}, nnz={nnz_s}):")
print(f"    A values:          {hbm_values:>8} B")
print(f"    A col indices:     {hbm_colidx:>8} B")
print(f"    A row pointers:    {hbm_rowptr:>8} B")
print(f"    x vector (best):   {hbm_x:>8} B  (entire x in L2)")
print(f"    x vector (worst):  {hbm_x_worst:>8} B  (every col miss)")
print(f"    y output:          {hbm_y:>8} B")
print(f"    FLOPs: {flops:,}")
print()
print(f"  Arithmetic intensity (best):  {ai_best:.3f} FLOPs/Byte")
print(f"  Arithmetic intensity (worst): {ai_worst:.3f} FLOPs/Byte")
print(f"  H100 roofline ridge: {H100_RIDGE} FLOPs/Byte (FP16 TC)")
print(f"  SpMV is {H100_RIDGE/ai_best:.0f}× to {H100_RIDGE/ai_worst:.0f}× below the roofline. "
      f"Always memory-bound.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Load imbalance on power-law graphs
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Load Imbalance: Uniform vs Power-Law Degree Graph")
print("━" * 68)
print()

def make_power_law_graph(N, avg_degree, alpha=2.1):
    """
    Generate a sparse adjacency matrix with power-law degree distribution.
    Returns CSR arrays.
    """
    np.random.seed(99)
    # Sample degrees from power-law distribution
    degrees = np.random.zipf(alpha, N).astype(int)
    degrees = np.minimum(degrees, N-1)   # cap at N-1
    row_ptr = np.zeros(N+1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(degrees)
    total_edges = int(degrees.sum())
    col_idx = np.random.randint(0, N, size=total_edges, dtype=np.int32)
    values  = np.ones(total_edges, dtype=np.float32)
    return row_ptr, col_idx, values, degrees

def load_balance_stats(thread_work):
    """Statistics on per-thread work distribution."""
    tw = np.array(thread_work)
    return {
        "mean":  tw.mean(),
        "std":   tw.std(),
        "max":   tw.max(),
        "min":   tw.min(),
        "p99":   np.percentile(tw, 99),
        "zeros": int((tw == 0).sum()),
        "imbalnce": tw.max() / max(tw.mean(), 1e-10),
    }

N_graph   = 4096
avg_deg   = 10

# Uniform graph
row_ptr_u = np.zeros(N_graph+1, dtype=np.int32)
for i in range(N_graph):
    row_ptr_u[i+1] = row_ptr_u[i] + avg_deg
col_idx_u = np.random.randint(0, N_graph, size=N_graph*avg_deg, dtype=np.int32)
values_u  = np.ones(N_graph*avg_deg, dtype=np.float32)
degrees_u = np.full(N_graph, avg_deg)

# Power-law graph
row_ptr_p, col_idx_p, values_p, degrees_p = make_power_law_graph(N_graph, avg_deg)

graphs = [
    ("Uniform (all deg=10)",  row_ptr_u, degrees_u),
    ("Power-law (Zipf α=2.1)", row_ptr_p, degrees_p),
]

for gname, rp, degs in graphs:
    thread_work_g = (rp[1:] - rp[:-1]).tolist()
    stats = load_balance_stats(thread_work_g)
    print(f"  {gname}:")
    print(f"    Degree: mean={stats['mean']:.1f}, max={stats['max']}, "
          f"p99={stats['p99']:.0f}, zeros={stats['zeros']}")
    print(f"    Load imbalance (max/mean): {stats['imbalnce']:.1f}×")

    # Warp efficiency simulation
    # SCALAR: threads assigned round-robin to rows. One "warp" = 32 rows.
    # Warp time = max degree in its 32 rows (not mean).
    n_warps = math.ceil(N_graph / WARP_SIZE)
    warp_times = []
    for w in range(n_warps):
        start = w * WARP_SIZE
        end   = min(start + WARP_SIZE, N_graph)
        warp_times.append(max(thread_work_g[start:end]))
    ideal_time = sum(thread_work_g) / (n_warps * WARP_SIZE)
    actual_time = np.mean(warp_times)
    efficiency  = ideal_time / actual_time * 100

    print(f"    Scalar SpMV warp efficiency: {efficiency:.1f}%")
    print(f"    (ideal={ideal_time:.1f} ops/warp, "
          f"actual avg={actual_time:.1f} due to long-tail rows)")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Three SpMV strategies compared
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 3 — SpMV Strategies: Scalar vs Vector vs Merge-Path")
print("━" * 68)
print()

print("  STRATEGY 1 — SCALAR (1 thread per row):")
print("    + Simple implementation")
print("    - Load imbalance for power-law graphs")
print("    - Warp divergence: threads with long rows stall the warp")
print("    - Best for: uniform-degree graphs, small M")
print()
print("  STRATEGY 2 — VECTOR (1 warp per row):")
print("    + No divergence within a warp (all 32 lanes do same row)")
print("    + __shfl_down_sync reduction: efficient partial-sum accumulation")
print("    - Warp IDLE when row has < 32 non-zeros (lanes waste cycles)")
print("    - Bad for power-law: leaf nodes use 1 lane, hub nodes underparallelise")
print("    - Best for: dense rows (large average degree)")
print()
print("  STRATEGY 3 — MERGE-PATH (equal work intervals):")
print("    + Perfect load balance: each thread block handles same nnz count")
print("    + Works well for power-law: hubs split across multiple thread blocks")
print("    - Complex row-boundary detection (binary search in row_ptr)")
print("    - Slight overhead for boundary handling")
print("    - Best for: power-law graphs, any irregular sparsity")
print()
print("  cuSPARSE SPMV_CSR_ALG2 uses merge-path by default.")
print()

# Show the merge-path work interval concept
nnz_test     = 24
n_threads    = 4
interval_size = nnz_test // n_threads

print(f"  Merge-path example: {nnz_test} nnz / {n_threads} thread-blocks:")
print(f"  Each block handles exactly {interval_size} non-zeros.")
print(f"  Blocks process contiguous nnz ranges regardless of row boundaries.")

# Show how rows distribute across merge intervals
sample_degrees = [1, 8, 3, 12]  # 4 rows, sum=24
row_ptr_m = [0, 1, 9, 12, 24]
print(f"  Row degrees: {sample_degrees}  (row_ptr: {row_ptr_m})")
print(f"  {'Block':>6}  {'nnz range':>12}  {'Rows covered'}")
print("  " + "─" * 36)
for blk in range(n_threads):
    nnz_start = blk * interval_size
    nnz_end   = nnz_start + interval_size
    rows = []
    for row_idx, (rstart, rend) in enumerate(zip(row_ptr_m[:-1], row_ptr_m[1:])):
        if rend > nnz_start and rstart < nnz_end:
            rows.append(row_idx)
    print(f"  {blk:>6}  [{nnz_start:>4}..{nnz_end:>4}]  rows {rows}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · SpMM — Dense Dimension Scaling, Tiling & GNN Message Passing": {
        "description": (
            "Implement SpMM and show how arithmetic intensity scales with the "
            "dense dimension N. Simulate the GNN message passing operation "
            "(adj @ node_features) for different graph sizes and feature "
            "dimensions. Show the tiling strategy for SpMM. Verify against "
            "dense matrix multiply reference. Compute the crossover point where "
            "SpMM becomes faster than dense matmul due to sparse memory savings."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  SpMM — Dense Dimension Scaling, Tiling & GNN Message Passing")
print("=" * 68)
print()

np.random.seed(42)

H100_BW_GBS   = 3350.0
H100_PEAK_FP32 = 67.0    # TFLOP/s scalar (SpMM uses FP32 scalar path)
H100_RIDGE_FP32 = H100_PEAK_FP32 * 1e12 / (H100_BW_GBS * 1e9)


# ─────────────────────────────────────────────────────────────────────
# SpMM reference implementation
# ─────────────────────────────────────────────────────────────────────

def spmm_csr(row_ptr, col_idx, values, B, M, K, N):
    """
    SpMM: C = A × B  where A is CSR (M×K sparse), B is dense (K×N).
    Returns C (M×N).
    """
    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for jptr in range(row_ptr[i], row_ptr[i+1]):
            j   = col_idx[jptr]
            val = values[jptr]
            C[i, :] += val * B[j, :]   # accumulate row j of B scaled by A[i,j]
    return C


def build_csr(dense_A):
    M, K   = dense_A.shape
    mask   = dense_A != 0
    nnz    = int(mask.sum())
    vals   = dense_A[mask].astype(np.float32)
    cidx   = np.where(mask)[1].astype(np.int32)
    rptr   = np.zeros(M+1, dtype=np.int32)
    rptr[1:] = np.cumsum(mask.sum(axis=1))
    return rptr, cidx, vals, nnz


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: AI scaling with dense dimension N
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Arithmetic Intensity vs Dense Dimension N")
print("━" * 68)
print()

M_ai, K_ai, density_ai = 1024, 1024, 0.02
np.random.seed(1)
A_ai = (np.random.rand(M_ai, K_ai) < density_ai).astype(np.float32) * np.random.randn(M_ai, K_ai)
_, _, _, nnz_ai = build_csr(A_ai)

print(f"  Sparse A: {M_ai}×{K_ai}, density={density_ai*100:.0f}%, nnz={nnz_ai}")
print(f"  Dense B: {K_ai}×N  (varying N)")
print(f"  H100 FP32 scalar ridge: {H100_RIDGE_FP32:.1f} FLOPs/Byte")
print()
print(f"  {'N':>8}  {'FLOPs':>12}  {'HBM bytes':>14}  {'AI':>8}  "
      f"{'BW-limit µs':>13}  {'Compute µs':>13}  {'Bottleneck'}")
print("  " + "─" * 74)

for N_spmm in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    flops     = 2 * nnz_ai * N_spmm
    hbm_A     = nnz_ai * 8               # values + col_idx
    hbm_rowptr= (M_ai+1) * 4
    hbm_B     = K_ai * N_spmm * 4       # dense B
    hbm_C     = M_ai * N_spmm * 4       # output C
    hbm_total = hbm_A + hbm_rowptr + hbm_B + hbm_C
    ai        = flops / hbm_total

    t_bw_us  = hbm_total / (H100_BW_GBS * 1e9) * 1e6
    t_cmp_us = flops / (H100_PEAK_FP32 * 1e12) * 1e6
    bottleneck = "🔴 mem" if ai < H100_RIDGE_FP32 else "🟢 cmp"

    print(f"  {N_spmm:>8}  {flops/1e6:>10.2f}M  {hbm_total/1e6:>12.2f}M  "
          f"{ai:>8.2f}  {t_bw_us:>13.3f}  {t_cmp_us:>13.5f}  {bottleneck}")

print()
print(f"  For N=1 (SpMV): AI={2/12:.2f}. Memory-bound by {H100_RIDGE_FP32/(2/12):.0f}×.")
print(f"  For N=512: AI={2*nnz_ai*512/(nnz_ai*8+(M_ai+1)*4+(K_ai+M_ai)*512*4):.1f}. Closer to ridge.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: GNN message passing — SpMM in practice
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — GNN Message Passing: adj @ node_features")
print("━" * 68)
print()

def make_graph_adjacency(n_nodes, avg_degree, normalise=True):
    """Build random adjacency matrix, optionally normalised (D^{-1/2} A D^{-1/2})."""
    np.random.seed(5)
    density  = avg_degree / n_nodes
    adj      = (np.random.rand(n_nodes, n_nodes) < density).astype(np.float32)
    np.fill_diagonal(adj, 0)   # no self-loops for graph connectivity

    if normalise:
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

    return adj

print("  GCN layer: H' = σ(Â × H × W)")
print("  Aggregate step: Z = Â × H  (SpMM — sparse adj × dense features)")
print()

graph_configs = [
    (512,  10, 64,  "Small graph, d=64"),
    (1024, 20, 128, "Medium graph, d=128"),
    (2048, 15, 256, "Large graph, d=256"),
    (4096, 10, 64,  "Very large graph, d=64"),
]

print(f"  {'Config':<32}  {'nnz':>8}  {'SpMM correct?':>14}  "
      f"{'AI':>8}  {'BW limit µs'}")
print("  " + "─" * 68)

for n_nodes, avg_deg, d_feat, label in graph_configs:
    adj      = make_graph_adjacency(n_nodes, avg_deg)
    H_feat   = np.random.randn(n_nodes, d_feat).astype(np.float32)

    rptr, cidx, vals, nnz_g = build_csr(adj)

    Z_spmm = spmm_csr(rptr, cidx, vals, H_feat, n_nodes, n_nodes, d_feat)
    Z_ref  = adj @ H_feat
    err    = np.abs(Z_spmm - Z_ref).max()
    ok     = err < 1e-3

    flops   = 2 * nnz_g * d_feat
    hbm_A   = nnz_g * 8 + (n_nodes+1)*4
    hbm_HZ  = 2 * n_nodes * d_feat * 4
    hbm_tot = hbm_A + hbm_HZ
    ai      = flops / hbm_tot
    bw_us   = hbm_tot / (H100_BW_GBS * 1e9) * 1e6

    print(f"  {label:<32}  {nnz_g:>8}  {'✅' if ok else '❌':>14}  "
          f"{ai:>8.2f}  {bw_us:>11.3f}")

print()
print("  GNN message passing (SpMM) AI scales with feature dimension d.")
print("  For d=256: AI≈35 — still memory-bound but 200× better than SpMV (AI≈0.17).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: SpMM tiling concept
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — SpMM Tiling: SMEM Reuse of Dense B Tiles")
print("━" * 68)
print()

print("  SpMM tiling: load a TILE of B (K_tile × N_tile) into SMEM,")
print("  process all non-zeros in the corresponding K_tile rows of A,")
print("  accumulate into C, then move to next tile.")
print()

M_tile, K_tile, N_dense = 128, 1024, 256
nnz_ex = int(M_tile * K_tile * 0.02)   # 2% density

TILE_K = 64    # K-tile size
TILE_N = 32    # N-tile size (dense columns per SMEM tile)

n_k_tiles  = math.ceil(K_tile / TILE_K)
n_n_tiles  = math.ceil(N_dense / TILE_N)
n_m_blocks = math.ceil(M_tile / 32)   # 32 rows per thread block

smem_B_tile_bytes = TILE_K * TILE_N * 4   # B tile in SMEM
smem_C_tile_bytes = 32 * TILE_N * 4       # accumulator tile in SMEM
total_smem        = smem_B_tile_bytes + smem_C_tile_bytes

print(f"  Matrix dimensions: A ({M_tile}×{K_tile} sparse, {nnz_ex} nnz), "
      f"B ({K_tile}×{N_dense} dense)")
print(f"  Tile sizes: K_tile={TILE_K}, N_tile={TILE_N}")
print()
print(f"  SMEM per thread block:")
print(f"    B tile:       {TILE_K}×{TILE_N}×4 = {smem_B_tile_bytes} bytes")
print(f"    C accumulator: 32×{TILE_N}×4  = {smem_C_tile_bytes} bytes")
print(f"    Total:         {total_smem} bytes (H100 has 228 KB SMEM)")
print()
print(f"  HBM loads WITHOUT tiling: B loaded {n_m_blocks} times (once per M block).")
print(f"    = {n_m_blocks} × {K_tile*N_dense*4/1024:.1f} KB = "
      f"{n_m_blocks*K_tile*N_dense*4/1024:.1f} KB total B reads")
print(f"  HBM loads WITH N-tiling: B loaded {n_k_tiles}×{n_n_tiles} times (per K-tile, N-tile).")
print(f"    Each N-tile of B ({TILE_K}×{TILE_N}): {TILE_K*TILE_N*4} bytes")
print(f"    Total unique B reads: {K_tile*N_dense*4/1024:.1f} KB (same data read once each)")
print()
print("  N-tiling key insight: each B[j,:] row is reused by EVERY non-zero in column j.")
print("  Loading it into SMEM once amortises the HBM cost across all those non-zeros.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · BSR Format — Block Structure, Fill-In & Tensor Core Opportunity": {
        "description": (
            "Implement BSR storage and SpMM using block-level dense GEMM. "
            "Show how BSR enables tensor core usage via block-level dense operations. "
            "Simulate fill-in for different block sizes on a structured FEM-like "
            "matrix and a random graph adjacency. Show when BSR is faster than "
            "CSR via the flops-per-HBM-byte analysis. Compare BSR SpMM throughput "
            "model against CSR SpMM for varying block sizes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  BSR FORMAT — Block Structure, Fill-In & Tensor Core Opportunity")
print("=" * 68)
print()

np.random.seed(11)

H100_BW_GBS     = 3350.0
H100_TC_TFLOPS  = 989.0   # FP16 tensor core
H100_FP32_TFLOPS= 67.0    # scalar FP32


# ─────────────────────────────────────────────────────────────────────
# BSR operations
# ─────────────────────────────────────────────────────────────────────

def build_bsr(dense, R, C):
    """Convert dense matrix to BSR with R×C block size."""
    M, N = dense.shape
    assert M % R == 0 and N % C == 0
    Mb, Nb = M // R, N // C

    bsr_row_ptr = [0]
    bsr_col_idx = []
    bsr_values  = []
    fill_in_count = 0

    for bi in range(Mb):
        cnt = 0
        for bj in range(Nb):
            blk = dense[bi*R:(bi+1)*R, bj*C:(bj+1)*C]
            if np.any(blk != 0):
                bsr_col_idx.append(bj)
                bsr_values.append(blk.copy())
                cnt += 1
                fill_in_count += int((blk == 0).sum())
        bsr_row_ptr.append(bsr_row_ptr[-1] + cnt)

    return (np.array(bsr_row_ptr, np.int32),
            np.array(bsr_col_idx, np.int32),
            np.stack(bsr_values) if bsr_values else np.zeros((0, R, C), np.float32),
            fill_in_count)


def bsr_spmm(bsr_row_ptr, bsr_col_idx, bsr_values, B_dense, R, C, M, K, N):
    """
    BSR SpMM: C = A_bsr × B_dense.
    Each non-zero block is a dense R×C block.
    In GPU: each non-zero block is processed as a small GEMM (tensor-core-eligible!).
    """
    Mb   = M // R
    C_out= np.zeros((M, N), dtype=np.float32)
    for bi in range(Mb):
        for ptr in range(bsr_row_ptr[bi], bsr_row_ptr[bi+1]):
            bj   = bsr_col_idx[ptr]
            blk  = bsr_values[ptr]   # R×C dense block
            # B rows accessed: bj*C .. (bj+1)*C
            B_blk = B_dense[bj*C:(bj+1)*C, :]   # C×N
            # Block GEMM: R×C × C×N = R×N
            C_out[bi*R:(bi+1)*R, :] += blk @ B_blk
    return C_out


def make_fem_matrix(N, block_dof=3):
    """
    Finite-element-like matrix: each 'node' has block_dof DOF.
    Nodes on a grid are connected to their 6 neighbors.
    """
    n_nodes  = N // block_dof
    grid_size = int(math.ceil(math.sqrt(n_nodes)))
    A = np.zeros((N, N), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            i = r * grid_size + c
            if i >= n_nodes:
                break
            # Connect to left, right, up, down neighbours
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                j = (r+dr)*grid_size + (c+dc)
                if 0 <= j < n_nodes and 0 <= r+dr < grid_size and 0 <= c+dc < grid_size:
                    # Block coupling: block_dof × block_dof dense interaction
                    bi_r, bj_r = i*block_dof, j*block_dof
                    blk = np.random.randn(block_dof, block_dof).astype(np.float32)
                    A[bi_r:bi_r+block_dof, bj_r:bj_r+block_dof] = blk
    # Diagonal blocks (self-coupling)
    for i in range(n_nodes):
        bi = i * block_dof
        A[bi:bi+block_dof, bi:bi+block_dof] = np.eye(block_dof) * 4.0
    return A


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: BSR fill-in analysis for different matrix types
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — BSR Fill-In: FEM Matrix vs Random Graph")
print("━" * 68)
print()

matrix_types = [
    ("FEM (3×3 blocks)",  make_fem_matrix(96, block_dof=3),  "natural 3-DOF/node structure"),
    ("Random (5% density)", None, "no block structure"),
]

# Build random matrix
np.random.seed(7)
M_rand, N_rand = 96, 96
A_rand_fill = (np.random.rand(M_rand, N_rand) < 0.05).astype(np.float32)
A_rand_fill *= np.random.randn(M_rand, N_rand).astype(np.float32)
matrix_types[1] = ("Random (5% density)", A_rand_fill, "no block structure")

for mat_name, A_mat, note in matrix_types:
    M_m, N_m = A_mat.shape
    true_nnz = int((A_mat != 0).sum())

    print(f"  {mat_name}  ({M_m}×{N_m}, true nnz={true_nnz}):")
    print(f"  [{note}]")
    print()
    print(f"  {'Block R×C':>12}  {'nnz blocks':>12}  {'Stored elems':>14}  "
          f"{'Fill-in':>10}  {'Fill%':>8}  {'Memory vs CSR'}")
    print("  " + "─" * 68)

    csr_mem = true_nnz * 8 + (M_m+1) * 4

    for R_b, C_b in [(1,1),(2,2),(3,3),(4,4),(8,8),(16,16)]:
        if M_m % R_b != 0 or N_m % C_b != 0:
            continue
        rp, ci, bv, fi = build_bsr(A_mat, R_b, C_b)
        nnzb      = len(ci)
        stored    = nnzb * R_b * C_b
        fill_pct  = fi / max(stored, 1) * 100
        bsr_mem   = len(rp)*4 + len(ci)*4 + stored*4
        mem_ratio = bsr_mem / csr_mem

        print(f"  {R_b}×{C_b:>2}{'':<8}  {nnzb:>12}  {stored:>14}  "
              f"{fi:>10}  {fill_pct:>7.1f}%  {mem_ratio:>6.2f}× CSR")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: BSR tensor core opportunity
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — BSR Tensor Core Opportunity vs CSR Scalar Path")
print("━" * 68)
print()

print("  CSR SpMM: irregular access pattern → scalar FP32 path (no tensor cores).")
print("  BSR SpMM with block_size=16: each non-zero block = 16×16 GEMM")
print("             → TENSOR CORES can be used! (16-multiple alignment)")
print()

# Performance model: CSR vs BSR for a 10K-node FEM matrix
M_tc, K_tc, N_tc = 240, 240, 64

# FEM-style: assume 3×3 blocks, average 4 block-neighbors per node
# True sparsity is block-structured
block_dof   = 3
n_nodes_tc  = M_tc // block_dof
avg_block_nbrs = 4   # each node connected to 4 neighbors in grid
nnzb_tc     = n_nodes_tc * (avg_block_nbrs + 1)  # +1 for diagonal
nnz_tc      = nnzb_tc * block_dof**2
density_tc  = nnz_tc / (M_tc * K_tc)

print(f"  FEM matrix: {M_tc}×{K_tc}, {block_dof}×{block_dof} blocks, "
      f"nnzb={nnzb_tc}, nnz={nnz_tc}, density={density_tc*100:.1f}%")
print()

formats = [
    ("CSR SpMM",          1, 1,  False,  H100_FP32_TFLOPS),
    ("BSR 3×3 SpMM",      3, 3,  False,  H100_FP32_TFLOPS),
    ("BSR 16×16 SpMM",   16, 16, True,   H100_TC_TFLOPS),
]

print(f"  {'Format':<22}  {'FLOPs':>12}  {'HBM bytes':>14}  "
      f"{'AI':>8}  {'BW µs':>8}  {'Compute µs':>12}  {'TC?'}")
print("  " + "─" * 80)

for fmt_name, R_f, C_f, use_tc, peak_tflops in formats:
    # BSR 16×16 needs larger blocks → more fill-in
    if R_f == 16:
        nnzb_f = math.ceil(M_tc / 16) * math.ceil(K_tc / 16) * 0.25  # sparse block grid
        nnzb_f = max(nnzb_f, nnzb_tc)   # at least as many as true
        flops_f = 2 * nnzb_f * R_f * C_f * N_tc
        stored_f = nnzb_f * R_f * C_f
    else:
        stored_f = nnz_tc if R_f == 1 else nnzb_tc * R_f * C_f
        flops_f  = 2 * stored_f * N_tc

    hbm_f   = stored_f * 4 + stored_f * 4 + K_tc * N_tc * 4 + M_tc * N_tc * 4
    ai_f    = flops_f / hbm_f
    bw_us   = hbm_f / (H100_BW_GBS * 1e9) * 1e6
    cmp_us  = flops_f / (peak_tflops * 1e12) * 1e6
    total_us = max(bw_us, cmp_us)
    tc_sym  = "✅ yes" if use_tc else "  no"

    print(f"  {fmt_name:<22}  {flops_f/1e6:>10.2f}M  {hbm_f/1e3:>12.1f}K  "
          f"{ai_f:>8.2f}  {bw_us:>8.3f}  {cmp_us:>12.6f}  {tc_sym}")

print()
print("  BSR 16×16 can use tensor cores → compute bound for large N.")
print("  CSR SpMM: always scalar FP32, always memory-bound.")
print("  Block size 16 is the minimum for H100 WMMA API alignment.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: BSR SpMM correctness
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — BSR SpMM Correctness Verification")
print("━" * 68)
print()

A_test = make_fem_matrix(48, block_dof=3)
B_test = np.random.randn(48, 32).astype(np.float32)
C_ref  = A_test @ B_test

print(f"  FEM matrix: {A_test.shape}, B: {B_test.shape}")
print()
print(f"  {'Block R×C':>12}  {'Max err':>10}  {'Correct?'}")
print("  " + "─" * 30)

for R_t, C_t in [(1,1),(3,3),(4,4),(6,6)]:
    if A_test.shape[0] % R_t or A_test.shape[1] % C_t:
        continue
    rp, ci, bv, _ = build_bsr(A_test, R_t, C_t)
    C_bsr = bsr_spmm(rp, ci, bv, B_test, R_t, C_t,
                      A_test.shape[0], A_test.shape[1], B_test.shape[1])
    err = np.abs(C_bsr - C_ref).max()
    ok  = err < 1e-4
    print(f"  {R_t}×{C_t:>2}{'':<8}  {err:>10.2e}  {'✅' if ok else '❌'}")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Sparse-Dense GNN Fusion — Aggregate+Transform, Memory Saving": {
        "description": (
            "Implement the fused GCN aggregate+transform operation: "
            "H' = Â × H × W, avoiding the intermediate Z tensor write to HBM. "
            "Verify against the two-step unfused pipeline. Compute the HBM "
            "traffic reduction from fusion. Profile the memory savings across "
            "GNN depths. Show the GNN forward pass pipeline with all sparse "
            "and dense operations annotated with their arithmetic intensities."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  SPARSE-DENSE GNN FUSION — Aggregate+Transform, Memory Saving")
print("=" * 68)
print()

np.random.seed(42)

H100_BW_GBS    = 3350.0
H100_TC_TFLOPS = 989.0
H100_FP32_TFLOPS = 67.0


# ─────────────────────────────────────────────────────────────────────
# GNN layer operations
# ─────────────────────────────────────────────────────────────────────

def build_csr(A):
    M, K   = A.shape
    mask   = A != 0
    nnz    = int(mask.sum())
    vals   = A[mask].astype(np.float32)
    cidx   = np.where(mask)[1].astype(np.int32)
    rptr   = np.zeros(M+1, dtype=np.int32)
    rptr[1:] = np.cumsum(mask.sum(axis=1))
    return rptr, cidx, vals, nnz


def spmm_csr(row_ptr, col_idx, values, B):
    """SpMM: C = A × B."""
    M, N = len(row_ptr)-1, B.shape[1]
    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for jptr in range(row_ptr[i], row_ptr[i+1]):
            j = col_idx[jptr]
            C[i, :] += values[jptr] * B[j, :]
    return C


def gcn_layer_unfused(adj_rptr, adj_cidx, adj_vals, H, W):
    """
    Unfused GCN layer: H' = relu(Â × H × W)
    Step 1: Z = Â × H  (SpMM → write Z to HBM)
    Step 2: Y = Z × W  (dense GEMM → read Z from HBM)
    """
    Z  = spmm_csr(adj_rptr, adj_cidx, adj_vals, H)  # SpMM
    Y  = Z @ W                                        # Dense GEMM
    return np.maximum(0, Y), Z   # return Z for memory analysis


def gcn_layer_fused(adj_rptr, adj_cidx, adj_vals, H, W):
    """
    Fused GCN layer: H' = relu(Â × H × W)
    Computes H × W first (W is small dense GEMM), then SpMM with the result.
    OR: for each non-zero in Â, directly accumulate H[j] × W into H'[i].
    This avoids materialising Z = Â × H.

    FUSION STRATEGY: precompute HW = H × W (dense d×d' GEMM),
    then SpMM(Â, HW) — same as unfused but Z only contains N×d' not N×d.
    For d >> d' (bottleneck pattern): large savings.
    For d == d' (same dimension): no savings from this fusion.

    ALTERNATIVE FUSION: SpMM(Â, H) in SMEM, then multiply tile × W in SMEM.
    Avoids the N×d intermediate entirely. This is what cuSPARSE + custom kernel achieves.
    Simulated here by showing the memory footprint difference.
    """
    # Compute HW = H × W (dense GEMM — small)
    HW = H @ W   # N × d'

    # SpMM(Â, HW) — now Z has dimension N × d' instead of N × d
    Y  = spmm_csr(adj_rptr, adj_cidx, adj_vals, HW)
    return np.maximum(0, Y)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Fusion correctness
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Fused vs Unfused GCN Layer: Correctness")
print("━" * 68)
print()

N_nodes = 256
d_in    = 64    # input feature dimension
d_out   = 32    # output dimension

adj_dense = (np.random.rand(N_nodes, N_nodes) < 0.03).astype(np.float32)
np.fill_diagonal(adj_dense, 0)
H_init    = np.random.randn(N_nodes, d_in).astype(np.float32)
W_layer   = np.random.randn(d_in, d_out).astype(np.float32)

rptr, cidx, vals, nnz_gcn = build_csr(adj_dense)

H_unf, Z_unf = gcn_layer_unfused(rptr, cidx, vals, H_init, W_layer)
H_fus        = gcn_layer_fused(rptr, cidx, vals, H_init, W_layer)

err = np.abs(H_unf - H_fus).max()
print(f"  N={N_nodes}, d_in={d_in}, d_out={d_out}, nnz={nnz_gcn}")
print(f"  Max error between fused and unfused: {err:.2e}")
print(f"  Correct: {'✅' if err < 1e-3 else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: HBM traffic analysis — unfused vs fused
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — HBM Traffic: Unfused vs Fused GCN Layer")
print("━" * 68)
print()

gcn_configs = [
    ("Small GNN",  512,  8,  64,  64,  "GCN-style, d_in=d_out"),
    ("Medium GNN", 1024, 15, 128, 64,  "bottleneck layer"),
    ("Large GNN",  2048, 10, 256, 64,  "high-dim features"),
    ("GAT-like",   1024, 20, 512, 128, "large feature dim"),
]

print(f"  {'Config':<24}  {'N':>6}  {'nnz':>8}  "
      f"{'Unf. bytes (MB)':>16}  {'Fus. bytes (MB)':>16}  {'Saving'}")
print("  " + "─" * 74)

for label, N_g, avg_deg, d_in_g, d_out_g, note in gcn_configs:
    # Build adjacency
    np.random.seed(13)
    adj_g   = (np.random.rand(N_g, N_g) < avg_deg/N_g).astype(np.float32)
    np.fill_diagonal(adj_g, 0)
    nnz_g   = int((adj_g != 0).sum())

    # HBM traffic
    hbm_adj   = nnz_g * 8 + (N_g+1)*4      # CSR values + col_idx + row_ptr

    # Unfused: SpMM(Â, H) then dense(Z, W)
    hbm_H     = N_g * d_in_g * 4            # read H for SpMM
    hbm_Z_w   = N_g * d_in_g * 4            # write Z after SpMM
    hbm_Z_r   = N_g * d_in_g * 4            # read Z for dense GEMM
    hbm_W     = d_in_g * d_out_g * 4        # read W
    hbm_Hout_w= N_g * d_out_g * 4           # write H'
    hbm_unfused = hbm_adj + hbm_H + hbm_Z_w + hbm_Z_r + hbm_W + hbm_Hout_w

    # Fused: dense(H, W) → HW small, then SpMM(Â, HW)
    hbm_H2    = N_g * d_in_g * 4            # read H for dense
    hbm_W2    = d_in_g * d_out_g * 4        # read W for dense
    hbm_HW_w  = N_g * d_out_g * 4           # write HW (d_out << d_in often)
    hbm_HW_r  = N_g * d_out_g * 4           # read HW for SpMM
    hbm_Hout2 = N_g * d_out_g * 4           # write H'
    hbm_fused = hbm_adj + hbm_H2 + hbm_W2 + hbm_HW_w + hbm_HW_r + hbm_Hout2

    saving = (hbm_unfused - hbm_fused) / hbm_unfused * 100

    print(f"  {label:<24}  {N_g:>6}  {nnz_g:>8}  "
          f"{hbm_unfused/1e6:>14.2f}  {hbm_fused/1e6:>14.2f}  "
          f"{saving:>5.1f}%")

print()
print("  Fusion savings = eliminated Z tensor (N × d_in bytes per layer).")
print("  Larger savings when d_in >> d_out (bottleneck architecture).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Full GNN pipeline annotation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — GNN Forward Pass: All Ops with AI Annotation")
print("━" * 68)
print()

N_gnn   = 1024
d_list  = [64, 64, 32]   # 2-layer GNN: 64 → 64 → 32
avg_deg_g = 15
nnz_gnn = N_gnn * avg_deg_g

print(f"  2-layer GCN: N={N_gnn}, avg_degree={avg_deg_g}, nnz={nnz_gnn}")
print(f"  Feature dims: input={d_list[0]} → hidden={d_list[1]} → output={d_list[2]}")
print()

ops_pipeline = []
for layer_idx in range(2):
    d_i = d_list[layer_idx]
    d_o = d_list[layer_idx+1]

    # SpMM: Â × H
    flops_spmm = 2 * nnz_gnn * d_i
    hbm_spmm   = nnz_gnn*8 + (N_gnn+1)*4 + N_gnn*d_i*4 + N_gnn*d_i*4
    ai_spmm    = flops_spmm / hbm_spmm

    # Dense GEMM: Z × W
    flops_gemm = 2 * N_gnn * d_i * d_o
    hbm_gemm   = N_gnn*d_i*4 + d_i*d_o*4 + N_gnn*d_o*4
    ai_gemm    = flops_gemm / hbm_gemm

    # ReLU
    flops_relu = N_gnn * d_o
    hbm_relu   = 2 * N_gnn * d_o * 4
    ai_relu    = flops_relu / hbm_relu

    ops_pipeline.extend([
        (f"Layer {layer_idx+1}: SpMM  (Â×H)",  flops_spmm, hbm_spmm, ai_spmm, "sparse"),
        (f"Layer {layer_idx+1}: GEMM  (Z×W)",  flops_gemm, hbm_gemm, ai_gemm, "dense TC"),
        (f"Layer {layer_idx+1}: ReLU",          flops_relu, hbm_relu, ai_relu, "elementwise"),
    ])

H100_RIDGE_FP32 = H100_FP32_TFLOPS * 1e12 / (H100_BW_GBS * 1e9)

print(f"  {'Operation':<30}  {'FLOPs':>12}  {'HBM (KB)':>10}  "
      f"{'AI':>8}  {'Bound':>8}  {'Kernel type'}")
print("  " + "─" * 76)

total_flops = total_hbm = 0
for name, flops, hbm, ai, ktype in ops_pipeline:
    bound = "🟢 cmp" if ai > H100_RIDGE_FP32 else "🔴 mem"
    total_flops += flops
    total_hbm   += hbm
    print(f"  {name:<30}  {flops/1e6:>10.2f}M  {hbm/1e3:>8.1f}  "
          f"{ai:>8.2f}  {bound:>8}  {ktype}")

print("  " + "─" * 76)
total_ai = total_flops / total_hbm
print(f"  {'TOTAL':<30}  {total_flops/1e6:>10.2f}M  {total_hbm/1e3:>8.1f}  "
      f"{total_ai:>8.2f}")
print()
print(f"  The GEMM layers are compute-bound at large N×d (tensor cores).")
print(f"  The SpMM layers are always memory-bound (no tensor core for CSR).")
print(f"  BSR with block_size=16 can bring SpMM to tensor-core territory for FEM.")
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