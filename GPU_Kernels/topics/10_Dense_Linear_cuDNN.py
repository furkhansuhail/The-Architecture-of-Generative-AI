"""
cuDNN — CUDA Deep Neural Network Library
==========================================

cuDNN (CUDA Deep Neural Network library) is NVIDIA's closed-source,
hand-optimised library of GPU primitives for deep learning. It is the
hidden engine beneath every major ML framework — when PyTorch executes
a convolution, it calls cuDNN. When TensorFlow runs batch normalisation,
it calls cuDNN. When TensorRT compiles a transformer, it calls cuDNN
for attention operations.

cuDNN is NOT an ML framework. It has no concept of backpropagation,
layers, or models. It is a collection of approximately 200 carefully
engineered CUDA kernels for the specific operations that dominate neural
network computation: convolution, matrix multiplication, attention,
activation functions, normalisation, RNNs, and their backward passes.

What makes cuDNN special is its algorithm selection engine: for any
given convolution shape (batch size, channel count, kernel size, etc.),
cuDNN internally benchmarks multiple CUDA algorithm implementations and
selects the fastest one for the specific GPU, input shape, and precision.
This auto-tuning — happening transparently every time you call a forward
pass — is why "just install CUDA and PyTorch" produces code that runs at
80-90% of theoretical peak performance without any manual kernel writing.

Understanding cuDNN means understanding the contract between ML frameworks
and GPU hardware — the layer where high-level operations become the
hand-optimised CUDA kernels that actually execute on silicon.

"""

import textwrap
import re

TOPIC_NAME   = "cuDNN — CUDA Deep Neural Network Library"
DISPLAY_NAME = "10 · cuDNN"
ICON         = "🔷"
SUBTITLE     = "NVIDIA's Hand-Optimised GPU Primitives for Deep Learning"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT cuDNN IS AND WHERE IT SITS IN THE STACK

### cuDNN's Position

    The ML software stack from Python to silicon:

    Python / JAX / PyTorch / TensorFlow  ← user writes code here
        ↓  framework dispatch
    cuDNN API calls                       ← cuDNN handles this layer
        ↓  internal algorithm selection
    Optimised CUDA kernels                ← hand-written PTX/CUDA C
        ↓  GPU hardware
    NVIDIA GPU (Tensor Cores, HBM, SMs)  ← executes here

    cuDNN sits between the framework and the hardware.
    Frameworks don't write their own convolution kernels — they call cuDNN.
    cuDNN doesn't understand "neural networks" — it just executes primitives.

### What cuDNN Provides

    cuDNN offers GPU-optimised implementations of:

    CONVOLUTION:
        Forward pass:       output = conv(input, filter)
        Backward (data):    ∂L/∂input  from ∂L/∂output
        Backward (filter):  ∂L/∂filter from ∂L/∂output
        Multiple algorithms: direct, FFT, Winograd, implicit GEMM

    ACTIVATION FUNCTIONS:
        Sigmoid, ReLU, Tanh, ELU, Softplus, SWISH, Clip
        Forward and backward in one kernel (fused)

    NORMALISATION:
        Batch Normalisation (forward: inference and training, backward)
        Layer Normalisation (forward and backward)
        Instance Normalisation
        Spatial Batch Normalisation (per-channel for CNNs)

    POOLING:
        Max pooling, Average pooling, Unpooling
        1D, 2D, and 3D variants
        Forward and backward

    ATTENTION (cuDNN 8.9+):
        Flash Attention (memory-efficient scaled dot-product attention)
        Multi-head attention forward and backward
        Ragged (variable-length sequence) attention

    RECURRENT NEURAL NETWORKS:
        LSTM, GRU, Vanilla RNN
        Bidirectional variants
        Multi-layer stacking
        Packed variable-length sequences

    LINEAR ALGEBRA:
        cuDNN wraps cuBLAS for GEMM operations used in FC layers, attention

    SOFTMAX:
        Full softmax (all classes)
        Log softmax
        Fast softmax (approximate)

### cuDNN vs cuBLAS vs CUTLASS

    cuBLAS:   NVIDIA's linear algebra library. GEMM, GEMV, TRSM, etc.
              Pure matrix/vector operations. No ML-specific primitives.
              Used by: linear layers, attention projections.
              Level: closer to hardware, less ML-specific.

    cuDNN:    Wraps cuBLAS for GEMM, adds ML-specific ops on top.
              Convolution, normalisation, attention, RNNs.
              Algorithm auto-selection engine.
              Used by: ALL major ML frameworks for high-level ops.
              Level: high-level ML operations.

    CUTLASS:  C++ template library for writing custom CUDA kernels.
              Not a black box — you write the kernel using CUTLASS abstractions.
              Used by: TensorRT, custom ML compilers, research kernels.
              Level: close to hardware, but ergonomic C++ API.

    Relationship:
        cuDNN internally uses cuBLAS and its own custom kernels.
        cuDNN itself is written using CUDA primitives (similar to what CUTLASS exposes).
        TensorRT and torch.compile may bypass cuDNN for some ops and call
        custom Triton/CUTLASS kernels that are faster for specific shapes.


##### PART 2 — cuDNN CONVOLUTION: ALGORITHMS AND SELECTION

### The Convolution Operation in cuDNN

    A 2D convolution (the most common in CNNs) has parameters:
        N:   batch size
        C:   input channels
        H,W: input height and width
        K:   output channels (number of filters)
        R,S: filter height and width
        P,Q: output height and width

    Compute:   output[n,k,p,q] = sum_{c,r,s} input[n,c,p*u+r,q*v+s] × filter[k,c,r,s]
    where u,v are stride values.

    FLOPs:     2 × N × K × P × Q × C × R × S

    For ResNet-50 first conv layer (N=32, C=3, H=W=224, K=64, R=S=7):
        FLOPs ≈ 2 × 32 × 64 × 112 × 112 × 3 × 7 × 7 ≈ 58 GFLOPs

### cuDNN's Convolution Algorithms

    cuDNN implements multiple convolution algorithms. The "best" depends
    on the specific shape, GPU, and available memory.

    1. IMPLICIT GEMM (CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM):
        Transforms convolution into a matrix multiplication implicitly.
        Input:  "im2col" transformation (expand image patches into columns)
                But does NOT materialise the im2col matrix in memory.
        Pros:   No extra memory, works for all sizes.
        Cons:   Slowest for most cases.

    2. IMPLICIT PRECOMP GEMM (CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM):
        Same as implicit GEMM but precomputes workspace for the index transformation.
        Generally faster than IMPLICIT_GEMM.
        Requires small workspace buffer.

    3. EXPLICIT GEMM (via im2col + cuBLAS GEMM):
        Actually materialises the im2col matrix.
        im2col: unroll input patches → matrix of shape (N×P×Q, C×R×S)
        Then: GEMM between im2col matrix and filter matrix.
        Pros:   Uses highly optimised cuBLAS GEMM.
        Cons:   Extra memory for the unrolled matrix (can be C×R×S × larger!)

    4. FFT (CUDNN_CONVOLUTION_FWD_ALGO_FFT):
        Convert input and filters to frequency domain via FFT.
        Element-wise multiply in frequency domain.
        Convert back via inverse FFT.
        Complexity: O(N×K×P×Q×log(P×Q)) instead of O(N×K×P×Q×C×R×S)
        Pros:   Fast for LARGE filters (R=S=11, 13, etc.)
        Cons:   Slow for small filters (R=S=1, 3). Large memory for FFT workspace.

    5. FFT TILING (CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING):
        Like FFT but tiles the spatial dimensions.
        Better for inputs that don't fit in memory as a whole FFT.

    6. WINOGRAD (CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD):
        Fast algorithm specifically for small filters (R=S=3, R=S=5).
        Winograd's minimal filtering algorithm reduces the number of multiplications.
        For a 3×3 filter: reduces from 9 multiplications to 4 (F(2×2, 3×3) transform).
        Pros:   ~2× faster than GEMM for 3×3 convolutions (very common in ResNets!)
        Cons:   Only works for specific filter sizes. Not numerically exact.

    7. WINOGRAD NON-FUSED:
        Winograd with transforms and multiplication in separate kernels.
        More memory, but sometimes faster due to better pipelining.

### Algorithm Selection: The Heuristic Engine

    cuDNN has two modes for algorithm selection:

    HEURISTIC (cudnnGetConvolutionForwardAlgorithm_v7):
        Uses a learned cost model to predict the best algorithm.
        Fast — no benchmarking needed.
        Result may not be globally optimal.

    EXHAUSTIVE SEARCH (cudnnFindConvolutionForwardAlgorithm):
        Benchmarks ALL available algorithms with the actual input.
        Returns the list sorted by measured execution time.
        SLOW (seconds on first call), but finds the true best.
        Results are cached in the framework (PyTorch's CUDNN_BENCHMARK mode).

    PyTorch benchmark mode:
        torch.backends.cudnn.benchmark = True
        → On the FIRST call for a new shape, runs exhaustive search.
        → Caches the best algorithm for that shape.
        → All subsequent calls with the SAME shape use the cached algorithm.
        → Fixed shapes (fixed batch, fixed spatial dims): HUGE speedup.
        → Variable shapes: re-benchmarks each unique shape (can be slow).

### Workspace Memory

    Several cuDNN algorithms require a temporary scratch buffer ("workspace"):
        FFT: needs memory for the frequency-domain representation
        Winograd: needs memory for transformed inputs and filters
        Implicit GEMM with precomputed indices: small workspace

    cuDNN allocates workspace from a user-provided pointer:
        cudnnGetConvolutionForwardWorkspaceSize() → tells you how much
        cudaMalloc(&workspace, workspace_size)    → you allocate
        cudnnConvolutionForward(..., workspace, workspace_size) → you provide

    PyTorch and TF manage this automatically. But if you're calling cuDNN
    directly (e.g., from C++ code), you must manage workspace yourself.

    Workspace memory can be significant:
        FFT for a 224×224 input might need several GB.
        cuDNN may choose a slower algorithm if workspace is limited.


##### PART 3 — cuDNN TENSOR DESCRIPTORS AND THE API MODEL

### The Descriptor Pattern

    cuDNN uses an opaque-handle design. Every operation requires:
        1. A cuDNN handle (library context, owns GPU resources)
        2. Descriptors (opaque structs describing data layout and op params)
        3. Raw GPU pointers (your actual data — cuDNN never owns your memory)

    This separation of description from data allows:
        - Reusing descriptors across multiple calls (same shape, different data)
        - cuDNN to optimise the descriptor at creation time
        - Clean separation between configuration and execution

### Tensor Descriptors (cudnnTensorDescriptor_t)

    A tensor descriptor describes the shape, datatype, and memory layout:

        cudnnTensorDescriptor_t input_desc;
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnSetTensor4dDescriptor(
            input_desc,
            CUDNN_TENSOR_NCHW,          // memory layout: N×C×H×W
            CUDNN_DATA_FLOAT,            // data type: float32
            N, C, H, W                  // dimensions
        );

    Data formats:
        CUDNN_TENSOR_NCHW:   batch × channels × height × width (default on GPU)
                             Channels are contiguous in memory per spatial location.
                             Best for most cuDNN operations on NVIDIA GPU.

        CUDNN_TENSOR_NHWC:   batch × height × width × channels
                             Spatial dims contiguous per channel location.
                             Better for some TPU workloads, and cuDNN INT8 ops.

        CUDNN_TENSOR_NCHW_VECT_C:  Vectorised channel format.
                             Channels packed in groups of 4 (int8) or 32 (int8).
                             Used for INT8 inference — enables Tensor Core efficiency.

    For custom strides (non-standard layout):
        cudnnSetTensorNdDescriptorEx() with explicit stride specification.
        Allows cuDNN to work with transposed, padded, or interleaved tensors.

### Filter Descriptors and Convolution Descriptors

    Filter descriptor describes the convolution weights:
        cudnnFilterDescriptor_t filter_desc;
        cudnnSetFilter4dDescriptor(
            filter_desc,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            K, C, R, S      // out_channels × in_channels × height × width
        );

    Convolution descriptor describes the operation parameters:
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            pad_h, pad_w,      // zero-padding
            stride_h, stride_w, // stride
            dilation_h, dilation_w, // dilation (for dilated/atrous convolution)
            CUDNN_CROSS_CORRELATION, // vs CUDNN_CONVOLUTION (flipped filter)
            CUDNN_DATA_FLOAT         // compute type
        );

        // Enable Tensor Core usage (Ampere: BF16, INT8):
        cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

### The Complete Convolution Forward Pass (C++ pseudocode)

    // 1. Create handle (once per session)
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // 2. Create and set descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    // ... (set as shown above)

    // 3. Get output dimensions
    int n_out, c_out, h_out, w_out;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc,
                                           &n_out, &c_out, &h_out, &w_out);

    // 4. Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(
        handle, x_desc, w_desc, conv_desc, y_desc,
        1, &returned_count, &algo_perf);
    algo = algo_perf.algo;

    // 5. Allocate workspace
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size);
    void *workspace;
    cudaMalloc(&workspace, workspace_size);

    // 6. Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(
        handle,
        &alpha,             // scale factor for input
        x_desc, x_data,    // input descriptor + GPU pointer
        w_desc, w_data,    // filter descriptor + GPU pointer
        conv_desc,          // convolution parameters
        algo,               // algorithm to use
        workspace, workspace_size,  // scratch memory
        &beta,              // scale factor for existing output (0 = overwrite)
        y_desc, y_data     // output descriptor + GPU pointer
    );

    // 7. Cleanup
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    // ... destroy other descriptors
    cudnnDestroy(handle);


##### PART 4 — cuDNN BATCH NORMALISATION: TRAINING VS INFERENCE

### Batch Normalisation Mathematics

    Given input X of shape (N, C, H, W):

    Training forward:
        μ_c  = (1/NHW) × sum_{n,h,w} X[n,c,h,w]        (per-channel mean)
        σ²_c = (1/NHW) × sum_{n,h,w} (X-μ_c)²           (per-channel variance)
        X̂   = (X - μ_c) / sqrt(σ²_c + ε)               (normalise)
        Y    = γ × X̂ + β                                 (scale and shift)

        Also updates running statistics:
        running_mean = (1-momentum) × running_mean + momentum × μ_c
        running_var  = (1-momentum) × running_var  + momentum × σ²_c

    Inference forward:
        Y = γ × (X - running_mean) / sqrt(running_var + ε) + β
        (Uses SAVED running statistics — NOT computed from the batch)
        Equivalent to a single linear transform: Y = a × X + b
        where a = γ / sqrt(running_var + ε), b = β - γ × running_mean / sqrt(...)

    Backward (training):
        ∂L/∂X:   complex expression involving μ, σ² computed in forward
        ∂L/∂γ:   sum over batch and spatial dims
        ∂L/∂β:   sum over batch and spatial dims

### cuDNN Batch Norm Modes

    CUDNN_BATCHNORM_PER_ACTIVATION:
        Normalise per neuron (not per channel).
        Used for fully-connected layers.
        Statistics shape: (1, C, H, W) — one stat per spatial+channel location.

    CUDNN_BATCHNORM_SPATIAL:
        Normalise per channel across N, H, W dimensions.
        Standard mode for convolutional layers.
        Statistics shape: (1, C, 1, 1) — one stat per channel.
        This is what PyTorch's nn.BatchNorm2d uses.

    CUDNN_BATCHNORM_SPATIAL_PERSISTENT:
        Same as SPATIAL but uses a different implementation optimised for
        small batch sizes that fits statistics in register files.
        Faster on Volta+ for batch sizes < 32.

### BN-Conv Fusion in cuDNN

    For inference, BatchNorm can be fused into the preceding Conv:
    Instead of:
        y = conv(x, W, b)
        y = gamma * (y - running_mean) / sqrt(running_var + eps) + beta

    Compute fused weights:
        W_fused = W × (gamma / sqrt(running_var + eps))
        b_fused = (b - running_mean) × (gamma / sqrt(running_var + eps)) + beta

    Then:
        y = conv(x, W_fused, b_fused)   (ONE operation instead of two!)

    cuDNN 8.0+ supports this fusion natively via the ConvBiasAct API.
    TensorRT performs this fusion automatically.
    Speedup: ~30-50% for typical ResNet convolution blocks.


##### PART 5 — cuDNN ATTENTION: FLASH ATTENTION IN cuDNN 8.9+

### cuDNN's Graph API for Attention

    cuDNN 8.9 introduced the "Flash Attention" implementation via the
    graph API. This is the backend that PyTorch's
    F.scaled_dot_product_attention() uses when available.

    The cuDNN graph API (different from the older descriptor API):
        Instead of separate descriptor objects, you build a computation GRAPH
        by connecting tensor nodes with operation nodes.

    Attention operation in cuDNN:
        graph = cudnn_frontend::graph::Graph();

        // Create tensor nodes (input)
        auto Q = graph.tensor(Q_descriptor);  // queries
        auto K = graph.tensor(K_descriptor);  // keys
        auto V = graph.tensor(V_descriptor);  // values

        // Create attention operation node
        auto [O, stats] = graph.sdpa(Q, K, V, {
            .is_inference    = true,
            .attn_scale      = 1.0f / sqrt(head_dim),
            .use_causal_mask = true,   // for autoregressive LLM
        });

        graph.validate();             // check graph is valid
        graph.build_operation_graph(handle);
        graph.execute(handle, {Q, K, V} → {O}, workspace);

### What cuDNN Flash Attention Does

    cuDNN's flash attention implementation mirrors the algorithmic ideas
    of FlashAttention (Dao et al.):
        - Tiles Q, K, V into SRAM-sized blocks
        - Computes partial softmax with online normalisation
        - Never materialises the N×N attention matrix in HBM
        - Supports: causal masks, dropout, sliding window attention

    Added in cuDNN 9.x:
        - Paged KV cache attention (for LLM inference with PagedAttention)
        - Variable-length sequence attention (ragged batches)
        - FP8 attention (H100 Tensor Cores at double speed)
        - Multi-query attention (MQA) and grouped-query attention (GQA)

### cuDNN Frontend Library (cudnn-frontend)

    Raw cuDNN C API is verbose. NVIDIA provides a header-only C++ wrapper:
    github.com/NVIDIA/cudnn-frontend

    Makes the graph API ergonomic:
        namespace fe = cudnn_frontend;

        auto graph = fe::graph::Graph();
        graph.set_io_data_type(fe::DataType_t::HALF)
             .set_compute_data_type(fe::DataType_t::FLOAT);

        auto Q = graph.tensor(fe::graph::Tensor_attributes()
                     .set_name("Q").set_dim({B, H, S, D}));
        // ...
        auto [O, _] = graph.sdpa(Q, K, V,
                         fe::graph::SDPA_attributes()
                         .set_scale(scale)
                         .set_causal_mask(true));

        graph.validate();
        graph.build({handle}, {fe::HeurMode_t::A});  // find best algorithm
        graph.execute(handle, variant_pack, workspace);


##### PART 6 — cuDNN RECURRENT NEURAL NETWORKS

### cuDNN's RNN Implementation

    cuDNN provides highly optimised LSTM, GRU, and RNN implementations.
    These are NOT naive loop-over-timesteps implementations.

    Key optimisations:
        1. Weight transposition: weights are stored in a layout that enables
           coalesced memory access during the matrix multiplications.
        2. Persistent RNN: for small hidden sizes, the RNN kernel keeps weights
           in L2 cache across timesteps (avoids re-loading from HBM each step).
        3. Fused kernel: the gate computations (input gate, forget gate, cell gate,
           output gate for LSTM) are fused into one kernel.
        4. Multi-layer unrolling: in multi-layer RNNs, cuDNN pipelines the
           computation of consecutive layers to overlap communication.

    RNN descriptor parameters:
        - Hidden size (number of hidden units)
        - Number of layers
        - Direction: UNIDIRECTIONAL or BIDIRECTIONAL
        - RNN mode: LSTM, GRU, ReLU, Tanh
        - Algorithm: STANDARD or PERSIST (for persistent RNN)
        - Data type: FP16, FP32, FP64

### Persistent RNN

    Standard LSTM at each timestep:
        1. Load weight matrices from HBM (expensive!)
        2. Compute gates: sigmoid and tanh matrix multiplications
        3. Update cell state and hidden state

    Persistent LSTM (for hidden_size ≤ L2 cache):
        1. Load weight matrices ONCE into L2 cache (first timestep)
        2. For all subsequent timesteps: weights stay in cache!
        3. Massive bandwidth savings for long sequences

    Condition for persistent:
        total_weights_bytes ≤ L2_cache_size
        For LSTM: weights ≈ 4 × hidden_size × (input_size + hidden_size) × dtype
        A100 has 40 MB L2 cache → fits LSTM with hidden_size ≈ 1024 in FP32

    Speedup from persistent RNN:
        For long sequences (T=500+): 3–5× faster than standard LSTM
        For short sequences (T=16): minimal benefit (no reuse savings)

### Packed Sequences for Variable-Length RNNs

    NLP inputs often have different lengths in a batch.
    Padding wastes compute — cuDNN supports packed (ragged) sequences:

        cudnnRNNDataDescriptor_t data_desc;
        cudnnSetRNNDataDescriptor(
            data_desc,
            CUDNN_DATA_FLOAT,
            CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
            max_seq_length,    // longest sequence in batch
            batch_size,
            input_size,
            seq_length_array,  // array of per-sample lengths [15, 23, 8, 31, ...]
            NULL               // padding value
        );

    cuDNN skips computation for padded positions.
    Saves compute proportional to average padding ratio.


##### PART 7 — cuDNN VERSIONS, COMPATIBILITY AND CONFIGURATION

### cuDNN Version History (ML-relevant milestones)

    cuDNN 1 (2014):   First release. Convolution forward pass only.
    cuDNN 2 (2015):   Backward passes. Multi-GPU support.
    cuDNN 5 (2016):   Winograd convolution. Massive speedup for 3×3 convs.
    cuDNN 6 (2016):   Dilated convolution (atrous). RNN APIs.
    cuDNN 7 (2017):   Tensor Core support (Volta V100). Grouped convolution.
    cuDNN 8 (2020):   Graph API. INT8 conv. BF16 support (Ampere).
                      cudnn-frontend C++ wrapper released.
    cuDNN 8.6 (2022): Flash Attention early support. MHA primitives.
    cuDNN 8.9 (2023): Full Flash Attention. Paged attention. GQA support.
    cuDNN 9.x (2024): FP8 attention (H100). Runtime fusion engine.
                      Improved multi-head attention with all mask types.

### Compatibility Rules

    cuDNN version is tied to CUDA version:
        cuDNN 9.x:    requires CUDA 12.x
        cuDNN 8.x:    requires CUDA 11.x or 12.x
        cuDNN 7.x:    requires CUDA 10.x or 11.x

    PyTorch ships its own bundled cuDNN:
        pip install torch → installs cuDNN ~7.6 or ~8.x (version varies)
        The bundled cuDNN is in: torch/lib/libcudnn.so.X

    Check cuDNN version in PyTorch:
        import torch
        print(torch.backends.cudnn.version())    # e.g., 8902 = cuDNN 8.9.2
        print(torch.backends.cudnn.is_available())

### cuDNN Configuration Flags in PyTorch

    torch.backends.cudnn.benchmark = True
        → Run exhaustive algorithm search on first call for each shape.
        → Cache results. Subsequent calls with same shape are fast.
        → BEST for fixed-shape training (CNNs with fixed image size).
        → BAD for variable-shape inputs (causes repeated benchmarking).

    torch.backends.cudnn.deterministic = True
        → Force deterministic (reproducible) algorithms only.
        → Some cuDNN algorithms are non-deterministic (random atomic ops).
        → Slightly slower than allowing non-deterministic algorithms.
        → Use when you need bit-exact reproducibility for debugging.

    torch.backends.cudnn.allow_tf32 = True (default on Ampere+)
        → Allow TF32 math in cuDNN (for convolutions).
        → TF32: FP32 exponent + 10 mantissa bits (same as FP16 mantissa).
        → 19× speedup vs FP32 on Ampere Tensor Cores.
        → Slight numerical difference (typically < 1% accuracy impact).

    torch.backends.cudnn.enabled = True (default)
        → When False, PyTorch uses its own (slower) CPU/CUDA implementations.
        → Only disable for debugging cuDNN-specific issues.

### cuDNN and Memory Fragmentation

    cuDNN may hold onto workspace memory between calls.
    PyTorch's CUDA memory allocator manages this via a memory pool.

    Memory management interaction:
        1. PyTorch allocates workspace from its CUDA memory pool.
        2. cuDNN uses the workspace during execution.
        3. PyTorch returns workspace to pool after the call.
        4. Pool does NOT return to CUDA immediately (reserved for reuse).

    Monitoring cuDNN workspace usage:
        torch.cuda.memory_stats()["reserved_bytes.all.current"]
        → includes workspace + activation memory + weight memory

    Reducing memory with smaller workspace:
        In PyTorch, you can limit workspace via environment:
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"


##### PART 8 — cuDNN IN PRACTICE: PYTORCH INTEGRATION

### How PyTorch Uses cuDNN

    When you call torch.nn.Conv2d.forward():
        1. PyTorch checks: is CUDA available and cuDNN enabled?
        2. PyTorch calls at::cudnn_convolution() (C++ backend)
        3. at::cudnn_convolution() calls:
            a. cudnnSetTensor4dDescriptor() for input/output/filter
            b. cudnnGetConvolutionForwardAlgorithm() or finds cached algo
            c. cudnnGetConvolutionForwardWorkspaceSize()
            d. Allocates workspace from PyTorch's memory pool
            e. cudnnConvolutionForward() ← actual GPU work
        4. Returns output tensor to Python

    PyTorch caches algorithm selections in:
        at::cuda::CUDAConvAlgos (a hashmap of (shape, device) → algo)
        When benchmark=True: populated by cudnnFindConvolutionForwardAlgorithmEx

### The Fastest PyTorch Conv Setup

    import torch
    import torch.nn as nn

    # 1. Enable benchmark mode (one-time benchmark, then cached forever)
    torch.backends.cudnn.benchmark = True

    # 2. Use BF16 or FP16 for Tensor Core acceleration
    model = model.to(dtype=torch.bfloat16)

    # 3. Allow TF32 on Ampere (default, but make explicit)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 4. Use channels_last memory format (NHWC) for NVIDIA's preferred layout
    model = model.to(memory_format=torch.channels_last)
    input = input.to(memory_format=torch.channels_last)

    # 5. Compile for additional fusion
    model = torch.compile(model)

    # Result: often 3-5× faster than default FP32 NCHW eager mode

### Channels Last Memory Format (NHWC)

    PyTorch tensors default to NCHW (channels first).
    cuDNN 8+ can use NHWC internally for convolutions.

    Why NHWC is faster on modern NVIDIA GPUs:
        NCHW: adjacent memory = adjacent channels of the SAME spatial location
              For conv: needs to access H×W different memory locations per channel
        NHWC: adjacent memory = all channels at the SAME spatial location
              For conv: reads all C channels at once (coalesced memory access)

    Converting to channels_last:
        x = x.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)

    Performance impact:
        ResNet-50, A100, BF16: NHWC is ~20% faster than NCHW

### Flash Attention via PyTorch

    PyTorch 2.0+ wraps cuDNN flash attention:
        F.scaled_dot_product_attention(Q, K, V,
                                        attn_mask=None,
                                        dropout_p=0.0,
                                        is_causal=True)

    Backend selection priority:
        1. cuDNN Flash Attention (if cuDNN 8.9+ and shape is supported)
        2. FlashAttention-2 (if flash_attn package is installed)
        3. Memory-efficient attention (xformers, if available)
        4. Math attention (slow, O(N²) memory fallback)

    Check which backend is used:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(Q, K, V)

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · cuDNN via PyTorch — Benchmarking Algorithms and Configuration": {
        "description": (
            "Interact with cuDNN through PyTorch's high-level interface. "
            "Benchmark convolution performance across different cuDNN configurations. "
            "Show the impact of benchmark mode, memory formats (NCHW vs NHWC), "
            "and data types (FP32, FP16, BF16) on throughput and latency. "
            "Query which cuDNN algorithms are selected. "
            "Demonstrate F.scaled_dot_product_attention with cuDNN backend."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  cuDNN VIA PYTORCH — BENCHMARKING ALGORITHMS & CONFIG")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print(f"  PyTorch:     {torch.__version__}")
    print(f"  CUDA:        {torch.version.cuda}")
    print(f"  cuDNN:       {torch.backends.cudnn.version()}")
    print(f"  cuDNN avail: {torch.backends.cudnn.is_available()}")
    print(f"  Device:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    HAS_CUDA = torch.cuda.is_available()
    device   = "cuda" if HAS_CUDA else "cpu"
    print()
except ImportError:
    print("  PyTorch not installed: pip install torch")
    HAS_CUDA = False
    device   = "cpu"

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: cuDNN algorithm selection — benchmark vs heuristic
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — cuDNN benchmark mode: algorithm selection impact")
print("━" * 65)
print()

print("  cuDNN Algorithm Selection Modes:")
print()
print("  HEURISTIC (default, benchmark=False):")
print("    cuDNN uses a cost model to PREDICT the best algorithm.")
print("    No benchmarking → fast startup, may not use fastest algorithm.")
print("    Use when: variable input shapes, development, quick experiments.")
print()
print("  EXHAUSTIVE SEARCH (benchmark=True):")
print("    cuDNN runs ALL candidate algorithms with real inputs.")
print("    Selects the MEASURED fastest algorithm for this exact shape.")
print("    Caches result → only slow on FIRST call per shape.")
print("    Use when: fixed input shapes in production training.")
print()

ALGORITHM_TABLE = """
  Convolution Algorithms in cuDNN:

  Algorithm                    | Condition for best use
  ─────────────────────────────┼────────────────────────────────────────────
  IMPLICIT_GEMM                │ Small inputs, no workspace available
  IMPLICIT_PRECOMP_GEMM        │ General purpose, slight improvement over above
  EXPLICIT_GEMM (im2col)       │ When cuBLAS GEMM is fast on your GPU
  FFT                          │ Large spatial dims (H=W>64), large filters
  FFT_TILING                   │ Very large spatial dims (H=W>128)
  WINOGRAD                     │ 3×3 or 5×5 filters — up to 2× speedup!
  WINOGRAD_NONFUSED            │ When Winograd requires large workspace

  PyTorch algorithm names (from TORCH_CUDNN_V8_API_DEBUG=1):
    ConvolutionForward_0:  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
    ConvolutionForward_1:  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    ConvolutionForward_4:  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
    ConvolutionForward_5:  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
"""
print(ALGORITHM_TABLE)

if not HAS_CUDA:
    print("  (Skipping live benchmark — no CUDA GPU detected)")
    print()
    BENCHMARK_REF = """
  BENCHMARK MODE EXAMPLE:

  import torch
  import torch.nn as nn

  # --- Setup ---
  torch.backends.cudnn.benchmark = True   # enable exhaustive search
  conv = nn.Conv2d(64, 128, 3, padding=1).cuda().half()
  x = torch.randn(16, 64, 56, 56, device='cuda', dtype=torch.float16)

  # First call: cuDNN benchmarks all algorithms (~100ms overhead)
  print("First call (benchmark + execute)...")
  t0 = time.perf_counter()
  y = conv(x)
  torch.cuda.synchronize()
  print(f"  {(time.perf_counter()-t0)*1000:.1f}ms (includes benchmarking)")

  # Second call: uses cached algorithm (fast)
  print("Second call (cached algorithm)...")
  t0 = time.perf_counter()
  for _ in range(100):
      y = conv(x)
  torch.cuda.synchronize()
  print(f"  {(time.perf_counter()-t0)/100*1000:.3f}ms per call")

  # --- memory format ---
  # NCHW (default): torch.channels_first
  x_nchw = torch.randn(16, 64, 56, 56, device='cuda', dtype=torch.float16)

  # NHWC: torch.channels_last — faster on modern NVIDIA
  x_nhwc = x_nchw.to(memory_format=torch.channels_last)
  conv_nhwc = conv.to(memory_format=torch.channels_last)

  # Benchmark shows NHWC is typically 15-25% faster on A100
"""
    print(BENCHMARK_REF)
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    BATCH = 16
    C_IN  = 64
    H = W = 56
    C_OUT = 128

    # ── Benchmark mode comparison ──────────────────────────────────────────
    results = {}

    for benchmark_mode in [False, True]:
        torch.backends.cudnn.benchmark = benchmark_mode
        conv = nn.Conv2d(C_IN, C_OUT, kernel_size=3, padding=1, bias=False).to(device)

        # FP32 baseline
        x_fp32 = torch.randn(BATCH, C_IN, H, W, device=device)
        conv.eval()

        with torch.no_grad():
            # Warmup
            for _ in range(5): conv(x_fp32)
            if HAS_CUDA: torch.cuda.synchronize()

            REPS = 200
            t0 = time.perf_counter()
            for _ in range(REPS): conv(x_fp32)
            if HAS_CUDA: torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) / REPS * 1000

        results[f"benchmark={'on' if benchmark_mode else 'off'}"] = t_ms

    # ── Precision comparison ───────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    for dtype_name, dtype in [("FP32", torch.float32),
                                ("FP16", torch.float16),
                                ("BF16", torch.bfloat16)]:
        if not HAS_CUDA and dtype != torch.float32:
            continue
        conv_d = nn.Conv2d(C_IN, C_OUT, 3, padding=1, bias=False).to(device).to(dtype)
        x_d    = torch.randn(BATCH, C_IN, H, W, device=device, dtype=dtype)
        conv_d.eval()
        with torch.no_grad():
            for _ in range(10): conv_d(x_d)
            if HAS_CUDA: torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200): conv_d(x_d)
            if HAS_CUDA: torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) / 200 * 1000
        results[dtype_name] = t_ms

    # ── Memory format comparison ───────────────────────────────────────────
    if HAS_CUDA:
        for fmt_name, fmt in [("NCHW (default)", torch.contiguous_format),
                               ("NHWC (channels_last)", torch.channels_last)]:
            conv_f = nn.Conv2d(C_IN, C_OUT, 3, padding=1, bias=False).cuda()
            conv_f = conv_f.to(memory_format=fmt)
            x_f    = torch.randn(BATCH, C_IN, H, W, device="cuda")
            x_f    = x_f.to(memory_format=fmt)
            conv_f.eval()
            with torch.no_grad():
                for _ in range(10): conv_f(x_f)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200): conv_f(x_f)
                torch.cuda.synchronize()
                t_ms = (time.perf_counter() - t0) / 200 * 1000
            results[fmt_name] = t_ms

    print(f"  Conv2d({C_IN}→{C_OUT}, 3×3, batch={BATCH}, spatial={H}×{W})")
    print()
    base = results.get("benchmark=off", results.get("FP32", 1.0))
    print(f"  {'Configuration':<30} | {'Latency (ms)':>14} | {'Speedup':>9}")
    print(f"  {'─'*57}")
    for name, t_ms in results.items():
        print(f"  {name:<30} | {t_ms:14.4f} | {base/t_ms:9.2f}×")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: cuDNN Flash Attention via F.scaled_dot_product_attention
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — cuDNN Flash Attention: F.scaled_dot_product_attention")
print("━" * 65)
print()

print("  PyTorch 2.0+ unified attention API (uses cuDNN under the hood):")
print()

SDPA_GUIDE = """
  import torch
  import torch.nn.functional as F

  # --- Standard self-attention (F.scaled_dot_product_attention) ---
  B, H, T, D = 4, 8, 512, 64    # batch, heads, seq_len, head_dim

  Q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
  K = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
  V = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)

  # Automatically dispatches to the best available backend:
  output = F.scaled_dot_product_attention(Q, K, V,
      attn_mask = None,      # optional attention mask
      dropout_p = 0.0,       # dropout (0 for inference)
      is_causal = True,      # causal (autoregressive) mask
  )
  # output shape: (B, H, T, D)

  # --- Backend control ---
  # Check which backends are enabled:
  print(torch.backends.cuda.flash_sdp_enabled())       # cuDNN flash attention
  print(torch.backends.cuda.mem_efficient_sdp_enabled())# xformers mem-efficient
  print(torch.backends.cuda.math_sdp_enabled())         # naive O(N²)

  # Force a specific backend:
  with torch.backends.cuda.sdp_kernel(
      enable_flash=True,         # use cuDNN flash attention
      enable_math=False,         # disable naive implementation
      enable_mem_efficient=False # disable xformers
  ):
      output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

  # --- Memory comparison ---
  T_LONG = 4096
  Q_long = torch.randn(B, H, T_LONG, D, device='cuda', dtype=torch.float16)

  # Without flash: needs N×N attention matrix = (4096×4096×2 bytes × B × H) ≈ 4 GB
  # With flash: only O(T) memory = 4096×64×2 bytes × B × H ≈ 4 MB

  # Flash attention is O(N) memory vs O(N²) for naive!
"""
print(SDPA_GUIDE)

if HAS_CUDA:
    B, H, D = 4, 8, 64

    print("  Sequence length scaling (cuDNN Flash vs naive math):")
    print()
    print(f"  {'Seq len':>8} | {'Flash (ms)':>12} | {'Memory (MB)':>13} | "
          f"{'Naive (ms)':>12} | {'Flash speedup':>14}")
    print(f"  {'─'*68}")

    for T in [128, 256, 512, 1024, 2048]:
        Q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)

        # Flash attention
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(50):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
                                                  enable_mem_efficient=False):
                try:
                    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                except RuntimeError:
                    break
        torch.cuda.synchronize()
        t_flash = (time.perf_counter() - t0) / 50 * 1000
        mem_mb  = torch.cuda.max_memory_allocated() / 1e6

        # Naive math
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                                  enable_mem_efficient=False):
                out_n = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        torch.cuda.synchronize()
        t_naive = (time.perf_counter() - t0) / 50 * 1000

        speedup = t_naive / t_flash if t_flash > 0 else 0
        print(f"  {T:>8} | {t_flash:>12.4f} | {mem_mb:>13.1f} | "
              f"{t_naive:>12.4f} | {speedup:>13.2f}×")

    print()
    print("  Flash attention's speedup grows with sequence length:")
    print("  At T=2048: ~2-4× faster than naive, using ~100× less memory.")
    print("  This is the cuDNN flash attention running under the hood.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · cuDNN Convolution & BatchNorm — Algorithms, Shapes, and Fusion": {
        "description": (
            "Deep dive into cuDNN convolution and batch normalisation. "
            "Show how filter size affects algorithm selection (Winograd vs GEMM). "
            "Measure arithmetic intensity across conv shapes. "
            "Demonstrate BatchNorm fusion into Conv for inference. "
            "Profile cuDNN workspace usage. "
            "Compare cuDNN against naive Python implementations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  cuDNN CONVOLUTION & BATCHNORM — ALGORITHMS AND FUSION")
print("=" * 65)
print()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_CUDA = torch.cuda.is_available()
    device   = "cuda" if HAS_CUDA else "cpu"
    if HAS_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"  Device: {device}")
    if HAS_CUDA:
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    HAS_CUDA  = False
    device    = "cpu"
    print("  PyTorch not available — showing reference code only.")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Filter size impact on cuDNN algorithm selection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Filter size and cuDNN algorithm selection")
print("━" * 65)
print()

print("  Winograd algorithm: fastest for 3×3 and 5×5 filters")
print("  Key insight: Winograd reduces multiplications from R×S to 4 (F(2×2,3×3))")
print()

WINOGRAD_MATH = """
  Standard direct convolution (3×3 filter):
    Each output element = 9 multiply-accumulate operations
    Total multiplications per output pixel: 9

  Winograd F(2×2, 3×3) transform:
    Transforms 4×4 input tile and 3×3 filter to 4×4 frequency domain
    Multiplies element-wise in frequency domain: 16 multiplications
    Transforms result back: produces 2×2 output tile
    So: 2×2=4 output pixels from 16 multiplications = 4 mult/pixel
    vs standard: 9 multiplications per pixel
    → ~2.25× fewer multiplications!

  But Winograd has overhead:
    - Pre-transforms: O(N×C×tiles × 4×4) additions
    - Post-transforms: O(N×K×tiles × 2×2) additions
    - For small N or C, transforms dominate → Winograd is NOT faster
    - For large N, C, K, P, Q → Winograd wins by ~2×

  Practical rule: cuDNN selects Winograd when:
    - Filter size = 3×3 or 5×5
    - Channel count >= 32 (enough parallelism to hide transform overhead)
    - Spatial output >= 7×7 (enough tiles to amortise overhead)
"""
print(WINOGRAD_MATH)

print("  Benchmark: different filter sizes (cuDNN auto-selects algorithm)")
print()

if HAS_TORCH and HAS_CUDA:
    BATCH = 32
    C_IN  = 64
    C_OUT = 128
    H = W = 56

    print(f"  Input: ({BATCH}, {C_IN}, {H}, {W}), Output channels: {C_OUT}, BF16")
    print()
    print(f"  {'Filter':>10} | {'Time (ms)':>12} | {'GFLOP/s':>10} | "
          f"{'Notes'}")
    print(f"  {'─'*55}")

    for kernel in [1, 3, 5, 7, 11]:
        pad = kernel // 2
        conv = nn.Conv2d(C_IN, C_OUT, kernel, padding=pad, bias=False
                          ).to(device).to(torch.bfloat16).eval()
        x = torch.randn(BATCH, C_IN, H, W, device=device, dtype=torch.bfloat16)

        # Compute FLOPs
        out_h = H  # same padding
        out_w = W
        flops = 2 * BATCH * C_OUT * out_h * out_w * C_IN * kernel * kernel
        gflops = flops / 1e9

        with torch.no_grad():
            for _ in range(10): conv(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200): conv(x)
            torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) / 200 * 1000

        gflops_per_s = gflops / (t_ms / 1000)
        notes = "→ Winograd likely" if kernel in [3, 5] else "→ GEMM/Direct"
        print(f"  {str(kernel)+'×'+str(kernel):>10} | {t_ms:>12.4f} | "
              f"{gflops_per_s:>10.1f} | {notes}")

    print()
    print("  3×3 and 5×5 filters often show higher GFLOP/s because")
    print("  cuDNN selects Winograd — fewer actual multiply operations,")
    print("  making the effective throughput appear higher.")
else:
    print("  (Requires CUDA GPU for live benchmark)")
    print()
    FILTER_SPEED_REF = """
  Approximate throughput (A100 80GB, BF16, N=32, C=64→128, HxW=56×56):
  Filter  Algorithm      Approx speed    Notes
  1×1     Implicit GEMM  ~400 TFLOPS     Efficient matmul
  3×3     Winograd       ~350 TFLOPS     Fewer mult ops (looks slower in raw TFLOPS
                                         but fewer actual operations done)
  5×5     Winograd       ~300 TFLOPS     Same idea
  7×7     FFT/GEMM       ~180 TFLOPS     Falls back to slower algorithm
  11×11   FFT            ~100 TFLOPS     FFT overhead dominates for large filters
"""
    print(FILTER_SPEED_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: BatchNorm fusion into Conv (inference optimisation)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — BatchNorm fusion into Conv (cuDNN inference fusion)")
print("━" * 65)
print()

print("  BatchNorm inference: Y = γ × (X - μ) / σ + β")
print("  This is a linear transform per channel: Y = a × X + b")
print("  Can be folded into the preceding conv weights!")
print()

FUSION_MATH = """
  Standard Conv + BN (two operations):
    Y_conv = X ⊗ W_conv + b_conv          (convolution)
    Y_bn   = γ × (Y_conv - μ) / σ + β    (batch norm)

  Fused: absorb BN into conv weights
    W_fused[k] = W_conv[k] × γ[k] / σ[k]
    b_fused[k] = (b_conv[k] - μ[k]) × γ[k] / σ[k] + β[k]
    Y = X ⊗ W_fused + b_fused              (ONE operation)

  Memory savings:
    Without fusion: intermediate Y_conv stored to HBM then read back
    With fusion:    Y_conv never written to HBM — computed in-flight!

  Speedup: typically 20–40% for the Conv+BN block in inference
"""
print(FUSION_MATH)

if HAS_TORCH:
    def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """
        Fold BatchNorm statistics into Conv weights for inference.
        This is what TensorRT and PyTorch torch.jit.freeze() do automatically.
        """
        # Get BN parameters
        gamma    = bn.weight.data
        beta     = bn.bias.data
        mean     = bn.running_mean.data
        var      = bn.running_var.data
        eps      = bn.eps

        # Compute the per-channel scale factor
        scale    = gamma / torch.sqrt(var + eps)   # shape: (C_out,)

        # Fuse into conv weights
        # W_conv shape: (C_out, C_in, H, W)
        W_fused  = conv.weight.data * scale[:, None, None, None]

        # Fuse into conv bias
        if conv.bias is not None:
            b_conv = conv.bias.data
        else:
            b_conv = torch.zeros(conv.out_channels)

        b_fused  = (b_conv - mean) * scale + beta

        # Create fused conv layer
        fused = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, bias=True
        )
        fused.weight.data = W_fused
        fused.bias.data   = b_fused
        return fused.to(next(conv.parameters()).device)

    # Create a Conv + BN block
    C_IN, C_OUT = 64, 128
    BATCH       = 16

    conv  = nn.Conv2d(C_IN, C_OUT, 3, padding=1, bias=True)
    bn    = nn.BatchNorm2d(C_OUT)
    bn.eval()   # freeze running statistics

    # Initialise BN with typical trained statistics
    bn.running_mean.data = torch.randn(C_OUT) * 0.1
    bn.running_var.data  = torch.abs(torch.randn(C_OUT)) + 0.5
    bn.weight.data       = torch.ones(C_OUT) + torch.randn(C_OUT) * 0.1
    bn.bias.data         = torch.randn(C_OUT) * 0.1

    # Test input
    x = torch.randn(BATCH, C_IN, 28, 28).to(device)
    conv = conv.to(device).eval()
    bn   = bn.to(device).eval()

    # Fuse
    fused_conv = fuse_conv_bn(conv, bn).eval()

    # Verify outputs match
    with torch.no_grad():
        out_original = bn(conv(x))
        out_fused    = fused_conv(x)

    max_diff = (out_original - out_fused).abs().max().item()
    print(f"  Fusion verification:")
    print(f"    Max absolute difference: {max_diff:.2e}  "
          f"{'✅ (within float32 tolerance)' if max_diff < 1e-4 else '❌'}")
    print()

    if HAS_CUDA:
        x = x.cuda()
        conv_cuda  = conv.cuda()
        bn_cuda    = bn.cuda()
        fused_cuda = fused_conv.cuda()
        REPS = 500

        # Benchmark: Conv + BN separately
        with torch.no_grad():
            for _ in range(20): bn_cuda(conv_cuda(x))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPS): bn_cuda(conv_cuda(x))
            torch.cuda.synchronize()
            t_separate = (time.perf_counter() - t0) / REPS * 1000

        # Benchmark: Fused Conv+BN
        with torch.no_grad():
            for _ in range(20): fused_cuda(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPS): fused_cuda(x)
            torch.cuda.synchronize()
            t_fused = (time.perf_counter() - t0) / REPS * 1000

        print(f"  Performance comparison (Conv({C_IN}→{C_OUT}, 3×3) + BN, "
              f"batch={BATCH}, spatial=28×28):")
        print(f"    Separate Conv + BN: {t_separate:.4f} ms")
        print(f"    Fused Conv (BN absorbed): {t_fused:.4f} ms")
        print(f"    Speedup: {t_separate/t_fused:.2f}×")
        print()

    # PyTorch automatic fusion
    print("  PyTorch automatic BN fusion (torch.jit.freeze):")
    FUSION_AUTO = """
  import torch
  import torch.nn as nn

  class ConvBNReLU(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv = nn.Conv2d(64, 128, 3, padding=1)
          self.bn   = nn.BatchNorm2d(128)
      def forward(self, x):
          return torch.relu(self.bn(self.conv(x)))

  model = ConvBNReLU().cuda().eval()

  # Method 1: torch.jit.trace + freeze (fuses BN into conv)
  scripted = torch.jit.trace(model, torch.randn(1, 64, 56, 56, device='cuda'))
  frozen   = torch.jit.freeze(scripted)
  # frozen now contains ONLY conv (BN parameters folded in)

  # Method 2: torch.compile (fuses ops + kernels)
  compiled = torch.compile(model)
  # torch.compile finds the conv+bn+relu pattern and emits one fused kernel

  # Method 3: TensorRT (most aggressive fusion)
  # TensorRT auto-detects Conv+BN+ReLU and fuses to single CudnnConvBiasActLayer
"""
    print(FUSION_AUTO)

    # ── Section 3: cuDNN workspace memory ─────────────────────────────────
    print("━" * 65)
    print("  SECTION 3 — cuDNN workspace: memory management")
    print("━" * 65)
    print()

    print("  cuDNN algorithms require scratch memory (workspace).")
    print("  PyTorch manages this automatically from its CUDA memory pool.")
    print()

    if HAS_CUDA:
        # Measure peak memory for different conv configurations
        configs = [
            ("Small conv (3×3, C=64→128)",  64, 128, 3, 56),
            ("Large conv (3×3, C=256→512)", 256, 512, 3, 28),
            ("1×1 conv (C=256→512)",         256, 512, 1, 56),
            ("7×7 large spatial",            3, 64, 7, 224),
        ]

        print(f"  {'Configuration':<35} | {'Memory (MB)':>12} | {'Time (ms)':>10}")
        print(f"  {'─'*62}")

        for desc, c_in, c_out, k, hw in configs:
            torch.cuda.reset_peak_memory_stats()
            conv_test = nn.Conv2d(c_in, c_out, k, padding=k//2, bias=False
                                   ).cuda().half().eval()
            x_test    = torch.randn(8, c_in, hw, hw, device='cuda', dtype=torch.half)
            with torch.no_grad():
                _ = conv_test(x_test)   # trigger workspace allocation
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(100): conv_test(x_test)
                torch.cuda.synchronize()
                t_ms = (time.perf_counter() - t0) / 100 * 1000
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"  {desc:<35} | {peak_mb:>12.1f} | {t_ms:>10.4f}")
        print()
        print("  Note: peak memory includes workspace + weights + activations.")
        print("  cuDNN workspace is returned to PyTorch's pool after each call.")
        print("  FFT-based algorithms need more workspace than GEMM-based ones.")

else:
    print("  (PyTorch required — reference code shown above)")
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