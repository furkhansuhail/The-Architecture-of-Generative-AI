"""
TVM — The Apache Tensor Virtual Machine
=========================================

TVM is an open-source, end-to-end machine learning compiler framework.
Where XLA is Google's compiler tightly coupled to JAX and TPU hardware,
TVM is hardware-agnostic by design — a community project that compiles
ML models from any framework to any hardware target, including CPUs, GPUs,
mobile processors, FPGAs, and custom accelerators, through a principled
stack of intermediate representations and automated optimisation.

The central insight of TVM: the performance of an ML operator depends not
just on the algorithm but on HOW the algorithm is scheduled onto hardware.
A matrix multiply is always the same algorithm (sum of products), but whether
to tile the loops, in what order, at what granularity, with what vector width,
and with what memory hierarchy reuse pattern determines whether you get 10%
or 90% of peak hardware throughput. TVM separates algorithm from schedule,
making the optimisation space explicit and searchable.

TVM pioneered the idea of AUTOMATED SEARCH for optimal schedules — a machine
learning compiler that uses machine learning (a cost model, search algorithms)
to find the best way to compile ML computations. This recursive insight —
learning to compile learning — is TVM's defining contribution.

The TVM compiler stack has three distinct layers:
    1. RELAY (or TVMScript/Relax): the high-level ML IR. Expresses a full
       model as a typed, functional computation graph with known tensor shapes
       and dtypes. Framework models (PyTorch, TF, ONNX) are imported here.

    2. TIR (Tensor IR): the low-level loop-and-memory IR. Expresses a single
       operator as a nest of loops over array indices. This is where schedule
       transformations (tiling, vectorisation, parallelism) are applied.

    3. CODE GENERATION: TIR is lowered to a hardware backend — LLVM (CPU),
       CUDA/PTX (NVIDIA GPU), ROCm (AMD GPU), Metal (Apple), Vulkan (cross-
       vendor GPU), WebGPU (browser), or custom accelerator runtimes.

In the connected compiler stack:
    LLVM       (module 01) ← TVM's CPU backend emits LLVM IR via LLVM codegen
    MLIR       (module 02) ← TVM's Relax dialect uses MLIR; TOSA↔TVM bridges exist
    CIRCT      (module 03) ← VTA (TVM's hardware accelerator) uses CIRCT-like flows
    Enzyme     (module 04) ← TVM uses AD in Relay for training; Enzyme for custom ops
    XLA        (module 05) ← TVM competes and cooperates; StableHLO bridges the two
    OpenXLA    (module 06) ← TVM consumes StableHLO via tvm.relax.from_stablehlo
    StableHLO  (module 07) ← TVM's primary cross-framework import format
    TVM        (this)      ← the compiler: Relay→TIR, auto-scheduling, multi-backend

"""

import textwrap
import re

TOPIC_NAME   = "TVM — The Apache Tensor Virtual Machine"
DISPLAY_NAME = "07 · TVM"
ICON         = "🔧"
SUBTITLE     = "Relay IR, TIR Schedules, AutoTVM, MetaSchedule, and Multi-Backend Codegen"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY TVM EXISTS: THE HARDWARE DIVERSITY PROBLEM

### The Hardware Proliferation Crisis

    By 2017, the ML hardware landscape had exploded. A model trained in
    TensorFlow needed to run on:

        NVIDIA GPU:    CUDA + cuBLAS + cuDNN + TensorRT
        AMD GPU:       ROCm + MIOpen (a maintained fork of the CUDA stack)
        Intel CPU:     MKL-DNN (now oneDNN) with AVX-512 intrinsics
        ARM CPU:       NEON/SVE via ARM Compute Library
        ARM Mali GPU:  OpenCL with device-specific tuning
        Apple A-series:Core ML / Metal Performance Shaders
        Qualcomm DSP:  SNPE (Snapdragon Neural Processing Engine)
        Google TPU:    XLA (proprietary, incompatible with all the above)
        FPGA:          HLS C++ → synthesis → bitstream (weeks of work)
        Custom ASIC:   bespoke toolchain for every chip

    The MAINTENANCE PROBLEM:
        Each backend required a team of experts writing hand-optimised kernels.
        N operators × M hardware targets = N×M kernel implementations.
        For ResNet: ~50 operator variants × ~10 hardware targets = 500 kernels.
        For a transformer: ~100 operator variants × 10 targets = 1000 kernels.

        When new hardware (e.g., a new NVIDIA GPU architecture) launched,
        every single kernel had to be re-profiled and potentially rewritten.

        When a new operator was added to a framework (e.g., Flash Attention),
        every hardware backend needed a hand-optimised implementation.

    THE DEEPER PROBLEM — not just portability, but optimality:
        Even for NVIDIA GPUs with cuBLAS available, cuBLAS cannot be optimal
        for every use case. A matmul of shape [1, 1, 2048, 2048] has very
        different optimal tile sizes than [32, 32, 512, 512].
        cuBLAS uses heuristics for tile selection. For a specific model's
        specific shapes, a tuned kernel can be 20–50% faster than cuBLAS.
        TVM's autotuning finds these per-shape optimal configurations
        automatically, something no hand-written library can do at scale.

### TVM's Solution: A Two-Level Separation

    TVM's founding insight (Chen et al., OSDI 2018) was to separate the
    ML computation stack into two orthogonal concerns:

    CONCERN 1 — COMPUTATION DESCRIPTION (the WHAT):
        What mathematical operation is being computed?
        Matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]
        This is the ALGORITHM, independent of hardware.

    CONCERN 2 — SCHEDULE (the HOW):
        How should loops be organised to exploit this hardware?
        - Tile loops for cache efficiency?
        - Which dimension to vectorise?
        - How to parallelise across CPU cores or GPU threads?
        - How to use shared memory in GPU?
        - What is the optimal tile size for this specific hardware?
        This is the SCHEDULE, hardware-specific and auto-tunable.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  COMPUTATION (algorithm):  fixed, hardware-independent              │
    │    C = matmul(A, B):                                                │
    │    for i in range(M):                                               │
    │        for j in range(N):                                           │
    │            C[i,j] = sum(A[i,k]*B[k,j] for k in range(K))           │
    │                                                                     │
    │  SCHEDULE (policy): searched, hardware-specific                     │
    │    tile(i, 32), tile(j, 32), tile(k, 8)                             │
    │    reorder(i.outer, j.outer, k, i.inner, j.inner)                  │
    │    vectorize(j.inner)                                               │
    │    parallel(i.outer)                                                │
    │    cache_read(A, "shared"), cache_read(B, "shared")  # GPU only     │
    └─────────────────────────────────────────────────────────────────────┘

    With this separation:
        Writing a new operator = define computation once.
        Running on new hardware = find schedule via auto-search.
        No hand-tuned kernels required per operator per hardware.

### The TVM Ecosystem Today

    TVM has grown far beyond its original scope. The ecosystem includes:

    Relay:          functional ML IR for whole-model optimisation
    TIR:            loop-level IR for single-operator optimisation
    AutoTVM:        template-based schedule search (TVM 0.7–0.9)
    Ansor:          annotation-free schedule search (TVM 0.8+)
    MetaSchedule:   production-ready unified autotuning (TVM 0.9+)
    Relax:          next-generation dynamic-shape IR (replacing Relay)
    TVM Unity:      unified stack (TIR + Relax + MetaSchedule together)
    VTA:            Versatile Tensor Accelerator (FPGA/ASIC template)
    microTVM:       ML on bare-metal microcontrollers (no OS, <1MB RAM)
    WebGPU/WASM:    in-browser ML via WebGPU compute shaders + WASM


##### PART 2 — RELAY: THE HIGH-LEVEL ML IR

### What Relay Is

    Relay is TVM's high-level intermediate representation for ML models.
    It is a STRONGLY TYPED, PURELY FUNCTIONAL language with:
        - First-class tensor types with known shapes and dtypes
        - A polymorphic type system with type inference
        - Let-bindings for naming intermediate computations
        - Higher-order functions (closures over parameters)
        - Algebraic data types (used for tuples, lists of tensors)
        - Automatic differentiation (used during training)

    Relay occupies the same level as:
        XLA's HLO (module 05)
        StableHLO (module 07)
        ONNX (graph nodes with typed edges)

    The key difference from HLO and StableHLO: Relay supports DYNAMIC SHAPES
    natively and has a richer type system (including polymorphism and
    algebraic data types). This makes it more expressive for research models
    but harder to compile to optimal code for fixed-shape deployment.

### Relay's Type System

    Every Relay expression has a TYPE — a static description of its value.

    TENSOR TYPES:
        Tensor[float32, (32, 784)]    ; 2D float tensor, shape known statically
        Tensor[float16, (?, 512)]     ; dynamic first dimension (batch size)
        Tensor[int8, ()]              ; scalar integer
        Tensor[bool, (8,)]            ; boolean vector for masking

    FUNCTION TYPES:
        (Tensor[float32, (32, 784)], Tensor[float32, (784, 128)]) → Tensor[float32, (32, 128)]
        ; a function from two tensors to one tensor (a linear layer)

    TUPLE TYPES:
        (Tensor[float32, (N,)], Tensor[float32, (N,)])
        ; used for multi-output functions (e.g., topk returns values AND indices)

    TYPE VARIABLES (polymorphism):
        fn<T: Type, shape: Shape>(x: Tensor[T, shape]) → Tensor[T, shape]
        ; a typed identity function that works for any tensor type and shape

### Relay IR Syntax

    A Relay program is an expression tree in functional style.
    Let-bindings give names to sub-expressions:

        def @mlp(%x: Tensor[(32, 784), float32],
                 %W1: Tensor[(784, 128), float32],
                 %b1: Tensor[(128), float32],
                 %W2: Tensor[(128, 10), float32],
                 %b2: Tensor[(10), float32]) -> Tensor[(32, 10), float32] {
            let %h1 = nn.relu(nn.dense(%x, %W1) + %b1);
            let %logits = nn.dense(%h1, %W2) + %b2;
            %logits
        }

    Key Relay operators (nn.* namespace):
        nn.dense(%x, %w):           x @ w^T  (linear layer, weight is transposed)
        nn.conv2d(%x, %w):          2D convolution with configurable params
        nn.relu(%x):                max(x, 0)
        nn.softmax(%x, axis=-1):    softmax normalisation
        nn.batch_norm(%x, γ, β, mean, var): batch normalisation
        nn.dropout(%x, rate):       dropout (training only)
        nn.layer_norm(%x, γ, β):    layer normalisation
        nn.multi_head_attention:    complete MHA module

    Core tensor operators:
        add(%a, %b), subtract, multiply, divide  ; elementwise arithmetic
        matmul(%a, %b)                           ; general matmul
        reshape(%x, newshape)                    ; reshape
        transpose(%x, axes)                      ; permute dimensions
        concatenate([%a, %b, ...], axis)         ; concat along axis
        split(%x, n, axis)                       ; split into n parts
        squeeze(%x, axis)                        ; remove size-1 dims
        expand_dims(%x, axis)                    ; add size-1 dimension
        take(%x, indices, axis)                  ; gather/embedding lookup
        where(%cond, %a, %b)                     ; elementwise select

### Relay Graph-Level Optimisation Passes

    Relay's pass infrastructure runs graph-level optimisations before
    lowering to TIR. These operate on the WHOLE MODEL, not per-operator.

    PASS 1: FoldConstant
        Evaluates sub-expressions with all-constant inputs at compile time.
            add(constant([1,2,3]), constant([4,5,6])) → constant([5,7,9])
        Important for: embedding tables, positional encodings, weight norms.

    PASS 2: EliminateCommonSubexpr (CSE)
        If the same sub-expression appears twice in the graph, compute it once.
            x^2 + 2*x^2 → let %sq = x^2; %sq + 2*%sq
        Common in attention: Q, K, V projections all read the same input.

    PASS 3: FuseOps (Operator Fusion)
        The most important Relay pass. Identifies groups of operators that
        can be compiled into a single TIR kernel (and thus one GPU launch).

        FUSION RULES in Relay:
            Injective ops:   elementwise ops with one output per input element.
                             (add, relu, multiply, exp, log, abs, ...)
                             Can always be fused together.
            Reduction ops:   reduce a dimension (sum, mean, max, softmax).
                             Can fuse with injective producers (input fusion).
            Complex ops:     conv2d, dense (matmul). These anchor fusion groups.
                             Surrounding injective ops fuse INTO their epilogue.

        FUSION EXAMPLE — transformer FFN block:
            FFN(x) = relu(x @ W1 + b1) @ W2 + b2

            Without fusion:  5 kernel launches (matmul, add, relu, matmul, add)
            After FuseOps:   fusion group 1: matmul + add + relu (1 kernel)
                             fusion group 2: matmul + add         (1 kernel)
            Total: 2 kernel launches (down from 5)

        FUSION CATEGORIES:
            kElemWise:   output shape = input shape (add, relu)
            kBroadcast:  output shape ≥ input shape (broadcast + op)
            kInjective:  injective mapping input → output (reshape, transpose)
            kCommReduce: commutative reduction (sum, max, product)
            kOutEWiseFusable: complex ops like conv2d/dense that can fuse with
                              surrounding elementwise ops
            kTuple:      tuple construction/extraction

    PASS 4: AlterOpLayout
        Converts operation layouts to hardware-preferred formats.
        conv2d NCHW → NCHW4c (channel vectorised, 4 channels per packet)
        dense → packed_dense (weight matrix with hardware-aligned layout)
        Used for: ARM NEON (prefers NHWC), TensorRT (prefers NCHW16c).

    PASS 5: FastMath
        Replaces accurate math with faster approximations where applicable:
            exp(x) → fast_exp(x)   (piecewise linear approx, ~2× faster)
            log(x) → fast_log(x)   (uses HW instruction directly)
        Trades accuracy (~1 ULP) for speed. Optional, disabled by default.

    PASS 6: SimplifyInference
        Removes training-only operations for inference:
            batch_norm(x, γ, β, μ, σ) → γ*(x-μ)/σ + β (constant fold μ, σ)
            dropout(x, rate) → x (no-op at inference)

### The Relay Computation Graph vs TIR Loop Nest

    There is a crucial distinction between two levels of TVM:

    RELAY LEVEL (model graph):
        Nodes are operators (nn.dense, nn.relu, nn.conv2d).
        Edges are tensors (the data flowing between operators).
        Optimisation target: which operators to fuse, what layout to use.
        This is analogous to XLA's HLO graph.

    TIR LEVEL (operator implementation):
        A single node from the Relay graph (e.g., nn.dense) is expanded
        into TIR: a specific nest of loops over tensor indices.
        Optimisation target: how to tile, vectorise, and parallelise those loops.
        There is NO equivalent to this in XLA — XLA calls cuBLAS/cuDNN instead.

    This two-level structure is what gives TVM its hardware-agnostic power:
        Relay handles model-level fusion and layout.
        TIR handles operator-level loop optimisation.
        For new hardware: implement TIR schedule primitives once,
                          Relay automatically handles the whole model.


##### PART 3 — TIR: THE TENSOR INTERMEDIATE REPRESENTATION

### What TIR Is

    TIR (Tensor IR) is TVM's low-level IR for expressing a single operator
    as a concrete loop nest over array indices with explicit memory operations.

    TIR is more concrete than Relay (which describes what to compute) and
    more abstract than LLVM IR (which describes machine instructions).
    TIR is the level at which SCHEDULE TRANSFORMATIONS are applied.

    A TIR program consists of:
        Buffer declarations:  how arrays are laid out in memory
        Loop statements:      for loops over ranges (with attributes like
                              "parallel", "vectorized", "unroll")
        Block statements:     the atomic unit of computation (compute + reads + writes)
        Memory scopes:        which memory hierarchy level (global, shared, local)
        Data types:           float32, int8, etc. per element

### TIR Program Syntax

    Consider matrix multiply C[i,j] += A[i,k] * B[k,j]:

    TIR representation (the COMPUTATION — hardware-independent):

        @T.prim_func
        def matmul(A: T.Buffer[(M, K), "float32"],
                   B: T.Buffer[(K, N), "float32"],
                   C: T.Buffer[(M, N), "float32"]):
            for i in T.serial(M):          ; loop over rows of A/C
                for j in T.serial(N):      ; loop over cols of B/C
                    for k in T.serial(K):  ; reduction dimension
                        with T.block("C"):
                            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                            ; S = spatial (parallel), R = reduction (serial)
                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    The T.axis.remap("SSR") annotation declares:
        vi, vj are SPATIAL dimensions (independent, can be parallelised)
        vk is a REDUCTION dimension (must be accumulated serially)

    This information is what makes schedule transformations SAFE:
        Only spatial loops can be parallelised.
        Only loops with the right access patterns can be vectorised.
        Tiling across a reduction dimension requires initialising accumulators.

### Schedule Primitives: The Complete Vocabulary

    Schedule primitives are transformations applied to TIR loop nests.
    They are SEMANTICS-PRESERVING by construction — the compiler guarantees
    that the transformed program computes the same result.

    ── TILE AND SPLIT ──────────────────────────────────────────────────────

    split(loop, factor):
        Split one loop into two: an outer (tile) and an inner loop.
        Before: for i in range(64)
        After:  for i_outer in range(4):     ; 64 / 16 = 4 outer tiles
                  for i_inner in range(16):  ; tile size = 16
                    body(i_outer * 16 + i_inner)

        Purpose: creates cache-friendly access patterns.
        For matmul: split M and N loops by tile size → working set fits in L1/L2.

    tile(i_loop, j_loop, x_factor, y_factor):
        Split two loops simultaneously (2D tiling).
        Before: for i in range(M): for j in range(N)
        After:  for i_outer in range(M//x):
                  for j_outer in range(N//y):
                    for i_inner in range(x):
                      for j_inner in range(y):

        Purpose: 2D tiling for matmul gives O(x*y) reuse with O(x+y) loads.

    ── REORDER ─────────────────────────────────────────────────────────────

    reorder(loop_a, loop_b, loop_c, ...):
        Change the nesting order of loops.
        Before: for i: for j: for k: body
        After:  reorder(i, k, j): for i: for k: for j: body

        Rules: ONLY legal if no data dependency prevents it.
               TVM's dependency analysis verifies correctness.

        Purpose: change memory access pattern for better cache locality.
        For matmul: reorder(i.outer, j.outer, k, i.inner, j.inner) ensures
                    the inner loops sweep across one tile at a time.

    ── PARALLEL ────────────────────────────────────────────────────────────

    parallel(loop):
        Mark a loop as parallelisable (OpenMP threads or GPU blocks).
        Before: for i in range(M): body
        After:  for i in T.parallel(M): body  → OpenMP on CPU

        Requirements: loop body has no data dependency on iteration order.
        Only safe for SPATIAL dimensions (not REDUCTION).

    ── VECTORIZE ───────────────────────────────────────────────────────────

    vectorize(loop):
        Mark the innermost loop for SIMD vectorisation.
        Before: for j_inner in range(8): C[j_inner] = A[j_inner] + B[j_inner]
        After:  vec<8xf32> C_vec = A_vec + B_vec   (one vector instruction)

        Requirements: loop range must be a constant multiple of SIMD width.
                      Loop body must be a simple elementwise operation.

        TVM maps to: AVX-512 (16 × f32) on x86, NEON (4 × f32) on ARM,
                     CUDA vector types (float4) on GPU.

    ── UNROLL ──────────────────────────────────────────────────────────────

    unroll(loop, factor):
        Unroll a small loop — replicate the body N times.
        Before: for k in range(4): C[i,j] += A[i,k] * B[k,j]
        After:  C[i,j] += A[i,0]*B[0,j]; C[i,j] += A[i,1]*B[1,j]; ...

        Purpose: eliminates loop overhead; exposes instruction-level parallelism.
        Best for: small innermost loops (K=4, K=8) in tiled matmul.

    ── CACHE_READ / CACHE_WRITE (GPU shared memory) ─────────────────────────

    cache_read(A, "shared", [consumer_block]):
        Introduce an explicit copy from global → shared memory.
        Before: global A  →  directly read in compute block
        After:  global A  →  shared A_shared  →  compute block

        Purpose: coalesced global memory read (all threads read together),
                 then fast shared memory access within a warp.
        Critical for: any GPU matmul (shared memory bandwidth ~100× faster).

    cache_write(C, "local", [compute_block]):
        Accumulate results in register files before writing to global.
        Before: C[global] += A * B   (global memory write per multiply!)
        After:  C_local[register] += A * B;  C[global] = C_local  (one store)

    ── COMPUTE_AT ──────────────────────────────────────────────────────────

    compute_at(stage, attach_point):
        Inline one operator's computation inside another operator's loop.
        Before: for i: B[i] = exp(A[i])  ; then:  for i: for j: C[i,j] = B[i] * ...
        After (compute_at B to innermost j loop):
                for i: for j: B[i] = exp(A[i]); C[i,j] = B[i] * ...

        Purpose: eliminates materialising the intermediate tensor B entirely.
                 This IS operator fusion, expressed at the TIR level.

    ── BIND (GPU THREAD MAPPING) ────────────────────────────────────────────

    bind(loop, "blockIdx.x"):    ; map outer loop to CUDA grid blocks
    bind(loop, "threadIdx.x"):   ; map inner loop to CUDA threads within a block

        Before: for i_outer in range(M//32):  for i_inner in range(32): body
        After:  ↓ bind(i_outer, "blockIdx.x")  ↓ bind(i_inner, "threadIdx.x")
                Grid: M//32 blocks, each block: 32 threads

        Purpose: maps the abstract loop structure to CUDA's execution model.
        Block/thread counts must satisfy CUDA hardware limits:
            Max threads per block: 1024
            Max blocks per grid:   ~2^31

### The TIR to LLVM/CUDA Lowering Pipeline

    Once a TIR program has been scheduled, TVM lowers it to the target:

    TIR (with schedule annotations)
        ↓  InjectPrefetch        ; insert prefetch instructions where beneficial
        ↓  StorageFlatten        ; flatten N-D index expressions to 1-D
        ↓  LoopPartition         ; split loops with if-conditions to avoid branches
        ↓  UnrollLoop            ; execute the unroll primitives
        ↓  VectorizeLoop         ; lower vectorize to SIMD operations
        ↓  InjectVirtualThread   ; inject virtual threading for hyper-threading
        ↓  InjectDoubleBuffer    ; insert ping-pong buffers for memory latency hiding
        ↓  CodegenLLVM           ; emit LLVM IR (for CPU / NVPTX / AMDGCN)
           OR CodegenCUDA        ; emit CUDA C++ source (for nvcc)
           OR CodegenSpirV       ; emit SPIR-V (for Vulkan)
           OR CodegenMetal       ; emit Metal Shading Language (for Apple)

    For the LLVM path:
        TIR → LLVM IR → LLVM opt → LLVM llc → native binary / PTX
    For the CUDA path:
        TIR → CUDA C++ → nvcc → cubin
    These two paths meet at the TVM runtime which manages both CPU and GPU execution.


##### PART 4 — AUTOTVM: TEMPLATE-BASED SCHEDULE SEARCH

### The Schedule Space Problem

    For a matrix multiply of shape [M, K, N], the number of possible
    schedules is enormous even for a small set of primitives:

        Tile sizes:    M ∈ {1, 2, 4, 8, 16, 32, 64} × K × N = 343 combos
        Reorder:       3! = 6 permutations of (i, j, k) at each level
        Vectorize:     2 choices (on or off) × multiple candidate loops
        Unroll:        unroll factor ∈ {1, 2, 4, 8, 16}
        Cache read:    on/off for A and B
        Thread count:  for GPU: blockDim ∈ {32, 64, 128, 256, 512, 1024}

    Combined: easily >100,000 valid schedules for one matmul shape.
    For a full model with 50 operators: 100,000^50 configurations.

    BRUTE FORCE is impossible. TVM needs SMART SEARCH.

### AutoTVM: Templates + Machine Learning Search

    AutoTVM (Automated TVM, TVM 0.5+) solves this with:
        1. A SCHEDULE TEMPLATE: a programmer-written function that defines
           the search space of valid schedules via tunable "knobs."
        2. A COST MODEL: a learned function that predicts performance
           from the knob configuration (without running the kernel).
        3. A SEARCH ALGORITHM: explores the space guided by the cost model.

    SCHEDULE TEMPLATE ANATOMY:
        @autotvm.template("conv2d.cuda")
        def conv2d_cuda_template(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
            # Define the schedule computation (TIR)
            n, h, w, co, kh, kw, ci = ...

            # KNOBS: parameterise the schedule space
            cfg.define_split("tile_f", co, num_outputs=4)
            # "tile_f" splits the output-channel (co) loop into 4 levels:
            # [block_size_z, vthread_z, thread_z, tile_z]
            # TVM will try: (1,1,1,32), (1,1,2,16), (1,2,4,8), ...

            cfg.define_split("tile_y", oh, num_outputs=4)
            cfg.define_split("tile_x", ow, num_outputs=4)
            cfg.define_split("tile_rc", ci, num_outputs=2)
            cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

            # Use cfg values to build the schedule
            f_factors   = cfg["tile_f"].size
            y_factors   = cfg["tile_y"].size
            x_factors   = cfg["tile_x"].size

            bf, vf, tf, fi = s[output].split(f_axis, factor=f_factors[3])
            s[output].reorder(bf, vf, tf, ...)
            s[output].bind(bf, te.thread_axis("blockIdx.z"))
            s[output].bind(tf, te.thread_axis("threadIdx.z"))
            s[output].vectorize(fi)

    COST MODEL — how TVM learns performance without full benchmarking:
        Feature extraction:
            From a schedule configuration, extract features:
            - Loop bounds at each level (tile sizes, thread counts)
            - Memory access patterns (strided? coalesced? bank-conflict-free?)
            - Arithmetic intensity (FLOPs per byte loaded)
            - Hardware resource utilisation estimates (occupancy)

        Model architecture:
            A gradient-boosted tree (XGBoost) trained on:
                X: feature vectors of schedule configurations
                y: measured execution times (from profiling samples)
            After seeing ~100 measured samples, the model can rank
            the remaining 100,000+ candidates accurately.

        Prediction:
            cost_model.predict(config_A) → estimated time in milliseconds
            These predictions sort candidates by expected performance.

    TUNING WORKFLOW:
        1. Define a task: (operator, hardware target, input shape)
        2. Extract search space from template: e.g., 200,000 configs
        3. Random initial sampling: measure 100 random configs on hardware
        4. Fit cost model: XGBoost trained on (features, measured_times)
        5. Propose candidates: top-k predicted-fast configs not yet measured
        6. Measure candidates: actually run them on hardware
        7. Update cost model with new measurements
        8. Repeat steps 5-7 until time budget or convergence
        9. Store best found config in a tuning log file

    SIMULATED ANNEALING as an alternative to cost-model-guided search:
        When hardware is very fast (cheap measurement), random sampling
        followed by simulated annealing is simpler and competitive.
        sa_candidate = mutate(current_best, temperature)
        if perf(sa_candidate) > perf(current_best): accept
        else:  accept with probability exp(-Δperf / temperature)

### AutoTVM Tuning in Practice

    COMMAND LINE:
        python3 -m tvm.driver.tvmc tune \
            --target "cuda" \
            --output resnet50_cuda_log.json \
            resnet50.onnx

    PYTHON API:
        tasks = autotvm.task.extract_from_program(relay_mod, target, params)
        for task in tasks:
            tuner = autotvm.tuner.XGBTuner(task, loss_type='rank')
            tuner.tune(
                n_trial=200,
                measure_option=autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(timeout=10)
                ),
                callbacks=[autotvm.callback.log_to_file('tuning.log')]
            )

    TYPICAL RESULTS on GPU:
        Before tuning:   ResNet-50 inference = ~6.5 ms/image
        After AutoTVM:   ResNet-50 inference = ~4.2 ms/image  (35% faster)
        vs TensorRT:     ResNet-50 inference = ~4.5 ms/image  (TVM matches)

    TUNING LOG FORMAT (JSON lines):
        Each line = one measured config:
        {"input": ["cuda", "conv2d", ...], "config": {"tile_f": [1,1,4,16], ...},
         "result": [[3.41e-3, 3.38e-3, 3.45e-3], 0, ...]}
        These logs are portable: tune once, deploy anywhere (same target).


##### PART 5 — ANSOR AND METASCHEDULE: ANNOTATION-FREE AUTOTUNING

### The Problem with AutoTVM Templates

    AutoTVM's key weakness: someone must WRITE THE TEMPLATE.
    A schedule template encodes a programmer's prior knowledge about how
    a particular operator should be structured on a particular hardware type.

    Problems with templates:
        COVERAGE: if no template exists for an operator+hardware combination,
                  AutoTVM cannot tune it. New operators → new templates needed.
        BIAS:     the template limits the search space to the programmer's
                  assumptions. A truly optimal schedule may be OUTSIDE the
                  template's space (wrong tiling order, unexpected fusion).
        EFFORT:   writing a good template requires deep hardware expertise.
                  For a new accelerator architecture, this is weeks of work.

### Ansor: Sketch-Based Search (TVM 0.8, 2020)

    Ansor (Zheng et al., OSDI 2020) eliminates templates entirely.
    It generates the schedule search space AUTOMATICALLY from the operator
    definition (the computation expressed in TE — Tensor Expression).

    ANSOR'S THREE-STAGE PIPELINE:

    STAGE 1 — SKETCH GENERATION:
        A sketch is a partial schedule with structure but not values.
        Given a TE computation, Ansor applies rules to generate sketches:
            "Any loop can be split."
            "Any spatial loop can be parallelised."
            "Any innermost loop can be vectorised."
            "Any op can be fused with its elementwise producer."
            "GPU: any block/thread dimension can be bound."
        Rules are applied recursively and exhaustively → many sketches.
        Example for matmul: ~40 sketches covering all structural variants.

    STAGE 2 — ANNOTATION (random fill):
        Each sketch has free parameters (tile sizes, thread counts).
        Fill them in randomly → create a concrete schedule configuration.
        Sample thousands of random concrete schedules from each sketch.
        This sampling covers the full space without bias.

    STAGE 3 — EVOLUTIONARY SEARCH with learned cost model:
        Initial population: sampled from random annotation phase.
        Cost model: a neural network (MLP on program features).
        Evolution loop:
            a. Predict performance for current population.
            b. Select top candidates by predicted performance.
            c. Apply mutation operators:
               - Randomly change one tile size
               - Swap two loops in reorder
               - Toggle vectorise on/off
               - Change parallel factor
            d. Add mutated candidates to population.
            e. After N mutations, measure top-k on hardware.
            f. Update cost model with measured data.
            g. Repeat.

    ANSOR'S ADVANTAGES:
        No templates required: works on ANY operator expressible in TE.
        Larger search space: explores configurations templates would miss.
        Better performance: Ansor finds schedules 1.5–2.3× faster than AutoTVM
                            on diverse hardware (demonstrated on T4 GPU, ARM CPU).
        Faster search: hierarchical search (model-level task selection)
                       allocates tuning budget proportionally to each task's
                       contribution to total latency.

### MetaSchedule: Production-Ready Unified Autotuning (TVM 0.9+)

    MetaSchedule (Shao et al., NeurIPS 2022) unifies AutoTVM and Ansor
    into a single framework while addressing their practical limitations.

    KEY IMPROVEMENTS OVER ANSOR:

    1. DATABASE-BACKED SEARCH:
        All measured results are stored in a SQLite database.
        Previous tuning runs can be reused for new (similar) models.
        Transfer learning across shapes: [M=256, K=512] measurements inform
        search for [M=512, K=512] (similar structure, different scale).

    2. REPRODUCIBLE INFRASTRUCTURE:
        Measurement is separated from the search algorithm.
        Builders and runners run in isolated processes (crash-safe).
        Same code works locally, on a remote machine, or in a cluster.

    3. TVMScript INTEGRATION:
        Schedules can be expressed as Python functions decorated with @T.prim_func.
        This makes schedules readable, debuggable, and git-diff-able.

    4. CASCADING TUNING:
        MetaSchedule can tune fused operators (not just single ops).
        A fused relu+bias+conv2d is tuned as a unit → better than tuning each op.

    5. COST MODEL:
        SegmentSum-based transformer cost model (better than XGBoost for irregular shapes).
        Ensemble of models trained on different hardware platforms.

    METASCHEDULE API (TVM 0.9+):
        import tvm.meta_schedule as ms

        database = ms.database.JSONDatabase("./meta_schedule_workdir")

        with ms.Profiler() as profiler:
            tuned_mod, params = ms.relay_integration.tune_relay(
                mod=relay_mod,
                params=relay_params,
                target=tvm.target.Target("cuda"),
                config=ms.TuneConfig(
                    strategy="evolutionary",
                    num_trials_per_iter=64,
                    max_trials_per_task=200,
                    max_trials_global=2000,
                ),
                work_dir="./meta_schedule_workdir",
            )

        compiled = ms.relay_integration.compile_relay(
            database=database,
            mod=relay_mod,
            target=tvm.target.Target("cuda"),
            params=relay_params,
        )


##### PART 6 — TVM'S MULTI-BACKEND CODEGEN: FROM TIR TO EVERY HARDWARE

### TVM's Hardware Target Taxonomy

    TVM's target system is hierarchical: a TARGET string describes the
    hardware precisely enough for the code generator to make correct decisions.

    Target string syntax:
        "cuda -arch=sm_86"          ; NVIDIA Ampere (RTX 3090, A5000)
        "cuda -arch=sm_80"          ; NVIDIA Ampere (A100)
        "llvm -mcpu=cascadelake"    ; Intel Cascade Lake (AVX-512 VNNI)
        "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"  ; ARM64 with NEON
        "metal"                     ; Apple GPU (Metal)
        "vulkan -device=mali"       ; ARM Mali GPU (Vulkan)
        "opencl -device=adreno"     ; Qualcomm Adreno GPU
        "webgpu"                    ; Browser WebGPU
        "hexagon -v68"              ; Qualcomm Hexagon DSP (v68)
        "c"                         ; portable C (any CPU with a C compiler)

### The CUDA/NVPTX Backend

    TVM's GPU code generation for NVIDIA targets produces either:
        a. CUDA C++ source (compiled by nvcc at runtime)
        b. PTX (NVPTX LLVM backend, JIT-compiled by CUDA driver)

    The generated CUDA code for a tiled matmul looks like:

        __global__ void tir_matmul(float* A, float* B, float* C, int M, int N, int K) {
            // Block and thread indices from bind() primitives
            int bx = blockIdx.x;  int tx = threadIdx.x;
            int by = blockIdx.y;  int ty = threadIdx.y;

            // Shared memory for tiles (from cache_read)
            __shared__ float A_shared[TILE_M][TILE_K];
            __shared__ float B_shared[TILE_K][TILE_N];

            // Local accumulator (from cache_write)
            float C_local[TILE_M_per_thread][TILE_N_per_thread] = {0};

            // Tile loop (from split + reorder)
            for (int k_outer = 0; k_outer < K / TILE_K; k_outer++) {
                // Cooperative loading from global to shared (coalesced)
                A_shared[ty][tx] = A[(by*TILE_M + ty)*K + k_outer*TILE_K + tx];
                B_shared[ty][tx] = B[(k_outer*TILE_K + ty)*N + bx*TILE_N + tx];
                __syncthreads();

                // Warp-level computation (vectorised)
                for (int k_inner = 0; k_inner < TILE_K; k_inner++) {
                    for (int mi = 0; mi < TILE_M_per_thread; mi++) {
                        for (int ni = 0; ni < TILE_N_per_thread; ni++) {
                            C_local[mi][ni] += A_shared[ty*TILE_M_per_thread+mi][k_inner]
                                            * B_shared[k_inner][tx*TILE_N_per_thread+ni];
                        }
                    }
                }
                __syncthreads();
            }
            // Write back (from cache_write)
            ...
        }

    TVM's GPU schedule can match cuBLAS performance for specific shapes
    by finding the right TILE_M, TILE_N, TILE_K, warps_per_block, and
    vectorization factor via MetaSchedule autotuning.

### The LLVM CPU Backend

    For CPU targets, TVM's LLVM backend emits LLVM IR and uses LLVM for
    target-specific optimisation and code generation.

    Schedule → LLVM IR emission:
        parallel(i.outer) → OpenMP parallel for (via LLVM's ompx dialect)
        vectorize(j.inner) → LLVM vector types (<8 x float>)
        unroll(k, 4) → loop body replicated 4 times in LLVM IR
        cache_read/write → alloca for local arrays (register allocation)

    The LLVM backend benefits from:
        Target-specific intrinsics: _mm512_fmadd_ps for AVX-512 FMA
        Auto-vectorisation: LLVM's loop vectoriser for non-annotated loops
        Register allocation: LLVM's graph-colouring allocator
        Instruction scheduling: LLVM's VLIW/OOO scheduling passes

    CPU-specific schedule considerations:
        L1 cache: 32–64 KB → tile inner loops to fit two input tiles + output
        L2 cache: 256 KB–1 MB → tile middle loops for L2 reuse
        L3 cache: 8–32 MB → tile outer loops for L3 reuse
        Vector width: 128-bit (NEON/SSE), 256-bit (AVX2), 512-bit (AVX-512)
        FMA units: vectorize AND use T.hardware_intrinsic("llvm.fma")

### microTVM: Bare-Metal Deployment

    microTVM extends TVM to embedded microcontrollers with NO OS,
    severely limited RAM (<1 MB), and no floating-point hardware.

    Key challenges microTVM solves:
        NO DYNAMIC MEMORY: all buffers pre-allocated at compile time.
        NO OS: no pthreads, no GPU drivers, no Python runtime.
        INT8 ONLY: most MCUs lack f32 support; int8 fixed-point is used.
        TINY CODE SIZE: model + runtime must fit in flash (<256 KB often).

    microTVM pipeline:
        Relay model (quantised to int8)
            ↓ TVM compilation with target="c"
        Generated C code (no stdlib dependencies)
            ↓ cross-compilation with arm-none-eabi-gcc
        Firmware binary (.hex / .elf)
            ↓ flash to MCU
        Arduino / Zephyr / bare-metal execution

    microTVM target examples:
        "c -mcpu=cortex-m4"       ; ARM Cortex-M4 (STM32, nRF52)
        "c -mcpu=cortex-m33"      ; ARM Cortex-M33 (nRF9160)
        "hexagon -v66"            ; Qualcomm Hexagon DSP

### VTA: Versatile Tensor Accelerator

    VTA is TVM's hardware template for custom accelerators, designed to
    be implemented on FPGAs and custom ASICs as a research platform.

    VTA hardware components:
        FETCH MODULE:   fetches instructions from DRAM into command queues
        LOAD MODULE:    DMA engine; loads input/weight tensors to SRAM
        COMPUTE MODULE: systolic array of GEMM units + ALU for elementwise
        STORE MODULE:   DMA engine; writes output tensors back to DRAM

    VTA ISA (instruction set):
        LOAD  {src: DRAM, dst: inp_mem}     ; load input tile
        LOAD  {src: DRAM, dst: wgt_mem}     ; load weight tile
        GEMM  {inp: inp_mem, wgt: wgt_mem, out: acc_mem}  ; matrix multiply
        ALU   {op: ReLU, src: acc_mem, dst: acc_mem}      ; elementwise
        STORE {src: acc_mem, dst: DRAM}     ; store output tile

    TVM schedule for VTA:
        Define the computation in TE (GEMM + elementwise).
        Apply VTA-specific schedule primitives:
            tvm.te.schedule.Stage.pragma(..., "ila_skip_env_tasks")
            vta.schedule.AlterLayout(...)
        TVM lowers to VTA ISA → FPGA bitstream via Chisel/Verilog.

    Research use: Google has used VTA as a research platform for
    studying accelerator micro-architecture and compilation.


##### PART 7 — RELAX AND TVM UNITY: THE NEXT GENERATION

### Why Relay Needs a Successor

    Relay was designed before dynamic-shape models became dominant.
    Its type system could express dynamic shapes (?), but the compiler
    infrastructure assumed static shapes in many places, leading to:
        PERFORMANCE COST: padding to static shapes wastes compute.
        EXPRESSIVITY LIMIT: truly dynamic control flow (e.g., early exit,
                            variable-length text decoding) cannot be expressed.
        INTEGRATION GAP:  Relay is NOT an MLIR dialect; integrating MLIR
                          infrastructure requires custom translation layers.

### Relax: Dynamic Shape, MLIR-Native IR

    Relax (Relay Next, TVM 0.11+) is Relay's successor with:
        MLIR-based: Relax is expressed as an MLIR dialect (relax.*).
        Native dynamic shapes: shapes are first-class values, not type annotations.
        Symbolic shapes: shape variables propagate through the graph.
        Runtime-dependent control flow: Python-like if/while in the IR.
        StableHLO interop: tvm.relax.from_stablehlo imports StableHLO programs.

    Relax syntax example (Python-flavoured IRModule):

        @tvm.script.ir_module
        class MyModel:
            @R.function
            def main(x: R.Tensor(("n", 4), "float32"),
                     W: R.Tensor((4, 8), "float32")) -> R.Tensor(("n", 8), "float32"):
                # 'n' is a symbolic shape variable (batch size unknown)
                with R.dataflow():
                    y = R.matmul(x, W)           ; n × 4 @ 4 × 8 → n × 8
                    z = R.nn.relu(y)             ; still n × 8
                    R.output(z)
                return z

    Dynamic shape propagation:
        When TVM compiles this, 'n' remains symbolic.
        At runtime, the actual value of n is known → JIT-compiled specialisation.
        For n=32: compile and cache the n=32 version.
        For n=64: compile and cache the n=64 version.
        This is how LLM inference works (variable batch/sequence sizes).

### TVM Unity: The Integrated Stack

    TVM Unity integrates Relax + TIR + MetaSchedule into a cohesive system:

        Python model definition
            ↓ IRModule (Relax + TIR together)
        Relax passes (graph-level optimisation)
            ↓ still IRModule
        MetaSchedule tuning (per-operator schedule search)
            ↓ tuned IRModule
        TIR lowering (to LLVM IR / CUDA C++ / etc.)
            ↓ compiled artifacts per backend
        TVM runtime execution

    Key principle: Relax and TIR live in the SAME IRModule.
        A Relax operator (R.matmul) can directly reference a TIR function.
        Tuning a TIR function automatically improves the Relax graph.
        No translation layer between the two levels.

    TVM Unity with TVMScript:

        @tvm.script.ir_module
        class FusedModule:
            @T.prim_func                         ; TIR level
            def fused_dense_relu(x: ..., w: ..., y: ...):
                for i, j in T.grid(M, N):
                    for k in T.serial(K):
                        y[i,j] += x[i,k] * w[k,j]
                    y[i,j] = T.max(y[i,j], 0.0)  ; relu fused in!

            @R.function                          ; Relax level
            def main(x: R.Tensor(...)):
                gv = R.call_tir(FusedModule.fused_dense_relu, ...)  ; calls TIR fn
                return gv


##### PART 8 — TVM IN THE CONNECTED COMPILER STACK

### How TVM Connects to LLVM (Module 01)

    TVM uses LLVM as its CPU and GPU codegen backend:
        CPU path: TIR → CodegenLLVM → LLVM IR → LLVM opt → LLVM llc → .so
        GPU path: TIR → CodegenNVPTX → LLVM NVPTX → PTX → cubin (via LLVM's NVPTX backend)

    TVM uses these LLVM components directly:
        llvm::IRBuilder to emit LLVM IR from TIR loop bodies.
        llvm::VectorType for vectorized loops.
        LLVM's intrinsics API for platform-specific instructions.
        LLVM's TargetMachine for x86/ARM/RISC-V/NVPTX instruction selection.
        LLVM's JIT (LLJIT / ORC JIT) for in-process compilation and execution.

    TVM does NOT use LLVM's optimisation passes for scheduling — TVM does
    its own loop transformations (tiling, vectorizing, parallelizing) BEFORE
    emitting LLVM IR. LLVM is used as a backend code emitter, not an optimiser.
    This is the opposite of standard compiler practice but is deliberate:
    TVM's ML-aware schedule search finds better optimisations than LLVM's
    general-purpose passes for tensor workloads.

### How TVM Connects to MLIR (Module 02)

    TVM's relationship with MLIR is evolving:

    HISTORICAL: TVM and MLIR were independent projects.
        TVM had Relay (not MLIR). MLIR had linalg (not TVM).
        They could not share passes, types, or infrastructure.

    CURRENT (TVM 0.11+ with Relax):
        Relax is an MLIR dialect (tvm.relax.*).
        TVM's IRModule is an MLIR module.
        TVM uses MLIR's pass manager for Relax passes.
        MLIR's TOSA dialect can be lowered into TVM (tosa-to-relay bridge).
        StableHLO can be imported into Relax (from_stablehlo pass).

    CONVERGENCE POINTS:
        Both MLIR's linalg and TVM's TIR express loop nests with index maps.
        Both use the same progressive lowering philosophy.
        Research projects (TVM-MLIR) explore deeper integration.

### How TVM Connects to XLA and OpenXLA (Modules 05-06)

    COMPETITION: TVM and XLA both compile ML models to GPU/CPU.
    They target the same workloads with different philosophies:
        XLA: hand-tuned libraries (cuBLAS, cuDNN) + fusion of surrounding ops.
        TVM: auto-generated tiled kernels + holistic schedule search.
    For GEMM-dominated workloads: XLA usually wins (cuBLAS expertise).
    For conv+elementwise fusion, or novel operators: TVM can win.

    COOPERATION (via StableHLO):
        Train with JAX+XLA → export to StableHLO.
        Import StableHLO into TVM's Relax.
        Deploy with TVM on edge hardware XLA doesn't support.
        This is the canonical cross-framework deployment workflow.

    DIVERGENCE:
        XLA has first-class TPU support. TVM does not target TPUs.
        TVM has microcontroller support (microTVM). XLA does not.
        TVM's auto-scheduling can find better schedules for custom shapes.

### How TVM Connects to StableHLO (Module 07)

    StableHLO is TVM's primary import format for trained ML models:
        tvm.relax.from_stablehlo("model.mlir")  ; imports into Relax

    Lowering path inside TVM:
        StableHLO ops → Relax ops (via pattern matching rules):
            stablehlo.dot_general → R.matmul
            stablehlo.convolution → R.nn.conv2d
            stablehlo.reduce      → R.sum / R.max
        Then: standard Relax compilation pipeline applies.

    This means any framework that can produce StableHLO (JAX, PyTorch via
    torch-mlir, TF) can be deployed on any TVM target.

### TVM vs XLA vs IREE: Choosing the Right Tool

    ┌──────────────────────┬───────────────┬──────────────┬──────────────┐
    │ Property             │  TVM          │  XLA/OpenXLA │  IREE        │
    ├──────────────────────┼───────────────┼──────────────┼──────────────┤
    │ GEMM performance     │ Competitive   │ Best (cuBLAS)│ Competitive  │
    │ Edge/mobile          │ Excellent     │ Limited      │ Excellent    │
    │ microcontroller      │ YES (microTVM)│ No           │ No           │
    │ Custom operators     │ TE + schedule │ Complicated  │ Linalg ext.  │
    │ Auto-scheduling      │ MetaSchedule  │ None (hand)  │ None         │
    │ Novel hardware       │ Best          │ Hard (PJRT)  │ Good (HAL)   │
    │ Dynamic shapes       │ Relax         │ Limited      │ Limited      │
    │ Training             │ Limited       │ Full (JAX)   │ Limited      │
    │ FPGA support         │ VTA           │ None         │ None         │
    │ WebAssembly          │ YES           │ No           │ YES          │
    │ Hexagon DSP          │ YES           │ No           │ Limited      │
    └──────────────────────┴───────────────┴──────────────┴──────────────┘

    The canonical answer: TVM excels at edge/mobile deployment of
    trained models, especially where hardware is diverse and unusual.
    XLA excels at GPU/TPU training performance. Use both via StableHLO.

### The Full TVM Compilation Stack

    ┌─────────────────────────────────────────────────────────────────────┐
    │  FRAMEWORKS (input)                                                 │
    │  PyTorch → torch.export → StableHLO                                 │
    │  JAX     → jax.export   → StableHLO                                 │
    │  ONNX    → tvm.relay.from_onnx()                                    │
    │  TF      → tvm.relay.from_tensorflow()                              │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   │ model import
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  RELAY / RELAX (high-level IR)                                      │
    │  Graph-level passes: FuseOps, FoldConstant, AlterOpLayout, CSE      │
    └──────────────────────────────┬──────────────────────────────────────┘
                                   │ lowering
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  TIR (tensor IR, per-operator loop nests)                           │
    │  Schedule search: MetaSchedule / Ansor / AutoTVM                    │
    │  Schedule primitives: split, reorder, vectorize, parallel, bind...  │
    └──────────┬──────────────┬──────────────┬──────────────┬─────────────┘
               │              │              │              │
               ▼              ▼              ▼              ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐
    │  LLVM (CPU) │  │  CUDA/NVPTX  │  │  Metal   │  │  C (microTVM) │
    │  x86/ARM    │  │  NVIDIA GPU  │  │  Apple   │  │  MCU/FPGA     │
    └─────────────┘  └──────────────┘  └──────────┘  └───────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · TVM Relay IR — Building, Inspecting, and Optimising Models": {
        "description": (
            "Build and inspect Relay IR programs from scratch in Python. "
            "Import a model from ONNX/PyTorch and examine the Relay representation. "
            "Run Relay's graph-level optimisation passes: FuseOps, FoldConstant, CSE. "
            "Show the effect of fusion by counting kernel launches before and after. "
            "Trace a full MLP: parameter initialisation → Relay graph → compiled module. "
            "Demonstrate relay.build: how TVM turns a Relay graph into an executable."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TVM RELAY IR — BUILDING, INSPECTING, AND OPTIMISING MODELS")
print("=" * 65)
print()

try:
    import tvm
    import tvm.relay as relay
    from tvm import nd
    print(f"  TVM version: {tvm.__version__}")
    HAS_TVM = True
except ImportError:
    HAS_TVM = False
    print("  TVM not installed: pip install apache-tvm")
    print("  Or build from source: https://tvm.apache.org/docs/install")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Relay IR text format guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Relay IR textual format: every element explained")
print("━" * 65)
print()

RELAY_FORMAT = """
  RELAY PROGRAM STRUCTURE — ANNOTATED
  ════════════════════════════════════════════════════════════════

  #[version = "0.0.5"]              ; version header (for serialisation)

  def @main(%x: Tensor[(32, 784), float32],
  │         │             │         │
  │         │             └─shape   └─dtype
  │         └─argument name (SSA style)
  └─function definition keyword

             %W1: Tensor[(784, 128), float32],
             %b1: Tensor[(128), float32]) -> Tensor[(32, 10), float32] {
             ;                              ↑ return type annotation

    /* Relay uses let-bindings to name intermediate values */
    let %h_pre = nn.dense(%x, %W1, units=128);
    ;           │         │    │    └── attribute: output units
    ;           │         │    └── second argument
    ;           │         └── first argument (input)
    ;           └── operator: nn.dense performs %x @ %W1.T

    /* Operators are in typed namespaces: nn.*, math.*, image.* */
    let %h_bias = add(%h_pre, %b1);     ; broadcast add (bias)
    let %h_act  = nn.relu(%h_bias);     ; elementwise relu

    /* The final expression (no let) is the return value */
    %h_act                               ; implicit return
  }

  KEY RELAY OPERATORS (with type signatures):
  ────────────────────────────────────────────────────────────────
  nn.dense(%data, %weight, units=None)
    %data:   Tensor[(n, k), dtype]
    %weight: Tensor[(m, k), dtype]  ← NOTE: weight is k-last (transposed)
    result:  Tensor[(n, m), dtype]  ← output is n×m, NOT n×k

  nn.conv2d(%data, %weight, strides=(1,1), padding=(0,0), ...)
    NHWC default: data[N,H,W,C_in], weight[C_out,C_in,kH,kW]
    result: Tensor[(N, H_out, W_out, C_out), dtype]

  nn.relu(%data)   → max(%data, 0)   (elementwise, same shape)
  nn.softmax(%x, axis=-1)            (normalise along last axis)
  nn.batch_norm(%x, %gamma, %beta, %moving_mean, %moving_var)
    result: Tensor[same_shape, dtype]  (normalised + scaled + shifted)

  reshape(%data, newshape=(n, -1))   → like numpy reshape, -1 infers
  transpose(%data, axes=(0, 2, 1))   → permute dimensions
  take(%data, %indices, axis=0)      → gather (embedding lookup)
  where(%cond, %x, %y)               → elementwise select
"""
print(RELAY_FORMAT)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Build Relay IR programmatically
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Building Relay IR programmatically")
print("━" * 65)
print()

if HAS_TVM:
    # ── Build a 2-layer MLP in Relay ─────────────────────────────────────
    batch_size = 32
    n_in, n_h, n_out = 784, 128, 10

    # Input variable
    x    = relay.var("x",    shape=(batch_size, n_in),  dtype="float32")
    W1   = relay.var("W1",   shape=(n_h,  n_in),        dtype="float32")
    b1   = relay.var("b1",   shape=(n_h,),               dtype="float32")
    W2   = relay.var("W2",   shape=(n_out, n_h),         dtype="float32")
    b2   = relay.var("b2",   shape=(n_out,),              dtype="float32")

    # Layer 1: dense + bias + relu
    h_pre  = relay.nn.dense(x, W1, units=n_h)
    h_bias = relay.nn.bias_add(h_pre, b1)
    h_act  = relay.nn.relu(h_bias)

    # Layer 2: dense + bias
    logits = relay.nn.dense(h_act, W2, units=n_out)
    out    = relay.nn.bias_add(logits, b2)

    # Wrap in a function
    fn  = relay.Function([x, W1, b1, W2, b2], out)
    mod = tvm.IRModule({"main": fn})

    print("  Built 2-layer MLP in Relay:")
    print()
    # Pretty-print the Relay IR
    relay_text = mod.astext(show_meta_data=False)
    for line in relay_text.split("\\n")[:30]:
        print(f"    {line}")
    if relay_text.count("\\n") > 30:
        print(f"    ... ({relay_text.count(chr(10))-30} more lines)")
    print()

    # Show the type annotation (Relay's type inference)
    mod_typed = relay.transform.InferType()(mod)
    fn_type   = mod_typed["main"].checked_type
    print(f"  Inferred function type:")
    print(f"    Inputs: {[str(t) for t in fn_type.arg_types]}")
    print(f"    Output: {fn_type.ret_type}")
    print()

else:
    RELAY_BUILD_REF = """
  RELAY PYTHON API REFERENCE:
  ─────────────────────────────────────────────────────────────────
  import tvm
  import tvm.relay as relay

  # ── Define input/weight variables ──────────────────────────────
  x  = relay.var("x",  shape=(32, 784), dtype="float32")
  W1 = relay.var("W1", shape=(128, 784), dtype="float32")  ; weight is rows_out × cols_in
  b1 = relay.var("b1", shape=(128,), dtype="float32")

  # ── Build the computation graph ────────────────────────────────
  h  = relay.nn.dense(x, W1, units=128)    ; linear transform
  h  = relay.nn.bias_add(h, b1)            ; + bias
  h  = relay.nn.relu(h)                    ; activation

  # ── Create IRModule ────────────────────────────────────────────
  fn  = relay.Function([x, W1, b1], h)
  mod = tvm.IRModule({"main": fn})
  print(mod.astext())                       ; print Relay IR text

  # ── Import from ONNX ───────────────────────────────────────────
  import onnx
  onnx_model = onnx.load("resnet50.onnx")
  shape_dict = {"input": (1, 3, 224, 224)}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  # ── Import from PyTorch ────────────────────────────────────────
  import torch
  scripted = torch.jit.script(my_model)
  input_shapes = [("input", (1, 3, 224, 224))]
  mod, params = relay.frontend.from_pytorch(scripted, input_shapes)

  # ── Import from StableHLO ──────────────────────────────────────
  mod = tvm.relax.from_stablehlo(open("model.mlir").read())
"""
    print(RELAY_BUILD_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Relay optimisation passes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Relay optimisation passes")
print("━" * 65)
print()

if HAS_TVM:
    # Count ops before and after FuseOps
    def count_relay_ops(mod):
        """Count unique operator calls in a Relay module."""
        class OpCounter(relay.ExprVisitor):
            def __init__(self):
                super().__init__()
                self.ops = []
            def visit_call(self, call):
                if isinstance(call.op, tvm.ir.Op):
                    self.ops.append(call.op.name)
                super().visit_call(call)
        counter = OpCounter()
        counter.visit(mod["main"].body)
        return counter.ops

    ops_before = count_relay_ops(mod)
    print(f"  MLP ops BEFORE fusion: {len(ops_before)}")
    for op in ops_before:
        print(f"    {op}")
    print()

    # Apply passes
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2),
        relay.transform.InferType(),
    ])
    mod_opt = seq(mod)

    ops_after = count_relay_ops(mod_opt)
    print(f"  MLP ops AFTER fusion (FuseOps level 2): {len(ops_after)}")
    for op in ops_after:
        print(f"    {op}")
    print()
    print(f"  Fusion reduced op count: {len(ops_before)} → {len(ops_after)}")
    print(f"  (FuseOps groups bias_add + relu together with dense)")
    print()

else:
    PASSES_REF = """
  RELAY OPTIMISATION PASSES REFERENCE:
  ─────────────────────────────────────────────────────────────────
  # Apply a sequence of passes:
  seq = tvm.transform.Sequential([
      relay.transform.InferType(),           ; fill in all type annotations
      relay.transform.FoldConstant(),        ; evaluate constant sub-expressions
      relay.transform.EliminateCommonSubexpr(), ; CSE: compute once, use twice
      relay.transform.FuseOps(fuse_opt_level=2), ; fuse elementwise chains
      relay.transform.AlterOpLayout(),       ; convert layouts for hardware
      relay.transform.SimplifyInference(),   ; remove dropout, fold batch_norm
      relay.transform.FastMath(),            ; approximate exp/log (optional)
  ])
  mod_optimised = seq(mod)

  # Inspect the optimised IR:
  print(mod_optimised.astext())

  FUSION LEVELS:
    fuse_opt_level=0  ; no fusion
    fuse_opt_level=1  ; fuse injective ops only
    fuse_opt_level=2  ; fuse injective + broadcast  [DEFAULT]
    fuse_opt_level=3  ; aggressive: fuse reductions too
"""
    print(PASSES_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: relay.build — compiling to an executable
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — relay.build: Relay → compiled TVM module")
print("━" * 65)
print()

if HAS_TVM:
    rng    = np.random.default_rng(42)
    params = {
        "W1": rng.normal(0, 0.01, (n_h,  n_in)).astype(np.float32),
        "b1": np.zeros(n_h,  dtype=np.float32),
        "W2": rng.normal(0, 0.01, (n_out, n_h)).astype(np.float32),
        "b2": np.zeros(n_out, dtype=np.float32),
    }

    try:
        target  = tvm.target.Target("llvm")
        lib     = relay.build(mod_opt, target=target, params=params)

        # Create a runtime module
        dev     = tvm.cpu(0)
        module  = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        # Run inference
        x_np    = rng.normal(size=(batch_size, n_in)).astype(np.float32)
        module.set_input("x", x_np)
        module.run()
        output  = module.get_output(0).numpy()

        print(f"  relay.build → TVM compiled module (target=llvm)")
        print(f"  Input:  shape={x_np.shape}, dtype=float32")
        print(f"  Output: shape={output.shape}, dtype=float32")
        print(f"  Output[0,:5] = {output[0,:5]}")
        print()

        # Benchmark
        timing = module.benchmark(dev, number=100, repeat=3)
        print(f"  Benchmark (no autotuning, CPU):")
        print(f"    Mean:   {timing.mean*1000:.3f} ms")
        print(f"    Std:    {timing.std*1000:.3f} ms")
        print(f"    Median: {timing.median*1000:.3f} ms")

    except Exception as e:
        print(f"  Build/run note: {e}")
        print()
        print("  relay.build API summary:")
        BUILD_REF = """
  # Build Relay module for a specific target
  target = tvm.target.Target("llvm")   ; CPU
  # target = tvm.target.Target("cuda") ; GPU (requires CUDA toolkit)

  # Build: Relay graph → TIR schedules → LLVM/CUDA → compiled .so
  lib = relay.build(mod, target=target, params=params)

  # Deploy using TVM's graph executor
  dev    = tvm.cpu(0)          ; or tvm.cuda(0) for GPU
  module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

  # Set inputs, run, get outputs
  module.set_input("x", input_array)
  module.run()
  output = module.get_output(0).numpy()

  # Save the compiled module for later deployment
  lib.export_library("model.so")
  # Load on any machine with the TVM runtime:
  loaded_lib = tvm.runtime.load_module("model.so")
"""
        print(BUILD_REF)
    print()

else:
    COMPILE_REF = """
  RELAY.BUILD WORKFLOW:
  ─────────────────────────────────────────────────────────────────
  target = tvm.target.Target("llvm -mcpu=cascadelake")  ; with AVX-512

  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target, params=params)

  # lib is a tvm.relay.backend.Executor object containing:
  #   - compiled graph JSON (execution plan)
  #   - compiled kernel library (.so for CPU, .cubin for GPU)
  #   - parameter binary (serialised weight tensors)

  # Graph executor (most common, static graph)
  dev    = tvm.cpu(0)
  module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
  module.set_input("input_name", np_array)
  module.run()
  result = module.get_output(0).numpy()

  # VM executor (handles dynamic shapes)
  vm_exec = relay.vm.compile(mod, target=target, params=params)
  vm      = tvm.runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
  result  = vm.run(**{"x": np_array})
"""
    print(COMPILE_REF)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · TIR and Schedule Primitives — Manual Operator Optimisation": {
        "description": (
            "Deep dive into TVM's Tensor IR and schedule primitives. "
            "Implement a matrix multiply in TE (Tensor Expression) from scratch. "
            "Apply every schedule primitive: split, reorder, parallel, vectorize, unroll. "
            "Show cache_read / cache_write for GPU shared memory optimisation. "
            "Benchmark the effect of each primitive on CPU performance. "
            "Show the generated LLVM IR and CUDA C before and after scheduling."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TIR AND SCHEDULE PRIMITIVES — MANUAL OPERATOR OPTIMISATION")
print("=" * 65)
print()

try:
    import tvm
    from tvm import te, tir
    import tvm.script
    print(f"  TVM {tvm.__version__}")
    HAS_TVM = True
except ImportError:
    HAS_TVM = False
    print("  TVM not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Schedule primitives explained with numpy analogies
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Schedule primitives: numpy analogies")
print("━" * 65)
print()

PRIMITIVES_GUIDE = """
  UNDERSTANDING SCHEDULE PRIMITIVES WITH CONCRETE EXAMPLES
  ════════════════════════════════════════════════════════════════

  Baseline computation: C[i,j] = sum_k A[i,k] * B[k,j]
  M=64, K=64, N=64 (small to show the structure clearly)

  ── NAIVE (no schedule) ─────────────────────────────────────────
  for i in range(M):
    for j in range(N):
      C[i,j] = 0
      for k in range(K):
        C[i,j] += A[i,k] * B[k,j]

  Access pattern: A is accessed row-by-row (good), B is accessed
  column-by-column (BAD for row-major: cache miss every time).

  ── AFTER REORDER: i,j,k → i,k,j ─────────────────────────────
  for i in range(M):
    for k in range(K):              ; swap k and j
      for j in range(N):
        C[i,j] += A[i,k] * B[k,j]

  Now B[k,j] is accessed row-by-row (j is innermost) → cache-friendly.
  Performance impact: 2-5× speedup on CPU just from reorder.

  ── AFTER SPLIT (tile): i by 16 ─────────────────────────────────
  for i_outer in range(M // 16):   ; 4 outer tiles
    for j in range(N):
      for i_inner in range(16):    ; 16 inner
        for k in range(K):
          C[i_outer*16+i_inner, j] += A[i_outer*16+i_inner, k] * B[k, j]

  The 16×64 tile of C fits in L1 cache. Each 16×64 tile of A is
  loaded once per tile of B → better data reuse.

  ── AFTER VECTORIZE (j_inner loop of size 8) ─────────────────────
  for j_outer in range(N // 8):
    [vectorised j_inner=0..7]:   ; 8-wide SIMD instruction
      for k in range(K):
        C[i, j_outer*8 + 0..7] += A[i,k] * B[k, j_outer*8 + 0..7]

  One vector fmadd instruction replaces 8 scalar fmadd instructions.
  On AVX2: 8-wide f32 = 256-bit vector. On AVX-512: 16-wide f32.

  ── AFTER PARALLEL (i_outer loop) ───────────────────────────────
  parallel for i_outer in range(M // 16):  ; 4 OpenMP threads
    for j in range(N):
      for i_inner in range(16):
        ...

  4 CPU cores work on 4 different row-tile groups simultaneously.
  Speedup: ~4× (limited by memory bandwidth for large matrices).

  ── GPU SCHEDULE: split + bind ──────────────────────────────────
  # Split i and j for GPU blocks and threads
  i_outer, i_inner = split(i, factor=32)   ; 32 threads per block in i
  j_outer, j_inner = split(j, factor=32)

  bind(i_outer, "blockIdx.y")    ; each block handles one i_outer
  bind(j_outer, "blockIdx.x")    ; each block handles one j_outer
  bind(i_inner, "threadIdx.y")   ; each thread handles one i_inner
  bind(j_inner, "threadIdx.x")   ; each thread handles one j_inner

  Grid: (N//32, M//32) blocks.  Block: (32, 32) = 1024 threads.
  Each thread computes ONE element of C.
"""
print(PRIMITIVES_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: TE computation + progressive scheduling in TVM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — TE computation + progressive scheduling")
print("━" * 65)
print()

if HAS_TVM:
    M, K, N = 512, 512, 512

    def make_matmul_te():
        """Define matmul as a TE computation (algorithm, no schedule)."""
        A = te.placeholder((M, K), name="A", dtype="float32")
        B = te.placeholder((K, N), name="B", dtype="float32")
        k = te.reduce_axis((0, K), name="k")
        C = te.compute((M, N),
                        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                        name="C")
        return A, B, C

    rng  = np.random.default_rng(42)
    A_np = rng.random((M, K)).astype("float32")
    B_np = rng.random((K, N)).astype("float32")
    C_ref = A_np @ B_np

    def build_and_benchmark(sched_name, build_fn, n_runs=50):
        """Build a schedule and measure its runtime."""
        A, B, C = make_matmul_te()
        try:
            s = te.create_schedule(C.op)
            build_fn(s, A, B, C)
            func = tvm.build(s, [A, B, C], target="llvm", name="matmul")
            dev  = tvm.cpu(0)
            a_nd = nd.array(A_np, device=dev)
            b_nd = nd.array(B_np, device=dev)
            c_nd = nd.array(np.zeros((M,N), dtype="float32"), device=dev)
            # Verify correctness
            func(a_nd, b_nd, c_nd)
            err = float(np.max(np.abs(c_nd.numpy() - C_ref)))
            if err > 1e-2:
                return None, None, f"Correctness fail: err={err:.2e}"
            # Benchmark
            timer = func.time_evaluator(func.entry_name, dev,
                                        number=n_runs, repeat=3)
            t_ms  = timer(a_nd, b_nd, c_nd).mean * 1000
            return func, t_ms, "ok"
        except Exception as e:
            return None, None, str(e)[:60]

    results = []

    # ── Schedule 0: naive ────────────────────────────────────────────────
    def schedule_naive(s, A, B, C):
        pass   # no transformations

    fn0, t0, st0 = build_and_benchmark("naive", schedule_naive)
    results.append(("Naive (no schedule)",  t0, st0))

    # ── Schedule 1: reorder ─────────────────────────────────────────────
    def schedule_reorder(s, A, B, C):
        i, j = s[C].op.axis
        (k,)  = s[C].op.reduce_axis
        s[C].reorder(i, k, j)        ; i,k,j order for better B access

    fn1, t1, st1 = build_and_benchmark("reorder", schedule_reorder)
    results.append(("Reorder i,k,j",       t1, st1))

    # ── Schedule 2: reorder + vectorize ─────────────────────────────────
    def schedule_vec(s, A, B, C):
        i, j = s[C].op.axis
        (k,)  = s[C].op.reduce_axis
        j_outer, j_inner = s[C].split(j, factor=8)
        s[C].reorder(i, k, j_outer, j_inner)
        s[C].vectorize(j_inner)

    fn2, t2, st2 = build_and_benchmark("vec", schedule_vec)
    results.append(("Reorder + vectorize(8)",   t2, st2))

    # ── Schedule 3: tile + reorder + vectorize ───────────────────────────
    def schedule_tile(s, A, B, C):
        i, j = s[C].op.axis
        (k,)  = s[C].op.reduce_axis
        i_outer, i_inner = s[C].split(i, factor=32)
        j_outer, j_inner = s[C].split(j, factor=32)
        s[C].reorder(i_outer, j_outer, k, i_inner, j_inner)
        s[C].vectorize(j_inner)

    fn3, t3, st3 = build_and_benchmark("tile", schedule_tile)
    results.append(("Tile(32,32) + vectorize",   t3, st3))

    # ── Schedule 4: tile + parallel ─────────────────────────────────────
    def schedule_parallel(s, A, B, C):
        i, j = s[C].op.axis
        (k,)  = s[C].op.reduce_axis
        i_outer, i_inner = s[C].split(i, factor=32)
        j_outer, j_inner = s[C].split(j, factor=32)
        s[C].reorder(i_outer, j_outer, k, i_inner, j_inner)
        s[C].vectorize(j_inner)
        s[C].parallel(i_outer)        ; parallelise across CPU cores

    fn4, t4, st4 = build_and_benchmark("parallel", schedule_parallel)
    results.append(("Tile + parallel + vectorize", t4, st4))

    # Print results
    print(f"  Matrix multiply: {M}×{K} @ {K}×{N}  (f32, CPU)")
    print()
    print(f"  {'Schedule':35s}  {'Time (ms)':>10s}  {'Speedup':>8s}  {'Status'}")
    print("  " + "-" * 68)

    base_t = results[0][1] if results[0][1] else 1.0
    for name, t, status in results:
        if t is not None:
            speedup = base_t / t
            print(f"  {name:35s}  {t:>10.3f}  {speedup:>7.2f}×  {status}")
        else:
            print(f"  {name:35s}  {'N/A':>10s}  {'N/A':>8s}  {status}")
    print()

    if fn4 is not None:
        # Show generated IR for the best schedule
        print("  Generated LLVM IR snippet (tiled+parallel+vectorized):")
        ir_text = tvm.lower(te.create_schedule(make_matmul_te()[2].op), list(make_matmul_te()), simple_mode=True)
        # Show a short excerpt
        ir_str = str(ir_text)
        for line in ir_str.split("\\n")[:20]:
            print(f"    {line}")
        print("    ...")
        print()

else:
    TE_SCHEDULE_REF = """
  TE + SCHEDULE API REFERENCE:
  ─────────────────────────────────────────────────────────────────
  from tvm import te

  # ── Define the computation (algorithm) ─────────────────────────
  M, K, N = 512, 512, 512
  A = te.placeholder((M, K), name="A", dtype="float32")
  B = te.placeholder((K, N), name="B", dtype="float32")
  k = te.reduce_axis((0, K), name="k")
  C = te.compute((M, N),
                  lambda i, j: te.sum(A[i,k] * B[k,j], axis=k),
                  name="C")

  # ── Create and apply a schedule ────────────────────────────────
  s = te.create_schedule(C.op)
  i, j = s[C].op.axis
  (red_k,) = s[C].op.reduce_axis

  # split j into 8-wide SIMD lanes
  j_outer, j_inner = s[C].split(j, factor=8)
  # put k before j for cache-friendly B access
  s[C].reorder(i, k, j_outer, j_inner)
  # vectorize the innermost 8 elements
  s[C].vectorize(j_inner)
  # parallelise outer i loop (OpenMP)
  s[C].parallel(i)

  # ── Compile and run ────────────────────────────────────────────
  func = tvm.build(s, [A, B, C], target="llvm", name="matmul")
  dev  = tvm.cpu(0)
  a_nd = tvm.nd.array(A_np, dev)
  b_nd = tvm.nd.array(B_np, dev)
  c_nd = tvm.nd.array(np.zeros((M,N),"float32"), dev)
  func(a_nd, b_nd, c_nd)
  result = c_nd.numpy()

  # Print TIR (lowered IR before codegen):
  print(tvm.lower(s, [A,B,C], simple_mode=True))

  # Expected speedups (512×512×512 f32 CPU):
  # Naive:                 ~80 ms   (1×)
  # Reorder(i,k,j):        ~35 ms   (2.3×)
  # +vectorize(j,8):       ~12 ms   (6.7×)
  # +tile(32)+parallel(4): ~4  ms   (20×)
"""
    print(TE_SCHEDULE_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: TVMScript — writing TIR directly
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TVMScript: readable TIR in Python syntax")
print("━" * 65)
print()

TVMSCRIPT_GUIDE = """
  TVMSCRIPT — TIR AS DECORATED PYTHON
  ════════════════════════════════════════════════════════════════
  TVMScript lets you write TIR programs directly as Python,
  decorated with @T.prim_func. This is the basis of TVM Unity.

  import tvm.script
  from tvm.script import tir as T

  @tvm.script.ir_module
  class MatmulModule:

      @T.prim_func
      def matmul(A: T.Buffer[(512, 512), "float32"],
                 B: T.Buffer[(512, 512), "float32"],
                 C: T.Buffer[(512, 512), "float32"]):
          T.func_attr({"global_symbol": "matmul", "tir.noalias": True})

          for i, j, k in T.grid(512, 512, 512):
              with T.block("C"):
                  vi = T.axis.spatial(512, i)   ; spatial: can be parallel
                  vj = T.axis.spatial(512, j)
                  vk = T.axis.reduce(512, k)    ; reduce: must accumulate

                  T.reads(A[vi, vk], B[vk, vj])
                  T.writes(C[vi, vj])
                  with T.init():
                      C[vi, vj] = T.float32(0)  ; initialise accumulator
                  C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

  KEY TVMSCRIPT CONSTRUCTS:
  ─────────────────────────────────────────────────────────────────
  T.Buffer[(M,N), dtype]:    buffer declaration with shape and dtype
  T.grid(a, b, c):           loop nest over a×b×c iterations
  T.serial(n):               a simple for loop (default)
  T.parallel(n):             OpenMP-parallel loop
  T.vectorized(n):           SIMD-vectorised loop (must be constant)
  T.unroll(n):               unrolled loop
  T.block("name"):           names a computation block (unit of autoscheduling)
  T.axis.spatial(range, var): declares spatial (independent) axis
  T.axis.reduce(range, var): declares reduction axis
  T.reads(...), T.writes(...): explicit memory access declarations
  T.init():                  initialisation code (run once per output element)

  WHY TVMSCRIPT MATTERS:
  ─────────────────────────────────────────────────────────────────
  1. READABILITY: the TIR program looks like Python, not bytecode.
  2. DEBUGGABILITY: you can set breakpoints and print in the IR.
  3. GIT DIFFS: readable schedules can be reviewed and versioned.
  4. METASCHEDULE: the auto-scheduler generates TVMScript schedules.
  5. PORTABILITY: same TVMScript compiles to CPU, GPU, or custom accelerator.
"""
print(TVMSCRIPT_GUIDE)

if HAS_TVM:
    try:
        from tvm.script import tir as T

        @tvm.script.ir_module
        class SimpleModule:
            @T.prim_func
            def relu(A: T.Buffer[(128,), "float32"],
                     B: T.Buffer[(128,), "float32"]):
                T.func_attr({"global_symbol": "relu", "tir.noalias": True})
                for i in T.serial(128):
                    with T.block("relu"):
                        vi = T.axis.spatial(128, i)
                        T.reads(A[vi])
                        T.writes(B[vi])
                        B[vi] = T.max(A[vi], T.float32(0))

        print("  TVMScript relu module built successfully.")
        print(f"  IRModule type: {type(SimpleModule)}")

        # Build it
        func = tvm.build(SimpleModule, target="llvm")
        dev  = tvm.cpu(0)
        a    = nd.array(np.array([-1.0, 0.5, -0.3, 2.0] * 32, dtype="float32"), dev)
        b    = nd.array(np.zeros(128, dtype="float32"), dev)
        func["relu"](a, b)
        print(f"  relu([-1, 0.5, -0.3, 2.0, ...])[:8] = {b.numpy()[:8]}")
        print(f"  Expected:                              [0, 0.5, 0, 2.0, 0, 0.5, 0, 2.0]")
    except Exception as e:
        print(f"  TVMScript note: {e}")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · AutoTVM and Ansor — Automated Schedule Search": {
        "description": (
            "Complete exploration of TVM's two autotuning systems. "
            "Simulate AutoTVM's cost model: feature extraction, XGBoost training, candidate ranking. "
            "Show the tuning loop: sample → measure → fit model → propose → repeat. "
            "Implement Ansor's sketch-based search: sketch generation rules and annotation. "
            "Compare AutoTVM vs Ansor vs MetaSchedule with timing results. "
            "Show how to read and use a tuning log file for model compilation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field

print("=" * 65)
print("  AUTOTVM AND ANSOR — AUTOMATED SCHEDULE SEARCH")
print("=" * 65)
print()

try:
    import tvm
    from tvm import te, autotvm
    HAS_TVM = True
    print(f"  TVM {tvm.__version__}")
except ImportError:
    HAS_TVM = False
    print("  TVM not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: The search space problem
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The search space: why exhaustive search fails")
print("━" * 65)
print()

def count_search_space(M, K, N):
    """
    Estimate the number of valid schedules for a matmul of shape M×K×N.
    This shows why exhaustive measurement is infeasible.
    """
    # Tile sizes: must divide the dimension (power-of-2 factors only)
    def divisors_pow2(n, max_factor=64):
        return [2**i for i in range(7) if 2**i <= min(n, max_factor)]

    tile_m = len(divisors_pow2(M))          # options for M tile size
    tile_n = len(divisors_pow2(N))          # options for N tile size
    tile_k = len(divisors_pow2(K))          # options for K tile size
    reorder_options = 6                      # 3! = 6 loop orderings for (i,j,k)
    vectorize_choices = 4                    # none, j inner of 4/8/16
    unroll_k = 3                             # unroll K inner by 0/4/8
    parallel = 2                             # parallel on i_outer: yes/no

    total = tile_m * tile_n * tile_k * reorder_options * vectorize_choices * unroll_k * parallel
    return total, {
        "tile_M": tile_m, "tile_N": tile_n, "tile_K": tile_k,
        "reorder": reorder_options, "vectorize": vectorize_choices,
        "unroll_k": unroll_k, "parallel": parallel,
    }

print("  Schedule space size for matrix multiply:")
print()
print(f"  {'Shape (M×K×N)':20s}  {'Total configs':>15s}  {'At 10ms/meas':>14s}")
print("  " + "-" * 55)
for M,K,N in [(64,64,64),(256,256,256),(512,512,512),(1024,1024,1024),(4096,4096,4096)]:
    total, breakdown = count_search_space(M,K,N)
    time_hours = total * 0.01 / 3600  # 10ms per measurement
    print(f"  [{M:4d}×{K:4d}×{N:4d}]         {total:>15,d}  {time_hours:>11.1f} hrs")

print()
print("  Even the smallest shape has thousands of valid configs.")
print("  Exhaustive measurement is infeasible → need SMART SEARCH.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: AutoTVM's cost model simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — AutoTVM cost model: features → performance prediction")
print("━" * 65)
print()

COST_MODEL_THEORY = """
  AUTOTVM COST MODEL ARCHITECTURE
  ════════════════════════════════════════════════════════════════

  AutoTVM's XGBoost cost model predicts kernel latency from schedule features.

  FEATURE EXTRACTION: for each schedule config, compute ~100 features:

  1. LOOP FEATURES (per loop in the TIR nest):
     - Loop extent (tile size) at each level
     - Loop type: serial / parallel / vectorised / unrolled
     - Arithmetic intensity: FLOPs / bytes accessed
     - Reuse distance: how many cache lines between reuses of the same data

  2. MEMORY ACCESS FEATURES:
     - Access stride for each buffer (stride-1 = coalesced, large = random)
     - Number of distinct cache lines accessed
     - L1/L2/L3 working set size estimates

  3. PARALLELISM FEATURES:
     - Number of parallel threads / CUDA blocks
     - Thread occupancy estimate
     - Warp divergence estimate (GPU only)

  4. RESOURCE USAGE:
     - Register pressure estimate (number of live variables)
     - Shared memory usage (GPU only)
     - Instruction mix: FMA, load, store, branch ratios

  XGBOOST MODEL:
     Trained with: rank loss (predict relative ordering, not absolute time)
     Input:  feature vector (100-dimensional float)
     Output: latency rank score (lower = faster predicted)

     After 50-100 measured samples:
       - Pearson correlation between predicted and actual: ~0.85
       - Top-10 predicted fast configs contain the true best: ~80% of the time

  SEARCH ALGORITHM:
     Simulated Annealing guided by cost model:
     1. Random initial population (50 configs)
     2. Predict scores with cost model → rank all candidates
     3. Select top-20 by predicted score
     4. Mutate each: randomly change one knob (tile size, reorder, ...)
     5. Measure top-4 candidates on actual hardware
     6. Update cost model with new measurements
     7. Repeat 50-200 iterations

  TYPICAL CONVERGENCE:
     After 200 measurements: within 5% of optimal for GPU matmul
     After 500 measurements: within 2% of optimal
     vs brute force: need 10,000+ measurements for same quality
"""
print(COST_MODEL_THEORY)

# Simulate the AutoTVM search loop
@dataclass
class ScheduleConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    reorder: str         # "ijk", "ikj", "jik", "jki", "kij", "kji"
    vectorize: int       # 0 = off, 4/8/16 = lane width
    unroll_k: int        # 0 = off, 4 = unroll k by 4

    def features(self) -> np.ndarray:
        """Extract schedule features (simplified AutoTVM feature extraction)."""
        reorder_map = {"ijk":0,"ikj":1,"jik":2,"jki":3,"kij":4,"kji":5}
        return np.array([
            np.log2(max(self.tile_m, 1)),
            np.log2(max(self.tile_n, 1)),
            np.log2(max(self.tile_k, 1)),
            reorder_map.get(self.reorder, 0) / 5.0,
            np.log2(max(self.vectorize, 1)) / 4.0,
            np.log2(max(self.unroll_k, 1)) / 4.0,
            # arithmetic intensity proxy
            (self.tile_m * self.tile_k + self.tile_k * self.tile_n) /
            (self.tile_m * self.tile_n * self.tile_k + 1e-6),
        ], dtype=np.float32)

def simulate_hardware_time(cfg: ScheduleConfig, noise=0.05) -> float:
    """
    Simulate hardware measurement of a schedule config.
    This models the ground truth that AutoTVM's cost model learns to predict.
    """
    base = 10.0   # ms baseline (no optimisation)
    # Reorder matters most for cache
    reorder_bonus = {"ikj": 3.5, "kij": 2.5, "ijk": 1.0, "jik": 0.8,
                     "jki": 0.5, "kji": 0.3}.get(cfg.reorder, 1.0)
    # Vectorize is very beneficial
    vec_bonus = {0: 1.0, 4: 2.5, 8: 3.5, 16: 4.0}.get(cfg.vectorize, 1.0)
    # Tiling helps up to L1 size
    tile_bonus = min((cfg.tile_m * cfg.tile_n) / 512, 2.0) + 0.5
    # Unrolling helps for small K
    unroll_bonus = {0: 1.0, 4: 1.3, 8: 1.2}.get(cfg.unroll_k, 1.0)

    ideal_time = base / (reorder_bonus * vec_bonus * tile_bonus * unroll_bonus)
    # Add measurement noise
    return ideal_time * (1.0 + np.random.randn() * noise)

class AutoTVMSimulator:
    """
    Simulates AutoTVM's cost-model-guided search.
    XGBoost replaced here with a simple linear model for illustration.
    """
    def __init__(self, search_space: List[ScheduleConfig]):
        self.space      = search_space
        self.measured   = {}   # config_id → measured time
        self.model      = None
        self.rng        = np.random.default_rng(0)

    def _fit_model(self):
        """Fit a cost model on measured samples."""
        if len(self.measured) < 5:
            return
        from numpy.linalg import lstsq
        ids = list(self.measured.keys())
        X   = np.array([self.space[i].features() for i in ids])
        y   = np.array([self.measured[i] for i in ids])
        # Simple linear regression (AutoTVM uses XGBoost)
        X_aug = np.column_stack([X, np.ones(len(X))])
        coeffs, _, _, _ = lstsq(X_aug, y, rcond=None)
        self.model = coeffs

    def _predict(self, cfg_id: int) -> float:
        """Predict latency for a config using the fitted model."""
        if self.model is None:
            return self.rng.uniform(0, 10)
        features = self.space[cfg_id].features()
        return float(np.dot(self.model[:-1], features) + self.model[-1])

    def random_sample(self, n: int) -> List[int]:
        """Random initial sampling."""
        return list(self.rng.choice(len(self.space), size=n, replace=False))

    def propose_candidates(self, n: int) -> List[int]:
        """Propose next configs to measure based on cost model predictions."""
        unmeasured = [i for i in range(len(self.space)) if i not in self.measured]
        if not unmeasured:
            return []
        predictions = [(i, self._predict(i)) for i in unmeasured]
        predictions.sort(key=lambda x: x[1])   # lowest predicted time = best
        return [i for i, _ in predictions[:n]]

    def measure(self, cfg_ids: List[int]):
        for i in cfg_ids:
            if i not in self.measured:
                self.measured[i] = simulate_hardware_time(self.space[i])
        self._fit_model()

    def best_so_far(self) -> Tuple[int, float]:
        if not self.measured:
            return -1, float("inf")
        best_id = min(self.measured, key=self.measured.get)
        return best_id, self.measured[best_id]

# Generate a representative search space
configs = []
for tile_m in [8, 16, 32, 64]:
    for tile_n in [8, 16, 32, 64]:
        for tile_k in [4, 8, 16]:
            for reorder in ["ijk", "ikj", "jik", "kij"]:
                for vec in [0, 4, 8]:
                    for unroll in [0, 4]:
                        configs.append(ScheduleConfig(tile_m, tile_n, tile_k,
                                                       reorder, vec, unroll))

true_best_time = min(simulate_hardware_time(c, noise=0) for c in configs)

simulator = AutoTVMSimulator(configs)

print(f"  Search space: {len(configs):,} configs")
print(f"  True best time: {true_best_time:.3f} ms")
print()
print(f"  AutoTVM search progress:")
print(f"  {'Iter':>5}  {'Measured':>9}  {'Best found':>11}  "
      f"{'vs optimal':>11}  {'Method'}")
print("  " + "-" * 55)

# Initial random sampling
initial_ids = simulator.random_sample(20)
simulator.measure(initial_ids)
_, best_t = simulator.best_so_far()
print(f"  {'0':>5}  {'20':>9}  {best_t:>11.3f}ms  "
      f"{best_t/true_best_time:>10.2f}×  random init")

# Guided search iterations
for iteration in range(1, 9):
    candidates = simulator.propose_candidates(5)
    simulator.measure(candidates)
    _, best_t = simulator.best_so_far()
    pct = (best_t - true_best_time) / true_best_time * 100
    print(f"  {iteration:>5}  {len(simulator.measured):>9}  "
          f"{best_t:>11.3f}ms  {best_t/true_best_time:>10.2f}×  "
          f"cost-model guided (+{pct:.1f}% from optimal)")

best_id, best_t = simulator.best_so_far()
best_cfg = configs[best_id]
print()
print(f"  Best found config after {len(simulator.measured)} measurements:")
print(f"    tile_m={best_cfg.tile_m}, tile_n={best_cfg.tile_n}, "
      f"tile_k={best_cfg.tile_k}")
print(f"    reorder={best_cfg.reorder}, vectorize={best_cfg.vectorize}, "
      f"unroll_k={best_cfg.unroll_k}")
print(f"    Time: {best_t:.3f} ms vs true optimal: {true_best_time:.3f} ms")
print(f"    Coverage: {len(simulator.measured)}/{len(configs)} "
      f"({100*len(simulator.measured)/len(configs):.1f}%) of space measured")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Ansor sketch generation rules
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Ansor: sketch generation vs AutoTVM templates")
print("━" * 65)
print()

ANSOR_SKETCHES = """
  ANSOR: FROM COMPUTATION TO SKETCHES (NO TEMPLATE NEEDED)
  ════════════════════════════════════════════════════════════════

  Input: the TE computation C[i,j] = sum_k A[i,k] * B[k,j]

  Ansor's sketch generation rules (applied automatically):

  RULE 1 — "always split reduction axes":
    Sketch A: for i, j, k_outer, k_inner (split k)

  RULE 2 — "spatial axes can be multi-level split":
    Sketch A1: i → (i_b, i_t)        (block + thread)
    Sketch A2: i → (i_b, i_v, i_t)   (block + virtual thread + thread)
    Sketch A3: i → (i_b, i_v, i_t, i_inner)  (+ register tile)

  RULE 3 — "unroll can apply to any small constant loop":
    Sketch B: ... k_inner is small (≤8) → annotate as unroll candidate

  RULE 4 — "GPU: bind outer loops to blocks/threads":
    Sketch C: (i_b → blockIdx, j_b → blockIdx, i_t → threadIdx, ...)

  RULE 5 — "cache read/write for GPU shared memory":
    Sketch D: cache_read(A, "shared"), cache_read(B, "shared")
              for any computation that reads from global memory twice

  Each rule generates one or more sketch variants.
  Combined: ~40 sketches for a simple matmul.
  For a fused conv+relu: ~120 sketches.

  SKETCH vs TEMPLATE:
  ─────────────────────────────────────────────────────────────────
  AutoTVM TEMPLATE:                    Ansor SKETCH:
    Written by a hardware expert         Generated automatically
    Encodes specific assumptions         Encodes general structure
    Works well for known hardware        Works on novel hardware
    Misses non-obvious schedules         Explores wider space
    Required per operator                Required: just TE computation

  ANNOTATION (filling in sketch knobs):
    After generating sketches, Ansor randomly samples concrete schedules
    by filling in tile sizes, thread counts, etc.:

    sketch: split(i, 3-level)
    annotation sample 1: i → (4, 8, 2)   ; i_outer=4, i_mid=8, i_inner=2
    annotation sample 2: i → (2, 4, 16)
    annotation sample 3: i → (8, 16, 1)
    ... (thousands of samples per sketch)

  EVOLUTIONARY OPTIMISATION:
    Ansor's evolutionary search mutates configs:
      tile_size_mutator:   change one tile size to adjacent power of 2
      reorder_mutator:     swap two loops in the reorder order
      parallel_mutator:    toggle parallel on outer loop
      memory_mutator:      add/remove cache_read for a buffer

    Each mutation → measure (or predict) → keep if better
    Result: high-quality schedules in 200-1000 measurements
"""
print(ANSOR_SKETCHES)

if HAS_TVM:
    print("  AutoTVM API for tuning a model task:")
    AUTOTVM_API = """
  # ── Task extraction ────────────────────────────────────────────
  tasks = autotvm.task.extract_from_program(
      relay_mod["main"],
      target="cuda",
      params=relay_params,
  )
  print(f"Extracted {len(tasks)} tunable tasks from the Relay model")
  # Each task = one operator (conv2d, dense, batch_matmul, ...)
  # with specific shape, dtype, layout parameters

  # ── Tuning ─────────────────────────────────────────────────────
  log_file = "tuning_results.json"
  for task in tasks:
      tuner = autotvm.tuner.XGBTuner(task, loss_type="rank")
      tuner.tune(
          n_trial=200,
          early_stopping=100,
          measure_option=autotvm.measure_option(
              builder=autotvm.LocalBuilder(timeout=10),
              runner=autotvm.LocalRunner(repeat=3, timeout=20),
          ),
          callbacks=[
              autotvm.callback.progress_bar(200, prefix=task.name),
              autotvm.callback.log_to_file(log_file),
          ],
      )

  # ── Compile with tuning logs ────────────────────────────────────
  with autotvm.apply_history_best(log_file):
      with tvm.transform.PassContext(opt_level=3):
          lib = relay.build(relay_mod, target="cuda", params=relay_params)
"""
    print(AUTOTVM_API)
    print()

    print("  Ansor / MetaSchedule API:")
    ANSOR_API = """
  # ── Ansor (requires tvm.auto_scheduler) ────────────────────────
  from tvm import auto_scheduler

  tasks, task_weights = auto_scheduler.extract_tasks(
      relay_mod["main"], relay_params, target="cuda")
  print(f"Ansor: {len(tasks)} tasks (weighted by latency contribution)")

  tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
  tune_option = auto_scheduler.TuningOptions(
      num_measure_trials=2000,     ; total across all tasks
      runner=auto_scheduler.LocalRunner(repeat=3),
      measure_callbacks=[auto_scheduler.RecordToFile("ansor_log.json")],
  )
  tuner.tune(tune_option)

  # Compile with Ansor schedule
  with auto_scheduler.ApplyHistoryBest("ansor_log.json"):
      with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
          lib = relay.build(relay_mod, target="cuda", params=relay_params)

  # ── MetaSchedule (TVM 0.9+, recommended) ───────────────────────
  import tvm.meta_schedule as ms

  database = ms.database.JSONDatabase("./meta_schedule_workdir")
  tuned_mod, params = ms.relay_integration.tune_relay(
      mod=relay_mod,
      params=relay_params,
      target=tvm.target.Target("cuda"),
      config=ms.TuneConfig(
          num_trials_per_iter=64,
          max_trials_per_task=200,
          max_trials_global=2000,
          strategy="evolutionary",    ; or "replay_trace"
      ),
      work_dir="./meta_schedule_workdir",
  )
"""
    print(ANSOR_API)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · TVM Targets — CPU, GPU, Mobile, and microTVM": {
        "description": (
            "Explore TVM's multi-target compilation capabilities. "
            "Compile the same model for CPU (LLVM), CUDA GPU, and ARM (cross-compile). "
            "Show target string syntax: all flags and their meanings. "
            "Demonstrate the quantisation pipeline: float32 → int8 for edge deployment. "
            "Show microTVM: compiling for a Cortex-M4 microcontroller. "
            "Benchmark TVM vs TensorRT vs ONNX Runtime on standard models."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TVM TARGETS — CPU, GPU, MOBILE, AND MICROTVM")
print("=" * 65)
print()

try:
    import tvm
    import tvm.relay as relay
    from tvm import nd
    HAS_TVM = True
    print(f"  TVM {tvm.__version__}")
except ImportError:
    HAS_TVM = False
    print("  TVM not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Target string anatomy
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Target string anatomy: every flag explained")
print("━" * 65)
print()

TARGET_GUIDE = """
  TVM TARGET STRINGS — COMPLETE REFERENCE
  ════════════════════════════════════════════════════════════════

  TARGET SYNTAX: "<backend> [-flag=value ...]"
  Backend determines the code generator; flags tune it.

  ── CPU TARGETS (LLVM backend) ──────────────────────────────────

  "llvm"
    Most generic CPU target. Uses LLVM IR → native code.
    Detects host CPU capabilities automatically.

  "llvm -mcpu=core-avx2"
    Intel Haswell / Broadwell / Skylake (AVX2, 256-bit vectors).
    8-wide float32 SIMD. Available on most 2014–2021 Intel CPUs.

  "llvm -mcpu=cascadelake"
    Intel Cascade Lake (AVX-512 + VNNI for int8 dot products).
    16-wide float32 SIMD. VNNI enables int8 matmul at 4× throughput.
    Used in: Intel Xeon Scalable 2nd gen (data centre inference).

  "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    ARM64 (AArch64) with NEON SIMD. 4-wide float32.
    Targets: Raspberry Pi 4, server ARM (Ampere, Graviton 2/3).

  "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    ARM Scalable Vector Extension (SVE). Variable width (128–2048 bit).
    Targets: AWS Graviton 3 (SVE), Apple M-series (via emulation).

  "llvm -mtriple=arm-none-eabi -mcpu=cortex-m4 -mfloat-abi=hard"
    ARM Cortex-M4 microcontroller. 32-bit DSP with FPU.
    Used with microTVM for embedded deployment.

  "llvm -mtriple=riscv64-unknown-linux-gnu -mattr=+v"
    RISC-V 64-bit with vector extension. Emerging mobile/edge target.

  ── GPU TARGETS ──────────────────────────────────────────────────

  "cuda"
    NVIDIA GPU (detected automatically). Generates CUDA C → cubin via nvcc.
    Default for NVIDIA GPU tuning.

  "cuda -arch=sm_86"
    NVIDIA Ampere (RTX 3090, A5000, A6000). sm_86 specific features:
    - Tensor cores for TF32, BF16, INT8
    - Async memory copies (cp.async)
    - Warp-level matrix operations

  "cuda -arch=sm_90"
    NVIDIA Hopper (H100). Adds:
    - FP8 tensor cores (twice the throughput of BF16)
    - Tensor Memory Accelerator (TMA) for warp-cooperative loads
    - Transformer Engine integration

  "rocm"
    AMD GPU (ROCm stack). Generates HIP C++ → HSA binary.
    Same schedule primitives work; just swap "cuda" → "rocm".

  "vulkan"
    Cross-vendor GPU via Vulkan compute shaders + SPIR-V.
    Supports: NVIDIA, AMD, Intel, Qualcomm Adreno, ARM Mali.
    Used for mobile GPU deployment (Android Vulkan).

  "metal"
    Apple GPU (Metal Shading Language). Supports A-series/M-series.
    Best for iOS/macOS deployment of tuned models.

  "opencl"
    Legacy cross-vendor. Widely supported but slower than Vulkan.
    Still used for older mobile hardware (Mali GPU pre-2018).

  ── SPECIAL TARGETS ──────────────────────────────────────────────

  "webgpu"
    WebGPU compute shaders for in-browser ML inference.
    Generates WGSL (WebGPU Shading Language) shaders.

  "c"
    Portable C code. Works anywhere with a C99 compiler.
    Used by microTVM for microcontrollers with no TVM runtime.

  "hexagon -v68"
    Qualcomm Hexagon DSP (version 68). Used in Snapdragon SoCs.
    Has VTCM (Vector Tightly Coupled Memory) for ultra-fast tensor ops.

  "llvm -device=micro_dev"
    microTVM bare-metal target. Generates C with TVM micro runtime.
    Supports: Arduino, Zephyr RTOS, bare-metal STM32, Nordic nRF.

  ── TARGET CREATION IN PYTHON ────────────────────────────────────
  target = tvm.target.Target("cuda -arch=sm_86")
  target = tvm.target.cuda(arch="sm_86")         ; shorthand
  target = tvm.target.Target("llvm", host="llvm"); for cross-compilation
  target = tvm.target.arm_cpu("rasp4b")           ; Raspberry Pi 4 preset
"""
print(TARGET_GUIDE)

if HAS_TVM:
    print("  Available TVM targets on this machine:")
    for t_str in ["llvm", "cuda", "vulkan", "metal", "rocm"]:
        try:
            t = tvm.target.Target(t_str)
            print(f"    {t_str:12s}: available ✅  ({t.kind.name})")
        except Exception:
            print(f"    {t_str:12s}: not available on this machine")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Compile for multiple targets
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Compiling the same model for multiple targets")
print("━" * 65)
print()

MULTI_TARGET = """
  SAME MODEL → MULTIPLE TARGETS
  ════════════════════════════════════════════════════════════════

  TVM's key advantage: the Relay model is HARDWARE-AGNOSTIC.
  The same Relay IR compiles to CPU, GPU, mobile, MCU.
  No code changes required between targets.

  import tvm, tvm.relay as relay
  import numpy as np

  # Build the model ONCE in Relay
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  # Compile for x86 CPU (local development)
  with tvm.transform.PassContext(opt_level=3):
      lib_cpu  = relay.build(mod, target="llvm -mcpu=core-avx2",
                              params=params)

  # Compile for NVIDIA GPU (production serving)
  with tvm.transform.PassContext(opt_level=3):
      lib_cuda = relay.build(mod, target="cuda -arch=sm_86",
                              params=params)

  # Cross-compile for ARM64 Android (mobile deployment)
  target_android = tvm.target.Target(
      "llvm -mtriple=aarch64-linux-android21")
  with tvm.transform.PassContext(opt_level=3):
      lib_android = relay.build(mod,
          target=tvm.target.Target(
              "vulkan -device=adreno",   ; Qualcomm Adreno GPU
              host=target_android),       ; Android CPU host
          params=params)
  # Export for Android deployment:
  lib_android.export_library("model_android.so",
      fcompile=tvm.contrib.ndk.create_shared,
      options=["-shared", "-fPIC"])

  # Compile for WebGPU / WASM (browser inference)
  with tvm.transform.PassContext(opt_level=3):
      lib_web = relay.build(mod,
          target=tvm.target.Target(
              "webgpu",
              host="llvm -mtriple=wasm32-unknown-unknown-wasm"),
          params=params)

  # ── What changes per target: NOTHING in user code ────────────────
  # The model graph (Relay) is identical.
  # TVM changes internally: schedule templates, code generator, runtime.
  # Output is target-specific .so / .cubin / .wasm / .hex

  # ── Saving and loading compiled models ─────────────────────────────
  lib_cpu.export_library("model_cpu.so")     ; shared library
  # --- on deployment machine (same TVM version, same OS) ---
  loaded = tvm.runtime.load_module("model_cpu.so")
  dev    = tvm.cpu(0)
  module = tvm.contrib.graph_executor.GraphModule(loaded["default"](dev))
"""
print(MULTI_TARGET)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Quantisation pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Quantisation: float32 → int8 for edge deployment")
print("━" * 65)
print()

QUANT_PIPELINE = """
  TVM QUANTISATION PIPELINE
  ════════════════════════════════════════════════════════════════

  TVM supports both post-training quantisation (PTQ) and
  quantisation-aware training (QAT) via the relay.quantize module.

  STEP 1: CALIBRATION
    Run a small representative dataset (100–1000 samples) through
    the float model to collect activation statistics (min/max per tensor).
    These statistics determine the quantisation scale factors.

    import tvm.relay.quantize as quantize

    # Define calibration function
    def calibrate_dataset():
        for i, (image, label) in enumerate(calibration_data):
            if i >= 100: break
            yield {"input": image}    ; yield input dict

    # Run calibration
    with relay.quantize.qconfig(
        calibrate_mode="kl_divergence",    ; minimise KL divergence (better)
        weight_scale="max",                ; scale weights by max abs value
        skip_conv_layers=[0],              ; keep first conv in float (fragile)
    ):
        mod_quantized = quantize.quantize(mod, params,
                                           dataset=calibrate_dataset())

  STEP 2: INSPECT THE QUANTISED RELAY GRAPH
    The quantized model has new ops:
      relay.quantize.simulated_quantize: float activations → quantised
      relay.quantize.quantize: float → int8
      relay.quantize.dequantize: int8 → float (at model output)

    In the quantised graph:
      nn.dense(int8, int8) → int32 accumulation → requantize → int8
      Memory: 4× smaller than float32 (int8 weights use 1 byte vs 4)
      Speed: 2–4× faster (INT8 Tensor Cores / INT8 DSP instructions)

  STEP 3: COMPILE THE QUANTISED MODEL
    Same as float model — just pass the quantised mod to relay.build:

    with tvm.transform.PassContext(opt_level=3):
        lib_int8 = relay.build(
            mod_quantized,
            target="llvm -mcpu=cascadelake",  ; has AVX-512 VNNI for int8
            params=params_quantized)

  STEP 4: VALIDATE ACCURACY (critical!)
    Float model accuracy:    72.1% top-1 (ResNet-50 ImageNet)
    Int8 PTQ (kl_divergence): 71.6% top-1  (< 0.5% drop: acceptable)
    Int8 PTQ (minmax):        70.2% top-1  (1.9% drop: investigate)
    Int8 QAT:                 71.9% top-1  (< 0.2% drop: best)

  QUANTISATION CONFIG OPTIONS:
    calibrate_mode:
      "global_scale":   use one global scale (fastest, least accurate)
      "kl_divergence":  find scale minimising KL(float_dist || quant_dist)
      "percentile":     use 99.99th percentile (robust to outliers)

    nbit_input, nbit_weight: 8 (standard int8), 4 (int4, experimental)

    skip_conv_layers: list of conv layer indices to keep in float32
      (useful for first/last layers that are sensitive to quantisation)

    do_simulation: if True, simulate int8 in float32 (for debugging)

  PERFORMANCE EXPECTATIONS:
  ────────────────────────────────────────────────────────────────
  Hardware         Float32 (ms)  Int8 (ms)  Speedup
  ARM Cortex-A55   8.3           2.1        3.95×   (mobile CPU)
  Cortex-M4 (MCU)  N/A           12.4       N/A     (float too slow)
  Intel Cascade Lake  0.9         0.22       4.1×    (AVX-512 VNNI)
  Qualcomm Adreno 650  3.2        0.85       3.8×    (Vulkan int8)
"""
print(QUANT_PIPELINE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: microTVM reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — microTVM: ML on bare-metal microcontrollers")
print("━" * 65)
print()

MICROTVM_GUIDE = """
  MICROTVM — DEPLOYING ML ON MICROCONTROLLERS
  ════════════════════════════════════════════════════════════════

  Target hardware examples:
    ARM Cortex-M4 (STM32F4xx): 168 MHz, 192 KB SRAM, 1 MB Flash
    ARM Cortex-M33 (nRF9160):  64 MHz, 256 KB SRAM, 1 MB Flash
    ARM Cortex-M7 (STM32H7):   480 MHz, 1 MB SRAM, 2 MB Flash

  TVM's micro runtime footprint:
    Runtime code:   ~16 KB (C code, links into firmware)
    Operator code:  ~50–200 KB (generated C for the model's ops)
    Model weights:  quantised (int8 → 1 byte/weight)
    Working memory: allocated from MCU SRAM

  KEY CONSTRAINTS vs server deployment:
    1. NO DYNAMIC MEMORY: malloc/free forbidden → all buffers are static.
       TVM pre-allocates all workspace buffers at compile time.
    2. NO FLOATING POINT (usually): must quantise to int8/int16.
    3. NO OS: no threads, no file system, no printf (sometimes).
    4. CODE SIZE LIMIT: everything must fit in flash.

  MICROTVM COMPILATION PIPELINE:

  # ── 1. Import model (TFLite most common for MCU models) ─────────
  import tflite
  tflite_model_buf = open("keyword_spotting.tflite", "rb").read()
  mod, params = relay.frontend.from_tflite(
      tflite_model_buf, shape_dict={"input": (1, 49, 10, 1)},
      dtype_dict={"input": "float32"})

  # ── 2. Quantise (float32 → int8) ─────────────────────────────────
  mod = relay.quantize.quantize(mod, params, dataset=calibration_data)

  # ── 3. Compile for microcontroller target ─────────────────────────
  target = tvm.target.Target(
      "c -keys=arm_cpu -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16")
  # The "c" backend generates portable C code (no LLVM assembly)
  # This C code is then compiled by arm-none-eabi-gcc

  runtime = tvm.relay.backend.Runtime("crt",   ; C runtime (no OS needed)
                                       {"system-lib": True})
  executor = tvm.relay.backend.Executor("aot", ; Ahead-of-Time (no graph executor)
                                         {"interface-api": "c",
                                          "unpacked-api": True})

  with tvm.transform.PassContext(
      opt_level=3,
      config={"tir.disable_vectorize": True,   ; MCU may not have SIMD
              "tir.usmp.enable": True,          ; unified static memory planner
              "tir.usmp.algorithm": "hill_climb"}):
      lib = relay.build(mod, target=target, runtime=runtime,
                         executor=executor, params=params)

  # ── 4. Extract generated C files ──────────────────────────────────
  lib.export_library("compiled_model.tar")
  # Contains: model.c, model.h, model_lib_c.c, params.bin

  # ── 5. Integrate into firmware project ────────────────────────────
  // In your Arduino/Zephyr/bare-metal C firmware:
  #include "tvmgen_default.h"

  // Inputs/outputs are statically allocated:
  static int8_t input_buffer[1 * 49 * 10 * 1];
  static int8_t output_buffer[12];   // 12 keyword classes

  // Inference (no dynamic memory, no OS calls):
  struct tvmgen_default_inputs inputs = {.input_1 = input_buffer};
  struct tvmgen_default_outputs outputs = {.output_0 = output_buffer};
  tvmgen_default_run(&inputs, &outputs);

  int keyword = argmax(output_buffer, 12);

  REAL-WORLD MICROTVM USE CASES:
  ─────────────────────────────────────────────────────────────────
  Keyword spotting:     "Hey Siri", "OK Google" on MCU (<1mW power)
  Anomaly detection:    industrial vibration sensor with ML on MCU
  Person detection:     camera trigger on Cortex-M7 (no WiFi needed)
  ECG classification:   heart arrhythmia on wearable MCU
  Predictive maintenance: edge inference on industrial controller

  AUTOTUNE FOR MCU (microTVM autotuning):
    TVM can autotune schedules for MCU by running measurements on
    actual hardware via USB/serial connection:
    tracker = tvm.rpc.tracker.Tracker(host="0.0.0.0", port=9190)
    remote  = tvm.rpc.connect("device_host", 9090)  ; device-side RPC server
    → AutoTVM finds optimal tiling for the specific MCU's cache sizes
"""
print(MICROTVM_GUIDE)
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · TVM in the Connected Stack — End-to-End Workflow and Benchmarks": {
        "description": (
            "Complete end-to-end workflow: PyTorch → StableHLO → TVM Relax → deployment. "
            "Show how TVM consumes StableHLO from JAX and PyTorch models. "
            "Benchmark TVM vs XLA vs ONNX Runtime vs TensorRT on ResNet/BERT. "
            "Demonstrate tvmc: TVM's unified command-line compiler. "
            "Show the Relax next-generation IR with dynamic shapes. "
            "Summarise TVM's position in the full connected compiler stack."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TVM IN THE CONNECTED STACK — END-TO-END WORKFLOW AND BENCHMARKS")
print("=" * 65)
print()

try:
    import tvm
    import tvm.relay as relay
    HAS_TVM = True
    print(f"  TVM {tvm.__version__}")
except ImportError:
    HAS_TVM = False
    print("  TVM not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: StableHLO → TVM Relax workflow
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — StableHLO → TVM: the cross-framework import path")
print("━" * 65)
print()

STABLEHLO_TVM = """
  STABLEHLO → TVM RELAX: THE CANONICAL IMPORT PATH (TVM 0.14+)
  ════════════════════════════════════════════════════════════════

  The StableHLO → TVM path enables training with ANY framework
  and deploying with TVM on ANY hardware.

  ── STEP 1: Produce StableHLO from any framework ─────────────────

  FROM JAX:
    import jax, jax.numpy as jnp

    def my_model(params, x):
        W1, b1, W2, b2 = params
        h = jnp.maximum(x @ W1 + b1, 0.0)
        return jax.nn.softmax(h @ W2 + b2, axis=-1)

    # Export to StableHLO bytes
    x_abs    = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    exported = jax.export.export(jax.jit(my_model))(params_abs, x_abs)
    stablehlo_bytes = exported.serialize()

  FROM PYTORCH (via torch-mlir):
    import torch_mlir
    scripted = torch.jit.script(pytorch_model)
    module = torch_mlir.compile(
        scripted, example_inputs,
        output_type=torch_mlir.OutputType.STABLEHLO)
    stablehlo_text = module.operation.get_asm()

  ── STEP 2: Import StableHLO into TVM Relax ─────────────────────

  import tvm
  import tvm.relax as relax

  # Parse StableHLO → TVM IRModule (with Relax ops)
  with open("model.mlir") as f:
      stablehlo_text = f.read()

  # Import (TVM translates stablehlo.* ops → relax.* ops):
  #   stablehlo.dot_general → relax.matmul
  #   stablehlo.reduce      → relax.sum / relax.max
  #   stablehlo.convolution → relax.nn.conv2d
  #   stablehlo.maximum     → relax.nn.relu (if pattern matches)
  ir_module = tvm.relax.from_stablehlo(stablehlo_text)
  print(ir_module.script())    ; print the Relax IR

  ── STEP 3: Relax passes → TIR lowering ──────────────────────────

  # Apply graph-level optimisations
  with tvm.transform.PassContext(opt_level=3):
      ir_module = relax.transform.FuseOps()(ir_module)
      ir_module = relax.transform.FoldConstant()(ir_module)

  # Lower to TIR (one TIR function per fused operator group)
  target = tvm.target.Target("llvm -mcpu=core-avx2")
  ex = relax.build(ir_module, target=target)

  ── STEP 4: Deploy ───────────────────────────────────────────────

  vm = tvm.runtime.vm.VirtualMachine(ex, tvm.cpu())
  output = vm["main"](params_tuple, input_tensor)

  # Export for deployment:
  ex.export_library("model.so")
  # On deployment machine:
  ex_loaded = tvm.runtime.load_module("model.so")
  vm_loaded = tvm.runtime.vm.VirtualMachine(ex_loaded, tvm.cpu())
"""
print(STABLEHLO_TVM)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Performance benchmarks (representative results)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Performance benchmarks: TVM vs alternatives")
print("━" * 65)
print()

BENCHMARKS = """
  REPRESENTATIVE INFERENCE BENCHMARKS
  ════════════════════════════════════════════════════════════════
  Sources: TVM papers (OSDI 2018, NeurIPS 2022), MLPerf inference,
  community benchmarks on github.com/tlc-pack/tvm-bench.
  All results are approximate and hardware/version dependent.
  Always benchmark on your specific hardware and model.

  ── GPU: NVIDIA A100 (CUDA 11.8, batch=1) ──────────────────────
  Model                  PyTorch  XLA/JAX  TensorRT  TVM(MetaSch.)
  ResNet-50 (ms)           2.1      1.8       1.4         1.7
  BERT-Base (ms)           4.3      3.9       3.1         3.5
  GPT-2 (token/s)        320       395       450         380
  MobileNetV2 (ms)         0.9      0.7       0.6         0.6
  ViT-Base/16 (ms)         4.8      4.1       3.5         3.8

  KEY INSIGHT:
    TensorRT wins for standard models (hand-tuned cuDNN kernels).
    TVM is competitive and beats TensorRT for:
      - Custom/novel operators (no cuDNN equivalent)
      - Specific shapes not optimised in cuDNN
      - Non-NVIDIA hardware (where TensorRT doesn't run)

  ── CPU: Intel Xeon (Cascade Lake, 1 core, batch=1) ────────────
  Model                  PyTorch  ONNX RT  OpenVINO   TVM(MetaSch.)
  ResNet-50 (ms)          42        18        14           12
  BERT-Base (ms)         185        52        44           38
  MobileNetV2 (ms)        15         7         6            5

  KEY INSIGHT:
    TVM wins on CPU because MetaSchedule finds tile sizes that
    perfectly fit the Cascade Lake's cache hierarchy and AVX-512 VNNI.
    OpenVINO (Intel's own tool) is close but TVM edges it out.

  ── MOBILE: Qualcomm Snapdragon 865 (Adreno 650 GPU) ───────────
  Model                  TFLite   PyTorch M  ONNX RT   TVM(Ansor)
  MobileNetV2 (ms)         6.2       8.1       5.8         4.3
  MobileNetV3-Large (ms)   5.1       7.2       5.2         3.9
  EfficientNet-Lite0 (ms)  8.4       11.3      7.9         6.1

  KEY INSIGHT:
    TVM with Ansor finds Vulkan schedules that outperform TFLite's
    hand-optimised kernels because Ansor discovers Adreno-specific
    tiling patterns that no human would write.

  ── MCU: ARM Cortex-M4 (STM32F4, 168 MHz) ─────────────────────
  Model                  TFLite Micro  CMSIS-NN  microTVM
  Keyword Spotting (ms)     18.2          9.3       7.8
  Person Detection (ms)    124           68        54
  Anomaly Det. (ms)          4.1          3.2       2.9

  KEY INSIGHT:
    microTVM outperforms even hand-optimised CMSIS-NN by finding
    better loop tiling for the M4's 16 KB L1 data cache.

  ── WHAT THESE NUMBERS MEAN ─────────────────────────────────────
  No tool is universally best. Choose based on:
    Standard NVIDIA GPU inference → TensorRT (best cuDNN integration)
    Research/custom models        → TVM (handles any operator)
    Mobile Vulkan/Metal           → TVM Ansor (best schedule search)
    Microcontrollers              → microTVM (the ONLY serious option)
    Training                      → XLA/JAX (far ahead)
    Intel CPU deployment          → TVM or OpenVINO (competitive)
"""
print(BENCHMARKS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: tvmc CLI — TVM's command-line compiler
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — tvmc: the unified TVM CLI")
print("━" * 65)
print()

TVMC_GUIDE = """
  TVMC — TVM MODEL COMPILER CLI
  ════════════════════════════════════════════════════════════════
  tvmc is TVM's unified command-line tool for compiling, tuning,
  and running ML models without writing Python code.

  BASIC COMPILATION:
  ─────────────────────────────────────────────────────────────────
  # ONNX → CPU (x86 with AVX2)
  tvmc compile resnet50.onnx \\
      --target "llvm -mcpu=core-avx2" \\
      --input-shapes "input:[1,3,224,224]" \\
      --output resnet50_cpu.tar

  # ONNX → CUDA GPU
  tvmc compile resnet50.onnx \\
      --target "cuda -arch=sm_86" \\
      --output resnet50_cuda.tar

  # TFLite → ARM mobile (cross-compile)
  tvmc compile mobilenet.tflite \\
      --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon" \\
      --output mobilenet_arm64.tar

  TUNING + COMPILATION:
  ─────────────────────────────────────────────────────────────────
  # Step 1: Auto-tune (measures on actual hardware)
  tvmc tune resnet50.onnx \\
      --target "cuda -arch=sm_86" \\
      --tuner xgb \\
      --trials 1000 \\
      --output resnet50_tuning_log.json \\
      --repeat 3

  # Step 2: Compile with tuning results
  tvmc compile resnet50.onnx \\
      --target "cuda -arch=sm_86" \\
      --tuning-records resnet50_tuning_log.json \\
      --output resnet50_tuned.tar

  RUNNING A COMPILED MODEL:
  ─────────────────────────────────────────────────────────────────
  # Run and benchmark
  tvmc run resnet50_tuned.tar \\
      --device gpu \\
      --inputs input.npz \\
      --output-format npz \\
      --print-time \\
      --repeat 100

  # Output:
  # Execution time summary:
  #  mean (ms)    median (ms)    max (ms)    min (ms)    std (ms)
  #  1.714         1.709          1.783        1.703       0.023

  SUPPORTED INPUT FORMATS:
  ─────────────────────────────────────────────────────────────────
  tvmc auto-detects the input format:
    .onnx       ONNX model (most common for deployment)
    .pb         TensorFlow SavedModel / frozen graph
    .tflite     TFLite FlatBuffer
    .pt         TorchScript (PyTorch)
    .relay      Relay IR text
    .mlir       MLIR / StableHLO
    .json+.params  Previously exported TVM Relay JSON

  tvmc can be installed standalone:
    pip install tlcpack   ; community binary builds
    # or build TVM from source for latest features

  TVMC VS PYTHON API:
    tvmc:   simpler, good for standard workflows, CI/CD pipelines
    Python: more flexible, required for custom schedules,
            MetaSchedule, advanced deployment scenarios
"""
print(TVMC_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Connected stack summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — TVM in the connected compiler stack")
print("━" * 65)
print()

STACK_SUMMARY = """
  TVM POSITION IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                     │
  │  TVM uses LLVM as CPU and GPU (NVPTX/AMDGCN) codegen backend.        │
  │  TVM's own loop transformations (tiling/vectorise) run BEFORE LLVM.  │
  │  LLVM is used as a code emitter, not as an optimiser.                │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                     │
  │  TVM Relax (TVM 0.11+) is an MLIR dialect (relax.*).                 │
  │  MLIR's pass manager drives Relax optimisation passes.               │
  │  TOSA dialect → Relay bridge enables TFLite/TOSA models in TVM.      │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 03: CIRCT                                                    │
  │  VTA (TVM's hardware accelerator) uses a CIRCT-like design flow.     │
  │  Custom ASICs targeted by TVM use CIRCT for RTL generation.          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 04: ENZYME                                                   │
  │  TVM's Relay has built-in AD for training workflows.                 │
  │  Custom TVM operators can use Enzyme for gradient computation.       │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 05: XLA / MODULE 06: OpenXLA                                 │
  │  Compete for GPU/CPU deployment (TVM stronger on edge/mobile).       │
  │  XLA dominates training; TVM dominates novel hardware deployment.    │
  │  StableHLO (OpenXLA) is TVM's primary cross-framework import format. │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 07: StableHLO                                                │
  │  TVM's primary import path: tvm.relax.from_stablehlo()               │
  │  Train with JAX/PyTorch → StableHLO → deploy with TVM on any HW.     │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM (THIS MODULE)                                        │
  │  Two IRs: Relay/Relax (graph) + TIR (loop nests).                    │
  │  Three search systems: AutoTVM, Ansor, MetaSchedule.                 │
  │  N backends: LLVM, CUDA, ROCm, Vulkan, Metal, WebGPU, C.             │
  │  Special targets: microTVM (MCU), VTA (ASIC), Hexagon (DSP).         │
  └──────────────────────────────────────────────────────────────────────┘

  QUICK REFERENCE:
"""
print(STACK_SUMMARY)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Task                     │ TVM API / Tool                        │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Import ONNX              │ relay.frontend.from_onnx()            │")
print("  │ Import PyTorch           │ relay.frontend.from_pytorch()         │")
print("  │ Import StableHLO         │ tvm.relax.from_stablehlo()            │")
print("  │ Compile model            │ relay.build(mod, target, params)      │")
print("  │ Run compiled model       │ GraphModule(lib['default'](dev))      │")
print("  │ AutoTVM tuning           │ autotvm.tuner.XGBTuner(task).tune()   │")
print("  │ Ansor tuning             │ auto_scheduler.TaskScheduler().tune() │")
print("  │ MetaSchedule tuning      │ ms.relay_integration.tune_relay()     │")
print("  │ Export library           │ lib.export_library('model.so')        │")
print("  │ CLI compile              │ tvmc compile model.onnx --target llvm │")
print("  │ CLI tune                 │ tvmc tune model.onnx --trials 1000    │")
print("  │ Quantise (int8)          │ relay.quantize.quantize(mod, params)  │")
print("  │ microTVM target          │ tvm.target.Target('c -mcpu=cortex-m4')│")
print("  │ Define TE computation    │ te.compute((M,N), lambda i,j: ...)    │")
print("  │ Create schedule          │ te.create_schedule(C.op)              │")
print("  │ Split loop               │ s[C].split(loop, factor=32)           │")
print("  │ Reorder loops            │ s[C].reorder(i, k, j)                 │")
print("  │ Vectorize loop           │ s[C].vectorize(j_inner)               │")
print("  │ Parallel loop            │ s[C].parallel(i_outer)                │")
print("  │ GPU shared memory        │ s[C].cache_read(A, 'shared', [C])     │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Project                  │ apache/tvm (github)                   │")
print("  │ Documentation            │ tvm.apache.org/docs                   │")
print("  │ Community                │ discuss.tvm.apache.org                │")
print("  │ Prebuilt binaries        │ pip install tlcpack                   │")
print("  │ Source build             │ cmake + python setup.py install       │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()

if HAS_TVM:
    print("  TVM runtime verification:")
    try:
        M, K, N = 64, 64, 64
        from tvm import te
        A = te.placeholder((M,K), name="A", dtype="float32")
        B = te.placeholder((K,N), name="B", dtype="float32")
        k = te.reduce_axis((0,K), name="k")
        C = te.compute((M,N), lambda i,j: te.sum(A[i,k]*B[k,j], axis=k), name="C")
        s = te.create_schedule(C.op)
        i,j = s[C].op.axis; (rk,) = s[C].op.reduce_axis
        j_o, j_i = s[C].split(j, factor=8)
        s[C].reorder(i, rk, j_o, j_i)
        s[C].vectorize(j_i)
        func = tvm.build(s, [A,B,C], target="llvm", name="matmul_verify")
        rng  = np.random.default_rng(0)
        a_np = rng.random((M,K)).astype("float32")
        b_np = rng.random((K,N)).astype("float32")
        dev  = tvm.cpu(0)
        a_nd = nd.array(a_np, dev); b_nd = nd.array(b_np, dev)
        c_nd = nd.array(np.zeros((M,N),"float32"), dev)
        func(a_nd, b_nd, c_nd)
        err  = float(np.max(np.abs(c_nd.numpy() - a_np @ b_np)))
        print(f"    TE matmul ({M}×{K}×{N}) correctness: max_err={err:.2e} "
              f"{'✅' if err < 1e-3 else '❌'}")
    except Exception as e:
        print(f"    Note: {e}")
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