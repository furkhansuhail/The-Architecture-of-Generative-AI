"""
TVM — Apache TVM: The End-to-End Deep Learning Compiler
=========================================================

Apache TVM is a compiler stack for deep learning that takes a neural
network model expressed in any major framework (PyTorch, ONNX, TensorFlow,
MXNet) and compiles it to optimised native code for any hardware target —
x86 CPUs, ARM CPUs, NVIDIA GPUs, AMD GPUs, Mali GPUs, RISC-V processors,
FPGAs, and custom ML accelerators.

TVM's unique contribution in the compiler stack is AUTO-SCHEDULING:
while XLA, MLIR, and LLVM apply deterministic analytical transformations,
TVM searches the space of possible implementations for each operator and
measures which one is fastest on the target hardware. No hand-written
hardware kernels, no manually tuned tile sizes — TVM finds them by
compiling and benchmarking thousands of candidate implementations.

This search-based approach means TVM can achieve performance competitive
with hand-tuned libraries like cuDNN or oneDNN on hardware where no such
libraries exist — which is the vast majority of hardware in the world.

In the connected compiler stack:
    LLVM (module 09)  ← TVM uses LLVM as its CPU backend
    MLIR (module 10)  ← TVM converges with MLIR via Relay/StableHLO
    XLA  (module 10)  ← XLA: analytical fusion. TVM: search-based tuning
    TVM  (this module)← end-to-end compiler with auto-scheduling

"""

import textwrap
import re

TOPIC_NAME   = "TVM — Apache TVM Deep Learning Compiler"
DISPLAY_NAME = "07a · TVM"
ICON         = "🔬"
SUBTITLE     = "Auto-Scheduling, Relay IR, and Universal Hardware Deployment"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT TVM IS AND THE PROBLEM IT SOLVES

### The Hardware Fragmentation Problem

    In 2017, deploying a trained neural network to production meant:
        NVIDIA GPU:    use cuDNN — hand-tuned by NVIDIA engineers for years.
        Intel CPU:     use oneDNN — hand-tuned by Intel engineers for years.
        ARM CPU:       use XNNPACK or ACL — if your architecture is supported.
        AMD GPU:       use MIOpen — if it has been ported.
        Google TPU:    use XLA — and only XLA.
        Custom chip:   write kernels by hand, or give up.

    The problem: the world has hundreds of ML hardware targets.
    Hand-tuned libraries exist for maybe ten of them.
    The other ninety get generic, unoptimised fallback code.

    TVM's answer: don't write kernels by hand. Let the compiler find the
    optimal implementation by searching the space of possible implementations
    and measuring performance directly on the target hardware.

### TVM's Core Philosophy

    ┌─────────────────────────────────────────────────────────────────┐
    │  "The performance of a neural network operator depends on the    │
    │  target hardware in ways that cannot be predicted analytically.  │
    │  We should MEASURE, not GUESS."                                  │
    │                                  — Tianqi Chen, TVM paper 2018  │
    └─────────────────────────────────────────────────────────────────┘

    This measurement-based approach (auto-tuning / auto-scheduling) is
    TVM's defining characteristic and its biggest difference from XLA.

    XLA:   algebraic transformations, layout assignment, analytical fusion.
            Fast to compile. Performance bounded by what the analytical
            model predicts.

    TVM:   compile many candidate implementations, benchmark each,
            keep the winner. Slow to tune (minutes to hours). But can
            find implementations that hand-written code would miss.

### Where TVM Fits

    TVM sits at the same level as XLA in the compiler stack —
    both compile ML computation graphs to hardware code.
    They are complementary, not competing:

        XLA:   best for Google's hardware (TPU), JAX workflows, fusion.
        TVM:   best for hardware diversity, custom chips, auto-scheduling.

    In practice, both use LLVM for CPU code generation,
    and both increasingly use MLIR/StableHLO as their IR.

    ┌────────────────────────────────────────────────────────────────┐
    │  ML model (PyTorch / ONNX / TF / MXNet / Keras)               │
    │     ↓  TVM frontend import                                     │
    │  Relay IR  (high-level graph IR — the computation graph)       │
    │     ↓  Graph-level optimisations (fusion, layout, const fold)  │
    │  TE (Tensor Expressions)  — operator-level compute definition  │
    │     ↓  Schedule lowering (tile, vectorise, unroll, parallelise)│
    │  TIR (Tensor IR)  — loop-level IR with explicit memory          │
    │     ↓  LLVM codegen / CUDA codegen / OpenCL codegen            │
    │  LLVM IR → x86/ARM  /  CUDA PTX → GPU  /  Metal → Apple       │
    └────────────────────────────────────────────────────────────────┘


##### PART 2 — RELAY IR: THE COMPUTATION GRAPH REPRESENTATION

### What Relay Is

    Relay is TVM's high-level intermediate representation for neural networks.
    It is a functional language with:
        - Static types (tensor shapes and dtypes known at compile time)
        - First-class functions (models are Relay functions)
        - Pattern matching and algebraic data types
        - Let bindings for naming intermediate computations

    Relay is designed for graph-level transformations:
        operator fusion, layout optimisation, constant folding,
        dead code elimination, and quantisation.

### Relay IR Structure

    A Relay program is a Module containing named global functions:

        def @main(%data: Tensor[(1, 3, 224, 224), float32],
                  %weight: Tensor[(64, 3, 3, 3), float32]) -> Tensor[(1, 64, 222, 222), float32] {
            %0 = nn.conv2d(%data, %weight, padding=[0, 0, 0, 0])
                  /* ty=Tensor[(1, 64, 222, 222), float32] */;
            %1 = nn.bias_add(%0, %bias)
                  /* ty=Tensor[(1, 64, 222, 222), float32] */;
            %2 = nn.relu(%1)
                  /* ty=Tensor[(1, 64, 222, 222), float32] */;
            %2   /* return value */
        }

    Key operators in Relay:
        nn.conv2d, nn.dense (matmul+bias), nn.batch_norm
        nn.relu, nn.softmax, nn.sigmoid, nn.tanh
        add, subtract, multiply (elementwise)
        reshape, transpose, expand_dims, squeeze
        nn.max_pool2d, nn.avg_pool2d, nn.global_avg_pool2d
        concatenate, split, stack
        image.resize2d (for super-resolution, detection)

### Relay Graph-Level Passes

    Relay runs a series of graph-level passes before lowering to TIR:

    AlterOpLayout:
        Converts tensor layouts for hardware efficiency.
        NCHW → NCHW4c (vectorised channels) for ARM.
        NCHW → NHWC for edge hardware.
        Same idea as XLA's layout assignment but more configurable.

    FuseOps:
        Identifies which operators can be fused into a single kernel.
        Uses a dataflow analysis similar to XLA's fusion pass.
        Fused ops become a single Relay Function marked with "Primitive".

    EliminateCommonSubexpr:
        CSE — same computation used twice → compute once, reuse.

    FoldConstant:
        Pre-compute any subgraph whose inputs are all constants.
        e.g., the position encoding table in a Transformer is constant.

    ConvertLayout:
        Inserts layout conversion ops to match hardware preference.

    QuantizeAnnotate / Quantize:
        Converts float ops to int8 ops with scale/zero_point metadata.
        Post-training quantisation or quantisation-aware training.

### Importing Models into Relay

    TVM imports from all major frameworks:

    From ONNX:
        import onnx, tvm.relay as relay
        onnx_model = onnx.load("resnet50.onnx")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    From PyTorch (via TorchScript):
        scripted = torch.jit.trace(model, example_input)
        mod, params = relay.frontend.from_pytorch(scripted, input_shapes)

    From TensorFlow:
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shapes)

    From MXNet:
        mod, params = relay.frontend.from_mxnet(symbol, arg_params, shape)

    From Keras:
        mod, params = relay.frontend.from_keras(model, shape_dict)

    All frontends produce the same Relay IR — the backend is identical
    regardless of which framework the model came from.


##### PART 3 — TENSOR EXPRESSIONS (TE) AND SCHEDULES

### What TE Is

    TE (Tensor Expressions) is TVM's DSL for defining operator computations.
    An operator is defined as a function from index coordinates to a value.

    This is the MATHEMATICAL SPECIFICATION of what to compute:

        # Matrix multiply C[i,k] = sum_j(A[i,j] * B[j,k])
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, n: te.sum(A[i, k] * B[k, n], axis=k),
            name='C'
        )

    The lambda body is the mathematical formula for each element.
    The reduce_axis marks the summation dimension.

    TE for elementwise relu:
        B = te.compute(
            A.shape,
            lambda *i: te.max(A(*i), tvm.tir.const(0, A.dtype)),
            name='relu'
        )

    TE for softmax (requires multiple stages):
        max_val  = te.compute((M,), lambda i: te.max(A[i, k], axis=k))
        exp_A    = te.compute(A.shape, lambda i,j: te.exp(A[i,j] - max_val[i]))
        sum_exp  = te.compute((M,), lambda i: te.sum(exp_A[i,k], axis=k))
        softmax  = te.compute(A.shape, lambda i,j: exp_A[i,j] / sum_exp[i])

### The Schedule: Separating What from How

    TE defines WHAT to compute. The Schedule defines HOW to compute it.

    The same matmul can be computed in many different ways:
        - Different tile sizes (8×8, 16×16, 32×32, ...)
        - Different loop orderings (i,j,k vs i,k,j vs k,i,j)
        - With or without vectorisation
        - With or without cache prefetching
        - With or without multi-threading

    The Schedule is the set of transformations applied to the loop nest:

        s = te.create_schedule(C.op)

        # Access the loop axes
        i, n = s[C].op.axis        # output dimensions
        k,   = s[C].op.reduce_axis # reduction dimension

        # Tile the outer loops (split into 32-element tiles)
        i_outer, i_inner = s[C].split(i, factor=32)
        n_outer, n_inner = s[C].split(n, factor=32)

        # Reorder for cache-friendly access
        s[C].reorder(i_outer, n_outer, i_inner, n_inner, k)

        # Vectorise the innermost loop (SIMD)
        s[C].vectorize(n_inner)

        # Parallelise the outer loop (multi-threading)
        s[C].parallel(i_outer)

### Schedule Primitives

    TVM's schedule primitives are the knobs the auto-scheduler searches:

    split(axis, factor):
        Split one loop into two: outer × inner.
        Creates tiling for cache locality.

    reorder(axes):
        Change loop ordering.
        i,j,k → k,i,j improves B matrix cache reuse in matmul.

    fuse(axis1, axis2):
        Merge two loops into one.
        Enables better parallelism granularity.

    vectorize(axis):
        Hint to LLVM to vectorise this loop (emit SIMD instructions).
        Works on the innermost loop when loop body is elementwise.

    unroll(axis):
        Unroll a loop a fixed number of times.
        Reduces loop overhead; enables instruction-level parallelism.

    parallel(axis):
        Mark an outer loop for parallel execution (OpenMP or pthreads).

    bind(axis, thread_axis):
        Bind a loop to a GPU thread/block dimension.
        s[C].bind(n_outer, te.thread_axis("blockIdx.x"))
        s[C].bind(n_inner, te.thread_axis("threadIdx.x"))

    compute_at(other_stage, axis):
        Move computation of an intermediate stage to inside another loop.
        Used for cache staging (compute shared memory tiles).

    cache_read / cache_write:
        Introduce explicit cache buffers (GPU shared memory, CPU L1 cache).

### Why Schedules Are the Key to Performance

    The SAME mathematical computation with different schedules can differ
    10–100× in speed because of:
        Cache behaviour:   reading data that fits in L1 cache vs HBM
        SIMD utilisation:  8 operations per cycle vs 1
        Thread occupancy:  100% GPU utilisation vs 20%
        Memory access:     coalesced GPU reads vs random access

    Hand-optimised CUDA kernels like cuBLAS contain:
        Carefully chosen tile sizes (tuned per GPU architecture)
        Register file usage maximised
        Shared memory banks organised to avoid conflicts
        Prefetch instructions to hide memory latency

    TVM's auto-scheduler FINDS these parameters automatically by search.


##### PART 4 — AUTO-TUNING: AUTOTVM

### What AutoTVM Is

    AutoTVM is TVM's first-generation auto-tuning system.
    It uses TEMPLATE-GUIDED search: a programmer writes a schedule template
    that defines WHICH schedule parameters to search, and AutoTVM
    searches those parameters using machine learning to guide the search.

    Key concepts:

    ConfigSpace:
        The set of all valid parameter combinations for a template.
        Example for a matrix multiply template:
            tile_x:   [1, 2, 4, 8, 16, 32, 64]    (7 choices)
            tile_y:   [1, 2, 4, 8, 16, 32, 64]    (7 choices)
            tile_k:   [1, 2, 4, 8, 16, 32]         (6 choices)
            unroll:   [1, 2, 4, 8]                  (4 choices)
            vectorise:[True, False]                  (2 choices)
        Total: 7 × 7 × 6 × 4 × 2 = 2,352 configurations

    ConfigEntity:
        One specific point in the ConfigSpace (e.g., tile_x=16, tile_y=8, ...).

    Measure:
        For each ConfigEntity, AutoTVM:
        1. Generates the schedule with those parameters
        2. Compiles to the target
        3. Runs on the target hardware N times
        4. Records the median execution time

    XGBoost cost model:
        After measuring a sample of configs, AutoTVM trains an XGBoost model
        to PREDICT performance of unmeasured configs.
        Uses this to guide subsequent measurements toward promising regions.

### AutoTVM Search Process

    for trial in range(n_trials):                # e.g., 2000 trials
        configs = cost_model.predict_best(k=64)  # predict top-64 candidates
        measured = measure(configs, target)       # benchmark on real hardware
        cost_model.update(configs, measured)      # update model with results
        history.record_best(configs, measured)   # track best seen so far

    Final step: load best config, rebuild with it.

    This process takes 2–4 hours for a full ResNet-50 on a GPU.
    But the resulting schedule often beats cuDNN by 5–15%.

### AutoTVM Limitations

    1. REQUIRES TEMPLATES: a programmer must write the schedule template
       defining which parameters to search. Not all operators have templates.

    2. OPERATOR INDEPENDENCE: AutoTVM tunes each operator independently.
       It doesn't consider cross-operator effects (cache pollution, etc.).

    3. SLOW: with 50+ operators in a network, tuning takes many hours.

    4. HARDWARE-SPECIFIC: tuning results don't transfer across hardware.
       Tune on RTX 3090 → logs are ONLY valid for RTX 3090.


##### PART 5 — AUTO-SCHEDULING: ANSOR (TEMPLATE-FREE SEARCH)

### Ansor's Breakthrough

    Ansor (2020) eliminates the need for human-written templates.
    Instead, it automatically generates the search space from the
    operator's computational definition (TE compute definition only).

    Input to Ansor: just the TE compute definition (no schedule template).
    Output: an optimised schedule found by search.

    How Ansor generates the search space:
        1. From the TE compute graph, enumerate all "sketch" structures.
           A sketch is a high-level schedule structure (which loops to tile,
           which loops to fuse, where to put cache buffers).
        2. Fill in each sketch with concrete parameters (tile sizes, etc.).
        3. This creates a hierarchical search space: discrete (sketch choice)
           + continuous (parameter values).

    Sketches for matrix multiply (partial list):
        - Tile i and j, vectorise j_inner, parallel i_outer
        - Tile i, j, k, use cache_write for output
        - Tile for GPU: bind i_outer→blockIdx, j_inner→threadIdx
        - Tile + shared memory + register tiles

### Ansor's Search Algorithm

    Ansor uses evolutionary search + neural network cost model:

        1. SKETCH GENERATION: enumerate valid sketch structures.
        2. POPULATION INIT: randomly sample concrete parameters for each sketch.
        3. EVOLUTIONARY MUTATION:
               mutate tile sizes (×2 or ÷2)
               swap two dimensions
               change vectorisation factor
        4. NEURAL COST MODEL: predict performance of each candidate.
           Trained on measurements, updated after each round.
        5. SELECT & MEASURE: take top predictions, measure on hardware.
        6. UPDATE: add measured results to training data, iterate.

    Key advantage over AutoTVM:
        No templates → finds schedules humans wouldn't think to write.
        Ansor discovered novel schedules for attention operators that
        outperformed hand-written kernels by 20–30%.

### Meta-Schedule (TVM Unity)

    TVM's next generation (TVM Unity, 2022+):
        Meta-Schedule generalises both AutoTVM and Ansor.
        Unified scheduling infrastructure for CPU, GPU, FPGA.
        Python-based schedule rules (more hackable than C++ backends).
        Supports Tensor Core (WMMA/MMA) operations natively.
        Better integration with MLIR and Relax (TVM's next-gen Relay).


##### PART 6 — TIR: TENSOR IR, THE LOOP-LEVEL IR

### What TIR Is

    TIR (Tensor IR) is TVM's low-level IR, positioned between the schedule
    and the final code generation (LLVM IR, CUDA PTX, etc.).

    TIR represents:
        - Explicit loop nests (for loops with bounds and step)
        - Buffer allocations (with shape, dtype, scope)
        - Explicit memory reads/writes (BufferLoad, BufferStore)
        - Data layout transformations (physical indexing)
        - Threading annotations (GPU thread/block bindings)
        - Atomic operations (for reductions)

    TIR is the "editable loop nest" before final code generation.
    Auto-scheduling operates by transforming the TIR loop structure.

### TIR Example: ReLU

    TE definition:
        B = te.compute(A.shape, lambda *i: te.max(A(*i), 0), name='relu')

    After schedule lowering to TIR (before LLVM codegen):

        # from tvm.ir.printer import astext
        @T.prim_func
        def relu(A: T.Buffer[(8,), "float32"],
                  B: T.Buffer[(8,), "float32"]):
            for i in T.serial(8):                   ; simple loop (scalar)
                B[i] = T.max(A[i], T.float32(0))

    After vectorise(i_inner=4) schedule:

        @T.prim_func
        def relu_vectorised(A: T.Buffer[(8,), "float32"],
                             B: T.Buffer[(8,), "float32"]):
            for i_outer in T.serial(2):             ; 2 outer iterations
                for i_inner in T.vectorized(4):     ; 4 vectorised inner
                    B[i_outer*4 + i_inner] = T.max(A[i_outer*4 + i_inner], 0.0)
        ; T.vectorized → LLVM loop-vectorize hint → <4 x float> maxps instruction

    After GPU bind schedule:

        @T.prim_func
        def relu_gpu(A: T.Buffer[(65536,), "float32"],
                      B: T.Buffer[(65536,), "float32"]):
            for i_outer in T.thread_binding(256, thread="blockIdx.x"):
                for i_inner in T.thread_binding(256, thread="threadIdx.x"):
                    B[i_outer*256 + i_inner] = T.max(A[i_outer*256 + i_inner], 0.0)
        ; Generates CUDA kernel with 256 blocks × 256 threads = 65536 parallel threads

### TIR Passes

    TVM's TIR pass pipeline (runs after scheduling, before codegen):

        StorageFlatten:     flatten multi-dim buffers to 1D (for C/CUDA arrays)
        LowerThreadAllreduce: lower parallel reductions to atomic ops
        InjectVirtualThread: virtualise parallel threads (for SIMD)
        VectorizeLoop:      convert vectorize annotations to vector types
        UnrollLoop:         unroll annotated loops
        Simplify:           constant folding, dead code, algebraic simplification
        LowerTVMBuiltin:    replace TVM built-ins with hardware intrinsics
        LowerIntrin:        lower intrinsics to target-specific calls


##### PART 7 — TVM TARGETS: DEPLOYING TO ANY HARDWARE

### Supported Targets

    CPU targets (via LLVM):
        x86-64:          "llvm -mcpu=core-avx2"   (AVX2 SIMD)
        x86-64 Skylake:  "llvm -mcpu=cascadelake"  (AVX-512)
        ARM Cortex-A:    "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
        ARM Cortex-M:    "c -keys=arm_cpu -mcpu=cortex-m7" (bare metal!)
        Apple M-series:  "llvm -mcpu=apple-m1"     (ARM + AMX tile engine)
        RISC-V:          "llvm -mtriple=riscv64-linux-gnu -mattr=+v"

    GPU targets:
        NVIDIA:     "cuda -arch=sm_80"   (Ampere A100)
        AMD:        "rocm -arch=gfx908"  (MI100)
        Apple GPU:  "metal"              (M1/M2 GPU, via Metal API)
        Qualcomm:   "opencl -device=adreno"  (Snapdragon Adreno GPU)
        ARM Mali:   "opencl -device=mali"    (Mali GPU, Android)

    Specialised accelerators:
        Hexagon:    Qualcomm DSP ("hexagon")
        MicroTVM:   bare-metal MCU without OS ("c --system-lib")
        FPGA:       via HLS (experimental)

### The Runtime and Module System

    Compiled TVM modules (.so / .tar) contain:
        - The optimised kernel functions (as native code)
        - Metadata (input/output shapes, dtypes)
        - Runtime dispatch logic

    Cross-compilation for edge:
        # Compile on x86 server, deploy to Raspberry Pi ARM
        target = tvm.target.Target("llvm -mtriple=armv7l-linux-gnueabihf")
        lib    = relay.build(mod, target=target, params=params)
        lib.export_library("resnet50_pi.tar")
        # Copy resnet50_pi.tar to Raspberry Pi, load with tvm.runtime

    MicroTVM for bare-metal:
        target  = tvm.target.Target("c --system-lib")
        runtime = tvm.runtime.create(executor="aot")  # ahead-of-time
        # Generates standalone C code + compiled .a library
        # No OS, no Python, no TVM runtime — just a C function call

### RPC: Remote Procedure Call for Device Benchmarking

    TVM's RPC system enables:
        - Compiling on a powerful x86 server
        - Benchmarking on a remote edge device (phone, Pi, TPU, FPGA)
        - Auto-tuning using the edge device as the measurement target

    Setup:
        # On the edge device:
        python -m tvm.exec.rpc_server --host=0.0.0.0 --port=9090

        # On the server (where you run TVM compilation):
        remote = rpc.connect("device_ip", 9090)
        target = tvm.target.arm_cpu("rasp4b")
        runner = autotvm.RPCRunner(target, remote)
        # AutoTVM now measures on the actual device over the network

    This is how TVM achieves hardware-specific optimisation without
    needing to run the compilation on the target device itself.


##### PART 8 — TVM IN THE CONNECTED COMPILER STACK

### How TVM Connects to LLVM (module 09)

    TVM's CPU backend generates LLVM IR from TIR:
        TIR loop nests → LLVM IR scalar loops
        LLVM loop-vectorize → SIMD instructions (SSE/AVX/NEON)
        LLVM instcombine → algebraic simplification
        LLVM LICM → loop-invariant code motion

    TVM uses LLVM's:
        - Target machine for platform detection (AVX2? AVX-512? NEON?)
        - JIT compiler (MCJIT) for auto-tuning candidate evaluation
        - Code emission for final compiled libraries (.so)

    TVM adds above LLVM:
        - ML operator schedule space (tile, vectorise, parallelise)
        - Auto-scheduling (Ansor) to find optimal tile sizes
        - Multi-stage compute pipelines (cache staging for GEMM)
        - Tensor Core support (not in LLVM — TVM emits WMMA intrinsics)

### How TVM Connects to MLIR (module 10)

    TVM and MLIR are converging in TVM Unity (2023+):

    Import path: StableHLO → Relax (TVM's next-gen IR) → TIR → code
    This means JAX/XLA-exported models can be compiled by TVM directly.

    TVM's Relax dialect (vs Relay):
        Like Relay but with dynamic shapes support.
        Designed to be more interoperable with MLIR dialects.
        Uses MLIR-like structural types.

    Linalg → TIR path:
        MLIR's Linalg ops (from the MLIR module) can be imported to TIR.
        TVM can then apply auto-scheduling to MLIR-expressed operators.

### How TVM Compares to XLA (module 11)

    ┌────────────────────────────────────────────────────────────────────┐
    │ Dimension         │ XLA                    │ TVM                  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Primary user      │ JAX, TensorFlow        │ Any framework (ONNX) │
    │ Optimisation      │ Analytical transforms  │ Search + measurement │
    │ Schedule          │ Deterministic          │ Auto-tuned           │
    │ Compile time      │ Seconds–minutes        │ Hours (first tune)   │
    │ Inference speed   │ Excellent (cuDNN)      │ Often beats cuDNN    │
    │ Hardware support  │ CPU, GPU, TPU          │ 30+ targets          │
    │ MLIR integration  │ StableHLO dialect      │ Relax + TIR          │
    │ Custom hardware   │ Hard (needs XLA port)  │ Easy (write TE + cfg)│
    │ TPU support       │ Native (best)          │ Experimental         │
    │ Edge/MCU          │ Via TFLite (indirect)  │ MicroTVM (direct)    │
    └────────────────────────────────────────────────────────────────────┘

### The Complete Compiler Stack (All Four Modules)

    ┌──────────────────────────────────────────────────────────────────┐
    │  LLVM (module 09): Universal CPU/GPU IR + optimisation passes    │
    │    JIT, SIMD auto-vectorise, loop optimise, code emit            │
    ├──────────────────────────────────────────────────────────────────┤
    │  MLIR (module 10): Multi-level dialect framework                 │
    │    StableHLO → Linalg → Affine → LLVM dialect                   │
    │    Infrastructure: pass manager, pattern rewriting, interfaces   │
    ├──────────────────────────────────────────────────────────────────┤
    │  XLA (module 11): Tensor compiler for JAX/TF/TPU                 │
    │    Analytical fusion, layout assign, SPMD, StableHLO export      │
    ├──────────────────────────────────────────────────────────────────┤
    │  TVM (module 12): Auto-scheduled ML compiler for all hardware    │
    │    Relay/Relax IR → TE schedules → TIR → Ansor search            │
    ├──────────────────────────────────────────────────────────────────┤
    │  Framework (other modules): PyTorch, JAX, TF, MXNet              │
    │  Deployment (other modules): TRT-LLM, vLLM, SGLang               │
    └──────────────────────────────────────────────────────────────────┘

    All four compiler-stack modules share LLVM as their CPU backend.
    All four are converging on StableHLO/MLIR as the interchange format.
    The boundaries are blurring: XLA uses MLIR, TVM imports StableHLO,
    MLIR outputs LLVM IR, LLVM JIT powers TVM auto-tuning.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Relay & TE — Import, Compile and Run a Model with TVM": {
        "description": (
            "End-to-end TVM compilation from ONNX or PyTorch to native code. "
            "Import a model into Relay IR and inspect the graph. "
            "Define a Tensor Expression kernel manually (relu, matmul). "
            "Build and benchmark against NumPy/PyTorch. "
            "Show the Relay → TIR lowering with schedule primitives."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  RELAY & TE — IMPORT, COMPILE AND RUN A MODEL WITH TVM")
print("=" * 65)
print()

try:
    import tvm
    from tvm import relay, te
    from tvm.relay import transform
    import tvm.runtime as runtime
    print(f"  Apache TVM version: {tvm.__version__}")
    HAS_TVM = True
except ImportError:
    HAS_TVM = False
    print("  TVM not installed.")
    print("  Install: pip install apache-tvm")
    print("  Or build from source: https://tvm.apache.org/docs/install")
    print()

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Relay IR — building and inspecting a graph manually
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Building and inspecting a Relay IR graph")
print("━" * 65)
print()

if HAS_TVM:
    # Build a simple 2-layer MLP directly in Relay
    dtype   = "float32"
    batch   = 4
    in_dim  = 16
    hid_dim = 32
    out_dim = 8

    # Inputs
    data = relay.var("data", shape=(batch, in_dim),   dtype=dtype)
    w1   = relay.var("w1",   shape=(hid_dim, in_dim), dtype=dtype)
    b1   = relay.var("b1",   shape=(hid_dim,),        dtype=dtype)
    w2   = relay.var("w2",   shape=(out_dim, hid_dim),dtype=dtype)
    b2   = relay.var("b2",   shape=(out_dim,),        dtype=dtype)

    # Build computation graph
    h1   = relay.nn.bias_add(relay.nn.dense(data, w1), b1)   # linear
    h1   = relay.nn.relu(h1)                                   # relu
    out  = relay.nn.bias_add(relay.nn.dense(h1, w2), b2)      # linear

    # Create a Relay module (the program)
    func   = relay.Function([data, w1, b1, w2, b2], out)
    module = tvm.IRModule({"main": func})

    print("  Relay module (2-layer MLP):")
    print()
    print(module)
    print()

    # Run Relay optimisation passes
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),          # propagate tensor types
        relay.transform.FoldConstant(),        # fold any constant subgraphs
        relay.transform.EliminateCommonSubexpr(),  # CSE
        relay.transform.FuseOps(),             # fuse compatible ops
    ])
    with tvm.transform.PassContext(opt_level=3):
        module_opt = seq(module)

    print("  After optimisation passes:")
    print(module_opt)
    print()

    # Compile for CPU
    target = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(module_opt, target=target)

    # Create runtime and run
    dev   = tvm.cpu(0)
    m     = runtime.GraphModule(lib["default"](dev))

    # Random inputs
    np.random.seed(0)
    data_np = np.random.randn(batch, in_dim).astype(np.float32)
    w1_np   = np.random.randn(hid_dim, in_dim).astype(np.float32)  * 0.1
    b1_np   = np.zeros(hid_dim, dtype=np.float32)
    w2_np   = np.random.randn(out_dim, hid_dim).astype(np.float32) * 0.1
    b2_np   = np.zeros(out_dim, dtype=np.float32)

    m.set_input("data", data_np)
    m.set_input("w1", w1_np)
    m.set_input("b1", b1_np)
    m.set_input("w2", w2_np)
    m.set_input("b2", b2_np)
    m.run()
    tvm_out = m.get_output(0).numpy()

    # Verify against NumPy reference
    h_ref  = np.maximum(data_np @ w1_np.T + b1_np, 0)
    out_ref = h_ref @ w2_np.T + b2_np
    match   = np.allclose(tvm_out, out_ref, atol=1e-5)
    print(f"  TVM output matches NumPy: {match} ✅")
    print(f"  Output shape: {tvm_out.shape}")
    print(f"  Max abs diff: {np.max(np.abs(tvm_out - out_ref)):.2e}")
    print()

else:
    RELAY_REF = """
  RELAY IR REFERENCE (without TVM installed):

  import tvm
  from tvm import relay

  # Define computation graph in Relay
  data  = relay.var("data",  shape=(4, 16), dtype="float32")
  w1    = relay.var("w1",    shape=(32, 16), dtype="float32")
  b1    = relay.var("b1",    shape=(32,), dtype="float32")

  h     = relay.nn.bias_add(relay.nn.dense(data, w1), b1)
  h     = relay.nn.relu(h)
  func  = relay.Function([data, w1, b1], h)
  mod   = tvm.IRModule({"main": func})
  print(mod)   # prints the Relay IR text

  # Compile
  target = tvm.target.Target("llvm")   # CPU
  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target)

  # Run
  import tvm.runtime as rt
  m   = rt.GraphModule(lib["default"](tvm.cpu(0)))
  m.set_input("data", data_np)
  m.set_input("w1", w1_np)
  m.set_input("b1", b1_np)
  m.run()
  output = m.get_output(0).numpy()

  # Relay IR output looks like:
  # def @main(%data: Tensor[(4, 16), float32],
  #           %w1: Tensor[(32, 16), float32],
  #           %b1: Tensor[(32), float32]) {
  #   %0 = nn.dense(%data, %w1);
  #   %1 = nn.bias_add(%0, %b1);
  #   nn.relu(%1)
  # }
"""
    print(RELAY_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Tensor Expressions — defining and scheduling operators
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Tensor Expressions: relu and matrix multiply")
print("━" * 65)
print()

if HAS_TVM:
    # ── ReLU in TE ────────────────────────────────────────────────────────
    n   = te.var("n")
    A   = te.placeholder((n,), dtype="float32", name="A")
    B   = te.compute(
        (n,),
        lambda i: te.max(A[i], tvm.tir.const(0, "float32")),
        name="B"
    )

    # Default schedule (scalar, no vectorisation)
    s_scalar = te.create_schedule(B.op)
    func_scalar = tvm.build(s_scalar, [A, B], target="llvm", name="relu_scalar")

    # Vectorised schedule
    s_vec = te.create_schedule(B.op)
    i, = s_vec[B].op.axis
    i_outer, i_inner = s_vec[B].split(i, factor=8)
    s_vec[B].vectorize(i_inner)
    s_vec[B].parallel(i_outer)
    func_vec = tvm.build(s_vec, [A, B], target="llvm -mcpu=native", name="relu_vec")

    # Benchmark
    N_TEST  = 1_000_000
    a_arr   = np.random.randn(N_TEST).astype(np.float32)
    dev_cpu = tvm.cpu(0)
    a_tvm   = tvm.nd.array(a_arr, dev_cpu)
    b_tvm_s = tvm.nd.array(np.zeros(N_TEST, np.float32), dev_cpu)
    b_tvm_v = tvm.nd.array(np.zeros(N_TEST, np.float32), dev_cpu)

    REPS = 100
    eval_scalar = func_scalar.time_evaluator(func_scalar.entry_name, dev_cpu, number=REPS)
    eval_vec    = func_vec.time_evaluator(func_vec.entry_name,    dev_cpu, number=REPS)

    t_scalar = eval_scalar(a_tvm, b_tvm_s).mean * 1000
    t_vec    = eval_vec(a_tvm, b_tvm_v).mean * 1000
    t0       = time.perf_counter()
    for _ in range(REPS): np.maximum(a_arr, 0)
    t_np = (time.perf_counter() - t0) / REPS * 1000

    print(f"  ReLU (N={N_TEST:,}) — schedule comparison:")
    print(f"  {'Schedule':<25} | {'Time (ms)':>12} | {'Speedup':>9}")
    print(f"  {'─'*50}")
    print(f"  {'Scalar (no schedule)':<25} | {t_scalar:12.4f} | {'1.00×':>9}")
    print(f"  {'Vectorised + parallel':<25} | {t_vec:12.4f} | {t_scalar/t_vec:9.2f}×")
    print(f"  {'NumPy (reference)':<25} | {t_np:12.4f} | {t_scalar/t_np:9.2f}×")
    print()

    # ── Matmul in TE ──────────────────────────────────────────────────────
    M_, K_, N_ = 256, 256, 256
    A_mm = te.placeholder((M_, K_), dtype="float32", name="A")
    B_mm = te.placeholder((K_, N_), dtype="float32", name="B")
    k    = te.reduce_axis((0, K_), name="k")
    C_mm = te.compute(
        (M_, N_),
        lambda i, j: te.sum(A_mm[i, k] * B_mm[k, j], axis=k),
        name="C"
    )

    # Naive schedule
    s_mm = te.create_schedule(C_mm.op)

    # Tiled + vectorised schedule
    s_mm_t = te.create_schedule(C_mm.op)
    i, j = s_mm_t[C_mm].op.axis
    k_ax, = s_mm_t[C_mm].op.reduce_axis
    i_o, i_i = s_mm_t[C_mm].split(i, factor=32)
    j_o, j_i = s_mm_t[C_mm].split(j, factor=32)
    s_mm_t[C_mm].reorder(i_o, j_o, k_ax, i_i, j_i)
    s_mm_t[C_mm].vectorize(j_i)
    s_mm_t[C_mm].parallel(i_o)

    fn_naive  = tvm.build(s_mm,   [A_mm, B_mm, C_mm], target="llvm", name="matmul_naive")
    fn_tiled  = tvm.build(s_mm_t, [A_mm, B_mm, C_mm], target="llvm -mcpu=native", name="matmul_tiled")

    a_m = tvm.nd.array(np.random.randn(M_, K_).astype(np.float32), dev_cpu)
    b_m = tvm.nd.array(np.random.randn(K_, N_).astype(np.float32), dev_cpu)
    c_m = tvm.nd.array(np.zeros((M_, N_), np.float32), dev_cpu)

    REPS2 = 50
    ev_naive  = fn_naive.time_evaluator(fn_naive.entry_name,  dev_cpu, number=REPS2)
    ev_tiled  = fn_tiled.time_evaluator(fn_tiled.entry_name,  dev_cpu, number=REPS2)

    t_naive = ev_naive(a_m, b_m, c_m).mean * 1000
    t_tiled = ev_tiled(a_m, b_m, c_m).mean * 1000
    t0 = time.perf_counter()
    for _ in range(REPS2): np.dot(a_m.numpy(), b_m.numpy())
    t_np_mm = (time.perf_counter() - t0) / REPS2 * 1000

    print(f"  MatMul ({M_}×{K_} @ {K_}×{N_}):")
    print(f"  {'Schedule':<25} | {'Time (ms)':>12} | {'Speedup':>9}")
    print(f"  {'─'*50}")
    print(f"  {'Naive (i,j,k)':<25} | {t_naive:12.4f} | {'1.00×':>9}")
    print(f"  {'Tiled+vec+parallel':<25} | {t_tiled:12.4f} | {t_naive/t_tiled:9.2f}×")
    print(f"  {'NumPy BLAS':<25} | {t_np_mm:12.4f} | {t_naive/t_np_mm:9.2f}×")
    print()
    print("  Auto-scheduling (Ansor) typically finds schedules that match or")
    print("  beat NumPy BLAS by searching 1000+ candidate configurations.")

else:
    TE_REFERENCE = """
  TENSOR EXPRESSION REFERENCE (without TVM):

  from tvm import te
  import tvm

  # ── ReLU ──────────────────────────────────────────────────────────────
  n = te.var("n")
  A = te.placeholder((n,), dtype="float32", name="A")
  B = te.compute((n,),
      lambda i: te.max(A[i], tvm.tir.const(0, "float32")),
      name="B")

  # Schedule: split into tiles of 8, vectorise inner loop
  s = te.create_schedule(B.op)
  i, = s[B].op.axis
  i_outer, i_inner = s[B].split(i, factor=8)
  s[B].vectorize(i_inner)    # emit SIMD
  s[B].parallel(i_outer)     # multi-thread

  func = tvm.build(s, [A, B], target="llvm -mcpu=native", name="relu")

  # ── Matrix Multiply ────────────────────────────────────────────────────
  M, K, N = 256, 256, 256
  A = te.placeholder((M, K), dtype="float32", name="A")
  B = te.placeholder((K, N), dtype="float32", name="B")
  k = te.reduce_axis((0, K), name="k")   # summation axis
  C = te.compute((M, N),
      lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
      name="C")

  s = te.create_schedule(C.op)
  i, j = s[C].op.axis
  k,   = s[C].op.reduce_axis
  # Tile for cache locality
  i_o, i_i = s[C].split(i, factor=32)
  j_o, j_i = s[C].split(j, factor=32)
  s[C].reorder(i_o, j_o, k, i_i, j_i)
  s[C].vectorize(j_i)
  s[C].parallel(i_o)

  func = tvm.build(s, [A, B, C], target="llvm -mcpu=native")

  Schedule primitives summary:
    split(axis, factor)   → tile loop (creates cache-friendly access)
    reorder(*axes)        → change loop ordering (i,j,k → i,k,j)
    vectorize(axis)       → SIMD hint (emit <8 x f32> ops via LLVM)
    parallel(axis)        → multi-thread (OpenMP / pthreads)
    unroll(axis)          → loop unrolling (reduce branch overhead)
    bind(axis, thread)    → GPU: axis → threadIdx.x / blockIdx.y
    compute_at(stage, ax) → move computation into another loop
"""
    print(TE_REFERENCE)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · AutoTVM and Ansor — Auto-Scheduling in Practice": {
        "description": (
            "How TVM's auto-scheduling systems work and how to use them. "
            "AutoTVM: define a schedule template with knobs, run the search. "
            "Ansor: template-free search on a TE compute definition. "
            "Show the tuning log format and how to use recorded schedules. "
            "Simulate the search process and cost model with Python. "
            "Compare hand-scheduled vs auto-scheduled vs BLAS performance."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from collections import defaultdict

print("=" * 65)
print("  AUTOTVM AND ANSOR — AUTO-SCHEDULING IN PRACTICE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: AutoTVM — template-guided search
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — AutoTVM: template-guided search")
print("━" * 65)
print()

try:
    import tvm
    from tvm import te, autotvm
    from tvm import relay
    import tvm.runtime as runtime

    # ── Define a schedule template ────────────────────────────────────────
    @autotvm.template("matmul_template")
    def matmul_template(N, L, M, dtype="float32"):
        """
        AutoTVM template for matrix multiply.
        The @autotvm.template decorator marks this as searchable.
        ConfigSpace is defined by the knob declarations.
        """
        A = te.placeholder((N, L), dtype=dtype, name="A")
        B = te.placeholder((L, M), dtype=dtype, name="B")
        k = te.reduce_axis((0, L), name="k")
        C = te.compute((N, M),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            name="C")

        s = te.create_schedule(C.op)
        i, j = s[C].op.axis
        k,   = s[C].op.reduce_axis

        # ── KNOB DECLARATIONS — these are the search dimensions ────────────
        # split_factor is chosen from the ConfigSpace
        cfg = autotvm.get_config()

        cfg.define_split("tile_i", i, num_outputs=2,
                         filter=lambda entity: entity.size[-1] <= 64)
        cfg.define_split("tile_j", j, num_outputs=2,
                         filter=lambda entity: entity.size[-1] <= 64)
        cfg.define_knob("unroll_k", [1, 2, 4, 8])
        cfg.define_knob("vectorize_j", [True, False])

        # Apply the chosen configuration
        i_o, i_i = cfg["tile_i"].apply(s, C, i)
        j_o, j_i = cfg["tile_j"].apply(s, C, j)
        s[C].reorder(i_o, j_o, k, i_i, j_i)

        if cfg["vectorize_j"].val:
            s[C].vectorize(j_i)

        s[C].unroll(k)
        s[C].parallel(i_o)

        return s, [A, B, C]

    # Print the config space
    task = autotvm.task.create("matmul_template",
                               args=(128, 128, 128, "float32"),
                               target="llvm -mcpu=native")

    print("  AutoTVM search space for matmul (128×128×128):")
    print(f"    Config space size: {len(task.config_space):,} configurations")
    print(f"    Dimensions:")
    for k_name, space in task.config_space.space_map.items():
        print(f"      {k_name:15s}: {space}")
    print()

    # ── Simulate the search without actual hardware measurement ────────────
    print("  Simulating AutoTVM search (10 candidates, synthetic costs):")
    print()

    np.random.seed(42)

    def synthetic_cost(tile_i, tile_j, unroll_k, vectorize_j):
        """
        Synthetic cost model: approximates the real performance landscape.
        Real AutoTVM measures actual execution time on the target hardware.
        """
        # Larger tiles generally better for cache (up to L1 cache size)
        cache_benefit  = min(tile_i * tile_j, 256) / 256
        # Vectorisation helps when inner loop is wide enough
        vec_benefit    = 0.3 if vectorize_j and tile_j >= 8 else 0
        # Unrolling helps for small k strides
        unroll_benefit = 0.1 * min(unroll_k, 4) / 4
        # Total performance (higher = faster)
        perf = cache_benefit + vec_benefit + unroll_benefit
        # Add noise (real hardware has variance)
        return perf + np.random.normal(0, 0.05)

    # Sample 10 random configs and evaluate
    configs_sampled = []
    for trial in range(10):
        tile_i  = np.random.choice([8, 16, 32, 64])
        tile_j  = np.random.choice([8, 16, 32, 64])
        unroll  = np.random.choice([1, 2, 4, 8])
        vec_j   = np.random.choice([True, False])
        cost    = synthetic_cost(tile_i, tile_j, unroll, vec_j)
        configs_sampled.append({
            "tile_i": tile_i, "tile_j": tile_j,
            "unroll_k": unroll, "vectorize_j": vec_j,
            "cost": cost
        })
        print(f"  Trial {trial+1:2d}: tile_i={tile_i:2d}, tile_j={tile_j:2d}, "
              f"unroll={unroll}, vec={str(vec_j):<5} → cost={cost:.4f}")

    best = max(configs_sampled, key=lambda c: c["cost"])
    print()
    print(f"  Best config found:")
    for k, v in best.items():
        if k != "cost":
            print(f"    {k}: {v}")
    print(f"  Best cost: {best['cost']:.4f}")
    print()

    # ── Build with best config ─────────────────────────────────────────────
    print("  Building with the best found configuration...")
    dispatch_ctx = autotvm.apply_history_best.ApplyHistoryBest.__new__(
        autotvm.apply_history_best.ApplyHistoryBest)
    # (In real usage: autotvm.apply_history_best("tuning.log"))

    M_size = 256
    a_np = np.random.randn(M_size, M_size).astype(np.float32)
    b_np = np.random.randn(M_size, M_size).astype(np.float32)

    with autotvm.apply_history_best([]):    # no pre-tuned log — use defaults
        s_best, args = matmul_template(M_size, M_size, M_size)
        fn_best = tvm.build(s_best, args, target="llvm -mcpu=native")

    dev = tvm.cpu(0)
    a_t = tvm.nd.array(a_np, dev)
    b_t = tvm.nd.array(b_np, dev)
    c_t = tvm.nd.array(np.zeros((M_size, M_size), np.float32), dev)

    evaluator = fn_best.time_evaluator(fn_best.entry_name, dev, number=100)
    t_tvm_ms  = evaluator(a_t, b_t, c_t).mean * 1000

    t0 = time.perf_counter()
    for _ in range(100): np.dot(a_np, b_np)
    t_np_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"  MatMul {M_size}×{M_size}:")
    print(f"    TVM (template, no tuning):  {t_tvm_ms:.4f} ms")
    print(f"    NumPy BLAS:                 {t_np_ms:.4f} ms")
    print()

except ImportError:
    print("  TVM not installed — showing AutoTVM reference code")
    print()

    AUTOTVM_REF = """
  AUTOTVM REFERENCE CODE:

  import tvm
  from tvm import te, autotvm

  @autotvm.template("matmul_cpu")
  def matmul_cpu(N, L, M, dtype="float32"):
      A = te.placeholder((N, L), dtype=dtype, name="A")
      B = te.placeholder((L, M), dtype=dtype, name="B")
      k = te.reduce_axis((0, L), name="k")
      C = te.compute((N, M), lambda i,j: te.sum(A[i,k]*B[k,j], axis=k), name="C")

      s = te.create_schedule(C.op)
      i, j = s[C].op.axis; k, = s[C].op.reduce_axis
      cfg = autotvm.get_config()

      # Define the search space (knobs)
      cfg.define_split("tile_i", i, num_outputs=2)
      cfg.define_split("tile_j", j, num_outputs=2)
      cfg.define_knob("unroll_k", [1, 4, 8])

      # Apply the config (values determined by search)
      i_o, i_i = cfg["tile_i"].apply(s, C, i)
      j_o, j_i = cfg["tile_j"].apply(s, C, j)
      s[C].reorder(i_o, j_o, k, i_i, j_i)
      s[C].vectorize(j_i)
      s[C].parallel(i_o)
      return s, [A, B, C]

  # Create tuning task
  task = autotvm.task.create("matmul_cpu",
                              args=(512, 512, 512, "float32"),
                              target="llvm")

  # Run tuning (measures on actual hardware)
  tuner = autotvm.tuner.XGBTuner(task)
  tuner.tune(
      n_trial=2000,
      measure_option=autotvm.measure_option(
          builder=autotvm.LocalBuilder(),
          runner=autotvm.LocalRunner(number=10)
      ),
      callbacks=[autotvm.callback.log_to_file("matmul.log")]
  )

  # Build with best config
  with autotvm.apply_history_best("matmul.log"):
      s, args = matmul_cpu(512, 512, 512)
      func = tvm.build(s, args, target="llvm -mcpu=native")

  # Tuning log entry format (JSON):
  # {"input": ["matmul_cpu", [512,512,512,"float32"]],
  #  "config": {"tile_i": [1,32], "tile_j": [1,16], "unroll_k": 4},
  #  "result": [[0.00142], 0, 2.3, 1234567890],  ← time in seconds
  #  "version": "v0.7"}
"""
    print(AUTOTVM_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Ansor — template-free search (simulated)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Ansor: template-free schedule search")
print("━" * 65)
print()

print("  Ansor vs AutoTVM:")
print()
print("  AutoTVM: programmer writes schedule template, Ansor searches parameters")
print("  Ansor:   no template needed — generates search space from TE definition only")
print()

try:
    import tvm
    from tvm import te, auto_scheduler

    # Define a TE workload (no template — just the math)
    @auto_scheduler.register_workload
    def relu_workload(N, dtype="float32"):
        A = te.placeholder((N,), dtype=dtype, name="A")
        B = te.compute((N,), lambda i: te.max(A[i], tvm.tir.const(0, dtype)), name="B")
        return [A, B]

    @auto_scheduler.register_workload
    def matmul_workload(M, K, N, dtype="float32"):
        A = te.placeholder((M, K), dtype=dtype, name="A")
        B = te.placeholder((K, N), dtype=dtype, name="B")
        k = te.reduce_axis((0, K), name="k")
        C = te.compute((M, N), lambda i, j: te.sum(A[i,k]*B[k,j], axis=k), name="C")
        return [A, B, C]

    target = tvm.target.Target("llvm -mcpu=native")
    N_mm   = 128

    task = auto_scheduler.SearchTask(
        func=matmul_workload, args=(N_mm, N_mm, N_mm, "float32"),
        target=target
    )

    print(f"  Ansor search task: matmul ({N_mm}×{N_mm}×{N_mm})")
    print(f"  Task workload key: {task.workload_key}")
    print()

    # Instead of full tuning (hours), show the sketch generation
    dag     = task.compute_dag
    print(f"  ComputeDAG stages: {len(dag.ops)} ops")
    for op in dag.ops:
        print(f"    {op.name}: {op}")
    print()

    # Run a SHORT tuning to show the mechanism
    log_file = "/tmp/ansor_demo.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20,          # in practice: 1000+
        runner=auto_scheduler.LocalRunner(repeat=3, min_repeat_ms=100),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )

    print("  Running Ansor search (20 trials — demonstration only)...")
    print("  In production: set num_measure_trials=1000–5000")
    print()

    t0 = time.perf_counter()
    task.tune(tune_option)
    t_tune = time.perf_counter() - t0
    print(f"  Search completed in {t_tune:.1f}s")

    # Load best schedule and build
    sch, args = task.apply_best(log_file)
    fn_ansor  = tvm.build(sch, args, target=target)

    dev  = tvm.cpu(0)
    a_nn = tvm.nd.array(np.random.randn(N_mm, N_mm).astype(np.float32), dev)
    b_nn = tvm.nd.array(np.random.randn(N_mm, N_mm).astype(np.float32), dev)
    c_nn = tvm.nd.array(np.zeros((N_mm, N_mm), np.float32), dev)

    ev_a = fn_ansor.time_evaluator(fn_ansor.entry_name, dev, number=200)
    t_ansor = ev_a(a_nn, b_nn, c_nn).mean * 1000
    print(f"  Ansor-tuned matmul {N_mm}×{N_mm}: {t_ansor:.4f} ms")
    print()
    print("  With full tuning (1000+ trials), Ansor finds schedules that")
    print("  often EXCEED hand-written cuDNN/BLAS performance by 10-30%.")

except Exception as e:
    print(f"  Ansor demo: {e}")
    print()

    ANSOR_REF = """
  ANSOR REFERENCE CODE (full workflow):

  from tvm import auto_scheduler

  @auto_scheduler.register_workload
  def conv2d_workload(N, H, W, CO, CI, KH, KW, dtype="float32"):
      data   = te.placeholder((N, CI, H, W), dtype=dtype, name="data")
      kernel = te.placeholder((CO, CI, KH, KW), dtype=dtype, name="kernel")
      # ... compute definition (no schedule template!)
      return [data, kernel, output]

  task = auto_scheduler.SearchTask(
      func=conv2d_workload, args=(1, 56, 56, 64, 64, 3, 3),
      target=tvm.target.Target("cuda")   # or "llvm", "opencl", etc.
  )

  # Configure search
  tune_option = auto_scheduler.TuningOptions(
      num_measure_trials=2000,   # how many configs to try
      runner=auto_scheduler.LocalRunner(
          repeat=3, min_repeat_ms=150, timeout=10
      ),
      measure_callbacks=[auto_scheduler.RecordToFile("ansor_conv2d.json")],
      verbose=1,
  )

  # Search — generates sketches, fills parameters, measures, updates model
  task.tune(tune_option)

  # Build with best schedule found
  sch, args = task.apply_best("ansor_conv2d.json")
  lib = tvm.build(sch, args, target="cuda")

  # Ansor search internally:
  # 1. Generate sketches (structural decisions): ~100 sketches for conv2d
  # 2. Fill parameters (tile sizes, cache levels): ~10 completions per sketch
  # 3. Apply evolutionary search: mutate, predict, measure
  # 4. Update neural cost model after every batch of measurements
  # 5. Total: 2000 trials × ~150ms = ~5 minutes for one conv2d layer
"""
    print(ANSOR_REF)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · TVM Deployment — ONNX Import, Cross-Compilation & the Full Stack": {
        "description": (
            "Production TVM: import from ONNX, compile for multiple targets, "
            "benchmark against framework dispatch, cross-compile for ARM. "
            "Show TIR lowering and how it connects to LLVM IR. "
            "MicroTVM reference for bare-metal MCU deployment. "
            "Complete connected-stack summary: TVM's place between LLVM, MLIR, XLA."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TVM DEPLOYMENT — ONNX, CROSS-COMPILE & THE FULL STACK")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Import from ONNX and compile for CPU
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — ONNX → Relay → TVM compile pipeline")
print("━" * 65)
print()

try:
    import tvm
    from tvm import relay
    import tvm.runtime as rt

    # Try to create a real ONNX model via PyTorch
    try:
        import torch
        import torch.nn as nn
        import io, onnx

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(32, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 10),
                )
            def forward(self, x): return self.features(x)

        model    = SimpleNet().eval()
        dummy    = torch.randn(1, 32)
        buf      = io.BytesIO()
        torch.onnx.export(model, dummy, buf, opset_version=13,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}})
        buf.seek(0)
        onnx_model = onnx.load(buf)
        print("  Created ONNX model from PyTorch ✅")
        has_model = True

    except ImportError:
        has_model = False
        print("  PyTorch/ONNX not available — using manually constructed Relay")

    if has_model:
        # Import to Relay
        shape_dict = {"input": (4, 32)}    # batch=4
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        print(f"  Imported to Relay: {len(mod.functions)} function(s)")
    else:
        # Manually construct Relay
        data = relay.var("input", shape=(4, 32), dtype="float32")
        w1   = relay.var("w1", shape=(64, 32))
        b1   = relay.var("b1", shape=(64,))
        h    = relay.nn.relu(relay.nn.bias_add(relay.nn.dense(data, w1), b1))
        w2   = relay.var("w2", shape=(10, 64))
        b2   = relay.var("b2", shape=(10,))
        out  = relay.nn.bias_add(relay.nn.dense(h, w2), b2)
        func = relay.Function([data, w1, b1, w2, b2], out)
        mod  = tvm.IRModule({"main": func})
        params = {}

    # ── Compile for different targets and benchmark ─────────────────────────
    targets = [
        ("llvm",                           "Generic x86 (no SIMD)"),
        ("llvm -mcpu=native",              "Native x86 (with SIMD)"),
    ]

    x_test = np.random.randn(4, 32).astype(np.float32)
    if not has_model:
        w1_np = np.random.randn(64, 32).astype(np.float32) * 0.1
        b1_np = np.zeros(64, dtype=np.float32)
        w2_np = np.random.randn(10, 64).astype(np.float32) * 0.1
        b2_np = np.zeros(10, dtype=np.float32)
        params = {"w1": w1_np, "b1": b1_np, "w2": w2_np, "b2": b2_np}

    print(f"\n  {'Target':<35} | {'Compile (s)':>12} | {'Infer (ms)':>12}")
    print(f"  {'─'*63}")

    for tgt_str, tgt_desc in targets:
        target = tvm.target.Target(tgt_str)
        t0 = time.perf_counter()
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        t_compile = time.perf_counter() - t0

        dev   = tvm.cpu(0)
        m     = rt.GraphModule(lib["default"](dev))
        m.set_input("input", x_test)
        ev    = m.module.time_evaluator("run", dev, number=500)
        t_ms  = ev().mean * 1000

        print(f"  {tgt_desc:<35} | {t_compile:12.3f} | {t_ms:12.4f}")

    print()

except ImportError as e:
    print(f"  TVM not available: {e}")
    ONNX_REF = """
  ONNX → TVM PIPELINE REFERENCE:

  import tvm, onnx
  from tvm import relay

  # Load ONNX model
  onnx_model = onnx.load("resnet50.onnx")

  # Import to Relay
  shape_dict = {"data": (1, 3, 224, 224)}
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  # Graph-level optimisations
  with tvm.transform.PassContext(opt_level=3):
      mod = relay.transform.InferType()(mod)
      mod = relay.transform.FuseOps()(mod)
      mod = relay.transform.EliminateCommonSubexpr()(mod)

  # Compile (with optional tuning log)
  target = tvm.target.Target("llvm -mcpu=cascadelake")
  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target, params=params)

  # Save compiled library
  lib.export_library("resnet50_tvm.so")

  # Load and run
  loaded = tvm.runtime.load_module("resnet50_tvm.so")
  m = tvm.runtime.GraphModule(loaded["default"](tvm.cpu()))
  m.set_input("data", input_data)
  m.run()
  output = m.get_output(0).numpy()
"""
    print(ONNX_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: TIR inspection — the loop-level IR
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — TIR: inspecting the loop-level intermediate repr")
print("━" * 65)
print()

try:
    import tvm
    from tvm import te
    from tvm.script import tir as T

    # ── Build a simple TE and lower to TIR ───────────────────────────────
    n   = te.var("n")
    A   = te.placeholder((n,), dtype="float32", name="A")
    B   = te.compute((n,), lambda i: te.max(A[i], tvm.tir.const(0, "float32")), name="B")

    s_plain = te.create_schedule(B.op)
    func_plain = tvm.lower(s_plain, [A, B], name="relu")

    print("  TIR for scalar relu (before vectorisation):")
    print()
    print(func_plain.script())
    print()

    # With vectorisation
    s_vec = te.create_schedule(B.op)
    i, = s_vec[B].op.axis
    i_o, i_i = s_vec[B].split(i, factor=4)
    s_vec[B].vectorize(i_i)
    func_vec = tvm.lower(s_vec, [A, B], name="relu_vectorised")

    print("  TIR for vectorised relu (factor=4):")
    print()
    print(func_vec.script())
    print()

    print("  Key TIR annotations:")
    print("    T.serial(n):       plain loop (sequential)")
    print("    T.vectorized(4):   inner loop → SIMD instruction hint")
    print("    T.parallel(n):     outer loop → multi-thread")
    print("    T.unroll(8):       loop unrolling")
    print("    T.thread_binding:  GPU thread/block binding")
    print()
    print("  TIR connects to LLVM: vectorized(4) → <4 x float> type in LLVM IR")
    print("  LLVM's backend then selects the appropriate SIMD instruction:")
    print("    x86:  movaps/maxps (SSE) or vmaxps (AVX)")
    print("    ARM:  vmaxq_f32 (NEON)")
    print("    RISC-V: vfmax.vv (RVV)")

except ImportError:
    TIR_REF = """
  TIR (TENSOR IR) REFERENCE:

  Lower TE to TIR for inspection:
    func = tvm.lower(schedule, [A, B, C], name="matmul")
    print(func.script())

  TIR output (simplified):
    @T.prim_func
    def matmul(A: T.Buffer[(M, K), "float32"],
               B: T.Buffer[(K, N), "float32"],
               C: T.Buffer[(M, N), "float32"]):
        for i in T.parallel(M // 32):        ; parallelised outer
            for j in T.serial(N // 32):      ; tiled outer j
                for k in T.serial(K):        ; reduction axis
                    for ii in T.serial(32):  ; inner tile i
                        for jj in T.vectorized(32):  ; SIMD inner j
                            C[i*32+ii, j*32+jj] += A[i*32+ii, k] * B[k, j*32+jj]

  TIR primitives → final code:
    T.vectorized(n)  → <n x float32> in LLVM IR → vaddps/vmulps (AVX2)
    T.parallel(n)    → OpenMP #pragma omp parallel for
    T.thread_binding → CUDA __global__ kernel launch parameters
    T.unroll(n)      → LLVM unroll metadata → #pragma unroll
"""
    print(TIR_REF)
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Cross-compilation and MicroTVM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Cross-compilation and MicroTVM")
print("━" * 65)
print()

CROSS_COMPILE = """
  TVM CROSS-COMPILATION (build on x86, run on ARM):

  # On development machine (x86 Linux):
  import tvm
  from tvm import relay

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  # Compile for Raspberry Pi 4 (ARMv8)
  target = tvm.target.Target(
      "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72 -mattr=+neon"
  )
  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target, params=params)

  # Export as cross-compiled shared library
  lib.export_library("model_rpi4.tar")   ; .tar contains .so + metadata

  # Transfer to Raspberry Pi (e.g., scp model_rpi4.tar pi@192.168.1.x:~/)

  # On Raspberry Pi:
  import tvm.runtime as rt
  import tvm
  loaded = tvm.runtime.load_module("model_rpi4.tar")
  m = rt.GraphModule(loaded["default"](tvm.cpu()))
  m.set_input("input", input_data)
  m.run()
  print(m.get_output(0).numpy())


  TVM CROSS-COMPILATION TARGETS:

  Platform          Target string
  ──────────────────────────────────────────────────────────────────────
  Raspberry Pi 4    "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72"
  Android ARM64     "llvm -mtriple=aarch64-linux-android -mcpu=generic"
  STM32 (Cortex-M)  "c --runtime=c --system-lib"  (MicroTVM)
  RISC-V Linux      "llvm -mtriple=riscv64-linux-gnu -mattr=+v"
  Apple M1          "llvm -mcpu=apple-m1"  (run on same machine)
  Qualcomm Adreno   "opencl -device=adreno"  (Android GPU)
  ARM Mali GPU      "opencl -device=mali"    (Android/embedded GPU)
"""
print(CROSS_COMPILE)

MICROTVM = """
  MICROTVM — BARE-METAL DEPLOYMENT (no OS, no Python):

  MicroTVM compiles models to standalone C code that runs on microcontrollers:
  STM32, Arduino, ESP32, Nordic nRF, Zephyr RTOS, FreeRTOS.

  # Compile to C for STM32 Cortex-M4
  from tvm.micro import build_utils

  target = tvm.target.Target("c --system-lib --runtime=c")
  with tvm.transform.PassContext(opt_level=3,
      config={"tir.disable_vectorize": True}):   ; MCU has no SIMD
      lib = relay.build(mod, target=target, params=params)

  # Generate C source files
  lib.export_library("model.tar")
  # Contains:
  #   model.c         (generated kernel code — pure C, no TVM runtime)
  #   model.h         (function declarations)
  #   runtime.h       (minimal TVM micro runtime, ~100 KB)

  # MCU application code:
  // model.h
  int model_run(float* input, float* output);

  // main.c on STM32:
  #include "model.h"
  float input[32], output[10];
  // ... fill input ...
  model_run(input, output);
  // argmax(output) → predicted class

  Why MicroTVM matters:
  - 1 billion+ microcontrollers deployed per year
  - Keyword detection, gesture recognition, anomaly detection
  - No Python, no OS, no TVM runtime on device
  - Just a compiled C function
"""
print(MICROTVM)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: The connected stack — final summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — The complete connected compiler stack")
print("━" * 65)
print()

CONNECTED_STACK = """
  THE COMPLETE ML COMPILER STACK (all four modules connected)

  ┌──────────────────────────────────────────────────────────────────────┐
  │                 USER WRITES PYTHON MODEL CODE                        │
  │   PyTorch / JAX / TF / MXNet / Keras (framework modules 06–08)      │
  └───────────────────────────┬──────────────────────────────────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
  │  XLA (module 11) │ │  TVM (module 12) │ │  TRT-LLM/vLLM/SGLang     │
  │                  │ │                  │ │  (deployment modules)     │
  │  HLO fusion      │ │  Relay/Relax IR  │ │  Specialised LLM         │
  │  Layout assign   │ │  TE + Schedules  │ │  inference runtimes      │
  │  SPMD/pjit       │ │  Auto-scheduling │ │  (PagedAttn, KV cache)   │
  │  TPU native      │ │  30+ targets     │ └──────────────────────────┘
  └────────┬─────────┘ └───────┬──────────┘
           │                   │
           └─────────┬─────────┘
                     ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │              MLIR (module 10)                                        │
  │  StableHLO / TOSA / Linalg / Affine / SCF / MemRef / LLVM dialects  │
  │  Pass manager, pattern rewriting, interface system                   │
  │  Progressive lowering from ML semantics to loop-level IR            │
  └───────────────────────────┬──────────────────────────────────────────┘
                              │  mlir-translate → LLVM IR
                              ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │              LLVM (module 09)                                        │
  │  Universal IR (SSA form), pass pipeline, backend code generation     │
  │  loop-vectorize → SIMD  ·  instcombine  ·  LICM  ·  JIT compiler   │
  └───────────────────────────┬──────────────────────────────────────────┘
                              │
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                     ▼
  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
  │  x86-64      │  │  ARM64 / ARM32   │  │  NVIDIA CUDA             │
  │  AVX-512     │  │  NEON / SVE      │  │  NVPTX → PTX → cubin     │
  │  (Intel/AMD) │  │  (mobile/server) │  │  cuBLAS, cuDNN calls     │
  └──────────────┘  └──────────────────┘  └──────────────────────────┘
          ▼                   ▼                     ▼
  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
  │  RISC-V      │  │  WebAssembly     │  │  Custom accelerators     │
  │  RVV vectors │  │  (browser/edge)  │  │  TPU, Hexagon, FPGA      │
  └──────────────┘  └──────────────────┘  └──────────────────────────┘

  CONVERGENCE POINT: StableHLO (MLIR dialect)
    JAX exports → StableHLO
    PyTorch/torch-mlir → StableHLO
    XLA reads → StableHLO (its new primary IR)
    TVM imports → StableHLO (via Relax)
    IREE compiles → StableHLO

  StableHLO is becoming the universal model exchange format,
  analogous to what ONNX tried to be but with compiler infrastructure.
"""
print(CONNECTED_STACK)

print("  TVM DECISION GUIDE — when to use TVM vs alternatives:")
print()
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ Situation                    │ Best choice                   │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ JAX + GPU/TPU                │ XLA (native, always compiled) │")
print("  │ PyTorch + NVIDIA GPU         │ torch.compile + TensorRT      │")
print("  │ Any model + ANY hardware     │ TVM (widest target support)   │")
print("  │ Custom ASIC / novel chip     │ TVM (write TE + target config)│")
print("  │ Bare-metal MCU               │ MicroTVM                      │")
print("  │ Beat cuDNN on NVIDIA         │ TVM Ansor (auto-schedule)     │")
print("  │ Fast iteration, no tuning    │ XLA or TorchInductor          │")
print("  │ Cross-compile to ARM/RISC-V  │ TVM (RPC remote tuning)       │")
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