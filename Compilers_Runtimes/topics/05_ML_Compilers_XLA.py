"""
XLA — Accelerated Linear Algebra: Google's Tensor Compiler
===========================================================

XLA (Accelerated Linear Algebra) is Google's domain-specific compiler for
linear algebra computations. It sits one level above LLVM and MLIR in the
compiler stack, operating on tensor programs expressed in HLO (High Level
Operations) and compiling them to highly optimised native code for CPUs,
GPUs, and TPUs.

XLA is not an optional optimisation — it is the mandatory compilation
layer for JAX (which compiles everything through XLA), and the recommended
acceleration path for TensorFlow. Every JAX program you write is compiled
to XLA HLO before any computation executes. Every jax.jit decorator is an
instruction to XLA: trace, optimise, and compile this function.

The central insight of XLA: ML models are not general programs. They are
computations on fixed-shape tensors with known data types, predictable
memory access patterns, and a finite vocabulary of operations. This
restriction is XLA's superpower — it enables the compiler to know things
that a general-purpose compiler like LLVM cannot:

    Which operations fuse into one GPU kernel (no intermediate allocation).
    The exact memory footprint at every point in the computation.
    The optimal layout for every tensor given the target hardware.
    How to partition computation across 1024 TPU cores.

In the connected stack:
    LLVM (module 09)  ← XLA uses LLVM as its CPU backend
    MLIR (module 10)  ← XLA adopts StableHLO/MHLO as its primary IR
    XLA  (this module)← the compiler for JAX, TF, and TPUs
    TVM  (module 12)  ← alternative ML compiler, converging via StableHLO

"""

import textwrap
import re

TOPIC_NAME   = "XLA — Accelerated Linear Algebra Compiler"
DISPLAY_NAME = "05 · XLA"
ICON         = "⚡"
SUBTITLE     = "HLO Fusion, TPU Compilation, and JAX's Execution Engine"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT XLA IS AND WHERE IT SITS IN THE STACK

### XLA's Origin and Role

    XLA was created at Google in 2016, initially to accelerate TensorFlow
    computations on internal hardware (TPUs). It has since become the
    exclusive compilation backend for JAX and an important path for TF.

    The problem XLA solves:
        TensorFlow's runtime dispatched each operation separately:
            tf.matmul → launch CUDA kernel
            tf.bias_add → launch another CUDA kernel
            tf.relu → launch another CUDA kernel

        Three kernel launches, three memory round-trips (read weights,
        read bias, read activations — write results — repeat).

        XLA instead compiles the ENTIRE computation graph:
            - Fuses matmul + bias_add + relu into ONE kernel
            - No intermediate memory allocation between fused ops
            - Kernel launch overhead: 1 instead of 3
            - Memory bandwidth: read once, write once

    This fusion-first approach is XLA's core contribution.

### XLA's Position in the ML Compiler Stack

    Framework level (Python):
        JAX:           jax.jit(f)(x)  → ALWAYS goes through XLA
        TensorFlow:    tf.function + XLA: @tf.function(jit_compile=True)
        PyTorch:       torch.compile with XLA backend (experimental)

    XLA level (this module):
        HLO IR:            the computation graph representation
        HLO passes:        algebraic simplifications, fusion, layout
        Backend codegen:   CPU (LLVM), GPU (NVPTX/AMDGCN), TPU (proprietary)

    Lower levels:
        LLVM (module 09):  XLA's CPU backend emits LLVM IR
        MLIR (module 10):  XLA adopts StableHLO/MHLO as its MLIR-based IR

    ┌────────────────────────────────────────────────────────────────┐
    │  JAX / TF program (Python)                                     │
    │        ↓  jax.jit / @tf.function(jit_compile=True)             │
    │  XLA HLO graph  ← this is what XLA compiles                    │
    │        ↓  HLO passes (fusion, layout, algebraic simplify)      │
    │  Optimised HLO                                                 │
    │        ↓  Backend codegen                                      │
    │  CPU: LLVM IR → LLVM → x86/ARM                                 │
    │  GPU: NVPTX → PTX → cubin  /  AMDGCN → GPU binary              │
    │  TPU: XLA HLO → LLO → TPU executable                           │
    └────────────────────────────────────────────────────────────────┘

### Key Consumers of XLA Today

    JAX:           Everything goes through XLA. No fallback.
                   jax.jit is jax.xla_computation + compile + cache.

    TensorFlow:    Optional via jit_compile=True on tf.function.
                   Google's TPU training always uses XLA.

    PyTorch/XLA:   torch_xla package runs PyTorch on TPUs via XLA.
                   Traces the PyTorch graph, compiles through XLA.

    Google internal: All Google production ML training (Search, Ads,
                     Translate, DeepMind) runs through XLA on TPUs.

    OpenXLA:       The open-source fork of XLA's compiler infrastructure.
                   Decoupled from TF/JAX to serve as shared ML compiler.


##### PART 2 — HLO: THE HIGH LEVEL OPERATIONS IR

### What HLO Is

    HLO (High Level Operations) is XLA's intermediate representation.
    It is a functional computation graph where:
        - Nodes are operations (add, dot, convolution, reduce, etc.)
        - Edges are tensors (values flowing between operations)
        - Shapes are FULLY KNOWN at compile time (static shapes)
        - Types are explicit (f32, f16, bf16, s8, u32, etc.)
        - No side effects — purely functional, no in-place mutation

    HLO's static shape requirement is both its constraint and its power:
        Constraint: dynamic shapes require recompilation (or padding).
        Power: the compiler knows EXACTLY how much memory each op uses,
               enabling perfect memory planning and buffer reuse.

    HLO is expressed as:
        1. HloModule:   the full program (one or more HloComputations)
        2. HloComputation: a function (one root instruction)
        3. HloInstruction: one operation (with shape, operands, params)

### HLO Operation Vocabulary

    XLA's ~150 HLO ops cover all of ML:

    Elementwise:
        add, subtract, multiply, divide, maximum, minimum
        exp, log, sqrt, rsqrt, tanh, sin, cos
        abs, negate, sign, floor, ceil, round
        compare (with direction: EQ, NE, LT, GT, LE, GE)
        select (ternary: condition ? true_val : false_val)

    Linear algebra:
        dot:            matrix multiply (generalized)
        convolution:    conv2d with configurable dimension numbers
        batch_norm_inference, batch_norm_training

    Reduction:
        reduce:         fold a dimension with an associative op
        reduce_window:  pooling (max/avg over sliding window)
        all_reduce:     cross-device gradient aggregation (distributed)

    Shape manipulation:
        reshape:        change shape, same elements
        transpose:      reorder dimensions
        broadcast:      expand along new or existing dimensions
        slice:          extract a rectangular sub-tensor
        pad:            add padding values around a tensor
        gather:         index into tensor at specified positions
        scatter:        write values to specified positions

    Control flow:
        while:          conditional iteration (loop)
        conditional:    if-then-else on a condition
        call:           call another HloComputation

    Communication (distributed):
        all_reduce, all_gather, reduce_scatter, all_to_all
        send, recv: point-to-point between devices

### HLO Textual Format

    HLO can be printed in human-readable form:

        HloModule relu_module

        ENTRY relu {
          x = f32[8] parameter(0)
          zero = f32[] constant(0)
          zeros = f32[8] broadcast(zero), dimensions={}
          ROOT result = f32[8] maximum(x, zeros)
        }

    More complex example — a 2-layer MLP forward pass:

        HloModule mlp_forward

        ENTRY main {
          input    = f32[32,784] parameter(0)   ; batch=32, features=784
          W1       = f32[784,128] parameter(1)
          b1       = f32[128] parameter(2)
          W2       = f32[128,10] parameter(3)
          b2       = f32[10] parameter(4)

          ; Layer 1: hidden = relu(input @ W1 + b1)
          matmul1  = f32[32,128] dot(input, W1),
                     lhs_contracting_dims={1}, rhs_contracting_dims={0}
          bias1_bc = f32[32,128] broadcast(b1), dimensions={1}
          pre_act1 = f32[32,128] add(matmul1, bias1_bc)
          zero     = f32[] constant(0)
          zeros1   = f32[32,128] broadcast(zero), dimensions={}
          hidden   = f32[32,128] maximum(pre_act1, zeros1)    ; relu

          ; Layer 2: logits = hidden @ W2 + b2
          matmul2  = f32[32,10] dot(hidden, W2),
                     lhs_contracting_dims={1}, rhs_contracting_dims={0}
          bias2_bc = f32[32,10] broadcast(b2), dimensions={1}
          ROOT logits = f32[32,10] add(matmul2, bias2_bc)
        }

### How JAX Generates HLO

    When you write:
        @jax.jit
        def forward(x, W1, b1):
            return jax.nn.relu(x @ W1 + b1)

    JAX:
        1. Traces forward() with abstract (shapeful) values (no real data yet)
        2. Records all JAX operations → builds a Jaxpr (JAX expression)
        3. Lowers Jaxpr → HLO instructions
        4. Hands the HLO module to XLA for compilation
        5. XLA compiles → executable
        6. First real call: runs the compiled executable

    You can inspect the HLO:
        computation = jax.xla_computation(forward)(x, W1, b1)
        print(computation.as_hlo_text())   # prints the HLO module above


##### PART 3 — XLA'S COMPILATION PASSES

### Overview of the HLO Optimisation Pipeline

    XLA runs a sequence of passes on the HLO graph before code generation.
    Each pass is a transformation that improves performance or correctness.

    Passes run in this approximate order:
        1. Algebraic simplification
        2. HLO Common Subexpression Elimination (CSE)
        3. Layout assignment
        4. Operation fusion
        5. Buffer assignment
        6. Code generation

### Pass 1: Algebraic Simplification

    Applies mathematical identities at the HLO level:

        x + 0 → x                    (add zero)
        x * 1 → x                    (multiply by one)
        x * 0 → zeros_like(x)        (multiply by zero)
        broadcast(scalar) + y → y    (scalar broadcast followed by add)
        transpose(transpose(x)) → x  (double transpose cancels)
        reshape(reshape(x, s1), s2) → reshape(x, s2)  (fold reshapes)
        reduce(broadcast(x)) → ...   (push reduce through broadcast)

    These simplifications create opportunities for subsequent passes.

### Pass 2: Common Subexpression Elimination (CSE)

    If the same HLO instruction (same opcode, same operands, same shape)
    appears twice in the graph, replace both uses with one instruction.

    Common in ML: the same embedding lookup or positional encoding is
    used in multiple attention heads. CSE ensures it's computed once.

### Pass 3: Layout Assignment

    This is one of XLA's most important and most unique passes.

    The problem: matrix multiply and convolution have preferred memory layouts.
        matmul(A, B):  A should be row-major, B column-major for efficiency.
        conv2d:        NHWC vs NCHW vs NCHW_VECT_C — depends on hardware.
        GPU cuDNN:     prefers NCHW (batch, channels, height, width).
        CPU:           often prefers NHWC.
        TPU:           has its own tiled layout preferences.

    Layout assignment propagates layout constraints through the graph:
        "cuDNN conv wants NCHW" → insert transpose before conv if input is NHWC
        "matmul backend wants column-major B" → assign column-major to B
        Propagate these constraints to producers (earlier ops)

    The goal: minimise the number of explicit layout-changing operations
    (transposes, copies) while satisfying each op's hardware preference.

### Pass 4: Operation Fusion — XLA's Core Optimisation

    Fusion is XLA's most impactful optimisation. It merges multiple HLO
    instructions into a single GPU kernel or CPU loop nest.

    Why fusion matters:
        Without fusion (element-wise chain: add → relu → multiply):
            Launch kernel 1: read A, B → compute A+B → write tmp1 (GPU memory)
            Launch kernel 2: read tmp1 → compute relu(tmp1) → write tmp2
            Launch kernel 3: read tmp2, C → compute tmp2*C → write result
            Memory traffic: 3 reads + 3 writes of the full tensor

        With fusion (all three in one kernel):
            Launch ONE kernel: read A, B, C → compute (A+B).relu()*C → write result
            Memory traffic: 2 reads + 1 write  (4× less memory traffic!)

    XLA's fusion rules:
        PRODUCER-CONSUMER fusion: if instruction B uses the output of A, and
        A is an elementwise op, fuse A into B's kernel.

        SIBLINGS fusion: if A and B both read the same input (with no deps
        between them), fuse them into one kernel that reads the input once.

    Fusion categories:
        kLoop:      simple elementwise fusion (one output element per thread)
        kInput:     reduction fusion (multiple input elements per thread)
        kOutput:    broadcast-like fusion

    What CAN be fused:
        Elementwise ops (add, relu, multiply, exp, log, sqrt, ...)
        Broadcasts
        Transposes (in some cases)
        Reductions (in an "input fusion" with a producer)

    What CANNOT be fused (hardware limits):
        dot (matmul) — uses dedicated CUDA GEMM library (cuBLAS/cuDNN)
        convolution — uses cuDNN
        These are library calls; the surrounding elementwise ops are fused.

    Real example (transformer attention):
        softmax(Q @ K^T / sqrt(d))
        → dot → scale → exp → reduce → divide → dot → output

        Fused: scale + exp fused into one kernel (producer-consumer)
               divide + second dot: partial fusion
               dot calls cannot be fused themselves (cuBLAS)

### Pass 5: Buffer Assignment

    XLA plans all memory allocations BEFORE running any computation.
    This is possible because HLO shapes are fully static.

    Buffer assignment determines:
        Which output buffers can ALIAS input buffers (in-place ops)
        Which temporary buffers can be REUSED between non-overlapping ops
        The total peak memory needed for the computation

    Example: in a chain A → B → C where B's output is only used by C,
    B and C can share a buffer (B's output buffer becomes C's output).

    This aggressive buffer reuse is why XLA-compiled programs often
    use less memory than framework-dispatched programs.


##### PART 4 — XLA BACKENDS: CPU, GPU, AND TPU

### The CPU Backend

    XLA's CPU backend compiles HLO to LLVM IR, then uses LLVM for
    architecture-specific optimisation and code generation.

    Pipeline:
        HLO → HloToIr (CPU emitter) → LLVM IR → LLVM passes → native code

    XLA adds above LLVM:
        Tiling:          split loops for cache efficiency (XLA's tiling pass)
        Vectorisation:   explicitly generate vector operations before LLVM
        Parallelism:     emit OpenMP or pthreads for multi-core
        GEMM dispatch:   call Eigen or BLAS for matrix multiply

    For small pointwise operations: XLA CPU generates tightly vectorised
    loops that outperform PyTorch's eager mode by avoiding kernel launch
    overhead and enabling cross-op fusion.

### The GPU Backend

    XLA's GPU backend compiles HLO to CUDA (NVIDIA) or ROCm (AMD) code.

    Pipeline:
        HLO → GpuCompiler → IR emitter → NVPTX/AMDGCN IR → GPU binary

    Key GPU-specific passes:
        GPU IR emitter:   generates PTX via LLVM's NVPTX backend
        cuBLAS/cuDNN:     for matmul and conv, XLA emits library CALLS
                          rather than generating GEMM kernels from scratch
        Custom kernels:   XLA auto-generates CUDA kernels for elementwise
                          fusion groups
        Triton:           XLA increasingly uses Triton for softmax and
                          custom attention kernels

    GPU memory management:
        XLA manages device memory explicitly (no PyTorch-style CUDA caching allocator).
        Buffer assignment determines all allocations upfront.
        Scratch space for temporary values is pre-allocated at compile time.

### The TPU Backend

    TPU (Tensor Processing Unit) is Google's custom ML accelerator.
    XLA is the ONLY way to run computation on TPUs — there is no PyTorch
    CUDA equivalent for TPU; you must go through XLA.

    TPU architecture (key differences from GPU):
        Systolic array:   special hardware for matrix multiply
                          A matrix flows through an array of multiply-accumulate units
                          No need for GEMM software; it's in hardware
        HBM:              high-bandwidth memory, accessed in large tiles
        MXU:              matrix multiply unit (128×128 on v3, 128×256 on v4)
        VPU:              vector processing unit (elementwise ops)
        Interconnect:     TPU pods connect chips for model parallelism

    XLA's TPU compilation:
        HLO → HLO optimisation → LLO (low-level operations) → TPU executable
        LLO describes the computation at the TPU hardware level:
            - Which operations go to MXU (matmul)
            - Which operations go to VPU (elementwise)
            - HBM tile layout for efficient access

    XLAdots' power on TPU:
        A TPU v4 chip: 275 TFLOPS BF16
        A TPU v4 pod:  10 exaFLOPS (1000+ chips connected)
        XLA coordinates data and computation across all chips automatically.


##### PART 5 — JAX AND XLA: THE UNIFIED INTERFACE

### JAX as XLA's Primary Consumer

    JAX (Just After eXecution) is designed from the ground up to compile
    everything through XLA. Unlike TF where XLA is optional, in JAX:
        - Every jax.jit call compiles to XLA
        - Every jax.vmap call compiles to XLA (with batch dimension expanded)
        - Every jax.grad call compiles BOTH forward AND backward through XLA
        - Every jax.pmap call compiles with XLA multi-device support

    JAX's design philosophy:
        1. Write pure Python/NumPy code
        2. jit-compile it through XLA
        3. Get performance comparable to hand-tuned CUDA

### The JAX Compilation Cache

    XLA compilation is expensive (seconds for large models).
    JAX caches compiled executables keyed on:
        - Input shapes (recompiles if shape changes)
        - Input dtypes
        - Static arguments (Python values marked as static)
        - XLA target (CPU/GPU/TPU)

    Avoiding recompilation:
        # BAD: different batch sizes → recompiles each time
        for batch_size in [8, 16, 32, 64]:
            result = jit_fn(data[:batch_size])  # 4 compilations!

        # GOOD: pad to fixed size or use static_argnums
        @partial(jax.jit, static_argnums=(1,))
        def fn(x, n_tokens):  # n_tokens is a Python int (static)
            return x[:n_tokens] @ W

### jax.xla_computation — Inspecting XLA Output

    You can extract and inspect the HLO that JAX generates:

        computation = jax.xla_computation(fn)(sample_input)
        print(computation.as_hlo_text())      # HLO text format
        print(computation.as_hlo_dot_graph()) # Graphviz DOT format

    This is invaluable for debugging performance:
        - Are ops being fused?
        - Is there an unexpected reshape or transpose?
        - Is the layout causing extra transposes?

### XLA's Effect on Training Throughput

    Comparing JAX+XLA vs PyTorch eager for transformer training:

        Transformer block forward pass (A100 GPU):
            PyTorch eager:  ~6 kernel launches per block
                            (matmul, matmul, softmax pieces, matmul, add, norm)
            JAX+XLA:        ~3 kernel launches per block
                            (matmul, fused-softmax+masking, matmul)

        Memory traffic reduction (via fusion):
            Eager: write activations after EVERY op (attention scores, softmax)
            XLA:   fuse attention-score → softmax → dropout into one kernel

        Typical end-to-end speedup:
            JAX+XLA over PyTorch eager: 20–40% faster on GPU
            JAX+XLA on TPUv4 vs A100:  3–5× faster (specialised hardware + XLA)


##### PART 6 — DISTRIBUTED COMPUTATION: SPMD AND PJIT

### SPMD: Single Program, Multiple Data

    XLA's distributed strategy is SPMD (Single Program Multiple Data):
        - ONE program is written (not N programs for N devices)
        - XLA replicates and shards it across N devices automatically
        - Collective communication (AllReduce, AllGather) is inserted by XLA

    This contrasts with PyTorch's DDP where the user explicitly:
        - Wraps the model in DistributedDataParallel
        - Manages gradient synchronisation
        - Handles device placement manually

    In JAX's SPMD model:
        - The user annotates tensors with sharding specs (PartitionSpec)
        - XLA's GSPMD compiler partitions the computation
        - Collectives are inserted where data needs to cross device boundaries

### jax.sharding — Annotating Tensor Distributions

    In JAX (and XLA under the hood):

        from jax.sharding import Mesh, PartitionSpec, NamedSharding

        # Create a mesh of 8 GPUs as a 2D grid (data × model parallel)
        devices = jax.devices()               # 8 GPUs
        mesh    = Mesh(np.array(devices).reshape(2, 4), ("data", "model"))

        # Shard weights across model dimension
        W_spec = PartitionSpec("model", None)   # rows across 4 model-parallel GPUs
        # Replicate across data dimension (same weights on all data-parallel copies)

        # Shard input batch across data dimension
        X_spec = PartitionSpec("data", None)    # batch across 2 data-parallel groups

        # jit with sharding: XLA inserts AllReduce for gradient sync
        f_parallel = jax.jit(fn, in_shardings=(X_sharding, W_sharding))

### HLO for Distributed Computation

    XLA inserts collective operations into the HLO graph automatically:

        all_reduce:   sum/average gradients across data-parallel replicas
        all_gather:   collect sharded tensors (gather across model-parallel GPUs)
        reduce_scatter: scatter-reduce for gradient sharding (ZeRO-style)
        all_to_all:   transpose data/model dimensions in model-parallel attention

    These collectives are represented as HLO instructions:
        %grad_synced = all-reduce(%grad_local),
                       channel_id=1, use_global_device_ids=true,
                       to_apply=add_computation


##### PART 7 — STABLEHLO: XLA'S MLIR-BASED FUTURE

### The Evolution from HLO to StableHLO

    Original XLA HLO:
        - Defined as C++ protobuf structures (not a formal IR)
        - No stability guarantee between XLA versions
        - Hard to extend or compose with other tools
        - Not portable (depended on XLA's internal protos)

    StableHLO (2022–present):
        - HLO operations redefined as an MLIR dialect (from MLIR module)
        - FORMAL STABILITY GUARANTEE: ops defined now work in 5 years
        - Portability: any tool reading StableHLO .mlir files works
        - Extensibility: MLIR infrastructure (passes, interfaces, patterns)
        - Composable: StableHLO programs can be embedded in larger MLIR modules

    Migration:
        Old: TF/JAX → HLO proto → XLA internal → GPU code
        New: TF/JAX → StableHLO .mlir → XLA (via StableHLO → HLO) → GPU code
             OR:      TF/JAX → StableHLO .mlir → IREE → GPU code
             OR:      TF/JAX → StableHLO .mlir → TVM → GPU code

### StableHLO as the Portability Layer

    StableHLO's stability guarantee makes it the first truly portable
    ML computation representation:

        Save:  jax.export(fn)(x).serialize()  → StableHLO bytes
        Load:  5 years later, new XLA version, still runs correctly

    This enables:
        Model zoos that survive framework upgrades.
        Compilation artifacts that can be shipped to arbitrary XLA versions.
        Third-party compilers (IREE, TVM, OpenXLA) consuming JAX models.

### XLA's Relationship to MLIR Infrastructure

    As covered in the MLIR module, XLA increasingly uses MLIR:
        - StableHLO/MHLO are MLIR dialects
        - XLA's passes are being migrated to MLIR's pass manager
        - Pattern rewriting (from MLIR module) used for HLO algebraic simplifications
        - The LLVM dialect bridges XLA's CPU backend to LLVM IR

    This is the "connected stack" in action:
        MLIR infrastructure → XLA uses it internally
        LLVM backend → XLA emits LLVM IR for CPU


##### PART 8 — XLA PERFORMANCE DEBUGGING AND PROFILING

### XLA Compilation Flags and Environment Variables

    XLA_FLAGS environment variable controls compilation:

        XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps"
            Dump HLO before/after each pass for inspection.
            Creates: module_XXXX.before_pass.txt, module_XXXX.after_pass.txt

        XLA_FLAGS="--xla_gpu_enable_triton_gemm=true"
            Use Triton-based GEMM instead of cuBLAS for some matmuls.

        XLA_FLAGS="--xla_gpu_autotune_level=4"
            Maximum autotuning: try all cuBLAS/cuDNN algorithm variants.

        XLA_FLAGS="--xla_gpu_cuda_graph_enable=1"
            Capture CUDA graphs for repeated execution (reduces launch overhead).

        XLA_FLAGS="--xla_force_host_platform_device_count=8"
            Simulate 8 devices on CPU for SPMD debugging.

### HLO Profiler and Memory Analysis

    XLA's HloProto tools:
        hlo-proto --graphviz:   render the HLO graph as Graphviz DOT
        xla-hlo-display:        interactive HLO viewer

    Memory usage analysis:
        XLA_FLAGS="--xla_hlo_profile=true"
        → After compilation, prints memory usage breakdown:
            Parameter memory: 1.2 GB  (model weights)
            Output memory:    0.05 GB (result tensor)
            Temp memory:      0.3 GB  (scratch buffers)
            Peak memory:      1.55 GB

    Fusion analysis:
        The HLO dump shows fusion groups explicitly:
            %fused_computation = f32[...] fusion(...)
            ;  contains: add, relu, multiply — 3 ops, 1 kernel launch

### GPU Profiling with nvprof / Nsight

    XLA-compiled GPU code can be profiled like any CUDA program:

        # Profile a JAX computation
        nsys profile --trace=cuda python jax_model.py

    What to look for:
        - Kernel names: XLA kernels are named xla__computation_XX
        - Fusion success: one kernel for a chain of elementwise ops
        - cuBLAS calls: GEMM operations
        - Memory copies: look for unexpected HtoD/DtoH transfers

    TensorBoard integration (TF + XLA):
        With TensorBoard's profile tab, you can see:
        - Per-op execution time
        - Kernel launch overhead
        - Memory bandwidth utilisation
        - Fusion groups (ops compiled into one kernel)

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · HLO Internals — Inspecting XLA via JAX": {
        "description": (
            "Use JAX to generate and inspect XLA HLO for real computations. "
            "jax.xla_computation() to extract HLO text. "
            "Trace a softmax, a matmul, and a transformer layer. "
            "Show what HLO looks like before and after fusion. "
            "Demonstrate how jit compilation caches and recompiles. "
            "Compare HLO op counts eager vs compiled."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  HLO INTERNALS — INSPECTING XLA VIA JAX")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    from functools import partial
    print(f"  JAX version:    {jax.__version__}")
    print(f"  XLA backend:    {jax.default_backend()}")
    print(f"  Devices:        {jax.devices()}")
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("  JAX not installed: pip install jax")
    print("  For GPU: pip install jax[cuda12] -f https://storage.googleapis.com/...")
    print()

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Extracting HLO from JAX computations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Extracting HLO with jax.xla_computation()")
print("━" * 65)
print()

if HAS_JAX:
    # ── ReLU HLO ──────────────────────────────────────────────────────────
    def relu(x):
        return jnp.maximum(x, 0.0)

    relu_hlo = jax.xla_computation(relu)(jnp.ones((8,), dtype=jnp.float32))
    hlo_text = relu_hlo.as_hlo_text()

    print("  HLO for relu(x) on shape (8,) f32:")
    print()
    for line in hlo_text.split('\n'):
        print(f"    {line}")
    print()

    # ── Softmax HLO — shows reduce + broadcast pattern ─────────────────
    def softmax(x):
        x_max = jnp.max(x, axis=-1, keepdims=True)
        x_exp = jnp.exp(x - x_max)
        return x_exp / jnp.sum(x_exp, axis=-1, keepdims=True)

    sm_input = jnp.ones((4, 8), dtype=jnp.float32)
    sm_hlo   = jax.xla_computation(softmax)(sm_input)
    sm_text  = sm_hlo.as_hlo_text()

    print("  HLO for softmax on shape (4,8) f32:")
    print()
    for line in sm_text.split('\n'):
        print(f"    {line}")
    print()

    # Count HLO instructions
    hlo_lines = [l.strip() for l in sm_text.split('\n')
                 if l.strip() and '=' in l and 'ROOT' not in l
                 and not l.strip().startswith('HloModule')
                 and not l.strip().startswith('ENTRY')]
    print(f"  HLO instruction count (softmax): ~{len(hlo_lines)} ops")
    print("  (reduce_max, broadcast, subtract, exp, reduce_add, broadcast, divide)")
    print()

else:
    # Reference HLO for when JAX is not available
    RELU_HLO = """
  HLO for relu(x) — shape (8,) f32:

  HloModule relu_module

  ENTRY relu {
    x    = f32[8] parameter(0)
    zero = f32[] constant(0)
    bc   = f32[8] broadcast(zero), dimensions={}
    ROOT result = f32[8] maximum(x, bc)
  }

  Breakdown:
    parameter(0)  — the input x
    constant(0)   — scalar zero
    broadcast      — expand scalar zero to shape [8]
    maximum        — elementwise max(x, 0) = relu
"""
    SOFTMAX_HLO = """
  HLO for softmax — shape (4,8) f32:

  HloModule softmax_module

  add_comp {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT sum = f32[] add(lhs, rhs)
  }
  max_comp {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT mx = f32[] maximum(lhs, rhs)
  }

  ENTRY softmax {
    x          = f32[4,8] parameter(0)
    neg_inf    = f32[] constant(-inf)
    row_max    = f32[4] reduce(x, neg_inf), dimensions={1}, to_apply=max_comp
    max_bc     = f32[4,8] broadcast(row_max), dimensions={0}
    x_shifted  = f32[4,8] subtract(x, max_bc)
    x_exp      = f32[4,8] exponential(x_shifted)
    zero       = f32[] constant(0)
    row_sum    = f32[4] reduce(x_exp, zero), dimensions={1}, to_apply=add_comp
    sum_bc     = f32[4,8] broadcast(row_sum), dimensions={0}
    ROOT result = f32[4,8] divide(x_exp, sum_bc)
  }

  7 HLO operations. With XLA fusion:
    x_shifted + x_exp can be fused (producer-consumer)
    row_sum reduction can be fused with x_exp computation
"""
    print(RELU_HLO)
    print(SOFTMAX_HLO)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: jit compilation caching and recompilation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — JIT caching: when XLA recompiles")
print("━" * 65)
print()

if HAS_JAX:
    compile_count = [0]
    shape_log     = []

    @jax.jit
    def tracked_fn(x):
        compile_count[0] += 1   # Python side-effect: only runs at trace time
        shape_log.append(x.shape)
        return jnp.relu(x) * 0.5

    print("  Calling tracked_fn with different shapes and dtypes:")
    print(f"  {'Call':>5} | {'Shape':>15} | {'dtype':>8} | {'Compiles':>10} | {'Cached?':>8}")
    print(f"  {'─'*55}")

    calls = [
        (jnp.ones((4,),    jnp.float32), "float32"),
        (jnp.ones((4,),    jnp.float32), "float32"),  # cached ✅
        (jnp.ones((8,),    jnp.float32), "float32"),  # new shape → recompile
        (jnp.ones((8,),    jnp.float32), "float32"),  # cached ✅
        (jnp.ones((8,),    jnp.float16), "float16"),  # new dtype → recompile
        (jnp.ones((4,),    jnp.float32), "float32"),  # old shape → cached ✅
    ]

    prev_count = 0
    for i, (x, dtype_label) in enumerate(calls):
        _ = tracked_fn(x)
        new_compile = compile_count[0] > prev_count
        cached      = not new_compile
        prev_count  = compile_count[0]
        print(f"  {i+1:5d} | {str(x.shape):>15} | {dtype_label:>8} | "
              f"{compile_count[0]:>10} | {'✅ cached' if cached else '🔄 recompiled':>8}")

    print()
    print(f"  Total compilations: {compile_count[0]}")
    print(f"  Total calls: {len(calls)}")
    print()
    print("  XLA recompiles when:")
    print("    • Input shape changes (different tensor sizes)")
    print("    • Input dtype changes (f32 → f16)")
    print("    • Static arguments change (marked with static_argnums)")
    print("    • Device changes (CPU → GPU)")
    print()
    print("  XLA reuses the cache when:")
    print("    • Shape and dtype are identical to a previous call")
    print("    • Even if the actual data values are different")
    print()

    # Demonstrate static_argnums
    print("  Using static_argnums to avoid recompilation on config changes:")

    @partial(jax.jit, static_argnums=(1,))
    def fn_with_static(x, mode: str):
        # 'mode' is a Python string — static, won't cause dtype recompile
        if mode == "relu":
            return jnp.maximum(x, 0)
        else:
            return jnp.tanh(x)

    x4 = jnp.ones((4,))
    r1 = fn_with_static(x4, "relu")    # compiles for mode="relu"
    r2 = fn_with_static(x4, "tanh")    # compiles for mode="tanh" (new static val)
    r3 = fn_with_static(x4, "relu")    # cached (same static val)
    print(f"  fn(x, 'relu') = {r1[:3].tolist()}...")
    print(f"  fn(x, 'tanh') = {r2[:3].tolist()}...")
    print(f"  fn(x, 'relu') — cached ✅")
    print()

else:
    print("  (JAX not available — showing compilation caching reference)")
    CACHE_REF = """
  JIT COMPILATION CACHING RULES:

  @jax.jit
  def fn(x):
      return jnp.relu(x)

  fn(jnp.ones((4,), jnp.float32))  # COMPILE (shape=(4,), dtype=f32)
  fn(jnp.ones((4,), jnp.float32))  # CACHE HIT ✅ (same shape+dtype)
  fn(jnp.ones((8,), jnp.float32))  # RECOMPILE (shape=(8,) is new)
  fn(jnp.ones((8,), jnp.float16))  # RECOMPILE (dtype=f16 is new)

  AVOID recompilation:
  1. Use fixed shapes (pad inputs to fixed size)
  2. Mark Python values as static:
     @partial(jax.jit, static_argnums=(1,))
     def fn(x, n): ...
  3. Use jax.ShapeDtypeStruct to pre-trace with expected shapes
"""
    print(CACHE_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Benchmarking XLA fusion vs eager
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — XLA fusion speedup: elementwise chains")
print("━" * 65)
print()

if HAS_JAX:
    N     = 1_000_000
    key   = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (N,), dtype=jnp.float32)

    # Eager: each op dispatches separately (like PyTorch eager)
    def elementwise_chain_eager(x):
        return (jnp.exp(x - jnp.max(x)) *
                jnp.tanh(x * 0.5 + 1.0) +
                jnp.sqrt(jnp.abs(x) + 1e-6))

    # JIT: XLA fuses the entire chain into ONE kernel
    elementwise_chain_jit = jax.jit(elementwise_chain_eager)

    # Warmup
    _ = elementwise_chain_eager(x_jax).block_until_ready()
    _ = elementwise_chain_jit(x_jax).block_until_ready()

    REPS = 100
    t0 = time.perf_counter()
    for _ in range(REPS):
        elementwise_chain_eager(x_jax).block_until_ready()
    t_eager = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        elementwise_chain_jit(x_jax).block_until_ready()
    t_jit = (time.perf_counter() - t0) / REPS * 1000

    print(f"  Elementwise chain: exp(x-max) * tanh(x*0.5+1) + sqrt(|x|+eps)")
    print(f"  N = {N:,} float32 elements")
    print()
    print(f"  {'Mode':<20} | {'Time (ms)':>12} | {'Speedup':>9} | {'Notes'}")
    print(f"  {'─'*65}")
    print(f"  {'Eager (no jit)':<20} | {t_eager:12.4f} | {'1.00×':>9} | Multiple kernel launches")
    print(f"  {'JIT (XLA fused)':<20} | {t_jit:12.4f} | {t_eager/t_jit:9.2f}× | Single fused kernel")
    print()
    print(f"  Fused kernel reads x once, writes result once.")
    print(f"  Eager launches {5} separate kernels, each reading the full tensor.")
    print()

    # Matmul + activation fusion
    M, K, N_dim = 256, 512, 256
    A   = jax.random.normal(key, (M, K), jnp.float32)
    B   = jax.random.normal(key, (K, N_dim), jnp.float32)
    b   = jax.random.normal(key, (N_dim,), jnp.float32)

    def linear_relu_eager(A, B, b):
        return jnp.maximum(A @ B + b, 0.0)

    linear_relu_jit = jax.jit(linear_relu_eager)

    _ = linear_relu_eager(A, B, b).block_until_ready()
    _ = linear_relu_jit(A, B, b).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(REPS):
        linear_relu_eager(A, B, b).block_until_ready()
    t_eg2 = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        linear_relu_jit(A, B, b).block_until_ready()
    t_jit2 = (time.perf_counter() - t0) / REPS * 1000

    print(f"  Linear + bias + ReLU: A@B + b → relu  ({M}×{K} @ {K}×{N_dim})")
    print(f"  {'Mode':<20} | {'Time (ms)':>12} | {'Speedup':>9}")
    print(f"  {'─'*45}")
    print(f"  {'Eager':<20} | {t_eg2:12.4f} | {'1.00×':>9}")
    print(f"  {'JIT (XLA)':<20} | {t_jit2:12.4f} | {t_eg2/t_jit2:9.2f}×")
    print()
    print("  XLA fuses bias_add + relu with matmul's output buffer.")
    print("  On GPU: bias+relu fused INTO the cuBLAS/custom matmul epilogue.")

else:
    print("  (JAX not available for live benchmarks)")
    FUSION_CONCEPT = """
  FUSION SPEEDUP CONCEPT:

  Elementwise chain: exp → tanh → sqrt → add (N=1M floats)
  ─────────────────────────────────────────────────────────
  Eager:  5 separate kernel launches
          Each kernel: read 4MB from GPU HBM, write 4MB back
          Total memory traffic: 5 × 8MB = 40MB
          GPU is MEMORY BANDWIDTH LIMITED (not compute limited)

  XLA JIT: 1 fused kernel
          Read 4MB from GPU HBM, write 4MB back
          Total memory traffic: 8MB  (5× less!)
          Expected speedup: ~3-5× on GPU

  Linear + bias + relu  (256×512 @ 512×256)
  ─────────────────────────────────────────────────────────
  Eager:  matmul → bias_add → relu  (3 kernel launches)
  XLA:    matmul → fused(bias+relu) (2 kernel launches)
          XLA fuses bias+relu into the matmul output epilogue
          Expected speedup: ~10-20% (matmul dominates)

  KEY INSIGHT: Fusion matters most for memory-bound ops (elementwise).
  For compute-bound ops (matmul), fusion of surrounding ops helps less.
"""
    print(FUSION_CONCEPT)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · XLA Compilation Pipeline — Passes, Fusion, and Layout": {
        "description": (
            "Deep dive into XLA's compilation passes. "
            "Algebraic simplification: show which transformations XLA applies. "
            "Fusion rules: producer-consumer vs siblings, what can and cannot fuse. "
            "Layout assignment: NHWC vs NCHW and its impact on convolution. "
            "Buffer assignment: how XLA plans memory before execution. "
            "XLA_FLAGS for dumping and inspecting the HLO pass pipeline."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  XLA COMPILATION PIPELINE — PASSES, FUSION AND LAYOUT")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Algebraic simplification — illustrated
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Algebraic simplification pass")
print("━" * 65)
print()

print("  XLA's algebraic simplifier applies mathematical identities")
print("  to reduce the HLO graph before code generation.")
print()

ALGEBRAIC_SIMPLIFICATIONS = """
  RULE                                BEFORE                                        AFTER
  ───────────────────────────────────────────────────────────────────────────────────────────────────
  Add-zero elimination                add(x, 0)                                     x
  Multiply-one elimination            multiply(x, 1)                                x
  Multiply-zero elimination           multiply(x, 0)                                zeros_like(x)
  Subtract-self elimination           subtract(x, x)                                zeros_like(x)
  Double negation                     negate(negate(x))                             x
  Double transpose                    transpose(transpose(x, {1,0}), {1,0})         x
  Consecutive reshapes                reshape(reshape(x,s1),s2)                     reshape(x,s2)
  Broadcast-reduce                    reduce(broadcast(x,d),d)                      x (if reduce-op is identity)
  Slice-full                          slice(x, [0..n])                              x (if n = size of x)
  Broadcast + elementwise             broadcast(a) + broadcast(b) → broadcast(a+b)
  Concatenate-one                     concatenate([x])                              x
  Constant folding                    add(5, 3)                                     8  (computed at compile time)
  Comparison with itself              equal(x, x)                                   ones_like(x) (type i1)
  Log of exp                          log(exp(x))                                   x  (exact inverse)
  Sqrt of square                      sqrt(multiply(x, x))                          abs(x)
"""
print(ALGEBRAIC_SIMPLIFICATIONS)

try:
    import jax
    import jax.numpy as jnp

    # Demonstrate: JAX+XLA eliminates redundant ops
    key = jax.random.PRNGKey(42)
    x   = jax.random.normal(key, (4,))

    # These patterns should be simplified by XLA
    @jax.jit
    def with_redundant_ops(x):
        a = x * 1.0                  # multiply by 1 → eliminated
        b = a + 0.0                  # add zero → eliminated
        c = jnp.transpose(jnp.transpose(b))  # double transpose → eliminated
        d = jnp.reshape(jnp.reshape(c, (2, 2)), (4,))  # double reshape → one reshape
        return d

    # Extract HLO and count instructions
    hlo = jax.xla_computation(with_redundant_ops)(x)
    text = hlo.as_hlo_text()
    instructions = [l for l in text.split('\n') if '=' in l and 'parameter' not in l
                    and 'ENTRY' not in l and 'HloModule' not in l]

    result = with_redundant_ops(x)
    print(f"  Input:  {x.tolist()}")
    print(f"  Output: {result.tolist()}")
    print(f"  After algebraic simplification: {len(instructions)} HLO instruction(s)")
    print(f"  (All redundant ops eliminated — effectively just returns x reshaped)")
    print()

except ImportError:
    print("  (JAX not available — showing concept only)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Fusion rules — what fuses and what doesn't
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Fusion rules: what fuses and what doesn't")
print("━" * 65)
print()

FUSION_RULES = """
  XLA FUSION RULES
  ═══════════════════════════════════════════════════════════════

  PRODUCER-CONSUMER FUSION (most common):
    If A's output feeds directly into B, and A is elementwise:
    → Fuse A into B's kernel

    CAN FUSE:
      relu(x) → multiply(_, 0.5) → add(_, bias)   ✅ all elementwise
      exp(x)  → log(_)                              ✅ elementwise chain
      add(x, b) → relu(_)                           ✅ classic bias+relu

    CANNOT FUSE (fusion-opaque ops):
      matmul(A, B) → relu(_)   → matmul starts a new fusion group
      conv(x, w)   → bias(_)   → conv is opaque; bias fused separately

    SPECIAL CASE — epilogue fusion:
      On modern NVIDIA GPUs (Ampere+), XLA can fuse elementwise ops
      directly into the cuBLAS GEMM epilogue:
      matmul(A,B) → add(_, b) → relu(_)  → ONE cuBLAS call with epilogue ✅

  SIBLING FUSION:
    If A and B both read the same input (no dependency between them):
    → Fuse A and B into one kernel that reads the input once

    Example: layer normalisation reads x to compute mean AND variance:
      mean = reduce(x, sum) / N          sibling: both read x
      var  = reduce((x-mean)^2, sum) / N
      → XLA may fuse mean+var computation into one pass over x

  INPUT FUSION (reduction + producer):
    If a producer feeds into a reduction op:
    → The producer can be fused INTO the reduction kernel

    Example: softmax numerator computation:
      exp(x - max(x)) → reduce(sum)
      → exp can be fused into the reduce kernel
      → input to reduce is computed on-the-fly, not stored

  WHAT STAYS AS SEPARATE KERNELS:
    • matmul / dot (cuBLAS call — library, not XLA kernel)
    • convolution (cuDNN call)
    • allreduce (NCCL call)
    • sort, scatter (complex enough to need separate kernels)
    • Any op with >1 output that's used in different fusion groups
"""
print(FUSION_RULES)

try:
    import jax
    import jax.numpy as jnp

    # Show fusion via operation counting in HLO
    def three_ops(x, w, b):
        """matmul + bias + relu — should fuse bias+relu"""
        return jnp.maximum(x @ w + b, 0.0)

    def five_elementwise(x):
        """Chain of 5 elementwise — should all fuse"""
        return jnp.log(jnp.exp(jnp.sqrt(jnp.abs(x) + 1e-6)) + 1.0) * jnp.tanh(x)

    M, K = 64, 64
    key  = jax.random.PRNGKey(0)
    x_mm = jax.random.normal(key, (M, K))
    w_mm = jax.random.normal(key, (K, K))
    b_mm = jax.random.normal(key, (K,))
    x_el = jax.random.normal(key, (K,))

    hlo_mm = jax.xla_computation(three_ops)(x_mm, w_mm, b_mm)
    hlo_el = jax.xla_computation(five_elementwise)(x_el)

    # Count fusion groups (fused computation blocks)
    def count_fusion_groups(hlo_text):
        return hlo_text.count("fused_computation") // 2  # each appears as def + call

    mm_fusions = count_fusion_groups(hlo_mm.as_hlo_text())
    el_fusions  = count_fusion_groups(hlo_el.as_hlo_text())

    print(f"  matmul + bias + relu: {mm_fusions} fusion group(s)")
    print(f"    (matmul = cuBLAS, bias+relu = fused into 1 kernel)")
    print()
    print(f"  5-op elementwise chain: {el_fusions} fusion group(s)")
    print(f"    (all 5 ops fused into 1 kernel — 1 GPU kernel launch)")
    print()

except ImportError:
    print("  (JAX not available)")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Layout assignment and its impact
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Layout assignment: NHWC vs NCHW")
print("━" * 65)
print()

LAYOUT_EXPLAINED = """
  TENSOR LAYOUTS IN XLA
  ══════════════════════════════════════════════════════════════

  A 4D image tensor has dimensions: (batch, channels, height, width)
  abbreviated: (N, C, H, W)

  NCHW layout:   channels are contiguous in memory
    Memory: [b0_c0_h0_w0, b0_c0_h0_w1, ..., b0_c0_h1_w0, ..., b0_c1_h0_w0, ...]
    Hardware preference:
      cuDNN (NVIDIA):     prefers NCHW → fastest convolution
      NVIDIA Tensor Cores: optimised for NCHW
      Older GPUs:         required NCHW for cuDNN

  NHWC layout:   spatial dimensions (height, width) are contiguous
    Memory: [b0_h0_w0_c0, b0_h0_w0_c1, ..., b0_h0_w0_cC, b0_h0_w1_c0, ...]
    Hardware preference:
      TPUs:              prefer NHWC
      ARM CPU (NEON):    prefer NHWC
      cuDNN (new):       also supports NHWC_VECT_C (channels packed in 32)
      TFLite (mobile):   prefers NHWC

  XLA layout assignment:
    1. Query each op's hardware preferences
    2. Propagate layout constraints backward through the graph
    3. Insert explicit layout changes (transposes) where constraints conflict
    4. Minimise the total number of transposes

  EXAMPLE: If your model receives NHWC input but cuDNN needs NCHW:
    input (NHWC) → [XLA inserts transpose] → NCHW → conv(cuDNN) → NCHW
                                                              ↓
                             [XLA inserts transpose] → NHWC (if next op wants NHWC)

  XLA flags for layout control:
    XLA_FLAGS="--xla_gpu_force_conv_nhwc=true"    force NHWC for all conv
    XLA_FLAGS="--xla_gpu_force_conv_nchw=true"    force NCHW for all conv

  Debugging layout issues:
    XLA_FLAGS="--xla_dump_to=/tmp/dumps"
    → Dumps HLO before and after layout assignment
    → Look for unexpected kTranspose instructions = layout conflict detected
"""
print(LAYOUT_EXPLAINED)

try:
    import jax
    import jax.numpy as jnp

    # Demonstrate layout impact on conv
    key = jax.random.PRNGKey(0)

    # NHWC input (JAX's default)
    x_nhwc  = jax.random.normal(key, (1, 28, 28, 1), dtype=jnp.float32)
    # NCHW input (PyTorch default)
    x_nchw  = jnp.transpose(x_nhwc, (0, 3, 1, 2))

    # Convolution in JAX (uses NHWC by default)
    from jax import lax
    kernel  = jax.random.normal(key, (3, 3, 1, 4), dtype=jnp.float32)  # (H,W,in,out)

    def conv_nhwc(x, k):
        return lax.conv_general_dilated(
            x, k,
            window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    t0 = time.perf_counter()
    out = jax.jit(conv_nhwc)(x_nhwc, kernel)
    out.block_until_ready()
    for _ in range(200): jax.jit(conv_nhwc)(x_nhwc, kernel).block_until_ready()
    t_nhwc = (time.perf_counter() - t0) / 200 * 1000

    print(f"  Conv2D (28×28, 1→4 channels, 3×3 kernel):")
    print(f"    NHWC input: output shape = {out.shape}")
    print(f"    Time:       {t_nhwc:.4f} ms")
    print()
    print("  XLA assigns optimal layout per backend:")
    print("    GPU:    cuDNN prefers NCHW → may insert NHWC→NCHW transpose")
    print("    CPU:    often prefers NHWC → no transpose needed")
    print("    TPU:    proprietary tiled layout → XLA handles all conversions")

except ImportError:
    print("  (JAX not available for conv demo)")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Buffer assignment — memory planning
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Buffer assignment: static memory planning")
print("━" * 65)
print()

BUFFER_ASSIGNMENT = """
  XLA BUFFER ASSIGNMENT
  ══════════════════════════════════════════════════════════════

  Before executing ANY computation, XLA plans ALL memory allocations.
  This is possible because HLO has fully static shapes.

  Key concepts:

  LIVE INTERVALS:
    For each buffer (HLO instruction output), compute the range of
    execution steps where it is "live" (needed by future ops).

    Example for A → B → C → D where B and C each use A:
      A: live from step 0 to step 2 (B and C both use it)
      B: live from step 1 to step 3 (C and D use B's output)
      C: live from step 2 to step 3 (D uses C's output)

  BUFFER REUSE:
    Two buffers can share physical memory if their live intervals DON'T OVERLAP.
    After A is no longer needed, its memory can be reused for D's output.

  IN-PLACE OPERATIONS:
    Some ops can reuse their INPUT buffer for output (in-place):
      elementwise ops: output has same size as input → reuse input buffer
      XLA marks these as "input/output aliases" in buffer assignment

  PEAK MEMORY:
    Buffer assignment computes the peak memory needed:
      peak_memory = max over all time steps of sum of live buffer sizes

  EXAMPLE — Residual connection:
    input (4MB) → conv1 → relu → conv2 → add(conv2, input)
    Buffer plan:
      input:  live [0..4]  — must survive until add
      conv1:  live [1..2]  — freed after relu
      relu:   live [2..3]  — freed after conv2
      conv2:  live [3..4]  — freed after add

    Memory timeline (MB):
      Step 0: input(4)             = 4MB
      Step 1: input(4) + conv1(4)  = 8MB  ← PEAK
      Step 2: input(4) + relu(4)   = 8MB  (conv1 freed, relu same size)
      Step 3: input(4) + conv2(4)  = 8MB  (relu freed)
      Step 4: output(4)            = 4MB  (input freed after add)

  Real XLA memory stats (via XLA_FLAGS="--xla_hlo_profile=true"):
    Parameter buffers:   ~1.2 GB  (model weights, never freed)
    Output buffer:       ~50 MB   (result)
    Temp/scratch:        ~300 MB  (intermediates, reused aggressively)
    Peak memory:         ~1.55 GB
"""
print(BUFFER_ASSIGNMENT)

try:
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)

    # Show memory efficiency via JAX's built-in memory reporting
    if jax.default_backend() == 'gpu':
        import jax.tools.colab_tpu  # only on GPU
    else:
        # Demonstrate that XLA pre-plans buffers
        @jax.jit
        def residual_block(x, w1, w2):
            h = jnp.tanh(x @ w1)
            return x + h @ w2   # skip connection: input x must survive until add

        M = 512
        x_  = jax.random.normal(key, (M, M))
        w1_ = jax.random.normal(key, (M, M))
        w2_ = jax.random.normal(key, (M, M))

        hlo = jax.xla_computation(residual_block)(x_, w1_, w2_)
        text = hlo.as_hlo_text()
        num_ops = len([l for l in text.split('\n') if '=' in l and not l.startswith('HloModule')])
        print(f"  Residual block HLO operations: ~{num_ops}")
        print(f"  XLA pre-plans memory for all {num_ops} ops at compile time.")
        print(f"  Temporary buffers are reused wherever live intervals don't overlap.")
        print()
        print("  To inspect buffer assignment:")
        print("    XLA_FLAGS='--xla_dump_hlo_pass_re=.*buffer.*'")
        print("    → Dumps buffer assignment to text showing which buffers share memory")

except ImportError:
    print("  (JAX not available)")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · XLA with JAX — Full Training, SPMD, and StableHLO Export": {
        "description": (
            "XLA in production via JAX. "
            "Full training loop: grad + jit + update, all compiled by XLA. "
            "pmap for multi-device data parallel training. "
            "jax.sharding for SPMD tensor parallelism. "
            "StableHLO export and reload for portable model deployment. "
            "XLA performance profiling with XLA_FLAGS dump options."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  XLA WITH JAX — TRAINING, SPMD AND STABLEHLO EXPORT")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    from functools import partial
    HAS_JAX = True
    print(f"  JAX {jax.__version__}, backend={jax.default_backend()}")
    print(f"  Devices: {len(jax.devices())} × {jax.devices()[0]}")
except ImportError:
    HAS_JAX = False
    print("  JAX not installed: pip install jax")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Full JAX+XLA training loop
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Full JAX+XLA training loop")
print("━" * 65)
print()

if HAS_JAX:
    # MLP model as pure functions (JAX/functional style)
    def init_params(key, layer_sizes):
        """Initialise parameters as a list of (W, b) tuples."""
        params = []
        keys   = jax.random.split(key, len(layer_sizes) - 1)
        for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            W = jax.random.normal(k, (n_in, n_out)) * np.sqrt(2.0 / n_in)
            b = jnp.zeros((n_out,))
            params.append((W, b))
        return params

    def forward(params, x):
        """Forward pass: sequence of linear + relu layers."""
        for i, (W, b) in enumerate(params):
            x = x @ W + b
            if i < len(params) - 1:
                x = jnp.maximum(x, 0.0)   # relu (not on last layer)
        return x

    def loss_fn(params, x, y):
        """Cross-entropy loss (logits → loss)."""
        logits   = forward(params, x)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot  = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    # Compile the ENTIRE training step through XLA in one jit
    @jax.jit
    def train_step(params, x, y, lr=0.01):
        """
        One gradient descent step.
        jax.value_and_grad computes loss AND gradients in ONE XLA compilation.
        Forward + backward pass compiled into a single XLA executable.
        """
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        # SGD update
        new_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g,
            params, grads
        )
        return new_params, loss

    # Dataset: classify 2D points by quadrant
    key     = jax.random.PRNGKey(42)
    N_DATA  = 2000
    key, k1, k2 = jax.random.split(key, 3)
    X_data  = jax.random.normal(k1, (N_DATA, 4), jnp.float32)
    y_data  = (X_data[:, 0] > 0).astype(jnp.int32) * 2 + (X_data[:, 1] > 0).astype(jnp.int32)

    # Init params and warmup compile
    params = init_params(key, [4, 64, 64, 4])
    print("  Model: [4→64→64→4] MLP, 4-class classification")
    print(f"  Parameters: {sum(w.size + b.size for w, b in params):,}")
    print()

    # First call: compiles the XLA executable
    print("  First train_step call (triggers XLA compilation)...")
    t0 = time.perf_counter()
    X_b = X_data[:64]; y_b = y_data[:64]
    params, loss = train_step(params, X_b, y_b)
    jax.block_until_ready(params[0][0])   # wait for GPU computation
    t_compile = time.perf_counter() - t0
    print(f"    Compile + first step: {t_compile*1000:.1f} ms")
    print()

    # Subsequent calls: use cached XLA executable
    REPS   = 500
    t0     = time.perf_counter()
    losses = []
    for _ in range(REPS):
        idx    = np.random.choice(N_DATA, 64, replace=False)
        X_b    = X_data[idx];  y_b = y_data[idx]
        params, loss = train_step(params, X_b, y_b)
        losses.append(float(loss))
    jax.block_until_ready(params[0][0])
    t_train = (time.perf_counter() - t0) / REPS * 1000

    print(f"  Cached step time: {t_train:.4f} ms/step  ({REPS} steps)")
    print(f"  Loss trajectory: {losses[0]:.4f} → {losses[-1]:.4f}")
    print()
    print("  What XLA compiled into ONE executable:")
    print("    forward pass (4 matmuls + relu activations)")
    print("  + backward pass (4 vjp computations via jax.grad)")
    print("  + parameter update (SGD)")
    print("  = 1 XLA executable, launched once per step")
    print()

else:
    JAX_TRAINING_REF = """
  JAX+XLA TRAINING LOOP REFERENCE:

  import jax, jax.numpy as jnp

  # All functions are pure Python — no state!
  def forward(params, x):
      for W, b in params:
          x = jnp.maximum(x @ W + b, 0.0)
      return x

  def loss(params, x, y):
      return jnp.mean((forward(params, x) - y) ** 2)

  # Compile EVERYTHING with jit — forward + backward + update = 1 XLA program
  @jax.jit
  def step(params, x, y, lr=0.01):
      loss_val, grads = jax.value_and_grad(loss)(params, x, y)
      new_params = jax.tree_util.tree_map(lambda p, g: p - lr*g, params, grads)
      return new_params, loss_val

  # What XLA does:
  # 1. Trace forward() with abstract shapes → HLO for forward pass
  # 2. Compute VJPs (vector-Jacobian products) for each op → HLO for backward
  # 3. Append parameter update ops → complete HLO program
  # 4. Compile the full HLO → one GPU executable
  # 5. Cache and reuse for all subsequent steps with same shapes
"""
    print(JAX_TRAINING_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: pmap for multi-device data parallelism
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — pmap: multi-device data parallelism")
print("━" * 65)
print()

PMAP_REFERENCE = """
  pmap PATTERN — data parallel training across N GPUs:

  from jax import pmap
  from jax.lax import pmean

  @pmap(axis_name='batch')
  def parallel_train_step(params, x, y, lr=0.01):
      loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

      # Synchronise gradients across all devices via AllReduce
      grads = jax.lax.pmean(grads, axis_name='batch')

      new_params = jax.tree_util.tree_map(
          lambda p, g: p - lr * g, params, grads)
      return new_params, pmean(loss, axis_name='batch')

  # Usage:
  n_devices = jax.device_count()   # e.g., 8 GPUs

  # Replicate params to all devices
  params_rep = jax.tree_util.tree_map(
      lambda p: jnp.stack([p] * n_devices), params)

  # Data must be split across devices (first dim = device dim)
  X_split = X_batch.reshape(n_devices, -1, X_batch.shape[-1])
  y_split = y_batch.reshape(n_devices, -1)

  # Each device processes its shard, gradients auto-AllReduced
  params_rep, loss = parallel_train_step(params_rep, X_split, y_split)

  WHAT XLA DOES:
  1. Compile the function body for one device (one HLO program)
  2. pmean(grads, 'batch') → insert AllReduce(mean) HLO instruction
  3. Replicate and run on all N devices simultaneously
  4. AllReduce happens via NCCL (GPU) or gRPC (TPU interconnect)

  DIFFERENCE FROM PYTORCH DDP:
  PyTorch DDP: Python code runs on each process, DDP wrapper syncs grads
  JAX pmap:    XLA compiles ONE program with AllReduce BUILT IN to the HLO
               No Python per-step overhead; AllReduce is a compiled HLO op
"""
print(PMAP_REFERENCE)

if HAS_JAX:
    n_dev = len(jax.devices())
    if n_dev > 1:
        @partial(jax.pmap, axis_name='batch')
        def pmap_mean(x):
            return jax.lax.pmean(x, axis_name='batch')

        x_sharded = jnp.ones((n_dev,))
        result    = pmap_mean(x_sharded)
        print(f"  pmap pmean across {n_dev} devices: {result.tolist()}")
    else:
        print(f"  Only 1 device available ({jax.devices()[0]})")
        print(f"  Run on multi-GPU to see pmap parallelism.")
        print(f"  Single-device: jax.jit is equivalent to pmap(n=1)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: StableHLO export for portable deployment
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — StableHLO export: portable model deployment")
print("━" * 65)
print()

STABLEHLO_EXPORT = """
  StableHLO export allows saving a JAX computation in a stable,
  portable format that works across XLA versions and frameworks.

  ── Export ────────────────────────────────────────────────────────────

  import jax
  import jax.numpy as jnp

  def my_model(params, x):
      W1, b1, W2, b2 = params
      h = jnp.maximum(x @ W1 + b1, 0.0)   ; relu
      return h @ W2 + b2

  # Create abstract example inputs (shapes only, no real data)
  x_abstract = jax.ShapeDtypeStruct((1, 4), jnp.float32)

  # Export to StableHLO
  exported = jax.export.export(jax.jit(my_model))(params_abstract, x_abstract)

  # Serialize to bytes (can be saved to disk, shipped to another system)
  serialized = exported.serialize()
  with open("model.stablehlo", "wb") as f:
      f.write(serialized)

  ── Reload and Run ────────────────────────────────────────────────────

  # Load on any system with XLA (even different XLA version, 5 years later)
  with open("model.stablehlo", "rb") as f:
      loaded = jax.export.deserialize(f.read())

  # Run directly through XLA — no Python model code needed
  result = loaded.call(params, x_test)

  ── Inspect the StableHLO ─────────────────────────────────────────────

  # The exported object contains the StableHLO MLIR module:
  print(exported.mlir_module())
  # Output: standard MLIR text with stablehlo.* ops
  # Can be processed by:
  #   mlir-opt, IREE, TVM, any StableHLO consumer

  ── Cross-framework use ────────────────────────────────────────────────

  # Train in JAX, export to StableHLO, run with IREE on mobile:
  #   1. exported.serialize() → "model.stablehlo"
  #   2. ireecc compile --target=arm_64-android model.stablehlo -o model.vmfb
  #   3. Deploy model.vmfb to Android device

  # Train in JAX, export to StableHLO, run with TVM on custom hardware:
  #   1. exported.serialize() → "model.stablehlo"
  #   2. tvm.relay.from_stablehlo("model.stablehlo") → Relay
  #   3. relay.build(target="fpga") → compiled artifact
"""
print(STABLEHLO_EXPORT)

if HAS_JAX:
    # Demonstrate xla_computation (precursor to stable export)
    def simple_net(x, w, b):
        return jnp.maximum(x @ w + b, 0.0)

    x_ex = jnp.ones((4,), jnp.float32)
    w_ex = jnp.ones((4, 2), jnp.float32)
    b_ex = jnp.zeros((2,), jnp.float32)

    comp = jax.xla_computation(simple_net)(x_ex, w_ex, b_ex)
    hlo  = comp.as_hlo_text()
    n_instructions = len([l for l in hlo.split('\n')
                           if '= ' in l and 'ENTRY' not in l])

    print(f"  simple_net HLO ({n_instructions} instructions):")
    for line in hlo.split('\n')[:20]:
        print(f"    {line}")
    if len(hlo.split('\n')) > 20:
        print("    ...")
    print()

print("━" * 65)
print("  XLA QUICK REFERENCE")
print("━" * 65)
print()
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ XLA Feature         │ JAX API                                │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ JIT compilation     │ @jax.jit / jax.jit(fn)                 │")
print("  │ Gradient            │ jax.grad / jax.value_and_grad          │")
print("  │ Vectorisation       │ jax.vmap                               │")
print("  │ Multi-device        │ jax.pmap / jax.sharding                │")
print("  │ Inspect HLO         │ jax.xla_computation(fn)(args)          │")
print("  │ Export model        │ jax.export.export(jax.jit(fn))(args)   │")
print("  │ Force GPU           │ jax.device_put(x, jax.devices('gpu'))  │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ XLA_FLAGS dump HLO  │ XLA_FLAGS='--xla_dump_to=/tmp/xla'     │")
print("  │ XLA_FLAGS profile   │ XLA_FLAGS='--xla_hlo_profile=true'     │")
print("  │ XLA_FLAGS fusion    │ XLA_FLAGS='--xla_enable_hlo_passes=1'  │")
print("  └──────────────────────────────────────────────────────────────┘")
print()
print("  CONNECTED STACK POSITION:")
print("  LLVM (module 09)  ← XLA uses LLVM for CPU backend")
print("  MLIR (module 10)  ← XLA adopts StableHLO/MHLO as MLIR dialects")
print("  XLA  (this)       ← compiles JAX/TF programs to GPU/TPU")
print("  TVM  (next)       ← alternative compiler, converging via StableHLO")
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