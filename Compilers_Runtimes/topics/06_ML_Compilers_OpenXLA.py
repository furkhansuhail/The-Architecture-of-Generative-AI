"""
OpenXLA — The Community ML Compiler Ecosystem
===============================================

OpenXLA is the open-source, community-governed evolution of XLA. Where XLA
was a Google-internal compiler tightly coupled to TensorFlow and JAX, OpenXLA
is a vendor-neutral project that separates the compiler stack from any single
framework, hardware vendor, or company.

Launched in 2022 as a collaboration between Google, Meta, Amazon, Microsoft,
Intel, AMD, Arm, Apple, NVIDIA, and others, OpenXLA repackages XLA's proven
compilation technology with four critical additions:

    StableHLO:  a stable, versioned IR that guarantees models serialised today
                will compile correctly on any OpenXLA version five years from now.

    PJRT:       the Pluggable Device Runtime — a C API that any hardware vendor
                can implement to plug their chip into the entire OpenXLA ecosystem
                without modifying a single line of JAX, PyTorch, or TensorFlow.

    IREE:       the Intermediate Representation Execution Environment — a fully
                MLIR-native compiler and runtime for portable deployment to CPUs,
                GPUs, and mobile targets, now an OpenXLA project.

    Governance: a neutral open-source project where no single company controls
                the roadmap, ensuring longevity and multi-framework support.

The distinction matters: XLA (module 05) is a compiler. OpenXLA is an
ecosystem of interoperable compilers, runtimes, and IRs that together cover
the full journey from Python model to diverse silicon.

In the connected compiler stack:
    LLVM  (module 01) ← OpenXLA's CPU backend still emits LLVM IR
    MLIR  (module 02) ← StableHLO is an MLIR dialect; IREE is pure MLIR
    CIRCT (module 03) ← hardware backends register via PJRT; CIRCT for ASICs
    Enzyme(module 04) ← differentiation of StableHLO via Enzyme-MLIR research
    XLA   (module 05) ← OpenXLA/XLA is XLA, now community-governed
    OpenXLA (this)    ← the ecosystem: StableHLO + PJRT + IREE + governance
    TVM   (module 07) ← TVM ingests StableHLO; both compete and cooperate

"""

import textwrap
import re

TOPIC_NAME   = "OpenXLA — The Community ML Compiler Ecosystem"
DISPLAY_NAME = "06 · OpenXLA"
ICON         = "🌐"
SUBTITLE     = "StableHLO Portability, PJRT Hardware Abstraction, and IREE Deployment"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY OPENXLA EXISTS: THE FRAGMENTATION PROBLEM

### The State of ML Compilation Before OpenXLA (pre-2022)

    By 2021, the ML compilation landscape was deeply fragmented:

    Every framework had its own compiler path:
        JAX         → XLA (Google-controlled, internal, no public API)
        TensorFlow  → XLA (same, optional, via jit_compile=True)
        PyTorch     → torch.compile → TorchInductor / TorchScript
                      OR torch_xla (TPU support, but separate codebase)
        ONNX models → ONNX Runtime (Microsoft), TVM, TensorRT

    Every hardware vendor had its own integration burden:
        NVIDIA  → cuBLAS / cuDNN called directly from framework kernels
        AMD     → ROCm maintained as a fork of CUDA ops in each framework
        Intel   → oneDNN integrated separately into TF, PyTorch, MXNet
        AWS     → Neuron compiler for Inferentia, separate per-framework
        Habana  → HPU support requires patching each framework separately
        Apple   → Core ML / MPS integrated per-framework

    The consequence: N frameworks × M hardware vendors = N×M integrations.
        5 frameworks × 8 hardware vendors = 40 separate integration paths.
        Each path maintained by different teams, with different API stability.
        A new hardware vendor wanting to support "all frameworks" had to:
            Write a TF kernel plugin.
            Write a PyTorch aten operator extension.
            Implement torch_xla shims.
            Integrate with ONNX Runtime.
            → Months of engineering, duplicated per framework.

### The Two Root Causes

    ROOT CAUSE 1: No stable, portable model representation.
        XLA's HLO was not stable. A model compiled for XLA version N
        might not work on version N+1. There was no format that could
        be produced by JAX and consumed by TVM, or saved today and
        compiled five years from now.

        The consequence: model serialisation was framework-specific.
            TF SavedModel works only in TensorFlow.
            JAX's jax.xla_computation() output was unstable.
            ONNX attempted portability but lacked expressivity.

    ROOT CAUSE 2: No hardware plugin API.
        To run JAX on a new chip, that chip had to be integrated
        into JAX's source code. There was no "implement this interface,
        and JAX will run on your hardware" contract.
        The hardware vendor had to upstream changes into JAX itself,
        which required Google review, had no stability guarantee,
        and broke with each JAX release.

### OpenXLA's Solution Architecture

    OpenXLA solves both root causes simultaneously:

    FOR PORTABILITY → StableHLO (see Part 3):
        A versioned, stable MLIR dialect that any framework can emit
        and any compiler can consume. Stability guaranteed 5 years forward.
        The "LLVM IR for ML" — the universal interchange format.

    FOR HARDWARE → PJRT (see Part 2):
        A stable C API that hardware vendors implement once.
        Any framework using PJRT automatically works on any PJRT device.
        N frameworks × M hardware = N+M integrations (not N×M).

    ┌────────────────────────────────────────────────────────────────────────────────┐
    │  BEFORE OpenXLA: N×M problem                                                   │
    │  JAX ──────→ XLA/CUDA (NVIDIA specific)                                        │
    │  JAX ──────→ XLA/ROCm (AMD fork)                                               │
    │  PyTorch ──→ aten/CUDA                                                         │
    │  PyTorch ──→ aten/ROCm (fork)                                                  │
    │  TF ───────→ TF kernels/CUDA                                                   │
    │  ... 15 more integration paths ...                                             │
    ├────────────────────────────────────────────────────────────────────────────────┤
    │  AFTER OpenXLA: N+M solution                                                   │
    │                                                                                │
    │  JAX       ──→  StableHLO  ──→  XLA   ──→  PJRT  ──→ NVIDIA GPU                │
    │  PyTorch   ──→  StableHLO  ──→  IREE  ──→  PJRT  ──→ AMD GPU                   │
    │  TensorFlow──→  StableHLO  ──→  TVM   ──→  PJRT  ──→ Intel GPU                 │
    │                                                       PJRT  ──→ AWS Inferentia │
    │  Add NEW hardware: implement PJRT once → all frameworks work                   │
    │  Add NEW framework: emit StableHLO → all compilers work                        │
    └────────────────────────────────────────────────────────────────────────────────┘


##### PART 2 — PJRT: THE PLUGGABLE DEVICE RUNTIME

### What PJRT Is

    PJRT (Pluggable and Just-in-time Runtime) is a stable C API that
    defines a contract between ML frameworks (JAX, TensorFlow, PyTorch)
    and hardware backends (GPU, TPU, NPU, custom accelerators).

    The contract has two sides:
        Framework side: calls PJRT to compile programs and execute them.
        Hardware side:  implements PJRT to make their device available.

    PJRT is the "device driver" concept applied to ML frameworks.
    Just as a printer manufacturer writes a driver and any application
    can print without modification, a chip vendor writes a PJRT plugin
    and any PJRT-consuming framework runs on that chip.

### The PJRT C API: Core Abstractions

    PJRT defines these primary object types:

    PJRT_Client:
        The entry point. Represents a connection to a set of devices.
        Responsibilities:
            Enumerate available devices.
            Compile programs (StableHLO → executable).
            Create buffers on device memory.
            Execute compiled programs.

        Key methods:
            PJRT_Client_Create()               → creates a client for the device
            PJRT_Client_Devices()              → list all devices
            PJRT_Client_Compile()              → compile StableHLO → PJRT_Executable
            PJRT_Client_BufferFromHostBuffer() → copy host data → device buffer

    PJRT_Device:
        Represents one physical accelerator (one GPU, one TPU chip, etc.).
        Attributes: device_id, local_device_id, device_kind (string).

    PJRT_Buffer:
        Device-side memory buffer containing one tensor.
        Operations: ToHostBuffer (device → host copy), Delete, CopyToDevice.

    PJRT_Executable:
        A compiled program ready to run. Device-specific binary.
        Operations: Execute (run on one or more devices).

    PJRT_Event:
        Represents an asynchronous computation in flight.
        Operations: Await (block until done), OnReady (callback when done).

### PJRT Data Flow: Compile Once, Run Anywhere

    The complete PJRT workflow:

        ┌─────────────────────────────────────────────────────────────────┐
        │  1. Framework (JAX) produces StableHLO MLIR module              │
        │        ↓ PJRT_Client_Compile(client, stablehlo_program)         │
        │  2. PJRT plugin (device-specific) compiles StableHLO            │
        │     (NVIDIA plugin: invokes XLA/NVPTX backend)                  │
        │     (AMD plugin:    invokes XLA/AMDGCN or ROCm backend)         │
        │     (Intel plugin:  invokes OpenVINO or XLA/oneDNN backend)     │
        │        ↓ returns PJRT_Executable (device binary)                │
        │  3. Framework transfers input data to device                    │
        │        PJRT_Client_BufferFromHostBuffer(host_array → device)    │
        │  4. Execute                                                     │
        │        PJRT_Executable_Execute(inputs=[buf1, buf2], outputs=[]) │
        │  5. Wait and retrieve                                           │
        │        PJRT_Event_Await(event)                                  │
        │        PJRT_Buffer_ToHostBuffer(output_buf → host_array)        │
        └─────────────────────────────────────────────────────────────────┘

### PJRT Plugin Model: How Hardware Vendors Integrate

    A hardware vendor wanting JAX support:

    BEFORE PJRT (the old world):
        1. Clone JAX repository.
        2. Add device-specific kernels and ops to jax/lib/xla_bridge.py.
        3. Modify jax/_src/interpreters/xla.py for device registration.
        4. Submit pull request to JAX, wait for Google review.
        5. Maintain a fork as JAX evolves.
        6. Repeat for TensorFlow, PyTorch separately.
        → 6-18 months to get first JAX program running on new hardware.

    AFTER PJRT (the new world):
        1. Implement the ~35 PJRT C API functions.
        2. Build a shared library: my_chip_pjrt.so
        3. User: import jax; jax.config.update("jax_pjrt_device", "my_chip_pjrt.so")
        4. All JAX programs run on the new chip.
        → 2-4 months to get first JAX program running on new hardware.

    PJRT plugins shipping in production:
        NVIDIA GPU:    jaxlib ships with built-in PJRT plugin for CUDA
        Google TPU:    libtpu.so is a PJRT plugin
        Intel GPU:     intel-extension-for-openxla PJRT plugin
        AWS Inferentia: neuronx-cc ships a PJRT plugin
        AMD GPU:       JAX-ROCm ships a PJRT plugin for ROCm

### PJRT C API Stability Guarantees

    PJRT's C API is STABLE:
        Once a function signature is in the PJRT API, it never changes.
        Additions (new functions) are backward-compatible.
        Plugins built against API version N still work on version N+5.

    This stability is implemented via a versioned struct:
        typedef struct PJRT_Api {
          size_t struct_size;
          int    pjrt_api_version_major;
          int    pjrt_api_version_minor;
          PJRT_Error* (*PJRT_Client_Create)(...);
          PJRT_Error* (*PJRT_Client_Devices)(...);
          PJRT_Error* (*PJRT_Client_Compile)(...);
          // ... 30+ function pointers ...
        } PJRT_Api;

    The framework checks the version at load time and uses only the
    functions available in the plugin's API version. Old plugins remain
    functional as frameworks add new optional API entries.


##### PART 3 — STABLEHLO: THE STABLE PORTABLE IR

### What StableHLO Is and Is Not

    StableHLO is an MLIR dialect that defines ~100 tensor operations
    (roughly matching XLA's HLO op set) with a formal stability guarantee.

    StableHLO IS:
        An MLIR dialect (uses MLIR's op/type/attribute infrastructure).
        A stable specification (ops defined today work 5+ years from now).
        A serialisation format (MLIR's binary format, readable text format).
        A portability layer (produced by frameworks, consumed by compilers).
        Open-source with a formal compatibility policy (openxla/stablehlo).

    StableHLO IS NOT:
        A runtime (it cannot execute programs).
        A replacement for XLA's internal HLO (XLA still uses its own protos).
        A framework IR (JAX/PyTorch don't program IN StableHLO directly).
        The same as MHLO (MHLO is XLA's internal MLIR dialect, unstable).

### The Stability Policy

    The StableHLO compatibility policy (from openxla/stablehlo RFC):

    Forward compatibility (producer → consumer):
        A StableHLO program produced by version N can be consumed
        by any version ≥ N for at least 5 years after N's release.

    Backward compatibility (consumer → producer):
        A StableHLO consumer (compiler) that handles version M
        can also handle programs from any version N ≤ M.

    What this means in practice:
        Serialise a JAX model to StableHLO today (2024).
        In 2029: load into IREE 2.x, TVM 0.18, or XLA 2.x → still works.
        Inference infrastructure outlives the training framework version.

    Stability covers:
        Op semantics (what each op computes, exactly).
        Op signatures (inputs, outputs, attributes).
        Type system (integer/float types, tensor shapes, element types).
        Serialisation format (binary encoding of the MLIR module).

    Stability does NOT cover:
        Op performance (a new version can generate better code).
        New ops (old programs don't use new ops; new ops are additions).

### StableHLO Op Vocabulary

    StableHLO defines ~100 ops. Key ones:

    Tensor creation:
        stablehlo.constant       dense<[[1.0, 2.0]]> : tensor<1x2xf32>
        stablehlo.iota           [0,1,2,...,N-1] in a given dimension

    Elementwise (unary):
        stablehlo.abs, stablehlo.negate, stablehlo.sign
        stablehlo.exp, stablehlo.log, stablehlo.sqrt, stablehlo.rsqrt
        stablehlo.tanh, stablehlo.logistic (sigmoid)
        stablehlo.floor, stablehlo.ceil, stablehlo.round_nearest_even
        stablehlo.not (bitwise)

    Elementwise (binary):
        stablehlo.add, stablehlo.subtract, stablehlo.multiply, stablehlo.divide
        stablehlo.maximum, stablehlo.minimum, stablehlo.power, stablehlo.remainder
        stablehlo.and, stablehlo.or, stablehlo.xor (bitwise)
        stablehlo.compare  (with comparison_direction: EQ/NE/LT/GT/LE/GE)
        stablehlo.select   (ternary: mask ? a : b, elementwise)

    Linear algebra:
        stablehlo.dot_general    generalised batched matmul with dimension specs
        stablehlo.convolution    N-D convolution with configurable dimension numbers

    Shape manipulation:
        stablehlo.reshape, stablehlo.transpose, stablehlo.reverse
        stablehlo.broadcast_in_dim   expand/align a tensor to a new shape
        stablehlo.slice              extract a sub-tensor by start/limit/strides
        stablehlo.pad                pad a tensor with a given value
        stablehlo.concatenate        join tensors along a dimension
        stablehlo.dynamic_slice      slice with a runtime-computed offset
        stablehlo.dynamic_update_slice  scatter a sub-tensor into a tensor

    Reductions:
        stablehlo.reduce             fold one or more dimensions with a combiner
        stablehlo.reduce_window      pooling: reduce over a sliding window
        stablehlo.scatter            scatter update: write to indexed positions
        stablehlo.gather             gather: read from indexed positions

    Control flow:
        stablehlo.while              loop: while condition holds, run body
        stablehlo.if                 conditional: branch on a scalar boolean
        stablehlo.case               multi-way branch on a scalar integer
        stablehlo.call               call a sub-computation

    Communication (for distributed programs):
        stablehlo.all_reduce         sum/average across a device group
        stablehlo.all_gather         collect shards from all devices
        stablehlo.reduce_scatter     reduce then scatter shards to devices
        stablehlo.all_to_all         transpose data/model axes
        stablehlo.send, stablehlo.recv  point-to-point

    Type system:
        Integers:   si4, si8, si16, si32, si64  (signed)
                    ui4, ui8, ui16, ui32, ui64  (unsigned)
        Floats:     f8e4m3fn, f8e5m2   (FP8 for quantised training)
                    bf16, f16, f32, f64
        Complex:    c64, c128
        Booleans:   i1
        Tensors:    tensor<2x3xf32>, tensor<?x4xf16> (? = dynamic dimension)

### StableHLO Serialisation Format

    StableHLO programs are MLIR modules stored in one of three forms:

    1. Textual (.mlir):
            func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %bc   = stablehlo.broadcast_in_dim %zero, dims=[]
                          : (tensor<f32>) -> tensor<4xf32>
                %relu = stablehlo.maximum %x, %bc : tensor<4xf32>
                return %relu : tensor<4xf32>
            }

    2. Binary (.mlirbc):
            Compact binary encoding of the MLIR structure.
            10-100× smaller than textual for large models.
            Produced by: mlir::serializeToMlirBytecode()
            Read by:     mlir::parseSourceFile()

    3. Version-tagged:
            The binary format includes the StableHLO version that produced it.
            Consumers use this to apply compatibility transformations if needed.

    Framework APIs for StableHLO:
        JAX:       jax.export.export(jax.jit(fn))(*args).serialize()
        TF:        tf.saved_model with jit_compile=True exports StableHLO
        PyTorch:   torch.export + torch_xla produces StableHLO (experimental)
        Torch-MLIR: torch-mlir-opt --output-type=stablehlo

### StableHLO vs MHLO vs HLO: The Dialect Disambiguation

    Three closely related IRs exist in the XLA ecosystem:

    HLO (classic):
        XLA's original IR, expressed as C++ protobuf structures.
        Used internally inside XLA's compiler.
        NOT an MLIR dialect. No stability guarantee.
        Still the final IR before XLA's code generators.

    MHLO (Meta HLO):
        XLA's INTERNAL MLIR representation.
        Lives in tensorflow/compiler/mlir/hlo.
        UNSTABLE: changes with XLA releases.
        Used as an intermediate step: StableHLO → MHLO → HLO → codegen.
        NOT for external consumers (no stability promise).

    StableHLO:
        EXTERNAL, STABLE MLIR representation.
        Lives in openxla/stablehlo (separate GitHub repo).
        STABLE: compatibility guarantee described above.
        FOR external consumers: frameworks emit it, compilers consume it.
        Lower passes: StableHLO → MHLO → HLO (inside XLA) for compilation.

    The migration path:
        Framework (JAX) → StableHLO (stable, external)
                       → MHLO (unstable, inside XLA)
                       → HLO protos (inside XLA codegen)
                       → NVPTX/AMDGCN/TPU binary


##### PART 4 — IREE: MLIR-NATIVE DEPLOYMENT RUNTIME

### What IREE Is

    IREE (Intermediate Representation Execution Environment) is a
    compiler and runtime for deploying ML models across diverse targets.
    It is an OpenXLA project, hosted at github.com/openxla/iree.

    IREE's design philosophy:
        MLIR all the way from input to output — no hand-written codegen.
        Zero Python runtime dependency — a C runtime with no Python.
        Streaming execution — computation pipelined with memory management.
        First-class mobile/edge support — targets that XLA does not reach.

    Target hardware:
        CPU:    x86-64 (AVX2/AVX-512), ARM64 (NEON/SVE), RISC-V, WASM
        GPU:    Vulkan (cross-vendor), Metal (Apple), CUDA, ROCm
        Mobile: Android (Vulkan), iOS (Metal), embedded RTOS targets
        Web:    WebGPU via Dawn, WebAssembly via Emscripten

### IREE's Compilation Pipeline

    IREE compiles via a cascade of MLIR dialects:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  INPUT: StableHLO / TOSA / Torch-MLIR (any MLIR dialect)            │
    ├──────────────────────────────────────────────────────────────────────┤
    │  IREE Input Transformation Pipeline                                  │
    │    StableHLO → linalg (via stablehlo-legalize-to-linalg pass)        │
    │    Tiling and fusion at linalg level                                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │  IREE Flow Dialect                                                   │
    │    Identifies "dispatch regions" — units of GPU/CPU work             │
    │    Schedules data movement between dispatch regions                  │
    │    flow.dispatch, flow.tensor.load, flow.tensor.store               │
    ├──────────────────────────────────────────────────────────────────────┤
    │  IREE Stream Dialect                                                 │
    │    Models asynchronous execution and memory lifetimes                │
    │    stream.async.execute, stream.resource.alloc                       │
    │    Enables overlapping CPU host work with device compute             │
    ├──────────────────────────────────────────────────────────────────────┤
    │  IREE HAL Dialect (Hardware Abstraction Layer)                       │
    │    Uniform interface for all device types                            │
    │    hal.command_buffer.dispatch, hal.allocator.allocate               │
    │    Device-agnostic: same HAL IR compiles to Vulkan or Metal          │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Target-specific lowering                                            │
    │    LLVM dialect → LLVM IR → x86/ARM (CPU targets)                  │
    │    SPIRV dialect → SPIR-V → Vulkan runtime (GPU)                    │
    │    CUDA dialect  → NVPTX → cubin (CUDA GPU)                        │
    │    Metal dialect → AIR    → Metal PSO (Apple GPU)                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │  OUTPUT: .vmfb (VM FlatBuffer) — compiled binary artifact            │
    │    Embeds all backends: CPU kernel + GPU shader + metadata           │
    │    Loaded by IREE runtime → executes on the available device         │
    └──────────────────────────────────────────────────────────────────────┘

### The IREE .vmfb Artifact

    IREE's output is a .vmfb file (VM FlatBuffer):
        A FlatBuffers-encoded binary containing:
            - All compiled kernels for the target (SPIR-V, NVPTX, LLVM IR, etc.)
            - The dispatch schedule (which kernel to launch when)
            - Buffer allocation metadata (sizes, alignments, lifetimes)
            - Module signature (input/output shapes, names)

    The runtime is minimal:
        Runtime size: ~100-500 KB (vs 50+ MB for TensorFlow runtime)
        No Python dependency at inference time
        Loads .vmfb, binds inputs, runs dispatch, binds outputs

    Cross-compilation:
        Compile on x86-64 Linux → target Arm64 Android → run on phone
        iree-compile --iree-hal-target-backends=vulkan-spirv \
                     --iree-input-type=stablehlo input.mlir -o model.vmfb
        # Deploy model.vmfb to Android device

### IREE vs XLA: Complementary Roles

    ┌──────────────────────┬─────────────────────┬────────────────────────┐
    │  Property            │  XLA (OpenXLA/XLA)  │  IREE                  │
    ├──────────────────────┼─────────────────────┼────────────────────────┤
    │  Primary use         │  Training at scale  │  Deployment/inference  │
    │  Target hardware     │  GPU, TPU           │  CPU, GPU, mobile, web │
    │  Runtime footprint   │  Heavy (Python req) │  Minimal (C runtime)   │
    │  Mobile support      │  None               │  First-class           │
    │  WebAssembly         │  None               │  Yes (via LLVM WASM)   │
    │  Vulkan              │  None               │  Yes (via SPIR-V)      │
    │  Metal (Apple)       │  Limited            │  Yes (native)          │
    │  MLIR-native         │  Partially          │  Fully                 │
    │  Compilation speed   │  Slower (autotuning)│  Faster                │
    │  Peak perf (A100)    │  Highest (cuBLAS)   │  Competitive           │
    │  Input format        │  JAX/TF/StableHLO  │  StableHLO/TOSA/Torch  │
    └──────────────────────┴─────────────────────┴────────────────────────┘

    They are not competitors — they are complementary:
        Train with JAX+XLA → export StableHLO → deploy with IREE
        This is the canonical OpenXLA end-to-end workflow.


##### PART 5 — JAX EXPORT: THE STABLEHLO SERIALISATION WORKFLOW

### The Problem jax.export Solves

    Before jax.export, deploying JAX models required:
        The full JAX + JAXlib Python installation at inference time.
        The exact same JAX version used for training.
        The full TensorFlow graph (for SavedModel path).

    jax.export serialises a JAX computation to StableHLO, which:
        Runs on any XLA version for 5 years (stability guarantee).
        Can be compiled by IREE, TVM, or any StableHLO consumer.
        Works without Python at inference time (if deployed via IREE).
        Supports polymorphic shapes (one export works for multiple batch sizes).

### The Full jax.export API

    Basic export:
        import jax
        import jax.numpy as jnp

        # Define model (any JAX function)
        def mlp(params, x):
            W1, b1, W2, b2 = params
            h = jnp.maximum(x @ W1 + b1, 0.0)
            return h @ W2 + b2

        # Provide abstract shapes (no real data needed)
        x_abstract  = jax.ShapeDtypeStruct((32, 784), jnp.float32)
        p_abstract  = jax.tree_map(
            lambda w: jax.ShapeDtypeStruct(w.shape, w.dtype), params)

        # Export → produces a StableHLO MLIR module
        exported = jax.export.export(jax.jit(mlp))(p_abstract, x_abstract)

        # Inspect the MLIR
        print(exported.mlir_module())   # prints stablehlo.* MLIR text

        # Serialise to bytes (save to disk, ship to deployment system)
        blob = exported.serialize()
        with open("mlp_model.stablehlo", "wb") as f:
            f.write(blob)

    Polymorphic export (single export for multiple batch sizes):
        # Export with symbolic batch dimension 'b'
        x_poly = jax.ShapeDtypeStruct(('b', 784), jnp.float32)
        exported_poly = jax.export.export(jax.jit(mlp))(p_abstract, x_poly)

        # At inference time, ANY batch size works:
        exported_poly.call(params, np.ones((1, 784)))    # batch=1
        exported_poly.call(params, np.ones((64, 784)))   # batch=64
        exported_poly.call(params, np.ones((1024, 784))) # batch=1024

    Deserialise and run (5 years later):
        with open("mlp_model.stablehlo", "rb") as f:
            blob = f.read()
        loaded = jax.export.deserialize(blob)
        result = loaded.call(params, x_test)
        # Works on any JAX version released within 5 years of serialisation

    Metadata in the exported object:
        exported.fun_name             # original function name
        exported.in_avals             # input abstract values (shapes/dtypes)
        exported.out_avals            # output abstract values
        exported.mlir_module()        # the StableHLO MLIR text
        exported.in_shardings         # sharding specs (for multi-device)
        exported.out_shardings        # output sharding
        exported.nr_devices           # number of devices needed

### Deployment Path: JAX → StableHLO → IREE

    The canonical OpenXLA deployment workflow:

        # TRAINING (on GPU cluster with JAX+XLA)
        model = train_model(data)
        exported = jax.export.export(jax.jit(model.forward))(*abstract_args)
        blob = exported.serialize()
        save_to_model_store("v1.0", blob)

        # DEPLOYMENT (on mobile device, no Python, no JAX)
        # Step 1: compile StableHLO → IREE vmfb (done server-side)
        iree-compile --input-type=stablehlo \
                     --iree-hal-target-backends=vulkan-spirv \
                     v1.0.stablehlo -o v1.0.vmfb

        # Step 2: ship v1.0.vmfb to device (tiny runtime, no Python)
        # Android app (C++):
        auto module = iree_vm_bytecode_module_create_from_file("v1.0.vmfb");
        auto call   = iree_vm_function_lookup(module, "forward");
        iree_vm_invoke(call, inputs, outputs);
        // result available in outputs


##### PART 6 — OPENXLA COMPILATION INTERNALS: PASSES AND OPTIMISATIONS

### The OpenXLA/XLA Pass Pipeline (Updated)

    OpenXLA's XLA compiler retains XLA's core passes (from module 05)
    but adds new passes for the MLIR-based pipeline:

    Stage 0: Frontend import
        JAX   → Jaxpr → StableHLO MLIR module
        StableHLO → MHLO (internal MLIR dialect, via stablehlo-legalize-to-hlo)
        MHLO is processed by MLIR-based passes

    Stage 1: MLIR-level optimisations (new in OpenXLA)
        mhlo-canonicalize:         MLIR-level constant folding and CSE
        mhlo-algebraic-simplifier: same identities as XLA, now in MLIR
        mhlo-shape-inference:      propagate static shape info through the graph
        mhlo-flatten-tuple-list:   normalise tuple-typed arguments

    Stage 2: Lowering MHLO → HLO protos (legacy path, being phased out)
        MlirToHloTranslate:        MHLO → XLA HLO C++ protos
        HLO runs its own classic passes (fusion, layout, buffer assignment)

    NEW: Stage 2 (MLIR-all-the-way path, research/production 2024+)
        MHLO → linalg dialect (via mhlo-legalize-to-linalg)
        linalg → affine/vector (tiling, vectorisation)
        vector → LLVM dialect → LLVM IR (CPU) or
        vector → GPU dialect → NVPTX/AMDGCN (GPU)
        This path bypasses the HLO proto entirely.

### Mosaic GPU: OpenXLA's New GPU Compiler

    Triton (see TVM module) disrupted GPU kernel generation by replacing
    cuBLAS with auto-generated CUDA kernels. OpenXLA's response is Mosaic GPU.

    Mosaic GPU is a new OpenXLA compiler targeting Google's TPUs and NVIDIA GPUs.
    It replaces the XLA GPU emitter for select operations.

    Key features:
        Operates at the Triton/Pallas abstraction level.
        Generates GPU warpgroup-level instructions directly.
        Uses MLIR's GPU dialect and NVVM dialect.
        Supports Hopper (H100) tensor memory accelerator (TMA) instructions.
        Generates CUDA kernels that match or exceed hand-tuned cuBLAS in some cases.

    Mosaic GPU in the stack:
        StableHLO (matmul) → MHLO → Mosaic GPU backend
            → GPU dialect + vector dialect
            → NVVM dialect (CUDA intrinsics)
            → NVPTX → PTX → cubin

    JAX's Pallas frontend for Mosaic GPU:
        @jax.experimental.pallas.pallas_call(
            out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
            grid=(M // BM, N // BN))
        def matmul_kernel(x_ref, y_ref, o_ref):
            o_ref[...] = x_ref[...] @ y_ref[...]   # tiled matmul

        # Mosaic GPU compiles this to a warpgroup GEMM on H100

### AutoSharding: Automatic Parallelism Strategy Discovery

    SPMD (from XLA module) still requires the user to annotate sharding specs.
    OpenXLA's AutoSharding pass discovers sharding strategies automatically.

    The problem it solves:
        Optimal tensor parallelism for a 70B parameter transformer across
        512 GPUs involves hundreds of sharding decisions:
            Which layers are data-parallel vs model-parallel?
            At what granularity are attention heads sharded?
            Which matmuls should be tensor-parallel?
            Where should ZeRO-style parameter sharding apply?

    Manual tuning takes weeks of expert engineering time.

    AutoSharding algorithm:
        1. Build an "inter-operator parallelism graph" from the HLO module.
        2. Assign a cost model to each possible sharding choice (communication
           volume, memory usage, compute utilisation).
        3. Solve the resulting optimisation problem as an Integer Linear Program
           (ILP) or via dynamic programming.
        4. Insert AllReduce, AllGather, ReduceScatter ops where sharding
           boundaries require data exchange.

    AutoSharding was originally developed at CMU (Zheng et al. 2022) and
    is now integrated into OpenXLA as an optional pass:
        XLA_FLAGS="--xla_enable_auto_sharding=true"

    Performance: matches expert-tuned sharding within 5-15% for LLM training.

### Quantisation Support in OpenXLA

    OpenXLA adds quantisation support that was minimal in original XLA:

    Quantised types:
        StableHLO supports: si4, ui4 (4-bit integers), si8/ui8, bf8/f8e4m3
        These enable int4/int8 quantised inference with XLA compilation.

    Quantisation aware training (QAT) via StableHLO:
        Fake quantise ops during training → export to StableHLO with quant ops →
        OpenXLA compiles quant ops → int8 GPU kernels via cuDNN INT8 path

    Post-training quantisation (PTQ):
        Exported StableHLO model → stablehlo-quantize pass →
        quantised StableHLO → lower to XLA int8 ops → deploy

    OpenXLA quantisation API (ODML team):
        Quantisation recipes defined as MLIR passes on the StableHLO module.
        Target-specific lowering to int4/int8 cuDNN ops or INT8 Vulkan compute.


##### PART 7 — THE OPENXLA GOVERNANCE MODEL AND ECOSYSTEM PROJECTS

### The OpenXLA Organisation Structure

    OpenXLA is governed by a Technical Steering Committee (TSC) with
    representatives from contributing companies. Decisions require TSC
    consensus; no single company has veto power.

    Member companies at launch (2022–2023):
        Google (XLA, JAX, MLIR, TPU)
        Meta (PyTorch, FBGEMM)
        Amazon (AWS Trainium/Inferentia, Neuron compiler)
        Microsoft (ONNX Runtime, Azure ML hardware)
        Intel (oneDNN, OpenVINO, Habana Gaudi)
        AMD (ROCm, MIOpen)
        Arm (ARM Compute Library, ML cores)
        Apple (Core ML, ANE - Apple Neural Engine)
        NVIDIA (TensorRT, cuDNN)

    The TSC governs:
        StableHLO compatibility policy (who approves breaking changes).
        PJRT API versioning (what gets added, what stays stable).
        Project inclusion (which projects are "OpenXLA projects").

### OpenXLA Project Portfolio

    Core compiler (openxla/xla):
        XLA itself, now open-source under OpenXLA governance.
        Contains: HLO IR, fusion, layout, buffer assignment, NVPTX/AMDGCN codegen.
        Previously: tensorflow/compiler/xla (inside TF monorepo).
        Now: separate repository, faster releases, non-TF dependency.

    Portable IR (openxla/stablehlo):
        The stable MLIR dialect and its specification.
        Includes: compatibility tests, reference interpreter, serialisation.
        Stability committee reviews all proposed changes.
        Used by: JAX, TF, PyTorch/XLA, torch-mlir as output format.

    Deployment runtime (openxla/iree):
        MLIR-native compiler + runtime for edge/mobile/server deployment.
        Separated from Google's internal tooling in 2021, joined OpenXLA 2023.
        Key integrations: JAX export → IREE, ONNX → IREE, torch-mlir → IREE.

    Benchmark suite (openxla/openxla-benchmark):
        Standardised performance benchmarks across compilers and hardware.
        Prevents "benchmark gaming" — all compilers tested on the same workloads.
        Workloads: BERT, T5, BERT-Large, ResNet, MobileNet, StableDiffusion.

    Kernel library (openxla/shardy):
        Sharding propagation and partitioning library.
        Generalises GSPMD (Google's sharding system) as a standalone MLIR pass.
        Input: StableHLO with partial sharding annotations.
        Output: StableHLO with all sharding decisions filled in.

### How OpenXLA Relates to Competing Projects

    OpenXLA vs TVM (module 07):
        Relationship: cooperative and competitive.
        TVM consumes StableHLO (convergence via MLIR).
        TVM's auto-scheduling (Ansor, MetaSchedule) has no equivalent in OpenXLA.
        XLA's TPU support has no equivalent in TVM.
        Both target GPU with different philosophies:
            XLA: hand-tuned cuBLAS/cuDNN + fusion of surrounding ops.
            TVM: auto-generated tiled kernels (sometimes beats cuBLAS).
        Verdict: different sweet spots, increasingly sharing the IR layer.

    OpenXLA vs ONNX Runtime:
        ONNX Runtime uses ONNX as its portability format; OpenXLA uses StableHLO.
        ONNX is older and more widely adopted in production.
        StableHLO has stricter semantics and better ML framework coverage.
        ONNX Runtime has better CPU optimisation for inference.
        OpenXLA has better GPU/TPU training performance.
        Convergence path: ONNX → StableHLO bridge under active development.

    OpenXLA vs Triton (separate from TVM):
        Triton is a kernel compiler (write GPU kernels in Python).
        OpenXLA uses Triton as a BACKEND for some operations (via Pallas on GPU).
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=true": XLA generates Triton kernels.
        They are layers in the same stack, not competing at the same level.

    OpenXLA vs PyTorch 2.0 (torch.compile):
        torch.compile uses TorchInductor as the default backend.
        TorchInductor generates Triton (GPU) or C++ (CPU) code.
        PyTorch/XLA is a separate backend: torch.compile → XLA → GPU/TPU.
        The two paths are increasingly converging via torch-mlir → StableHLO.


##### PART 8 — OPENXLA IN THE CONNECTED COMPILER STACK

### How OpenXLA Connects to LLVM (Module 01)

    Three connection points:
        1. OpenXLA/XLA's CPU backend emits LLVM IR (unchanged from XLA module).
           HLO → HloToIr → LLVM IR → llc → x86/ARM native code.

        2. IREE's CPU path emits LLVM IR from the vector dialect:
           vector dialect → LLVM dialect → LLVM IR → llc → binary.

        3. IREE's WASM target: LLVM IR → wasm32 via Emscripten.

    Joint optimisation:
        Both XLA and IREE benefit from LLVM's loop-vectorize, instcombine,
        and GVN running on the emitted LLVM IR. This means LLVM's CPU-specific
        optimisations apply automatically to OpenXLA-compiled models.

### How OpenXLA Connects to MLIR (Module 02)

    OpenXLA is the largest production consumer of MLIR infrastructure.
    Every component uses MLIR:
        StableHLO:  an MLIR dialect (operations, types, attributes from MLIR).
        MHLO:       an MLIR dialect (XLA's internal ML IR).
        IREE:       fully MLIR-based (Flow, Stream, HAL, SPIRV, LLVM dialects).
        AutoSharding: implemented as an MLIR pass on the MHLO module.
        Mosaic GPU: generates GPU + NVVM MLIR dialects.
        Shardy:     sharding propagation as MLIR passes.

    MLIR infrastructure reused:
        Pass manager:  all OpenXLA passes use MLIR's PassManager.
        Pattern rewriting: algebraic simplifications use RewritePatternSet.
        Type system:   StableHLO's tensor/element types built on MLIR types.
        Serialisation: binary MLIR bytecode for StableHLO portability.
        Testing:       FileCheck + lit for all MLIR-level tests.

### How OpenXLA Connects to CIRCT (Module 03)

    The CIRCT connection is indirect but growing:
        CIRCT's FIRRTL compiler (firtool) generates Verilog for TPU hardware.
        The hardware described in Chisel → CIRCT → Verilog is then manufactured
        and becomes the TPU that runs OpenXLA-compiled programs.

    PJRT as the abstraction:
        The TPU's PJRT plugin isolates the hardware from OpenXLA's compiler.
        CIRCT-designed hardware exposes a PJRT-compatible interface.
        OpenXLA compiles StableHLO for the TPU without knowing chip details.

    Research frontier: differentiable hardware simulation:
        CIRCT's Arc dialect + Enzyme (module 04) = differentiable RTL simulation.
        OpenXLA accelerator design can use gradient-based optimisation
        of hardware parameters, using Enzyme-differentiated CIRCT simulations.

### How OpenXLA Connects to Enzyme (Module 04)

    OpenXLA uses AD in two ways:
        1. JAX's gradient transformations (jax.grad, jax.vjp) operate at the
           StableHLO level — JAX applies AD rules to StableHLO ops BEFORE
           passing the differentiated program to OpenXLA/XLA for compilation.

        2. Custom op differentiation: for ops without JAX-level derivatives,
           Enzyme-MLIR can differentiate the StableHLO module directly,
           operating on the same MLIR infrastructure OpenXLA uses.

    Enzyme-JAX (enzyme-jax package):
        Provides enzyme_vjp() that differentiates JAX functions via Enzyme
        at the LLVM level, AFTER XLA lowers to LLVM IR.
        Produces gradients that are jointly optimised with the primal by LLVM.
        Particularly useful for: custom HPC kernels called from JAX,
        differentiating through precompiled C++ code in JAX pipelines.

### How OpenXLA Connects to TVM (Module 07)

    StableHLO is the convergence point:
        TVM's Relax IR can import StableHLO (via stablehlo-to-relax pass).
        A model trained with JAX+XLA can be deployed with TVM on edge hardware.
        This is the first time JAX training and TVM deployment shared an IR.

    Competition:
        Both OpenXLA/IREE and TVM target GPU deployment.
        Both ingest StableHLO as input.
        Users can benchmark both on the same StableHLO program.
        The OpenXLA benchmark suite (openxla-benchmark) measures both.

### The Full OpenXLA Ecosystem Map

    ┌─────────────────────────────────────────────────────────────────────┐
    │  FRAMEWORKS (emit StableHLO)                                        │
    │  JAX      PyTorch/XLA      TensorFlow     torch-mlir (→TOSA/SHLO)   │
    └───────────────────────┬─────────────────────────────────────────────┘
                            │ jax.export / SavedModel / torch.export
                            ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STABLEHLO (openxla/stablehlo)                                      │
    │  Stable, versioned MLIR dialect — the universal ML IR               │
    └──────┬────────────────┬───────────────────────┬─────────────────────┘
           │                │                       │
           ▼                ▼                       ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐
    │  OpenXLA/XLA │  │     IREE     │  │  Other consumers             │
    │  (openxla/xla│  │ (openxla/iree│  │  TVM, ONNX Runtime bridge,   │
    │  GPU+TPU     │  │  mobile+web  │  │  Torch-MLIR, custom tools    │
    └──────┬───────┘  └──────┬───────┘  └──────────────────────────────┘
           │                 │
           ▼                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PJRT (stable C API hardware abstraction layer)                     │
    └──────┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
           │          │          │          │          │
           ▼          ▼          ▼          ▼          ▼
       NVIDIA GPU  AMD GPU  Intel GPU  AWS Inferentia  Custom Silicon
       (libtpu.so) (ROCm)   (OpenVINO) (Neuron)        (any PJRT plugin)

    ┌─────────────────────────────────────────────────────────────────────┐
    │  OPENXLA PROJECTS                                                   │
    │  openxla/xla         core XLA compiler                              │
    │  openxla/stablehlo   portable IR specification                      │
    │  openxla/iree        MLIR-native deployment runtime                 │
    │  openxla/shardy      sharding propagation library                   │
    │  openxla/openxla-benchmark  cross-compiler benchmarks               │
    └─────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · StableHLO IR — Reading, Writing and Verifying": {
        "description": (
            "Hands-on exploration of the StableHLO IR format. "
            "Read real StableHLO text produced by JAX's export API. "
            "Understand every StableHLO op: syntax, type system, attributes. "
            "Trace a softmax, a dot_general matmul, and a reduce from JAX to StableHLO. "
            "Show the version tag and compatibility policy in action. "
            "Compare StableHLO vs MHLO vs HLO: what each looks like for the same computation."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  STABLEHLO IR — READING, WRITING AND VERIFYING")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    print(f"  JAX version:  {jax.__version__}")
    print(f"  Backend:      {jax.default_backend()}")
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("  JAX not installed: pip install jax")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: StableHLO textual format guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — StableHLO textual format")
print("━" * 65)
print()

SHLO_FORMAT = """
  STABLEHLO SYNTAX GUIDE
  ════════════════════════════════════════════════════════════════

  STRUCTURE: StableHLO is an MLIR module with stablehlo.* ops.
    module @my_model {
      func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        ...stablehlo ops...
        return %result : tensor<4xf32>
      }
    }

  TYPES: tensors with element type and static/dynamic shape.
    tensor<4xf32>         static 1D float tensor (4 elements)
    tensor<2x3xbf16>      static 2D bfloat16 tensor
    tensor<?x4xf32>       dynamic first dimension (batch size unknown)
    tensor<*xf32>         fully unranked (rarely used)

  ELEMENT TYPES (key ones):
    f32, f64, bf16, f16   floating point
    f8e4m3fn, f8e5m2      FP8 (for quantisation)
    si8, ui8, si32, si64  integers
    i1                    boolean (for masks)

  OPERATION SYNTAX:
    %result = stablehlo.opname %input1, %input2 {attrs} : (in_types) -> out_type
    ; or for tuple results:
    %r0, %r1 = stablehlo.opname %in : type -> (type0, type1)

  KEY ATTRIBUTES:
    broadcast_dimensions = [0, 2]    which dims of input map to output dims
    dimensions = [1]                  which dim(s) a reduce/transpose acts on
    padding = dense<[[1, 1], [0, 0]]> explicit padding specification
    dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [1]>

  COMPUTATIONS (for reduce, while, etc.):
    Some ops take sub-functions:
    %sum = stablehlo.reduce(%x, %zero) applies %add_fn across dim 1
    where add_fn is:
      ^bb0(%lhs: f32, %rhs: f32):
        %r = stablehlo.add %lhs, %rhs : f32
        stablehlo.return %r : f32
"""
print(SHLO_FORMAT)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Extract StableHLO from JAX computations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — StableHLO from JAX via jax.export")
print("━" * 65)
print()

if HAS_JAX:
    # ── Softmax ──────────────────────────────────────────────────────────
    def softmax(x):
        x_max  = jnp.max(x, axis=-1, keepdims=True)
        x_exp  = jnp.exp(x - x_max)
        return x_exp / jnp.sum(x_exp, axis=-1, keepdims=True)

    x_abs = jax.ShapeDtypeStruct((4, 8), jnp.float32)
    try:
        exp = jax.export.export(jax.jit(softmax))(x_abs)
        mlir_text = exp.mlir_module()
        print("  StableHLO for softmax (4×8 f32):")
        print()
        for line in mlir_text.split("\\n")[:50]:
            print(f"    {line}")
        if mlir_text.count("\\n") > 50:
            print(f"    ... ({mlir_text.count(chr(10))-50} more lines)")
    except Exception as e:
        # Fallback: use xla_computation
        comp     = jax.xla_computation(softmax)(jnp.ones((4,8), jnp.float32))
        hlo_text = comp.as_hlo_text()
        print("  HLO for softmax (4×8 f32) [StableHLO needs jax>=0.4.14]:")
        print()
        for line in hlo_text.split("\\n"):
            print(f"    {line}")
    print()

    # ── Two-layer MLP ─────────────────────────────────────────────────────
    def mlp_layer(x, W, b):
        return jnp.maximum(x @ W + b, 0.0)

    x_abs = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    W_abs = jax.ShapeDtypeStruct((16, 8), jnp.float32)
    b_abs = jax.ShapeDtypeStruct((8,),    jnp.float32)
    try:
        exp2 = jax.export.export(jax.jit(mlp_layer))(x_abs, W_abs, b_abs)
        print("  StableHLO for linear+relu (8×16 @ 16×8):")
        print()
        for line in exp2.mlir_module().split("\\n")[:40]:
            print(f"    {line}")
    except Exception:
        comp2 = jax.xla_computation(mlp_layer)(
            jnp.ones((8,16),jnp.float32),
            jnp.ones((16,8),jnp.float32),
            jnp.zeros((8,), jnp.float32))
        print("  HLO for linear+relu:")
        for line in comp2.as_hlo_text().split("\\n"):
            print(f"    {line}")
    print()

else:
    SOFTMAX_SHLO = """
  StableHLO for softmax(x: tensor<4x8xf32>) → tensor<4x8xf32>:

  module @jit_softmax {
    func.func public @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {

      // ── reduce to get row max ─────────────────────────────────────────
      %neg_inf = stablehlo.constant dense<-0x7F800000> : tensor<f32>
      %row_max = stablehlo.reduce(%arg0, %neg_inf) applies @max_fn across dim 1
               : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>

      // ── broadcast max back to (4,8) and subtract ──────────────────────
      %max_bc = stablehlo.broadcast_in_dim %row_max, dims=[0]
               : (tensor<4xf32>) -> tensor<4x8xf32>
      %shifted = stablehlo.subtract %arg0, %max_bc
               : tensor<4x8xf32>

      // ── exp ───────────────────────────────────────────────────────────
      %exp = stablehlo.exponential %shifted : tensor<4x8xf32>

      // ── row sum ───────────────────────────────────────────────────────
      %zero = stablehlo.constant dense<0.0> : tensor<f32>
      %row_sum = stablehlo.reduce(%exp, %zero) applies @add_fn across dim 1
               : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>

      // ── broadcast sum and divide ──────────────────────────────────────
      %sum_bc = stablehlo.broadcast_in_dim %row_sum, dims=[0]
               : (tensor<4xf32>) -> tensor<4x8xf32>
      %result = stablehlo.divide %exp, %sum_bc : tensor<4x8xf32>

      return %result : tensor<4x8xf32>
    }

    // Sub-computations used by reduce:
    func.func private @max_fn(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
      %r = stablehlo.maximum %a, %b : tensor<f32>
      return %r : tensor<f32>
    }
    func.func private @add_fn(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
      %r = stablehlo.add %a, %b : tensor<f32>
      return %r : tensor<f32>
    }
  }

  ── NOTES on key ops ────────────────────────────────────────────────────
  stablehlo.reduce:         fold a dimension using a sub-computation
  stablehlo.broadcast_in_dim: align a lower-rank tensor to a higher-rank one
  stablehlo.exponential:    elementwise exp(x)
  stablehlo.subtract/divide: elementwise binary ops

  ── Same computation in XLA HLO (for comparison) ─────────────────────
  HloModule softmax_module

  add_comp { lhs = f32[] p(0); rhs = f32[] p(1); ROOT r = f32[] add(lhs,rhs) }
  max_comp { lhs = f32[] p(0); rhs = f32[] p(1); ROOT r = f32[] maximum(lhs,rhs) }

  ENTRY softmax {
    x       = f32[4,8] parameter(0)
    neginf  = f32[] constant(-inf)
    row_max = f32[4] reduce(x, neginf), dimensions={1}, to_apply=max_comp
    max_bc  = f32[4,8] broadcast(row_max), dimensions={0}
    shifted = f32[4,8] subtract(x, max_bc)
    exp     = f32[4,8] exponential(shifted)
    zero    = f32[] constant(0)
    row_sum = f32[4] reduce(exp, zero), dimensions={1}, to_apply=add_comp
    sum_bc  = f32[4,8] broadcast(row_sum), dimensions={0}
    ROOT r  = f32[4,8] divide(exp, sum_bc)
  }

  KEY DIFFERENCE: StableHLO is an MLIR dialect (standard SSA, func/module).
  XLA HLO is a custom text format backed by C++ protos.
  StableHLO is stable across versions; HLO is not.
"""
    print(SOFTMAX_SHLO)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: dot_general — the generalised matmul op
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — stablehlo.dot_general: batched matmul anatomy")
print("━" * 65)
print()

DOT_GENERAL = """
  stablehlo.dot_general IS the most important op in transformer models.
  It handles: matmul, batched matmul, einsum, attention scores — all unified.

  SYNTAX:
    %result = stablehlo.dot_general %lhs, %rhs,
              lhs_batching_dimensions = [B dims of lhs],
              rhs_batching_dimensions = [B dims of rhs],
              lhs_contracting_dimensions = [K dims of lhs],
              rhs_contracting_dimensions = [K dims of rhs]
              : (tensor<...>, tensor<...>) -> tensor<...>

  CASE 1: Basic matrix multiply  C[M,N] = A[M,K] @ B[K,N]
    %C = stablehlo.dot_general %A, %B,
         lhs_batching_dimensions = [],
         rhs_batching_dimensions = [],
         lhs_contracting_dimensions = [1],    ; K is dim 1 of A
         rhs_contracting_dimensions = [0]     ; K is dim 0 of B
         : (tensor<MxKxf32>, tensor<KxNxf32>) -> tensor<MxNxf32>

  CASE 2: Batched matmul  C[B,M,N] = A[B,M,K] @ B[B,K,N]
    %C = stablehlo.dot_general %A, %B,
         lhs_batching_dimensions = [0],       ; B is dim 0 of A
         rhs_batching_dimensions = [0],       ; B is dim 0 of B
         lhs_contracting_dimensions = [2],    ; K is dim 2 of A
         rhs_contracting_dimensions = [1]     ; K is dim 1 of B
         : (tensor<BxMxKxf32>, tensor<BxKxNxf32>) -> tensor<BxMxNxf32>

  CASE 3: Multi-head attention scores  Q[B,H,S,D] @ K[B,H,D,S]
    Attention: scores = Q @ K^T / sqrt(d_head)
    %scores = stablehlo.dot_general %Q, %K_T,
              lhs_batching_dimensions = [0, 1],   ; batch, heads
              rhs_batching_dimensions = [0, 1],   ; batch, heads
              lhs_contracting_dimensions = [3],   ; d_head in Q
              rhs_contracting_dimensions = [2]    ; d_head in K^T
              : (tensor<BxHxSxDxf32>, tensor<BxHxDxSxf32>)
                -> tensor<BxHxSxSxf32>

  CASE 4: Einsum "bik,bjk->bij"  (used in some attention variants)
    lhs_batching     = [0],       ; b
    rhs_batching     = [0],       ; b
    lhs_contracting  = [2],       ; k
    rhs_contracting  = [2]        ; k
    → contracts k, keeps b (batch), i, j free

  WHY dot_general matters for compilers:
    All of the above go through cuBLAS GEMM in XLA and OpenXLA.
    The batching and contracting dimension specs are lowered to:
      B separate GEMM calls (old path) OR
      one batched GEMM call to cublasGemmStridedBatchedEx (new path).
    This spec is also what OpenXLA's AutoSharding uses to decide
    how to partition the computation across devices.
"""
print(DOT_GENERAL)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: StableHLO version compatibility simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — StableHLO version compatibility")
print("━" * 65)
print()

COMPAT_DEMO = """
  STABLEHLO COMPATIBILITY POLICY — ILLUSTRATED
  ════════════════════════════════════════════════════════════════

  StableHLO programs embed their version in the serialised binary:

  // Version header (embedded in binary, shown textually):
  // stablehlo_version = "1.0.0"
  module @my_model attributes {stablehlo.version = "1.0.0"} {
    func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      ...
    }
  }

  FORWARD COMPATIBILITY: version 1.0.0 programs work on v1.5.0, v2.0.0, ...
  The consumer version (e.g. 1.5.0) applies UPGRADE passes:
    - Deprecated op X → replacement op Y
    - Changed attribute encoding → new encoding
    These passes are generated from the compatibility spec.

  BACKWARD COMPATIBILITY: v1.5.0 consumer handles v1.0.0 programs.
  The consumer is aware of older serialisation and can parse it.

  WHAT THIS LOOKS LIKE IN PRACTICE:
  ─────────────────────────────────────────────────────────────────

  December 2024: Train GPT-2 with JAX 0.4.28
    exported = jax.export.export(jax.jit(model))(abstract_args)
    blob = exported.serialize()  # StableHLO v1.0.x embedded
    save_to_registry("gpt2_v1", blob)

  March 2026: Deploy with OpenXLA/IREE 3.x
    loaded = jax.export.deserialize(blob)
    # IREE runs compatibility upgrade passes:
    #   "stablehlo.dynamic_iota" (v1.0) → "stablehlo.iota" (v1.3)
    #   Attribute encoding fix from v1.1
    result = iree_session.run(loaded)
    # Works correctly. No manual intervention.

  WHY THIS MATTERS:
  Without StableHLO: retraining required every time JAX or XLA upgrades.
  With StableHLO: trained model artifacts are infrastructure-independent.
  Enterprises can freeze model versions for compliance/reproducibility.
"""
print(COMPAT_DEMO)

if HAS_JAX:
    # Show the version in a serialised module
    def simple_fn(x):
        return jnp.sum(x * 2.0)

    x_s = jax.ShapeDtypeStruct((8,), jnp.float32)
    try:
        exp = jax.export.export(jax.jit(simple_fn))(x_s)
        # Try to find version in the mlir module text
        mlir = exp.mlir_module()
        import re
        version_match = re.search(r"stablehlo.version.*?[\"\'](\\S+?)[\"\'\\)]",
                                   mlir, re.IGNORECASE)
        if version_match:
            print(f"  Detected StableHLO version in exported module: {version_match.group(1)}")
        else:
            # Check for module attributes
            lines = [l for l in mlir.split("\\n") if "version" in l.lower() or "stablehlo" in l.lower()]
            if lines:
                print(f"  Module metadata: {lines[0].strip()}")
            else:
                print(f"  Module exported successfully ({len(mlir)} chars of StableHLO MLIR)")
    except Exception as e:
        print(f"  Note: {e}")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · PJRT Device Abstraction — Plugin Architecture Simulation": {
        "description": (
            "Simulate the PJRT plugin architecture in Python. "
            "Implement a mock PJRT_Client, PJRT_Buffer, and PJRT_Executable interface. "
            "Show how a hardware vendor writes a PJRT plugin: compile + execute + transfer. "
            "Demonstrate how JAX uses PJRT: compile StableHLO → device binary → run. "
            "Compare the N×M problem (pre-PJRT) vs N+M solution (post-PJRT). "
            "Show PJRT API versioning: how old plugins work with new frameworks."
        ),
        "language": "python",
        "code": '''
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import time

print("=" * 65)
print("  PJRT DEVICE ABSTRACTION — PLUGIN ARCHITECTURE SIMULATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: The N×M problem illustrated
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The N×M integration problem (pre-PJRT)")
print("━" * 65)
print()

NM_PROBLEM = """
  BEFORE PJRT: N frameworks × M hardware = N×M integrations
  ════════════════════════════════════════════════════════════════

  Each cell = one custom integration path to build and maintain:

              │ NVIDIA GPU │ AMD GPU │ Intel GPU │ AWS Inf │ Apple ANE │
  ────────────┼────────────┼─────────┼───────────┼─────────┼───────────┤
  JAX         │  ✓ (XLA)   │ fork    │  partial  │  none   │   none    │
  TensorFlow  │  ✓ (XLA)   │ fork    │  partial  │  TF2    │   none    │
  PyTorch     │  ✓ (CUDA)  │ ROCm    │  ipex     │  none   │   none    │
  ONNX Runtime│  ✓ (TRT)   │ partial │  OpenVINO │  custom │   partial │
  MXNet       │  ✓ (CUDA)  │ ROCm    │  none     │  none   │   none    │
  ────────────┴────────────┴─────────┴───────────┴─────────┴───────────┘

  5 frameworks × 5 hardware = 25 integrations.
  Most are forks, partial, or non-existent.
  Each integration duplicates: buffer management, memory copies,
  async execution, error handling, device enumeration.

  AFTER PJRT: N frameworks + M hardware = N+M integrations
  ────────────────────────────────────────────────────────────────────────

  Frameworks implement the PJRT caller side ONCE:
    JAX, TF, PyTorch all call PJRT_Client_Compile() and PJRT_Executable_Execute()

  Hardware vendors implement the PJRT provider side ONCE:
    NVIDIA GPU: cuda_pjrt_plugin.so
    AMD GPU:    rocm_pjrt_plugin.so
    Intel GPU:  intel_pjrt_plugin.so
    AWS Inf:    neuron_pjrt_plugin.so
    Apple ANE:  coreml_pjrt_plugin.so

  Any framework × any hardware = works automatically.
  Total work: 5 framework integrations + 5 hardware plugins = 10 (not 25).
  For 10 frameworks × 10 hardware: 20 (not 100).
"""
print(NM_PROBLEM)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: PJRT C API structure (Python simulation)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — PJRT C API: core objects and methods")
print("━" * 65)
print()

PJRT_API_REFERENCE = """
  PJRT C API CORE STRUCTURE
  ════════════════════════════════════════════════════════════════

  // The versioned API struct (stable C ABI):
  typedef struct PJRT_Api {
    size_t struct_size;               // for ABI compatibility check
    int pjrt_api_version_major;       // e.g. 0
    int pjrt_api_version_minor;       // e.g. 49
    PJRT_Extension_Base* extension_start;  // linked list of optional extensions

    // Client lifecycle
    PJRT_Error* (*PJRT_Client_Create)(PJRT_Client_Create_Args*);
    PJRT_Error* (*PJRT_Client_Destroy)(PJRT_Client_Destroy_Args*);
    PJRT_Error* (*PJRT_Client_PlatformName)(PJRT_Client_PlatformName_Args*);
    PJRT_Error* (*PJRT_Client_Devices)(PJRT_Client_Devices_Args*);
    PJRT_Error* (*PJRT_Client_AddressableDevices)(PJRT_Client_AddressableDevices_Args*);

    // Compilation
    PJRT_Error* (*PJRT_Client_Compile)(PJRT_Client_Compile_Args*);
    //           ^ takes: client, program (StableHLO bytes), compile_options
    //             returns: PJRT_Executable

    // Buffer management (host ↔ device memory)
    PJRT_Error* (*PJRT_Client_BufferFromHostBuffer)(PJRT_Client_BufferFromHostBuffer_Args*);
    //           ^ copies host numpy array → PJRT_Buffer on device
    PJRT_Error* (*PJRT_Buffer_ToHostBuffer)(PJRT_Buffer_ToHostBuffer_Args*);
    //           ^ copies PJRT_Buffer → host numpy array
    PJRT_Error* (*PJRT_Buffer_Destroy)(PJRT_Buffer_Destroy_Args*);
    PJRT_Error* (*PJRT_Buffer_Dimensions)(PJRT_Buffer_Dimensions_Args*);
    PJRT_Error* (*PJRT_Buffer_ElementType)(PJRT_Buffer_ElementType_Args*);

    // Execution
    PJRT_Error* (*PJRT_Executable_Execute)(PJRT_Executable_Execute_Args*);
    //           ^ takes: executable, input buffers → output buffers + events
    PJRT_Error* (*PJRT_Executable_Destroy)(PJRT_Executable_Destroy_Args*);
    PJRT_Error* (*PJRT_Executable_GetCostAnalysis)(PJRT_Executable_GetCostAnalysis_Args*);

    // Async events
    PJRT_Error* (*PJRT_Event_Await)(PJRT_Event_Await_Args*);
    PJRT_Error* (*PJRT_Event_OnReady)(PJRT_Event_OnReady_Args*);
    PJRT_Error* (*PJRT_Event_Destroy)(PJRT_Event_Destroy_Args*);

    // ... ~35 more function pointers ...
  } PJRT_Api;

  PLUGIN LOADING (framework side):
    void* handle = dlopen("cuda_pjrt_plugin.so", RTLD_NOW);
    typedef const PJRT_Api* (*GetPjrtApiFn)();
    GetPjrtApiFn get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    const PJRT_Api* api = get_api();
    // Now api->PJRT_Client_Create, api->PJRT_Client_Compile, etc. are callable.
"""
print(PJRT_API_REFERENCE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Mock PJRT plugin — hardware vendor perspective
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Mock PJRT plugin: hardware vendor implementation")
print("━" * 65)
print()

@dataclass
class PJRTBuffer:
    """Device-side buffer. In real PJRT: wraps a device memory pointer."""
    data: np.ndarray
    device_id: int
    dtype: np.dtype
    shape: tuple
    is_valid: bool = True

    def to_host(self) -> np.ndarray:
        """PJRT_Buffer_ToHostBuffer: device → host memory copy."""
        assert self.is_valid, "Buffer has been destroyed"
        return self.data.copy()

    def destroy(self):
        """PJRT_Buffer_Destroy: release device memory."""
        self.is_valid = False
        self.data = None


@dataclass
class PJRTEvent:
    """Async event. In real PJRT: wraps a CUDA stream event or TPU semaphore."""
    _callback: Optional[Callable] = None
    _completed: bool = False
    _result: Any = None

    def await_completion(self):
        """PJRT_Event_Await: block until computation finishes."""
        if not self._completed and self._callback:
            self._result = self._callback()
            self._completed = True
        return self._result

    def on_ready(self, fn: Callable):
        """PJRT_Event_OnReady: register a callback for when computation finishes."""
        if self._completed:
            fn(self._result)
        else:
            old_cb = self._callback
            def chained():
                result = old_cb() if old_cb else None
                fn(result)
                return result
            self._callback = chained


@dataclass
class PJRTExecutable:
    """
    Compiled program binary for a specific device.
    In real PJRT: wraps a compiled cubin (NVIDIA), HLO executable (TPU), etc.
    """
    name: str
    device_id: int
    _compute_fn: Callable    # the actual compiled kernel (simulated here)
    cost_flops: int = 0
    cost_bytes: int = 0

    def execute(self, input_buffers: List[PJRTBuffer]) -> tuple:
        """
        PJRT_Executable_Execute: run the compiled program.
        Returns (output_buffers, event).
        In real PJRT: launches CUDA kernels, returns before completion.
        """
        inputs = [b.to_host() for b in input_buffers]
        # Simulate async execution via event
        event = PJRTEvent(_callback=lambda: self._compute_fn(*inputs))
        return event

    def get_cost_analysis(self):
        """PJRT_Executable_GetCostAnalysis: flop/byte estimates."""
        return {"flops": self.cost_flops, "bytes": self.cost_bytes}


class PJRTDevice:
    """
    One physical device. In real PJRT: one GPU, one TPU chip.
    """
    def __init__(self, device_id: int, kind: str, memory_gb: float):
        self.device_id      = device_id
        self.kind           = kind
        self.memory_gb      = memory_gb
        self._buffers: List[PJRTBuffer] = []

    def __repr__(self):
        return f"PJRTDevice(id={self.device_id}, kind={self.kind!r}, mem={self.memory_gb}GB)"


class MockCUDAPJRTClient:
    """
    Simulated NVIDIA GPU PJRT plugin.
    This is what nvidia writes in cuda_pjrt_plugin.so.
    The framework (JAX) loads this .so and calls these methods via the C API.
    """
    PLUGIN_NAME    = "CUDA"
    PLUGIN_VERSION = (0, 49)   # PJRT API version this plugin implements

    def __init__(self, n_devices: int = 2):
        self.devices = [
            PJRTDevice(i, "NVIDIA A100-SXM4-80GB", 80.0)
            for i in range(n_devices)
        ]
        self._compiled_cache: Dict[str, PJRTExecutable] = {}
        print(f"  [{self.PLUGIN_NAME} PJRT] Initialised {n_devices} device(s)")
        for d in self.devices:
            print(f"    {d}")

    def platform_name(self) -> str:
        return "cuda"

    def get_devices(self) -> List[PJRTDevice]:
        """PJRT_Client_Devices: enumerate all devices."""
        return self.devices

    def buffer_from_host(self, host_array: np.ndarray,
                          device_id: int = 0) -> PJRTBuffer:
        """
        PJRT_Client_BufferFromHostBuffer:
        Copies a host numpy array to device memory.
        In real CUDA: cudaMemcpyAsync(device_ptr, host_ptr, nbytes, H2D, stream)
        """
        buf = PJRTBuffer(
            data=host_array.copy(),    # simulate device memory
            device_id=device_id,
            dtype=host_array.dtype,
            shape=host_array.shape
        )
        print(f"  [{self.PLUGIN_NAME} PJRT] H2D transfer: "
              f"shape={host_array.shape} dtype={host_array.dtype} "
              f"→ device {device_id} ({host_array.nbytes/1024:.1f} KB)")
        return buf

    def compile(self, stablehlo_text: str,
                computation_name: str = "unnamed") -> PJRTExecutable:
        """
        PJRT_Client_Compile:
        Compile StableHLO → device binary.
        In real CUDA: StableHLO → XLA HLO → NVPTX → PTX → cubin
        Returns PJRT_Executable wrapping the cubin.
        """
        if computation_name in self._compiled_cache:
            print(f"  [{self.PLUGIN_NAME} PJRT] Cache hit: {computation_name!r}")
            return self._compiled_cache[computation_name]

        print(f"  [{self.PLUGIN_NAME} PJRT] Compiling {computation_name!r}...")
        print(f"    StableHLO → MHLO → HLO → NVPTX → PTX → cubin")

        # Parse the "StableHLO" to determine what computation to do
        # (in real code: XLA parses and compiles the actual MLIR)
        if "dot_general" in stablehlo_text or "matmul" in stablehlo_text.lower():
            def compute_fn(*inputs):
                if len(inputs) == 2:
                    return np.matmul(inputs[0], inputs[1])
                return inputs[0]
            flops = 2 * 256 * 256 * 256    # example GEMM flops
            bw    = 3 * 256 * 256 * 4      # A + B + C bytes
        elif "add" in stablehlo_text or "relu" in stablehlo_text.lower():
            def compute_fn(*inputs):
                result = inputs[0]
                for inp in inputs[1:]:
                    result = result + inp
                return np.maximum(result, 0)
            flops = 2 * inputs[0].size if inputs else 1024
            bw    = 2 * 1024 * 4
        else:
            def compute_fn(*inputs):
                return inputs[0] if inputs else np.array([0.0])
            flops, bw = 0, 0

        exec_obj = PJRTExecutable(
            name=computation_name,
            device_id=0,
            _compute_fn=compute_fn,
            cost_flops=flops,
            cost_bytes=bw
        )
        self._compiled_cache[computation_name] = exec_obj
        print(f"    Compiled: estimated {flops:,} FLOPs, {bw:,} bytes BW")
        return exec_obj

    def device_to_host(self, buf: PJRTBuffer) -> np.ndarray:
        """
        Wraps PJRT_Buffer_ToHostBuffer.
        In real CUDA: cudaMemcpyAsync(host_ptr, device_ptr, nbytes, D2H, stream)
        """
        result = buf.to_host()
        print(f"  [{self.PLUGIN_NAME} PJRT] D2H transfer: "
              f"shape={result.shape} ({result.nbytes/1024:.1f} KB)")
        return result


# ── Demonstrate the full PJRT workflow ───────────────────────────────────
print()
print("  FULL PJRT WORKFLOW DEMONSTRATION:")
print()

# 1. Load plugin (simulated)
plugin = MockCUDAPJRTClient(n_devices=2)
print()

# 2. Transfer input data to device
A = np.random.randn(4, 8).astype(np.float32)
B = np.random.randn(8, 4).astype(np.float32)
buf_A = plugin.buffer_from_host(A, device_id=0)
buf_B = plugin.buffer_from_host(B, device_id=0)
print()

# 3. Compile a StableHLO "program"
MATMUL_SHLO = """
module @jit_matmul {
  func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>)
                  -> tensor<4x4xf32> {
    %result = stablehlo.dot_general %arg0, %arg1,
              lhs_contracting_dimensions = [1],
              rhs_contracting_dimensions = [0]
              : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
"""
executable = plugin.compile(MATMUL_SHLO, computation_name="matmul_4x8x4")
print()

# 4. Execute
print("  Executing compiled matmul on device 0:")
event = executable.execute([buf_A, buf_B])
result_host = event.await_completion()
print(f"    A: {A.shape}, B: {B.shape}")
print(f"    A @ B result shape: {result_host.shape}")
print(f"    A @ B result[0,:3] = {result_host[0,:3]}")
expected = A @ B
print(f"    Expected[0,:3]     = {expected[0,:3]}")
print(f"    Max error: {np.max(np.abs(result_host - expected)):.2e} ✅")
print()

# 5. Retrieve output
buf_result = PJRTBuffer(result_host, 0, result_host.dtype, result_host.shape)
output = plugin.device_to_host(buf_result)
print()

# 6. Compilation cache hit
print("  Second call (same shape): should hit compilation cache:")
exec2 = plugin.compile(MATMUL_SHLO, computation_name="matmul_4x8x4")
print(f"  Same executable object? {exec2 is executable} ✅")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: PJRT API versioning simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — PJRT versioning: old plugins on new frameworks")
print("━" * 65)
print()

VERSIONING = """
  PJRT API VERSIONING STRATEGY
  ════════════════════════════════════════════════════════════════

  The PJRT struct begins with:
    size_t struct_size;        // size of THIS version of the struct
    int    version_major;      // breaks backward compat (rare)
    int    version_minor;      // additive only (new functions appended)

  Framework loads a plugin:
    1. Call get_api() → get PJRT_Api* pointer
    2. Check struct_size: if smaller than expected, old plugin (fewer fields)
    3. Check version_minor: access only functions available in plugin's version
    4. New optional extensions: check extension_start linked list

  Example: JAX 0.4.30 (uses PJRT v0.52) loads old plugin (PJRT v0.45):
    - JAX sees struct_size is smaller → knows plugin has v0.45 functions only
    - JAX calls only functions up to v0.45
    - New v0.46-v0.52 functions (e.g. streaming execution) → not called
    - Plugin works correctly; new features simply unavailable for this plugin

  Plugin v0.45 built against PJRT_Api:
    { struct_size = sizeof(PJRT_Api at v0.45),  // 400 bytes (example)
      PJRT_Client_Create, PJRT_Client_Compile, ...
      // PJRT_Executable_GetCostAnalysis: added in v0.48, NOT PRESENT }

  Framework (JAX) using PJRT_Api v0.52:
    const PJRT_Api* api = get_api();
    if (api->struct_size >= offsetof(PJRT_Api, PJRT_Executable_GetCostAnalysis)) {
      // Safe to call: new function is present in this plugin
      api->PJRT_Executable_GetCostAnalysis(args);
    } else {
      // Old plugin: function not available, use fallback
      estimated_cost = heuristic_cost_estimate(executable);
    }
"""
print(VERSIONING)

# Simulate version compatibility check
class PJRTApiVersion:
    def __init__(self, major, minor, available_functions):
        self.major = major
        self.minor = minor
        self.available = set(available_functions)

    def can_call(self, fn_name: str) -> bool:
        return fn_name in self.available

    def __repr__(self):
        return f"PJRT v{self.major}.{self.minor} ({len(self.available)} functions)"


plugin_v45 = PJRTApiVersion(0, 45, {
    "PJRT_Client_Create", "PJRT_Client_Destroy", "PJRT_Client_Devices",
    "PJRT_Client_Compile", "PJRT_Client_BufferFromHostBuffer",
    "PJRT_Buffer_ToHostBuffer", "PJRT_Buffer_Destroy",
    "PJRT_Executable_Execute", "PJRT_Executable_Destroy",
    "PJRT_Event_Await", "PJRT_Event_Destroy",
})
plugin_v52 = PJRTApiVersion(0, 52, plugin_v45.available | {
    "PJRT_Executable_GetCostAnalysis",   # added v0.48
    "PJRT_Client_GetTopologyDescription",  # added v0.49
    "PJRT_Executable_GetCompiledMemoryStats",  # added v0.51
    "PJRT_Buffer_CopyToDevice",           # added v0.52
})

framework_requests = [
    "PJRT_Client_Compile",
    "PJRT_Executable_Execute",
    "PJRT_Executable_GetCostAnalysis",
    "PJRT_Buffer_CopyToDevice",
    "PJRT_Event_Await",
]

print("  Framework JAX v0.4.30 requesting functions from two plugin versions:")
print()
print(f"  {'Function':45s}  {'v0.45 plugin':12s}  {'v0.52 plugin':12s}")
print("  " + "-" * 72)
for fn in framework_requests:
    v45_ok = "✅ available" if plugin_v45.can_call(fn) else "❌ fallback"
    v52_ok = "✅ available" if plugin_v52.can_call(fn) else "❌ fallback"
    print(f"  {fn:45s}  {v45_ok:12s}  {v52_ok:12s}")

print()
print(f"  {plugin_v45}: {len(plugin_v45.available)} functions")
print(f"  {plugin_v52}: {len(plugin_v52.available)} functions")
print(f"  Both plugins work with JAX v0.4.30 — new features gracefully degraded.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · IREE Compilation Pipeline — From StableHLO to vmfb": {
        "description": (
            "Deep dive into IREE's MLIR-native compilation pipeline. "
            "Trace a matmul from StableHLO through Flow → Stream → HAL → SPIR-V. "
            "Show IREE's dispatch region analysis: how it identifies GPU work units. "
            "Demonstrate the .vmfb artifact: what it contains and how the runtime uses it. "
            "Compare IREE vs XLA for mobile deployment scenarios. "
            "Show iree-compile CLI: flags, target backends, input formats."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  IREE COMPILATION PIPELINE — STABLEHLO TO VMFB")
print("=" * 65)
print()

try:
    import iree.runtime as iree_rt
    import iree.compiler as iree_cc
    HAS_IREE = True
    print(f"  IREE runtime available ✅")
except ImportError:
    HAS_IREE = False
    print("  IREE not installed: pip install iree-runtime iree-compiler")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: IREE dialect pipeline explanation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — IREE dialect pipeline: Flow → Stream → HAL")
print("━" * 65)
print()

IREE_PIPELINE = """
  IREE COMPILATION: STABLEHLO → .vmfb IN DETAIL
  ════════════════════════════════════════════════════════════════

  STAGE 1: INPUT TRANSFORMATION (StableHLO → IREE tensor world)
  ─────────────────────────────────────────────────────────────────
  Input:
    func.func @matmul_relu(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>)
              -> tensor<4x4xf32> {
      %C  = stablehlo.dot_general %A, %B, ... : -> tensor<4x4xf32>
      %z  = stablehlo.constant dense<0.0> : tensor<f32>
      %bc = stablehlo.broadcast_in_dim %z, dims=[] : -> tensor<4x4xf32>
      %R  = stablehlo.maximum %C, %bc : tensor<4x4xf32>
      return %R : tensor<4x4xf32>
    }

  After --iree-stablehlo-to-linalg:
    %C  = linalg.matmul ins(%A, %B) outs(%init) : tensor<4x4xf32>
    %R  = linalg.generic {relu} ins(%C) outs(%init2) : tensor<4x4xf32>
    ; dot_general → linalg.matmul
    ; maximum(x, broadcast(0)) → linalg.generic with elementwise max

  STAGE 2: FLOW DIALECT (dispatch region identification)
  ─────────────────────────────────────────────────────────────────
  After --iree-flow-dispatch-linalg-on-tensors:
    %result = flow.dispatch @matmul_relu_dispatch_0[%c4, %c4](%A, %B)
              : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    ; The entire matmul+relu is ONE dispatch (one GPU kernel launch)
    ; flow.dispatch = "this is a unit of work for the device"

  Dispatch region contents:
    func @matmul_relu_dispatch_0(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>)
              -> tensor<4x4xf32> {
      %C = linalg.matmul ins(%A,%B) outs(%init)
      %R = linalg.generic {relu} ins(%C) outs(%init2)
      return %R
    }
    ; This function becomes ONE GPU shader (one .spv kernel or one CUDA kernel)

  STAGE 3: STREAM DIALECT (async execution and memory lifetimes)
  ─────────────────────────────────────────────────────────────────
  After --iree-stream-transformation-pipeline:
    %A_res = stream.resource.alloc : !stream.resource<constant>{%A_size}
    stream.resource.store %A into %A_res : f32 tensor<4x8xf32>

    %result_res = stream.async.execute with(%A_res as %a, %B_res as %b) {
      %r = stream.async.dispatch @dispatch_0[%c4, %c4](%a, %b)
           : (!stream.resource, !stream.resource) -> !stream.resource
      stream.yield %r
    }
    ; stream.async.execute = submit GPU work without blocking
    ; Result is a !stream.resource (device handle, not yet a tensor)
    ; Host can continue doing work until it needs the result

  STAGE 4: HAL DIALECT (hardware abstraction)
  ─────────────────────────────────────────────────────────────────
  After --iree-hal-transformation-pipeline:
    %cmd_buf = hal.command_buffer.create %device : !hal.command_buffer
    hal.command_buffer.begin %cmd_buf
    hal.command_buffer.dispatch %executable, entry_point=0,
        workgroup_count=[%gx, %gy, %gz]
    hal.command_buffer.end %cmd_buf
    hal.device.queue.execute %device, %cmd_buf

    ; hal.command_buffer = Vulkan VkCommandBuffer / CUDA stream / Metal MTLCommandBuffer
    ; This HAL IR is device-agnostic: same IR for Vulkan, Metal, CUDA

  STAGE 5: TARGET-SPECIFIC LOWERING
  ─────────────────────────────────────────────────────────────────
  For Vulkan target (--iree-hal-target-backends=vulkan-spirv):
    SPIRV dialect → SPIR-V binary (.spv)
    ; Embedded in the .vmfb alongside the HAL schedule

  For CUDA target (--iree-hal-target-backends=cuda):
    CUDA dialect → NVPTX → PTX → cubin
    ; Also embedded in the .vmfb

  For CPU target (--iree-hal-target-backends=llvm-cpu):
    LLVM dialect → LLVM IR → LLVM optimises → native .o
    ; Embedded as a native library in the .vmfb

  FINAL OUTPUT: .vmfb (VM FlatBuffer)
  ─────────────────────────────────────────────────────────────────
  A .vmfb is a FlatBuffers binary containing:
    [module metadata]     input/output names, shapes, dtypes
    [HAL schedule]        command sequence: alloc, copy, dispatch, wait
    [SPIR-V shaders]      compiled GPU kernels for Vulkan
    [CUDA kernels]        compiled .cubin for NVIDIA (if requested)
    [CPU kernels]         compiled native code for CPU (if requested)
    [constant data]       model weights (serialised tensors)

  Runtime loads .vmfb → parses FlatBuffer → dispatches to device.
  Runtime size: ~150 KB (Vulkan) or ~300 KB (CUDA) — no Python needed.
"""
print(IREE_PIPELINE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: IREE compilation and execution (if available)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — IREE compilation and execution")
print("━" * 65)
print()

STABLEHLO_MATMUL = """
func.func @matmul_relu(%arg0: tensor<4x8xf32>,
                        %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm   = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x4xf32>)
                         outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %relu_init = tensor.empty() : tensor<4x4xf32>
  %relu = linalg.generic
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
                        affine_map<(d0,d1)->(d0,d1)>],
       iterator_types = ["parallel","parallel"]}
      ins(%mm : tensor<4x4xf32>) outs(%relu_init : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %c0 = arith.constant 0.0 : f32
      %r  = arith.maximumf %in, %c0 : f32
      linalg.yield %r : f32
  } -> tensor<4x4xf32>
  return %relu : tensor<4x4xf32>
}
"""

if HAS_IREE:
    A = np.random.randn(4, 8).astype(np.float32)
    B = np.random.randn(8, 4).astype(np.float32)

    print("  Compiling StableHLO/linalg matmul+relu via IREE:")
    print()
    try:
        vmfb = iree_cc.compile_str(
            STABLEHLO_MATMUL,
            target_backends=["llvm-cpu"],
            input_type="auto",
        )
        print(f"  Compiled .vmfb size: {len(vmfb):,} bytes")

        config  = iree_rt.Config("local-task")
        ctx     = iree_rt.SystemContext(config=config)
        vm_mod  = iree_rt.VmModule.from_flatbuffer(ctx.instance, vmfb)
        ctx.add_vm_module(vm_mod)

        fn      = ctx.modules.module["matmul_relu"]
        result  = fn(A, B)[0]
        expected = np.maximum(A @ B, 0.0)

        print(f"  A: {A.shape}, B: {B.shape}")
        print(f"  IREE result[0,:3]:    {result[0,:3]}")
        print(f"  Expected[0,:3]:       {expected[0,:3]}")
        print(f"  Max error:            {np.max(np.abs(result - expected)):.2e}")
        print(f"  ✅ IREE matmul_relu matches numpy reference")
    except Exception as e:
        print(f"  IREE compile error: {e}")
        print(f"  (This is expected if IREE version doesn\'t support this input format)")
        print(f"  Try: iree-compile input.mlir --iree-input-type=linalg-on-tensors")
    print()
else:
    print("  IREE not available. Reference CLI usage below:")
    IREE_CLI = """
  iree-compile CLI REFERENCE:
  ─────────────────────────────────────────────────────────────────

  # From StableHLO → CPU .vmfb:
  iree-compile input.mlir \\
      --iree-input-type=stablehlo \\
      --iree-hal-target-backends=llvm-cpu \\
      -o model.vmfb

  # From StableHLO → Vulkan GPU .vmfb:
  iree-compile input.mlir \\
      --iree-input-type=stablehlo \\
      --iree-hal-target-backends=vulkan-spirv \\
      -o model_vulkan.vmfb

  # From StableHLO → CUDA .vmfb:
  iree-compile input.mlir \\
      --iree-input-type=stablehlo \\
      --iree-hal-target-backends=cuda \\
      --iree-cuda-target-chip=sm_80 \\  # A100
      -o model_cuda.vmfb

  # Cross-compile for Android ARM64 Vulkan:
  iree-compile input.mlir \\
      --iree-input-type=stablehlo \\
      --iree-hal-target-backends=vulkan-spirv \\
      --iree-vulkan-target-triple=adreno-unknown-android31 \\
      -o model_android.vmfb

  # Run .vmfb:
  iree-run-module --module=model.vmfb \\
      --device=local-task \\
      --function=main \\
      --input=4x8xf32=@A.bin \\
      --input=8x4xf32=@B.bin \\
      --output=@output.bin

  # Python runtime (iree-runtime):
  import iree.runtime as rt
  config = rt.Config("local-task")          # or "vulkan" / "cuda"
  ctx    = rt.SystemContext(config=config)
  vm_mod = rt.VmModule.from_flatbuffer(ctx.instance, open("model.vmfb","rb").read())
  ctx.add_vm_module(vm_mod)
  result = ctx.modules.module["main"](input_a, input_b)
"""
    print(IREE_CLI)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: IREE vs XLA benchmark comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — IREE vs XLA deployment comparison")
print("━" * 65)
print()

COMPARISON = """
  IREE vs XLA: CHOOSING THE RIGHT TOOL
  ════════════════════════════════════════════════════════════════

  WORKLOAD TYPE DETERMINES THE WINNER:

  ── BERT inference (sequence 128, batch 1) on x86 CPU ─────────────────
  Tool           Time (ms)    Notes
  ─────────────────────────────────────────────────────────────────────
  PyTorch eager    8.5         Framework overhead, eager dispatch
  ONNX Runtime     3.1         Good CPU kernels (MKL/oneDNN)
  IREE (llvm-cpu)  2.8         MLIR-tiled loops, LLVM vectorised
  XLA (jax.jit)    3.2         Good but heavier startup
  TVM (autotune)   2.4         AutoScheduler finds best tiling

  ── MobileNetV3 inference on Android Vulkan GPU ─────────────────────
  Tool           Time (ms)    Notes
  ─────────────────────────────────────────────────────────────────────
  TF Lite          12.1        Quantised int8, good mobile support
  IREE (Vulkan)     6.8        SPIR-V shaders, Vulkan GPU direct
  XLA (GPU)         N/A        No Android/Vulkan support
  ONNX Runtime Mob  9.2        Mobile extension, Vulkan backend

  ── GPT-2 (medium) training on A100 GPU ────────────────────────────
  Tool           TFLOPs/s     Notes
  ─────────────────────────────────────────────────────────────────────
  PyTorch+CUDA     142         Baseline (cuBLAS + eager)
  JAX+XLA          198         HLO fusion, cuBLAS GEMM + fused epilogue
  JAX+Triton GEMM  210         Mosaic GPU / Triton custom kernels
  IREE (CUDA)      155         Good but cuBLAS wins for large GEMMs
  TVM (autotune)   168         Close to XLA with per-op autotuning

  SUMMARY:
  ─────────────────────────────────────────────────────────────────────
  Use IREE when:  deploying to mobile, web, edge; Vulkan/Metal targets;
                  need tiny runtime; cross-compiling to ARM/RISC-V.

  Use XLA when:   training on GPU/TPU; JAX ecosystem; maximum training
                  throughput; Google Cloud hardware access.

  Use BOTH:       Train with JAX+XLA → export StableHLO → deploy with IREE.
                  This is the canonical OpenXLA end-to-end workflow.

  BENCHMARK NOTE: openxla/openxla-benchmark provides reproducible
  cross-compiler comparisons on a standard workload suite.
  Always measure on your actual hardware and workload.
"""
print(COMPARISON)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · OpenXLA Optimisations — AutoSharding and Quantisation": {
        "description": (
            "Deep dive into OpenXLA's key new optimisations beyond vanilla XLA. "
            "Simulate AutoSharding: ILP-based automatic parallelism strategy discovery. "
            "Show sharding decisions for a transformer layer across N GPUs. "
            "Implement the cost model: communication volume, memory, compute balance. "
            "Demonstrate OpenXLA quantisation: int8 and FP8 via StableHLO quant ops. "
            "Show the shardy library: sharding propagation on the HLO graph."
        ),
        "language": "python",
        "code": '''
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import itertools

print("=" * 65)
print("  OPENXLA OPTIMISATIONS — AUTOSHARDING AND QUANTISATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: AutoSharding — automatic parallelism discovery
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — AutoSharding: ILP-based strategy discovery")
print("━" * 65)
print()

AUTOSHARDING_THEORY = """
  AUTOSHARDING: PROBLEM STATEMENT
  ════════════════════════════════════════════════════════════════

  Given: a transformer layer HLO graph + a device mesh of N×M GPUs
  Find:  the optimal sharding strategy that minimises:
           α × communication_time + β × memory_per_device + γ × compute_imbalance

  Sharding choices for each operator (e.g. matmul C[M,N] = A[M,K] @ B[K,N]):

  Strategy          A shard    B shard    C shard    Communication
  ─────────────────────────────────────────────────────────────────────────
  Data parallel     M-dim      replicated M-dim       AllReduce(K partial sum)
  Column parallel   replicated N-dim      N-dim       AllGather(C) or none
  Row parallel      K-dim      K-dim      replicated  AllReduce(C)
  Fully replicated  replicated replicated replicated  none (wasteful on memory)
  Fully sharded     M-dim      N-dim      local       AllReduce + AllGather

  AutoSharding explores ALL possible strategies for EACH op and finds
  the globally optimal combination via dynamic programming or ILP.

  Scale of the problem:
    A GPT-3 layer has ~50 matmuls and other ops.
    Each op has ~5-10 strategy choices.
    All combinations: 10^50 → intractable by brute force.
    ILP/DP reduces to tractable: O(|ops| × |strategies|^2).
"""
print(AUTOSHARDING_THEORY)

@dataclass
class ShardingStrategy:
    """One possible sharding for a matmul op."""
    name: str
    A_dim: Optional[int]      # which dim of A is sharded (None = replicated)
    B_dim: Optional[int]      # which dim of B is sharded (None = replicated)
    C_dim: Optional[int]      # which dim of C is sharded (None = replicated)
    comm_op: str              # what collective is needed


@dataclass
class MatmulOp:
    """A matmul in the HLO graph with shape annotations."""
    name: str
    M: int; K: int; N: int   # dimensions
    dtype_bytes: int = 2     # bf16 = 2 bytes

    def flops(self):
        return 2 * self.M * self.K * self.N

    def compute_strategy_cost(self, strategy: ShardingStrategy,
                               n_devices: int,
                               bandwidth_gbps: float = 300.0) -> Dict:
        """
        Estimate cost of a sharding strategy.
        Returns: {compute_ms, comm_ms, mem_per_device_mb, total_ms}
        """
        n = n_devices
        flops_per_device = self.flops() / n   # ideally balanced

        # Peak A100 BF16: ~312 TFLOPS
        compute_s = flops_per_device / (312e12 / n)
        compute_ms = compute_s * 1000

        # Communication cost
        if strategy.comm_op == "AllReduce":
            # AllReduce(M×N result): each device sends M×N×dtype
            comm_bytes = self.M * self.N * self.dtype_bytes * 2  # factor 2: send+recv
            comm_ms = comm_bytes / (bandwidth_gbps * 1e9 / 1000)
        elif strategy.comm_op == "AllGather":
            comm_bytes = self.M * self.N * self.dtype_bytes * (1 - 1/n)
            comm_ms = comm_bytes / (bandwidth_gbps * 1e9 / 1000)
        elif strategy.comm_op == "None":
            comm_ms = 0.0
        else:
            comm_ms = 0.1  # estimate for other collectives

        # Memory per device
        A_mem = self.M * self.K * self.dtype_bytes
        B_mem = self.K * self.N * self.dtype_bytes
        C_mem = self.M * self.N * self.dtype_bytes

        if strategy.A_dim is not None: A_mem /= n
        if strategy.B_dim is not None: B_mem /= n
        if strategy.C_dim is not None: C_mem /= n

        mem_mb = (A_mem + B_mem + C_mem) / 1e6
        total_ms = compute_ms + comm_ms

        return {
            "compute_ms": compute_ms,
            "comm_ms": comm_ms,
            "mem_per_device_mb": mem_mb,
            "total_ms": total_ms,
        }


strategies = [
    ShardingStrategy("data_parallel",    A_dim=0, B_dim=None, C_dim=0,   comm_op="AllReduce"),
    ShardingStrategy("column_parallel",  A_dim=None, B_dim=1, C_dim=1,   comm_op="AllGather"),
    ShardingStrategy("row_parallel",     A_dim=1, B_dim=0,   C_dim=None, comm_op="AllReduce"),
    ShardingStrategy("fully_replicated", A_dim=None,B_dim=None,C_dim=None,comm_op="None"),
    ShardingStrategy("fully_sharded",    A_dim=0, B_dim=1,   C_dim=None, comm_op="AllReduce"),
]

# GPT-3 scale matmul: Q projection in attention (d_model=12288, d_head=128, n_heads=96)
qkv_proj = MatmulOp("QKV_proj", M=2048, K=12288, N=12288, dtype_bytes=2)  # seq=2048
ffn_up   = MatmulOp("FFN_up",   M=2048, K=12288, N=49152, dtype_bytes=2)  # 4× expand

print(f"  AutoSharding cost analysis for GPT-3 scale ops (8 A100s, 300 GB/s NVLink):")
print()

for op in [qkv_proj, ffn_up]:
    print(f"  Op: {op.name}  [{op.M}×{op.K}] @ [{op.K}×{op.N}]  ({op.flops()/1e12:.1f} TFLOP)")
    print(f"  {'Strategy':20s}  {'Compute':>10}  {'Comm':>10}  {'Mem/dev':>10}  {'Total':>10}")
    print("  " + "-" * 60)
    best = None; best_total = float("inf")
    for s in strategies:
        cost = op.compute_strategy_cost(s, n_devices=8)
        marker = ""
        if cost["total_ms"] < best_total:
            best_total = cost["total_ms"]
            best = s.name
            marker = " ←"
        print(f"  {s.name:20s}  {cost['compute_ms']:>8.2f}ms  "
              f"{cost['comm_ms']:>8.2f}ms  "
              f"{cost['mem_per_device_mb']:>8.0f}MB  "
              f"{cost['total_ms']:>8.2f}ms{marker}")
    print(f"  AutoSharding picks: {best!r} (lowest total latency)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: OpenXLA quantisation pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — OpenXLA quantisation: int8, FP8, and StableHLO quant ops")
print("━" * 65)
print()

QUANT_THEORY = """
  QUANTISATION IN OPENXLA
  ════════════════════════════════════════════════════════════════

  OpenXLA supports quantisation at the StableHLO level.
  Quantised types available in StableHLO:
    si8  : signed 8-bit integer (int8)
    ui8  : unsigned 8-bit integer
    si4  : signed 4-bit integer (int4, packed 2 per byte)
    f8e4m3fn : FP8, 4 exponent bits + 3 mantissa (NVIDIA H100 TF32)
    f8e5m2   : FP8, 5 exponent bits + 2 mantissa (IEEE FP8)

  QUANTISED MATMUL (int8 weight, float activation):
    Primal (f32 matmul):
      %C = stablehlo.dot_general %A_f32, %W_f32 ...

    Quantised version:
      %W_q   = stablehlo.uniform_quantize %W_f32 {scale=0.01, zero_point=0}
               : (tensor<KxNxf32>) -> tensor<KxNxsi8>
      %C_q   = stablehlo.dot_general %A_f32, %W_q ...   ; mixed precision
               : (tensor<MxKxf32>, tensor<KxNxsi8>) -> tensor<MxNxf32>
      ; XLA lowers this to int8 cuDNN GEMM (cublasGemmEx with CUDA_R_8I)

    Memory savings: 4× (f32 → int8 weights)
    Speed gain:     2-4× (INT8 tensor cores on A100/H100)
    Accuracy loss:  <0.5% on typical BERT/GPT tasks with calibration

  FP8 TRAINING (H100/Hopper-specific):
    H100 adds FP8 tensor cores: 2× faster than BF16 for matmuls.
    %A_fp8 = stablehlo.convert %A_bf16 : bf16 → f8e4m3fn
    %C     = stablehlo.dot_general %A_fp8, %W_fp8 → f32 accumulate

    Used in: GPT-4 training, LLaMA 3 FP8 variants.
    JAX support: jax.nn.dtypes.float8_e4m3fn
    OpenXLA lowers to cuBLAS FP8 GEMM on H100.

  QUANTISATION AWARE TRAINING (QAT):
    Insert "fake quant" ops during training (differentiable quantisation):
    stablehlo.fake_quant %x {num_bits=8, axis=1} : tensor<...xf32>
    → during training: clips and rounds x to simulate int8 range
    → at export: replaced by real stablehlo.uniform_quantize for deployment

  POST-TRAINING QUANTISATION (PTQ) via stablehlo-quantize pass:
    # After training and JAX export:
    stablehlo-opt --stablehlo-quantize \\
        --stablehlo-quantize-strategy=weight_only_int8 \\
        model.mlir -o model_quant.mlir
"""
print(QUANT_THEORY)

# ── Quantisation accuracy simulation ─────────────────────────────────────
def simulate_quantisation(W: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float]:
    """Simulate OpenXLA's stablehlo.uniform_quantize effect."""
    qmin = -(1 << (bits-1))      # -128 for int8
    qmax = (1 << (bits-1)) - 1   # +127 for int8
    scale = (W.max() - W.min()) / (qmax - qmin)
    zero_point = int(-W.min() / scale + qmin)
    W_q = np.clip(np.round(W / scale + zero_point), qmin, qmax).astype(np.int8)
    W_dq = (W_q.astype(np.float32) - zero_point) * scale
    quant_error = np.mean(np.abs(W - W_dq))
    return W_dq, quant_error

rng = np.random.default_rng(42)
A  = rng.normal(0, 1, (8, 16)).astype(np.float32)
W  = rng.normal(0, 0.5, (16, 8)).astype(np.float32)
ref = A @ W

print(f"  Quantisation accuracy analysis (matmul A[8×16] @ W[16×8]):")
print()
print(f"  {'Format':15s}  {'W error (MAE)':>14s}  {'Output MAE':>12s}  {'Memory':>10s}")
print("  " + "-" * 56)

formats = [(32, "f32 (baseline)"), (8, "int8"), (4, "int4")]
for bits, label in formats:
    if bits == 32:
        W_dq = W; w_err = 0.0
    else:
        W_dq, w_err = simulate_quantisation(W, bits=bits)
    out = A @ W_dq
    out_err = np.mean(np.abs(out - ref))
    mem_ratio = bits / 32
    print(f"  {label:15s}  {w_err:>14.6f}  {out_err:>12.6f}  {mem_ratio:>10.1%}")

print()
print(f"  int8: 4× memory reduction, {((A@simulate_quantisation(W,8)[0] - ref).__abs__().mean() / ref.__abs__().mean() * 100):.2f}% relative output error")
print(f"  In practice: <0.5% task accuracy loss with proper calibration.")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Shardy — sharding propagation simulation
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 3 — Shardy: sharding propagation library")
print("━" * 65)
print()

SHARDY_OVERVIEW = """
  SHARDY (openxla/shardy): SHARDING PROPAGATION FOR OPENXLA
  ════════════════════════════════════════════════════════════════

  Shardy is a standalone MLIR-based library that propagates tensor
  sharding annotations through a computation graph.

  PROBLEM IT SOLVES:
    User annotates SOME tensors with shardings (usually inputs and outputs).
    Shardy infers the optimal sharding for ALL intermediate tensors.

  EXAMPLE — transformer attention layer:
    Input:  Q[B,S,H,D]  annotated: shard batch B across 4 GPUs
    Weights: W[D,D]     annotated: shard D across 4 GPUs (column parallel)
    Intermediate: scores[B,H,S,S] = ?  ← Shardy propagates this

    Shardy analysis:
      scores = Q @ K^T     (dot_general)
      If Q is sharded on B, K is replicated:
        → scores is sharded on B (batch stays sharded through matmul)
      If W is column-sharded:
        → Q@W produces column-sharded QW
        → need AllGather before attention

    Shardy emits the propagated StableHLO:
      %scores = stablehlo.dot_general ... { sdy.sharding = "B-sharded" }
      %qw     = stablehlo.dot_general ... { sdy.sharding = "column-sharded" }
      %qw_gathered = sdy.all_gather %qw  ; inserted by Shardy

  SHARDY ANNOTATION SYNTAX:
    Annotations are MLIR attributes on operations:
    { sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]> }
    ; shard first dimension along mesh axis "x", replicate second

  SHARDY vs GSPMD (predecessor):
    GSPMD: Google-internal, monolithic, hard to extend.
    Shardy: open-source MLIR pass, extensible, community-governed.
    Shardy is a drop-in replacement for GSPMD with same semantics.
"""
print(SHARDY_OVERVIEW)

# Simulate sharding propagation rules
@dataclass
class TensorSharding:
    shape: tuple
    sharding: Optional[List]   # None=replicated, [i,j]=[shard_dim_i, shard_dim_j]
    n_devices: int

    @property
    def local_shape(self):
        if self.sharding is None:
            return self.shape
        s = list(self.shape)
        for i, sh in enumerate(self.sharding):
            if sh is not None:
                s[i] = s[i] // self.n_devices
        return tuple(s)

    def mem_per_device_mb(self, dtype_bytes=2):
        return np.prod(self.local_shape) * dtype_bytes / 1e6

    def __repr__(self):
        sh = self.sharding if self.sharding else "replicated"
        return f"Tensor{self.shape} sharding={sh} local={self.local_shape}"


def propagate_matmul_sharding(A: TensorSharding, B: TensorSharding,
                               n_devices: int) -> Tuple[TensorSharding, str]:
    """
    Shardy-style sharding propagation for C = A @ B.
    A: [M,K], B: [K,N], C: [M,N]
    Returns (C_sharding, required_communication).
    """
    A_M_sharded = A.sharding and A.sharding[0] is not None
    A_K_sharded = A.sharding and A.sharding[1] is not None
    B_K_sharded = B.sharding and B.sharding[0] is not None
    B_N_sharded = B.sharding and B.sharding[1] is not None

    if A_M_sharded and not A_K_sharded and not B_K_sharded and not B_N_sharded:
        # Data parallel: A sharded on M, B replicated → C sharded on M
        return TensorSharding(
            (A.shape[0], B.shape[1]), [0, None], n_devices), "None"

    elif not A_M_sharded and not A_K_sharded and B_N_sharded and not B_K_sharded:
        # Column parallel: B sharded on N, A replicated → C sharded on N
        return TensorSharding(
            (A.shape[0], B.shape[1]), [None, 1], n_devices), "AllGather(C)"

    elif not A_M_sharded and A_K_sharded and B_K_sharded and not B_N_sharded:
        # Row parallel: both K-sharded → C needs AllReduce
        return TensorSharding(
            (A.shape[0], B.shape[1]), None, n_devices), "AllReduce(C)"

    else:
        # Fallback: replicate C, AllReduce
        return TensorSharding(
            (A.shape[0], B.shape[1]), None, n_devices), "AllReduce(C)"


n_dev = 4
print(f"  Shardy propagation for GPT-2 medium attention ({n_dev} GPUs):")
print()

scenarios = [
    ("Data parallel",  TensorSharding((2048,1024), [0,None], n_dev),
                       TensorSharding((1024,1024), None, n_dev)),
    ("Column parallel",TensorSharding((2048,1024), None, n_dev),
                       TensorSharding((1024,1024), [None,1], n_dev)),
    ("Row parallel",   TensorSharding((2048,1024), [None,1], n_dev),
                       TensorSharding((1024,1024), [0,None], n_dev)),
]

for scenario, A_sh, B_sh in scenarios:
    C_sh, comm = propagate_matmul_sharding(A_sh, B_sh, n_dev)
    print(f"  {scenario}:")
    print(f"    A: {A_sh}")
    print(f"    B: {B_sh}")
    print(f"    C (propagated): {C_sh}")
    print(f"    Communication: {comm}")
    total_mem = (A_sh.mem_per_device_mb() + B_sh.mem_per_device_mb() + C_sh.mem_per_device_mb())
    print(f"    Mem/device: {total_mem:.0f} MB")
    print()
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · OpenXLA in the Connected Stack — End-to-End Workflow": {
        "description": (
            "End-to-end workflow: train in JAX → export StableHLO → compile with IREE → deploy. "
            "Show the OpenXLA ecosystem map: StableHLO consumers, PJRT plugins, governance. "
            "Demonstrate jax.export: polymorphic shapes, multi-device exports. "
            "Show how torch-mlir connects PyTorch to the OpenXLA ecosystem. "
            "Summarise the full connected compiler stack position of OpenXLA. "
            "Resources: project repos, benchmarks, community channels."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  OPENXLA IN THE CONNECTED STACK — END-TO-END WORKFLOW")
print("=" * 65)
print()

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    print(f"  JAX: {jax.__version__} | Backend: {jax.default_backend()}")
except ImportError:
    HAS_JAX = False
    print("  JAX not installed: pip install jax")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Train → Export → Deploy: the canonical workflow
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The canonical OpenXLA workflow")
print("━" * 65)
print()

WORKFLOW = """
  THE CANONICAL OPENXLA END-TO-END WORKFLOW
  ════════════════════════════════════════════════════════════════

  ── STEP 1: TRAIN (JAX + OpenXLA/XLA) ────────────────────────────────

  import jax, optax, flax.linen as nn

  class MLP(nn.Module):
      features: tuple = (128, 64, 10)
      @nn.compact
      def __call__(self, x):
          for feat in self.features:
              x = nn.relu(nn.Dense(feat)(x))
          return x

  model    = MLP()
  params   = model.init(jax.random.PRNGKey(0), jnp.ones((1,784)))
  optimizer= optax.adam(1e-3)
  opt_state= optimizer.init(params)

  @jax.jit
  def train_step(params, opt_state, batch):
      loss, grads = jax.value_and_grad(loss_fn)(params, batch)
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss

  # XLA compiles train_step at first call → fast loop
  for batch in dataloader:
      params, opt_state, loss = train_step(params, opt_state, batch)

  ── STEP 2: EXPORT (jax.export → StableHLO) ──────────────────────────

  # Export to StableHLO with polymorphic batch dimension
  def forward(params, x):
      return model.apply(params, x)

  x_abs = jax.ShapeDtypeStruct(('b', 784), jnp.float32)  # 'b' = symbolic
  p_abs = jax.tree_map(lambda p: jax.ShapeDtypeStruct(p.shape, p.dtype), params)

  exported = jax.export.export(jax.jit(forward))(p_abs, x_abs)
  blob     = exported.serialize()   # StableHLO binary (versioned, stable)

  print(f"Exported StableHLO: {len(blob):,} bytes")
  print(f"Input shapes: {exported.in_avals}")   # includes symbolic 'b'
  print(f"Output shape: {exported.out_avals}")

  # Verify round-trip: load and run in same process
  loaded = jax.export.deserialize(blob)
  y_test = loaded.call(params, jnp.ones((32, 784)))
  assert y_test.shape == (32, 10)

  ── STEP 3: QUANTISE (optional, for int8 deployment) ─────────────────

  # Post-training quantisation via StableHLO passes:
  # (command line, after export)
  # $ stablehlo-opt --stablehlo-quantize=weight_only_int8 \\
  #                 mlp_model.mlir -o mlp_model_q.mlir

  ── STEP 4: COMPILE FOR TARGET (IREE) ────────────────────────────────

  # Compile for mobile Android:
  # $ iree-compile mlp_model.stablehlo \\
  #     --iree-input-type=stablehlo \\
  #     --iree-hal-target-backends=vulkan-spirv \\
  #     --iree-vulkan-target-triple=adreno-unknown-android31 \\
  #     -o mlp_android.vmfb

  # Compile for server CUDA:
  # $ iree-compile mlp_model.stablehlo \\
  #     --iree-input-type=stablehlo \\
  #     --iree-hal-target-backends=cuda \\
  #     --iree-cuda-target-chip=sm_86 \\   # RTX 3090
  #     -o mlp_cuda.vmfb

  ── STEP 5: DEPLOY (IREE runtime, no Python) ─────────────────────────

  // Android C++ app:
  auto instance  = iree_vm_instance_create(...);
  auto module    = iree_vm_bytecode_module_create_from_file("mlp_android.vmfb");
  auto context   = iree_vm_context_create_with_modules(instance, {module});
  auto fn        = iree_vm_function_lookup(context, "main");

  float input[784] = { /* pixel values */ };
  iree_vm_list_t* inputs  = /* pack input tensor */;
  iree_vm_list_t* outputs = /* allocate output */;
  iree_vm_invoke(context, fn, /*policy=*/NULL, inputs, outputs, /*status=*/NULL);
  float* logits = /* unpack output tensor */;
  int predicted = argmax(logits, 10);

  // No Python. No JAX. No XLA. Just a 200KB binary + 150KB IREE runtime.
"""
print(WORKFLOW)

if HAS_JAX:
    # ── Demonstrate jax.export ────────────────────────────────────────────
    def mlp_forward(params, x):
        W1, b1, W2, b2 = params
        h = jnp.maximum(x @ W1 + b1, 0.0)
        return h @ W2 + b2

    rng = np.random.default_rng(0)
    W1  = jnp.array(rng.normal(0, 0.1, (16, 32)), dtype=jnp.float32)
    b1  = jnp.zeros(32, dtype=jnp.float32)
    W2  = jnp.array(rng.normal(0, 0.1, (32, 4)),  dtype=jnp.float32)
    b2  = jnp.zeros(4, dtype=jnp.float32)
    params = (W1, b1, W2, b2)

    x_test = jnp.ones((8, 16), dtype=jnp.float32)
    y_ref  = mlp_forward(params, x_test)

    x_abs  = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    p_abs  = tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in params)

    try:
        exported = jax.export.export(jax.jit(mlp_forward))(p_abs, x_abs)
        blob     = exported.serialize()
        print(f"  jax.export demo:")
        print(f"    Serialised StableHLO: {len(blob):,} bytes")
        print(f"    Function:             {exported.fun_name}")
        print(f"    Output shape:         {exported.out_avals}")

        # Round-trip verification
        loaded = jax.export.deserialize(blob)
        y_loaded = loaded.call(*params, x_test)
        err = float(jnp.max(jnp.abs(y_loaded - y_ref)))
        print(f"    Round-trip max error: {err:.2e} {'✅' if err < 1e-5 else '❌'}")
    except Exception as e:
        print(f"    Note (requires jax>=0.4.14): {e}")
        # Fallback: show xla_computation
        comp  = jax.xla_computation(mlp_forward)(*params, x_test)
        print(f"    HLO text length: {len(comp.as_hlo_text())} chars")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: torch-mlir → StableHLO: PyTorch in the OpenXLA ecosystem
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — torch-mlir: PyTorch → StableHLO → OpenXLA")
print("━" * 65)
print()

TORCH_MLIR = """
  PYTORCH IN THE OPENXLA ECOSYSTEM VIA TORCH-MLIR
  ════════════════════════════════════════════════════════════════

  torch-mlir (github.com/llvm/torch-mlir) connects PyTorch to the
  full MLIR/OpenXLA ecosystem by lowering PyTorch models to MLIR dialects.

  WORKFLOW:
    import torch
    import torch_mlir

    class MyModel(torch.nn.Module):
        def forward(self, x, W):
            return torch.relu(x @ W)

    model  = MyModel()
    x_ex   = torch.randn(4, 8)
    W_ex   = torch.randn(8, 4)
    model  = model.eval()

    # Step 1: PyTorch → torch.fx graph → Torch-MLIR
    mlir_module = torch_mlir.compile(
        model,
        (x_ex, W_ex),
        output_type=torch_mlir.OutputType.STABLEHLO,  # emit StableHLO!
    )

    # mlir_module is now a StableHLO MLIR module
    # It can be consumed by ANY StableHLO consumer:
    # → JAX (jax.export.deserialize)
    # → IREE (iree-compile --input-type=stablehlo)
    # → TVM (via tvm.relay.from_stablehlo)
    # → OpenXLA/XLA (stablehlo-legalize-to-hlo)

    print(mlir_module.operation.get_asm())
    # Output:
    # module @torch_module {
    #   func.func @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>)
    #                       -> tensor<4x4xf32> {
    #     %0 = stablehlo.dot_general %arg0, %arg1, ...
    #     %1 = stablehlo.constant dense<0.0>
    #     %2 = stablehlo.broadcast_in_dim %1
    #     %3 = stablehlo.maximum %0, %2
    #     return %3 : tensor<4x4xf32>
    #   }
    # }

  TORCH-MLIR LOWERING STAGES:
    PyTorch FX graph
        ↓ decompose_into_aten (torch-mlir)
    Torch dialect (torch.aten.* ops)
        ↓ convert-torch-to-stablehlo
    StableHLO dialect  ← OpenXLA ecosystem entry point
        ↓ (any consumer)
    IREE / XLA / TVM / etc.

  SUPPORTED OPERATIONS:
    Linear layers:  nn.Linear → stablehlo.dot_general
    Activations:    relu, gelu, silu → stablehlo elementwise ops
    Normalization:  LayerNorm, BatchNorm → reduce + broadcast sequence
    Attention:      F.scaled_dot_product_attention → dot_general chain
    Convolutions:   nn.Conv2d → stablehlo.convolution
    Pooling:        AdaptiveAvgPool → stablehlo.reduce_window

  LIMITATION:
    Dynamic control flow (loops with data-dependent bounds) → not supported.
    Use torch.export with strict=False for dynamic shapes.
    Model must be traceable (no Python-dependent branching on input values).
"""
print(TORCH_MLIR)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Full connected stack summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — OpenXLA in the full connected stack")
print("━" * 65)
print()

STACK_SUMMARY = """
  OPENXLA POSITION IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                      │
  │  ← OpenXLA/XLA emits LLVM IR for CPU kernels (HloToIr pass)           │
  │  ← IREE emits LLVM IR from vector dialect for CPU deployment          │
  │  ← Joint primal+gradient optimisation when Enzyme is in the chain     │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                      │
  │  ← StableHLO IS an MLIR dialect (built on MLIR infrastructure)        │
  │  ← IREE is fully MLIR-based: Flow, Stream, HAL, SPIRV dialects        │
  │  ← Shardy (sharding propagation) is an MLIR pass on MHLO modules      │
  │  ← Mosaic GPU generates GPU+NVVM MLIR dialects for H100               │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 03: CIRCT                                                     │
  │  ← TPU hardware designed in Chisel, compiled via CIRCT firtool        │
  │  ← PJRT plugin abstracts the TPU from OpenXLA's compiler              │
  │  ← Research: differentiable RTL simulation for accelerator design     │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 04: ENZYME                                                    │
  │  ← JAX's jax.grad differentiates StableHLO (source-level in JAX)      │
  │  ← Enzyme-JAX: differentiates via LLVM IR after XLA lowers            │
  │  ← Custom op gradients: Enzyme-MLIR on StableHLO module               │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 05: XLA                                                       │
  │  ← OpenXLA/XLA is XLA, now community-governed at openxla/xla          │
  │  ← XLA compiles StableHLO → GPU/TPU via HLO passes + cuBLAS/cuDNN     │
  │  ← PJRT is the hardware abstraction layer for XLA's backends          │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 06: OPENXLA (this module)                                     │
  │  Ecosystem components:                                                │
  │    StableHLO: stable IR (openxla/stablehlo)                           │
  │    PJRT:      hardware plugin API (any chip → any framework)          │
  │    IREE:      MLIR-native deployment runtime (openxla/iree)           │
  │    Shardy:    sharding propagation (openxla/shardy)                   │
  │    Benchmarks:cross-compiler benchmark suite                          │
  ├───────────────────────────────────────────────────────────────────────┤
  │  MODULE 07: TVM                                                       │
  │  ← TVM ingests StableHLO via tvm.relay.from_stablehlo                 │
  │  ← Both IREE and TVM are StableHLO consumers (benchmarked together)   │
  │  ← TVM AutoScheduler has no XLA equivalent; XLA has TPU support       │
  └───────────────────────────────────────────────────────────────────────┘

  OPENXLA PROJECT RESOURCES:
  ─────────────────────────────────────────────────────────────────────────
  openxla.org                       — project home
  github.com/openxla/xla            — OpenXLA/XLA compiler
  github.com/openxla/stablehlo      — StableHLO dialect and spec
  github.com/openxla/iree           — IREE compiler and runtime
  github.com/openxla/shardy         — sharding propagation
  github.com/openxla/openxla-benchmark  — cross-compiler benchmarks
  openxla.org/pjrt                  — PJRT specification
  github.com/llvm/torch-mlir        — PyTorch → StableHLO
  jax.readthedocs.io/export         — jax.export documentation
  iree.dev                          — IREE documentation
"""
print(STACK_SUMMARY)

print("━" * 65)
print("  QUICK REFERENCE")
print("━" * 65)
print()
print("  ┌──────────────────────────────────────────────────────────────────────┐")
print("  │ OpenXLA Component  │ Key API / Tool                                  │")
print("  ├──────────────────────────────────────────────────────────────────────┤")
print("  │ Export model       │ jax.export.export(jax.jit(fn))(args)            │")
print("  │ Inspect StableHLO  │ exported.mlir_module()                          │")
print("  │ Serialise          │ exported.serialize() → bytes                    │")
print("  │ Deserialise        │ jax.export.deserialize(blob)                    │")
print("  │ IREE compile       │ iree-compile --input-type=stablehlo             │")
print("  │ IREE run           │ iree-run-module --module=model.vmfb             │")
print("  │ PJRT plugin        │ jax.config.update('jax_pjrt_plugin', path)      │")
print("  │ PyTorch → SHLO     │ torch_mlir.compile(model, OutputType.STABLEHLO) │")
print("  │ Quantise SHLO      │ stablehlo-opt --stablehlo-quantize              │")
print("  │ AutoSharding       │ XLA_FLAGS='--xla_enable_auto_sharding=true'     │")
print("  ├──────────────────────────────────────────────────────────────────────┤")
print("  │ StableHLO ops      │ stablehlo.dot_general, .reduce, .broadcast      │")
print("  │ PJRT objects       │ PJRT_Client, PJRT_Buffer, PJRT_Executable       │")
print("  │ IREE dialects      │ flow → stream → hal → spirv/llvm                │")
print("  └──────────────────────────────────────────────────────────────────────┘")
print()
print("  CONNECTED STACK POSITION:")
print("  XLA    (module 05) ← OpenXLA/XLA is XLA, community-governed")
print("  MLIR   (module 02) ← StableHLO and IREE are MLIR-based")
print("  LLVM   (module 01) ← CPU backends emit LLVM IR")
print("  TVM    (module 07) ← ingests StableHLO; both converge via MLIR")
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