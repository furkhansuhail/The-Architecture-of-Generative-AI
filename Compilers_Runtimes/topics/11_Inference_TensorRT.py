"""
TensorRT — High-Performance Deep Learning Inference
=====================================================

NVIDIA's production inference engine. Every millisecond you shave off
latency in deployment means real cost savings and better user experience.
TensorRT takes your trained model and compiles it into a GPU-optimised
"engine" that runs as fast as your hardware allows.

Understanding the build/execute lifecycle, precision modes, memory
management, and ONNX integration lets you deploy models that are
2–10× faster than naive framework inference — from first principles.

"""

import textwrap
import re

TOPIC_NAME   = "TensorRT — High-Performance Deep Learning Inference"
DISPLAY_NAME = "11 · TensorRT"
ICON         = "⚡"
SUBTITLE     = "Compile Once, Infer at the Speed of Your GPU"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT IS TENSORRT AND WHY IT EXISTS

### The Training vs Inference Gap

    Training:    maximise accuracy, use all the tricks, slow is OK
    Inference:   serve predictions in milliseconds, at scale, cheaply

    The problem: frameworks like PyTorch are built for flexibility during
    training (autograd, dynamic graphs, Python overhead). At inference time
    you pay for features you don't need.

    TensorRT solves this by:
    1. Knowing your model is FIXED at deployment time
    2. Knowing your TARGET GPU at build time
    3. Exploiting both to generate hardware-specific optimised code

### The Hardware Stack TensorRT Sits On

    ┌─────────────────────────────────────────────────────────┐
    │  Your application code (Python or C++)                  │
    ├─────────────────────────────────────────────────────────┤
    │  TensorRT Engine (optimised for YOUR GPU)               │
    ├─────────────────────────────────────────────────────────┤
    │  cuDNN  (NVIDIA deep learning primitives library)        │
    ├─────────────────────────────────────────────────────────┤
    │  CUDA   (general GPU computing toolkit)                  │
    ├─────────────────────────────────────────────────────────┤
    │  NVIDIA GPU Driver                                      │
    ├─────────────────────────────────────────────────────────┤
    │  NVIDIA GPU Hardware (Ampere, Ada, Hopper, etc.)         │
    └─────────────────────────────────────────────────────────┘

### The Two Phases: Build Time vs Runtime

    BUILD TIME (slow, done once):
        Input:  trained model (ONNX / PyTorch / TF)
        Action: TensorRT analyses, fuses, calibrates, kernel-selects
        Output: serialised .engine file (binary blob for your GPU)

    RUNTIME (fast, done on every request):
        Input:  .engine file + input tensor
        Action: load engine → allocate GPU memory → execute → read output
        Output: prediction tensor

    ┌─────────────────────────┐     ┌──────────────────────────┐
    │     BUILD PHASE          │     │     RUNTIME PHASE         │
    │  (minutes, once)         │     │  (milliseconds, always)   │
    │                         │     │                           │
    │  model.onnx             │     │  engine.plan              │
    │       ↓                 │     │       ↓                   │
    │  trt.Builder            │ ──► │  engine.create_context()  │
    │       ↓                 │     │       ↓                   │
    │  engine.serialize()     │     │  context.execute_v2()     │
    │       ↓                 │     │       ↓                   │
    │  engine.plan  (saved)   │     │  output tensor            │
    └─────────────────────────┘     └──────────────────────────┘


##### PART 2 — CORE TensorRT OBJECTS: Builder, Network, Config, Engine

### The Four Core Objects

    trt.Logger:
        Controls verbosity of TensorRT messages.
        Levels: VERBOSE > INFO > WARNING > ERROR > INTERNAL_ERROR
        Always create this first — everything else needs it.

    trt.Builder:
        The factory. Creates networks and configs.
        Tied to a specific GPU — build on the GPU you'll run on.
        Usage: builder = trt.Builder(logger)

    trt.INetworkDefinition:
        The computational graph.
        Nodes = layers, edges = tensors.
        You either define it manually (add_input, add_convolution, ...)
        or parse it from ONNX (the common real-world path).
        Usage: network = builder.create_network(flags)

    trt.IBuilderConfig:
        All knobs: precision (FP32/FP16/INT8), workspace memory, profiles.
        Usage: config = builder.create_builder_config()

    trt.ICudaEngine:
        The compiled, optimised result.
        Can be serialised to disk and reloaded later.
        Usage: engine = builder.build_engine(network, config)

    trt.IExecutionContext:
        A stateful handle for running inference.
        One engine can have multiple contexts (for concurrent execution).
        Usage: context = engine.create_execution_context()

### Object Lifecycle Diagram

    logger ──────────────────────────────────────────────┐
       │                                                  │
    builder ──► network ──► (add layers / parse ONNX)    │
       │           │                                      │
       │        config ──► (set precision, workspace)     │
       │                                                  │
       └──────► engine ──► serialise() → .plan file       │
                   │                                      │
               context ──► execute_v2(bindings) ──────────┘

### Network Flags (important!)

    When calling builder.create_network(), you must pass flags:

    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    This flag is REQUIRED for ONNX parsing and dynamic shapes.
    It tells TensorRT the batch dimension is explicitly part of the tensor shape,
    not handled implicitly. Always use it in modern TensorRT (≥ 7.1).

    Without it: shape (C, H, W) — batch implicit
    With it:    shape (N, C, H, W) — batch explicit ← always use this


##### PART 3 — PRECISION MODES: FP32, FP16, INT8

### Why Precision Matters

    Floating point numbers use bits to represent values.
    Fewer bits = less memory, faster math — but less precision.

    FP32 (float32):  32 bits = 1 sign + 8 exponent + 23 mantissa
        Precision:   ~7 decimal digits
        Range:       ~1.2×10⁻³⁸ to ~3.4×10³⁸
        GPU support: all NVIDIA GPUs

    FP16 (float16):  16 bits = 1 sign + 5 exponent + 10 mantissa
        Precision:   ~3 decimal digits
        Range:       ~5.96×10⁻⁸ to 65,504
        Speedup:     ~2–4× over FP32 on Tensor Core GPUs
        GPU support: Pascal and later (GTX 1080, V100, A100, ...)

    INT8 (int8):     8 bits = integer in [-128, 127]
        No fractional part — all values must be QUANTISED
        Speedup:     ~2–4× over FP16 (~4–8× over FP32)
        Accuracy:    small drop if calibrated properly
        GPU support: Turing and later (T4, A100, RTX 30xx, ...)

### FP32 vs FP16 Comparison

    ┌──────────┬──────────────┬───────────┬──────────┬─────────────────┐
    │ Mode     │ Memory/param │ Accuracy  │ Speedup  │ When to use     │
    ├──────────┼──────────────┼───────────┼──────────┼─────────────────┤
    │ FP32     │   4 bytes    │ Reference │   1×     │ Baseline/debug  │
    │ FP16     │   2 bytes    │ ≈ FP32    │  2–4×    │ Production std  │
    │ INT8     │   1 byte     │ -0.1–0.5% │  4–8×    │ Max throughput  │
    └──────────┴──────────────┴───────────┴──────────┴─────────────────┘

### INT8 Calibration — Why It's Needed

    INT8 cannot directly represent floats.
    We need a SCALE FACTOR: float32_value = int8_value × scale

    Finding the right scale requires CALIBRATION:
        Feed ~100–1000 representative input samples
        TensorRT observes the actual value range of each tensor
        Computes optimal scale to minimise quantisation error

    Calibration algorithms:
        MinMax:     scale = max(|x|) / 127   — simple, fast
        Entropy:    minimises KL divergence  — more accurate (IInt8EntropyCalibrator2)
        Percentile: clips outliers           — handles long-tail distributions

    ┌────────────────────────────────────────────────────────┐
    │  INT8 without calibration = random incorrect results   │
    │  INT8 with calibration    = nearly FP32 accuracy       │
    │                                                        │
    │  Think of it like: you need to know the "vocabulary"   │
    │  of values your model uses before you compress it      │
    └────────────────────────────────────────────────────────┘

### How TensorRT Mixes Precision (Layer-by-Layer)

    TensorRT doesn't force every layer into the same precision.
    It will automatically keep numerically sensitive layers in FP32
    even when you request FP16 or INT8:

    config.set_flag(trt.BuilderFlag.FP16)  ← hint, not a mandate

    Diagram — Mixed precision in a typical CNN:

    Input (FP32)
        ↓
    Conv1   [INT8 ← fast, many channels]
        ↓
    BN1     [FP32 ← sensitive to precision]
        ↓
    Conv2   [INT8]
        ↓
    Softmax [FP32 ← exponential, needs range]
        ↓
    Output (FP32)


##### PART 4 — MEMORY MANAGEMENT: HOST ↔ DEVICE TRANSFERS

### The Memory Hierarchy

    CPU RAM  (Host memory):      Large, slow, your Python arrays live here
    GPU VRAM (Device memory):    Small, blazing fast, computations happen here

    PCIe bus connects them:      transfers are ~15–32 GB/s (vs GPU's 1+ TB/s)

    The golden rule:
        Minimise host↔device transfers — they kill your latency budget!

### pycuda Memory Operations

    cuda.mem_alloc(n_bytes):     allocate n bytes on GPU, returns device pointer
    cuda.memcpy_htod(dst, src):  Host to Device (CPU → GPU)
    cuda.memcpy_dtoh(dst, src):  Device to Host (GPU → CPU)
    cuda.memcpy_htod_async():    async version — overlaps with compute

    Memory layout rules:
        Arrays must be C-contiguous (numpy default): arr = np.ascontiguousarray(arr)
        dtype must match the engine's expected type (float32 for FP32 engines)

### Bindings — Connecting Memory to the Engine

    TensorRT engine has "bindings" — numbered slots for inputs and outputs.
    You pass a list of GPU pointers in binding index order.

    binding index 0 → input  tensor  "images"
    binding index 1 → output tensor  "logits"

    context.execute_v2(bindings=[int(input_ptr), int(output_ptr)])

    How to find binding indices:

        for i in range(engine.num_bindings):
            name    = engine.get_binding_name(i)
            shape   = engine.get_binding_shape(i)
            is_inp  = engine.binding_is_input(i)
            print(f"  [{i}] {name} shape={shape}  input={is_inp}")

### CUDA Streams — Async Execution

    A CUDA stream is a queue of GPU operations executed in order.
    Multiple streams can run concurrently (if GPU has capacity).

    Why use streams?
        Async copies overlap with GPU computation → higher throughput

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()   ← wait for everything to finish

    Diagram — Synchronous vs Async pipeline:

    Synchronous:
        [H→D copy] [GPU compute] [D→H copy] [H→D copy] [GPU compute] [D→H copy]
        Time ──────────────────────────────────────────────────────────────────►

    Asynchronous (pipelined):
        [H→D copy #1][GPU compute #1]
                      [H→D copy #2] [D→H copy #1][GPU compute #2]
                                                  [H→D copy #3] [D→H copy #2]
        Time ──────────────────────────────────────────────────────────────────►
        Result: ~2× throughput with double-buffering


##### PART 5 — ONNX: THE REAL-WORLD WORKFLOW

### What is ONNX?

    Open Neural Network Exchange — a standard format for neural network graphs.
    Framework-agnostic: PyTorch → ONNX → TensorRT (or any other runtime)

    ONNX represents the model as a graph of operations (nodes) and tensors.
    Every major framework can export to ONNX and import from ONNX.

### The Canonical Deployment Pipeline

    Step 1: Train in PyTorch
        model = MyModel()
        model.train(...)

    Step 2: Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,          ← defines input shape
            "model.onnx",
            opset_version=17,     ← ONNX operator set version
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}}  ← optional
        )

    Step 3: Verify with onnxruntime (before TensorRT)
        import onnxruntime as ort
        sess = ort.InferenceSession("model.onnx")
        out  = sess.run(None, {"input": sample})   ← sanity check

    Step 4: Parse into TensorRT
        parser = trt.OnnxParser(network, logger)
        with open("model.onnx", "rb") as f:
            parser.parse(f.read())

    Step 5: Build engine & serialise
        engine = builder.build_engine(network, config)
        with open("model.plan", "wb") as f:
            f.write(engine.serialize())

    Step 6: Load & infer at runtime (forever after)
        runtime = trt.Runtime(logger)
        with open("model.plan", "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        # ... allocate memory, run inference

### Common ONNX Export Pitfalls

    ┌─────────────────────────────┬──────────────────────────────────────┐
    │ Problem                     │ Fix                                  │
    ├─────────────────────────────┼──────────────────────────────────────┤
    │ Unsupported op in TensorRT  │ Use an earlier opset; add plugin     │
    │ Shape mismatch at runtime   │ Use dynamic_axes or fixed shapes     │
    │ model.eval() forgotten      │ Always call model.eval() before export│
    │ Custom layers missing       │ Implement IPluginV2 or decompose     │
    │ Wrong opset_version         │ TRT 8.x supports opset ≤ 17          │
    └─────────────────────────────┴──────────────────────────────────────┘


##### PART 6 — GRAPH OPTIMISATIONS TENSORRT APPLIES

### Layer Fusion — The Biggest Win

    Many consecutive layers can be fused into a single GPU kernel.
    Fewer kernel launches = less overhead = faster execution.

    Common fusions:
        Conv + BN + ReLU → single fused kernel  (extremely common)
        Conv + Add       → fused in-place
        MatMul + Bias    → single GEMM with bias

    Before fusion (3 kernel launches, 3× memory round trips):
        Input → [Conv kernel] → temp1 → [BN kernel] → temp2 → [ReLU kernel] → Output

    After fusion (1 kernel launch, 1 memory round trip):
        Input → [Conv+BN+ReLU fused kernel] → Output

### Kernel Auto-Tuning (tactic selection)

    For each layer, TensorRT tries MULTIPLE implementations (tactics):
        - Direct convolution
        - Winograd convolution
        - FFT-based convolution
        - GEMM-based im2col convolution

    It benchmarks EACH tactic on YOUR GPU and picks the fastest.
    This is why building the engine takes time — it's running experiments.

    config.set_tactic_sources(...)  ← control which algorithm families are tried

### Constant Folding

    Any computation that doesn't depend on the input is pre-computed at build time.
    Example: a lookup table, a fixed bias addition, shape computations.

### Dead Code Elimination

    Layers whose outputs are never used are removed automatically.
    (Happens when ONNX exports auxiliary outputs you don't need at inference time.)

### Workspace Memory

    TensorRT needs scratch space for intermediate computations during optimisation.
    More workspace = more tactic options = potentially faster engine.

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    Rule of thumb:  set it to as much free GPU VRAM as you can spare.
    Minimum: 256 MB.  Sweet spot: 1–4 GB for most models.


##### PART 7 — DYNAMIC SHAPES

### Static vs Dynamic Shapes

    Static shapes:  engine compiled for ONE fixed input shape
        Fastest option — TensorRT knows everything at build time
        Problem: a new batch size requires rebuilding the engine

    Dynamic shapes:  engine handles a RANGE of input shapes
        Requires an OptimizationProfile describing min/opt/max shapes
        Small overhead per inference, but one engine handles many sizes

### Optimization Profile

    profile = builder.create_optimization_profile()

    profile.set_shape(
        "input_name",
        min=(1, 3, 224, 224),   ← smallest input you'll ever send
        opt=(8, 3, 224, 224),   ← the most COMMON input (engine optimised for this)
        max=(32, 3, 224, 224),  ← largest input you'll ever send
    )

    config.add_optimization_profile(profile)

    Critical rule:
        opt should match your MOST COMMON batch size in production.
        TensorRT optimises kernels for opt — performance degrades as you
        move away from it toward min or max.

### Multiple Profiles

    You can add multiple profiles to one engine:
        Profile 0: batch 1–4    (real-time, latency-sensitive)
        Profile 1: batch 16–64  (throughput mode)

    At runtime, select the active profile:
        context.active_optimization_profile = 0


##### PART 8 — PERFORMANCE BENCHMARKING AND PROFILING

### Latency vs Throughput

    Latency:    time for ONE inference to complete   → minimise for real-time
    Throughput: inferences per second                → maximise for batch jobs

    They trade off against each other:
        Batch size 1  → low latency,  low throughput
        Batch size 64 → high latency, high throughput

### How to Benchmark Correctly

    1. WARM UP FIRST — first few runs are slower (GPU cold start, caching)
       for _ in range(20):
           context.execute_v2(bindings)

    2. USE CUDA EVENTS for accurate GPU timing (not Python time.time())
       start = cuda.Event()
       end   = cuda.Event()
       start.record()
       context.execute_v2(bindings)
       end.record()
       end.synchronize()
       ms = start.time_till(end)   ← milliseconds on the GPU

    3. AVERAGE OVER MANY RUNS (≥ 100) to reduce variance

    4. MEASURE ON TARGET HARDWARE — engine is hardware-specific

### The Profiling Hierarchy

    Where is time being spent?

    Level 1: Total inference time (CUDA events)
    Level 2: Per-layer timing via IProfiler
        class MyProfiler(trt.IProfiler):
            def report_layer_time(self, layer_name, ms):
                print(f"  {layer_name}: {ms:.3f} ms")

        context.profiler = MyProfiler()
        context.execute_v2(bindings)   ← prints timing for every layer

    Level 3: NVIDIA Nsight Systems (full GPU timeline)
    Level 4: NVIDIA Nsight Compute  (kernel-level hardware counters)

### Key Metrics to Track

    ┌──────────────────────────┬───────────────────────────────────────┐
    │ Metric                   │ What it means                         │
    ├──────────────────────────┼───────────────────────────────────────┤
    │ Latency (ms)             │ Time per batch (lower is better)      │
    │ Throughput (imgs/s)      │ Batches/s × batch_size                │
    │ GPU utilisation (%)      │ Are you feeding the GPU fast enough?  │
    │ Memory bandwidth (GB/s)  │ Are you memory-bound?                 │
    │ Tensor Core util (%)     │ Are FP16/INT8 Tensor Cores active?    │
    └──────────────────────────┴───────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Hello World — Build a ReLU Engine from Scratch": {
        "description": (
            "The TensorRT 'Hello World'. Manually define a tiny 1-layer network "
            "(ReLU activation), compile it into a TensorRT engine, allocate GPU memory, "
            "run inference, and verify the result. Covers the full build/execute lifecycle. "
            "Requires: tensorrt, pycuda, numpy."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  TENSORRT HELLO WORLD — BUILD & RUN A RELU ENGINE")
print("=" * 65)
print()

# ── Imports with friendly error ────────────────────────────────────────────
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit          # initialises CUDA automatically
except ImportError as e:
    print(f"  ImportError: {e}")
    print()
    print("  TensorRT requires:")
    print("  1. An NVIDIA GPU with the appropriate driver")
    print("  2. CUDA toolkit installed at the system level")
    print("  3. cuDNN installed at the system level")
    print("  4. TensorRT installed at the system level")
    print("  5. pip install tensorrt pycuda numpy")
    print()
    print("  Easiest setup: use NVIDIA's official Docker container:")
    print("  docker pull nvcr.io/nvidia/tensorrt:24.01-py3")
    raise SystemExit(0)

print(f"  TensorRT version: {trt.__version__}")
print()

# ── STEP 1: Logger ─────────────────────────────────────────────────────────
# WARNING level: only print problems, not every internal decision
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
print("  [1/5] Logger created")

# ── STEP 2: Builder ────────────────────────────────────────────────────────
builder = trt.Builder(TRT_LOGGER)
print("  [2/5] Builder created  (tied to GPU: builder knows your hardware)")

# ── STEP 3: Network ────────────────────────────────────────────────────────
# EXPLICIT_BATCH flag: batch dimension IS part of the tensor shape (required modern TRT)
EXPLICIT = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network  = builder.create_network(EXPLICIT)

# Define the network: (4,) float32 input → ReLU activation → (4,) output
input_tensor = network.add_input(
    name  = "input",
    dtype = trt.float32,
    shape = (4,)          # 4-element vector
)
relu_layer = network.add_activation(input_tensor, trt.ActivationType.RELU)
network.mark_output(relu_layer.get_output(0))

print("  [3/5] Network defined:")
print(f"         Input  shape: {input_tensor.shape}")
print(f"         Layer:  ReLU activation")
print(f"         Output shape: {relu_layer.get_output(0).shape}")
print()

# ── STEP 4: Build Config & Engine ──────────────────────────────────────────
config = builder.create_builder_config()
# Give TensorRT 256 MB of scratch workspace for kernel selection
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * (1 << 20))

print("  [4/5] Building engine — TensorRT is selecting optimal GPU kernels...")
engine = builder.build_engine(network, config)
print("  ✅ Engine built successfully!")
print()

# Show binding information
print("  Engine bindings:")
for i in range(engine.num_bindings):
    name   = engine.get_binding_name(i)
    shape  = engine.get_binding_shape(i)
    is_inp = engine.binding_is_input(i)
    dtype  = engine.get_binding_dtype(i)
    tag    = "INPUT " if is_inp else "OUTPUT"
    print(f"    [{i}] {tag}  name={name!r}  shape={tuple(shape)}  dtype={dtype}")
print()

# ── STEP 5: Run Inference ──────────────────────────────────────────────────
print("  [5/5] Running inference:")
print()

context    = engine.create_execution_context()
input_data = np.array([-5.0, -1.5, 0.0, 3.7], dtype=np.float32)
print(f"  Input:  {input_data}")
print(f"         (ReLU zeroes the negatives: max(0, x))")

# Allocate GPU memory for input and output (both same shape here)
d_input  = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(input_data.nbytes)

# Transfer: CPU → GPU
cuda.memcpy_htod(d_input, input_data)

# Execute the engine
context.execute_v2(bindings=[int(d_input), int(d_output)])

# Transfer: GPU → CPU
output = np.empty_like(input_data)
cuda.memcpy_dtoh(output, d_output)

print(f"  Output: {output}")
print()

# Verify correctness with numpy
expected = np.maximum(0, input_data)
match    = np.allclose(output, expected)
print(f"  numpy ReLU: {expected}")
print(f"  ✅ TensorRT output matches numpy: {match}")
print()

print("  LIFECYCLE SUMMARY:")
print("  ┌──────────────────┬──────────────────────────────────────────┐")
print("  │  Object          │  Role                                    │")
print("  ├──────────────────┼──────────────────────────────────────────┤")
print("  │  Logger          │  Captures TensorRT diagnostic messages    │")
print("  │  Builder         │  Factory — creates networks and engines   │")
print("  │  Network         │  Computational graph (layers + tensors)   │")
print("  │  Config          │  Build options (precision, workspace)     │")
print("  │  Engine          │  Compiled, GPU-optimised model            │")
print("  │  Context         │  Execution state (can have many per eng.) │")
print("  │  cuda.mem_alloc  │  Allocate memory on the GPU              │")
print("  │  memcpy_htod     │  Host→Device transfer (CPU→GPU)          │")
print("  │  execute_v2      │  Run the engine                          │")
print("  │  memcpy_dtoh     │  Device→Host transfer (GPU→CPU)          │")
print("  └──────────────────┴──────────────────────────────────────────┘")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Precision Modes — FP32 vs FP16 Speed and Accuracy": {
        "description": (
            "Build the same small fully-connected network in FP32 and FP16. "
            "Measure build time, inference time, and output accuracy for both. "
            "Demonstrate that FP16 is ~2x faster with negligible accuracy loss. "
            "Requires: tensorrt, pycuda, numpy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TENSORRT PRECISION MODES — FP32 vs FP16")
print("=" * 65)
print()

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"  ImportError: {e}")
    print("  Install: tensorrt pycuda numpy")
    print("  Requires NVIDIA GPU + CUDA + cuDNN + TensorRT system install")
    raise SystemExit(0)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# ── Network factory ────────────────────────────────────────────────────────
def build_fc_network(builder, input_size=128, hidden=256, output_size=10):
    """
    Fully-connected network:  input(128) → Linear(256) → ReLU → Linear(10)
    Weights are random — we care about timing, not task accuracy here.
    """
    network = builder.create_network(EXPLICIT)
    weights = {}

    # Layer 1: Linear(128 → 256)
    w1 = np.random.randn(256, input_size).astype(np.float32) * 0.01
    b1 = np.zeros(256, dtype=np.float32)
    inp = network.add_input("input", trt.float32, (input_size,))
    fc1 = network.add_fully_connected(inp, 256, trt.Weights(w1), trt.Weights(b1))
    weights['w1'] = w1; weights['b1'] = b1

    # ReLU
    relu = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)

    # Layer 2: Linear(256 → 10)
    w2 = np.random.randn(10, 256).astype(np.float32) * 0.01
    b2 = np.zeros(10, dtype=np.float32)
    fc2 = network.add_fully_connected(relu.get_output(0), 10,
                                       trt.Weights(w2), trt.Weights(b2))
    weights['w2'] = w2; weights['b2'] = b2
    network.mark_output(fc2.get_output(0))
    return network, weights

# ── Benchmark helper ───────────────────────────────────────────────────────
def benchmark_engine(engine, input_size, n_warmup=20, n_runs=200):
    context   = engine.create_execution_context()
    inp_data  = np.random.randn(input_size).astype(np.float32)
    out_data  = np.empty(10, dtype=np.float32)
    d_input   = cuda.mem_alloc(inp_data.nbytes)
    d_output  = cuda.mem_alloc(out_data.nbytes)
    cuda.memcpy_htod(d_input, inp_data)

    # Warm up (GPU cold start — first runs are slower)
    for _ in range(n_warmup):
        context.execute_v2(bindings=[int(d_input), int(d_output)])

    # Timed runs using CUDA Events (more accurate than time.time())
    start_ev = cuda.Event()
    end_ev   = cuda.Event()
    start_ev.record()
    for _ in range(n_runs):
        context.execute_v2(bindings=[int(d_input), int(d_output)])
    end_ev.record()
    end_ev.synchronize()
    total_ms    = start_ev.time_till(end_ev)
    per_run_ms  = total_ms / n_runs

    # Read output for accuracy comparison
    cuda.memcpy_dtoh(out_data, d_output)
    return per_run_ms, out_data

# ── Check FP16 support ─────────────────────────────────────────────────────
fp16_supported = builder.platform_has_fast_fp16
print(f"  GPU FP16 Tensor Core support: {'✅ YES' if fp16_supported else '❌ NO (will fall back to FP32)'}")
print()

# ── Build FP32 engine ──────────────────────────────────────────────────────
INPUT_SIZE = 128
np.random.seed(42)

print("  Building FP32 engine...")
t0 = time.time()
network_fp32, weights = build_fc_network(builder, INPUT_SIZE)
config_fp32 = builder.create_builder_config()
config_fp32.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * (1 << 20))
engine_fp32  = builder.build_engine(network_fp32, config_fp32)
build_fp32_s = time.time() - t0
print(f"  FP32 engine built in {build_fp32_s:.2f}s")

# ── Build FP16 engine ──────────────────────────────────────────────────────
print("  Building FP16 engine...")
t0 = time.time()
network_fp16, _ = build_fc_network(builder, INPUT_SIZE)
config_fp16 = builder.create_builder_config()
config_fp16.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * (1 << 20))
if fp16_supported:
    config_fp16.set_flag(trt.BuilderFlag.FP16)  # hint: use FP16 where beneficial
engine_fp16  = builder.build_engine(network_fp16, config_fp16)
build_fp16_s = time.time() - t0
print(f"  FP16 engine built in {build_fp16_s:.2f}s")
print()

# ── Benchmark both ────────────────────────────────────────────────────────
print("  Benchmarking (200 runs each, after 20 warmup)...")
ms_fp32, out_fp32 = benchmark_engine(engine_fp32, INPUT_SIZE)
ms_fp16, out_fp16 = benchmark_engine(engine_fp16, INPUT_SIZE)

speedup    = ms_fp32 / ms_fp16 if ms_fp16 > 0 else float('nan')
max_diff   = np.max(np.abs(out_fp32 - out_fp16))
mean_diff  = np.mean(np.abs(out_fp32 - out_fp16))

print()
print("  RESULTS:")
print(f"  {'Metric':<30} {'FP32':>12} {'FP16':>12} {'Ratio':>10}")
print(f"  {'─'*65}")
print(f"  {'Avg latency (ms/inference)':<30} {ms_fp32:12.4f} {ms_fp16:12.4f} {speedup:9.2f}×")
print(f"  {'Throughput (inf/sec)':<30} {1000/ms_fp32:12.1f} {1000/ms_fp16:12.1f}")
print(f"  {'Build time (s)':<30} {build_fp32_s:12.2f} {build_fp16_s:12.2f}")
print(f"  {'Output max abs diff':<30} {'—':>12} {max_diff:12.6f}")
print(f"  {'Output mean abs diff':<30} {'—':>12} {mean_diff:12.6f}")
print()
print(f"  FP32 output (first 5): {out_fp32[:5]}")
print(f"  FP16 output (first 5): {out_fp16[:5]}")
print()
print("  PRECISION GUIDE:")
print("  ┌────────┬──────────┬───────────┬─────────────────────────────┐")
print("  │ Mode   │ Mem/wt   │ Speedup   │ When to use                 │")
print("  ├────────┼──────────┼───────────┼─────────────────────────────┤")
print("  │ FP32   │ 4 bytes  │ 1×        │ Baseline, debug, validation │")
print("  │ FP16   │ 2 bytes  │ ~2–4×     │ Standard production choice  │")
print("  │ INT8   │ 1 byte   │ ~4–8×     │ Max throughput, calibration │")
print("  └────────┴──────────┴───────────┴─────────────────────────────┘")
print()
if not fp16_supported:
    print("  NOTE: FP16 speedup not shown — your GPU lacks Tensor Core FP16 support.")
    print("  Tensor Cores appear on Pascal+ (P100, V100, A100, RTX series, T4, ...).")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · ONNX Pipeline — PyTorch → ONNX → TensorRT Engine": {
        "description": (
            "The real-world production workflow. Export a small CNN from PyTorch to ONNX, "
            "verify with onnxruntime, parse into TensorRT, build an engine, run inference, "
            "and compare outputs between all three runtimes. "
            "Requires: tensorrt, pycuda, torch, onnx, onnxruntime, numpy."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  ONNX PIPELINE: PyTorch → ONNX → TensorRT")
print("=" * 65)
print()

# ── Imports ────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    print(f"  PyTorch {torch.__version__} ✅")
except ImportError:
    print("  pip install torch  (CPU version sufficient for export)")
    raise SystemExit(0)

try:
    import onnx
    import onnxruntime as ort
    print(f"  ONNX {onnx.__version__}, onnxruntime {ort.__version__} ✅")
except ImportError:
    print("  pip install onnx onnxruntime")
    raise SystemExit(0)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    print(f"  TensorRT {trt.__version__} ✅")
except ImportError as e:
    print(f"  TensorRT/pycuda not available: {e}")
    print("  Showing ONNX steps only (skip TRT sections if no GPU)")
    trt = None

print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 1: Define and train a small PyTorch CNN
# ─────────────────────────────────────────────────────────────────────────
print("  STEP 1 — Define PyTorch model")
print()

class TinyCNN(nn.Module):
    """
    Small CNN for demonstration.
    Input:  (N, 1, 28, 28)  → grayscale 28×28 images (like MNIST)
    Output: (N, 10)         → class logits
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # → (N, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (N, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (N, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (N, 32, 7,  7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = TinyCNN().eval()   # CRITICAL: call .eval() before export!
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model: TinyCNN")
print(f"  Input: (N, 1, 28, 28)  →  Output: (N, 10)")
print(f"  Parameters: {total_params:,}")
print()

# Sample input for tracing
dummy_input = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    pytorch_output = model(dummy_input).numpy()
print(f"  PyTorch output (first 5 logits): {pytorch_output[0, :5]}")
print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 2: Export to ONNX
# ─────────────────────────────────────────────────────────────────────────
print("  STEP 2 — Export to ONNX")
ONNX_PATH = "/tmp/tiny_cnn.onnx"

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    opset_version=17,                      # TensorRT 8.x supports up to opset 17
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes={"images": {0: "batch"}, # batch dim is dynamic
                  "logits": {0: "batch"}},
    do_constant_folding=True,              # fold constants at export time
)

# Verify the ONNX file is valid
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print(f"  ONNX model saved and verified: {ONNX_PATH}")
print(f"  ONNX opset: {onnx_model.opset_import[0].version}")

# Print ONNX graph summary
print(f"  ONNX nodes: {len(onnx_model.graph.node)}")
print(f"  Node types used: { {n.op_type for n in onnx_model.graph.node} }")
print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 3: Validate with ONNX Runtime (before TensorRT — important sanity check)
# ─────────────────────────────────────────────────────────────────────────
print("  STEP 3 — Validate with ONNX Runtime")
sess     = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
ort_out  = sess.run(None, {"images": dummy_input.numpy()})[0]
ort_diff = np.max(np.abs(pytorch_output - ort_out))
print(f"  OnnxRuntime output (first 5):     {ort_out[0, :5]}")
print(f"  Max diff vs PyTorch: {ort_diff:.2e}  {'✅' if ort_diff < 1e-4 else '⚠️'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 4: Parse into TensorRT and Build Engine
# ─────────────────────────────────────────────────────────────────────────
if trt is None:
    print("  STEP 4–6 — Skipped (TensorRT not available)")
    raise SystemExit(0)

print("  STEP 4 — Parse ONNX into TensorRT network")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
builder    = trt.Builder(TRT_LOGGER)
network    = builder.create_network(EXPLICIT)
parser     = trt.OnnxParser(network, TRT_LOGGER)

with open(ONNX_PATH, "rb") as f:
    success = parser.parse(f.read())

if not success:
    for i in range(parser.num_errors):
        print(f"  PARSE ERROR: {parser.get_error(i)}")
    raise RuntimeError("ONNX parsing failed")

print(f"  ✅ Parsed {network.num_layers} layers from ONNX")
print(f"  Network input:  {network.get_input(0).name}  {network.get_input(0).shape}")
print(f"  Network output: {network.get_output(0).name} {network.get_output(0).shape}")
print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 5: Configure and build engine with dynamic shapes
# ─────────────────────────────────────────────────────────────────────────
print("  STEP 5 — Build TensorRT engine with dynamic batch")
import time

config  = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * (1 << 20))

# Optimization profile: support batch 1–16, optimised for batch 4
profile = builder.create_optimization_profile()
profile.set_shape("images",
    min=(1,  1, 28, 28),
    opt=(4,  1, 28, 28),    # ← optimise for this batch size
    max=(16, 1, 28, 28))
config.add_optimization_profile(profile)

# Enable FP16 if GPU supports it
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("  FP16 enabled ✅")

t0     = time.time()
engine = builder.build_engine(network, config)
t_build = time.time() - t0
print(f"  ✅ Engine built in {t_build:.1f}s")
print()

# Serialise to disk so you never need to rebuild
ENGINE_PATH = "/tmp/tiny_cnn.plan"
with open(ENGINE_PATH, "wb") as f:
    f.write(engine.serialize())
print(f"  Engine serialised → {ENGINE_PATH}")
print(f"  (Load later with: runtime.deserialize_cuda_engine(open(path,'rb').read()))")
print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 6: Run inference and compare all three runtimes
# ─────────────────────────────────────────────────────────────────────────
print("  STEP 6 — Inference comparison: PyTorch vs ONNX-RT vs TensorRT")
context = engine.create_execution_context()
context.set_binding_shape(0, (1, 1, 28, 28))   # set actual input shape

inp_np     = dummy_input.numpy().astype(np.float32)
out_np     = np.empty((1, 10), dtype=np.float32)
d_input    = cuda.mem_alloc(inp_np.nbytes)
d_output   = cuda.mem_alloc(out_np.nbytes)
cuda.memcpy_htod(d_input, inp_np)
context.execute_v2(bindings=[int(d_input), int(d_output)])
cuda.memcpy_dtoh(out_np, d_output)

trt_diff_vs_pt  = np.max(np.abs(pytorch_output - out_np))
trt_diff_vs_ort = np.max(np.abs(ort_out - out_np))

print(f"  PyTorch output:    {pytorch_output[0, :5]}")
print(f"  ONNX Runtime:      {ort_out[0, :5]}")
print(f"  TensorRT output:   {out_np[0, :5]}")
print()
print(f"  Max diff (TRT vs PyTorch):   {trt_diff_vs_pt:.2e}")
print(f"  Max diff (TRT vs OnnxRuntime): {trt_diff_vs_ort:.2e}")
print()
print("  PIPELINE SUMMARY:")
print("  PyTorch .train() → .eval() → torch.onnx.export() → onnx.checker")
print("         → onnxruntime (sanity) → trt.OnnxParser → build_engine()")
print("         → engine.serialize() → .plan (saved) → deploy forever")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · INT8 Calibration — Quantise a Network to 8-bit": {
        "description": (
            "Implement INT8 post-training quantisation with a custom calibrator. "
            "Show how calibration data is used to compute per-tensor scale factors. "
            "Compare INT8 vs FP32 accuracy and speed on a small FC network. "
            "Requires: tensorrt, pycuda, numpy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  INT8 CALIBRATION — QUANTISE A NETWORK TO 8-BIT")
print("=" * 65)
print()

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"  ImportError: {e}")
    print("  Requires NVIDIA GPU + system-level CUDA/cuDNN/TensorRT install")
    raise SystemExit(0)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# ─────────────────────────────────────────────────────────────────────────
# Calibrator implementation
# ─────────────────────────────────────────────────────────────────────────

class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator.
    Feeds batches of representative data to TensorRT so it can
    measure the actual value distribution of each tensor and
    compute optimal FP32 → INT8 scale factors.

    IInt8EntropyCalibrator2 minimises KL divergence between the
    FP32 and INT8 distributions — more accurate than MinMax.
    """

    def __init__(self, calibration_data: np.ndarray, batch_size: int = 16,
                 cache_file: str = "/tmp/calibration.cache"):
        super().__init__()
        self.data        = calibration_data.astype(np.float32)
        self.batch_size  = batch_size
        self.cache_file  = cache_file
        self.current_idx = 0

        # Pre-allocate GPU buffer for one batch
        one_batch     = self.data[:batch_size]
        self.d_input  = cuda.mem_alloc(one_batch.nbytes)

        print(f"  Calibrator initialised:")
        print(f"    Calibration samples: {len(self.data)}")
        print(f"    Batch size:          {batch_size}")
        print(f"    Num batches:         {len(self.data) // batch_size}")
        print(f"    Cache file:          {cache_file}")
        print()

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        """
        Called by TensorRT once per calibration batch.
        Returns list of GPU pointers (one per network input).
        Returns None when calibration data is exhausted.
        """
        if self.current_idx + self.batch_size > len(self.data):
            return None   # ← signals end of calibration

        batch = self.data[self.current_idx: self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        """Load cached calibration results (skip recomputing if available)."""
        try:
            with open(self.cache_file, "rb") as f:
                data = f.read()
                print(f"  ✅ Loaded calibration cache ({len(data)} bytes)")
                return data
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        """Save calibration results so future builds skip recalibration."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"  ✅ Calibration cache written ({len(cache)} bytes)")


# ─────────────────────────────────────────────────────────────────────────
# Build a simple FC network
# ─────────────────────────────────────────────────────────────────────────
INPUT_SIZE = 64
np.random.seed(0)

def make_fc_network(builder):
    network = builder.create_network(EXPLICIT)
    w1 = np.random.randn(128, INPUT_SIZE).astype(np.float32) * 0.1
    b1 = np.zeros(128, dtype=np.float32)
    w2 = np.random.randn(10,  128       ).astype(np.float32) * 0.1
    b2 = np.zeros(10, dtype=np.float32)

    inp = network.add_input("x", trt.float32, (INPUT_SIZE,))
    fc1 = network.add_fully_connected(inp,               128,
                                       trt.Weights(w1), trt.Weights(b1))
    relu = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)
    fc2  = network.add_fully_connected(relu.get_output(0), 10,
                                        trt.Weights(w2), trt.Weights(b2))
    network.mark_output(fc2.get_output(0))
    return network, (w1, b1, w2, b2)

# ─────────────────────────────────────────────────────────────────────────
# Check INT8 support
# ─────────────────────────────────────────────────────────────────────────
builder = trt.Builder(TRT_LOGGER)
int8_ok  = builder.platform_has_fast_int8
fp16_ok  = builder.platform_has_fast_fp16
print(f"  INT8 support: {'✅ YES (Turing+)' if int8_ok else '❌ NO (need Turing GPU or newer)'}")
print(f"  FP16 support: {'✅ YES' if fp16_ok else '❌ NO'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# Build FP32 engine (reference)
# ─────────────────────────────────────────────────────────────────────────
print("  Building FP32 reference engine...")
net_fp32, weights = make_fc_network(builder)
cfg_fp32 = builder.create_builder_config()
cfg_fp32.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * (1 << 20))
eng_fp32 = builder.build_engine(net_fp32, cfg_fp32)
print("  ✅ FP32 engine ready")

# ─────────────────────────────────────────────────────────────────────────
# Build INT8 engine with calibration
# ─────────────────────────────────────────────────────────────────────────
print()
print("  Building INT8 engine with calibration...")
print()

# Generate representative calibration data (normally this is real training data)
n_calib_samples = 256
calib_data = np.random.randn(n_calib_samples, INPUT_SIZE).astype(np.float32)

calibrator = SimpleCalibrator(calib_data, batch_size=16)

net_int8, _ = make_fc_network(builder)
cfg_int8    = builder.create_builder_config()
cfg_int8.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * (1 << 20))

if int8_ok:
    cfg_int8.set_flag(trt.BuilderFlag.INT8)
    cfg_int8.int8_calibrator = calibrator     # ← attach calibrator

t0      = time.time()
eng_int8 = builder.build_engine(net_int8, cfg_int8)
t_int8   = time.time() - t0
print(f"  ✅ INT8 engine built in {t_int8:.2f}s")
print()

# ─────────────────────────────────────────────────────────────────────────
# Compare outputs
# ─────────────────────────────────────────────────────────────────────────
def run_once(engine, input_data):
    ctx   = engine.create_execution_context()
    d_in  = cuda.mem_alloc(input_data.nbytes)
    out   = np.empty(10, dtype=np.float32)
    d_out = cuda.mem_alloc(out.nbytes)
    cuda.memcpy_htod(d_in, input_data)
    ctx.execute_v2([int(d_in), int(d_out)])
    cuda.memcpy_dtoh(out, d_out)
    return out

def bench(engine, x, n=200):
    ctx   = engine.create_execution_context()
    d_in  = cuda.mem_alloc(x.nbytes)
    out   = np.empty(10, dtype=np.float32)
    d_out = cuda.mem_alloc(out.nbytes)
    cuda.memcpy_htod(d_in, x)
    for _ in range(20): ctx.execute_v2([int(d_in), int(d_out)])  # warmup
    s = cuda.Event(); e = cuda.Event()
    s.record()
    for _ in range(n): ctx.execute_v2([int(d_in), int(d_out)])
    e.record(); e.synchronize()
    return s.time_till(e) / n

test_input = np.random.randn(INPUT_SIZE).astype(np.float32)
out_fp32   = run_once(eng_fp32, test_input)
out_int8   = run_once(eng_int8, test_input)

max_diff   = np.max(np.abs(out_fp32 - out_int8))
ms_fp32    = bench(eng_fp32, test_input)
ms_int8    = bench(eng_int8, test_input)
speedup    = ms_fp32 / ms_int8

print("  RESULTS:")
print(f"  FP32 output:  {out_fp32[:5]}")
print(f"  INT8 output:  {out_int8[:5]}")
print(f"  Max abs diff: {max_diff:.4f}   {'✅ Good' if max_diff < 0.1 else '⚠️  Large'}")
print()
print(f"  {'Metric':<25} {'FP32':>10} {'INT8':>10} {'Ratio':>8}")
print(f"  {'─'*55}")
print(f"  {'Latency (ms)':<25} {ms_fp32:10.4f} {ms_int8:10.4f} {speedup:7.2f}×")
print(f"  {'Throughput (inf/s)':<25} {1000/ms_fp32:10.1f} {1000/ms_int8:10.1f}")
print()
print("  INT8 KEY FACTS:")
print("  • Calibration: feed ~100–1000 representative samples")
print("  • IInt8EntropyCalibrator2 minimises KL divergence (best accuracy)")
print("  • Cache calibration results — reuse across builds for same model")
print("  • Accuracy drop: typically < 0.5% on classification tasks")
print("  • Not all layers may actually run INT8 (TensorRT decides per-layer)")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Dynamic Shapes — One Engine, Many Batch Sizes": {
        "description": (
            "Build a single engine that handles variable batch sizes using "
            "OptimizationProfile. Show how to set min/opt/max shapes, how to update "
            "the binding shape at runtime, and benchmark latency vs throughput "
            "across batch sizes 1, 4, 8, 16, 32. "
            "Requires: tensorrt, pycuda, numpy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  DYNAMIC SHAPES — ONE ENGINE, MANY BATCH SIZES")
print("=" * 65)
print()

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"  ImportError: {e}")
    raise SystemExit(0)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# ─────────────────────────────────────────────────────────────────────────
# Build a simple network with DYNAMIC batch dimension
# ─────────────────────────────────────────────────────────────────────────
FEAT = 64

builder  = trt.Builder(TRT_LOGGER)
network  = builder.create_network(EXPLICIT)
config   = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * (1 << 20))

# Input shape: (-1, FEAT) where -1 means DYNAMIC batch dimension
inp = network.add_input("x", trt.float32, (-1, FEAT))   # -1 = dynamic

np.random.seed(7)
w1 = np.random.randn(128, FEAT ).astype(np.float32) * 0.05
b1 = np.zeros(128, dtype=np.float32)
w2 = np.random.randn(10,  128  ).astype(np.float32) * 0.05
b2 = np.zeros(10, dtype=np.float32)

fc1  = network.add_fully_connected(inp, 128, trt.Weights(w1), trt.Weights(b1))
relu = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)
fc2  = network.add_fully_connected(relu.get_output(0), 10,
                                    trt.Weights(w2), trt.Weights(b2))
network.mark_output(fc2.get_output(0))

# ─────────────────────────────────────────────────────────────────────────
# Optimization Profile — defines the range of valid shapes
# ─────────────────────────────────────────────────────────────────────────
print("  Defining optimization profile:")
print(f"    min batch = 1   (smallest we'll ever serve)")
print(f"    opt batch = 8   (most common → engine optimised HERE)")
print(f"    max batch = 32  (largest we'll ever serve)")
print()
print("  Why opt matters: TensorRT benchmarks GPU kernels at opt.")
print("  Performance degrades smoothly as you move away from opt.")
print()

profile = builder.create_optimization_profile()
profile.set_shape(
    "x",
    min=(1,  FEAT),   # min input shape
    opt=(8,  FEAT),   # opt input shape ← kernels selected for this
    max=(32, FEAT),   # max input shape
)
config.add_optimization_profile(profile)

print("  Building dynamic engine (this may take ~30-60s for kernel selection)...")
t0     = time.time()
engine = builder.build_engine(network, config)
t_build = time.time() - t0
print(f"  ✅ Engine built in {t_build:.1f}s")
print()

# ─────────────────────────────────────────────────────────────────────────
# Inference with different batch sizes
# ─────────────────────────────────────────────────────────────────────────
print("  Benchmarking latency and throughput across batch sizes:")
print(f"  (100 warm-up + 300 timed runs each)")
print()
print(f"  {'Batch':>6} | {'Latency (ms)':>14} | {'Throughput (samp/s)':>22} | {'vs batch=1':>12}")
print(f"  {'─'*65}")

context = engine.create_execution_context()
results = {}

for batch_size in [1, 2, 4, 8, 16, 32]:
    inp_np  = np.random.randn(batch_size, FEAT).astype(np.float32)
    out_np  = np.empty((batch_size, 10), dtype=np.float32)

    # CRITICAL: tell the context the actual shape for THIS inference call
    context.set_binding_shape(0, (batch_size, FEAT))

    d_in  = cuda.mem_alloc(inp_np.nbytes)
    d_out = cuda.mem_alloc(out_np.nbytes)
    cuda.memcpy_htod(d_in, inp_np)

    # Warm up
    for _ in range(100):
        context.execute_v2([int(d_in), int(d_out)])

    # Time with CUDA events
    s = cuda.Event(); e = cuda.Event()
    s.record()
    for _ in range(300):
        context.execute_v2([int(d_in), int(d_out)])
    e.record(); e.synchronize()

    ms_per_batch   = s.time_till(e) / 300
    throughput     = batch_size / (ms_per_batch / 1000)   # samples/sec
    results[batch_size] = (ms_per_batch, throughput)

# Print results table
ref_latency     = results[1][0]
ref_throughput  = results[1][1]
for bs, (ms, tput) in results.items():
    speedup = tput / ref_throughput
    optimised_marker = " ← opt" if bs == 8 else ""
    print(f"  {bs:6d} | {ms:14.4f} | {tput:22.1f} | {speedup:10.2f}×{optimised_marker}")

print()
print("  OBSERVATIONS:")
print("  • Latency increases with batch (more data per call)")
print("  • Throughput ALSO increases (GPU becomes more utilised)")
print("  • Peak throughput near 'opt' batch size")
print("  • Batch size 1: lowest latency (best for real-time)")
print("  • Batch size 32: highest throughput (best for bulk jobs)")
print()
print("  DYNAMIC SHAPE KEY RULES:")
print("  1. Use shape (-1, feat) in network.add_input() for dynamic batch")
print("  2. Define OptimizationProfile with min/opt/max")
print("  3. Call context.set_binding_shape() before EVERY execute_v2()")
print("  4. Allocate GPU buffers for MAX shape (or reallocate as needed)")
print("  5. 'opt' should match your most common production batch size")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · Profiling — Layer-by-Layer Timing and Bottleneck Analysis": {
        "description": (
            "Implement a custom IProfiler to measure per-layer latency inside TensorRT. "
            "Identify the slowest layers (bottlenecks) and understand memory vs compute "
            "bound analysis. Compare profiling with CUDA Events vs Python time.time(). "
            "Requires: tensorrt, pycuda, numpy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  PROFILING — LAYER-BY-LAYER TIMING & BOTTLENECK ANALYSIS")
print("=" * 65)
print()

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"  ImportError: {e}")
    raise SystemExit(0)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# ─────────────────────────────────────────────────────────────────────────
# Custom Profiler — captures per-layer timing
# ─────────────────────────────────────────────────────────────────────────

class LayerProfiler(trt.IProfiler):
    """
    TensorRT calls report_layer_time() after every layer executes.
    We collect all timings so we can analyse bottlenecks.

    NOTE: profiling adds overhead — disable in production.
    Enable for diagnosis only:  context.profiler = profiler_instance
    """

    def __init__(self):
        super().__init__()
        self.timings = {}   # layer_name → [list of ms timings]

    def report_layer_time(self, layer_name: str, ms: float):
        """Called by TensorRT after each layer completes."""
        if layer_name not in self.timings:
            self.timings[layer_name] = []
        self.timings[layer_name].append(ms)

    def summary(self, top_n: int = 10):
        if not self.timings:
            print("  No profiling data collected yet. Run inference first.")
            return

        # Compute average time per layer
        avg = {name: np.mean(times) for name, times in self.timings.items()}
        total = sum(avg.values())
        sorted_layers = sorted(avg.items(), key=lambda x: x[1], reverse=True)

        print(f"  {'Layer Name':<40} {'Avg ms':>9} {'% Total':>9} {'Bar'}")
        print(f"  {'─'*75}")

        cumulative = 0
        for name, ms in sorted_layers[:top_n]:
            pct = (ms / total) * 100
            cumulative += pct
            bar = "█" * int(pct / 2)
            display_name = name[:38] + ".." if len(name) > 40 else name
            print(f"  {display_name:<40} {ms:9.4f} {pct:8.1f}% {bar}")

        if len(sorted_layers) > top_n:
            rest_ms  = sum(ms for _, ms in sorted_layers[top_n:])
            rest_pct = (rest_ms / total) * 100
            print(f"  {'... (' + str(len(sorted_layers)-top_n) + ' more layers)':<40} "
                  f"{rest_ms:9.4f} {rest_pct:8.1f}%")

        print(f"  {'─'*75}")
        print(f"  {'TOTAL':<40} {total:9.4f} {'100.0%':>9}")
        print()
        return total

# ─────────────────────────────────────────────────────────────────────────
# Build a multi-layer network to profile
# ─────────────────────────────────────────────────────────────────────────
print("  Building 4-layer FC network for profiling...")

np.random.seed(1)
SIZES = [128, 256, 512, 256, 64, 10]   # input 128 → hidden → output 10

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(EXPLICIT)
config  = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * (1 << 20))

x = network.add_input("x", trt.float32, (SIZES[0],))
prev = x
for i in range(len(SIZES) - 2):
    nin, nout = SIZES[i], SIZES[i + 1]
    w  = np.random.randn(nout, nin).astype(np.float32) * np.sqrt(2.0 / nin)
    b  = np.zeros(nout, dtype=np.float32)
    fc = network.add_fully_connected(prev, nout, trt.Weights(w), trt.Weights(b))
    fc.name = f"fc_{i+1}_{nin}→{nout}"
    rl = network.add_activation(fc.get_output(0), trt.ActivationType.RELU)
    rl.name = f"relu_{i+1}"
    prev = rl.get_output(0)

# Final layer (no activation)
nin, nout = SIZES[-2], SIZES[-1]
w  = np.random.randn(nout, nin).astype(np.float32) * np.sqrt(2.0 / nin)
b  = np.zeros(nout, dtype=np.float32)
fc_out = network.add_fully_connected(prev, nout, trt.Weights(w), trt.Weights(b))
fc_out.name = f"fc_out_{nin}→{nout}"
network.mark_output(fc_out.get_output(0))

engine = builder.build_engine(network, config)
print("  ✅ Engine built")
print()

# ─────────────────────────────────────────────────────────────────────────
# TIMING METHOD 1: Python time.time()  (WRONG for GPU)
# ─────────────────────────────────────────────────────────────────────────
print("  TIMING METHOD 1 — Python time.time()  [INCORRECT for GPU]")
print("  Problem: GPU is ASYNC — Python continues before GPU finishes")
print()

context = engine.create_execution_context()
inp_np  = np.random.randn(SIZES[0]).astype(np.float32)
out_np  = np.empty(SIZES[-1], dtype=np.float32)
d_in    = cuda.mem_alloc(inp_np.nbytes)
d_out   = cuda.mem_alloc(out_np.nbytes)
cuda.memcpy_htod(d_in, inp_np)

# Wrong: Python returns immediately, GPU still running
t0 = time.time()
for _ in range(100):
    context.execute_v2([int(d_in), int(d_out)])  # async!
t_python = (time.time() - t0) / 100 * 1000
print(f"  Python time.time():  {t_python:.4f} ms/inf  ← may be UNDERESTIMATED")
print()

# ─────────────────────────────────────────────────────────────────────────
# TIMING METHOD 2: CUDA Events (CORRECT)
# ─────────────────────────────────────────────────────────────────────────
print("  TIMING METHOD 2 — CUDA Events  [CORRECT]")
print("  CUDA events are stamped by the GPU itself — perfectly accurate")
print()

# Warm up
for _ in range(50):
    context.execute_v2([int(d_in), int(d_out)])

start_ev = cuda.Event()
end_ev   = cuda.Event()
times_ms = []

for _ in range(200):
    start_ev.record()
    context.execute_v2([int(d_in), int(d_out)])
    end_ev.record()
    end_ev.synchronize()             # ← WAIT for GPU to finish
    times_ms.append(start_ev.time_till(end_ev))

mean_ms = np.mean(times_ms)
std_ms  = np.std(times_ms)
p50_ms  = np.percentile(times_ms, 50)
p95_ms  = np.percentile(times_ms, 95)
p99_ms  = np.percentile(times_ms, 99)

print(f"  Latency stats over 200 runs:")
print(f"    Mean:    {mean_ms:.4f} ms")
print(f"    Std dev: {std_ms:.4f} ms")
print(f"    P50:     {p50_ms:.4f} ms")
print(f"    P95:     {p95_ms:.4f} ms  ← worst 5% of requests see this")
print(f"    P99:     {p99_ms:.4f} ms  ← worst 1% of requests")
print(f"    Throughput: {1000/mean_ms:.0f} inf/sec")
print()

# ─────────────────────────────────────────────────────────────────────────
# TIMING METHOD 3: IProfiler — per-layer breakdown
# ─────────────────────────────────────────────────────────────────────────
print("  TIMING METHOD 3 — IProfiler (per-layer breakdown)")
print("  NOTE: profiler adds overhead — only use for diagnosis, not production")
print()

profiler           = LayerProfiler()
context.profiler   = profiler   # attach profiler to context

# Run with profiler active (fewer runs — profiling is slower)
for _ in range(30):
    context.execute_v2([int(d_in), int(d_out)])

print("  Layer-by-layer timing breakdown:")
total_layer_ms = profiler.summary(top_n=15)

print("  PROFILING GUIDE:")
print("  ┌───────────────────────────────────┬──────────────────────────────┐")
print("  │ Tool                              │ Use case                     │")
print("  ├───────────────────────────────────┼──────────────────────────────┤")
print("  │ CUDA Events                       │ Total inference latency      │")
print("  │ trt.IProfiler                     │ Per-layer bottleneck finder  │")
print("  │ NVIDIA Nsight Systems             │ Full GPU timeline (overlap)  │")
print("  │ NVIDIA Nsight Compute             │ Kernel-level hardware stats  │")
print("  └───────────────────────────────────┴──────────────────────────────┘")
print()
print("  P99 LATENCY matters most in production:")
print("  If mean=1ms but P99=20ms, 1% of your users wait 20× longer.")
print("  Always measure tail latency, not just mean.")
''',
    },

    "7 · Basic TensorRT Example(Equivalent to Hello World)": {
        "description": "Basic TensorRT Example",
        "language": "python",
        "code": '''
        
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA

# ── 1. LOGGER ───────────────────────────────────────────────────────────────
# TensorRT needs a logger to report what it's doing
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ── 2. BUILD THE ENGINE ─────────────────────────────────────────────────────
# Think of this like "compiling" your model for the GPU
def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config  = builder.create_builder_config()

    # Define a simple network: one input → one ReLU layer → one output
    # Input: a tensor of 4 floats
    input_tensor = network.add_input(
        name="input",
        dtype=trt.float32,
        shape=(4,)
    )

    # Add a ReLU activation layer (like the "Hello" part — simple but real)
    relu_layer = network.add_activation(
        input_tensor,
        trt.ActivationType.RELU
    )

    # Mark the output
    network.mark_output(relu_layer.get_output(0))

    # Build and return the engine
    return builder.build_engine(network, config)

# ── 3. RUN INFERENCE ────────────────────────────────────────────────────────
def run_inference(engine, input_data):
    context = engine.create_execution_context()

    # Allocate memory on GPU for input and output
    input_buf  = cuda.mem_alloc(input_data.nbytes)
    output_buf = cuda.mem_alloc(input_data.nbytes)  # same shape as input here

    # Copy input data from CPU → GPU
    cuda.memcpy_htod(input_buf, input_data)

    # Run the engine
    context.execute_v2(bindings=[int(input_buf), int(output_buf)])

    # Copy results from GPU → CPU
    output = np.empty_like(input_data)
    cuda.memcpy_dtoh(output, output_buf)

    return output

# ── 4. MAIN ─────────────────────────────────────────────────────────────────
engine = build_engine()
print("✅ Engine built successfully!")

# Input: mix of negative and positive numbers
# ReLU will zero out the negatives — that's the "Hello World" transformation
input_data = np.array([-3.0, -1.0, 2.0, 5.0], dtype=np.float32)

output = run_inference(engine, input_data)

print(f"Input:  {input_data}")
print(f"Output: {output}")
print("(ReLU zeroed out the negatives ✓)")
'''
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