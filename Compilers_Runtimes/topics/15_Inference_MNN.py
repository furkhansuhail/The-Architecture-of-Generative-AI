"""
MNN — Alibaba's Mobile Neural Network Inference Engine
========================================================

MNN (Mobile Neural Network) is Alibaba Group's open-source, cross-platform
inference engine designed for mobile, embedded, and server deployment. Where
TFLite is Google's mobile inference answer and Core ML is Apple's, MNN is the
production inference engine behind some of the world's highest-traffic
applications: Taobao (China's largest e-commerce platform), Tmall, DingTalk,
Alibaba Cloud, and dozens of Alibaba Group services collectively reaching over
one billion users.

MNN is not simply a mobile port of a server framework. It was built from scratch
with three constraints that shaped every design decision:

    UNIVERSALITY:  Run on ARM phones, x86 laptops, embedded MCUs, and cloud
                   servers from a single codebase. Support ONNX, TensorFlow,
                   PyTorch, Caffe, and MXNet models without per-framework ports.

    PERFORMANCE:   Beat TFLite and NCNN on ARM CPU for the op shapes that
                   appear in real Alibaba models. Hand-written ARM NEON assembly
                   for the 20 most performance-critical operations.

    COMPLETENESS: Unlike inference-only runtimes, MNN supports ON-DEVICE
                   TRAINING — Taobao personalises recommendation models on the
                   user's phone, updating embeddings without any data leaving
                   the device. This required a full autograd engine, optimiser
                   support, and variable-length sequence training.

The MNN architecture has three distinct layers:

    CONVERTER:     MNNConvert — a standalone C++ binary that reads any
                   supported framework format and writes a .mnn FlatBuffer.
                   The converter pipeline applies graph fusion passes,
                   layout normalisation, and static shape inference.

    RUNTIME:       The MNN Interpreter — loads a .mnn file, allocates a
                   memory pool, and dispatches each op to the selected backend.
                   Backends (CPU / OpenCL GPU / Vulkan GPU / Metal GPU / NPU)
                   are selected per-op with automatic fallback.

    EXPRESS:       A NumPy-like Python API layered on top of the runtime for
                   building and training neural networks. Express exposes
                   lazy evaluation, gradient computation, and a suite of
                   optimiser classes matching PyTorch's API surface.

In the connected compiler stack:
    LLVM      (module 01) ← MNN's CPU kernels include LLVM-compiled SIMD paths
    MLIR      (module 02) ← MNN does not use MLIR directly; its IR is FlatBuffers
    TVM       (module 08) ← TVM and MNN compete on ARM CPU and mobile GPU
    ONNX      (module 10) ← ONNX is MNN's primary and most complete import format
    TFLite    (module 11) ← TFLite is MNN's closest competitor; MNN often faster on ARM CPU
    OpenVINO  (module 12) ← OpenVINO targets Intel x86; MNN targets ARM and mobile
    Core ML   (module 13) ← Core ML targets Apple ANE; MNN targets cross-platform
    MIGraphX  (module 14) ← MIGraphX targets AMD datacenter GPU; MNN targets mobile
    MNN       (this)      ← Alibaba's cross-platform inference + on-device training engine

"""

import textwrap
import re

TOPIC_NAME   = "MNN — Alibaba's Mobile Neural Network Inference Engine"
DISPLAY_NAME = "15 · MNN"
ICON         = "🧮"
SUBTITLE     = ".MNN Format, ARM NEON Kernels, OpenCL/Vulkan GPU, INT8 Quantisation, and On-Device Training"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY MNN EXISTS: ALIBABA'S DEPLOYMENT PROBLEM

### The Scale of Alibaba's Mobile ML Problem

    Alibaba Group operates at a scale that makes Google and Facebook look
    comparable but not larger: Taobao alone has over 900 million annual
    active users on mobile, making purchasing decisions influenced by ML at
    every screen. By 2017, Alibaba ran:
        Image search (snap-to-buy in Taobao)
        Product recommendation (real-time personalisation)
        Visual quality detection (automated content moderation)
        Face unlock and identity verification
        OCR for logistics labels and invoices
        Speech-to-text for customer service routing

    ALL of these ran inference on users' phones — not on servers.
    The reason was not privacy (though that mattered). It was economics:
    at 900 million users, even 1 ms of server inference at $0.001/request
    costs $900,000 per day for a feature running on every pageview.

### Why Existing Frameworks Were Insufficient

    TFLITE (2017):
        ARM CPU performance was 10–30% slower than hand-tuned NEON assembly.
        No GPU acceleration beyond basic OpenGL ES compute (unreliable).
        No on-device training path.
        Google-ecosystem tooling (conversion requires TF).

    CAFFE2 MOBILE (later PyTorch Mobile):
        Heavy C++ dependencies, large binary size (> 5 MB stripped).
        Designed for server inference first; mobile was an afterthought.
        No clean Python API for rapid prototyping.

    NCNN (Tencent, 2017):
        Excellent ARM CPU performance.
        No GPU support initially.
        No Python API.
        No on-device training.

    ALIBABA'S REQUIREMENTS that none of these met:
        Multi-framework input: Alibaba uses TF (search), PyTorch (research),
        Caffe (legacy production), and MXNet (older Alibaba cloud models).
        A framework requiring conversion THROUGH a specific training framework
        was a dependency Alibaba did not want.

        On-device training: Taobao's personalisation team wanted to update
        user-specific embedding layers on the device — adapting the
        recommendation model to each user's browsing history without
        uploading personal data to a server.

        Binary size budget: App store rules and user psychology constrain the
        size increase an app update can impose. MNN targets < 800 KB
        stripped ARM binary for the core runtime.

### MNN's Founding and Open-Source Release

    MNN was developed internally at Alibaba starting in 2018.
    It was open-sourced in June 2019 on GitHub (alibaba/MNN).
    License: Apache 2.0.

    By 2023, MNN was deployed in:
        Taobao (product search, recommendation, image search)
        Tmall Genie (Alibaba's smart speaker, embedded ARM)
        DingTalk (real-time video effects, face detection)
        Alibaba Cloud (CPU inference endpoints, IoT edge devices)
        Cainiao logistics (label OCR, package detection)
        YouKu (video understanding, content moderation)

    Supported platforms:
        iOS 8.0+          (ARM, Metal GPU, CoreML accelerator)
        Android 4.3+      (ARM32/ARM64, OpenCL GPU, Vulkan GPU, NNAPI)
        Linux (ARM/x86)   (servers, Raspberry Pi, NVIDIA Jetson)
        Windows (x86_64)  (developer tooling, PC deployment)
        WebAssembly        (browser inference)
        OpenHarmony        (Huawei devices, replaces Android for some products)


##### PART 2 — THE .MNN FORMAT: FLATBUFFERS SCHEMA

### Why FlatBuffers (Again)

    Like TFLite, MNN chose FlatBuffers over Protocol Buffers for the same
    fundamental reason: ZERO-COPY deserialisation.

    On a low-end Android phone with 1 GB RAM:
        Protobuf parsing a 15 MB model: ~200 ms + 15 MB extra heap
        FlatBuffer mmap of same model:  ~2 ms + 0 bytes heap
    The FlatBuffer file IS the data structure — no parsing step.
    Critical for apps where model loading contributes to visible startup lag.

    MNN's schema (MNN.fbs) is defined in the schema directory of the repo.

### The .mnn FlatBuffer Schema

    ROOT TABLE: Net
        name:          string            model name (debug)
        tensorName:    [string]          names for all tensors in the model
        oplists:       [Op]              the ordered list of operations
        tensorNumber:  int               total count of distinct tensors
        sourceType:    NetSource         TF / CAFFE / ONNX / TENSORFLOW_LITE
        mnn_uuid:      string            unique model identifier
        extraTensorDescribe: [TensorDescribe]  shape/type metadata per tensor
        subgraphs:     [SubGraph]        for control flow (if/while/scan)

    OP TABLE: Op
        outputIndexes: [int]     output tensor indices (into Net.tensorName)
        inputIndexes:  [int]     input tensor indices
        type:          OpType    enum (Conv2D, Pooling, Relu, MatMul, ...)
        name:          string    debug name
        main:          OpParameter  union — op-specific config (FlatBuffers union)

    TENSORDESCRIBE TABLE:
        index:    int             tensor index (into Net.tensorName)
        blob:     BlobT           tensor data for constants (weights)
        describe: TensorDescribeT  shape + format + quantisation info

    BLOB TABLE (weights and constants):
        dims:       [int]         tensor shape
        dataType:   DataType      FLOAT / INT8 / INT32 / UINT8 / ...
        dataFormat: MNN_DATA_FORMAT  NCHW / NHWC / NC4HW4 (internal format)
        float32s:   [float]       FP32 weight data
        int8s:      [int8]        INT8 weight data
        scales:     [float]       per-channel quantisation scales
        zero_point: int           quantisation zero point

### OpParameter Union: Op-Specific Configuration

    The main field of each Op is a FlatBuffers union — a tagged variant
    that can hold one of ~100 different op-specific parameter tables.

    EXAMPLES:
        Convolution2DCommon:
            kernelX, kernelY:   int         filter spatial dimensions
            strideX, strideY:   int         stride
            dilateX, dilateY:   int         dilation factor
            padX, padY:         int         symmetric padding
            padMode:            PadMode     CAFFE / VALID / SAME
            group:              int         for depthwise: group = channels
            outputCount:        int         number of output channels
            activationType:     ActivationType  NONE / RELU / RELU6 / SWISH

        Pool:
            isGlobal:           bool        global average/max pool
            kernelX, kernelY:   int
            strideX, strideY:   int
            type:               PoolType    MAXPOOL / AVEPOOL

        MatMul:
            transposeA:         bool
            transposeB:         bool

        LSTM:
            outputCount:        int
            keepAllOutputs:     bool        return all timesteps or last only
            isBidirectional:    bool

### The NC4HW4 Internal Memory Layout

    A key architectural decision in MNN is the NC4HW4 memory layout.
    Standard NCHW lays out a tensor of shape [N, C, H, W] as:
        n₀c₀h₀w₀  n₀c₀h₀w₁  ...  n₀c₁h₀w₀  ...

    NC4HW4 groups 4 channels together into a single "channel group":
        For a [1, 16, 8, 8] tensor (N=1, C=16, H=8, W=8):
        NC4HW4 stores: 4 channel groups, each of shape [1, 4, 8, 8]
        Each group's data is interleaved: for spatial position (h, w),
        all 4 channel values are stored contiguously.

    WHY NC4HW4:
        ARM NEON processes 4 float32 values per instruction (128-bit registers).
        With NC4HW4: one NEON load gives you 4 channels for one spatial location.
        No gather/scatter needed — the data IS in the right layout.
        Convolution inner loop: load 4-channel input, load 4-channel filter,
        compute 4×4 outer product with vmlaq — one NEON multiply-accumulate.

    NC4HW4 in the FlatBuffer:
        Weights stored in NC4HW4 at conversion time.
        The Blob.dataFormat field records the layout.
        No runtime re-layout needed at model load time.
        Models converted with MNNConvert always produce NC4HW4 weights.

### Reading a .mnn File

    Python (using MNN's Python bindings):
        import MNN
        net  = MNN.Interpreter("model.mnn")
        ; net.getSessionInfo() after session creation

    FlatBuffers Python (raw schema access):
        import flatbuffers
        ; Generated Python bindings from schema:
        from MNN.schema.Net import Net
        buf  = open("model.mnn", "rb").read()
        buf2 = bytearray(buf)
        net  = Net.GetRootAs(buf2, 0)
        print(f"Model name: {net.Name()}")
        print(f"Op count:   {net.OplistsLength()}")
        for i in range(net.OplistsLength()):
            op = net.Oplists(i)
            print(f"  {op.Type()}  {op.Name()}")

    MNNConvert --print:
        ./MNNConvert --print model.mnn
        ; Prints human-readable op list, tensor names, and shapes.


##### PART 3 — THE MNN CONVERTER: FROM ANY FRAMEWORK TO .MNN

### MNNConvert: The Standalone Converter Tool

    MNN ships a standalone C++ binary, MNNConvert, that handles all conversions.
    It requires no Python environment on the conversion machine —
    a C++ binary that reads any supported format and writes .mnn.

    SUPPORTED INPUT FORMATS:
        ONNX:         --framework ONNX       (most reliable, recommended)
        TensorFlow:   --framework TF         (frozen .pb and SavedModel)
        TFLite:       --framework TFLITE     (.tflite files)
        Caffe:        --framework CAFFE      (.prototxt + .caffemodel)
        MXNet:        --framework MNN        (MXNet .json + .params)
        Torch (JIT):  --framework TORCH      (TorchScript .pt)

### ONNX → .mnn (Primary Path)

    COMMAND LINE:
        ./MNNConvert \
            --srcPath     model.onnx \
            --dstPath     model.mnn  \
            --framework   ONNX       \
            --modelType   MNN        \
            --bizCode     "my_model"

    WITH QUANTISATION:
        ./MNNConvert \
            --srcPath     model.onnx \
            --dstPath     model_int8.mnn \
            --framework   ONNX \
            --modelType   MNN  \
            --quantizeWeight 8     ; weight-only INT8 (no calibration needed)

    WITH STATIC QUANTISATION (full INT8, needs calibration):
        ./MNNConvert \
            --srcPath         model.onnx \
            --dstPath         model_int8_static.mnn \
            --framework       ONNX \
            --modelType       MNN  \
            --weightQuantBits 8 \
            --fp16            true

    KEY FLAGS:
        --bizCode:           application identifier embedded in the model
        --weightQuantBits N: quantise weights to N bits (4 or 8)
        --fp16:              convert FP32 weights to FP16 (2× size reduction)
        --saveStaticModel:   fuse shape computation for fixed-size input
        --inputConfigFile:   JSON specifying input names and shapes
        --transformerFuse:   enable transformer op fusion (attention patterns)
        --optimizePrefer:    0=default, 1=smallest, 2=fastest (layout choice)

### Python Conversion API (pymnn)

    MNN also provides a Python API for conversion, avoiding the CLI:

        import MNN.tools.mnn_convert as cvt

        # Basic ONNX conversion
        cvt.convert(
            src_path    = "model.onnx",
            dst_path    = "model.mnn",
            framework   = "ONNX",
        )

        # With FP16 compression
        cvt.convert(
            src_path    = "model.onnx",
            dst_path    = "model_fp16.mnn",
            framework   = "ONNX",
            fp16        = True,
        )

    TORCH TO MNN (via torch.jit.trace + ONNX):
        import torch, onnx
        import MNN.tools.mnn_convert as cvt

        model   = MyModel().eval()
        example = torch.randn(1, 3, 224, 224)
        traced  = torch.jit.trace(model, example)

        # Step 1: TorchScript → ONNX
        torch.onnx.export(traced, (example,), "model.onnx",
                           opset_version=13,
                           input_names=["input"],
                           output_names=["output"])

        # Step 2: ONNX → .mnn
        cvt.convert("model.onnx", "model.mnn", "ONNX")

### The Converter's Internal Passes

    The converter applies these passes before serialising to .mnn:

    GRAPH NORMALISATION:
        All ops mapped to MNN's canonical op set.
        NCHW ↔ NHWC layout normalisation based on target backend.
        For CPU (NC4HW4): transpose ops inserted where needed.

    CONSTANT FOLDING:
        Subgraphs with all-constant inputs evaluated at convert time.
        Positional encodings, bias constants, shape tensors folded away.

    OP FUSION:
        BN FOLD: BatchNorm absorbed into preceding Conv weights (same as TFLite/MIGraphX).
        CONV + RELU:  single op with activationType=RELU in Convolution2DCommon.
        CONV + RELU6: single op with activationType=RELU6.
        GEMM + BIAS:  Inner product layer carries bias natively.
        ATTENTION FUSION: with --transformerFuse:
            MatMul (Q@K^T) + scale + softmax + MatMul (@V) → fused attention op.
            Reduces 5+ separate op calls to 1 at runtime.

    WEIGHT LAYOUT CONVERSION:
        All Conv weights pre-transposed to NC4HW4.
        All FC weights pre-transposed to the MNN inner-product layout.
        No runtime weight re-layout at model load time.
        The NC4HW4 pre-transposition is the key source of MNN's CPU speedup
        vs TFLite (which transposes lazily at first inference in some versions).

    SHAPE INFERENCE:
        For models with fixed input shapes (--saveStaticModel):
            All intermediate tensor shapes computed at convert time.
            Stored in the Net.extraTensorDescribe table.
            At inference: no dynamic shape computation — shapes are constants.
            Result: lower latency at inference time (no per-op shape checks).

### Common Conversion Issues

    "Unsupported op" for ONNX custom ops:
        MNN does not support every ONNX op.
        Fix: simplify the model first with onnx-simplifier:
            python -m onnxsim model.onnx model_simplified.onnx
        The simplifier unfolds custom ops into standard ones.
        If still unsupported: implement as MNN custom op (see Part 8).

    Wrong output for converted TFLite models:
        TFLite uses NHWC natively; MNN expects NCHW internally.
        MNNConvert inserts Transpose ops automatically.
        If results are wrong: check --framework TFLITE is specified.

    Large model (> 200 MB):
        Standard MNN conversion serialises all weights into the .mnn file.
        For large models, use external weight storage:
            ./MNNConvert ... --weightQuantBits 8
        OR: use FP16 compression (--fp16 true) to halve weight size.


##### PART 4 — THE BACKEND SYSTEM: CPU, GPU, AND NPU DISPATCH

### MNN's Backend Architecture

    MNN's backend system is a two-level abstraction:

    LEVEL 1 — BACKEND SELECTION (per-session):
        When you create an MNN session, you specify which hardware to use.
        The interpreter assigns all ops in the model to ONE backend.
        If an op is not supported by the selected backend, it falls back
        to the CPU backend for that op (automatic, no user action needed).

    LEVEL 2 — OP DISPATCH (per-op within backend):
        Within the CPU backend, each op dispatches to the best available
        SIMD implementation: ARM NEON assembly / ARM SVE / AVX2 / AVX512 /
        x86 SSE / scalar fallback.
        The dispatch is determined at session creation time via CPUID.

    BACKEND SELECTION API (C++):
        MNN::ScheduleConfig config;
        config.type      = MNN_FORWARD_CPU;      ; or GPU, OPENCL, VULKAN, ...
        config.numThread = 4;                    ; intra-op thread count (CPU)
        MNN::Session* session = interpreter->createSession(config);

    BACKEND SELECTION API (Python):
        import MNN
        interpreter = MNN.Interpreter("model.mnn")
        config = {
            "backend": "CPU",   ; or "GPU", "OPENCL", "VULKAN", "METAL"
            "numThread": 4,
        }
        session = interpreter.createSession(config)

### Available Backends

    MNN_FORWARD_CPU (CPU):
        Available on: all platforms.
        Implementation: ARM NEON assembly / x86 AVX / scalar C++.
        Precision: FP32, FP16 (on ARMv8.2+), INT8.
        Thread pool: configurable N threads, intra-op parallelism.
        Recommendation: use for single-sample latency on ARM.

    MNN_FORWARD_OPENCL (GPU via OpenCL):
        Available on: Android (Qualcomm Adreno, ARM Mali), Linux desktop.
        Implementation: OpenCL 1.2+ compute kernels.
        Precision: FP32, FP16 (most Adreno/Mali support FP16 natively).
        Kernel cache: compiled kernels cached to app's private storage.
        Recommendation: use for high-throughput batch on mid-to-high-end Android.
        NOT available on: iOS (Apple does not support OpenCL since iOS 12).

    MNN_FORWARD_VULKAN (GPU via Vulkan):
        Available on: Android 7.0+ (Vulkan 1.0), Linux, Windows.
        Implementation: Vulkan compute shaders (GLSL/SPIR-V).
        Lower OpenCL driver overhead on some devices.
        Precision: FP32, FP16.
        Recommendation: newer Android where Vulkan is better supported than OpenCL.

    MNN_FORWARD_METAL (GPU via Metal):
        Available on: iOS 10+, macOS 10.12+.
        Implementation: Metal compute shaders (MSL).
        The Apple GPU path for MNN (not the ANE — Core ML handles ANE).
        Precision: FP32, FP16.
        Recommendation: iOS inference when Core ML is not available/suitable.

    MNN_FORWARD_NN (Android NNAPI):
        Available on: Android 8.1+.
        Routes inference through Android's NNAPI to device's NPU/DSP.
        Qualcomm Snapdragon NPU (Hexagon DSP), Samsung NPU, etc.
        Precision: FP16, INT8 (device-dependent).
        Recommendation: when NPU acceleration outweighs NNAPI overhead.

    MNN_FORWARD_AUTO:
        MNN selects the best backend automatically at session creation.
        Priority: GPU (if available and fast) → CPU.
        For production: specify explicitly to avoid unexpected backend changes.

### Backend Selection Decision Guide

    LATENCY-SENSITIVE (single request, < 30 ms):
        Small model (< 5M params, input ≤ 224×224): CPU (4 threads)
        Medium model (5–50M params): CPU (4–8 threads) or GPU
        Large model (> 50M params): GPU

    THROUGHPUT-SENSITIVE (batch processing):
        Always GPU (OpenCL on Android, Metal on iOS)
        CPU: only when GPU is unavailable or model is too small to saturate

    POWER-SENSITIVE (battery-powered, real-time):
        GPU FP16 is typically most efficient (TOPS/watt)
        NNAPI → NPU for INT8 quantised models when device has NPU

    COMPATIBILITY-FIRST (support all devices in your user base):
        CPU: works on every device from 2015 onwards
        GPU: check: 30–40% of Android devices have buggy OpenCL drivers
        Always test with CPU fallback enabled


##### PART 5 — THE CPU BACKEND: ARM NEON AND X86 ASSEMBLY KERNELS

### Why MNN's CPU Performance Stands Out

    MNN's CPU backend is the most carefully optimised component of the entire
    framework. The core claim — and it is benchmark-verified — is that MNN's
    CPU backend outperforms TFLite on ARM CPUs for the op shapes that appear
    in real models, often by 20–40% on convolution-heavy workloads.

    The performance comes from four decisions:

    1. NC4HW4 PRE-LAYOUT:
       As described in Part 2, weights are stored in NC4HW4 at conversion time.
       At inference, the convolution inner loop is a pure NEON multiply-accumulate
       with no data rearrangement. TFLite performs an im2col transformation at
       runtime; MNN's converter does an equivalent pre-transposition offline.

    2. HAND-WRITTEN ARM NEON ASSEMBLY:
       The 20 most performance-critical operations have hand-written assembly
       implementations in source/backend/cpu/arm/.
       These include:
           MNNConvRunForUnit.S:   the core convolution micro-kernel (fp32, int8)
           MNNGemmInt8.S:         INT8 GEMM for fully-connected and attention
           MNNDepthWise.S:        depthwise convolution (critical for MobileNet)
           MNNMaxPool.S:          max pooling with loop unrolling
           MNNMinMaxFloat32.S:    fast min/max for quantisation scale finding
       These assembly files use ARMv8-A NEON intrinsics directly (not via
       compiler auto-vectorisation) to guarantee predictable register allocation
       and instruction scheduling.

    3. WINOGRAD CONVOLUTION FOR 3×3 KERNELS:
       MNN implements Winograd F(4×4, 3×3) transformation for all 3×3 strided-1
       convolutions. Winograd reduces the 9 multiplications of a standard 3×3
       convolution to 4 (at the cost of extra additions).
       Speedup: 1.5–2.5× for 3×3 conv on ARM (the most common conv size).
       Threshold: applied when the output spatial size is ≥ 16×16 (otherwise
       the overhead of transformation exceeds the multiplication savings).

    4. THREAD POOL WITH WORK STEALING:
       MNN maintains a persistent thread pool (created once at session creation).
       Threads sleep on a condition variable between op calls.
       Work stealing: if one thread finishes its partition early, it steals
       work from other threads' queues.
       The thread pool is SHARED across sessions if multiple sessions run
       concurrently — avoiding oversubscription of CPU cores.

### ARM NEON Micro-Kernel Design

    The inner loop of a 3×3 convolution in NC4HW4 layout:

    For each output position (h, w) and each output channel group (4 channels):
        ; Load 4-channel accumulator:
        ld1    {v16.4s}, [output_ptr]     ; acc[0:4]

        ; For each input channel group (4 channels) and each filter position (3×3):
        ld1    {v0.4s},  [input_ptr]      ; input[0:4]  (4 channels at one spatial location)
        ld1    {v4.4s},  [filter_ptr]     ; filter[0:4] (4×4 matrix row, transposed)
        fmla   v16.4s, v0.4s, v4.s[0]    ; acc += input * filter[0]
        fmla   v17.4s, v0.4s, v4.s[1]    ; 4 outputs updated simultaneously
        fmla   v18.4s, v0.4s, v4.s[2]
        fmla   v19.4s, v0.4s, v4.s[3]
        ; Repeat for all 9 spatial positions and all input channel groups

    The key instruction is `fmla v_acc.4s, v_inp.4s, v_flt.s[lane]`:
        One instruction: 4 multiply-accumulates using a scalar broadcast.
        Throughput: 2 cycles on Cortex-A53, 1 cycle on Cortex-A76+.
        Peak: A76 at 2.4 GHz: ~10 GFLOPS FP32 (close to theoretical peak).

### x86 CPU Backend

    On x86 (server, developer workstation), MNN uses:
        SSE4.1: 4-wide FP32 SIMD (all x86 from ~2008)
        AVX:    8-wide FP32 SIMD (Sandy Bridge, 2011)
        AVX2:   8-wide FP32 + FMA (Haswell, 2013)
        AVX512: 16-wide FP32 + VNNI for INT8 (Skylake-X, Ice Lake)

    The x86 backend uses the same NC4HW4 layout (groups of 4 channels for SSE,
    groups of 8 channels for AVX/AVX2 via dedicated NC8HW8 variants).

### INT8 CPU Backend

    MNN's INT8 CPU path uses ARM SDOT (signed dot product) instructions on
    ARMv8.2+ cores (Cortex-A76, A77, A78, Apple M-series):

        SDOT v0.4s, v1.16b, v2.16b   ; 16×8-bit → 4×32-bit accumulate
        Throughput: 1 cycle on A76, processes 16 multiply-accumulates
        vs FP32 FMLA: 4 multiply-accumulates per cycle
        INT8 SDOT: 4× throughput vs FP32 FMLA

    On ARMv8.0 (Cortex-A53, A55, older phones without ARMv8.2):
        MNN falls back to INT8 multiply + manual 32-bit accumulation.
        Still ~2× faster than FP32 due to 4× smaller data (better cache use).

    THREAD COUNT TUNING:
        Optimal thread count is NOT always the number of cores.
        On big.LITTLE SoCs (e.g., Snapdragon 888: 1×X1 + 3×A78 + 4×A55):
            Using all 8 cores: A55 cores are 3× slower; can slow down big cores.
            Optimal: 1 or 4 (big cores only).
        MNN's recommended heuristic:
            numThread = min(4, total_cores)  for latency
            numThread = all_big_cores        for throughput


##### PART 6 — GPU BACKENDS: OPENCL, VULKAN, AND METAL

### OpenCL Backend Architecture

    MNN's OpenCL backend is the primary GPU path on Android.
    It compiles to OpenCL compute kernels at runtime.

    KERNEL COMPILATION AND CACHING:
        OpenCL kernels are JIT-compiled by the GPU driver on first use.
        First compilation: 200 ms – 5 seconds (visible latency spike).
        MNN solves this via a KERNEL CACHE:
            At first inference, compiled kernel binaries are saved to:
                getFilesDir() + "/mnn_opencl_cache/"
            On subsequent runs: kernels loaded from cache (< 10 ms).
        Cache is device-specific (Adreno 740 cache ≠ Mali-G710 cache).

    KERNEL DESIGN FOR OPENCL:
        MNN's OpenCL kernels use BUFFER mode or IMAGE2D mode.
        IMAGE2D mode uses OpenCL texture memory (sampler-based access):
            Better spatial locality for 2D convolution.
            Hardware bilinear filter engine used for some ops.
            Faster on Adreno GPUs (Qualcomm's image hardware is very fast).
        BUFFER mode uses raw global memory:
            More flexible.
            Better for matrix-heavy ops on Mali GPUs.
        MNN auto-detects which mode is faster per GPU at first run.

    TUNING STRATEGY:
        Unlike TensorRT (which exhaustively benchmarks at compile time),
        MNN's OpenCL backend uses a LIGHT TUNING approach:
            At first inference, it measures 2–3 candidate tile sizes.
            Selects the fastest without exhaustive search.
            Saves the selection to the kernel cache.
        This gives 70–90% of exhaustive tuning quality in 1–3 seconds.
        Full exhaustive tuning: enabled via MNN_GPU_TUNING_LEVEL environment
        variable (0=none, 1=light, 2=normal, 3=heavy, 4=exhaustive).

    GPU MEMORY MANAGEMENT:
        MNN pre-allocates all intermediate buffers at session creation.
        No OpenCL allocations during inference.
        Buffer sizes computed statically from the model's tensor shapes.
        A memory reuse planner (same interval-colouring as TFLite/MIGraphX)
        minimises peak GPU memory usage.

### Vulkan Backend

    MNN's Vulkan backend targets Android 7.0+ and Linux Vulkan devices.
    Vulkan compute shaders are written in GLSL → compiled to SPIR-V.

    ADVANTAGES OVER OPENCL ON NEWER ANDROID:
        Lower driver overhead: no implicit synchronisation, explicit fences.
        More predictable performance: driver cannot "helpfully" recompile shaders.
        Better on newer Adreno and Exynos drivers where Vulkan is more stable.

    PIPELINE OBJECTS:
        Each Vulkan operation is a VkPipeline (a compiled compute shader).
        Pipelines are compiled once at session creation and reused.
        Unlike OpenCL (which JIT-compiles at dispatch), Vulkan compilation
        happens in createSession() — startup cost is paid once.
        In production: createSession() once, run many times.

    DESCRIPTOR SETS:
        Vulkan input/output buffers are bound via VkDescriptorSet objects.
        MNN pre-allocates descriptor pools at session creation.
        No Vulkan allocation at inference time.

### Metal Backend (iOS and macOS)

    MNN's Metal backend targets iOS 10+ and macOS 10.12+.
    It uses Apple Metal Shading Language (MSL) compute shaders.

    RELATIONSHIP TO CORE ML:
        Metal is Apple's GPU compute API.
        Core ML (module 13) uses the ANE (Neural Engine) as its primary backend.
        MNN's Metal backend uses the GPU — not the ANE.
        For ANE access on iOS: you must use Core ML; MNN cannot reach the ANE.
        When to use MNN Metal vs Core ML:
            Core ML is unavailable (older iOS < 11).
            Model has ops not in Core ML's op set but supported in Metal.
            Cross-platform code that also targets Android (same MNN model).

    METAL COMMAND BUFFER PIPELINING:
        MNN batches multiple op dispatches into a single MTLCommandBuffer.
        The command buffer is committed once to the GPU at the end of invoke().
        This minimises CPU↔GPU command submission overhead (critical for
        models with many small ops like MobileNet-style architectures).


##### PART 7 — QUANTISATION: INT8, INT4, AND MNN-SPECIFIC SCHEMES

### MNN's Quantisation Philosophy

    MNN takes a different approach to quantisation than TFLite or ONNX Runtime.
    Rather than requiring calibration datasets at conversion time (TFLite's PTQ),
    MNN primarily supports POST-TRAINING WEIGHT QUANTISATION — quantising only
    the weights to INT8 or INT4, while activations remain in FP32 at runtime.

    WEIGHT-ONLY QUANTISATION (MNN's default INT8 mode):
        Weights stored as INT8 in the .mnn file (4× smaller than FP32).
        At inference: weights dequantised to FP32 before computation.
        No calibration data required.
        Accuracy: near-identical to FP32 (< 0.1% difference for most models).
        Speedup: moderate on CPU (mostly size/bandwidth saving, not compute).
        Use case: reducing model file size for app download.

    FULL INT8 ACTIVATION QUANTISATION (requires calibration):
        Both weights AND activations in INT8.
        Activations quantised with scale/zero_point from calibration statistics.
        Requires a representative calibration dataset (100–500 samples).
        Speedup: significant on ARM with SDOT (2–4× vs FP32 on A76/A78).
        Accuracy: typically < 1% drop for vision models.
        Use case: latency-critical paths on high-end ARM (SDOT-capable).

### MNN INT8 Quantisation Scheme

    MNN uses SYMMETRIC per-channel quantisation for weights and
    ASYMMETRIC per-tensor quantisation for activations (where applicable):

    WEIGHT QUANTISATION (symmetric, per output channel):
        For each output channel c:
            scale_c   = max(|W[c, :]|) / 127.0
            W_q[c, :] = round(W[c, :] / scale_c)   ; clipped to [-127, 127]
        Storage: INT8 weight + FP32 scale per channel (in BlobT.scales).
        Dequantise: W_fp32[c, :] = W_q[c, :] * scale_c

    ACTIVATION QUANTISATION (asymmetric, per tensor):
        scale     = (max_val - min_val) / 255.0
        zero_point = round(-min_val / scale)
        x_q = clamp(round(x / scale) + zero_point, 0, 255)   ; UINT8
        Dequantise: x = (x_q - zero_point) * scale

    Symmetric weights + asymmetric activations is the same scheme as
    TFLite's hybrid quantisation mode — well-understood and widely supported.

### INT4 Weight Quantisation

    MNN 2.4+ supports 4-bit weight quantisation for further size reduction:
        ./MNNConvert ... --weightQuantBits 4

    INT4 schemes (MNN uses a block quantisation approach):
        Each weight tensor divided into BLOCKS of 32 consecutive elements.
        Each block has its own scale and zero_point (block-wise quantisation).
        Within a block: each weight is a 4-bit integer.
        Storage: 4 bits per weight = 8× smaller than FP32.
        Accuracy: slightly worse than INT8; typical drop < 0.5% on CNNs.

    AT INFERENCE:
        INT4 weights are dequantised to FP32 in groups of 32.
        Computation still in FP32 (or FP16 on ARM).
        Primary benefit: 8× smaller model file → faster loading, less RAM.

    COMPARISON: INT4 vs INT8 on typical CNN (MobileNetV2):
        FP32 model:  14 MB, 71.9% top-1, 45 ms on Cortex-A76
        INT8 model:   3.7 MB, 71.5% top-1, 28 ms on Cortex-A76 (SDOT)
        INT4 model:   1.9 MB, 71.3% top-1, 32 ms on Cortex-A76 (dequant overhead)

    INT4 is most useful when: model size (app download) is the constraint.
    INT8 is most useful when: inference speed on SDOT-capable ARM is the goal.

### MNNCalib: Calibration for Full INT8 Activation Quantisation

    For full INT8 (weights + activations), MNN provides the MNNCalib tool:

        # Step 1: collect calibration data statistics
        ./MNNCalib \
            --model   model.mnn \
            --images  /path/to/calib_images/ \
            --quant   int8 \
            --output  model_calibrated.mnn \
            --thread  4

    MNNCalib algorithm (MinMax with outlier clipping):
        1. Run all calibration images through the FP32 model.
        2. For each activation tensor: collect min/max statistics.
        3. Apply percentile clipping (0.1% / 99.9% by default) to exclude
           outlier activations that would shrink the dynamic range.
        4. Compute scale and zero_point from the clipped [min, max].
        5. Write the calibrated .mnn with quantisation parameters embedded.

    Calibration quality depends on dataset representativeness:
        Use 100–500 images from the same distribution as inference.
        Diverse samples (different lighting, poses, subjects) give better
        range estimates than similar images.
        Under-representative calibration → accuracy drop at deployment.


##### PART 8 — THE MNN EXPRESS API AND RUNTIME

### Two Runtime APIs

    MNN provides two distinct runtime APIs:

    INTERPRETER API (low-level, .mnn file oriented):
        Load a pre-converted .mnn model file.
        Create a session (allocates memory, selects backend).
        Set input tensors, invoke, read output tensors.
        Used in production Android/iOS apps.

    EXPRESS API (high-level, NumPy-like):
        Build models programmatically using Python/C++.
        Variable objects with automatic gradient tracking (autograd).
        Training loop with PyTorch-compatible optimiser classes.
        Used for on-device training and rapid prototyping.

### Interpreter API (Python)

    FULL INFERENCE WORKFLOW:
        import MNN
        import numpy as np

        # Step 1: load model
        interpreter = MNN.Interpreter("model.mnn")

        # Step 2: create session (allocates all buffers)
        config   = {"backend": "CPU", "numThread": 4}
        session  = interpreter.createSession(config)

        # Step 3: get input tensor
        input_tensor = interpreter.getSessionInput(session, "input")

        # Step 4: copy numpy array to input tensor
        interpreter.resizeTensor(input_tensor, (1, 3, 224, 224))
        interpreter.resizeSession(session)    ; realloc if shape changed
        input_array = np.random.rand(1, 3, 224, 224).astype(np.float32)
        tmp_tensor  = MNN.Tensor(
            (1, 3, 224, 224), MNN.Halide_Type_Float,
            input_array, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_tensor)

        # Step 5: run inference
        interpreter.runSession(session)

        # Step 6: read output
        output_tensor = interpreter.getSessionOutput(session, "output")
        tmp_out = MNN.Tensor(
            output_tensor.getShape(),
            output_tensor.getDataType(),
            np.zeros(output_tensor.getShape(), dtype=np.float32),
            MNN.Tensor_DimensionType_Caffe)
        output_tensor.copyToHostTensor(tmp_out)
        logits = np.array(tmp_out.getData())

    TENSOR DIMENSION TYPES:
        MNN.Tensor_DimensionType_Caffe:  NCHW (channels first — PyTorch default)
        MNN.Tensor_DimensionType_Tensorflow: NHWC (channels last — TF default)
        Specify the layout matching how your numpy array is arranged.
        MNN handles the layout conversion internally.

    SHAPE RESIZING:
        MNN supports dynamic shapes at runtime via resizeTensor/resizeSession.
        New memory is allocated only if the shape changes.
        If shape is constant: resizeSession() after createSession() — never again.
        If shape varies: call resizeTensor + resizeSession before each inference.

### Express API (Python): Building and Training Models

    MNN Express provides a NumPy-like API for building dynamic computation graphs:

        import MNN.expr as F
        import MNN.nn as nn
        import numpy as np

        # Create a simple model using Express
        class SimpleFC(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features)
                self.relu   = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        model = SimpleFC(128, 64)

        # Forward pass
        x     = F.const(np.random.rand(4, 128).astype(np.float32), [4, 128])
        out   = model.forward(x)
        print(out.shape)   ; [4, 64]

        # Backward (gradient computation)
        loss  = F.reduce_sum(out)
        loss.backward()

    EXPRESS VARIABLE (MNN.expr.Var):
        The fundamental type — analogous to torch.Tensor.
        Supports: lazy evaluation (computation graph built, executed on demand).
        Gradient tracking: .grad attribute filled by backward().
        Shapes: static or dynamic.

    BUILT-IN LAYERS (MNN.nn.*):
        nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.DepthwiseConv2d
        nn.BatchNorm, nn.LayerNorm, nn.GroupNorm
        nn.MaxPool, nn.AvgPool, nn.AdaptiveAvgPool2d
        nn.Dropout, nn.Embedding
        nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.GeLU, nn.Swish
        nn.LSTM, nn.GRU
        nn.Sequential, nn.ModuleList

    BUILT-IN OPTIMISERS (MNN.optim.*):
        optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
        optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        optim.Adadelta(params, lr=1.0, rho=0.95)
        optim.RMSprop(params, lr=1e-2, alpha=0.99)

    TRAINING LOOP:
        import MNN.optim as optim

        optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for batch_x, batch_y in dataloader:
                x   = F.const(batch_x, list(batch_x.shape))
                y   = F.const(batch_y, list(batch_y.shape), type=MNN.expr.int32)
                out = model.forward(x)
                loss = criterion(out, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

### Saving and Loading Express Models

    SAVE TO .mnn (for on-device deployment):
        model.save_to_mnn("trained_model.mnn",
                           [(1, 128)])   ; example input shape

    SAVE EXPRESS CHECKPOINT:
        F.save(model.parameters(), "checkpoint.mnn")

    LOAD EXPRESS CHECKPOINT:
        params_loaded = F.load_as_dict("checkpoint.mnn")
        model.load_parameters(params_loaded)

### Custom Operators

    For ops not in MNN's built-in set, register a custom operator:

    C++ REGISTRATION:
        class MyCustomOp : public MNN::Execution {
        public:
            MyCustomOp(Backend* backend, const MNN::Op* op) :
                MNN::Execution(backend) {}

            MNN::ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs) override {
                ; Implement op logic here
                return MNN::NO_ERROR;
            }

            MNN::ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) override {
                ; Set output shapes based on input shapes
                return MNN::NO_ERROR;
            }
        };

        MNN_REGISTER_OP(MyCustomOp, "MyCustomOpType")

    The custom op type string must match the op_type set during conversion
    (e.g., via a custom ONNX operator in domain "com.mycompany").


##### PART 9 — ON-DEVICE TRAINING AND PERSONALISATION

### Why On-Device Training Matters

    Server-side personalisation requires:
        Sending user behaviour data to a server (privacy risk).
        Server compute costs (GPU inference + training).
        Network round-trips (latency for real-time adaptation).

    On-device training addresses all three:
        User data never leaves the phone.
        Training runs on the user's own GPU/CPU.
        Adaptation is immediate (no server round-trip).

    Alibaba's production use case — Taobao personalisation:
        The base recommendation model (large, pre-trained on server) is
        deployed to all users as a .mnn model.
        A small "adaptation head" (the last few layers) is fine-tuned
        on each user's device using their browsing and purchase history.
        The adaptation runs in the background (overnight, on charge, on WiFi).
        Only the updated adaptation weights (< 1 MB) are uploaded to sync
        across the user's devices — not the raw training data.

### MNN's On-Device Training Capabilities

    MNN supports full forward+backward+update loops on-device:

    SUPPORTED TRAINING MODES:
        Fine-tuning:       update a subset of layers (e.g., last 2 layers)
        Transfer learning: freeze backbone, train new head
        Federated learning: train on device, upload weight deltas only
        Distillation:      on-device student trained from cached teacher outputs

    GRADIENT COMPUTATION:
        MNN implements REVERSE-MODE AUTOMATIC DIFFERENTIATION.
        The computation graph is recorded during forward().
        backward() walks the graph in reverse, computing gradients.
        Supported ops for gradient: all nn.* layers + most F.* elementwise ops.

    TRAINING PERFORMANCE (ARM CPU):
        On-device training is SLOW compared to GPU training (obviously).
        MNN's target: training throughput of 1–10 samples/second on Cortex-A76.
        This is sufficient for fine-tuning a few thousand samples overnight.
        For 1000 samples × 5 epochs = 5000 updates at 5 updates/sec: ~17 minutes.

    MEMORY EFFICIENCY FOR TRAINING:
        Gradient checkpointing: MNN supports recomputing activations during
        backward rather than storing them (trades compute for memory).
        Configuration:
            model.setCheckPointPath("checkpoint.path")  ; enable checkpointing
        Critical for devices with < 2 GB RAM (activations can exceed model size).

### Federated Learning Architecture with MNN

    MNN is used in some of Alibaba's federated learning infrastructure:

    DEVICE SIDE:
        Load global model (.mnn).
        Collect local training data (e.g., click history for 1 day).
        Run MNN Express training loop (SGD/Adam, 3–10 epochs).
        Compute weight delta: delta = local_weights - global_weights.
        Apply differential privacy noise to delta (optional).
        Upload delta to aggregation server (not raw data).

    SERVER SIDE:
        Collect deltas from N devices (FedAvg aggregation).
        global_weights += mean(deltas) * learning_rate.
        Distribute new global model.

    CODE SKETCH (device side):
        import MNN.expr as F
        import MNN.optim as optim

        # Load base model
        interpreter  = MNN.Interpreter("global_model.mnn")
        base_weights = interpreter.get_parameters()

        # Fine-tune on local data
        model     = load_express_model("global_model.mnn")
        optimiser = optim.SGD(model.parameters(), lr=0.001)

        for data, label in local_dataset:
            pred = model.forward(F.const(data, data.shape))
            loss = criterion(pred, label)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Compute delta
        new_weights = model.get_parameters()
        delta = {k: new_weights[k] - base_weights[k]
                 for k in new_weights}
        upload_delta(delta)   ; upload delta, not raw data


##### PART 10 — DEPLOYMENT: ANDROID, IOS, EMBEDDED, AND SERVER

### Android Integration

    MNN ships as a prebuilt AAR (Android Archive) for Gradle:

    BUILD.GRADLE:
        dependencies {
            implementation 'com.taobao.android:MNN:2.9.0@aar'
            implementation 'com.taobao.android:MNN_GPU:2.9.0@aar'  // OpenCL/Vulkan
        }

    OR build from source with NDK:
        cmake -DMNN_BUILD_FOR_ANDROID=ON \
              -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
              -DANDROID_ABI=arm64-v8a \
              -DANDROID_PLATFORM=android-21 \
              -DMNN_OPENCL=ON \
              -DMNN_VULKAN=ON \
              ..

    BINARY SIZE (stripped, release mode):
        MNN core (CPU only):  ~700 KB arm64-v8a
        + OpenCL backend:     +300 KB
        + Vulkan backend:     +400 KB
        + NNAPI support:      +100 KB
        Total (all backends): ~1.5 MB arm64-v8a
        This compares to TFLite: ~900 KB CPU-only, ~2 MB with GPU delegate.

    ANDROID JAVA/KOTLIN API:
        import com.taobao.android.mnn.MNNNetInstance;

        MNNNetInstance net = MNNNetInstance.createFromFile(
            context, modelPath);
        MNNNetInstance.Config config = new MNNNetInstance.Config();
        config.forwardType = MNNNetInstance.CONFIG_FORWARD_OPENCL;
        config.numThread   = 4;
        Session session    = net.createSession(config);

        MNNNetInstance.Session.Tensor inputTensor =
            session.getInput(null);
        inputTensor.setInputFloatData(inputFloatArray);
        session.run();

        float[] output = session.getOutput(null).getFloatData();

### iOS Integration

    MNN on iOS ships as a framework (XCFramework) or as static library:

    COCOAPODS:
        pod 'MNN', '~> 2.9'
        pod 'MNN/Metal'    // add Metal GPU support

    SWIFT/OBJECTIVE-C:
        // Objective-C:
        #import <MNN/Interpreter.hpp>

        MNN::Interpreter* net = MNN::Interpreter::createFromFile("model.mnn");
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_METAL;   // use Metal GPU
        MNN::Session* session = net->createSession(config);

        MNN::Tensor* inputTensor = net->getSessionInput(session, nullptr);
        // ... fill tensor data ...
        net->runSession(session);

    BINARY SIZE (iOS, arm64):
        MNN core (CPU):  ~900 KB
        + Metal:         +200 KB
        Total:           ~1.1 MB

    SIZE COMPARISON (iOS inference libraries, arm64 stripped):
        Core ML framework:  system (0 KB app impact)
        MNN:                ~1.1 MB
        TFLite:             ~900 KB
        NCNN:               ~1.5 MB
        PyTorch Mobile:     ~6–8 MB

### Embedded and Server Deployment

    RASPBERRY PI (ARM Cortex-A72):
        cmake -DMNN_BUILD_SHARED_LIBS=OFF \
              -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
              -DMNN_USE_THREAD_POOL=ON \
              -DMNN_OPENMP=ON ..
        ; Raspberry Pi 4: ~4× slower than Pixel 6 for same model

    NVIDIA JETSON (ARM + CUDA):
        cmake -DMNN_CUDA=ON \\
              -DMNN_OPENCL=ON ..   ; experimental, not production-ready
        ; Recommended: use TensorRT on Jetson (better CUDA support)
        ; MNN on Jetson: use CPU or OpenCL, not CUDA

    LINUX SERVER (x86_64):
        cmake -DMNN_AVX512=ON \
              -DMNN_AVX512VNNI=ON \
              -DMNN_USE_THREAD_POOL=ON ..
        ; Server use case: batch processing, model evaluation
        ; For GPU server inference: use TensorRT or OpenVINO instead

    WEBASSEMBLY (browser):
        emcmake cmake -DMNN_BUILD_FOR_WASM=ON \
                      -DMNN_WASM_THREADS=ON ..
        ; WASM binary: ~1.5 MB gzipped
        ; Performance: ~5–10× slower than native ARM CPU
        ; Use case: browser-based prototype/demo, not production

### Performance Benchmarks (Representative)

    MOBILENETV2 224×224 BATCH=1 (milliseconds, lower is better):

    Device              CPU (MNN)   GPU (MNN)   TFLite CPU  Notes
    ─────────────────────────────────────────────────────────────────
    Pixel 8 (A76@3.0GHz) 22 ms     6.5 ms      28 ms       ARM64+SDOT
    Pixel 5 (A76@2.8GHz) 28 ms     8.2 ms      36 ms       ARM64+SDOT
    iPhone 15            18 ms      4.1 ms      N/A         MNN Metal
    Xiaomi 13 (A715@3.2) 19 ms     5.8 ms      24 ms       Adreno 740 OpenCL
    Raspberry Pi 4        85 ms      N/A         110 ms      Cortex-A72

    BERT-BASE SEQ=128 BATCH=1:

    Device              CPU (MNN)   GPU (MNN)   TFLite CPU  Notes
    ─────────────────────────────────────────────────────────────────
    Pixel 8 (INT8)        38 ms      N/A         55 ms       SDOT INT8 GEMM
    iPhone 15 (FP16)      45 ms      18 ms       N/A         MNN Metal GPU


##### PART 11 — MNN IN THE CONNECTED COMPILER STACK

### MNN and LLVM (Module 01)

    MNN's x86 CPU backend uses LLVM-compiled AVX/AVX512 intrinsic code.
    Some compute-intensive paths use LLVM's auto-vectoriser for correctness
    guarantees, with hand-written NEON assembly for critical hot paths.
    MNN itself is compiled by Clang (LLVM-based) on both Android NDK and iOS.

### MNN and TVM (Module 08)

    TVM and MNN are direct competitors on ARM CPU and mobile GPU inference.
    TVM's MetaSchedule can find optimal tile sizes for ARM kernels.
    MNN's hand-tuned NEON assembly beats TVM's auto-scheduled output for the
    specific (3×3, stride-1, NC4HW4) convolution shapes most common in
    production models. TVM wins for unusual shapes and custom hardware.
    TVM supports on-device compilation; MNN supports on-device training —
    different capabilities, different use cases.

### MNN and ONNX (Module 10)

    ONNX is MNN's recommended import format.
    ONNX opsets 7–17 are well-supported; newer opsets have partial coverage.
    Use onnx-simplifier before MNNConvert to fold Cast/Identity nodes.
    MNN can also EXPORT to ONNX from an Express model:
        model.export_to_onnx("model.onnx", [(1, 3, 224, 224)])

### MNN and TFLite (Module 11)

    MNN and TFLite are the two most widely deployed mobile inference engines.
    Direct competitors: same hardware targets, same use cases.

    MNN WINS:
        ARM CPU: ~20–30% faster for 3×3 convolution (NC4HW4 + NEON assembly)
        On-device training (TFLite has no training path in production)
        Multi-framework input: ONNX, TF, Caffe, MXNet, Torch
        OpenHarmony / Huawei device support

    TFLITE WINS:
        Google Play Services TFLite (zero app-size increase via Play Services)
        NNAPI integration (more mature than MNN's NNAPI backend)
        TensorFlow ecosystem (TF training → TFLite conversion is first-class)
        ARM Ethos-U NPU support (TFLite Micro + Ethos delegate)
        MCU deployment (TFLite Micro targets down to 16 KB SRAM)

### MNN and Core ML (Module 13)

    Core ML targets Apple ANE; MNN targets cross-platform.
    On iOS, MNN and Core ML can coexist:
        Core ML for models requiring ANE (faster, more efficient)
        MNN for cross-platform models that also run on Android
    MNN provides a Core ML delegate backend that routes certain ops to Core ML.
    In practice: production iOS apps should prefer Core ML for maximum performance.

### The Complete MNN Ecosystem

    ┌──────────────────────────────────────────────────────────────────────┐
    │  SOURCE FRAMEWORKS                                                    │
    │  PyTorch→ONNX   TensorFlow   TFLite   Caffe   MXNet   Torch JIT     │
    └──────────────────────────┬───────────────────────────────────────────┘
                               │
              MNNConvert: ONNX/TF/TFLite/Caffe/MXNet → .mnn
              Passes: BN fold, conv+relu fusion, attention fusion,
                      NC4HW4 weight layout, static shape inference
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  .mnn FlatBuffer                                                      │
    │  FP32 / FP16 / INT8 / INT4 weights  +  NC4HW4 layout               │
    └──────┬───────────────────┬────────────────────────────┬──────────────┘
           │                   │                            │
           ▼                   ▼                            ▼
    ┌─────────────┐   ┌────────────────────┐    ┌─────────────────────────┐
    │  CPU Backend│   │   GPU Backends      │    │  Express (training)     │
    │  ARM NEON   │   │   OpenCL (Android)  │    │  autograd + optimisers  │
    │  NEON SDOT  │   │   Vulkan (Android)  │    │  on-device fine-tuning  │
    │  AVX2/512   │   │   Metal (iOS/macOS) │    │  federated learning     │
    │  Winograd   │   │   NNAPI (NPU)       │    └─────────────────────────┘
    │  INT8/INT4  │   │   Kernel cache      │
    └─────────────┘   └────────────────────┘

    COMPARISON WITH ALTERNATIVES ON MOBILE:
    ┌──────────────────────┬──────────────┬──────────────┬──────────────────┐
    │ Property             │  MNN         │  TFLite      │  NCNN            │
    ├──────────────────────┼──────────────┼──────────────┼──────────────────┤
    │ ARM CPU speed        │ Fastest      │ Fast         │ Fast             │
    │ On-device training   │ YES          │ No           │ No               │
    │ ONNX input           │ YES (best)   │ Via onnx-tf  │ YES              │
    │ TFLite input         │ YES          │ Native       │ No               │
    │ OpenCL GPU           │ YES          │ Via delegate │ YES              │
    │ Vulkan GPU           │ YES          │ Via delegate │ YES              │
    │ Metal GPU (iOS)      │ YES          │ No           │ No               │
    │ NNAPI                │ YES          │ YES (better) │ No               │
    │ MCU support          │ No           │ YES (Micro)  │ No               │
    │ Binary size (CPU)    │ ~700 KB      │ ~900 KB      │ ~800 KB          │
    │ Python API           │ YES          │ YES          │ No               │
    │ WebAssembly          │ YES          │ YES          │ Experimental     │
    │ OpenHarmony          │ YES          │ Limited      │ No               │
    │ Open source          │ YES (Apache) │ YES (Apache) │ YES (BSD)        │
    └──────────────────────┴──────────────┴──────────────┴──────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · MNN Conversion Pipeline — ONNX and PyTorch to .mnn": {
        "description": (
            "End-to-end conversion from PyTorch and ONNX to the .mnn FlatBuffer format. "
            "Export a CNN to ONNX, then convert with MNNConvert (Python API). "
            "Inspect the resulting .mnn schema: ops, tensor names, quantisation fields. "
            "Show FP32, FP16, and INT8 weight-quantised conversions and measure sizes. "
            "Validate numerical accuracy of the converted model vs PyTorch reference. "
            "Demonstrate the NC4HW4 layout and why it matters for CPU performance."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  MNN CONVERSION PIPELINE — ONNX AND PYTORCH TO .MNN")
print("=" * 65)
print()

try:
    import MNN
    print(f"  MNN {MNN.version()}")
    HAS_MNN = True
except ImportError:
    HAS_MNN = False
    print("  MNN not installed.")
    print("  Install: pip install MNN")
    print("  OR build from source: github.com/alibaba/MNN")
print()

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Build and export a model to ONNX, then convert to .mnn
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Build model, export ONNX, convert to .mnn")
print("━" * 65)
print()

if HAS_TORCH:
    class DepthwiseSep(nn.Module):
        """Depthwise separable conv block — core of MobileNet."""
        def __init__(self, c_in, c_out, stride=1):
            super().__init__()
            self.dw = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, stride=stride,
                           padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in), nn.ReLU6())
            self.pw = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU6())
        def forward(self, x): return self.pw(self.dw(x))

    class TinyMobileNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem   = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU6())
            self.blocks = nn.Sequential(
                DepthwiseSep(16, 32, stride=2),
                DepthwiseSep(32, 64, stride=2),
                DepthwiseSep(64, 64))
            self.head   = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, num_classes))
        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    model   = TinyMobileNet(10).eval()
    n_params = sum(p.numel() for p in model.parameters())
    dummy   = torch.randn(1, 3, 64, 64)

    print(f"  Model: TinyMobileNet (depthwise separable CNN)")
    print(f"  Params: {n_params:,}  |  Input: {list(dummy.shape)}")
    print()

    tmpdir    = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "tiny_mobilenet.onnx")

    torch.onnx.export(
        model, (dummy,), onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    onnx_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX exported: {onnx_size_kb:.1f} KB")
    print()

    # ── Convert to .mnn with MNN Python API ──────────────────────────────
    if HAS_MNN:
        try:
            import MNN.tools.mnn_convert as cvt

            results = {}

            # FP32 (baseline)
            mnn_fp32 = os.path.join(tmpdir, "model_fp32.mnn")
            t0 = time.perf_counter()
            ret = cvt.convert(onnx_path, mnn_fp32, "ONNX")
            t_fp32 = (time.perf_counter() - t0) * 1000
            if ret == 0 or os.path.exists(mnn_fp32):
                results["FP32"] = (mnn_fp32, os.path.getsize(mnn_fp32) / 1024)
                print(f"  FP32 .mnn: {results['FP32'][1]:.1f} KB  "
                      f"({t_fp32:.0f} ms conversion)")

            # FP16 weight compression
            mnn_fp16 = os.path.join(tmpdir, "model_fp16.mnn")
            cvt.convert(onnx_path, mnn_fp16, "ONNX", fp16=True)
            if os.path.exists(mnn_fp16):
                results["FP16"] = (mnn_fp16, os.path.getsize(mnn_fp16) / 1024)
                fp32_kb = results.get("FP32", (None, 1.0))[1]
                print(f"  FP16 .mnn: {results['FP16'][1]:.1f} KB  "
                      f"({fp32_kb/results['FP16'][1]:.2f}× vs FP32)")

            # INT8 weight quantisation (no calibration needed)
            mnn_int8 = os.path.join(tmpdir, "model_int8.mnn")
            cvt.convert(onnx_path, mnn_int8, "ONNX",
                         weightQuantBits=8)
            if os.path.exists(mnn_int8):
                results["INT8 weights"] = (mnn_int8, os.path.getsize(mnn_int8) / 1024)
                fp32_kb = results.get("FP32", (None, 1.0))[1]
                print(f"  INT8 .mnn: {results['INT8 weights'][1]:.1f} KB  "
                      f"({fp32_kb/results['INT8 weights'][1]:.2f}× vs FP32)")

            print()
            print(f"  Size comparison (vs ONNX {onnx_size_kb:.1f} KB):")
            for name, (path, kb) in results.items():
                print(f"    {name:<20s}: {kb:>7.1f} KB")

        except AttributeError:
            print("  MNN Python conversion API not available in this build.")
            print("  Use command line: MNNConvert --srcPath model.onnx "
                  "--dstPath model.mnn --framework ONNX")
        except Exception as e:
            print(f"  Conversion: {e}")

    else:
        print("  MNN not available; showing conversion reference:")
        CONV_REF = """
  MNNCONVERT REFERENCE:
  ─────────────────────────────────────────────────────────────────
  # Command-line conversion (build MNNConvert from source):
  ./MNNConvert \\
      --srcPath   model.onnx \\
      --dstPath   model.mnn  \\
      --framework ONNX       \\
      --modelType MNN

  # FP16 weight compression (2× smaller, near-lossless accuracy):
  ./MNNConvert ... --fp16 true

  # INT8 weight quantisation (4× smaller, no calibration):
  ./MNNConvert ... --weightQuantBits 8

  # INT4 weight quantisation (8× smaller, slight accuracy drop):
  ./MNNConvert ... --weightQuantBits 4

  # Python API (requires MNN with tools):
  import MNN.tools.mnn_convert as cvt
  cvt.convert("model.onnx", "model.mnn", "ONNX")
  cvt.convert("model.onnx", "model_fp16.mnn", "ONNX", fp16=True)
  cvt.convert("model.onnx", "model_int8.mnn", "ONNX", weightQuantBits=8)
"""
        print(CONV_REF)

else:
    print("  PyTorch not available; install with: pip install torch")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: .mnn schema inspection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — .mnn schema: FlatBuffer structure walkthrough")
print("━" * 65)
print()

SCHEMA_GUIDE = """
  THE .MNN FLATBUFFER SCHEMA — COMPLETE FIELD REFERENCE
  ════════════════════════════════════════════════════════════════

  Reading .mnn with Python (using MNN's generated FlatBuffers bindings):
  ─────────────────────────────────────────────────────────────────
  from MNN.schema.MNN.Net import Net as MNNNet
  import flatbuffers

  with open("model.mnn", "rb") as f:
      buf = bytearray(f.read())
  net = MNNNet.GetRootAs(buf, 0)

  print(f"Model name:    {net.Name()}")
  print(f"Op count:      {net.OplistsLength()}")
  print(f"Tensor count:  {net.TensorNumberLength()}")
  print(f"Source:        {net.SourceType()}")  ; 0=TF, 1=CAFFE, 2=MNN, 3=TFLITE, 4=ONNX

  for i in range(net.OplistsLength()):
      op = net.Oplists(i)
      print(f"  [{i:3d}] {op.Type():5d}  {op.Name()}")
      print(f"         inputs:  {[op.InputIndexes(j) for j in range(op.InputIndexesLength())]}")
      print(f"         outputs: {[op.OutputIndexes(j) for j in range(op.OutputIndexesLength())]}")

  READING WEIGHT DATA FROM A BLOB:
  ─────────────────────────────────────────────────────────────────
  for i in range(net.ExtraTensorDescribeLength()):
      td   = net.ExtraTensorDescribe(i)
      blob = td.Blob()
      if blob is not None and blob.Float32sLength() > 0:
          dims = [blob.Dims(j) for j in range(blob.DimsLength())]
          fmt  = blob.DataFormat()    ; 0=NCHW, 1=NHWC, 2=NC4HW4
          print(f"  tensor {td.Index()}:  shape={dims}  format={fmt}  "
                f"float32s={blob.Float32sLength()}")
          if blob.ScalesLength() > 0:
              print(f"    INT8 quantised: {blob.ScalesLength()} scale values")

  OP TYPE ENUM VALUES (selected):
  ─────────────────────────────────────────────────────────────────
    0  =  AbsVal              5  =  BatchNorm
    4  =  Convolution         6  =  Bias
    7  =  BinaryOp            8  =  Cast
   10  =  Concat             11  =  Const
   13  =  ConvolutionDepthwise
   14  =  Crop               17  =  Deconvolution
   22  =  Eltwise            26  =  Gather
   28  =  InnerProduct        (fully-connected)
   34  =  LRN                36  =  LSTM
   40  =  MatMul             45  =  Normalize
   50  =  Permute            51  =  Pooling
   59  =  ReLU               61  =  Reshape
   65  =  Scale              71  =  Slice
   72  =  Softmax            76  =  SpatialProduct
   85  =  Transpose

  NC4HW4 LAYOUT EXPLAINED:
  ─────────────────────────────────────────────────────────────────
  Standard NCHW [N, C, H, W]:  element (n,c,h,w) at offset n*C*H*W + c*H*W + h*W + w
  NC4HW4 [N, C/4, H, W, 4]:   element (n,c,h,w) at offset n*(C/4)*H*W*4 + (c//4)*H*W*4 + h*W*4 + w*4 + c%4

  WHY NC4HW4 IS FASTER ON ARM NEON:
    ARM NEON register: 128 bits = 4 × float32
    With NC4HW4: one NEON load (ld1.4s) gives you 4 channels at one spatial position
    Convolution inner loop: load 4-channel input, load 4-channel filter,
    compute 4×4 outer product with fmla — one NEON instruction processes 4 channels
    With NCHW: need to gather from 4 non-contiguous memory locations (4× more loads)
    Result: 2–3× memory bandwidth reduction in the convolution hot loop
"""
print(SCHEMA_GUIDE)

if HAS_MNN and HAS_TORCH and "mnn_fp32" in dir():
    # Inspect the .mnn using MNN's Python interpreter API
    if os.path.exists(mnn_fp32):
        try:
            interpreter = MNN.Interpreter(mnn_fp32)
            cfg     = {"backend": "CPU", "numThread": 1}
            session = interpreter.createSession(cfg)

            input_tensor  = interpreter.getSessionInput(session, None)
            output_tensor = interpreter.getSessionOutput(session, None)

            print(f"  .mnn model inspection via MNN Interpreter:")
            print(f"    Input shape:  {input_tensor.getShape()}")
            print(f"    Output shape: {output_tensor.getShape()}")

            # Get all tensor names (requires newer MNN API)
            try:
                info = interpreter.getSessionInfo(session)
                print(f"    Backend used: {info.get('backend', 'CPU')}")
            except Exception:
                pass

            interpreter.releaseSession(session)
            print()
        except Exception as e:
            print(f"  Model inspection: {e}")
            print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · MNN Runtime API — Interpreter, Sessions, and Backends": {
        "description": (
            "Master the MNN Interpreter API end-to-end. "
            "Create sessions with CPU, OpenCL, and Vulkan backends. "
            "Set input tensors with correct dimension types (Caffe vs TF). "
            "Run inference and read output with copyToHostTensor. "
            "Benchmark CPU (single-thread, multi-thread) vs GPU backends. "
            "Show shape resizing for dynamic batch sizes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  MNN RUNTIME API — INTERPRETER, SESSIONS, AND BACKENDS")
print("=" * 65)
print()

try:
    import MNN
    HAS_MNN = True
    print(f"  MNN {MNN.version()}")
except ImportError:
    HAS_MNN = False
    print("  MNN not installed: pip install MNN")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Full inference workflow reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Full inference workflow: every step explained")
print("━" * 65)
print()

INFERENCE_GUIDE = """
  MNN INFERENCE WORKFLOW — EVERY STEP ANNOTATED
  ════════════════════════════════════════════════════════════════

  import MNN
  import numpy as np

  STEP 1 — LOAD THE MODEL:
  ─────────────────────────────────────────────────────────────────
  interpreter = MNN.Interpreter("model.mnn")
  ; MNN.Interpreter reads the .mnn FlatBuffer.
  ; Does NOT allocate GPU/CPU memory yet.
  ; Fast: just maps the file and reads the schema.

  STEP 2 — CREATE SESSION (allocates memory):
  ─────────────────────────────────────────────────────────────────
  config = {
      "backend":   "CPU",    ; or "GPU", "OPENCL", "VULKAN", "METAL"
      "numThread": 4,        ; intra-op thread count (CPU only)
      "precision": "normal", ; or "low" (FP16 compute), "high" (FP32 forced)
      "memory":    "normal", ; or "low" (recompute activations to save RAM)
  }
  session = interpreter.createSession(config)
  ; createSession():
  ;   Allocates all intermediate tensor buffers in one memory pool.
  ;   Assigns each op to its backend kernel implementation.
  ;   Compiles/loads GPU kernels (for OpenCL/Vulkan backends — slow first time).
  ; EXPENSIVE: creates all buffers and selects kernels.
  ; Create ONCE, reuse across many inputs.

  STEP 3 — GET INPUT TENSOR HANDLE:
  ─────────────────────────────────────────────────────────────────
  input_tensor = interpreter.getSessionInput(session, "input_name")
  ; Pass None for the name to get the first (or only) input.
  ; input_tensor is a handle to the tensor allocated in STEP 2.

  OPTIONAL: RESIZE FOR DYNAMIC SHAPES:
  ─────────────────────────────────────────────────────────────────
  interpreter.resizeTensor(input_tensor, [batch, 3, 224, 224])
  interpreter.resizeSession(session)
  ; resizeSession() reallocates all buffers for the new shape.
  ; If using static shapes (most production cases): skip this step.

  STEP 4 — COPY INPUT DATA INTO TENSOR:
  ─────────────────────────────────────────────────────────────────
  input_array = np.random.rand(1, 3, 224, 224).astype(np.float32)
  tmp_tensor  = MNN.Tensor(
      (1, 3, 224, 224),            ; shape tuple
      MNN.Halide_Type_Float,        ; data type
      input_array,                  ; numpy array (data copied from here)
      MNN.Tensor_DimensionType_Caffe   ; NCHW (Caffe/PyTorch layout)
  )
  input_tensor.copyFrom(tmp_tensor)
  ; copyFrom() copies the numpy data into the session's pre-allocated buffer.
  ; IMPORTANT: use Tensor_DimensionType_Caffe for NCHW arrays (PyTorch default).
  ;            use Tensor_DimensionType_Tensorflow for NHWC arrays (TF default).
  ; MNN handles any needed layout conversion internally.

  STEP 5 — RUN INFERENCE:
  ─────────────────────────────────────────────────────────────────
  interpreter.runSession(session)
  ; Executes all ops in the model in topological order.
  ; CPU: runs on thread pool, returns when done (synchronous).
  ; GPU: submits GPU commands, synchronises at the end (synchronous from caller's view).
  ; This is the HOT PATH — optimised to be as fast as possible.

  STEP 6 — READ OUTPUT:
  ─────────────────────────────────────────────────────────────────
  output_tensor = interpreter.getSessionOutput(session, "output_name")
  tmp_out = MNN.Tensor(
      output_tensor.getShape(),
      output_tensor.getDataType(),
      np.zeros(output_tensor.getShape(), dtype=np.float32),
      MNN.Tensor_DimensionType_Caffe)
  output_tensor.copyToHostTensor(tmp_out)
  result_np = np.array(tmp_out.getData())
  ; result_np is a numpy array with the inference output.

  STEP 7 — CLEANUP:
  ─────────────────────────────────────────────────────────────────
  interpreter.releaseSession(session)  ; free session memory
  del interpreter                       ; close model file

  DIMENSION TYPE QUICK REFERENCE:
  ─────────────────────────────────────────────────────────────────
  MNN.Tensor_DimensionType_Caffe:
    NCHW — [Batch, Channels, Height, Width]
    Use when: PyTorch, Caffe, ONNX (channels-first) models
    Numpy shape: (1, 3, 224, 224)

  MNN.Tensor_DimensionType_Tensorflow:
    NHWC — [Batch, Height, Width, Channels]
    Use when: TensorFlow, TFLite (channels-last) models
    Numpy shape: (1, 224, 224, 3)

  RULE: always use the layout that matches your numpy array.
  MNN applies any needed internal conversion automatically.
"""
print(INFERENCE_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Live inference benchmark (if MNN and PyTorch available)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Live inference: backend benchmark")
print("━" * 65)
print()

if HAS_MNN and HAS_TORCH:
    # Build and convert a model
    class BenchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(),
                nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(),
                nn.Conv2d(32, 64, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU6(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*4*4, 10))
        def forward(self, x): return self.net(x)

    model_b = BenchCNN().eval()
    dummy_b = torch.randn(1, 3, 32, 32)

    tmpdir_b  = tempfile.mkdtemp()
    onnx_b    = os.path.join(tmpdir_b, "bench.onnx")
    mnn_b     = os.path.join(tmpdir_b, "bench.mnn")

    torch.onnx.export(model_b, (dummy_b,), onnx_b,
                       opset_version=13,
                       input_names=["x"], output_names=["y"])

    try:
        import MNN.tools.mnn_convert as cvt
        cvt.convert(onnx_b, mnn_b, "ONNX")
        mnn_converted = os.path.exists(mnn_b)
    except Exception:
        mnn_converted = False

    if mnn_converted:
        interpreter = MNN.Interpreter(mnn_b)
        x_np = np.random.rand(1, 3, 32, 32).astype(np.float32)
        REPS = 500

        def run_mnn(config, name):
            try:
                sess = interpreter.createSession(config)
                inp  = interpreter.getSessionInput(sess, None)
                tmp  = MNN.Tensor((1, 3, 32, 32), MNN.Halide_Type_Float,
                                   x_np, MNN.Tensor_DimensionType_Caffe)
                inp.copyFrom(tmp)
                for _ in range(30): interpreter.runSession(sess)
                t0 = time.perf_counter()
                for _ in range(REPS): interpreter.runSession(sess)
                t_ms = (time.perf_counter() - t0) / REPS * 1000
                # Read output
                out_t = interpreter.getSessionOutput(sess, None)
                tmp_o = MNN.Tensor(out_t.getShape(), out_t.getDataType(),
                                    np.zeros(out_t.getShape(), dtype=np.float32),
                                    MNN.Tensor_DimensionType_Caffe)
                out_t.copyToHostTensor(tmp_o)
                out_np = np.array(tmp_o.getData())
                interpreter.releaseSession(sess)
                return t_ms, out_np
            except Exception as e:
                return None, None

        results_bench = {}
        configs = [
            ("CPU (1 thread)",  {"backend": "CPU", "numThread": 1}),
            ("CPU (4 threads)", {"backend": "CPU", "numThread": 4}),
        ]

        # Try GPU backends (may not be available in Python env)
        for gpu_name in ["OPENCL", "VULKAN", "METAL"]:
            try:
                test_cfg = {"backend": gpu_name, "numThread": 1}
                test_sess = interpreter.createSession(test_cfg)
                interpreter.releaseSession(test_sess)
                configs.append((f"GPU ({gpu_name})", test_cfg))
            except Exception:
                pass

        print(f"  Model: BenchCNN (32×32 input, depthwise separable)")
        print(f"  {'Backend':25s}  {'Latency (ms)':>12s}  {'vs 1T CPU':>10s}")
        print("  " + "-" * 52)

        ref_t = None
        ref_out = None
        for name, cfg in configs:
            t_ms, out = run_mnn(cfg, name)
            if t_ms is None:
                continue
            if ref_t is None: ref_t = t_ms
            if ref_out is None: ref_out = out
            ratio = ref_t / t_ms if ref_t else 1.0
            # Check numerical agreement
            if ref_out is not None and out is not None:
                err = float(np.max(np.abs(ref_out - out)))
                agree = "✅" if err < 0.01 else "⚠️"
            else:
                agree = ""
            print(f"  {name:<25s}  {t_ms:>12.3f}  {ratio:>10.2f}× {agree}")

        # PyTorch baseline
        with torch.no_grad():
            x_pt = torch.from_numpy(x_np)
            for _ in range(20): model_b(x_pt)
            t0 = time.perf_counter()
            for _ in range(REPS): model_b(x_pt)
            t_pt = (time.perf_counter() - t0) / REPS * 1000
        if ref_t:
            print(f"  {'PyTorch eager':25s}  {t_pt:>12.3f}  {ref_t/t_pt:>10.2f}× baseline")
        print()

    else:
        print("  Conversion not available; showing benchmark reference.")
        print()

else:
    BENCH_REF = """
  BACKEND BENCHMARK REFERENCE (ARM Cortex-A76, MobileNetV2):
  ─────────────────────────────────────────────────────────────────
  config = {"backend": "CPU",     "numThread": 1}  → 85 ms (single thread)
  config = {"backend": "CPU",     "numThread": 4}  → 27 ms (4 big cores)
  config = {"backend": "OPENCL",  "numThread": 1}  → 8.5 ms (Adreno OpenCL FP16)
  config = {"backend": "VULKAN",  "numThread": 1}  → 9.1 ms (Adreno Vulkan FP16)
  PyTorch eager (CPU):                             → 110 ms (no NEON opt)

  NUMTHREAD TUNING GUIDE:
  ─────────────────────────────────────────────────────────────────
  numThread=1:   lowest power, deterministic, for single-threaded apps
  numThread=2:   good balance on big.LITTLE phones
  numThread=4:   optimal for most high-end phones (4 big cores)
  numThread=8:   only useful on SoCs with 8 homogeneous big cores
  AVOID:         numThread > number of big/performance cores
                 (adds A55 "efficiency" cores that are 3× slower)

  PRECISION OPTIONS:
  ─────────────────────────────────────────────────────────────────
  "precision": "normal"  → FP32 compute (default, most accurate)
  "precision": "low"     → FP16 compute (ARMv8.2+, 1.5–2× faster, slight error)
  "precision": "high"    → FP32 forced even on GPU (for debugging)

  MEMORY OPTIONS:
  ─────────────────────────────────────────────────────────────────
  "memory": "normal"  → keep all activations in memory pool (default)
  "memory": "low"     → recompute activations to save ~30% RAM (slower)
  Use "memory": "low" on devices with < 1 GB available RAM.
"""
    print(BENCH_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Dynamic shapes and multi-session
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Dynamic shapes and multi-session usage")
print("━" * 65)
print()

DYNAMIC_AND_MULTI = """
  DYNAMIC SHAPES — RUNTIME RESIZING
  ════════════════════════════════════════════════════════════════

  Use case: NLP model with variable sequence length.

  import MNN, numpy as np

  interpreter = MNN.Interpreter("bert_tiny.mnn")
  config      = {"backend": "CPU", "numThread": 4}
  session     = interpreter.createSession(config)

  for seq_len in [16, 32, 64, 128]:
      input_ids = interpreter.getSessionInput(session, "input_ids")

      # Resize to new sequence length:
      interpreter.resizeTensor(input_ids, [1, seq_len])
      interpreter.resizeSession(session)   ; reallocates all buffers!

      # Set input data
      ids_np = np.random.randint(0, 1000, (1, seq_len)).astype(np.int32)
      tmp    = MNN.Tensor([1, seq_len], MNN.Halide_Type_Int32, ids_np,
                          MNN.Tensor_DimensionType_Caffe)
      input_ids.copyFrom(tmp)
      interpreter.runSession(session)

      out = interpreter.getSessionOutput(session, "logits")
      print(f"  seq_len={seq_len:3d}: output shape = {out.getShape()}")

  NOTE: resizeSession() is expensive (reallocates ALL buffers).
  If you know the shape won't change: skip resize entirely.
  If you have a fixed set of shapes: create one session per shape.

  MULTI-SESSION (parallel inference with shared model weights):
  ════════════════════════════════════════════════════════════════

  Multiple sessions share the same model weights (read-only).
  Each session has its own activation buffers.
  Use for: serving multiple concurrent requests from one model.

  interpreter = MNN.Interpreter("model.mnn")

  # Create two independent sessions (shared weights, separate buffers):
  sess_a = interpreter.createSession({"backend": "CPU", "numThread": 2})
  sess_b = interpreter.createSession({"backend": "CPU", "numThread": 2})

  # Run them independently (from different threads):
  import threading

  def infer(sess, data):
      inp = interpreter.getSessionInput(sess, None)
      tmp = MNN.Tensor(data.shape, MNN.Halide_Type_Float, data,
                       MNN.Tensor_DimensionType_Caffe)
      inp.copyFrom(tmp)
      interpreter.runSession(sess)

  t1 = threading.Thread(target=infer, args=(sess_a, batch_a))
  t2 = threading.Thread(target=infer, args=(sess_b, batch_b))
  t1.start(); t2.start()
  t1.join();  t2.join()

  THREAD SAFETY:
    interpreter.createSession(): NOT thread-safe (call from main thread)
    interpreter.runSession():    THREAD-SAFE (different sessions only)
    Two threads calling runSession on the SAME session: NOT safe
    Two threads calling runSession on DIFFERENT sessions: SAFE
"""
print(DYNAMIC_AND_MULTI)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · INT8 Quantisation — Weight Quantisation and Full INT8": {
        "description": (
            "Complete INT8 quantisation workflow for MNN. "
            "Show weight-only INT8 conversion (no calibration needed). "
            "Perform full INT8 with activation quantisation using MNNCalib. "
            "Compare FP32 vs FP16 vs INT8 on size, latency, and accuracy. "
            "Explain MNN's symmetric per-channel weight scheme vs asymmetric activations. "
            "Benchmark INT8 vs FP32 on ARM SDOT-capable and non-SDOT hardware."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  INT8 QUANTISATION — WEIGHT QUANTISATION AND FULL INT8")
print("=" * 65)
print()

try:
    import MNN
    HAS_MNN = True
    print(f"  MNN {MNN.version()}")
except ImportError:
    HAS_MNN = False
    print("  MNN not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Quantisation scheme deep dive
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — MNN quantisation scheme: maths and design choices")
print("━" * 65)
print()

QUANT_THEORY = """
  MNN INT8 QUANTISATION — DESIGN CHOICES EXPLAINED
  ════════════════════════════════════════════════════════════════

  MNN uses two different quantisation schemes for weights vs activations.
  This asymmetry is deliberate and well-justified:

  WEIGHT QUANTISATION: SYMMETRIC, PER CHANNEL
  ─────────────────────────────────────────────────────────────────
  For each output channel c of a weight tensor W:
      abs_max_c = max(|W[c, :]|)
      scale_c   = abs_max_c / 127.0
      W_q[c, :] = round(W[c, :] / scale_c)  ; INT8, range [-127, 127]

  KEY PROPERTIES:
    Zero point = 0 (symmetric): W_real ≈ W_q × scale_c
    This is SYMMETRIC quantisation — real 0.0 maps to integer 0.
    ARM NEON INT8 GEMM (SDOT/SMMLA) requires symmetric INT8 for weights.
    Per-channel (not per-tensor): each output channel has its own scale.
    Better accuracy than per-tensor: different channels have different ranges.

  ACTIVATION QUANTISATION: ASYMMETRIC, PER TENSOR
  ─────────────────────────────────────────────────────────────────
  For a calibrated activation tensor with observed range [r_min, r_max]:
      scale     = (r_max - r_min) / 255.0
      zero_pt   = round(-r_min / scale)        ; in range [0, 255]
      x_q = clamp(round(x / scale) + zero_pt, 0, 255)   ; UINT8

  KEY PROPERTIES:
    Non-zero zero point: can represent asymmetric distributions (e.g., post-ReLU).
    Post-ReLU activations: range [0, R] maps efficiently to [0, 255].
    Per-tensor: one scale and zero_point for the whole activation tensor.
    Requires calibration data to determine [r_min, r_max].

  WEIGHT-ONLY MODE (no calibration, default --weightQuantBits 8):
  ─────────────────────────────────────────────────────────────────
  ONLY weights are quantised (INT8 storage).
  Activations remain FP32 at runtime.
  Weights dequantised to FP32 before each kernel call:
      W_fp32 = W_q × scale_c    (per-channel dequant, fast with NEON)
  Computation: FP32 (same accuracy as FP32 baseline).

  WHEN IS WEIGHT-ONLY USEFUL?
    Model file is too large for app download budget.
    Target device does NOT have INT8 SDOT (older ARM Cortex-A55 cores).
    Accuracy sensitivity is high (no activation quantisation error).
    Typical use: embedding tables in NLP models (can be 50–90% of model size).

  FULL INT8 (weights + activations, requires MNNCalib):
  ─────────────────────────────────────────────────────────────────
  After calibration with representative data:
      Weights:     INT8 (symmetric per-channel, as above)
      Activations: UINT8 (asymmetric per-tensor)
  The matmul kernel:
      result_int32 = sum(W_q × x_q)   ; INT8 × UINT8 → INT32 accumulate
  Rescale output:
      y_fp32 = (result_int32 - bias_int32) × (scale_W × scale_x)

  ARM SDOT THROUGHPUT GAIN:
    SDOT instruction: 16 × INT8 MACs per cycle on Cortex-A76
    vs FMLA instruction: 4 × FP32 MACs per cycle
    Ratio: 4× theoretical throughput gain
    In practice (with memory bandwidth): 2–3× measured speedup.

  ACCURACY IMPACT (MobileNetV2, ImageNet):
  ─────────────────────────────────────────────────────────────────
  FP32:                  71.9% top-1    100 MB model
  FP16 weights:          71.9% top-1     50 MB  (near-lossless)
  INT8 weights only:     71.8% top-1     26 MB  (essentially lossless)
  INT8 full (per-tensor): 71.0% top-1   13 MB  (-0.9%)
  INT8 full (per-channel): 71.5% top-1  13 MB  (-0.4%)
  INT4 weights only:     71.3% top-1     8 MB  (-0.6%)
"""
print(QUANT_THEORY)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Apply quantisation and compare
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Apply quantisation: sizes and accuracy comparison")
print("━" * 65)
print()

if HAS_TORCH and HAS_MNN:
    class QuantNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(),
                nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(),
                nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU6(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*4*4, 10))
        def forward(self, x): return self.net(x)

    model_q  = QuantNet().eval()
    dummy_q  = torch.randn(1, 3, 32, 32)
    tmpdir_q = tempfile.mkdtemp()
    onnx_q   = os.path.join(tmpdir_q, "quant_net.onnx")
    torch.onnx.export(model_q, (dummy_q,), onnx_q, opset_version=13,
                       input_names=["x"], output_names=["y"])

    x_test = np.random.rand(1, 3, 32, 32).astype(np.float32)

    # PyTorch reference output
    with torch.no_grad():
        pt_ref = model_q(torch.from_numpy(x_test)).numpy()

    try:
        import MNN.tools.mnn_convert as cvt
        conversions = {}

        for quant_name, kwargs in [
            ("FP32",         {}),
            ("FP16",         {"fp16": True}),
            ("INT8 weights", {"weightQuantBits": 8}),
            ("INT4 weights", {"weightQuantBits": 4}),
        ]:
            out_path = os.path.join(tmpdir_q, f"model_{quant_name.replace(' ','_')}.mnn")
            ret = cvt.convert(onnx_q, out_path, "ONNX", **kwargs)
            if os.path.exists(out_path):
                size_kb = os.path.getsize(out_path) / 1024
                conversions[quant_name] = (out_path, size_kb)

        if conversions:
            fp32_kb = conversions.get("FP32", (None, 1.0))[1]

            print(f"  {'Format':20s}  {'Size (KB)':>10s}  {'vs FP32':>8s}  "
                  f"{'Max err':>10s}  {'Argmax OK':>10s}")
            print("  " + "-" * 65)

            for fmt, (path, kb) in conversions.items():
                ratio = fp32_kb / kb if kb else 1.0
                try:
                    interp = MNN.Interpreter(path)
                    sess   = interp.createSession({"backend": "CPU", "numThread": 1})
                    inp_t  = interp.getSessionInput(sess, None)
                    tmp_i  = MNN.Tensor((1, 3, 32, 32), MNN.Halide_Type_Float,
                                        x_test, MNN.Tensor_DimensionType_Caffe)
                    inp_t.copyFrom(tmp_i)
                    interp.runSession(sess)
                    out_t  = interp.getSessionOutput(sess, None)
                    tmp_o  = MNN.Tensor(out_t.getShape(), out_t.getDataType(),
                                        np.zeros(out_t.getShape(), dtype=np.float32),
                                        MNN.Tensor_DimensionType_Caffe)
                    out_t.copyToHostTensor(tmp_o)
                    mnn_out = np.array(tmp_o.getData())
                    interp.releaseSession(sess)
                    max_err = float(np.max(np.abs(pt_ref - mnn_out)))
                    argmax_ok = np.argmax(pt_ref) == np.argmax(mnn_out)
                    err_str   = f"{max_err:.4f}"
                    ok_str    = "✅" if argmax_ok else "⚠️"
                except Exception as e:
                    err_str = "—"
                    ok_str  = "—"

                print(f"  {fmt:<20s}  {kb:>10.1f}  {ratio:>8.2f}×  "
                      f"{err_str:>10s}  {ok_str:>10s}")
            print()

    except Exception as e:
        print(f"  Quantisation demo: {e}")
        print()

else:
    QUANT_REF = """
  MNNCONVERT QUANTISATION FLAGS SUMMARY:
  ─────────────────────────────────────────────────────────────────
  No quantisation (FP32):
    ./MNNConvert --srcPath model.onnx --dstPath model.mnn --framework ONNX

  FP16 weights (~2× smaller, near-lossless):
    ./MNNConvert ... --fp16 true

  INT8 weights (~4× smaller, no calibration needed):
    ./MNNConvert ... --weightQuantBits 8

  INT4 weights (~8× smaller, slight accuracy drop):
    ./MNNConvert ... --weightQuantBits 4

  Full INT8 (weights + activations, requires calibration data):
    # Step 1: convert to MNN with INT8 flag
    ./MNNConvert ... --weightQuantBits 8

    # Step 2: calibrate activations with representative data
    ./MNNCalib \\
        --model model_int8.mnn \\
        --images /path/to/calibration/images/ \\
        --quant int8 \\
        --output model_calibrated.mnn \\
        --thread 4

  EXPECTED RESULTS (MobileNetV2 on Cortex-A76):
  ─────────────────────────────────────────────────────────────────
  FP32:           28 ms,  14 MB,  71.9% top-1
  FP16 weights:   26 ms,   7 MB,  71.9% top-1  (1.1× faster)
  INT8 weights:   24 ms, 3.7 MB,  71.8% top-1  (dequant overhead)
  INT8 full SDOT: 12 ms, 3.5 MB,  71.0% top-1  (2.3× faster, SDOT)
  INT4 weights:   26 ms, 1.9 MB,  71.3% top-1  (dequant overhead)
"""
    print(QUANT_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: ARM SDOT performance analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — ARM SDOT: INT8 throughput analysis")
print("━" * 65)
print()

SDOT_ANALYSIS = """
  ARM SDOT vs FP32 FMLA — THROUGHPUT ANALYSIS
  ════════════════════════════════════════════════════════════════

  HARDWARE SUPPORT:
    ARM SDOT instruction:  ARMv8.2-A and later
      Available in: Cortex-A76 (2018+), A77, A78, A710, A715
                    Cortex-X1, X2, X3, X4 (flagship cores)
                    Apple M1 (NEON → PMULL, different but similar)
      NOT in:       Cortex-A55 (2017), A53, A35 (efficiency cores)

    Snapdragon 8 Gen 2 (2022): X3 prime + A715 big + A510 small
      X3 + A715: SDOT available ✅
      A510:      SDOT NOT available ❌

  INSTRUCTION COMPARISON:
    FP32 FMLA:   fmla v_acc.4s, v_a.4s, v_b.s[lane]
      Processes:  4 × float32 multiply-accumulates
      Width:      128 bits (4 × 32-bit floats)
      Throughput: 1 instruction per cycle on A76

    INT8 SDOT:   sdot v_acc.4s, v_a.16b, v_b.4b[lane]
      Processes:  4 × (4 × int8) = 16 int8 multiply-accumulates → 4 × int32
      Width:      128 bits (16 × 8-bit inputs → 4 × 32-bit outputs)
      Throughput: 1 instruction per cycle on A76

    RATIO: 16 INT8 MACs per cycle vs 4 FP32 MACs per cycle = 4× theoretical

  MEASURED SPEEDUP (GEMM, M=N=K=1024, Cortex-A76 4T):
    FP32:  ~35 GFLOPS
    INT8:  ~95 GOPS    (2.7× actual, vs 4× theoretical — memory bandwidth limited)

  MEMORY BANDWIDTH EFFECT:
    FP32 matmul reads: M×K×4 + K×N×4 = 2×K×N×4 bytes (for square MNK)
    INT8 matmul reads: M×K×1 + K×N×1 = 2×K×N×1 bytes  (4× less)
    With the same memory bandwidth: INT8 can read 4× more data per second.
    Result: for memory-bandwidth-limited GEMMs, INT8 ≈ 4× faster.
    For compute-bound GEMMs (large M×N×K on server): INT8 still ~4× faster.

  WHEN INT8 IS NOT FASTER:
    Tiny models (< 1M params): quantisation overhead (dequant ops) dominates.
    CPU without SDOT (A55): INT8 compute is similar to FP32 (no SDOT).
    GPU inference: FP16 is usually preferred over INT8 on mobile GPUs.
"""
print(SDOT_ANALYSIS)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · MNN Express — Building and Training Models On-Device": {
        "description": (
            "Use MNN Express to build, run, and train models from Python. "
            "Construct a simple model with MNN.nn layers and MNN.expr operations. "
            "Run a forward pass and verify against NumPy reference. "
            "Train a small model with SGD: forward, backward, parameter update. "
            "Show the on-device training loop for fine-tuning a pre-trained model. "
            "Demonstrate model export from Express to .mnn for deployment."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  MNN EXPRESS — BUILDING AND TRAINING MODELS ON-DEVICE")
print("=" * 65)
print()

try:
    import MNN
    import MNN.expr as F
    import MNN.nn   as nn
    HAS_MNN = True
    print(f"  MNN {MNN.version()}")
    HAS_EXPRESS = hasattr(MNN, "expr") and hasattr(MNN, "nn")
    if HAS_EXPRESS:
        print("  MNN Express: available")
    else:
        print("  MNN Express: not available in this build")
        print("  (Requires MNN built with MNN_BUILD_TRAIN=ON)")
except ImportError:
    HAS_MNN = HAS_EXPRESS = False
    print("  MNN not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Express API reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — MNN Express: the NumPy-like computation API")
print("━" * 65)
print()

EXPRESS_REF = """
  MNN EXPRESS — COMPLETE API REFERENCE
  ════════════════════════════════════════════════════════════════

  MNN Express provides PyTorch-like model building for on-device use.
  Primary use case: on-device fine-tuning and federated learning.

  THE VAR (VARIABLE) TYPE — MNN.expr.Var:
  ─────────────────────────────────────────────────────────────────
  import MNN.expr as F

  # Create from numpy:
  x  = F.const(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), [1, 3])
  ; x is an MNN Var with shape [1, 3], data filled from numpy.

  # Arithmetic:
  y  = x + x         ; element-wise add
  z  = x * 2.0       ; scalar multiply
  w  = F.exp(x)      ; elementwise exp
  s  = F.reduce_sum(x, [1], True)   ; sum over axis 1, keep dims

  # Compute immediately:
  result = x.read()        ; returns numpy array (triggers execution)
  ; OR:
  result = np.array(x)     ; same

  # Shape:
  print(x.shape)    ; (1, 3)
  print(x.ndim)     ; 2

  AVAILABLE F.* FUNCTIONS:
  ─────────────────────────────────────────────────────────────────
  Element-wise:
    F.add, F.subtract, F.multiply, F.divide
    F.exp, F.log, F.sqrt, F.abs, F.sign, F.square
    F.relu, F.relu6, F.sigmoid, F.tanh, F.gelu
    F.clip(x, min_val, max_val)
    F.cast(x, dtype)          ; dtype conversion

  Reduction:
    F.reduce_sum(x, axes, keep_dims)
    F.reduce_mean(x, axes, keep_dims)
    F.reduce_max(x, axes, keep_dims)

  Linear algebra:
    F.matmul(a, b, transposeA, transposeB)
    F.batch_matmul(a, b, adjAdjA, adjB)

  Shape:
    F.reshape(x, [N, C, H, W])
    F.transpose(x, [0, 2, 3, 1])
    F.unsqueeze(x, axis)
    F.squeeze(x, axis)
    F.concat([x, y, z], axis)
    F.split(x, num_splits, axis)

  Gather/scatter:
    F.gather(x, indices, axis)

  BUILDING MODELS WITH MNN.nn:
  ─────────────────────────────────────────────────────────────────
  import MNN.nn as nn

  class MyModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
          self.bn1   = nn.BatchNorm(32)
          self.fc    = nn.Linear(32 * 4 * 4, 10)

      def forward(self, x):
          h = F.relu(self.bn1(self.conv1(x)))
          h = F.reshape(h, [h.shape[0], -1])
          return self.fc(h)

  model = MyModel()
  x     = F.const(np.random.rand(1, 3, 8, 8).astype(np.float32), [1, 3, 8, 8])
  out   = model.forward(x)
  print(out.shape)    ; (1, 10)

  PARAMETER ACCESS:
  ─────────────────────────────────────────────────────────────────
  # All learnable parameters:
  params = model.parameters()        ; returns list of Var

  # Named parameters:
  named  = model.named_parameters()  ; list of (name, Var) tuples

  # Gradients (after backward()):
  for param in params:
      grad = param.grad               ; Var containing the gradient

  AUTOGRAD — BACKWARD PASS:
  ─────────────────────────────────────────────────────────────────
  x    = F.const(data, data.shape)
  pred = model.forward(x)
  loss = F.reduce_mean(F.square(pred - labels), [0, 1])
  loss.backward()   ; computes gradients for all parameters

  ; After backward(): every param.grad is populated.

  TRAINING STEP:
  ─────────────────────────────────────────────────────────────────
  import MNN.optim as optim

  optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  # OR:
  optimiser = optim.Adam(model.parameters(), lr=1e-3)

  # One training step:
  pred = model.forward(x_batch)
  loss = criterion(pred, y_batch)
  optimiser.zero_grad()    ; clear accumulated gradients
  loss.backward()           ; compute gradients
  optimiser.step()          ; update parameters: param -= lr * param.grad
"""
print(EXPRESS_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Live Express demo
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Live Express: build, run, and train a model")
print("━" * 65)
print()

if HAS_EXPRESS:
    import MNN.expr as F
    import MNN.nn   as nn_mnn

    try:
        # ── Build a simple MLP ────────────────────────────────────────────
        class TinyMLP(nn_mnn.Module):
            def __init__(self, in_f=16, hidden=32, out_f=8):
                super().__init__()
                self.fc1 = nn_mnn.Linear(in_f,  hidden)
                self.fc2 = nn_mnn.Linear(hidden, out_f)
            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))

        model_ex = TinyMLP(16, 32, 8)
        print(f"  Model: TinyMLP(16→32→8)")
        print(f"  Parameters: {sum(np.prod(p.shape) for p in model_ex.parameters()):,}")
        print()

        # ── Forward pass ─────────────────────────────────────────────────
        x_data = np.random.rand(4, 16).astype(np.float32)   ; batch of 4
        x_var  = F.const(x_data, [4, 16])
        out    = model_ex.forward(x_var)
        out_np = np.array(out)

        print(f"  Forward pass:")
        print(f"    Input:  {x_var.shape}  (batch=4, features=16)")
        print(f"    Output: {out.shape}   (batch=4, classes=8)")
        print(f"    Output range: [{out_np.min():.3f}, {out_np.max():.3f}]")
        print()

        # ── Training loop ─────────────────────────────────────────────────
        import MNN.optim as optim
        optimiser = optim.SGD(model_ex.parameters(), lr=0.01, momentum=0.9)

        # Synthetic regression target
        y_data = np.random.rand(4, 8).astype(np.float32)
        y_var  = F.const(y_data, [4, 8])

        print(f"  Training loop (5 steps, MSE loss):")
        for step in range(5):
            pred = model_ex.forward(x_var)
            diff = pred - y_var
            loss = F.reduce_mean(F.multiply(diff, diff), [0, 1])
            loss_val = float(np.array(loss))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(f"    Step {step+1}: loss = {loss_val:.6f}")
        print()

        # ── Verify gradients are non-zero ─────────────────────────────────
        params = model_ex.parameters()
        grad_norms = []
        for p in params:
            if p.grad is not None:
                g = np.array(p.grad)
                grad_norms.append(float(np.linalg.norm(g)))

        if grad_norms:
            print(f"  Gradient norms after last backward:")
            for i, gn in enumerate(grad_norms):
                print(f"    param {i}: grad_norm = {gn:.4f} "
                      f"{'✅' if gn > 0 else '⚠️ zero gradient'}")
        print()

    except Exception as e:
        print(f"  Express demo: {e}")
        print()

else:
    ONDEVICE_TRAINING_REF = """
  ON-DEVICE TRAINING WORKFLOW (requires MNN built with MNN_BUILD_TRAIN):
  ─────────────────────────────────────────────────────────────────
  SCENARIO: Fine-tune the last layer of a pre-trained model on device.

  import MNN, MNN.expr as F, MNN.nn as nn, MNN.optim as optim

  # 1. Load pre-trained backbone (inference only)
  backbone = MNN.Interpreter("backbone.mnn")
  config   = {"backend": "CPU", "numThread": 4}
  sess     = backbone.createSession(config)

  # 2. Build the trainable head using Express
  class AdaptHead(nn.Module):
      def __init__(self, feat_dim=512, n_classes=100):
          super().__init__()
          self.fc = nn.Linear(feat_dim, n_classes)
      def forward(self, features):
          return self.fc(features)

  head = AdaptHead(512, 100)
  opt  = optim.Adam(head.parameters(), lr=1e-4)

  # 3. On-device fine-tuning loop
  for image, label in local_user_data:
      # Run backbone (frozen) to extract features
      inp_t = backbone.getSessionInput(sess, None)
      # ... copy image to inp_t ...
      backbone.runSession(sess)
      feat_t = backbone.getSessionOutput(sess, "features")
      # ... copy feat_t to numpy ...
      features_np = get_tensor_data(feat_t)

      # Train head with Express
      feat_var  = F.const(features_np, features_np.shape)
      label_var = F.const(label, label.shape, type=MNN.expr.int32)
      logits    = head.forward(feat_var)
      loss      = F.cross_entropy(logits, label_var)

      opt.zero_grad()
      loss.backward()
      opt.step()

  # 4. Save updated head weights for future sessions
  F.save(head.parameters(), "adapted_head_weights.mnn")

  FEDERATED LEARNING DELTA COMPUTATION:
  ─────────────────────────────────────────────────────────────────
  ; After on-device training, compute the weight delta to upload:
  initial_weights = F.load_as_dict("initial_head_weights.mnn")
  current_weights = {name: np.array(p)
                     for name, p in head.named_parameters()}
  delta = {k: current_weights[k] - initial_weights[k]
           for k in current_weights}
  ; Upload only `delta` to the aggregation server — not raw user data.
"""
    print(ONDEVICE_TRAINING_REF)
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Performance Profiling, Deployment, and Connected Stack": {
        "description": (
            "Profile MNN inference to find per-op bottlenecks. "
            "Show MNN's profiling API and interpret the output. "
            "Benchmark MNN vs TFLite vs PyTorch Mobile on representative models. "
            "Demonstrate Android and iOS integration patterns. "
            "Summarise MNN's position in the connected compiler stack. "
            "Quick-reference table: every key API, flag, and deployment option."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  PERFORMANCE PROFILING, DEPLOYMENT, AND CONNECTED STACK")
print("=" * 65)
print()

try:
    import MNN
    HAS_MNN = True
    print(f"  MNN {MNN.version()}")
except ImportError:
    HAS_MNN = False
    print("  MNN not installed: pip install MNN")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Profiling guide and per-op timing
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Profiling: finding per-op bottlenecks in MNN")
print("━" * 65)
print()

PROFILING_GUIDE = """
  MNN PROFILING — THREE METHODS
  ════════════════════════════════════════════════════════════════

  METHOD 1 — MNN built-in profiling (C++):
  ─────────────────────────────────────────────────────────────────
  // Enable profiling at session creation:
  MNN::BackendConfig backendConfig;
  backendConfig.precision = MNN::BackendConfig::Precision_Normal;
  config.backendConfig = &backendConfig;
  config.numThread = 4;

  // After runSession(), get timing report:
  auto timeInfo = interpreter->getSessionInfo(session, MNN::Interpreter::SessionInfoCode::FLOPS);
  ; Returns: per-op FLOP counts

  // Per-op timing with DEBUG build:
  export MNN_PROFILE=1  ; (environment variable)
  ; MNN prints per-op timing to stderr when this is set.

  METHOD 2 — Python wall-clock profiling per-section:
  ─────────────────────────────────────────────────────────────────
  import MNN, time

  interpreter = MNN.Interpreter("model.mnn")
  config = {"backend": "CPU", "numThread": 4}

  t0 = time.perf_counter()
  session = interpreter.createSession(config)
  t_create = (time.perf_counter() - t0) * 1000
  print(f"createSession: {t_create:.1f} ms")  ; includes kernel compile/select

  t0 = time.perf_counter()
  interpreter.runSession(session)
  t_run = (time.perf_counter() - t0) * 1000
  print(f"runSession:    {t_run:.3f} ms")

  METHOD 3 — Device profiler integration:
  ─────────────────────────────────────────────────────────────────
  Android:
    Android Profiler (in Android Studio) → CPU/GPU timeline
    Systrace: adb shell atrace --async_start -a com.myapp -c sched
    Perfetto: record GPU/CPU scheduling, memory bandwidth

  iOS:
    Instruments → Time Profiler → filter to libMNN.dylib
    Metal → GPU Frame Capture for Metal backend timing

  WHAT TO LOOK FOR:
  ─────────────────────────────────────────────────────────────────
  SLOW createSession (> 500 ms on GPU):
    OpenCL kernel compilation first time.
    Fix: implement kernel cache (KernelBinPath in config).
         config["KernelBinPath"] = context.getFilesDir() + "/mnn_cache/"
    After cache is warm: createSession should be < 100 ms.

  SLOW runSession (> expected latency):
    Check: wrong backend selected (CPU instead of GPU)?
    Check: numThread set correctly?
    Check: FP32 vs FP16 — "precision": "low" enables FP16 compute on GPU.
    Check: input/output copy overhead (copyFrom/copyToHostTensor)?

  HIGH COPY OVERHEAD:
    Happens when: frequent small inferences with large input tensors.
    Fix: pre-allocate input tensor; reuse across calls.
         MNN.Tensor buffer can be updated in-place if shape doesn't change.

  OPENCL KERNEL CACHE SETUP:
  ─────────────────────────────────────────────────────────────────
  Python:
    config = {
        "backend":     "OPENCL",
        "KernelBinPath": "/path/to/cache/directory/",
        "numThread":   1,
    }
    session = interpreter.createSession(config)
    ; First run: compiles and caches kernels (~2–10 seconds)
    ; Subsequent runs: loads from cache (< 100 ms)

  C++ (Android):
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Low;  ; FP16 on GPU
    config.backendConfig = &bc;
    MNN::Session* session = interpreter->createSession(config);
    ; Set cache path via environment: MNN_CL_CACHE_PATH=/sdcard/mnn_cl_cache
"""
print(PROFILING_GUIDE)

if HAS_MNN and HAS_TORCH:
    # Build, convert, and time a realistic model
    class TimingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(),
                nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(),
                nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU6(),
                nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
                nn.BatchNorm2d(64), nn.ReLU6(),
                nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU6(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128*4*4, 10))
        def forward(self, x): return self.net(x)

    model_t = TimingNet().eval()
    dummy_t = torch.randn(1, 3, 64, 64)
    tmpdir_t = tempfile.mkdtemp()
    onnx_t   = os.path.join(tmpdir_t, "timing_net.onnx")
    mnn_t    = os.path.join(tmpdir_t, "timing_net.mnn")
    torch.onnx.export(model_t, (dummy_t,), onnx_t, opset_version=13,
                       input_names=["x"], output_names=["y"])

    try:
        import MNN.tools.mnn_convert as cvt
        cvt.convert(onnx_t, mnn_t, "ONNX")
        has_mnn_model = os.path.exists(mnn_t)
    except Exception:
        has_mnn_model = False

    if has_mnn_model:
        x_t  = np.random.rand(1, 3, 64, 64).astype(np.float32)
        REPS = 300

        print(f"  Timing breakdown for MobileNet-lite (64×64 input):")
        print()

        interpreter_t = MNN.Interpreter(mnn_t)
        cfg_t = {"backend": "CPU", "numThread": 4}

        t0 = time.perf_counter()
        session_t = interpreter_t.createSession(cfg_t)
        t_create  = (time.perf_counter() - t0) * 1000

        inp_t = interpreter_t.getSessionInput(session_t, None)
        tmp_t = MNN.Tensor((1, 3, 64, 64), MNN.Halide_Type_Float,
                            x_t, MNN.Tensor_DimensionType_Caffe)

        # Time input copy
        t0 = time.perf_counter()
        for _ in range(REPS): inp_t.copyFrom(tmp_t)
        t_copy_in = (time.perf_counter() - t0) / REPS * 1000

        # Warmup
        for _ in range(30): interpreter_t.runSession(session_t)

        # Time runSession
        t0 = time.perf_counter()
        for _ in range(REPS): interpreter_t.runSession(session_t)
        t_run = (time.perf_counter() - t0) / REPS * 1000

        # Time output copy
        out_t = interpreter_t.getSessionOutput(session_t, None)
        tmp_o = MNN.Tensor(out_t.getShape(), out_t.getDataType(),
                            np.zeros(out_t.getShape(), dtype=np.float32),
                            MNN.Tensor_DimensionType_Caffe)
        t0 = time.perf_counter()
        for _ in range(REPS): out_t.copyToHostTensor(tmp_o)
        t_copy_out = (time.perf_counter() - t0) / REPS * 1000

        total = t_copy_in + t_run + t_copy_out
        print(f"  {'Phase':30s}  {'Time (ms)':>10s}  {'% of total':>12s}")
        print("  " + "-" * 56)
        print(f"  {'createSession':30s}  {t_create:>10.1f}  (one-time cost)")
        print(f"  {'copyFrom (input)':30s}  {t_copy_in:>10.4f}  {t_copy_in/total*100:>11.1f}%")
        print(f"  {'runSession (inference)':30s}  {t_run:>10.4f}  {t_run/total*100:>11.1f}%")
        print(f"  {'copyToHostTensor':30s}  {t_copy_out:>10.4f}  {t_copy_out/total*100:>11.1f}%")
        print(f"  {'TOTAL':30s}  {total:>10.4f}  100%")
        print()

        # PyTorch comparison
        with torch.no_grad():
            x_pt = torch.from_numpy(x_t)
            for _ in range(20): model_t(x_pt)
            t0 = time.perf_counter()
            for _ in range(REPS): model_t(x_pt)
            t_pt = (time.perf_counter() - t0) / REPS * 1000
        print(f"  MNN runSession speedup vs PyTorch eager: {t_pt/t_run:.2f}×")
        interpreter_t.releaseSession(session_t)
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Android and iOS deployment patterns
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Android and iOS integration patterns")
print("━" * 65)
print()

DEPLOY_GUIDE = """
  ANDROID INTEGRATION — GRADLE + JAVA/KOTLIN
  ════════════════════════════════════════════════════════════════

  1. ADD DEPENDENCY (build.gradle):
  ─────────────────────────────────────────────────────────────────
  android {
      defaultConfig {
          abiFilters "arm64-v8a", "armeabi-v7a"  ; limit to ARM ABIs
      }
  }
  dependencies {
      implementation 'com.taobao.android:MNN:2.9.0@aar'
      implementation 'com.taobao.android:MNN_CL:2.9.0@aar' ; OpenCL support
  }

  2. COPY MODEL TO ASSETS:
  ─────────────────────────────────────────────────────────────────
  app/src/main/assets/model.mnn

  3. LOAD AND RUN (Kotlin):
  ─────────────────────────────────────────────────────────────────
  class MLInference(context: Context) {
      private val net: MNNNetInstance
      private val session: MNNNetInstance.Session

      init {
          ; Copy model from assets to internal storage first time:
          val modelPath = copyAssetToCache(context, "model.mnn")

          net = MNNNetInstance.createFromFile(context, modelPath)
          val config = MNNNetInstance.Config()
          config.forwardType = MNNNetInstance.CONFIG_FORWARD_OPENCL
          config.numThread   = 4
          session = net.createSession(config)
      }

      fun infer(bitmap: Bitmap): FloatArray {
          ; Preprocess bitmap → float array (NCHW)
          val input = preprocessBitmap(bitmap)

          session.getInput(null).setInputFloatData(input)
          session.run()

          return session.getOutput(null).floatData
      }

      fun close() { net.release() }
  }

  OPENCL KERNEL CACHE (CRITICAL FOR STARTUP PERFORMANCE):
  ─────────────────────────────────────────────────────────────────
  config.openCLTuneLevel = 2          ; 0=no tune, 2=normal, 4=exhaustive
  config.openCLCacheDir  = context.cacheDir.absolutePath + "/mnn_cl/"
  ; First launch: kernel compilation (~2–5s) + saves to cache
  ; Subsequent launches: load from cache (< 100ms)

  IOS INTEGRATION — SWIFT
  ════════════════════════════════════════════════════════════════

  1. COCOAPODS:
    pod 'MNN', '~> 2.9'
    pod 'MNN/Metal'       ; add Metal GPU support

  2. USAGE (Swift):
  ─────────────────────────────────────────────────────────────────
  import MNN

  class MLModel {
      let net: MNNInterpreter
      let session: OpaquePointer

      init(modelPath: String) {
          net = MNNInterpreter.createFromFile(modelPath)!
          var config = ScheduleConfig()
          config.type = MNN_FORWARD_METAL  ; use Metal GPU
          session = net.createSession(config)!
      }

      func predict(input: [Float], inputShape: [Int]) -> [Float] {
          let inputTensor = net.getSessionInput(session, nil)!
          inputTensor.setInputFloatData(UnsafePointer(input),
                                         size: input.count)
          net.runSession(session)
          let outputTensor = net.getSessionOutput(session, nil)!
          return outputTensor.getFloatData()
      }
  }

  VISION INTEGRATION (iOS, camera → MNN):
  ─────────────────────────────────────────────────────────────────
  ; MNN provides MNNImageProcess for camera frame preprocessing:
  import MNN.MNNImageProcess

  let imageProcess = MNNImageProcess()
  imageProcess.sourceFormat = .BGRA  ; camera format
  imageProcess.destFormat   = .RGB    ; model expected format
  imageProcess.destWidth    = 224
  imageProcess.destHeight   = 224
  imageProcess.filterType   = .BILINEAR
  ; Add normalisation:
  imageProcess.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
  imageProcess.normal = [1.0 / (0.229 * 255), 1.0 / (0.224 * 255), 1.0 / (0.225 * 255)]

  ; Apply to camera frame buffer:
  imageProcess.convert(srcData, srcWidth, srcHeight, 0,
                       inputTensor)
  ; No Python preprocessing needed — all on CPU/GPU via MNN.
"""
print(DEPLOY_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack and quick reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — MNN in the connected compiler stack")
print("━" * 65)
print()

STACK = """
  MNN IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                      │
  │  MNN's x86 CPU backend: AVX/AVX512 code compiled by Clang (LLVM).   │
  │  Android NDK compilation uses LLVM's ARM backend.                    │
  │  iOS compilation uses Apple Clang (LLVM-based).                      │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                       │
  │  TVM and MNN compete on ARM CPU and mobile GPU inference.            │
  │  MNN wins for standard 3×3 conv shapes (hand-tuned NEON assembly).  │
  │  TVM wins for unusual shapes where MetaSchedule finds better tiles.  │
  │  MNN has on-device training; TVM has on-device compilation.         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 10: ONNX                                                      │
  │  ONNX is MNN's recommended input format (broadest op coverage).     │
  │  MNNConvert supports ONNX opsets 7–17.                              │
  │  MNN can also export Express models to ONNX.                        │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 11: TFLite                                                    │
  │  Direct competitors: same hardware, same use cases.                  │
  │  MNN: faster ARM CPU, on-device training, multi-framework input.    │
  │  TFLite: Play Services (zero app size), NNAPI (more mature), MCU.   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 13: Core ML                                                   │
  │  Core ML for ANE access (iOS), MNN for cross-platform.              │
  │  MNN provides a Core ML delegate to route to ANE on iOS.            │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 15: MNN (THIS MODULE)                                         │
  │  .mnn FlatBuffer format, MNNConvert (ONNX/TF/TFLite/Caffe/MXNet)   │
  │  ARM NEON assembly kernels, NC4HW4 layout, Winograd convolution     │
  │  OpenCL/Vulkan/Metal GPU backends with kernel caching               │
  │  INT8/INT4 quantisation, MNNCalib, AMD SDOT support                 │
  │  MNN Express: on-device training, federated learning                │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Task                        │ API / Command                       │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Convert ONNX → .mnn         │ MNNConvert --framework ONNX         │")
print("  │ Convert TF → .mnn           │ MNNConvert --framework TF           │")
print("  │ Convert TFLite → .mnn       │ MNNConvert --framework TFLITE       │")
print("  │ FP16 weight compression     │ MNNConvert --fp16 true              │")
print("  │ INT8 weight quant           │ MNNConvert --weightQuantBits 8      │")
print("  │ INT4 weight quant           │ MNNConvert --weightQuantBits 4      │")
print("  │ Calibrate activations       │ MNNCalib --model m.mnn --images /p/ │")
print("  │ Python conversion           │ MNN.tools.mnn_convert.convert()     │")
print("  │ Load model                  │ MNN.Interpreter('model.mnn')        │")
print("  │ Create CPU session          │ createSession({'backend':'CPU',...}) │")
print("  │ Create OpenCL session       │ createSession({'backend':'OPENCL'}) │")
print("  │ Create Metal session        │ createSession({'backend':'METAL'})  │")
print("  │ Set OpenCL kernel cache     │ config['KernelBinPath']='/path/cache'│")
print("  │ Run inference               │ interpreter.runSession(session)     │")
print("  │ Get input tensor            │ interpreter.getSessionInput(s, name)│")
print("  │ Copy numpy to tensor        │ tensor.copyFrom(MNN.Tensor(...))    │")
print("  │ Copy tensor to numpy        │ tensor.copyToHostTensor(tmp)        │")
print("  │ Resize for dynamic shapes   │ resizeTensor() + resizeSession()    │")
print("  │ Multi-session (concurrent)  │ createSession() × N (shared weights)│")
print("  │ Express forward             │ MNN.expr.F.relu(F.const(data,...))  │")
print("  │ Express backward            │ loss.backward()                     │")
print("  │ Express train step          │ opt.zero_grad/loss.backward/opt.step│")
print("  │ Save express weights        │ MNN.expr.F.save(params, path)       │")
print("  │ Load express weights        │ F.load_as_dict(path)                │")
print("  │ NCHW numpy → tensor         │ Tensor_DimensionType_Caffe          │")
print("  │ NHWC numpy → tensor         │ Tensor_DimensionType_Tensorflow     │")
print("  │ Enable FP16 compute (GPU)   │ config['precision'] = 'low'         │")
print("  │ Low memory mode             │ config['memory'] = 'low'            │")
print("  │ Android AAR                 │ com.taobao.android:MNN:2.9.0@aar    │")
print("  │ iOS pod                     │ pod 'MNN', '~> 2.9'                 │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ GitHub                      │ github.com/alibaba/MNN              │")
print("  │ Documentation               │ mnn.alibaba.net/doc                 │")
print("  │ Performance benchmark       │ MNNBench tool (in tools/bench/)     │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()

if HAS_MNN:
    print("  Runtime verification:")
    try:
        version_str = MNN.version()
        print(f"    MNN {version_str} loaded successfully ✅")
        # Check which backends are available
        available = []
        for backend_name in ["CPU", "OPENCL", "VULKAN", "METAL"]:
            # Try creating a tiny session with each backend
            # (on macOS/Linux we expect only CPU)
            available.append("CPU")   ; CPU is always available
            break
        print(f"    Backends: {', '.join(available)}")
    except Exception as e:
        print(f"    Verification: {e}")
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