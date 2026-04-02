"""
NCNN — Tencent's High-Performance Neural Network Inference Framework
======================================================================

NCNN is Tencent's open-source neural network inference framework optimised
for mobile platforms. Where MNN is Alibaba's answer to mobile inference,
NCNN is Tencent's — and the two embody fundamentally different philosophies.
MNN emphasises universality, multi-framework import, and on-device training.
NCNN emphasises a single design constraint above all else: ZERO EXTERNAL
DEPENDENCIES and the absolute minimum binary footprint possible.

NCNN was created by Nihui (a Tencent engineer) and released in July 2017,
predating most competing frameworks. The design goals were radical:
    No runtime dependency on any external library whatsoever.
    No Python runtime on the device — pure C++14 with only the STL.
    No dynamic memory allocation after model loading.
    No operating system abstraction layer — runs on bare metal if needed.
    Binary size: < 700 KB stripped for the core CPU-only runtime.

These constraints make NCNN uniquely suited to its production context:
WeChat (1.3 billion monthly active users), QQ, and Tencent's mobile game
portfolio (Honor of Kings, PUBG Mobile) all use NCNN for on-device AI —
face effects, real-time photo enhancement, game AI, and live filters running
on the oldest Android devices still in the user base (Android 4.0 era).

At that scale, a 1 MB larger binary means 1.3 million users downloading
an extra megabyte. A dependency on a shared library means a crash if that
library is missing on a particular OEM ROM. Zero dependencies is not an
aesthetic preference — it is a production reliability requirement.

NCNN's architecture is deliberately minimal:

    FORMAT:       .param (human-readable layer definition) +
                  .bin (raw weight blob). No FlatBuffers, no Protobuf.
                  The .param file can be edited in a text editor.

    RUNTIME:      A single Net class that loads .param + .bin and
                  creates Extractors for inference. No sessions, no sessions
                  factory, no backend plugin system — one implementation per
                  backend (CPU, Vulkan), selected at compile time.

    KERNELS:      Hand-written ARM NEON assembly for all performance-critical
                  paths. Vulkan compute shaders for GPU. No dependency on
                  MIOpen, oneDNN, or any external compute library.

    QUANTISATION: INT8 via a calibration-table approach, plus a new FP16
                  and BF16 path on newer ARM hardware.

In the connected compiler stack:
    LLVM      (module 01) ← NCNN kernels compiled with NDK Clang (LLVM-based)
    TVM       (module 08) ← TVM and NCNN compete on ARM CPU inference
    ONNX      (module 10) ← ONNX is NCNN's primary import path via onnx2ncnn
    TFLite    (module 11) ← TFLite and NCNN target the same hardware; NCNN lighter
    Core ML   (module 13) ← Core ML targets Apple ANE; NCNN targets all ARM
    MIGraphX  (module 14) ← MIGraphX targets AMD datacenter; NCNN targets mobile
    MNN       (module 15) ← MNN and NCNN are direct competitors; different trade-offs
    NCNN      (this)      ← Tencent's zero-dependency mobile inference framework

"""

import textwrap
import re

TOPIC_NAME   = "NCNN — Tencent's High-Performance Neural Network Inference Framework"
DISPLAY_NAME = "16 · NCNN"
ICON         = "⚡"
SUBTITLE     = ".param/.bin Format, ARM NEON Assembly, Vulkan GPU, INT8 Calibration, and pyncnn"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY NCNN EXISTS: TENCENT'S ZERO-DEPENDENCY CONSTRAINT

### Tencent's Deployment Reality

    WeChat is the infrastructure of daily life for 1.3 billion people.
    Updates cannot break on any phone. WeChat is installed on:
        Flagship Android phones (Samsung, Huawei, Xiaomi latest models)
        Mid-range phones (Redmi, OPPO, Vivo — the majority of the user base)
        Ancient budget devices: Android 4.0 phones with 512 MB RAM,
            ARMv7 CPUs without NEON, 8 GB internal storage
        iOS devices from iPhone 6 to iPhone 16 Pro Max
        HarmonyOS devices (Huawei's Android replacement)

    A library dependency in WeChat that crashes on a particular OEM ROM
    version affects millions of users. A 2 MB binary size increase triggers
    user complaints about storage usage on budget phones. A malloc inside
    inference that fails on a 512 MB RAM device causes a crash report.

    NCNN was designed with these constraints as first-class requirements,
    not afterthoughts.

### The Zero-Dependency Design Decision

    "Zero dependencies" means:
        No OpenBLAS, no MKL, no Eigen.
        No OpenCV (some utilities are provided but not required).
        No Protobuf, no FlatBuffers.
        No Python at runtime.
        No OpenCL at compile time (Vulkan is optional, compile-time flag).
        Standard C++ library only (std::vector, std::string, std::algorithm).

    WHY THIS IS HARDER THAN IT SOUNDS:
        Matrix multiply without BLAS: need to write GEMM from scratch.
        Model serialisation without Protobuf: need a custom format.
        SIMD optimisation without Eigen: need hand-written assembly.
        All three are necessary for competitive performance.
        NCNN does all three.

    BINARY SIZE COMPARISON (ARM64, stripped, release mode):
        NCNN (CPU only):    ~700 KB
        MNN (CPU only):     ~1.4 MB (2× larger)
        TFLite (CPU only):  ~900 KB
        ONNX Runtime:       ~8–15 MB (11–21× larger)
        PyTorch Mobile:     ~40 MB (57× larger)

    This matters: WeChat updates are reviewed for size increase by product.
    Every KB of the inference runtime is justified.

### Production Deployment at Tencent

    NCNN is deployed in:
        WeChat real-time photo filters (face enhancement, background blur)
        WeChat Live (real-time effects at 30fps on mid-range phones)
        QQ face recognition and augmented reality
        Honor of Kings (AI NPCs, real-time game effects)
        PUBG Mobile (AI-assisted features, scene understanding)
        Tencent Cloud edge inference (IoT devices)
        Tencent's medical imaging tools (on-device CT/MRI analysis on tablets)

    User base size requirements:
        Must run on Android 4.0 (API level 14+) — no modern API assumptions.
        Must run on ARMv7 without NEON (scalar fallback path exists).
        Must run on ARMv8 with NEON (primary optimised path).
        Must compile cleanly for iOS, Android, Linux, Windows, WebAssembly.

### NCNN vs MNN: A Design Philosophy Comparison

    Both are Chinese tech giant mobile inference engines from 2017–2018.
    They make opposite trade-offs:

    NCNN:
        Radical minimalism: zero deps, smallest possible binary.
        Pure C++14, no Python for runtime.
        Single-threaded design for deterministic latency.
        Custom text+binary format (.param + .bin).
        Inference ONLY (no training path).
        Excellent for: WeChat-style apps, games, IoT, embedded.

    MNN:
        Cross-platform universality: ONNX + TF + Caffe + MXNet + Torch.
        Python runtime API (pymnn), full training loop.
        OpenCL/Vulkan/Metal GPU backends.
        FlatBuffer format (same as TFLite).
        On-device training, federated learning.
        Excellent for: Taobao-style recommendation, NLP, server-side.

    WHEN TO CHOOSE NCNN:
        Binary size is the primary constraint.
        No Python environment at runtime.
        Targeting very old Android devices.
        Building games or apps where a dependency chain is unacceptable.
        Need Vulkan GPU on Android without OpenCL driver reliability issues.

    WHEN TO CHOOSE MNN:
        Need on-device training.
        Multiple framework models (ONNX + TF + TFLite all needed).
        iOS Metal GPU backend required.
        Python API for rapid prototyping.
        Alibaba cloud services integration.


##### PART 2 — THE .PARAM AND .BIN FORMAT

### Two Files, Zero Parsing Overhead

    An NCNN model is stored as exactly two files:

        model.param  —  human-readable text layer definition
        model.bin    —  raw binary weight data (little-endian floats)

    DESIGN RATIONALE:
        .param is TEXT: every layer, its connections, and its parameters
        are human-readable and editable in any text editor.
        No protobuf parser, no flatbuffers decoder — a simple text parser
        reads .param in microseconds.

        .bin is RAW BYTES: no headers, no framing, no schema.
        Raw float32/float16/int8 arrays concatenated sequentially.
        Reading: seek to the weight offset for layer N (offset computed
        by summing sizes of layers 0..N-1), read the bytes directly.
        No heap allocation — weights can be mmap'd from disk.

### The .param Text Format

    STRUCTURE:
        Line 1: magic number         "7767517"  (NCNN's magic constant)
        Line 2: layer_count blob_count   "47 42"   (47 layers, 42 weight blobs)
        Lines 3..N: one layer per line

    LAYER LINE FORMAT:
        layer_type  layer_name  input_count  output_count  input_names...  output_names...  [params...]

    EXAMPLE (ResNet-style block):
        7767517
        42 38
        Input            data             0 1 data
        Convolution      conv1            1 1 data conv1 0=64 1=7 2=1 3=2 4=3 5=1 6=9408
        BatchNorm        bn_conv1         1 1 conv1 bn_conv1 0=64
        Scale            scale_conv1      1 1 bn_conv1 scale_conv1 0=64 1=1
        ReLU             relu_conv1       1 1 scale_conv1 relu_conv1
        Pooling          pool1            1 1 relu_conv1 pool1 0=0 1=3 2=2 3=1 4=0
        ...
        Softmax          prob             1 1 fc8 prob 0=0
        ...

    LAYER PARAMETER ENCODING:
        Parameters follow a compact key=value syntax:
            0=64   : integer parameter with key 0, value 64
            1=0.1  : float parameter with key 1, value 0.1
           -23233=6,1,2,3   : array parameter (negative key = array, count first)

        EXAMPLE — Convolution layer parameters:
            0=num_output   (output channels)
            1=kernel_w     (kernel width)
            2=dilation_w   (dilation)
            3=stride_w     (stride)
            4=pad_left     (padding)
            5=bias_term    (0 or 1 — has bias or not)
            6=weight_data_size  (total weight elements = in_ch × out_ch × kH × kW)

    HUMAN READABILITY IN PRACTICE:
        You can open model.param in a text editor and:
            Understand the model architecture immediately.
            Edit layer names for debugging.
            Insert or remove layers (if you adjust the binary offsets).
            Trace the data flow from Input to output by following names.
        This is impossible with Protobuf (binary) or FlatBuffers (binary).

### The .bin Binary Format

    The .bin file is a flat concatenation of all weight blobs in the order
    they appear in .param. No headers, no framing, no random-access index.

    WEIGHT BLOB STRUCTURE:
        Each weight blob optionally starts with a 4-byte MAGIC NUMBER
        that indicates the data type and quantisation scheme:
            0x00000000:  raw float32 (legacy format)
            0x01306B47:  float32 with per-blob quantisation (NCNN internal)
            0x000D4B38:  float16 compressed weights
            0x0002C056:  INT8 quantised weights with scale tables

        After the magic: the raw weight data in the specified format.

    QUANTISATION IN THE BIN:
        INT8 quantised .bin stores:
            Per-output-channel INT8 weights (1 byte each).
            A float32 scale table: one scale per output channel.
        The scale table is appended AFTER the INT8 weight data.
        At inference: weight_float = weight_int8 × scale_channel.

    READING THE BIN IN PYTHON:
        import numpy as np

        with open("model.bin", "rb") as f:
            magic = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            if magic == 0x00000000:
                weights = np.frombuffer(f.read(n_elements * 4),
                                         dtype=np.float32)
            elif magic == 0x000D4B38:
                # float16 weights
                weights = np.frombuffer(f.read(n_elements * 2),
                                         dtype=np.float16).astype(np.float32)

    MMAP FOR ZERO-COPY LOADING:
        On Linux/Android, NCNN can memory-map the .bin file:
            net.load_param("model.param")
            net.load_model("model.bin")
        When opened with mmap: the weight data is not copied to heap.
        The OS maps the file directly into the process address space.
        Weight access: a read from mapped memory → hardware page fault →
        OS loads the page from storage on first access.
        For large models: only the pages actually touched are loaded.

### The .ncnn Encrypted Format and .mnn Packaging

    ENCRYPTED MODELS:
        Tencent provides tools to encrypt .param and .bin for IP protection:
            ncnnencrypt.py model.param model.bin model_enc.param model_enc.bin key
        The encrypted format uses AES-128-ECB.
        Production WeChat models are distributed in encrypted form.

    SINGLE-FILE .NCNN PACKAGING:
        For convenience, .param and .bin can be packed into a single file:
            ncnn_combine model.param model.bin model.ncnn
        NCNN can load the combined file directly.
        This is the format used by some NCNN Android libraries.


##### PART 3 — MODEL CONVERSION: ONNX, PYTORCH, AND CAFFE TO NCNN

### onnx2ncnn: The Primary Conversion Path

    NCNN's primary conversion tool is onnx2ncnn — a standalone C++ binary
    that reads an ONNX file and writes .param + .bin:

        onnx2ncnn model.onnx model.param model.bin

    ONNX OPSET SUPPORT:
        opsets 7–17 supported with broad operator coverage (~100 ops).
        Unknown ops: reported as warnings; the model fails to run on those ops.
        Best practice: simplify ONNX before converting:
            python -m onnxsim model.onnx model_simplified.onnx
            onnx2ncnn model_simplified.onnx model.param model.bin

    WHAT onnx2ncnn DOES:
        1. Reads the ONNX graph.
        2. Maps each ONNX op to the corresponding NCNN layer type.
        3. Translates ONNX attributes to NCNN's key=value parameter encoding.
        4. Writes weights to model.bin in NCNN's raw float32 format.
        5. Writes the layer graph to model.param in text format.

    UNSUPPORTED OPS:
        onnx2ncnn prints unsupported ops explicitly:
            [ERROR] unsupported op type STFT
        Options: (1) replace with supported ops in PyTorch before export,
                 (2) implement as NCNN custom layer (see Part 8),
                 (3) use onnx2ncnn-extended (community fork with more ops).

### ncnnoptimize: Post-Conversion Optimisation

    After conversion, always run ncnnoptimize:

        ncnnoptimize model.param model.bin model_opt.param model_opt.bin 0
        #                                                                 ^ flag: 0=fp32, 65536=fp16

    WHAT ncnnoptimize DOES:
        LAYER FUSION:
            Conv + BatchNorm → ConvolutionBatchNorm (merged weights)
            Conv + ReLU → Convolution with fuse_relu flag
            Conv + BatchNorm + ReLU → single fused layer
            These fusions are NCNN-specific and not done by onnx2ncnn.
            Without ncnnoptimize: BatchNorm remains a separate layer.

        LAYER ELIMINATION:
            Remove zero-padding layers (pad=0 is a no-op).
            Remove identity layers (Reshape with same shape).
            Remove dangling outputs.

        WEIGHT ABSORPTION:
            Scale + BatchNorm → fused Scale+BN layer with merged parameters.
            Bias terms absorbed into Conv or InnerProduct.

        FP16 WEIGHT COMPRESSION (flag=65536):
            All float32 weights in the .bin converted to float16.
            2× smaller .bin file.
            Activation computation still in FP32.
            Magic number in .bin updated to 0x000D4B38.

    RUN NCNNOPTIMIZE BEFORE DEPLOYMENT — ALWAYS:
        ncnnoptimize is not optional for production models.
        Without it: BatchNorm is a separate kernel call.
        With it: BatchNorm is folded into Conv weights at conversion time.
        Measured speedup from ncnnoptimize alone: 10–25% on typical CNNs.

### PyTorch → NCNN (via torch.onnx.export)

    STANDARD PATH:
        import torch
        torch.onnx.export(
            model.eval(),
            (dummy_input,),
            "model.onnx",
            opset_version=11,        ; prefer 11–13 for NCNN compat
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
        # Then: onnx2ncnn + ncnnoptimize

    IMPORTANT PYTORCH → NCNN SETTINGS:
        opset_version:  11 or 12 (NCNN's coverage is best here)
                        Avoid opset 17+ (newer ONNX ops may lack NCNN mapping)
        do_constant_folding=True:  fold shape computation into constants
        avoid dynamic control flow: no if/while inside forward()
        avoid .item() calls: these create ONNX If nodes
        prefer static shapes at export time (but dynamic batch is fine)

    TORCH2NCNN (direct conversion, bypasses ONNX):
        A community tool exists (ncnn/tools/pnnx) that converts TorchScript
        directly to NCNN without the ONNX intermediate step:
            pnnx model.pt input.shape="[1,3,224,224]"
        PNNX preserves more structural information than ONNX export and
        produces cleaner NCNN graphs for modern PyTorch models.
        Supported: TorchScript, torch.export.export output.

### Caffe → NCNN (Legacy Path)

    NCNN was originally designed for Caffe model deployment.
    The Caffe converter is mature and has the best op coverage:

        caffe2ncnn deploy.prototxt model.caffemodel model.param model.bin

    Most modern models are no longer in Caffe format, but NCNN's Caffe
    importer remains the most reliable path for legacy production models
    (AlexNet, VGG, older ResNets, FaceNet-style models).

### PNNX: The Modern Direct Converter

    PNNX (PyTorch Neural Network Exchange) is NCNN's answer to ONNX.
    It was developed specifically to overcome ONNX's limitations:

    ONNX PROBLEMS FOR PYTORCH → NCNN:
        ONNX flattens complex ops (attention, normalization) into many
        primitive ops. The resulting NCNN graph has 200+ layers for
        a transformer that should have ~30 meaningful operations.
        This reduces performance because NCNN dispatches each layer
        individually; more layers = more dispatch overhead.

    PNNX APPROACH:
        PNNX keeps high-level PyTorch ops as single NCNN layers:
            nn.MultiheadAttention → NCNN MultiHeadAttention (1 layer)
            nn.LayerNorm          → NCNN LayerNorm (1 layer)
            torch.nn.functional.scaled_dot_product_attention → 1 layer
        The resulting model has far fewer layers and dispatches.
        Transformer models: 5–10× fewer PNNX layers vs ONNX-derived.

    PNNX USAGE:
        # Trace the model with PNNX:
        import torch
        from pnnx import export

        model  = MyTransformer().eval()
        x      = torch.randn(1, 64, 256)
        export(model, "model.pnnx.param", "model.pnnx.bin",
               inputs=[x])

        # PNNX also exports directly to NCNN format:
        # model.ncnn.param + model.ncnn.bin are generated automatically.


##### PART 4 — THE LAYER SYSTEM: OPS, REGISTRATION, AND DISPATCH

### NCNN's Layer Architecture

    Every operation in NCNN is a LAYER — a C++ class that inherits from
    ncnn::Layer. There are approximately 200 built-in layer types.

    LAYER RESPONSIBILITIES:
        load_param():  read the key=value parameters from .param
        load_model():  read weight blobs from .bin
        forward():     execute the computation (CPU path)
        forward_inplace(): execute in-place (for elementwise ops)

    LAYER LIFECYCLE:
        PARSE TIME:  Net reads .param → creates Layer objects → calls load_param()
        LOAD TIME:   Net reads .bin → calls load_model() for each layer with weights
        INFERENCE:   Extractor calls forward() on each layer in topological order

    SHAPE INFERENCE:
        NCNN computes output shapes at model load time (not inference time).
        Each layer implements infer_output_shape() which is called once
        after load_model().
        At inference: no shape computation — all shapes are pre-determined.

### The Built-in Layer Catalogue

    CONVOLUTION FAMILY:
        Convolution:              standard 2D conv (handles all configs)
        ConvolutionDepthWise:     depthwise conv (groups = channels)
        DeformableConv2D:         deformable convolution
        ConvolutionBackpropInput: transposed conv (upsampling)
        InnerProduct:             fully connected (weight matrix multiply)

    NORMALISATION:
        BatchNorm:        inference-mode BN (usually fused into conv)
        GroupNorm:        group normalisation
        LayerNorm:        layer normalisation
        InstanceNorm:     instance normalisation
        LRN:              local response normalisation

    ACTIVATION:
        ReLU:      relu with optional negative slope (= leaky relu when slope > 0)
        SELU:      scaled exponential linear unit
        ELU:       exponential linear unit
        HardSigmoid, HardSwish, Swish, Mish, GELU (exact and tanh approx)
        Sigmoid:   logistic sigmoid
        TanH:      hyperbolic tangent
        Clip:      clamp to [min, max]
        AbsVal:    absolute value

    POOLING:
        Pooling:         max / average, with optional global flag
        Pooling1D:       1D pooling for sequence models

    ATTENTION AND TRANSFORMERS:
        MultiHeadAttention:   fused scaled-dot-product attention
        GLU:                  gated linear unit
        Gemm:                 general batched matmul

    NORMALISATION + ARITHMETIC:
        Scale:       element-wise scale+bias (= BN without mean/var)
        Bias:        add a bias vector
        BinaryOp:    element-wise binary: add/sub/mul/div/pow/max/min/...
        UnaryOp:     element-wise unary: abs/neg/floor/ceil/round/sqrt/...
        Reduction:   reduce sum/mean/max/min/L1/L2 over axes
        Softmax:     softmax over a specified axis

    SHAPE MANIPULATION:
        Reshape:     change shape
        Permute:     transpose axes
        Flatten:     flatten to 1D (from C×H×W to CHW)
        Squeeze:     remove size-1 dims
        ExpandDims:  add size-1 dim
        Concat:      concatenate along axis
        Split:       split into N parts
        Slice:       slice along axis
        Crop:        crop to subregion
        Padding:     pad with constant/reflect/replicate
        ShuffleChannel: channel shuffle (ShuffleNet operation)

    GATHER / SCATTER:
        Gather:      index gather (embedding lookup)
        GatherElements: per-element gather
        ScatterElements: per-element scatter

    RECURRENT:
        RNN:   simple RNN (forward and bidirectional)
        LSTM:  long short-term memory
        GRU:   gated recurrent unit

    DETECTION / VISION:
        Yolo:             YOLO detection output processing
        YoloV3DetectionOutput: YOLOv3 post-processing
        NMSSoftmax:       soft non-maximum suppression
        ROIPooling:       region of interest pooling
        PriorBox, DetectionOutput: SSD post-processing

    SPECIAL:
        Noop:        no operation (placeholder)
        Custom:      user-registered custom layer

### Layer Parameter Key Encoding

    NCNN's key=value parameter encoding is type-aware:

    KEY SPACE:
        0–29:    int parameters        (e.g., 0=num_output for Conv)
        30–59:   float parameters      (e.g., 30=eps for BatchNorm)
        -23233 and below: array params (negative → array, count at key+1)

    ARRAY PARAMETER SYNTAX:
        key=-count,v0,v1,v2,...
        Example: -23300=3,64,128,256
            key -23300 → output shape specification
            count 3 → 3 values follow
            values: 64, 128, 256

    FULL CONVOLUTION PARAMETER SPEC:
        0=num_output       (output channels)
        1=kernel_w         (kernel width; kernel_h = kernel_w unless split)
        11=kernel_h        (kernel height, optional)
        2=dilation_w       (dilation width)
        12=dilation_h      (dilation height, optional)
        3=stride_w         (stride width)
        13=stride_h        (stride height, optional)
        4=pad_left         (left padding)
        14=pad_right, 15=pad_top, 16=pad_bottom (other padding sides)
        5=bias_term        (0=no bias, 1=has bias)
        6=weight_data_size (total weight count for binary reading)
        9=activation_type  (0=none, 1=relu, 2=leaky_relu, 3=clip, 4=sigmoid)
        10=activation_params  (e.g., slope for leaky relu)
        8=int8_scale_term  (0=no INT8, 1=INT8 bottom, 2=INT8 top, 3=both)


##### PART 5 — THE CPU BACKEND: ARM NEON, PACKING, AND WINOGRAD

### NCNN's NEON Packing Strategy

    NCNN's primary performance insight is the same as MNN's but arrived
    at independently: group channels into packs of 4 (or 8 for AVX).

    NCNN's layout is called PACK4:
        For a tensor [C, H, W]:
            Divide C into groups of 4: c_groups = C / 4
            Rearrange data: for each spatial position (h, w),
            store all 4 channel values consecutively.
            Memory shape: [c_groups, H, W, 4]
            Access: data[c_group * H * W * 4 + h * W * 4 + w * 4 + lane]

        For ARM NEON (float32 × 4):
            One float32x4_t register holds one PACK4 channel group.
            Convolution inner loop: vfmaq_f32(acc, inp, flt) processes
            4 input channels × 4 output channels = 16 MACs per instruction.

    PACK8 (AVX / ARM SVE):
        On x86 with AVX2: groups of 8 channels (float × 8 per AVX register).
        On ARM with SVE (Cortex-X2, A710): groups of 8 with 256-bit vectors.
        Selecting between PACK4 and PACK8: done at Net.opt.use_bf16_storage
        or at compile time based on detected CPU features.

    PACK1 (fallback):
        When channels are not divisible by 4: PACK1 (standard NCHW).
        Used for the first conv layer (3 input channels = not divisible by 4).
        Transition layers: PACK1 → PACK4 when first conv outputs ≥ 4 channels.

### Winograd Convolution

    NCNN implements Winograd convolution for 3×3, stride-1 conv:
        Algorithm: F(4×4, 3×3) — 4×4 output tile, 3×3 kernel.
        Arithmetic reduction: 9 multiplies → 4 multiplies (2.25× fewer).
        Additional cost: input and filter transform (extra additions).
        Breakeven: profitable when output spatial size ≥ 16×16.

    NCNN's Winograd is PACK4-aware:
        The filter transform produces PACK4-layout transformed filters.
        The input transform produces PACK4-layout transformed inputs.
        The batched GEMM (4×4 tiles × PACK4 channels) uses NEON FMA.

    NCNN also implements Winograd F(6×6, 3×3):
        4 multiplies reduced further; good for large feature maps.
        Selected automatically based on output spatial size.

### ARM Feature Detection

    NCNN detects CPU features at runtime and selects the optimal path:

        #include "cpu.h"
        int neon    = ncnn::cpu_support_arm_neon();
        int fp16    = ncnn::cpu_support_arm_vfpv4();   ; FP16 storage
        int asimdhp = ncnn::cpu_support_arm_asimdhp(); ; FP16 compute (ARMv8.2)
        int dotprod = ncnn::cpu_support_arm_asimddp(); ; INT8 SDOT (ARMv8.2)
        int sve     = ncnn::cpu_support_arm_sve();      ; SVE (ARMv8.2-SVE)
        int sve2    = ncnn::cpu_support_arm_sve2();     ; SVE2

    DISPATCH:
        Each kernel has multiple implementations:
            Convolution::create_pipeline() selects between:
                conv_winograd_4x4_pack4 (NEON, output ≥ 16×16)
                conv_winograd_6x6_pack4 (NEON, output ≥ 24×24)
                conv_direct_pack4        (NEON, general)
                conv_direct_pack8        (AVX2/SVE, when available)
                conv_direct_pack1        (scalar, when no NEON)

### Thread Model: OpenMP

    NCNN uses OpenMP for intra-op parallelism on CPU:

        ncnn::Option opt;
        opt.num_threads = 4;
        net.opt = opt;

    NCNN THREADING IS MINIMAL:
        Unlike MNN's custom work-stealing pool, NCNN uses OpenMP directly.
        OpenMP team is created once and reused.
        Work distribution: parallel loops over output spatial tiles.

    AFFINITY CONTROL:
        NCNN provides CPU affinity helpers for big.LITTLE SoCs:
            ncnn::set_cpu_powersave(0);  ; use all cores
            ncnn::set_cpu_powersave(1);  ; prefer efficiency cores (low power)
            ncnn::set_cpu_powersave(2);  ; prefer big cores (high performance)
        For latency-sensitive inference: powersave=2 (big cores only).
        For background inference: powersave=1 (efficiency cores, saves battery).

### Memory Allocation Model

    NCNN pre-allocates all memory at load time:

    BLOB MEMORY:
        All input, output, and intermediate tensors (blobs) are allocated
        from an NCNN memory pool.
        The memory pool is computed by topological analysis:
            For each blob: lifetime [creation_layer, last_use_layer].
            Blobs with non-overlapping lifetimes share memory (interval colouring).
        After load(): ZERO dynamic allocations during inference.

    WEIGHT MEMORY:
        Weights either:
            a) Loaded into heap (default): copy from .bin to malloc'd buffer.
            b) Memory-mapped (opt.use_memory_mmap = true): mmap the .bin file.
        Memory-mapped weights: zero copy, pages loaded on demand.
        Best for: large models where not all weights are used every inference.

    SCRATCH MEMORY:
        Some layers (im2col for non-PACK4 conv, Winograd transform buffers)
        need temporary workspace.
        NCNN allocates these as thread-local "scratchpad" allocators.
        The scratchpad is reused across layers — at most one layer's scratch
        is alive at any time.


##### PART 6 — THE VULKAN GPU BACKEND

### Why Vulkan Instead of OpenCL

    NCNN chose Vulkan (not OpenCL) as its GPU backend, departing from MNN
    (which supports both). The reasons:

    RELIABILITY:
        Android OpenCL driver quality is notoriously inconsistent.
        On some OEM devices: OpenCL headers present but drivers crash.
        Vulkan is a mandatory Android API from Android 7.0 (API 24).
        A Vulkan API call that was valid in Android 7.0 is valid in Android 14.
        OpenCL has no such guarantee.

    EXPLICIT SYNCHRONISATION:
        OpenCL: implicit synchronisation between queued commands.
        Vulkan: explicit barriers and semaphores.
        NCNN uses explicit Vulkan synchronisation to pipeline:
            CPU preprocessing overlaps with GPU inference of previous batch.
        This pipelining is difficult to implement correctly in OpenCL.

    PRECOMPILED SHADERS:
        NCNN shaders are pre-compiled to SPIR-V at build time.
        No JIT shader compilation at runtime.
        Eliminates the 2–10 second first-run compilation delay that
        plagues OpenCL backends.

### Vulkan Backend Architecture

    NCNN'S SHADER PIPELINE:
        Each NCNN GPU layer has one or more SPIR-V compute shaders.
        Shaders are compiled from GLSL → SPIR-V at NCNN build time.
        The compiled SPIR-V is baked into the NCNN binary as a C array.
        At runtime: vkCreateComputePipeline from the pre-compiled SPIR-V.
        First pipeline creation: fast (no GLSL compilation needed).

    PIPELINE OBJECTS:
        Each shader becomes a VkPipeline object.
        VkPipelines are created once at model load time.
        Cached in the NCNN VulkanDevice object.
        Reused across inference calls.

    COMMAND BUFFER STRATEGY:
        Each inference call records a VkCommandBuffer.
        The command buffer contains: bind descriptors → dispatch → pipeline barrier.
        Submitted to the GPU command queue as a batch.
        Host waits for fence after submission.
        For asynchronous inference: NCNN provides async submit with callback.

    TENSOR STORAGE ON GPU:
        NCNN GPU uses VkBuffer (not VkImage/sampler).
        PACK4 layout is maintained on GPU.
        Input: CPU tensor → vkCmdCopyBuffer → GPU VkBuffer.
        Output: GPU VkBuffer → vkCmdCopyBuffer → CPU tensor.
        For pipelines where input comes from another GPU op: zero-copy.

### Enabling and Configuring the Vulkan Backend

    BUILD WITH VULKAN:
        cmake -DNCNN_VULKAN=ON ..
        ; Requires Vulkan SDK (vulkan_core.h) at build time.
        ; Generates SPIR-V shaders and embeds them in the binary.

    RUNTIME SETUP:
        #include "gpu.h"
        ncnn::create_gpu_instance();   ; initialise Vulkan runtime (once)

        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.load_param("model.param");
        net.load_model("model.bin");

        // After inference
        ncnn::destroy_gpu_instance();

    PYTHON (pyncnn):
        import ncnn
        ncnn.create_gpu_instance()
        net = ncnn.Net()
        net.opt.use_vulkan_compute = True
        net.load_param("model.param")
        net.load_model("model.bin")

    CHECKING VULKAN AVAILABILITY:
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0) {
            const ncnn::GpuInfo& info = ncnn::get_gpu_info(0);
            printf("GPU: %s\n", info.device_name());
            printf("GLSL: %d.%d\n", info.glsl_subgroup_size, ...);
        }

    GPU MEMORY MANAGEMENT:
        GPU blobs allocated in device-local VkBuffer (fastest GPU access).
        Input upload: host-visible staging buffer → device-local via copy.
        Output download: device-local → host-visible staging → CPU copy.
        For models where input is camera frame already on GPU (via
        Vulkan-capable camera or video decoder): zero-copy is possible
        by passing a VkBuffer directly to NCNN.

### Vulkan Performance Characteristics

    WHEN VULKAN IS FASTER THAN CPU:
        Large models (> 5M params): GPU parallelism dominates transfer cost.
        Batch size > 1: GPU scales better than CPU for batch inference.
        Image-heavy workloads: convolution on GPU outperforms CPU.
        High-resolution inputs (> 512×512): GPU occupancy is high.

    WHEN CPU IS FASTER THAN VULKAN:
        Tiny models (< 1M params): PCIe/memory copy overhead dominates.
        Single-sample, latency-critical: GPU kernel launch latency > CPU time.
        First inference: VkPipeline creation (even from pre-compiled SPIR-V).
        Models dominated by non-GPU-friendly ops (LSTM, complex gather).

    TYPICAL SPEEDUPS (Adreno 650, MobileNetV2 224×224, batch=1):
        CPU 4 threads:   28 ms
        Vulkan FP32:     10 ms  (2.8×)
        Vulkan FP16:      6 ms  (4.7×, with opt.use_fp16_storage=true)


##### PART 7 — INT8 QUANTISATION: THE CALIBRATION TABLE APPROACH

### NCNN's INT8 Philosophy

    NCNN's INT8 quantisation is simpler and more conservative than TFLite's
    or ONNX Runtime's:
        Weights: quantised to INT8 (symmetric, per-output-channel).
        Activations: quantised at specific layer boundaries only.
        Method: scale-based (same as TFLite symmetric INT8).
        No zero_point for weights (symmetric: zero always maps to zero).

    NCNN's INT8 is primarily a BANDWIDTH optimisation:
        INT8 weights are 4× smaller → 4× less memory bandwidth during weight reads.
        This is the dominant benefit on bandwidth-limited mobile CPUs.
        The GEMM itself still happens in INT32 (accumulate) with INT8 inputs,
        benefiting from ARM SDOT when available.

### INT8 Calibration Workflow

    NCNN uses a CALIBRATION TABLE approach:

    STEP 1 — PREPARE CALIBRATION DATA:
        A calibration dataset of 100–1000 representative images.
        Stored as a list of image paths in a text file (calib_list.txt).

    STEP 2 — GENERATE CALIBRATION TABLE:
        ncnn2table \
            --param model.param \
            --bin   model.bin   \
            --images calib_list.txt \
            --output model.table  \
            --mean   104,117,123 \
            --norm   1.0,1.0,1.0 \
            --size   224,224     \
            --thread 4

        ncnn2table:
            Runs all calibration images through the float model.
            For each activation tensor: records the max absolute value.
            Writes a text calibration table (model.table) with:
                layer_name  max_abs_value

    STEP 3 — QUANTISE THE MODEL:
        ncnn2int8 \
            model.param model.bin \
            model_int8.param model_int8.bin \
            model.table

        ncnn2int8:
            Reads the calibration table.
            For each eligible layer: computes scale = max_abs / 127.0.
            Quantises weights to INT8: w_q = round(w / scale_w).
            Embeds scale values in the .bin (after the INT8 weights).
            Sets int8_scale_term=3 in the layer parameters.
            Outputs model_int8.param + model_int8.bin.

    STEP 4 — DEPLOY:
        net.load_param("model_int8.param");
        net.load_model("model_int8.bin");
        ; NCNN automatically detects INT8 layers from the magic number in .bin.
        ; No special runtime configuration needed.

### Scale Computation Methods in ncnn2table

    ncnn2table supports several calibration methods:

    MAX (default):
        scale = max(|activation|) / 127.0
        Uses the absolute maximum seen across all calibration samples.
        Simple and fast. May produce poor scales if one sample has outliers.

    KL_DIVERGENCE:
        scale chosen to minimise KL divergence between quantised and float distributions.
        More robust to outlier activations.
        Slower than MAX (requires histogram computation).
        Better accuracy for models with wide activation distributions.

    PERCENTILE:
        scale = percentile(|activation|, 99.9) / 127.0
        Clips outlier values at the 99.9th percentile.
        Good middle ground: faster than KL, more robust than MAX.

    SELECTING THE METHOD:
        ncnn2table --calibration_method MAX      ; default
        ncnn2table --calibration_method KL       ; KL divergence
        ncnn2table --calibration_method PERCENTILE ; percentile clipping

### The INT8 Runtime Path

    When NCNN loads a model with int8_scale_term > 0:

    IN forward():
        1. Read INT8 weights from blob (magic 0x0002C056).
        2. Read scale table (float32 array appended after INT8 weights).
        3. Execute INT8 gemm: result_int32 = dotprod(input_q, weight_int8).
        4. Dequantise: result_fp32 = result_int32 × scale_w × scale_x.
        5. Apply activation (relu, etc.) in fp32.

    PERFORMANCE:
        On ARMv8.2+ with SDOT:  INT8 is 2–4× faster than FP32.
        On ARMv8.0 (no SDOT):   INT8 may be slightly SLOWER than FP32
                                 (manual expansion of 8-bit to 32-bit).
        Check at runtime:
            bool has_sdot = ncnn::cpu_support_arm_asimddp();

    ACCURACY:
        Typical accuracy drop: < 0.5% top-1 on ImageNet with good calibration.
        Bad calibration (too few samples, wrong distribution): up to 5% drop.
        Rule: always test on held-out validation data after INT8 conversion.


##### PART 8 — THE C++ AND PYTHON RUNTIME API

### The Net and Extractor Objects

    NCNN's runtime revolves around two objects:

    ncnn::Net:
        The model container. Loads .param + .bin. Holds all layer objects,
        all weight blobs, and the option configuration.
        One Net per model. Thread-safe for reads (inference).
        NOT thread-safe for writes (loading).

    ncnn::Extractor:
        An inference "lane" created from a Net.
        Holds the input/output blob handles.
        NOT thread-safe — one Extractor per thread.
        Lightweight to create: just stores references to the Net's layers.
        Create a new Extractor for each inference (or per thread).

### Full C++ Inference Workflow

    #include "net.h"

    // ── Load model (once) ──────────────────────────────────────────────
    ncnn::Net net;

    // Configure options BEFORE loading:
    net.opt.use_vulkan_compute = false;   ; or true for GPU
    net.opt.num_threads        = 4;
    net.opt.use_packing_layout = true;    ; enable PACK4 (default)
    net.opt.use_fp16_storage   = false;   ; or true for FP16 weights
    net.opt.use_bf16_storage   = false;   ; or true for BF16 compute

    net.load_param("model.param");
    net.load_model("model.bin");

    // ── Inference (called per request) ────────────────────────────────
    ncnn::Extractor ex = net.create_extractor();

    // Create input Mat
    ncnn::Mat input = ncnn::Mat(224, 224, 3);   ; W=224, H=224, C=3

    // Fill with preprocessed image data (subtract mean, divide std):
    ncnn::Mat::from_pixels_resize(image_bgr_ptr, ncnn::Mat::PIXEL_BGR2RGB,
                                   orig_w, orig_h, 224, 224, input);
    const float mean[3] = {0.485*255, 0.456*255, 0.406*255};
    const float std[3]  = {1.0/(0.229*255), 1.0/(0.224*255), 1.0/(0.225*255)};
    input.substract_mean_normalize(mean, std);

    // Set input, run, read output
    ex.input("input",  input);
    ncnn::Mat output;
    ex.extract("output", output);

    // output is a Mat with shape [num_classes]:
    float* scores = output.channel(0).row(0);   ; or just (float*)output.data
    int top1 = std::max_element(scores, scores + num_classes) - scores;

### ncnn::Mat: The Tensor Type

    NCNN's tensor type is ncnn::Mat — a reference-counted memory block
    with shape information.

    SHAPE CONVENTION:
        ncnn::Mat(w):           1D tensor [W]
        ncnn::Mat(w, h):        2D tensor [H, W]
        ncnn::Mat(w, h, c):     3D tensor [C, H, W]  — the standard image format
        ncnn::Mat(w, h, d, c):  4D tensor [C, D, H, W]

    IMPORTANT: w is the INNERMOST (fastest-varying) dimension.
    For images: w=width, h=height, c=channels (standard spatial order).

    CREATING MATS:
        ncnn::Mat m(224, 224, 3);           ; allocates 224×224×3 float32
        ncnn::Mat m_fp16(224, 224, 3, 2u);  ; 2 bytes per element = float16

    DATA ACCESS:
        float* ptr = m;             ; direct pointer to channel 0, row 0
        float* row0 = m.row(0);     ; pointer to first row of channel 0
        ncnn::Mat ch1 = m.channel(1); ; slice of channel 1 (shares memory)

    PIXEL CONVERSION (built-in, no OpenCV needed):
        ncnn::Mat::from_pixels(pixel_ptr, ncnn::Mat::PIXEL_BGR, w, h);
        ncnn::Mat::from_pixels_resize(ptr, PIXEL_BGR2RGB, sw, sh, dw, dh);
        ncnn::Mat::from_pixels_roi(ptr, PIXEL_BGR, w, h, rx, ry, rw, rh);
        ; PIXEL_ formats: BGR, RGB, GRAY, BGRA, RGBA, BGR2RGB, BGR2GRAY, ...

    MEAN/STD NORMALISATION:
        const float mean[3] = {104, 117, 123};
        const float std[3]  = {1.0f/58, 1.0f/57, 1.0f/57};
        in_mat.substract_mean_normalize(mean, std);
        ; Applied in-place: out[c][h][w] = (in[c][h][w] - mean[c]) * std[c]
        ; This is the canonical way — no numpy required on device.

### Python API (pyncnn)

    pyncnn provides Python bindings for NCNN's C++ API.

    INSTALLATION:
        pip install ncnn        ; prebuilt wheels for x86/ARM64 Linux/macOS
        # OR build from source:
        # cmake -DNCNN_PYTHON=ON ..

    BASIC INFERENCE:
        import ncnn
        import numpy as np

        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.load_param("model.param")
        net.load_model("model.bin")

        # Create numpy input (CHW float32, RGB, normalised)
        input_np = np.random.rand(3, 224, 224).astype(np.float32)

        with net.create_extractor() as ex:
            # Convert numpy to ncnn.Mat (CHW → ncnn internal PACK4)
            in_mat = ncnn.Mat.from_pixels(
                (input_np * 255).astype(np.uint8).transpose(1,2,0).tobytes(),
                ncnn.Mat.PixelType.PIXEL_RGB, 224, 224)
            # OR directly from numpy float32:
            in_mat = ncnn.Mat(input_np.shape[2],  ; w
                               input_np.shape[1],  ; h
                               input_np.shape[0],  ; c
                               input_np.ctypes.data_as(ncnn.ctypes.c_void_p))

            ex.input("input", in_mat)
            _, out_mat = ex.extract("output")

        out_np = np.array(out_mat)   ; converts ncnn.Mat to numpy array

    USING EXTRACTOR AS CONTEXT MANAGER:
        with net.create_extractor() as ex:
            ex.input("data", mat)
            _, result = ex.extract("prob")
        ; Context manager ensures the extractor is released after the block.
        ; Each inference call should use a fresh extractor (or one per thread).

### Custom Layer Registration

    For ops not in NCNN's built-in set:

    C++ CUSTOM LAYER:
        class MyCustomLayer : public ncnn::Layer {
        public:
            MyCustomLayer() {
                one_blob_only = true;     ; single input and output
            }

            int load_param(const ncnn::ParamDict& pd) override {
                my_param = pd.get(0, 0);   ; key=0, default=0
                return 0;
            }

            int forward(const ncnn::Mat& bottom_blob,
                         ncnn::Mat& top_blob,
                         const ncnn::Option& opt) const override {
                top_blob.create_like(bottom_blob);
                ; ... custom op implementation ...
                return 0;
            }

        private:
            int my_param;
        };

        NCNN_REGISTER_LAYER(MyCustomLayer)

    IN .PARAM:
        ; The layer type name matches the registered name:
        MyCustomLayer  my_layer  1 1  data output  0=42
        ;               name    in out inputs outputs  params

    PYTHON CUSTOM LAYER:
        import ncnn

        class MyCustomLayer(ncnn.Layer):
            def forward_with_extractor(self, bottom_blobs, top_blobs, opt):
                top_blobs[0] = bottom_blobs[0].clone()
                ; custom operation on ncnn.Mat objects
                return 0

        net.register_custom_layer("MyCustomLayer", MyCustomLayer)


##### PART 9 — BUILDING NCNN: CMAKE, ANDROID NDK, AND IOS

### Desktop Build (Linux / macOS / Windows)

    MINIMAL BUILD (CPU only, no Vulkan, no Python):
        git clone https://github.com/Tencent/ncnn.git
        cd ncnn && mkdir build && cd build
        cmake -DNCNN_DISABLE_RTTI=ON \
              -DNCNN_DISABLE_EXCEPTION=ON \
              -DNCNN_BUILD_TOOLS=ON \
              ..
        make -j8

    WITH VULKAN:
        cmake -DNCNN_VULKAN=ON \
              -DNCNN_BUILD_TOOLS=ON \
              ..

    WITH PYTHON BINDINGS:
        cmake -DNCNN_PYTHON=ON \
              -DNCNN_BUILD_TOOLS=ON \
              ..
        make -j8 && pip install .

    BUILD OUTPUTS:
        build/src/libncnn.a         ; static library (link into your app)
        build/tools/onnx2ncnn       ; ONNX converter
        build/tools/ncnnoptimize    ; optimiser/FP16 compressor
        build/tools/ncnn2table      ; INT8 calibration
        build/tools/ncnn2int8       ; INT8 quantiser

### Android NDK Build

    ANDROID_NDK=/path/to/ndk

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_PLATFORM=android-21 \
        -DNCNN_VULKAN=ON \
        -DNCNN_DISABLE_RTTI=ON \
        -DNCNN_DISABLE_EXCEPTION=ON \
        ..
    make -j8

    ABI OPTIONS:
        "armeabi-v7a":  ARMv7 with NEON (older 32-bit devices, still common)
        "arm64-v8a":    ARMv8 64-bit (all modern phones)
        "x86":          Android emulator
        "x86_64":       x86 64-bit emulator

    FOR MAXIMUM BINARY SIZE MINIMISATION:
        -DNCNN_DISABLE_RTTI=ON        ; disables RTTI (~50 KB savings)
        -DNCNN_DISABLE_EXCEPTION=ON   ; disables C++ exceptions (~30 KB savings)
        -DNCNN_TARGET_ARCH=arm         ; arm-specific optimisations
        After: strip the .so:
            $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip \
                --strip-all libncnn.a

    INTEGRATING INTO AN ANDROID PROJECT:
        Add to app/src/main/CMakeLists.txt:
            add_library(ncnn STATIC IMPORTED)
            set_target_properties(ncnn PROPERTIES
                IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a)
            target_link_libraries(${CMAKE_PROJECT_NAME} ncnn z log)

### iOS Build

    # iOS device (arm64):
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake \
        -DPLATFORM=OS64 \
        -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_BUILD_TOOLS=OFF \
        -DNCNN_VULKAN=OFF \
        ..   ; Vulkan not available on iOS
    make -j8

    FOR XCFRAMEWORK (universal iOS + iOS Simulator):
        Build separately for device (arm64) and simulator (x86_64/arm64).
        Use xcodebuild to package as XCFramework.
        Drag XCFramework into Xcode project.

    IOS OBJECTIVE-C:
        #import "ncnn/net.h"
        ncnn::Net net;
        net.load_param_bin([bundlePath cStringUsingEncoding:NSUTF8StringEncoding]);
        ; On iOS, load from app bundle, not file system.

    MEMORY MAPPED ON IOS:
        iOS file system supports mmap.
        net.opt.use_memory_mmap = true;
        net.load_model("model.bin");  ; memory-mapped, pages loaded on demand.
        Best for large models (> 20 MB): reduces startup memory spike.


##### PART 10 — PERFORMANCE BENCHMARKING AND OPTIMISATION

### ncnn_bench: The Built-in Benchmark Tool

    NCNN ships a benchmarking tool that reports per-op timing:

        ./benchncnn 10 4 0 -1 224 224 3  ; 10 runs, 4 threads, CPU, id=-1
        ./benchncnn 10 4 1  0 224 224 3  ; 10 runs, 4 threads, Vulkan GPU 0

    OUTPUT FORMAT:
        squeezenet      min =    8.35  max =    9.02  avg =    8.61
        mobilenet       min =   14.21  max =   15.33  avg =   14.88
        mobilenet_v2    min =   12.55  max =   13.01  avg =   12.78
        resnet18        min =   34.21  max =   35.44  avg =   34.87
        resnet50        min =   67.33  max =   69.12  avg =   68.02
        [per-op breakdown when --detail=1]

    NCNN BENCHMARK RESULTS (representative, ARM Cortex-A76 4 threads):
        MobileNetV2 (224×224):  12.8 ms    vs TFLite: 17.2 ms  (1.34× faster)
        ResNet-50:               68 ms      vs TFLite: 94 ms    (1.38× faster)
        YOLOv5-nano:             25 ms
        SqueezeNet 1.1:           9 ms

    WITH VULKAN (Adreno 650):
        MobileNetV2: 6.1 ms  (vs CPU 4T: 12.8 ms, 2.1× faster)
        ResNet-50:   22 ms   (vs CPU 4T: 68 ms,   3.1× faster)

### ncnn::Option: Fine-Grained Performance Control

    NCNN's option struct controls every performance-relevant aspect:

        ncnn::Option opt;

        ; THREADING:
        opt.num_threads = 4;          ; number of OpenMP threads (0=auto)

        ; PACK LAYOUT (critical for performance):
        opt.use_packing_layout = true; ; PACK4 for NEON, PACK8 for AVX (default ON)
        opt.use_shader_pack8   = true; ; PACK8 for Vulkan shaders (larger GPU registers)

        ; FLOAT16:
        opt.use_fp16_packed    = true; ; PACK4 with FP16 elements (ARM FP16)
        opt.use_fp16_storage   = true; ; store weights in FP16 (2× smaller)
        opt.use_fp16_arithmetic= true; ; compute in FP16 (requires ARMv8.2 fp16)

        ; BF16:
        opt.use_bf16_storage   = false; ; store in BF16 (BF16 hardware on some ARM)

        ; INT8:
        opt.use_int8_inference = true; ; enable INT8 kernels (if model is INT8)

        ; MEMORY:
        opt.blob_allocator    = &my_allocator;  ; custom memory allocator
        opt.workspace_allocator = &my_workspace; ; custom scratch allocator

        ; VULKAN:
        opt.use_vulkan_compute = true; ; GPU via Vulkan (if built with VULKAN)

        net.opt = opt;

### The use_fp16_storage Option

    FP16 weight storage halves the model's memory footprint:
        net.opt.use_fp16_storage = true;
    Effect:
        Weights read from .bin as float32, then stored as float16 internally.
        Computation still in float32 (unless use_fp16_arithmetic=true).
        Memory bandwidth: 2× less when reading weights (NEON can read 8 FP16 values vs 4 FP32).
        Typical speedup: 5–15% on bandwidth-limited models.

    use_fp16_arithmetic (ARMv8.2+ required):
        Activations also in FP16.
        Convolution accumulation in FP16 (faster but less accurate).
        Accuracy: typically within 0.2% of FP32.
        Use with caution for models sensitive to numerical precision.

### Profiling with ncnn::NCNN_TRACE

    In debug builds, NCNN can output per-layer timing:

        NCNN_TRACE=1 ./your_app
        ; Output: layer_name  forward_time_ms

    In code:
        net.opt.layer_time_measurement = true;
        ; After run:
        std::vector<double>& times = ex.get_per_layer_times();
        for (int i = 0; i < net.layers().size(); i++) {
            printf("%s: %.2f ms\n", net.layer(i)->name.c_str(), times[i]);
        }

    TYPICAL BOTTLENECKS IN MOBILE MODELS:
        Conv (3×3, stride=1): dominant for ResNet architectures.
        DepthwiseConv: smaller but numerous; many dispatch calls.
        InnerProduct (FC layers): memory-bandwidth-limited.
        LSTM/RNN: sequential, cannot parallelise over time steps.
        Softmax: trivial but present in every classifier.


##### PART 11 — NCNN IN THE CONNECTED COMPILER STACK

### NCNN and LLVM (Module 01)

    NCNN's C++ code is compiled by LLVM (via Android NDK Clang or iOS Clang).
    NCNN's hand-written ARM NEON assembly uses ARMv8 intrinsics that LLVM
    does not reorder (assembly is emitted as-is, bypassing the optimiser).
    NCNN does NOT use LLVM's auto-vectoriser for hot paths — it writes
    assembly manually to guarantee the exact instruction sequence.

### NCNN and TVM (Module 08)

    TVM and NCNN are direct competitors on ARM CPU inference.
    TVM's MetaSchedule finds optimal GEMM tile sizes for each input shape.
    NCNN's PACK4+Winograd+hand-written assembly beats TVM's auto-scheduled
    output for the specific 3×3, stride-1, PACK4 shapes common in MobileNet.
    TVM wins on unusual shapes and when the auto-scheduler finds very
    different tile configurations than NCNN's hand-tuned defaults.
    Neither framework integrates with the other.

### NCNN and ONNX (Module 10)

    ONNX is NCNN's recommended import format.
    The onnx2ncnn tool supports ONNX opsets 7–17.
    NCNN's op set is a subset of ONNX's — not every ONNX op has an NCNN equivalent.
    PNNX (Part 3) improves on ONNX for PyTorch models by preserving
    high-level ops that ONNX decomposes into primitives.

### NCNN and TFLite (Module 11)

    NCNN and TFLite are direct competitors:
    NCNN WINS:  ARM CPU speed (~20–40% on most vision models),
                smaller binary, zero dependencies, Vulkan GPU.
    TFLITE WINS: NNAPI integration (official Android NPU access),
                 TFLite Micro (microcontrollers), Python training ecosystem,
                 Play Services (zero app-size cost).
    TFLite models can be converted to NCNN:
        Use tflite2ncnn (in NCNN tools):
            tflite2ncnn model.tflite model.param model.bin

### NCNN and Core ML (Module 13)

    Core ML targets Apple ANE; NCNN targets cross-platform ARM.
    On iOS, NCNN and Core ML can coexist in the same app:
        Use Core ML for models requiring ANE.
        Use NCNN for cross-platform models (same .param + .bin on Android).
    NCNN does NOT have a CoreML backend delegate (unlike MNN).
    NCNN on iOS uses ARM NEON (CPU) only; no Metal, no ANE.

### NCNN and MNN (Module 15)

    NCNN and MNN are the two dominant Chinese mobile inference frameworks.
    They are complementary, not mutually exclusive:
        NCNN: use when binary size, zero deps, or Vulkan is the priority.
        MNN: use when multi-framework import, GPU OpenCL/Metal, or
              on-device training is needed.
    A cross-company comparison:
        NCNN (Tencent): games, messaging, consumer apps with tiny binary budget.
        MNN (Alibaba):  e-commerce, search, recommendation, server APIs.

### The Complete NCNN Ecosystem

    ┌──────────────────────────────────────────────────────────────────────┐
    │  SOURCE FRAMEWORKS                                                    │
    │  PyTorch(ONNX/PNNX)  TensorFlow(ONNX)  Caffe  TFLite  PaddlePaddle │
    └──────────────────────────┬───────────────────────────────────────────┘
                               │
           onnx2ncnn / caffe2ncnn / tflite2ncnn / pnnx
           + ncnnoptimize (BN fold, conv+relu fusion, FP16 compression)
           + ncnn2table (INT8 calibration) + ncnn2int8 (INT8 conversion)
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  .param (human-readable text) + .bin (raw weights)                  │
    │  FP32 / FP16 / INT8 / BF16 weights                                 │
    └──────┬───────────────────┬──────────────────────────────────────────┘
           │                   │
           ▼                   ▼
    ┌─────────────────┐   ┌──────────────────────────────────────────────┐
    │  CPU Backend    │   │   Vulkan GPU Backend                          │
    │  ARM NEON PACK4 │   │   SPIR-V compute shaders (pre-compiled)      │
    │  Winograd 3×3   │   │   PACK4 layout on GPU                        │
    │  INT8 SDOT      │   │   FP16 and FP32 precision                    │
    │  FP16/BF16      │   │   Android 7.0+ (API 24+)                    │
    │  x86 AVX PACK8  │   │   Pre-built SPIR-V (no JIT penalty)          │
    │  OpenMP threads │   └──────────────────────────────────────────────┘
    └─────────────────┘
           │
    PLATFORMS: Android (4.0+) · iOS · Linux ARM/x86 · Windows · WASM

    COMPARISON TABLE:
    ┌──────────────────────┬──────────────┬──────────────┬──────────────┐
    │ Property             │  NCNN        │  MNN         │  TFLite      │
    ├──────────────────────┼──────────────┼──────────────┼──────────────┤
    │ Binary size (CPU)    │ ~700 KB ✅   │ ~1.4 MB      │ ~900 KB      │
    │ External deps        │ Zero ✅      │ Minimal      │ FlatBuffers  │
    │ ARM CPU speed        │ Fastest ✅   │ Very fast    │ Fast         │
    │ GPU backend          │ Vulkan       │ OpenCL+Metal │ OpenCL+Vulkan│
    │ On-device training   │ No           │ YES ✅        │ No           │
    │ ONNX input           │ YES          │ YES (best)   │ Via onnx-tf  │
    │ TFLite input         │ YES          │ YES          │ Native       │
    │ Android NNAPI        │ No           │ YES          │ YES (best)   │
    │ MCU support          │ No           │ No           │ YES (Micro)  │
    │ Python API           │ pyncnn       │ pymnn        │ tflite-runtime│
    │ Format               │ .param+.bin  │ .mnn         │ .tflite      │
    │ Readable format      │ YES ✅ (.param)│ No (binary) │ No (binary) │
    │ Open source          │ YES (BSD)    │ YES (Apache) │ YES (Apache) │
    └──────────────────────┴──────────────┴──────────────┴──────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Model Conversion — ONNX to .param/.bin via onnx2ncnn": {
        "description": (
            "Convert a PyTorch model to NCNN's .param + .bin format. "
            "Export to ONNX, run onnx2ncnn, then ncnnoptimize. "
            "Inspect the .param text file: parse layer types, connections, parameters. "
            "Show what ncnnoptimize does: BN fold and conv+relu fusion visible in .param. "
            "Measure file sizes: FP32 vs FP16-compressed .bin. "
            "Demonstrate PNNX: direct PyTorch → NCNN for transformer models."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile
import subprocess
import shutil

print("=" * 65)
print("  MODEL CONVERSION — ONNX TO .PARAM/.BIN VIA ONNX2NCNN")
print("=" * 65)
print()

try:
    import ncnn
    HAS_NCNN = True
    print(f"  ncnn {ncnn.__version__}")
except ImportError:
    HAS_NCNN = False
    print("  pyncnn not installed: pip install ncnn")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
except ImportError:
    HAS_TORCH = False
print()

def find_tool(name):
    """Find an NCNN tool in PATH or common build locations."""
    return shutil.which(name)

tools = {t: find_tool(t) for t in ["onnx2ncnn", "ncnnoptimize", "ncnn2table", "ncnn2int8"]}
print(f"  NCNN tools found:")
for name, path in tools.items():
    status = f"✅ {path}" if path else "❌ not in PATH (build from source)"
    print(f"    {name:<20s}: {status}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Export PyTorch model to ONNX, then to NCNN
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Export PyTorch → ONNX → .param + .bin")
print("━" * 65)
print()

if HAS_TORCH:
    class MobileBlock(nn.Module):
        def __init__(self, c_in, c_out, stride=1):
            super().__init__()
            self.dw = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, stride=stride,
                           padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in), nn.ReLU6(inplace=True))
            self.pw = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU6(inplace=True))
        def forward(self, x): return self.pw(self.dw(x))

    class TinyMobileNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem   = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU6(inplace=True))
            self.blocks = nn.Sequential(
                MobileBlock(16, 32, 2), MobileBlock(32, 64, 2),
                MobileBlock(64, 64))
            self.head   = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, num_classes))
        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    model   = TinyMobileNet(10).eval()
    n_params = sum(p.numel() for p in model.parameters())
    dummy   = torch.randn(1, 3, 64, 64)
    print(f"  Model: TinyMobileNet  params={n_params:,}")

    tmpdir    = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "tiny_mobilenet.onnx")

    torch.onnx.export(
        model, (dummy,), onnx_path,
        opset_version=11,        ; NCNN works best with opset 11–13
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    print(f"  ONNX exported: {os.path.getsize(onnx_path)/1024:.1f} KB  (opset=11)")
    print()

    # ── Run onnx2ncnn ────────────────────────────────────────────────────
    param_path = os.path.join(tmpdir, "model.param")
    bin_path   = os.path.join(tmpdir, "model.bin")

    if tools["onnx2ncnn"]:
        ret = subprocess.run(
            [tools["onnx2ncnn"], onnx_path, param_path, bin_path],
            capture_output=True, text=True)
        if ret.returncode == 0:
            print(f"  onnx2ncnn: ✅ success")
            print(f"    model.param: {os.path.getsize(param_path)/1024:.1f} KB")
            print(f"    model.bin:   {os.path.getsize(bin_path)/1024:.1f} KB")
        else:
            print(f"  onnx2ncnn: ❌ error")
            if ret.stderr: print(f"  {ret.stderr[:200]}")
    else:
        print("  onnx2ncnn not found — showing expected workflow:")
        print("    onnx2ncnn model.onnx model.param model.bin")

    print()

    # ── Run ncnnoptimize ─────────────────────────────────────────────────
    opt_param = os.path.join(tmpdir, "model_opt.param")
    opt_bin   = os.path.join(tmpdir, "model_opt.bin")
    fp16_param= os.path.join(tmpdir, "model_fp16.param")
    fp16_bin  = os.path.join(tmpdir, "model_fp16.bin")

    if tools["ncnnoptimize"] and os.path.exists(param_path):
        # FP32 optimise
        subprocess.run(
            [tools["ncnnoptimize"], param_path, bin_path,
             opt_param, opt_bin, "0"],
            capture_output=True)
        # FP16 weight compression
        subprocess.run(
            [tools["ncnnoptimize"], param_path, bin_path,
             fp16_param, fp16_bin, "65536"],
            capture_output=True)

        if os.path.exists(opt_bin):
            orig_kb = os.path.getsize(bin_path) / 1024
            opt_kb  = os.path.getsize(opt_bin) / 1024
            fp16_kb = os.path.getsize(fp16_bin) / 1024 if os.path.exists(fp16_bin) else 0
            print(f"  ncnnoptimize results:")
            print(f"    Original .bin:    {orig_kb:.1f} KB")
            print(f"    Optimised .bin:   {opt_kb:.1f} KB  "
                  f"(BN fold, conv+relu fuse)")
            if fp16_kb > 0:
                print(f"    FP16 .bin:        {fp16_kb:.1f} KB  "
                      f"({orig_kb/fp16_kb:.2f}× compression)")
    else:
        print("  ncnnoptimize workflow:")
        print("    ncnnoptimize model.param model.bin model_opt.param model_opt.bin 0")
        print("    ncnnoptimize model.param model.bin model_fp16.param model_fp16.bin 65536")
        print("    ; 65536 flag = FP16 weight compression")

    print()

else:
    print("  PyTorch not available.")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Inspect the .param text format
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — .param text format: anatomy and inspection")
print("━" * 65)
print()

PARAM_GUIDE = """
  .PARAM TEXT FORMAT — COMPLETE ANATOMY
  ════════════════════════════════════════════════════════════════

  A .param file looks like this (MobileNetV2 excerpt):

    7767517            ; magic number (always on line 1)
    47 42              ; layer_count=47  blob_count=42
    Input            input      0 1 input
    Convolution      conv0      1 1 input conv0 0=32 1=3 2=1 3=2 4=1 5=0 6=864
    BatchNorm        bn_conv0   1 1 conv0 bn_conv0 0=32
    Scale            scale_conv0 2 1 bn_conv0 scale/conv0:0 scale/conv0:1 scale_conv0 0=32 1=1
    ReLU6            relu6_conv0 1 1 scale_conv0 relu6_conv0 0=6
    ConvolutionDepthWise conv1_dw 1 1 relu6_conv0 conv1_dw 0=32 1=3 2=1 3=1 4=1 5=0 6=288 7=32
    ...

  FIELD BREAKDOWN (layer line):
  ─────────────────────────────────────────────────────────────────
  Layer type:     "Convolution" — must match a registered layer class.
  Layer name:     "conv0" — unique identifier for this layer.
  input count:    1 — how many input blobs.
  output count:   1 — how many output blobs.
  input blobs:    "input" — names of input blobs (reference previous outputs).
  output blobs:   "conv0" — names of new output blobs created.
  params:         "0=32 1=3 2=1 3=2 4=1 5=0 6=864" — key=value parameters.

  CONVOLUTION PARAMETERS DECODED:
  ─────────────────────────────────────────────────────────────────
  Convolution  conv0  1 1  input  conv0  0=32 1=3 2=1 3=2 4=1 5=0 6=864
  ;                                      |    |    |    |    |    |    |
  ;                                      |    |    |    |    |    |    weight_data_size
  ;                                      |    |    |    |    |    bias_term (0=no bias)
  ;                                      |    |    |    |    pad_left (1)
  ;                                      |    |    |    stride_w (2)
  ;                                      |    |    dilation_w (1)
  ;                                      |    kernel_w (3)
  ;                                      num_output (32 output channels)

  weight_data_size = in_channels × out_channels × kH × kW
                   = 3 × 32 × 3 × 3 = 864  ✓

  WHAT ncnnoptimize CHANGES IN .PARAM:
  ─────────────────────────────────────────────────────────────────
  BEFORE optimise (from onnx2ncnn):
    Convolution  conv0  ...
    BatchNorm    bn0    ...      ; PRESENT as separate layer
    Scale        scale0 ...
    ReLU         relu0  ...     ; PRESENT as separate layer

  AFTER ncnnoptimize (BN fold + relu fuse):
    Convolution  conv0  ... 9=1  ; 9=activation_type (1=relu) — FUSED
    ; BatchNorm GONE (absorbed into conv weights)
    ; Scale GONE (absorbed into conv bias)
    ; ReLU GONE (activation_type=1 tells conv to apply relu after)

  COUNTING LAYERS BEFORE AND AFTER:
  ─────────────────────────────────────────────────────────────────
  Before ncnnoptimize:  ~60 layers (many BN + Scale + ReLU layers)
  After ncnnoptimize:   ~30 layers (BN/Scale absorbed, ReLU fused)
  Inference speedup:    10–25% fewer kernel dispatch calls
"""
print(PARAM_GUIDE)

# Parse and display the .param file if available
if HAS_TORCH and os.path.exists(opt_param) if "opt_param" in dir() else False:
    with open(opt_param) as f:
        lines = f.readlines()
    print(f"  model_opt.param contents:")
    print(f"  Magic: {lines[0].strip()}")
    counts = lines[1].strip().split()
    print(f"  Layers: {counts[0]}  Blobs: {counts[1]}")
    print()
    print(f"  Layer list:")
    for line in lines[2:2+min(12, len(lines)-2)]:
        parts = line.strip().split()
        if len(parts) >= 4:
            ltype  = parts[0]
            lname  = parts[1]
            n_in   = int(parts[2])
            n_out  = int(parts[3])
            params = [p for p in parts[4+n_in+n_out:] if "=" in p]
            print(f"    {ltype:<25s}  {lname:<20s}  {' '.join(params[:3])}")
    if len(lines) > 14:
        print(f"    ... ({len(lines)-14} more layers)")
    print()
elif "param_path" in dir() and os.path.exists(param_path):
    with open(param_path) as f:
        lines = f.readlines()
    print(f"  model.param (un-optimised) contents ({len(lines)} lines):")
    for line in lines[:12]:
        print(f"    {line}", end="")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · pyncnn Runtime — Net, Extractor, and ncnn::Mat": {
        "description": (
            "Master NCNN's Python runtime API end-to-end with pyncnn. "
            "Load .param + .bin, create extractor, run inference. "
            "Show all ncnn::Mat creation paths: from_pixels, from numpy, direct. "
            "Benchmark CPU single-thread vs multi-thread vs Vulkan GPU. "
            "Demonstrate ncnn::Option: packing layout, FP16, INT8, thread count. "
            "Show thread-safe multi-extractor pattern for concurrent inference."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  PYNCNN RUNTIME — NET, EXTRACTOR, AND NCNN::MAT")
print("=" * 65)
print()

try:
    import ncnn
    HAS_NCNN = True
    print(f"  ncnn {ncnn.__version__}")
    # Check Vulkan availability
    gpu_count = ncnn.get_gpu_count() if hasattr(ncnn, "get_gpu_count") else 0
    print(f"  Vulkan GPU count: {gpu_count}")
except ImportError:
    HAS_NCNN = False
    print("  pyncnn not installed: pip install ncnn")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Inference API reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — NCNN inference API: every detail explained")
print("━" * 65)
print()

API_GUIDE = """
  NCNN PYTHON (pyncnn) INFERENCE WORKFLOW
  ════════════════════════════════════════════════════════════════

  import ncnn, numpy as np

  STEP 1 — CREATE NET AND SET OPTIONS (before load):
  ─────────────────────────────────────────────────────────────────
  net = ncnn.Net()

  ; Performance options (must set BEFORE load_param):
  net.opt.use_vulkan_compute = False     ; True for Vulkan GPU
  net.opt.num_threads        = 4         ; OpenMP threads
  net.opt.use_packing_layout = True      ; PACK4/PACK8 NEON/AVX (default ON)
  net.opt.use_fp16_storage   = False     ; store weights as FP16
  net.opt.use_fp16_arithmetic = False    ; compute in FP16 (ARMv8.2+)
  net.opt.use_bf16_storage   = False     ; BF16 weights
  net.opt.use_int8_inference = False     ; enable INT8 kernels

  STEP 2 — LOAD MODEL:
  ─────────────────────────────────────────────────────────────────
  net.load_param("model.param")  ; parses text, registers layers
  net.load_model("model.bin")    ; loads weights into memory pool
  ;
  ; load_model() sets up ALL weight blobs for ALL layers.
  ; After this: the Net is READ-ONLY (thread-safe).
  ; One Net object is shared across all inference threads.

  STEP 3 — CREATE EXTRACTOR (per inference call or per thread):
  ─────────────────────────────────────────────────────────────────
  ex = net.create_extractor()
  ; Extractor holds references to the Net's layers.
  ; Lightweight to create: just references, no memory allocation.
  ; NOT thread-safe — each thread needs its own Extractor.
  ; Use as context manager for automatic release:
  with net.create_extractor() as ex:
      ...

  STEP 4 — SET INPUT:
  ─────────────────────────────────────────────────────────────────
  ; FROM NUMPY (most common in Python):
  in_mat = ncnn.Mat(w, h, c)   ; create empty Mat
  ; Copy numpy array into Mat:
  in_mat.fill(0.0)             ; or use numpy copy methods

  ; FROM UINT8 PIXELS (camera/image, built-in preprocessing):
  in_mat = ncnn.Mat.from_pixels(pixel_bytes, ncnn.Mat.PixelType.PIXEL_RGB,
                                 img_w, img_h)
  ; Pixel types: PIXEL_RGB, PIXEL_BGR, PIXEL_RGBA, PIXEL_BGRA, PIXEL_GRAY
  ; Conversion: PIXEL_BGR2RGB, PIXEL_RGB2BGR, PIXEL_BGR2GRAY, etc.

  ; WITH RESIZE (resize + pixel type conversion in one call):
  in_mat = ncnn.Mat.from_pixels_resize(
      pixel_bytes, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
      src_w, src_h, dst_w, dst_h)   ; bilinear resize

  ; MEAN/STD NORMALISATION (in-place):
  mean_vals = [0.485*255, 0.456*255, 0.406*255]
  norm_vals = [1.0/(0.229*255), 1.0/(0.224*255), 1.0/(0.225*255)]
  in_mat.substract_mean_normalize(mean_vals, norm_vals)
  ; After this: in_mat[c][h][w] = (pixel - mean[c]) * norm[c]

  ; SET INPUT BY BLOB NAME:
  ex.input("input_blob_name", in_mat)
  ; The blob name must match a blob name in .param (Input layer output name).

  STEP 5 — EXTRACT OUTPUT:
  ─────────────────────────────────────────────────────────────────
  ret, out_mat = ex.extract("output_blob_name")
  ; ret: 0=success, -1=error (blob name not found)
  ; out_mat: ncnn.Mat with the computed result.

  STEP 6 — CONVERT OUTPUT TO NUMPY:
  ─────────────────────────────────────────────────────────────────
  out_np = np.array(out_mat)
  ; out_np shape: (c, h, w) for 3D mats, (c,) for 1D (classifier output)

  NCNN.MAT DIMENSIONS:
  ─────────────────────────────────────────────────────────────────
  mat.w   : width  (innermost dimension)
  mat.h   : height
  mat.c   : channels
  mat.d   : depth (for 4D tensors)
  len(mat): total element count (w × h × c × d)

  ; Accessing data:
  mat[0]         ; first element (flat index)
  np.array(mat)  ; copy to numpy, shape (c, h, w)
  ; For 1D classifier output (shape = [num_classes]):
  ; mat is Mat(num_classes, 1, 1) in NCNN internal form
  ; np.array(mat) returns shape (1, 1, num_classes) or (num_classes,)
"""
print(API_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Build model, export, benchmark pyncnn
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — End-to-end: export model, benchmark pyncnn vs PyTorch")
print("━" * 65)
print()

if HAS_NCNN and HAS_TORCH:
    import subprocess, shutil

    class BenchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
                nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*4*4, 10))
        def forward(self, x): return self.net(x)

    model_b = BenchNet().eval()
    dummy_b  = torch.randn(1, 3, 64, 64)
    tmpdir_b = tempfile.mkdtemp()
    onnx_b   = os.path.join(tmpdir_b, "bench.onnx")
    param_b  = os.path.join(tmpdir_b, "bench.param")
    bin_b    = os.path.join(tmpdir_b, "bench.bin")
    opt_param_b = os.path.join(tmpdir_b, "bench_opt.param")
    opt_bin_b   = os.path.join(tmpdir_b, "bench_opt.bin")

    torch.onnx.export(model_b, (dummy_b,), onnx_b, opset_version=11,
                       input_names=["input"], output_names=["output"])

    # Try to convert with onnx2ncnn + ncnnoptimize
    o2n = shutil.which("onnx2ncnn")
    opt = shutil.which("ncnnoptimize")
    has_model = False

    if o2n:
        r1 = subprocess.run([o2n, onnx_b, param_b, bin_b],
                             capture_output=True)
        if r1.returncode == 0 and opt:
            subprocess.run([opt, param_b, bin_b, opt_param_b, opt_bin_b, "0"],
                            capture_output=True)
            has_model = os.path.exists(opt_param_b)
        elif r1.returncode == 0:
            has_model = os.path.exists(param_b)
            opt_param_b, opt_bin_b = param_b, bin_b

    if has_model:
        REPS = 300
        x_np = np.random.rand(1, 3, 64, 64).astype(np.float32)

        def bench_ncnn(param, bin_path, num_threads, use_vulkan=False):
            """Benchmark NCNN inference."""
            try:
                if use_vulkan:
                    ncnn.create_gpu_instance()
                net = ncnn.Net()
                net.opt.num_threads        = num_threads
                net.opt.use_vulkan_compute = use_vulkan
                net.opt.use_packing_layout = True
                net.load_param(param)
                net.load_model(bin_path)

                # Warmup
                for _ in range(20):
                    with net.create_extractor() as ex:
                        in_m = ncnn.Mat(64, 64, 3)
                        in_m.fill(0.5)
                        ex.input("input", in_m)
                        _, _ = ex.extract("output")

                t0 = time.perf_counter()
                for _ in range(REPS):
                    with net.create_extractor() as ex:
                        in_m = ncnn.Mat(64, 64, 3)
                        in_m.fill(0.5)
                        ex.input("input", in_m)
                        ret, out = ex.extract("output")
                t_ms = (time.perf_counter() - t0) / REPS * 1000

                out_np = np.array(out)
                if use_vulkan:
                    ncnn.destroy_gpu_instance()
                return t_ms, out_np
            except Exception as e:
                return None, None

        results_b = {}
        for n_threads in [1, 4]:
            t_ms, out = bench_ncnn(opt_param_b, opt_bin_b, n_threads)
            if t_ms: results_b[f"NCNN CPU {n_threads}T"] = (t_ms, out)

        if gpu_count > 0:
            t_ms, out = bench_ncnn(opt_param_b, opt_bin_b, 1, use_vulkan=True)
            if t_ms: results_b["NCNN Vulkan"] = (t_ms, out)

        # PyTorch baseline
        with torch.no_grad():
            x_pt = torch.from_numpy(x_np)
            for _ in range(20): model_b(x_pt)
            t0 = time.perf_counter()
            for _ in range(REPS): model_b(x_pt)
            t_pt = (time.perf_counter() - t0) / REPS * 1000
        pt_out = model_b(x_pt).numpy()

        print(f"  Benchmark (BenchNet, 64×64 input, {REPS} reps):")
        print(f"  {'Backend':25s}  {'Latency (ms)':>12s}  {'vs PT 1T':>9s}  {'vs PyTorch':>10s}")
        print("  " + "-" * 62)
        print(f"  {'PyTorch eager':25s}  {t_pt:>12.3f}  {'1.00×':>9s}")
        pt_ref = None
        for name, (t_ms, out) in results_b.items():
            ratio_pt = t_pt / t_ms
            if out is not None:
                err = float(np.max(np.abs(pt_out.flatten()[:10] -
                                          out.flatten()[:10]))) if len(out.flatten()) >= 10 else 0.0
                ok  = "✅" if err < 0.1 else "⚠️"
            else:
                ok = "—"
            print(f"  {name:<25s}  {t_ms:>12.3f}  {ratio_pt:>9.2f}×  {ok}")
        print()

    else:
        print("  onnx2ncnn not found in PATH.")
        print("  Install NCNN tools: github.com/Tencent/ncnn/releases")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Thread-safety and multi-extractor pattern
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Thread-safety: multi-extractor concurrent inference")
print("━" * 65)
print()

THREAD_GUIDE = """
  NCNN THREAD SAFETY RULES
  ════════════════════════════════════════════════════════════════

  ncnn::Net:         THREAD-SAFE for READ operations (inference).
                     NOT thread-safe for writes (loading).
                     One Net is SHARED across all threads.

  ncnn::Extractor:   NOT thread-safe.
                     Each thread MUST have its own Extractor.
                     Create with net.create_extractor() (cheap operation).

  CONCURRENT INFERENCE PATTERN (Python with threading):
  ─────────────────────────────────────────────────────────────────
  import ncnn, threading, queue

  # Load model once (shared):
  net = ncnn.Net()
  net.opt.num_threads = 2   ; threads PER EXTRACTOR (not total)
  net.load_param("model.param")
  net.load_model("model.bin")

  result_queue = queue.Queue()

  def worker(task_id, input_data):
      ; Each thread creates its own Extractor:
      with net.create_extractor() as ex:
          in_mat = ncnn.Mat(...)
          ; ... fill in_mat ...
          ex.input("input", in_mat)
          _, out_mat = ex.extract("output")
          result_queue.put((task_id, np.array(out_mat)))

  ; Dispatch 8 concurrent inferences:
  threads = []
  for i, data in enumerate(batch_data):
      t = threading.Thread(target=worker, args=(i, data))
      threads.append(t)
      t.start()

  for t in threads: t.join()
  results = {}
  while not result_queue.empty():
      tid, out = result_queue.get()
      results[tid] = out

  INTRA-OP vs INTER-OP PARALLELISM:
  ─────────────────────────────────────────────────────────────────
  net.opt.num_threads = N controls INTRA-OP parallelism:
    Each conv layer uses N OpenMP threads to process output tiles.
    N=4 means each layer uses 4 threads.

  Multi-extractor INTER-OP parallelism:
    If 4 extractors run simultaneously, each with num_threads=2:
    Total threads: 4 × 2 = 8 threads.
    This saturates an octa-core CPU effectively.

  RECOMMENDED CONFIGURATION:
  ─────────────────────────────────────────────────────────────────
  Latency-critical (interactive, 1 request at a time):
      net.opt.num_threads = all_big_cores   ; 1 extractor, many threads

  Throughput-critical (batch serving, queue of requests):
      net.opt.num_threads = 2               ; multiple extractors, few threads each
      ; Run N extractors concurrently where N = total_cores / 2
"""
print(THREAD_GUIDE)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · INT8 Quantisation — Calibration Tables and ncnn2int8": {
        "description": (
            "Complete INT8 quantisation workflow for NCNN. "
            "Generate a calibration table with ncnn2table. "
            "Quantise the model with ncnn2int8. "
            "Compare FP32 vs INT8 .bin size and latency. "
            "Show how INT8 scale data is embedded in the .bin binary. "
            "Explain ARM SDOT benefits and when INT8 is or isn't faster."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile
import struct

print("=" * 65)
print("  INT8 QUANTISATION — CALIBRATION TABLES AND NCNN2INT8")
print("=" * 65)
print()

try:
    import ncnn
    HAS_NCNN = True
    print(f"  ncnn {ncnn.__version__}")
except ImportError:
    HAS_NCNN = False
    print("  pyncnn not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: INT8 quantisation pipeline reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — INT8 quantisation pipeline: all four steps")
print("━" * 65)
print()

INT8_PIPELINE = """
  NCNN INT8 QUANTISATION — FOUR-STEP PIPELINE
  ════════════════════════════════════════════════════════════════

  STEP 1: PREPARE A FLOAT32 NCNN MODEL:
  ─────────────────────────────────────────────────────────────────
  ; Start from an optimised FP32 model (ncnnoptimize already run):
  model_opt.param + model_opt.bin

  STEP 2: GENERATE THE CALIBRATION TABLE (ncnn2table):
  ─────────────────────────────────────────────────────────────────
  ; Create a text file listing calibration image paths:
  echo "/path/to/calib_img_0.jpg" > calib_list.txt
  echo "/path/to/calib_img_1.jpg" >> calib_list.txt
  ; ... 100–1000 representative images ...

  ; Run ncnn2table:
  ncnn2table \\
      --param    model_opt.param \\
      --bin      model_opt.bin   \\
      --images   calib_list.txt  \\
      --output   model.table     \\
      --mean     104,117,123     \\  ; per-channel mean (BGR order)
      --norm     1.0,1.0,1.0     \\  ; per-channel scale (=1/std if normalised)
      --size     224,224         \\  ; input image size (W,H)
      --thread   4               \\
      --method   KL               ; KL, MAX, or PERCENTILE

  ; OUTPUT: model.table (text file)
  ; Format: one line per activation tensor:
  ;   conv0  3.1415
  ;   conv1  6.2831
  ;   bn_relu1  0.9876
  ;   fc8  12.3456
  ; Each line: layer_name  max_abs_activation_across_calibration_samples

  STEP 3: QUANTISE TO INT8 (ncnn2int8):
  ─────────────────────────────────────────────────────────────────
  ncnn2int8 \\
      model_opt.param model_opt.bin \\
      model_int8.param model_int8.bin \\
      model.table

  ; What ncnn2int8 does:
  ;   Reads model.table for activation scales.
  ;   Computes weight scales: scale_w = max(|W|) / 127.
  ;   Quantises weights: W_q = round(W / scale_w) → INT8.
  ;   Embeds weight data (INT8) + scale table (FP32) in model_int8.bin.
  ;   Sets int8_scale_term in model_int8.param for eligible layers.
  ;   Outputs model_int8.param + model_int8.bin.

  STEP 4: DEPLOY (NO SPECIAL RUNTIME NEEDED):
  ─────────────────────────────────────────────────────────────────
  ; NCNN automatically detects INT8 from magic number in .bin.
  ; No runtime config changes needed:
  net = ncnn.Net()
  net.opt.use_int8_inference = True   ; optional, enabled by default
  net.load_param("model_int8.param")
  net.load_model("model_int8.bin")
  ; Inference runs with INT8 kernels for quantised layers.
  ; Unquantised layers (first conv, last FC) run in FP32.

  THE CALIBRATION TABLE FORMAT:
  ─────────────────────────────────────────────────────────────────
  ; model.table is a plain text file:
  input          3.9142
  conv0          6.2831
  relu0          4.5612
  conv1_dw       3.1415
  relu1_dw       2.7182
  conv1_pw       5.5000
  relu1_pw       4.1230
  pool1          4.1230
  ...
  fc8_output    12.3456

  ; You can EDIT the calibration table manually:
  ;   Increase a value: less aggressive quantisation for that layer.
  ;   Set to a very high value: effectively skips INT8 for that layer.
  ;   This is useful for layers that are sensitive to quantisation.

  ; Inspect the table:
  cat model.table | sort -k2 -n | tail -10  ; show layers with largest range
  ; Layers with very large ranges are candidates for FP32 fallback.

  CALIBRATION METHOD SELECTION:
  ─────────────────────────────────────────────────────────────────
  --method MAX:
    scale = max(|activations|) / 127
    Fast. Sensitive to outliers.
    Use when: calibration data has no outliers; speed matters.

  --method KL:
    scale chosen by KL-divergence minimisation over activation histogram.
    More accurate for models with activation outliers.
    Use when: accuracy is critical; model has wide activation ranges.
    ~3× slower than MAX.

  --method PERCENTILE:
    scale = percentile(|activations|, p) / 127  (default p=99.9)
    Clips outlier values at the pth percentile.
    Good balance: robust to outliers, faster than KL.
    Use when: model has occasional outlier activations.

  TYPICAL ACCURACY IMPACT (ImageNet top-1, MobileNetV2):
  ─────────────────────────────────────────────────────────────────
  FP32:                      71.9%
  INT8 (MAX, 100 images):    71.0%  (-0.9%)
  INT8 (KL, 100 images):     71.5%  (-0.4%)
  INT8 (MAX, 1000 images):   71.4%  (-0.5%)
  INT8 (KL, 1000 images):    71.7%  (-0.2%)
"""
print(INT8_PIPELINE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Inspecting the INT8 .bin format
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — INT8 .bin structure: magic numbers and scale tables")
print("━" * 65)
print()

BIN_FORMAT = """
  NCNN .BIN MAGIC NUMBERS AND WEIGHT FORMATS
  ════════════════════════════════════════════════════════════════

  Each weight blob in the .bin starts with a 4-byte magic number:

  MAGIC          HEX          DESCRIPTION
  ─────────────────────────────────────────────────────────────────
  0x00000000     \x00\x00\x00\x00   Raw float32 (no header)
  0x01306B47     \x47\x6B\x30\x01   Float32 with per-blob quantisation
  0x000D4B38     \x38\x4B\x0D\x00   Float16 compressed weights (ncnnoptimize flag=65536)
  0x0002C056     \x56\xC0\x02\x00   INT8 quantised weights

  READING THE INT8 FORMAT (magic = 0x0002C056):
  ─────────────────────────────────────────────────────────────────
  import struct, numpy as np

  def read_ncnn_int8_blob(f, n_elements, n_output_channels):
      magic = struct.unpack('<I', f.read(4))[0]
      if magic == 0x0002C056:
          ; INT8 weights: n_elements bytes
          weights_int8 = np.frombuffer(f.read(n_elements), dtype=np.int8)
          ; Scale table: n_output_channels float32 values
          scales = np.frombuffer(f.read(n_output_channels * 4), dtype=np.float32)
          ; Reconstruct float32:
          ; Reshape to [out_ch, in_ch*kH*kW]:
          W_int8 = weights_int8.reshape(n_output_channels, -1)
          W_fp32 = W_int8.astype(np.float32) * scales[:, np.newaxis]
          return W_fp32, scales
      elif magic == 0x000D4B38:
          ; FP16 weights
          return np.frombuffer(f.read(n_elements * 2), dtype=np.float16).astype(float)
      else:
          ; FP32
          return np.frombuffer(f.read(n_elements * 4), dtype=np.float32)

  WHAT int8_scale_term MEANS IN .PARAM:
  ─────────────────────────────────────────────────────────────────
  Convolution  conv0  ...  8=3
  ;                        ^ int8_scale_term:
  ;  0 = FP32 (no INT8)
  ;  1 = bottom blob (input) quantised to INT8, top blob (output) FP32
  ;  2 = top blob quantised, bottom blob FP32 (output is INT8)
  ;  3 = both bottom and top blobs quantised (full INT8 layer)

  MIXED PRECISION IN NCNN:
  ─────────────────────────────────────────────────────────────────
  Not all layers must be INT8. ncnn2int8 selects layers automatically:
    Layers that appear in model.table AND have large enough activations
    to benefit from INT8 are quantised.
    Layers with very small activation ranges may stay FP32.
    The FIRST convolution (3 input channels) often stays FP32:
        3 input channels is too few for INT8 to be beneficial.
    The LAST fully-connected layer often stays FP32:
        Classifier output values are sensitive to precision.

  CHECKING WHICH LAYERS ARE INT8 IN A MODEL:
  ─────────────────────────────────────────────────────────────────
  grep "8=" model_int8.param | head -20
  ; Lines with 8=3 are INT8 layers.
  ; Lines without "8=" or with "8=0" are FP32 layers.
"""
print(BIN_FORMAT)

# Show file size simulation
if HAS_TORCH:
    # Simulate the size impact of INT8 for a toy model
    n_params_typical = 3_400_000   ; MobileNetV2 params
    fp32_kb = n_params_typical * 4 / 1024
    fp16_kb = n_params_typical * 2 / 1024
    int8_kb  = n_params_typical * 1 / 1024
    # INT8 .bin = int8 weights + FP16 scale tables
    # Scale tables: 1 float per output channel per layer
    # For MobileNetV2: ~2000 output channels total across all layers
    scale_kb = 2000 * 4 / 1024
    int8_total_kb = int8_kb + scale_kb

    print(f"  Model size estimate (MobileNetV2, {n_params_typical:,} params):")
    print(f"  {'Format':20s}  {'Weights':>10s}  {'Total .bin':>12s}  {'vs FP32':>8s}")
    print("  " + "-" * 55)
    print(f"  {'FP32':20s}  {fp32_kb:>8.0f} KB  {fp32_kb:>10.0f} KB  1.00×")
    print(f"  {'FP16 weights':20s}  {fp16_kb:>8.0f} KB  {fp16_kb:>10.0f} KB  "
          f"{fp32_kb/fp16_kb:.2f}×")
    print(f"  {'INT8 + scales':20s}  {int8_kb:>8.0f} KB  {int8_total_kb:>10.0f} KB  "
          f"{fp32_kb/int8_total_kb:.2f}×")
    print()
    print("  Note: INT8 compression is ~3.9× (not 4×) because scale tables")
    print("  add ~8 KB overhead for MobileNetV2-scale models.")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: ARM SDOT performance analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — ARM SDOT: when INT8 is (and isn't) faster")
print("━" * 65)
print()

SDOT_NCNN = """
  NCNN INT8 SDOT PERFORMANCE ANALYSIS
  ════════════════════════════════════════════════════════════════

  HARDWARE WITH ARM SDOT (sdot instruction, ARMv8.2+):
  ─────────────────────────────────────────────────────────────────
  Cortex-A76  (Snapdragon 855, 865, 888, 8 Gen 1)
  Cortex-A77  (Exynos 990, 9820)
  Cortex-A78  (Snapdragon 8 Gen 1, Dimensity 9000)
  Cortex-X1   (Snapdragon 888 prime core)
  Cortex-X2   (Snapdragon 8 Gen 1 prime core)
  Apple M1/M2 (different instruction set but similar benefit)

  NOT AVAILABLE ON:
  Cortex-A53  (budget phones 2016–2020)
  Cortex-A55  (all modern efficiency cores: still very common)
  Cortex-A57  (older mid-range)

  Implication: on a Snapdragon 888 (1×X1 + 3×A78 + 4×A55):
    INT8 with SDOT: fast on X1 and A78 cores.
    INT8 without SDOT: may be SLOWER than FP32 on A55 cores.
    NCNN detects SDOT at runtime and uses appropriate kernels.

  ARM SDOT INSTRUCTION:
  ─────────────────────────────────────────────────────────────────
  sdot v0.4s, v1.16b, v2.16b

  Meaning:
    v1.16b: 16 × signed int8 values (the input/activation batch)
    v2.16b: 16 × signed int8 values (the weight batch)
    v0.4s:  4 × int32 accumulator output

  Each SDOT computes 4 dot products of 4 int8 pairs:
    v0.s[0] += v1.b[0]*v2.b[0] + v1.b[1]*v2.b[1] + v1.b[2]*v2.b[2] + v1.b[3]*v2.b[3]
    v0.s[1] += v1.b[4]*v2.b[4] + ... (next 4 pairs)
    v0.s[2] += v1.b[8]*v2.b[8] + ...
    v0.s[3] += v1.b[12]*v2.b[12] + ...

  Throughput: 1 cycle on A76, 16 MACs per cycle.
  vs FP32 FMLA: 4 MACs per cycle.
  Ratio: 4× theoretical throughput.

  PRACTICAL SPEEDUP (measured, MobileNetV2, Cortex-A76, 4 threads):
  ─────────────────────────────────────────────────────────────────
  FP32:           12.8 ms (baseline)
  INT8 with SDOT:  7.1 ms  (1.80× faster)
  INT8 no SDOT:   14.2 ms  (1.11× SLOWER — dequant overhead without SDOT)

  Why INT8 no-SDOT is slower:
    Without SDOT: must manually expand int8 to int16/int32 before multiply.
    Extra instructions: SXTB (sign extend) for each element.
    Overhead exceeds the smaller data size benefit.

  NCNN RUNTIME DETECTION:
  ─────────────────────────────────────────────────────────────────
  import ncnn
  ; NCNN auto-detects and uses the best INT8 kernel:
  net = ncnn.Net()
  net.opt.use_int8_inference = True
  ; If SDOT is available: uses sdot-based INT8 GEMM (fast)
  ; If no SDOT: uses expanded INT16 path (moderate speedup)
  ; Switching behavior is automatic, no user action needed.

  RECOMMENDATION:
  ─────────────────────────────────────────────────────────────────
  ALWAYS run INT8 quantisation if:
    Target device has SDOT (Cortex-A76+ = flagship phones after 2019).
  SKIP INT8 if:
    Target is budget/mid-range only (Cortex-A53/A55 majority).
    In that case: FP16 weight storage (ncnnoptimize flag=65536) gives
    better benefit (smaller model) without the compute overhead.
"""
print(SDOT_NCNN)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Custom Layers and the Vulkan GPU Backend": {
        "description": (
            "Implement a custom NCNN layer in Python (pyncnn custom layer API). "
            "Register the custom layer and use it in a .param model. "
            "Configure the Vulkan backend: create_gpu_instance, use_vulkan_compute. "
            "Benchmark CPU vs Vulkan on realistic models. "
            "Show Vulkan option flags: fp16_packed, shader_pack8. "
            "Show NCNN's CPU affinity API for big.LITTLE SoCs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  CUSTOM LAYERS AND THE VULKAN GPU BACKEND")
print("=" * 65)
print()

try:
    import ncnn
    HAS_NCNN = True
    print(f"  ncnn {ncnn.__version__}")
    GPU_COUNT = ncnn.get_gpu_count() if hasattr(ncnn, "get_gpu_count") else 0
    print(f"  Vulkan GPUs: {GPU_COUNT}")
except ImportError:
    HAS_NCNN = False
    GPU_COUNT = 0
    print("  pyncnn not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Custom layer API reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Custom layers: registering new ops in pyncnn")
print("━" * 65)
print()

CUSTOM_LAYER_GUIDE = """
  NCNN CUSTOM LAYER API (C++ AND PYTHON)
  ════════════════════════════════════════════════════════════════

  USE CASES:
    Ops not in NCNN's 200 built-in layers.
    Domain-specific post-processing (NMS, custom decode, etc.).
    Experimental ops (new attention variants, new normalisation).
    Ops converted via PNNX/ONNX that NCNN can't match to a built-in.

  C++ CUSTOM LAYER:
  ─────────────────────────────────────────────────────────────────
  #include "layer.h"

  class SwiGLU : public ncnn::Layer {
  public:
      SwiGLU() { one_blob_only = true; }

      int forward(const ncnn::Mat& bottom_blob,
                  ncnn::Mat& top_blob,
                  const ncnn::Option& opt) const override {
          int w = bottom_blob.w;
          int h = bottom_blob.h;
          int c = bottom_blob.c / 2;   ; split channels in half

          top_blob.create(w, h, c, 4u, opt.blob_allocator);

          for (int q = 0; q < c; q++) {
              const float* gate  = bottom_blob.channel(q);
              const float* value = bottom_blob.channel(q + c);
              float* out = top_blob.channel(q);
              for (int i = 0; i < h * w; i++) {
                  float s = 1.0f / (1.0f + expf(-gate[i]));
                  out[i]  = gate[i] * s * value[i];   ; SwiGLU
              }
          }
          return 0;
      }
  };

  NCNN_REGISTER_LAYER_CLASS(SwiGLU)

  ; In .param, use the class name:
  ; SwiGLU  swiglu_0  1 1  input  output

  PYTHON CUSTOM LAYER (pyncnn):
  ─────────────────────────────────────────────────────────────────
  import ncnn, numpy as np

  class SwiGLULayer(ncnn.Layer):
      def __init__(self):
          super().__init__()
          self.one_blob_only = True

      def forward(self, bottom_blob, top_blob, opt):
          ; bottom_blob: ncnn.Mat (input)
          ; top_blob: ncnn.Mat (output, will be created here)
          in_np = np.array(bottom_blob)
          ; in_np shape: (2*c, h, w) — split in half along channels
          c = in_np.shape[0] // 2
          gate  = in_np[:c]   ; first half of channels
          value = in_np[c:]   ; second half of channels

          sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
          out_np = gate * sigmoid_gate * value   ; SwiGLU

          ; Create output Mat from numpy:
          top_blob.w = bottom_blob.w
          top_blob.h = bottom_blob.h
          top_blob.c = c
          ; Copy data back:
          ; (pyncnn provides assign from numpy)
          return 0  ; success

  ; Register BEFORE loading .param:
  net = ncnn.Net()
  net.register_custom_layer("SwiGLU", SwiGLULayer)
  net.load_param("model_with_swiglu.param")
  net.load_model("model_with_swiglu.bin")

  LOAD_PARAM / LOAD_MODEL AFTER REGISTRATION:
  ─────────────────────────────────────────────────────────────────
  ; CRITICAL ORDER: register BEFORE load_param.
  ; When load_param() encounters "SwiGLU  ...", it looks up the
  ; registered layer class immediately. If not registered: error.

  COMMON CUSTOM LAYER USE CASES:
  ─────────────────────────────────────────────────────────────────
  SwiGLU / GeGLU:          GLU variants for modern transformers
  RMSNorm:                  used in LLaMA, replacing LayerNorm
  RotaryEmbedding (RoPE):   position embeddings for LLaMA/Mistral
  GroupQueryAttention:      multi-query / grouped query attention
  YOLOv8 postprocess:       bounding box decode + NMS
  CenterPointDetection:     custom 3D object detection head
  FaceAlignment postproc:   landmark regression decode
"""
print(CUSTOM_LAYER_GUIDE)

if HAS_NCNN:
    # Demonstrate a simple custom layer in pyncnn
    try:
        class LeakySquareLayer(ncnn.Layer):
            """Custom layer: y = x^2 if x > 0 else 0.1 * x^2"""
            def __init__(self):
                super().__init__()
                self.one_blob_only = True

            def forward_with_extractor(self, bottom_blobs, top_blobs, opt):
                in_np  = np.array(bottom_blobs[0])
                out_np = np.where(in_np > 0, in_np**2, 0.1 * in_np**2)
                # Convert back to ncnn.Mat
                top_blob = ncnn.Mat(int(out_np.shape[-1]) if out_np.ndim >= 1 else 1)
                # pyncnn custom layer demo — actual data copy may vary by version
                return 0

        print(f"  Custom layer 'LeakySquare' class defined ✅")
        print(f"  (pyncnn custom layer API: net.register_custom_layer(name, cls))")
        print()
    except Exception as e:
        print(f"  Custom layer demo: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Vulkan backend configuration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Vulkan GPU backend: configuration and tuning")
print("━" * 65)
print()

VULKAN_CONFIG = """
  VULKAN BACKEND CONFIGURATION — EVERY OPTION EXPLAINED
  ════════════════════════════════════════════════════════════════

  INITIALISATION (once per process):
  ─────────────────────────────────────────────────────────────────
  import ncnn

  ncnn.create_gpu_instance()   ; initialise Vulkan; must call before any GPU use
  gpu_count = ncnn.get_gpu_count()
  if gpu_count == 0:
      print("No Vulkan GPU available")
  else:
      ; Get GPU info:
      gpu_info = ncnn.get_gpu_info(0)   ; GPU index 0
      print(f"GPU: {gpu_info.device_name()}")
      print(f"Heap memory: {gpu_info.heap_budget_mb()} MB")
      print(f"Subgroup size: {gpu_info.subgroup_size()}")

  ; At program exit:
  ncnn.destroy_gpu_instance()

  NET CONFIGURATION FOR VULKAN:
  ─────────────────────────────────────────────────────────────────
  net = ncnn.Net()

  ; Enable Vulkan:
  net.opt.use_vulkan_compute = True   ; required

  ; FP16 on GPU (Vulkan):
  net.opt.use_fp16_packed   = True    ; PACK4 with FP16 elements (fast on GPU)
  net.opt.use_fp16_storage  = True    ; store activations in FP16 (2× less GPU memory)
  net.opt.use_fp16_arithmetic = True  ; compute in FP16 on GPU (may have precision issues)

  ; PACK8 for wider SIMD on GPU:
  net.opt.use_shader_pack8  = True    ; use 8-wide SIMD instead of 4-wide on Vulkan
                                       ; Useful for wide GEMMs (transformer models)
                                       ; May be slower for depthwise conv (too narrow)

  ; Thread settings for Vulkan do not affect GPU execution:
  net.opt.num_threads = 1             ; Vulkan ignores num_threads

  net.load_param("model.param")
  net.load_model("model.bin")

  ; Then use as normal:
  with net.create_extractor() as ex:
      ex.input("input", in_mat)
      _, out = ex.extract("output")

  VULKAN DEVICE SELECTION (multi-GPU):
  ─────────────────────────────────────────────────────────────────
  ; NCNN uses GPU 0 by default.
  ; For a specific GPU:
  net.opt.vulkan_device_index = 1   ; use GPU 1

  WHEN TO USE use_shader_pack8=True:
  ─────────────────────────────────────────────────────────────────
  Model                  use_shader_pack8  Speedup vs pack4
  ─────────────────────────────────────────────────────────────────
  MobileNetV2 (many DW)  False             N/A (DW is pack4 anyway)
  ResNet-50 (standard)   True              +10–20% on Vulkan
  Transformer (wide FC)  True              +15–30% on Vulkan
  SqueezeNet             False             minimal benefit

  VULKAN PIPELINE CACHE (IMPORTANT FOR STARTUP TIME):
  ─────────────────────────────────────────────────────────────────
  ; NCNN pre-compiled SPIR-V shaders → no JIT compilation penalty.
  ; BUT: vkCreateComputePipeline still takes ~1–5ms per pipeline.
  ; For a model with 50 layers: ~50–250ms first time on Vulkan.

  ; Subsequent runs: pipeline cache kicks in (Vulkan driver caches).
  ; Android: Vulkan pipeline cache at /sdcard/Android/data/com.app/
  ; iOS: Vulkan not available (Metal only).

  ; NCNN manual pipeline cache:
  net.opt.pipeline_cache = ncnn.PipelineCache(ncnn.get_gpu_device(0))
  ; Share one PipelineCache across all nets for the same model family.
"""
print(VULKAN_CONFIG)

if HAS_NCNN and GPU_COUNT > 0:
    print(f"  Vulkan GPU detected ({GPU_COUNT} device(s)).")
    print(f"  To benchmark Vulkan vs CPU, load a model with:")
    print(f"    ncnn.create_gpu_instance()")
    print(f"    net.opt.use_vulkan_compute = True")
    print(f"    net.opt.use_fp16_packed = True")
    print()
else:
    print("  No Vulkan GPU available in this environment.")
    print("  Vulkan is available on Android 7.0+ and desktop Linux/Windows.")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: CPU affinity and big.LITTLE scheduling
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — CPU affinity and big.LITTLE scheduling")
print("━" * 65)
print()

CPU_AFFINITY = """
  BIG.LITTLE CPU SCHEDULING — NCNN AFFINITY API
  ════════════════════════════════════════════════════════════════

  MODERN MOBILE SoC ARCHITECTURE:
  ─────────────────────────────────────────────────────────────────
  Snapdragon 8 Gen 2 (2022 flagship):
    1 × Cortex-X3  @ 3.2 GHz (prime core, best single-thread perf)
    2 × Cortex-A715 @ 2.8 GHz (big cores, balanced)
    2 × Cortex-A710 @ 2.8 GHz (big cores)
    3 × Cortex-A510 @ 2.0 GHz (efficiency cores, 3× slower but low power)

  For ML inference:
    LATENCY-CRITICAL: use X3 + A715 + A710 only (skip A510).
    BACKGROUND/POWER: use A510 only (6× less power per inference).
    Using all 8 cores: A510 drags down the big cores via OpenMP sync.

  NCNN CPU AFFINITY API (C++):
  ─────────────────────────────────────────────────────────────────
  #include "cpu.h"

  ncnn::set_cpu_powersave(0);  ; use ALL cores (max performance, max power)
  ncnn::set_cpu_powersave(1);  ; prefer LITTLE (efficiency) cores
  ncnn::set_cpu_powersave(2);  ; prefer BIG cores (recommended for low latency)

  ; Manual core mask:
  ncnn::CpuSet cpu_set;
  cpu_set.enable(0);  ; core 0 (X3 prime)
  cpu_set.enable(1);  ; core 1 (A715)
  cpu_set.enable(2);  ; core 2 (A715)
  cpu_set.enable(3);  ; core 3 (A710)
  ncnn::set_thread_affinity(cpu_set);

  PYTHON (pyncnn):
  ─────────────────────────────────────────────────────────────────
  import ncnn

  ; Set powersave before any inference:
  ncnn.set_cpu_powersave(2)   ; prefer big cores
  net.opt.num_threads = 4     ; will use the 4 big cores

  RECOMMENDED CONFIGURATIONS:
  ─────────────────────────────────────────────────────────────────
  Real-time camera (30fps, interactive):
    powersave=2, num_threads=4   ; big cores, 4 threads
    Expected: < 33ms per frame (30fps budget)

  Background OCR/processing (battery life priority):
    powersave=1, num_threads=2   ; efficiency cores, 2 threads
    Expected: 2–3× slower but uses ~5× less battery

  Server/batch processing (throughput priority):
    powersave=0, num_threads=all_cores
    Expected: maximum throughput, highest power draw

  DETECTING CORE LAYOUT:
  ─────────────────────────────────────────────────────────────────
  import ncnn
  print(f"Big core count:    {ncnn.get_big_cpu_count()}")
  print(f"Little core count: {ncnn.get_little_cpu_count()}")
  print(f"Total CPU count:   {ncnn.get_cpu_count()}")
  ; Uses /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq
  ; to distinguish big (high frequency) from little (low frequency) cores.
"""
print(CPU_AFFINITY)

if HAS_NCNN and hasattr(ncnn, "get_cpu_count"):
    print(f"  CPU info on this machine:")
    print(f"    Total CPUs:      {ncnn.get_cpu_count()}")
    if hasattr(ncnn, "get_big_cpu_count"):
        print(f"    Big cores:       {ncnn.get_big_cpu_count()}")
        print(f"    Little cores:    {ncnn.get_little_cpu_count()}")
    print()
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · NCNN in Production — Benchmarks, Deployment, and Connected Stack": {
        "description": (
            "NCNN production deployment: Android NDK, iOS, and binary size. "
            "Show the complete ncnn::Option configuration for production. "
            "Benchmark NCNN vs MNN vs TFLite on representative ARM hardware. "
            "Demonstrate memory-mapped weight loading for large models. "
            "Summarise NCNN's position in the connected compiler stack. "
            "Quick-reference: every key tool, API, and build flag."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  NCNN IN PRODUCTION — BENCHMARKS, DEPLOYMENT, CONNECTED STACK")
print("=" * 65)
print()

try:
    import ncnn
    HAS_NCNN = True
    print(f"  ncnn {ncnn.__version__}")
except ImportError:
    HAS_NCNN = False
    print("  pyncnn not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Android and iOS integration guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Android and iOS integration")
print("━" * 65)
print()

DEPLOY_GUIDE = """
  ANDROID INTEGRATION WITH NCNN
  ════════════════════════════════════════════════════════════════

  1. BUILD NCNN FOR ANDROID (from source):
  ─────────────────────────────────────────────────────────────────
  ANDROID_NDK=/opt/android-ndk-r25b

  mkdir build-android && cd build-android
  cmake \\
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \\
      -DANDROID_ABI=arm64-v8a \\
      -DANDROID_PLATFORM=android-21 \\
      -DNCNN_VULKAN=ON \\
      -DNCNN_DISABLE_RTTI=ON \\
      -DNCNN_DISABLE_EXCEPTION=ON \\
      ..
  make -j8

  OUTPUTS:
    build-android/src/libncnn.a       ; static lib (~700 KB arm64-v8a stripped)
    build-android/include/            ; C++ headers

  2. ANDROID PROJECT INTEGRATION (CMakeLists.txt):
  ─────────────────────────────────────────────────────────────────
  cmake_minimum_required(VERSION 3.10)
  project(MyApp)

  ; Import NCNN:
  add_library(ncnn STATIC IMPORTED)
  set_target_properties(ncnn PROPERTIES
      IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a)

  ; Import Vulkan (if using GPU):
  find_library(vulkan-lib vulkan)

  add_library(myapp SHARED myapp.cpp)
  target_link_libraries(myapp ncnn ${vulkan-lib} android log z)

  3. ANDROID C++ INFERENCE CODE:
  ─────────────────────────────────────────────────────────────────
  #include "ncnn/net.h"
  #include "ncnn/gpu.h"   ; if using Vulkan

  ; Model assets → copy to internal storage at first launch:
  ; OR use AAsset to load directly from APK assets:
  #include <android/asset_manager.h>

  int loadAsset(AAssetManager* mgr, const char* name, std::vector<char>& data) {
      AAsset* asset = AAssetManager_open(mgr, name, AASSET_MODE_BUFFER);
      if (!asset) return -1;
      data.resize(AAsset_getLength(asset));
      AAsset_read(asset, data.data(), data.size());
      AAsset_close(asset);
      return 0;
  }

  ; In your JNI function:
  ncnn::Net net;
  net.opt.use_vulkan_compute = true;
  net.opt.use_fp16_storage   = true;
  net.opt.num_threads        = 4;

  std::vector<char> param_data, model_data;
  loadAsset(asset_mgr, "model.param", param_data);
  loadAsset(asset_mgr, "model.bin",   model_data);

  net.load_param_mem(param_data.data());
  net.load_model_mem(model_data.data());   ; load from memory (no file system)

  ; For memory-mapped loading (file must be on device storage):
  net.opt.use_memory_mmap = true;
  net.load_model("path/to/model.bin");   ; pages loaded on demand

  4. BINARY SIZE BREAKDOWN (arm64-v8a, stripped):
  ─────────────────────────────────────────────────────────────────
  Component              Size
  ─────────────────────────────────────────────────────────────────
  libncnn.a (CPU only)   ~700 KB
  Vulkan shaders         +200 KB (pre-compiled SPIR-V in binary)
  Total (CPU+Vulkan)     ~900 KB

  COMPARISON:
  libncnn.a (CPU only):  ~700 KB
  libMNN.so (CPU only):  ~1.4 MB  (2×)
  libonnxruntime.so:     ~8–15 MB (11–21×)
  libtorch.so:           ~40 MB   (57×)

  iOS INTEGRATION:
  ════════════════════════════════════════════════════════════════
  ; Build for iOS (requires macOS with Xcode):
  cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake \\
        -DPLATFORM=OS64 \\
        -DNCNN_BUILD_TOOLS=OFF \\
        ..
  make -j8

  ; Add libncnn.a to Xcode project → Build Phases → Link Binary with Libraries
  ; Add ncnn/include/ to Header Search Paths

  OBJECTIVE-C:
    NSString* paramPath = [[NSBundle mainBundle]
                           pathForResource:@"model" ofType:@"param"];
    NSString* binPath   = [[NSBundle mainBundle]
                           pathForResource:@"model" ofType:@"bin"];
    ncnn::Net net;
    net.load_param([paramPath UTF8String]);
    net.load_model([binPath   UTF8String]);

  WEBASSEMBLY:
  ════════════════════════════════════════════════════════════════
  emcmake cmake -DNCNN_BUILD_TESTS=OFF ..
  make -j8
  ; Output: ncnn.js + ncnn.wasm
  ; WASM binary: ~1.5 MB gzipped
  ; Performance: ~5–8× slower than native ARM64
"""
print(DEPLOY_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Representative benchmark table
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Benchmark: NCNN vs MNN vs TFLite on ARM")
print("━" * 65)
print()

BENCHMARK_TABLE = """
  INFERENCE LATENCY COMPARISON (milliseconds, lower is better)
  ════════════════════════════════════════════════════════════════

  DEVICE: Pixel 7 (Tensor G2, Cortex-X1 @ 2.85 GHz + 3×A710 + 4×A510)
  TEST: 4 threads, big cores, float32, batch=1

  Model              NCNN CPU   MNN CPU   TFLite    NCNN Vulkan  Notes
  ─────────────────────────────────────────────────────────────────────
  MobileNetV2 224    12.1 ms    14.8 ms   17.2 ms   6.2 ms       A710, NEON
  MobileNetV3-S 224   8.7 ms    10.2 ms   13.1 ms   4.8 ms
  SqueezeNet 1.1      8.2 ms    10.1 ms   11.4 ms   4.1 ms
  ResNet-18 224      31.4 ms    38.2 ms   44.8 ms   9.8 ms
  ResNet-50 224      65.2 ms    79.5 ms   94.1 ms   18.3 ms
  YOLOv5n 640        28.5 ms    35.1 ms   41.2 ms   12.1 ms      300 GFLOPS
  EfficientDet-D0    35.2 ms    43.8 ms   52.1 ms   15.4 ms

  DEVICE: Snapdragon 888 (X1 @ 2.84 GHz + 3×A78 + 4×A55)
  (INT8 with SDOT on X1 and A78 cores)

  Model              NCNN INT8  NCNN FP32  Speedup   Adreno 660  Notes
  ─────────────────────────────────────────────────────────────────────
  MobileNetV2 224     7.1 ms    12.8 ms    1.80×      5.9 ms     SDOT
  ResNet-50 224      34.2 ms    65.2 ms    1.91×     16.8 ms     SDOT
  BERT-Tiny (seq=64) 22.4 ms    51.2 ms    2.29×      N/A        SDOT GEMM
  YOLOv5n 640        15.2 ms    28.5 ms    1.88×     10.2 ms

  WHY NCNN IS FASTER THAN TFLITE ON ARM CPU:
  ─────────────────────────────────────────────────────────────────
  1. PACK4 pre-layout at conversion time vs TFLite's lazy im2col.
  2. Winograd F(4×4, 3×3) and F(6×6, 3×3) for stride-1 3×3 conv.
  3. Hand-written NEON assembly for the 20 hottest ops.
  4. BatchNorm folded and ReLU fused at ncnnoptimize time.
  5. Zero overhead memory allocator (no malloc during inference).
  6. OpenMP thread pool persistent across inference calls.

  WHY VULKAN IS FASTER THAN CPU (NCNN):
  ─────────────────────────────────────────────────────────────────
  1. Adreno 660: ~1.4 TFLOPS FP16 vs ~0.05 TFLOPS CPU (28×).
  2. GPU occupancy: many output channels run in parallel.
  3. FP16 on GPU: 2× throughput vs FP32 with use_fp16_packed=True.
  4. Pre-compiled SPIR-V: no JIT penalty on subsequent runs.

  WHERE NCNN LAGS vs TFLite/MNN:
  ─────────────────────────────────────────────────────────────────
  NNAPI (Android NPU): TFLite has mature NNAPI support.
    NCNN does NOT support NNAPI. On devices with a fast NPU
    (Google Tensor, Qualcomm HTP): TFLite+NNAPI may beat NCNN.
  Metal (iOS): NCNN uses ARM NEON (CPU) on iOS. MNN uses Metal GPU.
    For iOS GPU inference: MNN Metal > NCNN CPU.
  On-device training: NCNN has no training path.
    For federated learning: MNN > NCNN.
"""
print(BENCHMARK_TABLE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack and quick reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — NCNN in the connected compiler stack")
print("━" * 65)
print()

STACK = """
  NCNN IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                      │
  │  NCNN compiled by Android NDK Clang (LLVM-based).                   │
  │  NCNN's NEON assembly bypasses LLVM's vectoriser intentionally.      │
  │  NCNN controls every instruction in the hot path.                   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                       │
  │  Direct competitor on ARM CPU inference.                             │
  │  NCNN wins for standard MobileNet-style 3×3 conv shapes.            │
  │  TVM's MetaSchedule wins for unusual shapes and custom hardware.    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 10: ONNX                                                      │
  │  ONNX is NCNN's primary import format via onnx2ncnn.                │
  │  PNNX improves on ONNX for PyTorch → NCNN (preserves high-level ops)│
  │  NCNN supports ONNX opsets 7–17 (best coverage at opset 11–13).    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 11: TFLite                                                    │
  │  Direct competitor: same hardware, similar use cases.               │
  │  NCNN: smaller, faster on CPU, Vulkan GPU.                         │
  │  TFLite: NNAPI NPU, MCU (Micro), Play Services (zero app size).    │
  │  tflite2ncnn converts TFLite to NCNN format.                       │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 13: Core ML                                                   │
  │  Core ML targets Apple ANE; NCNN targets ARM NEON (CPU).            │
  │  NCNN runs on iOS via ARM NEON; cannot access ANE.                  │
  │  On iOS: use Core ML for maximum performance, NCNN for cross-platform│
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 15: MNN                                                       │
  │  NCNN and MNN are the two dominant Chinese mobile inference engines. │
  │  NCNN: zero deps, smaller, Vulkan-based GPU.                        │
  │  MNN: multi-framework, on-device training, OpenCL/Metal GPU.        │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 16: NCNN (THIS MODULE)                                        │
  │  .param (human-readable) + .bin (raw binary) format                 │
  │  onnx2ncnn + ncnnoptimize (BN fold, relu fuse, FP16)               │
  │  ARM NEON PACK4, Winograd 4×4/6×6, INT8 SDOT                       │
  │  Vulkan GPU: pre-compiled SPIR-V, explicit sync, PACK4 on GPU      │
  │  INT8 calibration tables (MAX/KL/Percentile) via ncnn2table         │
  │  Zero external dependencies, < 700 KB binary                        │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Task                        │ API / Tool                          │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Convert ONNX → NCNN         │ onnx2ncnn m.onnx m.param m.bin     │")
print("  │ Convert Caffe → NCNN        │ caffe2ncnn deploy.prototxt model.caffemodel │")
print("  │ Convert TFLite → NCNN       │ tflite2ncnn m.tflite m.param m.bin │")
print("  │ Direct PyTorch → NCNN       │ pnnx model.pt [input shapes]       │")
print("  │ Optimise (BN fold, fuse)    │ ncnnoptimize m.param m.bin opt.param opt.bin 0 │")
print("  │ FP16 weight compression     │ ncnnoptimize ... 65536 (flag)       │")
print("  │ INT8 calibration table      │ ncnn2table --method KL ...          │")
print("  │ INT8 quantise               │ ncnn2int8 m.param m.bin i8.param i8.bin table │")
print("  │ Load model (Python)         │ net.load_param() + net.load_model() │")
print("  │ Set CPU threads             │ net.opt.num_threads = 4             │")
print("  │ Enable PACK4                │ net.opt.use_packing_layout = True   │")
print("  │ Enable FP16 storage         │ net.opt.use_fp16_storage = True     │")
print("  │ Enable FP16 compute         │ net.opt.use_fp16_arithmetic = True  │")
print("  │ Enable INT8                 │ net.opt.use_int8_inference = True   │")
print("  │ Enable Vulkan               │ net.opt.use_vulkan_compute = True   │")
print("  │ Enable Vulkan FP16          │ net.opt.use_fp16_packed = True      │")
print("  │ Enable Vulkan PACK8         │ net.opt.use_shader_pack8 = True     │")
print("  │ Init Vulkan (Python)        │ ncnn.create_gpu_instance()          │")
print("  │ Query GPU count             │ ncnn.get_gpu_count()                │")
print("  │ Create inference lane       │ net.create_extractor()              │")
print("  │ Set input                   │ ex.input('blob_name', mat)          │")
print("  │ Run + get output            │ ret, out = ex.extract('blob_name')  │")
print("  │ numpy → ncnn.Mat            │ ncnn.Mat(w, h, c) + fill            │")
print("  │ pixels → ncnn.Mat           │ ncnn.Mat.from_pixels(bytes, fmt, w, h) │")
print("  │ ncnn.Mat → numpy            │ np.array(mat)                       │")
print("  │ Mean/std normalise          │ mat.substract_mean_normalize(m, n)  │")
print("  │ Register custom layer       │ net.register_custom_layer(name, cls)│")
print("  │ CPU powersave               │ ncnn.set_cpu_powersave(0/1/2)       │")
print("  │ Memory-mapped weights       │ net.opt.use_memory_mmap = True      │")
print("  │ Benchmark tool              │ benchncnn 10 4 0 -1 224 224 3       │")
print("  │ Android build               │ cmake -DANDROID_ABI=arm64-v8a ...   │")
print("  │ iOS build                   │ cmake -DPLATFORM=OS64 ...           │")
print("  │ WASM build                  │ emcmake cmake ...                   │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ GitHub                      │ github.com/Tencent/ncnn             │")
print("  │ Documentation               │ github.com/Tencent/ncnn/wiki        │")
print("  │ Pre-built Android AAR       │ github.com/nihui/ncnn-android-vulkan│")
print("  │ Releases (tools + libs)     │ github.com/Tencent/ncnn/releases    │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()

if HAS_NCNN:
    print("  Runtime verification:")
    try:
        print(f"    ncnn {ncnn.__version__} loaded ✅")
        gpu_count = ncnn.get_gpu_count() if hasattr(ncnn, "get_gpu_count") else 0
        cpu_count = ncnn.get_cpu_count() if hasattr(ncnn, "get_cpu_count") else "?"
        print(f"    CPU count: {cpu_count}")
        print(f"    Vulkan GPU count: {gpu_count}")
        if hasattr(ncnn, "get_big_cpu_count"):
            print(f"    Big cores: {ncnn.get_big_cpu_count()}")
            print(f"    Little cores: {ncnn.get_little_cpu_count()}")
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