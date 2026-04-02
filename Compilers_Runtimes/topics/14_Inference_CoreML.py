"""
Core ML — Apple's On-Device Machine Learning Runtime
======================================================

Core ML is Apple's end-to-end framework for deploying machine learning models
on Apple hardware: iPhone, iPad, Mac, Apple Watch, Apple TV, and Vision Pro.
Where ONNX Runtime is cross-platform and OpenVINO is Intel-specific, Core ML is
Apple-specific — and that restriction is its superpower. By targeting only
Apple silicon, Core ML can exploit the full vertical integration between hardware,
operating system, and software in a way no cross-platform framework can match.

The defining hardware advantage: every Apple SoC since the A11 Bionic (2017) —
and every Apple Silicon Mac since M1 (2020) — contains a dedicated NEURAL ENGINE
(ANE), a fixed-function hardware accelerator for neural network inference. The
A17 Pro delivers 35 TOPS INT8. The M4 delivers 38 TOPS. The entire Core ML
framework exists to maximise utilisation of this silicon.

But Core ML is more than an ANE dispatcher. It is a complete model lifecycle system:

    FORMAT:        .mlpackage — a directory bundle containing an MIL (Model
                   Intermediate Language) graph, compiled weights, and metadata.
                   Models are compiled to device-specific .mlmodelc bundles.

    CONVERSION:    coremltools (Python) converts any trained model — PyTorch,
                   TensorFlow, JAX, ONNX, scikit-learn — into Core ML format.
                   The conversion pipeline uses MIL as the intermediate
                   representation, running graph optimisation passes before
                   serialising to .mlpackage.

    QUANTISATION:  Float16, INT8 activations + weights, 4-bit palettization,
                   and mixed precision per-layer. All handled at conversion time
                   by coremltools.compression_utils.

    RUNTIME:       A Swift/Objective-C API. Models are loaded as MLModel objects.
                   Inference dispatches to the best available compute unit:
                   ANE (fastest, most power-efficient), GPU (Metal Performance
                   Shaders), or CPU (Accelerate/BNNS). The dispatch is automatic.

    PRIVACY:       On-device inference. Data never leaves the device.
                   No network required. No server costs.

In the connected compiler stack:
    LLVM      (module 01) ← Core ML CPU path uses Accelerate/BNNS, which Apple compiles with LLVM/Clang
    MLIR      (module 02) ← coremltools uses MLIR internally for MIL lowering passes
    XLA       (module 05) ← JAX models reach Core ML via ONNX or coremltools.convert()
    TVM       (module 08) ← TVM can target Metal (Apple GPU); Core ML targets ANE which TVM cannot
    ONNX      (module 10) ← ONNX is a primary input format for coremltools.convert()
    TFLite    (module 11) ← TFLite models can be converted to Core ML via coremltools
    OpenVINO  (module 12) ← OpenVINO targets Intel hardware; Core ML targets Apple; no overlap
    Core ML   (this)      ← Apple's on-device inference runtime for the entire Apple ecosystem

"""

import textwrap
import re

TOPIC_NAME   = "Core ML — Apple's On-Device Machine Learning Runtime"
DISPLAY_NAME = "14 · Core ML"
ICON         = "🍎"
SUBTITLE     = "MIL Format, ANE Dispatch, coremltools Conversion, Quantisation, and Apple Intelligence"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY CORE ML EXISTS: THE APPLE HARDWARE ADVANTAGE

### Apple's Vertical Integration Advantage

    Apple designs both the hardware and the software that runs on it. This
    vertical integration — unavailable to any cross-platform framework —
    creates unique opportunities for ML performance that Core ML exploits:

    UNIFIED MEMORY ARCHITECTURE:
        On traditional systems, CPU and GPU have separate memory pools.
        Moving data from CPU to GPU for inference requires an explicit copy:
            numpy array (CPU RAM) → cudaMemcpy → GPU VRAM → kernel → GPU VRAM
                                 → cudaMemcpy back → CPU RAM
        On Apple Silicon:
            CPU, GPU, ANE, and the ISP all access the SAME physical memory.
            There is no copy overhead between compute units.
            A camera frame captured by the ISP can be fed directly to the
            ANE without copying to CPU first.
        For ML inference: this eliminates the largest latency source
        (memory copies) in every pipeline that mixes CPU pre/postprocessing
        with GPU/ANE model execution.

    THE NEURAL ENGINE (ANE) — SILICON PURPOSE-BUILT FOR ML:
        The Apple Neural Engine is a dedicated hardware accelerator first
        introduced in the A11 Bionic chip (iPhone X, 2017). It is not a
        general-purpose compute engine — it cannot run arbitrary code.
        Instead, it executes a fixed instruction set of neural network
        operations: matrix multiply, convolution, element-wise ops,
        normalisation — with extreme energy efficiency.

        ANE generations and throughput:
            A11 Bionic (2017):  0.6 TOPS INT8  (2-core ANE)
            A12 Bionic (2018):  5 TOPS INT8    (8-core ANE, 9× improvement)
            A14 Bionic (2020):  11 TOPS INT8   (16-core ANE)
            A15 Bionic (2021):  15.8 TOPS INT8
            A16 Bionic (2022):  17 TOPS INT8
            A17 Pro    (2023):  35 TOPS INT8   (3-nanometre process)
            A18 Pro    (2024):  38 TOPS INT8   (used in iPhone 16 Pro)
            M1         (2020):  11 TOPS INT8
            M2         (2022):  15.8 TOPS INT8
            M3         (2023):  18 TOPS INT8
            M4         (2024):  38 TOPS INT8   (same ANE gen as A18 Pro)
            M4 Ultra   (2025):  ~76 TOPS (2× M4)

        The ANE runs at a fraction of the power of the GPU:
            Inference at 5W total SoC power (iPhone background)
            vs ~40W GPU inference on a discrete GPU

        POWER EFFICIENCY comparison (images/second per watt):
            NVIDIA RTX 4090:  ~1,200 img/s/W   (ResNet-50, INT8, 450W card)
            A17 Pro ANE:      ~7,000 img/s/W   (same model, ~2W ANE power)
        This 6× efficiency gap is why every iOS app that needs ML uses
        Core ML + ANE rather than shipping a server.

    PRIVACY AS A DESIGN CONSTRAINT:
        iOS App Store guidelines and Apple's own policies strongly prefer
        on-device inference. Cloud-based ML inference requires:
            Network connectivity (fails offline)
            Server costs (scale linearly with users)
            Data leaving the device (privacy implications, regulatory risk)
            Latency from round-trip (100–500 ms for single requests)
        Core ML enables all of this on-device:
            Works offline
            Zero marginal server cost
            User data never transmitted
            < 10 ms latency for most models

### What Core ML Is

    Core ML was introduced at WWDC 2017. It has three components:

    CORE ML FRAMEWORK:
        The runtime — loads models, dispatches to hardware, returns results.
        Available on: iOS 11+, macOS 10.13+, watchOS 4+, tvOS 11+,
                      visionOS 1.0+.
        Language bindings: Swift (primary), Objective-C.
        No Python runtime on-device — Python is used only for conversion.

    COREMLTOOLS:
        The Python conversion library (installed on developer's machine).
        Converts PyTorch, TensorFlow, scikit-learn, ONNX → Core ML format.
        Applies quantisation and compression.
        Builds and validates .mlpackage files.
        Install: pip install coremltools

    MIL (MODEL INTERMEDIATE LANGUAGE):
        Core ML's internal graph representation.
        Used by coremltools during conversion — you can inspect and modify it.
        Every supported framework is lowered to MIL before serialisation.
        MIL is also the public API for building models from scratch.

### Core ML vs Alternatives on Apple Hardware

    CORE ML vs ONNX RUNTIME on iOS/macOS:
        ONNX Runtime has a Core ML execution provider (backend).
        This actually routes inference through Core ML under the hood.
        Core ML directly is simpler and better integrated with Xcode/SwiftUI.
        ONNX RT on Apple: use when portability across platforms is required.
        Core ML: use when targeting Apple ecosystem exclusively.

    CORE ML vs METAL PERFORMANCE SHADERS (MPS / MPSGraph):
        MPS/MPSGraph is Apple's GPU compute framework (lower level).
        Core ML automatically uses MPS for GPU execution under the hood.
        MPS directly: gives more control over GPU kernels, useful for
                      custom ops not in Core ML's op set.
        Core ML: higher level, handles device dispatch automatically.

    CORE ML vs PYTORCH MOBILE / EXECUTORCH:
        PyTorch Mobile/ExecuTorch runs PyTorch models on device.
        Uses XNNPACK for CPU and MPS for GPU.
        CANNOT use the ANE (requires Core ML for ANE access).
        Core ML: required for ANE utilisation on Apple hardware.


##### PART 2 — THE CORE ML FORMAT: .MLPACKAGE AND .MLMODELC

### The .mlpackage Bundle

    Starting with coremltools 6.0 and Xcode 14, the primary Core ML format
    is the .mlpackage — a directory bundle (not a flat file):

        MyModel.mlpackage/
            manifest.json           ; bundle metadata, format version
            Data/
                com.apple.CoreML/
                    model.mlmodel   ; the serialised MIL graph (protobuf)
                    weights/
                        weight.bin  ; quantised weight data
                        ...

    The older .mlmodel format is a flat protobuf file (still supported).
    .mlpackage advantages:
        Weights stored separately — enables partial loading, delta updates.
        Can contain multiple precision variants (FP16 + INT4 side by side).
        Supports "preview" packages for development without compilation.
        Allows streaming weight download for large models.

### The Compilation Step: .mlmodelc

    When you add a .mlpackage to an Xcode project, Xcode compiles it at
    build time into a .mlmodelc bundle (ML Model Compiled):

        MyModel.mlmodelc/
            coremldata.bin          ; compiled model binary (device-specific)
            metadata.json
            analytics/
                ...

    The .mlmodelc contains a device-optimised binary, including:
        Tiled and vectorised kernel code for the CPU (Accelerate/BNNS).
        Pre-compiled Metal shaders for the GPU (mtllib format).
        A compiled ANE dataflow graph (the format is opaque/proprietary).
        A dispatch plan: which layers run on which compute unit.

    COMPILATION TIMING:
        BUILD TIME (Xcode):    when the app is built for a specific target.
        FIRST LAUNCH:          if the app ships the .mlpackage (not .mlmodelc),
                               compilation happens on first use (latency spike).
        MLModel.compileModel(at: URL): explicit compilation at runtime.

    DEVICE-SPECIFIC COMPILATION:
        The .mlmodelc is NOT portable between device families.
        A model compiled for iPhone 15 (A16) cannot run on M2 Mac.
        The Xcode build system handles this: it compiles per target in the
        app bundle.
        For OTA (over the air) model updates: ship .mlpackage and compile
        on-device at first use. The compiled .mlmodelc is cached in the
        app container.

### Model Specification: The Protobuf Schema

    The model.mlmodel file inside .mlpackage is a serialised Protocol Buffer.
    The schema is defined in CoreML.proto (available on GitHub: apple/coremltools).

    Top-level structure:
        message Model {
            int32 specificationVersion = 1;   ; spec version (7 for Core ML 7)
            ModelDescription description = 2;  ; input/output descriptions
            oneof Type {                       ; which model type this is
                NeuralNetwork neuralNetwork = 100;     ; legacy (pre-MIL)
                MLProgram mlProgram     = 200;         ; MIL-based (modern)
                Pipeline pipeline       = 300;         ; chained models
                // ... other types
            }
        }

    Model types:
        NeuralNetwork:   the original Core ML format. A sequence of layers
                         (ConvolutionLayerParams, InnerProductLayerParams, etc.)
        MLProgram:       MIL-based model. Functions and blocks of MIL ops.
                         The modern, preferred format.
        Pipeline:        multiple models chained sequentially.
        ItemSimilarity, TextClassifier, etc.: task-specific types.

    SPECIFICATION VERSIONS:
        Spec 1 (Core ML 1, iOS 11):     NeuralNetwork, basic ops
        Spec 2 (Core ML 2, iOS 12):     quantised weights, sequence models
        Spec 3 (Core ML 3, iOS 13):     updatable models, linked models
        Spec 4 (Core ML 4, iOS 14):     MLProgram (MIL) introduced
        Spec 5 (Core ML 5, iOS 15):     control flow, custom layers
        Spec 6 (Core ML 6, iOS 16):     improved quantisation
        Spec 7 (Core ML 7, iOS 17):     int4 weights, activation quantisation
        Spec 8 (Core ML 8, iOS 18):     stateful models (KV cache for LLMs)

    MINIMUM DEPLOYMENT TARGET:
        spec_version → minimum iOS version the model requires.
        Always choose the lowest spec that includes your required ops.
        coremltools.convert(..., minimum_deployment_target=ct.target.iOS16)

### ModelDescription: Inputs and Outputs

    The ModelDescription declares the model's interface:
        inputs:   list of FeatureDescription objects (name, type, shape)
        outputs:  list of FeatureDescription objects

    FeatureType variants:
        MultiArrayType:   N-dimensional float/int/double array.
                          Most common for image models and NLP.
        ImageType:        an image (width, height, colour space).
                          Handled natively by Core ML (BGRA8/FP16 internally).
        DictionaryType:   string/int keys → double values (classifiers).
        StringType:       a string (NLP models).
        SequenceType:     for RNNs and sequence models.

    Image vs MultiArray for CNN inputs:
        ImageType:      Core ML handles pixel format conversion automatically
                        (YCbCr from camera → RGB normalised automatically).
                        Uses VNImageFeaturePrintRequest for Vision integration.
        MultiArrayType: raw numerical array; preprocessing done by the app.
        For camera-fed models: always use ImageType for zero-copy camera feed.


##### PART 3 — MIL: THE MODEL INTERMEDIATE LANGUAGE

### What MIL Is

    MIL (Model Intermediate Language) is Core ML's typed, functional
    intermediate representation introduced in Core ML 4 (iOS 14).
    It is the language that coremltools uses internally to represent
    a model during conversion, optimisation, and serialisation.

    MIL is:
        TYPED:       every value has a type (fp16, fp32, int32, bool, ...)
                     and a shape (static or with symbolic dimensions).
        FUNCTIONAL:  no mutable state during forward pass (pure functions).
                     Side effects are explicit (stateful models use special ops).
        HIERARCHICAL: operations can contain sub-blocks (for if/while/scan).
        OPSET-VERSIONED: each op has a version, matching the spec version.

    MIL serves the same role as:
        ONNX's op set — defines the vocabulary of valid operations.
        TFLite's builtin ops — what the runtime can execute.
        OpenVINO's OV IR — the graph fed to the device plugins.

    Unlike ONNX (a file format designed for interchange) or OV IR (designed
    for Intel hardware), MIL is designed for Apple's full hardware range:
    its type system, shape system, and op set reflect ANE constraints.

### MIL Operations

    MIL has approximately 200 operations grouped into namespaces:

    TENSOR CREATION:
        mb.const(val, dtype):    a constant tensor
        mb.fill(shape, value):   a tensor filled with a constant
        mb.range_1d(end, start, step): like np.arange
        mb.one_hot(indices, depth): one-hot encoding

    LINEAR ALGEBRA:
        mb.linear(x, weight, bias):     y = x @ weight.T + bias
        mb.conv(x, weight, bias, ...):  N-D convolution
        mb.conv_transpose(...):         transposed convolution
        mb.matmul(x, y, ...):           general matrix multiply
        mb.einsum(values, equation):    Einstein summation
        mb.batch_norm(...):             batch normalisation
        mb.instance_norm(...):          instance normalisation
        mb.layer_norm(x, axes, gamma, beta, eps): layer normalisation
        mb.group_norm(...):             group normalisation

    ELEMENTWISE:
        mb.add(x, y), mb.sub, mb.mul, mb.real_div, mb.pow
        mb.relu(x), mb.relu6, mb.prelu, mb.elu, mb.selu
        mb.sigmoid(x), mb.tanh(x), mb.exp(x), mb.log(x), mb.sqrt(x)
        mb.gelu(x, mode):    exact or tanh approximation (mode="EXACT"/"TANH_APPROXIMATION")
        mb.silu(x):          x * sigmoid(x)
        mb.clip(x, alpha, beta): clamp
        mb.abs(x), mb.sign(x), mb.floor(x), mb.ceil(x), mb.round(x)

    REDUCTION:
        mb.reduce_sum(x, axes, keep_dims)
        mb.reduce_mean(x, axes, keep_dims)
        mb.reduce_max(x, axes, keep_dims)
        mb.reduce_min(x, axes, keep_dims)
        mb.reduce_l2_norm(x, axes, keep_dims)
        mb.reduce_log_sum_exp(x, axes, keep_dims)

    POOLING:
        mb.avg_pool(x, kernel_sizes, strides, pad_type)
        mb.max_pool(x, kernel_sizes, strides, pad_type)
        mb.l2_pool(x, kernel_sizes, strides, pad_type)

    SHAPE / INDEXING:
        mb.reshape(x, shape):    change shape
        mb.transpose(x, perm):   permute axes
        mb.squeeze(x, axes):     remove size-1 dims
        mb.expand_dims(x, axes): add size-1 dims
        mb.concat(values, axis): concatenate tensors
        mb.split(x, axis, ...):  split along axis
        mb.slice_by_index(x, begin, end, stride): strided slice
        mb.gather(x, indices, axis): index gather
        mb.scatter(data, indices, updates, axis): scatter update
        mb.pad(x, pad, mode, constant_val): padding

    ATTENTION (Core ML 7 / Spec 7):
        mb.scaled_dot_product_attention(q, k, v, mask, scale):
            Efficient attention — can use ANE flash-attention hardware path.
        mb.sdpa_with_kv_cache(q, k, v, kv_cache, ...):
            Stateful KV-cache attention for LLM generation (Spec 8 / iOS 18+).

    CONTROL FLOW:
        mb.cond(pred, _true_fn, _false_fn):  if/else
        mb.while_loop(cond_fn, body_fn, loop_vars):  while loop
        mb.map_fn(fn, inputs, axis):  map a function over a batch dimension

### Writing MIL Directly

    coremltools exposes a Python DSL for writing MIL programs:

        import coremltools as ct
        from coremltools.converters.mil import Builder as mb

        @mb.program(input_specs=[mb.TensorSpec((1, 8), dtype=ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32)])
        def my_program(x):
            W   = mb.const(val=np.random.randn(8, 4).astype(np.float32), name="W")
            b   = mb.const(val=np.zeros(4).astype(np.float32), name="b")
            y   = mb.linear(x=x, weight=W, bias=b, name="linear")
            out = mb.relu(x=y, name="relu")
            return out

        mlmodel = ct.convert(my_program)
        mlmodel.save("MyModel.mlpackage")

    This direct MIL authoring is useful for:
        Custom layers not expressible in any framework.
        Research prototypes that need ANE-specific ops.
        Replacing specific sub-graphs during conversion (graph surgery).
        Performance-critical ops that the automatic converter does not fuse.

### MIL Passes: The Optimisation Pipeline

    When coremltools converts a model, it applies a sequence of MIL PASSES —
    graph transformations analogous to LLVM's optimisation passes:

    DEAD CODE ELIMINATION:
        Remove ops whose outputs are never used.
        Triggered by: intermediate computation that was unused after
        constant folding.

    CONSTANT FOLDING:
        Evaluate operations with all-constant inputs at conversion time.
        Example: reshape(const([1,2,3]), shape=[3]) → const([1,2,3]).
        Eliminates positional encodings, lookup tables, etc. that never change.

    COMMON SUBEXPRESSION ELIMINATION (CSE):
        If two ops compute the same result: compute once, use twice.
        Important for transformers where Q, K, V projections share input.

    FUSE LAYERNORM:
        Pattern: reduce_mean → sub → reduce_mean → add(eps) → rsqrt → mul → mul(gamma) → add(beta)
        Fused into: mb.layer_norm(x, axes, gamma, beta, eps)
        Result: single ANE-native layer norm op (far faster than the chain).

    FUSE GELU:
        Fuses the piecewise polynomial GELU implementation into a single
        mb.gelu(mode="TANH_APPROXIMATION") op.
        Essential for transformer performance on ANE.

    CONV + BN FUSION:
        Absorbs batch normalisation into the preceding convolution weights.
        Eliminates one entire BN pass at inference time.

    DEDUP WEIGHTS:
        Detects weight tensors with identical values and collapses them
        to a single shared constant. Reduces model size.

    NOOP ELIMINATION:
        Removes: transpose([0,1,2,3]) → identity
                 reshape([N,C,H,W]) → reshape([N,C,H,W]) (same shape)
                 add(x, 0.0) → x
                 mul(x, 1.0) → x

    Passes run automatically. You can inspect pass results:
        from coremltools.converters.mil.mil import passes
        model.lower_to_mil()   # access the MIL program


##### PART 4 — COREMLTOOLS: CONVERTING ANY MODEL TO CORE ML

### The Unified Conversion API

    coremltools.convert() is the single entry point for all conversions:

        import coremltools as ct
        mlmodel = ct.convert(
            model,                              ; source model
            inputs=[ct.TensorType(...)],        ; input specifications
            outputs=[ct.TensorType(...)],       ; output specifications (optional)
            minimum_deployment_target=ct.target.iOS16,  ; target OS version
            compute_units=ct.ComputeUnit.ALL,   ; which hardware to use
            convert_to="mlprogram",             ; "mlprogram" (MIL) or "neuralnetwork"
        )
        mlmodel.save("Model.mlpackage")

    The source model can be:
        A PyTorch TorchScript model or ExportedProgram
        A TensorFlow 1.x frozen graph or SavedModel
        A Keras model (TF 2.x)
        An ONNX ModelProto or file path
        A MIL program (built directly)
        scikit-learn / LibSVM models (via legacy API)

### PyTorch → Core ML

    PATH 1 — TorchScript (legacy, still widely used):
        import torch, coremltools as ct

        model = MyModel().eval()
        example_input = torch.randn(1, 3, 224, 224)

        # Trace with TorchScript
        traced = torch.jit.trace(model, example_input)

        # Convert
        mlmodel = ct.convert(
            traced,
            inputs=[ct.ImageType(
                name="input_image",
                shape=example_input.shape,
                scale=1/255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB,
            )],
            minimum_deployment_target=ct.target.iOS16,
        )

    PATH 2 — torch.export (modern, handles more models):
        import torch, coremltools as ct

        model = MyModel().eval()
        example = (torch.randn(1, 3, 224, 224),)
        exported_program = torch.export.export(model, example)

        mlmodel = ct.convert(
            exported_program,
            inputs=[ct.TensorType(shape=example[0].shape, dtype=float)],
            minimum_deployment_target=ct.target.iOS17,
        )

    CHOOSING BETWEEN TORCHSCRIPT AND TORCH.EXPORT:
        TorchScript: better op coverage for older models; may fail on
                     models with complex control flow or custom ops.
        torch.export: TorchDynamo-based, handles more modern architectures.
                      Required for models using torch.compile or functorch.
                      More likely to support new PyTorch features.

    COMMON PYTORCH CONVERSION ISSUES:
        "Unsupported op" during tracing:
            TorchScript cannot trace data-dependent shapes.
            Fix: use torch.export.export instead.

        Slow conversion time (> 5 minutes):
            Large model with many ops. The MIL passes take linear time.
            Fix: reduce passes with ct.PassPipelineConfig.

        "Op not supported by ANE":
            Some ops (custom layers, certain reductions) fall back to GPU/CPU.
            Fix: restructure the model to use ANE-compatible ops.
            Inspect with: mlmodel.get_spec().description (check layer types).

    DYNAMIC SHAPES (variable sequence length):
        from coremltools.converters.mil.input_types import RangeDim

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="input_ids",
                shape=ct.Shape(shape=(1, RangeDim(1, 512)))  # seq len 1–512
            )],
            minimum_deployment_target=ct.target.iOS17,
        )

### TensorFlow and Keras → Core ML

    TF 2.x SAVEDMODEL:
        import tensorflow as tf, coremltools as ct

        saved_model = tf.saved_model.load("./saved_model/")
        mlmodel = ct.convert(saved_model,
                             inputs=[ct.TensorType(shape=(1, 224, 224, 3))],
                             minimum_deployment_target=ct.target.iOS16)

    KERAS MODEL IN MEMORY:
        model = tf.keras.applications.MobileNetV3Small(weights="imagenet")
        mlmodel = ct.convert(model,
                             inputs=[ct.ImageType(shape=(1, 224, 224, 3))],
                             minimum_deployment_target=ct.target.iOS16)

    TF 1.x FROZEN GRAPH:
        mlmodel = ct.convert(
            "model.pb",
            inputs=[ct.TensorType(name="input", shape=(1, 224, 224, 3))],
            outputs=["output/Softmax"],
            source="tensorflow",
        )

### ONNX → Core ML

    ONNX is the most portable path — works for any framework that exports ONNX:
        import onnx, coremltools as ct

        onnx_model = onnx.load("model.onnx")
        mlmodel = ct.convert(
            onnx_model,
            minimum_deployment_target=ct.target.iOS16,
        )

    ONNX opset support in coremltools:
        opset 7–19 supported (tracking ONNX releases).
        Some exotic ONNX ops may not have MIL equivalents.
        Check with: ct.convert(..., debug=True) to see unsupported ops.

### Input and Output Specifications

    ct.ImageType:
        For CNN models fed camera frames or images.
        Parameters:
            name:           input tensor name
            shape:          (batch, height, width, channels) or (batch, channels, H, W)
            scale:          pixel value scale factor (1/255.0 for [0,1] range)
            bias:           per-channel subtraction after scaling
            color_layout:   ct.colorlayout.RGB or BGR or GRAYSCALE
            channel_first:  True for NCHW (PyTorch), False for NHWC (TF)

        VNImageFeaturePrintRequest handles the camera → ImageType pathway.
        When using Vision framework: Core ML receives YCbCr data from the
        camera and converts it automatically — no preprocessing in Swift.

    ct.TensorType:
        For raw numerical arrays (NLP models, audio, time series).
        Parameters:
            name:   input name
            shape:  static shape or ct.Shape with RangeDim for dynamic dims
            dtype:  float (default), int, bool

    COLOUR PREPROCESSING BAKED IN:
        When you set scale and bias in ImageType, Core ML performs:
            normalised = (pixel / 255.0) * scale + bias_per_channel
        at inference time, on the same compute unit as the model.
        No Python preprocessing needed in the app.

### Classifier Convenience

    For classification models, coremltools can add probability output:
        mlmodel = ct.convert(traced_model, ...)
        mlmodel.user_defined_metadata["class_labels"] = json.dumps(label_list)

        # Or use the dedicated classifier_config:
        from coremltools.models.neural_network import quantization_utils
        classifier_config = ct.ClassifierConfig(label_list)
        mlmodel = ct.convert(traced, classifier_config=classifier_config)
        # Output: "classLabel" (string) + "classLabelProbs" (dict str→float)
        # Automatically provides top-1 label in Swift:
        //   let label = prediction.classLabel
        //   let probs = prediction.classLabelProbs


##### PART 5 — QUANTISATION AND MODEL COMPRESSION

### Why Quantisation Matters on Apple Silicon

    Model size directly impacts:
        App Store download size (ideally < 50MB, required < 4GB OTA).
        RAM usage at inference time (shared with OS and other apps).
        Flash storage on the device (limited on entry-level iPhones).
        Time-to-first-inference (larger models take longer to map to ANE).

    Quantisation also directly impacts ANE utilisation:
        ANE is optimised for INT8 and FP16 compute, not FP32.
        FP32 models on ANE may run FP16 internally anyway (automatic downcast).
        INT8 weight quantisation reduces memory bandwidth — critical for
        transformer models where weights are the bottleneck.

### Float16 Weight Compression

    The simplest quantisation: convert all float32 weights to float16.
    Activations remain float32 at runtime (dequantised before computation).

        import coremltools as ct
        from coremltools.models.neural_network import quantization_utils as quant

        mlmodel = ct.convert(model, ...)   # start with float32 model

        # Compress weights to float16:
        mlmodel_fp16 = ct.compression_utils.affine_quantize_weights(
            mlmodel,
            mode="linear_symmetric",
            dtype=ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16,
        )
        mlmodel_fp16.save("model_fp16.mlpackage")

    TYPICAL RESULTS:
        MobileNetV3:   14 MB FP32 → 7 MB FP16 (2× size reduction)
        BERT-Base:     420 MB FP32 → 210 MB FP16
        Accuracy loss: essentially zero for most models (< 0.01%)
        Latency:       similar to FP32 (ANE runs FP16 natively)

### INT8 Weight Quantisation (Post-Training Quantisation)

    More aggressive: quantise weights to INT8 (4× size reduction).

        mlmodel_int8 = ct.compression_utils.affine_quantize_weights(
            mlmodel,
            mode="linear",              # or "linear_symmetric"
            nbits=8,                    # 8-bit integer
            op_selector=lambda op: isinstance(op, (Conv, MatMul)),
        )

    WITH CALIBRATION DATA (better accuracy):
        from coremltools.optimize.coreml import PostTrainingQuantizer, OptimizationConfig, OpPalettizerConfig

        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
                granularity="per_channel",   ; per-channel more accurate than per-tensor
            )
        )
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
            granularity="per_channel",
        )
        quantizer = ct.optimize.coreml.PostTrainingQuantizer(mlmodel, config)
        quantised_model = quantizer.compress()

    CALIBRATION-BASED QUANTISATION (activation statistics):
        For models sensitive to quantisation, calibrate on representative data:

        from coremltools.optimize.coreml import PostTrainingQuantizer

        def data_generator():
            for image_path in calibration_images[:100]:
                img = PIL.Image.open(image_path).resize((224, 224))
                yield {"input_image": img}

        quantizer = ct.optimize.coreml.PostTrainingQuantizer(
            mlmodel,
            config,
            data_loader=data_generator(),
        )
        calibrated_model = quantizer.compress()

### Palettization (Lookup Table / k-Bit Precision)

    Palettization maps each weight to one of 2ᵏ centroids, storing only
    the centroid index (k bits per weight). The centroid values (the "palette")
    are stored separately and learned or computed by k-means clustering.

    This is DIFFERENT from uniform INT8 quantisation:
        INT8:           stores an 8-bit approximation of each weight directly.
        Palettization:  stores a 2, 4, or 8-bit INDEX into a codebook of 4/16/256 values.
                        The codebook values are float16 (not restricted to a uniform grid).

    WHY PALETTIZATION IS GOOD FOR ANE:
        The ANE can execute palettized models natively (Spec 7, A17/M4+).
        At inference: the ANE loads the 4-bit index, looks up the float16
        centroid value, and uses that for computation. All on-chip.
        No dequantisation step required by the CPU.

    PALETTIZATION MODES:
        KMEANS:    k-means clustering of weight values → find optimal centroids.
                   Best accuracy. Slower (runs k-means per layer).
        UNIFORM:   centroids are uniformly spaced in the weight range.
                   Fast. Slightly less accurate than k-means.
        UNIQUE:    uses the weight's unique values as centroids (up to 2ᵏ).
                   Best for weights that already have discrete values.
        CUSTOM:    provide your own centroid lookup table.

    USAGE:
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4,         # 4-bit: 16 centroids per weight
                mode="kmeans",   # or "uniform"
                granularity="per_grouped_channel",  # group every 32 channels
            )
        )
        palettized = ct.optimize.coreml.palettize_weights(mlmodel, config)

    BITS / COMPRESSION RATIO / ACCURACY:
        nbits=8:  256 centroids — near-lossless, 4× compression vs FP32
        nbits=4:  16 centroids — 8× compression, < 0.5% accuracy drop typical
        nbits=2:  4 centroids  — 16× compression, noticeable accuracy drop
        nbits=1:  2 centroids  — binary weights, 32× compression, significant drop

### Activation Quantisation (INT8 Activations)

    Core ML 7 (iOS 17 / Spec 7) introduced INT8 ACTIVATION quantisation.
    This quantises both weights AND activations to INT8 — the full-integer
    path analogous to TFLite's static INT8 mode.

        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
                granularity="per_channel",
                quantize_activations=True,   # enable INT8 activations
            )
        )

    CALIBRATION IS REQUIRED for activation quantisation:
        Must pass a data loader that yields representative inputs.
        The quantiser collects activation statistics (min/max per tensor).
        Without calibration: cannot determine scale/zero_point for activations.

    ANE INT8 NATIVE:
        On A17 Pro and M4+ (using Spec 7):
            The ANE executes INT8 activations natively.
            No dequantise/requantise at op boundaries.
            Full INT8 throughput (2× over FP16).

### Mixed Precision: Per-Layer Quantisation

    Different layers have different sensitivity to quantisation error.
    Mixed precision assigns different precision to each layer:

        from coremltools.optimize.coreml import OptimizationConfig, \
            OpLinearQuantizerConfig, OpPalettizerConfig

        config = ct.optimize.coreml.OptimizationConfig(
            global_config=OpPalettizerConfig(nbits=4, mode="kmeans"),
            op_type_configs={
                # Keep embedding layers at 8-bit (sensitive)
                "gather": OpLinearQuantizerConfig(dtype=np.int8),
                # Keep final linear at 8-bit (output sensitive)
                "linear": OpLinearQuantizerConfig(dtype=np.int8),
            },
            op_name_configs={
                # Keep first and last convolution at FP16
                "first_conv": None,    # None = no quantisation (keep FP16)
                "last_fc":    None,
            }
        )


##### PART 6 — COMPUTE UNITS: ANE, GPU, AND CPU DISPATCH

### The Three Compute Units

    Core ML can dispatch model layers to three hardware resources:

    CPU (Accelerate / BNNS):
        Apple's Accelerate framework provides optimised CPU kernels.
        BNNS (Basic Neural Network Subroutines): CPU primitives for
        convolution, matmul, pooling, normalisation.
        Available on ALL Apple devices.
        Precision: FP32, FP16, INT8 (hardware-dependent).
        Best for: ops not supported by ANE/GPU, small models, fallback.
        Power: highest per-operation power; not power-efficient for large models.

    GPU (Metal Performance Shaders):
        MPS / MPSGraph: Apple's GPU compute framework.
        Executes convolution, matmul, elementwise on the integrated GPU.
        Available on all devices with GPU (all iPhones, iPads, Macs).
        Precision: FP16 (native), FP32 on some GPUs.
        Best for: ops with large batch sizes, rendering + ML pipelines.
        Latency: higher first-inference (shader compilation on first call).
        Benefit: MPS shader cache after first compilation (warm subsequent calls).

    ANE (Neural Engine):
        Fixed-function hardware accelerator.
        Available on iPhone 8 (A11) and later; M1 Mac and later.
        Precision: FP16, INT8 (Spec 7+ on A17 Pro / M4).
        Best for: dense matmul and convolution at inference batch=1.
        Constraints: strict requirements on op types, shapes, and precision.
        Power: most efficient (TOPS/watt >> GPU >> CPU).

### ComputeUnit Configuration

    The compute_units argument to MLModel() controls which hardware is used:

        import CoreML

        let config = MLModelConfiguration()
        config.computeUnits = .all               // ANE first, GPU second, CPU fallback
        config.computeUnits = .cpuOnly           // CPU only (debugging, deterministic)
        config.computeUnits = .cpuAndGPU         // skip ANE (useful if model doesn't run well on ANE)
        config.computeUnits = .cpuAndNeuralEngine // skip GPU
        let model = try MLModel(contentsOf: modelURL, configuration: config)

    In Python (coremltools inference, macOS only):
        import coremltools as ct
        model = ct.models.MLModel("Model.mlpackage",
                                  compute_units=ct.ComputeUnit.ALL)
        # ALL = ANE + GPU + CPU (default)
        # CPU_ONLY = for testing/debugging
        # CPU_AND_GPU = bypass ANE
        # CPU_AND_NE  = bypass GPU

### How Core ML Decides Which Hardware to Use

    Core ML makes the dispatch decision AT COMPILE TIME, not at runtime.
    When compile_model is called (or at Xcode build time):

    STEP 1 — MODEL ANALYSIS:
        Core ML walks the MIL graph and checks each op against the
        ANE's supported op set (op type + shape + precision constraints).

    STEP 2 — PARTITION:
        Contiguous runs of ANE-compatible ops → ANE partition.
        Non-compatible ops → GPU or CPU partition.
        Goal: maximise the fraction of FLOPs on ANE.

    STEP 3 — GENERATE DEVICE CODE:
        ANE partition: compile to ANE dataflow graph (proprietary format).
        GPU partition: generate Metal shader code (.metallib).
        CPU partition: select BNNS primitives.

    STEP 4 — BUILD DISPATCH PLAN:
        The final dispatch plan (stored in .mlmodelc) describes:
        which tensor buffers live on which memory region (unified but
        different virtual addresses for CPU/ANE/GPU access).
        Inter-partition data handoffs (all zero-copy on Apple Silicon).

### ANE Compatibility Requirements

    The ANE has strict requirements. Ops and configurations that CANNOT
    run on the ANE (fall back to GPU/CPU):

    UNSUPPORTED OPERATIONS:
        Custom layers (Python, C++ extensions) — always CPU
        Gather with non-constant indices (dynamic lookup tables)
        TopK, Sort, ArgSort
        Non-batched dynamic shapes (some sequence models)
        Einsum with complex contraction patterns
        Some control flow ops (while_loop with variable iteration count)

    SHAPE CONSTRAINTS:
        Input channels: typically multiples of 8 or 16 preferred
        Batch size: ANE is SINGLE-SAMPLE optimised (batch=1 is fastest)
                    batch > 1 may run on GPU instead of ANE
        Spatial dimensions: multiples of 2 or 4 preferred for convolution
        Weight dimensions: some dimension combinations exclude ANE
        Sequence length > 2048: may exceed ANE buffer limits (model-dependent)

    PRECISION CONSTRAINTS:
        FP32 models: ANE downcasts to FP16 internally (may cause outlier errors)
        For full ANE utilisation: use FP16 weights (compress_to_fp16=True)
        INT8 ANE native: A17 Pro and M4 only, requires Spec 7

    VERIFYING ANE UTILISATION:
        Use Instruments (Xcode profiler) → Core ML Instrument:
            Shows per-layer execution device (ANE / GPU / CPU)
            Shows per-layer latency
            Shows energy consumption

        For a model running well on ANE:
            ANE should handle > 90% of total compute time
            Remaining 10% is preprocessing/postprocessing on CPU

### Performance Profiling with Instruments

    Instruments core ML profiling:
        1. Open Xcode → Product → Profile (Cmd+I)
        2. Choose "Core ML" instrument template
        3. Run your app on device (not simulator — ANE only on real hardware)
        4. The timeline shows: compute unit, duration, layer name

    TYPICAL PROFILE for a well-optimised transformer:
        MHA (multi-head attention):  ANE  8 ms   (fused sdpa_with_kv_cache)
        FFN (fully connected):       ANE  4 ms
        LayerNorm:                   ANE  0.5 ms
        Embedding lookup (gather):   CPU  0.1 ms  (gather falls back to CPU)
        Sampling (topk/argsort):     GPU  0.2 ms

    DIAGNOSING SLOW INFERENCE:
        "ANE partition is empty" → entire model on GPU/CPU
        Likely cause: FP32 input type, unsupported op in critical path
        Fix: use ImageType with scale+bias, audit ops for ANE compat

        "Many small CPU partitions" → frequent ANE↔CPU boundary crossings
        Each crossing: data handoff between memory regions (~0.2 ms each)
        Fix: restructure model to minimise non-ANE ops in the critical path


##### PART 7 — THE SWIFT/OBJECTIVE-C RUNTIME API

### Loading a Model

    BUNDLE-BASED LOADING (most common — model in Xcode project):
        let model = try! MyModel()   // auto-generated class from Xcode
        // Xcode generates MyModel.swift from MyModel.mlpackage
        // with strongly-typed input/output structs

    MANUAL LOADING (for OTA model updates):
        let modelURL = Bundle.main.url(forResource: "MyModel",
                                       withExtension: "mlmodelc")!
        let config   = MLModelConfiguration()
        config.computeUnits = .all
        let model    = try! MLModel(contentsOf: modelURL, configuration: config)

    RUNTIME COMPILATION (for .mlpackage shipped OTA):
        let packageURL = documentDirectory.appendingPathComponent("MyModel.mlpackage")
        let compiledURL = try! MLModel.compileModel(at: packageURL)
        let model = try! MLModel(contentsOf: compiledURL, configuration: config)
        // compiledURL is a temp directory — persist it to app container
        let savedURL = appContainer.appendingPathComponent("MyModel.mlmodelc")
        try! FileManager.default.copyItem(at: compiledURL, to: savedURL)

### The Prediction API

    SYNCHRONOUS INFERENCE:
        // Using auto-generated typed interface (simplest):
        let prediction = try model.prediction(input: MyModelInput(image: ciImage))
        let label      = prediction.classLabel    // String
        let probs      = prediction.classLabelProbs  // [String: Double]

        // Using MLFeatureProvider (flexible):
        let input   = try MLDictionaryFeatureProvider(dictionary: [
            "input_image": MLFeatureValue(cgImage: cgImage, orientation: .up,
                                         constraint: imageConstraint, options: [:])
        ])
        let output  = try model.prediction(from: input)
        let probs   = output.featureValue(for: "logits")?.multiArrayValue

    ASYNCHRONOUS INFERENCE (iOS 16+, avoids blocking main thread):
        Task {
            let prediction = try await model.prediction(input: myInput)
            await MainActor.run {
                updateUI(with: prediction.classLabel)
            }
        }

    BATCH INFERENCE:
        let batchInput  = [MyModelInput(image: img1), MyModelInput(image: img2)]
        let batchOutput = try model.predictions(inputs: batchInput)
        // Internally: Core ML may fuse these into a single batched inference

### MLMultiArray and MLShapedArray

    MLMultiArray is Core ML's N-dimensional array type.
    It is the type used for TensorType inputs and outputs.

    CREATING MLMultiArray:
        // From shape:
        let arr = try MLMultiArray(shape: [1, 512], dataType: .float16)

        // From UnsafePointer (zero-copy when possible):
        let arr = try MLMultiArray(dataPointer: rawPtr,
                                   shape: [1, 512],
                                   dataType: .float32,
                                   strides: [512, 1],
                                   deallocator: nil)

    ACCESSING DATA:
        arr[[0, 0]] = 1.0    // subscript setter
        let val = arr[[0, 0]] as! Float

        // Faster: typed pointer
        arr.withUnsafeMutableBytes { ptr in
            let floats = ptr.bindMemory(to: Float.self)
            floats.baseAddress?.initialize(repeating: 0.0, count: arr.count)
        }

    MLShapedArray (iOS 15+, Swift-native):
        // Strongly typed generic wrapper around MLMultiArray
        let shaped = MLShapedArray<Float>(repeating: 0.0, shape: [1, 512])
        let output  = prediction.outputTensor    // MLShapedArray<Float>
        let logits  = output[0, ..]              // row 0, all columns

### Core ML and the Vision Framework

    Apple's Vision framework (VN prefix) provides high-level CV pipelines
    that automatically handle camera-frame preprocessing before Core ML:

    IMAGE CLASSIFICATION:
        let request = VNCoreMLRequest(model: vnModel) { request, error in
            guard let results = request.results as? [VNClassificationObservation] else { return }
            let topResult = results.sorted { $0.confidence > $1.confidence }.first!
            print("\\(topResult.identifier): \\(topResult.confidence)")
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

    WHY VISION IS FASTER THAN MANUAL PREPROCESSING:
        Vision handles YCbCr → RGB conversion on the GPU.
        It handles resizing and normalisation in a single Metal pass.
        The result goes directly to Core ML without touching CPU memory.
        End-to-end: camera ISP → Metal → ANE → result. No CPU copies.

    OBJECT DETECTION (with VNCoreMLFeatureValueObservation):
        Let Core ML output bounding boxes as MLMultiArray.
        Vision wraps bounding box postprocessing with NMS optionally.

### Updatable Models (On-Device Learning)

    Core ML supports limited on-device fine-tuning via Updatable Models.
    Certain layer types (Linear, fully-connected) can be updated at runtime
    using a small labelled dataset collected on the device.

    This is used for personalisation:
        "Hey Siri" personalisation — adapts to your voice on device.
        Drawing app — adapts to your handwriting style.
        Recommendation systems — personalised without sending data to server.

    MAKING A MODEL UPDATABLE (coremltools):
        from coremltools.models import MLModel
        spec = mlmodel.get_spec()
        # Mark specific layers as updatable:
        ct.utils.make_pipeline_updatable(spec, layers=["final_dense"])
        mlmodel_updatable = MLModel(spec)
        mlmodel_updatable.save("UpdatableModel.mlpackage")

    RUNNING UPDATES IN SWIFT:
        let updateTask = try MLUpdateTask(forModelAt: modelURL,
                                          trainingData: trainingData,
                                          configuration: updateConfig,
                                          completionHandler: { context in
            let updatedModel = context.model
            try updatedModel.write(to: updatedModelURL)
        })
        updateTask.resume()


##### PART 8 — STATEFUL MODELS AND LLM INFERENCE ON DEVICE

### Stateful Models: Core ML Spec 8 (iOS 18)

    Standard Core ML inference is STATELESS: each call to predict() takes
    inputs and produces outputs, with no memory between calls.
    For LLM generation, this is inefficient: the KV-cache (key-value pairs
    from all previous attention layers) must be passed as an explicit input
    and re-read by the ANE on every decode step — a massive memory transfer.

    Core ML Spec 8 (iOS 18, macOS 15) introduces STATEFUL MODELS:
    models that maintain internal state (specifically KV-cache tensors)
    between prediction calls, entirely on the ANE.

    HOW STATEFUL KV-CACHE WORKS:
        The ANE has on-chip SRAM (~50–100 MB depending on chip generation).
        For LLM inference, the KV-cache lives in this SRAM.
        On each decode step:
            Input: new token embedding (tiny — just one vector)
            On-chip: read K,V from SRAM; compute new K,V; write back to SRAM
            Output: logits for next token prediction
        The KV-cache NEVER leaves the ANE's SRAM between steps.
        This eliminates the dominant memory transfer in LLM inference.

    EXPORTING A STATEFUL LLM MODEL:
        coremltools 7.0+ supports stateful model export:

        import coremltools as ct

        # The model must use mb.read_state / mb.coreml_update_state ops
        # OR use mb.sdpa_with_kv_cache which handles state automatically.

        # With torch.export (for models using torch.nn.KVCache):
        from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
        # Use the transformers-to-coremltools pipeline for LLaMA/Mistral/Phi models.

        # Apple provides first-party conversion scripts for supported models
        # via the apple/coremltools examples repository.

### Apple Intelligence and Foundation Models

    Apple Intelligence (announced WWDC 2024, shipped iOS 18.1) is Apple's
    on-device AI feature set. The technical infrastructure:

    ON-DEVICE MODELS (run on ANE, never leave the device):
        Apple's proprietary language model (~3B parameters, unpublished details)
        Runs on iPhone 15 Pro / M1 iPad Pro or later.
        Powers: Writing Tools, Smart Reply, Priority Notifications,
                Photo cleanup, Image Playground (some features).
        Quantisation: heavily quantised (believed to be 4-bit with
                       mixed precision for sensitive layers).
        Latency: ~20–30 tokens/second on A17 Pro ANE.

    PRIVATE CLOUD COMPUTE (server-side for complex requests):
        Larger models on Apple-designed servers with hardware attestation.
        Privacy guarantee: server cannot log prompts; auditable by researchers.
        Routes complex Writing Tools requests too large for on-device model.

    FOUNDATION MODELS FRAMEWORK (iOS 18.1+, macOS 15.1+):
        Apple exposes the on-device language model via a high-level API:

        import FoundationModels

        let session = LanguageModelSession()
        let response = try await session.respond(to: "Write a haiku about code.")
        print(response.content)

        // Structured output:
        @Generable struct BookRecommendation {
            @Guide("title of the book") var title: String
            @Guide("author name")       var author: String
            @Guide("one-sentence summary") var summary: String
        }
        let recommendation = try await session.respond(
            to: "Recommend a programming book.",
            generating: BookRecommendation.self)

        RESTRICTIONS (Apple Intelligence on-device model):
            Not a general API — restricted to predefined tasks.
            Cannot be accessed like a free-form LLM via FoundationModels.
            Foundation Models Framework exposes a limited set of intents.

    THIRD-PARTY LLM DEPLOYMENT (using Core ML directly):
        Developers can deploy their own LLMs via Core ML Spec 8:

        Supported open-source models (as of 2024):
            Llama 3.1 8B (quantised to 4-bit → ~4 GB)
            Mistral 7B (4-bit → ~3.8 GB)
            Phi-3 Mini (3.8B, 4-bit → ~2 GB)
            Gemma 2 2B (4-bit → ~1.3 GB)

        Requirements:
            iPhone 15 Pro (A17 Pro): 8 GB LPDDR5 unified memory, 35 TOPS ANE
            M1 iPad Pro:             8–16 GB unified memory
            M-series Mac:            8–128 GB unified memory

        Conversion via apple/ml-llm-convert:
            python convert.py --model-id meta-llama/Llama-3.1-8B-Instruct \\
                              --output ./Llama31_8B_4bit \\
                              --quant-type 4bit \\
                              --context-length 2048

        LLM generation speed (approximate, A17 Pro):
            Llama 3.1 8B 4-bit:   10–15 tokens/second
            Phi-3 Mini 4-bit:     25–35 tokens/second
            Gemma 2 2B 4-bit:     40–60 tokens/second


##### PART 9 — DEPLOYMENT WORKFLOW: FROM XCODE TO APP STORE

### The End-to-End Workflow

    STEP 1 — TRAIN (Python):
        Train your model in PyTorch, TensorFlow, or JAX.
        Validate accuracy on the test set.
        Save the trained weights.

    STEP 2 — CONVERT (coremltools, Python):
        Convert to .mlpackage using coremltools.convert().
        Apply quantisation (FP16, INT8, palettization).
        Validate numerically against original framework.
        Test on macOS using the Python API (coremltools.models.MLModel.predict()).

    STEP 3 — ADD TO XCODE:
        Drag .mlpackage into your Xcode project.
        Xcode auto-generates a Swift interface class (MyModel.swift).
        The generated class has:
            - A MyModelInput struct with strongly-typed inputs
            - A MyModelOutput struct with strongly-typed outputs
            - A prediction() method

    STEP 4 — COMPILE (Xcode build):
        Xcode compiles the .mlpackage to .mlmodelc for each target device.
        The .mlmodelc is embedded in the app bundle.
        Archive size impact: a quantised 224×224 classification model ≈ 5–15 MB.

    STEP 5 — TEST ON DEVICE:
        Use Instruments → Core ML template.
        Verify: which compute unit handles each layer.
        Check: latency, power usage, accuracy on real inputs.

    STEP 6 — SUBMIT TO APP STORE:
        Standard App Store submission.
        Apple reviews the app but NOT the model content.
        Large model note: models > 200 MB trigger App Store thin provisioning
        considerations (cellular download limit is 200 MB by default).

### Model Sizes and App Store Considerations

    App Store download limit:
        < 200 MB:   downloads over cellular automatically (no user prompt)
        200 MB–4 GB: displays size warning; user must confirm
        > 4 GB:     not allowed for initial download (use on-demand resources)

    Using On-Demand Resources (ODR) for large models:
        Mark the .mlpackage as an on-demand resource with a tag.
        The initial app download does NOT include the model.
        At runtime, request the model bundle:
            NSBundleResourceRequest(tags: ["large_model"]).beginAccessingResources { ... }
        The OS downloads and caches the model on first use.
        Subsequent launches use the cached version.
        Useful for: LLMs, large vision models.

    MODEL SIZE OPTIMISATION:
        FP16 compression: 2× reduction (mandatory for production)
        4-bit palettization: 8× reduction vs FP32 (recommended for LLMs)
        Selective quantisation: keep sensitive layers at FP16/INT8

### OTA Model Updates

    For production apps that need to update the model without an App Store
    release (A/B testing, model improvements, bug fixes):

    PATTERN 1 — CloudKit or S3 download:
        Store the .mlpackage in CloudKit / S3 / CDN.
        On first launch (or periodic): check for updates.
        Download new .mlpackage to app's Library/Application Support directory.
        Compile with MLModel.compileModel(at:) on first use.
        Cache the .mlmodelc.
        On next launch: load from cached .mlmodelc (fast).

    PATTERN 2 — Model personalisation loop:
        Ship a base model in the bundle.
        On-device: collect user-specific training data (with consent).
        Run MLUpdateTask to fine-tune the model.
        Save the updated .mlmodelc for future use.
        (Used by Health app, Siri personalisation, Keyboard.)


##### PART 10 — CORE ML ACROSS THE APPLE ECOSYSTEM

### Platform Coverage

    IPHONE / IPAD:
        All Core ML features available.
        ANE available from iPhone 8 / iPad Pro 2018.
        INT8 native ANE: iPhone 15 Pro (A17) and later.
        Typical models: vision classifiers, NLP, audio, AR depth estimation.

    APPLE WATCH:
        Core ML available from watchOS 4.
        No ANE in Watch — CPU and small GPU only.
        Model size budget: much tighter (watch has 32 MB RAM for apps).
        Recommended max model size: < 5 MB (quantised).
        Use case: heart rate classification, motion detection, gesture recognition.

    APPLE TV:
        Core ML available from tvOS 11.
        Similar capabilities to iPhone (A-series chip).
        Use case: content recommendation, image super-resolution.

    MAC (Apple Silicon — M1 / M2 / M3 / M4):
        Full Core ML support with high-performance ANE.
        coremltools.models.MLModel.predict() works on macOS for development.
        No ANE on Intel Macs (pre-2020) — CPU and GPU only.
        M4 Ultra: ~76 TOPS ANE; used by Final Cut Pro's ML noise reduction.

    APPLE VISION PRO (visionOS):
        Core ML available from visionOS 1.0.
        M2 chip (ANE available) + R1 co-processor for sensor fusion.
        Use case: hand tracking, scene understanding, spatial computing ML.
        Simulator: no ANE (developer workstation CPU only).

### Differences by Platform

    MODEL COMPILATION:
        iOS/iPadOS: Xcode compiles per device family. Single .ipa may contain
                    multiple .mlmodelc compiled variants.
        macOS: one .mlmodelc for the current Mac's chip architecture.
        watchOS: Xcode generates Watch-specific compiled model.

    COMPUTE UNIT AVAILABILITY:
        iOS (A11+):         CPU + GPU + ANE
        iOS (A10 and older): CPU + GPU (no ANE)
        watchOS (all):      CPU only (some GPU on recent models)
        macOS (Apple Silicon): CPU + GPU + ANE
        macOS (Intel):      CPU + GPU (no ANE)
        visionOS:           CPU + GPU + ANE (from M2 chip in Vision Pro)

### Metal Performance Shaders: The GPU Compute Path

    When Core ML dispatches ops to GPU, it uses Metal Performance Shaders (MPS).
    MPS is Apple's lower-level GPU compute framework. Understanding MPS
    helps when optimising Core ML models or building custom GPU ops.

    MPS PRIMITIVES used by Core ML GPU path:
        MPSCNNConvolution:      convolution on GPU
        MPSMatrixMultiplication: matrix multiply
        MPSCNNFullyConnected:   dense layer
        MPSCNNSoftMax:          softmax
        MPSCNNNeuron (subclasses): relu, sigmoid, tanh, gelu
        MPSNNAdd/Multiply:      elementwise ops

    MPSGRAPH (iOS 15+, macOS 12+):
        Higher-level API that builds computation graphs for GPU.
        Analogous to TensorFlow's computation graph but for Metal.
        Used by Core ML's GPU path internally.
        Also useful for custom GPU compute that Core ML doesn't support:

            let graph = MPSGraph()
            let xTensor = graph.placeholder(shape: [-1, 512], dataType: .float16, name: "x")
            let wTensor = graph.constant(weights, shape: [512, 256], dataType: .float16)
            let y = graph.matrixMultiplication(primary: xTensor, secondary: wTensor, name: "mm")
            let result = graph.sigmoidGradient(gradient: ..., forwardIn: y, name: "sig")
            let feed  = [xTensor: MPSGraphTensorData(inputBuffer, ...)]
            let fetch = graph.run(feeds: feed, targetTensors: [result], ...)


##### PART 11 — CORE ML IN THE CONNECTED COMPILER STACK

### Core ML and LLVM (Module 01)

    The CPU path in Core ML uses Accelerate/BNNS.
    Accelerate is Apple's optimised compute library, compiled with Apple's
    Clang (LLVM-based) with ARM Neon and ARMv8.3+ matrix multiply intrinsics.
    BNNS uses LLVM's auto-vectoriser for ARM SVE on M4+ chips.
    The coremltools Python library itself is LLVM/Clang compiled.

### Core ML and MLIR (Module 02)

    coremltools uses MLIR internally for MIL pass infrastructure.
    The MIL → device-specific lowering pipeline uses MLIR dialect concepts.
    Future coremltools versions are converging toward an MLIR-native frontend.
    MIL's block/region structure directly mirrors MLIR's region-based IR.

### Core ML and XLA / OpenXLA (Modules 05-06)

    JAX models reach Core ML via the ONNX pathway:
        JAX → jax.export → StableHLO → (stablehlo-onnx-bridge) → ONNX → coremltools.
    Or directly via TorchScript (via torch-xla side):
        This path is uncommon; ONNX is preferred.
    XLA itself does not target Core ML or the ANE.

### Core ML and TVM (Module 08)

    TVM has a Metal backend for Apple GPU compute.
    TVM's MetaSchedule can find optimal tile sizes for Apple GPU.
    However, TVM cannot target the ANE — that requires Core ML.
    For Apple hardware deployment:
        Maximum throughput (ANE utilisation) → Core ML.
        Custom GPU kernels with precise tile control → TVM + Metal.
    These are complementary: TVM for GPU research; Core ML for production.

### Core ML and ONNX (Module 10)

    ONNX is Core ML's primary import format in coremltools.
    The conversion path:
        ONNX model → coremltools ONNX frontend → MIL program → .mlpackage
    Supported: ONNX opsets 7–19.
    Limitations: ops without MIL equivalents need custom layers.
    Best practice: run onnx-simplifier before coremltools.convert() to
    remove ONNX artifacts (Cast, Identity nodes) that complicate conversion.

### Core ML and TFLite (Module 11)

    TFLite and Core ML serve overlapping use cases:
        TFLite targets Android + MCU + Linux edge.
        Core ML targets Apple devices exclusively.
    They are NOT alternatives for the same deployment target.
    Conversion bridges:
        TFLite (.tflite) → ONNX → coremltools → Core ML (lossy, some ops unsupported).
        TensorFlow → coremltools directly (preferred over TFLite roundtrip).
    TFLite's GPU delegate on iOS uses Metal internally — the same GPU as Core ML.
    Core ML is always preferred for iOS; TFLite on iOS is for cross-platform codebases.

### Core ML and OpenVINO (Module 12)

    OpenVINO targets Intel hardware exclusively.
    Core ML targets Apple hardware exclusively.
    There is zero hardware overlap — they never compete directly.
    Developers building for both Intel and Apple hardware need both tools.
    ONNX is the bridge: a model can be converted to both OpenVINO IR and
    Core ML format from the same ONNX file.

### The Complete Core ML Ecosystem

    ┌──────────────────────────────────────────────────────────────────────┐
    │  TRAINING (developer machine — Python)                               │
    │  PyTorch   TensorFlow  JAX   scikit-learn                           │
    │  (torch.export/TorchScript)  (SavedModel/Keras)  (ONNX export)     │
    └──────────────────────────┬───────────────────────────────────────────┘
                               │
             coremltools.convert()  +  compress/quantise
             MIL passes: fold BN, fuse LayerNorm, fuse GELU
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  .mlpackage (MIL graph + FP16/INT4/INT8 weights)                    │
    └────┬─────────────────────┬──────────────────────────────────────────┘
         │                     │
         ▼ Xcode build         ▼ On-device MLModel.compileModel()
    ┌──────────────────────────────────────────────────────────────────────┐
    │  .mlmodelc (device-specific binary: ANE plan + Metal shaders + BNNS)│
    └────┬─────────────────────┬─────────────────────┬────────────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
    ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐
    │  ANE        │    │  GPU (Metal  │    │  CPU (Accelerate/    │
    │  Neural     │    │  MPS/        │    │  BNNS)               │
    │  Engine     │    │  MPSGraph)   │    │  Fallback for all    │
    │  FP16/INT8  │    │  FP16        │    │  unsupported ops     │
    │  35–38 TOPS │    │  Unified mem │    │  ARM Neon / SVE      │
    └─────────────┘    └──────────────┘    └──────────────────────┘

    ZERO-COPY UNIFIED MEMORY: all three share the same physical DRAM.
    Inter-unit data handoffs require no memcpy — just virtual address remapping.

    COMPARISON TABLE:
    ┌──────────────────────┬──────────────┬──────────────┬──────────────┐
    │ Property             │  Core ML     │  TFLite iOS  │  ONNX RT iOS │
    ├──────────────────────┼──────────────┼──────────────┼──────────────┤
    │ ANE utilisation      │ YES (full)   │ No           │ Via CoreML EP│
    │ GPU (Metal)          │ YES          │ YES (MPS)    │ YES          │
    │ CPU (BNNS)           │ YES          │ YES (XNNPACK)│ YES          │
    │ Quantisation support │ FP16/INT8/4b │ INT8/FP16    │ INT8         │
    │ Stateful LLM         │ YES (iOS 18) │ No           │ No           │
    │ Xcode integration    │ Native       │ Manual       │ Manual       │
    │ On-device learning   │ YES          │ No           │ No           │
    │ Vision framework     │ Native       │ No           │ No           │
    │ Model size minimum   │ ~100 KB      │ ~50 KB       │ ~100 KB      │
    │ Simulator support    │ YES (CPU)    │ YES          │ YES          │
    └──────────────────────┴──────────────┴──────────────┴──────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · coremltools Conversion — PyTorch and ONNX to .mlpackage": {
        "description": (
            "End-to-end conversion from PyTorch and ONNX models to Core ML format. "
            "Convert a CNN with ImageType inputs (scale, bias, colour layout). "
            "Convert a transformer with TensorType and dynamic sequence length. "
            "Inspect the resulting MIL program: ops, types, and shapes. "
            "Validate numerical accuracy: coremltools predict vs PyTorch reference. "
            "Show the .mlpackage bundle layout and protobuf spec structure."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  COREMLTOOLS CONVERSION — PYTORCH AND ONNX TO .MLPACKAGE")
print("=" * 65)
print()

try:
    import coremltools as ct
    print(f"  coremltools {ct.__version__}")
    HAS_CT = True
except ImportError:
    HAS_CT = False
    print("  coremltools not installed: pip install coremltools")
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
# SECTION 1: Convert a CNN with ImageType input
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Convert CNN with ImageType: camera-ready model")
print("━" * 65)
print()

IMAGETYPE_GUIDE = """
  IMAGETYPE vs TENSORTYPE — CRITICAL CHOICE FOR CAMERA MODELS
  ════════════════════════════════════════════════════════════════

  When you declare an ImageType input in coremltools, Core ML handles
  the ENTIRE camera-to-tensor preprocessing pipeline automatically:

    Camera (YCbCr) → ISP → Metal (colour space convert + resize + normalise)
                  → ANE (model inference)

  vs TensorType:

    Camera → CPU (Python/Swift preprocessing: resize, /255, subtract mean)
           → Core ML → ANE

  The ImageType path is entirely on hardware — no CPU involvement.
  For real-time camera (30fps): the difference is 5–15 ms per frame.

  ImageType parameters:
    shape:        (batch, H, W, channels) for NHWC  OR  (batch, channels, H, W) NCHW
    scale:        multiply pixel values by this (e.g. 1/255.0 for [0,1] range)
    bias:         per-channel subtraction AFTER scaling
                  For ImageNet normalisation (mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]):
                      scale = 1/255.0
                      bias  = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
                      (because: normalised = (x/255 - mean) / std
                                           = (x/255) * (1/std) - mean/std
                                             ^scale            ^bias)
    color_layout: ct.colorlayout.RGB (R first) or BGR (B first, OpenCV default)

  CHECKING THE RESULT:
    with PIL.Image.open("test.jpg") as img:
        img_resized = img.resize((224, 224))
    prediction = coreml_model.predict({"input_image": img_resized})
    # No numpy operations needed — PIL Image is accepted directly!
"""
print(IMAGETYPE_GUIDE)

if HAS_CT and HAS_TORCH:
    # Define MobileNet-style CNN
    class MobileBlock(nn.Module):
        def __init__(self, c_in, c_out, stride=1):
            super().__init__()
            self.dw = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in), nn.ReLU6())
            self.pw = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU6())
        def forward(self, x): return self.pw(self.dw(x))

    class TinyCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU6())
            self.blocks = nn.Sequential(
                MobileBlock(16, 32, stride=2),
                MobileBlock(32, 64, stride=2),
                MobileBlock(64, 64))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, num_classes))
        def forward(self, x): return self.head(self.blocks(self.stem(x)))

    model = TinyCNN(10).eval()
    example = torch.randn(1, 3, 64, 64)

    print(f"  Model: TinyCNN  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  Input: {list(example.shape)}")
    print()

    try:
        traced = torch.jit.trace(model, example)

        # ── Convert with ImageType input ──────────────────────────────────
        print("  Converting with ct.ImageType (scale + bias for normalisation)...")
        t0 = time.perf_counter()
        mlmodel = ct.convert(
            traced,
            inputs=[ct.ImageType(
                name="input_image",
                shape=example.shape,
                scale=1.0 / (255.0 * 0.226),   # combined scale
                bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                color_layout=ct.colorlayout.RGB,
                channel_first=True,
            )],
            minimum_deployment_target=ct.target.iOS16,
            convert_to="mlprogram",
        )
        t_conv = (time.perf_counter() - t0) * 1000
        print(f"  Conversion time: {t_conv:.0f} ms")
        print()

        # ── Inspect the model spec ────────────────────────────────────────
        spec = mlmodel.get_spec()
        print(f"  Model spec version: {spec.specificationVersion}")
        print(f"  Model type:         {spec.WhichOneof('Type')}")
        print()

        # ── Show input/output descriptions ───────────────────────────────
        desc = spec.description
        print("  Inputs:")
        for inp in desc.input:
            img_type = inp.type.imageType
            arr_type = inp.type.multiArrayType
            if inp.type.HasField("imageType"):
                print(f"    {inp.name:<25s} ImageType "
                      f"[{img_type.width}×{img_type.height}]  "
                      f"colorSpace={img_type.colorSpaceModel}")
            else:
                print(f"    {inp.name:<25s} MultiArray "
                      f"shape={list(arr_type.shape)}")
        print("  Outputs:")
        for out in desc.output:
            arr_type = out.type.multiArrayType
            print(f"    {out.name:<25s} MultiArray "
                  f"shape={list(arr_type.shape)}  "
                  f"dtype={arr_type.dataType}")
        print()

        # ── Save and inspect the .mlpackage bundle ────────────────────────
        tmpdir   = tempfile.mkdtemp()
        pkg_path = os.path.join(tmpdir, "TinyCNN.mlpackage")
        mlmodel.save(pkg_path)

        print(f"  .mlpackage contents:")
        for root, dirs, files in os.walk(pkg_path):
            rel_root = os.path.relpath(root, tmpdir)
            indent   = "  " + "  " * rel_root.count(os.sep)
            if rel_root != ".":
                print(f"  {indent}{os.path.basename(root)}/")
            for fname in files:
                fpath = os.path.join(root, fname)
                fsize = os.path.getsize(fpath) / 1024
                print(f"  {indent}  {fname:<30s}  {fsize:.1f} KB")
        print()

        # ── Numerical validation ──────────────────────────────────────────
        import PIL.Image

        # Create a synthetic PIL image
        img_np  = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(img_np)

        # coremltools prediction (macOS only)
        try:
            ct_output = mlmodel.predict({"input_image": pil_img})
            ct_logits = ct_output[list(ct_output.keys())[0]]

            # PyTorch reference (with same normalisation applied manually)
            img_float = img_np.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_norm = (img_float - mean) / std
            x_pt = torch.from_numpy(img_norm.transpose(2, 0, 1)[np.newaxis]).float()
            with torch.no_grad():
                pt_logits = model(x_pt).numpy()

            max_diff = float(np.max(np.abs(ct_logits - pt_logits)))
            print(f"  Numerical validation: max_err = {max_diff:.2e} "
                  f"{'✅' if max_diff < 0.01 else '⚠️ check preprocessing'}")
        except Exception as e:
            print(f"  Prediction demo: {e}")
            print("  (Note: coremltools prediction requires macOS)")
        print()

    except Exception as e:
        print(f"  Conversion demo: {e}")
        print()

else:
    CNN_REF = """
  IMAGETYPE CONVERSION REFERENCE:
  ─────────────────────────────────────────────────────────────────
  import coremltools as ct, torch

  traced = torch.jit.trace(model.eval(), torch.randn(1, 3, 224, 224))
  mlmodel = ct.convert(
      traced,
      inputs=[ct.ImageType(
          name="input_image",
          shape=(1, 3, 224, 224),
          scale=1.0 / (255.0 * 0.226),
          bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
          color_layout=ct.colorlayout.RGB,
          channel_first=True,
      )],
      minimum_deployment_target=ct.target.iOS16,
      convert_to="mlprogram",
  )
  mlmodel.save("MyModel.mlpackage")

  # On macOS, test with PIL:
  import PIL.Image
  img = PIL.Image.open("test.jpg").resize((224, 224))
  result = mlmodel.predict({"input_image": img})
"""
    print(CNN_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Convert a transformer with TensorType + dynamic shapes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Convert transformer: TensorType with dynamic seq len")
print("━" * 65)
print()

TRANSFORMER_REF = """
  TRANSFORMER CONVERSION WITH DYNAMIC SEQUENCE LENGTH
  ════════════════════════════════════════════════════════════════

  Many NLP models take variable-length token sequences.
  coremltools supports this via ct.Shape with RangeDim:

  import coremltools as ct
  from coremltools.converters.mil.input_types import RangeDim
  import torch, torch.nn as nn

  class TinyTransformer(nn.Module):
      def __init__(self, d=128, n_heads=4, n_layers=2, vocab=1000, n_cls=5):
          super().__init__()
          self.embed = nn.Embedding(vocab, d)
          self.pos_enc = nn.Parameter(torch.randn(1, 512, d) * 0.02)
          encoder_layer = nn.TransformerEncoderLayer(d, n_heads,
              dim_feedforward=d*4, batch_first=True, dropout=0.0)
          self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
          self.classifier = nn.Linear(d, n_cls)

      def forward(self, input_ids):
          S = input_ids.shape[1]
          x = self.embed(input_ids) + self.pos_enc[:, :S, :]
          x = self.encoder(x)          ; [B, S, d]
          return self.classifier(x.mean(dim=1))  ; [B, n_cls]

  model  = TinyTransformer().eval()
  # IMPORTANT: trace with FIXED shape first (TorchScript requirement):
  example_ids = torch.randint(0, 1000, (1, 64))  ; seq len = 64 for trace
  traced = torch.jit.trace(model, example_ids)

  # Then convert with DYNAMIC seq len declared:
  mlmodel = ct.convert(
      traced,
      inputs=[ct.TensorType(
          name="input_ids",
          shape=ct.Shape(shape=(1, RangeDim(1, 512))),  ; seq 1–512
          dtype=int,
      )],
      minimum_deployment_target=ct.target.iOS16,
      convert_to="mlprogram",
  )

  # Verify dynamic shapes work:
  for seq_len in [1, 32, 128, 512]:
      ids = np.random.randint(0, 1000, (1, seq_len)).astype(np.int32)
      result = mlmodel.predict({"input_ids": ids})
      print(f"  seq_len={seq_len:3d}: output shape = {result['var_N'].shape}")

  DYNAMIC SHAPE CONSTRAINTS:
  ─────────────────────────────────────────────────────────────────
  RangeDim(min, max):       any value in [min, max]
  RangeDim(1, -1):          any positive integer (no upper bound)
  EnumeratedShapes([64, 128, 256, 512]):  only these specific values
                             EnumeratedShapes triggers compilation of
                             4 separate optimised kernels, one per shape.
                             Faster than RangeDim for known discrete lengths.
  Static shape (integer):   single fixed value, most ANE-compatible.

  ENUMERATEDSHAPES EXAMPLE:
  ct.TensorType(
      name="input_ids",
      shape=ct.EnumeratedShapes(
          shapes=[(1, 64), (1, 128), (1, 256), (1, 512)]),
  )
  ; Core ML compiles 4 separate kernels and selects at runtime based on input.
  ; On ANE: each kernel is fully optimised for its specific shape.
"""
print(TRANSFORMER_REF)

if HAS_CT and HAS_TORCH:
    try:
        from coremltools.converters.mil.input_types import RangeDim

        class TinyTransformer(nn.Module):
            def __init__(self, d=64, n_heads=4, n_layers=1, vocab=500, n_cls=4):
                super().__init__()
                self.embed = nn.Embedding(vocab, d)
                self.pos   = nn.Parameter(torch.randn(1, 256, d) * 0.02)
                enc_layer  = nn.TransformerEncoderLayer(d, n_heads,
                    dim_feedforward=d*2, batch_first=True, dropout=0.0)
                self.enc   = nn.TransformerEncoder(enc_layer, n_layers)
                self.clf   = nn.Linear(d, n_cls)
            def forward(self, ids):
                S = ids.shape[1]
                x = self.embed(ids) + self.pos[:, :S, :]
                return self.clf(self.enc(x).mean(1))

        tf_model = TinyTransformer().eval()
        example_ids = torch.randint(0, 500, (1, 32))
        traced_tf = torch.jit.trace(tf_model, example_ids)

        print(f"  Converting TinyTransformer (d=64, 1 layer)...")
        t0 = time.perf_counter()
        mlmodel_tf = ct.convert(
            traced_tf,
            inputs=[ct.TensorType(
                name="input_ids",
                shape=ct.Shape(shape=(1, RangeDim(1, 256))),
                dtype=int,
            )],
            minimum_deployment_target=ct.target.iOS16,
            convert_to="mlprogram",
        )
        t_conv = (time.perf_counter() - t0) * 1000
        print(f"  Conversion time: {t_conv:.0f} ms")

        # Count MIL ops
        mil_prog = mlmodel_tf._get_mil_internal()
        if mil_prog is not None:
            all_ops = [op for block in mil_prog.functions["main"].blocks
                       for op in block.operations]
            op_types = {}
            for op in all_ops:
                op_types[op.op_type] = op_types.get(op.op_type, 0) + 1
            print(f"  MIL program: {len(all_ops)} total ops")
            top_ops = sorted(op_types.items(), key=lambda x: -x[1])[:8]
            for op_name, count in top_ops:
                print(f"    {op_name:<35s} × {count}")

        # Test with multiple sequence lengths
        try:
            print()
            print("  Testing dynamic sequence lengths:")
            for seq_len in [4, 16, 64]:
                ids = np.random.randint(0, 500, (1, seq_len)).astype(np.int32)
                result = mlmodel_tf.predict({"input_ids": ids})
                out_key = list(result.keys())[0]
                out_shape = np.array(result[out_key]).shape
                print(f"    seq_len={seq_len:3d}: output {out_shape} ✅")
        except Exception as e:
            print(f"    Prediction: {e} (requires macOS)")

        print()

    except Exception as e:
        print(f"  Transformer conversion: {e}")
        print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · MIL Program Inspection and Graph Surgery": {
        "description": (
            "Deep dive into the MIL (Model Intermediate Language) IR. "
            "Inspect every operation in a converted model's MIL program. "
            "Show MIL op types, input/output types, and value shapes. "
            "Perform graph surgery: insert, delete, and replace MIL ops. "
            "Demonstrate writing a model directly in MIL using the Builder API. "
            "Show how MIL passes transform the graph: GELU fusion, BN fusion."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  MIL PROGRAM INSPECTION AND GRAPH SURGERY")
print("=" * 65)
print()

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types as mil_types
    HAS_CT = True
    print(f"  coremltools {ct.__version__}")
except ImportError:
    HAS_CT = False
    print("  coremltools not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: MIL program structure and op inspection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — MIL program structure: ops, types, and shapes")
print("━" * 65)
print()

MIL_ANATOMY = """
  THE MIL PROGRAM — STRUCTURE WALKTHROUGH
  ════════════════════════════════════════════════════════════════

  A MIL program has the following structure:

  Program
  └── Function: "main"
      └── Block
          ├── input vars (graph inputs)
          ├── Op 1: const      → value: [W matrix]  type: tensor<fp16, [8,4]>
          ├── Op 2: const      → value: [bias]      type: tensor<fp16, [4]>
          ├── Op 3: linear     inputs: (x, weight=W, bias=b)
          │             output: tensor<fp16, [1, 4]>
          ├── Op 4: relu       inputs: (x=linear_out)
          │             output: tensor<fp16, [1, 4]>
          └── output vars (graph outputs)

  ACCESSING OPERATIONS:
    mil_prog = mlmodel._get_mil_internal()   ; get the MIL program
    func     = mil_prog.functions["main"]    ; main function
    block    = func.blocks[0]                ; first (only) block

    for op in block.operations:
        print(f"  {op.op_type:25s} → {[(k, str(v.shape)) for k,v in op.outputs.items()]}")

  ACCESSING AN OP'S INPUTS:
    for op in block.operations:
        if op.op_type == "conv":
            x_var    = op.inputs["x"]         ; input activation
            w_var    = op.inputs["weight"]     ; weight constant
            groups   = op.inputs["groups"]     ; attribute
            strides  = op.inputs["strides"]    ; [stride_h, stride_w]
            print(f"  Conv: x={x_var.shape}  w={w_var.shape}  strides={strides.val}")

  MIL VARIABLE TYPES:
    var.dtype:    fp32, fp16, int8, bool, int32, ...
    var.shape:    tuple of integers (or symbols for dynamic dims)
    var.val:      numpy array for constants (None for activations)
    var.name:     string identifier

  MIL OP TYPES — WHAT FUSING ACHIEVES:
    Before fusion: 8 separate ops for LayerNorm
      (reduce_mean, sub, reduce_mean, add, rsqrt, mul, mul, add)
    After fusion:  1 op
      (layer_norm)
    Impact: 1 ANE kernel call vs 8 separate kernel calls.
    The fused kernel runs entirely on-chip with no intermediate HBM writes.
"""
print(MIL_ANATOMY)

if HAS_CT and HAS_TORCH:
    # Build a model with LayerNorm to show fusion
    class WithLayerNorm(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc   = nn.Linear(d, d)
        def forward(self, x):
            return self.fc(self.norm(x))

    model  = WithLayerNorm(32).eval()
    dummy  = torch.randn(1, 8, 32)

    try:
        traced = torch.jit.trace(model, dummy)
        mlmodel = ct.convert(traced,
                              inputs=[ct.TensorType(shape=dummy.shape,
                                                     dtype=float)],
                              minimum_deployment_target=ct.target.iOS16,
                              convert_to="mlprogram")

        mil_prog = mlmodel._get_mil_internal()
        if mil_prog is not None:
            func  = mil_prog.functions["main"]
            block = func.blocks[0]
            ops   = [op for op in block.operations]

            # Count by type
            from collections import Counter
            type_counts = Counter(op.op_type for op in ops)

            print(f"  Model: Linear(32) + LayerNorm(32)")
            print(f"  MIL ops after conversion ({len(ops)} total):")
            for op_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"    {op_type:<35s} × {count}")
            print()

            # Show layer_norm if it was fused
            ln_ops = [op for op in ops if op.op_type == "layer_norm"]
            if ln_ops:
                op = ln_ops[0]
                print(f"  ✅ LayerNorm FUSED: found 'layer_norm' op")
                print(f"     input shape:  {list(op.inputs['x'].shape)}")
                axes = op.inputs.get('axes')
                if axes is not None and hasattr(axes, 'val'):
                    print(f"     axes:         {axes.val}")
            else:
                print("  ℹ️  layer_norm op not found — checking for component ops:")
                rmu = [op for op in ops if "reduce" in op.op_type]
                for op in rmu[:3]:
                    print(f"    {op.op_type}")
            print()

    except Exception as e:
        print(f"  MIL inspection: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Writing MIL directly with the Builder API
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Writing MIL directly with the @mb.program decorator")
print("━" * 65)
print()

if HAS_CT:
    try:
        # ── Example 1: simple MLP in MIL ──────────────────────────────────
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 8), dtype=ct.proto.MIL_pb2.DataType.Value("FLOAT32"))],
        )
        def mlp_program(x):
            """A 2-layer MLP: relu(x @ W1 + b1) @ W2 + b2"""
            rng = np.random.default_rng(42)
            W1  = mb.const(val=rng.normal(size=(8, 16)).astype(np.float32), name="W1")
            b1  = mb.const(val=np.zeros(16, dtype=np.float32), name="b1")
            W2  = mb.const(val=rng.normal(size=(16, 4)).astype(np.float32), name="W2")
            b2  = mb.const(val=np.zeros(4, dtype=np.float32), name="b2")

            h   = mb.linear(x=x, weight=W1, bias=b1, name="hidden")
            a   = mb.relu(x=h, name="activation")
            out = mb.linear(x=a, weight=W2, bias=b2, name="output")
            return out

        mlmodel_mil = ct.convert(mlp_program)
        print("  Hand-written MIL MLP (8→16→4) converted successfully ✅")

        # Run prediction
        try:
            x_test  = np.random.randn(1, 8).astype(np.float32)
            inp_key = mlmodel_mil.get_spec().description.input[0].name
            result  = mlmodel_mil.predict({inp_key: x_test})
            out_key = list(result.keys())[0]
            print(f"  Input shape:  {x_test.shape}")
            print(f"  Output shape: {np.array(result[out_key]).shape}")
        except Exception as e:
            print(f"  Prediction: {e} (requires macOS)")
        print()

        # ── Example 2: demonstrating key MIL ops ─────────────────────────
        MIL_OPS = """
  KEY MIL BUILDER OPS — COMMON PATTERNS
  ════════════════════════════════════════════════════════════════

  ATTENTION (transformer block):
    @mb.program(input_specs=[mb.TensorSpec((1, 64, 128))])
    def attention_block(x):
        # Multi-head self-attention (manual):
        # Project Q, K, V
        Wq = mb.const(val=Wq_init, name="Wq")
        q  = mb.matmul(x=x, y=Wq, name="q_proj")          ; [1, 64, 128]
        k  = mb.matmul(x=x, y=Wk, name="k_proj")
        v  = mb.matmul(x=x, y=Wv, name="v_proj")

        # Scaled dot-product attention (fuses to ANE sdpa kernel):
        scale = mb.const(val=np.float32(1.0 / (128**0.5)))
        out   = mb.scaled_dot_product_attention(
            query=q, key=k, value=v, scale=scale, name="sdpa")
        return out

  GELU ACTIVATION:
    y = mb.gelu(x=h, mode="TANH_APPROXIMATION")
    ; mode="EXACT":             exact GELU (slower)
    ; mode="TANH_APPROXIMATION": tanh-based approx (standard for transformers)
    ; Both versions fuse to a single ANE op.

  RESHAPE AND TRANSPOSE:
    # Reshape: -1 infers the dim
    flat = mb.reshape(x=x, shape=[1, -1], name="flatten")
    # Transpose: permute axes
    transposed = mb.transpose(x=x, perm=[0, 2, 1], name="swap_seq_dim")

  GATHERING (embedding lookup):
    embeddings = mb.const(val=embed_table)  ; [vocab, d_model]
    x_embed    = mb.gather(x=embeddings, indices=token_ids, axis=0)
    ; NOTE: gather with non-const indices falls back to CPU (not ANE).
    ; For inference: embed lookup is typically fast on CPU (sequential access).

  LAYER NORM (will be fused by the pass pipeline):
    y = mb.layer_norm(x=x, axes=[-1], gamma=gamma, beta=beta, epsilon=1e-5)

  SOFTMAX:
    probs = mb.softmax(x=logits, axis=-1)
"""
        print(MIL_OPS)

    except Exception as e:
        print(f"  MIL builder: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Graph surgery — inserting and replacing MIL ops
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Graph surgery: modify the MIL graph post-conversion")
print("━" * 65)
print()

GRAPH_SURGERY = """
  MIL GRAPH SURGERY — MODIFYING CONVERTED MODELS
  ════════════════════════════════════════════════════════════════

  After coremltools.convert(), you can modify the MIL program before
  saving. This is useful for:
    - Replacing unsupported ops with equivalent supported ones
    - Inserting custom preprocessing/postprocessing
    - Changing output nodes (e.g., add argmax for classification)
    - Fusing patterns the automatic passes missed

  GRAPH SURGERY PATTERN:
  ─────────────────────────────────────────────────────────────────
  from coremltools.converters.mil.mil.passes.defs.optimize import \
      AbstractGraphPass

  # Access and modify the MIL program:
  mil_prog = mlmodel._get_mil_internal()
  func     = mil_prog.functions["main"]

  # Use mb context to insert new ops:
  with func:    ; sets the current function context
      # Iterate over a copy (list) to allow mutation:
      for block in func.blocks:
          for op in list(block.operations):
              if op.op_type == "relu":
                  # Replace relu with gelu:
                  with block:
                      new_op = mb.gelu(x=op.inputs["x"],
                                       mode="TANH_APPROXIMATION",
                                       before_op=op)
                  # Replace all users of relu's output:
                  op.outputs[0].replace_uses_with(new_op.outputs[0])
                  block.remove_ops([op])

  # After surgery, re-save:
  mlmodel.save("model_modified.mlpackage")

  COMMON SURGERY RECIPES:
  ─────────────────────────────────────────────────────────────────
  1. Add argmax output (for classifiers without top-1 label):
     with func:
         for block in func.blocks:
             final_softmax = [op for op in block.operations
                              if op.op_type == "softmax"][-1]
             with block:
                 argmax = mb.reduce_argmax(
                     x=final_softmax.outputs[0],
                     axis=1,
                     keep_dims=False,
                     after_op=final_softmax)
             block.set_outputs([argmax.outputs[0]])

  2. Insert normalisation before model input:
     Input arrives as uint8 [0,255]; model expects float32 [0,1]:
     with func:
         for block in func.blocks:
             raw_input = func.inputs["x"]  ; the graph input
             with block, mb.select_op_ctx(None, before_op=block.operations[0]):
                 scale = mb.const(val=np.float32(1.0 / 255.0))
                 norm  = mb.real_div(x=raw_input, y=scale)
             ; Replace all downstream uses of raw_input with norm:
             raw_input.replace_uses_with(norm.outputs[0])

  3. Change input dtype (int32 → int64 for embedding lookup):
     with func:
         for block in func.blocks:
             for op in list(block.operations):
                 if op.op_type == "cast" and op.inputs["dtype"].val == "int64":
                     ; Update the input var type annotation
                     pass  ; complex; usually easier to fix in the source model
"""
print(GRAPH_SURGERY)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Quantisation — FP16, INT8, and Palettization": {
        "description": (
            "Apply every quantisation technique coremltools supports. "
            "Compress weights to FP16: measure size reduction and verify accuracy. "
            "Apply INT8 weight quantisation with per-channel granularity. "
            "Demonstrate 4-bit palettization with k-means centroids. "
            "Show mixed precision: keep sensitive layers at FP16. "
            "Benchmark latency and model size across all compression modes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  QUANTISATION — FP16, INT8, AND PALETTIZATION")
print("=" * 65)
print()

try:
    import coremltools as ct
    from coremltools import optimize
    HAS_CT = True
    print(f"  coremltools {ct.__version__}")
except ImportError:
    HAS_CT = False
    print("  coremltools not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Compression pipeline comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — All compression modes: sizes and accuracy impact")
print("━" * 65)
print()

COMPRESSION_GUIDE = """
  COREMLTOOLS COMPRESSION MODES — COMPLETE COMPARISON
  ════════════════════════════════════════════════════════════════

  Mode                  | Storage | Inference | Accuracy loss | Notes
  ─────────────────────────────────────────────────────────────────────
  FP32 (baseline)       | 1.0×    | 1.0×     | 0%            | Default
  FP16 weights          | 0.5×    | ~1.0×    | < 0.01%       | Lossless in practice
  INT8 sym (per-channel)| 0.25×   | ~1.5×+   | < 0.1%        | Needs calibration
  INT8 asym (per-tensor)| 0.25×   | ~1.5×+   | < 0.3%        | No calibration needed
  4-bit palettized (kmeans)| 0.125×| ~2×+   | 0.2–0.5%      | Requires fine-tuning
  4-bit palettized (uniform)| 0.125×| ~2×+ | 0.5–1.0%      | No fine-tuning
  2-bit palettized      | 0.0625× | ~3×+    | 1–3%           | Mainly for LLM weights
  INT8 act + weights    | 0.25×   | ~2–4×   | < 0.5%         | Spec 7, A17 Pro only

  (Inference speedup vs FP32 on ANE, approximate, model-dependent)

  WHICH TO USE:
    App Store constraint tight (< 50 MB):  4-bit palettized with k-means
    Need speed + accuracy (production CNN): INT8 per-channel
    Simple deployment without calibration:  FP16 (always safe)
    LLMs (large weights, small activation): 4-bit palettized weights
    Maximum ANE throughput (A17/M4 only):  INT8 activations + weights

  COMPRESSION API QUICK REFERENCE:
  ─────────────────────────────────────────────────────────────────
  import coremltools as ct

  # FP16 weight compression (applied at save time):
  mlmodel.save("model.mlpackage")   ; FP16 is default for mlprogram

  # OR explicitly:
  config = ct.optimize.coreml.OptimizationConfig(
      global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
          dtype=np.float16,
          mode="linear_symmetric",
      ))

  # INT8 per-channel:
  config = ct.optimize.coreml.OptimizationConfig(
      global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
          dtype=np.int8,
          mode="linear_symmetric",
          granularity="per_channel",
      ))
  int8_model = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)

  # 4-bit palettization:
  config = ct.optimize.coreml.OptimizationConfig(
      global_config=ct.optimize.coreml.OpPalettizerConfig(
          nbits=4,
          mode="kmeans",
          granularity="per_grouped_channel",
          group_size=32,
      ))
  paletted = ct.optimize.coreml.palettize_weights(mlmodel, config)
"""
print(COMPRESSION_GUIDE)

if HAS_CT and HAS_TORCH:
    # Build a model with enough weights to show compression clearly
    class CompressibleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*4*4, 10))
        def forward(self, x): return self.head(self.features(x))

    model  = CompressibleNet().eval()
    n_params = sum(p.numel() for p in model.parameters())
    dummy  = torch.randn(1, 3, 32, 32)

    print(f"  Model: CompressibleNet  params={n_params:,}  "
          f"FP32 size={n_params*4/1024:.1f} KB")
    print()

    # Convert to Core ML
    try:
        traced   = torch.jit.trace(model, dummy)
        mlmodel  = ct.convert(traced,
                               inputs=[ct.TensorType(shape=dummy.shape)],
                               minimum_deployment_target=ct.target.iOS16,
                               convert_to="mlprogram")

        tmpdir  = tempfile.mkdtemp()
        results = {}

        def save_and_measure(m, name):
            path = os.path.join(tmpdir, f"{name}.mlpackage")
            m.save(path)
            # Measure bin size
            bin_size = 0
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith(".bin") or f == "weights":
                        bin_size += os.path.getsize(os.path.join(root, f))
            total_size = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(path) for f in files
            )
            return total_size / 1024

        # Baseline FP16 (default for mlprogram)
        size_fp16 = save_and_measure(mlmodel, "fp16")
        results["FP16 (default)"] = size_fp16

        # INT8 per-channel
        try:
            int8_config = ct.optimize.coreml.OptimizationConfig(
                global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                    dtype=np.int8,
                    mode="linear_symmetric",
                    granularity="per_channel",
                ))
            int8_model = ct.optimize.coreml.linear_quantize_weights(
                mlmodel, int8_config)
            size_int8  = save_and_measure(int8_model, "int8")
            results["INT8 sym per-channel"] = size_int8
        except Exception as e:
            print(f"  INT8 quantisation: {e}")

        # 4-bit palettization (uniform, no kmeans for speed)
        try:
            pal_config = ct.optimize.coreml.OptimizationConfig(
                global_config=ct.optimize.coreml.OpPalettizerConfig(
                    nbits=4,
                    mode="uniform",
                ))
            pal_model  = ct.optimize.coreml.palettize_weights(mlmodel, pal_config)
            size_4bit  = save_and_measure(pal_model, "4bit")
            results["4-bit palettized (uniform)"] = size_4bit
        except Exception as e:
            print(f"  4-bit palettization: {e}")

        # Print comparison
        print(f"  Compression comparison:")
        print(f"  {'Mode':35s}  {'Size (KB)':>10s}  {'vs FP16':>8s}")
        print("  " + "-" * 58)
        baseline = results.get("FP16 (default)", 1.0)
        for name, size_kb in results.items():
            ratio = size_kb / baseline if baseline else 1.0
            print(f"  {name:<35s}  {size_kb:>10.1f}  {ratio:>8.2f}×")
        print()

        # Verify accuracy after INT8 quantisation
        if "INT8 sym per-channel" in results:
            try:
                x_val   = np.random.rand(1, 3, 32, 32).astype(np.float32)
                inp_key = mlmodel.get_spec().description.input[0].name

                fp16_pred = mlmodel.predict({inp_key: x_val})
                int8_pred = int8_model.predict({inp_key: x_val})

                fp16_arr = np.array(list(fp16_pred.values())[0])
                int8_arr = np.array(list(int8_pred.values())[0])
                max_diff = float(np.max(np.abs(fp16_arr - int8_arr)))
                argmax_same = np.argmax(fp16_arr) == np.argmax(int8_arr)

                print(f"  Accuracy: FP16 vs INT8 per-channel:")
                print(f"    Max L1 diff:      {max_diff:.4f}")
                print(f"    Argmax agreement: {'✅' if argmax_same else '❌'}")
            except Exception as e:
                print(f"  Accuracy test: {e} (requires macOS)")
        print()

    except Exception as e:
        print(f"  Compression demo: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Palettization deep dive
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Palettization: lookup tables for ANE-native INT4")
print("━" * 65)
print()

PALETTIZATION_GUIDE = """
  PALETTIZATION — HOW 4-BIT WEIGHT STORAGE WORKS
  ════════════════════════════════════════════════════════════════

  Traditional INT8:
    Each weight → one 8-bit integer that approximates the float value.
    Stored: 8 bits per weight.
    Interpretation: real_value = scale × (int8_val - zero_point)
    Scale is stored separately (1 float per tensor or per channel).

  Palettization (look-up table):
    2ⁿ centroids stored as float16 values (the "palette").
    Each weight → n-bit INDEX into the palette.
    Stored: n bits per weight + 2ⁿ × 2 bytes for the palette.

    For 4-bit, 256 output channels, kernel 3×3:
      Weights: 256 × 1 × 3 × 3 = 2304 values
      Storage: 2304 × 4 bits = 1152 bytes (indices)
               16 × 2 bytes  = 32 bytes (palette for 16 centroids)
      vs FP16: 2304 × 2 bytes = 4608 bytes
      Compression: 4608 / (1152 + 32) ≈ 3.9×

  PALETTE MODES:
    KMEANS (recommended):
      Run k-means clustering on the layer's weights.
      The 16 cluster centres become the palette.
      Weights assigned to nearest centre.
      Minimises quantisation error for this specific weight distribution.
      Cost: 1 k-means run per layer at conversion time (slow for large models).

    UNIFORM:
      Centroids evenly spaced between min_weight and max_weight.
      No clustering — fast conversion.
      Less accurate than k-means (palette not matched to weight distribution).
      Good for: fast prototyping, LLM weights (often uniformly distributed).

    UNIQUE (for sparse or discrete weights):
      Uses the unique values already in the weight tensor as centroids.
      Best when weights are already quantised or clustered.
      Not applicable to freshly-trained FP32 weights.

  PER-GROUPED-CHANNEL PALETTIZATION (Core ML 7+):
    Instead of one global palette for the whole layer, use separate
    palettes for groups of output channels:

    group_size=32: every 32 output channels share a palette.
    Each group has its own 16 centroids optimised for that group.
    Better accuracy than per-tensor palette.
    Slightly larger storage (more palettes).

    Example: conv with 256 output channels, group_size=32:
      256 / 32 = 8 palette groups
      8 × 16 × 2 bytes = 256 bytes of palette storage
      vs one global palette: 16 × 2 = 32 bytes
      Storage overhead: 7× more palette bytes (negligible vs weight savings).

  ANE NATIVE LOOKUP (A17 Pro / M4+, Core ML Spec 7):
    On A17 Pro and M4, the ANE can execute palettized ops natively:
      1. Load 4-bit index from SRAM.
      2. Lookup FP16 centroid from palette (in registers).
      3. Use FP16 centroid directly in MAC operation.
    No dequantisation by CPU required.
    Full 4-bit storage with FP16 compute accuracy.
    Throughput: ~2× vs FP16 weights (2× more weights fit in bandwidth).

  ACCURACY IMPACT (MobileNetV2, ImageNet top-1):
    FP16 baseline:         71.9%
    4-bit uniform:         71.0%  (-0.9%)
    4-bit k-means:         71.5%  (-0.4%)
    4-bit k-means per-group: 71.7%  (-0.2%)
    2-bit k-means:         69.8%  (-2.1%)  ← significant, requires QAT
"""
print(PALETTIZATION_GUIDE)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · ANE Dispatch, Compute Units, and Performance Profiling": {
        "description": (
            "Understand how Core ML dispatches to ANE vs GPU vs CPU. "
            "Configure compute units and measure the impact on latency. "
            "Show ANE compatibility requirements: op types and shape constraints. "
            "Benchmark the same model across CPU-only, CPU+GPU, and ALL compute units. "
            "Show the Instruments profiling workflow for identifying slow layers. "
            "Demonstrate the stateful model API for KV-cache LLM inference."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  ANE DISPATCH, COMPUTE UNITS, AND PERFORMANCE PROFILING")
print("=" * 65)
print()

try:
    import coremltools as ct
    HAS_CT = True
    print(f"  coremltools {ct.__version__}")
except ImportError:
    HAS_CT = False
    print("  coremltools not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: ANE compatibility requirements and dispatch logic
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — ANE compatibility: what runs on ANE vs CPU/GPU")
print("━" * 65)
print()

ANE_COMPAT = """
  APPLE NEURAL ENGINE COMPATIBILITY REQUIREMENTS
  ════════════════════════════════════════════════════════════════

  The ANE is not a general compute engine. It executes a fixed set
  of neural network operations on its proprietary dataflow hardware.
  Understanding what DOES and DOES NOT run on ANE is critical for
  achieving maximum performance.

  ✅ RUNS ON ANE (direct hardware support):
  ─────────────────────────────────────────────────────────────────
  mb.conv / mb.conv_transpose:   2D convolution with:
      - input channels multiple of 4 (strongly preferred)
      - output channels multiple of 8 (strongly preferred)
      - kernel size 1×1 or 3×3 (highly optimised)
      - dilation 1 or 2 (larger values may fall back)

  mb.linear:                     matrix multiply (y = x @ W.T + b)
      - Inner dimension multiple of 16 strongly preferred
      - Used for all transformer FFN and projection layers

  mb.layer_norm:                 layer normalisation (FUSED by pass)
  mb.batch_norm:                 batch norm (fused into conv weights)
  mb.group_norm:                 group normalisation
  mb.relu, mb.relu6:             clipped ReLU activations
  mb.sigmoid, mb.tanh:           classical activations
  mb.gelu(mode="TANH_APPROXIMATION"):  transformer activation
  mb.silu:                       Swish activation
  mb.add, mb.mul, mb.sub:        elementwise arithmetic
  mb.softmax:                    softmax (standard, last axis)
  mb.reshape, mb.transpose:      shape manipulation (many patterns)
  mb.concat:                     concatenation (most configs)
  mb.avg_pool, mb.max_pool:      pooling operations
  mb.scaled_dot_product_attention: full fused attention (Core ML 7+)

  ⚠️ PARTIAL ANE SUPPORT (depends on shape/config):
  ─────────────────────────────────────────────────────────────────
  mb.reduce_mean / sum:          axes that don't match ANE's HWC layout
                                  → some reductions fall to GPU
  mb.slice_by_index:             complex strides may fall back
  mb.upsample_bilinear:          some scale factors fall to GPU
  mb.lstm / mb.rnn:              depends on hidden size and sequence length

  ❌ DOES NOT RUN ON ANE (always CPU or GPU):
  ─────────────────────────────────────────────────────────────────
  mb.gather (non-const indices): embedding lookups → CPU
  mb.scatter:                    scatter updates → CPU
  mb.sort / mb.argsort:          sorting → CPU or GPU
  mb.top_k:                      top-k sampling → CPU
  Custom layers (Python/C++):    always CPU
  mb.while_loop:                 dynamic iteration → CPU
  mb.reduce_argmax/argmin:       argmax → CPU
  Large integer arithmetic:      int64 ops → CPU
  Very large sequence lengths:   may exceed ANE SRAM

  PRACTICAL IMPLICATIONS:
  ─────────────────────────────────────────────────────────────────
  Transformer inference profile:
    Embedding lookup (gather):    CPU  ~0.1 ms  (acceptable, sequential)
    Attention (sdpa):             ANE  ~2–5 ms  (fully fused in MIL)
    FFN (2× linear + gelu):       ANE  ~1–3 ms
    LayerNorm:                    ANE  ~0.1 ms
    Sampling (topk + sample):     CPU  ~0.5 ms  (not on ANE)
    Token projection (linear):    ANE  ~0.5 ms

  The non-ANE ops (gather, topk) are typically fast enough on CPU
  that total inference is still ANE-dominated (>85% of latency on ANE).
  Focus optimisation effort on the ANE-bound layers.

  SHAPE ALIGNMENT GUIDE:
  ─────────────────────────────────────────────────────────────────
  Conv2D optimal for ANE:
    - batch size = 1           (ANE optimised for single-sample inference)
    - input channels: ×4       (ANE vectorisation width)
    - output channels: ×8      (ANE output tile size)
    - spatial dims: ×4         (ANE spatial tile constraints)
    Example: (1, 32, 56, 56) with 32-output-channel conv → ANE optimal

  Linear optimal for ANE:
    - input features: ×16      (ANE matrix multiply alignment)
    - output features: ×16
    Example: Linear(512, 2048) → optimal. Linear(100, 37) → suboptimal.
"""
print(ANE_COMPAT)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Compute unit benchmarking
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Compute unit benchmarking: CPU vs ALL (with ANE/GPU)")
print("━" * 65)
print()

COMPUTE_UNIT_BENCH = """
  COMPUTE UNIT BENCHMARK RESULTS (representative, real hardware)
  ════════════════════════════════════════════════════════════════

  Test: MobileNetV3-Small, 224×224, batch=1, float16 weights
  Device: iPhone 15 Pro (A17 Pro)

  ComputeUnit          Latency     Power    Notes
  ─────────────────────────────────────────────────────────────────
  cpuOnly              45.3 ms     350 mW   BNNS, Neon SIMD, 6 cores
  cpuAndGPU            12.1 ms     800 mW   Metal MPS, mostly GPU
  cpuAndNeuralEngine   3.8 ms      120 mW   ANE dominant (85% ops)
  all (default)        3.6 ms      125 mW   ANE, slight optimization

  cpuOnly  →  12× slower than ANE, but 3× more power
  all      →  recommended for production (best latency + efficiency)
  cpuOnly  →  use for debugging, deterministic results, simulator

  Test: BERT-Base NLP, seq_len=64
  Device: M2 MacBook Air

  ComputeUnit          Latency
  ─────────────────────────────────────────────────────────────────
  cpuOnly              35 ms    (8 CPU cores, BNNS)
  cpuAndGPU            18 ms    (GPU handles matmuls)
  all                  8.5 ms   (ANE handles transformer layers)

  Test: Llama 3 8B 4-bit, 1 new token generation
  Device: iPhone 15 Pro (A17 Pro)

  ComputeUnit          Tokens/sec
  ─────────────────────────────────────────────────────────────────
  cpuOnly              < 1 tok/s   (too slow for use)
  all (ANE)            10–15 tok/s (ANE + GPU + CPU all contribute)

  IMPORTANT: Compute unit selection only matters on REAL DEVICE.
  The iOS Simulator runs ALL code on the Mac CPU (cpuOnly always).
  Never benchmark Core ML on the simulator!

  SWIFT API FOR COMPUTE UNITS:
  ─────────────────────────────────────────────────────────────────
  let config = MLModelConfiguration()
  config.computeUnits = .all              // ANE > GPU > CPU (default)
  config.computeUnits = .cpuOnly          // CPU only
  config.computeUnits = .cpuAndGPU        // skip ANE (rare edge case)
  config.computeUnits = .cpuAndNeuralEngine  // skip GPU

  let model = try! MyModel(configuration: config)
"""
print(COMPUTE_UNIT_BENCH)

if HAS_CT and HAS_TORCH:
    # Build a simple model and benchmark on CPU (macOS dev machine)
    class BenchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU6(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*4*4, 10)
            )
        def forward(self, x): return self.net(x)

    try:
        bench_model = BenchNet().eval()
        dummy       = torch.randn(1, 3, 32, 32)
        traced      = torch.jit.trace(bench_model, dummy)
        mlmodel     = ct.convert(traced,
                                  inputs=[ct.TensorType(shape=dummy.shape)],
                                  minimum_deployment_target=ct.target.iOS16,
                                  convert_to="mlprogram")

        x_np    = np.random.rand(1, 3, 32, 32).astype(np.float32)
        inp_key = mlmodel.get_spec().description.input[0].name
        REPS    = 200

        print(f"  Benchmarking on macOS (CPU only — ANE requires real iPhone/iPad):")
        print(f"  Model: BenchNet (32×32 input)")
        print()

        units_to_test = [
            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
            ("ALL",      ct.ComputeUnit.ALL),
        ]

        results = {}
        for unit_name, unit in units_to_test:
            try:
                m = ct.models.MLModel(mlmodel.get_spec(),
                                       compute_units=unit)
                # Warmup
                for _ in range(20): m.predict({inp_key: x_np})

                t0 = time.perf_counter()
                for _ in range(REPS): m.predict({inp_key: x_np})
                t_ms = (time.perf_counter() - t0) / REPS * 1000
                results[unit_name] = t_ms
                print(f"  {unit_name:<20s}: {t_ms:.3f} ms/inference")
            except Exception as e:
                print(f"  {unit_name}: {e}")

        if results:
            print()
            print("  Note: on macOS, ALL and CPU_ONLY differences are minor")
            print("        (both run on CPU; ANE only accelerates on iOS/iPadOS/M-series Mac)")
            print("        Full ANE speedup visible only via Xcode Instruments on device.")
        print()

    except Exception as e:
        print(f"  Benchmark: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Instruments profiling guide and stateful models
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Profiling with Instruments and stateful KV-cache models")
print("━" * 65)
print()

PROFILING_AND_STATEFUL = """
  INSTRUMENTS PROFILING — FINDING PER-LAYER BOTTLENECKS
  ════════════════════════════════════════════════════════════════

  Xcode Instruments → Core ML instrument template:
    1. Xcode → Product → Profile  (Cmd+I)
    2. Select "Core ML" instrument template
    3. Run app on PHYSICAL DEVICE (not simulator — no ANE)
    4. The timeline shows:
         Y-axis: compute unit (ANE / GPU / CPU)
         X-axis: time
         Each bar: one layer's execution

  READING THE PROFILE:
    Green bars  → ANE execution (fast, power-efficient)
    Blue bars   → GPU execution (medium)
    Red bars    → CPU execution (slowest for large layers)

  HEALTHY PROFILE (vision transformer):
    ████████████████████████ ANE  95% of time (attention + FFN + LN)
    ██ CPU  4% (embedding gather + sampling)
    █ GPU   1% (preprocessing)

  UNHEALTHY PROFILE (needs optimisation):
    ████ ANE  30% of time
    ███████████████████ CPU  65% of time ← something is forcing CPU
    █ GPU  5%

  COMMON CAUSES OF CPU FALLBACK:
    1. FP32 model (not quantised to FP16):
       Fix: use compress_to_fp16=True or convert with FP16 dtype
    2. Gather op in critical path:
       Fix: move embedding lookup outside Core ML model
    3. Batch size > 1 for some ops:
       Fix: use batch=1 or check if batching triggers GPU fallback
    4. Custom layer in the model:
       Fix: replace with equivalent MIL built-in ops
    5. Unsupported activation function:
       Fix: replace GELU exact with GELU tanh approximation
            mb.gelu(mode="TANH_APPROXIMATION") instead of mb.gelu(mode="EXACT")

  STATEFUL MODELS — KV-CACHE FOR LLM INFERENCE (iOS 18+)
  ════════════════════════════════════════════════════════════════

  iOS 18 (Core ML Spec 8) introduced STATEFUL MODELS — models that
  maintain internal state between prediction calls.
  This is primarily used for KV-cache in LLM inference.

  WITHOUT STATEFUL API (iOS 17 and earlier):
    Each decode step must pass the ENTIRE KV-cache as inputs/outputs.
    For Llama-3-8B at 2048 tokens:
      KV-cache size ≈ 0.5 GB (32 layers × 2 × 8 heads × 128 head_dim × 2048 tokens × FP16)
    This 0.5 GB TRANSFERS from CPU to ANE and back EVERY TOKEN.
    At 15 tokens/sec: 0.5 GB × 15 = 7.5 GB/sec memory bandwidth consumed by KV-cache alone.
    Result: slow, power-hungry, bandwidth-limited generation.

  WITH STATEFUL API (iOS 18):
    The KV-cache lives in ANE SRAM between decode steps.
    Each decode step sends only the NEW token embedding (~2 KB).
    The ANE reads K/V from on-chip SRAM, updates it, stores it back.
    ZERO memory transfer for the KV-cache between steps.
    Result: ~3–5× faster generation, ~50% less power.

  STATEFUL MODEL SWIFT API:
  ─────────────────────────────────────────────────────────────────
  // Load stateful model
  let model = try MLModel(contentsOf: modelURL, configuration: config)

  // Create a persistent state (lives as long as generation session)
  let state = try model.makeState()

  // First call: process prompt (prefill)
  let prefillInput = MyModelInput(inputIds: promptTokens, ...)
  let prefillOutput = try model.prediction(from: prefillInput,
                                            using: state)

  // Decode loop: each call updates KV-cache in state
  var nextToken = prefillOutput.lastToken
  for _ in 0..<maxNewTokens {
      let decodeInput = MyModelInput(inputIds: [nextToken], ...)
      let decodeOutput = try model.prediction(from: decodeInput,
                                               using: state)
      // The state holds updated KV-cache — not copied to CPU
      nextToken = decodeOutput.nextToken
      if nextToken == eosToken { break }
  }
  // state goes out of scope → KV-cache freed from ANE SRAM

  BUILDING A STATEFUL MODEL (coremltools):
  ─────────────────────────────────────────────────────────────────
  For current best practice, use Apple's ml-llm-convert tool:
  github.com/apple/ml-ane-transformers (reference ANE transformer)
  github.com/apple/ml-llm-eval (LLM evaluation)

  Or use MIL's mb.sdpa_with_kv_cache op directly:
    @mb.program(input_specs=[...])
    def llm_decode_step(input_ids, kv_cache):
        # embed, add positions...
        q = mb.linear(x=h, weight=Wq)
        k = mb.linear(x=h, weight=Wk)
        v = mb.linear(x=h, weight=Wv)
        # Update KV cache and run attention in one op:
        attn_out = mb.sdpa_with_kv_cache(
            query=q, key=k, value=v,
            kv_cache=kv_cache,  ; state variable
            ...)
        return attn_out, updated_kv_cache
"""
print(PROFILING_AND_STATEFUL)
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Core ML in the Ecosystem — Benchmarks, Swift API, and Connected Stack": {
        "description": (
            "Complete Core ML workflow summary with practical Swift integration patterns. "
            "Show the Xcode auto-generated Swift interface and how to use it. "
            "Benchmark Core ML vs PyTorch Mobile vs ONNX Runtime on device (representative). "
            "Demonstrate the Vision framework integration for camera pipelines. "
            "Show OTA model updates: download, compile, cache, and swap. "
            "Summarise Core ML's position in the full connected compiler stack."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  CORE ML IN THE ECOSYSTEM — BENCHMARKS, SWIFT, CONNECTED STACK")
print("=" * 65)
print()

try:
    import coremltools as ct
    HAS_CT = True
    print(f"  coremltools {ct.__version__}")
except ImportError:
    HAS_CT = False
    print("  coremltools not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Swift runtime API reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Swift runtime API: the on-device side")
print("━" * 65)
print()

SWIFT_API = """
  CORE ML SWIFT RUNTIME — COMPLETE REFERENCE
  ════════════════════════════════════════════════════════════════

  XCODE AUTO-GENERATED INTERFACE (from .mlpackage in project):
  ─────────────────────────────────────────────────────────────────
  When you add MyModel.mlpackage to Xcode, it generates MyModel.swift:

  // Typed input struct
  class MyModelInput: MLFeatureProvider {
      var input_image: CVPixelBuffer    // ImageType → CVPixelBuffer
      var input_text: MLMultiArray      // TensorType → MLMultiArray
      init(input_image: CVPixelBuffer, input_text: MLMultiArray) { ... }
  }

  // Typed output struct
  class MyModelOutput: MLFeatureProvider {
      var classLabel: String             // ClassifierConfig → top-1 label
      var classLabelProbs: [String: Double]  // all class probabilities
  }

  // Model class
  class MyModel {
      var model: MLModel
      init(configuration: MLModelConfiguration = .init()) throws
      func prediction(input: MyModelInput) throws -> MyModelOutput
      func predictions(inputs: [MyModelInput]) throws -> [MyModelOutput]
  }

  INFERENCE PATTERNS:
  ─────────────────────────────────────────────────────────────────
  // 1. Simple single inference (synchronous):
  let model = try! MyModel()
  let prediction = try! model.prediction(input: MyModelInput(image: ciImage))
  label.text = prediction.classLabel

  // 2. Asynchronous inference (iOS 16+, avoids blocking main thread):
  Task {
      let prediction = try await model.prediction(input: myInput)
      await MainActor.run { updateUI(prediction) }
  }

  // 3. Camera frame inference (real-time, 30fps):
  func captureOutput(_ output: AVCaptureOutput,
                     didOutput sampleBuffer: CMSampleBuffer, ...) {
      guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
      let input = MyModelInput(cameraFrame: pixelBuffer)
      Task {
          let result = try? await model.prediction(input: input)
          // pixelBuffer comes from camera directly — zero copy to ANE
      }
  }

  // 4. Vision framework integration (highest performance):
  let request = VNCoreMLRequest(model: vnModel) { req, err in
      let classifications = req.results as! [VNClassificationObservation]
      let top = classifications.max { $0.confidence < $1.confidence }!
      print(top.identifier, top.confidence)
  }
  request.imageCropAndScaleOption = .centerCrop
  // VNImageRequestHandler handles colour conversion + resize on GPU:
  let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
  try handler.perform([request])   // entire pipeline: GPU preproc → ANE

  // 5. Batch inference (useful for offline processing):
  let inputs  = images.map { MyModelInput(image: $0) }
  let outputs = try model.predictions(inputs: inputs)
  // Core ML may batch these into fewer ANE dispatches internally

  LOADING A DYNAMICALLY-DOWNLOADED MODEL:
  ─────────────────────────────────────────────────────────────────
  // Compile once, cache, reuse:
  func loadOrCompileModel(at packageURL: URL) async throws -> MLModel {
      let cacheURL = FileManager.default
          .urls(for: .cachesDirectory, in: .userDomainMask)[0]
          .appendingPathComponent("CompiledModel.mlmodelc")

      if !FileManager.default.fileExists(atPath: cacheURL.path) {
          let compiledURL = try await MLModel.compileModel(at: packageURL)
          try FileManager.default.copyItem(at: compiledURL, to: cacheURL)
          try? FileManager.default.removeItem(at: compiledURL)
      }

      let config = MLModelConfiguration()
      config.computeUnits = .all
      return try MLModel(contentsOf: cacheURL, configuration: config)
  }
"""
print(SWIFT_API)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Competitive comparison on Apple hardware
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Core ML vs alternatives on Apple hardware")
print("━" * 65)
print()

BENCHMARK_TABLE = """
  BENCHMARK: MOBILENETV3-SMALL 224×224, BATCH=1
  Device: iPhone 15 Pro (A17 Pro), iOS 17
  ════════════════════════════════════════════════════════════════

  Framework               Latency    Throughput   Power    Device
  ─────────────────────────────────────────────────────────────────
  Core ML (ALL)           3.6 ms     278 fps      ~120 mW  ANE+GPU+CPU
  Core ML (cpuAndGPU)     12 ms       83 fps      ~400 mW  GPU (Metal MPS)
  Core ML (cpuOnly)       45 ms       22 fps      ~350 mW  CPU (BNNS, 6 cores)
  ONNX Runtime (CoreML EP)3.8 ms     263 fps      ~125 mW  Via Core ML
  PyTorch Mobile          38 ms       26 fps      ~420 mW  CPU (XNNPACK)
  TFLite (cpuOnly)        42 ms       24 fps      ~380 mW  CPU (XNNPACK)
  TFLite (GPU delegate)   14 ms       71 fps      ~450 mW  GPU (Metal)

  TAKEAWAYS:
  ─────────────────────────────────────────────────────────────────
  1. Core ML ALL is 10× faster than PyTorch Mobile (ANE vs XNNPACK)
  2. Core ML ALL uses 3× LESS power than GPU alternatives
  3. ONNX Runtime with CoreML EP ≈ Core ML directly (it IS Core ML under the hood)
  4. TFLite GPU delegate is competitive with Core ML cpuAndGPU (same Metal)
  5. PyTorch Mobile / ExecuTorch cannot access ANE

  BENCHMARK: BERT-BASE, SEQ=64, BATCH=1
  Device: M2 MacBook Air, macOS 14
  ════════════════════════════════════════════════════════════════

  Framework               Latency    Notes
  ─────────────────────────────────────────────────────────────────
  Core ML (ALL, FP16)     8.5 ms     ANE handles all transformer layers
  Core ML (cpuOnly)       35 ms      8 CPU cores, BNNS
  ONNX Runtime (CPU)      28 ms      oneDNN... wait, this is Apple.
  ONNX Runtime (CPU)      28 ms      Accelerate/BNNS under the hood
  PyTorch (MPS)           18 ms      GPU matmuls via Metal
  PyTorch (cpu)           55 ms      CPU PyTorch, no BNNS

  NOTES:
    On Apple Silicon: ONNX Runtime uses Core ML EP internally.
    Core ML ALL ≈ 3–4× faster than CPU-only alternatives.
    The ANE advantage is most dramatic for transformer architectures
    (where attention and FFN are ANE-native ops).

  BENCHMARK: LLAMA 3.1 8B, INT4, 1 TOKEN GENERATION
  Device: iPhone 15 Pro (A17 Pro)
  ════════════════════════════════════════════════════════════════

  Method                  Tokens/sec   Model size   Notes
  ─────────────────────────────────────────────────────────────────
  Core ML (iOS 18 state)  12–15 tok/s  ~4 GB        Stateful KV-cache on ANE
  Core ML (iOS 17 KV copy) 4–6 tok/s  ~4 GB        KV-cache copied each step
  llama.cpp (CPU only)    < 1 tok/s   ~4 GB        CPU-only, no ANE
  OpenVINO (NPU)          ~20 tok/s   ~4 GB        Lunar Lake NPU (PC, not iOS)
"""
print(BENCHMARK_TABLE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack summary and full reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Core ML in the connected compiler stack")
print("━" * 65)
print()

STACK = """
  CORE ML IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                      │
  │  Core ML CPU path uses Accelerate/BNNS, compiled with Apple Clang   │
  │  (LLVM-based). ARM Neon SIMD intrinsics are LLVM-generated.          │
  │  coremltools itself is Clang-compiled.                               │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                      │
  │  coremltools uses MLIR for MIL pass infrastructure.                  │
  │  The MIL lowering pipeline is MLIR-based.                            │
  │  MIL's block/region structure mirrors MLIR's dialect system.        │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 05/06: XLA / OpenXLA                                          │
  │  JAX models reach Core ML via ONNX (recommended).                   │
  │  XLA itself does not target Core ML or ANE.                          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                       │
  │  TVM can target Metal GPU on Apple Silicon.                          │
  │  TVM CANNOT target ANE — Core ML required for ANE.                  │
  │  TVM + Metal: custom GPU kernels; Core ML: production ANE inference. │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 10: ONNX                                                      │
  │  ONNX is Core ML's most widely-tested import format.                │
  │  coremltools.convert() accepts ONNX opsets 7–19.                    │
  │  ONNX Runtime on Apple Silicon routes through Core ML internally.   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 11: TFLite                                                    │
  │  TFLite GPU on iOS uses Metal MPS — the same GPU as Core ML.        │
  │  For maximum iOS performance: always prefer Core ML (ANE access).   │
  │  TFLite is better for cross-platform (Android + iOS) codebases.    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 12: OpenVINO                                                  │
  │  OpenVINO = Intel hardware. Core ML = Apple hardware. No overlap.   │
  │  ONNX bridges them: one ONNX model → both OV IR and .mlpackage.    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 13: Core ML (THIS MODULE)                                    │
  │  .mlpackage format (MIL graph + FP16/INT4/INT8 weights)             │
  │  coremltools conversion + MIL passes                                 │
  │  Compute units: ANE (fastest) > GPU (Metal) > CPU (BNNS)            │
  │  Quantisation: FP16 / INT8 / 4-bit palettized / mixed precision     │
  │  Stateful models (iOS 18): KV-cache on ANE for LLM inference        │
  │  Vision framework: camera → Metal → ANE, zero CPU copies            │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Task                         │ API / Tool                        │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Convert PyTorch → Core ML    │ ct.convert(traced, ...)           │")
print("  │ Convert ONNX → Core ML       │ ct.convert(onnx_model, ...)       │")
print("  │ Convert TF → Core ML         │ ct.convert(saved_model, ...)      │")
print("  │ Image input (camera-ready)   │ ct.ImageType(scale=, bias=, ...)  │")
print("  │ Tensor input with dyn shape  │ ct.TensorType(ct.Shape(RangeDim)) │")
print("  │ Dynamic seq len              │ ct.EnumeratedShapes([64, 128])     │")
print("  │ Classifier output            │ ct.ClassifierConfig(label_list)    │")
print("  │ Save model                   │ mlmodel.save('M.mlpackage')        │")
print("  │ FP16 compression (default)   │ ct.convert(..., convert_to='mlprogram') │")
print("  │ INT8 quantisation            │ ct.optimize.coreml.linear_quantize_weights() │")
print("  │ 4-bit palettization          │ ct.optimize.coreml.palettize_weights() │")
print("  │ Mixed precision              │ OptimizationConfig(op_name_configs={}) │")
print("  │ View MIL program             │ mlmodel._get_mil_internal()        │")
print("  │ Write MIL directly           │ @mb.program + mb.conv/linear/...  │")
print("  │ Load model (Swift)           │ let m = try MyModel()              │")
print("  │ Sync prediction (Swift)      │ try model.prediction(input: x)    │")
print("  │ Async prediction (Swift)     │ try await model.prediction(...)   │")
print("  │ Compute units (Swift)        │ config.computeUnits = .all        │")
print("  │ Vision integration           │ VNCoreMLRequest(model: vnModel)   │")
print("  │ Stateful model (Swift)       │ let state = try model.makeState() │")
print("  │ OTA model compile            │ await MLModel.compileModel(at:)   │")
print("  │ On-device update             │ MLUpdateTask(forModelAt:...)       │")
print("  │ Profile on device            │ Xcode Instruments → Core ML       │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ coremltools docs             │ coremltools.readthedocs.io         │")
print("  │ Core ML framework docs       │ developer.apple.com/documentation/coreml │")
print("  │ WWDC Core ML sessions        │ developer.apple.com/videos (search CoreML) │")
print("  │ MIL docs                     │ coremltools.readthedocs.io/mil    │")
print("  │ Model compression guide      │ coremltools.readthedocs.io/optimize │")
print("  │ ANE transformer reference    │ github.com/apple/ml-ane-transformers │")
print("  │ LLM convert tool             │ github.com/apple/ml-llm-convert   │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()

if HAS_CT and HAS_TORCH:
    print("  Runtime verification:")
    try:
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
        dummy = torch.randn(1, 8)
        traced = torch.jit.trace(m, dummy)
        mlm = ct.convert(traced, inputs=[ct.TensorType(shape=dummy.shape)],
                          minimum_deployment_target=ct.target.iOS16,
                          convert_to="mlprogram")

        spec = mlm.get_spec()
        inp_name = spec.description.input[0].name
        out_name = spec.description.output[0].name

        try:
            x_v = np.random.rand(1, 8).astype(np.float32)
            result_ct = mlm.predict({inp_name: x_v})
            ct_arr = np.array(list(result_ct.values())[0])

            with torch.no_grad():
                pt_arr = m(torch.from_numpy(x_v)).numpy()

            err = float(np.max(np.abs(ct_arr - pt_arr)))
            print(f"    Linear(8→16→4) ReLU: PyTorch vs CoreML max_err={err:.2e} "
                  f"{'✅' if err < 0.01 else '⚠️'}")
        except Exception as e:
            print(f"    Prediction (requires macOS): {e}")

        print(f"    Spec version: {spec.specificationVersion}")
        print(f"    Model type: {spec.WhichOneof('Type')}")
        print(f"    Input:  {inp_name}")
        print(f"    Output: {out_name}")

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