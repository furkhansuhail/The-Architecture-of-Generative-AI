"""
TFLite — TensorFlow Lite: Mobile and Edge ML Inference
========================================================

TensorFlow Lite (TFLite) is Google's purpose-built inference framework
for mobile devices, embedded systems, and edge hardware. Where TensorFlow
is optimised for training on servers and GPUs, TFLite is optimised for
inference under the hardest possible constraints: kilobytes of RAM, milliwatts
of power, tens of milliseconds of latency — on hardware that may have no
operating system, no heap allocator, and no floating-point unit.

TFLite is not simply TensorFlow with some features removed. It is a
completely separate stack, designed from first principles around the
requirements of deployment on constrained silicon:

    FILE FORMAT:   FlatBuffers (not protobuf). Schema-compiled, zero-copy,
                   no heap allocation required to parse a .tflite file.

    RUNTIME:       A portable C++ interpreter with a fixed-size memory arena.
                   No dynamic allocation during inference. Preallocates the
                   exact peak memory required at model load time.

    QUANTISATION:  INT8 weight and activation quantisation is a first-class
                   citizen, not an afterthought. The converter, operator
                   kernels, and delegate backends are all quantisation-aware.

    DELEGATES:     A pluggable hardware abstraction layer. NNAPI (Android NPU),
                   GPU (OpenCL/Metal), CoreML (Apple ANE), Hexagon (Qualcomm DSP),
                   and XNNPACK (SIMD CPU) all plug in as delegates.

    MICRO:         TFLite Micro strips the runtime further for microcontrollers
                   with no OS, no dynamic memory, and < 256KB of SRAM.

In 2023, Google rebranded TFLite as LiteRT (Lite RunTime) to signal a more
hardware-agnostic future, but the .tflite format, the converter API, and the
interpreter API remain backward-compatible and widely deployed.

In the connected stack:
    LLVM       (module 01) ← XNNPACK and some delegate backends use LLVM-compiled kernels
    MLIR       (module 02) ← TFLite converter pipeline is entirely MLIR-based
    XLA        (module 05) ← JAX models reach TFLite via StableHLO → MLIR → TFLite
    OpenXLA    (module 06) ← IREE can target the same hardware as TFLite delegates
    StableHLO  (module 07) ← TFLite's converter ingests StableHLO as one input format
    TVM        (module 08) ← TVM competes with TFLite on edge; microTVM targets MCUs
    ONNX       (module 10) ← ONNX ↔ TFLite conversion bridges the two ecosystems
    TFLite     (this)      ← the Google-ecosystem mobile/edge inference runtime

"""

import textwrap
import re

TOPIC_NAME   = "TFLite — TensorFlow Lite: Mobile and Edge ML Inference"
DISPLAY_NAME = "12 · TFLite"
ICON         = "📱"
SUBTITLE     = "FlatBuffers Format, INT8 Quantisation, Delegates, and TFLite Micro"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY TFLITE EXISTS: THE MOBILE DEPLOYMENT PROBLEM

### The Constraints of Deployment at the Edge

    Running a model in research and running it on a device people carry in
    their pocket are radically different engineering problems. The constraints
    of mobile and edge deployment are unlike anything in the data centre:

    MEMORY:
        Server GPU:   40–80 GB HBM + unlimited system RAM
        Mobile phone: 6–12 GB LPDDR5, shared with OS and all apps
                      ML inference typically limited to 100–500 MB
        MCU (microcontroller): 64 KB – 2 MB SRAM total
                               No heap. No malloc. No dynamic allocation.

    COMPUTE:
        Server GPU:   312 TFLOPS BF16 (A100)
        Mobile SoC:   2–10 TOPS INT8 (dedicated NPU like Google Tensor, Apple ANE)
                      0.1–1 TFLOPS FP32 GPU (Adreno, Mali, Apple GPU)
        MCU:          0.001–0.1 GFLOPS FP32 (Cortex-M4 at 168 MHz)
                      No FPU on smallest MCUs — FP32 requires software emulation.

    POWER:
        Server GPU:   300–700 W continuous power budget
        Mobile:       2–5 W average, 10 W peak — battery runs out in hours at peak
        MCU:          1–100 mW — battery-powered IoT devices run for months/years

    LATENCY:
        Server:       high throughput batch inference acceptable
        Mobile:       < 30 ms for interactive features (camera, AR, speech)
        MCU:          varies, but often < 10 ms for keyword spotting / gesture

    These constraints do not just require optimisation. They require a
    fundamentally different architecture from server-side inference engines.

### Why Standard TensorFlow Cannot Run on Mobile

    TensorFlow at training scale requires:
        Protocol Buffers for model serialisation: requires heap allocation to parse.
        Dynamic computation graphs: require heap allocation per operation.
        Python runtime: not available on Android/iOS at inference time.
        CUDA and cuDNN: unavailable on mobile silicon.
        Eager execution framework: per-op Python dispatch overhead unacceptable.

    The TF SavedModel format:
        A directory containing protocol buffer files and variables.
        Parsing requires multiple megabytes of working memory.
        Loading time measured in seconds (unacceptable for app startup).
        Assumes a standard OS file system (not available on MCUs).

    The fundamental problem: TF was designed for machines where memory,
    compute, and power are abundant. None of those assumptions hold at the edge.

### TFLite's Founding Design Decisions

    TFLite launched in May 2017. Every design decision was made in response
    to the constraints above:

    DECISION 1 — FlatBuffers instead of Protocol Buffers:
        FlatBuffers are ZERO-COPY: the model file can be read directly
        from flash storage without any heap allocation or parsing step.
        The .tflite file IS the model data structure — no deserialisation.
        Startup time: microseconds (mmap the file) vs seconds (parse protos).

    DECISION 2 — Fixed-size memory arena:
        TFLite pre-computes the MAXIMUM memory needed for all intermediate
        tensors at model load time. A single large allocation (the "arena")
        satisfies all intermediate buffers.
        After load: ZERO allocations during inference. No malloc, no GC.
        This makes inference latency DETERMINISTIC — critical for real-time use.

    DECISION 3 — INT8 quantisation as a first-class feature:
        4× memory reduction (float32 → int8).
        2–4× speedup on CPUs with SIMD integer multiply (ARM NEON, DSPs).
        NPUs (Google Tensor, Qualcomm Hexagon) are INT8-native — FP32
        would require dequantisation before every NPU operation.
        The quantisation scheme (scale + zero_point) is baked into the format.

    DECISION 4 — Delegate hardware abstraction:
        Hardware varies wildly: every phone has a different NPU, GPU, DSP.
        The delegate API lets hardware vendors plug their accelerator into the
        TFLite runtime without modifying any core code.
        The interpreter falls back to CPU for any op the delegate doesn't support.

    DECISION 5 — Minimal C++ runtime, no framework dependency:
        The TFLite interpreter has no dependency on TensorFlow proper.
        It is a standalone ~500KB C++ library (TFLite Micro: ~20KB).
        Ships as a .so on Android, a .framework on iOS, or a bare .a for MCUs.


##### PART 2 — THE FLATBUFFERS FORMAT: THE .TFLITE SCHEMA

### FlatBuffers vs Protocol Buffers

    ONNX uses Protocol Buffers (protobuf) for serialisation.
    TFLite uses FlatBuffers. The difference matters enormously for
    embedded deployment.

    PROTOCOL BUFFERS (protobuf):
        Serialisation:  explicit serialise-to-bytes step
        Deserialisation: must parse the entire binary, allocating objects
        Access:          through generated accessor classes
        Memory:          the binary + the parsed object tree (2× storage)
        Startup:         parse overhead proportional to model size

    FLATBUFFERS:
        Serialisation:   write directly to a binary with schema-defined layout
        Deserialisation: NONE. Access data directly via memory-mapped pointers.
        Access:          via generated accessor functions that compute byte offsets
        Memory:          the binary only (0× overhead)
        Startup:         mmap() the file, compute a single root offset — done

    ZERO-COPY in practice:
        FlatBuffer access: model->subgraphs()->Get(0)->operators()->Get(i)
        This computes: base_ptr + offset_table[0] + offset_table[0][i]
        No allocation, no copying, no parsing. Just pointer arithmetic.
        The flash storage of a microcontroller IS the model's memory.

### The .tflite FlatBuffer Schema

    The TFLite schema (schema.fbs) defines these key tables:

    Model (the top-level table):
        version:          uint    schema version (currently 3)
        operator_codes:   [OperatorCode]    the set of distinct ops used
        subgraphs:        [SubGraph]        computation graphs (usually 1)
        description:      string            human-readable model description
        buffers:          [Buffer]          raw data: weights, biases, constants
        metadata:         [Metadata]        key-value metadata (input range, etc.)
        signature_defs:   [SignatureDef]    named entry points (for multi-signature models)

    SubGraph:
        tensors:          [Tensor]          all tensors (inputs, outputs, intermediates)
        inputs:           [int]             indices into tensors[] for graph inputs
        outputs:          [int]             indices into tensors[] for graph outputs
        operators:        [Operator]        the ops in execution order
        name:             string            subgraph name

    Tensor:
        shape:            [int]             dimension sizes
        type:             TensorType        FLOAT32, INT8, INT16, BOOL, STRING, ...
        buffer:           int               index into Model.buffers (0 = no data)
        name:             string            debug name
        quantization:     QuantizationParameters  { scale, zero_point, min, max }
        shape_signature:  [int]             -1 for dynamic dimensions
        is_variable:      bool              mutable state (RNN hidden states)

    Operator:
        opcode_index:     int               index into Model.operator_codes
        inputs:           [int]             indices into SubGraph.tensors
        outputs:          [int]             indices into SubGraph.tensors
        builtin_options:  BuiltinOptions    op-specific config (union type)
        custom_options:   [byte]            flexbuffer for custom op config

    Buffer:
        data:             [byte]            raw little-endian tensor data
        offset:           ulong             offset into external file (optional)
        size:             ulong             size in bytes (for external buffers)

    OperatorCode:
        builtin_code:     BuiltinOperator   enum value (e.g., CONV_2D = 3)
        custom_code:      string            name for custom ops
        version:          int               op version (for backward compatibility)

### The Quantisation Parameters Table

    Every tensor in a quantised TFLite model carries quantisation metadata:

    QuantizationParameters:
        min:          [float]   per-channel minimum observed value (calibration)
        max:          [float]   per-channel maximum observed value (calibration)
        scale:        [float]   quantisation scale (1 per tensor, or per-channel)
        zero_point:   [int64]   quantisation zero point

    The mathematical relationship:
        real_value = (quantised_int8 - zero_point) × scale
        quantised   = clamp(round(real_value / scale) + zero_point, -128, 127)

    PER-TENSOR quantisation:    one scale and one zero_point for the whole tensor.
    PER-CHANNEL quantisation:   one scale and zero_point PER OUTPUT CHANNEL.
                                Better accuracy for weights; kernels are more complex.

    Signed INT8:  zero_point ∈ [-128, 127], range [-128, 127]
                  Used for weights and activations after ReLU6 calibration.
    Unsigned UINT8: zero_point ∈ [0, 255],  range [0, 255]
                   Legacy; still used for some activation quantisation.
    INT16 activations: higher precision for some sensitive ops (LSTM cells).

### Accessing the Schema Directly

    The Python TFLite Interpreter wraps the FlatBuffer. But you can also
    read the raw schema directly using the generated Python FlatBuffers bindings:

        import flatbuffers
        from tflite.Model import Model as TFLiteModel

        with open("model.tflite", "rb") as f:
            buf = f.read()

        buf_bytes = bytearray(buf)
        model = TFLiteModel.GetRootAs(buf_bytes, 0)

        print(f"Version: {model.Version()}")
        print(f"Subgraphs: {model.SubgraphsLength()}")
        sg   = model.Subgraphs(0)
        print(f"Operators: {sg.OperatorsLength()}")
        print(f"Tensors:   {sg.TensorsLength()}")

        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            print(f"  {t.Name().decode():30s}  {[t.Shape(j) for j in range(t.ShapeLength())]}")


##### PART 3 — THE TFLITE CONVERTER: FROM TF/PYTORCH/JAX TO .TFLITE

### The Converter's Job

    The TFLite converter takes a trained model in any of several source
    formats and produces a .tflite binary. This involves:
        1. Parsing the source format into MLIR (the converter's internal IR).
        2. Running graph optimisation passes (constant folding, fusion, etc.).
        3. Optionally quantising the graph (with or without calibration data).
        4. Lowering the MLIR graph to TFLite's builtin operator set.
        5. Handling unsupported ops (via Select TF Ops or custom ops).
        6. Serialising to the FlatBuffer schema.

### The MLIR-Based Converter Pipeline

    TFLite's converter was fully rewritten in MLIR (2020–2021). It is now
    a cascade of MLIR dialects and passes:

    TF SavedModel / Keras / concrete function
        ↓  TF importer (tf.saved_model → TF dialect MLIR)
    TF dialect (tf.Conv2D, tf.MatMul, tf.Relu, ...)
        ↓  --tf-executor-to-functional-conversion
        ↓  --tf-shape-inference
        ↓  --tf-standard-pipeline (constant folding, DCE, CSE)
    Optimised TF dialect
        ↓  --tf-lower-to-tflite-dialect
    TFLite dialect (tfl.conv_2d, tfl.fully_connected, tfl.relu, ...)
        ↓  --tfl-legalize-to-tflite (check op support, insert QDQ nodes)
        ↓  --tfl-post-quantization (fuse Q/DQ nodes, optimise quantised graph)
    Quantised TFLite dialect
        ↓  FlatBuffer serialiser
    model.tflite binary

    Each arrow is one or more MLIR passes (using MLIR's PassManager).
    The TF dialect → TFLite dialect lowering is the critical step that
    maps TF's rich op set (2000+ ops) to TFLite's smaller builtin set (~150 ops).

### Input Formats

    SOURCE 1 — TensorFlow SavedModel (most common):
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model("/path/to/saved_model/")
        tflite_model = converter.optimize(converter)
        with open("model.tflite", "wb") as f: f.write(tflite_model)

    SOURCE 2 — Keras model (in-memory):
        model = tf.keras.applications.MobileNetV3Small()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

    SOURCE 3 — TF concrete function (fine-grained control):
        @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
        def serving_fn(x): return model(x, training=False)

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [serving_fn.get_concrete_function()], model)

    SOURCE 4 — JAX / StableHLO (new path, TF 2.15+):
        The MLIR-based converter can ingest StableHLO directly:
        converter = tf.lite.TFLiteConverter.experimental_from_jax(
            serving_fn, [[("input", sample_input)]])

    SOURCE 5 — PyTorch (via ai-edge-torch, Google's library):
        import ai_edge_torch
        import torch

        class MyModel(torch.nn.Module):
            def forward(self, x): ...

        model  = MyModel().eval()
        sample = (torch.randn(1, 3, 224, 224),)
        edge_model = ai_edge_torch.convert(model, sample)
        edge_model.export("model.tflite")

    SOURCE 6 — ONNX (via onnx-tf):
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load("model.onnx")
        tf_rep     = prepare(onnx_model)  # ONNX → TF SavedModel
        tf_rep.export_graph("saved_model/")
        # Then convert saved_model/ → .tflite as above

### Converter Optimisation Options

    The converter accepts optimisation targets that control the output:

    NO OPTIMISATION (float32 model):
        converter.optimizations = []
        tflite_model = converter.convert()
        # Pure float32. Largest size, highest accuracy, most compatible.

    DEFAULT OPTIMISATION (dynamic range quantisation):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Quantises weights to INT8; activations remain float32 at runtime.
        # 4× size reduction, no calibration data needed.
        # Activations dequantised to float for each kernel call.
        # Speed: modest (1.5–2× over float32 on CPU with NEON).

    FULL INT8 QUANTISATION (static quantisation):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        # Both weights AND activations are INT8. Maximum speed on NPUs.
        # Requires a calibration dataset (100–1000 representative samples).

    FP16 QUANTISATION:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        # Weights quantised to FP16; activations remain float32.
        # 2× size reduction vs float32.
        # GPU delegate can execute natively in FP16 (faster on mobile GPU).

    INT16 ACTIVATIONS (high accuracy INT8):
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        # Activations in INT16, weights in INT8.
        # Better accuracy than pure INT8 for sensitive ops (LSTM, RNN cells).

### Representative Dataset for Static Quantisation

    Static quantisation requires CALIBRATION: running a small dataset through
    the model to measure the range of each activation.

    A representative_dataset is a generator that yields dictionaries
    mapping input tensor names to numpy arrays:

        def representative_dataset():
            for i in range(200):     # 100-1000 samples is typical
                sample = load_sample(i)
                sample = preprocess(sample).astype(np.float32)
                yield {"input_1": sample[np.newaxis, ...]}   # add batch dim

        converter.representative_dataset = representative_dataset

    What calibration computes for each activation tensor:
        min_val: smallest value seen across all calibration samples
        max_val: largest value seen across all calibration samples
        scale   = (max_val - min_val) / 255.0
        zero_pt = round(-min_val / scale)   clamped to [-128, 127]

    Calibration quality significantly affects accuracy:
        Too few samples (< 50): poor range estimation → accuracy drop
        Unrepresentative samples: model encounters out-of-range values at inference
        Best practice: use 100–1000 samples from the same distribution as inference data


##### PART 4 — QUANTISATION IN DEPTH: ARITHMETIC, SCHEMES, AND ACCURACY

### The Mathematics of Uniform Affine Quantisation

    TFLite uses UNIFORM AFFINE quantisation — the quantisation scheme
    described in the Gemmlowp paper and implemented across all TFLite kernels.

    The core relationship:
        real_value   ≈ scale × (quantised_value − zero_point)
        quantised_val = clamp( round(real_value / scale) + zero_point,
                                qmin, qmax )

    For signed INT8:   qmin = -128, qmax = 127
    For unsigned UINT8: qmin = 0,   qmax = 255

    Calculating scale and zero_point from observed range [r_min, r_max]:
        scale     = (r_max − r_min) / (qmax − qmin)
        zero_point = qmin − round(r_min / scale)
        zero_point = clamp(zero_point, qmin, qmax)

    THE ZERO_POINT ASYMMETRY:
        Asymmetric quantisation (zero_point ≠ 0 typically):
            Uses the full INT8 range [-128, 127] for the actual value range.
            Better for activations after ReLU (range [0, ∞) needs mapping to [-128, 127]).
            zero_point = -128 maps to 0.0 in real value.
            The hardware must perform: result = scale × (q − zero_point).

        Symmetric quantisation (zero_point = 0, forced):
            Real 0.0 always maps to integer 0.
            Arithmetic simplifies: result = scale × q (no subtraction needed).
            Better for weights (typically zero-centred by regularisation).
            TFLite uses symmetric quantisation for WEIGHTS by default.
            Uses only [-127, 127] (not -128) for safety.

### The INT8 Matmul Arithmetic

    Understanding why INT8 is fast requires seeing the arithmetic:

    FLOAT32 matmul:
        C[i,j] = Σₖ A[i,k] × B[k,j]
        Each multiply: 1 float32 multiply (4 bytes × 4 bytes → 4 bytes)
        Hardware: 32-bit floating-point unit (FPU)

    INT8 matmul with affine quantisation:
        C_q[i,j] = Σₖ (A_q[i,k] − za) × (B_q[k,j] − zb)

        Expanding:
        = Σₖ A_q[i,k]×B_q[k,j] - za×Σₖ B_q[k,j] - zb×Σₖ A_q[i,k] + K×za×zb

        The term Σₖ A_q[i,k]×B_q[k,j] is the core INT8×INT8→INT32 matmul.
        The correction terms (za, zb offsets) are computed separately.
        Hardware: 8-bit multiply, 32-bit accumulate (SIMD: 4 int8 per op)
        Dequantise: C_real[i,j] = scale_C × (C_q[i,j] − zc)

    WHY INT8 IS FAST:
        SIMD packing: ARM NEON processes 16 × int8 in one instruction vs 4 × float32.
        Hardware NPUs (Google Tensor, Apple ANE): 8-bit systolic arrays
                    run at 8–16× the throughput of float32 operations.
        Memory: 4× less data to read from DRAM → 4× less memory bandwidth needed.
        Cache: 4× more weights fit in L1/L2 cache → less cache miss stall.

### Dynamic Range Quantisation

    Dynamic range quantisation is the SIMPLEST quantisation mode:
        - Weights are quantised to INT8 at model conversion time.
        - Activations are quantised dynamically at RUNTIME per batch.
        - The scale for each activation is computed from min/max of the batch.

    Advantages:
        No calibration dataset required.
        Weights are 4× smaller (INT8 vs FP32).
        Latency improvement: moderate (1.5–2× on CPU).
        Accuracy: usually within 1% of float32 baseline.

    Disadvantages:
        Activations still processed in float at op boundaries.
        Dequantisation overhead between ops (float→int8→float per op).
        Cannot use INT8-only hardware delegates (NPU requires fully static INT8).
        Not as fast as full INT8.

    Use when:
        You have no calibration data.
        The model is large (LLM weights) — size matters more than speed.
        Accuracy tolerance is tight and static INT8 causes regression.

### Static (Full Integer) Quantisation

    Full INT8 quantisation quantises BOTH weights AND activations statically.
    Scale factors are fixed at conversion time; no runtime range computation.

    The conversion flow:
        1. Run calibration dataset through the float model.
        2. For each activation tensor, record min/max across all samples.
        3. Compute scale and zero_point for each tensor.
        4. Replace float ops with INT8 ops.
        5. Insert DEQUANTIZE nodes at model output if needed.

    At inference time:
        Input arrives (may be float32 or int8 depending on configuration).
        If float input: QUANTIZE node converts to INT8 at graph entry.
        All intermediate ops execute in INT8 (kernel reads/writes INT8 buffers).
        If float output needed: DEQUANTIZE node at graph exit.

    Memory layout for INT8 inference:
        All activation tensors live in the arena as INT8 buffers.
        Arena size = 4× smaller than float32 model.
        Total peak memory = INT8 peak activations + INT8 weights.

    Numerical accuracy:
        Typical classification accuracy loss: < 1% top-1 on ImageNet.
        Sensitive models (object detection, super-resolution): may need INT16
        activations or mixed-precision.
        Worst case: models with very wide activation ranges or outlier values.

### Quantisation-Aware Training (QAT)

    Post-training quantisation introduces error because weights and activations
    are quantised after training without the network adapting to the error.
    Quantisation-Aware Training (QAT) minimises this error by SIMULATING
    quantisation during training.

    The QAT mechanism:
        Insert "fake quantise" nodes at weight and activation points.
        Fake quantise: round to the nearest INT8 value, then dequantise back
                       to float. This IS differentiable (straight-through estimator).
        During forward: the model sees quantised values, learns to be robust.
        During backward: gradients flow as if fake quantise were identity.
        After QAT training: convert fake-quantised model to real INT8.

    TensorFlow QAT API:
        import tensorflow_model_optimization as tfmot

        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model  = quantize_model(model)
        q_aware_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        q_aware_model.fit(train_data, train_labels, epochs=5)   # fine-tune

        # Convert QAT model to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_qat  = converter.convert()

    QAT vs PTQ accuracy comparison (MobileNetV2 ImageNet):
        Float32 baseline:           71.9% top-1
        Post-training INT8 (static): 71.0% top-1  (−0.9%)
        QAT INT8:                   71.7% top-1  (−0.2%)

    QAT is necessary when:
        Static PTQ accuracy drop is > 1%.
        Model uses non-standard activations (Swish, H-Swish).
        Model is very small (depthwise separable convolutions are sensitive).
        Target hardware has unusual INT8 precision (some DSPs).


##### PART 5 — THE TFLITE BUILTIN OP SET

### The Builtin Operator Set

    TFLite has approximately 150 builtin operators, versioned per-op.
    Each operator has a BuiltinOperator enum value and a BuiltinOptions
    union table (FlatBuffer union) carrying its configuration.

    NEURAL NETWORK LAYERS:
        CONV_2D:             2D convolution, configurable stride/dilation/padding
        DEPTHWISE_CONV_2D:   depthwise separable conv (1 filter per input channel)
        TRANSPOSE_CONV:      transposed convolution (upsampling)
        FULLY_CONNECTED:     dense matrix multiply + optional bias + activation
        AVERAGE_POOL_2D:     spatial average pooling with stride/padding
        MAX_POOL_2D:         spatial max pooling with stride/padding
        SOFTMAX:             softmax normalisation with temperature parameter
        L2_NORMALIZATION:    L2 normalise along last dimension

    NORMALIZATION:
        BATCH_NORM (absorbed into FULLY_CONNECTED or CONV_2D at conversion)
        LOCAL_RESPONSE_NORMALIZATION: LRN used in AlexNet-era models
        LAYER_NORM (BuiltinOperator 128, added in TFLite schema v3)

    RECURRENT:
        LSTM:                standard LSTM cell with peephole connections
        UNIDIRECTIONAL_SEQUENCE_LSTM: full sequence LSTM
        BIDIRECTIONAL_SEQUENCE_LSTM: bidirectional LSTM
        RNN:                 basic RNN cell
        UNIDIRECTIONAL_SEQUENCE_RNN, BIDIRECTIONAL_SEQUENCE_RNN
        GRU (via CUSTOM op in some versions)

    ELEMENTWISE:
        ADD, SUB, MUL, DIV:  elementwise arithmetic with broadcast
        RELU, RELU6, RELU_N1_TO_1: clipped ReLU variants
        TANH, LOGISTIC (sigmoid): activation functions
        ELU, LEAKY_RELU, PRELU: more activation variants
        ABS, NEG, SQRT, RSQRT, SQUARE, LOG, EXP: math ops
        CEIL, FLOOR, ROUND:  rounding ops
        HARD_SWISH:          x * RELU6(x+3) / 6  (efficient Swish approximation)

    REDUCTION:
        MEAN, SUM, MAX (reduce), MIN (reduce): reduction along axes
        REDUCE_PROD, REDUCE_ANY, REDUCE_ALL: boolean and product reduce
        ARG_MAX, ARG_MIN:    index of max/min along an axis

    SHAPE MANIPULATION:
        RESHAPE:             reshape tensor (shape as a second input tensor)
        SQUEEZE:             remove size-1 dimensions
        EXPAND_DIMS:         add a size-1 dimension
        TRANSPOSE:           permute axes
        CONCATENATION:       concat along axis
        SPLIT, SPLIT_V:      split tensor into N parts along axis
        PACK:                stack tensors along a new dimension
        UNPACK:              unstack a tensor along a dimension
        SLICE:               static and dynamic slicing
        STRIDED_SLICE:       slice with start/stop/step and ellipsis mask
        PAD, PADV2:          zero/reflect/symmetric padding
        TILE:                repeat tensor N times along each axis
        BROADCAST_TO:        broadcast to a target shape

    GATHER / SCATTER:
        GATHER:              index into tensor with integer indices (embedding lookup)
        GATHER_ND:           multi-dimensional gather
        SCATTER_ND:          scatter values into a tensor at specified indices

    ATTENTION AND TRANSFORMERS (newer ops, TFLite schema v3+):
        SEGMENT_SUM:         sum tensor rows by segment IDs
        HASHTABLE_LOOKUP:    look up embeddings from a hash table

### Op Versioning

    Each builtin op has a VERSION field (1, 2, 3, ...). This allows the
    same op to have different semantics or input/output signatures across
    versions while remaining backward compatible.

    EXAMPLES:
        FULLY_CONNECTED v1:  no asymmetric quantisation support
        FULLY_CONNECTED v4:  asymmetric INT8 with per-channel quantisation
        FULLY_CONNECTED v9:  INT16 activations support

        CONV_2D v1:          NHWC float32 only
        CONV_2D v2:          asymmetric INT8 quantisation
        CONV_2D v5:          per-channel quantised weights
        CONV_2D v6:          dilated convolution with asymmetric quantisation

    The runtime checks the op version against its registered kernels.
    A model with CONV_2D v5 will fail on a runtime that only supports up to v4.
    This is why keeping TFLite runtime up to date matters for new models.

### The Select TF Ops Mechanism

    Not every TensorFlow op has a TFLite builtin equivalent.
    SELECT TF OPS (also called Flex ops) solve this:

    When the converter encounters an unsupported op, it can include the
    full TensorFlow kernel for that op in the .tflite model:

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,   # standard TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS,      # allow TF fallback ops
        ]

    The converted model then contains:
        Standard TFLite ops: executed by the fast TFLite interpreter.
        Flex ops: executed by TF's runtime (larger .so, slower, but correct).

    Size impact of SELECT_TF_OPS:
        Standard TFLite binary: ~600 KB
        With SELECT_TF_OPS:     +~1 MB (includes TF kernel libraries)

    When to use SELECT_TF_OPS:
        Models with string operations (text preprocessing in-graph).
        Models with tf.RaggedTensor outputs (variable-length sequences).
        Models with custom TF ops not yet ported to TFLite builtins.
        Prototyping: faster to use flex ops, then port critical ones to builtins.


##### PART 6 — THE TFLITE INTERPRETER AND MEMORY ARENA

### The Interpreter Lifecycle

    Using TFLite in Python:
        import tflite_runtime.interpreter as tflite
        # OR: from tensorflow.lite.python.interpreter import Interpreter as tflite.Interpreter

        interpreter = tflite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()   # ← CRITICAL: allocate the arena

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], input_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

    In C++ (Android/iOS production):
        // Load model from file
        auto model = tflite::FlatBufferModel::BuildFromFile("model.tflite");
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        std::unique_ptr<tflite::Interpreter> interpreter;
        builder(&interpreter);
        interpreter->AllocateTensors();

        // Set input, run, get output
        float* input_data = interpreter->typed_input_tensor<float>(0);
        std::copy(input_vec.begin(), input_vec.end(), input_data);
        interpreter->Invoke();
        float* output_data = interpreter->typed_output_tensor<float>(0);

### allocate_tensors(): The Critical Initialisation Step

    allocate_tensors() is not just memory allocation. It performs:

    STEP 1 — MEMORY PLANNING:
        Walk the model's computation graph.
        For each tensor, determine its LIFETIME — the range of operator
        indices during which the tensor must be alive.
        Two tensors can SHARE MEMORY if their lifetimes do not overlap.
        This is the interval graph colouring problem.
        TFLite's planner minimises the total arena size.

    STEP 2 — ARENA ALLOCATION:
        Compute the peak simultaneous memory needed (considering sharing).
        Allocate ONE contiguous block (the arena) of that size.
        Assign byte offsets within the arena to each tensor.
        Weights live in a SEPARATE read-only arena (mapped directly from the .tflite binary).

    STEP 3 — KERNEL INIT:
        For each op in topological order, call the kernel's Init() function.
        Init() allocates any persistent state the kernel needs (e.g., GEMM
        workspace buffers, scratchpad for FFT, delegate sub-graphs).

    WHY THIS MATTERS:
        After allocate_tensors(), ZERO dynamic allocation happens during invoke().
        Latency is deterministic (no GC pauses, no malloc latency spikes).
        Peak memory is known precisely before the first inference runs.
        On MCUs: no heap at all — the arena is a static array in BSS/RAM.

### The Memory Arena in Detail

    The TFLite arena is divided into three regions:

    REGION 1 — Persistent arena (model weights and static data):
        Contains: weight tensors, bias tensors, BatchNorm statistics.
        Source: directly mapped from the .tflite file's buffer section.
        Access: read-only, memory-mapped — no copy, no allocation.
        On MCU: stored in flash, accessed via direct memory map.

    REGION 2 — Activation arena (intermediate computation):
        Contains: all activation tensors (input→conv→relu→pool→…).
        Lifetime-based sharing: tensor A and tensor B share memory if A
        is consumed before B is produced.
        On phone: single aligned malloc() at model load.
        On MCU: static array, e.g.: uint8_t tensor_arena[256*1024];

    REGION 3 — Temporary arena (kernel scratchpad):
        Some kernels need temporary working memory (convolution im2col buffer,
        GEMM transpose buffer, FFT working arrays).
        Allocated once, shared across kernels sequentially (not concurrently).

    Memory planning visualisation for a MobileNetV2:
        Total model weights:      3.4 MB INT8
        Peak activation memory:   0.4 MB INT8   (shared across 50+ tensors)
        Temp scratchpad:          0.1 MB
        Total runtime memory:     3.9 MB INT8
        (vs ~14 MB float32 equivalent)

### Tensor Sharing and Memory Layout

    TFLite's memory planner performs OFFLINE TENSOR SHARING:
        It solves: given N tensors with lifetimes [start_op, end_op],
                   what is the minimum arena size such that no two tensors
                   with overlapping lifetimes share memory?
        Algorithm: greedy interval colouring (similar to register allocation).
        Result: a sorted list of (tensor_index, arena_offset, size) triples.

    Sharing example for a simple: Input → Conv → ReLU → Pool → Conv → Output:
        tensor_0: Input    lifetime [0, 0]    offset 0     size 3 KB
        tensor_1: Conv_out lifetime [0, 1]    offset 3     size 64 KB
        tensor_2: ReLU_out lifetime [1, 2]    offset 3     size 64 KB   ← SHARES with Conv_out!
        tensor_3: Pool_out lifetime [2, 3]    offset 3     size 16 KB   ← ALSO SHARES
        tensor_4: Output   lifetime [3, 3]    offset 3     size 4 KB

    Peak arena = max simultaneous live bytes = 3 KB + 64 KB = 67 KB
    Without sharing: 3 + 64 + 64 + 16 + 4 = 151 KB — 2.25× more memory.


##### PART 7 — HARDWARE DELEGATES: GPU, NNAPI, COREML, HEXAGON, XNNPACK

### What a Delegate Is

    A DELEGATE is a hardware-specific compilation and execution backend
    that integrates with the TFLite interpreter via a stable C++ interface.

    The delegate mechanism:
        1. Delegate object created with hardware-specific config.
        2. Delegate registered with the interpreter BEFORE allocate_tensors().
        3. Delegate walks the graph and "claims" the ops it can execute.
        4. Claimed ops are compiled into a hardware-specific sub-graph (a "plan").
        5. At invoke() time: interpreter dispatches claimed ops to the delegate.
                             unclaimed ops fall back to CPU kernels.
        6. Data moves from CPU memory to hardware memory at delegate boundaries.

    The interpreter's fallback guarantee:
        If the delegate fails to initialise (hardware unavailable), or if an op
        is not supported by the delegate, the interpreter AUTOMATICALLY FALLS BACK
        to the CPU implementation. No crash, no error (unless strict fallback
        mode is enabled). This makes delegates safe to deploy unconditionally.

### XNNPACK Delegate (CPU Acceleration)

    XNNPACK is Google's cross-platform SIMD acceleration library.
    It provides CPU kernels that are faster than TFLite's reference
    implementations by exploiting SIMD hardware directly.

    Supported hardware:
        x86/x86_64: SSE2, SSE4.1, AVX, AVX2, AVX512 (F, BW, VL, VNNI)
        ARM32 (Cortex-A): NEON, NEON FP16
        ARM64 (AArch64): NEON, NEON FP16, ARMv8.2 dot product, SVE
        RISC-V: RVV (RISC-V Vector extension)
        WebAssembly: SIMD128

    XNNPACK is the DEFAULT CPU delegate (enabled automatically in most builds).
    Speedups over reference TFLite kernels:
        Float32 CONV_2D:         2–4× on ARM NEON
        Float32 FULLY_CONNECTED: 2–6× on x86 AVX2
        INT8 CONV_2D:            4–8× on ARM with dot product instructions
        FP16 inference:          2–3× on devices with native FP16 NEON

    How to enable explicitly:
        import tflite_runtime.interpreter as tflite

        interpreter = tflite.Interpreter(
            model_path="model.tflite",
            experimental_delegates=[tflite.load_delegate('libxnnpack.so')]
        )
        # OR (Python TF):
        interpreter = tflite.Interpreter(
            model_path="model.tflite",
            num_threads=4)   # XNNPACK enabled by default

### GPU Delegate (Mobile GPU Acceleration)

    The GPU delegate runs inference on the mobile GPU (Adreno, Mali, Apple).
    It generates OpenCL or OpenGL compute shaders from the TFLite ops.

    Supported hardware:
        Android:  Qualcomm Adreno (OpenCL 2.0), ARM Mali (OpenCL 2.0)
        iOS:      Apple GPU (Metal compute shaders)
        Desktop:  Any OpenCL 1.2+ GPU

    Speedups (typical, float32 inference):
        MobileNetV2 on Adreno 650: 3–6× vs CPU
        EfficientDet on Mali-G78:  2–4× vs CPU
        Classification models:     3–8× vs CPU (memory bandwidth limited)
        Transformer attention:     1.5–3× (memory access pattern less GPU-friendly)

    GPU delegate requires FLOAT32 or FLOAT16 models.
    INT8 models require DEQUANTIZE→GPU→QUANTIZE at delegate boundaries.

    Usage:
        from tensorflow.lite.python.interpreter import load_delegate

        gpu_delegate = load_delegate('libtensorflowlite_gpu_delegate.so',
                                     {'is_precision_loss_allowed': 1})   # allow FP16
        interpreter  = tflite.Interpreter(
            model_path="model.tflite",
            experimental_delegates=[gpu_delegate])

    FP16 GPU inference:
        Setting 'is_precision_loss_allowed': 1 lets the GPU delegate run
        in FP16, which is 2× faster on most mobile GPUs.
        Accuracy impact: typically < 0.1% on classification tasks.

### NNAPI Delegate (Android Neural Networks API)

    NNAPI is Android's hardware abstraction layer for ML accelerators.
    It provides a unified API that routes computation to:
        The device's dedicated NPU (Neural Processing Unit)
        The DSP (Digital Signal Processor) if the NPU is unavailable
        The GPU via the NNAPI runtime
        The CPU (fallback)

    NNAPI availability:
        Android 8.1+: NNAPI 1.0 (basic CNNs)
        Android 9.0+: NNAPI 1.1 (recurrent models, quantised ops)
        Android 10+:  NNAPI 1.2 (INT8 quantisation, more op coverage)
        Android 11+:  NNAPI 1.3 (INT16, signed INT8 inputs)

    The NPU advantage:
        Modern SoCs have dedicated matrix multiply hardware with INT8 throughput
        many times faster than the CPU:
            Google Pixel 8 (Tensor G3 NPU):  7.3 TOPS INT8
            Samsung Galaxy S24 (Exynos NPU): 14.7 TOPS INT8
            Apple A17 Pro (ANE):             35 TOPS INT8

    NNAPI does NOT guarantee determinism:
        The OEM decides which hardware to use for each op.
        A given op may run on GPU, DSP, or CPU depending on the device.
        Numerical results may differ slightly from CPU reference.

    Usage:
        nnapi_delegate = load_delegate('libnnapi_util.so',
                                       {'execution_preference': 'fast_single_answer'})
        interpreter = tflite.Interpreter(
            model_path="model.tflite",
            experimental_delegates=[nnapi_delegate])

### CoreML Delegate (Apple Silicon)

    On iOS and macOS, the CoreML delegate routes TFLite inference to
    Apple's Neural Engine (ANE) via Core ML.

    Apple Neural Engine:
        iPhone 15 (A17 Pro):  35 TOPS INT8 (dedicated fixed-function hardware)
        M2 Mac:               15.8 TOPS INT8
        Much lower power than GPU — crucial for mobile battery life.

    CoreML delegate requirements:
        iOS 12+ / macOS 10.14+
        Model must be in float32 (CoreML handles internal conversion)
        Op coverage: CNNs, basic transformers (limited recurrent support)

    The CoreML delegate converts TFLite ops to Core ML's mlmodel format
    internally, then dispatches to the ANE via the Core ML framework.

    Usage (C++):
        TfLiteCoreMlDelegateOptions options = {0};
        options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
        auto delegate = TfLiteCoreMlDelegateCreate(&options);
        interpreter->ModifyGraphWithDelegate(delegate);

### Hexagon Delegate (Qualcomm DSP)

    The Hexagon delegate targets Qualcomm's Hexagon DSP (HVX extension),
    present in Snapdragon SoCs.

    HVX (Hexagon Vector Extensions):
        Processes 128 × INT8 values per clock cycle (1024-bit SIMD).
        At 1 GHz: 128 GOPS INT8 — faster than the Adreno GPU for many models.
        Very power-efficient: ~10 TOPS/W vs ~2 TOPS/W for GPU.

    Requirements:
        Qualcomm Snapdragon 835 or newer with HVX support.
        Hexagon Neural Network (HNN) libraries installed on device.
        INT8 quantised model (Hexagon cannot run float32 natively).

    The delegate compiles eligible INT8 ops to HVX-optimised instructions
    via Qualcomm's HNN SDK, then invokes them via the CDSP (Compute DSP).


##### PART 8 — CUSTOM OPERATORS AND THE OP REGISTRATION SYSTEM

### The OpResolver Architecture

    TFLite uses an OPRESOLVER to map from the (op_name, version) pair in
    a .tflite file to a kernel implementation. There are two resolvers:

    BuiltinOpResolver:
        Registers all ~150 builtin ops with their versioned kernels.
        Ships with the TFLite runtime; no model changes needed.
        Size: ~500 KB (all kernels compiled in).

    MutableOpResolver (custom ops):
        Starts empty. You register ops manually:
            tflite::MutableOpResolver resolver;
            resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                tflite::ops::builtin::Register_CONV_2D());
            resolver.AddCustom("MyCustomOp", Register_MyCustomOp());
        Allows minimum-size builds: only include kernels you use.

### Implementing a Custom Operator

    A TFLite custom op requires implementing four C++ functions:

    FUNCTION 1 — Init:
        Called once at allocate_tensors() time.
        Allocates any persistent state (e.g., workspace buffers).
        Returns an opaque void* (carried through the other functions).

    FUNCTION 2 — Free:
        Called at interpreter destruction.
        Deallocates the state returned by Init.

    FUNCTION 3 — Prepare:
        Called when the output tensor shapes need to be computed.
        Reads input shapes, computes and sets output shapes.
        May re-allocate workspace if shapes changed.
        Reads the custom_options FlatBuffer for op configuration.

    FUNCTION 4 — Eval:
        Called every inference (invoke()).
        Reads from input tensors, writes to output tensors.
        Must be re-entrant (called from multiple threads for parallel inference).

    Registration structure:
        TfLiteRegistration* Register_MyCustomOp() {
            static TfLiteRegistration reg = {
                Init, Free, Prepare, Eval
            };
            return &reg;
        }

    In the model, the custom op node has:
        op_type = "MyCustomOp"  (matches the name in AddCustom)
        custom_options = serialised FlexBuffer with op parameters

### Writing the Custom Op in Python

    For Python inference, you also register the op in Python:

        import tflite_runtime.interpreter as tflite
        import numpy as np

        # Register the custom op
        interpreter = tflite.Interpreter(
            model_path="model_with_custom_op.tflite",
            experimental_op_resolver_type=tflite.experimental.OpResolverType.AUTO,
            experimental_delegates=[
                tflite.load_delegate("libcustom_op.so")   # your custom op library
            ]
        )

### The FlatBuffer custom_options Encoding

    Custom op configuration is stored as a FlatBuffer (specifically a
    FlexBuffer) in the Operator.custom_options field.

    On the exporter side (tf.lite.TFLiteConverter):
        The converter calls the custom op's symbolic function (TF side)
        which serialises config into FlexBuffer bytes.

    On the runtime side (C++ Eval function):
        const uint8_t* buffer = node->custom_initial_data;
        const flexbuffers::Map& m = flexbuffers::GetRoot(buffer, size).AsMap();
        float alpha = m["alpha"].AsFloat();
        int   n_dim = m["n_dim"].AsInt32();

    FlexBuffer is a schema-free binary map format. Unlike FlatBuffers,
    it does not require a schema — you access values by string key.


##### PART 9 — TFLITE MICRO: INFERENCE ON MICROCONTROLLERS

### What TFLite Micro Is

    TFLite Micro (TFLM) is a port of TFLite to bare-metal microcontrollers
    with severe constraints:
        No operating system (no pthreads, no malloc, no file system, no printf)
        Total RAM: 16 KB – 2 MB (MCU SRAM, not DRAM)
        Storage: flash memory (read-only at runtime), 64 KB – 2 MB typical
        CPU: ARM Cortex-M0 to Cortex-M7 (60 MHz – 480 MHz)
              RISC-V, ESP32, Xtensa also supported
        Power: 0.001 mW – 100 mW (months to years on a battery)

    TFLM fits in:
        Runtime code: ~20 KB flash (without any op kernels)
        With keyword spotting kernels: ~50 KB flash
        With full CNN support (CONV, DEPTHWISE_CONV, FC, POOL): ~100 KB flash

### Key Differences from Standard TFLite

    1. NO DYNAMIC MEMORY ALLOCATION:
        No new, no malloc, no heap. Ever.
        The tensor arena is a STATIC ARRAY provided by the user:
            constexpr int kTensorArenaSize = 256 * 1024;  // 256 KB
            uint8_t tensor_arena[kTensorArenaSize];
        If the arena is too small: error at AllocateTensors(), not at runtime.

    2. NO FILE SYSTEM ACCESS:
        The .tflite file is compiled into the firmware as a C array:
            const unsigned char model_tflite[] = { 0x18, 0x00, ... };
            const int model_tflite_len = 12345;
        The model pointer is passed directly to the interpreter.

    3. NO STANDARD LIBRARY DEPENDENCIES:
        TFLM does not use <stdio.h>, <stdlib.h>, or C++ exceptions.
        Custom ErrorReporter and memory allocators can be provided.

    4. SUBSET OF OPS:
        Not all TFLite ops are implemented in TFLM (size budget).
        Commonly supported: CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED,
        SOFTMAX, POOLING, RESHAPE, CONCATENATION, INT8 activations.
        The TFLM library lets you include ONLY the ops your model uses.

    5. MICRO OP RESOLVER:
        Instead of BuiltinOpResolver (registers all ops):
            tflite::MicroMutableOpResolver<5> resolver;
            resolver.AddConv2D();
            resolver.AddDepthwiseConv2D();
            resolver.AddFullyConnected();
            resolver.AddSoftmax();
            resolver.AddReshape();
        The template parameter (5) is the maximum number of ops.
        This eliminates linking overhead for unused kernels.

### The TFLM Inference Loop

    Complete bare-metal inference (Cortex-M4, keyword spotting model):

        #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
        #include "tensorflow/lite/micro/micro_interpreter.h"
        #include "model_data.h"         // generated .h with model bytes
        #include "audio_features.h"     // your feature extraction

        // Static arena — the only heap
        constexpr int kArenaSize = 100 * 1024;
        uint8_t tensor_arena[kArenaSize];

        // Op resolver — only what this model uses
        tflite::MicroMutableOpResolver<4> resolver;
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddFullyConnected();
        resolver.AddSoftmax();

        // Load model (no malloc)
        const tflite::Model* model = tflite::GetModel(model_tflite);
        tflite::MicroInterpreter interpreter(model, resolver,
                                             tensor_arena, kArenaSize);
        interpreter.AllocateTensors();

        // Inference loop (runs forever on MCU)
        while (true) {
            // Get features from microphone (implementation-defined)
            float* input = interpreter.input(0)->data.f;
            ExtractMFCC(audio_buffer, input, kFeatureSize);

            // Run inference
            interpreter.Invoke();

            // Read output
            float* output = interpreter.output(0)->data.f;
            int keyword   = argmax(output, kNumKeywords);

            if (output[keyword] > kThreshold) {
                TriggerAction(keyword);
            }
        }

### TFLM Hardware Backends (CMSIS-NN)

    For ARM Cortex-M MCUs, TFLM integrates with CMSIS-NN:
    the ARM CMSIS Neural Network Library, which provides SIMD-optimised
    INT8 kernels using ARM DSP extensions.

    CMSIS-NN kernels use:
        ARM DSP instructions: SIMD 32-bit operations (2×16-bit or 4×8-bit per op)
        Available on: Cortex-M4, M7, M33, M55 (DSP extension)
        NOT on: Cortex-M0, M0+ (no DSP instructions)

    Speedups over reference kernels:
        INT8 CONV_2D (Cortex-M4):           2–4× vs reference
        INT8 DEPTHWISE_CONV_2D (Cortex-M4): 3–5× vs reference
        INT8 FULLY_CONNECTED (Cortex-M7):   4–8× vs reference

    Enabling CMSIS-NN in TFLM:
        cmake -DTFLITE_ENABLE_CMSIS_NN=ON
        # or in Arduino: use the EloquentTinyML library

    Ethos-U backend (Cortex-M55 + Ethos-U55/U65 NPU):
        ARM's microNPU for MCUs — 256 GOPS INT8 at ~10 mW.
        TFLM can dispatch supported ops to the Ethos-U via Vela compiler.
        Vela compiles the .tflite model to Ethos-U microcode at conversion time.

### TFLM Memory Benchmarking

    Measuring TFLM memory usage on real hardware:

    Keyword Spotting (DS-CNN-L, INT8):
        Flash (model):  46 KB
        Flash (code):   55 KB
        RAM (arena):    100 KB
        Latency (M4):   12 ms at 168 MHz
        Power:          ~20 mW peak

    Person Detection (MobileNetV1 0.25, INT8, 96×96):
        Flash (model):  250 KB
        Flash (code):   80 KB
        RAM (arena):    512 KB
        Latency (M7):   850 ms at 480 MHz
        Latency (Ethos-U55): 4 ms (200× faster with NPU)

    Anomaly Detection (fully-connected, INT8):
        Flash (model):  8 KB
        RAM (arena):    4 KB
        Latency (M0+):  0.1 ms at 64 MHz


##### PART 10 — LIBERT: THE REBRANDING AND MODERN DIRECTION

### The LiteRT Rebranding (2023)

    In late 2023, Google announced LiteRT (Lite RunTime) as the new name
    for TFLite. The technical contents are identical; the renaming signals:
        Framework independence: LiteRT accepts models from PyTorch, JAX,
                                 Keras, and TF — not just TensorFlow.
        Hardware breadth:       Expanded beyond mobile to any edge hardware.
        Governance:             Moved to an open governance model.

    BACKWARD COMPATIBILITY:
        All existing .tflite models work with LiteRT without modification.
        The Python API is unchanged: tf.lite.* and tflite_runtime.* still work.
        The C++ API is unchanged.
        The FlatBuffer schema is backward-compatible.

    New package names:
        pip install ai-edge-litert         # runtime (replaces tflite-runtime)
        pip install ai-edge-torch          # PyTorch → TFLite converter
        The tensorflow.lite.* Python API is retained for compatibility.

### ai-edge-torch: PyTorch Native Export

    The most significant new capability in LiteRT is ai-edge-torch:
    a Google library for converting PyTorch models to .tflite WITHOUT
    going through TensorFlow.

    Pipeline:
        PyTorch model
            ↓ torch.export.export (TorchDynamo graph capture)
        ExportedProgram (ATen IR)
            ↓ ai_edge_torch.convert (lowers to TFLite MLIR dialect)
        TFLite MLIR
            ↓ FlatBuffer serialiser
        model.tflite

    Usage:
        import ai_edge_torch
        import torch

        class MyModel(torch.nn.Module):
            def forward(self, x): return self.layers(x)

        model  = MyModel().eval()
        sample = (torch.randn(1, 3, 224, 224),)

        edge_model = ai_edge_torch.convert(model, sample)
        edge_model.export("model.tflite")

        # With quantisation:
        edge_model = ai_edge_torch.convert(
            model, sample,
            quant_config=ai_edge_torch.quantize.QuantizationConfig(
                global_config=ai_edge_torch.quantize.OpQuantizationConfig(
                    weight_dtype=ai_edge_torch.quantize.QuantDtype.AI_EDGE_W8A8_PT_WEIGHT
                )
            ))
        edge_model.export("model_int8.tflite")

    Supported models (as of 2024):
        MobileNet family, EfficientNet family
        BERT-style transformers
        Gemma 2B (Google's LLM, via MediaPipe LLM Inference)
        Custom PyTorch models using standard ops

### MediaPipe and On-Device LLMs

    LiteRT is the inference engine for MediaPipe — Google's framework
    for building on-device ML pipelines (face detection, pose estimation,
    hand tracking, segmentation).

    MediaPipe Tasks (high-level API built on TFLite):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Object detection
        detector = vision.ObjectDetector.create_from_model_path("model.tflite")
        result   = detector.detect(image)

        # Text classification
        classifier = text.TextClassifier.create_from_model_path("model.tflite")
        result     = classifier.classify("This movie is amazing!")

    On-Device LLM Inference with LiteRT:
        Google's Gemma 2B can run on-device via LiteRT:
            Model: Gemma 2B, 4-bit quantised → ~1.4 GB
            Inference: ~20 tokens/second on Pixel 8 (Tensor G3 NPU)
            API: MediaPipe LLM Inference Task

        from mediapipe.tasks.python.genai import inference as genai_inference
        options = genai_inference.LlmInferenceOptions(
            model_path="/data/local/tmp/gemma-2b-it-gpu-int4.bin",
            max_tokens=1024,
            topk=40,
            temperature=0.8,
        )
        llm = genai_inference.LlmInference.create_from_options(options)
        result = llm.generate_response("Tell me a joke:")


##### PART 11 — TFLITE IN THE CONNECTED ECOSYSTEM

### TFLite and MLIR (Module 02)

    TFLite's converter is the largest production user of MLIR outside XLA.
    The entire converter pipeline (TF → TFLite dialect → serialise) is
    implemented as MLIR passes.

    TFLite-specific MLIR dialects:
        TF dialect:      tf.Conv2D, tf.MatMul, tf.Relu, ...
        TFLite dialect:  tfl.conv_2d, tfl.fully_connected, tfl.relu, ...
        TOSA dialect:    tosa.conv2d, tosa.matmul (intermediate step for some paths)

    The quantisation passes are also MLIR-based:
        --tfl-prepare-quantize:   insert QuantizeOp/DequantizeOp nodes
        --tfl-quantize:           propagate quantisation parameters
        --tfl-post-quantize:      fuse Q/DQ nodes, clean up the graph

    onnx-mlir and the TFLite MLIR dialect share pass infrastructure.
    A model going through onnx-mlir → TFLite MLIR → .tflite is possible.

### TFLite and StableHLO (Module 07)

    TFLite's MLIR-based converter can now ingest StableHLO directly.
    This is the path for JAX and PyTorch (via torch-mlir) models:

        JAX model → jax.export → StableHLO
                             ↓ stablehlo-to-tfl pass
                          TFLite dialect
                             ↓ FlatBuffer serialiser
                          model.tflite

    The stablehlo-to-tfl lowering handles:
        stablehlo.convolution  → tfl.conv_2d
        stablehlo.dot_general  → tfl.fully_connected / tfl.batch_matmul
        stablehlo.reduce       → tfl.mean / tfl.sum / tfl.max
        stablehlo.broadcast_in_dim → tfl.broadcast_to

### TFLite and TVM (Module 08)

    TFLite and TVM are COMPETITIVE at the edge:
        Both target mobile CPUs, MCUs, and mobile NPUs.
        TFLite has deeper Android integration (NNAPI, Play Services).
        TVM has better auto-scheduling for custom hardware.

    TVM can IMPORT TFLite models:
        import tvm, tvm.relay as relay

        with open("model.tflite", "rb") as f:
            tflite_model_buf = f.read()

        mod, params = relay.frontend.from_tflite(tflite_model_buf)
        target      = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
        lib         = relay.build(mod, target=target, params=params)
        lib.export_library("model_tvm_arm64.so")

    This gives TVM's MetaSchedule auto-scheduling benefits to TFLite models.
    Useful when deploying to custom ARM hardware where TFLite's kernels
    are not optimised but TVM can find better schedules.

### TFLite and ONNX (Module 10)

    ONNX ↔ TFLite conversion:
        ONNX → TFLite: via onnx-tf bridge (ONNX → SavedModel → .tflite)
        TFLite → ONNX: via tensorflow-onnx (tf2onnx) after TFLite → SavedModel

    In practice:
        TFLite is the endpoint for deployment — not usually converted to ONNX.
        ONNX is the interchange format for moving between training frameworks.
        The typical path: PyTorch → ONNX → (onnx-tf) → SavedModel → .tflite.

### The Complete TFLite Ecosystem

    ┌──────────────────────────────────────────────────────────────────────┐
    │  TRAINING / SOURCE FRAMEWORKS                                        │
    │  TensorFlow    PyTorch     JAX      Keras      ONNX model            │
    │  (SavedModel)  (ai-edge)   (StHLO)  (Keras)    (via onnx-tf)         │
    └──────────────────────────┬───────────────────────────────────────────┘
                               │
                    TFLite Converter (MLIR-based)
                    + Quantisation (PTQ / QAT / dynamic range / FP16)
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  .tflite FlatBuffer binary                                           │
    │  Float32 / Dynamic INT8 / Static INT8 / FP16 / INT16 act             │
    └────┬─────────────────────┬───────────────────────────────────────────┘
         │                     │
         ▼                     ▼
    ┌────────────────┐  ┌───────────────────────────────────────────────────┐
    │  TFLite Micro  │  │  TFLite Interpreter (mobile/edge)                 │
    │  No OS / MCU   │  │  ┌──────────────────────────────────────────────┐ │
    │  <256 KB SRAM  │  │  │ Delegates (hardware acceleration)            │ │
    │  Cortex-M, RISC│  │  │  XNNPACK (SIMD CPU) GPU (OpenCL/Metal)       │ │
    │  CMSIS-NN/Ethos│  │  │  NNAPI (Android NPU)  CoreML (Apple ANE)     │ │
    └────────────────┘  │  │  Hexagon (Qualcomm DSP)                      │ │
                        │  └──────────────────────────────────────────────┘ │
                        └──────────────────────────────────────────┬────────┘
                                                                   │
                        ┌──────────────────────────────────────────┘
                        │
    ┌───────────────────▼──────────────────────────────────────────────────┐
    │  RUNTIME TARGETS                                                     │
    │  Android (Java/Kotlin/C++)    iOS (Swift/Obj-C/C++)                  │
    │  Linux (ARM64/x86)            Raspberry Pi      Desktop (CPU/GPU)    │
    │  Arduino / Zephyr (TFLM)      STM32 (TFLM)      ESP32 (TFLM)         │
    └──────────────────────────────────────────────────────────────────────┘

    COMPARISON WITH ALTERNATIVES:
    ┌──────────────────────┬──────────────┬──────────────┬──────────────────┐
    │ Property             │ TFLite       │ ONNX Runtime │ CoreML           │
    ├──────────────────────┼──────────────┼──────────────┼──────────────────┤
    │ File format          │ FlatBuffers  │ Protobuf     │ MIL (protobuf)   │
    │ Primary target       │ Android/MCU  │ Cross-platform│ iOS/macOS only  │
    │ MCU support          │ YES (TFLM)   │ Limited      │ No               │
    │ NNAPI integration    │ Built-in     │ Via EP       │ No (uses ANE)    │
    │ Apple ANE            │ Via CoreML D.│ Via CoreML EP│ Native           │
    │ Default quantisation │ INT8 first   │ INT8 optional│ FP16/INT8        │
    │ Model size (FP32)    │ Compact      │ Compact      │ Compact          │
    │ Op coverage          │ ~150 builtins│ ~250 ops     │ ~100 ops         │
    │ Custom ops           │ C++ required │ C++ required │ Swift/ObjC/Metal │
    │ Python inference     │ YES          │ YES          │ YES (macOS only) │
    └──────────────────────┴──────────────┴──────────────┴──────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · TFLite Conversion Pipeline — Float, FP16, and INT8 Models": {
        "description": (
            "End-to-end TFLite conversion from a Keras/PyTorch model. "
            "Convert the same model in float32, FP16, dynamic-range INT8, and static INT8. "
            "Compare .tflite file sizes and measure inference latency for each. "
            "Validate numerical accuracy against the float32 reference. "
            "Show the FlatBuffer schema structure: tensors, operators, and quantisation params."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import io
import os

print("=" * 65)
print("  TFLITE CONVERSION PIPELINE — FLOAT, FP16, AND INT8 MODELS")
print("=" * 65)
print()

try:
    import tensorflow as tf
    print(f"  TensorFlow {tf.__version__}")
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("  TensorFlow not installed: pip install tensorflow")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Model definition and float32 conversion
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Define model and convert to float32 TFLite")
print("━" * 65)
print()

if HAS_TF:
    # ── Define a small MobileNetV2-style CNN ──────────────────────────────
    def make_model():
        """Small depthwise-separable CNN (MobileNet style)."""
        inp = tf.keras.Input(shape=(32, 32, 3), name="input")
        x   = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
        x   = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
        x   = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
        x   = tf.keras.layers.GlobalAveragePooling2D()(x)
        x   = tf.keras.layers.Dense(10, activation="softmax")(x)
        return tf.keras.Model(inp, x, name="mini_mobilenet")

    model = make_model()
    print(f"  Model: {model.name}")
    print(f"  Params: {model.count_params():,}")
    print(f"  Input:  {model.input_shape}")
    print(f"  Output: {model.output_shape}")
    print()

    # Generate calibration data (100 random images)
    N_CALIB = 100
    calib_data = np.random.rand(N_CALIB, 32, 32, 3).astype(np.float32)

    # Representative dataset generator for static INT8
    def representative_dataset():
        for i in range(N_CALIB):
            yield [calib_data[i:i+1]]   # batch of 1

    # ── Convert to four formats ────────────────────────────────────────────
    results = {}

    def convert_and_measure(name, setup_fn, model, n_warmup=50, n_reps=500):
        """Convert model, run inference, return (bytes, latency_ms)."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        setup_fn(converter)
        try:
            tflite_bytes = converter.convert()
        except Exception as e:
            return None, None, str(e)[:60]

        # Run via Interpreter
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        inp_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]

        # Use correct input dtype
        x_test = np.random.rand(1, 32, 32, 3).astype(inp_d["dtype"])
        for _ in range(n_warmup):
            interp.set_tensor(inp_d["index"], x_test)
            interp.invoke()

        t0 = time.perf_counter()
        for _ in range(n_reps):
            interp.set_tensor(inp_d["index"], x_test)
            interp.invoke()
        latency = (time.perf_counter() - t0) / n_reps * 1000

        return tflite_bytes, latency, "ok"

    # Float32
    def setup_float32(c): pass

    # FP16
    def setup_fp16(c):
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]

    # Dynamic range INT8 (weights only)
    def setup_dynamic(c):
        c.optimizations = [tf.lite.Optimize.DEFAULT]

    # Static INT8 (weights + activations)
    def setup_static_int8(c):
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = representative_dataset
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type  = tf.float32   # keep float32 I/O for simplicity
        c.inference_output_type = tf.float32

    formats = [
        ("Float32",      setup_float32),
        ("FP16 weights", setup_fp16),
        ("Dynamic INT8", setup_dynamic),
        ("Static INT8",  setup_static_int8),
    ]

    print(f"  Converting to four quantisation formats...")
    print()
    float_bytes = None
    ref_output  = None

    for fmt_name, setup_fn in formats:
        tflite_bytes, latency, status = convert_and_measure(
            fmt_name, setup_fn, model)
        if tflite_bytes is None:
            print(f"  {fmt_name:<20s}: FAILED — {status}")
            continue

        size_kb = len(tflite_bytes) / 1024
        if fmt_name == "Float32":
            float_bytes = tflite_bytes
            size_ratio  = 1.0
        else:
            size_ratio  = len(tflite_bytes) / len(float_bytes) if float_bytes else 1.0

        results[fmt_name] = (tflite_bytes, latency)
        print(f"  {fmt_name:<20s}: {size_kb:7.1f} KB  "
              f"({size_ratio:.2f}× float32)  {latency:.3f} ms/step")

    print()
    print("  Size summary:")
    print("    Float32 → FP16 weights: ~2× compression (approx)")
    print("    Float32 → Dynamic INT8: ~4× compression (weights only)")
    print("    Float32 → Static INT8:  ~4× compression (weights+activations)")
    print()

else:
    CONV_REF = """
  TFLITE CONVERSION API REFERENCE:

  import tensorflow as tf

  # Build your model (Keras)
  model = tf.keras.Sequential([...])
  model.compile(...)
  model.fit(...)

  # ── Float32 (baseline) ──────────────────────────────────────────────
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  open("model.tflite", "wb").write(tflite_model)

  # ── FP16 (2× size reduction) ────────────────────────────────────────
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  tflite_fp16 = converter.convert()

  # ── Dynamic range INT8 (4× size, no calibration needed) ────────────
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_dynamic = converter.convert()

  # ── Static INT8 (4× size + faster kernels, needs calibration) ───────
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset_fn
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type  = tf.int8    # optional: int8 I/O
  converter.inference_output_type = tf.int8
  tflite_int8 = converter.convert()
"""
    print(CONV_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Inspecting the FlatBuffer schema
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — FlatBuffer inspection: tensors, ops, quantisation params")
print("━" * 65)
print()

if HAS_TF and "Float32" in results and results["Float32"][0] is not None:
    float_tflite = results["Float32"][0]

    try:
        # Use the TFLite Interpreter to inspect the model structure
        interp = tf.lite.Interpreter(model_content=float_tflite)
        interp.allocate_tensors()

        inp_details = interp.get_input_details()
        out_details = interp.get_output_details()
        all_tensors = interp.get_tensor_details()

        print(f"  Float32 model inspection:")
        print(f"    Total tensors:   {len(all_tensors)}")
        print(f"    Input tensors:   {len(inp_details)}")
        print(f"    Output tensors:  {len(out_details)}")
        print()

        print(f"  Inputs:")
        for d in inp_details:
            print(f"    [{d['index']:3d}] {d['name']:<30s} "
                  f"shape={d['shape']}  dtype={d['dtype'].__name__}")

        print(f"  Outputs:")
        for d in out_details:
            print(f"    [{d['index']:3d}] {d['name']:<30s} "
                  f"shape={d['shape']}  dtype={d['dtype'].__name__}")

        print(f"  All tensors (first 10):")
        print(f"  {'idx':>4}  {'name':35s}  {'shape':20s}  {'dtype':10s}  {'quant scale'}")
        print("  " + "-" * 85)
        for d in all_tensors[:10]:
            q = d.get("quantization_parameters", {})
            scales = q.get("scales", [])
            scale_str = f"{scales[0]:.4f}" if len(scales) == 1 else \
                        f"{len(scales)}ch" if scales else "—"
            print(f"  {d['index']:>4}  {d['name'][:34]:35s}  "
                  f"{str(list(d['shape'])):20s}  "
                  f"{d['dtype'].__name__:10s}  {scale_str}")
        if len(all_tensors) > 10:
            print(f"  ... ({len(all_tensors)-10} more tensors)")
        print()

    except Exception as e:
        print(f"  Inspection error: {e}")
        print()

if HAS_TF and "Static INT8" in results and results["Static INT8"][0] is not None:
    int8_tflite = results["Static INT8"][0]
    interp_q = tf.lite.Interpreter(model_content=int8_tflite)
    interp_q.allocate_tensors()
    all_q = interp_q.get_tensor_details()

    print(f"  Static INT8 quantisation parameters (weight tensors):")
    print(f"  {'idx':>4}  {'name':35s}  {'dtype':8s}  {'scale':>10s}  {'zero_pt':>8s}")
    print("  " + "-" * 72)
    shown = 0
    for d in all_q:
        q = d.get("quantization_parameters", {})
        scales     = q.get("scales", [])
        zero_points= q.get("zero_points", [])
        if scales and d["dtype"] == np.int8:
            scale_str = f"{scales[0]:.5f}" if len(scales) == 1 else f"{len(scales)}ch"
            zp_str    = str(zero_points[0]) if len(zero_points) == 1 else f"{len(zero_points)}ch"
            print(f"  {d['index']:>4}  {d['name'][:34]:35s}  "
                  f"{d['dtype'].__name__:8s}  {scale_str:>10s}  {zp_str:>8s}")
            shown += 1
            if shown >= 8: break
    if shown < sum(1 for d in all_q if d["dtype"] == np.int8):
        print(f"  ... (more INT8 tensors)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Numerical accuracy comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Accuracy: float32 reference vs quantised models")
print("━" * 65)
print()

if HAS_TF and len(results) >= 2:
    N_TEST = 200
    x_test = np.random.rand(N_TEST, 32, 32, 3).astype(np.float32)

    # Float32 reference outputs
    ref_outputs = []
    if "Float32" in results and results["Float32"][0]:
        interp_ref = tf.lite.Interpreter(model_content=results["Float32"][0])
        interp_ref.allocate_tensors()
        inp_idx = interp_ref.get_input_details()[0]["index"]
        out_idx = interp_ref.get_output_details()[0]["index"]
        for i in range(N_TEST):
            interp_ref.set_tensor(inp_idx, x_test[i:i+1])
            interp_ref.invoke()
            ref_outputs.append(interp_ref.get_tensor(out_idx).copy())
        ref_outputs = np.concatenate(ref_outputs, axis=0)

        print(f"  Numerical comparison vs float32 reference ({N_TEST} samples):")
        print(f"  {'Format':20s}  {'Mean L1':>10s}  {'Max L1':>10s}  "
              f"{'Argmax match':>13s}  {'Size KB':>8s}")
        print("  " + "-" * 70)

        for fmt_name, (tflite_bytes, _) in results.items():
            if tflite_bytes is None or fmt_name == "Float32":
                continue
            interp_q = tf.lite.Interpreter(model_content=tflite_bytes)
            interp_q.allocate_tensors()
            inp_d = interp_q.get_input_details()[0]
            out_d = interp_q.get_output_details()[0]

            q_outputs = []
            for i in range(N_TEST):
                x_in = x_test[i:i+1].astype(inp_d["dtype"])
                interp_q.set_tensor(inp_d["index"], x_in)
                interp_q.invoke()
                out = interp_q.get_tensor(out_d["index"])
                if out.dtype != np.float32:
                    q = inp_d.get("quantization_parameters", {})
                    sc = q.get("scales", [1.0])[0] or 1.0
                    zp = q.get("zero_points", [0])[0] or 0
                    out = (out.astype(np.float32) - zp) * sc
                q_outputs.append(out.copy())

            q_outputs  = np.concatenate(q_outputs, axis=0)
            mean_l1    = float(np.mean(np.abs(ref_outputs - q_outputs)))
            max_l1     = float(np.max(np.abs(ref_outputs - q_outputs)))
            argmax_acc = float(np.mean(np.argmax(ref_outputs, axis=1) ==
                                        np.argmax(q_outputs,  axis=1))) * 100
            size_kb    = len(tflite_bytes) / 1024

            print(f"  {fmt_name:<20s}  {mean_l1:>10.5f}  {max_l1:>10.4f}  "
                  f"{argmax_acc:>12.1f}%  {size_kb:>8.1f}")

        print()
        print("  Interpretation:")
        print("    FP16: near-lossless (float16 rounds differ from float32)")
        print("    Dynamic INT8: moderate deviation (activations still float)")
        print("    Static INT8: most deviation but argmax usually preserved")

else:
    ACC_REF = """
  ACCURACY COMPARISON REFERENCE:

  Typical results on MobileNetV2 on ImageNet:
  ┌─────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Format              │ Top-1 Acc    │ Model size   │ Speedup      │
  ├─────────────────────┼──────────────┼──────────────┼──────────────┤
  │ Float32             │ 71.9%        │ 14 MB        │ 1.0×         │
  │ FP16 weights        │ 71.9%        │  7 MB        │ 1.0–1.5× CPU │
  │ Dynamic range INT8  │ 71.0%        │  3.7 MB      │ 1.5–2× CPU   │
  │ Static INT8 (PTQ)   │ 71.0%        │  3.5 MB      │ 2–4× CPU     │
  │ Static INT8 (QAT)   │ 71.7%        │  3.5 MB      │ 2–4× CPU     │
  │ Static INT8 + NNAPI │ 71.0%        │  3.5 MB      │ 10–20× NPU   │
  └─────────────────────┴──────────────┴──────────────┴──────────────┘
"""
    print(ACC_REF)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · FlatBuffer Deep Dive — Schema, Memory Arena, and Op Registry": {
        "description": (
            "Deep exploration of the .tflite FlatBuffer format and runtime internals. "
            "Parse the FlatBuffer schema directly using flatbuffers Python bindings. "
            "Show every field: tensors, operators, operator_codes, buffers. "
            "Simulate TFLite's memory arena planning — the interval-based tensor sharing. "
            "Show the OpResolver mechanism: how op names map to C++ kernel functions. "
            "Visualise the quantisation parameters for every tensor in an INT8 model."
        ),
        "language": "python",
        "code": '''
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

print("=" * 65)
print("  FLATBUFFER DEEP DIVE — SCHEMA, MEMORY ARENA, AND OP REGISTRY")
print("=" * 65)
print()

try:
    import tensorflow as tf
    HAS_TF = True
    print(f"  TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("  TensorFlow not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: FlatBuffer schema structure (annotated)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — FlatBuffer schema: every field annotated")
print("━" * 65)
print()

SCHEMA_GUIDE = """
  THE .TFLITE FLATBUFFER SCHEMA — ANNOTATED
  ════════════════════════════════════════════════════════════════

  Root object: Model (accessed via Model.GetRootAs(buf, 0))

  ── Model ─────────────────────────────────────────────────────────
  model.Version()          → uint32  (schema version, currently 3)
  model.Description()      → bytes   (human-readable string)
  model.SubgraphsLength()  → int     (usually 1 main subgraph)
  model.BuffersLength()    → int     (count of data buffers)

  model.OperatorCodesLength() → int
  model.OperatorCodes(i).BuiltinCode()  → BuiltinOperator enum
  model.OperatorCodes(i).CustomCode()   → bytes (custom op name)
  model.OperatorCodes(i).Version()      → int (op kernel version)

  BuiltinOperator enum (selected):
    CONV_2D               = 3
    DEPTHWISE_CONV_2D     = 4
    FULLY_CONNECTED       = 9
    SOFTMAX               = 25
    AVERAGE_POOL_2D       = 1
    MAX_POOL_2D           = 17
    RESHAPE               = 22
    RELU                  = 19
    RELU6                 = 21
    CONCATENATION         = 2
    GATHER                = 36
    LSTM                  = 58
    TRANSPOSE_CONV        = 67
    HARD_SWISH            = 117

  ── SubGraph ─────────────────────────────────────────────────────
  sg = model.Subgraphs(0)

  sg.InputsLength()        → int    (number of model inputs)
  sg.Inputs(i)             → int    (index into sg.Tensors)
  sg.OutputsLength()       → int
  sg.Outputs(i)            → int

  sg.TensorsLength()       → int
  sg.OperatorsLength()     → int

  ── Tensor ───────────────────────────────────────────────────────
  t = sg.Tensors(i)

  t.Name()                 → bytes  ("serving_default_input:0")
  t.Buffer()               → int    (index into Model.Buffers)
                                     0 = no data (activation tensor)
                                     >0 = weight/bias tensor
  t.Type()                 → TensorType enum
                                     FLOAT32 = 0
                                     FLOAT16 = 1
                                     INT32   = 2
                                     UINT8   = 3
                                     INT64   = 4
                                     BOOL    = 6
                                     INT16   = 7
                                     COMPLEX64 = 8
                                     INT8    = 9
                                     FLOAT64 = 10
  t.ShapeLength()          → int    (number of dimensions)
  t.Shape(j)               → int    (size of dimension j)
  t.IsVariable()           → bool   (RNN hidden states are variable)

  t.Quantization()         → QuantizationParameters | None
    .ScaleLength()         → int    (1 for per-tensor, C for per-channel)
    .Scale(k)              → float  (scale factor for channel k)
    .ZeroPointLength()     → int
    .ZeroPoint(k)          → long   (zero point for channel k)
    .QuantizedDimension()  → int    (which axis is per-channel)
    .MinLength() / .Min(k) → float  (calibration min value)
    .MaxLength() / .Max(k) → float  (calibration max value)

  ── Operator ─────────────────────────────────────────────────────
  op = sg.Operators(i)

  op.OpcodeIndex()         → int    (index into model.OperatorCodes)
  op.InputsLength()        → int
  op.Inputs(j)             → int    (index into sg.Tensors, -1 = optional absent)
  op.OutputsLength()       → int
  op.Outputs(j)            → int

  op.BuiltinOptionsType()  → BuiltinOptions enum (which union variant)
  op.BuiltinOptions(obj)   → e.g., Conv2DOptions, FullyConnectedOptions, ...

  Example for Conv2D:
    opts = Conv2DOptions()
    op.BuiltinOptions(opts)
    opts.StrideW()          → int   (horizontal stride)
    opts.StrideH()          → int   (vertical stride)
    opts.Padding()          → Padding enum  (SAME=0, VALID=1)
    opts.FusedActivationFunction() → ActivationFunctionType (NONE=0, RELU=1, RELU6=3)

  ── Buffer ────────────────────────────────────────────────────────
  buf = model.Buffers(i)

  buf.DataLength()         → int    (0 for activation buffers)
  buf.Data(j)              → uint8  (j-th byte of weight data)
  buf.Offset()             → ulong  (for external data: file offset)
  buf.Size()               → ulong  (for external data: byte count)

  NOTE: Buffer(0) is always empty — it's the sentinel "no data" buffer.
        All activation tensors point to buffer 0.
        Weight/bias tensors point to buffers 1..N containing raw bytes.
"""
print(SCHEMA_GUIDE)

if HAS_TF:
    # Build and convert a tiny model to inspect
    inp = tf.keras.Input(shape=(8,), name="x")
    h   = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(inp)
    out = tf.keras.layers.Dense(4,  activation="softmax", name="output")(h)
    model = tf.keras.Model(inp, out, name="tiny_mlp")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    # Use TF's Interpreter to inspect the model
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    all_tensors = interp.get_tensor_details()

    print(f"  Inspecting tiny_mlp ({model.count_params()} params):")
    print(f"  Tensors ({len(all_tensors)}):")
    print(f"  {'idx':>4}  {'name':30s}  {'shape':18s}  {'dtype':10s}  {'buffer?'}")
    print("  " + "-" * 75)
    for t in all_tensors:
        # Buffer index > 0 means it carries weight data
        has_data = "weight/bias" if t["index"] not in \
                   [d["index"] for d in interp.get_input_details() +
                    interp.get_output_details()] else "I/O"
        print(f"  {t['index']:>4}  {t['name'][:29]:30s}  "
              f"{str(list(t['shape'])):18s}  "
              f"{t['dtype'].__name__:10s}  {has_data}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Memory arena simulation — tensor lifetime and sharing
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Memory arena planning: tensor sharing simulation")
print("━" * 65)
print()

ARENA_THEORY = """
  HOW TFLITE'S MEMORY PLANNER MINIMISES ACTIVATION MEMORY
  ════════════════════════════════════════════════════════════════

  Problem:
    A model has N activation tensors (intermediate results between ops).
    Each tensor is alive from the op that creates it until the op that
    last reads it. Two tensors can SHARE the same memory if they are
    never alive at the same time.

  Goal:
    Find the minimum total memory needed, given that tensors with
    non-overlapping lifetimes can share memory.

  This is equivalent to GRAPH COLOURING of the interval graph:
    Each tensor is an interval [creation_op, last_use_op].
    Two tensors conflict if their intervals overlap.
    Assign each tensor an "offset" in the arena such that
    conflicting tensors do not overlap in memory.

  Algorithm (greedy interval scheduling):
    Sort tensors by creation time.
    For each tensor:
        Find the earliest offset in the arena where it fits
        (i.e., no currently-alive tensor occupies that space).
    Track: currently_alive_tensors (with their offsets and sizes).
    When a tensor's lifetime ends, free its offset for reuse.

  Result: the arena high-water mark = peak simultaneous live bytes.
"""
print(ARENA_THEORY)

@dataclass
class SimTensor:
    """Simulates a TFLite activation tensor for memory planning."""
    name:       str
    size_bytes: int
    created_at: int   # op index when this tensor is written
    last_used:  int   # op index when this tensor is last read
    offset:     int = -1   # filled by planner

def plan_arena(tensors: List[SimTensor]) -> Tuple[int, List[SimTensor]]:
    """
    Greedy arena memory planner matching TFLite's algorithm.
    Returns (peak_bytes, tensors_with_offsets).
    """
    # Sort by creation time (earliest-created first)
    sorted_tensors = sorted(tensors, key=lambda t: t.created_at)

    # List of (offset, size) currently-occupied regions
    occupied: List[Tuple[int, int]] = []
    peak = 0

    for t in sorted_tensors:
        # Free any tensors that ended before this one starts
        free_offsets = []
        still_alive  = []
        for (off, sz, end_op) in [(o, s, e) for o, s, e in
                                   [(occ[0], occ[1], occ[2]) for occ in
                                    [(off, sz, end)
                                     for (off, sz), end in
                                     zip(occupied,
                                         [tensors[i].last_used
                                          for i in range(len(occupied))])]]]:
            # Rebuild with end info
            pass  # simplified below

        # Simple greedy: find first fit offset with no overlap
        def find_fit(size, avoid: List[Tuple[int, int]]) -> int:
            """Find the first offset where `size` bytes fit without overlap."""
            candidate = 0
            avoid_sorted = sorted(avoid)
            for (off, sz) in avoid_sorted:
                if candidate + size <= off:
                    break           # gap before off is big enough
                candidate = max(candidate, off + sz)
            return candidate

        # Collect currently alive tensors (those whose last_used >= current op)
        alive_ranges = []
        for prev_t in sorted_tensors:
            if prev_t is t: break
            if prev_t.offset >= 0 and prev_t.last_used >= t.created_at:
                alive_ranges.append((prev_t.offset, prev_t.size_bytes))

        t.offset = find_fit(t.size_bytes, alive_ranges)
        peak = max(peak, t.offset + t.size_bytes)

    return peak, sorted_tensors

# Simulate a small CNN's activation tensors
# Operations: [Conv, ReLU, Pool, Conv, ReLU, Flatten, Dense, Softmax]
#              0      1     2     3     4      5        6      7
sim_tensors = [
    SimTensor("input",        3*32*32*4, created_at=0, last_used=0),   # 12 KB float32
    SimTensor("conv1_out",    16*32*32*4,created_at=0, last_used=1),   # 64 KB
    SimTensor("relu1_out",    16*32*32*4,created_at=1, last_used=2),   # 64 KB (same shape)
    SimTensor("pool1_out",    16*16*16*4,created_at=2, last_used=3),   # 16 KB
    SimTensor("conv2_out",    32*16*16*4,created_at=3, last_used=4),   # 32 KB
    SimTensor("relu2_out",    32*16*16*4,created_at=4, last_used=5),   # 32 KB
    SimTensor("flat_out",     32*16*16*4,created_at=5, last_used=6),   # 32 KB (reshaped)
    SimTensor("dense_out",    10*4,       created_at=6, last_used=7),  # 40 bytes
    SimTensor("softmax_out",  10*4,       created_at=7, last_used=7),  # 40 bytes
]

peak_bytes, planned = plan_arena(sim_tensors)
naive_bytes = sum(t.size_bytes for t in planned)

print("  Activation tensors and arena placement:")
print(f"  {'Tensor':20s}  {'Size':>7s}  "
      f"{'Alive [op_in, op_out]':22s}  {'Offset':>8s}  {'End offset':>10s}")
print("  " + "-" * 76)
for t in planned:
    print(f"  {t.name:20s}  {t.size_bytes//1024:>5d}KB  "
          f"[{t.created_at}, {t.last_used}]                    "
          f"{t.offset//1024:>6d}KB  {(t.offset+t.size_bytes)//1024:>8d}KB")

print()
print(f"  Naive total (no sharing):  {naive_bytes/1024:.0f} KB")
print(f"  With arena sharing:        {peak_bytes/1024:.0f} KB")
print(f"  Memory reduction:          {naive_bytes/peak_bytes:.2f}×")
print()
print("  Key insight: conv1_out and relu1_out have the same shape.")
print("  Since relu1_out replaces conv1_out (relu is in-place here),")
print("  they can share memory. TFLite's planner finds this automatically.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Op registry simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Op registry: from builtin_code to kernel function")
print("━" * 65)
print()

OP_REGISTRY = """
  THE TFLITE OP REGISTRY — HOW OPS ARE RESOLVED AT RUNTIME
  ════════════════════════════════════════════════════════════════

  Each .tflite operator node has an opcode_index that maps into
  Model.operator_codes[]. The OperatorCode carries a BuiltinOperator
  enum value (e.g., CONV_2D = 3) and/or a custom_code string.

  At runtime, the Interpreter resolves ops via the OpResolver:

  // In C++ (simplified):
  const TfLiteRegistration* reg = resolver.FindOp(builtin_code, version);
  if (!reg) reg = resolver.FindOp(custom_code, version);
  if (!reg) return kTfLiteError;  // op not supported

  // Registration contains function pointers:
  struct TfLiteRegistration {
    void* (*init)(TfLiteContext*, const char* buffer, size_t length);
    void  (*free)(TfLiteContext*, void* buffer);
    TfLiteStatus (*prepare)(TfLiteContext*, TfLiteNode*);
    TfLiteStatus (*invoke)(TfLiteContext*, TfLiteNode*);  // = eval/run
    const char* builtin_name;
    int         version;
  };

  VERSION MATCHING:
    If a model has CONV_2D v5 but the runtime only has v4:
      The resolver checks: is v5 backward-compatible with v4?
      For TFLite builtins: NO — strict version check.
      kTfLiteError returned; inference fails at AllocateTensors().

  SELECTIVE LINKING (MutableOpResolver):
    For size-constrained builds (MCU, embedded):
      tflite::MutableOpResolver<6> resolver;
      resolver.AddConv2D();           // adds CONV_2D v1..v6
      resolver.AddDepthwiseConv2D();  // adds DEPTHWISE_CONV_2D v1..v7
      resolver.AddFullyConnected();   // adds FULLY_CONNECTED v1..v12
      resolver.AddSoftmax();          // adds SOFTMAX v1..v2
      resolver.AddReshape();          // adds RESHAPE v1
      resolver.AddMean();             // adds MEAN v1..v2
    Only these 6 ops' kernel code is compiled in.
    A model using LSTM would fail at load time (not a runtime error).

  CUSTOM OP REGISTRATION:
    resolver.AddCustom("RoFormer_Attention",
                        Register_RoFormer_Attention());
    // Register_RoFormer_Attention() returns a TfLiteRegistration*
    // with your custom Init, Free, Prepare, Invoke functions.

  BUILT-IN OP VERSIONS AND THEIR FEATURES:
    FULLY_CONNECTED:
      v1: float32, symmetric INT8 weights (static range)
      v4: asymmetric INT8 (per-tensor quantised activations)
      v7: INT16 activations
      v12: per-channel quantised weights with asymmetric activations

    CONV_2D:
      v1: float32 + UINT8 (asymmetric)
      v2: asymmetric INT8 quantisation
      v5: per-channel INT8 weight quantisation
      v6: dilated conv with per-channel quantisation

    GATHER (embedding lookup):
      v1: float32, INT32 indices
      v3: per-channel quantisation support
"""
print(OP_REGISTRY)

if HAS_TF:
    # Show op registry from a real model
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()

    # TFLite Python doesn't expose op codes directly via high-level API,
    # but we can infer them from the tensor connectivity
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    all_t   = {t["index"]: t for t in interp.get_tensor_details()}

    print(f"  tiny_mlp operator sequence:")
    print(f"  {'Op':>3}  {'Input tensors':35s}  {'Output tensors'}")
    print("  " + "-" * 65)
    # Infer op sequence from tensor order (heuristic)
    tensor_indices = sorted(all_t.keys())
    # Weight tensors: have non-zero size AND are not I/O
    io_indices = {d["index"] for d in
                  interp.get_input_details() + interp.get_output_details()}
    weight_tensors = [i for i in tensor_indices
                      if i not in io_indices and
                      np.prod(all_t[i]["shape"]) > 0]
    print(f"    Input:  {all_t[inp_idx]['name']} {list(all_t[inp_idx]['shape'])}")
    for i, wi in enumerate(weight_tensors[:4]):
        t = all_t[wi]
        print(f"    [{i}] weight/bias: {t['name'][:35]} {list(t['shape'])}")
    print(f"    Output: {all_t[out_idx]['name']} {list(all_t[out_idx]['shape'])}")
    print()
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Hardware Delegates — GPU, NNAPI, XNNPACK Benchmarking": {
        "description": (
            "Benchmark TFLite delegates: CPU reference, XNNPACK, and GPU. "
            "Show delegate configuration options and fallback behaviour. "
            "Simulate the delegate op partitioning: which ops go to hardware vs CPU. "
            "Demonstrate the NNAPI delegate compatibility checking. "
            "Measure delegate overhead vs speedup across batch sizes. "
            "Show how to check which delegate executed each op."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

print("=" * 65)
print("  HARDWARE DELEGATES — GPU, NNAPI, XNNPACK BENCHMARKING")
print("=" * 65)
print()

try:
    import tensorflow as tf
    HAS_TF = True
    print(f"  TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("  TensorFlow not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Delegate architecture and the partitioning mechanism
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Delegate partitioning: how ops are assigned to hardware")
print("━" * 65)
print()

PARTITIONING = """
  DELEGATE OP PARTITIONING — STEP BY STEP
  ════════════════════════════════════════════════════════════════

  When you call interpreter.ModifyGraphWithDelegate(delegate) (C++) or
  pass experimental_delegates=[delegate] (Python), TFLite:

  STEP 1 — CAPABILITY QUERY:
    Calls delegate.GetSupportedNodes(graph) → list of op indices.
    Each delegate can inspect op type, version, and tensor dtypes.
    Returns the set of ops it CAN execute (may be a subset of graph).

  STEP 2 — CONTIGUOUS PARTITION DETECTION:
    TFLite groups consecutive supported ops into "delegate partitions."
    A partition is a maximal contiguous run of delegate-supported ops.
    Why contiguous? Data must transfer between CPU and delegate memory
    at partition boundaries. Non-contiguous ops would create too many
    boundaries.

  STEP 3 — PARTITION COMPILATION:
    For each partition, the delegate compiles the ops into a
    hardware-specific plan (e.g., OpenCL kernel, NNAPI model).
    The compiled plan replaces the partition in the graph with a single
    "delegate node" that invokes the plan.

  STEP 4 — INVOKE (at inference time):
    Interpreter encounters a delegate node:
      1. Copy inputs from arena to delegate memory (if needed)
      2. Execute the compiled plan on hardware
      3. Copy outputs from delegate memory back to arena (if needed)
    For XNNPACK (shares CPU memory): steps 1 and 3 are zero-copy.
    For GPU delegate: steps 1 and 3 require CPU↔GPU memory copies.

  EXAMPLE — GPU delegate with a model that has unsupported Gather op:
    Op 0: Conv2D           → GPU partition 1  ✅
    Op 1: ReLU             → GPU partition 1  ✅ (continues)
    Op 2: Gather           → CPU fallback      ❌ (GPU can't do this)
    Op 3: Add              → GPU partition 2  ✅ (new partition)
    Op 4: Softmax          → GPU partition 2  ✅

  Execution order:
    [GPU part. 1: Conv2D+ReLU] → [MemcpyFromGPU] →
    [CPU: Gather] → [MemcpyToGPU] →
    [GPU part. 2: Add+Softmax] → [MemcpyFromGPU]

  COST of partition boundaries:
    Each boundary = 1 MemcpyToDevice + 1 MemcpyFromDevice.
    On Adreno 650: ~0.1–0.5 ms per boundary (small tensors).
    RULE: minimise partition boundaries. One big GPU partition > many small ones.
    RULE: move unsupported ops to the END of the model when possible.

  OP SUPPORT MATRIX (partial):
  ┌──────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ Op                   │ XNNPACK  │ GPU      │ NNAPI 1.3│ Hexagon  │
  ├──────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ CONV_2D              │ FP32/FP16│ FP32/F16 │ INT8     │ INT8     │
  │ DEPTHWISE_CONV_2D    │ FP32/FP16│ FP32/F16 │ INT8     │ INT8     │
  │ FULLY_CONNECTED      │ FP32/FP16│ FP32/F16 │ INT8     │ INT8     │
  │ AVERAGE_POOL_2D      │ FP32/FP16│ FP32/F16 │ INT8     │ INT8     │
  │ SOFTMAX              │ FP32     │ FP32/F16 │ FP32     │ ❌       │
  │ LSTM                 │ FP32     │ ❌       │ FP32     │ ❌       │
  │ GATHER               │ FP32     │ ❌       │ INT8/F32 │ ❌       │
  │ CUSTOM               │ ❌       │ ❌       │ ❌       │ ❌      │
  └──────────────────────┴──────────┴──────────┴──────────┴──────────┘
  (✅ = supported, ❌ = not supported, requires CPU fallback)
"""
print(PARTITIONING)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Benchmarking delegates (simulation + real where available)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Delegate benchmarking: CPU vs XNNPACK vs GPU")
print("━" * 65)
print()

if HAS_TF:
    # Build a model representative of mobile inference (MobileNet-style)
    def make_mobilenet_lite():
        inp = tf.keras.Input(shape=(64, 64, 3))
        x   = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same",
                                      activation="relu")(inp)
        x   = tf.keras.layers.DepthwiseConv2D(3, padding="same",
                                               activation="relu")(x)
        x   = tf.keras.layers.Conv2D(64, 1, activation="relu")(x)
        x   = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same",
                                               activation="relu")(x)
        x   = tf.keras.layers.Conv2D(128, 1, activation="relu")(x)
        x   = tf.keras.layers.GlobalAveragePooling2D()(x)
        x   = tf.keras.layers.Dense(100)(x)
        return tf.keras.Model(inp, x)

    mobile_model = make_mobilenet_lite()
    print(f"  Model: MobileNet-lite ({mobile_model.count_params():,} params)")

    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(mobile_model)
    tflite_bytes = converter.convert()
    model_size_kb = len(tflite_bytes) / 1024
    print(f"  Model size: {model_size_kb:.1f} KB float32")
    print()

    x_bench = np.random.rand(1, 64, 64, 3).astype(np.float32)
    REPS    = 200

    def run_benchmark(tflite_bytes, x, reps=200, delegate=None):
        """Run inference benchmark, return mean latency in ms."""
        try:
            kwargs = {"model_content": tflite_bytes}
            if delegate is not None:
                kwargs["experimental_delegates"] = [delegate]
            interp = tf.lite.Interpreter(**kwargs)
            interp.allocate_tensors()
            inp_d = interp.get_input_details()[0]
            out_d = interp.get_output_details()[0]

            # Warmup
            for _ in range(20):
                interp.set_tensor(inp_d["index"], x)
                interp.invoke()

            t0 = time.perf_counter()
            for _ in range(reps):
                interp.set_tensor(inp_d["index"], x)
                interp.invoke()
            return (time.perf_counter() - t0) / reps * 1000
        except Exception as e:
            return None

    # Test 1: CPU reference (no delegate)
    t_cpu = run_benchmark(tflite_bytes, x_bench)
    print(f"  Latency results (batch=1, 64×64×3 input):")
    print(f"  {'Delegate':25s}  {'Latency (ms)':>12s}  {'Speedup':>9s}  {'Notes'}")
    print("  " + "-" * 65)
    print(f"  {'CPU (no delegate)':25s}  {t_cpu:>12.3f}  {'1.00×':>9s}  "
          f"TFLite reference kernels")

    # Test 2: XNNPACK (default on most platforms)
    try:
        xnnpack_delegate = tf.lite.experimental.load_delegate(
            'libxnnpack_delegate.so') if False else None   ; not available in TF python API directly
        # XNNPACK is enabled by num_threads > 1 in TF Python
        interp_xnn = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=4)
        interp_xnn.allocate_tensors()
        inp_d = interp_xnn.get_input_details()[0]
        for _ in range(20):
            interp_xnn.set_tensor(inp_d["index"], x_bench)
            interp_xnn.invoke()
        t0 = time.perf_counter()
        for _ in range(REPS):
            interp_xnn.set_tensor(inp_d["index"], x_bench)
            interp_xnn.invoke()
        t_xnn = (time.perf_counter() - t0) / REPS * 1000
        print(f"  {'XNNPACK (4 threads)':25s}  {t_xnn:>12.3f}  "
              f"{t_cpu/t_xnn:>9.2f}×  SIMD-optimised CPU kernels")
    except Exception as e:
        print(f"  XNNPACK: not available in this environment ({e})")
        t_xnn = None

    # Test 3: Multi-thread CPU
    interp_mt = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=4)
    interp_mt.allocate_tensors()
    inp_d = interp_mt.get_input_details()[0]
    for _ in range(20):
        interp_mt.set_tensor(inp_d["index"], x_bench)
        interp_mt.invoke()
    t0 = time.perf_counter()
    for _ in range(REPS):
        interp_mt.set_tensor(inp_d["index"], x_bench)
        interp_mt.invoke()
    t_mt = (time.perf_counter() - t0) / REPS * 1000
    print(f"  {'CPU (4 threads)':25s}  {t_mt:>12.3f}  "
          f"{t_cpu/t_mt:>9.2f}×  parallel op execution")
    print()

else:
    DELEGATE_BENCHMARK_REF = """
  DELEGATE BENCHMARK REFERENCE:

  Typical latencies on Pixel 8 (MobileNetV2, 224×224, batch=1):
  ┌──────────────────────────┬──────────────┬──────────────┐
  │ Delegate                 │ Float32 (ms) │ INT8 (ms)    │
  ├──────────────────────────┼──────────────┼──────────────┤
  │ CPU reference (1 thread) │ 58.3         │ 12.1         │
  │ XNNPACK (4 threads)      │ 14.2         │  4.8         │
  │ GPU (FP32)               │ 18.1         │  N/A         │
  │ GPU (FP16)               │  9.7         │  N/A         │
  │ NNAPI → Tensor G3 NPU    │  N/A         │  1.4         │
  │ NNAPI → Adreno (GPU)     │  N/A         │  3.2         │
  └──────────────────────────┴──────────────┴──────────────┘

  Key observations:
  - XNNPACK CPU with NEON ≈ GPU FP32 for small models (memory-bound)
  - GPU FP16 ≈ 2× GPU FP32 (native FP16 SIMD on Adreno)
  - NPU INT8 ≈ 40× CPU reference (9 TOPS vs ~0.05 TOPS effective)
  - NNAPI overhead ~1 ms for model loading/caching (amortised)

  Python delegate usage:
    # XNNPACK (already enabled with num_threads):
    interpreter = tf.lite.Interpreter("model.tflite", num_threads=4)

    # GPU:
    from tensorflow.lite.python.interpreter import load_delegate
    gpu_delegate = load_delegate('libtensorflowlite_gpu_delegate.so',
                                  {'is_precision_loss_allowed': 1})
    interpreter = tf.lite.Interpreter(
        "model.tflite",
        experimental_delegates=[gpu_delegate])

    # NNAPI:
    nnapi_delegate = load_delegate('libnnapi_util.so',
                                    {'execution_preference': 'fast_single_answer'})
    interpreter = tf.lite.Interpreter(
        "model.tflite",
        experimental_delegates=[nnapi_delegate])
"""
    print(DELEGATE_BENCHMARK_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Thread count tuning
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Thread count tuning: intra-op parallelism")
print("━" * 65)
print()

THREADING = """
  TFLITE THREADING MODEL
  ════════════════════════════════════════════════════════════════

  TFLite uses INTRA-OP PARALLELISM (within a single operation):
    The most expensive ops (CONV_2D, FULLY_CONNECTED) partition their
    work across multiple threads via the TFLite thread pool.
    CONV_2D: partitions the output channel dimension across threads.
    FULLY_CONNECTED: partitions the output row dimension.

  Set via: interpreter = tf.lite.Interpreter(..., num_threads=N)
           In C++:  interpreter->SetNumThreads(N)

  CHOOSING N:
    N=1:  No threading overhead. Best for single tiny inference.
          Required for microcontrollers (no thread support).
    N=2:  Good for dual-core MCU/SoC (Cortex-A53 x2).
    N=4:  Standard for quad-core phones (4 performance cores).
    N=-1: Let TFLite choose based on available CPU cores.

  AMDAHL'S LAW APPLIES:
    A Conv2D with 32 output channels: max 32 threads useful.
    A Dense layer with output size 10: max 10 threads useful.
    A Reshape or Softmax: not parallelised (sequential).

    Speedup(N) ≈ 1 / (P_serial + (1-P_serial)/N)
    where P_serial = fraction of non-parallelisable work.

  TYPICAL RESULTS (MobileNetV2, 224×224, ARM Cortex-A55):
    threads=1: 85.3 ms    (1.0×  baseline)
    threads=2: 48.2 ms    (1.8×  near-linear, most work is CONV)
    threads=4: 31.4 ms    (2.7×  Amdahl limit kicking in)
    threads=8: 29.8 ms    (2.9×  diminishing returns, cache contention)

  BEST PRACTICE FOR MOBILE:
    Use num_threads equal to the number of BIG cores (not efficiency cores).
    Big cores on modern phones: 1–4 high-performance Cortex-A78/X1 cores.
    Running on efficiency cores adds contention without proportional benefit.

  THREAD AFFINITY (C++ only):
    For deterministic latency (production serving):
    interpreter->SetCustomAllocationOptions(
        {TfLiteCustomAllocationFlags_FORCE_NON_CACHED});
    // Pin to specific CPU cores via pthread_setaffinity_np()
"""
print(THREADING)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · TFLite Micro — Bare-Metal Inference on Microcontrollers": {
        "description": (
            "Complete guide to TFLite Micro for MCU deployment. "
            "Simulate the TFLM inference loop in Python: static arena, no-malloc pattern. "
            "Show the MicroMutableOpResolver selective linking mechanism. "
            "Convert a keyword spotting model and compute flash/RAM requirements. "
            "Compare TFLM vs TVM microTVM for the same MCU target. "
            "Show the model-as-C-array pattern for embedding models in firmware."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import struct

print("=" * 65)
print("  TFLITE MICRO — BARE-METAL INFERENCE ON MICROCONTROLLERS")
print("=" * 65)
print()

try:
    import tensorflow as tf
    HAS_TF = True
    print(f"  TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("  TensorFlow not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: The no-malloc pattern — simulated in Python
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The no-malloc pattern: TFLM's core design constraint")
print("━" * 65)
print()

NOMALLOC = """
  TFLITE MICRO: THE NO-MALLOC CONTRACT
  ════════════════════════════════════════════════════════════════

  The fundamental constraint of microcontrollers:
    No heap allocator. malloc() is either absent or unsafe.
    All memory must be static (BSS/data segment) or stack-allocated.
    TFLM strictly adheres to this: zero heap allocations at any point.

  HOW TFLM WORKS WITHOUT malloc():

  1. MODEL STORAGE — in flash (read-only):
     The .tflite binary is compiled into the firmware as a C array:

       // model_data.h (generated by xxd or bin2c)
       const unsigned char keyword_detection_model[] = {
           0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,
           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00,
           // ... thousands more bytes ...
       };
       const unsigned int keyword_detection_model_len = 46542;

     The FlatBuffer is read DIRECTLY from this array in flash.
     No copy to RAM required (mmap equivalent on MCU = direct flash access).

  2. TENSOR ARENA — in SRAM (static array):
     All activation tensors live in a single static byte array:

       // In your MCU firmware:
       constexpr int kTensorArenaSize = 100 * 1024;  // 100 KB
       uint8_t tensor_arena[kTensorArenaSize];        // static in BSS

     If kTensorArenaSize is too small: AllocateTensors() returns kTfLiteError.
     You can binary-search the minimum size by trying smaller values.
     TFLite's RecordingMicroInterpreter measures exact requirements.

  3. OP RESOLVER — compiled in:
     Only ops your model uses are linked:

       tflite::MicroMutableOpResolver<5> resolver;
       resolver.AddConv2D();               // +4 KB flash
       resolver.AddDepthwiseConv2D();      // +3 KB flash
       resolver.AddFullyConnected();       // +2 KB flash
       resolver.AddSoftmax();              // +1 KB flash
       resolver.AddReshape();              // +0.2 KB flash

     Total flash budget for these 5 ops: ~10 KB kernel code.
     vs BuiltinOpResolver (all ops): ~200 KB flash.

  4. INTERPRETER — no virtual dispatch, no exceptions:
     tflite::MicroInterpreter interpreter(
         model, resolver,
         tensor_arena, kTensorArenaSize);
     interpreter.AllocateTensors();   // one-time setup, no malloc

  5. INFERENCE LOOP — completely stack-based:
     while (true) {
         // Get feature from sensor (e.g., MFCC from microphone)
         float* input = interpreter.input(0)->data.f;
         ComputeMFCC(audio_buf, input, kFeatureLen);

         // Inference — zero allocation inside here
         TfLiteStatus status = interpreter.Invoke();

         // Read output — direct pointer to arena
         float* output = interpreter.output(0)->data.f;
         int label = std::max_element(output, output+kNumLabels) - output;
     }
"""
print(NOMALLOC)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Convert and analyse a keyword spotting model for MCU
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Convert and analyse a model for MCU deployment")
print("━" * 65)
print()

if HAS_TF:
    # Build a keyword spotting model (DS-CNN inspired)
    # Input: [1, 49, 10, 1] — MFCC spectrogram (49 frames, 10 coefficients)
    inp = tf.keras.Input(shape=(49, 10, 1), name="mfcc_input")
    x   = tf.keras.layers.Conv2D(64, (10,4), padding="same", activation="relu")(inp)
    x   = tf.keras.layers.DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    x   = tf.keras.layers.Conv2D(64, (1,1),  activation="relu")(x)
    x   = tf.keras.layers.DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    x   = tf.keras.layers.Conv2D(64, (1,1),  activation="relu")(x)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(12, activation="softmax")(x)   # 12 keywords
    kws_model = tf.keras.Model(inp, out, name="keyword_spotting")

    print(f"  Model: {kws_model.name}")
    print(f"  Input:  {kws_model.input_shape}  (MFCC spectrogram)")
    print(f"  Output: {kws_model.output_shape} (12 keyword classes)")
    print(f"  Params: {kws_model.count_params():,}")
    print()

    # Convert to INT8 (required for most MCU deployments)
    def kws_representative_dataset():
        for _ in range(100):
            yield [np.random.randn(1, 49, 10, 1).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(kws_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = kws_representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8_bytes = converter.convert()

    size_kb_float = kws_model.count_params() * 4 / 1024
    size_kb_int8  = len(tflite_int8_bytes) / 1024

    print(f"  Model sizes:")
    print(f"    Float32 weights:  {size_kb_float:.1f} KB")
    print(f"    INT8 .tflite:     {size_kb_int8:.1f} KB")
    print(f"    Compression:      {size_kb_float/size_kb_int8:.2f}× (vs raw float weights)")
    print()

    # Measure activation memory requirement
    interp = tf.lite.Interpreter(model_content=tflite_int8_bytes)
    interp.allocate_tensors()

    all_tensors = interp.get_tensor_details()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    # Activation tensors: not the input/output of the model, dtype=int8
    activation_sizes = [
        np.prod(t["shape"]) * np.dtype(t["dtype"]).itemsize
        for t in all_tensors
        if t["dtype"] in [np.int8, np.int16, np.float32]
    ]
    peak_activation_kb = max(activation_sizes) / 1024 if activation_sizes else 0

    print(f"  Tensor analysis:")
    print(f"    Total tensors:         {len(all_tensors)}")
    print(f"    INT8 weight tensors:   {sum(1 for t in all_tensors if t['dtype']==np.int8)}")
    print(f"    Largest activation:    {peak_activation_kb:.1f} KB")
    print()

else:
    TFLM_ANALYSIS_REF = """
  TFLM FLASH + RAM BUDGETS (representative models):
  ─────────────────────────────────────────────────────────────────

  Keyword Spotting (DS-CNN, INT8, 12 keywords):
    .tflite size:     46 KB  → flash storage
    Op kernel code:   55 KB  → flash (TFLM + CMSIS-NN kernels)
    Tensor arena:    100 KB  → SRAM
    Total SRAM:      100 KB  (plus ~8 KB stack)
    MCU target: STM32F4 (192 KB SRAM, 1 MB flash)  ✅ fits

  Person Detection (MobileNetV1 0.25, INT8, 96×96):
    .tflite size:    250 KB  → flash
    Op kernel code:   80 KB  → flash
    Tensor arena:    512 KB  → SRAM
    Total SRAM:      520 KB
    MCU target: STM32H7 (1 MB SRAM) ✅  Arduino Nano ❌ (2 KB SRAM!)

  Anomaly Detection (FC, INT8, 660 params):
    .tflite size:      4 KB  → flash
    Op kernel code:   25 KB  → flash
    Tensor arena:      4 KB  → SRAM
    Total SRAM:        5 KB
    MCU target: ANY Cortex-M0+ ✅ (even 6 KB SRAM Cortex-M0)

  MEASURING EXACT ARENA REQUIREMENTS:
    Use TFLite's RecordingMicroInterpreter:

    tflite::MicroInterpreter recording_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize,
        /*resource_variables_cache=*/nullptr,
        &error_reporter, /*arena_type=*/tflite::RecordingMicroAllocator::kRecordedAllocation
    );
    recording_interpreter.AllocateTensors();
    recording_interpreter.GetMicroAllocator().PrintAllocations();
    // Prints exact bytes used for tensors, ops, scratch buffers
"""
    print(TFLM_ANALYSIS_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Model as C array + the full deployment workflow
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Model as C array: embedding models in firmware")
print("━" * 65)
print()

if HAS_TF:
    # Convert the INT8 model to a C array
    c_array_lines = []
    c_array_lines.append(f"// TFLite INT8 keyword spotting model")
    c_array_lines.append(f"// Generated at build time — do not edit manually")
    c_array_lines.append(f"const unsigned char keyword_model[] = {{")

    bytes_data = tflite_int8_bytes
    hex_lines  = []
    for i in range(0, min(64, len(bytes_data)), 16):   # show first 64 bytes
        chunk = bytes_data[i:i+16]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"  {hex_str},")
    c_array_lines.extend(hex_lines)
    c_array_lines.append(f"  // ... ({len(bytes_data) - 64} more bytes)")
    c_array_lines.append(f"}};")
    c_array_lines.append(f"const unsigned int keyword_model_len = {len(bytes_data)};")

    print("  C array header (first 64 bytes of model):")
    print()
    for line in c_array_lines:
        print(f"  {line}")
    print()

    # Show how to generate this with xxd
    print("  Generation commands:")
    print("    # Using xxd (Linux/macOS):")
    print("    xxd -i model_int8.tflite > model_data.h")
    print()
    print("    # Using Python:")
    print("    with open('model_int8.tflite','rb') as f: data = f.read()")
    print("    hex_str = ','.join(f'0x{b:02x}' for b in data)")
    print("    with open('model_data.h','w') as f:")
    print("        f.write(f'const unsigned char model[] = {{{hex_str}}};\\n')")
    print("        f.write(f'const int model_len = {len(data)};\\n')")
    print()

TFLM_DEPLOYMENT = """
  COMPLETE TFLM DEPLOYMENT WORKFLOW
  ════════════════════════════════════════════════════════════════

  1. TRAIN (Python/TensorFlow):
     model = build_and_train_model(train_data, train_labels)

  2. QUANTISE (TFLiteConverter):
     converter = tf.lite.TFLiteConverter.from_keras_model(model)
     converter.optimizations = [tf.lite.Optimize.DEFAULT]
     converter.representative_dataset = calibration_data_fn
     converter.target_spec.supported_ops = [OpsSet.TFLITE_BUILTINS_INT8]
     converter.inference_input_type  = tf.int8
     converter.inference_output_type = tf.int8
     tflite_bytes = converter.convert()

  3. EMBED IN FIRMWARE (C array):
     xxd -i model_int8.tflite > model_data.cc
     // model_data.cc contains:
     //   const unsigned char model_int8_tflite[] = { 0x18, 0x00, ... };
     //   const unsigned int model_int8_tflite_len = 46542;

  4. CROSS-COMPILE (arm-none-eabi-gcc or PlatformIO):
     // CMakeLists.txt
     add_subdirectory(tensorflow/lite/micro)
     target_link_libraries(firmware tflite-micro)
     add_executable(firmware main.cc model_data.cc)

  5. INFERENCE CODE (main.cc, no OS needed):
     #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
     #include "model_data.h"

     constexpr int kArena = 100 * 1024;
     uint8_t arena[kArena];

     auto* model = tflite::GetModel(model_int8_tflite);
     tflite::MicroMutableOpResolver<5> resolver;
     resolver.AddConv2D(); resolver.AddDepthwiseConv2D();
     resolver.AddFullyConnected(); resolver.AddSoftmax();
     resolver.AddReshape();

     tflite::MicroInterpreter interpreter(model, resolver, arena, kArena);
     interpreter.AllocateTensors();

     while (true) {
         auto* input = interpreter.input(0);
         CollectMFCC(input->data.int8);
         interpreter.Invoke();
         auto* output = interpreter.output(0);
         int keyword = ArgMax(output->data.int8, 12);
     }

  6. FLASH TO MCU:
     # STM32: ST-Link programmer
     st-flash write firmware.bin 0x08000000
     # Arduino: Arduino CLI
     arduino-cli upload -p /dev/ttyUSB0
     # nRF52: nrfjprog
     nrfjprog --family NRF52 --program firmware.hex --chiperase --verify
"""
print(TFLM_DEPLOYMENT)
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · TFLite in the Ecosystem — Benchmarks, Profiling, and LiteRT": {
        "description": (
            "Comprehensive TFLite performance profiling and ecosystem integration. "
            "Use TFLite's built-in profiler to find per-op bottlenecks. "
            "Benchmark TFLite vs ONNX Runtime vs CoreML on standard models. "
            "Show ai-edge-torch: PyTorch → TFLite without going through TensorFlow. "
            "Demonstrate LiteRT's new API surface and MediaPipe Tasks integration. "
            "Summarise TFLite's position in the full connected compiler stack."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TFLITE IN THE ECOSYSTEM — BENCHMARKS, PROFILING, AND LIBERT")
print("=" * 65)
print()

try:
    import tensorflow as tf
    HAS_TF = True
    print(f"  TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("  TensorFlow not installed.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: TFLite profiling — per-op latency breakdown
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — TFLite profiling: finding per-op bottlenecks")
print("━" * 65)
print()

PROFILING = """
  TFLITE PROFILING — THREE METHODS
  ════════════════════════════════════════════════════════════════

  METHOD 1 — Python timing instrumentation (cross-platform):

    import time
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    inp_d = interpreter.get_input_details()
    out_d = interpreter.get_output_details()

    # Manual per-op timing (approximate — ops may be async)
    REPS = 100
    x    = np.random.rand(*inp_d[0]["shape"]).astype(inp_d[0]["dtype"])

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(inp_d[0]["index"], x)
        interpreter.invoke()

    # Timed benchmark
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    import numpy as np
    print(f"Mean latency: {np.mean(times):.3f} ms")
    print(f"P50 latency:  {np.percentile(times, 50):.3f} ms")
    print(f"P95 latency:  {np.percentile(times, 95):.3f} ms")
    print(f"P99 latency:  {np.percentile(times, 99):.3f} ms")

  METHOD 2 — TFLite benchmark tool (Android/Linux):
    # Build the benchmark tool:
    bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
    # Run:
    ./benchmark_model \\
        --graph=model.tflite \\
        --num_runs=50 \\
        --num_threads=4 \\
        --enable_op_profiling=true \\
        --use_xnnpack=true
    # Output:
    # Node Name         | Op Name              | Avg (ms) | Calls | Pct
    # CONV_2D_0         | CONV_2D              | 12.45    | 50    | 34.2%
    # DEPTHWISE_CONV_0  | DEPTHWISE_CONV_2D    | 6.78     | 50    | 18.6%
    # FULLY_CONNECTED   | FULLY_CONNECTED      | 4.12     | 50    | 11.3%
    # ...

  METHOD 3 — C++ ProfilingListener (in app code):
    #include "tensorflow/lite/profiling/profiler.h"

    tflite::profiling::Profiler profiler(/*max_events=*/1024);
    interpreter->SetProfiler(&profiler);

    profiler.StartProfiling();
    interpreter->Invoke();
    profiler.StopProfiling();

    auto events = profiler.GetProfileEvents();
    for (const auto& event : events) {
        printf("%-30s %7.3f ms\\n",
               event.tag, (event.end_time_us - event.begin_time_us) / 1000.0);
    }

  WHAT TO LOOK FOR IN PROFILES:
  ─────────────────────────────────────────────────────────────────
  Dominant Conv2D (>50% time):
    → Use INT8 quantisation + NNAPI NPU or XNNPACK acceleration.
    → Consider MobileNetV3 or EfficientNetLite (fewer channels).

  Many small ops (each <0.1ms, but >50 ops):
    → Operator launch overhead dominates.
    → Use tf.lite.Optimize.DEFAULT to fuse BatchNorm into Conv.
    → Consider larger models with fewer ops.

  Softmax / ArgMax slow:
    → Move classification postprocessing out of TFLite (do in Python/Java).
    → These are not accelerated by most delegates.

  RESHAPE / TRANSPOSE nodes:
    → May indicate layout mismatch (NCHW ↔ NHWC conversions).
    → Ensure your model uses NHWC throughout (TFLite prefers NHWC).
"""
print(PROFILING)

if HAS_TF:
    # Build a realistic CNN for profiling
    inp = tf.keras.Input(shape=(64, 64, 3))
    x   = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
    x   = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x   = tf.keras.layers.Conv2D(64, 1, activation="relu")(x)
    x   = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
    x   = tf.keras.layers.Conv2D(128, 1, activation="relu")(x)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(10)(x)
    cnn = tf.keras.Model(inp, out)

    converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
    tflite_bytes = converter.convert()

    interp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=2)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]

    x_test = np.random.rand(1, 64, 64, 3).astype(np.float32)

    # Per-op timing by measuring individual tensor updates
    REPS = 200
    for _ in range(20):
        interp.set_tensor(inp_d["index"], x_test)
        interp.invoke()

    times = []
    for _ in range(REPS):
        interp.set_tensor(inp_d["index"], x_test)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    times_arr = np.array(times)
    print(f"  CNN profiling (64×64×3 input, {cnn.count_params():,} params):")
    print(f"    Mean latency:    {times_arr.mean():.3f} ms")
    print(f"    P50 latency:     {np.percentile(times_arr, 50):.3f} ms")
    print(f"    P95 latency:     {np.percentile(times_arr, 95):.3f} ms")
    print(f"    Min / Max:       {times_arr.min():.3f} / {times_arr.max():.3f} ms")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: ai-edge-torch — PyTorch native TFLite export
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — ai-edge-torch: PyTorch → TFLite without TensorFlow")
print("━" * 65)
print()

AI_EDGE_TORCH = """
  AI-EDGE-TORCH — PYTORCH NATIVE EXPORT TO TFLITE
  ════════════════════════════════════════════════════════════════

  Traditionally: PyTorch → ONNX → TF SavedModel → .tflite  (3 steps, lossy)
  ai-edge-torch:  PyTorch → .tflite                          (1 step, native)

  Install:  pip install ai-edge-torch

  BASIC CONVERSION:
  ─────────────────────────────────────────────────────────────────
  import ai_edge_torch
  import torch
  import torch.nn as nn

  class MobileBlock(nn.Module):
      def __init__(self, c_in, c_out):
          super().__init__()
          self.dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in)
          self.pw = nn.Conv2d(c_in, c_out, 1)
          self.bn = nn.BatchNorm2d(c_out)
          self.act= nn.ReLU6()
      def forward(self, x):
          return self.act(self.bn(self.pw(self.dw(x))))

  model  = MobileBlock(16, 32).eval()
  sample = (torch.randn(1, 16, 56, 56),)

  # Convert to .tflite
  edge_model = ai_edge_torch.convert(model, sample)
  edge_model.export("block.tflite")

  UNDER THE HOOD:
  1. torch.export.export(model, sample) → ExportedProgram (ATen IR)
  2. TorchDynamo graph capture → FX graph with ATen ops
  3. Lower ATen ops → TFLite MLIR dialect (Google's lowering passes)
  4. Run TFLite MLIR optimisation passes (fusion, constant folding)
  5. FlatBuffer serialiser → model.tflite

  QUANTISATION WITH ai-edge-torch:
  ─────────────────────────────────────────────────────────────────
  from ai_edge_torch.quantize import QuantizationConfig, OpQuantizationConfig
  from ai_edge_torch.quantize import QuantDtype

  # INT8 weight + activation quantisation (PTQ):
  qconfig = QuantizationConfig(
      global_config=OpQuantizationConfig(
          weight_dtype     = QuantDtype.AI_EDGE_W8A8_PT_WEIGHT,
          activation_dtype = QuantDtype.AI_EDGE_W8A8_PT_ACTIVATION,
      ))

  # Run with calibration data:
  edge_model = ai_edge_torch.convert(
      model, sample,
      quant_config=qconfig,
      representative_dataset=calibration_generator,
  )
  edge_model.export("model_int8.tflite")

  # INT4 weight quantisation (for LLM deployment):
  qconfig_4bit = QuantizationConfig(
      global_config=OpQuantizationConfig(
          weight_dtype=QuantDtype.AI_EDGE_W4A8_BT_WEIGHT,   # INT4 weights
      ))

  SUPPORTED ARCHITECTURES (as of 2024):
    MobileNetV1/V2/V3, EfficientNet, EfficientDet
    BERT, DistilBERT (transformer blocks)
    GPT-2 (limited — full sequence generation needs MediaPipe GenAI)
    Vision Transformer (ViT) — partial support
    Custom models using standard PyTorch ops

  LIMITATIONS vs TFLite's native path:
    Python control flow inside forward() may cause graph breaks.
    Custom ops require registration on both PyTorch and TFLite sides.
    Very new ops may not have lowering rules yet.
    Dynamic-shape models require explicit Dim() annotations.
"""
print(AI_EDGE_TORCH)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack position and quick reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — TFLite in the connected compiler stack")
print("━" * 65)
print()

STACK_SUMMARY = """
  TFLITE / LIBERT IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 02: MLIR                                                     │
  │  The entire TFLite converter is MLIR-based.                          │
  │  TF dialect → TFLite dialect → optimisation passes → FlatBuffer.     │
  │  The quantisation insert/propagate/cleanup passes are MLIR passes.   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 05/06: XLA / OpenXLA                                         │
  │  JAX models → StableHLO → TFLite MLIR dialect → .tflite.             │
  │  IREE (OpenXLA project) targets overlapping hardware with TFLite.    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 07: StableHLO                                                │
  │  TFLite converter accepts StableHLO as an input format.              │
  │  stablehlo-to-tfl pass maps stablehlo ops → tfl dialect ops.         │
  │  Path: JAX → StableHLO → TFLite is fully supported.                  │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                      │
  │  tvm.relay.frontend.from_tflite() imports .tflite models.            │
  │  TVM's MetaSchedule finds better kernels than TFLite for custom HW.  │
  │  TVM microTVM targets the same MCUs as TFLite Micro.                 │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 09/10: TorchDynamo / TorchInductor                           │
  │  ai-edge-torch uses TorchDynamo (torch.export) to capture PyTorch.   │
  │  ATen ops from TorchDynamo are lowered to TFLite MLIR.               │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 10: ONNX                                                     │
  │  ONNX → TFLite: onnx-tf (ONNX → SavedModel → .tflite).               │
  │  TFLite → ONNX: tf2onnx (TFLite → SavedModel → ONNX).                │
  │  Indirect route — ONNX is not a direct TFLite input format.          │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 11: TFLite / LiteRT (THIS MODULE)                            │
  │  FlatBuffers format, MLIR converter, INT8 quantisation,              │
  │  memory arena, delegates (XNNPACK/GPU/NNAPI/Hexagon/CoreML),         │
  │  TFLite Micro (MCU), ai-edge-torch, LiteRT/MediaPipe LLMs.           │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK_SUMMARY)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Task                         │ API / Tool                        │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Convert Keras → TFLite       │ tf.lite.TFLiteConverter.from_keras_model() │")
print("  │ Convert PyTorch → TFLite     │ ai_edge_torch.convert()           │")
print("  │ Convert JAX → TFLite         │ TFLiteConverter.experimental_from_jax()   │")
print("  │ Dynamic range quantisation   │ converter.optimizations=[DEFAULT] │")
print("  │ Static INT8 quantisation     │ + representative_dataset + INT8 ops│")
print("  │ QAT                          │ tensorflow_model_optimization      │")
print("  │ Run inference (Python)       │ tf.lite.Interpreter                │")
print("  │ Enable XNNPACK               │ Interpreter(num_threads=4)         │")
print("  │ Enable GPU delegate          │ load_delegate('libgpu_delegate.so')│")
print("  │ Enable NNAPI                 │ load_delegate('libnnapi_util.so')  │")
print("  │ Save model to C array        │ xxd -i model.tflite > model_data.h│")
print("  │ MCU interpreter              │ tflite::MicroInterpreter           │")
print("  │ MCU op resolver              │ MicroMutableOpResolver<N>          │")
print("  │ Find per-op timings          │ TFLite benchmark tool + profiling  │")
print("  │ Inspect FlatBuffer           │ tf.lite.Interpreter.get_tensor_details() │")
print("  │ LiteRT package               │ pip install ai-edge-litert         │")
print("  │ MediaPipe Tasks              │ pip install mediapipe              │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Project (TFLite/LiteRT)      │ github.com/tensorflow/tensorflow   │")
print("  │                              │ tensorflow/lite/                   │")
print("  │ TFLite Micro                 │ github.com/tensorflow/tflite-micro │")
print("  │ ai-edge-torch (PyTorch)      │ github.com/google-ai-edge/ai-edge-torch │")
print("  │ LiteRT (new name)            │ ai.google.dev/edge/litert          │")
print("  │ MediaPipe Tasks              │ ai.google.dev/edge/mediapipe/solutions │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()

if HAS_TF:
    print("  Runtime verification:")
    # Quick end-to-end correctness check
    inp_v = tf.keras.Input(shape=(8,))
    out_v = tf.keras.layers.Dense(4, activation="softmax")(inp_v)
    m_v   = tf.keras.Model(inp_v, out_v)
    x_v   = np.random.rand(3, 8).astype(np.float32)

    with tf.no_grad := (lambda: None):
        pass

    # TF reference
    ref = m_v(x_v).numpy()

    # TFLite inference
    conv = tf.lite.TFLiteConverter.from_keras_model(m_v)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_b = conv.convert()
    interp_v = tf.lite.Interpreter(model_content=tflite_b)
    interp_v.allocate_tensors()
    inp_dd = interp_v.get_input_details()[0]
    out_dd = interp_v.get_output_details()[0]
    results_v = []
    for i in range(len(x_v)):
        interp_v.set_tensor(inp_dd["index"], x_v[i:i+1])
        interp_v.invoke()
        results_v.append(interp_v.get_tensor(out_dd["index"]).copy())
    tflite_out = np.concatenate(results_v)

    max_err = float(np.max(np.abs(ref - tflite_out)))
    print(f"    Dense(8→4 softmax): TF vs TFLite dynamic-INT8 max_err={max_err:.2e} "
          f"{'✅' if max_err < 0.02 else '⚠️'}")
    print(f"    Model size: {len(tflite_b)/1024:.1f} KB (dynamic INT8 quantised)")
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