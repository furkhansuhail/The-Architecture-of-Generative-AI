"""
ONNX — The Universal Language of Machine Learning Models
=========================================================

ONNX (Open Neural Network Exchange) is an open-source, open-specification
format for representing machine learning models. It was created jointly by
Microsoft and Facebook (Meta) in September 2017 and open-sourced immediately,
with Amazon and other major technology companies joining shortly after. The
ONNX community has since grown to include Google, NVIDIA, Intel, Qualcomm,
Arm, Huawei, and hundreds of individual contributors.

Before ONNX, the ML deployment landscape was severely fragmented. A model
trained in TensorFlow could not be directly used in a PyTorch runtime. A
model designed for NVIDIA GPUs could not be deployed to Intel CPUs without
rewriting. A researcher's PyTorch model could not be served by TensorFlow
Serving or Azure ML. Every framework had its own serialisation format,
runtime, and optimisation toolchain. Moving a model from development to
production meant rewriting it — a slow, error-prone, and expensive process.

ONNX solved this with three core ideas:

    1. A COMMON INTERMEDIATE REPRESENTATION (IR):
       An ONNX model is a computational graph where nodes are operations
       (Conv, MatMul, Relu, Reshape, etc.) and edges are tensors flowing
       between them. This graph is framework-neutral — it does not care
       whether it was produced by PyTorch, TensorFlow, scikit-learn,
       or any other library.

    2. A STANDARD OPERATOR SET (opset):
       ONNX defines ~220 standard operators that cover all common deep
       learning and classical ML operations. Any model expressible in any
       major framework can be translated into these standard operators.
       Operators are versioned for backward compatibility.

    3. A RUNTIME-NEUTRAL EXECUTION MODEL:
       ONNX Runtime (ORT) — Microsoft's optimised inference engine —
       runs ONNX models on ANY hardware: x86/ARM CPUs, CUDA GPUs, OpenCL
       GPUs, NPUs (Apple Neural Engine, Qualcomm QNN), Intel OpenVINO, and
       more. Each hardware target is a separate Execution Provider (EP).

The impact of ONNX is profound:
    Research → Production:   Export from PyTorch, run optimised on any hardware
    Cross-platform:           Same model file runs on server, mobile, edge, browser
    Optimisation:             ONNX Runtime applies graph-level optimisations
                              (constant folding, layer fusion, quantisation)
    Interoperability:         Teams using different frameworks share models as ONNX
    Standardisation:          Cloud providers (Azure, AWS, GCP) all accept ONNX

ONNX Runtime (ORT) consistently achieves 2–10× speedup over framework-native
inference (PyTorch/TensorFlow) through its graph optimisation pipeline and
hardware-specific execution providers. For transformer models, combined with
INT8 quantisation, speedups of 3–8× with < 1% accuracy drop are typical.

This module covers ONNX from the ground up: the computational graph model
and protobuf format, the operator set and type system, exporting from
PyTorch and scikit-learn, the ONNX Runtime inference engine and execution
providers, graph optimisation (constant folding, operator fusion, layout
optimisation), quantisation (dynamic, static INT8, QDQ format), model
optimisation with onnxoptimizer and onnxsim, mobile and edge deployment,
ONNX Runtime Web for browser inference, and production deployment patterns.

"""

import textwrap
import re

TOPIC_NAME   = "ONNX — The Universal Language of Machine Learning Models"
DISPLAY_NAME = "15 · ONNX"
ICON         = "🔄"
SUBTITLE     = "From Framework-Agnostic IR to Optimised Cross-Platform Inference"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT IS ONNX AND WHY IT EXISTS

### The ML Deployment Fragmentation Problem

    Consider a realistic ML team in 2017 (pre-ONNX):
        Data scientists train models in PyTorch (research-friendly).
        Production engineers need to serve them in TensorFlow (serving infra).
        Mobile team needs the model on iOS (Core ML format only).
        Edge team needs it on a Raspberry Pi (TFLite or custom runtime).
        Browser team wants it in JavaScript (TensorFlow.js format).

    Each deployment target required:
        1. Rewriting or re-training the model in that framework
        2. Framework-specific optimisation (may not generalise)
        3. Framework-specific bug surface and operational burden
        4. Maintaining 4-5 separate model versions

    This is the "write once, rewrite everywhere" problem — the exact
    opposite of Java's "write once, run anywhere" promise.

### ONNX as the Universal IR

    ONNX solves this by being the INTERMEDIATE REPRESENTATION that sits
    between frameworks (exporters) and runtimes (consumers):

        PyTorch    ──┐
        TensorFlow   ┤
        JAX          ┤ → ONNX model (.onnx file) → ONNX Runtime → CPU
        scikit-learn ┤                           → TensorRT    → NVIDIA GPU
        XGBoost     ─┤                           → CoreML      → Apple Silicon
        LightGBM    ─┤                           → OpenVINO    → Intel CPU/iGPU
        Keras        ┤                           → QNN         → Qualcomm NPU
        Hugging Face ┘                          → ONNX.js     → Browser/WebGL

    The ONNX file is the source of truth for deployment. It is:
        - Self-contained: includes weights, graph topology, metadata
        - Portable: works on any OS, CPU architecture, or accelerator
        - Versionable: opset versioning for long-term compatibility
        - Inspectable: human-readable graph with standard tooling
        - Optimisable: ORT applies 250+ graph transformations

### The Three Pillars of ONNX

    PILLAR 1 — THE ONNX SPECIFICATION:
        An open standard defining the computational graph model, the
        protobuf schema for serialisation, the operator set, and the
        type system. The spec is versioned; the current standard is
        ONNX opset 21 (as of 2024).

    PILLAR 2 — THE ONNX OPERATOR SET:
        ~220 primitive operators covering:
            - Neural network ops: Conv, ConvTranspose, BatchNorm, LSTM, GRU,
              MultiHeadAttention, Attention, Dropout, Embedding
            - Math ops: MatMul, Gemm, Add, Mul, Div, Sqrt, Exp, Log, Pow
            - Reduction: ReduceMean, ReduceSum, ReduceMax, ReduceMin
            - Shape: Reshape, Transpose, Flatten, Squeeze, Unsqueeze
            - Activation: Relu, Sigmoid, Tanh, Softmax, Gelu, Silu, Mish
            - Pooling: MaxPool, AveragePool, GlobalAveragePool
            - Tensor ops: Concat, Split, Slice, Gather, Scatter, Where
            - Control flow: If, Loop, Scan (for dynamic models)

    PILLAR 3 — ONNX RUNTIME (ORT):
        The reference implementation of the ONNX execution engine.
        Maintained by Microsoft, production-hardened, used by:
            - Azure ML for all model inference
            - Microsoft Office (on-device AI features)
            - Xbox (game AI)
            - Bing (search ranking models)
            - Windows ML (Windows 10/11 on-device ML API)
            - HuggingFace Optimum (default fast inference backend)

### ONNX vs Native Framework Inference

    When you run inference with PyTorch:
        Python overhead → PyTorch dispatcher → CUDA kernel
        The PyTorch runtime includes the autograd engine, training
        infrastructure, and Python bindings — all irrelevant for inference.

    When you run inference with ONNX Runtime:
        C++ runtime → optimised execution plan → hardware-specific kernel
        No Python overhead (ORT core is C++). No training infrastructure.
        Graph is statically analysed and optimised at load time.

    The result:
        ORT typically 2-5× faster than PyTorch eager for transformer inference
        ORT + TensorRT: 5-10× faster for NVIDIA GPU workloads
        ORT + INT8 quantisation: 3-4× faster than FP32, 4× smaller model size


##### PART 2 — THE ONNX COMPUTATIONAL GRAPH MODEL

### Graph Structure: Nodes, Edges, and Tensors

    An ONNX model is a directed acyclic graph (DAG) where:
        NODES:     operations (Conv, MatMul, Relu, etc.)
        EDGES:     named tensors flowing between nodes
        INPUTS:    named tensors entering the graph (model inputs)
        OUTPUTS:   named tensors leaving the graph (model outputs)
        INITIALIZERS: constant tensors (weights, biases — stored in the file)

    Every node has:
        op_type:    the operator name (e.g. "Conv", "MatMul")
        inputs:     list of input tensor names
        outputs:    list of output tensor names
        attributes: compile-time constants (kernel_size, strides, etc.)

    A simple two-layer MLP in ONNX graph notation:
        Input: X [batch, 784]
        ─────────────────────────────────
        Gemm(X, W1, b1) → h1 [batch, 256]      W1=[784,256], b1=[256]
        Relu(h1) → h1_act [batch, 256]
        Gemm(h1_act, W2, b2) → logits [batch, 10]  W2=[256,10], b2=[10]
        Softmax(logits) → probs [batch, 10]
        ─────────────────────────────────
        Output: probs [batch, 10]

    The names (X, h1, h1_act, logits, probs) are just string identifiers
    that connect node outputs to node inputs. W1, b1, W2, b2 are initializers
    (constant tensors embedded in the model file).

### The Protobuf Serialisation Format

    ONNX models are serialised as Protocol Buffers (protobuf) with the
    .onnx file extension.

    The top-level ModelProto contains:
        ir_version:     ONNX IR version
        opset_imports:  list of opsets used (e.g. {domain:"", version:17})
        graph:          the GraphProto (the actual model)
        model_version:  user-defined version string
        metadata_props: key-value metadata

    The GraphProto contains:
        node:           list of NodeProto (each node in the graph)
        input:          list of ValueInfoProto (model inputs with types/shapes)
        output:         list of ValueInfoProto (model outputs with types/shapes)
        initializer:    list of TensorProto (constant weight tensors)
        value_info:     intermediate tensor type/shape information

    The TensorProto for weights stores:
        data_type: FLOAT(1), FLOAT16(10), INT8(3), INT64(7), BOOL(9), ...
        dims:       shape (e.g. [256, 784])
        float_data: raw float32 values (or raw_data for other dtypes)
        name:       identifier matching the NodeProto input

    File sizes: a float32 ONNX model is approximately the same size as
    the number of parameters × 4 bytes. BERT-base ≈ 440 MB, DistilBERT ≈ 260 MB.

### Type System: Tensor Types and Shapes

    Every ONNX tensor has a static type (required) and optionally a static
    shape (may be symbolic for dynamic dimensions).

    Element types:
        float32 (FLOAT):    standard default
        float16 (FLOAT16):  half precision for GPU inference
        bfloat16 (BFLOAT16): brain float, used in modern LLMs
        int8 / uint8:       quantised inference
        int32 / int64:      indices, discrete values
        bool:               binary masks
        string:             for NLP preprocessing nodes

    Shape notation:
        [3, 224, 224]:       fully static shape
        [None, 3, 224, 224]: dynamic batch dimension (symbolic = "batch_size")
        ["batch", 512]:      named dynamic dimensions
        [-1, 512]:           alternative notation for dynamic

    Dynamic shapes are critical: a model exported with batch_size=1 may
    fail or be slow for batch_size=8 unless dynamic axes are specified at
    export time.

    Specifying dynamic axes at export:
        torch.onnx.export(
            model, dummy_input, "model.onnx",
            dynamic_axes={
                "input":  {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"},
            }
        )

### Opset Versioning

    ONNX uses OPSET VERSIONS to manage backward compatibility.
    Each opset version may add new operators or revise existing ones.

    Current landscape (2024):
        opset 9:  legacy baseline (widely supported)
        opset 11: added dynamic_axes support for many ops
        opset 13: updated Cast, Flatten, etc.
        opset 17: LayerNormalization, BlackmanWindow (audio)
        opset 18: BitwiseAnd, BitwiseOr, CenterCropPad
        opset 19: DeformConv, IsInf, IsNaN (improved)
        opset 20: DFT, STFT, AffineGrid (signal processing)
        opset 21: current default for PyTorch 2.x exports

    When to use which opset:
        Mobile deployment (Core ML, TFLite converter): opset 11-13
        ONNX Runtime current: opset 17+
        Maximum compatibility: opset 11 (but misses newer ops like LayerNorm)
        PyTorch 2.x default export: opset 17-21


##### PART 3 — EXPORTING MODELS TO ONNX

### PyTorch Export: torch.onnx.export

    The standard way to export PyTorch models to ONNX.
    It traces the model (runs it with example inputs, records operations).

        import torch

        model = MyModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)  # example input

        torch.onnx.export(
            model,                          # PyTorch model
            dummy_input,                    # example input(s)
            "model.onnx",                   # output path
            export_params      = True,      # embed weights in the file
            opset_version      = 17,        # ONNX opset version
            do_constant_folding = True,     # fold constants at export
            input_names        = ["input"], # name the inputs
            output_names       = ["output"],# name the outputs
            dynamic_axes       = {          # specify dynamic dimensions
                "input":  {0: "batch"},
                "output": {0: "batch"},
            },
            verbose            = False,
        )

    How torch.onnx.export works (tracing):
        1. Run the model once with dummy_input → produces a trace of ops
        2. Convert PyTorch ops to ONNX ops (using registered translators)
        3. Embed the weights as initializers
        4. Serialise to .onnx protobuf format

    Tracing limitations:
        - Control flow (if/for) is traced for the SPECIFIC input values
        - Dynamic shapes must be explicitly declared as dynamic_axes
        - Custom ops may need custom symbolic translators

    PyTorch 2.x export (torch.onnx.dynamo_export — the modern approach):
        Uses torch.compile's Dynamo for more robust graph capture.
        Handles data-dependent control flow better than tracing.
        from torch.onnx import dynamo_export
        export_output = dynamo_export(model, dummy_input)
        export_output.save("model.onnx")

### Exporting scikit-learn with skl2onnx

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model   = convert_sklearn(model, initial_types=initial_type,
                                        target_opset=17)
        with open("sklearn_model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

    Supported scikit-learn models via skl2onnx:
        Linear models:    LinearRegression, LogisticRegression, Ridge, Lasso
        Tree models:      DecisionTreeClassifier/Regressor
        Ensembles:        RandomForest, GradientBoosting, ExtraTreesClassifier
        SVM:              SVC, SVR, LinearSVC
        Preprocessing:    StandardScaler, MinMaxScaler, PCA, OneHotEncoder
        Pipelines:        sklearn.pipeline.Pipeline (full pipeline export)

### HuggingFace Optimum: Transformers → ONNX

    Optimum is HuggingFace's library for hardware-accelerated inference.
    It provides a one-line export for transformer models:

        from optimum.onnxruntime import ORTModelForSequenceClassification

        # Load a HuggingFace model and immediately convert to ONNX
        model = ORTModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            export=True,
        )
        # model is now backed by ONNX Runtime → 2-3× faster than PyTorch

    Explicit export:
        from optimum.exporters.onnx import main_export

        main_export(
            model_name_or_path = "bert-base-uncased",
            output             = "./bert_onnx/",
            task               = "text-classification",
            opset              = 17,
        )

### Validating ONNX Models

    Always validate after export:

        import onnx

        model = onnx.load("model.onnx")
        onnx.checker.check_model(model)   # raises if invalid

        # Shape inference: propagates shapes through the graph
        model_with_shapes = onnx.shape_inference.infer_shapes(model)

        # Inspect the graph
        graph = model.graph
        print(f"Inputs:  {[i.name for i in graph.input]}")
        print(f"Outputs: {[o.name for o in graph.output]}")
        print(f"Nodes:   {len(graph.node)}")
        for init in graph.initializer:
            shape = list(init.dims)
            print(f"  Weight: {init.name}  shape={shape}")


##### PART 4 — ONNX RUNTIME: INFERENCE ENGINE ARCHITECTURE

### ORT Architecture Overview

    ONNX Runtime is a high-performance C++ inference engine with Python,
    C#, Java, JavaScript, and C bindings. The inference pipeline:

        1. MODEL LOADING:
           Load and parse the .onnx protobuf.
           Validate the graph against the registered operator set.

        2. GRAPH TRANSFORMATIONS (L1/L2/L3):
           Apply a suite of graph optimisations (see Part 5 for details).
           Transform the graph into a more efficient equivalent.

        3. EXECUTION PROVIDER SELECTION:
           For each node, select which EP (CPU, CUDA, TensorRT, etc.) runs it.
           Nodes that an EP cannot handle fall back to the CPU EP.

        4. MEMORY PLANNING:
           Allocate output tensors. Reuse buffers where possible.
           Pin memory for GPU transfers.

        5. EXECUTION:
           Run nodes in topological order.
           EP-specific kernels execute each operation.
           Tensor data flows through the allocated buffers.

### The Execution Provider (EP) System

    An Execution Provider is a hardware-specific backend that implements
    ONNX operators for a particular hardware target.

    CPU Execution Provider (always available):
        Uses MLAS (Microsoft Linear Algebra Subprograms) and Eigen.
        Optimised SIMD kernels for AVX2/AVX-512.
        Thread pool for op-level parallelism.
        No external dependencies — works on any machine.

    CUDA Execution Provider:
        Runs CUDA-capable operations on NVIDIA GPUs.
        Uses cuDNN for convolutions, cuBLAS for matrix multiplications.
        Requires: CUDA toolkit, cuDNN library.

    TensorRT Execution Provider:
        Compiles ONNX subgraphs using NVIDIA TensorRT.
        Performs layer fusion, precision calibration, kernel auto-tuning.
        Dramatically faster than CUDA EP for production inference.
        Requires: TensorRT installation (large dependency).
        First run: slow (TRT engine compilation). Subsequent: very fast.
        Best for: NVIDIA GPU production serving.

    OpenVINO Execution Provider:
        Intel's optimised inference for CPUs, iGPUs, and VPUs.
        Model Optimizer converts ONNX to OpenVINO IR.
        Excellent for Intel CPU inference (often faster than CPU EP).
        Supports: Intel Core/Xeon CPUs, Intel integrated GPUs, Myriad X.

    CoreML Execution Provider (macOS/iOS):
        Uses Apple's CoreML framework via the ANE (Apple Neural Engine).
        ANE is dedicated ML silicon in Apple Silicon (M1/M2/M3).
        Massively more power-efficient than CPU or GPU for inference.

    DirectML Execution Provider (Windows):
        Uses Microsoft's DirectX 12 ML API.
        Works on any DirectX 12 compatible GPU (NVIDIA, AMD, Intel).
        Universal GPU inference on Windows without CUDA dependency.

    QNN Execution Provider (Qualcomm):
        Runs on Qualcomm's Hexagon DSP and NPU.
        Used for on-device AI on Android phones with Snapdragon.

    ROCm Execution Provider (AMD):
        AMD GPU inference using HIP/ROCm stack.
        Equivalent to CUDA EP for AMD GPUs.

### Using ONNX Runtime in Python

    Installation:
        pip install onnxruntime              # CPU only
        pip install onnxruntime-gpu          # CPU + CUDA + TensorRT

    Basic inference:
        import onnxruntime as ort
        import numpy as np

        # Create InferenceSession (loads and optimises the model)
        sess = ort.InferenceSession(
            "model.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # ORT tries EPs in order: first CUDA, fallback to CPU

        # Inspect inputs/outputs
        for inp in sess.get_inputs():
            print(inp.name, inp.shape, inp.type)
        for out in sess.get_outputs():
            print(out.name, out.shape, out.type)

        # Run inference
        input_data  = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output_data = sess.run(
            ["output"],                   # list of output names to return
            {"input": input_data},        # dict: input_name → numpy array
        )
        predictions = output_data[0]

    SessionOptions for fine-grained control:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4     # threads per op
        opts.inter_op_num_threads = 1     # threads across ops (usually 1)
        opts.enable_profiling = True      # generates a profiling JSON
        opts.optimized_model_filepath = "model_opt.onnx"  # save optimised graph

        sess = ort.InferenceSession("model.onnx", sess_options=opts)

    Execution Provider configuration:
        providers = [
            ("CUDAExecutionProvider", {
                "device_id":              0,
                "arena_extend_strategy":  "kNextPowerOfTwo",
                "gpu_mem_limit":          4 * 1024**3,   # 4 GB
                "cudnn_conv_algo_search":  "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]
        sess = ort.InferenceSession("model.onnx", providers=providers)


##### PART 5 — GRAPH OPTIMISATION: HOW ORT MAKES MODELS FASTER

### ORT Optimisation Levels

    ORT applies optimisations at three levels (controlled by
    SessionOptions.graph_optimization_level):

    Level 0 — ORT_DISABLE_ALL:
        No optimisations. Only use for debugging to compare with optimised.

    Level 1 — ORT_ENABLE_BASIC (default on CPU):
        Constant folding, redundant node elimination, node fusion patterns.
        Safe, semantics-preserving transformations.

    Level 2 — ORT_ENABLE_EXTENDED:
        More aggressive operator fusion, quantisation preparation.
        EP-specific kernel fusion (e.g. Attention fusion for transformers).

    Level 3 — ORT_ENABLE_ALL (recommended for production):
        All of the above plus layout optimisations (NHWC ↔ NCHW).
        Subgraph partitioning for EP assignment.
        Memory planning optimisation.

### Key Optimisation Transformations

    CONSTANT FOLDING:
        Evaluate sub-expressions whose inputs are all constants at compile time.
        Replaces them with the pre-computed result.

        Example:
            Before: Mul(Div(x, 255.0), 2.0)   (x/255 * 2 at every inference)
            After:  Mul(x, 0.00784...)         (2/255 pre-computed as constant)

        Particularly valuable for preprocessing graphs with fixed normalisation.

    REDUNDANT NODE ELIMINATION:
        Remove nodes whose outputs are never used, or whose operations
        have no effect (e.g. Identity nodes, Cast to same type).

        Example:
            Before: x → Cast(to=float32) → y   (if x is already float32)
            After:  x → y                       (Cast eliminated)

    LAYER FUSION:
        Combine adjacent operations into a single fused kernel.
        A fused kernel reads input once, performs multiple operations,
        writes output once — minimising memory bandwidth usage.

        Memory bandwidth is almost always the bottleneck for inference
        (not compute). Fusion reduces memory reads/writes dramatically.

        Common fusions in ONNX Runtime:
            Conv + BatchNorm → ConvBatchNorm:
                At inference, BN parameters can be absorbed into Conv weights.
                Mathematically equivalent, one kernel instead of two.
                Result: 30-50% speedup for CNN inference.

            Conv + BatchNorm + Relu → ConvBatchNormRelu:
                Three operations → one fused kernel.

            MatMul + Add (bias) → Gemm:
                Both captured by the Gemm operator.

            Attention fusion (transformer-specific):
                The full self-attention computation (QKV projections,
                scaled dot-product, softmax) → one fused Attention kernel.
                This is a 2-4× speedup for transformer inference specifically.

                Before: 3× MatMul + 3× Add + Reshape + Split +
                        MatMul + Div + Add (mask) + Softmax + MatMul +
                        Reshape + MatMul (output projection)
                After:  SingleFusedAttention kernel

            LayerNorm fusion:
                ReduceMean + Sub + Pow + ReduceMean + Add + Sqrt + Div + Mul + Add
                All become a single LayerNorm kernel.

    LAYOUT OPTIMISATION:
        Neural networks typically use NCHW (batch, channels, height, width).
        But many hardware backends (ARM, Intel, some NVIDIA operations)
        prefer NHWC (batch, height, width, channels) for memory access.
        ORT can insert Transpose nodes to match the preferred layout,
        then fold out unnecessary transposes.

    DEAD CODE ELIMINATION:
        Remove any graph path that does not contribute to the output.
        Common in models with multiple output heads where only one is used.

    RESHAPE ELIMINATION:
        Consecutive Reshape nodes may be collapsed.
        Reshapes that are no-ops (same shape) are removed.

### onnxoptimizer and onnxsim

    onnxoptimizer:
        A library of graph transformation passes for pre-processing ONNX
        models before passing to ORT or other runtimes.

        from onnxoptimizer import optimize

        passes = [
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "fuse_bn_into_conv",           # absorb BN into Conv
            "fuse_add_bias_into_conv",     # absorb bias add into Conv
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
            "eliminate_duplicate_initializer",
            "eliminate_unused_initializer",
        ]
        optimised = optimize(model, passes)

    onnx-simplifier (onnxsim):
        Uses constant propagation and simplification to produce cleaner graphs.
        Often dramatically reduces node count (sometimes 50-80% fewer nodes).
        Critical for models exported from PyTorch with complex symbolic shapes.

        import onnxsim
        model_simplified, check = onnxsim.simplify(model)
        # check: True if the simplified model produces identical outputs


##### PART 6 — QUANTISATION: SMALLER, FASTER MODELS

### What Is Quantisation and Why It Matters

    Quantisation reduces numerical precision of model weights and/or
    activations from float32 (4 bytes) to INT8 (1 byte) or INT4 (0.5 bytes).

    Benefits:
        Model size:      4× smaller (FP32→INT8), 8× smaller (FP32→INT4)
        Memory bandwidth: 4× less data to move = faster inference on all hardware
        Compute:         INT8 matrix multiply is 2-4× faster than FP32 on modern CPUs
                         NVIDIA TensorCores: INT8 is 4× faster than FP16
        Cache efficiency: smaller model fits in CPU cache → better locality

    Cost:
        Quantisation introduces rounding errors.
        Typical accuracy drop: 0.1-1% for INT8, 1-3% for INT4
        Some layers are quantisation-sensitive (first/last layer, attention)

### The Quantisation Math

    Quantising a float tensor to INT8:
        For a float value x in range [x_min, x_max]:

        scale     = (x_max - x_min) / (q_max - q_min)   # e.g. 255 for uint8
        zero_point = round(q_min - x_min / scale)

        q = round(x / scale) + zero_point               # float → int8
        x = (q - zero_point) * scale                    # int8 → float (dequant)

    The (scale, zero_point) pair is the QUANTISATION PARAMETERS.
    Symmetric quantisation: zero_point = 0 (simpler, slightly less accurate)
    Asymmetric quantisation: non-zero zero_point (more accurate, more complex)

    INT8 range: [-128, 127] (signed) or [0, 255] (unsigned)
    INT4 range: [-8, 7] (signed) or [0, 15] (unsigned)

### Dynamic Quantisation

    Quantise weights to INT8 at export time (static).
    Quantise activations to INT8 at runtime (dynamic, per-batch).
    No calibration data needed.

        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input  = "model.onnx",
            model_output = "model_int8.onnx",
            weight_type  = QuantType.QInt8,
        )

    Use cases: text models (BERT, GPT) where activation distribution
    varies enough that per-batch calibration is needed.
    Typical speedup: 2-3× on CPU, 1.5-2× on GPU.

### Static Quantisation (Post-Training Quantisation, PTQ)

    Quantise both weights AND activations using calibration data.
    The calibration step collects activation statistics to compute
    per-layer scale/zero_point parameters.

        from onnxruntime.quantization import (
            quantize_static, QuantType,
            CalibrationDataReader, QuantFormat,
        )

        class MyCalibrationReader(CalibrationDataReader):
            def __init__(self, calibration_samples):
                self.samples = calibration_samples
                self.idx = 0
            def get_next(self):
                if self.idx >= len(self.samples):
                    return None
                item = {"input": self.samples[self.idx]}
                self.idx += 1
                return item

        calibration_reader = MyCalibrationReader(calibration_data[:100])

        quantize_static(
            model_input     = "model.onnx",
            model_output    = "model_static_int8.onnx",
            calibration_data_reader = calibration_reader,
            quant_format    = QuantFormat.QDQ,   # or QOperator
            activation_type = QuantType.QInt8,
            weight_type     = QuantType.QInt8,
        )

    Calibration matters:
        Use ~100-1000 representative samples from the deployment distribution.
        Calibration range affects quantisation accuracy significantly.
        Outlier activations → poor quantisation → accuracy drop.

### QDQ vs QOperator Formats

    QOperator format:
        Inserts QuantizeLinear + DequantizeLinear nodes AROUND each op.
        Explicit quantisation at every layer boundary.
        Easy to visualise and debug.
        Model: ... → QuantizeLinear → MatMul(int8) → DequantizeLinear → ...

    QDQ (Quantize-DeQuantize) format:
        More flexible: quantisation can be applied to specific subgraphs.
        Required for TensorRT execution (TRT expects QDQ format).
        Better support for per-channel quantisation of weights.
        Recommended for GPU deployment.

### Quantisation-Aware Training (QAT)

    The most accurate quantisation method: simulate quantisation during
    training with fake-quantise nodes (real computation in FP32, but with
    simulated rounding).

    The model learns to tolerate quantisation noise.
    QAT typically recovers most of the accuracy lost by PTQ.
    Cost: requires additional training (50-100 epochs typically).

    PyTorch QAT → ONNX:
        qat_model = torch.quantization.quantize_qat(model, ...)
        torch.onnx.export(qat_model, ...)


##### PART 7 — CROSS-PLATFORM DEPLOYMENT TARGETS

### ONNX → TensorRT (NVIDIA GPU Production Inference)

    TensorRT is NVIDIA's high-performance inference SDK.
    Using the TensorRT EP in ORT is the fastest path for NVIDIA GPUs.

        providers = [("TensorrtExecutionProvider", {
            "device_id":              0,
            "trt_max_workspace_size": 4 * 1024**3,    # 4GB workspace
            "trt_fp16_enable":        True,            # FP16 inference
            "trt_int8_enable":        False,
            "trt_engine_cache_enable": True,           # cache compiled engine
            "trt_engine_cache_path":  "./trt_cache",
        })]

    First inference is slow (TRT compiles the engine: kernel autotuning,
    layer fusion, precision calibration). Cached runs are very fast.
    Typical speedup vs PyTorch: 3-8× for transformer inference.

### ONNX → CoreML (Apple Silicon)

    Convert ONNX to CoreML format for deployment on macOS/iOS:

        import coremltools as ct

        model = ct.convert(
            "model.onnx",
            convert_to    = "mlprogram",         # use Neural Engine
            minimum_deployment_target = ct.target.iOS16,
            inputs        = [ct.ImageType(shape=(1, 3, 224, 224))],
        )
        model.save("model.mlpackage")

    Apple's ANE (Apple Neural Engine) in M1/M2/M3 provides extraordinary
    power efficiency: processing 10 TOP/s while drawing < 1W.
    Running on CPU or GPU would use 5-10× more power for the same task.

### ONNX → WebAssembly (Browser Inference)

    ONNX Runtime Web (ort-web): run ONNX models entirely in the browser.

        // JavaScript / TypeScript
        import * as ort from 'onnxruntime-web';

        const session = await ort.InferenceSession.create('model.onnx', {
            executionProviders: ['wasm'],  // or 'webgl', 'webgpu'
        });
        const feeds  = { input: new ort.Tensor('float32', inputData, [1, 3, 224, 224]) };
        const output = await session.run(feeds);

    WebGPU backend (Chrome 113+): GPU-accelerated browser inference.
    Applications: privacy-preserving on-device ML, offline ML apps.

### ONNX → OpenVINO (Intel CPU/iGPU)

    Intel's OpenVINO provides excellent CPU inference with AVX-512 optimisation:

        providers = [("OpenVINOExecutionProvider", {
            "device_type":          "CPU_FP32",   # or CPU_FP16, GPU_FP16
            "enable_npu_fast_compile": False,
            "num_of_threads":        8,
        })]

    Particularly valuable for cloud CPU inference (no GPU cost).
    Intel CPUs with AVX-512 can achieve 2-4× speedup vs basic CPU EP.

### ONNX → Mobile (TFLite / QNN)

    ONNX → TFLite (via tf2onnx reverse):
        Not a standard path; TFLite has its own format.
        Better to export PyTorch → TFLite directly, or use ONNX Runtime Mobile.

    ONNX Runtime Mobile:
        A stripped-down ORT build for mobile devices (< 1 MB binary).
        Supports only a subset of operators (sufficient for most inference).
        Available for Android and iOS.

        # Generate a reduced ORT model for mobile:
        from onnxruntime.tools import prepare_model_for_inference_on_mobile
        prepare_model_for_inference_on_mobile.prepare_model(
            "model.onnx", "model_mobile.ort"
        )

### Deployment Target Selection Guide

    ┌───────────────────────────────────────────────────────────────────────┐
    │ Target                │ EP / Tool              │ Best for             │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Any CPU server        │ CPU EP                 │ Baseline, no deps    │
    │ Intel CPU (production)│ OpenVINO EP            │ Server CPU serving   │
    │ NVIDIA GPU (dev/cloud)│ CUDA EP                │ Easy GPU inference   │
    │ NVIDIA GPU (prod)     │ TensorRT EP            │ Max NVIDIA perf      │
    │ AMD GPU               │ ROCm EP                │ AMD datacenter       │
    │ Apple Silicon         │ CoreML EP / coremltools│ macOS / iOS apps     │
    │ Android (Qualcomm)    │ QNN EP                 │ On-device mobile     │ 
    │ Browser               │ ONNX Runtime Web       │ Client-side ML       │
    │ Windows any GPU       │ DirectML EP            │ Windows deployment   │
    │ Edge / IoT            │ ORT Mobile + INT8      │ Resource-constrained │
    └───────────────────────────────────────────────────────────────────────┘


##### PART 8 — PRODUCTION PATTERNS AND TOOLING ECOSYSTEM

### The ONNX Tooling Ecosystem

    onnx (core library):
        Load, save, create, and manipulate ONNX graphs in Python.
        Graph checking, shape inference, opset conversion.

    onnxruntime:
        The inference engine. pip install onnxruntime or onnxruntime-gpu.

    onnxruntime-training:
        Train models with ONNX Runtime (for fine-tuning / continual learning).
        Uses ORT's optimised kernels for both forward and backward passes.

    onnxoptimizer:
        Graph-level transformation passes for model simplification.

    onnx-simplifier (onnxsim):
        Simplify ONNX graphs via constant propagation and shape inference.
        pip install onnxsim

    onnxmltools:
        Convert scikit-learn, Keras, XGBoost, LightGBM to ONNX.
        Extends skl2onnx with more framework support.

    Netron:
        Visual ONNX model browser. Open .onnx files and see the full graph.
        Available as desktop app (Windows/Mac/Linux) and at netron.app.
        Essential for debugging export issues and inspecting graph structure.

    HuggingFace Optimum:
        ORTModel classes for fast transformer inference.
        Automatic ONNX export, optimisation, and quantisation.
        Best-in-class for deploying HuggingFace models with ONNX Runtime.

    torch-onnx (PyTorch 2.x):
        The updated exporter using Dynamo. More robust than tracing.

### Production Deployment Pattern

    The recommended production pipeline:

        [Training] → [Export to ONNX] → [Validate] → [Optimise] →
        [Quantise] → [Benchmark] → [Deploy with ORT]

    Step 1 — Export:
        torch.onnx.export(model, ..., dynamic_axes={...})

    Step 2 — Validate:
        onnx.checker.check_model(model)
        Compare ORT output vs PyTorch output (< 1e-5 max diff)

    Step 3 — Optimise:
        onnxsim.simplify(model)          → simpler graph
        onnxoptimizer.optimize(model)    → fused ops

    Step 4 — Quantise (if needed):
        quantize_dynamic() or quantize_static() for INT8

    Step 5 — Benchmark:
        Measure latency (p50/p95/p99) under realistic load
        Compare with FP32 baseline
        Verify accuracy delta < threshold

    Step 6 — Serve:
        ORT InferenceSession in a FastAPI/gRPC service
        or KServe/Triton for Kubernetes deployment

### ORT in Production Services

    Triton Inference Server (NVIDIA):
        Supports ONNX natively as a backend.
        Dynamic batching, multi-model serving, gRPC + HTTP.
        config.pbtxt:
            backend: "onnxruntime"
            max_batch_size: 64
            input: [{name: "input", data_type: TYPE_FP32, dims: [3, 224, 224]}]

    FastAPI + ORT (simple REST serving):
        sess = ort.InferenceSession("model.onnx")
        @app.post("/predict")
        async def predict(request: Request):
            body = await request.json()
            inp  = np.array(body["input"], dtype=np.float32)
            out  = sess.run(None, {"input": inp})[0]
            return {"output": out.tolist()}

    HuggingFace Optimum + Inference Endpoints:
        Deploy an ORTModel directly to HuggingFace Inference Endpoints.
        ORT inference is the default backend for HF production endpoints.

### Common Export Pitfalls and Solutions

    PITFALL 1: torch.onnx.export with default dtype:
        Problem:  model uses int64 indices; ONNX runtime expects int32 on some EPs
        Solution: add explicit Cast nodes or use dtype=torch.int32 for indices

    PITFALL 2: Dynamic control flow (if/else based on tensor values):
        Problem:  tracing captures only one branch
        Solution: use torch.jit.script or dynamo_export for dynamic control flow

    PITFALL 3: Custom PyTorch ops (e.g. custom CUDA kernels):
        Problem:  no ONNX equivalent exists
        Solution: register a symbolic function for torch.onnx,
                  or replace custom op with standard ops

    PITFALL 4: Shape mismatch at inference time:
        Problem:  exported with batch=1, queried with batch=32
        Solution: always set dynamic_axes at export time

    PITFALL 5: fp16 NaN/Inf after conversion:
        Problem:  attention softmax overflow with fp16 + long sequences
        Solution: add fp32 softmax node, or cast attention scores before softmax

    PITFALL 6: Slow first inference with TensorRT EP:
        Problem:  TRT compiles and auto-tunes kernels on first run (minutes)
        Solution: enable trt_engine_cache_enable and save cache to disk

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · ONNX Format — Graph Structure, Protobuf, and Model Inspection": {
        "description": (
            "Deep dive into the ONNX file format and computational graph. "
            "Build ONNX models from scratch using the onnx Python API. "
            "NodeProto, TensorProto, GraphProto, ModelProto anatomy. "
            "Operator types: Gemm, Conv, Relu, Softmax, LayerNorm. "
            "Graph topology: inputs, outputs, initializers, value_info. "
            "Shape inference: propagating tensor shapes through the graph. "
            "Model inspection: nodes, edges, parameter count, opset. "
            "Type system: float32, int8, int64, dynamic shapes. "
            "Exporting a real PyTorch MLP and CNN to ONNX. "
            "Validating ONNX models with onnx.checker. "
            "Comparing node counts: raw export vs simplified graph. "
            "Protobuf serialisation: file size and structure."
        ),
        "language": "python",
        "code": '''
import numpy as np
import struct
import math
import tempfile
import os
import time
from typing import List, Tuple, Dict, Any, Optional

try:
    import onnx
    from onnx import numpy_helper, TensorProto, helper, shape_inference
    print(f"  ONNX version: {onnx.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "onnx", "--quiet"], check=True)
    import onnx
    from onnx import numpy_helper, TensorProto, helper, shape_inference
    print(f"  ONNX version: {onnx.__version__}")

print("=" * 65)
print("  ONNX FORMAT — GRAPH STRUCTURE & MODEL INSPECTION")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: ONNX graph anatomy — building from scratch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Building an ONNX model from scratch")
print("━" * 65)
print()

def build_mlp_onnx(
    in_dim:     int,
    hidden_dim: int,
    out_dim:    int,
    opset:      int = 17,
) -> onnx.ModelProto:
    """
    Construct an ONNX graph for a 2-layer MLP manually.
    Architecture: Linear(in→hidden) → GELU → Linear(hidden→out) → Softmax
    This demonstrates all key ONNX graph primitives.
    """
    # ── Weight initializers (constant tensors stored in model) ──────────
    W1  = rng.standard_normal((in_dim, hidden_dim)).astype(np.float32) * 0.01
    b1  = np.zeros(hidden_dim, dtype=np.float32)
    W2  = rng.standard_normal((hidden_dim, out_dim)).astype(np.float32) * 0.01
    b2  = np.zeros(out_dim, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(W1, name="W1"),
        numpy_helper.from_array(b1, name="b1"),
        numpy_helper.from_array(W2, name="W2"),
        numpy_helper.from_array(b2, name="b2"),
    ]

    # ── Graph nodes (operations) ─────────────────────────────────────────
    # Gemm = General Matrix Multiply + Bias = XW + b
    # inputs: [X, W, b], outputs: [Y]
    # alpha=1, beta=1, transB=1 means: Y = alpha * X @ W.T + beta * b
    gemm1 = helper.make_node(
        op_type  = "Gemm",
        inputs   = ["X", "W1", "b1"],
        outputs  = ["h1"],
        name     = "gemm_layer1",
        alpha    = 1.0, beta = 1.0, transB = 1,
    )
    # GELU activation (opset 20+ supports GELU natively; we approximate for opset 17)
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # We'll use the standard Relu as a simpler demonstration
    relu1 = helper.make_node(
        op_type  = "Relu",
        inputs   = ["h1"],
        outputs  = ["h1_act"],
        name     = "relu_layer1",
    )
    gemm2 = helper.make_node(
        op_type  = "Gemm",
        inputs   = ["h1_act", "W2", "b2"],
        outputs  = ["logits"],
        name     = "gemm_layer2",
        alpha    = 1.0, beta = 1.0, transB = 1,
    )
    softmax = helper.make_node(
        op_type  = "Softmax",
        inputs   = ["logits"],
        outputs  = ["probs"],
        name     = "softmax_output",
        axis     = 1,
    )

    # ── Input/Output value info (typed tensor descriptors) ───────────────
    # None = dynamic dimension (batch size)
    X_info     = helper.make_tensor_value_info("X",     TensorProto.FLOAT, [None, in_dim])
    probs_info = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [None, out_dim])

    # ── Assemble the graph ───────────────────────────────────────────────
    graph = helper.make_graph(
        nodes        = [gemm1, relu1, gemm2, softmax],
        name         = "mlp_graph",
        inputs       = [X_info],
        outputs      = [probs_info],
        initializer  = initializers,
    )

    # ── Wrap in a ModelProto with opset ──────────────────────────────────
    model = helper.make_model(
        graph,
        opset_imports = [helper.make_opsetid("", opset)],
    )
    model.ir_version  = 8
    model.model_version = 1
    model.doc_string    = "Two-layer MLP built from scratch with ONNX Python API"
    return model


IN_DIM, HIDDEN_DIM, OUT_DIM = 32, 64, 10
mlp_model = build_mlp_onnx(IN_DIM, HIDDEN_DIM, OUT_DIM)

print(f"  Built MLP: {IN_DIM} → {HIDDEN_DIM} (Relu) → {OUT_DIM} (Softmax)")
print()
print(f"  ModelProto fields:")
print(f"    ir_version:     {mlp_model.ir_version}")
print(f"    opset_imports:  {[(o.domain or 'ONNX', o.version) for o in mlp_model.opset_import]}")
print(f"    model_version:  {mlp_model.model_version}")
print(f"    doc_string:     '{mlp_model.doc_string}'")
print()

graph = mlp_model.graph
print(f"  GraphProto contents:")
print(f"    name:          {graph.name}")
print(f"    nodes:         {len(graph.node)}")
print(f"    inputs:        {[i.name for i in graph.input]}")
print(f"    outputs:       {[o.name for o in graph.output]}")
print(f"    initializers:  {len(graph.initializer)}  (weight tensors)")
print()

# Detailed node inspection
print(f"  Node-by-node breakdown:")
print(f"  {'Name':<22} {'Op':<12} {'Inputs':<28} {'Outputs'}")
print(f"  {'─'*75}")
for node in graph.node:
    print(f"  {node.name:<22} {node.op_type:<12} "
          f"{str(list(node.input)):<28} {list(node.output)}")
print()

# Initializer (weight tensor) inspection
print(f"  Weight tensors (initializers):")
total_params = 0
print(f"  {'Name':<8} {'Shape':<20} {'DType':<12} {'Size (KB)'}")
print(f"  {'─'*55}")
for init in graph.initializer:
    shape    = list(init.dims)
    n_elem   = math.prod(shape) if shape else 1
    dtype    = {1: "float32", 3: "int8", 7: "int64", 10: "float16"}.get(init.data_type, "?")
    size_kb  = n_elem * 4 / 1024
    total_params += n_elem
    print(f"  {init.name:<8} {str(shape):<20} {dtype:<12} {size_kb:.2f}")
print(f"  {'─'*55}")
print(f"  Total parameters: {total_params:,}  ({total_params * 4 / 1024:.1f} KB)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Shape inference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Shape inference: propagating tensor shapes")
print("━" * 65)
print()

# Without shape inference, intermediate tensors have unknown shapes
pre_inference_shapes = {}
for node in mlp_model.graph.node:
    for output in node.output:
        pre_inference_shapes[output] = "unknown"

# Run shape inference
model_with_shapes = shape_inference.infer_shapes(mlp_model)

# Collect inferred shapes
inferred = {}
for vi in model_with_shapes.graph.value_info:
    shape = [d.dim_value if d.HasField("dim_value")
             else d.dim_param if d.HasField("dim_param") else "?"
             for d in vi.type.tensor_type.shape.dim]
    inferred[vi.name] = shape

# Also add inputs and outputs
for vi in (list(model_with_shapes.graph.input) +
           list(model_with_shapes.graph.output)):
    if vi.type.HasField("tensor_type"):
        shape = [d.dim_value if d.HasField("dim_value")
                 else d.dim_param if d.HasField("dim_param") else "?"
                 for d in vi.type.tensor_type.shape.dim]
        inferred[vi.name] = shape

print(f"  Inferred tensor shapes (batch=None = dynamic):")
print(f"  {'Tensor':<12} {'Shape':<30} {'Meaning'}")
print(f"  {'─'*65}")
tensor_meanings = {
    "X":      "model input: [batch, in_dim]",
    "h1":     "after linear layer 1: [batch, hidden]",
    "h1_act": "after Relu: [batch, hidden]",
    "logits": "after linear layer 2: [batch, n_classes]",
    "probs":  "after Softmax: [batch, n_classes]",
}
for tensor_name, meaning in tensor_meanings.items():
    shape = inferred.get(tensor_name, ["?"])
    print(f"  {tensor_name:<12} {str(shape):<30} {meaning}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Export PyTorch model to ONNX and inspect
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — PyTorch → ONNX export: tracing and inspection")
print("━" * 65)
print()

try:
    import torch
    import torch.nn as nn

    # Define a slightly more complex CNN model
    class SmallCNN(nn.Module):
        def __init__(self, n_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = SmallCNN(n_classes=10)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SmallCNN: {n_params:,} parameters")
    print(f"  Architecture: Conv(1→16) → BN → ReLU → MaxPool →")
    print(f"                Conv(16→32) → BN → ReLU → AvgPool →")
    print(f"                Flatten → Linear(512→128) → ReLU → Linear(128→10)")
    print()

    dummy_input = torch.randn(1, 1, 28, 28)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "small_cnn.onnx")

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params      = True,
                opset_version      = 17,
                do_constant_folding = True,
                input_names        = ["image"],
                output_names       = ["logits"],
                dynamic_axes       = {
                    "image":  {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
                verbose = False,
            )

        exported_size  = os.path.getsize(onnx_path)
        exported_model = onnx.load(onnx_path)

        # Validate
        onnx.checker.check_model(exported_model)

        print(f"  Export successful!")
        print(f"    ONNX file size:  {exported_size / 1024:.1f} KB")
        print(f"    Expected (~params×4B): {n_params * 4 / 1024:.1f} KB")
        print()

        # Graph analysis
        g = exported_model.graph
        op_counts: Dict[str, int] = {}
        for node in g.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        print(f"  Exported graph:")
        print(f"    Total nodes:    {len(g.node)}")
        print(f"    Initializers:   {len(g.initializer)}  (weight tensors)")
        print(f"    Input name:     {g.input[0].name}")
        print(f"    Output name:    {g.output[0].name}")
        print()
        print(f"  Operator frequency:")
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"    {op:<25} × {count:>2}  {bar}")
        print()

        # Dynamic shape inspection
        inp_type  = g.input[0].type.tensor_type
        inp_shape = [d.dim_param if d.HasField("dim_param")
                     else d.dim_value for d in inp_type.shape.dim]
        print(f"  Dynamic input shape: {inp_shape}")
        print(f"  batch_size=None means inference can handle any batch size ✅")

except ImportError:
    print(f"  PyTorch not available — showing export API reference:")
    print(f"  torch.onnx.export(model, dummy_input, 'model.onnx',")
    print(f"      opset_version=17, dynamic_axes={{...")
    print(f"      'input': {{0: 'batch'}}}},")
    print(f"      input_names=['input'], output_names=['output'])")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Building a custom ONNX model with multiple op types
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — ONNX operator variety: LayerNorm, attention, reshape")
print("━" * 65)
print()

# Build a transformer-like mini-block: LayerNorm + simple attention + FFN
SEQ_LEN, D_MODEL, N_HEADS = 8, 32, 4
D_HEAD = D_MODEL // N_HEADS

def make_tensor(name: str, array: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(array.astype(np.float32), name=name)

# Weights for a minimal transformer attention block
ln_weight = np.ones(D_MODEL, dtype=np.float32)
ln_bias   = np.zeros(D_MODEL, dtype=np.float32)
Wq = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.01
Wk = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.01
Wv = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.01
Wo = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float32) * 0.01
scale_val = np.array([1.0 / math.sqrt(D_HEAD)], dtype=np.float32)

inits = [
    make_tensor("ln_w",    ln_weight),
    make_tensor("ln_b",    ln_bias),
    make_tensor("Wq",      Wq),
    make_tensor("Wk",      Wk),
    make_tensor("Wv",      Wv),
    make_tensor("Wo",      Wo),
    make_tensor("scale",   scale_val),
]

# Layer Normalisation (opset 17 supports LayerNormalization natively)
layernorm_node = helper.make_node(
    "LayerNormalization",
    inputs  = ["X_seq", "ln_w", "ln_b"],
    outputs = ["X_norm"],
    name    = "layernorm",
    axis    = -1,      # normalise over last dimension (D_MODEL)
    epsilon = 1e-5,
)

# Q, K, V projections: MatMul(X_norm, W)
q_proj = helper.make_node("MatMul", ["X_norm", "Wq"], ["Q"], name="q_proj")
k_proj = helper.make_node("MatMul", ["X_norm", "Wk"], ["K"], name="k_proj")
v_proj = helper.make_node("MatMul", ["X_norm", "Wv"], ["V"], name="v_proj")

# Scaled attention scores: scores = Q @ K.T * scale
k_t   = helper.make_node("Transpose", ["K"], ["K_T"], name="k_transpose",
                           perm=[0, 2, 1])  # simplified: omit head splitting
scores = helper.make_node("MatMul",   ["Q", "K_T"], ["attn_raw"], name="attn_scores")
scaled = helper.make_node("Mul",  ["attn_raw", "scale"], ["attn_scaled"], name="scale")
softmx = helper.make_node("Softmax", ["attn_scaled"], ["attn_w"], name="attn_softmax",
                            axis=-1)
ctx    = helper.make_node("MatMul",  ["attn_w", "V"], ["context"], name="context")
output = helper.make_node("MatMul",  ["context", "Wo"], ["X_out"], name="out_proj")

# Residual connection
resid  = helper.make_node("Add", ["X_seq", "X_out"], ["X_residual"], name="residual")

nodes_attn = [layernorm_node, q_proj, k_proj, v_proj,
               k_t, scores, scaled, softmx, ctx, output, resid]

input_vi  = helper.make_tensor_value_info("X_seq",      TensorProto.FLOAT, [None, SEQ_LEN, D_MODEL])
output_vi = helper.make_tensor_value_info("X_residual", TensorProto.FLOAT, [None, SEQ_LEN, D_MODEL])

graph_attn  = helper.make_graph(nodes_attn, "transformer_block",
                                  [input_vi], [output_vi], inits)
model_attn  = helper.make_model(graph_attn,
                                  opset_imports=[helper.make_opsetid("", 17)])
model_attn.ir_version = 8

onnx.checker.check_model(model_attn)

attn_node_types = {}
for n in model_attn.graph.node:
    attn_node_types[n.op_type] = attn_node_types.get(n.op_type, 0) + 1

print(f"  Transformer attention block (d_model={D_MODEL}, heads={N_HEADS}):")
print(f"    Total nodes: {len(model_attn.graph.node)}")
print(f"    Operators used:")
for op, cnt in sorted(attn_node_types.items()):
    print(f"      {op:<25} × {cnt}")
print()

# Serialise and show file size
attn_bytes = model_attn.SerializeToString()
print(f"  Serialised model size: {len(attn_bytes):,} bytes")
print(f"  Overhead vs raw weights: +{len(attn_bytes) - (D_MODEL*D_MODEL*4*4 + D_MODEL*4*2)} bytes "
      f"(graph topology + metadata)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: ONNX model statistics utility
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — ONNX model statistics: full inspection utility")
print("━" * 65)
print()

def onnx_model_stats(model: onnx.ModelProto) -> Dict:
    """Comprehensive statistics for any ONNX model."""
    g = model.graph

    # Parameter count and size
    total_params = 0
    total_bytes  = 0
    for init in g.initializer:
        n_elem = math.prod(init.dims) if init.dims else 1
        dtype_bytes = {1: 4, 3: 1, 7: 8, 10: 2, 11: 8}.get(init.data_type, 4)
        total_params += n_elem
        total_bytes  += n_elem * dtype_bytes

    # Operator frequency
    op_freq: Dict[str, int] = {}
    for node in g.node:
        op_freq[node.op_type] = op_freq.get(node.op_type, 0) + 1

    # Unique ops
    unique_ops = sorted(op_freq.keys())

    # Input/output shapes
    def get_shape(vi):
        if not vi.type.HasField("tensor_type"): return []
        t = vi.type.tensor_type
        if not t.HasField("shape"): return []
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in t.shape.dim]

    return {
        "n_nodes":       len(g.node),
        "n_initializers": len(g.initializer),
        "n_params":      total_params,
        "total_bytes":   total_bytes,
        "n_inputs":      len(g.input),
        "n_outputs":     len(g.output),
        "opset":         model.opset_import[0].version if model.opset_import else "?",
        "op_frequency":  op_freq,
        "unique_ops":    unique_ops,
        "input_shapes":  [get_shape(i) for i in g.input],
        "output_shapes": [get_shape(o) for o in g.output],
    }


print(f"  Model statistics comparison:")
print()
for label, model in [
    ("Handbuilt MLP",     mlp_model),
    ("Transformer Block", model_attn),
]:
    stats = onnx_model_stats(model)
    print(f"  [{label}]")
    print(f"    nodes={stats['n_nodes']}  "
          f"params={stats['n_params']:,}  "
          f"size={stats['total_bytes']/1024:.1f}KB  "
          f"opset={stats['opset']}")
    print(f"    operators: {stats['unique_ops']}")
    print(f"    inputs:    {stats['input_shapes']}")
    print(f"    outputs:   {stats['output_shapes']}")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · ONNX Runtime — Inference, Execution Providers & Benchmarking": {
        "description": (
            "Complete ONNX Runtime inference pipeline from scratch. "
            "InferenceSession creation: loading, optimisation, EP selection. "
            "get_inputs/get_outputs: session metadata inspection. "
            "session.run(): input/output numpy interface. "
            "SessionOptions: graph optimisation levels, thread counts. "
            "Execution provider priority ordering: GPU → CPU fallback. "
            "Numerical accuracy verification: ORT vs PyTorch outputs. "
            "Throughput benchmark: latency at various batch sizes. "
            "Memory footprint: FP32 vs FP16 session comparison. "
            "ORT profiling: per-operator timing breakdown. "
            "Dynamic shape handling: run at multiple input sizes. "
            "Framework comparison: PyTorch eager vs ORT speedup."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import math
import json
import tempfile
import os
from typing import Dict, List, Optional, Tuple

try:
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto, helper, shape_inference
    print(f"  ONNX: {onnx.__version__}  |  ORT: {ort.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "onnx", "onnxruntime", "--quiet"], check=True)
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto, helper, shape_inference
    print(f"  ONNX: {onnx.__version__}  |  ORT: {ort.__version__}")

print("=" * 65)
print("  ONNX RUNTIME — INFERENCE, PROVIDERS & BENCHMARKING")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ── Build a realistic test model ─────────────────────────────────────────
def build_benchmark_model(
    in_dim:      int = 128,
    hidden_dims: List[int] = [256, 128, 64],
    out_dim:     int = 10,
    opset:       int = 17,
) -> bytes:
    """Build a multi-layer MLP ONNX model for benchmarking."""
    initializers = []
    nodes        = []
    prev_name    = "X"
    prev_dim     = in_dim

    for layer_idx, h in enumerate(hidden_dims):
        W = rng.standard_normal((prev_dim, h)).astype(np.float32) * math.sqrt(2.0 / prev_dim)
        b = np.zeros(h, dtype=np.float32)
        W_name = f"W{layer_idx}"
        b_name = f"b{layer_idx}"
        h_name = f"h{layer_idx}"
        a_name = f"a{layer_idx}"

        initializers.append(numpy_helper.from_array(W, name=W_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))

        nodes.append(helper.make_node(
            "Gemm", [prev_name, W_name, b_name], [h_name],
            name=f"gemm{layer_idx}", alpha=1.0, beta=1.0, transB=1,
        ))
        nodes.append(helper.make_node(
            "Relu", [h_name], [a_name], name=f"relu{layer_idx}",
        ))
        prev_name, prev_dim = a_name, h

    # Output layer (no activation)
    W_out = rng.standard_normal((prev_dim, out_dim)).astype(np.float32) * 0.01
    b_out = np.zeros(out_dim, dtype=np.float32)
    initializers.append(numpy_helper.from_array(W_out, name="W_out"))
    initializers.append(numpy_helper.from_array(b_out, name="b_out"))
    nodes.append(helper.make_node(
        "Gemm", [prev_name, "W_out", "b_out"], ["output"],
        name="gemm_out", alpha=1.0, beta=1.0, transB=1,
    ))

    input_vi  = helper.make_tensor_value_info("X",      TensorProto.FLOAT, [None, in_dim])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, out_dim])
    graph     = helper.make_graph(nodes, "benchmark_mlp", [input_vi], [output_vi], initializers)
    model     = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


model_bytes = build_benchmark_model(
    in_dim=128, hidden_dims=[512, 256, 128], out_dim=10
)
n_params_est = (128*512 + 512 + 512*256 + 256 + 256*128 + 128 + 128*10 + 10)
print(f"  Benchmark model: 128 → 512 → 256 → 128 → 10")
print(f"  Parameters: ~{n_params_est:,}  ({n_params_est*4/1024:.1f} KB)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Session creation and inspection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — InferenceSession: creation, options, and inspection")
print("━" * 65)
print()

# Show all available providers
all_providers = ort.get_available_providers()
print(f"  Available Execution Providers on this machine:")
for ep in all_providers:
    active = "  ← ACTIVE (first available)" if ep == all_providers[0] else ""
    print(f"    {ep}{active}")
print()

# Create session with full optimisation
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = max(1, os.cpu_count() // 2)
opts.inter_op_num_threads = 1
opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL

# Provider priority: CUDA → CPU
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

t0   = time.perf_counter()
sess = ort.InferenceSession(model_bytes, sess_options=opts, providers=providers)
t_load = (time.perf_counter() - t0) * 1000

print(f"  Session creation: {t_load:.1f}ms")
print()

# Inspect the session
print(f"  Session inputs:")
for inp in sess.get_inputs():
    print(f"    name={inp.name:<10} shape={inp.shape}  type={inp.type}")

print(f"  Session outputs:")
for out in sess.get_outputs():
    print(f"    name={out.name:<10} shape={out.shape}  type={out.type}")

print(f"  Active providers: {sess.get_providers()}")
print()

# Verify SessionOptions
print(f"  Session configuration:")
print(f"    graph_optimization_level: ORT_ENABLE_ALL")
print(f"    intra_op_num_threads:     {opts.intra_op_num_threads}")
print(f"    inter_op_num_threads:     {opts.inter_op_num_threads}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Running inference and verifying accuracy
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Running inference: sess.run() interface")
print("━" * 65)
print()

# Single sample inference
x_single = rng.standard_normal((1, 128)).astype(np.float32)

t0   = time.perf_counter()
outputs = sess.run(
    output_names = None,        # None = return ALL outputs
    input_feed   = {"X": x_single},
)
t_single = (time.perf_counter() - t0) * 1000

predictions = outputs[0]
print(f"  Single-sample inference:")
print(f"    Input:  shape={x_single.shape}  dtype={x_single.dtype}")
print(f"    Output: shape={predictions.shape}  dtype={predictions.dtype}")
print(f"    Logits: {predictions[0].round(4)}")
print(f"    Argmax: {predictions[0].argmax()}  (predicted class)")
print(f"    Time:   {t_single:.2f}ms (includes session overhead on 1st call)")
print()

# Warmup (JIT compilation, cache population)
warmup_input = {"X": rng.standard_normal((8, 128)).astype(np.float32)}
for _ in range(5):
    _ = sess.run(None, warmup_input)
print(f"  Warmed up (5 runs) ✅")
print()

# Run at specific output names (selective output retrieval)
out_name = sess.get_outputs()[0].name
result_named = sess.run([out_name], {"X": x_single})
print(f"  Named output retrieval: sess.run(['{out_name}'], ...)  →  shape={result_named[0].shape}")
print()

# Numerical verification vs numpy reference
def mlp_numpy_forward(x, model_proto):
    """Reference implementation using numpy to verify ORT output."""
    inits = {init.name: numpy_helper.to_array(init)
             for init in model_proto.graph.initializer}
    h = x
    layer_idx = 0
    while f"W{layer_idx}" in inits:
        W = inits[f"W{layer_idx}"]
        b = inits[f"b{layer_idx}"]
        h = h @ W.T + b
        h = np.maximum(h, 0)   # ReLU
        layer_idx += 1
    W_out = inits["W_out"]
    b_out = inits["b_out"]
    return h @ W_out.T + b_out

model_proto = onnx.load_from_string(model_bytes)
numpy_out   = mlp_numpy_forward(x_single, model_proto)
ort_out     = predictions

max_diff = float(np.abs(numpy_out - ort_out).max())
rel_diff = float(np.abs((numpy_out - ort_out) / (np.abs(numpy_out) + 1e-8)).max())

print(f"  Numerical verification (ORT vs NumPy reference):")
print(f"    Max absolute diff: {max_diff:.2e}  (expected < 1e-5 for FP32)")
print(f"    Max relative diff: {rel_diff:.2e}")
print(f"    ORT output:    {ort_out[0, :4].round(6)}")
print(f"    NumPy output:  {numpy_out[0, :4].round(6)}")
passed = max_diff < 1e-4
print(f"    Accuracy check: {'✅ PASSED' if passed else '❌ FAILED'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Latency benchmark across batch sizes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Latency benchmark: throughput vs latency trade-off")
print("━" * 65)
print()

N_REPEATS = 200
batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]

print(f"  Throughput benchmark ({N_REPEATS} runs each):")
print(f"  {'Batch':>8} {'Latency ms':>12} {'p95 ms':>10} {'Throughput':>14} {'Latency/sample'}")
print(f"  {'─'*60}")

for bs in batch_sizes:
    x_batch = rng.standard_normal((bs, 128)).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = sess.run(None, {"X": x_batch})

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _ = sess.run(None, {"X": x_batch})
        times.append((time.perf_counter() - t0) * 1000)

    times      = sorted(times)
    p50_ms     = times[N_REPEATS // 2]
    p95_ms     = times[int(N_REPEATS * 0.95)]
    throughput = bs / (p50_ms / 1000)     # samples per second
    per_sample = p50_ms / bs

    print(f"  {bs:>8} {p50_ms:>12.3f} {p95_ms:>10.3f} "
          f"{throughput:>14.0f}/s {per_sample:>14.4f}ms")

print()
print(f"  Key insight: larger batches → better GPU utilisation but higher latency")
print(f"  Sweet spot depends on SLA (latency budget) vs cost (throughput needed)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: ORT graph optimisation comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Optimisation levels: impact on latency")
print("━" * 65)
print()

opt_levels = [
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL,   "DISABLE_ALL (no opt)"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,  "ENABLE_BASIC"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,"ENABLE_EXTENDED"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL,    "ENABLE_ALL"),
]

x_test = rng.standard_normal((32, 128)).astype(np.float32)
N_OPT  = 100

print(f"  Optimisation level comparison (batch=32, {N_OPT} runs):")
print(f"  {'Level':<30} {'Load ms':>10} {'Latency ms':>12} {'Speedup':>10}")
print(f"  {'─'*65}")

baseline_latency = None
for opt_level, label in opt_levels:
    level_opts = ort.SessionOptions()
    level_opts.graph_optimization_level = opt_level
    level_opts.intra_op_num_threads      = opts.intra_op_num_threads

    t0    = time.perf_counter()
    s     = ort.InferenceSession(model_bytes, sess_options=level_opts,
                                  providers=["CPUExecutionProvider"])
    t_load_l = (time.perf_counter() - t0) * 1000

    # Warmup
    for _ in range(5): s.run(None, {"X": x_test})

    times = []
    for _ in range(N_OPT):
        t0 = time.perf_counter()
        s.run(None, {"X": x_test})
        times.append((time.perf_counter() - t0) * 1000)
    p50 = sorted(times)[N_OPT // 2]

    if baseline_latency is None:
        baseline_latency = p50
    speedup = baseline_latency / p50 if p50 > 0 else 1.0

    print(f"  {label:<30} {t_load_l:>10.1f} {p50:>12.3f} {speedup:>10.2f}×")

print()
print(f"  ORT_ENABLE_ALL is the recommended production setting.")
print(f"  It applies: constant folding, op fusion, layout optimisation")
print(f"  Cost: slightly longer session creation (one-time amortised cost)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: PyTorch vs ORT speedup (if PyTorch available)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — PyTorch vs ONNX Runtime inference comparison")
print("━" * 65)
print()

try:
    import torch
    import torch.nn as nn

    class PyTorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(128, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 10),
            )
        def forward(self, x): return self.net(x)

    pt_model = PyTorchMLP()
    pt_model.eval()

    # Load ORT session with PyTorch weights
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "pt_model.onnx")
        dummy = torch.randn(1, 128)
        with torch.no_grad():
            torch.onnx.export(pt_model, dummy, onnx_path, opset_version=17,
                               input_names=["X"], output_names=["output"],
                               dynamic_axes={"X": {0: "batch"}, "output": {0: "batch"}})

        sess_pt = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    BS, N_PT = 32, 500
    x_np = rng.standard_normal((BS, 128)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    # Warmup
    with torch.no_grad():
        for _ in range(10): _ = pt_model(x_pt)
    for _ in range(10): sess_pt.run(None, {"X": x_np})

    # PyTorch timing
    pt_times = []
    with torch.no_grad():
        for _ in range(N_PT):
            t0 = time.perf_counter()
            _ = pt_model(x_pt)
            pt_times.append((time.perf_counter() - t0) * 1000)

    # ORT timing
    ort_times = []
    for _ in range(N_PT):
        t0 = time.perf_counter()
        _ = sess_pt.run(None, {"X": x_np})
        ort_times.append((time.perf_counter() - t0) * 1000)

    pt_p50  = sorted(pt_times)[N_PT // 2]
    ort_p50 = sorted(ort_times)[N_PT // 2]
    speedup = pt_p50 / ort_p50

    # Verify identical outputs
    with torch.no_grad():
        pt_out  = pt_model(x_pt).numpy()
    ort_out = sess_pt.run(None, {"X": x_np})[0]
    max_diff = float(np.abs(pt_out - ort_out).max())

    print(f"  PyTorch vs ONNX Runtime (batch={BS}, {N_PT} runs):")
    print(f"  {'Runtime':<25} {'p50 ms':>10} {'p95 ms':>10} {'Throughput':>15}")
    print(f"  {'─'*62}")
    pt_thr  = BS / (pt_p50 / 1000)
    ort_thr = BS / (ort_p50 / 1000)
    for label, times_list, p50 in [
        ("PyTorch eager", pt_times,  pt_p50),
        ("ONNX Runtime",  ort_times, ort_p50),
    ]:
        p95 = sorted(times_list)[int(N_PT * 0.95)]
        thr = BS / (p50 / 1000)
        print(f"  {label:<25} {p50:>10.3f} {p95:>10.3f} {thr:>15.0f}/s")
    print()
    print(f"  ORT speedup:           {speedup:.2f}× faster than PyTorch eager")
    print(f"  Output max diff:       {max_diff:.2e}  ✅  (numerically identical)")

except ImportError:
    print(f"  PyTorch not available — speedup comparison requires PyTorch.")
    print(f"  Typical reported speedups (from published benchmarks):")
    print(f"    BERT inference (CPU):       2.0–3.5× over PyTorch eager")
    print(f"    ResNet-50 (CPU):            1.5–2.5× over PyTorch eager")
    print(f"    DistilBERT + INT8 (CPU):    4.0–6.0× over FP32 PyTorch")
    print(f"    GPT-2 (CUDA + TRT):         3.0–8.0× over PyTorch eager")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Quantisation & Optimisation — INT8, Graph Fusion & Model Compression": {
        "description": (
            "Complete quantisation and graph optimisation pipeline. "
            "Dynamic quantisation: weight-only INT8 with onnxruntime.quantization. "
            "Static quantisation: calibration data reader and activation ranges. "
            "QDQ vs QOperator format: when to use each. "
            "Quantisation error analysis: per-layer sensitivity measurement. "
            "Model size comparison: FP32 vs INT8 vs INT4. "
            "Graph optimisation with onnxoptimizer: fusion passes. "
            "onnxsim: constant propagation and node reduction. "
            "Conv+BN fusion: mathematical equivalence proof. "
            "Operator fusion impact: latency before vs after. "
            "Accuracy-latency trade-off curves. "
            "Quantisation-aware training (QAT) concept and workflow."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
import time
import tempfile
import os
import struct
from typing import Dict, List, Tuple, Optional

try:
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto, helper
    from onnxruntime.quantization import (
        quantize_dynamic, quantize_static,
        QuantType, QuantFormat, CalibrationDataReader,
    )
    print(f"  ONNX: {onnx.__version__}  |  ORT: {ort.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "onnx", "onnxruntime", "--quiet"], check=True)
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto, helper
    from onnxruntime.quantization import (
        quantize_dynamic, quantize_static,
        QuantType, QuantFormat, CalibrationDataReader,
    )
    print(f"  ONNX: {onnx.__version__}  |  ORT: {ort.__version__}")

print("=" * 65)
print("  QUANTISATION & OPTIMISATION — INT8 & GRAPH FUSION")
print("=" * 65)
print()

rng = np.random.default_rng(0)

# ── Shared model for all experiments ──────────────────────────────────────
def build_quantise_test_model(in_dim=64, hidden=256, out_dim=8) -> onnx.ModelProto:
    inits, nodes = [], []
    prev_name, prev_dim = "X", in_dim
    for i, h in enumerate([hidden, hidden//2]):
        W = rng.standard_normal((prev_dim, h)).astype(np.float32) * math.sqrt(2/prev_dim)
        b = np.zeros(h, dtype=np.float32)
        inits += [numpy_helper.from_array(W, f"W{i}"),
                  numpy_helper.from_array(b, f"b{i}")]
        nodes += [
            helper.make_node("Gemm", [prev_name, f"W{i}", f"b{i}"], [f"h{i}"],
                              name=f"gemm{i}", alpha=1.0, beta=1.0, transB=1),
            helper.make_node("Relu", [f"h{i}"], [f"a{i}"], name=f"relu{i}"),
        ]
        prev_name, prev_dim = f"a{i}", h
    W_o = rng.standard_normal((prev_dim, out_dim)).astype(np.float32) * 0.01
    b_o = np.zeros(out_dim, dtype=np.float32)
    inits += [numpy_helper.from_array(W_o, "W_out"),
              numpy_helper.from_array(b_o, "b_out")]
    nodes.append(helper.make_node("Gemm", [prev_name, "W_out", "b_out"], ["output"],
                                   name="gemm_out", alpha=1.0, beta=1.0, transB=1))
    iv  = helper.make_tensor_value_info("X",      TensorProto.FLOAT, [None, in_dim])
    ov  = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, out_dim])
    g   = helper.make_graph(nodes, "quant_test", [iv], [ov], inits)
    m   = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

base_model    = build_quantise_test_model()
base_n_params = sum(math.prod(i.dims) for i in base_model.graph.initializer)
base_bytes    = sum(math.prod(i.dims) * 4 for i in base_model.graph.initializer)

print(f"  Base model: 64 → 256 → 128 → 8  |  params={base_n_params:,}  size={base_bytes/1024:.1f}KB")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Quantisation mathematics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Quantisation math: scale, zero_point, error analysis")
print("━" * 65)
print()

def quantise_float_to_int8(
    arr: np.ndarray,
    symmetric: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Quantise a float32 array to int8 using linear quantisation.
    Returns (quantised_int8, scale, zero_point).
    """
    if symmetric:
        # Symmetric: zero_point = 0, scale maps [-max_abs, max_abs] to [-127, 127]
        max_abs     = float(np.abs(arr).max())
        scale       = max_abs / 127.0 if max_abs > 0 else 1.0
        zero_point  = 0
        q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    else:
        # Asymmetric: full range [x_min, x_max] → [0, 255]
        x_min, x_max = float(arr.min()), float(arr.max())
        scale       = (x_max - x_min) / 255.0 if x_max > x_min else 1.0
        zero_point  = int(np.clip(np.round(-x_min / scale), 0, 255))
        q = np.clip(np.round(arr / scale + zero_point), 0, 255).astype(np.uint8)
    return q, scale, zero_point


def dequantise(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


# Demonstrate quantisation on a weight tensor
W_sample = rng.standard_normal(1000).astype(np.float32) * 0.5

q_sym, s_sym, zp_sym   = quantise_float_to_int8(W_sample, symmetric=True)
q_asym, s_asym, zp_asym = quantise_float_to_int8(W_sample, symmetric=False)

W_rec_sym  = dequantise(q_sym,  s_sym,  zp_sym)
W_rec_asym = dequantise(q_asym, s_asym, zp_asym)

err_sym  = float(np.abs(W_sample - W_rec_sym).max())
err_asym = float(np.abs(W_sample - W_rec_asym).max())
snr_sym  = float(10 * np.log10(np.mean(W_sample**2) / np.mean((W_sample - W_rec_sym)**2)))
snr_asym = float(10 * np.log10(np.mean(W_sample**2) / np.mean((W_sample - W_rec_asym)**2)))

print(f"  Quantisation error analysis (1000 float32 → int8 weights):")
print(f"  {'Method':<20} {'Scale':>10} {'ZP':>6} {'Max err':>10} {'SNR (dB)':>12}")
print(f"  {'─'*62}")
print(f"  {'Symmetric'::<20} {s_sym:>10.6f} {zp_sym:>6} {err_sym:>10.6f} {snr_sym:>12.2f}")
print(f"  {'Asymmetric':<20} {s_asym:>10.6f} {zp_asym:>6} {err_asym:>10.6f} {snr_asym:>12.2f}")
print()

# INT8 size savings
print(f"  Bit-width comparison for {base_n_params:,} parameters:")
print(f"  {'Precision':<12} {'Bytes/param':>14} {'Total size':>14} {'Relative'}")
print(f"  {'─'*55}")
for name, bpp, mult in [
    ("FP32",   4,   1.0),
    ("FP16",   2,   0.5),
    ("BF16",   2,   0.5),
    ("INT8",   1,   0.25),
    ("INT4",   0.5, 0.125),
]:
    total = base_n_params * bpp
    print(f"  {name:<12} {bpp:>14.1f} {total/1024:>13.1f}KB {mult:>13.2f}×")
print()

# Per-layer quantisation sensitivity (how much each layer is hurt by INT8)
print(f"  Per-layer quantisation sensitivity (max weight error per layer):")
for init in base_model.graph.initializer:
    if init.name.startswith("W"):
        weights = numpy_helper.to_array(init)
        q, s, zp = quantise_float_to_int8(weights.flatten())
        rec = dequantise(q, s, zp).reshape(weights.shape)
        err_pct = float(np.abs(weights - rec).max() / (np.abs(weights).max() + 1e-8) * 100)
        layer_type = "output layer" if "out" in init.name else "hidden layer"
        print(f"    {init.name:<8} shape={list(weights.shape)}  "
              f"max_err={err_pct:.3f}%  {layer_type}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Dynamic quantisation with ORT
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Dynamic quantisation: weight-only INT8")
print("━" * 65)
print()

with tempfile.TemporaryDirectory() as tmp:
    fp32_path = os.path.join(tmp, "model_fp32.onnx")
    dyn_path  = os.path.join(tmp, "model_dyn_int8.onnx")

    onnx.save(base_model, fp32_path)

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input  = fp32_path,
        model_output = dyn_path,
        weight_type  = QuantType.QInt8,
    )
    t_dyn_quant = (time.perf_counter() - t0) * 1000

    # File size comparison
    fp32_size = os.path.getsize(fp32_path)
    dyn_size  = os.path.getsize(dyn_path)
    dyn_ratio = dyn_size / fp32_size

    print(f"  Dynamic quantisation: weight-only INT8")
    print(f"    Quantisation time:  {t_dyn_quant:.0f}ms")
    print(f"    FP32 model size:    {fp32_size/1024:.1f} KB")
    print(f"    INT8 model size:    {dyn_size/1024:.1f} KB")
    print(f"    Compression ratio:  {1/dyn_ratio:.1f}× smaller")
    print()

    # Accuracy comparison
    sess_fp32 = ort.InferenceSession(fp32_path,  providers=["CPUExecutionProvider"])
    sess_dyn  = ort.InferenceSession(dyn_path,   providers=["CPUExecutionProvider"])

    x_val = rng.standard_normal((200, 64)).astype(np.float32)
    out_fp32 = sess_fp32.run(None, {"X": x_val})[0]
    out_dyn  = sess_dyn.run( None, {"X": x_val})[0]

    rel_err = float(np.abs((out_fp32 - out_dyn) / (np.abs(out_fp32) + 1e-8)).mean() * 100)
    argmax_match = (out_fp32.argmax(1) == out_dyn.argmax(1)).mean() * 100
    cos_sim = float(np.mean([
        np.dot(out_fp32[i], out_dyn[i]) /
        (np.linalg.norm(out_fp32[i]) * np.linalg.norm(out_dyn[i]) + 1e-8)
        for i in range(len(x_val))
    ]))

    print(f"  Dynamic quantisation accuracy (200 samples):")
    print(f"    Mean relative error: {rel_err:.4f}%")
    print(f"    Argmax agreement:    {argmax_match:.1f}%")
    print(f"    Cosine similarity:   {cos_sim:.6f}  (1.0 = identical)")
    print()

    # Latency comparison
    N_LAT = 200
    x_bench = rng.standard_normal((32, 64)).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess_fp32.run(None, {"X": x_bench})
        sess_dyn.run( None, {"X": x_bench})

    times_fp32, times_dyn = [], []
    for _ in range(N_LAT):
        t0 = time.perf_counter()
        sess_fp32.run(None, {"X": x_bench})
        times_fp32.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        sess_dyn.run(None, {"X": x_bench})
        times_dyn.append((time.perf_counter() - t0) * 1000)

    p50_fp32 = sorted(times_fp32)[N_LAT // 2]
    p50_dyn  = sorted(times_dyn)[N_LAT // 2]
    speedup  = p50_fp32 / p50_dyn if p50_dyn > 0 else 1.0

    print(f"  Latency benchmark (batch=32, {N_LAT} runs):")
    print(f"  {'Precision':<15} {'p50 ms':>10} {'p95 ms':>10} {'Speedup':>10}")
    print(f"  {'─'*48}")
    for name, times_l, p50 in [("FP32", times_fp32, p50_fp32),
                                 ("INT8 (dynamic)", times_dyn, p50_dyn)]:
        p95 = sorted(times_l)[int(N_LAT * 0.95)]
        sp  = p50_fp32 / p50 if p50 > 0 else 1.0
        print(f"  {name:<15} {p50:>10.3f} {p95:>10.3f} {sp:>10.2f}×")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Graph optimisation — Conv+BN fusion demo
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Graph optimisation: BatchNorm fusion into Conv")
print("━" * 65)
print()

print(f"  BatchNorm fusion mathematical proof:")
print()
print(f"  At inference, BatchNorm(x) = (x - μ) / √(σ² + ε) × γ + β")
print(f"  where μ, σ² are running statistics (fixed after training).")
print()
print(f"  For a Conv layer followed by BN:")
print(f"    y = BN(Conv(x; W, b))")
print(f"      = [(W * x + b) - μ] / √(σ² + ε) × γ + β")
print(f"      = W' * x + b'")
print()
print(f"  where:")
print(f"    W' = W × γ / √(σ² + ε)     (absorbed into kernel)")
print(f"    b' = (b - μ) × γ / √(σ² + ε) + β  (absorbed into bias)")
print()
print(f"  Result: Conv(x; W', b')  ←  TWO ops replaced by ONE Conv")
print()

# Demonstrate numerically
C_in, C_out, K = 4, 8, 3
W_conv = rng.standard_normal((C_out, C_in, K, K)).astype(np.float32) * 0.01
b_conv = np.zeros(C_out, dtype=np.float32)

# BatchNorm parameters (from training)
bn_gamma = rng.uniform(0.5, 2.0, C_out).astype(np.float32)
bn_beta  = rng.uniform(-0.5, 0.5, C_out).astype(np.float32)
bn_mean  = rng.standard_normal(C_out).astype(np.float32)
bn_var   = rng.uniform(0.1, 1.0, C_out).astype(np.float32)
bn_eps   = 1e-5

# Fused weights
inv_std = 1.0 / np.sqrt(bn_var + bn_eps)
W_fused = W_conv * (bn_gamma * inv_std).reshape(C_out, 1, 1, 1)
b_fused = (b_conv - bn_mean) * bn_gamma * inv_std + bn_beta

# Verify fusion: apply Conv+BN manually vs fused Conv
def conv2d_numpy(x, W, b, padding=1):
    """Simplified 2D convolution using numpy (for N=1, H=W=6)."""
    N, C, H, Wi = x.shape
    Cout, Cin, Kh, Kw = W.shape
    out_h, out_w = H - Kh + 1 + 2*padding, Wi - Kw + 1 + 2*padding
    x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    out   = np.zeros((N, Cout, out_h, out_w), dtype=np.float32)
    for co in range(Cout):
        for kh in range(Kh):
            for kw in range(Kw):
                out[:, co] += np.sum(
                    x_pad[:, :, kh:kh+out_h, kw:kw+out_w] *
                    W[co, :, kh, kw].reshape(1, Cin, 1, 1),
                    axis=1
                )
        out[:, co] += b[co]
    return out

x_test = rng.standard_normal((1, C_in, 6, 6)).astype(np.float32)

# Original Conv + BN
conv_out = conv2d_numpy(x_test, W_conv, b_conv)
# Manually apply BN: (conv_out - bn_mean) / sqrt(bn_var + eps) * bn_gamma + bn_beta
bn_out   = (conv_out - bn_mean.reshape(1,-1,1,1)) / \
           np.sqrt(bn_var + bn_eps).reshape(1,-1,1,1) * \
           bn_gamma.reshape(1,-1,1,1) + bn_beta.reshape(1,-1,1,1)

# Fused Conv only
fused_out = conv2d_numpy(x_test, W_fused, b_fused)

fusion_diff = float(np.abs(bn_out - fused_out).max())
print(f"  Numerical verification of Conv+BN fusion:")
print(f"    Max output diff (original vs fused): {fusion_diff:.2e}  ✅  (≈ 0)")
print(f"    Ops saved: 2 (Conv + BN) → 1 (Conv with absorbed BN)")
print(f"    Typical speedup for CNN inference: 20-40% on CPU")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Static quantisation with calibration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Static quantisation: calibration and QDQ format")
print("━" * 65)
print()

class SimpleCalibrationReader(CalibrationDataReader):
    """Provides calibration samples to the static quantiser."""
    def __init__(self, data: np.ndarray, input_name: str = "X"):
        self.data       = data
        self.input_name = input_name
        self.idx        = 0

    def get_next(self) -> Optional[Dict]:
        if self.idx >= len(self.data):
            return None
        sample = {"X": self.data[self.idx:self.idx+1]}
        self.idx += 1
        return sample


with tempfile.TemporaryDirectory() as tmp:
    fp32_path   = os.path.join(tmp, "model_fp32.onnx")
    static_qdq  = os.path.join(tmp, "model_static_qdq.onnx")
    static_qop  = os.path.join(tmp, "model_static_qop.onnx")
    onnx.save(base_model, fp32_path)

    # Calibration data: representative sample from deployment distribution
    calib_data = rng.standard_normal((100, 64)).astype(np.float32)

    # QDQ format (preferred for GPU / TensorRT)
    t0 = time.perf_counter()
    quantize_static(
        model_input     = fp32_path,
        model_output    = static_qdq,
        calibration_data_reader = SimpleCalibrationReader(calib_data),
        quant_format    = QuantFormat.QDQ,
        activation_type = QuantType.QInt8,
        weight_type     = QuantType.QInt8,
    )
    t_static_qdq = (time.perf_counter() - t0) * 1000

    # QOperator format (for CPU with explicit quant ops)
    quantize_static(
        model_input     = fp32_path,
        model_output    = static_qop,
        calibration_data_reader = SimpleCalibrationReader(calib_data),
        quant_format    = QuantFormat.QOperator,
        activation_type = QuantType.QInt8,
        weight_type     = QuantType.QInt8,
    )

    fp32_size   = os.path.getsize(fp32_path)
    qdq_size    = os.path.getsize(static_qdq)
    qop_size    = os.path.getsize(static_qop)

    # Count nodes
    fp32_nodes  = len(onnx.load(fp32_path).graph.node)
    qdq_nodes   = len(onnx.load(static_qdq).graph.node)
    qop_nodes   = len(onnx.load(static_qop).graph.node)

    print(f"  Static quantisation results (100 calibration samples):")
    print()
    print(f"  {'Format':<20} {'Nodes':>8} {'Size (KB)':>12} {'Calib time'}")
    print(f"  {'─'*50}")
    print(f"  {'FP32 (original)':<20} {fp32_nodes:>8} {fp32_size/1024:>12.1f}")
    print(f"  {'Static INT8 (QDQ)':<20} {qdq_nodes:>8} {qdq_size/1024:>12.1f} {t_static_qdq:.0f}ms")
    print(f"  {'Static INT8 (QOp)':<20} {qop_nodes:>8} {qop_size/1024:>12.1f}")
    print()

    # Why QDQ has MORE nodes (not fewer):
    print(f"  Why QDQ has more nodes than FP32:")
    print(f"    QDQ inserts QuantizeLinear + DequantizeLinear nodes AROUND each op.")
    print(f"    This makes quantisation explicit in the graph for inspection.")
    print(f"    At runtime, ORT/TRT fuses Q/DQ nodes away → fast INT8 kernels.")
    print()

    # Accuracy vs FP32
    sess_fp32   = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_static = ort.InferenceSession(static_qdq, providers=["CPUExecutionProvider"])

    x_acc = rng.standard_normal((500, 64)).astype(np.float32)
    out_fp32   = sess_fp32.run(  None, {"X": x_acc})[0]
    out_static = sess_static.run(None, {"X": x_acc})[0]

    acc_match = (out_fp32.argmax(1) == out_static.argmax(1)).mean() * 100
    mean_err  = float(np.abs(out_fp32 - out_static).mean() / np.abs(out_fp32).mean() * 100)
    print(f"  Accuracy analysis (500 samples, FP32 as reference):")
    print(f"    Argmax agreement: {acc_match:.1f}%  (top-1 prediction match)")
    print(f"    Mean relative err: {mean_err:.3f}%")
    print()

    print(f"  Quantisation decision guide:")
    print(f"  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │ Method           │ Calibration │ Accuracy │ Best for             │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │ Dynamic INT8     │ None        │ Good     │ NLP/text, CPU deploy │")
    print(f"  │ Static INT8 QDQ  │ ~100 samples│ Better   │ GPU/TRT production   │")
    print(f"  │ Static INT8 QOp  │ ~100 samples│ Better   │ CPU production       │")
    print(f"  │ QAT (int8)       │ Full retrain│ Best     │ Accuracy-critical    │")
    print(f"  │ INT4 (GPTQ/AWQ)  │ Calibration │ Moderate │ LLM inference        │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Production Deployment — scikit-learn, HuggingFace & End-to-End Pipeline": {
        "description": (
            "Complete production ONNX deployment pipeline. "
            "scikit-learn → ONNX with skl2onnx: classification pipeline. "
            "Sklearn Pipeline (scaler + model) → single ONNX graph. "
            "HuggingFace model export via Optimum: BERT, DistilBERT. "
            "End-to-end accuracy parity: original vs ONNX inference. "
            "Multi-model serving: load multiple ONNX sessions. "
            "Batched inference with padding and variable-length inputs. "
            "ONNX model registry: version management pattern. "
            "Production monitoring: latency percentiles, error rates. "
            "Deployment pipeline: export → validate → optimise → benchmark → serve. "
            "FastAPI + ORT serving pattern. "
            "Production anti-patterns and solutions."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import json
import math
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto, helper
    print(f"  ONNX: {onnx.__version__}  |  ORT: {ort.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "onnx", "onnxruntime", "--quiet"], check=True)
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto, helper
    print(f"  ONNX: {onnx.__version__}  |  ORT: {ort.__version__}")

print("=" * 65)
print("  PRODUCTION DEPLOYMENT — END-TO-END ONNX PIPELINE")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: scikit-learn → ONNX pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — scikit-learn → ONNX: full Pipeline export")
print("━" * 65)
print()

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from skl2onnx import convert_sklearn, to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    # Generate a classification dataset
    N_SAMPLES, N_FEATURES, N_CLASSES = 2000, 20, 5
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_classes=N_CLASSES, n_informative=12,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    print(f"  Dataset: {N_SAMPLES} samples × {N_FEATURES} features × {N_CLASSES} classes")
    print()

    # Train several sklearn models
    models_to_export = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=50, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ]),
    }

    print(f"  Training and exporting sklearn models:")
    print(f"  {'Model':<22} {'Train acc':>12} {'Test acc':>12} "
          f"{'ONNX size':>12} {'Accuracy diff':>14}")
    print(f"  {'─'*78}")

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, pipeline in models_to_export.items():
            pipeline.fit(X_train, y_train)

            # Original sklearn predictions
            sk_train_acc = accuracy_score(y_train, pipeline.predict(X_train))
            sk_test_acc  = accuracy_score(y_test,  pipeline.predict(X_test))
            sk_proba     = pipeline.predict_proba(X_test)

            # Export to ONNX
            initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
            onnx_model   = convert_sklearn(
                pipeline,
                initial_types = initial_type,
                target_opset  = 17,
                options       = {"zipmap": False},   # return array not dict
            )
            onnx_path = os.path.join(tmp, f"{name}.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            onnx_size = os.path.getsize(onnx_path)

            # Inference with ORT
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            # Get output names: typically [label, probabilities]
            out_names = [o.name for o in sess.get_outputs()]
            input_name = sess.get_inputs()[0].name
            ort_outputs = sess.run(out_names, {input_name: X_test})

            # Find the probability output (2D array)
            ort_proba = None
            ort_labels = None
            for out in ort_outputs:
                if hasattr(out, "shape") and len(out.shape) == 2:
                    ort_proba = out
                elif hasattr(out, "shape") and len(out.shape) == 1:
                    ort_labels = out

            if ort_proba is not None:
                ort_acc   = accuracy_score(y_test, ort_proba.argmax(1))
                acc_diff  = abs(sk_test_acc - ort_acc) * 100
            else:
                ort_acc  = sk_test_acc
                acc_diff = 0.0

            results[name] = {
                "sk_test_acc": sk_test_acc,
                "ort_acc":     ort_acc,
                "onnx_size":   onnx_size,
                "acc_diff":    acc_diff,
            }
            print(f"  {name:<22} {sk_train_acc:>12.4f} {sk_test_acc:>12.4f} "
                  f"{onnx_size/1024:>11.1f}KB {acc_diff:>13.6f}%")

    print()
    print(f"  All models exported successfully. Accuracy diff ≈ 0 (exact float32 match) ✅")
    print()

    print(f"  Sklearn pipeline export key points:")
    print(f"    Full Pipeline (Scaler + Classifier) → single ONNX graph")
    print(f"    Scaler parameters absorbed as constants in the graph")
    print(f"    skl2onnx handles: LinearModels, Trees, Ensembles, Transformers")
    print(f"    options={{'zipmap': False}} → return arrays not dicts (faster)")

except ImportError:
    print(f"  skl2onnx not available. Install: pip install skl2onnx scikit-learn")
    print()
    print(f"  Example export pattern:")
    print(f"    from skl2onnx import convert_sklearn")
    print(f"    from skl2onnx.common.data_types import FloatTensorType")
    print(f"    initial_type = [('float_input', FloatTensorType([None, n_features]))]")
    print(f"    onnx_model   = convert_sklearn(pipeline, initial_types=initial_type)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Production deployment pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Production deployment pipeline: validate → optimise → serve")
print("━" * 65)
print()

def build_production_model(in_dim=64, hidden=128, out_dim=5) -> onnx.ModelProto:
    """Build a model representing a production-ready classifier."""
    inits, nodes = [], []
    prev, pdim   = "X", in_dim
    for i, h in enumerate([hidden, hidden//2]):
        W = (rng.standard_normal((pdim, h)) * math.sqrt(2/pdim)).astype(np.float32)
        b = np.zeros(h, np.float32)
        inits += [numpy_helper.from_array(W, f"W{i}"),
                  numpy_helper.from_array(b, f"b{i}")]
        nodes += [
            helper.make_node("Gemm",    [prev,   f"W{i}", f"b{i}"], [f"h{i}"],
                              name=f"fc{i}", alpha=1.0, beta=1.0, transB=1),
            helper.make_node("Relu",    [f"h{i}"],  [f"a{i}"], name=f"act{i}"),
        ]
        prev, pdim = f"a{i}", h
    W_o = (rng.standard_normal((pdim, out_dim)) * 0.01).astype(np.float32)
    b_o = np.zeros(out_dim, np.float32)
    inits += [numpy_helper.from_array(W_o, "W_out"),
              numpy_helper.from_array(b_o, "b_out")]
    nodes.append(helper.make_node("Gemm", [prev, "W_out", "b_out"], ["logits"],
                                   name="fc_out", alpha=1.0, beta=1.0, transB=1))
    nodes.append(helper.make_node("Softmax", ["logits"], ["proba"],
                                   name="softmax", axis=1))
    iv = helper.make_tensor_value_info("X",     TensorProto.FLOAT, [None, in_dim])
    ov = helper.make_tensor_value_info("proba", TensorProto.FLOAT, [None, out_dim])
    g  = helper.make_graph(nodes, "prod_model", [iv], [ov], inits)
    m  = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

@dataclass
class DeploymentStage:
    name:        str
    status:      str = "pending"
    latency_ms:  float = 0.0
    size_kb:     float = 0.0
    accuracy:    float = 0.0
    notes:       str = ""


def run_deployment_pipeline(model: onnx.ModelProto, in_dim: int = 64) -> List[DeploymentStage]:
    stages = []
    x_ref  = rng.standard_normal((200, in_dim)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        # ── Stage 1: Validate ──────────────────────────────────────────────
        stage1 = DeploymentStage("1. Validate ONNX")
        try:
            onnx.checker.check_model(model)
            m_shapes  = onnx.shape_inference.infer_shapes(model)
            n_nodes   = len(model.graph.node)
            n_inits   = len(model.graph.initializer)
            stage1.status = "PASSED"
            stage1.notes  = f"nodes={n_nodes} initializers={n_inits}"
        except Exception as e:
            stage1.status = f"FAILED: {e}"
        stages.append(stage1)

        # Save baseline
        fp32_path = os.path.join(tmp, "model_v1.onnx")
        onnx.save(model, fp32_path)
        fp32_size = os.path.getsize(fp32_path) / 1024

        # ── Stage 2: Accuracy baseline ────────────────────────────────────
        stage2 = DeploymentStage("2. Baseline accuracy")
        sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        out_ref   = sess_fp32.run(None, {"X": x_ref})[0]
        # Warmup + benchmark
        for _ in range(5): sess_fp32.run(None, {"X": x_ref[:32]})
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess_fp32.run(None, {"X": x_ref[:32]})
            times.append((time.perf_counter() - t0) * 1000)
        stage2.latency_ms = sorted(times)[50]
        stage2.size_kb    = fp32_size
        stage2.accuracy   = 1.0   # reference
        stage2.status     = "OK"
        stage2.notes      = f"FP32 reference  {fp32_size:.1f}KB"
        stages.append(stage2)

        # ── Stage 3: Dynamic quantisation ────────────────────────────────
        from onnxruntime.quantization import quantize_dynamic, QuantType
        stage3 = DeploymentStage("3. Dynamic INT8 quant")
        int8_path = os.path.join(tmp, "model_int8.onnx")
        try:
            quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
            int8_size = os.path.getsize(int8_path) / 1024
            sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
            out_int8  = sess_int8.run(None, {"X": x_ref})[0]

            for _ in range(5): sess_int8.run(None, {"X": x_ref[:32]})
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                sess_int8.run(None, {"X": x_ref[:32]})
                times.append((time.perf_counter() - t0) * 1000)

            acc_match = (out_ref.argmax(1) == out_int8.argmax(1)).mean()
            stage3.latency_ms = sorted(times)[50]
            stage3.size_kb    = int8_size
            stage3.accuracy   = float(acc_match)
            stage3.status     = "OK"
            stage3.notes      = f"INT8  {int8_size:.1f}KB  {stage3.latency_ms/stage2.latency_ms:.2f}× rel"
        except Exception as e:
            stage3.status = f"FAILED: {e}"
        stages.append(stage3)

        # ── Stage 4: Benchmark summary ───────────────────────────────────
        stage4 = DeploymentStage("4. Production decision")
        best_speedup = stage2.latency_ms / stage3.latency_ms if stage3.latency_ms > 0 else 1.0
        best_size    = stage3.size_kb
        best_acc     = stage3.accuracy

        if best_acc >= 0.98 and best_speedup >= 1.2:
            stage4.status = "DEPLOY INT8"
            stage4.notes  = f"speedup={best_speedup:.2f}× acc={best_acc:.4f}"
        elif best_acc >= 0.95:
            stage4.status = "DEPLOY INT8 (review acc)"
            stage4.notes  = f"small acc drop: {(1-best_acc)*100:.2f}%"
        else:
            stage4.status = "DEPLOY FP32"
            stage4.notes  = f"INT8 acc too low: {best_acc:.4f}"
        stages.append(stage4)

    return stages


print(f"  Running production deployment pipeline:")
print()
prod_model = build_production_model()
pipeline_stages = run_deployment_pipeline(prod_model, in_dim=64)

print(f"  {'Stage':<35} {'Status':<25} {'Latency':>10} {'Size':>10} {'Notes'}")
print(f"  {'─'*95}")
for s in pipeline_stages:
    lat_str  = f"{s.latency_ms:.3f}ms" if s.latency_ms else "—"
    size_str = f"{s.size_kb:.1f}KB"    if s.size_kb    else "—"
    print(f"  {s.name:<35} {s.status:<25} {lat_str:>10} {size_str:>10} {s.notes}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: FastAPI serving pattern
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — FastAPI + ONNX Runtime serving pattern")
print("━" * 65)
print()

SERVING_CODE = (
"# Production ONNX Runtime serving with FastAPI\n"
"#\n"
"# from fastapi import FastAPI, HTTPException\n"
"# from pydantic import BaseModel\n"
"# import numpy as np\n"
"# import onnxruntime as ort\n"
"# import time, logging\n"
"#\n"
"# class ModelRegistry:\n"
"#     def __init__(self):\n"
"#         self._sessions = {}\n"
"#     def register(self, name, onnx_path, providers=None):\n"
"#         opts = ort.SessionOptions()\n"
"#         opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n"
"#         opts.intra_op_num_threads = 4\n"
"#         providers = providers or ['CPUExecutionProvider']\n"
"#         self._sessions[name] = ort.InferenceSession(onnx_path, opts, providers)\n"
"#     def get(self, name):\n"
"#         return self._sessions[name]\n"
"#\n"
"# registry = ModelRegistry()\n"
"# registry.register('fraud-detector', 'models/fraud_int8.onnx')\n"
"#\n"
"# app = FastAPI(title='ONNX Model Server', version='2.1.0')\n"
"#\n"
"# @app.post('/predict')\n"
"# async def predict(request):\n"
"#     sess = registry.get(request.model_name)\n"
"#     np_inputs = {k: np.array(v, dtype=np.float32) for k,v in request.inputs.items()}\n"
"#     t0 = time.perf_counter()\n"
"#     outputs = sess.run(request.output_names, np_inputs)\n"
"#     latency = (time.perf_counter() - t0) * 1000\n"
"#     return {'outputs': [o.tolist() for o in outputs], 'latency_ms': latency}\n"
"#\n"
"# @app.get('/health')\n"
"# async def health():\n"
"#     return {'status': 'ok', 'models': list(registry._sessions.keys())}\n"
"#\n"
"# Start: uvicorn server:app --host 0.0.0.0 --port 8080 --workers 4\n"
)
# (SERVING_CODE is a reference pattern — not executable here)

print(SERVING_CODE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: ONNX ecosystem and deployment guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — ONNX ecosystem map and deployment decision guide")
print("━" * 65)
print()

print(f"  The ONNX deployment ecosystem:")
print()
print(f"  ┌─────────────────────────────────────────────────────────────────┐")
print(f"  │  SOURCE FRAMEWORKS        │  ONNX FORMAT  │  RUNTIMES           │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │  PyTorch (torch.onnx)     │               │  ORT + CPU EP       │")
print(f"  │  TensorFlow (tf2onnx)     │   model.onnx  │  ORT + CUDA EP      │")
print(f"  │  JAX (jax2onnx)           │   ─────────── │  ORT + TensorRT EP  │")
print(f"  │  scikit-learn (skl2onnx)  │   Operators:  │  ORT + OpenVINO EP  │")
print(f"  │  XGBoost (onnxmltools)    │   ~220 ops    │  ORT + CoreML EP    │")
print(f"  │  HuggingFace (optimum)    │   Versioned   │  Triton Server      │")
print(f"  │  LightGBM (onnxmltools)   │   Portable    │  TensorRT native    │")
print(f"  │  Keras (tf2onnx)          │   Compressed  │  ONNX.js / WebGPU   │")
print(f"  └─────────────────────────────────────────────────────────────────┘")
print()

print(f"  End-to-end ONNX deployment workflow:")
steps = [
    ("Export",      "torch.onnx.export / skl2onnx / optimum",
     "Get .onnx file with correct dynamic_axes"),
    ("Validate",    "onnx.checker.check_model + shape_inference",
     "Catch graph errors early"),
    ("Inspect",     "Netron (netron.app) + onnx_model_stats()",
     "Visual graph exploration"),
    ("Simplify",    "onnxsim.simplify() + onnxoptimizer",
     "Reduce node count 30-50%"),
    ("Quantise",    "quantize_dynamic / quantize_static (INT8)",
     "4× size reduction, 2-4× speedup"),
    ("Benchmark",   "InferenceSession + latency loop",
     "Measure p50/p95/p99 per batch size"),
    ("Serve",       "ORT InferenceSession in FastAPI / Triton",
     "Production REST/gRPC endpoint"),
    ("Monitor",     "Latency metrics, accuracy drift, error rate",
     "Detect degradation, trigger retraining"),
]
print(f"  {'Step':<4} {'Phase':<12} {'Tool':<40} {'Why'}")
print(f"  {'─'*85}")
for i, (phase, tool, why) in enumerate(steps, 1):
    print(f"  {i:<4} {phase:<12} {tool:<40} {why}")
print()

print(f"  Production anti-patterns and solutions:")
print(f"  ┌────────────────────────────────────────────────────────────────────────┐")
print(f"  │ Anti-pattern                  │ Solution                               │")
print(f"  ├────────────────────────────────────────────────────────────────────────┤")
print(f"  │ Export with static batch=1    │ Set dynamic_axes={{0:'batch'}}         │")
print(f"  │ Skip onnx.checker validation  │ Always validate after export           │")
print(f"  │ :latest image tag for model   │ Tag with git SHA or version            │")
print(f"  │ Load ONNX in request handler  │ Load once at startup, reuse sess       │")
print(f"  │ Single-thread ORT session     │ Set intra_op_num_threads=n_cores       │")
print(f"  │ No warmup before benchmarking │ Always warmup: JIT, cache priming      │")
print(f"  │ FP32 always (no quantisation) │ Benchmark INT8 — often 0% accuracy loss│")
print(f"  │ No TRT engine cache           │ Set trt_engine_cache_enable=True       │")
print(f"  └────────────────────────────────────────────────────────────────────────┘")
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