"""
ONNX — Open Neural Network Exchange
=====================================

ONNX (Open Neural Network Exchange) is an open-source format for
representing machine learning models. Where LLVM IR is the universal
assembly for general computation, ONNX is the universal assembly for
neural network computation — a file format, a type system, a graph
specification, and a runtime ecosystem that lets a model trained in
PyTorch run in TensorFlow Serving, compiled by TVM, optimised by
TensorRT, and deployed on an ARM CPU without any framework present.

Before ONNX, moving a model between frameworks required:
    - Manual weight copying layer by layer
    - Re-implementing the architecture in the target framework
    - Debugging numerical discrepancies between implementations
    - Maintaining two codebases (training + inference)

ONNX solves this with a single portable file (.onnx) that encodes:
    - The computation graph (nodes and edges)
    - Operator semantics (exactly what each op computes)
    - Tensor shapes and data types (with optional dynamic dimensions)
    - Trained weights (as initializers in the graph)
    - Metadata (model version, author, opset version)

ONNX is larger than just neural networks. It has two distinct specs:
    ONNX (standard):  neural network operators (Conv, MatMul, Attention …)
    ONNX-ML:          classical ML operators (trees, SVMs, linear models,
                      pipelines from sklearn, XGBoost, LightGBM …)

In the compiler stack:
    PyTorch/TF/JAX  →  export to ONNX  →  ONNX Runtime / TVM / TRT
    ONNX is the bridge between the training world and the inference world.

"""

import textwrap
import re

TOPIC_NAME   = "ONNX — Open Neural Network Exchange"
DISPLAY_NAME = "10 · ONNX"
ICON         = "🔀"
SUBTITLE     = "The Universal Model Exchange Format and Runtime Ecosystem"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY ONNX EXISTS: THE INTEROPERABILITY PROBLEM

### The Framework Fragmentation Problem

    By 2017, deep learning had fractured into a dozen frameworks:
    TensorFlow, PyTorch, Caffe, Caffe2, MXNet, Theano, CNTK, PaddlePaddle,
    Chainer, Keras, DL4J, and more. Each had:
        - Its own model format (SavedModel, .pt, .caffemodel, .params, ...)
        - Its own operator naming conventions
        - Its own type system and graph representation
        - Its own runtime library

    A model trained in PyTorch could not be served by TensorFlow Serving.
    A model in Caffe could not be optimised by TensorRT without manual work.
    A research model could not be deployed to a mobile device without
    rewriting it in TensorFlow Lite or Core ML.

    This created a concrete engineering problem in every AI organisation:
        Train    → PyTorch (researchers prefer it)
        Optimise → TensorRT (requires TensorFlow/ONNX/native format)
        Serve    → TF Serving (requires TensorFlow SavedModel)
        Mobile   → TFLite or Core ML (requires yet another conversion)

    Each arrow in that chain was a manual, error-prone, often lossy
    conversion step — or a weeks-long porting project.

### ONNX's Founding and Mission

    ONNX was created by Facebook (PyTorch team) and Microsoft in September 2017,
    then joined by Amazon, IBM, Intel, Nvidia, Qualcomm, and 40+ other companies.
    It became an open standard governed by the Linux Foundation.

    The central idea:
        ┌───────────────────────────────────────────────────────────────┐
        │  Define ONE common format that all frameworks can export to   │
        │  and all runtimes can import from.                            │
        │                                                               │
        │  Framework A → ONNX → Runtime / Framework B / Compiler        │
        │                                                               │
        │  N exporters + M importers  =  N+M integrations, not N×M.     │
        └───────────────────────────────────────────────────────────────┘

    This is the same insight that motivated LLVM — a shared IR eliminates
    the need for N×M tool chains.

### ONNX's Position in the ML Stack

    ONNX sits at the boundary between training and inference:

    TRAINING WORLD (framework-specific):
        PyTorch, JAX, TensorFlow, MXNet, PaddlePaddle, ...
        Optimised for: flexibility, research, gradient computation
        Output: trained weights + architecture definition

    ONNX (the bridge):
        Serialises the trained model in a framework-neutral format
        Validates the graph for operator correctness
        Optimises the graph (via ONNX Optimizer)

    INFERENCE WORLD (runtime-specific):
        ONNX Runtime (cross-platform)
        TensorRT (NVIDIA GPU)
        TVM (any hardware, with auto-scheduling)
        OpenVINO (Intel CPU/GPU/VPU)
        Core ML (Apple hardware)
        TFLite (mobile)
        DirectML (Windows GPU)
        ACL (ARM Compute Library)


##### PART 2 — THE ONNX FORMAT: PROTOBUF, GRAPH, AND NODES

### ONNX is Protobuf Under the Hood

    ONNX files (.onnx) are serialised Google Protocol Buffers (protobuf).
    The schema is defined in onnx.proto3 — the authoritative specification.

    Hierarchy:
        ModelProto
        └── GraphProto
            ├── NodeProto[]      (the computation graph — operations)
            ├── TensorProto[]    (initializers — trained weights)
            ├── ValueInfoProto[] (type/shape info for edges)
            └── metadata

    A .onnx file is binary by default. Human-readable text format:
        import onnx
        model = onnx.load("resnet50.onnx")
        print(onnx.helper.printable_graph(model.graph))

### ModelProto — Top-Level Container

    The ModelProto carries:
        ir_version:    ONNX IR spec version (e.g., 8 for ONNX 1.13+)
        opset_imports: which opset versions are used (e.g., opset 17)
        domain:        "" for standard ONNX ops, custom string for extensions
        model_version: user-defined version number
        doc_string:    human-readable description
        graph:         the GraphProto containing the computation
        metadata_props: key-value string pairs (author, training_info, etc.)

    Opset version is CRITICAL:
        Each ONNX opset version defines the exact semantics of all operators.
        opset_version=11: BatchNorm has slightly different training mode handling.
        opset_version=13: Squeeze/Unsqueeze have different input arrangements.
        opset_version=17: New ops added (LayerNormalization as a single op).
        The exporting framework must specify which opset it targets.
        The runtime must support that opset version.

### GraphProto — The Computation Graph

    The graph is a directed acyclic graph (DAG):
        - NODES: operations (MatMul, Conv, Relu, ...)
        - EDGES: named tensors flowing between nodes
        - INPUTS: graph inputs (model inputs + initializers/weights)
        - OUTPUTS: graph outputs (model predictions)
        - INITIALIZERS: constant tensors (weights, biases, BN statistics)

    Everything in ONNX is identified by NAME (a string).
    An edge is just a string that appears in one node's outputs and
    another node's inputs. No explicit edge objects.

### NodeProto — One Operation

    Each node has:
        op_type:    operation name (e.g., "Conv", "MatMul", "Relu")
        domain:     "" for standard ops, custom domain for extensions
        inputs:     list of input names (strings — reference to edges/initializers)
        outputs:    list of output names (strings — new edges created)
        attributes: op-specific configuration (kernel_size, dilations, group, ...)
        name:       optional human-readable node name (for debugging)

    Example node (Relu):
        node {
            op_type: "Relu"
            inputs:  ["conv_output"]     # incoming edge
            outputs: ["relu_output"]     # outgoing edge
        }

    Example node (Conv) with attributes:
        node {
            op_type: "Conv"
            inputs:  ["input", "W", "B"]   # input tensor, weights, bias
            outputs: ["conv_output"]
            attribute { name: "kernel_shape"  ints: [3, 3] }
            attribute { name: "pads"          ints: [1, 1, 1, 1] }
            attribute { name: "strides"       ints: [1, 1] }
            attribute { name: "group"         i: 1 }
        }

### TensorProto — Weights and Constants

    Initializers store trained weights as TensorProto:
        dims:       [64, 3, 3, 3]     shape of the weight tensor
        data_type:  FLOAT (1)          ONNX element type enum
        raw_data:   <binary bytes>     little-endian raw tensor data
        name:       "conv1.weight"     matches the name in node.inputs

    ONNX data types:
        FLOAT (1):       float32 — most common
        DOUBLE (11):     float64
        FLOAT16 (10):    float16 (FP16)
        BFLOAT16 (16):   bfloat16 (added in opset 13)
        INT8 (3):        int8 — quantised models
        UINT8 (2):       uint8 — quantised activations
        INT32 (6):       int32 — indices, counts
        INT64 (7):       int64 — shape tensors, gather indices
        BOOL (9):        boolean — mask tensors
        STRING (8):      for NLP token strings (rare)
        FLOAT8E4M3FN (17): FP8 (4-bit exponent, 3-bit mantissa) — H100 training
        FLOAT8E5M2   (19): FP8 (5-bit exponent, 2-bit mantissa)

### The External Data Format — Handling Models Larger Than 2 GB

    Protocol Buffers has a hard 2 GB serialisation limit. This affects
    any model with more than ~500M float32 parameters. The solution is
    EXTERNAL DATA FORMAT: weights are stored in separate binary files
    alongside the .onnx protobuf skeleton.

    Exporting with external data (PyTorch):
        torch.onnx.export(
            model, dummy, "model.onnx",
            opset_version=17,
        )
        # For large models, PyTorch automatically writes:
        #   model.onnx       — the graph structure (small, <100 MB)
        #   model.onnx.data  — the raw weight bytes (can be tens of GB)

        # Or explicitly:
        from torch.onnx import ExportOptions
        torch.onnx.dynamo_export(model, *args).save(
            "model.onnx",
            external_data=True,   # force external data
        )

    External TensorProto format:
        The TensorProto for an externally-stored weight has:
            data_location: EXTERNAL      (not DEFAULT)
            external_data: [
                { key: "location",  value: "model.onnx.data" },
                { key: "offset",    value: "0" },
                { key: "length",    value: "536870912" },
                { key: "checksum",  value: "<sha1>" },
            ]

    Loading external data:
        model = onnx.load("model.onnx", load_external_data=True)
        # OR, if the .data file is in a different directory:
        onnx.load_external_data_for_model(model, "/path/to/data_dir/")

    ORT with external data:
        ort.InferenceSession("model.onnx")
        # ORT automatically reads the .data file next to the .onnx file.
        # The weight directory must be co-located.

    Large model best practices:
        Keep the .onnx and .data files in the same directory.
        Use consistent relative paths (avoid absolute paths).
        Check with: onnx.checker.check_model("model.onnx", full_check=True)

### Non-Tensor Types: Sequences and Maps

    ONNX supports two non-tensor value types beyond TensorProto:

    SEQUENCE TYPE (TypeProto.Sequence):
        A variable-length list of tensors of the same element type.
        Used by: object detection models (variable number of detections),
                 LSTM outputs (when collecting per-step outputs),
                 ONNX-ML classifiers (returning class probabilities per class).
        Example:
            value_info {
                name: "detections"
                type { sequence_type {
                    elem_type { tensor_type { elem_type: FLOAT } }
                }}
            }
        Ops: SequenceInsert, SequenceAt, SequenceLength, SequenceEmpty,
             SplitToSequence, ConcatFromSequence, SequenceErase

    MAP TYPE (TypeProto.Map):
        A dictionary mapping string/int keys to tensor values.
        Used exclusively by ONNX-ML models (tree classifiers return
        probability maps: {class_label: probability}).
        Example:
            type { map_type {
                key_type: STRING
                value_type { tensor_type { elem_type: FLOAT } }
            }}
        Ops: ZipMap (creates a map from keys + values tensors)

    OPTIONAL TYPE (TypeProto.Optional, opset 15+):
        A value that may or may not be present at runtime.
        Used for: conditional outputs, optional model inputs.
        Ops: Optional, OptionalHasElement, OptionalGetElement

### ValueInfoProto — Type and Shape Information

    Each edge (named tensor) can have optional type/shape metadata:

        value_info {
            name:  "conv_output"
            type {
                tensor_type {
                    elem_type: FLOAT
                    shape {
                        dim { dim_value: 1   }   # static batch=1
                        dim { dim_value: 64  }   # 64 channels
                        dim { dim_value: 224 }   # height
                        dim { dim_value: 224 }   # width
                    }
                }
            }
        }

    Dynamic shapes use dim_param instead of dim_value:
        dim { dim_param: "batch_size" }   # symbolic — unknown at export

    Type/shape info is OPTIONAL but strongly recommended for:
        - Operator shape inference (needed by many runtimes)
        - Graph validation (catch shape mismatches early)
        - Visualisation tools (Netron needs shapes to display)


##### PART 3 — ONNX OPERATORS: THE STANDARD OP SET

### The Opset Version System

    ONNX operators are versioned by "opset":
        Opset 1–7:   early 2017–2018, basic ops (Conv, Relu, MatMul, ...)
        Opset 9:     BatchNormalization updated; LSTM/GRU standardised
        Opset 11:    Pad gains dynamic mode; Gather/Scatter expanded
        Opset 13:    Squeeze/Unsqueeze become dynamic; new reduce ops
        Opset 14:    Trilu, HardSigmoid updated
        Opset 17:    LayerNormalization added (previously decomposed)
        Opset 18:    GroupNormalization, BiasGelu added (transformer focus)
        Opset 19+:   Ongoing — more transformer and LLM ops

    When you export:
        torch.onnx.export(..., opset_version=17)
        # Chooses opset 17 operator definitions

    Key rules:
        - An op's behaviour is frozen once defined in a given opset.
        - New opsets may REDEFINE an op (e.g., more flexible Reshape).
        - A model with opset 17 import CANNOT use opset 18 features.
        - The runtime must support the model's opset.

### Standard Operators (Selected)

    Neural network layers:
        Conv:           N-D convolution with configurable pads, strides, dilations, group
        ConvTranspose:  Transposed convolution (for decoders, upsampling)
        BatchNormalization: normalise + scale + shift per channel
        LayerNormalization: normalise across last N dims (transformers)
        GroupNormalization: normalise across groups of channels
        InstanceNormalization: normalise per sample per channel
        Dropout:        randomly zero activations (rate attribute)
        MaxPool:        spatial max pooling
        AveragePool:    spatial average pooling
        GlobalMaxPool:  global max over spatial dims
        GlobalAveragePool: global average over spatial dims
        Flatten:        flatten all dims except batch
        Reshape:        change shape (with dynamic shape tensor)
        Transpose:      permute dimensions
        Squeeze / Unsqueeze: remove / add size-1 dimensions

    Linear algebra:
        MatMul:         general matrix multiply (N-D batched)
        Gemm:           generalised matrix multiply: Y = alpha*A@B + beta*C
        Einsum:         Einstein summation (flexible but less supported)

    Activation functions:
        Relu, Sigmoid, Tanh, Elu, Selu, LeakyRelu
        HardSigmoid, HardSwish
        Gelu (opset 20), BiasGelu (opset 18)
        Softmax, LogSoftmax (axis attribute specifies which dim)
        Mish, Swish (custom ops in some runtimes)

    Elementwise arithmetic:
        Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log, Abs, Neg
        Max, Min, Clip (elementwise maximum, minimum, clamp)
        And, Or, Not (boolean)
        Equal, Greater, Less, GreaterOrEqual, LessOrEqual

    Reduction:
        ReduceSum, ReduceMean, ReduceMax, ReduceMin
        ReduceProd, ReduceL1, ReduceL2
        All with keepdims and axes attributes

    Shape manipulation:
        Concat:         concatenate along an axis
        Split:          split into chunks along an axis
        Slice:          extract sub-tensor (supports step)
        Gather:         index into tensor (like numpy indexing)
        GatherElements: element-wise gather with per-element indices
        Scatter:        scatter updates into tensor
        Pad:            add constant / reflection / wrap padding
        Tile:           repeat tensor N times along each axis
        Expand:         broadcast to a given shape
        Flatten:        collapse dims after a given axis

    Sequence / NLP:
        LSTM, GRU, RNN:  recurrent cells (single-direction)
        Embed (via Gather on embedding table)
        Attention:       in opset 17+ (from transformers usage)

    Control flow (opset 13+):
        If:             conditional branch (with sub-graph attributes)
        Loop:           while loop (with sub-graph attribute)
        Scan:           sequential map (like jax.lax.scan)

### Custom Operators

    ONNX supports CUSTOM OPERATORS via a domain string:
        domain: "com.microsoft"    — used by ONNX Runtime for extensions
        domain: "ai.onnx.preview.training" — training ops
        domain: ""                — standard ONNX ops

    Custom ops must be registered in the runtime:
        # ONNX Runtime custom op registration
        session_options.register_custom_ops_library("libcustom_ops.so")

    Common custom ops in practice:
        com.microsoft:BiasSoftmax     — fused bias + softmax
        com.microsoft:BiasGelu        — fused bias + GELU
        com.microsoft:RotaryEmbedding — RoPE for LLMs
        com.microsoft:Attention        — multi-head attention (packed)
        com.microsoft:GroupQueryAttention — GQA for modern LLMs


##### PART 4 — EXPORTING TO ONNX: FRAMEWORK SPECIFICS

### PyTorch → ONNX (torch.onnx.export)

    PyTorch traces the model with TorchScript to capture the computation,
    then maps each traced op to its ONNX equivalent.

    Basic export:
        torch.onnx.export(
            model,                           # nn.Module in eval() mode!
            (dummy_input,),                  # example inputs (used for tracing)
            "model.onnx",                    # output path
            opset_version=17,                # ONNX opset to target
            input_names=["input"],           # name the graph inputs
            output_names=["output"],         # name the graph outputs
            dynamic_axes={                   # declare dynamic dimensions
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            do_constant_folding=True,        # fold constant subgraphs at export
            export_params=True,              # embed trained weights in the file
        )

    Critical rules:
        1. model.eval() BEFORE export — disables dropout and BN training mode.
        2. dummy_input must have the right shape and dtype.
        3. opset_version controls which operator set is used.
        4. dynamic_axes prevents shape baking (needed for variable batch sizes).
        5. do_constant_folding=True folds positional encodings and other constants.

    Common export problems:
        "Data-dependent shape" error:
            Model uses x.shape[0] to index — shape isn't known at trace time.
            Fix: replace with tensor operations that don't depend on Python ints.

        "Unsupported op" error:
            PyTorch op has no ONNX equivalent in the target opset.
            Fix: use an earlier opset, or implement a custom symbolic function.

        "Operator not implemented in runtime":
            ONNX has the op but ONNX Runtime hasn't implemented it.
            Fix: use custom ops or decompose into supported ops.

### PyTorch 2.x: dynamo_export (New Preferred Path)

    torch.onnx.dynamo_export uses TorchDynamo instead of TorchScript:
        - Handles more dynamic Python code (conditions, loops)
        - Better support for transformers and modern architectures
        - Produces cleaner graphs (no TorchScript artifacts)
        - Supports more opsets
        - Supports ONNX Program IR (the modern graph representation)

        from torch.onnx import dynamo_export
        export_output = dynamo_export(model, *args, **kwargs)
        export_output.save("model.onnx")

    LEGACY TRACE vs DYNAMO — key differences:

    LEGACY (torch.onnx.export with TorchScript):
        Uses torch.jit.trace internally to capture the model.
        Produces a "flat" ONNX graph where Python control flow is baked.
        Fails on data-dependent shapes and modern architectures.
        Output contains TorchScript artifacts (Cast, Identity nodes).
        Still required for opset < 9 and some production pipelines.

    DYNAMO (torch.onnx.dynamo_export):
        Uses TorchDynamo to trace, producing an FX graph first.
        FX graph is then lowered to ONNX via ATen decomposition.
        Handles graph breaks via sub-graphs or fallback ops.
        Cleaner output: no spurious Cast/Identity nodes.
        Supports dynamic shapes symbolically via torch.export.Dim().
        Required for: transformers with custom attention, models with
                      data-dependent control flow, torch 2.x features.

    Dynamic shapes with dynamo_export:
        from torch.export import Dim
        batch = Dim("batch", min=1, max=256)
        seq   = Dim("seq",   min=1, max=4096)
        export_output = dynamo_export(
            model, sample_input,
            dynamic_shapes={"x": {0: batch, 1: seq}})
        export_output.save("model.onnx")

### Registering Custom ONNX Symbolic Functions

    When PyTorch ops have no standard ONNX equivalent, you can register
    a SYMBOLIC FUNCTION that tells the exporter how to represent the op:

    LEGACY EXPORTER symbolic registration:
        from torch.onnx import register_custom_op_symbolic

        def my_custom_op_symbolic(g, x, scale, zero_point):
            return g.op("DequantizeLinear", x,
                         g.op("Constant", value_t=torch.tensor(scale)),
                         g.op("Constant", value_t=torch.tensor(zero_point)))

        register_custom_op_symbolic(
            "::my_quantised_op",          ; PyTorch op namespace::name
            my_custom_op_symbolic,
            opset_version=13)

    DYNAMO EXPORTER via onnxscript:
        import onnxscript
        from onnxscript import opset17 as op

        @torch.onnx.symbolic("torch_namespace::my_op", opset=17)
        def my_op_to_onnx(x: onnxscript.FLOAT[...],
                           scale: float) -> onnxscript.FLOAT[...]:
            return op.Mul(x, op.Constant(value_float=scale))

    This mechanism is how all ~2000 PyTorch ops are mapped to ONNX —
    the mapping table lives in torch/onnx/symbolic_opset*.py.

### ONNX Opset Version Conversion

    Sometimes you need to upgrade (or downgrade) a model's opset:

        from onnx import version_converter
        # Upgrade from opset 11 to opset 17:
        model_v17 = version_converter.convert_version(model_v11, 17)
        onnx.checker.check_model(model_v17)
        onnx.save(model_v17, "model_v17.onnx")

    What version_converter does:
        Applies ADAPTER functions for each op that changed between versions.
        Example: Unsqueeze changed its axes from an attribute (opset 11)
                 to an input tensor (opset 13).
        The adapter automatically rewrites the node + inserts a Constant
        node for the axes value.

    Limitations:
        Not all conversions are possible (some opset changes are lossy).
        Some ops were added in later opsets and cannot be downgraded.
        Always validate with onnx.checker after conversion.

### TensorFlow → ONNX (tf2onnx)

    tf-onnx converts TF SavedModels and concrete functions:
        import tf2onnx, tensorflow as tf

        model = tf.saved_model.load("saved_model/")
        spec  = (tf.TensorSpec((None, 4), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
        with open("model.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())

    Or via command line:
        python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx

### JAX → ONNX (jax2onnx)

    JAX models can be exported via StableHLO → ONNX converter:
        # Via jax2onnx package:
        import jax2onnx
        model_proto = jax2onnx.convert(jax_fn, sample_inputs)

    Or via the XLA → ONNX path (StableHLO → ONNX converter):
        exported = jax.export.export(jax.jit(fn))(*args)
        # Then use stablehlo-onnx-bridge to convert

### Verifying ONNX Models

    ALWAYS verify after export:
        import onnx
        model = onnx.load("model.onnx")
        onnx.checker.check_model(model)   # raises if graph is invalid

        # Also run shape inference to fill in all ValueInfo:
        from onnx import shape_inference
        model_inferred = shape_inference.infer_shapes(model)
        onnx.save(model_inferred, "model_inferred.onnx")

    Cross-check with the original framework:
        import onnxruntime as ort
        sess    = ort.InferenceSession("model.onnx")
        ort_out = sess.run(None, {"input": sample_input.numpy()})[0]
        pt_out  = model(sample_input).detach().numpy()
        assert np.allclose(ort_out, pt_out, atol=1e-5), "Output mismatch!"


##### PART 5 — ONNX RUNTIME: THE PRODUCTION INFERENCE ENGINE

### What ONNX Runtime Is

    ONNX Runtime (ORT) is Microsoft's production-grade inference engine for
    ONNX models. It is NOT just a reference implementation — it is a
    heavily optimised, production-grade runtime used in:
        - Microsoft Office (AI features)
        - Bing Search and Azure ML
        - Windows ML API
        - HuggingFace Transformers (ORT backend)
        - Azure ML endpoints

    Key properties:
        Platform:   Windows, Linux, macOS, Android, iOS, WebAssembly
        Hardware:   CPU (x86, ARM), NVIDIA GPU, AMD GPU, Intel, Apple Silicon
        Language:   Python, C++, C#, Java, JavaScript, Swift, Objective-C

### Execution Providers — Hardware Abstraction

    ORT dispatches operations to backend libraries via "Execution Providers":

        CPUExecutionProvider:    default, uses Eigen/MLAS for matmul
        CUDAExecutionProvider:   NVIDIA GPU via cuDNN/cuBLAS
        TensorrtExecutionProvider: TensorRT for maximum GPU throughput
        ROCMExecutionProvider:   AMD GPU
        DmlExecutionProvider:    DirectML (Windows GPU — any GPU)
        OpenVINOExecutionProvider: Intel CPU/GPU/VPU via OpenVINO
        CoreMLExecutionProvider: Apple M-series ANE/GPU
        NNAPIExecutionProvider:  Android NNAPI (hardware NPU)
        XNNPackExecutionProvider: ARM/x86 SIMD via Google XNNPACK
        AzureExecutionProvider:  Azure cloud inference endpoints

    Provider priority: if an op is not supported by GPU provider,
    it falls back to the next provider in the list (usually CPU).

    Usage:
        sess = ort.InferenceSession(
            "model.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            # Try CUDA first, fall back to CPU for unsupported ops
        )

### EP Graph Partitioning — How ORT Assigns Ops to Hardware

    When ORT loads a model with multiple EPs, it solves the OP ASSIGNMENT
    problem: which EP handles each node in the graph?

    STEP 1 — CAPABILITY QUERY:
        ORT iterates every node and asks each EP (in priority order):
            "Can you run this op with these input/output types?"
        Each EP responds with a CAPABILITY object indicating:
            - YES (fully supported)
            - YES, but with type constraints (e.g., only float32)
            - NO

    STEP 2 — SUBGRAPH PARTITIONING:
        ORT assigns contiguous runs of compatible nodes to the same EP.
        This minimises data transfers between EP memory spaces.
        A "subgraph" is a maximal contiguous set of nodes on one EP.

    STEP 3 — DATA COPY INSERTION:
        When the graph crosses EP boundary (CPU node → GPU node):
            ORT inserts a MemcpyToDevice node (CPU → GPU copy).
            When it crosses back: MemcpyFromDevice (GPU → CPU copy).
        These copies are invisible to the user but can be a bottleneck.

    EXAMPLE — mixed model on CUDA EP:
        Node 0: MatMul         → CUDA EP   ✓ (fully supported)
        Node 1: Custom_Rope    → CPU EP    (CUDA EP doesn't know this op)
        Node 2: Add            → CUDA EP   ✓
        Node 3: LayerNorm      → CUDA EP   ✓

        ORT partitions:
            Partition A (CUDA): nodes 0
            [MemcpyFromDevice]
            Partition B (CPU):  node 1
            [MemcpyToDevice]
            Partition C (CUDA): nodes 2, 3

        COST: 2 extra CPU↔GPU copies per forward pass.
        SOLUTION: register Custom_Rope as a custom CUDA kernel,
                  or decompose it into standard ops ORT supports on CUDA.

    PERFORMANCE TIP:
        Minimise EP boundary crossings. Even a small op on CPU between
        two GPU ops costs ~0.5ms per boundary (latency + sync overhead).
        Check boundaries with:
            ORTENV_ORT_DISABLE_PROVIDER_FALLBACK=1 → raises on fallback
            or inspect sess.get_providers() per node.

### ORT Threading Model

    ORT has TWO parallelism dimensions:

    INTRA-OP PARALLELISM (within a single op):
        How many threads a single operator uses internally.
        Controls parallelism of BLAS routines (matmul, conv).
        Set via: sess_options.intra_op_num_threads = N
        Default: 0 (= all logical cores)
        Recommendation:
            Server inference (batch=1, low latency): N=1 or N=2
            Batch inference (high throughput):       N=all cores
            Embedded/edge:                           N=1 or N=2

    INTER-OP PARALLELISM (concurrent execution of independent ops):
        How many ops can execute in parallel when the graph allows it.
        Uses ORT's graph scheduler to find independent subgraphs.
        Set via: sess_options.inter_op_num_threads = N
        Default: 0 (= disabled, sequential execution)
        Enable when: your graph has parallel branches (e.g., multi-head
                     attention with independent head projections).
        Caution: enable only for models where profiling shows benefit;
                 thread coordination overhead can hurt latency.

    EXECUTION MODE:
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            All ops run in topological order. Default, safest.
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            ORT's scheduler runs independent ops concurrently.
            Requires inter_op_num_threads > 1.

    MEMORY ARENA:
        ORT uses an ARENA ALLOCATOR for GPU memory by default:
            Pre-allocates a large GPU memory chunk at session creation.
            Subsequent allocations come from the arena (no cudaMalloc calls).
            Eliminates fragmentation and reduces allocation overhead.
        Configure:
            arena_config = {"initial_chunk_size_bytes": 256 * 1024 * 1024}
            providers = [("CUDAExecutionProvider", arena_config)]

### ORT Graph Optimisation

    Before inference, ORT applies graph-level optimisations:

    Level 0 (Disabled):        no optimisation
    Level 1 (Basic):           constant folding, redundant node elimination
    Level 2 (Extended):        op fusion (Conv+Relu, Attention fusion, ...)
    Level 99 (All):            all optimisations including layout transforms

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession("model.onnx", sess_options)

    WHAT EACH LEVEL DOES IN DETAIL:

    LEVEL 1 (Basic) — always safe, no accuracy impact:
        Eliminate Identity nodes (pass-through with no effect)
        Constant folding: evaluate sub-graphs with all-constant inputs
        Eliminate Slice + Gather patterns → single op
        Remove unused initializers (dead weight elimination)

    LEVEL 2 (Extended) — op fusion, layout changes:
        Conv + BatchNormalization → Conv (absorb BN into conv weights)
            W_fused = gamma * W / sqrt(var + eps)
            b_fused = beta - gamma * mean / sqrt(var + eps)
        MatMul + Add → Gemm (single cuBLAS call)
        Gelu: Erf-based chain → single Gelu kernel
        LayerNorm: chain of sub/mean/div/mul/add → single kernel
        Attention: QKV MatMul + Reshape → packed MultiHeadAttention kernel
        SkipLayerNorm: Add + LayerNorm → single fused kernel
        EmbedLayerNorm: Gather + Add + LayerNorm → fused

    LEVEL 99 (All) — includes:
        Layout transformations: NHWC → NCHW where cuDNN prefers it
        Transpose folding: eliminate redundant Transpose chains
        Reshape elimination: remove adjacent Reshape nodes

    SAVING THE OPTIMISED GRAPH:
        sess_options.optimized_model_filepath = "model_optimised.onnx"
        # ORT writes the fully-optimised graph to disk.
        # Load this directly next time — skips re-optimisation.
        # IMPORTANT: the saved model is EP-specific (e.g., CUDA-optimised).

    Key fusions ORT performs:
        Conv + BatchNorm → fused (BN absorbed into conv weights at init)
        MatMul + Add → Gemm (bias absorbed)
        Attention: QKV projections fused into one multi-head attention op
        LayerNorm: composed ops fused into single LayerNorm kernel
        Gelu: exp + mul + add chain → single Gelu kernel

### ORT Profiling — Finding Inference Bottlenecks

    ORT has a built-in profiler that measures per-op execution time:

        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = "/tmp/ort_profile"

        sess = ort.InferenceSession("model.onnx", sess_options)
        output = sess.run(None, {"input": x})

        # Flush and get profile path:
        profile_path = sess.end_profiling()
        # profile_path = "/tmp/ort_profile_<timestamp>.json"

    The profile file is a Chrome Trace JSON (compatible with chrome://tracing):
        Open chrome://tracing in Chrome
        Load the .json file
        See a timeline of every op with:
            Op name, execution time (µs), memory usage, input shapes

    Alternatively, read programmatically:
        import json
        with open(profile_path) as f:
            profile = json.load(f)

        # Each entry has:
        #   "name":     op name (e.g., "Conv_0", "MatMul_1", "Relu_2")
        #   "dur":      duration in microseconds
        #   "cat":      category ("Node" or "Op")
        #   "args":     {"input_type": "float", "output_size": [1, 64, 7, 7]}

        op_times = [(e["name"], e["dur"]) for e in profile if e.get("cat") == "Node"]
        op_times.sort(key=lambda x: x[1], reverse=True)
        print("Top 5 slowest ops:")
        for name, dur in op_times[:5]:
            print(f"  {name}: {dur} µs")

    TYPICAL FINDINGS from ORT profiling:
        MatMul / Gemm:  often 60-80% of total for transformer models
        Conv:           often 60-80% for CNN models
        Softmax:        5-15% for attention-heavy models
        LayerNorm:      3-8% for transformers
        Reshape/Gather: usually negligible unless very frequent

### Quantisation in ONNX/ORT

    ORT supports INT8 quantisation via onnxruntime.quantization:

    Post-training dynamic quantisation (weights INT8, activations dynamic):
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic("model.onnx", "model_int8.onnx")
        # Weights quantised at export; activations quantised per batch at runtime

    Post-training static quantisation (both INT8, needs calibration data):
        from onnxruntime.quantization import quantize_static, CalibrationDataReader
        quantize_static("model.onnx", "model_static_int8.onnx", calibration_data_reader)
        # Requires representative dataset to compute activation scale factors

    Memory savings: ~4× (FP32 → INT8)
    Inference speedup on CPU with VNNI: ~2–4×
    Accuracy impact: typically < 1% on classification tasks

### IO Binding — Eliminating CPU↔GPU Copies

    By default, ORT copies inputs from CPU to GPU and outputs back:
        input (CPU numpy) → GPU → compute → GPU output → CPU numpy

    IO binding keeps tensors on GPU:
        io_binding = sess.io_binding()
        io_binding.bind_input("input", device_type="cuda", ...)
        io_binding.bind_output("output", device_type="cuda", ...)
        sess.run_with_iobinding(io_binding)
        # No CPU-GPU copy overhead — critical for high-throughput serving


##### PART 6 — ONNX GRAPH MANIPULATION AND THE ONNX PYTHON API

### The onnx Python Package

    The onnx Python package provides full access to the protobuf structures:

    Loading and inspection:
        import onnx
        model = onnx.load("model.onnx")
        graph = model.graph
        print(f"Nodes:        {len(graph.node)}")
        print(f"Inputs:       {[i.name for i in graph.input]}")
        print(f"Outputs:      {[o.name for o in graph.output]}")
        print(f"Initializers: {len(graph.initializer)}")

    Iterating nodes:
        for node in graph.node:
            print(f"{node.op_type:20s}  {node.input} → {node.output}")

    Accessing weights:
        for init in graph.initializer:
            weight = onnx.numpy_helper.to_array(init)
            print(f"{init.name}: shape={weight.shape}, dtype={weight.dtype}")

### Building ONNX Graphs Programmatically

    onnx.helper provides utilities for building graphs:

        from onnx import helper, TensorProto
        import numpy as np

        # Create nodes
        relu_node = helper.make_node("Relu", inputs=["x"], outputs=["h"])
        matmul_node = helper.make_node("MatMul", inputs=["h", "W"], outputs=["y"])

        # Create value info (type/shape descriptors)
        x_info = helper.make_tensor_value_info("x",    TensorProto.FLOAT, [None, 8])
        W_info = helper.make_tensor_value_info("W",    TensorProto.FLOAT, [8, 4])
        y_info = helper.make_tensor_value_info("y",    TensorProto.FLOAT, [None, 4])
        W_init = helper.make_tensor("W", TensorProto.FLOAT, [8, 4],
                                    np.random.randn(8, 4).astype(np.float32).flatten())

        # Create graph
        graph = helper.make_graph([relu_node, matmul_node],
                                    "my_graph",
                                    inputs=[x_info, W_info],
                                    outputs=[y_info],
                                    initializer=[W_init])

        # Create model
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 17)
        ])
        onnx.checker.check_model(model)
        onnx.save(model, "my_model.onnx")

### ONNX Graph Surgeon (onnx-graphsurgeon)

    NVIDIA's onnx-graphsurgeon (gs) provides a higher-level API for
    modifying ONNX graphs without working with raw protobuf:

        import onnx_graphsurgeon as gs
        import numpy as np

        graph = gs.import_onnx(onnx.load("model.onnx"))

        # Find all Relu nodes and replace with Identity (ablation study)
        for node in graph.nodes:
            if node.op == "Relu":
                node.op = "Identity"

        # Insert a new node between two existing nodes
        x_out = graph.outputs[0]   # existing output
        scale  = gs.Constant("scale", np.array([2.0], dtype=np.float32))
        scaled = gs.Variable("scaled", np.float32, x_out.shape)
        mul_node = gs.Node("Mul", inputs=[x_out, scale], outputs=[scaled])
        graph.nodes.append(mul_node)
        graph.outputs = [scaled]

        graph.cleanup()   # remove orphaned nodes/tensors
        onnx.save(gs.export_onnx(graph), "modified.onnx")

### ONNX Optimizer

    onnx-simplifier and onnxoptimizer clean up and simplify graphs:

        import onnxsim
        model_simplified, check = onnxsim.simplify("model.onnx")
        # Runs: constant folding, dead node elimination, shape folding

        from onnxoptimizer import optimize
        passes = ["eliminate_identity", "fuse_bn_into_conv",
                  "eliminate_unused_initializer", "fuse_consecutive_squeezes"]
        optimised = optimize(model, passes)


##### PART 7 — DYNAMIC SHAPES AND LIMITATIONS

### Static vs Dynamic Shapes in ONNX

    ONNX supports three kinds of dimension specifications:
        dim_value: 4       — static (fixed integer)
        dim_param: "batch" — dynamic (symbolic, any integer)
        (unset)            — completely unknown

    Good practice:
        - Make batch dimension symbolic: {0: "batch_size"}
        - Make sequence length symbolic for NLP: {1: "seq_len"}
        - Keep everything else static where possible

    Why static shapes are preferred:
        - Runtime can pre-allocate all memory
        - Shape inference is complete and correct
        - Compilers (TVM, TensorRT) can optimise more aggressively
        - No recompilation needed across runs

    Why dynamic shapes are needed:
        - Variable batch sizes (serving different concurrencies)
        - NLP models with variable sequence lengths
        - Detection models with variable number of proposals

### Common ONNX Export Pitfalls

    1. Data-dependent shapes (most common):
        BAD:  x = tensor[:n_valid]   # n_valid unknown at trace time
        FIX:  use masking instead of slicing with data-dependent sizes

    2. Python int operations on tensor shapes:
        BAD:  b, n, d = x.shape; y = x.view(b*n, d)  # b*n is Python int
        FIX:  y = x.view(-1, d)  or  x.reshape([x.shape[0]*x.shape[1], -1])

    3. Control flow based on tensor values:
        BAD:  if x.sum() > 0: ...   # condition unknown at trace time
        FIX:  use torch.where() or express as tensor operations

    4. In-place operations:
        BAD:  x[i] = value          # in-place indexing
        FIX:  x = torch.scatter(x, dim, index, value)

    5. model.train() instead of model.eval():
        BAD:  exporting in training mode → dropout included in graph
        FIX:  ALWAYS call model.eval() before export

    6. Wrong opset for the target runtime:
        BAD:  opset_version=19 for a runtime that supports only up to opset 13
        FIX:  check the runtime's opset support table before choosing

### ONNX's Limitations

    1. NO GRADIENTS:
        ONNX represents inference graphs only.
        The forward computation is captured; no backward pass.
        ONNX-ML Training spec exists but is rarely used.

    2. STATIC COMPUTATION GRAPH:
        Python control flow (if/for) is baked into the graph at export.
        Truly dynamic computation (graph structure changing per input) is hard.
        Exception: Loop/If/Scan ops support dynamic control flow but are
        complex to export correctly.

    3. OPERATOR COVERAGE GAPS:
        New PyTorch ops may not have ONNX equivalents for months.
        Cutting-edge ops (FlashAttention, RoPE, quantised kernels) often
        require custom ops or decomposition into primitives.

    4. PRECISION DRIFT:
        Numerical differences between frameworks accumulate.
        A model that passes onnx.checker may still have incorrect outputs.
        Always benchmark numerically against the source framework.

    5. LARGE MODELS:
        Standard .onnx files use protobuf, which has a 2 GB limit.
        Solution: use external_data_format=True to split weights:
            torch.onnx.export(..., export_params=True,
                              opset_version=17,
                              f="model.onnx")
            # Weights stored in separate .bin files


##### PART 8 — ONNX IN THE CONNECTED ML ECOSYSTEM

### ONNX as the Hub of the Inference Ecosystem

    ONNX is the de facto interchange format between:
        Training frameworks  →  ONNX  →  Inference runtimes/compilers

    Exporters (produce ONNX):
        PyTorch (torch.onnx / dynamo_export)
        TensorFlow (tf2onnx)
        JAX (jax2onnx / StableHLO bridge)
        MXNet (mx2onnx)
        PaddlePaddle (paddle2onnx)
        Scikit-learn (sklearn-onnx — also tree models!)
        XGBoost, LightGBM (onnxmltools)
        SpaCy, Transformers (via PyTorch)

    Consumers (import ONNX):
        ONNX Runtime         — cross-platform inference
        TensorRT             — NVIDIA maximum-performance inference
        Apache TVM           — auto-scheduled hardware compilation
        OpenVINO             — Intel hardware optimisation
        Core ML              — Apple hardware acceleration
        TFLite               — mobile deployment (via conversion)
        DirectML             — Windows GPU inference
        WebNN                — browser-native ML inference
        Qualcomm SNPE        — Snapdragon NPU inference
        XNNPACK              — ARM/x86 SIMD acceleration
        onnx.js / onnxruntime-web — browser inference

### ONNX and the Compiler Stack Modules

    ONNX connects to each compiler stack module:

    LLVM (module 09):
        ORT's CPUExecutionProvider uses MLAS (Microsoft's LLVM-based
        linear algebra library) for matmul and convolution.
        TVM compiles ONNX models → Relay → TIR → LLVM IR → native code.

    MLIR (module 10):
        onnx-mlir project converts ONNX → MLIR's ONNX dialect → LLVM dialect.
        ONNX ops are represented as MLIR operations (onnx.MatMul, onnx.Conv).
        Lowered through MLIR's progressive lowering to machine code.

    XLA (module 11):
        StableHLO and ONNX can be interconverted (stablehlo-onnx-bridge).
        JAX models exported as StableHLO can be converted to ONNX for ORT.
        TF models exported to ONNX can be compiled by JAX/XLA pipeline.

    TVM (module 12):
        relay.frontend.from_onnx() imports any ONNX model into Relay IR.
        TVM then applies auto-scheduling (Ansor) and compiles to any target.
        ONNX → TVM is the standard path for deploying to custom hardware.

### ONNX for LLMs: Microsoft's Olive and Optimum

    The modern LLM ecosystem uses ONNX extensively:

    HuggingFace Optimum:
        from optimum.onnxruntime import ORTModelForCausalLM
        model = ORTModelForCausalLM.from_pretrained("gpt2", export=True)
        # Exports GPT-2 to ONNX, runs with ORT, 1.3–2× faster than PyTorch

    Microsoft Olive:
        Olive = full pipeline: export → simplify → quantise → benchmark
        Supports: ORT, TensorRT, DirectML targets
        Used for: deploying transformers on Windows Copilot+ PCs

    ONNX for LLM inference limitations:
        - FlashAttention not expressible in standard ONNX ops
        - KV-cache management requires custom extensions
        - Most LLM serving (vLLM, SGLang, TRT-LLM) uses native PyTorch or
          custom runtimes rather than ONNX for maximum throughput
        - ONNX is better for edge/on-device LLM inference (< 7B params)


##### PART 9 — THE ONNX TYPE SYSTEM AND OPERATOR SCHEMAS

### What Operator Schemas Are

    Every ONNX operator is defined by a SCHEMA — a formal specification
    registered in the ONNX runtime and available in the Python package.
    The schema specifies:
        name:           operator name string ("MatMul", "Conv", …)
        domain:         "" for standard, "ai.onnx.ml" for classical ML
        since_version:  opset where this version was introduced
        doc:            human-readable description
        inputs:         list of (name, type_str, optional, variadic)
        outputs:        list of (name, type_str, optional)
        attributes:     list of (name, type, default, required)
        type_constraints: list of (type_var, allowed_types, description)

    Access schemas in Python:
        import onnx
        schema = onnx.defs.get_schema("MatMul", version=17)
        print(schema.doc)
        print(schema.type_constraints)

### The Type Constraint System

    ONNX does not say "MatMul takes float32". It says "MatMul takes T,
    where T must be one of float16, float32, float64, int32, int64, ..."
    This is the TYPE CONSTRAINT system — a T-variable system.

    HOW IT WORKS:
        Each operator declares one or more type variables (T, T1, T2, …).
        Each type variable is constrained to a set of allowed element types.
        The operator inputs/outputs reference these variables.
        If two inputs reference the same variable, they must have the same type.

    EXAMPLE — MatMul schema:
        type_constraints: [
            ("T", ["tensor(float16)", "tensor(float)", "tensor(double)",
                   "tensor(uint32)", "tensor(uint64)", "tensor(int32)",
                   "tensor(int64)", "tensor(bfloat16)"],
             "Constrain input and output types to float/int tensors.")
        ]
        inputs:   [("A", "T", …), ("B", "T", …)]
        outputs:  [("Y", "T", …)]

    WHAT THIS MEANS: both inputs to MatMul must have the SAME type.
    You cannot MatMul a float32 tensor with a float16 tensor —
    a Cast node must be inserted first.

    EXAMPLE — Gather schema (different types for data vs indices):
        type_constraints: [
            ("T",  [...all tensor types...], "input type"),
            ("Tind", ["tensor(int32)", "tensor(int64)"], "index type"),
        ]
        inputs: [("data", "T", …), ("indices", "Tind", …)]
        outputs: [("output", "T", …)]

    WHY TYPE CONSTRAINTS MATTER FOR EXPORT:
        When exporting a mixed-precision model, the exporter must insert
        Cast nodes wherever types mismatch at an op boundary.
        torch.onnx.export does this automatically for most cases.
        But custom ops need to declare their type constraints explicitly
        so the graph verifier can check for mismatches.

### Attribute Types

    NodeProto attributes carry compile-time (not runtime) configuration.
    Each attribute has a strict type:

        INT:    a single integer    (e.g., Conv.group=1)
        FLOAT:  a single float      (e.g., LeakyRelu.alpha=0.01)
        STRING: a single string     (e.g., Cast.to="float16")
        TENSOR: an embedded tensor  (e.g., Constant.value=dense[1,2,3])
        GRAPH:  a sub-graph         (e.g., If.then_branch, Loop.body)
        INTS:   list of integers    (e.g., Conv.kernel_shape=[3,3])
        FLOATS: list of floats      (e.g., LSTM.activations_alpha=[…])
        STRINGS:list of strings     (e.g., Cast target types list)
        SPARSE_TENSOR: sparse weight(e.g., SparseMatMul weights)
        GRAPHS: list of sub-graphs  (rarely used)

    GRAPH attributes enable CONTROL FLOW:
        If node has: "then_branch" (GRAPH) and "else_branch" (GRAPH).
        Loop node has: "body" (GRAPH) executed each iteration.
        Scan node has: "body" (GRAPH) applied across a sequence.
        Each sub-graph is a complete GraphProto embedded inside the node.

### Shape Inference vs Type Inference

    ONNX has two kinds of inference that propagate information forward:

    TYPE INFERENCE:
        Given input types, determine the output type of each node.
        Usually straightforward: if inputs are float32, output is float32.
        Type inference is EXACT — it never produces "unknown type."
        Runs automatically during onnx.checker.check_model().

    SHAPE INFERENCE:
        Given input shapes (if known), determine output shapes.
        Can be PARTIAL — some ops cannot infer output shape from input shape
        alone (e.g., NonZero, TopK with unknown K at inference time).
        Triggered by: onnx.shape_inference.infer_shapes(model)
        Required for: runtimes that need to pre-allocate buffers,
                      compilers that need loop bounds, Netron visualisation.

    What shape inference fills in:
        ALL intermediate ValueInfoProto objects (the "edges" in the DAG).
        Before inference: only graph inputs/outputs have shape info.
        After inference:  every intermediate tensor has a known shape
                          (or a partial shape with some dynamic dims).

    Shape inference is implemented per-operator via SHAPE INFERENCE FUNCTIONS
    registered in the ONNX library. Each function implements the mathematical
    relationship between input and output shapes.

    Example (Reshape shape inference):
        input "data" has shape [2, 3, 4]
        input "shape" is a constant tensor [6, 4]
        → output has shape [6, 4]
        The inference function reads the "shape" constant and produces [6, 4].

    When shape inference CANNOT determine a shape:
        The ValueInfoProto gets an empty/partial shape.
        ORT still runs (dynamic shape mode).
        TVM and TensorRT may refuse to compile without complete shapes.
        Fix: run infer_shapes() with data_prop=True to use initializer values.


##### PART 10 — FUNCTION OPERATORS: COMPOSITE AND DECOMPOSABLE OPS

### What Function Operators Are

    Starting with ONNX 1.9 (opset 12), operators can be defined as
    FUNCTIONS — sub-graphs expressed in terms of other ONNX ops.
    This serves two roles simultaneously:

    ROLE 1 — SEMANTICS DOCUMENTATION:
        The function body IS the specification of what the op computes.
        It is the reference implementation — a runtime that does not
        have a native kernel for the op can EXPAND the function body
        and execute it as a sequence of primitive ops.

    ROLE 2 — RUNTIME OPTIMISATION:
        A runtime THAT HAS a native kernel for the op (e.g., ORT has a
        fused CUDA kernel for LayerNorm) will use that instead of expanding.
        The function body is only a fallback.

    This design decouples the op's semantics from its implementation —
    exactly the same split TVM makes between computation and schedule.

### The LayerNormalization Function Op

    LayerNormalization was added in opset 17 as a function op.
    Its function body (the reference implementation in ONNX primitives):

        function LayerNormalization(X, Scale, B, *, axis, epsilon, stash_type):
            # Step 1: compute mean across the normalised dims
            Mean = ReduceMean(X, axes=normalized_axes, keepdims=True)

            # Step 2: compute variance  Var[X] = E[(X - mean)^2]
            D          = Sub(X, Mean)
            DD         = Mul(D, D)
            Var        = ReduceMean(DD, axes=normalized_axes, keepdims=True)

            # Step 3: compute inverse standard deviation
            VarEps     = Add(Var, Constant(epsilon))
            StdDev     = Sqrt(VarEps)
            InvStdDev  = Reciprocal(StdDev)

            # Step 4: normalise, scale, shift
            Normalized = Mul(D, InvStdDev)
            NormScaled = Mul(Normalized, Scale)
            Y          = Add(NormScaled, B)    ; + bias

            return Y, Mean, InvStdDev          ; 3 outputs

    The function body has 10 ONNX primitive ops.
    ORT replaces all 10 with a single fused CUDA kernel at level 2 optimisation.
    A runtime WITHOUT a LayerNorm kernel expands and runs all 10 ops.

### Other Function Ops (opset 17+)

    GroupNormalization (opset 18):
        Similar to LayerNorm but normalises within channel groups.
        Function body: Reshape → InstanceNormalization → Reshape → Scale + Shift

    BiasGelu (com.microsoft domain):
        Adds a bias then applies GELU activation.
        Function body: Add(X, Bias) → Gelu
        ORT's CUDA kernel fuses Add + Gelu in a single pass.

    RotaryEmbedding (com.microsoft):
        Rotary positional embeddings (RoPE) used in LLaMA, Mistral, Phi.
        Function body: complex slice/gather/multiply/concatenate sequence.
        ORT's GenAI runtime has a fused CUDA kernel.

### Viewing Function Bodies

    In Python, function bodies are stored as FunctionProto objects:

        model = onnx.load("model.onnx")
        for func in model.functions:
            print(f"Function: {func.name} (domain: {func.domain})")
            print(f"  Nodes: {len(func.node)}")
            for node in func.node[:3]:
                print(f"    {node.op_type}: {list(node.input)} → {list(node.output)}")

    When you call shape_inference on a model with functions:
        The inferrer either:
          a) Uses the op's registered shape inference function (preferred), OR
          b) Inlines the function body and infers shapes through it (fallback)

### Inlining Function Ops

    To get a graph with NO function ops (pure primitives), inline them:

        from onnx.inliner import inline_local_functions
        model_inlined = inline_local_functions(model)
        # Now all LayerNorm, Gelu, etc. are expanded to their primitive ops.
        # Useful for: runtimes without function op support,
        #             debugging, manual graph editing.


##### PART 11 — ONNX-ML: CLASSICAL MACHINE LEARNING EXTENSION

### What ONNX-ML Is

    The standard ONNX spec covers neural network operators.
    ONNX-ML is a SEPARATE DOMAIN ("ai.onnx.ml") that extends ONNX with
    operators for classical machine learning models:
        Decision trees and tree ensembles
        SVMs (Support Vector Machines)
        Linear models (logistic regression, linear regression)
        Normalisation and preprocessing (scaler, binariser, imputer)
        Label encoders and one-hot encoders
        Full sklearn Pipeline representation

    This makes ONNX the single format for BOTH deep learning AND
    classical ML — you can serve a random forest and a BERT model
    from the same ONNX Runtime deployment.

### ONNX-ML Operators

    TreeEnsembleClassifier:
        Represents a scikit-learn RandomForestClassifier, GradientBoostingClassifier,
        XGBoost Classifier, LightGBM Classifier in a single ONNX node.
        Attributes:
            n_targets:         number of output classes
            nodes_truenodeids: for each node, which child if condition True
            nodes_falsenodeids: which child if condition False
            nodes_featureids:  which input feature to test at each node
            nodes_hitrates:    normalisation factor per node
            nodes_missing_value_tracks_true: handle NaN inputs
            nodes_nodeids:     unique node ID per tree
            nodes_treeids:     which tree each node belongs to
            nodes_modes:       decision type (LEQ, LT, GTE, GT, EQ, NEQ, MEMBER)
            nodes_values:      threshold value to compare against
            post_transform:    output transformation (NONE, SOFTMAX, LOGISTIC, ...)
            classlabels_int64s: the class label for each output class

    TreeEnsembleRegressor:
        Same structure but for regression tasks.
        Aggregation mode: AVERAGE, SUM, MIN, MAX across trees.

    SVMClassifier / SVMRegressor:
        Stores the support vectors, coefficients, and kernel parameters.
        Supports: Linear, Poly, RBF, Sigmoid kernels.
        Post-transform: Logistic, Softmax, or None.

    LinearClassifier:
        Stores weights and biases for linear/logistic regression.
        Multi-class: stores W matrix and b vector.
        Post-transform: same as SVM (softmax, sigmoid, etc.).

    Normalizer:
        Applies L1, L2, or Max normalisation to feature rows.
        Equivalent to sklearn.preprocessing.Normalizer.

    Binarizer:
        Thresholds numerical features to binary (0/1).

    Imputer:
        Replaces missing values (NaN) with mean, median, or constant.

    LabelEncoder:
        Maps string labels ↔ integer indices.
        Essential for models whose output is a class name string.

    ZipMap:
        Converts parallel arrays (class_labels, probabilities) into a
        sequence of maps {label: probability} — the standard ORT output
        format for classifiers.

### Exporting Classical ML Models with sklearn-onnx

    scikit-learn → ONNX via the skl2onnx package:

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Train a pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100)),
        ])
        pipe.fit(X_train, y_train)

        # Convert to ONNX
        initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
        onnx_model   = convert_sklearn(pipe, initial_types=initial_type,
                                        target_opset={"": 17, "ai.onnx.ml": 3})
        with open("classifier.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

    The converted model uses BOTH standard ONNX ops (for the scaler)
    and ONNX-ML ops (for the tree ensemble).

    Running in ORT — identical to neural network models:
        sess    = ort.InferenceSession("classifier.onnx")
        outputs = sess.run(None, {"float_input": X_test.astype(np.float32)})
        labels  = outputs[0]           # class labels
        probas  = outputs[1]           # list of {label: prob} dicts

### XGBoost and LightGBM → ONNX

    XGBoost:
        from onnxmltools import convert_xgboost
        from onnxconverter_common.data_types import FloatTensorType

        xgb_model   = xgboost.train(params, dtrain)
        onnx_model   = convert_xgboost(xgb_model,
                                        initial_types=[("input", FloatTensorType([None, n_features]))])

    LightGBM:
        from onnxmltools import convert_lightgbm
        lgbm_onnx = convert_lightgbm(lgbm_model,
                                      initial_types=[("input", FloatTensorType([None, n_features]))])

    Both produce TreeEnsembleClassifier nodes in the "ai.onnx.ml" domain.
    ORT runs these natively via its tree ensemble kernel, which is often
    FASTER than the native XGBoost/LGBM Python inference because:
        - No Python object overhead per tree traversal
        - SIMD-vectorised tree traversal (8 or 16 samples in parallel)
        - Better cache locality from compact array representation

### ONNX-ML and Deployment

    ONNX-ML enables an important deployment pattern:

        SINGLE SERVING ENDPOINT for mixed model types:
            One ORT session running a pipeline that includes:
                ONNX-ML preprocessing (StandardScaler, LabelEncoder)
                + standard ONNX neural network
                + ONNX-ML post-processing (probability calibration)

            This "all in one graph" approach eliminates Python glue code
            between preprocessing and the neural network at inference time,
            reducing latency and simplifying deployment.


##### PART 12 — ONNXSCRIPT: AUTHORING ONNX IN PYTHON

### What ONNXScript Is

    ONNXScript is Microsoft's Python-embedded DSL for writing ONNX
    operators, functions, and models. It replaced the raw protobuf API
    as the canonical way to:
        - Implement new ONNX operators from scratch
        - Write custom ops that torch.onnx.dynamo_export can use
        - Define ONNX function bodies that serve as reference implementations
        - Author new opset specifications (the ONNX spec itself is written
          in ONNXScript internally)

    Install:  pip install onnxscript

### Writing an ONNX Function in ONNXScript

    Compare: the same Selu activation function written two ways.

    RAW PROTOBUF (old way):
        selu_node = helper.make_node(
            "Selu", inputs=["x"], outputs=["y"],
            alpha=1.6732631, gamma=1.0507009)
        # To DEFINE selu in terms of primitives, you'd need to manually
        # construct an entire FunctionProto with 10+ protobuf nodes.

    ONNXSCRIPT (new way):
        import onnxscript
        from onnxscript import opset17 as op
        from onnxscript import FLOAT

        @onnxscript.script()
        def selu(x: FLOAT["N"]) -> FLOAT["N"]:
            alpha = op.Constant(value_float=1.6732631)
            gamma = op.Constant(value_float=1.0507009)
            pos   = op.Relu(x)
            neg   = op.Sub(x, pos)                   # x - relu(x) = min(x, 0)
            exp_n = op.Exp(neg)                       # exp(min(x, 0))
            scaled_neg = op.Mul(alpha, op.Sub(exp_n, op.Constant(value_float=1.0)))
            result= op.Mul(gamma, op.Add(pos, scaled_neg))
            return result

        # Convert to ONNX FunctionProto:
        onnx_selu = selu.to_model_proto()

    The @onnxscript.script() decorator:
        Analyses the function body using Python's AST.
        Maps Python operators (+, *, >) to ONNX ops.
        Maps op.* calls to ONNX operator nodes.
        Produces an ONNX FunctionProto or ModelProto.

### ONNXScript Type Annotations

    ONNXScript uses type annotations to declare tensor types:

        from onnxscript import FLOAT, INT64, BOOL, DOUBLE

        FLOAT["N", "M"]     ; 2D float tensor with symbolic dims N, M
        FLOAT[4, 8]         ; 2D float tensor with static shape [4,8]
        FLOAT[...]          ; float tensor of any rank
        INT64["N"]          ; 1D int64 tensor

    More complex types:
        from onnxscript.onnx_types import FLOAT16, BFLOAT16
        from onnxscript import script, opset18 as op

        @script()
        def attention_mask(x: FLOAT["B", "S", "D"],
                            mask: BOOL["B", "S"]) -> FLOAT["B", "S", "D"]:
            mask_f = op.Cast(mask, to=FLOAT.dtype)
            mask_bc = op.Unsqueeze(mask_f, axes=[-1])
            return op.Mul(x, mask_bc)

### Using ONNXScript with torch.onnx.dynamo_export

    The primary use case: registering custom op lowerings for the
    TorchDynamo-based ONNX exporter.

        import onnxscript
        from onnxscript import opset17 as op, FLOAT
        import torch

        # 1. Define the ONNX implementation in ONNXScript
        @onnxscript.script()
        def silu_onnx(x: FLOAT["N"]) -> FLOAT["N"]:
            sig = op.Sigmoid(x)
            return op.Mul(x, sig)

        # 2. Register it as the ONNX lowering for torch's silu
        @torch.onnx.symbolic("aten::silu", opset=17)
        def silu_symbolic(
                g: torch.onnx.ExportContext,
                x: torch.Value) -> torch.Value:
            return g.op("Silu", x)   ; now uses the ONNXScript definition

        # 3. Export — torch will use silu_onnx for all silu ops
        export_out = torch.onnx.dynamo_export(model, sample_input)
        export_out.save("model.onnx")

### ONNXScript for Opset Authoring

    The ONNX specification itself uses ONNXScript to define operator
    reference implementations (since ONNX 1.14). This means the spec is
    EXECUTABLE — you can run any ONNX operator through its reference
    implementation directly in Python.

        from onnx.reference import ReferenceEvaluator
        evaluator = ReferenceEvaluator("model.onnx")
        outputs   = evaluator.run(None, {"input": x_numpy})
        # Runs every op through its ONNXScript reference implementation.
        # Useful for: debugging precision issues, testing custom ops,
        #             validating that a new op implementation is correct.


##### PART 13 — ONNXRUNTIME-TRAINING AND FINE-TUNING WITH ORT

### What onnxruntime-training Is

    onnxruntime-training is a Microsoft extension of ORT that supports
    TRAINING and FINE-TUNING, not just inference. It adds:
        - A gradient graph builder (computes dL/dW for each parameter)
        - Optimisers (Adam, Adagrad, SGD with momentum) implemented in C++
        - Mixed-precision training (FP16/BF16 forward, FP32 master weights)
        - Gradient accumulation
        - Zero redundancy optimiser (ZeRO stage 1)

    Install:  pip install onnxruntime-training

    Use cases:
        Fine-tuning large pre-trained models on edge devices or
        resource-constrained servers where PyTorch is not available.
        Training in environments where the ONNX Runtime is already
        deployed for inference (add fine-tuning without adding PyTorch).
        ONNX-native training pipelines for reproducibility/certification.

### The Training API

    onnxruntime.training.api.Module wraps an ONNX model for training:

        from onnxruntime.training.api import Module, Optimizer

        # Export the model to ONNX for training (includes gradient graph)
        # Step 1: define training artifacts
        from onnxruntime.training.ortmodule import ORTModule
        import torch

        class MyModel(torch.nn.Module):
            def __init__(self): ...
            def forward(self, x): ...

        model     = MyModel()
        ort_model = ORTModule(model)   ; wraps the model for ORT-accelerated training

        # Training loop (same as PyTorch):
        optimizer = torch.optim.Adam(ort_model.parameters(), lr=1e-4)
        for batch in dataloader:
            output = ort_model(batch["input"])
            loss   = criterion(output, batch["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    ORTModule intercepts the forward pass:
        Traces the PyTorch model → ONNX graph.
        Builds a backward (gradient) graph automatically.
        Runs both graphs through ORT for accelerated execution.
        Falls back to PyTorch for unsupported ops transparently.

### Export-Based Training (ort.training.api)

    For completely PyTorch-free training:

        from onnxruntime.training.api import CheckpointState, Module, Optimizer

        # Export training artifacts (done once):
        from onnxruntime.training.artifacts import generate_artifacts, LossType

        generate_artifacts(
            model      = onnx.load("model.onnx"),
            loss       = LossType.CrossEntropyLoss,
            optimizer  = "Adam",
            artifact_directory = "./training_artifacts",
        )
        # Creates:
        #   training_model.onnx   — forward + loss + backward graph
        #   eval_model.onnx       — forward only (for validation)
        #   optimizer_model.onnx  — Adam update step
        #   checkpoint            — initial parameter state

        # Training (no PyTorch):
        state     = CheckpointState.load_checkpoint("./training_artifacts/checkpoint")
        module    = Module("training_model.onnx", state, "eval_model.onnx",
                            device="cuda")
        optimizer = Optimizer("optimizer_model.onnx", module)

        for X, y in dataloader:
            loss = module(X, y)         ; forward + backward
            optimizer.step()            ; Adam update
            optimizer.zero_grad()       ; clear gradients

    This entire loop runs in C++ via ORT — no Python tensor operations,
    no PyTorch, no framework overhead per batch.

### Mixed Precision Training with ORT

    ORT supports AUTOMATIC MIXED PRECISION (AMP) for training:

        from onnxruntime.training.ortmodule.experimental import _mix_precision
        from onnxruntime.training.ortmodule import ORTModule

        model   = ORTModule(MyModel())
        # Enable AMP:
        model.train()
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)

    ORT-specific AMP benefits:
        Parameters stored in FP32 (master weights).
        Forward/backward in FP16 (2× less memory, faster Tensor Cores).
        Gradient scaling to prevent underflow.
        Loss scaling applied automatically.
        Typical memory saving: 40-50% vs FP32 training.
        Typical speedup: 1.5-2× on NVIDIA Ampere and newer.


##### PART 14 — ORT GENAI: LLM INFERENCE WITH ONNX

### The LLM Inference Problem in ONNX

    Standard ONNX inference is STATELESS — you feed inputs and get outputs.
    Large language model inference is STATEFUL — the KV-cache from previous
    token generation steps must be preserved between calls.

    This creates a mismatch:
        Standard ONNX session: sess.run(None, {"input_ids": tokens})
        LLM inference needs:   decode_step(token, prev_kv_cache) → (next_token, new_kv_cache)

    The standard workaround (model with KV-cache as explicit I/O):
        Export the model with past_key_values as explicit graph inputs/outputs.
        Each decode step: output the new KV-cache and feed it back next step.
        Problem: KV-cache copying between Python and ORT adds latency.
                 For a 7B model with 32 layers: ~4 GB of KV tensors to copy!

### ORT GenAI (onnxruntime-genai)

    ORT GenAI is a dedicated package for LLM inference that handles
    KV-cache management natively inside the runtime — no Python copying.

    Install:  pip install onnxruntime-genai

    Key features:
        KV-CACHE MANAGED IN C++: stays on GPU, never copied to Python.
        BATCHED GENERATION: multiple sequences generated in parallel.
        GREEDY / BEAM SEARCH: decoding strategies implemented in C++.
        INT4/INT8/FP16: quantised model loading.
        MULTIPLE BACKENDS: CUDA, CPU, DirectML, MPS (Apple Silicon).

    Usage:
        import onnxruntime_genai as og

        model     = og.Model("./phi-2-onnx/")
        tokenizer = og.Tokenizer(model)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=200, temperature=0.8)
        params.input_ids = tokenizer.encode("Tell me a joke: ")

        generator = og.Generator(model, params)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

        output = tokenizer.decode(generator.get_sequence(0))

    SUPPORTED MODELS (via Microsoft's Optimum-ORT export):
        Phi-2, Phi-3 (Microsoft)
        Gemma (Google)
        LLaMA 2/3
        Mistral / Mixtral
        Falcon
        Qwen

### Exporting LLMs to ONNX for GenAI

    Microsoft's Olive or Optimum handles the export:

        # Via Optimum (recommended):
        from optimum.exporters.onnx import main_export

        main_export(
            "meta-llama/Llama-2-7b-hf",
            output="./llama2-onnx/",
            task="text-generation-with-past",   ; includes KV-cache I/O
            opset=17,
        )

        # Via Olive (adds quantisation):
        # olive run --config llama2_ort_genai.json
        # Config specifies: export → quantise to INT4 → benchmark

    The "with-past" model variant:
        The export generates TWO models:
            decoder_model.onnx:           processes the full prompt (prefill)
            decoder_with_past_model.onnx: single token decode with KV cache

        ORT GenAI automatically uses both: prefill model for the first call,
        then the with-past model for each subsequent token generation step.

### INT4 Quantisation for LLMs in ONNX

    ORT GenAI supports INT4 weight-only quantisation (GPTQ-style):

        from onnxruntime.quantization import MatMulNBits, quantize_static

        # INT4 with group quantisation (group_size=32 or 128):
        model_int4 = MatMulNBits.quantize(
            model=onnx_model,
            block_size=32,      ; quantise in groups of 32 weights
            bits=4,             ; 4-bit storage
            is_symmetric=True,  ; symmetric around 0 (no zero-point)
        )

    Memory comparison for Llama-2-7B:
        FP16:  13 GB VRAM
        INT8:   7 GB VRAM  (2× compression)
        INT4:   4 GB VRAM  (3.5× compression — fits on consumer GPU!)

    Quality: INT4 typically loses <1% accuracy on benchmarks like MMLU
             when using groupwise quantisation (group_size=32 or 128).

### The Complete ONNX LLM Deployment Stack

    ┌──────────────────────────────────────────────────────────────────┐
    │ TRAINING       PyTorch + HuggingFace Transformers                │
    │                (Llama, Phi, Mistral, Gemma, …)                   │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │ Optimum export / Olive pipeline
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ EXPORT         decoder.onnx + decoder_with_past.onnx             │
    │                + INT4 quantisation (optional)                    │
    │                + genai_config.json (model metadata)              │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │ model directory
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ SERVING        onnxruntime-genai                                 │
    │                KV-cache managed in C++ (no Python copies)        │
    │                Runs on: CUDA / CPU / DirectML / Apple MPS        │
    └──────────────────────────────────────────────────────────────────┘

    This stack runs on:
        NVIDIA GPU:   CUDA execution provider
        AMD GPU:      ROCm execution provider
        Apple M1/M2:  CoreML + Metal execution provider
        Intel GPU:    DirectML on Windows
        CPU only:     CPUExecutionProvider (slow but portable)
        Edge/NPU:     QNN execution provider (Qualcomm Snapdragon)

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · ONNX Export Pipeline — PyTorch to ONNX and Verification": {
        "description": (
            "End-to-end ONNX export from PyTorch with full validation. "
            "Export a CNN and a transformer encoder. Show static vs dynamic shapes. "
            "Inspect the protobuf graph structure (nodes, initializers, value_info). "
            "Run shape inference. Validate numerically against the original model. "
            "Demonstrate common export pitfalls and their fixes."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  ONNX EXPORT PIPELINE — PyTorch TO ONNX AND VERIFICATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Basic export and graph inspection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Export, verify, and inspect an ONNX graph")
print("━" * 65)
print()

try:
    import torch
    import torch.nn as nn
    import onnx
    import onnx.numpy_helper as nph
    from onnx import shape_inference
    import io

    print(f"  PyTorch {torch.__version__}, ONNX {onnx.__version__}")
    print()

    # ── Model definition ───────────────────────────────────────────────────
    class ConvBNRelu(nn.Module):
        """
        Simple CNN block: Conv2d → BatchNorm → ReLU
        BatchNorm is the interesting case: 4 parameter tensors in ONNX
        (weight, bias, running_mean, running_var)
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 8, kernel_size=1, bias=True),
                nn.Flatten(),
                nn.Linear(8 * 8 * 8, 10),
            )
        def forward(self, x):
            return self.net(x)

    model = ConvBNRelu().eval()   # CRITICAL: eval() mode!
    dummy = torch.randn(1, 3, 8, 8)
    print(f"  Model: ConvBNRelu ({sum(p.numel() for p in model.parameters()):,} params)")
    print()

    # ── Export to ONNX ─────────────────────────────────────────────────────
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (dummy,),
        buf,
        opset_version=17,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images":  {0: "batch_size"},   # dynamic batch
            "logits":  {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    buf.seek(0)
    onnx_model = onnx.load(buf)

    # ── Validation ─────────────────────────────────────────────────────────
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validation: ✅ PASSED")

    # Run shape inference (fills in value_info for all intermediate tensors)
    onnx_model_inferred = shape_inference.infer_shapes(onnx_model)
    print(f"  Shape inference: ✅ DONE")
    print()

    # ── Graph structure inspection ────────────────────────────────────────
    graph = onnx_model_inferred.graph
    print(f"  Graph summary:")
    print(f"    IR version:    {onnx_model.ir_version}")
    print(f"    Opset version: {onnx_model.opset_import[0].version}")
    print(f"    Nodes:         {len(graph.node)}")
    print(f"    Initializers:  {len(graph.initializer)}  (weight tensors)")
    print(f"    Inputs:        {[i.name for i in graph.input]}")
    print(f"    Outputs:       {[o.name for o in graph.output]}")
    print()

    # Print all nodes
    print(f"  Node-by-node breakdown:")
    print(f"  {'#':>4} | {'Op type':>20} | {'Inputs':>35} | {'Outputs'}")
    print(f"  {'─'*90}")
    for i, node in enumerate(graph.node):
        inputs  = ', '.join(node.input[:3])   # show first 3 inputs
        outputs = ', '.join(node.output)
        if len(node.input) > 3:
            inputs += f" (+{len(node.input)-3} more)"
        print(f"  {i:4d} | {node.op_type:>20} | {inputs:>35} | {outputs}")
    print()

    # Print initializer (weight) shapes
    print(f"  Initializers (trained weights):")
    total_params = 0
    for init in graph.initializer:
        arr    = nph.to_array(init)
        total_params += arr.size
        print(f"    {init.name:<45} {str(arr.shape):<20} {arr.dtype}")
    print(f"    Total: {total_params:,} parameters ({total_params*4/1e6:.2f} MB in float32)")
    print()

    # Print value_info for intermediate tensors
    print(f"  Intermediate tensor shapes (after shape inference):")
    for vi in list(graph.value_info)[:6]:   # first 6
        t = vi.type.tensor_type
        shape = [d.dim_value if d.dim_value else d.dim_param or "?"
                 for d in t.shape.dim]
        print(f"    {vi.name:<45} {str(shape):<20} dtype={t.elem_type}")
    if len(graph.value_info) > 6:
        print(f"    ... ({len(graph.value_info)-6} more)")

except ImportError as e:
    print(f"  Missing dependency: {e}")
    print("  Install: pip install torch onnx")
    print()

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Numerical validation and dynamic shapes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Numerical validation and dynamic batch testing")
print("━" * 65)
print()

try:
    import onnxruntime as ort
    print(f"  ONNX Runtime {ort.__version__}")
    print()

    # Run through ONNX Runtime and compare with PyTorch
    sess     = ort.InferenceSession(buf.getvalue(),
                 providers=["CPUExecutionProvider"])

    # Test multiple batch sizes (validates dynamic shape works)
    print(f"  Numerical comparison (PyTorch vs ORT) across batch sizes:")
    print(f"  {'Batch':>6} | {'Max abs diff':>14} | {'Mean abs diff':>15} | {'Pass?':>7}")
    print(f"  {'─'*50}")

    model.eval()
    for batch_size in [1, 2, 4, 8]:
        x_np = np.random.randn(batch_size, 3, 8, 8).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        # PyTorch reference
        with torch.no_grad():
            pt_out = model(x_pt).numpy()

        # ORT inference
        ort_out = sess.run(None, {"images": x_np})[0]

        max_diff  = np.max(np.abs(pt_out - ort_out))
        mean_diff = np.mean(np.abs(pt_out - ort_out))
        passed    = max_diff < 1e-4

        print(f"  {batch_size:6d} | {max_diff:14.2e} | {mean_diff:15.2e} | "
              f"{'✅' if passed else '❌':>7}")
    print()

    # Benchmark ORT vs PyTorch
    BATCH  = 16
    x_np_b = np.random.randn(BATCH, 3, 8, 8).astype(np.float32)
    x_pt_b = torch.from_numpy(x_np_b)
    REPS   = 500

    # PyTorch eager
    with torch.no_grad():
        for _ in range(10): model(x_pt_b)   # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(REPS): model(x_pt_b)
    t_pt = (time.perf_counter() - t0) / REPS * 1000

    # ORT
    for _ in range(10): sess.run(None, {"images": x_np_b})   # warmup
    t0 = time.perf_counter()
    for _ in range(REPS): sess.run(None, {"images": x_np_b})
    t_ort = (time.perf_counter() - t0) / REPS * 1000

    print(f"  Inference benchmark (batch={BATCH}):")
    print(f"    PyTorch eager:        {t_pt:.4f} ms")
    print(f"    ONNX Runtime (CPU):   {t_ort:.4f} ms")
    print(f"    Speedup:              {t_pt/t_ort:.2f}×")
    print()
    print("  ORT is often faster than PyTorch eager because:")
    print("    - BatchNorm fused into Conv at model load time")
    print("    - No Python overhead per operation")
    print("    - MLAS (ORT's BLAS) tuned for the specific CPU")

except ImportError:
    print("  onnxruntime not installed: pip install onnxruntime")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Common pitfalls — before/after
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 3 — Common export pitfalls and fixes")
print("━" * 65)
print()

PITFALLS = """
  PITFALL 1: model.train() mode during export
  ────────────────────────────────────────────────────────────────────────
  WRONG:
      model = MyModel()                           # default: training mode
      torch.onnx.export(model, dummy, "model.onnx")
      # → Dropout IS included in the graph (randomly zeros outputs!)
      # → BatchNorm uses batch statistics (not running stats)

  CORRECT:
      model = MyModel().eval()                    # switch to eval mode
      torch.onnx.export(model, dummy, "model.onnx")
      # → Dropout is identity (pass-through)
      # → BatchNorm uses saved running_mean / running_var

  PITFALL 2: Data-dependent shapes
  ────────────────────────────────────────────────────────────────────────
  WRONG:
      def forward(self, x, mask):
          valid = x[mask]        # shape depends on mask values → FAILS
          return valid.sum()

  CORRECT (use masking instead of indexing):
      def forward(self, x, mask):
          masked = x * mask.float()   # same shape as x
          return masked.sum()

  PITFALL 3: Python int from tensor shapes
  ────────────────────────────────────────────────────────────────────────
  WRONG:
      b, n, d = x.shape
      y = x.view(b * n, d)      # b*n is a Python int (baked into graph!)

  CORRECT:
      y = x.reshape(-1, x.shape[-1])   # dynamic, works for any batch

  PITFALL 4: in-place operations
  ────────────────────────────────────────────────────────────────────────
  WRONG:
      x += 1        # in-place — may confuse ONNX tracer
      x[0] = 0.0   # in-place indexing — not supported

  CORRECT:
      x = x + 1
      mask = torch.ones_like(x); mask[0] = 0.0; x = x * mask

  PITFALL 5: Wrong opset for runtime
  ────────────────────────────────────────────────────────────────────────
  CHECK: What opset does your target runtime support?
    ONNX Runtime:    opset 1–18 (all stable opsets)
    TensorRT:        opset 1–17 (check TRT version)
    OpenVINO:        opset 1–15 (some ops may be missing)
    Core ML tools:   opset 1–13 for most ops
    TFLite (via):    opset 1–13 typically

  FIX: Use the LOWEST opset that has all ops you need.
       Test with: onnxruntime.InferenceSession("model.onnx") # raises on unsupported ops
"""
print(PITFALLS)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · ONNX Graph Manipulation — Inspect, Modify and Optimise": {
        "description": (
            "Build ONNX graphs from scratch using the Python API. "
            "Inspect and traverse a real model's graph structure. "
            "Modify the graph: insert, delete, replace nodes. "
            "Run ONNX graph optimisations (simplification, constant folding). "
            "Quantise a model to INT8 with dynamic and static quantisation. "
            "Visualise graph structure programmatically."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  ONNX GRAPH MANIPULATION — INSPECT, MODIFY AND OPTIMISE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Build an ONNX graph from scratch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Building an ONNX graph with the Python API")
print("━" * 65)
print()

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper, shape_inference
    import numpy as np

    print("  Building: relu(x) @ W  (elementwise relu then matmul)")
    print()

    # ── Define graph components ────────────────────────────────────────────

    # 1. Node: Relu on input x
    relu_node = helper.make_node(
        "Relu",
        inputs=["x"],
        outputs=["relu_x"],
        name="relu_0",
    )

    # 2. Node: MatMul(relu_x, W) → y
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["relu_x", "W"],
        outputs=["y"],
        name="matmul_0",
    )

    # 3. Type/shape descriptors for graph edges
    x_info = helper.make_tensor_value_info("x",     TensorProto.FLOAT, [None, 8])
    W_info = helper.make_tensor_value_info("W",     TensorProto.FLOAT, [8, 4])
    y_info = helper.make_tensor_value_info("y",     TensorProto.FLOAT, [None, 4])

    # Intermediate edge (after relu) — optional but good practice
    relu_x_info = helper.make_tensor_value_info("relu_x", TensorProto.FLOAT, [None, 8])

    # 4. Weight initializer (trained weight W)
    W_data = np.random.randn(8, 4).astype(np.float32)
    W_init = numpy_helper.from_array(W_data, name="W")

    # 5. Build the graph
    graph = helper.make_graph(
        nodes       = [relu_node, matmul_node],
        name        = "relu_matmul_graph",
        inputs      = [x_info],         # W is an initializer, not an input
        outputs     = [y_info],
        initializer = [W_init],
        value_info  = [relu_x_info],    # intermediate shapes
    )

    # 6. Build the model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.doc_string = "Simple relu(x) @ W example"

    # 7. Validate and run shape inference
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    print("  Model validated ✅")
    print()

    # 8. Print the graph
    print("  Graph structure:")
    for node in model.graph.node:
        attrs = {a.name: a for a in node.attribute}
        print(f"    [{node.name}] {node.op_type}:")
        print(f"      inputs:  {list(node.input)}")
        print(f"      outputs: {list(node.output)}")
    print()

    # 9. Run with ORT
    import onnxruntime as ort
    import io
    buf = io.BytesIO()
    onnx.save(model, buf)
    buf.seek(0)

    sess   = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    x_test = np.random.randn(4, 8).astype(np.float32)
    out    = sess.run(None, {"x": x_test})[0]

    # Verify: relu(x) @ W
    expected = np.maximum(x_test, 0) @ W_data
    match    = np.allclose(out, expected, atol=1e-5)
    print(f"  ORT output shape: {out.shape}")
    print(f"  Matches numpy reference: {match} ✅")
    print()

except ImportError as e:
    print(f"  Missing: {e}  →  pip install onnx onnxruntime")
    print()

    BUILDER_REF = """
  ONNX GRAPH BUILDER REFERENCE:

  from onnx import helper, TensorProto, numpy_helper

  # --- Nodes (operations) ---
  conv_node = helper.make_node(
      "Conv",
      inputs=["x", "W", "B"],   # input, weight, bias
      outputs=["y"],
      kernel_shape=[3, 3],       # attributes go here as kwargs
      pads=[1, 1, 1, 1],
      strides=[1, 1],
      group=1,
  )

  # --- Type info (shapes for inputs/outputs) ---
  x_info = helper.make_tensor_value_info(
      "x", TensorProto.FLOAT, [None, 3, 224, 224])  # None = dynamic

  # --- Initializer (weights, constants) ---
  W = numpy_helper.from_array(np.random.randn(64, 3, 3, 3).astype(np.float32),
                               name="W")

  # --- Assemble ---
  graph = helper.make_graph(
      [conv_node],          # list of nodes (in topological order)
      "my_graph",
      inputs=[x_info],      # graph inputs (not including initializers)
      outputs=[y_info],
      initializer=[W],      # trained weights
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
  onnx.save(model, "model.onnx")
"""
    print(BUILDER_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Graph inspection and modification
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Graph inspection and node-level modification")
print("━" * 65)
print()

try:
    import torch, torch.nn as nn, io

    # Export a simple model
    class TwoBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 8)
        def forward(self, x):
            return torch.relu(self.fc2(torch.relu(self.fc1(x))))

    m   = TwoBlock().eval()
    buf = io.BytesIO()
    torch.onnx.export(m, torch.randn(1, 16), buf, opset_version=17,
                       input_names=["x"], output_names=["y"],
                       dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})
    buf.seek(0)
    model = onnx.load(buf)
    model = shape_inference.infer_shapes(model)

    print(f"  Model: TwoBlock (Linear→ReLU→Linear→ReLU)")
    print(f"  Nodes ({len(model.graph.node)}):")
    for i, n in enumerate(model.graph.node):
        print(f"    [{i}] {n.op_type:20s} {list(n.input)[:2]} → {list(n.output)}")
    print()

    # ── Count op types ────────────────────────────────────────────────────
    from collections import Counter
    op_counts = Counter(n.op_type for n in model.graph.node)
    print("  Op type distribution:")
    for op, count in op_counts.most_common():
        print(f"    {op:<20} × {count}")
    print()

    # ── Find all Relu nodes ────────────────────────────────────────────────
    relu_nodes = [n for n in model.graph.node if n.op_type == "Relu"]
    print(f"  Found {len(relu_nodes)} Relu node(s):")
    for rn in relu_nodes:
        print(f"    {rn.name if rn.name else '(unnamed)'}: {rn.input} → {rn.output}")
    print()

    # ── Replace Relu with Gelu ────────────────────────────────────────────
    import copy
    modified = copy.deepcopy(model)
    for node in modified.graph.node:
        if node.op_type == "Relu":
            node.op_type = "Gelu"   # simple op type swap
    print("  Modified: replaced all Relu → Gelu")
    onnx.checker.check_model(modified)
    print("  Validation after modification: ✅")
    print()

    # ── Verify original model numerics ────────────────────────────────────
    buf_orig = io.BytesIO()
    onnx.save(model, buf_orig)
    buf_orig.seek(0)
    sess_orig = ort.InferenceSession(buf_orig.getvalue(),
                                      providers=["CPUExecutionProvider"])
    x_test = np.random.randn(2, 16).astype(np.float32)
    out_orig = sess_orig.run(None, {"x": x_test})[0]
    print(f"  Original model output shape: {out_orig.shape}")
    print(f"  Original output[:2]: {out_orig[0, :4].round(4)}")

    buf_mod = io.BytesIO()
    onnx.save(modified, buf_mod)
    buf_mod.seek(0)
    sess_mod = ort.InferenceSession(buf_mod.getvalue(),
                                     providers=["CPUExecutionProvider"])
    out_mod = sess_mod.run(None, {"x": x_test})[0]
    print(f"  Modified model (Gelu) output[:2]: {out_mod[0, :4].round(4)}")
    print(f"  Outputs differ (expected — Gelu ≠ Relu): {not np.allclose(out_orig, out_mod)}")

except ImportError as e:
    print(f"  Missing: {e}")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: ONNX model simplification and quantisation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Graph optimisation and INT8 quantisation")
print("━" * 65)
print()

try:
    import onnxsim
    print(f"  onnx-simplifier available ✅")
    print()

    buf.seek(0)
    model_orig = onnx.load(buf)
    n_nodes_before = len(model_orig.graph.node)
    n_inits_before = len(model_orig.graph.initializer)

    model_simplified, ok = onnxsim.simplify(model_orig)
    n_nodes_after  = len(model_simplified.graph.node)
    n_inits_after  = len(model_simplified.graph.initializer)

    print(f"  onnx-simplifier results:")
    print(f"    Nodes:        {n_nodes_before} → {n_nodes_after}  (removed {n_nodes_before-n_nodes_after})")
    print(f"    Initializers: {n_inits_before} → {n_inits_after}  (folded {n_inits_before-n_inits_after})")
    print(f"    Check passed: {ok}")
    print()
    print("  onnx-simplifier removes:")
    print("    - Identity nodes (pass-through ops)")
    print("    - Redundant Cast nodes")
    print("    - Constant Reshape nodes (shape known at simplify time)")
    print("    - Fused BatchNorm into Conv (BN absorbed into conv weights)")

except ImportError:
    print("  onnx-simplifier not installed: pip install onnxsim")
    print()

print()
print("  DYNAMIC QUANTISATION REFERENCE:")
INT8_REF = """
  from onnxruntime.quantization import (
      quantize_dynamic, quantize_static,
      CalibrationDataReader, QuantType
  )

  # ── Dynamic quantisation (weights INT8, activations FP32 runtime) ──────
  quantize_dynamic(
      "model.onnx",
      "model_int8_dynamic.onnx",
      weight_type=QuantType.QInt8,         ; weights quantised to INT8
      nodes_to_quantize=["MatMul", "Gemm"], ; only quantise linear layers
  )
  # Memory: 4× smaller weights
  # Speed:  2-3× faster on Intel CPU with VNNI
  # Accuracy: < 0.5% drop typically

  # ── Static quantisation (both weights AND activations INT8) ────────────
  class MyCalibDataReader(CalibrationDataReader):
      def __init__(self, data):
          self.data = iter(data)
      def get_next(self):
          try:
              return {"x": next(self.data)}
          except StopIteration:
              return None

  calib_data = [np.random.randn(1, 16).astype(np.float32) for _ in range(100)]
  calibrator  = MyCalibDataReader(calib_data)

  quantize_static(
      "model.onnx",
      "model_int8_static.onnx",
      calibration_data_reader=calibrator,
      quant_format=QuantFormat.QDQ,       ; QuantDequant nodes (ORT preferred)
      per_channel=True,                   ; per-channel scale for weights
  )
  # Memory: 4× smaller
  # Speed:  2-4× faster on modern CPU (AVX-512 VNNI) or ARM (dotprod)
  # Accuracy: typically < 1% drop if calibration data is representative
"""
print(INT8_REF)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · ONNX Runtime — Execution Providers, IO Binding & Benchmarking": {
        "description": (
            "ONNX Runtime in production. Execution providers: CPU, CUDA, TensorRT. "
            "Session options: graph optimisation levels, intra/inter op threading. "
            "IO binding to eliminate CPU-GPU memory copies. "
            "Benchmark ORT vs PyTorch eager across batch sizes. "
            "HuggingFace Optimum: ORT-accelerated transformer inference. "
            "Complete ecosystem diagram: ONNX as the hub."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  ONNX RUNTIME — PROVIDERS, IO BINDING & BENCHMARKING")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: ORT session configuration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Session options and execution providers")
print("━" * 65)
print()

try:
    import onnxruntime as ort
    print(f"  ONNX Runtime {ort.__version__}")
    print()

    # Available providers
    available = ort.get_available_providers()
    print(f"  Available execution providers:")
    for p in available:
        print(f"    {'✅' if 'CUDA' in p or 'CPU' in p else '○'} {p}")
    print()

    # Session options explained
    print("  SessionOptions configuration:")
    OPTS_GUIDE = """
  opts = ort.SessionOptions()

  # ── Threading ────────────────────────────────────────────────────────
  opts.intra_op_num_threads = 4    ; threads within one op (e.g., matmul)
  opts.inter_op_num_threads = 1    ; threads across ops (parallel branches)
  # Recommendation:
  #   intra = num_physical_cores (max parallelism within GEMM kernels)
  #   inter = 1 (avoid context switching between ops)

  # ── Graph optimisation level ──────────────────────────────────────────
  opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  # Levels:
  #   ORT_DISABLE_ALL:        no optimisation (debugging only)
  #   ORT_ENABLE_BASIC:       constant folding, identity elimination
  #   ORT_ENABLE_EXTENDED:    op fusion (Conv+BN, Attention, LayerNorm)
  #   ORT_ENABLE_ALL:         + memory optimisation, layout transforms

  # ── Memory ────────────────────────────────────────────────────────────
  opts.enable_mem_pattern = True   ; memory pattern pre-allocation (faster)
  opts.enable_mem_reuse   = True   ; reuse buffers across runs

  # ── Profiling ─────────────────────────────────────────────────────────
  opts.enable_profiling = True
  opts.profile_file_prefix = "/tmp/ort_profile"
  # After session ends, creates /tmp/ort_profile.json (Chrome trace format)

  # ── Save optimised graph ──────────────────────────────────────────────
  opts.optimized_model_filepath = "/tmp/optimised.onnx"
  # Saves the graph AFTER fusion — useful for inspecting what ORT did

  # ── Provider configuration ────────────────────────────────────────────
  sess = ort.InferenceSession(
      "model.onnx",
      sess_options=opts,
      providers=[
          ("CUDAExecutionProvider", {
              "device_id": 0,
              "arena_extend_strategy": "kNextPowerOfTwo",
              "gpu_mem_limit": 2 * 1024 ** 3,    ; 2 GB GPU memory limit
              "cudnn_conv_algo_search": "EXHAUSTIVE",  ; find best cuDNN algo
          }),
          "CPUExecutionProvider",   ; fallback for unsupported GPU ops
      ]
  )
"""
    print(OPTS_GUIDE)

    # Run a real session with all optimisations
    import io, onnx, torch, torch.nn as nn

    class Benchmark(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 10),
            )
        def forward(self, x): return self.net(x)

    model = Benchmark().eval()
    dummy = torch.randn(1, 64)
    buf   = io.BytesIO()
    torch.onnx.export(model, (dummy,), buf, opset_version=17,
                       input_names=["x"], output_names=["y"],
                       dynamic_axes={"x": {0: "b"}, "y": {0: "b"}})
    buf.seek(0)
    onnx_bytes = buf.getvalue()

    # Compare optimisation levels
    print("  Optimisation level benchmark (4-layer MLP, batch=32):")
    print()
    x_np  = np.random.randn(32, 64).astype(np.float32)

    opt_levels = [
        ("ORT_DISABLE_ALL",    ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("ORT_ENABLE_BASIC",   ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("ORT_ENABLE_EXTENDED",ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("ORT_ENABLE_ALL",     ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]

    print(f"  {'Level':<25} | {'Latency (ms)':>14} | {'Speedup':>9}")
    print(f"  {'─'*52}")

    base_t = None
    for name, level in opt_levels:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = level
        opts.intra_op_num_threads      = 4

        sess = ort.InferenceSession(onnx_bytes, sess_options=opts,
                                     providers=["CPUExecutionProvider"])
        for _ in range(20): sess.run(None, {"x": x_np})   # warmup

        REPS = 500
        t0 = time.perf_counter()
        for _ in range(REPS): sess.run(None, {"x": x_np})
        t_ms = (time.perf_counter() - t0) / REPS * 1000

        if base_t is None: base_t = t_ms
        speedup = base_t / t_ms
        print(f"  {name:<25} | {t_ms:14.4f} | {speedup:9.2f}×")

    print()

except ImportError as e:
    print(f"  Missing: {e}  →  pip install onnxruntime torch")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: IO Binding — eliminating CPU-GPU copies
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — IO Binding: eliminate CPU↔GPU memory copies")
print("━" * 65)
print()

IO_BINDING_REF = """
  DEFAULT ORT (with GPU) DATA FLOW:
    CPU numpy array → [memcpy HtoD] → GPU → [ORT CUDA kernel] → GPU →
    [memcpy DtoH] → CPU numpy array

  Two PCIe transfers per inference call (slow for large tensors!)

  WITH IO BINDING:
    GPU tensor (from PyTorch/CUDA) → [ORT CUDA kernel] → GPU tensor

  Zero PCIe transfers when working with GPU tensors end-to-end.

  CODE:
  import onnxruntime as ort
  import numpy as np

  sess = ort.InferenceSession("model.onnx",
                               providers=["CUDAExecutionProvider"])

  # Create IO binding
  io_binding = sess.io_binding()

  # Bind input (GPU tensor)
  # Option A: from a CUDA numpy-like array
  input_gpu = ort.OrtValue.ortvalue_from_numpy(input_np, "cuda", 0)
  io_binding.bind_ortvalue_input("input", input_gpu)

  # Option B: from a PyTorch CUDA tensor (zero-copy!)
  import torch
  input_torch = torch.randn(1, 3, 224, 224, device="cuda")
  io_binding.bind_input(
      name="input",
      device_type="cuda",
      device_id=0,
      element_type=np.float32,
      shape=tuple(input_torch.shape),
      buffer_ptr=input_torch.data_ptr(),   # direct GPU memory pointer!
  )

  # Bind output (pre-allocated GPU buffer)
  output_torch = torch.empty(1, 1000, device="cuda")
  io_binding.bind_output(
      name="output",
      device_type="cuda",
      device_id=0,
      element_type=np.float32,
      shape=(1, 1000),
      buffer_ptr=output_torch.data_ptr(),
  )

  # Run without any CPU-GPU copies!
  sess.run_with_iobinding(io_binding)
  # output_torch now contains the result — directly on GPU

  # Measure speedup (typical for ResNet-50 on A100):
  # Standard run:   8.2 ms  (includes ~0.5ms HtoD + ~0.3ms DtoH)
  # IO binding:     7.4 ms  (pure compute, no transfer overhead)
  # Savings: ~10%  (more significant for smaller models with large I/O)
"""
print(IO_BINDING_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: ONNX ecosystem diagram and HuggingFace Optimum
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — ONNX ecosystem and HuggingFace Optimum")
print("━" * 65)
print()

HF_OPTIMUM_REF = """
  HUGGINGFACE OPTIMUM — ORT-accelerated transformer inference:

  # Install
  pip install optimum[onnxruntime]    ; CPU
  pip install optimum[onnxruntime-gpu]; GPU

  # Classification (e.g., DistilBERT sentiment analysis)
  from optimum.onnxruntime import ORTModelForSequenceClassification
  from transformers import AutoTokenizer

  model     = ORTModelForSequenceClassification.from_pretrained(
      "distilbert-base-uncased-finetuned-sst-2-english",
      export=True,             ; export to ONNX on first call
  )
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-...")

  inputs = tokenizer("I love this movie!", return_tensors="pt")
  logits = model(**inputs).logits
  # Same API as PyTorch HuggingFace! But faster via ORT.

  # Text generation (e.g., GPT-2)
  from optimum.onnxruntime import ORTModelForCausalLM
  model = ORTModelForCausalLM.from_pretrained("gpt2", export=True)
  # Generates text via ORT — typically 1.3-1.8× faster than PyTorch eager

  # With quantisation
  from optimum.onnxruntime.configuration import AutoQuantizationConfig
  from optimum.onnxruntime import ORTQuantizer
  quantizer = ORTQuantizer.from_pretrained(model)
  qconfig   = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
  quantizer.quantize(save_dir="./quantized_model", quantization_config=qconfig)
  # 4× smaller model, 2-3× faster on ARM (e.g., Apple M1/M2, Raspberry Pi 4)
"""
print(HF_OPTIMUM_REF)

ECOSYSTEM_DIAGRAM = """
  THE ONNX ECOSYSTEM — COMPLETE DIAGRAM
  ════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │                     TRAINING (produces ONNX)                         │
  │  PyTorch      TensorFlow      JAX        MXNet     Sklearn/XGBoost   │
  │  (torch.onnx) (tf2onnx)      (jax2onnx) (mx2onnx) (onnxmltools)      │
  └──────────────────────────┬───────────────────────────────────────────┘
                             │  .onnx file (protobuf binary)
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │                     ONNX MODEL HUB                                   │
  │  onnx.checker    shape_inference    onnx-simplifier    onnxoptimizer │
  │  Netron (visualiser)     onnx-graphsurgeon (graph surgery)           │
  └──────┬───────────────────┬────────────────────────────────┬──────────┘
         │                   │                                │
         ▼                   ▼                                ▼
  ┌────────────────┐ ┌──────────────────────┐    ┌──────────────────────┐
  │  ONNX Runtime  │ │   TVM (module 12)    │    │  TensorRT (NVIDIA)   │
  │  CPU/CUDA/OV   │ │   Relay.from_onnx    │    │  trtexec --onnx=     │
  │  GPU/CoreML/   │ │   + auto-scheduling  │    │  INT8/FP16/FP8 opt   │
  │  DirectML/NNAPI│ │   → any hardware     │    │  → .engine file      │
  └───────┬────────┘ └──────────────────────┘    └──────────────────────┘
          │
    ┌─────┴───────────────────────────────────────────────┐
    │              DOWNSTREAM TARGETS                     │
    │  x86 CPU (MLAS)    ARM CPU (XNNPACK/ACL)            │
    │  NVIDIA GPU (CUDA) AMD GPU (ROCm)                   │
    │  Intel VPU (OV)    Apple ANE (CoreML)               │
    │  Qualcomm NPU (SNPE/QNN)  Android NNAPI             │
    │  Browser (onnxruntime-web / WebNN)                  │
    └─────────────────────────────────────────────────────┘

  ONNX CONNECTIONS TO COMPILER STACK MODULES:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ LLVM  (module 09) ← ORT's CPU backend (MLAS) uses LLVM-based code    │
  │ MLIR  (module 10) ← onnx-mlir converts ONNX → MLIR ONNX dialect      │
  │ XLA   (module 11) ← StableHLO ↔ ONNX converters (jax→onnx→xla)       │
  │ TVM   (module 12) ← relay.frontend.from_onnx() is TVM's import path  │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(ECOSYSTEM_DIAGRAM)

try:
    import onnxruntime as ort, torch, torch.nn as nn, io

    # Final comprehensive benchmark
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 512), nn.GELU(),
                nn.Linear(512, 256), nn.GELU(),
                nn.Linear(256, 10))
        def forward(self, x): return self.layers(x)

    model = MLP().eval()
    buf   = io.BytesIO()
    torch.onnx.export(model, torch.randn(1,128), buf, opset_version=17,
                       input_names=["x"], output_names=["y"],
                       dynamic_axes={"x":{0:"b"},"y":{0:"b"}})
    buf.seek(0)
    onnx_bytes = buf.getvalue()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads      = 4
    sess = ort.InferenceSession(onnx_bytes, sess_options=opts,
                                 providers=["CPUExecutionProvider"])

    print("  Final benchmark: MLP (128→512→256→10) across batch sizes")
    print()
    print(f"  {'Batch':>6} | {'PyTorch eager (ms)':>20} | {'ORT (ms)':>10} | {'Speedup':>9}")
    print(f"  {'─'*55}")

    for bs in [1, 4, 16, 64, 256]:
        x_np = np.random.randn(bs, 128).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        REPS = 1000 if bs <= 16 else 200
        with torch.no_grad():
            for _ in range(20): model(x_pt)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(REPS): model(x_pt)
        t_pt = (time.perf_counter()-t0)/REPS*1000

        for _ in range(20): sess.run(None, {"x": x_np})
        t0 = time.perf_counter()
        for _ in range(REPS): sess.run(None, {"x": x_np})
        t_ort = (time.perf_counter()-t0)/REPS*1000

        print(f"  {bs:6d} | {t_pt:20.4f} | {t_ort:10.4f} | {t_pt/t_ort:9.2f}×")

    print()
    print("  ORT speedup is highest at small batch sizes where Python overhead")
    print("  (per-op dispatch) dominates. At large batches, BLAS dominates both.")

except ImportError as e:
    print(f"  Missing dependency: {e}")
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