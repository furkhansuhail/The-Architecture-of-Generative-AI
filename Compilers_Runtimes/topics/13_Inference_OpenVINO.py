"""
OpenVINO — Intel's Open Visual Inference and Neural Network Optimisation
=========================================================================

OpenVINO (Open Visual Inference and Neural Network Optimisation) is Intel's
end-to-end platform for deploying deep learning inference across the full
breadth of Intel hardware: CPUs from embedded Atom to server Xeon, integrated
graphics, discrete Arc GPUs, the Intel Neural Processing Unit (NPU) in AI PCs,
and legacy Myriad Vision Processing Units (VPUs) in edge cameras and drones.

OpenVINO is not simply a faster inference engine. It is a complete toolkit:
a model representation format (OpenVINO IR), a runtime with a pluggable
device abstraction layer, a neural network compression framework (NNCF) for
quantisation and pruning, a model serving infrastructure (OVMS), and a GenAI
pipeline for LLM inference — all unified under a single Python and C++ API.

The central problem OpenVINO solves: Intel's hardware is used everywhere in
the inference supply chain — from cloud servers running Intel Xeon to edge
boxes running Intel Core to IoT sensors running Intel Myriad — yet each
hardware tier speaks a different language of compute primitives. A model
trained in PyTorch cannot automatically use AVX-512 VNNI on Xeon, run INT8
on an Arc GPU, or accelerate on a Meteor Lake NPU without hand-written
backends for each. OpenVINO provides that translation, automatically, for
any model, targeting any Intel hardware.

The architecture has three levels:

    FRONTEND:   accepts any model format (PyTorch, TF, ONNX, PaddlePaddle,
                JAX) and converts it to the OpenVINO IR. The IR is a graph
                of typed, versioned operations — analogous to ONNX but
                with Intel-specific ops and richer type annotations.

    RUNTIME:    compiles the IR for a specific device at load time.
                Device plugins (CPU, GPU, NPU) apply device-specific passes:
                operator fusion, layout optimisation, INT8 kernel selection,
                cache-friendly memory layout planning.

    TOOLS:      NNCF (quantisation, pruning, sparsity), Benchmark App
                (throughput/latency measurement), Model Optimizer (legacy
                CLI), OpenVINO Model Server (production serving),
                OpenVINO GenAI (LLM pipelines).

In the connected compiler stack:
    LLVM      (module 01) ← OpenVINO's CPU plugin uses oneDNN which calls LLVM-compiled kernels
    MLIR      (module 02) ← OpenVINO uses MLIR internally for some passes; the OV IR mirrors MLIR concepts
    XLA       (module 05) ← JAX models reach OpenVINO via torch.export/ONNX → OV conversion
    TVM       (module 08) ← TVM and OpenVINO compete on CPU/edge; OV wins for Intel-specific hardware
    ONNX      (module 10) ← ONNX is OpenVINO's primary and most reliable import path
    TFLite    (module 11) ← TFLite models can be loaded directly by OpenVINO's TFLite frontend
    OpenVINO  (this)      ← Intel's inference runtime, optimisation toolkit, and serving infrastructure

"""

import textwrap
import re

TOPIC_NAME   = "OpenVINO — Intel's Open Visual Inference and Neural Network Optimisation"
DISPLAY_NAME = "13 · OpenVINO"
ICON         = "🔵"
SUBTITLE     = "OpenVINO IR, oneDNN CPU, NPU/GPU Plugins, NNCF Quantisation, and GenAI"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY OPENVINO EXISTS: INTEL'S HARDWARE DIVERSITY PROBLEM

### Intel's Fragmented Compute Landscape

    Intel manufactures the widest range of compute hardware of any single
    vendor. Unlike NVIDIA (primarily GPU) or Qualcomm (primarily mobile SoC),
    Intel builds:

    CPU (x86):
        Intel Atom (E-series):   ultra-low-power edge, 2–6W, IoT cameras
        Intel Core (i3/i5/i7):  mainstream PC, 15–65W, AI PCs with NPU
        Intel Core Ultra:        AI PCs with integrated NPU + Arc GPU
        Intel Xeon (Scalable):   server, 125–350W, data-centre inference
        Xeon with AMX:           Advanced Matrix Extensions (tile-based INT8/BF16)

    GPU (OpenCL / SYCL):
        Intel UHD (integrated):  in every Core CPU, OpenCL 3.0
        Intel Iris Xe:           in 11th-gen+ Core, stronger iGPU
        Intel Arc (discrete):    Alchemist/Battlemage, GPU class
        Intel Ponte Vecchio:     HPC accelerator (supercomputer class)

    NPU (Neural Processing Unit):
        Intel Meteor Lake NPU:   integrated in Core Ultra Gen 1, ~10 TOPS INT8
        Intel Lunar Lake NPU:    integrated in Core Ultra Gen 2, ~48 TOPS INT8
        Intel Movidius Myriad X: edge VPU (legacy, used in USB AI sticks)
        Intel Movidius Myriad 2: earlier edge VPU (still deployed in cameras)

    These hardware tiers are NOT compatible at the instruction level:
        Atom does not support AVX-512 (only SSE4.2 and some VNNI).
        Xeon with AMX uses tile registers unavailable on any GPU.
        NPU has a fixed hardware dataflow engine with no general compute.
        GPU uses OpenCL/SYCL kernel model — unrelated to CPU vector ops.

    Without OpenVINO, a developer deploying across this range would need:
        Custom VNNI kernels for Xeon inference
        Custom OpenCL kernels for GPU inference
        Custom Myriad SDK code for VPU inference
        Custom NPU runtime code for AI PC
    That is four separate engineering efforts — the N×M problem again.

### OpenVINO's Solution: One API, Many Devices

    OpenVINO was launched in 2018 as the Intel Distribution of OpenVINO
    Toolkit. It unifies all Intel inference targets behind a single API:

        import openvino as ov

        core = ov.Core()
        model = core.read_model("model.xml")
        compiled = core.compile_model(model, "CPU")    # or "GPU", "NPU", "AUTO"
        result  = compiled.create_infer_request().infer({"input": data})

    The same code, without modification, runs on:
        A Raspberry Pi with Intel M.2 Myriad X accelerator
        A laptop with Core Ultra NPU
        An edge server with Xeon Scalable + Arc GPU
        A cloud VM with Intel Xeon Gold

    What changes between devices is the PLUGIN — the device-specific
    compilation and execution engine — not the user-facing API.

### OpenVINO's Scope Beyond Inference

    OpenVINO is larger than just a runtime. The full toolkit includes:

    OPENVINO RUNTIME:        the core inference engine (read / compile / run)
    MODEL CONVERSION:        openvino.convert_model() for any input format
    NNCF:                    Neural Network Compression Framework —
                             PTQ, QAT, structured pruning, sparsity, GPTQ
    BENCHMARK APP:           CLI tool for latency/throughput measurement
    OPENVINO MODEL SERVER:   gRPC/REST inference serving (OpenAI API compatible)
    OPENVINO GENAI:          high-level LLM inference pipeline with
                             speculative decoding, beam search, KV cache
    OPENVINO NOTEBOOKS:      100+ Jupyter notebooks on kaggle/github
    TOKENIZERS:              fast CPU tokenizer library for LLM pipelines


##### PART 2 — THE OPENVINO IR: XML + BIN FORMAT

### Two-File Model Representation

    OpenVINO IR (Intermediate Representation) is a two-file format:

        model.xml   — the computation graph (operations and their connections)
                      human-readable XML; can be inspected in any text editor
        model.bin   — the binary weight data (float32, float16, INT8, ...)
                      little-endian raw tensor bytes; aligned to 64-byte boundaries

    The XML file references the BIN file by relative path. Both must be
    co-located (or the BIN path updated in the XML).

    This separation matters for deployment:
        The XML can be versioned in git (it is text).
        The BIN can be stored in a separate binary asset pipeline.
        The XML can be inspected and edited without a Python environment.
        The BIN can be memory-mapped directly into the runtime (zero-copy).

### XML Structure: Layer Graph

    The XML file contains:

    DOCUMENT STRUCTURE:
        <net name="resnet50" version="11">
            <layers>    <!-- all operation nodes -->
                <layer id="0"  name="input"   type="Parameter" .../>
                <layer id="1"  name="conv1"   type="Convolution" .../>
                <layer id="2"  name="relu1"   type="ReLU" .../>
                ...
                <layer id="N"  name="output"  type="Result" .../>
            </layers>
            <edges>     <!-- data flow connections -->
                <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
                ...
            </edges>
        </net>

    LAYER TYPES (the OpenVINO operation set):
        Special:       Parameter (graph input), Result (graph output), Const
        Linear ops:    Convolution, GroupConvolution, ConvolutionBackpropData
                       MatMul, Gemm (batched matmul)
        Activation:    Relu, Sigmoid, Tanh, Elu, Mish, Swish, Gelu, HSwish,
                       Clamp, PRelu, Selu
        Normalization: BatchNormInference, LayerNorm, GroupNorm, MVN
        Pooling:       MaxPool, AvgPool, AdaptiveAvgPool, AdaptiveMaxPool
        Elementwise:   Add, Multiply, Subtract, Divide, Power, Sqrt, Exp,
                       Log, Abs, Negative, Maximum, Minimum, Equal, Less
        Reduction:     ReduceMax, ReduceMin, ReduceSum, ReduceMean, ReduceProd
        Shape ops:     Reshape, Transpose, Unsqueeze, Squeeze, Flatten, Broadcast
        Data ops:      Gather, GatherElements, GatherND, ScatterNDUpdate,
                       Pad, Slice, StridedSlice, Split, VariadicSplit, Concat
        Attention:     MultiHeadAttention (OV 2023.2+), ScaledDotProductAttention
        Quantization:  FakeQuantize (for QAT), Convert (dtype cast)
        Control flow:  TensorIterator, Loop (sub-graph based)
        Custom:        Extension ops via OpenVINO extension API

    LAYER ANATOMY — a single Convolution layer:
        <layer id="3" name="conv2_1/conv" type="Convolution" version="opset1">
            <data
                strides="1,1"
                dilations="1,1"
                pads_begin="1,1"
                pads_end="1,1"
                auto_pad="explicit" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>  <!-- batch -->
                    <dim>64</dim> <!-- input channels -->
                    <dim>56</dim> <!-- height -->
                    <dim>56</dim> <!-- width -->
                </port>
                <port id="1" precision="FP32">  <!-- weight tensor -->
                    <dim>64</dim>  <dim>64</dim>
                    <dim>3</dim>   <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim> <dim>64</dim> <dim>56</dim> <dim>56</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="147456"/>   <!-- 64*64*3*3*4 bytes -->
                <biases  offset="147456" size="256"/> <!-- 64*4 bytes -->
            </blobs>
        </layer>

    EDGE FORMAT:
        <edge from-layer="3" from-port="2" to-layer="4" to-port="0"/>
        from-layer/to-layer: the layer id attributes
        from-port/to-port:   the port id attributes within that layer

### The OpenVINO Operation Set (opset)

    OpenVINO operations are versioned in "opsets":
        opset1:  the original operations (Conv, MatMul, Relu, ...)
        opset2:  added MVN, BatchToSpace, SpaceToBatch, ...
        opset4:  added Interpolate v4 (resize with anti-aliasing)
        opset6:  added ExperimentalDetectronROIFeatureExtractor
        opset8:  added DeformableConvolution v8, AdaptiveAvgPool
        opset12: added ScatterNDUpdate v12 semantics
        opset13: added MultiHeadAttention, PagedAttention (for LLMs)
        opset14: added Inverse, STFT, ISTFT (audio models)
        opset15: ongoing additions for transformers and audio

    A model's XML references the specific opset for each layer
    (e.g., version="opset8"). This guarantees backward compatibility:
    an older runtime that knows opset8 can still run models using opset8 ops
    even if a newer opset (opset15) is available.

### Accessing the IR in Python

    OpenVINO's Python API exposes the IR as a typed graph:

        import openvino as ov

        core  = ov.Core()
        model = core.read_model("resnet50.xml")  # or .onnx, .tflite, etc.

        # Graph structure
        print(f"Inputs:  {[i.get_any_name() for i in model.inputs]}")
        print(f"Outputs: {[o.get_any_name() for o in model.outputs]}")
        print(f"Ops:     {len(list(model.get_ordered_ops()))}")

        # Walk all operations
        for op in model.get_ordered_ops():
            print(f"  {op.get_type_name():30s}  "
                  f"{[str(o.partial_shape) for o in op.outputs()]}")

        # Access specific op attributes
        for op in model.get_ordered_ops():
            if op.get_type_name() == "Convolution":
                strides = op.get_attribute("strides")
                groups  = op.get_attribute("group")
                print(f"  Conv: strides={strides}")

    PARTIAL SHAPES:
        Unlike ONNX which marks dynamic dims as "?" strings,
        OpenVINO uses PartialShape objects with typed dimensions:
            ov.PartialShape([1, 3, 224, 224])     ; fully static
            ov.PartialShape([-1, 3, 224, 224])    ; dynamic batch
            ov.PartialShape([-1, 3, -1, -1])      ; dynamic H and W
            ov.PartialShape.dynamic()              ; completely unknown

        Setting input shapes:
            model.reshape({"input": ov.PartialShape([1, 3, 224, 224])})
            # OR set specific batch size:
            model.reshape({"input": [8, 3, 224, 224]})


##### PART 3 — MODEL CONVERSION: FROM ANY FRAMEWORK TO OPENVINO IR

### The Unified Conversion API

    OpenVINO 2022.1 introduced a unified model conversion API that accepts
    framework-native model objects directly — no CLI tool, no intermediate
    export step, no temp files for common cases.

    PYTHON API:
        import openvino as ov
        ov_model = ov.convert_model(source)
        ov.save_model(ov_model, "model.xml")  # saves .xml + .bin

    The source can be:
        A PyTorch nn.Module (with example_input)
        A TensorFlow/Keras SavedModel path, tf.Module, or Keras model
        An ONNX file path or onnx.ModelProto object
        A TFLite file path
        A PaddlePaddle model
        An OpenVINO IR file path (for re-optimization)

### PyTorch → OpenVINO

    DIRECT PYTORCH CONVERSION (OpenVINO 2023.0+):
        import torch, openvino as ov

        model    = MyModel().eval()
        example  = torch.randn(1, 3, 224, 224)
        ov_model = ov.convert_model(model, example_input=example)

    UNDER THE HOOD:
        OpenVINO calls torch.export.export(model, example_input) internally.
        The resulting ExportedProgram (ATen IR) is then lowered to OV IR.
        Dynamo-based tracing: handles more dynamic PyTorch code than TorchScript.

    DYNAMIC SHAPES:
        ov_model = ov.convert_model(
            model,
            example_input=example,
            input=[ov.InputCutInfo(
                name="input",
                shape=ov.PartialShape([-1, 3, 224, 224]),  # dynamic batch
                type=np.float32)])

    TORCHSCRIPT PATH (legacy, still supported):
        scripted = torch.jit.script(model)
        ov_model = ov.convert_model(scripted, example_input=example)

    COMMON PYTORCH CONVERSION ISSUES:
        "Unsupported operation" during tracing:
            TorchDynamo graph breaks caused by Python control flow.
            Fix: restructure with torch.where(), avoid .item() calls,
                 use torch.compile before ov.convert_model.

        "Type mismatch" on inputs:
            Make sure example_input dtype matches model's expected dtype.
            Fix: example = torch.randn(...).to(torch.float32)

        In-place operations:
            ops like x.add_(1) can confuse the tracer.
            Fix: use x = x + 1 throughout the model.

### TensorFlow and Keras → OpenVINO

    SAVEDMODEL:
        ov_model = ov.convert_model("/path/to/saved_model/")

    KERAS MODEL IN MEMORY:
        import tensorflow as tf
        keras_model = tf.keras.applications.MobileNetV2()
        ov_model    = ov.convert_model(keras_model)

    TF CONCRETE FUNCTION:
        @tf.function(input_signature=[tf.TensorSpec([1,224,224,3], tf.float32)])
        def serving(x): return model(x, training=False)
        ov_model = ov.convert_model(serving.get_concrete_function())

    CHANNEL ORDER NOTE:
        TensorFlow uses NHWC (channels last).
        OpenVINO internally uses NCHW (channels first) for CPU kernels.
        The conversion automatically inserts Transpose nodes where needed.
        To avoid runtime transposes: request NHWC layout from the plugin
        via ov.Layout and ov.preprocess.PrePostProcessor (see Part 4).

### ONNX → OpenVINO (Most Reliable Path)

    ONNX is OpenVINO's most tested and complete import path.
    Because ONNX defines precise semantics per-op, conversion is less
    ambiguous than direct framework import.

        ov_model = ov.convert_model("model.onnx")
        # OR with explicit input shapes (useful for static-shape compilation):
        ov_model = ov.convert_model("model.onnx",
                                     input=[("input", [1, 3, 224, 224], np.float32)])

    ONNX OPSET SUPPORT:
        OpenVINO supports ONNX opset 1–20 (tracking ONNX releases closely).
        Ops not natively supported are handled by:
            a) Decomposition into supported primitives (transparent to user)
            b) ONNX Runtime fallback for unknown ops

    WHY ONNX → OV IS PREFERRED OVER DIRECT PYTORCH → OV:
        ONNX export is a well-understood, debuggable step.
        The ONNX graph can be inspected in Netron before conversion.
        ONNX simplification (onnxsim) can remove artifacts before OV sees them.
        Wider op coverage: some PyTorch ops have ONNX equivalents but not OV.

### TFLite → OpenVINO

    OpenVINO can read .tflite files directly:
        ov_model = ov.convert_model("model.tflite")
    This makes OpenVINO usable as a TFLite backend for x86 deployment.
    INT8 quantised TFLite models retain their quantisation through the
    conversion and run with INT8 kernels on OpenVINO CPU.

### Saving and Loading OpenVINO IR

    SAVE:
        ov.save_model(ov_model, "model.xml", compress_to_fp16=False)
        # Creates: model.xml (graph structure)
        #          model.bin (weight data)
        # With FP16 compression:
        ov.save_model(ov_model, "model_fp16.xml", compress_to_fp16=True)
        # Weights are cast to FP16 → ~2× smaller BIN file

    LOAD:
        core     = ov.Core()
        ov_model = core.read_model("model.xml")
        # The core automatically finds model.bin next to model.xml


##### PART 4 — THE OPENVINO RUNTIME: CORE, COMPILED MODEL, INFER REQUEST

### The Three-Object API

    OpenVINO's Python runtime revolves around three objects:

    ov.Core:
        The global context. Loads plugins, reads models.
        One Core per application (expensive to create; device plugins loaded).
        Thread-safe: can be shared across threads.
        Methods:
            core.available_devices           → ['CPU', 'GPU.0', 'NPU']
            core.read_model("model.xml")     → ov.Model
            core.compile_model(model, "CPU") → ov.CompiledModel
            core.get_property("CPU", "DEVICE_ID")  → device info

    ov.CompiledModel:
        The model compiled for a specific device.
        Creation triggers device-specific optimisation: op fusion,
        layout transformation, kernel selection, INT8 calibration.
        Expensive to create (100ms–10s depending on model + device).
        Thread-safe: multiple InferRequests can be created from one CompiledModel.
        Methods:
            compiled.create_infer_request()  → ov.InferRequest
            compiled.inputs / .outputs       → list of input/output info

    ov.InferRequest:
        A single inference "lane" — holds input/output buffers and state.
        NOT thread-safe: one InferRequest per concurrent inference thread.
        Methods:
            request.infer({"input": array})              → dict of outputs
            request.set_input_tensor(tensor)             → in-place input set
            request.get_output_tensor()                  → output tensor
            request.start_async()                        → fire async
            request.wait() / request.wait_for(timeout)  → synchronise

### The Compilation Pipeline

    When core.compile_model(model, device, config) is called:

    STEP 1 — PREPROCESSING FUSION:
        If a PrePostProcessor was attached, fuse preprocessing ops
        (normalisation, resize, colour conversion) into the model graph.
        These then execute on the device alongside the model — not on CPU.

    STEP 2 — LAYOUT PROPAGATION:
        Determine optimal tensor layouts for each device:
        CPU: typically NCHW (channels-first) for AVX-512 convolutions.
        GPU: OpenCL kernels prefer channel-last variants for some ops.
        The compiler inserts transpose ops to enforce layout.

    STEP 3 — CONSTANT FOLDING:
        Evaluate all operations with known constant inputs.
        Absorbs BatchNorm into preceding Conv (BN fold).
        Absorbs bias into preceding MatMul/Conv.

    STEP 4 — OPERATOR FUSION:
        CPU plugin fuses common patterns:
            Conv + BatchNorm → Conv (with folded weights)
            Conv + Bias + ReLU → single oneDNN primitive call
            MatMul + Bias + Gelu → single oneDNN MKL-DNN call
            MVN + Scale + Shift → fused LayerNorm primitive
        GPU plugin fuses elementwise chains into single OpenCL kernels.

    STEP 5 — INT8 QUANTISATION (if model is quantised or config specifies):
        Select INT8 kernels from oneDNN (CPU) or OpenCL (GPU).
        Validate scale/zero_point values from FakeQuantize ops.
        Insert requantise operations at precision boundaries.

    STEP 6 — MEMORY PLANNING:
        Compute tensor lifetimes (like TFLite's arena planning).
        Assign buffer offsets to minimise peak memory.
        Pre-allocate all buffers for the request pool.

    STEP 7 — JIT CODE GENERATION (CPU only):
        oneDNN (MKL-DNN) generates SIMD-optimised kernel code at compile time.
        Selects instruction set: SSE4.2 / AVX2 / AVX-512 / AMX based on CPUID.
        Generates code specialised for the exact (M, N, K) shapes.

### PrePostProcessor: Embedding Preprocessing in the Graph

    A common performance trap: preprocessing (normalise, resize, convert BGR→RGB)
    runs in Python/NumPy on CPU while the model runs on GPU or NPU.
    This creates a CPU bottleneck that limits end-to-end throughput.

    OpenVINO's PrePostProcessor fuses preprocessing INTO the compiled model:

        from openvino.preprocess import PrePostProcessor, ColorFormat
        from openvino import Layout, Type

        ppp   = PrePostProcessor(model)

        # Declare that the real input comes as uint8 BGR from a camera:
        ppp.input("input").tensor() \\
            .set_element_type(Type.u8)       \\
            .set_layout(Layout("NHWC"))      \\
            .set_color_format(ColorFormat.BGR)

        # Declare what the model expects (float32, NCHW, RGB, normalised):
        ppp.input("input").model() \\
            .set_layout(Layout("NCHW"))

        # Specify the preprocessing steps:
        ppp.input("input").preprocess() \\
            .convert_color(ColorFormat.RGB) \\  ; BGR → RGB
            .resize(ResizeAlgorithm.RESIZE_LINEAR, 224, 224) \\  ; resize
            .convert_element_type(Type.f32) \\  ; uint8 → float32
            .mean([0.485, 0.456, 0.406]) \\     ; subtract ImageNet mean
            .scale([0.229, 0.224, 0.225])        ; divide by std

        model_with_preproc = ppp.build()
        # Now the compiled model takes uint8 BGR NHWC directly.
        # The entire preprocessing runs on the target device.

    This is critical for GPU/NPU: the preprocessing no longer stalls on CPU.
    GPU preprocessing via OpenCL is 5–10× faster than NumPy for HD video.

### Asynchronous Inference and Pipelining

    ov.InferRequest.start_async() submits inference without blocking.
    This enables PIPELINING: overlapping preprocessing, inference, and
    postprocessing for maximum throughput.

    PIPELINE PATTERN (double-buffered):
        # Create two InferRequests (ping-pong buffer)
        req0 = compiled.create_infer_request()
        req1 = compiled.create_infer_request()

        req0.set_input_tensor(batch_0)
        req0.start_async()         # start inference on batch 0

        while more_batches:
            preprocess(next_batch)            # CPU preprocessing (overlaps GPU)
            req0.wait()                       # wait for batch 0 to finish
            results_0 = req0.get_output_tensor().data
            postprocess(results_0)            # postprocess batch 0 (overlaps GPU)
            req0.set_input_tensor(next_batch)
            req0.start_async()                # start batch N+1
            req0, req1 = req1, req0           # swap

    CALLBACK-BASED (event-driven):
        def on_complete(req, userdata):
            result = req.get_output_tensor().data.copy()
            queue.put((userdata, result))

        for i, batch in enumerate(dataset):
            request = compiled.create_infer_request()
            request.set_callback(on_complete, userdata=i)
            request.set_input_tensor(ov.Tensor(batch))
            request.start_async()

    ASYNC QUEUE (OpenVINO's built-in pipeline):
        from openvino.runtime import AsyncInferQueue

        queue = AsyncInferQueue(compiled, jobs=4)   # 4 parallel requests
        queue.set_callback(callback_fn)

        for batch in dataset:
            queue.start_async({"input": batch})     # auto-selects free request
        queue.wait_all()                            # drain the queue


##### PART 5 — DEVICE PLUGINS: CPU, GPU, NPU, AND AUTO

### The CPU Plugin (oneDNN / MKL-DNN)

    The CPU plugin is OpenVINO's most mature and universally available backend.
    It uses oneDNN (formerly MKL-DNN / Intel MKL) as its compute library.

    INSTRUCTION SET AUTO-DETECTION:
        At compile_model() time, the CPU plugin calls CPUID and selects
        the best instruction set from those available:
            SSE4.2:   fallback, all modern x86 CPUs
            AVX2:     Intel Haswell+ and AMD Ryzen+ (FMA, 256-bit vectors)
            AVX-512:  Intel Skylake Scalable+ and Core i9 (512-bit vectors)
            AVX-512 VNNI: Cascade Lake+ and Ice Lake+ (INT8 dot product in 1 cycle)
            BF16:     Cooper Lake Xeon and Tiger Lake+ (native BF16 matmul)
            AMX-INT8: Sapphire Rapids Xeon (tile-based INT8, 8192 MACs/cycle)
            AMX-BF16: Sapphire Rapids Xeon (tile-based BF16)

    AMX (Advanced Matrix Extensions) — the critical Xeon advantage:
        AMX operates on 16×64 byte TILES stored in dedicated tile registers.
        AMX TMUL instruction: multiply two 16×64 INT8 tiles → 16×16 INT32 result.
        Throughput: one AMX TMUL per clock cycle.
        At 2.6 GHz with 2 AMX units: 2 × 16×16 × 2.6 GHz ≈ 1.3 TOPS INT8.
        vs AVX-512 VNNI: ~0.6 TOPS INT8 peak on same CPU.
        OpenVINO automatically uses AMX for MatMul and Conv2D when available.

    oneDNN PRIMITIVE TYPES (what the CPU plugin compiles to):
        matmul:        inner product primitive (GEMM → selected BLAS)
        convolution:   im2col + GEMM, or Winograd, or direct convolution
        pooling:       vectorised loop with SIMD
        eltwise:       fused elementwise chain (Add+ReLU compiled as one loop)
        layer_norm:    fused normalise + scale + shift via SIMD
        concat:        memory copy with optional format conversion
        binary:        element-wise binary op with broadcast support

    CPU PLUGIN CONFIGURATION:
        core.compile_model(model, "CPU", {
            "NUM_STREAMS":    "AUTO",         # number of parallel inference streams
            "INFERENCE_NUM_THREADS": 4,       # threads per stream
            "PERFORMANCE_HINT": "THROUGHPUT", # or "LATENCY", "CUMULATIVE_THROUGHPUT"
            "ENABLE_CPU_PINNING": True,       # pin threads to cores
        })

    NUMA awareness:
        On dual-socket Xeon systems, OpenVINO's CPU plugin is NUMA-aware.
        It partitions the model's computation across sockets to minimise
        cross-socket memory traffic (remote DRAM access is 2–3× slower).
        Configuration: core.compile_model(model, "CPU:0", config) for socket 0.

### The GPU Plugin (Intel Arc / iGPU)

    The GPU plugin targets Intel integrated and discrete GPUs via OpenCL / SYCL.

    SUPPORTED HARDWARE:
        Intel UHD Graphics (Gen9–12, in 6th–11th gen Core)
        Intel Iris Xe (Gen12.1, in 11th–12th gen Core)
        Intel Arc Alchemist (Xe-HPG, A-series discrete GPU)
        Intel Arc Battlemage (Xe2, B-series discrete GPU)
        Intel Data Center GPU Flex (server inference, Xe-HPC derivative)

    GPU PLUGIN COMPILATION:
        OpenCL kernels are JIT-compiled at compile_model() time.
        A kernel cache is maintained in ~/.cache/intel/openvino/:
            First compile: 5–30 seconds (OpenCL JIT compilation)
            Subsequent compiles of same model on same GPU: < 1 second

    PRECISION MODES:
        FP32: full precision (default if model is FP32)
        FP16: OpenVINO auto-converts FP32 models to FP16 on GPU by default
              (GPU FP16 is typically 2× faster than FP32 and usually safe)
        INT8: quantised models use INT8 kernels on GPU
        Disable FP16 auto-conversion:
            core.compile_model(model, "GPU",
                               {"INFERENCE_PRECISION_HINT": "f32"})

    GPU MEMORY MANAGEMENT:
        The GPU plugin manages memory allocation on the GPU device.
        Input data passed as CPU numpy arrays is automatically uploaded.
        For zero-copy GPU inference from camera/video:
            Use USM (Unified Shared Memory) tensors that are accessible
            from both CPU and GPU without explicit copies.
            (Requires Intel GPU with Shared Virtual Memory support.)

    GPU PLUGIN PERFORMANCE HINTS:
        "LATENCY":     optimise for minimum latency (batch=1, fewer streams)
        "THROUGHPUT":  optimise for maximum images/second (multiple batches,
                       pipelined execution, async queue)

### The NPU Plugin (Intel Neural Processing Unit)

    The NPU is a fixed-function, dataflow-style hardware accelerator
    specialised for transformer and CNN inference at very low power.

    AVAILABLE NPUs:
        Intel Meteor Lake (Core Ultra Gen 1):  ~10 TOPS INT8, ~3.7W
        Intel Lunar Lake (Core Ultra Gen 2):   ~48 TOPS INT8, included in 8W total
        Intel Arrow Lake (Core Ultra Gen 2 desktop): ~13 TOPS INT8

    HOW THE NPU DIFFERS FROM CPU AND GPU:
        The NPU is NOT programmable in general (no CUDA-equivalent).
        It executes a fixed hardware pipeline of matrix ops.
        Input: a compiled "blob" — a binary describing the dataflow graph.
        The OpenVINO NPU plugin compiles OV IR into this blob.
        The blob runs entirely in hardware with deterministic latency.
        No kernel launch overhead. No OS scheduling interference.
        VERY low power: inference at 1–3W (vs 15–45W for CPU inference).

    NPU CONSTRAINTS:
        Only a subset of operations supported natively:
            Supported:  Conv2D, DepthwiseConv, MatMul, Softmax, LayerNorm,
                        most elementwise ops, reshape, transpose, gather
            Fallback to CPU: complex ops (some scatter patterns, control flow)
        Static shapes STRONGLY preferred (recompile needed for shape change).
        Maximum model size: determined by NPU's on-chip SRAM (~50–100 MB).
        BF16 and INT8 are native; FP32 requires conversion.

    COMPILE AND USE:
        compiled_npu = core.compile_model(model, "NPU", {
            "PERFORMANCE_HINT": "LATENCY",    # NPU excels at low latency
        })
        # NPU compilation saves a blob cache to avoid re-compiling:
        # Cache location: set via CACHE_DIR property

    NPU CACHING (critical for startup time):
        First NPU compile: 10–60 seconds (AOT compilation to blob).
        Cached: < 1 second.
        core.set_property("NPU", {"CACHE_DIR": "./npu_cache"})
        # Subsequent runs with the same model load the pre-compiled blob.

### The AUTO and MULTI Plugins

    AUTO PLUGIN:
        Automatically selects the best available device for the model.
        Priority order: NPU → GPU → CPU (unless overridden).
        Warm-up: compiles for CPU first (fast), starts inference immediately.
        In parallel: compiles for the preferred device (slower but optimal).
        Switches to the better device when compilation finishes.
        For applications that cannot wait for NPU/GPU compile on startup.

        compiled = core.compile_model(model, "AUTO",
                                       {"PERFORMANCE_HINT": "LATENCY"})
        # Immediately usable on CPU; silently switches to NPU when ready.

    MULTI PLUGIN:
        Routes inference requests to MULTIPLE devices simultaneously.
        Useful when one device is not fast enough for the required throughput.
        Splits the inference stream: some requests go to CPU, some to GPU.
        Load balancing is automatic (each device gets work proportional to speed).

        compiled = core.compile_model(model, "MULTI:CPU,GPU",
                                       {"PERFORMANCE_HINT": "THROUGHPUT"})
        # CPU + GPU work in parallel; MULTI balances the load between them.

    HETERO PLUGIN:
        Routes DIFFERENT OPS in the same model to DIFFERENT devices.
        Similar to ONNX Runtime's EP partitioning:
        If NPU doesn't support an op, it falls back to CPU for that op.

        compiled = core.compile_model(model, "HETERO:NPU,CPU")
        # NPU-supported ops → NPU
        # Unsupported ops → CPU fallback
        # Data copies inserted at device boundaries (same as ORT EP graph partitioning)


##### PART 6 — NNCF: NEURAL NETWORK COMPRESSION FRAMEWORK

### What NNCF Is

    NNCF (Neural Network Compression Framework) is OpenVINO's companion
    library for model optimisation. It provides:
        Post-Training Quantisation (PTQ): INT8 with calibration data
        Quantisation-Aware Training (QAT): INT8 with fake-quantise nodes
        Structured Pruning: remove entire filter channels by importance
        Unstructured Sparsity: zero out individual weights
        GPTQ: quantisation for LLMs (weight-only INT4/INT8)
        Mixed Precision: assign different precision per layer

    NNCF works at the PyTorch model level (before conversion to OV IR).
    This ensures the optimisation is framework-aware and produces accurate
    quantised graphs before OpenVINO sees them.

    Install: pip install nncf

### Post-Training Quantisation (PTQ)

    NNCF's PTQ is the most common workflow: take a trained float model,
    run calibration data through it to measure activation ranges, then
    produce a quantised model that OpenVINO can run with INT8 kernels.

    WORKFLOW:
        import nncf, openvino as ov

        # Step 1: define calibration dataset (iterator of input dicts)
        def calibration_dataset():
            for image_batch in val_loader:
                yield {"input": image_batch.numpy()}

        # Step 2: load the float model
        core     = ov.Core()
        ov_model = core.read_model("model.xml")

        # Step 3: quantise with NNCF
        calibration_ds = nncf.Dataset(calibration_dataset())
        quantised_model = nncf.quantize(
            model           = ov_model,
            calibration_dataset = calibration_ds,
            preset          = nncf.QuantizationPreset.PERFORMANCE,  # or MIXED
            target_device   = nncf.TargetDevice.CPU,  # or GPU, NPU, VPU
            subset_size     = 300,     # number of calibration samples
        )

        # Step 4: save and compile
        ov.save_model(quantised_model, "model_int8.xml")
        compiled = core.compile_model(quantised_model, "CPU")

    QUANTISATION PRESETS:
        nncf.QuantizationPreset.PERFORMANCE:
            Quantises all layers aggressively (INT8 weights + activations).
            Best throughput. May have up to 1% accuracy drop.
            Use when maximum speed matters.

        nncf.QuantizationPreset.MIXED:
            Quantises weights to INT8 but activations to FP16 where needed.
            Better accuracy (closer to float baseline).
            Slightly less throughput than PERFORMANCE.
            Use when accuracy is the priority.

    TARGET DEVICE:
        nncf.TargetDevice.CPU:    targets oneDNN INT8 kernels on x86
        nncf.TargetDevice.GPU:    targets OpenCL INT8 kernels on Intel GPU
        nncf.TargetDevice.NPU:    targets Intel NPU INT8 dataflow
        nncf.TargetDevice.ANY:    device-agnostic (no device-specific tuning)
        The target device affects the quantisation granularity and
        which layers are quantised (some devices have constraints).

    ACCURACY-AWARE QUANTISATION:
        Sometimes INT8 quantisation drops accuracy beyond acceptable limits.
        NNCF's accuracy-aware mode automatically finds layers that are
        sensitive to quantisation and reverts them to FP32:

        from nncf.quantization import QuantizationMode, AccuracyRestorationAlgorithm

        quantised_model = nncf.quantize_with_accuracy_control(
            model               = ov_model,
            calibration_dataset = calibration_ds,
            validation_dataset  = validation_ds,
            validation_fn       = eval_fn,           # returns accuracy metric
            max_drop            = 0.01,              # tolerate up to 1% drop
        )
        # NNCF selects which layers to revert to FP32 to meet max_drop.
        # Produces a mixed FP32/INT8 model with minimal accuracy loss.

### Quantisation Internals: FakeQuantize Nodes

    NNCF represents quantisation symbolically via FakeQuantize nodes.
    A FakeQuantize node appears in the OV IR as:

        FakeQuantize (in_low, in_high, out_low, out_high, levels=256)
            = Quantize(Dequantize(x, in_low, in_high))

    The semantics:
        1. Scale input x to [0, levels-1]:
               q = round( (x - in_low) / (in_high - in_low) × (levels-1) )
        2. Clamp to [0, levels-1].
        3. Dequantise back to float:
               x_q = q / (levels-1) × (out_high - out_low) + out_low

    At inference time (compile_model), OpenVINO fuses adjacent FakeQuantize
    nodes into actual INT8 kernels. The FakeQuantize is not executed at runtime
    — it is absorbed into the preceding Conv or MatMul as INT8 accumulation.

    CALIBRATION ALGORITHM — how in_low, in_high are determined:
        MinMax: in_low = min(activations), in_high = max(activations)
        Simple, fast, may be affected by outliers.

        Percentile (default): in_low = 0.1th percentile,
                              in_high = 99.9th percentile
        More robust to outliers. Better INT8 accuracy.

        BiasCorrection: after quantisation, shifts bias values to correct
                        the mean shift introduced by quantisation error.
        Typically adds ~0.3–0.5% accuracy back.

### Structured Pruning (Filter Pruning)

    Structured pruning removes entire output CHANNELS (filters) from
    Conv2D/Linear layers whose importance score falls below a threshold.

    Unlike unstructured sparsity (zeroing individual weights), structured
    pruning reduces the actual computation: a Conv layer with 64 channels
    pruned to 48 channels genuinely runs faster (48/64 = 75% of the FLOPs).

    NNCF pruning workflow:
        import nncf
        from nncf.torch import create_compressed_model
        from nncf.config import NNCFConfig

        config = NNCFConfig.from_dict({
            "compression": [{
                "algorithm": "filter_pruning",
                "params": {
                    "prune_first_conv": True,
                    "prune_last_conv":  True,
                    "schedule":  "exponential",
                    "pruning_init": 0.1,   # start at 10% pruning
                    "pruning_target": 0.4, # target 40% pruning
                    "num_init_steps": 100,
                }
            }]
        })
        compressed_model, compression_ctrl = create_compressed_model(
            model, config)

        # Fine-tune (important: pruning needs fine-tuning to recover accuracy)
        for epoch in range(fine_tune_epochs):
            for batch in train_loader:
                output = compressed_model(batch)
                loss   = criterion(output, labels)
                loss.backward()
                optimizer.step()
                compression_ctrl.scheduler.step()   # update pruning masks

        # Convert to OpenVINO IR (pruned structure is baked in)
        compressed_model.eval()
        ov_model = ov.convert_model(compressed_model, example_input=sample)

    Typical pruning results:
        ResNet-50, 40% pruned:  83% of FLOPs, 79.7% top-1 (vs 76.1% baseline)
        (NNCF recovers accuracy beyond baseline due to fine-tuning regularisation)

### GPTQ / Weight-Only Quantisation for LLMs

    For large language models, activation quantisation causes severe accuracy
    drops. NNCF supports WEIGHT-ONLY quantisation (GPTQ and AWQ variants):

        compressed_model = nncf.compress_weights(
            ov_model,
            mode          = nncf.CompressWeightsMode.INT4_ASYM,   # 4-bit asymmetric
            group_size    = 128,   # quantise in groups of 128 weights
            ratio         = 0.8,  # 80% of layers at INT4, 20% at INT8 (mixed)
        )

    Modes:
        INT8_ASYM:  asymmetric INT8 (per-channel scale + zero_point). ~8GB for 7B LLM.
        INT8_SYM:   symmetric INT8 (per-channel scale only). Slightly less accurate.
        INT4_ASYM:  4-bit weights, group quantisation. ~4GB for 7B LLM.
        INT4_SYM:   4-bit symmetric. Most compressed. Lowest accuracy.
        NF4:        NormalFloat4 (like GGUF NF4). Optimal for normally-distributed weights.

    Only WEIGHTS are quantised (activations remain FP16/FP32).
    This gives 2–4× memory reduction with < 1% accuracy loss on most LLMs.
    Inference: weights dequantised on-the-fly to FP16 for each matmul.


##### PART 7 — PERFORMANCE CONFIGURATION AND BENCHMARKING

### Performance Hints: The Recommended Approach

    OpenVINO's preferred way to configure performance is via PERFORMANCE_HINT:
    a high-level intent that the plugin translates into the right combination
    of thread counts, batch sizes, and queue depths automatically.

    LATENCY:
        Optimise for minimum time from input to output for a single request.
        Inference is synchronous; model sees one request at a time.
        Plugin uses: 1 stream, all threads serving that stream, no batching.
        Best for: interactive applications, edge real-time processing.
        Usage:
            compiled = core.compile_model(model, "CPU",
                                          {"PERFORMANCE_HINT": "LATENCY"})

    THROUGHPUT:
        Optimise for maximum images/second for a stream of requests.
        Plugin uses: multiple streams, async queue, optional batching.
        Best for: batch video processing, cloud API endpoints.
        Usage:
            compiled = core.compile_model(model, "CPU",
                                          {"PERFORMANCE_HINT": "THROUGHPUT"})

    CUMULATIVE_THROUGHPUT:
        A variant of THROUGHPUT that uses more streams.
        Better when combined with MULTI or AUTO (multiple devices).
        Each device contributes to the total throughput independently.

### Streams and Threads: What They Are

    OpenVINO CPU STREAM: an independent parallel inference "lane."
    Each stream has its own oneDNN execution context, thread pool,
    and memory buffers for input/output tensors.

    STREAMS × THREADS = CPU CORES:
        If the CPU has 8 physical cores:
            1 stream × 8 threads:   latency-optimal (one request uses all 8 cores)
            4 streams × 2 threads:  throughput-optimal (4 requests run in parallel)
            8 streams × 1 thread:   maximum parallelism (8 requests simultaneously)

    The optimal stream/thread split depends on model size:
        Large model (ResNet-50, BERT-Base): LATENCY mode wins (1 stream)
        Small model (MobileNet, small FC):  THROUGHPUT mode wins (many streams)

    Querying the recommended configuration:
        compiled = core.compile_model(model, "CPU",
                                      {"PERFORMANCE_HINT": "THROUGHPUT"})
        n_streams  = compiled.get_property("NUM_STREAMS")
        n_infer_rs = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        print(f"Plugin chose: {n_streams} streams, "
              f"{n_infer_rs} optimal InferRequests")

### Benchmarking with the Benchmark App

    OpenVINO ships a command-line benchmark tool:

        benchmark_app -m model.xml -d CPU -hint latency -t 10
        # -m: model path  -d: device  -hint: performance hint  -t: duration (s)

        # Example output:
        # [ INFO ] First inference took 12.34 ms
        # Count:       500 iterations
        # Duration:    10234.56 ms
        # Latency:
        #   Median:    20.23 ms
        #   Average:   20.46 ms
        #   Min:       18.92 ms
        #   Max:       28.31 ms
        # Throughput:  48.89 FPS

    Benchmark flags:
        -api async          ; use async inference (for THROUGHPUT measurement)
        -b 8                ; batch size
        -niter 1000         ; number of iterations
        -nireq 4            ; number of parallel InferRequests
        -shape [1,3,224,224]; override input shape
        -data_type FP16     ; input data type

    Python benchmark (programmatic):
        from openvino.tools.benchmark import benchmarking_app
        result = benchmarking_app.main(["-m", "model.xml", "-d", "CPU",
                                        "-hint", "throughput", "-t", "5"])

### FP16 Compression and BF16 Inference

    FP16 WEIGHT COMPRESSION (save-time):
        ov.save_model(model, "model.xml", compress_to_fp16=True)
        Reduces model.bin size by 2× (float32 → float16 weights).
        At compile time: CPU plugin may use FP32 for actual computation
        but stores weights in FP16 to save memory bandwidth.
        No accuracy loss for most models (weights are quantised anyway
        to FP16 precision which is sufficient for inference).

    BF16 INFERENCE (runtime):
        On Intel CPUs with native BF16 support (Tiger Lake, Alder Lake, Xeon):
        core.compile_model(model, "CPU",
                           {"INFERENCE_PRECISION_HINT": "bf16"})
        BF16 has the same exponent range as FP32 (5 bits) but only 7 mantissa bits.
        Less overflow risk than FP16 (16 bits, 5-bit exponent).
        Typical accuracy: within 0.1% of FP32 for most vision/NLP models.
        Speed: 2× FP32 on BF16-capable CPUs (native BF16 multiply in AVX-512 BF16).


##### PART 8 — OPENVINO MODEL SERVER (OVMS): PRODUCTION SERVING

### What OVMS Is

    OpenVINO Model Server (OVMS) is an Intel-maintained production model
    serving infrastructure built on top of the OpenVINO runtime. It exposes:
        gRPC API:  TensorFlow Serving-compatible protocol (protobuf)
        REST API:  HTTP/JSON (also TF Serving-compatible format)
        OpenAI API: /v1/chat/completions endpoint for LLM serving

    OVMS is containerised:
        docker run -d --rm --name ovms \\
            -v /path/to/models:/models \\
            -p 9000:9000 -p 9001:9001 \\
            openvino/model_server:latest \\
            --model_path /models/resnet50 \\
            --model_name resnet50 \\
            --port 9000 --rest_port 9001

    OVMS is used in:
        Intel DevCloud inference endpoints
        Azure Machine Learning (OVMS as inference container)
        KServe (Kubernetes-based model serving)
        Edge deployments on Intel NUC / industrial PCs

### Model Repository Layout

    OVMS expects models in a specific directory structure:

        model_repository/
            resnet50/
                1/              ; version 1
                    model.xml
                    model.bin
                2/              ; version 2 (overrides 1 when traffic is migrated)
                    model.xml
                    model.bin
            bert_base/
                1/
                    model.onnx  ; OVMS can also serve ONNX directly
            gpt2/
                1/
                    model.xml
                    model.bin

    Version management:
        OVMS loads all version directories by default.
        Traffic routing to a version: config.json or --target_model_version flag.
        Rolling update: deploy v2 → validate → shift traffic → delete v1.

### Sending Requests to OVMS

    gRPC (Python client):
        import tritonclient.grpc as grpcclient   ; same protocol as Triton
        # OR: use tensorflow-serving-api:
        from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
        import grpc, numpy as np

        channel = grpc.insecure_channel("localhost:9000")
        stub    = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name    = "resnet50"
        request.model_spec.version.value = 1
        request.inputs["input"].CopyFrom(
            tf.make_tensor_proto(image_np, shape=image_np.shape))

        response = stub.Predict(request, timeout=10)
        result   = np.array(response.outputs["logits"].float_val)

    REST (curl):
        curl -X POST http://localhost:9001/v1/models/resnet50:predict \\
             -H "Content-Type: application/json" \\
             -d '{"instances": [{"input": [[[...pixel values...]]]}]}'

    OpenAI API compatible (for LLMs):
        curl http://localhost:9001/v1/chat/completions \\
             -H "Content-Type: application/json" \\
             -d '{"model": "llama3", "messages": [{"role": "user", "content": "Hello"}]}'

### DAG Pipelines in OVMS

    OVMS supports DIRECTED ACYCLIC GRAPH (DAG) PIPELINES:
    multiple models chained together, with the output of one feeding the input
    of the next — entirely server-side, with no network round-trips between steps.

    Example pipeline for OCR:
        detect_text_regions (detection model)
            → crop_and_resize (OpenCV preprocessing)
            → classify_characters (classification model)
            → postprocess_to_string

    DAG configuration (config.json):
        "pipeline_config_list": [{
            "name": "ocr_pipeline",
            "inputs": [{"name": "image", "mapping": "detect_text.input"}],
            "nodes": [
                {"name": "detect_text",  "model_name": "text_detector", ...},
                {"name": "crop_resize",  "library": "custom_lib.so",    ...},
                {"name": "read_chars",   "model_name": "ocr_model",     ...}
            ],
            "outputs": [{"name": "text", "mapping": "read_chars.output"}]
        }]

    Performance advantage:
        All models run in the same process.
        Tensor buffers are shared in-process (no serialisation overhead).
        Pipeline throughput matches the slowest stage.
        Custom C++ "calculators" (similar to MediaPipe) can be inserted
        between model nodes for preprocessing.


##### PART 9 — OPENVINO GENAI: LLM INFERENCE

### What OpenVINO GenAI Is

    OpenVINO GenAI is a high-level Python/C++ library built on top of the
    OpenVINO runtime for running language model inference efficiently.
    It handles the entire LLM inference pipeline:
        Tokenisation (using OpenVINO Tokenizers)
        Prefill (processing the full prompt)
        Decode (auto-regressive token generation)
        KV-cache management (stateful between decode steps)
        Beam search, top-k, top-p, temperature sampling
        Speculative decoding (draft + verification model)
        Continuous batching (different-length sequences simultaneously)
        LoRA adapter support

    Install:  pip install openvino-genai openvino-tokenizers

### Exporting LLMs for OpenVINO GenAI

    The recommended export path via Optimum-Intel (HuggingFace integration):

        pip install optimum[openvino]

        # Export to OpenVINO IR with INT4 weight compression:
        optimum-cli export openvino \\
            --model meta-llama/Llama-3-8B-Instruct \\
            --weight-format int4 \\
            --ratio 1.0 \\
            --group-size 128 \\
            ./llama3_8b_int4_ov/

        # The output directory contains:
        #   openvino_model.xml / .bin     (the LLM)
        #   openvino_tokenizer.xml / .bin (tokenizer as OV graph)
        #   tokenizer_config.json
        #   generation_config.json

    Weight format options:
        --weight-format fp32:    full precision (largest, most accurate)
        --weight-format fp16:    FP16 weights (~2× smaller)
        --weight-format int8:    INT8 weight-only (~4× smaller, slight accuracy drop)
        --weight-format int4:    INT4 weight-only (~8× smaller, tested on Llama/Mistral/Phi)
        --weight-format nf4:     NormalFloat4 (same size as INT4, slightly better accuracy)

    Memory requirements (Llama-3-8B):
        FP32: ~32 GB     (requires server-class hardware)
        FP16: ~16 GB     (Intel Arc A770 16 GB, or CPU with 32 GB RAM)
        INT8: ~8 GB      (fits on most AI PCs with 16 GB RAM on CPU)
        INT4: ~4 GB      (fits on Meteor Lake NPU + CPU combined)

### Running LLMs with OpenVINO GenAI

    SIMPLE GENERATION:
        import openvino_genai as ov_genai

        pipe = ov_genai.LLMPipeline("./llama3_8b_int4_ov/", "CPU")

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 200
        config.temperature    = 0.7
        config.top_p          = 0.9

        result = pipe.generate("Explain quantum entanglement:", config)
        print(result)

    STREAMING GENERATION (token-by-token):
        streamer = lambda token: print(token, end="", flush=True)

        pipe.generate("Write a Python function to sort a list:",
                      config, streamer)

    CHAT INTERFACE (with system prompt):
        pipe.start_chat()
        response = pipe.chat("What is OpenVINO?")
        response = pipe.chat("How does it compare to TensorRT?")
        pipe.finish_chat()

    MULTI-GPU (Arc + CPU fallback):
        pipe = ov_genai.LLMPipeline("./llama3_8b_int4_ov/", "GPU")
        # Falls back to CPU for unsupported ops (HETERO plugin under the hood)

### KV-Cache Management

    Auto-regressive LLM generation accumulates KEY and VALUE tensors
    for every input token in every attention layer. This KV-cache grows
    with each generated token and must persist between generation steps.

    OpenVINO GenAI handles KV-cache as STATEFUL TENSORS:
        The LLM model is exported with KV-cache as ReadValue/Assign op pairs.
        ReadValue reads the previous KV state from a persistent buffer.
        Assign writes the new KV state back to that buffer.
        The buffer is maintained across Invoke() calls automatically.

    KV-cache size estimate:
        seq_len × n_layers × 2 (K and V) × n_heads × head_dim × bytes_per_element
        Llama-3-8B (32 layers, 8 heads, 128 head_dim, 512 tokens, FP16):
            512 × 32 × 2 × 8 × 128 × 2 bytes = ~0.5 GB

    PAGED ATTENTION (OpenVINO 2024.3+):
        For long contexts and batched generation, OpenVINO GenAI supports
        paged KV-cache (inspired by vLLM's PagedAttention):
        KV-cache is divided into fixed-size "pages" allocated on demand.
        This eliminates memory fragmentation for variable-length sequences.
        Enables continuous batching: different requests share KV-cache pages.

### Speculative Decoding

    Speculative decoding uses a SMALL DRAFT MODEL to propose multiple tokens,
    then verifies them in parallel with the LARGE TARGET MODEL.

    Algorithm:
        1. Draft model generates N candidate tokens (fast, small model).
        2. Target model verifies all N tokens in ONE forward pass (parallel).
        3. If draft tokens are accepted: use them (N tokens generated at cost of 1 step).
        4. If a draft token is rejected: discard it and resample from target.

    Performance gain:
        When most draft tokens are accepted: ~3–4× throughput improvement.
        The target model runs the same total FLOPs but generates more tokens per step.

    OpenVINO GenAI speculative decoding:
        main_pipe  = ov_genai.LLMPipeline("./llama3_70b_int4_ov/", "CPU")
        draft_pipe = ov_genai.LLMPipeline("./llama3_8b_int4_ov/",  "CPU")

        config = ov_genai.GenerationConfig()
        config.num_assistant_tokens = 5   # draft model proposes 5 tokens

        # Speculative decoding: draft proposes, main verifies
        result = ov_genai.SpeculativeDecodingPipeline(
            main_pipe, draft_pipe).generate("Translate to French:", config)


##### PART 10 — OPENVINO EXTENSIONS AND CUSTOM OPS

### Why Custom Ops Are Needed

    OpenVINO supports ~250 built-in operations. New model architectures
    constantly introduce ops not yet in the OV op set:
        Rotary Position Embeddings (RoPE) — LLaMA, Mistral, Phi
        FlashAttention-2 — faster attention for long sequences
        KV-cache MHA variants — PagedAttention, Grouped Query Attention
        Custom normalisation layers
        Domain-specific signal processing ops

    Two mechanisms exist to handle these:

### Mechanism 1: OpenVINO Extension (C++ Plugin)

    A C++ Extension implements:
        a) An OP CLASS: the operation's semantics (shape inference, type inference)
        b) A KERNEL CLASS: the operation's computation (CPU implementation)

    STEP 1 — Define the op (C++):
        class CustomRoPE : public ov::op::Op {
        public:
            OPENVINO_OP("CustomRoPE", "user_extension");
            void validate_and_infer_types() override {
                set_output_type(0, get_input_element_type(0),
                                get_input_partial_shape(0));
            }
        };

    STEP 2 — Implement the kernel (C++):
        class CustomRoPEKernel : public ov::OpKernel {
            bool execute(ov::KernelContext& ctx) override {
                // access inputs via ctx.get_input(0), etc.
                apply_rope(ctx.get_input(0), ctx.get_output(0));
                return true;
            }
        };

    STEP 3 — Register (C++):
        ov::Extension ext = ov::Extension(std::make_shared<CustomRoPEKernel>());

    STEP 4 — Use in Python:
        core.add_extension("libcustom_rope.so")
        ov_model = core.read_model("model_with_rope.xml")
        # "CustomRoPE" nodes are now resolved

### Mechanism 2: Python Custom Op

    For prototyping and research, custom ops can be written in Python:

        from openvino.runtime import Op
        import numpy as np

        class SwiGLU(Op):
            class_type_info = ov.runtime.DiscreteTypeInfo("SwiGLU", "user")

            def __init__(self, x: ov.Output):
                super().__init__([x])
                self.constructor_validate_and_infer_types()

            def validate_and_infer_types(self):
                self.set_output_type(0, self.get_input_element_type(0),
                                     self.get_input_partial_shape(0))

            def evaluate(self, outputs, inputs):
                x = inputs[0].data
                # SwiGLU: split in half, apply swish to one half, multiply
                a, b = np.split(x, 2, axis=-1)
                outputs[0].data[:] = a * (b * np.exp(b)) / (1 + np.exp(b))
                return True

            def has_evaluate(self): return True

### Using Python Extensions for Frontend Conversion

    During model conversion (ov.convert_model), unknown PyTorch ops can
    be handled via CONVERSION EXTENSIONS:

        from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
        from openvino import frontend as ov_frontend

        # Register how to convert a custom PyTorch op to OV IR
        @ov_frontend.ConversionExtension("torch::custom_silu")
        def custom_silu_converter(node, *args, **kwargs):
            x = node.input(0)
            return ov.opset13.multiply(x, ov.opset13.sigmoid(x))

        core.add_extension(custom_silu_converter)
        ov_model = ov.convert_model(torch_model, example_input=sample)
        # "torch::custom_silu" nodes are now converted to Multiply+Sigmoid


##### PART 11 — OPENVINO IN THE CONNECTED COMPILER STACK

### OpenVINO and LLVM (Module 01)

    OpenVINO's CPU plugin is built on oneDNN (formerly MKL-DNN).
    oneDNN uses LLVM-compiled intrinsics for SIMD kernels:
        LLVM compiles AVX-512 VNNI code paths for INT8 matmul.
        LLVM compiles AVX2/AVX-512 FP32/FP16/BF16 kernels.
        At runtime: CPUID selects the best pre-compiled variant.
    OpenVINO itself is compiled with LLVM (Clang) for Intel hardware.

### OpenVINO and MLIR (Module 02)

    OpenVINO uses MLIR internally in parts of the compiler pipeline:
        The nGraph → OV IR lowering passes use MLIR infrastructure.
        The GPU plugin uses MLIR-based kernel generation for some ops.
        Future direction: OpenVINO is converging toward MLIR for its
        complete compilation pipeline (similar to TFLite's converter).
    The OV IR's operation versioning and region-based control flow ops
    are conceptually similar to MLIR's dialect and region system.

### OpenVINO and XLA/OpenXLA (Modules 05–06)

    OpenVINO and XLA/OpenXLA serve overlapping use cases on Intel hardware.
    JAX models compiled by XLA for CPU use different kernels than OV.
    XLA's oneDNN plugin (libxla_onednn.so) uses the SAME oneDNN library
    as OpenVINO's CPU plugin — they are direct competitors for the same silicon.
    For Intel CPU deployment of JAX models: OV often wins on throughput
    because it can use NNCF quantisation and oneDNN INT8 more aggressively.
    For TPU: XLA has no competition; OV does not target TPU at all.

### OpenVINO and TVM (Module 08)

    TVM and OpenVINO are direct competitors on Intel x86 CPU:
        TVM's MetaSchedule finds optimal tile sizes for each (M,N,K) shape.
        OpenVINO's oneDNN uses hand-tuned kernels per instruction set.
    Empirically: oneDNN wins for standard shapes (powers of 2),
    TVM sometimes wins for unusual shapes (prime-number dimensions).
    For edge IoT deployment: OV wins because it supports Intel Myriad VPU
    and Intel NPU — no TVM backend for these targets exists.

### OpenVINO and ONNX (Module 10)

    ONNX is OpenVINO's primary import format and is the most tested path.
    OpenVINO's ONNX frontend maps ONNX ops → OV IR ops.
    Every ONNX opset (1–20) is supported; new ONNX opsets are added
    within months of the ONNX release.
    OpenVINO can also EXPORT to ONNX (for cross-tool interoperability):
        import openvino as ov
        # Internally uses OV → ONNX exporter:
        ov_model = ov.compile_model(...)  # already optimised
        # For saving: save as OV IR (not ONNX), since OV-specific fusions
        # may not round-trip through ONNX correctly.

### OpenVINO and TFLite (Module 11)

    OpenVINO's TFLite frontend enables x86 deployment of TFLite models:
        ov_model = ov.convert_model("model.tflite")
        compiled  = core.compile_model(ov_model, "CPU")
    This is useful when a model was developed for Android deployment
    (TFLite) but needs to run on an Intel edge server.
    INT8 quantised TFLite models preserve their quantisation through
    conversion and run with INT8 kernels on OV CPU.

### The Complete OpenVINO Ecosystem

    ┌──────────────────────────────────────────────────────────────────────┐
    │  SOURCE FRAMEWORKS                                                   │
    │  PyTorch   TensorFlow   Keras   JAX/ONNX   TFLite   PaddlePaddle     │
    │  (torch.export)(SavedModel)(KerasModel)(ONNX file)(.tflite)(.pdmodel)│
    └──────────────────────────┬───────────────────────────────────────────┘
                               │
                  ov.convert_model() — MLIR-based frontend
                  + FP16 weight compression + NNCF quantisation
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  OpenVINO IR (.xml + .bin)                                          │
    │  FP32 / FP16 / INT8 / Mixed precision                               │
    └──────┬───────────────────┬──────────────────────────┬───────────────┘
           │                   │                          │
           ▼                   ▼                          ▼
    ┌─────────────┐  ┌────────────────────┐  ┌────────────────────────────┐
    │  CPU Plugin │  │   GPU Plugin       │  │   NPU Plugin               │
    │  oneDNN     │  │   OpenCL/SYCL      │  │   Intel NPU compiler       │
    │  SSE4→AMX   │  │   Iris Xe / Arc    │  │   Meteor/Lunar Lake NPU    │
    │  AVX512 VNNI│  │   FP32/FP16/INT8   │  │   INT8/BF16 dataflow       │
    └─────────────┘  └────────────────────┘  └────────────────────────────┘
           │
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PLUGINS: AUTO (best device) │ MULTI (all devices) │ HETERO (split) │
    ├─────────────────────────────────────────────────────────────────────┤
    │  TOOLS:   NNCF (INT8/INT4/pruning) │ Benchmark App │ OVMS (serving) │
    │           OpenVINO GenAI (LLMs)    │ Optimum-Intel (HF integration) │
    └─────────────────────────────────────────────────────────────────────┘

    COMPARISON WITH ALTERNATIVES:
    ┌──────────────────────┬───────────────┬──────────────┬──────────────┐
    │ Property             │  OpenVINO     │  ONNX Runtime│  TFLite      │
    ├──────────────────────┼───────────────┼──────────────┼──────────────┤
    │ Primary hardware     │ Intel CPU/GPU │ Cross-vendor │ Mobile/MCU   │
    │ Intel NPU support    │ YES (best)    │ Limited EP   │ No           │
    │ Myriad VPU support   │ YES           │ No           │ No           │
    │ INT8 quantisation    │ NNCF (best)   │ ORT quant    │ PTQ/QAT      │
    │ INT4 weight quant    │ YES (LLMs)    │ Limited      │ No           │
    │ LLM pipeline         │ GenAI         │ ORT-GenAI    │ MediaPipe    │
    │ Model serving        │ OVMS (native) │ Triton EP    │ MediaPipe    │
    │ HuggingFace integ.   │ Optimum-Intel │ Optimum-ORT  │ Limited      │
    │ Custom ops (Python)  │ YES           │ YES          │ C++ only     │
    │ Preprocessing fusion │ PrePostProc.  │ Partial      │ Limited      │
    └──────────────────────┴───────────────┴──────────────┴──────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · OpenVINO IR — Conversion, Inspection, and the XML/BIN Format": {
        "description": (
            "Convert models from PyTorch and ONNX to OpenVINO IR. "
            "Inspect the IR: walk all ops, print shapes, read attributes. "
            "Show the XML structure: layers, edges, and the BIN blob offsets. "
            "Demonstrate dynamic vs static shapes with PartialShape. "
            "Show FP16 weight compression and measure the size reduction. "
            "Validate numerical accuracy: OV output vs PyTorch reference."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os

print("=" * 65)
print("  OPENVINO IR — CONVERSION, INSPECTION, AND THE XML/BIN FORMAT")
print("=" * 65)
print()

try:
    import openvino as ov
    print(f"  OpenVINO {ov.__version__}")
    HAS_OV = True
except ImportError:
    HAS_OV = False
    print("  OpenVINO not installed: pip install openvino")
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
# SECTION 1: Convert a PyTorch model to OpenVINO IR
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Convert PyTorch model to OpenVINO IR")
print("━" * 65)
print()

if HAS_OV and HAS_TORCH:
    # Build a small ResNet-style model for demonstration
    class ConvBNReLU(nn.Module):
        def __init__(self, c_in, c_out, stride=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.block(x)

    class TinyResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = ConvBNReLU(3, 32, stride=2)    # 112×112
            self.body = nn.Sequential(
                ConvBNReLU(32, 64, stride=2),           # 56×56
                ConvBNReLU(64, 64),
                ConvBNReLU(64, 128, stride=2),          # 28×28
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes),
            )
        def forward(self, x):
            return self.head(self.body(self.stem(x)))

    pt_model = TinyResNet(10).eval()
    n_params  = sum(p.numel() for p in pt_model.parameters())
    example   = torch.randn(1, 3, 224, 224)

    print(f"  Model: TinyResNet ({n_params:,} params)")
    print(f"  Input: {list(example.shape)}, dtype=float32")
    print()

    # ── Convert to OpenVINO IR ────────────────────────────────────────────
    core = ov.Core()
    print("  Converting PyTorch model → OpenVINO IR...")
    t0 = time.perf_counter()
    ov_model = ov.convert_model(pt_model, example_input=example)
    t_conv = (time.perf_counter() - t0) * 1000
    print(f"  Conversion time: {t_conv:.1f} ms")
    print()

    # ── Inspect the IR graph ─────────────────────────────────────────────
    ops = list(ov_model.get_ordered_ops())
    print(f"  IR graph summary:")
    print(f"    Operations: {len(ops)}")
    print(f"    Inputs:   {[i.get_any_name() for i in ov_model.inputs]}")
    print(f"    Outputs:  {[o.get_any_name() for o in ov_model.outputs]}")
    print(f"    Input partial shape: {ov_model.inputs[0].get_partial_shape()}")
    print()

    # ── Print all operations with output shapes ──────────────────────────
    print(f"  Operation types in the IR:")
    from collections import Counter
    op_type_counts = Counter(op.get_type_name() for op in ops)
    for op_type, count in sorted(op_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {op_type:<35s} × {count}")
    print()

    # ── Show Conv operations with their attributes ───────────────────────
    print(f"  Convolution layers:")
    print(f"  {'Name':30s}  {'Output shape':22s}  {'Strides':8s}  {'Kernel'}")
    print("  " + "-" * 75)
    for op in ops:
        if op.get_type_name() == "Convolution":
            out_shape = str(op.output(0).get_partial_shape())
            try:
                strides = list(op.get_attribute("strides"))
                dilats  = list(op.get_attribute("dilations"))
                wshape  = list(op.input(1).get_partial_shape())
                ksize   = [wshape[2], wshape[3]] if len(wshape) >= 4 else "?"
            except Exception:
                strides, ksize = "?", "?"
            print(f"  {op.get_friendly_name()[:29]:30s}  "
                  f"{out_shape:22s}  {str(strides):8s}  {str(ksize)}")
    print()

    # ── Save to XML + BIN ────────────────────────────────────────────────
    import tempfile, pathlib
    tmpdir   = tempfile.mkdtemp()
    xml_path = os.path.join(tmpdir, "tiny_resnet.xml")
    bin_path = os.path.join(tmpdir, "tiny_resnet.bin")

    ov.save_model(ov_model, xml_path, compress_to_fp16=False)
    xml_size = os.path.getsize(xml_path)
    bin_size = os.path.getsize(bin_path)

    # Also save FP16 compressed version
    xml_fp16 = os.path.join(tmpdir, "tiny_resnet_fp16.xml")
    bin_fp16 = os.path.join(tmpdir, "tiny_resnet_fp16.bin")
    ov.save_model(ov_model, xml_fp16, compress_to_fp16=True)
    bin_fp16_size = os.path.getsize(bin_fp16)

    print(f"  Saved files:")
    print(f"    tiny_resnet.xml:       {xml_size/1024:.1f} KB  (graph structure)")
    print(f"    tiny_resnet.bin:       {bin_size/1024:.1f} KB  (FP32 weights)")
    print(f"    tiny_resnet_fp16.bin:  {bin_fp16_size/1024:.1f} KB  "
          f"(FP16 weights, {bin_size/bin_fp16_size:.2f}× smaller)")
    print()

    # ── Show XML structure snippet ────────────────────────────────────────
    with open(xml_path) as f:
        xml_lines = f.readlines()
    print(f"  XML structure (first 25 lines of {len(xml_lines)} total):")
    for line in xml_lines[:25]:
        print(f"  {line}", end="")
    print()
    if len(xml_lines) > 25:
        print(f"  ... ({len(xml_lines)-25} more lines)")
    print()

    # ── Reload and run inference ─────────────────────────────────────────
    model_reloaded = core.read_model(xml_path)
    compiled = core.compile_model(model_reloaded, "CPU")

    inp = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(inp)).numpy()

    # OpenVINO inference
    ov_out = compiled([inp])[compiled.output(0)]

    max_diff  = float(np.max(np.abs(pt_out - ov_out)))
    mean_diff = float(np.mean(np.abs(pt_out - ov_out)))
    print(f"  Numerical validation (PyTorch vs OpenVINO CPU):")
    print(f"    Max abs difference:  {max_diff:.2e}")
    print(f"    Mean abs difference: {mean_diff:.2e}")
    print(f"    Numerically equivalent: {'✅' if max_diff < 1e-4 else '⚠️'}")
    print()

else:
    CONV_REF = """
  OPENVINO CONVERSION API REFERENCE:
  ─────────────────────────────────────────────────────────────────
  import openvino as ov, torch, torch.nn as nn

  # ── From PyTorch (direct, uses torch.export internally) ──────────
  model    = MyModel().eval()
  example  = torch.randn(1, 3, 224, 224)
  ov_model = ov.convert_model(model, example_input=example)

  # ── From ONNX file ─────────────────────────────────────────────
  ov_model = ov.convert_model("model.onnx")

  # ── From TensorFlow SavedModel ─────────────────────────────────
  ov_model = ov.convert_model("/path/to/saved_model/")

  # ── From TFLite ────────────────────────────────────────────────
  ov_model = ov.convert_model("model.tflite")

  # ── Save to XML + BIN ──────────────────────────────────────────
  ov.save_model(ov_model, "model.xml", compress_to_fp16=True)

  # ── Inspect the IR ────────────────────────────────────────────
  for op in ov_model.get_ordered_ops():
      print(f"{op.get_type_name():30s} → {op.output(0).get_partial_shape()}")

  # ── Set dynamic batch ─────────────────────────────────────────
  ov_model.reshape({"input": ov.PartialShape([-1, 3, 224, 224])})
"""
    print(CONV_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Dynamic and static shapes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Dynamic vs static shapes: PartialShape API")
print("━" * 65)
print()

SHAPES_GUIDE = """
  SHAPES IN OPENVINO: PARTIAL SHAPES AND DIMENSION OBJECTS
  ════════════════════════════════════════════════════════════════

  PartialShape represents a tensor shape where some (or all) dimensions
  may be unknown or constrained to a range.

  CREATING PARTIAL SHAPES:
    ov.PartialShape([1, 3, 224, 224])     ; fully static — all dims known
    ov.PartialShape([-1, 3, 224, 224])   ; dynamic batch — batch is free
    ov.PartialShape([-1, 3, -1, -1])     ; dynamic H, W, and batch
    ov.PartialShape.dynamic()            ; completely unknown rank

  DIMENSION OBJECTS:
    ov.Dimension()            ; fully dynamic (any non-negative integer)
    ov.Dimension(4)           ; exactly 4
    ov.Dimension(1, 8)        ; range [1, 8] inclusive

    ov.PartialShape([ov.Dimension(1,16), 3, 224, 224])
    ; batch size between 1 and 16

  STATIC vs DYNAMIC COMPILATION:
  ─────────────────────────────────────────────────────────────────
  STATIC (fully specified shape):
    model.reshape({"input": [1, 3, 224, 224]})
    compiled = core.compile_model(model, "CPU")
    # Compiler knows all loop bounds → optimal tile sizes, no runtime overhead.
    # Faster inference. Recompile needed for different batch size.

  DYNAMIC (batch is symbolic):
    model.reshape({"input": ov.PartialShape([-1, 3, 224, 224])})
    compiled = core.compile_model(model, "CPU")
    # One compiled model works for batch=1, 2, 4, 8, 16 without recompilation.
    # Slightly less optimal (tile sizes must accommodate any batch).
    # Required for variable-length sequence models (NLP, audio).

  NPU CONSTRAINT:
    The Intel NPU ALWAYS requires static shapes.
    ov.PartialShape.dynamic() or -1 dims → NPU compilation error.
    Fix: set concrete shape before compiling for NPU:
      model.reshape({"input": [1, 3, 224, 224]})
      compiled_npu = core.compile_model(model, "NPU")

  QUERYING COMPILED SHAPE:
    for inp in compiled.inputs:
        print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")
    for out in compiled.outputs:
        print(f"  {out.get_any_name()}: {out.get_partial_shape()}")
"""
print(SHAPES_GUIDE)

if HAS_OV and HAS_TORCH:
    # Demonstrate static vs dynamic shape compilation
    print("  Demonstrating static vs dynamic shape compilation:")
    print()

    # Static shape
    static_model = ov.convert_model(pt_model, example_input=example)
    static_model.reshape({"x.1": [1, 3, 224, 224]})
    compiled_static = core.compile_model(static_model, "CPU")

    # Dynamic batch
    dyn_model = ov.convert_model(pt_model, example_input=example)
    try:
        input_name = dyn_model.inputs[0].get_any_name()
        dyn_model.reshape({input_name: ov.PartialShape([-1, 3, 224, 224])})
        compiled_dynamic = core.compile_model(dyn_model, "CPU")

        # Test with different batch sizes
        print(f"  Dynamic batch model: testing multiple batch sizes:")
        for bs in [1, 2, 4, 8]:
            x_bs = np.random.rand(bs, 3, 224, 224).astype(np.float32)
            out  = compiled_dynamic([x_bs])[compiled_dynamic.output(0)]
            print(f"    batch={bs}: input {x_bs.shape} → output {out.shape} ✅")
    except Exception as e:
        print(f"  Dynamic shape demo: {e}")
    print()
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · The Runtime — Core, CompiledModel, InferRequest, and Async": {
        "description": (
            "Master the three-object OpenVINO runtime API. "
            "Show Core: device enumeration, properties, and plugin configuration. "
            "Show CompiledModel: compile-time decisions, configuration options. "
            "Show InferRequest: synchronous and asynchronous inference. "
            "Implement the AsyncInferQueue pipeline pattern for throughput. "
            "Demonstrate PrePostProcessor: fusing normalisation into the graph."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
from typing import List

print("=" * 65)
print("  THE RUNTIME — CORE, COMPILEDMODEL, INFERREQUEST, AND ASYNC")
print("=" * 65)
print()

try:
    import openvino as ov
    from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
    from openvino import Layout, Type
    HAS_OV = True
    print(f"  OpenVINO {ov.__version__}")
except ImportError:
    HAS_OV = False
    print("  OpenVINO not installed: pip install openvino")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Core — device discovery and properties
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Core: device discovery and properties")
print("━" * 65)
print()

if HAS_OV:
    core = ov.Core()

    print(f"  Available devices: {core.available_devices}")
    print()

    # Query per-device properties
    for device in core.available_devices:
        print(f"  Device: {device}")
        for prop_key in ["FULL_DEVICE_NAME", "DEVICE_TYPE",
                         "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
                         "NUM_STREAMS", "RANGE_FOR_ASYNC_INFER_REQUESTS"]:
            try:
                val = core.get_property(device, prop_key)
                print(f"    {prop_key:<42s}: {val}")
            except Exception:
                pass
        print()

    # Supported optimisation capabilities
    for device in core.available_devices:
        try:
            caps = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
            print(f"  {device} optimisation capabilities: {caps}")
        except Exception:
            pass
    print()

else:
    CORE_REF = """
  OPENVINO CORE API REFERENCE:
  ─────────────────────────────────────────────────────────────────
  import openvino as ov
  core = ov.Core()

  # Device discovery
  print(core.available_devices)    ; ['CPU', 'GPU.0', 'NPU']

  # Device properties
  core.get_property("CPU", "FULL_DEVICE_NAME")
  ; → '12th Gen Intel(R) Core(TM) i7-1260P'
  core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
  ; → ['WINOGRAD', 'FP32', 'FP16', 'INT8', 'BIN', 'EXPORT_IMPORT']
  core.get_property("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
  ; → (1, 2147483647, 1)   ; (min, max, step)

  # Global settings
  core.set_property("CPU", {"NUM_STREAMS": "4"})
  core.set_property({"CACHE_DIR": "./ov_cache"})  ; cache for all devices
"""
    print(CORE_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Synchronous inference and performance hints
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Compile, infer, and performance hints")
print("━" * 65)
print()

if HAS_OV:
    try:
        import torch, torch.nn as nn

        # Build a small but realistic model
        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(4),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(), nn.Linear(64*4*4, 128), nn.ReLU(), nn.Linear(128, 10)
                )
            def forward(self, x): return self.classifier(self.features(x))

        pt_model  = SmallCNN().eval()
        example   = torch.randn(1, 3, 64, 64)
        ov_model  = ov.convert_model(pt_model, example_input=example)

        # Compile with different performance hints
        configs = [
            ("LATENCY",     {"PERFORMANCE_HINT": "LATENCY"}),
            ("THROUGHPUT",  {"PERFORMANCE_HINT": "THROUGHPUT"}),
        ]

        results = {}
        x_np    = np.random.rand(1, 3, 64, 64).astype(np.float32)
        REPS    = 500

        for hint_name, cfg in configs:
            compiled = core.compile_model(ov_model, "CPU", cfg)

            # Get the plugin's chosen configuration
            try:
                n_streams  = compiled.get_property("NUM_STREAMS")
                n_threads  = compiled.get_property("INFERENCE_NUM_THREADS")
                n_opt_req  = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            except Exception:
                n_streams = n_threads = n_opt_req = "?"

            req = compiled.create_infer_request()

            # Warmup
            for _ in range(20):
                req.infer({"x.1": x_np})

            # Benchmark
            t0 = time.perf_counter()
            for _ in range(REPS):
                req.infer({"x.1": x_np})
            t_ms = (time.perf_counter() - t0) / REPS * 1000
            results[hint_name] = t_ms

            print(f"  PERFORMANCE_HINT = {hint_name}")
            print(f"    Plugin chose:  {n_streams} stream(s), "
                  f"{n_threads} thread(s), {n_opt_req} optimal InferReq")
            print(f"    Latency:       {t_ms:.3f} ms/request")
            print()

        print(f"  Summary:")
        print(f"    LATENCY mode is {results['LATENCY']:.3f} ms vs "
              f"THROUGHPUT mode {results['THROUGHPUT']:.3f} ms (batch=1)")
        print(f"    (THROUGHPUT wins when using AsyncInferQueue with multiple requests)")
        print()

    except Exception as e:
        print(f"  Inference demo: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: AsyncInferQueue — the throughput pipeline pattern
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — AsyncInferQueue: maximum throughput pipeline")
print("━" * 65)
print()

ASYNC_GUIDE = """
  ASYNCINFERQUEUE — OPENVINO'S BUILT-IN PIPELINING
  ════════════════════════════════════════════════════════════════

  AsyncInferQueue manages a pool of N InferRequests automatically.
  You call start_async() as fast as batches arrive.
  The queue assigns each batch to a free request, blocks if all busy.

  HOW IT WORKS INTERNALLY:
    AsyncInferQueue(compiled, jobs=4)
    → Creates 4 InferRequest objects.
    → Maintains a free-request queue.
    start_async(inputs):
    → Waits for a free request (blocks if all 4 are running).
    → Sets the input tensor.
    → Calls infer_request.start_async().
    → Returns immediately.
    set_callback(fn):
    → fn(request, userdata) is called when each request finishes.
    wait_all():
    → Waits until all queued requests complete.

  PATTERN — THROUGHPUT WITH CALLBACKS:

    from openvino.runtime import AsyncInferQueue
    import threading

    results = {}
    lock    = threading.Lock()

    def callback(request: ov.InferRequest, item_id: int):
        output = request.get_output_tensor(0).data.copy()
        with lock:
            results[item_id] = output

    # Create queue with 4 parallel slots
    queue = AsyncInferQueue(compiled, jobs=4)
    queue.set_callback(callback)

    N_IMAGES = 1000
    batch_data = [np.random.rand(1, 3, 64, 64).astype(np.float32)
                  for _ in range(N_IMAGES)]

    t0 = time.perf_counter()
    for i, batch in enumerate(batch_data):
        queue.start_async({"input": batch}, userdata=i)
    queue.wait_all()
    t_total = time.perf_counter() - t0

    throughput = N_IMAGES / t_total
    print(f"  Processed {N_IMAGES} images in {t_total:.2f} s")
    print(f"  Throughput: {throughput:.1f} images/sec")

  DOUBLE-BUFFER PATTERN (lower latency variant):

    req0 = compiled.create_infer_request()
    req1 = compiled.create_infer_request()

    req0.set_input_tensor(ov.Tensor(first_batch))
    req0.start_async()

    for i, batch in enumerate(batches[1:], 1):
        # While GPU processes previous batch, CPU prepares next
        next_tensor = ov.Tensor(batch)            ; preprocessing here
        req0.wait()
        result_0 = req0.get_output_tensor(0).data.copy()
        postprocess(result_0)                     ; postprocessing here
        req0.set_input_tensor(next_tensor)
        req0.start_async()
        req0, req1 = req1, req0                   ; swap

    req0.wait()
    postprocess(req0.get_output_tensor(0).data)
"""
print(ASYNC_GUIDE)

if HAS_OV:
    try:
        from openvino.runtime import AsyncInferQueue
        import threading

        # Benchmark sync vs async
        results_async = {}
        lock  = threading.Lock()
        N_IMG = 200

        def callback(request, item_id):
            out = request.get_output_tensor(0).data.copy()
            with lock:
                results_async[item_id] = out

        compiled_thr = core.compile_model(ov_model, "CPU",
                                           {"PERFORMANCE_HINT": "THROUGHPUT"})
        queue = AsyncInferQueue(compiled_thr, jobs=4)
        queue.set_callback(callback)

        batches = [np.random.rand(1,3,64,64).astype(np.float32)
                   for _ in range(N_IMG)]

        # Warmup
        for b in batches[:10]:
            compiled_thr.create_infer_request().infer({list(compiled_thr.inputs)[0].get_any_name(): b})

        # Sync baseline
        req_sync = compiled_thr.create_infer_request()
        inp_name = list(compiled_thr.inputs)[0].get_any_name()
        t0 = time.perf_counter()
        for b in batches:
            req_sync.infer({inp_name: b})
        t_sync = time.perf_counter() - t0

        # Async with queue
        results_async.clear()
        t0 = time.perf_counter()
        for i, b in enumerate(batches):
            queue.start_async({inp_name: b}, userdata=i)
        queue.wait_all()
        t_async = time.perf_counter() - t0

        print(f"  Throughput comparison ({N_IMG} images, 64×64):")
        print(f"    Sync  (sequential):    {N_IMG/t_sync:.0f} imgs/sec  "
              f"({t_sync*1000/N_IMG:.2f} ms/img)")
        print(f"    Async (queue=4 jobs):  {N_IMG/t_async:.0f} imgs/sec  "
              f"({t_async*1000/N_IMG:.2f} ms/img)")
        print(f"    Async speedup:         {t_sync/t_async:.2f}×")
        print()
    except Exception as e:
        print(f"  Async queue demo: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: PrePostProcessor — preprocessing fused into the model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — PrePostProcessor: fuse preprocessing into the graph")
print("━" * 65)
print()

PPP_GUIDE = """
  PREPOSTPROCESSOR — ELIMINATING CPU PREPROCESSING BOTTLENECKS
  ════════════════════════════════════════════════════════════════

  WITHOUT PrePostProcessor:
    camera → uint8 BGR numpy → [Python: resize, BGR→RGB, /255, normalise]
           → float32 numpy → OpenVINO → output
    The preprocessing runs on CPU BEFORE the model, as a serial bottleneck.
    For HD video: preprocessing can exceed model inference time.

  WITH PrePostProcessor:
    camera → uint8 BGR numpy → OpenVINO [resize+BGR→RGB+normalise+model]
           → output
    Preprocessing is PART of the compiled model.
    On GPU: OpenCL kernel for resize+colour convert runs before the DNN.
    Zero CPU involvement between camera and output.

  FULL EXAMPLE:
  ─────────────────────────────────────────────────────────────────
  from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
  from openvino import Layout, Type

  ppp = PrePostProcessor(model)

  # Declare what the ACTUAL input looks like (from camera/video):
  ppp.input(0).tensor()\\
      .set_element_type(Type.u8)          \\  ; uint8 (0–255)
      .set_shape([1, 480, 640, 3])        \\  ; 640×480 HD
      .set_layout(Layout("NHWC"))         \\  ; channel-last (camera default)
      .set_color_format(ColorFormat.BGR)      ; OpenCV default

  # Declare what the MODEL expects:
  ppp.input(0).model()\\
      .set_layout(Layout("NCHW"))             ; channel-first (PyTorch default)

  # Define the preprocessing steps (run on device):
  ppp.input(0).preprocess()\\
      .convert_color(ColorFormat.RGB)     \\  ; BGR → RGB
      .resize(ResizeAlgorithm.RESIZE_LINEAR, 224, 224)  \\  ; 640×480 → 224×224
      .convert_element_type(Type.f32)     \\  ; uint8 → float32
      .mean([0.485*255, 0.456*255, 0.406*255]) \\  ; ImageNet mean (unnormalised)
      .scale([0.229*255, 0.224*255, 0.225*255])    ; ImageNet std

  model_with_preproc = ppp.build()
  compiled = core.compile_model(model_with_preproc, "GPU")

  # Now pass RAW uint8 BGR frames directly:
  frame = cv2.VideoCapture(0).read()[1]   ; uint8 BGR, 480×640×3
  result = compiled([frame])[compiled.output(0)]
  # NO Python preprocessing at all!

  WHAT PrePostProcessor CAN DO:
    Input side:
      .set_element_type(Type.u8/f32/f16/...)  ; input data type
      .set_layout("NHWC" / "NCHW" / ...)     ; memory layout
      .set_color_format(BGR/RGB/NV12/I420/...) ; colour format
      .convert_element_type(Type.f32)         ; dtype conversion
      .convert_color(RGB)                     ; colour space conversion
      .resize(LINEAR/NEAREST, h, w)           ; spatial resize
      .mean([r, g, b]) / .scale([r, g, b])    ; normalisation

    Output side:
      .convert_element_type(Type.f32)         ; output dtype conversion
      .convert_layout("NHWC")                 ; output layout change

  SPECIAL FORMATS — NV12 (YUV from camera/video):
    NV12 is the native format of most cameras, video decoders, and NPUs.
    OpenVINO can accept NV12 directly and convert to RGB as part of model:

    ppp.input(0).tensor()\\
        .set_element_type(Type.u8)\\
        .set_shape([1, 480*3//2, 640])   ; NV12: H*3/2 rows
        .set_layout(Layout("NCHW"))\\
        .set_color_format(ColorFormat.NV12_TWO_PLANES)   ; Y + UV planes
    ppp.input(0).preprocess()\\
        .convert_color(ColorFormat.BGR)   ; NV12 → BGR automatically
"""
print(PPP_GUIDE)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · NNCF Quantisation — INT8 PTQ, Accuracy-Aware, and Weight Compression": {
        "description": (
            "Complete NNCF quantisation workflow from float to INT8. "
            "Run post-training quantisation with a calibration dataset. "
            "Compare float32 vs INT8 accuracy and latency side-by-side. "
            "Demonstrate accuracy-aware quantisation that automatically reverts sensitive layers. "
            "Show INT4 weight-only compression for LLM deployment. "
            "Visualise FakeQuantize nodes inserted by NNCF in the IR."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  NNCF QUANTISATION — INT8 PTQ, ACCURACY-AWARE, WEIGHT COMPRESSION")
print("=" * 65)
print()

try:
    import openvino as ov
    HAS_OV = True
    print(f"  OpenVINO {ov.__version__}")
except ImportError:
    HAS_OV = False
    print("  OpenVINO not installed.")

try:
    import nncf
    HAS_NNCF = True
    print(f"  NNCF {nncf.__version__}")
except ImportError:
    HAS_NNCF = False
    print("  NNCF not installed: pip install nncf")

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: NNCF PTQ — the standard quantisation workflow
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — NNCF PTQ: the complete quantisation pipeline")
print("━" * 65)
print()

NNCF_PTQ_THEORY = """
  NNCF POST-TRAINING QUANTISATION — STEP BY STEP
  ════════════════════════════════════════════════════════════════

  STEP 1 — BUILD THE FLOAT MODEL:
    Any framework: PyTorch, TF, Keras — or load existing OV IR.

  STEP 2 — CONVERT TO OPENVINO IR:
    ov_model = ov.convert_model(model, example_input=example)
    This gives NNCF a clean FP32 OV graph to work with.

  STEP 3 — PREPARE CALIBRATION DATASET:
    def calibration_dataset_fn():
        for batch, labels in val_loader:
            yield {"input_name": batch.numpy()}
    # 200–1000 samples is typically sufficient.
    # Samples must come from the SAME distribution as real inference data.
    # More diverse samples → better range estimation → higher accuracy.

  STEP 4 — QUANTISE:
    calibration_ds = nncf.Dataset(calibration_dataset_fn())
    quantised = nncf.quantize(
        model               = ov_model,
        calibration_dataset = calibration_ds,
        preset              = nncf.QuantizationPreset.PERFORMANCE,
        target_device       = nncf.TargetDevice.CPU,
        subset_size         = 300,
    )

  STEP 5 — COMPILE AND MEASURE:
    compiled_int8 = core.compile_model(quantised, "CPU")

  WHAT NNCF DOES INTERNALLY:
  ─────────────────────────────────────────────────────────────────
  1. GRAPH ANALYSIS:
     Walk the OV IR graph, identify all ops where quantisation
     can be applied (MatMul, Convolution, DepthwiseConvolution,
     FullyConnected, AddV2 after MatMul, etc.).

  2. ACTIVATION RANGE COLLECTION:
     Run all calibration samples through the FLOAT model.
     For each identified activation tensor: record min and max
     across all calibration samples.
     Per default: uses percentile (99.9th / 0.1th) to reduce outlier impact.

  3. FAKEQUANTIZE INSERTION:
     Insert FakeQuantize nodes at identified points:
     BEFORE MatMul inputs:         quantise activations (unsigned INT8, range [0,255])
     BEFORE Convolution weights:   quantise weights (signed INT8, range [-128,127])
     The FakeQuantize carries: in_low, in_high, out_low, out_high, levels=256.

  4. BIAS CORRECTION (optional, default=True):
     After inserting FakeQuantize nodes, run calibration data again.
     Measure the SHIFT in each layer's mean output (quantisation error).
     Adjust the bias tensor to compensate for this shift.
     Typically recovers 0.2–0.5% accuracy for free.

  5. RESULT:
     The quantised OV IR contains FakeQuantize nodes.
     When compiled by OpenVINO, these are fused into INT8 kernels.
     The final compiled model executes fully in INT8 — no FakeQuantize overhead.

  QUANTISATION PRECISION IMPACT (ImageNet classification):
  ─────────────────────────────────────────────────────────────────
  Model                Float32   INT8 PERF  INT8 MIXED   INT8+BiasCorr
  ResNet-50            76.1%     75.2%      75.8%        75.9%
  MobileNetV2          71.9%     71.0%      71.4%        71.5%
  EfficientNet-B0      77.1%     76.3%      76.7%        76.8%
  BERT-Base (F1)       88.3%     87.8%      88.1%        88.2%
"""
print(NNCF_PTQ_THEORY)

if HAS_OV and HAS_NNCF and HAS_TORCH:
    print("  Running live NNCF PTQ demo...")
    print()

    # Build a simple model for demonstration
    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*4*4, 10),
            )
        def forward(self, x): return self.net(x)

    pt_model = SmallNet().eval()
    example  = torch.randn(1, 3, 32, 32)
    ov_model = ov.convert_model(pt_model, example_input=example)

    # Build calibration dataset (random in this demo — real data gives better results)
    CALIB_SIZE = 150
    calib_data = np.random.rand(CALIB_SIZE, 3, 32, 32).astype(np.float32)

    def calibration_dataset_fn():
        inp_name = list(ov_model.inputs)[0].get_any_name()
        for i in range(CALIB_SIZE):
            yield {inp_name: calib_data[i:i+1]}

    # PTQ
    print("  Running NNCF quantisation (PTQ)...")
    t0 = time.perf_counter()
    try:
        calib_ds  = nncf.Dataset(calibration_dataset_fn())
        quantised = nncf.quantize(
            ov_model,
            calib_ds,
            preset        = nncf.QuantizationPreset.PERFORMANCE,
            target_device = nncf.TargetDevice.CPU,
            subset_size   = CALIB_SIZE,
        )
        t_quant = (time.perf_counter() - t0) * 1000
        print(f"  Quantisation time: {t_quant:.0f} ms")

        # Count FakeQuantize nodes
        fq_count = sum(1 for op in quantised.get_ordered_ops()
                       if op.get_type_name() == "FakeQuantize")
        total_ops = len(list(quantised.get_ordered_ops()))
        print(f"  FakeQuantize nodes inserted: {fq_count} "
              f"(out of {total_ops} total ops)")
        print()

        # Compile float32 and INT8 models
        core = ov.Core()
        inp_name = list(ov_model.inputs)[0].get_any_name()

        compiled_fp32 = core.compile_model(ov_model,   "CPU",
                                            {"PERFORMANCE_HINT": "LATENCY"})
        compiled_int8 = core.compile_model(quantised,  "CPU",
                                            {"PERFORMANCE_HINT": "LATENCY"})

        # Accuracy comparison
        N_TEST   = 200
        x_test   = np.random.rand(N_TEST, 3, 32, 32).astype(np.float32)
        req_fp32 = compiled_fp32.create_infer_request()
        req_int8 = compiled_int8.create_infer_request()

        fp32_outs, int8_outs = [], []
        for i in range(N_TEST):
            x_i = x_test[i:i+1]
            fp32_outs.append(req_fp32.infer({inp_name: x_i})[compiled_fp32.output(0)].copy())
            int8_outs.append(req_int8.infer({inp_name: x_i})[compiled_int8.output(0)].copy())

        fp32_arr = np.concatenate(fp32_outs)
        int8_arr = np.concatenate(int8_outs)
        mean_l1  = float(np.mean(np.abs(fp32_arr - int8_arr)))
        argmax_agree = float(np.mean(np.argmax(fp32_arr, axis=1) ==
                                     np.argmax(int8_arr, axis=1))) * 100

        # Latency comparison
        REPS = 500
        x_bench = np.random.rand(1, 3, 32, 32).astype(np.float32)
        for _ in range(50): req_fp32.infer({inp_name: x_bench})
        t0 = time.perf_counter()
        for _ in range(REPS): req_fp32.infer({inp_name: x_bench})
        t_fp32 = (time.perf_counter() - t0) / REPS * 1000

        for _ in range(50): req_int8.infer({inp_name: x_bench})
        t0 = time.perf_counter()
        for _ in range(REPS): req_int8.infer({inp_name: x_bench})
        t_int8 = (time.perf_counter() - t0) / REPS * 1000

        print(f"  {'Metric':30s}  {'FP32':>10s}  {'INT8':>10s}  {'Change'}")
        print("  " + "-" * 60)
        print(f"  {'Latency (ms)':30s}  {t_fp32:>10.3f}  {t_int8:>10.3f}  "
              f"{t_fp32/t_int8:.2f}× faster")
        print(f"  {'Mean L1 vs FP32':30s}  {'0.0':>10s}  {mean_l1:>10.5f}  "
              f"quantisation error")
        print(f"  {'Argmax agreement':30s}  {'100.0%':>10s}  {argmax_agree:>9.1f}%  "
              f"{'✅' if argmax_agree > 98 else '⚠️'}")
        print()

    except Exception as e:
        print(f"  NNCF quantisation: {e}")
        print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Weight-only compression for LLMs
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Weight-only compression for LLM deployment")
print("━" * 65)
print()

LLM_COMPRESSION = """
  NNCF WEIGHT-ONLY COMPRESSION FOR LARGE LANGUAGE MODELS
  ════════════════════════════════════════════════════════════════

  LLMs have a special characteristic: the weights are enormous (7B–70B params)
  but activations are relatively small (seq_len × d_model).
  Standard INT8 activation quantisation causes accuracy drops for LLMs
  because language model activations have extreme outliers (~100× normal range).
  Solution: quantise only WEIGHTS, keep activations in FP16.

  NNCF WEIGHT COMPRESSION API:
  ─────────────────────────────────────────────────────────────────
  import openvino as ov
  import nncf

  core     = ov.Core()
  ov_model = core.read_model("llama3_8b.xml")

  # INT8 weight-only (symmetric, per-channel):
  int8_model = nncf.compress_weights(
      ov_model,
      mode       = nncf.CompressWeightsMode.INT8_SYM,
  )

  # INT4 weight-only (asymmetric, group quantisation):
  int4_model = nncf.compress_weights(
      ov_model,
      mode       = nncf.CompressWeightsMode.INT4_ASYM,
      group_size = 128,    ; quantise in groups of 128 weights
      ratio      = 0.8,    ; 80% of MatMul layers at INT4, 20% at INT8
  )
  # ratio < 1.0: keeps first/last few layers at INT8 (more sensitive to quantisation)

  # NF4 (NormalFloat4, optimal for normally-distributed weights like LLaMA):
  nf4_model = nncf.compress_weights(
      ov_model,
      mode       = nncf.CompressWeightsMode.NF4,
      group_size = 64,
  )

  MEMORY COMPARISON (Llama-3-8B):
  ─────────────────────────────────────────────────────────────────
  Precision     Model size    Inference RAM    Accuracy (MMLU)
  FP32          32 GB         40 GB            65.3%
  FP16          16 GB         20 GB            65.2% (≈ identical)
  INT8_SYM       8 GB         10 GB            64.8% (-0.5%)
  INT4_ASYM      4 GB          5 GB            63.9% (-1.4%)
  INT4_ASYM r0.8 4.5 GB       5.6 GB           64.5% (-0.8%)  ← best tradeoff
  NF4            4 GB          5 GB            64.2% (-1.1%)

  HARDWARE REQUIREMENTS FOR LLAMA-3-8B:
  ─────────────────────────────────────────────────────────────────
  FP16:    Intel Arc A770 (16 GB VRAM)     or  64 GB system RAM on Xeon
  INT8:    Intel Arc A380 (6 GB VRAM)      or  32 GB system RAM on Core
  INT4:    Intel Iris Xe (4 GB iGPU VRAM)  or  16 GB system RAM on Core Ultra
           → runs on a standard AI PC laptop (16 GB RAM, Meteor Lake NPU+CPU)

  GENERATION SPEED (Llama-3-8B on Intel Core Ultra 155H):
  ─────────────────────────────────────────────────────────────────
  Device     Precision  Tokens/sec
  CPU        INT4       ~8 tok/s
  iGPU       FP16       ~12 tok/s
  iGPU       INT4       ~22 tok/s
  NPU        INT4       ~20 tok/s  (lower latency, lower power)
"""
print(LLM_COMPRESSION)

if HAS_OV and HAS_NNCF and HAS_TORCH:
    # Demonstrate weight compression on a small linear model
    class TinyLM(nn.Module):
        """Simulates the core Linear layers of an LLM."""
        def __init__(self, d=128):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)
            self.gate   = nn.Linear(d, d*2, bias=False)
            self.down   = nn.Linear(d*2, d, bias=False)
        def forward(self, x):
            q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
            attn = (q + k + v) / 3  # simplified
            o    = self.o_proj(attn)
            g    = torch.relu(self.gate(o))
            return self.down(g) + x

    tiny_lm = TinyLM(128).eval()
    example = torch.randn(1, 16, 128)   # [batch, seq, d_model]
    ov_lm   = ov.convert_model(tiny_lm, example_input=example)

    print("  Weight compression on tiny LLM-style model:")
    print()

    original_size = sum(np.prod(op.shape) * 4
                        for op in ov_lm.get_ordered_ops()
                        if op.get_type_name() == "Const"
                        and np.prod(op.shape) > 100) / 1024

    try:
        int8_lm = nncf.compress_weights(ov_lm,
                                          mode=nncf.CompressWeightsMode.INT8_SYM)
        int8_size = sum(np.prod(op.shape)
                        for op in int8_lm.get_ordered_ops()
                        if op.get_type_name() == "Const"
                        and np.prod(op.shape) > 100) / 1024

        print(f"  {'Precision':12s}  {'Weight size (KB)':>18s}  {'Compression':>12s}")
        print("  " + "-" * 46)
        print(f"  {'FP32':12s}  {original_size:>18.1f}  {'1.00×':>12s}")
        print(f"  {'INT8_SYM':12s}  {int8_size:>18.1f}  "
              f"{original_size/int8_size if int8_size else 0:>12.2f}×")
        print()

    except Exception as e:
        print(f"  Weight compression demo: {e}")
        print()
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Device Benchmarking and Performance Configuration": {
        "description": (
            "Systematic performance measurement across CPU, GPU, and AUTO. "
            "Benchmark latency vs throughput trade-off: stream count effect. "
            "Show BF16 and FP16 precision modes and their impact. "
            "Demonstrate the compile-time caching mechanism. "
            "Measure the overhead of model compilation vs cached loading. "
            "Show how to use the benchmark_app equivalent in Python."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  DEVICE BENCHMARKING AND PERFORMANCE CONFIGURATION")
print("=" * 65)
print()

try:
    import openvino as ov
    HAS_OV = True
    print(f"  OpenVINO {ov.__version__}")
except ImportError:
    HAS_OV = False
    print("  OpenVINO not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Latency vs throughput: stream count impact
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Streams and threads: the latency/throughput trade-off")
print("━" * 65)
print()

PERF_CONFIG = """
  OPENVINO CPU PERFORMANCE CONFIGURATION
  ════════════════════════════════════════════════════════════════

  CPU inference performance is controlled by two key parameters:

  NUM_STREAMS (number of parallel inference lanes):
    Each stream has its own execution context, thread pool, and buffers.
    More streams → more parallelism → higher THROUGHPUT.
    More streams → each stream has fewer threads → higher LATENCY per request.
    Rule: NUM_STREAMS × INFERENCE_NUM_THREADS ≤ total physical cores.

  INFERENCE_NUM_THREADS (threads per stream):
    Each thread handles one portion of the current operation (parallelised loop).
    More threads per stream → each request finishes faster (lower latency).
    But: more threads × more cache pressure = diminishing returns.

  OPTIMAL CONFIGURATIONS:
  ─────────────────────────────────────────────────────────────────
  8-core CPU, latency scenario (online inference, single request):
    NUM_STREAMS=1, INFERENCE_NUM_THREADS=8
    → All 8 cores focus on finishing one request as fast as possible.
    → Best single-request latency.

  8-core CPU, throughput scenario (batch processing):
    NUM_STREAMS=4, INFERENCE_NUM_THREADS=2
    → 4 requests run in parallel, each using 2 cores.
    → 4× more work done per unit time vs latency config.
    → Each request takes longer, but the overall throughput is higher.

  8-core CPU, balanced:
    PERFORMANCE_HINT="LATENCY" → plugin auto-selects 1 stream
    PERFORMANCE_HINT="THROUGHPUT" → plugin auto-selects optimal N streams

  WHEN TO USE WHICH:
    LATENCY:     Online API endpoint (single user), interactive apps,
                 edge real-time (camera, microphone).
    THROUGHPUT:  Batch processing (offline video analysis, document OCR),
                 high-traffic serving endpoint with request queue,
                 cloud inference where requests can be batched.
    AUTO:        When you're not sure — plugin benchmarks both and picks.

  VERIFYING THE CHOICE:
    compiled = core.compile_model(model, "CPU",
                                   {"PERFORMANCE_HINT": "THROUGHPUT"})
    print(compiled.get_property("NUM_STREAMS"))
    print(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    ; Use EXACTLY optimal_infer_reqs async requests for best throughput.
"""
print(PERF_CONFIG)

if HAS_OV and HAS_TORCH:
    # Build benchmark model
    class BenchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128*4*4, 10)
            )
        def forward(self, x): return self.net(x)

    pt_model = BenchNet().eval()
    example  = torch.randn(1, 3, 64, 64)
    ov_model = ov.convert_model(pt_model, example_input=example)

    core     = ov.Core()
    inp_name = list(ov_model.inputs)[0].get_any_name()
    x_bench  = np.random.rand(1, 3, 64, 64).astype(np.float32)
    REPS     = 300

    print(f"  Model: BenchNet (64×64 input), {sum(p.numel() for p in pt_model.parameters()):,} params")
    print()
    print(f"  {'Config':45s}  {'Latency (ms)':>12s}  {'Throughput':>12s}")
    print("  " + "-" * 74)

    perf_configs = [
        ("LATENCY hint (1 stream)",  {"PERFORMANCE_HINT": "LATENCY"}),
        ("THROUGHPUT hint (N streams)", {"PERFORMANCE_HINT": "THROUGHPUT"}),
    ]

    for cfg_name, config in perf_configs:
        try:
            compiled = core.compile_model(ov_model, "CPU", config)
            req      = compiled.create_infer_request()

            # Warmup
            for _ in range(30): req.infer({inp_name: x_bench})

            # Latency (sequential, 1 request at a time)
            t0 = time.perf_counter()
            for _ in range(REPS): req.infer({inp_name: x_bench})
            latency_ms = (time.perf_counter() - t0) / REPS * 1000
            throughput  = 1000 / latency_ms

            try:
                n_streams = compiled.get_property("NUM_STREAMS")
                cfg_detail = f"{cfg_name} [{n_streams} stream(s)]"
            except Exception:
                cfg_detail = cfg_name

            print(f"  {cfg_detail[:44]:45s}  {latency_ms:>12.3f}  "
                  f"{throughput:>10.1f} rps")
        except Exception as e:
            print(f"  {cfg_name}: {e}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Compilation cache — startup time optimisation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Compilation cache: amortising startup overhead")
print("━" * 65)
print()

CACHE_GUIDE = """
  OPENVINO COMPILATION CACHE
  ════════════════════════════════════════════════════════════════

  compile_model() is expensive: it includes JIT code generation,
  graph optimisation, and memory planning. This costs:
    Small model (MobileNet):    50–200 ms
    Medium model (ResNet-50):   200–500 ms
    Large model (BERT-Base):    1–5 seconds
    NPU (first compile):        10–60 seconds (AOT to blob)
    GPU (OpenCL JIT):           5–30 seconds (first run per GPU)

  The CACHE_DIR property serialises the compiled artifact to disk.
  On subsequent runs: compiled artifact is loaded directly.
    CPU cache hit:   10–50 ms   (load + validate)
    NPU cache hit:   < 1 second
    GPU cache hit:   < 2 seconds

  ENABLING THE CACHE:
    # Global (all devices):
    core.set_property({"CACHE_DIR": "./ov_model_cache"})
    compiled = core.compile_model(ov_model, "CPU")
    ; First run: compiles + saves to ./ov_model_cache/*.blob
    ; Next run:  loads from cache, skips compilation

    # Per-device:
    core.compile_model(ov_model, "NPU",
                       {"CACHE_DIR": "./npu_cache"})

  CACHE INVALIDATION:
    The cache is keyed by: (model hash, device, configuration hash).
    Any change to: model weights, model structure, configuration, or
    OpenVINO version → cache miss → recompilation.
    You can force cache invalidation by deleting the cache directory.

  PRODUCTION BEST PRACTICE:
    # Warm up the cache during model deployment (not serving):
    # 1. Pre-compile: run compile_model() once with CACHE_DIR set.
    # 2. Container startup: subsequent compile_model() is fast (cache hit).
    # This avoids the first-request latency spike in production.
"""
print(CACHE_GUIDE)

if HAS_OV and HAS_TORCH:
    tmpdir   = tempfile.mkdtemp()
    cache_dir = os.path.join(tmpdir, "ov_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"  Compilation cache benchmark (cache at {cache_dir}):")
    print()

    # First compilation (no cache)
    core_nocache = ov.Core()
    t0 = time.perf_counter()
    compiled_1 = core_nocache.compile_model(ov_model, "CPU",
                                             {"PERFORMANCE_HINT": "LATENCY"})
    t_cold = (time.perf_counter() - t0) * 1000
    print(f"  Cold compile (no cache):   {t_cold:.1f} ms")

    # Second compilation (with cache enabled)
    core_cache = ov.Core()
    core_cache.set_property({"CACHE_DIR": cache_dir})
    t0 = time.perf_counter()
    compiled_2 = core_cache.compile_model(ov_model, "CPU",
                                           {"PERFORMANCE_HINT": "LATENCY"})
    t_first_cached = (time.perf_counter() - t0) * 1000
    print(f"  First run with cache:      {t_first_cached:.1f} ms  "
          f"(compile + save cache)")

    # Third compilation (cache hit)
    core_cache2 = ov.Core()
    core_cache2.set_property({"CACHE_DIR": cache_dir})
    t0 = time.perf_counter()
    compiled_3 = core_cache2.compile_model(ov_model, "CPU",
                                            {"PERFORMANCE_HINT": "LATENCY"})
    t_warm = (time.perf_counter() - t0) * 1000
    print(f"  Cache hit (warm):          {t_warm:.1f} ms  "
          f"(load from cache, skip JIT)")

    # Verify cache files were created
    cache_files = os.listdir(cache_dir)
    total_cache_mb = sum(os.path.getsize(os.path.join(cache_dir, f))
                         for f in cache_files) / 1024 / 1024
    print()
    print(f"  Cache files: {len(cache_files)} file(s), "
          f"{total_cache_mb:.2f} MB total")
    print(f"  Startup speedup with cache: {t_cold/t_warm:.1f}×")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Precision modes and their accuracy/speed impact
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Precision modes: FP32, FP16, BF16 on CPU")
print("━" * 65)
print()

PRECISION_GUIDE = """
  OPENVINO PRECISION MODES
  ════════════════════════════════════════════════════════════════

  INFERENCE_PRECISION_HINT controls the arithmetic precision:

  "f32" (Float32):
    Full precision. Universal fallback.
    Speed:    1.0× baseline.
    Accuracy: reference.
    When to use: debugging, models sensitive to precision loss.

  "f16" (Float16):
    GPU default. Also available on Intel CPU with native FP16.
    Speed:    1.5–2× vs FP32 on GPU (SIMD processes 2× values per instruction).
    Accuracy: within 0.01% of FP32 for most models.
    When to use: GPU inference, memory-bandwidth-limited workloads.

  "bf16" (BFloat16):
    Available on Intel CPUs with AVX-512 BF16 extension:
      Ice Lake (11th gen Core): AVX512-BF16
      Tiger Lake (11th gen Core): AVX512-BF16
      Alder Lake+ (12th gen+): AMX-BF16
      Sapphire Rapids Xeon: AMX-BF16 (highest throughput)
    Speed:    1.5–2× vs FP32 on capable CPUs.
    Accuracy: within 0.1% of FP32 (BF16 has same exponent range as FP32).
    When to use: Intel CPU deployment where hardware supports BF16.

  CHECKING WHAT'S SUPPORTED:
    caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    ; e.g., ['FP32', 'FP16', 'BF16', 'INT8', 'BIN']

  SETTING PRECISION:
    compiled = core.compile_model(model, "CPU",
                                   {"INFERENCE_PRECISION_HINT": "bf16"})

    # For GPU, FP16 is auto-selected by default.
    # To force FP32 on GPU (for debugging):
    compiled = core.compile_model(model, "GPU",
                                   {"INFERENCE_PRECISION_HINT": "f32"})

  WHEN PRECISION HINT IS IGNORED:
    If the model is already INT8 (from NNCF), precision hint has no effect
    (INT8 kernels are used regardless).
    If the hardware doesn't support BF16, the plugin falls back to FP32.
    The runtime does NOT crash — it silently uses the best available precision.
"""
print(PRECISION_GUIDE)

if HAS_OV and HAS_TORCH:
    print("  Precision mode benchmark:")
    print()
    x_prec = np.random.rand(1, 3, 64, 64).astype(np.float32)
    REPS_P  = 300

    precision_modes = [
        ("FP32 (INFERENCE_PRECISION_HINT=f32)",  "f32"),
        ("BF16 (INFERENCE_PRECISION_HINT=bf16)", "bf16"),
    ]

    # Check if BF16 is supported
    try:
        caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
        bf16_avail = "BF16" in caps if isinstance(caps, (list, tuple)) else "BF16" in str(caps)
    except Exception:
        bf16_avail = False

    fp32_t = None
    for mode_name, mode_str in precision_modes:
        try:
            comp = core.compile_model(ov_model, "CPU",
                                       {"INFERENCE_PRECISION_HINT": mode_str,
                                        "PERFORMANCE_HINT": "LATENCY"})
            req  = comp.create_infer_request()
            for _ in range(30): req.infer({inp_name: x_prec})

            t0 = time.perf_counter()
            for _ in range(REPS_P): req.infer({inp_name: x_prec})
            t_ms = (time.perf_counter() - t0) / REPS_P * 1000

            if fp32_t is None: fp32_t = t_ms
            ratio = fp32_t / t_ms
            avail = "" if mode_str == "f32" else (" (available)" if bf16_avail else " (may fallback to FP32)")

            print(f"  {mode_name:<48s}  {t_ms:.3f} ms  {ratio:.2f}×{avail}")
        except Exception as e:
            print(f"  {mode_name}: {e}")
    print()
    print("  Note: BF16 speedup visible on Ice Lake+ and Tiger Lake+ CPUs.")
    print("  On CPUs without BF16, the hint is silently ignored (falls back to FP32).")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · OpenVINO GenAI and the Connected Stack": {
        "description": (
            "Run LLM inference with OpenVINO GenAI on CPU/GPU. "
            "Export a HuggingFace model to OpenVINO IR with Optimum-Intel. "
            "Show streaming generation and the async generator API. "
            "Demonstrate OVMS: starting the server and sending REST requests. "
            "Summarise OpenVINO's position in the full connected compiler stack. "
            "Quick-reference table: every key API, config, and tool."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  OPENVINO GENAI AND THE CONNECTED STACK")
print("=" * 65)
print()

try:
    import openvino as ov
    HAS_OV = True
    print(f"  OpenVINO {ov.__version__}")
except ImportError:
    HAS_OV = False
    print("  OpenVINO not installed.")

HAS_GENAI = False
try:
    import openvino_genai as ov_genai
    HAS_GENAI = True
    print(f"  OpenVINO GenAI: available")
except ImportError:
    print("  OpenVINO GenAI: not installed (pip install openvino-genai)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: OpenVINO GenAI — LLM inference pipeline
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — OpenVINO GenAI: the complete LLM pipeline")
print("━" * 65)
print()

GENAI_GUIDE = """
  OPENVINO GENAI — FROM EXPORT TO GENERATION
  ════════════════════════════════════════════════════════════════

  STEP 1 — EXPORT THE MODEL (using Optimum-Intel):
  ─────────────────────────────────────────────────────────────────
  pip install optimum[openvino]

  # Export Phi-3 Mini (3.8B, fast on Intel hardware) with INT4 weights:
  optimum-cli export openvino \\
      --model microsoft/Phi-3-mini-4k-instruct \\
      --weight-format int4 \\
      --group-size 128 \\
      --ratio 1.0 \\
      --trust-remote-code \\
      ./phi3_mini_int4_ov/

  # Output directory:
  #   openvino_model.xml/.bin      (quantised LLM weights)
  #   openvino_tokenizer.xml/.bin  (tokenizer as OV graph — no HF dep at runtime)
  #   tokenizer.json               (HF tokenizer config)
  #   generation_config.json       (default generation parameters)

  MODEL SIZE COMPARISON (Phi-3 Mini 3.8B):
    FP16:  ~7 GB    (too large for 8 GB RAM machines)
    INT8:  ~4 GB    (fits in 8 GB RAM)
    INT4:  ~2 GB    (fits in 4 GB RAM; runs on budget AI PCs)

  STEP 2 — LOAD AND GENERATE:
  ─────────────────────────────────────────────────────────────────
  import openvino_genai as ov_genai

  # Load model (compilation happens here — use CACHE_DIR for faster startup):
  pipe = ov_genai.LLMPipeline("./phi3_mini_int4_ov/", "CPU")
  # OR GPU: ov_genai.LLMPipeline("./phi3_mini_int4_ov/", "GPU")
  # OR AUTO: ov_genai.LLMPipeline("./phi3_mini_int4_ov/", "AUTO")

  # Basic generation:
  config = ov_genai.GenerationConfig()
  config.max_new_tokens = 100
  config.temperature    = 0.8
  config.top_k          = 50
  config.top_p          = 0.9

  result = pipe.generate("Explain the Pythagorean theorem in one sentence:",
                          config)
  print(result)

  # Streaming (token-by-token output):
  def streamer(token: str) -> bool:
      print(token, end="", flush=True)
      return False   ; return True to stop generation early

  pipe.generate("Write a haiku about silicon:", config, streamer)

  # Chat mode (maintains conversation history):
  pipe.start_chat(system_message="You are a helpful AI assistant.")
  response1 = pipe.chat("What is Intel OpenVINO?")
  response2 = pipe.chat("How does it compare to ONNX Runtime?")
  response3 = pipe.chat("Give me a code example.")
  pipe.finish_chat()

  GENERATION CONFIG OPTIONS:
  ─────────────────────────────────────────────────────────────────
  config.max_new_tokens:    maximum tokens to generate (required)
  config.temperature:       sampling randomness (0.0=greedy, 1.0=full random)
  config.top_k:             keep only top-k tokens by probability
  config.top_p:             nucleus sampling — keep tokens summing to top_p prob
  config.repetition_penalty: penalise repeated tokens (>1.0 reduces repetition)
  config.num_beams:         beam search width (1=greedy, >1=beam search)
  config.do_sample:         True=sampling, False=greedy/beam
  config.eos_token_id:      stop generation at this token ID

  PERFORMANCE:
  ─────────────────────────────────────────────────────────────────
  # Query TTFT (time-to-first-token) and throughput:
  pipe.get_metrics()
  # Returns: GenerateMetrics with:
  #   .mean_ttft_ms:          Time To First Token (prefill latency)
  #   .mean_tpot_ms:          Time Per Output Token (decode latency)
  #   .throughput:            tokens per second
  #   .generate_duration_ms:  total generation time
"""
print(GENAI_GUIDE)

if HAS_GENAI:
    # If GenAI is available, show it working with a mock
    print("  OpenVINO GenAI is installed. Example (requires model files):")
    EXAMPLE = """
    import openvino_genai as ov_genai

    pipe = ov_genai.LLMPipeline("./phi3_mini_int4_ov/", "CPU")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100

    result = pipe.generate("What is 2+2? Answer briefly:", config)
    print(result)
"""
    print(EXAMPLE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: OVMS — production model serving
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — OpenVINO Model Server (OVMS)")
print("━" * 65)
print()

OVMS_GUIDE = """
  OPENVINO MODEL SERVER — PRODUCTION DEPLOYMENT
  ════════════════════════════════════════════════════════════════

  OVMS serves OpenVINO, ONNX, TFLite, and TF SavedModel files via:
    gRPC (port 9000): TensorFlow Serving-compatible protocol
    REST (port 9001): HTTP/JSON API
    OpenAI API (LLMs): /v1/chat/completions endpoint

  DEPLOYING WITH DOCKER:
  ─────────────────────────────────────────────────────────────────
  # Directory structure:
  # ./models/
  #     resnet50/
  #         1/
  #             model.xml
  #             model.bin

  docker run -d \\
      -v "$(pwd)/models:/models" \\
      -p 9000:9000 -p 9001:9001 \\
      openvino/model_server:latest \\
      --model_path /models/resnet50 \\
      --model_name resnet50 \\
      --port 9000 \\
      --rest_port 9001

  # Health check:
  curl http://localhost:9001/v1/config
  # → {"resnet50": {"model_version_status": [{"state": "AVAILABLE"}]}}

  SENDING REQUESTS (REST):
  ─────────────────────────────────────────────────────────────────
  import requests, json, numpy as np

  image = np.random.rand(1, 3, 224, 224).astype(np.float32)
  payload = {"inputs": {"input": image.tolist()}}

  response = requests.post(
      "http://localhost:9001/v1/models/resnet50:predict",
      json=payload)
  result   = response.json()["outputs"]["output"]
  probas   = np.array(result)
  top_class = np.argmax(probas)

  SENDING REQUESTS (gRPC):
  ─────────────────────────────────────────────────────────────────
  # pip install tritonclient[grpc] OR tensorflow-serving-api

  from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

  client = InferenceServerClient("localhost:9000")
  input_tensor = InferInput("input", [1, 3, 224, 224], "FP32")
  input_tensor.set_data_from_numpy(image)

  response = client.infer("resnet50",
                            inputs=[input_tensor],
                            outputs=[InferRequestedOutput("output")])
  result = response.as_numpy("output")

  LLM SERVING (OpenAI-compatible):
  ─────────────────────────────────────────────────────────────────
  # Deploy Llama-3 in OVMS:
  docker run -d \\
      -v "$(pwd)/llama3_int4_ov:/models/llama3/1" \\
      -p 9001:9001 \\
      openvino/model_server:latest \\
      --model_path /models/llama3 \\
      --model_name llama3 \\
      --target_device CPU \\
      --plugin_config '{"PERFORMANCE_HINT": "THROUGHPUT"}' \\
      --rest_port 9001

  # Use OpenAI client:
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:9001/v1", api_key="unused")
  response = client.chat.completions.create(
      model="llama3",
      messages=[{"role": "user", "content": "What is Intel OpenVINO?"}])
  print(response.choices[0].message.content)

  MULTI-MODEL CONFIG (config.json):
  ─────────────────────────────────────────────────────────────────
  {
    "model_config_list": [
      {"config": {"name": "detection",   "base_path": "/models/yolov8",
                  "plugin_config": {"PERFORMANCE_HINT": "THROUGHPUT"},
                  "batch_size": "auto", "max_batch_size": 8}},
      {"config": {"name": "embeddings",  "base_path": "/models/bert",
                  "nireq": 4}},
      {"config": {"name": "llm",         "base_path": "/models/phi3",
                  "target_device": "CPU",
                  "plugin_config": {"CACHE_DIR": "/tmp/ov_cache"}}}
    ]
  }

  docker run ... --config_path /models/config.json
"""
print(OVMS_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack and quick reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — OpenVINO in the connected compiler stack")
print("━" * 65)
print()

STACK = """
  OPENVINO IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                     │
  │  OpenVINO's CPU plugin (oneDNN) uses LLVM-compiled SIMD kernels.     │
  │  AVX-512/BF16/AMX code paths are LLVM-compiled intrinsics.           │
  │  The OpenVINO toolkit itself is compiled by Intel Clang (LLVM-based).│
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                     │
  │  OpenVINO uses MLIR internally for graph transformation passes.      │
  │  The NPU compiler pipeline is MLIR-based.                            │
  │  Future: OV is converging toward an MLIR-first architecture.         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 05/06: XLA / OpenXLA                                         │
  │  JAX models reach OV via JAX → ONNX → ov.convert_model().            │
  │  Both XLA and OV use oneDNN for Intel CPU — they share the library.  │
  │  For Intel CPU: OV often wins (NNCF INT8 outperforms XLA's FP32).    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                      │
  │  TVM and OV compete on Intel CPU/edge inference.                     │
  │  OV wins: Intel NPU and Myriad VPU (no TVM backend).                 │
  │  TVM wins: custom hardware where MetaSchedule finds better tiles.    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 10: ONNX                                                     │
  │  ONNX is OV's primary and most reliable import format.               │
  │  OV supports ONNX opset 1–20.                                        │
  │  ONNX → OV is the recommended path for PyTorch models.               │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 11: TFLite                                                   │
  │  OV reads .tflite files directly via ov.convert_model().             │
  │  INT8 TFLite → OV INT8: quantisation preserved through conversion.   │
  │  Use case: TFLite models originally for Android, now run on Intel.   │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 12: OpenVINO (THIS MODULE)                                   │
  │  Two-file IR (XML + BIN), MLIR-based converter, oneDNN CPU plugin,   │
  │  OpenCL GPU plugin, Intel NPU plugin, NNCF quantisation,             │
  │  PrePostProcessor preprocessing fusion, AsyncInferQueue pipelining,  │
  │  OVMS production serving, OpenVINO GenAI LLM inference.              │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK)

print("  ┌──────────────────────────────────────────────────────────────────────────────────┐")
print("  │ Task                        │ API / Tool                                         │")
print("  ├──────────────────────────────────────────────────────────────────────────────────┤")
print("  │ Convert from PyTorch        │ ov.convert_model(model, example_input=)            │")
print("  │ Convert from ONNX           │ ov.convert_model('model.onnx')                     │")
print("  │ Convert from TFLite         │ ov.convert_model('model.tflite')                   │")
print("  │ Save IR                     │ ov.save_model(model, 'model.xml')                  │")
print("  │ Load IR                     │ core.read_model('model.xml')                       │")
print("  │ Compile for CPU             │ core.compile_model(model, 'CPU')                   │")
print("  │ Compile for GPU             │ core.compile_model(model, 'GPU')                   │")
print("  │ Compile for NPU             │ core.compile_model(model, 'NPU')                   │")
print("  │ Best device auto-select     │ core.compile_model(model, 'AUTO')                  │")
print("  │ Latency hint                │ config={'PERFORMANCE_HINT':'LATENCY'}              │")
print("  │ Throughput hint             │ config={'PERFORMANCE_HINT':'THROUGHPUT'}           │")
print("  │ Enable cache                │ core.set_property({'CACHE_DIR':'./cache'})         │")
print("  │ Sync inference              │ request.infer({'input': array})                    │")
print("  │ Async inference             │ request.start_async(); .wait()                     │")
print("  │ Async queue pipeline        │ AsyncInferQueue(compiled, jobs=4)                  │")
print("  │ Fuse preprocessing          │ PrePostProcessor(model).build()                    │")
print("  │ INT8 quantisation           │ nncf.quantize(model, calib_ds)                     │")
print("  │ Accuracy-aware INT8         │ nncf.quantize_with_accuracy_control()              │")
print("  │ LLM weight compression      │ nncf.compress_weights(model, INT4_ASYM)            │")
print("  │ LLM export (HF)             │ optimum-cli export openvino --weight-format int4   │")
print("  │ LLM inference               │ ov_genai.LLMPipeline(path, device)                 │")
print("  │ LLM streaming               │ pipe.generate(prompt, config, streamer_fn)         │")
print("  │ Model serving               │ docker openvino/model_server                       │")
print("  │ Benchmark CLI               │ benchmark_app -m model.xml -d CPU                  │")
print("  ├──────────────────────────────────────────────────────────────────────────────────┤")
print("  │ Documentation               │ docs.openvino.ai                                   │")
print("  │ GitHub                      │ github.com/openvinotoolkit/openvino                │")
print("  │ NNCF GitHub                 │ github.com/openvinotoolkit/nncf                    │")
print("  │ OVMS GitHub                 │ github.com/openvinotoolkit/model_server            │")
print("  │ GenAI GitHub                │ github.com/openvinotoolkit/openvino.genai          │")
print("  │ Optimum Intel               │ github.com/huggingface/optimum-intel               │")
print("  │ Notebooks                   │ github.com/openvinotoolkit/openvino_notebooks      │")
print("  └──────────────────────────────────────────────────────────────────────────────────┘")
print()

if HAS_OV:
    print("  Runtime verification:")
    try:
        import torch, torch.nn as nn

        # Minimal end-to-end test
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
        x = torch.randn(2, 8)

        with torch.no_grad():
            pt_out = m(x).numpy()

        ov_m     = ov.convert_model(m, example_input=x)
        core_v   = ov.Core()
        comp_v   = core_v.compile_model(ov_m, "CPU")
        inp_n    = list(comp_v.inputs)[0].get_any_name()
        ov_out   = comp_v([x.numpy()])[comp_v.output(0)]

        max_err  = float(np.max(np.abs(pt_out - ov_out)))
        print(f"    Linear(8→16→4): PyTorch vs OpenVINO CPU max_err={max_err:.2e} "
              f"{'✅' if max_err < 1e-4 else '⚠️'}")
        print(f"    Devices available: {core_v.available_devices}")
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