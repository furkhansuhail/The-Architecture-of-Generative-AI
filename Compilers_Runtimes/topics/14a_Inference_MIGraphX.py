"""
MIGraphX — AMD's GPU Inference Engine
=======================================

MIGraphX is AMD's purpose-built deep learning inference engine for AMD GPUs.
Where TensorRT is NVIDIA's answer to the question "how do we make GPU inference
as fast as possible on our hardware?", MIGraphX is AMD's answer to the same
question — on ROCm, targeting Radeon Instinct, Radeon Pro, and RDNA datacenter
GPUs.

MIGraphX sits at the convergence point of three things AMD needed simultaneously:

    An inference runtime that exploits AMD GPU architecture — specifically the
    GCN/CDNA memory hierarchy, wavefront execution model, and the matrix
    FMA units in CDNA2/CDNA3 that deliver competitive INT8 and FP8 throughput.

    A bridge to the ROCm software stack — MIGraphX is the inference-facing
    component of ROCm, consuming ONNX models and dispatching kernels through
    MIOpen (AMD's cuDNN equivalent), rocBLAS (AMD's cuBLAS), and composable
    kernel (AMD's hand-tuned GEMM/convolution library).

    An open-source counterweight to TensorRT — unlike TensorRT, MIGraphX is
    fully open source (Apache 2.0), making its compiler passes, fusion rules,
    and kernel selection logic inspectable, forkable, and contributable.

The MIGraphX compilation pipeline transforms an ONNX (or TF/PyTorch) graph
through a series of C++ passes into a sequence of compiled GPU kernels. The
key passes are: dead code elimination, constant propagation, operator fusion
(elementwise chains, attention patterns), layout optimisation (NCHW↔NHWC),
and finally kernel lowering to MIOpen/rocBLAS/composable_kernel dispatch.

At runtime, MIGraphX serialises the compiled program to a binary (`.mxr` file)
that can be deserialized and re-executed without recompilation — critical for
production serving where startup latency matters.

In the connected compiler stack:
    LLVM      (module 01) ← ROCm's HIP compiler (hipcc) is LLVM-based; MIGraphX kernels compile via LLVM
    MLIR      (module 02) ← MIGraphX uses MLIR for some internal lowering; ONNX-MLIR is a related project
    TVM       (module 08) ← TVM supports AMD GPU via ROCm backend; MIGraphX and TVM compete on AMD hardware
    ONNX      (module 10) ← ONNX is MIGraphX's primary and most complete import format
    TFLite    (module 11) ← TFLite targets mobile; MIGraphX targets AMD datacenter GPU
    OpenVINO  (module 12) ← OpenVINO targets Intel; MIGraphX targets AMD; both are vendor-specific runtimes
    Core ML   (module 13) ← Core ML targets Apple ANE; MIGraphX targets AMD GPU; no hardware overlap
    MIGraphX  (this)      ← AMD's inference engine: ROCm-native, ONNX-first, open-source

"""

import textwrap
import re

TOPIC_NAME   = "MIGraphX — AMD's GPU Inference Engine"
DISPLAY_NAME = "14a · MIGraphX"
ICON         = "🔴"
SUBTITLE     = "ROCm Runtime, ONNX Import, Operator Fusion, INT8 Quantisation, and MIOpen Dispatch"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY MIGRAPHX EXISTS: AMD'S INFERENCE PROBLEM

### The NVIDIA Inference Monopoly (and Why It Existed)

    By 2016, deep learning inference on GPU meant NVIDIA + TensorRT.
    The combination was unassailable for three reasons:

    CUDA ECOSYSTEM LOCK-IN:
        PyTorch, TensorFlow, JAX all targeted CUDA as their primary GPU backend.
        Libraries like cuDNN, cuBLAS, NCCL were NVIDIA-exclusive.
        A model trained in PyTorch could not run on AMD hardware without
        porting every kernel — a prohibitive effort for any organisation.

    TENSORRT PERFORMANCE ADVANTAGE:
        TensorRT's closed-source kernel library was hand-tuned for each NVIDIA
        architecture (Turing, Ampere, Hopper). AMD had no equivalent.
        The gap was not small: TensorRT on an A100 could be 3–5× faster
        than naive ONNX Runtime on the same hardware.

    DRIVER + HARDWARE ECOSYSTEM:
        CUDA drivers were mature, stable, and ubiquitous in cloud VMs.
        AMD GPU availability in cloud (AWS, GCP, Azure) was limited.

### The ROCm Strategy and MIGraphX's Role

    AMD's response was ROCm (Radeon Open Compute) — an open-source GPU
    compute platform designed to make AMD GPUs viable for ML workloads.

    ROCm provides:
        HIP:              a CUDA-compatible programming language.
                          Most CUDA code can be ported via hipify.
        rocBLAS:          BLAS (matrix multiply) for AMD GPU.
        MIOpen:           convolution, normalisation, pooling, attention kernels.
        composable_kernel: hand-tuned GEMM/conv kernels for CDNA architecture.
        RCCL:             collective communications (AMD's NCCL).
        ROCm-aware MPI:   distributed training and inference.

    MIGraphX is the INFERENCE ENGINE layer on top of ROCm:
        It accepts trained models (ONNX, TF, PyTorch).
        It compiles them to sequences of ROCm kernel dispatches.
        It applies graph-level fusion and layout optimisations that no
        individual library call can do.
        It produces a compiled binary (.mxr) for production deployment.

### AMD Hardware: RDNA vs CDNA

    AMD ships two architectures that MIGraphX targets:

    CDNA (Compute DNA) — datacenter inference target:
        MI100 (CDNA1, 2020):  23.1 TFLOPS FP32, 184 TOPS INT8
        MI200 series (CDNA2, 2021): 47.9 TFLOPS FP64, ~340 TOPS INT8
                              HBM2e memory (Frontier supercomputer GPU)
        MI300X (CDNA3, 2023): 163.4 TFLOPS FP16, 1.3 POPS INT8 (FP8)
                              192 GB HBM3 unified memory — the largest GPU memory ever
                              AMD's direct answer to H100
        MI300A (CDNA3, 2023): CPU+GPU APU design, 128 GB HBM3 unified memory
        MI325X (CDNA3+, 2024): improved memory bandwidth

    RDNA (Render DNA) — consumer/workstation (limited inference support):
        RX 7900 XTX (RDNA3): 61 TFLOPS FP16, no native INT8 matrix unit
        ROCm supports selected RDNA3 GPUs but MIGraphX is CDNA-optimised.

    MI300X key specs relevant to inference:
        HBM3: 5.3 TB/s memory bandwidth (vs H100's 3.35 TB/s)
        192 GB: fits Llama-3 70B in FP16 on a single GPU (H100 SXM needs 2)
        INT8: 2.6 POPS (2,600 TOPS) via matrix units
        FP8:  2.6 PFLOPS (inference-oriented training)

### MIGraphX vs TensorRT: The Key Differences

    OPEN SOURCE:
        MIGraphX: fully open source (Apache 2.0, github.com/ROCm/AMDMIGraphX)
        TensorRT: closed-source (binary only, NVIDIA proprietary)
        Impact: MIGraphX fusion logic is auditable; TensorRT is a black box.

    OPERATOR COVERAGE:
        Both support: CNN ops, transformer ops, basic NLP
        MIGraphX gaps vs TensorRT: fewer custom plugin mechanisms,
        some exotic ops unsupported or falling to slower fallbacks
        MIGraphX advantage: ONNX import is the primary path (no TF conversion needed)

    COMPILATION SPEED:
        TensorRT: notorious for very long compile times (minutes for large models)
        MIGraphX: faster compile (seconds to tens of seconds for most models)
        because MIGraphX uses exhaustive tuning as opt-in, not mandatory.

    INTEGRATION:
        TensorRT: integrates with NVIDIA Triton, CUDA, TF-TRT, Torch-TensorRT
        MIGraphX: integrates with PyTorch via torch_migraphx,
                  ONNX Runtime via ROCm execution provider,
                  and ROCm-native C++/Python APIs


##### PART 2 — THE MIGRAPHX PROGRAM MODEL: IR, MODULES, AND SHAPES

### MIGraphX's Internal Representation

    MIGraphX represents a model as a PROGRAM (migraphx::program).
    A program contains one or more MODULES (migraphx::module).
    The main execution path lives in the "main" module.
    Sub-graphs (for if/loop ops) live in named sub-modules.

    PROGRAM STRUCTURE:
        program
        └── module["main"]
            ├── instruction 0:  @param "input"  shape<float, {1,3,224,224}>
            ├── instruction 1:  @literal <weights>  (weight tensor)
            ├── instruction 2:  convolution(0, 1)  → shape<float, {1,64,112,112}>
            ├── instruction 3:  relu(2)             → shape<float, {1,64,112,112}>
            ├── instruction 4:  @literal <bn_scale> ...
            ├── instruction 5:  batch_norm_inference(3, 4, ...)
            │   ...
            └── instruction N:  @return (softmax output)

    INSTRUCTION (migraphx::instruction):
        Each instruction has:
            operator:   the operation (convolution, relu, batch_norm_inference, ...)
            inputs:     list of instructions this reads from (the def-use graph)
            output_shape: migraphx::shape describing the result tensor
            name:       optional debug name

        Instructions form a DIRECTED ACYCLIC GRAPH (DAG).
        Execution order: topological sort of the instruction DAG.
        No explicit control flow at the instruction level
        (control flow is handled via sub-modules + if/loop ops).

### migraphx::shape

    Every value in MIGraphX has a migraphx::shape that carries:
        TYPE (element type):
            float_type   (float32)
            half_type    (float16)
            double_type  (float64)
            int8_type    (int8)
            uint8_type   (uint8)
            int16_type   (int16)
            int32_type   (int32)
            int64_type   (int64)
            bool_type    (bool)

        LENS (dimensions):
            A std::vector<std::size_t> of dimension sizes.
            e.g., {1, 3, 224, 224} for a batch-1 ImageNet input.

        STRIDES:
            A std::vector<std::size_t> for non-contiguous layout.
            Default: row-major strides computed from lens.
            Transposed matrix: non-default strides (no data copy needed).
            Broadcasting: stride of 0 along a broadcast dimension.

    Python API:
        import migraphx
        s = migraphx.shape(type=migraphx.shape.float_type,
                           lens=[1, 3, 224, 224])
        print(s.type())    ; float_type
        print(s.lens())    ; [1, 3, 224, 224]
        print(s.strides()) ; [150528, 50176, 224, 1]
        print(s.elements()) ; 150528

    DYNAMIC SHAPES (MIGraphX 2.8+):
        migraphx::dynamic_dimension allows dimensions to be specified as
        a range [min, max] rather than a fixed value:

        Python:
            dd = migraphx.dynamic_dimension(min=1, max=16)  ; batch 1–16
            ds = migraphx.shape(type=migraphx.shape.float_type,
                                dyn_dims=[dd,
                                          migraphx.dynamic_dimension(3),
                                          migraphx.dynamic_dimension(224),
                                          migraphx.dynamic_dimension(224)])

        To use dynamic shapes, set when parsing:
            prog = migraphx.parse_onnx(
                "model.onnx",
                map_input_dims={"input": [[1,16], [3], [224], [224]]})

### Operator Coverage

    MIGraphX implements approximately 120 operators. Key operators:

    CONVOLUTION:
        convolution:         standard N-D convolution (groups supported)
        convolution_backwards: gradient conv (for training or encoder-decoder)
        deconvolution:       transposed convolution
        dot:                 general matmul (batch dimensions supported)

    NORMALISATION:
        batch_norm_inference: inference-mode BatchNorm (fuses into conv)
        batch_norm_training:  training-mode BatchNorm (for training)
        layer_norm:           layer normalisation
        group_norm:           group normalisation
        instance_norm:        instance normalisation
        lrn:                  local response normalisation (AlexNet-era)

    POOLING:
        pooling:       max/average/lpnorm pooling (configurable)
        avg_pool:      average pooling shortcut

    ATTENTION:
        multihead_attention:  fused multi-head attention (from ONNX opset 17)
        dot:                  used for Q*K^T and (Q*K^T)*V
        softmax:              numerically stable softmax (logsumexp)

    ELEMENTWISE:
        add, sub, mul, div, pow, sqrt, exp, log, abs, neg, sign
        relu, leaky_relu, prelu, elu, selu, celu, hard_sigmoid
        sigmoid, tanh, gelu (exact and tanh approximation)
        clip, floor, ceil, round, isnan, isinf
        equal, less, greater, and, or, not
        where (ternary select), max (elementwise), min (elementwise)

    REDUCTION:
        reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod
        all with axes and keepdims attributes

    SHAPE / INDEXING:
        reshape, flatten, unsqueeze, squeeze, transpose
        broadcast, multibroadcast (handles all ONNX broadcast semantics)
        concat, split, slice, gather, scatter_elements, scatter_nd
        pad, tile, expand, reverse

    RECURRENT:
        rnn, lstm, gru (all with bidirectional support)

    QUANTISATION:
        quantizelinear, dequantizelinear (ONNX QDQ format)
        The quantised path: float → quantize → INT8 matmul → dequantize → float


##### PART 3 — IMPORTING MODELS: ONNX, TENSORFLOW, AND PYTORCH

### ONNX Import (Primary Path)

    ONNX is MIGraphX's primary and most complete import format.
    The parser maps ONNX operators directly to MIGraphX operators.
    Supported opsets: 1 through 17 (most ops), with partial opset 18+.

    BASIC IMPORT:
        import migraphx

        # Load and parse an ONNX model
        prog = migraphx.parse_onnx("resnet50.onnx")
        print(prog)   ; prints the instruction list

        # Parse with custom input shapes (overrides shapes in the ONNX file):
        prog = migraphx.parse_onnx(
            "model.onnx",
            map_input_dims={"input": [1, 3, 224, 224]})
        ; Always specify input shapes when the ONNX file has dynamic dims.

    DYNAMIC SHAPE IMPORT:
        # Parse with a range for the batch dimension:
        prog = migraphx.parse_onnx(
            "model.onnx",
            map_input_dims={"input": [[1, 8], [3], [224], [224]]})
        ; [[1,8], [3], [224], [224]] means batch ∈ [1,8], others fixed.
        ; MIGraphX compiles a single kernel that handles any batch in range.

    EXTERNAL DATA (models > 2 GB):
        ONNX models with external weight files are supported:
        prog = migraphx.parse_onnx("model.onnx",
                                    external_data_dir="/path/to/weights/")
        ; The external .bin files referenced in the ONNX are read from
        ; external_data_dir rather than alongside the .onnx file.

    CHECKING IMPORT RESULTS:
        # Print the MIGraphX IR after import:
        print(prog)

        # Access instructions:
        main_mod = prog.get_main_module()
        for ins in main_mod:
            print(f"  {ins.name():<30s} {ins.get_shape()}")

        # Find instructions by operator name:
        conv_ins = [i for i in main_mod if i.name() == "convolution"]
        print(f"  Convolution layers: {len(conv_ins)}")

### TensorFlow Import

    MIGraphX can import TensorFlow frozen graphs (.pb files):

        prog = migraphx.parse_tf(
            "model.pb",
            is_nhwc=True,               ; TF default: channels-last
            batch_size=1,
            output_names=["output/Softmax"])

    TF import limitations:
        Only frozen graphs (not SavedModels) are supported directly.
        For TF 2.x SavedModels: convert to ONNX first (tf2onnx), then use parse_onnx.
        TF import covers most classification and detection models
        but has less coverage than the ONNX parser.

### PyTorch Integration: torch_migraphx

    torch_migraphx is a TorchScript/torch.compile backend for MIGraphX.
    It intercepts PyTorch models and compiles subgraphs with MIGraphX.

    INSTALLATION:
        pip install torch_migraphx   ; AMD ROCm environment required

    TORCH.COMPILE BACKEND:
        import torch
        import torch_migraphx

        model = MyModel().eval().cuda()   ; AMD GPU

        # Compile with MIGraphX backend:
        compiled = torch.compile(model, backend="migraphx")

        # Run — graph captured by Dynamo, compiled by MIGraphX:
        output = compiled(input_tensor)

    TORCHSCRIPT DIRECT PATH:
        import torch_migraphx

        traced  = torch.jit.trace(model.eval(), example_input)
        mg_model = torch_migraphx.compile(traced, inputs=[example_input])
        output  = mg_model(input_tensor)

    HOW torch_migraphx WORKS:
        1. TorchDynamo (or TorchScript) captures the FX graph.
        2. torch_migraphx receives the FX graph as a backend.
        3. It converts the FX graph to ONNX internally.
        4. It passes the ONNX to migraphx.parse_onnx.
        5. It compiles the program for the target AMD GPU.
        6. It wraps the compiled program as a callable PyTorch module.
        The conversion through ONNX is transparent; from the user's
        perspective it looks like a standard torch.compile backend.

### Saving and Loading Compiled Programs

    MIGraphX compiled programs can be serialised to binary (.mxr files)
    for fast loading in production (avoids recompilation on every startup).

    SAVE:
        import migraphx

        prog = migraphx.parse_onnx("model.onnx")
        migraphx.compile(prog, migraphx.get_target("gpu"))
        prog.save("model.mxr")   ; save compiled binary

    LOAD:
        prog = migraphx.load("model.mxr")   ; fast load, no recompilation
        ; Load time: milliseconds (vs seconds for recompilation)

    .MXR FORMAT:
        The .mxr file is a MIGraphX-specific binary format (NOT ONNX).
        It contains:
            The compiled instruction sequence.
            The GPU kernel binaries (ROCm/HIP compiled code).
            Weight tensors (the model's parameters).
        The .mxr is ARCHITECTURE-SPECIFIC:
            A program compiled for MI300X will not run on MI100.
            Recompile when changing GPU architecture.
        Version compatibility:
            The .mxr is tied to the MIGraphX version that created it.
            Upgrade MIGraphX → recompile all .mxr files.


##### PART 4 — THE COMPILATION PIPELINE: PASSES AND OPTIMISATIONS

### Compilation Overview

    migraphx.compile(prog, target) triggers the full compilation pipeline.
    The pipeline is a sequence of PASSES — each pass transforms the
    instruction DAG, aiming to produce a faster equivalent program.

    Python API:
        target = migraphx.get_target("gpu")   ; or "cpu" for CPU target
        migraphx.compile(prog, target)
        ; prog is modified in-place; after this call it is ready to run.

    With options:
        migraphx.compile(prog, target,
            migraphx.compile_options(
                offload_copy=True,    ; auto-copy inputs to GPU and outputs back
                fast_math=False,     ; allow unsafe math optimisations (e.g. ftz)
            ))

    After compilation:
        prog is no longer the ONNX-imported instruction sequence.
        It is now a sequence of MIOpen/rocBLAS/composable_kernel calls
        interspersed with HIP kernel launches for fused elementwise chains.

### Pass 1 — Simplification and Normalisation

    DEAD CODE ELIMINATION:
        Remove any instruction whose result is never used.
        Triggered most often by: constant-folded subgraphs, optional outputs.

    CONSTANT PROPAGATION:
        Evaluate operations with all-constant inputs at compile time.
        Removes: positional encodings that are fixed, lookup tables,
                 shape computation subgraphs.

    IDENTITY ELIMINATION:
        Remove redundant reshape/transpose pairs that cancel each other:
            transpose([0,1,2,3]) → identity (permute to same order)
            reshape([N,C,H,W]) then reshape([N,C,H,W]) → identity

    COMMON SUBEXPRESSION ELIMINATION (CSE):
        Detect duplicate computations and collapse to single instruction.
        Common in transformers where Q, K, V all apply the same pre-normalization.

    SIMPLIFY ALGEBRA:
        Arithmetic identities:
            x + 0   → x
            x * 1   → x
            x / 1   → x
            0 * x   → 0
            1^x     → 1
        Shape identities:
            broadcast(x, [1,1]) where x has shape [1,1] → x (no broadcast needed)

### Pass 2 — BatchNorm and Bias Fusion

    BATCH_NORM INTO CONVOLUTION:
        BatchNormInference following a Convolution is provably equivalent to a
        single modified Convolution with scaled weights and shifted biases.
        MIGraphX folds the four BN parameters (scale, bias, mean, variance)
        directly into the Conv weights and bias at compile time.

        After folding:
            W_new = W * (scale / sqrt(variance + eps))
            b_new = (bias - mean) * (scale / sqrt(variance + eps)) + bn_bias
        The BN instruction is removed entirely.
        The Conv now carries the folded weights as its literal inputs.
        RESULT: one fewer kernel launch per BN layer at inference time.

    BIAS ADD INTO MATMUL/CONV:
        If a bias Add follows immediately after a MatMul or Convolution,
        the Add is merged into the bias argument of the underlying
        rocBLAS/MIOpen call. rocBLAS supports alpha*A*B + beta*C (GEMM),
        where C is the bias tensor. This eliminates a separate Add kernel.

### Pass 3 — Operator Fusion (the Core Optimisation)

    Fusion is MIGraphX's primary performance lever. It reduces kernel
    launch overhead and eliminates intermediate buffers in HBM.

    POINTWISE FUSION:
        Contiguous elementwise operations are fused into a single HIP kernel.
        Rules:
            Producer P and consumer C can fuse if:
                P is elementwise (add, relu, exp, mul, gelu, etc.)
                C reads only P's output (no other consumers of P's tensor)
        The fused kernel reads inputs once, writes outputs once.
        All intermediate values live in GPU registers.

        Example: add → relu → mul → exp (4 separate ops)
        Fused:   one HIP kernel, reads 2 inputs, writes 1 output.

    REDUCTION + ELEMENTWISE FUSION:
        An elementwise producer that feeds a reduction is inlined:
            sum(exp(x))  →  one kernel: load x, compute exp, accumulate sum.
        The intermediate exp tensor is never written to HBM.

    ATTENTION PATTERN FUSION (MIGraphX 2.5+):
        MIGraphX detects the scaled-dot-product-attention pattern:
            Q @ K^T → scale → mask → softmax → @ V
        and replaces it with a single fused attention kernel backed by
        MIOpen's fused attention implementation (or composable_kernel).
        This enables flash-attention-like memory efficiency on AMD GPUs.

    GELU FUSION:
        Both the exact GELU and the tanh approximation are detected and
        fused from their multi-op forms into single kernel calls:
            Exact:    x * 0.5 * (1 + erf(x/sqrt(2)))
            Approx:   x * sigmoid(1.702 * x)  or tanh-based

    CONV + ACTIVATION FUSION:
        MIOpen supports activation epilogues for convolution kernels.
        relu(conv(x, W)) is compiled to a single MIOpen convolution call
        with the activation function specified as an epilogue parameter.
        No separate relu kernel is launched.

### Pass 4 — Layout Optimisation

    NCHW vs NHWC:
        By default, MIGraphX uses NCHW (channels-first) internally,
        matching PyTorch's convention.
        MIOpen can run convolutions in either layout.
        The pass selects the layout that MIOpen prefers for each conv
        (depends on filter sizes, batch size, and GPU architecture).
        If layout switches are needed, transpose ops are inserted.

    LAYOUT PROPAGATION:
        The layout pass propagates layout decisions forward through the graph.
        Goal: minimise the number of transpose ops by keeping consecutive ops
        in the same preferred layout.

### Pass 5 — Kernel Selection

    After fusion and layout decisions, each instruction is lowered to a
    specific GPU kernel call.

    AVAILABLE KERNELS:
        MIOpen:           Conv, pooling, batch_norm, LSTMs, attention
        rocBLAS:          GEMM (MatMul, FC layers), trsm, symm
        composable_kernel: highly-tuned GEMM/conv for CDNA2/CDNA3 architectures
                           specifically designed for MI250/MI300 performance
        HIP custom kernel: generated for fused elementwise chains
        Reference CPU:    fallback for unsupported ops (rare, slow)

    KERNEL SELECTION LOGIC:
        For each MatMul of size (M, N, K), MIGraphX looks up in its
        tuning database for the optimal rocBLAS/composable_kernel configuration.
        If no tuning entry: uses heuristic selection.
        With exhaustive tuning (--exhaustive_tune): benchmarks all candidates.

### Pass 6 — Memory Planning

    After kernel selection, MIGraphX assigns buffer memory for all intermediate
    tensors. The memory planner works like TFLite's arena planner:

        1. Compute instruction LIFETIMES: [first_use, last_use] for each tensor.
        2. Greedily assign memory locations, reusing memory when lifetimes
           do not overlap.
        3. Allocate one large GPU buffer (the workspace) for all intermediates.
        4. Weight tensors are mapped to separate read-only GPU memory.

    After memory planning: inference requires ZERO dynamic GPU allocations.
    Every buffer is pre-assigned to a fixed offset in the workspace.


##### PART 5 — THE PYTHON AND C++ RUNTIME API

### Python API: End-to-End Inference

    FULL WORKFLOW:
        import migraphx
        import numpy as np

        # Step 1: Import the model
        prog = migraphx.parse_onnx("resnet50.onnx",
                                    map_input_dims={"data": [1, 3, 224, 224]})

        # Step 2: Compile for GPU
        target = migraphx.get_target("gpu")
        migraphx.compile(prog, target)

        # Step 3: Prepare input
        x_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # Step 4: Run inference
        results = prog.run({"data": migraphx.to_gpu(x_np)})

        # Step 5: Read output (copies from GPU to CPU)
        logits = np.array(results[0])
        top1   = np.argmax(logits)

    ALTERNATIVE: run_async + synchronise:
        # Fire-and-forget (overlaps with CPU work):
        results = prog.run({"data": gpu_tensor}, run_async=True)
        # Do other work here...
        results.synchronize()   ; wait for GPU to finish
        output = np.array(results[0])

    ALTERNATIVE: compile with offload_copy (simpler but slower for pipelining):
        migraphx.compile(prog, target,
            migraphx.compile_options(offload_copy=True))
        ; Now prog.run accepts plain numpy arrays directly:
        results = prog.run({"data": x_np})   ; auto-copies to/from GPU
        output  = np.array(results[0])

### GPU Memory Management

    Without offload_copy, the caller manages GPU memory explicitly.
    This enables zero-copy pipelining:

        UPLOADING DATA TO GPU:
            gpu_tensor = migraphx.to_gpu(numpy_array)
            ; Creates a GPU allocation and copies numpy_array to it.

        CREATING AN EMPTY GPU BUFFER:
            shape     = migraphx.shape(type=migraphx.shape.float_type,
                                       lens=[1, 1000])
            gpu_empty = migraphx.allocate(shape)

        DOWNLOADING RESULTS FROM GPU:
            cpu_array = np.array(gpu_tensor)   ; copies GPU → CPU
            ; OR:
            result_np = migraphx.from_gpu(gpu_tensor)

        PRE-ALLOCATED OUTPUT BUFFERS:
            # For high-throughput serving: pre-allocate output buffers
            # and reuse them across calls to avoid GPU malloc overhead.
            out_shape  = prog.get_output_shapes()[0]
            output_buf = migraphx.allocate(out_shape)
            prog.run({"data": input_gpu}, outputs=[output_buf])
            ; output_buf is updated in-place; no allocation per call.

### Pipelined Inference for Maximum Throughput

    For serving, overlapping data transfer with GPU computation maximises
    throughput. The pattern with MIGraphX:

        import threading, queue

        result_queue = queue.Queue()

        def inference_worker(prog, input_queue):
            while True:
                item_id, data = input_queue.get()
                if data is None: break
                gpu_in  = migraphx.to_gpu(data)
                results = prog.run({"data": gpu_in}, run_async=True)
                results.synchronize()
                result_queue.put((item_id, np.array(results[0])))

        # Double-buffered pipeline:
        input_queue = queue.Queue(maxsize=2)
        worker = threading.Thread(target=inference_worker,
                                   args=(prog, input_queue))
        worker.start()

        for i, batch in enumerate(dataset):
            input_queue.put((i, batch))

        input_queue.put((None, None))   ; sentinel
        worker.join()

### C++ API

    MIGraphX's primary production API is C++. The Python bindings wrap it.

        #include <migraphx/migraphx.hpp>
        #include <migraphx/onnx.hpp>
        #include <migraphx/target.hpp>

        // Parse ONNX
        migraphx::program prog = migraphx::parse_onnx(
            "resnet50.onnx",
            migraphx::onnx_options{}.set_input_dimensions(
                "data", {1, 3, 224, 224}));

        // Compile
        prog.compile(migraphx::make_target("gpu"));

        // Create parameter map
        migraphx::program_parameters params;
        migraphx::argument input_arg = ...;   // created from raw GPU pointer
        params.add("data", input_arg);

        // Run
        auto results = prog.eval(params);

        // Read output
        float* output_ptr = results[0].cast<float>();

    C++ MIGRAPHX::ARGUMENT:
        An argument wraps a raw pointer + shape:
        migraphx::argument arg{shape, data_ptr};
        ; data_ptr can be a GPU pointer (hipMalloc result) or CPU pointer.
        ; MIGraphX does NOT own the memory — the caller is responsible.

### The migraphx_driver CLI

    MIGraphX ships a command-line tool (migraphx-driver) for testing and
    benchmarking without writing any code:

        # Compile and run a model:
        migraphx-driver run --onnx resnet50.onnx \
                            --input-dim @data 1 3 224 224

        # Benchmark (repeated runs, report throughput):
        migraphx-driver perf --onnx resnet50.onnx \
                             --input-dim @data 1 3 224 224 \
                             --iterations 100

        # Print the compiled instruction sequence:
        migraphx-driver print --onnx resnet50.onnx \
                              --input-dim @data 1 3 224 224

        # Check correctness vs reference (CPU):
        migraphx-driver verify --onnx resnet50.onnx \
                               --input-dim @data 1 3 224 224

        # Compile and save to .mxr:
        migraphx-driver compile --onnx resnet50.onnx \
                                --input-dim @data 1 3 224 224 \
                                --output resnet50.mxr

        # Benchmark a pre-compiled .mxr:
        migraphx-driver perf --migraphx resnet50.mxr


##### PART 6 — MIOPEN AND THE ROCM KERNEL STACK

### MIOpen: AMD's cuDNN Equivalent

    MIOpen (Machine Intelligence Library for OpenCL/HIP) is AMD's open-source
    deep learning primitive library — the layer that sits between MIGraphX
    and the raw GPU hardware.

    MIOpen provides:
        Convolution forward and backward (all paddings, dilations, groups)
        Pooling (max, average, adaptive)
        Batch normalisation (forward/backward)
        Activation functions (relu, sigmoid, tanh, elu, leaky_relu, softplus, ...)
        Local response normalisation
        LSTM, GRU, RNN (all variants, bidirectional)
        Softmax
        Fused operations (conv+BN+activation in one call)
        Multi-head attention (MIOpen 2.19+, composable_kernel backed)

    MIOpen is the AMD equivalent of NVIDIA's cuDNN.
    MIGraphX dispatches to MIOpen for all convolution and pooling operations.

### MIOpen Auto-Tuning (the "Find" Database)

    Like cuDNN, MIOpen supports multiple algorithm implementations for each
    operation. The optimal algorithm depends on:
        Input shape (N, C, H, W)
        Filter shape (K, C, R, S)
        Stride, padding, dilation
        GPU architecture (MI100 vs MI250 vs MI300X)
        Precision (FP32, FP16, INT8)

    MIOpen's FIND DATABASE stores per-shape algorithm selections.
    The database lives at:
        ~/.config/miopen/  (user-level cache)
        /etc/miopen/       (system-level cache, for shared deployments)
        MIOPEN_USER_DB_PATH env var overrides the location.

    FINDING THE BEST ALGORITHM:
        First call with a new shape: MIOpen benchmarks candidate algorithms
        and records the winner in the find database.
        Subsequent calls: loads from the database (zero benchmarking overhead).

    CONTROLLING FIND BEHAVIOUR:
        MIOPEN_FIND_MODE=1  (default): benchmark on first encounter, cache result
        MIOPEN_FIND_MODE=0: always use heuristic (no benchmarking)
        MIOPEN_FIND_MODE=2: always benchmark, never cache (reproducible but slow)

    MIGraphX exposes this via compile options:
        migraphx.compile(prog, target,
            migraphx.compile_options(exhaustive_tune=True))
        ; exhaustive_tune=True: benchmark ALL candidate algorithms per op
        ; This can take minutes for the first compilation of a model
        ; but saves to the find database for subsequent runs.

### rocBLAS: GEMM for Dense Layers and Transformers

    rocBLAS is AMD's BLAS library — the core of all matrix-matrix multiply
    operations in MIGraphX. Every transformer attention dot-product, every
    fully-connected layer, and every linear projection uses rocBLAS.

    MIGraphX dispatches to rocBLAS for:
        dot (general matmul, any batch dims)
        Fused GEMM+bias (via rocBLAS extension API)
        INT8 GEMM (for quantised inference)

    rocBLAS algorithm selection:
        Similar to MIOpen: rocBLAS maintains a kernel database keyed by
        (M, N, K, alpha, beta, layout) shape.
        rocBLAS-bench tool: standalone benchmarking for specific GEMM shapes.

### composable_kernel: Highly-Tuned CDNA Kernels

    composable_kernel (CK) is AMD's library of hand-tuned GPU kernels
    specifically designed for the CDNA2/CDNA3 architecture (MI250, MI300X).

    CK differs from MIOpen and rocBLAS:
        MIOpen / rocBLAS: auto-generated kernels, tuned at a macro level.
        CK: hand-written kernel templates that directly exploit CDNA
            memory hierarchy (LDS, shared memory, register file layout).
            CK kernels achieve close-to-theoretical peak throughput.

    CK provides:
        Fused GEMM + activation epilogues (add_relu, add_gelu, etc.)
        Fused multi-head attention (flash-attention style, O(N) memory)
        Mixed-precision GEMM (FP16 accumulate, INT8 compute, BF16)
        Batched GEMM patterns for transformer attention heads

    MIGraphX uses CK kernels:
        When MIGraphX determines a GEMM shape will benefit from CK
        vs rocBLAS, it dispatches to CK.
        Typically: large batch sizes, attention head patterns.
        The selection is automatic based on shape heuristics.

### The ROCm Software Stack Diagram

    ┌──────────────────────────────────────────────────────────────────────┐
    │  USER LEVEL                                                           │
    │  MIGraphX (this module)     PyTorch (ROCm)      TensorFlow (ROCm)   │
    └──────────────────────────────┬────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────▼────────────────────────────────────────┐
    │  COMPUTE PRIMITIVE LAYER                                               │
    │  MIOpen (conv/pool/BN/attn)  rocBLAS (GEMM)  composable_kernel (CDNA)│
    │  RCCL (collectives)          rocRAND (random)  rocSPARSE              │
    └──────────────────────────────┬────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────▼────────────────────────────────────────┐
    │  HIP RUNTIME LAYER                                                     │
    │  HIP API (hipMalloc, hipMemcpy, hipLaunchKernel, ...)                  │
    │  hipcc compiler (LLVM-based, generates AMD GCN/CDNA/RDNA ISA)        │
    └──────────────────────────────┬────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────▼────────────────────────────────────────┐
    │  AMD GPU HARDWARE                                                      │
    │  MI300X (CDNA3)   MI250 (CDNA2)   MI100 (CDNA1)   RX 7900 (RDNA3)   │
    └────────────────────────────────────────────────────────────────────────┘


##### PART 7 — QUANTISATION: INT8 AND FP8 INFERENCE

### INT8 Quantisation in MIGraphX

    MIGraphX supports INT8 inference via the ONNX QDQ (Quantize-Dequantize)
    format. The QDQ format inserts explicit QuantizeLinear and DequantizeLinear
    operators in the ONNX graph to annotate quantisation points.

    QDQ FORMAT:
        float_input → QuantizeLinear (scale, zero_point) → int8_tensor
        int8_tensor → MatMul (INT8 kernel) → int32_output
        int32_output → DequantizeLinear (scale, zero_point) → float_output

    WORKFLOW — GENERATE A QUANTISED ONNX MODEL:
        Use any standard ONNX quantisation tool to produce the QDQ model.
        Recommended tools:
            ONNX Runtime: onnxruntime.quantization.quantize_static()
            Intel Neural Compressor: inc.compress(...)
            Brevitas (for QAT): then export to ONNX QDQ format

        Then load into MIGraphX normally:
            prog = migraphx.parse_onnx("model_int8_qdq.onnx",
                                        map_input_dims={"input": [1, 3, 224, 224]})
            migraphx.compile(prog, migraphx.get_target("gpu"))
            ; MIGraphX recognises the QDQ pattern and compiles to INT8 kernels.

    WHAT MIGRAPHX DOES WITH QDQ:
        1. Detects QuantizeLinear → MatMul/Conv → DequantizeLinear patterns.
        2. Fuses them: the QDQ nodes are removed.
        3. The MatMul/Conv is compiled to INT8 (rocBLAS INT8 or CK INT8 GEMM).
        4. Scale/zero_point are absorbed into the kernel as epilogue parameters.
        5. The output is dequantised at the rocBLAS level (no separate kernel).

    VERIFICATION AFTER QUANTISATION:
        migraphx-driver verify --onnx model_int8_qdq.onnx \
                               --input-dim @input 1 3 224 224 \
                               --tolerance 0.01
        ; Compares INT8 output vs FP32 reference.
        ; tolerance=0.01 is typical for INT8 models.

### FP8 Inference (MIGraphX 2.10+ / MI300X)

    AMD's MI300X (CDNA3) natively supports FP8 (8-bit floating point) in
    hardware, delivering up to 1.3 PFLOPS FP8 peak throughput.
    Two FP8 formats are supported:
        E4M3: 4-bit exponent, 3-bit mantissa — better range, used for activations
        E5M2: 5-bit exponent, 2-bit mantissa — better for gradients (training)

    FP8 inference in MIGraphX:
        Currently most accessible via models exported with FP8 quantisation
        tools (e.g., AMD's Quark quantiser or NVIDIA's modelopt for ONNX FP8).
        The ONNX FP8 format uses Cast to/from float8e4m3fn dtype.

        MIGraphX compiles Cast + MatMul patterns to native FP8 GEMM
        using composable_kernel FP8 kernels.

    MEMORY SAVINGS (Llama-3-70B):
        FP16:   ~140 GB  (needs 2× MI300X)
        FP8:    ~70 GB   (fits on single MI300X with 192 GB HBM3)
        INT4:   ~35 GB   (future; not yet fully supported in MIGraphX)

### AMD Quark: Production Quantisation Toolkit

    AMD Quark is AMD's post-training quantisation tool, analogous to
    NVIDIA's modelopt. It produces ONNX QDQ models optimised for MIGraphX.

        pip install amd-quark

        from quark.onnx import ModelQuantizer, QuantizationConfig

        config = QuantizationConfig(
            quant_format="QDQ",
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method="MinMax",  ; or "Percentile", "MSE"
        )
        quantizer = ModelQuantizer(config)
        quantized_model = quantizer.quantize_model(
            model=onnx_model,
            save_path="model_int8.onnx",
            calibration_data_reader=calib_reader,
        )

    AMD Quark also supports:
        SmoothQuant: smooths activation outliers before INT8 quantisation,
                     significantly improving accuracy on LLMs.
        AWQ (Activation-aware Weight Quantisation): INT4 weight quantisation
            for LLMs with < 1% accuracy drop on most benchmarks.
        GPTQ: group-quantised INT4 for LLM weights.


##### PART 8 — ONNX RUNTIME WITH ROCM EXECUTION PROVIDER

### Why Use ORT + MIGraphX Together

    ONNX Runtime with the MIGraphX Execution Provider (EP) combines:
        ORT's operator coverage (250+ ops, including complex control flow)
        MIGraphX's AMD-optimised kernel library

    The EP architecture:
        1. ORT loads the ONNX model.
        2. ORT queries the MIGraphX EP: "which ops can you run?"
        3. MIGraphX EP claims subgraphs it supports.
        4. ORT partitions the graph: MIGraphX subgraphs + CPU fallback.
        5. At runtime: ORT dispatches each subgraph to the right backend.

    This is the same EP partitioning mechanism as ONNX Runtime's CUDA EP —
    just targeting AMD ROCm instead of NVIDIA CUDA.

### Installation and Setup

    INSTALL ORT WITH ROCM:
        # For ROCm 5.7/6.0 with AMD GPU:
        pip install onnxruntime-rocm

        # OR build from source for latest ROCm:
        # https://github.com/microsoft/onnxruntime/blob/main/docs/build/eps.md#amd-migraphx

    BASIC USAGE:
        import onnxruntime as ort
        import numpy as np

        # Create session with MIGraphX EP
        sess = ort.InferenceSession(
            "model.onnx",
            providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"])

        inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out = sess.run(None, {"input": inp})[0]

    PROVIDER CONFIGURATION:
        provider_options = {
            "device_id": 0,               ; which GPU (0-indexed)
            "exhaustive_tune": False,      ; enable MIOpen exhaustive tuning
            "fp16_enable": True,          ; cast FP32 model to FP16 for GPU
        }
        sess = ort.InferenceSession(
            "model.onnx",
            providers=[("MIGraphXExecutionProvider", provider_options),
                       "CPUExecutionProvider"])

    CHECKING WHICH PROVIDER EXECUTED EACH NODE:
        # After running, inspect which EP handled each node:
        for i, node in enumerate(sess.get_providers()):
            print(f"  Provider {i}: {node}")
        ; Unlike TensorRT EP, MIGraphX EP does not expose per-node EP info
        ; directly, but you can use ORT's session-level profiling.

### ORT Profiling on AMD GPUs

    ENABLE PROFILING:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = "/tmp/ort_migraphx_profile"
        sess = ort.InferenceSession("model.onnx", sess_options=opts,
                                     providers=["MIGraphXExecutionProvider"])
        out    = sess.run(None, {"input": inp})
        profile_path = sess.end_profiling()

    The output JSON (chrome://tracing format) shows:
        Which ops ran on MIGraphX EP vs CPU fallback.
        Per-op execution times including HIP kernel overhead.
        Memory transfer events (host-to-device for inputs).


##### PART 9 — PERFORMANCE TUNING AND EXHAUSTIVE TUNE

### The Tuning Database Architecture

    MIGraphX performance is database-driven at two levels:

    LEVEL 1 — MIGraphX compile-time selection:
        MIGraphX's compiler has shape-based heuristics for choosing
        between rocBLAS, MIOpen, and composable_kernel for each op.
        These heuristics are compiled into the MIGraphX binary and
        updated with each release.

    LEVEL 2 — MIOpen algorithm selection (the "Find" database):
        For each unique (shape, config) combination, MIOpen benchmarks
        all available algorithm implementations and records the winner.
        This is the fine-grained tuning that adapts to each exact model shape.
        Location: ~/.config/miopen/2.0/

    Both databases are PERSISTENT: once a shape is tuned, subsequent
    compilations of models with the same shapes skip benchmarking.

### Exhaustive Tuning

    DEFAULT COMPILATION (no exhaustive_tune):
        MIGraphX uses heuristic algorithm selection.
        Compile time: fast (seconds for most models).
        Performance: good, within 10–20% of optimal.

    EXHAUSTIVE TUNE:
        migraphx.compile(prog, target,
            migraphx.compile_options(exhaustive_tune=True))

        What this does:
            For every Conv/MatMul in the model, benchmarks ALL candidate
            MIOpen and rocBLAS algorithm implementations.
            Selects the fastest algorithm for the exact (M, N, K) shape
            on the current GPU.
            Saves results to the MIOpen find database.

        Compile time with exhaustive_tune:
            ResNet-50:    2–5 minutes (first time)
            BERT-Base:    5–15 minutes (first time)
            LLaMA-3-8B:  20–60 minutes (first time)
            Subsequent compilations with same shapes: < 1 second (database hit)

        When to use exhaustive_tune:
            Whenever inference latency matters more than compile time.
            In production deployments: compile once (exhaustive), deploy fast.
            Avoid in CI/CD pipelines where compile time matters.

### Environment Variables for Performance

    MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0    ; disable a slow fallback conv algorithm
    MIOPEN_FIND_MODE=1                   ; cache find results (default)
    MIOPEN_FIND_ENFORCE=1                ; re-run benchmarks even if cached
    MIOPEN_LOG_LEVEL=4                   ; log level (0=off, 4=info, 5=trace)
    MIOPEN_USER_DB_PATH=/fast/ssd/db     ; move find database to faster storage
    HIP_VISIBLE_DEVICES=0,1,2,3          ; which GPUs MIGraphX sees
    ROCR_VISIBLE_DEVICES=0               ; same, different env var name
    AMD_SERIALIZE_KERNEL=1               ; serialise kernel launches (for debugging)
    GPU_MAX_HW_QUEUES=4                  ; number of hardware command queues

### FP16 vs FP32 for Production

    DEFAULT: MIGraphX runs the model in the precision it was exported in.
    For FP32 ONNX models on GPU: inference uses FP32 (slower, less memory).
    For FP16 ONNX models: inference uses FP16 natively.

    FORCING FP16 AT RUNTIME:
        ORT MIGraphX EP:
            provider_options = {"fp16_enable": True}
        MIGraphX directly:
            Currently: export FP16 ONNX model before import.
            MIGraphX 2.10+: fp16_enable option in compile_options.

    FP16 ACCURACY:
        For most vision and NLP models: FP16 causes < 0.1% accuracy drop.
        For sensitive regression models or models with large activation ranges:
            test carefully before deploying FP16.
        Use half the memory of FP32:
            ResNet-50: 98 MB (FP32) → 49 MB (FP16)
            BERT-Base: 418 MB (FP32) → 209 MB (FP16)
            LLaMA-3-8B: 16 GB (FP32) → 8 GB (FP16)

### Benchmarking Best Practices

    1. WARM UP before measuring:
        ; First run: MIOpen find database may be cold (triggers benchmarking).
        ; Second run: fast (database loaded).
        ; Measure from the 3rd–5th run onwards.

    2. HIP_VISIBLE_DEVICES for isolation:
        ; Pin the process to one GPU to avoid interference from other workloads.

    3. Use migraphx-driver for baseline:
        migraphx-driver perf --onnx model.onnx \
                             --input-dim @data 1 3 224 224 \
                             --iterations 1000
        ; Reports: mean, min, max latency across 1000 iterations.

    4. rocm-smi for GPU utilisation:
        rocm-smi --showuse   ; GPU compute utilisation
        rocm-smi --showmemuse  ; GPU memory usage
        watch -n1 'rocm-smi --showuse --showmemuse'

    5. hipprof for GPU kernel profiling:
        hipprof --stats migraphx-driver run --onnx model.onnx ...
        ; Reports per-kernel timing, memory transfer, occupancy.


##### PART 10 — DOCKER, ROCM SETUP, AND PRODUCTION DEPLOYMENT

### ROCm Installation

    MIGraphX requires a working ROCm installation. ROCm is Linux-only
    (Ubuntu 20.04/22.04 and RHEL/CentOS are best supported).

    QUICK INSTALL (Ubuntu 22.04, ROCm 6.x):
        # Install ROCm:
        sudo apt-get install -y wget
        wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | \
            sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
            https://repo.radeon.com/rocm/apt/6.1 jammy main" | \
            sudo tee /etc/apt/sources.list.d/rocm.list
        sudo apt-get update
        sudo apt-get install -y rocm-hip-sdk miopen-hip migraphx

        # Add user to render/video groups:
        sudo usermod -aG render,video $USER

        # Verify GPU is visible:
        rocm-smi

    INSTALL MIGRAPHX PYTHON:
        pip install migraphx   ; bindings only, requires ROCm installed
        ; OR install the full ROCm Python stack:
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

### Docker for Development

    The easiest way to use MIGraphX without a full ROCm install:

        # Pull AMD's official ROCm + MIGraphX Docker image:
        docker pull rocm/migraphx:latest

        # Run interactively with GPU access:
        docker run -it \
            --network=host \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            --group-add render \
            -v $(pwd):/workspace \
            -w /workspace \
            rocm/migraphx:latest \
            /bin/bash

        # Inside the container:
        python3 -c "import migraphx; print(migraphx.__version__)"
        migraphx-driver --version

    Available Docker tags:
        rocm/migraphx:latest                   latest ROCm + MIGraphX
        rocm/migraphx:rocm6.1.1-ubuntu22.04    pinned ROCm version
        rocm/pytorch:latest                     PyTorch + ROCm (no MIGraphX)
        rocm/onnxruntime:latest                ORT + ROCm EP

### Production Deployment Patterns

    PATTERN 1 — Pre-compiled .mxr bundle:
        DEVELOPER MACHINE (compile once):
            prog = migraphx.parse_onnx("model.onnx", ...)
            migraphx.compile(prog, target,
                migraphx.compile_options(exhaustive_tune=True))
            prog.save("model.mxr")

        PRODUCTION SERVER (load fast):
            prog = migraphx.load("model.mxr")
            ; Startup: milliseconds
            ; No recompilation, no benchmarking

        The .mxr must be compiled on the same GPU architecture as the
        production server. Use Docker image pinned to matching ROCm version.

    PATTERN 2 — Triton Inference Server with MIGraphX backend:
        AMD provides a Triton backend plugin for MIGraphX:
        github.com/ROCm/triton-mlir (AMD's Triton fork)
        github.com/ROCmSoftwarePlatform/AMDMIGraphX (native Triton backend)

        tritonserver \
            --model-repository=/models \
            --backend-config=migraphx,exhaustive_tune=true \
            --log-verbose=1

        Model repository format:
            /models/
                resnet50/
                    1/
                        model.onnx   ; Triton compiles to .mxr on first load

    PATTERN 3 — ONNX Runtime Serving with MIGraphX EP:
        OVMS (OpenVINO Model Server) and Triton both support
        ONNX Runtime as a backend, which in turn uses the MIGraphX EP.
        This gives you: ORT's broad ONNX coverage + MIGraphX's AMD kernels
        + Triton's batching and serving infrastructure.

        docker run -d \
            --device=/dev/kfd --device=/dev/dri --group-add video \
            -p 8000:8000 -p 8001:8001 \
            nvcr.io/nvidia/tritonserver:23.10-rocm-py3 \
            tritonserver --model-repository=/models \
            --backend-config=onnxruntime,execution-provider=migraphx


##### PART 11 — MIGRAPHX IN THE CONNECTED COMPILER STACK

### MIGraphX and LLVM (Module 01)

    ROCm's HIP compiler (hipcc) is built on LLVM.
    When MIGraphX generates custom HIP kernels (for fused elementwise chains),
    those kernels are compiled by hipcc → LLVM → AMD GCN/CDNA ISA.
    composable_kernel templates are also compiled via LLVM's GPU backend.
    MIGraphX itself is compiled by Clang (LLVM-based).
    The entire ROCm software stack from user code to GPU instruction stream
    flows through LLVM.

### MIGraphX and MLIR (Module 02)

    AMD contributes to ONNX-MLIR — an MLIR-based ONNX compiler that partly
    overlaps with MIGraphX's functionality.
    MIGraphX itself uses MLIR passes internally for some optimisation phases.
    The future direction of MIGraphX is MLIR-first compilation:
        ONNX → MLIR ONNX dialect → MIGraphX MLIR dialect → GPU code.
    AMD's IREE port also targets AMD GPU, competing internally with MIGraphX
    for the AMD GPU inference space.

### MIGraphX and TVM (Module 08)

    TVM supports AMD ROCm via its OpenCL and ROCm backends.
    TVM's MetaSchedule finds optimal tiling for AMD GPU compute primitives.
    MIGraphX and TVM are direct competitors on AMD GPU inference:
        MIGraphX wins: standard ops (conv, matmul, attention) on CDNA hardware
                       where MIOpen/rocBLAS/CK are highly tuned.
        TVM wins: custom ops, exotic shapes, models requiring unusual tiling.
    Both can accept ONNX as input. Both produce ROCm kernel sequences.

### MIGraphX and ONNX (Module 10)

    ONNX is MIGraphX's primary import format and best-supported path.
    MIGraphX supports ONNX opsets 1–17 with broad operator coverage.
    Key limitation: ONNX opset 18+ ops (newer transformers, LLM-specific)
    may require fallback to ORT CPU EP or onnx-simplifier pre-processing.
    Best practice: use onnx-simplifier before MIGraphX to remove Cast and
    Identity nodes that inflate the graph without adding computation.

### MIGraphX and OpenVINO (Module 12)

    OpenVINO targets Intel hardware; MIGraphX targets AMD hardware.
    No hardware overlap — they are complementary, not competing.
    Both use ONNX as a primary import format.
    An organisation deploying on mixed Intel/AMD hardware uses both:
        Intel Xeon for CPU inference → OpenVINO
        AMD MI300X for GPU inference → MIGraphX
    ONNX is the portable representation that feeds both.

### The Complete MIGraphX Ecosystem

    ┌──────────────────────────────────────────────────────────────────────┐
    │  SOURCE FRAMEWORKS                                                    │
    │  PyTorch (torch_migraphx)   TensorFlow (ONNX export)                │
    │  JAX (ONNX export)          ONNX Hub / Hugging Face Optimum          │
    └──────────────────────────┬───────────────────────────────────────────┘
                               │
             migraphx.parse_onnx / parse_tf / torch_migraphx
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  MIGraphX Compilation Pipeline                                        │
    │  constant-prop → BN fusion → op fusion → layout → kernel-select     │
    │  → memory planning                                                    │
    └──────┬───────────────────┬──────────────────────────────────────────┘
           │                   │
           ▼                   ▼
    ┌─────────────┐   ┌────────────────────────────────────────────────────┐
    │  .mxr file  │   │  In-memory compiled program                         │
    │  (serialise │   │  prog.run({"data": gpu_tensor})                     │
    │   + redeploy│   └────────────────────────────────────────────────────┘
    │   fast)     │              ↓ dispatches to:
    └─────────────┘   ┌──────────────────────────────┐
                      │ MIOpen │ rocBLAS │ comp.kernel │
                      └──────────────────────────────┘
                                ↓
    ┌──────────────────────────────────────────────────────────────────────┐
    │  AMD GPU HARDWARE                                                     │
    │  MI300X (192 GB HBM3, 1.3 POPS INT8)                                │
    │  MI250 (128 GB HBM2e, CDNA2)                                        │
    │  MI100 (32 GB HBM2, CDNA1)                                          │
    └──────────────────────────────────────────────────────────────────────┘

    COMPARISON WITH ALTERNATIVES ON AMD GPU:
    ┌──────────────────────┬──────────────┬──────────────┬──────────────────┐
    │ Property             │  MIGraphX    │  ORT ROCm EP │  TVM (ROCm)      │
    ├──────────────────────┼──────────────┼──────────────┼──────────────────┤
    │ Primary format       │ ONNX         │ ONNX         │ ONNX/TFLite/etc  │
    │ Op coverage          │ ~120 ops     │ 250+ (ORT)   │ ~200+ ops        │
    │ Kernel backend       │ MIOpen/CK    │ MIGraphX EP  │ MetaSchedule     │
    │ Custom ops           │ Limited      │ ORT custom   │ Full via Python  │
    │ INT8 support         │ QDQ format   │ ORT quant    │ Experimental     │
    │ FP8 support          │ MI300X only  │ Limited      │ Limited          │
    │ .mxr serialisation   │ YES          │ No           │ .so export       │
    │ Open source          │ YES (Apache) │ YES (MIT)    │ YES (Apache)     │
    │ PyTorch backend      │ torch_mgx    │ ORT backend  │ torch.compile    │
    │ Triton integration   │ YES (native) │ Via ORT      │ YES              │
    └──────────────────────┴──────────────┴──────────────┴──────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · MIGraphX Core API — Import, Compile, and Run": {
        "description": (
            "End-to-end MIGraphX inference from ONNX model to GPU output. "
            "Show parse_onnx with explicit input shapes and dynamic shape ranges. "
            "Inspect the MIGraphX IR: walk instructions, print shapes, count op types. "
            "Compile for GPU with and without exhaustive_tune. "
            "Run inference: migraphx.to_gpu, prog.run, numpy conversion. "
            "Save to .mxr and reload to demonstrate serialisation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  MIGRAPHX CORE API — IMPORT, COMPILE, AND RUN")
print("=" * 65)
print()

try:
    import migraphx
    print(f"  MIGraphX {migraphx.__version__}")
    HAS_MX = True
    # Detect available target
    try:
        GPU_TARGET = migraphx.get_target("gpu")
        HAS_GPU    = True
        print("  Target: GPU (ROCm)")
    except Exception:
        GPU_TARGET = migraphx.get_target("cpu")
        HAS_GPU    = False
        print("  Target: CPU (no AMD GPU detected)")
except ImportError:
    HAS_MX  = False
    HAS_GPU = False
    print("  MIGraphX not installed.")
    print("  Install via ROCm: pip install migraphx")
    print("  Or use Docker:    docker run -it --device=/dev/kfd --device=/dev/dri")
    print("                     rocm/migraphx:latest")
print()

try:
    import torch, torch.nn as nn, io
    HAS_TORCH = True
    print(f"  PyTorch {torch.__version__}")
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Build a model and export to ONNX for MIGraphX
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Build and export an ONNX model for MIGraphX")
print("━" * 65)
print()

if HAS_TORCH:
    class ResBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(c)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(c)
        def forward(self, x):
            h = torch.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return torch.relu(h + x)   ; residual connection

    class TinyResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem  = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU())
            self.layer = nn.Sequential(ResBlock(32), ResBlock(32))
            self.head  = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(), nn.Linear(32*4*4, num_classes))
        def forward(self, x):
            return self.head(self.layer(self.stem(x)))

    model   = TinyResNet(10).eval()
    n_params = sum(p.numel() for p in model.parameters())
    example = torch.randn(1, 3, 64, 64)

    print(f"  Model: TinyResNet  params={n_params:,}")

    # Export to ONNX
    tmpdir   = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "tiny_resnet.onnx")
    torch.onnx.export(
        model, (example,), onnx_path,
        opset_version=17,
        input_names=["data"],
        output_names=["logits"],
        dynamic_axes={"data": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    onnx_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX file: {onnx_path}  ({onnx_size_kb:.1f} KB)")
    print()
else:
    # Fallback: use a pre-existing ONNX model if available
    onnx_path = None
    print("  PyTorch not available; using ONNX model if present.")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Parse and inspect the MIGraphX IR
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Parse ONNX and inspect MIGraphX IR")
print("━" * 65)
print()

if HAS_MX and onnx_path and os.path.exists(onnx_path):
    # Parse with explicit static shape
    prog = migraphx.parse_onnx(
        onnx_path,
        map_input_dims={"data": [1, 3, 64, 64]})

    main_mod = prog.get_main_module()

    # Count instructions and gather op type statistics
    from collections import Counter
    all_ins   = list(main_mod)
    op_counts = Counter(ins.name() for ins in all_ins)

    print(f"  MIGraphX IR after ONNX import:")
    print(f"    Total instructions: {len(all_ins)}")
    print()
    print(f"  Operation type counts:")
    for op_name, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        if op_name.startswith("@"): continue   ; skip params/literals/return
        print(f"    {op_name:<40s} × {cnt}")
    print()

    # Show first N instructions with shapes
    print(f"  First 12 instructions (name → output shape):")
    print(f"  {'#':>3}  {'Op name':30s}  {'Output shape'}")
    print("  " + "-" * 60)
    for i, ins in enumerate(all_ins[:12]):
        try:
            shp = ins.get_shape()
            shape_str = f"{shp.type()} {list(shp.lens())}"
        except Exception:
            shape_str = "(n/a)"
        print(f"  {i:>3}  {ins.name()[:29]:30s}  {shape_str}")
    if len(all_ins) > 12:
        print(f"  ... ({len(all_ins)-12} more instructions)")
    print()

    # Count convolution and matmul ops specifically
    conv_count   = op_counts.get("convolution", 0)
    matmul_count = op_counts.get("dot", 0)
    bn_count     = op_counts.get("batch_norm_inference", 0)
    print(f"  Key ops:  convolution={conv_count}  "
          f"dot={matmul_count}  batch_norm_inference={bn_count}")
    print()

else:
    MX_PARSE_REF = """
  MIGRAPHX PARSE AND INSPECT REFERENCE:
  ─────────────────────────────────────────────────────────────────
  import migraphx

  # Parse ONNX model with explicit input shapes:
  prog = migraphx.parse_onnx(
      "model.onnx",
      map_input_dims={"data": [1, 3, 224, 224]})

  # Access the main module:
  main_mod = prog.get_main_module()
  for ins in main_mod:
      shp = ins.get_shape()
      print(f"  {ins.name():<40s} {shp.type()} {list(shp.lens())}")

  # Count ops by type:
  from collections import Counter
  ops = Counter(ins.name() for ins in main_mod)
  for name, count in ops.most_common(10):
      print(f"  {name}: {count}")

  # Dynamic batch size (1–8):
  prog_dyn = migraphx.parse_onnx(
      "model.onnx",
      map_input_dims={"data": [[1, 8], [3], [224], [224]]})
"""
    print(MX_PARSE_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Compile and run
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Compile for GPU and run inference")
print("━" * 65)
print()

if HAS_MX and onnx_path and os.path.exists(onnx_path):
    print("  Compiling for target:", "GPU" if HAS_GPU else "CPU")
    t0 = time.perf_counter()
    migraphx.compile(prog, GPU_TARGET,
                      migraphx.compile_options(offload_copy=True))
    t_compile = (time.perf_counter() - t0) * 1000
    print(f"  Compile time: {t_compile:.0f} ms  (exhaustive_tune=False)")
    print()

    # Run inference
    x_np = np.random.randn(1, 3, 64, 64).astype(np.float32)

    # Warmup
    for _ in range(5):
        prog.run({"data": x_np})

    # Benchmark
    REPS = 200
    t0   = time.perf_counter()
    for _ in range(REPS):
        results = prog.run({"data": x_np})
    t_inf = (time.perf_counter() - t0) / REPS * 1000
    out   = np.array(results[0])

    print(f"  Input:   {x_np.shape}  dtype={x_np.dtype}")
    print(f"  Output:  {out.shape}  dtype={out.dtype}")
    print(f"  Latency: {t_inf:.3f} ms/inference  "
          f"({1000/t_inf:.0f} inf/sec)")
    print()

    # Verify against PyTorch reference
    if HAS_TORCH:
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x_np)).numpy()
        max_diff = float(np.max(np.abs(out - pt_out)))
        print(f"  Numerical validation: MIGraphX vs PyTorch max_err={max_diff:.2e} "
              f"{'✅' if max_diff < 1e-3 else '⚠️'}")
        print()

    # ── Save and reload .mxr ─────────────────────────────────────────────
    mxr_path = os.path.join(tmpdir, "tiny_resnet.mxr")
    prog.save(mxr_path)
    mxr_size_kb = os.path.getsize(mxr_path) / 1024

    print(f"  Saved .mxr: {mxr_path}  ({mxr_size_kb:.1f} KB)")

    t0 = time.perf_counter()
    prog_loaded = migraphx.load(mxr_path)
    t_load = (time.perf_counter() - t0) * 1000
    print(f"  Load from .mxr: {t_load:.1f} ms  (vs {t_compile:.0f} ms compile)")
    print(f"  Load speedup:   {t_compile/t_load:.1f}× faster than recompile")
    print()

    # Verify loaded model gives same result
    results2 = prog_loaded.run({"data": x_np})
    out2      = np.array(results2[0])
    roundtrip_err = float(np.max(np.abs(out - out2)))
    print(f"  .mxr roundtrip fidelity: max_err={roundtrip_err:.2e} "
          f"{'✅' if roundtrip_err == 0.0 else '⚠️'}")
    print()

else:
    COMPILE_REF = """
  COMPILATION AND INFERENCE REFERENCE:
  ─────────────────────────────────────────────────────────────────
  import migraphx, numpy as np

  prog   = migraphx.parse_onnx("model.onnx",
                                map_input_dims={"data": [1, 3, 224, 224]})
  target = migraphx.get_target("gpu")

  # Compile (auto-copies inputs to GPU when offload_copy=True):
  migraphx.compile(prog, target,
      migraphx.compile_options(offload_copy=True))

  # Run with numpy input directly:
  x_np   = np.random.randn(1, 3, 224, 224).astype(np.float32)
  result = prog.run({"data": x_np})
  out    = np.array(result[0])

  # Save compiled program:
  prog.save("model.mxr")

  # Load without recompiling:
  prog2  = migraphx.load("model.mxr")
  result2 = prog2.run({"data": x_np})
"""
    print(COMPILE_REF)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Compilation Passes — Fusion, BN Fold, and IR Inspection": {
        "description": (
            "Inspect MIGraphX's IR before and after compilation to see what each pass does. "
            "Show BatchNorm fold: BN instructions disappear after compile. "
            "Show operator fusion: elementwise chains collapse to single fused_main ops. "
            "Count kernel dispatch types: MIOpen, rocBLAS, fused HIP kernels. "
            "Demonstrate exhaustive_tune: compile options and timing comparison. "
            "Show the migraphx-driver CLI equivalent for each Python operation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  COMPILATION PASSES — FUSION, BN FOLD, AND IR INSPECTION")
print("=" * 65)
print()

try:
    import migraphx
    HAS_MX = True
    print(f"  MIGraphX {migraphx.__version__}")
    try:
        TARGET = migraphx.get_target("gpu")
        HAS_GPU = True
        print("  Target: GPU (ROCm)")
    except Exception:
        TARGET  = migraphx.get_target("cpu")
        HAS_GPU = False
        print("  Target: CPU fallback")
except ImportError:
    HAS_MX  = False
    HAS_GPU = False
    print("  MIGraphX not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Pass reference — what each pass does
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Compilation pass reference: what each pass does")
print("━" * 65)
print()

PASSES_GUIDE = """
  MIGRAPHX COMPILATION PASSES — ANNOTATED
  ════════════════════════════════════════════════════════════════

  The compilation pipeline runs these passes in order.
  Each pass modifies the instruction DAG and is followed by
  dead-code-elimination to clean up unreferenced instructions.

  PASS 1 — simplify_reshapes:
    Eliminates pairs of reshape/transpose that cancel out.
    Example: reshape(reshape(x, [N,C,HW]), [N,C,H,W]) → reshape(x, [N,C,H,W])
    Impact: reduces IR size, fewer intermediate buffers.

  PASS 2 — eliminate_dead_code:
    Removes any instruction whose output is not consumed.
    Runs after EVERY other pass to keep the IR clean.

  PASS 3 — propagate_constant:
    Evaluates instructions with all-constant inputs at compile time.
    Output: the instruction is replaced with a @literal constant.
    Example: shape computation subgraphs for fixed-size models.

  PASS 4 — simplify_algebra:
    Arithmetic identities: x + 0 → x, x * 1 → x, x / 1 → x, 0 * x → 0.
    Enables downstream passes to see simpler patterns.

  PASS 5 — fuse_batch_norm:
    Absorbs batch_norm_inference into preceding convolution.
    After: the conv instruction has updated weights/bias literals.
    The batch_norm_inference instruction is removed.
    Impact: one fewer kernel per BN layer at inference time.

    BEFORE fusion:
      conv(x, W, bias)           → conv_out
      batch_norm_inference(conv_out, scale, b, mean, var, eps) → bn_out

    AFTER fusion:
      @literal W_new  (W * scale/sqrt(var+eps))
      @literal b_new  (bias-related scaled bias)
      conv(x, W_new, b_new)      → bn_out
      ; batch_norm_inference instruction DELETED

  PASS 6 — fuse_pointwise:
    Groups contiguous elementwise instructions into fused kernels.
    The fused group becomes a single "pointwise" instruction.
    A custom HIP kernel is generated for the fused group.

    BEFORE fusion:
      add(x, bias) → h
      relu(h)      → a
      mul(a, gate) → out

    AFTER fusion:
      pointwise_fused_add_relu_mul(x, bias, gate) → out
      ; ONE HIP kernel launch, all intermediates in registers

  PASS 7 — fuse_reduce:
    Inline pointwise producers feeding into a reduction.
    Example: reduce_sum(exp(x)) → one kernel (no intermediate exp buffer).

  PASS 8 — fuse_ops (general):
    Fuses conv + relu, matmul + bias, attention patterns.
    Matches pre-registered fusion patterns against the graph.

  PASS 9 — select_target:
    Maps each instruction to a specific kernel implementation:
      convolution       → MIOpen forward convolution
      dot               → rocBLAS GEMM or composable_kernel
      batch_norm_inference → MIOpen (if not already fused into conv)
      pointwise_fused_* → HIP custom kernel (JIT compiled)

  PASS 10 — schedule:
    Determines execution order for maximum GPU occupancy.
    Avoids unnecessary synchronisation between independent instructions.

  PASSES RUN TOGETHER (migraphx.compile does all automatically):
    migraphx.compile(prog, target)
    ; You do not call passes individually in normal use.

  TO INSPECT BETWEEN PASSES (C++ only, Python not exposed):
    In C++: access pass results via migraphx::pass_manager inspection.
    In Python: compare IR before and after compile() to infer pass effects.
"""
print(PASSES_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Observe BN fold and fusion in practice
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Observe BN fold and op fusion before vs after compile")
print("━" * 65)
print()

if HAS_MX and HAS_TORCH:
    # Build a model with BatchNorm and elementwise chain
    class FusionDemo(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
            self.bn   = nn.BatchNorm2d(32)
            self.fc   = nn.Linear(32*4*4, 8)
        def forward(self, x):
            h = torch.relu(self.bn(self.conv(x)))       ; conv+BN+relu chain
            h = torch.relu(h * 2.0 + h * 0.5)          ; elementwise chain
            return self.fc(h.flatten(1))

    model_fused = FusionDemo().eval()
    dummy_fused = torch.randn(1, 16, 8, 8)

    tmpdir2 = tempfile.mkdtemp()
    path2   = os.path.join(tmpdir2, "fusion_demo.onnx")
    torch.onnx.export(model_fused, (dummy_fused,), path2,
                       opset_version=17,
                       input_names=["x"], output_names=["out"],
                       dynamic_axes={"x":{0:"b"},"out":{0:"b"}})

    prog2 = migraphx.parse_onnx(path2,
                                  map_input_dims={"x": [1, 16, 8, 8]})

    # Snapshot IR BEFORE compilation
    from collections import Counter
    mod_before = prog2.get_main_module()
    ins_before = list(mod_before)
    ops_before = Counter(i.name() for i in ins_before)

    print(f"  BEFORE compilation ({len(ins_before)} instructions):")
    for op, cnt in sorted(ops_before.items(), key=lambda x: -x[1]):
        if op.startswith("@"): continue
        print(f"    {op:<40s} × {cnt}")
    print()

    # Compile
    migraphx.compile(prog2, TARGET,
                      migraphx.compile_options(offload_copy=True))

    # Snapshot IR AFTER compilation
    mod_after = prog2.get_main_module()
    ins_after = list(mod_after)
    ops_after = Counter(i.name() for i in ins_after)

    print(f"  AFTER compilation ({len(ins_after)} instructions):")
    for op, cnt in sorted(ops_after.items(), key=lambda x: -x[1]):
        if op.startswith("@"): continue
        print(f"    {op:<40s} × {cnt}")
    print()

    # Analysis
    bn_before  = ops_before.get("batch_norm_inference", 0)
    bn_after   = ops_after.get("batch_norm_inference", 0)
    fused_after = sum(v for k, v in ops_after.items() if "pointwise" in k or "fused" in k)

    print(f"  Pass results:")
    print(f"    batch_norm_inference:  {bn_before} → {bn_after}  "
          f"({'✅ folded into conv' if bn_after == 0 else 'not folded'})")
    print(f"    Total instructions:    {len(ins_before)} → {len(ins_after)}  "
          f"({len(ins_before)-len(ins_after)} eliminated by passes)")
    print(f"    Fused elementwise ops: {fused_after} fused kernel(s)")
    print()

else:
    FUSION_REF = """
  OBSERVING PASSES IN PRACTICE — EXPECTED RESULTS:
  ─────────────────────────────────────────────────────────────────
  Model: Conv2d → BatchNorm2d → ReLU → elementwise chain

  BEFORE compile (after parse_onnx):
    convolution               × 1
    batch_norm_inference      × 1   ; present as separate instruction
    relu                      × 1
    mul                       × 2   ; elementwise chain not yet fused
    add                       × 1

  AFTER compile (migraphx.compile):
    convolution               × 1   ; same conv, but with folded BN weights
    (batch_norm_inference     × 0)  ; GONE — folded into conv weights/bias
    pointwise_fused_*         × 1   ; relu+mul+add FUSED into one kernel

  KEY OBSERVATIONS:
    batch_norm_inference disappears: BN absorbed into conv via fuse_batch_norm.
    relu/mul/add disappear: replaced by pointwise_fused_* via fuse_pointwise.
    Total instruction count DROPS significantly (typically 40–60% reduction).
    Each "pointwise_fused_*" is one custom HIP kernel launch.
"""
    print(FUSION_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Exhaustive tune reference and CLI guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Exhaustive tune and migraphx-driver CLI reference")
print("━" * 65)
print()

TUNE_AND_CLI = """
  EXHAUSTIVE TUNING — WHEN AND HOW
  ════════════════════════════════════════════════════════════════

  WHY TUNE MATTERS:
    For each unique (M, N, K) GEMM shape, MIOpen and rocBLAS have
    50–200 candidate kernel implementations differing in:
      - Tile sizes (M_TILE, N_TILE, K_TILE)
      - Vectorisation width (1, 2, 4, or 8 elements per thread)
      - Pipeline depth (prefetch stages)
      - Accumulator precision (FP32 vs FP16 accumulate)
    The BEST algorithm for a 1024×1024 GEMM may be 30% faster than
    the SECOND BEST. Without tuning, MIGraphX uses heuristics that
    are usually within 15% of optimal.

  ENABLING EXHAUSTIVE TUNE:
    import migraphx

    # Option 1: MIGraphX Python API
    migraphx.compile(prog, target,
        migraphx.compile_options(exhaustive_tune=True))

    # Option 2: ORT MIGraphX EP
    providers = [("MIGraphXExecutionProvider",
                  {"exhaustive_tune": True})]
    sess = ort.InferenceSession("model.onnx", providers=providers)

    # Option 3: migraphx-driver CLI
    migraphx-driver perf --onnx model.onnx \\
                         --input-dim @data 1 3 224 224 \\
                         --exhaustive-tune

  TUNING TIME (first compilation with exhaustive_tune=True):
    ResNet-50 (25 unique GEMM/conv shapes):  3–8 minutes
    BERT-Base (8 unique shapes):             5–12 minutes
    LLaMA-3-8B (many shapes):               30–90 minutes
    Subsequent compilations (same GPU):      < 5 seconds (DB hit)

  MANAGING THE TUNE DATABASE:
    Database location:
      ~/.config/miopen/    (default, user-specific)
    Override:
      export MIOPEN_USER_DB_PATH=/shared/miopen_db
    Sharing tuned databases (team workflow):
      Compile once on the production GPU with exhaustive_tune.
      Copy ~/.config/miopen/ to a shared location.
      Set MIOPEN_USER_DB_PATH to the shared location on all servers.
      No retuning needed; all servers get the optimal kernels.

  MIGRAPHX-DRIVER CLI REFERENCE:
  ─────────────────────────────────────────────────────────────────
  # Run a model and print output stats:
  migraphx-driver run \\
      --onnx resnet50.onnx \\
      --input-dim @data 1 3 224 224

  # Benchmark (mean latency across N iterations):
  migraphx-driver perf \\
      --onnx resnet50.onnx \\
      --input-dim @data 1 3 224 224 \\
      --iterations 100

  # Print compiled IR:
  migraphx-driver print \\
      --onnx resnet50.onnx \\
      --input-dim @data 1 3 224 224

  # Verify numerics vs CPU reference:
  migraphx-driver verify \\
      --onnx resnet50.onnx \\
      --input-dim @data 1 3 224 224 \\
      --tolerance 0.001

  # Compile and save .mxr:
  migraphx-driver compile \\
      --onnx resnet50.onnx \\
      --input-dim @data 1 3 224 224 \\
      --output resnet50_mi300x.mxr

  # Perf test pre-compiled .mxr:
  migraphx-driver perf \\
      --migraphx resnet50_mi300x.mxr \\
      --iterations 500

  # With exhaustive tuning (slow first run, fast after DB is warm):
  migraphx-driver perf \\
      --onnx bert_base.onnx \\
      --input-dim @input_ids 1 128 \\
      --exhaustive-tune \\
      --iterations 100
"""
print(TUNE_AND_CLI)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · INT8 Quantisation and the QDQ Pipeline": {
        "description": (
            "End-to-end INT8 quantisation workflow for MIGraphX. "
            "Quantise a model using ONNX Runtime's static quantisation tool. "
            "Load the QDQ ONNX model into MIGraphX and verify QDQ pattern detection. "
            "Compare FP32 vs INT8 model size, latency, and numerical accuracy. "
            "Show the AMD Quark API for production quantisation. "
            "Explain per-channel vs per-tensor granularity on AMD hardware."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  INT8 QUANTISATION AND THE QDQ PIPELINE")
print("=" * 65)
print()

try:
    import migraphx
    HAS_MX = True
    print(f"  MIGraphX {migraphx.__version__}")
    try:
        TARGET  = migraphx.get_target("gpu")
        HAS_GPU = True
        print("  Target: GPU (ROCm)")
    except Exception:
        TARGET  = migraphx.get_target("cpu")
        HAS_GPU = False
        print("  Target: CPU fallback")
except ImportError:
    HAS_MX = HAS_GPU = False
    print("  MIGraphX not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_static, quantize_dynamic,
        CalibrationDataReader, QuantFormat, QuantType
    )
    HAS_ORT = True
    print(f"  ONNX Runtime {ort.__version__} (for quantisation)")
except ImportError:
    HAS_ORT = False
    print("  ONNX Runtime not installed: pip install onnxruntime")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: INT8 quantisation theory and QDQ format
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — INT8 quantisation: QDQ format and MIGraphX's handling")
print("━" * 65)
print()

QDQ_GUIDE = """
  THE QDQ (QUANTIZE-DEQUANTIZE) FORMAT — HOW MIGRAPHX USES IT
  ════════════════════════════════════════════════════════════════

  MIGraphX reads INT8 via the ONNX QDQ format:
    float_input → QuantizeLinear → int8_tensor
    int8_tensor → [ConvolutionOrMatMul] → int32_output
    int32_output → DequantizeLinear → float_output

  WHAT QUANTIZELINEAR DOES:
    q = clamp(round(x / scale) + zero_point, -128, 127)
    scale and zero_point are stored as initializer tensors in the ONNX model.

  WHAT DEQUANTIZELINEAR DOES:
    x = (q - zero_point) * scale

  HOW MIGRAPHX HANDLES QDQ PATTERNS:
  ─────────────────────────────────────────────────────────────────
  When migraphx.compile() detects:
    QuantizeLinear → Convolution → DequantizeLinear

  It applies the fuse_qdq pass, which:
    1. Removes the QuantizeLinear and DequantizeLinear ops.
    2. Marks the Convolution as INT8 mode.
    3. Sets the scale and zero_point from the removed ops as kernel params.
    4. The convolution dispatches to rocBLAS INT8 GEMM or MIOpen INT8 conv.
    5. The INT32 accumulator output is rescaled to FP16/FP32 by the kernel.

  RESULT: a single INT8 kernel call with no separate Q/DQ overhead.

  QUANTISATION GRANULARITY:
  ─────────────────────────────────────────────────────────────────
  PER-TENSOR quantisation:
    One scale and one zero_point per activation tensor.
    scale = (max_val - min_val) / 255
    Fast, simple, but accuracy limited by outliers.
    QuantizeLinear has scalar (shape []) scale and zero_point.

  PER-CHANNEL quantisation:
    One scale and one zero_point per output channel of a weight tensor.
    scale_c = (max_channel_c - min_channel_c) / 255  for each c
    Better accuracy: each channel uses its own dynamic range.
    QuantizeLinear has scale and zero_point of shape [C] (one per channel).
    MIGraphX detects per-channel QDQ and dispatches to per-channel INT8 kernels.

  PER-CHANNEL ACCURACY ADVANTAGE (MobileNetV2, ImageNet):
    FP32 baseline:           71.9% top-1
    INT8 per-tensor PTQ:     71.0% top-1  (-0.9%)
    INT8 per-channel PTQ:    71.5% top-1  (-0.4%)
    INT8 QAT per-channel:    71.7% top-1  (-0.2%)

  CALIBRATION DATA REQUIREMENTS:
    Static quantisation requires a calibration dataset:
    - 100–1000 samples from the SAME distribution as inference.
    - Too few: poor range estimates → accuracy drop.
    - Too many: diminishing returns.
    MIOpen / rocBLAS INT8 requires SIGNED INT8 (range [-128, 127]).
    Ensure zero_point = 0 for symmetric quantisation (MIOpen requirement).
"""
print(QDQ_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Quantise a model with ORT and load into MIGraphX
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Quantise with ONNX Runtime, then compile with MIGraphX")
print("━" * 65)
print()

if HAS_TORCH and HAS_ORT:
    import onnx
    from onnx import numpy_helper

    # Build and export a model to ONNX
    class QuantNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64*4*4, 10))
        def forward(self, x): return self.net(x)

    model_q  = QuantNet().eval()
    dummy_q  = torch.randn(1, 3, 32, 32)
    tmpdir_q = tempfile.mkdtemp()
    fp32_path = os.path.join(tmpdir_q, "model_fp32.onnx")

    torch.onnx.export(model_q, (dummy_q,), fp32_path,
                       opset_version=17,
                       input_names=["input"], output_names=["output"],
                       dynamic_axes={"input":{0:"b"}, "output":{0:"b"}})
    print(f"  FP32 ONNX: {os.path.getsize(fp32_path)/1024:.1f} KB")

    # ── Dynamic range quantisation (no calibration needed) ────────────────
    int8_dynamic_path = os.path.join(tmpdir_q, "model_int8_dynamic.onnx")
    try:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_dynamic_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        )
        dyn_size = os.path.getsize(int8_dynamic_path) / 1024
        print(f"  INT8 dynamic ONNX:  {dyn_size:.1f} KB "
              f"(weights quantised, activations float)")
    except Exception as e:
        print(f"  Dynamic quantisation: {e}")
        int8_dynamic_path = None

    # ── Static QDQ quantisation (with calibration data) ──────────────────
    int8_static_path = os.path.join(tmpdir_q, "model_int8_qdq.onnx")
    N_CALIB = 100
    calib_data = [np.random.randn(1, 3, 32, 32).astype(np.float32)
                  for _ in range(N_CALIB)]

    class SimpleCalib(CalibrationDataReader):
        def __init__(self, data):
            self.data = data
            self.idx  = 0
        def get_next(self):
            if self.idx >= len(self.data): return None
            item = {"input": self.data[self.idx]}
            self.idx += 1
            return item

    try:
        quantize_static(
            model_input=fp32_path,
            model_output=int8_static_path,
            calibration_data_reader=SimpleCalib(calib_data),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,           ; per-channel for better accuracy
            reduce_range=False,
        )
        static_size = os.path.getsize(int8_static_path) / 1024
        fp32_size   = os.path.getsize(fp32_path) / 1024
        print(f"  INT8 static QDQ:    {static_size:.1f} KB  "
              f"({fp32_size/static_size:.2f}× compression vs FP32)")
    except Exception as e:
        print(f"  Static quantisation: {e}")
        int8_static_path = None
    print()

    # ── Load QDQ model into MIGraphX ─────────────────────────────────────
    if HAS_MX and int8_static_path and os.path.exists(int8_static_path):
        from collections import Counter

        # FP32 model stats
        prog_fp32 = migraphx.parse_onnx(fp32_path,
                                          map_input_dims={"input": [1, 3, 32, 32]})
        mod_fp32  = prog_fp32.get_main_module()
        ops_fp32  = Counter(i.name() for i in mod_fp32)

        # QDQ model stats
        prog_qdq  = migraphx.parse_onnx(int8_static_path,
                                          map_input_dims={"input": [1, 3, 32, 32]})
        mod_qdq   = prog_qdq.get_main_module()
        ops_qdq   = Counter(i.name() for i in mod_qdq)

        print(f"  IR comparison before compilation:")
        print(f"  {'Op':30s}  {'FP32':>8s}  {'QDQ INT8':>10s}")
        print("  " + "-" * 52)
        all_ops = sorted(set(ops_fp32) | set(ops_qdq))
        for op in all_ops:
            if op.startswith("@"): continue
            f  = ops_fp32.get(op, 0)
            q  = ops_qdq.get(op, 0)
            if f == 0 and q == 0: continue
            marker = "  ← quantised" if q > f else ""
            print(f"  {op[:29]:30s}  {f:>8d}  {q:>10d}{marker}")
        print()

        # Compile QDQ model
        migraphx.compile(prog_qdq, TARGET,
                          migraphx.compile_options(offload_copy=True))
        migraphx.compile(prog_fp32, TARGET,
                          migraphx.compile_options(offload_copy=True))

        # Benchmark FP32 vs INT8
        x_val = np.random.randn(1, 3, 32, 32).astype(np.float32)
        REPS  = 300

        for _ in range(20): prog_fp32.run({"input": x_val})
        t0 = time.perf_counter()
        for _ in range(REPS): prog_fp32.run({"input": x_val})
        t_fp32 = (time.perf_counter() - t0) / REPS * 1000

        for _ in range(20): prog_qdq.run({"input": x_val})
        t0 = time.perf_counter()
        for _ in range(REPS): prog_qdq.run({"input": x_val})
        t_int8 = (time.perf_counter() - t0) / REPS * 1000

        # Accuracy comparison
        fp32_out = np.array(prog_fp32.run({"input": x_val})[0])
        int8_out = np.array(prog_qdq.run({"input": x_val})[0])
        max_diff = float(np.max(np.abs(fp32_out - int8_out)))
        argmax_same = np.argmax(fp32_out) == np.argmax(int8_out)

        print(f"  {'Metric':30s}  {'FP32':>10s}  {'INT8 QDQ':>10s}")
        print("  " + "-" * 54)
        print(f"  {'Latency (ms)':30s}  {t_fp32:>10.3f}  {t_int8:>10.3f}")
        print(f"  {'Speedup':30s}  {'1.00×':>10s}  {t_fp32/t_int8:>10.2f}×")
        print(f"  {'Max L1 diff vs FP32':30s}  {'0':>10s}  {max_diff:>10.4f}")
        print(f"  {'Argmax agreement':30s}  {'✅':>10s}  "
              f"{'✅' if argmax_same else '⚠️':>10s}")
        print()

else:
    AMD_QUARK_REF = """
  AMD QUARK QUANTISATION (production tool):
  ─────────────────────────────────────────────────────────────────
  pip install amd-quark

  from quark.onnx import ModelQuantizer, QuantizationConfig
  from quark.onnx.quantization.config.config import (
      QuantType, CalibrationMethod
  )

  config = QuantizationConfig(
      quant_format="QDQ",
      activation_type=QuantType.QInt8,
      weight_type=QuantType.QInt8,
      per_channel=True,
      calibrate_method=CalibrationMethod.MinMax,   ; or Percentile, MSE
  )

  quantizer = ModelQuantizer(config)
  quantizer.quantize_model(
      model_input="model_fp32.onnx",
      model_output="model_int8_qdq.onnx",
      calibration_data_reader=MyCalibReader(),
  )

  # Then load into MIGraphX as usual:
  import migraphx
  prog   = migraphx.parse_onnx("model_int8_qdq.onnx",
                                map_input_dims={"input": [1, 3, 224, 224]})
  target = migraphx.get_target("gpu")
  migraphx.compile(prog, target)

  SMOOTHQUANT (for LLMs — handles activation outliers):
  ─────────────────────────────────────────────────────────────────
  ; SmoothQuant redistributes quantisation difficulty from
  ; activations (which have outliers) to weights (which don't).
  ; Result: much better INT8 accuracy on transformer models.

  config = QuantizationConfig(
      quant_format="QDQ",
      activation_type=QuantType.QInt8,
      weight_type=QuantType.QInt8,
      smooth_quant=True,         ; enable SmoothQuant
      smooth_quant_alpha=0.5,    ; balance between activation and weight smoothing
  )
"""
    print(AMD_QUARK_REF)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · ONNX Runtime MIGraphX Execution Provider": {
        "description": (
            "Use MIGraphX through the ONNX Runtime MIGraphX Execution Provider. "
            "Show EP configuration: device_id, exhaustive_tune, fp16_enable. "
            "Demonstrate the EP partitioning: which ops go to MIGraphX vs CPU. "
            "Benchmark ORT+MIGraphX EP vs pure MIGraphX vs ORT CPU. "
            "Show IO binding to eliminate host-device copy overhead. "
            "Demonstrate ORT profiling to find per-op bottlenecks."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  ONNX RUNTIME MIGRAPHX EXECUTION PROVIDER")
print("=" * 65)
print()

try:
    import onnxruntime as ort
    HAS_ORT = True
    print(f"  ONNX Runtime {ort.__version__}")
    available_providers = ort.get_available_providers()
    print(f"  Available providers: {available_providers}")
    HAS_MX_EP = "MIGraphXExecutionProvider" in available_providers
    HAS_ROCM  = "ROCMExecutionProvider"    in available_providers
    if HAS_MX_EP: print("  ✅ MIGraphX EP available")
    if HAS_ROCM:  print("  ✅ ROCm EP available")
except ImportError:
    HAS_ORT = HAS_MX_EP = HAS_ROCM = False
    print("  ONNX Runtime not installed: pip install onnxruntime")
print()

try:
    import migraphx
    HAS_MX = True
    try:
        TARGET  = migraphx.get_target("gpu")
        HAS_GPU = True
    except Exception:
        TARGET  = migraphx.get_target("cpu")
        HAS_GPU = False
except ImportError:
    HAS_MX = HAS_GPU = False
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: MIGraphX EP configuration and usage
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — MIGraphX EP: configuration and provider options")
print("━" * 65)
print()

EP_CONFIG = """
  ORT MIGRAPHX EXECUTION PROVIDER — COMPLETE REFERENCE
  ════════════════════════════════════════════════════════════════

  BASIC USAGE:
  ─────────────────────────────────────────────────────────────────
  import onnxruntime as ort

  # Simplest: just add MIGraphX EP to provider list
  sess = ort.InferenceSession(
      "model.onnx",
      providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"])

  inp    = {"data": np.random.randn(1, 3, 224, 224).astype(np.float32)}
  output = sess.run(None, inp)[0]

  FULL CONFIGURATION:
  ─────────────────────────────────────────────────────────────────
  provider_options = {
      "device_id":        0,        ; which GPU (0-indexed, for multi-GPU)
      "exhaustive_tune":  False,    ; enable MIOpen exhaustive algorithm search
                                    ; True: slower first compile, optimal perf
      "fp16_enable":      True,     ; cast FP32 ops to FP16 on GPU
                                    ; 2× less memory, ~same accuracy, faster
  }

  sess = ort.InferenceSession(
      "model.onnx",
      providers=[
          ("MIGraphXExecutionProvider", provider_options),
          "CPUExecutionProvider",     ; fallback for unsupported ops
      ],
      sess_options=ort.SessionOptions())

  EP PARTITIONING — HOW ORT ASSIGNS OPS:
  ─────────────────────────────────────────────────────────────────
  When ORT loads a model with MIGraphX EP:
    1. ORT walks every node in the ONNX graph.
    2. For each node: queries MIGraphX EP "can you run this?"
    3. MIGraphX EP responds: "yes" or "no" (based on supported ops).
    4. ORT groups consecutive "yes" nodes into MIGraphX EP subgraphs.
    5. "no" nodes → CPUExecutionProvider fallback.
    6. At runtime: MIGraphX compiles and executes each subgraph.
       CPU EP executes fallback nodes on CPU.

  BOUNDARY COST:
    Each MIGraphX-subgraph → CPU boundary requires:
      - Copying tensor from GPU to CPU (DeviceToHost memcpy).
      - Continuing on CPU.
    Each CPU → MIGraphX boundary:
      - Copying tensor from CPU to GPU (HostToDevice memcpy).
    These copies are ~0.1–0.5 ms each on fast PCIe 4.0.
    GOAL: minimise boundaries → maximise fraction of graph on MIGraphX EP.

  CHECKING WHICH PROVIDER EXECUTED A NODE:
  ─────────────────────────────────────────────────────────────────
  Use ORT profiling to see per-node provider assignment:
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    sess = ort.InferenceSession("model.onnx", sess_options=opts,
                                 providers=["MIGraphXExecutionProvider",
                                            "CPUExecutionProvider"])
    out          = sess.run(None, {"data": x})
    profile_path = sess.end_profiling()
    ; Open profile_path in chrome://tracing or parse JSON
    ; Nodes executed by MIGraphX EP have cat="Node" and ep="MIGraphX"

  IO BINDING (eliminates host-device copy overhead):
  ─────────────────────────────────────────────────────────────────
  For high-throughput serving where inputs are already on GPU:

    io_binding = sess.io_binding()

    ; Bind input from an existing GPU buffer:
    import ctypes
    io_binding.bind_input(
        name="data",
        device_type="hip",       ; "hip" for AMD GPU
        device_id=0,
        element_type=np.float32,
        shape=(1, 3, 224, 224),
        buffer_ptr=gpu_data_ptr, ; raw HIP device pointer
    )

    ; Bind output to a pre-allocated GPU buffer:
    io_binding.bind_output(
        name="output",
        device_type="hip",
        device_id=0,
        element_type=np.float32,
        shape=(1, 1000),
        buffer_ptr=gpu_out_ptr,
    )

    sess.run_with_iobinding(io_binding)
    ; Zero host-device copies for inputs or outputs!
    ; Critical for camera → GPU inference pipelines.
"""
print(EP_CONFIG)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Live benchmark — EP vs native MIGraphX vs CPU
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Benchmark: MIGraphX EP vs native API vs CPU")
print("━" * 65)
print()

if HAS_TORCH and HAS_ORT:
    class BenchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128*4*4, 10))
        def forward(self, x): return self.net(x)

    tmpdir_b = tempfile.mkdtemp()
    onnx_b   = os.path.join(tmpdir_b, "bench.onnx")
    model_b  = BenchNet().eval()
    dummy_b  = torch.randn(1, 3, 64, 64)
    torch.onnx.export(model_b, (dummy_b,), onnx_b,
                       opset_version=17,
                       input_names=["x"], output_names=["y"],
                       dynamic_axes={"x":{0:"b"},"y":{0:"b"}})
    x_b   = np.random.rand(1, 3, 64, 64).astype(np.float32)
    REPS  = 300
    results = {}

    def bench(fn, n_warmup=30, n_reps=REPS):
        for _ in range(n_warmup): fn()
        t0 = time.perf_counter()
        for _ in range(n_reps): fn()
        return (time.perf_counter() - t0) / n_reps * 1000

    # ORT CPU
    try:
        sess_cpu = ort.InferenceSession(onnx_b, providers=["CPUExecutionProvider"])
        t_cpu    = bench(lambda: sess_cpu.run(None, {"x": x_b}))
        results["ORT CPU"] = t_cpu
    except Exception as e:
        print(f"  ORT CPU: {e}")

    # ORT MIGraphX EP
    if HAS_MX_EP:
        try:
            opts_mx = ort.SessionOptions()
            sess_mx = ort.InferenceSession(
                onnx_b,
                sess_options=opts_mx,
                providers=[("MIGraphXExecutionProvider", {"fp16_enable": False}),
                           "CPUExecutionProvider"])
            t_mx = bench(lambda: sess_mx.run(None, {"x": x_b}))
            results["ORT MIGraphX EP"] = t_mx
        except Exception as e:
            print(f"  ORT MIGraphX EP: {e}")

    # Native MIGraphX Python
    if HAS_MX:
        try:
            prog_b = migraphx.parse_onnx(onnx_b, map_input_dims={"x": [1,3,64,64]})
            migraphx.compile(prog_b, TARGET,
                              migraphx.compile_options(offload_copy=True))
            t_native = bench(lambda: prog_b.run({"x": x_b}))
            results["Native MIGraphX"] = t_native
        except Exception as e:
            print(f"  Native MIGraphX: {e}")

    print(f"  Benchmark (batch=1, 64×64 CNN):")
    print(f"  {'Backend':30s}  {'Latency (ms)':>12s}  {'vs CPU':>8s}")
    print("  " + "-" * 54)
    cpu_t = results.get("ORT CPU", None)
    for name, t in results.items():
        ratio = f"{cpu_t/t:.2f}×" if cpu_t and t else "—"
        print(f"  {name:<30s}  {t:>12.3f}  {ratio:>8s}")
    print()

else:
    BENCHMARK_REF = """
  REPRESENTATIVE BENCHMARK RESULTS (MI300X, ROCm 6.1):
  ─────────────────────────────────────────────────────────────────
  Model: ResNet-50, batch=1, FP32

  Backend                         Latency   Throughput
  ─────────────────────────────────────────────────────────────────
  ORT CPU (32-core EPYC)          8.2 ms    122 inf/s
  ORT MIGraphX EP (MI300X, FP32)  1.9 ms    526 inf/s   (4.3× vs CPU)
  ORT MIGraphX EP (MI300X, FP16)  1.1 ms    909 inf/s   (7.5× vs CPU)
  Native MIGraphX (MI300X, FP32)  1.7 ms    588 inf/s   (4.8× vs CPU)
  Native MIGraphX (exhaustive)    1.4 ms    714 inf/s   (5.9× vs CPU)

  Model: BERT-Base, seq=128, batch=1, FP32

  Backend                         Latency
  ─────────────────────────────────────────────────────────────────
  ORT CPU                         38 ms
  ORT MIGraphX EP (FP32)          9.4 ms   (4.0× vs CPU)
  ORT MIGraphX EP (FP16)          5.2 ms   (7.3× vs CPU)
  Native MIGraphX (exhaustive)    4.8 ms   (7.9× vs CPU)

  Model: LLaMA-3-8B, 1 token generation, FP16, MI300X

  Backend                         Tokens/sec
  ─────────────────────────────────────────────────────────────────
  Native MIGraphX (FP16)          ~40 tok/s
  vLLM + ROCm                     ~55 tok/s  (vLLM uses PA + custom ops)
  TGI + ROCm                      ~48 tok/s
"""
    print(BENCHMARK_REF)
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Production Deployment, ROCm Setup, and Connected Stack": {
        "description": (
            "Complete MIGraphX production deployment patterns. "
            "Show Docker-based setup: which image tags, device flags, volume mounts. "
            "Demonstrate the .mxr compile-once-deploy-many pattern with timing. "
            "Show rocm-smi for GPU monitoring and hipprof for kernel profiling. "
            "Summarise MIGraphX's position in the full connected compiler stack. "
            "Quick-reference table: every key API, config flag, and CLI command."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import os
import tempfile

print("=" * 65)
print("  PRODUCTION DEPLOYMENT, ROCM SETUP, AND CONNECTED STACK")
print("=" * 65)
print()

try:
    import migraphx
    HAS_MX = True
    print(f"  MIGraphX {migraphx.__version__}")
    try:
        TARGET  = migraphx.get_target("gpu")
        HAS_GPU = True
        print("  GPU target: available")
    except Exception:
        TARGET  = migraphx.get_target("cpu")
        HAS_GPU = False
        print("  CPU target: (no AMD GPU)")
except ImportError:
    HAS_MX = HAS_GPU = False
    print("  MIGraphX not installed.")
print()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Docker setup and ROCm environment guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Docker setup: the fastest path to MIGraphX")
print("━" * 65)
print()

DOCKER_GUIDE = """
  ROCM + MIGRAPHX WITH DOCKER — COMPLETE SETUP GUIDE
  ════════════════════════════════════════════════════════════════

  PREREQUISITES (host system):
    AMD GPU with ROCm support (check: https://rocm.docs.amd.com/projects/install-on-linux)
    Supported GPUs: MI300X, MI250, MI100, RX 7900 XTX (limited), ...
    Linux kernel: 5.15+ (Ubuntu 22.04 LTS recommended)
    AMD GPU driver:
      sudo apt-get install amdgpu-dkms   ; installs kernel driver

  DOCKER IMAGE TAGS:
  ─────────────────────────────────────────────────────────────────
  rocm/migraphx:latest
      Latest ROCm + MIGraphX build (may not be stable)

  rocm/migraphx:rocm6.1.3-ubuntu22.04
      Pinned: ROCm 6.1.3, Ubuntu 22.04 (production-safe)

  rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_staging
      PyTorch + ROCm (no MIGraphX by default, but can install)
      pip install migraphx   ; add MIGraphX inside this container

  ROCm/onnxruntime:rocm6.1.3-ubuntu22.04
      ORT with ROCm + MIGraphX EP

  STARTING A CONTAINER (with AMD GPU access):
  ─────────────────────────────────────────────────────────────────
  docker run -it \\
      --network=host \\
      --device=/dev/kfd \\            ; AMD Kernel Fusion Driver
      --device=/dev/dri \\            ; Direct Rendering Infrastructure
      --group-add=video \\            ; group for GPU access
      --group-add=render \\           ; group for render node access
      --ipc=host \\                   ; shared memory (for PyTorch DataLoader)
      -v $(pwd):/workspace \\         ; mount current dir
      -w /workspace \\
      -e HIP_VISIBLE_DEVICES=0 \\    ; use GPU 0 only
      rocm/migraphx:rocm6.1.3-ubuntu22.04 \\
      /bin/bash

  VERIFY GPU IS ACCESSIBLE (inside container):
  ─────────────────────────────────────────────────────────────────
  # Check ROCm sees the GPU:
  rocm-smi

  # Verify MIGraphX:
  python3 -c "import migraphx; print(migraphx.__version__)"
  migraphx-driver --version

  # Quick end-to-end test:
  python3 -c "
  import migraphx, numpy as np
  ; Create a tiny test via the Python API
  target = migraphx.get_target('gpu')
  print('GPU target OK:', target)
  "

  MULTI-GPU SETUP:
  ─────────────────────────────────────────────────────────────────
  # Expose all GPUs:
  --device=/dev/kfd --device=/dev/dri
  -e HIP_VISIBLE_DEVICES=0,1,2,3

  # In Python: select GPU 1 specifically
  import migraphx
  ; (MIGraphX currently compiles for the default GPU device)
  ; For multi-GPU: use separate processes, each with HIP_VISIBLE_DEVICES=N

  ROCM-SMI MONITORING:
  ─────────────────────────────────────────────────────────────────
  rocm-smi                    ; overview: utilisation, memory, temp
  rocm-smi --showuse          ; GPU compute utilisation (%)
  rocm-smi --showmemuse       ; GPU VRAM usage (used/total MB)
  rocm-smi --showpower        ; GPU power draw (Watts)
  rocm-smi --showtemp         ; GPU temperature (°C)
  rocm-smi --showall          ; everything
  watch -n 0.5 rocm-smi      ; live refresh every 0.5 seconds

  ; During inference, you should see:
  ;   GPU use: 80–100%  (well-utilised)
  ;   Memory:  model_weights + activations + workspace
  ;   Power:   200–400W (MI300X under load)
  ;   Temp:    60–85°C  (normal operating range)
"""
print(DOCKER_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Production deployment: compile-once pattern
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Production pattern: compile once, deploy everywhere")
print("━" * 65)
print()

DEPLOY_PATTERNS = """
  PRODUCTION DEPLOYMENT PATTERN — THREE APPROACHES
  ════════════════════════════════════════════════════════════════

  APPROACH 1 — .MXR BUNDLE (recommended for stable models):
  ─────────────────────────────────────────────────────────────────
  COMPILE PHASE (CI/CD or build server, runs once):
    import migraphx

    prog = migraphx.parse_onnx("model.onnx",
                                map_input_dims={"data": [1, 3, 224, 224]})
    target = migraphx.get_target("gpu")
    migraphx.compile(prog, target,
        migraphx.compile_options(exhaustive_tune=True))   ; run once, slow
    prog.save("model_mi300x.mxr")
    ; model_mi300x.mxr: pre-compiled for MI300X, ready for production

  SERVING PHASE (production server, runs on every startup):
    import migraphx
    prog = migraphx.load("model_mi300x.mxr")             ; fast: milliseconds
    ; Server is ready for inference immediately

  CAVEATS:
    .mxr is GPU-architecture-specific (MI300X .mxr ≠ MI250 .mxr).
    .mxr is MIGraphX-version-specific (upgrade → recompile).
    .mxr does NOT contain the ROCm kernel source — it's a compiled binary.
    Solution: compile as part of Docker image build:
        FROM rocm/migraphx:rocm6.1.3-ubuntu22.04
        COPY model.onnx /app/
        RUN python3 -c "
            import migraphx
            prog = migraphx.parse_onnx('/app/model.onnx',
                map_input_dims={'data': [1,3,224,224]})
            migraphx.compile(prog, migraphx.get_target('gpu'),
                migraphx.compile_options(exhaustive_tune=True))
            prog.save('/app/model.mxr')
        "
        ; model.mxr is now in the Docker image, ready for fast startup

  APPROACH 2 — COMPILE AT STARTUP WITH CACHE:
  ─────────────────────────────────────────────────────────────────
  Compile every startup, but use MIOpen find database to skip tuning:
    export MIOPEN_USER_DB_PATH=/persistent/miopen_db   ; shared volume
    python3 server.py   ; compile_model() at startup
    ; First startup: MIOpen tunes and caches (slow)
    ; Subsequent startups: cache hit (fast)

    Advantage: no .mxr version management.
    Disadvantage: startup latency on first cold start (minutes with exhaustive_tune).

  APPROACH 3 — ONNX RUNTIME WITH MIGRAPHX EP:
  ─────────────────────────────────────────────────────────────────
  For serving with ONNX Runtime (Triton, OVMS, custom):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    sess = ort.InferenceSession(
        "model.onnx",
        sess_options=opts,
        providers=[
            ("MIGraphXExecutionProvider", {
                "device_id":      0,
                "exhaustive_tune": False,
                "fp16_enable":    True,
            }),
            "CPUExecutionProvider",
        ])

    Advantage: ORT's broad op coverage (250+ ops) + MIGraphX performance.
    Best for: complex models with ops MIGraphX doesn't support natively.

  TRITON INFERENCE SERVER WITH MIGRAPHX:
  ─────────────────────────────────────────────────────────────────
  # model_repository/resnet50/config.pbtxt:
  name: "resnet50"
  backend: "onnxruntime"
  max_batch_size: 8
  input [{ name: "data" dims: [3, 224, 224] data_type: TYPE_FP32 }]
  output [{ name: "logits" dims: [1000] data_type: TYPE_FP32 }]
  parameters { key: "execution_provider" value { string_value: "migraphx" }}
  parameters { key: "migraphx_fp16_enable" value { string_value: "1" }}
"""
print(DEPLOY_PATTERNS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Connected stack and quick reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — MIGraphX in the connected compiler stack")
print("━" * 65)
print()

STACK = """
  MIGRAPHX IN THE CONNECTED COMPILER STACK
  ════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE 01: LLVM                                                      │
  │  ROCm's hipcc (HIP compiler) is LLVM-based.                          │
  │  MIGraphX's fused HIP kernels are compiled by hipcc → LLVM → GCN.   │
  │  composable_kernel uses LLVM template metaprogramming.               │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 02: MLIR                                                      │
  │  MIGraphX uses MLIR for some internal lowering passes.               │
  │  AMD's ONNX-MLIR project is an MLIR-based ONNX compiler.            │
  │  Future: MIGraphX converging toward full MLIR compilation pipeline. │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 08: TVM                                                       │
  │  TVM supports AMD GPU via ROCm backend.                              │
  │  TVM and MIGraphX compete on AMD GPU inference.                     │
  │  TVM wins: custom ops, unusual shapes, MetaSchedule tile search.    │
  │  MIGraphX wins: standard shapes where MIOpen/CK are hand-tuned.    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 10: ONNX                                                      │
  │  ONNX is MIGraphX's primary and most complete import format.         │
  │  parse_onnx is the recommended entry point for all frameworks.      │
  │  Supported: ONNX opsets 1–17 (most ops), partial 18+.              │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 12: OpenVINO                                                  │
  │  Intel hardware → OpenVINO.  AMD hardware → MIGraphX.               │
  │  ONNX bridges both: one model, two vendor-specific runtimes.        │
  │  No overlap in target hardware.                                      │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 13: Core ML                                                   │
  │  Apple hardware → Core ML.  AMD hardware → MIGraphX.                │
  │  Zero overlap. ONNX bridges both.                                    │
  ├──────────────────────────────────────────────────────────────────────┤
  │  MODULE 14: MIGraphX (THIS MODULE)                                   │
  │  ONNX/TF/PyTorch → parse → compile (BN fold, fusion, layout)       │
  │  → MIOpen/rocBLAS/composable_kernel dispatch → AMD GPU              │
  │  .mxr serialisation for fast production deployment                  │
  │  INT8 via ONNX QDQ format, FP8 on MI300X                           │
  │  ORT MIGraphX EP for broad op coverage + AMD kernels               │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(STACK)

print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Task                        │ API / Command                       │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ Import ONNX model           │ migraphx.parse_onnx('m.onnx', ...) │")
print("  │ Import TF frozen graph      │ migraphx.parse_tf('m.pb', ...)     │")
print("  │ Set input shapes            │ map_input_dims={'data':[1,3,224,224]}│")
print("  │ Dynamic batch range         │ map_input_dims={'x':[[1,8],[3],..]} │")
print("  │ Compile for GPU             │ migraphx.compile(prog, get_target('gpu'))│")
print("  │ Enable auto-copy            │ compile_options(offload_copy=True)  │")
print("  │ Exhaustive tuning           │ compile_options(exhaustive_tune=True)│")
print("  │ Upload array to GPU         │ migraphx.to_gpu(numpy_array)        │")
print("  │ Run inference               │ prog.run({'data': gpu_tensor})      │")
print("  │ Async run                   │ prog.run({...}, run_async=True)     │")
print("  │ Download from GPU           │ np.array(gpu_tensor)                │")
print("  │ Save compiled program       │ prog.save('model.mxr')              │")
print("  │ Load compiled program       │ migraphx.load('model.mxr')          │")
print("  │ Inspect IR                  │ for i in prog.get_main_module(): ...│")
print("  │ Op type count               │ Counter(i.name() for i in module)  │")
print("  │ ORT MIGraphX EP             │ providers=['MIGraphXExecutionProvider']│")
print("  │ ORT EP with FP16            │ {'fp16_enable': True}               │")
print("  │ INT8 via QDQ                │ parse_onnx('model_qdq.onnx', ...)   │")
print("  │ Benchmark CLI               │ migraphx-driver perf --onnx m.onnx  │")
print("  │ Print IR CLI                │ migraphx-driver print --onnx m.onnx │")
print("  │ Verify numerics CLI         │ migraphx-driver verify --onnx m.onnx│")
print("  │ Compile + save CLI          │ migraphx-driver compile --output m.mxr│")
print("  │ Monitor GPU                 │ rocm-smi --showuse --showmemuse     │")
print("  │ Profile kernels             │ hipprof --stats migraphx-driver ... │")
print("  │ MIOpen find DB location     │ MIOPEN_USER_DB_PATH=/path/to/db     │")
print("  │ GPU visibility              │ HIP_VISIBLE_DEVICES=0,1             │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ GitHub (MIGraphX)           │ github.com/ROCm/AMDMIGraphX        │")
print("  │ ROCm docs                   │ rocm.docs.amd.com                   │")
print("  │ MIOpen docs                 │ rocm.docs.amd.com/projects/miopen   │")
print("  │ Docker image                │ rocm/migraphx:latest                │")
print("  │ AMD Quark (quantisation)    │ github.com/amd/quark                │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()

if HAS_MX and HAS_TORCH:
    print("  Runtime verification:")
    try:
        model_v = nn.Sequential(nn.Linear(8, 16), nn.ReLU(),
                                 nn.Linear(16, 4)).eval()
        dummy_v = torch.randn(1, 8)
        tmpdir_v = tempfile.mkdtemp()
        path_v   = os.path.join(tmpdir_v, "verify.onnx")
        torch.onnx.export(model_v, (dummy_v,), path_v,
                           opset_version=17,
                           input_names=["x"], output_names=["y"])
        with torch.no_grad():
            pt_out = model_v(dummy_v).numpy()

        prog_v = migraphx.parse_onnx(path_v,
                                       map_input_dims={"x": [1, 8]})
        migraphx.compile(prog_v, TARGET,
                          migraphx.compile_options(offload_copy=True))
        mx_out = np.array(prog_v.run({"x": dummy_v.numpy()})[0])

        err = float(np.max(np.abs(pt_out - mx_out)))
        print(f"    Linear(8→16→4) ReLU: PyTorch vs MIGraphX max_err={err:.2e} "
              f"{'✅' if err < 1e-3 else '⚠️'}")
        print(f"    Target: {'GPU (ROCm)' if HAS_GPU else 'CPU'}")
        print(f"    Available devices: GPU={HAS_GPU}")
    except Exception as e:
        print(f"    Verification: {e}")
elif HAS_MX:
    print("  Runtime verification:")
    print(f"    MIGraphX {migraphx.__version__} loaded")
    print(f"    Target: {'GPU (ROCm)' if HAS_GPU else 'CPU'}")
    try:
        t = migraphx.get_target("gpu" if HAS_GPU else "cpu")
        print(f"    Target name: {t.name()}")
    except Exception as e:
        print(f"    Target: {e}")
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