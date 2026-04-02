import textwrap
import re

TOPIC_NAME = "Order for Compilers & Runtimes"
DISPLAY_NAME = "00 · Order for Compilers & Runtimes"
ICON = "⚙️"
SUBTITLE = "Compilers, Runtimes & Hardware Backends — priorities and ordering"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### Compilers & Runtimes in AI/ML

A compiler takes the high-level operations your framework describes and transforms them into something 
a specific piece of hardware can execute efficiently. A runtime is the engine that actually carries out 
that execution. Together they form the invisible layer that determines whether your model runs in 
milliseconds or minutes — and whether it runs at all on a given device.

## The Core Problem

A PyTorch model is a Python program. A GPU understands CUDA kernels. An edge device understands 
ARM assembly. The gap between these representations is enormous, and bridging it naively — 
executing each operation one at a time in Python — leaves most of the hardware idle. 
Compilers exist to analyze the full computation graph, fuse operations, eliminate redundancy, 
schedule memory transfers, and generate tight hardware-specific code. Without them, modern ML 
at scale would be economically impossible.

## The Compilation Pipeline

Every ML compiler follows roughly the same pipeline, just at different levels of abstraction.
You start with a high-level representation — a computation graph captured from your framework.
That graph goes through a series of transformations: operator fusion merges adjacent ops to avoid 
memory round-trips, constant folding eliminates redundant computation, layout optimization 
reorders tensor dimensions for cache efficiency, and quantization reduces numerical precision 
to trade accuracy for speed. At the bottom, a code generation backend emits hardware-specific 
instructions — CUDA PTX for NVIDIA GPUs, ROCm HIP for AMD, Metal for Apple Silicon.

## Compiler Infrastructure

**LLVM** is the backbone of modern compiler engineering. Most ML compilers ultimately lower 
to LLVM IR before emitting machine code, inheriting decades of optimization passes for free. 
Understanding LLVM is understanding the foundation everything else builds on.

**MLIR** (Multi-Level Intermediate Representation) is Google's answer to the proliferation of 
incompatible compiler IRs. Rather than building one monolithic IR, MLIR is a framework for 
building IRs — it lets you define custom dialects at different abstraction levels and write 
passes that lower progressively from high-level ML ops down to hardware instructions. 
XLA, IREE, and TensorFlow all use MLIR internally.

**Enzyme** brings automatic differentiation into the compiler itself. Instead of your framework 
tracking gradients in Python, Enzyme differentiates the compiled LLVM IR directly — 
giving you gradients through arbitrary low-level code that frameworks can't trace.

## ML Compilers

**XLA / OpenXLA** is the compiler that powers JAX and is deeply integrated with TensorFlow and 
increasingly with PyTorch via `torch_xla`. It takes HLO (High-Level Operations) as input and 
emits highly optimized code for TPUs, GPUs, and CPUs. When JAX code runs fast, XLA is the reason.

**TVM (Apache)** is the most widely deployed open-source ML compiler. Its key innovation is 
separating algorithm description from schedule — how computation is mapped to hardware. 
AutoTVM and MetaSchedule automate the search for optimal schedules across diverse hardware targets, 
making TVM the go-to choice for deploying models on non-standard hardware.

**Triton** is OpenAI's GPU kernel language. Where CUDA requires you to think in terms of threads 
and warps, Triton lets you write GPU kernels at the level of tiles and blocks, with the compiler 
handling the low-level mapping. Flash Attention — the attention algorithm that makes training 
long-context transformers tractable — is implemented in Triton.

**TorchDynamo + TorchInductor** is PyTorch 2.x's compilation story. TorchDynamo captures 
Python-level computation graphs by tracing bytecode, preserving Python's dynamism while enabling 
compilation. TorchInductor then lowers those graphs to Triton kernels on GPU or C++ on CPU. 
Calling `torch.compile(model)` invokes this entire stack transparently.

**IREE** (Intermediate Representation Execution Environment) is Google's end-to-end ML compiler 
and runtime stack built on MLIR. It targets a wide range of hardware and is particularly strong 
for deployment scenarios where you need a single portable binary that runs efficiently across devices.

## Inference Runtimes — General Purpose

**ONNX Runtime** is Microsoft's cross-platform inference engine. Export your model to ONNX format 
and ONNX Runtime handles execution across CPU, GPU, and edge devices through pluggable 
execution providers (CUDA, TensorRT, OpenVINO, CoreML). It's the most commonly used runtime 
for production inference outside of NVIDIA-only stacks.

**TensorRT** is NVIDIA's inference optimizer and runtime. It takes a trained model, applies 
aggressive optimizations — layer fusion, precision calibration, kernel auto-tuning — 
and produces an engine file optimized for a specific GPU. For maximum throughput on NVIDIA hardware, 
nothing beats TensorRT, but the optimization step is hardware-specific and can be time-consuming.

**OpenVINO** is Intel's inference toolkit. It optimizes models for Intel CPUs, integrated GPUs, 
VPUs, and FPGAs. For deployments on Intel edge hardware or data center CPUs, OpenVINO consistently 
outperforms generic runtimes.

**TFLite** targets mobile and embedded devices. It quantizes models aggressively and minimizes 
binary size, making it the standard runtime for on-device ML on Android and microcontrollers.

**Core ML** is Apple's on-device inference framework. Models converted to Core ML format run 
on the Neural Engine in iPhones and Apple Silicon Macs, which provides dramatic power efficiency 
compared to GPU execution for supported architectures.

## Inference Runtimes — LLM Specific

LLMs introduced inference challenges that general-purpose runtimes weren't designed for: 
autoregressive decoding is memory-bandwidth-bound rather than compute-bound, KV caches grow 
dynamically, and batching requests with different sequence lengths is non-trivial.

**vLLM** solved the KV cache memory problem with PagedAttention — borrowing the virtual memory 
concept from operating systems to manage KV cache blocks non-contiguously, dramatically reducing 
fragmentation and enabling higher throughput batch serving.

**TensorRT-LLM** is NVIDIA's LLM-specific inference library. It brings TensorRT's optimization 
philosophy — custom kernels, quantization, in-flight batching — specifically to transformer 
architectures, and is the highest-throughput option for LLM serving on NVIDIA hardware.

**llama.cpp / GGML** made running LLMs on consumer hardware possible. By quantizing weights to 
4-bit integers and writing hand-optimized CPU inference code, Georgi Gerganov enabled 7B parameter 
models to run on a MacBook. GGUF is the successor format — more flexible, with better metadata 
support and improved quantization schemes.

**MLX** is Apple's ML framework designed specifically for Apple Silicon. Like JAX, it uses lazy 
evaluation and functional transforms, but its memory model is unified — CPU and GPU share the 
same memory pool, eliminating the copy overhead that makes other frameworks inefficient on Apple hardware.

**Ollama** is a runtime wrapper that makes running local LLMs as simple as `ollama run llama3`. 
It handles model downloading, quantization format detection, hardware detection, and serving, 
abstracting over llama.cpp and other backends. It's the entry point for most developers working 
with local LLMs.

## Serving & Deployment Runtimes

**Triton Inference Server** (NVIDIA, distinct from the Triton compiler) is a production model 
serving platform. It supports multiple frameworks simultaneously, handles dynamic batching, 
manages model ensembles, and exposes gRPC and HTTP endpoints. It's the infrastructure layer 
for large-scale multi-model serving.

**BentoML** focuses on the developer experience of packaging and deploying models. 
A BentoML service wraps your model with its dependencies, preprocessing logic, and API definition 
into a self-contained artifact that can run locally or deploy to any cloud.

**Ray Serve** brings Ray's distributed computing model to inference. It's the natural choice 
when your serving needs are heterogeneous — routing requests across multiple models, 
composing model pipelines, or scaling inference across a cluster you're already managing with Ray.

## Interchange Formats & IRs

**ONNX** is the lingua franca of trained models. The ONNX specification defines a standard set 
of ML operators and a protobuf serialization format. Any framework that exports to ONNX can 
interoperate with any runtime that imports it — decoupling the training ecosystem from 
the inference ecosystem.

**SafeTensors** is Hugging Face's replacement for pickle-based model weights. It's safe 
(no arbitrary code execution on load), fast (zero-copy memory mapping), and simple. 
The ecosystem has largely converged on SafeTensors for distributing model weights.

**GGUF** is llama.cpp's format for quantized models. It bundles weights, tokenizer, and model 
metadata into a single file, supports a wide range of quantization schemes, and is the standard 
format for sharing quantized LLMs via Hugging Face.

## Hardware Backends

**CUDA / PTX** is NVIDIA's compute platform. PTX is the portable assembly language that CUDA 
compiles to before being further optimized for a specific GPU microarchitecture. 
Understanding CUDA threading — blocks, warps, shared memory — is essential for writing 
high-performance ML kernels.

**ROCm / HIP** is AMD's GPU compute stack. HIP is syntactically close to CUDA and many CUDA 
programs can be ported automatically via hipify. The ecosystem is maturing rapidly as AMD's 
MI300 series has become a serious alternative to NVIDIA in data centers.

**Metal / MPS** is Apple's GPU compute framework. PyTorch's MPS backend routes tensor operations 
through Metal Performance Shaders, giving GPU acceleration on Apple Silicon without CUDA.

**WebGPU** brings GPU compute to the browser. It's a modern replacement for WebGL, designed 
specifically for compute workloads. Running inference directly in the browser — without a server 
— is increasingly viable through WebGPU, enabling privacy-preserving on-device ML at web scale.

## How the Layers Connect

The path from a PyTorch model to a deployed prediction travels through many of these layers. 
`torch.compile` captures the graph via TorchDynamo, lowers it through TorchInductor to Triton kernels, 
which compile to CUDA PTX for the GPU. For deployment, the model is exported to ONNX, 
optimized by TensorRT, and served via Triton Inference Server. For edge deployment, 
the same model might go through TFLite or Core ML instead, targeting completely different hardware.

Understanding this stack means understanding where your model's performance actually comes from — 
and where to look when it isn't fast enough.


    compilers_runtimes/
    ├── compiler_infrastructure/
    │   ├── llvm
    │   ├── mlir
    │   ├── circt
    │   └── enzyme
    ├── ml_compilers/
    │   ├── xla_openxla
    │   ├── tvm
    │   ├── triton
    │   ├── torchdynamo_inductor
    │   ├── glow
    │   ├── iree
    │   ├── halide
    │   └── nnfusion
    ├── inference_runtimes/
    │   ├── general_purpose/
    │   │   ├── onnx_runtime
    │   │   ├── tensorrt
    │   │   ├── tflite
    │   │   ├── openvino
    │   │   ├── coreml
    │   │   └── migraphx
    │   └── llm_specific/
    │       ├── vllm
    │       ├── tensorrt_llm
    │       ├── llama_cpp_ggml
    │       ├── gguf
    │       ├── exllamav2
    │       ├── ctranslate2
    │       ├── mlx
    │       ├── deepspeed_inference
    │       └── ollama
    ├── serving_deployment/
    │   ├── triton_inference_server
    │   ├── torchserve
    │   ├── tf_serving
    │   ├── bentoml
    │   ├── ray_serve
    │   └── onnx_runtime_server
    ├── interchange_formats/
    │   ├── onnx
    │   ├── hlo_stablehlo
    │   ├── tosa
    │   ├── spirv
    │   ├── safetensors
    │   └── gguf
    └── hardware_backends/
        ├── cuda_ptx
        ├── rocm_hip
        ├── oneapi_dpcpp
        ├── metal_mps
        ├── webgpu_wgsl
        └── opencl

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }


"""
compilers_runtimes
│
├── Compiler Infrastructure
│   ├── LLVM                        ← backbone for most ML compilers
│   ├── MLIR                        ← multi-level IR framework (Google)
│   ├── CIRCT                       ← HW-oriented extension of MLIR
│   └── Enzyme                      ← LLVM-based automatic differentiation
│
├── ML Compilers
│   ├── XLA / OpenXLA               ← Google (JAX, TF, PyTorch)
│   ├── StableHLO                   ← portable HLO op layer
│   ├── TVM (Apache)                ← cross-platform ML compiler
│   ├── Triton                      ← OpenAI GPU kernel compiler
│   ├── TorchDynamo                 ← PyTorch 2.x graph capture
│   ├── TorchInductor               ← PyTorch 2.x codegen backend
│   ├── Glow                        ← Meta's ML compiler
│   ├── IREE                        ← Google portable ML compiler+runtime
│   ├── Halide                      ← scheduling-based array compiler
│   └── NNFusion                    ← Microsoft deep learning compiler
│
├── Inference Runtimes & Engines
│   ├── General Purpose
│   │   ├── ONNX Runtime            ← Microsoft cross-platform runtime
│   │   ├── TensorRT                ← NVIDIA inference optimizer
│   │   ├── TFLite                  ← TensorFlow mobile/edge runtime
│   │   ├── OpenVINO                ← Intel inference toolkit
│   │   ├── Core ML                 ← Apple on-device runtime
│   │   ├── MIGraphX                ← AMD inference engine
│   │   ├── MNN                     ← Alibaba mobile inference
│   │   ├── NCNN                    ← Tencent mobile inference
│   │   └── Paddle Lite             ← Baidu edge inference
│   │
│   └── LLM-Specific
│       ├── vLLM                    ← PagedAttention serving engine
│       ├── TensorRT-LLM            ← NVIDIA LLM inference
│       ├── llama.cpp / GGML        ← CPU-first LLM runtime
│       ├── GGUF                    ← quantized model format+runtime
│       ├── ExLlamaV2               ← quantized GPU LLM runtime
│       ├── CTranslate2             ← fast Transformer inference
│       ├── MLX                     ← Apple Silicon ML framework
│       ├── DeepSpeed Inference     ← Microsoft LLM serving
│       └── Ollama                  ← local LLM runtime wrapper
│
├── Serving & Deployment Runtimes
│   ├── Triton Inference Server     ← NVIDIA model serving (≠ Triton compiler)
│   ├── TorchServe                  ← PyTorch model serving
│   ├── TF Serving                  ← TensorFlow model serving
│   ├── BentoML                     ← model packaging + serving
│   ├── Ray Serve                   ← distributed model serving
│   └── ONNX Runtime Server         ← ONNX serving endpoint
│
├── IRs & Interchange Formats
│   ├── ONNX                        ← cross-framework model format
│   ├── HLO / StableHLO             ← XLA's computation IR
│   ├── TOSA                        ← Tensor Operator Set Architecture
│   ├── SPIR-V                      ← GPU shader/compute IR
│   ├── SafeTensors                 ← safe model weight format
│   └── GGUF                        ← quantized weight format (llama.cpp)
│
└── Hardware Backends & SDKs
    ├── CUDA / PTX                  ← NVIDIA GPU compute
    ├── ROCm / HIP                  ← AMD GPU compute
    ├── oneAPI / DPC++              ← Intel unified programming model
    ├── Metal / MPS                 ← Apple GPU compute
    ├── WebGPU / WGSL               ← browser GPU compute
    └── OpenCL                      ← cross-vendor compute (legacy)

"""