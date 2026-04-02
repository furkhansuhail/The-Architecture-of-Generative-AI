"""
GPU-Accelerated AI Inference — From CUDA Cores to Production Serving
=====================================================================

AI inference is the act of running a trained neural network to produce
predictions. Training is done once; inference runs billions of times per
day across every product that uses AI — search, recommendations, voice
assistants, image recognition, language models.

The performance gap between GPU and CPU for inference is enormous:
    - An NVIDIA A100 delivers 312 TFLOPS of BF16 compute
    - A high-end Intel Xeon delivers ~2 TFLOPS
    - Ratio: 150× more raw compute on the GPU

But raw FLOPS is only part of the story. Getting a model from "trained
checkpoint" to "serving 10,000 requests/second at under 10ms latency"
requires understanding the full inference stack:
    GPU hardware:          how CUDA cores, Tensor Cores, and HBM work
    Memory hierarchy:      why bandwidth matters more than FLOPS for LLMs
    Quantisation:          INT8/FP8/INT4 to fit models in GPU memory
    Kernel optimisation:   FlashAttention, fused ops, custom CUDA kernels
    Batching strategies:   static, dynamic, continuous (for LLMs)
    Runtime systems:       TensorRT, vLLM, ONNX Runtime, TorchScript
    Multi-GPU deployment:  tensor parallelism, pipeline parallelism

This module builds understanding from the GPU hardware up through every
layer of the inference stack, with concrete techniques for maximising
throughput and minimising latency in production systems.

"""

import textwrap
import re

TOPIC_NAME   = "GPU-Accelerated AI Inference"
DISPLAY_NAME = "21 · GPU Inference"
ICON         = "🚀"
SUBTITLE     = "From CUDA Architecture to Production-Scale Model Serving"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — GPU ARCHITECTURE: WHY GPUS DOMINATE AI INFERENCE

### CPU vs GPU: Fundamental Design Philosophy

    CPU (Central Processing Unit):
        Designed for LATENCY — complete one complex task as fast as possible.
        4–64 powerful cores, each with:
            Large out-of-order execution engine (re-orders instructions for speed)
            Deep branch prediction (predicts if/else outcomes)
            Huge L1/L2/L3 caches (32 MB+ to avoid memory stalls)
            Very fast single-thread performance (~GHz clock)
        Memory: DDR5 ~50–100 GB/s bandwidth

    GPU (Graphics Processing Unit):
        Designed for THROUGHPUT — complete thousands of simple tasks in parallel.
        1000s–10000s of simple cores (CUDA cores), each with:
            Tiny execution engine (no branch prediction, small cache)
            Very efficient at SIMD (same instruction, different data)
            Excellent at floating-point arithmetic
        Memory: HBM (High Bandwidth Memory) ~1–3 TB/s bandwidth

    For matrix multiplication (A @ B where A=1024×1024, B=1024×1024):
        Each output element needs 1024 multiply-accumulate operations.
        1024×1024 = 1M output elements, all INDEPENDENT.
        GPU: dispatch all 1M output elements to parallel CUDA threads.
        CPU: compute sequentially with SIMD, filling 8–16 elements per cycle.
        Result: GPU is 100–500× faster for large matrix operations.

### NVIDIA GPU Architecture Deep Dive

    KEY UNIT: Streaming Multiprocessor (SM)
        The SM is the fundamental compute unit of an NVIDIA GPU.
        One GPU has tens to hundreds of SMs.

        Inside one SM (Ampere A100):
            ┌───────────────────────────────────────────────────────┐
            │  64 × FP32 CUDA cores (64 mul-add per clock cycle)    │
            │  64 × INT32 cores                                     │
            │  32 × FP64 cores                                     │
            │   4 × Tensor Cores (for matrix math — the key for AI) │
            │   1 × Special Function Unit (sin, cos, sqrt, ...)     │
            │  L1 cache + Shared Memory (192 KB configurable)       │
            │  Register file (256 KB)                               │
            └───────────────────────────────────────────────────────┘

        A100 has 108 SMs:
            108 × 64 FP32 CUDA cores = 6,912 CUDA cores total
            108 ×  4 Tensor Cores   =   432 Tensor Cores total

    TENSOR CORES — The AI Acceleration Engine:
        Tensor Cores perform WARP-LEVEL matrix multiply-accumulate (MMA):
            D = A × B + C
        where A, B, C, D are small matrices (4×4, 8×16, etc.).

        Comparison (A100):
            FP32 CUDA cores:   19.5 TFLOPS (1 multiply+1 add per core per cycle)
            BF16 Tensor Cores: 312 TFLOPS (16× more throughput!)
            INT8 Tensor Cores: 624 TFLOPS (32× more!)
            FP8  Tensor Cores: 1248 TFLOPS (64×!)  (H100 only)

        This is WHY quantisation (INT8, FP8) is so valuable:
        Not just memory savings — Tensor Cores get faster too.

    MEMORY HIERARCHY:
        Registers (per thread):     Fastest. ~256 KB per SM. Private.
        Shared Memory (per SM):     Fast. 48–192 KB. Shared among threads in block.
        L1 Cache (per SM):          Fast. Part of the 192 KB shared/L1.
        L2 Cache (per GPU):         40 MB on H100. All SMs share this.
        HBM (GPU global memory):    ~TB/s bandwidth. 80 GB on A100/H100.
        System RAM (via PCIe):      ~32 GB/s. Slow CPU↔GPU transfers.

### CUDA Execution Model

    CUDA organises threads into a hierarchy:

        Thread:      One worker. Runs one instance of the kernel function.
        Warp:        32 threads that execute TOGETHER (SIMD unit).
                     All 32 threads execute the SAME instruction each cycle.
        Block:       1–1024 threads. Share Shared Memory. One SM handles one block.
        Grid:        All blocks in a kernel launch. Can be millions of blocks.

    Warp divergence — the main pitfall:
        If threads in a warp take different branches (if/else):
            ALL threads in the warp must execute BOTH branches.
            One branch is masked out for some threads.
            Half the throughput wasted.

        For ML kernels: avoid data-dependent branches in inner loops.
        Design kernels so ALL threads take the same code path.

    Memory coalescing — critical for bandwidth:
        When threads in a warp access memory consecutively (address i, i+1, i+2...),
        the hardware combines these into ONE memory transaction.
        When threads scatter-access random addresses: ONE transaction PER THREAD.
        64× bandwidth difference between coalesced and scattered access.

        For matrix multiplication: the order of loops determines coalescing.
        Transposing input matrices before multiplication improves coalescing.

### GPU Generations and Their ML Relevance

    Tesla (2006):     First GPGPU. No Tensor Cores. Historical.
    Volta (2017):     First Tensor Cores. V100. FP16 Tensor Cores (125 TFLOPS FP16).
    Turing (2018):    RTX series. Added INT8 Tensor Cores. DLSS.
    Ampere (2020):    A100. BF16 Tensor Cores, sparsity support.
                      312 TFLOPS BF16 dense, 624 TFLOPS with 2:4 sparsity.
    Hopper (2022):    H100. FP8 Tensor Cores (1978 TFLOPS FP8).
                      Transformer Engine: dynamic FP8 scaling per layer.
                      NVLink 4.0: 900 GB/s GPU-to-GPU bandwidth.
    Ada Lovelace (2022): RTX 4090. Consumer. 1321 TFLOPS FP8.
    Blackwell (2024): B200. FP4 Tensor Cores (~20 PFLOPS).
                      GB200 "Grace-Blackwell": CPU+GPU on same die (NVL72 system).

    Memory comparison:
        V100:  900 GB/s HBM2,   16/32 GB
        A100:  2000 GB/s HBM2e, 40/80 GB
        H100:  3350 GB/s HBM3,  80 GB
        H200:  4800 GB/s HBM3e, 141 GB
        B200:  8000 GB/s HBM3e, 192 GB


##### PART 2 — THE INFERENCE BOTTLENECK: COMPUTE VS MEMORY BANDWIDTH

### Roofline Model: Are You Compute or Memory Bound?

    Every operation falls into one of two regimes:

    COMPUTE BOUND: arithmetic intensity is HIGH.
        GPU spends most time doing FLOPs.
        Performance limited by Tensor Core throughput.
        Example: large matrix multiply with tall/wide matrices.
        To improve: use Tensor Cores, increase precision, fuse ops.

    MEMORY BOUND: arithmetic intensity is LOW.
        GPU spends most time waiting for data from HBM.
        Performance limited by HBM bandwidth.
        Example: elementwise ops (relu, add), LLM decode step.
        To improve: reduce memory traffic (fusion, quantisation, caching).

    Arithmetic intensity = FLOPs / Bytes accessed
        A100: 312 TFLOPS compute / 2000 GB/s bandwidth = 156 FLOPS/byte
        H100: 1979 TFLOPS FP8 / 3350 GB/s = ~590 FLOPS/byte (FP8 ridge)

    For an operation to be compute-bound, its arithmetic intensity must
    exceed the GPU's compute-to-bandwidth ratio (the "ridge point").

### Why LLM Decode Is Memory Bound

    During LLM token generation (decode phase):
        Model: LLaMA-2 7B, batch=1
        Weights: 7B × 2 bytes (BF16) = 14 GB
        Each decode step reads ALL 14 GB of weights to produce ONE token.
        Arithmetic: 7B × 2 FLOPs = 14 GFLOPs
        Intensity: 14 GFLOPs / 14 GB = 1 FLOP/byte

    Ridge point of A100: 156 FLOPS/byte.
    LLM decode: 1 FLOP/byte → deeply memory bound!

    Throughput = min(peak_compute, peak_bandwidth × arithmetic_intensity)
    With batch=1:  A100 BW × 1 FLOP/byte = 2000 GB/s × 1 = 2 TFLOPS
    Actual A100:   312 TFLOPS available but only 2 TFLOPS used.
    GPU utilisation for LLM decode at batch=1: < 1%!

    The fix: BATCHING. With batch=32:
        Intensity = 32 FLOP/byte (still memory bound but 32× better)
    With batch=200+:
        Intensity > 156 FLOP/byte → compute bound, full GPU utilisation.

    This is why throughput-optimised LLM servers (vLLM, SGLang, TRT-LLM)
    work so hard to batch as many requests as possible together.

### The KV Cache Memory Problem

    LLM attention requires access to ALL previous tokens' keys and values.
    This KV cache grows as generation proceeds:

    KV cache size per token:
        = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
        LLaMA-2 7B:  2 × 32 × 32 × 128 × 2 bytes = 524,288 bytes = 0.5 MB/token

    For batch=64, max_seq_len=2048:
        KV cache = 64 × 2048 × 0.5 MB = 65,536 MB = 64 GB ← exceeds A100!

    Solutions (covered in vLLM and TRT-LLM modules):
        PagedAttention (vLLM): allocate KV cache in pages like virtual memory
        Grouped Query Attention (GQA): fewer KV heads = proportionally less cache
        Sliding window attention: only attend to last N tokens
        Quantised KV cache: store in INT8 instead of BF16 (2× smaller)


##### PART 3 — QUANTISATION: FITTING MORE MODEL INTO LESS MEMORY

### Why Quantise?

    The same LLaMA-2 70B model at different precisions:
        FP32:  70B × 4 bytes = 280 GB → needs 4× A100 80GB
        BF16:  70B × 2 bytes = 140 GB → needs 2× A100 80GB
        INT8:  70B × 1 byte  = 70 GB  → fits in 1× A100 80GB (barely)
        INT4:  70B × 0.5 byte = 35 GB → fits in 1× A100 40GB
        FP8:   70B × 1 byte  = 70 GB  → same as INT8 but higher accuracy

    Benefits of quantisation:
        1. Memory: fit larger models on fewer GPUs
        2. Bandwidth: read less data per forward pass (decode speed)
        3. Compute: INT8/FP8 Tensor Cores are 2–4× faster than BF16

### Quantisation Mathematics

    Linear quantisation (most common):
        float_value ≈ scale × (int_value - zero_point)

        Symmetric quantisation (zero_point = 0):
            scale = max(|W|) / (2^(bits-1) - 1)
            q(W) = round(W / scale)
            dq(Q) = Q × scale

        Asymmetric quantisation:
            scale = (W_max - W_min) / (2^bits - 1)
            zero_point = round(-W_min / scale)
            q(W) = clamp(round(W / scale) + zero_point, 0, 2^bits - 1)

    Per-tensor vs per-channel quantisation:
        Per-tensor: one scale/zero_point for the ENTIRE weight matrix.
                    Simplest, but large accuracy loss if weight ranges vary.
        Per-channel (per-row or per-column): one scale per output channel.
                    Better accuracy — each channel has its own range.
                    Standard for most INT8 quantisation.
        Per-group:  one scale per group of (e.g.) 128 weights.
                    Used in GPTQ, AWQ for INT4 (better accuracy than per-channel INT4).

### Quantisation Methods

    POST-TRAINING QUANTISATION (PTQ) — no retraining:

    Dynamic Quantisation:
        Weights quantised to INT8 offline.
        Activations quantised to INT8 DYNAMICALLY at runtime (per batch).
        No calibration data needed.
        Works well for: linear layers, LSTM, Transformer encoders.
        Tools: PyTorch quantize_dynamic, ONNX Runtime dynamic quant.

    Static Quantisation:
        Weights AND activations quantised offline.
        Requires a calibration dataset to determine activation ranges.
        Uses statistical calibration: percentile, entropy, MSE minimisation.
        Better throughput than dynamic (no runtime quantisation overhead).
        Tools: PyTorch quantize_static, TensorRT, ONNX Runtime static.

    GPTQ (Generative Post-Training Quantisation):
        Compresses LLM weights to INT4 using layer-wise second-order info.
        Uses Hessian information to find weight errors caused by quantisation
        and compensates them by updating remaining unquantised weights.
        Process: quantise one column of the weight matrix at a time,
                 update remaining columns to compensate for the error.
        Result: INT4 models with accuracy very close to FP16.
        Used for: 4-bit LLaMA, Mistral, Falcon inference.

    AWQ (Activation-aware Weight Quantisation):
        Key insight: not all weights are equally important.
        Weights corresponding to LARGE activations matter more.
        AWQ protects the ~1% of salient weights by keeping them in higher precision
        or by pre-scaling to make them easier to quantise accurately.
        Result: INT4 with better accuracy than GPTQ in many cases.
        Tools: llm-awq, AutoAWQ (HuggingFace integrated).

    TRAINING-AWARE:

    QAT (Quantisation-Aware Training):
        Simulate INT8 quantisation during training with "fake quantise" ops.
        Model learns to be robust to quantisation noise.
        Best accuracy — often within 0.1% of FP32.
        Cost: requires retraining (expensive for large models).
        Tools: PyTorch quantization, TensorFlow model optimisation.

    FP8 Training + Inference (H100 Transformer Engine):
        NVIDIA's Transformer Engine dynamically scales FP8 per tensor.
        No explicit quantisation step — the hardware handles it.
        Uses FP8-E4M3 for weights (better range), FP8-E5M2 for gradients.
        Result: same accuracy as BF16 with 2× the throughput.

### Quantisation Accuracy Impact

    Typical accuracy loss (ImageNet classification, ResNet-50):
        FP32 → BF16:    0.0% accuracy drop (numerically equivalent)
        FP32 → INT8:    0.1–0.5% accuracy drop (post-training static)
        FP32 → INT4:    0.5–2% accuracy drop (per-group GPTQ/AWQ)
        FP32 → INT4:    0.1–0.5% accuracy drop (with QAT)

    For LLMs (perplexity change on WikiText-2, LLaMA-2 7B):
        FP16 → BF16:    0.0% change
        FP16 → INT8:    ~0.01 perplexity increase
        FP16 → INT4 (GPTQ, group=128): ~0.1–0.3 perplexity increase
        FP16 → INT4 (naïve, no group): large perplexity increase
        FP16 → INT4 (AWQ):   similar to GPTQ, sometimes better


##### PART 4 — KERNEL OPTIMISATION: FLASHATTENTION AND OP FUSION

### Why Standard Attention Is Slow

    Standard multi-head attention for sequence length N, head dim d:
        1. Compute Q, K, V: O(Nd²) FLOPs — fast (matmul)
        2. Compute S = QK^T: O(N²d) FLOPs — quadratic in N
        3. Write S to HBM (N×N matrix): O(N²) memory writes
        4. Compute P = softmax(S): reads S back from HBM
        5. Write P to HBM: O(N²) memory writes
        6. Compute output = PV: reads P from HBM

    Problem: steps 3–5 read/write N² elements.
    For N=2048, d=64: N² = 4M elements × 2 bytes = 8 MB PER HEAD.
    With 32 heads, 32 layers: 8 MB × 32 × 32 = 8 GB of HBM traffic
    just for attention score intermediates.

    At 2000 GB/s bandwidth: 8 GB / 2000 GB/s = 4 ms per forward pass.
    This is MEMORY BANDWIDTH bound — the bottleneck.

### FlashAttention — The Solution

    FlashAttention (Dao et al., 2022) never materialises the full N×N matrix.

    Key insight: TILE the computation to fit in SRAM (Shared Memory on GPU).
        Shared Memory: ~192 KB (fast, ~10 TB/s effective bandwidth)
        vs HBM:        ~80 GB (slow, ~2 TB/s)

    Algorithm:
        1. Split Q into blocks of size Br rows
        2. Split K, V into blocks of size Bc rows
        3. For each Q block, iterate over all K, V blocks:
            a. Load Q block, K block, V block into SRAM
            b. Compute partial attention: S_block = Q_block × K_block^T
            c. Apply online softmax (maintains running max + sum across blocks)
            d. Accumulate partial output: O_block += softmax_block × V_block
        4. Only write O (the final output) to HBM — never write S or P!

    Memory reduction:
        Standard: O(N²) HBM reads/writes for S and P
        FlashAttention: O(N) HBM reads/writes (only O goes to HBM)

    Speed improvement:
        2–4× faster wall-clock time for attention
        For N=4096: ~5.5× speedup over standard PyTorch attention

    FlashAttention-2 (2023):
        Better work partitioning between warps (fewer idle warps)
        Reduced shared memory synchronisation
        ~2× speedup over FlashAttention-1

    FlashAttention-3 (2024, H100):
        Exploits H100's async memory copies
        Overlaps data transfers with Tensor Core computation
        ~1.5–2× faster than FlashAttention-2 on H100

    Usage in frameworks:
        PyTorch 2.0+: F.scaled_dot_product_attention() uses FlashAttention
        HuggingFace: automatically used when flash_attention_2 is enabled
        vLLM, TRT-LLM: FlashAttention as the default attention backend

### Operator Fusion

    Fusion combines multiple CUDA kernel launches into one:

    BEFORE fusion (bias + GELU activation):
        Launch kernel 1: matmul(X, W) → intermediate [GPU global memory]
        Launch kernel 2: add(intermediate, bias) → intermediate2 [GPU global memory]
        Launch kernel 3: gelu(intermediate2) → output [GPU global memory]
        3 kernel launches, 3 HBM round-trips

    AFTER fusion (BiasGELU fused kernel):
        Launch 1 kernel: fused_bias_gelu(X, W, bias) → output
        1 kernel launch, 1 HBM write (reads W once, writes output once)
        Memory bandwidth: 3× less HBM traffic
        Latency: typically 1.5–3× faster for elementwise chains

    Common fusions in LLM inference:
        RMSNorm + QKV projection: normalise then project in one pass
        Attention + softmax: FlashAttention (the ultimate attention fusion)
        Residual + LayerNorm: add residual and normalise together
        SwiGLU / GeGLU: gated activation as one op
        GEMM + bias + activation: key for MLP layers in transformers

    Tools for fusion:
        TensorRT:  aggressive automatic fusion + custom fusion rules
        XLA:       analytical fusion (see XLA module)
        Triton:    write custom fused kernels in Python-like language
        torch.compile: TorchInductor generates fused code automatically

### Triton — Python-Level GPU Kernel Programming

    Triton (OpenAI, 2019) lets you write custom GPU kernels in Python:
        No CUDA C++ needed — Python with Triton decorators
        Automatic shared memory tiling and vectorisation
        Performance competitive with hand-written CUDA

    Triton kernel example (elementwise GELU):
        @triton.jit
        def gelu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
            pid  = tl.program_id(0)             # which block is this?
            offs = pid * BLOCK + tl.arange(0, BLOCK)   # elements this block handles
            mask = offs < N                     # handle last block edge case
            x    = tl.load(x_ptr + offs, mask=mask)    # load from HBM
            # GELU: x * 0.5 * (1 + tanh(0.797885 * (x + 0.044715 * x³)))
            y    = x * 0.5 * (1 + tl.math.tanh(0.7978845 * (x + 0.044715 * x*x*x)))
            tl.store(out_ptr + offs, y, mask=mask)      # write to HBM

    PyTorch 2.0's torch.compile backend (TorchInductor) generates Triton kernels
    automatically from pure Python PyTorch code.


##### PART 5 — BATCHING STRATEGIES FOR MAXIMUM THROUGHPUT

### Static Batching

    Simplest form: collect N requests, run one forward pass.
        Wait for N requests to arrive → group into batch → run → return results.

    Problem 1: PADDING WASTE
        If request A is 10 tokens and request B is 100 tokens:
        Pad A to 100 tokens with zeros → 90% of A's computation is wasted.
        With diverse sequence lengths: can waste 50%+ of compute.

    Problem 2: LATENCY VS THROUGHPUT TRADEOFF
        Wait for large batch → high latency for early arrivals.
        Small batch → low latency but low GPU utilisation.
        No good middle ground for variable-arrival rate serving.

    When to use: offline batch processing where latency doesn't matter.
    NOT for: real-time serving with diverse input lengths.

### Dynamic Batching

    Group requests arriving within a time window:
        Server waits up to T ms for more requests.
        Groups all arrived requests into one batch.
        Pads to longest sequence in the batch.

    Better than static for serving:
        Short window (2–5 ms) balances latency and throughput.
        Still wastes compute on padding.

    Padding waste mitigation:
        Sort requests by length before batching.
        Group similar-length requests together → less padding per batch.
        TensorRT Dynamic Shapes: compile one engine for a range of shapes.

### Continuous Batching (Iteration-Level Scheduling)

    The breakthrough for LLM serving. Pioneered by Orca (2022).
    Now standard in vLLM, TRT-LLM, SGLang.

    Key insight: in LLM generation, different requests finish at different steps.
    Don't waste GPU cycles running already-finished requests.

    WITHOUT continuous batching:
        Batch: [req A (2000 tokens), req B (50 tokens), req C (1500 tokens)]
        Must run ALL 2000 steps even after B finishes at step 50.
        B's slot wastes compute for 1950 steps.

    WITH continuous batching:
        After req B finishes at step 50: immediately insert req D into that slot.
        GPU never runs empty slots.
        Result: 2–5× higher throughput at same GPU count.

    Implementation: the scheduler makes decisions at EVERY decode step:
        "Which requests should execute this step?"
        "Which finished requests should be replaced with new ones?"
        "Which requests should be paused (preempted) for memory?"

### Speculative Decoding

    Key observation: LLM autoregressive generation is memory bound at batch=1.
    Idea: use a small DRAFT model to propose multiple tokens, then verify with
    the large ORACLE model in one forward pass.

    Algorithm:
        1. Draft model generates K tokens autoregressively (fast, small model)
        2. Oracle model verifies ALL K tokens in ONE forward pass
           (batch K+1 tokens together → only 1 KV cache miss)
        3. Accept tokens up to the first mismatch (rejection sampling)
        4. On mismatch: use oracle's distribution for that token
        5. Repeat from step 1

    Speedup = K_accepted / 1 (oracle forward passes)
        If acceptance rate α ≈ 0.9 and draft generates K=4:
        Expected accepted = 3.6 tokens per oracle call
        vs. 1 token per oracle call normally
        → ~3.6× speedup in wall-clock token generation

    Works best when:
        - Draft and oracle have similar output distributions (same family)
        - Acceptance rate is high (task-specific — factual > creative)
        - Draft model is small (< 1/10 oracle size)

    Examples:
        Llama-2 70B + Llama-2 7B draft: 2–3× speedup
        Google PaLM2 uses this in production
        vLLM, TRT-LLM support speculative decoding


##### PART 6 — TENSORRT: MAXIMUM GPU INFERENCE PERFORMANCE

### What TensorRT Does

    TensorRT takes a trained model (ONNX, TF, PyTorch) and:
        1. Parses the model graph
        2. Applies graph optimisations (fusion, dead node elimination)
        3. Selects optimal CUDA kernels for each op (among 1000s of candidates)
        4. Plans memory allocation (minimal peak memory)
        5. Compiles everything into an optimised .engine file

    Result: 2–10× faster inference vs PyTorch eager on the same GPU.

### TensorRT Optimisation Techniques

    Layer and tensor fusion:
        Conv + BN + ReLU → single CudnnConvBiasActLayer
        MatMul + BiasAdd + GELU → single FusedMatmulBiasActivation
        Multiple small operations → single custom CUDA kernel

    Kernel auto-selection:
        For each operation, TensorRT benchmarks multiple algorithm implementations:
            For Conv: direct convolution, FFT-based, Winograd, GEMM-based
            Chooses the fastest one for the specific input shape and GPU
        This is why TensorRT build time is slow (minutes to hours):
            It's benchmarking thousands of kernel combinations

    Precision modes:
        FP32:  full precision (slowest, most accurate)
        FP16:  2× throughput on Tensor Cores, ~1% accuracy loss on average
        BF16:  same range as FP32, good for LLMs
        INT8:  4× throughput, requires calibration, ~1% loss
        FP8:   available on H100, highest throughput
        Mixed: FP16 for most ops, FP32 for sensitive ops (softmax, layernorm)

    Dynamic shapes:
        OptimisationProfile: define (min, opt, max) shapes
        TensorRT builds kernels for the full range
        At runtime: selects the best kernel for the actual shape

### TensorRT-LLM

    TRT-LLM is NVIDIA's TensorRT-based serving stack specifically for LLMs:
        Pre-built engines for LLaMA, Mistral, Falcon, GPT, BERT, etc.
        In-flight batching (continuous batching)
        Custom attention kernels (FlashAttention + masked attention)
        INT8/FP8/INT4 quantisation (SmoothQuant, AWQ, GPTQ)
        Multi-GPU tensor parallelism and pipeline parallelism
        Triton Inference Server integration for production serving

    Build and serve workflow:
        # 1. Convert weights
        python convert_checkpoint.py --model llama-2-7b --dtype float16

        # 2. Build TRT-LLM engine
        trtllm-build --checkpoint_dir ./ckpt \
                     --output_dir ./engine \
                     --max_batch_size 32 \
                     --max_input_len 2048 \
                     --max_output_len 512 \
                     --gemm_plugin float16

        # 3. Run with Triton Inference Server
        tritonserver --model-repository ./triton_repo

### Key TensorRT Performance Numbers

    LLaMA-2 7B on A100 80GB (from NVIDIA benchmarks):
        PyTorch eager (BF16):           ~50 tokens/sec, batch=1
        TRT-LLM (FP16, continuous):     ~3000 tokens/sec, batch=128
        TRT-LLM (INT8, continuous):     ~5000 tokens/sec, batch=128
        Improvement from eager: 60–100× higher throughput

    ResNet-50 on A100 (ImageNet, batch=64):
        PyTorch eager (FP32):           ~2000 images/sec
        TensorRT (FP32):                ~4000 images/sec (2×)
        TensorRT (FP16):                ~9000 images/sec (4.5×)
        TensorRT (INT8):                ~18000 images/sec (9×)


##### PART 7 — MULTI-GPU INFERENCE: PARALLELISM STRATEGIES

### When You Need Multiple GPUs

    A model that doesn't fit in one GPU's memory must be split.
    Also: to achieve lower latency via parallel computation.

    GPU memory requirements (BF16):
        LLaMA-2 7B:   14 GB → fits in A100 40GB (with KV cache)
        LLaMA-2 13B:  26 GB → fits in A100 40GB (barely)
        LLaMA-2 70B:  140 GB → needs 2× A100 80GB
        LLaMA-2 70B INT4: 35 GB → fits in one A100 40GB
        Llama-3 405B: 810 GB → needs 10× A100 80GB minimum

### Data Parallelism (DP)

    Each GPU holds a FULL copy of the model.
    Different batches of requests are assigned to different GPUs.

    GPU 0: model copy → batch 0 → result 0
    GPU 1: model copy → batch 1 → result 1
    GPU 2: model copy → batch 2 → result 2

    Pros: simple, scales throughput linearly with GPUs.
    Cons: model must fit on one GPU. No latency improvement.
    Use: when model fits in one GPU and you need more throughput.

### Tensor Parallelism (TP)

    Each GPU holds a SHARD of the model weights.
    One request is processed across all GPUs simultaneously.

    Example — Linear layer (Y = X @ W, W is 4096×4096):
        GPU 0: W[:, :1024] → partial_output_0   (columns 0–1023)
        GPU 1: W[:, 1024:2048] → partial_output_1
        GPU 2: W[:, 2048:3072] → partial_output_2
        GPU 3: W[:, 3072:] → partial_output_3
        AllReduce: partial outputs summed across all GPUs

    For attention layers (Megatron-LM style):
        QKV projection: split across heads — each GPU handles some attention heads
        Output projection: split by input dimension

    Each forward pass requires AllReduce communications.
    NVLink (GPU-to-GPU interconnect): 600–900 GB/s per GPU.
    PCIe (when no NVLink): 32–64 GB/s → much slower AllReduce.

    Latency improvement: N-GPU TP reduces layer computation by ~N×.
    Communication overhead: adds AllReduce latency (NVLink: ~1 ms).
    Net latency reduction: 2–4× for TP=4–8 within one NVLink-connected node.

### Pipeline Parallelism (PP)

    Different layers placed on different GPUs.
    GPU 0 handles layers 0–8, GPU 1 handles layers 9–16, etc.

    GPU 0: layers 0-8   → activations
    GPU 1: layers 9-16  → activations (receives from GPU 0)
    GPU 2: layers 17-24 → activations (receives from GPU 1)
    GPU 3: layers 25-32 → output      (receives from GPU 2)

    Naive PP: only one GPU busy at a time. Inefficient.
    Micro-batching (GPipe): split batch into micro-batches.
        While GPU 3 processes micro-batch 0, GPU 2 processes micro-batch 1, etc.
        Reduces pipeline bubble from O(N_stages) to O(N_stages/N_microbatches).

    Use: when model doesn't fit across NVLink nodes.
          Cross-node (InfiniBand) for the layer boundary transfers.

### Expert Parallelism (EP) — for MoE Models

    Mixture-of-Experts (Mixtral, DeepSeek, DBRX) use multiple "expert" FFNs.
    Each token is routed to 2–8 of the N experts.

    Place different experts on different GPUs:
        GPU 0: experts 0, 1, 2, 3
        GPU 1: experts 4, 5, 6, 7
        etc.

    All-to-All communication: route each token to the GPU holding its expert.
    This is MORE communication than TP but enables much larger models.

    Combined parallelism in production:
        LLM with 400B parameters, 32 experts, 16 GPUs:
        TP=2 × PP=4 × EP=2 — all three strategies simultaneously.


##### PART 8 — INFERENCE RUNTIME ECOSYSTEM AND DEPLOYMENT PATTERNS

### Runtime Comparison

    ┌────────────────────────────────────────────────────────────────────┐
    │ Runtime       │ Best for           │ Hardware     │ Latency/Thput  │
    ├────────────────────────────────────────────────────────────────────┤
    │ TensorRT      │ CNN/ViT, fixed shape│ NVIDIA only  │ Highest thput  │
    │ TRT-LLM       │ LLM serving        │ NVIDIA only  │ Highest thput  │
    │ vLLM          │ LLM serving        │ NVIDIA/AMD   │ High thput     │
    │ SGLang        │ Multi-call LLMs    │ NVIDIA/AMD   │ High thput     │
    │ ONNX Runtime  │ Cross-platform     │ Any hardware │ Good           │
    │ torch.compile │ Research → prod    │ Any          │ Good           │
    │ TFLite        │ Mobile/edge        │ ARM/NPU      │ Edge-optimised │
    │ OpenVINO      │ Intel hardware     │ Intel CPU/VPU│ Intel-optimised│
    │ CoreML        │ Apple devices      │ Apple ANE    │ Apple-optimised│
    └────────────────────────────────────────────────────────────────────┘

### Key Inference Metrics

    LATENCY metrics:
        TTFT (Time To First Token): prefill latency. Perceived as "responsiveness."
        ITL / TPOT (Inter-Token Latency / Time Per Output Token): decode latency.
        E2E latency: TTFT + N × ITL (total time for N output tokens).
        P50/P95/P99: percentile latencies (P99 = worst 1% of requests).

    THROUGHPUT metrics:
        Requests/second (RPS): how many requests completed per second.
        Output tokens/second: total tokens generated per second across all requests.
        Tokens/second/GPU: normalised by GPU count (efficiency metric).

    EFFICIENCY metrics:
        GPU utilisation: what fraction of compute cycles doing useful work.
        MFU (Model FLOP Utilisation): actual FLOPs / theoretical peak FLOPs.
        GPU memory utilisation: fraction of HBM used.

    The tradeoff: latency vs throughput
        Low batch size → low latency, low throughput, low GPU utilisation
        High batch size → high latency, high throughput, high GPU utilisation
        Continuous batching → approaches optimal balance

### GPU Memory Management Best Practices

    Avoid memory fragmentation:
        Allocate tensors in large contiguous blocks where possible.
        Use memory pools (PyTorch's CUDA caching allocator does this).
        Pre-allocate KV cache buffers at server startup (TRT-LLM, vLLM).

    Reduce peak memory:
        Activation checkpointing: don't store all activations (for long context).
        Offloading: move KV cache to CPU RAM when GPU is full (slower).
        Quantised KV cache: store in INT8 instead of FP16 (2× less memory).

    Monitor GPU memory:
        nvidia-smi            → live GPU memory usage
        torch.cuda.memory_summary()  → PyTorch memory breakdown
        nsys profile          → NVIDIA Nsight Systems profiling
        ncu profile           → NVIDIA Nsight Compute kernel profiling

### Production Inference Architecture

    A production LLM serving system:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Load Balancer  (nginx, HAProxy, Envoy)                         │
    │  Rate limiting, request routing, health checks                  │
    └───────────────────┬─────────────────────────────────────────────┘
                        │  HTTP/gRPC requests
    ┌───────────────────▼─────────────────────────────────────────────┐
    │  Inference Gateway  (router + request queue)                    │
    │  Auth, caching (semantic cache), queue management               │
    └───────────────────┬─────────────────────────────────────────────┘
                        │  Batch of requests
    ┌───────────────────▼─────────────────────────────────────────────┐
    │  Inference Server  (vLLM / TRT-LLM / Triton)                   │
    │  Continuous batching scheduler                                  │
    │  KV cache manager (PagedAttention)                              │
    │  Model execution engine                                         │
    └───────────────────┬─────────────────────────────────────────────┘
                        │  GPU kernels
    ┌───────────────────▼─────────────────────────────────────────────┐
    │  GPU(s): NVIDIA A100/H100 with NVLink                           │
    │  TensorRT / custom CUDA kernels                                 │
    │  FlashAttention, Tensor Cores, HBM3                             │
    └─────────────────────────────────────────────────────────────────┘

    Monitoring stack:
        Prometheus + Grafana: metrics dashboard
        DCGM (Data Center GPU Manager): GPU health and utilisation
        MLflow / W&B: model performance tracking

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · GPU Profiling — Roofline Analysis, Memory Bandwidth & Arithmetic Intensity": {
        "description": (
            "Understand GPU performance limits from first principles. "
            "Implement the Roofline model to classify ops as compute vs memory bound. "
            "Measure arithmetic intensity for common ML operations. "
            "Profile GPU memory bandwidth utilisation. "
            "Show how batch size shifts ops from memory to compute bound. "
            "Calculate theoretical speedups from quantisation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  GPU PROFILING — ROOFLINE, BANDWIDTH & ARITHMETIC INTENSITY")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Roofline model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Roofline model: classify ops as compute/memory bound")
print("━" * 65)
print()

# GPU specifications (representative values)
GPU_SPECS = {
    "V100 (Volta)":   {"tflops_fp16": 125.0,  "bandwidth_gb": 900.0,  "hbm_gb": 32},
    "A100 (Ampere)":  {"tflops_bf16": 312.0,  "bandwidth_gb": 2000.0, "hbm_gb": 80},
    "H100 (Hopper)":  {"tflops_bf16": 989.0,  "bandwidth_gb": 3350.0, "hbm_gb": 80,
                        "tflops_fp8": 1979.0},
    "RTX 4090 (Ada)": {"tflops_fp16": 330.0,  "bandwidth_gb": 1008.0, "hbm_gb": 24},
    "A10G (Ampere)":  {"tflops_bf16": 31.2,   "bandwidth_gb": 600.0,  "hbm_gb": 24},
}

print("  GPU Hardware Specifications:")
print(f"  {'GPU':<22} | {'Peak BF16 (TFLOPS)':>20} | {'HBM BW (GB/s)':>15} | "
      f"{'Ridge (FLOP/B)':>15} | {'HBM (GB)':>9}")
print(f"  {'─'*87}")
for name, specs in GPU_SPECS.items():
    tflops = specs.get("tflops_bf16", specs.get("tflops_fp16", 0))
    bw     = specs["bandwidth_gb"]
    ridge  = tflops * 1000 / bw   # convert TFLOPS to GFLOPS then to FLOP/Byte
    print(f"  {name:<22} | {tflops:>20.1f} | {bw:>15.0f} | {ridge:>15.1f} | "
          f"{specs['hbm_gb']:>9}")
print()

print("  Roofline model — an operation is:")
print("    COMPUTE BOUND   if arithmetic intensity > ridge point")
print("    MEMORY BOUND    if arithmetic intensity < ridge point")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Arithmetic intensity of common ML operations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Arithmetic intensity of ML operations")
print("━" * 65)
print()

def matmul_intensity(M, K, N, bytes_per_elem=2):
    """
    Matrix multiply C = A @ B
    A: M×K, B: K×N, C: M×N
    FLOPs: 2 × M × K × N  (multiply + add for each element)
    Bytes: read A + read B + write C = (M×K + K×N + M×N) × bpe
    """
    flops = 2 * M * K * N
    bytes_accessed = (M*K + K*N + M*N) * bytes_per_elem
    return flops, bytes_accessed, flops / bytes_accessed

def elementwise_intensity(N, bytes_per_elem=2, ops_per_elem=1):
    """
    Elementwise op (relu, gelu, etc.) on N-element tensor
    FLOPs: ops_per_elem × N
    Bytes: read + write = 2 × N × bpe
    """
    flops = ops_per_elem * N
    bytes_accessed = 2 * N * bytes_per_elem
    return flops, bytes_accessed, flops / bytes_accessed

def attention_intensity(B, H, N, d, bytes_per_elem=2):
    """
    Standard self-attention: softmax(QK^T / sqrt(d)) V
    FLOPs: 4 × B × H × N² × d  (QK^T, softmax, AV, plus output)
    Bytes: read Q,K,V + read/write S (N×N matrix) + write output
    FlashAttention eliminates the N×N read/write!
    """
    # Standard attention
    flops_qkt   = 2 * B * H * N * N * d      # Q @ K^T
    flops_av    = 2 * B * H * N * N * d      # A @ V
    flops_total = flops_qkt + flops_av

    # Standard: must read/write N×N attention matrix
    bytes_qkv   = 3 * B * H * N * d * bytes_per_elem
    bytes_attn  = 2 * B * H * N * N * bytes_per_elem   # write S, read S back
    bytes_out   = B * H * N * d * bytes_per_elem
    bytes_std   = bytes_qkv + bytes_attn + bytes_out

    # FlashAttention: no N×N materialization
    bytes_flash = bytes_qkv + bytes_out   # only QKV and output

    return flops_total, bytes_std, flops_total / bytes_std, bytes_flash, flops_total / bytes_flash

print("  Arithmetic intensity for key ML operations (BF16, A100 ridge=156 FLOP/B):")
print()

# LLM-relevant shapes
BATCH_SIZES = [1, 8, 64, 512]
D_MODEL     = 4096    # LLaMA-2 7B hidden dim
D_FF        = 11008   # LLaMA-2 7B FFN intermediate

print("  Matrix Multiply (Linear layer): Y = X @ W")
print(f"  W shape: ({D_MODEL}×{D_FF}) — FFN first projection in LLaMA-2 7B")
print()
print(f"  {'Batch':>6} | {'FLOPs (G)':>11} | {'Bytes (MB)':>11} | "
      f"{'Intensity (F/B)':>17} | {'Regime':>12}")
print(f"  {'─'*65}")
for B in BATCH_SIZES:
    flops, bytes_acc, intensity = matmul_intensity(B, D_MODEL, D_FF)
    regime = "COMPUTE BOUND ✅" if intensity > 156 else "MEMORY BOUND  ⚠️ "
    print(f"  {B:6d} | {flops/1e9:11.2f} | {bytes_acc/1e6:11.2f} | "
          f"{intensity:17.2f} | {regime}")
print()
print("  Key insight: batch=1 is deeply memory bound (intensity=0.73 F/B vs ridge=156)")
print("  You need batch≈200+ to be compute bound for this layer.")
print()

# Elementwise ops
print("  Elementwise ops (ReLU, GELU, Add) — ALWAYS memory bound:")
print(f"  {'Operation':>18} | {'N elements':>12} | {'F/B':>8} | {'Regime'}")
print(f"  {'─'*55}")
for op_name, n, ops_per in [("ReLU (N=1M)",  1_000_000, 1),
                              ("GELU (N=1M)",  1_000_000, 8),
                              ("Add  (N=1M)",  1_000_000, 1),
                              ("Add  (N=100M)",100_000_000, 1)]:
    f, b, i = elementwise_intensity(n, ops_per_elem=ops_per)
    regime   = "COMPUTE" if i > 156 else "MEMORY"
    print(f"  {op_name:>18} | {n:>12,} | {i:>8.3f} | {regime} BOUND")
print()
print("  Elementwise ops are ALWAYS memory bound on any GPU.")
print("  Fusion (FlashAttention, op fusion) is essential — read/write ONCE.")
print()

# Attention: standard vs FlashAttention
print("  Attention: Standard vs FlashAttention intensity comparison:")
print(f"  (LLaMA-2 7B: H=32 heads, d=128, BF16)")
print()
print(f"  {'Batch × Seq':>12} | {'Standard F/B':>14} | {'Flash F/B':>11} | {'Flash speedup':>14}")
print(f"  {'─'*60}")
for B, N in [(1, 512), (1, 2048), (8, 512), (8, 2048)]:
    flops, bytes_std, i_std, bytes_flash, i_flash = attention_intensity(B, 32, N, 128)
    speedup = bytes_std / bytes_flash   # fewer bytes = proportionally faster (memory bound)
    label   = f"{B}×{N:4d}"
    print(f"  {label:>12} | {i_std:>14.3f} | {i_flash:>11.3f} | {speedup:>13.2f}×")
print()
print("  FlashAttention's speedup grows with sequence length (more N² savings at large N).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Quantisation impact analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Quantisation: memory, bandwidth and throughput impact")
print("━" * 65)
print()

MODEL_PARAMS = {
    "LLaMA-2 7B":   7_000_000_000,
    "LLaMA-2 13B":  13_000_000_000,
    "LLaMA-2 70B":  70_000_000_000,
    "Llama-3 70B":  70_000_000_000,
    "Llama-3 405B": 405_000_000_000,
    "Mistral 7B":   7_000_000_000,
}

PRECISIONS = {
    "FP32":  {"bpe": 4, "tensor_core_mult": 0.5,  "label": "FP32 (no TC)"},
    "BF16":  {"bpe": 2, "tensor_core_mult": 1.0,  "label": "BF16 Tensor Cores"},
    "INT8":  {"bpe": 1, "tensor_core_mult": 2.0,  "label": "INT8 Tensor Cores"},
    "FP8":   {"bpe": 1, "tensor_core_mult": 4.0,  "label": "FP8 Tensor Cores (H100)"},
    "INT4":  {"bpe": 0.5,"tensor_core_mult": 4.0, "label": "INT4 weight-only"},
}

print("  Model memory requirements by precision (GPU memory needed):")
print()
print(f"  {'Model':<18} | {'FP32':>9} | {'BF16':>9} | {'INT8':>9} | "
      f"{'FP8':>9} | {'INT4':>9}")
print(f"  {'─'*68}")
for model_name, params in MODEL_PARAMS.items():
    sizes = []
    for prec, info in PRECISIONS.items():
        gb = params * info["bpe"] / 1e9
        sizes.append(f"{gb:8.1f}G")
    print(f"  {model_name:<18} | {'|'.join(sizes)}")

print()
print("  Which GPUs can fit which models at which precision?")
print()
GPU_MEMORY = {"A10G 24GB": 24, "A100 40GB": 40, "A100 80GB": 80,
               "H100 80GB": 80, "H200 141GB": 141, "B200 192GB": 192}

print(f"  (✅ = fits, ⚠️ = tight, ❌ = doesn't fit)")
print()
print(f"  {'Model × Precision':<28} | " + " | ".join(f"{g:<12}" for g in GPU_MEMORY))
print(f"  {'─'*100}")

check_cases = [
    ("LLaMA-2 7B × BF16",   14),
    ("LLaMA-2 7B × INT4",   3.5),
    ("LLaMA-2 70B × BF16",  140),
    ("LLaMA-2 70B × INT4",  35),
    ("Llama-3 405B × BF16", 810),
    ("Llama-3 405B × INT4", 202),
]
for label, gb_needed in check_cases:
    statuses = []
    for gpu_name, gpu_gb in GPU_MEMORY.items():
        if gb_needed <= gpu_gb * 0.8:   # 80% — leave room for KV cache
            statuses.append("✅")
        elif gb_needed <= gpu_gb:
            statuses.append("⚠️ ")
        else:
            statuses.append("❌")
    print(f"  {label:<28} | " + "  |  ".join(s.center(10) for s in statuses))

print()
print("  Throughput multiplier from quantisation (A100, memory-bound decode):")
print()
print(f"  {'Precision':>10} | {'Bytes/param':>12} | {'BW utilised':>13} | "
      f"{'Rel throughput':>16} | {'TC speedup'}")
print(f"  {'─'*72}")
base_throughput = None
for prec, info in PRECISIONS.items():
    bpe    = info["bpe"]
    tc_m   = info["tensor_core_mult"]
    rel_bw = 2.0 / bpe      # relative bandwidth (2 bytes = BF16 baseline)
    if base_throughput is None:
        base_throughput = rel_bw
    rel_t  = rel_bw / base_throughput
    print(f"  {prec:>10} | {bpe:>12.1f} | {rel_bw:>13.1f}× | "
          f"{rel_t:>16.1f}× (BW) | {tc_m:.1f}× (compute)")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Inference Optimisation — Quantisation, Batching & Latency Measurement": {
        "description": (
            "Practical GPU inference optimisation. "
            "Implement INT8 quantisation from scratch (symmetric and asymmetric). "
            "Demonstrate dynamic vs static quantisation error. "
            "Measure real inference latency with CUDA events. "
            "Show batching effects on throughput and latency. "
            "Benchmark torch.compile speedup. "
            "Demonstrate continuous batching concept simulation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time
import math

print("=" * 65)
print("  INFERENCE OPTIMISATION — QUANTISATION, BATCHING & LATENCY")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Quantisation from scratch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Quantisation: INT8 from first principles")
print("━" * 65)
print()

class SymmetricQuantiser:
    """
    Symmetric INT8 quantisation.
    scale = max(|W|) / 127
    q(W) = round(W / scale).clip(-128, 127)
    dq(Q) = Q × scale
    """
    def __init__(self, bits=8):
        self.bits  = bits
        self.qmax  = 2 ** (bits - 1) - 1  # 127 for INT8
        self.scale = None

    def quantise(self, W: np.ndarray, per_channel=False):
        if per_channel:
            # One scale per output channel (first dimension)
            abs_max = np.max(np.abs(W), axis=tuple(range(1, W.ndim)), keepdims=True)
        else:
            abs_max  = np.max(np.abs(W))
        self.scale = abs_max / self.qmax
        Q          = np.round(W / self.scale).clip(-self.qmax - 1, self.qmax)
        return Q.astype(np.int8)

    def dequantise(self, Q: np.ndarray):
        return Q.astype(np.float32) * self.scale


class AsymmetricQuantiser:
    """
    Asymmetric UINT8 quantisation.
    Maps [W_min, W_max] to [0, 255].
    scale = (W_max - W_min) / 255
    zero_point = round(-W_min / scale)
    q(W) = round(W / scale) + zero_point
    dq(Q) = (Q - zero_point) × scale
    """
    def __init__(self):
        self.scale      = None
        self.zero_point = None

    def quantise(self, W: np.ndarray):
        W_min, W_max    = W.min(), W.max()
        self.scale      = (W_max - W_min) / 255.0
        self.zero_point = int(np.round(-W_min / self.scale))
        Q               = np.round(W / self.scale) + self.zero_point
        return Q.clip(0, 255).astype(np.uint8)

    def dequantise(self, Q: np.ndarray):
        return (Q.astype(np.float32) - self.zero_point) * self.scale


np.random.seed(42)

# Simulate a weight matrix (like a transformer linear layer)
W_fp32 = np.random.randn(64, 64).astype(np.float32) * 0.1   # typical weight scale

print("  Weight matrix statistics:")
print(f"    Shape:  {W_fp32.shape}")
print(f"    Min:    {W_fp32.min():.4f}")
print(f"    Max:    {W_fp32.max():.4f}")
print(f"    StdDev: {W_fp32.std():.4f}")
print()

# Symmetric per-tensor
sym_qt = SymmetricQuantiser(bits=8)
W_q_pt = sym_qt.quantise(W_fp32, per_channel=False)
W_dq_pt = sym_qt.dequantise(W_q_pt)
err_pt  = np.abs(W_fp32 - W_dq_pt)

# Symmetric per-channel
sym_qc = SymmetricQuantiser(bits=8)
W_q_pc = sym_qc.quantise(W_fp32, per_channel=True)
W_dq_pc = sym_qc.dequantise(W_q_pc)
err_pc  = np.abs(W_fp32 - W_dq_pc)

# Asymmetric
asym_q = AsymmetricQuantiser()
W_q_as = asym_q.quantise(W_fp32)
W_dq_as = asym_q.dequantise(W_q_as)
err_as  = np.abs(W_fp32 - W_dq_as)

print("  Quantisation error comparison:")
print(f"  {'Method':<30} | {'Scale':>10} | {'Max error':>11} | "
      f"{'Mean error':>12} | {'SQNR (dB)':>11}")
print(f"  {'─'*80}")

methods = [
    ("Symmetric per-tensor (INT8)",   sym_qt.scale.item(), err_pt),
    ("Symmetric per-channel (INT8)",  sym_qc.scale.mean().item(), err_pc),
    ("Asymmetric UINT8",              asym_q.scale, err_as),
]
for name, scale, err in methods:
    sqnr = 10 * np.log10(np.mean(W_fp32**2) / np.mean(err**2))
    print(f"  {name:<30} | {scale:>10.6f} | {err.max():>11.6f} | "
          f"{err.mean():>12.8f} | {sqnr:>10.2f}")

print()
print("  Per-channel quantisation has lower error because each channel's")
print("  own weight range is captured separately — avoiding one large range")
print("  from dominating the scale and wasting precision on outlier channels.")
print()

# INT4 group quantisation
print("  INT4 group quantisation (GPTQ/AWQ style):")
print()

def group_quantise_int4(W, group_size=128):
    """Per-group INT4 symmetric quantisation (as used in GPTQ/AWQ)."""
    rows, cols = W.shape
    n_groups   = cols // group_size
    scales     = []
    Q_grouped  = np.zeros_like(W, dtype=np.float32)
    W_dq       = np.zeros_like(W)

    for r in range(rows):
        for g in range(n_groups):
            start = g * group_size
            end   = start + group_size
            w_g   = W[r, start:end]
            s     = np.max(np.abs(w_g)) / 7.0   # 4-bit: qmax = 7
            Q_g   = np.round(w_g / s).clip(-8, 7)
            W_dq[r, start:end] = Q_g * s
            scales.append(s)

    return W_dq, np.array(scales)

W_large = np.random.randn(32, 512).astype(np.float32) * 0.1
W_dq_i4, scales_i4 = group_quantise_int4(W_large, group_size=128)
err_i4 = np.abs(W_large - W_dq_i4)

# Compare INT8 per-tensor vs INT4 group
sym_qt_large = SymmetricQuantiser(8)
W_q8  = sym_qt_large.quantise(W_large)
W_dq8 = sym_qt_large.dequantise(W_q8)
err_i8 = np.abs(W_large - W_dq8)

sqnr_i8 = 10 * np.log10(np.mean(W_large**2) / np.mean(err_i8**2))
sqnr_i4 = 10 * np.log10(np.mean(W_large**2) / np.mean(err_i4**2))

print(f"  INT8 per-tensor:       SQNR = {sqnr_i8:.2f} dB  (max err = {err_i8.max():.4f})")
print(f"  INT4 group-128:        SQNR = {sqnr_i4:.2f} dB  (max err = {err_i4.max():.4f})")
print(f"  INT4 memory savings:   {32*512*2/1e6:.2f} MB → {32*512*0.5/1e6:.2f} MB (4× reduction)")
print()
print("  INT4 has ~6 dB less SQNR but 4× memory savings.")
print("  Group quantisation (group=128) recovers ~3 dB vs naive INT4.")
print("  This is why GPTQ and AWQ use group sizes of 64–128.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Batching effects on throughput
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Batching: latency vs throughput tradeoff")
print("━" * 65)
print()

try:
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print()

    # Build a simple transformer-like model
    class TransformerBlock(nn.Module):
        def __init__(self, d=512, heads=8, ff_mult=4):
            super().__init__()
            self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.ff    = nn.Sequential(
                nn.Linear(d, d * ff_mult), nn.GELU(),
                nn.Linear(d * ff_mult, d),
            )
        def forward(self, x):
            x = x + self.attn(x, x, x, need_weights=False)[0]
            x = self.norm1(x)
            x = x + self.ff(x)
            return self.norm2(x)

    class TinyTransformer(nn.Module):
        def __init__(self, n_layers=4, d=512, heads=8):
            super().__init__()
            self.blocks = nn.ModuleList([TransformerBlock(d, heads) for _ in range(n_layers)])
            self.head   = nn.Linear(d, 32000)   # vocab head
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.head(x[:, -1, :])   # last token logits

    model = TinyTransformer(n_layers=4, d=512).to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: 4-layer transformer, d=512, {total_params/1e6:.1f}M params")
    print()

    SEQ_LEN    = 32
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    REPS       = 50
    WARMUP     = 10

    if device == "cuda":
        # CUDA timing with events (accurate GPU timing)
        def benchmark_batch(batch_size, reps, warmup):
            x = torch.randn(batch_size, SEQ_LEN, 512, device=device)
            with torch.no_grad():
                for _ in range(warmup): model(x)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                for _ in range(reps): model(x)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / reps   # ms
    else:
        # CPU timing
        def benchmark_batch(batch_size, reps, warmup):
            x = torch.randn(batch_size, SEQ_LEN, 512, device=device)
            with torch.no_grad():
                for _ in range(warmup): model(x)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(reps): model(x)
            return (time.perf_counter() - t0) / reps * 1000   # ms

    print(f"  Batching effects (seq_len={SEQ_LEN}, {REPS} iterations):")
    print()
    print(f"  {'Batch':>6} | {'Latency (ms)':>14} | {'Throughput (req/s)':>20} | "
          f"{'Tokens/s':>12} | {'vs batch=1':>11}")
    print(f"  {'─'*75}")

    base_throughput = None
    for bs in BATCH_SIZES:
        try:
            lat_ms    = benchmark_batch(bs, REPS, WARMUP)
            req_per_s = 1000 / lat_ms * bs   # batch_size requests per 1000ms
            tok_per_s = req_per_s * SEQ_LEN
            if base_throughput is None:
                base_throughput = req_per_s
            speedup = req_per_s / base_throughput
            print(f"  {bs:6d} | {lat_ms:14.3f} | {req_per_s:20.1f} | "
                  f"{tok_per_s:12.0f} | {speedup:10.2f}×")
        except RuntimeError:
            print(f"  {bs:6d} | OOM")
            break
    print()
    print("  Key observations:")
    print("  - Latency increases with batch (linearly past a point)")
    print("  - Throughput increases super-linearly at first (GPU fills up)")
    print("  - Throughput plateaus at large batch (compute bound)")
    print("  - The 'sweet spot' depends on your latency SLA")

except ImportError:
    print("  PyTorch not available — showing batching concept:")
    BATCHING_CONCEPT = """
  BATCHING TRADEOFF (simulated LLM decode, A100 80GB):

  Batch | Latency | Throughput | GPU util | Notes
  ──────┼─────────┼────────────┼──────────┼───────────────────────────
    1   |  15 ms  |   67 req/s |     <1%  | Deeply memory bound
    8   |  20 ms  |  400 req/s |     5%   | Still memory bound
   64   |  60 ms  | 1067 req/s |    40%   | Moving toward compute bound
  256   | 200 ms  | 1280 req/s |    90%+  | Compute bound, high util
  512   | 400 ms  | 1280 req/s |    95%   | No more throughput gain

  The inflection point where throughput plateaus ≈ the batch size
  where arithmetic intensity crosses the GPU's ridge point.

  With continuous batching (vLLM, TRT-LLM):
    Effective batch is always as large as the queue allows.
    Long requests don't stall new short requests.
    GPU utilisation stays consistently 70–95%.
"""
    print(BATCHING_CONCEPT)

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: torch.compile for inference speedup
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — torch.compile and inference optimisation")
print("━" * 65)
print()

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class MLP(nn.Module):
        def __init__(self, d=1024):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d*4), nn.GELU(),
                nn.Linear(d*4, d*4), nn.GELU(),
                nn.Linear(d*4, d),
            )
        def forward(self, x): return self.net(x)

    BATCH = 64
    DIM   = 1024
    x_ref = torch.randn(BATCH, DIM, device=device)

    model_eager    = MLP(DIM).to(device).eval()
    model_compiled = torch.compile(MLP(DIM).to(device).eval())

    # Copy weights to compiled model
    model_compiled.load_state_dict(model_eager.state_dict())

    def bench(m, x, n=200, warmup=20):
        with torch.no_grad():
            for _ in range(warmup): m(x)
        if device == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n): m(x)
            torch.cuda.synchronize()
        else:
            t0 = time.perf_counter()
            for _ in range(n): m(x)
        return (time.perf_counter() - t0) / n * 1000

    print("  Compiling model with torch.compile() ...")
    # Trigger compilation
    with torch.no_grad():
        _ = model_compiled(x_ref)
    print("  Compilation complete.")
    print()

    t_eager = bench(model_eager,    x_ref)
    t_comp  = bench(model_compiled, x_ref)

    # Verify outputs match
    with torch.no_grad():
        out_e = model_eager(x_ref)
        out_c = model_compiled(x_ref)
    match = torch.allclose(out_e, out_c, atol=1e-4)

    print(f"  MLP (d={DIM}, batch={BATCH}): 3 linear layers + GELU")
    print(f"    Eager:    {t_eager:.4f} ms")
    print(f"    Compiled: {t_comp:.4f} ms")
    print(f"    Speedup:  {t_eager/t_comp:.2f}×")
    print(f"    Outputs match: {match} ✅")
    print()
    print("  torch.compile fuses ops via TorchInductor → Triton kernels.")
    print("  Speedup is larger on GPU (3-5×) and for memory-bound models.")
    print("  On CPU: typically 1.5-2.5× via LLVM vectorisation.")
    print()

    # Mixed precision
    if device == "cuda":
        model_fp16 = MLP(DIM).to(device).to(torch.float16).eval()
        model_fp16.load_state_dict({k: v.half() for k, v in model_eager.state_dict().items()})
        x_fp16 = x_ref.half()
        t_fp16 = bench(model_fp16, x_fp16)
        print(f"  FP16 inference:  {t_fp16:.4f} ms  ({t_eager/t_fp16:.2f}× vs FP32 eager)")
        print(f"  Memory: FP32={sum(p.numel()*4 for p in model_eager.parameters())/1e6:.1f}MB  "
              f"FP16={sum(p.numel()*2 for p in model_fp16.parameters())/1e6:.1f}MB")

except ImportError:
    print("  PyTorch not available")
    COMPILE_REF = """
  torch.compile INFERENCE PATTERNS:

  import torch

  # Basic compilation
  model = MyModel().eval()
  model = torch.compile(model)       # default mode
  model = torch.compile(model, mode="reduce-overhead")  # minimize launch overhead
  model = torch.compile(model, mode="max-autotune")     # maximize speed (slow compile)

  # For LLM generation (dynamic shapes):
  model = torch.compile(model, dynamic=True)            # handles varying seq lengths

  # Mixed precision
  model = model.half()         # FP16 weights
  with torch.autocast("cuda"):  # FP16 activations
      output = model(input)

  # torch.compile + FlashAttention
  F.scaled_dot_product_attention(Q, K, V)  # auto-uses FlashAttention when compiled

  Typical speedups (A100 GPU):
    Small MLP (inference):     1.5–2×
    Large transformer:          2–3×
    LLM prefill:               1.5–2×
    LLM decode (memory bound): 1.1–1.3× (mostly memory bound, less benefit)
"""
    print(COMPILE_REF)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Production Inference — TensorRT, Multi-GPU & Monitoring": {
        "description": (
            "Production GPU inference workflows. "
            "TensorRT compilation from ONNX: build engine, run inference. "
            "Multi-GPU tensor parallelism simulation. "
            "Speculative decoding throughput estimation. "
            "GPU monitoring with nvidia-smi and PyTorch memory tools. "
            "Production metrics: TTFT, ITL, throughput calculation. "
            "Deployment decision guide."
        ),
        "language": "python",
        "code": '''
import numpy as np
import subprocess
import time

print("=" * 65)
print("  PRODUCTION INFERENCE — TensorRT, MULTI-GPU & MONITORING")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: TensorRT workflow reference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — TensorRT: build and run workflow")
print("━" * 65)
print()

TRT_WORKFLOW = """
  TENSORRT WORKFLOW (complete reference):

  ── Step 1: Export model to ONNX ───────────────────────────────────────
  import torch, onnx

  model = MyModel().eval().cuda()
  dummy = torch.randn(1, 3, 224, 224).cuda()

  torch.onnx.export(
      model, (dummy,), "model.onnx",
      opset_version=17,
      input_names=["images"],
      output_names=["logits"],
      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
  )

  ── Step 2: Build TensorRT Engine ──────────────────────────────────────
  import tensorrt as trt

  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder    = trt.Builder(TRT_LOGGER)
  network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser     = trt.OnnxParser(network, TRT_LOGGER)

  # Parse the ONNX file
  with open("model.onnx", "rb") as f:
      parser.parse(f.read())

  # Configure builder
  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

  # Enable FP16 precision (2× faster, <1% accuracy loss)
  config.set_flag(trt.BuilderFlag.FP16)

  # Enable INT8 with calibration
  # config.set_flag(trt.BuilderFlag.INT8)
  # config.int8_calibrator = MyCalibrator(calibration_data)

  # Dynamic shape profile (batch can be 1–64)
  profile = builder.create_optimization_profile()
  profile.set_shape("images",
      min=(1, 3, 224, 224),    # minimum
      opt=(8, 3, 224, 224),    # optimal (most common)
      max=(64, 3, 224, 224),   # maximum
  )
  config.add_optimization_profile(profile)

  # Build engine (SLOW — may take minutes to hours!)
  engine = builder.build_serialized_network(network, config)
  with open("model.engine", "wb") as f:
      f.write(engine)

  ── Step 3: Run inference ───────────────────────────────────────────────
  import tensorrt as trt
  import pycuda.driver as cuda
  import pycuda.autoinit

  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  runtime = trt.Runtime(TRT_LOGGER)

  # Load engine
  with open("model.engine", "rb") as f:
      engine = runtime.deserialize_cuda_engine(f.read())

  context = engine.create_execution_context()

  # Set dynamic batch size for this inference
  context.set_input_shape("images", (8, 3, 224, 224))

  # Allocate I/O buffers (GPU memory)
  input_np  = np.random.randn(8, 3, 224, 224).astype(np.float32)
  output_np = np.zeros((8, 1000), dtype=np.float32)

  d_input  = cuda.mem_alloc(input_np.nbytes)
  d_output = cuda.mem_alloc(output_np.nbytes)

  # Copy input to GPU
  cuda.memcpy_htod(d_input, input_np)

  # Run inference
  context.execute_v2([int(d_input), int(d_output)])

  # Copy output from GPU
  cuda.memcpy_dtoh(output_np, d_output)

  # Benchmark with CUDA streams
  stream  = cuda.Stream()
  start   = cuda.Event()
  end     = cuda.Event()

  start.record(stream)
  for _ in range(100):
      cuda.memcpy_htod_async(d_input, input_np, stream)
      context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
      cuda.memcpy_dtoh_async(output_np, d_output, stream)
  end.record(stream)
  stream.synchronize()
  print(f"  Average: {start.time_till(end)/100:.3f} ms per inference")
"""
print(TRT_WORKFLOW)

try:
    import tensorrt as trt
    print(f"  TensorRT version: {trt.__version__} ✅")
except ImportError:
    print("  TensorRT not installed (pip install tensorrt — requires CUDA)")
    print("  Install: pip install tensorrt nvidia-pyindex")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Multi-GPU tensor parallelism simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Tensor parallelism: communication overhead analysis")
print("━" * 65)
print()

def simulate_tensor_parallel(M, K, N, tp_degree,
                               compute_tflops_per_gpu=312.0,
                               nvlink_bw_gb=600.0):
    """
    Simulate tensor parallelism for a linear layer Y = X @ W.
    Each GPU computes a shard of W (column-parallel).
    After the local matmul, an AllReduce is needed to sum partial results.
    """
    # Compute time (each GPU handles W[:, K//tp : K//tp * (tp+1)])
    # Each GPU does M × K × (N/tp) multiply-adds = 2 × M × K × N / tp FLOPs
    flops_per_gpu = 2 * M * K * N / tp_degree
    compute_s     = flops_per_gpu / (compute_tflops_per_gpu * 1e12)

    # AllReduce communication (Ring-AllReduce algorithm)
    # Communicates 2 × (tp-1)/tp × output_size bytes
    output_bytes  = M * N * 2   # BF16 = 2 bytes per element
    allreduce_bytes = 2 * (tp_degree - 1) / tp_degree * output_bytes
    # NVLink bandwidth is shared among all links
    effective_bw  = nvlink_bw_gb * 1e9 / tp_degree   # simplified
    comm_s        = allreduce_bytes / (nvlink_bw_gb * 1e9)

    total_s       = compute_s + comm_s
    speedup       = (2 * M * K * N / (compute_tflops_per_gpu * 1e12)) / total_s

    return compute_s * 1000, comm_s * 1000, total_s * 1000, speedup


# Transformer attention QKV projection (LLaMA-2 70B-like)
# One attention layer: Linear(8192, 8192) in BF16
M_dim = 64   # batch × seq tokens in this step
K_dim = 8192
N_dim = 8192

print(f"  Layer: Linear({K_dim}→{N_dim}), batch tokens={M_dim}")
print(f"  Hardware: A100 SXM (312 TFLOPS BF16, NVLink 600 GB/s per GPU)")
print()
print(f"  {'TP degree':>10} | {'Compute (ms)':>14} | {'AllReduce (ms)':>16} | "
      f"{'Total (ms)':>12} | {'Speedup':>9} | {'Efficiency':>12}")
print(f"  {'─'*82}")

for tp in [1, 2, 4, 8]:
    c_ms, comm_ms, tot_ms, speedup = simulate_tensor_parallel(
        M_dim, K_dim, N_dim, tp_degree=tp)
    efficiency = speedup / tp * 100
    print(f"  {tp:>10} | {c_ms:>14.4f} | {comm_ms:>16.4f} | "
          f"{tot_ms:>12.4f} | {speedup:>9.2f}× | {efficiency:>10.1f}%")

print()
print("  Efficiency decreases with more GPUs because AllReduce overhead grows.")
print("  With NVLink: TP=4 maintains ~70-80% efficiency.")
print("  Without NVLink (PCIe): TP=4 may only achieve 30-50% efficiency.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Speculative decoding analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Speculative decoding: throughput estimation")
print("━" * 65)
print()

def speculative_decoding_speedup(
    alpha,              # acceptance rate per draft token
    K,                  # draft tokens per oracle call
    t_draft_ms,         # time for draft model to generate K tokens
    t_oracle_ms,        # time for oracle to verify K+1 tokens
):
    """
    Expected tokens per oracle call:
        E[accepted] = sum_{i=1}^{K} alpha^i × 1   (geometric series)
                    ≈ min(K, 1/(1-alpha))  (approximately)
    """
    expected_accepted = sum(alpha**i for i in range(1, K+1))
    # Add the oracle's own token (always accepted if all K draft tokens accepted)
    expected_total = expected_accepted + alpha**K  # +1 if all accepted (oracle picks)

    # Time per oracle step = draft time + oracle verification time
    t_per_oracle_call = t_draft_ms + t_oracle_ms

    # Baseline: oracle generates one token per step
    t_baseline_per_token = t_oracle_ms

    # Speculative: generates ~expected_total tokens per step
    t_spec_per_token = t_per_oracle_call / expected_total

    speedup = t_baseline_per_token / t_spec_per_token
    return expected_total, speedup


print("  Speculative decoding speedup analysis:")
print("  (Oracle: LLaMA-2 70B, Draft: LLaMA-2 7B)")
print()
print("  Timing assumptions (A100 80GB, batch=1):")
print("    Oracle (70B, 1 token):  ~80 ms")
print("    Draft  (7B,  K tokens): ~15 ms per token (but processed together)")
print()

# Oracle time to process K+1 tokens (prefill-like, memory bound)
T_ORACLE = 80   # ms for one oracle forward pass (verifies K+1 tokens — nearly same cost as 1 token)
T_DRAFT_PER_TOKEN = 15  # ms for draft model per token

print(f"  {'K (draft)':>10} | {'α (accept)':>12} | {'E[tokens]':>11} | "
      f"{'Speedup':>9} | {'Effective tokens/s'}")
print(f"  {'─'*67}")
for K in [2, 4, 6, 8]:
    for alpha in [0.7, 0.8, 0.9]:
        t_draft = K * T_DRAFT_PER_TOKEN   # draft generates K tokens sequentially
        exp_tok, speedup = speculative_decoding_speedup(alpha, K, t_draft, T_ORACLE)
        baseline_tok_s = 1000 / T_ORACLE     # tokens/sec baseline
        spec_tok_s     = baseline_tok_s * speedup
        print(f"  {K:>10} | {alpha:>12.1f} | {exp_tok:>11.2f} | "
              f"{speedup:>9.2f}× | {spec_tok_s:>10.1f} tok/s")
    print()

print("  Key insight: acceptance rate α is task-dependent.")
print("    Code generation:    α ≈ 0.9 (draft model very aligned with oracle)")
print("    Factual Q&A:        α ≈ 0.8")
print("    Creative writing:   α ≈ 0.6 (high diversity, draft less useful)")
print("    K=4–5 is usually the sweet spot (diminishing returns at K>6)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: GPU monitoring commands and metrics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — GPU monitoring and production metrics")
print("━" * 65)
print()

MONITORING_GUIDE = """
  GPU MONITORING COMMANDS:

  nvidia-smi                           Live GPU status (all GPUs)
  nvidia-smi -l 1                      Refresh every 1 second
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu \\
             --format=csv,noheader     CSV format for scripting
  nvidia-smi dmon                      Device monitoring (streaming, compact)
  nvidia-smi pmon                      Process monitoring (GPU per process)
  nvidia-smi nvlink --status           NVLink status and bandwidth
  nvidia-smi topo --matrix             GPU topology and connectivity

  DCGM (Data Center GPU Manager):
  dcgmi dmon -e 1004,1005,200,201      Monitor GPU/memory utilisation + power
  dcgmi health --check                  GPU health check (good for alerts)

  PYTORCH MEMORY TOOLS:
  torch.cuda.memory_allocated()        Bytes currently allocated
  torch.cuda.memory_reserved()         Bytes in CUDA memory pool (may be higher)
  torch.cuda.max_memory_allocated()    Peak allocation since last reset
  torch.cuda.memory_summary()          Detailed breakdown by tensor size
  torch.cuda.reset_peak_memory_stats() Reset peak stats

  # Memory snapshot (shows allocation call stacks):
  torch.cuda.memory._record_memory_history()
  # ... run model ...
  snapshot = torch.cuda.memory._snapshot()
  torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")

  NSIGHT PROFILING:
  nsys profile --trace=cuda,cudnn,cublas \\
       -o my_profile python my_inference.py   Capture profile
  nsys stats my_profile.nsys-rep              Summary stats
  nsight-sys my_profile.nsys-rep              Open GUI

  ncu --target-processes all \\
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
                l1tex__t_bytes.sum python my_inference.py   Per-kernel metrics

  PRODUCTION METRICS TO TRACK:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Metric                    │ How to measure           │ Target        │
  ├──────────────────────────────────────────────────────────────────────┤
  │ TTFT (Time To First Token)│ t(first_token) - t(recv) │ < 1s (chat)  │
  │ ITL (Inter-Token Latency) │ avg time between tokens  │ < 50ms       │
  │ P99 TTFT                  │ 99th percentile TTFT     │ < 3s         │
  │ Throughput (tokens/s)     │ total_tokens / duration  │ maximize     │
  │ GPU utilisation           │ nvidia-smi utilization   │ > 70%        │
  │ GPU memory utilisation    │ memory.used/memory.total │ 80-90%       │
  │ Request queue depth       │ server queue length      │ < 100        │
  │ KV cache hit rate         │ server stats endpoint    │ > 50%        │
  │ Token rejection rate      │ requests throttled/total │ < 5%         │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(MONITORING_GUIDE)

# Live nvidia-smi if available
print("  Live GPU status (nvidia-smi):")
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
if result.returncode == 0:
    for i, line in enumerate(result.stdout.strip().split('\n')):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            print(f"    GPU {i}: {parts[0]}")
            print(f"           Memory:      {parts[2]} / {parts[1]} MB used")
            print(f"           Utilisation: {parts[3]}%")
            print(f"           Temperature: {parts[4]}°C")
else:
    print("    (nvidia-smi not available — no NVIDIA GPU detected)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Deployment decision guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — GPU inference deployment decision guide")
print("━" * 65)
print()

DECISION_GUIDE = """
  CHOOSING YOUR INFERENCE STACK:

  ┌──────────────────────────────────────────────────────────────────────┐
  │ Scenario                  │ Recommended Stack         │ Key Benefit  │
  ├──────────────────────────────────────────────────────────────────────┤
  │ LLM serving (NVIDIA)      │ vLLM or TRT-LLM           │ Throughput   │
  │ LLM + multi-call programs │ SGLang                    │ RadixAttn    │
  │ CNN/ViT inference (NVIDIA)│ TensorRT                  │ Max speed    │
  │ Cross-platform serving    │ ONNX Runtime              │ Portability  │
  │ Research → quick deploy   │ torch.compile             │ Ease of use  │
  │ Intel CPU/GPU             │ OpenVINO                  │ Intel optim  │
  │ Apple Silicon             │ Core ML / MPS             │ ANE access   │
  │ Mobile (Android/iOS)      │ TFLite / ExecuTorch       │ On-device    │
  │ Browser                   │ ONNX Runtime Web          │ No server    │
  │ Bare metal MCU            │ TVM MicroTVM              │ No OS needed │
  └──────────────────────────────────────────────────────────────────────┘

  LATENCY vs THROUGHPUT OPTIMISATION:

  Optimise for LATENCY:
    Use small batch sizes (batch=1 for interactive chat)
    Use speculative decoding (reduces tokens per oracle call)
    Use tensor parallelism (split model across GPUs for lower per-request time)
    Use shorter KV cache (sliding window attention)
    Pre-warm model: keep it loaded, avoid cold starts

  Optimise for THROUGHPUT:
    Use large batch sizes (continuous batching, max concurrency)
    Use quantisation (INT8/FP8 for 2-4× more tokens/s)
    Use data parallelism (multiple GPU replicas)
    Maximise KV cache size (serve longer sequences concurrently)
    Tune --max-num-seqs (vLLM) or batch_scheduler_policy

  QUANTISATION DECISION:
    BF16 → always use over FP32 (same accuracy, 2× faster, half memory)
    INT8 (W8A8, SmoothQuant): safe for most models (<1% accuracy loss)
    FP8: prefer on H100 (native FP8 Tensor Cores, best accuracy+speed)
    INT4 (GPTQ/AWQ, group=128): significant memory savings, ~1% loss
    INT4 weight-only (W4A16): good balance of speed and accuracy

  GPU MEMORY SIZING:
    Rule of thumb: model_gb × 1.2 (20% overhead for KV cache + activations)
    For large batches: model_gb × 2 (generous KV cache for concurrency)
    Monitor: nvidia-smi memory.used — should stay under 90% to avoid OOM

  SCALING RULE OF THUMB:
    Target throughput < 1 GPU capacity → 1 GPU + dynamic batching
    Target throughput > 1 GPU capacity → data parallel (N replicas)
    Model > 1 GPU VRAM → tensor parallel (split model)
    Model > 1 node → pipeline + tensor parallel
    Latency SLA < 200ms → avoid large batch sizes, use speculative decoding
"""
print(DECISION_GUIDE)

print("  PERFORMANCE TARGETS (ballpark, A100 80GB):")
print()
PERF_TABLE = [
    ("ResNet-50 (FP16)",        "BS=64",  "~9000 img/s",    "~7ms",   "TensorRT"),
    ("ViT-L/16 (FP16)",         "BS=32",  "~2000 img/s",    "~16ms",  "TensorRT"),
    ("BERT-Base (INT8)",         "BS=64",  "~10000 seq/s",   "~6ms",   "TRT or ORT"),
    ("LLaMA-2 7B (BF16)",       "BS=1",   "~50 tok/s",      "~20ms",  "vLLM"),
    ("LLaMA-2 7B (BF16)",       "BS=128", "~3000 tok/s",    "~800ms", "vLLM"),
    ("LLaMA-2 7B (INT8)",       "BS=128", "~5000 tok/s",    "~500ms", "TRT-LLM"),
    ("LLaMA-2 70B (BF16, 2GPU)","BS=32",  "~500 tok/s",     "~500ms", "TRT-LLM"),
    ("Stable Diff XL (FP16)",   "BS=1",   "~2 img/s",       "~500ms", "ORT/TRT"),
]
print(f"  {'Model':<30} | {'Config':>7} | {'Throughput':>14} | {'Latency':>9} | {'Runtime'}")
print(f"  {'─'*80}")
for model, cfg, thput, lat, runtime in PERF_TABLE:
    print(f"  {model:<30} | {cfg:>7} | {thput:>14} | {lat:>9} | {runtime}")
print()
print("  Note: actual numbers vary by GPU generation, batch size, sequence")
print("  length, quantisation level, and serving framework version.")
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