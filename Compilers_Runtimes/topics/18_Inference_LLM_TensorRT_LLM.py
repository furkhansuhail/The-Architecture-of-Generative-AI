"""
TensorRT-LLM — NVIDIA's Production LLM Inference Engine
=========================================================

TensorRT for large language models. Where the base TensorRT module covers
general neural network optimisation, TRT-LLM is purpose-built for the
autoregressive generation loop that makes LLMs unique: tokens generated
one at a time, each attending back to everything before it.

Every design decision in TRT-LLM — in-flight batching, KV cache management,
tensor parallelism, FP8 quantisation — exists because that generation loop
has different bottlenecks from anything in computer vision or classical DL.
Understanding those bottlenecks is understanding TRT-LLM.

"""

import textwrap
import re

TOPIC_NAME   = "TensorRT-LLM — NVIDIA Production LLM Inference"
DISPLAY_NAME = "18 · TensorRT-LLM"
ICON         = "🚀"
SUBTITLE     = "Autoregressive Inference at NVIDIA GPU Speed"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY LLM INFERENCE IS DIFFERENT

### The Autoregressive Generation Problem

    Standard neural network inference (classification, detection):
        INPUT → [model] → OUTPUT
        One forward pass, done. Batch size N, run N samples in parallel.

    LLM text generation (autoregressive):
        "The cat" → [model] → "sat"
        "The cat sat" → [model] → "on"
        "The cat sat on" → [model] → "the"
        ...repeat until <EOS> token

    This creates a fundamental problem:
        - EACH token requires a FULL forward pass through the model
        - A 128-token response = 128 sequential forward passes
        - Each pass attends to ALL previous tokens (attention is O(n²))
        - You cannot run these passes in parallel (each depends on the last)

    This is why LLM inference is slow, expensive, and hard to optimise.
    Everything in TRT-LLM is an attempt to fight this constraint.

### Two Distinct Phases of LLM Inference

    PREFILL (prompt processing):
        Input:   the user's prompt (e.g., 512 tokens)
        Action:  process ALL prompt tokens simultaneously in one forward pass
        Output:  hidden states for all prompt tokens + KV cache populated
        Compute: highly parallel — matrix multiply dominated (compute-bound)
        Bottleneck: GPU FLOPS

    DECODE (token generation):
        Input:   the LAST generated token (just 1 token)
        Action:  run one forward pass, attend to all prior tokens via KV cache
        Output:  next token probability distribution
        Compute: minimal per step — mostly reading the KV cache (memory-bound)
        Bottleneck: GPU memory BANDWIDTH, not compute

    This split matters enormously for system design:
        Prefill: you want maximum throughput → batch many requests together
        Decode:  you want minimum latency   → minimise memory reads per step

    Diagram — The two phases for "Paris is the capital of France":

    Prefill (one parallel pass):
    [Paris] [is] [the] [capital] [of] → all processed simultaneously
                                        KV cache filled for all 5 tokens

    Decode (sequential passes):
    [of] → [France]    KV cache: 6 tokens, read all to attend
    [France] → [.]     KV cache: 7 tokens, read all to attend
    [.] → <EOS>        KV cache: 8 tokens, read all to attend


##### PART 2 — THE KV CACHE: THE CENTRAL DATA STRUCTURE

### What is the KV Cache?

    Every transformer layer computes attention:
        Q = x @ W_Q     (query)
        K = x @ W_K     (key)
        V = x @ W_V     (value)
        Attention = softmax(Q @ Kᵀ / √d) @ V

    During decode, when generating token t:
        - We need keys and values for tokens 0, 1, ..., t-1
        - We computed these ALREADY in previous decode steps
        - Re-computing them would be wasteful (pure wasted FLOPS)

    The KV cache stores every layer's K and V tensors from previous steps:
        On step 1: cache = {K₀, V₀}
        On step 2: cache = {K₀, V₀, K₁, V₁}
        On step t: cache = {K₀...Kₜ₋₁, V₀...Vₜ₋₁}

    At each decode step:
        1. Compute Q, K, V for the NEW token only
        2. Append new K, V to the cache
        3. Run attention: Q_new @ [K_cache]ᵀ → attends to ALL previous tokens

### KV Cache Memory Cost

    For a single request with S tokens in a model with:
        L layers,  H attention heads,  d head dimension

    KV cache size = 2 × L × H × d × S × bytes_per_element

    Example: LLaMA-2 70B, sequence length 4096, FP16:
        L=80, H=64, d=128, S=4096, bytes=2
        = 2 × 80 × 64 × 128 × 4096 × 2 bytes
        = 2 × 80 × 64 × 128 × 4096 × 2
        ≈ 21.5 GB for ONE request at full context

    For a batch of B requests: B × 21.5 GB
    This is why serving LLMs at scale requires careful memory management.

### KV Cache — The Memory vs Throughput Tension

    More GPU memory → larger KV cache → more concurrent requests
    Larger model → less memory for KV cache → fewer concurrent requests

    ┌────────────────────────────────────────────────────────┐
    │  Total GPU VRAM = Model Weights + KV Cache + Workspace │
    │                                                        │
    │  A100 80 GB:                                           │
    │    LLaMA-2 70B FP16:   ~140 GB  (needs 2×A100!)        │
    │    LLaMA-2 13B FP16:   ~26 GB   → ~54 GB for KV cache  │
    │    LLaMA-2 7B  FP16:   ~14 GB   → ~66 GB for KV cache  │
    │                                                        │
    │  More concurrency = more KV cache = more throughput    │
    └────────────────────────────────────────────────────────┘


##### PART 3 — IN-FLIGHT BATCHING (CONTINUOUS BATCHING)

### The Problem with Naive Static Batching

    Static batching: collect N requests, run them all together, return all results.

    Problem: requests have DIFFERENT output lengths.

    Diagram — Static batching timeline (4 requests, padded to max length):

    Request A: ████████████████████████████████ (long)
    Request B: ████████──────────────────────── (short, done early)
    Request C: ████████████────────────────────
    Request D: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (very short, WAITING)

    B, C, D finish early but MUST WAIT for A.
    GPU is running wasted compute on padding tokens.
    New requests E, F, G must WAIT for the whole batch to finish.

### In-Flight Batching (Continuous Batching)

    Requests join and leave the batch DYNAMICALLY between decode steps.
    No waiting for a batch to fully complete.

    Diagram — In-flight batching:

    Step 1: [A, B, C, D]
    Step 2: [A, B, C, D]  ← D finishes, E enters
    Step 3: [A, B, C, E]  ← B finishes, F enters
    Step 4: [A, C, E, F]  ← C finishes, G enters
    Step 5: [A, E, F, G]
    ...

    GPU always has a full batch of active requests.
    No padding waste. No request waits for others to finish.

    Throughput improvement in practice: 2–10× over static batching,
    depending on output length variance in the request distribution.

### Iteration-Level vs Request-Level Scheduling

    Static batching:   schedule at REQUEST level (whole responses)
    In-flight batching: schedule at ITERATION level (one decode step)

    This is the conceptual shift that makes modern LLM serving work.
    TRT-LLM implements this natively; the scheduler runs every decode step.


##### PART 4 — TENSOR PARALLELISM AND PIPELINE PARALLELISM

### Why Parallelism is Mandatory for Large Models

    LLaMA-2 70B in FP16: 140 GB of weights.
    NVIDIA A100: 80 GB VRAM.
    → Model does NOT fit on one GPU. Parallelism is not optional.

    Even models that fit (7B = 14 GB) benefit from multi-GPU:
        Multiple GPUs = larger combined KV cache = more concurrent requests
        = higher throughput.

### Tensor Parallelism (TP) — Split Matrices Across GPUs

    Megatron-LM style parallelism. Split the weight matrices themselves.

    For a Linear layer Y = X @ W (W is [hidden, hidden]):
        GPU 0: Y₀ = X @ W[:, :hidden/2]    (first half of output)
        GPU 1: Y₁ = X @ W[:, hidden/2:]    (second half of output)
        AllReduce: Y = concat(Y₀, Y₁)      (communicate across GPUs)

    For attention (multi-head):
        GPU 0: compute heads 0...(H/2 - 1)
        GPU 1: compute heads H/2...(H - 1)
        Communication only after each layer (one AllReduce per layer)

    TP degree 2: split across 2 GPUs
    TP degree 4: split across 4 GPUs (LLaMA-2 70B typically uses TP=4 or TP=8)

    Communication cost: AllReduce after each layer
        NVLink:  600 GB/s bidirectional — fast enough to hide latency
        PCIe:    ~64 GB/s — slower; TP across nodes is expensive

    ┌────────────────────────────────────────────────────────┐
    │  RULE: Use NVLink (within a node) for tensor parallelism│
    │  Never span tensor parallel across nodes (too slow)     │
    └────────────────────────────────────────────────────────┘

### Pipeline Parallelism (PP) — Split Layers Across GPUs

    Different layers run on different GPUs. Model is split by depth.

    GPU 0: layers 0–19    (first 20 layers)
    GPU 1: layers 20–39   (last 20 layers)

    Execution: GPU0 processes a micro-batch → sends activations to GPU1
               GPU1 processes → sends to GPU2 → ...

    Bubble problem:
        While GPU1 processes batch 1, GPU0 is idle.
        Overcome by micro-batching: GPU0 starts batch 2 while GPU1 has batch 1.

    PP is for multi-NODE deployments (between machines, not within).
    TP handles within-node, PP handles across-node.

### Combined: Tensor × Pipeline Parallelism

    Production LLaMA-2 70B on 16 × A100 80GB:

        TP = 8 (8 GPUs per node, model split via NVLink)
        PP = 2 (2 nodes, pipeline across InfiniBand)

    TRT-LLM configuration:
        tensor_parallel_size  = 8
        pipeline_parallel_size = 2

    This is how GPT-4, Claude, and every major LLM deployment works at scale.


##### PART 5 — QUANTISATION FOR LLMs: FP8, INT8, GPTQ, AWQ

### Why Quantisation Matters More for LLMs Than CNNs

    A ResNet-50 has 25M parameters → 100 MB in FP32 → trivial on any GPU.
    LLaMA-2 70B has 70B parameters → 140 GB in FP16 → requires 2×A100.

    Quantisation from FP16 → INT4:
        140 GB → 35 GB → fits on ONE A100!
        Decode throughput: memory-bandwidth limited → 4× fewer bytes = 4× faster

### Quantisation Schemes in TRT-LLM

    W8A16 (SmoothQuant style):
        Weights stored in INT8, activations in FP16.
        Dequantise weights on-the-fly before matrix multiply.
        Memory: 2× smaller weights; Compute: still FP16 matmuls.
        Simple and widely supported.

    W4A16 (GPTQ / AWQ):
        Weights in INT4 (2 bits fewer), activations FP16.
        GPTQ: minimise layer-wise reconstruction error using Hessian info.
        AWQ:  protects the ~1% of weights that are "salient" (large activation channels).
        4× smaller model; noticeable accuracy drop vs W8A16 at small scale.

    FP8 (H100 native):
        8-bit floating point — 1 sign + 4 exponent + 3 mantissa.
        BOTH weights and activations in FP8 (W8A8).
        H100 Tensor Cores natively support FP8 → no dequantise overhead.
        ~2× throughput vs FP16 on H100; accuracy nearly identical to FP16.
        TRT-LLM's recommended mode on H100.

    ┌───────────────────────────────────────────────────────────────────┐
    │  Precision  │ Size/param │ Accuracy │ Speed    │ Hardware req.    │
    ├─────────────┼────────────┼──────────┼──────────┼──────────────────┤
    │  FP16       │  2 bytes   │ Reference│  1×      │ All modern GPUs  │
    │  W8A16      │  1 byte    │ ≈ FP16   │ ~1.5×    │ Ampere+          │
    │  FP8 W8A8   │  1 byte    │ ≈ FP16   │ ~2×      │ H100 only        │
    │  W4A16 AWQ  │  0.5 bytes │ -0.5–1%  │ ~2–3×    │ Ampere+          │
    │  W4A16 GPTQ │  0.5 bytes │ -0.5–2%  │ ~2–3×    │ Ampere+          │
    └───────────────────────────────────────────────────────────────────┘

### Activation Outliers — The LLM Quantisation Challenge

    In CNNs, weight distributions are smooth → INT8 quantisation is easy.
    In LLMs (especially OPT, BLOOM), some activation channels have
    MASSIVE outliers: values 100× larger than the median.

    Problem: quantising a tensor with outliers wastes precision.
        Range: [-100, 100]  → scale = 200/255 ≈ 0.78
        Small value 0.01    → quantised to 0    (complete loss of precision)

    Solutions:
        SmoothQuant: "migrate" difficulty from activations to weights
                     Divide activation by s, multiply weight by s
                     Activations become smoother → easier to quantise
        AWQ:         Find and protect the ~1% of weight channels
                     corresponding to large-activation input channels


##### PART 6 — ATTENTION OPTIMISATIONS: FLASH ATTENTION & MULTI-QUERY

### Standard Attention — The Memory Bottleneck

    Standard attention for sequence length N:
        1. S = Q @ Kᵀ              (N×N score matrix — written to HBM)
        2. S = softmax(S / √d)     (read S from HBM, write back)
        3. O = S @ V               (read S from HBM again)

    Memory traffic: O(N²) reads/writes to HBM for the N×N matrix.
    For N=8192: 8192² × 2 bytes × 2 (read+write) = 268 MB per layer
    For 80 layers: 21 GB of HBM traffic just for attention scores!

### Flash Attention — Tile and Fuse

    Key insight: we don't need to materialise the full N×N matrix.
    Compute softmax(Q @ Kᵀ) @ V in TILES that fit in SRAM (fast on-chip memory).

    Algorithm:
        Divide Q into tiles of size B_r
        For each Q-tile:
            Iterate over K, V tiles
            Compute partial attention in SRAM
            Accumulate with numerically stable running softmax
        Write only O (output) to HBM — N×N matrix never written

    Memory traffic: O(N) instead of O(N²) — massive reduction.
    Speed: 2–4× faster than standard attention on long sequences.
    Memory: O(N) instead of O(N²) — enables much longer context windows.

    TRT-LLM uses Flash Attention 2 (and Flash Attention 3 on H100)
    for both prefill and optimised variants for decode.

### Multi-Head vs Multi-Query vs Grouped-Query Attention

    Standard MHA (multi-head attention):
        Each head has its OWN K and V projection matrices.
        Memory for KV cache: num_heads × head_dim × sequence_len

    MQA (multi-query attention, Shazeer 2019):
        ALL heads SHARE a single K and V projection.
        KV cache memory: 1/num_heads of MHA.
        Decode is num_heads× faster (less KV cache memory to read).
        Accuracy slightly reduced.
        Used in: PaLM, Falcon, some Mistral variants.

    GQA (grouped-query attention, Ainslie 2023):
        Heads grouped into G groups; each group shares K, V.
        KV cache: G/num_heads fraction of MHA.
        G=1 → MQA.  G=num_heads → MHA.
        Balance between MHA accuracy and MQA speed.
        Used in: LLaMA-2 70B (G=8), Mistral 7B (G=8), Gemma, Qwen.

    ┌──────────────────────────────────────────────────────────┐
    │  For decode throughput: GQA >> MHA (fewer KV cache reads)│
    │  TRT-LLM supports all three natively.                    │
    └──────────────────────────────────────────────────────────┘


##### PART 7 — SPECULATIVE DECODING

### The Decode Bottleneck Is Not Compute

    During decode, the GPU is memory-bandwidth limited.
    A single decode step with batch size 1:
        Model weights: 14 GB (LLaMA-2 7B FP16) — all must be read
        Useful compute: ~0.1% of the GPU's peak FLOPS
        GPU utilisation: ~1–5%

    Adding more compute per step is FREE — we're not using it.
    Speculative decoding exploits this slack.

### Speculative Decoding Algorithm

    Idea: use a SMALL "draft" model to guess the next K tokens cheaply.
    Then verify ALL K guesses in ONE pass of the big "target" model.

    Since the big model must read all its weights once anyway for verification,
    verification of K tokens costs barely more than verification of 1 token.

    Algorithm (each step):
        1. DRAFT:  small model generates K token guesses [t₁, t₂, ..., tₖ]
        2. VERIFY: large model runs ONE forward pass, computing probabilities
                   for all K positions simultaneously
        3. ACCEPT/REJECT each guess using rejection sampling:
               Accept tᵢ if random(0,1) < p_target(tᵢ) / p_draft(tᵢ)
               Stop at first rejection, sample the correct token
        4. RESULT: at least 1 token accepted, up to K tokens accepted

    Speedup depends on ACCEPTANCE RATE:
        Acceptance rate α → expected tokens per step: (1-αᴷ⁺¹)/(1-α) / K
        α = 0.9, K = 4 → ~3.8 accepted per step → ~3.8× speedup!

    Draft models used in practice:
        Tiny LLMs of the same family (e.g., LLaMA-3.2 1B drafts LLaMA-3.1 70B)
        Medusa heads: multiple prediction heads on the target model itself
        Eagle: draft using target model's internal features (highest acceptance)

### When Speculative Decoding Helps

    ✅  Low concurrency (batch size 1–4): GPU is idle; draft cost is negligible
    ✅  Predictable output: code, structured formats, templates
    ❌  High concurrency: GPU is already busy; draft model adds overhead
    ❌  Creative/diverse generation: acceptance rate drops → little gain


##### PART 8 — TRT-LLM ARCHITECTURE AND WORKFLOW

### TRT-LLM vs Base TensorRT

    Base TensorRT (see TensorRT module):
        General neural network graphs
        User manually defines network or parses ONNX
        One input shape → one output shape

    TRT-LLM:
        Purpose-built for transformer decoder stacks
        Natively handles dynamic sequence lengths and KV cache
        Built-in implementations of all major LLM architectures
        In-flight batching scheduler included
        Multi-GPU tensor/pipeline parallelism built-in

### The TRT-LLM Build Workflow

    Step 1: Model source
        Load weights from HuggingFace Hub, or checkpoint files

    Step 2: Convert to TRT-LLM checkpoint format
        trtllm-build --model_dir ./llama-2-7b
                     --output_dir ./trtllm-engine
                     --dtype float16
                     --tp_size 1
                     --pp_size 1
                     --max_batch_size 32
                     --max_input_len 2048
                     --max_output_len 512

    Step 3: Engine built — serialised .engine files (per GPU)

    Step 4: Serve with Triton Inference Server (recommended) or Python runtime
        tritonserver --model-repository=./triton_model_repo

    Diagram — Full TRT-LLM deployment stack:

    ┌─────────────────────────────────────────────────────────────┐
    │  Client (HTTP/gRPC)                                         │
    ├─────────────────────────────────────────────────────────────┤
    │  NVIDIA Triton Inference Server                             │
    │  (routing, batching, health checks, metrics)                │
    ├─────────────────────────────────────────────────────────────┤
    │  TRT-LLM Executor (in-flight batching, KV cache manager)    │
    ├─────────────────────────────────────────────────────────────┤
    │  TRT-LLM Engine (compiled, GPU-specific, per-GPU .engine)   │
    ├─────────────────────────────────────────────────────────────┤
    │  NCCL (GPU-GPU communication for tensor parallelism)        │
    ├─────────────────────────────────────────────────────────────┤
    │  NVIDIA GPU (A100 / H100 / L40S)                            │
    └─────────────────────────────────────────────────────────────┘

### Key Configuration Parameters

    ┌────────────────────────────┬──────────────────────────────────────────┐
    │ Parameter                  │ What it controls                         │
    ├────────────────────────────┼──────────────────────────────────────────┤
    │ --dtype                    │ Weight precision (float16, bfloat16, fp8)│
    │ --tp_size                  │ Tensor parallel degree (GPUs per node)   │
    │ --pp_size                  │ Pipeline parallel degree (nodes)         │
    │ --max_batch_size           │ Max concurrent requests in scheduler     │
    │ --max_input_len            │ Maximum prompt token length              │
    │ --max_output_len           │ Maximum generation token length          │
    │ --kv_cache_free_gpu_mem_fraction │ % of free VRAM reserved for KV     │
    │ --enable_chunked_context   │ Chunked prefill for long prompts         │
    │ --use_paged_context_fmha   │ Paged KV cache (like vLLM) in prefill    │
    └────────────────────────────┴──────────────────────────────────────────┘

### When to Use TRT-LLM vs Alternatives

    ✅  Maximum throughput on NVIDIA hardware — TRT-LLM is fastest on A100/H100
    ✅  Production enterprise deployments with Triton
    ✅  When you need FP8 (H100) or INT8 W8A16 quantisation
    ✅  Controlled, locked-down NVIDIA environments
    ✅  Models: LLaMA, Mistral, Falcon, GPT, Gemma, Qwen, Phi, Mamba, Whisper
    ❌  Research / rapid iteration — compilation is slow (10–30 min per config)
    ❌  AMD GPUs — NVIDIA-only
    ❌  Frequent model updates — recompilation required on every weight change
    ❌  Complex multi-model orchestration — vLLM or SGLang are easier to compose

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · KV Cache — Memory Sizing and the Concurrency/VRAM Tradeoff": {
        "description": (
            "Calculate KV cache memory requirements for any model. "
            "Show how KV cache size scales with sequence length, model size, "
            "batch size, and precision. Derive maximum concurrent requests given "
            "available VRAM. Compare MHA vs GQA KV cache sizes."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  KV CACHE — MEMORY SIZING & CONCURRENCY TRADEOFF")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# Model configuration presets
# ─────────────────────────────────────────────────────────────────────────
MODELS = {
    "LLaMA-2 7B":  dict(layers=32,  heads=32,  kv_heads=32,  head_dim=128, param_b=7),
    "LLaMA-2 13B": dict(layers=40,  heads=40,  kv_heads=40,  head_dim=128, param_b=13),
    "LLaMA-2 70B": dict(layers=80,  heads=64,  kv_heads=8,   head_dim=128, param_b=70),
    "Mistral 7B":  dict(layers=32,  heads=32,  kv_heads=8,   head_dim=128, param_b=7),
    "Falcon 40B":  dict(layers=60,  heads=64,  kv_heads=1,   head_dim=64,  param_b=40),
    "GPT-3 175B":  dict(layers=96,  heads=96,  kv_heads=96,  head_dim=128, param_b=175),
}

GPUS = {
    "A10G   (24 GB)":  24,
    "A100   (40 GB)":  40,
    "A100   (80 GB)":  80,
    "H100   (80 GB)":  80,
    "H100  (141 GB)":  141,
}

BYTES_PER_ELEMENT = {
    "FP16 (2 bytes)": 2,
    "INT8 (1 byte)":  1,
    "FP8  (1 byte)":  1,
    "INT4 (0.5 bytes)": 0.5,
}

def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, bpe):
    """
    KV cache for ONE request.
    Factor 2: K and V tensors.
    """
    return 2 * layers * kv_heads * head_dim * seq_len * bpe

def model_weight_bytes(param_b, bpe):
    """Approximate weight size: param_billions × bytes_per_element."""
    return param_b * 1e9 * bpe

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: KV cache size for one request
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — KV cache per request (FP16, seq_len=4096)")
print("━" * 65)
print()

SEQ_LEN = 4096
BPE     = 2   # FP16

print(f"  KV cache = 2 × layers × kv_heads × head_dim × seq_len × bpe")
print(f"  seq_len={SEQ_LEN}, precision=FP16 (2 bytes/element)")
print()
print(f"  {'Model':<18} | {'Layers':>7} | {'KV heads':>9} | {'Attn type':>10} | {'KV cache/req':>13}")
print(f"  {'─'*65}")

for name, cfg in MODELS.items():
    kv_gb   = kv_cache_bytes(cfg['layers'], cfg['kv_heads'],
                              cfg['head_dim'], SEQ_LEN, BPE) / 1e9
    attn    = ("MHA" if cfg['kv_heads'] == cfg['heads'] else
               ("MQA" if cfg['kv_heads'] == 1 else f"GQA(g={cfg['kv_heads']})"))
    print(f"  {name:<18} | {cfg['layers']:7d} | {cfg['kv_heads']:9d} | "
          f"{attn:>10} | {kv_gb:11.3f} GB")

print()
print("  Observation: LLaMA-2 70B uses GQA(g=8) — 8× smaller KV cache than MHA")
print("  Mistral 7B uses GQA(g=8) — enables longer contexts on same hardware")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Max concurrent requests given VRAM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Max concurrent requests = (VRAM - weights) / KV_per_req")
print("━" * 65)
print()
print("  Formula: max_reqs = floor((VRAM_bytes × 0.85 - model_bytes) / kv_per_req)")
print("  (0.85: leave 15% for activations, workspace, CUDA overhead)")
print()

TARGET_SEQ = 2048   # assume 2048 tokens (input+output combined) per request

print(f"  seq_len per request = {TARGET_SEQ} tokens, precision = FP16")
print()

for gpu_name, vram_gb in GPUS.items():
    print(f"  GPU: {gpu_name}  ({vram_gb} GB VRAM)")
    print(f"  {'Model':<18} | {'Weights (GB)':>13} | {'KV/req (GB)':>12} | {'Max reqs':>9} | {'VRAM util':>10}")
    print(f"  {'─'*70}")
    for model_name, cfg in list(MODELS.items())[:4]:   # first 4 models
        w_gb    = model_weight_bytes(cfg['param_b'], BPE) / 1e9
        kv_gb   = kv_cache_bytes(cfg['layers'], cfg['kv_heads'],
                                  cfg['head_dim'], TARGET_SEQ, BPE) / 1e9
        avail   = vram_gb * 0.85 - w_gb
        if avail < 0:
            print(f"  {model_name:<18} | {w_gb:11.1f} GB | {'—':>12} | {'DOES NOT FIT':>9}")
            continue
        max_req  = int(avail / kv_gb)
        used_gb  = w_gb + max_req * kv_gb
        util_pct = used_gb / vram_gb * 100
        print(f"  {model_name:<18} | {w_gb:11.1f} GB | {kv_gb:10.3f} GB | "
              f"{max_req:9d} | {util_pct:9.1f}%")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Precision comparison — FP16 vs INT8 vs FP8 vs INT4
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Precision impact on capacity (LLaMA-2 7B, A100 80GB)")
print("━" * 65)
print()

cfg     = MODELS["LLaMA-2 7B"]
vram_gb = 80

print(f"  Model: LLaMA-2 7B on A100 80GB, seq_len={TARGET_SEQ}")
print()
print(f"  {'Precision':<22} | {'Weight (GB)':>12} | {'KV/req (GB)':>12} | "
      f"{'Max reqs':>9} | {'vs FP16':>8}")
print(f"  {'─'*72}")

ref_reqs = None
for prec_name, bpe in BYTES_PER_ELEMENT.items():
    w_gb    = model_weight_bytes(cfg['param_b'], bpe) / 1e9
    kv_gb   = kv_cache_bytes(cfg['layers'], cfg['kv_heads'],
                              cfg['head_dim'], TARGET_SEQ, bpe) / 1e9
    avail   = vram_gb * 0.85 - w_gb
    max_req = max(0, int(avail / kv_gb))
    if ref_reqs is None:
        ref_reqs = max_req
    ratio = f"{max_req/ref_reqs:.2f}×" if ref_reqs > 0 else "—"
    print(f"  {prec_name:<22} | {w_gb:12.2f} | {kv_gb:12.3f} | {max_req:9d} | {ratio:>8}")

print()
print("  INT4 quantisation (GPTQ/AWQ) enables ~4× more concurrent requests")
print("  than FP16 on the same GPU — critical for cost-efficient serving.")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: KV cache growth per decode step
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 4 — KV cache grows each decode step (LLaMA-2 7B)")
print("━" * 65)
print()
print("  KV cache is NOT static — it grows by one 'row' per generated token.")
print()

cfg     = MODELS["LLaMA-2 7B"]
tokens  = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

print(f"  {'Tokens in ctx':>14} | {'KV cache (FP16)':>16} | {'KV cache (INT8)':>16} | {'% of 80GB':>10}")
print(f"  {'─'*60}")
for t in tokens:
    kv_fp16 = kv_cache_bytes(cfg['layers'], cfg['kv_heads'], cfg['head_dim'], t, 2) / 1e9
    kv_int8 = kv_cache_bytes(cfg['layers'], cfg['kv_heads'], cfg['head_dim'], t, 1) / 1e9
    pct     = kv_fp16 / 80 * 100
    print(f"  {t:14,} | {kv_fp16:14.4f} GB | {kv_int8:14.4f} GB | {pct:9.2f}%")

print()
print("  At 8192 tokens, the KV cache for LLaMA-2 7B is ~16 GB (FP16).")
print("  This is why long-context models are so expensive to serve.")
print("  GQA reduces this proportionally to kv_heads/heads.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · In-Flight Batching — Simulate the Continuous Batching Scheduler": {
        "description": (
            "Simulate static batching vs in-flight (continuous) batching side by side. "
            "Generate requests with random arrival times and output lengths. "
            "Show GPU utilisation, time-to-first-token, and throughput for each strategy. "
            "Demonstrate why in-flight batching dominates modern LLM serving."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import deque

print("=" * 65)
print("  IN-FLIGHT BATCHING — STATIC vs CONTINUOUS BATCHING SIMULATOR")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────────────────────────────────

class Request:
    """Simulates one LLM inference request."""
    def __init__(self, req_id, arrival_time, output_len):
        self.req_id       = req_id
        self.arrival      = arrival_time
        self.output_len   = output_len     # total tokens to generate
        self.tokens_done  = 0
        self.start_time   = None           # when first token generated (TTFT)
        self.finish_time  = None

    def is_done(self):
        return self.tokens_done >= self.output_len

    def __repr__(self):
        return f"Req({self.req_id}, len={self.output_len}, done={self.tokens_done})"


def generate_requests(n_requests, arrival_rate=0.8, mean_output=100):
    """
    Generate requests with Poisson arrivals and log-normal output lengths.
    arrival_rate: requests per time step
    mean_output:  mean output tokens
    """
    requests = []
    arrival_time = 0
    for i in range(n_requests):
        # Poisson inter-arrival (exponential gaps)
        gap         = np.random.exponential(1.0 / arrival_rate)
        arrival_time += max(1, int(gap))
        # Log-normal output lengths: heavy tail (some very long, most short)
        output_len  = max(10, int(np.random.lognormal(
            mean=np.log(mean_output) - 0.5, sigma=1.0)))
        output_len  = min(output_len, 512)
        requests.append(Request(i, arrival_time, output_len))
    return requests


# ─────────────────────────────────────────────────────────────────────────
# STRATEGY 1: Static (Naive) Batching
# ─────────────────────────────────────────────────────────────────────────
def simulate_static_batching(requests, batch_size=8, prefill_steps=5,
                              decode_steps_per_token=1):
    """
    Collect up to batch_size requests, run them together until ALL finish,
    then collect the next batch. Classic batching strategy.
    """
    queue        = sorted(requests, key=lambda r: r.arrival)
    waiting      = deque(queue)
    completed    = []
    step         = 0
    current_batch = []

    metrics = dict(ttfts=[], throughputs=[], gpu_util_steps=[])

    while waiting or current_batch:
        step += 1

        # ── Fill new batch if empty ───────────────────────────────────────
        if not current_batch:
            while waiting and len(current_batch) < batch_size:
                req = waiting[0]
                if req.arrival <= step:
                    current_batch.append(waiting.popleft())
                else:
                    break
            if not current_batch:
                step = waiting[0].arrival if waiting else step + 1
                continue

            # Prefill phase
            for req in current_batch:
                req.start_time = step + prefill_steps
            step += prefill_steps

        # ── Decode one token for all requests in batch ────────────────────
        active = [r for r in current_batch if not r.is_done()]
        gpu_util = len(active) / batch_size   # how full is the batch?
        metrics['gpu_util_steps'].append(gpu_util)

        for req in active:
            req.tokens_done += 1

        # Remove finished requests
        finished = [r for r in current_batch if r.is_done()]
        for req in finished:
            req.finish_time = step
            completed.append(req)
            metrics['ttfts'].append(req.start_time - req.arrival)

        current_batch = [r for r in current_batch if not r.is_done()]
        step += decode_steps_per_token

        # Wait for ALL in batch to finish before taking new ones
        # (This is the static batching constraint)

    total_time = step
    metrics['total_time']  = total_time
    metrics['completed']   = completed
    metrics['throughput']  = len(completed) / total_time * 1000
    metrics['mean_gpu_util'] = np.mean(metrics['gpu_util_steps']) if metrics['gpu_util_steps'] else 0
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# STRATEGY 2: In-Flight (Continuous) Batching
# ─────────────────────────────────────────────────────────────────────────
def simulate_inflight_batching(requests, batch_size=8, prefill_steps=5,
                                decode_steps_per_token=1):
    """
    Each decode step: remove finished requests, add waiting ones up to batch_size.
    GPU is ALWAYS processing a full batch (if requests are available).
    """
    queue     = sorted(requests, key=lambda r: r.arrival)
    waiting   = deque(queue)
    active    = []
    completed = []
    step      = 0

    metrics = dict(ttfts=[], gpu_util_steps=[])

    while waiting or active:
        step += 1

        # ── Admit new requests if we have capacity ────────────────────────
        while waiting and len(active) < batch_size:
            req = waiting[0]
            if req.arrival <= step:
                req = waiting.popleft()
                req.start_time = step + prefill_steps  # first token after prefill
                active.append(req)
            else:
                break

        if not active:
            step = waiting[0].arrival if waiting else step + 1
            continue

        # ── Decode one step for all active requests ───────────────────────
        gpu_util = len(active) / batch_size
        metrics['gpu_util_steps'].append(gpu_util)

        for req in active:
            req.tokens_done += 1

        # ── Remove finished requests (no waiting!) ────────────────────────
        finished = [r for r in active if r.is_done()]
        for req in finished:
            req.finish_time = step
            completed.append(req)
            metrics['ttfts'].append(req.start_time - req.arrival)

        active = [r for r in active if not r.is_done()]

    total_time = step
    metrics['total_time']  = total_time
    metrics['completed']   = completed
    metrics['throughput']  = len(completed) / total_time * 1000
    metrics['mean_gpu_util'] = np.mean(metrics['gpu_util_steps'])
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────────────────
N_REQUESTS  = 100
BATCH_SIZE  = 8
ARRIVAL_RATE = 0.5

requests_static   = generate_requests(N_REQUESTS, arrival_rate=ARRIVAL_RATE)
requests_inflight = [Request(r.req_id, r.arrival, r.output_len) for r in requests_static]

print(f"  Simulating {N_REQUESTS} requests, max batch size = {BATCH_SIZE}")
print(f"  Arrival rate: {ARRIVAL_RATE} req/step, mean output = 100 tokens (log-normal)")
print()

# Output length distribution
out_lens = [r.output_len for r in requests_static]
print(f"  Output length distribution:")
print(f"    Min:    {min(out_lens):4d} tokens")
print(f"    P25:    {int(np.percentile(out_lens, 25)):4d} tokens")
print(f"    Median: {int(np.percentile(out_lens, 50)):4d} tokens")
print(f"    P75:    {int(np.percentile(out_lens, 75)):4d} tokens")
print(f"    P95:    {int(np.percentile(out_lens, 95)):4d} tokens")
print(f"    Max:    {max(out_lens):4d} tokens")
print(f"    Std:    {np.std(out_lens):.1f}  (high variance → in-flight wins)")
print()

r_static   = simulate_static_batching(requests_static, batch_size=BATCH_SIZE)
r_inflight = simulate_inflight_batching(requests_inflight, batch_size=BATCH_SIZE)

# ─────────────────────────────────────────────────────────────────────────
# Results comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  RESULTS")
print("━" * 65)
print()

ttft_s  = r_static['ttfts']
ttft_i  = r_inflight['ttfts']

print(f"  {'Metric':<35} {'Static':>14} {'In-Flight':>14} {'Ratio':>8}")
print(f"  {'─'*75}")
print(f"  {'Total simulation time (steps)':<35} {r_static['total_time']:>14,} "
      f"{r_inflight['total_time']:>14,} "
      f"{r_static['total_time']/r_inflight['total_time']:>7.2f}×")
print(f"  {'Throughput (reqs/1000 steps)':<35} {r_static['throughput']:>14.2f} "
      f"{r_inflight['throughput']:>14.2f} "
      f"{r_inflight['throughput']/r_static['throughput']:>7.2f}×")
print(f"  {'Mean GPU batch utilisation':<35} {r_static['mean_gpu_util']*100:>13.1f}% "
      f"{r_inflight['mean_gpu_util']*100:>13.1f}%")
print()
print(f"  Time-to-first-token (TTFT) — latency from arrival to first token:")
print(f"  {'TTFT metric':<35} {'Static':>14} {'In-Flight':>14}")
print(f"  {'─'*65}")
for label, pct in [("Mean TTFT", 50), ("P50 TTFT", 50), ("P95 TTFT", 95), ("P99 TTFT", 99)]:
    s = np.percentile(ttft_s, pct) if ttft_s else 0
    i = np.percentile(ttft_i, pct) if ttft_i else 0
    print(f"  {label:<35} {s:>14.1f} {i:>14.1f}")

print()
print("  HOW TO READ THESE RESULTS:")
print()
print("  Static batching GPU util < 100%:")
print("    Short requests finish early → GPU runs wasted padding compute")
print("    while waiting for the longest request in the batch to finish.")
print()
print("  In-flight batching GPU util ≈ 100%:")
print("    When any request finishes, a new one immediately fills its slot.")
print("    GPU never waits. Full batch, every step.")
print()
print("  In-flight batching TTFT improvement:")
print("    Static:  new request waits for ENTIRE current batch to finish.")
print("    In-flight: request admitted at NEXT decode step.")
print("    Result: P99 TTFT drops dramatically — tail latency is the win.")
print()
print("  TRT-LLM's in-flight batching scheduler runs at EVERY decode step.")
print("  This is one of its core advantages over naive batch execution.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Tensor Parallelism — Sharding Strategy and Communication Cost": {
        "description": (
            "Implement Megatron-LM style tensor parallelism for a transformer layer. "
            "Show column-parallel and row-parallel linear layers. "
            "Measure computation vs communication time for different TP degrees. "
            "Calculate the theoretical speedup and communication bottleneck on PCIe vs NVLink."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  TENSOR PARALLELISM — SHARDING STRATEGY & COMMUNICATION COST")
print("=" * 65)
print()

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: MEGATRON-LM COLUMN/ROW PARALLELISM — THEORY
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Megatron-LM Tensor Parallelism (theory)")
print("━" * 65)
print()

print("  Transformer MLP block: Y = GeLU(X @ W1) @ W2")
print()
print("  Standard (single GPU):")
print("    X: [batch, hidden]    W1: [hidden, ffn]    W2: [ffn, hidden]")
print()
print("  Column-parallel split of W1 (split along output dim):")
print("    GPU0: W1[:, :ffn/2]  →  Y0 = X @ W1[:,  :ffn/2]   (first half of GeLU output)")
print("    GPU1: W1[:, ffn/2:]  →  Y1 = X @ W1[:, ffn/2:]   (second half)")
print("    No communication needed here — X is replicated on both GPUs.")
print()
print("  Row-parallel split of W2 (split along input dim):")
print("    GPU0: W2[:ffn/2, :]  →  Z0 = Y0 @ W2[:ffn/2, :]  (partial output)")
print("    GPU1: W2[ffn/2:, :]  →  Z1 = Y1 @ W2[ffn/2:, :]  (partial output)")
print("    AllReduce: Z = Z0 + Z1   ← ONE communication per layer")
print()
print("  Key insight:")
print("    2 matmuls → 2 GPU-parallel matmuls + 1 AllReduce")
print("    If communication time << compute time → near-linear speedup")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: SIMULATE TP COMPUTE vs COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Compute time vs AllReduce communication time")
print("━" * 65)
print()

def matmul_time_ms(M, K, N, flops_per_sec):
    """Time for Y = X @ W where X: (M, K), W: (K, N)."""
    flops = 2 * M * K * N   # multiply-add = 2 ops
    return flops / flops_per_sec * 1000

def allreduce_time_ms(tensor_bytes, bandwidth_bytes_per_sec):
    """
    AllReduce latency for ring-allreduce.
    Ring AllReduce transmits 2 × (n-1)/n × data_size bytes per GPU.
    Approximate as 2 × data_size / bandwidth for simplicity.
    """
    return 2 * tensor_bytes / bandwidth_bytes_per_sec * 1000

# Hardware specs
HARDWARE = {
    "A100 NVLink  (600 GB/s)": dict(
        gpu_tflops=312e12,      # A100 FP16 Tensor Core TFLOPS
        bandwidth=600e9,        # NVLink4 bidirectional
    ),
    "A100 PCIe    (64 GB/s)":  dict(
        gpu_tflops=312e12,
        bandwidth=64e9,         # PCIe 4.0 x16
    ),
    "H100 NVLink  (900 GB/s)": dict(
        gpu_tflops=989e12,      # H100 FP16
        bandwidth=900e9,        # NVLink5
    ),
}

# LLaMA-2 70B MLP layer parameters
HIDDEN = 8192
FFN    = 28672    # ffn_dim = 3.5 × hidden in LLaMA-2 70B
BATCH  = 4        # typical decode batch size
SEQ    = 1        # single decode step (one token)

TP_DEGREES = [1, 2, 4, 8]

print(f"  Model: LLaMA-2 70B MLP layer (hidden={HIDDEN}, ffn={FFN})")
print(f"  Batch × seq: {BATCH} × {SEQ} tokens (decode step)")
print()

for hw_name, hw in HARDWARE.items():
    print(f"  Hardware: {hw_name}")
    print(f"  {'TP':>4} | {'Compute (ms)':>14} | {'AllReduce (ms)':>16} | "
          f"{'Total (ms)':>12} | {'Speedup':>8} | {'Comm %':>8}")
    print(f"  {'─'*70}")

    base_total = None
    for tp in TP_DEGREES:
        # Each GPU handles hidden/tp output columns for W1
        # and hidden/tp input rows for W2
        local_ffn = FFN // tp
        m         = BATCH * SEQ
        k1, n1    = HIDDEN, local_ffn    # W1 shard: (hidden, ffn/tp)
        k2, n2    = local_ffn, HIDDEN    # W2 shard: (ffn/tp, hidden)

        t_compute  = matmul_time_ms(m, k1, n1, hw['gpu_tflops'])
        t_compute += matmul_time_ms(m, k2, n2, hw['gpu_tflops'])

        # AllReduce on the output tensor (batch × hidden × float16 bytes)
        output_bytes = BATCH * SEQ * HIDDEN * 2   # float16
        t_comm       = allreduce_time_ms(output_bytes, hw['bandwidth'])

        t_total = t_compute + t_comm
        if base_total is None:
            base_total = t_total
        speedup  = base_total / t_total
        comm_pct = t_comm / t_total * 100

        print(f"  {tp:4d} | {t_compute:14.4f} | {t_comm:16.4f} | "
              f"{t_total:12.4f} | {speedup:8.2f}× | {comm_pct:7.1f}%")
    print()

print("  INTERPRETATION:")
print("  NVLink:  communication is tiny fraction of compute → near-linear TP speedup")
print("  PCIe:    AllReduce dominates at TP=8 → diminishing returns beyond TP=2 or 4")
print()
print("  RULE: Only use tensor parallelism within a node (NVLink).")
print("  Across nodes (InfiniBand ~200 GB/s), TP is too slow → use pipeline parallel.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: WEIGHT SHARDING — ACTUAL MEMORY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Memory per GPU at different TP degrees (LLaMA-2 70B)")
print("━" * 65)
print()

# LLaMA-2 70B model architecture
LLAMA_70B = dict(
    layers=80, hidden=8192, ffn=28672,
    heads=64, kv_heads=8, head_dim=128,
    vocab=32000, param_b=70
)

def llama_layer_params(cfg):
    """Parameters in one transformer layer."""
    h, f = cfg['hidden'], cfg['ffn']
    # Self-attention: Q, K, V projections + output
    # Note: K,V use kv_heads not heads (GQA)
    q_params = h * cfg['heads']    * cfg['head_dim']
    k_params = h * cfg['kv_heads'] * cfg['head_dim']
    v_params = h * cfg['kv_heads'] * cfg['head_dim']
    o_params = cfg['heads'] * cfg['head_dim'] * h
    attn     = q_params + k_params + v_params + o_params
    # MLP: gate, up, down projections (SwiGLU uses 3 matrices)
    mlp      = 3 * h * f
    # LayerNorm (tiny)
    ln       = 2 * 2 * h
    return attn + mlp + ln

total_params = (
    llama_layer_params(LLAMA_70B) * LLAMA_70B['layers']
    + LLAMA_70B['vocab'] * LLAMA_70B['hidden'] * 2  # embed + lm_head
)

print(f"  LLaMA-2 70B parameter count verification:")
print(f"    Calculated: {total_params/1e9:.1f}B params  (nominal: 70B)")
print()

print(f"  {'TP degree':>10} | {'GPUs needed':>12} | {'Params/GPU (B)':>15} | "
      f"{'Weights/GPU (FP16)':>20} | {'Fits in A100 80GB?':>20}")
print(f"  {'─'*85}")

for tp in [1, 2, 4, 8]:
    params_per_gpu = total_params / tp
    gb_per_gpu     = params_per_gpu * 2 / 1e9  # FP16
    fits           = "✅ YES" if gb_per_gpu <= 80 else "❌ NO"
    print(f"  {tp:10d} | {tp:12d} | {params_per_gpu/1e9:15.1f} | "
          f"{gb_per_gpu:18.1f} GB | {fits:>20}")

print()
print("  LLaMA-2 70B REQUIRES TP≥2 on A100 80GB (weights alone: 140GB).")
print("  TP=8 on 8×A100 80GB is the standard config for LLaMA-2 70B.")
print("  After accounting for KV cache and workspace, TP=4 on A100 80GB")
print("  leaves very little room → TP=8 preferred for serving.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Speculative Decoding — Theory, Simulation & Speedup Analysis": {
        "description": (
            "Implement speculative decoding from scratch. Simulate the draft/verify "
            "loop. Show how acceptance rate drives speedup. Analyse when speculative "
            "decoding helps vs hurts (batch size effect). Compare Medusa vs standard "
            "draft model approach on code vs creative text distributions."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import defaultdict

print("=" * 65)
print("  SPECULATIVE DECODING — SIMULATION & SPEEDUP ANALYSIS")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: CORE ALGORITHM
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The algorithm: draft → verify → accept/reject")
print("━" * 65)
print()

print("  Standard decode (1 token per target model call):")
print("    Step 1: target model reads ALL weights (memory bandwidth limited)")
print("    Result: 1 token")
print()
print("  Speculative decode (up to K tokens per target model call):")
print("    Step 1: draft model generates K candidate tokens cheaply")
print("    Step 2: target model verifies ALL K tokens in ONE forward pass")
print("    Step 3: accept tokens via rejection sampling until first mismatch")
print("    Result: 1 to K+1 tokens")
print()
print("  Cost of verification ≈ cost of generating 1 token with target model")
print("  (target must read its weights either way — marginal cost per extra token is tiny)")
print()

VOCAB_SIZE = 50257  # GPT-2 / LLaMA vocab

def draft_token(draft_probs: np.ndarray) -> int:
    """Sample from draft model distribution."""
    return np.random.choice(VOCAB_SIZE, p=draft_probs)

def target_verify(token: int, target_probs: np.ndarray,
                  draft_probs: np.ndarray) -> tuple[bool, int]:
    """
    Rejection sampling: accept with probability min(1, p_target/p_draft).
    If rejected, sample from adjusted target distribution.
    Returns (accepted, final_token)
    """
    acceptance_ratio = min(1.0, target_probs[token] / (draft_probs[token] + 1e-10))
    accepted = np.random.random() < acceptance_ratio
    if accepted:
        return True, token
    else:
        # Sample from the "corrected" distribution: p_target - p_draft (clipped to >=0)
        corrected = np.maximum(0, target_probs - draft_probs)
        corrected /= corrected.sum()
        return False, np.random.choice(VOCAB_SIZE, p=corrected)

def speculative_decode_step(draft_probs_seq, target_probs_seq, K=4):
    """
    One speculative decoding step.
    draft_probs_seq: K distributions from the draft model
    target_probs_seq: K+1 distributions from the target model (verified in parallel)
    Returns: list of accepted tokens (1 to K+1)
    """
    draft_tokens = [draft_token(draft_probs_seq[k]) for k in range(K)]

    accepted_tokens = []
    for k in range(K):
        accepted, token = target_verify(
            draft_tokens[k], target_probs_seq[k], draft_probs_seq[k]
        )
        accepted_tokens.append(token)
        if not accepted:
            return accepted_tokens   # stop at first rejection

    # All K accepted — sample bonus token from target distribution at K+1
    bonus = np.random.choice(VOCAB_SIZE, p=target_probs_seq[K])
    accepted_tokens.append(bonus)
    return accepted_tokens

# Demonstrate with a single step
K = 4
# Simulate: draft and target mostly agree (high acceptance scenario)
target_p = np.random.dirichlet(np.ones(VOCAB_SIZE) * 0.05)   # peaked distribution
draft_p  = target_p.copy()
draft_p += np.random.dirichlet(np.ones(VOCAB_SIZE) * 0.1) * 0.2
draft_p /= draft_p.sum()   # slightly perturbed (realistic draft)

step_tokens = speculative_decode_step(
    [draft_p] * K,
    [target_p] * (K + 1),
    K=K
)
print(f"  Demo (K={K}, high-acceptance scenario):")
print(f"  Draft tokens proposed:  {K}")
print(f"  Tokens accepted (incl. bonus): {len(step_tokens)}")
print(f"  Token IDs: {step_tokens}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: ACCEPTANCE RATE → EXPECTED SPEEDUP
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Acceptance rate α → expected speedup")
print("━" * 65)
print()

print("  Expected tokens per spec-decode step (K draft tokens):")
print("  E[accepted] = (1 - α^(K+1)) / (1 - α)")
print("  where α = per-token acceptance rate")
print()
print("  Speedup = E[accepted] / (cost_ratio)")
print("  cost_ratio = (draft_cost × K + target_cost) / target_cost")
print("  For tiny draft (1/10 target size): cost_ratio ≈ K×0.1 + 1")
print()

DRAFT_COST_RATIO = 0.1   # draft model ≈ 10% of target cost

print(f"  (Assuming draft model costs {DRAFT_COST_RATIO*100:.0f}% of target model)")
print()
print(f"  {'α':>6} | {'K=1':>8} | {'K=2':>8} | {'K=3':>8} | {'K=4':>8} | "
      f"{'K=6':>8} | {'Ideal (∞K)':>12}")
print(f"  {'─'*70}")

for alpha in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    row = [f"  {alpha:6.2f} |"]
    for K in [1, 2, 3, 4, 6]:
        if alpha < 1.0:
            expected_tokens = (1 - alpha**(K+1)) / (1 - alpha)
        else:
            expected_tokens = K + 1
        step_cost = K * DRAFT_COST_RATIO + 1.0   # K draft + 1 target verify
        speedup   = expected_tokens / step_cost
        row.append(f" {speedup:6.2f}× |")
    # Ideal: infinite K, no draft cost (theoretical max)
    ideal = 1.0 / (1.0 - alpha) if alpha < 1 else float('inf')
    row.append(f" {ideal:10.2f}×")
    print("".join(row))

print()
print("  KEY OBSERVATIONS:")
print("  α=0.7: even moderate acceptance gives ~1.8× speedup with K=4")
print("  α=0.9: excellent acceptance → ~3.2× speedup with K=4")
print("  α<0.5: speedup < 1× — speculative decoding HURTS performance")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: ACCEPTANCE RATE BY TASK TYPE
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Acceptance rate by task type (simulation)")
print("━" * 65)
print()

def simulate_acceptance_rate(n_samples, alpha):
    """
    Simulate the per-token acceptance rate for a given task.
    alpha represents the average agreement between draft and target.
    """
    rates = []
    for _ in range(n_samples):
        accepted = int(np.random.random() < alpha)
        rates.append(accepted)
    return np.mean(rates)

TASKS = {
    "Python code generation":     0.88,   # deterministic, formulaic → high α
    "SQL query generation":       0.91,   # very constrained vocabulary → very high α
    "Structured JSON output":     0.93,   # near-deterministic for common schemas
    "Technical documentation":    0.82,
    "Formal reasoning (chain-of-thought)": 0.78,
    "General conversation":       0.72,
    "Creative writing":           0.61,   # diverse vocabulary → low α
    "Poetry / lyrical text":      0.55,   # highly unpredictable → spec-decode hurts
}

N_SIM = 2000
K_VAL = 4

print(f"  K={K_VAL}, draft cost = {DRAFT_COST_RATIO*100:.0f}% of target, N={N_SIM} samples each")
print()
print(f"  {'Task':<40} | {'Est. α':>7} | {'Expected speedup':>17} | {'Verdict':>12}")
print(f"  {'─'*85}")

for task, alpha in TASKS.items():
    expected_tokens = (1 - alpha**(K_VAL+1)) / (1 - alpha)
    step_cost       = K_VAL * DRAFT_COST_RATIO + 1.0
    speedup         = expected_tokens / step_cost
    verdict = ("✅ Strong" if speedup >= 2.5 else
               "✅ Good"   if speedup >= 1.8 else
               "⚠️  Marginal" if speedup >= 1.2 else
               "❌ Avoid")
    print(f"  {task:<40} | {alpha:7.2f} | {speedup:15.2f}×   | {verdict}")

print()
print("  SQL/code/JSON → highest speedup (80–90% acceptance rate)")
print("  Poetry/creative → avoid speculative decoding (adds cost without gain)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: BATCH SIZE EFFECT
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Why batch size kills speculative decoding speedup")
print("━" * 65)
print()

print("  With batch size B:")
print("  - ALL B requests must use the SAME K draft tokens")
print("    (we can't run K different futures for B requests simultaneously)")
print("  - Acceptance drops if requests diverge → speedup collapses")
print("  - Draft model cost multiplies by B → dominates at high B")
print()

ALPHA_CODE = 0.88  # code generation

print(f"  Task: code generation (α≈{ALPHA_CODE}), K=4, draft=10% target cost")
print()
print(f"  {'Batch size':>12} | {'Effective α':>13} | {'Expected speedup':>17} | {'Notes'}")
print(f"  {'─'*70}")

batch_data = [
    (1,  0.88, ""),
    (2,  0.84, "slight divergence"),
    (4,  0.78, "different requests diverge"),
    (8,  0.68, ""),
    (16, 0.55, "high diversity"),
    (32, 0.45, "spec decode hurts here"),
]

for bs, eff_alpha, note in batch_data:
    expected_tokens = (1 - eff_alpha**(K_VAL+1)) / (1 - eff_alpha)
    # Draft cost per step scales with batch size, but less than linear
    # (still one target forward pass at larger batch)
    draft_overhead  = K_VAL * DRAFT_COST_RATIO * min(1.0, np.log(bs+1)/np.log(2)/5 + 0.5)
    step_cost       = draft_overhead + 1.0
    speedup         = expected_tokens / step_cost
    verdict = "✅" if speedup > 1.2 else "❌"
    print(f"  {bs:12d} | {eff_alpha:13.2f} | {speedup:15.2f}×   {verdict} | {note}")

print()
print("  CONCLUSION: Speculative decoding is for LOW CONCURRENCY serving.")
print("  Batch size 1–4: excellent speedup (chatbot, API with 1 user at a time)")
print("  Batch size 16+: often hurts — disable spec decoding above threshold")
print()
print("  TRT-LLM's speculative decoding implementation automatically")
print("  falls back to standard decoding when batch size exceeds threshold.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Quantisation for LLMs — INT8 SmoothQuant & AWQ Weight-Only": {
        "description": (
            "Understand why LLM activations are harder to quantise than CNN activations. "
            "Implement SmoothQuant migration factor. Simulate AWQ weight saliency scoring. "
            "Show the per-channel scale factor calculation. Measure activation outlier "
            "distributions for different model families."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  LLM QUANTISATION — SmoothQuant & AWQ")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: WHY LLM ACTIVATIONS ARE HARD TO QUANTISE
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Activation outliers: the LLM quantisation problem")
print("━" * 65)
print()

np.random.seed(0)
HIDDEN   = 512
N_TOKENS = 64

def generate_activations(model_type="cnn"):
    """
    Simulate activation distributions for different model types.
    CNNs: smooth, few outliers. LLMs: heavy-tail, channel-specific outliers.
    """
    X = np.random.randn(N_TOKENS, HIDDEN).astype(np.float32)
    if model_type == "cnn":
        # Standard CNN activations: roughly Gaussian, no extreme outliers
        return X * 2.0
    elif model_type == "llm_mild":
        # LLM (GPT-2 style): some channels with larger variance
        channel_scales = np.ones(HIDDEN)
        outlier_channels = np.random.choice(HIDDEN, size=10, replace=False)
        channel_scales[outlier_channels] = np.random.uniform(5, 15, size=10)
        return X * channel_scales
    elif model_type == "llm_severe":
        # LLM (OPT/BLOOM style): massive outliers in ~1% of channels
        channel_scales = np.ones(HIDDEN)
        outlier_channels = np.random.choice(HIDDEN, size=5, replace=False)
        channel_scales[outlier_channels] = np.random.uniform(80, 200, size=5)
        return X * channel_scales

def quantise_int8(X):
    """Simple MinMax INT8 quantisation (per-tensor)."""
    scale    = np.max(np.abs(X)) / 127.0
    X_int8   = np.clip(np.round(X / scale), -127, 127).astype(np.int8)
    X_dequant = X_int8.astype(np.float32) * scale
    return X_dequant, scale

def quantise_int8_per_channel(X):
    """Per-channel INT8 quantisation (more accurate for outlier channels)."""
    scales    = np.max(np.abs(X), axis=0, keepdims=True) / 127.0
    X_int8    = np.clip(np.round(X / (scales + 1e-8)), -127, 127).astype(np.int8)
    X_dequant = X_int8.astype(np.float32) * scales
    return X_dequant, scales.flatten()

def quantisation_error(X_orig, X_dequant):
    """Mean squared reconstruction error."""
    return float(np.mean((X_orig - X_dequant) ** 2))

print(f"  Activations: {N_TOKENS} tokens × {HIDDEN} hidden dims")
print()
print(f"  {'Model type':<22} | {'Max |X|':>10} | {'|X| P99':>10} | {'Outlier ratio':>14} | {'Description'}")
print(f"  {'─'*80}")

for mtype, desc in [("cnn", "Smooth, bounded"),
                     ("llm_mild", "Moderate channel outliers"),
                     ("llm_severe", "Massive channel outliers (OPT/BLOOM)")]:
    X    = generate_activations(mtype)
    pct99 = np.percentile(np.abs(X), 99)
    pct999 = np.percentile(np.abs(X), 99.9)
    outlier_ratio = np.mean(np.abs(X) > 10 * np.std(X))
    print(f"  {mtype:<22} | {np.max(np.abs(X)):10.2f} | {pct99:10.2f} | "
          f"{outlier_ratio*100:13.3f}% | {desc}")

print()
print("  Quantisation error comparison (per-tensor vs per-channel):")
print()
print(f"  {'Model type':<22} | {'Per-tensor MSE':>16} | {'Per-channel MSE':>17} | {'Improvement':>12}")
print(f"  {'─'*75}")

for mtype in ["cnn", "llm_mild", "llm_severe"]:
    X      = generate_activations(mtype)
    Xd_pt, _ = quantise_int8(X)
    Xd_pc, _ = quantise_int8_per_channel(X)
    err_pt = quantisation_error(X, Xd_pt)
    err_pc = quantisation_error(X, Xd_pc)
    improvement = err_pt / (err_pc + 1e-10)
    print(f"  {mtype:<22} | {err_pt:16.4f} | {err_pc:17.4f} | {improvement:10.2f}×")

print()
print("  Per-channel quantisation is essential for LLMs with outlier channels.")
print("  But even per-channel activation quantisation is hard when W8A8 is needed.")
print("  → SmoothQuant migrates the difficulty from activations to weights.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: SMOOTHQUANT — MIGRATE DIFFICULTY FROM ACTIVATIONS TO WEIGHTS
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — SmoothQuant (W8A8 with activation migration)")
print("━" * 65)
print()

print("  Key insight: Y = X @ W = (X / s) @ (W × s)")
print("               where s is a per-channel migration scale")
print("  Divide activations by s → smoother activations (easy to quantise)")
print("  Multiply weights by s   → harder weights (but weights are static!)")
print()
print("  s = max(|X|)^α / max(|W|)^(1-α)")
print("  α = migration strength: α=0 → no migration, α=1 → full migration")
print("  α=0.5 is the recommended default (SmoothQuant paper)")
print()

def compute_smoothquant_scales(X, W, alpha=0.5):
    """
    X: activations (N, hidden)   — smooth after dividing by s
    W: weights    (hidden, out)  — scale by s
    Returns per-channel scale s (length: hidden)
    """
    max_x = np.max(np.abs(X), axis=0)          # (hidden,)
    max_w = np.max(np.abs(W), axis=1)           # (hidden,) — per input channel
    s     = (max_x ** alpha) / (max_w ** (1 - alpha) + 1e-8)
    return s

def apply_smoothquant(X, W, s):
    """Apply SmoothQuant transformation."""
    X_smooth = X / s                  # activations become smoother
    W_scaled = W * s[:, np.newaxis]   # weights absorb the scale
    return X_smooth, W_scaled

np.random.seed(1)
X_test    = generate_activations("llm_severe")
W_test    = np.random.randn(HIDDEN, 256).astype(np.float32) * 0.02

# Reference output (FP32)
Y_ref = X_test @ W_test

# Without SmoothQuant (naive W8A8)
Xq_naive, _ = quantise_int8(X_test)
Wq_naive, _ = quantise_int8(W_test.T)
Y_naive     = Xq_naive @ Wq_naive.T.astype(np.float32)

# With SmoothQuant
s           = compute_smoothquant_scales(X_test, W_test.T, alpha=0.5)
X_smooth, W_scaled = apply_smoothquant(X_test, W_test, s)
Xq_smooth, _ = quantise_int8(X_smooth)
Wq_scaled, _ = quantise_int8(W_scaled.T)
Y_smooth    = (Xq_smooth @ Wq_scaled.T.astype(np.float32)) * np.outer(
    np.ones(X_test.shape[0]), s)   # rescale output

print(f"  Test: {X_test.shape[0]} tokens × {X_test.shape[1]} hidden → {W_test.shape[1]} output")
print()

print(f"  {'Method':<25} | {'Output MSE':>12} | {'vs FP32 err':>13} | {'Notes'}")
print(f"  {'─'*65}")

mse_naive  = quantisation_error(Y_ref, Y_naive)
mse_smooth = quantisation_error(Y_ref, Y_smooth)

print(f"  {'FP32 (reference)':<25} | {'—':>12} | {'—':>13} |")
print(f"  {'Naive W8A8 (no smooth)':<25} | {mse_naive:12.4f} | {mse_naive:13.4f} | Outliers crush accuracy")
print(f"  {'SmoothQuant W8A8':<25} | {mse_smooth:12.6f} | {mse_smooth:13.6f} | ~{mse_naive/mse_smooth:.0f}× better")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: AWQ — WEIGHT-ONLY QUANTISATION WITH SALIENCY SCORING
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — AWQ: Activation-aware Weight Quantisation")
print("━" * 65)
print()

print("  AWQ insight: not all weights are equally important.")
print("  Weights corresponding to LARGE activation channels matter more.")
print("  Protect the top ~1% of weight columns → nearly FP16 accuracy at INT4.")
print()
print("  AWQ algorithm:")
print("  1. Collect activation statistics from calibration data")
print("  2. Score each weight column by: importance = mean(|X_channel|) × max(|W_col|)")
print("  3. Find per-channel scale s that minimises quantisation error")
print("  4. Quantise W using per-group INT4 with these protective scales")
print()

np.random.seed(2)
X_calib = generate_activations("llm_severe")   # calibration activations
W_awq   = np.random.randn(HIDDEN, 128).astype(np.float32) * 0.02

def awq_saliency_score(X_calib, W):
    """
    Score each input channel by activation magnitude × weight sensitivity.
    High score → this channel is important → needs protection.
    """
    act_mag   = np.mean(np.abs(X_calib), axis=0)  # (hidden,) mean activation per channel
    weight_sensitivity = np.max(np.abs(W), axis=1)  # (hidden,) max weight per input channel
    return act_mag * weight_sensitivity

def awq_find_optimal_scales(X, W, n_grid=20, alpha=0.5):
    """
    Grid search for optimal per-channel scale s in [0, 1].
    s chosen to minimise: ||Y_fp32 - quant(W × s) / s × X||
    """
    Y_ref_awq   = X @ W
    act_mag     = np.mean(np.abs(X), axis=0)   # (hidden,)
    best_scales = np.ones(HIDDEN)
    best_err    = float('inf')

    for grid_val in np.linspace(0.1, 2.0, n_grid):
        # Try scaling all channels by this factor (simplified — real AWQ is per-channel)
        candidate_s = act_mag ** alpha * grid_val
        W_scaled_c  = W * candidate_s[:, np.newaxis]
        W_q, _      = quantise_int8(W_scaled_c.T)
        # Undo the scale in the output
        Y_approx    = (X / candidate_s) @ W_q.T.astype(np.float32) * candidate_s.mean()
        err         = quantisation_error(Y_ref_awq, Y_approx)
        if err < best_err:
            best_err    = err
            best_scales = candidate_s
    return best_scales, best_err

saliency = awq_saliency_score(X_calib, W_awq)
top_pct  = 1.0   # top 1% of channels
threshold = np.percentile(saliency, 100 - top_pct)
protected = np.sum(saliency > threshold)

print(f"  Calibration data: {X_calib.shape[0]} tokens × {X_calib.shape[1]} channels")
print(f"  Weight matrix: {W_awq.shape}")
print()
print(f"  Top {top_pct:.0f}% salient channels: {protected} of {HIDDEN}")
print(f"  Saliency score range: [{saliency.min():.4f}, {saliency.max():.4f}]")
print(f"  Salient channel threshold: {threshold:.4f}")
print()

# Compare naive INT8 vs AWQ
optimal_scales, awq_err = awq_find_optimal_scales(X_calib, W_awq)
Y_ref_a    = X_calib @ W_awq
Xq_a, _    = quantise_int8(X_calib)
Wq_a, _    = quantise_int8(W_awq.T)
naive_err  = quantisation_error(Y_ref_a, Xq_a @ Wq_a.T.astype(np.float32))

print(f"  Output error comparison:")
print(f"    Naive INT8 (no AWQ): MSE = {naive_err:.6f}")
print(f"    AWQ INT8:            MSE = {awq_err:.6f}  ({naive_err/awq_err:.1f}× improvement)")
print()
print("  QUANTISATION SCHEME DECISION GUIDE:")
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │ Scheme     │ Hardware │ Acc. loss │ Best for                │")
print("  ├─────────────────────────────────────────────────────────────┤")
print("  │ FP16/BF16  │ Any A100+│ None      │ Baseline, quality       │")
print("  │ SmoothQuant│ A100+    │ ~0%       │ W8A8 on A100 (fastest)  │")
print("  │ FP8 W8A8   │ H100 only│ ~0%       │ H100 production         │")
print("  │ AWQ W4A16  │ Any A100+│ ~0.5%     │ Memory-constrained      │")
print("  │ GPTQ W4A16 │ Any A100+│ ~0.5–1%   │ Offline quantisation    │")
print("  └─────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · TRT-LLM Config & Triton Deployment — Build Commands and Configs": {
        "description": (
            "Generate correct trtllm-build commands for common models and hardware. "
            "Show the Triton model repository structure for LLM serving. "
            "Walk through a complete deployment checklist: model conversion, "
            "engine build, Triton config, health check, and benchmark. "
            "Cover the complete production deployment workflow."
        ),
        "language": "python",
        "code": '''
import json, os

print("=" * 65)
print("  TRT-LLM CONFIG & TRITON DEPLOYMENT REFERENCE")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: trtllm-build command generator
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — trtllm-build command generator")
print("━" * 65)
print()

CONFIGS = [
    dict(
        model="LLaMA-2 7B",    hf_path="meta-llama/Llama-2-7b-chat-hf",
        dtype="float16",       tp=1, pp=1,
        gpu="A100 40GB",       max_batch=32, max_in=2048, max_out=1024,
        kv_frac=0.9,
    ),
    dict(
        model="LLaMA-2 13B",   hf_path="meta-llama/Llama-2-13b-chat-hf",
        dtype="float16",       tp=2, pp=1,
        gpu="A100 80GB ×2",    max_batch=16, max_in=2048, max_out=1024,
        kv_frac=0.85,
    ),
    dict(
        model="LLaMA-2 70B",   hf_path="meta-llama/Llama-2-70b-chat-hf",
        dtype="bfloat16",      tp=4, pp=1,
        gpu="A100 80GB ×4",    max_batch=8,  max_in=4096, max_out=2048,
        kv_frac=0.8,
    ),
    dict(
        model="LLaMA-3.1 70B", hf_path="meta-llama/Meta-Llama-3.1-70B-Instruct",
        dtype="float16",       tp=4, pp=1,
        gpu="H100 80GB ×4",    max_batch=16, max_in=8192, max_out=4096,
        kv_frac=0.85,
    ),
    dict(
        model="Mistral 7B",    hf_path="mistralai/Mistral-7B-Instruct-v0.2",
        dtype="float16",       tp=1, pp=1,
        gpu="A10G 24GB",       max_batch=16, max_in=4096, max_out=1024,
        kv_frac=0.85,
    ),
]

def gen_trtllm_commands(cfg):
    """Generate the full TRT-LLM build + convert command sequence."""
    out_dir   = f"./trtllm_engines/{cfg['model'].lower().replace(' ', '_').replace('.', '')}"
    ckpt_dir  = f"./trtllm_ckpts/{cfg['model'].lower().replace(' ', '_').replace('.', '')}"

    lines = []
    lines.append(f"# ── {cfg['model']} on {cfg['gpu']} ──────────────────────")
    lines.append("")
    lines.append("# Step 1: Convert HuggingFace checkpoint to TRT-LLM format")
    lines.append(f"python convert_checkpoint.py \\")
    lines.append(f"    --model_dir {cfg['hf_path']} \\")
    lines.append(f"    --output_dir {ckpt_dir} \\")
    lines.append(f"    --dtype {cfg['dtype']} \\")
    lines.append(f"    --tp_size {cfg['tp']} \\")
    lines.append(f"    --pp_size {cfg['pp']}")
    lines.append("")
    lines.append("# Step 2: Build the TensorRT engine")
    lines.append(f"trtllm-build \\")
    lines.append(f"    --checkpoint_dir {ckpt_dir} \\")
    lines.append(f"    --output_dir {out_dir} \\")
    lines.append(f"    --max_batch_size {cfg['max_batch']} \\")
    lines.append(f"    --max_input_len {cfg['max_in']} \\")
    lines.append(f"    --max_output_len {cfg['max_out']} \\")
    lines.append(f"    --kv_cache_free_gpu_mem_fraction {cfg['kv_frac']} \\")
    lines.append(f"    --use_inflight_batching \\")
    lines.append(f"    --paged_kv_cache enable \\")
    if cfg['tp'] > 1:
        lines.append(f"    --workers {cfg['tp']} \\")
    lines.append(f"    --gemm_plugin {cfg['dtype']} \\")
    lines.append(f"    --gpt_attention_plugin {cfg['dtype']}")
    lines.append("")
    lines.append(f"# Engine written to: {out_dir}/")
    lines.append(f"# Files: rank0.engine [rank1.engine ...] + config.json")
    return "\n".join(lines)

for cfg in CONFIGS:
    print(gen_trtllm_commands(cfg))
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Triton model repository structure
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Triton model repository structure")
print("━" * 65)
print()

print("""
  triton_model_repo/
  ├── ensemble/              ← orchestrates the 3 backends below
  │   ├── config.pbtxt
  │   └── 1/
  ├── preprocessing/         ← tokenise raw text → token IDs
  │   ├── config.pbtxt
  │   └── 1/
  │       └── model.py
  ├── tensorrt_llm/          ← the actual TRT-LLM engine
  │   ├── config.pbtxt
  │   └── 1/
  │       ├── rank0.engine
  │       └── config.json
  └── postprocessing/        ← token IDs → decoded text
      ├── config.pbtxt
      └── 1/
          └── model.py

  Request flow:
  Client → [preprocessing] → [tensorrt_llm] → [postprocessing] → Client
              (tokenise)         (generate)         (decode)
""")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Triton config.pbtxt for TRT-LLM backend
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — tensorrt_llm/config.pbtxt (annotated)")
print("━" * 65)
print()

CONFIG_PBTXT = 
"""
    name: "tensorrt_llm"
    backend: "tensorrtllm"
    max_batch_size: 32        # must match --max_batch_size from trtllm-build
    
    model_transaction_policy {
      decoupled: True          # streaming output (token by token)
    }
    
    dynamic_batching {         # in-flight batching configuration
      preferred_batch_size: [1, 2, 4, 8, 16, 32]
      max_queue_delay_microseconds: 5000   # wait up to 5ms to form a larger batch
    }
    
    parameters {
      key: "gpt_model_type"
      value: { string_value: "inflight_fused_batching" }
    }
    parameters {
      key: "gpt_model_path"
      value: { string_value: "/engines/llama-2-7b/" }   # path to rank*.engine
    }
    parameters {
      key: "max_tokens_in_paged_kv_cache"
      value: { string_value: "20000" }   # total KV tokens pooled across all requests
    }
    parameters {
      key: "kv_cache_free_gpu_mem_fraction"
      value: { string_value: "0.9" }
    }
    parameters {
      key: "max_num_sequences"           # max concurrent in-flight requests
      value: { string_value: "32" }
    }
    parameters {
      key: "enable_chunked_context"      # chunk long prefills for fairer scheduling
      value: { string_value: "false" }
    }
    """
print(CONFIG_PBTXT)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Production deployment checklist
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Production deployment checklist")
print("━" * 65)
print()

CHECKLIST = [
    ("PRE-BUILD",  [
        "Verify GPU count and VRAM: `nvidia-smi -a`",
        "Set TP = number of GPUs on the node (for models requiring multiple GPUs)",
        "Choose dtype: float16/bfloat16 for A100, float8 for H100",
        "Set max_batch_size based on expected peak concurrency × 1.5",
        "Set max_input_len = max expected prompt length (longer = more memory)",
        "Set kv_cache_free_gpu_mem_fraction = 0.85 (leave room for activations)",
    ]),
    ("BUILD",      [
        "Run convert_checkpoint.py (creates TRT-LLM format weights)",
        "Run trtllm-build — expect 10–30 min for first build",
        "Verify rank*.engine files exist in output_dir",
        "Check config.json: num_attention_heads, num_kv_heads, hidden_size",
        "Validate engine: run summarize.py or a quick test inference",
    ]),
    ("TRITON SETUP", [
        "Clone tensorrtllm_backend repo for Triton configs",
        "Copy engine files to Triton model repo",
        "Set correct paths in tensorrt_llm/config.pbtxt",
        "Configure preprocessing/postprocessing with matching tokeniser",
        "Set ensemble/config.pbtxt to wire pre → trtllm → post",
    ]),
    ("LAUNCH", [
        "docker run nvcr.io/nvidia/tritonserver:24.xx-trtllm-python-py3",
        "tritonserver --model-repository=/path/to/triton_model_repo",
        "Check health: curl localhost:8000/v2/health/ready",
        "Check models loaded: curl localhost:8000/v2/models",
    ]),
    ("BENCHMARK", [
        "Use genai-perf (NVIDIA) or locust for load testing",
        "Measure: time-to-first-token (TTFT), inter-token latency (ITL)",
        "Measure: tokens/sec/GPU at P50 and P99 concurrency",
        "Check nvidia-smi: GPU utilisation should be >80% under load",
        "Tune max_queue_delay_microseconds: higher = more batching, more latency",
    ]),
    ("MONITORING", [
        "Export Triton metrics: curl localhost:8002/metrics  (Prometheus format)",
        "Key metrics: nv_inference_queue_duration_us, nv_inference_compute_duration_us",
        "Alert on: TTFT > 5s, queue depth > max_batch_size × 3",
        "Log GPU memory: watch for KV cache evictions (request failures)",
    ]),
]

for phase, items in CHECKLIST:
    print(f"  [{phase}]")
    for item in items:
        print(f"    ☐  {item}")
    print()

print("  TROUBLESHOOTING QUICK REFERENCE:")
print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │ Symptom                     │ Likely cause + fix                 │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │ OOM during build             │ Reduce max_batch or max_len       │")
print("  │ OOM during inference         │ Reduce kv_cache_free_frac         │")
print("  │ Low GPU util (<50%)          │ Increase max_num_sequences        │")
print("  │ High TTFT (>2s)              │ Requests queuing; scale GPUs      │")
print("  │ Accuracy degraded            │ Check dtype; verify quantisation  │")
print("  │ Engine build takes >1hr      │ Expected for large TP configs     │")
print("  │ NCCL timeout with TP>1       │ Check NVLink topology; peers      │")
print("  └──────────────────────────────────────────────────────────────────┘")
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