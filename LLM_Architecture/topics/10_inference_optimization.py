"""
Inference Optimisation - Making LLMs Fast and Efficient
========================================================

Inference optimisation is the collection of engineering and algorithmic techniques
that make model serving fast enough and cheap enough to be practical at scale.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Inference Optimisation"
DISPLAY_NAME = "10 · Inference Optimisation"
ICON         = "⚡"
SUBTITLE     = "Sampling, Quantisation, KV Cache, Batching & Beyond"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER — converts local images to base64 HTML for st.markdown()
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    """Convert a local image file to an HTML <img> tag with base64 data.
    This allows images to render inside st.markdown() with unsafe_allow_html=True.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Introduction to Inference Optimisation

When a large language model (LLM) is deployed in production, the challenges are no longer
only about model quality — they are about speed, cost, and scalability. Inference
optimisation is the collection of engineering and algorithmic techniques that make model
serving fast enough and cheap enough to be practical at scale.

This deep dive covers every major lever in the inference stack: from how tokens are
generated one-by-one, through the sampling strategies that govern creativity, to the
hardware-level tricks that compress models and parallelise requests. Each topic builds on
the previous, so by the end you will have a coherent mental model of why modern LLM
serving looks the way it does.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  KEY INSIGHT                                                        │
    │                                                                     │
    │  Inference is not just model execution — it is a systems           │
    │  engineering discipline. Understanding each concept here will      │
    │  help you reason about latency, cost, and quality trade-offs       │
    │  in any LLM deployment.                                             │
    └─────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
1. AUTOREGRESSIVE GENERATION
═══════════════════════════════════════════════════════════════════════════════

### What it is

Autoregressive generation is the fundamental mechanism by which all modern
transformer-based language models produce text. The model generates output one token at
a time, and each new token is conditioned on every token that came before it — both the
original prompt tokens and all previously generated tokens.

Formally, for a sequence of tokens x₁, x₂, ..., xₙ, the model learns the joint
probability distribution decomposed as:

    P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, x₂, ..., xᵢ₋₁)

At each step, the model:

    1. Runs a full forward pass over all previous tokens
    2. Produces a probability distribution (logits) over the vocabulary
    3. Samples or picks the next token from that distribution
    4. Appends the token and repeats


### Why it works this way

This approach is rooted in how transformers are trained. The model is trained with teacher
forcing on causal (left-to-right) masked self-attention, so it learns to predict the next
token given a left context. At inference time, this training objective is replicated
exactly: you always have a left context and need to predict what comes next.

The causal mask ensures that token i cannot attend to tokens i+1, i+2, etc. This makes
the model fundamentally sequential at generation time, even though training over the full
sequence can be parallelised.


### The Performance Bottleneck

The core bottleneck of autoregressive generation is that every new token requires a full
attention computation across all previous tokens. For a sequence of length n, this means
O(n²) attention operations per token generated. This is why long sequences are expensive.

    ┌─────────────────────────────┬─────────────────────────────────────────────┐
    │ Concept                     │ Details                                     │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ Prefill Phase               │ Processing the input prompt in one parallel │
    │                             │ forward pass (fast — exploits GPU fully)    │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ Decode Phase                │ Generating each output token sequentially   │
    │                             │ — the slow part of inference               │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ Time-to-First-Token (TTFT)  │ Latency of the prefill phase; driven by     │
    │                             │ prompt length and model size                │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ Tokens-per-Second (TPS)     │ Throughput of the decode phase; the key     │
    │                             │ production metric                           │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ Memory Bandwidth Bound      │ During decoding, the GPU is often waiting   │
    │                             │ on memory reads, not compute                │
    └─────────────────────────────┴─────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
2. GREEDY DECODING VS. SAMPLING
═══════════════════════════════════════════════════════════════════════════════

After the model produces a probability distribution over the vocabulary, a decoding
strategy must choose which token to emit. This choice has enormous consequences for
the quality and character of the output.


### Greedy Decoding

Greedy decoding always picks the single highest-probability token at each step.
It is deterministic: given the same prompt, you will always get the same output.

    next_token = argmax(logits)

    Advantages:
      - Deterministic and reproducible
      - Fast (no sampling overhead)
      - Good for factual Q&A, classification, structured extraction

    Disadvantages:
      - Can get stuck in repetition loops
      - Tends to produce generic, predictable text
      - Local optimum — maximising each step doesn't maximise sequence quality


### Sampling

Sampling draws the next token stochastically from the probability distribution.
This introduces randomness, which can be a feature (creativity) or a bug (hallucination).

Sampling allows unlikely but good tokens to sometimes be chosen. This is critical for
creative writing, code that explores different valid solutions, and avoiding the
degenerate repetition that plagues greedy decoding.


### Why Not Just Always Sample?

Unmodified sampling from the raw distribution has its own problems. The model assigns
non-trivial probability mass to extremely unlikely or incoherent tokens. Raw sampling
will occasionally emit garbage. This is why temperature, top-k, and top-p are needed —
they reshape the distribution before sampling.

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  TRADE-OFF                                                                   │
    │                                                                              │
    │  Greedy   = consistent & accurate                                            │
    │  Sampling = creative & diverse                                               │
    │                                                                              │
    │  Most production systems use sampling with constraints                       │
    │  (temperature + top-p) to get the best of both worlds.                       │
    └──────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
3. TEMPERATURE
═══════════════════════════════════════════════════════════════════════════════

### What Temperature Does

Temperature is a scalar applied to the raw logits (unnormalised log-probabilities)
before the softmax operation. It is one of the most important knobs for controlling
model behaviour at inference time.

    # Standard softmax:
    P(xᵢ) = exp(zᵢ) / Σ exp(zⱼ)

    # Temperature-scaled softmax:
    P(xᵢ) = exp(zᵢ / T) / Σ exp(zⱼ / T)

    where T is temperature, zᵢ is the raw logit for token i


### Effects Across the Range

    ┌──────────────┬──────────────────────────────────────────────────────────────┐
    │ Temperature  │ Effect                                                       │
    ├──────────────┼──────────────────────────────────────────────────────────────┤
    │ T = 0.0      │ Deterministic — equivalent to greedy decoding (argmax).      │
    │              │ The highest-logit token wins every time.                     │
    ├──────────────┼──────────────────────────────────────────────────────────────┤
    │ T < 1.0      │ Sharpens the distribution. The model becomes more confident  │
    │              │ and conservative. Repetition risk increases.                 │
    ├──────────────┼──────────────────────────────────────────────────────────────┤
    │ T = 1.0      │ The raw distribution unchanged — standard sampling from the  │
    │              │ model's learned probabilities.                               │
    ├──────────────┼──────────────────────────────────────────────────────────────┤
    │ T > 1.0      │ Flattens the distribution. More tokens become plausible.     │
    │              │ Creativity and incoherence both increase.                   │
    ├──────────────┼──────────────────────────────────────────────────────────────┤
    │ T = 2.0      │ Near-uniform distribution. Output becomes largely random     │
    │              │ and typically nonsensical.                                   │
    └──────────────┴──────────────────────────────────────────────────────────────┘

    Diagram — Visualising logit sharpening and flattening:

                    T = 0.3 (sharp)       T = 1.0 (raw)         T = 2.0 (flat)
                  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    P(token)      │  ██              │  │  ██              │  │  ███             │
                  │  ██              │  │  ██  ██          │  │  ███ ███ ███     │
                  │  ██  █           │  │  ██  ██  █  █    │  │  ███ ███ ███ ███ │
                  └──────────────────┘  └──────────────────┘  └──────────────────┘
                   One token dominates   Natural spread          Near-uniform


### Intuition

Imagine the logits as a bar chart of token preferences. Temperature 0 just picks the
tallest bar. Temperature 1 samples proportional to bar height. Temperature 2 shrinks all
the bars towards equal height — the model acts as if it is much less certain about
everything.

High temperature does not make the model more "creative" in a structured sense — it just
makes it less confident. True creativity comes from the model's training; temperature
just unlocks more of the distribution.


### Practical Recommendations

    •   Factual tasks (RAG, Q&A, extraction)    →   T = 0.0 – 0.3
    •   Code generation                         →   T = 0.2 – 0.6
    •   Conversational AI                       →   T = 0.7 – 1.0
    •   Creative writing, brainstorming         →   T = 0.9 – 1.3
    •   Never use T > 1.5 in production without very specific need


═══════════════════════════════════════════════════════════════════════════════
4. TOP-K AND TOP-P (NUCLEUS SAMPLING)
═══════════════════════════════════════════════════════════════════════════════

### The Problem with Raw Sampling

Even with a good temperature setting, the vocabulary has tens of thousands of tokens and
many of them will have non-zero (if tiny) probability. Occasionally sampling from this
long tail produces incoherent or offensive outputs. Top-K and Top-P both solve this by
truncating the candidate pool before sampling.


### Top-K Sampling

Top-K sampling retains only the K most probable tokens, zeroes out the rest,
re-normalises to a valid probability distribution, and then samples from those K
candidates.

    Algorithm:
    1. Sort tokens by probability (descending)
    2. Keep only the top K
    3. Set all others to probability 0
    4. Renormalise: divide by sum of kept probabilities
    5. Sample

    Typical values: K = 20, 40, 50
    K = 1 is equivalent to greedy decoding

The weakness of top-K is that K is a fixed count. In some contexts the model is very
confident and the top 3 tokens cover 95% of the probability mass — using K=50 still
includes many junk tokens. In other contexts the model is genuinely uncertain and 50
tokens might still miss good options.


### Top-P (Nucleus) Sampling

Top-P, also called nucleus sampling, addresses the rigidity of top-K by using a dynamic
threshold. Instead of a fixed count, it keeps the smallest set of tokens whose cumulative
probability exceeds p.

    Algorithm:
    1. Sort tokens by probability (descending)
    2. Compute cumulative probabilities
    3. Find the nucleus: smallest set where cumsum >= p
    4. Zero out everything outside the nucleus
    5. Renormalise and sample

    Typical values: p = 0.9 or p = 0.95
    p = 1.0 is equivalent to no truncation
    p = 0.0 is effectively greedy

    Diagram — How nucleus size adapts with model confidence:

    CONFIDENT (model knows the answer)    UNCERTAIN (many valid continuations)

    Token    Prob   CumSum                Token    Prob   CumSum
    ─────────────────────────             ─────────────────────────────────
    "Paris"  0.82   0.82  ← nucleus       "run"    0.18   0.18
    "Lyon"   0.09   0.91  ←  ends here    "walk"   0.14   0.32
    "Rome"   0.04   0.95                  "jump"   0.12   0.44
    ...                                   "move"   0.10   0.54
                                          "go"     0.09   0.63
                                          "step"   0.08   0.71
                                          "rush"   0.07   0.78
                                          "dash"   0.06   0.84
                                          "flee"   0.05   0.89
                                          "head"   0.04   0.93  ← nucleus ends
                                          ...

    Nucleus = 2 tokens               Nucleus = 10 tokens
    (model is sure → small pool)     (model is unsure → larger pool)

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  ANALOGY                                                                     │
    │                                                                              │
    │  Top-P is like saying "I'll only consider the likely suspects" — the        │
    │  nucleus adapts in size. When the model is sure, the nucleus is tiny        │
    │  (3-5 tokens). When uncertain, it expands to include many plausible         │
    │  options.                                                                    │
    └──────────────────────────────────────────────────────────────────────────────┘


### Combining Top-K, Top-P, and Temperature

In practice, these three are often applied together in the following order:

    1. Apply temperature scaling to the raw logits
    2. Apply top-K to hard-limit the candidate count
    3. Apply top-P to further trim the nucleus
    4. Sample from the remaining distribution

A common production configuration is temperature=0.7, top-k=50, top-p=0.95,
which gives coherent but diverse output.


### Additional Sampling Strategy: Min-P

A newer alternative is Min-P sampling, which filters out tokens whose probability is
less than a fraction of the top token's probability. If the top token has 60% probability
and min-p=0.05, any token with less than 3% probability is discarded. This scales
dynamically with confidence and often outperforms top-p in practice.


═══════════════════════════════════════════════════════════════════════════════
5. SPECULATIVE DECODING
═══════════════════════════════════════════════════════════════════════════════

### The Core Problem

The autoregressive decode loop is fundamentally sequential — you cannot generate token
N+1 until token N is known. This means the large model sits mostly idle between steps,
waiting. Speculative decoding exploits this idle time by using a much smaller, faster
"draft" model to propose several tokens ahead, which the large model then verifies in a
single parallel pass.


### How It Works

    Step 1 — DRAFT
      Small draft model (e.g. 7B) generates K tokens autoregressively.
      This is cheap: K forward passes of a tiny model.
      Result: a proposed sequence of K tokens.

    Step 2 — VERIFY
      Large target model (e.g. 70B) runs ONE forward pass over the full
      context + K draft tokens in parallel.
      Result: K+1 probability distributions (one per position).

    Step 3 — ACCEPT / REJECT
      For each draft token, compare: P_target(t) vs P_draft(t)
      Accept token if: random() <= P_target(t) / P_draft(t)
      On first rejection, resample from a corrected distribution and stop.

    Key property: Accepted tokens are EXACTLY distributed as if the large
    model generated them — no quality loss.

    Diagram — Comparing standard decode vs. speculative decode:

    STANDARD DECODING (one token per large-model pass):

    │◄── Large Model Pass ──►│◄── Large Model Pass ──►│◄── Large Model Pass ──►│
    │       Token 1          │       Token 2          │       Token 3          │
    Total = 3 × T_large

    SPECULATIVE DECODING (K=4 draft tokens, acceptance rate ~80%):

    │◄─ Draft (4 tokens) ─►│◄────────── Large Model Verify (1 pass) ──────────►│
    │  t1  t2  t3  t4      │  ✓ Accept  ✓ Accept  ✓ Accept  ✗ Reject → resample │
    Total ≈ T_draft×4 + T_large  →  effectively ~3.2 tokens per large model pass


### Why This Is Faster

The large model's forward pass takes roughly the same time whether it processes 1 token
or K+1 tokens in parallel (because the GPU is parallelising across the sequence
dimension). If the draft model achieves an acceptance rate of 80%, you effectively get
4-5 tokens per large model pass instead of 1 — a 4-5x decode speedup.


### Variants and Extensions

    ┌──────────────────────────────┬──────────────────────────────────────────────┐
    │ Variant                      │ Description                                  │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │ Self-speculative / Medusa    │ The large model itself generates draft tokens │
    │                              │ using auxiliary heads — no separate draft     │
    │                              │ model needed                                 │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │ Draft model selection        │ Draft model must be fast AND aligned with     │
    │                              │ the target model's distribution. Same         │
    │                              │ training data helps.                          │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │ Speculative streaming        │ Stream accepted tokens immediately for lower  │
    │                              │ perceived latency                             │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │ Tree-based speculation       │ Draft multiple branches simultaneously,       │
    │                              │ verify with a tree attention mask             │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │ Lookahead decoding           │ Use n-gram matching from existing context to  │
    │                              │ generate draft tokens without a separate      │
    │                              │ model                                         │
    └──────────────────────────────┴──────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  SPEEDUP RANGE                                                               │
    │                                                                              │
    │  Typical gains: 2x–4x for code generation (highly predictable).            │
    │  Lower for creative tasks (fewer acceptances).                              │
    │  Draft model quality is the main bottleneck.                                │
    └──────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
6. QUANTISATION
═══════════════════════════════════════════════════════════════════════════════

### The Memory Problem

A 70B parameter model in 32-bit floating point (FP32) requires approximately 280 GB of
VRAM — more than most server-grade GPU clusters. Even in 16-bit (BF16/FP16) it requires
140 GB. Quantisation compresses the model's weights to lower bit-width integers,
dramatically reducing memory footprint while preserving most of the model's quality.


### How Quantisation Works

Floating point values have a wide dynamic range but many of those values cluster in a
much smaller range. Quantisation maps this continuous range to a discrete set of integer
values. At inference time, the weights are dequantised back to floating point for the
actual matrix multiplication (or specialised integer kernels handle the arithmetic
directly).

    FP32 weight:  0.3745...
    FP16 weight:  0.374...     (2 bytes — standard training precision)
    BF16 weight:  0.375...     (2 bytes — better dynamic range than FP16)
    INT8 weight:  47            (1 byte — zero_point=0, scale=0.00796)
    INT4 weight:  3             (4 bits — 2x more aggressive)


### Quantisation Methods

    ┌────────────────────────────────────┬──────────────────────────────────────────┐
    │ Method                             │ Description                              │
    ├────────────────────────────────────┼──────────────────────────────────────────┤
    │ Post-Training Quantisation (PTQ)   │ Quantise a pre-trained model without     │
    │                                    │ retraining. Simple and fast. Some        │
    │                                    │ accuracy loss.                           │
    ├────────────────────────────────────┼──────────────────────────────────────────┤
    │ GPTQ                               │ Minimises quantisation error layer-by-   │
    │                                    │ layer using a small calibration dataset. │
    │                                    │ Best accuracy for INT4 PTQ.              │
    ├────────────────────────────────────┼──────────────────────────────────────────┤
    │ GGUF / llama.cpp                   │ Mixed-precision PTQ popular for CPU and  │
    │                                    │ consumer GPU inference. Flexible bit-    │
    │                                    │ depth per layer.                         │
    ├────────────────────────────────────┼──────────────────────────────────────────┤
    │ AWQ (Activation-aware)             │ Protects the most important weight       │
    │                                    │ channels from aggressive quantisation    │
    │                                    │ based on activation statistics.          │
    ├────────────────────────────────────┼──────────────────────────────────────────┤
    │ QLoRA                              │ INT4 quantised base model + low-rank     │
    │                                    │ FP16 adapter. Enables fine-tuning of     │
    │                                    │ large models on consumer hardware.       │
    ├────────────────────────────────────┼──────────────────────────────────────────┤
    │ Quantisation-Aware Training (QAT)  │ Simulates quantisation during training.  │
    │                                    │ Best quality but requires full training  │
    │                                    │ run.                                     │
    └────────────────────────────────────┴──────────────────────────────────────────┘


### Memory Savings

    ┌───────────────────────┬─────────────────────────────────────────────────────┐
    │ Precision             │ 70B Parameter Model Memory Footprint                │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ FP32  (32-bit)        │ 280 GB VRAM                                         │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ FP16 / BF16  (16-bit) │ 140 GB VRAM                                         │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ INT8  (8-bit)         │ 70 GB VRAM  (fits on 2x A100 80GB)                  │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ INT4  (4-bit)         │ 35 GB VRAM  (fits on 1x A100 80GB)                  │
    ├───────────────────────┼─────────────────────────────────────────────────────┤
    │ INT2  (2-bit)         │ 17.5 GB VRAM  (quality degradation is significant)  │
    └───────────────────────┴─────────────────────────────────────────────────────┘


### Quality vs. Compression Trade-off

INT8 quantisation typically incurs less than 1% perplexity degradation on most
benchmarks. INT4 incurs 2-5% degradation depending on model size. Larger models tolerate
quantisation better than smaller ones — a 70B model at INT4 often outperforms a 7B model
at FP16 on the same hardware.

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  RULE OF THUMB                                                               │
    │                                                                              │
    │  If you can, prefer INT8 for quality-sensitive production.                  │
    │  Use INT4 when memory is the hard constraint.                               │
    │  Avoid INT2/INT1 unless you have specific calibration and accept            │
    │  significant quality loss.                                                  │
    └──────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
7. BATCHING STRATEGIES
═══════════════════════════════════════════════════════════════════════════════

### Why Batching Matters

GPU hardware is massively parallel. A single request uses only a small fraction of
available compute. Batching groups multiple inference requests together so they can share
a single forward pass, dramatically improving hardware utilisation and overall throughput.


### Static Batching

The naive approach: collect N requests, pad them all to the same length, run one forward
pass, return results. Simple to implement but wasteful — short sequences waste compute
waiting for long ones to finish, and the batch cannot accept new requests mid-flight.


### Continuous Batching (Iteration-Level Batching)

Continuous batching, introduced by the Orca paper (2022), solves static batching's
efficiency problem. Instead of batching at the request level, the server batches at the
iteration level — every single decode step can incorporate new requests and evict
completed ones.

    Static Batching Timeline:
      t=0:  [Req A (100 tok), Req B (20 tok), Req C (80 tok)] START
      t=20: B finishes. A and C still running. GPU sits partially idle.
      t=80: C finishes. Only A running now.
      t=100:A finishes. Batch complete.

    Continuous Batching Timeline:
      t=0:  [A, B, C] start
      t=20: B finishes. New request D joins immediately.
      t=35: D finishes. E joins.
      ...and so on — GPU is always full.

Continuous batching is now the default in production serving frameworks including vLLM,
TGI (Text Generation Inference), and TensorRT-LLM. It typically achieves 5-10x higher
throughput compared to static batching at equivalent latency.


### In-flight Batching Challenges

    ┌───────────────────────────────┬────────────────────────────────────────────┐
    │ Challenge                     │ Description                                │
    ├───────────────────────────────┼────────────────────────────────────────────┤
    │ Variable sequence lengths     │ Requests in the same batch are at          │
    │                               │ different decode steps — requires careful  │
    │                               │ memory management                          │
    ├───────────────────────────────┼────────────────────────────────────────────┤
    │ KV cache memory pressure      │ Each active sequence holds its own KV      │
    │                               │ cache — limits how many can be batched     │
    │                               │ simultaneously                             │
    ├───────────────────────────────┼────────────────────────────────────────────┤
    │ Request prioritisation        │ SLA-sensitive requests may need to         │
    │                               │ preempt lower-priority ones mid-batch      │
    ├───────────────────────────────┼────────────────────────────────────────────┤
    │ Chunked prefill               │ Split long prompt prefills into chunks so  │
    │                               │ they don't block the decode loop for other │
    │                               │ requests                                   │
    └───────────────────────────────┴────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
8. KV CACHE & PAGEDATTENTION
═══════════════════════════════════════════════════════════════════════════════

### What the KV Cache Is

During the attention computation, the model computes Key and Value matrices for each
token in the context. In autoregressive generation, the context grows by one token each
step. Without caching, the model would recompute K and V for all previous tokens on every
step — an O(n) redundant computation per step.

The KV cache stores these K and V matrices after they are computed, so each new step only
needs to compute K and V for the single new token and then run attention over the cached
values. This reduces per-step compute from O(n) to O(1) for the context processing.

    Memory cost of KV cache:
      2 (K and V) × num_layers × num_heads × head_dim × seq_len × bytes_per_element

    Example: LLaMA-2 70B, BF16, sequence of 4096 tokens:
      2 × 80 × 64 × 128 × 4096 × 2 bytes = ~10.7 GB

    For 16 concurrent sequences: ~171 GB — easily exceeds GPU memory.
    This is the central constraint on batch size.


### The Memory Fragmentation Problem

Before PagedAttention, serving frameworks pre-allocated a contiguous block of GPU memory
for each request's KV cache at request start, sized for the maximum possible sequence
length. This led to two inefficiencies: internal fragmentation (unused space within an
over-allocated block) and external fragmentation (wasted gaps between blocks).
Utilisation was typically 20-40%.


### PagedAttention

PagedAttention (from the vLLM paper, 2023) borrows the concept of virtual memory paging
from operating systems. Instead of contiguous pre-allocation, the KV cache is managed in
fixed-size pages (blocks) that can be allocated, freed, and shared non-contiguously.

    Key ideas:

    1. BLOCK TABLE: Each sequence has a mapping from logical block
       number to physical block location in GPU memory.

    2. ON-DEMAND ALLOCATION: New blocks are allocated only when
       needed (as the sequence actually grows).

    3. COPY-ON-WRITE SHARING: For beam search or parallel sampling,
       multiple sequences can share the same physical KV blocks
       until they diverge. Only then are blocks copied.

    4. ZERO WASTE: No internal fragmentation (blocks are small and
       fixed), near-zero external fragmentation.

    Diagram — PagedAttention block management:

    TRADITIONAL (contiguous, pre-allocated):
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Req A [████████████████░░░░░░░░░] 60% used  → 40% wasted            │
    │ Req B [██████░░░░░░░░░░░░░░░░░░░] 25% used  → 75% wasted            │
    │ Req C [███████████████████████░░] 90% used  → 10% wasted            │
    └──────────────────────────────────────────────────────────────────────┘

    PAGEDATTENTION (non-contiguous, on-demand):
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ A │ A │ B │ A │ C │ C │ A │ B │ C │ C │ A │ C │ C │ B │   │   │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
    Blocks assigned on demand. Near-zero waste. >90% utilisation.

PagedAttention increases KV cache utilisation from ~20-40% to over 90%, enabling far
more concurrent requests on the same hardware. Combined with continuous batching, it is
the reason vLLM achieves up to 24x higher throughput than naive serving.


### Flash Attention

Flash Attention is a complementary optimisation at the attention computation level.
Standard attention materialises the full O(n²) attention matrix in GPU high-bandwidth
memory (HBM). Flash Attention instead tiles the computation to keep as much as possible
in the fast on-chip SRAM, dramatically reducing memory reads and writes.

    ┌────────────────────────┬──────────────────────────────────────────────────┐
    │ Version                │ Description                                      │
    ├────────────────────────┼──────────────────────────────────────────────────┤
    │ Standard Attention     │ O(n²) memory — writes full attention matrix to   │
    │                        │ HBM then reads it back                           │
    ├────────────────────────┼──────────────────────────────────────────────────┤
    │ Flash Attention v1     │ Tiled computation, O(n) memory, same             │
    │                        │ mathematical result, 2-4x faster                 │
    ├────────────────────────┼──────────────────────────────────────────────────┤
    │ Flash Attention v2     │ Better parallelism across heads and sequence     │
    │                        │ length, ~2x faster than v1                       │
    ├────────────────────────┼──────────────────────────────────────────────────┤
    │ Flash Attention v3     │ Further GPU-specific optimisations for H100      │
    │                        │ architecture                                     │
    └────────────────────────┴──────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
9. BEAM SEARCH
═══════════════════════════════════════════════════════════════════════════════

### Motivation

Both greedy decoding and random sampling are one-shot strategies — they commit to one
sequence and cannot backtrack. Beam search explores multiple candidate sequences
simultaneously, keeping the B most probable partial sequences (beams) at each step,
in hopes of finding a better final sequence.


### How Beam Search Works

    B = beam width (typically 4–10)

    Step 1: At each decode step, expand EVERY active beam.
      - For beam i with score s_i, generate top B next tokens
      - Create B×B candidate extensions

    Step 2: Score each candidate.
      - score = s_i + log P(next_token | beam_i)
      (log probabilities sum = probability product)

    Step 3: Keep top B candidates across all beams.
      - Prune to the B highest-scoring sequences

    Step 4: Repeat until all beams hit EOS or max length.

    Step 5: Return the highest-scoring completed sequence.

    Diagram — Beam search tree (B=2):

                  [START]
                 /       \\
           "The"           "A"
           /   \\           /   \\
        "cat" "dog"    "cat" "bird"
        /  \\    \\         |
    "sat" "ran" "ran"  "flew"
         ↑
         Prune after each step, keep only top-B paths


### When Beam Search Helps

Beam search is most valuable when the output space has a clear objective score (like
translation BLEU) and when longer-range coherence matters. Committing greedily to a
slightly worse token early might preclude a much better sequence later; beam search
hedges against this.


### Beam Search's Weaknesses

    ┌──────────────────────┬────────────────────────────────────────────────────┐
    │ Weakness             │ Details                                            │
    ├──────────────────────┼────────────────────────────────────────────────────┤
    │ Memory cost          │ B active beams means B KV caches simultaneously   │
    │                      │ — expensive at high beam widths                    │
    ├──────────────────────┼────────────────────────────────────────────────────┤
    │ Compute cost         │ B × B expansions per step — scales poorly,         │
    │                      │ rarely used with B > 10                            │
    ├──────────────────────┼────────────────────────────────────────────────────┤
    │ Generic outputs      │ Maximising likelihood often produces safe,         │
    │                      │ repetitive, low-diversity text                     │
    ├──────────────────────┼────────────────────────────────────────────────────┤
    │ Not always better    │ For open-ended generation, sampling often          │
    │                      │ produces more natural text than beam search        │
    ├──────────────────────┼────────────────────────────────────────────────────┤
    │ Length bias          │ Shorter sequences tend to have higher probability; │
    │                      │ need length normalisation to counteract            │
    └──────────────────────┴────────────────────────────────────────────────────┘


### Diverse Beam Search

An extension that penalises beams for being too similar to each other, explicitly
encouraging diversity in the output. Useful when you want to generate multiple distinct
candidate answers.


### When to Use Each Decoding Strategy

    ┌──────────────────────────────────┬──────────────────────────────────────────┐
    │ Strategy                         │ Best Use Case                            │
    ├──────────────────────────────────┼──────────────────────────────────────────┤
    │ Greedy                           │ Fast lookup, factual Q&A, structured     │
    │                                  │ extraction where speed matters most      │
    ├──────────────────────────────────┼──────────────────────────────────────────┤
    │ Temperature + Top-P sampling     │ Conversational AI, creative writing,     │
    │                                  │ most general-purpose generation          │
    ├──────────────────────────────────┼──────────────────────────────────────────┤
    │ Beam search (B=4–8)              │ Machine translation, text summarisation, │
    │                                  │ constrained generation tasks             │
    ├──────────────────────────────────┼──────────────────────────────────────────┤
    │ Diverse beam search              │ Generating a ranked list of candidate    │
    │                                  │ responses for re-ranking downstream      │
    └──────────────────────────────────┴──────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
10. ADDITIONAL OPTIMISATION TOPICS
═══════════════════════════════════════════════════════════════════════════════

### 10.1 Memory Hierarchy Awareness

Modern GPU inference must be designed around three levels of memory:

    •   HBM (High-Bandwidth Memory, 40-80 GB, ~2-3 TB/s bandwidth)
        Main GPU VRAM where weights and KV cache live.

    •   SRAM (on-chip cache, ~10-50 MB, ~10-20 TB/s bandwidth)
        Ultra-fast but tiny; Flash Attention keeps data here.

    •   CPU RAM / NVMe
        Overflow storage for offloading; used in speculative decoding draft
        model caching.

The key insight: LLM decode is memory bandwidth-bound, not compute-bound. Moving data
from HBM to compute units is the bottleneck, which is why reducing model size
(quantisation) and reducing memory reads (Flash Attention, KV cache) are so impactful.


### 10.2 Tensor Parallelism & Pipeline Parallelism

For models too large for a single GPU, parallelism strategies split the model across
devices:

    ┌────────────────────────────┬────────────────────────────────────────────────┐
    │ Strategy                   │ Description                                    │
    ├────────────────────────────┼────────────────────────────────────────────────┤
    │ Tensor Parallelism (TP)    │ Split individual weight matrices across GPUs   │
    │                            │ — each GPU computes part of each matrix        │
    │                            │ multiply. Requires fast NVLink between GPUs.   │
    ├────────────────────────────┼────────────────────────────────────────────────┤
    │ Pipeline Parallelism (PP)  │ Assign different layers to different GPUs.     │
    │                            │ Good for multiple nodes. Introduces pipeline   │
    │                            │ bubbles that reduce utilisation.               │
    ├────────────────────────────┼────────────────────────────────────────────────┤
    │ Sequence Parallelism       │ Split the sequence dimension across GPUs.      │
    │                            │ Used for extremely long contexts.              │
    ├────────────────────────────┼────────────────────────────────────────────────┤
    │ Expert Parallelism         │ For Mixture-of-Experts models: different GPUs  │
    │                            │ hold different expert weights.                 │
    └────────────────────────────┴────────────────────────────────────────────────┘


### 10.3 Prefix Caching (Prompt Caching)

When many requests share a common prefix (e.g., a long system prompt), recomputing its
KV cache for every request is wasteful. Prefix caching stores the KV cache for common
prefixes and reuses it across requests, reducing time-to-first-token and compute cost
for shared-prefix workloads by up to 90%.

This is especially valuable for:
    •   Multi-turn conversations (reuse conversation history)
    •   RAG systems (reuse retrieved documents)
    •   A/B testing (reuse system prompts)


### 10.4 Latency vs. Throughput Trade-offs

Inference serving involves a fundamental tension between optimising for individual
request latency and maximising system throughput:

    ┌──────────────────────────┬──────────────────────────────────────────────────┐
    │ Mode                     │ Description                                      │
    ├──────────────────────────┼──────────────────────────────────────────────────┤
    │ Low Latency Mode         │ Smaller batches, less prefill chunking, reserved │
    │                          │ capacity. Good for interactive chatbots. Lower   │
    │                          │ GPU utilisation.                                 │
    ├──────────────────────────┼──────────────────────────────────────────────────┤
    │ High Throughput Mode     │ Larger batches, aggressive continuous batching.  │
    │                          │ Good for batch processing. Higher latency per    │
    │                          │ request.                                         │
    ├──────────────────────────┼──────────────────────────────────────────────────┤
    │ Latency SLOs             │ Service Level Objectives: e.g., P95 TTFT < 500ms │
    │                          │ AND P50 output rate > 50 tok/s                   │
    ├──────────────────────────┼──────────────────────────────────────────────────┤
    │ Cost per Token           │ Throughput-optimised serving minimises cost.     │
    │                          │ Latency-optimised serving maximises UX quality.  │
    └──────────────────────────┴──────────────────────────────────────────────────┘


### 10.5 Structured Output Generation (Constrained Decoding)

Constrained decoding forces the model's output to conform to a grammar or schema (e.g.,
valid JSON). At each step, tokens that would violate the constraint are zeroed out before
sampling. Libraries like Outlines and Guidance implement this. It has near-zero quality
impact and eliminates post-processing parsing errors.


═══════════════════════════════════════════════════════════════════════════════
SUMMARY — THE INFERENCE OPTIMISATION TOOLKIT
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────┬──────────────────────────┬─────────────────────────┬──────────────────────────┐
    │ Technique                   │ Problem Solved           │ Primary Benefit         │ Trade-off / Cost         │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Greedy Decoding             │ Choose next token        │ Deterministic, fast     │ Repetitive outputs       │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Temperature                 │ Output diversity control │ Quality/creativity dial │ Too high = incoherence   │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Top-K Sampling              │ Trim probability tail    │ Prevent rare token      │ Fixed K is inflexible    │
    │                             │                          │ disasters               │                          │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Top-P Sampling              │ Adaptive candidate pool  │ Better quality vs top-K │ Adds one sort per step   │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Min-P Sampling              │ Adaptive lower bound     │ Scales with confidence  │ Less well-known          │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Beam Search                 │ Better sequence quality  │ Higher BLEU/ROUGE       │ B× memory and compute    │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Speculative Decoding        │ Sequential decode        │ 2–4× decode speedup     │ Needs draft model        │
    │                             │ bottleneck               │                         │                          │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ INT8 Quantisation           │ VRAM memory pressure     │ ~2× memory reduction    │ <1% quality loss         │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ INT4 Quantisation           │ Severe memory constraint │ ~4× memory reduction    │ 2–5% quality loss        │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ KV Cache                    │ Redundant recomputation  │ Linear vs O(n²) decode  │ Proportional VRAM cost   │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ PagedAttention              │ KV cache fragmentation   │ 90%+ memory utilisation │ Complex implementation   │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Flash Attention             │ HBM bandwidth bottleneck │ 2–4× attention speedup  │ Requires custom kernels  │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Continuous Batching         │ GPU underutilisation     │ 5–10× throughput gain   │ Scheduling complexity    │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Prefix Caching              │ Shared prompt recompute  │ Up to 90% TTFT saving   │ Cache management         │
    ├─────────────────────────────┼──────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ Tensor Parallelism          │ Single GPU size limit    │ Scale to multi-GPU      │ Requires NVLink          │
    └─────────────────────────────┴──────────────────────────┴─────────────────────────┴──────────────────────────┘


### Closing Thoughts

Inference optimisation is not a single technique — it is a stack of complementary
improvements that each target a different bottleneck. The best production systems layer
all of these on top of each other:

    •   Quantised models (INT8/INT4) to fit more in VRAM
    •   PagedAttention to maximise KV cache utilisation
    •   Continuous batching to keep the GPU busy
    •   Flash Attention to reduce memory bandwidth pressure
    •   Speculative decoding to hide sequential decode latency
    •   Prefix caching to avoid redundant prefill compute
    •   Carefully tuned sampling (temperature + top-p) to control output quality

Understanding each layer independently is important, but understanding how they interact
is what separates a good ML engineer from a great one. VRAM you save through quantisation
can be used for larger batches; larger batches benefit more from continuous batching;
better batching makes speculative decoding's draft acceptance rate more valuable.
These techniques compound.

The field moves fast. Flash Attention 3, speculative streaming, and novel quantisation
formats (like FP8) are all recent developments. The conceptual framework in this module
will help you quickly understand any new technique: what bottleneck does it target,
what is the quality/cost trade-off, and where in the serving stack does it live?

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
None
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────


OPERATIONS = {

    # ─────────────────────────────────────────────────────────────────────────
    "1. Sampling Strategies — Temperature, Top-K, Top-P": {
        "description": "Implement and compare temperature scaling, top-K, and nucleus (top-P) sampling from scratch",
        "runnable": True,
        "code": '''
import math, random

# ── SOFTMAX ──────────────────────────────────────────────────────────────────
def softmax(logits):
    """Convert raw logits to a probability distribution (numerically stable)."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]

# ── TEMPERATURE ───────────────────────────────────────────────────────────────
def apply_temperature(logits, T):
    """
    Scale logits by temperature T before softmax.
      T < 1.0  →  sharpens distribution  (model more confident, less creative)
      T = 1.0  →  no change              (raw probabilities unchanged)
      T > 1.0  →  flattens distribution  (model less confident, more creative)
    """
    if T <= 0:
        raise ValueError("Temperature must be > 0.")
    return [z / T for z in logits]

# ── TOP-K ─────────────────────────────────────────────────────────────────────
def top_k_filter(logits, k):
    """
    Zero out all but the top-K highest logits.
    Sets excluded logits to -inf so exp(-inf) = 0 after softmax.
    """
    if k >= len(logits):
        return logits
    threshold = sorted(logits, reverse=True)[k - 1]
    NEG_INF = float("-inf")
    return [z if z >= threshold else NEG_INF for z in logits]

# ── TOP-P (NUCLEUS) ───────────────────────────────────────────────────────────
def top_p_filter(logits, p):
    """
    Keep the smallest set of tokens whose cumulative probability >= p.
    Nucleus size adapts: small when model is confident, large when uncertain.
    """
    probs = softmax(logits)
    sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    nucleus, cum = set(), 0.0
    for idx in sorted_idx:
        nucleus.add(idx)
        cum += probs[idx]
        if cum >= p:
            break
    NEG_INF = float("-inf")
    return [z if i in nucleus else NEG_INF for i, z in enumerate(logits)]

# ── COMBINED PIPELINE ─────────────────────────────────────────────────────────
def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Full production pipeline: temperature → top-K → top-P → sample.
    Returns: index of sampled token in the vocabulary.
    """
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    probs = softmax(logits)
    r, cum = random.random(), 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1

# ── DEMONSTRATION ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VOCAB  = ["cat", "dog", "the", "a", "sat", "ran", "XYZ", "???"]
    logits = [2.5,   2.1,   4.0,  3.5,  1.0,   0.8,  -3.0,  -5.0]

    print("=" * 65)
    print("  SAMPLING STRATEGIES DEMONSTRATION")
    print("=" * 65)
    print("  Vocab :", VOCAB)
    print("  Logits:", logits)

    # --- Temperature effect ---
    print()
    print("  --- TEMPERATURE EFFECT ---")
    header = f"  {'Token':<10} {'T=0.3':>8} {'T=1.0':>8} {'T=2.0':>8}"
    print(header)
    print("  " + "-" * 40)
    p03  = softmax(apply_temperature(logits, 0.3))
    p10  = softmax(logits)
    p20  = softmax(apply_temperature(logits, 2.0))
    for i, tok in enumerate(VOCAB):
        print(f"  {tok:<10} {p03[i]:>8.3f} {p10[i]:>8.3f} {p20[i]:>8.3f}")
    print()
    print("  T=0.3: 'the' dominates strongly (conservative)")
    print("  T=1.0: raw distribution (balanced)")
    print("  T=2.0: distribution flattened (even XYZ gets a chance)")

    # --- Top-K ---
    print()
    print("  --- TOP-K FILTERING (K=3) ---")
    fk = top_k_filter(logits, k=3)
    pk = softmax(fk)
    print(f"  {'Token':<10} {'Original':>10} {'After K=3':>10}  In Pool?")
    print("  " + "-" * 44)
    for i, tok in enumerate(VOCAB):
        mark = "yes" if pk[i] > 0 else " no"
        print(f"  {tok:<10} {p10[i]:>10.3f} {pk[i]:>10.3f}  {mark}")

    # --- Top-P ---
    print()
    print("  --- TOP-P NUCLEUS SAMPLING (P=0.90) ---")
    fp = top_p_filter(logits, p=0.90)
    pp = softmax(fp)
    print(f"  {'Token':<10} {'Prob':>8}  In Nucleus?")
    print("  " + "-" * 30)
    for i, tok in enumerate(VOCAB):
        mark = "yes" if pp[i] > 0 else " no"
        print(f"  {tok:<10} {p10[i]:>8.3f}  {mark}")

    # --- Combined 10 samples ---
    print()
    print("  --- COMBINED (temp=0.8, top_k=5, top_p=0.95) — 10 samples ---")
    counts = {}
    for _ in range(10):
        t = sample_next_token(logits, temperature=0.8, top_k=5, top_p=0.95)
        counts[VOCAB[t]] = counts.get(VOCAB[t], 0) + 1
    for tok, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * n
        print(f"  {tok:<10} {bar} ({n})")
    print()
    print("  (Output is stochastic — re-run to see different samples)")
''',
    },


    # ─────────────────────────────────────────────────────────────────────────
    "2. Autoregressive Generation Loop": {
        "description": "Simulate the autoregressive token-by-token generation loop with a KV cache stub",
        "runnable": True,
        "code": '''
import math, random, time

# ── TINY VOCABULARY ──────────────────────────────────────────────────────────
VOCAB = {
    0: "<PAD>", 1: "<BOS>", 2: "<EOS>",
    3: "The",   4: "quick", 5: "brown",  6: "fox",
    7: "jumps", 8: "over",  9: "the",   10: "lazy",
   11: "dog",  12: ".",    13: "A",     14: "fast",
   15: "cat",  16: "sits", 17: "quietly", 18: "nearby",
}
EOS_ID, BOS_ID, VOCAB_SIZE = 2, 1, len(VOCAB)

# ── KV CACHE STUB ─────────────────────────────────────────────────────────────
class KVCache:
    """
    Simulates the KV cache. In a real transformer this stores:
        K matrix: [num_layers, num_heads, seq_len, head_dim]
        V matrix: [num_layers, num_heads, seq_len, head_dim]

    Memory formula:
        2 x layers x heads x head_dim x seq_len x bytes_per_param
    Example: LLaMA-2 70B, BF16, 4096 tokens → ~10.7 GB per sequence
    """
    def __init__(self):
        self.tokens = []    # token IDs processed so far
        self.hits   = 0     # cumulative reuse count (K/V lookups saved)

    def store(self, token_id):
        self.tokens.append(token_id)

    def attend(self):
        # Each decode step attends to ALL cached tokens — O(n) per step
        self.hits += len(self.tokens)
        return len(self.tokens)

    def size(self):
        return len(self.tokens)

# ── SOFTMAX + SAMPLING ────────────────────────────────────────────────────────
def softmax(logits):
    m = max(logits)
    e = [math.exp(z - m) for z in logits]
    s = sum(e)
    return [x / s for x in e]

def sample(logits, T=0.85):
    probs = softmax([z / T for z in logits])
    r, cum = random.random(), 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1

# ── FAKE FORWARD PASS ─────────────────────────────────────────────────────────
# Markov-style transitions to produce coherent demo output
TRANSITIONS = {
    BOS_ID: {3: 4.0, 13: 3.5},
    3:  {4: 3.5, 14: 3.0, 10: 2.5},   # The  → quick / fast / lazy
    13: {4: 3.5, 14: 3.0, 15: 2.5},   # A    → quick / fast / cat
    4:  {5: 4.0, 15: 2.0},            # quick → brown / cat
    14: {5: 4.0, 15: 2.0},
    5:  {6: 4.5},   6:  {7: 4.5},     # brown → fox → jumps
    7:  {8: 4.5},   8:  {9: 4.5},     # jumps → over → the
    9:  {10: 4.0, 18: 3.0},           # the  → lazy / nearby
   10:  {11: 4.5, 15: 2.5},           # lazy → dog / cat
   15:  {16: 4.0, 17: 3.0},           # cat  → sits / quietly
   11:  {12: 4.5},  16: {17: 4.0},    # dog → .   sits → quietly
   17:  {18: 4.0, 12: 3.0},  18: {12: 4.5},
   12:  {EOS_ID: 5.0},
}

def fake_forward(context):
    last = context[-1]
    z    = [-5.0] * VOCAB_SIZE
    for tok, val in TRANSITIONS.get(last, {}).items():
        z[tok] = val + random.gauss(0, 0.2)
    z[EOS_ID] = max(z[EOS_ID], -2.0)
    return z

# ── PREFILL ───────────────────────────────────────────────────────────────────
def prefill(prompt_ids, kv_cache):
    """
    Process the full prompt in ONE parallel forward pass.
    This is fast because the GPU can process all prompt tokens simultaneously.
    Latency metric: Time-To-First-Token (TTFT)
    """
    print("  [PREFILL] Processing", len(prompt_ids),
          "prompt tokens in parallel...")
    for tid in prompt_ids:
        kv_cache.store(tid)
    time.sleep(0.04 * len(prompt_ids))   # simulate prefill compute
    return fake_forward(prompt_ids)

# ── DECODE LOOP ───────────────────────────────────────────────────────────────
def decode_loop(init_logits, kv_cache, max_new=10):
    """
    Generate tokens ONE AT A TIME — the autoregressive bottleneck.
    Each step: attend to KV cache → sample → append → repeat.
    Latency metric: Tokens-Per-Second (TPS)
    """
    generated, logits, ctx = [], init_logits, list(kv_cache.tokens)
    print()
    print("  [DECODE] Generating tokens (max =", max_new, ")")
    print(f"  {'Step':>4}  {'Token':>10}  {'Attn Span':>10}  Status")
    print("  " + "-" * 44)

    for step in range(max_new):
        next_id   = sample(logits)
        tok_text  = VOCAB.get(next_id, f"[{next_id}]")
        span      = kv_cache.attend()
        status    = "STOP (EOS)" if next_id == EOS_ID else "continue"
        print(f"  {step+1:>4}  {tok_text:>10}  {span:>10} tok  {status}")

        if next_id == EOS_ID:
            break

        generated.append(next_id)
        ctx.append(next_id)
        kv_cache.store(next_id)
        time.sleep(0.015)
        logits = fake_forward(ctx)

    return generated

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AUTOREGRESSIVE GENERATION LOOP")
    print("=" * 55)

    prompt_ids  = [BOS_ID, 3]   # <BOS> The
    prompt_text = " ".join(VOCAB[t] for t in prompt_ids)
    print("  Prompt:", repr(prompt_text))

    kv = KVCache()

    t0 = time.time()
    init_logits = prefill(prompt_ids, kv)
    ttft = time.time() - t0
    print(f"  Prefill done: {ttft*1000:.0f} ms  |  KV cache: {kv.size()} tokens")

    t1   = time.time()
    gen  = decode_loop(init_logits, kv, max_new=12)
    tdec = time.time() - t1

    full  = " ".join(VOCAB.get(t, "?") for t in prompt_ids + gen)
    gen_t = " ".join(VOCAB.get(t, "?") for t in gen)

    print()
    print("  ── RESULTS ──────────────────────────────────────")
    print("  Prompt:   ", repr(prompt_text))
    print("  Generated:", repr(gen_t))
    print("  Full text:", repr(full))
    print()
    tps = len(gen) / tdec if tdec > 0 else 0
    print(f"  TTFT:          {ttft*1000:.0f} ms")
    print(f"  Decode time:   {tdec*1000:.0f} ms  ({len(gen)} tokens)")
    print(f"  TPS:           {tps:.1f} tokens/sec")
    print(f"  KV cache size: {kv.size()} tokens  |  Reuse hits: {kv.hits}")
    print()
    print("  Key observations:")
    print("    1. Prefill processed the whole prompt in one pass (fast)")
    print("    2. Decode generated each token sequentially (slow)")
    print("    3. KV cache prevented recomputing attention from scratch each step")
    print("    4. Attention span grew from", len(prompt_ids),
          "->", kv.size(), "tokens over generation")
''',
    },


    # ─────────────────────────────────────────────────────────────────────────
    "3. INT8 Quantisation — Weights and Dequantisation": {
        "description": "Demonstrate INT8 quantisation: quantise FP32 weights, dequantise, and measure error",
        "runnable": True,
        "code": '''
import random, math

# ── QUANTISE ──────────────────────────────────────────────────────────────────
def quantise_int8(weights):
    """
    Symmetric per-tensor INT8 quantisation.

    Algorithm:
        scale     = max(|W|) / 127        maps FP32 range → INT8 range [-127, 127]
        W_int8    = round(W / scale)      quantise
        W_recon   = W_int8 * scale        dequantise (approximate original)

    Memory: FP32=4 bytes/param → INT8=1 byte/param → 75% reduction
    """
    if not weights:
        return [], 1.0
    max_abs = max(abs(w) for w in weights)
    if max_abs == 0:
        return [0] * len(weights), 1.0
    scale = max_abs / 127
    q = [max(-128, min(127, round(w / scale))) for w in weights]
    return q, scale

def dequantise_int8(q, scale):
    """Reconstruct approximate FP32 weights: W_approx = q * scale"""
    return [v * scale for v in q]

def quantise_int4(weights):
    """INT4: range -7 to 7 (4 bits). 4× memory reduction vs FP32."""
    if not weights:
        return [], 1.0
    max_abs = max(abs(w) for w in weights)
    if max_abs == 0:
        return [0] * len(weights), 1.0
    scale = max_abs / 7
    q = [max(-8, min(7, round(w / scale))) for w in weights]
    return q, scale

def measure_error(original, reconstructed):
    """Return mean absolute error and relative error (% of weight range)."""
    errs  = [abs(o - r) for o, r in zip(original, reconstructed)]
    mae   = sum(errs) / len(errs)
    rng   = max(original) - min(original)
    rel   = (mae / rng * 100) if rng > 0 else 0
    return mae, max(errs), rel

# ── DEMONSTRATION ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    N = 20
    fp32 = [random.gauss(0, 0.5) for _ in range(N)]

    print("=" * 65)
    print("  INT8 QUANTISATION DEMONSTRATION")
    print("=" * 65)
    print(f"  Weights: {N} parameters")
    print(f"  Range:   [{min(fp32):.4f}, {max(fp32):.4f}]")
    print(f"  Sample:  {[round(w,4) for w in fp32[:5]]} ...")

    # ── INT8 ─────────────────────────────────────────────────────────────────
    q8, s8   = quantise_int8(fp32)
    recon8   = dequantise_int8(q8, s8)
    mae8, mx8, rel8 = measure_error(fp32, recon8)

    print()
    print(f"  Scale factor:  {s8:.6f}  (= max_abs / 127)")
    print(f"  Sample INT8:   {q8[:5]} ...")
    print()
    print(f"  Memory per weight:  FP32=4 bytes  INT8=1 byte  Saving=75%")

    # ── INT4 ─────────────────────────────────────────────────────────────────
    q4, s4   = quantise_int4(fp32)
    recon4   = dequantise_int8(q4, s4)
    mae4, mx4, rel4 = measure_error(fp32, recon4)

    # ── Side-by-side table ───────────────────────────────────────────────────
    print()
    print(f"  {'Idx':>3}  {'FP32':>12}  {'INT8':>6}  {'Recon':>12}  {'Error':>10}")
    print("  " + "-" * 50)
    for i in range(min(10, N)):
        err = abs(fp32[i] - recon8[i])
        print(f"  {i:>3}  {fp32[i]:>12.6f}  {q8[i]:>6}  {recon8[i]:>12.6f}  {err:>10.6f}")

    # ── Precision comparison ─────────────────────────────────────────────────
    print()
    print("  PRECISION COMPARISON")
    print()
    print(f"  {'Format':<14} {'Bits':>6}  {'Memory(20w)':>12}  {'Rel Error':>10}")
    print("  " + "-" * 50)
    rows = [
        ("FP32",          32,  4*N,   0.0),
        ("FP16",          16,  2*N,   0.01),
        ("INT8",           8,  1*N,   rel8),
        ("INT4",           4,  N//2,  rel4),
    ]
    for fmt, bits, mem, rel in rows:
        print(f"  {fmt:<14} {bits:>6}  {mem:>10} bytes  {rel:>9.3f}%")

    print()
    print(f"  INT8 error ({rel8:.3f}%) → <1% perplexity impact in practice")
    print(f"  INT4 error ({rel4:.3f}%) → 2-5% perplexity impact (worse on small models)")
    print()
    print("  Rule: 70B model @ INT4  often beats  7B model @ FP16  on same hardware")
''',
    },


    # ─────────────────────────────────────────────────────────────────────────
    "4. Speculative Decoding — Draft and Verify": {
        "description": "Simulate the speculative decoding accept/reject protocol and measure effective speedup",
        "runnable": True,
        "code": '''
import random, math, time

VOCAB_SIZE = 12

# ── SOFTMAX ───────────────────────────────────────────────────────────────────
def softmax(logits):
    m = max(logits)
    e = [math.exp(z - m) for z in logits]
    s = sum(e)
    return [x / s for x in e]

def sample_from(probs):
    r, cum = random.random(), 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1

# ── SIMULATED MODELS ─────────────────────────────────────────────────────────
def draft_logits(context):
    """Fast small model — cheap but imperfect."""
    z = [random.gauss(0, 1.5) for _ in range(VOCAB_SIZE)]
    for i in [2, 5, 7]: z[i] += 2.0   # draft model biases
    return z

def target_logits(context):
    """Slow large model — expensive but authoritative."""
    z = [random.gauss(0, 1.0) for _ in range(VOCAB_SIZE)]
    for i in [2, 5, 8]: z[i] += 2.0   # target model biases (slightly different)
    return z

# ── ACCEPT / REJECT ───────────────────────────────────────────────────────────
def accept_reject(draft_tok, P_draft, P_target):
    """
    Speculative decoding acceptance criterion (Leviathan et al., 2023).

    Accept probability:  min(1, P_target(t) / P_draft(t))

    On rejection, resample from corrected distribution:
        P_corrected = normalise(max(0, P_target - P_draft))

    GUARANTEE: Accepted tokens are identically distributed to P_target.
    No quality loss — only a speed trade-off.
    """
    p_d = P_draft[draft_tok]
    p_t = P_target[draft_tok]
    accept_prob = min(1.0, p_t / (p_d + 1e-10))

    if random.random() < accept_prob:
        return True, draft_tok
    else:
        corrected = [max(0.0, pt - pd) for pt, pd in zip(P_target, P_draft)]
        total = sum(corrected)
        if total < 1e-10:
            corrected = P_target
            total = 1.0
        normed = [c / total for c in corrected]
        return False, sample_from(normed)

# ── SPECULATIVE DECODE ROUND ─────────────────────────────────────────────────
def spec_round(context, K):
    """
    One speculative decoding round:
      1. Draft model proposes K tokens  (K × T_draft)
      2. Target model verifies all K    (1 × T_target, parallel)
      3. Accept/reject in order, stop at first rejection
    """
    # Step 1: Draft
    draft_toks, draft_probs, ctx = [], [], list(context)
    for _ in range(K):
        dz  = draft_logits(ctx)
        pd  = softmax(dz)
        tok = sample_from(pd)
        draft_toks.append(tok)
        draft_probs.append(pd)
        ctx.append(tok)

    # Step 2: Target verify (one parallel pass)
    target_probs, vctx = [], list(context)
    for tok in draft_toks:
        tz = target_logits(vctx)
        target_probs.append(softmax(tz))
        vctx.append(tok)

    # Step 3: Accept / reject
    accepted = []
    for i, dt in enumerate(draft_toks):
        ok, final = accept_reject(dt, draft_probs[i], target_probs[i])
        accepted.append(final)
        if not ok:
            break

    return accepted, len(accepted), K

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SPECULATIVE DECODING DEMONSTRATION")
    print("=" * 60)

    K         = 4       # draft tokens proposed per round
    N_TOTAL   = 40      # total tokens to generate
    T_LARGE   = 0.040   # large model latency per pass (sec)
    T_DRAFT   = 0.005   # draft model latency per token (sec)

    context = [0, 1, 2]

    # Standard decoding baseline
    std_time = N_TOTAL * T_LARGE
    std_tps  = N_TOTAL / std_time

    print(f"  Standard decoding: {N_TOTAL} tokens x {T_LARGE*1000:.0f}ms/pass")
    print(f"    Total:      {std_time*1000:.0f} ms")
    print(f"    Throughput: {std_tps:.1f} tok/sec")

    # Speculative decoding
    print()
    print(f"  Speculative decoding (K={K}):")
    generated, rounds, accepted_tot, proposed_tot = 0, 0, 0, 0

    while generated < N_TOTAL:
        acc, n_acc, n_prop = spec_round(context, K)
        generated     += n_acc
        rounds        += 1
        accepted_tot  += n_acc
        proposed_tot  += n_prop
        context.extend(acc)

    alpha    = accepted_tot / proposed_tot
    spec_t   = rounds * (T_LARGE + K * T_DRAFT)
    spec_tps = accepted_tot / spec_t
    speedup  = spec_tps / std_tps

    print(f"    Proposals:      {proposed_tot}")
    print(f"    Accepted:       {accepted_tot}")
    print(f"    Accept rate:    {alpha:.0%}")
    print(f"    LM passes:      {rounds}  (vs {N_TOTAL} for standard)")
    print(f"    Total:          {spec_t*1000:.0f} ms")
    print(f"    Throughput:     {spec_tps:.1f} tok/sec")
    print(f"    Speedup:        {speedup:.2f}x")

    print()
    print("  ┌────────────────────────┬──────────────┬──────────────┐")
    print("  │ Method                 │   Time (ms)  │   Tok/sec    │")
    print("  ├────────────────────────┼──────────────┼──────────────┤")
    print(f"  │ Standard               │ {std_time*1000:>12.0f} │ {std_tps:>12.1f} │")
    print(f"  │ Speculative (K={K})     │ {spec_t*1000:>12.0f} │ {spec_tps:>12.1f} │")
    print("  └────────────────────────┴──────────────┴──────────────┘")
    print()
    print(f"  Speedup: {speedup:.2f}x  |  Accept rate: {alpha:.0%}  |  K={K}")
    print()
    print("  Key insight: accepted tokens are IDENTICALLY distributed to")
    print("  what the large model would have produced — zero quality loss.")
    print("  Higher acceptance rate → larger speedup. Draft model quality")
    print("  is the main bottleneck.")
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
# ─────────────────────────────────────────────────────────────────────────────

for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if "tok_step_status"  not in st.session_state:
        st.session_state.tok_step_status  = {}
    if "tok_step_outputs" not in st.session_state:
        st.session_state.tok_step_outputs = {}

    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**{op_data['description']}**")
            st.markdown("---")
            st.code(op_data["code"], language=op_data.get("language", "python"))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 1300
    try:
        from Deep_Learning.visuals.inference_optimisation import (
            INFERENCE_VISUAL_HTML,
            INFERENCE_VISUAL_HEIGHT,
        )
        visual_html   = INFERENCE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = INFERENCE_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[10_Inference_Optimisation.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    OPERATIONS,
    }