"""
Modern Architectures — LLaMA, Mistral, and Phi
================================================

The original Transformer paper left many design choices open. Over 2021–2024,
a set of architectural innovations converged into what is now the de-facto
standard decoder-only LLM blueprint: Pre-Norm with RMSNorm, SwiGLU FFN,
RoPE, and GQA. LLaMA, Mistral, and the Phi family each made distinct
engineering choices that are worth understanding deeply.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Modern LLM Architectures — LLaMA, Mistral, Phi"
DISPLAY_NAME = "05 · Modern Architectures"
ICON         = "🦙"
SUBTITLE     = "LLaMA, Mistral, and Phi Design Choices"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### From the Original Transformer to the Modern Blueprint

The 2017 "Attention Is All You Need" Transformer differed from today's LLMs
in almost every sub-component that wasn't self-attention:

    Component          Original (2017)          Modern LLM (2024)
    ─────────────────────────────────────────────────────────────────
    Normalisation      Post-LayerNorm            Pre-RMSNorm
    Norm function      LayerNorm (γ, β)          RMSNorm (γ only)
    FFN activation     ReLU                      SwiGLU / GeGLU
    FFN structure      2-layer (W1, W2)           3-matrix gated (W1, W2, Wg)
    Positional enc.    Sinusoidal (fixed)         RoPE (relative, rotary)
    Attention heads    MHA (h K/V per Q)          GQA (g K/V shared per group)
    Biases             Yes (all projections)      No (weight-only)
    Vocabulary         ~32k WordPiece             32k–128k BPE/SP
    ─────────────────────────────────────────────────────────────────

Each change was motivated by a specific problem observed at scale. This module
traces those changes through LLaMA 1→2→3, Mistral, and Phi.


### Innovation 1 — RMSNorm (replacing LayerNorm)

**LayerNorm** (Ba et al., 2016) normalises across the feature dimension:

    LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β

where μ and σ are the mean and variance computed over the d_model features
of a single token vector, and γ, β are learned scale + shift parameters.

**RMSNorm** (Zhang & Sennrich, 2019) removes the mean-centring entirely:

    RMSNorm(x) = γ · x / RMS(x)     where RMS(x) = √(mean(x²) + ε)

The motivation is both empirical and theoretical:
    •   Re-centring (subtracting μ) was found to contribute little to
        training stability at large scale.
    •   Without mean subtraction, RMSNorm is ~10–40% faster (one fewer
        pass over the data; simpler backward pass).
    •   RMSNorm also removes the shift parameter β, reducing parameter count
        by d_model per norm (trivial in total, but cleaner).

Used by: LLaMA 1/2/3, Mistral, Falcon, Qwen, DeepSeek, Gemma.
Not used by: GPT-2/3 (LayerNorm), original Transformer (LayerNorm).


    **Diagram 1 — LayerNorm vs RMSNorm:**

    LAYERNORM vs RMSNORM
    ════════════════════════════════════════════════════════════════

    Input vector x = [x₁, x₂, …, x_d]

    LayerNorm:
        μ    = mean(x)                    ← requires a full pass to compute mean
        σ²   = mean((x - μ)²)            ← another pass for variance
        x̂ᵢ  = (xᵢ - μ) / sqrt(σ² + ε)  ← re-centre AND scale
        out  = γ ⊙ x̂ + β               ← scale (γ) and shift (β)
        Parameters: γ ∈ ℝ^d, β ∈ ℝ^d

    RMSNorm:
        rms  = sqrt(mean(x²) + ε)        ← ONE pass, no mean subtraction
        x̂ᵢ  = xᵢ / rms                  ← scale only, no re-centring
        out  = γ ⊙ x̂                    ← scale only
        Parameters: γ ∈ ℝ^d  (no β)

    Speed: RMSNorm ≈ 10–40% faster than LayerNorm on modern hardware
    Quality: Empirically indistinguishable at LLM scale


### Innovation 2 — SwiGLU Feed-Forward Networks

The standard FFN is:

    FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂     (ReLU)
    FFN(x) = GELU(x·W₁ + b₁) · W₂ + b₂        (GPT-2 style)

**GLU (Gated Linear Unit):** Dauphin et al. (2017) introduced gating:

    GLU(x) = (x·W₁) ⊙ σ(x·W₂)

where σ is sigmoid and ⊙ is element-wise multiplication. The second branch
acts as a soft "gate" that controls how much of the first branch passes through.

**SwiGLU** (Noam Shazeer, 2020) replaces sigmoid with SiLU (Swish):

    SiLU(x) = x · σ(x)     (smooth, non-monotone activation)

    SwiGLU(x) = SiLU(x·W₁) ⊙ (x·Wg) · W₂

Three matrices instead of two:
    W₁ ∈ ℝ^(d_model × d_ff)    "up" projection
    Wg ∈ ℝ^(d_model × d_ff)    "gate" projection
    W₂ ∈ ℝ^(d_ff × d_model)    "down" projection

To maintain the same total parameter count as a 4× standard FFN, d_ff is
reduced to 8/3 × d_model ≈ 2.67 × d_model (sometimes rounded to nearest 256).

**Why does gating help?**
The gate Wg effectively learns a per-neuron "switch" — it amplifies
information that passes through W₁ when that information is relevant, and
suppresses it otherwise. This provides a form of dynamic computation that
vanilla two-layer MLPs lack. In practice, SwiGLU outperforms ReLU/GELU
at every scale that has been tested.

Used by: LLaMA 1/2/3, PaLM, Mistral, Qwen, DeepSeek, Gemma.


    **Diagram 2 — Standard FFN vs SwiGLU:**

    STANDARD FFN vs SWIGLU
    ════════════════════════════════════════════════════════════════

    Standard FFN (ReLU/GELU):
                           W₁
    x ─────────────────────────► [ d_ff neurons ]
                                         │
                                    activation(·)
                                         │  W₂
                                         └────────► output ∈ ℝ^d

    SwiGLU:
                           W₁
    x ──────────────────────────► [ d_ff neurons ]
    │                                     │
    │                                  SiLU(·)
    │                                     │
    │                      Wg             ⊙ (element-wise multiply)
    └──────────────────────────► [ d_ff neurons ]         │
                                                     W₂   │
                                         ┌───────────────► output ∈ ℝ^d
                                         │
                              "gate controls information flow"

    • W₁ branch: transforms features
    • Wg branch: learns when to let features through
    • ⊙:         dynamic, content-dependent gating
    • W₂:        project back to d_model


### LLaMA 1 (Touvron et al., February 2023)

LLaMA-1 was a landmark release — not for being the largest model, but for
being the first openly released model to match (or exceed) GPT-3 quality
while being dramatically more efficient. It packaged together every
architectural improvement known at the time:

**Architecture:**
    •   Pre-RMSNorm (not Post-LayerNorm)
    •   SwiGLU FFN (not ReLU/GELU)
    •   RoPE positional encoding (not sinusoidal or learned)
    •   No biases anywhere (all Linear layers are weight-only)
    •   MHA attention (GQA was added in LLaMA-2)
    •   SentencePiece BPE tokenizer, 32k vocabulary

**Training:**
    •   Trained only on publicly available data (no proprietary data)
    •   1T–1.4T tokens depending on model size
    •   AdamW optimiser, cosine LR schedule
    •   All models trained to "over-train" past the Chinchilla optimal
        point — smaller models trained longer are more inference-efficient

**Model sizes:**
    7B, 13B, 33B, 65B parameters

**Key insight from LLaMA-1:** The Chinchilla scaling laws (Hoffmann et al.,
2022) defined compute-optimal training — but that optimises for training
compute, not inference cost. If you are going to run a model billions of times,
you want the smallest model that achieves a target quality, not the one trained
with optimal compute. Training a 7B model on 1T tokens is more
inference-efficient than training a 70B model on 100B tokens, even if the
latter has lower perplexity per FLOP of training.


### LLaMA 2 (Touvron et al., July 2023)

LLaMA-2 added two critical improvements:

**1. Grouped-Query Attention (GQA) for the 70B model:**
    •   34B and 70B use GQA with g=8 KV heads
    •   7B and 13B still use MHA (at these sizes, KV-cache is not the bottleneck)
    •   70B: 64 Q heads, 8 KV heads → 8× KV-cache reduction
    •   Converted from an MHA checkpoint via mean-pooling + uptraining

**2. Extended context window:**
    •   LLaMA-1: 2048 tokens
    •   LLaMA-2: 4096 tokens
    •   Achieved by training with 4k-length sequences and adjusting RoPE base

**3. Safety-focused fine-tuning:**
    •   Released with both base and chat variants
    •   Chat models fine-tuned with SFT + RLHF (PPO) + rejection sampling
    •   Introduced Ghost Attention (GAtt) for multi-turn consistency

**Model sizes:** 7B, 13B, 34B, 70B


### LLaMA 3 (Meta AI, April 2024)

LLaMA-3 represented a step change in training data quality and scale:

**Architecture changes from LLaMA-2:**
    •   All sizes use GQA (even 8B): 32 Q heads, 8 KV heads
    •   New tiktoken-based tokenizer: 128k vocabulary (up from 32k)
        → much better non-English efficiency, better code tokenization
    •   Extended context: 8k (base), up to 128k with fine-tuning
    •   RoPE base frequency increased to 500,000 (enables long context)

**Training data:**
    •   15T tokens of training data (10× LLaMA-2)
    •   >5% code in training mix (vs ~1% in LLaMA-2)
    •   Careful data curation: deduplication, quality filtering

**Model sizes:** 8B, 70B (405B released later as LLaMA-3.1)

**Key change — vocabulary size:**
    LLaMA-2 used 32k tokens. LLaMA-3 uses 128k. This means:
    •   Embedding table: 128k × 4096d = 524M params (vs 131M for LLaMA-2)
    •   Much better multilingual and code tokenization efficiency
    •   Average English fertility drops from ~1.4× to ~1.2×


    **Diagram 3 — LLaMA Architecture Evolution:**

    LLAMA ARCHITECTURE EVOLUTION
    ════════════════════════════════════════════════════════════════

    Component          LLaMA-1       LLaMA-2       LLaMA-3 (8B)
    ─────────────────────────────────────────────────────────────
    Norm               RMSNorm       RMSNorm       RMSNorm
    FFN                SwiGLU        SwiGLU        SwiGLU
    Position enc.      RoPE          RoPE          RoPE
    RoPE base          10,000        10,000        500,000 ←
    Attention          MHA           MHA/GQA       GQA (all) ←
    Context            2048          4096          8192 ←
    Vocabulary         32k           32k           128k ←
    Biases             None          None          None
    Tokenizer          SentencePiece SentencePiece tiktoken ←
    Training tokens    1T–1.4T       2T            15T ←
    Public data only   Yes           Yes           Yes
    ─────────────────────────────────────────────────────────────


### Mistral 7B (Mistral AI, October 2023)

Mistral 7B is notable for introducing two architectural innovations:

**1. Sliding Window Attention (SWA):**
    •   Each token attends only to the W=4096 most recent tokens
    •   KV-cache is capped at W entries per layer — O(W) not O(T) memory
    •   Information flows beyond W tokens through N layers of sliding windows:
        with N=32 layers, effective receptive field = 32 × 4096 ≈ 128k tokens
    •   In practice: a rolling buffer KV-cache, evicting old entries

**2. GQA with n_kv_heads=8:**
    •   32 Q heads, 8 KV heads
    •   Combined with SWA: fixed KV-cache size regardless of generation length

**Architecture:**
    d_model=4096, n_heads=32, n_kv_heads=8, n_layers=32
    d_ff=14336 (3.5× instead of 4× — slightly unusual ratio)
    RMSNorm, SwiGLU, RoPE, 32k vocab

**Performance:** Despite being 7B parameters, Mistral-7B outperformed
LLaMA-2 13B on most benchmarks — attributed to high-quality training data
filtering and the architectural efficiency gains.

**Mistral Mixture of Experts (Mixtral 8×7B):** Mistral later released a
Sparse MoE model where each Transformer block has 8 expert FFNs, with
a router selecting 2 per token. Active parameters = ~13B, total = 46B.


### Phi Family (Microsoft Research)

The Phi models (Gunasekar et al., 2023 onwards) pursued a different axis:
**maximum quality per parameter through data quality**, not scale.

**Phi-1 (1.3B):**
    •   Trained on "textbook-quality" synthetic Python code generated by GPT-4
    •   1B parameters outperformed 7B models on Python coding benchmarks
    •   Key finding: data quality can substitute for model size

**Phi-1.5 (1.3B) and Phi-2 (2.7B):**
    •   Extended to general reasoning using "textbook-style" synthetic data
    •   Phi-2 matched or outperformed 7B models on many reasoning benchmarks
    •   Used for: rapid experimentation, on-device inference

**Phi-3 Mini (3.8B):**
    Architecture:
        d_model=3072, n_heads=32, n_kv_heads=32 (MHA), n_layers=32
        Context window: 4k (standard), 128k (long-context variant)
        Vocabulary: 32k (tiktoken-compatible)
        No SWA, standard MHA (smaller model → KV-cache less critical)

    Training data:
        3.3T tokens of heavily filtered web text + synthetic data
        "Filtered web": text that reads like books/tutorials, not spam

**Phi-3 Medium (14B) and Phi-3 Small (7B):**
    Both use GQA. Phi-3 Medium matches GPT-3.5 class performance.

**The Phi thesis:** For a given inference budget (latency, memory), the
optimal strategy may be a smaller model trained on higher-quality data,
rather than a larger model trained on noisier web data. This challenges the
assumption that scale is always the primary lever.


    **Diagram 4 — Modern Architecture Comparison:**

    MODERN ARCHITECTURE COMPARISON
    ════════════════════════════════════════════════════════════════

    Model           Params  Layers  d_model  Heads  KV heads  FFN      Vocab
    ──────────────────────────────────────────────────────────────────────────
    LLaMA-2 7B        7B     32     4096      32      32       SwiGLU   32k
    LLaMA-2 70B      70B     80     8192      64       8       SwiGLU   32k
    LLaMA-3 8B        8B     32     4096      32       8       SwiGLU  128k
    LLaMA-3 70B      70B     80     8192      64       8       SwiGLU  128k
    Mistral 7B        7B     32     4096      32       8       SwiGLU   32k
    Mixtral 8×7B     46B*    32     4096      32       8       MoE-SGL  32k
    Phi-3 Mini       3.8B    32     3072      32      32       SwiGLU   32k
    Phi-3 Medium     14B     40     5120      40      10       SwiGLU   32k
    Gemma 7B          8B     28     3072      16       1       GeGLU    256k
    Qwen-1.5 7B       7B     32     4096      32       4       SwiGLU  152k
    ──────────────────────────────────────────────────────────────────────────
    * Mixtral: 46B total, ~13B active per token (2 of 8 experts)

    Shared by ALL modern LLMs above:
    ✓ RMSNorm (Pre-Norm)
    ✓ Gated FFN (SwiGLU or GeGLU)
    ✓ RoPE
    ✓ No biases in attention/FFN projections
    ✓ GQA or MQA (all except Phi-3 Mini)


### The No-Bias Decision

One quiet change that deserves attention: modern LLMs remove bias terms from
all attention and FFN projection matrices. The Linear layers are weight-only.

Reasons:
    1.  **Memory**: eliminates d_model bias vectors per projection — small
        savings but cleaner accounting.
    2.  **Training stability**: biases can cause issues with gradient flow
        at scale, particularly in the early training stages.
    3.  **Distributed training**: weight tensors shard cleanly across GPUs;
        bias vectors are 1D and harder to shard consistently.
    4.  **Empirical**: no quality degradation has been found from removing biases
        in LLMs at scale.

Note: Layer normalisation retains its scale parameter γ (and sometimes β).
Only the attention and FFN projections drop biases.


### Reading a model's config.json

Every HuggingFace model ships a config.json that encodes the architecture.
Knowing what to look for lets you reconstruct the full architecture from 10
lines of JSON:

    {
      "hidden_size":             4096,     → d_model
      "intermediate_size":      11008,     → d_ff  (SwiGLU ratio ≈ 2.67×)
      "num_hidden_layers":          32,    → N layers
      "num_attention_heads":        32,    → n_heads (Q heads)
      "num_key_value_heads":         8,    → n_kv_heads (GQA groups)
      "max_position_embeddings":  4096,    → max context window
      "rms_norm_eps":            1e-05,    → ε in RMSNorm
      "rope_theta":             10000.0,   → RoPE base frequency
      "vocab_size":             32000,     → vocabulary size
      "hidden_act":            "silu",     → SwiGLU activation (SiLU gate)
    }

This is a LLaMA-2 7B config. You can immediately reconstruct:
    •   GQA with 4 Q heads per KV group (32/8=4)
    •   d_ff/d_model ratio = 11008/4096 ≈ 2.69 ≈ 8/3  (SwiGLU ratio)
    •   RMSNorm, not LayerNorm
    •   RoPE with standard base 10k
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Architecture Component Decision Guide

| Scenario                              | Recommended choice         | Why                                         |
|---------------------------------------|----------------------------|---------------------------------------------|
| Normalisation                         | RMSNorm (Pre-Norm)         | 10-40% faster, equally stable               |
| FFN activation                        | SwiGLU                     | Best empirical quality at all tested scales |
| Positional encoding                   | RoPE                       | Relative, extensible, KV-cache compatible   |
| Attention (large model, inference)    | GQA (g=4–8)                | Near-MHA quality, 4–8× KV-cache reduction   |
| Attention (small model, <4B)          | MHA or GQA g=2             | KV-cache not bottleneck at small scale      |
| Long context                          | RoPE + YaRN or LongRoPE    | Smooth extrapolation beyond training length |
| Fixed memory budget at inference      | GQA + SWA (Mistral style)  | Capped KV-cache regardless of gen length    |
| Maximum quality/param (edge/mobile)   | Phi-style data curation    | High-quality synthetic data > scale         |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "RMSNorm vs LayerNorm — Speed and Correctness": {
        "description": "Implement both norms from scratch, verify numerical equivalence of the normalisation effect, and benchmark speed.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
RMSNORM vs LAYERNORM — IMPLEMENTATION AND COMPARISON
================================================================================

Implements both from scratch, verifies the normalisation behaviour,
and benchmarks the speed difference.
================================================================================
"""

import math
import time
import torch
import torch.nn as nn


# ── From-scratch implementations ──────────────────────────────────────────────

class LayerNormScratch(nn.Module):
    """LayerNorm with explicit mean-centring and variance scaling."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNormScratch(nn.Module):
    """RMSNorm: no mean subtraction, no shift parameter beta."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        rms    = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm


# ── Verification ──────────────────────────────────────────────────────────────

def check_normalisation_properties():
    d = 512
    torch.manual_seed(0)

    x = torch.randn(4, 32, d) * 10 + 5   # mean ≠ 0, std ≠ 1

    ln = LayerNormScratch(d)
    rn = RMSNormScratch(d)

    with torch.no_grad():
        y_ln = ln(x)
        y_rn = rn(x)

    print("=" * 60)
    print("  NORMALISATION PROPERTIES CHECK")
    print("=" * 60)
    print(f"\n  Input  — mean: {x.mean():.3f},  std: {x.std():.3f}")
    print()

    # LayerNorm should produce zero-mean, unit-variance outputs (before γ/β)
    # Since γ=1 and β=0 at init, output should be normalised
    print(f"  LayerNorm output:")
    print(f"    mean per token: {y_ln.mean(dim=-1).abs().mean():.6f}  (≈0 expected)")
    print(f"    std  per token: {y_ln.std(dim=-1).mean():.6f}         (≈1 expected)")

    print()
    print(f"  RMSNorm output:")
    # RMSNorm does NOT zero-mean (no μ subtraction), but normalises the L2 norm
    mean_abs = y_rn.mean(dim=-1).abs().mean()
    rms_val  = y_rn.pow(2).mean(dim=-1).sqrt().mean()
    print(f"    mean per token: {mean_abs:.6f}  (NOT necessarily 0 — no centring)")
    print(f"    RMS  per token: {rms_val:.6f}       (≈1 expected — this is what's normalised)")

    print()
    print("  Key difference: LayerNorm centres AND scales; RMSNorm only scales.")
    print("  At LLM scale, the centring provides negligible benefit.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(name: str, module: nn.Module, x: torch.Tensor, n_iter: int = 200):
    # Warmup
    for _ in range(10):
        _ = module(x)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = module(x)
        _ = y.sum()   # prevent dead-code elimination
    elapsed = (time.perf_counter() - t0) / n_iter * 1000
    return elapsed


if __name__ == "__main__":
    # 1. Properties check
    check_normalisation_properties()

    # 2. Speed benchmark
    print()
    print("=" * 60)
    print("  SPEED BENCHMARK")
    print("=" * 60)

    for d_model in [512, 2048, 4096]:
        x = torch.randn(8, 512, d_model)  # batch=8, T=512

        ln      = LayerNormScratch(d_model)
        rn      = RMSNormScratch(d_model)
        ln_pt   = nn.LayerNorm(d_model)    # PyTorch built-in (fused kernel)

        t_ln    = benchmark("LayerNorm (scratch)", ln,    x)
        t_rn    = benchmark("RMSNorm   (scratch)", rn,    x)
        t_ln_pt = benchmark("LayerNorm (PyTorch)", ln_pt, x)

        speedup = t_ln / t_rn
        print(f"\n  d_model = {d_model}")
        print(f"    LayerNorm scratch: {t_ln:.3f} ms")
        print(f"    RMSNorm   scratch: {t_rn:.3f} ms  ({speedup:.2f}× faster)")
        print(f"    LayerNorm PyTorch: {t_ln_pt:.3f} ms  (fused CUDA kernel)")

    print()
    print("  Note: real speedup is larger on GPU with fused RMSNorm kernels")
    print("  (e.g. torch.nn.RMSNorm or Triton kernels used in production).")

    # 3. Parameter count comparison
    print()
    print("=" * 60)
    print("  PARAMETER COUNT")
    print("=" * 60)
    d = 4096
    n_layers = 32
    ln_params = 2 * d    # gamma + beta
    rn_params = 1 * d    # gamma only
    # 2 norms per block (pre-attn, pre-ffn) + 1 final norm
    total_norms = 2 * n_layers + 1
    print(f"  LayerNorm per norm: {ln_params:,} params (γ + β)")
    print(f"  RMSNorm   per norm: {rn_params:,} params (γ only)")
    print(f"  Total norms in {n_layers}-layer model: {total_norms}")
    print(f"  LayerNorm total: {total_norms * ln_params:,}")
    print(f"  RMSNorm   total: {total_norms * rn_params:,}")
    print(f"  Savings: {total_norms * (ln_params - rn_params):,} params "
          f"(negligible but architecturally cleaner)")
''',
    },

    "SwiGLU FFN — Implementation and Ablation": {
        "description": "Implement SwiGLU from scratch alongside standard ReLU and GELU FFNs, and run a small ablation showing forward-pass diversity.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SWIGLU FFN — IMPLEMENTATION AND ABLATION
================================================================================

Implements four FFN variants:
    1. Standard ReLU       (original Transformer)
    2. GELU                (GPT-2)
    3. SwiGLU              (LLaMA, PaLM, Mistral)
    4. GeGLU               (Gemma, some T5 variants)

Shows parameter parity and the gating behaviour.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Activation functions ───────────────────────────────────────────────────────

def swish(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish: x * sigmoid(x). Smooth, non-monotone."""
    return x * torch.sigmoid(x)


# ── FFN Variants ───────────────────────────────────────────────────────────────

class FFN_ReLU(nn.Module):
    """Standard 2-layer FFN with ReLU (original Transformer)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff,    bias=False)
        self.W2 = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class FFN_GELU(nn.Module):
    """2-layer FFN with GELU (GPT-2 style)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff,    bias=False)
        self.W2 = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))


class FFN_SwiGLU(nn.Module):
    """
    SwiGLU FFN (LLaMA, Mistral, PaLM).
    Three matrices: W1 (up), Wg (gate), W2 (down).
    d_ff is set to 8/3 * d_model to maintain param parity with 4× standard FFN.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff,    bias=False)   # "up"
        self.Wg = nn.Linear(d_model, d_ff,    bias=False)   # "gate"
        self.W2 = nn.Linear(d_ff,    d_model, bias=False)   # "down"

    def forward(self, x):
        # Gate: SiLU(x·W1) element-wise multiplied by (x·Wg)
        gate    = swish(self.W1(x))     # activated "up" branch
        content = self.Wg(x)            # gate branch (no activation)
        return self.W2(gate * content)


class FFN_GeGLU(nn.Module):
    """
    GeGLU FFN (Gemma, some T5 variants).
    Same as SwiGLU but uses GELU instead of SiLU for the gate.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff,    bias=False)
        self.Wg = nn.Linear(d_model, d_ff,    bias=False)
        self.W2 = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x):
        gate    = F.gelu(self.W1(x))
        content = self.Wg(x)
        return self.W2(gate * content)


# ── Parameter-parity configuration ────────────────────────────────────────────

def swiglu_d_ff(d_model: int, multiple_of: int = 256) -> int:
    """
    Compute d_ff for SwiGLU that matches the parameter count of a 4× standard FFN.

    Standard FFN params:  2 × d_model × d_ff_standard  (W1 + W2)
                          where d_ff_standard = 4 × d_model

    SwiGLU FFN params:    3 × d_model × d_ff_swiglu  (W1 + Wg + W2)

    Setting equal: 2 × 4 × d² = 3 × d_ff_swiglu × d
    → d_ff_swiglu = 8/3 × d_model

    In practice, round up to the nearest multiple_of for hardware efficiency.
    """
    raw = int(8 / 3 * d_model)
    return ((raw + multiple_of - 1) // multiple_of) * multiple_of


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    d_model       = 4096
    d_ff_standard = 4 * d_model       # 16384  (standard)
    d_ff_swiglu   = swiglu_d_ff(d_model)  # 11008 (SwiGLU parity)

    print("=" * 62)
    print("  FFN VARIANT COMPARISON")
    print(f"  d_model = {d_model}")
    print("=" * 62)

    variants = [
        ("ReLU  (standard)",  FFN_ReLU(d_model,  d_ff_standard), d_ff_standard),
        ("GELU  (GPT-2)",     FFN_GELU(d_model,  d_ff_standard), d_ff_standard),
        ("SwiGLU (LLaMA)",    FFN_SwiGLU(d_model, d_ff_swiglu),  d_ff_swiglu),
        ("GeGLU (Gemma)",     FFN_GeGLU(d_model,  d_ff_swiglu),  d_ff_swiglu),
    ]

    x = torch.randn(2, 16, d_model)

    print(f"\n  {'Variant':<22} {'d_ff':>8}  {'Params':>12}  {'d_ff/d':>8}  {'Output OK':>10}")
    print(f"  {'':─<22} {'':─>8}  {'':─>12}  {'':─>8}  {'':─>10}")
    for name, ffn, d_ff in variants:
        n_params = sum(p.numel() for p in ffn.parameters())
        out      = ffn(x)
        ok       = "✓" if out.shape == x.shape else "✗"
        ratio    = d_ff / d_model
        print(f"  {name:<22} {d_ff:>8,}  {n_params:>12,}  {ratio:>8.2f}×  {ok:>10}")

    # Gating behaviour analysis
    print()
    print("=" * 62)
    print("  SWIGLU GATING BEHAVIOUR")
    print("=" * 62)
    print()
    print("  The gate controls which 'features' pass through to the down-projection.")
    print()

    ffn_swiglu = FFN_SwiGLU(d_model, d_ff_swiglu)
    x_single   = torch.randn(1, 1, d_model)

    with torch.no_grad():
        gate_values = swish(ffn_swiglu.W1(x_single)).squeeze()   # (d_ff,)
        content_values = ffn_swiglu.Wg(x_single).squeeze()        # (d_ff,)
        gated = gate_values * content_values                       # element-wise

    active_pct = (gate_values.abs() > 0.1).float().mean() * 100
    strong_pct = (gate_values.abs() > 0.5).float().mean() * 100

    print(f"  Gate statistics (after SiLU activation):")
    print(f"    d_ff neurons:          {d_ff_swiglu:,}")
    print(f"    Neurons with |gate|>0.1: {active_pct:.1f}%  (moderately active)")
    print(f"    Neurons with |gate|>0.5: {strong_pct:.1f}%  (strongly active)")
    print(f"    Dead neurons (gate≈0):   {100-active_pct:.1f}%")
    print()
    print(f"  Mean gate value:  {gate_values.mean():.4f}")
    print(f"  Std  gate value:  {gate_values.std():.4f}")
    print()
    print("  The gate acts as a learned, input-dependent sparsity mask.")
    print("  Unlike ReLU (hard 0/1), SwiGLU gates are soft and continuous.")

    # SwiGLU d_ff computation demo
    print()
    print("=" * 62)
    print("  SWIGLU d_ff COMPUTATION (parameter parity)")
    print("=" * 62)
    print()
    for d in [512, 768, 1024, 2048, 4096, 8192]:
        d_ff_std = 4 * d
        d_ff_sgl = swiglu_d_ff(d)
        params_std = 2 * d * d_ff_std   # W1 + W2
        params_sgl = 3 * d * d_ff_sgl   # W1 + Wg + W2
        print(f"  d={d:5d}:  standard 4× d_ff={d_ff_std:6d} ({params_std:>10,} params)  "
              f"SwiGLU d_ff={d_ff_sgl:6d} ({params_sgl:>10,} params)  "
              f"ratio={params_sgl/params_std:.3f}")
''',
    },

    "Build a LLaMA-Style Model Block": {
        "description": "Assemble a complete LLaMA-3-style Transformer block: Pre-RMSNorm + RoPE + GQA + SwiGLU FFN + no biases.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LLAMA-3 STYLE TRANSFORMER BLOCK — FULL ASSEMBLY
================================================================================

Combines all the modern innovations into one canonical block:
    •   Pre-RMSNorm (not Post-LayerNorm)
    •   RoPE on Q and K (relative positional encoding)
    •   GQA with configurable n_kv_heads
    •   SwiGLU FFN (3-matrix gated design)
    •   No biases in any Linear layer

This is functionally equivalent to a single LLaMA-3 Transformer block.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration matching LLaMA-3 8B (scaled down for demo)."""
    d_model:     int = 1024    # 4096 in LLaMA-3 8B
    n_heads:     int = 8       # 32   in LLaMA-3 8B
    n_kv_heads:  int = 2       # 8    in LLaMA-3 8B
    n_layers:    int = 2       # 32   in LLaMA-3 8B
    vocab_size:  int = 512     # 128k in LLaMA-3 8B
    max_seq_len: int = 64      # 8192 in LLaMA-3 8B
    rope_base:   float = 500_000.0  # 500k in LLaMA-3 (was 10k in LLaMA-2)
    multiple_of: int = 256

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def d_ff(self) -> int:
        raw = int(8 / 3 * self.d_model)
        return ((raw + self.multiple_of - 1) // self.multiple_of) * self.multiple_of


# ── RMSNorm ────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


# ── RoPE ──────────────────────────────────────────────────────────────────────

def precompute_rope(d_head: int, max_seq_len: int,
                    base: float = 500_000.0) -> tuple:
    i      = torch.arange(0, d_head, 2, dtype=torch.float32)
    thetas = 1.0 / (base ** (i / d_head))
    pos    = torch.arange(max_seq_len, dtype=torch.float32)
    freqs  = torch.outer(pos, thetas)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    cos    = cos[None, :, None, :]
    sin    = sin[None, :, None, :]
    out    = torch.stack([
        x_even * cos - x_odd  * sin,
        x_even * sin + x_odd  * cos,
    ], dim=-1).flatten(-2)
    return out


# ── GQA Attention ─────────────────────────────────────────────────────────────

class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # No biases — LLaMA style
        self.Wq = nn.Linear(cfg.d_model, cfg.n_heads    * cfg.d_head, bias=False)
        self.Wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.Wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.Wo = nn.Linear(cfg.n_heads  * cfg.d_head,  cfg.d_model,  bias=False)

        cos, sin = precompute_rope(cfg.d_head, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos_table", cos)
        self.register_buffer("sin_table", sin)

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        cfg      = self.cfg

        q = self.Wq(x).view(B, T, cfg.n_heads,    cfg.d_head)
        k = self.Wk(x).view(B, T, cfg.n_kv_heads, cfg.d_head)
        v = self.Wv(x).view(B, T, cfg.n_kv_heads, cfg.d_head)

        # Apply RoPE to Q and K
        cos = self.cos_table[:T]
        sin = self.sin_table[:T]
        q   = apply_rope(q, cos, sin)
        k   = apply_rope(k, cos, sin)

        # Expand KV heads to match Q head count (GQA → repeat)
        if cfg.n_groups > 1:
            k = k.repeat_interleave(cfg.n_groups, dim=2)
            v = v.repeat_interleave(cfg.n_groups, dim=2)

        # (B, h, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale  = math.sqrt(cfg.d_head)
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0, float("-inf")
        )
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.Wo(out)


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # No biases — LLaMA style
        self.W1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.Wg = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.W2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.silu(self.W1(x)) * self.Wg(x))


# ── Transformer Block ─────────────────────────────────────────────────────────

class LlamaBlock(nn.Module):
    """
    One LLaMA-3 style Transformer block.
    Layout:
        x = x + Attention(RMSNorm(x))    ← Pre-Norm + residual
        x = x + FFN(RMSNorm(x))          ← Pre-Norm + residual
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn( self.norm2(x))
        return x


# ── Full LLaMA-style LM ───────────────────────────────────────────────────────

class LlamaLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor, targets=None):
        x      = self.embed(idx)
        for block in self.blocks:
            x  = block(x)
        x      = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = ModelConfig()
    model = LlamaLM(cfg)

    total = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print("  LLAMA-3 STYLE MODEL — PARAMETER BREAKDOWN")
    print(f"  Config: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"n_kv={cfg.n_kv_heads}, n_layers={cfg.n_layers}")
    print(f"  d_ff={cfg.d_ff}  (SwiGLU 8/3× = {cfg.d_ff/cfg.d_model:.2f}×)")
    print(f"  RoPE base: {cfg.rope_base:,}  (LLaMA-3 style, enables long context)")
    print("=" * 60)

    print(f"\n  Component                    Params")
    print(f"  {'':─<40}")
    print(f"  Token embedding:             {model.embed.weight.numel():>12,}")
    for i, block in enumerate(model.blocks):
        attn_p = sum(p.numel() for p in block.attn.parameters())
        ffn_p  = sum(p.numel() for p in block.ffn.parameters())
        norm_p = sum(p.numel() for p in block.norm1.parameters()) + \
                 sum(p.numel() for p in block.norm2.parameters())
        print(f"  Block {i} — attn: {attn_p:>9,}  ffn: {ffn_p:>9,}  norm: {norm_p:>6,}")
    print(f"  Final RMSNorm:               {sum(p.numel() for p in model.norm.parameters()):>12,}")
    print(f"  LM head (weight-tied):       {'[shared with embed]':>12}")
    print(f"  {'':─<40}")
    print(f"  TOTAL:                       {total:>12,}")

    # Forward pass
    print()
    B, T = 2, 16
    idx  = torch.randint(0, cfg.vocab_size, (B, T))
    tgt  = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, tgt)
    print(f"  Forward pass: input={tuple(idx.shape)}  logits={tuple(logits.shape)}")
    print(f"  Loss: {loss.item():.4f}  (random init, expected ≈ ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f})")

    # Verify no biases anywhere in attention/FFN
    has_bias = [(n, p.shape) for n, p in model.named_parameters()
                if "bias" in n and "norm" not in n]
    print()
    if has_bias:
        print(f"  ⚠️  Unexpected biases found: {has_bias}")
    else:
        print(f"  ✓  No biases in attention or FFN projections (LLaMA style)")
    print(f"  ✓  RMSNorm gamma present:  yes")
    print(f"  ✓  SwiGLU gating:          yes (3 matrices per FFN block)")
    print(f"  ✓  GQA ratio:              {cfg.n_heads}/{cfg.n_kv_heads} = {cfg.n_groups}× fewer KV heads")
    print(f"  ✓  RoPE base:              {cfg.rope_base:,} (high base → long-context capable)")
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


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
    visual_height = 600
    # try:
    #     from llm_training.visuals.modern_architectures import (
    #         ARCH_VISUAL_HTML,
    #         ARCH_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ARCH_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ARCH_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[05_modern_architectures_llama_mistral_phi.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }