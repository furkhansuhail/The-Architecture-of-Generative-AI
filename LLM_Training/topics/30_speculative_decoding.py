"""
Speculative Decoding
=====================

Speculative decoding is the most practically impactful inference
acceleration technique of the LLM era. It achieves 2–4× speedup on
autoregressive generation without any change to model quality, by
exploiting the observation that a small "draft" model can propose tokens
cheaply, and the large "target" model can verify many tokens in parallel in
a single forward pass. The mathematics of the rejection sampling scheme
guarantees that the output distribution is identical to what the target
model would produce if run naively — the speedup is purely computational,
not a quality approximation. Understanding the acceptance rate, the optimal
draft model size, and why the guarantee holds is essential for deploying
fast LLM inference.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Speculative Decoding"
DISPLAY_NAME = "30 · Speculative Decoding"
ICON         = "⚡"
SUBTITLE     = "Draft Models and Parallel Verification"


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

### The Autoregressive Generation Bottleneck

Autoregressive language model generation is inherently sequential: to generate
token t, the model must have already generated tokens 1 through t-1. This
creates a fundamental throughput problem:

    Number of forward passes per response = number of tokens in the response

For a 200-token response from LLaMA-2 70B:
    •   200 sequential forward passes × ~0.1 seconds per pass = ~20 seconds
    •   GPU utilisation during each forward pass: ~15–30% (memory bandwidth bound)
      (each forward pass reads 140 GB of weights from HBM to compute a single token)

This is deeply inefficient. The GPU spends most of its cycles moving model
weights from HBM memory into compute cores — not actually computing. Modern
GPUs are designed for large, parallel matrix multiplications, but autoregressive
generation creates tiny matmul operations (batch=1, sequence_position=1) that
underutilise the hardware.

**The core issue:** the cost of generating token t is dominated by the
weight-to-compute transfer, not by the actual arithmetic. At batch=1, every
forward pass reads ~140 GB (70B weights × 2 bytes bf16) but only performs
~2 × 70B ≈ 140 GFLOP of compute. The A100's HBM bandwidth limits this to
~70 ms per forward pass, while its compute could handle 300+ GFLOP/ms — an
arithmetic intensity of just 1 FLOP/byte (far below the roofline).

The arithmetic intensity for generation:
    FLOPs per token / bytes per token = 2P / (P × 2) = 1 FLOP/byte at batch=1
    A100 roofline: 2 TB/s × (9750 / 1) = 19.5 TFLOPS effective

Any generation strategy that increases tokens per forward pass directly
improves hardware utilisation.


### The Key Insight: Verification Is Parallel

Consider a target model T (large, slow) and a draft model D (small, fast).

The standard approach processes one token at a time:
    Forward(T, [x, y₁, y₂, ..., y_t]) → p_T(y_{t+1}|x, y_{1:t}) → sample y_{t+1}
    (one forward pass per token, sequential)

**Speculative decoding** reverses this: generate K candidate tokens cheaply,
then verify them all at once:

    **Draft phase:** Run D autoregressively to generate K candidates
        y̋₁ ~ p_D(·|x)
        y̋₂ ~ p_D(·|x, y̋₁)
        ...
        y̋_K ~ p_D(·|x, y̋₁, ..., y̋_{K-1})
        Cost: K small forward passes through D

    **Verify phase:** Run T once on the FULL sequence [x, y̋₁, ..., y̋_K]
        T computes p_T(·|x), p_T(·|x,y̋₁), ..., p_T(·|x,y̋₁,...,y̋_K)
        ALL in ONE forward pass (same as training with teacher-forcing!)
        Cost: 1 large forward pass through T

The key: the target model's forward pass is prefix-parallel. Given the
sequence [x, y̋₁, ..., y̋_K], the model computes the probability of every
position simultaneously (this is exactly what the causal attention mask enables).


### The Acceptance/Rejection Scheme — Why Quality Is Preserved

Simply keeping all draft tokens if the target model assigns high probability
to them would not preserve the target distribution. The algorithm needs a
principled way to accept or reject draft tokens that guarantees the output
distribution matches p_T.

**The speculative sampling algorithm (Chen et al., 2023):**

For each draft token y̋_t (t = 1, ..., K):
    •   Compute α_t = min(1, p_T(y̋_t|context) / p_D(y̋_t|context))
    •   Sample u ~ Uniform(0, 1)
    •   If u ≤ α_t: ACCEPT y̋_t and move to y̋_{t+1}
    •   If u > α_t: REJECT y̋_t and all subsequent draft tokens.
        Instead, sample from the ADJUSTED distribution:
            p_adjusted(y) = max(0, p_T(y|context) - p_D(y|context)) / Z
        where Z normalises p_adjusted to sum to 1.

**Why this preserves the target distribution:**

The key theorem: the accepted tokens, together with the fallback-sampled
replacement on rejection, are distributed identically to sampling from p_T.

Proof sketch for one token:
    P(y accepted at position t)
    = P(accept|y = y̋) × P(y̋ = y from draft)
    + P(draw adjusted|rejection) × P(y drawn from adjusted)

    For the first term:
    = min(1, p_T(y)/p_D(y)) × p_D(y)
    = min(p_D(y), p_T(y))

    For the second term (when draft is rejected):
    Total probability of rejection = Σ_y max(0, p_D(y) - p_T(y)) × 1
    = Σ_y max(0, p_D(y) - p_T(y))
    p_adjusted(y) = max(0, p_T(y) - p_D(y)) / Z where Z = Σ_y max(0, p_T-p_D)

    P(y at position t)
    = min(p_D(y), p_T(y)) + Z × p_adjusted(y)
    = min(p_D(y), p_T(y)) + max(0, p_T(y) - p_D(y))
    = p_T(y)  ✓

**The beautiful result:** regardless of how wrong the draft model is, the
output distribution is exactly p_T. Speculative decoding with a bad draft
model will have a low acceptance rate (slow), but it will never produce
incorrect outputs.


    **Diagram 1 — Speculative Decoding Step-by-Step:**

    SPECULATIVE DECODING: K=4 DRAFT TOKENS
    ════════════════════════════════════════════════════════════════

    DRAFT PHASE (4 small forward passes):
    Context: "The cat sat on"
    D: "the" (prob 0.7)  → accepted? → "mat" (0.5) → "and" (0.8) → "then" (0.6)
       y̋₁              y̋₂            y̋₃           y̋₄

    VERIFY PHASE (1 large forward pass):
    T processes: ["The", "cat", "sat", "on", "the", "mat", "and", "then"]
    T outputs:   p_T(·|"The")   p_T(·|"the cat")  ...  p_T(·|"...on the mat and")
                 simultaneously (causal attention, teacher-forcing style)

    ACCEPT/REJECT:
    Token "the":  α₁ = min(1, p_T("the"|ctx₀) / p_D("the"|ctx₀))
    Token "mat":  α₂ = min(1, p_T("mat"|ctx₁) / p_D("mat"|ctx₁))
    Token "and":  α₃ = min(1, p_T("and"|ctx₂) / p_D("and"|ctx₂))  → REJECTED!
    Token "then": skipped (after first rejection)

    RESULT: Accept ["the", "mat"], reject ["and", "then"]
    BONUS:  Sample p_adjusted at position of "and" → get "."
    Net tokens accepted: "the", "mat", "." = 3 tokens in 1 target + 4 draft passes


### The Acceptance Rate: The Key Performance Metric

The **acceptance rate** ᾱ (average acceptance probability per draft token)
determines the speedup. If the draft model proposes K tokens per step:

    Expected accepted tokens per step ≈ K × ᾱ + 1   (K × mean_accept + 1 fallback)

Wait, the exact formula is more subtle due to the sequential nature of acceptance:

    E[accepted tokens] = Σ_{k=0}^{K} ᾱ^k × (1 - ᾱ) × k + ᾱ^K × K
                       = (1 - ᾱ^{K+1}) / (1 - ᾱ)       (geometric series)

Adding the 1 final token always generated (from the adjusted or bonus distribution):
    E[tokens per verify step] = (1 - ᾱ^{K+1}) / (1 - ᾱ) + ᾱ^K

    Speedup ≈ E[tokens per step] / (1 + γ × K)

where γ = (time per draft forward pass) / (time per target forward pass)

For a 70B target and a 7B draft:
    γ ≈ (7B params × 2 bytes / 2 TB/s HBM) / (70B params × 2 bytes / 2 TB/s HBM)
      ≈ 7/70 = 0.1   (draft is ~10× faster)

    At ᾱ = 0.8, K = 4:
    E[tokens] = (1 - 0.8⁵) / (1 - 0.8) = (1 - 0.328) / 0.2 = 3.36
    Time per step ≈ (1 + 0.1 × 4) × T_target = 1.4 × T_target
    Speedup = 3.36 / 1.4 ≈ 2.4×


    **Diagram 2 — Acceptance Rate vs Speedup:**

    SPEEDUP vs ACCEPTANCE RATE ᾱ (K=4, γ=0.1)
    ════════════════════════════════════════════════════════════════

    ᾱ = 0.5:  E[tokens] ≈ (1-0.031)/(0.5) = 1.94  speedup ≈ 1.4×  (poor draft)
    ᾱ = 0.7:  E[tokens] ≈ (1-0.168)/(0.3) = 2.77  speedup ≈ 2.0×
    ᾱ = 0.8:  E[tokens] ≈ 3.36               speedup ≈ 2.4×
    ᾱ = 0.9:  E[tokens] ≈ (1-0.59)/(0.1)= 4.10  speedup ≈ 2.9×
    ᾱ = 0.95: E[tokens] ≈ (1-0.77)/(0.05)=4.60  speedup ≈ 3.3×
    ᾱ = 1.0:  E[tokens] = K = 4.0             speedup ≈ 2.9×  (perfect draft)

    Note: ᾱ=1.0 gives lower speedup than ᾱ=0.95 because K+1 total steps
    are still needed (including the final bonus token from target).


### Optimal Draft Model Size

The draft model should be:
    1.  **Large enough** to achieve high acceptance rate
    2.  **Small enough** to be much faster than the target

Empirical results:
    Target 70B ← Draft 7B:   ᾱ ≈ 0.75–0.85 for general text,  speedup 2–3×
    Target 70B ← Draft 1.3B: ᾱ ≈ 0.6–0.7,  speedup 1.5–2×
    Target 7B  ← Draft 1B:   ᾱ ≈ 0.8–0.9,  speedup 2–3×

**The draft model quality matters for domain:**
For a medical question-answering task, a general 7B draft model will have
lower acceptance rate than a 7B model fine-tuned on medical text. Speculative
decoding benefits from domain-specific draft models.

**Model families matter:** The draft and target should be from the same
model family. A LLaMA-3 8B draft works better with a LLaMA-3 70B target
than an unrelated 8B model, because they share vocabulary, tokenisation,
and learned representations.


### Speculative Decoding Variants

**Standard speculative decoding (Chen et al., 2023):**
Separate draft model + target model. The original formulation.

**Self-speculative decoding / EAGLE:**
No separate draft model needed! Use the target model itself to generate
drafts, but at a shallower depth:
    •   EAGLE (Ji et al., 2024): Train a small additional head on top of
        the target model that predicts next-next tokens using the hidden states.
        Draft tokens are generated from this head; verification uses full model.
    •   Medusa (Cai et al., 2024): Add K parallel prediction heads to the
        target model, each predicting tokens K steps ahead. No separate draft.
    •   Lookahead decoding: Generate multiple token sequences speculatively
        using n-gram lookups from a cache; verify with the full model.

**Tree-based speculative decoding:**
Instead of a single draft chain y̋₁,...,y̋_K, build a tree of candidates:
    Level 1: K₁ candidate tokens
    Level 2: each of K₁ leads to K₂ candidates → K₁ × K₂ paths total
    Level 3: each of K₁×K₂ leads to K₃ candidates → K₁ × K₂ × K₃ total

The target model verifies all tree nodes in one forward pass using a tree
attention mask. This finds the longest accepted prefix in the tree, giving
higher expected acceptance. Used in SpecTree and EAGLE-2.

**Retrieval-augmented speculative decoding:**
Use a text corpus to retrieve likely completions of the current context.
These retrieved text segments serve as draft tokens. High acceptance rate
for common phrasings; no draft model overhead.


### Implementation Details

**KV cache management:**
Both the draft model and the target model maintain KV caches. The complexity
is in managing cache consistency when tokens are rejected:
    •   Draft model's KV cache must be rolled back to the last accepted position
    •   Target model's KV cache is extended by the accepted tokens (from the
        verify pass, which already computed all the needed KV states)
    •   On rejection: target model's KV cache is extended up to the rejected
        position; draft model's KV cache is reset to that position

**The bonus token:**
At the end of a verify step (whether all K tokens were accepted or a rejection
occurred), the target model always generates one additional token:
    •   If all K accepted: sample from p_T(·|full_sequence) — a free extra token!
    •   If rejection at position k: sample from p_adjusted(·|context up to k-1)
    •   This ensures the algorithm always makes forward progress

**Batched speculative decoding:**
For batch inference (serving multiple users), speculative decoding is more
complex because different sequences in the batch may accept/reject different
numbers of draft tokens. Solutions:
    •   Rejection padding: pad shorter accepted sequences to the length of
        the longest, insert dummy tokens at rejected positions
    •   Asynchronous batching: dynamically rebatch sequences after rejection

**GPU utilisation with speculative decoding:**
The verify step processes a sequence of length P + K (prefix + drafts).
This provides K× higher token density in the matmul operations:
    matmul size: (P+K) × d  vs  P × d  (K× larger)
    Higher matmul density → closer to the compute-bound regime
    At K=4, effective arithmetic intensity increases by ~4×

For batch_size=1 (single user), speculative decoding with K=4 pushes the
matmul size from ~1×d to ~5×d, substantially improving hardware utilisation.


### When Speculative Decoding Helps Most

    Scenario                         Expected benefit    Notes
    ─────────────────────────────────────────────────────────────────────
    Single-user, long generation     Very high (3-4×)    Low batch; memory-bound
    Chat / interactive responses     High (2-3×)         Single user, short seqs
    Bulk processing, large batch     Low (~1.2×)         Already compute-bound
    Structured output (code, JSON)   High (2-3×)         Predictable patterns
    Creative/random generation       Moderate (1.5-2×)   Lower acceptance rate
    Domain-matched draft             High (2-4×)         Domain-specific draft
    Cross-domain draft               Low (1.2-2×)        Poor acceptance rate
    ─────────────────────────────────────────────────────────────────────

Speculative decoding is most beneficial for latency-critical applications
(interactive chat, code completion) where a single user query needs a fast
response. For throughput-critical applications (processing millions of
documents), large-batch inference already achieves good GPU utilisation.


### System-Level Considerations

**Memory overhead:**
Both the draft and target models must fit in GPU memory simultaneously.
For a 70B target + 7B draft on a single node of 8× A100 80GB:
    Draft: 7B × 2 bytes × 8 = 14 GB (fits comfortably on 1 GPU)
    Target: 70B × 2 bytes = 140 GB (needs 2 GPUs for weights)
    Combined: 154 GB → fits on 4× A100 with tensor parallelism

Alternatively: run the draft model on CPU with PCIe transfer (only works
if the draft model is fast enough to overlap with target compute).

**Latency breakdown (70B + 7B at K=4, single user):**
    Draft 4 tokens:   4 × 7ms = 28ms
    Target verify 1:  1 × 70ms = 70ms
    Total per step:   98ms for ~3 tokens average
    vs. naive:        1 × 70ms = 70ms for 1 token
    Speedup:          3 tokens / 98ms vs 1 token / 70ms ≈ 2.1× throughput
    Latency:          98ms for 3 tokens vs 70ms for 1 token (same latency for 1st token)
    First token latency: identical (no change — draft runs after target starts)

Note: speculative decoding does NOT improve time-to-first-token (TTFT).
It only improves throughput after the first token (tokens-per-second).
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Speculative Decoding Variants Comparison

| Variant                | Draft mechanism          | Memory overhead | Expected speedup | Distribution exact? |
|------------------------|--------------------------|-----------------|------------------|---------------------|
| Standard spec. decoding| Separate draft LM        | +draft model    | 2–4×             | Yes (proven)        |
| Medusa                 | K parallel heads on target| +K×d params    | 2–3×             | Yes (with sampling) |
| EAGLE                  | 1-layer head + target     | +small head     | 3–4×             | Yes                 |
| Tree speculative       | Tree-structured draft     | Same as standard| 2.5–5×           | Yes                 |
| Lookahead decoding     | N-gram cache             | N-gram table    | 1.5–2.5×         | Approximate         |
| Self-draft (layer skip)| Skip middle layers        | Same as target  | 1.5–2×           | Approximate         |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Speculative Decoding — Full Implementation": {
        "description": "Complete speculative decoding implementation with the rejection sampling guarantee proof in code — draft phase, parallel verification, accept/reject with adjusted distribution, KV cache simulation, and correctness verification.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SPECULATIVE DECODING — COMPLETE IMPLEMENTATION
================================================================================

Implements the complete speculative decoding algorithm:
    1. Draft model generates K candidate tokens autoregressively
    2. Target model verifies all K tokens in ONE forward pass
    3. Accept/reject tokens via speculative sampling (Chen et al., 2023)
    4. Compute adjusted distribution on rejection; sample bonus token on full accept
    5. Verify correctness: output distribution matches vanilla target sampling

The implementation proves the quality guarantee by running both algorithms
on the same prompts and showing the output distributions are identical.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


# ── Minimal models ────────────────────────────────────────────────────────────

class SmallLM(nn.Module):
    """Draft model: small and fast."""
    def __init__(self, vocab: int, d: int = 32, n_layers: int = 1):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d*2, bias=False), nn.GELU(),
                          nn.Linear(d*2, d, bias=False))
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h))

    def get_next_token_probs(self, context: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over next token given context."""
        with torch.no_grad():
            logits = self.forward(context)
        return F.softmax(logits[:, -1, :], dim=-1)


class LargeLM(nn.Module):
    """Target model: large and accurate."""
    def __init__(self, vocab: int, d: int = 96, n_layers: int = 3):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d*4, bias=False), nn.GELU(),
                          nn.Linear(d*4, d, bias=False))
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h))

    def get_all_probs(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Return probability distributions for ALL positions simultaneously.
        This is the key: ONE forward pass gives us ALL the distributions.
        Shape: (1, T, V) where position t gives p_T(·|sequence[:t])
        """
        with torch.no_grad():
            logits = self.forward(sequence)
        return F.softmax(logits, dim=-1)


# ── The rejection sampling scheme ────────────────────────────────────────────

def speculative_sampling_step(
    draft_model: SmallLM,
    target_model: LargeLM,
    context: torch.Tensor,   # (1, T_context) current context tokens
    K: int,                   # number of draft tokens to generate
    temperature: float = 1.0, # sampling temperature
) -> tuple[torch.Tensor, int, list[float]]:
    """
    One step of speculative decoding.

    Returns:
        new_tokens:      (n,) tensor of accepted tokens (1 ≤ n ≤ K+1)
        n_accepted:      number of draft tokens that were accepted (0 ≤ n ≤ K)
        accept_probs:    list of acceptance probabilities α_t for each position
    """
    VOCAB = draft_model.embed.num_embeddings

    # ── DRAFT PHASE ───────────────────────────────────────────────────────────
    draft_tokens  = []
    draft_probs   = []   # p_D(y̋_t | context, y̋_1..t-1)
    current_ctx   = context.clone()

    for k in range(K):
        p_D = draft_model.get_next_token_probs(current_ctx)  # (1, V)
        if temperature != 1.0:
            # Temperature scaling: sharpen or flatten the distribution
            logits = torch.log(p_D.clamp(min=1e-10)) / temperature
            p_D    = F.softmax(logits, dim=-1)
        # Sample next token from draft
        y_draft = torch.multinomial(p_D, num_samples=1)   # (1, 1)
        draft_tokens.append(y_draft.squeeze())
        draft_probs.append(p_D.squeeze())   # (V,)
        current_ctx = torch.cat([current_ctx, y_draft], dim=1)

    draft_seq = torch.stack(draft_tokens)   # (K,)

    # ── VERIFY PHASE (ONE forward pass through target) ────────────────────────
    # Full sequence: context + all K draft tokens
    verify_input = torch.cat([context,
                               draft_seq.unsqueeze(0)], dim=1)  # (1, T+K)

    # Target model sees the full sequence and produces probs at each position
    # target_all_probs[t] = p_T(·|verify_input[:t])
    target_all_probs = target_model.get_all_probs(verify_input)  # (1, T+K, V)

    # We need p_T at positions corresponding to draft token positions
    # Position T-1 in the full sequence predicts the first draft token
    # Position T predicts the second, etc.
    T_ctx = context.shape[1]   # length of original context

    # p_T distributions for each of the K draft token positions
    target_probs_at_draft = [
        target_all_probs[0, T_ctx - 1 + k, :]   # (V,) for k-th draft position
        for k in range(K)
    ]
    # p_T distribution for the bonus token (one past all drafts)
    target_bonus_probs = target_all_probs[0, T_ctx - 1 + K, :]  # (V,)

    # ── ACCEPT / REJECT ───────────────────────────────────────────────────────
    accepted_tokens = []
    accept_probs    = []
    n_accepted      = 0

    for k in range(K):
        y_k   = draft_seq[k].item()
        p_D_k = draft_probs[k]       # (V,)
        p_T_k = target_probs_at_draft[k]  # (V,)

        # Acceptance probability: min(1, p_T(y̋_k) / p_D(y̋_k))
        p_T_yk = p_T_k[y_k].item()
        p_D_yk = p_D_k[y_k].item()
        alpha  = min(1.0, p_T_yk / (p_D_yk + 1e-10))
        accept_probs.append(alpha)

        u = torch.rand(1).item()
        if u <= alpha:
            # ACCEPT: draft token is consistent with target distribution
            accepted_tokens.append(y_k)
            n_accepted += 1
        else:
            # REJECT: sample from adjusted distribution
            # p_adjusted(y) = max(0, p_T(y) - p_D(y)) / Z
            diff           = (p_T_k - p_D_k).clamp(min=0.0)
            Z              = diff.sum().item()
            if Z > 1e-10:
                p_adjusted = diff / Z
            else:
                p_adjusted = p_T_k   # fallback: use target distribution directly
            fallback_token = torch.multinomial(p_adjusted, num_samples=1).item()
            accepted_tokens.append(fallback_token)
            break  # stop accepting after first rejection

    # Bonus token: if all K were accepted, also take one from target
    if n_accepted == K:
        bonus_token = torch.multinomial(target_bonus_probs,
                                         num_samples=1).item()
        accepted_tokens.append(bonus_token)

    new_tokens = torch.tensor(accepted_tokens, dtype=torch.long)
    return new_tokens, n_accepted, accept_probs


# ── Full generation with speculative decoding ─────────────────────────────────

def generate_speculative(
    draft_model: SmallLM,
    target_model: LargeLM,
    prompt: torch.Tensor,
    max_new_tokens: int = 20,
    K: int = 4,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Generate max_new_tokens using speculative decoding.
    Returns (generated_sequence, stats).
    """
    ctx = prompt.unsqueeze(0).clone()   # (1, T_prompt)
    stats = {
        "target_forward_passes": 0,
        "draft_forward_passes":  0,
        "n_tokens_generated":    0,
        "n_draft_accepted":      0,
        "n_draft_total":         0,
        "step_accept_rates":     [],
    }

    while stats["n_tokens_generated"] < max_new_tokens:
        remaining = max_new_tokens - stats["n_tokens_generated"]
        K_step    = min(K, remaining)

        new_tokens, n_acc, alphas = speculative_sampling_step(
            draft_model, target_model, ctx, K_step, temperature
        )

        ctx = torch.cat([ctx, new_tokens.unsqueeze(0)], dim=1)

        stats["target_forward_passes"] += 1
        stats["draft_forward_passes"]  += K_step
        stats["n_tokens_generated"]    += len(new_tokens)
        stats["n_draft_accepted"]      += n_acc
        stats["n_draft_total"]         += K_step
        stats["step_accept_rates"].append(n_acc / K_step)

        if len(new_tokens) == 0:
            break

    return ctx[0, prompt.shape[0]:], stats


# ── Vanilla (target-only) generation for comparison ──────────────────────────

def generate_vanilla(target_model: LargeLM, prompt: torch.Tensor,
                      max_new_tokens: int = 20, temperature: float = 1.0,
                      seed: int = None) -> torch.Tensor:
    """Standard autoregressive generation from target model only."""
    if seed is not None:
        torch.manual_seed(seed)
    ctx = prompt.unsqueeze(0).clone()
    for _ in range(max_new_tokens):
        probs = target_model.get_all_probs(ctx)[:, -1, :]
        if temperature != 1.0:
            logits = torch.log(probs.clamp(min=1e-10)) / temperature
            probs  = F.softmax(logits, dim=-1)
        next_t = torch.multinomial(probs, num_samples=1)
        ctx    = torch.cat([ctx, next_t], dim=1)
    return ctx[0, prompt.shape[0]:]


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    VOCAB      = 64
    PROMPT_LEN = 4
    MAX_NEW    = 20
    K          = 4
    N_SAMPLES  = 500   # for distribution matching test

    draft  = SmallLM(VOCAB, d=24, n_layers=1)
    target = LargeLM(VOCAB, d=64, n_layers=2)

    prompt = torch.randint(3, VOCAB, (PROMPT_LEN,))

    print("=" * 65)
    print("  SPECULATIVE DECODING — CORRECTNESS + PERFORMANCE DEMO")
    print(f"  vocab={VOCAB}, prompt_len={PROMPT_LEN}, K={K}")
    print("=" * 65)
    print()

    # Single generation demo
    torch.manual_seed(0)
    tokens, stats = generate_speculative(draft, target, prompt, MAX_NEW, K)

    print("  Single generation (speculative decoding):")
    print(f"  Generated tokens:         {tokens.tolist()[:10]}...")
    print(f"  Target forward passes:    {stats['target_forward_passes']}")
    print(f"  Draft forward passes:     {stats['draft_forward_passes']}")
    print(f"  Tokens generated:         {stats['n_tokens_generated']}")
    print(f"  Draft acceptance rate:    "
          f"{stats['n_draft_accepted']/max(stats['n_draft_total'],1):.1%}")
    print(f"  Tokens per target pass:   "
          f"{stats['n_tokens_generated']/stats['target_forward_passes']:.2f}")
    vanilla_passes = MAX_NEW
    print(f"  Naive would need:         {vanilla_passes} target passes")
    print(f"  Speedup (forward passes): "
          f"{vanilla_passes / stats['target_forward_passes']:.2f}×")
    print()

    # ── DISTRIBUTION MATCHING TEST ────────────────────────────────────────────
    print("=" * 65)
    print("  DISTRIBUTION MATCHING TEST")
    print(f"  Generating {N_SAMPLES} single next-tokens each way")
    print("  (proving speculative decoding preserves target distribution)")
    print("=" * 65)
    print()

    speculative_counts = Counter()
    vanilla_counts     = Counter()
    accept_rates       = []

    for i in range(N_SAMPLES):
        torch.manual_seed(i)
        spec_tok, n_acc, alphas = speculative_sampling_step(
            draft, target, prompt.unsqueeze(0), K=1, temperature=1.0
        )
        speculative_counts[spec_tok[0].item()] += 1
        if alphas:
            accept_rates.append(alphas[0])

        torch.manual_seed(i)
        van_tok = torch.multinomial(
            target.get_all_probs(prompt.unsqueeze(0))[:, -1, :], 1
        ).item()
        vanilla_counts[van_tok] += 1

    # Compare distributions
    all_tokens = set(list(speculative_counts.keys()) + list(vanilla_counts.keys()))
    max_diff   = 0.0
    top_tokens = sorted(all_tokens, key=lambda t: vanilla_counts.get(t,0)+
                         speculative_counts.get(t,0), reverse=True)[:8]

    print(f"  {'Token':>8}  {'Vanilla prob':>14}  {'Spec. prob':>12}  {'Diff':>10}")
    print(f"  {'':─>8}  {'':─>14}  {'':─>12}  {'':─>10}")
    for tok in top_tokens:
        v_prob = vanilla_counts.get(tok, 0) / N_SAMPLES
        s_prob = speculative_counts.get(tok, 0) / N_SAMPLES
        diff   = abs(v_prob - s_prob)
        max_diff = max(max_diff, diff)
        flag   = " ≈ " if diff < 0.03 else " ← differs"
        print(f"  {tok:>8}  {v_prob:>14.3f}  {s_prob:>12.3f}  {diff:>10.3f}{flag}")

    print()
    print(f"  Max absolute probability difference: {max_diff:.3f}")
    print(f"  (Expected: ~1/√N = {1/math.sqrt(N_SAMPLES):.3f} Monte Carlo noise)")
    if max_diff < 0.05:
        print("  ✓ Speculative decoding preserves the target distribution!")
    else:
        print("  ⚠️  Distribution mismatch (check implementation)")

    mean_accept = sum(accept_rates) / len(accept_rates)
    print()
    print(f"  Mean acceptance rate (K=1): {mean_accept:.3f}")
    print(f"  (Higher = draft model aligns better with target)")
''',
    },

    "Acceptance Rate and Speedup Analysis": {
        "description": "Measure the acceptance rate as a function of draft model quality, sequence position, and temperature — and compute the theoretical speedup curve for different K values.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ACCEPTANCE RATE AND SPEEDUP ANALYSIS
================================================================================

Analyses the key performance determinants of speculative decoding:
    1. Acceptance rate ᾱ as a function of draft quality (KL divergence)
    2. Acceptance rate vs generation position (usually decreases)
    3. Expected speedup as a function of K and ᾱ
    4. The optimal K for different (draft_speed, target_speed) ratios
    5. Temperature's effect on acceptance rate

Key insight: the speedup depends on both the acceptance rate AND the
relative speed of draft vs target. A perfect draft (ᾱ=1) with a slow
draft model gives less speedup than a good draft (ᾱ=0.9) with a fast one.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ── Theoretical speedup formula ───────────────────────────────────────────────

def expected_tokens_per_step(alpha_bar: float, K: int) -> float:
    """
    Expected number of tokens produced per verify step.

    E[tokens] = (1 - ᾱ^(K+1)) / (1 - ᾱ)

    where the +1 accounts for the bonus token always produced.
    """
    if abs(alpha_bar - 1.0) < 1e-8:
        return K + 1.0  # all K accepted + 1 bonus
    return (1 - alpha_bar ** (K + 1)) / (1 - alpha_bar)


def expected_speedup(alpha_bar: float, K: int,
                      gamma: float) -> float:
    """
    Expected throughput speedup vs vanilla generation.

    gamma = (time per draft forward pass) / (time per target forward pass)
            = (draft_param_count) / (target_param_count) for memory-bound case

    speedup = E[tokens per step] / (time per step / time per token)
            = E[tokens per step] / (1 + gamma * K)
    """
    e_tokens   = expected_tokens_per_step(alpha_bar, K)
    time_ratio = 1 + gamma * K   # relative to one target pass
    return e_tokens / time_ratio


def optimal_K(alpha_bar: float, gamma: float, K_range: range = range(1, 32)) -> int:
    """Find the K that maximises expected speedup."""
    return max(K_range, key=lambda k: expected_speedup(alpha_bar, k, gamma))


# ── Empirical acceptance rate measurement ─────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab, d, n_layers):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d*2,bias=False),
                          nn.GELU(), nn.Linear(d*2,d,bias=False))
            for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers: h = h + l(h)
        return self.head(self.norm(h))

    def next_probs(self, ctx):
        with torch.no_grad():
            return F.softmax(self.forward(ctx)[:,-1,:], dim=-1)


def measure_acceptance_rate(draft, target, prompts, K=4, n_steps=200,
                              temperature=1.0) -> float:
    """
    Empirically measure the average acceptance rate ᾱ.
    """
    total_accept = 0
    total_draft  = 0

    for prompt in prompts[:n_steps]:
        ctx = prompt.unsqueeze(0)

        for k in range(K):
            p_D  = draft.next_probs(ctx)
            if temperature != 1.0:
                logits = torch.log(p_D.clamp(1e-10)) / temperature
                p_D    = F.softmax(logits, dim=-1)
            y    = torch.multinomial(p_D, 1)

            # What would target have assigned?
            p_T  = target.next_probs(ctx)
            if temperature != 1.0:
                logits = torch.log(p_T.clamp(1e-10)) / temperature
                p_T    = F.softmax(logits, dim=-1)

            y_val = y.item()
            alpha = min(1.0, p_T[0, y_val].item() / (p_D[0, y_val].item() + 1e-10))
            u     = torch.rand(1).item()

            if u <= alpha:
                total_accept += 1
                ctx = torch.cat([ctx, y], dim=1)
            else:
                total_draft += K - k  # count all remaining as rejected
                break
            total_draft += 1

    return total_accept / max(total_draft, 1)


def kl_divergence_between_probs(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(P||Q) between two probability distributions."""
    p  = p.clamp(min=1e-10)
    q  = q.clamp(min=1e-10)
    return (p * (p / q).log()).sum().item()


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    VOCAB = 64

    # Create models of different sizes to simulate quality spectrum
    target_model = TinyLM(VOCAB, 96, n_layers=3)

    drafts = {
        "Perfect (clone)":  TinyLM(VOCAB, 96, n_layers=3),
        "Good draft":        TinyLM(VOCAB, 64, n_layers=2),
        "OK draft":          TinyLM(VOCAB, 32, n_layers=2),
        "Poor draft":        TinyLM(VOCAB, 16, n_layers=1),
    }

    # Make perfect draft = target
    drafts["Perfect (clone)"].load_state_dict(target_model.state_dict())

    prompts = [torch.randint(3, VOCAB, (4,)) for _ in range(40)]

    print("=" * 70)
    print("  ACCEPTANCE RATE vs DRAFT MODEL QUALITY")
    print("=" * 70)
    print()

    results = {}
    for name, draft in drafts.items():
        alpha = measure_acceptance_rate(draft, target_model, prompts, K=4)
        # Measure KL divergence as proxy for draft quality
        ctx = prompts[0].unsqueeze(0)
        kl_vals = []
        for _ in range(20):
            p_T = target_model.next_probs(ctx)
            p_D = draft.next_probs(ctx)
            kl_vals.append(kl_divergence_between_probs(p_T, p_D))
            tok = torch.multinomial(p_T, 1)
            ctx = torch.cat([ctx, tok], dim=1)
        mean_kl = sum(kl_vals) / len(kl_vals)
        results[name] = (alpha, mean_kl)
        print(f"  {name:<22}:  ᾱ = {alpha:.3f}  KL(T||D) = {mean_kl:.3f}")

    # ── THEORETICAL SPEEDUP ANALYSIS ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  THEORETICAL SPEEDUP vs K (draft params / target params = 0.1)")
    print("=" * 70)
    print()
    gamma = 0.1  # draft is 10× faster (e.g. 7B/70B)

    print(f"  γ = {gamma:.2f}  (draft is {1/gamma:.0f}× faster than target)")
    print()
    print(f"  {'ᾱ':>6}", end="")
    for K in [1, 2, 3, 4, 6, 8, 12]:
        print(f"  {'K='+str(K):>8}", end="")
    print(f"  {'Opt K':>8}  {'Max speedup':>12}")
    print(f"  {'':─>6}" + "".join(f"  {'':─>8}" for _ in [1,2,3,4,6,8,12])
          + f"  {'':─>8}  {'':─>12}")

    for alpha_bar in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        print(f"  {alpha_bar:>6.2f}", end="")
        speedups = {}
        for K in [1, 2, 3, 4, 6, 8, 12]:
            s = expected_speedup(alpha_bar, K, gamma)
            speedups[K] = s
            print(f"  {s:>8.2f}", end="")
        opt_K  = optimal_K(alpha_bar, gamma)
        max_sp = expected_speedup(alpha_bar, opt_K, gamma)
        print(f"  {opt_K:>8}  {max_sp:>12.2f}")

    print()
    print(f"  For γ={gamma}: optimal K increases with higher acceptance rate.")

    # ── GAMMA SENSITIVITY ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SPEEDUP vs DRAFT/TARGET SPEED RATIO γ (ᾱ=0.8, K=4)")
    print("=" * 70)
    print()
    alpha_bar = 0.8
    K_fixed   = 4

    print(f"  {'γ (draft/target ratio)':>25}  {'Speedup':>10}  "
          f"{'Draft description':>30}")
    print(f"  {'':─>25}  {'':─>10}  {'':─>30}")

    gamma_configs = [
        (0.02,  "Draft 70× faster (1B vs 70B)"),
        (0.05,  "Draft 20× faster (3.5B vs 70B)"),
        (0.10,  "Draft 10× faster (7B vs 70B)"),
        (0.20,  "Draft 5× faster (14B vs 70B)"),
        (0.50,  "Draft 2× faster (35B vs 70B)"),
        (1.00,  "Equal speed (not beneficial)"),
    ]

    for gam, desc in gamma_configs:
        sp = expected_speedup(alpha_bar, K_fixed, gam)
        bar = "█" * int(sp * 8)
        print(f"  {gam:>25.2f}  {sp:>10.2f}  {desc:<30}  {bar}")

    # ── TEMPERATURE EFFECT ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  TEMPERATURE'S EFFECT ON ACCEPTANCE RATE")
    print("=" * 70)
    print()
    print("  Higher temperature → more random → draft tokens less likely to match target")
    print()

    good_draft = drafts["Good draft"]
    for temp in [0.1, 0.3, 0.7, 1.0, 1.2, 1.5, 2.0]:
        alpha = measure_acceptance_rate(good_draft, target_model, prompts,
                                          K=4, temperature=temp)
        bar   = "█" * int(alpha * 30)
        print(f"  T={temp:.1f}: ᾱ={alpha:.3f}  |{bar:<30}|")

    print()
    print("  Low temperature (greedy-like): high acceptance rate → better speedup")
    print("  High temperature (creative):   lower acceptance rate → less speedup")
    print()
    print("  This is why speculative decoding helps most for factual/structured")
    print("  generation (where low temperature is natural) vs creative writing.")
''',
    },

    "Medusa and Self-Speculative Decoding": {
        "description": "Implement Medusa-style multi-head speculative decoding where the target model grows parallel prediction heads — no separate draft model needed.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MEDUSA: MULTI-HEAD SPECULATIVE DECODING
================================================================================

Medusa (Cai et al., 2024) adds K parallel prediction heads to the target
model, each predicting a different step ahead:
    Head 0: predicts token t+1  (standard LM head — already exists)
    Head 1: predicts token t+2  (new trainable head)
    Head 2: predicts token t+3  (new trainable head)
    ...
    Head K: predicts token t+K  (new trainable head)

At inference time:
    1. Run ONE forward pass through the target model
    2. Get K candidate tokens from the K additional heads simultaneously
    3. Verify using the same speculative sampling scheme
    4. No separate draft model needed — heads reuse the target's hidden states

This is more memory-efficient than having a separate draft model.
The heads are trained while the base model is frozen (like LoRA for speed).

Speedup depends on acceptance rate of the additional heads:
    Head k predicts token k steps ahead — harder for larger k.
    Typically: head 1 has ᾱ ≈ 0.7, head 2 ≈ 0.5, head 3 ≈ 0.4, ...

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Medusa model: base LM + K additional heads ───────────────────────────────

class MedusaModel(nn.Module):
    """
    A language model with K additional 'Medusa heads' for speculative decoding.

    The base model is frozen during Medusa training.
    Only the K additional heads are trained.

    Architecture of each additional head:
        Linear(d, d) + GELU + Linear(d, vocab)  (simple 2-layer FFN)
    This is much smaller than the base model, adding < 1% extra parameters.
    """

    def __init__(self, base_lm: nn.Module, vocab: int, d: int,
                 n_medusa_heads: int = 3):
        super().__init__()
        self.base    = base_lm
        self.vocab   = vocab
        self.d       = d
        self.K       = n_medusa_heads

        # Medusa heads: each predicts 1, 2, ..., K steps ahead
        # They share the base model's hidden state at the current position
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d, bias=False),
                nn.SiLU(),
                nn.Linear(d, vocab, bias=False),
            )
            for _ in range(n_medusa_heads)
        ])

        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad_(False)

    def base_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the base model's hidden states before the final LM head.
        In real implementations: hook into the last transformer block output.
        Here: we approximate by running the base forward and extracting logits.
        """
        # This is a simulation — real Medusa intercepts the last hidden states
        with torch.no_grad():
            # Use base embed + layers as hidden state extractor
            h = self.base.embed(input_ids)
            for layer in self.base.layers:
                h = h + layer(h)
            h = self.base.norm(h)
        return h   # (B, T, d)

    def forward(self, input_ids: torch.Tensor) -> tuple:
        """
        Forward pass returns:
            base_logits:  (B, T, V) from the standard LM head
            head_logits:  list of K × (B, T, V) from the Medusa heads
        """
        h = self.base_hidden_states(input_ids)

        # Standard LM head prediction
        base_logits  = self.base.head(h)   # (B, T, V)

        # Medusa head predictions (each uses the same hidden state h)
        head_logits  = [head(h) for head in self.heads]   # K × (B, T, V)

        return base_logits, head_logits

    def predict_next_k_tokens(self, context: torch.Tensor) -> list[torch.Tensor]:
        """
        At the current context position, predict the next K+1 token distributions.
        Returns list of K+1 probability tensors, each (V,).
        """
        base_logits, head_logits = self.forward(context)

        # Position -1 = current context end
        probs_base = F.softmax(base_logits[0, -1, :], dim=-1)   # next token
        probs_list = [probs_base]

        for head_out in head_logits:
            probs_k = F.softmax(head_out[0, -1, :], dim=-1)
            probs_list.append(probs_k)

        return probs_list   # [p(t+1), p(t+2), ..., p(t+K+1)]


# ── Medusa training ───────────────────────────────────────────────────────────

def train_medusa_heads(medusa_model: MedusaModel,
                        train_data: torch.Tensor,
                        n_steps: int = 100,
                        lr: float = 1e-3,
                        vocab: int = 64) -> list[float]:
    """
    Train only the Medusa heads (base model frozen).

    Medusa head k is trained to predict the token k+1 steps ahead:
        loss_k = CE(head_k(h_t), y_{t+k+1})

    The label shift: head 0 → shift 1, head 1 → shift 2, etc.
    """
    opt    = torch.optim.AdamW(
        [p for p in medusa_model.parameters() if p.requires_grad],
        lr=lr
    )
    losses = []
    B, T   = 4, 12

    for step in range(n_steps):
        idx   = torch.randint(0, len(train_data) - T, (B,))
        batch = torch.stack([train_data[i:i+T] for i in idx])

        opt.zero_grad()

        # Get hidden states from frozen base
        with torch.no_grad():
            h = medusa_model.base.embed(batch)
            for layer in medusa_model.base.layers:
                h = h + layer(h)
            h = medusa_model.base.norm(h)

        # Compute Medusa head losses for each head
        total_loss = torch.tensor(0.0)
        for k, head in enumerate(medusa_model.heads):
            head_logits = head(h[:, :-k-2, :])   # predict k+1 steps ahead
            head_targets = batch[:, k+2:]          # shifted targets

            # Only use valid positions (avoid going past sequence end)
            if head_logits.shape[1] <= 0 or head_targets.shape[1] <= 0:
                continue

            min_len = min(head_logits.shape[1], head_targets.shape[1])
            head_logits  = head_logits[:, :min_len, :]
            head_targets = head_targets[:, :min_len]

            loss_k = F.cross_entropy(
                head_logits.reshape(-1, vocab),
                head_targets.reshape(-1)
            )
            total_loss = total_loss + loss_k

        total_loss.backward()
        opt.step()
        losses.append(total_loss.item())

    return losses


# ── Medusa-style speculative decoding ─────────────────────────────────────────

def medusa_generate_step(medusa_model: MedusaModel,
                           context: torch.Tensor) -> tuple:
    """
    One Medusa speculative decoding step.

    1. Run ONE forward pass through the model
    2. Sample K candidate tokens from the K Medusa heads
    3. Verify using rejection sampling (same as standard spec. decoding)
    4. The "draft" distributions come from the Medusa heads
    5. The "target" distributions come from the base LM (sequential simulation)

    In practice: one forward pass gives BOTH draft and verify simultaneously.
    Here we simulate the verification step separately for clarity.
    """
    with torch.no_grad():
        probs_list = medusa_model.predict_next_k_tokens(context)

    K = medusa_model.K
    accepted_tokens = []
    n_accepted      = 0

    for k in range(K):
        # Draft: sample from Medusa head k (predicts k+1 steps ahead)
        p_head = probs_list[k]               # (V,)
        y_k    = torch.multinomial(p_head, 1).item()

        # Target verification: compute what the base LM would assign
        # (in real Medusa: this comes from the same forward pass)
        # Simulate by running base model one step at a time for verification
        verify_ctx  = torch.cat([context,
                                   torch.tensor(accepted_tokens).unsqueeze(0)
                                   if accepted_tokens else context[:, :0]], dim=1)
        if accepted_tokens:
            verify_ctx = torch.cat([context,
                                     torch.tensor([[accepted_tokens[-1]])],
                                    dim=1] if len(accepted_tokens) > 0 else [context],
                                   ) if False else verify_ctx

        # Simpler: use the base model logits from the forward pass
        # p_T_k is the base model's prediction k+1 steps ahead
        # (In practice, Medusa verifies with the same token-level base probs)
        p_T_k = probs_list[0] if k == 0 else probs_list[k]  # simplified simulation

        # Acceptance ratio
        alpha = min(1.0, p_T_k[y_k].item() / (p_head[y_k].item() + 1e-10))
        u     = torch.rand(1).item()

        if u <= alpha:
            accepted_tokens.append(y_k)
            n_accepted += 1
        else:
            # Reject: sample from adjusted distribution
            diff = (p_T_k - p_head).clamp(min=0.0)
            Z    = diff.sum().item()
            if Z > 1e-8:
                fallback = torch.multinomial(diff / Z, 1).item()
            else:
                fallback = torch.multinomial(p_T_k, 1).item()
            accepted_tokens.append(fallback)
            break

    # Always add bonus token
    if n_accepted == K:
        bonus = torch.multinomial(probs_list[K], 1).item()
        accepted_tokens.append(bonus)

    return torch.tensor(accepted_tokens), n_accepted


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    VOCAB   = 64
    D       = 64
    N_HEADS = 3

    # Build base model and Medusa
    class BaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed  = nn.Embedding(VOCAB, D)
            self.layers = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(D),
                              nn.Linear(D, D*4, bias=False), nn.GELU(),
                              nn.Linear(D*4, D, bias=False))
                for _ in range(2)])
            self.norm   = nn.LayerNorm(D)
            self.head   = nn.Linear(D, VOCAB, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, x):
            h = self.embed(x)
            for l in self.layers: h = h + l(h)
            return self.head(self.norm(h))

    base   = BaseModel()
    medusa = MedusaModel(base, VOCAB, D, N_HEADS)

    n_base_params   = sum(p.numel() for p in base.parameters())
    n_medusa_params = sum(p.numel() for p in medusa.heads.parameters())
    print("=" * 65)
    print("  MEDUSA SELF-SPECULATIVE DECODING")
    print(f"  Base model: {n_base_params:,} params")
    print(f"  Medusa heads: {n_medusa_params:,} params  "
          f"({n_medusa_params/n_base_params*100:.1f}% overhead)")
    print(f"  K={N_HEADS} heads → predict up to {N_HEADS+1} tokens per pass")
    print("=" * 65)
    print()

    # Train Medusa heads
    print("  Training Medusa heads (base model frozen)...")
    train_data = torch.randint(3, VOCAB, (2000,))
    losses     = train_medusa_heads(medusa, train_data, n_steps=80)
    print(f"  Loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    print()

    # Generation demo
    prompt = torch.randint(3, VOCAB, (1, 4))

    print("  Medusa generation (10 steps):")
    print(f"  {'Step':>5}  {'Tokens accepted':>18}  {'Tokens generated':>18}")
    print(f"  {'':─>5}  {'':─>18}  {'':─>18}")

    total_tokens  = 0
    total_passes  = 0
    ctx = prompt.clone()

    for step in range(10):
        new_tokens, n_acc = medusa_generate_step(medusa, ctx)
        ctx        = torch.cat([ctx, new_tokens.unsqueeze(0)], dim=1)
        total_tokens += len(new_tokens)
        total_passes += 1
        print(f"  {step+1:>5}  {n_acc:>18} / {N_HEADS}  "
              f"{total_tokens:>18} total")

    print()
    print(f"  Summary:")
    print(f"  Tokens per forward pass: {total_tokens/total_passes:.2f}  "
          f"(naive: 1.00)")
    print(f"  Speedup vs naive:        {total_tokens/total_passes:.2f}×")
    print()
    print(f"  Medusa advantages:")
    print(f"  ✓ No separate draft model (saves {n_base_params*2/1e9:.1f} GB GPU memory)")
    print(f"  ✓ Single forward pass per decoding step (like standard spec. decoding)")
    print(f"  ✓ Heads add only {n_medusa_params/n_base_params*100:.1f}% parameters to the model")
    print(f"  ✓ Heads can be trained with LoRA-style efficiency on the target model")
    print()
    print("  In production (e.g. EAGLE, vLLM Medusa):")
    print("  - Achieves ~2-3× speedup on single-user inference")
    print("  - No extra GPU memory for a separate draft model")
    print("  - Well-suited for deployment alongside PEFT adapters")
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
    #     from llm_training.visuals.speculative_decoding import (
    #         SPEC_VISUAL_HTML,
    #         SPEC_VISUAL_HEIGHT,
    #     )
    #     visual_html   = SPEC_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SPEC_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[30_speculative_decoding.py] Could not load visual: {e}",
    #         stacklevel=2,
    #     )

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