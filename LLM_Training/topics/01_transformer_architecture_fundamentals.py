"""
Transformer Architecture Fundamentals
======================================

The Transformer is the foundational architecture behind every modern large
language model. Understanding its structure — from the embedding table to
the final logit projection — is the prerequisite for every other topic in
this collection.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Transformer Architecture Fundamentals"
DISPLAY_NAME = "01 · Transformer Architecture"
ICON         = "🏗️"
SUBTITLE     = "Encoder, Decoder, and the Full Forward Pass"


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

### What Is a Transformer?

The Transformer is a neural network architecture introduced in the 2017 paper
*"Attention Is All You Need"* (Vaswani et al.). Before it, sequence modelling
was dominated by RNNs and LSTMs, which processed tokens one at a time — making
it impossible to parallelise training across the length dimension and hard to
model long-range dependencies.

The Transformer's key insight was radical: **throw away recurrence entirely**.
Instead of processing left-to-right, let every token attend directly to every
other token in the sequence simultaneously. This single idea unlocked massive
parallelism (training on thousands of GPUs at once) and near-perfect long-range
memory (every token sees every other token in one step).

Every modern LLM — GPT-4, LLaMA, Gemini, Mistral, Claude — is a Transformer or
a direct descendant. Understanding this architecture is therefore the foundation
for everything else in this course.


### The Two Flavours: Encoder vs Decoder

The original paper described an **Encoder–Decoder** architecture designed for
machine translation:

    •   Encoder: reads the source sentence and builds a rich contextual
        representation. Uses *bidirectional* self-attention — every token can
        see every other token in both directions. Examples: BERT, RoBERTa.

    •   Decoder: generates the target sentence, one token at a time. Uses
        *causal* (masked) self-attention — each token can only attend to past
        tokens. Also has cross-attention to query the encoder's output.
        Examples: GPT series.

    •   Encoder–Decoder: both components combined. Still used for
        seq2seq tasks. Examples: T5, BART, mT5.

Modern LLMs for language generation are **decoder-only** — the encoder half is
dropped entirely. The remainder of this module focuses on the decoder-only
Transformer, which is what you will train.


### The 30,000-Foot View: One Forward Pass

Here is the complete journey a batch of token IDs takes through a decoder-only
Transformer:

    ```
    Token IDs  →  Embedding Table  →  + Positional Encoding
               →  [ Transformer Block × N ]
               →  Final LayerNorm
               →  Logit Projection (unembed)
               →  Softmax  →  Next-Token Probabilities
    ```

Each Transformer Block contains exactly two sub-layers:
    1. Multi-Head Self-Attention  (MHSA)
    2. Feed-Forward Network       (FFN)

Each sub-layer is wrapped with a residual connection and a normalisation layer.
That is the entire recipe. Let's build it up from scratch.


    **Diagram 1 — Decoder-Only Transformer, Full Stack:**

    DECODER-ONLY TRANSFORMER
    ════════════════════════════════════════════════════════════════

    Input token IDs: [42, 1891, 307, 257, ...]
          │
          ▼
    ┌─────────────────────────────────────┐
    │   Token Embedding  E ∈ ℝ^(V × d)    │  Look up a d-dim vector per token
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │   + Positional Encoding             │  Add position information
    └─────────────────────────────────────┘
          │
          ▼
    ┌═════════════════════════════════════╗
    ║   Transformer Block  ×  N layers    ║
    ║  ┌───────────────────────────────┐  ║
    ║  │  LayerNorm                    │  ║
    ║  │  Multi-Head Self-Attention    │  ║
    ║  │  + Residual                   │  ║
    ║  ├───────────────────────────────┤  ║
    ║  │  LayerNorm                    │  ║
    ║  │  Feed-Forward Network (MLP)   │  ║
    ║  │  + Residual                   │  ║
    ║  └───────────────────────────────┘  ║
    ╚═════════════════════════════════════╝
          │
          ▼
    ┌─────────────────────────────────────┐
    │   Final LayerNorm                   │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │   Logit Projection  W ∈ ℝ^(d × V)   │  Project back to vocabulary size
    └─────────────────────────────────────┘
          │
          ▼
    Next-token logits: [v₁, v₂, ..., v_V]   (one score per vocabulary item)


### Step 1 — Token Embeddings

The first step converts discrete token IDs into continuous vectors. The model
maintains an **embedding table** E of shape (V, d_model) where:

    •   V  = vocabulary size   (typically 32 000 – 128 000)
    •   d  = model dimension   (e.g. 512 for small, 4096 for LLaMA-7B, 8192 for LLaMA-70B)

For a sequence of T tokens, the embedding layer performs T independent table
lookups, producing a matrix X ∈ ℝ^(T × d).

This table is learned — it starts randomly initialised and the embeddings are
updated by backpropagation just like any other weight matrix.

**Weight Tying:** Many models (GPT-2, LLaMA) reuse the same matrix for both the
input embedding and the final logit projection (transposed). This halves the
parameter count for this component and often improves performance, because the
model learns that "what goes in" and "what comes out" share the same semantic
space.


### Step 2 — Positional Encoding

Self-attention is permutation-invariant: if you shuffle the tokens, the
attention computation produces the same result (up to shuffled outputs). The
model has no built-in notion of "this token comes before that one."

Positional encodings inject position information. The original paper used fixed
sinusoidal functions; modern LLMs typically use learned or rotary encodings
(covered in Module 03). For now, the key idea is:

    X_pos = X_embed + PE

where PE ∈ ℝ^(T × d) encodes each position 0, 1, …, T-1.


### Step 3 — The Transformer Block

This is the heart of the architecture. A single Transformer Block transforms
its input tensor of shape (T, d) into an output tensor of the same shape,
stacking rich contextual information with each pass.

**3a — Multi-Head Self-Attention (MHSA)**

Attention asks: "For each token, which other tokens are most relevant to
understanding it?" It computes this as a weighted sum over all tokens, where
the weights are determined by learned similarity.

For a single attention head:

    Q = X · W_Q    (Query: what am I looking for?)
    K = X · W_K    (Key:   what do I contain?)
    V = X · W_V    (Value: what do I contribute if matched?)

    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

    where d_k = d_model / num_heads

The scaling factor √d_k prevents the dot products from growing large in
magnitude (which would push softmax into near-zero gradient regions).

In **Multi-Head** attention, this operation is run h times in parallel with
different learned projection matrices, allowing the model to simultaneously
attend to different aspects of the input (syntax, coreference, semantics, etc.).
The h outputs are concatenated and projected back to d_model.

**Causal Masking:** In decoder-only models, the attention matrix is masked so
that position i cannot attend to any position j > i. This is implemented by
setting those entries to −∞ before the softmax, ensuring they contribute zero
weight. This is what makes next-token prediction possible: at training time,
all positions are processed in parallel, but each position only sees past tokens.


    **Diagram 2 — Single Attention Head:**

    SINGLE ATTENTION HEAD
    ════════════════════════════════════════════════════════════════

    Input X ∈ ℝ^(T × d)
    │
    ├──── × W_Q  ──►  Q ∈ ℝ^(T × d_k)   "What am I looking for?"
    ├──── × W_K  ──►  K ∈ ℝ^(T × d_k)   "What do I contain?"
    └──── × W_V  ──►  V ∈ ℝ^(T × d_k)   "What do I contribute?"

                        Q · Kᵀ
    Scores  =  ─────────────────   ∈ ℝ^(T × T)
                         √d_k

    For each row i: score[i, j] = "how relevant is token j to token i?"

    Causal mask (decoder):
    ┌                           ┐
    │  s₀₀   -∞   -∞   -∞  … │   position 0 sees only itself
    │  s₁₀  s₁₁   -∞   -∞  … │   position 1 sees 0 and 1
    │  s₂₀  s₂₁  s₂₂   -∞  … │   position 2 sees 0, 1, 2
    │  s₃₀  s₃₁  s₃₂  s₃₃  … │   position 3 sees 0, 1, 2, 3
    └                           ┘

    Weights = softmax(Masked Scores)  ∈ ℝ^(T × T)

    Output  = Weights · V             ∈ ℝ^(T × d_k)


    **Diagram 3 — Multi-Head Attention:**

    MULTI-HEAD ATTENTION
    ════════════════════════════════════════════════════════════════

    Input X
    │
    ├── Head 1: Q₁K₁V₁ ──► out₁ ∈ ℝ^(T × d_k)   "syntactic role?"
    ├── Head 2: Q₂K₂V₂ ──► out₂ ∈ ℝ^(T × d_k)   "coreference?"
    ├── Head 3: Q₃K₃V₃ ──► out₃ ∈ ℝ^(T × d_k)   "semantic type?"
    │     ⋮
    └── Head h: QₕKₕVₕ ──► outₕ ∈ ℝ^(T × d_k)

    Concat([out₁, out₂, …, outₕ])  ∈ ℝ^(T × d_model)
           │
           × W_O  ──►  MHSA output  ∈ ℝ^(T × d_model)

    d_k = d_model / h    (each head operates in a subspace)


**3b — The Residual Connection**

After MHSA, the output is added back to the input:

    X = X + MHSA(LayerNorm(X))

This is the **residual** or **skip connection** (introduced in ResNets, 2015).
It is not a minor implementation detail — it is architecturally critical:

    •   It lets gradients flow directly from the loss to early layers without
        vanishing through dozens of non-linearities.
    •   It creates a "gradient highway" that makes training very deep networks
        (100+ layers) tractable.
    •   It means each Transformer Block only needs to learn a *correction* to
        its input, not a full transformation from scratch.

Without residual connections, 48-layer GPT-3 scale models would be untrainable.


**3c — Layer Normalisation**

LayerNorm normalises each token's feature vector independently:

    LayerNorm(x) = γ · (x − μ) / (σ + ε) + β

where μ and σ are computed over the d_model features of that single token, and
γ, β are learned scale and shift parameters.

**Pre-Norm vs Post-Norm:**
The original paper placed LayerNorm *after* the residual addition (Post-Norm).
Modern LLMs almost universally use **Pre-Norm** — LayerNorm is applied *before*
the sub-layer, with the residual bypassing it. Pre-Norm significantly improves
training stability at scale by keeping the residual path clean.

    Post-Norm:  X = LayerNorm( X + SubLayer(X) )    ← original paper
    Pre-Norm:   X = X + SubLayer( LayerNorm(X) )    ← modern LLMs (LLaMA etc.)


**3d — Feed-Forward Network (FFN)**

The second sub-layer in each Transformer Block is a two-layer MLP applied
independently to each token position:

    FFN(x) = activation( x · W₁ + b₁ ) · W₂ + b₂

Shapes:
    W₁ ∈ ℝ^(d_model × d_ff)    expand    (d_ff ≈ 4 × d_model typically)
    W₂ ∈ ℝ^(d_ff × d_model)    contract

Despite looking simple, the FFN component contains **~⅔ of all parameters**
in a standard Transformer. The attention layers handle routing — deciding which
tokens to mix — while the FFN layers do the heavy lifting of storing and
transforming information.

**Modern variant — SwiGLU / GeGLU:**
LLaMA, Mistral, and most modern LLMs replace the standard FFN with a *gated*
variant:

    FFN_SwiGLU(x) = ( SiLU(x · W₁) ⊙ (x · W_gate) ) · W₂

The gating mechanism (⊙ = element-wise product) gives the network fine-grained
control over which information flows through. To keep total parameters constant,
d_ff is typically reduced to ~(8/3) × d_model when using SwiGLU.


### Step 4 — Stacking Blocks (Depth)

A modern LLM stacks N identical Transformer Blocks:

    Model Size    Layers (N)    d_model    Heads    d_ff
    ─────────────────────────────────────────────────────
    GPT-2 Small       12          768        12     3072
    GPT-2 XL          48         1600        25     6400
    LLaMA-2 7B        32         4096        32    11008
    LLaMA-2 70B       80         8192        64    28672
    GPT-3 175B        96        12288        96    49152
    ─────────────────────────────────────────────────────

Each successive layer builds increasingly abstract representations. Early layers
tend to capture local syntax; later layers capture semantics and world knowledge.
This is why fine-tuning often targets later layers — they hold the highest-level
representations.


### Step 5 — Final LayerNorm and Logit Projection

After all N blocks:

    1.  Final LayerNorm is applied to the last hidden state.

    2.  The **logit projection** (also called the *language model head* or
        *unembedding matrix*) projects from d_model back to vocabulary size V:

            logits = H · Wᵤ    where Wᵤ ∈ ℝ^(d_model × V)

    3.  The resulting vector of V logits represents the model's unnormalised
        scores for each possible next token.

    4.  Softmax converts these to probabilities:

            P(next token = v | context) = exp(logits_v) / Σ exp(logits_i)

During training, these probabilities are compared to the ground-truth next
token using **cross-entropy loss**. The goal: make the correct next token's
probability as high as possible.


### Parameter Count Estimation

It is useful to be able to estimate a model's parameter count from its
hyperparameters. The dominant terms are:

    Embedding table:        V  × d
    Per-block attention:    4  × d² (Q, K, V, O projections, assuming d_k=d/h)
    Per-block FFN:          8  × d² (two weight matrices at 4×d width each)
    Per-block total:       ~12 × d²
    All N blocks:           N × 12 × d²
    Logit projection:       V  × d  (often weight-tied with embeddings)

    Total ≈ 12 × N × d²  +  2 × V × d   (ignoring biases and LayerNorm)

Example — LLaMA-2 7B:
    12 × 32 × 4096²  ≈  6.44 B params from Transformer blocks
    + 32 000 × 4096 × 2 ≈ 0.26 B from embeddings
    ≈ 6.7 B total  ✓ (matches the reported ~7B)

This rule of thumb (12Nd²) is a quick sanity check when reading papers or
designing model configurations.


    **Diagram 4 — Parameter Budget Breakdown (LLaMA-2 7B):**

    PARAMETER BUDGET BREAKDOWN
    ════════════════════════════════════════════════════════════════

    Component                    Params        % of Total
    ─────────────────────────────────────────────────────
    Token Embeddings             131 M          ~1.9 %
    ─────────────────────────────────────────────────────
    × 32 Transformer Blocks:
      Attention (Q,K,V,O)        536 M / block
      FFN (SwiGLU, W1,W2,Wgate) 905 M / block
      LayerNorm (tiny)             ~0 M
      Block subtotal:            ~45 M each
      All 32 blocks:            1440 M         ~85.5 %
    ─────────────────────────────────────────────────────
    Final LayerNorm                ~0 M
    Logit Projection             131 M (tied)   ~1.9 %
    ─────────────────────────────────────────────────────
    TOTAL                       ~6.7 B          100 %


### Why Does Depth Help More Than Width?

Depth (more layers, same d) vs. width (larger d, same layers) is an empirical
finding from scaling research:

    •   More layers → more sequential computation steps → model can reason
        through more "steps" of information transformation.
    •   Larger width → more capacity per step, but diminishing returns faster.
    •   Optimal scaling keeps depth and width growing together (roughly
        d ∝ √N in some analyses), which is reflected in the table above.

This is why you don't see models with d=32768 and N=2 — depth is not
replaceable by width alone.


### The Residual Stream View

A powerful mental model is to think of the Transformer as a sequence of
modifications to a **residual stream**:

    •   The residual stream starts as the token+position embeddings.
    •   Each attention and FFN sub-layer reads from the stream, computes a
        delta, and adds it back.
    •   The final hidden state is the sum of all these deltas.

This view (from Elhage et al., 2021, "A Mathematical Framework for Transformer
Circuits") makes it clear why residuals are load-bearing: they are the
communication channel that every layer reads and writes.

    Layer 0 Attn writes:   residual += attn_delta_0
    Layer 0 FFN  writes:   residual += ffn_delta_0
    Layer 1 Attn writes:   residual += attn_delta_1
    ...
    Final state: X₀ + Σ all deltas


### Key Hyperparameters and Their Effects

    d_model   — controls representational capacity; dominates parameter count
    N (layers)— controls depth of computation; key for complex reasoning
    h (heads) — number of attention heads; each head = d_model/h subspace
    d_ff      — FFN hidden size; usually 4× or 8/3× d_model
    V         — vocabulary size; larger V = finer token granularity, more params
    max_seq_len— context window; large contexts need efficient attention
    dropout   — regularisation; often 0 for LLM pre-training at large scale

Understanding these gives you the ability to read any model card or config.json
and immediately understand the architecture before loading a single weight.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Component Comparison: Attention vs FFN

| Property          | Self-Attention          | Feed-Forward Network        |
|-------------------|-------------------------|-----------------------------|
| Primary role      | Token mixing / routing  | Per-token transformation    |
| Parameter share   | ~33% of block           | ~67% of block               |
| Compute cost      | O(T² · d) in sequence   | O(T · d · d_ff)             |
| Cross-token?      | Yes — all-to-all        | No — each token independent |
| Bottleneck at     | Long sequences (T large)| Large d_ff                  |
| Parallelisable?   | Yes (within a layer)    | Yes (fully)                 |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Minimal Decoder-Only Transformer (from scratch)": {
        "description": "A complete, runnable decoder-only Transformer in pure PyTorch — embedding, causal attention, FFN, residuals, logit head.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DECODER-ONLY TRANSFORMER — BUILT FROM SCRATCH
================================================================================

A minimal but complete decoder-only Transformer implemented in pure PyTorch.
No HuggingFace, no external libraries beyond torch.

Architecture overview:
    Token IDs
        → Embedding + Positional Encoding
        → N × TransformerBlock (CausalSelfAttention + FFN + LayerNorm + Residual)
        → Final LayerNorm
        → Logit Projection
        → Cross-Entropy Loss

This is the exact same structure used by GPT-2, LLaMA (with minor variants),
and virtually every modern autoregressive LLM.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

class TransformerConfig:
    vocab_size   = 256       # byte-level for this demo
    d_model      = 128       # embedding / hidden dimension
    n_heads      = 4         # number of attention heads
    n_layers     = 2         # number of Transformer blocks
    d_ff         = 512       # FFN hidden dimension (4 × d_model)
    max_seq_len  = 64        # maximum context length
    dropout      = 0.0       # no dropout for this demo

    @property
    def d_head(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


# ── Causal Self-Attention ─────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Each token attends to all previous tokens (and itself) but NOT to future
    tokens. This is enforced by the causal mask, which sets future positions
    to -inf before the softmax.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_head
        self.d_model = cfg.d_model

        # Single fused projection for Q, K, V — faster than three separate ones
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.dropout   = nn.Dropout(cfg.dropout)

        # Register the causal mask as a buffer (not a parameter; not updated)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, D = x.shape  # batch, sequence length, d_model

        # 1. Compute Q, K, V for all heads in one go
        qkv = self.qkv_proj(x)                   # (B, T, 3*D)
        q, k, v = qkv.split(self.d_model, dim=-1) # each: (B, T, D)

        # 2. Reshape into (B, n_heads, T, d_head) for batched attention
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # shapes: (B, h, T, d_head)

        # 3. Scaled dot-product attention
        scale   = math.sqrt(self.d_head)
        scores  = (q @ k.transpose(-2, -1)) / scale  # (B, h, T, T)

        # 4. Apply causal mask: future positions → -inf → softmax → 0
        scores  = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)           # (B, h, T, T)
        weights = self.dropout(weights)

        # 5. Weighted sum over values
        out = weights @ v                              # (B, h, T, d_head)

        # 6. Concatenate heads and project back to d_model
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        return self.out_proj(out)


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FFN(nn.Module):
    """
    Two-layer MLP applied independently at each token position.

    Expand from d_model → d_ff, apply GELU, contract back to d_model.
    GELU (Gaussian Error Linear Unit) is smooth everywhere and works well
    in practice — used by GPT-2/3 and many other models.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_ff,    bias=False)
        self.fc2  = nn.Linear(cfg.d_ff,    cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One Transformer block = Pre-Norm MHSA + Pre-Norm FFN, both with residuals.

    Pre-Norm layout (used by modern LLMs):
        x = x + Attention( LayerNorm(x) )
        x = x + FFN(       LayerNorm(x) )
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention sub-layer + residual
        x = x + self.ffn( self.ln2(x))   # FFN sub-layer + residual
        return x


# ── Full Decoder-Only Transformer ─────────────────────────────────────────────

class DecoderTransformer(nn.Module):
    """
    Complete decoder-only Transformer language model.

    Input:  token IDs  (B, T)           long tensor
    Output: logits     (B, T, V)        float tensor (one logit per vocab item)

    Training loss: cross-entropy over all (T-1) next-token predictions in parallel.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed   = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks      = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_ln    = nn.LayerNorm(cfg.d_model)

        # Logit projection — weight-tied with token embedding (GPT-2 style)
        self.lm_head     = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        # 1. Embeddings
        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x  = self.token_embed(idx) + self.pos_embed(positions)        # (B, T, D)

        # 2. Transformer blocks
        x  = self.blocks(x)

        # 3. Final normalisation
        x  = self.final_ln(x)

        # 4. Project to vocabulary
        logits = self.lm_head(x)   # (B, T, V)

        # 5. Optionally compute loss
        loss = None
        if targets is not None:
            # Flatten: (B*T, V) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressively generate max_new_tokens tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]  # trim to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature    # last position only
            probs  = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = TransformerConfig()
    model = DecoderTransformer(cfg)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {total_params:,}  (trainable: {trainable:,})")

    # Print architecture
    print()
    print(model)
    print()

    # Forward pass with dummy data
    B, T  = 2, 16   # batch of 2, sequence length 16
    idx   = torch.randint(0, cfg.vocab_size, (B, T))
    tgts  = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, tgts)

    print(f"Input shape  : {idx.shape}")
    print(f"Logits shape : {logits.shape}  (B={B}, T={T}, V={cfg.vocab_size})")
    print(f"Training loss: {loss.item():.4f}  (random init, expected ≈ ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f})")

    # Generate a few tokens from a single prompt
    prompt  = torch.tensor([[65, 66, 67]])   # 3-token prompt
    output  = model.generate(prompt, max_new_tokens=10)
    print(f"\\nGeneration example:")
    print(f"  Prompt  : {prompt[0].tolist()}")
    print(f"  Output  : {output[0].tolist()}")
    print(f"  New tokens: {output[0, 3:].tolist()}")
''',
    },

    "Parameter Count Calculator": {
        "description": "Estimate the parameter count of any Transformer from its hyperparameters using the 12Nd² rule and exact formulas.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
TRANSFORMER PARAMETER COUNT CALCULATOR
================================================================================

Given a model config, compute the exact (and approximate) parameter count.
Useful for quick sanity checks when reading papers or designing architectures.
================================================================================
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name:        str
    vocab_size:  int
    d_model:     int
    n_layers:    int
    n_heads:     int
    d_ff:        int
    ffn_type:    str = "standard"   # "standard" or "swiglu"
    weight_tying: bool = True


def count_params(cfg: ModelConfig) -> dict:
    d, N, V, h = cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.n_heads
    d_ff = cfg.d_ff

    # --- Embeddings ---
    embed = V * d

    # --- Per-block attention ---
    # Q, K, V projections: each d × d  →  3d²
    # Output projection:   d × d        →  d²
    attn_per_block = 4 * d * d  # 4d²

    # --- Per-block FFN ---
    if cfg.ffn_type == "swiglu":
        # SwiGLU has 3 matrices: W1, W_gate (both d→d_ff) and W2 (d_ff→d)
        ffn_per_block = 3 * d * d_ff
    else:
        # Standard: W1 (d→d_ff) + W2 (d_ff→d)
        ffn_per_block = 2 * d * d_ff

    # LayerNorm parameters (2 × d per norm, 2 norms per block) — tiny
    ln_per_block = 2 * 2 * d

    block_total  = attn_per_block + ffn_per_block + ln_per_block
    all_blocks   = N * block_total

    # --- Final LN + head ---
    final_ln    = 2 * d
    lm_head     = 0 if cfg.weight_tying else V * d   # tied = no extra params

    total = embed + all_blocks + final_ln + lm_head

    # Rule-of-thumb approximation (ignores biases, LN, weight tying)
    approx = 12 * N * d ** 2

    return {
        "token_embed":     embed,
        "attn_all_blocks": N * attn_per_block,
        "ffn_all_blocks":  N * ffn_per_block,
        "layernorms":      N * ln_per_block + final_ln,
        "lm_head":         lm_head,
        "total_exact":     total,
        "approx_12Nd2":    approx,
        "pct_attn":        100 * N * attn_per_block / total,
        "pct_ffn":         100 * N * ffn_per_block  / total,
        "pct_embed":       100 * embed / total,
    }


def fmt(n):
    if n >= 1e9:  return f"{n/1e9:.2f} B"
    if n >= 1e6:  return f"{n/1e6:.1f} M"
    return f"{n:,}"


KNOWN_MODELS = [
    ModelConfig("GPT-2 Small",    50257,   768, 12, 12,  3072, "standard", True),
    ModelConfig("GPT-2 XL",       50257,  1600, 48, 25,  6400, "standard", True),
    ModelConfig("LLaMA-2 7B",     32000,  4096, 32, 32, 11008, "swiglu",  False),
    ModelConfig("LLaMA-2 13B",    32000,  5120, 40, 40, 13824, "swiglu",  False),
    ModelConfig("LLaMA-2 70B",    32000,  8192, 80, 64, 28672, "swiglu",  False),
    ModelConfig("Mistral 7B",     32000,  4096, 32, 32, 14336, "swiglu",  False),
]


if __name__ == "__main__":
    print(f"{'Model':<20} {'Exact':>10} {'12Nd² est':>10} {'% Attn':>8} {'% FFN':>8} {'% Embed':>8}")
    print("─" * 72)
    for cfg in KNOWN_MODELS:
        p = count_params(cfg)
        print(
            f"{cfg.name:<20} "
            f"{fmt(p['total_exact']):>10} "
            f"{fmt(p['approx_12Nd2']):>10} "
            f"{p['pct_attn']:>7.1f}% "
            f"{p['pct_ffn']:>7.1f}% "
            f"{p['pct_embed']:>7.1f}%"
        )
    print()
    print("Detailed breakdown for LLaMA-2 7B:")
    p = count_params(KNOWN_MODELS[2])
    for k, v in p.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.1f}%")
        else:
            print(f"  {k:<25} {fmt(v)}")
''',
    },

    "Attention Mask Visualiser": {
        "description": "Visualise the causal attention mask and softmax weights for a short sequence, step by step.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ATTENTION MASK AND WEIGHT VISUALISER
================================================================================

Shows exactly what the causal mask looks like, and traces one full attention
computation step-by-step for a tiny sequence.
================================================================================
"""

import math
import torch
import torch.nn.functional as F


def softmax_manual(x):
    """Numerically stable softmax (subtract max before exp)."""
    x = x - x.max()
    e = torch.exp(x)
    return e / e.sum()


def show_matrix(mat, title, fmt=".3f", col_labels=None, row_labels=None):
    T = mat.shape[0]
    labels = col_labels or [f"t{i}" for i in range(T)]
    print(f"\n  {title}")
    header = "        " + "  ".join(f"{l:>7}" for l in labels)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, row in enumerate(mat):
        rl = row_labels[i] if row_labels else f"  t{i}"
        vals = "  ".join(f"{v:{fmt}}" for v in row)
        print(f"  {rl:<6}  {vals}")


if __name__ == "__main__":
    T       = 5    # sequence length
    d_model = 8    # tiny model dimension
    n_heads = 2
    d_head  = d_model // n_heads

    torch.manual_seed(42)

    # ── 1. Causal mask ──────────────────────────────────────────────────────
    causal_mask = torch.tril(torch.ones(T, T))
    print("=" * 60)
    print("  CAUSAL MASK  (1 = allowed, 0 = masked/future)")
    print("=" * 60)
    show_matrix(causal_mask, "Lower-triangular mask:", fmt=".0f")

    # ── 2. Random Q and K for one head ─────────────────────────────────────
    Q = torch.randn(T, d_head)
    K = torch.randn(T, d_head)
    V = torch.randn(T, d_head)

    print("\n" + "=" * 60)
    print("  STEP-BY-STEP SINGLE-HEAD ATTENTION  (T=5, d_head=4)")
    print("=" * 60)

    # Step 1: raw dot products
    raw_scores = Q @ K.T                                     # (T, T)
    show_matrix(raw_scores, "Step 1 — Raw scores  Q·Kᵀ:")

    # Step 2: scale
    scaled = raw_scores / math.sqrt(d_head)
    show_matrix(scaled, f"Step 2 — Scaled  / √{d_head} = / {math.sqrt(d_head):.2f}:")

    # Step 3: apply causal mask
    masked = scaled.masked_fill(causal_mask == 0, float("-inf"))
    show_matrix(masked, "Step 3 — After causal mask  (−∞ = future):")

    # Step 4: softmax row-by-row
    weights = torch.stack([softmax_manual(masked[i]) for i in range(T)])
    show_matrix(weights, "Step 4 — Softmax weights  (rows sum to 1.0):")

    # Step 5: verify rows sum to 1
    row_sums = weights.sum(dim=-1)
    print(f"\n  Row sums: {row_sums.tolist()}  (all ≈ 1.0 ✓)")

    # Step 6: output = weights × V
    out = weights @ V
    print(f"\n  Step 5 — Output = Weights × V")
    print(f"  Shape: {out.shape}   (each of the {T} tokens has a new {d_head}-dim representation)")

    # ── 3. Intuition summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INTUITION SUMMARY")
    print("=" * 60)
    print(f"  Token t0 can only see: t0")
    print(f"  Token t1 can only see: t0, t1")
    print(f"  Token t2 can only see: t0, t1, t2")
    print(f"  Token t{T-1} can see all {T} tokens")
    print()
    print(f"  Future tokens are masked to −inf → softmax → 0 weight")
    print(f"  → No information leaks from future positions")
    print(f"  → All T positions can be processed IN PARALLEL at training time")
    print(f"     (the mask does NOT mean sequential processing)")
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
    #     from llm_training.visuals.transformer_architecture import (
    #         TRANSFORMER_VISUAL_HTML,
    #         TRANSFORMER_VISUAL_HEIGHT,
    #     )
    #     visual_html   = TRANSFORMER_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = TRANSFORMER_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[01_transformer_architecture_fundamentals.py] Could not load visual: {e}", stacklevel=2)

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