"""
Attention Variants — MHA, MQA, and Grouped-Query Attention
===========================================================

The original Transformer used Multi-Head Attention (MHA) where every head
has its own Q, K, and V projections. As models scaled and inference speed
became critical, two memory-efficient variants emerged: Multi-Query Attention
(MQA) and Grouped-Query Attention (GQA). Understanding all three is essential
because the choice directly determines KV-cache size, memory bandwidth, and
throughput at inference time.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Attention Variants — MHA, MQA, GQA"
DISPLAY_NAME = "04 · Attention Variants"
ICON         = "👁️"
SUBTITLE     = "MHA, MQA, and Grouped-Query Attention"


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

### The Attention Memory Problem

During autoregressive generation, a language model processes tokens one at a
time. At each new token, it must attend to every previous token. To avoid
recomputing those keys and values from scratch on every step, they are stored
in the **KV-cache** — a buffer that grows with every generated token.

For a model with:
    •   n_heads = 32 attention heads
    •   d_head  = 128 features per head
    •   seq_len = 4096 tokens
    •   batch   = 32 concurrent requests
    •   dtype   = fp16 (2 bytes)

KV-cache size per layer:
    2 × n_heads × d_head × seq_len × batch × 2 bytes
    = 2 × 32 × 128 × 4096 × 32 × 2
    ≈ 2.1 GB per Transformer layer

For a 32-layer model: ~67 GB just for the KV-cache. On an 80 GB A100,
this leaves almost nothing for the model weights, drastically limiting
batch size and therefore throughput.

The solution: reduce the number of K and V heads while keeping Q heads the
same. This is the core idea behind MQA and GQA.


### Multi-Head Attention (MHA)

MHA is the original formulation from Vaswani et al. 2017. Each of the h
attention heads has its own independent projection matrices:

    For head k:
        Q_k = X · W_Q_k    W_Q_k ∈ ℝ^(d_model × d_head)
        K_k = X · W_K_k    W_K_k ∈ ℝ^(d_model × d_head)
        V_k = X · W_V_k    W_V_k ∈ ℝ^(d_model × d_head)

    head_k = Attention(Q_k, K_k, V_k)
    MHA(X) = Concat(head_1, …, head_h) · W_O

where d_head = d_model / h.

Total projection parameters:
    Q: h × d_model × d_head = d_model²
    K: h × d_model × d_head = d_model²
    V: h × d_model × d_head = d_model²
    O: d_model × d_model     = d_model²
    Total: 4 × d_model²

Total KV-cache size: proportional to h × d_head = d_model per layer.


    **Diagram 1 — MHA: Every Head Has Full Q, K, V:**

    MULTI-HEAD ATTENTION (MHA)   — h=4 heads shown
    ════════════════════════════════════════════════════════════════

    Input X ∈ ℝ^(T × d_model)
    │
    ├──────────────────────────────────────────────────────┐
    │                                                      │
    ▼                                                      ▼
    Q projections          K projections          V projections
    ┌──────┐               ┌──────┐               ┌──────┐
    │ W_Q1 │ → Q₁          │ W_K1 │ → K₁          │ W_V1 │ → V₁
    ├──────┤               ├──────┤               ├──────┤
    │ W_Q2 │ → Q₂          │ W_K2 │ → K₂          │ W_V2 │ → V₂
    ├──────┤               ├──────┤               ├──────┤
    │ W_Q3 │ → Q₃          │ W_K3 │ → K₃          │ W_V3 │ → V₃
    ├──────┤               ├──────┤               ├──────┤
    │ W_Q4 │ → Q₄          │ W_K4 │ → K₄          │ W_V4 │ → V₄
    └──────┘               └──────┘               └──────┘
                                │                      │
                           KV-cache                KV-cache
                           (4 K heads)             (4 V heads)
    Attn₁(Q₁, K₁, V₁)   Attn₂(Q₂, K₂, V₂)  ...
          │                      │
          └──── Concat ──────────┘ → × W_O → output

    KV-cache: 4 heads × 2 (K+V) × T × d_head  per layer


### Multi-Query Attention (MQA)

MQA (Shazeer, 2019) keeps h independent Q projections (one per head) but
collapses K and V down to **a single shared projection** used by all heads:

    For all heads k:
        Q_k = X · W_Q_k    (k independent Q projections)
        K   = X · W_K      (ONE shared K projection)
        V   = X · W_V      (ONE shared V projection)

    head_k = Attention(Q_k, K, V)   ← all heads use the same K and V

KV-cache reduction:
    MHA:  h × d_head per K, per V → h × d_head total K, h × d_head total V
    MQA:  1 × d_head per K, per V → d_head total K, d_head total V
    Reduction factor: h×  (for h=32 heads → 32× smaller KV-cache)

Parameter reduction:
    K projection: d_model² → d_model × d_head    (save ~(h-1)/h of W_K)
    V projection: d_model² → d_model × d_head    (save ~(h-1)/h of W_V)
    Total params: 4d² → (h+2)/h × d²  ≈ 2.06d² for h=32

Quality impact: MQA's KV sharing means different Q heads look through
the same "keyhole" for keys and values. Empirically, MQA training quality
is slightly lower than MHA at the same model size, but the throughput
improvement at inference is so large (2–10× in practice) that MQA dominated
for a period. Falcon, PaLM, StarCoder all use MQA.


    **Diagram 2 — MQA: One K and V Shared Across All Heads:**

    MULTI-QUERY ATTENTION (MQA)   — h=4 Q heads, 1 K/V head
    ════════════════════════════════════════════════════════════════

    Input X
    │
    ├── Q projections (h=4):          ├── K projection (1):  ├── V projection (1):
    │   ┌──────┐                      │   ┌──────┐           │   ┌──────┐
    │   │ W_Q1 │ → Q₁                 │   │ W_K  │ → K  ─────┼───│ W_V  │ → V
    │   ├──────┤                      │   └──────┘           │   └──────┘
    │   │ W_Q2 │ → Q₂                 │        ↓             │        ↓
    │   ├──────┤                      │     KV-cache         │     KV-cache
    │   │ W_Q3 │ → Q₃                 │   (1 head only!)     │   (1 head only!)
    │   ├──────┤                      │        │             │        │
    │   │ W_Q4 │ → Q₄                 │        └─────────────┘        │
    │   └──────┘                      │              │                 │
    │       │                         │              ▼                 ▼
    └───────┼─────────────────────────┘
            │
    Attn₁(Q₁, K, V)   ← all heads share the same K and V
    Attn₂(Q₂, K, V)
    Attn₃(Q₃, K, V)
    Attn₄(Q₄, K, V)
            │
         Concat → × W_O → output

    KV-cache: 1 head × 2 (K+V) × T × d_head  per layer   (4× smaller!)


### Grouped-Query Attention (GQA)

GQA (Ainslie et al., 2023) is the sweet spot between MHA and MQA. Instead
of one KV head (MQA) or h KV heads (MHA), GQA uses **g groups** of KV heads,
where 1 ≤ g ≤ h. Each group of (h/g) Q heads shares one K and one V head.

    n_kv_heads = g            (number of KV heads, g divides h evenly)
    heads_per_group = h / g   (Q heads that share one KV head)

    For KV group j (j = 0..g-1):
        K_j = X · W_K_j
        V_j = X · W_V_j

    For Q head k in group j:
        Q_k  = X · W_Q_k
        head_k = Attention(Q_k, K_j, V_j)   ← shares KV with its group

KV-cache size:
    MHA: h   × 2 × T × d_head
    GQA: g   × 2 × T × d_head    (g/h of MHA)
    MQA: 1   × 2 × T × d_head    (1/h of MHA)

For h=32, g=8:  GQA uses 8/32 = 25% of MHA's KV-cache.

**GQA is now the standard for frontier LLMs:**
    •   LLaMA-2 70B: h=64 Q heads, g=8 KV heads
    •   LLaMA-3 8B:  h=32 Q heads, g=8 KV heads
    •   Mistral 7B:  h=32 Q heads, g=8 KV heads
    •   Gemma:       h=8  Q heads, g=1 KV heads (MQA equivalent)

Special cases:
    g = h  →  MHA  (every Q head has its own KV head)
    g = 1  →  MQA  (all Q heads share one KV head)
    1 < g < h → GQA (interpolates between the two)


    **Diagram 3 — GQA: Grouped Sharing:**

    GROUPED-QUERY ATTENTION (GQA)   — h=8 Q heads, g=2 KV groups
    ════════════════════════════════════════════════════════════════

    Q heads:    Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇  Q₈
                │   │   │   │   │   │   │   │
                └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
                  │         │       │         │
              Group 0    Group 0  Group 1  Group 1
                  │         │       │         │
              ┌───┴───┐         ┌───┴───┐
              │ K₀ V₀ │         │ K₁ V₁ │
              └───────┘         └───────┘

    Q₁ and Q₂ share K₀, V₀    (Group 0)
    Q₃ and Q₄ share K₀, V₀    (Group 0)
    Q₅ and Q₆ share K₁, V₁    (Group 1)
    Q₇ and Q₈ share K₁, V₁    (Group 1)

    KV-cache: 2 groups × 2 (K+V) × T × d_head   (4× smaller than MHA)


    **Diagram 4 — KV-Cache Size Comparison:**

    KV-CACHE SIZE (h=32 heads, d_head=128, T=4096, batch=1, fp16)
    ════════════════════════════════════════════════════════════════

    Variant     KV heads    Cache / layer    Cache (32 layers)
    ─────────────────────────────────────────────────────────
    MHA          32          32 MB            1024 MB (~1 GB)
    GQA (g=8)     8           8 MB             256 MB
    GQA (g=4)     4           4 MB             128 MB
    MQA (g=1)     1           1 MB              32 MB (~32× smaller!)
    ─────────────────────────────────────────────────────────

    At batch=32: multiply all numbers by 32.
    → MHA:  ~32 GB just for KV cache
    → GQA:   ~8 GB
    → MQA:   ~1 GB


### How GQA Implements K/V Expansion

When implementing GQA in code, Q has shape (B, T, h, d_head) but K and V
have shape (B, T, g, d_head) where g < h. To compute attention, we need
K and V to match Q's head count.

Two implementation strategies:

**Strategy 1 — Repeat/expand (simple, more memory):**
    Repeat each KV head (h//g) times to get (B, T, h, d_head).
    This is what most reference implementations do for clarity.
    Memory cost: temporarily h heads × d_head for K and V.

**Strategy 2 — Reshape Q (efficient, less memory):**
    Instead of expanding K and V, reshape Q from
    (B, T, h, d_head) → (B, T, g, h//g, d_head)
    Then compute attention within each group:
    Q[..., group, :, :] vs K[..., group, :] and V[..., group, :]
    This avoids materialising the expanded K and V tensors.
    Used in production implementations (FlashAttention-2 supports this natively).


### Flash Attention — Efficient Implementation of All Variants

All three variants (MHA, MQA, GQA) have the same theoretical complexity but
differ in memory bandwidth requirements. The bottleneck in practice is not
FLOPs but **memory bandwidth**: reading Q, K, V matrices from GPU HBM is slow.

**FlashAttention** (Dao et al., 2022) restructures the attention computation
to use on-chip SRAM (much faster) by processing attention in tiles:

    Instead of:     materialise full (T × T) attention matrix → HBM read/write
    FlashAttention: stream tiles of Q, K, V through SRAM, accumulate output
                    Never write the full attention matrix to HBM

Result:
    •   Memory: O(T) instead of O(T²) — no full attention matrix in HBM
    •   Speed:  2–4× faster wall clock on long sequences
    •   Exact:  numerically identical to standard attention (online softmax)

FlashAttention-2 adds improved parallelism and explicit GQA support.
FlashAttention-3 targets Hopper (H100) architecture specifics.

All three attention variants (MHA, MQA, GQA) benefit from FlashAttention
because the bottleneck is memory bandwidth, not the number of heads.


### Converting MHA Models to GQA (Uptraining)

When Ainslie et al. introduced GQA, they also showed that an existing MHA
model can be **uptrained** to use GQA with relatively little compute:

    1.  Take the trained MHA model.
    2.  For each group of (h/g) KV heads, mean-pool them into one KV head.
        K_group = mean(K₁, K₂, …, K_{h/g})
    3.  Fine-tune on ~5% of the original training data to recover quality.

This is how LLaMA-2 70B-GQA was derived from the MHA checkpoint. The quality
recovery is near-complete because the mean-pooled initialisation is close to
the optimal GQA solution — the KV heads within a group were already partially
redundant in the MHA model.


### Sliding Window Attention (Bonus: Mistral's SWA)

Mistral 7B introduced **Sliding Window Attention** as a complement to GQA:
instead of attending to all previous tokens (O(T) KV-cache growth), each
token only attends to the W most recent tokens (fixed window W = 4096).

    •   KV-cache does not grow beyond W tokens: memory is O(W), not O(T).
    •   For generation beyond W tokens, old KV entries are evicted (rolling buffer).
    •   Information beyond W still flows through the network via stacking:
        with N layers and window W, the effective receptive field is N × W.

For Mistral 7B: W=4096, N=32 layers → receptive field = 131k tokens.
Combined with GQA (g=8), Mistral achieves excellent quality with dramatically
lower memory requirements than a comparable MHA model.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Attention Variant Comparison (h = total Q heads, g = KV head groups, d = d_model)

| Property              | MHA              | GQA                   | MQA              |
|-----------------------|------------------|-----------------------|------------------|
| Q heads               | h                | h                     | h                |
| K heads               | h                | g  (1 < g < h)        | 1                |
| V heads               | h                | g  (1 < g < h)        | 1                |
| KV-cache              | h × 2 × T × d_h | g × 2 × T × d_h       | 1 × 2 × T × d_h |
| KV params (W_K, W_V)  | 2d²              | 2d²×(g/h)             | 2d²/h            |
| Quality vs MHA        | Baseline         | Near-identical (g≥4)  | Slightly lower   |
| Used by               | GPT-2, original  | LLaMA-2/3, Mistral    | Falcon, PaLM     |
|                       | Transformer      | Gemma, Qwen           | StarCoder        |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "MHA, MQA, GQA — Side-by-Side Implementation": {
        "description": "All three attention variants implemented in a single unified PyTorch module with a shared interface. Shows exactly how KV head count changes.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MHA / MQA / GQA — UNIFIED IMPLEMENTATION
================================================================================

All three attention variants in one file, sharing the same interface.
The only difference: how many K and V heads are created.

    MHA:  n_kv_heads = n_heads         (every Q head has its own KV)
    GQA:  n_kv_heads = n_heads // g    (groups of Q heads share KV)
    MQA:  n_kv_heads = 1               (all Q heads share one KV)

This is exactly how LLaMA-3, Mistral, and other modern LLMs implement it.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Unified attention module supporting MHA, GQA, and MQA.

    Args:
        d_model:      total model dimension
        n_heads:      number of query heads
        n_kv_heads:   number of key/value heads
                      = n_heads  → MHA
                      = 1        → MQA
                      = anything in between → GQA
        max_seq_len:  for causal mask pre-allocation
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int = 512):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups   = n_heads // n_kv_heads   # Q heads per KV group
        self.d_head     = d_model // n_heads

        variant = (
            "MHA" if n_kv_heads == n_heads else
            "MQA" if n_kv_heads == 1       else
            f"GQA (g={n_kv_heads})"
        )
        self.variant = variant

        # Q projection: n_heads full projections
        self.Wq = nn.Linear(d_model, n_heads    * self.d_head, bias=False)
        # K, V: only n_kv_heads projections
        self.Wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        # Output projection: same for all variants
        self.Wo = nn.Linear(n_heads * self.d_head, d_model,    bias=False)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Project to Q, K, V
        q = self.Wq(x).view(B, T, self.n_heads,    self.d_head)  # (B,T,h,d_h)
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.d_head)  # (B,T,g,d_h)
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.d_head)  # (B,T,g,d_h)

        # 2. Expand K and V to match Q head count (for GQA/MQA)
        #    Each KV head is repeated n_groups times
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=2)  # (B,T,h,d_h)
            v = v.repeat_interleave(self.n_groups, dim=2)  # (B,T,h,d_h)

        # 3. Transpose for batched attention: (B, h, T, d_h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Scaled dot-product attention
        scale  = math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) / scale   # (B, h, T, T)

        # 5. Causal mask
        scores = scores.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)           # (B, h, T, T)
        out = weights @ v                              # (B, h, T, d_h)

        # 6. Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, h*d_h)
        return self.Wo(out)

    def kv_cache_size_bytes(self, seq_len: int, batch: int,
                            dtype_bytes: int = 2) -> int:
        """Estimate KV-cache memory usage in bytes."""
        # K cache + V cache
        return 2 * self.n_kv_heads * self.d_head * seq_len * batch * dtype_bytes

    def param_count(self) -> dict:
        return {
            "Wq": self.Wq.weight.numel(),
            "Wk": self.Wk.weight.numel(),
            "Wv": self.Wv.weight.numel(),
            "Wo": self.Wo.weight.numel(),
            "total": sum(p.numel() for p in self.parameters()),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

def fmt_bytes(n):
    if n >= 1e9:  return f"{n/1e9:.2f} GB"
    if n >= 1e6:  return f"{n/1e6:.1f} MB"
    return f"{n/1e3:.1f} KB"


if __name__ == "__main__":
    d_model     = 4096
    n_heads     = 32
    max_seq_len = 512

    configs = [
        ("MHA",        n_heads),     # all 32 KV heads
        ("GQA (g=8)",  8),           # 8 KV heads
        ("GQA (g=4)",  4),           # 4 KV heads
        ("MQA",        1),           # 1 KV head
    ]

    print("=" * 68)
    print("  ATTENTION VARIANT COMPARISON")
    print(f"  d_model={d_model}, n_heads={n_heads}, d_head={d_model//n_heads}")
    print("=" * 68)

    x = torch.randn(1, 16, d_model)  # batch=1, T=16 for quick test

    print(f"\n  {'Variant':<16} {'n_kv':>6}  {'W_K params':>12}  "
          f"{'W_V params':>12}  {'Total attn':>12}  {'KV cache @4096,b32':>20}")
    print(f"  {'':─<16} {'':─>6}  {'':─>12}  {'':─>12}  {'':─>12}  {'':─>20}")

    for name, n_kv in configs:
        attn = GroupedQueryAttention(d_model, n_heads, n_kv, max_seq_len)
        p    = attn.param_count()

        kv_cache = attn.kv_cache_size_bytes(seq_len=4096, batch=32)

        # Forward pass check
        out = attn(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

        print(f"  {name:<16} {n_kv:>6}  {p['Wk']:>12,}  "
              f"{p['Wv']:>12,}  {p['total']:>12,}  {fmt_bytes(kv_cache):>20}")

    print()
    print("  Note: Total attn params decrease because W_K and W_V shrink.")
    print("        MQA saves ~2×d² params from K+V projections.")
    print()

    # Detailed breakdown for GQA g=8
    attn_gqa = GroupedQueryAttention(d_model, n_heads, 8, max_seq_len)
    print("=" * 68)
    print("  DETAILED PARAMETER BREAKDOWN — GQA (g=8)")
    print("=" * 68)
    p = attn_gqa.param_count()
    for name, count in p.items():
        shape_info = ""
        if name == "Wq":
            shape_info = f"({d_model} × {n_heads * (d_model//n_heads)})"
        elif name in ("Wk", "Wv"):
            shape_info = f"({d_model} × {8 * (d_model//n_heads)})"
        elif name == "Wo":
            shape_info = f"({n_heads * (d_model//n_heads)} × {d_model})"
        print(f"  {name:<8} {count:>10,}  {shape_info}")
    print()

    # KV cache scaling demo
    print("=" * 68)
    print("  KV-CACHE SIZE AT DIFFERENT BATCH SIZES (seq_len=4096, fp16)")
    print("=" * 68)
    print(f"  {'Variant':<14}", end="")
    for bs in [1, 8, 32, 128]:
        print(f"  {'batch='+str(bs):>14}", end="")
    print()
    print(f"  {'':─<14}" + "  " + "  ".join(["─" * 14] * 4))

    for name, n_kv in configs:
        attn = GroupedQueryAttention(d_model, n_heads, n_kv, max_seq_len)
        print(f"  {name:<14}", end="")
        for bs in [1, 8, 32, 128]:
            sz = attn.kv_cache_size_bytes(4096, bs)
            print(f"  {fmt_bytes(sz):>14}", end="")
        print()
''',
    },

    "KV-Cache Simulation": {
        "description": "Simulate autoregressive generation with a KV-cache for MHA vs GQA — show how the cache grows and measure memory savings.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
KV-CACHE SIMULATION — MHA vs GQA MEMORY COMPARISON
================================================================================

Simulates the KV-cache mechanics during autoregressive generation:
    •   At each new token, compute new K and V for that position.
    •   Append to the KV-cache buffer (never recompute past tokens).
    •   Track memory usage as the sequence grows.

Demonstrates why GQA is critical for serving large batches.
================================================================================
"""

import math
import torch
import torch.nn.functional as F


class KVCache:
    """
    A simple KV-cache for one Transformer layer.

    Stores K and V tensors for all past positions.
    At each generation step, appends the new K/V and returns the full K/V
    so that the new query can attend to all past tokens.
    """

    def __init__(self, n_kv_heads: int, d_head: int, max_seq_len: int,
                 dtype: torch.dtype = torch.float16):
        self.n_kv_heads = n_kv_heads
        self.d_head     = d_head
        self.max_seq_len = max_seq_len
        self.dtype      = dtype
        self.current_len = 0

        # Pre-allocate the full cache (avoids repeated memory allocation)
        # Shape: (n_kv_heads, max_seq_len, d_head)
        self.k_cache = torch.zeros(n_kv_heads, max_seq_len, d_head, dtype=dtype)
        self.v_cache = torch.zeros(n_kv_heads, max_seq_len, d_head, dtype=dtype)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple:
        """
        Add new K/V for the current position and return the full cache.

        Args:
            new_k: (n_kv_heads, d_head) — K for the new token
            new_v: (n_kv_heads, d_head) — V for the new token

        Returns:
            k_full: (n_kv_heads, seq_len, d_head)
            v_full: (n_kv_heads, seq_len, d_head)
        """
        if self.current_len >= self.max_seq_len:
            raise RuntimeError(f"KV-cache full at max_seq_len={self.max_seq_len}")

        pos = self.current_len
        self.k_cache[:, pos, :] = new_k
        self.v_cache[:, pos, :] = new_v
        self.current_len += 1

        return (
            self.k_cache[:, :self.current_len, :],
            self.v_cache[:, :self.current_len, :],
        )

    @property
    def bytes_used(self) -> int:
        bytes_per_elem = 2 if self.dtype == torch.float16 else 4
        return (2 * self.n_kv_heads * self.current_len *
                self.d_head * bytes_per_elem)

    @property
    def bytes_allocated(self) -> int:
        bytes_per_elem = 2 if self.dtype == torch.float16 else 4
        return (2 * self.n_kv_heads * self.max_seq_len *
                self.d_head * bytes_per_elem)


def simulate_generation(n_heads: int, n_kv_heads: int, d_head: int,
                         n_layers: int, n_steps: int, batch_size: int,
                         name: str):
    """
    Simulate n_steps of autoregressive generation and track KV-cache memory.
    Returns a list of (step, total_bytes) tuples.
    """
    # One KV-cache per layer per batch element
    caches = [
        [KVCache(n_kv_heads, d_head, max_seq_len=n_steps)
         for _ in range(n_layers)]
        for _ in range(batch_size)
    ]

    history = []
    for step in range(n_steps):
        total_bytes = 0
        for b in range(batch_size):
            for layer in range(n_layers):
                # Fake new K, V for this step
                new_k = torch.randn(n_kv_heads, d_head, dtype=torch.float16)
                new_v = torch.randn(n_kv_heads, d_head, dtype=torch.float16)
                caches[b][layer].update(new_k, new_v)
                total_bytes += caches[b][layer].bytes_used

        history.append((step + 1, total_bytes))

    return history


def fmt_bytes(n: int) -> str:
    if n >= 1e9:  return f"{n/1e9:.2f} GB"
    if n >= 1e6:  return f"{n/1e6:.1f} MB"
    return f"{n/1e3:.1f} KB"


if __name__ == "__main__":
    # LLaMA-2 7B-style dimensions (scaled down for speed)
    N_HEADS   = 32
    D_HEAD    = 128
    N_LAYERS  = 32
    N_STEPS   = 512    # tokens to generate
    BATCH     = 8

    configs = [
        ("MHA",        N_HEADS),
        ("GQA (g=8)",  8),
        ("GQA (g=4)",  4),
        ("MQA",        1),
    ]

    print("=" * 65)
    print(f"  KV-CACHE GROWTH SIMULATION")
    print(f"  n_heads={N_HEADS}, d_head={D_HEAD}, n_layers={N_LAYERS}")
    print(f"  batch_size={BATCH}, generating {N_STEPS} tokens")
    print("=" * 65)

    all_histories = {}
    for name, n_kv in configs:
        history = simulate_generation(
            N_HEADS, n_kv, D_HEAD, N_LAYERS, N_STEPS, BATCH, name
        )
        all_histories[name] = history

    # Print table at key milestones
    milestones = [64, 128, 256, 512]
    print(f"\n  {'Step':>6}  " +
          "  ".join(f"{n:>15}" for n, _ in configs))
    print(f"  {'':─>6}  " + "  ".join(["─" * 15] * len(configs)))

    for step in milestones:
        row = f"  {step:>6}  "
        for name, _ in configs:
            hist_dict = dict(all_histories[name])
            row += f"{fmt_bytes(hist_dict[step]):>15}  "
        print(row)

    # Summary at final step
    print()
    print("=" * 65)
    print(f"  FINAL STATE AT {N_STEPS} TOKENS")
    print("=" * 65)

    mha_final = dict(all_histories["MHA"])[N_STEPS]
    for name, _ in configs:
        final_bytes = dict(all_histories[name])[N_STEPS]
        ratio = mha_final / final_bytes
        bar = "█" * int(ratio)
        print(f"  {name:<14}  {fmt_bytes(final_bytes):>10}  "
              f"  {ratio:.1f}× smaller than MHA  {bar}")

    print()
    print("  Practical implication for serving:")
    mha_batch_limit = 80e9 / (mha_final / BATCH)   # 80GB A100
    gqa_batch_limit = 80e9 / (dict(all_histories["GQA (g=8)"])[N_STEPS] / BATCH)
    print(f"  On an 80 GB A100 (KV-cache only, ignoring model weights):")
    print(f"  MHA     can serve ≈ {int(mha_batch_limit):>6} concurrent requests")
    print(f"  GQA g=8 can serve ≈ {int(gqa_batch_limit):>6} concurrent requests")
    print(f"  → {gqa_batch_limit/mha_batch_limit:.1f}× more concurrent users with GQA")
    print()
    print("  This is why GQA is now standard in production LLMs.")
''',
    },

    "Attention Head Analysis": {
        "description": "Inspect what individual attention heads learn — show head specialisation, redundancy patterns, and why MQA/GQA KV sharing works.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ATTENTION HEAD ANALYSIS
================================================================================

Analyses the diversity and specialisation of attention heads in MHA vs GQA.

Key questions:
    1. How similar are K/V projections across heads in a trained MHA model?
       (High similarity → redundancy → KV sharing is lossless)
    2. How does GQA's within-group head diversity compare to MHA?
    3. What attention patterns emerge from different heads?
================================================================================
"""

import math
import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened weight matrices."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat / (a_flat.norm() * b_flat.norm() + 1e-9)).item()


def analyse_head_similarity(weight_matrix: torch.Tensor, n_heads: int,
                             d_head: int, name: str):
    """
    Given a projection weight of shape (n_heads * d_head, d_model),
    split into per-head matrices and compute pairwise cosine similarities.
    """
    # Split into per-head matrices: list of (d_head, d_model)
    heads = weight_matrix.view(n_heads, d_head, -1)

    similarities = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            sim = cosine_similarity(heads[i], heads[j])
            similarities.append(sim)

    if not similarities:
        return

    avg_sim = sum(similarities) / len(similarities)
    max_sim = max(similarities)
    min_sim = min(similarities)

    print(f"  {name}:")
    print(f"    Avg pairwise cosine sim: {avg_sim:+.4f}")
    print(f"    Max:                     {max_sim:+.4f}")
    print(f"    Min:                     {min_sim:+.4f}")
    print(f"    Interpretation: {'High similarity → redundant → KV sharing OK' if avg_sim > 0.3 else 'Low similarity → diverse → GQA may lose some diversity'}")
    print()


def attention_pattern_demo():
    """
    Simulate different attention patterns that different heads might learn:
    - Local window (looks at last few tokens)
    - Global (looks at all tokens)
    - First-token sink (attends strongly to token 0)
    - Diagonal (attends to self and immediate neighbours)
    """
    T = 16

    # Pattern 1: local window (last 3 tokens)
    local = torch.zeros(T, T)
    for i in range(T):
        for j in range(max(0, i-2), i+1):
            local[i, j] = 1.0
    local = local / local.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    # Pattern 2: global (uniform over all past)
    global_pat = torch.tril(torch.ones(T, T))
    global_pat = global_pat / global_pat.sum(dim=-1, keepdim=True)

    # Pattern 3: first-token sink (strong attention to position 0)
    sink = torch.tril(torch.ones(T, T)) * 0.1
    sink[:, 0] += 0.5
    sink = sink / sink.sum(dim=-1, keepdim=True)

    # Pattern 4: previous token only
    prev = torch.zeros(T, T)
    for i in range(1, T):
        prev[i, i-1] = 1.0
    prev[0, 0] = 1.0

    patterns = {
        "Local window (last 3)":     local,
        "Global (uniform past)":     global_pat,
        "First-token sink":          sink,
        "Previous token":            prev,
    }

    def render_row(weights, T_show=12):
        chars = " ·▒█"
        out = ""
        for j in range(T_show):
            w = weights[j].item()
            idx = min(int(w * len(chars) * 3), len(chars) - 1)
            out += chars[idx] * 2
        return out

    print("=" * 62)
    print("  COMMON ATTENTION HEAD PATTERNS")
    print("=" * 62)
    print()
    print("  Each row = 'which past positions does token i attend to?'")
    print("  Dark = high weight, Light = low weight")
    print()

    for pat_name, pat in patterns.items():
        print(f"  Pattern: {pat_name}")
        for i in [0, 3, 7, 11, 15]:
            row = render_row(pat[i])
            print(f"    token {i:2d} │{row}│")
        print()


def gqa_head_grouping_analysis():
    """
    Simulate what happens when we group MHA heads for GQA conversion.
    Shows that within-group heads are more similar than across-group heads
    in well-structured models.
    """
    print("=" * 62)
    print("  GQA GROUPING ANALYSIS")
    print("=" * 62)
    print()
    print("  When converting MHA → GQA, we pool heads within a group.")
    print("  Ideally, heads in the same group should be similar.")
    print()

    n_heads = 8
    d_head  = 16
    n_groups = 2
    heads_per_group = n_heads // n_groups

    # Simulate heads with within-group similarity
    torch.manual_seed(42)
    heads = []
    for g in range(n_groups):
        base = torch.randn(d_head)          # group "prototype"
        for _ in range(heads_per_group):
            head = base + 0.2 * torch.randn(d_head)   # small variation
            heads.append(F.normalize(head, dim=0))

    print(f"  n_heads={n_heads}, n_groups={n_groups}, heads_per_group={heads_per_group}")
    print()
    print(f"  Cosine similarities:")
    print(f"  {'Pair':<22}  {'Similarity':>12}  {'Same group?':>12}")
    print(f"  {'':─<22}  {'':─>12}  {'':─>12}")

    for i in range(n_heads):
        for j in range(i+1, n_heads):
            sim = cosine_similarity(heads[i], heads[j])
            same_group = (i // heads_per_group) == (j // heads_per_group)
            marker = "✓ same" if same_group else "✗ diff"
            print(f"  Head {i} vs Head {j:<14}  {sim:>12.4f}  {marker:>12}")

    within = [cosine_similarity(heads[i], heads[j])
              for i in range(n_heads) for j in range(i+1, n_heads)
              if (i // heads_per_group) == (j // heads_per_group)]
    across = [cosine_similarity(heads[i], heads[j])
              for i in range(n_heads) for j in range(i+1, n_heads)
              if (i // heads_per_group) != (j // heads_per_group)]

    print()
    print(f"  Avg within-group similarity:  {sum(within)/len(within):.4f}")
    print(f"  Avg across-group similarity:  {sum(across)/len(across):.4f}")
    print()
    print(f"  Within-group heads are more similar → mean-pooling is valid!")
    print(f"  This is why GQA uptraining from MHA works with minimal quality loss.")


if __name__ == "__main__":
    # 1. Head similarity in random (untrained) weights — baseline
    print("=" * 62)
    print("  HEAD SIMILARITY — RANDOM INITIALISATION BASELINE")
    print("=" * 62)
    print()

    n_heads, d_head, d_model = 8, 32, 256
    torch.manual_seed(0)

    Wq_mha = torch.randn(n_heads * d_head, d_model) * 0.02
    Wk_mha = torch.randn(n_heads * d_head, d_model) * 0.02
    Wv_mha = torch.randn(n_heads * d_head, d_model) * 0.02

    analyse_head_similarity(Wq_mha, n_heads, d_head, "Q projections (MHA)")
    analyse_head_similarity(Wk_mha, n_heads, d_head, "K projections (MHA)")
    analyse_head_similarity(Wv_mha, n_heads, d_head, "V projections (MHA)")

    # 2. Common attention patterns
    attention_pattern_demo()

    # 3. GQA grouping analysis
    gqa_head_grouping_analysis()
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
    #     from llm_training.visuals.attention_variants import (
    #         ATTENTION_VISUAL_HTML,
    #         ATTENTION_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ATTENTION_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ATTENTION_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[04_attention_variants_mha_mqa_gqa.py] Could not load visual: {e}", stacklevel=2)

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