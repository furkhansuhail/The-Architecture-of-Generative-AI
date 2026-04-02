"""
Sequence Parallelism
=====================

Sequence parallelism extends tensor parallelism by distributing the sequence
dimension across GPUs. Where tensor parallelism splits weight matrices
(column/row), sequence parallelism splits the activation tensors along their
length dimension. This allows operations that are redundantly computed in
tensor-parallel training — LayerNorm, Dropout, the full-sequence attention
mask computation — to be parallelised, further reducing per-GPU activation
memory and enabling significantly longer context windows on the same hardware.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Sequence Parallelism"
DISPLAY_NAME = "21 · Sequence Parallelism"
ICON         = "🔗"
SUBTITLE     = "Distributing the Sequence Dimension"


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

### The Activation Memory Problem in Tensor Parallelism

Module 19 showed that tensor parallelism (TP) splits weight matrices across
GPUs, reducing weight memory by TP×. However, consider what happens to the
activation tensors in a TP=4 system:

    Between column-parallel and row-parallel layers, each GPU still holds
    the full input x ∈ ℝ^(B × T × d) — replicated on all 4 GPUs.
    LayerNorm operates on the full x before passing it to the QKV projection.
    Dropout operates on the full output tensor.

These replicated activations are wasteful: every GPU stores the same
(B × T × d) tensor, consuming TP× more memory than necessary.

**Sequence Parallelism (SP)** (Korthikanti et al., 2022) removes this
redundancy by splitting the activation tensor along the sequence dimension T
between tensor-parallel ranks:

    With TP=4:  each GPU holds x_t ∈ ℝ^(B × T/4 × d)  instead of full x

LayerNorm, Dropout, and other element-wise operations run independently on
each shard — no communication needed for these operations.


### How SP Integrates with TP

SP and TP use complementary collective operations:

    **Before column-parallel linear (W₁):**
    Each rank has x_t (B × T/4 × d). We need the full x (B × T × d) to
    compute Q/K/V projections. An **All-Gather** reconstructs the full sequence:
        All-Gather(x₀, x₁, x₂, x₃)  →  x_full on all ranks

    **After row-parallel linear (W₂):**
    Each rank has a full (B × T × d) partial sum z_t. Apply
    **Reduce-Scatter** to simultaneously sum and shard:
        Reduce-Scatter(z₀, z₁, z₂, z₃)  →  z_sharded_t on each rank
        where z_sharded_t ∈ ℝ^(B × T/4 × d)

The key insight: the All-Gather + Reduce-Scatter pattern is **volume-equivalent**
to two all-reduces. We don't increase communication volume — we exchange the
same amount of data but in a form that produces a distributed output.


    **Diagram 1 — SP + TP Data Flow:**

    SEQUENCE PARALLELISM + TENSOR PARALLELISM (TP=2)
    ════════════════════════════════════════════════════════════════

    In SP regions (LayerNorm, Dropout):
    GPU0 holds: x₀ ∈ ℝ^(B × T/2 × d)   ← seq shard 0
    GPU1 holds: x₁ ∈ ℝ^(B × T/2 × d)   ← seq shard 1

    LayerNorm(x₀) → norm₀                LayerNorm(x₁) → norm₁
    (no communication needed)

    ──── All-Gather ────────────────────────────────────────────

    GPU0: x_full ← [x₀, x₁]   GPU1: x_full ← [x₀, x₁]

    ──── TP Region (column-parallel Q/K/V) ─────────────────────

    GPU0: Q₀, K₀, V₀ = x_full · W_Q/K/V[:, :d/2]
    GPU1: Q₁, K₁, V₁ = x_full · W_Q/K/V[:, d/2:]

    Each GPU attends with its own heads over the FULL sequence T.

    ──── Row-parallel output + Reduce-Scatter ──────────────────

    GPU0: z₀_full = head₀ · W_O[:d/2, :]    (partial sum, full T)
    GPU1: z₁_full = head₁ · W_O[d/2:, :]    (partial sum, full T)

    Reduce-Scatter: sum z₀+z₁ and shard along T:
    GPU0: z₀ ∈ ℝ^(B × T/2 × d)  ← shard of summed output
    GPU1: z₁ ∈ ℝ^(B × T/2 × d)

    ──── Back to SP region ──────────────────────────────────────

    Dropout(z₀)   no comm                   Dropout(z₁)

    ════════════════════════════════════════════════════════════════
    Activation memory per GPU: B × T/2 × d  (vs B × T × d without SP)


### Memory Savings from Sequence Parallelism

Without SP (standard TP):
    Non-TP activations (LayerNorm input/output, Dropout):
        2 × B × T × d per layer × N layers  ← replicated on all TP ranks!

With SP:
    Same activations now sharded: 2 × B × T/TP × d per layer × N layers

**Activation memory reduction: TP× for the SP-eligible components.**

For LLaMA-2 70B (N=80 layers, d=8192, B=1, T=4096, TP=8):

    Without SP activation contribution:
        LayerNorm tensors ≈ 2 × 1 × 4096 × 8192 × 2 × 80 = 10.7 GB per GPU

    With SP:
        ≈ 10.7 GB / 8 = 1.3 GB per GPU   → 9.4 GB saved per GPU!

This is not the only activation memory, but it is the component that grows
most rapidly with sequence length T, making SP essential for very long contexts.


### Ring Attention: SP for Long-Context (> 128k tokens)

Standard attention requires all T² attention scores to fit in memory, which
is prohibitive for T > 32k tokens even with FlashAttention.

**Ring Attention** (Liu et al., 2023) extends SP to the attention computation
itself. Each GPU holds T/TP tokens. The KV pairs are passed around a ring of
TP GPUs, and each GPU attends to the KV pairs it currently holds:

    Round 0: GPU0 attends to KV₀ (its own tokens)
    Round 1: GPU0 receives KV₁ from GPU1, attends, passes KV₀ to GPU(TP-1)
    Round 2: GPU0 receives KV₂, attends...
    ...

After TP rounds, each GPU has attended to all tokens.

This enables contexts of 1M+ tokens by distributing both the computation
and the memory across TP GPUs. Used in:
    •   LLaMA-3 long-context models (128k context)
    •   Research models with 1M+ context (e.g., Gemini 1.5)

**Communication cost:** TP rounds × KV shard size per round
    = TP × (B × T/TP × d) = B × T × d  per layer
    (same as without SP — communication is not increased)


    **Diagram 2 — Ring Attention:**

    RING ATTENTION (TP=4, T=16 tokens per GPU)
    ════════════════════════════════════════════════════════════════

    Initial: each GPU holds its T/4 query tokens + T/4 KV tokens

    GPU0: Q₀,K₀,V₀    GPU1: Q₁,K₁,V₁    GPU2: Q₂,K₂,V₂    GPU3: Q₃,K₃,V₃

    Round 0: attend with own KV
    GPU0: O₀ += Attn(Q₀, K₀, V₀)  ← attend to first T/4 tokens

    Round 1: receive KV from left, send KV to right
    GPU0 receives K₁,V₁ from GPU1:
    GPU0: O₀ += Attn(Q₀, K₁, V₁)  ← attend to second T/4 tokens

    Round 2:
    GPU0 receives K₂,V₂ from GPU2:
    GPU0: O₀ += Attn(Q₀, K₂, V₂)  ← attend to third T/4 tokens

    Round 3:
    GPU0 receives K₃,V₃ from GPU3:
    GPU0: O₀ += Attn(Q₀, K₃, V₃)  ← attend to fourth T/4 tokens

    Each GPU now has the complete attention output for its T/4 tokens. ✓
    Memory: O(T/TP) per GPU instead of O(T) ✓


### Context Parallelism (CP) in Practice

NVIDIA Megatron-Core and recent PyTorch nightly builds implement sequence
parallelism under the name **Context Parallelism (CP)**. The implementation:

    1.  Split input batch along sequence dimension: each CP rank gets T/CP
    2.  Run LayerNorm, FFN (element-wise parts) on local shards
    3.  For attention: use Ring Attention or All-Gather-based attention
    4.  Reduce-Scatter the attention output back to sequence shards

**CP vs TP vs SP distinction in practice:**
    •   TP: splits weight matrices (d dimension)
    •   SP (Megatron definition): splits non-TP activations along T, uses
        All-Gather before TP region and Reduce-Scatter after
    •   CP (context parallelism): extends SP to include attention itself
        using Ring Attention

In practice, modern frameworks combine all three:
    SP handles LayerNorm/Dropout regions
    TP handles weight matrix computation
    CP handles attention with Ring Attention for very long contexts


### Collective Operations: All-Gather vs Reduce-Scatter

SP introduces two new collective operations to the TP workflow:

    **All-Gather** (SP before TP region):
        Each rank contributes its T/TP sequence shard.
        Every rank receives the full T-length sequence.
        Volume per rank: T/TP → T  (data received = (TP-1)/TP × T × d)

    **Reduce-Scatter** (SP after TP region):
        Each rank contributes its partial result over the full sequence.
        Each rank receives a different T/TP shard of the summed result.
        Volume per rank: T × d → T/TP × d  (same as All-Gather)

    **All-Reduce** (standard TP without SP):
        Each rank contributes its partial result.
        Every rank receives the full summed result.
        Volume per rank: T × d (same total data moved)

The total communication volume of SP (All-Gather + Reduce-Scatter) is
identical to standard TP (two All-Reduces). SP does not increase communication
— it just changes the output format from replicated to sharded.


### When to Use Sequence Parallelism

    Scenario                    Use SP?   Why
    ──────────────────────────────────────────────────────────────────
    Short context (T ≤ 2048)    Maybe     Small absolute saving
    Medium context (T = 4096)   Yes       Meaningful activation saving
    Long context (T ≥ 8192)     Strongly  Activation memory scales with T
    Very long (T ≥ 32768)       Required  Need Ring Attention too
    TP=1 (no tensor parallel)   No        SP only makes sense combined with TP
    TP ≥ 2, any context         Yes       Free memory saving, same comm
    ──────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
SP vs TP vs CP Comparison

| Property                  | TP (no SP)                  | TP + SP                      | TP + SP + CP (Ring Attn)     |
|---------------------------|-----------------------------|------------------------------ |------------------------------|
| Weight memory per GPU     | Model / TP                  | Model / TP                    | Model / TP                   |
| Non-TP activation / GPU   | B × T × d  (replicated)    | B × T/TP × d  (sharded)      | B × T/TP × d  (sharded)      |
| Attention activation / GPU| B × T × d_head             | B × T × d_head                | B × T/TP × d_head            |
| Max context length        | Limited by T² memory        | Limited by T² memory          | Scales with TP               |
| Communication per layer   | 2 All-Reduce               | All-Gather + Reduce-Scatter   | 2(AG+RS) + Ring sends        |
| Comm volume               | 2 × B × T × d              | 2 × B × T × d  (same!)       | 2 × B × T × d + Ring KV      |
| Enables M-token contexts? | No                          | No (attn still T²)            | Yes                          |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Sequence Parallelism: All-Gather and Reduce-Scatter": {
        "description": "Implement SP's All-Gather (before column-parallel) and Reduce-Scatter (after row-parallel) using threads, and verify the SP+TP pipeline produces correct results.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SEQUENCE PARALLELISM: ALL-GATHER + REDUCE-SCATTER
================================================================================

Implements the two core operations of Sequence Parallelism:
    1. All-Gather: reconstruct full sequence before column-parallel linear
    2. Reduce-Scatter: sum partial results and shard sequence after row-parallel

Also implements a SP+TP block that handles:
    SP region:  LayerNorm on T/TP shard
    All-Gather: reconstruct full T sequence
    TP region:  column-parallel → activation → row-parallel
    Reduce-Scatter: sum and re-shard to T/TP per GPU

Verifies mathematical equivalence to single-GPU computation.
================================================================================
"""

import threading
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Collective operations ─────────────────────────────────────────────────────

class CollectiveBarrier:
    """Thread-safe barrier supporting All-Gather and Reduce-Scatter."""

    def __init__(self, n: int):
        self.n       = n
        self.lock    = threading.Lock()
        self.barrier = threading.Barrier(n)
        self.store   = {}

    def all_gather(self, key: str, rank: int,
                   shard: torch.Tensor) -> torch.Tensor:
        """
        Each rank contributes a shard along dim=1 (sequence).
        All ranks receive the full concatenation.
        """
        with self.lock:
            if key not in self.store:
                self.store[key] = [None] * self.n
            self.store[key][rank] = shard.detach().clone()

        self.barrier.wait()
        # Concatenate along sequence dimension (dim=1: B × T_local × d)
        full = torch.cat(self.store[key], dim=1)

        with self.lock:
            if key in self.store:
                del self.store[key]
        return full

    def reduce_scatter(self, key: str, rank: int,
                       partial: torch.Tensor) -> torch.Tensor:
        """
        Each rank contributes a partial sum over the full sequence.
        Sum across ranks, then each rank gets shard rank * T/n : (rank+1) * T/n.
        """
        with self.lock:
            if key not in self.store:
                self.store[key] = [None] * self.n
            self.store[key][rank] = partial.detach().clone()

        self.barrier.wait()
        # Sum all partial results (this is the "reduce" step)
        summed = torch.stack(self.store[key]).sum(dim=0)  # (B, T, d)
        # Scatter: each rank gets its chunk along sequence dim
        T      = summed.shape[1]
        T_local = T // self.n
        shard  = summed[:, rank * T_local : (rank + 1) * T_local, :]

        with self.lock:
            if key in self.store:
                del self.store[key]
        return shard


# ── SP + TP forward pass ──────────────────────────────────────────────────────

def sp_tp_rank(rank: int, tp: int,
               x_shard: torch.Tensor,      # (B, T/tp, d) — SP input shard
               W1: torch.Tensor,            # (d, d_ff) full weight
               W2: torch.Tensor,            # (d_ff, d) full weight
               barrier: CollectiveBarrier,
               results: dict, step: int):
    """
    One SP+TP rank forward pass:
        1. LayerNorm on x_shard (SP region — no comm)
        2. All-Gather to get full sequence for TP region
        3. Column-parallel FFN W1 on full sequence
        4. Row-parallel FFN W2 on full sequence → partial result
        5. Reduce-Scatter to sum partial results and re-shard sequence
    """
    d, d_ff = W1.shape
    d_ff_local = d_ff // tp

    # ── Step 1: LayerNorm in SP region (no comm) ────────────────────────────
    ln = nn.LayerNorm(d, elementwise_affine=False)
    x_normed = ln(x_shard)   # (B, T/tp, d)

    # ── Step 2: All-Gather to reconstruct full sequence ──────────────────────
    x_full = barrier.all_gather(f"s{step}_ag", rank, x_normed)  # (B, T, d)

    # ── Step 3: Column-parallel W1 (each rank owns d_ff/tp output columns) ──
    W1_local = W1[:, rank * d_ff_local : (rank + 1) * d_ff_local]
    y_local  = F.gelu(x_full @ W1_local)    # (B, T, d_ff/tp)

    # ── Step 4: Row-parallel W2 (each rank owns d_ff/tp input rows) ─────────
    W2_local = W2[rank * d_ff_local : (rank + 1) * d_ff_local, :]
    z_full   = y_local @ W2_local           # (B, T, d) — partial sum

    # ── Step 5: Reduce-Scatter to sum and re-shard ───────────────────────────
    z_shard = barrier.reduce_scatter(f"s{step}_rs", rank, z_full)  # (B, T/tp, d)

    results[rank] = z_shard


# ── Reference: single GPU ────────────────────────────────────────────────────

def reference_forward(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor):
    """Full single-GPU FFN with LayerNorm."""
    ln = nn.LayerNorm(x.shape[-1], elementwise_affine=False)
    return F.gelu(ln(x) @ W1) @ W2


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    TP   = 4
    B, T = 2, 16     # T must be divisible by TP
    D    = 64
    D_FF = 128

    assert T % TP == 0, "Sequence length must be divisible by TP"
    T_local = T // TP

    W1 = torch.randn(D, D_FF)
    W2 = torch.randn(D_FF, D)
    x  = torch.randn(B, T, D)

    print("=" * 62)
    print(f"  SEQUENCE PARALLELISM + TENSOR PARALLELISM")
    print(f"  TP={TP}, d={D}, d_ff={D_FF}, B={B}, T={T}")
    print(f"  Each GPU holds T/TP = {T_local} sequence tokens in SP regions")
    print("=" * 62)
    print()

    # Reference
    ref_out = reference_forward(x, W1, W2)
    print(f"  Reference output shape: {ref_out.shape}")
    print(f"  Reference norm:         {ref_out.norm():.4f}")
    print()

    # Shard x along sequence dimension for SP
    x_shards = [x[:, r * T_local : (r + 1) * T_local, :] for r in range(TP)]

    # SP + TP simulation
    barrier = CollectiveBarrier(TP)
    results = {}
    threads = [
        threading.Thread(
            target=sp_tp_rank,
            args=(r, TP, x_shards[r], W1, W2, barrier, results, 0)
        )
        for r in range(TP)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    # Reconstruct full output from shards for comparison
    full_out = torch.cat([results[r] for r in range(TP)], dim=1)

    diff = (full_out - ref_out).abs().max().item()
    print(f"  Reconstructed SP+TP output shape: {full_out.shape}")
    print(f"  Max diff from reference:          {diff:.2e}  "
          f"{'✓ correct' if diff < 1e-4 else '✗ error'}")

    print()
    print("  Data flow summary:")
    print(f"  SP region:  each GPU holds B×{T_local}×{D} = "
          f"{B*T_local*D*2/1024:.1f} KB  (bf16)")
    print(f"  TP region:  each GPU uses full B×{T}×{D} = "
          f"{B*T*D*2/1024:.1f} KB  (bf16)")
    print(f"  After RS:   each GPU holds B×{T_local}×{D} = "
          f"{B*T_local*D*2/1024:.1f} KB  (bf16)")
    print()
    print(f"  Memory saving vs standard TP: {TP}× in SP regions")
    print(f"  Communication: All-Gather + Reduce-Scatter = same as 2× All-Reduce")

    # Verify All-Gather + Reduce-Scatter = All-Reduce
    print()
    print("=" * 62)
    print("  VERIFY: (All-Gather → TP → Reduce-Scatter) = All-Reduce")
    print("=" * 62)
    print()
    print("  Standard TP uses All-Reduce after row-parallel linear.")
    print("  SP+TP uses Reduce-Scatter instead, producing sharded output.")
    print("  Concatenating the shards gives the same result as All-Reduce. ✓")
    print()
    # Simulate what All-Reduce would give
    z_partials = []
    for r in range(TP):
        W1_l = W1[:, r * D_FF//TP : (r+1) * D_FF//TP]
        W2_l = W2[r * D_FF//TP : (r+1) * D_FF//TP, :]
        ln   = nn.LayerNorm(D, elementwise_affine=False)
        y    = F.gelu(ln(x) @ W1_l) @ W2_l
        z_partials.append(y)

    z_allreduce = torch.stack(z_partials).sum(dim=0)   # All-Reduce result
    diff2 = (z_allreduce - ref_out).abs().max().item()
    diff3 = (full_out - z_allreduce).abs().max().item()

    print(f"  All-Reduce result matches reference:  {diff2:.2e}")
    print(f"  SP+TP reconstruction matches AR:      {diff3:.2e}")
    print(f"  Both correct: {'✓' if diff2 < 1e-4 and diff3 < 1e-4 else '✗'}")
''',
    },

    "Ring Attention Simulation": {
        "description": "Simulate Ring Attention for long-context sequence parallelism — each GPU attends to rotating KV shards, accumulating the full attention output.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
RING ATTENTION SIMULATION
================================================================================

Simulates Ring Attention:
    1. Each GPU holds T/TP query tokens and T/TP KV tokens
    2. In each round, KV tensors rotate to the next GPU in the ring
    3. Each GPU accumulates partial attention outputs using online softmax
    4. After TP rounds, each GPU has the complete attention for its tokens

Uses online softmax (FlashAttention-style) for numerical stability.
Verifies correctness against standard full-sequence attention.
================================================================================
"""

import threading
import math
import torch
import torch.nn.functional as F


def attention_chunk(Q: torch.Tensor, K: torch.Tensor,
                    V: torch.Tensor, causal_mask_fn=None,
                    q_offset: int = 0, kv_offset: int = 0) -> tuple:
    """
    Compute attention for (Q, K, V) chunks with optional causal masking.
    Returns (output, log_sum_exp) for online softmax accumulation.

    Q shape: (B, h, T_q, d_head)
    K shape: (B, h, T_k, d_head)
    V shape: (B, h, T_k, d_head)
    """
    scale  = math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) / scale   # (B, h, T_q, T_k)

    if causal_mask_fn is not None:
        mask = causal_mask_fn(q_offset, kv_offset, Q.shape[-2], K.shape[-2])
        scores = scores.masked_fill(mask, float("-inf"))

    # Compute softmax statistics for online accumulation
    m      = scores.max(dim=-1, keepdim=True).values     # max for stability
    exp_s  = torch.exp(scores - m)                        # (B, h, T_q, T_k)
    l      = exp_s.sum(dim=-1, keepdim=True)              # (B, h, T_q, 1)
    o      = exp_s @ V                                    # (B, h, T_q, d_head)

    return o, m.squeeze(-1), l.squeeze(-1)   # outputs and statistics


def merge_attention_chunks(o1, m1, l1, o2, m2, l2):
    """
    Merge two partial attention results using online softmax update.
    Online softmax: update running (output, max, log_sum_exp).
    """
    # New global max
    m_new = torch.maximum(m1, m2)

    # Rescale each contribution
    alpha1 = torch.exp(m1 - m_new).unsqueeze(-1)
    alpha2 = torch.exp(m2 - m_new).unsqueeze(-1)

    l_new = alpha1.squeeze(-1) * l1 + alpha2.squeeze(-1) * l2
    o_new = (alpha1 * l1.unsqueeze(-1) * o1 + alpha2 * l2.unsqueeze(-1) * o2)
    o_new = o_new / l_new.unsqueeze(-1).clamp(min=1e-9)

    return o_new, m_new, l_new


# ── Causal mask for Ring Attention ────────────────────────────────────────────

def causal_ring_mask(q_offset: int, kv_offset: int,
                     T_q: int, T_k: int) -> torch.Tensor:
    """
    Causal mask for a (T_q, T_k) chunk where:
        Q tokens are at positions [q_offset, q_offset + T_q)
        K tokens are at positions [kv_offset, kv_offset + T_k)

    Mask[i, j] = True (mask out) if q_pos < kv_pos (future token).
    q_pos = q_offset + i,  kv_pos = kv_offset + j
    """
    q_pos  = torch.arange(q_offset, q_offset + T_q).unsqueeze(1)
    kv_pos = torch.arange(kv_offset, kv_offset + T_k).unsqueeze(0)
    return q_pos < kv_pos   # True = mask out (future tokens)


def ring_attention_rank(rank: int, tp: int,
                         Q_shard: torch.Tensor,
                         K_shard: torch.Tensor,
                         V_shard: torch.Tensor,
                         barrier, results: dict, step: int,
                         causal: bool = True):
    """
    One GPU's execution of Ring Attention.

    Q_shard: (B, h, T_local, d_head) — this GPU's query shard
    K/V shards rotate around the ring for TP rounds.
    """
    B, h, T_local, d_head = Q_shard.shape
    T_total = T_local * tp

    # Initialise accumulation buffers
    o_acc = torch.zeros_like(Q_shard)
    m_acc = torch.full((B, h, T_local), float("-inf"))
    l_acc = torch.zeros(B, h, T_local)

    # Local KV starts at this rank's shard
    K_curr = K_shard.clone()
    V_curr = V_shard.clone()
    kv_rank = rank   # which rank's KV we currently hold

    q_offset  = rank * T_local   # position offset of our Q tokens in global seq

    for round_idx in range(tp):
        kv_offset = kv_rank * T_local

        # Compute attention for this (Q_shard, K_curr, V_curr) pair
        mask_fn = (lambda qo, kvo, tq, tk: causal_ring_mask(qo, kvo, tq, tk)
                    .unsqueeze(0).unsqueeze(0)) if causal else None

        o_chunk, m_chunk, l_chunk = attention_chunk(
            Q_shard, K_curr, V_curr,
            causal_mask_fn=mask_fn,
            q_offset=q_offset, kv_offset=kv_offset,
        )

        # Merge with accumulated result
        if round_idx == 0:
            o_acc, m_acc, l_acc = o_chunk, m_chunk, l_chunk
        else:
            o_acc, m_acc, l_acc = merge_attention_chunks(
                o_acc, m_acc, l_acc, o_chunk, m_chunk, l_chunk
            )

        # Pass KV to next rank in ring (blocking barrier for sync)
        with barrier.lock:
            if f"s{step}_kv_r{round_idx}_in{rank}" not in barrier.store:
                barrier.store[f"s{step}_kv_r{round_idx}_in{rank}"] = (K_curr.clone(), V_curr.clone())

        barrier.barrier.wait()

        # Receive KV from previous rank in ring
        prev_rank    = (rank - 1) % tp
        key_recv     = f"s{step}_kv_r{round_idx}_in{prev_rank}"
        # Wait briefly for previous rank to write
        import time as _time
        while key_recv not in barrier.store:
            _time.sleep(0.0001)

        K_curr, V_curr = barrier.store[key_recv]
        kv_rank = (kv_rank - 1) % tp

        barrier.barrier.wait()

        # Cleanup
        with barrier.lock:
            barrier.store.pop(f"s{step}_kv_r{round_idx}_in{rank}", None)

    results[rank] = o_acc


class RingBarrier:
    def __init__(self, n):
        self.n       = n
        self.lock    = threading.Lock()
        self.barrier = threading.Barrier(n)
        self.store   = {}


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    TP        = 4
    B         = 1
    H         = 4         # attention heads (divisible by TP)
    T_TOTAL   = 16        # total sequence length
    D_HEAD    = 8
    T_LOCAL   = T_TOTAL // TP

    assert T_TOTAL % TP == 0

    print("=" * 62)
    print(f"  RING ATTENTION SIMULATION")
    print(f"  TP={TP}, T={T_TOTAL}, T_local={T_LOCAL}, h={H}, d_head={D_HEAD}")
    print("=" * 62)
    print()

    # Create full Q, K, V for reference
    Q_full = torch.randn(B, H, T_TOTAL, D_HEAD)
    K_full = torch.randn(B, H, T_TOTAL, D_HEAD)
    V_full = torch.randn(B, H, T_TOTAL, D_HEAD)

    # Reference: standard causal attention on full sequence
    scale     = math.sqrt(D_HEAD)
    scores    = (Q_full @ K_full.transpose(-2, -1)) / scale
    causal_m  = torch.triu(torch.ones(T_TOTAL, T_TOTAL, dtype=torch.bool), diagonal=1)
    scores    = scores.masked_fill(causal_m.unsqueeze(0).unsqueeze(0), float("-inf"))
    weights   = F.softmax(scores, dim=-1)
    ref_out   = weights @ V_full    # (B, H, T_TOTAL, D_HEAD)

    print(f"  Reference output shape: {ref_out.shape}")
    print(f"  Reference norm:         {ref_out.norm():.4f}")
    print()

    # Shard Q, K, V along sequence dimension for Ring Attention
    Q_shards = [Q_full[:, :, r*T_LOCAL:(r+1)*T_LOCAL, :] for r in range(TP)]
    K_shards = [K_full[:, :, r*T_LOCAL:(r+1)*T_LOCAL, :] for r in range(TP)]
    V_shards = [V_full[:, :, r*T_LOCAL:(r+1)*T_LOCAL, :] for r in range(TP)]

    # Ring Attention simulation
    ring_barrier = RingBarrier(TP)
    ring_results = {}
    threads = [
        threading.Thread(
            target=ring_attention_rank,
            args=(r, TP, Q_shards[r], K_shards[r], V_shards[r],
                  ring_barrier, ring_results, 0, True)
        )
        for r in range(TP)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    # Reconstruct full output from shards
    ring_out = torch.cat([ring_results[r] for r in range(TP)], dim=2)

    diff = (ring_out - ref_out).abs().max().item()
    print(f"  Ring Attention output shape: {ring_out.shape}")
    print(f"  Max diff from reference:     {diff:.2e}  "
          f"{'✓ correct' if diff < 1e-3 else '✗ error'}")

    print()
    print("  Ring Attention properties:")
    print(f"  • Memory per GPU:  B×{T_LOCAL}×(K+V) tensors = "
          f"{2*B*T_LOCAL*H*D_HEAD*4/1024:.1f} KB  (fp32)")
    print(f"  • vs standard:     B×{T_TOTAL}×(K+V) tensors = "
          f"{2*B*T_TOTAL*H*D_HEAD*4/1024:.1f} KB  ({TP}× more)")
    print(f"  • Rounds:          {TP} (TP degree)")
    print(f"  • Communication:   {TP-1} KV tensor exchanges per layer")
    print()
    print("  Scaling advantage:")
    for t in [1024, 4096, 16384, 65536, 131072]:
        mem_std  = 2 * B * t * H * D_HEAD * 4 / 1e6    # MB
        mem_ring = mem_std / TP
        print(f"    T={t:>7}: standard {mem_std:>8.1f} MB / GPU  "
              f"Ring Attn {mem_ring:>8.1f} MB / GPU ({TP}× saving)")
''',
    },

    "SP Memory Savings Analysis": {
        "description": "Quantify the activation memory savings from sequence parallelism at different sequence lengths and TP degrees, showing when SP becomes essential.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SEQUENCE PARALLELISM: MEMORY SAVINGS ANALYSIS
================================================================================

Computes activation memory breakdown for:
    1. Standard TP (no SP)
    2. TP + SP (non-attention ops sharded)
    3. TP + SP + CP (ring attention, full sharding)

Shows when SP becomes essential (large T) and the combined savings.
================================================================================
"""

import math


def activation_memory_tp_only(d: int, n_layers: int, n_heads: int,
                               seq_len: int, batch: int,
                               dtype_bytes: int = 2) -> dict:
    """
    Activation memory per GPU with TP only (no SP).
    Every GPU holds the full (B × T × d) activations.
    """
    B, T, N = batch, seq_len, n_layers

    # Replicated tensors (same on all TP GPUs)
    layernorm_per_layer = 2 * B * T * d * dtype_bytes       # 2 LNs per block
    dropout_per_layer   = B * T * d * dtype_bytes            # 1 dropout

    # Attention activations (T² grows quadratically)
    d_head = d // n_heads
    attn_scores  = B * n_heads * T * T * dtype_bytes         # Q·Kᵀ (T×T!)
    attn_out     = B * T * d * dtype_bytes                   # post-attn

    # FFN activations
    ffn_hidden = B * T * d * 4 * dtype_bytes                 # d_ff ≈ 4d

    per_layer = (layernorm_per_layer + dropout_per_layer +
                 attn_scores + attn_out + ffn_hidden)

    return {
        "layernorm_mb":   N * layernorm_per_layer / 1e6,
        "dropout_mb":     N * dropout_per_layer / 1e6,
        "attn_scores_mb": N * attn_scores / 1e6,
        "attn_out_mb":    N * attn_out / 1e6,
        "ffn_hidden_mb":  N * ffn_hidden / 1e6,
        "total_mb":       N * per_layer / 1e6,
    }


def activation_memory_sp_tp(d: int, n_layers: int, n_heads: int,
                              seq_len: int, batch: int,
                              tp: int, dtype_bytes: int = 2) -> dict:
    """
    Activation memory per GPU with SP+TP.
    Non-TP ops (LN, Dropout) are sharded → divided by TP.
    Attention still needs full T (unless Ring Attention).
    """
    B, T, N = batch, seq_len, n_layers
    T_local = T // tp   # sharded sequence length per GPU

    # SP-sharded tensors (T/tp per GPU)
    layernorm_per_layer = 2 * B * T_local * d * dtype_bytes
    dropout_per_layer   = B * T_local * d * dtype_bytes

    # TP region: attention still needs full sequence context
    # (each GPU attends over full T but only for its h/tp heads)
    d_head = d // n_heads
    h_local = n_heads // tp
    attn_scores  = B * h_local * T * T * dtype_bytes   # per-rank heads, full T
    attn_out     = B * T_local * d * dtype_bytes        # re-sharded after RS

    # FFN: column-parallel intermediate (d_ff/tp per GPU)
    ffn_hidden = B * T * d * 4 // tp * dtype_bytes     # TP-sharded

    per_layer = (layernorm_per_layer + dropout_per_layer +
                 attn_scores + attn_out + ffn_hidden)

    return {
        "layernorm_mb":   N * layernorm_per_layer / 1e6,
        "dropout_mb":     N * dropout_per_layer / 1e6,
        "attn_scores_mb": N * attn_scores / 1e6,
        "attn_out_mb":    N * attn_out / 1e6,
        "ffn_hidden_mb":  N * ffn_hidden / 1e6,
        "total_mb":       N * per_layer / 1e6,
    }


def activation_memory_ring(d: int, n_layers: int, n_heads: int,
                             seq_len: int, batch: int,
                             tp: int, dtype_bytes: int = 2) -> dict:
    """
    Activation memory per GPU with SP+TP+Ring Attention.
    All activations are T/tp per GPU (full sharding).
    """
    B, T, N = batch, seq_len, n_layers
    T_local = T // tp

    d_head  = d // n_heads
    h_local = n_heads // tp

    layernorm_per_layer = 2 * B * T_local * d * dtype_bytes
    dropout_per_layer   = B * T_local * d * dtype_bytes

    # Ring attention: only T_local KV at a time (rotated)
    attn_scores  = B * h_local * T_local * T_local * dtype_bytes  # local chunk only
    attn_out     = B * T_local * d * dtype_bytes

    ffn_hidden = B * T_local * d * 4 // tp * dtype_bytes

    per_layer = (layernorm_per_layer + dropout_per_layer +
                 attn_scores + attn_out + ffn_hidden)

    return {
        "layernorm_mb":   N * layernorm_per_layer / 1e6,
        "dropout_mb":     N * dropout_per_layer / 1e6,
        "attn_scores_mb": N * attn_scores / 1e6,
        "attn_out_mb":    N * attn_out / 1e6,
        "ffn_hidden_mb":  N * ffn_hidden / 1e6,
        "total_mb":       N * per_layer / 1e6,
    }


def fmt_mb(mb: float) -> str:
    if mb >= 1000: return f"{mb/1024:.1f} GB"
    return f"{mb:.0f} MB"


if __name__ == "__main__":
    # LLaMA-2 70B dimensions
    D       = 8192
    N_HEADS = 64
    N_LAY   = 80
    B       = 1
    TP      = 8

    print("=" * 72)
    print(f"  SEQUENCE PARALLELISM MEMORY SAVINGS — LLaMA-2 70B")
    print(f"  d={D}, n_heads={N_HEADS}, N={N_LAY}, B={B}, TP={TP}")
    print("=" * 72)
    print()
    print(f"  {'Seq len':>8}  {'TP only':>10}  {'TP+SP':>10}  {'TP+SP+Ring':>12}  "
          f"{'SP saves':>10}  {'Ring saves':>12}")
    print(f"  {'':─>8}  {'':─>10}  {'':─>10}  {'':─>12}  {'':─>10}  {'':─>12}")

    for T in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        m_tp    = activation_memory_tp_only(D, N_LAY, N_HEADS, T, B)
        m_sp    = activation_memory_sp_tp(D, N_LAY, N_HEADS, T, B, TP)
        m_ring  = activation_memory_ring(D, N_LAY, N_HEADS, T, B, TP)

        sp_save   = m_tp["total_mb"] / m_sp["total_mb"]
        ring_save = m_tp["total_mb"] / m_ring["total_mb"]

        print(f"  {T:>8,}  {fmt_mb(m_tp['total_mb']):>10}  "
              f"{fmt_mb(m_sp['total_mb']):>10}  "
              f"{fmt_mb(m_ring['total_mb']):>12}  "
              f"{sp_save:>9.1f}×  {ring_save:>11.1f}×")

    print()
    print("  Key observations:")
    print("  1. TP+SP already saves TP× for LN/Dropout components")
    print("  2. Standard attention (T²) dominates at T > 8k even with SP")
    print("  3. Ring Attention eliminates T² problem → saves TP² in attention")
    print("  4. At T=65536: TP+SP+Ring uses ~100× less memory than TP only!")
    print()

    # Breakdown for T=8192
    T = 8192
    print("=" * 72)
    print(f"  COMPONENT BREAKDOWN AT T={T}")
    print("=" * 72)
    print()
    m_tp   = activation_memory_tp_only(D, N_LAY, N_HEADS, T, B)
    m_sp   = activation_memory_sp_tp(D, N_LAY, N_HEADS, T, B, TP)
    m_ring = activation_memory_ring(D, N_LAY, N_HEADS, T, B, TP)

    components = ["layernorm_mb", "dropout_mb", "attn_scores_mb",
                  "attn_out_mb", "ffn_hidden_mb"]
    labels     = ["LayerNorm", "Dropout", "Attn scores (T²)", "Attn output", "FFN hidden"]

    print(f"  {'Component':<22} {'TP only':>10}  {'TP+SP':>10}  {'TP+SP+Ring':>12}")
    print(f"  {'':─<22} {'':─>10}  {'':─>10}  {'':─>12}")
    for comp, label in zip(components, labels):
        print(f"  {label:<22} {fmt_mb(m_tp[comp]):>10}  "
              f"{fmt_mb(m_sp[comp]):>10}  "
              f"{fmt_mb(m_ring[comp]):>12}")
    print(f"  {'TOTAL':<22} {fmt_mb(m_tp['total_mb']):>10}  "
          f"{fmt_mb(m_sp['total_mb']):>10}  "
          f"{fmt_mb(m_ring['total_mb']):>12}")
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
    #     from llm_training.visuals.sequence_parallelism import (
    #         SP_VISUAL_HTML,
    #         SP_VISUAL_HEIGHT,
    #     )
    #     visual_html   = SP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SP_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[21_sequence_parallelism.py] Could not load visual: {e}",
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