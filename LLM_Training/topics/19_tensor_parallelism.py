"""
Tensor Parallelism
==================

Tensor parallelism splits individual weight matrices across multiple GPUs
rather than replicating them. Where data parallelism replicates the entire
model and splits the data, tensor parallelism splits the model itself —
each GPU holds a horizontal slice of every weight matrix and contributes
to every forward pass together. This allows training models whose individual
weight matrices exceed the memory of a single GPU, and reduces per-GPU
peak memory without the communication overhead of recomputing full weights.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Tensor Parallelism"
DISPLAY_NAME = "19 · Tensor Parallelism"
ICON         = "🧩"
SUBTITLE     = "Splitting Weight Matrices Across GPUs"


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

### Why Tensor Parallelism?

Data parallelism (DDP/FSDP) requires that the model fit on a single GPU,
at least transiently during the forward pass. A 70B model in bf16 occupies
140 GB of weights alone — already exceeding two A100 80GB GPUs. ZeRO Stage 3
shards the weights but must all-gather them before each layer computation,
adding communication overhead.

**Tensor parallelism** takes a different approach: the weight matrix itself
is split column-wise or row-wise across GPUs. Each GPU holds a permanent
shard of every weight, and the forward/backward passes are computed
collaboratively using small, targeted collective operations (not full all-gathers
of the whole weight).

For a weight matrix W ∈ ℝ^(d_in × d_out) split across T GPUs:
    •   Each GPU holds W_t ∈ ℝ^(d_in × d_out/T)   (column partition)
    •   Or W_t ∈ ℝ^(d_in/T × d_out)               (row partition)
    •   The forward pass for each GPU is a partial matmul: y_t = x · W_t
    •   Aggregation: all-reduce or all-gather to get the full output

Used in: Megatron-LM, LLaMA-2 70B training, PaLM, GPT-4 (estimated).


### Megatron-LM Tensor Parallelism (Column + Row Split)

The canonical tensor parallelism design from Megatron-LM (Shoeybi et al.,
2019) applies complementary splits to the two weight matrices in each FFN:

**Column-parallel linear (first FFN layer):**
    W₁ ∈ ℝ^(d × d_ff) is split column-wise: W₁ = [W₁⁰ | W₁¹ | … | W₁ᵀ⁻¹]
    Each GPU t computes: y_t = x · W₁ᵗ    (shape: B × T × d_ff/T)
    No communication needed — each GPU uses the same full input x.

**Row-parallel linear (second FFN layer):**
    W₂ ∈ ℝ^(d_ff × d) is split row-wise: W₂ = [W₂⁰; W₂¹; …; W₂ᵀ⁻¹]
    Each GPU t computes: z_t = y_t · W₂ᵗ   (shape: B × T × d)
    All-reduce: z = Σ z_t  (sum partial results to get final output)

The key insight: no communication is needed between the column-parallel
and row-parallel layers because W₁ᵗ produces exactly the y_t needed by W₂ᵗ.
The all-reduce happens only once at the end of the two-matrix computation.


    **Diagram 1 — Column-Parallel and Row-Parallel Linear Layers:**

    MEGATRON-LM TENSOR PARALLELISM (T=2 GPUs)
    ════════════════════════════════════════════════════════════════

    Input x (full, on both GPUs via broadcast or replicated):

    ──── Column-Parallel (W₁ split by columns) ────────────────────

    GPU 0:  x · W₁⁰  →  y₀  ∈ ℝ^(d_ff/2)
    GPU 1:  x · W₁¹  →  y₁  ∈ ℝ^(d_ff/2)

    (No communication — each GPU can compute independently)

    Apply activation: y₀ = act(y₀),  y₁ = act(y₁)

    ──── Row-Parallel (W₂ split by rows) ──────────────────────────

    GPU 0:  y₀ · W₂⁰  →  z₀  ∈ ℝ^d
    GPU 1:  y₁ · W₂¹  →  z₁  ∈ ℝ^d

    All-Reduce:  z = z₀ + z₁   ← each GPU gets the full d-dim output

    ────────────────────────────────────────────────────────────────
    Result:  z = x · W₁ · act(·) · W₂  ← mathematically identical
                                           to single-GPU computation ✓
    Communication: 1 all-reduce per FFN block  (after row-parallel)


### Tensor Parallelism for Attention

Multi-head attention has a natural tensor-parallel decomposition: split
heads across GPUs. With h attention heads and T-way tensor parallelism,
each GPU holds h/T heads.

**Q, K, V projections (column-parallel):**
    W_Q ∈ ℝ^(d × d)  →  each GPU holds W_Q_t ∈ ℝ^(d × d/T)
    Each GPU computes: Q_t = x · W_Q_t   (its h/T query heads)
    Similarly for K_t and V_t

**Attention computation:**
    Each GPU computes attention independently for its h/T heads:
    head_t = Attention(Q_t, K_t, V_t)

**Output projection (row-parallel):**
    W_O ∈ ℝ^(d × d)  →  each GPU holds W_O_t ∈ ℝ^(d/T × d)
    GPU t computes: out_t = head_t · W_O_t    ∈ ℝ^d
    All-Reduce: out = Σ out_t

Exactly one all-reduce per Transformer block (same pattern as FFN).


    **Diagram 2 — Tensor-Parallel Attention:**

    TENSOR-PARALLEL MULTI-HEAD ATTENTION (T=2, h=4 total heads)
    ════════════════════════════════════════════════════════════════

    Input x (same on both GPUs)

    GPU 0 owns heads 0,1:              GPU 1 owns heads 2,3:
    Q₀ = x · W_Q[0:d/2]               Q₁ = x · W_Q[d/2:d]
    K₀ = x · W_K[0:d/2]               K₁ = x · W_K[d/2:d]
    V₀ = x · W_V[0:d/2]               V₁ = x · W_V[d/2:d]

    head₀ = Attn(Q₀,K₀,V₀)           head₁ = Attn(Q₁,K₁,V₁)

    out₀ = head₀ · W_O[0:d/2, :]      out₁ = head₁ · W_O[d/2:d, :]

    ←────── All-Reduce: out = out₀ + out₁ ──────→

    Full output out ∈ ℝ^d available on both GPUs ✓


### Communication Pattern and Cost

Each Transformer block (attention + FFN) requires exactly **2 all-reduces**
per block in the forward pass:
    •   1 all-reduce after the attention output projection
    •   1 all-reduce after the FFN second linear layer

During the backward pass, the communication pattern is mirrored:
    •   Gradient of the all-reduce is an all-gather (or another all-reduce)
    •   Total: 4 collectives per block (2 forward + 2 backward)

For a model with N blocks and tensor-parallel degree T:

    Communication volume per step ≈ 4 × N × (B × seq × d) × (T-1)/T × 2 bytes

For LLaMA-2 70B (N=80, d=8192, B=1, T=4096, T_TP=4):
    ≈ 4 × 80 × 1 × 4096 × 8192 × 3/4 × 2 ≈ 16 GB per forward+backward

Compare to DDP all-reduce:
    ≈ 2 × 70B × 2 bytes ≈ 280 GB per step

**Tensor parallelism produces far less communication than DDP for large models**,
which is why it is preferred for models that don't fit on a single GPU.


### Synchronous vs Asynchronous Communication

Tensor parallelism communication is **synchronous** — the all-reduce must
complete before the next layer can begin. There is no overlap with the layer
compute (unlike DDP, where bucket-level all-reduce overlaps with backward).

This is why tensor parallelism works well only when communication is fast:
    •   Within a single NVLink node: all-reduce for 8 GB ≈ 13 ms on NVLink 3.0
    •   Across nodes via InfiniBand: 8 GB ≈ 320 ms → completely dominates

**The golden rule of tensor parallelism:** Only apply it within a single
NVLink domain (typically 8 GPUs). Never apply tensor parallelism across
InfiniBand — use pipeline parallelism for cross-node distribution.


### Sequence Parallelism: Complementary to TP

When tensor parallelism is applied, the model dimension is split but
the sequence dimension remains full on each GPU. However, certain operations
(LayerNorm, Dropout) still process the full sequence redundantly.

**Sequence Parallelism** (Korthikanti et al., 2022) extends tensor
parallelism by also partitioning the sequence dimension:

    •   Between all-reduces in TP, each GPU holds only seq/T of the sequence
    •   This reduces activation memory from B×T×d to B×(T/TP)×d per GPU
    •   LayerNorm is applied to the local sequence shard (no communication)

Combined with TP, sequence parallelism reduces activation memory by TP×,
enabling much larger batch sizes for the same GPU memory.
The collectives become:
    •   All-Gather before column-parallel linear (reconstruct full sequence)
    •   Reduce-Scatter after row-parallel linear (re-shard the sequence)


    **Diagram 3 — Sequence Parallelism + Tensor Parallelism:**

    SEQUENCE PARALLELISM COMBINED WITH TENSOR PARALLELISM
    ════════════════════════════════════════════════════════════════

    Without SP:
    GPU0, GPU1: each holds x ∈ ℝ^(B×T×d)    ← 2× redundant storage!
    LayerNorm runs on full T on both GPUs     ← redundant compute!

    With SP:
    GPU0: x₀ ∈ ℝ^(B×T/2×d)   GPU1: x₁ ∈ ℝ^(B×T/2×d)
    LayerNorm on local shard → no communication needed
          │                          │
    All-Gather x₀,x₁  ─────────────→ x full on both GPUs
          │                          │
    Column-parallel matmul           Column-parallel matmul
          │                          │
    Row-parallel matmul              Row-parallel matmul
          │                          │
    Reduce-Scatter ─────────────────→ x₀', x₁' (sharded again)

    Activation memory: halved per GPU ✓
    Communication: All-Gather + Reduce-Scatter (same volume as 2 all-reduces)


### Tensor Parallelism Degree: How to Choose

    TP degree    When appropriate
    ─────────────────────────────────────────────────────────────────
    TP=1         Model fits on one GPU; use DDP or ZeRO instead
    TP=2         Model marginally too large for 1 GPU; modest memory saving
    TP=4         Standard for 30B–70B models on 8-GPU nodes
    TP=8         Maximum within one NVLink domain (8 GPUs)
    TP>8         NEVER — requires cross-node communication; prohibitively slow
    ─────────────────────────────────────────────────────────────────

**Real-world configurations:**
    LLaMA-2 70B:     TP=4  (or TP=8 for faster training, lower batch)
    GPT-3 175B:      TP=8  (Megatron-LM reference config)
    PaLM 540B:       TP=8  per pod, multiple pods via DP
    Megatron 1T:     TP=8 per node, combined with PP across nodes


### Constraints on TP Degree

    1.  d_model must be divisible by TP
    2.  n_heads must be divisible by TP (for attention head splitting)
    3.  n_kv_heads (for GQA) must be divisible by TP
    4.  d_ff must be divisible by TP
    5.  TP must divide the number of GPUs on each node evenly

Typical model dimensions (d=4096, h=32, h_kv=8) support TP ∈ {1,2,4,8}.
TP=8 requires h_kv ≥ 8 (GQA must have at least 8 KV heads).
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Tensor Parallelism vs Data Parallelism vs ZeRO

| Property                    | Data Parallel (DDP)    | ZeRO Stage 3          | Tensor Parallel (TP)         |
|-----------------------------|------------------------|-----------------------|------------------------------|
| Memory per GPU (weights)    | Full model             | Model / N             | Model / TP                   |
| Communication per step      | 2 × grad_size          | 2 × param_size        | 4 × N × B×T×d (small)       |
| Communication frequency     | Once per step          | Per-layer (all-gather)| Per-layer (all-reduce)       |
| Works across nodes?         | Yes                    | Yes (slower)          | Only within NVLink domain     |
| Overlap comm/compute?       | Yes (DDP buckets)      | Limited               | No (synchronous)             |
| Max efficient TP degree     | N/A                    | N/A                   | 8 (NVLink nodes)             |
| Best for                    | Model fits on 1 GPU    | Large models          | Very large single-layer ops  |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Column-Parallel and Row-Parallel Linear — From Scratch": {
        "description": "Implement column-parallel and row-parallel linear layers from scratch, verify mathematical equivalence to the full single-GPU computation, and show the communication pattern.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
COLUMN-PARALLEL AND ROW-PARALLEL LINEAR — FROM SCRATCH
================================================================================

Implements Megatron-LM style tensor parallelism using Python threads to
simulate multiple GPU ranks.

Shows:
    1. Column-parallel linear: split W column-wise, no comm needed
    2. Row-parallel linear: split W row-wise, all-reduce at end
    3. Combined FFN: column-parallel → activation → row-parallel → all-reduce
    4. Mathematical equivalence to single-GPU computation

No actual GPUs needed — runs on CPU with threads.
================================================================================
"""

import threading
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared all-reduce ─────────────────────────────────────────────────────────

class AllReduceSync:
    """Thread-based all-reduce for TP simulation."""

    def __init__(self, tp_degree: int):
        self.tp      = tp_degree
        self.lock    = threading.Lock()
        self.barrier = threading.Barrier(tp_degree)
        self.store   = {}

    def all_reduce_sum(self, key: str, rank: int,
                        tensor: torch.Tensor) -> torch.Tensor:
        """Sum tensors across all TP ranks."""
        with self.lock:
            if key not in self.store:
                self.store[key] = [None] * self.tp
            self.store[key][rank] = tensor.detach().clone()

        self.barrier.wait()
        result = torch.stack(self.store[key]).sum(dim=0)

        with self.lock:
            if key in self.store:
                del self.store[key]
        return result


# ── Tensor-parallel FFN layer ─────────────────────────────────────────────────

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear: W split along output (column) dimension.
    Each rank holds W[:, rank*d_out_local : (rank+1)*d_out_local].
    Input x is the same on all ranks (replicated).
    Output y_local is different per rank (partial d_out).
    No communication needed in the forward pass.
    """
    def __init__(self, d_in: int, d_out: int, rank: int, tp: int,
                 bias: bool = False):
        super().__init__()
        assert d_out % tp == 0
        d_out_local = d_out // tp
        self.linear = nn.Linear(d_in, d_out_local, bias=bias)
        # Initialise with the same slice as the reference model (for verification)
        self.d_in        = d_in
        self.d_out       = d_out
        self.d_out_local = d_out_local
        self.rank        = rank
        self.tp          = tp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)   # y_local = x · W_t


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear: W split along input (row) dimension.
    Each rank holds W[rank*d_in_local : (rank+1)*d_in_local, :].
    Input y_local comes from the column-parallel layer (no full gather).
    Output z_local is partial sum; all-reduce needed to get full z.
    """
    def __init__(self, d_in: int, d_out: int, rank: int, tp: int,
                 bias: bool = False):
        super().__init__()
        assert d_in % tp == 0
        d_in_local = d_in // tp
        self.linear = nn.Linear(d_in_local, d_out, bias=bias)
        self.d_in        = d_in
        self.d_out       = d_out
        self.d_in_local  = d_in_local
        self.rank        = rank
        self.tp          = tp

    def forward(self, y_local: torch.Tensor) -> torch.Tensor:
        return self.linear(y_local)   # z_local = y_local · W_t (partial sum)


# ── Single-rank TP forward pass ───────────────────────────────────────────────

def tp_rank_forward(rank: int, tp: int, x: torch.Tensor,
                     col_weight: torch.Tensor,   # full W1
                     row_weight: torch.Tensor,   # full W2
                     allreduce: AllReduceSync,
                     results: dict, step: int):
    """
    One TP rank's contribution to the FFN forward pass.
    col_weight: (d, d_ff) — we use the column slice for this rank
    row_weight: (d_ff, d) — we use the row slice for this rank
    """
    d, d_ff = col_weight.shape
    d_ff_local = d_ff // tp
    d_in_local = d_ff // tp  # for row-parallel, d_ff split by rows

    # Column-parallel: each rank owns d_ff_local output columns of W1
    W1_local = col_weight[:, rank * d_ff_local : (rank + 1) * d_ff_local]
    y_local  = x @ W1_local              # (B, T, d_ff_local)
    y_local  = F.gelu(y_local)           # activation (no comm)

    # Row-parallel: each rank owns d_in_local input rows of W2
    W2_local = row_weight[rank * d_in_local : (rank + 1) * d_in_local, :]
    z_local  = y_local @ W2_local        # (B, T, d)

    # All-reduce to get full output
    z_full   = allreduce.all_reduce_sum(f"step{step}_z", rank, z_local)

    results[rank] = z_full


# ── Reference: single-GPU full computation ───────────────────────────────────

def single_gpu_forward(x, W1, W2):
    """Reference: full computation on a single GPU."""
    return F.gelu(x @ W1) @ W2


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    TP   = 4
    B, T = 2, 8
    D    = 64
    D_FF = 128

    # Create weight matrices
    W1 = torch.randn(D, D_FF)
    W2 = torch.randn(D_FF, D)

    # Input (same on all ranks — replicated)
    x  = torch.randn(B, T, D)

    print("=" * 60)
    print(f"  TENSOR PARALLELISM: COLUMN + ROW PARALLEL FFN")
    print(f"  TP={TP}, d={D}, d_ff={D_FF}, B={B}, T={T}")
    print("=" * 60)
    print()

    # Reference output
    ref_out = single_gpu_forward(x, W1, W2)
    print(f"  Reference output shape: {ref_out.shape}")
    print(f"  Reference norm:         {ref_out.norm():.4f}")

    # TP simulation
    allreduce = AllReduceSync(TP)
    results   = {}
    threads   = [
        threading.Thread(
            target=tp_rank_forward,
            args=(r, TP, x, W1, W2, allreduce, results, 0)
        )
        for r in range(TP)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    print()
    print("  TP outputs (should all be identical to reference):")
    for r in range(TP):
        diff = (results[r] - ref_out).abs().max().item()
        print(f"    Rank {r}: max diff from reference = {diff:.2e}  "
              f"{'✓ identical' if diff < 1e-4 else '✗ differs'}")

    # Memory analysis
    print()
    print("=" * 60)
    print("  MEMORY SAVINGS FROM TENSOR PARALLELISM")
    print("=" * 60)
    print()
    D_MODEL = 4096
    D_FF_REAL = 11008
    dtype_bytes = 2

    print(f"  {'TP degree':>10}  {'W1 per GPU':>14}  {'W2 per GPU':>14}  "
          f"{'Total per GPU':>16}  {'vs TP=1':>10}")
    print(f"  {'':─>10}  {'':─>14}  {'':─>14}  {'':─>16}  {'':─>10}")

    for tp in [1, 2, 4, 8]:
        w1_mb  = D_MODEL * D_FF_REAL / tp * dtype_bytes / 1e6
        w2_mb  = D_FF_REAL * D_MODEL / tp * dtype_bytes / 1e6
        total  = w1_mb + w2_mb
        ratio  = (D_MODEL * D_FF_REAL * 2 * dtype_bytes / 1e6) / total
        print(f"  {tp:>10}  {w1_mb:>14.1f} MB  {w2_mb:>14.1f} MB  "
              f"{total:>16.1f} MB  {ratio:>9.1f}×")

    print()
    print("  Communication pattern per block (forward pass):")
    print("  • Column-parallel W1:  no communication (input x replicated)")
    print("  • Row-parallel W2:     1 all-reduce (sum partial z_t)")
    print("  • Similarly for attention: 1 all-reduce per attention block")
    print("  Total: 2 all-reduces per Transformer block (forward pass)")
''',
    },

    "Tensor-Parallel Attention Layer": {
        "description": "Implement tensor-parallel multi-head attention — split Q/K/V heads across ranks, compute local attention, all-reduce the output projection.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
TENSOR-PARALLEL MULTI-HEAD ATTENTION
================================================================================

Implements tensor-parallel MHA:
    1. Column-parallel Q, K, V projections (heads split across ranks)
    2. Local attention computation (each rank handles its heads)
    3. Row-parallel output projection + all-reduce

Verifies mathematical equivalence to standard MHA.
================================================================================
"""

import threading
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── All-reduce helper ─────────────────────────────────────────────────────────

class AllReduceBarrier:
    def __init__(self, n: int):
        self.n       = n
        self.lock    = threading.Lock()
        self.barrier = threading.Barrier(n)
        self.store   = {}

    def all_reduce(self, key: str, rank: int, tensor: torch.Tensor,
                    op: str = "sum") -> torch.Tensor:
        with self.lock:
            if key not in self.store:
                self.store[key] = [None] * self.n
            self.store[key][rank] = tensor.detach().clone()
        self.barrier.wait()
        stacked = torch.stack(self.store[key])
        result  = stacked.sum(dim=0) if op == "sum" else stacked.mean(dim=0)
        with self.lock:
            if key in self.store:
                del self.store[key]
        return result


# ── Standard MHA (reference) ─────────────────────────────────────────────────

def standard_mha(x: torch.Tensor,
                  W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor,
                  W_O: torch.Tensor) -> torch.Tensor:
    """Full MHA on one device."""
    B, T, d = x.shape
    d_model  = W_Q.shape[0]
    h        = W_Q.shape[1] // (d_model // 32 if d_model >= 128 else 4)

    # Q, K, V
    Q = x @ W_Q   # (B, T, d)
    K = x @ W_K
    V = x @ W_V

    # Reshape to (B, h, T, d_head)
    d_head = d // h
    def split(t):
        return t.view(B, T, h, d_head).transpose(1, 2)

    Q, K, V = split(Q), split(K), split(V)

    # Scaled dot-product attention
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)
    mask   = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out    = (weights @ V).transpose(1, 2).contiguous().view(B, T, d)

    return out @ W_O   # (B, T, d)


# ── TP rank forward pass ──────────────────────────────────────────────────────

def tp_attention_rank(rank: int, tp: int, x: torch.Tensor,
                       W_Q: torch.Tensor, W_K: torch.Tensor,
                       W_V: torch.Tensor, W_O: torch.Tensor,
                       n_heads: int, allreduce: AllReduceBarrier,
                       results: dict, step: int):
    """
    One TP rank's attention computation.
    Rank t owns heads [t*h_local, (t+1)*h_local).
    """
    B, T, d = x.shape
    h_local = n_heads // tp
    d_head  = d // n_heads
    d_local = h_local * d_head   # local head dimension

    # Column-parallel Q, K, V: each rank gets d_local output columns
    Q_local = x @ W_Q[:, rank * d_local : (rank + 1) * d_local]  # (B,T,d_local)
    K_local = x @ W_K[:, rank * d_local : (rank + 1) * d_local]
    V_local = x @ W_V[:, rank * d_local : (rank + 1) * d_local]

    # Reshape to (B, h_local, T, d_head)
    def reshape(t):
        return t.view(B, T, h_local, d_head).transpose(1, 2)

    Q_l, K_l, V_l = reshape(Q_local), reshape(K_local), reshape(V_local)

    # Local attention (only over the local heads)
    scores = (Q_l @ K_l.transpose(-2, -1)) / math.sqrt(d_head)
    mask   = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out_local = (weights @ V_l).transpose(1, 2).contiguous().view(B, T, d_local)

    # Row-parallel output projection: each rank owns d_local input rows
    W_O_local = W_O[rank * d_local : (rank + 1) * d_local, :]
    z_local   = out_local @ W_O_local    # (B, T, d) — partial sum

    # All-reduce to get full output
    z_full = allreduce.all_reduce(f"step{step}_attn", rank, z_local)
    results[rank] = z_full


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    TP      = 4
    B, T    = 2, 8
    D       = 64
    N_HEADS = 8    # must be divisible by TP
    D_HEAD  = D // N_HEADS

    assert N_HEADS % TP == 0, "n_heads must be divisible by TP degree"

    # Create full weight matrices (same as reference)
    W_Q = torch.randn(D, D)
    W_K = torch.randn(D, D)
    W_V = torch.randn(D, D)
    W_O = torch.randn(D, D)
    x   = torch.randn(B, T, D)

    print("=" * 60)
    print(f"  TENSOR-PARALLEL MULTI-HEAD ATTENTION")
    print(f"  TP={TP}, n_heads={N_HEADS}, heads/rank={N_HEADS//TP}")
    print(f"  d={D}, d_head={D_HEAD}, B={B}, T={T}")
    print("=" * 60)
    print()

    # Reference output
    ref_out = standard_mha(x, W_Q, W_K, W_V, W_O)
    print(f"  Reference output shape: {ref_out.shape}")
    print(f"  Reference norm:         {ref_out.norm():.4f}")

    # TP simulation
    allreduce = AllReduceBarrier(TP)
    results   = {}
    threads   = [
        threading.Thread(
            target=tp_attention_rank,
            args=(r, TP, x, W_Q, W_K, W_V, W_O, N_HEADS, allreduce, results, 0)
        )
        for r in range(TP)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    print()
    print("  TP attention outputs (all ranks should match reference):")
    for r in range(TP):
        diff = (results[r] - ref_out).abs().max().item()
        print(f"    Rank {r}: max diff = {diff:.2e}  "
              f"{'✓ correct' if diff < 1e-4 else '✗ error'}")

    print()
    print("=" * 60)
    print("  WORK DISTRIBUTION PER RANK")
    print("=" * 60)
    print()
    print(f"  Full model:   {N_HEADS} heads, d_head={D_HEAD}")
    print(f"  Per rank:     {N_HEADS//TP} heads, d_local={D//TP}")
    print()
    print(f"  Q,K,V matmul:  {D} × {D}  → {D} × {D//TP}  per rank")
    print(f"  Attn compute:  ({N_HEADS//TP}, {T}, {T}) attn matrix per rank")
    print(f"  O matmul:      {D//TP} × {D}  per rank → partial sum")
    print(f"  All-reduce:    1× per attention block")
    print()
    print("  Head parallelism is clean: no cross-head dependencies in attention.")
    print("  This makes attention naturally tensor-parallelisable.")
''',
    },

    "TP Communication Cost and Degree Selection": {
        "description": "Analytical comparison of TP communication cost vs DDP, showing optimal TP degree for different model sizes, and the NVLink-only constraint.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
TP COMMUNICATION COST AND OPTIMAL DEGREE SELECTION
================================================================================

Computes:
    1. Communication volume per step for TP vs DDP
    2. Latency at different TP degrees and interconnects
    3. Optimal TP degree for common model configurations
    4. Memory savings per GPU as a function of TP degree

================================================================================
"""

import math


def tp_comm_per_step(n_layers: int, batch: int, seq_len: int,
                     d_model: int, tp: int,
                     dtype_bytes: int = 2) -> float:
    """
    Estimate TP communication volume per training step (GB).
    Each layer needs 2 all-reduces in forward and 2 in backward.
    Each all-reduce carries (B × T × d) per rank.
    Ring all-reduce volume: 2 × (N-1)/N × message_size per rank.
    """
    message_size_gb  = batch * seq_len * d_model * dtype_bytes / 1e9
    # 2 all-reduces per layer (attn + FFN) × 2 (forward + backward)
    # = 4 all-reduces per layer, each with ring volume 2 × (tp-1)/tp × msg
    ring_factor      = 2 * (tp - 1) / tp
    comm_per_layer   = 4 * ring_factor * message_size_gb
    return n_layers * comm_per_layer


def ddp_comm_per_step(n_params: int, dtype_bytes: int = 2) -> float:
    """DDP gradient all-reduce: 2 × grad_size GB."""
    return 2 * n_params * dtype_bytes / 1e9


def latency_s(comm_gb: float, bandwidth_gbps: float) -> float:
    return comm_gb / bandwidth_gbps


def tp_memory_per_gpu(n_params: int, tp: int, dtype_bytes: int = 2) -> float:
    """Weight memory per GPU with TP (GB). Each GPU holds 1/TP of each layer."""
    return n_params * dtype_bytes / tp / 1e9


def fmt_gb(gb: float) -> str:
    if gb >= 1: return f"{gb:.1f} GB"
    return f"{gb*1000:.0f} MB"


def fmt_ms(s: float) -> str:
    return f"{s*1000:.1f} ms"


if __name__ == "__main__":
    # LLaMA-2 70B reference config
    N_PARAMS   = 70_000_000_000
    N_LAYERS   = 80
    D_MODEL    = 8192
    SEQ_LEN    = 2048
    BATCH      = 1

    bws = [
        ("NVLink 3.0 (A100)", 600.0),
        ("InfiniBand HDR",     25.0),
        ("100GbE Ethernet",    12.0),
    ]

    print("=" * 70)
    print("  TENSOR PARALLELISM vs DDP COMMUNICATION COMPARISON")
    print("  LLaMA-2 70B: N=80 layers, d=8192, B=1, T=2048")
    print("=" * 70)
    print()

    # DDP baseline
    ddp_vol = ddp_comm_per_step(N_PARAMS)
    print(f"  DDP all-reduce volume per step: {fmt_gb(ddp_vol)}")
    print()

    for bw_name, bw in bws:
        ddp_lat = latency_s(ddp_vol, bw)
        print(f"  {bw_name} ({bw:.0f} GB/s):")
        print(f"    DDP all-reduce: {fmt_gb(ddp_vol)} → {fmt_ms(ddp_lat)}")
        print()
        print(f"    {'TP degree':>10}  {'TP comm (GB)':>14}  {'TP latency':>12}  "
              f"{'vs DDP':>10}  {'In bandwidth?':>14}")
        print(f"    {'':─>10}  {'':─>14}  {'':─>12}  {'':─>10}  {'':─>14}")
        for tp in [1, 2, 4, 8]:
            if tp == 1:
                print(f"    {tp:>10}  {'(none)':>14}  {'(none)':>12}  {'baseline':>10}  {'N/A':>14}")
                continue
            vol = tp_comm_per_step(N_LAYERS, BATCH, SEQ_LEN, D_MODEL, tp)
            lat = latency_s(vol, bw)
            ratio = ddp_lat / lat
            ok = "✓ within BW" if bw >= 100 else "⚠️ slow" if bw < 50 else "ok"
            print(f"    {tp:>10}  {fmt_gb(vol):>14}  {fmt_ms(lat):>12}  "
                  f"{ratio:>9.1f}×  {ok:>14}")
        print()

    # Memory savings
    print("=" * 70)
    print("  MEMORY PER GPU vs TP DEGREE")
    print("=" * 70)
    print()
    print(f"  {'TP':>5}  {'Weights/GPU':>14}  {'Reduction':>12}  {'Works on':>20}")
    print(f"  {'':─>5}  {'':─>14}  {'':─>12}  {'':─>20}")

    full_mem = tp_memory_per_gpu(N_PARAMS, 1)
    for tp in [1, 2, 4, 8, 16]:
        mem_gpu = tp_memory_per_gpu(N_PARAMS, tp)
        reduction = full_mem / mem_gpu
        if tp <= 8:
            hardware = "1 NVLink node ✓"
        else:
            hardware = "cross-node ✗ (too slow)"
        print(f"  {tp:>5}  {fmt_gb(mem_gpu):>14}  {reduction:>11.0f}×  {hardware:>20}")

    print()
    print("  For LLaMA-2 70B (140 GB at bf16):")
    print("    TP=1: 140 GB per GPU — doesn't fit on A100 80GB")
    print("    TP=2:  70 GB per GPU — barely fits")
    print("    TP=4:  35 GB per GPU — comfortable fit")
    print("    TP=8:  18 GB per GPU — lots of headroom for activations")
    print()

    # Valid TP degrees for common model configs
    print("=" * 70)
    print("  VALID TP DEGREES FOR COMMON MODEL CONFIGURATIONS")
    print("  (must evenly divide d_model, n_heads, and n_kv_heads)")
    print("=" * 70)
    print()

    configs = [
        ("LLaMA-2 7B",  4096, 32, 32),
        ("LLaMA-2 70B", 8192, 64,  8),
        ("LLaMA-3 8B",  4096, 32,  8),
        ("LLaMA-3 70B", 8192, 64,  8),
        ("GPT-3 175B",  12288, 96, 96),
    ]

    print(f"  {'Model':<18} {'d_model':>8}  {'n_heads':>8}  {'n_kv':>6}  {'Valid TP degrees':>20}")
    print(f"  {'':─<18} {'':─>8}  {'':─>8}  {'':─>6}  {'':─>20}")
    for name, d, h, kv in configs:
        valid = [tp for tp in [1, 2, 4, 8, 16, 32]
                 if d % tp == 0 and h % tp == 0 and kv % tp == 0]
        valid_in_node = [tp for tp in valid if tp <= 8]
        print(f"  {name:<18} {d:>8,}  {h:>8}  {kv:>6}  "
              f"{str(valid_in_node):>20}  (all: {valid})")

    print()
    print("  n_kv_heads is often the limiting factor for GQA models.")
    print("  LLaMA-2 70B has n_kv=8, so max practical TP = 8.")
    print("  Models with n_kv=4 can only use TP ≤ 4 for full correctness.")
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
    try:
        from llm_training.visuals.tensor_parallelism import (
            TP_VISUAL_HTML,
            TP_VISUAL_HEIGHT,
        )
        visual_html   = TP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = TP_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(
            f"[19_tensor_parallelism.py] Could not load visual: {e}",
            stacklevel=2,
        )

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