"""
Activation Checkpointing
=========================

Activations — the intermediate tensors stored during the forward pass for
use in the backward pass — often consume more memory than the model weights
themselves. For a 7B model at sequence length 2048, activations can exceed
50 GB. Activation checkpointing trades compute for memory by discarding
most activations during the forward pass and recomputing them on-demand
during the backward pass. It is one of the most important memory reduction
techniques and is universally used in large-scale LLM training.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Activation Checkpointing"
DISPLAY_NAME = "16 · Activation Checkpointing"
ICON         = "💾"
SUBTITLE     = "Trading Compute for Memory"


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

### The Activation Memory Problem

During the forward pass, every intermediate tensor (activation) is stored
in GPU memory so it can be used to compute gradients during the backward
pass. For a deep network, this accumulates rapidly.

**Why activations are so large:**
Consider one Transformer block processing a batch of B sequences, each
of length T tokens, with model dimension d:

    Component                    Tensor shape        Memory (fp16)
    ──────────────────────────────────────────────────────────────────
    Input to block               (B, T, d)           B×T×d×2 bytes
    Q, K, V projections (each)   (B, T, d)           B×T×d×2 bytes × 3
    Attention scores             (B, h, T, T)         B×h×T²×2 bytes
    Attention output             (B, T, d)           B×T×d×2 bytes
    FFN intermediate             (B, T, d_ff)         B×T×d_ff×2 bytes
    FFN output                   (B, T, d)           B×T×d×2 bytes
    ──────────────────────────────────────────────────────────────────
    Per-block total              ≈ B×T×(12d + h×T) × 2 bytes

For LLaMA-2 7B (d=4096, h=32, d_ff=11008, T=2048, B=1):
    Per-block ≈ 1 × 2048 × (12×4096 + 32×2048) × 2
              ≈ 1 × 2048 × (49152 + 65536) × 2
              ≈ 470 MB per Transformer block

    × 32 blocks = 15 GB activations per training step

At batch size 8: 120 GB — far exceeding an A100's 80 GB.

Even at batch size 1 with sequence length 4096, activation memory
dominates and prevents training without memory reduction.


### How Backpropagation Uses Activations

During backpropagation, the backward pass computes gradients layer by
layer in reverse order. For each layer, it needs the *activation from
the forward pass* of that layer:

    Layer L forward:   y = f(x; W)
    Layer L backward:  ∂L/∂W = ∂L/∂y × ∂y/∂W  = ∂L/∂y × g(x, W)

The function g requires x (the input activation to this layer). Without
storing x during the forward pass, the backward pass cannot compute the
gradient for W.

**Naive approach (standard autograd):** Store ALL activations during
the forward pass. Peak memory = sum of all intermediate tensors.

**Activation checkpointing:** Store only a subset of activations.
Recompute the discarded ones when needed during the backward pass.


### The Checkpointing Idea

The key insight: discarding activations and recomputing them from a saved
"checkpoint" costs one extra forward pass per checkpointed segment, but
reduces peak activation memory dramatically.

**Trade-off:**
    •   Memory saved:   (1 - 1/√N) × N ≈ most activations (for √N checkpoints)
    •   Compute added:  ~33% more FLOPs (one extra forward pass per segment)

For large models where memory is the bottleneck, paying 33% more compute
to reduce activation memory by 10–30× is a highly favourable trade.


    **Diagram 1 — Standard vs Checkpointed Forward/Backward Pass:**

    STANDARD (store all activations):
    ════════════════════════════════════════════════════════════════

    Forward:  x0 → [L1] → x1 → [L2] → x2 → [L3] → x3 → Loss
                          ↓            ↓            ↓
                       stored       stored       stored    ← 3 activations

    Backward: Loss → [L3_bwd, uses x2] → [L2_bwd, uses x1] → [L1_bwd, uses x0]
              Peak memory: x0 + x1 + x2 + x3 = 4 activations simultaneously

    CHECKPOINTED (checkpoint at x0 only, discard x1, x2):
    ════════════════════════════════════════════════════════════════

    Forward:  x0 → [L1] → x1 → [L2] → x2 → [L3] → x3 → Loss
              ↓            ↑            ↑
           checkpoint   discard      discard       ← only 1 stored!

    Backward:
      Step 1: Need x2 → recompute: x0→[L1]→x1→[L2]→x2  (uses x0 checkpoint)
              [L3_bwd, uses x2 just computed]
      Step 2: Need x1 → already have x0, recompute: x0→[L1]→x1
              [L2_bwd, uses x1 just computed]
      Step 3: [L1_bwd, uses x0 (still stored)]

    Peak memory: x0 (checkpoint) + 1 recomputed activation = 2 ≪ 4 ✓
    Extra compute: 2 extra forward passes (x0→x2, x0→x1) ≈ 33% overhead


### Granularity of Checkpointing

The granularity of checkpointing determines the memory/compute trade-off:

    **Per-block checkpointing (standard for Transformers):**
    Checkpoint at the input to each Transformer block. During backward,
    recompute one block at a time. Memory reduction: ~N× (N = num blocks),
    compute overhead: ~33%.

    **Selective checkpointing:**
    Only checkpoint computationally expensive layers (attention), not cheap
    ones (LayerNorm, residual add). This gives most of the memory benefit
    with less compute overhead. Used in FlashAttention-2.

    **Segment checkpointing:**
    Checkpoint every k layers, recompute k layers per backward segment.
    k=1 (per-layer) is the most memory-efficient but highest overhead.
    k=√N (one per √N layers) is the optimal trade-off for general networks.

    **No checkpointing (for inference):**
    Not needed during inference since no backward pass is required.


### PyTorch's `checkpoint` API

PyTorch provides `torch.utils.checkpoint.checkpoint()`:

    from torch.utils.checkpoint import checkpoint

    # Without checkpointing:
    x = transformer_block(x)

    # With checkpointing:
    x = checkpoint(transformer_block, x,
                   use_reentrant=False)  # prefer use_reentrant=False

The `use_reentrant` flag:
    •   `use_reentrant=True` (old default): uses PyTorch autograd hooks;
        can cause issues with certain modules (e.g. those using `_saved_tensors`)
    •   `use_reentrant=False` (recommended): uses a cleaner re-entrant-safe
        implementation that works better with FSDP and compiled models


### Memory Reduction: Exact Numbers

For a standard Transformer layer in bf16 with B=1, T=2048, d=4096:

    Without checkpointing:  ~470 MB per layer × 32 = ~15 GB total
    With checkpointing:     ~30 MB per layer × 32 +   ← only block inputs
                            ~470 MB × 1 at a time     ← max 1 recomputed
                            ≈ 1.4 GB steady-state peak

    Memory reduction: ~10× (from 15 GB to ~1.4 GB of activations)

The remaining ~30 MB per layer is the stored checkpoint tensor (the input
to each block), which is necessary as the starting point for recomputation.

**With FlashAttention + checkpointing:**
    FlashAttention already avoids materialising the full (T×T) attention
    matrix (saving another ~h×T² bytes). Combined with checkpointing:
    activation memory ≈ 2 MB per layer → 32 layers → ~64 MB total (!)

This is why FlashAttention + activation checkpointing together enable
training very large models at reasonable batch sizes.


    **Diagram 2 — Memory Profile: Standard vs Checkpointed:**

    MEMORY PROFILE DURING TRAINING (32-layer model)
    ════════════════════════════════════════════════════════════════

    STANDARD:
    Memory
    ▲
    │                          ██ peak (all activations stored)
    │                  ████████
    │          ████████
    │  ████████
    └──────────────────────────────────────────────────► step
       Forward pass            Backward pass

    CHECKPOINTED:
    Memory
    ▲
    │         █ one checkpoint stored
    │         ▐▌ + one recomputed layer at a time
    │  ████████▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌▐▌
    └──────────────────────────────────────────────────► step
       Forward pass            Backward pass
       (just storing          (recompute + immediately free)
        checkpoints)

    Peak memory reduction: ~10× for full checkpointing


### Compute Overhead in Practice

The theoretical 33% overhead assumes recomputing one complete forward pass
for the checkpointed segment. In practice:

    •   Recomputation uses the same kernels as the original forward pass
    •   On modern GPUs with fast memory, the overhead is often 20–30%
    •   FlashAttention's recompute in backward is kernel-fused — the overhead
        is hidden in the attention kernel itself
    •   Gradient checkpointing with `use_reentrant=False` has slightly less
        overhead than `True` due to fewer Python interpreter invocations

**Real-world overhead measurements (approximate):**
    •   Per-block checkpointing: +20–35% training time
    •   Selective checkpointing (attn only): +10–15% training time
    •   No checkpointing (large batch): N/A (OOM without it)


### Offloading Activations to CPU

An alternative to recomputation: store activations in CPU RAM rather than
discarding them. This is called **activation offloading**.

    •   GPU → CPU transfer: activations stored in pinned CPU memory
    •   CPU → GPU transfer: activations fetched back when needed
    •   Cost: PCIe bandwidth (limited at ~32 GB/s vs 2 TB/s HBM)

In practice, activation offloading is only beneficial when:
    1.  CPU RAM is plentiful (common: 1–2 TB on DGX nodes)
    2.  PCIe bandwidth is not the bottleneck (i.e., computation takes longer
        than the transfer)
    3.  The batch size is very large (transfer can be pipelined with compute)

For most LLM training runs, recomputation is preferred over offloading
because PCIe is a severe bottleneck at the activation sizes involved.


### Integration with FSDP and ZeRO

Activation checkpointing composes cleanly with sharded optimiser states:

    •   FSDP + activation checkpointing: standard combination for 7–70B models
    •   ZeRO-3 + activation checkpointing: maximum memory savings, used for
        models too large for a single node
    •   Pipeline parallelism: activations of pipeline-stage inputs are usually
        checkpointed since they must be kept until the backward bubble arrives

The order of application:
    1.  Apply per-block activation checkpointing in the model definition
    2.  Wrap with FSDP (shards parameters and gradients)
    3.  Apply ZeRO (shards optimiser state)
    4.  Add gradient accumulation if batch size is still too small

Each layer reduces memory from a different source; they stack multiplicatively.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Activation Checkpointing Strategy Comparison

| Strategy                 | Memory saving      | Compute overhead | Best for                         |
|--------------------------|--------------------|------------------|----------------------------------|
| No checkpointing         | 0× (baseline)      | 0%               | Small models / inference         |
| Per-block checkpointing  | ~10×               | +20–33%          | Standard LLM training            |
| Selective (attn only)    | ~5×                | +10–15%          | When compute is the bottleneck   |
| √N checkpoints           | ~√N ×              | +√N fwd passes   | Optimal general trade-off        |
| CPU offloading           | ~10× (no recompute)| PCIe bandwidth   | Huge CPU RAM, large batch        |
| FlashAttention built-in  | Avoids T² storage  | ~0% (kernel)     | Always — use FlashAttention      |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Activation Checkpointing — From Scratch": {
        "description": "Implement gradient checkpointing from scratch using PyTorch autograd hooks. Shows exactly how activations are discarded and recomputed, with memory and timing comparison.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ACTIVATION CHECKPOINTING — FROM SCRATCH
================================================================================

Implements gradient checkpointing manually using PyTorch's custom autograd:
    1. A manual checkpoint function that discards and recomputes activations
    2. Memory measurement showing reduction
    3. Timing comparison: standard vs checkpointed training
    4. Comparison with PyTorch's built-in checkpoint

Also demonstrates the √N optimal checkpoint placement strategy.
================================================================================
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ── Manual checkpoint implementation ─────────────────────────────────────────

class ManualCheckpoint(torch.autograd.Function):
    """
    A manual implementation of gradient checkpointing.

    Forward: runs the function and discards intermediate activations.
    Backward: recomputes the forward pass to recover activations, then
              computes gradients.
    """

    @staticmethod
    def forward(ctx, fn, *args):
        # Save inputs (not outputs) for recomputation
        ctx.fn   = fn
        ctx.save_for_backward(*args)
        # Run forward without gradient tracking to avoid storing a graph
        with torch.no_grad():
            output = fn(*args)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        args = ctx.saved_tensors
        # Recompute the forward pass WITH gradient tracking this time
        with torch.enable_grad():
            detached_args = tuple(a.detach().requires_grad_(a.requires_grad)
                                  for a in args)
            output = ctx.fn(*detached_args)

        # Compute gradients via standard backprop on the recomputed output
        torch.autograd.backward(output, grad_outputs)
        return (None,) + tuple(a.grad for a in detached_args)


def manual_checkpoint(fn, *args):
    """Drop-in replacement for torch.utils.checkpoint.checkpoint."""
    return ManualCheckpoint.apply(fn, *args)


# ── Model with and without checkpointing ─────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, d: int, n_layers: int, use_checkpoint: bool = False,
                 checkpoint_fn=None):
        super().__init__()
        self.blocks          = nn.ModuleList([TransformerBlock(d) for _ in range(n_layers)])
        self.use_checkpoint  = use_checkpoint
        self.checkpoint_fn   = checkpoint_fn or torch_checkpoint
        self.head            = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint:
                x = self.checkpoint_fn(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x)


# ── Memory measurement helpers ────────────────────────────────────────────────

def measure_peak_memory_cpu(fn):
    """Measure peak CPU memory (approximate) using tracemalloc."""
    import tracemalloc
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def count_tensors_in_graph(tensor: torch.Tensor) -> int:
    """Count the number of saved tensors in the computation graph."""
    visited = set()
    count   = [0]

    def visit(node):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        if hasattr(node, "saved_tensors"):
            count[0] += len(node.saved_tensors)
        if hasattr(node, "next_functions"):
            for fn, _ in node.next_functions:
                visit(fn)

    visit(tensor.grad_fn)
    return count[0]


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    D       = 256
    N_LAYERS = 8
    B, T    = 2, 64

    torch.manual_seed(42)
    x = torch.randn(B, T, D, requires_grad=True)
    x_ckpt = x.clone().detach().requires_grad_(True)

    print("=" * 60)
    print(f"  ACTIVATION CHECKPOINTING COMPARISON")
    print(f"  d={D}, n_layers={N_LAYERS}, B={B}, T={T}")
    print("=" * 60)
    print()

    # 1. Standard (no checkpointing)
    model_std = TinyTransformer(D, N_LAYERS, use_checkpoint=False)
    out_std   = model_std(x)
    loss_std  = out_std.mean()

    saved_tensors_std = count_tensors_in_graph(loss_std)
    print(f"  Without checkpointing:")
    print(f"    Saved tensors in graph:  {saved_tensors_std}")

    # Time the backward pass
    t0 = time.perf_counter()
    for _ in range(10):
        out = model_std(x.clone().requires_grad_(True))
        out.mean().backward()
    t_std = (time.perf_counter() - t0) / 10 * 1000
    print(f"    Backward time:           {t_std:.2f} ms")

    # 2. With PyTorch built-in checkpointing
    model_ckpt = TinyTransformer(D, N_LAYERS, use_checkpoint=True,
                                  checkpoint_fn=torch_checkpoint)
    out_ckpt = model_ckpt(x_ckpt)
    loss_ckpt = out_ckpt.mean()

    saved_tensors_ckpt = count_tensors_in_graph(loss_ckpt)
    print()
    print(f"  With checkpointing (torch built-in):")
    print(f"    Saved tensors in graph:  {saved_tensors_ckpt}")

    t0 = time.perf_counter()
    for _ in range(10):
        xc = x.clone().requires_grad_(True)
        out = model_ckpt(xc)
        out.mean().backward()
    t_ckpt = (time.perf_counter() - t0) / 10 * 1000
    print(f"    Backward time:           {t_ckpt:.2f} ms")
    print(f"    Compute overhead:        {(t_ckpt/t_std - 1)*100:.1f}%")
    print(f"    Graph tensor reduction:  {saved_tensors_std/max(saved_tensors_ckpt,1):.1f}×")

    # 3. √N optimal checkpoint placement
    print()
    print("=" * 60)
    print("  √N OPTIMAL CHECKPOINT PLACEMENT")
    print("=" * 60)
    print()
    print("  For N layers, saving one checkpoint every √N layers minimises")
    print("  peak memory while bounding recomputation cost.")
    print()

    for N in [4, 8, 16, 32, 64]:
        sqrt_n = int(math.sqrt(N))
        n_checkpoints = N // sqrt_n
        # Peak memory: sqrt_n activations at once during recomputation
        peak_relative = sqrt_n          # relative to 1 activation
        recompute_cost = N // sqrt_n    # extra forward passes
        overhead_pct = (recompute_cost - 1) / (recompute_cost) * 100
        print(f"  N={N:2d} layers:  checkpoint every {sqrt_n} layers  "
              f"→  peak mem ≈ {peak_relative} activations  "
              f"recompute {recompute_cost-1} fwd passes")

    print()
    print("  Per-block (every 1 layer):")
    for N in [8, 32]:
        print(f"  N={N:2d} layers:  checkpoint every 1 layer  "
              f"→  peak mem ≈ 2 activations  recompute {N-1} fwd passes")

    print()
    print("  For Transformers, per-block is standard because:")
    print("  • Each block is a natural unit of computation")
    print("  • The 33% overhead is acceptable given 10× memory savings")

    # 4. Memory estimation
    print()
    print("=" * 60)
    print("  ACTIVATION MEMORY ESTIMATION")
    print("=" * 60)
    print()
    print(f"  {'Model':<22} {'d':>6}  {'T':>6}  {'N':>4}  "
          f"{'No ckpt (GB)':>14}  {'Ckpt (GB)':>12}  {'Savings':>10}")
    print(f"  {'':─<22} {'':─>6}  {'':─>6}  {'':─>4}  "
          f"{'':─>14}  {'':─>12}  {'':─>10}")

    configs = [
        ("GPT-2 Small",   768,  1024, 12),
        ("LLaMA-2 7B",   4096,  2048, 32),
        ("LLaMA-2 7B*4", 4096,  4096, 32),
        ("LLaMA-2 70B",  8192,  2048, 80),
    ]

    for name, d, T_len, N in configs:
        d_ff = d * 4
        h    = d // 64
        B_sz = 1
        dtype_bytes = 2   # bf16

        # Per-layer activation memory (approximate)
        attn_scores = B_sz * h * T_len * T_len * dtype_bytes  # attention matrix
        qkv         = 3 * B_sz * T_len * d * dtype_bytes
        ffn_inter   = B_sz * T_len * d_ff * dtype_bytes
        per_layer   = attn_scores + qkv + ffn_inter

        no_ckpt_gb  = N * per_layer / 1e9
        ckpt_gb     = (per_layer + (d * T_len * B_sz * dtype_bytes)) / 1e9  # 1 layer + inputs
        savings     = no_ckpt_gb / ckpt_gb if ckpt_gb > 0 else 0

        print(f"  {name:<22} {d:>6,}  {T_len:>6,}  {N:>4}  "
              f"{no_ckpt_gb:>14.1f}  {ckpt_gb:>12.2f}  {savings:>9.1f}×")
''',
    },

    "Selective Checkpointing and FlashAttention Integration": {
        "description": "Show how selective checkpointing targets expensive layers (attention), how FlashAttention eliminates the T² activation matrix, and their combined memory savings.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SELECTIVE CHECKPOINTING AND FLASHATTENTION INTEGRATION
================================================================================

Demonstrates:
    1. Selective checkpointing — only checkpoint attention, not FFN
    2. How FlashAttention avoids materialising the T×T attention matrix
    3. Combined memory savings of FlashAttention + per-block checkpointing
    4. Which components dominate activation memory at different seq lengths
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ── Activation memory analysis ────────────────────────────────────────────────

def activation_memory_breakdown(d: int, n_heads: int, d_ff: int,
                                  seq_len: int, batch: int = 1,
                                  dtype_bytes: int = 2) -> dict:
    """
    Compute the activation memory for each component of a Transformer block.
    """
    T = seq_len
    B = batch
    h = n_heads
    d_h = d // h

    # ── Attention components ──────────────────────────────────────────────────
    # Q, K, V projections: 3 tensors of shape (B, T, d)
    qkv_proj    = 3 * B * T * d * dtype_bytes

    # Attention scores (before softmax): (B, h, T, T)
    attn_scores = B * h * T * T * dtype_bytes  # THE BIG ONE

    # Attention weights (after softmax): (B, h, T, T)
    attn_weights = B * h * T * T * dtype_bytes

    # Output projection input: (B, T, d)
    attn_out    = B * T * d * dtype_bytes

    attn_total  = qkv_proj + attn_scores + attn_weights + attn_out

    # ── FFN components ────────────────────────────────────────────────────────
    # FFN intermediate: (B, T, d_ff)
    ffn_hidden  = B * T * d_ff * dtype_bytes

    # FFN output: (B, T, d)
    ffn_out     = B * T * d * dtype_bytes

    ffn_total   = ffn_hidden + ffn_out

    # ── LayerNorm ─────────────────────────────────────────────────────────────
    # Mean and variance for each LayerNorm: (B, T)
    ln_total    = 2 * 2 * B * T * dtype_bytes  # 2 LNs, 2 stats each

    total = attn_total + ffn_total + ln_total

    return {
        "qkv_proj_mb":       qkv_proj    / 1e6,
        "attn_scores_mb":    attn_scores / 1e6,
        "attn_weights_mb":   attn_weights / 1e6,
        "attn_out_mb":       attn_out    / 1e6,
        "ffn_hidden_mb":     ffn_hidden  / 1e6,
        "ffn_out_mb":        ffn_out     / 1e6,
        "ln_mb":             ln_total    / 1e6,
        "attn_total_mb":     attn_total  / 1e6,
        "ffn_total_mb":      ffn_total   / 1e6,
        "total_mb":          total       / 1e6,
        # FlashAttention avoids materialising attn_scores and attn_weights
        "flash_total_mb":    (total - attn_scores - attn_weights) / 1e6,
    }


def bar_mb(mb: float, max_mb: float, width: int = 25) -> str:
    n = int(mb / max_mb * width) if max_mb > 0 else 0
    return "█" * n + "░" * (width - n)


# ── Selective checkpointing ───────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    """Standard multi-head attention."""
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, x):
        xn = self.norm(x)
        return x + self.attn(xn, xn, xn, need_weights=False)[0]


class FFNLayer(nn.Module):
    """Standard FFN layer."""
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class SelectiveCheckpointBlock(nn.Module):
    """
    Transformer block with selective checkpointing:
        - Attention: checkpointed (expensive, large activations)
        - FFN: NOT checkpointed (cheaper relative to its activation cost)
    """

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.attn = AttentionLayer(d, n_heads)
        self.ffn  = FFNLayer(d)

    def forward(self, x, checkpoint_attn: bool = True, checkpoint_ffn: bool = False):
        if checkpoint_attn:
            x = checkpoint(self.attn, x, use_reentrant=False)
        else:
            x = self.attn(x)

        if checkpoint_ffn:
            x = checkpoint(self.ffn, x, use_reentrant=False)
        else:
            x = self.ffn(x)

        return x


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Activation memory breakdown by component
    print("=" * 65)
    print("  ACTIVATION MEMORY BREAKDOWN — LLaMA-2 7B BLOCK")
    print("  d=4096, n_heads=32, d_ff=11008, B=1")
    print("=" * 65)
    print()

    for T in [512, 1024, 2048, 4096, 8192]:
        stats = activation_memory_breakdown(4096, 32, 11008, T)
        max_mb = stats["total_mb"]
        print(f"  seq_len = {T:>5}")
        items = [
            ("QKV projections",  stats["qkv_proj_mb"]),
            ("Attn scores (T²)", stats["attn_scores_mb"]),
            ("Attn weights(T²)", stats["attn_weights_mb"]),
            ("Attn output",      stats["attn_out_mb"]),
            ("FFN hidden",       stats["ffn_hidden_mb"]),
            ("FFN output",       stats["ffn_out_mb"]),
        ]
        for name, mb in items:
            print(f"    {name:<22} {bar_mb(mb, max_mb)}  {mb:>8.1f} MB")
        print(f"    {'TOTAL':<22} {'█'*25}  {stats['total_mb']:>8.1f} MB")
        flash = stats["flash_total_mb"]
        print(f"    {'TOTAL (FlashAttn)':<22} {'█'*int(flash/max_mb*25):<25}  {flash:>8.1f} MB  ← T² avoided")
        print()

    # 2. Scaling analysis: how T² dominates at long contexts
    print("=" * 65)
    print("  ATTENTION SCORE FRACTION vs SEQUENCE LENGTH")
    print("  (Motivating FlashAttention's importance)")
    print("=" * 65)
    print()
    print(f"  {'seq_len':>8}  {'Total MB':>10}  {'Attn T²':>10}  {'Attn fraction':>14}  {'FA saves':>10}")
    print(f"  {'':─>8}  {'':─>10}  {'':─>10}  {'':─>14}  {'':─>10}")

    for T in [512, 1024, 2048, 4096, 8192, 16384]:
        s = activation_memory_breakdown(4096, 32, 11008, T)
        attn_frac = (s["attn_scores_mb"] + s["attn_weights_mb"]) / s["total_mb"]
        fa_saving = s["total_mb"] - s["flash_total_mb"]
        print(f"  {T:>8,}  {s['total_mb']:>10.1f}  "
              f"{s['attn_scores_mb']+s['attn_weights_mb']:>10.1f}  "
              f"{attn_frac:>14.1%}  {fa_saving:>10.1f} MB")

    print()
    print("  At T=8192, the T² attention matrices = >80% of activation memory!")
    print("  FlashAttention eliminates this by computing attention in tiles,")
    print("  never materialising the full T×T matrix in HBM.")

    # 3. Selective vs full checkpointing timing
    print()
    print("=" * 65)
    print("  SELECTIVE CHECKPOINTING: ATTN-ONLY vs FULL BLOCK")
    print("=" * 65)
    print()

    import time
    D, H, T, B = 256, 4, 64, 2
    model_block = SelectiveCheckpointBlock(D, H)
    x = torch.randn(B, T, D)

    configs_ckpt = [
        ("No checkpoint",   False, False),
        ("Attn only",       True,  False),
        ("FFN only",        False, True),
        ("Full block",      True,  True),
    ]

    print(f"  {'Config':<18}  {'Fwd+Bwd (ms)':>14}")
    print(f"  {'':─<18}  {'':─>14}")

    for name, ck_attn, ck_ffn in configs_ckpt:
        times = []
        for _ in range(20):
            xc = x.clone().requires_grad_(True)
            t0 = time.perf_counter()
            out = model_block(xc, checkpoint_attn=ck_attn, checkpoint_ffn=ck_ffn)
            out.mean().backward()
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times[5:]) / len(times[5:])  # skip warmup
        print(f"  {name:<18}  {avg:>14.2f}")

    print()
    print("  Attention checkpointing adds more overhead than FFN checkpointing")
    print("  because attention has more complex operations to recompute.")
    print("  However, attention also saves the most memory (T² tensors).")
    print("  → Selective attention checkpointing is the best memory/compute trade-off.")
''',
    },

    "Memory Budget Planning with Checkpointing": {
        "description": "Build a complete memory budget planner that shows how activation checkpointing, mixed precision, and batch size interact — find the maximum batch size for a given GPU memory budget.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MEMORY BUDGET PLANNER
================================================================================

Given a GPU memory budget and model configuration, compute:
    1. Memory components: weights, gradients, optimiser state, activations
    2. Impact of activation checkpointing on activation memory
    3. Maximum achievable batch size for each configuration
    4. Recommendations for hitting a target batch size

Helps answer: "Can I train this model on these GPUs with this batch size?"
================================================================================
"""

import math
from dataclasses import dataclass


@dataclass
class MemoryBudget:
    gpu_memory_gb: float          # total GPU HBM (e.g. 80 for A100)
    reserve_gb:    float = 2.0    # reserved for CUDA overhead, buffers


@dataclass
class ModelSpec:
    name:           str
    n_params:       int
    n_layers:       int
    d_model:        int
    n_heads:        int
    d_ff:           int
    vocab_size:     int = 32000
    dtype_model:    str = "bf16"   # "fp32" or "bf16" or "fp16"
    dtype_optim:    str = "fp32"   # optimiser state always fp32


DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}


def compute_memory(spec: ModelSpec, budget: MemoryBudget,
                   seq_len: int, batch: int,
                   use_checkpointing: bool = False,
                   use_flash_attn: bool = False,
                   grad_accum_steps: int = 1) -> dict:
    """
    Compute memory breakdown for training configuration.

    Returns dict with per-component memory in GB and feasibility.
    """
    mb = DTYPE_BYTES[spec.dtype_model]  # model dtype bytes
    ob = DTYPE_BYTES[spec.dtype_optim]  # optimiser dtype bytes
    T  = seq_len
    B  = batch
    d  = spec.d_model
    h  = spec.n_heads
    df = spec.d_ff
    N  = spec.n_layers

    # ── Fixed costs (independent of batch/seq) ────────────────────────────────
    weight_gb  = spec.n_params * mb / 1e9
    grad_gb    = spec.n_params * mb / 1e9
    master_gb  = spec.n_params * 4  / 1e9    # fp32 master weights
    moment_gb  = spec.n_params * ob * 2 / 1e9  # m + v in AdamW

    # Note: if dtype_model == fp32, master = weight (no duplication)
    if spec.dtype_model == "fp32":
        master_gb = 0.0   # no separate master needed

    optim_total_gb = master_gb + moment_gb

    # ── Activation memory (per layer, per token) ──────────────────────────────
    # Full activation per layer (without FlashAttention)
    def activations_per_layer(T: int, B: int) -> float:
        qkv        = 3 * B * T * d * mb
        attn_mat   = 2 * B * h * T * T * mb   # scores + weights (T²!)
        attn_out   = B * T * d * mb
        ffn_hidden = B * T * df * mb
        ffn_out    = B * T * d * mb
        total      = qkv + attn_mat + attn_out + ffn_hidden + ffn_out
        if use_flash_attn:
            total -= attn_mat   # FlashAttention avoids T² storage
        return total / 1e9

    act_per_layer = activations_per_layer(T, B)

    if use_checkpointing:
        # Only store block inputs (B × T × d) + one recomputed layer at a time
        block_input = B * T * d * mb / 1e9
        act_gb      = N * block_input + act_per_layer  # checkpoints + max 1 recomputed
    else:
        act_gb = N * act_per_layer

    # ── Total ─────────────────────────────────────────────────────────────────
    total_gb      = weight_gb + grad_gb + optim_total_gb + act_gb
    available_gb  = budget.gpu_memory_gb - budget.reserve_gb
    fits          = total_gb <= available_gb
    headroom_gb   = available_gb - total_gb

    return {
        "weights_gb":    weight_gb,
        "grads_gb":      grad_gb,
        "master_gb":     master_gb,
        "moments_gb":    moment_gb,
        "optim_total_gb":optim_total_gb,
        "activations_gb":act_gb,
        "total_gb":      total_gb,
        "available_gb":  available_gb,
        "fits":          fits,
        "headroom_gb":   headroom_gb,
        "act_per_layer": act_per_layer,
    }


def max_batch_size(spec: ModelSpec, budget: MemoryBudget,
                   seq_len: int,
                   use_checkpointing: bool = False,
                   use_flash_attn: bool = False,
                   max_batch: int = 256) -> int:
    """Binary search for the maximum batch size that fits in memory."""
    lo, hi = 1, max_batch
    best   = 0
    while lo <= hi:
        mid  = (lo + hi) // 2
        mem  = compute_memory(spec, budget, seq_len, mid,
                               use_checkpointing, use_flash_attn)
        if mem["fits"]:
            best = mid
            lo   = mid + 1
        else:
            hi   = mid - 1
    return best


def fmt_gb(gb: float) -> str:
    return f"{gb:.1f} GB"


if __name__ == "__main__":
    A100 = MemoryBudget(gpu_memory_gb=80.0, reserve_gb=3.0)

    models = [
        ModelSpec("LLaMA-2 7B",   7_000_000_000, 32, 4096, 32, 11008, dtype_model="bf16"),
        ModelSpec("LLaMA-2 13B", 13_000_000_000, 40, 5120, 40, 13824, dtype_model="bf16"),
        ModelSpec("LLaMA-2 70B", 70_000_000_000, 80, 8192, 64, 28672, dtype_model="bf16"),
    ]

    SEQ_LEN = 2048

    # 1. Memory breakdown for 7B at batch=1
    print("=" * 68)
    print("  MEMORY BREAKDOWN — LLaMA-2 7B, B=1, T=2048, BF16, A100 80GB")
    print("=" * 68)
    print()

    configs = [
        ("No ckpt, no FA",  False, False),
        ("Ckpt only",       True,  False),
        ("FA only",         False, True),
        ("Ckpt + FA",       True,  True),
    ]

    model7b = models[0]
    print(f"  {'Config':<20} {'Weights':>8}  {'Grad+Optim':>10}  "
          f"{'Activ':>8}  {'TOTAL':>8}  {'Fits?':>6}")
    print(f"  {'':─<20} {'':─>8}  {'':─>10}  "
          f"{'':─>8}  {'':─>8}  {'':─>6}")

    for name, ckpt, fa in configs:
        m = compute_memory(model7b, A100, SEQ_LEN, 1, ckpt, fa)
        wo = m["weights_gb"] + m["grads_gb"] + m["optim_total_gb"]
        print(f"  {name:<20} {fmt_gb(m['weights_gb']):>8}  {fmt_gb(wo):>10}  "
              f"{fmt_gb(m['activations_gb']):>8}  {fmt_gb(m['total_gb']):>8}  "
              f"{'✓' if m['fits'] else '✗':>6}")

    # 2. Maximum batch size comparison
    print()
    print("=" * 68)
    print("  MAXIMUM BATCH SIZE ON A100 80GB, T=2048")
    print("=" * 68)
    print()
    print(f"  {'Model':<16} {'No ckpt':>10}  {'Ckpt only':>10}  "
          f"{'FA only':>10}  {'Ckpt+FA':>12}")
    print(f"  {'':─<16} {'':─>10}  {'':─>10}  {'':─>10}  {'':─>12}")

    for spec in models:
        b_none   = max_batch_size(spec, A100, SEQ_LEN, False, False)
        b_ckpt   = max_batch_size(spec, A100, SEQ_LEN, True,  False)
        b_fa     = max_batch_size(spec, A100, SEQ_LEN, False, True)
        b_both   = max_batch_size(spec, A100, SEQ_LEN, True,  True)

        def fmt_b(b):
            return f"B={b}" if b > 0 else "OOM"

        print(f"  {spec.name:<16} {fmt_b(b_none):>10}  {fmt_b(b_ckpt):>10}  "
              f"{fmt_b(b_fa):>10}  {fmt_b(b_both):>12}")

    # 3. How batch size scales with sequence length (7B, ckpt+FA)
    print()
    print("=" * 68)
    print("  BATCH SIZE vs SEQUENCE LENGTH — LLaMA-2 7B, Ckpt+FA, A100 80GB")
    print("=" * 68)
    print()
    print(f"  {'seq_len':>8}  {'Max B':>8}  {'Global tokens (B=max)':>24}  "
          f"{'Activation GB':>15}")
    print(f"  {'':─>8}  {'':─>8}  {'':─>24}  {'':─>15}")

    for T in [512, 1024, 2048, 4096, 8192]:
        b    = max_batch_size(model7b, A100, T, True, True, max_batch=32)
        m    = compute_memory(model7b, A100, T, max(b, 1), True, True)
        gtok = b * T
        print(f"  {T:>8,}  {b:>8}  {gtok:>24,}  {fmt_gb(m['activations_gb']):>15}")

    print()
    print("  Recommendation for production LLM training:")
    print("  • Always use activation checkpointing — 33% compute overhead is")
    print("    almost always worth the 5–10× memory reduction.")
    print("  • Combine with FlashAttention for additional T² savings.")
    print("  • Increase grad_accum_steps to achieve target effective batch")
    print("    size if memory limits the per-GPU batch size.")
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
    #     from llm_training.visuals.activation_checkpointing import (
    #         ACTCKPT_VISUAL_HTML,
    #         ACTCKPT_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ACTCKPT_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ACTCKPT_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[16_activation_checkpointing.py] Could not load visual: {e}",
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