"""
Gradient Accumulation
======================

Gradient accumulation simulates a large effective batch size by running
multiple small micro-batches before each parameter update. It is the primary
technique for achieving the large batch sizes required for stable LLM training
when GPU memory limits the per-step batch size. Understanding how accumulation
interacts with learning rate scaling, BatchNorm (irrelevant for LLMs but worth
knowing), mixed precision, and distributed training is essential for correctly
implementing large-scale training recipes.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Gradient Accumulation"
DISPLAY_NAME = "17 · Gradient Accumulation"
ICON         = "🪣"
SUBTITLE     = "Simulating Large Batches on Small Memory"


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

### The Batch Size Problem in LLM Training

LLM training recipes typically target a **global batch size** of 1–8 million
tokens per optimiser step. This large batch size:
    •   Reduces gradient noise (more accurate gradient estimate)
    •   Enables higher learning rates (linear scaling rule)
    •   Provides better sample efficiency (fewer total steps needed)
    •   Is required to fill GPU Tensor Cores efficiently

But achieving 4M tokens per step on an A100 80GB GPU is often impossible
directly due to memory constraints:

    LLaMA-3 8B, T=4096, B=?:
        With activation checkpointing + FlashAttention: max B ≈ 4
        Tokens per step per GPU: 4 × 4096 = 16,384 tokens
        Target global batch: 4,000,000 tokens
        GPUs needed if no grad accum: 4M / 16K = 244 GPUs!

With gradient accumulation (grad_accum = 16):
    •   Per-GPU memory: same as B=4
    •   Tokens per step per GPU: 4 × 4096 × 16 = 262,144 tokens
    •   GPUs needed: 4M / 262K ≈ 15 GPUs

Gradient accumulation allows small teams to achieve the same effective batch
size as large clusters, just with more wall-clock time per effective step.


### The Mathematics of Gradient Accumulation

In a standard training step with batch size B:

    Loss    = (1/B) Σᵢ L(xᵢ; θ)
    Gradient = ∂Loss/∂θ = (1/B) Σᵢ ∂L(xᵢ; θ)/∂θ

With gradient accumulation over K micro-batches, each of size M (so B = K×M):

    For k = 1, …, K:
        g_k = ∂/∂θ [(1/M) Σᵢ∈micro_k L(xᵢ; θ)]

    Accumulated gradient = (1/K) Σₖ g_k = (1/(K×M)) Σᵢ ∂L(xᵢ; θ)/∂θ

This is **identical** to the gradient you would get from a single forward+backward
pass on all K×M samples simultaneously, assuming the model is the same before
each micro-batch (which it is, since we don't update between micro-batches).

**Critical point:** The loss for each micro-batch must be divided by K before
calling `.backward()` so the accumulated gradients sum to the correct average:

    loss = forward(micro_batch) / K
    loss.backward()   # adds loss/K 's gradient to .grad

After K such calls, `.grad` contains the mean gradient over all K×M samples.
Alternatively, divide by K after accumulation during the update step.


    **Diagram 1 — Gradient Accumulation vs Single Large Batch:**

    GRADIENT ACCUMULATION (K=4 micro-batches of M=2 = effective B=8)
    ════════════════════════════════════════════════════════════════

    Memory on GPU: holds only M=2 samples at a time

    Step 1  (micro-batch 1):  forward(x1,x2) → loss/4 → backward → grad += g1/4
    Step 2  (micro-batch 2):  forward(x3,x4) → loss/4 → backward → grad += g2/4
    Step 3  (micro-batch 3):  forward(x5,x6) → loss/4 → backward → grad += g3/4
    Step 4  (micro-batch 4):  forward(x7,x8) → loss/4 → backward → grad += g4/4
                                                                      ─────────────
                                                              total:  (g1+g2+g3+g4)/4
    Optimiser step → update θ using accumulated gradient
    Zero gradients → ready for next effective batch

    Memory usage:             M samples at a time  (not K×M!)
    Effective gradient:       same as using K×M samples  ✓


### The Division Trap — The Most Common Implementation Bug

The single most common mistake in gradient accumulation is forgetting to
divide the loss by the number of accumulation steps before calling backward.

**Wrong (gradients are K× too large):**
    for k in range(K):
        loss = forward(micro_batch[k])
        loss.backward()                    # ← loss is not divided by K
    optimizer.step()

**What happens:** `.grad` accumulates the sum of K gradients, not the mean.
The effective gradient is K× larger than it should be — equivalent to using
a learning rate of K×lr. This causes training to diverge or oscillate.

**Correct approach 1 — divide inside the loop:**
    for k in range(K):
        loss = forward(micro_batch[k]) / K  # ← divide by K
        loss.backward()
    optimizer.step()

**Correct approach 2 — divide after accumulation:**
    for k in range(K):
        loss = forward(micro_batch[k])
        loss.backward()                    # accumulate sum
    # Before optimizer step, scale gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(K)
    optimizer.step()

Both are mathematically equivalent. Approach 1 is more common and is what
HuggingFace Trainer and other frameworks implement.


### Interaction with Mixed Precision (GradScaler)

When using fp16 with GradScaler, loss scaling must be applied correctly
across micro-batches:

    scaler = GradScaler()

    for k in range(K):
        with autocast():
            loss = forward(micro_batch[k]) / K     # ← divide by K
        scaler.scale(loss).backward()              # ← scale before backward

    scaler.unscale_(optimizer)
    clip_grad_norm_(params, max_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

**Why unscale once at the end (not per micro-batch):**
The GradScaler multiplies the loss by a scale factor S before backward.
After K micro-batches, gradients = S × (true_grad / K × K) = S × true_grad.
The single `unscale_` call at the end divides all gradients by S, giving
the correct true_grad. Calling unscale_ inside the loop would be wrong.


### Interaction with Distributed Training (DDP)

In DDP, gradients are all-reduced across GPUs at each `.backward()` call.
With gradient accumulation, we want to delay the all-reduce until the last
micro-batch to avoid K all-reduce operations instead of 1:

    # Efficient: delay all-reduce until last micro-batch
    for k in range(K):
        # Disable all-reduce for all but the last micro-batch
        with model.no_sync() if k < K - 1 else contextlib.nullcontext():
            loss = forward(micro_batch[k]) / K
            loss.backward()
    # DDP all-reduce happens here (during the last backward)
    optimizer.step()

Without `model.no_sync()`, DDP would perform K all-reduce operations instead
of 1, multiplying communication overhead by K. The `no_sync()` context manager
tells DDP to accumulate gradients locally and defer the all-reduce.

**Communication savings with no_sync():**
    With    no_sync: 1 all-reduce per effective batch = same as without grad accum
    Without no_sync: K all-reduces per effective batch = K× more communication!


    **Diagram 2 — DDP + Gradient Accumulation with no_sync():**

    DDP + GRADIENT ACCUMULATION (K=3, 2 GPUs)
    ════════════════════════════════════════════════════════════════

    WITHOUT no_sync (WRONG — K× communication):
    ──────────────────────────────────────────────────────────────
    GPU0:  fwd1 → bwd1 ─[all-reduce]─ fwd2 → bwd2 ─[all-reduce]─ fwd3 → bwd3 ─[all-reduce]─ step
    GPU1:  fwd1 → bwd1 ─[all-reduce]─ fwd2 → bwd2 ─[all-reduce]─ fwd3 → bwd3 ─[all-reduce]─ step
    Cost:  3 × all-reduce (3× communication overhead!)

    WITH no_sync (CORRECT — 1 communication):
    ──────────────────────────────────────────────────────────────
    GPU0:  fwd1→bwd1 (local) → fwd2→bwd2 (local) → fwd3→bwd3 ─[all-reduce]─ step
    GPU1:  fwd1→bwd1 (local) → fwd2→bwd2 (local) → fwd3→bwd3 ─[all-reduce]─ step
    Cost:  1 × all-reduce ✓


### Gradient Accumulation and Learning Rate

When doubling the effective batch size via gradient accumulation (or more
GPUs), the learning rate should be scaled accordingly:

    **Linear scaling rule (Goyal et al., 2017):**
        lr_new = lr_base × (effective_batch / base_batch)

    **Square-root scaling (more conservative, often better for LLMs):**
        lr_new = lr_base × √(effective_batch / base_batch)

    Example: If the base recipe uses B=256 with lr=1e-4, and you want
    effective B=4096 via gradient accumulation:
        Linear: lr = 1e-4 × (4096/256) = 1.6e-3
        Sqrt:   lr = 1e-4 × √(4096/256) = 4e-4

**Warmup and batch size:** When using a larger effective batch size, the
warmup period in terms of *tokens* should stay the same, which means
fewer warmup *steps*:
    warmup_tokens = constant (e.g., 500M tokens)
    warmup_steps  = warmup_tokens / effective_batch_tokens


### Global Batch Size Calculation

Keeping track of the effective batch size across all its contributors:

    effective_batch_tokens = (
        micro_batch_size      # sequences per GPU per micro-step
        × seq_len             # tokens per sequence
        × n_gpus              # total GPUs
        × grad_accum_steps    # accumulation steps
    )

    Example (LLaMA-3 8B training recipe):
        micro_batch_size = 2
        seq_len          = 4096
        n_gpus           = 512
        grad_accum       = 1
        ────────────────────────────────
        effective_batch  = 2 × 4096 × 512 × 1 = 4,194,304 ≈ 4M tokens ✓

For a team with 8 GPUs trying to replicate this:
        micro_batch_size = 4        (limited by A100 memory)
        seq_len          = 4096
        n_gpus           = 8
        grad_accum       = 128      (!) → very slow
        ────────────────────────────────
        effective_batch  = 4 × 4096 × 8 × 128 = 16,777,216 ≈ 16M tokens
        (or reduce accum to 16 for 2M tokens — smaller but more feasible)

The trade-off: achieving the exact same global batch size with fewer GPUs
requires more accumulation steps (more sequential computation, lower GPU
utilisation during the accumulation steps themselves).
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Gradient Accumulation Trade-offs

| Property                  | No accumulation   | With accumulation (K steps)        |
|---------------------------|-------------------|------------------------------------|
| Memory per step           | B × seq × d bytes | M × seq × d bytes  (M = B/K)       |
| Steps to update           | 1                 | K                                  |
| Gradient quality          | Same              | Same (mathematically identical)    |
| Wall-clock per update     | 1 fwd+bwd         | K × fwd+bwd                        |
| DDP comm. with no_sync    | 1 all-reduce      | 1 all-reduce (same as no accum)    |
| DDP comm. without no_sync | 1 all-reduce      | K × all-reduces (avoid this!)      |
| GradScaler interaction    | Standard          | Unscale once after K micro-batches |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Gradient Accumulation — Correctness Proof": {
        "description": "Prove that gradient accumulation produces identical gradients to a full large-batch forward pass — with and without the division trick. Show the common bug and fix.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
GRADIENT ACCUMULATION — CORRECTNESS VERIFICATION
================================================================================

Proves that gradient accumulation is mathematically identical to a
single large-batch forward pass, and demonstrates:
    1. Correct implementation (divide by K before backward)
    2. Common bug (forget to divide — gradients K× too large)
    3. Alternative correct implementation (divide after accumulation)
    4. Numerical verification with max absolute difference

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model_and_data(d: int = 64, vocab: int = 256,
                         B: int = 16, T: int = 32,
                         seed: int = 42):
    """Create identical model and data for fair comparison."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Embedding(vocab, d),
        nn.Linear(d, d),
        nn.GELU(),
        nn.Linear(d, vocab),
    )
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    return model, x, y


def full_batch_gradient(model, x, y):
    """Compute gradient using the full batch B at once."""
    model.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}


def accumulate_correct(model, x, y, K: int):
    """
    CORRECT: divide loss by K inside each micro-batch loop.
    Gradient = mean over all B = K*M samples.
    """
    model.zero_grad()
    B  = x.size(0)
    M  = B // K  # micro-batch size
    for k in range(K):
        micro_x = x[k*M : (k+1)*M]
        micro_y = y[k*M : (k+1)*M]
        logits  = model(micro_x)
        loss    = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   micro_y.view(-1)) / K  # ← divide by K
        loss.backward()  # accumulates into .grad
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}


def accumulate_correct_postdiv(model, x, y, K: int):
    """
    CORRECT (alternative): accumulate sum, divide by K after all micro-batches.
    """
    model.zero_grad()
    B, M = x.size(0), x.size(0) // K
    for k in range(K):
        micro_x = x[k*M : (k+1)*M]
        micro_y = y[k*M : (k+1)*M]
        logits  = model(micro_x)
        loss    = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   micro_y.view(-1))
        loss.backward()  # accumulates sum (not divided by K yet)
    # Divide accumulated gradients by K
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(K)
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}


def accumulate_buggy(model, x, y, K: int):
    """
    BUGGY: forgot to divide by K — gradients are K× too large!
    Equivalent to using learning_rate × K — causes divergence.
    """
    model.zero_grad()
    B, M = x.size(0), x.size(0) // K
    for k in range(K):
        micro_x = x[k*M : (k+1)*M]
        micro_y = y[k*M : (k+1)*M]
        logits  = model(micro_x)
        loss    = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   micro_y.view(-1))
        loss.backward()  # ← no division by K!
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}


def max_abs_diff(grads_a: dict, grads_b: dict) -> float:
    diffs = []
    for name in grads_a:
        if name in grads_b:
            diffs.append((grads_a[name] - grads_b[name]).abs().max().item())
    return max(diffs) if diffs else float("nan")


def grad_ratio(grads_a: dict, grads_b: dict) -> float:
    """Ratio of first parameter's gradient norms."""
    key = next(iter(grads_a))
    na  = grads_a[key].norm().item()
    nb  = grads_b[key].norm().item()
    return na / nb if nb > 0 else float("inf")


if __name__ == "__main__":
    import copy

    D, VOCAB, B, T = 64, 256, 16, 32

    print("=" * 62)
    print("  GRADIENT ACCUMULATION CORRECTNESS VERIFICATION")
    print(f"  Batch B={B}, T={T}, d={D}")
    print("=" * 62)
    print()

    # Test different accumulation factors
    for K in [1, 2, 4, 8]:
        if B % K != 0:
            continue

        model_ref, x, y = make_model_and_data(D, VOCAB, B, T)
        model_c1  = copy.deepcopy(model_ref)
        model_c2  = copy.deepcopy(model_ref)
        model_bug = copy.deepcopy(model_ref)

        g_ref    = full_batch_gradient(model_ref, x, y)
        g_c1     = accumulate_correct(model_c1, x, y, K)
        g_c2     = accumulate_correct_postdiv(model_c2, x, y, K)
        g_bug    = accumulate_buggy(model_bug, x, y, K)

        diff_c1  = max_abs_diff(g_ref, g_c1)
        diff_c2  = max_abs_diff(g_ref, g_c2)
        diff_bug = max_abs_diff(g_ref, g_bug)
        ratio_bug = grad_ratio(g_bug, g_ref)

        print(f"  K={K} (micro_batch M={B//K}):")
        print(f"    Full batch:                     grad norm ref")
        print(f"    Correct (÷K in loop):           max diff = {diff_c1:.2e}  "
              f"{'✓ identical' if diff_c1 < 1e-5 else '✗ DIFFERENT'}")
        print(f"    Correct (÷K after):             max diff = {diff_c2:.2e}  "
              f"{'✓ identical' if diff_c2 < 1e-5 else '✗ DIFFERENT'}")
        print(f"    BUGGY (no division):            max diff = {diff_bug:.2e}  "
              f"⚠️  gradients are {ratio_bug:.1f}× too large!")
        print()

    # Show practical convergence impact of the bug
    print("=" * 62)
    print("  CONVERGENCE IMPACT OF THE DIVISION BUG")
    print("=" * 62)
    print()
    print("  Training a tiny model with K=4 accumulation steps.")
    print("  Comparing correct vs buggy implementation.")
    print()

    LR   = 1e-3
    K    = 4
    N_STEPS = 30

    model_correct, x_tr, y_tr = make_model_and_data(D, VOCAB, B, T, seed=0)
    model_buggy   = copy.deepcopy(model_correct)
    opt_c = torch.optim.AdamW(model_correct.parameters(), lr=LR)
    opt_b = torch.optim.AdamW(model_buggy.parameters(),   lr=LR)

    M = B // K

    print(f"  {'Step':>5}  {'Loss (correct)':>16}  {'Loss (buggy)':>14}")
    print(f"  {'':─>5}  {'':─>16}  {'':─>14}")

    for step in range(N_STEPS):
        # Correct
        opt_c.zero_grad()
        for k in range(K):
            micro_x = x_tr[k*M:(k+1)*M]
            micro_y = y_tr[k*M:(k+1)*M]
            logits  = model_correct(micro_x)
            loss    = F.cross_entropy(logits.view(-1, VOCAB),
                                       micro_y.view(-1)) / K
            loss.backward()
        opt_c.step()
        loss_c = F.cross_entropy(model_correct(x_tr).view(-1,VOCAB), y_tr.view(-1)).item()

        # Buggy
        opt_b.zero_grad()
        for k in range(K):
            micro_x = x_tr[k*M:(k+1)*M]
            micro_y = y_tr[k*M:(k+1)*M]
            logits  = model_buggy(micro_x)
            loss    = F.cross_entropy(logits.view(-1, VOCAB), micro_y.view(-1))
            loss.backward()  # no division!
        opt_b.step()
        loss_b = F.cross_entropy(model_buggy(x_tr).view(-1,VOCAB), y_tr.view(-1)).item()

        if step % 5 == 0 or step < 3:
            flag = " ← diverging!" if loss_b > loss_c * 2 else ""
            print(f"  {step:>5}  {loss_c:>16.4f}  {loss_b:>14.4f}{flag}")

    print()
    print(f"  Summary:")
    print(f"    Correct: converging normally")
    print(f"    Buggy:   {'diverged (loss exploded)' if loss_b > 5 else 'oscillating / poor convergence'}")
    print(f"    Root cause: effective LR for buggy model = {K}× too high")
''',
    },

    "Production Gradient Accumulation Training Loop": {
        "description": "A complete production-quality training loop with gradient accumulation, mixed precision, DDP no_sync, gradient clipping, and step-level metric logging.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PRODUCTION GRADIENT ACCUMULATION TRAINING LOOP
================================================================================

A complete training loop combining:
    1. Gradient accumulation (K micro-batches per optimiser step)
    2. Mixed precision (autocast + GradScaler)
    3. DDP no_sync for communication efficiency
    4. Gradient clipping (after accumulation, not per micro-batch)
    5. Step-level metric logging (loss, grad norm, lr, tokens/sec)
    6. Correct GradScaler interaction with accumulation

This is the template used in production LLM training.
================================================================================
"""

import math
import time
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast


# ── Tiny LM for demo ─────────────────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab: int = 512, d: int = 128,
                 n_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos   = nn.Embedding(max_len, d)
        dec_layer  = nn.TransformerDecoderLayer(
            d_model=d, nhead=4, dim_feedforward=d*4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.body  = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm  = nn.LayerNorm(d)
        self.head  = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.max_len = max_len
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.embed(idx) + self.pos(pos)
        mask = self.causal_mask[:T, :T]
        x    = self.body(x, x, tgt_mask=mask, memory_mask=mask,
                         tgt_is_causal=True, memory_is_causal=True)
        return self.head(self.norm(x))


# ── Production training loop ──────────────────────────────────────────────────

class GradAccumTrainer:
    """
    A production-ready trainer with gradient accumulation.

    Handles:
        - Micro-batch splitting
        - Loss normalisation (divide by K)
        - GradScaler interaction
        - DDP no_sync (if model is DDP-wrapped)
        - Gradient clipping
        - Step-level metrics
    """

    def __init__(self, model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 grad_accum_steps: int = 4,
                 max_grad_norm: float = 1.0,
                 use_amp: bool = False,
                 amp_dtype: torch.dtype = torch.bfloat16,
                 device: str = "cpu"):
        self.model            = model
        self.optimizer        = optimizer
        self.K                = grad_accum_steps
        self.max_grad_norm    = max_grad_norm
        self.device           = device
        self.use_amp          = use_amp
        self.amp_dtype        = amp_dtype

        self.scaler           = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
        self.is_ddp           = isinstance(model, torch.nn.parallel.DistributedDataParallel)

        # Metrics
        self.global_step      = 0
        self.tokens_processed = 0
        self.t_start          = time.perf_counter()

    def _is_ddp_sync_step(self, micro_step: int) -> bool:
        """True only on the last micro-batch (when DDP should all-reduce)."""
        return micro_step == self.K - 1

    def _get_sync_context(self, micro_step: int):
        """Return no_sync context for all but the last micro-batch."""
        if self.is_ddp and not self._is_ddp_sync_step(micro_step):
            return self.model.no_sync()
        return contextlib.nullcontext()

    def train_step(self, x_full: torch.Tensor,
                   y_full: torch.Tensor) -> dict:
        """
        Run one full training step (K micro-batches + 1 optimiser update).

        Returns dict with step metrics.
        """
        x_full = x_full.to(self.device)
        y_full = y_full.to(self.device)

        B = x_full.size(0)
        M = B // self.K   # micro-batch size

        assert B % self.K == 0, \
            f"Batch size {B} must be divisible by grad_accum_steps {self.K}"

        self.optimizer.zero_grad(set_to_none=True)

        total_loss    = 0.0
        total_tokens  = 0

        # ── K micro-batch forward/backward passes ─────────────────────────────
        for k in range(self.K):
            micro_x = x_full[k * M : (k + 1) * M]
            micro_y = y_full[k * M : (k + 1) * M]

            with self._get_sync_context(k):
                with autocast(device_type=self.device,
                               dtype=self.amp_dtype,
                               enabled=self.use_amp):
                    logits = self.model(micro_x)  # (M, T, V)
                    loss   = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        micro_y[:, 1:].reshape(-1),
                    )
                    # Divide by K to get the mean over the full effective batch
                    loss_scaled = loss / self.K

                self.scaler.scale(loss_scaled).backward()

            total_loss   += loss.item()
            total_tokens += micro_x.numel()

        # ── Post-accumulation: unscale, clip, step ────────────────────────────
        self.scaler.unscale_(self.optimizer)   # unscale ONCE, after all micro-batches

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        ).item()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # ── Metrics ──────────────────────────────────────────────────────────
        self.global_step      += 1
        self.tokens_processed += total_tokens
        elapsed = time.perf_counter() - self.t_start

        return {
            "step":        self.global_step,
            "loss":        total_loss / self.K,   # mean loss over all micro-batches
            "grad_norm":   grad_norm,
            "tokens_sec":  self.tokens_processed / elapsed,
            "scale":       self.scaler.get_scale() if self.scaler.is_enabled() else 1.0,
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
    VOCAB    = 512
    D        = 128
    T        = 32
    K        = 4      # accumulation steps
    B_FULL   = 16     # effective batch size (M = B_FULL // K = 4 per micro-batch)
    LR       = 3e-4
    N_STEPS  = 50

    print(f"Device: {DEVICE}")
    print(f"Gradient accumulation: K={K} micro-batches of M={B_FULL//K}")
    print(f"Effective batch: {B_FULL} sequences × {T} tokens = "
          f"{B_FULL*T:,} tokens/step")
    print()

    model   = TinyLM(VOCAB, D, n_layers=2, max_len=T).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    trainer = GradAccumTrainer(
        model=model, optimizer=opt,
        grad_accum_steps=K,
        max_grad_norm=1.0,
        use_amp=False,   # set True if CUDA available and want to demo AMP
        device=DEVICE,
    )

    x_data = torch.randint(0, VOCAB, (B_FULL, T))
    y_data = torch.randint(0, VOCAB, (B_FULL, T))

    print("=" * 60)
    print(f"  {'Step':>5}  {'Loss':>10}  {'Grad norm':>12}  {'Tok/s':>10}")
    print("=" * 60)

    for step in range(N_STEPS):
        # Resample data each step (in real training, from DataLoader)
        x = torch.randint(0, VOCAB, (B_FULL, T))
        y = torch.randint(0, VOCAB, (B_FULL, T))

        metrics = trainer.train_step(x, y)

        if step % 10 == 0 or step < 3:
            print(f"  {metrics['step']:>5}  {metrics['loss']:>10.4f}  "
                  f"{metrics['grad_norm']:>12.4f}  {metrics['tokens_sec']:>10.0f}")

    print()
    print("Training complete!")
    print()
    print("=" * 60)
    print("  KEY IMPLEMENTATION NOTES")
    print("=" * 60)
    notes = [
        ("loss / K",           "Divide loss by K before backward in each micro-step"),
        ("zero_grad once",     "Call zero_grad() once BEFORE the K micro-batch loop"),
        ("unscale_ once",      "Call scaler.unscale_() AFTER the K micro-batch loop"),
        ("clip after accum",   "Gradient clipping happens AFTER all micro-batches"),
        ("step once",          "optimizer.step() called ONCE after K micro-batches"),
        ("model.no_sync()",    "Defer DDP all-reduce to last micro-batch only"),
        ("tokens_per_step",    "= micro_batch × seq_len × n_gpus × K"),
        ("LR scaling",         "Scale lr ∝ √(effective_batch / base_batch)"),
    ]
    for key, note in notes:
        print(f"  ✓  {key:<22}  {note}")
''',
    },

    "Effective Batch Size Calculator and GPU Scaling": {
        "description": "Calculate how gradient accumulation, GPU count, and micro-batch size combine into an effective batch size, and find the optimal accumulation factor for a target.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
EFFECTIVE BATCH SIZE CALCULATOR AND GPU SCALING
================================================================================

Answers key practical questions:
    1. What is my effective batch size given my GPU setup?
    2. How many gradient accumulation steps do I need to hit a target?
    3. How should my learning rate scale with my effective batch size?
    4. What is the throughput vs batch size trade-off?

================================================================================
"""

import math


def effective_batch_tokens(micro_batch: int, seq_len: int,
                            n_gpus: int, grad_accum: int) -> int:
    return micro_batch * seq_len * n_gpus * grad_accum


def required_grad_accum(target_tokens: int, micro_batch: int,
                         seq_len: int, n_gpus: int) -> int:
    """Compute grad_accum needed to hit a target effective batch size."""
    per_gpu_per_step = micro_batch * seq_len
    total_per_step   = per_gpu_per_step * n_gpus
    return math.ceil(target_tokens / total_per_step)


def lr_for_batch(lr_base: float, batch_base_tokens: int,
                  batch_new_tokens: int,
                  rule: str = "sqrt") -> float:
    """Compute scaled learning rate for a new effective batch size."""
    ratio = batch_new_tokens / batch_base_tokens
    if rule == "linear":
        return lr_base * ratio
    elif rule == "sqrt":
        return lr_base * math.sqrt(ratio)
    else:
        return lr_base


def steps_per_second(micro_batch: int, seq_len: int, n_gpus: int,
                      grad_accum: int, tokens_per_sec_per_gpu: float) -> float:
    """Effective optimiser steps per second."""
    tps_total     = tokens_per_sec_per_gpu * n_gpus
    tokens_per_step = effective_batch_tokens(micro_batch, seq_len, n_gpus, grad_accum)
    return tps_total / tokens_per_step


def fmt_k(n: int) -> str:
    if n >= 1_000_000: return f"{n/1e6:.1f}M"
    if n >= 1_000:     return f"{n/1e3:.0f}K"
    return str(n)


if __name__ == "__main__":
    # ── 1. Effective batch size for common setups ────────────────────────────
    print("=" * 68)
    print("  EFFECTIVE BATCH SIZE ACROSS COMMON TRAINING SETUPS")
    print("  (seq_len=4096, target=4M tokens/step)")
    print("=" * 68)
    print()

    TARGET_TOKENS = 4_000_000
    SEQ_LEN       = 4096

    setups = [
        # (desc,          micro_batch, n_gpus)
        ("1× A100 80GB",  4,    1),
        ("4× A100 80GB",  4,    4),
        ("8× A100 80GB",  4,    8),
        ("16× A100 80GB", 4,   16),
        ("64× A100 80GB", 4,   64),
        ("512×A100 80GB", 4,  512),
    ]

    print(f"  {'Setup':<20} {'Micro B':>8}  {'Max K=1':>10}  {'K needed':>10}  {'Feasible?':>12}")
    print(f"  {'':─<20} {'':─>8}  {'':─>10}  {'':─>10}  {'':─>12}")

    for desc, mb, n_gpu in setups:
        native_tokens = effective_batch_tokens(mb, SEQ_LEN, n_gpu, 1)
        K_needed      = required_grad_accum(TARGET_TOKENS, mb, SEQ_LEN, n_gpu)
        feasible      = "✓ K=1" if native_tokens >= TARGET_TOKENS else f"K={K_needed}"
        print(f"  {desc:<20} {mb:>8}  {fmt_k(native_tokens):>10}  {K_needed:>10}  {feasible:>12}")

    # ── 2. LR scaling with batch size ────────────────────────────────────────
    print()
    print("=" * 68)
    print("  LR SCALING WITH BATCH SIZE (base: 4M tokens at lr=3e-4)")
    print("=" * 68)
    print()

    BASE_TOKENS  = 4_000_000
    BASE_LR      = 3e-4

    batch_sizes = [500_000, 1_000_000, 2_000_000, 4_000_000,
                   8_000_000, 16_000_000]

    print(f"  {'Tokens/step':>12}  {'Linear LR':>12}  {'Sqrt LR':>12}  {'Ratio':>8}")
    print(f"  {'':─>12}  {'':─>12}  {'':─>12}  {'':─>8}")

    for tokens in batch_sizes:
        lr_lin = lr_for_batch(BASE_LR, BASE_TOKENS, tokens, "linear")
        lr_sqt = lr_for_batch(BASE_LR, BASE_TOKENS, tokens, "sqrt")
        ratio  = tokens / BASE_TOKENS
        marker = " ← base" if tokens == BASE_TOKENS else ""
        print(f"  {fmt_k(tokens):>12}  {lr_lin:>12.2e}  {lr_sqt:>12.2e}  "
              f"{ratio:>7.1f}×{marker}")

    print()
    print("  Note: Linear scaling can be too aggressive for LLMs.")
    print("  Sqrt scaling is more conservative and usually more stable.")
    print("  Always pair LR scaling with an appropriate warmup schedule.")

    # ── 3. Throughput vs accumulation ────────────────────────────────────────
    print()
    print("=" * 68)
    print("  THROUGHPUT vs GRAD ACCUMULATION STEPS")
    print("  (4× A100, micro_batch=4, seq_len=2048, 380 tok/s/GPU)")
    print("=" * 68)
    print()

    N_GPUS = 4
    MICRO  = 4
    SEQ    = 2048
    TPS    = 380   # tokens/sec per GPU

    print(f"  {'K (accum)':>10}  {'Eff. batch':>12}  {'Steps/sec':>12}  "
          f"{'Steps/hr':>12}  {'Tokens/hr':>12}")
    print(f"  {'':─>10}  {'':─>12}  {'':─>12}  {'':─>12}  {'':─>12}")

    for K in [1, 2, 4, 8, 16, 32, 64]:
        eff_tokens = effective_batch_tokens(MICRO, SEQ, N_GPUS, K)
        sps        = steps_per_second(MICRO, SEQ, N_GPUS, K, TPS)
        sph        = sps * 3600
        tph        = TPS * N_GPUS * 3600
        print(f"  {K:>10}  {fmt_k(eff_tokens):>12}  {sps:>12.3f}  "
              f"{sph:>12,.0f}  {fmt_k(int(tph)):>12}")

    print()
    print("  Key insight: larger K means fewer optimiser steps per second,")
    print("  but the SAME total throughput (tokens/hr doesn't change with K).")
    print("  Grad accumulation doesn't slow training — it just changes the")
    print("  frequency of weight updates, not the rate of data consumption.")
    print()

    # ── 4. Total training time ────────────────────────────────────────────────
    print("=" * 68)
    print("  TOTAL TRAINING TIME ESTIMATES")
    print("  (Target: 1T tokens, 4× A100, 380 tok/s/GPU)")
    print("=" * 68)
    print()

    TOTAL_TOKENS = 1_000_000_000_000   # 1T tokens
    TPS_TOTAL    = TPS * N_GPUS        # total tokens/sec across all GPUs

    time_seconds   = TOTAL_TOKENS / TPS_TOTAL
    time_hours     = time_seconds / 3600
    time_days      = time_hours   / 24

    print(f"  Total tokens:        {TOTAL_TOKENS/1e12:.0f}T")
    print(f"  Effective throughput:{TPS_TOTAL:,} tokens/sec ({N_GPUS} GPUs × {TPS} tok/s)")
    print(f"  Training time:       {time_days:.1f} days  ({time_hours:.0f} hours)")
    print()
    print("  Gradient accumulation does NOT change training time.")
    print("  It only affects memory usage and batch size, not throughput.")
    print()
    print("  To train faster: add more GPUs (scales linearly with perfect scaling)")
    print("  To use larger batch: increase grad_accum_steps")
    print("  To reduce memory: decrease micro_batch_size + increase grad_accum")
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
    #     from llm_training.visuals.gradient_accumulation import (
    #         GRADACCUM_VISUAL_HTML,
    #         GRADACCUM_VISUAL_HEIGHT,
    #     )
    #     visual_html   = GRADACCUM_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = GRADACCUM_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[17_gradient_accumulation.py] Could not load visual: {e}",
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