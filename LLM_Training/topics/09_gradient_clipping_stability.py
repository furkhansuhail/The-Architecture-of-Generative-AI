"""
Gradient Clipping and Training Stability
==========================================

Gradient clipping is a one-line operation that prevents the single most
common cause of LLM training failure: exploding gradients. Understanding
why gradients explode, how clipping stops them, what the right threshold
is, and how to use the gradient norm as a training health signal will save
you from countless mysterious divergences.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Gradient Clipping and Training Stability"
DISPLAY_NAME = "09 · Gradient Clipping"
ICON         = "✂️"
SUBTITLE     = "Preventing Exploding Gradients"


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

### What Are Exploding Gradients?

During backpropagation, gradients are computed by chaining derivatives from
the loss back through each layer. In a deep network, this chain involves
repeated multiplication:

    ∂L/∂W₁ = ∂L/∂a_N × ∂a_N/∂a_{N-1} × … × ∂a_2/∂a_1 × ∂a_1/∂W₁

If the individual Jacobians have spectral norm > 1, their product grows
exponentially with depth — a gradient that starts as 0.1 can become 10⁶
by the time it reaches the first layer.

**What triggers gradient explosions in LLMs?**

    1.  **Anomalous training batches:** A batch that happens to contain
        unusually long sequences, rare tokens, or adversarial patterns can
        produce atypically large activations and correspondingly large
        gradients.

    2.  **Loss spikes:** A sudden jump in loss (caused by a hard batch,
        a data quality issue, or a learning rate that is momentarily too
        high) creates a large gradient signal that propagates explosively.

    3.  **Early training instability:** Before Adam's second moment v_t is
        well-estimated (first few hundred steps), the adaptive scaling can
        fail for certain parameters, producing large effective updates.

    4.  **Numerical precision issues:** In fp16/bf16 training, gradients
        can overflow to inf or NaN, which then propagate and corrupt all
        parameters.

Without protection, a single explosive gradient update rewrites the model
weights into garbage, and training never recovers.


### Global Gradient Norm Clipping

The standard solution is **global gradient norm clipping**, which rescales
the entire gradient vector if its L2 norm exceeds a threshold:

    g = concatenate(∂L/∂W₁, ∂L/∂W₂, …, ∂L/∂W_n)   ← all gradients flattened
    ‖g‖ = sqrt(sum of squared values)                ← global L2 norm

    if ‖g‖ > max_norm:
        g ← g × (max_norm / ‖g‖)                    ← rescale to max_norm

    (equivalently, each parameter's gradient is scaled by the same factor)

**Key properties:**

    •   The *direction* of the gradient is preserved — only the magnitude
        is capped. The update still points toward lower loss.
    •   All parameters are scaled by the *same factor* — the relative
        magnitude of different gradients is unchanged.
    •   This is a hard ceiling, not a soft penalty: gradients up to max_norm
        are untouched; only large gradient events are clipped.
    •   Global (not per-parameter): a spike in one layer's gradient affects
        the clipping of all layers, which preserves gradient direction.


    **Diagram 1 — Global vs Per-Parameter Clipping:**

    GLOBAL vs PER-PARAMETER GRADIENT CLIPPING
    ════════════════════════════════════════════════════════════════

    Suppose max_norm = 1.0.

    Layer gradients (L2 norms):
        Layer 1:  ‖g₁‖ = 0.3    (small, normal)
        Layer 2:  ‖g₂‖ = 0.4    (small, normal)
        Layer 3:  ‖g₃‖ = 9.0    (LARGE — spike in this layer!)

    Global clipping:
        ‖g_total‖ = sqrt(0.3² + 0.4² + 9.0²) ≈ 9.03
        Scale factor = 1.0 / 9.03 ≈ 0.111
        g₁_clipped = g₁ × 0.111  → 0.033   (scaled down, direction preserved)
        g₂_clipped = g₂ × 0.111  → 0.044   (scaled down, direction preserved)
        g₃_clipped = g₃ × 0.111  → 1.0     (capped at max_norm contribution)

    Per-parameter clipping (NOT used in practice):
        g₁_clipped = clip(g₁, -1.0, 1.0)   (clips elementwise — distorts direction!)
        g₃_clipped = clip(g₃, -1.0, 1.0)   (layer 3 direction destroyed ✗)

    Global clipping preserves the direction of the full gradient vector. ✓


### Choosing max_norm

    Value          When to use
    ─────────────────────────────────────────────────────────────────────
    0.5            Conservative; useful if training is frequently unstable
    1.0            Standard for most LLM pre-training (LLaMA, GPT-3, etc.)
    5.0            Looser; common for fine-tuning or small models
    ∞ (disabled)   Baseline / ablation only — not recommended for LLMs
    ─────────────────────────────────────────────────────────────────────

**The key rule:** The gradient norm should be clipped only on large outlier
events, not on every step. If your gradient norm is consistently > max_norm,
the clipping threshold is too low — or your learning rate is too high.

A healthy training run has gradient norms well below max_norm on most steps,
with occasional spikes that get clipped. If clipping fires on every step,
something is wrong.


### The Gradient Norm as a Training Health Signal

Logging the pre-clip gradient norm at every step provides invaluable
diagnostic information:

    Pattern                        Meaning
    ─────────────────────────────────────────────────────────────────────
    Stable, slowly decreasing      Healthy training ✓
    Sudden spike then recovery     Hard batch; clipping handled it ✓
    Continuous high norm           LR too high, or unstable init ⚠️
    Sudden spike + loss spike      Real instability — check data pipeline ⚠️
    Norm = NaN                     Overflow; check mixed precision settings ✗
    Norm = 0.0 every step          Gradient vanishing — check activations ✗
    ─────────────────────────────────────────────────────────────────────

The relationship between loss spikes and gradient norm spikes is a key
diagnostic:

    •   Loss spike + gradient spike: a hard batch caused both; clipping
        absorbed the gradient; loss usually recovers within 50-200 steps.
    •   Gradient spike WITHOUT loss spike: unusual; may indicate a numerical
        precision issue or a pathological batch.
    •   Loss spike WITHOUT gradient spike: clipping was too aggressive and
        prevented the model from responding to genuine signal.


    **Diagram 2 — Healthy vs Unhealthy Gradient Norm Profiles:**

    GRADIENT NORM PROFILES ACROSS TRAINING
    ════════════════════════════════════════════════════════════════

    HEALTHY:  Norm declines as training stabilises, with occasional spikes
    ──────────────────────────────────────────────────────────────────────
    ‖g‖
     ▲
     │ ██
     │  ████  █   █
     │      █████  ██  █     █
     │           ████████ █████    ████
     │                         ████    ████████████
     └──────────────────────────────────────────────► steps
       Start               Middle             End
       (high variance)     (stabilising)      (low, steady)

    UNHEALTHY: Constant high norm — LR too large or training diverging
    ──────────────────────────────────────────────────────────────────────
    ‖g‖
     ▲
     │ ████████████████████████████████████████████
     └──────────────────────────────────────────────► steps
       ⚠️  norm constantly at max_norm → clipping on every step → bad

    CATASTROPHIC: Norm → ∞ (NaN) — training failed
    ──────────────────────────────────────────────────────────────────────
    ‖g‖
     ▲
     │                             ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄∞
     │                    ▄▄▄▄▄▄▄▄
     │ ████████████████████
     └──────────────────────────────────────────────► steps
       ✗  divergence — training cannot continue


### Where Clipping Fits in the Training Step

Gradient clipping must happen AFTER `.backward()` (gradients are computed)
and BEFORE `.step()` (gradients are applied):

    loss.backward()                                   ← compute gradients
    scaler.unscale_(optimizer)                        ← unscale (if fp16)
    torch.nn.utils.clip_grad_norm_(params, max_norm)  ← CLIP HERE
    scaler.step(optimizer)                            ← apply (already clipped)
    scaler.update()

**Mixed precision note:** With fp16 and GradScaler, the gradients are stored
in scaled form (multiplied by a large factor to prevent underflow). They must
be unscaled before clipping, otherwise you are comparing the scaled norm to
max_norm, which is meaningless. The `scaler.unscale_()` call handles this.


### Value Clipping vs Norm Clipping

Two different approaches exist:

**Value clipping (element-wise):**

    g ← clip(g, -threshold, +threshold)   element by element

    •   Changes the direction of the gradient (each element independently).
    •   Simple but crude — distorts the gradient direction.
    •   Rarely used for LLMs.

**Norm clipping (global L2, described above):**

    g ← g × min(1, max_norm / ‖g‖)

    •   Preserves gradient direction.
    •   Scales the entire gradient uniformly.
    •   Standard for LLM training.

PyTorch's `torch.nn.utils.clip_grad_norm_()` implements norm clipping.
`torch.nn.utils.clip_grad_value_()` implements value clipping (avoid for LLMs).


### Loss Spikes — Causes and Recovery

A **loss spike** is a sudden jump in training loss (typically 2–10× the
running mean) followed by gradual recovery. They are common in LLM training
and usually benign if they recover within a few hundred steps.

**Common causes:**
    1.  A hard training batch (long sequences, unusual content, rare tokens)
    2.  A data quality issue (corrupted document, encoding error)
    3.  LR momentarily too high (e.g. at LR schedule peak)
    4.  Gradient accumulation sync error (DDP desync)

**Recovery strategies:**
    •   If recovery is rapid (50–200 steps): do nothing — clipping handled it.
    •   If recovery is slow (> 500 steps): inspect the data batch that caused
        the spike; consider reducing lr or increasing max_norm.
    •   If no recovery: checkpoint rollback. Save checkpoints every N steps
        so you can resume from before the spike.
    •   If persistent spikes: reduce lr by 10–30%, check data pipeline for
        corrupted batches, verify mixed precision settings.

**Checkpoint-and-rollback is the ultimate safety net.** Most production LLM
training systems save checkpoints every 1,000–5,000 steps precisely to handle
irrecoverable spikes.


### z-Loss: A Complementary Stability Technique

PaLM (Chowdhery et al., 2022) introduced **z-loss**, an auxiliary loss term
that penalises large logit values:

    L_z = ε × log²( sum_i exp(logit_i) )

This prevents the softmax denominator from growing unboundedly, which can
cause the cross-entropy loss to behave erratically. z-loss is inexpensive
(adds one scalar to the loss) and was found to significantly reduce loss
spike frequency in large models.


### Gradient Clipping in Distributed Training

In distributed training (DDP, FSDP, tensor parallelism), gradient clipping
requires careful coordination:

    •   **DDP (data parallel):** Gradients are already averaged across
        GPUs by the time `.backward()` completes. Call `clip_grad_norm_()`
        once on any GPU — the norm is the same everywhere.

    •   **FSDP (fully sharded):** Each GPU holds only a shard of each
        parameter's gradient. The global norm requires an all-reduce across
        GPUs. FSDP provides `clip_grad_norm_()` that does this correctly.

    •   **Tensor/pipeline parallel:** Each GPU may hold only part of the
        gradient for a given layer. The gradient norm must be computed and
        clipped across all devices that own parts of the same parameter.

Getting clipping wrong in distributed training silently produces incorrect
updates — each GPU clips independently to the wrong norm.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Gradient Clipping Methods

| Method              | Preserves direction? | Granularity  | PyTorch API                      | Use for LLMs? |
|---------------------|----------------------|--------------|----------------------------------|---------------|
| Global norm clip    | Yes ✓                | All params   | clip_grad_norm_(params, max_norm)| Yes — standard|
| Per-layer norm clip | Yes ✓                | Per layer    | manual loop                      | Rare          |
| Value clip          | No ✗                 | Per element  | clip_grad_value_(params, val)    | Avoid         |
| Adaptive clip       | Yes ✓                | All params   | custom (based on grad history)   | Research      |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Gradient Clipping — From Scratch": {
        "description": "Implement global norm clipping from scratch, verify against PyTorch, and demonstrate how a gradient spike is contained without destroying the update direction.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
GRADIENT CLIPPING — FROM SCRATCH
================================================================================

Implements global gradient norm clipping from scratch and demonstrates:
    1. How the global norm is computed
    2. How the scale factor is applied uniformly
    3. That gradient direction is preserved
    4. Comparison with PyTorch's built-in

Also shows what happens without clipping during a simulated gradient spike.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── From-scratch implementation ───────────────────────────────────────────────

def clip_grad_norm_scratch(parameters, max_norm: float, norm_type: float = 2.0):
    """
    Clip the gradient norm of an iterable of parameters.
    Returns the pre-clip global gradient norm.

    This is functionally identical to torch.nn.utils.clip_grad_norm_.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return torch.tensor(0.0)

    # Compute global L2 norm across all parameter gradients
    total_norm_sq = sum(
        p.grad.detach().norm(norm_type).item() ** norm_type
        for p in params_with_grad
    )
    total_norm = total_norm_sq ** (1.0 / norm_type)

    # Compute clip coefficient: min(1, max_norm / total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)   # don't amplify gradients

    # Scale all gradients by the same factor
    for p in params_with_grad:
        p.grad.detach().mul_(clip_coef)

    return torch.tensor(total_norm)


# ── Verification ──────────────────────────────────────────────────────────────

def verify_clipping():
    torch.manual_seed(0)
    model_a = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
    model_b = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))

    # Copy identical weights
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        pb.data.copy_(pa.data)

    x = torch.randn(16, 8)
    y = torch.randn(16, 1)

    # Identical forward + backward
    for model in (model_a, model_b):
        model.zero_grad()
        F.mse_loss(model(x), y).backward()

    max_norm = 0.5

    # Our implementation
    norm_a = clip_grad_norm_scratch(model_a.parameters(), max_norm)
    # PyTorch implementation
    norm_b = torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm)

    print("=" * 60)
    print("  VERIFICATION: scratch vs PyTorch")
    print("=" * 60)
    print(f"  Pre-clip norm (scratch):  {norm_a.item():.6f}")
    print(f"  Pre-clip norm (PyTorch):  {norm_b.item():.6f}")
    print(f"  Norms match: {abs(norm_a.item() - norm_b.item()) < 1e-4}")

    # Compare final gradient values
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        if pa.grad is not None:
            diff = (pa.grad - pb.grad).abs().max().item()
            if diff > 1e-5:
                print(f"  ⚠️  Gradient mismatch! max diff = {diff:.2e}")
                break
    else:
        print(f"  Gradient values match ✓")


# ── Direction preservation demo ───────────────────────────────────────────────

def direction_preservation_demo():
    """
    Show that global norm clipping preserves the gradient direction.
    """
    print()
    print("=" * 60)
    print("  DIRECTION PRESERVATION DEMO")
    print("=" * 60)
    print()

    torch.manual_seed(1)
    model = nn.Linear(4, 4, bias=False)
    x     = torch.randn(8, 4)
    y     = torch.randn(8, 4)

    F.mse_loss(model(x), y).backward()
    g_before = model.weight.grad.clone()
    norm_before = g_before.norm().item()

    max_norm = 0.1   # aggressive clip — force clipping
    clip_grad_norm_scratch(list(model.parameters()), max_norm)
    g_after = model.weight.grad.clone()
    norm_after = g_after.norm().item()

    # Cosine similarity between before and after
    cos_sim = F.cosine_similarity(
        g_before.flatten().unsqueeze(0),
        g_after.flatten().unsqueeze(0)
    ).item()

    print(f"  Gradient norm before clip: {norm_before:.4f}")
    print(f"  Gradient norm after  clip: {norm_after:.4f}  (max_norm={max_norm})")
    print(f"  Scale factor applied:      {norm_after/norm_before:.4f}")
    print()
    print(f"  Cosine similarity (before vs after): {cos_sim:.6f}")
    print(f"  Direction preserved: {'✓ YES' if abs(cos_sim - 1.0) < 1e-4 else '✗ NO'}")
    print()
    print("  The gradient vector is scaled down uniformly —")
    print("  every element is multiplied by the same factor.")
    print("  The direction (angle in weight space) is unchanged.")


# ── Gradient spike simulation ─────────────────────────────────────────────────

def gradient_spike_simulation():
    """
    Simulate a training run where a hard batch causes a gradient spike.
    Compare: with and without clipping.
    """
    print()
    print("=" * 60)
    print("  GRADIENT SPIKE SIMULATION")
    print("=" * 60)
    print()

    torch.manual_seed(42)
    MAX_NORM = 1.0

    def make_model():
        m = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
        return m

    def make_opt(m):
        return torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)

    model_clipped   = make_model()
    model_unclipped = make_model()

    # Copy same init
    for pc, pu in zip(model_clipped.parameters(),
                      model_unclipped.parameters()):
        pu.data.copy_(pc.data)

    opt_c = make_opt(model_clipped)
    opt_u = make_opt(model_unclipped)

    N_STEPS   = 80
    SPIKE_AT  = 40   # inject a hard batch at step 40

    x_normal  = torch.randn(64, 16)
    y_normal  = torch.randn(64, 1)
    # "Hard batch": inputs with 50× the normal scale → ~2500× gradient magnitude
    x_hard    = torch.randn(64, 16) * 50.0
    y_hard    = torch.randn(64, 1)  * 50.0

    print(f"  {'Step':>5}  {'Loss (clipped)':>16}  {'Loss (no clip)':>16}  "
          f"{'Grad norm (clip)':>18}  {'Grad norm (no)':>16}")
    print(f"  {'':─>5}  {'':─>16}  {'':─>16}  {'':─>18}  {'':─>16}")

    for step in range(N_STEPS):
        x = x_hard if step == SPIKE_AT else x_normal
        y = y_hard if step == SPIKE_AT else y_normal

        for model, opt, clip in (
            (model_clipped,   opt_c, True),
            (model_unclipped, opt_u, False),
        ):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()

        if step % 5 == 0 or step == SPIKE_AT or step == SPIKE_AT + 1:
            # Compute norms for reporting (recompute before clip for display)
            model_clipped.zero_grad()
            model_unclipped.zero_grad()
            for m in (model_clipped, model_unclipped):
                F.mse_loss(m(x_normal), y_normal).backward()
            norm_c = sum(p.grad.norm().item()**2
                         for p in model_clipped.parameters()
                         if p.grad is not None) ** 0.5
            norm_u = sum(p.grad.norm().item()**2
                         for p in model_unclipped.parameters()
                         if p.grad is not None) ** 0.5
            model_clipped.zero_grad(); model_unclipped.zero_grad()

            loss_c = F.mse_loss(model_clipped(x_normal),   y_normal).item()
            loss_u = F.mse_loss(model_unclipped(x_normal), y_normal).item()

            flag = " ← SPIKE" if step == SPIKE_AT else ""
            print(f"  {step:>5}  {loss_c:>16.4f}  {loss_u:>16.4f}  "
                  f"{norm_c:>18.4f}  {norm_u:>16.4f}{flag}")

    print()
    print(f"  Gradient clipping absorbed the spike at step {SPIKE_AT}.")
    print(f"  The unclipped model may diverge or take much longer to recover.")


if __name__ == "__main__":
    verify_clipping()
    direction_preservation_demo()
    gradient_spike_simulation()
''',
    },

    "Gradient Norm Monitoring and Diagnostics": {
        "description": "Build a gradient norm monitor that tracks per-layer and global norms across training, flags anomalies, and visualises the health of gradient flow.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
GRADIENT NORM MONITOR
================================================================================

A production-style gradient norm monitor that:
    1. Tracks global and per-layer gradient norms at each step
    2. Detects anomalies (spikes, vanishing, NaN)
    3. Generates a training health report
    4. Shows the gradient norm profile as an ASCII sparkline

This is the kind of monitoring used in real LLM training dashboards.
================================================================================
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GradStats:
    step:           int
    global_norm:    float
    was_clipped:    bool
    clip_ratio:     float          # global_norm / max_norm
    per_layer:      dict           # layer_name → norm


class GradNormMonitor:
    """
    Tracks gradient norms across training and provides diagnostic signals.

    Usage:
        monitor = GradNormMonitor(max_norm=1.0, window=100)

        # In training loop, after backward() before optimizer.step():
        stats = monitor.record(model, step)
        if monitor.is_anomalous():
            print("WARNING:", monitor.anomaly_description())
    """

    def __init__(self, max_norm: float = 1.0, window: int = 100):
        self.max_norm  = max_norm
        self.window    = window
        self.history:  list[GradStats] = []
        self.recent:   deque  = deque(maxlen=window)

    def record(self, model: nn.Module, step: int) -> GradStats:
        """
        Record gradient norms for the current step.
        Does NOT clip — call clip_grad_norm_ separately.
        """
        per_layer  = {}
        total_sq   = 0.0
        has_nan    = False

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g_norm = param.grad.detach().norm(2).item()
            if math.isnan(g_norm) or math.isinf(g_norm):
                has_nan = True
                g_norm  = float("nan")
            per_layer[name] = g_norm
            if not math.isnan(g_norm):
                total_sq += g_norm ** 2

        global_norm = math.sqrt(total_sq) if not has_nan else float("nan")
        was_clipped = (not has_nan) and (global_norm > self.max_norm)
        clip_ratio  = global_norm / self.max_norm if not has_nan else float("nan")

        stats = GradStats(step, global_norm, was_clipped, clip_ratio, per_layer)
        self.history.append(stats)
        if not has_nan:
            self.recent.append(global_norm)
        return stats

    def is_anomalous(self) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        if math.isnan(last.global_norm):
            return True
        if len(self.recent) >= 10:
            mean = sum(self.recent) / len(self.recent)
            if last.global_norm > 5 * mean:
                return True
        return False

    def anomaly_description(self) -> Optional[str]:
        if not self.history:
            return None
        last = self.history[-1]
        if math.isnan(last.global_norm):
            return "NaN gradient norm — overflow in fp16 or numerical instability"
        if len(self.recent) >= 10:
            mean = sum(self.recent) / len(self.recent)
            if last.global_norm > 5 * mean:
                return (f"Gradient spike: {last.global_norm:.2f} vs "
                        f"recent mean {mean:.2f} ({last.global_norm/mean:.1f}×)")
        return None

    def running_mean(self) -> float:
        if not self.recent:
            return 0.0
        return sum(self.recent) / len(self.recent)

    def clip_frequency(self) -> float:
        """Fraction of steps where clipping fired (last window steps)."""
        if not self.history:
            return 0.0
        recent_stats = self.history[-self.window:]
        clipped = sum(1 for s in recent_stats if s.was_clipped)
        return clipped / len(recent_stats)

    def sparkline(self, last_n: int = 60) -> str:
        """ASCII sparkline of recent gradient norms."""
        bars  = " ▁▂▃▄▅▆▇█"
        norms = [s.global_norm for s in self.history[-last_n:]
                 if not math.isnan(s.global_norm)]
        if not norms:
            return ""
        lo, hi = min(norms), max(norms)
        span   = hi - lo if hi > lo else 1.0
        result = ""
        for v in norms:
            idx     = int((v - lo) / span * (len(bars) - 1))
            idx     = max(0, min(len(bars) - 1, idx))
            result += bars[idx]
        return result

    def health_report(self) -> str:
        if not self.history:
            return "No data recorded yet."
        lines   = []
        n_steps = len(self.history)
        mean_n  = self.running_mean()
        clip_f  = self.clip_frequency()
        last    = self.history[-1]

        lines.append(f"  Steps recorded:      {n_steps}")
        lines.append(f"  Current norm:        {last.global_norm:.4f}")
        lines.append(f"  Recent mean norm:    {mean_n:.4f}")
        lines.append(f"  Clip frequency:      {clip_f*100:.1f}%  "
                     f"({'⚠️ very high' if clip_f > 0.5 else '✓ normal' if clip_f < 0.1 else 'moderate'})")
        lines.append(f"  max_norm threshold:  {self.max_norm}")
        lines.append(f"  Norm sparkline:")
        lines.append(f"    [{self.sparkline()}]")

        if self.is_anomalous():
            lines.append(f"  ⚠️  ANOMALY: {self.anomaly_description()}")
        else:
            lines.append(f"  ✓  No anomalies detected")

        return "\n".join(lines)


# ── Demo: simulate a training run with a spike ────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(16, 64),  nn.ReLU(),
        nn.Linear(64, 64),  nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-3)
    monitor = GradNormMonitor(max_norm=1.0, window=50)

    x = torch.randn(64, 16)
    y = torch.randn(64, 1)

    print("=" * 65)
    print("  TRAINING WITH GRADIENT NORM MONITORING")
    print("=" * 65)
    print()
    print(f"  {'Step':>5}  {'Grad Norm':>12}  {'Clipped?':>9}  {'Status':>20}")
    print(f"  {'':─>5}  {'':─>12}  {'':─>9}  {'':─>20}")

    SPIKE_STEPS = {30, 31}   # inject artificial spike

    for step in range(1, 101):
        opt.zero_grad(set_to_none=True)

        if step in SPIKE_STEPS:
            # Simulate hard batch: scale the loss to create a gradient spike
            loss = F.mse_loss(model(x * 20), y * 20)
        else:
            loss = F.mse_loss(model(x), y)

        loss.backward()

        # Record BEFORE clipping
        stats = monitor.record(model, step)

        # Now clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), monitor.max_norm)
        opt.step()

        if step % 10 == 0 or step in SPIKE_STEPS or step == SPIKE_STEPS.copy().pop() + 1:
            anomaly = monitor.anomaly_description()
            status  = f"⚠️ {anomaly[:15]}.." if anomaly else "✓ healthy"
            clipped_str = "YES ✂️" if stats.was_clipped else "no"
            print(f"  {step:>5}  {stats.global_norm:>12.4f}  {clipped_str:>9}  {status:>20}")

    print()
    print("=" * 65)
    print("  HEALTH REPORT")
    print("=" * 65)
    print(monitor.health_report())

    print()
    print("=" * 65)
    print("  PER-LAYER GRADIENT NORMS (last step)")
    print("=" * 65)
    last_stats = monitor.history[-1]
    for layer_name, norm in sorted(last_stats.per_layer.items()):
        bar = "█" * int(min(norm / 0.1, 30))
        print(f"  {layer_name:<35}  {norm:.4f}  {bar}")
''',
    },

    "Loss Spike Anatomy and Recovery": {
        "description": "Simulate a loss spike, trace its cause in the gradient, and demonstrate checkpoint-rollback as the recovery strategy.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LOSS SPIKE ANATOMY AND RECOVERY
================================================================================

Loss spikes are common in LLM training. This demo:
    1. Simulates a realistic training run with an injected spike
    2. Shows the relationship between gradient norm and loss
    3. Demonstrates natural recovery (clipping + AdamW)
    4. Simulates checkpoint-rollback when recovery fails
    5. Shows the effect of different max_norm values on spike absorption
================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerBlock(nn.Module):
    """Minimal Transformer block for realistic gradient dynamics."""
    def __init__(self, d: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                           need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=64, n_layers=2, max_len=32):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TinyTransformerBlock(d) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def train_run(model, max_norm, n_steps=200, spike_at=100,
              spike_scale=30.0, checkpoint_every=20):
    """
    Run training with a simulated spike and optional clipping.
    Returns (loss_history, norm_history, checkpoint_info).
    """
    opt       = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    x_normal  = torch.randint(0, 256, (16, 16))
    x_spike   = torch.randint(0, 256, (16, 16))

    loss_history = []
    norm_history = []
    checkpoints  = {}

    for step in range(n_steps):
        # Save checkpoint
        if step % checkpoint_every == 0:
            checkpoints[step] = copy.deepcopy(model.state_dict())

        # Select batch
        is_spike = (step == spike_at)
        x = x_spike if is_spike else x_normal

        # Forward
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if is_spike:
            # Simulate "hard batch": amplify the loss
            loss = F.cross_entropy(
                logits.view(-1, 256) * spike_scale,
                x.view(-1)
            )
        else:
            loss = F.cross_entropy(logits.view(-1, 256), x.view(-1))

        loss.backward()

        # Record pre-clip norm
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        norm_history.append(total_norm)

        # Clip
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        opt.step()

        # Eval loss on normal batch
        with torch.no_grad():
            eval_loss = F.cross_entropy(
                model(x_normal).view(-1, 256), x_normal.view(-1)
            ).item()
        loss_history.append(eval_loss)

    return loss_history, norm_history, checkpoints


def ascii_timeseries(values, title, width=60, height=8, spike_at=None):
    """ASCII plot of a time series."""
    n   = len(values)
    lo  = min(v for v in values if not math.isnan(v) and not math.isinf(v))
    hi  = max(v for v in values if not math.isnan(v) and not math.isinf(v))
    if hi == lo:
        hi = lo + 1e-9

    # Downsample
    cols = []
    for c in range(width):
        idx = int(c / width * n)
        cols.append(values[min(idx, n-1)])

    grid = [[" "] * width for _ in range(height)]
    for c, val in enumerate(cols):
        if math.isnan(val) or math.isinf(val):
            for r in range(height):
                grid[r][c] = "?"
            continue
        row = height - 1 - int((val - lo) / (hi - lo) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][c] = "█"
        if spike_at is not None:
            spike_col = int(spike_at / n * width)
            for r in range(height):
                if grid[r][spike_col] == " ":
                    grid[r][spike_col] = "│"

    lines = [f"  {title}  (lo={lo:.3f}, hi={hi:.3f})"]
    for row in grid:
        lines.append("  │" + "".join(row) + "│")
    lines.append("  └" + "─" * width + "┘")
    return "\n".join(lines)


if __name__ == "__main__":
    torch.manual_seed(42)
    N_STEPS   = 200
    SPIKE_AT  = 100
    SPIKE_SCL = 25.0

    configs = [
        ("No clipping (max_norm=∞)", None),
        ("max_norm=5.0  (loose)",    5.0),
        ("max_norm=1.0  (standard)", 1.0),
        ("max_norm=0.3  (tight)",    0.3),
    ]

    print("=" * 65)
    print(f"  LOSS SPIKE SIMULATION  (spike at step {SPIKE_AT})")
    print(f"  spike_scale={SPIKE_SCL}×  total_steps={N_STEPS}")
    print("=" * 65)

    base_model = TinyLM()
    results    = {}

    for name, max_norm in configs:
        model  = copy.deepcopy(base_model)
        losses, norms, ckpts = train_run(
            model, max_norm, N_STEPS, SPIKE_AT, SPIKE_SCL
        )
        results[name] = (losses, norms, ckpts, model)

    # Print loss at key milestones
    print(f"\n  {'Step':>5}", end="")
    for name, _ in configs:
        short = name[:16]
        print(f"  {short:>16}", end="")
    print()
    print(f"  {'':─>5}" + "".join(f"  {'':─>16}" for _ in configs))

    milestones = [50, 99, SPIKE_AT, 101, 110, 130, 150, 199]
    for s in milestones:
        if s >= N_STEPS:
            continue
        flag = " ← spike" if s == SPIKE_AT else ""
        print(f"  {s:>5}", end="")
        for name, _ in configs:
            losses = results[name][0]
            l = losses[s] if s < len(losses) else float("nan")
            inf_str = "  ∞ (diverged)" if math.isinf(l) or l > 1e6 else f"  {l:>16.4f}"
            print(inf_str, end="")
        print(flag)

    # ASCII plots
    print()
    for name, _ in configs:
        losses, norms, _, _ = results[name]
        # Replace inf/nan with a large but plottable value for display
        safe_losses = [min(l, 50.0) if not math.isnan(l) else 50.0 for l in losses]
        print(ascii_timeseries(safe_losses, f"Loss — {name}", spike_at=SPIKE_AT))
        print()

    # Checkpoint rollback demo
    print("=" * 65)
    print("  CHECKPOINT ROLLBACK DEMO")
    print("=" * 65)
    print()
    print("  If max_norm=None (no clipping) and the model diverges,")
    print("  we can roll back to the checkpoint before the spike.")
    print()

    no_clip_losses, _, ckpts, no_clip_model = results["No clipping (max_norm=∞)"]
    post_spike_loss = no_clip_losses[min(SPIKE_AT + 20, N_STEPS - 1)]
    pre_spike_ckpt  = max(k for k in ckpts if k <= SPIKE_AT - 1)

    print(f"  Loss at step {SPIKE_AT+20} (after spike, no clip): {post_spike_loss:.4f}")
    print(f"  Nearest checkpoint before spike:  step {pre_spike_ckpt}")
    print()
    print(f"  Rolling back to checkpoint at step {pre_spike_ckpt}...")
    no_clip_model.load_state_dict(ckpts[pre_spike_ckpt])

    # Quick eval after rollback
    x_eval = torch.randint(0, 256, (16, 16))
    with torch.no_grad():
        rollback_loss = F.cross_entropy(
            no_clip_model(x_eval).view(-1, 256), x_eval.view(-1)
        ).item()
    print(f"  Loss after rollback: {rollback_loss:.4f}")
    print()
    print("  Recommendation:")
    print("  • Save checkpoints every 1000–5000 steps in production.")
    print("  • Use max_norm=1.0 to absorb most spikes automatically.")
    print("  • Reserve rollback for unrecoverable divergences (norm→NaN).")
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
        from llm_training.visuals.gradient_clipping import (
            GRADCLIP_VISUAL_HTML,
            GRADCLIP_VISUAL_HEIGHT,
        )
        visual_html   = GRADCLIP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = GRADCLIP_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(
            f"[09_gradient_clipping_stability.py] Could not load visual: {e}",
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