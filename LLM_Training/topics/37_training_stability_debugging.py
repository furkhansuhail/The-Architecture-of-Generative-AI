"""
Training Stability and Debugging
==================================

LLM training at scale is fragile. Loss spikes, NaN gradients, divergence,
and silent numerical errors can waste days of expensive compute before being
detected. This module is a systematic guide to diagnosing and fixing the most
common training failures — from gradient explosions and loss plateaus to
subtle numerical instabilities that only manifest at long training horizons.
Understanding the root causes and their signatures separates practitioners
who can debug training runs from those who restart and hope.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Training Stability and Debugging"
DISPLAY_NAME = "37 · Training Stability & Debugging"
ICON         = "🔧"
SUBTITLE     = "Loss Spikes, NaN Gradients, and Fixes"


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

### The Taxonomy of Training Failures

Training failures fall into a small number of fundamental categories,
each with distinct signatures and remedies:

    Category              Signature                       Primary cause
    ────────────────────────────────────────────────────────────────────────
    Gradient explosion    Loss spike → NaN → divergence   LR too high, bad batch
    Gradient vanishing    Loss plateau, tiny grad norms   Architecture, LR too low
    Loss spike (recovers) Sudden large loss, then returns Bad batch in data
    Divergence            Monotonically increasing loss   Catastrophically wrong HP
    Loss plateau          No improvement for 1000+ steps  LR too low, dead optimum
    NaN in activations    Loss is NaN from step 1         Numerical issue, bad init
    Slow convergence      Good direction but too slow     LR too low, batch too small
    Overfitting           Train loss ↓ but val loss ↑     Too small dataset, no reg
    ────────────────────────────────────────────────────────────────────────

**Diagnosis is not guesswork.** Each failure leaves specific fingerprints
in the metric history. Learning to read these fingerprints is the core
diagnostic skill of LLM training.


### Gradient Explosion: The Most Common Acute Failure

**Signature:**
    •   Gradient norm suddenly spikes from ~1 to >100 or to NaN
    •   Loss immediately jumps 5–20× in one step
    •   After the spike: loss is NaN or stuck at a high value

**Causes:**
    1.  **Learning rate too high:** The weight update step overshoots the
        loss minimum, landing far from it. Next step also overshoots,
        amplifying the error. This is the most common cause.

    2.  **Bad data batch:** A single batch with unusual statistics (all
        tokens from a rare category, extremely long sequences, a corrupted
        example) produces an unusually large gradient.

    3.  **Insufficient warmup:** Starting training with the full LR (no
        warmup) often causes early-step explosions when the randomly
        initialised model has inconsistent gradients.

    4.  **Exploding activations:** If any intermediate activation grows
        very large (e.g., due to a very large weight in a residual connection),
        the gradient of the loss with respect to that activation is also large.

**Fixes:**
    1.  **Gradient clipping:** The most reliable defence. Setting max_norm=1.0
        prevents any gradient update from being larger than 1.0 in L2 norm.
        This clips the largest possible step size without changing direction.

    2.  **Reduce learning rate:** Usually a 3–10× reduction stops explosions.

    3.  **Increase warmup steps:** Use 1–5% of total training steps for warmup.

    4.  **Data quality filtering:** Remove or down-sample the pathological batches.

    5.  **Z-loss (auxiliary loss for logit scale):** Add a small penalty on the
        magnitude of the LM head logits to prevent them from growing large:
            z_loss = 1e-4 × mean(log_Z²)   where log_Z = logsumexp(logits, dim=-1)
        Used in PaLM and T5 to prevent softmax saturation at scale.


    **Diagram 1 — Gradient Explosion Signature:**

    GRADIENT EXPLOSION: WHAT IT LOOKS LIKE IN METRICS
    ════════════════════════════════════════════════════════════════

    Step    Loss      Grad norm    Note
    ──────────────────────────────────────────────────────────────
    ...     2.31      0.82         normal training
    ...     2.28      0.79         normal training
    499     2.26      0.77         normal training
    500     8.43      142.7        ← EXPLOSION! bad batch or LR spike
    501     NaN       NaN          ← diverged; all subsequent steps useless
    502     NaN       NaN

    With gradient clipping (max_norm=1.0):
    500     2.34      1.00*        ← clipped; loss bumps but doesn't explode
    501     2.29      0.82         recovers normally
    (* gradient was clipped: true norm was 142.7, clipped to 1.0)


### NaN Gradients: Systematic Debugging

NaN gradients can originate from many sources. Systematically isolating
the source is essential:

**Step 1: Is the NaN in the forward pass or backward pass?**
    # Forward pass check
    with torch.no_grad():
        out = model(x)
    if torch.isnan(out).any():
        print("NaN in forward pass output")
    # If no NaN here, the problem is in the backward pass

**Step 2: Which layer produces the first NaN?**
    for name, module in model.named_modules():
        module.register_forward_hook(
            lambda m, inp, out, n=name: (
                print(f"NaN at {n}") if torch.is_tensor(out) and torch.isnan(out).any()
                else None
            )
        )

**Step 3: Which batch triggers NaN?**
    for step, batch in enumerate(loader):
        if torch.isnan(batch["input_ids"].float()).any():
            print(f"NaN in input at step {step}")

**Common NaN sources and their fixes:**

    Source                          Fix
    ──────────────────────────────────────────────────────────────────
    log(0) in cross-entropy         Label smoothing or add eps
    sqrt(0) in normalization        Add eps to denominator
    0/0 in attention (all -inf)     Check attention mask logic
    exp(large_number) overflow      Check temperature / logit scale
    log softmax on all-zero input   Verify input is not all padding
    Division by near-zero norm      Add eps in RMSNorm, LayerNorm
    ──────────────────────────────────────────────────────────────────

**The "anomaly detection" mode:**
    torch.autograd.set_detect_anomaly(True)
    # This makes PyTorch show the exact operation that produced NaN
    # and the full forward stack trace. Very slow but invaluable for diagnosis.
    # Disable before production training.


### Loss Spikes That Recover

Some loss spikes are benign: the model temporarily outputs unusual distributions
for one bad batch, then recovers. These manifest as isolated spikes in the
loss curve that return to the trend within 5–20 steps.

**Diagnostic:** Plot the smoothed loss curve. If spikes are isolated and the
underlying trend is decreasing, they are usually caused by bad data batches.

**The "bad batch" analysis:**
    1.  Log the seed/indices of batches that caused large gradient norms
    2.  Inspect those batches for unusual properties:
        •   Very long sequences (near max_seq_len)
        •   Repeated rare tokens
        •   Language switches (English model seeing Chinese text)
        •   Corrupted examples (random bytes, HTML tags, etc.)

**Fixes:**
    1.  Gradient clipping (limits damage from any single bad batch)
    2.  Data filtering (remove pathological examples before training)
    3.  Skip-and-continue: if grad_norm > threshold, skip the optimizer step
    4.  Loss spike detection + rollback (Module 33)


### The Learning Rate: The Central Stability Knob

The learning rate is the single most important hyperparameter for stability.
Too high → divergence; too low → no learning.

**The loss landscape perspective:**
The loss function is a high-dimensional surface. The optimal learning rate
is roughly: LR ≈ 1/H_max, where H_max is the maximum eigenvalue of the Hessian.
In practice, you find LR empirically, but understanding this relationship
helps:
    •   At the optimal LR, the gradient step lands near the minimum of
        the local parabolic approximation
    •   Beyond 2/H_max, gradient descent diverges (oscillates around minimum)
    •   The condition number H_max/H_min determines how different the
        optimal LR is in different directions (motivates Adam over SGD)

**The LR range test (fast empirical LR search):**
    Start with a very small LR (1e-7), increase exponentially each step,
    plot loss vs LR. Find:
        •   The LR where loss first decreases steeply (lower bound)
        •   The LR just before loss increases or becomes unstable (upper bound)
        •   Set training LR to ≈ 10× below the upper bound

**AdamW-specific considerations:**
AdamW's effective step size is (LR / √v_t), where v_t is the running second
moment. During warmup (when v_t is very small), even small gradients produce
large steps. This is why warmup is essential with Adam-based optimisers.

Without warmup: early steps have unboundedly large effective LR → explosion
With warmup: v_t has time to accumulate, normalising the gradient scale


### Diagnosing Slow Convergence

**Slow convergence** (loss decreasing but very slowly) has different causes
from divergence. Common diagnostics:

**1. LR too low:**
    Symptom: loss decreases monotonically but at a glacial pace.
    Fix: increase LR. Consider LR range test.

**2. Batch size too small:**
    Symptom: noisy loss curve, high variance in gradient estimates.
    Fix: increase batch size or gradient accumulation steps.

**3. Warmup too long:**
    Symptom: loss barely moves for the first 10% of training.
    Fix: reduce warmup_steps to 0.5–1% of total training.

**4. Data leakage / memorisation:**
    Symptom: train loss very low, val loss much higher. Model memorising.
    Fix: shuffle more aggressively, increase regularisation.

**5. Dead neurons in ReLU layers:**
    Symptom: certain neurons always output zero; gradients don't flow through.
    Fix: use GELU or SiLU instead of ReLU; reduce LR; check initialisation.

**6. Insufficient model capacity:**
    Symptom: loss decreases but plateaus well above expected value (> 3.0 for LLMs).
    Fix: increase model size (d_model, n_layers).


### Weight and Activation Monitoring

Beyond loss and gradient norms, monitoring intermediate values reveals
early warning signs:

**Weight norm growth:**
    ideal:    ||W||_F grows slowly and steadily (~5–10% per 10% training)
    warning:  rapidly growing norms → instability risk
    critical: ||W||_F → ∞ → model diverging

**Attention entropy:**
    ideal:    attention weights spread across many tokens (high entropy)
    warning:  attention collapses to ≈ 1 token (low entropy)
    meaning:  "attention sinks" — model puts all weight on a few tokens
    fix:      add attention sink tokens (like [BOS]), use sliding window attention

**Logit scale:**
    ideal:    max logit ≈ 5–20 (reasonable softmax probabilities)
    warning:  max logit > 50 (softmax becomes ≈ one-hot → unstable)
    fix:      z_loss auxiliary term, logit capping, LR reduction

**LayerNorm/RMSNorm scale factors:**
    ideal:    scale weights (γ) near 1.0 at start, drifting slowly
    warning:  scale collapsing to 0 → layer essentially disabled
    fix:      check learning rate; don't apply weight decay to norm params


    **Diagram 2 — Health Dashboard for LLM Training:**

    LLM TRAINING HEALTH DASHBOARD (what to monitor and what's normal)
    ════════════════════════════════════════════════════════════════

    Metric          Healthy range          Warning sign       Critical sign
    ─────────────────────────────────────────────────────────────────────────
    Train loss      Decreasing trend       Plateau > 1k steps NaN / diverging
    Val loss        Slightly > train       >> train (overfit)  NaN
    Grad norm       0.5 – 10               > 50                > 1000 / NaN
    Weight norm     Slow growth            Rapid growth        Exponential growth
    LR              Per schedule           Stuck too high      Stuck too low
    Loss scale (FP16) 2^16 – 2^24          Frequently halving  Hits 1.0
    GPU util        > 40% (compute-bound) 20–40% (check data) < 20% (bottleneck)


### The Debugging Checklist

When a training run fails, work through this checklist:

    STEP 1: Identify when the failure started
        →  Plot metrics from the beginning. Was it sudden or gradual?

    STEP 2: Check gradient norms
        →  Large norm: gradient explosion. Check LR, batch quality.
        →  Zero norm: vanishing. Check architecture, LR.

    STEP 3: Check loss scale (if using FP16)
        →  If GradScaler halved the scale many times → fp16 overflow.
        →  Consider switching to bf16 or adding GradScaler.

    STEP 4: Check for NaN in inputs
        →  Are any input token IDs out of range? Any NaN in embeddings?

    STEP 5: Run anomaly detection on a single batch
        →  torch.autograd.set_detect_anomaly(True)
        →  This will show exactly where NaN was produced.

    STEP 6: Reduce to minimum reproducible case
        →  Can you reproduce the failure with batch_size=1?
        →  Can you reproduce with a different model configuration?

    STEP 7: Check data pipeline
        →  Log the raw batch that caused the failure.
        →  Is there a pattern in the problematic batches?
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Training Stability Interventions

| Problem              | Primary diagnosis           | Fix                              | Severity  |
|----------------------|-----------------------------|----------------------------------|-----------|
| Loss spike → NaN     | Gradient explosion          | Gradient clipping, reduce LR     | Critical  |
| Loss plateau early   | LR too low / warmup too long| Increase LR, reduce warmup       | High      |
| Loss plateau late    | Cosine decay bottomed out   | LR restart (SGDR), cycle         | Medium    |
| Loss increasing      | LR too high                 | Reduce LR 3–10×                  | Critical  |
| NaN from step 1      | Init / numerical issue      | anomaly detection, check log(0)  | Critical  |
| Train >> val loss    | Overfitting                 | More data, dropout, regularize   | High      |
| Slow GPU util        | Data loading bottleneck     | Increase num_workers, pin_memory | Medium    |
| NaN only in fp16     | Overflow                    | Switch to bf16 or add GradScaler | High      |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Gradient and Loss Anomaly Detector": {
        "description": "Build a comprehensive training anomaly detector that monitors gradient norms, loss spikes, NaN propagation, and weight divergence — with automatic root-cause analysis.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
TRAINING ANOMALY DETECTOR
================================================================================

A comprehensive monitoring system that:
    1. Detects gradient explosions and NaN gradients
    2. Identifies loss spikes (sudden deviations from rolling average)
    3. Detects NaN propagation through the network (layer-by-layer)
    4. Monitors weight norm growth (divergence early warning)
    5. Performs automated root-cause analysis
    6. Suggests fixes based on detected patterns

This is the monitoring system that catches problems before they waste GPU hours.

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Anomaly types ─────────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    """One detected training anomaly."""
    step:       int
    kind:       str       # "nan_loss", "grad_explosion", "loss_spike", etc.
    severity:   str       # "warning" or "critical"
    message:    str
    value:      float
    suggestion: str


# ── The anomaly detector ──────────────────────────────────────────────────────

class TrainingAnomalyDetector:
    """
    Monitors training metrics and detects common failure modes.

    Attach to your training loop:
        detector = TrainingAnomalyDetector()
        # In training loop:
        anomalies = detector.check(
            step=step, loss=loss, grad_norm=grad_norm, model=model
        )
        for a in anomalies:
            print(f"⚠️  {a.message}")
    """

    def __init__(self,
                 loss_spike_window:     int   = 20,
                 loss_spike_threshold:  float = 3.0,
                 grad_explosion_thresh: float = 100.0,
                 grad_vanish_thresh:    float = 1e-7,
                 weight_growth_thresh:  float = 0.5,   # 50% growth in one check
                 check_weights_every:   int   = 50):
        self.loss_window   = deque(maxlen=loss_spike_window)
        self.grad_history  = deque(maxlen=loss_spike_window)
        self.weight_norms  = {}   # name → last seen norm
        self.loss_spike_thr= loss_spike_threshold
        self.grad_exp_thr  = grad_explosion_thresh
        self.grad_van_thr  = grad_vanish_thresh
        self.wt_growth_thr = weight_growth_thresh
        self.check_wt_every= check_weights_every
        self.anomalies     = []
        self._step_count   = 0

    def check(self, step: int, loss: float,
               grad_norm: float = None,
               model: nn.Module = None) -> list[Anomaly]:
        """
        Check all monitored quantities for anomalies.
        Returns list of newly detected anomalies.
        """
        new_anomalies = []
        self._step_count += 1

        # ── 1. NaN loss ────────────────────────────────────────────────────────
        if math.isnan(loss) or math.isinf(loss):
            a = Anomaly(
                step     = step,
                kind     = "nan_loss",
                severity = "critical",
                message  = f"Loss is {'NaN' if math.isnan(loss) else 'Inf'} at step {step}",
                value    = loss,
                suggestion = (
                    "1. Check for log(0) in loss function (add epsilon).\n"
                    "   2. Check GradScaler scale: if using FP16, it may have underflowed.\n"
                    "   3. Enable anomaly detection: torch.autograd.set_detect_anomaly(True)\n"
                    "   4. Run forward pass with no_grad to isolate forward vs backward."
                )
            )
            new_anomalies.append(a)
            self.anomalies.append(a)
            return new_anomalies   # no point checking more if loss is NaN

        self.loss_window.append(loss)

        # ── 2. Loss spike ──────────────────────────────────────────────────────
        if len(self.loss_window) >= 5:
            # Rolling mean of previous window (exclude current)
            prev = list(self.loss_window)[:-1]
            rolling_mean = sum(prev) / len(prev)
            if rolling_mean > 0 and loss > self.loss_spike_thr * rolling_mean:
                a = Anomaly(
                    step     = step,
                    kind     = "loss_spike",
                    severity = "warning",
                    message  = (f"Loss spike at step {step}: {loss:.3f} "
                                 f"(rolling mean: {rolling_mean:.3f}, "
                                 f"ratio: {loss/rolling_mean:.1f}×)"),
                    value    = loss,
                    suggestion = (
                        "1. If spike is isolated: likely a bad data batch. Add gradient clipping.\n"
                        "   2. If spikes are frequent: LR is too high. Reduce by 3×.\n"
                        "   3. Log the batch that caused the spike for inspection.\n"
                        "   4. Consider rollback to last checkpoint if spike doesn't recover."
                    )
                )
                new_anomalies.append(a)
                self.anomalies.append(a)

        # ── 3. Loss plateau ────────────────────────────────────────────────────
        if len(self.loss_window) == self.loss_window.maxlen:
            window = list(self.loss_window)
            first_half = sum(window[:len(window)//2]) / (len(window)//2)
            second_half = sum(window[len(window)//2:]) / (len(window)//2)
            improvement = (first_half - second_half) / max(first_half, 1e-8)
            if improvement < 0.001:   # < 0.1% improvement
                a = Anomaly(
                    step     = step,
                    kind     = "loss_plateau",
                    severity = "warning",
                    message  = (f"Loss plateau detected at step {step}: "
                                 f"only {improvement*100:.3f}% improvement "
                                 f"over last {self.loss_window.maxlen} steps"),
                    value    = improvement,
                    suggestion = (
                        "1. LR may have decayed too much. Try a warm restart.\n"
                        "   2. Model may have converged. Check validation loss quality.\n"
                        "   3. LR cycle (SGDR) can escape local minima.\n"
                        "   4. Data: are you seeing the same examples repeatedly?"
                    )
                )
                new_anomalies.append(a)
                self.anomalies.append(a)

        # ── 4. Gradient explosion / vanishing ──────────────────────────────────
        if grad_norm is not None:
            self.grad_history.append(grad_norm)

            if math.isnan(grad_norm):
                a = Anomaly(
                    step     = step,
                    kind     = "nan_gradient",
                    severity = "critical",
                    message  = f"NaN gradient at step {step}",
                    value    = grad_norm,
                    suggestion = (
                        "1. Run torch.autograd.set_detect_anomaly(True) to find the source.\n"
                        "   2. Check for operations that produce NaN: log(0), 0/0, sqrt(0).\n"
                        "   3. Check attention mask: all-inf row → NaN after softmax.\n"
                        "   4. Check data: NaN in input tokens or embeddings?"
                    )
                )
                new_anomalies.append(a)
                self.anomalies.append(a)

            elif grad_norm > self.grad_exp_thr:
                a = Anomaly(
                    step     = step,
                    kind     = "gradient_explosion",
                    severity = "critical",
                    message  = f"Gradient explosion at step {step}: norm={grad_norm:.1f}",
                    value    = grad_norm,
                    suggestion = (
                        "1. Gradient clipping (max_norm=1.0) — add if not present.\n"
                        "   2. Reduce learning rate by 3–10×.\n"
                        "   3. Increase warmup steps to 1–5% of total training.\n"
                        "   4. Check for bad data batch: log batch indices."
                    )
                )
                new_anomalies.append(a)
                self.anomalies.append(a)

            elif grad_norm < self.grad_van_thr:
                a = Anomaly(
                    step     = step,
                    kind     = "gradient_vanishing",
                    severity = "warning",
                    message  = f"Vanishing gradients at step {step}: norm={grad_norm:.2e}",
                    value    = grad_norm,
                    suggestion = (
                        "1. Increase learning rate.\n"
                        "   2. Check for dead neurons (all-zero activations after ReLU).\n"
                        "   3. Verify residual connections are properly configured.\n"
                        "   4. Check weight initialisation."
                    )
                )
                new_anomalies.append(a)
                self.anomalies.append(a)

        # ── 5. Weight norm growth ──────────────────────────────────────────────
        if model is not None and self._step_count % self.check_wt_every == 0:
            for name, param in model.named_parameters():
                if param.data is not None:
                    current_norm = param.data.norm().item()
                    if name in self.weight_norms:
                        prev_norm = self.weight_norms[name]
                        if prev_norm > 0:
                            growth = (current_norm - prev_norm) / prev_norm
                            if growth > self.wt_growth_thr:
                                a = Anomaly(
                                    step     = step,
                                    kind     = "weight_growth",
                                    severity = "warning",
                                    message  = (f"Rapid weight norm growth in {name}: "
                                                 f"{prev_norm:.3f} → {current_norm:.3f} "
                                                 f"({growth*100:.0f}% growth)"),
                                    value    = growth,
                                    suggestion = (
                                        "1. Reduce learning rate.\n"
                                        "   2. Increase weight decay for this parameter group.\n"
                                        "   3. Check if this layer is producing outlier gradients."
                                    )
                                )
                                new_anomalies.append(a)
                                self.anomalies.append(a)
                    self.weight_norms[name] = current_norm

        return new_anomalies

    def summary(self) -> str:
        """Print a summary of all detected anomalies."""
        if not self.anomalies:
            return "  No anomalies detected. ✓"
        lines = [f"  {len(self.anomalies)} anomaly/anomalies detected:"]
        counts = {}
        for a in self.anomalies:
            counts[a.kind] = counts.get(a.kind, 0) + 1
        for kind, count in sorted(counts.items()):
            emoji = "🚨" if any(a.severity == "critical" and a.kind == kind
                                for a in self.anomalies) else "⚠️"
            lines.append(f"    {emoji}  {kind}: {count} occurrence(s)")
        return "\n".join(lines)


# ── Demo: simulate various training failures ──────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=64, n_layers=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2, bias=False),
                          nn.GELU(), nn.Linear(d*2, d, bias=False))
            for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.vocab  = vocab

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers: h = h + l(h)
        return self.head(self.norm(h))


def run_demo_training(scenario: str, n_steps: int = 60) -> list[Anomaly]:
    """
    Simulate a training run under different scenarios:
        "healthy":    normal training
        "lr_too_high": learning rate is 10× too high → explosion
        "nan_input":  corrupted batch at step 30
        "plateau":    LR too low → no learning
    """
    import random
    random.seed(42)
    torch.manual_seed(42)

    VOCAB, D = 256, 64
    model    = TinyLM(VOCAB, D)
    detector = TrainingAnomalyDetector(loss_spike_window=10, check_weights_every=20)

    lr = {
        "healthy":    3e-4,
        "lr_too_high": 3e-2,    # 100× too high
        "nan_input":  3e-4,
        "plateau":    3e-8,     # pathologically low
    }.get(scenario, 3e-4)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"  Scenario: '{scenario}'")
    loss_val = None
    for step in range(1, n_steps + 1):
        x = torch.randint(0, VOCAB, (4, 16))
        y = torch.randint(0, VOCAB, (4, 16))

        # Inject NaN input at step 30 for "nan_input" scenario
        if scenario == "nan_input" and step == 30:
            x[0, 0] = -1   # invalid token ID → embedding lookup issue

        opt.zero_grad()
        try:
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
            loss.backward()
        except RuntimeError as e:
            loss = torch.tensor(float("nan"))

        # Gradient clipping only for healthy scenario
        if scenario == "healthy":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5 if any(p.grad is not None for p in model.parameters()) else 0.0

        opt.step()
        loss_val = loss.item()

        anomalies = detector.check(step, loss_val, grad_norm, model)
        if anomalies:
            for a in anomalies:
                print(f"    [step {step:>4}] {'🚨' if a.severity == 'critical' else '⚠️ '} "
                      f"{a.kind}: {a.message.split(':', 1)[-1].strip()[:60]}")

    print(detector.summary())
    return detector.anomalies


if __name__ == "__main__":
    print("=" * 65)
    print("  TRAINING ANOMALY DETECTION DEMO")
    print("=" * 65)
    print()

    for scenario in ["healthy", "lr_too_high", "nan_input", "plateau"]:
        print("─" * 65)
        anomalies = run_demo_training(scenario, n_steps=50)
        print()

    # Show fix suggestions for a critical anomaly
    print("=" * 65)
    print("  EXAMPLE FIX SUGGESTION FOR GRADIENT EXPLOSION")
    print("=" * 65)
    print()
    dummy_anomaly = Anomaly(
        step=500, kind="gradient_explosion", severity="critical",
        message="Gradient explosion at step 500: norm=142.7",
        value=142.7,
        suggestion=(
            "1. Gradient clipping (max_norm=1.0) — add if not present.\n"
            "   2. Reduce learning rate by 3–10×.\n"
            "   3. Increase warmup steps to 1–5% of total training.\n"
            "   4. Check for bad data batch: log batch indices."
        )
    )
    print(f"  Anomaly: {dummy_anomaly.message}")
    print(f"  Suggestions:")
    for line in dummy_anomaly.suggestion.split("\n"):
        print(f"  {line}")
''',
    },

    "LR Range Test and Warm Restart": {
        "description": "Implement the LR range test (find optimal LR empirically), cosine warm restarts (SGDR), and demonstrate how warm restarts escape loss plateaus.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LR RANGE TEST AND WARM RESTARTS (SGDR)
================================================================================

Implements:
    1. LR range test: exponentially sweep LR, find optimal range
    2. Cosine Annealing with Warm Restarts (SGDR)
    3. Demonstration: warm restart escapes a loss plateau

The LR range test (Smith, 2017) provides a data-driven way to find the
optimal LR without grid search. SGDR (Loshchilov & Hutter, 2016) enables
periodic learning rate restarts that help escape local minima.

================================================================================
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── LR range test ────────────────────────────────────────────────────────────

def lr_range_test(model: nn.Module,
                   data_fn,          # callable: returns (x, y) batch
                   n_steps: int = 100,
                   start_lr: float = 1e-7,
                   end_lr: float = 10.0,
                   smoothing: float = 0.05) -> dict:
    """
    LR range test (Smith, 2017).
    Sweep learning rate from start_lr to end_lr over n_steps,
    record the loss at each step.

    Returns dict with lr_values and loss_values for plotting.

    Interpretation:
        - Find the LR where loss starts to decrease (lower bound)
        - Find the LR just before loss starts to increase (upper bound)
        - Use upper_bound / 10 as your training LR
    """
    model_copy  = copy.deepcopy(model)
    optimizer   = torch.optim.SGD(model_copy.parameters(), lr=start_lr)

    lr_mult   = (end_lr / start_lr) ** (1 / n_steps)
    lr_values = []
    loss_values = []
    smoothed_loss = None
    best_loss   = float("inf")

    for step in range(n_steps):
        current_lr = start_lr * (lr_mult ** step)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        x, y = data_fn()
        optimizer.zero_grad()
        out  = model_copy(x)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        # Exponential smoothing
        if smoothed_loss is None:
            smoothed_loss = loss_val
        else:
            smoothed_loss = (1 - smoothing) * smoothed_loss + smoothing * loss_val

        lr_values.append(current_lr)
        loss_values.append(smoothed_loss)

        # Early stop if loss is exploding
        if smoothed_loss > 4 * best_loss:
            break
        best_loss = min(best_loss, smoothed_loss)

    # Find optimal range
    min_loss_idx  = loss_values.index(min(loss_values))
    # Steepest descent: biggest drop
    max_drop_idx  = max(range(1, len(loss_values)),
                         key=lambda i: loss_values[i-1] - loss_values[i])

    return {
        "lr_values":    lr_values,
        "loss_values":  loss_values,
        "min_loss_lr":  lr_values[min_loss_idx],
        "steep_lr":     lr_values[max_drop_idx],
        "suggested_lr": lr_values[max_drop_idx] / 10,
    }


# ── SGDR: Cosine Annealing with Warm Restarts ──────────────────────────────

class SGDRScheduler:
    """
    Cosine Annealing with Warm Restarts (SGDR, Loshchilov & Hutter 2016).

    LR follows a cosine curve from max_lr to min_lr.
    After T_0 steps, restarts from max_lr again.
    Each restart doubles the cycle length (T_mult).

    Why this helps:
    - Standard cosine decay commits to a minimum and never leaves it.
    - SGDR periodically restarts with a high LR, allowing the model to
      escape local minima and explore other regions of the loss landscape.
    - Each restart snapshot can be saved and ensembled (snapshot ensembles).
    """

    def __init__(self, optimizer, T_0: int, T_mult: int = 2,
                  max_lr: float = 3e-4, min_lr: float = 3e-5):
        self.optimizer  = optimizer
        self.T_0        = T_0
        self.T_mult     = T_mult
        self.max_lr     = max_lr
        self.min_lr     = min_lr
        self.step_count = 0
        self.T_cur      = T_0   # current cycle length
        self.T_i        = 0     # steps into current cycle
        self.n_restarts = 0

    def get_lr(self) -> float:
        """Compute LR for current step."""
        progress = self.T_i / self.T_cur
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self):
        """Advance one training step."""
        self.step_count += 1
        self.T_i        += 1

        # Restart cycle
        if self.T_i >= self.T_cur:
            self.T_i    = 0
            self.T_cur *= self.T_mult
            self.n_restarts += 1

        current_lr = self.get_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = current_lr
        return current_lr


# ── Demo ──────────────────────────────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=64, n_layers=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2, bias=False),
                          nn.GELU(), nn.Linear(d*2, d, bias=False))
            for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.vocab  = vocab

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers: h = h + l(h)
        return self.head(self.norm(h))


if __name__ == "__main__":
    torch.manual_seed(42)
    VOCAB, D = 256, 64

    # Shared data function
    def get_batch():
        x = torch.randint(0, VOCAB, (4, 16))
        y = torch.randint(0, VOCAB, (4, 16))
        return x, y

    # ── 1. LR Range Test ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  LR RANGE TEST")
    print("=" * 65)
    print()

    model = TinyLM(VOCAB, D)
    result = lr_range_test(model, get_batch, n_steps=80,
                            start_lr=1e-7, end_lr=5.0)

    print(f"  LR at minimum loss:     {result['min_loss_lr']:.2e}")
    print(f"  LR at steepest descent: {result['steep_lr']:.2e}")
    print(f"  Suggested training LR:  {result['suggested_lr']:.2e}")
    print()

    # ASCII plot of loss vs LR
    bars   = " ▁▂▃▄▅▆▇█"
    losses = result["loss_values"]
    lrs    = result["lr_values"]
    lo, hi = min(losses), max(losses)
    span   = hi - lo + 1e-8
    print("  Loss vs LR (lower = better):")
    print("  " + "─" * 52)
    # Sample 25 points
    step = max(1, len(losses) // 25)
    for i in range(0, len(losses), step):
        bar_h   = int((losses[i] - lo) / span * 7)
        lr_str  = f"{lrs[i]:.1e}"
        bar_str = "█" * (8 - bar_h) + "░" * bar_h
        marker  = " ← suggested" if abs(lrs[i] - result["suggested_lr"]) < result["suggested_lr"] * 0.5 else ""
        print(f"  {lr_str:>10}: {bar_str}{marker}")
    print()

    # ── 2. SGDR: Warm Restarts ────────────────────────────────────────────────
    print("=" * 65)
    print("  SGDR: COSINE ANNEALING WITH WARM RESTARTS")
    print("  T_0=20, T_mult=2 (cycles: 20, 40, 80 steps)")
    print("=" * 65)
    print()

    # Compare: cosine decay vs SGDR
    N_STEPS = 120

    # Train with plain cosine decay
    torch.manual_seed(0)
    model_cosine = TinyLM(VOCAB, D)
    opt_cosine   = torch.optim.AdamW(model_cosine.parameters(), lr=3e-4)

    # Train with SGDR
    torch.manual_seed(0)
    model_sgdr   = TinyLM(VOCAB, D)
    opt_sgdr     = torch.optim.AdamW(model_sgdr.parameters(), lr=3e-4)
    sgdr_sched   = SGDRScheduler(opt_sgdr, T_0=20, T_mult=2, max_lr=3e-4, min_lr=3e-6)

    cosine_losses = []
    sgdr_losses   = []

    for step in range(N_STEPS):
        # LR for cosine model
        lr_cosine = 3e-6 + 0.5 * (3e-4 - 3e-6) * (1 + math.cos(math.pi * step / N_STEPS))
        for g in opt_cosine.param_groups:
            g["lr"] = lr_cosine

        x, y = get_batch()

        # Cosine step
        opt_cosine.zero_grad()
        loss_c = F.cross_entropy(model_cosine(x).view(-1, VOCAB), y.view(-1))
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(model_cosine.parameters(), 1.0)
        opt_cosine.step()
        cosine_losses.append(loss_c.item())

        # SGDR step
        sgdr_sched.step()
        opt_sgdr.zero_grad()
        loss_s = F.cross_entropy(model_sgdr(x).view(-1, VOCAB), y.view(-1))
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(model_sgdr.parameters(), 1.0)
        opt_sgdr.step()
        sgdr_losses.append(loss_s.item())

    # Report
    print(f"  {'Step':>6}  {'Cosine loss':>12}  {'SGDR loss':>12}  {'SGDR LR':>12}")
    print(f"  {'':─>6}  {'':─>12}  {'':─>12}  {'':─>12}")
    for step in [0, 20, 40, 60, 80, 100, 119]:
        sgdr_lr = sgdr_sched.get_lr()
        # Recompute SGDR LR at this step (approximate)
        t_cur = 20; t_i = step % t_cur if step < 20 else 0
        est_lr = 3e-6 + 0.5 * (3e-4 - 3e-6) * (1 + math.cos(math.pi * min(1, step/20)))
        print(f"  {step:>6}  {cosine_losses[step]:>12.4f}  {sgdr_losses[step]:>12.4f}  {est_lr:>12.2e}")

    print()
    final_c = min(cosine_losses[-20:])
    final_s = min(sgdr_losses[-20:])
    print(f"  Best loss (last 20 steps): cosine={final_c:.4f}  SGDR={final_s:.4f}")
    print()
    print("  SGDR's warm restarts help escape flat regions in the loss landscape.")
    print("  Each restart snapshot can also be saved and used for model ensembling.")

    # ── 3. LR schedule visualisation ─────────────────────────────────────────
    print()
    print("  SGDR LR SCHEDULE (T_0=20, T_mult=2):")
    sched_viz = SGDRScheduler(None, T_0=20, T_mult=2, max_lr=3e-4, min_lr=3e-6)
    bars_lr   = " ▁▂▃▄▅▆▇█"
    for step in range(N_STEPS):
        lr = sched_viz.get_lr()
        sched_viz.step()
        frac = (lr - 3e-6) / (3e-4 - 3e-6)
        bar  = bars_lr[int(frac * 7)]
        if step % 5 == 0:
            restart = " ← restart" if (step in [0, 20, 60, 120]) else ""
            print(f"  step {step:>4}: {bar} {lr:.2e}{restart}")
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
    #     from llm_training.visuals.training_stability import (
    #         STAB_VISUAL_HTML,
    #         STAB_VISUAL_HEIGHT,
    #     )
    #     visual_html   = STAB_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = STAB_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[37_training_stability_debugging.py] Could not load visual: {e}",
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