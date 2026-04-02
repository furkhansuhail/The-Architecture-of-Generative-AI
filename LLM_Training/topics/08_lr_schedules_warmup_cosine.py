"""
Learning Rate Schedules — Warmup, Cosine Decay, and WSD
=========================================================

The learning rate is the single most impactful hyperparameter in LLM
training. A fixed learning rate is never used in practice — the schedule
determines how the learning rate evolves across hundreds of thousands of
steps. Getting the schedule wrong causes slow convergence, instability,
or failure to reach the model's quality ceiling.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Learning Rate Schedules — Warmup, Cosine, WSD"
DISPLAY_NAME = "08 · LR Schedules"
ICON         = "📈"
SUBTITLE     = "Warmup, Cosine Decay, and WSD"


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

### Why a Fixed Learning Rate Fails

A single fixed learning rate creates an unavoidable trade-off:

    •   Too large: training diverges or oscillates. Gradients are noisy,
        especially early in training when the loss landscape is steep and
        the gradient estimates from small batches are unreliable. A large
        step can overshoot minima and cause loss spikes.

    •   Too small: training converges, but slowly. The model never reaches
        the quality achievable with a well-tuned schedule.

The solution is to vary the learning rate across training: start small to
stabilise early dynamics, increase to a peak for fast convergence, then
gradually decrease to settle into a sharp minimum.


### Phase 1 — Linear Warmup

At the very start of training:
    •   Model parameters are random — gradients point in unstable directions.
    •   Adam's m and v buffers are zero — biased estimates cause large noisy
        updates (the bias correction partially mitigates this, but the
        gradient directions themselves are unreliable).
    •   The loss is high and the landscape is rough — large steps overshoot.

**Linear warmup** ramps the learning rate from 0 (or a small fraction of lr_max)
up to lr_max over T_warmup steps:

    lr(t) = lr_max × (t / T_warmup)     for t ≤ T_warmup

Typical warmup lengths:
    •   Small models (< 1B):   500 – 2,000 steps
    •   Large models (> 10B):  2,000 – 5,000 steps
    •   GPT-3 (175B):          375 steps (only! — short warmup at scale)
    •   LLaMA-1/2:             2,000 steps

Why does warmup help? The key is Adam's second moment v_t: at step 1,
v_t ≈ 0, so the effective learning rate 1/√v̂_t is enormous. Warmup
keeps the raw lr small while v_t builds up to a meaningful estimate over
the first few hundred steps. Once v_t is well-estimated, it's safe to use
the full lr_max.


### Phase 2 — Cosine Decay

After warmup, the learning rate decays from lr_max to lr_min following a
cosine curve:

    lr(t) = lr_min + ½ × (lr_max − lr_min) × (1 + cos(π × t' / T_decay))

where t' = t − T_warmup is the number of steps since warmup ended, and
T_decay is the total decay length (usually total_steps − T_warmup).

The cosine schedule has two desirable properties:

    1.  **Slow start, fast middle, slow end:** The cosine function decreases
        slowly at first (near lr_max, giving the model time to find a good
        basin), more steeply in the middle (fast convergence), and slowly
        again at the end (careful settling into the minimum).

    2.  **Smooth:** No discontinuities, no sudden drops. Smooth schedules
        produce stable training curves with fewer loss spikes.

**lr_min:** Typically set to 10% of lr_max (e.g. lr_max=3e-4, lr_min=3e-5)
or sometimes 0. A non-zero lr_min prevents the model from completely stopping
learning at the end.


    **Diagram 1 — Standard Warmup + Cosine Decay Schedule:**

    WARMUP + COSINE DECAY LEARNING RATE SCHEDULE
    ════════════════════════════════════════════════════════════════

    lr
    ▲
    │                  ╭──────╮
    │               ╭─╯       ╲
    lr_max ──────────────╯         ╲
    │        ╱                      ╲
    │       ╱                        ╲_____
    lr_min─┤──────────────────────────────────────── steps →
    │  0  T_warm         T_total
    │  └────┘└──────────────────────────┘
    │  warmup     cosine decay
    │
    │  Warmup: 0 → lr_max (linear)
    │  Decay:  lr_max → lr_min (cosine)

    Steps:     0    500   5000  10000  50000  100000
    lr:        0   0.1×  1.0×  0.98×   0.6×    0.1×  (× lr_max)


### Phase 3 — The Tail: Does LR = 0 Matter?

A well-known empirical finding: the final quality of a model depends strongly
on how low the learning rate is at the end of training.

**Chinchilla (Hoffmann et al., 2022)** found that models trained with a
final lr that's too high leave significant quality on the table. The final
few percent of decay often provide disproportionate quality improvement.

Rule of thumb for cosine decay: the final lr should be ≤ 10% of lr_max.
Setting lr_min = 0 is fine for pre-training if you have a known total budget.


### The Warmup-Stable-Decay (WSD) Schedule

Traditional cosine decay has a critical limitation: you must know the total
training budget upfront. If you decide to train for longer, you cannot simply
extend the cosine — the lr is already at lr_min and extending flat at lr_min
is suboptimal.

**WSD** (Hu et al., 2024, used in MiniCPM and many modern recipes) splits
training into three explicit phases:

    Phase 1 — Warmup:    T_warmup steps, lr: 0 → lr_max  (linear)
    Phase 2 — Stable:    T_stable steps, lr: lr_max  (constant)
    Phase 3 — Decay:     T_decay steps,  lr: lr_max → lr_min (cosine/sqrt)

The key innovation: **the stable phase can be extended indefinitely** (just
train more at lr_max), and the decay phase is applied only at the very end.
This decouples the "how long to train" decision from the schedule shape.

    •   Train at lr_max for as long as budget allows
    •   Apply a short decay phase (10–20% of total training) at the end
    •   This is effectively "cosine restart at the last checkpoint"

**Why does training at constant lr work?**
At a high, stable lr, the model is actively exploring the loss landscape.
The final decay phase then "anneals" the weights into a sharp minimum.
Empirically, a WSD model trained to N tokens with 10% decay matches or
exceeds a cosine-decay model trained to the same N tokens.

**Continuous training advantage:** WSD naturally supports the common
real-world scenario where a model is trained, then later extended on more
data. Each extension just runs more stable-phase training followed by a
fresh decay phase.


    **Diagram 2 — WSD vs Cosine Decay:**

    WSD vs COSINE DECAY
    ════════════════════════════════════════════════════════════════

    COSINE DECAY (fixed budget):
    lr ▲
       │   ╭──╮
       │  ╱    ╲
       │ ╱      ╲_________
       └──────────────────── steps
       │warmup│── cosine decay ──│
              Must decide total budget upfront ⚠️

    WSD (flexible budget):
    lr ▲
       │   ╭──────────────╮
       │  ╱                ╲
       │ ╱                  ╲____
       └──────────────────────── steps
       │warm│── stable ──│decay│
                         ↑ can extend "stable" as long as you want ✓

    WSD extension (train longer):
    lr ▲
       │   ╭──────────────────────╮
       │  ╱                        ╲
       │ ╱                          ╲____
       └────────────────────────────────── steps
       Extended stable → better final model, no schedule rework needed ✓


### Cosine with Restarts (SGDR)

SGDR (Loshchilov & Hutter, 2017) adds periodic restarts to the cosine
schedule. After each full cosine cycle, the lr resets to lr_max and the
cycle length is typically multiplied by a factor T_mult:

    Cycle 1: T₀ steps
    Cycle 2: T₀ × T_mult steps
    Cycle 3: T₀ × T_mult² steps  …

The restarts allow the optimiser to escape local minima and explore different
regions of the loss landscape. Less commonly used for LLM pre-training (WSD
and simple cosine dominate), but often used in fine-tuning.


### Reciprocal-Square-Root Schedule (Inverse Square Root)

Used by the original Transformer paper for machine translation:

    lr(t) = d_model^(-0.5) × min(t^(-0.5), t × T_warmup^(-1.5))

This combines warmup with a slow t^(-0.5) decay. It scales the peak learning
rate with d_model^(-0.5) — larger models automatically get smaller lr_max.
Less used for LLM pre-training but still common in translation models.


### The Relationship Between LR, Batch Size, and Tokens

**Linear scaling rule (Goyal et al., 2017):** When multiplying the batch
size by k, multiply lr by k. Keeps the per-token gradient variance constant.

    lr_new = lr_base × (batch_new / batch_base)

**Square-root scaling (more conservative, often preferred for LLMs):**

    lr_new = lr_base × √(batch_new / batch_base)

**Token-normalised perspective:** Some practitioners express lr in terms of
"learning rate per token" rather than per step:

    lr_per_token = lr_per_step / batch_size_in_tokens

This makes it easier to compare schedules across different batch sizes.


    **Diagram 3 — Common Schedule Shapes:**

    COMMON LR SCHEDULE SHAPES (all same total area ≈ same total "learning")
    ════════════════════════════════════════════════════════════════

    lr
    ▲
    │   ╭────────────────╮          Constant (never used for LLMs)
    │   │                 │
    ├───┤─────────────────┤──────
    │
    │   ╭──╮              ╮          Cosine (most common for pre-training)
    │  ╱    ╲             │╲
    ├──┤      ╲___________┤  ╲───
    │
    │   ╭──────────────╮  ╮          WSD (best for unknown total budget)
    │  ╱               │  │╲
    ├──┤               │  ╰──╲──
    │
    │   ╭──╮      ╭──╮      ╭──    Cosine with restarts (fine-tuning)
    │  ╱    ╲    ╱    ╲    ╱
    ├──┤      ╲──┤      ╲──┤
    │
    0                         steps


### Practical Recommendations

    Scenario                    Recommended Schedule
    ────────────────────────────────────────────────────────────────
    Pre-training, fixed budget   Warmup (2k steps) + cosine decay
    Pre-training, open-ended     WSD with 10% decay at the end
    Fine-tuning, one epoch       Short warmup + cosine to lr/10
    Fine-tuning, few-shot SFT    Linear decay or constant lr
    RLHF / PPO                   Constant small lr (1e-6 – 5e-6)
    ────────────────────────────────────────────────────────────────

**Key numbers for reference (from published LLM recipes):**

    Model           lr_max   T_warmup   Schedule     lr_min
    ──────────────────────────────────────────────────────────
    GPT-3 175B      6e-5     375        Cosine       6e-6
    Chinchilla 70B  1e-4     1500       Cosine       1e-5
    LLaMA-1 7B      3e-4     2000       Cosine       3e-5
    LLaMA-2 7B      3e-4     2000       Cosine       3e-5
    LLaMA-3 8B      3e-4     2000       Cosine       3e-5
    Mistral 7B      3e-4     ~2000      Cosine       3e-5
    ──────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
LR Schedule Comparison

| Schedule              | Fixed Budget? | Restartable? | Quality Ceiling | Used By              |
|-----------------------|---------------|--------------|-----------------|----------------------|
| Constant              | No            | N/A          | Low             | Legacy / baselines   |
| Linear decay          | Yes           | No           | Moderate        | BERT fine-tuning     |
| Cosine decay          | Yes           | No           | High            | GPT-3, LLaMA 1/2/3  |
| Cosine with restarts  | No            | Yes          | High            | Fine-tuning          |
| WSD                   | No            | Yes          | High            | MiniCPM, modern recipes|
| Inv. square root      | No            | No           | High            | Original Transformer |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "LR Schedule Library — All Variants": {
        "description": "Implement every common LR schedule (warmup+cosine, WSD, cosine restarts, inverse sqrt) from scratch with ASCII visualisation.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LR SCHEDULE LIBRARY — ALL COMMON VARIANTS
================================================================================

Implements and visualises:
    1. Linear warmup + cosine decay   (LLaMA, GPT-3 style)
    2. Warmup-Stable-Decay (WSD)       (MiniCPM style, flexible budget)
    3. Cosine with warm restarts (SGDR)
    4. Inverse square root             (original Transformer)
    5. Linear warmup + linear decay    (BERT fine-tuning)

Each scheduler is a pure function: get_lr(step) → float
================================================================================
"""

import math


# ── Schedule implementations ──────────────────────────────────────────────────

def cosine_schedule(step: int, total_steps: int, warmup_steps: int,
                    lr_max: float, lr_min: float) -> float:
    """
    Linear warmup followed by cosine decay.
    The standard schedule for LLM pre-training.
    """
    if step < warmup_steps:
        # Linear warmup: 0 → lr_max
        return lr_max * step / max(1, warmup_steps)

    if step >= total_steps:
        return lr_min

    # Cosine decay: lr_max → lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def wsd_schedule(step: int, warmup_steps: int, stable_steps: int,
                 decay_steps: int, lr_max: float, lr_min: float) -> float:
    """
    Warmup-Stable-Decay (WSD) schedule.
    Phase 1: Linear warmup  (0         → warmup_steps)
    Phase 2: Constant       (warmup    → warmup + stable)
    Phase 3: Cosine decay   (last decay_steps of training)
    """
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)

    stable_end = warmup_steps + stable_steps
    if step < stable_end:
        return lr_max

    decay_progress = (step - stable_end) / max(1, decay_steps)
    decay_progress = min(decay_progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return lr_min + (lr_max - lr_min) * cosine


def cosine_restarts(step: int, warmup_steps: int, T0: int,
                    T_mult: float, lr_max: float, lr_min: float) -> float:
    """
    Cosine schedule with warm restarts (SGDR).
    T0:     initial cycle length (steps)
    T_mult: cycle length multiplier after each restart
    """
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)

    s = step - warmup_steps
    # Find which cycle we are in
    cycle_len = T0
    cycle_start = 0
    cycle_idx = 0
    while cycle_start + cycle_len <= s:
        cycle_start += cycle_len
        cycle_len = int(cycle_len * T_mult)
        cycle_idx += 1

    progress = (s - cycle_start) / max(1, cycle_len)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def inverse_sqrt_schedule(step: int, warmup_steps: int,
                           lr_max: float, d_model: int = None) -> float:
    """
    Inverse square root decay (original Transformer paper).
    If d_model provided, lr_max is automatically scaled by d_model^(-0.5).
    """
    if d_model is not None:
        peak = d_model ** (-0.5) * min(
            step ** (-0.5),
            step * warmup_steps ** (-1.5)
        )
        return peak

    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    return lr_max * math.sqrt(warmup_steps) / math.sqrt(step)


def linear_warmup_linear_decay(step: int, warmup_steps: int,
                                total_steps: int,
                                lr_max: float, lr_min: float) -> float:
    """Linear warmup + linear decay. Common for BERT-style fine-tuning."""
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return lr_max - (lr_max - lr_min) * progress


# ── ASCII visualiser ──────────────────────────────────────────────────────────

def ascii_plot(lr_values: list[float], title: str, width: int = 60,
               height: int = 10) -> str:
    """Render a list of lr values as an ASCII line plot."""
    n     = len(lr_values)
    lo    = min(lr_values)
    hi    = max(lr_values)
    span  = hi - lo if hi > lo else 1.0

    # Downsample to width columns
    cols  = []
    for c in range(width):
        idx = int(c / width * n)
        cols.append(lr_values[min(idx, n - 1)])

    # Map to rows
    grid = [[" "] * width for _ in range(height)]
    prev_row = None
    for c, val in enumerate(cols):
        row = height - 1 - int((val - lo) / span * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][c] = "●"
        # Connect with vertical line if gap
        if prev_row is not None and abs(row - prev_row) > 1:
            r0, r1 = min(row, prev_row), max(row, prev_row)
            for r in range(r0, r1 + 1):
                if grid[r][c] == " ":
                    grid[r][c] = "│"
        prev_row = row

    lines = ["  " + title]
    lines.append("  lr_max │" + "─" * width + "│")
    for row_idx, row in enumerate(grid):
        prefix = "        │" if row_idx not in (0, height - 1) else (
            "  lr_max│" if row_idx == 0 else "  lr_min│"
        )
        lines.append("        │" + "".join(row) + "│")
    lines.append("  lr_min │" + "─" * width + "│")
    lines.append("         0" + " " * (width - 10) + "total_steps")
    return "\n".join(lines)


if __name__ == "__main__":
    TOTAL   = 10_000
    WARMUP  = 500
    LR_MAX  = 3e-4
    LR_MIN  = 3e-5

    schedules = {
        "Cosine (LLaMA style)": [
            cosine_schedule(t, TOTAL, WARMUP, LR_MAX, LR_MIN)
            for t in range(TOTAL)
        ],
        "WSD (stable=7000, decay=2000)": [
            wsd_schedule(t, WARMUP, 7000, 2000, LR_MAX, LR_MIN)
            for t in range(10_500)   # slightly longer to show full decay
        ],
        "Cosine Restarts (T0=2000, mult=2)": [
            cosine_restarts(t, WARMUP, 2000, 2.0, LR_MAX, LR_MIN)
            for t in range(TOTAL)
        ],
        "Inverse Sqrt": [
            inverse_sqrt_schedule(t, WARMUP, LR_MAX)
            for t in range(TOTAL)
        ],
        "Linear Warmup + Linear Decay (BERT)": [
            linear_warmup_linear_decay(t, WARMUP, TOTAL, LR_MAX, LR_MIN)
            for t in range(TOTAL)
        ],
    }

    for name, lrs in schedules.items():
        print()
        print(ascii_plot(lrs, name))
        peak_step = lrs.index(max(lrs))
        print(f"         peak lr={max(lrs):.2e} at step {peak_step}  |  "
              f"final lr={lrs[-1]:.2e}")

    # Numeric comparison at key milestones
    print()
    print("=" * 68)
    print("  NUMERIC LR VALUES AT KEY TRAINING MILESTONES")
    print("  (lr_max=3e-4, lr_min=3e-5, warmup=500, total=10000)")
    print("=" * 68)
    milestones = [0, 250, 500, 1000, 2500, 5000, 7500, 9000, 9999]
    print(f"\n  {'Step':>6}  {'Cosine':>12}  {'WSD':>12}  {'InvSqrt':>12}  {'Linear':>12}")
    print(f"  {'':─>6}  {'':─>12}  {'':─>12}  {'':─>12}  {'':─>12}")
    for s in milestones:
        c = cosine_schedule(s, TOTAL, WARMUP, LR_MAX, LR_MIN)
        w = wsd_schedule(s, WARMUP, 7000, 2000, LR_MAX, LR_MIN)
        i = inverse_sqrt_schedule(s, WARMUP, LR_MAX)
        l = linear_warmup_linear_decay(s, WARMUP, TOTAL, LR_MAX, LR_MIN)
        print(f"  {s:>6}  {c:>12.2e}  {w:>12.2e}  {i:>12.2e}  {l:>12.2e}")
''',
    },

    "PyTorch Scheduler Integration": {
        "description": "Wire all common LR schedules into a real PyTorch training loop with AdamW — LambdaLR, OneCycleLR, and a custom cosine implementation.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PYTORCH LR SCHEDULER INTEGRATION
================================================================================

Shows how to use LR schedulers in a real PyTorch training loop:
    1. Custom cosine+warmup via LambdaLR (most flexible)
    2. Built-in CosineAnnealingLR
    3. OneCycleLR (fast experimentation)
    4. Logging lr alongside loss at each step
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Custom warmup + cosine via LambdaLR ───────────────────────────────────────

def make_cosine_schedule(warmup_steps: int, total_steps: int,
                         lr_min_ratio: float = 0.1):
    """
    Returns a lambda function for use with torch.optim.lr_scheduler.LambdaLR.

    LambdaLR multiplies the base lr (set in the optimiser) by the returned
    factor at each step. Setting base lr = lr_max, the lambda returns a
    factor in [lr_min_ratio, 1.0].
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return lr_lambda


def make_wsd_schedule(warmup_steps: int, stable_steps: int,
                      decay_steps: int, lr_min_ratio: float = 0.1):
    """WSD schedule as a LambdaLR-compatible lambda."""
    stable_end = warmup_steps + stable_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < stable_end:
            return 1.0
        progress = (step - stable_end) / max(1, decay_steps)
        progress = min(progress, 1.0)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return lr_lambda


# ── Tiny model for demo ────────────────────────────────────────────────────────

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


# ── Training loop with LR logging ─────────────────────────────────────────────

def train_with_schedule(schedule_name: str, scheduler_fn, total_steps: int,
                        lr_max: float = 3e-4):
    torch.manual_seed(0)
    model     = TinyMLP()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr_max,
                                   betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = scheduler_fn(optimiser)

    x = torch.randn(128, 16)
    y = torch.randn(128, 1)

    log_steps    = set([1, 50, 100, 250, 500, 750, 1000])
    loss_history = []
    lr_history   = []

    for step in range(1, total_steps + 1):
        optimiser.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        loss_history.append(loss.item())
        lr_history.append(current_lr)

    return loss_history, lr_history


if __name__ == "__main__":
    TOTAL   = 1_000
    WARMUP  = 100
    LR_MAX  = 3e-4

    # Define schedulers
    schedule_configs = {
        "Cosine (LambdaLR)": lambda opt: torch.optim.lr_scheduler.LambdaLR(
            opt, make_cosine_schedule(WARMUP, TOTAL, lr_min_ratio=0.1)
        ),
        "WSD (LambdaLR)": lambda opt: torch.optim.lr_scheduler.LambdaLR(
            opt, make_wsd_schedule(WARMUP, int(0.7 * TOTAL),
                                   int(0.2 * TOTAL), lr_min_ratio=0.1)
        ),
        "CosineAnnealingLR (built-in)": lambda opt: (
            torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        opt, start_factor=1e-4, total_iters=WARMUP),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=TOTAL - WARMUP, eta_min=LR_MAX * 0.1),
                ],
                milestones=[WARMUP],
            )
        ),
        "OneCycleLR (built-in)": lambda opt: torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR_MAX, total_steps=TOTAL,
            pct_start=0.1, anneal_strategy="cos",
        ),
    }

    print("=" * 70)
    print("  PYTORCH LR SCHEDULER COMPARISON")
    print(f"  total_steps={TOTAL}, warmup={WARMUP}, lr_max={LR_MAX}")
    print("=" * 70)
    print()
    print(f"  {'Step':>6}", end="")
    for name in schedule_configs:
        short = name.split(" ")[0][:14]
        print(f"  {short:>14}", end="")
    print()
    print(f"  {'':─>6}" + "".join(f"  {'':─>14}" for _ in schedule_configs))

    all_lrs = {}
    all_losses = {}
    for name, sched_fn in schedule_configs.items():
        losses, lrs = train_with_schedule(name, sched_fn, TOTAL, LR_MAX)
        all_lrs[name]    = lrs
        all_losses[name] = losses

    for step in [1, 50, 100, 200, 500, 750, 999]:
        print(f"  {step:>6}", end="")
        for name in schedule_configs:
            lr = all_lrs[name][step - 1]
            print(f"  {lr:>14.2e}", end="")
        print()

    print()
    print(f"  Final loss comparison:")
    for name in schedule_configs:
        final_loss = all_losses[name][-1]
        peak_lr    = max(all_lrs[name])
        min_lr     = min(all_lrs[name])
        print(f"  {name:<35}  loss={final_loss:.4f}  "
              f"lr range: [{min_lr:.1e}, {peak_lr:.1e}]")

    # Best practice guide
    print()
    print("=" * 70)
    print("  IMPLEMENTATION CHECKLIST")
    print("=" * 70)
    checklist = [
        ("scheduler.step()", "Call AFTER optimiser.step(), once per training step"),
        ("get_last_lr()[0]", "Get current lr for logging (not param_groups[0]['lr'])"),
        ("LambdaLR",         "Most flexible — define any schedule as a lambda function"),
        ("SequentialLR",     "Chain warmup + decay schedulers cleanly"),
        ("OneCycleLR",       "Needs total_steps upfront; good for quick experiments"),
        ("warmup_steps",     "≈ 1-2% of total steps; 500-2000 for most LLM runs"),
        ("lr_min",           "Set to 10% of lr_max (e.g. 3e-5 if lr_max=3e-4)"),
        ("log lr each step", "Always log lr to W&B/TensorBoard — critical for debugging"),
    ]
    for item, note in checklist:
        print(f"  ✓  {item:<28}  {note}")
''',
    },

    "WSD Schedule and Budget-Extension Demo": {
        "description": "Show how WSD enables extending a training run without changing the schedule — train to 10k tokens, extend to 15k, compare to cosine baseline.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
WSD SCHEDULE: FLEXIBLE BUDGET EXTENSION
================================================================================

Demonstrates the practical advantage of WSD over cosine for real-world training:
    Scenario: You planned to train for 10,000 steps but have budget for 15,000.
    Problem:  With cosine, the lr is near 0 at step 10k — extending flat is bad.
    Solution: WSD — extend the stable phase, re-apply decay at the new endpoint.

Also shows the lr_min impact on final model quality.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ── Schedule functions ────────────────────────────────────────────────────────

def cosine_lr(step, total, warmup, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    if step >= total:
        return lr_min
    p = (step - warmup) / max(1, total - warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * p))


def wsd_lr(step, warmup, stable_end, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    if step < stable_end:
        return lr_max
    decay_len = total - stable_end
    p = (step - stable_end) / max(1, decay_len)
    p = min(p, 1.0)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * p))


# ── Tiny trainable model ──────────────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 32),
        )
    def forward(self, x):
        return self.layers(x)


def run_training(model, schedule_fn, n_steps, x, y):
    """Train model using the given lr schedule, return loss history."""
    opt = torch.optim.AdamW(model.parameters(), lr=1.0,  # lr=1 — schedule handles it
                             betas=(0.9, 0.999), weight_decay=0.01)
    # LambdaLR scales the base lr=1.0 by schedule_fn(step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=schedule_fn)

    losses = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == n_steps - 1:
            losses.append((step, loss.item(), sched.get_last_lr()[0]))

    return losses


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(256, 32)
    y = torch.randn(256, 32)

    LR_MAX  = 3e-4
    LR_MIN  = 3e-5
    WARMUP  = 200

    # ── Scenario 1: Planned 10k steps ─────────────────────────────────────────
    PLANNED = 2_000   # scaled down for speed
    print("=" * 65)
    print("  SCENARIO: Planned budget = 2000 steps")
    print("=" * 65)

    model_cos  = TinyLM()
    model_wsd  = copy.deepcopy(model_cos)

    cos_sched_fn = lambda s: cosine_lr(s, PLANNED, WARMUP, LR_MAX, LR_MIN) / LR_MAX
    wsd_sched_fn = lambda s: wsd_lr(s, WARMUP,
                                     int(0.7 * PLANNED),  # stable until 70%
                                     PLANNED,              # decay in last 30%
                                     LR_MAX, LR_MIN) / LR_MAX

    print(f"\n  Training both models for {PLANNED} steps...")
    losses_cos_1 = run_training(model_cos, cos_sched_fn, PLANNED, x, y)
    losses_wsd_1 = run_training(model_wsd, wsd_sched_fn, PLANNED, x, y)

    cos_final = losses_cos_1[-1]
    wsd_final = losses_wsd_1[-1]
    print(f"  Cosine final loss @ step {PLANNED}: {cos_final[1]:.5f}  lr={cos_final[2]:.2e}")
    print(f"  WSD    final loss @ step {PLANNED}: {wsd_final[1]:.5f}  lr={wsd_final[2]:.2e}")

    # ── Scenario 2: Extended to 3k steps ──────────────────────────────────────
    EXTENDED = 3_000
    print()
    print("=" * 65)
    print(f"  SCENARIO: Budget extended to {EXTENDED} steps")
    print("=" * 65)
    print()
    print("  Option A — Cosine extended flat (bad): lr stays at lr_min")
    print("  Option B — Cosine restarted (ok):      re-run cosine from scratch")
    print("  Option C — WSD extended (best):        extend stable, new decay")
    print()

    # Cosine extended: flat at lr_min after planned endpoint
    def cos_extended(s):
        if s < PLANNED:
            return cosine_lr(s, PLANNED, WARMUP, LR_MAX, LR_MIN) / LR_MAX
        return LR_MIN / LR_MAX   # flat at lr_min — bad!

    # WSD extended: longer stable phase, decay at new end
    def wsd_extended(s):
        return wsd_lr(s, WARMUP,
                      int(0.7 * EXTENDED),  # stable until 70% of NEW budget
                      EXTENDED,
                      LR_MAX, LR_MIN) / LR_MAX

    model_cos_ext  = TinyLM()   # fresh model for fair comparison
    model_wsd_ext  = copy.deepcopy(model_cos_ext)

    losses_cos_ext = run_training(model_cos_ext, cos_extended, EXTENDED, x, y)
    losses_wsd_ext = run_training(model_wsd_ext, wsd_extended, EXTENDED, x, y)

    print(f"  {'Step':>6}  {'Cosine-ext loss':>18}  {'WSD-ext loss':>14}  "
          f"{'Cosine lr':>12}  {'WSD lr':>10}")
    print(f"  {'':─>6}  {'':─>18}  {'':─>14}  {'':─>12}  {'':─>10}")
    for (s1, l1, lr1), (s2, l2, lr2) in zip(losses_cos_ext, losses_wsd_ext):
        print(f"  {s1:>6}  {l1:>18.5f}  {l2:>14.5f}  {lr1:>12.2e}  {lr2:>10.2e}")

    print()
    print("  Key observation:")
    print("  • Cosine-extended: lr drops to lr_min at step 2000, then FLAT")
    print("    → model stops learning; loss plateaus or stagnates")
    print("  • WSD-extended: continues at lr_max until step ~2100, then decays")
    print("    → model keeps exploring, then anneals properly at end")
    print()
    print("  WSD advantage: just re-run with new total_steps — no rescheduling!")

    # ── lr_min impact ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  IMPACT OF lr_min ON FINAL QUALITY")
    print("=" * 65)
    print()

    lr_min_values = [0.0, 0.05, 0.10, 0.20, 0.50]
    print(f"  {'lr_min / lr_max':>16}  {'lr_min':>10}  {'Final loss':>12}")
    print(f"  {'':─>16}  {'':─>10}  {'':─>12}")

    for ratio in lr_min_values:
        lr_min_v = LR_MAX * ratio
        m = TinyLM()
        fn = lambda s, r=ratio: cosine_lr(s, PLANNED, WARMUP,
                                          LR_MAX, LR_MAX * r) / LR_MAX
        hist = run_training(m, fn, PLANNED, x, y)
        print(f"  {ratio:>16.2f}  {lr_min_v:>10.2e}  {hist[-1][1]:>12.5f}")

    print()
    print("  Typical best: lr_min = 0.1 × lr_max (10%)")
    print("  lr_min = 0 is acceptable but may slightly underperform")
    print("  lr_min > 0.2 × lr_max leaves quality on the table")
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
    #     from llm_training.visuals.lr_schedules import (
    #         LR_VISUAL_HTML,
    #         LR_VISUAL_HEIGHT,
    #     )
    #     visual_html   = LR_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = LR_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[08_lr_schedules_warmup_cosine.py] Could not load visual: {e}", stacklevel=2)

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