"""
Training Stability
==================

A model that trains is not the same as a model that trains reliably.
Training stability is the property that a run converges to a good
solution consistently — not just once, not just with one random seed,
not just under ideal conditions. Instability wastes compute, corrupts
results, and produces models that work on the developer's machine but
fail in production. Understanding the causes of instability, reading
the signals it produces, and knowing how to restore order when training
breaks down is essential practical knowledge for anyone training models
beyond toy problems.

"""
import textwrap
import re

TOPIC_NAME   = "Training Stability"
DISPLAY_NAME = "16 · Training Stability"
ICON         = "📈"
SUBTITLE     = "Loss Curves · LR Finding · Divergence · Debugging · Reproducibility"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """


##### PART 1 — WHAT STABILITY MEANS AND WHY IT IS HARD

### Defining Training Stability

A training run is STABLE if:
1. The loss decreases monotonically (with allowable stochastic fluctuation).
2. Gradient norms remain bounded and roughly constant across depth.
3. Activation and weight statistics remain in a healthy range throughout.
4. The run converges to a similar final performance across different seeds.
5. Performance degrades gracefully under small changes to hyperparameters.

A training run is UNSTABLE when any of the following occur:
- Loss spikes: sudden large increases followed by partial or no recovery.
- Loss divergence: monotone increase that does not reverse.
- NaN propagation: loss or weights become NaN and training collapses.
- Oscillation: loss fluctuates without a downward trend.
- Silent degradation: training appears normal but the model never
  improves beyond a trivially bad baseline (e.g., always predicting
  the majority class).

    Diagram 1 — Stability Spectrum:

    Loss │
         │   A: Stable          B: Spikes         C: Divergence
         │  ╲                  ╲  ╱╲              ╲╱──────────
         │   ╲──────           ╲╱   ╲──           ╱
         │                     ↑spike              ↑ never recovers
         └────────────────────────────────── step

         │   D: Oscillation     E: Silent fail     F: NaN collapse
         │  ╲╱╲╱╲╱             ────────────────   ╲     .
         │                     (high plateau)          ↑ NaN
         └────────────────────────────────── step

    A is the goal. B is manageable. C, D, E, F require diagnosis.


### Why Instability Occurs: A Systems View

Training instability is almost never random. It is the predictable
consequence of a specific misconfiguration. The sources map onto the
components of the training system:

    Optimiser instability:
    Learning rate too large → gradient steps overshoot minima → oscillation.
    Gradient explosion → single large step destroys learned weights.
    Incorrect loss reduction (sum vs mean) → gradients scale with batch size.

    Architecture instability:
    No normalisation in deep networks → activations grow or shrink with depth.
    Poor initialisation → gradients vanish or explode from the very first step.
    Residual scaling issues → residual stream variance grows with depth.

    Data instability:
    Outlier examples → spike in loss on a single corrupted batch.
    NaN or Inf in inputs → NaN propagates through the computation graph.
    Incorrect normalisation → inputs to the model are on wildly different scales.
    Label errors or mismatched label/input indexing → loss cannot decrease.

    Precision instability:
    FP16 overflow → activations or gradients exceed the FP16 range (±65,504).
    Loss scale is too large in FP16 training → gradients overflow.
    Loss scale is too small → gradients underflow to zero.

    The diagnostic process is to ISOLATE which component is responsible.
    This requires systematic elimination, not guesswork.



##### PART 2 — READING LOSS CURVES IN DETAIL

### The Baseline Sanity Check

Before training for even one full epoch, verify that the loss starts
at the theoretically correct random-chance value.

    For cross-entropy classification with K classes:
    Expected initial loss ≈ log(K)
    (uniform prediction assigns 1/K probability to each class)

    K=2  (binary):   expected loss ≈ 0.693
    K=10 (CIFAR-10): expected loss ≈ 2.303
    K=1000 (ImageNet): expected loss ≈ 6.908

    For MSE regression:
    Expected initial loss ≈ Var(y)  (predicting the mean of y)

    If the initial loss is FAR from the expected value, there is a bug:
    Initial loss << expected: the model or data has leaked information.
                              The model is not randomly initialised.
    Initial loss >> expected: a bug in the loss function, label indexing,
                              or output normalisation.
    Initial loss = NaN:       initialisation problem, NaN in data,
                              or a computational graph issue.

    This check costs one forward pass. It catches the majority of
    setup bugs before wasting hours of compute.


### Smoothed vs Raw Loss Curves

Mini-batch loss is NOISY — each batch is a random sample from the
data distribution. The per-step loss has high variance that obscures
the underlying trend.

    Raw mini-batch loss: informative about spikes but noisy for trends.
    Exponential moving average (EMA): reveals the trend.
    Per-epoch average: smooth but masks within-epoch dynamics.

    EMA smoothing:
    smoothed_loss_t = β · smoothed_loss_{t-1} + (1-β) · raw_loss_t

    β = 0.99: very smooth, lags behind sharp changes by ~100 steps.
              Best for overall trend.
    β = 0.9:  moderately smooth, lags ~10 steps.
              Best for detecting spikes while seeing the trend.
    β = 0.0:  raw (no smoothing).

    In TensorBoard: use the smoothing slider.
    For manual plotting: rolling window mean or EMA with β≈0.95.

    Diagram 2 — Raw vs Smoothed Loss:

    Loss
      │  raw: ╱╲╱╲╱╲  ╱╲╱╲╱╲  ╱╲╱╲╱╲
      │       ╲╱╲╱╲╱  ╲╱╲╱╲╱  ╲╱╲╱╲╱
      │                                 ← noisy, trend unclear
      │  EMA: ╲────────╲────────╲──── ← trend visible, spikes smoothed
      │
      └───────────────────────────── step


### Loss Curve Anatomy: What Each Segment Reveals

A healthy training run has recognisable segments:

    Diagram 3 — Full Training Run Loss Anatomy:

    Loss
      │
      │ ①╲     ② ╲           ③ ──────── (plateau)
      │   ╲────   ╲────────╮
      │                    ╰──────────  (val loss, slight gap)
      │
      └─────────────────────────────────── Epoch
        warm  fast    gradual   converged
        up    learn   decay

    ① Warmup / initial phase:
       Loss drops rapidly from the random-chance baseline.
       Slope is steep: the model is learning obvious features.
       If LR warmup is used, the curve is initially slower.

    ② Fast learning phase:
       Steady monotone decrease. The model is building representations.
       Val loss tracks train loss closely (or within a small gap).
       The slope of the descent indicates learning rate effectiveness.

    ③ Convergence / plateau:
       Loss decrease slows or stops. The model has extracted most
       learnable signal from the data given current regularisation.
       Train and val loss diverge slightly (the generalisation gap).
       LR decay (cosine or step) should occur here to fine-tune.

    Knowing which segment you are in determines the correct action:
    Poor ①: initialisation or LR problem.
    Poor ②: LR too large or too small, regularisation mismatched.
    Poor ③: model capacity wrong, regularisation wrong, or correct.


### Diagnosing Specific Loss Curve Shapes

    SHAPE: Flat then sudden drop (delayed learning):
    ┌──────────────────────────────────────────────┐
    │ Loss │                                       │
    │      │─────────╲────────────                 │
    │      │          ╲ (sudden drop after flat)   │
    └──────────────────────────────────────────────┘
    Cause: LR warmup taking too long. Or the model is stuck in a
           flat region (saddle point) before finding the gradient.
    Fix:   Shorten warmup. Check for vanishing gradients in early layers.

    SHAPE: Fast drop then plateau at high loss (underfitting):
    ┌──────────────────────────────────────────────┐
    │ Loss │ ╲                                     │
    │      │  ╲────────────────────── (too high)   │
    └──────────────────────────────────────────────┘
    Cause: Model too small, LR too small, or training stopped too early.
    Fix:   Increase model capacity. Increase LR. Train longer.

    SHAPE: Oscillating (high-frequency zigzag):
    ┌──────────────────────────────────────────────┐
    │ Loss │ ╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱                      │
    └──────────────────────────────────────────────┘
    Cause: LR too large (steps overshoot the minimum repeatedly).
    Fix:   Reduce LR by 2-10×. Add LR warmup. Add gradient clipping.

    SHAPE: Periodic bumps (every N epochs):
    ┌──────────────────────────────────────────────┐
    │ Loss │ ╲  ╱╲  ╱╲  ╱ (regular bumps)          │
    │      │  ╲╱  ╲╱  ╲╱                           │
    └──────────────────────────────────────────────┘
    Cause: LR schedule with warm restarts (SGDR). Expected behaviour.
           Or: dataset has periodic structure correlated with epoch length.
    Fix:   If unintended, disable warm restarts. Check for data ordering.

    SHAPE: Good train loss, diverging val loss (classic overfit):
    ┌──────────────────────────────────────────────┐
    │ Loss │ train: ╲────────                      │
    │      │ val:   ╲───╱────── (val rises)        │
    └──────────────────────────────────────────────┘
    Cause: Model overfitting (high variance).
    Fix:   Add regularisation, more data, early stopping.

    SHAPE: Single spike then recovery:
    ┌───────────────────────────────────────────────┐
    │ Loss │ ╲─────╱╲────────── (spike then resume) │
    └───────────────────────────────────────────────┘
    Cause: Single corrupted batch, gradient explosion on rare example.
    Fix:   Add gradient clipping. Check dataset for corrupted files.
           Filter examples with extreme loss values.

    SHAPE: Gradual drift upward after initial convergence:
    ┌────────────────────────────────────────────────┐
    │ Loss │ ╲──────╲───────── (slow rise after min) │
    └────────────────────────────────────────────────┘
    Cause: Distribution shift in data stream (online learning setting).
           Or: catastrophic forgetting in continual learning.
    Fix:   Add replay buffer. Check for distributional shift in batches.



##### PART 3 — THE LEARNING RATE AS THE PRIMARY STABILITY LEVER

### Why LR Is the First Thing to Adjust

Of all the hyperparameters that affect stability, the learning rate has
the most direct relationship with the optimiser's behaviour:

    Too large:   steps overshoot → oscillation → divergence.
    Too small:   steps too tiny → no learning → plateau.
    Just right:  monotone convergence to a flat, generalisable minimum.

    The sensitivity is multiplicative and spans several orders of magnitude:
    LR = 1.0  → almost always diverges (for Adam with standard scale)
    LR = 0.1  → likely diverges or oscillates
    LR = 0.01 → often works but may be slow
    LR = 0.001 → sweet spot for Adam on most tasks
    LR = 0.0001 → often works but may be slow
    LR = 0.00001 → almost certainly too slow (learning rate too small)

    The 5-orders-of-magnitude range means a naive guess has perhaps a
    20% chance of being in the productive region. The LR finder
    (Module 15) narrows this to a principled choice in minutes.


### The LR-Stability Phase Diagram

For a fixed architecture and dataset, the loss landscape has a
STABILITY REGION — a range of LR values where training converges:

    Diagram 4 — LR Phase Diagram:

    Final validation loss
      │
    High│   divergence       productive region    too slow
      │ │───────────────╮                    ╭──────────────
      │ │                ╲  ╭─────────────╮  ╲
      │ │                 ╲─╯             ╰───
      │ │                        ↑
    Low │                   optimal LR
      └─────────────────────────────────────────── LR (log scale)
              1e-1       1e-3         1e-5

    Left of the stability boundary: divergence.
    Within the productive region: loss decreases, lower LR → lower final loss.
    Right of the productive region: loss barely decreases (too slow to converge).
    The stability boundary moves left (lower max stable LR) as the model gets:
    - Deeper (more layers to multiply gradients through)
    - Wider (larger gradient magnitudes per step)
    - More poorly conditioned (imbalanced loss landscape)


### Learning Rate and Batch Size Interaction

As discussed in Module 14 (Data Pipeline), the optimal LR scales with
batch size. This interacts directly with stability:

    LINEAR SCALING RULE: LR_optimal ≈ LR_base × (B / B_base)

    If you increase batch size by 4× without scaling LR, the effective
    LR is too small — training is unnecessarily slow.
    If you increase batch size by 4× and scale LR by 4× without warmup,
    the first steps have a 4× larger update with poorly conditioned
    gradients — training may diverge.

    The LINEAR SCALING + WARMUP protocol:
    1. Set LR_target = LR_base × (B / B_base)
    2. Start at LR_start = LR_base  (or smaller)
    3. Linearly ramp from LR_start to LR_target over warmup_steps
    4. Then apply the chosen LR decay schedule from LR_target downward.

    This sequence is the standard protocol for all large-batch training
    and is used by every major production training system.


### Per-Parameter Learning Rate Adaptation (Adam, RMSProp)

Adaptive optimisers like Adam divide the global LR by a per-parameter
scale derived from gradient history:

    Effective LR for parameter i:
    η_eff_i = η / (sqrt(v̂ᵢ) + ε)

    Where v̂ᵢ = exponential moving average of squared gradients for param i.

    If v̂ᵢ is large (parameter has seen large gradients historically):
    η_eff_i is small → this parameter takes small steps → stable.

    If v̂ᵢ is small (parameter has seen small gradients historically):
    η_eff_i is large → this parameter takes large steps → potentially unstable.

    This per-parameter adaptation is what makes Adam more stable than SGD
    across many hyperparameter settings. But it also means:

    1. The ε parameter matters more than you might expect.
       Default ε = 1e-8. If gradients are very small (e.g., in the
       early steps of a poorly scaled network), v̂ᵢ ≈ 0 and the effective
       LR is ≈ η/ε = η × 10^8 — enormous. This is why warmup is critical.

    2. Very different gradient magnitudes across layers can cause some
       layers to receive enormous effective LR while others receive tiny LR.
       This is one reason why normalisation (LayerNorm, BatchNorm) stabilises
       adaptive optimiser training — it equalises gradient magnitudes.

    3. Adam can fail on non-stationary objectives because v̂ᵢ accumulates
       history. In transfer learning with domain shift, old gradient history
       may be irrelevant. Solution: reset Adam state at domain boundaries,
       or use a short warmup after loading a checkpoint.


##### PART 4 — DIAGNOSING AND FIXING DIVERGENCE

### Systematic Divergence Diagnosis

When a training run diverges, work through this elimination sequence:

    STEP 1 — Check if the first step is already bad:
    Print loss after exactly ONE gradient step.
    Expected: loss should be close to the random-chance baseline.
    If loss explodes immediately: initialisation bug, LR too large,
    or NaN in input data.

    STEP 2 — Check for NaN in data:
    import torch
    for batch in train_loader:
        if torch.isnan(batch['inputs']).any() or torch.isnan(batch['labels']).any():
            print("NaN found in batch")
            break
    NaN in data propagates silently through the network and produces
    NaN loss. A single corrupted file can kill a multi-day training run.

    STEP 3 — Verify the loss function:
    Check that:
    - Loss output is a scalar (not a tensor requiring manual .mean() call)
    - Loss is using mean reduction (not sum, which scales with batch size)
    - The output logits and target labels have matching shapes and dtypes
    - Labels are 0-indexed (not 1-indexed) for CrossEntropyLoss

    Common mistakes:
    nn.CrossEntropyLoss expects (N, C) logits and (N,) integer labels.
    If you pass (N, C) labels (one-hot), you get wrong gradients silently.

    STEP 4 — Reduce LR by 10×:
    The single most common cause of divergence is LR too large.
    Try LR × 0.1 before investigating anything else.
    If the run now converges: LR was the problem.

    STEP 5 — Overfit one batch:
    Overfit the model to a single batch of 10-32 examples:
    for step in range(1000):
        loss = model(single_batch)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        print(f"step {step}: loss={loss.item():.6f}")

    This should converge to near-zero loss within 100-500 steps.
    If it does NOT converge:
    - Bug in the model (wrong architecture, wrong activation)
    - Bug in the loss function (wrong reduction, wrong output shape)
    - LR too small to converge even on a tiny problem
    - Model cannot represent the target function (capacity issue)

    If it DOES converge: the model is capable of learning. The
    divergence on the full dataset is a data or regularisation issue.

    STEP 6 — Enable gradient anomaly detection:
    torch.autograd.set_detect_anomaly(True)
    This makes PyTorch raise an exception at the first NaN or Inf
    in the computation graph, with a traceback showing exactly where
    the numerical error occurred. Expensive (2-3× slower) — use only
    for debugging.

    STEP 7 — Check gradient norms before and after clipping:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"grad norm: {grad_norm:.4f}")
    If grad_norm >> 1.0 on most steps: gradient explosion is the cause.
    Fix: add gradient clipping (or it's already there — reduce LR).


### The Batch Sanity Check

When the loss curve looks completely wrong (loss not decreasing at all
or decreasing very slowly on what should be an easy task):

    Test 1 — Does the model output the right shape?
    outputs = model(batch['inputs'])
    print(f"output shape: {outputs.shape}")
    print(f"target shape: {batch['labels'].shape}")

    Test 2 — Does the loss decrease when you intentionally overfit?
    (See STEP 5 above — the single-batch overfit test.)

    Test 3 — Is the target the right type?
    CrossEntropyLoss: targets must be torch.long (int64)
    BCEWithLogitsLoss: targets must be torch.float32
    MSELoss: targets must be torch.float32

    Test 4 — Is the data loaded correctly?
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    img_grid = make_grid(batch['inputs'][:16])
    plt.imshow(img_grid.permute(1,2,0))  # Visualise the first batch
    Labels = batch['labels'][:16]        # Verify they match the images

    Visualising the first batch is one of the most effective debugging
    steps available. It reveals:
    - Incorrect normalisation (images appear black, all-white, or inverted)
    - Wrong augmentation (images are corrupted or unrecognisable)
    - Label mismatch (label says "cat" but image shows a dog)
    - Shape mismatch (images are transposed or have wrong channels)


### Loss Scale Bugs (Batch Size)

A subtle and common instability cause: summing instead of averaging
the loss across the batch:

    WRONG (sum reduction — scales with batch size):
    loss = criterion(outputs, targets)   # criterion uses sum reduction
    loss.backward()

    With B=256, the gradient is 256× larger than intended.
    If the LR was tuned with B=32: effective LR is now 8× too large.
    Training diverges as batch size increases.

    CORRECT (mean reduction — batch size invariant):
    criterion = nn.CrossEntropyLoss(reduction='mean')  # default
    loss = criterion(outputs, targets)
    loss.backward()

    Verifying reduction mode:
    print(criterion.reduction)  # should be 'mean'

    For custom losses:
    loss = ((outputs - targets) ** 2).sum()   # WRONG — scales with B
    loss = ((outputs - targets) ** 2).mean()  # CORRECT


##### PART 5 — NUMERICAL PRECISION AND MIXED PRECISION STABILITY

### FP16 Precision Hazards

FP16 (float16) stores numbers in 16 bits with a range of ±65,504.
Gradient values during training can exceed this:

    FP32 can represent: ±3.4 × 10^38
    FP16 can represent: ±65,504 ≈ 6.5 × 10^4

    When a gradient value exceeds ±65,504:
    FP16 represents it as ±Inf  (overflow)
    Inf propagates through all subsequent operations as NaN

    When a gradient value is smaller than the smallest FP16 value (≈ 6×10^-8):
    FP16 represents it as 0  (underflow)
    The gradient signal is lost → effectively zero gradient for that parameter

    Both overflow and underflow cause training instability.
    FP16 training without loss scaling fails silently or diverges.


### Loss Scaling for FP16

The solution: LOSS SCALING. Multiply the loss by a large scale factor S
before calling loss.backward(). This multiplies all gradients by S,
shifting them into the FP16 representable range. Unscale before the
optimiser step.

    Forward:  loss_scaled = loss × S
    Backward: gradients are S× larger (fit in FP16)
    Before step: gradient ← gradient / S  (unscale)
    Step: weight -= η × (gradient / S)

    Dynamic loss scaling (the standard approach):
    Start with a large S (e.g., S=65536).
    If any gradient is Inf or NaN: skip the step, halve S.
    If no Inf/NaN for N steps: double S.
    S adapts to the gradient magnitudes seen during training.

    In PyTorch:
    scaler = torch.cuda.amp.GradScaler()

    # Training step:
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.unscale_(optimiser)                    # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimiser)
    scaler.update()                               # update S

    The scaler.update() call adjusts S based on whether any Inf/NaN
    were detected in the current step.


### BF16: The Better Alternative

BF16 (bfloat16) stores numbers in 16 bits with the SAME exponent as FP32:

    FP32:  1 sign bit | 8 exponent bits | 23 mantissa bits → range ±3.4×10^38
    FP16:  1 sign bit | 5 exponent bits | 10 mantissa bits → range ±65,504
    BF16:  1 sign bit | 8 exponent bits |  7 mantissa bits → range ±3.4×10^38

    BF16 has the same dynamic range as FP32 but less precision per value.
    For training, dynamic range matters far more than precision:
    - Gradient values can be very large or very small → need large range.
    - The exact decimal precision of a gradient is less critical.

    BF16 training requires NO LOSS SCALING — it cannot overflow where
    FP32 would be fine. This dramatically simplifies the training loop.

    BF16 is supported on:
    NVIDIA Ampere GPUs (A100, A10, A30) and newer
    NVIDIA Hopper GPUs (H100) — also supports FP8
    Google TPU v3 and v4 (natively BF16)

    Modern recommendation:
    If BF16 is available on your hardware: use BF16, no loss scaling needed.
    If only FP16 is available: use FP16 with GradScaler.
    If neither: use FP32 (most stable, least efficient).

    torch.autocast(device_type='cuda', dtype=torch.bfloat16)


### Numerical Precision Checklist

    □ Use BF16 if available (A100/H100/TPU). No loss scaling needed.
    □ Use FP16 with GradScaler on older GPUs.
    □ Keep weight master copies in FP32 (mixed precision: compute in FP16,
      store weights in FP32, update in FP32).
    □ Keep loss computation in FP32 (autocast exits for loss function
      by default in PyTorch's autocast implementation).
    □ Monitor the loss scale: if it drops below 1.0, overflow is frequent.
      Reduce LR or check for problematic layers.
    □ If training in FP16 and loss is NaN: GradScaler is not active,
      or a layer is computing outside the autocast context.
    □ Keep normalisation (LayerNorm, BatchNorm) in FP32 even in FP16 mode.
      PyTorch autocast does this automatically for these modules.


##### PART 6 — ACTIVATION AND WEIGHT MONITORING

### What to Monitor and Why

Loss alone is a coarse signal. The following statistics, logged
periodically during training, give early warning of instability
before it becomes catastrophic:

    GRADIENT NORM (global, per-layer):
    Total gradient norm = sqrt(Σᵢ ||∇ᵢL||²)
    Should be: roughly constant, bounded between 0.01 and 10.
    Rising trend: explosion risk.
    Falling to zero: vanishing gradient or dead neurons.
    Per-layer norms should be similar; monotone decrease with depth = vanishing.

    ACTIVATION STATISTICS (mean and std per layer):
    Mean: should be near zero (especially after normalisation).
          Large mean: bias terms or activations are all positive/negative.
    Std: should be O(1) — not growing, not shrinking with depth.
         Std → 0: activation collapse (vanishing).
         Std >> 1: activation explosion.

    WEIGHT STATISTICS (mean and std per parameter group):
    Weights should not grow unboundedly.
    Std growing continuously: weight explosion despite gradient clipping
    (LR is too large relative to the clipping threshold).

    GRADIENT-TO-WEIGHT RATIO (update ratio):
    update_ratio = (η × grad_norm) / weight_norm
    Target: ~1e-3 (weights change by ~0.1% per step).
    >> 1e-2: updates too large, instability risk.
    << 1e-4: updates too small, learning stalled.
    Proposed and used by Andrej Karpathy as a primary diagnostic.

    LOSS SCALE (for FP16 training):
    Should stay in a healthy range (typically 128 to 65536).
    Continuously decreasing → frequent overflow, instability.
    Bottoming out at 1.0 → training is barely stable.


### Implementing Activation Monitoring with Hooks

PyTorch forward hooks allow inserting monitoring code at any layer
without modifying the model:

    class ActivationMonitor:
        def __init__(self, model, writer, log_interval=500):
            self.writer = writer
            self.step = 0
            self.log_interval = log_interval
            self.hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d,
                                        nn.LayerNorm, nn.ReLU, nn.GELU)):
                    hook = module.register_forward_hook(
                        self._make_hook(name)
                    )
                    self.hooks.append(hook)

        def _make_hook(self, name):
            def hook(module, input, output):
                if self.step % self.log_interval == 0:
                    out = output.detach().float()
                    self.writer.add_scalar(f'activation/mean/{name}',
                                           out.mean(), self.step)
                    self.writer.add_scalar(f'activation/std/{name}',
                                           out.std(), self.step)
                    self.writer.add_scalar(f'activation/max_abs/{name}',
                                           out.abs().max(), self.step)
            return hook

        def step_end(self):
            self.step += 1

        def remove(self):
            for h in self.hooks:
                h.remove()


### Healthy vs Unhealthy Statistics: Reference Ranges

    ┌───────────────────────────────────────────────────────────────────────┐
    │ Statistic           │ Healthy range    │ Warning           │ Critical │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Global gradient norm│ 0.05 – 5.0       │ > 10 or < 0.001   │ > 100    │
    │ (pre-clipping)      │                  │                   │          │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Activation std      │ 0.5 – 2.0        │ < 0.1 or > 5.0    │ < 0.01   │
    │ (post-LayerNorm)    │ (should be ~1)   │                   │  or > 50 │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Activation std      │ 0.5 – 3.0        │ < 0.1 or > 10     │ < 0.01   │
    │ (no normalisation)  │                  │                   │  or Inf  │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Weight std          │ 0.01 – 0.5       │ Growing trend     │ > 10     │
    │                     │                  │ consistently      │          │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Update ratio        │ 1e-4 – 1e-2      │ > 1e-1 or < 1e-5  │ > 1.0    │
    │ η·grad/weight       │                  │                   │          │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Loss scale (FP16)   │ 128 – 65536      │ < 64 or declining │ = 1.0    │
    └───────────────────────────────────────────────────────────────────────┘


##### PART 7 — REPRODUCIBILITY

### Why Reproducibility Is Part of Stability

A run that succeeds with one random seed but fails with another is
UNSTABLE — it is sensitive to initialisation in a way that indicates
the training dynamics are fragile. True stability means the run
converges reliably across multiple seeds, within an acceptable
variance band.

    Reproducibility serves two purposes:
    1. DEBUGGING: if you cannot reproduce a failure, you cannot fix it.
    2. REPORTING: claimed results must be reproducible by others.

    Variance across seeds is a real effect:
    A model trained 5 times with different seeds may produce validation
    accuracies of [84.1, 84.5, 83.9, 84.3, 84.7]. The mean ± std is
    84.3 ± 0.3%. Reporting a single run at 84.7% is misleading.
    Results should always report mean ± std across seeds.


### Sources of Non-Reproducibility

    1. Random weight initialisation:
       Different seeds → different starting weights → different solutions.
       This is the INTENDED source of variance and is acceptable.
       Not a bug — it is the seed-to-seed variance you report.

    2. Stochastic data ordering:
       Different random shuffles → different batch compositions.
       Fix: set seed for the DataLoader sampler.

    3. Dropout and data augmentation:
       Random masking and transforms use the global random state.
       Fix: seed the RNG before training.

    4. CUDA non-determinism:
       GPU operations (cuDNN convolution, atomics in scatter) are
       non-deterministic even with the same seed because of parallel
       execution order.
       Fix: torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark    = False
       Cost: significant (10-30%) slowdown. Use only when exact
       reproducibility is required.

    5. Multi-GPU training:
       Gradient synchronisation order can vary between runs.
       Floating-point addition is not associative — different summation
       orders produce slightly different results.
       Fix: use deterministic collective operations (NCCL deterministic mode).

    6. num_workers > 0 in DataLoader:
       Workers are separate processes with independent RNG state.
       Without seeding workers, they use the same seed as the main process
       → all workers produce the same random augmentations.
       Fix: use worker_init_fn to set unique seeds per worker.


### The Reproducibility Seed Setup

    def set_seed(seed: int, deterministic: bool = False):
        '''
        Set all random seeds for reproducibility.

        Args:
            seed:          The global random seed.
            deterministic: If True, forces exact CUDA reproducibility
                           at the cost of ~20% training slowdown.
        '''
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)    # for multi-GPU

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
            torch.use_deterministic_algorithms(True)
        else:
            # Allow non-determinism for speed, but seed the RNG
            torch.backends.cudnn.benchmark = True  # benchmark=True finds
                                                    # fastest convolution algo
                                                    # May vary between runs

    # Usage:
    set_seed(42, deterministic=False)  # for speed (acceptable variation)
    set_seed(42, deterministic=True)   # for exact reproducibility (debugging)


### Measuring Seed-to-Seed Variance

Before reporting results or comparing models, characterise the
seed-to-seed variance:

    1. Train the model with k=3 to 5 different seeds:
       seeds = [42, 123, 456, 789, 1024]
       results = []
       for seed in seeds:
           set_seed(seed)
           model = build_model()
           train(model, ...)
           results.append(evaluate(model, test_loader))

    2. Report mean ± std:
       mean_acc = np.mean(results)
       std_acc  = np.std(results)
       print(f"Accuracy: {mean_acc:.2f} ± {std_acc:.2f}")

    3. Determine if a difference between two models is real:
       Use a two-sample t-test or Mann-Whitney U test on the
       seed distributions.
       If p > 0.05: the difference is not statistically significant.
       Do NOT claim model B is better than model A based on a single seed.

    Rule of thumb: if std > 0.5% on a classification task, differences
    smaller than 1.5× std cannot be claimed with confidence.
    Always run at least 3 seeds when comparing methods.


##### PART 8 — DEBUGGING WORKFLOW: FROM SYMPTOM TO FIX

### The "Make It Work, Make It Right, Make It Fast" Protocol

A principled debugging order prevents wasting time on wrong hypotheses:

    PHASE 1 — MAKE IT WORK (verify correctness):
    Goal: confirm the model can learn SOMETHING.
    Method: overfit a single batch of 10-32 examples to near-zero loss.
    If this fails: there is a fundamental bug (model architecture, loss,
    data loading). Fix this before anything else.
    If this succeeds: the model is capable of learning.

    PHASE 2 — MAKE IT RIGHT (verify generalisation):
    Goal: confirm the model learns on a small training subset and
          generalises to a small validation subset.
    Method: train on 100-1000 examples for a full number of epochs.
    Expected: train loss drops, val loss follows.
    If train loss does not drop: LR too small, vanishing gradients,
    or the dataset is too hard for the model capacity.
    If train loss drops but val loss is very high: overfitting (normal
    for small data — add regularisation or more data).

    PHASE 3 — MAKE IT FAST (scale to full dataset):
    Only scale to the full dataset after phases 1 and 2 succeed.
    Profile the pipeline (GPU utilisation, DataLoader throughput).
    Scale up batch size, apply mixed precision, add gradient accumulation.

    This protocol prevents a common mistake: spending 3 days debugging
    at full scale when the bug would have been caught in 5 minutes
    with a single-batch overfit test.


### Step-by-Step Debugging Checklist

    BEFORE TRAINING:
    □ Verify the initial loss equals log(K) for K-class cross-entropy.
    □ Print output and target shapes after the first batch.
    □ Visualise the first batch (images, sequence, tabular rows).
    □ Confirm labels are 0-indexed.
    □ Check that augmentation is not applied at validation time.
    □ Verify model.train() is called before training, model.eval() before val.

    FIRST TRAINING STEPS:
    □ Run the single-batch overfit test.
    □ Check gradient norms after the first backward pass.
    □ Verify loss is decreasing after 10 steps (even if slowly).
    □ Enable torch.autograd.set_detect_anomaly(True) if loss is NaN.

    ONGOING MONITORING:
    □ Log gradient norms every step (cheap).
    □ Log activation statistics every 100-500 steps (moderate cost).
    □ Log gradient-to-weight ratios every 500 steps.
    □ Log loss scale (if using FP16 + GradScaler).
    □ Check val loss after every epoch. Compare to train loss.

    WHEN SOMETHING GOES WRONG:
    □ Reduce LR by 10×. Does it fix the problem?
    □ Add gradient clipping (max_norm=1.0). Does it fix the problem?
    □ Check for NaN in inputs: torch.isnan(batch).any().
    □ Check for NaN in outputs: torch.isnan(outputs).any().
    □ Run torch.autograd.set_detect_anomaly(True) to trace NaN origin.
    □ Look at raw per-batch losses — is there a systematic outlier batch?
    □ Reduce model depth by half. Does it stabilise?
    □ Switch to a simpler model (linear → 2-layer MLP → full model).


### Systematic Isolation: The Binary Search for Bugs

When a run that used to work stops working (after a code change):

    Binary search across recent commits:
    Identify the last commit where training was known to work.
    Identify the current broken state.
    Binary search the commit history to find the first bad commit.

    When no commit history is available (new setup):
    Start with the simplest possible version:
    - Single linear layer, random data
    - Then add the model, keep random data
    - Then add the real dataset, keep simple model
    - Then combine full model + real dataset

    At each step, verify training stability. The step that breaks
    training identifies the component responsible.

    Common components to isolate:
    Model architecture vs. Loss function vs. Dataloader vs. Optimiser
    Each can be swapped for a known-good reference implementation.


##### PART 9 — STABILITY ACROSS THE FULL TRAINING LIFECYCLE

### Stability During Fine-tuning

Fine-tuning a pre-trained model introduces unique stability concerns:

    CATASTROPHIC FORGETTING:
    Fine-tuning with a large LR and many steps causes the model to
    overwrite its pre-trained representations entirely.
    The model converges to the fine-tuning task but loses its prior
    general representations.

    Symptom: val loss on fine-tuning task is good, but the model
    no longer performs on the original task (if measured).

    Mitigation strategies:
    1. Small LR for fine-tuning: typically 10-100× smaller than pre-training LR.
       Common: 3e-4 (pre-training) → 2e-5 (fine-tuning).
    2. Discriminative learning rates (ULMFiT approach):
       Use lower LR for early layers (closer to input) and higher LR
       for later layers (task-specific head).
       Early layers: LR × 0.01 to LR × 0.1 of the base LR.
       Later layers: LR × 1.0 (full LR).
    3. LoRA / adapter methods:
       Freeze the pre-trained weights entirely. Only train small adapter
       modules inserted into each layer. No catastrophic forgetting.

    INITIAL INSTABILITY FROM RANDOM HEAD:
    When fine-tuning a pre-trained model on a new task, the classification
    head is randomly initialised. Its output is random → loss is high →
    gradient for the head is large → can corrupt the pre-trained backbone
    if the LR is large.
    Fix: use LR warmup. Or freeze the backbone for the first few epochs,
    train only the head, then unfreeze and train everything with a small LR.

    ADAM STATE MISMATCH:
    If loading a checkpoint trained with Adam and continuing training on
    a different dataset, the saved Adam state (momentum estimates) reflects
    the OLD dataset. These stale momentum estimates may cause instability.
    Fix: reset the Adam state (do not load the optimiser state from the
    checkpoint) and add LR warmup for the first 1-5% of fine-tuning steps.


### Stability at Very Large Scale

Large-scale pre-training (100B+ parameter models) introduces stability
challenges that smaller models rarely encounter:

    LOSS SPIKES (LLM training):
    Even well-configured large runs experience occasional loss spikes —
    single or few-step increases of 0.1-0.5 nats followed by recovery.
    These are expected and generally harmless.

    Catastrophic spikes (>1 nat, no recovery): caused by bad batches,
    corrupted data, or gradient explosion. Typically handled by:
    - Reverting to the most recent checkpoint before the spike.
    - Skipping or filtering the problematic batch.
    - Reducing LR temporarily.

    DATA QUALITY AT SCALE:
    Web-scale datasets contain a small fraction of highly pathological
    examples (extremely long sequences, all-identical tokens, encoding
    errors, adversarial content). At 1 trillion tokens, even a 0.001%
    pathological rate means 10 billion bad tokens.
    Standard practice: statistical filtering, deduplication, quality
    scoring, and removing examples with anomalous loss values.

    CHECKPOINT FREQUENCY:
    At large scale, a training failure that loses 12 hours of compute
    is costly. Standard practice: checkpoint every 100-500 steps.
    Asynchronous checkpointing (saving in the background while training
    continues) avoids adding overhead to the critical path.


### The Stability Manifest: Rules for a Stable Training Run

    Architecture:
    □ Use residual connections for depth > 6 layers.
    □ Use LayerNorm (pre-norm placement for transformers).
    □ Use ReLU, GELU, or Swish — not sigmoid/tanh in hidden layers.
    □ Use He (ReLU) or Xavier (tanh) initialisation. Scaled init for LLMs.
    □ Keep model output in the same scale as the target (no sigmoid squeeze
      when targets span [0, 100]).

    Optimiser:
    □ Use AdamW (not Adam) with weight_decay applied only to weights.
    □ Use gradient clipping (max_norm=1.0 for transformers, up to 5.0 for others).
    □ Use LR warmup for the first 1-5% of training steps.
    □ Use cosine annealing or ReduceLROnPlateau for LR decay.
    □ Monitor gradient norm and update ratio every step.

    Data:
    □ Validate data for NaN/Inf before training.
    □ Use mean loss reduction (not sum).
    □ Verify labels are correctly indexed and shaped.
    □ Apply normalisation consistently (train stats → val and test).

    Precision:
    □ Use BF16 if hardware supports it (A100/H100/TPU).
    □ Use FP16 + GradScaler if BF16 is unavailable.
    □ Monitor loss scale (FP16). If falling: reduce LR.

    Reproducibility:
    □ Set random seeds for Python, NumPy, PyTorch, CUDA.
    □ Report mean ± std across ≥ 3 seeds.
    □ Log configuration (all hyperparameters, model architecture, data split)
      with each run for future comparison.
    □ Save full config alongside each checkpoint.

"""


# ─────────────────────────────────────────────────────────────────────────────
# No OPERATIONS (theory-only module)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has ~20 leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None # _MAIN_SCRIPT

    scripts_available = main_script.exists()

    if "tok_step_status"  not in st.session_state:
        st.session_state.tok_step_status  = {}
    if "tok_step_outputs" not in st.session_state:
        st.session_state.tok_step_outputs = {}

    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**{op_data['description']}**")
            st.markdown("---")
            st.code(op_data["code"], language=op_data.get("language", "python"))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────


def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 1150
    try:
        from Training_Core.visuals.Training_stability_visual import (
            TRAINING_STABILITY_VISUAL_HTML,
            TRAINING_STABILITY_VISUAL_HEIGHT,
        )
        visual_html   = TRAINING_STABILITY_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = TRAINING_STABILITY_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    OPERATIONS,
    }