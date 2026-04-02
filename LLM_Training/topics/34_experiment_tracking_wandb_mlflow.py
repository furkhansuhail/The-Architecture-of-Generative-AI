"""
Experiment Tracking: W&B, MLflow, and TensorBoard
===================================================

Large-scale training experiments are complex, long-running, and expensive.
Without systematic tracking, it is nearly impossible to understand why one
run outperformed another, reproduce a key result, or efficiently search
the hyperparameter space. Experiment tracking systems — W&B, MLflow, and
TensorBoard — provide the infrastructure to log, compare, and analyse
training runs. This module covers what to track, how to track it efficiently,
and the deeper interpretability patterns that separate actionable logging
from noise.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Experiment Tracking"
DISPLAY_NAME = "34 · Experiment Tracking"
ICON         = "📊"
SUBTITLE     = "W&B, MLflow, and TensorBoard"


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

### Why Experiment Tracking Matters

A single LLM pre-training run involves dozens of hyperparameter decisions:
learning rate, warmup schedule, batch size, weight decay, gradient clipping,
data mix, architecture choices. Changing any one can significantly affect
final model quality. Without tracking, you will inevitably:

    •   Re-run experiments you've already run (wasting GPU-hours)
    •   Forget which configuration produced your best result
    •   Be unable to explain why a result was good or bad
    •   Lose the ability to reproduce a published or deployed result

**The minimum viable tracking rule:**
Every training run should have a unique identifier, and every metric and
configuration that could explain its behaviour should be recorded.

The cost of tracking is ~0% of training time (background async logging).
The cost of NOT tracking is measured in weeks of lost work and GPU dollars.


### What to Track: A Hierarchy of Signal

Not all metrics are equally informative. A useful mental model is to think
in tiers:

**Tier 1 — Training health (check every 10–100 steps):**
    •   train_loss:        Raw cross-entropy on training batch
    •   grad_norm:         Global gradient norm (should stay in [0.1, 10])
    •   learning_rate:     Current LR (verify schedule is applied correctly)
    •   loss_scale:        AMP GradScaler's current scale (should stay large)
    •   gpu_utilisation:   MFU proxy (should be > 40% for healthy training)

**Tier 2 — Training dynamics (check every 100–1000 steps):**
    •   val_loss:          Held-out validation perplexity
    •   tokens_per_second: Overall throughput metric
    •   param_grad_ratio:  Mean |param| / mean |grad| (stability indicator)
    •   layer_grad_norms:  Per-layer gradient norms (identifies dead layers)
    •   weight_norms:      Per-layer weight norms (growing = instability risk)

**Tier 3 — Quality benchmarks (check every 10k–100k steps):**
    •   perplexity:        On held-out text (WikiText-103, etc.)
    •   few_shot_accuracy: HellaSwag, ARC, MMLU (downstream capability proxy)
    •   code_eval:         HumanEval pass@k (if training on code)

**Tier 4 — System metrics (check every 1–10 steps):**
    •   gpu_memory_allocated: MB of GPU HBM used
    •   gpu_memory_reserved:  MB reserved (fragmentation indicator)
    •   cpu_memory_rss:       CPU RAM usage (for offloading setups)
    •   disk_io_bytes:        Data loading bottleneck indicator
    •   step_time_ms:         Wall-clock time per step


### The Three Systems: W&B, MLflow, TensorBoard

**Weights & Biases (W&B):**
    •   Cloud-based SaaS with a rich web UI
    •   Excellent experiment comparison (parallel coordinates, scatter plots)
    •   Hyperparameter sweeps (Bayesian optimisation built-in)
    •   Artifact versioning (model checkpoints, datasets)
    •   Team features (sharing runs, commenting, alerts)
    •   Cost: free for individuals, paid for teams/large data

    Best for: collaborative research, hyperparameter search, experiment comparison
    API:      `import wandb; wandb.init(); wandb.log({"loss": 0.5})`

**MLflow:**
    •   Open-source, self-hosted or Databricks cloud
    •   MLflow Tracking: experiment logging
    •   MLflow Models: model packaging and registry
    •   MLflow Projects: reproducible experiment definitions
    •   MLflow Registry: model lifecycle management (staging, production)

    Best for: enterprise MLOps, model deployment workflows, self-hosted
    API:      `import mlflow; mlflow.log_metric("loss", 0.5, step=100)`

**TensorBoard:**
    •   Open-source, Google, runs as a local web server
    •   Originally designed for TensorFlow, fully supports PyTorch
    •   Excellent for: loss curves, image samples, histograms, profiling
    •   No cloud infrastructure needed
    •   Can be embedded in Jupyter notebooks

    Best for: local development, quick visual inspection, compute clusters
    API:      `from torch.utils.tensorboard import SummaryWriter; writer.add_scalar("loss", 0.5, 100)`


### Logging Patterns: What, When, and How Often

**The logging overhead problem:**
Logging every step to W&B/MLflow sends an HTTP request per log call.
At 10 ms/step, logging every step adds ~1% overhead. Logging every 10 steps
drops this to 0.1%.

But more critically: if the logger is synchronous, a slow network connection
or API rate limit can stall training. Always use:
    •   Asynchronous logging (W&B and MLflow both support this)
    •   Local buffering with periodic flush
    •   Log less frequently for expensive metrics (validation perplexity)

**The logging frequency matrix:**

    Metric group            Frequency    Why
    ──────────────────────────────────────────────────────────────────
    Step metrics (loss, LR) Every step   Fast, critical for monitoring
    Gradient norms          Every 10     Slightly expensive, important
    Validation loss         Every 100    Expensive (requires full eval pass)
    Benchmarks (MMLU, etc.) Every 10k    Very expensive (1+ GPU-hours)
    System metrics          Every step   Cheap (pre-computed)
    Histograms/distributions Every 100   Expensive (large data to send)
    ──────────────────────────────────────────────────────────────────


    **Diagram 1 — Experiment Tracking Architecture:**

    EXPERIMENT TRACKING DATA FLOW
    ════════════════════════════════════════════════════════════════

    Training loop (GPU)
        │
        ├─ every step: {loss, lr, grad_norm, step_time}
        │   → buffer in CPU RAM
        │
        ├─ every 10 steps: flush buffer → async HTTP → W&B/MLflow server
        │
        ├─ every 100 steps: validation loss → eval GPU → log
        │
        └─ every checkpoint: save artifact → W&B artifact store / MLflow model registry

    W&B/MLflow server (cloud or local)
        ├─ Real-time dashboard
        ├─ Alert on anomalies (loss spike, training stall)
        └─ Compare runs side-by-side


### Gradient and Weight Monitoring: Early Warning System

Beyond simple loss curves, gradient and weight statistics provide early
warning of training problems before they become catastrophic:

**Gradient norm monitoring:**
    •   Increasing: learning rate too high, about to diverge
    •   Decreasing toward zero: vanishing gradients, dead layers
    •   Sudden spike: single bad batch, hardware error, or numerical overflow
    •   Recommended: log mean and max grad norm, separated by layer group

**Per-layer gradient norms (identifying dead/dominant layers):**
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_grad_norm = param.grad.norm().item()
            writer.add_scalar(f"grad_norm/{name}", layer_grad_norm, step)

    A layer with near-zero gradients is contributing nothing to learning.
    A layer with 10× the gradients of other layers is dominating learning.

**Weight norms over training:**
    The norm of each weight matrix should grow slowly and steadily.
    Rapidly growing norms = instability.
    Norm shrinking → 0 = layer collapsing.

**The "param_grad_ratio" diagnostic:**
    For each parameter group, compute: ||params|| / ||grads||
    This ratio estimates how many gradient steps until weights change by 100%.
    Healthy range: 10–1000 (weights change slowly relative to their current value)
    Below 10: gradients are very large relative to weights (risk of instability)
    Above 10000: gradients are tiny (learning has essentially stopped)


### Hyperparameter Sweeps

A **sweep** is a systematic search over the hyperparameter space.
W&B Sweeps and Optuna are the dominant tools.

**Types of search:**
    Grid search:   Try every combination  (2^n runs for n binary hyperparams)
    Random search: Sample randomly        (works surprisingly well)
    Bayesian:      Use GP to suggest next (most sample-efficient)
    Population-based (PBT): Evolve population of runs in parallel

**W&B sweep configuration (YAML):**
    method: bayes
    metric:
      name: val/loss
      goal: minimize
    parameters:
      learning_rate:
        distribution: log_uniform_values
        min: 1e-5
        max: 1e-3
      warmup_steps:
        values: [100, 500, 1000, 2000]
      weight_decay:
        distribution: uniform
        min: 0.0
        max: 0.1

    sweep_id = wandb.sweep(sweep_config, project="llm-training")
    wandb.agent(sweep_id, function=train_function)

**The LLM hyperparameter sweep challenge:**
Full LLM training runs are expensive. Sweeps should use:
    •   Proxy tasks: train for 5% of the full duration, use early loss as proxy
    •   Small-scale proxies: train a 125M model as a proxy for 7B behaviour
    •   Multi-fidelity methods: filter bad runs early, extend promising ones


### MLflow for Production Workflows

MLflow adds model lifecycle management beyond experiment tracking:

**MLflow Tracking:**
    •   Runs, experiments, metrics, params, artifacts
    •   Auto-logging for PyTorch, sklearn, etc.
    •   REST API for programmatic access

**MLflow Models:**
    •   Standard model formats (MLmodel spec)
    •   Flavors: pytorch, sklearn, huggingface (transformers)
    •   Deploy as REST endpoint, batch, or streaming

**MLflow Model Registry:**
    •   Stage transitions: None → Staging → Production → Archived
    •   Model versioning with lineage
    •   Comments and annotations per version
    •   Webhooks for CI/CD integration

**Model registration (HuggingFace flavour):**
    import mlflow.transformers
    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="llm",
            registered_model_name="llama-7b-instruct-v1",
        )


### Profiling: Finding the True Bottleneck

Before optimising training, profile to find where time is actually spent.
The PyTorch Profiler integrates with TensorBoard for visualisation.

**Common bottlenecks revealed by profiling:**
    •   Data loading: `DataLoader` workers insufficient → GPU starves
    •   CPU-GPU synchronisation: too many `.item()` or `.numpy()` calls
    •   Memory allocation: frequent `torch.empty()` / GC pressure
    •   Communication: DDP all-reduce dominates step time

**The PyTorch Profiler + TensorBoard:**
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True) as prof:
        for i, batch in enumerate(loader):
            with record_function("forward"):
                loss = model(batch)
            with record_function("backward"):
                loss.backward()
            prof.step()
            if i == 10: break


### Alerting: Automated Training Monitoring

For long-running training, manual monitoring is insufficient. Alerts should
fire automatically for:

    Event                           Threshold       Action
    ─────────────────────────────────────────────────────────────
    Loss spike                      > 3× rolling    Slack/email alert
    Loss plateau (no improvement)   > 5k steps       LR restart or stop
    Gradient norm explosion         > 100            Auto-rollback
    GPU utilisation drop            < 30%            Check data loading
    OOM error                       any              Restart with smaller batch
    ─────────────────────────────────────────────────────────────

W&B supports alert rules via webhooks. A simple implementation:
    if grad_norm > 100:
        wandb.alert(title="Gradient explosion", text=f"Step {step}: grad_norm={grad_norm:.2f}")
        trigger_checkpoint_and_rollback()
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Experiment Tracking Systems Comparison

| Feature                  | Weights & Biases    | MLflow               | TensorBoard          |
|--------------------------|---------------------|----------------------|----------------------|
| Hosting                  | Cloud (SaaS)        | Self-hosted/Databricks| Local server         |
| Experiment comparison    | Excellent           | Good                 | Basic                |
| Hyperparameter sweeps    | Built-in (Bayesian) | None (use Optuna)    | HParams plugin       |
| Model registry           | Artifacts           | Full registry        | No                   |
| Cost                     | Free/paid           | Free (open-source)   | Free                 |
| Team collaboration       | Excellent           | Good                 | Limited              |
| Profiling support        | No                  | No                   | Excellent            |
| LLM-specific features    | Tables, media       | Log transformers     | Text/attention plots  |
| Offline support          | Yes (sync later)    | Yes                  | Yes (local files)    |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Unified Experiment Tracker": {
        "description": "A backend-agnostic experiment tracker that writes to W&B, MLflow, or TensorBoard depending on what is available — with async logging, metric grouping, and automatic health alerts.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
UNIFIED EXPERIMENT TRACKER
================================================================================

A production-ready experiment tracker that:
    1. Auto-detects available backends (wandb, mlflow, tensorboard, or console)
    2. Groups metrics into namespaced categories (train/, val/, system/)
    3. Logs asynchronously to avoid training stalls
    4. Computes and logs training health diagnostics automatically
    5. Issues alerts for common training failure modes
    6. Supports W&B Sweeps / Optuna-style hyperparameter logging

Works without any ML framework installed (falls back to console logging).

================================================================================
"""

import os
import math
import time
import threading
import queue
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from collections import deque
import torch
import torch.nn as nn


# ── Backend detection ─────────────────────────────────────────────────────────

def detect_backends() -> list[str]:
    """Detect which tracking backends are available."""
    available = []
    try:
        import wandb
        available.append("wandb")
    except ImportError:
        pass
    try:
        import mlflow
        available.append("mlflow")
    except ImportError:
        pass
    try:
        from torch.utils.tensorboard import SummaryWriter
        available.append("tensorboard")
    except ImportError:
        pass
    available.append("console")   # always available
    return available


# ── Health diagnostics ────────────────────────────────────────────────────────

@dataclass
class HealthAlert:
    level:   str    # "warning" or "critical"
    message: str
    metric:  str
    value:   float
    threshold: float


class TrainingHealthMonitor:
    """
    Monitors training metrics and issues alerts for common failure modes.

    Tracks rolling statistics and compares current values against thresholds.
    """

    def __init__(self, window: int = 50):
        self.window         = window
        self.loss_history   = deque(maxlen=window)
        self.grad_history   = deque(maxlen=window)
        self.util_history   = deque(maxlen=window)
        self.stall_counter  = 0

    def update(self, loss: float, grad_norm: float,
                gpu_util: float = 1.0) -> list[HealthAlert]:
        """Update monitoring state and return any new alerts."""
        alerts = []

        self.loss_history.append(loss)
        self.grad_history.append(grad_norm)
        self.util_history.append(gpu_util)

        if len(self.loss_history) < 10:
            return alerts

        rolling_loss = sum(list(self.loss_history)[:-1]) / max(len(self.loss_history) - 1, 1)
        rolling_grad = sum(self.grad_history) / len(self.grad_history)

        # Loss spike detection
        if loss > 3.0 * rolling_loss and rolling_loss > 0:
            alerts.append(HealthAlert("warning", f"Loss spike: {loss:.3f} (rolling: {rolling_loss:.3f})",
                                       "train/loss", loss, 3.0 * rolling_loss))

        # Gradient explosion
        if grad_norm > 100.0:
            alerts.append(HealthAlert("critical", f"Gradient explosion: {grad_norm:.1f}",
                                       "train/grad_norm", grad_norm, 100.0))

        # Gradient vanishing
        if grad_norm < 1e-7:
            alerts.append(HealthAlert("warning", f"Vanishing gradients: {grad_norm:.2e}",
                                       "train/grad_norm", grad_norm, 1e-7))

        # Loss plateau detection
        if len(self.loss_history) == self.window:
            first_half_mean = sum(list(self.loss_history)[:self.window//2]) / (self.window//2)
            second_half_mean = sum(list(self.loss_history)[self.window//2:]) / (self.window//2)
            if abs(first_half_mean - second_half_mean) / max(first_half_mean, 1e-8) < 0.001:
                self.stall_counter += 1
                if self.stall_counter >= 5:  # 5 consecutive windows with no improvement
                    alerts.append(HealthAlert("warning",
                                               f"Training may have stalled (loss flat for {self.window * 5} steps)",
                                               "train/loss", second_half_mean, first_half_mean))
            else:
                self.stall_counter = 0

        # GPU utilisation
        if len(self.util_history) >= 10:
            avg_util = sum(list(self.util_history)[-10:]) / 10
            if avg_util < 0.3:
                alerts.append(HealthAlert("warning", f"Low GPU utilisation: {avg_util:.0%}",
                                           "system/gpu_util", avg_util, 0.3))

        return alerts


# ── Async metric buffer ────────────────────────────────────────────────────────

class MetricBuffer:
    """
    Buffer metrics in memory and flush periodically.
    Avoids calling the backend API at every step.
    """

    def __init__(self, flush_every: int = 10):
        self.flush_every   = flush_every
        self._buffer: list[tuple[dict, int]] = []   # (metrics, step)
        self._lock         = threading.Lock()
        self._step_count   = 0

    def add(self, metrics: dict, step: int) -> bool:
        """Add metrics to buffer. Returns True if flush is needed."""
        with self._lock:
            self._buffer.append((metrics, step))
            self._step_count += 1
        return self._step_count % self.flush_every == 0

    def drain(self) -> list[tuple[dict, int]]:
        """Take all buffered items and clear the buffer."""
        with self._lock:
            items = self._buffer.copy()
            self._buffer.clear()
        return items


# ── The unified tracker ───────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Unified experiment tracker supporting multiple backends.

    Usage:
        tracker = ExperimentTracker(project="llm-training", config={...})
        tracker.start()
        # In training loop:
        tracker.log({"train/loss": 2.3, "train/lr": 3e-4}, step=100)
        tracker.log_model_diagnostics(model, step=100)
        tracker.finish()
    """

    def __init__(self, project: str = "llm-training",
                 run_name: str = None,
                 config: dict = None,
                 preferred_backend: str = "auto",
                 log_dir: str = "./runs",
                 flush_every: int = 10):
        self.project   = project
        self.run_name  = run_name or f"run-{int(time.time())}"
        self.config    = config or {}
        self.log_dir   = log_dir
        self.backend   = None
        self._backend_name = preferred_backend
        self.buffer    = MetricBuffer(flush_every=flush_every)
        self.health    = TrainingHealthMonitor(window=50)
        self._step_0   = None
        self._t0       = None

        # For console fallback
        self._console_history: list[dict] = []

    def start(self):
        """Initialise the chosen backend."""
        self._t0     = time.time()
        available    = detect_backends()

        if self._backend_name == "auto":
            chosen = available[0]
        elif self._backend_name in available:
            chosen = self._backend_name
        else:
            chosen = "console"

        self._backend_name = chosen
        print(f"  [Tracker] Using backend: {chosen}")

        if chosen == "wandb":
            import wandb
            wandb.init(project=self.project, name=self.run_name, config=self.config)
            self.backend = wandb
        elif chosen == "mlflow":
            import mlflow
            mlflow.set_experiment(self.project)
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(self.config)
            self.backend = mlflow
        elif chosen == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            tb_path = os.path.join(self.log_dir, self.run_name)
            self.backend = SummaryWriter(log_dir=tb_path)
            print(f"  [Tracker] TensorBoard dir: {tb_path}")
        else:
            self.backend = None   # console mode

    def log(self, metrics: dict, step: int) -> list[HealthAlert]:
        """
        Log metrics at a given step.
        Buffers internally and flushes every flush_every steps.
        Returns any health alerts triggered.
        """
        # Health monitoring
        alerts = []
        if "train/loss" in metrics or "train/grad_norm" in metrics:
            loss      = metrics.get("train/loss", float("nan"))
            grad_norm = metrics.get("train/grad_norm", float("nan"))
            gpu_util  = metrics.get("system/gpu_util", 1.0)
            if not math.isnan(loss):
                alerts = self.health.update(loss, grad_norm, gpu_util)

        # Add derived metrics
        if "train/loss" in metrics:
            metrics["train/perplexity"] = math.exp(min(metrics["train/loss"], 20))

        if self._step_0 is None:
            self._step_0 = step

        # Throughput
        if self._t0 is not None:
            elapsed = time.time() - self._t0
            metrics["system/steps_per_second"] = (step - self._step_0) / max(elapsed, 1e-6)

        should_flush = self.buffer.add(metrics, step)
        if should_flush:
            self._flush()

        # Handle alerts
        for alert in alerts:
            print(f"  [{'⚠️ WARNING' if alert.level == 'warning' else '🚨 CRITICAL'}] "
                  f"{alert.message}")

        return alerts

    def _flush(self):
        """Write buffered metrics to the backend."""
        items = self.buffer.drain()
        if not items:
            return

        if self._backend_name == "wandb":
            for metrics, step in items:
                self.backend.log(metrics, step=step)

        elif self._backend_name == "mlflow":
            import mlflow
            for metrics, step in items:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        mlflow.log_metric(key.replace("/", "."), value, step=step)

        elif self._backend_name == "tensorboard":
            for metrics, step in items:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        self.backend.add_scalar(key, value, global_step=step)

        else:
            # Console: accumulate for summary
            self._console_history.extend(items)

    def log_model_diagnostics(self, model: nn.Module, step: int,
                                log_histograms: bool = False):
        """
        Compute and log per-layer gradient and weight statistics.
        These are the early-warning metrics for training health.
        """
        metrics = {}
        total_grad_norm = 0.0
        total_weight_norm = 0.0
        n_params = 0

        for name, param in model.named_parameters():
            if param.data is not None:
                w_norm = param.data.norm().item()
                total_weight_norm += w_norm ** 2
                n_params += 1

                if log_histograms and self._backend_name == "tensorboard":
                    self.backend.add_histogram(f"weights/{name}", param.data, step)

            if param.grad is not None:
                g_norm = param.grad.norm().item()
                total_grad_norm += g_norm ** 2
                metrics[f"grad_norms/{name}"] = g_norm

                if log_histograms and self._backend_name == "tensorboard":
                    self.backend.add_histogram(f"grads/{name}", param.grad, step)

        if total_grad_norm > 0:
            metrics["train/total_grad_norm"] = math.sqrt(total_grad_norm)
        if total_weight_norm > 0:
            metrics["train/total_weight_norm"] = math.sqrt(total_weight_norm)

        if metrics:
            self.log(metrics, step)

    def log_config(self):
        """Log the full configuration as a table or artifact."""
        if self._backend_name == "wandb":
            self.backend.config.update(self.config)
        elif self._backend_name == "mlflow":
            import mlflow
            mlflow.log_params(self.config)
        print(f"  [Tracker] Config logged: {len(self.config)} parameters")

    def finish(self):
        """Flush remaining metrics and close the backend."""
        self._flush()   # ensure all buffered metrics are written

        if self._backend_name == "wandb":
            self.backend.finish()
        elif self._backend_name == "mlflow":
            import mlflow
            mlflow.end_run()
        elif self._backend_name == "tensorboard":
            self.backend.close()
        elif self._backend_name == "console":
            # Print a summary table
            self._print_console_summary()

        print(f"  [Tracker] Run {self.run_name} finished.")

    def _print_console_summary(self):
        """Print a training summary when using console backend."""
        if not self._console_history:
            return
        print()
        print("  Training Summary (console backend):")
        print(f"  {'Step':>8}  {'Loss':>10}  {'PPL':>10}  {'LR':>12}  {'Grad norm':>12}")
        print(f"  {'':─>8}  {'':─>10}  {'':─>10}  {'':─>12}  {'':─>12}")
        for metrics, step in self._console_history:
            if "train/loss" in metrics:
                print(f"  {step:>8}  {metrics.get('train/loss', float('nan')):>10.4f}  "
                      f"{metrics.get('train/perplexity', float('nan')):>10.2f}  "
                      f"{metrics.get('train/lr', 0):>12.2e}  "
                      f"{metrics.get('train/grad_norm', float('nan')):>12.4f}")


# ── Demo training loop ────────────────────────────────────────────────────────

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

    def forward(self, x):
        import torch.nn.functional as F
        h = self.embed(x)
        for l in self.layers: h = h + l(h)
        return self.head(self.norm(h))


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    import math

    torch.manual_seed(42)
    VOCAB, D = 256, 64
    N_STEPS  = 60
    B, T     = 4, 16

    config = {
        "model":        "TinyLM",
        "vocab_size":   VOCAB,
        "d_model":      D,
        "lr":           3e-4,
        "batch_size":   B,
        "seq_len":      T,
        "n_steps":      N_STEPS,
    }

    tracker = ExperimentTracker(
        project           = "llm-demo",
        run_name          = "baseline-run",
        config            = config,
        preferred_backend = "console",
        flush_every       = 10,
    )
    tracker.start()
    tracker.log_config()

    model = TinyLM(VOCAB, D)
    opt   = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

    print()
    print("  Training with unified tracker...")
    print()

    all_alerts = []
    for step in range(1, N_STEPS + 1):
        x = torch.randint(0, VOCAB, (B, T))
        y = torch.randint(0, VOCAB, (B, T))

        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))

        # Inject a loss spike at step 30 to test alerting
        if step == 30:
            with torch.no_grad():
                loss_val = loss.item() * 5.0   # simulate spike
        else:
            loss_val = loss.item()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()

        # Log core metrics
        alerts = tracker.log({
            "train/loss":      loss_val,
            "train/grad_norm": grad_norm,
            "train/lr":        sched.get_last_lr()[0],
        }, step=step)

        all_alerts.extend(alerts)

        # Log model diagnostics every 20 steps
        if step % 20 == 0:
            tracker.log_model_diagnostics(model, step=step)

    tracker.finish()

    print()
    print(f"  Total alerts issued: {len(all_alerts)}")
    for alert in all_alerts:
        print(f"    [{alert.level.upper()}] {alert.message}")
    print()
    print("  In production:")
    print("  - Replace 'console' backend with 'wandb' or 'mlflow'")
    print("  - Add alerts → Slack/email via webhook")
    print("  - Log benchmarks every 10k steps (MMLU, HumanEval)")
''',
    },

    "Hyperparameter Sweep and Comparison": {
        "description": "Implement a mini hyperparameter sweep framework that runs multiple configurations, logs results, and identifies the best hyperparameter setting using parallel coordinates analysis.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
HYPERPARAMETER SWEEP AND COMPARISON
================================================================================

Implements a lightweight hyperparameter sweep:
    1. Grid and random search over a hyperparameter space
    2. Early stopping: kill underperforming runs early (successive halving)
    3. Multi-run comparison: identify which HPs matter most
    4. Correlation analysis: which hyperparameters correlate with final loss?
    5. Parallel coordinates visualisation (text-based)

This is the W&B Sweep / Optuna workflow implemented from scratch.

================================================================================
"""

import math
import copy
import random
import itertools
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RunResult:
    """Results of one hyperparameter configuration."""
    run_id:    int
    config:    dict
    losses:    list[float]     # loss at each eval step
    final_loss: float
    best_loss:  float
    n_steps:   int
    stopped_early: bool = False

    @property
    def summary(self) -> dict:
        return {
            "run_id":       self.run_id,
            "final_loss":   self.final_loss,
            "best_loss":    self.best_loss,
            "stopped":      self.stopped_early,
            **{f"hp/{k}": v for k, v in self.config.items()},
        }


class HyperparameterSweep:
    """
    Lightweight hyperparameter sweep with:
        - Grid/random/Bayesian-style search
        - Successive halving (early stopping of bad runs)
        - Feature importance (which HP correlates most with final loss?)
    """

    def __init__(self, param_space: dict, seed: int = 42):
        """
        param_space: dict of {param_name: list_of_values}
        Example:
            {"lr": [1e-4, 3e-4, 1e-3],
             "weight_decay": [0.0, 0.01, 0.1],
             "warmup": [0, 100, 500]}
        """
        self.param_space = param_space
        self.rng         = random.Random(seed)
        self.results: list[RunResult] = []

    def grid_configs(self) -> list[dict]:
        """All combinations in param_space."""
        keys   = list(self.param_space.keys())
        values = list(self.param_space.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def random_configs(self, n: int) -> list[dict]:
        """n randomly sampled configurations."""
        configs = []
        for _ in range(n):
            config = {k: self.rng.choice(v) for k, v in self.param_space.items()}
            configs.append(config)
        return configs

    def run(self, train_fn, configs: list[dict],
             n_steps: int = 50, eval_every: int = 10,
             successive_halving: bool = True,
             halving_fraction: float = 0.5) -> list[RunResult]:
        """
        Run all configurations and collect results.
        With successive halving: after the first eval, keep only the top
        fraction of runs; after the second eval, keep the top of those, etc.
        """
        active_runs = list(range(len(configs)))
        partial_losses: dict[int, list[float]] = {i: [] for i in active_runs}
        steps_run     = {i: 0 for i in active_runs}

        # Successive halving schedule
        eval_steps  = list(range(eval_every, n_steps + 1, eval_every))
        halving_pts = set()
        if successive_halving and len(eval_steps) >= 3:
            halving_pts = {eval_steps[len(eval_steps)//3],
                            eval_steps[2*len(eval_steps)//3]}

        print(f"  Running {len(configs)} configs × {n_steps} steps each")
        if successive_halving:
            print(f"  Successive halving at steps: {halving_pts}")
        print()

        # Run all active configs step by step
        models_and_opts = {}
        for run_id in active_runs:
            torch.manual_seed(configs[run_id].get("seed", run_id))
            model  = SimpleModel(configs[run_id])
            lr     = configs[run_id].get("lr", 3e-4)
            wd     = configs[run_id].get("weight_decay", 0.01)
            opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            models_and_opts[run_id] = (model, opt)

        for step in range(1, n_steps + 1):
            # Process one step for all active runs
            x_batch = torch.randint(0, 64, (4, 8))
            y_batch = torch.randint(0, 64, (4, 8))

            for run_id in active_runs:
                model, opt = models_and_opts[run_id]
                train_loss = train_fn(model, opt, x_batch, y_batch,
                                       config=configs[run_id], step=step)
                steps_run[run_id] = step

                if step % eval_every == 0:
                    partial_losses[run_id].append(train_loss)

            # Successive halving: kill bottom fraction after halving point
            if successive_halving and step in halving_pts and len(active_runs) > 1:
                # Sort by most recent loss
                ranked = sorted(active_runs,
                                  key=lambda r: partial_losses[r][-1] if partial_losses[r] else float("inf"))
                n_keep = max(1, int(len(active_runs) * halving_fraction))
                eliminated = ranked[n_keep:]
                active_runs = ranked[:n_keep]
                if eliminated:
                    print(f"    Step {step}: Eliminated {len(eliminated)} run(s), "
                          f"keeping top {len(active_runs)}")

        # Collect results
        results = []
        for run_id, config in enumerate(configs):
            losses    = partial_losses[run_id]
            final     = losses[-1] if losses else float("inf")
            best      = min(losses) if losses else float("inf")
            stopped   = steps_run[run_id] < n_steps
            result    = RunResult(run_id, config, losses, final, best,
                                   steps_run[run_id], stopped)
            results.append(result)
            self.results.append(result)

        return results

    def feature_importance(self) -> dict[str, float]:
        """
        Compute Spearman-like rank correlation between each HP value and final loss.
        Higher absolute correlation = more important hyperparameter.
        """
        if not self.results:
            return {}

        losses = [r.final_loss for r in self.results]
        importances = {}

        for param, values in self.param_space.items():
            param_vals = [r.config[param] for r in self.results]
            # Convert to ranks
            unique_vals = sorted(set(param_vals))
            val_to_rank = {v: i for i, v in enumerate(unique_vals)}
            param_ranks = [val_to_rank[v] for v in param_vals]
            loss_ranks  = sorted(range(len(losses)), key=lambda i: losses[i])
            loss_ranks_dict = {orig: rank for rank, orig in enumerate(loss_ranks)}

            # Spearman correlation
            n = len(param_ranks)
            mean_p = sum(param_ranks) / n
            mean_l = sum(loss_ranks_dict[i] for i in range(n)) / n

            cov  = sum((param_ranks[i] - mean_p) * (loss_ranks_dict[i] - mean_l)
                        for i in range(n))
            std_p = math.sqrt(sum((p - mean_p)**2 for p in param_ranks) + 1e-10)
            std_l = math.sqrt(sum((loss_ranks_dict[i] - mean_l)**2 for i in range(n)) + 1e-10)

            importances[param] = abs(cov / (std_p * std_l))

        return dict(sorted(importances.items(), key=lambda x: -x[1]))

    def print_results(self) -> None:
        """Print a comparison table of all run results."""
        sorted_r = sorted(self.results, key=lambda r: r.final_loss)
        params   = list(self.param_space.keys())

        header = f"  {'Rank':>4}  {'Final loss':>12}  {'Best loss':>12}  {'Steps':>6}  "
        header += "  ".join(f"{p[:8]:>8}" for p in params)
        print(header)
        print(f"  {'':─>4}  {'':─>12}  {'':─>12}  {'':─>6}  " + "  ".join("─"*8 for _ in params))

        for rank, r in enumerate(sorted_r):
            row = f"  {rank+1:>4}  {r.final_loss:>12.4f}  {r.best_loss:>12.4f}  {r.n_steps:>6}  "
            row += "  ".join(f"{str(r.config.get(p, '?'))[:8]:>8}" for p in params)
            if r.stopped_early:
                row += "  [early stop]"
            print(row)


# ── Simple model for sweep ────────────────────────────────────────────────────

class SimpleModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d = config.get("d_model", 32)
        self.embed = nn.Embedding(64, d)
        self.fc    = nn.Linear(d, 64, bias=False)

    def forward(self, x):
        return self.fc(self.embed(x).mean(1))


def train_step(model, opt, x, y, config, step=1):
    """One training step with warmup (simulated)."""
    warmup  = config.get("warmup", 0)
    dropout = config.get("dropout", 0.0)
    scale   = min(1.0, step / max(warmup, 1))

    for g in opt.param_groups:
        g["lr"] = config.get("lr", 3e-4) * scale

    opt.zero_grad()
    h       = model.embed(x[:, 0])   # use first token
    if dropout > 0:
        h   = F.dropout(h, p=dropout, training=True)
    logits  = model.fc(h)
    loss    = F.cross_entropy(logits, y[:, 0])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    param_space = {
        "lr":           [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 0.01, 0.1],
        "warmup":       [0, 50, 200],
    }

    sweep = HyperparameterSweep(param_space, seed=42)
    configs = sweep.random_configs(n=12)   # sample 12 random configs

    print("=" * 65)
    print(f"  HYPERPARAMETER SWEEP ({len(configs)} configurations)")
    print("=" * 65)
    print()

    results = sweep.run(
        train_fn          = train_step,
        configs           = configs,
        n_steps           = 80,
        eval_every        = 20,
        successive_halving = True,
        halving_fraction   = 0.5,
    )

    print()
    print("  RESULTS TABLE:")
    sweep.print_results()

    # Feature importance
    print()
    importances = sweep.feature_importance()
    print("  HYPERPARAMETER IMPORTANCE (correlation with final loss):")
    for param, imp in importances.items():
        bar = "█" * int(imp * 30)
        print(f"  {param:<15}: {imp:.3f}  |{bar:<30}|")

    # Best config
    best = min(results, key=lambda r: r.final_loss)
    print()
    print(f"  BEST CONFIGURATION (final loss = {best.final_loss:.4f}):")
    for k, v in best.config.items():
        print(f"    {k:<20}: {v}")
    print()
    print("  In production: replace with W&B Sweeps or Optuna for")
    print("  Bayesian optimisation and TPE (Tree-structured Parzen Estimator).")
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
    #     from llm_training.visuals.experiment_tracking import (
    #         TRACK_VISUAL_HTML,
    #         TRACK_VISUAL_HEIGHT,
    #     )
    #     visual_html   = TRACK_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = TRACK_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[34_experiment_tracking_wandb_mlflow.py] Could not load visual: {e}",
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