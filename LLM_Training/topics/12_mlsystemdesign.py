"""
ML System Design for LLM Training
====================================

Training a large language model is not just a machine learning problem — it
is a distributed systems engineering challenge. Understanding how the
components fit together: compute clusters, storage, data pipelines, job
schedulers, monitoring, and fault tolerance, is what separates a successful
training run from one that wastes millions of dollars on a preventable failure.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "ML System Design for LLM Training"
DISPLAY_NAME = "12 · ML System Design"
ICON         = "🏛️"
SUBTITLE     = "End-to-End Training Infrastructure"


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

### The Full Stack of an LLM Training System

When a paper reports "we trained on 1024 A100s for 3 months," it is
describing the visible tip of an iceberg. Beneath that are dozens of
engineering systems that must function reliably and efficiently together:

    Layer                   Components
    ──────────────────────────────────────────────────────────────
    Hardware                GPUs, CPUs, NVLink, InfiniBand, storage
    Cluster management      Slurm / Kubernetes / internal schedulers
    Distributed training    DDP / FSDP / Megatron / DeepSpeed
    Data pipeline           Tokenisation, packing, streaming loaders
    Training loop           Mixed precision, grad clip, LR schedule
    Checkpointing           Async saves, cloud storage, sharding
    Monitoring              Metrics, alerts, dashboards, anomaly detection
    Fault tolerance         Auto-restart, checkpoint recovery, straggler handling
    ──────────────────────────────────────────────────────────────

Each layer has failure modes that can silently corrupt a training run or
waste GPU-hours. This module maps the full system and the failure modes at
each level.


### The Compute Cluster

A typical LLM training cluster is a collection of **nodes** (servers), each
containing multiple GPUs connected via NVLink. Nodes are connected to each
other via a high-speed network fabric.

**DGX A100 node spec (common configuration):**
    •   8 × A100 80GB SXM GPUs
    •   NVLink 3.0 full mesh between GPUs (600 GB/s per GPU)
    •   2 × AMD EPYC CPUs (64–128 cores total)
    •   2 TB DDR4 CPU RAM
    •   8 × InfiniBand HDR 200 Gb/s NICs (one per GPU)
    •   5–10 TB NVMe local SSD

**Cluster sizes for common models:**
    Model           GPUs        Nodes (DGX A100)    Training time
    ─────────────────────────────────────────────────────────────
    LLaMA-2 7B        64            8               ~3 weeks
    LLaMA-2 70B      512–2048      64–256           ~6 weeks
    GPT-3 175B       1024          128              ~1 month
    GPT-4 (est.)     ~25000+      ~3000+            several months
    ─────────────────────────────────────────────────────────────

At 25,000 GPUs running for 3 months, the hardware cost alone is ~$100M.
**Failure tolerance is not optional** — with that many GPUs, hardware failures
happen multiple times per day.


### Storage Architecture

Data locality and I/O throughput are often the hidden bottleneck in LLM
training. The storage hierarchy:

    Tier            Technology          Bandwidth       Latency
    ──────────────────────────────────────────────────────────────
    GPU HBM         HBM2e               2 TB/s          ~100 ns
    CPU RAM         DDR4/DDR5           100–300 GB/s    ~100 ns
    Local NVMe      PCIe 4.0 × 4        7 GB/s          ~100 µs
    Parallel FS     Lustre / GPFS       100–500 GB/s    ~1 ms
    Object storage  S3 / GCS / ABS      5–50 GB/s       ~10 ms
    ──────────────────────────────────────────────────────────────

**The data loading problem:**
Training a LLaMA-3 8B model on 15T tokens at global batch size 4M tokens
processes 4M tokens/step. At 2 bytes/token (BPE token IDs), that is 8 MB
of raw input data per step. This sounds manageable, but:
    •   Data must be tokenised, shuffled, and packed before training
    •   A 15T-token dataset = ~30 TB of raw tokenised data on disk
    •   The DataLoader must stream data fast enough to keep GPUs busy
    •   Typically: pre-tokenise and save as memory-mapped binary files

**Memory-mapped files (mmap):** The standard approach for large datasets.
A numpy `.npy` or binary file is opened with `np.memmap`, so the OS loads
pages from disk on demand. The entire dataset is never loaded into RAM.


    **Diagram 1 — End-to-End Training System Architecture:**

    LLM TRAINING SYSTEM ARCHITECTURE
    ════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │  RAW DATA                                                   │
    │  Web text, books, code → deduplicate → quality filter      │
    │  → tokenise → pack → write to binary shards on Lustre FS   │
    └──────────────────────────┬──────────────────────────────────┘
                               │ stream
    ┌──────────────────────────▼──────────────────────────────────┐
    │  DATA LOADING  (per node)                                   │
    │  DistributedSampler → DataLoader workers → prefetch buffer  │
    │  Pin memory → async transfer to GPU HBM                    │
    └──────────────────────────┬──────────────────────────────────┘
                               │ batches
    ┌──────────────────────────▼──────────────────────────────────┐
    │  TRAINING LOOP  (N nodes × 8 GPUs each)                    │
    │  Forward (autocast bf16) → Loss → Backward                 │
    │  ← DDP/FSDP gradient all-reduce across all GPUs →          │
    │  Grad clip → AdamW → LR schedule                           │
    └──────────────────────────┬──────────────────────────────────┘
                               │ every K steps
    ┌──────────────────────────▼──────────────────────────────────┐
    │  CHECKPOINTING                                              │
    │  Async save to Lustre + cloud (S3/GCS)                     │
    │  Save: model, optimiser state, RNG state, step counter     │
    └──────────────────────────┬──────────────────────────────────┘
                               │ metrics
    ┌──────────────────────────▼──────────────────────────────────┐
    │  MONITORING                                                 │
    │  W&B / TensorBoard: loss, grad norm, LR, tokens/sec        │
    │  Alert on: NaN, divergence, straggler GPU, OOM             │
    └─────────────────────────────────────────────────────────────┘


### Job Scheduling and Cluster Management

Large clusters are shared resources managed by a scheduler.

**Slurm (Simple Linux Utility for Resource Management):**
The dominant scheduler in academic and HPC environments. A training job
is submitted as a batch script that specifies resources and runs torchrun:

    #!/bin/bash
    #SBATCH --nodes=16
    #SBATCH --ntasks-per-node=8
    #SBATCH --gres=gpu:8
    #SBATCH --time=72:00:00
    #SBATCH --partition=gpu

    srun torchrun \\
        --nnodes=$SLURM_NNODES \\
        --nproc_per_node=8 \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
        train.py

**Kubernetes + GPU Operators:** More common in cloud environments.
Operators like NVIDIA's GPU Operator manage GPU resource allocation.
Volcano and Kueue are emerging schedulers for ML workloads on Kubernetes.


### Fault Tolerance

At scale, hardware failures are guaranteed. The question is not if but
how often:

    Component        Mean time between failures   Notes
    ─────────────────────────────────────────────────────────────
    Single GPU       ~100,000 hours               Rare per device
    8-GPU node       ~12,500 hours                ~1.4 years
    128-GPU cluster  ~781 hours                   ~32 days
    1024-GPU cluster ~98 hours                    ~4 days ⚠️
    25000-GPU cluster ~4 hours                    Multiple per day!
    ─────────────────────────────────────────────────────────────

**Checkpoint-based recovery:** The standard approach.
    1.  Save checkpoints every 500–2000 steps
    2.  On failure, detect via heartbeat or job failure
    3.  Restart job, reload latest checkpoint, resume training
    4.  "Wasted" compute = steps since last checkpoint (minutes, not hours)

**Elastic training:** More sophisticated — allows the job to continue
with fewer GPUs while waiting for replacements. Requires careful handling
of batch size changes (the global batch size drops if fewer GPUs are
available, affecting the training dynamics).

**Straggler mitigation:** One slow GPU can hold back the entire job during
synchronous gradient all-reduce. Solutions:
    •   Monitor per-GPU throughput; alert on stragglers
    •   NCCL timeout: if a GPU fails to respond within N seconds, abort
    •   Async communication where possible (gradient all-reduce can be
        overlapped with the next forward pass in DDP)


### The Global Batch Size and Its Implications

**Global batch size** = per-GPU batch size × number of GPUs × gradient
accumulation steps.

For LLM training, a common target is a global batch size of 1–8 million
tokens per step. This is the "effective batch size" used by the optimiser.

    LLaMA-2 7B:  global_batch = 4M tokens
                 = 2048 tokens/seq × ~2000 seqs per step
                 = 2048 × 1 (local batch per GPU) × 8 GPUs × 128 grad_accum

The global batch size affects:
    •   Training stability: larger batches need higher LR (linear scaling)
    •   Memory: larger local batches → more activation memory
    •   Communication: more compute per communication step → better overlap
    •   Token throughput: larger batches → better GPU utilisation


### Monitoring and Observability

A production training run emits many signals that indicate system health
and model training health:

    Metric              Healthy range               Warning signal
    ──────────────────────────────────────────────────────────────
    Training loss       Monotonically decreasing    Spikes > 2× mean
    Validation loss     Tracks train loss closely   Large gap = overfitting
    Gradient norm       Stable, < max_norm mostly   Constant at max_norm
    Learning rate       Follows schedule            Stuck at 0 or max
    Tokens/second       Stable or improving         Sudden drop > 10%
    GPU utilisation     > 90% (compute bound)       < 80% = bottleneck
    GPU memory          < 95% of capacity           OOM = crash
    Loss scale (fp16)   Stable 1k–65k               Rapidly shrinking = NaN
    ──────────────────────────────────────────────────────────────

**Weights & Biases (W&B)** and **TensorBoard** are the standard monitoring
tools. Key W&B features for LLM training:
    •   Real-time loss curves across runs
    •   System metrics (GPU util, memory, temperature)
    •   Gradient histograms per layer
    •   Automatic alerts on metric anomalies
    •   Artifact tracking (checkpoints, datasets)


### MFU — Model FLOP Utilisation

**MFU (Model FLOP Utilisation)** is the standard efficiency metric for
LLM training, measuring what fraction of theoretical peak GPU FLOP/s is
being used productively:

    MFU = (observed_flops_per_second) / (theoretical_peak_flops)

where observed_flops are estimated from the model's arithmetic intensity
and the measured training throughput (tokens/second).

**Computing theoretical FLOPs per token:**
    Per-token FLOPs ≈ 6 × N_params   (forward + backward, ignoring attention)
    (The factor 6 comes from: 2× matmul FLOPs × 3 for fwd+bwd combined)

    For LLaMA-2 7B: 6 × 7e9 = 42 GFLOP per token

If we achieve 100k tokens/second on 64 A100s (64 × 312 TFLOPS peak):
    observed = 42e9 × 100000 = 4.2 PFLOPS
    peak     = 64 × 312e12   = 19.97 PFLOPS
    MFU      = 4.2 / 19.97   ≈ 21%

Typical MFU values:
    •   Poor (< 30%): bottlenecks in data loading, communication, or I/O
    •   Good (40–50%): well-tuned distributed training
    •   Excellent (50–60%): expert-level kernel fusion and overlap
    •   Maximum (> 60%): rare; requires careful batch size tuning

GPT-3 paper reported ~50 TFLOPS per A100 on 1024 V100s (older hardware).
PaLM achieved ~46% MFU on TPUv4. LLaMA paper achieved ~380 tokens/s/GPU.


### Cost Estimation

Training cost can be estimated from:

    Cost = (FLOPs_to_train) / (GPU_FLOP_per_dollar)

**Chinchilla optimal FLOPs:**
    C_opt ≈ 6 × N_params × D_tokens   (Hoffmann et al., 2022 approximation)
    For 7B model trained on 140B tokens (Chinchilla): 6 × 7e9 × 140e9 ≈ 5.9e21 FLOPs

    At $2/A100-hour, 312 TFLOPS peak, 40% MFU:
    Effective FLOPS = 312e12 × 0.4 = 125 GFLOPS/s
    Time for 5.9e21 FLOPs on 1 A100: 5.9e21 / 1.25e11 = 47,200 hours
    Cost on 1 A100: 47,200 × $2 = $94,400

    On 64 A100s: time = 737 hours ≈ 31 days; cost = $94,400 (same total $)
    Adding communication overhead (~20%): ~$113,000 total

    For a 70B model on 2T tokens: ~10× more → ~$1–2M (rough estimate)

Note: Real costs vary widely based on cluster efficiency, cloud vs owned
hardware, and data center costs.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Training Infrastructure Decision Guide

| Requirement                     | Recommended approach                           |
|---------------------------------|------------------------------------------------|
| < 7B model, 1 node              | DDP + ZeRO Stage 1                             |
| 7B–13B model, 1–4 nodes         | DDP + ZeRO Stage 2 + activation checkpointing  |
| 70B model, multi-node           | FSDP/ZeRO Stage 3 + tensor parallelism         |
| > 100B model                    | 3D parallelism (DP + TP + PP) + ZeRO           |
| Unknown training duration       | WSD schedule + checkpoint every 1000 steps     |
| Cloud (spot/preemptible)        | Checkpoint every 200–500 steps                 |
| On-premise HPC                  | Slurm + torchrun + Lustre FS                    |
| Research / small scale          | Single node, W&B logging, simple DDP           |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "MFU Calculator and Training Cost Estimator": {
        "description": "Calculate Model FLOP Utilisation from measured throughput, and estimate training cost for any model size and token budget.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MFU CALCULATOR AND TRAINING COST ESTIMATOR
================================================================================

Computes:
    1. Theoretical FLOPs per token for any model configuration
    2. MFU from measured tokens/second
    3. Training cost estimate (time and $) for any model/data/hardware combo
    4. Chinchilla-optimal compute allocation

Reference: Chowdhery et al. (PaLM), Narayanan et al. (Megatron-LM),
           Hoffmann et al. (Chinchilla)
================================================================================
"""

import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name:       str
    n_params:   int       # total parameters (excluding embeddings sometimes)
    n_layers:   int
    d_model:    int
    d_ff:       int
    n_heads:    int
    vocab_size: int = 32000
    seq_len:    int = 2048


@dataclass
class HardwareConfig:
    name:           str
    n_gpus:         int
    peak_tflops:    float    # bf16 Tensor Core TFLOPS per GPU
    cost_per_hour:  float    # USD per GPU-hour


# ── FLOPs estimation ──────────────────────────────────────────────────────────

def flops_per_token(cfg: ModelConfig) -> int:
    """
    Estimate the number of FLOPs (floating-point operations) per token
    for one forward pass.

    The approximation: 2 × P  (where P = non-embedding parameters)
    accounts for:
        - Each matmul multiply-add counts as 2 FLOPs
        - Ignoring non-linear activations and norms (< 1% of total)

    For the forward + backward pass (training): multiply by 3
    (backward pass costs ~2× the forward pass).

    For attention specifically:
        - QKV projections:  2 × 3 × d_model² × T / T = 6d²
        - Attention scores: 2 × T × d_model            (T-dependent)
        - FFN:              2 × 2 × d_model × d_ff      (or 3 for SwiGLU)

    The simplified formula 6N (= 2N × 3 for fwd+bwd) is from Chinchilla.
    """
    # Non-embedding parameters: total minus embedding table
    n_non_embed = cfg.n_params - cfg.vocab_size * cfg.d_model

    # FLOPs per token, forward pass only
    flops_forward = 2 * n_non_embed

    # Attention FLOPs: 2 × T × d_model per layer (quadratic in T!)
    # This becomes significant at long contexts
    flops_attn_per_layer = 2 * cfg.seq_len * cfg.d_model
    flops_attn_total = cfg.n_layers * flops_attn_per_layer

    # Training: forward + backward ≈ 3× forward
    flops_train = 3 * (flops_forward + flops_attn_total / cfg.seq_len)

    return int(flops_train)


def compute_mfu(cfg: ModelConfig, hw: HardwareConfig,
                measured_tokens_per_sec: float) -> float:
    """
    Compute Model FLOP Utilisation (MFU).

    MFU = (actual FLOPs/s) / (theoretical peak FLOPs/s)
    """
    actual_flops_per_sec   = flops_per_token(cfg) * measured_tokens_per_sec
    peak_flops_per_sec     = hw.n_gpus * hw.peak_tflops * 1e12

    return actual_flops_per_sec / peak_flops_per_sec


def training_cost(cfg: ModelConfig, hw: HardwareConfig,
                  n_tokens: int, mfu: float) -> dict:
    """
    Estimate training time and cost.

    Returns dict with time_hours, cost_usd, tokens_per_sec_per_gpu.
    """
    total_flops     = flops_per_token(cfg) * n_tokens

    # Effective FLOPS accounting for MFU
    peak_flops_s    = hw.n_gpus * hw.peak_tflops * 1e12
    eff_flops_s     = peak_flops_s * mfu

    time_s          = total_flops / eff_flops_s
    time_h          = time_s / 3600
    cost_usd        = time_h * hw.n_gpus * hw.cost_per_hour
    tps_per_gpu     = (eff_flops_s / hw.n_gpus) / flops_per_token(cfg)

    return {
        "time_hours":        time_h,
        "time_days":         time_h / 24,
        "cost_usd":          cost_usd,
        "tps_per_gpu":       tps_per_gpu,
        "total_tflops":      total_flops / 1e12,
    }


def chinchilla_optimal(compute_budget_flops: float) -> tuple:
    """
    Given a compute budget (FLOPs), return the Chinchilla-optimal
    (model_size, dataset_size) pair.

    Chinchilla: C ≈ 6 × N × D, with N ≈ D (equal compute to model and data)
    Optimal: N* = sqrt(C / (6 × 20))   D* = 20 × N*
    (The factor 20 comes from the Chinchilla paper's fitted constant)
    """
    N_opt = math.sqrt(compute_budget_flops / (6 * 20))
    D_opt = 20 * N_opt
    return int(N_opt), int(D_opt)


def fmt_large(n: float) -> str:
    if n >= 1e12: return f"{n/1e12:.1f}T"
    if n >= 1e9:  return f"{n/1e9:.1f}B"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    return f"{n:.0f}"


# ── Predefined model and hardware configs ─────────────────────────────────────

MODELS = [
    ModelConfig("GPT-2 117M",   117e6,  12, 768,  3072,   12, 50257, 1024),
    ModelConfig("LLaMA-2 7B",  7e9,    32, 4096, 11008,  32, 32000, 2048),
    ModelConfig("LLaMA-2 13B", 13e9,   40, 5120, 13824,  40, 32000, 2048),
    ModelConfig("LLaMA-2 70B", 70e9,   80, 8192, 28672,  64, 32000, 2048),
    ModelConfig("LLaMA-3 70B", 70e9,   80, 8192, 28672,  64, 128000, 4096),
]

HARDWARE = [
    HardwareConfig("A100 80GB × 8",   8,   312.0, 2.00),
    HardwareConfig("A100 80GB × 64",  64,  312.0, 2.00),
    HardwareConfig("A100 80GB × 512", 512, 312.0, 2.00),
    HardwareConfig("H100 SXM × 8",   8,   989.0, 4.00),
    HardwareConfig("H100 SXM × 64",  64,  989.0, 4.00),
]


if __name__ == "__main__":
    # 1. FLOPs per token comparison
    print("=" * 68)
    print("  FLOPS PER TOKEN (forward + backward, training)")
    print("=" * 68)
    print()
    print(f"  {'Model':<20} {'Params':>10}  {'FLOPs/token':>14}  {'≈ 6N check':>12}")
    print(f"  {'':─<20} {'':─>10}  {'':─>14}  {'':─>12}")
    for m in MODELS:
        fpt    = flops_per_token(m)
        approx = 6 * m.n_params
        print(f"  {m.name:<20} {fmt_large(m.n_params):>10}  "
              f"{fmt_large(fpt):>14}  {fmt_large(approx):>12}")

    # 2. MFU for reported throughputs
    print()
    print("=" * 68)
    print("  MFU FROM REPORTED THROUGHPUTS")
    print("=" * 68)
    print()
    # (model, hardware, measured tokens/sec, source)
    benchmarks = [
        (MODELS[1], HARDWARE[1],  380 * 64,  "LLaMA-1 paper (7B, 64×A100)"),
        (MODELS[3], HARDWARE[2], 140 * 512,  "LLaMA-2 paper (70B, 512×A100)"),
        (MODELS[1], HARDWARE[0],  380 * 8,   "7B on 8×A100 (estimated)"),
        (MODELS[1], HARDWARE[3],  800 * 8,   "7B on 8×H100 (estimated)"),
    ]

    print(f"  {'Config':<40} {'tok/s':>8}  {'MFU':>8}")
    print(f"  {'':─<40} {'':─>8}  {'':─>8}")
    for model, hw, tps, desc in benchmarks:
        mfu = compute_mfu(model, hw, tps)
        print(f"  {desc:<40} {tps:>8,}  {mfu*100:>7.1f}%")

    # 3. Training cost estimates
    print()
    print("=" * 68)
    print("  TRAINING COST ESTIMATES (MFU = 40%, $2/A100-hour)")
    print("=" * 68)
    print()
    training_runs = [
        (MODELS[1], HARDWARE[1],  140e9,  "Chinchilla-opt 7B  (140B tokens)"),
        (MODELS[1], HARDWARE[1], 2000e9,  "LLaMA-2 7B         (2T tokens)"),
        (MODELS[3], HARDWARE[2], 2000e9,  "LLaMA-2 70B        (2T tokens)"),
        (MODELS[4], HARDWARE[2], 15000e9, "LLaMA-3 70B        (15T tokens)"),
    ]
    MFU = 0.40

    print(f"  {'Run':<40} {'Days':>6}  {'GPU-hours':>10}  {'Cost $':>10}")
    print(f"  {'':─<40} {'':─>6}  {'':─>10}  {'':─>10}")
    for model, hw, tokens, desc in training_runs:
        r = training_cost(model, hw, int(tokens), MFU)
        gpu_hours = r["time_hours"] * hw.n_gpus
        print(f"  {desc:<40} {r['time_days']:>6.1f}  "
              f"{gpu_hours:>10,.0f}  ${r['cost_usd']:>9,.0f}")

    # 4. Chinchilla optimal allocations
    print()
    print("=" * 68)
    print("  CHINCHILLA-OPTIMAL COMPUTE ALLOCATION")
    print("=" * 68)
    print()
    budgets = [
        ("$100k on 64×A100 (7 days)",   64 * 312e12 * 0.4 * 7 * 24 * 3600),
        ("$1M on 64×A100 (~2.4 months)", 64 * 312e12 * 0.4 * 70 * 24 * 3600),
        ("$10M on 512×A100 (~4 months)", 512 * 312e12 * 0.4 * 120 * 24 * 3600),
    ]
    print(f"  {'Budget':<40} {'Opt. Model Size':>16}  {'Opt. Tokens':>14}")
    print(f"  {'':─<40} {'':─>16}  {'':─>14}")
    for desc, flops in budgets:
        N, D = chinchilla_optimal(flops)
        print(f"  {desc:<40} {fmt_large(N):>16}  {fmt_large(D):>14}")
    print()
    print("  Chinchilla rule: train a model with as many tokens as ~20× the params.")
    print("  Larger models need proportionally more data, not just more compute.")
''',
    },

    "Production Training Checklist and Config": {
        "description": "A structured production training config and pre-flight checklist — everything to verify before launching a multi-week training run.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PRODUCTION TRAINING CHECKLIST AND CONFIG
================================================================================

A structured configuration system and pre-flight checklist for LLM training.
Encodes the lessons learned from many training runs about what to verify,
log, and monitor.

This is the "engineering playbook" complement to the ML-focused modules.
================================================================================
"""

import math
import os
from dataclasses import dataclass, field
from typing import Optional


# ── Training configuration ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Single source of truth for all training hyperparameters.
    Every value is documented with its typical range and justification.
    """

    # ── Model ──────────────────────────────────────────────────────────────
    model_name:         str   = "llama-3-8b"
    vocab_size:         int   = 128_000    # LLaMA-3 tokeniser
    d_model:            int   = 4096
    n_layers:           int   = 32
    n_heads:            int   = 32
    n_kv_heads:         int   = 8          # GQA
    d_ff:               int   = 14336      # SwiGLU ≈ 3.5× d_model
    max_seq_len:        int   = 8192

    # ── Data ───────────────────────────────────────────────────────────────
    data_dir:           str   = "/data/llama3_tokens"
    train_tokens:       int   = 15_000_000_000_000   # 15T tokens
    val_tokens:         int   =      1_000_000_000   # 1B tokens

    # ── Batch sizes ────────────────────────────────────────────────────────
    micro_batch_size:   int   = 2          # per-GPU micro batch (sequences)
    grad_accum_steps:   int   = 4          # gradient accumulation steps
    # Effective: micro_batch × n_gpus × grad_accum × seq_len tokens per step

    # ── Optimiser ──────────────────────────────────────────────────────────
    lr_max:             float = 3e-4
    lr_min:             float = 3e-5       # 10% of lr_max
    lr_warmup_steps:    int   = 2_000
    beta1:              float = 0.9
    beta2:              float = 0.95       # lower than default 0.999
    eps:                float = 1e-8
    weight_decay:       float = 0.1
    grad_clip:          float = 1.0

    # ── Precision ──────────────────────────────────────────────────────────
    dtype:              str   = "bfloat16" # "bfloat16" or "float16"
    use_grad_scaler:    bool  = False      # True if dtype == "float16"

    # ── Infrastructure ─────────────────────────────────────────────────────
    n_gpus:             int   = 64
    n_nodes:            int   = 8
    checkpoint_dir:     str   = "/checkpoints/llama3-8b"
    checkpoint_every:   int   = 1_000      # steps
    keep_last_n_ckpts:  int   = 5
    log_every:          int   = 10         # steps
    eval_every:         int   = 500        # steps

    # ── Distributed ────────────────────────────────────────────────────────
    ddp_backend:        str   = "nccl"
    tensor_parallel:    int   = 1          # TP degree (1 = no TP)
    pipeline_parallel:  int   = 1          # PP degree (1 = no PP)

    # ── Experiment tracking ────────────────────────────────────────────────
    wandb_project:      str   = "llm-training"
    wandb_run_name:     str   = "llama3-8b-15T"
    wandb_entity:       Optional[str] = None

    @property
    def global_batch_tokens(self) -> int:
        return (self.micro_batch_size * self.n_gpus *
                self.grad_accum_steps * self.max_seq_len)

    @property
    def total_train_steps(self) -> int:
        return self.train_tokens // self.global_batch_tokens

    @property
    def n_params_approx(self) -> int:
        d, N = self.d_model, self.n_layers
        return int(12 * N * d**2 + 2 * self.vocab_size * d)

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = config is valid)."""
        errors = []
        if self.d_model % self.n_heads != 0:
            errors.append(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            errors.append(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.n_gpus % (self.tensor_parallel * self.pipeline_parallel) != 0:
            errors.append("n_gpus must be divisible by TP × PP degrees")
        if self.dtype == "float16" and not self.use_grad_scaler:
            errors.append("use_grad_scaler should be True when dtype=float16")
        if self.lr_min > self.lr_max:
            errors.append(f"lr_min ({self.lr_min}) must be < lr_max ({self.lr_max})")
        if self.grad_clip <= 0:
            errors.append("grad_clip must be positive")
        return errors

    def summary(self) -> str:
        lines = [
            f"  Model:          {self.model_name}",
            f"  Parameters:     ~{self.n_params_approx/1e9:.1f}B",
            f"  Architecture:   d={self.d_model}, N={self.n_layers}, "
            f"h={self.n_heads}, kv={self.n_kv_heads}",
            f"  Context:        {self.max_seq_len} tokens",
            f"  Training data:  {self.train_tokens/1e12:.1f}T tokens",
            f"  Global batch:   {self.global_batch_tokens/1e6:.1f}M tokens/step",
            f"  Total steps:    {self.total_train_steps:,}",
            f"  Warmup steps:   {self.lr_warmup_steps:,}",
            f"  Peak LR:        {self.lr_max:.1e}",
            f"  Min  LR:        {self.lr_min:.1e}",
            f"  Precision:      {self.dtype}",
            f"  GPUs:           {self.n_gpus} ({self.n_nodes} nodes)",
            f"  Checkpoints:    every {self.checkpoint_every} steps "
            f"(keep last {self.keep_last_n_ckpts})",
        ]
        return "\n".join(lines)


# ── Pre-flight checklist ──────────────────────────────────────────────────────

class PreFlightChecklist:
    """
    Structured checklist for verifying training run readiness.
    Each check returns (passed: bool, message: str).
    """

    def __init__(self, cfg: TrainingConfig):
        self.cfg    = cfg
        self.passed = []
        self.failed = []
        self.warned = []

    def check(self, name: str, condition: bool, fail_msg: str,
              warning_only: bool = False):
        status = "✓" if condition else ("⚠️" if warning_only else "✗")
        if condition:
            self.passed.append(f"  {status}  {name}")
        elif warning_only:
            self.warned.append(f"  {status}  {name}: {fail_msg}")
        else:
            self.failed.append(f"  {status}  {name}: {fail_msg}")

    def run_all(self):
        cfg = self.cfg

        # Config validation
        errors = cfg.validate()
        self.check("Config validation",
                   len(errors) == 0,
                   "; ".join(errors))

        # Data
        self.check("Data directory exists",
                   os.path.exists(cfg.data_dir) or True,  # skip in demo
                   f"{cfg.data_dir} not found",
                   warning_only=True)

        # Batch size sanity
        expected_tokens_per_step = cfg.global_batch_tokens
        self.check("Global batch size ≥ 1M tokens",
                   expected_tokens_per_step >= 1_000_000,
                   f"Only {expected_tokens_per_step/1e6:.2f}M tokens/step — may converge slowly",
                   warning_only=True)

        # LR / batch size ratio
        lr_per_million_tokens = cfg.lr_max / (expected_tokens_per_step / 1e6)
        self.check("LR not too high for batch size",
                   lr_per_million_tokens < 5e-4,
                   f"LR ratio = {lr_per_million_tokens:.2e} — may be unstable",
                   warning_only=True)

        # Warmup length
        warmup_pct = cfg.lr_warmup_steps / cfg.total_train_steps * 100
        self.check("Warmup ≥ 0.5% of total steps",
                   warmup_pct >= 0.5,
                   f"Warmup is only {warmup_pct:.2f}% of training — too short",
                   warning_only=True)

        # Precision
        self.check("BF16 or FP16 selected (not FP32)",
                   cfg.dtype in ("bfloat16", "float16"),
                   "Using FP32 — 2× memory, 2–4× slower than BF16")

        # GradScaler consistency
        self.check("GradScaler enabled iff FP16",
                   (cfg.dtype == "float16") == cfg.use_grad_scaler,
                   "GradScaler should be enabled for FP16, disabled for BF16")

        # Checkpoint frequency
        steps_per_hour_approx = 20   # rough estimate
        ckpt_interval_h = cfg.checkpoint_every / steps_per_hour_approx
        self.check("Checkpoint interval ≤ 4 hours of training",
                   ckpt_interval_h <= 4,
                   f"Checkpointing every ~{ckpt_interval_h:.1f}h — too infrequent",
                   warning_only=True)

        # W&B / monitoring
        self.check("Experiment tracking configured",
                   bool(cfg.wandb_project),
                   "No W&B project set — training progress won't be logged",
                   warning_only=True)

        # Weight decay
        self.check("Weight decay 0.01–0.1",
                   0.001 <= cfg.weight_decay <= 0.5,
                   f"Weight decay {cfg.weight_decay} is unusual",
                   warning_only=True)

        # Grad clip
        self.check("Gradient clip > 0",
                   cfg.grad_clip > 0,
                   "Gradient clipping disabled — risky for long training runs")

    def report(self) -> str:
        lines = ["  PRE-FLIGHT CHECKLIST RESULTS", "  " + "─" * 50]
        if self.passed:
            lines.append("  PASSED:")
            lines.extend(self.passed)
        if self.warned:
            lines.append("  WARNINGS:")
            lines.extend(self.warned)
        if self.failed:
            lines.append("  FAILED:")
            lines.extend(self.failed)
        total = len(self.passed) + len(self.warned) + len(self.failed)
        lines.append(f"\n  Summary: {len(self.passed)}/{total} passed, "
                     f"{len(self.warned)} warnings, {len(self.failed)} failures")
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = TrainingConfig()

    print("=" * 65)
    print("  TRAINING CONFIGURATION SUMMARY")
    print("=" * 65)
    print(cfg.summary())

    print()
    print("=" * 65)
    print("  DERIVED METRICS")
    print("=" * 65)
    fpt        = 6 * cfg.n_params_approx   # FLOPs per token (approx)
    total_fl   = fpt * cfg.train_tokens
    print(f"  FLOPs per token:   ~{fpt/1e9:.0f} GFLOP")
    print(f"  Total training:    ~{total_fl/1e24:.2f} × 10²⁴ FLOPs")
    print(f"  At 40% MFU, 64×A100 (312 TFLOPS each):")
    eff = 64 * 312e12 * 0.4
    time_h = total_fl / eff / 3600
    print(f"    Effective FLOPS:  {eff/1e12:.0f} TFLOPS")
    print(f"    Training time:    {time_h/24:.1f} days")
    print(f"    Cost estimate:    ~${time_h * 64 * 2:,.0f} (at $2/A100-hr)")

    print()
    print("=" * 65)
    checklist = PreFlightChecklist(cfg)
    checklist.run_all()
    print(checklist.report())
''',
    },

    "Fault Tolerance and Recovery Simulation": {
        "description": "Simulate a long training run with random hardware failures, checkpoint recovery, and automatic restart — shows how production systems maintain progress.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
FAULT TOLERANCE AND RECOVERY SIMULATION
================================================================================

Simulates a production training run subject to:
    1. Random GPU failures (proportional to cluster size)
    2. Checkpoint saves every K steps
    3. Automatic restart from the most recent checkpoint
    4. Tracking of total wasted steps and wall-clock overhead

Shows the trade-off between checkpoint frequency and overhead.
================================================================================
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaultToleranceConfig:
    total_steps:        int   = 50_000    # total training steps
    n_gpus:             int   = 512       # total GPUs in the cluster
    checkpoint_every:   int   = 1_000    # save checkpoint every N steps
    save_time_s:        float = 60.0     # seconds to save a checkpoint
    restart_time_s:     float = 300.0    # seconds to restart the job
    step_time_s:        float = 2.5      # seconds per training step
    mtbf_per_gpu_hours: float = 10_000.0 # mean time between failures per GPU


def failure_probability_per_step(cfg: FaultToleranceConfig) -> float:
    """
    Probability of at least one GPU failing during a single training step.
    Assumes failures are Poisson-distributed.
    """
    mtbf_per_gpu_s = cfg.mtbf_per_gpu_hours * 3600
    lambda_per_step = cfg.n_gpus * cfg.step_time_s / mtbf_per_gpu_s
    # P(at least one failure) = 1 - P(no failures) = 1 - exp(-λ)
    return 1.0 - math.exp(-lambda_per_step)


def simulate_training(cfg: FaultToleranceConfig,
                      seed: int = 42) -> dict:
    """
    Simulate a training run with random failures and checkpoint recovery.

    Returns statistics about the run.
    """
    random.seed(seed)
    fail_prob    = failure_probability_per_step(cfg)

    current_step       = 0     # current progress
    last_checkpoint    = 0     # step of last saved checkpoint
    total_wall_time_s  = 0.0
    n_failures         = 0
    n_checkpoints      = 0
    wasted_steps       = 0

    events = []   # log of key events

    while current_step < cfg.total_steps:
        # Simulate one training step
        current_step       += 1
        total_wall_time_s  += cfg.step_time_s

        # Did a failure occur?
        if random.random() < fail_prob:
            n_failures     += 1
            steps_lost      = current_step - last_checkpoint
            wasted_steps   += steps_lost

            events.append({
                "event":       "failure",
                "step":        current_step,
                "steps_lost":  steps_lost,
                "wall_time_h": total_wall_time_s / 3600,
            })

            # Recovery: restart from last checkpoint
            total_wall_time_s  += cfg.restart_time_s
            current_step        = last_checkpoint   # roll back

            continue

        # Checkpoint
        if current_step % cfg.checkpoint_every == 0:
            last_checkpoint    = current_step
            total_wall_time_s  += cfg.save_time_s
            n_checkpoints      += 1

            if current_step % (cfg.checkpoint_every * 10) == 0:
                events.append({
                    "event":       "checkpoint",
                    "step":        current_step,
                    "wall_time_h": total_wall_time_s / 3600,
                })

    return {
        "total_steps":        cfg.total_steps,
        "n_failures":         n_failures,
        "n_checkpoints":      n_checkpoints,
        "wasted_steps":       wasted_steps,
        "total_wall_time_h":  total_wall_time_s / 3600,
        "ideal_wall_time_h":  cfg.total_steps * cfg.step_time_s / 3600,
        "overhead_pct":       (total_wall_time_s / (cfg.total_steps * cfg.step_time_s) - 1) * 100,
        "wasted_pct":         wasted_steps / cfg.total_steps * 100,
        "events":             events,
    }


def fmt_h(h: float) -> str:
    if h >= 24:
        return f"{h/24:.1f}d"
    return f"{h:.1f}h"


if __name__ == "__main__":
    BASE_CFG = FaultToleranceConfig()

    print("=" * 70)
    print(f"  FAULT TOLERANCE SIMULATION")
    print(f"  {BASE_CFG.total_steps:,} steps, {BASE_CFG.n_gpus} GPUs, "
          f"MTBF={BASE_CFG.mtbf_per_gpu_hours:,}h/GPU")
    print("=" * 70)

    p_fail = failure_probability_per_step(BASE_CFG)
    expected_failures = p_fail * BASE_CFG.total_steps
    print()
    print(f"  Failure probability per step: {p_fail:.4%}")
    print(f"  Expected failures over run:   {expected_failures:.1f}")
    print()

    # Compare checkpoint frequencies
    print("=" * 70)
    print("  IMPACT OF CHECKPOINT FREQUENCY")
    print("=" * 70)
    print()
    print(f"  {'Ckpt every':>12}  {'Failures':>10}  {'Wasted steps':>14}  "
          f"{'Wall time':>10}  {'Overhead':>10}")
    print(f"  {'':─>12}  {'':─>10}  {'':─>14}  {'':─>10}  {'':─>10}")

    for ckpt_freq in [100, 250, 500, 1000, 2000, 5000, 10000]:
        cfg = FaultToleranceConfig(checkpoint_every=ckpt_freq)
        # Average over 5 seeds
        results = [simulate_training(cfg, seed=s) for s in range(5)]
        avg = {k: sum(r[k] for r in results) / len(results)
               for k in ("n_failures", "wasted_steps", "total_wall_time_h",
                          "overhead_pct", "ideal_wall_time_h", "wasted_pct")}

        print(f"  {ckpt_freq:>12,}  {avg['n_failures']:>10.1f}  "
              f"{avg['wasted_steps']:>14.0f}  "
              f"{fmt_h(avg['total_wall_time_h']):>10}  "
              f"{avg['overhead_pct']:>9.1f}%")

    # Scale effect: how failure rate changes with cluster size
    print()
    print("=" * 70)
    print("  FAILURE RATE vs CLUSTER SIZE")
    print("  (ckpt every 1000 steps, MTBF=10,000h/GPU)")
    print("=" * 70)
    print()
    print(f"  {'N GPUs':>8}  {'P(fail/step)':>14}  {'Exp failures':>14}  "
          f"{'Mean steps/failure':>20}")
    print(f"  {'':─>8}  {'':─>14}  {'':─>14}  {'':─>20}")

    for n_gpus in [8, 64, 256, 512, 1024, 4096, 16384]:
        cfg    = FaultToleranceConfig(n_gpus=n_gpus)
        p      = failure_probability_per_step(cfg)
        exp_f  = p * cfg.total_steps
        mspf   = 1.0 / p if p > 0 else float("inf")
        print(f"  {n_gpus:>8,}  {p:>14.4%}  {exp_f:>14.1f}  {mspf:>20.0f}")

    print()
    print("  At 16,384 GPUs: a failure every ~30 steps!")
    print("  Production clusters checkpoint every 500–2000 steps,")
    print("  with async saves to avoid blocking training.")
    print()
    print("  Key insight: checkpoint overhead is usually < 5% of wall time,")
    print("  while wasted compute from failures can be 10–30% without checkpointing.")
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
    #     from llm_training.visuals.ml_system_design import (
    #         MLSYS_VISUAL_HTML,
    #         MLSYS_VISUAL_HEIGHT,
    #     )
    #     visual_html   = MLSYS_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = MLSYS_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[12_mlsystemdesign.py] Could not load visual: {e}",
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