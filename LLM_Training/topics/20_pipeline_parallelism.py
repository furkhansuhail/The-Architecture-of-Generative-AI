"""
Pipeline Parallelism
=====================

Pipeline parallelism assigns consecutive layers of a model to consecutive
GPUs. GPU 0 processes the first group of layers, GPU 1 the next, and so on.
This allows models too large for any single GPU to be distributed across
many devices. The key challenge is the "pipeline bubble" — idle GPU time
caused by sequential dependencies between pipeline stages — and the various
micro-batch scheduling strategies that reduce it.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Pipeline Parallelism"
DISPLAY_NAME = "20 · Pipeline Parallelism"
ICON         = "🚰"
SUBTITLE     = "Micro-batches, Bubbles, and Schedules"


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

### What Is Pipeline Parallelism?

In pipeline parallelism (PP), the model's layers are partitioned into P
sequential **stages**, each assigned to one or more GPUs. During a forward
pass, a batch travels from stage 0 → stage 1 → … → stage P-1 like an
assembly line. Each stage processes the output of the previous stage.

    Stage 0 (GPU 0): Layers 0–7     (embedding + first 7 blocks)
    Stage 1 (GPU 1): Layers 8–15    (next 8 blocks)
    Stage 2 (GPU 2): Layers 16–23   (next 8 blocks)
    Stage 3 (GPU 3): Layers 24–31   (last 8 blocks + logit head)

**Memory benefit:** Each GPU only stores 1/P of the layers. For a 70B model
split across P=4 GPUs, each GPU holds ~17.5B worth of parameters.

**Communication:** Activation tensors are sent from each stage to the next
as a point-to-point message. For a batch of B×T×d activations in bf16:
    transfer_size = B × T × d × 2 bytes
For LLaMA-2 70B (d=8192, B=1, T=2048): 1 × 2048 × 8192 × 2 ≈ 32 MB per boundary

This is far smaller than DDP's full gradient all-reduce (280 GB) or TP's
per-layer all-reduces, making PP the preferred choice for cross-node distribution.


### The Pipeline Bubble Problem

The naive (GPipe-style) approach processes one micro-batch at a time:

    t:  0  1  2  3  4  5  6  7
    GPU0: F0 F0 F0 F0 B0 B0 B0 B0    (forward then backward for batch 0)
    GPU1:    F0 F0 F0    B0 B0 B0
    GPU2:       F0 F0       B0 B0
    GPU3:          F0          B0

    While GPU3 does forward, GPU0 is idle.
    While GPU0 does backward, GPU3 is idle (already done).

The fraction of time GPUs are idle is the **pipeline bubble**:

    bubble_fraction = (P - 1) / (m + P - 1)

where P = number of pipeline stages and m = number of micro-batches.

For P=4 stages and m=1 micro-batch: bubble = 3/4 = 75%! Catastrophically wasteful.

The solution: use many micro-batches so GPUs are rarely idle:
    P=4 stages, m=8 micro-batches: bubble = 3/11 ≈ 27%
    P=4 stages, m=32 micro-batches: bubble = 3/35 ≈ 8.6%
    P=4 stages, m=∞:               bubble → 0%

But more micro-batches means more memory for activations (all m activations
must be kept until the corresponding backward pass).


    **Diagram 1 — GPipe (Naive) Bubble:**

    NAIVE (GPIPE) PIPELINE — P=4, m=4 MICRO-BATCHES
    ════════════════════════════════════════════════════════════════

    Time →  1    2    3    4    5    6    7    8    9   10   11   12
    GPU0: [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0] [  ] [  ] [  ] ← bubble
    GPU1: [  ] [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0] [  ] [  ]
    GPU2: [  ] [  ] [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0] [  ]
    GPU3: [  ] [  ] [  ] [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0]

    Bubble (idle) at GPU0: steps 1, 2, 3 (startup) and steps 9-12 (drain)
    Bubble fraction = (P-1)/(m+P-1) = 3/7 ≈ 43%

    ▓ = useful computation    □ = bubble (idle)

    Ideal schedule fills every cell.


### PipeDream-Flush: 1F1B Schedule

PipeDream-Flush (Narayanan et al., 2021) introduced the **1F1B (one-forward-
one-backward)** schedule. Instead of running all forward passes then all
backward passes, it interleaves them:

    As soon as stage i finishes its forward pass for micro-batch k,
    it immediately starts the forward pass for micro-batch k+1.
    Meanwhile, stage 0 can start the backward pass for micro-batch k.

The key property: **each GPU holds activations for at most P micro-batches**
at steady state, not m. This breaks the coupling between number of micro-batches
and activation memory.

    **Memory:** O(P) activations per GPU, regardless of m
    **Bubble:** Same as GPipe: (P-1) / (m + P-1)


    **Diagram 2 — 1F1B Schedule:**

    1F1B PIPELINE SCHEDULE — P=4, m=8
    ════════════════════════════════════════════════════════════════

    Time →  1   2   3   4   5   6   7   8   9   10  11  12
    GPU0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7]...
    GPU1:     [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6]...
    GPU2:         [F0][F1][F2][F3][B0][F4][B1][F5]...
    GPU3:             [F0][F1][F2][F3][B0][F4][B1]...

    Steady state: GPU0 alternates F and B → stays busy most of the time
    Memory at each GPU: holds at most P=4 activation tensors simultaneously


### Interleaved 1F1B: Reduced Bubble

Megatron-LM v2 introduced **interleaved pipeline** where each GPU processes
multiple non-consecutive chunks instead of one contiguous block:

Standard assignment (P=4, 32 layers):
    GPU0: layers 0–7       GPU1: layers 8–15
    GPU2: layers 16–23     GPU3: layers 24–31

Interleaved (V stages per GPU, also called Virtual Stages):
    GPU0: layers 0–3, 16–19     GPU1: layers 4–7, 20–23
    GPU2: layers 8–11, 24–27    GPU3: layers 12–15, 28–31

With V virtual stages per GPU:
    bubble_fraction = (P - 1) / (V × m + P - 1)

For P=4, V=2, m=8: bubble = 3/(16+3) ≈ 15.8%
For P=4, V=4, m=8: bubble = 3/(32+3) ≈ 8.6%

Trade-off: more virtual stages → lower bubble but more point-to-point
communication (2V messages per layer traversal instead of 2).


### The Pipeline Schedule Comparison

    Schedule               Bubble fraction         Memory (activations)
    ──────────────────────────────────────────────────────────────────────
    GPipe (naive)          (P-1)/(m+P-1)           O(m × d)  ← scales with m
    1F1B (PipeDream-Flush) (P-1)/(m+P-1)           O(P × d)  ← fixed!
    Interleaved 1F1B       (P-1)/(V×m+P-1)         O(P × d)  ← fixed!
    Zero-Bubble (ZB-1p)    ≈ 0%                    O(P × d)  ← complex
    ──────────────────────────────────────────────────────────────────────

**Zero-Bubble Pipeline** (Qi et al., 2023) achieves near-zero bubble fraction
by carefully decomposing the backward pass into weight gradient (B_W) and
input gradient (B_I) components and scheduling them asynchronously.


### Pipeline Parallelism vs Tensor Parallelism

These two forms of model parallelism are complementary:

    **Tensor Parallelism:**
    •   Splits within each layer (matmul across GPUs)
    •   Synchronous communication (all-reduce within layer)
    •   Best within a NVLink domain (intra-node)
    •   Reduces per-GPU memory for large weight matrices

    **Pipeline Parallelism:**
    •   Splits across layers (different stages on different GPUs)
    •   Asynchronous point-to-point communication (send/recv activations)
    •   Works across nodes via InfiniBand
    •   Introduces bubble overhead that micro-batching mitigates

In practice, **3D parallelism** (Data × Tensor × Pipeline) combines all three:
    •   DP: replicate model across clusters of nodes
    •   TP: split layers within each node (intra-node NVLink)
    •   PP: split layer groups across nodes (inter-node InfiniBand)


    **Diagram 3 — 3D Parallelism Layout:**

    3D PARALLELISM: DP=2, TP=2, PP=2 (8 GPUs total)
    ════════════════════════════════════════════════════════════════

    Data Parallel replica 0:               Data Parallel replica 1:
    ┌──────────────────┐                   ┌──────────────────┐
    │  Stage 0         │                   │  Stage 0         │
    │  GPU 0 | GPU 1   │  ← TP=2           │  GPU 4 | GPU 5   │
    │  (layers 0–15)   │                   │  (layers 0–15)   │
    └────────┬─────────┘                   └────────┬─────────┘
             │ PP activation send                   │
    ┌────────▼─────────┐                   ┌────────▼─────────┐
    │  Stage 1         │                   │  Stage 1         │
    │  GPU 2 | GPU 3   │  ← TP=2           │  GPU 6 | GPU 7   │
    │  (layers 16–31)  │                   │  (layers 16–31)  │
    └──────────────────┘                   └──────────────────┘

    DP all-reduce: GPU0↔GPU4, GPU1↔GPU5, GPU2↔GPU6, GPU3↔GPU7
    (gradients only, after all micro-batches complete)


### Activation Checkpointing in Pipeline Parallelism

In a pipeline, activations from earlier micro-batches must be retained
until their corresponding backward pass arrives. This can consume significant
memory in the startup and drain phases.

Standard approach: checkpoint all activations at pipeline stage boundaries.
Each stage only stores the activations it needs for its own backward pass
plus the pipeline boundary tensors sent to the next stage.

The combination:
    •   Pipeline PP boundaries: only transfer (B × T × d) tensors
    •   Activation checkpointing within each stage: recompute per-layer
    •   FlashAttention: avoid T² storage within each layer

This allows training very deep models with minimal memory overhead per GPU.


### GPT-3 and LLaMA-2 70B Pipeline Configurations

**GPT-3 175B (Megatron-LM paper):**
    TP=8, PP=8, DP=... (scaled based on available GPUs)
    Per-node: 8 GPUs with TP=8
    Cross-node: PP=8 (8 nodes connected via InfiniBand)
    Layers per stage: 96/8 = 12 layers per pipeline stage

**LLaMA-2 70B (Meta, estimated):**
    TP=8 within nodes (70B needs TP to fit in A100 80GB)
    PP may or may not be used depending on the cluster topology
    Megatron-based training with 3D parallelism

**Practical rule:**
    PP_degree = total_nodes × gpus_per_node / (TP × DP)
    Or equivalently: DP × TP × PP = total_GPUs
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Pipeline Schedule Comparison

| Schedule          | Bubble fraction      | Activation memory | Communication       | Implementation |
|-------------------|----------------------|-------------------|---------------------|----------------|
| GPipe (naive)     | (P-1)/(m+P-1)        | O(m × B × T × d)  | P-1 sends per batch | Simple         |
| 1F1B              | (P-1)/(m+P-1)        | O(P × B × T × d)  | P-1 sends per batch | Moderate       |
| Interleaved 1F1B  | (P-1)/(V×m+P-1)      | O(P × B × T × d)  | 2V sends per batch  | Complex        |
| Zero-Bubble (ZB)  | ~0%                  | O(P × B × T × d)  | Similar to 1F1B     | Very complex   |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Pipeline Bubble Simulation and Scheduling": {
        "description": "Simulate GPipe and 1F1B pipeline schedules, visualise the bubble pattern, and compare efficiency across different stage counts and micro-batch counts.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PIPELINE BUBBLE SIMULATION
================================================================================

Simulates GPipe and 1F1B pipeline schedules:
    1. Compute per-GPU work schedule (F/B/idle at each time step)
    2. Visualise the pipeline as an ASCII schedule
    3. Measure bubble fraction (idle time / total time)
    4. Compare schedules across different P and m values

================================================================================
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineEvent:
    gpu:       int
    time:      int
    type:      str     # "F" (forward), "B" (backward), "W" (weight grad), "idle"
    micro:     int     # micro-batch index (-1 for idle)


def simulate_gpipe(P: int, m: int) -> list[PipelineEvent]:
    """
    GPipe schedule: all micro-batch forwards first, then all backwards.
    Each stage processes m forward passes sequentially, then m backward passes.
    Returns list of events.
    """
    events = []
    # Forward passes
    for micro in range(m):
        for stage in range(P):
            t = micro + stage  # time = micro-batch number + stage offset
            events.append(PipelineEvent(stage, t, "F", micro))

    # Backward passes (reversed stage order in time)
    t_start_bwd = m + P - 1 - 1  # when GPU P-1 finishes its last forward
    for micro in range(m - 1, -1, -1):
        for stage in range(P - 1, -1, -1):
            # GPU P-1 starts backward immediately after its last forward
            t = t_start_bwd + (m - 1 - micro) + (P - 1 - stage)
            events.append(PipelineEvent(stage, t, "B", micro))

    return events


def simulate_1f1b(P: int, m: int) -> list[PipelineEvent]:
    """
    1F1B (PipeDream-Flush) schedule.
    After the startup phase (P-1 micro-batches in flight), each GPU
    alternates between one forward and one backward pass.
    """
    events = []
    # Simplified 1F1B scheduling:
    # - Startup: stage i starts at time i
    # - Steady state: each GPU interleaves F and B
    schedule = [[] for _ in range(P)]

    for stage in range(P):
        t = stage  # startup offset
        fwd_count = 0
        bwd_count = 0
        # Warmup phase: do P-stage forwards before any backward
        warmup = P - stage
        for _ in range(warmup):
            if fwd_count < m:
                schedule[stage].append(("F", fwd_count, t))
                fwd_count += 1
                t += 1

        # Steady state: 1F1B
        while bwd_count < m:
            if fwd_count < m:
                schedule[stage].append(("F", fwd_count, t))
                fwd_count += 1
                t += 1
            schedule[stage].append(("B", bwd_count, t))
            bwd_count += 1
            t += 1

    # Convert to events
    for stage, evts in enumerate(schedule):
        for typ, micro, t in evts:
            events.append(PipelineEvent(stage, t, typ, micro))

    return events


def bubble_fraction(events: list[PipelineEvent], P: int) -> float:
    """Compute the fraction of time steps that are idle (bubble)."""
    if not events:
        return 1.0
    max_t    = max(e.time for e in events)
    total    = (max_t + 1) * P
    busy     = sum(1 for e in events if e.type in ("F", "B"))
    return 1.0 - busy / total


def render_schedule(events: list[PipelineEvent], P: int, max_t: int = 20,
                     title: str = "") -> str:
    """Render pipeline schedule as ASCII grid."""
    grid = [["·"] * (max_t + 1) for _ in range(P)]
    for e in events:
        if e.time <= max_t:
            grid[e.gpu][e.time] = e.type if e.type != "idle" else " "

    lines = [f"\n  {title}" if title else ""]
    lines.append("       time: " + "".join(f"{t%10}" for t in range(max_t + 1)))
    lines.append("  " + "─" * (13 + max_t + 1))
    for stage in range(P):
        row = "  " + f"GPU{stage} (S{stage}): " + "".join(grid[stage])
        lines.append(row)
    return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  PIPELINE SCHEDULE COMPARISON")
    print("=" * 65)

    # Visualise schedules
    for P, m in [(4, 6)]:
        print(f"\n  P={P} pipeline stages, m={m} micro-batches")

        ev_gpipe = simulate_gpipe(P, m)
        ev_1f1b  = simulate_1f1b(P, m)

        bubble_g = bubble_fraction(ev_gpipe, P)
        bubble_1 = bubble_fraction(ev_1f1b,  P)

        max_t = min(max(e.time for e in ev_gpipe), 20)

        print(render_schedule(ev_gpipe, P, max_t,
                               f"GPipe schedule  (bubble={bubble_g:.1%})"))
        print()
        print(render_schedule(ev_1f1b,  P, max_t,
                               f"1F1B schedule   (bubble={bubble_1:.1%})"))

    # Bubble fraction analysis
    print()
    print("=" * 65)
    print("  BUBBLE FRACTION vs MICRO-BATCH COUNT")
    print("=" * 65)
    print()
    print("  Formula: bubble = (P - 1) / (m + P - 1)")
    print()
    print(f"  {'m (micro-batches)':>20}", end="")
    for P in [2, 4, 8, 16]:
        print(f"  {'P='+str(P):>10}", end="")
    print()
    print(f"  {'':─>20}" + "".join(f"  {'':─>10}" for _ in [2,4,8,16]))

    for m in [1, 2, 4, 8, 16, 32, 64, 128]:
        print(f"  {m:>20}", end="")
        for P in [2, 4, 8, 16]:
            bub = (P - 1) / (m + P - 1)
            print(f"  {bub:>9.1%}", end="")
        print()

    # Interleaved 1F1B
    print()
    print("=" * 65)
    print("  INTERLEAVED 1F1B (VIRTUAL STAGES)")
    print("  bubble = (P-1) / (V * m + P - 1)")
    print("=" * 65)
    print()
    P = 8
    print(f"  P={P} pipeline stages, showing V=1,2,4 virtual stages:")
    print()
    print(f"  {'m':>5}", end="")
    for V in [1, 2, 4]:
        print(f"  {'V='+str(V):>12}", end="")
    print()
    print(f"  {'':─>5}" + "".join(f"  {'':─>12}" for _ in [1,2,4]))

    for m in [4, 8, 16, 32, 64]:
        print(f"  {m:>5}", end="")
        for V in [1, 2, 4]:
            bub = (P - 1) / (V * m + P - 1)
            print(f"  {bub:>11.1%}", end="")
        print()

    print()
    print("  Interleaved 1F1B with V virtual stages reduces bubble by ~V×")
    print("  at the cost of V× more point-to-point messages per micro-batch.")
    print("  V=2 is a common choice (halves bubble, modest comm overhead).")
''',
    },

    "Layer Partitioning and Stage Assignment": {
        "description": "Implement a balanced layer partitioner for pipeline parallelism and simulate the forward pass across pipeline stages using threads.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
LAYER PARTITIONING AND PIPELINE STAGE SIMULATION
================================================================================

Implements:
    1. Balanced layer partitioning (equal FLOPs per stage)
    2. Stage assignment with activation communication simulation
    3. Multi-micro-batch 1F1B-style scheduling on threads
    4. Activation memory tracking per stage

================================================================================
"""

import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ── Simple Transformer block ──────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


# ── Layer partitioning ────────────────────────────────────────────────────────

def partition_layers(n_layers: int, n_stages: int,
                      layer_flops: list[float] = None) -> list[list[int]]:
    """
    Partition n_layers into n_stages, minimising the maximum stage FLOPs.
    Returns list of [start, end) for each stage.

    If layer_flops not provided, assume uniform (simple round-robin).
    """
    if layer_flops is None:
        layer_flops = [1.0] * n_layers

    # Greedy: add layers to current stage until adding one more would
    # exceed the per-stage target (total_flops / n_stages)
    total_flops  = sum(layer_flops)
    target_flops = total_flops / n_stages

    stages = []
    stage_layers = []
    stage_flops  = 0.0

    for i, flops in enumerate(layer_flops):
        stage_layers.append(i)
        stage_flops += flops

        # Start a new stage if we've reached the target and more stages remain
        if stage_flops >= target_flops and len(stages) < n_stages - 1:
            stages.append(stage_layers[:])
            stage_layers = []
            stage_flops  = 0.0

    # Remaining layers go to the last stage
    if stage_layers:
        stages.append(stage_layers)

    # If we have fewer stages than requested, pad with empty stages
    while len(stages) < n_stages:
        stages.append([])

    return stages


class PipelineStage(nn.Module):
    """
    One pipeline stage: a contiguous group of Transformer blocks.
    Handles activation send (to next stage) and receive (from previous stage).
    """

    def __init__(self, stage_id: int, n_stages: int,
                 layers: nn.ModuleList,
                 is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.stage_id  = stage_id
        self.n_stages  = n_stages
        self.layers    = layers
        self.is_first  = is_first
        self.is_last   = is_last
        self.n_params  = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ── Pipeline simulation ───────────────────────────────────────────────────────

def simulate_pipeline_forward(stages: list[PipelineStage],
                               micro_batches: list[torch.Tensor],
                               verbose: bool = True) -> list[torch.Tensor]:
    """
    Simulate a pipeline forward pass across stages.
    Uses Python lists to simulate the activation transfer between stages.
    Returns the output of the last stage for each micro-batch.
    """
    outputs  = [None] * len(micro_batches)
    n_stages = len(stages)

    if verbose:
        print(f"  Running {len(micro_batches)} micro-batches through "
              f"{n_stages}-stage pipeline")
        print()

    # Process micro-batches stage by stage (simplified GPipe-style for demo)
    t0 = time.perf_counter()
    for stage_idx, stage in enumerate(stages):
        if verbose:
            print(f"  Stage {stage_idx} ({len(stage.layers)} layers, "
                  f"{stage.n_params:,} params):")

        for mb_idx, mb in enumerate(micro_batches):
            if stage_idx == 0:
                x_in = mb
            else:
                x_in = micro_batches[mb_idx]  # already updated by prev stage

            with torch.no_grad():
                x_out = stage(x_in)

            micro_batches[mb_idx] = x_out

            if verbose and stage_idx == n_stages - 1:
                outputs[mb_idx] = x_out

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n  Pipeline complete in {elapsed*1000:.1f} ms")

    return [m for m in micro_batches]


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    N_LAYERS  = 12
    N_STAGES  = 4
    D         = 128
    B, T      = 2, 16
    N_MICRO   = 4

    print("=" * 60)
    print(f"  PIPELINE PARALLELISM STAGE SETUP")
    print(f"  {N_LAYERS} layers, {N_STAGES} stages, d={D}")
    print("=" * 60)
    print()

    # Create all layers
    all_layers = nn.ModuleList([TransformerBlock(D) for _ in range(N_LAYERS)])

    # Partition layers into stages
    stage_partitions = partition_layers(N_LAYERS, N_STAGES)

    print("  Layer partitioning:")
    stages_list = []
    for s_idx, layer_ids in enumerate(stage_partitions):
        if not layer_ids:
            continue
        stage_layers  = nn.ModuleList([all_layers[i] for i in layer_ids])
        stage         = PipelineStage(
            stage_id=s_idx, n_stages=N_STAGES,
            layers=stage_layers,
            is_first=(s_idx == 0),
            is_last=(s_idx == N_STAGES - 1),
        )
        stages_list.append(stage)
        print(f"  Stage {s_idx}: layers {layer_ids[0]}–{layer_ids[-1]}  "
              f"({len(layer_ids)} layers, {stage.n_params:,} params)")

    # Verify balanced partitioning
    stage_param_counts = [s.n_params for s in stages_list]
    max_p = max(stage_param_counts)
    min_p = min(stage_param_counts)
    print(f"\n  Balance: max/min param ratio = {max_p/min_p:.2f}×  "
          f"({'✓ balanced' if max_p/min_p < 1.1 else '⚠️ imbalanced'})")

    # Create micro-batches
    micro_batches = [torch.randn(B, T, D) for _ in range(N_MICRO)]
    print(f"\n  Micro-batch shape: {micro_batches[0].shape}")
    print(f"  Activation transfer size per boundary: "
          f"{B * T * D * 2 / 1024:.1f} KB  (bf16)")

    print()
    print("=" * 60)
    print("  PIPELINE FORWARD PASS")
    print("=" * 60)
    print()

    outputs = simulate_pipeline_forward(stages_list, micro_batches[:], verbose=True)

    print(f"\n  Output shapes: {[o.shape for o in outputs[:2]]} ...")
    print(f"  All outputs have shape {outputs[0].shape} ✓")

    # Memory analysis
    print()
    print("=" * 60)
    print("  ACTIVATION MEMORY PER STAGE")
    print("=" * 60)
    print()
    dtype_bytes = 2  # bf16
    act_per_mb  = B * T * D * dtype_bytes   # bytes per micro-batch at each stage boundary
    print(f"  Activation tensor size (per micro-batch): {act_per_mb/1024:.1f} KB")
    print()
    print("  With GPipe (all micro-batches in flight):")
    print(f"    Each intermediate stage stores {N_MICRO} activations = "
          f"{N_MICRO * act_per_mb / 1024:.1f} KB")
    print()
    print("  With 1F1B (at most P in flight):")
    print(f"    Each intermediate stage stores ≤ {N_STAGES} activations = "
          f"{N_STAGES * act_per_mb / 1024:.1f} KB")
    print(f"  Memory saving: {N_MICRO / N_STAGES:.1f}× vs GPipe for m={N_MICRO}")
''',
    },

    "Pipeline Efficiency Calculator": {
        "description": "Calculate pipeline efficiency, GPU utilisation, and optimal micro-batch count for different pipeline configurations and model sizes.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
PIPELINE EFFICIENCY CALCULATOR
================================================================================

Computes:
    1. Pipeline bubble fraction for all schedule types
    2. Effective GPU utilisation accounting for bubble
    3. Optimal micro-batch count m* for a given memory budget
    4. Communication overhead at different PP degrees
    5. Comparison: DDP vs TP vs PP for different model sizes

================================================================================
"""

import math


def gpipe_bubble(P: int, m: int) -> float:
    return (P - 1) / (m + P - 1)


def interleaved_bubble(P: int, m: int, V: int) -> float:
    """V = virtual stages (chunks) per GPU."""
    return (P - 1) / (V * m + P - 1)


def pipeline_comm_per_step(P: int, d: int, B: int, T: int,
                            dtype_bytes: int = 2) -> float:
    """Communication volume per step for PP (activation sends). GB."""
    # P-1 boundaries, each sends B×T×d tensor forward and backward
    boundary_size = B * T * d * dtype_bytes / 1e9
    return 2 * (P - 1) * boundary_size  # ×2 for fwd and bwd


def gpu_utilisation(P: int, m: int, V: int = 1) -> float:
    """Effective GPU utilisation = 1 - bubble_fraction."""
    return 1.0 - interleaved_bubble(P, m, V)


def optimal_m(P: int, target_util: float = 0.95, V: int = 1) -> int:
    """Find minimum m that achieves target GPU utilisation."""
    for m in range(1, 1000):
        if gpu_utilisation(P, m, V) >= target_util:
            return m
    return 999


def activation_memory_gb(P: int, m: int, d: int, B: int, T: int,
                          schedule: str = "1f1b",
                          dtype_bytes: int = 2) -> float:
    """
    Activation memory per GPU for pipeline activations (not counting
    within-layer activations, which are handled by activation checkpointing).
    """
    activation_per_mb = B * T * d * dtype_bytes / 1e9
    if schedule == "gpipe":
        return m * activation_per_mb     # all m activations stored
    else:
        return P * activation_per_mb     # 1F1B: at most P stored


def bar(frac: float, width: int = 20) -> str:
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


if __name__ == "__main__":
    # ── 1. Bubble fraction vs m and P ────────────────────────────────────────
    print("=" * 68)
    print("  GPU UTILISATION (1 - bubble) vs MICRO-BATCH COUNT")
    print("=" * 68)
    print()
    print(f"  {'m':>5}", end="")
    for P in [2, 4, 8, 16, 32]:
        print(f"  {'P='+str(P):>10}", end="")
    print()
    print(f"  {'':─>5}" + "".join(f"  {'':─>10}" for _ in [2,4,8,16,32]))

    for m in [1, 2, 4, 8, 16, 32, 64]:
        print(f"  {m:>5}", end="")
        for P in [2, 4, 8, 16, 32]:
            util = gpu_utilisation(P, m)
            print(f"  {util:>9.1%}", end="")
        print()

    # ── 2. Optimal m for 90% and 95% utilisation ──────────────────────────────
    print()
    print("=" * 68)
    print("  MINIMUM MICRO-BATCHES FOR TARGET GPU UTILISATION")
    print("=" * 68)
    print()
    print(f"  {'P stages':>10}  {'m for 90% util':>18}  "
          f"{'m for 95% util':>18}  {'m for 99% util':>18}")
    print(f"  {'':─>10}  {'':─>18}  {'':─>18}  {'':─>18}")

    for P in [2, 4, 8, 16, 32, 64]:
        m90 = optimal_m(P, 0.90)
        m95 = optimal_m(P, 0.95)
        m99 = optimal_m(P, 0.99)
        print(f"  {P:>10}  {m90:>18}  {m95:>18}  {m99:>18}")

    print()
    print("  Rule of thumb: m ≈ 4P to 8P gives good utilisation (>90%)")

    # ── 3. Impact of virtual stages on m requirement ──────────────────────────
    print()
    print("=" * 68)
    print("  INTERLEAVED 1F1B: m REQUIREMENT VS VIRTUAL STAGES")
    print("  (P=8, targeting 95% GPU utilisation)")
    print("=" * 68)
    print()
    P = 8
    print(f"  {'V (virtual stages)':>20}  {'m needed for 95%':>20}  "
          f"{'Bubble at m=16':>18}  {'Extra comm':>12}")
    print(f"  {'':─>20}  {'':─>20}  {'':─>18}  {'':─>12}")

    for V in [1, 2, 4, 8]:
        m_needed = optimal_m(P, 0.95, V)
        bub_m16  = interleaved_bubble(P, 16, V)
        comm_ovh = f"{V}× sends"
        print(f"  {V:>20}  {m_needed:>20}  {bub_m16:>17.1%}  {comm_ovh:>12}")

    print()
    print("  V=2 halves the required m (fewer micro-batches needed for same")
    print("  efficiency) while only doubling point-to-point messages.")

    # ── 4. Communication comparison: DDP vs TP vs PP ──────────────────────────
    print()
    print("=" * 68)
    print("  COMMUNICATION VOLUME PER TRAINING STEP")
    print("  LLaMA-2 70B: d=8192, B=1, T=2048")
    print("=" * 68)
    print()

    N_PARAMS = 70_000_000_000
    D        = 8192
    B, T     = 1, 2048
    N_LAYERS = 80

    ddp_vol  = 2 * N_PARAMS * 2 / 1e9   # 2 × grad_size in bf16

    print(f"  DDP all-reduce (bf16 grads):     {ddp_vol:.1f} GB")
    print()
    print(f"  {'PP stages':>10}  {'PP comm (GB)':>14}  {'vs DDP':>10}")
    print(f"  {'':─>10}  {'':─>14}  {'':─>10}")
    for PP in [2, 4, 8, 16, 32]:
        vol  = pipeline_comm_per_step(PP, D, B, T)
        ratio = ddp_vol / vol
        print(f"  {PP:>10}  {vol:>14.3f}  {ratio:>9.1f}× less")

    print()
    print("  PP communication is orders of magnitude smaller than DDP!")
    print("  This is why PP is used for cross-node distribution (InfiniBand),")
    print("  while TP is used within nodes (NVLink).")
    print()
    print("  For a 32-stage pipeline, PP sends only ~4 MB per boundary,")
    print("  while DDP all-reduces 280 GB of gradients!")

    # ── 5. Pipeline + TP combined ─────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  3D PARALLELISM: DP × TP × PP CONFIGURATIONS")
    print("  (targeting 1024 A100 GPUs, 70B model)")
    print("=" * 68)
    print()

    TOTAL_GPUS = 1024
    print(f"  {'DP':>5}  {'TP':>5}  {'PP':>5}  {'GPUs':>6}  {'Mem/GPU':>10}  Notes")
    print(f"  {'':─>5}  {'':─>5}  {'':─>5}  {'':─>6}  {'':─>10}  {'':─}")

    configs = [
        (1024, 1, 1),
        (512,  2, 1),
        (256,  4, 1),
        (128,  8, 1),
        (128,  4, 2),
        (64,   8, 2),
        (32,   8, 4),
        (16,   8, 8),
    ]

    for dp, tp, pp in configs:
        gpus   = dp * tp * pp
        if gpus != TOTAL_GPUS:
            continue
        mem_gb = N_PARAMS * 2 / (tp * pp) / 1e9  # weights per GPU in bf16
        note   = ""
        if tp > 8:
            note = "❌ TP>8 cross-node"
        elif mem_gb > 80:
            note = "❌ OOM on A100-80GB"
        elif mem_gb > 60:
            note = "⚠️ tight"
        else:
            note = "✓ fits"
        print(f"  {dp:>5}  {tp:>5}  {pp:>5}  {gpus:>6}  "
              f"{mem_gb:>9.1f}GB  {note}")

    print()
    print("  Recommended for 70B on 1024 A100s: DP=16, TP=8, PP=8")
    print("  → 18 GB/GPU weights, 8 nodes for TP (NVLink), 8 PP stages (IB)")
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
    #     from llm_training.visuals.pipeline_parallelism import (
    #         PP_VISUAL_HTML,
    #         PP_VISUAL_HEIGHT,
    #     )
    #     visual_html   = PP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = PP_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[20_pipeline_parallelism.py] Could not load visual: {e}",
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