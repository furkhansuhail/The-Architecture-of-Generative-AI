"""
3D Parallelism — Megatron-LM and DeepSpeed
============================================

Training models at the scale of GPT-3 (175B), PaLM (540B), or Llama-4
requires combining all three axes of model parallelism simultaneously:
Data Parallelism for throughput, Tensor Parallelism for large weight
matrices within a node, and Pipeline Parallelism for layer distribution
across nodes. The orchestration of these three strategies — their
interactions, communication patterns, and optimal degree selection —
is the topic of this module.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "3D Parallelism — Megatron-LM and DeepSpeed"
DISPLAY_NAME = "23 · 3D Parallelism"
ICON         = "🌐"
SUBTITLE     = "Megatron-LM and DeepSpeed Combined"


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

### The Three Axes of Parallelism

Each form of parallelism addresses a different bottleneck:

    Parallelism Type   What it parallelises       Memory saved        Communication
    ─────────────────────────────────────────────────────────────────────────────────
    Data (DP)          Training data              Optimizer state     All-reduce (grads)
    Tensor (TP)        Individual weight matrices  Weights + activations All-reduce (layers)
    Pipeline (PP)      Layers (stage assignment)  Weights             Send/recv (activations)
    ─────────────────────────────────────────────────────────────────────────────────

None of these is sufficient alone for frontier-scale models:
    •   DP alone: model must fit on one GPU (70B = 140 GB — doesn't fit on A100 80GB)
    •   TP alone: all-reduce is intra-node fast but cross-node slow; limited to TP=8
    •   PP alone: bubble overhead wastes GPU time

**3D parallelism** combines all three optimally, exploiting the network
hierarchy of a modern GPU cluster:
    •   TP operates within one NVLink node  (high bandwidth, synchronous)
    •   PP operates across nodes            (InfiniBand, point-to-point)
    •   DP operates across PP×TP groups    (IB or Ethernet, infrequent)


### The Megatron-LM 3D Parallelism Layout

Megatron-LM (Narayanan et al., 2021) introduced the canonical 3D parallelism
layout for transformers. Consider a cluster of 16 GPUs (2 nodes × 8 GPUs/node):

    DP=2, TP=4, PP=2  →  2 × 4 × 2 = 16 GPUs

**Node layout:**
    Node 0: [GPU00 GPU01 GPU02 GPU03] [GPU04 GPU05 GPU06 GPU07]
                 ↑── TP group 0 ──↑        ↑── TP group 1 ──↑
             Stage 0 (layers 0–15)         Stage 1 (layers 16–31)

    Node 1: [GPU10 GPU11 GPU12 GPU13] [GPU14 GPU15 GPU16 GPU17]
             Stage 0 (layers 0–15)         Stage 1 (layers 16–31)
             ↑ DP replica of Node 0 ↑

    TP communication: within [GPU00–GPU03] (NVLink, fast)
    PP communication: GPU03→GPU04 (NVLink, fast) or GPU07→GPU10 (IB, slower)
    DP communication: GPU00↔GPU10 (IB, infrequent — once per PP pass)


    **Diagram 1 — 3D Parallelism GPU Assignment:**

    3D PARALLELISM: DP=2, TP=4, PP=2 (16 GPUs, 2 nodes)
    ════════════════════════════════════════════════════════════════

                   ← TP group (within node, NVLink) →

    DP replica 0   [ GPU0  GPU1  GPU2  GPU3 ]  Stage 0  (layers 0–15)
    Node 0         [ GPU4  GPU5  GPU6  GPU7 ]  Stage 1  (layers 16–31)
                        ↕ PP (between stages, fast NVLink)

    DP replica 1   [ GPU8  GPU9  GPU10 GPU11 ] Stage 0  (layers 0–15)
    Node 1         [ GPU12 GPU13 GPU14 GPU15 ] Stage 1  (layers 16–31)
                        ↕ PP (between stages, IB)

    DP all-reduce: GPU0↔GPU8, GPU1↔GPU9, ..., GPU7↔GPU15
    (only once per effective batch, after all micro-batches complete)

    Total GPU count: DP × TP × PP = 2 × 4 × 2 = 16 ✓


### How 3D Parallelism Decomposes

For a model with N total layers and DP×TP×PP = total_GPUs:

    **Layers per pipeline stage:**
        layers_per_stage = N / PP
        Each stage is further TP-parallel within it.

    **Memory per GPU:**
        weights    ≈ P_total / (TP × PP) bytes    (each GPU owns 1/(TP×PP) of weights)
        optimizer  ≈ P_total / (TP × PP × DP)     (with ZeRO Stage 1)
        activations ≈ B × T × d × layers_per_stage / TP   (with SP+ckpt)

    **Throughput:**
        Effective tokens per step = micro_batch × T × DP × PP_stages × m
        (m = micro-batches per PP pass, PP handles the temporal pipelining)


### Communication Pattern Hierarchy

The 3D communication hierarchy matches the hardware bandwidth hierarchy:

    Operation             Frequency    Volume per step    Bandwidth used
    ──────────────────────────────────────────────────────────────────────────
    TP all-reduce (fwd)   Per layer    B × T × d          NVLink (600 GB/s)
    PP send/recv          Per stage    B × T × d_boundary  NVLink or IB
    DP all-reduce (grad)  Per batch    P / (TP × PP) × 2   IB (25 GB/s)
    ──────────────────────────────────────────────────────────────────────────

Key insight: DP all-reduce is the most expensive (large volume, slow IB)
but it happens least frequently (only after completing all micro-batches
in a pipeline pass). By choosing DP as the "outermost" parallelism (across
nodes), we make the expensive operation infrequent.


### Megatron-LM: The Reference Implementation

Megatron-LM (NVIDIA) is the de-facto reference implementation for 3D
parallelism in LLM training. Key features:

    **Tensor Parallelism:**
    •   Column-parallel + Row-parallel FFN (Module 19)
    •   Head-split attention (Module 19)
    •   Sequence parallelism for non-TP activations (Module 21)

    **Pipeline Parallelism:**
    •   1F1B and interleaved 1F1B schedules (Module 20)
    •   Activation checkpointing at stage boundaries
    •   Communication-computation overlap in PP

    **Data Parallelism:**
    •   DDP + ZeRO Stage 1 (optimizer state sharding across DP replicas)
    •   Optional ZeRO Stage 2 for gradient sharding

    **Process Group Layout:**
    Every GPU belongs to exactly three process groups:
        1. TP group (fast, intra-node)
        2. PP group (sequential, point-to-point)
        3. DP group (infrequent, global all-reduce)


### DeepSpeed ZeRO in 3D Parallelism

While Megatron-LM handles TP and PP, DeepSpeed's ZeRO handles the
data-parallel dimension by sharding the optimizer state, gradients,
and even parameters across DP replicas.

The combination "Megatron-DeepSpeed" or "3D parallelism + ZeRO":

    ZeRO Stage 1: optimizer state sharded across DP
        → Memory per GPU: Optimizer / DP  (e.g., divide 56 GB by DP=32)

    ZeRO Stage 2: + gradients sharded
        → Further reduces per-GPU memory

    ZeRO Stage 3: + parameters sharded
        → Each GPU holds only 1/(TP×PP×DP) of the model
        → Maximum memory reduction, highest communication overhead

For most 3D parallelism setups, ZeRO Stage 1 or 2 is used (not Stage 3,
which conflicts with TP's assumption that weights are permanently resident).


### GPU Assignment and Process Group Construction

Given a flat list of GPU ranks [0, 1, 2, …, total-1], the process groups are
constructed as follows (for DP×TP×PP layout):

    **TP groups** (ranks within same TP degree, same stage):
        For DP=2, TP=4, PP=2 (total=16):
        TP groups: [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]

    **PP groups** (ranks across stages, same TP position and DP replica):
        PP groups: [0,4], [1,5], [2,6], [3,7],
                   [8,12], [9,13], [10,14], [11,15]

    **DP groups** (ranks with same TP position and stage):
        DP groups: [0,8], [1,9], [2,10], [3,11],
                   [4,12], [5,13], [6,14], [7,15]

Each GPU belongs to exactly one group of each type.


### Optimal Degree Selection

For a cluster of N total GPUs training a model of P parameters:

    Step 1: Set TP = min(N_per_node, max_useful_TP)
            max_useful_TP ≤ 8 (NVLink bound)
            max_useful_TP ≤ min(n_heads, n_kv_heads)
            Example: TP = 8 for a 70B+ model on DGX A100

    Step 2: Set PP to fit the model per TP group
            Min PP: ceil(P × bytes / (TP × GPU_memory_GB))
            Larger PP reduces per-GPU memory but increases bubble
            Example: PP = 4 for 70B on 80GB A100s with TP=8

    Step 3: Set DP = total_GPUs / (TP × PP)
            DP provides throughput and converges faster (larger effective batch)
            Example: with 512 GPUs, DP = 512 / (8 × 4) = 16

    Step 4: Set micro-batches m per PP pass to reduce bubble
            Recommended: m ≥ 4×PP for < 20% bubble
            Example: m = 16 for PP=4 → bubble ≈ 15%

    Step 5: Compute global batch size
            global_batch = micro_batch × T × DP × m
            Check: this should be 1M–8M tokens for LLM training


### Real-World 3D Parallelism Configurations

    Model         Params   GPUs    DP    TP    PP    Notes
    ──────────────────────────────────────────────────────────────────────
    GPT-3 175B     175B    1024   8     8     16    Megatron paper
    Megatron 530B  530B    2240   —     8     35    —
    PaLM 540B      540B    ~6k    —     12    —     TPU-based, different topology
    LLaMA-2 70B    70B     512    8     8     8     Internal estimates
    DeepSeek-V3    671B    ~2k    —     1(EP) TP    MoE + heavy TP, no PP
    ──────────────────────────────────────────────────────────────────────

Note: many production configs are proprietary. These are estimates/publications.


### Communication Overlap in 3D Parallelism

The key to efficiency is maximising overlap between computation and communication:

    **TP communication overlap:**
    TP all-reduces are synchronous and cannot be overlapped easily within a layer.
    However, SP (Module 21) replaces TP all-reduce with AG + RS, which can
    partially overlap with the next layer's computation.

    **PP communication overlap:**
    Pipeline send/recv can be overlapped with backward computation using
    non-blocking NCCL operations. As Stage i computes backward for micro-batch k,
    it sends forward activation of micro-batch k+1 to Stage i+1 simultaneously.

    **DP all-reduce overlap:**
    DDP overlaps gradient all-reduce with backward computation (bucket-level,
    Module 18). In 3D, the DP gradient reduce happens after all micro-batches
    in a PP pass, so it is naturally overlapped with the next PP pass's startup.

A well-tuned 3D training run achieves >80% MFU, meaning 80%+ of peak GPU
compute is used productively. Poor overlap drops this to 40–60%.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
3D Parallelism Degree Selection Guide

| Constraint                        | Recommended action                                      |
|-----------------------------------|---------------------------------------------------------|
| Model too large for 1 GPU         | Increase TP (up to 8, intra-node NVLink)                |
| Model still too large after TP=8  | Increase PP (inter-node, any size)                      |
| Need more throughput              | Increase DP (multiply total GPUs, cheap all-reduce)     |
| Too much bubble                   | Increase micro-batch m per PP pass (m ≥ 4×PP)           |
| Communication bound (DP)          | Add ZeRO Stage 1/2 to reduce DP communication volume    |
| MFU below 50%                     | Profile: likely data loading or communication bottleneck|
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "3D Parallelism Process Group Calculator": {
        "description": "Given DP, TP, PP degrees and total GPUs, compute the exact process group membership for each GPU rank — TP group, PP group, and DP group.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
3D PARALLELISM PROCESS GROUP CALCULATOR
================================================================================

Computes the process group assignments for all GPUs in a 3D parallel setup.
Replicates the logic used by Megatron-LM's parallel_state module.

For a DP × TP × PP grid, each GPU rank belongs to exactly:
    - One TP group  (intra-node, synchronous all-reduce)
    - One PP group  (point-to-point pipeline communication)
    - One DP group  (gradient synchronisation across replicas)

================================================================================
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProcessGroups:
    """Process group assignments for all GPUs in a 3D parallel setup."""
    dp:   int
    tp:   int
    pp:   int
    total_gpus: int = 0

    # Per-rank assignments
    tp_groups:  list[list[int]] = field(default_factory=list)
    pp_groups:  list[list[int]] = field(default_factory=list)
    dp_groups:  list[list[int]] = field(default_factory=list)

    # Reverse lookup: rank → group index
    rank_to_tp: dict[int, int] = field(default_factory=dict)
    rank_to_pp: dict[int, int] = field(default_factory=dict)
    rank_to_dp: dict[int, int] = field(default_factory=dict)

    # Per-rank local ranks within each group
    rank_to_tp_rank: dict[int, int] = field(default_factory=dict)
    rank_to_pp_rank: dict[int, int] = field(default_factory=dict)
    rank_to_dp_rank: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.total_gpus = self.dp * self.tp * self.pp


def build_process_groups(dp: int, tp: int, pp: int) -> ProcessGroups:
    """
    Build process group assignments for a DP×TP×PP parallel setup.

    GPU rank assignment (Megatron convention):
        rank = dp_rank * (tp * pp) + pp_rank * tp + tp_rank

    Or equivalently, iterating over a 3D grid [dp][pp][tp]:
        rank = dp_idx * (tp * pp) + pp_idx * tp + tp_idx
    """
    g = ProcessGroups(dp=dp, tp=tp, pp=pp)
    N = dp * tp * pp

    # Build flat rank → 3D position mapping
    rank_to_pos = {}
    for dp_r in range(dp):
        for pp_r in range(pp):
            for tp_r in range(tp):
                rank = dp_r * (tp * pp) + pp_r * tp + tp_r
                rank_to_pos[rank] = (dp_r, pp_r, tp_r)

    # ── TP groups: same dp_rank and pp_rank, all tp_ranks ─────────────────────
    tp_group_map = {}   # (dp_r, pp_r) → [ranks]
    for rank, (dp_r, pp_r, tp_r) in rank_to_pos.items():
        key = (dp_r, pp_r)
        tp_group_map.setdefault(key, []).append(rank)

    g.tp_groups = [sorted(v) for v in tp_group_map.values()]
    for gidx, group in enumerate(g.tp_groups):
        for local_rank, rank in enumerate(group):
            g.rank_to_tp[rank]      = gidx
            g.rank_to_tp_rank[rank] = local_rank

    # ── PP groups: same dp_rank and tp_rank, all pp_ranks ─────────────────────
    pp_group_map = {}   # (dp_r, tp_r) → [ranks]
    for rank, (dp_r, pp_r, tp_r) in rank_to_pos.items():
        key = (dp_r, tp_r)
        pp_group_map.setdefault(key, []).append(rank)

    g.pp_groups = [sorted(v) for v in pp_group_map.values()]
    for gidx, group in enumerate(g.pp_groups):
        for local_rank, rank in enumerate(sorted(group,
                key=lambda r: rank_to_pos[r][1])):  # sort by pp_rank
            g.rank_to_pp[rank]      = gidx
            g.rank_to_pp_rank[rank] = local_rank

    # ── DP groups: same pp_rank and tp_rank, all dp_ranks ─────────────────────
    dp_group_map = {}   # (pp_r, tp_r) → [ranks]
    for rank, (dp_r, pp_r, tp_r) in rank_to_pos.items():
        key = (pp_r, tp_r)
        dp_group_map.setdefault(key, []).append(rank)

    g.dp_groups = [sorted(v) for v in dp_group_map.values()]
    for gidx, group in enumerate(g.dp_groups):
        for local_rank, rank in enumerate(group):
            g.rank_to_dp[rank]      = gidx
            g.rank_to_dp_rank[rank] = local_rank

    return g


def print_process_groups(g: ProcessGroups):
    N = g.total_gpus
    print(f"  Configuration: DP={g.dp}, TP={g.tp}, PP={g.pp}")
    print(f"  Total GPUs:    {N}")
    print()

    print(f"  {'Rank':>5}  {'TP grp':>8}  {'TP local':>9}  "
          f"{'PP grp':>8}  {'PP local':>9}  {'DP grp':>8}  {'DP local':>9}")
    print(f"  {'':─>5}  {'':─>8}  {'':─>9}  "
          f"{'':─>8}  {'':─>9}  {'':─>8}  {'':─>9}")

    for rank in range(N):
        print(f"  {rank:>5}  {g.rank_to_tp[rank]:>8}  {g.rank_to_tp_rank[rank]:>9}  "
              f"{g.rank_to_pp[rank]:>8}  {g.rank_to_pp_rank[rank]:>9}  "
              f"{g.rank_to_dp[rank]:>8}  {g.rank_to_dp_rank[rank]:>9}")

    print()
    print(f"  TP groups ({len(g.tp_groups)} total):")
    for i, grp in enumerate(g.tp_groups):
        print(f"    [{i}]: {grp}")

    print(f"\n  PP groups ({len(g.pp_groups)} total):")
    for i, grp in enumerate(g.pp_groups):
        print(f"    [{i}]: {sorted(grp, key=lambda r: g.rank_to_pp_rank[r])}")

    print(f"\n  DP groups ({len(g.dp_groups)} total):")
    for i, grp in enumerate(g.dp_groups):
        print(f"    [{i}]: {grp}")


if __name__ == "__main__":
    # ── Example 1: Small cluster (16 GPUs) ────────────────────────────────────
    print("=" * 65)
    print("  EXAMPLE 1: DP=2, TP=4, PP=2 (16 GPUs)")
    print("=" * 65)
    g1 = build_process_groups(dp=2, tp=4, pp=2)
    print_process_groups(g1)

    # ── Example 2: Verify properties ─────────────────────────────────────────
    print()
    print("=" * 65)
    print("  VERIFICATION: Every rank belongs to exactly one group of each type")
    print("=" * 65)
    for dp, tp, pp in [(2,4,2), (4,8,1), (2,8,4), (8,8,8)]:
        g   = build_process_groups(dp, tp, pp)
        N   = dp * tp * pp
        ok1 = all(r in g.rank_to_tp for r in range(N))
        ok2 = all(r in g.rank_to_pp for r in range(N))
        ok3 = all(r in g.rank_to_dp for r in range(N))
        ok4 = len(g.tp_groups) == dp * pp   # one TP group per (dp_r, pp_r) pair
        ok5 = len(g.pp_groups) == dp * tp   # one PP group per (dp_r, tp_r) pair
        ok6 = len(g.dp_groups) == pp * tp   # one DP group per (pp_r, tp_r) pair
        all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6])
        print(f"  DP={dp:2d} TP={tp:2d} PP={pp:2d}: {N:>5} GPUs  "
              f"TP groups={len(g.tp_groups):>4}  PP groups={len(g.pp_groups):>4}  "
              f"DP groups={len(g.dp_groups):>4}  {'✓' if all_ok else '✗'}")

    # ── Example 3: Large cluster (512 GPUs) ──────────────────────────────────
    print()
    print("=" * 65)
    print("  EXAMPLE 3: GPT-3 style — DP=8, TP=8, PP=8 (512 GPUs)")
    print("=" * 65)
    g2 = build_process_groups(dp=8, tp=8, pp=8)
    print(f"  Total GPUs:    {g2.total_gpus}")
    print(f"  TP groups:     {len(g2.tp_groups)} (each with {g2.tp} ranks)")
    print(f"  PP groups:     {len(g2.pp_groups)} (each with {g2.pp} ranks)")
    print(f"  DP groups:     {len(g2.dp_groups)} (each with {g2.dp} ranks)")
    print(f"  First TP group: {g2.tp_groups[0]}")
    print(f"  First PP group: {sorted(g2.pp_groups[0], key=lambda r: g2.rank_to_pp_rank[r])}")
    print(f"  First DP group: {g2.dp_groups[0]}")
''',
    },

    "Optimal 3D Configuration Finder": {
        "description": "Given a model size, GPU count, and hardware specs, compute all valid 3D configurations and rank them by efficiency — memory, MFU, and communication overhead.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
OPTIMAL 3D PARALLELISM CONFIGURATION FINDER
================================================================================

For a given model + cluster, enumerate all valid DP×TP×PP configurations and
rank them by:
    1. Memory feasibility (does the model fit per GPU?)
    2. Estimated MFU (accounting for PP bubble and TP communication)
    3. Communication balance (DP all-reduce volume)

================================================================================
"""

import math
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name:         str
    n_params:     int
    n_layers:     int
    d_model:      int
    n_heads:      int
    n_kv_heads:   int
    d_ff:         int
    dtype_bytes:  int = 2    # bf16 default


@dataclass
class HardwareSpec:
    name:            str
    n_gpus:          int
    gpus_per_node:   int
    gpu_mem_gb:      float
    nvlink_bw_gbps:  float   # intra-node
    ib_bw_gbps:      float   # inter-node
    peak_tflops:     float   # per GPU, bf16


def estimate_weight_mem_gb(model: ModelSpec, tp: int, pp: int) -> float:
    """Weight memory per GPU in GB (bf16)."""
    return model.n_params * model.dtype_bytes / (tp * pp) / 1e9


def estimate_activation_mem_gb(model: ModelSpec, tp: int, pp: int,
                                 seq_len: int, micro_batch: int,
                                 use_ckpt: bool = True) -> float:
    """Activation memory per GPU (rough estimate)."""
    layers_per_stage = model.n_layers / pp
    d, T, B = model.d_model, seq_len, micro_batch

    if use_ckpt:
        # With checkpointing: O(1 activation per layer)
        act_per_layer = B * T * d * model.dtype_bytes / 1e9
    else:
        # Without: O(T² attention + FFN activations)
        h = model.n_heads // tp
        act_per_layer = (B * h * T * T * model.dtype_bytes     # attention T²
                         + B * T * model.d_ff // tp * model.dtype_bytes) / 1e9

    return act_per_layer * layers_per_stage


def estimate_bubble_pct(pp: int, m: int) -> float:
    """Pipeline bubble fraction."""
    if pp == 1:
        return 0.0
    return (pp - 1) / (m + pp - 1) * 100


def estimate_tp_overhead_pct(tp: int, hw: HardwareSpec,
                               model: ModelSpec, seq_len: int,
                               micro_batch: int) -> float:
    """TP communication overhead (fraction of compute time spent on TP comms)."""
    if tp == 1:
        return 0.0
    # 2 all-reduces per layer, each B×T×d in volume
    tp_comm_vol_gb = 2 * model.n_layers * micro_batch * seq_len * model.d_model * model.dtype_bytes / 1e9
    # Ring all-reduce: 2(tp-1)/tp × vol per step
    tp_comm_vol_gb *= 2 * (tp - 1) / tp

    # Intra-node bandwidth (NVLink)
    tp_comm_time_s = tp_comm_vol_gb / hw.nvlink_bw_gbps

    # Compute time (rough estimate: 6P FLOPs / (total_tflops × mfu))
    mfu             = 0.45  # assumed
    compute_flops   = 6 * model.n_params * micro_batch * seq_len
    compute_time_s  = compute_flops / (hw.n_gpus * hw.peak_tflops * 1e12 * mfu)

    return min(tp_comm_time_s / compute_time_s * 100, 50.0)


def find_valid_configs(model: ModelSpec, hw: HardwareSpec,
                        seq_len: int = 2048, micro_batch: int = 1,
                        max_m: int = 32) -> list[dict]:
    """
    Enumerate all valid DP×TP×PP configurations for this model+cluster.
    Returns list of config dicts sorted by estimated efficiency.
    """
    N = hw.n_gpus
    configs = []

    for tp in [1, 2, 4, 8]:
        if tp > hw.gpus_per_node:
            break
        if model.n_heads % tp != 0 or model.n_kv_heads % tp != 0:
            continue

        for pp in range(1, N // tp + 1):
            if (N % (tp * pp)) != 0:
                continue
            dp = N // (tp * pp)

            # Memory check
            weight_gb = estimate_weight_mem_gb(model, tp, pp)
            act_gb    = estimate_activation_mem_gb(model, tp, pp, seq_len, micro_batch)
            optim_gb  = model.n_params * 8 / (tp * pp * dp) / 1e9  # ZeRO Stage 1

            total_gb  = weight_gb + act_gb + optim_gb
            if total_gb > hw.gpu_mem_gb * 0.95:   # 5% reserve
                continue

            # Find optimal m (micro-batches per PP pass)
            m = max(1, min(max_m, 4 * pp))    # target: m ≈ 4×PP

            # Efficiency estimates
            bubble    = estimate_bubble_pct(pp, m)
            tp_over   = estimate_tp_overhead_pct(tp, hw, model, seq_len, micro_batch)
            effective_util = (1 - bubble / 100) * (1 - tp_over / 100) * 100

            # Global batch size
            global_batch = micro_batch * seq_len * dp * m
            global_batch_M = global_batch / 1e6

            configs.append({
                "dp": dp, "tp": tp, "pp": pp, "m": m,
                "weight_gb":  weight_gb,
                "act_gb":     act_gb,
                "optim_gb":   optim_gb,
                "total_gb":   total_gb,
                "bubble_pct": bubble,
                "tp_over_pct": tp_over,
                "eff_pct":    effective_util,
                "global_M":   global_batch_M,
            })

    return sorted(configs, key=lambda c: -c["eff_pct"])


def fmt_gb(gb: float) -> str:
    return f"{gb:.1f}GB"


if __name__ == "__main__":
    models = [
        ModelSpec("LLaMA-2 7B",  7_000_000_000, 32, 4096, 32, 32, 11008),
        ModelSpec("LLaMA-2 70B", 70_000_000_000, 80, 8192, 64, 8, 28672),
        ModelSpec("GPT-3 175B",  175_000_000_000, 96, 12288, 96, 96, 49152),
    ]

    clusters = [
        HardwareSpec("8×A100 80GB  (1 node)",   8,  8, 80.0, 600.0, 0.0,  312.0),
        HardwareSpec("64×A100 80GB (8 nodes)",  64, 8, 80.0, 600.0, 25.0, 312.0),
        HardwareSpec("512×A100 80GB",           512, 8, 80.0, 600.0, 25.0, 312.0),
    ]

    for model in models:
        for hw in clusters:
            configs = find_valid_configs(model, hw, seq_len=2048, micro_batch=2)
            if not configs:
                continue
            print(f"\n  {model.name} on {hw.name}")
            print(f"  {'DP':>4}  {'TP':>4}  {'PP':>4}  {'m':>4}  "
                  f"{'Mem/GPU':>8}  {'Bubble':>8}  {'TP ovhd':>9}  "
                  f"{'Eff%':>8}  {'Batch(M)':>10}")
            print(f"  {'':─>4}  {'':─>4}  {'':─>4}  {'':─>4}  "
                  f"{'':─>8}  {'':─>8}  {'':─>9}  {'':─>8}  {'':─>10}")
            for c in configs[:5]:  # top 5
                print(f"  {c['dp']:>4}  {c['tp']:>4}  {c['pp']:>4}  {c['m']:>4}  "
                      f"{fmt_gb(c['total_gb']):>8}  {c['bubble_pct']:>7.1f}%  "
                      f"{c['tp_over_pct']:>8.1f}%  {c['eff_pct']:>7.1f}%  "
                      f"{c['global_M']:>9.1f}M")
''',
    },

    "Memory Budget and Communication Volume Analysis": {
        "description": "Detailed breakdown of memory and communication costs for 3D parallelism at different scales, showing how each axis helps and their interaction.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
3D PARALLELISM: MEMORY AND COMMUNICATION ANALYSIS
================================================================================

Quantifies:
    1. Per-GPU memory at different DP×TP×PP configurations
    2. Communication volume for each type (TP, PP, DP)
    3. How the three axes compose to enable any model scale
    4. Comparison with single-axis approaches

================================================================================
"""

import math


def fmt_gb(gb: float) -> str:
    if gb >= 1000: return f"{gb/1024:.1f} TB"
    if gb >= 1:   return f"{gb:.1f} GB"
    return f"{gb*1024:.0f} MB"


def analyze_3d_config(n_params: int, dp: int, tp: int, pp: int,
                       d: int, T: int, B: int, n_layers: int,
                       dtype_bytes: int = 2) -> dict:
    """Full memory and communication analysis for a 3D config."""
    N = dp * tp * pp

    # Memory
    weight_per_gpu_gb = n_params * dtype_bytes / (tp * pp) / 1e9
    grad_per_gpu_gb   = weight_per_gpu_gb   # same as weights
    opt_per_gpu_gb    = n_params * 4 * 2 / (tp * pp * dp) / 1e9  # ZeRO-1: m+v in fp32 / dp

    layers_pp = n_layers / pp
    act_per_gpu_gb = B * T * d * n_layers / pp / tp * dtype_bytes / 1e9  # rough, with ckpt

    total_mem_gb = weight_per_gpu_gb + grad_per_gpu_gb + opt_per_gpu_gb + act_per_gpu_gb

    # Communication volumes per step
    # TP: 2 all-reduces per layer × n_layers × B×T×d per rank (ring volume = 2(tp-1)/tp)
    tp_vol_gb = (2 * n_layers * B * T * d * dtype_bytes / 1e9
                 * 2 * (tp - 1) / max(tp, 1))

    # PP: 2 (fwd+bwd) × (PP-1) stage boundaries × B×T×d
    pp_vol_gb = 2 * (pp - 1) * B * T * d * dtype_bytes / 1e9

    # DP: gradient all-reduce = 2 × params/TP/PP × dtype
    dp_vol_gb = 2 * n_params * dtype_bytes / (tp * pp) / 1e9

    # m (micro-batches): target 4×PP for ~20% bubble
    m         = max(1, 4 * pp)
    bubble    = (pp - 1) / (m + pp - 1) * 100 if pp > 1 else 0.0

    return {
        "n_gpus":          N,
        "dp": dp, "tp": tp, "pp": pp,
        "weight_gb":       weight_per_gpu_gb,
        "grad_gb":         grad_per_gpu_gb,
        "optim_gb":        opt_per_gpu_gb,
        "act_gb":          act_per_gpu_gb,
        "total_mem_gb":    total_mem_gb,
        "tp_comm_gb":      tp_vol_gb,
        "pp_comm_gb":      pp_vol_gb,
        "dp_comm_gb":      dp_vol_gb,
        "bubble_pct":      bubble,
        "m":               m,
    }


if __name__ == "__main__":
    # LLaMA-2 70B configuration
    N_PARAMS = 70_000_000_000
    N_LAYERS = 80
    D        = 8192
    T        = 2048
    B        = 1

    print("=" * 78)
    print("  LLAMA-2 70B MEMORY BREAKDOWN BY 3D CONFIGURATION")
    print("  (per GPU, ZeRO Stage 1, activation checkpointing, bf16 weights)")
    print("=" * 78)
    print()
    print(f"  {'DP':>4}  {'TP':>4}  {'PP':>4}  {'GPUs':>6}  "
          f"{'Weights':>9}  {'Grad':>7}  {'Optim':>7}  {'Act':>7}  "
          f"{'TOTAL':>8}  {'A100 80GB?':>12}")
    print(f"  {'':─>4}  {'':─>4}  {'':─>4}  {'':─>6}  "
          f"{'':─>9}  {'':─>7}  {'':─>7}  {'':─>7}  "
          f"{'':─>8}  {'':─>12}")

    configs = [
        (1, 1, 1),   # no parallelism
        (1, 2, 1),   # TP only
        (1, 4, 1),   # TP only
        (1, 8, 1),   # TP only
        (1, 8, 2),   # TP + PP
        (2, 8, 2),   # TP + PP + DP
        (4, 8, 2),
        (8, 8, 2),
        (8, 8, 4),
        (16, 8, 4),
    ]

    for dp, tp, pp in configs:
        c = analyze_3d_config(N_PARAMS, dp, tp, pp, D, T, B, N_LAYERS)
        fits = "✓ fits" if c["total_mem_gb"] <= 80 else "✗ OOM"
        print(f"  {dp:>4}  {tp:>4}  {pp:>4}  {c['n_gpus']:>6}  "
              f"{fmt_gb(c['weight_gb']):>9}  {fmt_gb(c['grad_gb']):>7}  "
              f"{fmt_gb(c['optim_gb']):>7}  {fmt_gb(c['act_gb']):>7}  "
              f"{fmt_gb(c['total_mem_gb']):>8}  {fits:>12}")

    # Communication analysis
    print()
    print("=" * 78)
    print("  COMMUNICATION VOLUME PER TRAINING STEP")
    print("  (LLaMA-2 70B, representative configurations)")
    print("=" * 78)
    print()
    print(f"  {'Config':>16}  {'TP comm':>10}  {'PP comm':>10}  "
          f"{'DP comm':>10}  {'Total':>10}  {'Bottleneck':>12}")
    print(f"  {'':─>16}  {'':─>10}  {'':─>10}  "
          f"{'':─>10}  {'':─>10}  {'':─>12}")

    for dp, tp, pp in [(1,8,1), (1,8,4), (4,8,4), (16,8,4)]:
        c = analyze_3d_config(N_PARAMS, dp, tp, pp, D, T, B, N_LAYERS)
        total = c["tp_comm_gb"] + c["pp_comm_gb"] + c["dp_comm_gb"]
        if total == 0:
            bottleneck = "compute"
        else:
            bns = max(["tp", "pp", "dp"],
                       key=lambda x: c[f"{x}_comm_gb"])
            bottleneck = f"{bns.upper()} ({c[f'{bns}_comm_gb']/total:.0%})"
        lbl = f"DP{dp}×TP{tp}×PP{pp}"
        print(f"  {lbl:>16}  {fmt_gb(c['tp_comm_gb']):>10}  "
              f"{fmt_gb(c['pp_comm_gb']):>10}  "
              f"{fmt_gb(c['dp_comm_gb']):>10}  "
              f"{fmt_gb(total):>10}  {bottleneck:>12}")

    # Scaling trajectory
    print()
    print("=" * 78)
    print("  SCALING TRAJECTORY: HOW TO SCALE TO 1024 GPUS")
    print("  LLaMA-2 70B, starting from 8 GPUs")
    print("=" * 78)
    print()
    trajectory = [
        (8,   8, 1, 1, "Baseline: TP only (8 GPUs, TP=8)"),
        (16,  8, 2, 1, "Add PP=2 (16 GPUs)"),
        (32,  8, 4, 1, "Add PP=4 (32 GPUs)"),
        (64,  8, 4, 2, "Add DP=2 (64 GPUs)"),
        (128, 8, 4, 4, "Add DP=4 (128 GPUs)"),
        (256, 8, 4, 8, "Add DP=8 (256 GPUs)"),
        (512, 8, 4, 16, "Add DP=16 (512 GPUs)"),
        (1024, 8, 4, 32, "Add DP=32 (1024 GPUs)"),
    ]

    for total, tp, pp, dp, desc in trajectory:
        c   = analyze_3d_config(N_PARAMS, dp, tp, pp, D, T, B, N_LAYERS)
        mem = c["total_mem_gb"]
        fits = "✓" if mem <= 80 else "✗"
        tokens_step = B * T * dp * max(1, 4 * pp)
        print(f"  {total:>5} GPUs  DP={dp:>2} TP={tp} PP={pp}  "
              f"mem={fmt_gb(mem):>8}  {fits}  "
              f"~{tokens_step/1000:.0f}K tok/step  {desc}")
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
    #     from llm_training.visuals.parallelism_3d import (
    #         THREED_VISUAL_HTML,
    #         THREED_VISUAL_HEIGHT,
    #     )
    #     visual_html   = THREED_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = THREED_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[23_3D_parallelism_megatron_deepspeed.py] Could not load visual: {e}",
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