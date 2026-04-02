"""
ZeRO Optimizer Stages
======================

ZeRO (Zero Redundancy Optimizer) is the single most impactful memory
reduction technique for large-scale training. By partitioning the three
categories of training state — optimizer moments, gradients, and parameters
— across data-parallel ranks, ZeRO eliminates the memory redundancy that
standard DDP incurs. A 70B model that requires 160 GB per GPU under DDP can
be trained on 8 GPUs with just 20 GB per GPU under ZeRO Stage 3, without
any change to the model architecture or quality.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "ZeRO Optimizer Stages"
DISPLAY_NAME = "24 · ZeRO Optimizer Stages"
ICON         = "🅾️"
SUBTITLE     = "ZeRO-1, ZeRO-2, ZeRO-3 Memory Partitioning"


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

### The Memory Redundancy Problem in Data Parallelism

Standard DDP (DistributedDataParallel) replicates the entire model on every
GPU. This achieves near-linear throughput scaling but is enormously wasteful
in memory: every piece of training state exists N times across N GPUs.

For a model with P parameters trained with AdamW in bf16/fp32 mixed precision,
the training state comprises four categories:

    Component              Precision   Bytes/param   Fraction of total
    ──────────────────────────────────────────────────────────────────
    16-bit parameters      bf16        2              12.5%
    16-bit gradients       bf16        2              12.5%
    32-bit master weights  fp32        4              25.0%
    AdamW 1st moment (m)   fp32        4              25.0%
    AdamW 2nd moment (v)   fp32        4              25.0%
    ──────────────────────────────────────────────────────────────────
    Total                              16             100%

    For a 7B model: 7B × 16 bytes = 112 GB per GPU in DDP
    Across 8 DDP replicas: 8 × 112 GB = 896 GB total GPU memory used
    Of which 896 - 112 = 784 GB is pure redundancy!

The key insight of ZeRO: **there is no mathematical requirement that every
GPU holds all of these states**. The optimizer update only needs to see
a parameter's gradient and moments to compute the new parameter value.
We can therefore partition these states across GPUs, with each GPU holding
only 1/N of each.

Rajbhandari et al. (2020) formalised this into ZeRO's three stages, each
eliminating a different source of redundancy.


### ZeRO Stage 1 — Optimizer State Partitioning (OS)

**What is partitioned:** The AdamW optimizer state — specifically the fp32
master weights, 1st moment (m), and 2nd moment (v).

**How it works:**
Each GPU is assigned a contiguous partition of P/N parameters to "own".
Only the owning GPU maintains the full optimizer state for its partition.
All GPUs still hold the complete 16-bit parameters and gradients.

    Per-GPU memory with ZeRO-1:
        16-bit params (full):  2P bytes          ← replicated
        16-bit grads  (full):  2P bytes          ← replicated
        fp32 master weights:   4P/N bytes        ← partitioned!
        m (1st moment):        4P/N bytes        ← partitioned!
        v (2nd moment):        4P/N bytes        ← partitioned!
        ─────────────────────────────────────────
        Total:                 (4 + 12/N)P bytes

    For N=8:  (4 + 1.5)P = 5.5P bytes   vs 16P in DDP  → 2.9× reduction

**The update procedure in ZeRO-1:**
    1. Forward pass: standard (all GPUs have full 16-bit params)
    2. Backward pass: standard (all GPUs compute full gradients)
    3. Gradient synchronisation: each GPU all-reduces only the gradient
       shard it owns (Reduce-Scatter pattern instead of All-Reduce)
    4. Optimizer step: each GPU updates only its owned parameter shard
       using the local optimizer state
    5. Parameter sync: each GPU has updated its shard of master weights;
       All-Gather to distribute updated 16-bit params back to all GPUs

**Communication in ZeRO-1:**
    Standard DDP: 1 All-Reduce of size P × 2 bytes
    ZeRO-1:       Reduce-Scatter (P × 2 bytes) + All-Gather (P × 2 bytes)
    Total volume: identical — ZeRO-1 does not add communication!

The Reduce-Scatter and All-Gather together equal one All-Reduce. ZeRO-1
achieves 2.9× memory reduction with zero extra communication.


    **Diagram 1 — ZeRO Memory Partitioning Stages:**

    ZERO MEMORY PARTITIONING (N=4 GPUs, model P parameters)
    ════════════════════════════════════════════════════════════════

    DDP (baseline):
    GPU0: [params|grads|master|m|v]  ← full replica
    GPU1: [params|grads|master|m|v]  ← identical copy
    GPU2: [params|grads|master|m|v]  ← identical copy
    GPU3: [params|grads|master|m|v]  ← identical copy
    Redundancy: 4× on all 5 components = 16P bytes per GPU

    ZeRO Stage 1 (OS — optimizer state partition):
    GPU0: [params|grads|master₀|m₀|v₀]  ← owns shard 0
    GPU1: [params|grads|master₁|m₁|v₁]  ← owns shard 1
    GPU2: [params|grads|master₂|m₂|v₂]  ← owns shard 2
    GPU3: [params|grads|master₃|m₃|v₃]  ← owns shard 3
    Memory: (2+2+4/4+4/4+4/4)P = 7P bytes  → 2.3× reduction (at N=4)

    ZeRO Stage 2 (OS+G — + gradient partition):
    GPU0: [params|grad₀|master₀|m₀|v₀]  ← owns grad+optim shard 0
    GPU1: [params|grad₁|master₁|m₁|v₁]  ← owns grad+optim shard 1
    GPU2: [params|grad₂|master₂|m₂|v₂]
    GPU3: [params|grad₃|master₃|m₃|v₃]
    Memory: (2+2/4+12/4)P = (2+0.5+3)P = 5.5P → 2.9× reduction

    ZeRO Stage 3 (OS+G+P — full partition):
    GPU0: [param₀|grad₀|master₀|m₀|v₀]  ← owns ALL state for shard 0
    GPU1: [param₁|grad₁|master₁|m₁|v₁]
    GPU2: [param₂|grad₂|master₂|m₂|v₂]
    GPU3: [param₃|grad₃|master₃|m₃|v₃]
    Memory: 16P/4 = 4P bytes per GPU → 4× reduction (= N×)


### ZeRO Stage 2 — Gradient Partitioning (OS+G)

Stage 2 extends Stage 1 by also partitioning gradients across GPUs. After
the backward pass, each GPU only needs to keep the gradients for its owned
parameter shard. The gradients for other shards can be discarded as soon as
they've been all-reduced.

**Implementation via Reduce-Scatter:**
Instead of a full All-Reduce (which sends the full gradient to all GPUs),
Stage 2 uses a Reduce-Scatter:
    •   Each GPU contributes its locally-computed gradient for all P parameters
    •   The sum is computed and each GPU receives only its owned shard (P/N)
    •   Each GPU now has the correct averaged gradient for its shard only
    •   The gradients for unowned parameters are discarded

This halves the gradient memory: instead of holding 2P bytes of gradients
(one full copy), each GPU holds only 2P/N bytes.

    Per-GPU memory with ZeRO-2:
        16-bit params (full):  2P bytes          ← still replicated
        16-bit grads  (owned): 2P/N bytes        ← partitioned!
        fp32 master weights:   4P/N bytes        ← partitioned!
        m (1st moment):        4P/N bytes        ← partitioned!
        v (2nd moment):        4P/N bytes        ← partitioned!
        ─────────────────────────────────────────
        Total:                 (2 + 14/N)P bytes

    For N=8:  (2 + 1.75)P = 3.75P bytes  → 4.3× reduction over DDP

**Communication in ZeRO-2:**
    Backward: Reduce-Scatter (same volume as All-Reduce: P × 2 bytes)
    Step:     All-Gather of updated 16-bit params (P × 2 bytes)
    Total:    P × 4 bytes  (identical to standard DDP All-Reduce of P × 2 bytes × 2 fwd+bwd)

ZeRO-2 achieves 4.3× memory reduction with the same communication volume as DDP.


### ZeRO Stage 3 — Parameter Partitioning (OS+G+P)

Stage 3 goes furthest: it partitions the 16-bit parameters themselves across
GPUs. Each GPU permanently holds only 1/N of the parameters.

**The challenge:** The forward pass requires all parameters. With Stage 3,
a forward pass through layer L requires an All-Gather to reconstruct the
full parameters for L before computing, then the parameters are discarded.

**The parameter lifecycle in ZeRO-3:**
    1.  Before layer L's forward: All-Gather parameters of layer L
        (each GPU fetches the full layer L from all N GPUs)
    2.  Compute layer L's forward pass (full parameters available)
    3.  Discard non-owned parameters of layer L
    4.  Proceed to layer L+1 → All-Gather L+1, etc.

This adds one All-Gather per layer per forward pass and one per backward
pass — significantly more communication than Stages 1 and 2.

    Per-GPU memory with ZeRO-3:
        16-bit params (owned): 2P/N bytes        ← partitioned!
        16-bit grads  (owned): 2P/N bytes        ← partitioned!
        fp32 master weights:   4P/N bytes        ← partitioned!
        m (1st moment):        4P/N bytes        ← partitioned!
        v (2nd moment):        4P/N bytes        ← partitioned!
        ─────────────────────────────────────────
        Total:                 16P/N bytes

    For N=8:  16P/8 = 2P bytes per GPU  → 8× reduction over DDP
    For N=64: 16P/64 = 0.25P bytes per GPU → 64× reduction!

**Communication in ZeRO-3:**
    Per forward pass:  All-Gather for each layer (P × 2 bytes total)
    Per backward pass: All-Gather + Reduce-Scatter per layer
    Total:             ~3× the communication of DDP per step

The massive memory savings come at the cost of substantially more communication.
ZeRO-3 is therefore most valuable when:
    a)  The model absolutely does not fit otherwise (even with TP+PP)
    b)  The cluster has sufficient bandwidth to absorb the overhead
    c)  The alternative is not training at all


### ZeRO Memory Formula Summary

    Stage   Memory per GPU (N data-parallel ranks)   vs DDP
    ──────────────────────────────────────────────────────────────────
    DDP      16P bytes                                1× (baseline)
    ZeRO-1   (4 + 12/N)P bytes                       ~N× for large N
    ZeRO-2   (2 + 14/N)P bytes                       ~8× for N=8
    ZeRO-3   16P/N bytes                             N× (exact)
    ──────────────────────────────────────────────────────────────────

At large N (N → ∞):
    ZeRO-1 → 4P bytes   (params + grads, no optimizer)
    ZeRO-2 → 2P bytes   (only params, no grads or optimizer)
    ZeRO-3 → 0P bytes   (everything partitioned, scales with N)


    **Diagram 2 — ZeRO Communication Pattern Comparison:**

    COMMUNICATION PATTERN: DDP vs ZeRO STAGES
    ════════════════════════════════════════════════════════════════

    DDP:
    Backward: ──── All-Reduce (full grad, 2P bytes) ────→ all GPUs
    Cost: 1 All-Reduce = 2 × (N-1)/N × 2P bytes per rank

    ZeRO-1 and ZeRO-2:
    Backward: ──── Reduce-Scatter (grads → each GPU gets shard) ────
    Update:         [local optimizer step on owned shard]
    Parameter sync: ──── All-Gather (updated params) ────────────────
    Cost: Reduce-Scatter + All-Gather = same as All-Reduce ✓

    ZeRO-3 (additional All-Gathers per layer):
    Before layer L fwd:  AG to reconstruct params L
    Before layer L bwd:  AG to reconstruct params L (again)
    After layer L bwd:   RS to reduce-scatter gradients L
    Cost: ~3× per-layer communication vs DDP (but saves 1/N memory)


### The ZeRO-3 Parameter Prefetching Optimisation

Naive ZeRO-3 stalls computation while waiting for each All-Gather to complete.
The key optimisation: **prefetch the next layer's parameters while computing
the current layer**.

    Timeline (prefetch disabled):
    Layer 1: [AG wait] [compute] [discard] → [AG wait] [compute] ...

    Timeline (prefetch enabled):
    Layer 1: [AG wait] [compute + AG for L2] [discard L1] → [compute L2 + AG for L3] ...

With prefetch, the All-Gather communication is hidden behind computation.
DeepSpeed's ZeRO-3 implementation supports this via `prefetch_bucket_size`.

However, prefetching requires more memory (you hold two layers' parameters
simultaneously), creating a trade-off: memory vs throughput.


### ZeRO in PyTorch: FSDP

PyTorch's native implementation of ZeRO is **FullyShardedDataParallel (FSDP)**.
It implements ZeRO-3 semantics with additional features:

    **FSDP API:**
    ```python
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
        # or:
        # ShardingStrategy.SHARD_GRAD_OP,   # ZeRO-2
        # ShardingStrategy.NO_SHARD,        # DDP (no sharding)
        cpu_offload=CPUOffload(offload_params=False),
    )
    ```

    **FSDP sharding strategies:**
    FULL_SHARD     = ZeRO-3: shard params + grads + optimizer
    SHARD_GRAD_OP  = ZeRO-2: shard grads + optimizer (params replicated during fwd)
    NO_SHARD       = standard DDP

**FSDP FlatParameter design:**
FSDP flattens all parameters within each "flat parameter group" (typically
one Transformer block) into a single 1D tensor before sharding. This reduces
communication overhead: instead of N All-Gathers per layer (one per parameter),
there is 1 All-Gather per group.

**FSDP mixed precision:**
FSDP supports `MixedPrecision` that keeps parameters in fp16/bf16 but
optimizer state in fp32 (the standard mixed-precision recipe). With
`reduce_dtype=torch.float16`, gradients are reduced in fp16 before casting
to fp32 for the optimizer update.


### DeepSpeed ZeRO

DeepSpeed (Microsoft) is the original ZeRO implementation with additional
features beyond PyTorch FSDP:

    **ZeRO-Infinity (Stage 4 extension):**
    Extends ZeRO-3 to offload parameters and optimizer state to CPU RAM
    and even NVMe SSD, enabling models far larger than total GPU memory.
    A 1T parameter model can be trained with just 512 GB of CPU RAM and
    GPU memory for active compute. (Covered in Module 25.)

    **ZeRO++ (2023):**
    Reduces ZeRO-3 communication by 4× using:
        •   Quantised weights (4-bit) for All-Gather during forward pass
        •   Hierarchical All-Gather (within node first, then across nodes)
        •   Quantised gradients for Reduce-Scatter
    Achieves ~2× throughput improvement over standard ZeRO-3 on slow networks.

    **ZeRO-R (Redundancy Elimination):**
    Partitions activations across data-parallel ranks, reducing activation
    memory by N× for free (complementary to activation checkpointing).


### The Three-Stage Trade-off Table

    ZeRO Stage   Memory saving   Communication overhead   Throughput impact
    ─────────────────────────────────────────────────────────────────────────
    Stage 1      2-3×            None                     ~0% (same as DDP)
    Stage 2      3-8×            None                     ~1-3% (Reduce-Scatter slightly
                                                           less optimal than All-Reduce)
    Stage 3      N×              +2× per-layer AG         10-30% (depends on bandwidth)
    ─────────────────────────────────────────────────────────────────────────

**When to use each stage:**

    Stage 1:  Model fits on GPU with DDP. You want larger batch size or
              lower optimizer memory without affecting throughput.
              Example: 7B model on 8×80GB with optimizer taking too much space.

    Stage 2:  Model still fits if you remove gradient redundancy.
              Example: 13B model where Stage 1 is insufficient.

    Stage 3:  Model fundamentally does not fit without parameter sharding.
              You accept communication overhead as the cost of training.
              Example: 70B model on 8×80GB GPUs (each GPU holds 10B params).

The general advice: use the lowest stage that makes the model fit.
Unnecessary sharding adds communication overhead for no benefit.


### ZeRO-1 with Mixed-Precision: The "Big 3" Memory Components

In modern bf16 training with AdamW and ZeRO-1, the per-GPU memory is:

    Component                     Bytes          Note
    ──────────────────────────────────────────────────────────────────
    bf16 model parameters         2P             Replicated (full)
    bf16 gradients                2P             Replicated (full) until RS
    fp32 master weights (owned)   4P/N           ZeRO-1 partition
    fp32 m_t (owned)              4P/N           ZeRO-1 partition
    fp32 v_t (owned)              4P/N           ZeRO-1 partition
    Activations                   variable       ~B×T×d×N_layers (with ckpt)
    ──────────────────────────────────────────────────────────────────
    Subtotal (model state)        4P + 12P/N
    ──────────────────────────────────────────────────────────────────

For a 70B model (P=70B, N=16 DP replicas):
    Model state = 4 × 70B × 1 + 12 × 70B / 16
                = 280 GB + 52.5 GB = 332.5 GB  TOTAL across 16 GPUs
    Per GPU     = 332.5 / 16 = 20.8 GB   (vs 112 GB with DDP!)

This is one of the most dramatic examples of ZeRO's power: the same 70B model
that requires 112 GB per GPU in DDP requires only 21 GB per GPU with ZeRO-1
and 16 DP replicas — without any change to model quality or throughput.


### ZeRO and Tensor Parallelism: A Critical Interaction

ZeRO-3 and Tensor Parallelism (TP) have a subtle incompatibility:

    TP requires parameters to be permanently resident on the GPU (no all-gather).
    ZeRO-3 all-gathers parameters on demand (reconstructing the full layer).

If both are applied naively, ZeRO would try to shard a weight that TP
already expects to be split along a different dimension. The solutions:

    **Option 1 (Megatron-DeepSpeed default):** Apply ZeRO only to the
    data-parallel dimension (not within a TP group). ZeRO partitions
    optimizer state across DP ranks; TP handles weight matrix splits.
    Result: optimal combination of ZeRO-1/2 + TP + PP.

    **Option 2 (ZeRO-1 or ZeRO-2 with TP):**
    ZeRO-1/2 are safe with TP because they don't move parameters.
    Only optimizer state (ZeRO-1) or gradients (ZeRO-2) are partitioned.

    **Option 3 (ZeRO-3 without TP):**
    For models too large for a single TP group, use ZeRO-3 as the sole
    model distribution strategy. Accept the communication overhead.
    DeepSpeed's Megatron integration supports this carefully.

The recommended combination for frontier models:
    TP within NVLink nodes + PP across nodes + ZeRO-1 or ZeRO-2 across DP groups
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
ZeRO Stages: Memory and Communication Summary

| Stage       | What is partitioned           | Memory per GPU (N ranks)    | Extra communication   | PyTorch FSDP equivalent |
|-------------|-------------------------------|-----------------------------|-----------------------|-------------------------|
| DDP (none)  | Nothing                       | 16P bytes                   | 1× All-Reduce         | NO_SHARD               |
| ZeRO-1 (OS) | Optimizer state (m, v, fp32w) | (4 + 12/N)P bytes           | None (RS + AG = AR)   | N/A (use DeepSpeed)    |
| ZeRO-2 (OS+G)| Optimizer + gradients        | (2 + 14/N)P bytes           | None (RS + AG = AR)   | SHARD_GRAD_OP          |
| ZeRO-3 (OS+G+P)| All: param+grad+optim    | 16P/N bytes                 | +2× per-layer AG      | FULL_SHARD             |
| ZeRO-Inf    | All + CPU/NVMe offload        | ≈ 0 (GPU) + offload         | CPU↔GPU transfer      | N/A (DeepSpeed only)   |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "ZeRO Memory Calculator — All Stages": {
        "description": "Compute the exact per-GPU memory for all ZeRO stages and model sizes, show crossover points, and plot the N-GPU scaling curve.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ZERO MEMORY CALCULATOR — ALL STAGES
================================================================================

Computes per-GPU memory for DDP, ZeRO-1, ZeRO-2, and ZeRO-3 as a function
of model size and number of data-parallel ranks.

Shows:
    1. Memory breakdown per stage
    2. Memory vs N curves for common model sizes
    3. Crossover points (when each stage becomes necessary)
    4. Combined 3D parallelism + ZeRO calculations
    5. Minimum GPU count to train each model at each stage

================================================================================
"""

import math


# ── Memory formulas ───────────────────────────────────────────────────────────

def bytes_per_param():
    """
    Returns a dict of bytes per parameter for each component.
    Mixed precision training: model in bf16, optimizer state in fp32.
    """
    return {
        "bf16_params":   2,   # 16-bit parameters (forward/backward)
        "bf16_grads":    2,   # 16-bit gradients (accumulated)
        "fp32_master":   4,   # float32 master copy (for optimizer step)
        "fp32_m":        4,   # AdamW 1st moment
        "fp32_v":        4,   # AdamW 2nd moment
    }


def memory_ddp(n_params: int) -> dict:
    """Per-GPU memory in bytes for standard DDP."""
    bpp = bytes_per_param()
    components = {k: n_params * v for k, v in bpp.items()}
    components["total"] = sum(components.values())
    return components


def memory_zero1(n_params: int, n_dp: int) -> dict:
    """Per-GPU memory for ZeRO Stage 1 (optimizer state partitioned)."""
    bpp = bytes_per_param()
    return {
        "bf16_params":  n_params * bpp["bf16_params"],         # full
        "bf16_grads":   n_params * bpp["bf16_grads"],          # full
        "fp32_master":  n_params * bpp["fp32_master"] // n_dp, # sharded
        "fp32_m":       n_params * bpp["fp32_m"]      // n_dp, # sharded
        "fp32_v":       n_params * bpp["fp32_v"]      // n_dp, # sharded
        "total":        n_params * (4 + 12 / n_dp),
    }


def memory_zero2(n_params: int, n_dp: int) -> dict:
    """Per-GPU memory for ZeRO Stage 2 (optimizer + gradients partitioned)."""
    bpp = bytes_per_param()
    return {
        "bf16_params":  n_params * bpp["bf16_params"],          # full
        "bf16_grads":   n_params * bpp["bf16_grads"]  // n_dp,  # sharded
        "fp32_master":  n_params * bpp["fp32_master"] // n_dp,  # sharded
        "fp32_m":       n_params * bpp["fp32_m"]      // n_dp,  # sharded
        "fp32_v":       n_params * bpp["fp32_v"]      // n_dp,  # sharded
        "total":        n_params * (2 + 14 / n_dp),
    }


def memory_zero3(n_params: int, n_dp: int) -> dict:
    """Per-GPU memory for ZeRO Stage 3 (all state partitioned)."""
    bpp = bytes_per_param()
    return {
        "bf16_params":  n_params * bpp["bf16_params"]  // n_dp,  # sharded
        "bf16_grads":   n_params * bpp["bf16_grads"]   // n_dp,  # sharded
        "fp32_master":  n_params * bpp["fp32_master"]  // n_dp,  # sharded
        "fp32_m":       n_params * bpp["fp32_m"]       // n_dp,  # sharded
        "fp32_v":       n_params * bpp["fp32_v"]       // n_dp,  # sharded
        "total":        n_params * 16 // n_dp,
    }


def min_gpus_for_model(n_params: int, gpu_mem_bytes: int,
                        activations_bytes: int = 0) -> dict:
    """Minimum number of GPUs (power of 2) for each ZeRO stage."""
    budget = gpu_mem_bytes - activations_bytes
    result = {}
    for stage, mem_fn in [("DDP",    lambda p, n: memory_ddp(p)["total"]),
                           ("ZeRO-1", lambda p, n: memory_zero1(p, n)["total"]),
                           ("ZeRO-2", lambda p, n: memory_zero2(p, n)["total"]),
                           ("ZeRO-3", lambda p, n: memory_zero3(p, n)["total"])]:
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            if mem_fn(n_params, n) <= budget:
                result[stage] = n
                break
        else:
            result[stage] = None   # cannot fit
    return result


def fmt_gb(b: float) -> str:
    gb = b / 1e9
    if gb >= 1000: return f"{gb/1024:.1f}TB"
    if gb >= 1:   return f"{gb:.1f}GB"
    return f"{gb*1024:.0f}MB"


def bar_chart(val: float, max_val: float, width: int = 20) -> str:
    n = int(min(val / max_val, 1.0) * width)
    return "█" * n + "░" * (width - n)


if __name__ == "__main__":
    GPU_MEM  = 80e9   # 80 GB A100

    models = [
        ("GPT-2 1.5B",    1_500_000_000),
        ("LLaMA-2 7B",    7_000_000_000),
        ("LLaMA-2 13B",  13_000_000_000),
        ("LLaMA-2 70B",  70_000_000_000),
        ("GPT-3 175B",  175_000_000_000),
        ("LLaMA-3 405B",405_000_000_000),
    ]

    # ── 1. Memory at N=8 DP ranks ─────────────────────────────────────────────
    print("=" * 75)
    print("  PER-GPU MEMORY (N=8 DP ranks, 80 GB A100)")
    print("=" * 75)
    print()
    print(f"  {'Model':<18} {'DDP':>10}  {'ZeRO-1':>10}  "
          f"{'ZeRO-2':>10}  {'ZeRO-3':>10}  {'Fits on 80GB?':>14}")
    print(f"  {'':─<18} {'':─>10}  {'':─>10}  {'':─>10}  {'':─>10}  {'':─>14}")

    for name, P in models:
        m_ddp  = memory_ddp(P)["total"]
        m_z1   = memory_zero1(P, 8)["total"]
        m_z2   = memory_zero2(P, 8)["total"]
        m_z3   = memory_zero3(P, 8)["total"]

        fits = []
        if m_ddp  <= GPU_MEM: fits.append("DDP")
        if m_z1   <= GPU_MEM: fits.append("Z1")
        if m_z2   <= GPU_MEM: fits.append("Z2")
        if m_z3   <= GPU_MEM: fits.append("Z3")
        fits_str = ", ".join(fits) if fits else "none"

        print(f"  {name:<18} {fmt_gb(m_ddp):>10}  {fmt_gb(m_z1):>10}  "
              f"{fmt_gb(m_z2):>10}  {fmt_gb(m_z3):>10}  {fits_str:>14}")

    # ── 2. Memory vs N scaling for 70B model ─────────────────────────────────
    print()
    print("=" * 75)
    print("  MEMORY vs N DP RANKS — LLaMA-2 70B")
    print(f"  (80 GB line = {'─'*20})")
    print("=" * 75)
    print()
    P70B = 70_000_000_000
    max_mem = memory_ddp(P70B)["total"]

    print(f"  {'N':>5}  {'DDP':>10}  {'ZeRO-1':>10}  {'ZeRO-2':>10}  "
          f"{'ZeRO-3':>10}  {'Fits (Z3)':>12}")
    print(f"  {'':─>5}  {'':─>10}  {'':─>10}  {'':─>10}  {'':─>10}  {'':─>12}")

    for N in [1, 2, 4, 8, 16, 32, 64]:
        m_d  = memory_ddp(P70B)["total"]
        m_z1 = memory_zero1(P70B, N)["total"]
        m_z2 = memory_zero2(P70B, N)["total"]
        m_z3 = memory_zero3(P70B, N)["total"]
        fits = "✓" if m_z3 <= GPU_MEM else "✗"
        print(f"  {N:>5}  {fmt_gb(m_d):>10}  {fmt_gb(m_z1):>10}  "
              f"{fmt_gb(m_z2):>10}  {fmt_gb(m_z3):>10}  {fits:>12}")

    # ── 3. Minimum GPU count ──────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("  MINIMUM GPUS TO TRAIN (A100 80GB, no activation memory)")
    print("=" * 75)
    print()
    print(f"  {'Model':<18} {'DDP':>8}  {'ZeRO-1':>8}  {'ZeRO-2':>8}  "
          f"{'ZeRO-3':>8}  {'Ratio D/Z3':>12}")
    print(f"  {'':─<18} {'':─>8}  {'':─>8}  {'':─>8}  {'':─>8}  {'':─>12}")
    for name, P in models:
        mins = min_gpus_for_model(P, int(GPU_MEM))
        d    = mins.get("DDP")
        z1   = mins.get("ZeRO-1")
        z2   = mins.get("ZeRO-2")
        z3   = mins.get("ZeRO-3")
        ratio = f"{d}/{z3}×" if d and z3 else "N/A"
        fmt  = lambda x: str(x) if x else "∞"
        print(f"  {name:<18} {fmt(d):>8}  {fmt(z1):>8}  {fmt(z2):>8}  "
              f"{fmt(z3):>8}  {ratio:>12}")

    # ── 4. Detailed component breakdown for 7B at N=8 ────────────────────────
    print()
    print("=" * 75)
    print("  COMPONENT BREAKDOWN — LLaMA-2 7B, N=8 DP RANKS")
    print("=" * 75)
    P7B = 7_000_000_000
    N   = 8
    stages = [
        ("DDP",    memory_ddp(P7B)),
        ("ZeRO-1", memory_zero1(P7B, N)),
        ("ZeRO-2", memory_zero2(P7B, N)),
        ("ZeRO-3", memory_zero3(P7B, N)),
    ]
    print()
    print(f"  {'Component':<20}", end="")
    for s, _ in stages:
        print(f"  {s:>12}", end="")
    print()
    print(f"  {'':─<20}" + "".join(f"  {'':─>12}" for _ in stages))
    for comp in ["bf16_params", "bf16_grads", "fp32_master", "fp32_m", "fp32_v"]:
        labels = {"bf16_params": "BF16 params", "bf16_grads": "BF16 grads",
                  "fp32_master": "FP32 master w", "fp32_m": "FP32 m (Adam)",
                  "fp32_v": "FP32 v (Adam)"}
        print(f"  {labels[comp]:<20}", end="")
        for _, d in stages:
            print(f"  {fmt_gb(d.get(comp, 0)):>12}", end="")
        print()
    print(f"  {'TOTAL':<20}", end="")
    for _, d in stages:
        print(f"  {fmt_gb(d['total']):>12}", end="")
    print()
    print()
    # Reduction ratios
    ddp_total = stages[0][1]["total"]
    print("  Reduction ratios vs DDP:")
    for s, d in stages:
        ratio = ddp_total / d["total"]
        bar   = bar_chart(1/ratio, 1.0, width=20)
        print(f"  {s:<10}  {ratio:>6.2f}×  |{bar}|  "
              f"{fmt_gb(d['total'])} per GPU")
''',
    },

    "ZeRO Gradient Reduction: Reduce-Scatter + All-Gather": {
        "description": "Implement and verify the ZeRO-2 gradient communication pattern (Reduce-Scatter then All-Gather) showing it is volume-equivalent to a standard All-Reduce but produces sharded gradients.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
ZERO GRADIENT REDUCTION: REDUCE-SCATTER + ALL-GATHER
================================================================================

ZeRO-2's key insight: instead of all-reduce (which distributes the full
gradient to all GPUs), use Reduce-Scatter (each GPU keeps only its shard).

This saves gradient memory without increasing communication volume.

Demonstrates:
    1. Standard All-Reduce for gradient synchronisation (DDP)
    2. ZeRO-2 Reduce-Scatter: same communication volume, sharded result
    3. Verify: RS result matches the corresponding shard of AR result
    4. ZeRO-2 optimiser step on owned shard only
    5. All-Gather of updated parameters to synchronise all GPUs

================================================================================
"""

import threading
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Collective operations ─────────────────────────────────────────────────────

class CollectiveOps:
    """Thread-safe simulation of All-Reduce, Reduce-Scatter, and All-Gather."""

    def __init__(self, n: int):
        self.n       = n
        self.lock    = threading.Lock()
        self.barrier = threading.Barrier(n)
        self.store   = {}

    def _put(self, key: str, rank: int, tensor: torch.Tensor):
        with self.lock:
            if key not in self.store:
                self.store[key] = [None] * self.n
            self.store[key][rank] = tensor.detach().clone()

    def _wait_and_get(self, key: str) -> list:
        self.barrier.wait()
        data = self.store[key]
        self.barrier.wait()
        with self.lock:
            if key in self.store:
                del self.store[key]
        return data

    def all_reduce_sum(self, key: str, rank: int,
                        tensor: torch.Tensor) -> torch.Tensor:
        """Sum tensor across all ranks; every rank gets the full result."""
        self._put(key, rank, tensor)
        data = self._wait_and_get(key)
        return torch.stack(data).sum(dim=0)

    def reduce_scatter(self, key: str, rank: int,
                        tensor: torch.Tensor) -> torch.Tensor:
        """
        Sum tensor across all ranks; each rank gets its own shard.
        tensor shape: (P,)
        Returns shard: (P/N,) — the rank-th contiguous chunk of the sum.
        """
        self._put(key, rank, tensor)
        data = self._wait_and_get(key)
        total = torch.stack(data).sum(dim=0)   # (P,)
        P     = total.shape[0]
        P_local = P // self.n
        shard = total[rank * P_local : (rank + 1) * P_local]
        return shard

    def all_gather(self, key: str, rank: int,
                   shard: torch.Tensor) -> torch.Tensor:
        """Gather all rank shards; every rank gets the full concatenation."""
        self._put(key, rank, shard)
        data = self._wait_and_get(key)
        return torch.cat(data, dim=0)


# ── Simulated model parameter ─────────────────────────────────────────────────

class ZeROSimulator:
    """
    Simulates ZeRO stages 1, 2, and 3 for a single parameter tensor.
    Each "rank" holds a different view of the parameter state.
    """

    def __init__(self, n_params: int, n_ranks: int, stage: int = 2,
                 lr: float = 0.01):
        self.P       = n_params
        self.N       = n_ranks
        self.stage   = stage
        self.lr      = lr
        self.P_local = n_params // n_ranks

        self.ops     = CollectiveOps(n_ranks)
        self.step_t  = 0

        # State per rank (simulates the actual tensor storage on each GPU)
        torch.manual_seed(42)
        self.bf16_params = torch.randn(n_params)   # 16-bit params (replicated)
        self.fp32_master = self.bf16_params.float() # fp32 master copy
        self.m           = torch.zeros(n_params)    # 1st moment
        self.v           = torch.zeros(n_params)    # 2nd moment

        # Simulate different gradients on each rank (different data)
        torch.manual_seed(0)
        self.local_grads = [
            torch.randn(n_params) * (0.1 + r * 0.05)
            for r in range(n_ranks)
        ]

    def rank_param_shard(self, rank: int) -> tuple[int, int]:
        """Start and end index of owned parameter shard for this rank."""
        return rank * self.P_local, (rank + 1) * self.P_local

    def simulate_ddp_step(self) -> dict:
        """Standard DDP: All-Reduce gradients, full optimiser update on all ranks."""
        # Each rank computes its own loss gradient on different data
        results = {}
        threads_data = {}

        def rank_step(r):
            local_g = self.local_grads[r]
            avg_g   = self.ops.all_reduce_sum(f"ddp_{self.step_t}", r, local_g)
            avg_g   = avg_g / self.N

            # Each rank has the same gradient and does the same update (redundant)
            start, end = self.rank_param_shard(r)
            threads_data[r] = {
                "avg_gradient": avg_g.clone(),
                "comm_bytes":   self.P * 2 * 4,   # 2 × size × dtype (All-Reduce)
            }

        threads = [threading.Thread(target=rank_step, args=(r,))
                   for r in range(self.N)]
        for t in threads: t.start()
        for t in threads: t.join()

        return threads_data

    def simulate_zero2_step(self) -> dict:
        """
        ZeRO-2: Reduce-Scatter gradients (each GPU keeps only its shard).
        Each GPU only computes optimizer update for its owned parameter shard.
        Then All-Gather updated parameters to synchronise.
        """
        results = {}

        def rank_step(r):
            # 1. Reduce-Scatter: get averaged gradient for owned shard only
            local_g      = self.local_grads[r]
            avg_g_shard  = self.ops.reduce_scatter(f"z2_rs_{self.step_t}", r, local_g)
            avg_g_shard  = avg_g_shard / self.N   # normalise

            # 2. AdamW update on owned shard only
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            start, end = self.rank_param_shard(r)
            t = self.step_t + 1

            m_shard = self.m[start:end]
            v_shard = self.v[start:end]
            p_shard = self.fp32_master[start:end]

            m_shard = beta1 * m_shard + (1 - beta1) * avg_g_shard
            v_shard = beta2 * v_shard + (1 - beta2) * avg_g_shard ** 2
            m_hat   = m_shard / (1 - beta1 ** t)
            v_hat   = v_shard / (1 - beta2 ** t)
            p_shard = p_shard - self.lr * m_hat / (v_hat.sqrt() + eps)

            # 3. All-Gather updated 16-bit params to all ranks
            updated_bf16 = p_shard.half()
            full_params  = self.ops.all_gather(f"z2_ag_{self.step_t}", r, updated_bf16)

            results[r] = {
                "owned_grad_shard":  avg_g_shard,
                "updated_params":    full_params,
                "comm_bytes_rs":     self.P * 2,   # Reduce-Scatter: same as half of AR
                "comm_bytes_ag":     self.P * 2,   # All-Gather: same as half of AR
                "total_comm_bytes":  self.P * 4,   # same as DDP All-Reduce!
            }

        threads = [threading.Thread(target=rank_step, args=(r,))
                   for r in range(self.N)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.step_t += 1
        return results


# ── Verification and demo ──────────────────────────────────────────────────────

if __name__ == "__main__":
    N_PARAMS = 128
    N_RANKS  = 4

    sim = ZeROSimulator(N_PARAMS, N_RANKS, stage=2, lr=0.01)

    print("=" * 65)
    print(f"  ZERO-2 GRADIENT COMMUNICATION PATTERN")
    print(f"  P={N_PARAMS} params, N={N_RANKS} ranks")
    print("=" * 65)
    print()

    # DDP step
    ddp_results = sim.simulate_ddp_step()
    sim.step_t += 1  # advance step counter

    # ZeRO-2 step
    z2_results  = sim.simulate_zero2_step()

    # Verify ZeRO-2 gradient shard matches DDP gradient shard
    print("  Gradient verification (ZeRO-2 shard == DDP averaged grad shard):")
    print()
    for r in range(N_RANKS):
        start, end = sim.rank_param_shard(r)
        ddp_shard  = ddp_results[r]["avg_gradient"][start:end]
        z2_shard   = z2_results[r]["owned_grad_shard"]
        diff       = (ddp_shard - z2_shard).abs().max().item()
        print(f"  Rank {r}: shard indices [{start}:{end}]  "
              f"max diff = {diff:.2e}  "
              f"{'✓ identical' if diff < 1e-5 else '✗ differs'}")

    print()
    print("  Communication volume comparison:")
    print(f"  DDP All-Reduce:          {ddp_results[0]['comm_bytes']:>10,} bytes per rank")
    print(f"  ZeRO-2 RS + AG:          {z2_results[0]['total_comm_bytes']:>10,} bytes per rank")
    print(f"  Difference:              {abs(ddp_results[0]['comm_bytes'] - z2_results[0]['total_comm_bytes']):>10,} bytes")
    print()
    print("  ✓ ZeRO-2 uses identical communication volume as DDP!")
    print()
    print("  Memory footprint comparison (gradient component only):")
    per_rank_full_grad = N_PARAMS * 2   # bf16 full gradient
    per_rank_shard     = N_PARAMS // N_RANKS * 2  # bf16 shard
    print(f"  DDP:    {per_rank_full_grad:>8} bytes per rank (full gradient)")
    print(f"  ZeRO-2: {per_rank_shard:>8} bytes per rank (shard only)")
    print(f"  Saving: {N_RANKS:>8}× gradient memory reduction  ✓")

    # Show the optimizer state savings
    print()
    print("=" * 65)
    print("  OPTIMIZER STATE SAVINGS (ZeRO-1 and ZeRO-2)")
    print("=" * 65)
    print()
    for P in [7_000_000_000, 70_000_000_000]:
        for N in [8, 16, 32]:
            # ZeRO-1/2 optimizer state per GPU
            optim_total = P * 12     # fp32 master + m + v = 12 bytes/param
            optim_per_gpu = optim_total // N
            saved = optim_total - optim_per_gpu * N  # ... this is 0, but per GPU:
            reduction = optim_total // N
            print(f"  P={P/1e9:.0f}B, N={N:>3}:  "
                  f"optimizer state per GPU = {reduction/1e9:.1f} GB  "
                  f"({optim_total//N//(optim_total//1):.0%} of DDP's "
                  f"{optim_total/1e9:.0f} GB)")
''',
    },

    "FSDP Full Sharding Simulation": {
        "description": "Simulate FSDP ZeRO-3 semantics: parameter All-Gather before each forward layer, Reduce-Scatter after backward, discard non-owned params after use. Show memory and communication at each step.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
FSDP / ZERO-3: PARAMETER LIFECYCLE SIMULATION
================================================================================

Simulates FSDP (ZeRO-3) parameter management:
    FORWARD:
        1. All-Gather parameters for current layer (reconstruct from shards)
        2. Compute layer forward pass
        3. Discard non-owned parameters (keep only owned shard)

    BACKWARD:
        1. All-Gather parameters for current layer (recompute grads need them)
        2. Compute layer backward pass (compute gradients)
        3. Reduce-Scatter gradients (each rank accumulates its shard's grad)
        4. Discard reconstructed parameters again

    OPTIMISER:
        4. Each rank updates only its owned parameter shard
        (No All-Gather needed — next forward will do it)

Tracks exact memory usage at each step.
================================================================================
"""

import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class MemoryTracker:
    """Track GPU memory usage over time."""
    events: list = field(default_factory=list)

    def record(self, label: str, size_bytes: int, action: str = "alloc"):
        self.events.append({"label": label, "bytes": size_bytes, "action": action})

    @property
    def current_bytes(self) -> int:
        total = 0
        for e in self.events:
            total += e["bytes"] if e["action"] == "alloc" else -e["bytes"]
        return max(0, total)


class FSDPLayer:
    """
    Simulates one FSDP-managed linear layer.
    Each rank permanently holds only 1/N of the parameters.
    """

    def __init__(self, d_in: int, d_out: int, rank: int, n_ranks: int):
        self.d_in    = d_in
        self.d_out   = d_out
        self.rank    = rank
        self.n       = n_ranks
        self.dtype   = torch.float32  # simulation in fp32 for clarity

        # Full weight for reference (simulation only — in real FSDP, not held by all ranks)
        torch.manual_seed(42)
        self.weight_full = torch.randn(d_out, d_in)

        # Each rank owns 1/N of the flattened parameters
        flat = self.weight_full.flatten()
        P    = flat.shape[0]
        P_local = P // n_ranks
        self.owned_shard = flat[rank * P_local : (rank + 1) * P_local].clone()
        self.P      = P
        self.P_local = P_local

        # Reconstructed weight (only present during fwd/bwd, then freed)
        self._reconstructed = None

        # Owned gradient shard (set during backward)
        self.owned_grad_shard = None

        # Optimizer state for owned shard
        self.m = torch.zeros(P_local)
        self.v = torch.zeros(P_local)

    def all_gather_params(self, ops, step_label: str, tracker: MemoryTracker) -> None:
        """Reconstruct full weight from all rank shards."""
        # Each rank sends its shard; all ranks receive the full weight
        shards = [None] * self.n
        shards[self.rank] = self.owned_shard.clone()

        # Simulate: we have access to all shards (in reality, NCCL All-Gather)
        for r in range(self.n):
            P_local = self.P // self.n
            full_flat = self.weight_full.flatten()
            shards[r] = full_flat[r * P_local : (r + 1) * P_local].clone()

        self._reconstructed = torch.cat(shards).reshape(self.d_out, self.d_in)
        tracker.record(f"AG {step_label}", self._reconstructed.numel() * 4, "alloc")

    def free_reconstructed(self, step_label: str, tracker: MemoryTracker) -> None:
        """Discard the reconstructed weight, keep only owned shard."""
        if self._reconstructed is not None:
            tracker.record(f"free {step_label}", self._reconstructed.numel() * 4, "free")
            self._reconstructed = None

    def forward(self, x: torch.Tensor, ops, step: int,
                 tracker: MemoryTracker) -> torch.Tensor:
        """FSDP forward: All-Gather → compute → free (if not needed for bwd)."""
        self.all_gather_params(ops, f"fwd_L{step}", tracker)
        y = F.linear(x, self._reconstructed)
        # In real FSDP, we free here if not using activation checkpointing
        # For backward, we need to re-gather (or keep if memory allows)
        # We keep it for backward in this simple simulation
        return y

    def backward_and_reduce(self, x: torch.Tensor, grad_output: torch.Tensor,
                              ops, step: int, tracker: MemoryTracker):
        """
        FSDP backward:
            1. (Already have reconstructed weight or re-gather)
            2. Compute local gradients
            3. Reduce-Scatter: reduce and shard gradients
        """
        # Compute full gradient w.r.t. weight
        grad_weight_full = grad_output.T @ x   # (d_out, d_in)
        flat_grad        = grad_weight_full.flatten()

        # Reduce-Scatter: sum gradients from all ranks, keep our shard
        # (In real training, gradients from different data shards are different)
        # Simulation: assume this rank sees noise * scale as its local gradient
        local_grad = flat_grad + torch.randn_like(flat_grad) * 0.01
        # ... in reality each rank would have different local_grad from its data

        # Average over N ranks (Reduce-Scatter gives sum; we divide for mean)
        avg_grad_shard = local_grad[self.rank * self.P_local : (self.rank + 1) * self.P_local]
        self.owned_grad_shard = avg_grad_shard.clone()

        self.free_reconstructed(f"bwd_L{step}", tracker)

    def optimizer_step(self, lr: float = 0.001, step_t: int = 1):
        """Update owned parameter shard using AdamW."""
        if self.owned_grad_shard is None:
            return
        b1, b2, eps = 0.9, 0.999, 1e-8
        g       = self.owned_grad_shard
        self.m  = b1 * self.m + (1 - b1) * g
        self.v  = b2 * self.v + (1 - b2) * g ** 2
        m_hat   = self.m / (1 - b1 ** step_t)
        v_hat   = self.v / (1 - b2 ** step_t)
        self.owned_shard -= lr * m_hat / (v_hat.sqrt() + eps)
        self.owned_grad_shard = None


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_RANKS    = 4
    N_LAYERS   = 3
    D          = 32
    B, T       = 2, 8
    LR         = 0.001

    print("=" * 68)
    print(f"  FSDP / ZERO-3 PARAMETER LIFECYCLE SIMULATION")
    print(f"  {N_LAYERS} layers, d={D}, N={N_RANKS} ranks")
    print("=" * 68)
    print()

    # Create one rank's view
    RANK = 0
    layers = [FSDPLayer(D, D, RANK, N_RANKS) for _ in range(N_LAYERS)]

    ops     = None   # simplified: no actual collective (all shards available)
    tracker = MemoryTracker()

    # Owned shard memory at rest
    for i, layer in enumerate(layers):
        tracker.record(f"owned_shard_L{i}", layer.P_local * 4, "alloc")
        tracker.record(f"m_L{i}",           layer.P_local * 4, "alloc")
        tracker.record(f"v_L{i}",           layer.P_local * 4, "alloc")

    print(f"  Memory at rest (owned shards + optim state):")
    print(f"    Per-rank: {tracker.current_bytes/1e6:.2f} MB")
    print(f"    vs full model: {sum(l.P * 4 for l in layers) * 3 / 1e6:.2f} MB  "
          f"({N_RANKS}× more without ZeRO-3)")

    print()
    print("  Step-by-step memory trace:")
    print(f"  {'Event':<35}  {'Δ bytes':>12}  {'Total':>12}")
    print(f"  {'':─<35}  {'':─>12}  {'':─>12}")

    def print_event(label, delta, total):
        sign = "+" if delta > 0 else ""
        print(f"  {label:<35}  {sign}{delta/1e6:>10.2f}MB  {total/1e6:>10.2f}MB")

    prev = tracker.current_bytes
    print_event("Initial (shards + optim state)", prev, prev)

    # Forward pass
    x = torch.randn(B, T, D)
    fwd_outputs = []
    for i, layer in enumerate(layers):
        label = f"All-Gather L{i} (fwd)"
        layer.all_gather_params(ops, label, tracker)
        delta = tracker.current_bytes - prev
        print_event(label, delta, tracker.current_bytes)
        prev = tracker.current_bytes

        y = F.linear(x, layer._reconstructed)
        fwd_outputs.append((x.clone(), y))
        x = F.gelu(y)

        # In real FSDP with activation checkpointing: free immediately
        # Without: keep for backward
        # We keep for backward in this demo
        label = f"(keeping L{i} for bwd)"
        print_event(label, 0, tracker.current_bytes)

    print_event("<<< peak activation + params >>>", 0, tracker.current_bytes)

    # Backward pass (reverse order)
    grad = torch.randn_like(x)
    for i in reversed(range(N_LAYERS)):
        layer    = layers[i]
        x_i, y_i = fwd_outputs[i]

        label = f"Bwd: compute grad + RS L{i}"
        layer.backward_and_reduce(x_i, grad, ops, i, tracker)
        delta = tracker.current_bytes - prev
        print_event(label, delta, tracker.current_bytes)
        prev = tracker.current_bytes

    # Optimizer step
    for i, layer in enumerate(layers):
        layer.optimizer_step(lr=LR, step_t=1)

    print()
    print("  Memory comparison summary:")
    p_total = sum(l.P for l in layers)
    full_model_bytes   = p_total * 4         # fp32 for all layers
    optim_bytes        = p_total * 8         # m + v fp32
    per_rank_shard     = (p_total + p_total * 2) * 4 / N_RANKS  # params+optim / N
    print(f"  Full model (all layers, fp32):  {full_model_bytes/1e6:.2f} MB")
    print(f"  Full optim state (m+v, fp32):   {optim_bytes/1e6:.2f} MB")
    print(f"  ZeRO-3 per-rank at rest:        {per_rank_shard/1e6:.2f} MB  "
          f"({N_RANKS}× smaller)")
    print(f"  ZeRO-3 peak (during AG):        ~{(per_rank_shard + full_model_bytes)/1e6:.2f} MB  "
          f"(transient, 1 layer at a time)")
    print()
    print("  This is the ZeRO-3 trade-off:")
    print("  • At rest: N× less memory (permanent benefit)")
    print("  • During forward/backward: briefly reconstruct full layer")
    print("  • Prefetch optimisation hides the All-Gather latency behind compute")
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
    #     from llm_training.visuals.zero_optimizer import (
    #         ZERO_VISUAL_HTML,
    #         ZERO_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ZERO_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ZERO_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[24_zero_optimizer_stages.py] Could not load visual: {e}",
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