"""
Multi-GPU LLM Training: DDP + ZeRO + Mixed Precision
======================================================

Scaling from one GPU to many requires distributed training infrastructure.
This module presents the complete multi-GPU training recipe used in
production LLM training: DistributedDataParallel (DDP) for synchronising
gradients, ZeRO/FSDP for memory efficiency, mixed precision, gradient
accumulation, and the torchrun launcher. Every distributed-specific detail
is explained and every configuration decision is documented.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Multi-GPU LLM Training"
DISPLAY_NAME = "36 · Multi-GPU Training"
ICON         = "⚡"
SUBTITLE     = "DDP + ZeRO + Mixed Precision"


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

### From Single-GPU to Multi-GPU

Module 35 established the complete single-GPU training loop. Scaling to
multiple GPUs requires three changes:

    1.  **Process coordination:** Each GPU runs its own Python process
        (torchrun spawns one process per GPU).
    2.  **Data partitioning:** Each process handles a different shard of the
        training data (DistributedSampler).
    3.  **Gradient synchronisation:** After each backward pass, gradients
        are averaged across all processes (DDP's all-reduce).

Everything else — the model, loss function, optimizer, LR schedule —
remains identical to the single-GPU recipe.

**The relationship between modules:**
    Module 18: DDP internals (buckets, all-reduce, no_sync)
    Module 24: ZeRO/FSDP memory partitioning
    Module 17: Gradient accumulation (essential with small per-GPU batches)
    Module 35: Single-GPU training template (this module extends it)
    Module 36: Complete multi-GPU integration of all the above


### The torchrun Launcher

`torchrun` (replacing the older `torch.distributed.launch`) is the standard
way to launch distributed training:

    # Single node, 4 GPUs:
    torchrun --nproc_per_node=4 train.py

    # Multi-node (2 nodes × 4 GPUs = 8 GPUs total):
    # Node 0 (master):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
             --master_addr=<node0_ip> --master_port=29500 train.py
    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
             --master_addr=<node0_ip> --master_port=29500 train.py

torchrun sets these environment variables that your training script reads:
    RANK:             Global rank (0 to world_size-1)
    LOCAL_RANK:       Local rank on this node (0 to nproc_per_node-1)
    WORLD_SIZE:       Total number of processes
    MASTER_ADDR:      Address of rank-0 process
    MASTER_PORT:      Port for rendezvous


### Process Group Initialisation

Every distributed training script begins with:

    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    device      = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # At the end of training:
    dist.destroy_process_group()

**Backend choice:**
    NCCL:   Best for CUDA GPUs (uses NVLink/InfiniBand directly)
    Gloo:   CPU tensors, debugging, no CUDA
    MPI:    Legacy, rarely used with PyTorch

Always use NCCL for CUDA training.

**The rank hierarchy:**
    WORLD_SIZE = 8 (2 nodes × 4 GPUs):
    Global rank 0 = node 0, GPU 0
    Global rank 1 = node 0, GPU 1
    ...
    Global rank 4 = node 1, GPU 0
    ...
    Global rank 7 = node 1, GPU 3

    LOCAL_RANK = rank within the node (always 0–3 on a 4-GPU node)
    The LOCAL_RANK determines which GPU this process uses.


### DDP vs FSDP: Which to Use?

    Model size     Training mode    Recommendation
    ──────────────────────────────────────────────────────────────
    < 3B params    Full fine-tuning  DDP (simple, fast)
    3B–13B params  Full fine-tuning  DDP + ZeRO-2, or FSDP SHARD_GRAD_OP
    > 13B params   Full fine-tuning  FSDP FULL_SHARD (ZeRO-3)
    Any size       LoRA fine-tuning  DDP (LoRA params are small)
    ──────────────────────────────────────────────────────────────

For pre-training large models from scratch (70B+), Megatron-LM's 3D
parallelism (Module 23) is used. FSDP FULL_SHARD handles up to ~70B
models on 8×80GB A100s without pipeline parallelism.


### DDP Training Loop Specifics

**Key differences from single-GPU:**

    1.  Wrap model with DDP after moving to GPU:
            model = DDP(model, device_ids=[local_rank])

    2.  Use DistributedSampler for the data loader:
            sampler = DistributedSampler(dataset, shuffle=True)
            # CRITICAL: call sampler.set_epoch(epoch) each epoch

    3.  Use no_sync() context for gradient accumulation:
            with model.no_sync():   # disable all-reduce for K-1 micro-batches
                loss.backward()
            loss.backward()   # trigger all-reduce on final micro-batch

    4.  Only rank 0 should log and save checkpoints:
            if rank == 0:
                print(f"Loss: {loss:.4f}")
                save_checkpoint(model.module.state_dict(), ...)
            # Note: model.module to access the wrapped model

    5.  Synchronise metrics across ranks before logging:
            loss_tensor = torch.tensor(loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            # loss_tensor now contains the mean loss across all ranks


### Effective Batch Size and Learning Rate Scaling

With N GPUs, the effective batch size scales by N (each GPU processes its own
micro-batch, and all gradients are averaged). This changes the effective LR:

    single GPU: batch=4, grad_accum=8,  eff_batch=32, LR=3e-4
    8 GPUs:     batch=4, grad_accum=8,  eff_batch=256, LR = ?

**Linear scaling rule:** LR ∝ effective_batch
    LR = 3e-4 × (256 / 32) = 2.4e-3

**Square-root scaling rule (safer for LLMs):**
    LR = 3e-4 × √(256 / 32) = 8.5e-4

In practice, scale linearly with N GPUs and a careful warmup period to
avoid instability at the start of training with the higher LR.


### FSDP: The Modern Alternative

For large models that don't fit in memory even with DDP, FSDP (FullyShardedDataParallel)
is PyTorch's implementation of ZeRO Stage 3:

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    # Wrap each transformer block independently
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # FSDP with ZeRO-3 and bf16
    mp_policy = MixedPrecision(
        param_dtype    = torch.bfloat16,
        reduce_dtype   = torch.float32,   # fp32 gradient reduction
        buffer_dtype   = torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy       = ShardingStrategy.FULL_SHARD,  # ZeRO-3
        auto_wrap_policy        = auto_wrap,
        mixed_precision         = mp_policy,
        backward_prefetch       = BackwardPrefetch.BACKWARD_PRE,
        device_id               = torch.cuda.current_device(),
    )

**Key FSDP considerations:**
    •   auto_wrap_policy tells FSDP which submodules to wrap independently
        (each wrapped module's state is sharded separately)
    •   BACKWARD_PRE prefetches the next layer's parameters during backward
    •   Each rank only holds 1/N of the model at rest, all-gathers on demand
    •   Checkpoint saving requires special handling (Module 33)


    **Diagram 1 — DDP vs FSDP Memory:**

    DDP (8 GPUs, 7B model):
    ════════════════════════════════════════════════════════════════
    Each GPU holds:  14 GB (weights) + 14 GB (grads) + 56 GB (optimizer)
                   = 84 GB  ← barely fits A100 80GB!

    FSDP FULL_SHARD (8 GPUs, 7B model):
    ════════════════════════════════════════════════════════════════
    Each GPU holds:  14/8 GB (weights) + 14/8 GB (grads) + 56/8 GB (optimizer)
                   = 10.5 GB  ← massive memory savings!
    Plus activations and transient all-gather buffers.


### Gradient Accumulation with DDP: The no_sync() Trick

When using gradient accumulation with DDP, the all-reduce should only happen
on the LAST micro-batch, not every micro-batch (Module 17):

    for k in range(grad_accum_steps):
        is_last = (k == grad_accum_steps - 1)
        # Suppress all-reduce for all but the last micro-batch
        ctx = contextlib.nullcontext() if is_last else model.no_sync()
        with ctx:
            loss = model(batch[k]) / grad_accum_steps
            loss.backward()

    # All-reduce triggered by the final loss.backward()
    optimizer.step()

Without `no_sync()`, DDP would perform grad_accum × all-reduce operations
instead of 1, multiplying communication cost by grad_accum.


### Mixed Precision in Distributed Training

Mixed precision with DDP is identical to single-GPU (just add autocast):

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, targets=y)
    loss = loss / grad_accum_steps
    loss.backward()

For FSDP, mixed precision is specified in the MixedPrecision policy rather
than via autocast. The policy ensures all-gather operations use the right dtype.


### Monitoring Distributed Training

Key metrics to monitor across all ranks:

    **Per-rank metrics (logged from rank 0 only):**
        train_loss: all-reduced across ranks → mean loss
        grad_norm: all-reduced → max or mean across ranks
        tokens_per_second: rank 0 measures overall throughput

    **Synchronisation health:**
        All losses should be identical across ranks after all-reduce.
        If rank 0 sees NaN but others don't → hardware error on rank 0's GPU.
        If all ranks see NaN → numerical overflow in forward or backward.

    **The distributed barrier:**
        dist.barrier()   # all ranks must reach this before any proceeds
        Use sparingly (adds synchronisation overhead) but essential before
        checkpoint saves and eval passes.


### Common Distributed Training Bugs

    Bug                         Symptom                     Fix
    ────────────────────────────────────────────────────────────────────────
    Missing set_epoch()         Same data order every epoch  sampler.set_epoch(epoch)
    Saving from all ranks       Multiple conflicting files   if rank == 0: save
    Using model.state_dict()    Gets only local shard (FSDP)  Use model.module or DCP
    Forgetting no_sync()        K× communication overhead    Use no_sync() for K-1 steps
    Different seeds per rank    Non-reproducible training    Use rank-specific seeds
    Wrong device_ids in DDP     Incorrect GPU assignment     DDP(model, device_ids=[local_rank])
    Missing dist.barrier()      Race condition in checkpt    Add barrier before/after save
    ────────────────────────────────────────────────────────────────────────


### Production Multi-GPU Recipe Summary

    # 1. Initialise distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # 2. Create model and move to GPU
    model = LLM(config).to(device)
    model = DDP(model, device_ids=[local_rank])   # or FSDP for large models

    # 3. Distributed sampler
    sampler = DistributedSampler(dataset, shuffle=True)
    loader  = DataLoader(dataset, sampler=sampler, ...)

    # 4. Training loop
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)          # CRITICAL
        for step, batch in enumerate(loader):
            # Gradient accumulation with no_sync
            ...
            # All-reduce metrics before logging (rank 0 only)
            if rank == 0:
                log(metrics)
            # Checkpoint (rank 0 only, with barrier)
            dist.barrier()
            if rank == 0:
                save_checkpoint(model.module.state_dict(), ...)

    # 5. Cleanup
    dist.destroy_process_group()
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
DDP vs FSDP Configuration Guide

| Scenario                     | Method           | Memory/GPU  | Speed  | Complexity |
|------------------------------|------------------|-------------|--------|------------|
| < 3B, any GPU count          | DDP              | P×16/N (opt)| Fast   | Low        |
| 7B on 8×40GB (fine-tune)     | DDP + ZeRO-2     | ~10 GB      | Fast   | Low        |
| 7B on 8×80GB (pre-train)     | DDP              | ~84 GB      | Fastest| Very low   |
| 13B on 8×80GB                | FSDP SHARD_GRAD_OP| ~40 GB     | Fast   | Medium     |
| 70B on 8×80GB                | FSDP FULL_SHARD  | ~20 GB      | Medium | High       |
| 70B+ (production scale)      | Megatron 3D      | Varies      | Best   | Very high  |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "DDP Training Script — Production Ready": {
        "description": "A complete multi-GPU DDP training script that handles process group init, DistributedSampler, no_sync gradient accumulation, metric reduction, and checkpoint saving — runnable as-is with torchrun.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MULTI-GPU DDP TRAINING SCRIPT — PRODUCTION READY
================================================================================

Run with torchrun:
    torchrun --nproc_per_node=4 this_script.py

Or single-process simulation (no GPU required):
    python this_script.py

Features:
    - Process group init with environment variable detection
    - DDP-wrapped model with correct device placement
    - DistributedSampler with set_epoch() for reproducible shuffling
    - Gradient accumulation with model.no_sync() (K-1 steps)
    - Metric all-reduction for accurate global loss logging
    - Rank-0-only logging and checkpointing
    - Distributed barrier before/after checkpoint saves

================================================================================
"""

import os
import math
import time
import contextlib
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ── Distributed setup ────────────────────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int, torch.device]:
    """
    Initialise the distributed process group.
    Returns (rank, local_rank, world_size, device).
    Falls back to single-process mode if not launched with torchrun.
    """
    if "RANK" in os.environ and dist.is_available():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank       = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, local_rank, world_size, device


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def all_reduce_mean(tensor: torch.Tensor) -> float:
    """Average a scalar tensor across all ranks."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


# ── Model (same as Module 35) ─────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt() * self.w


class Block(nn.Module):
    def __init__(self, d: int, d_ff: int, heads: int):
        super().__init__()
        self.n1   = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2   = RMSNorm(d)
        self.ffn  = nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.GELU(),
                                    nn.Linear(d_ff, d, bias=False))

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


class LLM(nn.Module):
    def __init__(self, vocab, d, n_layers, heads, d_ff, max_len):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, d_ff, heads) for _ in range(n_layers)])
        self.norm   = RMSNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x, y=None):
        h      = self.embed(x)
        for b in self.blocks:
            h  = b(h)
        logits = self.head(self.norm(h))
        if y is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                    y[:, 1:].reshape(-1))
            return logits, loss
        return logits, None

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, n_tokens: int, vocab: int, seq_len: int, seed: int):
        torch.manual_seed(seed)
        self.data    = torch.randint(0, vocab, (n_tokens,))
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.data[start : start + self.seq_len + 1]


def collate_fn(batch):
    seqs = torch.stack(batch)
    return seqs[:, :-1], seqs[:, 1:]


# ── Cosine LR schedule ────────────────────────────────────────────────────────

def get_lr(step: int, max_lr: float, min_lr: float,
            warmup: int, n_steps: int) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= n_steps:
        return min_lr
    p = (step - warmup) / (n_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


# ── DDP training loop ─────────────────────────────────────────────────────────

@dataclass
class Config:
    vocab:          int   = 512
    d:              int   = 128
    n_layers:       int   = 4
    heads:          int   = 4
    d_ff:           int   = 512
    max_len:        int   = 64
    n_steps:        int   = 100
    micro_batch:    int   = 4
    grad_accum:     int   = 2
    max_lr:         float = 3e-4
    min_lr:         float = 3e-5
    warmup:         int   = 10
    weight_decay:   float = 0.1
    grad_clip:      float = 1.0
    n_train_tokens: int   = 30_000
    n_val_tokens:   int   = 5_000
    log_every:      int   = 10
    val_every:      int   = 50


def main():
    rank, local_rank, world_size, device = setup_distributed()
    cfg = Config()

    if is_main(rank):
        print("=" * 60)
        print(f"  MULTI-GPU DDP TRAINING ({world_size} GPU(s))")
        print(f"  rank={rank}, local_rank={local_rank}, device={device}")
        print("=" * 60)
        print()

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(42)   # same init on all ranks
    model = LLM(cfg.vocab, cfg.d, cfg.n_layers, cfg.heads, cfg.d_ff, cfg.max_len)
    model = model.to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    if is_main(rank):
        base_model = model.module if dist.is_initialized() else model
        print(f"  Parameters: {base_model.n_params:,}")
        eff_batch = cfg.micro_batch * cfg.grad_accum * world_size
        print(f"  Effective batch: {cfg.micro_batch} × {cfg.grad_accum} accum "
              f"× {world_size} GPUs = {eff_batch} sequences")
        print(f"  Effective tokens/step: {eff_batch * cfg.max_len:,}")
        print()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = TokenDataset(cfg.n_train_tokens, cfg.vocab, cfg.max_len, seed=0)
    val_ds   = TokenDataset(cfg.n_val_tokens,   cfg.vocab, cfg.max_len, seed=1)

    # DistributedSampler partitions data across ranks
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.micro_batch,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=collate_fn, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.micro_batch, shuffle=False,
        collate_fn=collate_fn, drop_last=True, num_workers=0,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    params = model.module.parameters() if dist.is_initialized() else model.parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg.max_lr, weight_decay=cfg.weight_decay)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    t0          = time.perf_counter()
    train_iter  = iter(train_loader)
    amp_ctx     = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                   if device.type == "cuda" else contextlib.nullcontext())

    while global_step < cfg.n_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(global_step)   # different shuffle each step

        current_lr = get_lr(global_step, cfg.max_lr, cfg.min_lr,
                             cfg.warmup, cfg.n_steps)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        # ── Gradient accumulation with no_sync() ──────────────────────────────
        for k in range(cfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                if train_sampler:
                    train_sampler.set_epoch(global_step + k)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # Suppress DDP all-reduce for all but the last micro-batch
            is_last_accum = (k == cfg.grad_accum - 1)
            sync_ctx = (contextlib.nullcontext()
                        if (not dist.is_initialized() or is_last_accum)
                        else model.no_sync())

            with sync_ctx:
                with amp_ctx:
                    _, loss = model(x, targets=y)
                    loss    = loss / cfg.grad_accum
                loss.backward()

            step_loss += loss.item()

        # Clip + step
        if dist.is_initialized():
            # Gradient already all-reduced by DDP; clip the averaged gradient
            pass
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (model.module if dist.is_initialized() else model).parameters(),
            cfg.grad_clip
        ).item()

        optimizer.step()
        global_step += 1

        # ── Metric synchronisation + logging ──────────────────────────────────
        if global_step % cfg.log_every == 0:
            # Average loss across all ranks
            loss_t = torch.tensor(step_loss * cfg.grad_accum, device=device)
            avg_loss = all_reduce_mean(loss_t)

            if is_main(rank):
                elapsed  = time.perf_counter() - t0
                tok_ps   = (global_step * cfg.grad_accum * cfg.micro_batch
                             * cfg.max_len * world_size) / elapsed
                print(f"  step {global_step:>4}/{cfg.n_steps}  "
                      f"loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                      f"grad={grad_norm:.3f}  tok/s={tok_ps:,.0f}")

        # ── Validation ────────────────────────────────────────────────────────
        if global_step % cfg.val_every == 0:
            model.eval()
            val_loss = 0.0
            n_val    = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    with amp_ctx:
                        _, loss = model(xv, targets=yv)
                    val_loss += loss.item()
                    n_val += 1

            # Synchronise val loss across ranks
            val_t = torch.tensor(val_loss / max(n_val, 1), device=device)
            avg_val = all_reduce_mean(val_t)

            if is_main(rank):
                print(f"  >>> VAL step={global_step}: "
                      f"loss={avg_val:.4f}  ppl={math.exp(avg_val):.2f}")

            # Checkpoint: only rank 0 saves, all ranks barrier
            if dist.is_initialized():
                dist.barrier()   # ensure all ranks finish val before rank 0 saves
            if is_main(rank):
                base_model = model.module if dist.is_initialized() else model
                # In production: torch.save(base_model.state_dict(), path)
                print(f"  [Checkpoint saved at step {global_step}]")
            if dist.is_initialized():
                dist.barrier()   # all ranks wait for rank 0 to finish saving

            model.train()

    if is_main(rank):
        print()
        print("  Training complete!")
        print(f"  Total wall time: {(time.perf_counter()-t0)/60:.1f} minutes")

    cleanup()


if __name__ == "__main__":
    main()
''',
    },

    "Distributed Metric Synchronisation and Monitoring": {
        "description": "Implement distributed metric reduction patterns — all-reduce for scalars, barrier-based synchronisation, and a distributed training monitor that aggregates metrics across all ranks.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DISTRIBUTED METRIC SYNCHRONISATION
================================================================================

In distributed training, each rank computes metrics (loss, accuracy, grad norms)
independently. To report a single global metric, these must be synchronised:

    •   All-reduce (sum/mean): gradient synchronisation, loss averaging
    •   All-gather: collect per-rank metrics for analysis
    •   Broadcast: share rank-0 values with all ranks

This module shows the correct pattern for each case and demonstrates how
distributed metrics differ from single-process metrics.

================================================================================
"""

import torch
import torch.distributed as dist
import os
import math
from dataclasses import dataclass, field
from typing import Optional
import threading


# ── Metric types and reduction patterns ──────────────────────────────────────

class DistributedMetrics:
    """
    Accumulated metrics that are periodically reduced across all ranks.

    Usage:
        metrics = DistributedMetrics(device)
        metrics.update("loss", loss.item(), count=batch_size)
        metrics.update("correct", n_correct, count=batch_size)

        # Reduce and get results across all ranks
        results = metrics.reduce_and_reset()
        if rank == 0:
            print(f"Global loss: {results['loss']:.4f}")
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device  = device
        self._sums:  dict[str, float] = {}
        self._counts: dict[str, int]  = {}

    def update(self, key: str, value: float, count: int = 1):
        """Accumulate a metric value weighted by count."""
        self._sums[key]   = self._sums.get(key, 0.0)   + value * count
        self._counts[key] = self._counts.get(key, 0)   + count

    def reduce_and_reset(self) -> dict[str, float]:
        """
        All-reduce accumulated metrics across all ranks.
        Returns mean values per metric.
        Only rank 0 gets meaningful values (others can use them too after reduce).
        """
        is_distributed = dist.is_available() and dist.is_initialized()
        results = {}

        for key in self._sums:
            # Package sum and count into a tensor for efficient all-reduce
            tensor = torch.tensor(
                [self._sums[key], self._counts[key]],
                dtype=torch.float64, device=self.device
            )
            if is_distributed:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            total_sum, total_count = tensor[0].item(), tensor[1].item()
            results[key] = total_sum / max(total_count, 1)

        # Reset accumulators
        self._sums.clear()
        self._counts.clear()

        return results


def simulate_distributed_metrics(world_size: int = 4) -> None:
    """
    Simulate distributed metric accumulation using threads.
    Demonstrates how all-reduce combines per-rank metrics.
    """
    import random

    # Simulate each rank having different loss values (different data batches)
    torch.manual_seed(42)
    rank_losses = {r: [random.gauss(2.0, 0.3) for _ in range(10)] for r in range(world_size)}
    rank_batches = {r: [4] * 10 for r in range(world_size)}

    print("  Per-rank individual losses (first 5 steps):")
    print(f"  {'Step':>6}", end="")
    for r in range(world_size):
        print(f"  {'Rank '+str(r):>10}", end="")
    print(f"  {'Global mean':>13}")
    print(f"  {'':─>6}" + "".join(f"  {'':─>10}" for _ in range(world_size)) + f"  {'':─>13}")

    for step in range(5):
        rank_l = {r: rank_losses[r][step] for r in range(world_size)}
        global_mean = sum(rank_l[r] * rank_batches[r][step]
                           for r in range(world_size)) / sum(rank_batches[r][step]
                           for r in range(world_size))
        print(f"  {step+1:>6}", end="")
        for r in range(world_size):
            print(f"  {rank_l[r]:>10.4f}", end="")
        print(f"  {global_mean:>13.4f}")

    print()

    # Show what goes wrong without synchronisation
    naive_mean_r0 = sum(rank_losses[0]) / len(rank_losses[0])
    true_global   = sum(
        sum(rank_losses[r]) for r in range(world_size)
    ) / (world_size * len(rank_losses[0]))

    print(f"  Rank 0 loss (unsynchronised): {naive_mean_r0:.4f}")
    print(f"  True global mean (all-reduce): {true_global:.4f}")
    print(f"  Relative error if we only use rank 0: "
          f"{abs(naive_mean_r0 - true_global)/true_global*100:.1f}%")
    print()
    print("  For small world_size and similar data: error is small.")
    print("  For very imbalanced data or large world_size: error compounds.")
    print("  Always all-reduce metrics before logging from rank 0!")


# ── Per-rank gradient norm analysis ──────────────────────────────────────────

def simulate_gradient_health(world_size: int = 4) -> None:
    """
    Show how gradient norms can differ across ranks and what all-reduce gives.

    In DDP: gradients are all-reduced BEFORE optimizer step.
    So all ranks do the same optimizer step with the same global gradient.
    But individual gradient norms BEFORE all-reduce may differ.
    """
    import random
    random.seed(0)

    print("  GRADIENT NORMS: BEFORE vs AFTER ALL-REDUCE")
    print()
    print("  (In DDP, all-reduce happens inside loss.backward())")
    print()

    # Simulate per-rank local gradients (from different data batches)
    n_params = 1000
    rank_grads = {
        r: torch.randn(n_params) * (0.5 + r * 0.2)  # rank r has different grad magnitude
        for r in range(world_size)
    }

    print("  Local gradient norms (before DDP all-reduce):")
    for r in range(world_size):
        local_norm = rank_grads[r].norm().item()
        print(f"    Rank {r}: grad_norm = {local_norm:.4f}")

    # All-reduce (average across ranks)
    avg_grad = torch.stack(list(rank_grads.values())).mean(0)
    global_norm = avg_grad.norm().item()
    print()
    print(f"  Global gradient norm (after DDP all-reduce): {global_norm:.4f}")
    print()
    print("  After all-reduce: all ranks have the same gradient and grad_norm.")
    print("  The global norm is what gets clipped by clip_grad_norm_().")


# ── Communication volume analysis ─────────────────────────────────────────────

def communication_volume_analysis(n_params: int = 7_000_000_000,
                                    world_sizes: list = [1, 2, 4, 8, 16, 32, 64]) -> None:
    """
    Compute the communication volume for DDP gradient all-reduce
    as a function of world_size and parameter count.
    """
    print("  DDP ALL-REDUCE COMMUNICATION VOLUME (7B model, bf16)")
    print()

    # Ring all-reduce: each rank sends 2*(N-1)/N × grad_size bytes
    dtype_bytes = 2  # bf16
    grad_size_gb = n_params * dtype_bytes / 1e9

    print(f"  {'World size':>12}  {'Comm/rank (GB)':>16}  "
          f"{'Bandwidth req.':>18}  {'Time (IB 25GB/s)':>20}")
    print(f"  {'':─>12}  {'':─>16}  {'':─>18}  {'':─>20}")

    for N in world_sizes:
        # Ring all-reduce: 2*(N-1)/N per rank
        ring_factor = 2 * (N - 1) / max(N, 1)
        comm_gb     = ring_factor * grad_size_gb
        # Effective bandwidth: parallel across all links
        time_ib     = comm_gb / 25.0   # InfiniBand 25 GB/s
        time_nvlink = comm_gb / 600.0  # NVLink 600 GB/s

        note = "same node (NVLink)" if N <= 8 else "cross-node (IB)"
        time_str = f"{time_nvlink*1000:.0f}ms (NVLink)" if N <= 8 else f"{time_ib*1000:.0f}ms (IB)"
        print(f"  {N:>12}  {comm_gb:>16.1f}  {comm_gb:>18.1f}  {time_str:>20}")

    print()
    print("  Key insight: communication volume per rank is roughly constant")
    print("  (scales as 2×(N-1)/N ≈ 2 for large N).")
    print("  But the bottleneck shifts from NVLink to InfiniBand beyond 8 GPUs.")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  DISTRIBUTED METRIC SYNCHRONISATION PATTERNS")
    print("=" * 65)
    print()

    # 1. Per-rank vs global metrics
    print("  1. PER-RANK vs GLOBAL METRIC COMPARISON")
    print("  " + "─" * 50)
    simulate_distributed_metrics(world_size=4)

    # 2. Gradient norms
    print("  2. GRADIENT NORM DISTRIBUTION ACROSS RANKS")
    print("  " + "─" * 50)
    simulate_gradient_health(world_size=4)

    # 3. Communication volume
    print("  3. COMMUNICATION VOLUME ANALYSIS")
    print("  " + "─" * 50)
    communication_volume_analysis(n_params=7_000_000_000)

    # 4. Correct barrier pattern
    print()
    print("=" * 65)
    print("  CORRECT CHECKPOINT BARRIER PATTERN")
    print("=" * 65)
    print()
    print("  # Correct pattern (prevents race conditions):")
    print("  dist.barrier()              # wait for all ranks to finish")
    print("  if rank == 0:")
    print("      torch.save(...)         # rank 0 saves checkpoint")
    print("  dist.barrier()              # wait for rank 0 to finish saving")
    print("                              # now all ranks can safely reload")
    print()
    print("  # Why both barriers?")
    print("  # 1st barrier: ensure no rank starts a new forward pass")
    print("  #              while rank 0 is still saving the checkpoint")
    print("  # 2nd barrier: ensure rank 0 finishes before any rank needs")
    print("  #              to load the checkpoint (e.g., after preemption)")
    print()

    # 5. DistributedMetrics class demo
    print("  DistributedMetrics class usage:")
    device = torch.device("cpu")
    metrics = DistributedMetrics(device)
    for step in range(5):
        # Simulate each step generating different values
        loss   = 2.5 - step * 0.1 + torch.randn(1).item() * 0.05
        n_toks = 256
        metrics.update("loss",     loss,   count=n_toks)
        metrics.update("n_tokens", n_toks, count=1)

    results = metrics.reduce_and_reset()
    print(f"  Accumulated over 5 steps:")
    print(f"    Mean loss:   {results['loss']:.4f}")
    print(f"    Total tokens: {int(results['n_tokens'] * 5)}")
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
    #     from llm_training.visuals.multigpu_training import (
    #         MULTIGPU_VISUAL_HTML,
    #         MULTIGPU_VISUAL_HEIGHT,
    #     )
    #     visual_html   = MULTIGPU_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = MULTIGPU_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[36_multigpu_llm_training.py] Could not load visual: {e}",
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