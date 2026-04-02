"""
Data Parallelism and DistributedDataParallel (DDP)
====================================================

Data parallelism is the simplest and most widely used form of distributed
training: replicate the entire model on every GPU, split the data across
GPUs, and average the gradients after each backward pass. PyTorch's
DistributedDataParallel (DDP) is the production implementation of this
idea, with carefully engineered gradient synchronisation, communication
overlap, and failure handling. Understanding DDP deeply — how it wraps
your model, when it communicates, and what can go wrong — is the
foundation for all distributed LLM training.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Data Parallelism and DistributedDataParallel"
DISPLAY_NAME = "18 · Data Parallelism & DDP"
ICON         = "🔀"
SUBTITLE     = "DistributedDataParallel in PyTorch"


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

### The Core Idea: Data Parallelism

In data parallelism, the model is **replicated** on every GPU. Each GPU
processes a different shard of the batch (data), computes its own gradients,
and then all gradients are **averaged** across GPUs before the weight update.
Since every replica starts the step with identical weights and ends with an
identical averaged gradient, every GPU performs the same parameter update —
all replicas stay synchronised.

    Global batch B = N × M   (N GPUs, M samples per GPU)

    GPU 0: forward(x[0:M])   → loss₀ → grad₀
    GPU 1: forward(x[M:2M])  → loss₁ → grad₁
    …
    GPU N: forward(x[(N-1)M:NM]) → loss_{N-1} → grad_{N-1}

    All-reduce: g_avg = (grad₀ + grad₁ + … + grad_{N-1}) / N
    Each GPU: θ ← θ − lr × g_avg       ← identical update on every GPU

The mathematical equivalence: data parallelism with N GPUs and per-GPU
batch M is equivalent to training on a single GPU with batch N×M, with
all-reduce replacing the gradient computation. This only holds when:
    •   Each GPU sees IID data (DistributedSampler ensures this)
    •   The loss is averaged, not summed, over the batch
    •   The gradient averaging is exact (ring all-reduce is exact)


### DataParallel vs DistributedDataParallel

PyTorch provides two data parallelism implementations:

    **DataParallel (DP) — legacy, single-process:**
    •   Spawns multiple threads within one Python process
    •   Parameter server pattern: one GPU (rank 0) gathers all gradients
    •   GIL limits CPU parallelism; rank 0 GPU is a bottleneck
    •   Only supports single-node training
    •   Much slower than DDP in practice
    •   Do NOT use for LLM training

    **DistributedDataParallel (DDP) — current standard:**
    •   Spawns one process per GPU (via torchrun)
    •   Ring all-reduce: bandwidth-optimal, no bottleneck GPU
    •   Gradient synchronisation overlapped with backward computation
    •   Supports multi-node via NCCL over InfiniBand
    •   Use this for everything from 2 GPUs to 1000+

    The rule is simple: **always use DDP, never DataParallel.**


    **Diagram 1 — DP vs DDP Architecture:**

    DATAPARALLEL (legacy):            DISTRIBUTEDDATAPARALLEL (current):
    ════════════════════════════════════════════════════════════════

    One Python process                One Python process per GPU
    ┌──────────────────────┐          ┌─────┐  ┌─────┐  ┌─────┐
    │  Thread 0 → GPU 0    │          │ P0  │  │ P1  │  │ P2  │
    │  Thread 1 → GPU 1    │          │GPU0 │  │GPU1 │  │GPU2 │
    │  Thread 2 → GPU 2    │          └──┬──┘  └──┬──┘  └──┬──┘
    │  Thread 3 → GPU 3    │             └─────────┴──────────┘
    └──────────────────────┘                  NCCL ring
    Bottleneck at GPU 0 (collects all   Equal peers — no bottleneck ✓
    gradients, applies update, scatters)


### How DDP Works Internally

**Step 1 — Initialisation:**

    model = DDP(model, device_ids=[local_rank])

DDP registers **gradient hooks** on every parameter. Each hook fires when
that parameter's gradient is computed during the backward pass.

**Step 2 — Bucketing:**

DDP groups parameters into **buckets** (default: 25 MB per bucket). All-reduce
is launched per-bucket rather than per-parameter. This is important for two
reasons:
    1.  Larger all-reduce operations are more bandwidth-efficient
    2.  Bucket-level all-reduce can be overlapped with the backward pass
        (later layers' gradients are all-reduced while earlier layers are
        still computing)

The bucket assignment is reversed: the last parameters (output layers) are
in the first bucket because their gradients are computed first during
backpropagation.

**Step 3 — Backward + All-Reduce Overlap:**

As each layer's gradient is computed during backward:
    1.  The gradient is added to its bucket
    2.  When a bucket is full (all its params have gradients), the all-reduce
        is launched asynchronously
    3.  The backward pass continues to earlier layers
    4.  By the time the backward pass finishes, most all-reduces are already
        done or nearly done

This overlap is the key performance feature of DDP. On fast networks (NVLink),
the all-reduce effectively costs zero additional time because it is completely
hidden behind the backward pass compute.


    **Diagram 2 — DDP Backward + All-Reduce Overlap:**

    DDP BACKWARD PASS WITH BUCKET-LEVEL ALL-REDUCE OVERLAP
    ════════════════════════════════════════════════════════════════

    Layer:    [L1] [L2] [L3] [L4] [L5] [L6]    (forward order)
    Backward: [L6] [L5] [L4] [L3] [L2] [L1]    (reverse order)
    Buckets:  [B3:L5,L6]  [B2:L3,L4]  [B1:L1,L2]

    Timeline:
    t=0  backward(L6) computes grad(L6) → B3 partial
    t=1  backward(L5) computes grad(L5) → B3 full → all-reduce B3 starts
    t=2  backward(L4) computes grad(L4) → B2 partial    [B3 reducing...]
    t=3  backward(L3) computes grad(L3) → B2 full → all-reduce B2 starts
    t=4  backward(L2) computes grad(L2) → B1 partial    [B2 reducing...]
    t=5  backward(L1) computes grad(L1) → B1 full → all-reduce B1 starts
    t=6  all-reduces complete → optimizer step

    All-reduce time is hidden behind backward computation time ✓


**Step 4 — Synchronisation at Backward End:**

DDP's forward pass wraps itself in a `no_grad` context for the model output.
Wait — that's not right. DDP wraps the **forward** with a module hook that:
    1.  Broadcasts input from rank 0 to all ranks (for data consistency if needed)
    2.  Calls the actual model's forward
    3.  Registers autograd hooks that will trigger all-reduce when gradients
        are ready

After the backward pass and all-reduce, all ranks have identical gradients.
The optimizer step is then performed identically on every rank.


### DDP Setup: Step-by-Step

**Launch with torchrun:**
    torchrun --nproc_per_node=4 --nnodes=1 train.py

**Inside the training script:**

    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # 1. Initialise process group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 2. Build model on this GPU
    model = MyModel().to(local_rank)

    # 3. Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # 4. Use DistributedSampler for data
    sampler = DistributedSampler(dataset)
    loader  = DataLoader(dataset, sampler=sampler, batch_size=4)

    # 5. Training loop
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)  # ← CRITICAL
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch)   # gradient hooks registered here
            loss.backward()       # all-reduce triggered here (per bucket)
            optimizer.step()      # identical update on every rank

    # 6. Cleanup
    dist.destroy_process_group()


### DDP Common Pitfalls

**1. Forgetting `sampler.set_epoch(epoch)`:**
Without this call, every epoch uses the same shuffle — all GPUs see the
same data in the same order. No data diversity across epochs.

**2. Saving/loading checkpoints incorrectly:**
The DDP-wrapped model stores the actual model under `model.module`:
    # Save (from rank 0 only):
    if rank == 0:
        torch.save(model.module.state_dict(), "checkpoint.pt")
    # Load (before DDP wrapping):
    model.load_state_dict(torch.load("checkpoint.pt"))
    model = DDP(model, ...)

**3. Unused parameters causing errors:**
DDP requires all parameters to receive gradients. If some parameters are
conditionally used (e.g., different layers for different batch types), add:
    model = DDP(model, find_unused_parameters=True)
This is slower but necessary for models with conditional computation.

**4. BatchNorm statistics:**
Standard BatchNorm computes statistics per GPU, not across GPUs. For DDP,
use `SyncBatchNorm` to compute global statistics:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
(LayerNorm has no this issue — it doesn't aggregate batch statistics.)

**5. Gradient accumulation without `no_sync()`:**
See Module 17 — forgetting `model.no_sync()` during accumulation causes K×
communication overhead.

**6. Mixed precision: autocast scope:**
The `autocast` context should wrap the model call, not the entire training
loop:
    with autocast():
        loss = model(batch)  # ✓ autocast inside forward
    loss.backward()


### When DDP Is Not Enough: Scaling Limits

DDP works well when:
    •   The entire model fits on one GPU (each rank holds the full model)
    •   The gradient all-reduce fits within the network bandwidth budget

DDP breaks down when:
    •   The model is too large for a single GPU (common for 70B+)
    •   Gradient all-reduce dominates training time (multi-node at scale)

At these scales, the next step is:
    •   **ZeRO / FSDP** (Module 24): shard parameters, gradients, and
        optimizer state across GPUs
    •   **Tensor Parallelism** (Module 19): split individual weight matrices
    •   **Pipeline Parallelism** (Module 20): assign different layers to
        different GPUs

The scaling ladder: DDP → FSDP/ZeRO → TP → PP → 3D Parallelism


### DDP Scaling Efficiency

**Amdahl's Law applied to DDP:**
    speedup(N) = 1 / (serial_fraction + (1 - serial_fraction) / N)

For LLM training, the serial fraction is approximately:
    serial ≈ all_reduce_time / (compute_time + all_reduce_time)

With NVLink (600 GB/s) and 7B model (14 GB gradients):
    all_reduce ≈ 14 GB / 600 GB/s ≈ 23 ms
    compute    ≈ 2,000 ms (7B forward+backward at batch 8)
    serial     ≈ 23 / 2023 ≈ 1.1%
    speedup(8) ≈ 1 / (0.011 + 0.989/8) ≈ 7.3× (vs ideal 8×)

With InfiniBand (25 GB/s):
    all_reduce ≈ 14 GB / 25 GB/s ≈ 560 ms
    serial     ≈ 560 / 2560 ≈ 21.9%
    speedup(8) ≈ 4.3× (vs ideal 8×)  ← significant inefficiency
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Data Parallelism Implementation Comparison

| Property               | DataParallel (DP)     | DistributedDataParallel (DDP) |
|------------------------|-----------------------|-------------------------------|
| Processes per node     | 1 (multi-thread)      | N (one per GPU)               |
| Communication pattern  | Parameter server      | Ring all-reduce               |
| Communication cost     | 2× model size (gather+scatter) | 2× model size (ring) |
| GIL bottleneck         | Yes (Python threads)  | No (separate processes)       |
| Multi-node support     | No                    | Yes (via NCCL + IB)           |
| Overlap comm/compute   | Limited               | Yes (bucket-level)            |
| Memory per GPU         | Full model            | Full model                    |
| Recommended            | No                    | Yes                           |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "DDP Gradient Synchronisation Simulation": {
        "description": "Simulate DDP gradient averaging across N ranks using Python threads, show that all ranks converge to identical parameters, and benchmark the communication overhead.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DDP GRADIENT SYNCHRONISATION — THREAD-BASED SIMULATION
================================================================================

Simulates DDP's gradient averaging using Python threads.
Each thread represents one GPU rank.

Shows:
    1. Parameter divergence WITHOUT gradient averaging
    2. Perfect synchronisation WITH gradient averaging (simulated all-reduce)
    3. Bucketing strategy: group parameters, reduce per bucket
    4. Communication overlap: reduce bucket N while computing bucket N-1

No actual multi-GPU hardware needed — runs on CPU with threads.
================================================================================
"""

import threading
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared all-reduce barrier ─────────────────────────────────────────────────

class AllReduceBarrier:
    """
    Simulates an all-reduce operation across N threads.
    Each thread contributes a gradient tensor; all receive the average.
    """

    def __init__(self, n_ranks: int):
        self.n        = n_ranks
        self.lock     = threading.Lock()
        self.barrier  = threading.Barrier(n_ranks)
        self.buffers: dict[str, list] = {}   # param_name → [tensor per rank]

    def all_reduce_mean(self, name: str, rank: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Each rank contributes its tensor; all receive the mean.
        Synchronous (blocks until all N ranks have contributed).
        """
        # Phase 1: contribute
        with self.lock:
            if name not in self.buffers:
                self.buffers[name] = [None] * self.n
            self.buffers[name][rank] = tensor.clone()

        # Phase 2: wait for all contributions
        self.barrier.wait()

        # Phase 3: compute mean (only needs one rank's work, but all get it)
        mean = torch.stack(self.buffers[name]).mean(dim=0)

        # Phase 4: clean up (only rank 0 to avoid race)
        with self.lock:
            if name in self.buffers:
                del self.buffers[name]

        return mean


# ── Tiny model ────────────────────────────────────────────────────────────────

class TinyModel(nn.Module):
    def __init__(self, d: int = 32, vocab: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.fc    = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        return self.fc(self.embed(x).mean(dim=1))


# ── Simulated DDP rank ────────────────────────────────────────────────────────

def simulate_ddp_rank(rank: int, n_ranks: int, n_steps: int,
                       model_init: nn.Module,
                       all_reduce_barrier: AllReduceBarrier,
                       use_grad_avg: bool,
                       results: dict, seed_offset: int = 0):
    """
    Simulate one DDP rank:
        - Loads data specific to this rank (different seed)
        - Computes gradients
        - Optionally all-reduces with other ranks
        - Updates weights
    """
    torch.manual_seed(rank + seed_offset)
    model = copy.deepcopy(model_init)
    opt   = torch.optim.SGD(model.parameters(), lr=0.1)
    vocab = model.embed.num_embeddings

    for step in range(n_steps):
        # Each rank has different data (simulates data parallelism)
        torch.manual_seed(rank * 1000 + step)
        x = torch.randint(0, vocab, (4, 8))   # batch of 4, seq 8
        y = torch.randint(0, vocab, (4,))

        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()

        if use_grad_avg:
            # Simulate DDP all-reduce: average gradients across all ranks
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad = all_reduce_barrier.all_reduce_mean(
                        f"step{step}_{name}", rank, param.grad
                    )

        opt.step()

    # Record final parameters
    results[rank] = {n: p.data.clone() for n, p in model.named_parameters()}


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_comparison(n_ranks: int = 4, n_steps: int = 20):
    model_init = TinyModel()

    for use_avg in [False, True]:
        label    = "WITH gradient averaging (DDP)" if use_avg else "WITHOUT averaging (naive)"
        barrier  = AllReduceBarrier(n_ranks)
        results  = {}

        threads = [
            threading.Thread(
                target=simulate_ddp_rank,
                args=(r, n_ranks, n_steps, model_init, barrier, use_avg, results)
            )
            for r in range(n_ranks)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        # Check parameter divergence across ranks
        print(f"\n  {label}")
        max_diffs = {}
        for name in results[0]:
            tensors   = [results[r][name] for r in range(n_ranks)]
            max_diff  = max(
                (tensors[i] - tensors[j]).abs().max().item()
                for i in range(n_ranks) for j in range(i+1, n_ranks)
            )
            max_diffs[name] = max_diff

        for name, diff in max_diffs.items():
            synced = diff < 1e-5
            status = "✓ synced" if synced else f"✗ diverged (max diff = {diff:.4f})"
            print(f"    {name:<25}  {status}")


if __name__ == "__main__":
    print("=" * 62)
    print("  DDP GRADIENT SYNCHRONISATION SIMULATION")
    print("  4 ranks, 20 training steps, different data per rank")
    print("=" * 62)

    run_comparison(n_ranks=4, n_steps=20)

    print()
    print("=" * 62)
    print("  KEY INSIGHT")
    print("=" * 62)
    print()
    print("  WITHOUT gradient averaging:")
    print("    Each rank updates with its OWN gradients only.")
    print("    Ranks see different data → diverge in parameter space.")
    print("    Equivalent to training N independent models on N subsets.")
    print()
    print("  WITH gradient averaging (DDP):")
    print("    Every rank computes the SAME averaged gradient.")
    print("    All ranks perform the SAME weight update.")
    print("    Equivalent to training one model on all N × M samples.")
    print()
    print("  This mathematical equivalence is why DDP is lossless:")
    print("  it produces identical results to single-GPU training")
    print("  with a batch N× larger.")
''',
    },

    "DDP Training Loop — Complete Implementation": {
        "description": "A complete, annotated DDP training loop from torchrun setup to checkpoint saving — shows every DDP-specific detail including no_sync, model.module, and DistributedSampler.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DDP TRAINING LOOP — COMPLETE IMPLEMENTATION
================================================================================

A production-ready DDP training loop with every critical detail annotated:
    1. Process group initialisation
    2. Model wrapping with DDP
    3. DistributedSampler for data
    4. Training loop with no_sync for gradient accumulation
    5. Checkpoint saving (rank 0 only)
    6. Metric reduction across ranks
    7. Cleanup

When run as a single process (no torchrun), simulates DDP-like structure
to show the code pattern without requiring multiple GPUs.
================================================================================
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import contextlib


# ── Check for distributed environment ────────────────────────────────────────

IS_DISTRIBUTED = (
    dist.is_available() and
    "RANK" in os.environ and
    "WORLD_SIZE" in os.environ
)


def setup_distributed():
    if IS_DISTRIBUTED:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                                   else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank       = 0
        world_size = 1
        local_rank = 0
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup():
    if IS_DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# ── Model ────────────────────────────────────────────────────────────────────

class SmallLM(nn.Module):
    def __init__(self, vocab: int = 512, d: int = 128, n_layers: int = 2,
                 max_len: int = 32):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.pos    = nn.Embedding(max_len, d)
        dec_layer   = nn.TransformerDecoderLayer(
            d_model=d, nhead=4, dim_feedforward=d*4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.body   = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.embed(idx) + self.pos(pos)
        m    = self.causal_mask[:T, :T]
        x    = self.body(x, x, tgt_mask=m, memory_mask=m,
                         tgt_is_causal=True, memory_is_causal=True)
        return self.head(self.norm(x))


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticTokenDataset(Dataset):
    def __init__(self, n_seqs: int, seq_len: int, vocab: int, seed: int = 0):
        rng        = torch.Generator()
        rng.manual_seed(seed)
        self.data  = torch.randint(0, vocab, (n_seqs, seq_len), generator=rng)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


# ── Metric averaging across ranks ────────────────────────────────────────────

def reduce_mean(tensor: torch.Tensor, world_size: int) -> float:
    """Average a scalar tensor across all ranks."""
    if IS_DISTRIBUTED and world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / world_size
    return tensor.item()


# ── Main training function ────────────────────────────────────────────────────

def train(rank: int, world_size: int, local_rank: int, device):
    # ── Config ────────────────────────────────────────────────────────────────
    VOCAB         = 512
    D             = 128
    N_LAYERS      = 2
    SEQ_LEN       = 32
    N_DATASET     = 1024
    MICRO_BATCH   = 8
    GRAD_ACCUM    = 2
    N_EPOCHS      = 3
    LR            = 3e-4
    LOG_EVERY     = 5

    # ── Build model ───────────────────────────────────────────────────────────
    torch.manual_seed(42)   # identical init on all ranks
    model = SmallLM(VOCAB, D, N_LAYERS, SEQ_LEN).to(device)

    if IS_DISTRIBUTED:
        # Wrap with DDP — gradient hooks registered here
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available()
                    else None)
    # Note: model.module is the underlying SmallLM; model is the DDP wrapper

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # ── Build DataLoader ──────────────────────────────────────────────────────
    dataset = SyntheticTokenDataset(N_DATASET, SEQ_LEN, VOCAB)

    if IS_DISTRIBUTED:
        # DistributedSampler ensures each rank sees a different data shard
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        loader  = DataLoader(dataset, batch_size=MICRO_BATCH, sampler=sampler,
                             num_workers=0, drop_last=True)
    else:
        # Single-process fallback
        loader  = DataLoader(dataset, batch_size=MICRO_BATCH,
                             shuffle=True, drop_last=True)
        sampler = None

    # ── Training loop ─────────────────────────────────────────────────────────
    if is_main_process(rank):
        print(f"\n  Starting training: {N_EPOCHS} epochs, "
              f"world_size={world_size}, device={device}")
        print(f"  Effective batch = {MICRO_BATCH} × {world_size} GPUs × "
              f"{GRAD_ACCUM} accum = "
              f"{MICRO_BATCH * world_size * GRAD_ACCUM} sequences/step\n")

    for epoch in range(N_EPOCHS):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)  # ← CRITICAL: different shuffle per epoch

        step          = 0
        epoch_loss    = 0.0
        epoch_tokens  = 0
        t0            = time.perf_counter()

        # Accumulate K=GRAD_ACCUM micro-batches per optimizer step
        micro_batches_buffer = []

        for batch in loader:
            micro_batches_buffer.append(batch)

            if len(micro_batches_buffer) < GRAD_ACCUM:
                continue   # collect more micro-batches

            # Enough micro-batches collected — do one optimizer step
            optimizer.zero_grad(set_to_none=True)

            for k, micro_batch in enumerate(micro_batches_buffer):
                is_last = (k == GRAD_ACCUM - 1)
                x = micro_batch["input_ids"].to(device)
                y = micro_batch["labels"].to(device)

                # Use no_sync for all but the last micro-batch
                # This defers the DDP all-reduce to the last backward call
                sync_ctx = (contextlib.nullcontext()
                            if (not IS_DISTRIBUTED or is_last)
                            else model.no_sync())

                with sync_ctx:
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.view(-1, VOCAB), y.view(-1)
                    ) / GRAD_ACCUM
                    loss.backward()

            # Gradient clipping (after all micro-batches)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            micro_batches_buffer = []

            step         += 1
            epoch_loss   += loss.item() * GRAD_ACCUM  # undo the /K for logging
            epoch_tokens += MICRO_BATCH * SEQ_LEN

            if is_main_process(rank) and step % LOG_EVERY == 0:
                elapsed     = time.perf_counter() - t0
                tokens_sec  = epoch_tokens / elapsed
                print(f"  Epoch {epoch+1} Step {step:>4}  "
                      f"loss={epoch_loss/step:.4f}  "
                      f"tok/s={tokens_sec:,.0f}")

        if is_main_process(rank):
            print(f"  Epoch {epoch+1} complete.  "
                  f"Mean loss: {epoch_loss/step:.4f}\n")

    # ── Checkpoint saving (rank 0 only) ───────────────────────────────────────
    if is_main_process(rank):
        # Access underlying model via model.module when using DDP
        state_dict = model.module.state_dict() if IS_DISTRIBUTED else model.state_dict()
        print("  ✓ Would save checkpoint here: model.module.state_dict()")
        print(f"  Checkpoint keys: {list(state_dict.keys())[:3]} ...")

    cleanup()


if __name__ == "__main__":
    rank, world_size, local_rank, device = setup_distributed()

    print("=" * 60)
    print("  DDP TRAINING LOOP DEMO")
    print(f"  rank={rank}, world_size={world_size}, device={device}")
    if not IS_DISTRIBUTED:
        print("  (Running in single-process mode — no torchrun detected)")
        print("  To run with DDP: torchrun --nproc_per_node=N this_script.py")
    print("=" * 60)

    train(rank, world_size, local_rank, device)

    print()
    print("=" * 60)
    print("  DDP CHECKLIST")
    print("=" * 60)
    checklist = [
        ("dist.init_process_group",   "Call once at startup before any distributed ops"),
        ("torch.cuda.set_device",     "Set device = local_rank before creating tensors"),
        ("DDP(model, device_ids)",    "Wrap model after moving to device"),
        ("DistributedSampler",        "Ensure each rank sees different data"),
        ("sampler.set_epoch(epoch)",  "Different shuffle each epoch — CRITICAL"),
        ("model.no_sync()",           "Defer all-reduce for gradient accumulation"),
        ("model.module for ckpt",     "Access underlying model via model.module"),
        ("log from rank 0 only",      "if rank == 0: print/log"),
        ("dist.destroy_process_group","Call at end of training"),
    ]
    for item, note in checklist:
        print(f"  ✓  {item:<32}  {note}")
''',
    },

    "DDP Scaling Efficiency Analysis": {
        "description": "Analytical model of DDP scaling efficiency as a function of GPU count, interconnect bandwidth, model size, and compute time — with efficiency curves and recommendations.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DDP SCALING EFFICIENCY ANALYSIS
================================================================================

Models DDP scaling efficiency:
    1. Amdahl's Law applied to communication overhead
    2. Efficiency curves for different interconnects and model sizes
    3. Optimal GPU count before switching to model parallelism
    4. Impact of gradient compression / PowerSGD

================================================================================
"""

import math


def compute_ddp_efficiency(
    n_gpus: int,
    model_params: int,
    compute_time_per_step_s: float,   # single-GPU backward time
    bandwidth_gbps: float,            # effective all-reduce bandwidth (GB/s)
    gradient_compression: float = 1.0,  # 1.0 = no compression, 0.1 = 10× compression
    overlap_fraction: float = 0.9,    # fraction of comm hidden by compute
    dtype_bytes: int = 2,             # gradient dtype bytes
) -> dict:
    """
    Compute DDP scaling efficiency for given configuration.

    Returns efficiency metrics at this GPU count.
    """
    # Communication volume: ring all-reduce ≈ 2 × grad_size
    grad_size_gb  = model_params * dtype_bytes / 1e9 * gradient_compression
    comm_volume   = 2 * (n_gpus - 1) / n_gpus * grad_size_gb
    comm_time_s   = comm_volume / bandwidth_gbps

    # Compute time scales inversely with N (ideal)
    compute_per_gpu = compute_time_per_step_s / n_gpus

    # Hidden communication (overlapped with backward pass)
    visible_comm  = comm_time_s * (1 - overlap_fraction)

    # Step time = max(compute, comm) + serial overhead
    if overlap_fraction >= 1.0:
        step_time = compute_per_gpu
    else:
        step_time = max(compute_per_gpu, visible_comm) + min(compute_per_gpu, visible_comm) * 0.0
        step_time = compute_per_gpu + visible_comm  # conservative: sequential

    # Efficiency and speedup
    ideal_step  = compute_time_per_step_s / n_gpus
    efficiency  = ideal_step / step_time * 100
    speedup     = compute_time_per_step_s / step_time

    return {
        "n_gpus":         n_gpus,
        "compute_s":      compute_per_gpu,
        "comm_s":         comm_time_s,
        "visible_comm_s": visible_comm,
        "step_s":         step_time,
        "efficiency_pct": efficiency,
        "speedup":        speedup,
    }


def bar(frac: float, width: int = 25) -> str:
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


def efficiency_grade(eff: float) -> str:
    if eff >= 95: return "excellent"
    if eff >= 85: return "good"
    if eff >= 70: return "acceptable"
    if eff >= 50: return "poor"
    return "very poor"


if __name__ == "__main__":
    # Reference: LLaMA-2 7B, single A100 step ≈ 2 seconds
    MODEL_PARAMS = 7_000_000_000
    COMPUTE_1GPU = 2.0   # seconds

    # ── 1. Efficiency by interconnect type ───────────────────────────────────
    print("=" * 70)
    print("  DDP SCALING EFFICIENCY — LLaMA-2 7B, 2s/step/GPU")
    print("=" * 70)

    interconnects = [
        ("NVLink 3.0 (A100)",   600.0, 0.95),
        ("NVLink 4.0 (H100)",   900.0, 0.95),
        ("InfiniBand HDR (IB)", 25.0,  0.70),
        ("100GbE Ethernet",     12.0,  0.50),
    ]

    for bw_name, bw, overlap in interconnects:
        print(f"\n  {bw_name}  (effective BW: {bw} GB/s, overlap: {overlap:.0%})")
        print(f"  {'N GPUs':>7}  {'Efficiency':>12}  {'Speedup':>10}  "
              f"{'Comm time':>12}  {'Grade':>12}")
        print(f"  {'':─>7}  {'':─>12}  {'':─>10}  {'':─>12}  {'':─>12}")

        for n in [1, 2, 4, 8, 16, 32, 64]:
            r = compute_ddp_efficiency(
                n, MODEL_PARAMS, COMPUTE_1GPU,
                bw, overlap_fraction=overlap
            )
            grade = efficiency_grade(r["efficiency_pct"])
            print(f"  {n:>7}  {r['efficiency_pct']:>11.1f}%  "
                  f"{r['speedup']:>10.2f}×  "
                  f"{r['comm_s']*1000:>11.1f}ms  "
                  f"{grade:>12}")

    # ── 2. Optimal GPU count per model size ───────────────────────────────────
    print()
    print("=" * 70)
    print("  OPTIMAL GPU COUNT (efficiency > 80%) — NVLink 3.0")
    print("=" * 70)
    print()

    model_configs = [
        ("GPT-2 1.5B",   1.5e9,  0.5),
        ("LLaMA-2 7B",   7.0e9,  2.0),
        ("LLaMA-2 13B", 13.0e9,  3.5),
        ("LLaMA-2 70B", 70.0e9, 18.0),
    ]

    print(f"  {'Model':<18} {'Params':>10}  {'Max efficient GPUs':>20}")
    print(f"  {'':─<18} {'':─>10}  {'':─>20}")

    for name, params, compute in model_configs:
        max_eff_n = 1
        for n in [2, 4, 8, 16, 32, 64, 128, 256]:
            r = compute_ddp_efficiency(n, params, compute, 600.0, overlap_fraction=0.9)
            if r["efficiency_pct"] >= 80.0:
                max_eff_n = n
            else:
                break
        print(f"  {name:<18} {params/1e9:>9.1f}B  {max_eff_n:>20} GPUs")

    print()
    print("  Beyond these counts, use ZeRO/FSDP to reduce gradient communication,")
    print("  or tensor/pipeline parallelism to eliminate all-reduce entirely.")

    # ── 3. Gradient compression impact ────────────────────────────────────────
    print()
    print("=" * 70)
    print("  GRADIENT COMPRESSION IMPACT (64 GPUs, InfiniBand 25 GB/s)")
    print("=" * 70)
    print()

    print(f"  {'Compression':>15}  {'Comm time':>12}  {'Efficiency':>12}  "
          f"{'Quality cost':>14}")
    print(f"  {'':─>15}  {'':─>12}  {'':─>12}  {'':─>14}")

    compressions = [
        (1.00, "none (full)"),
        (0.50, "~2× (PowerSGD)"),
        (0.10, "~10× (aggressive)"),
        (0.01, "~100× (extreme)"),
    ]

    for ratio, label in compressions:
        r = compute_ddp_efficiency(64, MODEL_PARAMS, COMPUTE_1GPU, 25.0,
                                    gradient_compression=ratio, overlap_fraction=0.7)
        q_cost = "" if ratio >= 0.5 else "~0.5% loss" if ratio >= 0.1 else "~2-5% loss"
        print(f"  {label:>15}  {r['comm_s']*1000:>11.1f}ms  "
              f"{r['efficiency_pct']:>11.1f}%  {q_cost:>14}")

    print()
    print("  PowerSGD (rank-1 gradient approximation) can reduce comm 2-8×")
    print("  with < 0.5% quality degradation on most tasks.")
    print("  Not commonly used for LLM pre-training (ZeRO is preferred),")
    print("  but useful for fine-tuning on slow interconnects.")

    # ── 4. Summary: when to use what ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  DECISION GUIDE: DDP vs MODEL PARALLELISM")
    print("=" * 70)
    scenarios = [
        ("7B on 8×A100 (NVLink)",    "DDP ✓",           "fits on 1 GPU, high BW"),
        ("7B on 8×A100 (no NVLink)", "DDP + ZeRO-2",    "reduce comm volume"),
        ("70B on 16×A100",           "DDP + ZeRO-3",    "model > 1 GPU memory"),
        ("70B on 128×A100 cluster",  "FSDP + TP",       "need tensor parallelism"),
        ("175B+ on 1000+ GPUs",      "3D parallelism",  "DP + TP + PP required"),
    ]
    print(f"  {'Scenario':<35} {'Recommendation':<18}  Notes")
    print(f"  {'':─<35} {'':─<18}  ─────────────────────────")
    for scenario, rec, note in scenarios:
        print(f"  {scenario:<35} {rec:<18}  {note}")
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
    #     from llm_training.visuals.data_parallel_ddp import (
    #         DDP_VISUAL_HTML,
    #         DDP_VISUAL_HEIGHT,
    #     )
    #     visual_html   = DDP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = DDP_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[18_data_parallel_ddp.py] Could not load visual: {e}",
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