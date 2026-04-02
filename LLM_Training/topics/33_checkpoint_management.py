"""
Checkpoint Management
======================

Checkpointing is the safety net of large-scale training. A 70B model trained
for three months on 512 GPUs can lose weeks of work to a hardware failure, a
software bug, or even a poorly-timed cloud spot instance preemption. Robust
checkpoint management encompasses not just saving weights, but saving the
entire resumable training state — optimizer moments, learning rate schedules,
random number generator states, data loader positions — and doing so
efficiently enough that checkpointing overhead does not dominate training time.
At scale, checkpointing interacts with every distributed training primitive:
DDP, FSDP, tensor parallelism, and pipeline parallelism each require
different strategies to correctly save and restore state.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Checkpoint Management"
DISPLAY_NAME = "33 · Checkpoint Management"
ICON         = "💾"
SUBTITLE     = "Saving, Resuming, and Sharding Large Model State"


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

### What Must Be Saved for a Complete Checkpoint

A "complete checkpoint" is one from which training can resume and produce
results identical to an uninterrupted run. This requires saving more than
just model weights:

    Component                  Format       Size (7B fp16)  Required for exact resume?
    ─────────────────────────────────────────────────────────────────────────────────
    Model parameters           fp16/bf16    14 GB           Yes (obviously)
    Optimizer first moment m   fp32         28 GB           Yes (affects next update)
    Optimizer second moment v  fp32         28 GB           Yes (affects next update)
    FP32 master weights        fp32         28 GB           Yes (for optimizer step)
    LR scheduler state         dict         <1 KB           Yes (current LR)
    Random number states       tuple        <1 KB           Yes (reproducibility)
    Data loader position       int          <1 KB           Yes (skip already-seen data)
    Global step counter        int          <1 KB           Yes (logging, eval timing)
    AMP GradScaler state       dict         <1 KB           Yes (for fp16 training)
    ─────────────────────────────────────────────────────────────────────────────────
    TOTAL                                   ~99 GB          Full training state

**The "weights only" checkpoint** (14 GB) is sufficient for inference but NOT
for resuming training. Resuming from weights only causes two problems:
    1.  Optimizer moments are re-initialised to zero → training restarts from
        a cold start even if you're at step 50,000. This causes a loss spike.
    2.  LR schedule position is lost → you may restart with a very high LR
        that was supposed to be in its decay phase, causing divergence.

**Why the optimizer state is so large:**
AdamW stores two fp32 tensors per parameter — the first moment (m) and second
moment (v). Since these must be fp32 for numerical precision:
    optimizer_state_bytes = 2 × 4 bytes × P parameters
    For 7B model: 2 × 4 × 7B = 56 GB

Combined with fp32 master weights (another 28 GB), the full checkpoint is
14 + 28 + 56 = 98 GB — 7× larger than the model weights alone.


### The Random Number State Problem

One often-overlooked requirement for exact training reproducibility: the
random number generator (RNG) state must be checkpointed. This includes:

    1.  **Python random state:** `random.getstate()`
    2.  **NumPy random state:** `np.random.get_state()`
    3.  **PyTorch CPU state:** `torch.get_rng_state()`
    4.  **PyTorch GPU states:** `torch.cuda.get_rng_state(device)` for each GPU

Why? Operations like:
    •   Dropout masks (different each forward pass)
    •   Data augmentation (random crops, flips)
    •   Data shuffling (DistributedSampler's per-epoch shuffle)

All depend on the RNG state. Without restoring it, resumed training produces
different dropout patterns and different data order — the model sees "different"
training from what it would have seen without the interruption.

For research reproducibility: always checkpoint RNG states.
For production training: at minimum, checkpoint the data loader position so
the model does not re-see training examples after a restart.


### Checkpoint Frequency: The Cost-Safety Trade-off

Every checkpoint writes hundreds of gigabytes to storage. With multiple GPUs
and a distributed file system, this has non-trivial cost:

    Checkpoint write time ≈ checkpoint_size_GB / storage_bandwidth_GB_s
    For a 98 GB checkpoint to NFS at 5 GB/s: ~20 seconds per checkpoint

    If checkpointing every 100 steps at 30 seconds per step:
        overhead = 20s / (100 × 30s) ≈ 0.67%   (negligible)

    If checkpointing every 10 steps:
        overhead = 20s / (10 × 30s) ≈ 6.7%   (significant!)

**Typical checkpointing strategies:**
    •   Every N steps (e.g., every 500 or 1000 steps)
    •   Every M hours of wall-clock time (e.g., every 2 hours)
    •   Before any risky operation (changing LR, code update)
    •   On detected instability (loss spike detection → save + rollback)

**The rolling checkpoint pattern:**
Keep only the last K checkpoints (e.g., K=3) to limit storage usage.
The delete-oldest policy ensures you can always go back up to K checkpoints:
    After step 1000: save ckpt-1000, delete ckpt-700
    After step 1300: save ckpt-1300, delete ckpt-1000
    ...

**The milestone checkpoint pattern:**
Keep all checkpoints at milestone steps (1M, 5M, 10M, 50M, 100M tokens),
plus the rolling last K. This allows studying training dynamics retrospectively.


    **Diagram 1 — Checkpoint Storage Organisation:**

    CHECKPOINT DIRECTORY STRUCTURE
    ════════════════════════════════════════════════════════════════

    checkpoints/
    ├── step-001000/
    │   ├── model.pt          ← model state_dict (fp16)
    │   ├── optimizer.pt      ← optimizer state + master weights (fp32)
    │   ├── scheduler.pt      ← LR scheduler state
    │   ├── rng_states.pt     ← CPU + GPU RNG states
    │   ├── dataloader.pt     ← data loader position/seed
    │   └── metadata.json     ← step, tokens, loss, timestamp, git hash
    ├── step-002000/ ...
    ├── step-003000/ ...
    ├── latest → step-003000  ← symlink to most recent
    └── milestone-010000000/  ← permanent checkpoint at 10M tokens


### Asynchronous Checkpointing: Eliminating Training Stalls

The naive approach stalls training while writing the checkpoint:
    1.  Training halts (barrier synchronisation across all GPUs)
    2.  CPU threads write checkpoint to storage
    3.  Training resumes

At 20 seconds per checkpoint every 500 steps, this is a 0.67% overhead —
tolerable. But at scale (LLaMA-2 scale, 2048 GPUs, full checkpoint is TBs),
checkpoint writes can take minutes, stalling thousands of GPUs.

**Asynchronous checkpointing:**
    1.  At checkpoint step: copy model state to CPU RAM (fast: GPU→CPU at PCIe speed)
    2.  Resume training immediately
    3.  Background threads write CPU RAM copy to storage (no GPU stall)

The GPU-to-CPU copy takes a few seconds (at 32 GB/s PCIe, 98 GB ≈ 3 seconds).
Storage write happens asynchronously without blocking compute.

**Implementation sketch:**
    import threading

    def async_save(state_dict: dict, path: str):
        # Copy to CPU (main thread, fast)
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        # Write to storage (background thread, slow but non-blocking)
        def write():
            torch.save(cpu_state, path)
        threading.Thread(target=write, daemon=True).start()

This requires that the next checkpoint is not triggered before the previous
write completes. In practice, checkpointing every 500+ steps gives more than
enough time.

**Torch Distributed Checkpoint (DCP):** PyTorch's native async checkpointing
library. Uses multiple writer threads (one per shard) and can write to
distributed storage (S3, GCS, distributed filesystems) natively.


### Sharded Checkpoints: DDP, FSDP, and Tensor Parallelism

With distributed training, the model state is distributed across GPUs. This
creates a fundamental question: how should checkpoints be saved and loaded?

**Strategy 1 — Gather then save (single file):**
    •   All-gather model weights to rank 0 (or one designated saver)
    •   Rank 0 saves a single full model checkpoint
    •   All other ranks save nothing (no parallelism for saving)
    •   Load: rank 0 loads and broadcasts/scatters to all ranks

    Pros:  Simple; compatible with any downstream use (inference, fine-tuning)
    Cons:  Rank 0 needs enough memory to hold the full model; serial save

    Used by: DDP (Module 18), small-scale distributed training

**Strategy 2 — Save shards independently (sharded checkpoint):**
    •   Each rank saves its own shard of the model state
    •   Creates N checkpoint files for N ranks
    •   Load: each rank loads its own shard (preserves the distributed structure)

    Pros:  No memory overhead on any single rank; fully parallel save
    Cons:  Checkpoint is only usable with the same parallelism configuration;
           must re-shard if topology changes (N GPUs → M GPUs)

    Used by: FSDP (Module 24), large-scale training

**Strategy 3 — Torch Distributed Checkpoint (DCP) with resharding:**
    PyTorch's DCP allows saving sharded checkpoints but loading with a
    different topology. It stores metadata about how shards map to model
    parameters, enabling resharding at load time.
    Used by: LLaMA and modern PyTorch distributed workflows.


    **Diagram 2 — Sharded vs Consolidated Checkpoints:**

    SHARDED CHECKPOINT (4 GPUs, FSDP):
    ════════════════════════════════════════════════════════════════
    GPU 0: saves shard_0.pt  (contains W[0:P/4])
    GPU 1: saves shard_1.pt  (contains W[P/4:P/2])
    GPU 2: saves shard_2.pt  (contains W[P/2:3P/4])
    GPU 3: saves shard_3.pt  (contains W[3P/4:P])

    Load (same topology):
    GPU 0 reads shard_0.pt  → directly usable ✓

    Load (different topology, 8 GPUs):
    Old shards need resharding: shard_0.pt → new_shard_0.pt + new_shard_1.pt
    DCP handles this automatically ✓

    CONSOLIDATED CHECKPOINT (rank 0 gathers all):
    ════════════════════════════════════════════════════════════════
    GPU 1,2,3: all-gather to GPU 0
    GPU 0: saves model.pt  (full 140 GB for 70B model!)
    GPU 0 needs 140 GB free memory ← OOM for large models!

    → Use sharded checkpoints for models that don't fit on one GPU.


### Safe Tensors Format: The Modern Standard

PyTorch's default checkpoint format (`.pt` / `.pth`) uses Python's `pickle`
module. This has two significant problems:

    1.  **Security:** Loading a `.pt` file executes arbitrary Python code
        (pickle can call `__reduce__` methods that run shell commands).
        Loading untrusted checkpoints is a security vulnerability.

    2.  **Portability:** Pickle format is tied to the Python object structure
        that was used to create it. If the model class changes (even a small
        refactor), old checkpoints may fail to load.

**SafeTensors** (Hugging Face, 2023) solves both problems:
    •   Pure tensor storage: no Python objects, no arbitrary code execution
    •   Memory-mapped loading: tensors can be loaded lazily from disk
        without reading the entire file into RAM
    •   Language-agnostic: can be read from Rust, C++, Python, JS
    •   Fast: memory-mapped loading is ~3× faster than torch.load() for large files

Format:
    •   Header: JSON metadata (tensor names, shapes, dtypes, offsets)
    •   Data: raw bytes of each tensor (no compression by default)

Usage:
    from safetensors.torch import save_file, load_file
    save_file(model.state_dict(), "model.safetensors")
    state_dict = load_file("model.safetensors")

SafeTensors is now the standard format for all Hugging Face model repositories
and is supported by vLLM, llama.cpp, and all modern inference frameworks.


### Checkpoint Metadata and Lineage Tracking

A checkpoint without metadata is nearly useless for large-scale training. At
minimum, each checkpoint should record:

    {
        "step":           50000,
        "tokens_seen":    26_214_400_000,   # ~26B tokens
        "train_loss":     2.143,
        "val_loss":       2.287,
        "learning_rate":  1.2e-5,
        "wall_clock_s":   432000,            # 5 days
        "git_hash":       "a3f4b2c",         # exact code version
        "config_hash":    "e7d1a09",         # hash of hyperparameter config
        "hardware":       "8×A100 80GB",
        "parallelism":    {"dp": 4, "tp": 1, "pp": 1},
        "framework":      "torch 2.1.0",
        "parent_ckpt":    "step-049000",     # for lineage tracking
    }

**Why git hash matters:**
If a training run produces unexpected results, the git hash tells you exactly
which version of the code was running. Combined with the config hash, you can
exactly reproduce any checkpoint.

**Training lineage:**
Record the parent checkpoint path in each checkpoint's metadata. This creates
a chain that shows exactly how the model evolved:
    base_pretrained → SFT_step_1000 → SFT_step_2000 → RLHF_step_500 → ...


### Checkpoint Rollback and Recovery

When training diverges (loss spike, NaN gradients, sudden quality degradation),
the recovery procedure is:

    1.  Detect the anomaly (automated monitoring or human observation)
    2.  Stop training immediately
    3.  Load the most recent healthy checkpoint
    4.  Diagnose the cause (usually: LR too high, bad data batch, hardware error)
    5.  Fix the issue (skip the problematic data, reduce LR, replace bad GPU)
    6.  Resume from the healthy checkpoint

**Automated spike detection and rollback:**
    if loss > 3 × rolling_mean_loss:
        save_emergency_checkpoint()
        load_checkpoint(last_healthy)
        skip_current_batch = True
        lr = lr × 0.9   # reduce LR after spike

**The "zombie training" failure mode:**
Training appears to continue normally, but loss is not decreasing.
This is harder to detect than a spike. Symptoms:
    •   Loss curve has plateaued for 10k+ steps
    •   Gradient norms are very small (model is at a local minimum)
    •   Evaluation metrics haven't improved

Recovery: load an earlier checkpoint (before the plateau began) and try
different hyperparameters (higher LR, different scheduler, different data mix).


### FSDP Checkpoint: Practical Details

PyTorch FSDP has specific requirements for checkpointing that differ from DDP.

**The FSDP full_state_dict issue:**
By default, FSDP keeps parameters sharded across GPUs. Calling `model.state_dict()`
returns only the local shard — the full model cannot be reconstructed from one rank.

**Solution 1: FULL_STATE_DICT with offloading**
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        if rank == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, "model.pt")

    This all-gathers on CPU and saves from rank 0. Works but is slow for large models.

**Solution 2: SHARDED_STATE_DICT (recommended)**
    from torch.distributed.fsdp import ShardedStateDictConfig

    with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True)
    ):
        state_dict = model.state_dict()
        # Each rank saves its own shard
        torch.save(state_dict, f"shard_{rank}.pt")

    Sharded saves are fast and memory-efficient but require DCP-style loading
    when resuming with a different number of GPUs.

**Torch Distributed Checkpoint (DCP) — the modern approach:**
    import torch.distributed.checkpoint as DCP

    # Save (all ranks participate, each saves its own state)
    DCP.save({"model": model, "optimizer": optimizer}, checkpoint_id="./ckpt/")

    # Load (automatically reshards to match current topology)
    DCP.load({"model": model, "optimizer": optimizer}, checkpoint_id="./ckpt/")

DCP handles resharding transparently — you can save with 4 GPUs and load
with 8 GPUs, and DCP will correctly remap the shards.


### Cloud Storage and Checkpoint Streaming

Production training runs on cloud infrastructure (AWS, GCP, Azure) where
local NVMe is scarce but object storage (S3, GCS) is effectively unlimited.

**Checkpoint streaming to S3:**
    import boto3
    from io import BytesIO

    def save_to_s3(state_dict, bucket, key):
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        boto3.client("s3").upload_fileobj(buffer, bucket, key)

**The write bandwidth challenge:**
A 99 GB checkpoint at 5 GB/s upload to S3 = ~20 seconds.
With 2048 GPUs all writing simultaneously to S3: massive I/O contention.

Solution: only rank 0 (or shard leaders in FSDP) write to S3. All other ranks
write nothing. This avoids S3 rate limiting and reduces egress costs.

**Checkpoint caching:**
Load from S3 once; cache in local NVMe for fast recovery:
    if not os.path.exists("/nvme/ckpt/model.pt"):
        download_from_s3("s3://bucket/ckpt/model.pt", "/nvme/ckpt/model.pt")
    load_checkpoint("/nvme/ckpt/model.pt")


### Tensor Parallelism and Pipeline Parallelism Checkpoints

With 3D parallelism (TP+PP+DP), each "model rank" (unique TP+PP combination)
holds a different slice of the model:

    TP rank t, PP stage p has parameters W[t/TP × d_out, p × layers/PP : ...]

**Saving with 3D parallelism (Megatron-style):**
    •   Each (TP, PP) pair saves its own checkpoint shard
    •   Total shards = TP × PP (one per model rank, not per DP rank)
    •   DP replicas are identical → only save from DP rank 0 per model rank

    For TP=8, PP=4, DP=16: save 8 × 4 = 32 checkpoint shards
    (Not 512 = 8 × 4 × 16 — DP replicas are identical, only save one)

**The shard naming convention (Megatron):**
    mp_rank_{tp_rank:02d}_{pp_rank:03d}/model_optim_rng.pt

**Resharding checkpoints when topology changes:**
Going from TP=4, PP=2 to TP=8, PP=1 (same total GPUs, different topology):
    •   Load old shards
    •   Concatenate/split along the appropriate tensor dimension
    •   Save in new topology
    Megatron provides `tools/convert_megatron_checkpoint.py` for this.

This is one of the most error-prone operations in large-scale training —
always verify the quality of the re-sharded checkpoint with a few validation
steps before committing to a long training run.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Checkpoint Strategies by Training Setup

| Setup                | Save strategy             | Load strategy          | Memory overhead | Reshardable? |
|----------------------|---------------------------|------------------------|-----------------|--------------|
| Single GPU / DDP     | torch.save full model     | torch.load             | 1× model        | N/A          |
| FSDP (same topology) | SHARDED_STATE_DICT        | DCP.load               | 1/N per rank    | With DCP     |
| FSDP (any topology)  | Torch DCP                 | DCP.load (auto-reshard)| 1/N per rank    | Yes          |
| Megatron 3D          | Per-MP-rank shards        | Load matching topology | 1/(TP×PP)       | With tools   |
| DeepSpeed            | ds_checkpoint.save        | ds_checkpoint.load     | 1/N per rank    | Limited      |
| Inference only       | SafeTensors (rank 0 only) | load_file              | Full model      | N/A          |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Complete Checkpoint Manager": {
        "description": "A production-ready checkpoint manager that saves and loads the complete training state — model, optimizer, scheduler, RNG states, data loader position, and metadata — with async writing and rolling deletion.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
COMPLETE CHECKPOINT MANAGER
================================================================================

A production-ready checkpoint manager that handles:
    1. Complete training state (model + optimizer + scheduler + RNG + data)
    2. Metadata logging (step, loss, tokens, git hash, timestamp)
    3. Asynchronous writing to avoid training stalls
    4. Rolling checkpoint deletion (keep last K)
    5. Milestone checkpoints (permanent at key steps)
    6. Resumption with exact state restoration
    7. Verification that resumed state is correct

================================================================================
"""

import os
import json
import time
import shutil
import hashlib
import threading
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Metadata dataclass ────────────────────────────────────────────────────────

@dataclass
class CheckpointMetadata:
    """Complete metadata for one checkpoint."""
    step:             int
    global_step:      int
    tokens_seen:      int
    train_loss:       float
    val_loss:         Optional[float]
    learning_rate:    float
    wall_clock_s:     float
    timestamp:        str
    framework_ver:    str
    config_hash:      str    # hash of training config
    parent_ckpt:      Optional[str]  # path to previous checkpoint
    git_hash:         str = "unknown"
    hardware:         str = "unknown"
    extra:            dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CheckpointMetadata":
        return cls(**json.loads(json_str))


def get_config_hash(config: dict) -> str:
    """Deterministic hash of a configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def get_git_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


# ── RNG state management ──────────────────────────────────────────────────────

def save_rng_states() -> dict:
    """
    Save all RNG states needed for exact reproducibility.
    Includes Python, NumPy, PyTorch CPU, and PyTorch GPU states.
    """
    import random
    import numpy as np

    states = {
        "python":       random.getstate(),
        "numpy":        np.random.get_state(),
        "torch_cpu":    torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = {
            i: torch.cuda.get_rng_state(i)
            for i in range(torch.cuda.device_count())
        }
    return states


def restore_rng_states(states: dict) -> None:
    """Restore all RNG states from a saved state dict."""
    import random
    import numpy as np

    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        for device_id, rng_state in states["torch_cuda"].items():
            torch.cuda.set_rng_state(rng_state, int(device_id))


# ── Checkpoint manager ────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Production checkpoint manager.

    Features:
        - Saves model, optimizer, scheduler, RNG, data loader state
        - Asynchronous writing (doesn't stall training)
        - Rolling deletion (keep last K checkpoints)
        - Milestone checkpoints (permanent at configurable steps)
        - Metadata logging with training lineage
        - Atomic save (write to temp dir, then rename)
    """

    def __init__(self,
                 checkpoint_dir: str,
                 keep_last_k: int = 3,
                 milestone_steps: list = None,
                 async_save: bool = True,
                 config: dict = None):
        self.ckpt_dir        = Path(checkpoint_dir)
        self.keep_last_k     = keep_last_k
        self.milestone_steps = set(milestone_steps or [])
        self.async_save      = async_save
        self.config_hash     = get_config_hash(config or {})
        self.git_hash        = get_git_hash()
        self.start_time      = time.time()
        self.history: list[str] = []      # list of checkpoint paths saved so far
        self._write_thread: Optional[threading.Thread] = None

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _wait_for_write(self):
        """Wait for any pending async write to complete."""
        if self._write_thread is not None and self._write_thread.is_alive():
            self._write_thread.join()

    def _atomic_write(self, state: dict, ckpt_path: Path):
        """
        Write checkpoint atomically: write to temp dir, then rename.
        This prevents incomplete checkpoints if interrupted mid-write.
        """
        # Write to temporary path first
        tmp_path = ckpt_path.parent / f"_tmp_{ckpt_path.name}"
        tmp_path.mkdir(parents=True, exist_ok=True)

        for key, value in state.items():
            if isinstance(value, str):
                # JSON files
                (tmp_path / key).write_text(value)
            else:
                # Tensor/dict files
                torch.save(value, tmp_path / key)

        # Atomic rename (on POSIX systems)
        if ckpt_path.exists():
            shutil.rmtree(ckpt_path)
        tmp_path.rename(ckpt_path)

    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             scheduler: Any,
             step: int,
             tokens_seen: int,
             train_loss: float,
             val_loss: Optional[float] = None,
             dataloader_state: Optional[dict] = None,
             extra_state: Optional[dict] = None):
        """
        Save a complete training checkpoint.

        If async_save=True: copy state to CPU immediately (fast),
        then write to disk in a background thread (non-blocking).
        """
        self._wait_for_write()   # ensure previous async write is done

        ckpt_name = f"step-{step:07d}"
        ckpt_path = self.ckpt_dir / ckpt_name

        # ── Collect state (on CPU for portability) ────────────────────────────
        # Model state
        model_state  = {k: v.cpu().clone()
                         for k, v in model.state_dict().items()}

        # Optimizer state
        opt_state = optimizer.state_dict()

        # Scheduler state
        sched_state = scheduler.state_dict() if hasattr(scheduler, "state_dict") else {}

        # RNG states (for exact reproducibility)
        rng_states = save_rng_states()

        # Metadata
        meta = CheckpointMetadata(
            step          = step,
            global_step   = step,
            tokens_seen   = tokens_seen,
            train_loss    = train_loss,
            val_loss      = val_loss,
            learning_rate = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0,
            wall_clock_s  = time.time() - self.start_time,
            timestamp     = time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            framework_ver = f"torch {torch.__version__}",
            config_hash   = self.config_hash,
            git_hash      = self.git_hash,
            parent_ckpt   = self.history[-1] if self.history else None,
        )

        state = {
            "model.pt":       model_state,
            "optimizer.pt":   opt_state,
            "scheduler.pt":   sched_state,
            "rng_states.pt":  rng_states,
            "metadata.json":  meta.to_json(),
        }

        if dataloader_state:
            state["dataloader.pt"] = dataloader_state
        if extra_state:
            state["extra.pt"] = extra_state

        # ── Write checkpoint ───────────────────────────────────────────────────
        def do_write():
            t0 = time.perf_counter()
            self._atomic_write(state, ckpt_path)
            write_time = time.perf_counter() - t0

            # Update latest symlink
            latest = self.ckpt_dir / "latest"
            if latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ckpt_name)

            # Size report
            total_bytes = sum(
                f.stat().st_size
                for f in ckpt_path.rglob("*") if f.is_file()
            )
            print(f"  [Checkpoint] Saved step {step} to {ckpt_path.name} "
                  f"({total_bytes/1e9:.2f} GB in {write_time:.1f}s)")

        if self.async_save:
            self._write_thread = threading.Thread(target=do_write, daemon=True)
            self._write_thread.start()
        else:
            do_write()

        # Update history
        self.history.append(str(ckpt_path))

        # Rolling deletion: remove old checkpoints (unless milestone)
        if len(self.history) > self.keep_last_k:
            to_delete = self.history[-(self.keep_last_k + 1)]
            step_to_delete = int(Path(to_delete).name.replace("step-", ""))
            if step_to_delete not in self.milestone_steps:
                if Path(to_delete).exists():
                    shutil.rmtree(to_delete)
                    print(f"  [Checkpoint] Deleted old checkpoint: {Path(to_delete).name}")

        return ckpt_path

    def load(self,
             checkpoint_path: str,
             model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler=None,
             restore_rng: bool = True,
             map_location: str = "cpu") -> CheckpointMetadata:
        """
        Load a complete training checkpoint.

        Returns the metadata for the loaded checkpoint.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            # Try resolving as symlink
            ckpt_path = self.ckpt_dir / checkpoint_path

        print(f"  [Checkpoint] Loading from {ckpt_path.name}...")
        t0 = time.perf_counter()

        # Load model
        model_state = torch.load(ckpt_path / "model.pt", map_location=map_location)
        model.load_state_dict(model_state)

        # Load optimizer
        if optimizer is not None and (ckpt_path / "optimizer.pt").exists():
            opt_state = torch.load(ckpt_path / "optimizer.pt", map_location=map_location)
            optimizer.load_state_dict(opt_state)

        # Load scheduler
        if scheduler is not None and (ckpt_path / "scheduler.pt").exists():
            sched_state = torch.load(ckpt_path / "scheduler.pt", map_location=map_location)
            if sched_state:
                scheduler.load_state_dict(sched_state)

        # Restore RNG states
        if restore_rng and (ckpt_path / "rng_states.pt").exists():
            rng_states = torch.load(ckpt_path / "rng_states.pt")
            restore_rng_states(rng_states)

        # Load metadata
        meta_json = (ckpt_path / "metadata.json").read_text()
        meta      = CheckpointMetadata.from_json(meta_json)

        load_time = time.perf_counter() - t0
        print(f"  [Checkpoint] Loaded step {meta.step} "
              f"(loss={meta.train_loss:.4f}, "
              f"tokens={meta.tokens_seen:,}) in {load_time:.1f}s")

        return meta

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints with their metadata."""
        checkpoints = []
        for path in sorted(self.ckpt_dir.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            meta_file = path / "metadata.json"
            if meta_file.exists():
                meta = CheckpointMetadata.from_json(meta_file.read_text())
                checkpoints.append({
                    "path":       str(path),
                    "name":       path.name,
                    "step":       meta.step,
                    "tokens":     meta.tokens_seen,
                    "train_loss": meta.train_loss,
                    "timestamp":  meta.timestamp,
                })
        return sorted(checkpoints, key=lambda x: x["step"])


# ── Tiny model and training loop ──────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=512, d=64, n_layers=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d),
                          nn.Linear(d, d*2, bias=False), nn.GELU(),
                          nn.Linear(d*2, d, bias=False))
            for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers: h = h + l(h)
        return self.head(self.norm(h))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    torch.manual_seed(42)
    VOCAB, D = 512, 64
    BATCH, T = 4, 16

    config = {"vocab": VOCAB, "d": D, "lr": 3e-4, "batch": BATCH}

    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 65)
        print("  CHECKPOINT MANAGER DEMO")
        print(f"  Checkpoint dir: {tmpdir[:30]}...")
        print("=" * 65)
        print()

        model   = TinyLM(VOCAB, D)
        opt     = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

        manager = CheckpointManager(
            checkpoint_dir  = tmpdir,
            keep_last_k     = 2,
            milestone_steps = {500},
            async_save      = False,   # synchronous for demo
            config          = config,
        )

        # ── Phase 1: Training steps 1-3 ───────────────────────────────────────
        print("  Phase 1: Training steps 1-3...")
        step         = 0
        tokens_seen  = 0

        for step in [100, 200, 300]:
            x = torch.randint(0, VOCAB, (BATCH, T))
            y = torch.randint(0, VOCAB, (BATCH, T))
            opt.zero_grad()
            loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
            loss.backward(); opt.step(); sched.step()
            tokens_seen += BATCH * T

            manager.save(model, opt, sched,
                          step=step, tokens_seen=tokens_seen,
                          train_loss=loss.item(), val_loss=loss.item() * 0.95)

        print()

        # Show checkpoints
        ckpts = manager.list_checkpoints()
        print("  Available checkpoints:")
        for ck in ckpts:
            print(f"  {'→' if ck == ckpts[-1] else ' '} "
                  f"{ck['name']}: step={ck['step']}, "
                  f"loss={ck['train_loss']:.4f}")
        print()

        # ── Phase 2: Simulate crash and recovery ──────────────────────────────
        print("  Simulating crash — model state corrupted...")
        # Corrupt model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float("nan"))

        print(f"  Model output after corruption: "
              f"{model(torch.randint(0, VOCAB, (1, 4)))[0, 0, :3].tolist()}")

        # Reload from last checkpoint
        print()
        print("  Recovering from last checkpoint...")
        last_ckpt = ckpts[-1]["path"]
        meta = manager.load(last_ckpt, model, opt, sched, restore_rng=True)

        # Verify recovery
        with torch.no_grad():
            test_out = model(torch.randint(0, VOCAB, (1, 4)))[0, 0, :3]
        print(f"  Model output after recovery: {test_out.tolist()}")
        recovered = not any(x != x for x in test_out.tolist())  # check for NaN
        print(f"  Recovery successful: {'✓' if recovered else '✗'}")
        print(f"  Resuming from step: {meta.step}, loss: {meta.train_loss:.4f}")
        print()

        # ── Phase 3: Verify RNG reproducibility ───────────────────────────────
        print("  Verifying RNG state restoration (exact reproducibility)...")
        import random
        torch.manual_seed(99)
        random.seed(99)

        # Save state mid-training
        ckpt3 = manager.save(model, opt, sched, step=400, tokens_seen=tokens_seen,
                               train_loss=0.9)
        manager._wait_for_write()

        # Generate some samples
        samples_original = [torch.randint(0, VOCAB, (3,)).tolist() for _ in range(3)]

        # Restore and regenerate
        manager.load(str(ckpt3), model, restore_rng=True)
        samples_restored = [torch.randint(0, VOCAB, (3,)).tolist() for _ in range(3)]

        match = samples_original == samples_restored
        print(f"  Original samples:  {samples_original}")
        print(f"  Restored samples:  {samples_restored}")
        print(f"  RNG states match:  {'✓' if match else '✗'}")
        print()

        # ── Summary ───────────────────────────────────────────────────────────
        print("  Checkpoint directory contents:")
        for path in sorted(Path(tmpdir).iterdir()):
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                print(f"    {path.name:<25}  {size/1e3:.1f} KB")
            elif path.is_symlink():
                print(f"    {path.name:<25}  → {path.readlink().name}")
''',
    },

    "Sharded Checkpoint Simulation for FSDP": {
        "description": "Simulate sharded FSDP-style checkpointing where each rank saves its own parameter shard, and show how to consolidate or reshard for different GPU counts.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
SHARDED CHECKPOINT: FSDP-STYLE SHARD SAVE AND LOAD
================================================================================

Simulates the key concepts of sharded checkpointing:
    1. Each GPU rank holds and saves only its parameter shard
    2. Shard metadata tells us how to reassemble the full model
    3. Consolidation: merge N shards into one full checkpoint
    4. Resharding: convert a checkpoint from N shards to M shards
    5. Consistency checking: verify all shards are compatible

This is the core of FSDP's SHARDED_STATE_DICT and Torch DCP.

================================================================================
"""

import json
import math
import tempfile
import os
from pathlib import Path
import torch
import torch.nn as nn


# ── Shard metadata ────────────────────────────────────────────────────────────

def compute_shard_slices(param_shape: tuple, n_shards: int,
                          shard_dim: int = 0) -> list[tuple]:
    """
    Compute the slice each shard holds.
    Shards divide the first dimension (or specified dim) evenly.
    Returns list of (start, end) tuples, one per shard.
    """
    total   = param_shape[shard_dim]
    per_shard = math.ceil(total / n_shards)
    slices  = []
    for i in range(n_shards):
        start = i * per_shard
        end   = min(start + per_shard, total)
        slices.append((start, end))
    return slices


def shard_state_dict(full_state_dict: dict, rank: int, n_shards: int,
                      shard_dim: int = 0) -> tuple[dict, dict]:
    """
    Partition a full state dict into the slice belonging to `rank`.

    Returns:
        shard_dict:    the tensors belonging to this rank
        shard_metadata: shape/slice information for reconstruction
    """
    shard_dict     = {}
    shard_metadata = {
        "rank":     rank,
        "n_shards": n_shards,
        "shard_dim": shard_dim,
        "params":  {}
    }

    for name, tensor in full_state_dict.items():
        if tensor.dim() == 0:
            # Scalar — every rank holds a copy
            shard_dict[name] = tensor.clone()
            shard_metadata["params"][name] = {
                "shape":   list(tensor.shape),
                "dtype":   str(tensor.dtype),
                "type":    "scalar",
            }
        else:
            slices   = compute_shard_slices(tensor.shape, n_shards, shard_dim)
            start, end = slices[rank]
            shard    = tensor.narrow(shard_dim, start, end - start).clone()

            shard_dict[name] = shard
            shard_metadata["params"][name] = {
                "full_shape": list(tensor.shape),
                "shard_shape": list(shard.shape),
                "dtype":       str(tensor.dtype),
                "start":       start,
                "end":         end,
                "type":        "sharded",
            }

    return shard_dict, shard_metadata


def unshard_state_dict(shards: list[dict],
                        metadatas: list[dict]) -> dict:
    """
    Reconstruct a full state dict from a list of shards.
    Concatenates tensors along the shard dimension.
    """
    full_dict  = {}
    n_shards   = metadatas[0]["n_shards"]
    shard_dim  = metadatas[0]["shard_dim"]

    # Get all param names from the first shard's metadata
    param_names = list(metadatas[0]["params"].keys())

    for name in param_names:
        meta = metadatas[0]["params"][name]

        if meta["type"] == "scalar":
            full_dict[name] = shards[0][name]
        else:
            # Sort shards by rank to concatenate in order
            pieces = []
            for rank_idx in range(n_shards):
                if name in shards[rank_idx]:
                    pieces.append(shards[rank_idx][name])
            full_dict[name] = torch.cat(pieces, dim=shard_dim)

    return full_dict


def reshard_state_dict(full_state_dict: dict, new_n_shards: int,
                        shard_dim: int = 0) -> list[tuple]:
    """
    Create shards for a new topology.
    Returns list of (shard_dict, shard_metadata) for each new shard.
    """
    return [
        shard_state_dict(full_state_dict, rank, new_n_shards, shard_dim)
        for rank in range(new_n_shards)
    ]


# ── Tiny model ────────────────────────────────────────────────────────────────

class TinyModel(nn.Module):
    def __init__(self, vocab=64, d=32, n_layers=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.fc1    = nn.Linear(d, d*2, bias=False)
        self.fc2    = nn.Linear(d*2, d, bias=False)
        self.head   = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return self.head(h)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    VOCAB, D = 64, 32
    N_SHARDS_ORIG = 4   # original training used 4 "GPUs"
    N_SHARDS_NEW  = 2   # want to load on 2 "GPUs"

    model = TinyModel(VOCAB, D)
    full_sd = model.state_dict()

    print("=" * 65)
    print(f"  SHARDED CHECKPOINT SIMULATION")
    print(f"  Original topology: {N_SHARDS_ORIG} shards → New: {N_SHARDS_NEW}")
    print("=" * 65)
    print()

    print("  Full model state_dict shapes:")
    for k, v in full_sd.items():
        print(f"    {k:<30}: {list(v.shape)}  ({v.numel():,} params)")
    print()

    # ── Save: shard across 4 ranks ────────────────────────────────────────────
    print(f"  Saving as {N_SHARDS_ORIG} shards (simulating {N_SHARDS_ORIG}-GPU training):")
    shards   = []
    metas    = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for rank in range(N_SHARDS_ORIG):
            shard, meta = shard_state_dict(full_sd, rank, N_SHARDS_ORIG)
            shards.append(shard)
            metas.append(meta)

            # Save to disk
            shard_path = os.path.join(tmpdir, f"rank_{rank:02d}_shard.pt")
            meta_path  = os.path.join(tmpdir, f"rank_{rank:02d}_meta.json")
            torch.save(shard, shard_path)
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            shard_bytes = sum(v.numel() * v.element_size()
                               for v in shard.values() if hasattr(v, 'numel'))
            print(f"    Rank {rank}: saved {shard_bytes/1e3:.1f} KB  "
                  f"({shard_bytes / (sum(v.numel()*v.element_size() for v in full_sd.values())/N_SHARDS_ORIG)*100:.0f}% of expected per-rank)")

        # Show one shard's contents
        print()
        print(f"  Rank 0 shard contents:")
        for k, v in shards[0].items():
            meta_k = metas[0]["params"][k]
            if meta_k["type"] == "sharded":
                print(f"    {k:<30}: {list(v.shape)}  [{meta_k['start']}:{meta_k['end']}] of full {meta_k['full_shape']}")
            else:
                print(f"    {k:<30}: {list(v.shape)}  (scalar, replicated)")

        # ── Consolidate: merge all shards back ────────────────────────────────
        print()
        print("  Consolidating shards back to full model...")
        reconst = unshard_state_dict(shards, metas)
        max_diff = max(
            (full_sd[k] - reconst[k]).abs().max().item()
            for k in full_sd if k in reconst
        )
        print(f"  Max reconstruction error: {max_diff:.2e}  "
              f"{'✓ perfect reconstruction' if max_diff < 1e-6 else '✗ error!'}")

        # ── Reshard to different topology ──────────────────────────────────────
        print()
        print(f"  Resharding to {N_SHARDS_NEW} shards (topology change):")
        new_shards_data = reshard_state_dict(full_sd, N_SHARDS_NEW)

        for rank, (new_shard, new_meta) in enumerate(new_shards_data):
            print(f"  New rank {rank} shard sizes:")
            for k, v in new_shard.items():
                if new_meta["params"][k]["type"] == "sharded":
                    m = new_meta["params"][k]
                    print(f"    {k:<30}: {list(v.shape)}  [{m['start']}:{m['end']}]")

        # Verify new shards reconstruct correctly
        new_shards_list  = [s for s, _ in new_shards_data]
        new_metas_list   = [m for _, m in new_shards_data]
        reconst_new = unshard_state_dict(new_shards_list, new_metas_list)
        max_diff_new = max(
            (full_sd[k] - reconst_new[k]).abs().max().item()
            for k in full_sd if k in reconst_new
        )
        print()
        print(f"  Resharded reconstruction error: {max_diff_new:.2e}  "
              f"{'✓ correct' if max_diff_new < 1e-6 else '✗ error!'}")

    # ── Memory analysis ────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  MEMORY COMPARISON: CONSOLIDATED vs SHARDED")
    print("=" * 65)
    print()

    total_bytes = sum(v.numel() * v.element_size() for v in full_sd.values())
    print(f"  Full model: {total_bytes/1e3:.1f} KB")
    print()
    print(f"  {'N shards':>10}  {'Memory per rank':>18}  {'Save time (est)':>18}  "
          f"{'Load OOM risk':>16}")
    print(f"  {'':─>10}  {'':─>18}  {'':─>18}  {'':─>16}")

    for n in [1, 2, 4, 8, 16, 32]:
        per_rank   = total_bytes / n
        save_time  = per_rank / (5e9 / n)   # parallel writes
        oom_risk   = "✓ safe" if per_rank < 80e9 else "✗ OOM"
        print(f"  {n:>10}  {per_rank/1e3:>16.1f} KB  "
              f"{save_time*1000:>16.1f} ms  {oom_risk:>16}")

    print()
    print("  For large models (70B+): consolidated save requires rank 0 to hold")
    print("  the FULL model in RAM → OOM for models > GPU memory.")
    print("  → Always use sharded checkpoints for models that don't fit on 1 GPU.")
''',
    },

    "Checkpoint Metadata, Lineage, and Rollback": {
        "description": "Build a checkpoint registry that tracks training lineage, detects loss spikes for automatic rollback, and finds the best checkpoint for fine-tuning.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
CHECKPOINT REGISTRY: LINEAGE, SPIKE DETECTION, AND ROLLBACK
================================================================================

Implements a checkpoint registry that:
    1. Tracks the full training lineage (parent-child checkpoint relationships)
    2. Detects loss spikes and identifies the last healthy checkpoint
    3. Suggests the best checkpoint for fine-tuning (best validation loss)
    4. Plots training history from metadata alone (no model loading needed)
    5. Estimates training cost from metadata

This shows how checkpoint metadata becomes a powerful tool for managing
long training runs where you may have hundreds of checkpoints.

================================================================================
"""

import json
import math
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import tempfile


@dataclass
class CheckpointRecord:
    """One entry in the checkpoint registry."""
    path:        str
    step:        int
    tokens_seen: int
    train_loss:  float
    val_loss:    Optional[float]
    lr:          float
    wall_clock_s: float
    parent_path: Optional[str]
    git_hash:    str
    timestamp:   str
    flags:       list = field(default_factory=list)   # e.g., ["spike", "best_val"]


class CheckpointRegistry:
    """
    Manages a collection of checkpoints with lineage tracking,
    spike detection, and quality-based selection.
    """

    def __init__(self):
        self.records: list[CheckpointRecord] = []
        self._by_path: dict[str, CheckpointRecord] = {}
        self._by_step: dict[int, CheckpointRecord] = {}

    def register(self, record: CheckpointRecord) -> None:
        """Add a checkpoint to the registry."""
        self.records.append(record)
        self._by_path[record.path] = record
        self._by_step[record.step] = record

    def get_lineage(self, path: str) -> list[CheckpointRecord]:
        """
        Get the lineage chain from root to the given checkpoint.
        Returns [root, ..., parent, checkpoint].
        """
        chain = []
        current_path = path
        while current_path is not None:
            record = self._by_path.get(current_path)
            if record is None:
                break
            chain.append(record)
            current_path = record.parent_path
        return list(reversed(chain))

    def detect_spikes(self, window: int = 5,
                       threshold: float = 3.0) -> list[CheckpointRecord]:
        """
        Detect loss spikes: steps where loss exceeds threshold × rolling mean.

        Returns list of spike checkpoints.
        """
        if len(self.records) < window:
            return []

        spikes = []
        sorted_records = sorted(self.records, key=lambda r: r.step)

        for i, record in enumerate(sorted_records):
            if i < window:
                continue
            window_losses = [sorted_records[j].train_loss
                              for j in range(max(0, i - window), i)]
            rolling_mean  = sum(window_losses) / len(window_losses)
            if record.train_loss > threshold * rolling_mean:
                record.flags.append("spike")
                spikes.append(record)

        return spikes

    def last_healthy_before(self, step: int) -> Optional[CheckpointRecord]:
        """
        Find the most recent checkpoint before `step` that is not flagged
        as a spike. Used for rollback.
        """
        candidates = [r for r in self.records
                       if r.step < step and "spike" not in r.flags]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.step)

    def best_for_finetuning(self) -> Optional[CheckpointRecord]:
        """
        Find the checkpoint with the best validation loss.
        This is typically the best starting point for fine-tuning.
        """
        valid = [r for r in self.records if r.val_loss is not None]
        if not valid:
            return None
        best = min(valid, key=lambda r: r.val_loss)
        best.flags.append("best_val")
        return best

    def training_summary(self) -> dict:
        """
        Compute training statistics from checkpoint metadata alone.
        No model loading required.
        """
        if not self.records:
            return {}
        sorted_r = sorted(self.records, key=lambda r: r.step)
        first, last = sorted_r[0], sorted_r[-1]
        return {
            "total_steps":        last.step - first.step,
            "total_tokens":       last.tokens_seen,
            "total_wall_hours":   last.wall_clock_s / 3600,
            "tokens_per_hour":    last.tokens_seen / max(last.wall_clock_s / 3600, 1e-6),
            "initial_loss":       first.train_loss,
            "final_loss":         last.train_loss,
            "loss_reduction":     (first.train_loss - last.train_loss) / first.train_loss * 100,
            "n_checkpoints":      len(self.records),
        }

    def print_history(self, n_rows: int = 20) -> None:
        """Print a summary table of training history."""
        sorted_r = sorted(self.records, key=lambda r: r.step)
        stride   = max(1, len(sorted_r) // n_rows)

        print(f"  {'Step':>8}  {'Tokens':>12}  {'Train loss':>12}  "
              f"{'Val loss':>10}  {'LR':>10}  {'Flags'}")
        print(f"  {'':─>8}  {'':─>12}  {'':─>12}  "
              f"{'':─>10}  {'':─>10}  {'':─}")

        for i, r in enumerate(sorted_r):
            if i % stride == 0 or i == len(sorted_r) - 1:
                val_str  = f"{r.val_loss:.4f}" if r.val_loss else "—"
                flag_str = " ".join(f"[{f}]" for f in r.flags)
                print(f"  {r.step:>8,}  {r.tokens_seen:>12,}  "
                      f"{r.train_loss:>12.4f}  {val_str:>10}  "
                      f"{r.lr:>10.2e}  {flag_str}")


# ── Simulate a training run with checkpoints ──────────────────────────────────

def simulate_training_run(n_steps: int = 50,
                            spike_at: int = 30,
                            seed: int = 42) -> CheckpointRegistry:
    """
    Simulate a training run with periodic checkpoints and one loss spike.
    Returns a populated CheckpointRegistry.
    """
    rng = random.Random(seed)
    registry = CheckpointRegistry()

    # Simulate training loss (decreasing with noise)
    loss     = 4.0
    lr_start = 3e-4
    prev_path = None

    for step_idx, step in enumerate(range(0, n_steps * 100, 100)):
        # Simulate LR decay
        lr = lr_start * (1 - step / (n_steps * 100 * 1.2))

        # Simulate loss evolution
        loss *= (0.98 + rng.gauss(0, 0.005))   # decay with noise
        loss  = max(loss, 0.8)

        # Inject a loss spike
        if step == spike_at * 100:
            loss *= 3.5   # spike!

        # Simulate val loss (slightly higher than train)
        val_loss = loss * (1.05 + rng.uniform(0, 0.02))

        # Simulate tokens seen
        tokens = step * 512 * 16   # B*T per step

        path = f"/ckpts/step-{step:07d}"
        record = CheckpointRecord(
            path        = path,
            step        = step,
            tokens_seen = tokens,
            train_loss  = loss,
            val_loss    = val_loss,
            lr          = max(lr, 1e-6),
            wall_clock_s = step * 30,   # 30 seconds per step
            parent_path  = prev_path,
            git_hash     = "abc1234",
            timestamp    = "2024-01-01T00:00:00Z",
        )
        registry.register(record)
        prev_path = path

    return registry


if __name__ == "__main__":
    print("=" * 65)
    print("  CHECKPOINT REGISTRY: LINEAGE AND SPIKE DETECTION")
    print("=" * 65)
    print()

    registry = simulate_training_run(n_steps=40, spike_at=20)

    # Detect spikes
    spikes = registry.detect_spikes(window=5, threshold=2.5)
    print(f"  Training history ({len(registry.records)} checkpoints):")
    registry.print_history(n_rows=12)
    print()

    print(f"  Loss spikes detected: {len(spikes)}")
    for spike in spikes:
        print(f"    Step {spike.step:,}: loss={spike.train_loss:.4f}")
        # Find last healthy checkpoint before the spike
        healthy = registry.last_healthy_before(spike.step)
        if healthy:
            print(f"    → Last healthy checkpoint: step {healthy.step:,} "
                  f"(loss={healthy.train_loss:.4f})")
            print(f"    → Rollback recommendation: load from step {healthy.step:,}")
    print()

    # Best for fine-tuning
    best = registry.best_for_finetuning()
    print(f"  Best checkpoint for fine-tuning:")
    if best:
        print(f"    Step {best.step:,}: val_loss={best.val_loss:.4f}, "
              f"tokens={best.tokens_seen:,}")
    print()

    # Training summary
    summary = registry.training_summary()
    print("  Training statistics:")
    print(f"    Total steps:         {summary['total_steps']:,}")
    print(f"    Total tokens:        {summary['total_tokens']:,}")
    print(f"    Wall-clock time:     {summary['total_wall_hours']:.1f} hours")
    print(f"    Throughput:          {summary['tokens_per_hour']:,.0f} tokens/hour")
    print(f"    Loss reduction:      {summary['loss_reduction']:.1f}%  "
          f"({summary['initial_loss']:.4f} → {summary['final_loss']:.4f})")
    print()

    # Lineage demo
    last_path = sorted(registry.records, key=lambda r: r.step)[-1].path
    lineage   = registry.get_lineage(last_path)
    print(f"  Lineage chain (last 5 of {len(lineage)}):")
    for r in lineage[-5:]:
        print(f"    Step {r.step:,} ← ", end="")
    print("(root)")
    print()
    print("  Key insight: with complete metadata, you can reconstruct the")
    print("  full training story without loading any model weights.")
    print("  This is invaluable for debugging and retrospective analysis.")
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
    #     from llm_training.visuals.checkpoint_management import (
    #         CKPT_VISUAL_HTML,
    #         CKPT_VISUAL_HEIGHT,
    #     )
    #     visual_html   = CKPT_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = CKPT_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[33_checkpoint_management.py] Could not load visual: {e}",
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