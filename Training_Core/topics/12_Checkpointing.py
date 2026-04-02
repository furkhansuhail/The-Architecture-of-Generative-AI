"""
Checkpointing
=============

A checkpoint is a snapshot of a model's complete state at a point
in time. Without checkpointing, every training run is a gamble: a
hardware failure, a power outage, or a training divergence means
starting from scratch. With checkpointing done correctly, training
is resumable from any saved state, the best-performing model is
always preserved, and experiments are reproducible from any
intermediate point. Checkpointing is also the mechanism by which
trained models are transferred from training infrastructure to
production — the checkpoint is the deliverable.

"""
import textwrap
import re

TOPIC_NAME   = "Checkpointing"
DISPLAY_NAME = "12 · Checkpointing"
ICON         = "💾"
SUBTITLE     = "Save · Resume · Best-Model · State Dicts · Formats · Deployment"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT A COMPLETE CHECKPOINT CONTAINS

### The Components of Training State

Resuming a training run from a checkpoint requires restoring EVERY
component of the training state — not just the model weights:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Component              │ What it contains                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Model state dict       │ All learnable parameters (weights, biases) │
    │                        │ and persistent buffers (BatchNorm running  │
    │                        │ mean/var, positional embeddings)           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Optimiser state dict   │ Momentum buffers (SGD), first/second       │
    │                        │ moment estimates (Adam m̂ᵢ, v̂ᵢ), step       │
    │                        │ count, parameter group LR values           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ LR scheduler state     │ Current step/epoch count, warmup state,    │
    │                        │ current LR, decay history                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ GradScaler state       │ Current loss scale value, growth interval  │
    │ (FP16 training only)   │ counter, backoff history                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Training metadata      │ Epoch number, global step count, best      │
    │                        │ validation metric, random seeds            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ RNG states             │ Python random, NumPy, PyTorch CPU and      │
    │                        │ CUDA RNG states (for exact reproducibility)│
    └─────────────────────────────────────────────────────────────────────┘

    Omitting any component produces a subtly broken resume:

    Missing OPTIMISER STATE:
    Adam's moment estimates (m̂ᵢ, v̂ᵢ) reflect the history of gradients
    seen so far. Resetting them means the resumed run starts with incorrect
    per-parameter learning rates. The model takes many steps to re-warm
    the moment estimates. Training effectively regresses 5-10% of its
    progress at the resume point.

    Missing LR SCHEDULER STATE:
    The scheduler resumes from step 0 rather than the checkpoint step.
    If using cosine annealing, the LR jumps back to LR_max after resuming,
    potentially causing instability or unnecessary backtracking.

    Missing EPOCH COUNT:
    The training loop restarts from epoch 0. The data sampler is re-seeded
    from the beginning, causing the model to see the same data ordering it
    already saw. This breaks the guarantee that each example is seen
    equally often per epoch.

    Missing GRADSCALER STATE (FP16):
    The loss scale resets to its initial value (typically 65536). If the
    training had settled on a lower stable scale (e.g., 512), the too-large
    initial scale produces gradient overflow in the first few steps.


### State Dict vs Full Model Serialisation

    PyTorch offers two approaches to saving a model:

    APPROACH 1 — State dict (RECOMMENDED):
    torch.save(model.state_dict(), 'weights.pt')
    model.load_state_dict(torch.load('weights.pt'))

    The state dict is a Python OrderedDict mapping parameter names
    to tensors. It contains ONLY the data, not the model architecture.
    Advantages:
    - Portable: can be loaded into any model that defines the same
      architecture, even if defined in a different file or framework.
    - Small: no code is serialised.
    - Stable: not broken by Python version changes or code refactors.
    Disadvantage:
    - Requires the model class definition to be available when loading.

    APPROACH 2 — Full model (AVOID for long-term storage):
    torch.save(model, 'model.pt')
    model = torch.load('model.pt')

    This uses Python's pickle protocol to serialise the entire object
    including the class definition and all its dependencies.
    Advantages:
    - Convenient for quick experiments.
    Disadvantages:
    - Brittle: breaks if the model class is renamed, moved, or refactored.
    - Couples the checkpoint to a specific codebase version.
    - Security risk: loading arbitrary pickled objects can execute
      arbitrary code (do not load untrusted checkpoints with torch.load).
    - Does not work across Python versions.

    Rule: ALWAYS use state dicts for any checkpoint that will outlive
    a single experiment session.


### What a Complete Checkpoint Save Looks Like

    def save_checkpoint(path, model, optimiser, scheduler,
                        scaler, epoch, step, best_val_metric, config):
        '''
        Save a complete, resumable training checkpoint.

        Args:
            path:            File path (e.g., 'checkpoints/epoch_42.pt').
            model:           The model (nn.Module).
            optimiser:       The optimiser.
            scheduler:       The LR scheduler (or None).
            scaler:          GradScaler for FP16 (or None).
            epoch:           Current epoch number (0-indexed).
            step:            Global gradient step count.
            best_val_metric: Best validation metric seen so far.
            config:          Dict of all training hyperparameters.
        '''
        checkpoint = {
            # Core training state
            'model_state_dict':     model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'epoch':                epoch,
            'step':                 step,
            'best_val_metric':      best_val_metric,

            # LR scheduling state
            'scheduler_state_dict': scheduler.state_dict()
                                    if scheduler is not None else None,

            # FP16 loss scaling state
            'scaler_state_dict':    scaler.state_dict()
                                    if scaler is not None else None,

            # RNG states (for exact reproducibility on resume)
            'rng_state': {
                'python':   random.getstate(),
                'numpy':    np.random.get_state(),
                'torch':    torch.get_rng_state(),
                'cuda':     torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None,
            },

            # Human-readable metadata
            'config':   config,
            'timestamp': datetime.utcnow().isoformat(),
        }
        # Atomic write: write to temp file, then rename
        tmp_path = path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)   # atomic on POSIX systems


### Why Atomic Writes Matter

    The MOST DANGEROUS MOMENT in checkpointing is the write itself.
    If the process is killed mid-write (power failure, OOM killer,
    SIGKILL), the file on disk is CORRUPT — partially written.

    Non-atomic write:
    torch.save(checkpoint, path)   ← if killed here, path is corrupt
    The checkpoint file is overwritten in-place. A partial write
    leaves a corrupt file at path. The PREVIOUS checkpoint was
    already overwritten. BOTH checkpoints are now gone.

    Atomic write (POSIX rename):
    torch.save(checkpoint, path + '.tmp')   ← write to temp file
    os.replace(path + '.tmp', path)         ← atomic rename

    os.replace() is atomic on POSIX systems (Linux, macOS): the kernel
    guarantees that the rename is either fully done or not done — there
    is no intermediate state where path contains a partial file.
    If the process is killed during torch.save(), path.tmp is corrupt
    but the ORIGINAL checkpoint at path is intact.

    On Windows: os.replace() is not guaranteed atomic. Use a more
    robust approach (write to temp, validate, then replace) or use
    the filelock library.


##### PART 2 — LOADING AND RESUMING FROM A CHECKPOINT

### The Complete Resume Procedure

    def load_checkpoint(path, model, optimiser=None, scheduler=None,
                        scaler=None, device='cpu', strict=True):
        '''
        Load a checkpoint and restore all training state.

        Args:
            path:      Path to the checkpoint file.
            model:     The model to load weights into.
            optimiser: The optimiser to restore state into (or None
                       for inference-only loading).
            scheduler: The LR scheduler to restore (or None).
            scaler:    GradScaler to restore (or None).
            device:    Device to map tensors to ('cpu', 'cuda:0', etc.)
            strict:    If True, keys in state dict must exactly match
                       the model's parameters. Set False for partial loading.

        Returns:
            dict with 'epoch', 'step', 'best_val_metric', 'config'.
        '''
        checkpoint = torch.load(path, map_location=device)

        # 1 — Restore model weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        # 2 — Restore optimiser state (only when resuming training)
        if optimiser is not None and 'optimiser_state_dict' in checkpoint:
            optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            # Move optimiser state to the correct device
            # (optimiser states are saved on CPU by default in older PyTorch)
            for state in optimiser.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        # 3 — Restore LR scheduler state
        if scheduler is not None and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 4 — Restore GradScaler state (FP16)
        if scaler is not None and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 5 — Restore RNG states (for exact reproducibility)
        rng = checkpoint.get('rng_state', {})
        if rng.get('python'):
            random.setstate(rng['python'])
        if rng.get('numpy') is not None:
            np.random.set_state(rng['numpy'])
        if rng.get('torch') is not None:
            torch.set_rng_state(rng['torch'])
        if rng.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])

        return {
            'epoch':           checkpoint.get('epoch', 0),
            'step':            checkpoint.get('step', 0),
            'best_val_metric': checkpoint.get('best_val_metric', None),
            'config':          checkpoint.get('config', {}),
        }


### map_location: Loading Across Devices

    When a checkpoint is saved on GPU, all tensors are stored as CUDA
    tensors. Loading on a machine with a different GPU configuration
    (or no GPU) requires remapping:

    # Saved on GPU 0, loading on GPU 0 (same machine):
    checkpoint = torch.load('ckpt.pt')   # works, but may use wrong device

    # Saved on GPU 0, loading on GPU 1 (different GPU):
    checkpoint = torch.load('ckpt.pt', map_location='cuda:1')

    # Saved on GPU, loading on CPU (inference or different machine):
    checkpoint = torch.load('ckpt.pt', map_location='cpu')

    # Loading and mapping to the current default device:
    checkpoint = torch.load('ckpt.pt', map_location=lambda s, _: s)

    # Recommended general-purpose pattern:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('ckpt.pt', map_location=device)

    CRITICAL: After loading the optimiser state (which is on CPU by default
    in checkpoints saved with map_location='cpu'), the optimiser states
    must be manually moved to the correct device:
    for state in optimiser.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    Forgetting this leaves optimiser state on CPU while model is on GPU,
    causing either slow CPU-GPU transfers or runtime errors.


### Strict vs Non-Strict Loading

    strict=True (default):
    Every key in the checkpoint must exactly match a parameter in the model.
    Every parameter in the model must be present in the checkpoint.
    If any key is missing or unexpected: RuntimeError.

    Use when: resuming from an exact checkpoint of the same model.

    strict=False:
    Missing keys (in checkpoint but not in model): silently skipped.
    Unexpected keys (in model but not in checkpoint): left at their
    initialised values.
    No error is raised.

    Use when:
    - Loading a pre-trained backbone into a larger model with a new head.
    - Loading weights from a model with a slightly different architecture.
    - Transfer learning: the classification head has a different number
      of classes.

    Checking what was and was not loaded:
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False)
    print(f"Missing: {missing_keys}")      ← params NOT loaded from ckpt
    print(f"Unexpected: {unexpected_keys}")← ckpt keys not in model

    Always print these lists when using strict=False to verify
    that the right parameters were loaded and the right ones were skipped.


##### PART 3 — BEST-MODEL TRACKING

### Why Best-Model Tracking Matters

The model at the END of training is not necessarily the best model.
Late in training, the model may have overfit past its optimal point —
the best checkpoint is at the epoch where validation performance peaked.

    Diagram 1 — Best Checkpoint vs Final Checkpoint:

    Val loss
      │
      │  ╲
      │   ╲──╮                    ← best val loss (best checkpoint)
      │      ╰──────────────╮
      │                      ╰─── ← val loss at end of training
      │                            (worse due to overfit)
      └──────────────────────────── Epoch
                  ↑
             best epoch
             (save this checkpoint)

    The difference between best and final checkpoint can be:
    Trivial: 0.1% when strong regularisation prevents overfitting.
    Significant: 2-5% when the model is prone to late-epoch overfitting.

    Without best-model tracking, the model deployed to production is
    the final checkpoint — which may not be the best model produced
    during training. This is a systematic performance cost with zero
    benefit.


### Best-Model Checkpoint Manager

    class BestModelTracker:
        '''
        Tracks the best validation metric and saves the corresponding
        checkpoint. Supports both 'min' (loss) and 'max' (accuracy) modes.
        '''

        def __init__(self, save_path, mode='min', min_delta=0.0,
                     save_full=True, verbose=True):
            '''
            Args:
                save_path:  Path to save the best checkpoint.
                mode:       'min' to track loss (lower = better),
                            'max' to track accuracy (higher = better).
                min_delta:  Minimum change to qualify as an improvement.
                save_full:  If True, save the full training state.
                            If False, save only model weights (for deployment).
                verbose:    Print a message when the best model is updated.
            '''
            self.save_path = save_path
            self.mode = mode
            self.min_delta = min_delta
            self.save_full = save_full
            self.verbose = verbose
            self.best_metric = float('inf') if mode == 'min' else float('-inf')
            self.best_epoch = -1

        def is_improvement(self, metric):
            if self.mode == 'min':
                return metric < self.best_metric - self.min_delta
            else:
                return metric > self.best_metric + self.min_delta

        def step(self, metric, model, optimiser=None, scheduler=None,
                 scaler=None, epoch=0, step=0, config=None):
            '''
            Call after each validation pass.
            Saves a checkpoint if the metric has improved.

            Returns:
                True if the best model was updated, False otherwise.
            '''
            if self.is_improvement(metric):
                self.best_metric = metric
                self.best_epoch = epoch
                if self.save_full and optimiser is not None:
                    save_checkpoint(
                        self.save_path, model, optimiser, scheduler,
                        scaler, epoch, step, metric, config or {}
                    )
                else:
                    # Inference-only checkpoint (weights only)
                    torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"  ✓ Best model saved "
                          f"(epoch {epoch}, {self.mode} metric={metric:.6f})")
                return True
            return False

        def summary(self):
            return (f"Best {self.mode} metric: {self.best_metric:.6f} "
                    f"at epoch {self.best_epoch}")


### What Metric to Track

    The best-model tracker should monitor the metric that most closely
    reflects your deployment objective:

    ┌────────────────────────────────────────────────────────────────────┐
    │ Task                      │ Track metric    │ Mode                 │
    ├────────────────────────────────────────────────────────────────────┤
    │ Classification (balanced) │ val_loss        │ min (preferred over  │
    │                           │ or val_accuracy │ accuracy — smoother) │
    ├────────────────────────────────────────────────────────────────────┤
    │ Classification (imbalanced│ val_macro_f1    │ max                  │
    │ or rare class critical)   │ or val_PRAUC    │ max                  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Regression                │ val_loss (MSE)  │ min                  │
    │                           │ or val_MAE      │ min                  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Language modelling        │ val_perplexity  │ min                  │
    │                           │ or val_loss     │ min                  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Object detection          │ val_mAP         │ max                  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Generative (image)        │ val_FID         │ min                  │
    └────────────────────────────────────────────────────────────────────┘

    ALWAYS track the primary LOSS, not just the accuracy or task metric.
    Loss is smoother and more sensitive to small improvements.
    Supplementary metrics (F1, mAP) can be tracked in parallel but the
    primary checkpoint decision should be loss-driven unless the task
    metric is specifically what matters for deployment.


##### PART 4 — CHECKPOINT FREQUENCY AND RETENTION STRATEGY

### How Often to Checkpoint

    The checkpoint frequency is a tradeoff between:
    1. RECOVERY WINDOW: how much compute is lost on failure.
    2. STORAGE COST: checkpoint files are large (model size × 2–5×).
    3. I/O OVERHEAD: saving a large checkpoint takes time.

    ┌────────────────────────────────────────────────────────────────────┐
    │ Scenario                  │ Recommended frequency                  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Training for < 1 hour     │ End of training only (or every epoch)  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Training for 1-8 hours    │ Every epoch or every N steps (~hourly) │
    ├────────────────────────────────────────────────────────────────────┤
    │ Training for days         │ Every 30-60 minutes (step-based)       │
    │                           │ + best model checkpoint on val metric  │
    ├────────────────────────────────────────────────────────────────────┤
    │ Pre-training LLMs (weeks) │ Every 500-1000 steps; async to avoid   │
    │                           │ blocking the critical path             │
    ├────────────────────────────────────────────────────────────────────┤
    │ Cloud preemptible VMs     │ Every 10-30 minutes                    │
    │ (can be killed at any time│ (preemption loss = last checkpoint     │
    │ with ~30s warning)        │  gap × hourly compute cost)            │
    └────────────────────────────────────────────────────────────────────┘

    For cloud preemptible instances (spot instances, preemptible TPUs):
    The cost of losing 30 minutes of training is (30 min × hourly rate).
    At $10/hr: losing 30 min costs $5. A 100 GB checkpoint on fast storage
    takes ~1 min to save, costing 1 min × $10/hr ≈ $0.17. Savings are clear.
    Checkpoint aggressively on preemptible hardware.


### Checkpoint Retention Policies

Saving every checkpoint indefinitely is impractical — a 7B parameter
model checkpoint is ~14 GB. Over 1000 checkpoints: 14 TB.

    Common retention policies:

    KEEP LAST K:
    Maintain only the K most recent periodic checkpoints.
    When a new checkpoint is saved, delete the oldest.
    K=3 is standard (rolling window of recent state).

    KEEP BEST + LAST K:
    Always keep the best-metric checkpoint.
    Also keep the last K periodic checkpoints for fault tolerance.
    The best checkpoint is never deleted by the rolling window.
    This is the standard for most training runs.

    KEEP MILESTONE CHECKPOINTS:
    Retain checkpoints at significant training milestones: 
    epoch 1, 10, 25, 50, 100; the checkpoint before and after
    a major hyperparameter change; checkpoints used for published results.

    EXPONENTIAL RETENTION:
    Keep the most recent checkpoints at high frequency, with
    decreasing frequency for older ones:
    Last 10 checkpoints: keep all (every step)
    10-100 steps ago: keep every 10th
    100-1000 steps ago: keep every 100th
    > 1000 steps ago: keep only milestones

    This gives dense coverage of recent history (for debugging)
    and sparse coverage of distant history (for reference).


### Checkpoint Manager Implementation

    import glob

    class CheckpointManager:
        '''
        Manages periodic checkpoints with a rolling retention window,
        plus a separate best-model checkpoint that is never deleted.
        '''

        def __init__(self, save_dir, keep_last=3, prefix='ckpt'):
            '''
            Args:
                save_dir:  Directory to save checkpoints.
                keep_last: Number of recent checkpoints to retain.
                prefix:    Filename prefix (e.g., 'ckpt' → 'ckpt_step_1000.pt').
            '''
            os.makedirs(save_dir, exist_ok=True)
            self.save_dir = save_dir
            self.keep_last = keep_last
            self.prefix = prefix
            self.saved_paths = []   # ordered list of periodic checkpoint paths

        def save(self, model, optimiser, scheduler, scaler,
                 epoch, step, val_metric, config):
            '''Save a periodic checkpoint and enforce the retention policy.'''
            fname = f"{self.prefix}_step_{step:07d}_ep_{epoch:04d}.pt"
            path  = os.path.join(self.save_dir, fname)
            save_checkpoint(path, model, optimiser, scheduler,
                            scaler, epoch, step, val_metric, config)
            self.saved_paths.append(path)

            # Enforce keep_last policy
            while len(self.saved_paths) > self.keep_last:
                oldest = self.saved_paths.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                    print(f"  Deleted old checkpoint: {oldest}")

            return path

        def latest(self):
            '''Return the path to the most recent checkpoint, or None.'''
            return self.saved_paths[-1] if self.saved_paths else None

        @classmethod
        def from_dir(cls, save_dir, keep_last=3, prefix='ckpt'):
            '''Reconstruct manager state from an existing checkpoint directory.'''
            manager = cls(save_dir, keep_last, prefix)
            pattern = os.path.join(save_dir, f'{prefix}_step_*.pt')
            existing = sorted(glob.glob(pattern))   # sorted by step number
            manager.saved_paths = existing[-keep_last:]   # only last K
            return manager

##### PART 5 — CHECKPOINTING IN DISTRIBUTED TRAINING

### DDP (DistributedDataParallel) Checkpointing

In DDP training, the model is replicated across N GPUs. Each GPU
holds an IDENTICAL copy of the model weights — gradients are
synchronised at each step to keep all copies in sync.

    Key rule: ONLY THE RANK 0 PROCESS saves checkpoints.
    Every other process has an identical model, so saving from all
    processes would produce N identical files (wasted disk I/O) and
    can cause race conditions.

    Correct DDP checkpointing pattern:
    if dist.get_rank() == 0:       ← only rank 0 saves
        save_checkpoint(
            path,
            model.module,          ← model.module, NOT model
            optimiser, scheduler, scaler, epoch, step, val_metric, config
        )

    DDP wraps the model in a DDP container:
    model = DDP(model, device_ids=[local_rank])

    model.state_dict()        → returns the DDP wrapper's state dict
                                (includes 'module.' prefix on all keys)
    model.module.state_dict() → returns the INNER model's state dict
                                (no prefix — this is what you want)

    Saving model.state_dict() instead of model.module.state_dict()
    produces checkpoints with keys like 'module.layer1.weight' instead
    of 'layer1.weight'. Loading these into a non-DDP model (e.g., at
    inference) requires stripping the 'module.' prefix:

    # Fix for checkpoints accidentally saved with 'module.' prefix:
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', '', 1): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


### Synchronisation Before Saving

In DDP, all processes must reach the checkpoint save point together.
If rank 0 saves while other ranks are still in the middle of a step,
the checkpoint may be inconsistent:

    # Barrier: all processes wait here until all have arrived
    dist.barrier()
    if dist.get_rank() == 0:
        save_checkpoint(...)
    dist.barrier()   # second barrier: all wait for rank 0 to finish saving

    The second barrier ensures ranks 1..N-1 do not proceed to the next
    step while rank 0 is still writing the checkpoint file. This matters
    because the next step will change the model state — without the
    barrier, the saved state could be inconsistent.


### FSDP (Fully Sharded Data Parallel) Checkpointing

FSDP shards model parameters across GPUs — each GPU holds only a
FRACTION of the parameters. This fundamentally changes checkpointing:

    Naive approach (WRONG for FSDP):
    Each rank saves its local shard. To reload, you need the same
    number of GPUs with the same sharding configuration.
    This creates a GPU-count dependency: a 64-GPU checkpoint cannot
    be loaded on 32 GPUs.

    Correct FSDP approach — consolidated saving:
    Use FSDP's built-in save/load utilities to gather shards and
    save a unified checkpoint:

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    # Save: gather all shards to rank 0 and save a unified state dict
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save({'model_state_dict': state_dict}, path)

    The offload_to_cpu=True option moves parameters to CPU during
    gathering, preventing OOM from having all parameters on one GPU.
    The rank0_only=True ensures only rank 0 gets the full state dict.

    SHARDED CHECKPOINTING (alternative for very large models):
    Save each rank's shard separately. Loading requires the same number
    of GPUs and the same FSDP configuration.
    Pros: fast to save (no communication overhead for gathering).
    Cons: not portable across GPU count changes.
    Used when: the model is too large to fit on one GPU even temporarily.


### Asynchronous Checkpointing

For large models (100B+ parameters), saving a checkpoint can take
minutes. Blocking training for minutes every N steps is unacceptable.

    ASYNCHRONOUS CHECKPOINTING: save in a background thread/process
    while training continues on the GPU.

    Simple threading approach:
    import threading

    def async_save(checkpoint, path):
        tmp = path + '.tmp'
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)

    def save_async(model, optimiser, ...):
        # Snapshot the state dict on CPU (fast copy from GPU)
        snapshot = {
            'model_state_dict': {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()},
            'optimiser_state_dict': copy.deepcopy(optimiser.state_dict()),
            ...
        }
        t = threading.Thread(target=async_save, args=(snapshot, path))
        t.daemon = True
        t.start()
        return t    # caller can join() before the next save if needed

    The CPU copy (v.cpu().clone()) is the key: this moves the tensor
    data to CPU WHILE the next training step begins on GPU. CPU and GPU
    can work simultaneously.

    Limitation: two saves must not overlap (the second would overwrite
    the temp file of the first). Track the thread and wait for it to
    complete before starting the next save:
    if save_thread is not None:
        save_thread.join()   ← wait for previous save to complete
    save_thread = save_async(...)


##### PART 6 — CHECKPOINT FORMATS FOR DEPLOYMENT

### PyTorch Native: .pt / .pth

    The default PyTorch format. Saves state dicts using Python pickle.

    Pros:
    - Native to PyTorch — no extra dependencies.
    - Supports all PyTorch tensor types and metadata.
    - State dicts are human-inspectable (keys are readable strings).

    Cons:
    - Not cross-framework (cannot load in TensorFlow or JAX).
    - pickle-based — potential security risk with untrusted sources.
    - Not optimised for large-file streaming.

    Conventions:
    .pt  = any PyTorch checkpoint (common generic extension)
    .pth = legacy; functionally identical to .pt
    Use .pt for all new code — there is no technical distinction.


### SafeTensors

    SafeTensors (Hugging Face, 2022) is a security-first format for
    storing tensors:

    WHY IT EXISTS:
    torch.load() with Python pickle can execute arbitrary code during
    loading. A malicious checkpoint file distributed online can run
    code on the loading machine. This is a real attack vector.

    SafeTensors uses a simple binary format (header + raw tensor data)
    with NO code execution during loading. It is safe to load from
    untrusted sources.

    ADDITIONAL BENEFITS:
    - Zero-copy memory mapping: tensors can be memory-mapped rather than
      copied into RAM. Loading a 13 GB model that was previously a 30-second
      operation can take < 1 second with mmap=True.
    - Lazy loading: load only the tensors you need (e.g., load one layer
      at a time for quantisation or analysis).
    - Cross-framework: same file can be loaded in PyTorch, TensorFlow,
      JAX, or Flax.

    In code:
    from safetensors.torch import save_file, load_file

    # Save
    save_file(model.state_dict(), 'model.safetensors')

    # Load
    state_dict = load_file('model.safetensors', device='cpu')
    model.load_state_dict(state_dict)

    # Memory-mapped load (large models)
    from safetensors import safe_open
    with safe_open('model.safetensors', framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)   # loaded on demand

    SafeTensors is now the default format for Hugging Face models
    (all models on the Hub are available in .safetensors in addition to .bin).
    Recommendation: use SafeTensors for any checkpoint distributed publicly.


### ONNX (Open Neural Network Exchange)

    ONNX is a cross-framework, cross-runtime model exchange format.
    It serialises not just weights but also the MODEL GRAPH — the
    computation DAG including all operations.

    USE CASE: deploy a PyTorch model in a non-PyTorch runtime.
    ONNX models can run in:
    - ONNX Runtime (optimised for CPU and GPU inference)
    - TensorRT (NVIDIA GPU acceleration)
    - OpenVINO (Intel CPU/VPU)
    - Core ML (Apple Neural Engine)
    - Mobile frameworks (ONNX for Android/iOS)

    Exporting to ONNX:
    dummy_input = torch.randn(1, 3, 224, 224)   # example input
    torch.onnx.export(
        model,
        dummy_input,
        'model.onnx',
        export_params   = True,           # include weights in export
        opset_version   = 17,             # ONNX opset version
        input_names     = ['input'],
        output_names    = ['output'],
        dynamic_axes    = {               # allow variable batch size
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    IMPORTANT CONSTRAINTS:
    - Model must be in inference mode: model.eval() before export.
    - Only operations supported by the target opset can be exported.
      Custom operations require custom ONNX operators.
    - The exported graph corresponds to ONE forward pass with the dummy
      input. Control flow (if/else based on input shape) is traced, not
      symbolically represented. Dynamic shapes require the dynamic_axes argument.

    Validation after export:
    import onnxruntime as ort
    sess = ort.InferenceSession('model.onnx')
    output = sess.run(None, {'input': dummy_input.numpy()})
    # Compare output with model(dummy_input) — should match closely.


### TorchScript

    TorchScript compiles a PyTorch model to a STATIC GRAPH that can
    run without the Python interpreter.

    TWO MODES:
    TRACING (torch.jit.trace):
    Run the model on dummy input. Record which operations are called.
    The recorded trace becomes the static graph.
    Limitation: if-else control flow based on tensor VALUES is not
    captured correctly — the trace only records the path taken for
    the specific dummy input.

    SCRIPTING (torch.jit.script):
    Statically analyse the Python source code.
    Supports control flow (if/else, loops, recursion).
    Limitation: not all Python code is scriptable — only a subset of
    Python syntax is supported. Some libraries and operations fail.

    torch.jit.trace(model, dummy_input)   # tracing
    torch.jit.script(model)               # scripting

    Saving and loading:
    scripted = torch.jit.script(model)
    scripted.save('model_scripted.pt')
    model = torch.jit.load('model_scripted.pt')

    USE CASES:
    - Mobile deployment (torch.jit models can run on iOS/Android via LibTorch)
    - C++ inference (load with torch::jit::load in C++)
    - Removing Python dependency from the inference stack
    - Performance: TorchScript can be compiled and optimised ahead of time


### Hugging Face Model Hub Format

    Models distributed via the Hugging Face Hub follow a standard structure:
    model_directory/
    ├── config.json               ← architecture configuration
    ├── tokenizer.json            ← tokeniser (for NLP models)
    ├── tokenizer_config.json     ← tokeniser metadata
    ├── model.safetensors         ← weights (safetensors format, preferred)
    │   or pytorch_model.bin      ← weights (pickle, legacy)
    │   or model-00001-of-00002.safetensors  ← sharded (large models)
    │      model-00002-of-00002.safetensors
    │      model.safetensors.index.json      ← shard index
    └── generation_config.json    ← generation parameters (optional)

    Loading from HF Hub:
    from transformers import AutoModel, AutoTokenizer
    model     = AutoModel.from_pretrained('model_directory/')
    tokenizer = AutoTokenizer.from_pretrained('model_directory/')

    Saving to HF Hub format:
    model.save_pretrained('model_directory/')
    tokenizer.save_pretrained('model_directory/')

    Sharding (for models too large for a single file):
    model.save_pretrained('model_directory/', max_shard_size='2GB')
    → Automatically splits into multiple .safetensors files with an index.

    The HF format is now the de facto standard for distributing pre-trained
    transformer models. Any model intended for public distribution or for
    use with the Hugging Face ecosystem should be saved in this format.


##### PART 7 — TRAIN VS INFERENCE MODE AND MODEL EXPORT

### model.train() vs model.eval(): What Changes

These two calls switch the model between TRAINING and INFERENCE modes.
They are not just for documentation — they materially change the
model's computation.

    COMPONENTS THAT CHANGE BEHAVIOUR:

    Dropout:
    TRAIN mode: randomly zeros activations with probability p.
    EVAL mode:  all activations pass through unchanged (no dropout).
    Getting this wrong at inference: output is stochastic and ~p% lower
    in magnitude than expected. Predictions are wrong and irreproducible.

    BatchNorm:
    TRAIN mode: computes mean and variance from the CURRENT MINI-BATCH.
                Updates running_mean and running_var (exponential moving average).
    EVAL mode:  uses the stored running_mean and running_var accumulated
                during training. No update to running statistics.
    Getting this wrong at inference: BN normalises by the batch statistics
    of whatever inference batch it sees, not the training distribution.
    With a batch size of 1, this produces zero normalisation (mean=x,
    var=0) and completely garbage outputs.

    LayerNorm, GroupNorm, InstanceNorm:
    These normalise per-example (not per-batch) and do NOT change
    between train and eval mode. They behave identically in both.

    Diagram 2 — train() vs eval() Behavioural Differences:

    TRAIN mode:                         EVAL mode:
    ┌──────────────────────────┐         ┌───────────────────────────┐
    │ Input → Dropout (random) │         │ Input → (no dropout)      │
    │       → BN (batch stats) │         │       → BN (running stats)│
    │       → output (noisy)   │         │       → output (stable)   │
    └──────────────────────────┘         └───────────────────────────┘

    The switch is GLOBAL — model.eval() recursively sets eval mode for
    ALL submodules, including those nested inside other modules.
    Verify: model.training → True (train mode) or False (eval mode).

    For inference:
    model.eval()                          ← always
    with torch.no_grad():                 ← always (saves memory + compute)
        outputs = model(inputs)


### Verifying a Checkpoint Before Deployment

Before deploying a model to production from a checkpoint, validate:

    STEP 1 — Load and verify no missing or unexpected keys:
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'], strict=True)
    assert len(missing) == 0,    f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    STEP 2 — Verify the model is in eval mode:
    model.eval()
    assert not model.training, "Model not in eval mode"

    STEP 3 — Run a smoke test on a known input:
    with torch.no_grad():
        test_output = model(test_input)
    assert test_output.shape == expected_shape
    assert torch.isfinite(test_output).all(), "Output contains NaN/Inf"

    STEP 4 — Verify against the training validation metric:
    val_metric = evaluate(model, val_loader)
    expected_metric = checkpoint['best_val_metric']
    tolerance = 1e-3  # small floating point variance is acceptable
    assert abs(val_metric - expected_metric) < tolerance, (
        f"Checkpoint metric mismatch: "
        f"expected {expected_metric:.4f}, got {val_metric:.4f}")

    STEP 5 — Check memory footprint:
    param_bytes = sum(p.numel() * p.element_size()
                      for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size()
                       for b in model.buffers())
    print(f"Model size: {(param_bytes + buffer_bytes) / 1e9:.2f} GB")

    STEP 6 — Benchmark inference latency:
    import time
    model = model.to(device)
    warmup_runs = 10
    timed_runs  = 100
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(test_input.to(device))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(timed_runs):
            _ = model(test_input.to(device))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) / timed_runs * 1000
    print(f"Inference latency: {latency_ms:.2f} ms/batch")


##### PART 8 — COMMON CHECKPOINTING BUGS AND THEIR FIXES

### Taxonomy of Common Checkpointing Mistakes

    BUG 1 — Saving model instead of model.module in DDP:
    Symptom: loading the checkpoint adds 'module.' prefix to all keys.
             RuntimeError: Missing key(s): 'layer1.weight'.
    Cause:   torch.save(model.state_dict()) instead of
             torch.save(model.module.state_dict()) in DDP.
    Fix:     Always save model.module.state_dict() in DDP.
             Or strip 'module.' prefix when loading:
             state_dict = {k[7:] if k.startswith('module.') else k: v
                           for k, v in checkpoint['model_state_dict'].items()}

    BUG 2 — Not calling model.eval() before inference:
    Symptom: inference output is stochastic (different every call).
             Inference performance is worse than training validation.
    Cause:   Dropout is still active in train mode.
    Fix:     Always call model.eval() after loading a checkpoint for inference.

    BUG 3 — Loading checkpoint weights but not optimiser state:
    Symptom: resumed training is slower than original training for
             the first 10-50 epochs. Performance temporarily regresses.
    Cause:   Adam's moment estimates were reset. The optimiser restarts
             from zero gradient history.
    Fix:     Always restore optimiser state dict when resuming training.
             Only skip optimiser state when fine-tuning on a new task.

    BUG 4 — Loading checkpoint on wrong device:
    Symptom: RuntimeError: Expected all tensors to be on the same device.
             Or: training is extremely slow (CPU-GPU transfers on every step).
    Cause:   checkpoint = torch.load(path) without map_location.
             Tensors are loaded onto the device they were saved on.
    Fix:     torch.load(path, map_location=device) always.
             Then move optimiser states manually to device.

    BUG 5 — Overwriting a good checkpoint with a bad one:
    Symptom: the best-epoch checkpoint is gone; only a later, worse
             checkpoint exists.
    Cause:   checkpoint_manager.save() called every epoch, overwriting
             the best model with a later overfitted model.
    Fix:     Separate periodic checkpoints (for resume) from best-model
             checkpoint (saved only on metric improvement). Never overwrite
             the best checkpoint path with periodic checkpoints.

    BUG 6 — Comparing model after loading to pre-load:
    Symptom: torch.load → model.load_state_dict → model outputs differ
             slightly from the saved model.
    Cause:   BatchNorm running statistics are not fully deterministic
             across devices or precision formats. Also: model was saved
             in train mode (running stats still updating during inference).
    Fix:     Save in eval mode (call model.eval() before saving for
             the deployment checkpoint). Validate by comparing outputs
             on a fixed input before and after save/load.

    BUG 7 — Not saving the epoch/step count:
    Symptom: resumed training restarts LR schedule from epoch 0.
             LR jumps back to LR_max in cosine annealing.
    Cause:   Checkpoint was saved without 'epoch' or 'step' fields.
    Fix:     Always include epoch and step in the checkpoint dict.
             Always restore these and pass them to the scheduler on load.

    BUG 8 — Partial write leaving corrupt checkpoint:
    Symptom: torch.load fails with: RuntimeError: PytorchStreamReader
             failed reading zip archive.
    Cause:   Process was killed during torch.save().
    Fix:     Always use atomic writes (save to .tmp, then os.replace).
             Validate checkpoint integrity on load:
             try:
                 ckpt = torch.load(path)
             except Exception as e:
                 print(f"Corrupt checkpoint {path}: {e}")
                 # Fall back to second-most-recent checkpoint


##### PART 9 — THE COMPLETE CHECKPOINTING SYSTEM

### Putting It All Together: Training Loop with Full Checkpointing

    # ── Setup ────────────────────────────────────────────────────────────
    checkpoint_manager = CheckpointManager(
        save_dir  = 'checkpoints/run_001',
        keep_last = 3,
    )
    best_model_tracker = BestModelTracker(
        save_path = 'checkpoints/run_001/best_model.pt',
        mode      = 'min',   # minimise val loss
        min_delta = 1e-4,
    )
    scaler = torch.cuda.amp.GradScaler()   # for FP16

    # Resume from checkpoint if one exists
    start_epoch = 0
    start_step  = 0
    latest = checkpoint_manager.latest()
    if latest:
        meta = load_checkpoint(latest, model, optimiser, scheduler, scaler)
        start_epoch = meta['epoch'] + 1
        start_step  = meta['step']
        print(f"Resumed from {latest} (epoch {start_epoch}, step {start_step})")

    # ── Training loop ────────────────────────────────────────────────────
    global_step = start_step
    for epoch in range(start_epoch, config['max_epochs']):

        # Training phase
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['grad_clip'])
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            # Periodic checkpoint (every N steps)
            if global_step % config['ckpt_every_steps'] == 0:
                checkpoint_manager.save(
                    model, optimiser, scheduler, scaler,
                    epoch, global_step, best_model_tracker.best_metric, config)

        # Validation phase
        model.eval()
        val_loss = run_validation(model, val_loader, criterion, device)

        # Best-model tracking
        best_model_tracker.step(
            val_loss, model, optimiser, scheduler, scaler,
            epoch, global_step, config)

        # Epoch-level periodic checkpoint
        checkpoint_manager.save(
            model, optimiser, scheduler, scaler,
            epoch, global_step, val_loss, config)

        print(f"Epoch {epoch}: val_loss={val_loss:.4f} | "
              f"{best_model_tracker.summary()}")

    # ── Final export ─────────────────────────────────────────────────────
    # Load best model for deployment
    best_ckpt = torch.load('checkpoints/run_001/best_model.pt',
                           map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()
    # Validate, export to SafeTensors or ONNX as needed.


### Checkpointing Decision Summary

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Decision                    │ Recommendation                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │ What to save                │ Model + optimiser + scheduler +       │
    │                             │ scaler + epoch + step + RNG states    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ How to save                 │ Atomic write (.tmp then os.replace)   │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Format (training resume)    │ .pt (PyTorch state dict)              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Format (public distribution)│ SafeTensors                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Format (cross-framework)    │ ONNX                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Format (C++/mobile)         │ TorchScript                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Format (HF ecosystem)       │ Hugging Face save_pretrained()        │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Periodic save frequency     │ Every 30-60 min (or every N steps)    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Retention policy            │ Keep last 3 + always keep best model  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ DDP: who saves              │ Only rank 0; save model.module, not   │
    │                             │ model                                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │ DDP: synchronisation        │ dist.barrier() before and after save  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Large models (FSDP)         │ FullStateDictConfig with              │
    │                             │ offload_to_cpu=True, rank0_only=True  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Inference deployment        │ Load best model, call model.eval(),   │
    │                             │ validate outputs, benchmark latency   │
    └─────────────────────────────────────────────────────────────────────┘

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
        from Training_Core.visuals.checkpoint_visualization import (
            CHECKPOINTING_VISUAL_HTML,
            CHECKPOINTING_VISUAL_HEIGHT,
        )
        visual_html   = CHECKPOINTING_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = CHECKPOINTING_VISUAL_HEIGHT
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