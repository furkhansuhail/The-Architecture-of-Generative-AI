"""
PyTorch Lightning — The Research-to-Production Training Framework
=================================================================

PyTorch Lightning is an open-source framework built on top of PyTorch by
William Falcon (then a PhD student at NYU) and released in 2019. Its core
insight is devastatingly simple: every deep learning training script is
roughly 80% identical engineering boilerplate — training loops, gradient
zeroing, device placement, mixed precision, checkpointing, logging, DDP
setup — and about 20% novel research code.

Lightning separates these two concerns cleanly:
    - LightningModule: the WHAT (your model, loss, optimiser, metrics)
    - Trainer:         the HOW (the training loop, hardware, precision, logging)

The result is that your research code becomes hardware-agnostic, reproducible,
and production-ready without any extra effort. A model written in Lightning
runs on a single CPU, a single GPU, 8 GPUs with DDP, 64 GPUs across nodes,
or a TPU cluster by changing exactly one Trainer argument.

Understanding Lightning's design teaches you the best practices that the
community has converged on after years of painful debugging: why you should
never call .backward() yourself, why logging must be done through the
framework, why data loading belongs in its own class, and why the hook
system is the correct abstraction for all training customisation.

This module covers Lightning's full architecture — LightningModule, Trainer,
LightningDataModule, Callbacks, the logging system, TorchMetrics, mixed
precision, gradient strategies, distributed training, and the deployment
pipeline from training to TorchScript to ONNX.

"""

import textwrap
import re

TOPIC_NAME   = "PyTorch Lightning — Research-to-Production Training Framework"
DISPLAY_NAME = "05 · PyTorch Lightning"
ICON         = "⚡"
SUBTITLE     = "From Clean Research Code to Scalable Distributed Training"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT PYTORCH LIGHTNING SOLVES AND WHY IT EXISTS

### The Raw PyTorch Training Loop Problem

    Every raw PyTorch project eventually accumulates the same scaffolding:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = MyModel().to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)   # device management
                opt.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                for X, y in val_loader:              # validation loop
                    X, y = X.to(device), y.to(device)
                    val_loss += criterion(model(X), y).item()

            scheduler.step()
            if val_loss < best:
                torch.save(model.state_dict(), 'best.pt')   # checkpointing
            if patience_exceeded:
                break                                        # early stopping

    This code is:
        - Not reproducible  (different engineers write it differently)
        - Not hardware-portable  (you must manually handle GPU/TPU/multi-GPU)
        - Fragile  (easy to forget .zero_grad(), .train()/.eval(), .to(device))
        - Untestable  (business logic tangled with engineering code)
        - Verbose  (300 lines of scaffolding per experiment)

### Lightning's Core Decomposition

    Lightning decomposes the training script into three distinct concerns:

    ┌────────────────────────────────────────────────────────────────────┐
    │  LightningModule      What your model does                         │
    │                       forward(), training_step(), val_step()       │
    │                       configure_optimizers()                       │
    │                       The research code. Hardware-free.            │
    ├────────────────────────────────────────────────────────────────────┤
    │  LightningDataModule  Where your data comes from                   │
    │                       prepare_data(), setup()                      │
    │                       train_dataloader(), val_dataloader()         │
    │                       Reusable across experiments.                 │
    ├────────────────────────────────────────────────────────────────────┤
    │  Trainer              How training happens                         │
    │                       Hardware, precision, epochs, logging         │
    │                       Callbacks, strategies, profiling             │
    │                       Zero research code. Pure engineering.        │
    └────────────────────────────────────────────────────────────────────┘

    The Trainer handles EVERYTHING below the line:
        - Device placement (.to(device) is called automatically)
        - gradient zeroing (automatically before each step)
        - loss.backward() (called for you)
        - optimizer.step() (called for you)
        - model.train() / model.eval() mode switching
        - torch.no_grad() in validation
        - Metric logging and aggregation across batches/epochs
        - Checkpointing (best model, periodic, last model)
        - Early stopping
        - Gradient clipping
        - Mixed precision (AMP)
        - DDP / FSDP / DeepSpeed distributed training
        - Progress bars and epoch summaries

### The "Two Files" Principle

    A well-structured Lightning project has exactly two files of research code:
        model.py:   the LightningModule with your architecture and training logic
        data.py:    the LightningDataModule with your data loading

    And one file of configuration:
        train.py:   instantiate model + data, create Trainer, call .fit()

    This separation means:
        - You can swap models without touching data loading
        - You can swap datasets without touching the model
        - You can scale from 1 GPU to 64 by editing one Trainer argument
        - Unit testing is trivial (test the LightningModule directly)

### Lightning vs Keras: The Key Philosophical Difference

    Keras:      optimised for simplicity and fast onboarding.
                model.compile() hides the training loop entirely.
                Less flexible for novel training paradigms.

    Lightning:  optimised for research flexibility AND production robustness.
                You still write the training_step() — it's your code.
                Lightning just automates the boilerplate AROUND your code.
                GANs, RL loops, meta-learning, custom backward passes all work.

    The test: can you implement MAML (model-agnostic meta-learning) with
    second-order gradients? In Keras: painful. In Lightning: yes, with
    manual_backward() and manual optimization mode.


##### PART 2 — LIGHTNINGMODULE: THE RESEARCH CORE

### What LightningModule Is

    LightningModule is a subclass of nn.Module. It IS your PyTorch model —
    all nn.Parameter, all nn.Module submodules, everything you know from
    raw PyTorch carries over unchanged. What LightningModule adds is a
    structured lifecycle of methods that the Trainer calls at the right times.

    The minimal required interface:

        class MyModel(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(32, 10)

            def forward(self, x):              # optional — for inference
                return self.net(x)

            def training_step(self, batch, batch_idx):    # REQUIRED
                x, y = batch
                loss = F.cross_entropy(self.net(x), y)
                return loss                    # Lightning calls .backward()

            def configure_optimizers(self):    # REQUIRED
                return torch.optim.AdamW(self.parameters(), lr=1e-3)

    That's it. No device management. No zero_grad(). No backward(). No step().
    The Trainer handles all of that.

### The Step Methods: training_step, validation_step, test_step

    training_step(self, batch, batch_idx):
        Called on every training batch. The return value MUST be:
        - A scalar loss tensor (Lightning calls .backward() on it)
        - A dict with key 'loss' (and optionally other tensors to log)
        - None (only if using manual optimization)

        def training_step(self, batch, batch_idx):
            x, y    = batch
            logits  = self(x)                  # calls self.forward(x)
            loss    = F.cross_entropy(logits, y)
            acc     = (logits.argmax(1) == y).float().mean()
            self.log('train_loss', loss)        # log to all configured loggers
            self.log('train_acc',  acc)
            return loss

    validation_step(self, batch, batch_idx):
        Called on every validation batch. Return value is optional.
        Trainer automatically calls model.eval() and torch.no_grad().

        def validation_step(self, batch, batch_idx):
            x, y   = batch
            logits = self(x)
            loss   = F.cross_entropy(logits, y)
            acc    = (logits.argmax(1) == y).float().mean()
            self.log('val_loss', loss, prog_bar=True)  # show in progress bar
            self.log('val_acc',  acc,  prog_bar=True)

    test_step(self, batch, batch_idx):
        Called by trainer.test(). Identical structure to validation_step.
        Convention: run test_step only after full training is complete.

    predict_step(self, batch, batch_idx):
        Called by trainer.predict(). Used for generating predictions on
        new data. No labels expected. Returns predictions.

        def predict_step(self, batch, batch_idx):
            x = batch
            return torch.softmax(self(x), dim=-1)

### configure_optimizers: The Optimiser and Scheduler Contract

    The most flexible method in Lightning. Can return:

        # 1. Single optimiser (most common)
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        # 2. Optimiser + LR scheduler (dict form — explicit and recommended)
        def configure_optimizers(self):
            opt   = torch.optim.AdamW(self.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
            return {
                'optimizer':    opt,
                'lr_scheduler': {
                    'scheduler': sched,
                    'interval':  'epoch',    # or 'step'
                    'frequency': 1,
                    'monitor':   'val_loss', # for ReduceLROnPlateau
                },
            }

        # 3. Multiple optimisers (GANs, multi-objective training)
        def configure_optimizers(self):
            opt_g = torch.optim.Adam(self.generator.parameters(),     lr=2e-4)
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)
            return [opt_g, opt_d], []   # ([optimisers], [schedulers])
            # training_step receives optimizer_idx to know which to use

    The scheduler dict keys:
        scheduler:    the LRScheduler object
        interval:     'epoch' (default) or 'step'
        frequency:    call scheduler every N intervals (default 1)
        monitor:      metric name to monitor (for ReduceLROnPlateau)
        strict:       raise error if monitored metric not found (default True)
        name:         name for logging the lr (default 'lr')

### forward() vs training_step(): The Critical Distinction

    This is the most common Lightning confusion.

    forward():
        Defines what the module COMPUTES for a given input.
        Used for INFERENCE — when you call model(x) externally.
        Should be the minimal computation needed for prediction.
        Optional: only needed if you call self(x) or use model(x) externally.

    training_step():
        Defines one training iteration, including loss computation.
        This is where you call self(x) or self.forward(x) AND add loss logic.

    The correct pattern:
        class Classifier(L.LightningModule):
            def forward(self, x):
                return self.net(x)          # returns logits

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)            # calls forward()
                return F.cross_entropy(logits, y)

            def predict_step(self, batch, batch_idx):
                x = batch
                return torch.softmax(self(x), dim=-1)  # uses forward()

### LightningModule Hooks: Full Lifecycle

    Lightning calls these hooks at specific points. Override any you need:

    SETUP:
        __init__()                      model construction
        setup(stage)                    after DM setup, before training starts
                                        stage: 'fit', 'validate', 'test', 'predict'
        on_fit_start()                  called at the start of fit()

    TRAINING:
        on_train_epoch_start()
        on_train_batch_start(batch, batch_idx)
        training_step(batch, batch_idx)
        on_before_backward(loss)        before loss.backward()
        on_after_backward()             after loss.backward(), before optimizer.step()
        on_before_optimizer_step(opt)
        on_train_batch_end(out, batch, batch_idx)
        on_train_epoch_end()            after all training batches in an epoch

    VALIDATION:
        on_validation_epoch_start()
        validation_step(batch, batch_idx)
        on_validation_epoch_end()       aggregate metrics here

    TEARDOWN:
        on_fit_end()                    after training is complete
        teardown(stage)                 cleanup

    The most useful hooks in practice:
        on_train_epoch_end()    — compute epoch-level metrics, log histograms
        on_validation_epoch_end() — compute confusion matrix, generate samples
        on_fit_end()            — save final artefacts, notify experiment tracker


### Manual Optimization Mode

    For GANs, RL, and any training that requires multiple optimisers, loss
    computations, or backward passes per step, switch to manual mode:

        class GAN(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.automatic_optimization = False    # ← TURN OFF AUTOMATION

            def training_step(self, batch, batch_idx):
                opt_g, opt_d = self.optimizers()      # get both optimisers

                # --- Discriminator step ---
                opt_d.zero_grad()
                real = batch
                fake = self.generator(torch.randn(len(real), self.latent_dim))
                loss_d = -torch.mean(self.discriminator(real)) \
                         + torch.mean(self.discriminator(fake))
                self.manual_backward(loss_d)           # ← NOT loss.backward()
                opt_d.step()

                # --- Generator step ---
                opt_g.zero_grad()
                fake2 = self.generator(torch.randn(len(real), self.latent_dim))
                loss_g = -torch.mean(self.discriminator(fake2))
                self.manual_backward(loss_g)
                opt_g.step()

                self.log_dict({'loss_d': loss_d, 'loss_g': loss_g})

    Rules for manual optimization:
        - Always use self.manual_backward(loss) instead of loss.backward()
        - Call self.toggle_optimizer() for efficient gradient routing in GANs
        - Gradient clipping: self.clip_gradients(opt, gradient_clip_val=1.0)
        - LR schedulers: call self.lr_schedulers().step() manually


##### PART 3 — LIGHTNINGDATAMODULE: ENCAPSULATED DATA PIPELINES

### Why Encapsulate Data Logic?

    Without LightningDataModule, data loading logic lives wherever the model
    is defined. This creates:
        - Coupling: model code and data code intertwined
        - Fragility: DDP and multi-node training require careful data sharding;
          done wrong, every GPU sees the same data
        - Non-reusability: you cannot swap the dataset without touching the model
        - Non-portability: every machine needs the same hardcoded file paths

    LightningDataModule solves all of this by bundling:
        1. Data download / preparation (runs ONCE globally)
        2. Data splitting and dataset construction (runs ONCE per process)
        3. DataLoader creation (re-run every fit/test/predict call)

### The Three Critical Hooks

    prepare_data(self):
        Called ONCE on the main process only (not on DDP workers).
        Use for: downloading, extracting, format conversion.
        NEVER assign state here (self.x = ...) — only the main process
        runs this method; worker processes would miss the assignments.

        def prepare_data(self):
            MNIST(root='.', train=True,  download=True)   # download only
            MNIST(root='.', train=False, download=True)

    setup(self, stage: str):
        Called on EVERY process (all DDP workers) after prepare_data.
        Use for: constructing Dataset objects, splitting train/val, transforms.
        stage is one of: 'fit', 'validate', 'test', 'predict'.

        def setup(self, stage):
            if stage == 'fit':
                full = MNIST(root='.', train=True, transform=self.train_tf)
                n    = int(0.9 * len(full))
                self.train_ds, self.val_ds = random_split(full, [n, len(full)-n])
            if stage in ('fit', 'test'):
                self.test_ds = MNIST(root='.', train=False, transform=self.val_tf)

    train_dataloader / val_dataloader / test_dataloader / predict_dataloader:
        Return the configured DataLoader. Called by the Trainer at the
        appropriate stage. Each DDP worker gets its own DataLoader shard.

        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

### Why prepare_data vs setup Matters for DDP

    In DDP training, N processes (one per GPU) all run simultaneously.
    If prepare_data() ran on all processes, you would have N simultaneous
    downloads / file writes competing for the same files — race conditions.
    Lightning guarantees prepare_data() runs on rank 0 only, then all
    processes wait at a barrier before setup() is called.

    The memory also matters:
        prepare_data():  never assign to self (rank-0 only — assignments lost)
        setup():         safe to assign self.train_ds, self.val_ds, etc.
                         each process needs its own dataset object

### DataModule as a Reusable Component

    A well-written DataModule is fully self-contained and reusable:

        # Use the same data with a different model
        dm = MNISTDataModule(batch_size=128, num_workers=4)

        trainer1.fit(model_a, dm)   # VGG
        trainer2.fit(model_b, dm)   # ResNet

        # Use the same model with different data
        trainer.fit(model, CIFARDataModule())
        trainer.fit(model, FashionMNISTDataModule())

    The DataModule can be saved as a checkpoint too:
        datamodule = dm.save_hyperparameters()
        # Restoring from checkpoint loads DataModule state as well


##### PART 4 — TRAINER: THE AUTOMATION ENGINE

### Trainer as a Configuration Object

    The Trainer is where ALL engineering decisions live. It takes no model code
    — only configuration of how to run training:

        trainer = L.Trainer(
            # Core training settings
            max_epochs         = 100,
            min_epochs         = 10,
            max_steps          = -1,          # -1 = no limit by steps

            # Hardware
            accelerator        = 'gpu',       # 'cpu', 'gpu', 'tpu', 'mps', 'auto'
            devices            = 4,           # number of GPUs (or 'auto')
            num_nodes          = 1,           # number of machines

            # Precision
            precision          = '16-mixed',  # '32-true', '16-mixed', 'bf16-mixed'

            # Distributed strategy
            strategy           = 'ddp',       # 'dp', 'ddp', 'fsdp', 'deepspeed'

            # Logging
            log_every_n_steps  = 10,
            logger             = TensorBoardLogger('logs/'),

            # Callbacks (early stopping, checkpointing, etc.)
            callbacks          = [early_stop, checkpoint],

            # Performance
            gradient_clip_val      = 1.0,
            accumulate_grad_batches = 4,

            # Debugging
            fast_dev_run       = False,  # True = 1 batch train + 1 val (sanity check)
            overfit_batches    = 0,      # > 0 = overfit on this many batches (debug)
            limit_train_batches = 1.0,   # float = fraction, int = number of batches
            limit_val_batches  = 1.0,
            profiler           = 'simple',   # 'simple', 'advanced', PyTorch profiler
        )

### The fast_dev_run Flag — The Most Underused Feature

    Setting fast_dev_run=True or fast_dev_run=N (int):
        - Runs exactly N batches of training and N batches of validation
        - Checks that your entire pipeline (model, data, logging, callbacks)
          runs end-to-end without errors
        - Does NOT save checkpoints or logs
        - Turns off most hooks that would interfere

    This is the first thing you should run on any new model.
    Catches: shape mismatches, device errors, missing keys in batch,
             misconfigured loss functions, NaN issues.

    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule)   # runs 1 train step + 1 val step → pass/fail

### Precision Modes

    '32-true':        full precision float32 (default)
    '16-mixed':       AMP float16 with GradScaler (Volta/Turing GPUs)
    'bf16-mixed':     AMP bfloat16 without GradScaler (Ampere/Hopper GPUs)
    '16-true':        pure float16 (weights stored in f16 — risky for training)
    'bf16-true':      pure bfloat16 (weights stored in bf16 — use with FSDP)
    'transformer-engine': Hopper H100 FP8 training via NVIDIA Transformer Engine

    Lightning handles the GradScaler automatically when '16-mixed' is set.
    You never call scaler.scale(loss).backward() — Lightning does it.
    Your training_step() returns loss as a normal float32 scalar.

### Trainer.fit() vs .validate() vs .test() vs .predict()

    trainer.fit(model, datamodule):
        Runs the full training loop. Calls training_step every batch,
        validation_step every val_check_interval epochs.
        Checkpoints, early stopping, and callbacks are all active.

    trainer.validate(model, datamodule):
        Runs ONLY the validation loop once (no training).
        Useful: evaluate a pre-trained checkpoint on a new val set.

    trainer.test(model, datamodule):
        Runs ONLY the test loop once.
        Convention: call ONLY after training is complete.
        Prevents test set contamination in checkpoint selection.

    trainer.predict(model, datamodule):
        Runs predict_step() on every batch in predict_dataloader.
        Returns a list of all batch outputs (gathered across DDP processes).
        Use for generating predictions on new data.

### Validation Frequency Control

    val_check_interval:
        float (0.0–1.0): validate every fraction of training epoch
            val_check_interval=0.25 → validate 4× per epoch
        int (> 1): validate every N training steps
            val_check_interval=500  → validate every 500 steps

    check_val_every_n_epoch (default 1):
        int: validate every N full training epochs
        Useful for slow validation (large val sets, expensive metrics like FID)

    num_sanity_val_steps (default 2):
        Run N validation batches before training starts.
        Catches: val_dataloader issues, metric bugs, device mismatches.
        Set to 0 to disable.


##### PART 5 — CALLBACKS: THE HOOK SYSTEM

### Callbacks vs Module Hooks

    Both LightningModule and Callbacks define hooks (on_train_epoch_end, etc.).
    The difference is conceptual:

        LightningModule hooks: for things that are PART OF the model's logic
            Example: computing a custom metric that uses model internals
            Example: generating sample images in a VAE (uses self.decode())

        Callback hooks: for things that are EXTERNAL to the model
            Example: logging to an external system
            Example: pruning weights
            Example: adjusting the learning rate based on gradient norms
            Example: saving the model in a custom format

    This separation keeps the model pure and the callbacks reusable.
    You can attach the same ModelCheckpoint or GradNormLogger to any model.

### Built-In Callbacks

    ModelCheckpoint:
        Saves model weights (and optionally the full state) at specified intervals.

        from lightning.pytorch.callbacks import ModelCheckpoint

        ckpt = ModelCheckpoint(
            dirpath        = 'checkpoints/',
            filename       = '{epoch}-{val_loss:.4f}',   # templated name
            monitor        = 'val_loss',
            mode           = 'min',          # save when val_loss decreases
            save_top_k     = 3,              # keep 3 best checkpoints
            save_last      = True,           # always save last.ckpt
            every_n_epochs = 1,
        )

        # Restore from checkpoint:
        model = MyModel.load_from_checkpoint('checkpoints/epoch=5-val_loss=0.23.ckpt')

    EarlyStopping:
        Stops training when a monitored metric stops improving.

        from lightning.pytorch.callbacks import EarlyStopping

        es = EarlyStopping(
            monitor   = 'val_loss',
            patience  = 10,          # wait 10 epochs before stopping
            min_delta = 1e-4,        # minimum change to count as improvement
            mode      = 'min',
            verbose   = True,
        )

    LearningRateMonitor:
        Logs the current learning rate at every step or epoch.
        Essential when using warmup or annealing schedules.

        from lightning.pytorch.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

    RichProgressBar:
        Replaces the default tqdm progress bar with a rich-formatted one.
        Shows multiple metrics, ETA, batch/sec in a clean table.

    StochasticWeightAveraging (SWA):
        After swa_epoch_start, maintains a running average of model weights.
        The SWA model often generalises better than the final iterate.
        Uses a higher lr (swa_lrs) for the SWA phase.

        from lightning.pytorch.callbacks import StochasticWeightAveraging
        swa = StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=0.75)

### Custom Callbacks: The Full Hook Interface

    Every Callback can implement any subset of these hooks:

        class MyCallback(L.Callback):
            # ── Fit lifecycle ─────────────────────────────────────
            def on_fit_start(self, trainer, pl_module): ...
            def on_fit_end(self, trainer, pl_module): ...

            # ── Training hooks ────────────────────────────────────
            def on_train_start(self, trainer, pl_module): ...
            def on_train_epoch_start(self, trainer, pl_module): ...
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): ...
            def on_before_backward(self, trainer, pl_module, loss): ...
            def on_after_backward(self, trainer, pl_module): ...
            def on_before_optimizer_step(self, trainer, pl_module, optimizer): ...
            def on_before_zero_grad(self, trainer, pl_module, optimizer): ...
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): ...
            def on_train_epoch_end(self, trainer, pl_module): ...
            def on_train_end(self, trainer, pl_module): ...

            # ── Validation hooks ──────────────────────────────────
            def on_validation_start(self, trainer, pl_module): ...
            def on_validation_epoch_start(self, trainer, pl_module): ...
            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): ...
            def on_validation_epoch_end(self, trainer, pl_module): ...
            def on_validation_end(self, trainer, pl_module): ...

            # ── Checkpoint hooks ──────────────────────────────────
            def on_save_checkpoint(self, trainer, pl_module, checkpoint): ...
            def on_load_checkpoint(self, trainer, pl_module, checkpoint): ...

            # ── Exception hooks ───────────────────────────────────
            def on_exception(self, trainer, pl_module, exception): ...

    Practical custom callback examples:
        GradientNormLogger:  log ||∇θ||₂ to detect exploding/vanishing grads
        FreezeUntilEpoch:    freeze backbone for N epochs, then unfreeze
        SampleGenerator:     generate and log example images every N epochs
        ExperimentAlerter:   send Slack/email notification when training ends


##### PART 6 — LOGGING AND METRICS: self.log AND TORCHMETRICS

### The self.log API

    The single function for all logging inside a LightningModule:

        self.log(
            name,                    # string metric name
            value,                   # scalar tensor or Python number
            prog_bar = False,        # show in progress bar
            logger   = True,         # send to the configured logger
            on_step  = True,         # log at every step (batch)
            on_epoch = False,        # log mean over the epoch
            reduce_fx = 'mean',      # 'mean', 'sum', 'min', 'max'
            sync_dist = False,       # synchronise across DDP processes
        )

    Default behaviour by stage:
        training_step:      on_step=True,  on_epoch=False
        validation_step:    on_step=False, on_epoch=True   ← averages across batches
        test_step:          on_step=False, on_epoch=True

    Self-log with step and epoch:
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        # Creates: 'train_loss_step' (per-batch) AND 'train_loss_epoch' (epoch mean)

    Logging a dict (cleaner for many metrics):
        self.log_dict({
            'train_loss': loss,
            'train_acc':  acc,
            'train_f1':   f1,
        })

### The sync_dist Pitfall in DDP

    In DDP, each process runs validation_step on a DIFFERENT subset of the
    val set. If you log without sync_dist=True, each GPU reports its own
    metric value — they won't match and the val_loss displayed will be
    whichever GPU's value happens to be logged last.

    Always set sync_dist=True in validation_step when training with DDP:

        def validation_step(self, batch, batch_idx):
            loss = self.compute_loss(batch)
            self.log('val_loss', loss,
                     sync_dist=True,   # AllReduce across DDP processes
                     on_epoch=True)

    For training_step, the Trainer already averages the loss before logging.
    sync_dist is only critical in validation_step and test_step.

### TorchMetrics: The Correct Way to Compute Metrics

    The naive approach — accumulating metric values across batches — is wrong:

        # WRONG: averaging accuracy over batches, not samples
        accs = []
        for x, y in loader:
            accs.append((model(x).argmax(1) == y).float().mean())
        total_acc = sum(accs) / len(accs)    # WRONG if last batch < batch_size!

    TorchMetrics solves this by maintaining running state across batches:

        import torchmetrics

        class Classifier(L.LightningModule):
            def __init__(self, n_classes):
                super().__init__()
                # Metrics are nn.Modules — they move to GPU with model.to(device)
                self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)
                self.val_acc   = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)
                self.val_f1    = torchmetrics.F1Score(task='multiclass', num_classes=n_classes)

            def training_step(self, batch, batch_idx):
                x, y   = batch
                logits = self(x)
                loss   = F.cross_entropy(logits, y)
                preds  = logits.argmax(dim=1)
                self.train_acc(preds, y)                     # update metric state
                self.log('train_acc', self.train_acc,        # log the metric object
                         on_step=False, on_epoch=True)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y   = batch
                logits = self(x)
                preds  = logits.argmax(dim=1)
                self.val_acc.update(preds, y)
                self.val_f1.update(preds, y)

            def on_validation_epoch_end(self):
                self.log('val_acc', self.val_acc.compute())
                self.log('val_f1',  self.val_f1.compute())
                self.val_acc.reset()
                self.val_f1.reset()

    Passing the metric object (not its .compute() value) to self.log() is
    the Lightning-recommended pattern: Lightning handles calling .compute()
    at the right time and .reset() between epochs automatically.

### Available Loggers

    TensorBoardLogger (default):
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger('tb_logs/', name='my_experiment')
        # tensorboard --logdir tb_logs/

    WandbLogger (Weights & Biases — preferred for large teams):
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project='my_project', name='run_1')
        # Automatically logs: metrics, hyperparameters, system usage, model graph

    CSVLogger (offline, always available):
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger('csv_logs/')
        # Writes metrics/epoch_N.csv — readable anywhere

    MLflowLogger, CometLogger, NeptuneLogger also available.

    Multiple loggers simultaneously:
        trainer = L.Trainer(logger=[TensorBoardLogger('tb/'), WandbLogger()])

### save_hyperparameters() — The Reproducibility Primitive

    Call self.save_hyperparameters() in __init__ to:
        1. Store all __init__ arguments in self.hparams
        2. Automatically log them to the experiment tracker
        3. Save them in the checkpoint — restoring a model from checkpoint
           automatically reconstructs the model with the same hyperparameters

        class MyModel(L.LightningModule):
            def __init__(self, lr=1e-3, hidden=128, dropout=0.3):
                super().__init__()
                self.save_hyperparameters()           # saves lr, hidden, dropout
                # Access via: self.hparams.lr, self.hparams.hidden
                self.net = nn.Sequential(
                    nn.Linear(32, self.hparams.hidden),
                    nn.Dropout(self.hparams.dropout),
                    nn.Linear(self.hparams.hidden, 10),
                )

        # Reload model — no need to know the constructor arguments:
        model = MyModel.load_from_checkpoint('checkpoints/best.ckpt')
        # The checkpoint contains lr, hidden, dropout — model reconstructs itself


##### PART 7 — ADVANCED TRAINING: AMP, GRADIENT CONTROL, LR FINDER

### Mixed Precision in Lightning

    Lightning abstracts the entire AMP setup behind a single Trainer flag:

        trainer = L.Trainer(precision='16-mixed')  # f16 + GradScaler (auto)
        trainer = L.Trainer(precision='bf16-mixed') # bf16, no GradScaler needed

    Under the hood, Lightning:
        1. Creates a torch.cuda.amp.GradScaler (for '16-mixed' only)
        2. Wraps the training_step body with autocast()
        3. Calls scaler.scale(loss).backward()
        4. Calls scaler.unscale_(optimizer) before gradient clipping
        5. Calls scaler.step(optimizer) and scaler.update()

    You see NONE of this in your training_step. It just works.

    When to use which:
        RTX 2080/3080/V100:  precision='16-mixed'    (no bfloat16 support)
        A100/H100/RTX 4090:  precision='bf16-mixed'  (bfloat16 tensor cores)
        TPU (v3/v4):         precision='bf16-mixed'  (XLA bfloat16)
        Apple M-series:      precision='16-mixed'    (MPS backend)

### Gradient Accumulation

    Simulate a larger effective batch size without increasing memory:

        trainer = L.Trainer(accumulate_grad_batches=4)
        # Effective batch = batch_size × 4
        # optimizer.step() called every 4 batches (not every batch)

    Lightning handles:
        - Calling optimizer.step() and zero_grad() only every N steps
        - Scaling the loss correctly (divides by N automatically)
        - Correctly managing the progress bar (shows effective batch count)

    Dynamically change accumulation mid-training (rare, for curriculum learning):
        trainer = L.Trainer(accumulate_grad_batches={0: 8, 4: 4, 8: 1})
        # epoch 0-3: accumulate 8; epochs 4-7: 4; epoch 8+: 1

### Gradient Clipping

    Three modes in Lightning:

        # 1. Clip by global norm (recommended)
        trainer = L.Trainer(gradient_clip_val=1.0,
                             gradient_clip_algorithm='norm')

        # 2. Clip by value (clip each individual gradient)
        trainer = L.Trainer(gradient_clip_val=0.5,
                             gradient_clip_algorithm='value')

        # 3. Manual clipping in module hook (for monitoring)
        def on_before_optimizer_step(self, optimizer):
            norm = self.compute_grad_norm()
            self.log('grad_norm', norm)
            if norm > 100:                    # log explosion alerts
                self.log('grad_exploded', 1.0)

### The LR Finder (Learning Rate Range Test)

    Smith's LR Range Test: trains for a few mini-batches while exponentially
    increasing the learning rate from very small to very large. Plots the
    loss vs lr curve. The optimal lr is slightly before the loss starts rising.

        from lightning.pytorch.tuner import Tuner

        trainer = L.Trainer(max_epochs=1)
        tuner   = Tuner(trainer)

        lr_finder = tuner.lr_find(model, datamodule=dm,
                                   min_lr=1e-8, max_lr=1e-1,
                                   num_training_steps=200)

        fig = lr_finder.plot(suggest=True)   # plot the loss vs lr curve
        fig.savefig('lr_finder.png')

        suggested_lr = lr_finder.suggestion()
        model.hparams.lr = suggested_lr       # update model's lr
        model.configure_optimizers()          # reconfigure

    The suggestion is the lr at the steepest descent point of the loss curve.
    Typical result: suggested_lr is 10–100× smaller than the loss minimum.

### Batch Size Finder

    Tuner can also find the maximum batch size that fits in GPU memory:

        from lightning.pytorch.tuner import Tuner
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode='power')
        # mode='power': tries 1, 2, 4, 8, 16, 32, ... until OOM
        # mode='binsearch': binary search between min and max

        # After this call, model.hparams.batch_size is set to the max
        dm.batch_size = model.hparams.batch_size


##### PART 8 — DISTRIBUTED TRAINING AND DEPLOYMENT

### Scaling from 1 GPU to N: The One-Line Change

    The entire power of Lightning's hardware abstraction:

        # Single CPU
        trainer = L.Trainer()

        # Single GPU
        trainer = L.Trainer(accelerator='gpu', devices=1)

        # 4 GPUs, DDP (same machine)
        trainer = L.Trainer(accelerator='gpu', devices=4, strategy='ddp')

        # 8 machines × 8 GPUs = 64 GPUs, DDP
        trainer = L.Trainer(accelerator='gpu', devices=8, num_nodes=8, strategy='ddp')

        # TPU v3 with 8 cores
        trainer = L.Trainer(accelerator='tpu', devices=8)

        # FSDP for large models
        trainer = L.Trainer(strategy='fsdp', precision='bf16-true', devices=4)

    Your LightningModule code is IDENTICAL for all of these.
    No if torch.cuda.is_available(), no DDP wrapping, no DistributedSampler setup.

### Strategy Guide

    DDP (DistributedDataParallel):
        strategy='ddp'
        Full model replica on each GPU. Gradient AllReduce after backward.
        Use: models that fit on a single GPU, standard training.
        Synchronous: all GPUs must complete each step before proceeding.

    DDP with find_unused_parameters:
        strategy=DDPStrategy(find_unused_parameters=True)
        Required when some parameters don't participate in loss computation.
        Slight overhead (scans param graph). Disable for pure dense networks.

    FSDP (Fully Sharded Data Parallel):
        strategy='fsdp'   or   strategy=FSDPStrategy(...)
        Shards parameters, gradients, and optimiser states across GPUs.
        Use: models that don't fit on a single GPU (7B+ parameters).
        Requires: precision='bf16-true' for best performance.

    DeepSpeed:
        strategy='deepspeed_stage_2'    (shard optimiser states + gradients)
        strategy='deepspeed_stage_3'    (shard everything including parameters)
        strategy='deepspeed_stage_3_offload'  (offload to CPU/NVMe — extreme scale)
        Use: cutting-edge LLM training. Most memory-efficient option.
        Requires: pip install deepspeed

    DDPSpawnStrategy:
        Spawns new processes (fork). Use in notebooks or Windows where fork fails.
        Slower than 'ddp' (no pre-spawned processes).

### DDP Best Practices in Lightning

    1. Metric syncing: use sync_dist=True in validation_step
    2. Data: Lightning handles DistributedSampler automatically
    3. Checkpointing: only rank-0 saves checkpoints (Lightning handles this)
    4. Logging: Lightning broadcasts logs from all ranks to rank-0
    5. prepare_data(): runs on rank-0 only — never assign state here
    6. Random seeds: call seed_everything(42, workers=True) for reproducibility

### Deployment Pipeline

    From trained LightningModule to production:

    Option 1: TorchScript
        model = MyModel.load_from_checkpoint('best.ckpt')
        model.eval()
        script = model.to_torchscript(method='script')
        torch.jit.save(script, 'model.pt')

    Option 2: ONNX
        model = MyModel.load_from_checkpoint('best.ckpt')
        model.to_onnx('model.onnx',
                       input_sample = torch.randn(1, 32),
                       export_params = True,
                       opset_version = 17,
                       input_names  = ['input'],
                       output_names = ['logits'],
                       dynamic_axes = {'input': {0: 'batch'}})

    Option 3: TorchServe
        torch-model-archiver --model-name my_model \
            --version 1.0 \
            --serialized-file model.pt \
            --handler custom_handler.py
        torchserve --start --model-store model_store --models my_model.mar

    Option 4: Hugging Face Hub (for research and sharing)
        model.push_to_hub("username/my-model")

### Strategy Decision Guide

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Strategy        │ strategy=...         │ When to use                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Single GPU      │ (default)            │ Development, small models   │
    │ DDP             │ 'ddp'                │ Standard multi-GPU          │
    │ FSDP            │ 'fsdp'               │ 7B+ param models            │
    │ DeepSpeed S2    │ 'deepspeed_stage_2'  │ Large models + efficiency   │
    │ DeepSpeed S3    │ 'deepspeed_stage_3'  │ Extreme scale (100B+)       │
    │ TPU             │ accelerator='tpu'    │ Google Cloud TPU            │
    └──────────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · LightningModule Deep Dive — Steps, Hooks & Manual Optimization": {
        "description": (
            "Complete LightningModule patterns from minimal to advanced. "
            "training_step / validation_step / test_step lifecycle. "
            "configure_optimizers with LR schedulers (dict form). "
            "on_train_epoch_end and on_validation_epoch_end hooks. "
            "save_hyperparameters() for reproducible checkpoints. "
            "Manual optimization mode (GAN training with two optimizers). "
            "on_before_optimizer_step for gradient norm monitoring. "
            "Benchmarking eager vs AMP in a Lightning training loop."
        ),
        "timeout": 300,
        "language": "python",
        "code": '''
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    print(f"  Lightning version: {L.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "lightning", "--quiet"], check=True)
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    print(f"  Lightning version: {L.__version__}")

print("=" * 65)
print("  LIGHTNINGMODULE DEEP DIVE — STEPS, HOOKS & MANUAL OPT")
print("=" * 65)
print()

torch.manual_seed(42)
np.random.seed(42)
torch.set_float32_matmul_precision('medium')   # suppress Tensor Core precision warning on RTX GPUs
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  PyTorch: {torch.__version__} | Device: {device_str}")
print()

# ── Shared synthetic dataset ───────────────────────────────────────────
N, FEAT, CLS = 2000, 32, 5
X = torch.randn(N, FEAT)
y = torch.randint(0, CLS, (N,))
full_ds  = TensorDataset(X, y)
n_train  = int(0.8 * N)
n_val    = N - n_train
train_ds, val_ds = random_split(full_ds, [n_train, n_val])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False)
print(f"  Dataset: {n_train} train / {n_val} val | {FEAT} features | {CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Minimal LightningModule
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Minimal LightningModule (the contract)")
print("━" * 65)
print()

class MinimalClassifier(L.LightningModule):
    """
    The smallest valid LightningModule.
    Only three things are required: forward, training_step, configure_optimizers.
    """
    def __init__(self, in_features=32, n_classes=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()     # saves lr, in_features, n_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):              # called by self(x) — for inference
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y  = batch
        loss  = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss                    # Lightning calls .backward() on this

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc',  acc,  on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

model_min = MinimalClassifier(in_features=FEAT, n_classes=CLS, lr=1e-3)
print(f"  MinimalClassifier params: {sum(p.numel() for p in model_min.parameters()):,}")
print(f"  hparams: {dict(model_min.hparams)}")
print()

# fastdevrun: runs 1 train batch + 1 val batch — sanity check only
trainer_sanity = L.Trainer(fast_dev_run=True, enable_progress_bar=False)
trainer_sanity.fit(model_min, train_dl, val_dl)
print("  fast_dev_run=True PASSED ✅  (1 train + 1 val batch ran without errors)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Full LightningModule with hooks and scheduler
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Full LightningModule with hooks and cosine LR")
print("━" * 65)
print()

import tempfile, os

class FullClassifier(L.LightningModule):
    """
    Demonstrates: hooks, LR scheduler (dict form), epoch-level metrics,
    gradient norm monitoring, configure_optimizers advanced usage.
    """
    def __init__(self, in_features=32, hidden=128, n_classes=5,
                 lr=1e-3, weight_decay=1e-2, max_epochs=15):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_classes),
        )

        # Epoch-level metric accumulators
        self._train_correct = 0
        self._train_total   = 0
        self._val_losses    = []

    def forward(self, x):
        return self.net(x)

    # ── Training step ─────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        preds  = logits.argmax(1)

        # Accumulate for epoch-level accuracy (manual — no TorchMetrics)
        self._train_correct += (preds == y).sum().item()
        self._train_total   += len(y)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """Called after ALL training batches in an epoch."""
        epoch_acc = self._train_correct / self._train_total
        self.log('train_acc_epoch', epoch_acc, prog_bar=True)
        self._train_correct = 0
        self._train_total   = 0

    # ── Validation step ───────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc',  acc,  on_epoch=True, prog_bar=True)
        self._val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        """Called after ALL validation batches. Log aggregated metrics."""
        if self._val_losses:
            mean_val_loss = np.mean(self._val_losses)
            # Already logged via self.log above, but we can compute extras here
            self._val_losses.clear()

    # ── Optimizer + scheduler ─────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr           = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        # CosineAnnealingLR: smoothly decreases lr from lr → ~0 over T_max epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {
            'optimizer':    opt,
            'lr_scheduler': {
                'scheduler': sched,
                'interval':  'epoch',    # step scheduler once per epoch
                'frequency': 1,
                'name':      'cosine_lr',  # appears in TensorBoard as 'lr/cosine_lr'
            },
        }

    # ── Gradient norm monitoring hook ─────────────────────────────────
    def on_before_optimizer_step(self, optimizer):
        """Hook called after backward but before optimizer.step().
        Perfect place to log/clip gradient norms."""
        total_norm = torch.sqrt(
            sum(p.grad.norm() ** 2
                for p in self.parameters() if p.grad is not None)
        )
        self.log('grad_norm', total_norm, on_step=True, on_epoch=False)

with tempfile.TemporaryDirectory() as tmp:
    csv_logger = CSVLogger(tmp, name='full_classifier')

    trainer_full = L.Trainer(
        max_epochs         = 15,
        accelerator        = 'auto',
        log_every_n_steps  = 5,
        logger             = csv_logger,
        enable_progress_bar = False,
        enable_model_summary = False,
    )

    model_full = FullClassifier(in_features=FEAT, hidden=128, n_classes=CLS, max_epochs=15)

    t0 = time.perf_counter()
    trainer_full.fit(model_full, train_dl, val_dl)
    t_train = time.perf_counter() - t0

    print(f"  FullClassifier trained 15 epochs in {t_train:.2f}s")
    print()

    # Read logged metrics from CSV
    metrics_path = os.path.join(tmp, 'full_classifier', 'version_0', 'metrics.csv')
    if os.path.exists(metrics_path):
        import csv
        rows = []
        with open(metrics_path) as f:
            rows = list(csv.DictReader(f))
        # Get val metrics from epoch rows
        epoch_rows = [r for r in rows if r.get('epoch') and r.get('val_acc')]
        if epoch_rows:
            print(f"  {'Epoch':<8} {'val_loss':>12} {'val_acc':>12} {'train_acc':>14}")
            print(f"  {'─'*50}")
            for row in epoch_rows[-5:]:      # show last 5 epochs
                e   = int(float(row['epoch']))
                vl  = float(row.get('val_loss', 0) or 0)
                va  = float(row.get('val_acc',  0) or 0)
                ta  = float(row.get('train_acc_epoch', 0) or 0)
                print(f"  {e+1:<8} {vl:12.4f} {va:12.4f} {ta:14.4f}")

print()
print(f"  hparams saved in checkpoint: {dict(model_full.hparams)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Manual optimization — GAN training
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Manual optimization: GAN training pattern")
print("━" * 65)
print()

class SimpleGAN(L.LightningModule):
    """
    Vanilla GAN with manual optimization.
    Demonstrates: automatic_optimization=False, self.manual_backward(),
    self.toggle_optimizer() for efficient gradient routing.
    """
    def __init__(self, latent_dim=16, data_dim=32, hidden=64, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False    # ← KEY: disable automation

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, data_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
        self.g_losses = []
        self.d_losses = []

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        x_real, _ = batch
        batch_size = x_real.size(0)
        opt_g, opt_d = self.optimizers()   # unpack in order returned by configure_optimizers
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)

        # ── Train Discriminator ──────────────────────────────────
        # toggle_optimizer: zeros only opt_d.params, sets generator.requires_grad=False
        self.toggle_optimizer(opt_d)
        x_fake   = self.generator(z).detach()     # detach: no G gradients
        pred_real = self.discriminator(x_real)
        pred_fake = self.discriminator(x_fake)
        loss_d = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real)) \
               + F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        self.manual_backward(loss_d)              # ← NOT loss_d.backward()
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ── Train Generator ──────────────────────────────────────
        self.toggle_optimizer(opt_g)
        x_fake2   = self.generator(z)
        pred_fake2 = self.discriminator(x_fake2)
        loss_g     = F.binary_cross_entropy_with_logits(
            pred_fake2, torch.ones_like(pred_fake2))   # fool discriminator
        self.manual_backward(loss_g)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.g_losses.append(loss_g.item())
        self.d_losses.append(loss_d.item())
        self.log_dict({'loss_g': loss_g, 'loss_d': loss_d}, prog_bar=False)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                  lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                  lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]     # list order matches self.optimizers() unpack order

# Build a fake "real data" distribution
real_data = torch.randn(1000, 32) * 2 + 1.0    # N(1, 2)
fake_ds   = TensorDataset(real_data, torch.zeros(1000))
gan_dl    = DataLoader(fake_ds, batch_size=64, shuffle=True)

gan = SimpleGAN(latent_dim=16, data_dim=32)
trainer_gan = L.Trainer(max_epochs=10, enable_progress_bar=False,
                         enable_model_summary=False)
trainer_gan.fit(gan, gan_dl)

g_loss_final = np.mean(gan.g_losses[-20:])
d_loss_final = np.mean(gan.d_losses[-20:])
print(f"  GAN trained (10 epochs, manual optimization):")
print(f"    Final G loss: {g_loss_final:.4f}  (target ≈ log(2) = {np.log(2):.4f})")
print(f"    Final D loss: {d_loss_final:.4f}  (target ≈ log(2) = {np.log(2):.4f})")
print(f"    Nash equilibrium both converge to log(2) in perfect GAN")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: configure_optimizers — param groups for fine-tuning
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — configure_optimizers: param groups for fine-tuning")
print("━" * 65)
print()

class FineTuningModel(L.LightningModule):
    """
    Demonstrates differential learning rates per layer — critical for
    fine-tuning pre-trained models (BERT, ResNet, ViT, etc.)
    """
    def __init__(self, backbone_lr=1e-5, head_lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Simulated pre-trained backbone (frozen initially)
        self.backbone = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # New classification head
        self.head = nn.Linear(64, 5)

    def forward(self, x):
        return self.head(self.backbone(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Param groups: different lr for backbone vs head
        param_groups = [
            {
                'params': self.backbone.parameters(),
                'lr': self.hparams.backbone_lr,     # tiny lr: fine-tune slowly
                'name': 'backbone',
            },
            {
                'params': self.head.parameters(),
                'lr': self.hparams.head_lr,         # large lr: train head fast
                'name': 'head',
            },
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-2)

        # OneCycleLR needs total_steps = steps_per_epoch × max_epochs
        steps_per_epoch = len(train_dl)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr         = [self.hparams.backbone_lr * 10,
                               self.hparams.head_lr],
            steps_per_epoch = steps_per_epoch,
            epochs         = 10,
        )
        return {
            'optimizer':    opt,
            'lr_scheduler': {
                'scheduler': sched,
                'interval':  'step',    # OneCycleLR must step per BATCH
            },
        }

ft_model = FineTuningModel(backbone_lr=1e-5, head_lr=1e-3)

# Print param group lrs
for i, pg in enumerate(ft_model.configure_optimizers()['optimizer'].param_groups):
    print(f"  Param group '{pg['name']}': lr={pg['lr']}, params={sum(p.numel() for p in pg['params']):,}")
print()
print("  Fine-tuning pattern: backbone lr (1e-5) is 100× smaller than head lr (1e-3)")
print("  Preserves pre-trained representations while adapting the final layers.")
print()

trainer_ft = L.Trainer(max_epochs=5, enable_progress_bar=False,
                        enable_model_summary=False)
trainer_ft.fit(ft_model, train_dl, val_dl)
print("  Fine-tuning training complete ✅")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · LightningDataModule + Trainer — Full Pipeline with Callbacks": {
        "description": (
            "Complete end-to-end Lightning training pipeline. "
            "LightningDataModule with prepare_data / setup / dataloaders. "
            "Trainer configuration: accelerator, precision, strategies. "
            "ModelCheckpoint: save top-k, monitor val_loss, load best. "
            "EarlyStopping with patience and min_delta. "
            "LearningRateMonitor logging lr to CSV. "
            "Checkpoint saving and loading via load_from_checkpoint. "
            "trainer.test() and trainer.predict() usage. "
            "fast_dev_run, overfit_batches debugging modes."
        ),
        "timeout": 300,
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import lightning as L
    from lightning.pytorch.callbacks import (
        ModelCheckpoint, EarlyStopping, LearningRateMonitor
    )
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "lightning", "--quiet"], check=True)
    import lightning as L
    from lightning.pytorch.callbacks import (
        ModelCheckpoint, EarlyStopping, LearningRateMonitor
    )
    from lightning.pytorch.loggers import CSVLogger

print("=" * 65)
print("  LIGHTNINGDATAMODULE + TRAINER — FULL PIPELINE")
print("=" * 65)
print()

torch.manual_seed(42)
np.random.seed(42)
print(f"  PyTorch: {torch.__version__}  |  Lightning: {L.__version__}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LightningDataModule
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — LightningDataModule (encapsulated data pipeline)")
print("━" * 65)
print()

class SyntheticDataset(Dataset):
    """Synthetic Gaussian cluster dataset with n_classes cluster centres."""
    def __init__(self, n_samples=1000, n_features=32, n_classes=5,
                 noise=0.4, seed=42):
        rng = np.random.RandomState(seed)
        centres = rng.randn(n_classes, n_features)
        labels  = rng.randint(0, n_classes, n_samples)
        X = centres[labels] + noise * rng.randn(n_samples, n_features)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):           return len(self.X)
    def __getitem__(self, idx):  return self.X[idx], self.y[idx]


class SyntheticDataModule(L.LightningDataModule):
    """
    Encapsulates the full data pipeline:
      prepare_data() → download / precompute (rank-0 only)
      setup(stage)   → build Dataset objects (all ranks)
      *_dataloader() → return DataLoader objects
    """
    def __init__(self, n_samples=2000, n_features=32, n_classes=5,
                 batch_size=64, num_workers=0, val_split=0.15, test_split=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = self.val_ds = self.test_ds = None

    def prepare_data(self):
        # Called ONCE on rank-0. Download / preprocess ONLY.
        # Do NOT assign self.x = ... here (worker processes won't see it).
        print("  [DataModule] prepare_data() — would download data here")

    def setup(self, stage: str):
        # Called on EVERY process. Build Dataset objects here.
        print(f"  [DataModule] setup(stage='{stage}') — building datasets")
        full = SyntheticDataset(
            n_samples  = self.hparams.n_samples,
            n_features = self.hparams.n_features,
            n_classes  = self.hparams.n_classes,
        )
        n_test  = int(self.hparams.test_split * len(full))
        n_val   = int(self.hparams.val_split  * len(full))
        n_train = len(full) - n_test - n_val
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"  [DataModule] train={len(self.train_ds)} "
              f"val={len(self.val_ds)} test={len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size  = self.hparams.batch_size,
                          shuffle     = True,
                          num_workers = self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size  = self.hparams.batch_size * 2,
                          shuffle     = False,
                          num_workers = self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size  = self.hparams.batch_size * 2,
                          shuffle     = False,
                          num_workers = self.hparams.num_workers)

    def predict_dataloader(self):
        # For inference: use test set (or any new data)
        return self.test_dataloader()

dm = SyntheticDataModule(n_samples=2000, n_features=32, n_classes=5, batch_size=64)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Model with test_step and predict_step
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Model with all four steps")
print("━" * 65)
print()

class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm   = nn.LayerNorm(d)
        self.linear = nn.Linear(d, d)
    def forward(self, x):
        return x + F.gelu(self.linear(self.norm(x)))


class ClassifierWithAllSteps(L.LightningModule):
    def __init__(self, in_features=32, hidden=128, n_classes=5,
                 lr=1e-3, n_blocks=3):
        super().__init__()
        self.save_hyperparameters()
        self.embed  = nn.Linear(in_features, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        self.head   = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.gelu(self.embed(x))
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

    def _shared_step(self, batch, stage):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=(stage != 'train'))
        self.log(f'{stage}_acc',  acc,  on_epoch=True, prog_bar=(stage != 'train'))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        # test_step: identical to val but called by trainer.test()
        self._shared_step(batch, 'test')

    def predict_step(self, batch, batch_idx):
        # No labels expected — returns probabilities
        x, _ = batch
        return torch.softmax(self(x), dim=-1)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer':    opt,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor':   'val_loss',   # required for ReduceLROnPlateau
                'interval':  'epoch',
            },
        }

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Trainer with callbacks
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Trainer with callbacks and checkpointing")
print("━" * 65)
print()

with tempfile.TemporaryDirectory() as tmp:
    # ── Callbacks ──────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath    = os.path.join(tmp, 'checkpoints'),
        filename   = 'epoch={epoch:02d}-val_loss={val_loss:.4f}',
        monitor    = 'val_loss',
        mode       = 'min',
        save_top_k = 2,           # keep 2 best checkpoints
        save_last  = True,        # always save last.ckpt
    )

    early_stop_cb = EarlyStopping(
        monitor   = 'val_loss',
        patience  = 8,
        min_delta = 1e-4,
        mode      = 'min',
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

    csv_logger = CSVLogger(tmp, name='run')

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs          = 50,
        accelerator         = 'auto',
        precision           = '32-true',
        gradient_clip_val   = 1.0,
        log_every_n_steps   = 5,
        num_sanity_val_steps = 2,      # validate 2 batches before training starts
        callbacks           = [checkpoint_cb, early_stop_cb, lr_monitor_cb],
        logger              = csv_logger,
        enable_progress_bar  = False,
        enable_model_summary = False,
    )

    model = ClassifierWithAllSteps(in_features=32, hidden=128, n_classes=5)

    t0 = time.perf_counter()
    trainer.fit(model, dm)
    t_fit = time.perf_counter() - t0

    stopped_epoch  = trainer.current_epoch
    best_ckpt_path = checkpoint_cb.best_model_path
    best_score     = checkpoint_cb.best_model_score

    print(f"  Training complete in {t_fit:.2f}s")
    print(f"  Stopped at epoch: {stopped_epoch} (early stopping patience=8)")
    print(f"  Best val_loss:    {float(best_score):.4f}")
    print()

    # List saved checkpoints
    ckpt_dir = os.path.join(tmp, 'checkpoints')
    if os.path.exists(ckpt_dir):
        ckpts = sorted(os.listdir(ckpt_dir))
        print(f"  Saved checkpoints ({len(ckpts)} files):")
        for c in ckpts:
            size = os.path.getsize(os.path.join(ckpt_dir, c)) / 1024
            print(f"    {c}  ({size:.1f} KB)")
    print()

    # ── Load best checkpoint ────────────────────────────────────────────
    best_model = ClassifierWithAllSteps.load_from_checkpoint(best_ckpt_path)
    best_model.eval()
    print(f"  Loaded from checkpoint: {os.path.basename(best_ckpt_path)}")
    print(f"  Loaded hparams: {dict(best_model.hparams)}")
    print()

    # ── Trainer.test() ─────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 4 — trainer.test() and trainer.predict()")
    print("━" * 65)
    print()

    test_results = trainer.test(best_model, dm, verbose=False)
    print(f"  Test results: {test_results}")
    print()

    # ── Trainer.predict() ──────────────────────────────────────────────
    predictions = trainer.predict(best_model, dm)
    all_probs   = torch.cat(predictions, dim=0)   # gather all batches
    all_classes = all_probs.argmax(dim=1)
    class_dist  = torch.bincount(all_classes, minlength=5)
    print(f"  Predictions on test set:")
    print(f"    Output shape: {all_probs.shape}  (n_test, n_classes)")
    print(f"    Predicted class distribution: {class_dist.numpy()}")
    print(f"    Max confidence: {all_probs.max():.4f}")
    print(f"    Min confidence: {all_probs.min():.4f}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Debugging modes
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Debugging modes")
print("━" * 65)
print()

DEBUG_PATTERN = """
# ─── DEBUG MODE 1: fast_dev_run ───────────────────────────────────────
# Runs exactly 1 batch train + 1 batch val. Checks pipeline end-to-end.
trainer = L.Trainer(fast_dev_run=True)
# Or set fast_dev_run=5 to run 5 batches of each.

# ─── DEBUG MODE 2: overfit_batches ───────────────────────────────────
# Intentionally overfit on a fixed subset of training data.
# If loss doesn't go to ~0: model has a bug (wrong loss, bad init, etc.)
trainer = L.Trainer(overfit_batches=0.01)  # use 1% of training data
# or:
trainer = L.Trainer(overfit_batches=10)    # use exactly 10 batches

# ─── DEBUG MODE 3: limit_train/val_batches ────────────────────────────
# Quickly iterate with a fraction of the data.
trainer = L.Trainer(
    limit_train_batches = 0.1,   # use 10% of training data per epoch
    limit_val_batches   = 0.5,   # use 50% of val data per epoch
)

# ─── DEBUG MODE 4: num_sanity_val_steps ──────────────────────────────
# Run N val batches before training (catches val bugs early, default=2)
trainer = L.Trainer(num_sanity_val_steps=5)

# ─── DEBUG MODE 5: detect_anomaly ─────────────────────────────────────
# Detect NaN/Inf in gradients and tensors. Slow but invaluable for debugging.
trainer = L.Trainer(detect_anomaly=True)

# ─── DEBUG MODE 6: profiler ──────────────────────────────────────────
trainer = L.Trainer(profiler='simple')    # operation-level timing
trainer = L.Trainer(profiler='advanced')  # line-by-line timing
from lightning.pytorch.profilers import PyTorchProfiler
trainer = L.Trainer(profiler=PyTorchProfiler(on_trace_ready=...))  # Chrome trace
"""
print(DEBUG_PATTERN)
print("  Decision guide for debugging modes:")
print("  ┌───────────────────────────────────────────────────────────────┐")
print("  │ Problem                      │ Debug mode                    │")
print("  ├───────────────────────────────────────────────────────────────┤")
print("  │ Pipeline won't run at all    │ fast_dev_run=True             │")
print("  │ Loss not decreasing          │ overfit_batches=10            │")
print("  │ NaN in loss/gradients        │ detect_anomaly=True           │")
print("  │ Training is slow             │ profiler='simple'             │")
print("  │ Test before full run         │ limit_train_batches=0.1       │")
print("  └───────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Logging, Metrics & Custom Callbacks — TorchMetrics and Hooks": {
        "description": (
            "Production-quality logging and metrics pipeline. "
            "TorchMetrics: Accuracy, F1Score, ConfusionMatrix across batches. "
            "self.log / self.log_dict with on_step and on_epoch. "
            "MetricCollection for clean multi-metric management. "
            "CSVLogger parsing for post-training analysis. "
            "Custom Callback full hook lifecycle demo. "
            "GradientNormCallback with explosion alerting. "
            "FreezeBackboneCallback for staged fine-tuning. "
            "SampleVisualisationCallback for image/data generation."
        ),
        "timeout": 300,
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    import torchmetrics
    print(f"  Lightning: {L.__version__} | TorchMetrics: {torchmetrics.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "lightning", "torchmetrics", "--quiet"], check=True)
    import lightning as L
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    import torchmetrics

print("=" * 65)
print("  LOGGING, METRICS & CUSTOM CALLBACKS")
print("=" * 65)
print()

torch.manual_seed(42)
N, FEAT, CLS = 2400, 32, 4
X = torch.randn(N, FEAT)
y = torch.randint(0, CLS, (N,))
n_tr = int(0.75 * N)
n_va = int(0.15 * N)
n_te = N - n_tr - n_va
tr_ds, va_ds, te_ds = random_split(TensorDataset(X, y), [n_tr, n_va, n_te])
tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
va_dl = DataLoader(va_ds, batch_size=128, shuffle=False)
te_dl = DataLoader(te_ds, batch_size=128, shuffle=False)

print(f"  Dataset: {n_tr}/{n_va}/{n_te} train/val/test | {FEAT} feat | {CLS} classes")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: TorchMetrics — the correct way to compute metrics in Lightning
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — TorchMetrics: per-sample vs per-batch correctness")
print("━" * 65)
print()

# Demonstrate WHY naive batch-averaging is wrong
batch_sizes  = [64] * 31 + [16]   # 31 full batches + 1 half batch
batch_accs   = np.random.uniform(0.6, 0.9, 32)

# Wrong: simple average of batch accuracies
wrong_acc = batch_accs.mean()

# Correct: weighted by batch size
correct_acc = np.average(batch_accs, weights=batch_sizes)

print(f"  Naive average of batch accuracies:   {wrong_acc:.4f}")
print(f"  Weighted (correct, per-sample) avg:  {correct_acc:.4f}")
print(f"  Error: {abs(wrong_acc - correct_acc)*100:.2f}% — small here but grows with")
print(f"  imbalanced last batches or class-weighted sampling.")
print()

class MetricRichClassifier(L.LightningModule):
    """
    Demonstrates TorchMetrics best practices:
    - MetricCollection for clean grouping
    - Passing metric objects (not .compute()) to self.log
    - Separate train/val metrics (avoid state contamination across splits)
    - ConfusionMatrix accumulated over the full validation set
    """
    def __init__(self, in_features=32, n_classes=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64, n_classes),
        )

        # MetricCollection groups multiple metrics — each resets together
        task_args = dict(task='multiclass', num_classes=n_classes)
        self.train_metrics = torchmetrics.MetricCollection(
            prefix='train_',
            metrics={
                'acc': torchmetrics.Accuracy(**task_args),
                'f1':  torchmetrics.F1Score(**task_args, average='macro'),
            }
        )
        self.val_metrics = torchmetrics.MetricCollection(
            prefix='val_',
            metrics={
                'acc': torchmetrics.Accuracy(**task_args),
                'f1':  torchmetrics.F1Score(**task_args, average='macro'),
                'prec': torchmetrics.Precision(**task_args, average='macro'),
                'rec':  torchmetrics.Recall(**task_args, average='macro'),
            }
        )
        # ConfusionMatrix: keep separate (different compute/reset semantics)
        self.val_cm = torchmetrics.ConfusionMatrix(**task_args)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        preds  = logits.argmax(1)

        # Pass metric objects directly — Lightning handles compute() + reset()
        self.train_metrics.update(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        preds  = logits.argmax(1)

        self.val_metrics.update(preds, y)
        self.val_cm.update(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self):
        cm = self.val_cm.compute()
        # Log diagonal (per-class accuracy) as individual scalars
        for cls_i in range(self.hparams.n_classes):
            cls_acc = cm[cls_i, cls_i].float() / cm[cls_i].sum().clamp(min=1)
            self.log(f'val_acc_cls{cls_i}', cls_acc)
        self.val_cm.reset()    # must reset manually for non-MetricCollection metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom Callbacks — full lifecycle
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Custom Callbacks: lifecycle and practical patterns")
print("━" * 65)
print()

class GradientNormCallback(Callback):
    """
    Monitors gradient norms per epoch. Logs mean, max, and flags
    gradient explosions (norm > threshold).

    Hook used: on_after_backward — gradients exist but optimizer hasn't stepped.
    """
    def __init__(self, threshold=10.0, log_per_step=False):
        self.threshold    = threshold
        self.log_per_step = log_per_step
        self._norms       = []

    def on_after_backward(self, trainer, pl_module):
        norms = [p.grad.norm().item()
                 for p in pl_module.parameters()
                 if p.grad is not None]
        if not norms: return
        total_norm = (sum(n**2 for n in norms)) ** 0.5
        self._norms.append(total_norm)
        if self.log_per_step:
            pl_module.log('grad_norm_step', total_norm, on_step=True, on_epoch=False)
        if total_norm > self.threshold:
            pl_module.log('grad_explosion', 1.0, on_step=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._norms:
            pl_module.log('grad_norm_mean', np.mean(self._norms))
            pl_module.log('grad_norm_max',  np.max(self._norms))
            self._norms.clear()


class FreezeBackboneCallback(Callback):
    """
    Freezes the backbone for the first `freeze_epochs` epochs,
    then unfreezes it for fine-tuning.

    Implements a common fine-tuning curriculum: train head fast first,
    then fine-tune the full model at a lower learning rate.
    """
    def __init__(self, backbone_attr='backbone', freeze_epochs=3):
        self.backbone_attr  = backbone_attr
        self.freeze_epochs  = freeze_epochs
        self._frozen        = False

    def on_train_epoch_start(self, trainer, pl_module):
        if not hasattr(pl_module, self.backbone_attr):
            return
        backbone = getattr(pl_module, self.backbone_attr)

        if trainer.current_epoch < self.freeze_epochs and not self._frozen:
            for p in backbone.parameters():
                p.requires_grad = False
            self._frozen = True
            print()
            print(f"  [FreezeCallback] Epoch {trainer.current_epoch}: backbone FROZEN")

        elif trainer.current_epoch == self.freeze_epochs and self._frozen:
            for p in backbone.parameters():
                p.requires_grad = True
            self._frozen = False
            print()
            print(f"  [FreezeCallback] Epoch {trainer.current_epoch}: backbone UNFROZEN")


class EpochTimerCallback(Callback):
    """
    Tracks time per epoch — training and validation separately.
    Logs epoch_train_secs and epoch_val_secs.
    """
    def __init__(self):
        self._train_start = None
        self._val_start   = None
        self.epoch_times  = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_start = time.perf_counter()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._train_start:
            t = time.perf_counter() - self._train_start
            pl_module.log('epoch_train_secs', t)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_start:
            t = time.perf_counter() - self._val_start
            pl_module.log('epoch_val_secs', t)
            self.epoch_times.append(t)

    def on_fit_end(self, trainer, pl_module):
        if self.epoch_times:
            print()
            print(f"  [TimerCallback] Avg val time: {np.mean(self.epoch_times)*1000:.1f}ms/epoch")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Train with all custom callbacks
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Full training run with metrics and callbacks")
print("━" * 65)
print()

with tempfile.TemporaryDirectory() as tmp:
    grad_cb   = GradientNormCallback(threshold=5.0, log_per_step=False)
    timer_cb  = EpochTimerCallback()
    ckpt_cb   = ModelCheckpoint(dirpath=tmp, monitor='val_acc',
                                 mode='max', save_top_k=1)
    csv_log   = CSVLogger(tmp, name='metrics_run')

    trainer = L.Trainer(
        max_epochs         = 20,
        callbacks          = [grad_cb, timer_cb, ckpt_cb],
        logger             = csv_log,
        log_every_n_steps  = 3,
        enable_progress_bar = False,
        enable_model_summary = False,
    )

    model = MetricRichClassifier(in_features=FEAT, n_classes=CLS, lr=1e-3)

    t0 = time.perf_counter()
    trainer.fit(model, tr_dl, va_dl)
    t_total = time.perf_counter() - t0

    # Parse CSV logs
    log_dir = os.path.join(tmp, 'metrics_run', 'version_0')
    metrics_file = os.path.join(log_dir, 'metrics.csv')

    epoch_data = {}
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            for row in csv.DictReader(f):
                ep = row.get('epoch')
                if ep and row.get('val_acc'):
                    epoch_data[ep] = row

    print(f"  Training: {trainer.current_epoch} epochs | {t_total:.2f}s total")
    print()
    if epoch_data:
        print(f"  {'Ep':>4} {'val_loss':>10} {'val_acc':>10} {'val_f1':>10} "
              f"{'val_prec':>10} {'val_rec':>10}")
        print(f"  {'─'*55}")
        for ep, row in sorted(epoch_data.items(), key=lambda x: int(x[0]))[-5:]:
            vl  = float(row.get('val_loss', 0) or 0)
            va  = float(row.get('val_acc',  0) or 0)
            vf  = float(row.get('val_f1',   0) or 0)
            vp  = float(row.get('val_prec', 0) or 0)
            vr  = float(row.get('val_rec',  0) or 0)
            print(f"  {int(ep)+1:>4} {vl:>10.4f} {va:>10.4f} {vf:>10.4f} "
                  f"{vp:>10.4f} {vr:>10.4f}")
    print()

    # Gradient norm stats from callback
    print(f"  Gradient norm stats (last recorded by callback):")
    print(f"    GradientNorm callback hooks fired on {len(grad_cb._norms) + 20*len(tr_dl)} backward passes")
    print()

    # Best checkpoint from val_acc monitoring
    best_path = ckpt_cb.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"  Best checkpoint (val_acc={float(ckpt_cb.best_model_score):.4f}):")
        print(f"    {os.path.basename(best_path)}")
        reloaded = MetricRichClassifier.load_from_checkpoint(best_path)
        print(f"    Reloaded hparams: {dict(reloaded.hparams)}")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Advanced Lightning — AMP, LR Finder, Grad Accumulation & Export": {
        "description": (
            "Advanced Trainer features and production deployment. "
            "Automatic Mixed Precision (AMP) — '16-mixed' vs 'bf16-mixed'. "
            "accumulate_grad_batches: simulating large batches. "
            "Gradient clipping: norm vs value mode. "
            "LR Finder (Smith range test) with Tuner. "
            "Batch size finder with Tuner.scale_batch_size. "
            "TorchScript export via model.to_torchscript(). "
            "ONNX export via model.to_onnx(). "
            "Distributed training strategy patterns and decision guide."
        ),
        "timeout": 300,
        "language": "python",
        "code": '''
import numpy as np
import os, tempfile, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.tuner import Tuner
    print(f"  Lightning: {L.__version__}")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "lightning", "--quiet"], check=True)
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.tuner import Tuner

print("=" * 65)
print("  ADVANCED LIGHTNING — AMP, LR FINDER, GRAD ACCUM & EXPORT")
print("=" * 65)
print()

torch.manual_seed(42)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  PyTorch: {torch.__version__}  |  Device: {device_str}")
print()

# ── Dataset ───────────────────────────────────────────────────────────
N, FEAT, CLS = 3200, 64, 8
X = torch.randn(N, FEAT)
y = torch.randint(0, CLS, (N,))
n_tr = int(0.8 * N); n_va = N - n_tr
tr_ds, va_ds = random_split(TensorDataset(X, y), [n_tr, n_va])
tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
va_dl = DataLoader(va_ds, batch_size=128, shuffle=False)
print(f"  Dataset: {n_tr} train / {n_va} val | {FEAT} features | {CLS} classes")
print()

# ── Shared model class ────────────────────────────────────────────────
class BenchmarkModel(L.LightningModule):
    def __init__(self, in_features=64, hidden=256, n_classes=8, lr=1e-3,
                 batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),       nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2),  nn.GELU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y  = batch
        loss  = F.cross_entropy(self(x), y)
        acc   = (self(x).argmax(1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc',  acc,  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                  weight_decay=1e-2)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Precision comparison — fp32 vs AMP
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Precision modes: fp32 vs 16-mixed vs bf16-mixed")
print("━" * 65)
print()

precision_results = {}

for prec_label, prec_arg in [
    ('fp32 (32-true)',  '32-true'),
    ('fp16 AMP (16-mixed)', '16-mixed'),
    ('bf16 AMP (bf16-mixed)', 'bf16-mixed'),
]:
    model_p = BenchmarkModel(in_features=FEAT, n_classes=CLS)
    trainer_p = L.Trainer(
        max_epochs          = 5,
        precision           = prec_arg,
        accelerator         = 'auto',
        enable_progress_bar  = False,
        enable_model_summary = False,
    )
    t0 = time.perf_counter()
    trainer_p.fit(model_p, tr_dl, va_dl)
    t_total = time.perf_counter() - t0

    # memory estimate (params * bytes)
    bytes_per_param = {'32-true': 4, '16-mixed': 4,  # weights still fp32 in mixed
                       'bf16-mixed': 4}  # weights fp32, activations bf16
    mem_bytes = sum(p.numel() for p in model_p.parameters()) * bytes_per_param[prec_arg]

    precision_results[prec_label] = {
        'time_s':    t_total,
        'mem_mb':    mem_bytes / 1e6,
    }

print(f"  {'Precision':<25} {'5-epoch time':>14} {'Param memory':>14}")
print(f"  {'─'*55}")
fp32_time = precision_results['fp32 (32-true)']['time_s']
for label, r in precision_results.items():
    rel = r['time_s'] / fp32_time
    print(f"  {label:<25} {r['time_s']:12.2f}s  {r['mem_mb']:12.2f} MB")
print()
print("  Note: AMP speedups are most visible on CUDA with Tensor Core GPUs.")
print("  On CPU / MPS, the overhead may outweigh gains for small models.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Gradient accumulation (simulating large batch sizes)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Gradient accumulation and gradient clipping")
print("━" * 65)
print()

# Simulate memory constraint: only 32 samples fit in memory
# But we want effective batch = 256 → accumulate 8 micro-batches
small_dl  = DataLoader(tr_ds, batch_size=32, shuffle=True)   # 32 per step

ACCUM = {
    'no_accum':    dict(accumulate_grad_batches=1,  name='batch=32 (no accum)'),
    'accum_4':     dict(accumulate_grad_batches=4,  name='batch=32×4=128'),
    'accum_8':     dict(accumulate_grad_batches=8,  name='batch=32×8=256'),
}

accum_results = {}
for key, cfg in ACCUM.items():
    m = BenchmarkModel(in_features=FEAT, n_classes=CLS)
    t = L.Trainer(
        max_epochs              = 8,
        accumulate_grad_batches = cfg['accumulate_grad_batches'],
        gradient_clip_val       = 1.0,
        gradient_clip_algorithm = 'norm',
        enable_progress_bar     = False,
        enable_model_summary    = False,
    )
    t0 = time.perf_counter()
    t.fit(m, small_dl, va_dl)
    t_run = time.perf_counter() - t0
    # Get last val_acc from trainer callback_metrics
    val_acc = t.callback_metrics.get('val_acc', torch.tensor(0.0)).item()
    accum_results[cfg['name']] = {'time': t_run, 'val_acc': val_acc}

print(f"  Gradient accumulation comparison (micro-batch=32, 8 epochs):")
print(f"  {'Config':<22} {'Time (s)':>12} {'Final val_acc':>15}")
print(f"  {'─'*52}")
for name, r in accum_results.items():
    print(f"  {name:<22} {r['time']:12.2f} {r['val_acc']:15.4f}")
print()
print("  Rule: larger effective batch → smoother gradients, needs higher lr")
print("  Scale lr by sqrt(effective_batch / base_batch) — the square root rule.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: LR Finder (Smith Range Test)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — LR Finder (Smith Range Test via Tuner)")
print("━" * 65)
print()

model_lrf = BenchmarkModel(in_features=FEAT, n_classes=CLS, lr=1e-3)
trainer_lrf = L.Trainer(
    max_epochs          = 1,
    enable_progress_bar  = False,
    enable_model_summary = False,
)
tuner = Tuner(trainer_lrf)

lr_finder = tuner.lr_find(
    model_lrf, tr_dl, va_dl,
    min_lr          = 1e-7,
    max_lr          = 1e-1,
    num_training    = 100,
    early_stop_threshold = 4.0,    # stop if loss increases 4× from minimum
)

suggested = lr_finder.suggestion()
print(f"  LR Finder ran over {len(lr_finder.results['lr'])} learning rates")
print(f"  Suggested LR: {suggested:.2e}")
print()

# Show the loss vs lr curve (text table)
lrs    = lr_finder.results['lr']
losses = lr_finder.results['loss']
# Sample 10 evenly-spaced points
step = max(1, len(lrs) // 10)
print(f"  Loss vs LR curve (sampled):")
print(f"  {'LR':>12} {'Loss':>12}")
print(f"  {'─'*28}")
for lr_val, loss_val in zip(lrs[::step], losses[::step]):
    marker = " ← suggested" if abs(lr_val - suggested) / (suggested + 1e-12) < 0.1 else ""
    print(f"  {lr_val:12.2e} {loss_val:12.4f}{marker}")
print()
print(f"  Strategy: set lr = suggested / 10 for conservative start,")
print(f"             or lr = suggested directly for aggressive training.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Deployment — TorchScript and ONNX via Lightning API
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Deployment: TorchScript and ONNX export")
print("━" * 65)
print()

# Train a model first
model_deploy = BenchmarkModel(in_features=FEAT, n_classes=CLS, lr=3e-3)
trainer_deploy = L.Trainer(max_epochs=5, enable_progress_bar=False,
                             enable_model_summary=False)
trainer_deploy.fit(model_deploy, tr_dl, va_dl)
model_deploy.eval()
print(f"  Trained model: val_acc = {trainer_deploy.callback_metrics.get('val_acc', torch.tensor(0)).item():.4f}")
print()

with tempfile.TemporaryDirectory() as tmp:

    # ── TorchScript (method='script') ──────────────────────────────────
    try:
        script_path = os.path.join(tmp, 'model.pt')
        scripted    = model_deploy.to_torchscript(method='script',
                                                    file_path=script_path)
        size_kb_s = os.path.getsize(script_path) / 1024

        # Reload with no Python class needed
        loaded_script = torch.jit.load(script_path)
        loaded_script.eval()

        x_test = torch.randn(8, FEAT)
        with torch.no_grad():
            out_orig   = model_deploy(x_test)
            out_script = scripted(x_test)
            diff_s = (out_orig - out_script).abs().max().item()

        print(f"  TorchScript export:")
        print(f"    Method:     script  (static type analysis, handles control flow)")
        print(f"    File size:  {size_kb_s:.1f} KB")
        print(f"    Max diff:   {diff_s:.2e}  ✅")
        print()
    except Exception as e:
        print(f"  TorchScript script: {e}")
        print()

    # ── ONNX export via Lightning .to_onnx() ───────────────────────────
    try:
        onnx_path    = os.path.join(tmp, 'model.onnx')
        input_sample = torch.randn(1, FEAT)
        model_deploy.to_onnx(
            onnx_path,
            input_sample  = input_sample,
            export_params = True,
            opset_version = 17,
            input_names   = ['features'],
            output_names  = ['logits'],
            dynamic_axes  = {
                'features': {0: 'batch_size'},
                'logits':   {0: 'batch_size'},
            },
        )
        size_kb_o = os.path.getsize(onnx_path) / 1024
        print(f"  ONNX export:")
        print(f"    Opset:      17")
        print(f"    File size:  {size_kb_o:.1f} KB")

        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            x_np = x_test.numpy()
            ort_out = sess.run(None, {'features': x_np})[0]
            with torch.no_grad():
                pt_out = model_deploy(x_test).numpy()
            diff_o = np.abs(ort_out - pt_out).max()
            print(f"    ORT max diff:  {diff_o:.2e}  ✅")
        except ImportError:
            print(f"    (install onnxruntime to run inference)")

    except Exception as e:
        print(f"  ONNX export: {e}")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Distributed training patterns
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Distributed training strategy patterns")
print("━" * 65)
print()

DIST_PATTERN = """
# ─── Single GPU ───────────────────────────────────────────────────────
trainer = L.Trainer(accelerator='gpu', devices=1)

# ─── DDP: 4 GPUs on 1 machine ─────────────────────────────────────────
trainer = L.Trainer(
    accelerator = 'gpu',
    devices     = 4,
    strategy    = 'ddp',
    precision   = 'bf16-mixed',
)
# Launch: python train.py  (Lightning handles torchrun internally)
# OR:     torchrun --nproc_per_node=4 train.py

# ─── DDP: 4 machines × 8 GPUs = 32 GPUs ──────────────────────────────
trainer = L.Trainer(
    accelerator = 'gpu',
    devices     = 8,
    num_nodes   = 4,
    strategy    = 'ddp',
)
# Launch on each node:
#   torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 train.py  (node 0)
#   torchrun --nnodes=4 --nproc_per_node=8 --node_rank=1 train.py  (node 1)

# ─── FSDP: sharding 7B+ parameter models ─────────────────────────────
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from lightning.pytorch.strategies import FSDPStrategy

fsdp_strategy = FSDPStrategy(
    auto_wrap_policy = {TransformerBlock},   # shard each block separately
    mixed_precision  = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.float32,
    ),
    sharding_strategy = 'FULL_SHARD',        # maximum memory savings
    cpu_offload      = False,               # True: offload to CPU (slower)
)
trainer = L.Trainer(
    strategy  = fsdp_strategy,
    precision = 'bf16-true',
    devices   = 8,
)

# ─── DeepSpeed: for extreme scale LLM training ────────────────────────
trainer = L.Trainer(
    strategy  = 'deepspeed_stage_2',    # shard optimiser states + gradients
    precision = 'bf16-mixed',
    devices   = 8,
)
# stage_3 also shards parameters (more memory, more communication)
# stage_3_offload CPU offloads to support models > GPU memory

# ─── TPU (Google Cloud / Colab) ───────────────────────────────────────
trainer = L.Trainer(
    accelerator = 'tpu',
    devices     = 8,        # 8 TPU cores
    precision   = 'bf16-mixed',
)
"""
print(DIST_PATTERN)
print("  DISTRIBUTED STRATEGY DECISION GUIDE:")
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ Scenario                      │ strategy= / accelerator=       │")
print("  ├─────────────────────────────────────────────────────────────────┤")
print("  │ 1 GPU, any model              │ accelerator='gpu', devices=1   │")
print("  │ N GPUs, model fits on 1 GPU   │ strategy='ddp', devices=N      │")
print("  │ Model too large for 1 GPU     │ strategy='fsdp', bf16-true     │")
print("  │ Extreme scale (100B+)         │ strategy='deepspeed_stage_3'   │")
print("  │ Multi-node cluster            │ ddp + num_nodes=K              │")
print("  │ Google TPU                    │ accelerator='tpu', devices=8   │")
print("  │ Apple Silicon (M1/M2/M3)      │ accelerator='mps', devices=1   │")
print("  └─────────────────────────────────────────────────────────────────┘")
print()
print("  LIGHTNING PRECISION CHEATSHEET:")
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ Hardware         │ precision=           │ GradScaler needed?    │")
print("  ├─────────────────────────────────────────────────────────────────┤")
print("  │ V100 / T4        │ '16-mixed'           │ Yes (auto by trainer) │")
print("  │ A100 / H100      │ 'bf16-mixed'         │ No                    │")
print("  │ RTX 3090/4090    │ 'bf16-mixed'         │ No                    │")
print("  │ TPU v3/v4        │ 'bf16-mixed'         │ No (XLA handles it)   │")
print("  │ CPU / MPS        │ '32-true'            │ N/A                   │")
print("  │ H100 + FP8       │ 'transformer-engine' │ No                    │")
print("  └─────────────────────────────────────────────────────────────────┘")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }