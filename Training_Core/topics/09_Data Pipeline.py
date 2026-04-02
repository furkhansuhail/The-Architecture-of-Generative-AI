"""
Data Pipeline
=============

The data pipeline is everything that happens to your data between
storage and the model's forward pass. It is the part of the training
system that practitioners most consistently underestimate. A poorly
designed pipeline starves the GPU, introduces subtle biases, causes
training instability through inconsistent preprocessing, and produces
models that behave differently at inference than during training.
Getting the pipeline right is not bookkeeping — it is a precondition
for everything else working correctly.

"""
import textwrap
import re

TOPIC_NAME   = "Data Pipeline"
DISPLAY_NAME = "09 · Data Pipeline"
ICON         = "🔁"
SUBTITLE     = "Batching · Shuffling · Augmentation · DataLoaders · Prefetch · Leakage"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — THE PIPELINE AS A SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Full Data Flow

Every training step involves the same sequence of operations, and a
bottleneck at any stage limits the entire system:

    Diagram 1 — End-to-End Data Pipeline:

    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Storage  │────►│  Load &  │────►│  Pre-    │────►│  Batch   │
    │  (disk/  │     │  Decode  │     │  process │     │  Collate │
    │  network)│     │          │     │          │     │          │
    └──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                              │
                                                              ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Model   │◄────│  GPU     │◄────│  Transfer │◄────│  Augment │
    │  Forward │     │  Memory  │     │  (H2D)   │     │          │
    │  Pass    │     │          │     │          │     │          │
    └──────────┘     └──────────┘     └──────────┘     └──────────┘

    Each arrow is a potential bottleneck. The GPU sits idle if it is
    waiting for any upstream stage. The goal: keep the GPU at >95%
    utilisation. Everything else is in service of that goal.

    Identifying the bottleneck:
    1. Monitor GPU utilisation (nvidia-smi, PyTorch Profiler).
    2. If GPU util is < 80%: the pipeline is the bottleneck.
    3. If GPU util is ~100%: the model compute is the bottleneck
       (this is the correct situation — the pipeline is not a problem).

    GPU idle = money wasted. On a $2/hour A100, a 50% idle rate
    doubles your training cost for zero benefit.


### CPU vs GPU Work Split

    CPU work (DataLoader workers):
    - Reading files from disk
    - Decompressing images (JPEG decode, etc.)
    - Data augmentation (random crops, flips, colour jitter)
    - Normalisation and type conversion
    - Batching and collating tensors

    GPU work:
    - Forward pass computation
    - Loss computation
    - Backward pass (gradient computation)
    - Weight update (optimiser step)

    The pipeline must produce batches faster than the GPU consumes them.
    If one forward+backward step takes 100ms, the pipeline must prepare
    the next batch in < 100ms — otherwise the GPU waits.

    Modern GPU training throughput (rough estimates):
    Image classification (ResNet-50, batch=256, A100):
        Forward+backward: ~30ms
        DataLoader must deliver a batch in < 30ms.
    Language model (GPT-2 medium, batch=32, A100):
        Forward+backward: ~50ms
        DataLoader must tokenise and pad sequences in < 50ms.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — BATCHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What Batching Does and Why It Exists

Mini-batch gradient descent computes the gradient on a SUBSET of the
training data at each step, rather than the full dataset (batch GD)
or a single example (stochastic GD):

    Full-batch GD:   gradient = (1/N) · Σᵢ ∇ L(xᵢ, yᵢ)
    Stochastic GD:   gradient = ∇ L(x_rand, y_rand)
    Mini-batch GD:   gradient = (1/B) · Σᵢ∈batch ∇ L(xᵢ, yᵢ)

    Mini-batch is preferred for three reasons:
    1. GPU parallelism: GPUs execute matrix operations on B examples
       simultaneously. A batch of 256 takes almost the same wall-clock
       time as a batch of 1 (up to the point of memory saturation).
    2. Gradient noise: stochastic noise in mini-batch gradients acts as
       implicit regularisation — it helps escape sharp minima.
    3. Memory: full-batch GD requires the entire dataset in memory simultaneously.


### Batch Size and Its Effects on Training

    BATCH SIZE affects:
    1. Gradient variance
    2. Generalisation
    3. Learning rate requirements
    4. Training speed (in wall-clock time)

    GRADIENT VARIANCE:
    The mini-batch gradient is an ESTIMATOR of the full-batch gradient.
    Variance of the estimator = σ²_full / B
    Larger B → lower variance → more accurate gradient estimate.
    Smaller B → higher variance → noisier updates.

    GENERALISATION (the "generalization gap"):
    Large-batch training (B > 2048 for ImageNet) tends to converge to
    SHARP minima — regions where the loss surface has high curvature
    in many directions. Sharp minima generalise poorly: a small shift
    in the test distribution moves the operating point out of the
    minimum.

    Small-batch training tends to find FLAT minima — broad regions of
    low loss. Flat minima generalise better: the test distribution
    shift still lands in the low-loss region.

    This is the "large-batch generalisation gap" (Keskar et al., 2017).
    It is real and significant: training with B=8192 on ImageNet without
    compensating techniques typically loses 1-2% accuracy vs B=256.

    Diagram 2 — Sharp vs Flat Minima:

    Loss
      │     Sharp minimum         Flat minimum
      │         │                ╭──────────────╮
      │      ╭──┴──╮           ╭─╯              ╰─╮
      │    ╭─╯     ╰─╮       ╭─╯                  ╰─╮
      └────────────────────────────────────────────── weights
              ↑                        ↑
       train and test             train ● still in minimum
       at same minimum            test  ○ has drifted but
       → narrow basin                     still low loss
       → test drifts OUT                → flat basin
                                        → good generalisation

    COMPENSATING FOR LARGE BATCH:
    Linear scaling rule (Goyal et al., 2017):
    If you multiply batch size by k, multiply learning rate by k.
    Rationale: each step covers the same "dataset fraction" in terms
    of gradient signal, so the learning rate should scale proportionally.

    B=256 → LR=0.1  (baseline)
    B=512 → LR=0.2
    B=2048 → LR=0.8

    BUT the linear scaling rule breaks at very large batch sizes.
    LR warmup is REQUIRED: start at small LR and linearly ramp to the
    scaled LR over the first 5-10 epochs. Without warmup, large-batch
    training with a large LR diverges immediately.


### Choosing Batch Size in Practice

    Rule 1: Fill the GPU memory.
    The optimal batch size is the largest batch that fits in GPU memory
    without OOM, rounded down to a power of 2 (for efficiency).
    Memory ≈ (model parameters + activations per example × B) × bytes_per_param.

    Rule 2: Keep it ≤ a generalisation-safe threshold.
    For most image classification tasks: B ≤ 4096 with linear LR scaling.
    For language modelling: B measured in tokens, typically 256K–2M tokens
    per step for large models.

    Rule 3: Use gradient accumulation for very large effective batches.
    If you want effective_batch = 2048 but only B=256 fits in memory:
    accumulate gradients over 8 steps without calling optimiser.step().
    (See Module 10 — Advanced Training for gradient accumulation.)

    Rule 4: Adjust LR with batch size.
    Always re-tune the LR when changing the batch size.
    Pretrained hyperparameter sets specify their batch size for this reason.

    Typical batch sizes by task:
    Image classification (ImageNet):   B = 256 to 1024
    Object detection:                  B = 16 to 64 (large images)
    NLP fine-tuning (BERT-scale):      B = 16 to 64 sequences
    LLM pre-training (GPT-3 scale):    B = 2048 sequences = ~2M tokens
    RL (policy gradient):              B = 64 to 512 trajectories


### Variable-Length Sequences: Padding and Packing

For NLP tasks, sequences in a batch have different lengths.
Two strategies:

    PADDING:
    Pad all sequences in the batch to the length of the longest sequence.
    Use a special PAD token (usually id=0).
    Use an attention mask to ignore PAD positions in the loss and attention.

    Example batch (sequences of length 3, 7, 5 → padded to 7):
    [A B C PAD PAD PAD PAD]
    [D E F G H I J      ]
    [K L M N O PAD PAD  ]

    Loss must mask out PAD positions:
    loss = criterion(logits, targets, ignore_index=pad_id)

    PADDING WASTE: if sequences vary widely in length, a batch may be
    50% PAD tokens. This wastes GPU compute on padding that contributes
    nothing to learning.

    PACKING (sequence packing):
    Concatenate multiple short sequences into a single long sequence
    up to the context length limit:

    [A B C | D E | F G H I | J K]   ← 4 short sequences packed into 1
    (| = separator token; attention is blocked between sequences)

    Benefit: nearly 100% of compute goes to real tokens.
    Complication: requires modifying the attention mask to prevent
    cross-sequence attention (using block-diagonal attention masks).
    Used by: most modern LLM pre-training pipelines.

    DYNAMIC BATCHING:
    Sort sequences by length and group similar-length sequences together.
    Reduces padding waste without the complexity of full packing.
    Each batch has similar-length sequences → the longest sequence in
    the batch is not much longer than the shortest.
    Reduces wasted compute by 20-40% vs random batching for NLP tasks.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — SHUFFLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Shuffling Is Non-Optional

Without shuffling, mini-batches are formed from consecutive examples
in the dataset. If the dataset is ordered (all class-0 examples first,
then all class-1 examples, etc.), consecutive batches will be dominated
by a single class:

    Batch 1:   [class 0, class 0, class 0, class 0, ...]  → gradient biased
    Batch 2:   [class 0, class 0, class 0, class 0, ...]  → still biased
    ...
    Batch N/2: [class 1, class 1, class 1, class 1, ...]  → sudden class switch

    Effects of unshuffled training:
    1. The model trains to predict the dominant class of each batch,
       not the true task distribution.
    2. BatchNorm statistics oscillate wildly between batches with
       different class distributions.
    3. The optimiser sees a strongly non-stationary loss landscape —
       gradients point in completely different directions from batch
       to batch, making convergence erratic.
    4. The effective learning signal per epoch is much lower than it
       should be.

    The fix: shuffle the entire dataset at the start of each epoch
    so that each mini-batch is a random sample of the training distribution.


### Epoch-Level vs Step-Level Shuffling

    EPOCH-LEVEL SHUFFLING (standard):
    At the start of each epoch, generate a random permutation of the
    dataset indices. Use this permutation to determine the order of
    examples for the entire epoch.

    Epoch 1: indices = [42, 7, 813, 2, 999, ...]   ← shuffled
    Epoch 2: indices = [301, 45, 7, 888, 2, ...]   ← re-shuffled

    Each example appears EXACTLY ONCE per epoch.
    After N steps, the model has seen every training example exactly once.
    This is the standard mode for most training.

    STEP-LEVEL SHUFFLING (sampling with replacement):
    At each step, randomly sample B examples from the entire dataset.
    Examples may appear multiple times within an epoch and some may
    never appear.

    Benefit: Simpler to implement for distributed training (no epoch
    boundary coordination needed).
    Cost: Some examples are never seen (wasted data) unless many epochs
    are run.
    Used by: some large-scale distributed training setups.

    IN PYTORCH: DataLoader(dataset, shuffle=True) uses epoch-level
    shuffling (permutation). It generates a new permutation each epoch
    automatically.


### Shuffling and BatchNorm: The Critical Interaction

BatchNorm computes per-batch mean and variance. If batches are not
shuffled, BN statistics see highly unrepresentative samples:

    Unshuffled (all-class-0 batch): mean and variance computed on class-0
    BN running mean/variance accumulates class-0 statistics
    Later batches (all-class-1) see incorrect normalisation

    Shuffled: each batch is a random mixture of all classes
    BN statistics are computed on a representative mini-sample of the
    data distribution
    Running statistics converge to the true dataset statistics

    This effect is especially severe with small batch sizes.
    BatchNorm + no shuffling is one of the most damaging combinations
    in practice — the model appears to train but converges to a biased
    solution.


### Shuffling in Distributed Training

When training across multiple GPUs/machines, each worker sees a
different subset of the data. A naive global shuffle followed by
partitioning is correct but requires coordination:

    DistributedSampler (PyTorch):
    1. Generate a single global permutation (seeded with epoch number).
    2. Partition the permutation into N non-overlapping chunks
       (one per worker).
    3. Each worker iterates through its chunk.
    4. At the end of the epoch, every example has been seen exactly once
       across all workers combined.

    CRITICAL: DistributedSampler requires setting the epoch number
    before each epoch to update the random permutation:
    sampler.set_epoch(epoch)   ← MUST be called at the start of each epoch
    If forgotten: all epochs use the same permutation → no re-shuffling.

    In code:
    sampler = torch.utils.data.DistributedSampler(dataset)
    loader  = DataLoader(dataset, sampler=sampler, batch_size=B)
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)   ← required for correct shuffling
        for batch in loader:
            train_step(batch)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — DATA AUGMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What Augmentation Does

Data augmentation applies RANDOM TRANSFORMATIONS to training examples,
creating modified versions that preserve the correct label. The model
must learn to classify both the original and its augmented variants
consistently — forcing it to learn INVARIANCES rather than memorising
specific pixel arrangements.

    Augmentation is regularisation in input space.
    Dropout, L2 regularise in parameter space.
    Augmentation regularises what the model must be invariant to.

    Example: horizontal flip augmentation on a cat image.
    The model sees [cat → left-facing] and [cat → right-facing].
    It must learn features that are invariant to horizontal orientation.
    Without augmentation: it may memorise "cats face left."
    With augmentation: it learns orientation-invariant cat features.

    Effective augmentation = domain knowledge encoded as code.
    Useful augmentations are those that create plausible realistic
    variations of the input. Implausible augmentations (e.g., extreme
    colour distortion that makes an image look nothing like natural
    images) may hurt performance.


### Standard Image Augmentation Operations

    GEOMETRIC TRANSFORMS:
    Random horizontal flip (p=0.5):
        50% chance of reflecting image left-right.
        Valid for: most object recognition (not text, not medical left/right).
        Typical impact: +0.5 to +1.5% accuracy on ImageNet.

    Random crop:
        Pad image by a few pixels, then randomly crop back to original size.
        Or crop a random region of the image (RandomResizedCrop).
        Forces the model to recognise objects that are partially cut off
        and at varying scales.

    Random rotation:
        Rotate by ±θ degrees. θ=10-15° is standard for most tasks.
        Not appropriate for tasks where orientation is meaningful
        (e.g., digit recognition — a rotated 6 becomes a 9).

    Random perspective / affine:
        Apply a random affine transformation (translate, shear, scale).
        More general than rotation. Used in document recognition,
        handwriting recognition.

    COLOUR TRANSFORMS:
    Colour jitter:
        Randomly change brightness, contrast, saturation, and hue.
        torch.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4, hue=0.1)
        Teaches the model that colour is not the defining feature of
        most objects. Very effective for outdoor/natural image datasets.

    Random grayscale (p=0.1-0.2):
        Occasionally convert to grayscale. Forces learning of shape
        features, not colour features.

    Gaussian blur:
        Apply a random Gaussian blur. Useful for training models that
        must be robust to image quality variation.

    ERASING TRANSFORMS:
    Random erasing / cutout:
        Randomly set a rectangular patch of the image to zero (or random noise).
        Forces the model to use information from the entire image,
        not just one discriminative region.
        Particularly effective for fine-grained recognition tasks.

    Diagram 3 — Common Augmentation Operations:

    Original:      Flipped:       Cropped:       Colour jitter:
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ 🐱 ──► │    │ ◄── 🐱 │    │  🐱 ──►│    │ 🐱──► │
    │         │    │         │    │   crop  │    │(bright) │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
    label: cat     label: cat     label: cat     label: cat
    Same label — but the model must recognise all four variants.


### Advanced Augmentation Strategies

    MIXUP (Zhang et al., 2018):
    Create a new example by linearly interpolating two training examples:
    x_mix = λ·x₁ + (1-λ)·x₂,   y_mix = λ·y₁ + (1-λ)·y₂
    where λ ~ Beta(α, α), typically α=0.2 or α=0.4.

    Effect: the model must output a mixture of labels for a mixture of inputs.
    This enforces SMOOTH INTERPOLATION in the learned feature space —
    no sharp decision boundaries between classes.
    Mixup is one of the most effective regularisation techniques
    for image classification (+0.5 to +1.5% on ImageNet).

    CutMix (Yun et al., 2019):
    Cut a rectangular patch from one image and paste it into another.
    Labels are mixed proportionally to the patch area.

    x_new = mask · x₁ + (1-mask) · x₂
    y_new = (area_ratio) · y₁ + (1-area_ratio) · y₂

    More natural than Mixup (the image contains two clearly distinct
    regions rather than an unnatural blend). Often outperforms Mixup.

    RandAugment (Cubuk et al., 2020):
    Sample N augmentation operations uniformly from a fixed library
    of K operations. Apply them sequentially with magnitude M.

    Hyperparameters: N (number of ops, typically 2) and M (magnitude, 1-10).
    Library: rotate, shear, translate, flip, colour, contrast, brightness,
             posterise, equalize, autocontrast, solarise, etc.
    Eliminates the need to manually design augmentation policies.
    Standard for ViT and modern CNN training.

    AugMix (Hendrycks et al., 2020):
    Creates augmented examples by MIXING several augmented chains:
    x_aug = Σᵢ wᵢ · Aug_chain_i(x)   (weighted sum of augmentation chains)
    Improves robustness to distribution shift (corruptions at test time).

    TrivialAugment (Müller et al., 2021):
    Even simpler than RandAugment: randomly sample ONE operation and
    apply it at a random magnitude sampled uniformly from the full range.
    Despite its simplicity, matches or exceeds RandAugment on most benchmarks.
    Shows that the search for complex augmentation policies is often
    not worth the effort.

    Test-Time Augmentation (TTA):
    At INFERENCE, apply multiple augmentations and average the predictions.
    For example, predict on: original, horizontal flip, and 4 crops.
    Average the 6 predictions. This is augmentation used for prediction
    improvement, not training. Often gives +0.3 to +0.5% accuracy.


### NLP Augmentation

NLP augmentation is harder than image augmentation because text is
discrete — small changes can change the meaning entirely.

    SYNONYM REPLACEMENT:
    Randomly replace k words with synonyms (from WordNet or similar).
    "The dog ran fast" → "The canine sprinted rapidly"
    Risk: synonyms are rarely truly interchangeable in context.

    BACK-TRANSLATION:
    Translate the sentence to another language, then back to the original.
    Produces semantically equivalent but lexically different text.
    Effective but expensive (requires a translation model).

    RANDOM DELETION / INSERTION / SWAP (EDA, Wei & Zou, 2019):
    Easy Data Augmentation — four simple operations:
    SR: Synonym Replacement of n random non-stop words
    RI: Random Insertion of synonyms into random positions
    RS: Random Swap of two words n times
    RD: Random Deletion of words with probability p
    Particularly effective for small datasets (< 1000 training examples).

    SPAN MASKING (used in pre-training, not fine-tuning):
    Mask random spans of tokens (MLM, BERT-style) for pre-training.
    Not augmentation for fine-tuning but a form of self-supervised
    augmentation for pre-training.

    NOTE: For NLP, most modern fine-tuning pipelines on large datasets
    do not use explicit augmentation — the pre-trained model is already
    robust to lexical variation from its pre-training exposure.
    Augmentation is most beneficial for small fine-tuning datasets.


### Augmentation Train/Test Asymmetry

A critical pipeline principle: augmentation is applied ONLY during training.
At validation and test time, no random transformations should be applied
(except for TTA, which is explicit and intentional).

    WRONG: applying random augmentations at test time
    → val loss is stochastic (different every evaluation pass)
    → worse val loss than the model's true capability
    → early stopping triggers too early
    → reported test performance is a noisy estimate

    CORRECT: at test time, apply only DETERMINISTIC preprocessing
    (resize, center crop, normalise). Nothing random.

    Typical image preprocessing:
    TRAINING:    RandomResizedCrop(224) → RandomHorizontalFlip() →
                 ColorJitter(...) → ToTensor() → Normalize(mean, std)

    VALIDATION:  Resize(256) → CenterCrop(224) →
                 ToTensor() → Normalize(mean, std)

    The normalisation (mean and std) is IDENTICAL for train and val.
    The random augmentations exist only in the training transform.

    Normalisation values must be computed on the TRAINING SET only.
    Using dataset-wide statistics (including val/test) is a form
    of data leakage (minor in practice but conceptually wrong).

    Standard ImageNet normalisation:
    mean = [0.485, 0.456, 0.406]   (per-channel, in [0,1] scale)
    std  = [0.229, 0.224, 0.225]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — THE DATALOADER: PARALLELISM AND PREFETCHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### PyTorch DataLoader Architecture

The DataLoader is the component that bridges the Dataset (which returns
individual examples) and the training loop (which expects batches).

    Components:
    Dataset:    Defines __getitem__(idx) — how to load one example.
    Sampler:    Defines the order of indices (shuffled or sequential).
    Collate fn: Defines how to stack a list of examples into a batch.
    Workers:    Parallel processes that call __getitem__ and collate.

    Diagram 4 — DataLoader Worker Architecture:

    Main process (training loop)
         │
         │  request batch
         ▼
    ┌─────────────────────────────────────────────────────┐
    │              DataLoader                             │
    │                                                     │
    │  Worker 0: load idx 42, 7, 813 → batch fragment     │
    │  Worker 1: load idx 2, 999, 31 → batch fragment     │
    │  Worker 2: load idx 56, 401, 8 → batch fragment     │
    │  Worker 3: load idx 77, 23, 500→ batch fragment     │
    │       │         │         │         │               │
    │       └─────────┴────┬────┴─────────┘               │
    │                      │ collate into batch            │
    │                      ▼                              │
    │              batch ready in prefetch buffer         │
    └─────────────────────────────────────────────────────┘
         │
         │  deliver batch
         ▼
    GPU training step


### num_workers: Parallel Data Loading

    num_workers=0: no parallel loading. Loading happens synchronously
                   in the main process during the training loop.
                   GPU waits for loading → low GPU utilisation.
                   ONLY use for debugging.

    num_workers=N: N separate CPU processes load data in parallel.
                   While the GPU processes batch k, workers prepare batch k+1.
                   Ideal: workers are always ahead by ≥ 1 batch.

    Choosing num_workers:
    Rule of thumb: num_workers = number of CPU cores / number of GPUs
    For a 16-core CPU, 1 GPU: num_workers ≈ 8-16
    For a 16-core CPU, 4 GPUs: num_workers ≈ 4 per GPU

    Practical approach: start at num_workers=4, increase until GPU
    utilisation saturates or training speed stops improving.

    Overhead of large num_workers:
    - Each worker has its own copy of the dataset object in memory
    - Spawning workers has startup cost (amortised over a long training run)
    - IPC (inter-process communication) overhead for passing batches
    - Context switching between many workers

    For very fast I/O (e.g., NVMe SSD, data in shared memory):
    num_workers=4 is often sufficient.
    For slow I/O (network storage, spinning disk, heavy JPEG decode):
    num_workers=16 or higher may be needed.

    KNOWN BUG: On some systems, too many workers with JPEG decoding
    can cause deadlocks or excessive memory use. If training crashes
    with SIGKILL or segfaults, reduce num_workers.


### pin_memory and Host-to-Device Transfer

    pin_memory=True: allocates batch tensors in PINNED (page-locked) memory.

    Standard (pageable) memory: CPU→GPU transfer requires the OS to
    first copy to a temporary pinned buffer, then DMA to GPU.
    Two copies: pageable → pinned → GPU.

    Pinned memory: the CPU→GPU DMA transfer is direct (one copy).
    Throughput improvement: typically 20-30% faster H2D transfer.

    Cost: pinned memory is non-swappable. Allocating too much pinned
    memory reduces available memory for other processes and the OS.

    Rule: always use pin_memory=True when training on GPU, unless
    the system has very limited RAM.

    Combined with non_blocking=True on .to(device):
    batch = batch.to(device, non_blocking=True)
    → The CPU→GPU transfer is initiated but the CPU does not wait
      for it to complete. The CPU continues to the next operation
      (e.g., starting the next data load). GPU operations that
      depend on the tensor will wait automatically.
    → Allows CPU and GPU work to overlap further.


### prefetch_factor and the Prefetch Buffer

    prefetch_factor (default=2):
    Each worker pre-loads prefetch_factor × num_workers examples ahead
    of what the training loop has consumed.

    With num_workers=4, prefetch_factor=2:
    Up to 8 batches are prepared and waiting in the buffer.
    The training loop never waits for data as long as it processes
    batches slower than the workers produce them.

    Increasing prefetch_factor:
    - Reduces the chance of the GPU waiting for data
    - Increases memory use (more batches held in RAM simultaneously)
    - Rarely needs to go above 4

    Diagram 5 — Prefetch Buffer Timeline:

    Time:      0    1    2    3    4    5    6    7    8
    GPU:       [B0] [B1] [B2] [B3] [B4] [B5] [B6] [B7]
    Workers:  [B1][B2][B3][B4][B5][B6][B7][B8]...
               ↑ Workers are always 1+ batches ahead
               GPU never waits = 100% GPU utilisation


### persistent_workers

    persistent_workers=True:
    Keep worker processes alive between epochs.

    Default (persistent_workers=False):
    Workers are killed at the end of each epoch and re-spawned at the
    start of the next. Spawning workers has overhead (~1-2 seconds for
    typical configurations). For short epochs (small datasets), this
    overhead is significant.

    With persistent_workers=True:
    Workers persist across epoch boundaries. Dataset state is preserved.
    Eliminates spawning overhead.

    Caveat: the Dataset's internal random state must be reset each epoch
    if augmentations use a per-worker random seed. This is automatic
    in PyTorch's default implementation.

    Rule: always use persistent_workers=True when training for many
    epochs on a small-to-medium dataset.


### Recommended DataLoader Configuration

    # Production configuration for GPU training
    loader = DataLoader(
        dataset,
        batch_size       = B,
        shuffle          = True,          # epoch-level shuffling
        num_workers      = 8,             # start at 4-8, tune to GPU util
        pin_memory       = True,          # faster H2D transfer
        persistent_workers = True,        # avoid re-spawn overhead
        prefetch_factor  = 2,             # default, increase if GPU waits
        drop_last        = True,          # drop incomplete final batch
                                          # (important for BatchNorm)
    )

    drop_last=True:
    The last batch of an epoch typically has fewer than B examples.
    With BatchNorm, a very small batch (e.g., B=2) produces unreliable
    statistics. Dropping it avoids this. The cost is at most (B-1)
    examples discarded per epoch — negligible for large datasets.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — PREPROCESSING AND NORMALISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Fit-on-Train-Only Rule

All preprocessing statistics (means, standard deviations, min/max values,
vocabulary sizes, TF-IDF weights, PCA components) must be computed using
ONLY the training split.

    WRONG: compute normalisation stats on the full dataset, then split.
    WHY: the test set's statistics influence the normalisation of the
    training set. The model sees test set information during training.
    This is DATA LEAKAGE. Reported test performance is inflated.

    CORRECT:
    1. Split the data into train/val/test FIRST.
    2. Compute all preprocessing statistics using ONLY the training split.
    3. Apply those statistics to normalise train, val, and test.

    In code:
    scaler = StandardScaler()
    scaler.fit(X_train)              ← fit on train only
    X_train_norm = scaler.transform(X_train)
    X_val_norm   = scaler.transform(X_val)   ← apply same transform
    X_test_norm  = scaler.transform(X_test)  ← apply same transform

    This pattern applies to EVERY preprocessing step:
    Tokeniser vocabulary:  built from training text only
    Image normalisation:   mean/std from training images only
    PCA components:        fit on training features only
    Imputation values:     median/mean from training rows only


### Why Normalisation Matters for Gradient Flow

Unnormalised inputs create CONDITIONING problems in the loss landscape:

    Example: two features with very different scales.
    Feature 1: salary in dollars, range [30,000, 500,000]
    Feature 2: age in years, range [18, 80]

    Without normalisation, the weight for Feature 1 must be ~10,000×
    smaller than the weight for Feature 2 to produce comparable outputs.
    The loss landscape has very different curvature in each weight direction:

    Diagram 6 — Poorly Conditioned vs Well-Conditioned Loss Landscape:

    w₂ (salary weight)          w₂ (normalised)
      │                           │
      │   elongated ellipses      │   circular contours
      │   ─────────────           │       ╭────╮
      │ ──────────────            │    ╭──╯    ╰──╮
      │──────────────             │  ╭─╯          ╰─╮
      └──────────────── w₁        └──────────────────── w₁
         ← requires tiny             ← all directions
           steps in w₁ direction        equally fast
           to avoid overshooting

    Poorly conditioned: gradient descent must take tiny steps (to avoid
    overshooting w₂) and makes slow progress (in the w₁ direction).
    Well conditioned: all directions have similar scale → converges faster.

    Standard normalisation for tabular data:
    z-score:   x_norm = (x - μ) / σ        → mean=0, std=1
    min-max:   x_norm = (x - min) / (max - min)  → range [0, 1]
    Use z-score for neural networks (preferred).
    Use min-max when the range is known and meaningful.


### Online Normalisation vs Offline Normalisation

    OFFLINE NORMALISATION (precompute on disk):
    Compute normalised versions of ALL examples and save to disk.
    At training time, load pre-normalised examples.
    Pros: zero normalisation overhead during training.
    Cons: storage duplication, inflexible (can't change normalisation
          without recomputing all files), not applicable to on-the-fly
          augmentation.

    ONLINE NORMALISATION (apply per-batch at load time):
    Compute normalisation inside __getitem__ or the collate function.
    Pros: flexible, no storage overhead, compatible with augmentation.
    Cons: small CPU overhead per batch.

    For image data: online normalisation is standard.
    The computation (subtract mean, divide std) takes microseconds per image.
    Any overhead is dominated by JPEG decompression.

    For tabular data with very large datasets (> 100M rows):
    offline normalisation is preferred to avoid repeated CPU overhead.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — IMBALANCED DATASETS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Problem

Class imbalance occurs when some classes have far more examples than others.
Most real-world classification problems are imbalanced:

    Medical imaging: 95% healthy, 5% pathology
    Fraud detection: 99.9% legitimate, 0.1% fraud
    Defect detection: 98% good, 2% defective
    Sentiment: 60% neutral, 30% positive, 10% negative

    Without intervention, a model trained on an imbalanced dataset
    will learn to predict the majority class for most inputs:
    - Minimises training loss by exploiting class frequencies
    - Achieves high accuracy (98% correct for the 98% healthy class!)
    - But is useless: it never detects what matters (the rare class)


### Strategy 1 — Class-Weighted Loss

Assign higher loss weight to minority class examples:

    weight_class_k = n_total / (n_classes × n_class_k)

    For a dataset with 9,500 negative and 500 positive examples:
    weight_neg = 10,000 / (2 × 9,500) = 0.526
    weight_pos = 10,000 / (2 × 500)   = 10.0

    Each positive example now contributes 10× more to the loss than
    each negative example. The model can no longer minimise loss by
    ignoring positives.

    In PyTorch:
    class_weights = torch.tensor([0.526, 10.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    BENEFIT: Simple, no data resampling needed, works well in practice.
    LIMITATION: Does not change what the model sees — still mostly
    majority examples, just weighted differently in the loss.


### Strategy 2 — Oversampling the Minority Class

Repeat minority class examples so each class appears roughly equally
in training:

    NAIVE OVERSAMPLING: duplicate minority examples exactly.
    Risk: the model overfits to the specific minority examples seen
    many times. The minority class representations are memorised
    rather than generalised.

    SMOTE (Synthetic Minority Oversampling Technique):
    Generate SYNTHETIC minority examples by interpolating between
    real minority examples in feature space:
    x_synthetic = x₁ + λ · (x₂ - x₁)   where x₁, x₂ are minority examples
    Creates new examples that are "between" real ones.
    Effective for tabular data. Less used for image/text.

    WeightedRandomSampler (PyTorch):
    Instead of duplicating data, assign higher sampling probability
    to minority examples. Each minority example is drawn more often
    per epoch.

    # Compute per-example weights
    class_counts = [9500, 500]
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
    loader  = DataLoader(dataset, sampler=sampler, batch_size=B)

    Effect: each batch has approximately equal numbers of each class.
    The model sees minority examples much more frequently.


### Strategy 3 — Undersampling the Majority Class

Remove majority class examples so classes are balanced:

    Pros: smaller dataset → faster training.
    Cons: discards potentially useful data.
    Use when: the majority class has abundant data and training
    speed is a concern.

    Random undersampling: randomly remove majority examples.
    Informed undersampling: remove majority examples that are
    most similar to minority examples (near the decision boundary),
    keeping informative difficult examples.


### Focal Loss for Extreme Imbalance

Focal Loss (Lin et al., 2017, introduced for RetinaNet object detection)
modifies cross-entropy to down-weight easy examples and focus on hard ones:

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Where:
        p_t = model probability for the correct class
        α_t = class-balancing weight (like class-weighted loss)
        γ   = focusing parameter (typically 0.5 to 5)

    (1 - p_t)^γ is the MODULATING FACTOR:
    For easy examples (p_t → 1): factor → 0 → loss is near-zero → ignored
    For hard examples (p_t → 0): factor → 1 → normal cross-entropy loss

    At γ=0: Focal Loss = standard cross-entropy (modulating factor = 1)
    At γ=2 (standard): easy examples receive 1,000× less weight than hard ones

    Focal Loss forces the model to focus its learning capacity on
    difficult examples — which are disproportionately the minority class
    examples (they are harder to classify correctly).

    Used in: RetinaNet, many one-stage object detectors, and any
    classification problem with extreme class imbalance.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — DATASET AND DATALOADER BUGS: A TAXONOMY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Most Common Data Pipeline Bugs

These bugs are extremely common, often silent (training appears to
proceed normally), and cause significant performance degradation:

    BUG 1 — Augmentation applied at test time:
    Symptom: Val loss is higher than expected and stochastic
             (different every evaluation pass on the same data).
    Fix: use two separate transform objects — one with augmentation
         (training) and one without (validation/test).

    BUG 2 — Normalisation not applied at test time:
    Symptom: Model performs well on validation but fails catastrophically
             in deployment. Training data was normalised; production
             data is not (or uses different statistics).
    Fix: apply the same normalisation transform at test time.
         Save the normalisation statistics alongside the model weights.

    BUG 3 — Label index mismatch:
    Symptom: Training appears normal but accuracy never rises above
             chance level. Loss decreases but the model is learning
             to predict wrong classes.
    Cause: the Dataset returns labels starting from 1 (not 0) but
           CrossEntropyLoss expects labels starting from 0. Or class
           ordering differs between train and test split.
    Fix: verify label distributions match between the splits.
         Verify labels are 0-indexed for PyTorch loss functions.

    BUG 4 — Insufficient num_workers causing GPU starvation:
    Symptom: GPU utilisation is 30-50%. Training is slower than expected.
             Profiling shows the GPU is idle waiting for data.
    Fix: increase num_workers. Add pin_memory=True.

    BUG 5 — Missing model.eval() during validation:
    Symptom: Validation loss is stochastic (varies each time the same
             validation batch is processed). Val loss is higher than
             the true val loss because dropout is still active.
    Fix: always call model.eval() before the validation loop and
         model.train() before the training loop.

    BUG 6 — Data leakage through preprocessing:
    Symptom: Validation performance looks excellent during development
             but test performance is significantly worse.
    Cause: normalisation statistics were computed on the full dataset
           (including validation/test), or train/val splits were made
           after a preprocessing step that used global statistics.
    Fix: split first. Fit preprocessing only on training data.

    BUG 7 — Workers sharing random state:
    Symptom: All workers produce the same augmentation pattern —
             the augmentation appears ordered or repetitive.
    Cause: DataLoader workers inherit the main process's random seed.
           Multiple workers start with the same seed and produce
           identical (or correlated) random transformations.
    Fix: set a unique worker seed in the worker_init_fn:
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)
    DataLoader(..., worker_init_fn=worker_init_fn)

    BUG 8 — Tensor dtype mismatch:
    Symptom: RuntimeError: expected scalar type Float but got Double,
             or silent precision loss.
    Cause: dataset returns torch.float64 tensors but the model uses
           torch.float32 parameters.
    Fix: add .float() cast in __getitem__ or collate function.
         Model parameters: model = model.float()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — PIPELINE PERFORMANCE OPTIMISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Profiling the Pipeline Bottleneck

Before optimising, measure where the time is spent:

    Method 1 — Timing baseline:
    # How fast is the DataLoader alone (no model)?
    import time
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i == 100: break
    t1 = time.perf_counter()
    print(f"DataLoader: {(t1-t0)/100*1000:.1f} ms/batch")

    Compare with:
    # How long does one GPU step take (model only)?
    batch = next(iter(loader))
    batch = batch.to(device)
    t0 = time.perf_counter()
    for _ in range(100):
        out = model(batch)
        loss = criterion(out, targets)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    t1 = time.perf_counter()
    print(f"GPU step: {(t1-t0)/100*1000:.1f} ms/batch")

    If DataLoader ms/batch > GPU step ms/batch:
    → Pipeline is the bottleneck. Increase num_workers, use pin_memory,
      cache data to RAM, or switch to faster storage.
    If DataLoader ms/batch < GPU step ms/batch:
    → GPU is the bottleneck. Pipeline is adequate. Optimise the model.

    Method 2 — PyTorch Profiler:
    with torch.profiler.profile(activities=[...]) as prof:
        train_one_step(...)
    print(prof.key_averages().table(sort_by="cpu_time_total"))
    → Shows exact time breakdown per operation including data loading.


### Caching Strategies

    IN-MEMORY CACHING:
    Load the entire dataset into RAM at startup. Each epoch reads
    from RAM (nanoseconds) rather than disk (milliseconds).
    Applicable when dataset fits in RAM (< system RAM).
    Speedup: 10-100× for disk-bound pipelines.

    class CachedDataset(Dataset):
        def __init__(self, base_dataset):
            self.cache = [base_dataset[i] for i in range(len(base_dataset))]
        def __getitem__(self, idx):
            return self.cache[idx]   # transform applied after cache hit

    HDF5 / LMDB CACHING:
    Store preprocessed examples in a binary format optimised for
    sequential/random access:
    HDF5: good for array data (images as float arrays); supports
          partial reads and compression.
    LMDB: memory-mapped key-value store. Extremely fast random access.
          Standard format for large-scale vision datasets.

    WEBDATASET:
    Store examples as tar archives on disk or object storage (S3, GCS).
    Stream data without random access — tar archives are read sequentially.
    Enables training directly from S3/GCS without downloading the dataset.
    Scales to datasets larger than local disk capacity.
    Widely used for large vision and multimodal model training.


### The Gold Standard Pipeline Configuration

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Component          │ Recommended setting                             │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Storage format     │ Binary (HDF5, LMDB, WebDataset) not raw JPEG    │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Preprocessing      │ Fit on training split only                      │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Augmentation       │ Applied in training transform only              │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Shuffling          │ shuffle=True (training) / False (val/test)      │
    │                    │ sampler.set_epoch(epoch) for distributed        │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Batch size         │ Largest that fits in GPU memory; tune LR to     │
    │                    │ match (linear scaling rule)                     │
    ├──────────────────────────────────────────────────────────────────────┤
    │ num_workers        │ 8–16 (tune to CPU count / GPU count)            │
    ├──────────────────────────────────────────────────────────────────────┤
    │ pin_memory         │ True (always for GPU training)                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ persistent_workers │ True (avoid re-spawn overhead)                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ prefetch_factor    │ 2 (default); increase to 4 if GPU still waits   │
    ├──────────────────────────────────────────────────────────────────────┤
    │ drop_last          │ True (avoid small final batch with BatchNorm)   │
    ├──────────────────────────────────────────────────────────────────────┤
    │ worker_init_fn     │ Set unique per-worker random seed               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ H2D transfer       │ batch.to(device, non_blocking=True)             │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Imbalance strategy │ Class-weighted loss + WeightedRandomSampler     │
    │                    │ (or Focal Loss for extreme imbalance)           │
    └──────────────────────────────────────────────────────────────────────┘

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
        from Training_Core.visuals.data_pipeline_visual import (
            DATA_PIPELINE_VISUAL_HTML,
            DATA_PIPELINE_VISUAL_HEIGHT,
        )
        visual_html   = DATA_PIPELINE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = DATA_PIPELINE_VISUAL_HEIGHT
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