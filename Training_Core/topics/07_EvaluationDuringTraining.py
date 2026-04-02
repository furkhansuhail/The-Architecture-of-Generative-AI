"""
Evaluation During Training
==========================

Training a model without a rigorous evaluation loop is like navigating
without a compass — you are moving, but you cannot tell if you are
heading toward the destination or away from it. Evaluation during
training is the feedback mechanism that tells you whether the model is
learning, overfitting, underfitting, or converging — and gives you the
evidence to decide what to change.

"""
import textwrap
import re

TOPIC_NAME   = "Evaluation During Training"
DISPLAY_NAME = "07 · Evaluation During Training"
ICON         = "📊"
SUBTITLE     = "Val Loops · Metrics · Splits · Overfitting · Calibration · Debugging"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — THE THREE-WAY DATA SPLIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why You Need Three Sets, Not Two

The fundamental constraint of generalisation measurement:

    Any dataset that influences ANY decision during model development
    can no longer serve as an unbiased estimate of generalisation performance.

    If you use the test set to pick the best model, tune hyperparameters,
    or decide when to stop training — the test set has been LEAKED into
    the development process. Your reported test performance is now an
    optimistic estimate of true generalisation.

    This is not a philosophical concern — it is a practical one.
    Researchers who repeatedly evaluate on the same test set and adjust
    their models accordingly will inevitably overfit to that test set,
    even without accessing individual test labels directly.

The three-way split enforces a clean separation:

    ┌────────────────────────────────────────────────────────────────────┐
    │ Split        │ Used for                     │ Touched by           │
    ├────────────────────────────────────────────────────────────────────┤
    │ Training set │ Computing gradients,          │ Optimiser, every     │
    │              │ updating weights              │ training step        │
    ├────────────────────────────────────────────────────────────────────┤
    │ Validation   │ Hyperparameter tuning,        │ You (the researcher),│
    │ set          │ model selection, early        │ but NOT the optimiser│
    │              │ stopping, architecture        │                      │
    │              │ decisions                     │                      │
    ├────────────────────────────────────────────────────────────────────┤
    │ Test set     │ Final, one-time evaluation    │ Nobody — until the   │
    │              │ reported as the result        │ very end             │
    └────────────────────────────────────────────────────────────────────┘

    The test set must be HELD OUT completely until after all
    development decisions are finalised. It is the one honest answer
    to "how well does this model generalise?"

    Contaminating the test set (using it for ANY decision) means you
    are reporting in-sample performance on a set you think is out-of-sample.


### Standard Split Ratios

    Exact ratios depend on total dataset size. General guidance:

    ┌───────────────────────────────────────────────────────────────────┐
    │ Dataset size  │ Train  │ Validation │ Test    │ Notes             │
    ├───────────────────────────────────────────────────────────────────┤
    │ < 1,000       │ 60%    │ 20%        │ 20%     │ Consider k-fold   │
    │               │        │            │         │ CV instead        │
    ├───────────────────────────────────────────────────────────────────┤
    │ 1K – 100K     │ 70%    │ 15%        │ 15%     │ Standard          │
    ├───────────────────────────────────────────────────────────────────┤
    │ 100K – 1M     │ 80%    │ 10%        │ 10%     │ Standard          │
    ├───────────────────────────────────────────────────────────────────┤
    │ > 1M          │ 98%    │ 1%         │ 1%      │ 1% of 10M is      │
    │               │        │            │         │ still 100K samples│
    └───────────────────────────────────────────────────────────────────┘

    As the dataset grows, the validation and test sets only need to be
    large enough to give a STATISTICALLY STABLE estimate — not a large
    fraction of the data.

    Key constraint: the validation and test sets must be large enough that
    performance differences of practical interest are statistically detectable.
    Rule of thumb: ≥ 1,000 samples per evaluation set for classification.


### Stratified Splitting

For classification, splits must be STRATIFIED — each split must contain
the same CLASS DISTRIBUTION as the original dataset:

    Without stratification (random split):
    Dataset: 90% class A, 10% class B
    By chance, a 10% test set might contain:
        92% class A, 8% class B   → biased toward A
        88% class A, 12% class B  → biased toward B
    Performance estimates fluctuate based on which classes ended up
    in the test set, not just how good the model is.

    With stratification:
    Each split is guaranteed to have exactly 90%/10% class distribution.
    Performance estimates are stable across different random seeds.

    Stratification is especially critical when:
    - Classes are highly imbalanced (common in real data)
    - The dataset is small (stochastic imbalance is larger)
    - You report per-class metrics (each class needs enough examples)

    In sklearn: train_test_split(..., stratify=y)
    In PyTorch: use a custom sampler or split indices with stratification.


### Temporal and Group Splits

For time-series data or grouped data, random splits are WRONG:

    Time-series data:
    Random split leaks FUTURE data into training — the model learns
    from examples that occurred after the examples it must predict.
    This inflates performance dramatically and the model fails in deployment.

    CORRECT: Chronological split.
    Train on the first T% of time, validate on the next V%, test on the rest.
    Or: rolling window evaluation (train on months 1–6, test on month 7;
    train on months 1–7, test on month 8; etc.).

    Grouped data (e.g., multiple examples per patient, per document):
    Random split can place examples from the SAME patient in both
    train and test. The model memorises patient-specific features
    rather than learning generalisable patterns.
    Reported test accuracy is inflated — the model "knows" the patient.

    CORRECT: Group split (GroupShuffleSplit in sklearn).
    All examples from a given patient appear in ONLY ONE split.

    Diagram 1 — Temporal Split vs Random Split:

    Timeline:   [──────── PAST ────────────────── FUTURE ──────]

    CORRECT:    [████████████████ TRAIN ████████][VAL][  TEST  ]
                 ← chronological order maintained

    WRONG:      [TRAIN ██ TEST ██ TRAIN ██ VAL ██ TRAIN ██ TEST]
                 ← random — future leaks into training


### Cross-Validation for Small Datasets

When n < 5,000, a single validation split is too noisy to be reliable.
k-fold cross-validation gives a more stable estimate:

    k-fold procedure:
    1. Split the dataset into k equal folds.
    2. For fold i: train on folds {1,...,k} \ {i}, validate on fold i.
    3. Repeat for all k folds.
    4. Report the mean and standard deviation of the k validation scores.

    Diagram 2 — 5-Fold Cross-Validation:

    Fold 1: [VAL ][TRAIN][TRAIN][TRAIN][TRAIN]
    Fold 2: [TRAIN][VAL ][TRAIN][TRAIN][TRAIN]
    Fold 3: [TRAIN][TRAIN][VAL ][TRAIN][TRAIN]
    Fold 4: [TRAIN][TRAIN][TRAIN][VAL ][TRAIN]
    Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][VAL ]
             ↓ each fold produces one validation score
    Final score: mean ± std of 5 scores

    k=5 or k=10 are standard. k=n (Leave-One-Out CV) gives the lowest
    bias but the highest variance and is computationally expensive.

    The standard deviation across folds is a model selection tool:
    If model A has 85.0 ± 0.5% and model B has 85.2 ± 3.1%, model A
    is more reliable despite the lower mean.

    For deep learning: full k-fold CV is expensive (requires k complete
    training runs). A practical compromise is stratified 3-fold CV.

    CRITICAL: The test set is NEVER part of the cross-validation loop.
    CV is used to select the model and hyperparameters. The test set
    is still used once at the very end on the final selected model.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — THE VALIDATION LOOP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Anatomy of the Training + Validation Loop

Every well-structured training run interleaves training steps with
periodic validation passes:

    for epoch in range(max_epochs):

        # ── TRAINING PHASE ───────────────────────────────────────────────
        model.train()                   ← enables dropout, BN train mode
        train_loss_accum = 0.0

        for batch in train_loader:
            optimiser.zero_grad()
            outputs = model(batch.inputs)
            loss = criterion(outputs, batch.targets)
            loss.backward()
            optimiser.step()
            train_loss_accum += loss.item()

        train_loss = train_loss_accum / len(train_loader)

        # ── VALIDATION PHASE ─────────────────────────────────────────────
        model.eval()                    ← disables dropout, BN eval mode
        val_loss_accum = 0.0

        with torch.no_grad():           ← no gradient tape needed
            for batch in val_loader:
                outputs = model(batch.inputs)
                loss = criterion(outputs, batch.targets)
                val_loss_accum += loss.item()

        val_loss = val_loss_accum / len(val_loader)

        # ── LOGGING & DECISIONS ──────────────────────────────────────────
        log(epoch, train_loss, val_loss)
        scheduler.step(val_loss)        ← LR decay (if ReduceLROnPlateau)
        checkpoint.step(val_loss)       ← save best model
        early_stop.step(val_loss)       ← check stopping criterion


### The Three Critical Switches

These three lines are the most commonly forgotten and most damaging bugs
in training loops:

    1. model.train()  ←  before the training loop
       Activates: Dropout (random masking ON)
                  BatchNorm (computes batch statistics, updates running stats)
       If forgotten: Dropout is disabled during training (no regularisation).
                     BN uses inference statistics — statistics mismatch.

    2. model.eval()   ←  before the validation loop
       Activates: Dropout OFF (all neurons active)
                  BatchNorm uses running mean/variance (not batch statistics)
       If forgotten: Validation loss is computed WITH dropout — different neurons
                     are dropped each time, making val loss stochastic and
                     higher than the true val loss. Model selection is corrupted.

    3. torch.no_grad()  ←  context manager around the validation loop
       Effect: Disables gradient computation and the autograd tape.
       Why it matters:
           - Memory: autograd stores intermediate activations for backward pass.
             During eval, you never call .backward() — storing activations wastes
             memory. On large models, forgetting no_grad() can cause OOM errors.
           - Speed: gradient computation adds ~20–30% overhead. Eval is faster
             without it.
       If forgotten: Correct results, but slower and more memory-hungry.


### How Often to Validate

    Validation frequency is a tradeoff between information and overhead:

    Per-epoch validation:
        Standard for most tasks. After every pass through the training set,
        evaluate on the validation set. Loss curves are smooth and
        directly comparable epoch-to-epoch.

    Per-N-steps validation (step-based):
        For very large datasets where one epoch takes hours.
        Validate every N gradient steps (e.g., every 1,000 steps).
        Allows more frequent feedback without waiting for a full epoch.

    Validate MORE frequently when:
    - Training is fast (small model, small dataset)
    - You suspect instability (val loss might spike and recover)
    - Using a learning rate schedule that triggers on val loss (ReduceLROnPlateau)

    Validate LESS frequently when:
    - Each validation pass is expensive (large val set, slow model)
    - The model is large and the overhead is significant
    - Training is stable and you only need end-of-epoch summaries


### Accumulating Metrics Correctly

A subtle bug: do NOT average per-batch losses naively if batches
have different sizes (e.g., the last batch is smaller):

    WRONG (unweighted average of batch losses):
    total_loss = sum(batch_loss for each batch) / num_batches
    → Each batch gets equal weight regardless of size.
    → The last batch (if smaller) is over-weighted.

    CORRECT (weighted by sample count):
    total_loss = sum(batch_loss × batch_size for each batch) / total_samples
    → Each SAMPLE gets equal weight.

    In code:
    val_loss_sum = 0.0
    val_n = 0
    for batch in val_loader:
        outputs = model(batch.inputs)
        loss = criterion(outputs, batch.targets)   # mean loss per batch
        val_loss_sum += loss.item() * len(batch.targets)
        val_n += len(batch.targets)
    val_loss = val_loss_sum / val_n


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — LOSS AS THE PRIMARY TRAINING SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Why Monitor Loss, Not Just Accuracy

A common mistake: monitoring only accuracy and ignoring loss.
Loss is a MORE INFORMATIVE signal during training for several reasons:

    1. Continuity:
       Accuracy changes in discrete jumps of 1/n (one example flips).
       Loss changes continuously — every gradient update moves loss,
       even if no prediction changes. Small improvements are invisible
       in accuracy but visible in loss.

    2. Sensitivity to confidence:
       If the model correctly predicts class A with probability 0.51
       on epoch 1, and with probability 0.99 on epoch 100 — accuracy
       is unchanged (both correct), but loss has dropped dramatically.
       Loss tracks CALIBRATION progress that accuracy cannot see.

    3. Early warning of problems:
       Loss spikes (sudden large increases) signal instability, gradient
       explosions, or corrupted batches. Accuracy spikes are rarely visible
       because a spike in loss may not flip any individual prediction.

    Diagram 3 — Loss vs Accuracy: What Each Reveals

    Epoch:          1     10    20    30    40    50
    Train accuracy: 60%   72%   80%   82%   83%   83%   ← looks like it plateaued
    Train loss:     1.42  0.91  0.52  0.38  0.27  0.19  ← still improving!
    Val accuracy:   58%   70%   78%   79%   79%   78%
    Val loss:       1.51  0.98  0.61  0.55  0.59  0.68  ← overfitting since epoch 30!

    Reading accuracy alone: model looks stable from epoch 30–50.
    Reading loss: the model is overfitting from epoch 30 (val loss rising
    while train loss falls). Early stopping on loss catches this.
    Early stopping on accuracy would miss it entirely.


### The Four Loss Curve Patterns

    Diagram 4 — Canonical Loss Curve Shapes:

    Pattern 1: GOOD CONVERGENCE
    Loss
      │╲
      │ ╲
      │  ╲─╮
      │    ╰──────────────────── both curves flat and close
      └─────────────────── Epoch
    → Training and validation converge to similar values.
      Stop here. The model has generalised well.

    Pattern 2: OVERFITTING
    Loss
      │  training loss           validation loss
      │╲                              ╱────────
      │ ╲                        ────╯
      │  ╲──────────────────────╯
      └─────────────────── Epoch
    → Validation loss reaches a minimum then rises.
      Training loss keeps falling.
    → Add regularisation, reduce model capacity, add data.

    Pattern 3: UNDERFITTING (HIGH BIAS)
    Loss
      │
      │╲  ╲ training + val loss both plateau HIGH
      │ ╲──╰────────────────────────────────────
      └─────────────────── Epoch
    → Both curves plateau at a high loss value.
      Model cannot represent the target function.
    → Increase model capacity, train longer,
      reduce regularisation, check for data issues.

    Pattern 4: UNSTABLE TRAINING
    Loss
      │    ╱╲     ╱╲   ╱╲
      │╲  ╱  ╲   ╱  ╲ ╱
      │ ╲╱    ╲ ╱    ╲╱
      │         ╱
      └─────────────────── Epoch
    → Loss oscillates or spikes.
    → Learning rate too high, gradient explosion,
      corrupted batches, or inappropriate loss function.


### Diagnosing Spikes and Anomalies in Loss Curves

    LOSS SPIKE (single sharp increase, then recovery):
    Cause: a single corrupted batch (NaN or Inf in data), gradient
           explosion on a rare example, or a very large learning rate step.
    Action: check data preprocessing for NaN/Inf, add gradient clipping,
            inspect the dataloader for corrupted files.

    LOSS CLIFF (sudden catastrophic increase, no recovery):
    Cause: gradient explosion. Gradients have become so large that a
           single step moves the weights to a completely different region
           of the loss landscape.
    Action: add gradient clipping (clip_norm = 1.0), reduce learning rate.

    LOSS PLATEAU (training loss stops decreasing but val loss is also flat):
    Cause: learning rate too low, trapped in a flat region (saddle point),
           or the model has genuinely converged.
    Action: check if the learning rate is too small. Try a cyclic LR
            or a learning rate warmup restart. If val loss is also flat
            and at an acceptable value, convergence is genuine.

    LOSS DIVERGENCE (both train and val loss increase monotonically):
    Cause: learning rate far too large, or fundamentally incorrect setup
           (wrong loss function, incorrect label mapping).
    Action: reduce learning rate by 10×. Check label and output shapes.
            Verify the loss function matches the task.

    VAL LOSS LOWER THAN TRAIN LOSS (unusual):
    Cause: (a) dropout is active during training (training loss is measured
               WITH dropout noise; val loss is not) — this is normal.
           (b) training set is harder than the validation set — check your
               split for distribution mismatch.
           (c) data augmentation is only applied during training — training
               loss is measured on augmented (harder) inputs; val loss on
               clean inputs. This is expected and benign.
    Action: (a) is harmless; (b) requires checking the split; (c) is fine.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — CLASSIFICATION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Confusion Matrix

All classification metrics derive from the confusion matrix.
For binary classification (positive = event of interest):

    ┌────────────────────────────────────────────────────────┐
    │                   │ Predicted Positive │ Predicted Neg. │
    ├────────────────────────────────────────────────────────┤
    │ Actual Positive   │ True Positive (TP) │ False Neg (FN) │
    │ Actual Negative   │ False Positive (FP)│ True Neg. (TN) │
    └────────────────────────────────────────────────────────┘

    TP: Model says positive, it IS positive.  (correct)
    TN: Model says negative, it IS negative.  (correct)
    FP: Model says positive, it is NEGATIVE.  (Type I error — false alarm)
    FN: Model says negative, it is POSITIVE.  (Type II error — missed detection)

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
             = fraction of all predictions that were correct

    Accuracy is MISLEADING for imbalanced classes:
    Dataset: 990 negatives, 10 positives.
    A model that always predicts "negative" achieves 99% accuracy
    while being completely useless (TP=0, detects nothing).


### Precision, Recall, and the Precision-Recall Tradeoff

    Precision = TP / (TP + FP)
               "Of all predictions of positive, what fraction are correct?"
               Measures the QUALITY of positive predictions.

    Recall    = TP / (TP + FN)
               "Of all actual positives, what fraction did we detect?"
               Measures the COVERAGE of positive predictions.

    These two metrics are in fundamental tension:

    ↑ Classification threshold (stricter about what counts as positive):
        → Fewer positives predicted → FP decreases → PRECISION improves
        → Some true positives are now missed → FN increases → RECALL drops

    ↓ Classification threshold (more liberal about what counts as positive):
        → More positives predicted → FP increases → PRECISION drops
        → More true positives are detected → FN decreases → RECALL improves

    Diagram 5 — Precision-Recall Tradeoff at Different Thresholds:

                    threshold=0.9   threshold=0.5   threshold=0.1
    Precision:         0.95             0.80             0.45
    Recall:            0.30             0.72             0.93

    High precision (threshold=0.9): only very confident positives are
    flagged. Almost all flags are correct, but many true positives are missed.

    High recall (threshold=0.1): almost all positives are flagged.
    But many flags are false alarms.

    DOMAIN DETERMINES WHICH TO OPTIMISE:
    Medical diagnosis (cancer screening): HIGH RECALL
        → Missing a positive (FN) is life-threatening.
        → False alarms (FP) cause extra tests but not death.

    Email spam filter: HIGH PRECISION
        → A false alarm (important email in spam) is very costly.
        → Missing some spam (FN) is annoying but acceptable.

    Fraud detection: context-dependent
        → Low-value transactions: high precision (user friction)
        → High-value transactions: high recall (financial loss)


### F1 Score and Fβ

F1 is the HARMONIC MEAN of precision and recall:

    F1 = 2 · (Precision · Recall) / (Precision + Recall)

    The harmonic mean penalises EXTREME IMBALANCE between the two.
    A model with Precision=1.0 and Recall=0.01 has F1=0.02 — penalised
    heavily for the near-zero recall.
    A model with Precision=0.0 and Recall=1.0 also has F1=0.0.
    The harmonic mean forces both to be high simultaneously.

    Why harmonic, not arithmetic mean?
    Arithmetic: (1.0 + 0.01) / 2 = 0.505 — falsely high for a useless model.
    Harmonic:   2·(1.0·0.01)/(1.0+0.01) = 0.0198 — correctly low.

    Fβ generalises F1 by weighting recall β² times more than precision:

    Fβ = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall)

    β = 1:   Standard F1 (equal weight)
    β = 2:   F2 — recall twice as important (use when missing positives
              is more costly than false alarms — e.g., medical diagnosis)
    β = 0.5: F0.5 — precision twice as important (use when false alarms
              are more costly — e.g., spam filtering, recommendation systems)


### Multi-class Averaging Strategies

For K > 2 classes, precision, recall, and F1 are computed per-class
then aggregated:

    MACRO averaging:
        Compute metric for each class, take the unweighted mean.
        F1_macro = (F1_class1 + F1_class2 + ... + F1_classK) / K
        Treats all classes EQUALLY regardless of frequency.
        Use when: every class matters equally (rare class is as important
        as frequent classes). Will be pulled down by poor rare-class performance.

    MICRO averaging:
        Pool all TP, FP, FN across classes, then compute the metric.
        F1_micro = 2·TP_total / (2·TP_total + FP_total + FN_total)
        Equivalent to accuracy for multi-class problems.
        Treats all EXAMPLES equally regardless of class.
        Use when: the majority class dominates and you care about overall
        correctness. A model that ignores a tiny class has high micro F1.

    WEIGHTED averaging:
        Weight each class's metric by its support (number of examples):
        F1_weighted = Σk (support_k / n) · F1_k
        Treats all examples equally but shows the aggregate picture
        accounting for class frequencies.
        Use when: class imbalance exists and you want to account for it
        in a single-number summary without ignoring rare classes.

    EXAMPLE (3-class, imbalanced: class A=800, B=150, C=50):
    F1_A=0.92, F1_B=0.71, F1_C=0.35
    Macro F1    = (0.92 + 0.71 + 0.35) / 3 = 0.660
    Micro F1    ≈ 0.88  (dominated by class A)
    Weighted F1 = (800×0.92 + 150×0.71 + 50×0.35) / 1000 = 0.865


### ROC-AUC: Threshold-Free Classification Performance

The Receiver Operating Characteristic (ROC) curve plots:
    x-axis: False Positive Rate = FP / (FP + TN)  [= 1 - Specificity]
    y-axis: True Positive Rate  = TP / (TP + FN)  [= Recall]
at every possible classification threshold.

    Diagram 6 — ROC Curve Interpretation:

    TPR
    1.0 │         ╭──────────────────────
        │      ╭──╯
        │   ╭──╯  ← good classifier (AUC ≈ 0.9)
        │  ╭╯
        │ ╱        ← random classifier (AUC = 0.5, diagonal)
        │╱
    0.0 └──────────────────────── FPR
        0.0                       1.0

    AUC (Area Under the Curve) = probability that the model ranks
    a randomly chosen positive example HIGHER than a randomly chosen
    negative example.

    AUC = 1.0: perfect classifier at some threshold
    AUC = 0.5: random classifier (no discriminative power)
    AUC < 0.5: worse than random (labels may be inverted)

    AUC is THRESHOLD-INVARIANT — it measures the overall separability of
    the model's score distribution, regardless of where you set the decision
    boundary. This makes it useful for:
    - Comparing models when you haven't decided on a threshold yet
    - Imbalanced classes (robust to class frequency)
    - When different deployment scenarios require different thresholds

    LIMITATION of AUC: it averages performance across ALL thresholds,
    including thresholds that are never practically used. In high-precision
    applications (rare events), PRAUC (area under precision-recall curve)
    is more informative than ROC-AUC.


### PRAUC: When Positive Class Is Rare

For highly imbalanced problems (< 5% positive rate), ROC-AUC can be
misleadingly optimistic because the TN population is large:

    Consider: 1% positive rate, AUC = 0.95.
    At FPR = 0.05 (5% of negatives are false alarms), the model still
    finds most positives — but 5% of 99% = ~5% of ALL predictions are
    false alarms, swamping the true positives.

    PRAUC (area under Precision-Recall curve) directly measures
    performance on the positive class:
    - Precision depends on TP/(TP+FP)  — does not include TN
    - Recall depends on TP/(TP+FN)     — does not include TN
    - Both metrics are blind to the (usually large) TN population

    Baseline PRAUC for a random classifier = fraction of positives = p
    At 1% positive rate: random PRAUC = 0.01, not 0.5 (unlike ROC-AUC).
    This makes PRAUC harder to inflate artificially.

    Rule of thumb:
    Positive rate > 10%: ROC-AUC is appropriate
    Positive rate < 10%: PRAUC is more informative and harder to game


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — REGRESSION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Core Regression Metrics

    MSE  (Mean Squared Error):
    MSE = (1/n) · Σᵢ (yᵢ - ŷᵢ)²

    Properties:
    - Differentiable everywhere → directly usable as a loss function
    - PENALISES OUTLIERS heavily (errors are squared)
    - Same units as y² (not directly interpretable in the same units as y)
    - Sensitive to outliers: a single prediction off by 10× has 100× the
      contribution of a prediction off by 1×

    RMSE (Root Mean Squared Error):
    RMSE = sqrt(MSE)

    Properties:
    - Same units as y — directly interpretable
    - Still penalises outliers (still derived from squared errors)
    - The most widely reported regression metric

    MAE  (Mean Absolute Error):
    MAE = (1/n) · Σᵢ |yᵢ - ŷᵢ|

    Properties:
    - Robust to outliers (errors are not squared)
    - Corresponds to the MEDIAN as the optimal predictor (vs MSE which
      corresponds to the MEAN)
    - Not differentiable at 0 (requires subgradient; use Huber loss
      as a smooth approximation)
    - Better choice when outliers are expected and should not dominate


### R² (Coefficient of Determination)

    R² = 1 - SS_res / SS_tot
       = 1 - Σᵢ(yᵢ - ŷᵢ)² / Σᵢ(yᵢ - ȳ)²

    Where SS_res = sum of squared residuals (model error)
          SS_tot = total sum of squares (variance of y)

    Interpretation:
    R² = 1.0: model explains all variance in y (perfect fit)
    R² = 0.0: model explains no variance — equivalent to always
               predicting the mean of y
    R² < 0:   model is WORSE than always predicting the mean
               (possible when evaluating on out-of-distribution data)

    R² is DIMENSIONLESS and SCALE-FREE — it measures relative improvement
    over the simplest possible baseline (predicting the mean).

    R² is useful for:
    - Comparing models across different datasets (scale-free)
    - Reporting what fraction of variance the model explains
    - Sanity-checking whether the model is better than the trivial baseline


### MAPE and Percentage Errors

    MAPE = (100/n) · Σᵢ |(yᵢ - ŷᵢ) / yᵢ|

    Useful when the RELATIVE error matters more than the absolute error.
    Example: an error of $100 on a $10,000 prediction (1%) is very
    different from an error of $100 on a $200 prediction (50%).

    LIMITATION: Undefined when yᵢ = 0 (common in many real datasets).
    Asymmetric: over-predictions (ŷ > y) are bounded at 100% error,
    but under-predictions (ŷ < y) are unbounded.

    Better alternative: sMAPE (symmetric MAPE) or MASE (Mean Absolute
    Scaled Error, which normalises by the in-sample MAE of a naive forecast).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — METRICS FOR LANGUAGE AND GENERATION TASKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Perplexity (Language Models)

Perplexity is the primary training-time metric for language models:

    Perplexity = exp(-1/N · Σᵢ log P(wᵢ | w₁,...,wᵢ₋₁))
               = exp(cross-entropy loss)

    Interpretation: the MODEL'S AVERAGE UNCERTAINTY per predicted token.
    Perplexity = K means the model is, on average, as uncertain as
    choosing uniformly among K equally likely tokens.

    Perplexity = 10:   good language model
    Perplexity = 100:  mediocre model
    Perplexity = 1:    perfectly predicting the next token (impossible in practice)
    Perplexity = |V|:  the model is completely confused (uniform over vocabulary)

    Perplexity is a monotone function of cross-entropy loss — minimising
    loss is equivalent to minimising perplexity. It is preferred as a
    reporting metric because it is more interpretable than a raw loss value.

    IMPORTANT CAVEAT: Perplexity is not comparable across models that
    use different tokenisers or different context lengths. A model with
    a byte-level tokeniser will have very different perplexity than one
    with a word-level tokeniser, even with identical language modelling ability.


### BLEU (Machine Translation, Text Generation)

BLEU (Bilingual Evaluation Understudy, Papineni et al., 2002) measures
the overlap between a generated sequence and one or more reference sequences
using n-gram precision:

    BLEU = BP · exp(Σₙ wₙ · log pₙ)

    Where:
        pₙ = modified n-gram precision (fraction of generated n-grams
             that appear in the reference, clipped to prevent gaming)
        wₙ = weight for each n-gram order (typically 1/4 for n=1,2,3,4)
        BP = Brevity Penalty (penalises short outputs that "cherry-pick"
             easy n-grams: BP = exp(1 - r/c) if c < r, else 1)

    BLEU score ranges from 0 to 1 (or 0 to 100 as a percentage).

    Typical BLEU values in machine translation:
    < 10:  almost unusable output
    10–19: very rough translation
    20–29: understandable but with grammatical errors
    30–40: good translation, comparable to human quality
    > 40:  very high quality, approaching human level

    KNOWN LIMITATIONS of BLEU:
    - Measures word overlap, not semantic meaning.
      "The cat sat" and "A feline was seated" have BLEU=0 despite
      equivalent meaning.
    - Sensitive to tokenisation — comparisons are only valid when
      tokenisation is identical across evaluated models.
    - Does not correlate well with human judgement on short texts.
    - Not suitable for open-ended generation (chat, summarisation)
      where many valid outputs exist.

    Alternatives: ROUGE (summarisation), METEOR, BERTScore (embedding-based).


### BERTScore and Embedding-Based Metrics

BERTScore (Zhang et al., 2020) computes the similarity between
generated and reference texts using CONTEXTUAL EMBEDDINGS:

    For each token in the generated text, find the most similar token
    in the reference text (using cosine similarity of BERT embeddings).
    Precision = average max similarity for generated tokens.
    Recall = average max similarity for reference tokens.
    F1 = harmonic mean of precision and recall.

    Advantages over BLEU:
    - Captures SEMANTIC similarity, not just lexical overlap
    - More robust to paraphrase ("car" ~ "vehicle")
    - Correlates better with human judgements
    - Language-agnostic with multilingual models

    Disadvantage: requires a pre-trained BERT model — adds overhead
    and a dependency on the quality of the embedding model.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — MODEL CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What Is Calibration?

A model is WELL-CALIBRATED if its predicted probabilities match the
empirical frequencies of the events it predicts:

    "Of all examples where the model predicted 80% confidence,
    approximately 80% should actually be in the positive class."

    Calibration is separate from accuracy:
    - A model can be accurate but poorly calibrated
      (correct predictions but with the wrong confidence levels)
    - A model can be poorly calibrated but still rank examples correctly
      (AUC is high but probabilities are meaningless)

    Why calibration matters:
    - Decision thresholds: if the model says 70% confidence, you need
      that to mean approximately 70% probability to set a meaningful threshold.
    - Risk-weighted decisions: in insurance, medicine, and finance,
      decisions are made based on the MAGNITUDE of the predicted probability,
      not just its rank.
    - Uncertainty quantification: downstream systems that use model outputs
      as inputs assume the probabilities are meaningful.


### Reliability Diagram (Calibration Curve)

Group predictions into M bins by predicted probability
(e.g., 0–10%, 10–20%, ..., 90–100%). In each bin, compare:
    - Mean predicted probability (x-axis)
    - Fraction of actual positives (y-axis)

    Diagram 7 — Calibration Curve Patterns:

    Actual positive rate
    1.0 │                   ╱  ← perfect calibration (diagonal)
        │                 ╱
        │          ╱──●──●   ← overconfident (above diagonal)
        │       ●──╯         (model says 80%, only 60% are positive)
        │    ╱
        │ ●──          ← underconfident (below diagonal)
        │              (model says 30%, 50% are actually positive)
    0.0 └───────────────────────────── Predicted probability
        0.0                            1.0

    Perfect calibration: points lie on the diagonal y = x.
    Overconfident: points curve above the diagonal (model too confident).
    Underconfident: points curve below the diagonal.

    Neural networks trained with cross-entropy tend to be OVERCONFIDENT —
    this is a direct consequence of the logit-pushing behaviour of
    cross-entropy loss (see Module 15 — Label Smoothing for the mechanism).


### Expected Calibration Error (ECE)

ECE is a scalar summary of calibration error:

    ECE = Σₘ (|Bₘ| / n) · |acc(Bₘ) - conf(Bₘ)|

    Where:
        Bₘ = set of examples in bin m
        acc(Bₘ) = fraction of correct predictions in bin m (observed freq.)
        conf(Bₘ) = mean predicted confidence in bin m
        |Bₘ| / n = weight (fraction of examples in bin m)

    ECE = 0.0: perfectly calibrated
    ECE = 0.1: typical for uncalibrated deep neural networks
    ECE = 0.3: severely overconfident

    Limitation: sensitive to binning strategy. Some bins may have
    very few examples (especially the extreme 0–5% and 95–100% bins),
    making those bin-level estimates noisy.


### Post-hoc Calibration Methods

If a model is trained and found to be poorly calibrated, calibration
can be improved WITHOUT retraining:

    TEMPERATURE SCALING (Guo et al., 2017):
    The simplest and most effective method for neural networks.
    Divide all logits by a single scalar T before the softmax:

    p_calibrated(k) = softmax(z / T)

    T > 1: softens the distribution → reduces overconfidence
    T < 1: sharpens the distribution → increases confidence (rarely needed)
    T = 1: original output (no change)

    T is found by minimising NLL on the VALIDATION SET only.
    Only one parameter — cannot overfit. Takes seconds.
    Temperature scaling does not change ACCURACY — it only rescales
    probabilities. The argmax of the softmax is unchanged.

    PLATT SCALING:
    Fit a logistic regression on the model's raw logit outputs
    using the validation set. More flexible than temperature scaling
    but can overfit if the validation set is small.

    ISOTONIC REGRESSION:
    Non-parametric — fits a piecewise-constant monotone function
    mapping predicted probabilities to calibrated probabilities.
    Most flexible but most prone to overfitting. Use only with large
    validation sets.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — OVERFITTING VS UNDERFITTING: PRACTICAL DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Generalisation Gap as a Diagnostic Tool

    Generalisation gap = val_metric - train_metric
    (for loss: positive gap = overfitting;
     for accuracy: negative gap = overfitting)

    Gap = 0:        Perfect generalisation (or identical train/val distributions)
    Small gap:      Mild overfitting, acceptable in most settings
    Large gap:      Significant overfitting, requires intervention
    Gap is negative
    (val BETTER than train): Unusual — see Part 3 for explanations

    The gap should be interpreted relative to the absolute performance level:
    Train accuracy = 60%, Val accuracy = 58%: gap = 2%  (small; model underfits)
    Train accuracy = 98%, Val accuracy = 96%: gap = 2%  (small; both high)
    Train accuracy = 98%, Val accuracy = 60%: gap = 38% (large; severe overfit)


### Decision Tree: What Action to Take

    ┌────────────────────────────────────────────────────────────────────────┐
    │                   START                                                │
    │                     │                                                  │
    │          Is training loss low?                                         │
    │         ┌────────────────────┐                                         │
    │         NO                  YES                                        │
    │         │                    │                                         │
    │    Underfitting          Is val loss also low?                         │
    │    (high bias)          ┌──────────────────┐                          │
    │         │               YES                NO                         │
    │   ┌─────┴──────────┐    │                  │                          │
    │   │ Increase        │  GOOD FIT         Overfitting                   │
    │   │ capacity        │  (done)           (high variance)               │
    │   │ Train longer    │                   │                             │
    │   │ Reduce λ        │             ┌─────┴─────────────────────┐       │
    │   │ Check data      │             │ Add regularisation         │       │
    │   └─────────────────┘             │ Reduce model capacity      │       │
    │                                   │ Get more data              │       │
    │                                   │ Use early stopping         │       │
    │                                   │ Add data augmentation      │       │
    │                                   └────────────────────────────┘       │
    └────────────────────────────────────────────────────────────────────────┘


### Reading Validation Metrics: Common Mistakes

    MISTAKE 1: Reporting the final epoch metric (not the best):
    The model at the final epoch has often overfit past the optimal point.
    Always report the BEST VALIDATION METRIC achieved during training,
    which corresponds to the best checkpoint (early stopping target).

    MISTAKE 2: Evaluating on the validation set used for early stopping:
    If early stopping monitored val_loss and selected the best model
    by val_loss, then reporting that model's val_loss is slightly biased
    (you selected the best-by-chance low point). A true unbiased estimate
    requires a separate test set.

    MISTAKE 3: Model selection across many hyperparameter configurations
    using a single validation set:
    If you try 100 hyperparameter configurations and pick the one with
    the best validation performance, you have implicitly overfit to the
    validation set. This is called VALIDATION SET OVERFITTING or
    HYPERPARAMETER OVERFITTING. The more configurations you try,
    the larger the gap between reported val performance and true test performance.
    Solution: use a held-out test set that is NEVER used for selection.

    MISTAKE 4: Comparing models trained with different random seeds
    without accounting for variance:
    Model A (seed 1): 84.3%  vs  Model B (seed 1): 84.5%
    Is model B genuinely better?
    Run each model with 3-5 different seeds and compare distributions.
    If the confidence intervals overlap, neither model is clearly better.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — EVALUATION BEST PRACTICES & COMMON BUGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Evaluation Checklist

Before reporting any evaluation result, verify:

    DATA:
    □ Train/val/test splits are non-overlapping and stratified (or time-ordered)
    □ No data leakage: preprocessing fitted only on the training split
    □ Validation set distribution matches test set distribution
    □ Sufficient examples in each split for statistically stable estimates

    MODEL STATE:
    □ model.eval() is called before evaluation
    □ torch.no_grad() context is active during evaluation
    □ All model components (including submodules) are in eval mode
    □ The correct checkpoint is loaded (best, not final)

    METRIC COMPUTATION:
    □ Metrics are accumulated correctly (weighted by sample count)
    □ The threshold (for classification) is appropriate for the task
    □ Per-class metrics are reported for imbalanced problems
    □ Baseline performance is reported for comparison

    REPORTING:
    □ Val metrics are from the best checkpoint, not the final epoch
    □ Test metrics are reported only ONCE, on the FINAL model
    □ Multiple seeds are used if comparing models (report mean ± std)
    □ Statistical significance is assessed for small differences


### Data Leakage: The Silent Killer of Evaluation Validity

Data leakage is when information from the validation or test set
INFLUENCES the training process, making reported metrics overly optimistic.

    PREPROCESSING LEAKAGE (most common):
    Wrong:  Fit normalisation (mean, std) on the full dataset,
            then split into train/val/test.
    Effect: The training set's normalisation uses statistics from
            val and test examples. The model is indirectly "told"
            about val/test values during training.
    Correct: Split first. Fit preprocessing ONLY on the training split.
             Apply the training-fit transform to val and test.

    TARGET LEAKAGE:
    Including features in the training data that would not be available
    at prediction time (they are derived from the target or measured
    after the event of interest).
    Example: predicting hospital readmission using "discharge_notes"
    that include phrases like "patient likely to return for follow-up" —
    the note was written AFTER the outcome was known.

    TEMPORAL LEAKAGE (see Part 1):
    Using future data to predict past events due to incorrect splitting.

    DUPLICATION LEAKAGE:
    Near-duplicate examples in both training and test sets.
    The model memorises these examples rather than generalising.
    Solution: deduplication before splitting.


### Choosing the Right Metric for the Task

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Task type              │ Primary metric    │ Secondary / check       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Binary classification  │ ROC-AUC or PRAUC  │ F1, Precision, Recall   │
    │ (balanced classes)     │                   │                         │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Binary classification  │ PRAUC             │ F1 at operating         │
    │ (rare positive class)  │                   │ threshold               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Multi-class classif.   │ Macro F1          │ Per-class F1,           │
    │ (balanced)             │                   │ Accuracy                │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Multi-class classif.   │ Macro F1 +        │ Confusion matrix        │
    │ (imbalanced)           │ per-class metrics │                         │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Regression (general)   │ RMSE              │ MAE, R²                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Regression (outliers)  │ MAE               │ Huber loss, R²          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Language modelling     │ Perplexity        │ Val cross-entropy loss  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Machine translation    │ BLEU              │ BERTScore               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Text summarisation     │ ROUGE-L           │ BERTScore               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Object detection       │ mAP               │ AP per class, IoU       │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Image generation       │ FID               │ IS (Inception Score)    │
    └──────────────────────────────────────────────────────────────────────┘

    Golden rule: choose the metric that MOST CLOSELY REFLECTS THE COST
    of the types of errors in your application, not the metric that makes
    your model look best.

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
        from Training_Core.visuals.Evaluation_visual import (
            EVALUATION_VISUAL_HTML,
            EVALUATION_VISUAL_HEIGHT,
        )
        visual_html   = EVALUATION_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = EVALUATION_VISUAL_HEIGHT
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