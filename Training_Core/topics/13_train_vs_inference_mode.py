"""
Train vs Inference Mode
=======================

A model trained with PyTorch is not a static artefact — it is a
stateful object that behaves differently depending on which mode it
is operating in. The distinction between training mode and inference
mode is one of the most consequential and most frequently misunderstood
aspects of the PyTorch model lifecycle. Getting it wrong does not
always produce an obvious error. More often, it produces a model that
silently underperforms, outputs that are stochastic when they should
be deterministic, or a deployment that behaves nothing like the model
that was validated. Understanding exactly what changes between modes,
why it changes, and how to move between them correctly is essential
practical knowledge.

"""
import textwrap
import re

TOPIC_NAME   = "Train vs Inference Mode"
DISPLAY_NAME = "13 · Train vs Inference Mode"
ICON         = "🔀"
SUBTITLE     = "model.train() · model.eval() · no_grad · Dropout · BatchNorm · Deployment"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE FUNDAMENTAL DISTINCTION

### Why Two Modes Exist

A neural network serves two distinct purposes that have conflicting
requirements:

    DURING TRAINING:
    The model must explore the data distribution noisily. Stochasticity
    (from Dropout, batch statistics from BatchNorm) acts as regularisation.
    The model must also build a computation graph for backpropagation —
    storing intermediate activations so gradients can be computed.

    DURING INFERENCE:
    The model must produce DETERMINISTIC, STABLE predictions from a
    fixed set of weights. Stochasticity would make predictions unreliable.
    The computation graph is not needed — no gradients will be computed.
    Memory and compute efficiency matter: no intermediate activations
    need to be retained.

    These requirements are incompatible in a single fixed configuration.
    PyTorch resolves this by giving every nn.Module a .training flag
    that switches behaviour between the two modes.

    model.train()   → sets model.training = True  (recursively on all submodules)
    model.eval()    → sets model.training = False (recursively on all submodules)

    This is not merely a convention or a documentation hint.
    It changes the mathematical operations the model performs.


### The .training Flag

Every nn.Module in PyTorch exposes a .training attribute:

    model.training         → True or False
    model.layer1.training  → same as model.training (set recursively)

    model.train()  sets model.training = True  for self AND every submodule.
    model.eval()   sets model.training = False for self AND every submodule.

    You can verify the current mode at any layer:
    for name, module in model.named_modules():
        print(f"{name:30s}: training={module.training}")

    Important: model.train() and model.eval() DO NOT affect:
    - The values of the model's parameters (weights and biases).
    - The model's gradient tracking state (that is controlled by
      torch.no_grad() and parameter.requires_grad).
    - The device the model is on.
    They ONLY change the behavioural mode of mode-sensitive layers.

    Diagram 1 — What .training controls:

    model.train()                          model.eval()
    ┌───────────────────────────┐          ┌───────────────────────────┐
    │ Dropout:   ACTIVE         │          │ Dropout:   OFF            │
    │ BatchNorm: batch stats    │          │ BatchNorm: running stats  │
    │ Autograd:  UNAFFECTED  ←──┼──────────┼──► still active           │
    │ Weights:   UNAFFECTED  ←──┼──────────┼──► same values            │
    └───────────────────────────┘          └───────────────────────────┘


### The Three Controls and Their Separation

Three independent controls govern training vs inference behaviour.
They are commonly confused because all three should be set together
for inference — but they are conceptually separate:

    CONTROL 1 — Model mode (model.train() / model.eval()):
    Governs: Dropout, BatchNorm, and any custom mode-sensitive layers.
    Scope:   The model's entire computation graph.
    Default: nn.Module initialises in TRAIN mode (training=True).

    CONTROL 2 — Gradient computation (torch.no_grad() context):
    Governs: whether PyTorch builds a computation graph and retains
             intermediate activations for backpropagation.
    Scope:   All operations within the context manager.
    Default: gradients ARE computed (not disabled).

    CONTROL 3 — Parameter gradient tracking (param.requires_grad):
    Governs: whether a specific parameter will receive gradient updates.
    Scope:   A specific parameter tensor.
    Default: True for parameters created with nn.Module.

    These three controls are INDEPENDENT:
    model.eval() does NOT disable gradient computation.
    torch.no_grad() does NOT switch Dropout off.
    requires_grad=False does NOT switch BatchNorm to running statistics.

    ALL THREE must be set correctly for safe, efficient inference:
    model.eval()                        ← Dropout off, BN uses running stats
    with torch.no_grad():               ← no graph built, no activations stored
        outputs = model(inputs)         ← deterministic, memory-efficient


##### PART 2 — DROPOUT IN TRAIN VS EVAL MODE

### What Dropout Does in Each Mode

    TRAIN MODE (model.training = True):
    For each neuron, sample mᵢ ~ Bernoulli(1 - p):
        mᵢ = 1:  neuron output passes through, scaled by 1/(1-p)
        mᵢ = 0:  neuron output is zeroed out

    output_i = mᵢ · xᵢ / (1 - p)

    The mask is sampled INDEPENDENTLY for each example and each
    forward pass. Two identical inputs will produce different outputs.
    Expected value: E[output_i] = (1-p) · xᵢ/(1-p) = xᵢ  (unbiased)

    EVAL MODE (model.training = False):
    No mask is sampled. All neurons are active:
    output_i = xᵢ

    Because of inverted dropout (the 1/(1-p) scaling in train mode),
    the expected output in eval mode exactly matches the expected output
    in train mode. No additional rescaling is needed at test time.

    Diagram 2 — Dropout: Train vs Eval (p=0.5, 6 neurons):

    TRAIN (each pass different):         EVAL (always the same):
    Pass 1:  [x₁·2, 0,   x₃·2, 0,   x₅·2, 0  ]
    Pass 2:  [0,   x₂·2, 0,   x₄·2, 0,   x₆·2]
    Pass 3:  [x₁·2, x₂·2, 0,   0,   x₅·2, x₆·2]
                                          [x₁, x₂, x₃, x₄, x₅, x₆]

    Stochastic at train time → regularisation.
    Deterministic at eval time → stable predictions.


### What Happens When You Forget model.eval()

    Symptom 1 — Stochastic predictions:
    Running the same input through the model twice produces different
    outputs. A diagnostic check:

    outputs_1 = model(test_input)
    outputs_2 = model(test_input)
    are_equal = torch.allclose(outputs_1, outputs_2)
    print(f"Outputs deterministic: {are_equal}")
    # If False: model is in train mode (or uses other stochasticity)

    Symptom 2 — Systematically lower prediction magnitudes:
    With dropout rate p=0.5, on average half the neurons are zeroed.
    The output activation is (1-p) = 50% of what it would be without dropout.
    With inverted dropout this is corrected by the 1/(1-p) scaling —
    but only if the model was using inverted dropout (PyTorch's default).
    If the model used a non-inverted dropout variant, the output scale
    at inference will be half the training scale.

    Symptom 3 — Poor calibration:
    Dropout in train mode adds noise to the final logits. This inflates
    the variance of predictions. The model appears LESS confident than
    it actually is, pulling probabilities toward 0.5.

    Symptom 4 — Lower reported validation accuracy than expected:
    Each validation batch is evaluated with a different random mask.
    Some correct predictions may be flipped to wrong by an unlucky mask.
    The resulting validation loss is HIGHER and accuracy LOWER than the
    true (dropout-disabled) validation performance.
    This can cause early stopping to trigger prematurely, killing a
    training run that was actually performing well.


### Intentional Dropout at Inference: Monte Carlo Dropout

Keeping dropout ACTIVE at inference is intentional in ONE specific
scenario: Monte Carlo Dropout for uncertainty estimation.

    MC Dropout (Gal & Ghahramani, 2016):
    The model is treated as a Bayesian approximation. Multiple forward
    passes with different dropout masks approximate samples from the
    posterior distribution over model parameters.

    # MC Dropout: keep model in train mode during inference
    model.train()   ← intentionally NOT calling eval()

    predictions = []
    with torch.no_grad():           ← still disable gradient computation
        for _ in range(T):          ← T = number of MC samples (e.g. 100)
            pred = model(test_input)
            predictions.append(pred)

    predictions = torch.stack(predictions)  # shape (T, batch, classes)
    mean_pred   = predictions.mean(dim=0)   # mean prediction
    uncertainty = predictions.var(dim=0)    # epistemic uncertainty

    This gives a distribution over predictions, not a single prediction.
    The variance across T passes estimates how uncertain the model is.

    Use MC Dropout when:
    ✓ Epistemic uncertainty is needed (medical, autonomous systems, active learning)
    ✓ The model uses Dropout (MCDropout cannot be applied to Dropout-free models)
    ✓ T forward passes are affordable at inference time

    Do NOT confuse MC Dropout with forgetting model.eval(). They look
    identical in code (both call model.train() during inference) but have
    entirely different intent. MC Dropout requires deliberate design;
    forgetting eval() is always a bug.


##### PART 3 — BATCHNORM IN TRAIN VS EVAL MODE

### BatchNorm's Two Sets of Statistics

BatchNorm (Ioffe & Szegedy, 2015) maintains TWO distinct sets of statistics:

    BATCH STATISTICS (used in train mode):
    μ_B  = mean of the current mini-batch: (1/B) · Σᵢ xᵢ
    σ²_B = variance of the current mini-batch: (1/B) · Σᵢ (xᵢ - μ_B)²
    Computed fresh from the current batch on every forward pass.

    RUNNING STATISTICS (used in eval mode):
    running_mean = exponential moving average of μ_B across all batches
    running_var  = exponential moving average of σ²_B across all batches

    Update rule (applied during train mode):
    running_mean ← (1-momentum) · running_mean + momentum · μ_B
    running_var  ← (1-momentum) · running_var  + momentum · σ²_B

    Default momentum = 0.1 in PyTorch (a higher value gives more
    weight to recent batches).

    The BN transformation:
    x̂ᵢ = (xᵢ - μ) / sqrt(σ² + ε)
    yᵢ = γ · x̂ᵢ + β

    TRAIN mode: μ = μ_B, σ² = σ²_B  (batch statistics)
    EVAL mode:  μ = running_mean, σ² = running_var  (accumulated statistics)


### Why BatchNorm Behaviour Changes Between Modes

    TRAIN MODE — batch statistics:
    Why: the running statistics are not yet reliable at the start of
    training (they start at 0 and 1). Using batch statistics gives
    an unbiased normalisation from step 1.
    Also: the stochasticity from using batch statistics (each batch
    has slightly different mean/variance) is an implicit regulariser.

    EVAL MODE — running statistics:
    Why: at inference, there is no "batch" in the meaningful sense.
    A single-example inference (batch size 1) would have μ_B = x₁
    and σ²_B = 0 — completely degenerate statistics that produce
    x̂ = 0/sqrt(0+ε) ≈ 0 for every example. This collapses the
    normalised output to near-zero, destroying the network's learned
    representations.

    The running statistics accumulated over training represent a stable
    estimate of the data distribution's mean and variance. At inference,
    these stable estimates are used instead.

    Diagram 3 — BatchNorm at Batch Size 1:

    TRAIN mode (batch=1):             EVAL mode (batch=1):
    μ_B   = x₁  (trivially)           μ = running_mean  (stable estimate)
    σ²_B  = 0   (trivially)            σ² = running_var  (stable estimate)
    x̂ = (x₁ - x₁) / sqrt(0+ε) = 0   x̂ = (x₁ - μ_run) / sqrt(σ²_run+ε)
    → OUTPUT IS ALWAYS ZERO!           → correct normalisation


### The Running Statistics Accumulation Problem

Running statistics only represent the data distribution accurately
if they were updated over a REPRESENTATIVE sample of the training data.

    Problem 1 — Not training long enough:
    If a model is evaluated before the running statistics have converged,
    the running mean/var are still far from the true dataset statistics.
    Eval mode uses these inaccurate statistics → poor normalisation.
    Rule: running statistics need approximately 1000/momentum steps to
    converge. At momentum=0.1: ~10,000 steps for convergence.

    Problem 2 — Domain shift:
    The running statistics represent the training data distribution.
    If the test distribution has different mean/variance, the normalisation
    is wrong for the test data.
    Symptom: model works well on val (same distribution as train) but
    fails on held-out test data from a different source.

    Problem 3 — Fine-tuning with frozen BatchNorm:
    When fine-tuning on a new dataset with BatchNorm layers frozen in
    eval mode, the running statistics from the original dataset may not
    match the new dataset.
    Fix: either unfreeze BatchNorm layers (let running stats update) or
    switch frozen BN layers to use batch statistics during fine-tuning.

    Problem 4 — Small batch size causing poor running stat estimates:
    With batch size 2-4, each batch's statistics are very noisy.
    The running statistics accumulate noise rather than signal.
    Fix: use GroupNorm or LayerNorm instead (both are batch-size independent).


### Manually Resetting Running Statistics

After a domain shift or when re-using a pre-trained BN model on new data:

    def reset_bn_running_stats(model):
        '''Reset all BatchNorm running statistics to their initial values.'''
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                    nn.BatchNorm3d, nn.SyncBatchNorm)):
                module.reset_running_stats()
                # Sets: running_mean = 0, running_var = 1, num_batches_tracked = 0

    After resetting: run the model in TRAIN mode over the new dataset for
    at least 1000/momentum steps before switching to EVAL mode for evaluation.

    Alternatively, use track_running_stats=False to always use batch statistics:
    nn.BatchNorm2d(channels, track_running_stats=False)
    → This disables running stat accumulation entirely. The BN always uses
      batch statistics, even in eval mode. Useful for quick experiments but
      not safe for inference at batch size 1.


### BatchNorm Alternatives and Their Mode Behaviour

Not all normalisation layers change between train and eval modes:

    ┌────────────────────────────────────────────────────────────────────┐
    │ Normalisation   │ Train mode     │ Eval mode      │ Mode-sensitive │
    ├────────────────────────────────────────────────────────────────────┤
    │ BatchNorm       │ Batch stats    │ Running stats  │ YES            │
    │ SyncBatchNorm   │ Global batch   │ Running stats  │ YES            │
    │                 │ stats (multi-  │                │                │
    │                 │ GPU)           │                │                │
    ├────────────────────────────────────────────────────────────────────┤
    │ LayerNorm       │ Instance stats │ Instance stats │ NO             │
    │ GroupNorm       │ Group stats    │ Group stats    │ NO             │
    │ InstanceNorm    │ Instance stats │ Instance stats │ NO             │
    └────────────────────────────────────────────────────────────────────┘

    LayerNorm, GroupNorm, and InstanceNorm normalise per-example (or per
    group within an example) and do not maintain running statistics.
    They behave IDENTICALLY in train and eval mode.
    This is one reason LayerNorm is preferred in transformers: there is
    one fewer mode-sensitive component to manage.


##### PART 4 — torch.no_grad() AND THE AUTOGRAD ENGINE

### What PyTorch Builds During a Forward Pass

During a forward pass in TRAIN mode (with gradient computation enabled),
PyTorch does two things simultaneously:

    1. Computes the numerical output of each operation (the forward pass).
    2. Builds a COMPUTATION GRAPH — a directed acyclic graph that records:
       - Which operations were applied.
       - Which tensors were inputs and outputs.
       - The intermediate activation values needed to compute gradients.

    This computation graph is stored in memory until .backward() is called.
    It is the mechanism by which PyTorch computes gradients via reverse-mode
    automatic differentiation (backpropagation).

    Diagram 4 — What Autograd Builds During Forward:

    input → Linear → ReLU → Linear → output → loss
        ↓         ↓       ↓        ↓
    [saves]   [saves]  [saves]  [saves]  ← intermediate activations
                                           stored for backward pass
        └─────────────────────────────────┘
              computation graph in memory

    Memory cost of the computation graph:
    Roughly proportional to the number of activations × batch size × depth.
    For a large ResNet-50 at batch size 256: ~4 GB of activation memory.
    For a GPT-2 medium at batch size 32: ~8 GB of activation memory.
    This activation memory is WASTED at inference — no backward pass will
    be called and the graph will be immediately discarded.


### torch.no_grad(): Disabling the Autograd Engine

    torch.no_grad() is a context manager that disables gradient computation
    for all operations within its scope:

    with torch.no_grad():
        outputs = model(inputs)   ← no graph built, no activations stored

    What changes with torch.no_grad():
    1. No computation graph is built. Intermediate activations are NOT stored.
    2. Tensors computed within the context have requires_grad=False.
    3. Memory usage drops by ~40-60% (no activation storage).
    4. Compute time drops by ~20-30% (no graph bookkeeping overhead).
    5. Tensor operations are slightly faster (simpler internal dispatch).

    What DOES NOT change with torch.no_grad():
    1. model.training flag — Dropout and BatchNorm are unaffected.
    2. The numerical values computed — outputs are mathematically identical
       to a forward pass with gradients enabled.
    3. Existing tensors' requires_grad status.

    Incorrect assumption: "torch.no_grad() disables Dropout."
    Dropout is controlled by model.training, not by gradient tracking.
    model.eval() + torch.no_grad() are BOTH required for safe inference.


### torch.no_grad() vs torch.inference_mode()

PyTorch offers a stronger version: torch.inference_mode()

    torch.no_grad():
    - Disables gradient computation.
    - Created tensors have requires_grad=False but are still tracked.
    - Tensors created inside can be used outside (gradients computed there).
    - Can be nested with gradient contexts (grad re-enabled within no_grad).

    torch.inference_mode():
    - Disables gradient computation MORE aggressively.
    - Created tensors are in a special "inference mode" — they cannot
      be used in any operation that would require gradient tracking.
    - Slightly faster than no_grad (fewer internal checks).
    - Cannot be nested with gradient contexts.

    In code:
    with torch.inference_mode():
        outputs = model(inputs)   # fastest, safest inference path

    RECOMMENDATION:
    Use torch.inference_mode() when:
    - You are doing pure inference and the outputs will never be used
      in gradient computation (standard deployment scenario).
    Use torch.no_grad() when:
    - You need the output tensors to potentially participate in gradient
      computation later (e.g., validation loss computation where you might
      need the gradient with respect to the loss for analysis, not for update).
    - You are implementing algorithms like MAML where test-time computation
      must remain in the gradient graph.

    Diagram 5 — Gradient Context Comparison:

    Context              │ Graph built │ requires_grad │ Speed    │ Nestable
    ─────────────────────┼─────────────┼───────────────┼──────────┼─────────
    Default (no context) │ YES         │ YES           │ slowest  │ N/A
    torch.no_grad()      │ NO          │ NO            │ fast     │ YES
    torch.inference_mode │ NO          │ NO (strict)   │ fastest  │ NO


### requires_grad: Parameter-Level Control

    requires_grad is a property of individual PARAMETER TENSORS, not
    of the model's mode:

    param.requires_grad = True   ← this parameter will receive gradients
    param.requires_grad = False  ← this parameter will NOT be updated

    Freezing specific layers (e.g., a pre-trained backbone):
    for param in model.backbone.parameters():
        param.requires_grad = False

    Only the classifier head is trained:
    for param in model.head.parameters():
        param.requires_grad = True

    model.requires_grad_(False)   ← freeze the entire model
    model.requires_grad_(True)    ← unfreeze the entire model

    IMPORTANT: requires_grad=False does NOT switch BatchNorm to eval mode.
    A frozen BatchNorm layer with requires_grad=False in TRAIN mode:
    - Still uses batch statistics (not running stats).
    - Still updates running_mean and running_var.
    - Still adds stochasticity from batch statistics.
    → Call model.eval() or freeze BN explicitly if you want eval-mode BN
      behaviour during fine-tuning with a frozen backbone.

    Freezing BatchNorm specifically:
    def freeze_bn(model):
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                    nn.BatchNorm3d)):
                module.eval()             ← use running stats
                module.requires_grad_(False)  ← do not update γ and β


##### PART 5 — CUSTOM LAYERS AND IMPLEMENTING MODE SENSITIVITY

### Writing a Mode-Sensitive Custom Module

Any custom nn.Module can read self.training to implement different
behaviour in train and eval modes:

    class GaussianNoise(nn.Module):
        '''
        Add Gaussian noise during training for regularisation.
        Pass through unchanged at inference.
        '''

        def __init__(self, std=0.1):
            super().__init__()
            self.std = std

        def forward(self, x):
            if self.training:             ← checks self.training flag
                noise = torch.randn_like(x) * self.std
                return x + noise
            return x                      ← deterministic at inference


    class FeatureDropout(nn.Module):
        '''
        Drop entire feature channels (spatial dropout) during training.
        Identity during inference.
        '''

        def __init__(self, p=0.1):
            super().__init__()
            self.p = p

        def forward(self, x):
            # x shape: (batch, channels, height, width)
            if self.training and self.p > 0:
                # Sample a binary mask per channel
                mask = torch.bernoulli(
                    torch.full((x.shape[0], x.shape[1], 1, 1),
                               1 - self.p, device=x.device)
                ) / (1 - self.p)        # inverted dropout scaling
                return x * mask
            return x

    The key principle: always read self.training, never pass training
    as a constructor argument or a manual flag. The parent nn.Module
    machinery handles propagation of the flag via train()/eval().


### Mode-Sensitive Behaviour in Common Architectures

    TRANSFORMER ATTENTION:
    Attention dropout (applied to the attention weight matrix):
    attn_weights = softmax(QK^T / sqrt(d_k))
    attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)

    In eval mode (self.training=False): attention weights are not dropped.
    Every token attends to every other token with the full computed weight.
    Predictions are fully deterministic and stable.

    VARIATIONAL AUTOENCODER (VAE) — REPARAMETERISATION:
    The VAE encoder samples from a latent distribution during training.
    At inference, the model may use the mean (MAP estimate) rather than
    a sample:

    def encode(self, x):
        mu, log_var = self.encoder(x)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z   = mu + eps * std          ← stochastic during training
        else:
            z = mu                        ← deterministic at inference
        return z, mu, log_var

    BATCH RENORMALIZATION (for variable batch sizes):
    The batch renormalization module clamps the batch statistics to
    be within a factor of the running statistics, with the clamping
    factor increasing over training:
    In train mode: uses clamped batch statistics + updates running stats.
    In eval mode:  uses running statistics only.

    SPECTRAL NORMALISATION (for GANs):
    The spectral norm constraint is computed and applied during the
    forward pass using a cached singular vector estimate.
    In train mode:  singular vector estimate is UPDATED.
    In eval mode:   singular vector estimate is FROZEN (the cached value
                    from training is used).


### The nn.Module.apply() Pattern for Mode Switching

When fine-tuning with mixed frozen/unfrozen layers, model.eval() and
model.train() set ALL submodules to the same mode. To set specific
submodules to different modes:

    # Fine-tuning: all layers in train mode EXCEPT BatchNorm
    model.train()                    ← set all to train
    model.apply(lambda m:            ← then override specific layers
        m.eval() if isinstance(m, nn.BatchNorm2d) else None)

    # Or: freeze a specific backbone while keeping the head in train mode
    model.eval()                     ← start from eval
    model.head.train()               ← override just the head to train

    nn.Module.apply(fn) calls fn on EVERY submodule recursively (self first,
    then children depth-first). It returns self, enabling chaining.

    CRITICAL: calling model.train() after model.apply() would reset
    all submodules back to train mode, undoing the selective eval().
    The apply() call must come AFTER model.train(), not before.


##### PART 6 — TRAINING LOOP STRUCTURE: PLACING THE MODE SWITCHES

### The Canonical Training Loop

    for epoch in range(max_epochs):

        # ── TRAINING PHASE ────────────────────────────────────────────
        model.train()          ← [1] switch to train mode ONCE per epoch
                                     not inside the batch loop
        train_loss = 0.0
        for batch in train_loader:
            inputs  = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            optimiser.zero_grad(set_to_none=True)

            outputs = model(inputs)                   ← forward pass
            loss    = criterion(outputs, targets)
            loss.backward()                           ← backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── VALIDATION PHASE ──────────────────────────────────────────
        model.eval()           ← [2] switch to eval mode ONCE per epoch
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():  ← [3] disable gradient computation
            for batch in val_loader:
                inputs  = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                val_loss    += loss.item() * len(targets)
                val_correct += (outputs.argmax(1) == targets).sum().item()

        val_loss    /= len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_accuracy:.4f}")

    Annotations for each numbered call:

    [1] model.train():
        Called ONCE at the start of the training phase, not on every batch.
        Calling it inside the batch loop is harmless but wasteful.
        It is especially important if the previous phase was validation
        (which called model.eval()).

    [2] model.eval():
        Called ONCE at the start of the validation phase.
        Switches all Dropout and BatchNorm layers to inference behaviour.
        MUST be called before the validation loop, not after.

    [3] torch.no_grad():
        Wraps the ENTIRE validation loop, not just individual operations.
        The context manager is re-entered for each batch automatically.
        Do NOT wrap the training loop in no_grad — gradients are needed.


### Common Structural Mistakes

    MISTAKE 1 — Calling model.eval() inside the batch loop:
    for batch in val_loader:
        model.eval()    ← wasteful but correct (sets mode redundantly)
        ...
    Effect: harmless but adds per-batch overhead.

    MISTAKE 2 — Forgetting model.train() after validation:
    evaluate()         ← calls model.eval() internally
    for batch in train_loader:
        loss = model(batch)   ← model is still in eval mode!
    Effect: Dropout is off during training (no regularisation).
    BatchNorm uses running stats during training (may hurt early in training).

    MISTAKE 3 — Using with torch.no_grad() in the training loop:
    for batch in train_loader:
        with torch.no_grad():    ← WRONG: no gradients computed
            outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()          ← fails: loss has no grad_fn
    Effect: RuntimeError: element 0 of tensors does not require grad
            and does not have a grad_fn.

    MISTAKE 4 — Not using torch.no_grad() in the validation loop:
    model.eval()
    for batch in val_loader:
        outputs = model(batch)   ← graph is built, activations stored
        loss    = criterion(...)
    Effect: correct numerically, but 2× memory usage (activations stored).
    On large models, may cause OOM during validation.

    MISTAKE 5 — Calling model.eval() once at the start and never
    switching back:
    model.eval()          ← called at the beginning of the script
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model(batch)   ← model is in eval mode!
    Effect: identical to Mistake 2.


##### PART 7 — INFERENCE IN PRODUCTION

### Preparing a Model for Deployment

The transition from a training checkpoint to a production inference
model involves several steps beyond just loading the weights:

    STEP 1 — Load the checkpoint:
    model = MyModel(config)
    state_dict = torch.load('best_model.pt', map_location='cpu')
    model.load_state_dict(state_dict['model_state_dict'])

    STEP 2 — Switch to eval mode:
    model.eval()
    # Verify:
    assert not model.training, "Model must be in eval mode for deployment"

    STEP 3 — Move to the inference device:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    STEP 4 — (Optional) Convert to half precision for speed:
    model = model.half()    ← FP16 inference (2× faster, 2× less memory)
    # Note: some layers (e.g., softmax, layer norm) may need FP32.
    # Use autocast for mixed precision inference instead:
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)

    STEP 5 — Wrap in inference function:
    @torch.inference_mode()     ← decorator equivalent to with-block
    def predict(model, inputs):
        inputs = inputs.to(device)
        return model(inputs)

    STEP 6 — Validate outputs:
    test_output = predict(model, test_input)
    assert torch.isfinite(test_output).all(), "NaN or Inf in model output"
    assert test_output.shape == expected_output_shape


### Serving Multiple Requests: Batching at Inference

In production, inference requests arrive one at a time (or in small
groups). Batching them together improves GPU utilisation:

    DYNAMIC BATCHING:
    Collect requests until a batch is full (max_batch=32) or a timeout
    is reached (max_wait_ms=10). Process the batch together.

    The model handles this without any special code — it sees a batch
    of B inputs exactly as during training. Ensure:
    1. model.eval() is set (it is, since this is deployment).
    2. torch.no_grad() or torch.inference_mode() wraps the forward call.
    3. Inputs are padded to the same length if variable-length sequences.
    4. Outputs are split back into per-request responses.

    SINGLE EXAMPLE INFERENCE (batch size 1):
    The input must still have a batch dimension:
    single_input = image_tensor.unsqueeze(0)  # shape (1, C, H, W)
    output = model(single_input)             # shape (1, num_classes)
    prediction = output.squeeze(0)           # shape (num_classes,)

    BatchNorm with batch size 1:
    BatchNorm in TRAIN mode with B=1 collapses (see Part 3).
    BatchNorm in EVAL mode with B=1 is safe — it uses running statistics.
    This is another reason model.eval() is essential for deployment.


### The @torch.inference_mode() Decorator

For functions that are always called at inference, the decorator
version is cleaner than wrapping every call in a context manager:

    # Context manager version (verbose):
    def predict(model, x):
        with torch.inference_mode():
            return model(x)

    # Decorator version (preferred for production code):
    @torch.inference_mode()
    def predict(model, x):
        return model(x)   # inference_mode is automatically active

    # Class method version:
    class InferenceEngine:
        def __init__(self, model):
            self.model = model
            self.model.eval()

        @torch.inference_mode()
        def __call__(self, x):
            return self.model(x)

    The decorator applies torch.inference_mode() to every call of the
    function, not just once. It cannot be bypassed from outside the function.


### Thread Safety and Stateful Components

PyTorch models are NOT thread-safe when stateful components are present:

    Thread-UNSAFE stateful components:
    - BatchNorm (updates running_mean and running_var during train mode)
    - RNNs with hidden state that persists across calls
    - Any module with mutable internal state

    For inference serving with multiple threads:
    Option 1: Use one model replica per thread.
    Option 2: Ensure all inference calls use model.eval() (no running stat
              updates in eval mode → BN becomes effectively stateless).
    Option 3: Use torch.jit.script() compiled models — TorchScript
              models have a different (more carefully managed) state model.

    Safe concurrent inference:
    All inference threads use model.eval() and torch.no_grad().
    The model's forward pass is pure (no mutation of model state) in eval mode
    for most architectures. BatchNorm in eval mode reads but does not write
    to running statistics → read-only → thread-safe.

    Unsafe concurrent training:
    Multiple threads calling loss.backward() on the same model simultaneously
    will corrupt gradients. Use DataParallel or DistributedDataParallel
    for multi-GPU training, not raw threads.


##### PART 8 — FINE-TUNING: MIXED MODE CONFIGURATIONS

### Fine-Tuning Patterns and Their Mode Requirements

Fine-tuning introduces the most complex mode configurations because
different parts of the model may be in different operational states:

    PATTERN 1 — Full fine-tuning (all layers train):
    model.train()           ← all layers in train mode
    optimiser = AdamW(model.parameters(), lr=2e-5)
    # Train as normal. No special mode handling needed.

    PATTERN 2 — Linear probing (frozen backbone, new head only):
    model.eval()            ← frozen backbone: eval mode (no BN updates,
                                no dropout) — matches inference behaviour
    model.head.train()      ← new head in train mode (has Dropout)
    for param in model.backbone.parameters():
        param.requires_grad = False    ← no gradient for backbone
    optimiser = AdamW(model.head.parameters(), lr=1e-3)

    WHY eval() for the backbone:
    If the backbone is in train mode with frozen weights:
    - BatchNorm STILL UPDATES running statistics (even though γ and β are
      frozen). Running stats may drift from pre-trained values.
    - Dropout still adds stochasticity (though this may be desired for
      regularisation of the intermediate representations).

    Generally: frozen backbone → eval mode (preserves pre-trained BN stats).
    Exception: if the fine-tuning dataset is large and similar in distribution,
    letting BN adapt (train mode) may be beneficial.

    PATTERN 3 — Gradual unfreezing (unfreeze layers progressively):
    # Epoch 1: only head is in train mode
    model.eval()
    model.head.train()

    # Epoch 2: unfreeze last transformer block
    model.layers[-1].train()
    for param in model.layers[-1].parameters():
        param.requires_grad = True

    # Epoch 3: unfreeze more layers...

    This is the ULMFiT approach. Each newly unfrozen block switches to
    train mode at the epoch it is activated.

    PATTERN 4 — LoRA / adapter fine-tuning:
    # Frozen base model; only LoRA adapters are trainable
    model.eval()                          ← base model in eval mode
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.train()                ← adapters in train mode
    # Only LoRA parameters have requires_grad=True
    optimiser = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)

    LoRA layers contain no BatchNorm, so the train/eval distinction is
    mainly for dropout (if any is added to LoRA layers).


### A Fine-Tuning Mode Verification Utility

    def audit_model_modes(model, verbose=True):
        '''
        Print a summary of which modules are in train vs eval mode,
        and which parameters have requires_grad=True.
        Useful for verifying fine-tuning configurations.
        '''
        mode_summary    = {'train': 0, 'eval': 0}
        grad_summary    = {'requires_grad': 0, 'frozen': 0}
        mode_sensitive  = []

        for name, module in model.named_modules():
            mode = 'train' if module.training else 'eval'
            mode_summary[mode] += 1

            # Flag mode-sensitive layers explicitly
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d,
                                    nn.BatchNorm2d, nn.BatchNorm3d,
                                    nn.SyncBatchNorm)):
                mode_sensitive.append((name, type(module).__name__, mode))

        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_summary['requires_grad'] += 1
            else:
                grad_summary['frozen'] += 1

        if verbose:
            print(f"Module modes:   train={mode_summary['train']}, "
                  f"eval={mode_summary['eval']}")
            print(f"Parameters:     trainable={grad_summary['requires_grad']}, "
                  f"frozen={grad_summary['frozen']}")
            if mode_sensitive:
                print(f"Mode-sensitive layers:")
                for name, cls, mode in mode_sensitive:
                    print(f"  {name:40s} ({cls:20s}) → {mode}")

        return mode_summary, grad_summary, mode_sensitive


##### PART 9 — COMPLETE REFERENCE AND DECISION GUIDE

### Mode Configuration Lookup Table

    ┌───────────────────────────────────────────────────────────────────────┐
    │ Situation                    │ model.   │ torch.      │ requires_grad │
    │                              │ mode     │ context     │               │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Standard training step       │ .train() │ (none)      │ True          │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Validation / evaluation      │ .eval()  │ no_grad()   │ True (ok)     │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Production inference         │ .eval()  │ inference_  │ False (opt)   │
    │                              │          │ mode()      │               │
    ├───────────────────────────────────────────────────────────────────────┤
    │ MC Dropout uncertainty       │ .train() │ no_grad()   │ False (opt)   │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Test-time augmentation (TTA) │ .eval()  │ no_grad()   │ False (opt)   │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Linear probing (frozen body) │ body:    │ (none)      │ body: False   │
    │                              │ .eval()  │             │ head: True    │
    │                              │ head:    │             │               │
    │                              │ .train() │             │               │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Full fine-tuning             │ .train() │ (none)      │ True          │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Gradient penalty (GAN)       │ .train() │ (none)      │ True          │
    │ (needs grad w.r.t. input)    │          │             │               │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Feature extraction (no train)│ .eval()  │ inference_  │ False         │
    │                              │          │ mode()      │               │
    └───────────────────────────────────────────────────────────────────────┘


### Quick Reference: What Each Layer Does in Each Mode

    ┌────────────────────────────────────────────────────────────────────┐
    │ Layer            │ TRAIN mode behaviour     │ EVAL mode behaviour  │
    ├────────────────────────────────────────────────────────────────────┤
    │ nn.Dropout       │ Random zeroing + scale   │ Identity (pass-thru) │
    │ nn.Dropout2d     │ Random channel drop      │ Identity             │
    │ nn.AlphaDropout  │ Random zeroing (SELU)    │ Identity             │
    ├────────────────────────────────────────────────────────────────────┤
    │ nn.BatchNorm1d/  │ Uses batch mean/var.     │ Uses running         │
    │ 2d/3d            │ Updates running stats.   │ mean/var. No update. │
    │ nn.SyncBatchNorm │ Uses cross-GPU batch     │ Uses running stats.  │
    │                  │ mean/var.                │                      │
    ├────────────────────────────────────────────────────────────────────┤
    │ nn.LayerNorm     │ Instance stats           │ Instance stats       │
    │ nn.GroupNorm     │ Group stats              │ Group stats          │
    │ nn.InstanceNorm  │ Instance stats           │ Instance stats       │
    │ (no change)      │                          │                      │
    ├────────────────────────────────────────────────────────────────────┤
    │ nn.Linear        │ Identical                │ Identical            │
    │ nn.Conv2d        │ Identical                │ Identical            │
    │ nn.Embedding     │ Identical                │ Identical            │
    │ nn.MultiheadAttn │ attn dropout active      │ attn dropout off     │
    │ (via Dropout)    │                          │                      │
    └────────────────────────────────────────────────────────────────────┘


### The Mode Switch Checklist

    BEFORE TRAINING:
    □ model.train() is called before the training loop starts.
    □ No torch.no_grad() wraps any part of the training loop.
    □ All parameters that should be updated have requires_grad=True.
    □ If fine-tuning: frozen layers are in eval mode, trainable layers
      in train mode. Verified with audit_model_modes().

    BEFORE VALIDATION / EVALUATION:
    □ model.eval() is called before the validation loop.
    □ torch.no_grad() (or torch.inference_mode()) wraps the validation loop.
    □ Loss is computed correctly (mean reduction, correct output/target types).

    BEFORE RETURNING TO TRAINING:
    □ model.train() is called again after the validation phase.
    □ If fine-tuning with mixed modes: re-apply the selective mode
      configuration after model.train() (since train() overrides all modes).

    BEFORE DEPLOYMENT:
    □ model.eval() is set. Verify: assert not model.training.
    □ torch.inference_mode() wraps all inference calls.
    □ Output of model(test_input) is verified to be finite and correctly shaped.
    □ Inference output matches the validation metric from the checkpoint.
    □ Thread safety: confirmed that eval mode makes forward pass read-only.

    DEBUGGING STOCHASTIC INFERENCE:
    □ Check model.training — it should be False.
    □ Check all submodules: any submodule with training=True?
      for name, m in model.named_modules():
          if m.training: print(f"Still in train mode: {name}")
    □ Check for any custom layer that samples randomly without respecting
      self.training.
    □ Verify no MC Dropout is being used unintentionally.

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
        from Training_Core.visuals.Train_vs_inference_visual import (
            TRAIN_VS_INFERENCE_VISUAL_HTML,
            TRAIN_VS_INFERENCE_VISUAL_HEIGHT,
        )
        visual_html   = TRAIN_VS_INFERENCE_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = TRAIN_VS_INFERENCE_VISUAL_HEIGHT
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