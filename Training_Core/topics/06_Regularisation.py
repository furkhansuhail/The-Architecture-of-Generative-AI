"""
Regularisation
==============

Regularisation is the collection of techniques that close the gap between
training performance and generalisation performance. A model that achieves
zero training loss but fails on unseen data has learned to memorise rather
than to understand. Regularisation is the set of constraints, penalties,
and structural choices that force the model to learn compact, transferable
representations instead.

"""
import textwrap
import re

TOPIC_NAME   = "Regularisation"
DISPLAY_NAME = "06 · Regularisation"
ICON         = "🛡️"
SUBTITLE     = "L1 · L2 · Dropout · DropPath · Early Stopping · Bias-Variance"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — THE OVERFITTING PROBLEM & BIAS-VARIANCE TRADEOFF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### What Is Overfitting?

A model overfits when it learns the training set — including its noise,
labelling errors, and sample-specific idiosyncrasies — so thoroughly that
it fails to generalise to new examples drawn from the same distribution.

The diagnostic signature:

    Training loss  → very low (approaches 0)
    Validation loss → high (and often rising while training loss falls)

    Diagram 1 — Overfitting in Loss Curves:

    Loss
     │
     │  ┌─ training loss
     │  │
    ▓▓ │                          validation loss ─┐
    ▓▓ │                         ╱                 │
    ▓▓ │                        ╱                  │
    ▓▓ │  ╲                    ╱                   │
    ▓▓ │   ╲                  ╱                    │
    ▓▓ │    ╲                ╱    ← divergence      │
    ▓▓ │     ╲______________╱       = overfitting   │
    ▓▓ │                                            │
     └──────────────────────────────────────── Epoch

    The gap between training and validation loss is the GENERALISATION GAP.
    Regularisation techniques aim to reduce or eliminate this gap.


### Why Neural Networks Overfit

Neural networks are Universal Function Approximators — given enough
capacity, they can memorise any finite dataset exactly. A network with
more parameters than training examples has the capacity to assign a
unique output to every training point without learning any generalising
rule.

    Concrete example: N=1000 training samples, model has 10M parameters.
    The model has 10,000× more degrees of freedom than constraints.
    Unconstrained gradient descent will find a solution that perfectly
    fits the training data, but that solution is one of infinitely many
    possible solutions — most of which generalise poorly.

    This is the fundamental tension in supervised learning:
    Expressiveness (capacity to fit complex patterns) vs
    Simplicity    (capacity to generalise to unseen data)

    Regularisation resolves this tension by CONSTRAINING the search space
    of solutions — biasing gradient descent toward simpler solutions that
    are more likely to generalise.


### The Bias-Variance Decomposition

For a regression model, the expected test error can be decomposed
exactly into three terms:

    E[(y - ŷ)²] = Bias²  +  Variance  +  Irreducible Noise

    Bias²:      Error from the model's systematic wrong assumptions.
                A linear model fit to a cubic function has high bias.
                It cannot represent the true function regardless of
                how much data you show it.

    Variance:   Error from the model's sensitivity to the training set.
                A degree-15 polynomial fit to 20 points has high variance —
                small changes in the training set produce wildly different
                fitted curves.

    Irreducible Noise:
                The inherent randomness in the target (measurement error,
                label noise). No model can reduce this.

    Diagram 2 — The Bias-Variance Tradeoff:

    Error
      │
      │╲                                    Total error (U-shaped)
      │ ╲                              ╱
      │  ╲                            ╱
      │   ╲              ┌───────────╯   ← variance
      │    ╲            ╱
      │     ╲__________╱                 ← bias²
      │
      └──────────────────────────────────────── Model complexity
                ↑
          sweet spot = optimal regularisation strength

    High regularisation → high bias, low variance   (underfitting)
    Low regularisation  → low bias, high variance   (overfitting)
    The goal: tune regularisation to land at the bottom of the U-curve.

    IMPORTANT NUANCE: Deep neural networks have a more complex relationship
    with this tradeoff than classical models. The "double descent" phenomenon
    (Belkin et al., 2019) shows that for sufficiently large models, test
    error can DECREASE again after the interpolation threshold — even without
    explicit regularisation. Modern large models operate in this
    "over-parameterised" regime where classical bias-variance intuitions
    break down. Nonetheless, regularisation still consistently improves
    both the stability and the generalisation of practical models.


### Why Regularisation Works: The PAC-Learning Perspective

From statistical learning theory (PAC learning), the generalisation error
is bounded by:

    Test error ≤ Training error  +  O(sqrt(C / n))

    Where C is the complexity of the hypothesis class (e.g. VC dimension
    or Rademacher complexity), and n is the number of training samples.

    Regularisation REDUCES C — it restricts the hypothesis class that
    gradient descent is searching over. A model with L2 regularisation
    cannot use large weights, effectively restricting itself to a smaller
    class of functions. The tighter the hypothesis class, the tighter
    the generalisation bound, and the less data you need to generalise.

    Practical takeaway: regularisation is especially critical when n is small.
    With millions of training examples, a large network may generalise
    without explicit regularisation. With thousands, it almost certainly
    needs it.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — L2 REGULARISATION (WEIGHT DECAY / RIDGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Idea

L2 regularisation adds a penalty proportional to the SQUARED MAGNITUDE of
every weight to the loss function:

    L_total = L_task  +  (λ/2) · Σᵢ wᵢ²

    Where:
        L_task = original task loss (cross-entropy, MSE, etc.)
        λ      = regularisation strength (hyperparameter)
        Σᵢ wᵢ² = sum of squared weights (the L2 norm squared)

    The (λ/2) factor is a convention that simplifies the gradient.

The penalty discourages large weights. A model with weight magnitudes
of [10, -8, 7] pays a much larger penalty than one with [0.1, -0.08, 0.07]
for the same task loss reduction.


### Gradient Update with L2: Weight Decay

The gradient of the total loss:

    ∂L_total/∂wᵢ = ∂L_task/∂wᵢ  +  λ · wᵢ

    The SGD update rule becomes:

    wᵢ ← wᵢ - η · (∂L_task/∂wᵢ + λ · wᵢ)
           = wᵢ · (1 - η·λ)  -  η · ∂L_task/∂wᵢ

    The term (1 - η·λ) is the WEIGHT DECAY factor. Every step, each weight
    is shrunk toward zero by a multiplicative factor before the gradient
    update. This is why L2 regularisation is often called WEIGHT DECAY —
    the weights literally decay toward zero at each update.

    For typical values η=0.001, λ=0.01:
    Weight decay factor = 1 - (0.001 × 0.01) = 1 - 0.00001 = 0.99999
    Each weight shrinks by 0.001% per step — tiny per step, significant
    over thousands of steps.


### L2 Regularisation as a MAP Estimate

From a Bayesian perspective, L2 regularisation is equivalent to placing
a GAUSSIAN PRIOR on the weights:

    Prior: p(w) = N(0, σ²_prior)     [weights are expected to be near 0]
    Likelihood: p(D|w)               [data fit, the standard loss]

    Maximum A Posteriori (MAP) estimate:
    log p(w|D) = log p(D|w) + log p(w)
               = -L_task  -  (1/2σ²_prior) · Σᵢ wᵢ²

    Comparing: L_total = L_task + (λ/2) · Σᵢ wᵢ²
    → λ = 1/σ²_prior

    A SMALLER σ²_prior (stronger belief that weights should be near 0)
    corresponds to a LARGER λ (stronger regularisation).

    This interpretation is powerful: you are encoding prior knowledge
    that the true model is simple (weights close to zero). If you have
    domain knowledge that the model should be complex, you can reduce λ.


### L2 Effect on Weights: Shrinkage, Not Sparsity

L2 regularisation SHRINKS weights proportionally — it does not zero them out.

    Optimal weight with L2 (closed-form for linear regression):
    w_L2 = (X^T X + λI)^{-1} X^T y

    The λI term "inflates" the diagonal of X^T X, ensuring invertibility
    (solving the ill-conditioning problem when features are correlated).

    Effect on each weight:
    w_L2_i = w_OLS_i · (σᵢ² / (σᵢ² + λ))

    Where σᵢ² is the variance explained by feature i.
    High-variance features (informative): shrinkage is small.
    Low-variance features (noisy):        shrinkage is large.

    KEY PROPERTY: L2 distributes weight across correlated features.
    If two features are highly correlated, L2 splits weight between them.
    L1 (next section) tends to pick one and zero the other.

    Diagram 3 — L2 Geometry (Weight Shrinkage):

    w₂
     │      ╭─────────────╮
     │   ╭──┤ L2 constraint├──╮
     │  ╭┤  │   sphere    │  ╰╮
     │  ││  ╰─────────────╯  ││
     │  │╰────────────────────╯│
    ─┼──╯─────────────────────╰──── w₁
     │          ●  ← OLS solution
     │        ●     ← L2 solution (pulled toward origin, not zeroed)
     │

    The L2 constraint is a SPHERE (circle in 2D). The unconstrained
    OLS solution is projected onto the boundary of this sphere —
    pulled toward the origin but rarely landing exactly on an axis,
    so weights are rarely exactly zero.


### L2 vs AdamW: A Critical Distinction

Standard L2 regularisation adds λ·wᵢ to the gradient, which then feeds
into the adaptive gradient computation. With Adam, this interacts with
the per-parameter learning rate adaptation:

    Standard Adam + L2:
    m̂ᵢ = corrected gradient estimate (includes λ·wᵢ term)
    v̂ᵢ = variance of gradient (includes λ·wᵢ influence)
    wᵢ ← wᵢ - η · m̂ᵢ / (sqrt(v̂ᵢ) + ε)

    The weight decay signal is ADAPTED by the gradient statistics.
    Features with large gradients get a smaller effective decay.
    This breaks the clean "shrink toward zero" interpretation.

AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient:

    AdamW:
    m̂ᵢ, v̂ᵢ computed from task gradient ONLY (no λ·wᵢ)
    wᵢ ← wᵢ · (1 - η·λ)  -  η · m̂ᵢ / (sqrt(v̂ᵢ) + ε)

    Weight decay is applied DIRECTLY to the weight, before the gradient step.
    Every weight decays at the same rate regardless of its gradient variance.

    Empirical result: AdamW consistently outperforms Adam + L2 for
    training transformers and other large models.
    Rule: Use AdamW when using Adam. Apply λ directly to the weight, not the gradient.

    Typical λ values for AdamW:
    Language models (GPT-2, BERT): 0.01
    Image models (ViT):            0.01 to 0.1
    CNNs:                          1e-4 to 1e-3
    Default starting point:        0.01


### What NOT to Regularise with Weight Decay

Not all parameters should be penalised:

    DO regularise:   weight matrices (Linear, Conv2d)
    DO NOT regularise:
        - Bias terms (they do not cause overfitting in the same way)
        - BatchNorm scale (γ) and shift (β) parameters
        - LayerNorm parameters
        - Embedding tables (unless explicitly intended)

    Regularising normalisation parameters can destabilise training because
    γ=0 collapses the layer's output scale, disrupting gradient flow.

    In PyTorch:
        decay_params = [p for n, p in model.named_parameters()
                        if 'bias' not in n and 'norm' not in n]
        no_decay_params = [p for n, p in model.named_parameters()
                           if 'bias' in n or 'norm' in n]
        optimiser = AdamW([
            {'params': decay_params,    'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=3e-4)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — L1 REGULARISATION (LASSO / SPARSITY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Idea

L1 regularisation penalises the SUM OF ABSOLUTE VALUES of weights:

    L_total = L_task  +  λ · Σᵢ |wᵢ|

    The absolute value, unlike the square, gives a CONSTANT gradient
    magnitude regardless of weight size:

    ∂(|wᵢ|)/∂wᵢ = sign(wᵢ)  =  +1 if wᵢ > 0
                                 -1 if wᵢ < 0
                                  0 if wᵢ = 0

    The gradient update:
    wᵢ ← wᵢ - η · ∂L_task/∂wᵢ  -  η · λ · sign(wᵢ)

    At every step, a CONSTANT amount η·λ is subtracted from |wᵢ|.
    If |wᵢ| < η·λ, the next step would overshoot zero — in practice,
    the weight is clipped to exactly 0 (the proximal gradient method).

    This constant pull is what creates SPARSITY: small weights get
    driven all the way to zero, large weights get shrunk linearly.


### Why L1 Produces Sparse Solutions

    Diagram 4 — L1 vs L2 Geometry (why L1 gives sparse solutions):

    w₂                              w₂
     │     ●  OLS solution           │     ●  OLS solution
     │    ╱                          │    ╱
     │   ╱                           │   ╱
     │  ╱  L1 diamond                │  ╱   L2 sphere
     │ ╱ ╱╲                          │ ╱  ╭───╮
    ─┼─╱──╱╲──── w₁                 ─┼─╱──╭──╮─── w₁
     │ ╲  ╱  │                       │    │  │
     │  ╲╱   │                       │    ╰──╯
     │   ↑                           │     ↑
     │ corner at (0, w₂*)            │  Smooth boundary
     │ = sparse solution             │  = non-sparse solution

    The L1 feasible set is a DIAMOND (L1 ball) with CORNERS on the axes.
    When you project the unconstrained solution onto the L1 ball, the
    closest point is very likely to be a CORNER — where one or more
    weights are exactly zero.

    The L2 feasible set is a SPHERE with no corners. The projection
    almost never lands exactly on an axis, so weights are almost never zero.

    This geometric insight generalises to high dimensions:
    L1 ball in p dimensions has 2p corners on the coordinate axes.
    As p increases, the probability that the optimal solution lies at
    a corner — and is thus sparse — also increases.


### L1 as a Bayesian Laplace Prior

Analogous to the Gaussian prior for L2, L1 corresponds to a LAPLACE prior:

    Prior: p(wᵢ) = (λ/2) · exp(-λ|wᵢ|)     [Laplace / double exponential]

    The Laplace prior has a SHARP PEAK at zero and HEAVY TAILS compared
    to the Gaussian. This means it:
    1. Strongly encourages weights to be near zero (sharp peak)
    2. But allows occasional large weights without as heavy a penalty
       as the Gaussian (heavy tails)

    Diagram 5 — Gaussian (L2) vs Laplace (L1) Priors:

    Density
      │    L1 (Laplace)        L2 (Gaussian)
      │       │                   ╭───╮
      │       │                 ╭─╯   ╰─╮
      ▓▓      │                ╭╯       ╰╮
      ▓▓     ╱ ╲              ╭╯         ╰╮
      ▓▓    ╱   ╲            ─╯           ╰─
      └──────────────────────────────────── w
              ↑↑↑
         Sharp peak at 0
         → strongly promotes sparsity


### When to Use L1 in Neural Networks

L1 is less commonly used in deep learning than L2 for several reasons:

    1. Non-differentiability at w=0: the gradient is undefined at exactly
       zero. In practice, the subgradient (sign function) is used, but
       this can cause training instability near zero.

    2. Deep networks rarely benefit from strict sparsity: unlike linear
       models where feature selection is meaningful, neural networks
       learn distributed representations where no single weight is
       interpretable in isolation.

    3. Practical sparsity tools exist at a higher level: structured pruning
       (removing entire neurons, channels, or heads) is more hardware-
       efficient than unstructured weight-level sparsity from L1.

    USE L1 regularisation when:
    ✓ Training linear or shallow models where feature selection matters
    ✓ Building explicit sparse representations (dictionary learning)
    ✓ The output should be interpretable and few-feature
    ✓ Combined with L2 (Elastic Net — see Part 4)

    PREFER L2 (weight decay) for:
    ✓ Deep neural networks (the default, always-on regulariser)
    ✓ Transformers and large models (use AdamW with weight decay)
    ✓ When training stability is paramount


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — ELASTIC NET (L1 + L2 COMBINED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Combined Penalty

Elastic Net (Zou & Hastie, 2005) combines both penalties:

    L_total = L_task  +  λ₁ · Σᵢ |wᵢ|  +  (λ₂/2) · Σᵢ wᵢ²

    This is parameterised as:
    L_total = L_task  +  λ · [α · Σᵢ|wᵢ|  +  (1-α)/2 · Σᵢwᵢ²]

    Where α ∈ [0,1] controls the L1/L2 mix:
        α = 1: pure L1 (Lasso)
        α = 0: pure L2 (Ridge)
        α = 0.5: equal mix

### Properties and When It Is Useful

    L1 ALONE has a problem with correlated features:
    If two features are highly correlated, Lasso tends to arbitrarily
    select one and discard the other (random which survives).

    L2 ALONE does not produce sparsity:
    All features survive; only small weights are obtained.

    ELASTIC NET solves both:
    - The L2 term handles correlated features (distributes weight, stable)
    - The L1 term still drives some weights to exactly zero (sparse)
    - Grouping effect: correlated features tend to be selected or dropped
      together, rather than arbitrarily

    Useful in neural networks for:
    - Regularising embedding layers (sparse + bounded embeddings)
    - Convolutional filter regularisation in small-data regimes
    - Any setting where you want both bounded magnitudes AND sparsity


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 5 — DROPOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Mechanism

Dropout (Srivastava et al., 2014) is the most widely used regularisation
technique in deep learning. During each forward pass in training, each
neuron's activation is independently set to zero with probability p
(the DROP probability, also called the dropout rate):

    For each neuron i in each forward pass:
    if Bernoulli(p) = 1:  output = 0        (dropped)
    else:                  output = aᵢ       (kept)

    Equivalently, a mask mᵢ ~ Bernoulli(1-p):
    output_i = mᵢ · aᵢ

    The mask is sampled INDEPENDENTLY for each neuron and each sample
    in the batch. Two different samples in the same batch get different masks.


### Inverted Dropout: The Rescaling Fix

Naive dropout has a critical problem: the expected output of a neuron
during training is (1-p) × aᵢ (only (1-p) fraction of the time is it
non-zero), but during INFERENCE all neurons are active, producing
expected output 1.0 × aᵢ.

This mismatch means the downstream layers see inputs at different scales
during training vs inference. Inverted dropout fixes this:

    During TRAINING:
    output_i = mᵢ · aᵢ / (1 - p)

    The surviving outputs are SCALED UP by 1/(1-p) to compensate
    for the dropped units. Expected value = (1-p) · aᵢ/(1-p) = aᵢ.
    Training and test expectations MATCH.

    During INFERENCE:
    output_i = aᵢ    (no masking, no scaling)

    Diagram 6 — Inverted Dropout: Training vs Inference:

    TRAINING (p=0.5, scale=2.0):          INFERENCE:
    ┌────────────┐                         ┌────────────┐
    │ a₁ = 0.4  │ → mask=0 → 0.0          │ a₁ = 0.4  │ → 0.4
    │ a₂ = 0.7  │ → mask=1 → 0.7×2 = 1.4 │ a₂ = 0.7  │ → 0.7
    │ a₃ = 0.3  │ → mask=1 → 0.3×2 = 0.6 │ a₃ = 0.3  │ → 0.3
    │ a₄ = 0.9  │ → mask=0 → 0.0          │ a₄ = 0.9  │ → 0.9
    │ a₅ = 0.5  │ → mask=1 → 0.5×2 = 1.0 │ a₅ = 0.5  │ → 0.5
    └────────────┘                         └────────────┘
    Expected sum = 0+1.4+0.6+0+1.0 = 3.0  Expected sum = 0.4+0.7+0.3+0.9+0.5 = 2.8
    ≈ same scale ✓

    Note: PyTorch nn.Dropout uses inverted dropout by default.
    Setting model.eval() is essential — it switches off the mask.
    Forgetting model.eval() is one of the most common bugs in practice.


### Why Dropout Regularises: Three Interpretations

    INTERPRETATION 1 — Ensemble of exponentially many models:
    A network with n dropout-eligible units can produce 2^n different
    sub-networks (each unit either dropped or not). Dropout trains
    an ensemble of all these sub-networks SIMULTANEOUSLY, with shared
    weights. At inference, using all neurons is an approximation to
    averaging the predictions of the entire ensemble.
    This is analogous to random forests — a tree ensemble often
    generalises better than any single tree.

    INTERPRETATION 2 — Prevents co-adaptation:
    Without dropout, neurons can develop co-dependencies — neuron A
    learns to fix the mistake of neuron B. These co-adaptations are
    specific to the training set and do not generalise.
    Dropout randomly removes neurons, so each neuron cannot rely on
    any other specific neuron being present. Every neuron must learn
    a feature that is useful on its own, independent of its
    co-neurons. The resulting features are more ROBUST and REDUNDANT.

    INTERPRETATION 3 — Adds multiplicative noise to features:
    Dropout multiplies each feature by Bernoulli noise (0 or 1/(1-p)).
    This is a form of DATA AUGMENTATION in feature space — the model
    learns to be invariant to the presence or absence of individual
    features, which is a useful form of robustness.

    Research by Wager et al. (2013) formalised this: dropout applied
    to linear models is equivalent to an adaptive L2 regularisation
    where the regularisation strength for each feature is proportional
    to its input variance.


### Where to Place Dropout

    After fully-connected layers:  ✓  Standard and highly effective
    After convolutional layers:    ⚠   Less common; use lower p (0.1-0.2)
                                        or use SpatialDropout instead
    After the embedding layer:     ✓  Common in NLP models
    Inside attention (attention    ✓  Attention dropout — standard in
    dropout in transformers):           transformer models
    After BatchNorm / LayerNorm:   ⚠   Controversial — some evidence that
                                        dropout + BN can interact negatively
    After the final output layer:  ✗   Never — dropout on logits prevents
                                        the model from making predictions

    For CNNs specifically — SpatialDropout2d:
    Standard dropout drops individual pixel activations independently.
    For 2D feature maps, it is better to drop ENTIRE CHANNELS (feature maps)
    because adjacent pixels in a channel are highly correlated —
    standard dropout is not very effective (dropped pixels can be inferred
    from neighbours). SpatialDropout2d drops entire channels at once.


### Typical Dropout Rates by Architecture

    Fully connected layers (MLP):  p = 0.3 to 0.5
                                   p = 0.5 is the original recommendation
    Convolutional layers:          p = 0.1 to 0.25
    Transformer (residual stream): p = 0.1 (GPT-2, BERT)
    Transformer attention:         p = 0.1 (attention dropout)
    After embedding layers (NLP):  p = 0.1 to 0.3
    Vision transformer (ViT):      p = 0.0 (ViT-Base trained on ImageNet
                                   without dropout; modern practice uses
                                   DropPath instead — see Part 6)


### Dropout at Test Time: Monte Carlo Dropout

Standard use: disable dropout at inference (model.eval()).
But dropout can be KEPT ENABLED at inference for uncertainty estimation:

    MC Dropout (Gal & Ghahramani, 2016):
    Run the same input through the model T times (each with random masks).
    The variance across T predictions estimates EPISTEMIC UNCERTAINTY
    (uncertainty about the model parameters).

    For T=100 forward passes:
    mean(predictions)     → final prediction
    var(predictions)      → model uncertainty

    This gives a Bayesian approximation at almost zero extra cost —
    the only overhead is T forward passes.

    Used in:
    - Medical imaging (when model must report confidence)
    - Active learning (select examples where the model is most uncertain)
    - Autonomous systems (detect out-of-distribution inputs)


### DropConnect: A Generalisation of Dropout

Dropout zeroes out ACTIVATIONS. DropConnect (Wan et al., 2013) zeroes
out WEIGHTS instead:

    Standard dropout: mask applied to outputs of a layer
    DropConnect:      mask applied to the weight matrix W
                      → each weight wᵢⱼ is independently set to 0
                         with probability p during the forward pass

    DropConnect is a strict generalisation of Dropout (Dropout is
    equivalent to DropConnect on a specific layer structure).
    In practice, DropConnect is rarely used — the implementation overhead
    is high relative to the marginal gain over standard Dropout.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 6 — DROPPATH / STOCHASTIC DEPTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Motivation: Why Standard Dropout Fails in Residual Networks

In residual networks (ResNets, Transformers), each layer adds a residual:

    output = x + F(x)     [F is the layer's transformation]

    Standard dropout applied inside F(x) zeroes out individual activations.
    But x (the residual path) always passes through — so even with heavy
    dropout on F(x), the network still has full depth at every step.
    The network never learns to "work without" specific layers.

DropPath (Stochastic Depth, Huang et al., 2016) drops ENTIRE LAYERS
(residual branches) rather than individual neurons:

    Standard residual:   output = x + F(x)

    DropPath:            b ~ Bernoulli(1 - p_drop)
                         output = x + b · F(x)

    When b=0: the entire layer is skipped. The input passes through
              unchanged (only the skip connection remains).
    When b=1: the layer operates normally.

    Diagram 7 — DropPath vs Standard Dropout in a Residual Block:

    Standard Dropout:                  DropPath:
    ┌─────────────────────┐            ┌─────────────────────────┐
    │    x (input)        │            │    x (input)            │
    │    │        │       │            │    │           │         │
    │    ▼        │       │            │    ▼           │         │
    │  F(x)       │skip   │            │  F(x)          │skip     │
    │  with       │       │            │ ×b (b=0: skip  │         │
    │  neuron     │       │            │  entire block) │         │
    │  dropout    │       │            │    │           │         │
    │    │        │       │            │    │           │         │
    │    └──┬─────┘       │            │    └────┬──────┘         │
    │       ▼             │            │         ▼                │
    │  x + drop(F(x))     │            │   x + b·F(x)            │
    └─────────────────────┘            └─────────────────────────┘
    (F is always computed,             (b=0: F is never computed,
     some neurons zeroed)               entire branch skipped)


### Linear Drop Rate Schedule

For a network with L layers, DropPath is applied with a LINEARLY INCREASING
drop probability from the first to the last layer:

    Layer l drop probability:  p_l = (l / L) · p_max

    Layer 1 (closest to input):  p₁ ≈ 0 (rarely dropped)
    Layer L (deepest):           p_L = p_max

    Rationale: early layers compute basic features (edges, textures)
    that every example needs. Dropping them would be too destructive.
    Deeper layers compute task-specific features that can be skipped
    more safely — the network learns to work without them, which both
    regularises and reduces effective depth.

    Typical p_max values:
    ResNet-50:       0.2
    DeiT-S (ViT):    0.1
    DeiT-B (ViT):    0.1
    Swin-Tiny:       0.2
    Swin-Base:       0.3 to 0.5


### Stochastic Depth as an Implicit Ensemble

At inference, all layers are active (b=1 always, no rescaling needed
beyond the standard 1/(1-p) scaling applied per-layer).

During training, a network with L layers and stochastic depth implicitly
trains an ensemble of networks ranging from depth 1 to depth L.
Each forward pass uses a randomly selected subset of layers.

    Expected depth during training:
    E[depth] = Σ_l (1 - p_l) = L - Σ_l p_l  ≈  L · (1 - p_max/2)

    A 24-layer network with p_max=0.5:
    Expected depth ≈ 24 × (1 - 0.25) = 18 layers per forward pass.

    This also means training is FASTER — skipped layers require no
    computation. Stochastic depth gives both a regularisation benefit
    AND a training speed benefit (roughly proportional to 1 - p_max/2).


### DropPath Implementation Detail

    The drop decision is made PER SAMPLE, not per batch:

    # Correct implementation:
    def drop_path(x, drop_prob, training):
        if not training or drop_prob == 0.0:
            return x
        keep_prob = 1 - drop_prob
        # Shape: (batch_size, 1, 1, ...) — broadcast over spatial dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob   # rescale to maintain expected value

    The mask has batch dimension but all spatial dimensions = 1,
    so it drops the entire residual for selected samples in the batch
    while keeping it for others. This is the "per-sample" property —
    different samples in the same batch experience different effective depths.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 7 — EARLY STOPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Core Idea

Early stopping halts training when a monitored metric (typically validation
loss) stops improving, and restores the model weights from the best
observed checkpoint.

    It is the simplest possible regulariser: reduce model complexity by
    training less. Gradient descent finds progressively more complex
    functions as training continues — early stopping limits this.

    Diagram 8 — Early Stopping Decision:

    Loss
     │
     │              validation loss
     │             ╱──────────╲ ← early stop here (best val loss)
     │            ╱            ╲__________
     │           ╱                         ╲____
     │          ╱
     │         ╱ training loss
     │    ____╱____________________________ → keeps decreasing
     │
     └──────────────────────────────────────── Epoch
                         ↑
                    PATIENCE window
              (monitor N epochs, stop if
               no improvement within N)

    Without early stopping: training continues until the training loss
    stops decreasing or a fixed epoch budget is exhausted.
    With early stopping: training stops when val loss begins consistently
    increasing, indicating that further training is overfitting.


### Patience: The Key Hyperparameter

Early stopping requires a PATIENCE parameter — the number of epochs to
wait after the last improvement before stopping:

    patience = 1:   Stop at the first epoch with no improvement.
                    Too aggressive — stochastic training has noisy val loss.
    patience = 10:  Wait 10 consecutive non-improving epochs before stopping.
                    Standard for most tasks.
    patience = 50:  Wait 50 epochs. Use for tasks where val loss
                    fluctuates significantly (e.g., small datasets).

    The patience must be long enough to outlast the natural noise in the
    validation metric. If the validation loss typically fluctuates by
    ±0.5% epoch-to-epoch, a patience of 5 would stop training due to noise
    rather than genuine overfitting.


### Early Stopping as L2 Regularisation (Exact Equivalence)

For linear models trained with gradient descent, early stopping is
mathematically EQUIVALENT to L2 regularisation:

    Proof sketch (Bishop, 1995):
    After t steps of gradient descent with learning rate η:
    wᵢ(t) = Σₖ (1 - (1 - η·λₖ)^t) · (uₖ^T y) · uₖ

    Where λₖ are the eigenvalues of X^T X and uₖ are the eigenvectors.

    As t → ∞: wᵢ(t) → OLS solution (no regularisation)
    As t → 0:  wᵢ(t) → 0 (maximum regularisation)

    This is equivalent to L2 regularisation with λ = 1/(η·t):
    Stopping early (small t) ↔ Strong L2 regularisation (large λ)
    Training long  (large t) ↔ Weak L2 regularisation   (small λ)

    For non-linear deep networks, the equivalence is not exact but
    the intuition holds: early stopping implicitly constrains the model
    to a neighbourhood of its random initialisation, which tends to be
    a low-complexity region of the loss landscape.


### The min_delta Parameter

Beyond patience, early stopping typically monitors SIGNIFICANT improvement:

    min_delta: minimum change in the monitored metric to qualify as
               an improvement.

    Example: min_delta=0.001, patience=10
    → An improvement from val_loss=0.502 to val_loss=0.5019 is
      NOT counted as an improvement (delta = 0.0001 < 0.001)
    → Only improvements larger than 0.001 reset the patience counter

    Without min_delta: training could continue indefinitely by making
    tiny improvements (1e-6 per epoch) that have no practical significance.


### Restore Best Weights

The model at the stopping epoch is NOT the same as the model at
the best validation epoch — training continued for patience epochs after
the best checkpoint. Always restore from the best checkpoint:

    Callback pseudocode:
    best_val_loss = +inf
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_one_epoch()
        val_loss = evaluate_on_validation_set()

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_weights = copy(model.weights)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            model.weights = best_weights  ← restore before returning
            break

    Final model is from the best checkpoint, not the stopping epoch.


### What Metric to Monitor

    Regression:        validation MSE or validation MAE
    Classification:    validation cross-entropy loss (not accuracy!)
                       → accuracy is coarse (integer steps); loss is smooth
    Object detection:  validation mAP (but noisy — use larger patience)
    Language models:   validation perplexity or validation loss
    Generative models: domain-specific (FID for images, BLEU for translation)

    WHY use loss not accuracy: Accuracy changes in discrete jumps (each
    misclassified example costs exactly 1 accuracy point). Loss changes
    continuously and reflects how CONFIDENT the model is, not just whether
    it is right. A model that changes a prediction from 51% to 99% correct
    shows no accuracy improvement but a significant loss improvement.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 8 — OTHER REGULARISATION TECHNIQUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Data Augmentation as Regularisation

Data augmentation is regularisation applied in INPUT SPACE rather than
parameter space. By transforming training examples (flips, crops, colour
jitter, rotations), augmentation creates new training examples that the
model must learn to classify consistently.

    Effect: enforces INVARIANCES in the learned representation.
    A model that sees horizontally-flipped cats during training learns
    that "cat-ness" is not orientation-specific.

    Most powerful augmentation strategies (in terms of regularisation strength):
    Mixup (Zhang et al., 2018):
        x_mix = λ·x₁ + (1-λ)·x₂      [interpolate two images]
        y_mix = λ·y₁ + (1-λ)·y₂      [interpolate their labels]
        The model must output a mixture of labels for a mixture of inputs.
        Forces smooth, linear interpolation in the learned feature space.

    CutMix (Yun et al., 2019):
        Cuts a rectangular patch from one image and pastes it into another.
        Labels are mixed proportionally to the patch area.
        More aggressive than Mixup — the model sees a genuinely
        multi-content image.

    RandAugment (Cubuk et al., 2020):
        Uniformly samples N augmentation operations from a fixed library
        (rotate, shear, contrast, colour, etc.) and applies them with
        magnitude M. Both N and M are hyperparameters.
        Eliminates the need to design augmentation policies manually.

    Data augmentation is often more effective than weight regularisation
    (L1/L2) for image models because it regularises the INPUT distribution
    directly rather than the parameter space.


### Batch Normalisation as Implicit Regularisation

BatchNorm is primarily a training stability technique (see Module 10 —
Weight Init & Normalisation), but it has an IMPLICIT regularisation effect:

    During training, BatchNorm computes statistics over the current mini-batch.
    The mean and variance of the batch change each step due to stochastic
    sampling — each example is normalised by slightly different statistics
    each time it appears (because its batch partners change).

    This is equivalent to adding STOCHASTIC NOISE to each example's
    normalised representation. The noise has zero mean and variance
    proportional to (1 - 1/B) where B is the batch size.

    Consequence: BatchNorm acts as a regulariser — models with BN
    often need LESS dropout or L2 regularisation. The two techniques
    partially overlap.

    Interaction warning: Dropout + BatchNorm can INTERFERE:
    Dropout changes the mean and variance of its input to BN
    (dropped units shift the distribution). BN then normalises
    based on these shifted statistics. At inference, dropout is off
    but BN still expects the shifted statistics, creating a mismatch.
    Solutions: use BN before dropout, or avoid dropout in BN layers.


### Gradient Noise

Adding Gaussian noise to gradients during training:

    g_noisy = g + N(0, σ²)

    Where σ² is typically annealed:  σ²(t) = η / (1 + t)^0.55

    This helps the optimiser escape sharp minima and find flatter
    minima that generalise better. It is rarely used explicitly because
    the stochasticity of mini-batch sampling already provides this effect.
    Gradient noise becomes relevant for very small batch sizes or
    for deterministic (full-batch) gradient descent.


### Weight Tying as Structural Regularisation

In language models, the input embedding matrix (vocabulary × dimension)
and the output projection matrix (dimension × vocabulary) are often
shared (tied):

    E_in = E_out^T

    Both matrices have the same shape (vocabulary × embedding_dim).
    Tying them halves the parameter count for the embedding/unembedding
    layers, which are typically the largest in small transformer models.

    This is structural regularisation — a hard constraint on the model
    architecture that prevents these two large matrices from diverging.
    Used in: GPT-2, BERT, most transformer language models.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 9 — COMBINED STRATEGY & DECISION FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Diagnosing Whether You Need Regularisation

Before adding regularisation, diagnose the problem:

    Step 1: Train to convergence without regularisation.
    Step 2: Compare training loss and validation loss.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Training loss │ Validation loss │ Diagnosis      │ Fix              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ High          │ High            │ Underfitting   │ More capacity,   │
    │               │                 │ (high bias)    │ more training,   │
    │               │                 │                │ less regularisa. │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Low           │ High            │ Overfitting    │ More regularisa. │
    │               │                 │ (high variance)│ more data,       │
    │               │                 │                │ less capacity    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Low           │ Low             │ Good fit       │ No action needed │
    ├─────────────────────────────────────────────────────────────────────┤
    │ High          │ Lower than train│ Distribution   │ Check data       │
    │               │ (unusual)       │ shift or bug   │ pipeline         │
    └─────────────────────────────────────────────────────────────────────┘


### Regularisation Strength vs Data Size

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Dataset size  │ Recommended regularisation approach                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ < 1,000       │ Strong L2 (λ=0.1), heavy dropout (p=0.5),          │
    │               │ aggressive data augmentation, early stopping,        │
    │               │ consider a smaller model or transfer learning        │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 1K – 100K     │ Moderate L2 (λ=0.01), dropout (p=0.1–0.3),        │
    │               │ data augmentation, early stopping with patience=10  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 100K – 10M    │ Light L2 (AdamW λ=0.01), DropPath for transformers, │
    │               │ standard augmentation. Early stopping optional.      │
    ├──────────────────────────────────────────────────────────────────────┤
    │ > 10M         │ Light regularisation only (AdamW λ=0.01).          │
    │               │ Modern large models often train without dropout.      │
    │               │ Data augmentation is still beneficial.               │
    └──────────────────────────────────────────────────────────────────────┘


### Standard Regularisation Stacks by Architecture Type

    MLP (dense networks):
        AdamW weight decay (λ = 0.01)
        Dropout after every hidden layer (p = 0.3–0.5)
        Early stopping (patience = 10–20)

    CNN (image classification):
        AdamW weight decay (λ = 1e-4 to 1e-3)
        SpatialDropout2d after conv blocks (p = 0.1–0.2)
        Aggressive data augmentation (RandAugment, Mixup, CutMix)
        Label smoothing (α = 0.1) for final classification head
        Early stopping (patience = 15–30)

    Transformer (vision — ViT, Swin):
        AdamW weight decay (λ = 0.05–0.1)
        DropPath / Stochastic Depth (p_max = 0.1–0.3)
        Attention dropout (p = 0.0–0.1)
        Mixup + CutMix augmentation
        Label smoothing (α = 0.1)

    Transformer (language — GPT, BERT):
        AdamW weight decay (λ = 0.01) on non-norm, non-bias params
        Residual dropout (p = 0.1)
        Attention dropout (p = 0.1)
        Embedding dropout (p = 0.1)
        Early stopping on validation perplexity

    Graph Neural Networks:
        L2 weight decay (λ = 5e-4)
        Dropout on node features (p = 0.5)
        DropEdge (randomly drop edges, analogous to DropPath)


### Regularisation Interaction Rules

    ┌───────────────────────────────────────────────────────────────────────┐
    │ Technique A    │ Technique B   │ Interaction                          │
    ├───────────────────────────────────────────────────────────────────────┤
    │ L2 / AdamW     │ Dropout       │ Complement each other well.          │
    │                │               │ Use both; tune λ and p independently. │
    ├───────────────────────────────────────────────────────────────────────┤
    │ BatchNorm      │ Dropout       │ Interact poorly. Prefer to avoid     │
    │                │               │ using dropout after BN layers.        │
    │                │               │ Use one or the other per block.       │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Label smoothing│ Distillation  │ DO NOT combine. Label smoothing on   │
    │                │               │ the teacher degrades soft targets.    │
    ├───────────────────────────────────────────────────────────────────────┤
    │ DropPath       │ Dropout       │ Can be combined but rarely both are  │
    │                │               │ needed. DropPath is preferred for     │
    │                │               │ residual architectures.               │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Data augment.  │ L2 / Dropout  │ Complement each other. Augmentation  │
    │                │               │ regularises input space; L2/dropout  │
    │                │               │ regularise parameter space.           │
    ├───────────────────────────────────────────────────────────────────────┤
    │ Early stopping │ Any           │ Compatible with all. Acts as a        │
    │                │               │ safety net — limits overfitting when  │
    │                │               │ other techniques are insufficient.    │
    └───────────────────────────────────────────────────────────────────────┘


### Regularisation Strength Tuning Protocol

    1. Start with a standard stack (AdamW λ=0.01, dropout p=0.1 for
       transformers, p=0.3 for MLPs).

    2. Train to convergence and measure the generalisation gap:
       gap = val_loss - train_loss

    3. If gap is large (overfitting):
       → Increase dropout p by 0.1, or increase λ by 3-10×, one at a time.
       → Add or strengthen data augmentation.
       → Reduce model capacity if the gap remains large.

    4. If gap is near zero but performance is still poor (underfitting):
       → Reduce regularisation (lower λ, lower p).
       → Increase model capacity.
       → Train for more epochs (raise early stopping patience).

    5. Never tune regularisation and architecture simultaneously — you
       will not be able to attribute changes in performance to either.

    6. Use cross-validation for small datasets (< 5,000 examples) to
       get a stable estimate of validation performance before tuning
       regularisation strength.

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
        from Training_Core.visuals.Regularisation_visual import (
            REGULARISATION_VISUAL_HTML,
            REGULARISATION_VISUAL_HEIGHT,
        )
        visual_html   = REGULARISATION_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = REGULARISATION_VISUAL_HEIGHT
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