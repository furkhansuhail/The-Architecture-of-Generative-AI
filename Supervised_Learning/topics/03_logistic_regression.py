"""
Module 02 · Logistic Regression — Linear Classification via Probability
============================================================

Logistic Regression is the natural evolution of linear regression when the output
is a category rather than a continuous value. It answers the question: given what
I know about this input, what is the probability it belongs to class 1?

Despite its name, logistic regression is a classification algorithm. The word
"regression" refers to the underlying linear model — it regresses a linear
combination of inputs, then squashes the result into a probability with the sigmoid.

"""

import base64
import os
import textwrap
import re

DISPLAY_NAME = "03 · Logistic Regression"
ICON         = "🔀"
SUBTITLE     = "Linear regression's classification cousin — probabilities via the sigmoid"

THEORY = """

### What is Logistic Regression?

Despite the name, Logistic Regression is a **classification** algorithm. 
It models the probability that an input belongs to a class by squashing a linear function through the
**sigmoid** (logistic) function.

Logistic Regression is a binary classification algorithm that models the probability
that an input x belongs to class 1. It takes the exact same linear combination you
saw in linear regression — w·x + b — and passes it through the sigmoid function to
compress the output into the range (0, 1), where it can be interpreted as a probability.

The name is slightly misleading: this is a classification algorithm, not a regression
algorithm. The "regression" refers to the internal linear model. The classification
decision comes from thresholding the output probability.

Think of it like a doctor making a diagnosis. They look at many features — age, blood
pressure, test results — weight each one by its importance (the weights), combine them
into a single risk score (the linear combination z), and then convert that score into
a probability of disease (the sigmoid). If the probability exceeds 50%, the diagnosis
is positive.

    Things that exist inside the model (learnable parameters):
        - Weights (w₁, w₂, ..., wₚ)  — importance of each feature for classification
        - Bias (w₀)                   — baseline log-odds of class 1

    Things that exist only at setup time (hyperparameters / configuration):
        - Learning rate (α)           — controls gradient descent step size
        - Regularisation strength (C or λ) — controls overfitting penalty
        - Decision threshold          — probability cutoff for class 1 (default 0.5)
        - Solver                      — optimisation algorithm (lbfgs, sgd, newton, etc.)

### Logistic Regression as Empirical Risk Minimisation (ERM)

Just as with linear regression, we can precisely state what logistic regression *is*
within the ERM framework:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Hypothesis class:  H = { f(x) = σ(w·x + b) | w ∈ ℝᵖ }      │
    │                     (sigmoid-of-linear functions)           │
    │                                                             │
    │  Loss function:     L(y, ŷ) = −[y log(ŷ) + (1−y) log(1−ŷ)]  │
    │                     (binary cross-entropy / log loss)       │
    │                                                             │
    │  Training objective:                                        │
    │      min_w  (1/n) Σᵢ −[yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]       │
    │                                                             │
    │  Optimiser:  Gradient Descent (no closed-form solution)     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Compared to linear regression (which uses the same hypothesis class structure but
different output and loss), the two changes are: (1) the prediction is squashed
through sigmoid to produce a probability, and (2) the loss is cross-entropy instead
of MSE. Everything else — the linear model, gradient descent, regularisation — is
structurally identical.

This is the ERM template in action: same framework, different modelling choices.


### The Inductive Bias of Logistic Regression

Every model encodes beliefs about the world before seeing data. For logistic regression:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Logistic regression assumes:                               │
    │                                                             │
    │  1. LINEAR SEPARABILITY (soft) — the two classes can be     │
    │     approximately separated by a straight hyperplane in     │
    │     feature space (the decision boundary is always linear)  │
    │                                                             │
    │  2. ADDITIVITY — each feature contributes independently     │
    │     to the log-odds of class membership                     │
    │                                                             │
    │  3. LOG-ODDS LINEARITY — the log-odds ln(P/(1-P)) is a      │
    │     linear function of the features (this is what the       │
    │     model actually fits, not the probabilities directly)    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

When these beliefs are approximately true, logistic regression is fast, interpretable,
and generalises well. When the true decision boundary is curved or non-linear, you need
feature engineering (adding polynomial terms), kernels, or neural networks.

---
### Part 1 . The Sigmoid Function - The Core Components 

### The Sigmoid Function:

The sigmoid function (also called the logistic function) is the heart of the model:

            σ(z) = 1 / (1 + e^(-z))     where z = wᵀx

Where z = w·x + b is the linear combination of inputs and weights (the same z you
computed in linear regression). The sigmoid maps any real number z to the interval (0, 1).

Key properties of σ(z):
    - σ(0) = 0.5        → when z = 0, the model is maximally uncertain
    - σ(z) → 1 as z → +∞ → large positive z = confident class 1 prediction
    - σ(z) → 0 as z → -∞ → large negative z = confident class 0 prediction
    - σ(-z) = 1 - σ(z)  → symmetric around z = 0

The derivative has a beautiful closed form:

                    σ'(z) = σ(z) · (1 − σ(z))

This is crucial for backpropagation in neural networks — the derivative of sigmoid
is expressible entirely in terms of the sigmoid output itself, making it cheap to
compute.


    # ======================================================================================= # 
    **Diagram 1 — The Sigmoid Function:**

    OUTPUT OF σ(z) = 1 / (1 + e⁻ᶻ)
    ══════════════════════════════════════════════════════════════

    σ(z)
    1.0 │                          ___________
        │                      ___/
        │                   __/
    0.5 │─────────────────-/───────────────── ← Decision threshold (default)
        │               _/
        │            __/
        │          _/
    0.0 │__________/
        └──────────────────────────────────────→ z
              -6   -4   -2    0    2    4    6

    Key values:
    ──────────────────────────────────────────
    z = -6   →  σ = 0.002  (very confident: class 0)
    z = -2   →  σ = 0.119  (leaning toward class 0)
    z =  0   →  σ = 0.500  (maximally uncertain)
    z = +2   →  σ = 0.881  (leaning toward class 1)
    z = +6   →  σ = 0.998  (very confident: class 1)
    ──────────────────────────────────────────

    Compare with the perceptron's step function:
    ──────────────────────────────────────────
    Step function: hard 0 or 1 at z = 0 (not differentiable)
    Sigmoid:       smooth 0-to-1 curve  (differentiable everywhere)
    ──────────────────────────────────────────
    # ======================================================================================= # 

- Output is always in `(0, 1)` — interpretable as a probability
- Decision boundary: predict class 1 if `σ(z) ≥ 0.5`, i.e. if `z ≥ 0`


### The Full Prediction Pipeline:

                    z = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ       (linear combination)

                    ŷ = σ(z) = 1 / (1 + e⁻ᶻ)                 (probability of class 1)

                    class = 1  if ŷ ≥ 0.5  (equivalently, if z ≥ 0)
                    class = 0  if ŷ < 0.5  (equivalently, if z < 0)


    # ======================================================================================= # 
    **Diagram 2 — Full Logistic Regression Pipeline:**

    FULL PIPELINE
    ══════════════════════════════════════════════════════════════

    Inputs X          Weights W         Linear sum    Sigmoid      Probability
    ┌──────┐         ┌──────┐
    │x₁=0.8├────×────┤w₁=1.2│──┐
    └──────┘         └──────┘  │
                               ├──► z = Σwᵢxᵢ + b ──► σ(z) ──► ŷ ∈ (0,1) ──► class
    ┌──────┐         ┌──────┐  │          ▲
    │x₂=0.3├────×────┤w₂=0.5│──┘          │
    └──────┘         └──────┘         ┌──┴───┐
                                       │b=0.1 │
                                       └──────┘

    Step by step for this example:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. LINEAR: z = (1.2×0.8) + (0.5×0.3) + 0.1              │
    │            z = 0.96 + 0.15 + 0.1 = 1.21                 │
    │                                                         │
    │ 2. SIGMOID: ŷ = 1 / (1 + e⁻¹·²¹)                        │
    │             ŷ = 1 / (1 + 0.298) = 0.770                 │
    │                                                         │
    │ 3. CLASSIFY: ŷ = 0.770 ≥ 0.5  →  predict class 1        │
    │              "77% confident this is class 1"            │
    └─────────────────────────────────────────────────────────┘
    # ======================================================================================= # 


### Why Not Use Linear Regression for Classification?

A natural question: why not just use linear regression and threshold at 0.5?

There are three reasons this fails:

1. **Unbounded outputs**: Linear regression predicts any real number. Predictions of
   1.7 or -0.3 have no probabilistic interpretation and make the threshold arbitrary.

2. **Wrong loss function**: MSE treats being wrong by 0.1 the same whether you're at
   0.4 (near the boundary) or 0.99 (very confident and very wrong). For classification,
   a confident wrong prediction should be penalised far more than an uncertain one.

3. **Non-convex loss**: If you minimise MSE with a step function output (to get 0/1
   predictions), the loss surface has discontinuities and no gradient. Gradient descent
   can't work. The sigmoid fixes this by making the output smooth.

---

### Part 2: The Loss Function — Binary Cross-Entropy
 
Instead we use:

L = -(1/n) Σ [ yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ) ]

This is **log loss** — it heavily penalises confident wrong predictions.

In the previous module, we justified MSE through MLE: minimising MSE is equivalent to
maximising likelihood under Gaussian noise. The same reasoning tells us which loss to
use for classification — we just need the right noise model.

For binary classification, the output y is not a continuous number with Gaussian noise.
It's a coin flip: y ∈ {0, 1}. The correct probabilistic model is Bernoulli.

**The Probabilistic Interpretation:**

Assume the data is generated by:

            P(y = 1 | x; w) = σ(w·x + b) = ŷ

That is: given input x and weights w, the probability of observing class 1 is ŷ.
Equivalently:

            P(y = 0 | x; w) = 1 − ŷ

For a single training example, the likelihood of observing label yᵢ is:

            P(yᵢ | xᵢ; w) = ŷᵢʸⁱ · (1 − ŷᵢ)¹⁻ʸⁱ

This compact formula handles both cases:
    if yᵢ = 1:   P = ŷᵢ¹ · (1 − ŷᵢ)⁰ = ŷᵢ        ← probability of class 1
    if yᵢ = 0:   P = ŷᵢ⁰ · (1 − ŷᵢ)¹ = 1 − ŷᵢ    ← probability of class 0

Taking the log-likelihood over all n training examples:

    log L = Σᵢ [ yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ) ]

Maximising log-likelihood is the same as minimising the negative log-likelihood,
which is binary cross-entropy:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  BCE = -(1/n) Σᵢ [ yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ) ]          │
    │                                                             │
    │  Logistic Regression + BCE                                  │
    │      = Maximum Likelihood Estimation                        │
    │           under the Bernoulli distribution                  │
    │                                                             │
    │  Compare: Linear Regression + MSE = MLE under Gaussian      │
    │           Logistic Regression + BCE = MLE under Bernoulli   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The distribution assumption determines the loss function. This pattern generalises
to every regression and classification model in machine learning.


### Unpacking the Cross-Entropy Loss:

    BCE = -(1/n) Σᵢ [ yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ) ]

For any single example, only one of the two terms is active:

    When yᵢ = 1:   loss = -log(ŷᵢ)
    When yᵢ = 0:   loss = -log(1 − ŷᵢ)

The −log function is the key. It is zero when its argument is 1 (confident and correct)
and grows to infinity as its argument approaches 0 (confident and wrong).


    # ======================================================================================= # 
    **Diagram 3 — Why Cross-Entropy Punishes Confident Mistakes:**

    LOSS vs PREDICTED PROBABILITY (for true label y = 1)
    ══════════════════════════════════════════════════════════════

    Loss = -log(ŷ)

    ∞ │
      │
    4 │●
      │ ╲
    3 │  ╲
      │   ╲
    2 │    ╲
      │     ╲
    1 │      ╲____
      │            ‾‾‾──────...
    0 │                         ────────────────●
      └──────────────────────────────────────────→ ŷ
      0   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0

    ──────────────────────────────────────────────────
    ŷ = 0.95  (confident, correct)   → loss =  0.051 ← small penalty
    ŷ = 0.70  (moderate confidence)  → loss =  0.357
    ŷ = 0.50  (uncertain)            → loss =  0.693
    ŷ = 0.20  (leaning wrong way)    → loss =  1.609
    ŷ = 0.05  (confident, WRONG)     → loss =  2.996 ← severe penalty
    ──────────────────────────────────────────────────

    MSE would give:  ŷ = 0.95 → loss = 0.0025
                     ŷ = 0.05 → loss = 0.9025
    Much smaller penalty for confident wrong predictions.
    Cross-entropy screams loud when you're confidently wrong. MSE barely notices.
    # =======================================================================================# 


### Why Is Cross-Entropy Convex? (And Why MSE Is Not):

When using sigmoid output, MSE creates a non-convex loss surface — gradient descent
can get stuck in local minima and never find the global optimum. Cross-entropy with
sigmoid output is provably convex: there is exactly one global minimum, and gradient
descent is guaranteed to find it.

This is not a coincidence. It follows directly from the MLE perspective: cross-entropy
is the canonical loss for Bernoulli outputs, and MLE with Bernoulli gives a log-concave
likelihood — which means a convex negative log-likelihood.

---

### Part 3: Training — Gradient Descent

The gradient of log loss w.r.t. weights is elegantly simple:

∇L = (1/n) Xᵀ (ŷ - y)
w  := w - α ∇L

Same form as linear regression — only the prediction `ŷ = σ(Xw)` differs.


### The Gradient of Binary Cross-Entropy:

Despite the intimidating formula for BCE, its gradient with respect to the weights
is elegantly simple:

                        ∇L = (1/n) Xᵀ(ŷ − y)

This is almost identical to the gradient we derived for MSE in linear regression:

    Linear Regression:    ∇L = (2/n) Xᵀ(Xw − y)     where ŷ = Xw
    Logistic Regression:  ∇L = (1/n) Xᵀ(ŷ − y)       where ŷ = σ(Xw)

The only differences are the constant (2/n vs 1/n, absorbed into α anyway) and the
fact that ŷ is now a sigmoid of the linear combination rather than the linear
combination itself. The update rule is otherwise structurally identical.


**Derivation Sketch (Why the Gradient is So Clean):**

The chain rule on the BCE + sigmoid combination produces a miraculous cancellation.

Step 1 — derivative of BCE with respect to ŷ:
    ∂L/∂ŷᵢ = −yᵢ/ŷᵢ + (1−yᵢ)/(1−ŷᵢ)

Step 2 — derivative of sigmoid with respect to z:
    ∂ŷᵢ/∂zᵢ = σ(zᵢ)(1 − σ(zᵢ)) = ŷᵢ(1 − ŷᵢ)

Step 3 — chain rule (∂L/∂z = ∂L/∂ŷ · ∂ŷ/∂z):
    ∂L/∂zᵢ = [−yᵢ/ŷᵢ + (1−yᵢ)/(1−ŷᵢ)] · ŷᵢ(1−ŷᵢ)
            = −yᵢ(1−ŷᵢ) + (1−yᵢ)ŷᵢ
            = ŷᵢ − yᵢ                                   ← the cancellation!

The ŷᵢ(1−ŷᵢ) in step 2 exactly cancels the denominators in step 1. This is why
cross-entropy and sigmoid were designed to pair together — they are the canonical
matched pair for Bernoulli MLE, and their combination produces the cleanest possible
gradient.

Step 4 — chain rule down to weights (∂z/∂wⱼ = xᵢⱼ):
    ∂L/∂wⱼ = (1/n) Σᵢ (ŷᵢ − yᵢ) · xᵢⱼ

In matrix form:  ∇L = (1/n) Xᵀ(ŷ − y)   ← clean, elegant, computationally simple.


### Weight Update Rule:

                        w := w − α · (1/n) Xᵀ(ŷ − y)

where ŷ = σ(Xw) must be recomputed at each step since ŷ depends on the current w.


### Concrete Training Walkthrough:

Let's trace two full iterations training a logistic regression from scratch on a
2D binary classification problem.

    Dataset (n=4, p=2 features):
    ──────────────────────────────────────────────────────────────
    x₁     x₂     y_true    Interpretation
    ──────────────────────────────────────────────────────────────
    2.0    1.0     1         Tumour (malignant)
    1.0    2.0     1         Tumour (malignant)
    -1.0  -1.0     0         Benign
    -2.0  -2.0     0         Benign
    ──────────────────────────────────────────────────────────────

    Initial weights: w₀ = 0 (bias), w₁ = 0.0, w₂ = 0.0
    Learning rate: α = 0.1

    === EPOCH 1 ===

    Step 1 — Compute z (linear combination) for all 4 examples:
        z₁ = 0 + 0×2 + 0×1 = 0.0
        z₂ = 0 + 0×1 + 0×2 = 0.0
        z₃ = 0 + 0×(-1) + 0×(-1) = 0.0
        z₄ = 0 + 0×(-2) + 0×(-2) = 0.0

    Step 2 — Apply sigmoid to get predicted probabilities:
        ŷ₁ = σ(0) = 0.500
        ŷ₂ = σ(0) = 0.500
        ŷ₃ = σ(0) = 0.500
        ŷ₄ = σ(0) = 0.500

    Step 3 — Compute errors (ŷ - y_true):
        error₁ = 0.5 - 1 = -0.5   (should be 1, predicted 0.5)
        error₂ = 0.5 - 1 = -0.5   (should be 1, predicted 0.5)
        error₃ = 0.5 - 0 = +0.5   (should be 0, predicted 0.5)
        error₄ = 0.5 - 0 = +0.5   (should be 0, predicted 0.5)

    Step 4 — Compute gradients:

        ∂L/∂w₀ = (1/4)[(-0.5)(1) + (-0.5)(1) + (0.5)(1) + (0.5)(1)]
               = (1/4)[−0.5 − 0.5 + 0.5 + 0.5] = 0.0

        ∂L/∂w₁ = (1/4)[(-0.5)(2) + (-0.5)(1) + (0.5)(-1) + (0.5)(-2)]
               = (1/4)[−1 − 0.5 − 0.5 − 1] = (1/4)(−3) = −0.75

        ∂L/∂w₂ = (1/4)[(-0.5)(1) + (-0.5)(2) + (0.5)(-1) + (0.5)(-2)]
               = (1/4)[−0.5 − 1 − 0.5 − 1] = (1/4)(−3) = −0.75

    Step 5 — Update weights (w := w − α · ∇L):
        w₀ := 0.0 − 0.1 × 0.0  = 0.000
        w₁ := 0.0 − 0.1 × (−0.75) = +0.075
        w₂ := 0.0 − 0.1 × (−0.75) = +0.075

    After epoch 1: w₀=0.000, w₁=0.075, w₂=0.075

    Intuition: the gradient is negative for w₁ and w₂, so we increase both weights.
    The positive-class examples (2,1) and (1,2) have larger feature values than the
    negative-class examples (-1,-1) and (-2,-2). Increasing w₁ and w₂ will produce
    larger z for positive examples and smaller z for negative examples — correct!

    === EPOCH 2 ===

    Step 1 — New z values with updated weights:
        z₁ = 0 + 0.075×2 + 0.075×1 = 0.225
        z₂ = 0 + 0.075×1 + 0.075×2 = 0.225
        z₃ = 0 + 0.075×(-1) + 0.075×(-1) = -0.150
        z₄ = 0 + 0.075×(-2) + 0.075×(-2) = -0.300

    Step 2 — Apply sigmoid:
        ŷ₁ = σ(0.225) ≈ 0.556
        ŷ₂ = σ(0.225) ≈ 0.556
        ŷ₃ = σ(−0.15) ≈ 0.463
        ŷ₄ = σ(−0.30) ≈ 0.426

    Already better: class 1 examples (≈0.556) getting higher probability than
    class 0 examples (≈0.426−0.463). The model is learning the boundary.

    After many more iterations, the weights will converge to values that clearly
    separate the two classes, with w₁ and w₂ both positive and large enough that
    positive-class examples produce σ(z) ≫ 0.5.

---

### Part 4: The Decision Boundary — What Is the Model Geometry?


### The Decision Boundary Is Always a Hyperplane:

The logistic regression model predicts class 1 when σ(z) ≥ 0.5, which is when z ≥ 0.
The decision boundary is therefore:

                        w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ = 0

This is the equation of a hyperplane — in 2D, a straight line; in 3D, a flat plane.
The decision boundary of logistic regression is always linear, regardless of the
probability values on either side.

This is the same structural insight from the perceptron: the weight vector W is
perpendicular to the decision boundary and points toward the positive class. The bias
w₀ shifts the boundary away from the origin.


    # =======================================================================================# 
    **Diagram 4 — The Linear Decision Boundary in 2D:**

    BINARY CLASSIFICATION: Two features, linear boundary
    ══════════════════════════════════════════════════════════════

    x₂
    ↑
    │        ●  ●          Decision boundary: w₁x₁ + w₂x₂ + w₀ = 0
    │      ●  ●            (straight line in 2D)
    │   ● ●      \
    │          ●  \← boundary
    │     ○     ○  \u005c         ● = Class 1
    │   ○   ○      \u005c         ○ = Class 0
    │         ○  ○  \
    │                \
    └───────────────────────→ x₁

    ──────────────────────────────────────────────────────────
    Above the boundary (w·x > 0):  σ(z) > 0.5  →  Class 1
    Below the boundary (w·x < 0):  σ(z) < 0.5  →  Class 0
    On the boundary (w·x = 0):     σ(z) = 0.5  →  Uncertain
    ──────────────────────────────────────────────────────────

    PROBABILITY CONTOURS (iso-probability lines):

    x₂                ↑ high ŷ (confident class 1)
    ↑         0.95
    │      0.80
    │   0.65
    │0.50 ← decision boundary
    │   0.35
    │      0.20
    │         0.05
    └──────────────────────────────────→ x₁
                     ↓ low ŷ (confident class 0)

    Probabilities "spread out" from the decision boundary symmetrically.
    The further from the boundary, the more confident the prediction.
    # =======================================================================================# 


### The Log-Odds (Logit) Interpretation:

There is a beautiful algebraic way to see what logistic regression is actually fitting.
Define the odds of class 1 as P/(1-P). Taking the log:

                log[ŷ/(1 − ŷ)] = w·x + b

The left side is the log-odds (called the logit). The right side is a linear function.
So logistic regression is actually fitting a linear model for the log-odds, not for the
probability directly:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Logistic regression is linear regression for the log-odds  │
    │                                                             │
    │  logit(P(y=1|x)) = w·x + b                                  │
    │                                                             │
    │  A unit increase in feature j multiplies the odds by eʷʲ    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

This is why the weights in logistic regression are interpreted as log-odds ratios.
If w₁ = 0.5 for the "age" feature, then each additional year of age multiplies
the odds of the outcome by e^0.5 ≈ 1.65 — i.e., increases odds by 65%.


    # =======================================================================================# 
    **Diagram 5 — Log-Odds (Logit) as Linear Function:**

    WHAT LOGISTIC REGRESSION IS FITTING:

    log(P/(1-P))                               P (probability)
    ↑                                          ↑
    4│        /                               1│        ─────
    3│       /                                 │     ─/
    2│      /                                  │   /
    1│     /                                   │  /
    0│────/──────→ z (linear score)          .5│─/──────────→ z
   -1│   /                                     │/
   -2│  /                                     0│───────────
   -3│ /                                       │
   -4│/                                        │

    Log-odds are LINEAR in z.          Probability is NONLINEAR (S-curve).
    This is what the model fits.       This is what we interpret.
    # =======================================================================================# 


### When Does Logistic Regression Fail?

The linear decision boundary is both the strength and limitation of logistic regression.

**Fails on XOR (non-linear separation):**

    Class 0:  (−1, −1), (+1, +1)     ← same sign inputs → class 0
    Class 1:  (−1, +1), (+1, −1)     ← different sign inputs → class 1

No straight line can separate these classes. The optimal boundary is curved (hyperbolic).
Logistic regression will perform no better than random on this problem.

**Fixes for non-linear boundaries:**

1. Feature engineering: add x₁² , x₂², x₁x₂ as new features (Polynomial Logistic Regression)
2. Kernel methods: implicitly map to high-dimensional space (Kernel SVM)
3. Stack multiple logistic regression units with non-linear activation → Neural Network

This is the direct bridge to neural networks: a hidden layer transforms the input
into a new representation where the classes may be linearly separable, and then
logistic regression is applied in that transformed space.


### Part 5: Regularisation

Same L1/L2 penalty as linear regression. In sklearn, the parameter is `C = 1/λ`
(inverse of regularisation strength — larger C = less regularisation).

### Why Regularisation Matters More for Logistic Regression:

In linear regression, overfitting typically manifests as large weight magnitudes but
the model still gives real-valued outputs. In logistic regression, large weights are
more dangerous: if w is very large, σ(z) is almost exactly 0 or 1 for most training
examples. The model becomes extremely overconfident.

Worse: with large weights, the gradient σ(z)(1−σ(z)) approaches zero for most
examples (since σ(z) ≈ 0 or ≈ 1 → σ(1−σ) ≈ 0). Gradient descent essentially
stops updating — the model is saturated. Regularisation prevents this.

**L2 Regularisation (Ridge Logistic Regression):**

        L_Ridge = BCE + (λ/2n) Σⱼ wⱼ²

The gradient adds a term λ/n · wⱼ to each weight update:

        w := w − α · [(1/n) Xᵀ(ŷ − y) + (λ/n) w]

**L1 Regularisation (Lasso Logistic Regression):**

        L_Lasso = BCE + (λ/n) Σⱼ |wⱼ|

Forces some weights to exactly zero — useful for high-dimensional problems where
many features are irrelevant (e.g., text classification with 50,000 word features).

**sklearn Convention:** sklearn uses C = 1/λ (inverse of regularisation strength).
    - C = 1 (default): moderate regularisation
    - C → ∞: no regularisation (equivalent to plain logistic regression)
    - C → 0: very heavy regularisation, weights approach zero


### The Generalisation Gap:

Just as in linear regression, the difference between training loss and test loss is
the generalisation gap. Logistic regression can overfit severely when:
    - The number of features p is large relative to n (high-dimensional data)
    - Features are very predictive of training labels but noisy on new data
    - Training data is not representative of the test distribution

Regularisation (L1 or L2) directly reduces the generalisation gap by penalising
complexity, trading a small increase in training loss for a larger decrease in test loss.

---

### Part 6: Multi-Class Extensions

- **One-vs-Rest (OvR)** — train k binary classifiers (one per class)
- **Softmax Regression** — direct multi-class generalisation

softmax(zⱼ) = e^zⱼ / Σₖ e^zₖ


### One-vs-Rest (OvR):

For k classes, train k independent binary classifiers. Classifier j answers:
"Is this example class j or not?" To predict, run all k classifiers and pick
the one with the highest predicted probability.

    Class A classifier: P(y = A | x) = σ(wₐ·x)
    Class B classifier: P(y = B | x) = σ(w_b·x)
    Class C classifier: P(y = C | x) = σ(w_c·x)
    
    Predict: argmax over {A, B, C} of their respective σ outputs

Simple but the k probabilities don't sum to 1 (they're from independent models).


### Softmax Regression (Multinomial Logistic Regression):

The natural multi-class generalisation. Instead of k independent sigmoids, use one
joint model with k weight vectors {w₁, ..., wₖ}:

            P(y = j | x) = exp(wⱼ·x) / Σₖ exp(wₖ·x) = softmax(z)ⱼ

Properties of softmax:
- All class probabilities are in (0, 1)
- They sum to exactly 1 → valid probability distribution
- The largest zⱼ gets the largest probability (argmax-like but smooth)
- Reduces to sigmoid when k = 2


    # ======================================================================================= # 
    **Diagram 6 — Softmax Spreads Probability Across Classes:**

    SOFTMAX: k=3 classes, linear scores z = [2.0, 1.0, 0.5]
    ══════════════════════════════════════════════════════════════

    Raw scores z:    Class A = 2.0    Class B = 1.0    Class C = 0.5

    Step 1 — Exponentiate:
        exp(2.0) = 7.39    exp(1.0) = 2.72    exp(0.5) = 1.65

    Step 2 — Sum: 7.39 + 2.72 + 1.65 = 11.76

    Step 3 — Divide:
        P(A) = 7.39 / 11.76 = 0.629
        P(B) = 2.72 / 11.76 = 0.231
        P(C) = 1.65 / 11.76 = 0.140
        ─────────────────────────────
        Sum  =                1.000 ✓

    The class with the highest score (A) gets the majority of probability,
    but the others are not ignored — their probabilities reflect their scores.

    COMPARE: Binary sigmoid gives P(class 1) = 0.88 for z = 2.0
             Softmax with k=2 gives identical results as sigmoid
             Softmax with k > 2 is the natural generalisation
    # ======================================================================================= # 


### Cross-Entropy for Multi-Class (Categorical Cross-Entropy):

The loss extends naturally from binary BCE to k classes:

        L = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)

Where yᵢⱼ = 1 if example i belongs to class j (and 0 otherwise). This is a
one-hot encoding over classes. For a correctly classified example, only the
true class term is active — the others are multiplied by zero. The loss reduces
to: -log(probability assigned to the true class).

This is the same loss used in the output layer of every multi-class neural network.

---

Decision Boundary

The boundary is the set of points where `wᵀx + bias = 0`. It's always **linear** in feature space.
For non-linear boundaries, you need:
- Polynomial feature expansion
- Kernels (SVM)
- Neural networks

---

### Part 7: From Logistic Regression to Deep Learning


### The Direct Connection:

A single logistic regression unit and a single neuron in a neural network are the
same computation:

                    output = σ(w·x + b)

The only difference is where in the architecture they appear:
- Logistic regression: this is the entire model
- Neural network: this is one neuron in one layer

A neural network is literally stacks of logistic regression units (with various
activation functions) organised in layers, where the output of each layer is the
input to the next. The last layer's sigmoid (or softmax) is the direct descendant
of logistic regression.

the output layer of a binary classification network is logistic regression; 
hidden neurons historically used sigmoid before ReLU became dominant.

This is the full progression:

    Perceptron (1958)          ← step activation, binary prediction, no gradient
          │
          │  (replace step with sigmoid, use BCE loss)
          ↓
    Logistic Regression         ← sigmoid activation, probability output, MLE training
          │
          │  (stack multiple logistic regression units in layers)
          ↓
    Multi-Layer Perceptron      ← hidden layers transform input into separable space
          │
          │  (replace sigmoid with ReLU, add depth)
          ↓
    Deep Neural Networks        ← universal function approximators


**Why Logistic Regression is Not Enough:**

Logistic regression can only find a linear decision boundary. Composing multiple
logistic regression units — one layer — still gives a linear boundary (linear
compositions of linear functions are linear). The key breakthrough of neural networks
is that adding non-linear activation functions between layers creates non-linear
transformations of the input, allowing the final linear classifier to operate in a
"transformed" space where the classes may be separable.

This is why the activation function is non-negotiable in deep learning: without it,
the depth of a network is irrelevant.


**Comparing the Three Models Side-by-Side:**

    ────────────────────────────────────────────────────────────────────────
    Property              Perceptron        Logistic Reg.     Neural Net
    ────────────────────────────────────────────────────────────────────────
    Output                0 or 1 (hard)     P ∈ (0,1) (soft)  P ∈ (0,1)
    Activation            Step (not diff.)  Sigmoid           ReLU/Sigmoid
    Loss function         Perceptron rule   Cross-entropy     Cross-entropy
    Optimiser             Error correction  Gradient descent  Backprop
    Training guarantee    Converges if sep. Global minimum    Local min only
    Decision boundary     Linear            Linear            Non-linear
    Probabilistic output  No                Yes (MLE)         Yes (MLE)
    ────────────────────────────────────────────────────────────────────────



"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Runnable code demonstrations
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Logistic Regression from Scratch": {
        "description": "Full implementation with sigmoid, BCE loss, gradient descent, and step-by-step trace",
        "runnable": True,
        "code": '''
"""
================================================================================
LOGISTIC REGRESSION FROM SCRATCH — GRADIENT DESCENT
================================================================================
Unlike linear regression where the solution is computed analytically in one step, 
logistic regression uses gradient descent (an iterative optimizer).

We implement everything using only NumPy. No sklearn, no black boxes.

Architecture:

    x₁ ──(w₁)───┐
                │
    x₂ ──(w₂)───┼──► z = Σwᵢxᵢ + b ──► σ(z) ──► ŷ ∈ (0,1) ──► class {0,1}
                │
    xₚ ──(wₚ)────┘

Forward:  z = Xw,   ŷ = σ(z) = 1 / (1 + e⁻ᶻ)
Loss:     BCE = -(1/n) Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]
Gradient: ∇L = (1/n) Xᵀ(ŷ − y)     ← identical form to linear regression
Update:   w := w − α · ∇L

================================================================================
"""

import numpy as np
import math


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def sigmoid(z):
    """
    Sigmoid function: σ(z) = 1 / (1 + e^(-z))

    Maps any real number to (0, 1) — interpretable as a probability.

    Numerically stable implementation:
        For large negative z, e^(-z) overflows.
        We clip z to prevent this.
    """
    z = np.clip(z, -500, 500)   # prevent overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy loss.

    BCE = -(1/n) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]

    We clip predictions away from 0 and 1 to avoid log(0) = -∞.
    """
    eps = 1e-12   # numerical stability
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# =============================================================================
# LOGISTIC REGRESSION CLASS
# =============================================================================

class LogisticRegressionGD:
    """
    Binary Logistic Regression trained with Gradient Descent.

    Learnable parameters:
        self.weights : np.array of shape (p+1,)
                       weights[0] = bias w₀
                       weights[1:] = feature weights w₁...wₚ

    Hyperparameters (not learned):
        learning_rate : step size for gradient descent
        n_iterations  : number of gradient steps
        lambda_       : L2 regularisation strength (0 = no regularisation)
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, lambda_=0.0):
        self.lr      = learning_rate
        self.n_iter  = n_iterations
        self.lambda_ = lambda_
        self.weights = None
        self.loss_history = []

    def _add_bias(self, X):
        """Prepend a column of 1s for the bias term."""
        return np.c_[np.ones(len(X)), X]

    def fit(self, X, y, verbose=True):
        """
        Train via gradient descent.

        Forward pass:  z = Xw,  ŷ = σ(z)
        Gradient:      ∇L = (1/n) Xᵀ(ŷ - y)  +  (λ/n) w  (last term = L2)
        Update:        w := w - α · ∇L
        """
        X_b = self._add_bias(X)           # (n, p+1)
        n, p = X_b.shape

        # Initialise weights to small random values
        self.weights = np.random.randn(p) * 0.01

        print(f"  Initial weights: {self.weights.round(4)}")
        print(f"  Initial BCE loss: {binary_cross_entropy(y, sigmoid(X_b @ self.weights)):.4f}\\n")

        for i in range(self.n_iter):

            # ── Forward pass ─────────────────────────────────────────────────
            z     = X_b @ self.weights            # linear score, shape (n,)
            y_hat = sigmoid(z)                    # predicted probability, (n,)

            # ── Loss ─────────────────────────────────────────────────────────
            loss = binary_cross_entropy(y, y_hat)
            if self.lambda_ > 0:
                loss += (self.lambda_ / (2 * n)) * np.sum(self.weights[1:] ** 2)
            self.loss_history.append(loss)

            # ── Gradient ─────────────────────────────────────────────────────
            # Core gradient: (1/n) Xᵀ(ŷ − y)
            grad = (1 / n) * X_b.T @ (y_hat - y)

            # L2 regularisation gradient: (λ/n) w  (skip bias: index 0)
            if self.lambda_ > 0:
                reg_grad = (self.lambda_ / n) * self.weights
                reg_grad[0] = 0.0   # never regularise the bias
                grad += reg_grad

            # ── Weight update ─────────────────────────────────────────────────
            self.weights -= self.lr * grad

            if verbose and i % 200 == 0:
                preds = (y_hat >= 0.5).astype(int)
                acc = np.mean(preds == y)
                print(f"  Iter {i:>4} | BCE: {loss:.4f} | Acc: {acc:.3f}")

    def predict_proba(self, X):
        """Return predicted probabilities."""
        X_b = self._add_bias(X)
        return sigmoid(X_b @ self.weights)

    def predict(self, X, threshold=0.5):
        """Return binary class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


# =============================================================================
# DATASET: Linearly separable 2D binary classification
# =============================================================================
# True boundary: x₁ + x₂ = 0 (diagonal line through origin)
# Class 1: x₁ + x₂ > 0    Class 0: x₁ + x₂ < 0

np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float)   # true boundary: x₁ + x₂ = 0

print("=" * 60)
print("  LOGISTIC REGRESSION FROM SCRATCH")
print("=" * 60)
print(f"\\n  Dataset: {n} points, 2 features")
print(f"  True decision boundary: x₁ + x₂ = 0")
print(f"  Class 1: {int(y.sum())} points    Class 0: {int((1-y).sum())} points")
print()

model = LogisticRegressionGD(learning_rate=0.5, n_iterations=1000)
model.fit(X, y)

print(f"\\n  Final weights:")
print(f"    w₀ (bias)   = {model.weights[0]:.4f}  (expected ≈ 0)")
print(f"    w₁          = {model.weights[1]:.4f}  (expected > 0)")
print(f"    w₂          = {model.weights[2]:.4f}  (expected > 0)")

final_acc = np.mean(model.predict(X) == y)
print(f"\\n  Training accuracy: {final_acc:.4f}")

# The learned decision boundary: w₀ + w₁x₁ + w₂x₂ = 0
# Normalise to show that it's approximately x₁ + x₂ = 0
w1, w2 = model.weights[1], model.weights[2]
ratio = w1 / w2 if w2 != 0 else float("inf")
print(f"  Weight ratio w₁/w₂ = {ratio:.4f}  (expected ≈ 1.0 for x₁ + x₂ boundary)")


# =============================================================================
# STEP-BY-STEP SINGLE ITERATION TRACE
# =============================================================================
print("\\n" + "=" * 60)
print("  STEP-BY-STEP: ONE GRADIENT DESCENT ITERATION")
print("=" * 60)

# Tiny dataset matching the walkthrough in THEORY
X_tiny = np.array([[2.0, 1.0],
                    [1.0, 2.0],
                    [-1.0, -1.0],
                    [-2.0, -2.0]])
y_tiny = np.array([1.0, 1.0, 0.0, 0.0])
X_tiny_b = np.c_[np.ones(4), X_tiny]

w = np.array([0.0, 0.0, 0.0])   # initial weights
lr = 0.1

print(f"""
  Dataset (4 points, true boundary ≈ x₁ + x₂ = 0):
    x₁   x₂   y
     2.0  1.0  1  (Class 1)
     1.0  2.0  1  (Class 1)
    -1.0 -1.0  0  (Class 0)
    -2.0 -2.0  0  (Class 0)

  Initial weights: w₀={w[0]}, w₁={w[1]}, w₂={w[2]}
  Learning rate: α = {lr}
""")

# Forward pass
z     = X_tiny_b @ w
y_hat = sigmoid(z)
errors = y_hat - y_tiny
loss  = binary_cross_entropy(y_tiny, y_hat)

print("  STEP 1 — Linear scores z = Xw:")
for i in range(4):
    print(f"    z_{i+1} = {w[0]}×1 + {w[1]}×{X_tiny[i,0]} + {w[2]}×{X_tiny[i,1]} = {z[i]:.3f}")

print("\\n  STEP 2 — Sigmoid → predicted probabilities:")
for i in range(4):
    print(f"    ŷ_{i+1} = σ({z[i]:.3f}) = {y_hat[i]:.3f}    "
          f"(true y={int(y_tiny[i])}, error={errors[i]:+.3f})")

print(f"\\n  STEP 3 — BCE loss: {loss:.4f}")

grad = (1 / 4) * X_tiny_b.T @ errors
print(f"\\n  STEP 4 — Gradient ∇L = (1/4) Xᵀ(ŷ - y):")
print(f"    ∂L/∂w₀ = {grad[0]:.4f}")
print(f"    ∂L/∂w₁ = {grad[1]:.4f}")
print(f"    ∂L/∂w₂ = {grad[2]:.4f}")

w_new = w - lr * grad
print(f"\\n  STEP 5 — Weight update (w := w - α·∇L):")
print(f"    w₀ := {w[0]:.4f} - {lr} × {grad[0]:.4f} = {w_new[0]:.4f}")
print(f"    w₁ := {w[1]:.4f} - {lr} × {grad[1]:.4f} = {w_new[1]:.4f}")
print(f"    w₂ := {w[2]:.4f} - {lr} × {grad[2]:.4f} = {w_new[2]:.4f}")

z_new     = X_tiny_b @ w_new
y_hat_new = sigmoid(z_new)
loss_new  = binary_cross_entropy(y_tiny, y_hat_new)
print(f"\\n  New BCE loss: {loss_new:.4f}  (was {loss:.4f} — decreased ✓)")
print(f"  w₁ and w₂ both increased → positive class gets higher scores next step")
''',
    },

    "Bare Bone Logistic Regression":{
        "description": "How the learned weights define the linear decision boundary",
        "runnable": True,
        "code":'''

# =============================================================================
# Logistic Regression from Scratch
# =============================================================================
#
# WHAT IS LOGISTIC REGRESSION?
# -----------------------------------------------------------------------------
# Logistic Regression is a supervised machine learning algorithm used for
# CLASSIFICATION tasks — predicting which category an input belongs to.
#
# Despite having "regression" in its name, it is used to predict discrete
# class labels (e.g., 0 or 1, spam or not spam, disease or no disease),
# not continuous values.
#
#
# HOW IS IT DIFFERENT FROM LINEAR REGRESSION?
# -----------------------------------------------------------------------------
#
#   Feature               Linear Regression         Logistic Regression
#   ─────────────────────────────────────────────────────────────────────
#   Goal                  Predict a number          Predict a category
#   Output                Any real value            A probability (0 to 1)
#   Output example        House price = $320,000    Spam probability = 0.91
#   Final prediction      The number itself         Class 0 or 1 (threshold)
#   Core equation         y = w*x + b               y = sigmoid(w*x + b)
#   Training method       Analytical (OLS formula)  Iterative (Gradient Descent)
#   Loss function         Mean Squared Error (MSE)  Binary Cross-Entropy
#
# In short:
#   - Linear Regression draws a straight line to fit continuous data
#   - Logistic Regression draws an S-curve to separate two classes
#
#
# THE CORE IDEA
# -----------------------------------------------------------------------------
# Logistic Regression starts the same way as linear regression:
#       z = w * x + b          (a simple straight line)
#
# But instead of outputting z directly, it passes it through the SIGMOID
# function to squash the output into a probability between 0 and 1:
#
#       sigmoid(z) = 1 / (1 + e^(-z))
#
#   Input z        Sigmoid Output    Meaning
#   ────────────────────────────────────────────────
#   Very negative  ≈ 0.0             Strongly class 0
#   0              = 0.5             Uncertain / boundary
#   Very positive  ≈ 1.0             Strongly class 1
#
# The output is interpreted as:
#       P(y=1 | x) — the probability that the input x belongs to class 1
#
#
# HOW THE MODEL LEARNS (GRADIENT DESCENT)
# -----------------------------------------------------------------------------
# Unlike Linear Regression where slope and intercept are solved in ONE step
# using a formula, Logistic Regression learns iteratively over many epochs:
#
#   Step 1 — Forward pass:   Compute predictions using current w and b
#   Step 2 — Compute loss:   Measure how wrong the predictions are
#   Step 3 — Backward pass:  Compute gradients (direction to improve)
#   Step 4 — Update weights: Nudge w and b slightly in the right direction
#   Step 5 — Repeat for N epochs until the loss stops improving
#
# The weight update rule is:
#       w = w - learning_rate * dL/dw
#       b = b - learning_rate * dL/db
#
# where dL/dw and dL/db are the gradients of the loss with respect to w and b.
#
#
# LOSS FUNCTION — Binary Cross-Entropy
# -----------------------------------------------------------------------------
# Instead of Mean Squared Error (used in Linear Regression), Logistic
# Regression uses Binary Cross-Entropy loss, which heavily penalizes
# confident wrong predictions:
#
#       Loss = -mean( y*log(ŷ) + (1-y)*log(1-ŷ) )
#
# where y is the true label and ŷ is the predicted probability.
#
#   If y=1 and ŷ=0.99 → Loss is very small  (confident and correct)
#   If y=1 and ŷ=0.01 → Loss is very large  (confident and wrong)
#
#
# MAKING A PREDICTION
# -----------------------------------------------------------------------------
# Once trained, for any new input x:
#
#   1. Compute probability:  p = sigmoid(w*x + b)
#   2. Apply threshold:      if p >= 0.5 → predict class 1
#                            if p <  0.5 → predict class 0
#
#
# Example:
#       x_train = [1, 2, 3, 4, 5]   (input feature)
#       y_train = [0, 0, 0, 1, 1]   (binary labels)
#       --> The model learns to separate 0s from 1s and predicts
#           the class for any new input value
# =============================================================================


import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        return -np.mean(
            y_true * np.log(y_pred + 1e-9) +
            (1 - y_true) * np.log(1 - y_pred + 1e-9)
        )

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1] if x.ndim > 1 else 1)  # supports multi-feature
        for epoch in range(self.epochs):
            linear_model = self.w * x + self.b
            predictions = self.sigmoid(linear_model)

            dw = np.mean((predictions - y) * x)
            db = np.mean(predictions - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 100 == 0:
                loss = self.compute_loss(y, predictions)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, x):
        if self.w is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.sigmoid(self.w * x + self.b)

    def predict(self, x):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(x)]


if __name__ == '__main__':
    x_train = np.array([1, 2, 3, 4, 5])
    y_train = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(x_train, y_train)

    x_test = np.array([1, 2, 3, 4, 5, 6])
    print("\nProbabilities:", model.predict_proba(x_test))
    print("Predicted Classes:", model.predict(x_test))
        
        '''
    },


    "Decision Boundary Visualisation": {
        "description": "How the learned weights define the linear decision boundary",
        "runnable": True,
        "code": '''
"""
================================================================================
DECISION BOUNDARY — GEOMETRIC UNDERSTANDING
================================================================================

The decision boundary of logistic regression is the hyperplane where z = 0:

    w₀ + w₁x₁ + w₂x₂ = 0

We train models on three different datasets to show:
    1. Linearly separable  → logistic regression works perfectly
    2. NOT separable (XOR) → logistic regression fails (linear limit)
    3. Effect of regularisation on the boundary

================================================================================
"""

import numpy as np

np.random.seed(42)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(X, y, lr=0.5, n_iter=2000, lambda_=0.0):
    X_b = np.c_[np.ones(len(X)), X]
    w = np.zeros(X_b.shape[1])
    n = len(X)
    for _ in range(n_iter):
        y_hat = sigmoid(X_b @ w)
        grad  = (1 / n) * X_b.T @ (y_hat - y)
        if lambda_ > 0:
            reg = (lambda_ / n) * w
            reg[0] = 0
            grad += reg
        w -= lr * grad
    return w

def evaluate(X, y, w):
    X_b = np.c_[np.ones(len(X)), X]
    preds = (sigmoid(X_b @ w) >= 0.5).astype(int)
    return np.mean(preds == y)


# =============================================================================
# DATASET 1: Linearly Separable
# =============================================================================

print("=" * 65)
print("  EXPERIMENT 1: LINEARLY SEPARABLE DATA")
print("=" * 65)

X_lin = np.random.randn(200, 2)
y_lin = (X_lin[:, 0] + X_lin[:, 1] > 0).astype(float)

w_lin = train_logistic(X_lin, y_lin)
acc_lin = evaluate(X_lin, y_lin, w_lin)

print(f"""
  True boundary:    x₁ + x₂ = 0   (w₁ = w₂ = 1, bias = 0)
  Learned weights:  w₀={w_lin[0]:.3f}, w₁={w_lin[1]:.3f}, w₂={w_lin[2]:.3f}
  Training accuracy: {acc_lin:.4f}

  The learned decision boundary: {w_lin[0]:.3f} + {w_lin[1]:.3f}x₁ + {w_lin[2]:.3f}x₂ = 0
  Normalised:  x₁ + {w_lin[2]/w_lin[1]:.3f}x₂ = {-w_lin[0]/w_lin[1]:.3f}
               ≈ x₁ + x₂ = 0  ← correctly recovered! ✓

  Weight vector W = [{w_lin[1]:.3f}, {w_lin[2]:.3f}] is perpendicular to the
  decision boundary and points toward the class 1 region.
""")


# =============================================================================
# DATASET 2: XOR — Not Linearly Separable
# =============================================================================

print("=" * 65)
print("  EXPERIMENT 2: XOR — NOT LINEARLY SEPARABLE")
print("=" * 65)

n_xor = 200
X_xor = np.random.randn(n_xor, 2)
y_xor = ((X_xor[:, 0] > 0) ^ (X_xor[:, 1] > 0)).astype(float)   # XOR

w_xor = train_logistic(X_xor, y_xor, n_iter=3000)
acc_xor = evaluate(X_xor, y_xor, w_xor)

# Random baseline for comparison
baseline = max(y_xor.mean(), 1 - y_xor.mean())

print(f"""
  XOR classification:
    Class 1: x₁>0 XOR x₂>0 (different signs)
    Class 0: same signs

  Learned weights:   w₀={w_xor[0]:.3f}, w₁={w_xor[1]:.3f}, w₂={w_xor[2]:.3f}
  Training accuracy: {acc_xor:.4f}
  Random baseline:   {baseline:.4f}

  {"✗ As expected: logistic regression FAILS on XOR." if acc_xor < 0.6 else "Unexpected result."}
  Accuracy ≈ 50% = random — no linear boundary separates XOR classes.

  Fix: add a quadratic feature x₁×x₂ as a new column.
  This is Polynomial Feature Expansion — a form of feature engineering.
""")


# =============================================================================
# FIX XOR WITH FEATURE ENGINEERING
# =============================================================================

print("=" * 65)
print("  FIX: POLYNOMIAL FEATURE EXPANSION FOR XOR")
print("=" * 65)

# Add x₁×x₂ as a new feature — now the classes become linearly separable!
X_xor_poly = np.column_stack([X_xor, X_xor[:, 0] * X_xor[:, 1]])

w_xor_poly = train_logistic(X_xor_poly, y_xor, n_iter=3000)
acc_xor_poly = evaluate(X_xor_poly, y_xor, w_xor_poly)

print(f"""
  Added feature: x₃ = x₁ × x₂ (interaction term)

  In the new 3D feature space [x₁, x₂, x₁x₂]:
    Class 1 (XOR): x₁x₂ < 0  (different signs → negative product)
    Class 0 (XOR): x₁x₂ > 0  (same signs → positive product)

  These ARE linearly separable in 3D!

  Learned weights: w₀={w_xor_poly[0]:.3f}, w₁={w_xor_poly[1]:.3f},
                   w₂={w_xor_poly[2]:.3f}, w₃={w_xor_poly[3]:.3f} (x₁x₂ feature)

  Training accuracy: {acc_xor_poly:.4f}  ← massively improved

  The w₃ weight on x₁x₂ is large and negative: when x₁x₂ < 0 (class 1),
  this drives z to be positive (class 1 region). Elegant.

  Key insight: logistic regression is linear in feature space, but we can
  engineer non-linear features. Neural networks automate this — they learn
  the feature transformations instead of us hand-crafting them.
""")


# =============================================================================
# REGULARISATION EFFECT ON DECISION BOUNDARY
# =============================================================================

print("=" * 65)
print("  REGULARISATION: EFFECT ON LEARNED WEIGHTS")
print("=" * 65)

# Generate near-separable data with some noise
X_reg = np.random.randn(100, 2)
y_reg = (X_reg[:, 0] + X_reg[:, 1] + np.random.randn(100) * 0.2 > 0).astype(float)

print(f"\\n  {'λ (regularisation)':<24} | {'w₀':>8} | {'w₁':>8} | {'w₂':>8} | "
      f"{'‖w‖':>8} | {'Train Acc':>10}")
print(f"  {'-'*24}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

for lam in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
    w = train_logistic(X_reg, y_reg, lambda_=lam, n_iter=3000)
    acc = evaluate(X_reg, y_reg, w)
    norm = np.linalg.norm(w[1:])
    print(f"  {lam:<24.3f} | {w[0]:>8.4f} | {w[1]:>8.4f} | {w[2]:>8.4f} | "
          f"{norm:>8.4f} | {acc:>10.4f}")

print(f"""
  Observations:
    λ = 0.0    → large weights, model overconfident, may overfit
    λ = 0.1–1  → moderate shrinkage, good generalisation (typical sweet spot)
    λ = 100    → weights near zero, model predicts ~50% everywhere (underfit)

  In sklearn:  C = 1/λ   (inverse convention)
    C = 100 ≈ no regularisation    (same as λ = 0.01)
    C = 1   = default               (same as λ = 1.0)
    C = 0.01 = heavy regularisation (same as λ = 100)
""")
''',
    },

    "Probabilistic Interpretation & BCE": {
        "description": "MLE derivation of cross-entropy, comparison with MSE, and calibration",
        "runnable": True,
        "code": '''
"""
================================================================================
PROBABILISTIC INTERPRETATION — BCE vs MSE, MLE, CALIBRATION
================================================================================

Binary cross-entropy is not an arbitrary loss function choice.
It is the Maximum Likelihood Estimator under the Bernoulli distribution.

This script demonstrates:
    1. Why MSE fails for classification (non-convex with sigmoid)
    2. BCE vs MSE: penalising confident wrong predictions
    3. Model calibration — are the probabilities meaningful?

================================================================================
"""

import numpy as np
import math

np.random.seed(42)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# =============================================================================
# PART 1: BCE vs MSE — Why BCE is the Right Choice
# =============================================================================

print("=" * 65)
print("  PART 1: BCE vs MSE FOR CLASSIFICATION")
print("=" * 65)

print("""
  For a TRUE label y = 1, we compare how BCE and MSE penalise
  predictions ŷ (predicted probability of class 1).

  BCE(ŷ, y=1) = -log(ŷ)
  MSE(ŷ, y=1) = (1 - ŷ)²
""")

print(f"  {'ŷ (predicted P(class 1))':<28} | {'BCE loss':>10} | {'MSE loss':>10} | Note")
print(f"  {'-'*28}-+-{'-'*10}-+-{'-'*10}-+-{'-'*25}")

cases = [
    (0.99, "Very confident, correct"),
    (0.80, "Confident, correct"),
    (0.60, "Mildly confident, correct"),
    (0.50, "Maximally uncertain"),
    (0.40, "Leaning wrong"),
    (0.20, "Confident, WRONG"),
    (0.01, "Very confident, WRONG"),
]

for y_hat, note in cases:
    bce = -math.log(max(y_hat, 1e-12))
    mse = (1 - y_hat) ** 2
    print(f"  {y_hat:<28.2f} | {bce:>10.4f} | {mse:>10.4f} | {note}")

print(f"""
  Key insight:
    At ŷ = 0.01 (very confident, WRONG):
      BCE = 4.6  — SCREAMS about this mistake
      MSE = 0.98 — barely registers

    At ŷ = 0.99 (very confident, correct):
      BCE ≈ 0    — essentially no penalty
      MSE ≈ 0    — same

  BCE punishes confident mistakes exponentially more.
  MSE treats a confident wrong prediction almost the same as a moderate one.
  For classification, confident mistakes are catastrophic → use BCE.
""")


# =============================================================================
# PART 2: MLE DERIVATION — WHY BCE FOLLOWS FROM PROBABILITY THEORY
# =============================================================================

print("=" * 65)
print("  PART 2: MLE DERIVATION OF BINARY CROSS-ENTROPY")
print("=" * 65)

print("""
  Assumption: y|x ~ Bernoulli(σ(w·x + b))
    P(y=1 | x; w) = ŷ
    P(y=0 | x; w) = 1 - ŷ

  Unified form: P(y | x; w) = ŷʸ · (1-ŷ)^(1-y)

  Log-likelihood over n examples:
    ℓ(w) = Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

  Maximising ℓ(w) = Minimising -ℓ(w)/n = Minimising BCE

  This is not an approximation or a heuristic — it is the exact MLE
  for a Bernoulli-distributed binary target.

  Noise model → Loss function (the pattern):
  ────────────────────────────────────────────────────────────────
  Gaussian noise  →  MSE         (Linear Regression)
  Bernoulli noise →  BCE         (Logistic Regression)
  Poisson count   →  Poisson NLL (Poisson Regression)
  Laplace noise   →  MAE         (Least Absolute Deviations)
  ────────────────────────────────────────────────────────────────
  Choose your noise model → get your loss function for free.
""")


# =============================================================================
# PART 3: TRAINING COMPARISON — BCE vs MSE on CLASSIFICATION
# =============================================================================

print("=" * 65)
print("  PART 3: BCE vs MSE TRAINING ON A CLASSIFICATION TASK")
print("=" * 65)

X = np.random.randn(300, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float)
X_b = np.c_[np.ones(len(X)), X]
n = len(X)

def train_with_loss(loss_type, n_iter=2000, lr=0.3):
    w = np.zeros(3)
    losses = []
    for _ in range(n_iter):
        z     = X_b @ w
        y_hat = sigmoid(z)
        if loss_type == "bce":
            eps   = 1e-12
            yh    = np.clip(y_hat, eps, 1-eps)
            loss  = -np.mean(y * np.log(yh) + (1-y) * np.log(1-yh))
            grad  = (1/n) * X_b.T @ (y_hat - y)
        else:  # mse
            loss  = np.mean((y_hat - y)**2)
            # MSE gradient through sigmoid: dL/dw = (2/n) Xᵀ[(ŷ-y)·ŷ·(1-ŷ)]
            grad  = (2/n) * X_b.T @ ((y_hat - y) * y_hat * (1 - y_hat))
        losses.append(loss)
        w -= lr * grad
    acc = np.mean((sigmoid(X_b @ w) >= 0.5).astype(int) == y)
    return w, losses, acc

w_bce, losses_bce, acc_bce = train_with_loss("bce")
w_mse, losses_mse, acc_mse = train_with_loss("mse")

print(f"""
  Training on linearly separable 2D data:
  (n=300, lr=0.3, 2000 iterations)

  ────────────────────────────────────────────────────────────
  Metric                    BCE             MSE
  ────────────────────────────────────────────────────────────
  Final training accuracy   {acc_bce:.4f}          {acc_mse:.4f}
  Initial loss              {losses_bce[0]:.4f}          {losses_mse[0]:.4f}
  Final loss                {losses_bce[-1]:.4f}          {losses_mse[-1]:.4f}
  Loss at iter 100          {losses_bce[100]:.4f}          {losses_mse[100]:.4f}
  Loss at iter 500          {losses_bce[500]:.4f}          {losses_mse[500]:.4f}
  ────────────────────────────────────────────────────────────

  BCE converges faster and more reliably because:
    1. The gradient ∇L = (1/n)Xᵀ(ŷ-y) doesn't vanish (no σ(1-σ) factor)
    2. The loss surface is convex under BCE
    3. MSE gradient has an extra ŷ(1-ŷ) term which approaches 0
       when the model is confident — gradient vanishes, learning stalls
""")


# =============================================================================
# PART 4: CALIBRATION — ARE THE PROBABILITIES TRUSTWORTHY?
# =============================================================================

print("=" * 65)
print("  PART 4: PROBABILITY CALIBRATION")
print("=" * 65)

print("""
  A well-calibrated model: when it says P=0.8, it's right 80% of the time.
  Logistic regression is inherently well-calibrated — a direct consequence
  of MLE training with the correct Bernoulli likelihood.

  Test: group predictions into bins, check empirical accuracy per bin.
""")

# Train a model
X_cal = np.random.randn(1000, 2)
y_cal = (X_cal[:, 0] + X_cal[:, 1] + np.random.randn(1000) * 0.5 > 0).astype(float)
X_cal_b = np.c_[np.ones(1000), X_cal]
w_cal, _, _ = train_with_loss("bce")
# Retrain on calibration data
w_cal2 = np.zeros(3)
for _ in range(3000):
    z = X_cal_b @ w_cal2
    yh = sigmoid(z)
    grad = (1/1000) * X_cal_b.T @ (yh - y_cal)
    w_cal2 -= 0.3 * grad

probs = sigmoid(X_cal_b @ w_cal2)

print(f"  {'Predicted prob bin':<22} | {'# examples':>12} | {'Actual accuracy':>16}")
print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*16}")

bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for lo, hi in bins:
    mask = (probs >= lo) & (probs < hi)
    if mask.sum() > 0:
        actual_acc = y_cal[mask].mean()
        mid = (lo + hi) / 2
        calibration_gap = abs(actual_acc - mid)
        marker = "✓" if calibration_gap < 0.1 else "⚠"
        print(f"  [{lo:.1f} – {hi:.1f})              | {mask.sum():>12} | "
              f"{actual_acc:>14.3f}  {marker}")

print(f"""
  A well-calibrated model's actual accuracy should match the predicted
  probability in each bin. Small gaps → trustworthy probabilities.

  This calibration property is critical in medicine, finance, and any
  domain where you need to act on the predicted probability, not just
  the binary classification.
""")
''',
    },

    "Multi-Class: Softmax Regression": {
        "description": "Softmax from scratch, categorical cross-entropy, and comparison with One-vs-Rest",
        "runnable": True,
        "code": '''
"""
================================================================================
MULTI-CLASS LOGISTIC REGRESSION: SOFTMAX FROM SCRATCH
================================================================================

For k > 2 classes, the natural generalisation of sigmoid is softmax:

    softmax(z)ⱼ = exp(zⱼ) / Σₖ exp(zₖ)

The loss is categorical cross-entropy:

    L = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)

where yᵢⱼ = 1 if example i belongs to class j (one-hot encoding).

The gradient is as elegant as binary logistic regression:

    ∇L_wⱼ = (1/n) Xᵀ (ŷⱼ - yⱼ)     for each class j

================================================================================
"""

import numpy as np

np.random.seed(42)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def softmax(Z):
    """
    Numerically stable softmax for a batch of examples.

    Input:  Z of shape (n, k)  — n examples, k class scores each
    Output: P of shape (n, k)  — n examples, k probabilities (rows sum to 1)

    Numeric stability trick: subtract max per row before exponentiating.
    exp(z - max) / Σ exp(z - max) = exp(z) / Σ exp(z)  (identical value)
    but prevents overflow in exp() for large z values.
    """
    Z_shifted = Z - Z.max(axis=1, keepdims=True)   # subtract row max
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def categorical_cross_entropy(Y_onehot, Y_hat):
    """
    Categorical cross-entropy loss.

    L = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)

    Since Y is one-hot, only the true class term contributes.
    Simplified: L = -(1/n) Σᵢ log(ŷᵢ[true_class_i])
    """
    eps = 1e-12
    return -np.mean(np.sum(Y_onehot * np.log(Y_hat + eps), axis=1))


def one_hot(y, k):
    """Convert integer labels to one-hot matrix of shape (n, k)."""
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1
    return Y


# =============================================================================
# SOFTMAX REGRESSION CLASS
# =============================================================================

class SoftmaxRegression:
    """
    Multi-class logistic regression using softmax.

    Architecture:
        W: shape (p+1, k)   — one weight vector per class
        For each example x: z = W.T @ x  → softmax(z) → k probabilities

    Training:
        Gradient: ∇L_W = (1/n) Xᵀ (P - Y)    where P = softmax(XW), Y = one-hot
        Update:   W := W - α · ∇L_W
    """

    def __init__(self, learning_rate=0.5, n_iterations=1000):
        self.lr     = learning_rate
        self.n_iter = n_iterations
        self.W      = None

    def fit(self, X, y, verbose=True):
        n, p = X.shape
        k = len(np.unique(y))
        X_b = np.c_[np.ones(n), X]        # (n, p+1)
        Y   = one_hot(y, k)                # (n, k)

        # Initialise weight matrix: (p+1) × k
        self.W = np.random.randn(p + 1, k) * 0.01

        for i in range(self.n_iter):
            # Forward pass
            Z = X_b @ self.W               # (n, k) raw scores
            P = softmax(Z)                 # (n, k) probabilities

            # Loss
            loss = categorical_cross_entropy(Y, P)

            # Gradient: (1/n) Xᵀ(P - Y), shape (p+1, k)
            grad = (1 / n) * X_b.T @ (P - Y)

            # Update
            self.W -= self.lr * grad

            if verbose and i % 200 == 0:
                preds = np.argmax(P, axis=1)
                acc = np.mean(preds == y)
                print(f"  Iter {i:>4} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    def predict_proba(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return softmax(X_b @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# =============================================================================
# DATASET: 3-class problem (Iris-inspired, 2 features for clarity)
# =============================================================================

print("=" * 65)
print("  SOFTMAX REGRESSION FROM SCRATCH — 3-CLASS PROBLEM")
print("=" * 65)

# Generate 3 separable Gaussian clusters
n_per_class = 150
centers = [(-3, 0), (3, 0), (0, 4)]
X_list, y_list = [], []
for j, (cx, cy) in enumerate(centers):
    X_list.append(np.random.randn(n_per_class, 2) + np.array([cx, cy]))
    y_list.append(np.full(n_per_class, j))

X_multi = np.vstack(X_list)
y_multi = np.concatenate(y_list)

# Shuffle
idx = np.random.permutation(len(X_multi))
X_multi, y_multi = X_multi[idx], y_multi[idx]

print(f"""
  Dataset: {len(X_multi)} examples, 2 features, 3 classes
  Class 0 centre: (-3, 0)    Class 1 centre: (3, 0)    Class 2 centre: (0, 4)
""")

model = SoftmaxRegression(learning_rate=0.5, n_iterations=1000)
model.fit(X_multi, y_multi)

final_acc = np.mean(model.predict(X_multi) == y_multi)
print(f"\\n  Final accuracy: {final_acc:.4f}")


# =============================================================================
# STEP-BY-STEP SOFTMAX COMPUTATION
# =============================================================================

print("\\n" + "=" * 65)
print("  STEP-BY-STEP: SOFTMAX COMPUTATION")
print("=" * 65)

k = 3
z = np.array([2.0, 1.0, 0.5])   # raw scores for 3 classes

exp_z      = np.exp(z)
sum_exp_z  = exp_z.sum()
probs      = exp_z / sum_exp_z

print(f"""
  Raw scores z = {z}

  Step 1 — Exponentiate:
    exp(2.0) = {exp_z[0]:.4f}
    exp(1.0) = {exp_z[1]:.4f}
    exp(0.5) = {exp_z[2]:.4f}

  Step 2 — Sum: {exp_z[0]:.4f} + {exp_z[1]:.4f} + {exp_z[2]:.4f} = {sum_exp_z:.4f}

  Step 3 — Normalise:
    P(class 0) = {exp_z[0]:.4f} / {sum_exp_z:.4f} = {probs[0]:.4f}
    P(class 1) = {exp_z[1]:.4f} / {sum_exp_z:.4f} = {probs[1]:.4f}
    P(class 2) = {exp_z[2]:.4f} / {sum_exp_z:.4f} = {probs[2]:.4f}
    ────────────────────────
    Sum         =                              {probs.sum():.4f} ✓
""")

# Show categorical cross-entropy for this example
y_true_onehot = np.array([1, 0, 0])   # true class is 0
loss_example = -np.sum(y_true_onehot * np.log(probs + 1e-12))
print(f"  If true class = 0:  CE loss = -log({probs[0]:.4f}) = {loss_example:.4f}")
print(f"  If true class = 1:  CE loss = -log({probs[1]:.4f}) = {-np.log(probs[1]):.4f}")
print(f"  If true class = 2:  CE loss = -log({probs[2]:.4f}) = {-np.log(probs[2]):.4f}")
print(f"""
  Cross-entropy = -log(probability assigned to the TRUE class).
  High probability on the correct class → small loss.
  Low probability on the correct class → large loss.
""")


# =============================================================================
# ONE-VS-REST vs SOFTMAX COMPARISON
# =============================================================================

print("=" * 65)
print("  ONE-VS-REST vs SOFTMAX: SIDE-BY-SIDE")
print("=" * 65)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

ovr = LogisticRegression(multi_class="ovr", solver="lbfgs", C=1.0, max_iter=1000)
softmax_sk = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=1000)

ovr.fit(X_tr, y_tr)
softmax_sk.fit(X_tr, y_tr)

acc_ovr = np.mean(ovr.predict(X_te) == y_te)
acc_softmax = np.mean(softmax_sk.predict(X_te) == y_te)

print(f"""
  Train/test split: 80/20

  Method                     Test Accuracy
  ────────────────────────────────────────────
  One-vs-Rest (OvR)           {acc_ovr:.4f}
  Softmax (Multinomial)       {acc_softmax:.4f}
  Our scratch implementation  {final_acc:.4f} (train acc)
  ────────────────────────────────────────────

  OvR trains k independent binary classifiers — fast, simple.
  Softmax trains one joint model — probabilities sum to 1, often better.

  Key difference in probability outputs:
""")

# Show OvR probabilities (don't sum to 1)
test_pt = X_te[[0]]
ovr_prob = ovr.predict_proba(test_pt)[0]
sm_prob  = softmax_sk.predict_proba(test_pt)[0]

print(f"  For a single test point:")
print(f"    OvR probabilities:     {ovr_prob.round(4)}  (sum = {ovr_prob.sum():.4f})")
print(f"    Softmax probabilities: {sm_prob.round(4)}   (sum = {sm_prob.sum():.4f})")
print(f"""
  OvR sum ≠ 1: the k classifiers are independent, no normalisation constraint.
  Softmax sum = 1: enforced by the softmax normalisation.

  For tasks requiring a valid probability distribution over classes
  (e.g., the output of a neural network), softmax is correct.
""")
''',
    },

    # ── Trigger: Logistic Regression ─────────────────────────────────
    "▶ Run: Logistic Regression": {
        "description": (
            "Runs 00_LogisticRegression.py from the Implementation folder. "
            "Logistic Regression for Classification "
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Classification"
    / "00_LogisticRegression.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Implementation/Supervised Model/Classification/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },

#

}

VISUAL_HTML = ""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Standalone demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np

    print("\n" + "=" * 65)
    print("  LOGISTIC REGRESSION: LINEAR CLASSIFICATION VIA PROBABILITY")
    print("=" * 65)
    print("""
  This script demonstrates logistic regression from the ground up.

  Key Concepts:
    • Sigmoid squashes z = w·x + b into a probability ∈ (0, 1)
    • BCE = MLE under Bernoulli distribution (not an arbitrary choice)
    • Gradient ∇L = (1/n)Xᵀ(ŷ-y) — identical form to linear regression
    • Decision boundary is always linear: w·x + b = 0
    • Regularisation prevents overconfident, saturated weights
    • Softmax extends to k > 2 classes with probabilities summing to 1
    """)

    np.random.seed(42)

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ─── Tiny dataset: 4 points ──────────────────────────────────────────────
    X = np.array([[2.0, 1.0],
                  [1.0, 2.0],
                  [-1.0, -1.0],
                  [-2.0, -2.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])

    X_b = np.c_[np.ones(4), X]
    w = np.zeros(3)
    lr = 0.1
    n = len(y)

    print("=" * 65)
    print("  DATASET: 4 points, true boundary ≈ x₁ + x₂ = 0")
    print("=" * 65)
    print(f"\n  {'x₁':>6}  {'x₂':>6}  {'y':>4}")
    print(f"  {'------':>6}  {'------':>6}  {'----':>4}")
    for xi, yi in zip(X, y):
        print(f"  {xi[0]:>6.1f}  {xi[1]:>6.1f}  {int(yi):>4}")

    # ─── Manual training loop ────────────────────────────────────────────────
    print(f"\n  Training (100 iterations, α={lr})...")
    for epoch in range(100):
        z = X_b @ w
        y_hat = sigmoid(z)
        grad = (1 / n) * X_b.T @ (y_hat - y)
        w -= lr * grad

    # ─── Results ─────────────────────────────────────────────────────────────
    print(f"\n  Final weights: w₀={w[0]:.4f}, w₁={w[1]:.4f}, w₂={w[2]:.4f}")

    print(f"\n  Predictions:")
    print(f"  {'x':>12}  {'y_true':>8}  {'P(class 1)':>12}  {'predicted':>10}  {'correct?':>8}")
    print(f"  {'------------':>12}  {'--------':>8}  {'------------':>12}  {'----------':>10}  {'--------':>8}")
    all_correct = True
    for xi, yi, xbi in zip(X, y, X_b):
        prob = sigmoid(xbi @ w)
        pred = int(prob >= 0.5)
        correct = "✓" if pred == int(yi) else "✗"
        if pred != int(yi):
            all_correct = False
        print(f"  {str(xi.tolist()):>12}  {int(yi):>8}  {prob:>12.4f}  {pred:>10}  {correct:>8}")

    eps = 1e-12
    y_hat_final = sigmoid(X_b @ w)
    bce = -np.mean(y * np.log(np.clip(y_hat_final, eps, 1-eps)) +
                   (1-y) * np.log(np.clip(1-y_hat_final, eps, 1-eps)))

    print(f"\n  BCE loss:  {bce:.4f}")
    print(f"  {'All correct ✓' if all_correct else 'Some errors ✗'}")

    # ─── Decision boundary ───────────────────────────────────────────────────
    print(f"\n  Decision boundary: {w[0]:.4f} + {w[1]:.4f}x₁ + {w[2]:.4f}x₂ = 0")
    if abs(w[1]) > 0.01:
        print(f"  Normalised: x₁ + {w[2]/w[1]:.4f}x₂ = {-w[0]/w[1]:.4f}")
        print(f"  Expected:   x₁ + 1.0000x₂ = 0.0000  (true boundary)")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1. Logistic regression = ERM with linear hypothesis + sigmoid + BCE loss
  2. BCE is MLE under Bernoulli — not an arbitrary choice
  3. Gradient ∇L = (1/n)Xᵀ(ŷ-y) is identical in form to linear regression
  4. Decision boundary is always linear: the model carves space in half
  5. Fails on XOR and non-linear problems without feature engineering
  6. Softmax extends to k > 2 classes; probabilities sum to 1
  7. Every output neuron in a neural network is logistic regression
  8. Logistic regression is the direct predecessor of neural network classifiers
    """)


# # ─────────────────────────────────────────────────────────────────────────────
# # CONTENT EXPORT
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_content():
#     """Return all content for this topic module."""
#     return {
#         "theory":               THEORY,
#         "theory_raw":           THEORY,
#         "operations":           OPERATIONS,
#         "interactive_components": [],
#     }

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
        main_script = None

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
    visual_height = 400
    try:
        from Supervised_Learning.visuals.logistic_regression_visual import (   # ← match your exact folder casing
            LOR_VISUAL_HTML,
            LOR_VISUAL_HEIGHT,
        )
        visual_html   = LOR_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = LOR_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[02_logistic_regression.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None, # COMPLEXITY,
        "operations":    OPERATIONS,
    }