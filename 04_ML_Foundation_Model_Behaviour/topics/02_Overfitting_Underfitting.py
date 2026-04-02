"""
Overfitting, Underfitting & the Bias-Variance Tradeoff
=======================================================

Two of the most important failure modes in machine learning, and the
fundamental tension that governs how well a model can learn from data.

"""

import textwrap
import re

TOPIC_NAME = "Overfitting, Underfitting & Bias-Variance Tradeoff"
DISPLAY_NAME = "02 · Overfitting & Underfitting"
ICON = "📉"
SUBTITLE = "The Core Tension in Machine Learning"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Problem: Generalisation

The entire goal of machine learning is *generalisation* — training on a
finite set of examples and then making accurate predictions on **new,
unseen data**. This is much harder than it sounds.

A model that perfectly memorises the training data is useless if it fails on
anything new. A model that is too simple never captures the real patterns at
all. The challenge is finding the sweet spot between these two failure modes.

There are only two ways a model can go wrong:

    •   Underfitting  — model is too simple to capture the true pattern
    •   Overfitting   — model is too complex and memorises noise, not signal


──────────────────────────────────────────────────────────────────────────────
### What is Underfitting?

Underfitting occurs when a model is **not complex enough** to represent the
underlying structure of the data. It performs poorly on both training data
and new data — it hasn't even learned the training examples properly.

**Symptoms:**
    - High training error
    - High validation/test error
    - Training and validation errors are close to each other (both bad)
    - The model's predictions look "flat" or "too smooth"

**Causes:**
    - Model is too simple (e.g., linear model on non-linear data)
    - Too few features
    - Too much regularisation (penalising complexity too aggressively)
    - Training stopped too early (under-trained)
    - Learning rate is too high and the model overshoots the minimum


    Diagram 1 — Underfitting Visualised (Polynomial Regression):

    TRUE RELATIONSHIP: a curved parabola
    FITTED MODEL: a straight line (degree-1 polynomial)

                                    y
                                    |              ● True data point
                                    |         ●
                                    |    ●          ●
                                    | ●                 ●
                                    |●                      ●
                                    ●────────────────────────── ← fitted line
                                   /|
                                  / |                            x
                    ─────────────────────────────────────────────

    The straight line CANNOT capture the curve, no matter how much
    data you add. High bias — wrong structural assumption.

    Error Analysis:
    ┌─────────────────────────────────────────────────────────┐
    │  Training Error:    HIGH   (~35%)                       │
    │  Validation Error:  HIGH   (~37%)                       │
    │  Gap:               SMALL  (both are bad)               │
    │  Diagnosis:         UNDERFIT — increase model capacity  │
    └─────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### What is Overfitting?

Overfitting occurs when a model is **too complex** — it learns the training
data so well that it captures random noise and idiosyncrasies as if they were
real patterns. It performs excellently on training data but poorly on new data.

Think of a student who memorises every past exam paper word-for-word instead
of understanding the underlying concepts. They ace practice exams but fail
on any new question phrased differently.

**Symptoms:**
    - Very low training error
    - Much higher validation/test error
    - Large gap between training and validation performance
    - Model output is "wiggly" or overly complex

**Causes:**
    - Model has too many parameters relative to the amount of training data
    - Training for too many epochs
    - Too little regularisation
    - Too few training examples
    - Data has a lot of noise that the model has learned as signal


    Diagram 2 — Overfitting Visualised (Polynomial Regression):

    TRUE RELATIONSHIP: a smooth curve
    FITTED MODEL: a degree-9 polynomial — "memorises" every point

                                    y
                                    |    ╭─╮       ╭╮
                                    |   ╱   ╲  ╭──╯  ╲  ╭─
                                    | ╱       ╲╯        ╲╯
                                    |╱                        x
                                    |
                    ─────────────────────────────────────────────

    Every training point sits exactly on the curve — but the curve
    oscillates wildly between them. On new test points the predictions
    are catastrophically wrong.

    Error Analysis:
    ┌─────────────────────────────────────────────────────────┐
    │  Training Error:    VERY LOW  (~1%)                     │
    │  Validation Error:  HIGH      (~40%)                    │
    │  Gap:               LARGE     (this gap IS overfitting) │
    │  Diagnosis:         OVERFIT — reduce capacity or add    │
    │                     regularisation / more data          │
    └─────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### The Just-Right Model (Goldilocks Zone)

    Diagram 3 — Three Models Side by Side:

    DATA: points sampled from a quadratic + noise

    UNDERFIT (degree 1)   GOOD FIT (degree 2)   OVERFIT (degree 9)
    ──────────────────    ───────────────────    ──────────────────

          ●   ●                 ●   ●                 ●   ●
        ●                     ●                     ●
      ●         ●           ●         ●           ●         ●
    ──────────────     ──╮──────────╭──     ──╮──────────╭──
                        ╰──────────╯              ╰──╯╭─╯╰╮╯

    Line misses curve.    Matches curve well.    Wiggles through noise.
    High bias.            Low bias + low var.    Low bias, high variance.
    Can't generalise.     Generalisees well.    Can't generalise.


──────────────────────────────────────────────────────────────────────────────
### The Bias-Variance Tradeoff

This is the theoretical framework that formally explains overfitting and
underfitting. Every model's generalisation error can be decomposed into
three components:

                Total Error = Bias² + Variance + Irreducible Noise


**Bias** — error from wrong assumptions in the learning algorithm.

    A HIGH-BIAS model makes strong assumptions about the data structure
    (e.g., "the relationship is linear"). If those assumptions are wrong,
    the model systematically predicts incorrectly — even with infinite data.

    → Underfitting is a high-bias problem.

**Variance** — error from sensitivity to fluctuations in the training data.

    A HIGH-VARIANCE model is extremely sensitive to the specific training
    examples it sees. Train it on a slightly different dataset and you get
    a completely different model. It chases noise.

    → Overfitting is a high-variance problem.

**Irreducible Noise** — the inherent randomness in the data.

    No model can predict perfectly because the real world has noise
    (measurement error, missing features, randomness). This is a floor
    that no algorithm can beat. We can only minimise bias and variance.


    Diagram 4 — The Bias-Variance Bullseye:

    Each dot = one model trained on a DIFFERENT sample of the data.
    The bullseye (centre) = the true target value we want to predict.

    HIGH BIAS, LOW VARIANCE        LOW BIAS, LOW VARIANCE
    (Underfitting)                 (IDEAL)

         ┌────────────┐               ┌─────────────┐
         │            │               │             │
         │  ×  ×  ×   │               │      ●      │
         │  ×  ×  ×   │               │    ● ● ●    │
         │  ×  ×  ×   │               │      ●      │
         │            │               │             │
         └────────────┘               └─────────────┘
    Consistent predictions,           Consistent predictions,
    but all WRONG (off-centre)        all CORRECT (on-target)


    LOW BIAS, HIGH VARIANCE        HIGH BIAS, HIGH VARIANCE
    (Overfitting)                  (Worst case — broken)

         ┌─────────────┐               ┌─────────────┐
         │   ×         │               │ ×           │
         │       ×     │               │        ×    │
         │   ×    ×    │               │  ×      ×   │
         │       ×     │               │     ×       │
         │  ×          │               │          ×  │
         └─────────────┘               └─────────────┘
    Predictions scattered            Scattered AND off-centre.
    around the centre but            Neither accurate nor stable.
    each one is different.


    Diagram 5 — Bias-Variance as Model Complexity Increases:

    Error
      │
      │  ╲                                    ← Validation Error
      │   ╲                          ╱
      │    ╲                        ╱ ← Overfitting zone
      │     ╲                      ╱
      │      ╲                    ╱
      │       ╲──────────────────╱   ← Training Error
      │        ╲               ╱
      │         ╲            ╱
      │          ╲──────────╯   ← SWEET SPOT (minimum val. error)
      │
      └──────────────────────────────────────── Model Complexity →
            Underfit     Good fit     Overfit


    As model complexity increases:
    ┌──────────────┬────────────────────┬───────────────────┐
    │ Complexity   │ Bias               │ Variance          │
    ├──────────────┼────────────────────┼───────────────────┤
    │ Low          │ HIGH (rigid)       │ LOW (stable)      │
    │ Medium       │ MEDIUM             │ MEDIUM            │
    │ High         │ LOW (flexible)     │ HIGH (erratic)    │
    └──────────────┴────────────────────┴───────────────────┘

    The "tradeoff" — decreasing bias increases variance, and vice versa.
    This is a fundamental law, not a limitation of current algorithms.


──────────────────────────────────────────────────────────────────────────────
### Diagnosing Overfitting and Underfitting with Learning Curves

A *learning curve* plots training and validation error as a function of the
number of training examples. It's the single most powerful diagnostic tool.

    Diagram 6 — Learning Curves for Underfitting:

    Error
      │
      │──────────────────────────────  ← Validation error (high, flat)
      │
      │
      │──────────────────────────────  ← Training error (high, flat)
      │
      │
      └────────────────────────────── Training set size →

    BOTH curves are high and flat. Adding more data does NOT help.
    The model is too simple — fix: increase model capacity.


    Diagram 7 — Learning Curves for Overfitting:

    Error
      │
      │╲
      │ ╲
      │  ╲___________________________  ← Validation error (still high)
      │
      │
      │                     ╭─────── ← Training error (very low)
      │──────────────────────
      └────────────────────────────── Training set size →

    Large GAP between the two curves.
    Fix: add more data, add regularisation, or reduce model complexity.


    Diagram 8 — Learning Curves for a Good Model:

    Error
      │
      │╲
      │ ╲___________________________  ← Validation error (decreasing)
      │  ╲─────────────────────────
      │   ╲──────────────────────── ← Training error (increasing slightly)
      │    ╲────────────────────────
      │     ─────────────────────── ← Curves converge to a low value
      └────────────────────────────── Training set size →

    Small gap, both curves converge to a low error.
    Adding more data continues to help (curves still moving down).


──────────────────────────────────────────────────────────────────────────────
### How to Fix Overfitting

1. **Get more training data**
   The most reliable fix. More data constrains what the model can memorise.
   Alternatively, use data augmentation to artificially expand the dataset.

2. **Reduce model complexity**
   Use fewer layers, fewer neurons, lower-degree polynomials.
   Less capacity = less room to memorise noise.

3. **Regularisation (L1 / L2 / Dropout)**
   Add a penalty to the loss function that discourages large weights.
   Forces the model to learn simpler, more general patterns.
   (Covered in depth in Module 04.)

4. **Early stopping**
   Monitor validation error during training. Stop when it starts rising
   even though training error is still falling. The rising validation error
   is the exact moment overfitting begins.

5. **Feature selection / dimensionality reduction**
   Remove irrelevant or redundant features. Fewer noisy features =
   fewer things the model can overfit to.

6. **Cross-validation**
   Use k-fold cross-validation instead of a single train/val split to get
   a more reliable estimate of generalisation performance.
   (Covered in Module 04.)

7. **Ensemble methods**
   Average predictions from many models (bagging, random forests).
   Individual models overfit to different parts of the noise;
   averaging cancels out the noise.

    Fix Comparison:
    ┌─────────────────────────┬────────────────────────────────────────┐
    │ Fix                     │ What it does                           │
    ├─────────────────────────┼────────────────────────────────────────┤
    │ More data               │ Reduces variance (more signal/noise)   │
    │ Simpler model           │ Reduces variance, increases bias       │
    │ L2 regularisation       │ Shrinks weights → reduces variance     │
    │ Dropout                 │ Ensemble effect → reduces variance     │
    │ Early stopping          │ Stops before variance explodes         │
    │ Feature selection       │ Removes noisy dimensions               │
    └─────────────────────────┴────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### How to Fix Underfitting

1. **Increase model complexity**
   Add more layers, more neurons, higher-degree terms.
   Give the model the capacity to represent the true function.

2. **Add more / better features**
   If the raw features are insufficient, engineer new ones.
   Example: add x² and x³ to a linear model.

3. **Reduce regularisation**
   If the regularisation penalty is too strong it will prevent the model
   from fitting even the training data. Relax it.

4. **Train longer**
   Sometimes the model is capable but hasn't converged yet.
   More epochs with a stable learning rate will close the gap.

5. **Tune the learning rate**
   A learning rate that is too large causes oscillation and prevents
   convergence. A schedule (decay) or an adaptive optimizer (Adam) helps.

6. **Check for data problems**
   Mislabelled data, wrong feature scaling, or missing values can make
   any model appear to underfit.


──────────────────────────────────────────────────────────────────────────────
### The Double Descent Phenomenon (Modern Insight)

Classical wisdom says: "once you have more parameters than data points, the
model will catastrophically overfit." In practice, modern deep networks
routinely have billions of parameters trained on far fewer examples — and
they generalise beautifully. Why?

It turns out the bias-variance tradeoff curve has a second descent:

    Error
      │                          ┌── Interpolation Threshold
      │                          │   (# params ≈ # data points)
      │  ╲                       │
      │   ╲              ╭───────┼──────────────╮
      │    ╲            ╱       │                ╲
      │     ╲──────────╯        │                 ╲────────────
      │                         │
      └────────────────────────────────────────── Model Size →
           Classical regime      Modern regime (overparameterised)

    In the "modern" overparameterised regime, the model is so large it
    can fit the training data in MANY ways. Gradient descent naturally
    finds the SMOOTHEST solution — which turns out to generalise well.

    This is closely related to *implicit regularisation* — the optimiser
    itself (especially SGD) introduces a bias toward simpler solutions
    even without explicit regularisation penalties.

    Key insight: the classical bias-variance tradeoff still holds
    WITHIN the modern regime, but the whole curve shifts down as model
    size increases past the threshold. Bigger is not always worse.

---


Prevents Overfitting  

1. Collect More Data
    
    More data helps the model learn true underlying patterns rather than noise.

    Example:
        A neural network trained on 1,000 images may overfit.        
        Training on 100,000 images reduces overfitting.
    
    Methods:
        Data collection

        Data augmentation (images, text, audio)


2. Simpler Models

Complex models fit noise easily.

    Examples
        * Reduce number of layers in neural networks
        * Reduce polynomial degree in regression
        * Use shallow trees instead of deep trees

3. Regularization

Regularization penalizes large model parameters.

L1 Regularization

Encourages sparse models.
        
        Loss=MSE+λ∑∣w∣
        
    * Removes irrelevant features
    * Used in Lasso Regression


L2 Regularization

Penalizes large weights.

        Loss=MSE+λ∑w^2

    * Prevents extreme parameter values
    * Used in Ridge Regression


4. Dropout (Neural Networks)

Randomly disables neurons during training.

    Example:
        Layer neurons: 100
        Dropout rate: 0.5
        Only 50 neurons active each iteration
        
    Benefits
        Prevents neuron co-adaptation
        Improves generalization
    
    
5. Early Stopping

Stop training when validation error starts increasing.

Typical training behavior:
    
    Epoch        Training Loss       Validation Loss
    1                ↓                     ↓
    10               ↓                     ↓
    30               ↓                     ↑  ← stop here
    
This prevents the model from memorizing training noise.

6. Cross Validation

Split data multiple times to ensure the model generalizes.

Example:

    K-Fold Cross Validation
    
    Dataset → 5 folds

    Train: 4 folds
    Test : 1 fold
    
    Repeat 5 times
    
Average performance gives reliable results.
    

7. Data Augmentation

Artificially increase dataset size.

Examples

    Images
        Rotation
        Cropping
        Flipping
        Color jitter
    
    Text
        Synonym replacement
        Back translation


8. Pruning (Decision Trees)

Remove unnecessary branches.

    Without pruning:
        Deep tree → memorizes training data
    
    With pruning:
        Simpler tree → better generalization
        
9. Ensembling

Combine multiple models.

    Examples    
        Random Forest
        Bagging
        Boosting
    Ensembles reduce variance and overfitting.


10. Feature Selection

Remove irrelevant or noisy features.

    Methods
        L1 regularization
        Recursive feature elimination
        Correlation filtering


    +----------------------+----------------------------------+
    | Technique            | How it Prevents Overfitting      |
    +----------------------+----------------------------------+
    | More Data            | Improves generalization          |
    | Regularization       | Penalizes large parameters       |
    | Dropout              | Prevents neuron co-adaptation    |
    | Early Stopping       | Stops memorization               |
    | Cross Validation     | Ensures stable performance       |
    | Data Augmentation    | Expands dataset                  |
    | Simpler Models       | Reduces model variance           |
    | Pruning              | Simplifies decision trees        |
    | Ensembling           | Reduces variance                 |
    | Feature Selection    | Removes noisy inputs             |
    +----------------------+----------------------------------+

---

Prevents Underfitting  

Underfitting occurs when a model is too simple to capture the underlying patterns in the data.
It performs poorly on both training data and test data.

Typical symptom:

    Training Error   → High
    Validation Error → High
    
Ways to Prevent Underfitting

1. Increase Model Complexity

If the model is too simple, it cannot learn the data structure.

Examples

    Model	                    Fix
    Linear regression	        Use polynomial regression
    Shallow neural network	    Add more layers/neurons
    Small decision tree	        Increase tree depth

Example:

    y = ax + b          → may underfit
    y = ax² + bx + c    → captures curvature
    
2. Reduce Regularization

Too much regularization restricts the model too strongly.

Regularized loss:

    Loss=Error+λ⋅Penalty

If λ is too large, weights become very small → model cannot learn.

Solution:

    Reduce L1 / L2 regularization strength

3. Train for More Iterations

Sometimes the model just has not trained long enough.

Example

        Epoch 5  → High error
        Epoch 100 → Lower error

Solution:

    * Increase epochs
    * Allow optimizer to converge

4. Improve Feature Engineering

Poor features lead to underfitting even with good models.

    Solutions
        * Add informative features
        * Create interaction features
        * Use nonlinear transformations
    
    Example
        * Original feature: income
        * Better features: income², income / age
        
5. Reduce Dropout

If dropout rate is too high, the network cannot learn enough structure.

    Example
        * Dropout = 0.7  → too aggressive
        * Dropout = 0.2  → better learning

6. Use More Powerful Models

Some algorithms are inherently more flexible.

Example progression

    Linear Regression
          ↓
    Polynomial Regression
          ↓
    Decision Trees
          ↓
    Gradient Boosting
          ↓
    Neural Networks

7. Reduce Data Noise

If the dataset has too much noise or poor labels, the model struggles to learn patterns.

Solutions
    * Clean labels
    * Remove corrupted samples
    * Improve preprocessing

Visual Intuition

Underfitting
    True Pattern:      ~~~~~~~
    Model Prediction:  ------

The model is too rigid to capture the curve.


    +---------------------------+---------------------------------------------+
    | Technique                 | How it Prevents Underfitting                |
    +---------------------------+---------------------------------------------+
    | Increase Model Complexity | Allows model to capture more complex        |
    |                           | patterns in the data                        |  
    |                           |                                             |                             
    | Reduce Regularization     | Prevents weights from being overly          |
    |                           | restricted                                  |      
    |                           |                                             |  
    | Train Longer              | Gives the optimizer more time to learn      |
    |                           | the data patterns                           |
    |                           |                                             |              
    | Add Better Features       | Provides more informative inputs            |
    |                           | to the model                                |
    |                           |                                             |  
    | Reduce Dropout            | Allows more neurons to participate          |
    |                           | in learning                                 |
    |                           |                                             |
    | Use More Powerful Models  | Enables learning of nonlinear               |
    |                           | and complex relationships                   |
    |                           |                                             |  
    | Feature Engineering       | Creates interaction and nonlinear           |
    |                           | transformations of features                 |
    |                           |                                             |  
    | Reduce Data Noise         | Improves signal quality in training data    |
    |                           |                                             |
    | Increase Model Capacity   | Adds more parameters (layers, neurons,      |
    |                           | tree depth, etc.)                           |
    +---------------------------+---------------------------------------------+

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Polynomial Underfitting vs Overfitting": {
        "description": (
            "Fit polynomials of degree 1, 4, and 15 to noisy quadratic data. "
            "Watch training error fall and validation error diverge as degree increases."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — swap to "TkAgg" for a window
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval

np.random.seed(42)

# ── Ground truth: y = x² + noise ──────────────────────────────────────────
x_all = np.linspace(-3, 3, 200)
y_true = x_all ** 2

# Training data — 30 noisy points
n_train = 30
x_train = np.sort(np.random.uniform(-3, 3, n_train))
y_train = x_train ** 2 + np.random.normal(0, 1.5, n_train)

# Validation data — 50 fresh noisy points
x_val = np.sort(np.random.uniform(-3, 3, 50))
y_val = x_val ** 2 + np.random.normal(0, 1.5, 50)

degrees = [1, 4, 15]
labels  = ["Degree 1 (Underfit)", "Degree 4 (Good fit)", "Degree 15 (Overfit)"]

print("=" * 60)
print("  POLYNOMIAL FIT: UNDERFIT vs GOOD vs OVERFIT")
print("=" * 60)
print(f"  Training samples: {n_train}")
print(f"  True function:    y = x²  (+Gaussian noise σ=1.5)")
print()

results = []
for deg, label in zip(degrees, labels):
    # np.polyfit uses descending-degree convention
    coeffs = np.polyfit(x_train, y_train, deg)
    y_pred_train = np.polyval(coeffs, x_train)
    y_pred_val   = np.polyval(coeffs, x_val)

    train_mse = np.mean((y_train - y_pred_train) ** 2)
    val_mse   = np.mean((y_val   - y_pred_val  ) ** 2)

    results.append((label, train_mse, val_mse, coeffs))

    print(f"  {label}")
    print(f"    Train MSE : {train_mse:.3f}")
    print(f"    Val   MSE : {val_mse:.3f}")
    gap = val_mse - train_mse
    diagnosis = (
        "UNDERFIT  ← both errors high"    if deg == 1  else
        "GOOD FIT  ← low gap"             if deg == 4  else
        "OVERFIT   ← large val/train gap"
    )
    print(f"    Gap       : {gap:+.3f}  →  {diagnosis}")
    print()

# ── Visual ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Polynomial Regression: Underfitting → Good Fit → Overfitting",
             fontsize=14, fontweight="bold")

x_line = np.linspace(-3.2, 3.2, 500)
colours = ["steelblue", "seagreen", "crimson"]

for ax, (label, tr_mse, v_mse, coeffs), colour in zip(axes, results, colours):
    y_line = np.polyval(coeffs, x_line)
    ax.scatter(x_train, y_train, s=30, alpha=0.6, color="gray",
               label="Train data", zorder=3)
    ax.plot(x_line, x_line ** 2, "k--", lw=1.5, label="True (x²)", zorder=2)
    ax.plot(x_line, y_line, colour, lw=2, label="Model", zorder=4)
    ax.set_title(f"{label}\\nTrain MSE={tr_mse:.2f}  Val MSE={v_mse:.2f}",
                 fontsize=10)
    ax.set_ylim(-3, 14)
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("polynomial_fit_comparison.png", dpi=120)
print("  Plot saved → polynomial_fit_comparison.png")
print()
print("  Key Observations:")
print("  - Degree 1: Both errors are high (can't capture the curve)")
print("  - Degree 4: Low train AND val error  → generalises well")
print("  - Degree 15: Near-zero train error, high val error → memorised noise")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Bias-Variance Decomposition": {
        "description": (
            "Simulate the bias-variance decomposition by training 200 models "
            "on different random training sets and measuring how much their "
            "predictions vary (variance) versus how far they are from the truth (bias)."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Setup ──────────────────────────────────────────────────────────────────
def true_function(x):
    return np.sin(2 * np.pi * x)

N_EXPERIMENTS = 200          # number of independent training runs
N_TRAIN       = 25           # training points per experiment
NOISE_STD     = 0.3          # label noise
DEGREES       = [1, 4, 10]   # model complexities to compare

x_test = np.linspace(0, 1, 100)
y_test_true = true_function(x_test)

print("=" * 65)
print("  BIAS-VARIANCE DECOMPOSITION")
print("  True function: y = sin(2πx)")
print(f"  {N_EXPERIMENTS} independent models per degree, "
      f"{N_TRAIN} training points each")
print("=" * 65)
print()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Bias-Variance Decomposition across Model Complexities",
             fontsize=13, fontweight="bold")

for col_idx, deg in enumerate(DEGREES):
    predictions = np.zeros((N_EXPERIMENTS, len(x_test)))

    for exp in range(N_EXPERIMENTS):
        x_tr = np.random.uniform(0, 1, N_TRAIN)
        y_tr = true_function(x_tr) + np.random.normal(0, NOISE_STD, N_TRAIN)
        coeffs = np.polyfit(x_tr, y_tr, deg)
        predictions[exp] = np.polyval(coeffs, x_test)

    mean_pred = predictions.mean(axis=0)
    variance   = predictions.var(axis=0).mean()       # avg over test points
    bias_sq    = ((mean_pred - y_test_true) ** 2).mean()
    noise      = NOISE_STD ** 2
    total_expected = bias_sq + variance + noise

    print(f"  Degree {deg:2d} polynomial")
    print(f"    Bias²        : {bias_sq:.4f}")
    print(f"    Variance     : {variance:.4f}")
    print(f"    Noise        : {noise:.4f}  (irreducible)")
    print(f"    Expected MSE : {total_expected:.4f}  (Bias²+Var+Noise)")
    print()

    # ── Top plot: individual model fits ───────────────────────────────────
    ax_top = axes[0, col_idx]
    for i in range(min(30, N_EXPERIMENTS)):
        ax_top.plot(x_test, predictions[i], alpha=0.15, color="steelblue",
                    linewidth=0.8)
    ax_top.plot(x_test, y_test_true, "k-",  linewidth=2.5, label="True function")
    ax_top.plot(x_test, mean_pred,   "r--", linewidth=2,   label="Mean prediction")
    ax_top.set_title(f"Degree {deg}  |  Bias²={bias_sq:.3f}  Var={variance:.3f}",
                     fontsize=10)
    ax_top.set_ylim(-2.5, 2.5)
    ax_top.legend(fontsize=8)
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("y")

    # ── Bottom plot: bias² and variance stacked bar ───────────────────────
    ax_bot = axes[1, col_idx]
    bars = ax_bot.bar(["Bias²", "Variance", "Noise"],
                      [bias_sq, variance, noise],
                      color=["tomato", "steelblue", "grey"])
    ax_bot.set_title(f"Error Components — Degree {deg}", fontsize=10)
    ax_bot.set_ylabel("MSE contribution")
    ax_bot.set_ylim(0, max(bias_sq + variance + noise, 0.4) * 1.3)
    for bar in bars:
        ax_bot.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("bias_variance_decomposition.png", dpi=120)
print("  Plot saved → bias_variance_decomposition.png")
print()
print("  Key Takeaways:")
print("  - Degree 1  : High bias dominates. Model too rigid → underfitting.")
print("  - Degree 4  : Balance. Reasonable bias AND reasonable variance.")
print("  - Degree 10 : Variance explodes. Model chases noise → overfitting.")
print("  - Noise (σ²) is constant — NO model can do better than this floor.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Learning Curves Diagnostic": {
        "description": (
            "Plot learning curves (train vs validation error vs training set size) "
            "for underfitting, good fit, and overfitting models. "
            "This is the primary diagnostic tool for identifying each failure mode."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy

class LinearRegression:
    def __init__(self,fit_intercept=True): self.fit_intercept=fit_intercept
    def fit(self,X,y):
        Xa=np.column_stack([np.ones(len(X)),X]) if self.fit_intercept else X
        w,_,_,_=np.linalg.lstsq(Xa,y,rcond=None)
        if self.fit_intercept: self.intercept_=w[0]; self.coef_=w[1:]
        else: self.intercept_=0.; self.coef_=w
        return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"fit_intercept":self.fit_intercept}

class Ridge:
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        mu=X.mean(0); Xc=X-mu; yc=y-y.mean(); n,d=Xc.shape
        self.coef_=np.linalg.solve(Xc.T@Xc+self.alpha*np.eye(d),Xc.T@yc)
        self.intercept_=y.mean()-mu@self.coef_; return self
    def predict(self,X): return X@self.coef_+self.intercept_
    def get_params(self,deep=True): return {"alpha":self.alpha}

from itertools import combinations_with_replacement as _cwr
class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True): self.degree=degree; self.include_bias=include_bias
    def _combos(self,n):
        c=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: c.append(())
            else: c.extend(_cwr(range(n),d))
        return c
    def fit(self,X,y=None): self._c=self._combos(X.shape[1]); return self
    def transform(self,X):
        cols=[]
        for combo in self._c:
            if not combo: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)
    def get_params(self,deep=True): return {"degree":self.degree,"include_bias":self.include_bias}

class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,s in self.steps[:-1]: Xt=s.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def get_params(self,deep=True):
        p={}
        for nm,s in self.steps: p.update({nm+"__"+k:v for k,v in s.get_params().items()})
        return p

def mean_squared_error(y_true,y_pred): return np.mean((np.asarray(y_true)-np.asarray(y_pred))**2)

np.random.seed(7)

# ── Data: y = sin(x) + noise ───────────────────────────────────────────────
def make_data(n):
    x = np.sort(np.random.uniform(0, 6, n))
    y = np.sin(x) + np.random.normal(0, 0.35, n)
    return x.reshape(-1, 1), y

X_train_full, y_train_full = make_data(300)
X_val,        y_val        = make_data(200)

train_sizes = np.arange(10, 280, 10)

# ── Three models ───────────────────────────────────────────────────────────
models = {
    "Underfit  (Degree 1)":    Pipeline([("poly", PolynomialFeatures(1)),
                                          ("reg",  LinearRegression())]),
    "Good Fit  (Degree 5)":    Pipeline([("poly", PolynomialFeatures(5)),
                                          ("reg",  Ridge(alpha=0.1))]),
    "Overfit   (Degree 15)":   Pipeline([("poly", PolynomialFeatures(15)),
                                          ("reg",  LinearRegression())]),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Learning Curves: Train vs Validation Error vs Dataset Size",
             fontsize=13, fontweight="bold")

print("=" * 65)
print("  LEARNING CURVE DIAGNOSTICS")
print("=" * 65)
print()

for ax, (name, model) in zip(axes, models.items()):
    tr_errors, val_errors = [], []

    for size in train_sizes:
        X_sub, y_sub = X_train_full[:size], y_train_full[:size]
        model.fit(X_sub, y_sub)

        tr_pred  = model.predict(X_sub)
        val_pred = model.predict(X_val)

        tr_errors.append(mean_squared_error(y_sub,  tr_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))

    ax.plot(train_sizes, tr_errors,  "b-o", ms=4, label="Train Error",
            linewidth=2)
    ax.plot(train_sizes, val_errors, "r-s", ms=4, label="Val Error",
            linewidth=2)
    ax.axhline(0.35 ** 2, color="gray", linestyle="--", lw=1,
               label=f"Noise floor (σ²={0.35**2:.3f})")
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.9)

    final_gap = val_errors[-1] - tr_errors[-1]
    print(f"  {name}")
    print(f"    Final train error : {tr_errors[-1]:.4f}")
    print(f"    Final val   error : {val_errors[-1]:.4f}")
    print(f"    Gap (val-train)   : {final_gap:+.4f}")
    if "Underfit" in name:
        print("    Diagnosis: UNDERFIT — both errors high & flat")
        print("               Adding more data WON'T help much.")
        print("               Fix: increase model complexity.")
    elif "Good" in name:
        print("    Diagnosis: GOOD FIT — small gap, both converge low")
        print("               More data continues to improve generalisation.")
    else:
        print("    Diagnosis: OVERFIT  — large gap, train near zero")
        print("               More data DOES help (val curve declining).")
        print("               Also try: regularisation, simpler model.")
    print()

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=120)
print("  Plot saved → learning_curves.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Early Stopping Demo": {
        "description": (
            "Train a neural network and track validation loss each epoch. "
            "Visualise the exact moment overfitting starts and where "
            "early stopping would have intervened."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(99)

# ── Simulate a realistic training / validation loss curve ─────────────────
# (mimics a neural network that starts overfitting around epoch 40)
n_epochs = 120

# Validation loss: decreases then increases
val_inflection = 40
epoch_arr = np.arange(1, n_epochs + 1)

# Training loss decreases smoothly
train_loss = 1.8 * np.exp(-0.03 * epoch_arr) + 0.12 + \
             np.random.normal(0, 0.008, n_epochs)

# Validation loss: decreases, hits minimum, then rises (overfitting)
val_base  = 1.9 * np.exp(-0.035 * epoch_arr) + 0.25
overfit   = np.where(epoch_arr > val_inflection,
                     0.0018 * (epoch_arr - val_inflection) ** 1.6, 0)
val_loss  = val_base + overfit + np.random.normal(0, 0.015, n_epochs)

# ── Find best epoch (minimum val loss) ────────────────────────────────────
best_epoch = int(np.argmin(val_loss)) + 1
best_val   = val_loss[best_epoch - 1]

# ── Patience-based early stopping ─────────────────────────────────────────
PATIENCE = 10
best_seen, patience_counter, stop_epoch = np.inf, 0, n_epochs
for ep in range(n_epochs):
    if val_loss[ep] < best_seen - 1e-4:
        best_seen        = val_loss[ep]
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        stop_epoch = ep + 1
        break

print("=" * 60)
print("  EARLY STOPPING DEMONSTRATION")
print("=" * 60)
print(f"  Total epochs available    : {n_epochs}")
print(f"  Best val loss at epoch    : {best_epoch}   (loss = {best_val:.4f})")
print(f"  Early-stop triggers at    : {stop_epoch}   (patience = {PATIENCE})")
print()
print("  Epoch     Train Loss    Val Loss")
print("  ──────────────────────────────")
for ep in [1, 10, 20, best_epoch, stop_epoch, n_epochs]:
    ep = min(ep, n_epochs)
    marker = ""
    if ep == best_epoch:
        marker = "  ← BEST VAL LOSS"
    elif ep == stop_epoch:
        marker = "  ← EARLY STOP"
    elif ep == n_epochs:
        marker = "  ← FINAL (overfit)"
    print(f"  {ep:5d}     {train_loss[ep-1]:.4f}        "
          f"{val_loss[ep-1]:.4f}{marker}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(epoch_arr, train_loss, "b-",  lw=2, label="Training loss",   alpha=0.9)
ax.plot(epoch_arr, val_loss,   "r-",  lw=2, label="Validation loss", alpha=0.9)

# Shade overfitting region
ax.axvspan(best_epoch, n_epochs, alpha=0.08, color="red",
           label="Overfitting zone")

# Mark key points
ax.axvline(best_epoch, color="green",  ls="--", lw=2,
           label=f"Best checkpoint (epoch {best_epoch})")
ax.axvline(stop_epoch, color="orange", ls="--", lw=2,
           label=f"Early stop (epoch {stop_epoch}, patience={PATIENCE})")

ax.annotate(f"Min val loss\\n{best_val:.3f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 8, best_val + 0.12),
            arrowprops=dict(arrowstyle="->", color="green"), fontsize=9,
            color="green")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Early Stopping: Catching Overfitting Before It Hurts",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0.05, 1.4)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("early_stopping_demo.png", dpi=120)
print()
print("  Plot saved → early_stopping_demo.png")
print()
print("  Key Takeaways:")
print("  - Training loss ALWAYS decreases (model keeps memorising)")
print("  - Validation loss hits a minimum then RISES → overfitting started")
print("  - Best checkpoint = minimum validation loss")
print(f"  - Early stopping (patience={PATIENCE}) stops {n_epochs - stop_epoch} "
      f"epochs early, saving compute")
print("    and returning a model that generalises much better than the final one.")
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
    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }