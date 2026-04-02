"""Module 01 · Linear Regression"""

"""
Linear Regression — The Foundation of Supervised Learning
==========================================================

Linear Regression is the bedrock of machine learning. Before neural networks,
before SVMs, before gradient boosting — there was fitting a line to data.
Every more complex model you will ever study is a generalisation or extension
of the ideas introduced here.
"""

import base64
import math
import os
import re
import textwrap

DISPLAY_NAME = "02 · Linear Regression"
ICON         = "📈"
SUBTITLE     = "The foundation of prediction — fitting a line to minimize error"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## 01 · Linear Regression

Linear Regression is the bedrock of supervised learning. It models the relationship between
a continuous target variable `y` and one or more input features `X` by fitting a linear equation.

### Part 1: The Model — What Linear Regression Computes

### What is Linear Regression?

Linear Regression is the simplest and most foundational supervised learning algorithm.
Given a set of input features and a continuous output value, it finds the best-fitting
straight line (or hyperplane) through the data — one that can then be used to predict
the output for new, unseen inputs.

The word "regression" comes from Francis Galton (1886), who discovered that the heights
of children tended to "regress toward the mean" compared to their parents. He fit a line
to that relationship — and in doing so, invented one of the most important tools in all
of statistics and machine learning.

The Real-World Analogy:

Imagine you're trying to predict house prices. You gather data on 500 houses: their
size (square feet) and sale price. You plot them on a graph — size on the X axis,
price on the Y axis. The data forms a rough diagonal cloud. Linear regression finds
the single line that passes through that cloud with the smallest possible error.

Once you have that line, predicting the price of a new 2000 sq ft house is trivial:
just read off the Y value at X = 2000.

    Things that exist inside the model (learnable parameters):
        - Weights (w₁, w₂, ..., wₚ) — the slope/coefficient for each feature
        - Bias (w₀) — the intercept, shifts the line up/down

    Things that exist only at setup time (hyperparameters / configuration):
        - Learning rate (α) — controls step size during gradient descent
        - Number of iterations — how long to run gradient descent
        - Regularisation strength (λ) — controls overfitting penalty
        - Which solver to use: Normal Equation vs Gradient Descent

### Linear Regression as Empirical Risk Minimisation (ERM)

Before diving into the math, it's worth naming what linear regression *is* at a
theoretical level. This framing will unify every supervised learning algorithm you
encounter from here forward.

In the ERM framework, learning always has three components:

    1. A hypothesis class H  — the set of functions we're allowed to fit
    2. A loss function L      — how we measure the cost of a wrong prediction
    3. An optimiser           — the procedure that finds the best function in H

For Linear Regression these choices are:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Hypothesis class:  H = { f(x) = w·x + b  |  w ∈ ℝᵖ }       │
    │                     (all linear functions of the input)     │
    │                                                             │
    │  Loss function:     L(y, ŷ) = (y − ŷ)²                      │
    │                     (squared loss — penalises large errors) │
    │                                                             │
    │  Training objective:                                        │
    │      min_w  (1/n) Σᵢ (yᵢ − w·xᵢ − b)²                       │
    │             ↑                                               │
    │             this is empirical risk — average loss on data   │
    │                                                             │
    │  Optimiser:  Normal Equation (exact)                        │
    │              or Gradient Descent (iterative)                │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Linear regression is therefore a specific instance of ERM:
    Model class:  linear functions
    Loss:         squared error
    Optimiser:    Normal Equation or Gradient Descent

This matters because every supervised algorithm you will study — logistic regression,
SVMs, neural networks — follows the same three-component template. They differ only
in their choice of hypothesis class and loss function. Understanding linear regression
as ERM means understanding the *architecture* of all supervised learning, not just one
algorithm.

---

### The Math Behind It : The Core Idea

## Linear Regression computes its prediction in one step:


Given `n` training examples `{(x₁, y₁), ..., (xₙ, yₙ)}`, we want to find a function:

```
                ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ
```

where `w₀` is the **bias** (intercept) and `w₁...wₚ` are **weights** (coefficients).
In matrix notation: `ŷ = Xw`

Or in vector form:

                ŷ = w · x    (dot product of weight vector and input vector)

Or in matrix form (for all n training examples at once):

                ŷ = Xw

Where:
    - X is the (n × p+1) design matrix — each row is one training example, with a leading column of 1s for the bias term
    - w is the (p+1 × 1) weight vector — one weight per feature, plus the bias
    - ŷ is the (n × 1) prediction vector

** ------------ Detailed Explanation ------------ **


### Step 1 — The Linear Equation:

For a single feature (simple linear regression):

                            ŷ = w₀ + w₁x₁

This is the equation of a straight line. w₁ is the slope (how much ŷ changes when
x₁ increases by 1) and w₀ is the y-intercept (the value of ŷ when x₁ = 0).

For multiple features (multiple linear regression):

                    ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ

In 2D (two features), this is the equation of a plane. In higher dimensions, it
becomes a hyperplane — but the intuition is always the same: a flat surface that
best separates high-output from low-output regions of the input space.


    # =======================================================================================# 
    **Diagram 1 — Visualising the Linear Fit and Residuals:**

    SIMPLE LINEAR REGRESSION: House Size vs. Price
    ══════════════════════════════════════════════════════════════

        Price
        ($k) ↑
             │                                        ● ← actual value
         500 │                                   ●  ↕ ← residual (error)
             │                              ●   ╱───── fitted line (ŷ)
         400 │                         ●   ╱  ●
             │                    ● ● ╱
         300 │               ●   ╱  ●
             │          ●   ╱ ●
         200 │     ● ● ╱  ●
             │    ╱  ●
         100 │  ╱●
             │╱
           0 └──────────────────────────────────────────────────────────────→
             0    500   1000   1500   2000   2500   3000   4000   5000   6000
                                                                  Size (sq ft)

    Each residual = actual yᵢ minus predicted ŷᵢ
    The goal: find the line that minimises the total residual error.

    ──────────────────────────────────────────────────────────────
    Point     Size   Actual Price   Predicted ŷ    Residual
    ──────────────────────────────────────────────────────────────
    House 1   1000      $210k          $200k          +10k
    House 2   1500      $280k          $300k          -20k
    House 3   2000      $410k          $400k          +10k
    House 4   2500      $480k          $500k          -20k
    ──────────────────────────────────────────────────────────────
    # =======================================================================================# 


### Step 2 — Understanding the Dot Product Intuitively:

Just like the perceptron, the core computation in linear regression is a dot product:

                            ŷ = w · x

This has the same geometric meaning: it measures alignment between the weight vector
and the input vector. But unlike the perceptron, we don't squash it through a step
function — we use the raw value as our prediction.

Each weight wᵢ encodes: "how much does feature i contribute to the output?"

A weight w₁ = 150 on the "house size" feature means: every extra square foot adds
$150 to the predicted price. The bias w₀ is the baseline price for a house with
zero square feet — not physically meaningful here, but mathematically necessary.


### The Role of the Bias:

The bias w₀ (also called the intercept) shifts the entire prediction line up or down.
Without it, every prediction line would be forced to pass through the origin (0, 0).

In the house price example: with no bias, a 0 sq ft house must cost $0, which forces
the line to start at the origin and may distort all other predictions. The bias frees
the line to sit wherever the data actually lives.

In matrix form, we include the bias by prepending a column of 1s to X:

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  X_augmented = [ 1  x₁  x₂ ... xₚ ]     w = [w₀]        │
    │                                             [w₁]        │
    │                                             [w₂]        │
    │                                             [...]       │
    │                                             [wₚ]        │
    │                                                         │
    │  ŷ = X_augmented · w  ← one clean matrix multiplication │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

This trick (prepending 1s) lets us treat the bias as just another weight and
handle everything in a single matrix multiply. You will see this trick everywhere
in machine learning.

---

## Part 2: Loss Function — Mean Squared Error

### The Loss Function — How Do We Measure "Best Fit"?

We minimise the **Mean Squared Error (MSE)**:

        MSE = (1/n) Σ (yᵢ - ŷᵢ)²

Why squared? It penalises large errors more, is differentiable everywhere, and has a
unique closed-form solution.

## Breakdown: 

There are infinitely many lines you could draw through a cloud of points. We need
a mathematical criterion to decide which one is *best*. That criterion is the Loss
Function.


### Mean Squared Error (MSE):

The standard loss function for linear regression is Mean Squared Error:

                    MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

In words: for each training example, compute the difference between the true label
and our prediction (this is the residual), square it, and then average across all
n examples.

Why do we square the residuals? Three reasons:

1. **Sign cancellation**: residuals can be positive or negative. Without squaring,
   a +10 error and a -10 error would cancel out and look like zero total error.
   Squaring makes every error contribute positively.

2. **Penalise large errors more**: a residual of 10 contributes 100 to the sum.
   A residual of 20 contributes 400 — four times as much. This makes the model
   especially keen to avoid catastrophic mispredictions.

3. **Mathematical convenience**: the squared function is differentiable everywhere,
   allowing us to take its derivative and find the exact minimum analytically.
   This is critical for the Normal Equation (Part 3).


    # ======================================================================================= # 
    **Diagram 2 — The MSE Loss Surface is a Convex Bowl:**

    MSE Loss (simplified, one weight w₁)
    ══════════════════════════════════════════════════════════════

        Loss
          ↑
          │  ╲                        ╱
          │    ╲                    ╱
          │      ╲                ╱
          │        ╲            ╱
          │          ╲        ╱
          │            ╲    ╱
          │              ╲╱  ← global minimum (optimal w₁*)
          │
          └──────────────────────────────→  w₁

    This bowl shape (convex) is crucial:
    • There is ONLY ONE minimum — no local minima to get stuck in
    • Gradient descent is guaranteed to find the global optimum
    • The minimum is exactly at w* = (XᵀX)⁻¹Xᵀy  (the Normal Equation)

    In 2D (two weights w₀, w₁), the loss surface is a 3D bowl (paraboloid).
    In higher dimensions (p weights), it's still a bowl — just harder to visualise.

    COMPARE with neural networks:
    Linear regression loss → convex bowl → one global minimum (easy)
    Neural network loss    → bumpy landscape → many local minima (hard)
    # ======================================================================================= # 


### Why Not Use Mean Absolute Error (MAE) instead?

MAE = (1/n) Σ |yᵢ - ŷᵢ|    uses the absolute value instead of squaring.

MAE is more robust to outliers (a huge residual has linear not quadratic impact).
But the absolute value has a sharp kink at zero — its derivative is undefined there.
This means MAE has no closed-form solution and requires more careful gradient methods.

The choice between MSE and MAE depends on whether outliers in your data represent
genuine signal (use MAE) or noise you want to down-weight (MSE is fine either way,
though MSE does get pulled by extreme values).

### Probabilistic Interpretation — Why MSE is Not Just Convenient

There is a deeper reason to use MSE beyond the three practical reasons above.
Minimising MSE is mathematically equivalent to Maximum Likelihood Estimation under
Gaussian noise. This single fact upgrades linear regression from an algorithm to a
principled statistical model.

**The Generative Assumption:**

Assume the data is generated by a true linear function plus random Gaussian noise:

                    y = w·x + b + ε       where ε ~ N(0, σ²)

That is: for any input x, the true output y is the linear prediction plus a small
random perturbation drawn from a Normal distribution centred at zero.

Under this model, the probability of observing a particular yᵢ given xᵢ is:

            P(yᵢ | xᵢ; w) = (1/√(2πσ²)) · exp( -(yᵢ - w·xᵢ - b)² / (2σ²) )

This is the Gaussian PDF — the probability of yᵢ being "explained" by our model.

**Maximum Likelihood Estimation (MLE):**

MLE says: find the weights w that maximise the probability of observing the entire
training set. For n independent examples, this is:

                P(y | X; w) = Πᵢ P(yᵢ | xᵢ; w)

Taking the log (log-likelihood, monotonically equivalent):

            log P(y | X; w) = -n·log(√(2πσ²)) - (1/2σ²) Σᵢ (yᵢ - w·xᵢ - b)²

Maximising log-likelihood is the same as minimising the last term:

                            min_w  Σᵢ (yᵢ - w·xᵢ - b)²

Which is exactly minimising MSE (up to the 1/n constant, which doesn't affect w*).

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Linear Regression + MSE                                    │
    │      = Maximum Likelihood Estimation                        │
    │           under the assumption of Gaussian noise            │
    │                                                             │
    │  The Gaussian noise assumption IS the justification for MSE │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

**What Changes If the Noise is Not Gaussian?**

This is exactly the bridge to other models:
    ε ~ Gaussian      → MSE loss           → Linear Regression
    ε ~ Laplace       → MAE loss           → Least Absolute Deviations
    y ~ Bernoulli     → Log loss           → Logistic Regression
    y ~ Poisson       → Poisson deviance   → Poisson Regression

The noise assumption *determines* the right loss function. When you understand this,
choosing a loss function becomes a modelling decision — "what do I believe about the
distribution of my errors?" — not an arbitrary choice.

---

*(The LINE assumptions — Linearity, Independence, Normality, Equal Variance — are covered in full depth in Part 5.)*


### Part 3: Solving It — Two Approaches

### 3.1 Normal Equation (Analytical)

    w* = (XᵀX)⁻¹ Xᵀy

Direct solution — no iterations needed. But inverting `XᵀX` costs **O(p³)**, so it
breaks down for high-dimensional data (p > ~10,000).

### 3.2 Gradient Descent (Iterative)

    w := w - α · ∇L(w)
    ∇L = (2/n) Xᵀ(Xw - y)

Scales to millions of features. Step size controlled by learning rate `α`.

Once we have a loss function, finding the best weights means minimising it.
There are two fundamentally different approaches.


#### The Two Approaches in Depth 

### Approach 1 — The Normal Equation (Analytical Solution):

If the loss is MSE and the model is linear, calculus gives us an exact closed-form
formula for the optimal weights:

                        w* = (XᵀX)⁻¹ Xᵀy

This is the Normal Equation. "Normal" here is from geometry: the residual vector
is perpendicular (normal) to the column space of X at the optimal solution.

**Derivation sketch:**
The MSE loss in matrix form is:

                    L(w) = (1/n) ||Xw - y||²

Taking the gradient and setting it to zero:

                    ∇L = (2/n) Xᵀ(Xw - y) = 0
                    XᵀXw = Xᵀy
                    w* = (XᵀX)⁻¹ Xᵀy

No iterations, no learning rate, no hyperparameters — just one matrix calculation.


    # =======================================================================================# 
    **Diagram 3 — The Normal Equation: Geometric Intuition:**

    WHY IS THE RESIDUAL PERPENDICULAR AT THE OPTIMUM?

    Think of the column space of X as a flat plane in n-dimensional space.
    The target vector y probably doesn't lie in this plane (if it did, a perfect
    fit would exist). The best prediction ŷ = Xw is the point IN the plane
    closest to y — the orthogonal projection of y onto the column space.

                              y (true target, outside the plane)
                              │
                              │ ← residual vector (y - ŷ) is perpendicular
                              │   to the plane at the optimal point
                              ↓
                    ──────────●──────────  ← Column space of X (the "reachable" plane)
                              ↑
                              ŷ = Xw* (orthogonal projection of y)

    The Normal Equation is literally computing this orthogonal projection.
    # ======================================================================================= # 


**The Cost of the Normal Equation:**

The bottleneck is computing (XᵀX)⁻¹, which requires inverting a (p × p) matrix.
Matrix inversion scales as O(p³) — cubically in the number of features.

    p = 100 features   → (100)³ = 1,000,000 operations    ← fast
    p = 1,000 features → (1000)³ = 1,000,000,000 ops       ← noticeable
    p = 10,000 features→ (10000)³ = 10^12 operations       ← prohibitive

Rule of thumb: use the Normal Equation when p < ~10,000. Beyond that, switch to
gradient descent. For modern deep learning with millions of parameters, the Normal
Equation is completely infeasible — gradient descent is the only option.

Also: if XᵀX is singular (non-invertible), the Normal Equation breaks down entirely.
This happens when features are perfectly collinear (one feature is a linear combination
of others). Ridge regularisation (Part 5) fixes this by adding λI to XᵀX, guaranteeing
invertibility.

**Numerical Stability — Why We Never Actually Invert XᵀX:**

In practice, computing (XᵀX)⁻¹ explicitly is numerically fragile. When XᵀX is
ill-conditioned (nearly singular — features nearly collinear), floating-point errors
accumulate during inversion and the result can be wildly wrong even though it looks
valid.

Instead, real implementations solve the linear system XᵀXw = Xᵀy using two
numerically stable decompositions:

    QR Decomposition:
        Decompose X = QR, where Q is orthogonal and R is upper triangular.
        Then w* = R⁻¹Qᵀy.
        Inverting a triangular matrix is stable and cheap. O(np²) cost.
        Used by sklearn's LinearRegression by default.

    SVD (Singular Value Decomposition):
        Decompose X = UΣVᵀ, giving the pseudoinverse X⁺ = VΣ⁺Uᵀ.
        Then w* = X⁺y.
        Most robust — handles singular and near-singular cases gracefully.
        Used by numpy.linalg.lstsq.

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Never write:   w = inv(X.T @ X) @ X.T @ y                  │
    │  Always write:  w = lstsq(X, y)   ← uses SVD internally     │
    │                                                             │
    │  The formula (XᵀX)⁻¹Xᵀy is for derivation and intuition.     │
    │  The implementation should always use QR or SVD.            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

The condition number of XᵀX measures how ill-conditioned the system is. A high
condition number (> 10⁶) means small perturbations in the data can cause large
changes in w* — your solution is numerically unreliable. Regularisation and feature
scaling both help reduce the condition number.

### Approach 2 — Gradient Descent (Iterative Solution):

Instead of solving for w* in one shot, gradient descent starts with a random guess
and iteratively nudges the weights in the direction that reduces the loss:

                        w := w - α · ∇L(w)

Where:
- α (alpha) is the learning rate — how large each step is
- ∇L(w) is the gradient of the loss — which direction is "uphill"
- We subtract the gradient to go downhill toward the minimum


**The Gradient of MSE:**

For MSE = (1/n) Σ (yᵢ - ŷᵢ)², the gradient with respect to weights w is:

                    ∇L = (2/n) Xᵀ(Xw - y)

Breaking this down:
- (Xw - y) is the vector of residuals for the current weights
- Multiplying by Xᵀ "projects" those residuals back into weight space
- The (2/n) is a constant scaling factor (often absorbed into α)

This is the Batch Gradient Descent formula — it uses all n training examples
at each step. Variants exist:
- Stochastic Gradient Descent (SGD): one example per step — noisy but fast
- Mini-Batch GD: k examples per step — balances speed and stability


    # ======================================================================================= # 
    **Diagram 4 — Gradient Descent on the Loss Surface:**

    LOSS SURFACE (2 weights: w₀ and w₁)
    ══════════════════════════════════════════════════════════════

    w₁
    ↑
    │     (start here, random init)
    │           ★
    │          ↙
    │         ★
    │        ↙
    │       ★
    │      ↙
    │     ★
    │    ↙
    │   ★  ← converging...
    │  ↙
    │ ●  ← global minimum (optimal w*)
    │
    └─────────────────────────────────────→  w₀

    Each ★ is one iteration. The arrow points in the direction
    -∇L(w) (the negative gradient = steepest downhill direction).
    The step size is controlled by the learning rate α.


    LEARNING RATE EFFECTS:
    ══════════════════════════════════════════════════════════════

    α too LARGE:                       α too SMALL:
    (overshoots, oscillates)           (crawls, takes forever)

    w₁                                  w₁
    ↑   ★           ★                   ↑   ★★★★
    │     ↘       ↗                     │       ★★★★
    │       ↘   ↗                       │           ★★★
    │         ★                         │              ★★
    │       ↗   ↘                       │                ★★
    │     ↗       ↘                     │                  ★★
    │   ★           ★                   │                    ★●
    │                                   │
    └───────────────────→  w₀           └─────────────────────────→  w₀

    α JUST RIGHT:
    Smooth convergence in a reasonable number of steps.

    # ======================================================================================= # 


**Why is the Gradient (2/n) Xᵀ(Xw - y)?**

Let's derive it step by step to demystify it.

MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

For one weight wⱼ, take the partial derivative using the chain rule:

    ∂MSE/∂wⱼ = (1/n) Σᵢ  2(yᵢ - ŷᵢ) · ∂(yᵢ - ŷᵢ)/∂wⱼ
             = (1/n) Σᵢ  2(yᵢ - ŷᵢ) · (-xᵢⱼ)
             = -(2/n) Σᵢ xᵢⱼ(yᵢ - ŷᵢ)

Writing this for all weights simultaneously in matrix form:

             ∇L = -(2/n) Xᵀ(y - Xw)
                = (2/n) Xᵀ(Xw - y)      ← flip sign to get this form

The gradient says: for each weight wⱼ, its gradient is the correlation between
feature j's values and the current residuals. If feature j consistently has high
values where our errors are also high, wⱼ should increase. This is elegant credit
assignment — the gradient automatically focuses each weight's update on its own
contribution to the error.


### Concrete Training Walkthrough:

Let's manually trace one full gradient descent step on a tiny dataset.

    Dataset (n=3, p=1 feature + bias):
    ──────────────────────────────────────────────────
    x    y_true    y_hat (initial)    residual
    ──────────────────────────────────────────────────
    1      3         1.0               2.0
    2      5         2.0               3.0
    3      7         3.0               4.0
    ──────────────────────────────────────────────────

    Initial weights: w₀ = 0 (bias), w₁ = 1.0
    Learning rate: α = 0.01

    Step 1 — Compute predictions:
        ŷ₁ = 0 + 1.0 × 1 = 1.0    residual₁ = 3 - 1.0 = 2.0
        ŷ₂ = 0 + 1.0 × 2 = 2.0    residual₂ = 5 - 2.0 = 3.0
        ŷ₃ = 0 + 1.0 × 3 = 3.0    residual₃ = 7 - 3.0 = 4.0

    Step 2 — Compute MSE:
        MSE = (1/3) × (2.0² + 3.0² + 4.0²) = (1/3) × (4 + 9 + 16) = 9.67

    Step 3 — Compute gradients:
        ∂L/∂w₀ = (2/3) × [(-1×2.0) + (-1×3.0) + (-1×4.0)]
               = (2/3) × (-9.0) = -6.0

        ∂L/∂w₁ = (2/3) × [(-1×2.0) + (-2×3.0) + (-3×4.0)]
               = (2/3) × (-2.0 - 6.0 - 12.0) = (2/3) × (-20.0) = -13.33

    Step 4 — Update weights:
        w₀ := 0   - 0.01 × (-6.0)  =  0.06
        w₁ := 1.0 - 0.01 × (-13.33) = 1.133

    New predictions:
        ŷ₁ = 0.06 + 1.133 × 1 = 1.193    residual₁ = 3 - 1.193 = 1.807
        ŷ₂ = 0.06 + 1.133 × 2 = 2.326    residual₂ = 5 - 2.326 = 2.674
        ŷ₃ = 0.06 + 1.133 × 3 = 3.459    residual₃ = 7 - 3.459 = 3.541

    New MSE = (1/3) × (1.807² + 2.674² + 3.541²) = (1/3) × (3.265 + 7.15 + 12.54) = 7.65

    Loss dropped from 9.67 → 7.65 in a single step. After ~1000 steps,
    the weights will converge to w₀ ≈ 1.0, w₁ ≈ 2.0 (the true underlying slope).

---

### Part 4: Inductive Bias and Geometric Interpretation — What Is the Model Learning?

### What is Inductive Bias?

Every learning algorithm embeds assumptions about the world before seeing any data.
These assumptions are called the **inductive bias** of the model — they determine what
kinds of patterns the algorithm can even find, and what it *believes* about unseen data.

Without inductive bias, generalisation is impossible. A model with no assumptions about
structure could fit any training set perfectly but would have no principled basis for
predicting anything new. (This is the No Free Lunch Theorem.)

**Linear Regression's Inductive Bias:**

Linear regression encodes a specific belief about the world:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Linear regression assumes:                                 │
    │                                                             │
    │  1. ADDITIVITY — the effect of each feature is              │
    │     independent of all other features                       │
    │     (feature j contributes wⱼxⱼ regardless of x₁,...,xₚ)      │
    │                                                             │
    │  2. LINEARITY — doubling a feature value doubles its        │
    │     contribution to the output                              │
    │                                                             │
    │  3. GLOBAL SMOOTHNESS — the same linear relationship        │
    │     holds everywhere in input space                         │
    │     (no local patterns, no interactions)                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

These beliefs are powerful when true (the model is fast, interpretable, and
generalisable) and limiting when false (no amount of data will help a linear model
fit y = x² or y = x₁ × x₂).

**Comparing Inductive Biases Across Models:**

    ──────────────────────────────────────────────────────────────
    Model               Inductive Bias
    ──────────────────────────────────────────────────────────────
    Linear Regression   Relationships are additive and linear
    Decision Tree       Relationships are axis-aligned step functions
    k-NN                Nearby points have similar outputs
    Neural Network      Hierarchical compositions of local patterns
    ──────────────────────────────────────────────────────────────

None of these biases is universally correct. Choosing a model means choosing a belief.
The art of machine learning is matching your model's inductive bias to the true
structure of your problem.

### The Decision Boundary as a Hyperplane:

In simple linear regression (one feature), the model fits a line through 2D space.
In multiple linear regression (two features), the model fits a plane through 3D space.
With p features, the model fits a p-dimensional hyperplane through (p+1)-dimensional space.

The key insight: linear regression is always finding a flat surface. The weights
determine the orientation (tilt) of that surface, and the bias determines its height.

For classification (logistic regression, later), we use a similar flat surface as a
decision boundary. The perceptron you studied earlier is doing exactly this — the
weights define a hyperplane, and the step function decides which side of the plane
an input falls on.

Linear regression, the perceptron, logistic regression, and SVMs are all, at their
core, finding the best hyperplane for their respective objectives.


    # ======================================================================================= # 
    **Diagram 5 — The Hyperplane in 2D Feature Space:**

    TWO FEATURES: Size (x₁) and Age (x₂) predicting House Price (y)
    ══════════════════════════════════════════════════════════════════

    The model: ŷ = w₀ + w₁·size + w₂·age
    This is a plane in 3D (size, age, price) space.

    What happens if we project that plane down to 2D (size vs age)?

    Age
    (years) ↑
            │
         50 │  $200k   $250k   $300k   $350k   $400k
            │    ×        ×       ×       ×       ×
         40 │  $220k   $270k   $320k   $370k   $420k
            │    ×        ×       ×       ×       ×
         30 │  $240k   $290k   $340k   $390k   $440k
            │    ×        ×       ×       ×       ×
         20 │  $260k   $310k   $360k   $410k   $460k
            │
          0 └────────────────────────────────────────→
            0    500    1000    1500    2000    2500
                                              Size (sq ft)

    Each cell shows the model's predicted price.
    Moving right (larger house) → price increases (w₁ > 0)
    Moving up (older house) → price decreases (w₂ < 0, older = worth less)

    The model learns BOTH relationships simultaneously.
    # ======================================================================================= # 

---

### Part 5: Assumptions — When Does Linear Regression Work?

The LINE acronym describes the four assumptions:

**L — Linearity**
The relationship between X and y must actually be linear. If the true relationship is
y = x², a straight line will fit poorly. You can often fix this by transforming features
(e.g., adding x² as a new feature — this is Polynomial Regression, which is still
linear regression because it's linear in the *weights*).

**I — Independence**
Each training example must be independent of the others. The most common violation is
time-series data: today's stock price depends on yesterday's. Violating this means your
standard errors are wrong and confidence intervals are unreliable.

**N — Normality**
The residuals (errors) should be approximately normally distributed. This matters mostly
for statistical inference (hypothesis testing, confidence intervals). For pure prediction,
violating normality has minimal impact.

**E — Equal Variance (Homoscedasticity)**
The variance of residuals should be constant across all input values. If variance grows
with x (e.g., residuals are larger for bigger houses than smaller ones), this is
heteroscedasticity. It doesn't bias predictions, but makes confidence intervals wrong
and large-error regions of the data dominate the MSE disproportionately.


    # ======================================================================================= # 
    **Diagram 6 — Homoscedasticity vs. Heteroscedasticity:**

    GOOD: Homoscedastic (equal variance)     BAD: Heteroscedastic (unequal variance)

    Residual ↑                               Residual ↑
             │                                        │                     ●
           2 │  ●  ●    ●   ●  ●  ●                   │              ● ●  ●
             │    ●   ●   ●    ●                      │         ●  ●
           0 │──────────────────────────→ ŷ        0  │──────────────────────→ ŷ
             │     ●   ●   ●    ●                     │    ● ●
          -2 │  ●  ●    ●   ●  ●                      │
             │                                        │
    Residuals spread evenly around 0             Residuals fan out for larger ŷ
    at all predicted values. ✅                   Standard errors are unreliable. ⚠️
    
    # ======================================================================================= # 


Violating assumptions doesn't always break the model:
- For **prediction**: linearity matters most. The others matter far less.
- For **inference** (p-values, confidence intervals): all four matter.

Violating assumptions doesn't always break the model — but *which* assumptions matter
depends entirely on what you're trying to do. The three use cases have different
requirements:

 ───────────────────────────────────────────────────────────────────────────────
    Assumption          Pure Prediction   Unbiased Coefficients    Inference
 ───────────────────────────────────────────────────────────────────────────────
    L — Linearity       Critical ❗       Critical ❗              Critical ❗
    I — Independence    Matters           Critical ❗              Critical ❗
    N — Normality       Irrelevant ✓      Irrelevant ✓             Matters
    E — Equal Variance  Irrelevant ✓      Irrelevant ✓             Matters
 ───────────────────────────────────────────────────────────────────────────────

What this table means in practice:

- If you are **predicting house prices** and only care about RMSE on a test set:
  only Linearity matters significantly. You can violate I, N, and E freely and
  your test-set predictions will be fine.

- If you want to **interpret coefficients** ("a 1 sq ft increase adds $X to price"):
  Linearity and Independence matter. Normality and Equal Variance still don't affect
  whether the coefficients point in the right direction.

- If you want **confidence intervals and p-values** ("is this coefficient
  statistically significant?"): all four matter. Standard errors are only valid under
  the full set of Gauss-Markov assumptions. Heteroscedasticity, non-normality, and
  autocorrelation all invalidate classical inference.

The common mistake is applying inference-level assumptions to prediction tasks and
concluding the model is broken when it isn't — or vice versa, using a model for
inference when its assumptions are violated.

---

## Part 6: Regularisation — Preventing Overfitting

When you train a model, it learns patterns from your training data. But sometimes it learns too well — it memorizes the 
training data including all its noise and quirks, and then performs poorly on new unseen data. This is overfitting.

Think of it like a student who memorizes every past exam paper word for word, but then fails when the real exam asks 
slightly different questions. They over-fitted to the past papers.

A model that overfits will have:

Very low error on training data
Very high error on test/unseen data

        Total Loss = Original Loss + λ * Penalty Term

Where λ (lambda) is a hyperparameter that controls how strongly you want to regularize. 
A higher lambda means stronger regularization.


What Regularization Does
Regularization adds a penalty term to the loss function to discourage the model from becoming too complex. 
It essentially says to the model: "Yes, fit the data — but don't go overboard."

The general idea looks like this:

Overfitting = model memorises training noise. Fix by adding a penalty term to the loss:

| Method         | Penalty        | Effect                                          |
|----------------|----------------|-------------------------------------------------|
| **Ridge (L2)** | `λ Σ wᵢ²`      | Shrinks weights, keeps all features             |
| **Lasso (L1)** | `λ Σ |wᵢ|`     | Forces some weights to zero (feature selection) |
| **ElasticNet** | Mix of L1 + L2 | Best of both                                    |



**The Bias-Variance Tradeoff:**

Every model's generalisation error can be decomposed as:

                Total Error = Bias² + Variance + Irreducible Noise

- **Bias**: error from wrong assumptions (e.g., fitting a line to a quadratic curve).

        High bias = underfitting.

- **Variance**: error from sensitivity to small fluctuations in training data.

        High variance = overfitting.

- **Irreducible noise**: randomness in the data itself. Cannot be fixed.

Regularisation increases bias slightly (simpler model) to reduce variance a lot
(more stable model). The net effect is better generalisation.


    # ======================================================================================= # 
    **Diagram 7 — Overfitting vs Underfitting vs Just Right:**

    y ↑                  y ↑                  y ↑
      │  •               │  •                 │  •
      │    •  •          │ /  •  •            │    /‾‾•
      │       •          │/       •           │   / •
      │•   •             │•   •               │• /   •
      └──────────→ x     └──────────→ x       └──────────→ x

    UNDERFITTING               OVERFITTING          JUST RIGHT
    (High Bias)            (High Variance)       (Balanced)
    Line too simple        Wiggly curve fits      Smooth curve
    Misses the pattern     every training point   captures the signal
                           Fails on new data.     Generalises well.
                           
    # ======================================================================================= # 

### Ridge Regression (L2 Regularisation):

        Total Loss = Original Loss + λ * Σ(weights²)

L2 Regularization (Ridge) adds the sum of the squared values of the weights as the penalty:

Add the sum of squared weights as a penalty to the MSE loss:

        L_Ridge = MSE + λ Σⱼ wⱼ²   (sum excludes the bias w₀)

Where λ (lambda) is the regularisation strength (a hyperparameter you choose):
- λ = 0: pure linear regression, no regularisation
- λ → ∞: all weights forced to zero, model predicts only the mean

The gradient of the Ridge loss adds a term to the standard gradient:

        ∇L_Ridge = (2/n) Xᵀ(Xw - y)  +  2λw

The closed-form Ridge solution (a modified Normal Equation) is:

        w* = (XᵀX + λI)⁻¹ Xᵀy

Adding λI to XᵀX before inverting guarantees invertibility — solving the
collinearity problem mentioned in Part 3.

**Geometric Intuition:**

Ridge shrinks all weights toward zero proportionally but never forces any weight
to exactly zero. It keeps all features in the model, just with diminished influence.
Think of it as pulling all weights toward the origin simultaneously.

### Lasso Regression (L1 Regularisation):

    Total Loss = Original Loss + λ * Σ|weights|

L1 Regularization (Lasso) adds the sum of the absolute values of the weights as the penalty:

Replace the squared penalty with the sum of absolute values:

        L_Lasso = MSE + λ Σⱼ |wⱼ|

The key difference: L1 creates sparse solutions — it forces some weights to exactly
zero, performing automatic feature selection. Features with zero weights are
completely excluded from the model.

Why does L1 produce zeros but L2 does not? Geometry:

    # ======================================================================================= # 
    **Diagram 8 — Why Lasso Produces Zeros (L1 vs L2 Geometry):**

    CONTOUR PLOT: Loss landscape + regularisation constraint

    The optimal solution lies where the MSE ellipse first touches the constraint region.

    RIDGE (L2) — constraint is a circle:     LASSO (L1) — constraint is a diamond:

    w₂ ↑                                     w₂ ↑
       │       MSE contours                     │       MSE contours
       │   ~~~~●~~~~                            │   ~~~~●~~~~
       │ ~~  ↗   ~~                             │ ~~  ↗   ~~
       │~~  ╱     ~~                            │~~  ╱     ~~
       │   ╱ ●                                  │   ╱
       │  (   ) ← circle                        │  ╱ ← diamond corner!
       │   ╲   ╱                                │◆
       │    ╲ ╱                                 │  ╲
       │─────●─────→ w₁                         │───●───────→ w₁
                                                     ↑
                                               Optimum hits the CORNER
                                               at exactly (w₁=0, w₂=some value)
                                               → sparse solution!

    The L2 (circle) constraint is smooth — the optimum rarely hits the axes.
    The L1 (diamond) constraint has corners ON the axes — the optimum often
    lands exactly on a corner, making that weight exactly zero.
    # ======================================================================================= # 


**When to use Ridge vs Lasso:**

| Scenario                              | Use      | Reason                              |
|---------------------------------------|----------|-------------------------------------|
| All features probably relevant        | Ridge    | Keep all, just shrink them          |
| Suspect many features are irrelevant  | Lasso    | Automatic zero = feature selection  |
| High p (many features), high n        | Either   | Both work well                      |
| Features are highly correlated        | Ridge    | Lasso picks one arbitrarily         |
| Want interpretable sparse model       | Lasso    | Fewer non-zero weights = clearer    |
| Unsure                                | ElasticNet| Mix of both, robust                |


### ElasticNet:

Combines both penalties:

        L_ElasticNet = MSE + λ [ρ Σ|wⱼ| + (1-ρ)/2 Σwⱼ²]

Where ρ (rho) controls the mix between L1 and L2. When ρ=1, it's pure Lasso;
when ρ=0, it's pure Ridge. ElasticNet inherits the sparsity of Lasso and the
grouping behaviour of Ridge (correlated features get similar non-zero weights).


**Other Forms of Regularization**

Beyond L1 and L2, regularization is a broader concept that appears in many forms across machine learning:

Dropout is used in neural networks. During training, random neurons are switched off at each step, 
forcing the network to not rely too heavily on any single neuron. This prevents co-dependency between neurons.

Early Stopping monitors validation loss during training and stops the training process before the model starts to overfit — 
even if it could keep training.

Data Augmentation artificially increases the size and variety of your training data (flipping images, adding noise, etc.), 
making it harder for the model to memorize the training set.

Batch Normalization normalizes the inputs to each layer, which has a regularizing effect by reducing internal covariate shift.

---

### Part 7: Evaluation Metrics — How Good is Our Model ?

- **MSE** — Mean Squared Error (penalises outliers heavily)
- **RMSE** — Root MSE (same units as y)
- **MAE** — Mean Absolute Error (robust to outliers)
- **R²** — Proportion of variance explained (1.0 = perfect, 0 = predicts mean)

**Mean Squared Error (MSE):**

        MSE = (1/n) Σ (yᵢ - ŷᵢ)²

Units: squared units of y (if y is in $, MSE is in $²). Hard to interpret directly.
Use for: loss function during training; comparing models on the same dataset.


**Root Mean Squared Error (RMSE):**

        RMSE = √MSE

Units: same as y. A RMSE of $20k means our average prediction error is ~$20k.
Use for: reporting model quality in human-readable terms.

**Mean Absolute Error (MAE):**

        MAE = (1/n) Σ |yᵢ - ŷᵢ|

More robust to outliers than MSE (errors aren't squared, so extreme errors don't
dominate). Easier to interpret: MAE of $15k means our predictions are off by
$15k on average.

**R² (Coefficient of Determination):**

        R² = 1 - SS_res / SS_tot
           = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²

Where ȳ is the mean of y. Interpretation:
- R² = 1.0: perfect fit, model explains 100% of variance
- R² = 0.0: model is no better than predicting the mean every time
- R² < 0:   model is worse than predicting the mean (severe underfitting)

Practical caution: R² always increases when you add more features to the model,
even if they're irrelevant. Use Adjusted R² when comparing models with different
numbers of features:

        Adjusted R² = 1 - (1 - R²)(n-1)/(n-p-1)

This penalises adding features that don't improve prediction.


    # ======================================================================================= # 
    **Diagram 9 — Visualising R²:**

    SS_tot: Total variance                     SS_res: Residual variance
    (how much y varies from its mean)          (how much error the model has LEFT)

    y ↑                                        y ↑
      │    ●     ↕                               │    ●   ↕
      │   ●  ↕   ← total                        │   ● ↕  ← residual
    ȳ │──────────── spread                    ŷ │────●──── error
      │  ●   ↕   from mean                      │  ●  ↕   after fit
      │ ●    ↕                                  │ ●   ↕
      └──────────→ x                            └──────────→ x

    R² = 1 - (SS_res / SS_tot) = "fraction of variance the model explains"

    ──────────────────────────────────────────────────────
    R² value      Interpretation
    ──────────────────────────────────────────────────────
    0.95 – 1.0    Excellent (may indicate overfitting)
    0.80 – 0.95   Very good
    0.60 – 0.80   Moderate
    0.40 – 0.60   Weak
    < 0.40        Poor (consider non-linear models)
    ───────────────────────────────────────────────────────
    # ======================================================================================= # 

---

### Part 8: From Linear Regression to Deep Learning

        Linear Regression
              ↓
        Logistic Regression (classification boundary)
              ↓
        SVM (maximum margin classifier)
              ↓
        Decision Trees → Ensembles
              ↓
        Neural Networks (universal approximator)


**The Learning Path:**

Linear regression is not just a standalone tool — it is the conceptual ancestor of
all of machine learning. Understanding it deeply means you understand the skeleton
of every model that comes after.

    Linear Regression            ← what we learned here
          │
          │  (add sigmoid activation)
          ↓
    Logistic Regression          ← same model, different output activation
          │                         predicts probabilities instead of continuous values
          │  (use max-margin objective)
          ↓
    Support Vector Machine       ← same linear hyperplane, different loss function
          │
          │  (stack many linear models + non-linear activations)
          ↓
    Neural Networks              ← many perceptrons (each = linear regression + activation)
          │                         stacked in layers
          │  (add convolutional filters)
          ↓
    Convolutional Neural Nets    ← shared-weight linear models over local patches
          │
          │  (add attention mechanism)
          ↓
    Transformers                 ← attention = weighted linear combination of values


Every step adds one idea. But the core — a weighted sum of inputs — never disappears.


**Why Every Neuron in a Deep Network IS Linear Regression:**

A single neuron in a neural network computes exactly:

                    output = activation(w · x + b)

With no activation function (or a linear one), this is pure linear regression.
The activation function (ReLU, sigmoid, etc.) is the only thing making a neuron
"non-linear." Remove it, and you have a linear regression unit.

This is why a deep neural network with no activation functions (all linear layers)
collapses to a single linear transformation — stacking linear operations only ever
produces another linear operation. Non-linear activations are what give neural
networks their expressive power.


**The Fundamental Difference: Loss Functions and Activation:**

    Linear Regression    :    ŷ = w·x + b                   (raw output, no activation)
                              Loss = MSE

    Logistic Regression  :    ŷ = σ(w·x + b)                (sigmoid activation)
                              Loss = Binary Cross-Entropy

    Perceptron           :    ŷ = step(w·x + b)             (step activation)
                              Loss = Perceptron rule (not gradient descent)

    Neural Network Neuron:    output = ReLU(w·x + b)        (ReLU activation)
                              Loss = task-dependent (MSE for regression, CE for classification)

Same structure. Different choices. Each choice is optimal for a specific task.



##### Extra

## **Simple Linear Regression vs Bayesian Linear Regression**

The best way to understand the difference is through how each one thinks about uncertainty.

**Simple Linear Regression — "Give me one answer"**

Simple Linear Regression is a deterministic approach. You feed it data, it crunches the OLS formula, 
and it hands you back one fixed number for the slope and one fixed number for the intercept. 
That's it. It's confident and committed — it gives you a single best-fit line and never looks back.

For example, after training it might tell you:
Slope     = 2.847
Intercept = 5.123

These are point estimates. There is no sense of "maybe the slope is somewhere between 2.5 and 3.1" — it just picks one 
value and moves on. If your data is noisy or limited, this can be dangerously overconfident because the model doesn't 
acknowledge that uncertainty at all.


**Bayesian Linear Regression — "Give me a range of plausible answers"**

Bayesian Linear Regression takes a completely different philosophical stance. Instead of treating slope and intercept 
as fixed unknowns to be solved, it treats them as random variables with probability distributions. 
It never gives you one answer — it gives you a distribution of answers, each with a probability attached.

**It works in three stages:**

**1. Prior** — Before seeing any data, you state your belief about what the slope and intercept might be. 
In the Pyro code you shared, this looked like:

slope     = pyro.sample("slope",     dist.Normal(0., 10.))
intercept = pyro.sample("intercept", dist.Normal(0., 10.))

This says "I believe the slope is somewhere around 0, but it could reasonably be anywhere within a wide range." 
This is your starting belief before the data speaks.

**2. Likelihood** — The model then looks at the actual data and asks: given this data, how probable are different 
values of slope and intercept? This updates your beliefs.

**3. Posterior** — After combining the prior and the data, you get a posterior distribution — your updated belief 
about what the slope and intercept are. Instead of slope = 2.847, you get something like 
slope ~ Normal(mean=2.85, std=0.12), which tells you both the best guess AND how uncertain you are about it.

THE CORE PHILOSOPHICAL DIFFERENCE
(Simple Linear Regression vs Bayesian Linear Regression)

    
    COMPONENT                │ SIMPLE LINEAR REGRESSION            │ BAYESIAN LINEAR REGRESSION
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────
    Parameters               │ Fixed single values (point estimate)│ Probability distributions over weights
                             │ w, b are deterministic              │ w, b are random variables
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────
    Output                   │ One regression line                 │ A family of plausible regression lines
                             │ Single best-fit solution            │ Posterior distribution of functions
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────
    Uncertainty              │ Ignored                             │ Explicitly modeled
                             │ No uncertainty quantification       │ Predictive uncertainty intervals
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────
    Prior Knowledge          │ Not used                            │ Incorporated via prior distributions
                             │ Data-only learning                  │ Prior + Data → Posterior
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────
    Data Requirement         │ Works well with large datasets      │ Handles small data more gracefully
                             │ Needs sufficient samples            │ Priors stabilize limited data regimes
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────
    Computation              │ Closed-form OLS or GD               │ Inference engine required
                             │ (XᵀX)⁻¹Xᵀy                          │ SVI, MCMC, or Variational Inference
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────
    Interpretability         │ Very easy                           │ More complex
                             │ Direct coefficients                 │ Must interpret posterior distributions
    ─────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────

A Real-World Analogy
Imagine you're trying to estimate how much a house is worth based on its size.

Simple Linear Regression is like a junior appraiser who looks at the data and confidently says: "It's worth exactly £450,000."

Bayesian Linear Regression is like a senior appraiser who says: "Based on what I already know about the market (prior), 
and looking at this data, 
I'd say it's most likely around £450,000 — but it could reasonably be anywhere from £420,000 to £480,000, 
and here's the full probability breakdown."

The Bayesian approach is more honest about what it doesn't know.

**When to use which**

Use Simple Linear Regression when you have plenty of clean data, you need a fast and interpretable result, 
and you don't need to quantify uncertainty.

Use Bayesian Linear Regression when you have limited or noisy data, you want to incorporate domain knowledge through 
priors, or your decisions depend on understanding the range of possible outcomes rather than just a single best guess — 
for example in medicine, finance, or scientific research.

"""
# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ──────────────────────────────────────────────────────────────────────────
    Component              Complexity        Notes
    ──────────────────────────────────────────────────────────────────────────
    Normal Equation        O(p³ + np²)       p = num features; breaks for large p
    Gradient Descent       O(n·p) per iter   Scales to millions of features
    Ridge/Lasso (sklearn)  O(n·p) iter       Uses coordinate descent or SGD
    Prediction (inference) O(p)              One dot product per example
    ──────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Runnable code demonstrations
# ─────────────────────────────────────────────────────────────────────────────
"""
================================================================================
  ADDITIONS TO: supervised/topics/01_linear_regression.py
================================================================================

HOW TO INTEGRATE
─────────────────────────────────────────────────────────────────────────────
  1. Open supervised/topics/01_linear_regression.py
  2. Find the OPERATIONS dict — it ends with a closing  }
  3. Paste each block below BEFORE that final closing brace  }
  4. Make sure every existing operation block ends with a comma  },
     before you paste the new ones.

WHAT THIS DOES
─────────────────────────────────────────────────────────────────────────────
  Adds one OPERATIONS entry per regression implementation file.
  When you click ▶️ Run in the Step-by-Step tab, the app reads that
  implementation file from disk (relative to CWD = app.py's directory)
  and executes it, streaming its full output to the UI.

  Each operation:
    • Resolves the file path at runtime using os.getcwd()
    • Checks the file exists and shows a clear error if not
    • Exec's it in an isolated namespace so there's no variable leakage
    • Prints the file path it ran, for easy debugging

FOLDER STRUCTURE ASSUMED (matches your image)
─────────────────────────────────────────────────────────────────────────────
  app.py
  supervised/
    Implementation/
      Regression/
        00_Simple Linear Regression.py
        01_Bayesian Linear Regression.py
        03_Ridge Regression-L2.py
        04_Lasso Regression-L1.py
        05_Elastic Net-L1andL2Comb.py
    topics/
      01_linear_regression.py   ← you are editing this file
================================================================================
"""


VISUAL_HTML = ""  # Add your HTML visual breakdown here


OPERATIONS = {

    "Linear Regression from Scratch": {
        "description": "Full gradient descent implementation with step-by-step commentary",
        "runnable": True,
        "code": '''
"""
================================================================================
LINEAR REGRESSION FROM SCRATCH — GRADIENT DESCENT
================================================================================

We build everything from first principles using only NumPy.
No sklearn, no black boxes. Every step is explained.

Architecture:
                                     ŷᵢ = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ
    x₁ ──(w₁)──┐
               │
    x₂ ──(w₂)──┼──► [ Σ weighted sum + bias w₀ ] ──► ŷ  (no activation!)
               │
    xₚ ──(wₚ)───┘

Loss:   MSE = (1/n) Σ (yᵢ - ŷᵢ)²
Update: w  := w - α · ∇L(w)        where ∇L = (2/n) Xᵀ(Xw - y)

================================================================================
"""

import numpy as np

# =============================================================================
# DATA GENERATION
# =============================================================================
# True relationship: y = 4 + 3x + noise
# After training, we expect w₀ ≈ 4 (bias) and w₁ ≈ 3 (weight)

np.random.seed(42)
n = 100
X_raw = 2 * np.random.rand(n, 1)             # 100 points in [0, 2]
y = 4 + 3 * X_raw + np.random.randn(n, 1)    # y = 4 + 3x + Gaussian noise

# Add bias column (column of 1s) to X so bias is handled as just another weight
# X_b has shape (100, 2): column 0 = all 1s (for bias), column 1 = x values
X_b = np.c_[np.ones((n, 1)), X_raw]

print("=" * 60)
print("  LINEAR REGRESSION FROM SCRATCH")
print("=" * 60)
print(f"\\n  Dataset: {n} points generated from y = 4 + 3x + noise")
print(f"  X_b shape: {X_b.shape}   (100 examples, 2 columns: [1, x])")
print(f"  y shape:   {y.shape}")


# =============================================================================
# GRADIENT DESCENT IMPLEMENTATION
# =============================================================================

class LinearRegressionGD:
    """
    Linear Regression trained with Batch Gradient Descent.

    Parameters (learnable):
        self.weights : np.array of shape (p+1, 1)
                       weights[0] = bias (w₀)
                       weights[1:] = feature weights (w₁, w₂, ...)

    Hyperparameters (not learned, set by us):
        learning_rate : how large each gradient step is
        n_iterations  : how many steps to take
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        # ─── Hyperparameters ────────────────────────────────────────────────
        self.lr      = learning_rate
        self.n_iter  = n_iterations

        # ─── Learnable parameters (initialised to None, set in fit()) ───────
        self.weights = None

        # ─── Training history (for diagnostics) ─────────────────────────────
        self.loss_history = []

    def _mse(self, X, y):
        """Compute Mean Squared Error for current weights."""
        residuals = X @ self.weights - y          # shape (n, 1)
        return float(np.mean(residuals ** 2))

    def _gradient(self, X, y):
        """
        Compute MSE gradient with respect to weights.

        ∇L = (2/n) Xᵀ(Xw - y)

        Shape analysis:
            X      : (n, p+1)
            w      : (p+1, 1)
            Xw - y : (n, 1)   ← residuals
            Xᵀ     : (p+1, n)
            Xᵀ(Xw-y): (p+1, 1) ← one gradient value per weight
        """
        n = len(X)
        residuals = X @ self.weights - y          # (n, 1)
        return (2 / n) * X.T @ residuals          # (p+1, 1)

    def fit(self, X, y, verbose=True):
        """
        Train the model using gradient descent.

        Args:
            X (np.array): Design matrix, shape (n, p+1), with bias column prepended
            y (np.array): Target values, shape (n, 1)
            verbose (bool): Print progress every 100 iterations
        """
        # ─── Initialise weights randomly ────────────────────────────────────
        # Shape: (p+1, 1) — one weight per column of X (including bias)
        self.weights = np.random.randn(X.shape[1], 1)
        print(f"\\n  Initial weights: {self.weights.flatten().round(3)}")
        print(f"  Initial MSE: {self._mse(X, y):.4f}")
        print()

        # ─── Main gradient descent loop ─────────────────────────────────────
        for i in range(self.n_iter):

            # Compute gradient
            grad = self._gradient(X, y)

            # Update weights: w := w - α · ∇L
            self.weights -= self.lr * grad

            # Track loss every 100 steps
            if i % 100 == 0:
                loss = self._mse(X, y)
                self.loss_history.append(loss)
                if verbose:
                    print(f"  Iter {i:>4} | MSE: {loss:.4f} | "
                          f"w₀={self.weights[0,0]:.3f}, w₁={self.weights[1,0]:.3f}")

    def predict(self, X):
        """Make predictions: ŷ = Xw"""
        return X @ self.weights


# =============================================================================
# TRAIN AND EVALUATE
# =============================================================================

model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
model.fit(X_b, y)

print("\\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\\n  Learned  | bias={model.weights[0,0]:.3f}, weight={model.weights[1,0]:.3f}")
print(f"  True     | bias=4.000, weight=3.000")
print(f"  Final MSE: {model._mse(X_b, y):.4f}")

# Predict on a new point
x_new = np.array([[1, 1.5]])   # x = 1.5, with bias column prepended
y_pred = model.predict(x_new)
y_true = 4 + 3 * 1.5
print(f"\\n  Prediction at x=1.5:")
print(f"    Model:  ŷ = {y_pred[0,0]:.3f}")
print(f"    True:   y ≈ {y_true:.3f}  (before noise)")


# =============================================================================
# STEP-BY-STEP SINGLE ITERATION TRACE
# =============================================================================
print("\\n" + "=" * 60)
print("  STEP-BY-STEP: ONE GRADIENT DESCENT ITERATION")
print("=" * 60)

# Use tiny dataset for clarity
X_tiny = np.array([[1, 1],
                    [1, 2],
                    [1, 3]])          # column 0 = bias 1s, column 1 = x
y_tiny = np.array([[3], [5], [7]])   # y = 1 + 2x (true: w₀=1, w₁=2)

w = np.array([[0.0], [1.0]])         # initial weights
lr = 0.01

print(f"""
  Dataset (3 points, true: y = 1 + 2x):
    x = [1, 2, 3],  y = [3, 5, 7]

  Initial weights: w₀ = {w[0,0]},  w₁ = {w[1,0]}
  Learning rate: α = {lr}
""")

preds = X_tiny @ w
residuals = preds - y_tiny
mse = float(np.mean(residuals ** 2))
grad = (2 / 3) * X_tiny.T @ residuals
w_new = w - lr * grad

print(f"  STEP 1 — Forward Pass (predictions):")
for i in range(3):
    print(f"    ŷ_{i+1} = {w[0,0]}×1 + {w[1,0]}×{i+1} = {preds[i,0]:.2f}"
          f"   residual = {y_tiny[i,0]} - {preds[i,0]:.2f} = {residuals[i,0]:.2f}")

print(f"\\n  STEP 2 — Loss:")
print(f"    MSE = (1/3) × ({residuals[0,0]:.2f}² + {residuals[1,0]:.2f}² + {residuals[2,0]:.2f}²)")
print(f"    MSE = {mse:.4f}")

print(f"\\n  STEP 3 — Gradient:")
print(f"    ∂L/∂w₀ = {grad[0,0]:.4f}")
print(f"    ∂L/∂w₁ = {grad[1,0]:.4f}")

print(f"\\n  STEP 4 — Weight Update (w := w - α·∇L):")
print(f"    w₀ := {w[0,0]:.4f} - {lr} × {grad[0,0]:.4f} = {w_new[0,0]:.4f}")
print(f"    w₁ := {w[1,0]:.4f} - {lr} × {grad[1,0]:.4f} = {w_new[1,0]:.4f}")

preds_new = X_tiny @ w_new
mse_new = float(np.mean((preds_new - y_tiny) ** 2))
print(f"\\n  STEP 5 — New MSE: {mse_new:.4f}  (was {mse:.4f} — loss decreased ✓)")
''',
    },

    "Normal Equation Solution": {
        "description": "Closed-form analytical solution — no iterations, no learning rate",
        "runnable": True,
        "code": '''
"""
================================================================================
NORMAL EQUATION — ANALYTICAL SOLUTION
================================================================================

The Normal Equation finds the exact optimal weights in ONE step using algebra.
No iterations. No learning rate. No convergence concerns.

Formula:   w* = (XᵀX)⁻¹ Xᵀy

This is derived by setting the gradient of MSE to zero:
    ∇L = (2/n) Xᵀ(Xw - y) = 0
    XᵀXw = Xᵀy
    w* = (XᵀX)⁻¹ Xᵀy

When to use:
    ✓ Number of features p < ~10,000
    ✓ When you need exact solution without tuning a learning rate
    ✗ Avoid for large p (matrix inversion is O(p³))
    ✗ Avoid when XᵀX is singular (use Ridge instead)

================================================================================
"""

import numpy as np
import time

# =============================================================================
# SIDE-BY-SIDE COMPARISON: Normal Equation vs. Gradient Descent
# =============================================================================

np.random.seed(42)
n = 200
X_raw = 2 * np.random.rand(n, 1)
y = 4 + 3 * X_raw + np.random.randn(n, 1)
X_b = np.c_[np.ones((n, 1)), X_raw]    # add bias column

print("=" * 65)
print("  NORMAL EQUATION vs GRADIENT DESCENT")
print("=" * 65)

# ─── Method 1: Normal Equation ───────────────────────────────────────────────
t0 = time.perf_counter()

XtX = X_b.T @ X_b                           # shape (2, 2)
XtX_inv = np.linalg.inv(XtX)                # (XᵀX)⁻¹
Xty = X_b.T @ y                             # shape (2, 1)
w_normal = XtX_inv @ Xty                    # w* = (XᵀX)⁻¹Xᵀy

t_normal = (time.perf_counter() - t0) * 1000

print(f"\\n  Normal Equation:")
print(f"    w₀ (bias)  = {w_normal[0,0]:.6f}  (true: 4.0)")
print(f"    w₁ (weight)= {w_normal[1,0]:.6f}  (true: 3.0)")
print(f"    Time: {t_normal:.3f} ms")

# ─── Method 2: NumPy lstsq (numerically stable alternative to inv) ────────────
# lstsq uses SVD under the hood — more robust than explicit inversion
w_lstsq, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

print(f"\\n  Normal Equation (lstsq / SVD — numerically stable):")
print(f"    w₀ (bias)  = {w_lstsq[0,0]:.6f}")
print(f"    w₁ (weight)= {w_lstsq[1,0]:.6f}")

# ─── Method 3: Gradient Descent for comparison ───────────────────────────────
t0 = time.perf_counter()

w_gd = np.random.randn(2, 1)
for _ in range(2000):
    grad = (2 / n) * X_b.T @ (X_b @ w_gd - y)
    w_gd -= 0.1 * grad

t_gd = (time.perf_counter() - t0) * 1000

print(f"\\n  Gradient Descent (2000 iterations, α=0.1):")
print(f"    w₀ (bias)  = {w_gd[0,0]:.6f}")
print(f"    w₁ (weight)= {w_gd[1,0]:.6f}")
print(f"    Time: {t_gd:.3f} ms")


# =============================================================================
# DEMONSTRATE THE COST OF NORMAL EQUATION AT SCALE
# =============================================================================

print("\\n" + "=" * 65)
print("  SCALING COMPARISON: O(p³) vs O(n·p)")
print("=" * 65)
print("""
  The Normal Equation becomes infeasible for large p (many features).
  Below we measure actual inversion time for different p values.
""")

for p in [10, 100, 500, 1000]:
    X_big = np.random.rand(1000, p)
    XtX_big = X_big.T @ X_big

    t0 = time.perf_counter()
    try:
        _ = np.linalg.inv(XtX_big)
        t_inv = (time.perf_counter() - t0) * 1000
        print(f"  p={p:>5} features | Inversion time: {t_inv:>8.2f} ms")
    except np.linalg.LinAlgError:
        print(f"  p={p:>5} features | SINGULAR MATRIX — inversion failed!")

print("""
  → For p > 10,000, inversion takes seconds to minutes.
  → For modern deep learning (p = millions), use gradient descent.

  Rule of thumb:
    p < 10,000   → Normal Equation is fine
    p >= 10,000  → Use Gradient Descent
""")


# =============================================================================
# SINGULAR XtX — WHEN NORMAL EQUATION BREAKS
# =============================================================================

print("=" * 65)
print("  EDGE CASE: Singular XᵀX (collinear features)")
print("=" * 65)
print("""
  If two features are perfectly correlated, XᵀX is singular (non-invertible).
  The Normal Equation fails. Ridge regression fixes this.
""")

# Create perfectly collinear data (x₂ = 2·x₁)
X_collinear = np.c_[np.ones(50), np.random.rand(50), np.zeros(50)]
X_collinear[:, 2] = 2 * X_collinear[:, 1]   # x₂ = 2x₁ → linear dependence

XtX_col = X_collinear.T @ X_collinear
print(f"  Determinant of XᵀX (collinear): {np.linalg.det(XtX_col):.6f}")
print(f"  (Near zero → nearly singular)")

# Ridge fix: add λI to XᵀX before inverting
lambda_ = 1.0
XtX_ridge = XtX_col + lambda_ * np.eye(3)
print(f"  Determinant after Ridge (λ={lambda_}): {np.linalg.det(XtX_ridge):.4f}")
print(f"  (Non-zero → invertible again ✓)")
''',
    },

    "Ridge & Lasso Regularisation": {
        "description": "Compare L2 and L1 regularisation — shrinkage, sparsity, and the bias-variance tradeoff",
        "runnable": True,
        "code": '''
"""
================================================================================
REGULARISATION: RIDGE (L2) vs LASSO (L1) vs ELASTICNET
================================================================================

Overfitting: model learns training noise, fails on new data.
Fix: add a penalty term to the loss that punishes large weights.

    Ridge (L2):     Loss = MSE + λ Σ wᵢ²     → shrinks weights, keeps all features
    Lasso (L1):     Loss = MSE + λ Σ |wᵢ|    → forces some weights to exactly 0
    ElasticNet:     Loss = MSE + λ [ρ|w| + (1-ρ)w²]  → mix of both

Key question: why does L1 produce zeros but L2 doesn't? (Geometry, see THEORY.)

================================================================================
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# =============================================================================
# DATASET: 20 features, but only 5 are actually informative
# =============================================================================
# This simulates the real-world scenario: you have many candidate features
# but only a few genuinely predict the outcome.

n_samples  = 300
n_features = 20
n_informative = 5

# Generate true weights: 5 non-zero, 15 zero
true_weights = np.zeros(n_features)
true_weights[:n_informative] = [3.0, -2.5, 1.8, -1.2, 0.9]

# Generate features and target
X = np.random.randn(n_samples, n_features)
y = X @ true_weights + 2.0 + 0.5 * np.random.randn(n_samples)  # + bias 2.0

# Scale features (important for regularisation — unscaled features are unfair)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 70)
print("  REGULARISATION COMPARISON")
print("=" * 70)
print(f"\\n  Dataset: {n_samples} samples, {n_features} features")
print(f"  True non-zero weights: features 0–{n_informative-1}")
print(f"  True zero weights:     features {n_informative}–{n_features-1}  (noise features)")


# =============================================================================
# FIT ALL FOUR MODELS
# =============================================================================

models = {
    "OLS (no regularisation)": LinearRegression(),
    "Ridge  (L2, λ=1.0)":      Ridge(alpha=1.0),
    "Lasso  (L1, λ=0.1)":      Lasso(alpha=0.1, max_iter=10000),
    "ElasticNet (λ=0.1, ρ=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
}

print(f"\\n  {'Model':<30} | {'Train R²':>8} | {'Test R²':>8} | "
      f"{'Test RMSE':>10} | {'Zero weights':>14}")
print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*14}")

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_r2   = r2_score(y_train, y_train_pred)
    test_r2    = r2_score(y_test, y_test_pred)
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))
    n_zeros    = np.sum(np.abs(model.coef_) < 1e-4)

    results[name] = {"model": model, "coef": model.coef_.copy()}
    print(f"  {name:<30} | {train_r2:>8.4f} | {test_r2:>8.4f} | "
          f"{test_rmse:>10.4f} | {n_zeros:>6}/{n_features} zero")


# =============================================================================
# WEIGHT COMPARISON: TRUE vs LEARNED
# =============================================================================

print(f"\\n{'=' * 70}")
print(f"  LEARNED WEIGHTS vs TRUE WEIGHTS")
print(f"{'=' * 70}")
print(f"  (showing first 8 features)")
print()
print(f"  {'Feature':<10}", end="")
print(f"  {'True':>8}", end="")
for name in models:
    short = name.split("(")[0].strip()[:12]
    print(f"  {short:>12}", end="")
print()
print(f"  {'-'*10}", end="")
print(f"  {'--------':>8}", end="")
for _ in models:
    print(f"  {'------------':>12}", end="")
print()

for i in range(8):
    marker = "← real" if i < n_informative else "← noise"
    print(f"  w{i:<9}", end="")
    print(f"  {true_weights[i]:>8.3f}", end="")
    for name in models:
        w = results[name]["coef"][i]
        print(f"  {w:>12.4f}", end="")
    print(f"   {marker}")

print(f"""
  Key observations:
    OLS        — large weights on noise features (overfitting)
    Ridge (L2) — shrinks all weights but none to zero
    Lasso (L1) — zeros out noise features automatically ← sparse!
    ElasticNet — balance of both behaviours
""")


# =============================================================================
# λ SENSITIVITY: HOW REGULARISATION STRENGTH AFFECTS WEIGHTS
# =============================================================================

print("=" * 70)
print("  EFFECT OF λ ON RIDGE WEIGHTS (weight path)")
print("=" * 70)
print()
print(f"  {'λ':<10} | {'w₀':>8} | {'w₁':>8} | {'w₂':>8} | {'w₃':>8} | {'w₄':>8} | {'Test R²':>8}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    r = Ridge(alpha=lam).fit(X_train, y_train)
    r2 = r2_score(y_test, r.predict(X_test))
    coef = r.coef_
    print(f"  {lam:<10.3f} | {coef[0]:>8.4f} | {coef[1]:>8.4f} | "
          f"{coef[2]:>8.4f} | {coef[3]:>8.4f} | {coef[4]:>8.4f} | {r2:>8.4f}")

print(f"""
  As λ increases:
    → All weights shrink toward zero (L2 shrinkage)
    → Model becomes simpler (more bias, less variance)
    → Test R² first improves (reduces overfitting),
      then decreases (model becomes too simple / underfit)
    → The sweet spot is found by cross-validation
""")
''',
    },

    "Evaluation Metrics Deep Dive": {
        "description": "MSE, RMSE, MAE, and R² — understanding what each metric actually tells you",
        "runnable": True,
        "code": '''
"""
================================================================================
EVALUATION METRICS DEEP DIVE
================================================================================

Metrics tell us how good our model is — but each metric emphasises different
aspects of model quality. Knowing which to use and why is as important as
knowing how to train the model.

    MSE   = (1/n) Σ (yᵢ - ŷᵢ)²       penalises large errors heavily
    RMSE  = √MSE                       same units as y, more interpretable
    MAE   = (1/n) Σ |yᵢ - ŷᵢ|         robust to outliers
    R²    = 1 - SS_res/SS_tot          fraction of variance explained (0–1)

================================================================================
"""

import numpy as np

np.random.seed(42)

# =============================================================================
# DEMO DATASET: Predict salary from years of experience
# =============================================================================
years_exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
true_salary_k = np.array([45, 50, 55, 60, 68, 75, 82, 88, 95, 100], dtype=float)

# Three candidate models (already fitted, showing their predictions)
preds_good    = np.array([44, 51, 56, 61, 67, 74, 83, 87, 96, 101], dtype=float)
preds_biased  = np.array([40, 45, 50, 55, 63, 70, 77, 83, 90, 95], dtype=float)   # consistently low
preds_outlier = np.array([44, 51, 56, 61, 67, 74, 83, 87, 96, 135], dtype=float)  # one huge error


def compute_metrics(y_true, y_pred, model_name):
    """Compute and print all four metrics for a model."""
    residuals = y_true - y_pred
    n = len(y_true)

    mse  = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"  {model_name}")
    print(f"    MSE  = {mse:>8.2f}   (squared error — penalises outliers heavily)")
    print(f"    RMSE = {rmse:>8.2f}   (in $k — avg error ~${rmse:.1f}k)")
    print(f"    MAE  = {mae:>8.2f}   (in $k — avg absolute deviation ~${mae:.1f}k)")
    print(f"    R²   = {r2:>8.4f}   (explains {r2*100:.1f}% of variance)")
    print()
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


print("=" * 65)
print("  METRIC COMPARISON ACROSS THREE MODELS")
print("=" * 65)
print(f"  Task: Predict salary ($k) from years of experience\\n")

m1 = compute_metrics(true_salary_k, preds_good,    "Model 1: Good fit")
m2 = compute_metrics(true_salary_k, preds_biased,  "Model 2: Consistently biased (off by ~5k)")
m3 = compute_metrics(true_salary_k, preds_outlier, "Model 3: One catastrophic outlier (off by 35k)")


# =============================================================================
# WHY MSE AND MAE TELL DIFFERENT STORIES WITH OUTLIERS
# =============================================================================

print("=" * 65)
print("  OUTLIER SENSITIVITY: MSE vs MAE")
print("=" * 65)
print(f"""
  Model 2 (biased by 5k) vs Model 3 (one 35k outlier):

    Model 2:  MSE={m2["mse"]:.1f}, MAE={m2["mae"]:.1f}
    Model 3:  MSE={m3["mse"]:.1f}, MAE={m3["mae"]:.1f}

  Model 3 has a MUCH higher MSE (35k error → 35²=1225 contribution!)
  But Model 3's MAE may be similar to Model 2 across most points.

  Practical implication:
    → If outliers represent RARE GENUINE ERRORS you want to flag: use MSE
      (it screams loudest about big mistakes)
    → If outliers represent UNUSUAL BUT VALID data you don't want to
      dominate your loss: use MAE (treats all errors proportionally)
""")


# =============================================================================
# UNDERSTANDING R² — WHAT DOES "EXPLAINS VARIANCE" ACTUALLY MEAN?
# =============================================================================

print("=" * 65)
print("  DEEP DIVE: WHAT DOES R² ACTUALLY MEASURE?")
print("=" * 65)

y = true_salary_k
y_hat = preds_good
y_mean = np.mean(y)

ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_hat) ** 2)
ss_reg = np.sum((y_hat - y_mean) ** 2)  # variance explained BY the model
r2 = 1 - ss_res / ss_tot

print(f"""
  y_mean (baseline: always predict the mean) = {y_mean:.1f}k

  SS_tot = Σ(yᵢ - ȳ)²  = {ss_tot:.1f}   ← total variance in y
  SS_res = Σ(yᵢ - ŷᵢ)²  = {ss_res:.1f}   ← variance LEFT UNEXPLAINED by model
  SS_reg = Σ(ŷᵢ - ȳ)²  = {ss_reg:.1f}   ← variance EXPLAINED by the model

  R² = 1 - SS_res/SS_tot = 1 - {ss_res:.1f}/{ss_tot:.1f} = {r2:.4f}

  Interpretation: the model explains {r2*100:.1f}% of the variance in salary.
  The remaining {(1-r2)*100:.1f}% is either:
      (a) inherent randomness in salary (irreducible noise), or
      (b) signal we haven't captured yet (missing features)

  SS_tot = SS_reg + SS_res   (Decomposition of variance)
  {ss_tot:.1f}  =  {ss_reg:.1f}  +  {ss_res:.1f}   ← check: {ss_reg + ss_res:.1f} ✓
""")


# =============================================================================
# R² EDGE CASES
# =============================================================================

print("=" * 65)
print("  R² EDGE CASES")
print("=" * 65)

# Perfect predictions
preds_perfect = true_salary_k.copy()
ss_res_perfect = np.sum((true_salary_k - preds_perfect) ** 2)
r2_perfect = 1 - ss_res_perfect / ss_tot
print(f"  R² = {r2_perfect:.4f}  ← Perfect predictions (R²=1.0)")

# Always predict the mean
preds_mean = np.full_like(true_salary_k, y_mean)
ss_res_mean = np.sum((true_salary_k - preds_mean) ** 2)
r2_mean = 1 - ss_res_mean / ss_tot
print(f"  R² = {r2_mean:.4f}  ← Always predict mean (R²=0.0: baseline, no skill)")

# Terrible predictions (worse than mean)
preds_terrible = true_salary_k[::-1]  # reversed — actively wrong
ss_res_terrible = np.sum((true_salary_k - preds_terrible) ** 2)
r2_terrible = 1 - ss_res_terrible / ss_tot
print(f"  R² = {r2_terrible:.4f}  ← Terrible predictions (R² < 0: worse than mean!)")

print(f"""
  R² summary:
    1.0   → Perfect fit
    0.0   → Model has no predictive power (equivalent to just predicting ȳ)
    < 0   → Model is actively harmful — predictions worse than the mean baseline
    > 1.0 → Impossible (would require SS_res < 0)
""")
''',
    },

    # ── Trigger: Simple Linear Regression ───────────────────────────────────
    "▶ Run Module: Simple Linear Regression": {
        "description": (
            "Runs 00_Simple Linear Regression.py from the Implementation folder. "
            "Linear Regression from scratch — slope, intercept, MSE, R², and a regression plot."
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Regression"
    / "00_Simple Linear Regression.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Supervised_Learning/Regression/")
    # Path(os.getcwd()) / "Supervised_Learning" / "Implementation" / "Regression"
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },


    # ── Trigger: Bayesian Linear Regression ─────────────────────────────────
    "▶ Run Module: Bayesian Linear Regression": {
        "description": (
            "Runs 01_Bayesian Linear Regression.py from the Implementation folder. "
            "Probabilistic weight estimation, posterior distributions, and "
            "uncertainty quantification — contrasted against OLS."
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Regression"
    / "01_Bayesian Linear Regression.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Supervised_Learning/Regression/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },


    # ── Trigger: Ridge Regression (L2) ──────────────────────────────────────
    "▶ Run Module: Ridge Regression (L2)": {
        "description": (
            "Runs 03_Ridge Regression-L2.py from the Implementation folder. "
            "Closed-form (XᵀX + λI)⁻¹Xᵀy, weight shrinkage vs OLS, "
            "and how lambda controls the bias-variance tradeoff."
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Regression"
    / "03_Ridge Regression-L2.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Supervised_Learning/Regression/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },


    # ── Trigger: Lasso Regression (L1) ──────────────────────────────────────
    "▶ Run Module: Lasso Regression (L1)": {
        "description": (
            "Runs 04_Lasso Regression-L1.py from the Implementation folder. "
            "Coordinate descent, soft thresholding, automatic feature selection, "
            "and the regularisation path showing which features survive as λ increases."
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Regression"
    / "04_Lasso Regression-L1.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Supervised_Learning/Regression/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },

    # ── Trigger: Elastic Net - L1 and L2 Combination ──────────────────────────────────────
    "▶ Run Module: ElasticNetRegression": {
        "description": (
            "Runs 05_Elastic Net-L1andL2Combo.py from the Implementation folder. "
            "Coordinate descent, soft thresholding, automatic feature selection, "
            "and the regularisation path showing which features survive as λ increases."
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Regression"
    / "05_Elastic Net-L1andL2Comb.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Supervised_Learning/Regression/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },


# ── Trigger: Ridge Regression (L2) ──────────────────────────────────────
    "▶ Run Module: KNN Regression": {
        "description": (
            "Runs 06_KNN_Regression.py from the Implementation folder. "
            "Knn Regression Model Explanation"
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Supervised_Learning"
    / "Implementation"
    / "Regression"
    / "06_KNN_Regression.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Supervised_Learning/Regression/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },


}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Standalone demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np

    print("\n" + "=" * 65)
    print("  LINEAR REGRESSION: THE FOUNDATION OF MACHINE LEARNING")
    print("=" * 65)
    print("""
  This script demonstrates linear regression from the ground up.

  Key Concepts:
    • Weighted sum of features → continuous prediction (no activation)
    • MSE loss: penalises large errors quadratically
    • Normal Equation: exact closed-form solution in one step
    • Gradient Descent: iterative solution, scales to any size
    • Regularisation: Ridge (L2) shrinks weights; Lasso (L1) zeros them
    • R² measures how much variance the model explains
    """)

    np.random.seed(42)

    # ─── Tiny dataset: y = 2 + 5x ────────────────────────────────────────────
    X_raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y     = np.array([7.1, 12.0, 17.2, 21.9, 27.1])   # ≈ 2 + 5x + noise

    # Design matrix with bias column
    X_b = np.c_[np.ones(5), X_raw]

    print("=" * 65)
    print("  DATASET: y ≈ 2 + 5x  (5 training points)")
    print("=" * 65)
    print(f"\n  {'x':>6}   {'y (true)':>10}")
    print(f"  {'------':>6}   {'----------':>10}")
    for xi, yi in zip(X_raw, y):
        print(f"  {xi:>6.1f}   {yi:>10.1f}")

    # ─── Normal Equation ─────────────────────────────────────────────────────
    w_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    print(f"\n  Normal Equation solution:")
    print(f"    w₀ (bias)   = {w_normal[0]:.4f}  (true: 2.0)")
    print(f"    w₁ (weight) = {w_normal[1]:.4f}  (true: 5.0)")

    # ─── Gradient Descent ────────────────────────────────────────────────────
    w_gd = np.array([0.0, 0.0])
    lr   = 0.01
    for _ in range(5000):
        preds = X_b @ w_gd
        grad  = (2 / 5) * X_b.T @ (preds - y)
        w_gd -= lr * grad

    print(f"\n  Gradient Descent solution (5000 iterations, α={lr}):")
    print(f"    w₀ (bias)   = {w_gd[0]:.4f}  (true: 2.0)")
    print(f"    w₁ (weight) = {w_gd[1]:.4f}  (true: 5.0)")

    # ─── Predictions ─────────────────────────────────────────────────────────
    print(f"\n  Predictions using Normal Equation weights:")
    print(f"  {'x':>6}   {'y_true':>10}   {'y_hat':>10}   {'residual':>10}")
    print(f"  {'------':>6}   {'----------':>10}   {'----------':>10}   {'----------':>10}")
    for xi, yi in zip(X_raw, y):
        y_hat = w_normal[0] + w_normal[1] * xi
        print(f"  {xi:>6.1f}   {yi:>10.2f}   {y_hat:>10.4f}   {yi - y_hat:>10.4f}")

    mse = np.mean((y - X_b @ w_normal) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - X_b @ w_normal) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n  MSE  = {mse:.6f}")
    print(f"  RMSE = {np.sqrt(mse):.6f}")
    print(f"  R²   = {r2:.6f}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1. Linear regression predicts: ŷ = w₀ + w₁x₁ + ... + wₚxₚ  (no activation)
  2. MSE loss is a convex bowl — only one global minimum
  3. Normal Equation solves it exactly: w* = (XᵀX)⁻¹Xᵀy
     → Use when p (features) < ~10,000
  4. Gradient Descent solves it iteratively: w := w - α·∇L
     → Use for large-scale problems; requires tuning α
  5. Ridge (L2) shrinks all weights; Lasso (L1) zeros some out
  6. R² tells you how much of y's variance your model explains
  7. Every neuron in a deep network is linear regression + activation function
  8. Understanding this module deeply = understanding the skeleton of all ML
    """)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """\
    ──────────────────────────────────────────────────────────────────────────
    Solver / Component          Complexity          Notes
    ──────────────────────────────────────────────────────────────────────────
    Normal Equation             O(p³ + np²)         p = features; infeasible p>10k
    Gradient Descent (1 iter)   O(n·p)              n = samples; scales to any size
    Ridge closed-form           O(p³ + np²)         (XᵀX + λI)⁻¹ — always invertible
    Prediction (inference)      O(p)                single dot product
    ──────────────────────────────────────────────────────────────────────────
"""

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
        from Supervised_Learning.visuals.linear_regression_visual import (   # ← match your exact folder casing
            LR_VISUAL_HTML,
            LR_VISUAL_HEIGHT,
        )
        visual_html   = LR_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = LR_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[01_linear_regression.py] Could not load visual: {e}", stacklevel=2)

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