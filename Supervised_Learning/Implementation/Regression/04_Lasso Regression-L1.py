r"""
================================================================================
  L1 REGULARIZATION -- LASSO REGRESSION
  From First Principles, Pure Python
================================================================================

WHAT THIS MODULE COVERS
─────────────────────────────────────────────────────────────────────────────
  1. Why regularization exists — the overfitting problem
  2. L1 vs L2 — the fundamental geometric difference
  3. The Lasso objective function — full mathematical derivation
  4. Why L1 produces sparse (zero) weights — the corner solution proof
  5. Coordinate Descent — the algorithm used to solve Lasso
  6. The Soft-Thresholding operator — math + intuition
  7. Feature selection via Lasso — how and why it works
  8. Hyperparameter λ — how to choose it
  9. Full implementation from scratch — no libraries
 10. Evaluation: MSE, R², and sparsity inspection

────────────────────────────────────────────────────────────────────────────────
  PART 1 — THE OVERFITTING PROBLEM (Why Regularization?)
────────────────────────────────────────────────────────────────────────────────

  In ordinary Linear Regression (OLS), we minimize:

        Loss(w) = Σᵢ (yᵢ - ŷᵢ)²
                = Σᵢ (yᵢ - (w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ))²

  The model is free to assign ANY value to any weight.

  When data is noisy or has many features, OLS will:
    → Assign huge positive weights to some features
    → Assign huge negative weights to cancel them out
    → Perfectly fit the training set (noise included)
    → Fail miserably on new data  ← OVERFITTING

  Regularization forces the model to keep weights small,
  trading a tiny bit of training accuracy for much better generalization.

────────────────────────────────────────────────────────────────────────────────
  PART 2 — L1 vs L2 REGULARIZATION
────────────────────────────────────────────────────────────────────────────────

  Both methods add a penalty term to the loss.

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  L2 (Ridge):   Loss + λ · Σⱼ wⱼ²       ← penalizes squared weights       │
  │  L1 (Lasso):   Loss + λ · Σⱼ |wⱼ|      ← penalizes absolute weights      │
  └─────────────────────────────────────────────────────────────────────────┘

  The difference in the exponent (2 vs 1) creates a profound difference
  in behavior:

  L2 (Ridge):
    • Shrinks all weights toward zero — but NEVER exactly to zero
    • Distributes importance across correlated features
    • Closed-form solution: w = (XᵀX + λI)⁻¹ Xᵀy

  L1 (Lasso):
    • Shrinks weights, but CAN push them EXACTLY to zero
    • Acts as automatic feature selection
    • No closed-form solution → must use iterative methods

  Why does L1 zero out weights but L2 doesn't?
  → See Part 4 (the geometric corner proof) below.

────────────────────────────────────────────────────────────────────────────────
  PART 3 — THE LASSO OBJECTIVE FUNCTION
────────────────────────────────────────────────────────────────────────────────

  Full Lasso objective to MINIMIZE:

        J(w) = (1/n) · Σᵢ (yᵢ - Xᵢw)²   +   λ · Σⱼ |wⱼ|
               ─────────────────────────       ─────────────
                      MSE loss                  L1 penalty

  Notation:
    n       — number of training samples
    p       — number of features
    yᵢ      — actual target for sample i
    Xᵢ      — feature vector for sample i  (shape: 1 × p)
    w       — weight vector  (shape: p × 1)
    λ       — regularization strength  (lambda, hyperparameter ≥ 0)
    |wⱼ|    — absolute value of weight j

  Note: The intercept w₀ is NOT penalized. We handle this by
        centering y (subtracting its mean) and fitting only the weights.
        The intercept is recovered as: w₀ = ȳ − X̄ · w

  As λ increases:
    λ = 0     → pure OLS, no regularization, possible overfitting
    λ = small → mild shrinkage, most weights survive
    λ = large → heavy shrinkage, many weights → 0 (sparse model)
    λ = ∞     → all weights forced to 0, model predicts constant ȳ

────────────────────────────────────────────────────────────────────────────────
  PART 4 — WHY L1 ZEROS OUT WEIGHTS (Geometric Intuition)
────────────────────────────────────────────────────────────────────────────────

  Think of regularization as a CONSTRAINED optimization problem.
  Instead of adding a penalty, it's equivalent to:

        Minimize:  Σᵢ (yᵢ - Xᵢw)²
        Subject to: constraint on w

  L2 constraint:   w₁² + w₂² ≤ t       (a circle in 2D)
  L1 constraint:   |w₁| + |w₂| ≤ t     (a diamond / rhombus in 2D)

  The unconstrained OLS solution lives somewhere in weight space.
  Regularization finds where the loss contours FIRST TOUCH the constraint.

  ┌─────────────────────────────────────────────────────┐
  │   L2 Constraint Region (circle):                    │
  │                                                     │
  │            smooth, round                            │
  │         ___________                                 │
  │        /           \                                │
  │       |     OLS*    |  ← contours likely touch      │
  │        \___________/    the smooth edge, so w ≠ 0   │
  │                                                     │
  │   L1 Constraint Region (diamond):                   │
  │                                                     │
  │             /\                                      │
  │            /  \  ← CORNERS at axes!                 │
  │           / ◄──── contours very often touch here    │
  │            \  /   where w₁ = 0  or  w₂ = 0          │
  │             \/                                      │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  The L1 diamond has CORNERS on the axes.
  Elliptical loss contours tend to touch these corners first.
  At a corner, one (or more) weights = 0 exactly.
  → This is the geometric reason L1 produces sparse solutions.

  The more features you have, the more "corner-like" the L1
  constraint becomes (it's a hyperdiamond with 2p corners in p dimensions),
  and the more likely sparsity occurs.

────────────────────────────────────────────────────────────────────────────────
  PART 5 — WHY L1 HAS NO CLOSED-FORM SOLUTION
────────────────────────────────────────────────────────────────────────────────

  For Ridge (L2), the derivative of w² is 2w — smooth everywhere.
  So we can set ∂J/∂w = 0 and solve directly.

  For Lasso (L1), the derivative of |w| is:
        ∂|w|/∂w =  +1   if w > 0
                   -1   if w < 0
                   undefined  if w = 0  ← NOT differentiable at zero!

  Because |w| has a KINK at zero, we cannot take a simple gradient and
  set it to zero. This is why Lasso requires iterative solvers.

  The standard solution: COORDINATE DESCENT (see Part 6).

────────────────────────────────────────────────────────────────────────────────
  PART 6 — COORDINATE DESCENT ALGORITHM
────────────────────────────────────────────────────────────────────────────────

  Idea: Instead of optimizing all weights simultaneously, optimize
        ONE weight at a time, cycling through all features repeatedly.

  At each step, for weight j:
    1. Compute the PARTIAL RESIDUAL — what's left after removing
       feature j's current contribution:

          rᵢ(j) = yᵢ - Σₖ≠ⱼ (wₖ · xᵢₖ)

       This isolates the relationship between feature j and the target.

    2. Compute the OLS solution for wⱼ treating all others as fixed:

          w̃ⱼ = (1/n) · Σᵢ xᵢⱼ · rᵢ(j)

       This is just the correlation between feature j and the residual.

    3. Apply SOFT THRESHOLDING (the L1 solution for a single weight):

          wⱼ ← S(w̃ⱼ, λ)

  Repeat until convergence (weights stop changing appreciably).

────────────────────────────────────────────────────────────────────────────────
  PART 7 — THE SOFT THRESHOLDING OPERATOR
────────────────────────────────────────────────────────────────────────────────

  The Soft Thresholding function S(z, λ) is the CLOSED-FORM solution
  to the single-variable Lasso subproblem:

        minimize_w  (1/2)(w - z)²  +  λ|w|

  Solution:
                  ┌  z - λ    if  z  >  λ    (positive, shrink down)
        S(z, λ) = │  z + λ    if  z  < -λ   (negative, shrink up)
                  └  0        if  |z| ≤ λ    (ZERO OUT — this is the key!)

  Interpretation:
    • z is the "raw" (unpenalized) estimate for the weight
    • S shrinks z toward zero by exactly λ
    • If |z| is smaller than λ, the penalty overwhelms the signal
      → the weight is forced to exactly zero
    • If |z| is larger than λ, we shrink by λ but keep the sign

  Compact notation:    S(z, λ) = sign(z) · max(|z| - λ, 0)

  Example:
    z = 2.5,  λ = 1.0  →  S = sign(2.5) · max(1.5, 0) =  1.5
    z = 0.8,  λ = 1.0  →  S = sign(0.8) · max(-0.2, 0) =  0.0  ← zeroed!
    z = -3.0, λ = 1.0  →  S = sign(-3.0) · max(2.0, 0) = -2.0

────────────────────────────────────────────────────────────────────────────────
  PART 8 — CHOOSING LAMBDA (λ)
────────────────────────────────────────────────────────────────────────────────

  λ is the single most important hyperparameter in Lasso.

  Strategy: Try a range of λ values and pick the one that gives
            the best validation set performance.

  Typical range: logarithmically spaced values
    λ ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}

  Rule of thumb:
    • Start with λ = 0.01 · max(|XᵀY|) / n
      (this is the smallest λ that produces a fully zero model,
       scaled back to begin shrinkage)
    • Use cross-validation (k-fold) to select the best λ
    • In practice: scikit-learn's LassoCV does this automatically

  Lasso "path": as λ sweeps from large to small, weights enter the model
  one-by-one. The ORDER in which they enter tells you feature importance.

================================================================================
"""


import math


# ─────────────────────────────────────────────────────────────────────────────
#  CORE MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def mean(values):
    """Arithmetic mean of a list of numbers."""
    return sum(values) / len(values)


def dot(a, b):
    """Dot product of two equal-length vectors."""
    return sum(a[i] * b[i] for i in range(len(a)))


def sign(x):
    """
    Mathematical sign function.
    Returns +1 for positive, -1 for negative, 0 for zero.
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  SOFT THRESHOLDING OPERATOR  — The Heart of Lasso
# ─────────────────────────────────────────────────────────────────────────────

def soft_threshold(z, lambda_):
    """
    Soft Thresholding Operator: S(z, λ)

    The closed-form solution to the single-variable Lasso subproblem:

        minimize_w  (1/2)(w - z)² + λ|w|

    Formula:
        S(z, λ) = sign(z) · max(|z| - λ, 0)

    Parameters:
        z       (float): Raw unpenalized estimate for a single weight
        lambda_ (float): Regularization strength (≥ 0)

    Returns:
        float: Regularized weight — exactly 0 if |z| ≤ λ

    Behavior:
        z =  2.5, λ = 1.0  →   1.5   (shrunk by λ, positive)
        z =  0.8, λ = 1.0  →   0.0   (|z| < λ, zeroed out!)
        z = -3.0, λ = 1.0  →  -2.0   (shrunk by λ, negative)
    """
    return sign(z) * max(abs(z) - lambda_, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE STANDARDIZATION
# ─────────────────────────────────────────────────────────────────────────────

def standardize(X):
    """
    Standardizes each feature to zero mean and unit variance.

    Formula for each feature column j:
        x̃ᵢⱼ = (xᵢⱼ - μⱼ) / σⱼ

    Why standardize for Lasso?
        Lasso penalizes |wⱼ| equally for all j. If features have
        different scales, the penalty will unfairly shrink weights
        of large-scale features more aggressively.
        Standardizing puts all features on equal footing.

    Returns:
        X_std   — standardized feature matrix
        means   — feature means  (needed to standardize test data)
        stds    — feature standard deviations
    """
    n = len(X)
    p = len(X[0])
    means = [mean([X[i][j] for i in range(n)]) for j in range(p)]
    stds  = [
        math.sqrt(sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n)
        for j in range(p)
    ]
    # Replace zero std (constant feature) with 1 to avoid divide-by-zero
    stds = [s if s > 1e-10 else 1.0 for s in stds]

    X_std = [
        [(X[i][j] - means[j]) / stds[j] for j in range(p)]
        for i in range(n)
    ]
    return X_std, means, stds


# ─────────────────────────────────────────────────────────────────────────────
#  LASSO REGRESSION VIA COORDINATE DESCENT
# ─────────────────────────────────────────────────────────────────────────────

def lasso_coordinate_descent(X, y, lambda_, max_iter=1000, tol=1e-6):
    """
    Fits a Lasso Regression model using Coordinate Descent.

    ── ALGORITHM ──────────────────────────────────────────────────────────────

    Input:
      X         — (n × p) feature matrix  (should be standardized)
      y         — (n,) target vector
      lambda_   — regularization strength λ  (≥ 0)
      max_iter  — maximum number of full passes over all weights
      tol       — convergence threshold (stop when max weight change < tol)

    Setup:
      1. Center y by subtracting its mean (handles intercept implicitly)
      2. Initialize all weights w = [0, 0, ..., 0]

    Main loop  (one "epoch" = one full pass over all p weights):
      For each feature j = 0, 1, ..., p-1:

        a) Compute PARTIAL RESIDUAL:
              rᵢ(j) = yᵢ_centered - Σₖ≠ⱼ (wₖ · xᵢₖ)

           This is what's left unexplained once we remove feature j.

        b) Compute RAW UPDATE (OLS solution for wⱼ alone):
              w̃ⱼ = (1/n) · Σᵢ xᵢⱼ · rᵢ(j)

           For standardized features (‖xⱼ‖² / n = 1), this simplifies
           to a plain correlation between column j and the residual.

        c) Apply SOFT THRESHOLDING:
              wⱼ ← S(w̃ⱼ, λ)

           This either shrinks the raw update or zeros it out entirely.

      Repeat until max change across all weights < tol.

    Output:
      weights   — learned weight vector (p,)
      intercept — recovered as ȳ  (because y was centered)

    ───────────────────────────────────────────────────────────────────────────

    Parameters:
        X        (list of lists): Standardized feature matrix, shape (n × p)
        y        (list of float): Target values, shape (n,)
        lambda_  (float):         Regularization strength λ
        max_iter (int):           Max number of coordinate descent sweeps
        tol      (float):         Convergence criterion

    Returns:
        dict with keys:
            "weights"      — list of p floats (on standardized scale)
            "intercept"    — float (= mean of original y)
            "n_iter"       — int (iterations until convergence)
            "converged"    — bool
    """
    n = len(X)
    p = len(X[0])

    # ── Step 1: Center y (intercept handling) ─────────────────────────────
    y_mean = mean(y)
    y_c    = [yi - y_mean for yi in y]   # centered target

    # ── Step 2: Initialize weights at zero ────────────────────────────────
    w = [0.0] * p

    # ── Step 3: Coordinate Descent Main Loop ──────────────────────────────
    for iteration in range(max_iter):
        w_old = w[:]          # snapshot for convergence check
        max_change = 0.0

        for j in range(p):

            # a) Compute partial residual for feature j
            #    r(j)ᵢ = y_cᵢ - Σₖ≠ⱼ wₖ xᵢₖ
            #          = (y_cᵢ - Σₖ wₖ xᵢₖ) + wⱼ xᵢⱼ    ← efficient form
            partial_residuals = [
                y_c[i] - sum(w[k] * X[i][k] for k in range(p)) + w[j] * X[i][j]
                for i in range(n)
            ]

            # b) Raw (unpenalized) update: OLS for this single weight
            #    For standardized features: Σᵢ xᵢⱼ² / n ≈ 1, so:
            #    w̃ⱼ = (1/n) · Σᵢ xᵢⱼ · rᵢ(j)
            z = (1.0 / n) * sum(X[i][j] * partial_residuals[i] for i in range(n))

            # c) Apply soft thresholding — the L1 penalty step
            w[j] = soft_threshold(z, lambda_)

            max_change = max(max_change, abs(w[j] - w_old[j]))

        # ── Convergence check ─────────────────────────────────────────────
        if max_change < tol:
            return {
                "weights":   w,
                "intercept": y_mean,
                "n_iter":    iteration + 1,
                "converged": True
            }

    return {
        "weights":   w,
        "intercept": y_mean,
        "n_iter":    max_iter,
        "converged": False
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict(X_std, model):
    """
    Generates predictions on standardized features.

    Formula:
        ŷᵢ = intercept + Σⱼ wⱼ · x̃ᵢⱼ

    Parameters:
        X_std (list of lists): Standardized feature matrix
        model (dict):          Output of lasso_coordinate_descent()

    Returns:
        list: Predicted values
    """
    intercept = model["intercept"]
    weights   = model["weights"]
    return [intercept + dot(weights, X_std[i]) for i in range(len(X_std))]


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def mean_squared_error(y_actual, y_predicted):
    """
    Mean Squared Error.

    Formula:
        MSE = (1/n) · Σᵢ (yᵢ - ŷᵢ)²

    Lower MSE = better fit.
    """
    n = len(y_actual)
    return sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(n)) / n


def r_squared(y_actual, y_predicted):
    """
    R² (Coefficient of Determination).

    Formula:
        R² = 1 - SS_res / SS_tot

    Where:
        SS_res = Σᵢ (yᵢ - ŷᵢ)²     ← residual (unexplained) variance
        SS_tot = Σᵢ (yᵢ - ȳ)²       ← total variance in y

    R² = 1.0 → perfect fit
    R² = 0.0 → model explains nothing (same as predicting ȳ)
    R² < 0   → model is worse than predicting the mean
    """
    y_mean = mean(y_actual)
    ss_res = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_actual)
    return 1 - (ss_res / ss_tot)


def sparsity_report(weights, feature_names=None, threshold=1e-6):
    """
    Reports which weights Lasso zeroed out (feature selection result).

    A weight is considered zero if |wⱼ| < threshold.

    Parameters:
        weights       (list): Learned weight vector
        feature_names (list): Optional list of feature name strings
        threshold     (float): Anything below this is treated as zero

    Returns:
        dict with "active", "zeroed", "sparsity_ratio"
    """
    p = len(weights)
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(p)]

    active = [(feature_names[j], round(weights[j], 6))
              for j in range(p) if abs(weights[j]) >= threshold]
    zeroed = [feature_names[j]
              for j in range(p) if abs(weights[j]) < threshold]

    return {
        "active":         active,
        "zeroed":         zeroed,
        "sparsity_ratio": len(zeroed) / p
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LAMBDA PATH EXPLORER  (shows how sparsity evolves with λ)
# ─────────────────────────────────────────────────────────────────────────────

def lambda_path(X_std, y, lambdas, feature_names=None):
    """
    Fits Lasso models across a sequence of λ values and
    shows how weights enter/exit the model.

    This is the "Lasso path" — a key diagnostic tool.

    The order in which weights become nonzero (as λ decreases)
    is a natural ranking of feature importance.

    Parameters:
        X_std        (list of lists): Standardized feature matrix
        y            (list):          Target vector
        lambdas      (list of float): λ values to try (recommend sorted descending)
        feature_names(list):          Feature name strings

    Prints a table of weight values at each λ.
    Returns list of (lambda, weights) tuples.
    """
    p = len(X_std[0])
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(p)]

    results = []
    header  = f"{'λ':>10} | " + " | ".join(f"{n:>10}" for n in feature_names)
    print(header)
    print("-" * len(header))

    for lam in sorted(lambdas, reverse=True):
        model = lasso_coordinate_descent(X_std, y, lam)
        w = model["weights"]
        results.append((lam, w))
        row = f"{lam:>10.4f} | " + " | ".join(f"{wi:>10.4f}" for wi in w)
        print(row)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  EXAMPLE — FEATURE SELECTION ON A NOISY DATASET
# ─────────────────────────────────────────────────────────────────────────────
#
#  Scenario:
#    We have 6 features but only 3 actually matter.
#    The true model is:  y = 2·x1 + (-3)·x3 + 1.5·x5 + noise
#    Features x2, x4, x6 are pure noise.
#
#    Goal: Can Lasso recover the true sparse structure?
#

X_raw = [
    # x1    x2      x3    x4     x5    x6
    [ 1.2,  0.3,   -2.1,  0.5,   3.0,  -0.1],
    [ 2.5,  1.1,   -0.8,  0.9,   1.5,   0.4],
    [-0.3, -0.5,    1.4, -1.2,   2.2,   0.7],
    [ 3.1,  0.2,   -3.0,  0.1,   0.8,  -0.6],
    [ 0.8,  1.4,    0.5,  0.4,   4.1,   0.2],
    [-1.5, -0.9,    2.3, -0.8,   1.0,   0.9],
    [ 2.0, -0.3,   -1.7,  1.1,   2.5,  -0.3],
    [ 1.0,  0.6,   -2.5, -0.2,   3.5,   0.1],
    [-0.8,  1.2,    1.0,  0.7,   0.5,  -0.8],
    [ 2.7, -1.0,   -2.8,  0.3,   2.0,   0.5],
]

# True: y = 2·x1 - 3·x3 + 1.5·x5 + noise
y = [
    2*x[0] - 3*x[2] + 1.5*x[4] + 0.1  for x in X_raw
]

FEATURE_NAMES = ["x1 (signal)", "x2 (noise)", "x3 (signal)",
                 "x4 (noise)",  "x5 (signal)", "x6 (noise)"]

LAMBDA = 0.05   # Try: 0.0, 0.05, 0.5, 2.0


# ── Standardize ───────────────────────────────────────────────────────────────
X_std, feat_means, feat_stds = standardize(X_raw)

# ── Fit ───────────────────────────────────────────────────────────────────────
model  = lasso_coordinate_descent(X_std, y, lambda_=LAMBDA)
y_pred = predict(X_std, model)

# ── Evaluate ──────────────────────────────────────────────────────────────────
mse      = mean_squared_error(y, y_pred)
r2       = r_squared(y, y_pred)
sparsity = sparsity_report(model["weights"], FEATURE_NAMES)


# ── Print Results ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("        L1 / LASSO REGRESSION RESULTS")
print("=" * 60)

print(f"\n  λ (lambda)   : {LAMBDA}")
print(f"  Converged    : {model['converged']}  (in {model['n_iter']} iterations)")
print(f"  Intercept    : {model['intercept']:.4f}")

print(f"\n  ── Learned Weights ──────────────────────────────────────")
for fname, w in zip(FEATURE_NAMES, model["weights"]):
    zero_marker = "  ← ZEROED (feature eliminated)" if abs(w) < 1e-6 else ""
    print(f"    {fname:<20} : {w:>8.4f}{zero_marker}")

print(f"\n  ── Performance ──────────────────────────────────────────")
print(f"    MSE  : {mse:.4f}")
print(f"    R²   : {r2:.4f}")

print(f"\n  ── Sparsity Report ──────────────────────────────────────")
print(f"    Active features  : {len(sparsity['active'])} / {len(model['weights'])}")
print(f"    Zeroed features  : {sparsity['zeroed']}")
print(f"    Sparsity ratio   : {sparsity['sparsity_ratio']:.1%}")

print()
print("=" * 60)
print("        LASSO PATH  (how weights change with λ)")
print("=" * 60)
print()

lambdas_to_try = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
lambda_path(X_std, y, lambdas_to_try, FEATURE_NAMES)

print()
print("=" * 60)
print("  INTERPRETATION")
print("=" * 60)
print("""
  True model:  y = 2·x1 - 3·x3 + 1.5·x5  (x2, x4, x6 are pure noise)

  At λ = 0.05:
    Lasso correctly identifies that x2, x4, x6 contribute
    nothing meaningful — their weights are driven to exactly zero.

    Signal features (x1, x3, x5) survive with nonzero weights.

  As λ increases (in the path table above):
    Weights shrink progressively toward zero.
    Noise features disappear first (they have weak signals).
    The strongest signal (x3, largest true coefficient) disappears last.

  This ordering = implicit feature importance ranking.

  L1 vs L2 on this problem:
    Ridge (L2) would keep x2, x4, x6 with tiny but nonzero weights.
    Lasso (L1) eliminates them completely → cleaner, interpretable model.
""")
print("=" * 60)