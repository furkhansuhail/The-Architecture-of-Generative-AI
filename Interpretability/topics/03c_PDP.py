"""
Partial Dependence Plots (PDP) — Visualising Feature Effects in ML Models
==========================================================================

Partial Dependence Plots (PDPs) answer one of the most fundamental questions
in model interpretability: "How does changing feature X affect the model's
predictions, all else being equal?" They reveal the marginal relationship
between a feature (or pair of features) and the predicted outcome, averaged
over the distribution of all other features in the training data.

Introduced by Jerome Friedman in his 2001 paper on Gradient Boosting Machines,
PDPs have become one of the most widely used global model explanation tools
precisely because they are visual, intuitive, and model-agnostic. Unlike SHAP
or LIME — which explain individual predictions — PDPs describe the model's
global behaviour: its average response as a feature sweeps across its range.

A PDP for income in a loan model answers: "On average, across all loan
applicants in our training data, how does approval probability change as we
increase income from $20K to $200K, while the distribution of all other
features (age, credit score, debt ratio) remains as it actually is in the data?"

This module goes deep: the mathematical foundation (marginalisation over the
feature distribution), the connection to ALE (Accumulated Local Effects) plots
that fix PDP's extrapolation problem, Individual Conditional Expectation (ICE)
curves that reveal heterogeneity hidden by the PDP average, 2D PDPs for
interaction visualisation, and the complete set of diagnostic techniques for
reading, validating, and communicating PDP-based insights.

"""

import textwrap
import re

TOPIC_NAME   = "Partial Dependence Plots (PDP)"
DISPLAY_NAME = "03c · PDP"
ICON         = "📈"
SUBTITLE     = "Marginal Feature Effects, ICE Curves, ALE Plots, and 2D Interactions"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE QUESTION PDPs ANSWER AND THE MARGINALISATION IDEA

### The Core Question

    You have trained a model f on features X = {x_1, x_2, ..., x_p}.
    You want to understand: "What is the relationship between feature x_j
    and the prediction, ignoring the effect of all other features?"

    This is a global question — not about one specific prediction (that is
    LIME/SHAP territory), but about the model's average behaviour across
    the entire feature space.

### The Marginalisation Approach

    The partial dependence function for feature x_j is defined as:

    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   PD_j(x_j) = E_{X_{-j}} [ f(x_j, X_{-j}) ]                        │
    │                                                                    │
    │             = ∫ f(x_j, x_{-j}) × p(x_{-j}) dx_{-j}                 │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

    Where:
        x_j    = the feature of interest (varies along the x-axis of the plot)
        X_{-j} = all features EXCEPT x_j (the "complement")
        p(x_{-j}) = marginal distribution of the complement features
        The integral marginalises out (averages over) all other features

    Intuition:
        For each value v of feature x_j, we compute the model's prediction
        for EVERY training sample but with x_j fixed at v. We then average
        these predictions. This gives the "average prediction when x_j = v."

    By repeating this for many values of v, we trace out how the average
    prediction changes as x_j varies — the partial dependence.

### The Monte Carlo Approximation

    The integral is approximated using the training data:

    PD_j(v) ≈ (1/n) × Σ_{i=1}^{n} f(v, x_{-j}^{(i)})

    Algorithm:
        1. Choose a grid of values G = {v_1, v_2, ..., v_k} for feature j.
           Typically: k=50 evenly spaced points from min(x_j) to max(x_j).
        2. For each grid value v_k:
              For each training sample i = 1 to n:
                  Create modified sample: replace x_j^{(i)} with v_k.
                  Compute prediction f(v_k, x_{-j}^{(i)}).
           Average over all n training samples.
        3. Plot: x-axis = v, y-axis = PD_j(v).

    This requires n × k model evaluations (e.g., 1000 samples × 50 grid
    points = 50,000 forward passes). Manageable for most models.

### What the PDP x-axis and y-axis Mean

    x-axis: the range of values of the feature of interest.
            For continuous features: a grid from min to max.
            For categorical features: one point per category.

    y-axis: the average predicted outcome at that feature value.
            For regression: average predicted value (e.g., house price in $).
            For classification: average predicted probability (e.g., 0.0–1.0).
            For log-loss classifiers: sometimes average predicted log-odds.

    The y-axis is CENTERED in most implementations: subtract the overall mean
    prediction so the y-axis shows the DEVIATION from average rather than
    the absolute prediction level. Centered PDPs from different features
    can be compared on the same scale.

### What PDPs Tell You (and What They Don't)

    PDPs REVEAL:
        The direction of a feature's effect (positive/negative/non-monotone).
        The shape of the relationship (linear, logarithmic, threshold).
        The range over which a feature has most impact.
        Whether a feature has a consistent effect across its range.

    PDPs DON'T REVEAL:
        Per-sample variation (that's ICE curves).
        Feature interactions (need 2D PDPs or ICE).
        Whether the model's learned relationship is causally valid.
        What happens for feature combinations outside the training distribution.


##### PART 2 — ICE CURVES: INDIVIDUAL CONDITIONAL EXPECTATION

### The Heterogeneity Problem with PDPs

    PDPs average over all training samples. This averaging can hide
    important heterogeneity in how different subgroups of the data
    respond to a feature.

    Classic failure case: suppressor effect.
        For subgroup A: prediction RISES as feature x_j increases.
        For subgroup B: prediction FALLS as feature x_j increases.
        PDP average: approximately flat — feature appears unimportant!
        Reality: feature has strong but OPPOSITE effects for different subgroups.

    A flat PDP can mean: (a) feature truly doesn't matter, OR
    (b) feature has strong but cancelling effects across subgroups.
    Without ICE curves, you cannot tell these apart.

### ICE Curves: One Line Per Sample

    ICE (Individual Conditional Expectation) plots display the PDP
    without the aggregation step — one prediction curve per sample:

    ICE_{ij}(x_j) = f(x_j, x_{-j}^{(i)})   for sample i, feature value x_j

    For n samples and k grid points: n × k model evaluations (same as PDP).
    The PDP is simply the column-wise mean of all ICE curves.

    Reading ICE plots:
        PARALLEL lines (all same shape/direction):
            Feature has a consistent, homogeneous effect.
            PDP is reliable — it represents every individual.

        DIVERGING lines (spread increases/decreases along x_j):
            Feature's effect is amplified/dampened at certain values.
            Suggests an interaction with another feature.

        CROSSING lines (lines cross each other):
            Feature has OPPOSITE effects for different individuals.
            PDP is misleading — the average masks subgroup differences.
            Strong evidence of an interaction with another feature.

### Centered ICE (c-ICE) Plots

    Raw ICE curves can be hard to compare because they start at different
    prediction levels (different intercepts). Centering removes this:

    c-ICE_{i}(x_j) = ICE_{i}(x_j) - ICE_{i}(x_j^{(reference)})

    Typically centered at the minimum value of x_j (left edge of the plot).
    All c-ICE lines start at 0 — now the SHAPE is directly comparable.

    c-ICE reveals: as x_j increases from its minimum, how much does
    the prediction change for each individual? This isolates the effect
    of the feature from baseline differences between individuals.

### When to Use PDPs vs ICE

    Always compute ICE alongside PDP. Workflow:
        Step 1: Plot PDPs to get the average picture.
        Step 2: Overlay ICE curves to check for heterogeneity.
        Step 3: If lines are parallel → PDP is trustworthy.
        Step 4: If lines cross → investigate subgroup structure.
                Colour ICE lines by a suspected interaction feature.
                Cluster ICE lines to find subgroups with different effects.


##### PART 3 — THE EXTRAPOLATION PROBLEM AND ALE PLOTS

### PDPs and Correlated Features

    PDPs have a fundamental problem when features are correlated:
    the marginalisation creates UNREALISTIC data combinations.

    Example: features are age and credit_score. They are correlated —
    older people tend to have higher credit scores.

    When computing PDP for age with age=25:
        For each training sample, replace age with 25.
        Sample 42 might have age=60, credit_score=750.
        After replacement: age=25, credit_score=750.
        But 25-year-olds with credit_score=750 are extremely rare!

    The PDP evaluates the model at many unrealistic feature combinations.
    If the model's behaviour in these unrealistic regions is unpredictable
    (extrapolation), the PDP may not reflect the model's behaviour on
    real data.

    Impact: can be severe when features are strongly correlated.
    Correlation > 0.5 between the target feature and any other feature
    should prompt switching from PDP to ALE plots.

### Accumulated Local Effects (ALE) Plots

    ALE plots (Apley & Zhu, 2016) fix the extrapolation problem by
    estimating the feature effect using LOCAL (conditional) differences
    rather than MARGINAL (unconditional) averages.

    Key idea: instead of fixing feature j to value v across ALL samples,
    ALE considers only samples where feature j is NEAR v, and computes
    how the prediction changes as j varies LOCALLY around v.

    ALE formula:
        ALE_j(x_j) = ∫_{z_0}^{x_j} E_{X_{-j}|x_j=z} [∂f(z, X_{-j})/∂z] dz

    Monte Carlo approximation:
        For each interval [z_{k-1}, z_k] in the grid:
            Select samples with x_j^{(i)} in [z_{k-1}, z_k].
            For each such sample, compute:
                f(z_k, x_{-j}^{(i)}) - f(z_{k-1}, x_{-j}^{(i)})
                (prediction change when x_j moves from z_{k-1} to z_k)
            Average these differences over the samples in the interval.
        Accumulate (integrate) these average differences from left to right.

    Why ALE is better for correlated features:
        Only uses samples with x_j^{(i)} NEAR v.
        These samples have realistic joint distributions of (x_j, x_{-j}).
        No extrapolation into regions the model hasn't seen.

    Trade-off:
        ALE requires sufficient samples in each interval (sparse data → noisy).
        ALE measures a CONDITIONAL effect (given x_j ≈ v).
        PDP measures a MARGINAL effect (averaging over all x_{-j}).
        For independent features: PDP ≈ ALE. For correlated: ALE is better.

### PDP vs ALE: When to Use Which

    Features are INDEPENDENT (correlation ≈ 0):
        → Use PDP. Equivalent to ALE, more widely understood.

    Features are CORRELATED (|correlation| > 0.4):
        → Use ALE. Avoids extrapolation into unrealistic regions.

    Features have STRONG correlations (|correlation| > 0.7):
        → Definitely use ALE. PDP may be severely misleading.

    Real-world rule of thumb:
        Always compute both. If they differ substantially, the correlation
        is causing the PDP to extrapolate — trust ALE.


##### PART 4 — 2D PDPS: VISUALISING FEATURE INTERACTIONS

### Two-Feature PDPs

    A 2D PDP shows the joint partial dependence on two features simultaneously:

    PD_{j,k}(x_j, x_k) = E_{X_{-j,-k}} [f(x_j, x_k, X_{-j,-k})]
    ≈ (1/n) × Σ_i f(x_j, x_k, x_{-j,-k}^{(i)})

    Visualised as a heatmap or surface plot:
        X-axis: values of feature j
        Y-axis: values of feature k
        Color/height: average predicted outcome

    Requires a grid of size k_j × k_k (e.g., 20×20 = 400 grid points).
    Each grid point requires n model evaluations.
    Total: n × k_j × k_k (e.g., 1000 × 400 = 400,000 forward passes).
    Computationally expensive — use a sample of training data if needed.

### Detecting Interactions from 2D PDPs

    An interaction between features j and k exists if the effect of
    feature j depends on the value of feature k. The 2D PDP reveals this.

    NO INTERACTION: the 2D PDP looks like a SUM of two 1D patterns.
        PD_{j,k}(x_j, x_k) ≈ PD_j(x_j) + PD_k(x_k)
        On a heatmap: horizontal stripes (if k matters) or
        vertical stripes (if j matters), never diagonal patterns.

    INTERACTION EXISTS: the effect of j changes depending on k.
        The heatmap shows DIAGONAL patterns, curved ridges,
        or regions where colour changes depend on both axes simultaneously.
        PD_{j,k}(x_j, x_k) ≠ PD_j(x_j) + PD_k(x_k)

    H-statistic (Friedman & Popescu, 2008):
        Measures interaction strength between features j and k:
        H²_{jk} = Σ [PD_{j,k}(x_j, x_k) - PD_j(x_j) - PD_k(x_k)]² /
                   Σ [PD_{j,k}(x_j, x_k)]²
        H² = 0: no interaction.
        H² = 1: fully interaction-driven (PD is entirely non-additive).
        H² > 0.1: practically significant interaction.

### Reading 2D PDP Heatmaps

    ADDITIVE MODEL (no interaction):
        Horizontal bands: feature k matters, j doesn't (or vice versa).
        Or: bands diagonal but perfectly predictable from 1D PDPs.

    MULTIPLICATIVE INTERACTION:
        Top-right corner much higher/lower than either top-left or bottom-right.
        "Both features must be high together for a big effect" pattern.
        Example: income × credit_score — need BOTH high for approval.

    THRESHOLD × THRESHOLD:
        Flat everywhere except a corner — both features must cross thresholds.
        Common in tree ensembles (threshold splits create corner effects).

    SUPPRESSOR INTERACTION:
        High j + low k ≈ low j + high k (they cancel).
        One feature compensates for the other.


##### PART 5 — PDP VARIANTS AND EXTENSIONS

### M-Plots (Marginal Plots)

    M-plots (Marginal plots) plot E[f(X) | X_j = v] — the CONDITIONAL
    expected prediction, not the marginal.

    Difference from PDP:
        PDP: E_{X_{-j}}[f(v, X_{-j})] — fix j=v, marginalise over x_{-j}
        M-plot: E[f(X) | X_j=v] — condition on j=v, use realistic x_{-j}

    M-plots use only samples where x_j ≈ v. This is more realistic but
    conflates the effect of x_j with correlated features.
    Example: if age and income are correlated, the M-plot for income at
    high values reflects both the income effect AND the age effect
    (because high-income samples also tend to be older).

    PDP separates the feature's own effect. M-plot shows the combined effect
    including correlated features. Neither is "wrong" — they answer different
    questions.

### Categorical Features in PDPs

    For categorical features, the PDP is computed as a bar chart:
        One bar per category value.
        Height = average prediction when category is set to that value.

    Watch out: if a category has very few samples, the average prediction
    at that category value will be noisy.

    For ordinal categoricals (small/medium/large), preserve the order.
    For nominal categoricals (countries, product types), sort by PD value
    for readability.

### PDPs for Multi-Output Models

    For multi-class classification (K classes):
        Compute a separate PDP for each class.
        This gives K curves on K plots (or overlaid with different colours).
        Useful for understanding which features drive each class separately.

    For multi-output regression:
        Compute a separate PDP for each output.

### Smoothing PDPs

    For noisy models (e.g., neural nets, small trees), PDPs can be jagged.
    Apply a smoother (LOESS, Gaussian process, spline) to reduce noise:

        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(grid, pdp_values, s=0.5)
        pdp_smooth = spline(grid)

    Only smooth for visualisation. Use raw values for analysis.

### Derivative PDPs (d-PDPs)

    The derivative of the PDP reveals where the feature has the most impact:

    d-PDP_j(x_j) = ∂PD_j(x_j) / ∂x_j ≈ (PD_j(v+h) - PD_j(v-h)) / (2h)

    Large absolute derivative: prediction changes rapidly around this value.
    Near-zero derivative: feature has little effect here.
    Sign of derivative: direction of effect (positive → prediction increases).

    d-PDPs identify THRESHOLD VALUES — points where the model transitions
    from one behaviour to another. These are particularly meaningful for
    decision support systems.


##### PART 6 — READING AND INTERPRETING PDP SHAPES

### Common PDP Shapes and Their Interpretations

    MONOTONE INCREASING:
        Higher feature value → higher prediction (all else equal).
        Example: income → loan approval probability.
        Implication: the model has learned a positive correlation.

    MONOTONE DECREASING:
        Higher feature value → lower prediction.
        Example: debt-to-income ratio → loan approval probability.
        Implication: negative correlation learned.

    STEP FUNCTION (one or more jumps):
        Prediction is flat then jumps at a specific value.
        Very common in tree ensembles — reflects a split threshold.
        The jump point is a learned decision boundary.
        Example: approval probability jumps at credit_score = 680.

    UNIMODAL (hill shape):
        Optimal value in the middle — prediction increases then decreases.
        Example: age → athletic performance (peaks in 20s-30s).
        Or: ideal temperature for crop yield.

    U-SHAPED:
        Minimum effect at intermediate values, stronger at extremes.
        Example: extreme political views → election turnout.

    FLAT (near-zero):
        Feature has no (or minimal) average effect on prediction.
        Could mean: (a) truly unimportant, or (b) cancelling subgroup effects.
        CHECK ICE curves before concluding the feature is unimportant.

    NON-MONOTONE (complex shape):
        Multiple peaks and valleys — complex non-linear learned relationship.
        Could reflect genuine data patterns or overfitting.

### Domain Knowledge Alignment

    Always compare PDP shapes against domain knowledge:
        Does the direction make sense? (Income → approval should be positive)
        Does the threshold make sense? (Credit score 680 is a real boundary)
        Does the shape make sense? (Is U-shaped age realistic for this outcome?)

    Surprising PDP shapes are diagnostic signals:
        Unexpected positive effect: model may have learned a spurious correlation.
        Threshold at a round number: may reflect a business rule in the data.
        Non-monotone where monotone expected: possible data quality issue.


##### PART 7 — PDPs vs SHAP DEPENDENCE PLOTS AND OTHER METHODS

### PDP vs SHAP Dependence Plot

    SHAP dependence plot:
        X-axis: actual feature value.
        Y-axis: SHAP value for that feature for each sample.
        Colour: a second feature (interaction partner).
        Points: individual training/test samples.

    PDP:
        X-axis: feature value (grid).
        Y-axis: average model prediction for that feature value.
        No interaction colouring by default.
        No individual sample points.

    Key differences:
        PDP = averaged over samples (one curve).
        SHAP dependence = individual points (scatter plot).
        PDP y-axis = absolute prediction.
        SHAP y-axis = attribution (contribution ABOVE baseline).

    Which to use?
        PDP: presenting to non-technical stakeholders ("here's how income
             affects the model's approval rate on average").
        SHAP dependence: investigating interactions ("how does income's
             SHAP value change depending on credit score?").

### PDP vs ICE vs ALE — Summary Table

    ┌───────────────────────────────────────────────────────────────────┐
    │ Method │ Aggregation │ Correlation │ Interaction │ Computation    │
    ├───────────────────────────────────────────────────────────────────┤
    │ PDP    │ Mean        │ Problematic │ Hidden      │ n × k          │
    │ ICE    │ None        │ Problematic │ Visible     │ n × k          │
    │ c-ICE  │ None        │ Problematic │ Visible+    │ n × k          │
    │ ALE    │ Mean local  │ Handles it  │ Hidden      │ n × k (local)  │
    │ 2D PDP │ Mean        │ Problematic │ Visible     │ n × k_j × k_k  │
    └───────────────────────────────────────────────────────────────────┘

    Recommended workflow:
        1. Compute PDP for all features: get the global picture.
        2. Compute ICE for top features: check for heterogeneity.
        3. If features are correlated: compute ALE to validate PDP.
        4. For top interacting feature pairs: compute 2D PDPs.
        5. Confirm with domain knowledge and SHAP dependence plots.


##### PART 8 — IMPLEMENTATION DETAILS AND PRODUCTION CONSIDERATIONS

### Computational Cost and Optimisation

    STANDARD COST:
        n samples × k grid points × 1 model call = n×k calls total.
        For n=10,000 and k=50: 500,000 forward passes.
        For neural networks: can be slow. For tree models: fast.

    OPTIMISATION STRATEGIES:

    1. Subsample the training data:
        Use a random sample of 500–2000 training points instead of all n.
        ICE heterogeneity is captured well even with 200 samples.
        For PDPs, 1000 samples is usually sufficient for smooth curves.

    2. Reduce grid resolution:
        k=20 is often sufficient for smooth PDPs.
        k=100 only needed if you want to detect sharp thresholds.

    3. Parallelise:
        Each grid point is independent — embarrassingly parallelisable.
        Use joblib.Parallel with n_jobs=-1 for multicore CPU utilisation.

    4. Cache predictions:
        For 2D PDPs: precompute a matrix of predictions, then slice.

    5. Vectorised prediction:
        Batch all modified samples for a single grid value into one array.
        Call model.predict(X_modified) once per grid value, not n times.

### Scikit-learn Implementation

    sklearn.inspection provides PDPs directly:

        from sklearn.inspection import partial_dependence, PartialDependenceDisplay

        # 1D PDP + ICE
        pd_result = partial_dependence(
            estimator,              # fitted sklearn model
            X,                      # training or test data
            features=[0, 3, 5],     # feature indices to explain
            kind='average',         # 'average'=PDP, 'individual'=ICE, 'both'
            grid_resolution=50,     # number of grid points
            percentiles=(0.05, 0.95) # clip extreme values
        )

        # Display
        PartialDependenceDisplay.from_estimator(
            estimator, X,
            features=[(0,), (3,), (0, 3)],  # 1D and 2D
            kind='both',            # show PDP + ICE
            subsample=500,          # use 500 random samples
        )

    The sklearn implementation is optimised with Cython and joblib.
    It handles: regression, binary classification, multi-class classification.

### Third-Party Libraries

    PDPbox:         Full-featured PDP/ICE/interaction library.
    PyALE:          ALE plots for Python.
    InterpretML:    Microsoft's interpretability library (includes PDPs).
    DALEX:          R and Python, model-agnostic, includes PDPs + ALE.

### Communicating PDP Results

    For data scientists:
        Show PDP + ICE overlay with rug plot (actual data density shown at bottom).
        Include confidence bands (bootstrap PDP ± 1.96σ).
        Show derivative PDP to highlight threshold regions.

    For domain experts:
        Single PDP per feature with meaningful axis labels.
        Annotate threshold points ("model switches from reject to approve here").
        Compare with empirical data distribution (histogram below x-axis).

    For executives/regulators:
        Bar chart of PDP "effect size" (max - min PDP value) per feature.
        Narrative: "Increasing income from $40K to $80K increases approval
        rate by 23 percentage points, on average across all applicants."

    For regulatory reporting (GDPR, model documentation):
        Include PDP plots in model card with description of the training data
        used to compute them, the grid resolution, and any subsampling applied.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · PDP from Scratch — Implementing the Marginalisation Algorithm": {
        "description": (
            "Build PDP computation from first principles. "
            "Implement the Monte Carlo marginalisation exactly as defined. "
            "Compute PDPs for multiple features on a real dataset. "
            "Show the grid construction and vectorised prediction approach. "
            "Visualise as ASCII plots. Compare with sklearn's partial_dependence. "
            "Measure computation time and show subsampling optimisation."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  PDP FROM SCRATCH — IMPLEMENTING THE MARGINALISATION ALGORITHM")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: PDP algorithm from first principles
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The PDP algorithm step by step")
print("━" * 65)
print()

print("  PDP FORMULA:")
print("    PD_j(v) ≈ (1/n) × Σ_i f(v, x_{-j}^{(i)})")
print()
print("  ALGORITHM:")
print("    For each grid value v in {v_1, ..., v_k}:")
print("      Copy training data X to X_modified")
print("      Set column j of X_modified to v (for ALL rows)")
print("      Compute predictions: preds = model.predict(X_modified)")
print("      PD_j(v) = mean(preds)")
print()


def compute_pdp(model_fn, X, feature_idx, grid_resolution=50,
                percentile_range=(0.05, 0.95)):
    """
    Compute 1D Partial Dependence for feature `feature_idx`.

    Returns:
        grid_values: array of shape (grid_resolution,) — x-axis values
        pdp_values:  array of shape (grid_resolution,) — average predictions
    """
    # Build the grid: evenly spaced between the 5th and 95th percentiles
    # (avoid extreme outliers distorting the x-axis)
    lo = np.percentile(X[:, feature_idx], percentile_range[0] * 100)
    hi = np.percentile(X[:, feature_idx], percentile_range[1] * 100)
    grid_values = np.linspace(lo, hi, grid_resolution)

    pdp_values = np.zeros(grid_resolution)

    for k, v in enumerate(grid_values):
        # Create modified dataset: fix feature j to v, keep all other features
        X_mod = X.copy()
        X_mod[:, feature_idx] = v

        # Predict for all modified samples, average
        preds = model_fn(X_mod)
        pdp_values[k] = preds.mean()

    return grid_values, pdp_values


def compute_ice(model_fn, X, feature_idx, grid_resolution=50,
                n_ice_samples=None, percentile_range=(0.05, 0.95), seed=42):
    """
    Compute ICE curves for feature `feature_idx`.
    Each row in the output is one sample's prediction curve.

    Returns:
        grid_values: array (grid_resolution,)
        ice_curves:  array (n_samples, grid_resolution)
    """
    rng = np.random.RandomState(seed)
    if n_ice_samples is not None and n_ice_samples < len(X):
        idx = rng.choice(len(X), size=n_ice_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    lo = np.percentile(X[:, feature_idx], percentile_range[0] * 100)
    hi = np.percentile(X[:, feature_idx], percentile_range[1] * 100)
    grid_values = np.linspace(lo, hi, grid_resolution)

    ice_curves = np.zeros((len(X_sub), grid_resolution))

    for k, v in enumerate(grid_values):
        X_mod = X_sub.copy()
        X_mod[:, feature_idx] = v
        ice_curves[:, k] = model_fn(X_mod)

    return grid_values, ice_curves


# ─────────────────────────────────────────────────────────────────────────
# Setup: dataset and model
# ─────────────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    housing  = fetch_california_housing()
    X, y     = housing.data, housing.target
    fn       = list(housing.feature_names)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, random_state=42)
    gbm.fit(X_tr, y_tr)
    r2 = gbm.score(X_te, y_te)
    model_fn = gbm.predict
    print(f"  Dataset: California Housing Regression")
    print(f"  Features: {fn}")
    print(f"  GBM R²:   {r2:.4f}")
    HAS_SKL = True
except ImportError:
    # Fallback synthetic dataset
    HAS_SKL = False
    np.random.seed(42)
    n, p      = 800, 5
    X_tr      = np.random.randn(n, p)
    X_te      = np.random.randn(200, p)
    fn        = [f"Feature_{i}" for i in range(p)]
    # Non-linear model with interaction
    def model_fn(X):
        return (2*X[:,0] + 1.5*X[:,1]**2 - 1.5*X[:,2]
                + 1.0*X[:,0]*X[:,1] + np.random.randn(len(X))*0.1)
    y_tr = model_fn(X_tr)
    print("  Using synthetic dataset (sklearn not available)")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Compute and display PDPs
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Computing PDPs for all features")
print("━" * 65)
print()

# Use training data for PDP (standard practice)
X_pdp = X_tr[:500]   # subsample for speed

print(f"  Computing PDPs using {len(X_pdp)} samples, 30-point grid...")
print()

t0 = time.perf_counter()
pdp_results = {}
for j in range(min(X_pdp.shape[1], 8)):
    grid, pdp = compute_pdp(model_fn, X_pdp, feature_idx=j, grid_resolution=30)
    pdp_results[j] = (grid, pdp)
t_pdp = time.perf_counter() - t0

print(f"  Computed {len(pdp_results)} PDPs in {t_pdp:.2f}s")
print()

# ASCII visualisation of PDPs
def ascii_plot(x_vals, y_vals, title, width=55, height=10,
               x_label="", y_label=""):
    """Compact ASCII line plot."""
    y_min, y_max = y_vals.min(), y_vals.max()
    y_range      = y_max - y_min if y_max != y_min else 1.0
    x_min, x_max = x_vals.min(), x_vals.max()

    grid = [[' '] * width for _ in range(height)]
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = height - 1 - int((y - y_min) / y_range * (height - 1))
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1,  col))
        grid[row][col] = '●'

    lines = [''.join(r) for r in grid]
    print(f"  {title}")
    print(f"  y={y_max:.3f} |{'─'*width}|")
    for row_str in lines:
        print(f"          |{row_str}|")
    print(f"  y={y_min:.3f} |{'─'*width}|")
    print(f"          {x_min:.3f}{' '*(width//2-8)}{(x_min+x_max)/2:.3f}"
          f"{' '*(width//2-8)}{x_max:.3f}")
    print()

# Show PDPs for top features
for j in range(min(4, len(pdp_results))):
    grid, pdp = pdp_results[j]
    fname = fn[j] if j < len(fn) else f"Feature_{j}"
    ascii_plot(grid, pdp, f"PDP: {fname}", width=50, height=8)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Effect sizes from PDPs
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Feature effect sizes from PDPs")
print("━" * 65)
print()

print("  Effect size = max(PDP) - min(PDP) over the grid.")
print("  Larger effect = feature has more impact on average prediction.")
print()
print(f"  {'Rank':>4} | {'Feature':<22} | {'PDP min':>9} | {'PDP max':>9} | "
      f"{'Effect':>9} | {'Shape'}")
print(f"  {'─'*72}")

effects = []
for j, (grid, pdp) in pdp_results.items():
    fname  = fn[j] if j < len(fn) else f"Feature_{j}"
    effect = pdp.max() - pdp.min()
    diffs  = np.diff(pdp)

    # Classify shape
    if (diffs > 0).sum() > 0.9 * len(diffs):
        shape = "↑ monotone +"
    elif (diffs < 0).sum() > 0.9 * len(diffs):
        shape = "↓ monotone -"
    elif diffs.max() > 0.3 * abs(diffs).max() and diffs.min() < -0.3 * abs(diffs).max():
        shape = "↕ non-monotone"
    else:
        shape = "≈ flat"
    effects.append((effect, j, fname, pdp.min(), pdp.max(), shape))

effects.sort(reverse=True)
for rank, (effect, j, fname, pmin, pmax, shape) in enumerate(effects):
    print(f"  {rank+1:>4} | {fname:<22} | {pmin:>9.4f} | {pmax:>9.4f} | "
          f"{effect:>9.4f} | {shape}")

print()
print("  ⚠️  Note: A 'flat' PDP may still indicate a feature with strong but")
print("  OPPOSITE effects across subgroups (check ICE curves!).")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Compare from-scratch PDP with sklearn
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Validation: from-scratch vs sklearn partial_dependence")
print("━" * 65)
print()

if HAS_SKL:
    from sklearn.inspection import partial_dependence

    print("  Comparing PDP from-scratch vs sklearn for feature 0...")
    feat_idx = 0

    # Our implementation
    grid_our, pdp_our = compute_pdp(model_fn, X_pdp, feat_idx, grid_resolution=30)
    pdp_our_centered  = pdp_our - pdp_our.mean()

    # sklearn implementation
    pd_sk = partial_dependence(gbm, X_pdp, features=[feat_idx],
                                grid_resolution=30, percentiles=(0.05, 0.95))
    grid_sk = pd_sk['grid_values'][0]
    pdp_sk  = pd_sk['average'][0]
    pdp_sk_centered = pdp_sk - pdp_sk.mean()

    # Compare at interpolated common grid
    from scipy.interpolate import interp1d
    lo = max(grid_our.min(), grid_sk.min())
    hi = min(grid_our.max(), grid_sk.max())
    common = np.linspace(lo, hi, 20)

    f_our = interp1d(grid_our, pdp_our_centered, fill_value='extrapolate')
    f_sk  = interp1d(grid_sk,  pdp_sk_centered,  fill_value='extrapolate')
    our_interp = f_our(common)
    sk_interp  = f_sk(common)
    corr       = np.corrcoef(our_interp, sk_interp)[0, 1]
    mae        = np.mean(np.abs(our_interp - sk_interp))

    print(f"  Feature: {fn[feat_idx]}")
    print(f"  Correlation (centered PDP): {corr:.6f}  {'✅' if corr > 0.99 else '⚠️'}")
    print(f"  Mean absolute error:        {mae:.4f}")
    print()
    print("  Small differences expected due to:")
    print("    - Different grid spacing (our percentile vs sklearn's)")
    print("    - Different centering implementation")
    print("    - sklearn uses the full X_pdp; we subsampled")

else:
    print("  (sklearn not available — skipping comparison)")

print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Computational efficiency
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Computational cost and optimisation")
print("━" * 65)
print()

print("  Cost: n_samples × grid_resolution forward passes per feature.")
print()
print(f"  {'n_samples':>10} | {'grid_pts':>9} | {'Total calls':>13} | "
      f"{'Approx time':>13} | {'Recommendation'}")
print(f"  {'─'*70}")

configs = [
    (100,   20, "Prototype"),
    (500,   30, "Development"),
    (1000,  50, "Standard"),
    (5000,  50, "High-quality"),
    (10000, 100, "Research-grade"),
]
for n, k, label in configs:
    total    = n * k
    est_time = total / 200_000 * 1.0   # assume ~200k predictions/sec
    print(f"  {n:>10,} | {k:>9} | {total:>13,} | {est_time:>11.2f}s | {label}")

print()
print("  Vectorisation tip:")
print("    SLOW:  for each sample i: predict([v, x_{-j}^{(i)}])")
print("    FAST:  X_mod[:,j] = v; predict(X_mod)  (one batch call)")
print()

# Show speed comparison
print("  Speed comparison: loop vs vectorised:")
feat_idx = 0
v_test   = X_pdp[:, feat_idx].mean()

# Loop version
t0 = time.perf_counter()
preds_loop = []
for i in range(min(100, len(X_pdp))):
    x_mod = X_pdp[i].copy()
    x_mod[feat_idx] = v_test
    preds_loop.append(model_fn(x_mod.reshape(1, -1))[0])
t_loop = time.perf_counter() - t0

# Vectorised version
t0 = time.perf_counter()
X_mod = X_pdp[:100].copy()
X_mod[:, feat_idx] = v_test
preds_vec = model_fn(X_mod)
t_vec = time.perf_counter() - t0

print(f"    Loop (100 samples):       {t_loop*1000:.3f}ms")
print(f"    Vectorised (100 samples): {t_vec*1000:.3f}ms")
print(f"    Speedup:                  {t_loop/max(t_vec,1e-9):.1f}×")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · ICE Curves, Interactions, and the ALE vs PDP Comparison": {
        "description": (
            "ICE curves: reveal hidden heterogeneity in feature effects. "
            "Show parallel vs crossing ICE patterns and what each means. "
            "Compute centered ICE (c-ICE) to isolate effect shapes. "
            "Demonstrate the ALE vs PDP difference under correlated features. "
            "Implement the H-statistic to quantify interaction strength. "
            "Build 2D PDPs to visualise feature interactions visually."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  ICE CURVES, INTERACTIONS, AND ALE vs PDP")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Setup: synthetic dataset with KNOWN interaction
# ─────────────────────────────────────────────────────────────────────────
def make_dataset_with_interaction(n=800):
    """
    y = 2*x0 + x1**2 + x0*x2 - x3 + noise
    x0 and x2 have a STRONG interaction (product term).
    x0 effect depends on x2 value.
    """
    X = np.random.randn(n, 5)
    y = (2.0  * X[:,0]           # linear
         + 1.0 * X[:,1]**2        # quadratic (U-shaped PDP)
         + 1.5 * X[:,0] * X[:,2]  # INTERACTION: x0's effect depends on x2
         - 0.8 * X[:,3]           # linear negative
         + 0.2 * np.random.randn(n))  # noise
    return X, y

X_data, y_data = make_dataset_with_interaction(1000)
fn             = ["x0 (main+interaction)", "x1 (quadratic)", "x2 (interacts w/x0)",
                  "x3 (linear neg)", "x4 (noise)"]

try:
    from sklearn.ensemble import GradientBoostingRegressor
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    gbm.fit(X_data, y_data)
    model_fn = gbm.predict
    r2 = 1 - np.mean((y_data - model_fn(X_data))**2) / np.var(y_data)
    print(f"  GBM R² on synthetic data: {r2:.4f}")
    HAS_SKL = True
except ImportError:
    # Fallback: use the true function + noise
    def model_fn(X):
        return (2.0*X[:,0] + 1.0*X[:,1]**2 + 1.5*X[:,0]*X[:,2]
                - 0.8*X[:,3])
    HAS_SKL = False
    print("  Using true model function (sklearn not available)")

print()

# ─────────────────────────────────────────────────────────────────────────
def compute_ice(model_fn, X, feature_idx, n_grid=40,
                n_ice=None, pct=(0.05,0.95)):
    lo   = np.percentile(X[:, feature_idx], pct[0]*100)
    hi   = np.percentile(X[:, feature_idx], pct[1]*100)
    grid = np.linspace(lo, hi, n_grid)
    if n_ice and n_ice < len(X):
        idx  = np.random.choice(len(X), n_ice, replace=False)
        Xsub = X[idx]
    else:
        Xsub = X
    curves = np.zeros((len(Xsub), n_grid))
    for k, v in enumerate(grid):
        Xm = Xsub.copy(); Xm[:, feature_idx] = v
        curves[:, k] = model_fn(Xm)
    pdp = curves.mean(axis=0)
    return grid, curves, pdp


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — ICE curves: detecting hidden heterogeneity")
print("━" * 65)
print()

print("  Computing ICE for x0 (has interaction with x2)...")
grid0, ice0, pdp0 = compute_ice(model_fn, X_data, 0, n_ice=200, n_grid=40)

print("  Computing ICE for x3 (linear, no interaction)...")
grid3, ice3, pdp3 = compute_ice(model_fn, X_data, 3, n_ice=200, n_grid=40)

def ice_stats(ice_curves, pdp):
    """Statistics that detect heterogeneity / crossing."""
    # Monotone fraction: what fraction of ICE curves are monotone increasing?
    pct_mono_inc = (np.diff(ice_curves, axis=1) > 0).all(axis=1).mean()
    pct_mono_dec = (np.diff(ice_curves, axis=1) < 0).all(axis=1).mean()

    # Crossing index: fraction of pairs that cross
    n = len(ice_curves)
    # Sample 100 pairs for efficiency
    cross_count = 0
    total_pairs = 500
    for _ in range(total_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        # Crossing: first point i > j but last point i < j (or vice versa)
        diff_start = ice_curves[i, 0]  - ice_curves[j, 0]
        diff_end   = ice_curves[i, -1] - ice_curves[j, -1]
        if diff_start * diff_end < 0:
            cross_count += 1
    crossing_frac = cross_count / total_pairs

    # ICE spread at each grid point (vs PDP)
    std_per_grid = ice_curves.std(axis=0)
    mean_spread  = std_per_grid.mean()

    return pct_mono_inc, pct_mono_dec, crossing_frac, mean_spread

print()
for name, grid, ice, pdp in [("x0 (interaction)", grid0, ice0, pdp0),
                               ("x3 (linear/no interaction)", grid3, ice3, pdp3)]:
    pmi, pmd, cf, ms = ice_stats(ice, pdp)
    print(f"  Feature: {name}")
    print(f"    ICE curves:                {len(ice)}")
    print(f"    Monotone increasing:       {pmi:.1%}")
    print(f"    Monotone decreasing:       {pmd:.1%}")
    print(f"    Crossing fraction:         {cf:.1%}  "
          f"{'⚠️  INTERACTION LIKELY' if cf > 0.2 else '✅ likely additive'}")
    print(f"    Mean ICE spread (std):     {ms:.4f}  "
          f"{'Wide spread → heterogeneous effect' if ms > 0.5 else 'Narrow → homogeneous'}")
    print()

# Show text-based ICE visualisation
print("  ICE curves for x0 (interaction feature) — first 10 samples:")
print()
print(f"  Grid position:  {'  '.join([f'{v:.1f}' for v in grid0[::8]])}")
for i in range(min(10, len(ice0))):
    vals = '  '.join([f'{ice0[i,k]:+.2f}' for k in range(0, len(grid0), 8)])
    print(f"  Sample {i+1:>3}:    {vals}")
print(f"  PDP (mean):     {'  '.join([f'{v:+.2f}' for v in pdp0[::8]])}")
print()
print("  If samples have very different signs, lines are crossing → interaction.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Centered ICE (c-ICE): isolating effect shapes")
print("━" * 65)
print()

print("  Centered ICE subtracts each curve's value at the leftmost grid point.")
print("  All c-ICE lines start at zero — directly comparable across samples.")
print()

# Center at the leftmost grid point
c_ice0 = ice0 - ice0[:, 0:1]   # subtract first grid point value per sample
c_pdp0 = pdp0 - pdp0[0]        # center the PDP too

print("  c-ICE for x0 (first 10 samples, starting at 0.000):")
print()
print(f"  Grid →      {grid0[0]:.2f}  {'':>5}  {grid0[len(grid0)//2]:.2f}  "
      f"{'':>5}  {grid0[-1]:.2f}")
print(f"  {'─'*55}")
for i in range(min(10, len(c_ice0))):
    v_start = c_ice0[i, 0]
    v_mid   = c_ice0[i, len(grid0)//2]
    v_end   = c_ice0[i, -1]
    direction = "↑" if v_end > 0.5 else ("↓" if v_end < -0.5 else "≈")
    print(f"  Sample {i+1:>3}: {v_start:+7.3f}  {'':>4}  {v_mid:+7.3f}  "
          f"{'':>4}  {v_end:+7.3f}  {direction}")
print(f"  c-PDP:      {c_pdp0[0]:+7.3f}  {'':>4}  {c_pdp0[len(grid0)//2]:+7.3f}  "
      f"{'':>4}  {c_pdp0[-1]:+7.3f}")
print()
print("  Mixed ↑ and ↓ directions → feature has OPPOSITE effects for different samples.")
print("  This is the interaction: x0's direction depends on x2's value.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — The extrapolation problem: PDP vs ALE")
print("━" * 65)
print()

print("  Demonstrating the extrapolation problem with correlated features.")
print()

# Create dataset with correlated features
n_corr = 600
x1 = np.random.randn(n_corr)
x2 = x1 + np.random.randn(n_corr) * 0.3   # x1 and x2 are highly correlated!
X_corr = np.column_stack([x1, x2])

# True relationship: y = x1 + x2 (perfectly additive)
def true_model(X):
    return X[:,0] + X[:,1]

# Check correlation
corr_val = np.corrcoef(x1, x2)[0,1]
print(f"  Correlation between x1 and x2: {corr_val:.3f} (highly correlated)")
print()

# PDP for x1
grid_pdp, _, pdp_vals = compute_ice(true_model, X_corr, 0, n_grid=30)
# What the CORRECT answer should be: PD_1(v) = v + E[X2] = v + ~0
pdp_true = grid_pdp + X_corr[:,1].mean()

print("  PDP vs TRUE partial dependence (additive model y=x1+x2):")
print()
print(f"  {'x1 value':>10} | {'PDP estimate':>14} | {'True PD':>10} | "
      f"{'Error':>8} | {'Extrapolation?'}")
print(f"  {'─'*65}")

grid_sample = grid_pdp[::6]
pdp_sample  = pdp_vals[::6]
true_sample = pdp_true[::6]

for v, est, true in zip(grid_sample, pdp_sample, true_sample):
    err  = est - true
    # Check if this grid value has realistic (x1, x2) combinations near it
    mask = np.abs(X_corr[:,0] - v) < 0.5
    n_near = mask.sum()
    realistic = "✅ well-sampled" if n_near > 20 else "⚠️  sparse/extrapolating"
    print(f"  {v:>10.3f} | {est:>14.4f} | {true:>10.4f} | {err:>+8.4f} | {realistic}")

print()
print("  The PDP uses samples with x2 values from the FULL distribution,")
print("  even when x1 is fixed at an extreme value.")
print("  At x1=2.0: most training data has x2≈2.0 (correlated).")
print("  But PDP uses x2 values from its full range, including x2=-2.0.")
print("  This (x1=2.0, x2=-2.0) pair barely exists in training data.")
print("  → The model is evaluated in extrapolation territory.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — ALE plots: the correlated-feature solution")
print("━" * 65)
print()

def compute_ale(model_fn, X, feature_idx, n_bins=30):
    """
    Compute Accumulated Local Effects for feature j.
    Uses local differences within narrow bins — avoids extrapolation.
    """
    x_j      = X[:, feature_idx]
    quantiles = np.percentile(x_j, np.linspace(5, 95, n_bins+1))
    quantiles = np.unique(quantiles)  # remove duplicates
    n_bins    = len(quantiles) - 1

    ale_vals  = np.zeros(n_bins)
    bin_mids  = np.zeros(n_bins)

    for b in range(n_bins):
        lo_b   = quantiles[b]
        hi_b   = quantiles[b+1]
        mask   = (x_j >= lo_b) & (x_j < hi_b)
        if mask.sum() == 0:
            continue

        X_lo       = X[mask].copy()
        X_hi       = X[mask].copy()
        X_lo[:, feature_idx] = lo_b
        X_hi[:, feature_idx] = hi_b

        # Local difference: effect of moving x_j from lo_b to hi_b
        # using ONLY samples that are actually in this bin (realistic!)
        local_diff = model_fn(X_hi) - model_fn(X_lo)
        ale_vals[b] = local_diff.mean()
        bin_mids[b] = (lo_b + hi_b) / 2

    # Accumulate (integrate) the local differences
    ale_accumulated = np.cumsum(ale_vals)

    # Centre: subtract mean (conventional)
    weights       = np.array([((x_j >= quantiles[b]) & (x_j < quantiles[b+1])).sum()
                               for b in range(n_bins)])
    weights       = weights / weights.sum()
    ale_mean      = np.sum(ale_accumulated * weights)
    ale_centered  = ale_accumulated - ale_mean

    return bin_mids, ale_centered


# Compare PDP vs ALE for the correlated case
grid_ale, ale = compute_ale(true_model, X_corr, 0, n_bins=20)
grid_pdp_c, _, pdp_c = compute_ice(true_model, X_corr, 0, n_grid=30)

# Center PDP for comparison
pdp_c_centered = pdp_c - pdp_c.mean()

print("  Correlated features (correlation=0.9): PDP vs ALE comparison")
print(f"  True model: y = x1 + x2  (PD_1 should be linear = grid value + constant)")
print()
print(f"  {'x1 value':>10} | {'PDP (centered)':>16} | {'ALE':>8} | {'Diff':>8}")
print(f"  {'─'*48}")

# Interpolate for comparison
from scipy.interpolate import interp1d
f_pdp = interp1d(grid_pdp_c, pdp_c_centered, fill_value='extrapolate')
f_ale = interp1d(grid_ale,    ale,             fill_value='extrapolate')
common_grid = np.linspace(max(grid_pdp_c.min(), grid_ale.min()),
                           min(grid_pdp_c.max(), grid_ale.max()), 10)

for v in common_grid:
    p = f_pdp(float(v))
    a = f_ale(float(v))
    print(f"  {v:>10.3f} | {p:>16.4f} | {a:>8.4f} | {p-a:>+8.4f}")

print()
print("  For y=x1+x2, PD_1 should be ≈ linear (slope 1).")
print("  If PDP is non-linear: correlation is causing extrapolation error.")
print("  ALE stays close to the true linear relationship.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — 2D PDPs and the H-statistic for interactions")
print("━" * 65)
print()

def compute_2d_pdp(model_fn, X, j, k, n_grid=15, pct=(0.1, 0.9)):
    """2D PDP: joint partial dependence on features j and k."""
    lo_j, hi_j = np.percentile(X[:,j], [pct[0]*100, pct[1]*100])
    lo_k, hi_k = np.percentile(X[:,k], [pct[0]*100, pct[1]*100])
    grid_j = np.linspace(lo_j, hi_j, n_grid)
    grid_k = np.linspace(lo_k, hi_k, n_grid)
    pdp_2d = np.zeros((n_grid, n_grid))
    for a, vj in enumerate(grid_j):
        for b, vk in enumerate(grid_k):
            Xm = X.copy()
            Xm[:, j] = vj; Xm[:, k] = vk
            pdp_2d[a, b] = model_fn(Xm).mean()
    return grid_j, grid_k, pdp_2d


def h_statistic(grid_j, grid_k, pdp_2d, pdp_j, pdp_k):
    """
    H-statistic: quantifies interaction between features j and k.
    H²= Σ(PD_{jk} - PD_j - PD_k)² / Σ PD_{jk}²
    """
    from scipy.interpolate import interp1d
    f_j = interp1d(grid_j, pdp_j, fill_value='extrapolate')
    f_k = interp1d(grid_k, pdp_k, fill_value='extrapolate')
    numerator   = 0.0
    denominator = 0.0
    for a, vj in enumerate(grid_j):
        for b, vk in enumerate(grid_k):
            interaction = pdp_2d[a, b] - f_j(vj) - f_k(vk)
            numerator   += interaction ** 2
            denominator += pdp_2d[a, b] ** 2
    return np.sqrt(numerator / denominator) if denominator > 0 else 0.0


print("  Computing 2D PDP for x0 × x2 (known interaction)...")
_, _, _ = compute_ice(model_fn, X_data, 0, n_grid=12)
grid_j0, _, pdp_j0 = compute_ice(model_fn, X_data, 0, n_grid=12)
grid_k2, _, pdp_k2 = compute_ice(model_fn, X_data, 2, n_grid=12)

print("  Computing 2D PDP for x0 × x3 (no interaction)...")
_, _, pdp_j0b = compute_ice(model_fn, X_data, 0, n_grid=12)
grid_k3, _, pdp_k3 = compute_ice(model_fn, X_data, 3, n_grid=12)

gj0, gk2, pdp2d_02 = compute_2d_pdp(model_fn, X_data, 0, 2, n_grid=12)
gj0b, gk3, pdp2d_03 = compute_2d_pdp(model_fn, X_data, 0, 3, n_grid=12)

h_02 = h_statistic(gj0, gk2, pdp2d_02, pdp_j0, pdp_k2)
h_03 = h_statistic(gj0b, gk3, pdp2d_03, pdp_j0b, pdp_k3)

print()
print(f"  H-statistic results:")
print(f"    x0 × x2 (known interaction):  H = {h_02:.4f}  "
      f"{'✅ Interaction detected' if h_02 > 0.1 else '❌ No interaction'}")
print(f"    x0 × x3 (no interaction):     H = {h_03:.4f}  "
      f"{'⚠️ Spurious?' if h_03 > 0.1 else '✅ No interaction (expected)'}")
print()
print("  H-statistic interpretation:")
print("    H < 0.05:   no meaningful interaction")
print("    H = 0.1–0.3: moderate interaction")
print("    H > 0.3:    strong interaction")
print()

# Text-based 2D PDP heatmap
print("  2D PDP heatmap for x0 × x2 (ASCII):")
print("  (rows=x0 values, cols=x2 values, numbers=avg prediction)")
print()
print(f"  x2 →   {' '.join([f'{v:6.1f}' for v in gk2[::3]])}")
print(f"  {'─'*55}")
for a in range(0, len(gj0), 3):
    row = ' '.join([f'{pdp2d_02[a,b]:+6.2f}' for b in range(0, len(gk2), 3)])
    print(f"  x0={gj0[a]:4.1f} | {row}")
print()
print("  If the matrix shows clear diagonal patterns, there is an interaction.")
print("  Additive features produce rows/columns, not diagonal gradients.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · PDPs with Sklearn, Interpretation Pitfalls, and Full Workflow": {
        "description": (
            "Production PDP workflow using sklearn.inspection. "
            "PartialDependenceDisplay for multiple features simultaneously. "
            "PDPs for regression and classification models. "
            "Demonstrate and explain common PDP interpretation pitfalls. "
            "Rug plots and confidence bands for PDP uncertainty. "
            "Build a complete PDP-based model interpretation report."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  PDPs WITH SKLEARN, PITFALLS, AND FULL WORKFLOW")
print("=" * 65)
print()

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
    from sklearn.datasets import load_breast_cancer, fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import partial_dependence
    HAS_SKL = True
    print("  sklearn available ✅")
except ImportError:
    HAS_SKL = False
    print("  pip install scikit-learn")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — sklearn.inspection API for PDPs")
print("━" * 65)
print()

SKLEARN_API = """
  SKLEARN PARTIAL DEPENDENCE API:

  from sklearn.inspection import partial_dependence, PartialDependenceDisplay

  # --- Compute PDP values programmatically ---
  result = partial_dependence(
      estimator,                     # fitted model (any sklearn estimator)
      X,                             # dataset to marginalise over
      features=[0, 3, (0, 3)],       # list of feature indices (tuple = 2D)
      kind='average',                # 'average' (PDP), 'individual' (ICE), 'both'
      grid_resolution=50,            # number of grid points per feature
      percentiles=(0.05, 0.95),      # clip feature range to these percentiles
      method='auto',                 # 'brute' (model-agnostic) or 'recursion' (tree)
  )
  # result['average']     shape: (n_features, grid_resolution)
  # result['individual']  shape: (n_features, n_samples, grid_resolution)
  # result['grid_values'] list of arrays, one per feature

  # --- Display PDPs (requires matplotlib) ---
  PartialDependenceDisplay.from_estimator(
      estimator, X,
      features=[0, 3, (0, 3)],      # 1D and 2D plots
      kind='both',                   # PDP + ICE overlay
      subsample=500,                 # subsample n points for ICE (speed)
      n_cols=3,                      # columns in subplot grid
      centered=True,                 # center PDP/ICE at first grid point
      ice_lines_kw={'color': 'tab:blue', 'alpha': 0.3},
      pd_line_kw={'color': 'red', 'linewidth': 2},
  )

  # --- Key parameters explained ---
  kind='average':    Only PDP (one curve). Best for global summary.
  kind='individual': Only ICE curves (n lines). Reveals heterogeneity.
  kind='both':       ICE + PDP overlaid. Best for full picture.
  centered=True:     c-ICE: subtract leftmost value so all start at 0.
  subsample=200:     Random subsample for ICE (speed vs completeness).
  method='recursion':Fast O(n) algorithm for tree ensembles (PDPs only).
  method='brute':    O(n*k) model-agnostic (works for any model, ICE too).
"""
print(SKLEARN_API)

if HAS_SKL:
    # Breast cancer classification
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    fn_bc = list(bc.feature_names)
    X_tr_bc, X_te_bc, y_tr_bc, y_te_bc = train_test_split(
        X_bc, y_bc, test_size=0.2, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gbm.fit(X_tr_bc, y_tr_bc)
    acc = gbm.score(X_te_bc, y_te_bc)
    print(f"  GBM on breast cancer: accuracy = {acc:.4f}")
    print()

    print("  Computing PDPs for top 5 features...")
    top5 = np.argsort(gbm.feature_importances_)[::-1][:5]
    t0   = time.perf_counter()
    for j in top5:
        pd_res = partial_dependence(gbm, X_tr_bc, features=[j],
                                     grid_resolution=40, percentiles=(0.05, 0.95))
        grid   = pd_res['grid_values'][0]
        pdp    = pd_res['average'][0]

        # Show key statistics
        lo, hi = pdp.min(), pdp.max()
        effect = hi - lo
        diffs  = np.diff(pdp)
        mono   = "↑" if (diffs >= 0).mean() > 0.9 else ("↓" if (diffs <= 0).mean() > 0.9 else "↕")

        # Find threshold: largest jump
        jump_idx = np.argmax(np.abs(diffs))
        jump_val = grid[jump_idx]
        jump_mag = abs(diffs[jump_idx])

        print(f"  Feature [{fn_bc[j][:30]}]:")
        print(f"    PDP range: [{lo:.4f}, {hi:.4f}]   effect={effect:.4f}   shape={mono}")
        if jump_mag > 0.02:
            print(f"    Largest jump: x={jump_val:.3f}  Δprob={diffs[jump_idx]:+.4f}  "
                  f"(decision threshold region)")
        print()
    t_sk = time.perf_counter() - t0
    print(f"  Computed 5 PDPs in {t_sk:.3f}s")

# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 2 — PDP interpretation pitfalls")
print("━" * 65)
print()

PITFALLS = """
  COMMON PDP PITFALLS AND HOW TO AVOID THEM:

  ── PITFALL 1: Flat PDP ≠ Unimportant Feature ─────────────────────────
  A flat PDP means the AVERAGE prediction is constant across the feature range.
  This can happen when:
    (a) Feature truly has no effect on average (correct conclusion)
    (b) Feature has OPPOSITE effects for different subgroups that CANCEL OUT

  FIX: Always plot ICE curves alongside PDPs.
    If ICE lines are parallel and flat: feature truly unimportant.
    If ICE lines cross and flatten the average: interaction masking true effect.

  ── PITFALL 2: PDP Shows Causal Effect ────────────────────────────────
  WRONG: "The PDP shows that increasing income by $10K raises approval by 5%."
  RIGHT: "On average, applicants with $10K higher income have 5% higher
          predicted approval. This is correlation, not causation."

  The PDP reflects the model's learned relationships, which include:
    - Causal effects (income genuinely enables loan repayment)
    - Spurious correlations (income correlates with education correlates with...)
    - Interaction effects (income only matters if credit score is adequate)

  FIX: Interpret PDPs as describing the MODEL, not the data-generating process.
       Use causal inference methods if you need causal effects.

  ── PITFALL 3: Extrapolation with Correlated Features ─────────────────
  If feature j is correlated with features k, l, m:
    Computing PD_j(v) uses samples with ALL values of k, l, m.
    At extreme values of v, these (v, x_{-j}) combinations may not exist
    in the training data → the model is EXTRAPOLATING.

  FIX: Use ALE plots (PyALE package) when features are correlated.
       Or: restrict the PDP grid to percentiles where coverage is good.
       Check: correlation matrix of features before trusting PDPs.

  ── PITFALL 4: Ignoring the X-axis Distribution ───────────────────────
  The PDP looks at ALL grid points with equal weight.
  But if 95% of data has income < $80K, what happens at $200K matters little
  for actual predictions (but it dominates the visual impression).

  FIX: Add a rug plot (tick marks at actual training data values) below the PDP.
       This shows WHERE data actually is — weight your interpretation accordingly.
       Restrict grid to a narrower percentile range (e.g., 10th–90th percentile).

  ── PITFALL 5: PDPs for Categorical Features ──────────────────────────
  For categorical features, PDP creates bar charts.
  Bar height includes the marginal distribution of all other features.
  Rare categories may have very few samples → high variance, unreliable bars.

  FIX: Show sample count per category alongside the bar chart.
       Flag categories with < 30 samples as "low confidence."
       Consider grouping rare categories before computing PDP.

  ── PITFALL 6: Model-Specific PDP Artefacts ───────────────────────────
  Tree ensembles: PDP shows STEP FUNCTIONS (each step = a split threshold).
    If the step function is too coarse → many grid points in the flat region.
  Linear models: PDP is always a straight line.
    A straight line is perfectly trustworthy but tells you nothing new.
  Neural networks: PDP may be very smooth but slow to compute.
    Consider gradient-based methods (SHAP, IG) instead for neural nets.
"""
print(PITFALLS)

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Bootstrap confidence bands for PDPs")
print("━" * 65)
print()

print("  Bootstrap confidence bands quantify PDP uncertainty.")
print("  Re-train or re-sample many times, compute PDP each time.")
print("  Use the spread across iterations as a confidence interval.")
print()

if HAS_SKL:
    # Bootstrap PDP for one feature
    from sklearn.utils import resample

    feat_idx = top5[0]
    n_boot   = 20   # small for speed (use 100+ in practice)

    pd_ref = partial_dependence(gbm, X_tr_bc, features=[feat_idx],
                                  grid_resolution=30)
    grid  = pd_ref['grid_values'][0]
    pdp_ref = pd_ref['average'][0]

    boot_pdps = []
    for b in range(n_boot):
        X_b, y_b = resample(X_tr_bc, y_tr_bc, random_state=b)
        gbm_b    = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               random_state=b)
        gbm_b.fit(X_b, y_b)
        pd_b   = partial_dependence(gbm_b, X_b, features=[feat_idx],
                                     grid_resolution=30, percentiles=(0.05,0.95))
        # Align grids (may differ slightly)
        from scipy.interpolate import interp1d
        f_b = interp1d(pd_b['grid_values'][0], pd_b['average'][0],
                       fill_value='extrapolate')
        boot_pdps.append(f_b(grid))

    boot_pdps = np.array(boot_pdps)
    pdp_lo    = np.percentile(boot_pdps, 5,  axis=0)
    pdp_hi    = np.percentile(boot_pdps, 95, axis=0)
    pdp_mean  = boot_pdps.mean(axis=0)

    print(f"  Bootstrap PDP ({n_boot} iterations) for '{fn_bc[feat_idx][:30]}':")
    print()
    print(f"  {'x value':>10} | {'PDP mean':>10} | {'5th pctile':>12} | "
          f"{'95th pctile':>13} | {'CI width':>10}")
    print(f"  {'─'*62}")
    for k in range(0, len(grid), 4):
        width = pdp_hi[k] - pdp_lo[k]
        print(f"  {grid[k]:>10.4f} | {pdp_mean[k]:>10.4f} | "
              f"{pdp_lo[k]:>12.4f} | {pdp_hi[k]:>13.4f} | {width:>10.4f}")
    print()
    print("  Wide CI: model is sensitive to training data → PDP is unstable.")
    print("  Narrow CI: PDP is robust and trustworthy.")
    print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Complete PDP workflow and interpretation report")
print("━" * 65)
print()

WORKFLOW = """
  COMPLETE PDP INTERPRETATION WORKFLOW:

  STEP 1: Compute feature importance (any method)
    Use permutation importance or mean|SHAP| to rank features.
    Focus PDP analysis on the TOP 5-10 features.

  STEP 2: Check feature correlations
    Correlation matrix: if |corr| > 0.4 between target feature and others
    → switch to ALE plots for those features.

  STEP 3: Compute 1D PDPs + ICE curves
    kind='both', subsample=200-500
    For each feature:
      - Note the PDP shape (monotone, U, threshold, flat)
      - Check if ICE lines cross (crossing → interaction)
      - Measure effect size (max - min PDP)

  STEP 4: For crossing ICE features
    Colour ICE lines by the suspected interaction partner.
    Compute 2D PDP for the feature pair.
    Compute H-statistic to quantify interaction strength.

  STEP 5: Compute bootstrap confidence bands (optional)
    For the top 3 features, bootstrap 50-100 times.
    Report CI width as a measure of PDP reliability.

  STEP 6: Validate against domain knowledge
    Does direction make sense? Does threshold value match business logic?
    Flag any surprising patterns for investigation.

  STEP 7: Create interpretation report
    Feature effect table: feature name, shape, effect size, interaction flag
    PDP plots for top features (with ICE, rug plot, CI bands)
    2D PDP heatmaps for significant interactions
    ALE plots for correlated features

  SUMMARY TABLE STRUCTURE:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Feature       │ Shape     │ Effect │ Interaction? │ Trustworthy? │
  ├──────────────────────────────────────────────────────────────────────┤
  │ income        │ ↑ monotone│ 0.32   │ ✅ with age   │ ⚠️ correlated │
  │ credit_score  │ threshold │ 0.28   │ ✅ none        │ ✅ yes         │
  │ debt_ratio    │ ↓ monotone│ 0.19   │ ✅ none        │ ✅ yes         │
  │ age           │ U-shaped  │ 0.08   │ ✅ with income │ ⚠️ correlated │
  │ employment    │ step      │ 0.11   │ ✅ none        │ ✅ yes         │
  └──────────────────────────────────────────────────────────────────────┘

  CODE TEMPLATE:
  from sklearn.inspection import partial_dependence, PartialDependenceDisplay
  import numpy as np

  # 1. Compute PDPs
  top_features = np.argsort(model.feature_importances_)[::-1][:10]
  pdp_results  = {}
  for j in top_features:
      result = partial_dependence(model, X_train, features=[j],
                                   kind='both', grid_resolution=50)
      pdp_results[j] = result

  # 2. Effect sizes
  for j, result in pdp_results.items():
      pdp  = result['average'][0]
      ice  = result['individual'][0]
      print(f"{feature_names[j]}: effect={pdp.max()-pdp.min():.3f}, "
            f"ice_std={ice.std():.3f}")

  # 3. Display
  PartialDependenceDisplay.from_estimator(
      model, X_train,
      features=[(j,) for j in top_features[:6]] + [(j, k)],  # add 2D
      kind='both', subsample=300, centered=True,
  )
"""
print(WORKFLOW)

print("━" * 65)
print("  PDP QUICK REFERENCE")
print("━" * 65)
print()
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ Question                    │ Tool                           │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ Average feature effect      │ 1D PDP                         │")
print("  │ Is effect homogeneous?      │ ICE curves (crossing check)    │")
print("  │ What's the PDP uncertainty? │ Bootstrap CI bands             │")
print("  │ Features are correlated?    │ ALE plots instead of PDP       │")
print("  │ Do two features interact?   │ 2D PDP + H-statistic           │")
print("  │ Where is the threshold?     │ Derivative PDP (d-PDP)         │")
print("  │ Summary for executives      │ Effect size bar chart          │")
print("  ├──────────────────────────────────────────────────────────────┤")
print("  │ sklearn API:                                                 │")
print("  │   partial_dependence(model, X, features=[j], kind='both')    │")
print("  │   PartialDependenceDisplay.from_estimator(model, X, ...)     │")
print("  └──────────────────────────────────────────────────────────────┘")
''',
    },
}

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