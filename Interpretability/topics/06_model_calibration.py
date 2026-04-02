"""
Model Calibration — When Confidence Matches Accuracy
=====================================================

A perfectly calibrated model is one where, among all predictions made with
confidence p, exactly a fraction p turn out to be correct. Calibration is
distinct from accuracy: a model can be highly accurate yet poorly calibrated
(overconfident or underconfident), or poorly accurate yet well calibrated
(uncertain about the right answers). For deployed systems — especially in
medicine, finance, law, and autonomous vehicles — knowing HOW MUCH TO TRUST
a model's confidence is as important as the model's accuracy itself.

Calibration matters enormously in production:
    Medical diagnosis:    "85% probability of malignancy" directly informs
                          biopsy decisions. An over-confident model wastes
                          resources; an under-confident one misses disease.
    Credit scoring:       "30% default probability" determines interest rates.
                          Systematic miscalibration transfers wealth from
                          borrowers to lenders or vice versa.
    Autonomous vehicles:  Object detection confidence gates further processing.
                          An over-confident detector misses obstacles.
    Weather forecasting:  Probability of precipitation calibration is the
                          gold standard that meteorologists are evaluated on.
    Recommender systems:  Click probability estimates determine bid prices
                          in real-time advertising auctions worth billions.
    Bayesian pipelines:   Downstream systems that combine model output with
                          other evidence need calibrated probabilities as input.
                          Uncalibrated probabilities corrupt the entire pipeline.

The problem: modern machine learning models are systematically miscalibrated.
Guo et al. (2017) showed that neural networks trained with modern techniques
(batch normalisation, weight decay, residual connections) are significantly
MORE over-confident than older, shallower networks — the very methods that
improved accuracy also worsened calibration.

This module covers the complete calibration pipeline from first principles:
formal definitions, measuring calibration (reliability diagrams, ECE/MCE/ACE/
OE/KCE metrics, Brier scores), diagnosing the pattern of miscalibration, the
full family of post-hoc correction methods (Platt scaling, isotonic regression,
temperature scaling, beta calibration, histogram binning, Dirichlet calibration),
proper scoring rules, Bayesian approaches, out-of-distribution calibration,
training-time calibration, and deploying and monitoring calibration in production.
"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME   = "Model Calibration — When Confidence Matches Accuracy"
DISPLAY_NAME = "06 . Model Calibration"
ICON         = "🎯"
SUBTITLE     = "ECE, reliability diagrams, temperature scaling, and recalibration methods"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHAT CALIBRATION IS AND WHY ACCURACY IS NOT ENOUGH

### The Calibration Definition

A hospital deploys a neural network to predict sepsis risk from patient vitals.
The model outputs "91% probability of sepsis" for a patient who, in fact,
does not have sepsis. Is 91% a reasonable confidence for this prediction?
Only if the model is CALIBRATED: if the model outputs 91% for a class of
similar patients, exactly 91% of those patients should actually have sepsis.

    A classifier is PERFECTLY CALIBRATED if, among all predictions where
    the model outputs probability p, the fraction that are actually positive
    equals p:

    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   P(Y = 1 | f(X) = p) = p    for all p ∈ [0, 1]                    │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

    In English: "When the model says 70%, the event happens 70% of the time."

    This is a FREQUENTIST concept — calibration is about long-run frequencies,
    not about individual predictions. A single prediction of 70% cannot be
    right or wrong in itself; calibration is a property of the distribution
    of predictions over many events.

    f(x) = 0.70 means: "I assign 70% probability to Y=1."

    Perfect calibration REQUIRES: if you collect ALL inputs where f(x) = 0.70
    (or within a small window around 0.70), then exactly 70% of those inputs
    have true label Y=1.

    If instead 90% of those inputs have Y=1: the model is UNDERCONFIDENT
    (its 70% is too low; it should say ~90%).

    If instead 50% of those inputs have Y=1: the model is OVERCONFIDENT
    (its 70% is too high; it should say ~50%).


### Perfect Calibration — Multiclass Case

For a K-class classifier with predicted probability vector ŷ = (ŷ₁,...,ŷK)
and true label Y ∈ {1,...,K}, perfect calibration for class k is:

    P(Y = k | ŷₖ = p) = p    for all k ∈ {1,...,K}, all p ∈ [0, 1]

This is CLASSWISE calibration. A weaker requirement is CONFIDENCE calibration,
which only requires that the TOP-1 predicted class's confidence is calibrated:

    P(Y = ŷ* | max_k(ŷₖ) = p) = p

where ŷ* = argmax_k(ŷₖ) is the predicted class.

Confidence calibration is easier to achieve and most commonly studied.
A model can be confidence-calibrated but not classwise-calibrated if,
for example, its 90% predictions are always correct (good confidence
calibration) but its second-place probabilities are systematically wrong
(poor classwise calibration).


### Accuracy vs Calibration: They Are Independent

    These two properties are ORTHOGONAL — a model can be:

    HIGH accuracy, GOOD calibration:  ideal. Predicts correctly AND
        assigns confident probabilities only when actually correct.

    HIGH accuracy, BAD calibration:  common with modern neural nets.
        Gets the right CLASS (accuracy ≥ threshold) but assigns
        probabilities that don't match true frequencies.
        Example: predicts 0.95 when true probability is 0.75.

    LOW accuracy, GOOD calibration:  a model that says 0.51 for every
        prediction on a balanced task. Always correctly uncertain.
        Useless for decision-making despite being "well-calibrated."

    LOW accuracy, BAD calibration:  the worst case.

    Implication: you must evaluate BOTH accuracy/AUC AND calibration.
    High AUC does NOT imply good calibration.


##### PART 2 — MEASURING CALIBRATION: RELIABILITY DIAGRAMS AND METRICS

### The Reliability Diagram (Calibration Curve)

The reliability diagram (DeGroot & Fienberg, 1983; Niculescu-Mizil & Caruana,
2005) is the standard visual tool for assessing calibration. It plots:

    X-axis: predicted confidence (binned into intervals, e.g., [0,0.1), [0.1,0.2), ...)
    Y-axis: observed accuracy within each bin (fraction of correct predictions)

A perfectly calibrated model produces a DIAGONAL LINE (the identity line):
predicted confidence = observed accuracy for every bin.

    Algorithm:
        1. Collect predicted probabilities p̂_i and true labels y_i.
        2. Sort predictions into M equal-width bins (e.g., [0,0.1), [0.1,0.2), ...).
        3. For each bin m:
               mean_confidence_m = average of p̂_i for samples in bin m
               fraction_positive_m = fraction of y_i = 1 in bin m
        4. Plot: x = mean_confidence, y = fraction_positive.
        5. The diagonal (y=x) is perfect calibration.

    Reading the plot:
        Points ABOVE the diagonal: UNDER-CONFIDENT (model is too low)
            → Actual frequency higher than predicted probability.
        Points BELOW the diagonal: OVER-CONFIDENT (model is too high)
            → Actual frequency lower than predicted probability.
        Points ON the diagonal: perfectly calibrated.

    Common patterns:
        S-curve (below left, above right): sigmoid-shaped miscalibration.
            Model is too extreme at both ends.
            Typical of models trained to maximise log-likelihood.
        Flat curve near 0.5: under-confident (random forests).
        Systematically below diagonal: over-confident (neural nets).

The reliability diagram should also show a HISTOGRAM of prediction counts per
bin. Bins with very few predictions produce unreliable accuracy estimates and
should be treated with caution.


### Expected Calibration Error (ECE)

The most widely used calibration metric (Naeini, Cooper & Hauskrecht, 2015).
It discretises the confidence range into M bins and computes the weighted
average absolute difference between accuracy and confidence within each bin:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   ECE = Σ_{m=1}^{M}  (|B_m| / n)  × |acc(B_m) - conf(B_m)|      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Where:
        M           = number of bins (typically 10 or 15)
        B_m         = set of predictions in bin m
        |B_m|       = number of predictions in bin m
        n           = total number of predictions
        acc(B_m)    = fraction of correct predictions in bin m
        conf(B_m)   = mean predicted confidence in bin m
        |B_m|/n     = weight (fraction of total samples in this bin)

    ECE = 0.0: perfect calibration.
    ECE = 0.1: on average, predicted probabilities are 10 percentage
               points away from actual frequencies.
    ECE < 0.02: well-calibrated (competitive benchmarks).
    ECE > 0.05: poor calibration (usually worth fixing).

    Computing ECE by hand — a minimal example (M=3 bins, n=10 predictions):

    Bin 1: [0, 0.4)  —  3 predictions, avg confidence = 0.25, accuracy = 0/3 = 0.0
    Bin 2: [0.4,0.7) —  4 predictions, avg confidence = 0.55, accuracy = 2/4 = 0.5
    Bin 3: [0.7, 1.0] — 3 predictions, avg confidence = 0.85, accuracy = 3/3 = 1.0

    ECE = (3/10)|0.0 − 0.25| + (4/10)|0.5 − 0.55| + (3/10)|1.0 − 0.85|
        = 0.3×0.25        + 0.4×0.05         + 0.3×0.15
        = 0.075 + 0.020 + 0.045
        = 0.140   →   ECE = 14.0% — moderate miscalibration.


### Limitations of ECE

    PROBLEM 1 — BIN BOUNDARY SENSITIVITY:
    ECE depends on where you place the bin boundaries. Equal-width bins
    treat each 10% range equally, but most predictions may cluster near the
    high end for overconfident models. Equal-mass bins (each bin contains n/M
    predictions) avoid sparse bins but change the measured ECE.

    PROBLEM 2 — OVERCONFIDENT MODELS CAN GAME ECE:
    A model with 100% accuracy can have low ECE even if it always predicts 1.0.
    ECE rewards accuracy alongside calibration.

    PROBLEM 3 — FINITE SAMPLE BIAS:
    ECE is a biased estimator; bias is downward. With one prediction per bin,
    ECE = 0 by definition — clearly wrong.

    PROBLEM 4 — INSENSITIVITY TO TAILS:
    ECE weights bins by prediction count, underweighting tail bins. But
    calibration at extreme confidences matters most for safety decisions.


### Alternative and Complementary Calibration Metrics

    MAXIMUM CALIBRATION ERROR (MCE):
    MCE = max_m |acc(B_m) − conf(B_m)|
    Reports the WORST-CASE bin rather than the average.
    More conservative; more sensitive to tail miscalibration.
    Preferred in safety-critical applications.

    ADAPTIVE ECE (ACE / AECE, Nixon et al., 2019):
    Uses equal-mass bins (each bin has the same number of predictions).
    Avoids the sparse-bin problem of equal-width ECE.
    More stable estimates for skewed confidence distributions.

    OVERCONFIDENCE ERROR (OE):
    OE = Σ_m (|B_m|/n) × max(conf(B_m) − acc(B_m), 0)
    Only penalises overconfidence (confidence > accuracy).
    Preferred when overconfidence is the primary safety concern.

    KERNEL CALIBRATION ERROR (KCE, Widmann et al., 2019):
    Replaces discrete bins with a kernel smoothing of the calibration function.
    Produces a smooth, continuous estimate of the calibration curve.
    Avoids all binning artifacts but requires kernel bandwidth selection.

    EXPECTED NORMALIZED CALIBRATION ERROR (ENCE, Levi et al., 2022):
    For regression calibration (predicting continuous outputs with uncertainty).
    Measures whether predicted variance matches actual error variance.

    NEGATIVE LOG-LIKELIHOOD (NLL) / LOG-LOSS:
    NLL = −(1/n) Σᵢ yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)
    A proper scoring rule that jointly rewards accuracy AND calibration.
    Minimising NLL produces calibrated probabilities (in theory).

    BRIER SCORE:
    BS = (1/n) Σᵢ (ŷᵢ − yᵢ)²
    Another proper scoring rule. Decomposable as:
    BS = Uncertainty − Resolution + Reliability (Murphy decomposition)
    The "Reliability" component IS the calibration error.


##### PART 3 — WHY MODERN NEURAL NETWORKS ARE MISCALIBRATED

### The Guo et al. (2017) Finding

Guo, Pleiss, Sun & Weinberger — "On Calibration of Modern Neural Networks"
(ICML 2017): modern deep neural networks have gotten MORE ACCURATE over the
past decade, but also MORE OVERCONFIDENT (worse calibration), simultaneously.

    • LeNet (1998): test accuracy 99.0%, ECE ≈ 1.5%.
    • VGG-16 (2014): test accuracy 90.4%, ECE ≈ 4.7%.
    • ResNet-110 (2016): test accuracy 93.6%, ECE ≈ 4.4%.

Deeper, more accurate models are MORE miscalibrated. Understanding why:


### Root Cause 1 — Softmax Does Not Produce Calibrated Probabilities

The softmax function maps logits (unbounded scores) to probabilities:

    p̂ₖ = exp(zₖ) / Σⱼ exp(zⱼ)

Softmax is a normalisation function, not a probability computation. The ratio
exp(zₖ)/Σⱼ exp(zⱼ) reflects the RELATIVE ranking of logits, not the absolute
probability of correctness.

    The softmax illusion:

    Three-class problem. True class = class 1.
    Case A — logits: [2.0, 1.0, 0.0]
    Softmax output: [0.665, 0.245, 0.090]
    Model predicts class 1 with confidence 0.665.

    Case B — logits: [20.0, 10.0, 0.0]  (same RANKING, 10× magnitude)
    Softmax output: [0.99995, 0.0000454, 0.0000000206]
    Model predicts class 1 with confidence ≈ 1.000.

    Both logit vectors rank class 1 first, but Case B's softmax output
    is near 1.0 while Case A is 0.665. Large logit magnitudes → high
    softmax confidence → overconfidence.


### Root Cause 2 — Overparameterisation and Cross-Entropy Pressure

Modern neural networks are overparameterised — vastly more parameters than
training examples. Cross-entropy loss NEVER stops pushing toward overconfidence:

    Loss for a correct prediction ŷ is: −log(ŷ).
    This reaches 0 only as ŷ → 1, which requires logit magnitude → ∞.
    The loss ALWAYS has a gradient pushing logits larger for correct predictions.

    Cross-entropy training never stops pushing toward overconfidence.
    Weight decay and dropout IMPROVE calibration as side effects (they limit
    logit magnitude growth) but are indirect and insufficient.


### Root Cause 3 — The Train-Test Accuracy Gap

If the model achieves 98% training accuracy but 88% test accuracy:

    During training:  model achieves 98% accuracy → logits are calibrated
                      for 98% confidence → model "knows" it is right a lot.

    At test time:     true accuracy is 88%.
                      The model STILL emits confidence levels calibrated
                      for 98% accuracy. The gap (10%) appears as
                      systematic overconfidence in the reliability diagram.


### Root Cause 4 — Dataset Shift (Distribution Shift)

Even a well-calibrated model at training distribution becomes miscalibrated
under distribution shift — when test inputs differ from training inputs.
In-distribution calibration ≠ Out-of-distribution calibration.
This is one of the most important practical limitations of all post-hoc
calibration methods: they calibrate on a held-out set from the SAME
distribution as training, but do not guarantee calibration under shift.


### Miscalibration by Model Family

    GRADIENT BOOSTING: systematically over-confident.
        As boosting progresses, predictions become more extreme.
        The final ensemble assigns probabilities near 0 or 1 more often
        than the true distribution warrants.

    LOGISTIC REGRESSION: often well-calibrated when its assumptions hold.
        Can become miscalibrated with: multicollinearity, non-linear features,
        class imbalance without re-weighting.

    RANDOM FORESTS: tends to be UNDER-confident.
        Predictions are averages of 0/1 leaf values, compressing toward 0.5.
        More trees = more averaging = predictions compressed toward 0.5.


### The Consequence of Miscalibration in Practice

    OVER-CONFIDENCE (predicted p > actual frequency):
        Model says 90% confident — but only right 70% of the time.
        Downstream decisions are made with false certainty.

    UNDER-CONFIDENCE (predicted p < actual frequency):
        Model says 60% confident — but actually right 80% of the time.
        In bidding systems: under-bidding → lost revenue.
        In medical: unnecessary consultations or tests.


##### PART 4 — POST-HOC CALIBRATION METHODS

### The Post-Hoc Calibration Framework

Post-hoc calibration methods transform a model's OUTPUTS (logits or probabilities)
using a second, small model trained on a held-out calibration set, WITHOUT
modifying the original model's weights.

    Training data → TRAIN main model f → f is overconfident
    Calibration data → TRAIN calibrator g on f's outputs → g corrects confidence
    New input → f(x) → g(f(x)) = calibrated probability

The calibration set is typically a held-out validation set, SEPARATE from
both the training set (used to train f) and the test set (used for final
evaluation). Using the training set to calibrate would cause overfitting.


### Method 1 — Temperature Scaling

Temperature scaling (Guo et al., 2017) is the simplest, most widely used,
and often most effective post-hoc calibration method.

The idea: divide ALL logits by a single scalar temperature parameter T > 0
before applying softmax:

    p̂ₖ = exp(zₖ / T) / Σⱼ exp(zⱼ / T)

T is chosen to MINIMISE the negative log-likelihood on the calibration set.
This is a ONE-PARAMETER optimisation, solvable exactly by line search.

    T > 1:  SOFTER distribution — probabilities become less extreme.
            Used for overconfident models.
            Example: T=2 applied to [0.97, 0.02, 0.01] → [0.73, 0.15, 0.12].

    T < 1:  SHARPER distribution — probabilities become more extreme.
            Used for underconfident models (rare in deep learning).

    T = 1:  Original softmax output (no change).

    Temperature has a physical analogy from statistical mechanics:
    the Boltzmann distribution for energy states at temperature T is:
      P(state = k) ∝ exp(−E_k / T)

    High T → uniform distribution (high entropy, maximum uncertainty)
    Low T  → concentrated distribution (low entropy, high confidence)

    For a 2-class problem with logits [2.5, −2.5]:
    T=1.0: softmax → [0.993, 0.007]  (very confident)
    T=2.0: softmax → [0.924, 0.076]  (less confident, more calibrated)
    T=4.0: softmax → [0.731, 0.269]  (much less confident)
    T=10.0: softmax → [0.537, 0.463] (near-uniform)

    KEY PROPERTY — Temperature scaling preserves accuracy:
    argmax_k(zₖ/T) = argmax_k(zₖ) for any T > 0.
    TOP-1 accuracy is COMPLETELY PRESERVED. Only confidence values change.

    LIMITATION: Temperature scaling is a GLOBAL adjustment. If overconfidence
    is different at different confidence levels, a single T cannot fix
    heterogeneous miscalibration across the probability range.


### Method 2 — Platt Scaling

Platt scaling (Platt, 1999) trains a logistic regression model on the
model's logits (or scores) to produce calibrated probabilities:

    p̂ = σ(A × s + B) = 1 / (1 + exp(−(A×s + B)))

Parameters A and B are found by minimising binary cross-entropy on the
calibration set. It is the TWO-PARAMETER generalisation of temperature scaling:
setting A=1/T, B=0 recovers temperature scaling.

    If A > 0 and B = 0: just scaling the logit — compresses or expands.
    If A < 1: model was over-confident (shrinks logit toward 0).
    B ≠ 0: adjusts for prior probability shift (base rate mismatch).

    STRENGTHS:
        Very few parameters (just A and B) → doesn't overfit easily.
        Works well with small calibration datasets (few hundred samples).
        Preserves ranking (monotone transformation of scores).
        Fast: just one logistic regression fit.

    LIMITATIONS:
        Parametric: assumes the calibration curve is sigmoid-shaped.
        If the miscalibration is more complex (S-curve with local bumps),
        Platt scaling cannot capture it.
        May not work well for multi-class problems (need per-class calibration
        or a multinomial extension like Dirichlet calibration).

    For large calibration sets: Platt scaling can outperform temperature scaling.
    For small calibration sets: temperature scaling is safer (1 param vs 2).


### Why a Separate Calibration Set?

    If you calibrate on the SAME data used to train the model:
        The model has already fit those probabilities.
        The calibration fitter also overfits those probabilities.
        Result: calibration appears good on training data but is
        still wrong on test data.

    Option A: 3-way split: train / calibration / test.
    Option B: k-fold cross-validated calibration (sklearn's CalibratedClassifierCV).


### Method 3 — Isotonic Regression

Isotonic regression (Zadrozny & Elkan, 2002) is a NON-PARAMETRIC calibration
method that fits a piecewise-constant, monotonically non-decreasing function
from predicted probability to calibrated probability.

Given pairs (p̂ᵢ, yᵢ) on the calibration set, find the function g such that:
    Σᵢ (yᵢ − g(p̂ᵢ))² is minimised
    subject to:  g(p̂ᵢ) ≤ g(p̂ⱼ)  when  p̂ᵢ ≤ p̂ⱼ

The POOL ADJACENT VIOLATORS (PAV) algorithm solves this in O(n log n).

    PAV algorithm worked example:

    Calibration data (sorted by p̂ ascending):
    p̂ = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
    y  = [0,   1,   0,   1,   1,   1  ]

    Start: each prediction in its own group.
    Group accuracies: [0, 1, 0, 1, 1, 1]

    Violating pair (0→1): merge groups 1 and 2:
    Groups: [0,1] [0] [1] [1] [1]    Accuracies: [0.5, 0, 1, 1, 1]

    Violating pair (0.5→0): merge groups 1-2 and 3:
    Groups: [0,1,0] [1] [1] [1]      Accuracies: [0.33, 1, 1, 1]

    No more violations. Final calibration:
    p̂=0.2 → 0.33, p̂=0.3 → 0.33, p̂=0.5 → 0.33
    p̂=0.6 → 1.0,  p̂=0.7 → 1.0,  p̂=0.8 → 1.0

Isotonic regression is MORE FLEXIBLE than Platt scaling — it can correct
any monotone calibration error, not just linear scaling. But it has higher
variance and is prone to overfitting with small calibration sets.

    Platt < Isotonic in bias (isotonic fits better in principle)
    Platt > Isotonic in variance (fewer params, less overfitting risk)
    Use isotonic regression with ≥ 5,000 calibration examples.
    Use Platt scaling / temperature scaling with fewer.


### Method 4 — Beta Calibration

Beta calibration (Kull, Silva Filho & Flach, 2017) is specifically designed
for binary classifiers where the uncalibrated outputs have a beta-like
distribution. The calibration function:

    p̂_calibrated = σ(a · log(p̂) + b · log(1 − p̂) + c)

with three parameters a, b, c. This generalises logistic regression and
can correct for sigmoid-shape miscalibration — a common S-shape pattern
that simple scaling cannot fix.

    Special cases:
    a=1, b=1: reduces to Platt scaling.
    a=b, c=0: temperature scaling.

Beta calibration typically outperforms Platt scaling when the uncalibrated
probability distribution has substantial skewness or bimodality.


### Method 5 — Histogram Binning

Histogram binning (Zadrozny & Elkan, 2001) divides the predicted probability
range [0,1] into M bins and replaces all predictions within a bin by the
empirical accuracy of predictions in that bin on the calibration set:

    ĝ(p̂) = acc(Bᵢ)   if p̂ ∈ Bᵢ

This guarantees that the reliability diagram shows a perfect diagonal
(by definition). BUT it produces a degenerate solution: the calibrated output
takes only M values. It has poor resolution and high variance on small
calibration sets. Primarily used as a diagnostic comparison rather than a
production calibration method.


### Method 6 — Dirichlet Calibration (Multi-Class)

Dirichlet calibration fits a K×K linear map on the log-probability simplex.
More accurate than per-class calibration for multi-class settings.
Implementation: dirichletcal Python package.


### sklearn Integration

    from sklearn.calibration import CalibratedClassifierCV, calibration_curve

    # Cross-validated calibration (recommended for production)
    cal_model = CalibratedClassifierCV(base_model,
                    method='sigmoid',  # Platt scaling; use 'isotonic' for n > 1000
                    cv=5)              # or 'prefit' if you have a separate cal set
    cal_model.fit(X_train, y_train)
    p_cal = cal_model.predict_proba(X_test)[:, 1]

    # Reliability diagram
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10)


##### PART 5 — PROPER SCORING RULES AND THE MURPHY DECOMPOSITION

### Proper Scoring Rules

A PROPER SCORING RULE is a loss function S(p̂, y) for probabilistic predictions
such that the expected score is MINIMISED by the true probability:

    E_y[S(p*, y)] ≤ E_y[S(p̂, y)]   for any p̂ ≠ p*

Proper scoring rules incentivise honest probability reporting: you cannot
improve your expected score by reporting something other than your true belief.

    STRICTLY PROPER: the minimum is achieved ONLY at p̂ = p*.

    LOG-LOSS (Negative Log-Likelihood):
    S(p̂, y) = −y log(p̂) − (1−y) log(1−p̂)
    Expected log-loss is minimised at the true probability p*.
    Penalises confident wrong predictions HEAVILY (log penalty → ∞ as p̂ → 0).

    BRIER SCORE:
    S(p̂, y) = (p̂ − y)²
    Expected Brier score is minimised at the true probability.
    Penalises wrong predictions QUADRATICALLY (bounded, unlike log-loss).
    Decomposable into calibration and resolution components.

Any proper scoring rule gives the right gradient signal for calibration.
Cross-entropy training with infinite data and a perfect model would produce
perfectly calibrated outputs. The miscalibration we observe comes from
finite data, finite model capacity, and optimisation stopping before convergence.

Log-loss penalises confident wrong predictions EXPONENTIALLY.
A perfectly calibrated model minimises log-loss for a given AUC.
Over-confident predictions incur much higher log-loss than calibrated ones.


### The Murphy Decomposition of the Brier Score

The Brier score decomposes into three interpretable components:

    BS = UNC − RES + REL

where:
  • UNC (Uncertainty) = p̄(1 − p̄), where p̄ is the base rate.
    This is IRREDUCIBLE — a property of the dataset, not the model.
    Higher base rate variance → higher irreducible uncertainty.

  • RES (Resolution) = (1/n) Σₖ |Bₖ| (p̄ₖ − p̄)²
    How much the model's predictions DISCRIMINATE from the base rate.
    Higher resolution = better at separating positives from negatives.
    This is analogous to AUC — measuring discrimination.

  • REL (Reliability = Calibration) = (1/n) Σₖ |Bₖ| (p̄ₖ − fₖ)²
    How well the model's confidence matches its accuracy.
    REL = 0 means perfect calibration. Higher REL = worse calibration.

    Murphy decomposition — reading the numbers:

    Model A: BS = 0.12, UNC = 0.25, REL = 0.04, RES = 0.17
    Interpretation: fairly well calibrated (REL=4%), good discrimination.

    Model B: BS = 0.08, UNC = 0.25, REL = 0.15, RES = 0.32
    Interpretation: better Brier score, but achieved via high discrimination
    (RES=32%) DESPITE poor calibration (REL=15%). Model B is overconfident.

    Model C: BS = 0.25, UNC = 0.25, REL = 0.00, RES = 0.00
    Interpretation: perfectly calibrated (REL=0) but useless — it always
    predicts the base rate. Resolution = 0 means no discrimination.

    For most applications: prefer Model A (balanced calibration + discrimination).

This decomposition shows that Brier score is a joint measure of accuracy
(resolution) AND calibration (reliability). Improving calibration directly
improves Brier score without changing model discrimination.


##### PART 6 — BAYESIAN APPROACHES TO CALIBRATION

### Why Bayesian Methods Are Naturally Calibrated

A perfectly specified Bayesian model with the correct prior and likelihood
produces calibrated posteriors by definition. The posterior P(Y|X, data)
represents the actual uncertainty given the observed evidence.

    Single model: f_ŵ(x) = [0.97, 0.02, 0.01]  (very overconfident)

    Bayesian ensemble of 5 models (different weight samples):
    f_{w₁}(x) = [0.97, 0.02, 0.01]
    f_{w₂}(x) = [0.85, 0.10, 0.05]
    f_{w₃}(x) = [0.91, 0.06, 0.03]
    Mean: [0.91, 0.06, 0.03] — less confident than any individual model.
    Disagreement across models (variance in predictions) represents model
    uncertainty. High variance → averaged output is less confident.


### Monte Carlo Dropout (Gal & Ghahramani, 2016)

Apply dropout at TEST TIME and run multiple forward passes. Each pass with
different dropout masks approximates a sample from the posterior over weights.

    For N stochastic forward passes with dropout:
    p̂_mean(y|x) = (1/N) Σᵢ₌₁ᴺ f_{dropout_i}(x)   ← predictive mean
    p̂_uncertainty(x) = Var_i[f_{dropout_i}(x)]     ← predictive variance

The predictive mean is better calibrated than a single forward pass.
Cost: N forward passes instead of 1. Typical N: 30-100.
Important caveat: MC Dropout does NOT always improve calibration — on many
benchmarks, temperature scaling outperforms MC Dropout for calibration.


### Deep Ensembles (Lakshminarayanan et al., 2017)

Train M independently initialised models on the SAME dataset. At test time,
average their predictions:

    p̂(y|x) = (1/M) Σₘ f_m(x)

Deep ensembles are the empirical gold standard for calibration:
  • Better calibration than any single model (averaging reduces overconfidence)
  • Better accuracy than any single model (averaging reduces variance)
  • Better out-of-distribution detection (disagreement across models signals OOD)

Cost: M × training and inference cost. Typical M: 5.

Why ensembles improve calibration:
  1. DIVERSITY: different random initialisations explore different loss-landscape regions.
  2. AVERAGING: the mean of M predictions with different biases is less biased.
  3. VARIANCE REDUCTION: averaging reduces estimate variance → more reliable confidence.


##### PART 7 — MULTI-CLASS AND REGRESSION CALIBRATION

### Multi-Class Calibration

For a K-class classifier producing probabilities p = (p_1, ..., p_K):

    CLASSWISE-ECE (CW-ECE):
        Compute ECE separately for each class (one-vs-rest).
        Average over K classes. Most commonly reported.

    TOP-LABEL ECE:
        Only consider the predicted class (the argmax).
        Bin by the maximum predicted probability.
        acc(B_m) = fraction of predictions in bin m where argmax was correct.
        This is what sklearn's calibration_curve implements.

    ALL-CLASS ECE:
        Consider all K predicted probabilities, not just the top.
        Much more stringent — requires calibration across all class-confidence pairs.


### Regression Calibration

For regression models producing point predictions, calibration is framed
around predictive intervals:

    INTERVAL CALIBRATION:
        A 90% predictive interval should contain the true value 90% of the time.
        P(y in [L, U]) = 0.90 for a 90% interval.

    QUANTILE CALIBRATION:
        A predicted α-quantile should be exceeded (1-α) fraction of the time.
        P(y > q_α(x)) = 1 - α.

The two sources of uncertainty in regression:
  • ALEATORIC UNCERTAINTY: inherent noise in the data (irreducible). Even with
    a perfect model, y is not fully determined by x.
  • EPISTEMIC UNCERTAINTY: uncertainty due to limited data. Could be reduced
    with more training data or a better model.

Common regression calibration metrics:
    Coverage rate: fraction of true values inside predicted intervals.
    PICP (Prediction Interval Coverage Probability) = empirical coverage.
    Average interval width: wider intervals are easier to calibrate.
    Sharpness: how narrow the calibrated intervals are (narrower = better).

For a sequence of nominal coverage levels α ∈ {0.1, 0.2, ..., 0.9}:
    Compute the α-prediction interval [l_α(x), u_α(x)] for each test input.
    Measure empirical coverage = fraction of y ∈ [l_α, u_α].
    Plot: nominal (x-axis) vs empirical (y-axis). Calibrated = diagonal.

Regression calibration methods:
    Conformal prediction: distribution-free coverage guarantees.
    Normalising flows: calibrated density estimation.
    Natural gradient / Bayesian methods: principled uncertainty.


##### PART 8 — OUT-OF-DISTRIBUTION CALIBRATION AND SELECTIVE PREDICTION

### The In-Distribution vs Out-of-Distribution Calibration Gap

A model calibrated on the training distribution may be severely miscalibrated
on inputs that differ from that distribution (OOD inputs). Worse: models are
typically OVERCONFIDENT on OOD inputs.

    OOD overconfidence — the phenomenon:

    Model trained on ImageNet (1000 object classes).
    Test input: a random Gaussian noise image (not a real object).

    Expected behaviour: roughly uniform distribution over 1000 classes.

    Actual behaviour (typical DNN): one class receives probability 0.95+.
    This is a consequence of the softmax architecture: softmax ALWAYS produces
    a distribution over the training classes, regardless of whether the input
    resembles any training class. It cannot output "none of the above."

Ovadia et al. (2019): All methods DEGRADE under distribution shift.
  1. Deep ensembles degrade the LEAST under moderate shift.
  2. Temperature scaling calibrated in-distribution can become WORSE than
     no calibration under large distribution shifts.
  3. MC Dropout is unreliable under shift — sometimes better, sometimes worse.

The practical implication: post-hoc calibration should be recalibrated
periodically on data from the actual deployment distribution.


### Selective Prediction — The Abstention Option

A complementary approach to calibration: allow the model to ABSTAIN on
inputs where it is uncertain, rather than forcing a low-confidence prediction.

    Selective prediction framework:
    At confidence threshold τ:
      If p̂_max ≥ τ: make the prediction.
      If p̂_max < τ: abstain ("I don't know").

    Coverage = fraction of inputs the model predicts on.
    Quality (accuracy on covered inputs) typically increases with τ.

    Coverage-Quality trade-off curve: analogous to precision-recall.
    At τ=0: predict everything, quality = overall accuracy.
    At τ=1: predict nothing (zero coverage).

For calibrated models, the coverage-quality curve is more predictable and
reliable. Miscalibrated models have unreliable coverage-quality curves.


### Conformal Prediction — Distribution-Free Calibration

Conformal prediction (Vovk, Gammerman & Shafer, 2005) produces calibrated
prediction intervals/sets with FINITE-SAMPLE COVERAGE GUARANTEES — no
distributional assumptions required.

    SPLIT CONFORMAL PREDICTION:

    Given a calibration set {(xᵢ, yᵢ)}ᵢ₌₁ⁿ and a trained model f:

    Step 1: Compute nonconformity scores for each calibration example:
      sᵢ = |yᵢ − f(xᵢ)|   (absolute residual for regression)
      sᵢ = 1 − p̂(true class)  (for classification)

    Step 2: Find the (1−α)-quantile of the nonconformity scores:
      q̂ = quantile(s₁, ..., sₙ, level = ⌈(n+1)(1−α)⌉/n)

    Step 3: For a new input x:
      Regression: prediction interval [f(x) − q̂, f(x) + q̂]
      Classification: prediction set = {k : 1 − p̂ₖ ≤ q̂}

    THEOREM: The conformal prediction interval covers the true y with
    probability ≥ 1−α, for any distribution of (x, y), with any model f.

    This is a FINITE-SAMPLE GUARANTEE: not asymptotic, not approximate.
    Unlike ECE or temperature scaling, this guarantee is non-asymptotic.

    Bayesian 90% credible interval: contains y with 90% probability GIVEN
    correct model specification. Fails if the model is misspecified.

    90% conformal prediction interval: contains y with AT LEAST 90%
    probability, period. No model specification assumptions required.
    Can be applied to ANY trained model.


##### PART 9 — TRAINING-TIME CALIBRATION

### Label Smoothing

Label smoothing (Szegedy et al., 2016) replaces hard labels (0 or 1) with
soft targets during training:

    ỹₖ = (1 − ε) · yₖ + ε/K

where yₖ is the one-hot label, K is the number of classes, and ε is the
smoothing parameter (typically ε = 0.1).

    Without smoothing: training pushes p̂_correct → 1.0 (logit → ∞).
    With ε=0.1 smoothing: training pushes p̂_correct → 0.9 (finite logit).
    The maximum "correct" confidence the model can achieve is now 0.9 − ε/K.

    This bounds logit magnitudes, preventing the most extreme overconfidence.
    Müller et al. (2019): label smoothing produces better-calibrated models
    but can slightly harm accuracy if ε is too large.

    Approximate equivalence: training with label smoothing ε ≈ post-hoc
    temperature scaling with T ≈ 1/(1-ε) (they have similar effects).


### Focal Loss and Its Calibration Implications

Focal loss (Lin et al., 2017) was designed for class imbalance:

    FL(p̂, y) = −y (1 − p̂)^γ log(p̂) − (1−y) p̂^γ log(1−p̂)

The (1−p̂)^γ factor reduces the loss for high-confidence predictions. This
keeps the model focused on hard examples, but can worsen overconfidence on
easy examples. Calibration effect is mixed: depends on γ and the data.


### Explicitly Calibrated Training Objectives

    CALIBRATION LOSS REGULARISATION (Kumar et al., 2018):
    Add ECE as a regulariser to the cross-entropy loss:
    L = CE(p̂, y) + λ × ECE(p̂, y)

    The problem: ECE is not differentiable (it involves sorting and binning).
    Approximations: use kernel-smoothed ECE (MMCE) or differentiable surrogates.

    MAXIMUM MEAN CALIBRATION ERROR (MMCE, Kumar et al., 2018):
    A kernel-based, differentiable calibration regulariser that can be added
    directly to the training loss and backpropagated through:
    MMCE = ‖E[(p̂ − y) k(p̂, ·)]‖_H
    where k is a kernel function and H is a reproducing kernel Hilbert space.


##### PART 10 — DIAGNOSING MISCALIBRATION PATTERNS

### Pattern 1: Systematic Over-Confidence

    Reliability diagram: ALL points consistently BELOW the diagonal.
    ECE contribution: high in middle confidence bins (0.4–0.8).
    Cause: model assigns higher probabilities than observed frequencies.
    Common in: neural networks, gradient boosting, SVMs.
    Fix: temperature scaling (T > 1) or Platt scaling.

### Pattern 2: Systematic Under-Confidence

    Reliability diagram: ALL points consistently ABOVE the diagonal.
    ECE contribution: points cluster near 0.5 (centred distribution).
    Cause: predictions compressed toward 0.5 (averaging effect).
    Common in: random forests, naive Bayes with strong feature independence.
    Fix: temperature scaling (T < 1) or Platt scaling.

### Pattern 3: S-Shaped Miscalibration

    Reliability diagram: S-curve — below diagonal at low confidence,
    above diagonal at high confidence. Extreme probabilities are wrong in
    BOTH directions.
    Cause: log-loss training with insufficient regularisation.
    Fix: isotonic regression captures this complex shape. Platt scaling
    is insufficient (cannot model the S).

### Pattern 4: Middle-Range Miscalibration Only

    Reliability diagram: ends are calibrated, middle is off.
    Usually indicates class imbalance issues — the prior probability
    hasn't been correctly accounted for.
    Fix: isotonic regression, or prior adjustment.

### Pattern 5: High-Confidence Miscalibration

    Reliability diagram: mostly on diagonal, but high-confidence end (0.9+)
    deviates strongly.
    ECE is low, but MCE is high.
    Most dangerous for decision-making: the model is wrong when it's most sure.
    Fix: focus on the 0.8–1.0 range; add uncertainty quantification methods
    (conformal prediction, deep ensembles).


### The Calibration-Sharpness Trade-off

    A model that ALWAYS predicts the base rate (e.g., always 0.3) is
    perfectly calibrated but has zero discriminative value (zero resolution).

    The goal: find the BEST-CALIBRATED model among those with HIGH discrimination.

    This is why calibration metrics (ECE) should ALWAYS be reported alongside
    discrimination metrics (AUC, accuracy).

    Proper scoring rules (like Brier score and log-loss) reward BOTH:
        Good discrimination (resolution) AND good calibration (reliability).
        They are the theoretically correct objectives for probabilistic forecasting.


##### PART 11 — PRODUCTION MONITORING AND RECALIBRATION

### The Calibration Monitoring Problem

A model calibrated at deployment time may become miscalibrated over time
due to distribution shift.

    Types of shift affecting calibration:
        Covariate shift: feature distribution p(X) changes.
            → May not affect calibration if model's p(Y|X) is correct.
        Label shift: class frequencies p(Y) change.
            → Directly affects calibration. A model calibrated at 30%
               prevalence becomes over-confident at 10% prevalence.
        Concept drift: the true relationship p(Y|X) changes.
            → Calibration may deteriorate in unpredictable ways.

    WHAT TO MONITOR:
        ECE over rolling windows (weekly, monthly).
        Reliability diagram — has the shape changed?
        Prediction distribution — has the histogram of predicted probabilities shifted?
        Label rate (if labels available) — does observed frequency match expected?

    WHEN TO TRIGGER RECALIBRATION:
        ECE increases by > 0.03 from baseline.
        Reliability diagram points shift significantly off the diagonal.
        Label rate changes by > 5 percentage points from training baseline.

    RECALIBRATION STRATEGY:
        Keep the original model weights unchanged.
        Only update the calibration layer (Platt/isotonic/temperature).
        Use recent labelled data (last 30–90 days).
        This is much cheaper than full retraining.
        Version-control the calibration parameters (not just the model).


### Best Practices Checklist

    AT MODEL DEVELOPMENT:
        □ Use 3-way split (train / calibration / test) or CV-calibration.
        □ Evaluate both ECE and AUC/log-loss — not just accuracy.
        □ Plot the reliability diagram for all model variants.
        □ Try temperature scaling first (simplest, often sufficient).
        □ If complex miscalibration: try isotonic regression.
        □ Report both calibrated and uncalibrated ECE in model card.
        □ Check calibration on demographic subgroups separately.

    AT DEPLOYMENT:
        □ Deploy the calibration layer as part of the model serving pipeline.
        □ Monitor ECE on labelled production data (if labels available).
        □ Set up alerts for ECE drift > threshold.
        □ Have a recalibration procedure ready without full retraining.
        □ Version-control the calibration parameters.

    IN REGULATED INDUSTRIES:
        □ Calibration must be documented in model risk management docs.
        □ Calibration should be tested on demographic subgroups separately.
        □ Recalibration events must be version-controlled and auditable.

    METHOD SELECTION GUIDE:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Scenario                          │ Recommended Method               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Neural network (any size)         │ Temperature scaling first        │
    │ < 500 calibration samples         │ Platt scaling (sigmoid, 2 params)│
    │ ≥ 1000 calibration samples        │ Isotonic regression              │
    │ Gradient boosting model           │ Platt or isotonic                │
    │ Random forest (under-confident)   │ Platt (finds T < 1)              │
    │ Multi-class neural network        │ Temperature scaling              │
    │ Multi-class non-neural            │ Dirichlet calibration            │
    │ Regression uncertainty            │ Conformal prediction             │
    │ Production with sklearn           │ CalibratedClassifierCV(cv=5)     │
    │ Need guaranteed coverage          │ Conformal prediction             │
    ├──────────────────────────────────────────────────────────────────────┤
    │ What to try FIRST: Temperature Scaling                               │
    │   - 1 parameter: cannot overfit                                      │
    │   - Preserves multi-class structure                                  │
    │   - Preserves ranking (accuracy unchanged)                           │
    │   - Fast: binary search or LBFGS on scalar                           │
    │   - If ECE still > 0.05: try Platt or isotonic                       │
    └──────────────────────────────────────────────────────────────────────┘

    CALIBRATION QUICK REFERENCE:
    ECE < 0.02: excellent    ECE 0.02–0.05: good
    ECE 0.05–0.10: fair      ECE > 0.10: poor

    sklearn temperature (numpy):
      from scipy.optimize import minimize_scalar
      T_opt = minimize_scalar(lambda T: log_loss(y, sigmoid(logits/T)),
                              bounds=(0.1, 10.0), method='bounded').x


##### PART 12 — COMPLETE WORKED EXAMPLE: CALIBRATION BY HAND

### Computing ECE, Reliability Diagram, and Temperature Scaling by Hand

    BINARY CLASSIFIER — 20 predictions on held-out calibration set:
    ─────────────────────────────────────────────────────────────────────
    #   Confidence p̂    True label y   Correct?
    1       0.95              1           ✓
    2       0.92              1           ✓
    3       0.91              1           ✓
    4       0.90              0           ✗  ← wrong despite 90% confidence
    5       0.88              1           ✓
    6       0.85              0           ✗
    7       0.82              1           ✓
    8       0.80              1           ✓
    9       0.78              0           ✗
    10      0.75              1           ✓
    11      0.60              1           ✓
    12      0.58              0           ✗
    13      0.55              1           ✓
    14      0.52              0           ✗
    15      0.51              1           ✓
    16      0.30              0           ✓
    17      0.25              0           ✓
    18      0.22              0           ✓
    19      0.20              1           ✗
    20      0.15              0           ✓
    ─────────────────────────────────────────────────────────────────────

    Overall accuracy: 15/20 = 75%

    BINNING (M=3 equal-width bins):
    Bin 1: [0, 0.4)   → predictions #16–20
      Average confidence: (0.15+0.20+0.22+0.25+0.30)/5 = 0.224
      Accuracy: 4 correct out of 5 = 0.800
      Calibration error: |0.800 − 0.224| = 0.576  ← severely underconfident!

    Bin 2: [0.4, 0.7) → predictions #11–15
      Average confidence: (0.51+0.52+0.55+0.58+0.60)/5 = 0.552
      Accuracy: 3 correct out of 5 = 0.600
      Calibration error: |0.600 − 0.552| = 0.048  ← well calibrated!

    Bin 3: [0.7, 1.0] → predictions #1–10
      Average confidence: 8.56/10 = 0.856
      Accuracy: 7 correct out of 10 = 0.700
      Calibration error: |0.700 − 0.856| = 0.156  ← overconfident!

    ECE COMPUTATION:
    ECE = (5/20)|0.576| + (5/20)|0.048| + (10/20)|0.156|
        = 0.25×0.576 + 0.25×0.048 + 0.50×0.156
        = 0.144 + 0.012 + 0.078 = 0.234  →  ECE ≈ 23.4%

    The reliability diagram shows:
    Bin 1 (conf=0.224, acc=0.800): BAR IS ABOVE DIAGONAL (underconfident)
    Bin 2 (conf=0.552, acc=0.600): bar is NEAR DIAGONAL (well calibrated)
    Bin 3 (conf=0.856, acc=0.700): BAR IS BELOW DIAGONAL (overconfident)

    TEMPERATURE SCALING:
    For binary, logit zᵢ = log(p̂ᵢ/(1−p̂ᵢ)).
    Bin 3 average confidence: 0.856. Logit = log(0.856/0.144) = 1.782.
    At T=1.5: p̂_new = σ(1.782/1.5) = σ(1.188) = 0.766.
    This moves bin 3 average confidence from 0.856 to ~0.766 (closer to 0.700).

    LIMITATION: temperature scaling is a GLOBAL adjustment. In this example,
    bin 1 is underconfident (needs T < 1) while bin 3 is overconfident
    (needs T > 1). A single T cannot fix both simultaneously.
    Isotonic regression can fix heterogeneous miscalibration.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔══════════════════════╦════════════╦════════════╦══════════════╦══════════════╗
    ║ Method               ║ Params     ║ Flexibility║ Min cal. set ║ Accuracy hit ║
    ╠══════════════════════╬════════════╬════════════╬══════════════╬══════════════╣
    ║ Histogram binning    ║ M (bins)   ║ High       ║ Large        ║ None         ║
    ║ Isotonic regression  ║ Non-param  ║ High       ║ 5000+        ║ None         ║
    ║ Beta calibration     ║ 3          ║ Medium     ║ 1000+        ║ None         ║
    ║ Platt scaling        ║ 2          ║ Low-med    ║ 500+         ║ None         ║
    ║ Temperature scaling  ║ 1          ║ Low        ║ 100+         ║ NONE ✓       ║
    ║ Dirichlet (multi-cls)║ K×K        ║ Medium     ║ 1000+        ║ None         ║
    ║ Label smoothing      ║ 1 (ε)      ║ Low        ║ Training     ║ Slight       ║
    ║ MC Dropout           ║ 0 (N runs) ║ Medium     ║ None needed  ║ Slight       ║
    ║ Deep ensembles       ║ M models   ║ High       ║ None needed  ║ None (+ acc) ║
    ╚══════════════════════╩════════════╩════════════╩══════════════╩══════════════╝

    Key metrics:
      ECE = Σᵢ (|Bᵢ|/n) × |acc(Bᵢ) − conf(Bᵢ)|    (lower is better, 0 = perfect)
      MCE = max_i |acc(Bᵢ) − conf(Bᵢ)|              (worst bin; safety-critical)
      OE  = Σᵢ (|Bᵢ|/n) × max(conf(Bᵢ) − acc(Bᵢ), 0)  (overconfidence only)
      NLL = −(1/n) Σᵢ yᵢ log(p̂ᵢ)                    (proper scoring rule)
      Brier = (1/n) Σᵢ (p̂ᵢ − yᵢ)²                   (proper scoring rule)

    Murphy decomposition: Brier = UNC − RES + REL
      UNC = base rate variance (irreducible)
      RES = discrimination (higher is better)
      REL = calibration error (lower is better)

    ECE thresholds:
      < 0.02: excellent    0.02–0.05: good    0.05–0.10: fair    > 0.10: poor
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Measuring Calibration — ECE, Reliability Diagrams, and Brier Score": {
        "description": (
            "Implement calibration measurement from scratch. "
            "Compute ECE, MCE, and ACE from first principles. "
            "Build reliability diagrams with equal-width and equal-mass bins. "
            "Compute the Brier score and its Murphy decomposition (UNC/RES/REL). "
            "Compare calibration of different model families. "
            "Show how AUC and ECE are independent — high AUC does not mean good calibration."
        ),
        "language": "python",
        "code": '''
import numpy as np
import time

print("=" * 65)
print("  MEASURING CALIBRATION — ECE, RELIABILITY DIAGRAMS, BRIER")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: ECE, MCE, ACE from scratch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — ECE, MCE, ACE: implementing calibration metrics")
print("━" * 65)
print()

def compute_ece(y_true, y_prob, n_bins=10, strategy="uniform"):
    """
    Expected Calibration Error.

    strategy="uniform": equal-width bins (standard ECE)
    strategy="quantile": equal-mass bins (ACE - Adaptive CE)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).clip(1e-9, 1 - 1e-9)

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # quantile bins (ACE)
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bins[0]  = 0.0
        bins[-1] = 1.0

    ece      = 0.0
    mce      = 0.0
    n        = len(y_true)
    bin_data = []

    for i in range(len(bins) - 1):
        if i < len(bins) - 2:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i+1])

        if mask.sum() == 0:
            bin_data.append(None)
            continue

        bin_size    = mask.sum()
        bin_conf    = y_prob[mask].mean()
        bin_acc     = y_true[mask].mean()
        bin_gap     = abs(bin_acc - bin_conf)
        weight      = bin_size / n

        ece += weight * bin_gap
        mce  = max(mce, bin_gap)
        bin_data.append({
            "lo":      bins[i],
            "hi":      bins[i+1],
            "n":       bin_size,
            "conf":    bin_conf,
            "acc":     bin_acc,
            "gap":     bin_acc - bin_conf,  # signed gap
            "abs_gap": bin_gap,
            "weight":  weight,
        })

    return ece, mce, [b for b in bin_data if b is not None]


def brier_score(y_true, y_prob):
    """Brier score = mean squared error of probabilities."""
    return np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2)


def brier_decomposition(y_true, y_prob, n_bins=10):
    """
    Murphy decomposition: BS = Uncertainty - Resolution + Reliability
        Uncertainty = var(y) = p̄(1 - p̄)
        Resolution  = (1/n) Σ_m |B_m| × (acc_m - p̄)²
        Reliability = (1/n) Σ_m |B_m| × (conf_m - acc_m)²  (calibration error)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    p_bar  = y_true.mean()
    n      = len(y_true)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)

    uncertainty  = p_bar * (1 - p_bar)
    resolution   = 0.0
    reliability  = 0.0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        if mask.sum() == 0:
            continue
        n_m    = mask.sum()
        acc_m  = y_true[mask].mean()
        conf_m = y_prob[mask].mean()
        resolution  += (n_m / n) * (acc_m - p_bar) ** 2
        reliability += (n_m / n) * (conf_m - acc_m) ** 2

    bs_decomp = uncertainty - resolution + reliability
    bs_direct = brier_score(y_true, y_prob)

    return {
        "uncertainty": uncertainty,
        "resolution":  resolution,
        "reliability": reliability,
        "bs_decomp":   bs_decomp,
        "bs_direct":   bs_direct,
    }


def text_reliability_diagram(bin_data, title="Reliability Diagram"):
    print(f"  {title}")
    print(f"  {'Bin center':>12} | {'Conf':>6} | {'Acc':>6} | "
          f"{'Gap':>7} | {'N':>5} | {'Calibration bar'}")
    print(f"  {'─'*75}")
    for bd in bin_data:
        center = (bd['lo'] + bd['hi']) / 2
        gap    = bd['gap']
        n      = bd['n']
        gap_s  = f"{gap:+.3f}"
        if abs(gap) < 0.02:
            bar_str = "█" * 1 + "  ≈ calibrated"
        elif gap > 0:
            bar = "▲" * min(int(abs(gap) * 40), 20)
            bar_str = bar + "  UNDER-confident"
        else:
            bar = "▼" * min(int(abs(gap) * 40), 20)
            bar_str = bar + "  OVER-confident"
        print(f"  {center:>12.2f} | {bd['conf']:>6.3f} | {bd['acc']:>6.3f} | "
              f"{gap_s:>7} | {n:>5} | {bar_str}")
    print()


np.random.seed(42)
N_SAMPLES = 3000

X_latent = np.random.randn(N_SAMPLES)
y_true   = (X_latent + np.random.randn(N_SAMPLES) * 0.5 > 0).astype(int)
base_rate = y_true.mean()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

p_oracle    = sigmoid(2.5 * X_latent)
p_overconf  = sigmoid(4.0 * X_latent)
p_underconf = sigmoid(0.8 * X_latent)
p_s_shape   = sigmoid(2.5 * X_latent) ** 1.5

noise  = np.random.randn(N_SAMPLES) * 0.3
p_gbm  = sigmoid(2.5 * X_latent + noise).clip(0.05, 0.95)
p_gbm  = np.where(p_gbm > 0.5, p_gbm + 0.08*(p_gbm-0.5), p_gbm - 0.08*(0.5-p_gbm))
p_gbm  = p_gbm.clip(0.01, 0.99)

models = {
    "Oracle (perfect)":        p_oracle,
    "Over-confident (NN)":     p_overconf,
    "Under-confident (RF)":    p_underconf,
    "S-shaped miscalibration": p_s_shape,
    "Gradient Boosting":       p_gbm,
}

print(f"  Dataset: {N_SAMPLES} samples, base rate = {base_rate:.3f}")
print()

print("  Calibration metrics comparison:")
print(f"  {'Model':<30} | {'ECE(10)':>8} | {'ECE(15)':>8} | {'ACE':>8} | "
      f"{'MCE':>8} | {'Brier':>8} | {'Pattern'}")
print(f"  {'─'*90}")

all_metrics = {}
for name, probs in models.items():
    ece10, mce10, bd10 = compute_ece(y_true, probs, n_bins=10)
    ece15, _,    _     = compute_ece(y_true, probs, n_bins=15)
    ace,   _,    _     = compute_ece(y_true, probs, n_bins=10, strategy="quantile")
    bs                  = brier_score(y_true, probs)

    signed_gaps = [bd['gap'] for bd in bd10]
    if all(g > -0.01 for g in signed_gaps): pattern = "UNDER-conf ▲"
    elif all(g < 0.01 for g in signed_gaps): pattern = "OVER-conf  ▼"
    elif signed_gaps[0] < -0.02 and signed_gaps[-1] > 0.02: pattern = "S-shape    ↕"
    else: pattern = "Mixed      ≈"

    all_metrics[name] = {"ece": ece10, "mce": mce10, "ace": ace, "bs": bs, "bd": bd10}
    print(f"  {name:<30} | {ece10:>8.4f} | {ece15:>8.4f} | {ace:>8.4f} | "
          f"{mce10:>8.4f} | {bs:>8.4f} | {pattern}")

print()
print("  ECE interpretation: 0.0=perfect, <0.02=excellent, <0.05=good, >0.05=poor")
print()

print("━" * 65)
print("  SECTION 2 — Reliability diagrams (text-based)")
print("━" * 65)
print()

for name in ["Oracle (perfect)", "Over-confident (NN)", "Under-confident (RF)"]:
    bd = all_metrics[name]["bd"]
    text_reliability_diagram(bd, title=f"  {name}")

print("━" * 65)
print("  SECTION 3 — Brier score Murphy decomposition")
print("━" * 65)
print()

print("  BS = Uncertainty - Resolution + Reliability")
print("  Uncertainty:  irreducible noise (base rate variance)")
print("  Resolution:   discriminative power (higher = more informative)")
print("  Reliability:  calibration error (lower = better calibrated)")
print()
print(f"  {'Model':<30} | {'Uncert':>8} | {'Resolut':>8} | "
      f"{'Reliab':>8} | {'BS (decomp)':>12} | {'BS (direct)':>12}")
print(f"  {'─'*90}")

for name, probs in models.items():
    dec = brier_decomposition(y_true, probs)
    print(f"  {name:<30} | {dec['uncertainty']:>8.4f} | {dec['resolution']:>8.4f} | "
          f"{dec['reliability']:>8.4f} | {dec['bs_decomp']:>12.4f} | "
          f"{dec['bs_direct']:>12.4f}")

print()
print("  All models share the same Uncertainty (same true labels).")
print("  Better model = higher Resolution AND lower Reliability.")
print("  Perfect calibration → Reliability ≈ 0.")
print()

print("━" * 65)
print("  SECTION 4 — AUC vs ECE: high AUC ≠ good calibration")
print("━" * 65)
print()

def _roc_auc(y_true, y_score):
    yt = np.asarray(y_true); ys = np.asarray(y_score)
    desc = np.argsort(ys)[::-1]; yt = yt[desc]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt)
    tp_r = tp / tp[-1]; fp_r = fp / fp[-1]
    return float(np.trapezoid(tp_r, fp_r))

print(f"  {'Model':<30} | {'AUC':>8} | {'ECE':>8} | {'Calibrated?':>13} | {'Discriminates?'}")
print(f"  {'─'*75}")
for name, probs in models.items():
    auc  = _roc_auc(y_true, probs)
    ece  = all_metrics[name]["ece"]
    cal  = "✅ Good" if ece < 0.03 else ("⚠️  Fair" if ece < 0.06 else "❌ Poor")
    disc = "✅ Good" if auc > 0.80 else ("⚠️  Fair" if auc > 0.70 else "❌ Poor")
    print(f"  {name:<30} | {auc:>8.4f} | {ece:>8.4f} | {cal:>13} | {disc}")
print()
print("  Key insight: Over-confident NN can have HIGH AUC but HIGH ECE.")
print("  The model discriminates well but assigns wrong probabilities.")
print("  For decisions based on probability thresholds, calibration matters!")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Calibration Methods — Platt Scaling, Isotonic, and Temperature Scaling": {
        "description": (
            "Implement and compare the three main calibration methods. "
            "Platt scaling: fit logistic regression on held-out calibration scores. "
            "Isotonic regression: non-parametric pool-adjacent-violators algorithm. "
            "Temperature scaling: single-parameter neural network calibration. "
            "Show before/after reliability diagrams and ECE for each method. "
            "sklearn CalibratedClassifierCV for production use. "
            "Cross-validated calibration to avoid data leakage."
        ),
        "timeout": 300,
        "language": "python",
        "code": '''
import numpy as np
from scipy.optimize import minimize_scalar

print("=" * 65)
print("  CALIBRATION METHODS — PLATT, ISOTONIC, TEMPERATURE SCALING")
print("=" * 65)
print()

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def compute_ece(y_true, y_prob, n_bins=10):
    y_prob = np.clip(y_prob, 1e-9, 1-1e-9)
    bins   = np.linspace(0, 1, n_bins+1)
    ece    = 0.0
    n      = len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if i == n_bins-1: m = (y_prob >= bins[i]) & (y_prob <= 1.0)
        if m.sum() == 0: continue
        ece += (m.sum()/n) * abs(y_true[m].mean() - y_prob[m].mean())
    return ece

def log_loss(y, p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

import numpy as _np2
import scipy.optimize as _sp2
import copy as _copy2

def _sig2(z): return 1.0/(1.0+_np2.exp(-_np2.clip(z,-500,500)))

def _make_cls(n_samples=100,n_features=20,n_informative=10,n_redundant=5,random_state=None,**kw):
    rng=_np2.random.default_rng(random_state); y=rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=n_redundant
    Xr=(Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else _np2.empty((n_samples,0)))
    nn2=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn2)) if nn2>0 else _np2.empty((n_samples,0))
    parts=[a for a in [Xi,Xr,Xn] if a.shape[1]>0]
    return _np2.hstack(parts)[:,:n_features],y.astype(int)

def _tts(*arrays,test_size=0.4,random_state=None,**kw):
    rng=_np2.random.default_rng(random_state); n=len(arrays[0])
    n_tr=int(n*(1-test_size)); idx=rng.permutation(n); ti,vi=idx[:n_tr],idx[n_tr:]
    out=[]
    for a in arrays: out+=[a[ti],a[vi]]
    return out

class _DTR2:
    def __init__(self,max_depth=3,min_s=5): self.max_depth=max_depth; self.min_s=min_s
    def _split(self,X,r):
        best=None
        for f in range(X.shape[1]):
            vals=_np2.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>10: ts=ts[_np2.linspace(0,len(ts)-1,10).astype(int)]
            for t in ts:
                l=X[:,f]<=t; rr=~l
                if l.sum()<self.min_s or rr.sum()<self.min_s: continue
                g=_np2.var(r)*len(r)-_np2.var(r[l])*l.sum()-_np2.var(r[rr])*rr.sum()
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np2.var(r)<1e-12: return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np2.array([self._p1(x,self._tree) for x in X])

class GradientBoostingClassifier:
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=min(n_estimators,40); self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np2.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np2.log(p0/(1-p0)); F=_np2.full(len(y),self._f0); self._trees=[]
        self._classes=_np2.unique(y)
        for _ in range(self.n_estimators):
            r=yf-_sig2(F); t=_DTR2(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict_proba(self,X):
        F=_np2.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        p=_sig2(F); return _np2.c_[1-p,p]
    def predict(self,X): return self._classes[(self.predict_proba(X)[:,1]>=0.5).astype(int)]
    def score(self,X,y): return _np2.mean(self.predict(X)==y)

class CalibratedClassifierCV:
    def __init__(self,estimator,method='sigmoid',cv=3,**kw):
        self.estimator=estimator; self.method=method; self.cv=cv
    def fit(self,X,y):
        n=len(X); idx=_np2.arange(n); fs=n//self.cv
        oof=_np2.zeros(n); oofy=_np2.zeros(n)
        for k in range(self.cv):
            vi=idx[k*fs:(k+1)*fs]; ti=_np2.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
            est=_copy2.deepcopy(self.estimator); est.fit(X[ti],y[ti])
            oof[vi]=est.predict_proba(X[vi])[:,1]; oofy[vi]=y[vi]
        if self.method=='isotonic':
            order=_np2.argsort(oof); ss=oof[order]; ys=oofy[order].astype(float)
            blocks=[[i] for i in range(len(ys))]; means=list(ys.copy()); i=0
            while i<len(means)-1:
                if means[i]>means[i+1]:
                    mg=blocks[i]+blocks[i+1]; mm=ys[mg].mean()
                    blocks[i]=mg; means[i]=mm; blocks.pop(i+1); means.pop(i+1); i=max(0,i-1)
                else: i+=1
            cal=_np2.zeros_like(ys)
            for block,mv in zip(blocks,means): cal[block]=mv
            self._iso_x=ss; self._iso_y=cal; self._method='iso'
        else:
            from scipy.optimize import minimize as _m2
            def nll(params):
                A,B=params; p=_sig2(A*oof+B)
                return -_np2.mean(oofy*_np2.log(p+1e-12)+(1-oofy)*_np2.log(1-p+1e-12))
            res=_m2(nll,[1.0,0.0],method='L-BFGS-B')
            self._A,self._B=res.x; self._method='platt'
        self._final=_copy2.deepcopy(self.estimator); self._final.fit(X,y)
        return self
    def predict_proba(self,X):
        p=self._final.predict_proba(X)[:,1]
        if self._method=='iso':
            pc=_np2.interp(p,self._iso_x,self._iso_y,left=self._iso_y[0],right=self._iso_y[-1])
        else:
            pc=_sig2(self._A*p+self._B)
        return _np2.c_[1-pc,pc]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

X, y = _make_cls(n_samples=5000, n_features=20, n_informative=10,
                  n_redundant=5, random_state=42)
X_tr, X_tmp, y_tr, y_tmp = _tts(X, y, test_size=0.4, random_state=42)
X_cal, X_te, y_cal, y_te = _tts(X_tmp, y_tmp, test_size=0.5, random_state=42)

gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                  learning_rate=0.05, random_state=42)
gbm.fit(X_tr, y_tr)
s_cal = gbm.predict_proba(X_cal)[:, 1]
s_te  = gbm.predict_proba(X_te)[:, 1]
print(f"  GBM trained: {len(X_tr)} train, {len(X_cal)} calibration, {len(X_te)} test")
print(f"  Uncalibrated ECE (test): {compute_ece(y_te, s_te):.4f}")
print()

print("━" * 65)
print("  SECTION 1 — Platt Scaling: logistic regression on scores")
print("━" * 65)
print()

print("  Platt scaling: p_calibrated = σ(A × s + B)")
print("  Fit A and B on calibration set by minimising log-loss.")
print()

class PlattScaler:
    def __init__(self): self.A = 1.0; self.B = 0.0
    def fit(self, s_cal, y_cal):
        from scipy.optimize import minimize
        def neg_log_loss(params):
            A, B = params
            p = sigmoid(A * s_cal + B)
            return log_loss(y_cal, p)
        result = minimize(neg_log_loss, [1.0, 0.0], method='L-BFGS-B')
        self.A, self.B = result.x
        return self
    def predict_proba(self, s):
        return sigmoid(self.A * np.asarray(s) + self.B)

platt = PlattScaler().fit(s_cal, y_cal)
p_platt = platt.predict_proba(s_te)

print(f"  Fitted parameters:")
print(f"    A = {platt.A:.4f}  ({'< 1 → model was over-confident' if platt.A < 1 else '> 1 → under-confident'})")
print(f"    B = {platt.B:.4f}  ({'< 0 → shift down' if platt.B < 0 else '>= 0 → shift up'})")
print()
print(f"  Before calibration — ECE: {compute_ece(y_te, s_te):.4f}  Log-loss: {log_loss(y_te, s_te):.4f}")
print(f"  After Platt scaling   — ECE: {compute_ece(y_te, p_platt):.4f}  Log-loss: {log_loss(y_te, p_platt):.4f}")
print()

print("  Score → Calibrated probability mapping (Platt):")
for s_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    p_cal = platt.predict_proba(s_val)
    print(f"    raw={s_val:.2f}  →  calibrated={p_cal:.4f}  (Δ = {p_cal-s_val:+.4f})")
print()

print("━" * 65)
print("  SECTION 2 — Isotonic Regression: non-parametric calibration")
print("━" * 65)
print()

print("  Isotonic regression: monotone step function, pool-adjacent-violators.")
print("  No shape assumption — can fit ANY monotone calibration curve.")
print()

class IsotonicCalibrator:
    def __init__(self): self.breakpoints = None; self.values = None
    def _pav(self, x, y):
        n = len(y)
        block_sum   = y.copy().astype(float)
        block_count = np.ones(n)
        block_mean  = block_sum.copy()
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(block_mean) - 1:
                if block_mean[i] > block_mean[i+1]:
                    new_sum   = block_sum[i]   + block_sum[i+1]
                    new_count = block_count[i] + block_count[i+1]
                    new_mean  = new_sum / new_count
                    block_sum   = np.delete(block_sum,   i+1)
                    block_count = np.delete(block_count, i+1)
                    block_mean  = np.delete(block_mean,  i+1)
                    block_sum[i]   = new_sum
                    block_count[i] = new_count
                    block_mean[i]  = new_mean
                    changed = True
                else:
                    i += 1
        result = np.zeros(n)
        idx = 0
        for mean, count in zip(block_mean, block_count):
            result[idx:idx+int(count)] = mean
            idx += int(count)
        return result
    def fit(self, s_cal, y_cal):
        order    = np.argsort(s_cal)
        s_sorted = s_cal[order]
        y_sorted = y_cal[order].astype(float)
        y_iso  = self._pav(s_sorted, y_sorted)
        unique_s, unique_idx = np.unique(s_sorted, return_index=True)
        self.breakpoints = s_sorted[unique_idx]
        self.values      = y_iso[unique_idx]
        return self
    def predict_proba(self, s):
        return np.clip(np.interp(s, self.breakpoints, self.values), 0, 1)

iso = IsotonicCalibrator().fit(s_cal, y_cal)
p_iso = iso.predict_proba(s_te)

print(f"  After isotonic regression — ECE: {compute_ece(y_te, p_iso):.4f}  Log-loss: {log_loss(y_te, p_iso):.4f}")
print()
print("  Isotonic mapping (selected points):")
for s_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    p_cal = iso.predict_proba(np.array([s_val]))[0]
    print(f"    raw={s_val:.2f}  →  calibrated={p_cal:.4f}")
print()

print("━" * 65)
print("  SECTION 3 — Temperature Scaling: single-parameter calibration")
print("━" * 65)
print()

print("  Temperature scaling: p_cal = σ(logit(p_raw) / T)")
print("  T > 1: softer (for over-confident models)")
print("  T < 1: sharper (for under-confident models)")
print()

class TemperatureScaler:
    def __init__(self): self.T = 1.0
    def fit(self, s_cal, y_cal):
        s_logit = np.log(np.clip(s_cal, 1e-9, 1-1e-9) / (1 - np.clip(s_cal, 1e-9, 1-1e-9)))
        def nll(T):
            if T <= 0: return 1e9
            p = sigmoid(s_logit / T)
            return log_loss(y_cal, p)
        result = minimize_scalar(nll, bounds=(0.01, 10.0), method='bounded')
        self.T = result.x
        return self
    def predict_proba(self, s):
        s_logit = np.log(np.clip(s, 1e-9, 1-1e-9) / (1 - np.clip(s, 1e-9, 1-1e-9)))
        return sigmoid(s_logit / self.T)

temp = TemperatureScaler().fit(s_cal, y_cal)
p_temp = temp.predict_proba(s_te)

print(f"  Optimal temperature T = {temp.T:.4f}  "
      f"({'> 1: model was over-confident' if temp.T > 1 else '< 1: under-confident'})")
print(f"  After temperature scaling — ECE: {compute_ece(y_te, p_temp):.4f}  Log-loss: {log_loss(y_te, p_temp):.4f}")
print()

print("━" * 65)
print("  SECTION 4 — Comprehensive method comparison")
print("━" * 65)
print()

results = {
    "Uncalibrated":         s_te,
    "Platt Scaling":        p_platt,
    "Isotonic Regression":  p_iso,
    "Temperature Scaling":  p_temp,
}

print(f"  {'Method':<25} | {'ECE':>8} | {'Log-loss':>10} | {'Brier':>8} | {'vs uncal ECE'}")
print(f"  {'─'*72}")

ece_base = compute_ece(y_te, s_te)
for name, probs in results.items():
    ece  = compute_ece(y_te, probs)
    ll   = log_loss(y_te, probs)
    bs   = np.mean((probs - y_te)**2)
    impr = (ece_base - ece) / ece_base * 100 if name != "Uncalibrated" else 0
    flag = f"↓ {impr:.0f}% better" if impr > 0 else ("→ baseline" if impr == 0 else "↑ worse")
    print(f"  {name:<25} | {ece:>8.4f} | {ll:>10.4f} | {bs:>8.4f} | {flag}")

print()
print("  sklearn CalibratedClassifierCV (cross-validated calibration):")
print()

gbm_raw = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
gbm_raw.fit(X_tr, y_tr)
p_raw = gbm_raw.predict_proba(X_te)[:, 1]

gbm_platt = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    method='sigmoid', cv=3)
gbm_platt.fit(X_tr, y_tr)
p_platt_cv = gbm_platt.predict_proba(X_te)[:, 1]

gbm_iso = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    method='isotonic', cv=3)
gbm_iso.fit(X_tr, y_tr)
p_iso_cv = gbm_iso.predict_proba(X_te)[:, 1]

print(f"  {'Method':<35} | {'ECE':>8} | {'Log-loss':>10}")
print(f"  {'─'*58}")
for nm, pp in [("GBM (raw)", p_raw), ("GBM + Platt (cv=3)", p_platt_cv),
               ("GBM + Isotonic (cv=3)", p_iso_cv)]:
    ece = compute_ece(y_te, pp)
    ll  = log_loss(y_te, pp)
    print(f"  {nm:<35} | {ece:>8.4f} | {ll:>10.4f}")
print()
print("  sklearn API reference:")
print("    from sklearn.calibration import CalibratedClassifierCV")
print("    cal_model = CalibratedClassifierCV(base_model,")
print("                    method='sigmoid',  # or 'isotonic'")
print("                    cv=5)              # or 'prefit'")
print("    cal_model.fit(X_train, y_train)")
print("    p_cal = cal_model.predict_proba(X_test)[:, 1]")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Temperature Scaling for Neural Networks and Production Workflow": {
        "description": (
            "Temperature scaling on neural network logits. "
            "Complete PyTorch implementation (with pure-numpy fallback). "
            "Calibration under distribution shift: covariate shift, prior shift, concept drift. "
            "Production monitoring: detect calibration drift over 12 monthly batches. "
            "Method selection guide and ECE alert thresholds."
        ),
        "language": "python",
        "code": r"""
import numpy as np
from scipy.optimize import minimize_scalar

print("=" * 65)
print("  TEMPERATURE SCALING AND PRODUCTION CALIBRATION WORKFLOW")
print("=" * 65)
print()

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def log_loss(y, p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

def compute_ece(y_true, y_prob, n_bins=10):
    y_prob = np.clip(y_prob, 1e-9, 1-1e-9)
    bins = np.linspace(0, 1, n_bins+1)
    ece  = 0.0; n = len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if i == n_bins-1: m = (y_prob >= bins[i]) & (y_prob <= 1.0)
        if m.sum() == 0: continue
        ece += (m.sum()/n) * abs(y_true[m].mean() - y_prob[m].mean())
    return ece

print("━" * 65)
print("  SECTION 1 — Temperature scaling: PyTorch implementation")
print("━" * 65)
print()

PYTORCH_TEMP = '''
  COMPLETE PYTORCH TEMPERATURE SCALING:

  import torch, torch.nn as nn
  from torch import optim

  class TemperatureScaler(nn.Module):
      def __init__(self, model):
          super().__init__()
          self.model       = model
          self.temperature = nn.Parameter(torch.ones(1) * 1.5)

      def forward(self, input):
          logits = self.model(input)
          return self.temperature_scale(logits)

      def temperature_scale(self, logits):
          T = self.temperature.unsqueeze(1).expand_as(logits)
          return logits / T

      def set_temperature(self, valid_loader):
          self.model.eval()
          nll_criterion = nn.CrossEntropyLoss()
          logits_list, labels_list = [], []
          with torch.no_grad():
              for X, y in valid_loader:
                  logits_list.append(self.model(X))
                  labels_list.append(y)
          logits = torch.cat(logits_list)
          labels = torch.cat(labels_list)

          optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
          def eval_fn():
              optimizer.zero_grad()
              loss = nll_criterion(self.temperature_scale(logits), labels)
              loss.backward()
              return loss
          optimizer.step(eval_fn)
          print(f'Optimal T = {self.temperature.item():.4f}')
          return self

  # Usage:
  scaled_model = TemperatureScaler(pretrained_model)
  scaled_model.set_temperature(val_loader)  # finds T on val set
  # model weights NEVER modified — only T is trained
  # accuracy ALWAYS preserved (argmax unchanged)
'''
print(PYTORCH_TEMP)

try:
    import torch
    import torch.nn as nn

    N_CAL, N_TEST = 1000, 2000
    X_c = np.random.randn(N_CAL); X_t = np.random.randn(N_TEST)
    y_cal = (X_c > 0).astype(int); y_te = (X_t > 0).astype(int)
    logits_cal = torch.FloatTensor(4.0*X_c + np.random.randn(N_CAL)*0.5)
    logits_te  = torch.FloatTensor(4.0*X_t + np.random.randn(N_TEST)*0.5)
    p_te_before = torch.sigmoid(logits_te).numpy()

    print(f"  Before temperature scaling:")
    print(f"    ECE: {compute_ece(y_te, p_te_before):.4f}  Log-loss: {log_loss(y_te, p_te_before):.4f}")

    T = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    criterion = nn.BCEWithLogitsLoss()
    y_cal_t   = torch.FloatTensor(y_cal)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_cal / T, y_cal_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    T_opt = T.item()
    p_te_after = torch.sigmoid(logits_te / T_opt).detach().numpy()

    print(f"  Optimal T = {T_opt:.4f}")
    print(f"  After temperature scaling:")
    print(f"    ECE: {compute_ece(y_te, p_te_after):.4f}  Log-loss: {log_loss(y_te, p_te_after):.4f}")
    print()
    print("  Effect on extreme predictions:")
    print(f"  {'Raw prob':>10} | {'Calibrated':>12} | {'Change':>10}")
    print(f"  {'─'*38}")
    for raw in [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]:
        logit = np.log(raw / (1 - raw))
        cal   = sigmoid(logit / T_opt)
        print(f"  {raw:>10.2f} | {cal:>12.4f} | {cal-raw:>+10.4f}")

except ImportError:
    N_CAL, N_TEST = 1000, 2000
    X_c = np.random.randn(N_CAL); X_t = np.random.randn(N_TEST)
    y_cal_np = (X_c > 0).astype(int); y_te_np = (X_t > 0).astype(int)
    logits_c = 4.0*X_c + np.random.randn(N_CAL)*0.5
    logits_t = 4.0*X_t + np.random.randn(N_TEST)*0.5

    def nll_T(T):
        return log_loss(y_cal_np, sigmoid(logits_c / max(T, 0.01)))

    result = minimize_scalar(nll_T, bounds=(0.1, 10.0), method='bounded')
    T_opt  = result.x
    p_before = sigmoid(logits_t); p_after = sigmoid(logits_t / T_opt)
    print(f"  Optimal T = {T_opt:.4f}")
    print(f"  Before: ECE={compute_ece(y_te_np, p_before):.4f}  Log-loss={log_loss(y_te_np, p_before):.4f}")
    print(f"  After:  ECE={compute_ece(y_te_np, p_after):.4f}  Log-loss={log_loss(y_te_np, p_after):.4f}")

print()
print("━" * 65)
print("  SECTION 2 — Calibration under distribution shift")
print("━" * 65)
print()

N = 1000
X_base = np.random.randn(N)
y_base = (X_base + np.random.randn(N)*0.5 > 0).astype(int)
logits_base = 4.0 * X_base

def find_temperature(logits, y):
    def nll(T): return log_loss(y, sigmoid(logits / max(T, 0.01)))
    return minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded').x

T_fitted = find_temperature(logits_base, y_base)

shifts = [
    ("In-distribution (no shift)", 0.0, 0.0),
    ("Mild covariate shift",       0.5, 0.1),
    ("Strong covariate shift",     1.5, 0.3),
    ("Prior shift (class imbal.)", 0.0, 1.0),
    ("Severe concept drift",       0.0, 2.0),
]

print(f"  Calibration fitted with T = {T_fitted:.4f}")
print()
print(f"  {'Test scenario':<35} | {'ECE before':>12} | {'ECE after':>11} | {'Still good?'}")
print(f"  {'─'*75}")

for scenario, covar_shift, noise_amp in shifts:
    X_s    = np.random.randn(N) + covar_shift
    noise_extra = np.random.randn(N) * noise_amp
    y_s    = (X_s + np.random.randn(N)*0.5 + noise_extra > 0).astype(int)
    logits_s = 4.0 * X_s
    p_before = sigmoid(logits_s)
    p_after  = sigmoid(logits_s / T_fitted)
    ece_before = compute_ece(y_s, p_before)
    ece_after  = compute_ece(y_s, p_after)
    good = "✅ Good" if ece_after < 0.05 else ("⚠️  Fair" if ece_after < 0.10 else "❌ Poor")
    print(f"  {scenario:<35} | {ece_before:>12.4f} | {ece_after:>11.4f} | {good}")

print()
print("  Key finding: calibration transfers well for covariate shift,")
print("  but degrades under prior shift or concept drift.")
print("  → Recalibrate periodically with recent labelled data.")
print()

print("━" * 65)
print("  SECTION 3 — Production monitoring: detecting calibration drift")
print("━" * 65)
print()

n_periods = 12; period_size = 300; ece_history = []
X_deploy = np.random.randn(500)
y_deploy = (X_deploy + np.random.randn(500)*0.5 > 0).astype(int)
T_deployed = find_temperature(4.0 * X_deploy, y_deploy)

print(f"  Deployed with T = {T_deployed:.4f}")
print()
print(f"  {'Month':>6} | {'ECE':>8} | {'Base rate':>10} | {'Status'}")
print(f"  {'─'*42}")

alert_threshold = 0.06
for period in range(n_periods):
    drift_rate = period / n_periods
    base_rate  = 0.50 - 0.15 * drift_rate
    X_p = np.random.randn(period_size)
    threshold = np.percentile(X_p + np.random.randn(period_size)*0.5, (1-base_rate)*100)
    y_p = ((X_p + np.random.randn(period_size)*0.5) > threshold).astype(int)
    logits_p = 4.0 * X_p
    p_after_p = sigmoid(logits_p / T_deployed)
    ece_p = compute_ece(y_p, p_after_p)
    ece_history.append(ece_p)
    status = "✅ OK" if ece_p < alert_threshold else "🚨 ALERT — recalibrate!"
    print(f"  {period+1:>6} | {ece_p:>8.4f} | {y_p.mean():>10.3f} | {status}")

print()
print(f"  Alert threshold: ECE > {alert_threshold}")
print(f"  Alerts triggered at months: {[i+1 for i, e in enumerate(ece_history) if e > alert_threshold]}")
print()
print("  Monitoring strategy:")
print("    1. Compute ECE on every scored batch (rolling window)")
print("    2. Alert if ECE > threshold vs deployment baseline")
print("    3. Keep a 'recalibration dataset' of recent labelled examples")
print("    4. When alert fires: refit T (or Platt) on recent data")
print("    5. Version-control the calibration parameters (not just the model)")
print()

print("━" * 65)
print("  SECTION 4 — Calibration method selection guide")
print("━" * 65)
print()

GUIDE = '''
  CALIBRATION METHOD SELECTION:

  ┌──────────────────────────────────────────────────────────────────────┐
  │ Scenario                          │ Recommended Method               │
  ├──────────────────────────────────────────────────────────────────────┤
  │ Neural network (any size)         │ Temperature scaling first        │
  │ < 500 calibration samples         │ Platt scaling (sigmoid, 2 params)│
  │ ≥ 1000 calibration samples        │ Isotonic regression              │
  │ Gradient boosting model           │ Platt or isotonic                │
  │ Random forest (under-confident)   │ Platt (finds T < 1)              │
  │ Multi-class neural network        │ Temperature scaling              │
  │ Multi-class non-neural            │ Dirichlet calibration            │
  │ Regression uncertainty            │ Conformal prediction             │
  │ Production with sklearn           │ CalibratedClassifierCV(cv=5)     │
  │ Need guaranteed coverage          │ Conformal prediction             │
  ├──────────────────────────────────────────────────────────────────────┤
  │ What to try FIRST: Temperature Scaling                               │
  │   - 1 parameter: cannot overfit                                      │
  │   - Preserves multi-class structure and accuracy                     │
  │   - Fast: binary search or LBFGS on scalar                           │
  │   - If ECE still > 0.05: try Platt or isotonic                       │
  ├──────────────────────────────────────────────────────────────────────┤
  │ ALWAYS:                                                              │
  │   □ Calibrate on HELD-OUT data (not training data)                   │
  │   □ Use cross-validation if calibration set is small                 │
  │   □ Report ECE + AUC together (not just accuracy)                    │
  │   □ Plot reliability diagram before AND after calibration            │
  │   □ Check calibration on demographic subgroups separately            │
  │   □ Monitor ECE in production with labelled data                     │
  └──────────────────────────────────────────────────────────────────────┘
'''
print(GUIDE)
print("  ECE thresholds: < 0.02 excellent  |  0.02–0.05 good  |  0.05–0.10 fair  |  > 0.10 poor")
""",
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Calibration from Scratch — All Methods, Murphy Decomposition, and Conformal Prediction": {
        "description": (
            "Complete calibration pipeline in pure Python (zero dependencies). "
            "Synthesises a deliberately miscalibrated binary classifier output. "
            "Computes ECE, MCE, and Brier score with full Murphy decomposition (UNC/RES/REL). "
            "Renders an ASCII reliability diagram showing over/underconfidence per bin. "
            "Applies all four recalibration methods from scratch: "
            "temperature scaling (bisection), Platt scaling (gradient descent), "
            "isotonic regression (PAV algorithm), and histogram binning. "
            "Compares ECE and NLL before and after each method on a held-out test set. "
            "Murphy decomposition before and after temperature scaling. "
            "Conformal prediction: non-asymptotic coverage verification at multiple alpha levels."
        ),
        "runnable": True,
        "pipeline_cmd": "calibration",
        "code": '''
"""
================================================================================
CALIBRATION FROM SCRATCH — ALL METHODS, MURPHY DECOMPOSITION, CONFORMAL PREDICTION
================================================================================
Pure Python only (no numpy/scipy required).
  1. Compute ECE, MCE, Brier score with Murphy decomposition
  2. Draw an ASCII reliability diagram
  3. Apply four recalibration methods:
     (a) Temperature scaling — 1-param bisection
     (b) Platt scaling       — 2-param gradient descent
     (c) Isotonic regression — Pool Adjacent Violators algorithm
     (d) Histogram binning   — non-parametric bin replacement
  4. Compare all methods on a held-out test set
  5. Murphy decomposition before and after temperature scaling
  6. Conformal prediction — non-asymptotic coverage verification
================================================================================
"""

import math
import random

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC MISCALIBRATED CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

def generate_data(n, seed=0):
    """
    Generate (confidence, label) pairs from a systematically miscalibrated classifier.

    True model: P(y=1|x) = sigmoid(2x) for x ~ N(0,1)
    Overconfident output: the classifier inflates confidence by 2.5× logit magnitude.
    """
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.gauss(0, 1)
        true_prob = sigmoid(2 * x)
        y = 1 if random.random() < true_prob else 0
        p_hat = sigmoid(2.5 * 2 * x)      # overconfident prediction
        data.append((p_hat, y))
    return data

ALL_DATA = generate_data(2000, seed=42)
CAL_DATA  = ALL_DATA[:1000]
TEST_DATA = ALL_DATA[1000:]


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece_mce(data, n_bins=10, equal_mass=False):
    """Compute ECE, MCE, and bin statistics. Returns (ECE, MCE, bins)."""
    n = len(data)
    if equal_mass:
        sorted_data = sorted(data, key=lambda d: d[0])
        bin_size = n // n_bins
        bins_data = [sorted_data[i*bin_size:(i+1)*bin_size] for i in range(n_bins)]
    else:
        bins_data = [[] for _ in range(n_bins)]
        for p_hat, y in data:
            b = min(int(p_hat * n_bins), n_bins-1)
            bins_data[b].append((p_hat, y))

    bin_stats = []
    for bdata in bins_data:
        if not bdata:
            continue
        avg_conf = sum(p for p, y in bdata) / len(bdata)
        avg_acc  = sum(y for p, y in bdata) / len(bdata)
        count    = len(bdata)
        bin_stats.append((avg_conf, avg_acc, count))

    ece = sum((cnt/n) * abs(acc - conf) for conf, acc, cnt in bin_stats)
    mce = max(abs(acc - conf) for conf, acc, cnt in bin_stats)
    return ece, mce, bin_stats

def compute_brier(data):
    """Compute Brier score and Murphy decomposition (UNC, RES, REL)."""
    n = len(data)
    p_bar = sum(y for _, y in data) / n
    bs = sum((p - y)**2 for p, y in data) / n
    _, _, bins = compute_ece_mce(data, n_bins=10)
    unc = p_bar * (1 - p_bar)
    res = sum((cnt/n) * (acc - p_bar)**2 for conf, acc, cnt in bins)
    rel = sum((cnt/n) * (acc - conf)**2  for conf, acc, cnt in bins)
    return bs, unc, res, rel

def compute_nll(data):
    """Negative log-likelihood."""
    eps = 1e-12
    return -sum(y*math.log(p+eps) + (1-y)*math.log(1-p+eps) for p, y in data) / len(data)


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("  INITIAL CALIBRATION DIAGNOSTICS (before recalibration)")
print("=" * 68)
print()

ece_cal, mce_cal, bins_cal = compute_ece_mce(CAL_DATA)
bs_cal, unc, res, rel = compute_brier(CAL_DATA)
nll_cal = compute_nll(CAL_DATA)
acc_cal  = sum(y for _, y in CAL_DATA) / len(CAL_DATA)
avg_conf = sum(p for p, _ in CAL_DATA) / len(CAL_DATA)

print(f"  Calibration set (n={len(CAL_DATA)}):")
print(f"    Average confidence:   {avg_conf:.4f}")
print(f"    Average accuracy:     {acc_cal:.4f}")
print(f"    Overconfidence gap:   {avg_conf - acc_cal:+.4f}")
print()
print(f"    ECE (10 bins):        {ece_cal:.4f}  (0 = perfect)")
print(f"    MCE:                  {mce_cal:.4f}  (worst-bin error)")
print(f"    NLL:                  {nll_cal:.4f}  (lower is better)")
print(f"    Brier score:          {bs_cal:.4f}")
print(f"      ├── UNC:            {unc:.4f}  (irreducible dataset uncertainty)")
print(f"      ├── RES:            {res:.4f}  (discrimination, higher=better)")
print(f"      └── REL:            {rel:.4f}  (reliability/calibration, lower=better)")


# ─────────────────────────────────────────────────────────────────────────────
# ASCII RELIABILITY DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  ASCII RELIABILITY DIAGRAM (calibration set)")
print("=" * 68)
print()
print("  Each row = one confidence bin (10 bins, equal-width)")
print("  C = average confidence position   A = accuracy position")
print("  - = overconfident (C right of A)  + = underconfident (A right of C)")
print()

for conf, acc, cnt in bins_cal:
    conf_pos  = int(conf * 40)
    acc_pos   = int(acc  * 40)
    direction = "← OVER" if conf > acc + 0.02 else ("UNDER →" if acc > conf + 0.02 else "  OK  ")
    bar = ["·"] * 41
    bar[min(conf_pos, 40)] = "C"
    bar[min(acc_pos,  40)] = "A"
    if acc_pos < conf_pos:
        for k in range(acc_pos+1, conf_pos):
            if bar[k] == "·": bar[k] = "-"
    elif conf_pos < acc_pos:
        for k in range(conf_pos+1, acc_pos):
            if bar[k] == "·": bar[k] = "+"
    bar_str = "".join(bar)
    print(f"  [{conf:.2f}] {bar_str}  acc={acc:.2f} {direction} (n={cnt})")

print()


# ─────────────────────────────────────────────────────────────────────────────
# RECALIBRATION METHOD 1 — TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

def p_hat_with_temp(p_hat, T):
    """Apply temperature scaling to a probability (binary case)."""
    if p_hat <= 0 or p_hat >= 1: return p_hat
    logit = math.log(p_hat / (1 - p_hat))
    return sigmoid(logit / T)

def nll_for_temp(T, data):
    """NLL on data after applying temperature T."""
    eps = 1e-12
    total = 0.0
    for p, y in data:
        p_cal = p_hat_with_temp(p, T)
        total -= y * math.log(p_cal + eps) + (1 - y) * math.log(1 - p_cal + eps)
    return total / len(data)

# Find optimal T via bisection (NLL is unimodal in T)
T_lo, T_hi = 0.1, 10.0
for _ in range(60):
    T_mid = (T_lo + T_hi) / 2
    eps_T = 1e-4
    if nll_for_temp(T_mid + eps_T, CAL_DATA) < nll_for_temp(T_mid - eps_T, CAL_DATA):
        T_lo = T_mid
    else:
        T_hi = T_mid
T_opt = (T_lo + T_hi) / 2

data_temp_test = [(p_hat_with_temp(p, T_opt), y) for p, y in TEST_DATA]
ece_temp, _, _ = compute_ece_mce(data_temp_test)
nll_temp = compute_nll(data_temp_test)


# ─────────────────────────────────────────────────────────────────────────────
# RECALIBRATION METHOD 2 — PLATT SCALING
# ─────────────────────────────────────────────────────────────────────────────

# Gradient descent to find (a, b): p_cal = sigmoid(a*logit + b)
a_platt, b_platt = 1.0, 0.0
lr_p = 0.05
for step in range(1000):
    da, db = 0.0, 0.0
    eps = 1e-12
    for p, y in CAL_DATA:
        if p <= 0 or p >= 1: continue
        logit = math.log(p / (1 - p))
        p_cal = sigmoid(a_platt * logit + b_platt)
        grad_base = p_cal - y
        da += grad_base * logit
        db += grad_base
    n = len(CAL_DATA)
    a_platt -= lr_p * da / n
    b_platt -= lr_p * db / n

def platt_transform(p, a, b):
    if p <= 0 or p >= 1: return p
    logit = math.log(p / (1 - p))
    return sigmoid(a * logit + b)

data_platt_test = [(platt_transform(p, a_platt, b_platt), y) for p, y in TEST_DATA]
ece_platt, _, _ = compute_ece_mce(data_platt_test)
nll_platt = compute_nll(data_platt_test)


# ─────────────────────────────────────────────────────────────────────────────
# RECALIBRATION METHOD 3 — ISOTONIC REGRESSION (PAV ALGORITHM)
# ─────────────────────────────────────────────────────────────────────────────

def isotonic_regression_pav(data):
    """Pool Adjacent Violators algorithm for isotonic regression."""
    sorted_data = sorted(data, key=lambda d: d[0])
    groups = [([sorted_data[i][0]], [sorted_data[i][1]]) for i in range(len(sorted_data))]

    merged = True
    while merged:
        merged = False
        new_groups = []
        i = 0
        while i < len(groups):
            if (i + 1 < len(groups) and
                sum(groups[i][1]) / len(groups[i][1]) >
                sum(groups[i+1][1]) / len(groups[i+1][1])):
                new_groups.append((groups[i][0] + groups[i+1][0],
                                   groups[i][1] + groups[i+1][1]))
                i += 2; merged = True
            else:
                new_groups.append(groups[i]); i += 1
        groups = new_groups

    return [(min(ps), max(ps), sum(ys)/len(ys)) for ps, ys in groups]

def isotonic_transform(p_hat, mapping):
    for lo, hi, cal_prob in mapping:
        if p_hat <= hi: return cal_prob
    return mapping[-1][2]

iso_mapping = isotonic_regression_pav(CAL_DATA)
data_iso_test = [(isotonic_transform(p, iso_mapping), y) for p, y in TEST_DATA]
ece_iso, _, _ = compute_ece_mce(data_iso_test)
nll_iso = compute_nll(data_iso_test)


# ─────────────────────────────────────────────────────────────────────────────
# RECALIBRATION METHOD 4 — HISTOGRAM BINNING
# ─────────────────────────────────────────────────────────────────────────────

def histogram_calibrate(cal_data, n_bins=10):
    edges = [i/n_bins for i in range(n_bins+1)]
    bins  = [[] for _ in range(n_bins)]
    for p, y in cal_data:
        b = min(int(p * n_bins), n_bins-1)
        bins[b].append(y)
    mapping = []
    for i in range(n_bins):
        lo = edges[i]; hi = edges[i+1]
        cal_prob = sum(bins[i]) / max(len(bins[i]), 1) if bins[i] else (lo+hi)/2
        mapping.append((lo, hi, cal_prob))
    return mapping

def hist_transform(p_hat, mapping):
    for lo, hi, cal_prob in mapping:
        if p_hat < hi: return cal_prob
    return mapping[-1][2]

hist_mapping = histogram_calibrate(CAL_DATA, n_bins=10)
data_hist_test = [(hist_transform(p, hist_mapping), y) for p, y in TEST_DATA]
ece_hist, _, _ = compute_ece_mce(data_hist_test)
nll_hist = compute_nll(data_hist_test)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

ece_base, _, _ = compute_ece_mce(TEST_DATA)
nll_base = compute_nll(TEST_DATA)

print()
print("=" * 68)
print("  RECALIBRATION RESULTS — TEST SET (n=1000)")
print("=" * 68)
print()
print(f"  {'Method':<22}  {'Params':>7}  {'ECE':>8}  {'NLL':>8}  {'ECE reduc':>11}")
print(f"  {'-'*58}")

results = [
    ("No calibration",       "—",       ece_base,  nll_base),
    (f"Temp. scaling (T={T_opt:.2f})", "1", ece_temp,  nll_temp),
    (f"Platt (a={a_platt:.2f},b={b_platt:.2f})","2", ece_platt, nll_platt),
    ("Isotonic regression",  "non-par", ece_iso,   nll_iso),
    ("Histogram binning",    "10 bins", ece_hist,  nll_hist),
]

for name, params, ece, nll in results:
    pct = (ece_base - ece) / ece_base * 100 if name != "No calibration" else 0
    print(f"  {name:<22}  {params:>7}  {ece:>8.4f}  {nll:>8.4f}  "
          f"{pct:>+10.1f}%")

print()
print(f"  Optimal temperature T = {T_opt:.3f}")
print(f"    T > 1 → softer distribution → less overconfident")
print(f"  Platt: a={a_platt:.3f}, b={b_platt:.3f}")
print(f"    a < 1 → scaling down logit magnitude (like T > 1)")
print(f"    b ≠ 0 → shifting baseline probability (additional bias correction)")
print()
print(f"  KEY TAKEAWAY:")
best = min(results[1:], key=lambda r: r[2])
print(f"  Best ECE: {best[0]} ({best[2]:.4f})")
print(f"  All methods improve over uncalibrated baseline ({ece_base:.4f}).")
print(f"  Temperature scaling is the simplest and often competitive.")


# ─────────────────────────────────────────────────────────────────────────────
# MURPHY DECOMPOSITION BEFORE AND AFTER TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  MURPHY BRIER DECOMPOSITION — BEFORE AND AFTER TEMPERATURE SCALING")
print("=" * 68)
print()

bs_before, unc_b, res_b, rel_b = compute_brier(TEST_DATA)
bs_after,  unc_a, res_a, rel_a = compute_brier(data_temp_test)

print(f"  {'Component':<20}  {'Before':>9}  {'After (T-scaled)':>16}  {'Change'}")
print(f"  {'-'*58}")
print(f"  {'Brier score':20}  {bs_before:>9.5f}  {bs_after:>16.5f}  {bs_after-bs_before:>+8.5f}")
print(f"  {'  UNC (irreducible)':20}  {unc_b:>9.5f}  {unc_a:>16.5f}  {unc_a-unc_b:>+8.5f}")
print(f"  {'  RES (discrimination)':20}  {res_b:>9.5f}  {res_a:>16.5f}  {res_a-res_b:>+8.5f}")
print(f"  {'  REL (reliability)':20}  {rel_b:>9.5f}  {rel_a:>16.5f}  {rel_a-rel_b:>+8.5f}")
print()
print(f"  UNC is unchanged (property of the dataset, not the model).")
print(f"  RES changes slightly (temperature scaling can shift discrimination).")
print(f"  REL improves substantially (calibration is directly targeted).")
print(f"  Net Brier score change: {bs_after-bs_before:+.5f}")
print(f"  Temperature scaling improves REL (calibration) without harming RES (discrimination).")


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION COVERAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  CONFORMAL PREDICTION — COVERAGE VERIFICATION")
print("=" * 68)
print()
print("  Approach: include class 1 in prediction set if p̂ ≥ threshold.")
print("  Threshold computed from calibration set to guarantee (1-α) coverage.")
print("  Nonconformity score: 1 - p̂(true class).")
print()

for alpha in [0.10, 0.15, 0.20]:
    scores = []
    for p, y in CAL_DATA:
        score = 1 - p if y == 1 else p
        scores.append(score)

    n_cal = len(scores)
    scores_sorted = sorted(scores)
    quantile_idx = math.ceil((n_cal + 1) * (1 - alpha)) - 1
    quantile_idx = min(max(quantile_idx, 0), n_cal - 1)
    q_hat = scores_sorted[quantile_idx]

    covered = 0
    for p, y in TEST_DATA:
        score_test = 1 - p if y == 1 else p
        if score_test <= q_hat:
            covered += 1
    empirical_coverage = covered / len(TEST_DATA)

    print(f"  Nominal (1−α) = {1-alpha:.2f}:  "
          f"threshold q̂ = {q_hat:.4f},  "
          f"empirical coverage = {empirical_coverage:.3f}  "
          f"({'≥' if empirical_coverage >= 1-alpha else '<'} {1-alpha:.2f} {'✓' if empirical_coverage >= 1-alpha else '✗'})")

print()
print("  Conformal prediction GUARANTEES coverage ≥ (1-α) for any model,")
print("  any distribution, any n ≥ 1 (given exchangeability).")
print("  Unlike ECE or temperature scaling, this guarantee is non-asymptotic.")
''',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    import streamlit as st

    st.markdown("---")
    st.subheader("⚙️ Operations")

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

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }