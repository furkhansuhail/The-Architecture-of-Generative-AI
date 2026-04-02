"""
Statistical Testing in Machine Learning
========================================

Comparing classifiers rigorously: paired t-test, McNemar's test,
Wilcoxon signed-rank test, the 5×2cv paired test, bootstrap confidence
intervals, Bonferroni / Holm-Šidák corrections for multiple comparisons,
and the practical danger of drawing conclusions from a single test-set
evaluation.

"""

import textwrap
import re

TOPIC_NAME = "Statistical Testing in Machine Learning"
DISPLAY_NAME = "11 · Statistical Testing"
ICON = "📊"
SUBTITLE = "Rigorous Model Comparison Beyond Test-Set Accuracy"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Problem: Is My Model Actually Better?

Suppose model A achieves 87.3% accuracy on the test set and model B
achieves 86.9%. Is A genuinely better? Or did it just get lucky on
these particular test examples?

    SCENARIO:
    - 1000 test examples
    - Model A: 873 correct
    - Model B: 869 correct
    - Difference: 4 examples

    Those 4 examples could easily flip the other way on a different
    sample. We need statistical machinery to distinguish real differences
    from sampling noise.

    THE FUNDAMENTAL ISSUE:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Machine learning evaluation is STATISTICAL INFERENCE.           │
    │  A test set is a SAMPLE from the data-generating process.        │
    │  Conclusions from it are subject to sampling error.              │
    │  Without uncertainty quantification, all comparisons are         │
    │  potentially misleading.                                         │
    └──────────────────────────────────────────────────────────────────┘

    Key questions that statistical tests answer:
    ─ Is model A significantly better than model B?
    ─ What is the 95% confidence interval for model A's accuracy?
    ─ After testing 10 hyperparameter settings, which improvements are real?
    ─ Does the difference hold across different data folds?

    Diagram 1 — The Estimation Pipeline:

    True distribution P(X, Y)
          │
          │  sample
          ▼
    Test set Dtest = {(x₁,y₁), ..., (xₙ,yₙ)}
          │
          │  evaluate
          ▼
    Observed accuracy θ̂ = (1/n) Σ 𝟙[f(xᵢ) = yᵢ]
          │
          │  infer
          ▼
    Confidence interval [θ_lo, θ_hi] for true accuracy θ*


──────────────────────────────────────────────────────────────────────────────
### Confidence Intervals for a Single Classifier

Before comparing two models, we need to measure one model's uncertainty.

**Normal approximation (Wald interval):**

    θ̂ = k/n   where k correct out of n examples.

    SE(θ̂) = √(θ̂(1−θ̂)/n)

    95% CI:   [θ̂ − 1.96 · SE,  θ̂ + 1.96 · SE]

    ┌──────────────────────────────────────────────────────────────────┐
    │  Example: k=873, n=1000                                          │
    │  θ̂ = 0.873                                                       │
    │  SE = √(0.873 × 0.127 / 1000) = 0.0105                           │
    │  95% CI: [0.852, 0.894]                                          │
    └──────────────────────────────────────────────────────────────────┘

    Caveats: the normal approximation breaks down when k or n−k < 5.

**Wilson interval (better for small n or extreme proportions):**

    Let z = 1.96, ñ = n + z².

    θ̃ = (k + z²/2) / ñ

    half-width = (z / ñ) · √(k(n−k)/n + z²/4)

    The Wilson interval has better coverage properties near 0 and 1.

**Bootstrap confidence interval:**

    1. Resample the test set with replacement B times.
    2. Compute accuracy θ̂_b for each bootstrap sample b.
    3. CI: [quantile(θ̂_b, 0.025), quantile(θ̂_b, 0.975)]

    Bootstrap requires no distributional assumptions.
    It is the most reliable method for small or unusual test sets.

    Diagram 2 — Bootstrap Sampling:

    Test set: [✓ ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓]   θ̂ = 7/10
                    │
                    │ resample with replacement (×1000)
                    ▼
    Bootstrap 1: [✓ ✓ ✓ ✗ ✓ ✓ ✓ ✗ ✓ ✓]   θ̂₁ = 8/10
    Bootstrap 2: [✗ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ ✓]   θ̂₂ = 8/10
    Bootstrap 3: [✓ ✗ ✗ ✓ ✓ ✓ ✗ ✗ ✓ ✓]   θ̂₃ = 6/10
    ...
    Bootstrap 1000:                          θ̂₁₀₀₀
                    │
                    ▼
    95% CI = [2.5th percentile, 97.5th percentile] of {θ̂_b}


──────────────────────────────────────────────────────────────────────────────
### Comparing Two Classifiers on the SAME Test Set

When both models are evaluated on the same test set, their errors are
CORRELATED — errors on hard examples occur for both models.
Ignoring this correlation (treating errors as independent) inflates
the apparent precision of comparisons.

**McNemar's Test — for paired binary outcomes:**

    McNemar's test uses only the DISCORDANT pairs:
    ─ nᵢⱼ = examples where model i was correct, model j was wrong.

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │                  Model B correct    Model B wrong              │
    │  Model A correct    n₁₁               n₁₀                      │
    │  Model A wrong      n₀₁               n₀₀                      │
    │                                                                │
    │  McNemar statistic (with continuity correction):               │
    │  χ² = (|n₁₀ − n₀₁| − 1)² / (n₁₀ + n₀₁)                         │
    │                                                                │
    │  Distributed as χ²(1) under H₀: models are equivalent.         │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    WHY ONLY DISCORDANT PAIRS?
    When both models are right (n₁₁) or both are wrong (n₀₀),
    those examples provide no information about which model is better.
    The test power comes from n₁₀ and n₀₁.

    EXACT MCNEMAR (small samples):
    When n₁₀ + n₀₁ < 25, use the exact binomial test:
    p = P(X ≥ max(n₁₀, n₀₁))   where X ~ Binomial(n₁₀+n₀₁, 0.5)
    This tests whether one model being right while the other is wrong
    is significantly more likely than a coin flip.

    STRENGTHS:
    ─ Naturally accounts for correlation between models.
    ─ Very powerful — uses all available discordant information.
    ─ Non-parametric, no normality assumption.

    WEAKNESS:
    ─ Only applicable when both models are evaluated on the same test set.


──────────────────────────────────────────────────────────────────────────────
### The Paired t-Test for Cross-Validated Results

When using k-fold cross-validation, each fold gives a performance
estimate. The k estimates are paired (same data, same random seed).

    PROCEDURE:
    1. Run k-fold CV for model A and model B.
    2. Compute per-fold differences: dᵢ = accᵢ(A) − accᵢ(B).
    3. t-statistic: t = d̄ / (s_d / √k)
       where d̄ = mean of differences, s_d = standard deviation.
    4. p-value from t-distribution with k−1 degrees of freedom.

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  t = d̄ / (s_d / √k)    ~  t(k−1)  under H₀: μ_d = 0              │
    │                                                                  │
    │  Confidence interval for true difference Δ:                      │
    │  d̄ ± t_{α/2, k-1} · (s_d / √k)                                   │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    CRITICAL ASSUMPTION: the k fold differences must be independent.
    In k-fold CV they are NOT truly independent — training sets overlap.
    This causes the test to be anti-conservative (too many false positives).

    CORRECTED RESAMPLED t-TEST (Nadeau & Bengio, 2003):
    Correct the standard error for the overlap:

    SE_corrected = √( (1/k + n_test/n_train) · (1/k) Σᵢ (dᵢ − d̄)² )

    where n_train and n_test are the train/test sizes per fold.

    This correction inflates the standard error, making the test
    conservative rather than anti-conservative.


──────────────────────────────────────────────────────────────────────────────
### The 5×2cv Paired Test (Dietterich, 1998)

The 5×2cv test addresses the independence problem directly.
Instead of correlated k-fold differences, it uses 5 independent 2-fold
cross-validation runs.

    PROCEDURE:
    1. Repeat 5 times (each with a different random split):
       a. Randomly split data 50/50 into sets S₁ and S₂.
       b. Train A on S₁, evaluate on S₂. Train A on S₂, evaluate on S₁.
          Same for B. Record differences p₁(i), p₂(i).
    2. For each run i:
       p̄(i) = (p₁(i) + p₂(i)) / 2
       s²(i) = (p₁(i) − p̄(i))² + (p₂(i) − p̄(i))²
    3. Test statistic:
       t = p₁(1) / √((1/5) Σᵢ s²(i))
       Distributed approximately as t(5) under H₀.

    WHY IT WORKS:
    ─ Each 2-fold pair is computed on an independent data split.
    ─ The 5 runs are truly independent → no correlation issue.
    ─ Uses only 5 degrees of freedom → conservative.
    ─ Better type I error control than standard paired t-test on k-fold CV.

    Diagram 3 — 5×2cv Design:

    Run 1:    S₁ ───train A,B──▶ eval on S₂ → diff₁
              S₂ ───train A,B──▶ eval on S₁ → diff₂  →  s²(1)
    Run 2:    (fresh split) ...                         →  s²(2)
    Run 3:    (fresh split) ...                         →  s²(3)
    Run 4:    (fresh split) ...                         →  s²(4)
    Run 5:    (fresh split) ...                         →  s²(5)
                                                             │
                                              t = diff₁ / √(mean(s²)) ~ t(5)


──────────────────────────────────────────────────────────────────────────────
### Wilcoxon Signed-Rank Test — Non-Parametric Alternative

The Wilcoxon signed-rank test is a non-parametric alternative to the
paired t-test. It does not assume normality of the differences.

    PROCEDURE:
    1. Compute differences dᵢ = perfᵢ(A) − perfᵢ(B).
    2. Exclude dᵢ = 0 (ties).
    3. Rank |dᵢ| from 1 (smallest) to m (largest).
    4. Assign each rank the sign of dᵢ.
    5. W⁺ = sum of ranks where dᵢ > 0 (A better).
       W⁻ = sum of ranks where dᵢ < 0 (B better).
    6. Test statistic: W = min(W⁺, W⁻).
       Under H₀, E[W⁺] = E[W⁻] = m(m+1)/4.
    7. Look up or approximate critical value / p-value.

    ┌──────────────────────────────────────────────────────────────────┐
    │  WHEN TO USE WILCOXON vs PAIRED t-TEST:                          │
    │                                                                  │
    │  t-test:      assumes normality, parametric, more powerful       │
    │               if normality holds (central limit theorem helps).  │
    │  Wilcoxon:    no distributional assumptions, robust to           │
    │               outliers, better when differences are skewed.      │
    │               Preferred when k is small (k < 20 folds).          │
    └──────────────────────────────────────────────────────────────────┘

    CRITICAL: Wilcoxon still suffers from the correlation problem in
    k-fold CV. The corrected standard error fix should be applied, or
    use the 5×2cv design to ensure independence.


──────────────────────────────────────────────────────────────────────────────
### Multiple Comparisons — The Family-Wise Error Rate Problem

When comparing m models or testing m hypotheses simultaneously, the
probability that at least one test gives a false positive inflates.

    INFLATION OF FALSE POSITIVES:
    Single test α = 0.05 → P(false positive) = 0.05.
    20 independent tests at α = 0.05:
    P(at least one false positive) = 1 − (1−0.05)²⁰ ≈ 0.64

    Researchers testing many models and reporting only significant ones
    are committing MULTIPLE COMPARISONS / data dredging errors.

    ┌──────────────────────────────────────────────────────────────────┐
    │  FAMILY-WISE ERROR RATE (FWER):                                  │
    │  P(at least one false rejection among all H₀) ≤ α                │
    │                                                                  │
    │  Bonferroni correction:  αᵢ = α / m                              │
    │  Reject H₀ᵢ if pᵢ ≤ α/m.  Controls FWER ≤ α.                     │
    │  Very conservative — loses power with large m.                   │
    │                                                                  │
    │  Holm–Šidák stepdown:  more powerful than Bonferroni.            │
    │  1. Sort p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎.                        │
    │  2. Compare p₍ᵢ₎ against α/(m−i+1).                               │
    │  3. Reject H₍ᵢ₎ and continue while p₍ᵢ₎ ≤ α/(m−i+1).               │
    │  4. Stop at first non-rejection (do not reject remaining).       │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

**False Discovery Rate (FDR) — Benjamini-Hochberg:**

    ┌──────────────────────────────────────────────────────────────────┐
    │  FDR = E[V/R]  where V = false rejections, R = total rejections  │
    │                                                                  │
    │  Benjamini-Hochberg procedure:                                   │
    │  1. Sort p-values: p₍₁₎ ≤ ... ≤ p₍ₘ₎.                              │
    │  2. Find largest k where p₍ₖ₎ ≤ kα/m.                             │
    │  3. Reject H₍₁₎, ..., H₍ₖ₎.                                        │
    │  4. Controls FDR ≤ α (not FWER).                                 │
    │  Less conservative than Bonferroni — appropriate when many       │
    │  tests are expected to be true (e.g., feature selection).        │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    FWER vs FDR:
    ─ FWER control (Bonferroni, Holm): controls P(any false positive).
      Use when a single false positive is catastrophic.
    ─ FDR control (BH): controls expected FRACTION of false positives.
      Use when some false positives are acceptable (discovery setting).

    Diagram 4 — FWER Inflation with Number of Tests:

    P(at least 1 false positive)
    1.0 │                             ────── (α=0.05, no correction)
        │                          ──
    0.8 │                       ──
        │                    ──
    0.6 │                 ──
        │              ──
    0.4 │          ──
        │      ──
    0.2 │   ──
        │──
    0.0 │────────────────────────────────────── m (number of tests)
        0        10        20        30

    With Bonferroni correction (α/m): P stays at α = 0.05 for all m.


──────────────────────────────────────────────────────────────────────────────
### Practical Guidelines — Choosing the Right Test

    ┌──────────────────────────────────────────────────────────────────┐
    │  SCENARIO                       RECOMMENDED TEST                 │
    │                                                                  │
    │  Single test set, both models   McNemar's test                   │
    │  evaluated on same examples     (exact version if n₁₀+n₀₁<25)    │
    │                                                                  │
    │  k-fold CV, large k (≥30)       Corrected paired t-test          │
    │                                 (Nadeau-Bengio correction)       │
    │                                                                  │
    │  k-fold CV, small k (<20)       Wilcoxon signed-rank             │
    │                                 (still apply correction)         │
    │                                                                  │
    │  Principled paired comparison   5×2cv paired test                │
    │  with type I error control      (Dietterich 1998)                │
    │                                                                  │
    │  Single accuracy estimate CI    Bootstrap (B=10,000)             │
    │                                 or Wilson interval               │
    │                                                                  │
    │  Multiple model comparison      Bonferroni/Holm + McNemar        │
    │                                 or FDR (Benjamini-Hochberg)      │
    └──────────────────────────────────────────────────────────────────┘

**Common pitfalls:**

    ─ SELECTION BIAS: choosing the best model on the test set, then
      reporting its test accuracy as if it were unbiased. The test set
      has been used for model selection → it is no longer a valid
      estimate of generalisation performance. Solution: use a separate
      hold-out set that is NEVER used during development.

    ─ SINGLE SPLIT: one train/test split is a single noisy estimate.
      Report cross-validated performance with confidence intervals.

    ─ IGNORING PRACTICAL SIGNIFICANCE: a statistically significant
      difference of 0.1% accuracy may be practically meaningless.
      Always report effect size alongside p-values.

    ─ P-HACKING: running many tests and reporting only significant ones.
      Apply multiple-comparison correction and preregister hypotheses.

    ─ INAPPROPRIATE TEST: using unpaired tests when data is paired,
      or assuming normality when checking it is warranted.

    Diagram 5 — The Right Comparison Framework:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  STEP 1: Define a single primary comparison (not 20 tests).     │
    │  STEP 2: Choose an appropriate test (see table above).          │
    │  STEP 3: Pre-specify α (typically 0.05).                        │
    │  STEP 4: Evaluate. Apply correction if multiple hypotheses.     │
    │  STEP 5: Report effect size + CI, not just p-value.             │
    │  STEP 6: Sanity check — is the difference meaningful?           │
    │  STEP 7: If possible, replicate on independent data.            │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Effect Size — Beyond Statistical Significance

Statistical significance tells us that an effect is unlikely to be
zero. It does NOT tell us the magnitude of the effect.

    COHEN'S d  (for continuous metrics like accuracy difference):
        d = (μ_A − μ_B) / s_pooled
        s_pooled = √[(s_A² + s_B²)/2]

        |d| < 0.2:   small effect
        |d| ∈ [0.2, 0.5): medium effect
        |d| ≥ 0.5:   large effect

    ODDS RATIO (for binary outcomes like errors):
        OR = (n₁₀/n₀₁) — how many times more likely is A correct when B is wrong?
        OR = 1: models are equivalent.
        OR > 1: A wins on discordant examples.

    PRACTICAL SIGNIFICANCE:
    An accuracy improvement of +0.1% (statistically significant on a
    very large test set) may not justify retraining, redeployment, or
    the added complexity of a new model.

    Always ask: does the measured difference matter in the application?

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Confidence Intervals — Bootstrap, Wilson, and Normal Approximation": {
        "description": (
            "Compares three confidence interval methods for classifier accuracy: "
            "the normal (Wald) approximation, the Wilson score interval, and the "
            "bootstrap percentile interval. Shows how CI width depends on test set "
            "size and how well each method maintains nominal coverage (95%). "
            "Includes a coverage simulation: generates 1000 repetitions of a "
            "classification experiment and measures how often each CI method "
            "contains the true accuracy. Visualises the three CIs across different "
            "test set sizes from n=50 to n=5000."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Confidence interval functions
# ─────────────────────────────────────────────────────────────────────────────

def ci_wald(k, n, alpha=0.05):
    """Normal (Wald) confidence interval for a proportion."""
    z   = stats.norm.ppf(1 - alpha / 2)
    p   = k / n
    se  = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se

def ci_wilson(k, n, alpha=0.05):
    """Wilson score interval — better coverage near 0 and 1."""
    z  = stats.norm.ppf(1 - alpha / 2)
    p  = k / n
    nt = n + z**2
    pt = (k + z**2 / 2) / nt
    hw = (z / nt) * np.sqrt(k * (n - k) / n + z**2 / 4)
    return max(0, pt - hw), min(1, pt + hw)

def ci_bootstrap(errors, alpha=0.05, B=5000, rng=None):
    """
    Nonparametric bootstrap CI for accuracy.
    errors: binary array (0=correct, 1=wrong)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n   = len(errors)
    acc = 1 - errors.mean()
    boot_accs = np.array([
        1 - rng.choice(errors, n, replace=True).mean()
        for _ in range(B)
    ])
    return (np.percentile(boot_accs, 100 * alpha / 2),
            np.percentile(boot_accs, 100 * (1 - alpha / 2)))

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: CIs at different test-set sizes (true accuracy = 0.80)
# ─────────────────────────────────────────────────────────────────────────────

true_acc = 0.80
n_vals   = [50, 100, 200, 500, 1000, 2000, 5000]

print("=" * 65)
print("  CONFIDENCE INTERVALS FOR CLASSIFIER ACCURACY")
print("=" * 65)
print()
print(f"  True accuracy: {true_acc:.2f}")
print()
print(f"  {'n':>6}  {'Wald':>18}  {'Wilson':>18}  {'Bootstrap':>18}")
print("  " + "─" * 64)

wald_lo_list = []; wald_hi_list = []
wils_lo_list = []; wils_hi_list = []
boot_lo_list = []; boot_hi_list = []

rng_ci = np.random.default_rng(0)
for n in n_vals:
    # Simulate one evaluation run
    errors = (rng_ci.random(n) > true_acc).astype(int)
    k      = n - errors.sum()
    w_lo, w_hi = ci_wald(k, n)
    wl_lo, wl_hi = ci_wilson(k, n)
    b_lo, b_hi = ci_bootstrap(errors, B=500, rng=np.random.default_rng(n))
    print(f"  {n:>6}  [{w_lo:.3f}, {w_hi:.3f}] {w_hi-w_lo:.3f}"
          f"  [{wl_lo:.3f}, {wl_hi:.3f}] {wl_hi-wl_lo:.3f}"
          f"  [{b_lo:.3f}, {b_hi:.3f}] {b_hi-b_lo:.3f}")
    wald_lo_list.append(w_lo); wald_hi_list.append(w_hi)
    wils_lo_list.append(wl_lo); wils_hi_list.append(wl_hi)
    boot_lo_list.append(b_lo); boot_hi_list.append(b_hi)

print()

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Coverage simulation — how often does each CI contain the truth?
# ─────────────────────────────────────────────────────────────────────────────

print("  COVERAGE SIMULATION (target: 95%):")
print(f"  {'n':>6}  {'Wald coverage':>15}  {'Wilson coverage':>17}  {'Bootstrap cov':>15}")
print("  " + "─" * 57)

N_reps = 100
rng_cov = np.random.default_rng(1)

for n in [50, 100, 200, 500, 1000]:
    wald_cover = 0; wils_cover = 0; boot_cover = 0
    for rep in range(N_reps):
        errs = (rng_cov.random(n) > true_acc).astype(int)
        k    = n - errs.sum()
        w_l, w_h  = ci_wald(k, n)
        wl_l, wl_h = ci_wilson(k, n)
        b_l, b_h  = ci_bootstrap(errs, B=200, rng=np.random.default_rng(rep))
        wald_cover += int(w_l  <= true_acc <= w_h)
        wils_cover += int(wl_l <= true_acc <= wl_h)
        boot_cover += int(b_l  <= true_acc <= b_h)
    print(f"  {n:>6}  {wald_cover/N_reps:>15.3f}  {wils_cover/N_reps:>17.3f}  "
          f"{boot_cover/N_reps:>15.3f}")

print()
print("  OBSERVATIONS:")
print("  - Wald interval under-covers at small n (discrete binomial vs normal).")
print("  - Wilson interval has better coverage, especially for small n.")
print("  - Bootstrap is distribution-free and has good coverage across n.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Part 3: CI width as a function of n — sample size planning
# ─────────────────────────────────────────────────────────────────────────────

print("  SAMPLE SIZE NEEDED for CI half-width ε:")
alphas = [0.01, 0.05, 0.10]
epsilons = [0.01, 0.02, 0.05]
print()
print(f"  {'α':>6}  {'ε':>6}  {'n_required (Wald)':>20}")
print("  " + "─" * 38)
for al in alphas:
    z = stats.norm.ppf(1 - al / 2)
    for ep in epsilons:
        # Worst case p=0.5, SE = sqrt(p(1-p)/n) => n = z²p(1-p)/ε²
        n_req = int(np.ceil((z**2 * 0.25) / ep**2))
        print(f"  {al:>6.2f}  {ep:>6.3f}  {n_req:>20,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Confidence Intervals for Classifier Accuracy",
             fontsize=13, fontweight="bold")

n_arr = np.array(n_vals)

# Panel 0: CI width vs n for all methods
ax = axes[0]
wald_widths = np.array(wald_hi_list) - np.array(wald_lo_list)
wils_widths = np.array(wils_hi_list) - np.array(wils_lo_list)
boot_widths = np.array(boot_hi_list) - np.array(boot_lo_list)
ax.plot(n_arr, wald_widths, "o-", color="steelblue", lw=2, ms=7, label="Wald")
ax.plot(n_arr, wils_widths, "s-", color="seagreen",  lw=2, ms=7, label="Wilson")
ax.plot(n_arr, boot_widths, "^-", color="tomato",    lw=2, ms=7, label="Bootstrap")
n_theory = np.linspace(50, 5000, 200)
ax.plot(n_theory, 2 * 1.96 * np.sqrt(0.2 * 0.8 / n_theory), "k--",
        lw=1.5, label="Theoretical (Wald, p=0.8)")
ax.set_xscale("log"); ax.set_xlabel("Test set size n")
ax.set_ylabel("CI width"); ax.set_title("CI Width vs Test Set Size\\n(1/√n decay)",
                                          fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 1: Coverage at different n (horizontal bar chart from simulation)
ax = axes[1]
n_cov = [50, 100, 200, 500, 1000]
rng_sim = np.random.default_rng(99)

# Re-run coverage for plot
wald_covs = []; wils_covs = []; boot_covs = []
for n in n_cov:
    wc = 0; wlc = 0; bc = 0
    for _ in range(80):
        errs = (rng_sim.random(n) > true_acc).astype(int)
        k    = n - errs.sum()
        wl_v, wh_v = ci_wald(k, n)
        wll_v,wlh_v= ci_wilson(k, n)
        bl_v, bh_v = ci_bootstrap(errs, B=200, rng=np.random.default_rng(k+n))
        wc  += int(wl_v  <= true_acc <= wh_v)
        wlc += int(wll_v <= true_acc <= wlh_v)
        bc  += int(bl_v  <= true_acc <= bh_v)
    wald_covs.append(wc / 80)
    wils_covs.append(wlc / 80)
    boot_covs.append(bc / 80)

x_pos = np.arange(len(n_cov)); w = 0.25
ax.bar(x_pos - w, wald_covs, w, color="steelblue", alpha=0.8, label="Wald")
ax.bar(x_pos,     wils_covs, w, color="seagreen",  alpha=0.8, label="Wilson")
ax.bar(x_pos + w, boot_covs, w, color="tomato",    alpha=0.8, label="Bootstrap")
ax.axhline(0.95, color="black", lw=2, ls="--", label="Target 95%")
ax.set_xticks(x_pos); ax.set_xticklabels([str(n) for n in n_cov])
ax.set_ylim(0.8, 1.0); ax.set_xlabel("Test set size n")
ax.set_ylabel("Empirical coverage (80 simulations)")
ax.set_title("Coverage: Which CIs Achieve 95%?\\n(below line = undercoverage)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# Panel 2: Specific CIs for one evaluation (n=200, varying true acc)
ax = axes[2]
true_accs_range = np.linspace(0.5, 0.95, 12)
n_fixed = 200

for i, ta in enumerate(true_accs_range):
    errors_ex = (np.random.RandomState(i).random(n_fixed) > ta).astype(int)
    k_ex      = n_fixed - errors_ex.sum()
    p_obs     = k_ex / n_fixed
    wl_lo_e, wl_hi_e = ci_wilson(k_ex, n_fixed)
    ax.plot([wl_lo_e, wl_hi_e], [ta, ta], "-", color="steelblue",
            lw=3, alpha=0.7)
    ax.scatter(p_obs, ta, c="steelblue", s=40, zorder=5)
    # Mark whether CI contains true value
    if not (wl_lo_e <= ta <= wl_hi_e):
        ax.scatter(p_obs, ta, c="tomato", s=80, zorder=6, marker="x")

ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=2, label="True accuracy (diagonal)")
ax.set_xlabel("Observed accuracy / Wilson CI")
ax.set_ylabel("True accuracy")
ax.set_title(f"Wilson CIs for n={n_fixed}\\n(× marks CIs that miss true value)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("confidence_intervals.png", dpi=110)
print("  Plot saved → confidence_intervals.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · McNemar's Test and Paired t-Test for Model Comparison": {
        "description": (
            "Implements McNemar's test (both approximate χ² and exact binomial "
            "versions) and the paired t-test on cross-validated results from "
            "scratch. Trains three classifier pairs on a dataset — a genuinely "
            "different pair (LR vs GBM) and a trivially different pair (LR vs "
            "LR with a tiny regularisation change) — and applies both tests. "
            "Shows the contingency table, discordant pair counts, χ² statistic, "
            "p-value, and 95% CI for the accuracy difference. Also shows the "
            "corrected paired t-test (Nadeau-Bengio) vs uncorrected."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,random_state=None,**kw):
    rng=_np_impl.random.default_rng(random_state)
    y=rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=n_redundant
    Xr=(Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else _np_impl.empty((n_samples,0)))
    nn2=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    parts=[a for a in [Xi,Xr,Xn] if a.shape[1]>0]
    return _np_impl.hstack(parts)[:,:n_features],y.astype(int)

def train_test_split(*arrays,test_size=0.4,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=len(arrays[0])
    n_tr=int(n*(1-test_size)); idx=rng.permutation(n); ti,vi=idx[:n_tr],idx[n_tr:]
    out=[]
    for a in arrays: out+=[a[ti],a[vi]]
    return out

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y):
        y=_np_impl.asarray(y); classes=_np_impl.unique(y)
        class_idx={c:_np_impl.where(y==c)[0] for c in classes}
        if self.shuffle:
            rng=_np_impl.random.default_rng(self.random_state)
            for c in classes: rng.shuffle(class_idx[c])
        fold_idx=[[] for _ in range(self.n_splits)]
        for c in classes:
            for j,i in enumerate(class_idx[c]):
                fold_idx[j%self.n_splits].append(i)
        for k in range(self.n_splits):
            val=_np_impl.array(fold_idx[k])
            train=_np_impl.concatenate([fold_idx[j] for j in range(self.n_splits) if j!=k])
            yield train.astype(int), val.astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None,**kw):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict(self,X): return (_sig(X@self.coef_.ravel()+self.intercept_[0])>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"LogisticRegression(C={self.C})"

class _DTR:
    def __init__(self,max_depth=3,min_s=5): self.max_depth=max_depth; self.min_s=min_s
    def _split(self,X,r):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>10: ts=ts[_np_impl.linspace(0,len(ts)-1,10).astype(int)]
            for t in ts:
                l=X[:,f]<=t; rr=~l
                if l.sum()<self.min_s or rr.sum()<self.min_s: continue
                g=_np_impl.var(r)*len(r)-_np_impl.var(r[l])*l.sum()-_np_impl.var(r[rr])*rr.sum()
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np_impl.var(r)<1e-12: return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class GradientBoostingClassifier:
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=min(n_estimators,30); self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        self._classes=_np_impl.unique(y)
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return self._classes[(_sig(F)>=0.5).astype(int)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"GradientBoostingClassifier(n_estimators={self.n_estimators})"

class _DTC:
    def __init__(self,max_depth=8,max_features=None,min_s=2):
        self.max_depth=max_depth; self.max_features=max_features; self.min_s=min_s
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y,rng):
        d=X.shape[1]
        feats=(rng.choice(d,min(self.max_features,d),replace=False)
               if self.max_features else _np_impl.arange(d))
        best=None
        for f in feats:
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>8: ts=ts[_np_impl.linspace(0,len(ts)-1,8).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_s or r.sum()<self.min_s: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,d,rng):
        cls,cnt=_np_impl.unique(y,return_counts=True)
        prob=_np_impl.zeros(len(self._classes))
        for c,n in zip(cls,cnt): prob[_np_impl.searchsorted(self._classes,c)]=n/len(y)
        if d>=self.max_depth or len(_np_impl.unique(y))==1 or len(y)<=self.min_s: return ('L',prob)
        sp=self._best(X,y,rng)
        if sp is None: return ('L',prob)
        _,f,t=sp; l=X[:,f]<=t
        return ('N',f,t,self._build(X[l],y[l],d+1,rng),self._build(X[~l],y[~l],d+1,rng))
    def fit(self,X,y,classes,rng):
        self._classes=classes; self._tree=self._build(X,y,0,rng); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict_proba(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=100,max_depth=None,random_state=None,**kw):
        self.n_estimators=min(n_estimators,20); self.max_depth=max_depth or 8
        self.random_state=random_state
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); rng=_np_impl.random.default_rng(self.random_state)
        n,d=X.shape; mf=max(1,int(_np_impl.sqrt(d))); self._trees=[]
        for _ in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_DTC(max_depth=self.max_depth,max_features=mf)
            t.fit(X[idx],y[idx],self._classes,rng); self._trees.append(t)
        return self
    def predict(self,X):
        proba=_np_impl.mean([t.predict_proba(X) for t in self._trees],axis=0)
        return self._classes[proba.argmax(1)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"RandomForestClassifier(n_estimators={self.n_estimators})"

class DecisionTreeClassifier:
    def __init__(self,max_depth=None,random_state=None,**kw):
        self.max_depth=max_depth or 999; self.random_state=random_state
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>15: ts=ts[_np_impl.linspace(0,len(ts)-1,15).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<2 or r.sum()<2: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,d):
        cls,cnt=_np_impl.unique(y,return_counts=True); pred=cls[cnt.argmax()]
        if d>=self.max_depth or len(_np_impl.unique(y))==1 or len(y)<=2: return ('L',pred)
        sp=self._best(X,y)
        if sp is None: return ('L',pred)
        _,f,t=sp; l=X[:,f]<=t
        return ('N',f,t,self._build(X[l],y[l],d+1),self._build(X[~l],y[~l],d+1))
    def fit(self,X,y): self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"DecisionTreeClassifier(max_depth={self.max_depth})"


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Statistical test implementations
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(correct_A, correct_B, exact=False):
    """
    McNemar's test for two paired binary classifiers.
    correct_A, correct_B: binary arrays (1=correct, 0=wrong).
    Returns: n10, n01, statistic, p_value, odds_ratio.
    """
    # Contingency table
    n11 = ((correct_A == 1) & (correct_B == 1)).sum()
    n10 = ((correct_A == 1) & (correct_B == 0)).sum()  # A right, B wrong
    n01 = ((correct_A == 0) & (correct_B == 1)).sum()  # A wrong, B right
    n00 = ((correct_A == 0) & (correct_B == 0)).sum()
    n   = len(correct_A)
    acc_A = correct_A.mean()
    acc_B = correct_B.mean()

    if exact or (n10 + n01) < 25:
        # Exact binomial test: H0: p(n10) = 0.5 among discordant
        total = n10 + n01
        k     = max(n10, n01)
        p_val = 2 * binom.sf(k - 1, total, 0.5)  # two-sided
        stat  = None
    else:
        # Approximate McNemar with continuity correction
        stat  = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
        p_val = stats.chi2.sf(stat, df=1)

    odds_ratio = (n10 / n01) if n01 > 0 else float("inf")
    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "n": n, "acc_A": acc_A, "acc_B": acc_B,
        "stat": stat, "p_val": p_val, "odds_ratio": odds_ratio,
        "n_discordant": n10 + n01,
    }

def paired_t_test(diffs, n_train=None, n_test=None, corrected=False):
    """
    Paired t-test on per-fold accuracy differences.
    diffs: array of (acc_A - acc_B) per fold.
    corrected: apply Nadeau-Bengio correction if True.
    """
    k     = len(diffs)
    d_bar = diffs.mean()
    s_d   = diffs.std(ddof=1)

    if corrected and n_train is not None and n_test is not None:
        # Nadeau-Bengio corrected variance
        rho   = n_test / n_train
        s2_c  = (1/k + rho) * (s_d**2)
        se    = np.sqrt(s2_c)
    else:
        se = s_d / np.sqrt(k)

    t_stat = d_bar / (se + 1e-15)
    p_val  = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    ci_hw  = stats.t.ppf(0.975, df=k - 1) * se
    return {
        "d_bar": d_bar, "s_d": s_d, "se": se,
        "t_stat": t_stat, "p_val": p_val,
        "ci_lo": d_bar - ci_hw, "ci_hi": d_bar + ci_hw,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: McNemar test on a single test set
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  McNEMAR'S TEST AND PAIRED t-TEST FOR MODEL COMPARISON")
print("=" * 65)
print()

X, y = make_classification(n_samples=600, n_features=20, n_informative=10,
                            n_redundant=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
sc   = StandardScaler().fit(X_tr)
X_trs = sc.transform(X_tr); X_tes = sc.transform(X_te)

# Train several models
lr_model  = LogisticRegression(C=1.0,   max_iter=500).fit(X_trs, y_tr)
lr_tiny   = LogisticRegression(C=1.001, max_iter=500).fit(X_trs, y_tr)
gbm_model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X_trs, y_tr)
rf_model  = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_trs, y_tr)
dt_model  = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_trs, y_tr)

# Correct arrays
correct = {
    "LR (C=1.0)":   (lr_model.predict(X_tes) == y_te).astype(int),
    "LR (C=1.001)": (lr_tiny.predict(X_tes) == y_te).astype(int),
    "GBM":          (gbm_model.predict(X_tes) == y_te).astype(int),
    "RF":           (rf_model.predict(X_tes) == y_te).astype(int),
    "DT (depth=3)": (dt_model.predict(X_tes) == y_te).astype(int),
}

print("  MODEL ACCURACIES ON TEST SET:")
for name, corr in correct.items():
    print(f"    {name:>14}: {corr.mean():.4f}  ({corr.sum()}/{len(corr)} correct)")
print()

# McNemar comparisons
comparisons = [
    ("LR (C=1.0)", "LR (C=1.001)", "Trivial difference (expect non-significant)"),
    ("LR (C=1.0)", "GBM",          "Real difference (expect significant)"),
    ("GBM",        "RF",           "Similar strong models"),
    ("GBM",        "DT (depth=3)", "Strong vs weak (expect significant)"),
]

print("  McNEMAR TEST RESULTS:")
all_mcn = []
for A_name, B_name, note in comparisons:
    r = mcnemar_test(correct[A_name], correct[B_name])
    all_mcn.append((A_name, B_name, r, note))
    sig = "***" if r["p_val"] < 0.001 else "**" if r["p_val"] < 0.01 else "*" if r["p_val"] < 0.05 else "n.s."
    print(f"\\n  {A_name} vs {B_name}  [{note}]")
    print(f"    Contingency: n₁₁={r['n11']} n₁₀={r['n10']} n₀₁={r['n01']} n₀₀={r['n00']}")
    print(f"    Discordant pairs: {r['n_discordant']}   Odds ratio: {r['odds_ratio']:.3f}")
    print(f"    Acc(A)={r['acc_A']:.4f}  Acc(B)={r['acc_B']:.4f}  Diff={r['acc_A']-r['acc_B']:+.4f}")
    print(f"    p-value = {r['p_val']:.6f}  {sig}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Paired t-test on k-fold CV results
# ─────────────────────────────────────────────────────────────────────────────

print("  PAIRED t-TEST ON 5-FOLD CROSS-VALIDATION:")
k_folds  = 5
skf      = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)

# Collect per-fold accuracies for LR vs GBM
fold_acc_lr  = []
fold_acc_gbm = []
fold_acc_dt  = []
n_train_folds = []; n_test_folds = []

for tr_idx, te_idx in skf.split(X, y):
    X_f_tr, X_f_te = X[tr_idx], X[te_idx]
    y_f_tr, y_f_te = y[tr_idx], y[te_idx]
    sc_f = StandardScaler().fit(X_f_tr)
    X_ftr = sc_f.transform(X_f_tr); X_fte = sc_f.transform(X_f_te)
    lr_f  = LogisticRegression(C=1.0, max_iter=500).fit(X_ftr, y_f_tr)
    gbm_f = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X_ftr, y_f_tr)
    dt_f  = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_ftr, y_f_tr)
    fold_acc_lr.append(lr_f.score(X_fte, y_f_te))
    fold_acc_gbm.append(gbm_f.score(X_fte, y_f_te))
    fold_acc_dt.append(dt_f.score(X_fte, y_f_te))
    n_train_folds.append(len(tr_idx))
    n_test_folds.append(len(te_idx))

fold_acc_lr  = np.array(fold_acc_lr)
fold_acc_gbm = np.array(fold_acc_gbm)
fold_acc_dt  = np.array(fold_acc_dt)
n_tr_avg     = int(np.mean(n_train_folds))
n_te_avg     = int(np.mean(n_test_folds))

print()
print(f"  {'Fold':>6}  {'LR':>8}  {'GBM':>8}  {'Diff(GBM-LR)':>14}  {'DT':>8}  {'Diff(GBM-DT)':>14}")
print("  " + "─" * 62)
for i, (a, b, c) in enumerate(zip(fold_acc_lr, fold_acc_gbm, fold_acc_dt)):
    print(f"  {i+1:>6}  {a:>8.4f}  {b:>8.4f}  {b-a:>14.4f}  {c:>8.4f}  {b-c:>14.4f}")
print()

for label, diffs in [("GBM vs LR", fold_acc_gbm - fold_acc_lr),
                      ("GBM vs DT", fold_acc_gbm - fold_acc_dt)]:
    print(f"  {label}:")
    t_unc = paired_t_test(diffs, corrected=False)
    t_cor = paired_t_test(diffs, n_train=n_tr_avg, n_test=n_te_avg, corrected=True)
    sig_u = "***" if t_unc["p_val"] < 0.001 else "**" if t_unc["p_val"] < 0.01 else "*" if t_unc["p_val"] < 0.05 else "n.s."
    sig_c = "***" if t_cor["p_val"] < 0.001 else "**" if t_cor["p_val"] < 0.01 else "*" if t_cor["p_val"] < 0.05 else "n.s."
    print(f"    Mean diff: {t_unc['d_bar']:+.4f}  95% CI: [{t_unc['ci_lo']:+.4f}, {t_unc['ci_hi']:+.4f}]")
    print(f"    Standard t-test:  t={t_unc['t_stat']:.3f}  p={t_unc['p_val']:.5f}  {sig_u}")
    print(f"    Corrected t-test: t={t_cor['t_stat']:.3f}  p={t_cor['p_val']:.5f}  {sig_c}  (Nadeau-Bengio)")
    if t_unc["p_val"] < 0.05 and t_cor["p_val"] >= 0.05:
        print(f"    → Standard test significant but CORRECTED test is not!")
        print(f"      (uncorrected test anti-conservative: correlation ignored)")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("McNemar's Test and Paired t-Test for Model Comparison",
             fontsize=13, fontweight="bold")

# Panel (0,0): Contingency table visualisation (LR vs GBM)
ax = axes[0, 0]
r = all_mcn[1][2]   # LR vs GBM
matrix = np.array([[r["n11"], r["n10"]], [r["n01"], r["n00"]]])
im = ax.imshow(matrix, cmap="Blues", aspect="auto")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["GBM correct", "GBM wrong"], fontsize=9)
ax.set_yticklabels(["LR correct", "LR wrong"], fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"n={matrix[i,j]}", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if matrix[i,j] > matrix.max() * 0.6 else "black")
labels = [["n₁₁ (both right)", "n₁₀ (LR right, GBM wrong)"],
          ["n₀₁ (LR wrong, GBM right)", "n₀₀ (both wrong)"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i + 0.35, labels[i][j], ha="center", va="center",
                fontsize=7, color="gray")
ax.set_title(f"McNemar Contingency Table: LR vs GBM\\np={r['p_val']:.5f} — {'significant' if r['p_val']<0.05 else 'not significant'}",
             fontweight="bold")
plt.colorbar(im, ax=ax)

# Panel (0,1): Per-fold accuracy — LR vs GBM
ax = axes[0, 1]
folds_x = np.arange(1, k_folds + 1)
ax.plot(folds_x, fold_acc_lr,  "o-", color="steelblue", lw=2, ms=8, label="LR")
ax.plot(folds_x, fold_acc_gbm, "s-", color="tomato",    lw=2, ms=8, label="GBM")
ax.plot(folds_x, fold_acc_dt,  "^-", color="seagreen",  lw=2, ms=8, label="DT (depth=3)")
ax.axhline(fold_acc_lr.mean(),  color="steelblue", lw=1, ls="--", alpha=0.6)
ax.axhline(fold_acc_gbm.mean(), color="tomato",    lw=1, ls="--", alpha=0.6)
ax.axhline(fold_acc_dt.mean(),  color="seagreen",  lw=1, ls="--", alpha=0.6)
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
ax.set_title("Per-Fold Accuracy: LR vs GBM vs DT\\n(lines show mean per model)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xticks(folds_x)

# Panel (0,2): Per-fold differences and CI
ax = axes[0, 2]
diffs_gbm_lr = fold_acc_gbm - fold_acc_lr
diffs_gbm_dt = fold_acc_gbm - fold_acc_dt
ax.axhline(0, color="black", lw=2, ls="--", label="No difference")
ax.plot(folds_x, diffs_gbm_lr, "o-", color="tomato",    lw=2, ms=8, label="GBM−LR")
ax.plot(folds_x, diffs_gbm_dt, "s-", color="seagreen",  lw=2, ms=8, label="GBM−DT")
ax.fill_between(folds_x, diffs_gbm_lr, alpha=0.15, color="tomato")
ax.fill_between(folds_x, diffs_gbm_dt, alpha=0.15, color="seagreen")
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy difference")
ax.set_title("Per-Fold Differences\\n(positive = GBM better than other)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(folds_x)

# Panel (1,0): p-value comparison — standard vs corrected t-test
ax = axes[1, 0]
labels_pairs = ["GBM vs LR", "GBM vs DT"]
diffs_all    = [fold_acc_gbm - fold_acc_lr, fold_acc_gbm - fold_acc_dt]
p_std_list   = []; p_cor_list = []
for diffs_pair in diffs_all:
    p_std_list.append(paired_t_test(diffs_pair, corrected=False)["p_val"])
    p_cor_list.append(paired_t_test(diffs_pair, n_train=n_tr_avg,
                                     n_test=n_te_avg, corrected=True)["p_val"])
x_pair = np.arange(len(labels_pairs))
ax.bar(x_pair - 0.2, p_std_list, 0.35, color="steelblue", alpha=0.8, label="Standard t-test")
ax.bar(x_pair + 0.2, p_cor_list, 0.35, color="tomato",    alpha=0.8, label="Corrected t-test (NB)")
ax.axhline(0.05, color="black", lw=2, ls="--", label="α=0.05")
ax.set_xticks(x_pair); ax.set_xticklabels(labels_pairs)
ax.set_ylabel("p-value"); ax.set_ylim(0, 0.5)
ax.set_title("Standard vs Corrected Paired t-Test\\n(correction inflates p-values — more conservative)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for i, (p_s, p_c) in enumerate(zip(p_std_list, p_cor_list)):
    ax.text(i - 0.2, p_s + 0.01, f"{p_s:.3f}", ha="center", fontsize=9)
    ax.text(i + 0.2, p_c + 0.01, f"{p_c:.3f}", ha="center", fontsize=9)

# Panel (1,1): McNemar p-values across all comparisons
ax = axes[1, 1]
comp_names = [f"{a} vs {b}" for a, b, _, _ in all_mcn]
p_values   = [r["p_val"] for _, _, r, _ in all_mcn]
colors_bar = ["seagreen" if p < 0.05 else "gray" for p in p_values]
bars = ax.barh(comp_names, [-np.log10(p) for p in p_values], color=colors_bar, alpha=0.8)
ax.axvline(-np.log10(0.05), color="red", lw=2, ls="--", label="α=0.05")
ax.axvline(-np.log10(0.001), color="orange", lw=1.5, ls=":", label="α=0.001")
ax.set_xlabel("−log₁₀(p-value)  [larger = more significant]")
ax.set_title("McNemar p-Values (−log scale)\\n(green = significant, gray = not significant)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")
for i, (p, bar) in enumerate(zip(p_values, bars)):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"p={p:.4f}", va="center", fontsize=8)

# Panel (1,2): Effect sizes and 95% CIs
ax = axes[1, 2]
ax.axis("off")
rows_t = []
for name_pair, diffs_p in [("GBM vs LR", fold_acc_gbm - fold_acc_lr),
                              ("GBM vs DT", fold_acc_gbm - fold_acc_dt)]:
    r_std = paired_t_test(diffs_p, corrected=False)
    r_cor = paired_t_test(diffs_p, n_train=n_tr_avg, n_test=n_te_avg, corrected=True)
    cohen_d = r_std["d_bar"] / (diffs_p.std() + 1e-10)
    rows_t.append([name_pair,
                    f"{r_std['d_bar']:+.4f}",
                    f"[{r_std['ci_lo']:+.4f}, {r_std['ci_hi']:+.4f}]",
                    f"{cohen_d:.3f}",
                    f"{r_std['p_val']:.4f}",
                    f"{r_cor['p_val']:.4f}"])
headers_t = ["Comparison", "Mean Δ", "95% CI (std)", "Cohen d", "p (std)", "p (corr)"]
table = ax.table(cellText=rows_t, colLabels=headers_t, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.1, 2.8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
ax.set_title("Effect Sizes and Confidence Intervals", fontweight="bold")

plt.tight_layout()
plt.savefig("mcnemar_paired_ttest.png", dpi=110)
print("  Plot saved → mcnemar_paired_ttest.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Multiple Comparisons — Bonferroni, Holm, and Benjamini-Hochberg": {
        "description": (
            "Demonstrates the multiple comparisons problem in ML model selection. "
            "Simulates testing 20 random 'improvements' to a baseline (where the "
            "true effect is zero) and shows the inflated false positive rate. "
            "Implements Bonferroni correction, Holm-Šidák stepdown, and "
            "Benjamini-Hochberg FDR procedure from scratch. Applies all three "
            "to a real comparison: 10 hyperparameter settings of a classifier. "
            "Visualises p-value distributions under H₀, FWER vs FDR tradeoffs, "
            "and which improvements survive correction."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,random_state=None,**kw):
    rng=_np_impl.random.default_rng(random_state)
    y=rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=n_redundant
    Xr=(Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else _np_impl.empty((n_samples,0)))
    nn2=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    parts=[a for a in [Xi,Xr,Xn] if a.shape[1]>0]
    return _np_impl.hstack(parts)[:,:n_features],y.astype(int)

def train_test_split(*arrays,test_size=0.4,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=len(arrays[0])
    n_tr=int(n*(1-test_size)); idx=rng.permutation(n); ti,vi=idx[:n_tr],idx[n_tr:]
    out=[]
    for a in arrays: out+=[a[ti],a[vi]]
    return out

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y):
        y=_np_impl.asarray(y); classes=_np_impl.unique(y)
        class_idx={c:_np_impl.where(y==c)[0] for c in classes}
        if self.shuffle:
            rng=_np_impl.random.default_rng(self.random_state)
            for c in classes: rng.shuffle(class_idx[c])
        fold_idx=[[] for _ in range(self.n_splits)]
        for c in classes:
            for j,i in enumerate(class_idx[c]):
                fold_idx[j%self.n_splits].append(i)
        for k in range(self.n_splits):
            val=_np_impl.array(fold_idx[k])
            train=_np_impl.concatenate([fold_idx[j] for j in range(self.n_splits) if j!=k])
            yield train.astype(int), val.astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None,**kw):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict(self,X): return (_sig(X@self.coef_.ravel()+self.intercept_[0])>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"LogisticRegression(C={self.C})"

class _DTR:
    def __init__(self,max_depth=3,min_s=5): self.max_depth=max_depth; self.min_s=min_s
    def _split(self,X,r):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>10: ts=ts[_np_impl.linspace(0,len(ts)-1,10).astype(int)]
            for t in ts:
                l=X[:,f]<=t; rr=~l
                if l.sum()<self.min_s or rr.sum()<self.min_s: continue
                g=_np_impl.var(r)*len(r)-_np_impl.var(r[l])*l.sum()-_np_impl.var(r[rr])*rr.sum()
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np_impl.var(r)<1e-12: return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class GradientBoostingClassifier:
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=min(n_estimators,30); self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        self._classes=_np_impl.unique(y)
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return self._classes[(_sig(F)>=0.5).astype(int)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"GradientBoostingClassifier(n_estimators={self.n_estimators})"

class _DTC:
    def __init__(self,max_depth=8,max_features=None,min_s=2):
        self.max_depth=max_depth; self.max_features=max_features; self.min_s=min_s
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y,rng):
        d=X.shape[1]
        feats=(rng.choice(d,min(self.max_features,d),replace=False)
               if self.max_features else _np_impl.arange(d))
        best=None
        for f in feats:
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>8: ts=ts[_np_impl.linspace(0,len(ts)-1,8).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_s or r.sum()<self.min_s: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,d,rng):
        cls,cnt=_np_impl.unique(y,return_counts=True)
        prob=_np_impl.zeros(len(self._classes))
        for c,n in zip(cls,cnt): prob[_np_impl.searchsorted(self._classes,c)]=n/len(y)
        if d>=self.max_depth or len(_np_impl.unique(y))==1 or len(y)<=self.min_s: return ('L',prob)
        sp=self._best(X,y,rng)
        if sp is None: return ('L',prob)
        _,f,t=sp; l=X[:,f]<=t
        return ('N',f,t,self._build(X[l],y[l],d+1,rng),self._build(X[~l],y[~l],d+1,rng))
    def fit(self,X,y,classes,rng):
        self._classes=classes; self._tree=self._build(X,y,0,rng); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict_proba(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=100,max_depth=None,random_state=None,**kw):
        self.n_estimators=min(n_estimators,20); self.max_depth=max_depth or 8
        self.random_state=random_state
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); rng=_np_impl.random.default_rng(self.random_state)
        n,d=X.shape; mf=max(1,int(_np_impl.sqrt(d))); self._trees=[]
        for _ in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_DTC(max_depth=self.max_depth,max_features=mf)
            t.fit(X[idx],y[idx],self._classes,rng); self._trees.append(t)
        return self
    def predict(self,X):
        proba=_np_impl.mean([t.predict_proba(X) for t in self._trees],axis=0)
        return self._classes[proba.argmax(1)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"RandomForestClassifier(n_estimators={self.n_estimators})"

class DecisionTreeClassifier:
    def __init__(self,max_depth=None,random_state=None,**kw):
        self.max_depth=max_depth or 999; self.random_state=random_state
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>15: ts=ts[_np_impl.linspace(0,len(ts)-1,15).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<2 or r.sum()<2: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,d):
        cls,cnt=_np_impl.unique(y,return_counts=True); pred=cls[cnt.argmax()]
        if d>=self.max_depth or len(_np_impl.unique(y))==1 or len(y)<=2: return ('L',pred)
        sp=self._best(X,y)
        if sp is None: return ('L',pred)
        _,f,t=sp; l=X[:,f]<=t
        return ('N',f,t,self._build(X[l],y[l],d+1),self._build(X[~l],y[~l],d+1))
    def fit(self,X,y): self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"DecisionTreeClassifier(max_depth={self.max_depth})"


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Multiple comparison correction procedures (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def bonferroni(p_values, alpha=0.05):
    """Bonferroni correction: reject H₀ if p ≤ α/m."""
    m = len(p_values)
    return np.array(p_values) <= (alpha / m)

def holm_sidak(p_values, alpha=0.05):
    """
    Holm stepdown procedure — more powerful than Bonferroni.
    Sort p-values, compare each against α/(m−rank+1).
    Stop rejecting at first non-rejection.
    """
    m      = len(p_values)
    order  = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    rejected = np.zeros(m, dtype=bool)
    for i, p in enumerate(sorted_p):
        threshold = alpha / (m - i)
        if p <= threshold:
            rejected[order[i]] = True
        else:
            break   # stepdown: stop at first non-rejection
    return rejected

def benjamini_hochberg(p_values, alpha=0.05):
    """
    BH procedure — controls FDR ≤ alpha.
    Reject all H₀ with p-value rank ≤ k, where k is the largest
    rank satisfying p₍ₖ₎ ≤ k·α/m.
    """
    m       = len(p_values)
    order   = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    thresholds = np.arange(1, m + 1) * alpha / m
    # Find largest k where p₍ₖ₎ ≤ k*alpha/m
    reject_mask = sorted_p <= thresholds
    if not reject_mask.any():
        return np.zeros(m, dtype=bool)
    last_reject = np.where(reject_mask)[0][-1]
    final_reject = np.zeros(m, dtype=bool)
    final_reject[order[:last_reject + 1]] = True
    return final_reject

# ─────────────────────────────────────────────────────────────────────────────
# Simulation: FWER inflation under H₀
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  MULTIPLE COMPARISONS: BONFERRONI, HOLM, BENJAMINI-HOCHBERG")
print("=" * 65)
print()

n_sims = 2000
alpha  = 0.05

print("  SIMULATION: 2000 experiments, all H₀ TRUE (no real effect)")
print()
print(f"  {'m tests':>8}  {'No correction':>15}  {'Bonferroni':>12}  {'Holm':>8}  {'BH-FDR':>8}")
print("  " + "─" * 57)

for m in [1, 5, 10, 20, 50]:
    fwer_none   = 0; fwer_bonf = 0; fwer_holm = 0; fwer_bh = 0
    for _ in range(n_sims):
        # All H₀ true: p-values uniformly distributed under H₀
        p_vals = np.random.uniform(0, 1, m)
        any_rej_none = (p_vals <= alpha).any()
        any_rej_bonf = bonferroni(p_vals, alpha).any()
        any_rej_holm = holm_sidak(p_vals, alpha).any()
        any_rej_bh   = benjamini_hochberg(p_vals, alpha).any()
        fwer_none += int(any_rej_none)
        fwer_bonf += int(any_rej_bonf)
        fwer_holm += int(any_rej_holm)
        fwer_bh   += int(any_rej_bh)
    print(f"  {m:>8}  {fwer_none/n_sims:>15.3f}  {fwer_bonf/n_sims:>12.3f}  "
          f"{fwer_holm/n_sims:>8.3f}  {fwer_bh/n_sims:>8.3f}")

print()
print("  Without correction: FWER grows to ~92% at m=50 (catastrophic!).")
print("  Bonferroni/Holm: FWER ≤ α = 0.05 for all m.")
print("  BH: controls FDR (expected fraction of false discoveries) not FWER.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Real experiment: compare 12 hyperparameter settings against a baseline
# ─────────────────────────────────────────────────────────────────────────────

print("  REAL EXPERIMENT: 8 HYPERPARAMETER SETTINGS vs BASELINE (LR)")
print()

X, y = make_classification(n_samples=500, n_features=20, n_informative=8,
                            n_redundant=5, random_state=0)
sc   = StandardScaler()

def cv_scores(clf, X, y, k=5):
    """Return per-fold accuracy scores."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    sc_inner = StandardScaler()
    scores   = []
    for tr, te in skf.split(X, y):
        X_tr2 = sc_inner.fit_transform(X[tr])
        X_te2 = sc_inner.transform(X[te])
        clf.fit(X_tr2, y[tr])
        scores.append(clf.score(X_te2, y[te]))
    return np.array(scores)

baseline_clf = LogisticRegression(C=1.0, max_iter=500)
baseline_scores = cv_scores(baseline_clf, X, y, k=5)
baseline_mean   = baseline_scores.mean()

# 8 alternatives: some real improvements (GBM), some trivial (LR variants)
alternatives = {
    "LR C=0.01":    LogisticRegression(C=0.01,  max_iter=500),
    "LR C=0.1":     LogisticRegression(C=0.1,   max_iter=500),
    "LR C=1.1":     LogisticRegression(C=1.1,   max_iter=500),
    "LR C=2.0":     LogisticRegression(C=2.0,   max_iter=500),
    "LR C=10.0":    LogisticRegression(C=10.0,  max_iter=500),
    "GBM 20est":    GradientBoostingClassifier(n_estimators=20, random_state=0),
    "GBM 30est":    GradientBoostingClassifier(n_estimators=30, random_state=0),
    "GBM d=5":      GradientBoostingClassifier(n_estimators=20, max_depth=5, random_state=0),
}

raw_p_vals = []
model_diffs  = []
model_names  = []

print(f"  Baseline (LR C=1.0): mean accuracy = {baseline_mean:.4f}")
print()
print(f"  {'Model':>14}  {'Mean acc':>10}  {'Diff':>8}  {'p-value (t)':>13}  {'Reject?':>8}")
print("  " + "─" * 58)

for alt_name, alt_clf in alternatives.items():
    alt_scores = cv_scores(alt_clf, X, y, k=5)
    diffs_fold = alt_scores - baseline_scores
    d_bar   = diffs_fold.mean()
    t_res   = stats.ttest_rel(alt_scores, baseline_scores)
    raw_p_vals.append(t_res.pvalue)
    model_diffs.append(d_bar)
    model_names.append(alt_name)
    sig = "*" if t_res.pvalue < 0.05 else ""
    print(f"  {alt_name:>14}  {alt_scores.mean():>10.4f}  {d_bar:>+8.4f}  {t_res.pvalue:>13.6f}  {sig:>8}")

raw_p_vals   = np.array(raw_p_vals)
model_diffs  = np.array(model_diffs)
m_tests      = len(raw_p_vals)

print()
print(f"  {(raw_p_vals < 0.05).sum()} significant WITHOUT correction (α=0.05)")

# Apply corrections
bonf_rej = bonferroni(raw_p_vals, alpha)
holm_rej = holm_sidak(raw_p_vals, alpha)
bh_rej   = benjamini_hochberg(raw_p_vals, alpha)

print()
print("  AFTER MULTIPLE COMPARISON CORRECTION:")
print(f"  {'Method':>22}  {'# rejected':>12}  {'Models passing correction'}")
print("  " + "─" * 60)

for meth_name, rej in [("No correction",          raw_p_vals < 0.05),
                         ("Bonferroni (α/m)",        bonf_rej),
                         ("Holm stepdown",            holm_rej),
                         ("BH-FDR (α=0.05)",          bh_rej)]:
    passing = [model_names[i] for i in range(m_tests) if rej[i]]
    print(f"  {meth_name:>22}  {rej.sum():>12}  {', '.join(passing) if passing else 'none'}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Multiple Comparisons: Correction Methods for ML Model Selection",
             fontsize=13, fontweight="bold")

# Panel (0,0): FWER inflation simulation (visualise false positive rate)
ax = axes[0, 0]
m_range  = np.arange(1, 51)
fwer_no_corr   = 1 - (1 - alpha) ** m_range
fwer_bonferroni = np.full_like(m_range, alpha, dtype=float)
ax.plot(m_range, fwer_no_corr, "tomato",    lw=2.5, label="No correction")
ax.plot(m_range, fwer_bonferroni, "steelblue", lw=2.5, ls="--", label="Bonferroni (≤ α)")
ax.axhline(alpha, color="black", lw=1.5, ls=":", label=f"α={alpha}")
ax.fill_between(m_range, alpha, fwer_no_corr, alpha=0.15, color="tomato",
                label="Inflated false positives")
ax.set_xlabel("Number of hypotheses tested m")
ax.set_ylabel("P(at least one false positive)")
ax.set_title("FWER Inflation Without Correction\\n(grows rapidly with m)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (0,1): p-values for 12 model comparisons
ax = axes[0, 1]
sort_idx = np.argsort(raw_p_vals)
x_pos    = np.arange(m_tests)
short_names_sorted = [model_names[i] for i in sort_idx]
p_sorted = raw_p_vals[sort_idx]
col_bar = ["seagreen" if bh_rej[sort_idx[i]] else
           "steelblue" if bonf_rej[sort_idx[i]] else
           "orange" if raw_p_vals[sort_idx[i]] < 0.05 else "gray"
           for i in range(m_tests)]
ax.bar(x_pos, -np.log10(p_sorted + 1e-15), color=col_bar, alpha=0.8)
ax.axhline(-np.log10(alpha), color="black", lw=2, ls="--", label=f"α={alpha} (uncorrected)")
ax.axhline(-np.log10(alpha / m_tests), color="tomato", lw=2, ls=":",
           label=f"Bonferroni α/m={alpha/m_tests:.4f}")
bh_thresh = (np.arange(1, m_tests+1) * alpha / m_tests)
# BH threshold for sorted p-values
ax.step(x_pos, -np.log10(bh_thresh + 1e-15), color="seagreen", lw=2,
        label="BH threshold k·α/m")
ax.set_xticks(x_pos); ax.set_xticklabels(short_names_sorted, rotation=45,
                                           ha="right", fontsize=7)
ax.set_ylabel("−log₁₀(p-value)")
ax.set_title("p-Values: 12 Models vs Baseline\\n(green=BH, blue=Bonferroni, orange=uncorr only)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# Panel (0,2): Correction comparison — which models survive?
ax = axes[0, 2]
correction_matrix = np.column_stack([
    raw_p_vals < 0.05, bonf_rej, holm_rej, bh_rej
])
correction_labels = ["None", "Bonferroni", "Holm", "BH-FDR"]
im = ax.imshow(correction_matrix.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(m_tests)); ax.set_xticklabels(model_names, rotation=45,
                                                    ha="right", fontsize=7)
ax.set_yticks(range(4)); ax.set_yticklabels(correction_labels, fontsize=9)
for i in range(4):
    for j in range(m_tests):
        ax.text(j, i, "✓" if correction_matrix[j, i] else "✗",
                ha="center", va="center", fontsize=10)
ax.set_title("Rejection Matrix: Which Models Pass?\\n(✓=significant, ✗=not significant)",
             fontweight="bold")
plt.colorbar(im, ax=ax)

# Panel (1,0): BH procedure step function
ax = axes[1, 0]
bh_threshs = np.arange(1, m_tests + 1) * alpha / m_tests
ax.scatter(range(1, m_tests+1), p_sorted, color="steelblue", s=60, zorder=5,
           label="Sorted p-values")
ax.step(range(1, m_tests+1), bh_threshs, color="tomato", lw=2.5,
        label="BH threshold k·α/m", where="mid")
ax.axhline(alpha, color="black", lw=1.5, ls="--", label=f"α={alpha}")
ax.axhline(alpha / m_tests, color="gray", lw=1.5, ls=":",
           label=f"Bonferroni α/m={alpha/m_tests:.4f}")
# Mark BH rejections
bh_count = bh_rej.sum()
if bh_count > 0:
    ax.scatter(range(1, bh_count+1), p_sorted[:bh_count], color="seagreen",
               s=100, zorder=6, label="BH rejected")
ax.set_xlabel("Rank i"); ax.set_ylabel("p-value")
ax.set_title("Benjamini-Hochberg Procedure\\n(reject all p₍ᵢ₎ ≤ i·α/m up to last crossing)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,1): Power vs FWER — theoretical tradeoff
ax = axes[1, 1]
# Simulate: m=10 tests, 5 truly different, 5 null
n_reps_power = 1000
true_effects = [0] * 5 + [0.05] * 5   # 0 = null, 0.05 = small true effect
n_per_fold   = 200   # each test uses n_per_fold observations

power_none = np.zeros(10); fwer_none_2 = 0
power_bonf = np.zeros(10); fwer_bonf_2 = 0
power_holm = np.zeros(10); fwer_holm_2 = 0
power_bh   = np.zeros(10); fwer_bh_2   = 0

rng_power = np.random.default_rng(42)
for _ in range(n_reps_power):
    p_vals_sim = []
    for eff in true_effects:
        # Simulate paired differences with effect = eff
        diffs_sim = rng_power.normal(eff, 0.1, 20)
        t_s, p_s  = stats.ttest_1samp(diffs_sim, 0)
        p_vals_sim.append(p_s)
    p_vals_sim = np.array(p_vals_sim)

    rej_none = p_vals_sim < 0.05
    rej_bonf = bonferroni(p_vals_sim, 0.05)
    rej_holm = holm_sidak(p_vals_sim, 0.05)
    rej_bh   = benjamini_hochberg(p_vals_sim, 0.05)

    for i in range(10):
        power_none[i] += int(rej_none[i]); power_bonf[i] += int(rej_bonf[i])
        power_holm[i] += int(rej_holm[i]); power_bh[i]   += int(rej_bh[i])

    # FWER: any false positive among nulls (i=0..4)?
    fwer_none_2 += int(rej_none[:5].any()); fwer_bonf_2 += int(rej_bonf[:5].any())
    fwer_holm_2 += int(rej_holm[:5].any()); fwer_bh_2   += int(rej_bh[:5].any())

power_none /= n_reps_power; power_bonf /= n_reps_power
power_holm /= n_reps_power; power_bh   /= n_reps_power

x_tests = np.arange(10)
ax.plot(x_tests[:5],  power_none[:5],  "o", color="gray",      ms=10, label="None (nulls)")
ax.plot(x_tests[5:],  power_none[5:],  "o", color="steelblue", ms=10, label="None (true eff.)")
ax.plot(x_tests[:5],  power_bonf[:5],  "^", color="lightcoral", ms=10, label="Bonferroni (nulls)")
ax.plot(x_tests[5:],  power_bonf[5:],  "^", color="tomato",    ms=10, label="Bonferroni (true eff.)")
ax.plot(x_tests[5:],  power_bh[5:],    "s", color="seagreen",  ms=10, label="BH (true eff.)")
ax.axvline(4.5, color="black", lw=1.5, ls="--", label="Null | Alternative")
ax.set_xlabel("Test index (0-4: null, 5-9: true effect)")
ax.set_ylabel("Power (true alt.) / FWER rate (null)")
ax.set_title("Power vs Error Control\\n(BH retains more power on true effects than Bonferroni)",
             fontweight="bold")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

# Panel (1,2): Summary table
ax = axes[1, 2]
ax.axis("off")
rows_s = [
    ["Bonferroni",      "α/m",       "FWER ≤ α",  "Very conservative",    "Few true discoveries"],
    ["Holm stepdown",   "adaptive",  "FWER ≤ α",  "Less conservative",    "More power than Bonf."],
    ["BH",              "k·α/m",     "FDR ≤ α",   "Permissive",           "More power, some FP ok"],
    ["No correction",   "α",         "None",       "Anti-conservative",    "Many false positives!"],
    ["5×2cv test",      "n/a",       "FWER ≤ α",  "Design-based",         "Two-model comparison only"],
]
headers_s = ["Method", "Threshold", "Controls", "Conservatism", "Notes"]
table = ax.table(cellText=rows_s, colLabels=headers_s, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.05, 2.2)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif r == 4:
        cell.set_facecolor("#ffe8e8")
ax.set_title("Multiple Comparison Methods Summary", fontweight="bold")

plt.tight_layout()
plt.savefig("multiple_comparisons.png", dpi=110)
print("  Plot saved → multiple_comparisons.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · The 5×2cv Test and Wilcoxon Signed-Rank Test": {
        "description": (
            "Implements the 5×2cv paired test (Dietterich 1998) from scratch "
            "and the Wilcoxon signed-rank test. Runs the 5×2cv test on three "
            "classifier pairs — a genuinely different pair, a similar pair, and "
            "an identical pair — and compares its p-values against the standard "
            "paired t-test and the corrected Nadeau-Bengio t-test. Shows that "
            "the 5×2cv test has better type I error control. Also demonstrates "
            "the Wilcoxon test on the same folds. Summarises all tests in a "
            "comprehensive comparison table."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,random_state=None,**kw):
    rng=_np_impl.random.default_rng(random_state)
    y=rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=n_redundant
    Xr=(Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else _np_impl.empty((n_samples,0)))
    nn2=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    parts=[a for a in [Xi,Xr,Xn] if a.shape[1]>0]
    return _np_impl.hstack(parts)[:,:n_features],y.astype(int)

def train_test_split(*arrays,test_size=0.4,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=len(arrays[0])
    n_tr=int(n*(1-test_size)); idx=rng.permutation(n); ti,vi=idx[:n_tr],idx[n_tr:]
    out=[]
    for a in arrays: out+=[a[ti],a[vi]]
    return out

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y):
        y=_np_impl.asarray(y); classes=_np_impl.unique(y)
        class_idx={c:_np_impl.where(y==c)[0] for c in classes}
        if self.shuffle:
            rng=_np_impl.random.default_rng(self.random_state)
            for c in classes: rng.shuffle(class_idx[c])
        fold_idx=[[] for _ in range(self.n_splits)]
        for c in classes:
            for j,i in enumerate(class_idx[c]):
                fold_idx[j%self.n_splits].append(i)
        for k in range(self.n_splits):
            val=_np_impl.array(fold_idx[k])
            train=_np_impl.concatenate([fold_idx[j] for j in range(self.n_splits) if j!=k])
            yield train.astype(int), val.astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None,**kw):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict(self,X): return (_sig(X@self.coef_.ravel()+self.intercept_[0])>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"LogisticRegression(C={self.C})"

class _DTR:
    def __init__(self,max_depth=3,min_s=5): self.max_depth=max_depth; self.min_s=min_s
    def _split(self,X,r):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>10: ts=ts[_np_impl.linspace(0,len(ts)-1,10).astype(int)]
            for t in ts:
                l=X[:,f]<=t; rr=~l
                if l.sum()<self.min_s or rr.sum()<self.min_s: continue
                g=_np_impl.var(r)*len(r)-_np_impl.var(r[l])*l.sum()-_np_impl.var(r[rr])*rr.sum()
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np_impl.var(r)<1e-12: return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class GradientBoostingClassifier:
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=min(n_estimators,30); self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        self._classes=_np_impl.unique(y)
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return self._classes[(_sig(F)>=0.5).astype(int)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"GradientBoostingClassifier(n_estimators={self.n_estimators})"

class _DTC:
    def __init__(self,max_depth=8,max_features=None,min_s=2):
        self.max_depth=max_depth; self.max_features=max_features; self.min_s=min_s
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y,rng):
        d=X.shape[1]
        feats=(rng.choice(d,min(self.max_features,d),replace=False)
               if self.max_features else _np_impl.arange(d))
        best=None
        for f in feats:
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>8: ts=ts[_np_impl.linspace(0,len(ts)-1,8).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_s or r.sum()<self.min_s: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,d,rng):
        cls,cnt=_np_impl.unique(y,return_counts=True)
        prob=_np_impl.zeros(len(self._classes))
        for c,n in zip(cls,cnt): prob[_np_impl.searchsorted(self._classes,c)]=n/len(y)
        if d>=self.max_depth or len(_np_impl.unique(y))==1 or len(y)<=self.min_s: return ('L',prob)
        sp=self._best(X,y,rng)
        if sp is None: return ('L',prob)
        _,f,t=sp; l=X[:,f]<=t
        return ('N',f,t,self._build(X[l],y[l],d+1,rng),self._build(X[~l],y[~l],d+1,rng))
    def fit(self,X,y,classes,rng):
        self._classes=classes; self._tree=self._build(X,y,0,rng); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict_proba(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=100,max_depth=None,random_state=None,**kw):
        self.n_estimators=min(n_estimators,20); self.max_depth=max_depth or 8
        self.random_state=random_state
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); rng=_np_impl.random.default_rng(self.random_state)
        n,d=X.shape; mf=max(1,int(_np_impl.sqrt(d))); self._trees=[]
        for _ in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); t=_DTC(max_depth=self.max_depth,max_features=mf)
            t.fit(X[idx],y[idx],self._classes,rng); self._trees.append(t)
        return self
    def predict(self,X):
        proba=_np_impl.mean([t.predict_proba(X) for t in self._trees],axis=0)
        return self._classes[proba.argmax(1)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"RandomForestClassifier(n_estimators={self.n_estimators})"

class DecisionTreeClassifier:
    def __init__(self,max_depth=None,random_state=None,**kw):
        self.max_depth=max_depth or 999; self.random_state=random_state
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>15: ts=ts[_np_impl.linspace(0,len(ts)-1,15).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<2 or r.sum()<2: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,d):
        cls,cnt=_np_impl.unique(y,return_counts=True); pred=cls[cnt.argmax()]
        if d>=self.max_depth or len(_np_impl.unique(y))==1 or len(y)<=2: return ('L',pred)
        sp=self._best(X,y)
        if sp is None: return ('L',pred)
        _,f,t=sp; l=X[:,f]<=t
        return ('N',f,t,self._build(X[l],y[l],d+1),self._build(X[~l],y[~l],d+1))
    def fit(self,X,y): self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)
    def __repr__(self): return f"DecisionTreeClassifier(max_depth={self.max_depth})"


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 5×2cv paired test (Dietterich 1998)
# ─────────────────────────────────────────────────────────────────────────────

def five_by_two_cv(clf_A, clf_B, X, y, seed=0):
    """
    5×2cv paired t-test (Dietterich 1998).
    Returns: t_stat, p_val, and per-run differences.
    """
    rng    = np.random.default_rng(seed)
    n      = len(X)
    p1_run = np.zeros(5)   # difference on fold 1 of each run
    s2_run = np.zeros(5)   # per-run variance estimate

    for i in range(5):
        # Fresh random 50/50 split
        idx  = rng.permutation(n)
        half = n // 2
        idx1, idx2 = idx[:half], idx[half:]

        # Fit on each half, evaluate on the other
        sc = StandardScaler()
        p1_A = []; p1_B = []; p2_A = []; p2_B = []
        for tr_idx, te_idx in [(idx1, idx2), (idx2, idx1)]:
            X_tr = sc.fit_transform(X[tr_idx]); X_te = sc.transform(X[te_idx])
            y_tr = y[tr_idx];                   y_te = y[te_idx]
            clf_A.fit(X_tr, y_tr); clf_B.fit(X_tr, y_tr)
            acc_A = clf_A.score(X_te, y_te)
            acc_B = clf_B.score(X_te, y_te)
            if len(p1_A) == 0:
                p1_A.append(acc_A); p1_B.append(acc_B)
            else:
                p2_A.append(acc_A); p2_B.append(acc_B)

        d1 = p1_A[0] - p1_B[0]   # diff on fold 1
        d2 = p2_A[0] - p2_B[0]   # diff on fold 2
        p_bar = (d1 + d2) / 2
        s2_run[i] = (d1 - p_bar) ** 2 + (d2 - p_bar) ** 2
        p1_run[i] = d1

    t_stat = p1_run[0] / np.sqrt((1/5) * s2_run.sum() + 1e-15)
    p_val  = 2 * stats.t.sf(abs(t_stat), df=5)
    return t_stat, p_val, p1_run, s2_run

# ─────────────────────────────────────────────────────────────────────────────
# Paired t-test utilities
# ─────────────────────────────────────────────────────────────────────────────

def paired_t_test(diffs, n_train=None, n_test=None, corrected=False):
    k     = len(diffs)
    d_bar = diffs.mean()
    s_d   = diffs.std(ddof=1)
    if corrected and n_train is not None:
        rho = n_test / n_train
        se  = np.sqrt((1/k + rho) * s_d**2)
    else:
        se = s_d / np.sqrt(k)
    t_stat = d_bar / (se + 1e-15)
    p_val  = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    ci_hw  = stats.t.ppf(0.975, df=k-1) * se
    return {"d_bar": d_bar, "t_stat": t_stat, "p_val": p_val,
            "ci_lo": d_bar - ci_hw, "ci_hi": d_bar + ci_hw}

def wilcoxon_test(diffs):
    """Wilcoxon signed-rank test on per-fold differences."""
    stat, p_val = stats.wilcoxon(diffs, alternative="two-sided")
    return stat, p_val

# ─────────────────────────────────────────────────────────────────────────────
# Experiment setup
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  5×2cv AND WILCOXON SIGNED-RANK TEST")
print("=" * 65)
print()

X, y = make_classification(n_samples=400, n_features=20, n_informative=8,
                            n_redundant=5, random_state=0)

pairs = [
    ("LR vs GBM (genuinely different)",
     LogisticRegression(C=1.0, max_iter=500),
     GradientBoostingClassifier(n_estimators=20, random_state=0)),
    ("LR vs RF (similar performance)",
     LogisticRegression(C=1.0, max_iter=500),
     RandomForestClassifier(n_estimators=20, random_state=0)),
    ("LR(C=1.0) vs LR(C=1.001) (identical)",
     LogisticRegression(C=1.0,   max_iter=500),
     LogisticRegression(C=1.001, max_iter=500)),
]

# Get 5-fold CV per-fold scores for all models
k_folds = 5
skf     = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)

fold_results = {}
n_tr_avg = 0; n_te_avg = 0
for clf_A, clf_B in [(p[1], p[2]) for p in pairs]:
    for clf in [clf_A, clf_B]:
        key = str(clf)[:30]
        if key not in fold_results:
            fold_results[key] = []
            for tr_idx, te_idx in skf.split(X, y):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X[tr_idx]); X_te = sc.transform(X[te_idx])
                y_tr = y[tr_idx];                   y_te = y[te_idx]
                clf.fit(X_tr, y_tr)
                fold_results[key].append(clf.score(X_te, y_te))
                n_tr_avg = len(tr_idx); n_te_avg = len(te_idx)
            fold_results[key] = np.array(fold_results[key])

all_test_results = []

for pair_name, clf_A, clf_B in pairs:
    key_A = str(clf_A)[:30]
    key_B = str(clf_B)[:30]
    cv_A  = fold_results[key_A]
    cv_B  = fold_results[key_B]
    diffs = cv_A - cv_B

    # 1. Standard paired t-test
    t_std = paired_t_test(diffs, corrected=False)
    # 2. Corrected paired t-test
    t_cor = paired_t_test(diffs, n_train=n_tr_avg, n_test=n_te_avg, corrected=True)
    # 3. 5×2cv test
    t52_stat, t52_p, p1_runs, s2_runs = five_by_two_cv(
        clf_A, clf_B, X, y, seed=0)
    # 4. Wilcoxon
    w_stat, w_p = wilcoxon_test(diffs) if len(diffs[diffs != 0]) > 0 else (np.nan, 1.0)

    all_test_results.append({
        "pair_name": pair_name,
        "mean_diff": diffs.mean(),
        "cv_A": cv_A, "cv_B": cv_B, "diffs": diffs,
        "t_std_p": t_std["p_val"], "t_cor_p": t_cor["p_val"],
        "t52_p": t52_p, "t52_stat": t52_stat,
        "w_p": w_p, "w_stat": w_stat,
        "p1_runs": p1_runs, "s2_runs": s2_runs,
    })

    def sig_star(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    print(f"  {pair_name}")
    print(f"  {'─' * 55}")
    print(f"  5-fold CV: mean(A)={cv_A.mean():.4f}  mean(B)={cv_B.mean():.4f}  "
          f"diff={diffs.mean():+.4f}")
    print()
    print(f"  {'Test':>30}  {'Statistic':>12}  {'p-value':>12}  {'Sig':>6}")
    print(f"  {'─' * 65}")
    print(f"  {'Standard paired t-test':>30}  {t_std['t_stat']:>12.4f}  "
          f"{t_std['p_val']:>12.6f}  {sig_star(t_std['p_val']):>6}")
    print(f"  {'Corrected t (Nadeau-Bengio)':>30}  {t_cor['t_stat']:>12.4f}  "
          f"{t_cor['p_val']:>12.6f}  {sig_star(t_cor['p_val']):>6}")
    print(f"  {'5×2cv paired t-test':>30}  {t52_stat:>12.4f}  "
          f"{t52_p:>12.6f}  {sig_star(t52_p):>6}")
    print(f"  {'Wilcoxon signed-rank':>30}  {w_stat:>12.2f}  "
          f"{w_p:>12.6f}  {sig_star(w_p):>6}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Type I error simulation: all H₀ true (identical models)
# ─────────────────────────────────────────────────────────────────────────────

print("  TYPE I ERROR RATE SIMULATION (H₀ TRUE, 50 repetitions):")
print("  (Both classifiers are identical LR C=1.0 — any rejection is false positive)")
print()

n_type1 = 50
alpha   = 0.05
t_std_fp = 0; t_cor_fp = 0; t52_fp = 0; w_fp = 0

rng_t1 = np.random.default_rng(99)
for rep in range(n_type1):
    # Same model, different random seed → should give p > 0.05
    X_r, y_r = make_classification(n_samples=200, n_features=10, random_state=rep)
    clf_same  = LogisticRegression(C=1.0, max_iter=500)
    skf_r     = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
    fold_s_A = []; fold_s_B = []
    nt_r = 0; nte_r = 0
    for tr_idx, te_idx in skf_r.split(X_r, y_r):
        sc_r = StandardScaler()
        X_tr_r = sc_r.fit_transform(X_r[tr_idx]); X_te_r = sc_r.transform(X_r[te_idx])
        y_tr_r = y_r[tr_idx];                     y_te_r = y_r[te_idx]
        clf_same.fit(X_tr_r, y_tr_r)
        acc  = clf_same.score(X_te_r, y_te_r)
        # Add small independent noise to simulate two "different" runs of same model
        noise_A = rng_t1.normal(0, 0.01)
        noise_B = rng_t1.normal(0, 0.01)
        fold_s_A.append(min(1, max(0, acc + noise_A)))
        fold_s_B.append(min(1, max(0, acc + noise_B)))
        nt_r = len(tr_idx); nte_r = len(te_idx)
    fold_s_A = np.array(fold_s_A); fold_s_B = np.array(fold_s_B)
    diffs_r  = fold_s_A - fold_s_B
    t_std_fp += int(paired_t_test(diffs_r)["p_val"] < alpha)
    t_cor_fp += int(paired_t_test(diffs_r, nt_r, nte_r, True)["p_val"] < alpha)
    _, t52_p_r, _, _ = five_by_two_cv(clf_same, clf_same, X_r, y_r, seed=rep)
    t52_fp += int(t52_p_r < alpha)
    if (diffs_r != 0).sum() >= 1:
        _, w_p_r = stats.wilcoxon(diffs_r)
        w_fp   += int(w_p_r < alpha)

print(f"  {'Test':>30}  {'FP rate':>10}  {'Target':>8}")
print("  " + "─" * 52)
print(f"  {'Standard paired t-test':>30}  {t_std_fp/n_type1:>10.3f}  {alpha:>8.2f}")
print(f"  {'Corrected t (Nadeau-Bengio)':>30}  {t_cor_fp/n_type1:>10.3f}  {alpha:>8.2f}")
print(f"  {'5×2cv paired t-test':>30}  {t52_fp/n_type1:>10.3f}  {alpha:>8.2f}")
print(f"  {'Wilcoxon signed-rank':>30}  {w_fp/n_type1:>10.3f}  {alpha:>8.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("5×2cv and Wilcoxon Test: Comparison of Statistical Tests for ML Models",
             fontsize=13, fontweight="bold")

# Panel (0,0): Per-fold differences for all three pairs
ax = axes[0, 0]
folds_x = np.arange(1, k_folds + 1)
for res, col in zip(all_test_results, ["steelblue", "seagreen", "tomato"]):
    label = res["pair_name"].split("(")[0].strip()
    ax.plot(folds_x, res["diffs"], "o-", color=col, lw=2, ms=8, label=label)
ax.axhline(0, color="black", lw=1.5, ls="--")
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy difference (A−B)")
ax.set_title("Per-Fold Differences: Three Classifier Pairs\\n(positive = A better)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xticks(folds_x)

# Panel (0,1): 5×2cv per-run differences (first pair: LR vs GBM)
ax = axes[0, 1]
res0 = all_test_results[0]
runs_x = np.arange(1, 6)
ax.bar(runs_x - 0.15, res0["p1_runs"], 0.3, color="steelblue", alpha=0.8,
       label="p₁(i) = diff, fold 1 of run i")
ax.bar(runs_x + 0.15, res0["s2_runs"], 0.3, color="tomato", alpha=0.8,
       label="s²(i) = within-run variance")
ax.axhline(0, color="black", lw=1.5)
ax.set_xlabel("5×2cv Run i")
ax.set_ylabel("Difference / Variance estimate")
ax.set_title(f"5×2cv Components: LR vs GBM\\n"
             f"t={res0['t52_stat']:.3f}  p={res0['t52_p']:.5f}",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xticks(runs_x)

# Panel (0,2): p-value comparison across pairs and tests
ax = axes[0, 2]
test_labels_short = ["Std t", "Corr t", "5×2cv", "Wilcoxon"]
x_pos = np.arange(3)
w_bar = 0.2
colours_t = ["steelblue", "seagreen", "tomato", "purple"]
for ti, (tname, key) in enumerate([("Std t", "t_std_p"), ("Corr t", "t_cor_p"),
                                     ("5×2cv", "t52_p"),  ("Wilcoxon", "w_p")]):
    p_vals_t = [res[key] for res in all_test_results]
    ax.bar(x_pos + (ti - 1.5) * w_bar, p_vals_t, w_bar,
           color=colours_t[ti], alpha=0.8, label=tname)
ax.axhline(0.05, color="black", lw=2, ls="--", label="α=0.05")
pair_short = [r["pair_name"].split("(")[0].strip()[:15] for r in all_test_results]
ax.set_xticks(x_pos); ax.set_xticklabels(pair_short, fontsize=8, rotation=10)
ax.set_ylabel("p-value"); ax.set_ylim(0, 1.0)
ax.set_title("p-Values Across Tests and Pairs\\n(below 0.05 = significant)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# Panel (1,0): Type I error rates bar chart
ax = axes[1, 0]
tests    = ["Standard t", "Corrected t", "5×2cv", "Wilcoxon"]
fp_rates = [t_std_fp/n_type1, t_cor_fp/n_type1, t52_fp/n_type1, w_fp/n_type1]
col_bars = ["tomato" if r > alpha else "seagreen" for r in fp_rates]
ax.bar(tests, fp_rates, color=col_bars, alpha=0.8)
ax.axhline(alpha, color="black", lw=2, ls="--", label=f"Target α={alpha}")
ax.axhline(alpha * 1.2, color="gray", lw=1, ls=":", alpha=0.7,
           label="20% buffer")
ax.set_ylabel("Empirical Type I error rate")
ax.set_title(f"Type I Error (H₀ True, n={n_type1} sims)\\n"
             f"(green = controlled at α, red = inflated)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for i, r in enumerate(fp_rates):
    ax.text(i, r + 0.005, f"{r:.3f}", ha="center", fontsize=10)

# Panel (1,1): 5×2cv variance pooling illustration
ax = axes[1, 1]
runs_v = np.arange(1, 6)
res_v  = all_test_results[0]
ax.bar(runs_v, res_v["s2_runs"], color="steelblue", alpha=0.8, label="s²(i) per run")
ax.axhline(res_v["s2_runs"].mean(), color="tomato", lw=2.5, ls="--",
           label=f"Mean s²={res_v['s2_runs'].mean():.6f}")
ax.set_xlabel("Run i"); ax.set_ylabel("s²(i) = within-run variance estimate")
ax.set_title("5×2cv Variance Pooling (LR vs GBM)\\n"
             "t = p₁(1) / √(mean(s²)) ~ t(5)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xticks(runs_v)

# Panel (1,2): Summary recommendation table
ax = axes[1, 2]
ax.axis("off")
rows_r = [
    ["Paired t-test (std)",   "k-fold CV",          "anti-conservative",  "→ too many false pos."],
    ["Paired t-test (NB)",    "k-fold CV",           "conservative",       "good default"],
    ["5×2cv paired t",        "5 fresh splits",      "controlled",         "best type-I control"],
    ["Wilcoxon",              "k-fold CV",            "moderate",           "non-parametric"],
    ["McNemar's test",        "single test set",      "controlled",         "paired binary outcomes"],
    ["Bootstrap CI",          "single test set",      "controlled",         "for single model CI"],
]
headers_r = ["Test", "Evaluation design", "Type I error", "Notes"]
table = ax.table(cellText=rows_r, colLabels=headers_r, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.05, 2.0)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif r == 3:
        cell.set_facecolor("#e8f5e8")
ax.set_title("Test Selection Guide", fontweight="bold")

plt.tight_layout()
plt.savefig("five_by_two_cv.png", dpi=110)
print("  Plot saved → five_by_two_cv.png")
print()
print("  KEY TAKEAWAYS — STATISTICAL TESTING IN ML:")
print("  1. Test set accuracy is an estimate with uncertainty — always report CIs.")
print("  2. McNemar's test for same test-set comparisons — uses discordant pairs.")
print("  3. Paired t-test on k-fold CV is anti-conservative (apply NB correction).")
print("  4. 5×2cv test has the best type I error control for paired comparisons.")
print("  5. Wilcoxon signed-rank is non-parametric — preferred for small k or outliers.")
print("  6. Multiple testing inflates false positives — always apply Bonferroni/Holm/BH.")
print("  7. Statistical significance ≠ practical significance. Report effect size.")
print("  8. Never use the same data for model selection AND reporting final accuracy.")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent all code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
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