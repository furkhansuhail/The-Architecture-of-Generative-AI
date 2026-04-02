"""
Sampling Theory
===============

How to draw conclusions about a population from a subset of it —
the mathematical foundation beneath every statistic, ML metric,
A/B test, and confidence interval you will ever compute.

"""

import textwrap
import re

TOPIC_NAME = "Sampling Theory"
DISPLAY_NAME = "03 · Sampling Theory"
ICON = "🎲"
SUBTITLE = "From Populations to Estimates — the Math Behind Every Statistic"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """


### PART 1 — THE CORE FRAMEWORK: POPULATION, SAMPLE, ESTIMATOR


### Population vs Sample

A population is the complete set of all objects you care about.
A sample is a subset you actually observe. Almost everything in
statistics, ML evaluation, and A/B testing involves reasoning
about a population from a sample.

    Population (all N units, usually unknown or infinite):
    ┌───────────────────────────────────────────────────────────┐
    │  μ  = population mean          (Greek = fixed but unknown)│
    │  σ² = population variance                                 │
    │  Any quantity computed on the full population             │
    │  is called a PARAMETER.                                   │
    └───────────────────────────────────────────────────────────┘
                              ↓  draw n units
    Sample (n units you observe):
    ┌───────────────────────────────────────────────────────────┐
    │  x̄  = sample mean              (Roman = computed, known)  │
    │  s² = sample variance                                     │
    │  Any quantity computed on the sample                      │
    │  is called a STATISTIC (or ESTIMATOR).                    │
    └───────────────────────────────────────────────────────────┘

The goal of inferential statistics: use the statistic (x̄, s², p̂ …)
to make a confident claim about the parameter (μ, σ², p …).

Key notational convention used throughout this module:
    Greek letters  (μ, σ, θ, β)  →  true population parameters
    Roman letters  (x̄, s, θ̂, b)  →  sample-based estimates


──────────────────────────────────────────────────────────────────────────────
### What Makes a Good Estimator?

An estimator θ̂ is a function of the sample data used to approximate a
population parameter θ. Three properties define estimator quality:

**1. Bias** — how far the estimator is from the true value ON AVERAGE:

    Bias(θ̂) = E[θ̂] − θ

    Unbiased: E[θ̂] = θ   →  the estimator is correct on average
    Biased:   E[θ̂] ≠ θ   →  even with infinite data, it drifts

    Classic example: why we divide by (n−1) not n for sample variance.
    The sample mean x̄ is already estimated from the same data, so raw
    deviations are slightly too small. Dividing by (n−1) corrects this:

        s² = Σ(xᵢ − x̄)² / (n−1)    ← unbiased estimate of σ²
        s² = Σ(xᵢ − x̄)² / n        ← biased (underestimates σ²)

**2. Variance** — how much the estimator fluctuates across different samples:

    Var(θ̂) = E[(θ̂ − E[θ̂])²]

    An estimator with high variance is unreliable even if unbiased:
    one sample gives 2.1, another gives 9.8, another gives 0.3 —
    each is close to the truth on average, but any single estimate
    could be far off.

**3. MSE — Mean Squared Error** — the single number that combines both:

    MSE(θ̂) = Var(θ̂) + [Bias(θ̂)]²

    This is the bias-variance trade-off at the level of estimators.
    A slightly biased estimator with much lower variance can have
    smaller MSE than an unbiased but highly variable one.

    Diagram — Bias vs Variance as a shooting analogy:

        Low Bias / Low Variance    Low Bias / High Variance
        ┌───────────────────┐      ┌────────────────────┐
        │       ●           │      │  ●                 │
        │      ●●●          │      │           ●        │
        │       ●           │      │      ●             │
        │   (clustered      │      │  (spread around    │
        │    at target)     │      │   the target)      │
        └───────────────────┘      └────────────────────┘

        High Bias / Low Variance   High Bias / High Variance
        ┌───────────────────┐      ┌────────────────────┐
        │                   │      │ ●                  │
        │      ●●●          │      │           ●        │
        │       ●           │      │  ●         ●       │
        │  (clustered but   │      │  (scattered AND    │
        │   off-target)     │      │   off-target)      │
        └───────────────────┘      └────────────────────┘



### PART 2 — THE SAMPLING DISTRIBUTION & THE CENTRAL LIMIT THEOREM

### The Sampling Distribution

Imagine drawing thousands of independent samples from the same population,
computing x̄ for each. The distribution of those x̄ values is the
SAMPLING DISTRIBUTION of the mean.

    Population (fixed, unknown):   any shape — skewed, bimodal, uniform …
                                   mean = μ,  variance = σ²

    Sample 1:  x₁₁ x₁₂ … x₁ₙ  →  x̄₁
    Sample 2:  x₂₁ x₂₂ … x₂ₙ  →  x̄₂      The distribution of
    Sample 3:  x₃₁ x₃₂ … x₃ₙ  →  x̄₃   ←  all possible x̄ values
    …          …                  …         is the Sampling Distribution.

Two facts about the sampling distribution of x̄ (exact, no assumptions):
    1.  E[x̄] = μ             (sample mean is unbiased for population mean)
    2.  Var(x̄) = σ²/n        (variance SHRINKS as sample size grows)

The quantity √(σ²/n) = σ/√n is called the STANDARD ERROR of the mean.
It quantifies the typical gap between a single sample mean and the truth.
More data → smaller standard error → more precise estimates.


──────────────────────────────────────────────────────────────────────────────
### The Central Limit Theorem (CLT)

The most important result in statistics:

    If x₁, x₂, …, xₙ are i.i.d. with mean μ and variance σ² < ∞,
    then as n → ∞:

        (x̄ − μ) / (σ/√n)  →  N(0, 1)

    Equivalently:   x̄  ~  N(μ,  σ²/n)   approximately, for large n.

Why this is profound: it doesn't matter what shape the population has.
Exponential, Poisson, heavy-tailed, bimodal — the sample mean's distribution
approaches a bell curve as n grows.

    Diagram — CLT in action across population shapes:

    Population distribution → Sampling distribution of x̄ (n=30)

    Uniform [0,1]:
    ████████████████  →  ░░░▒▒▓▓████▓▓▒▒░░░   (bell shape)

    Exponential (right-skewed):
    ████░░░           →  ░░░▒▒▓▓████▓▓▒▒░░░   (bell shape)

    Bimodal (two peaks):
    ████  ████        →  ░░░▒▒▓▓████▓▓▒▒░░░   (bell shape)

    In every case the sample mean distribution is approximately normal.

How large must n be? A rough but widely-used guideline:
    • Symmetric populations:   n ≥ 10  is often sufficient
    • Moderately skewed:       n ≥ 30  is the classic rule-of-thumb
    • Heavily skewed/outliers: n ≥ 100+ for reliable normality

The CLT is what justifies using z-scores and t-tests on sample means
regardless of the original data's distribution.



### PART 3 — CONFIDENCE INTERVALS

### What a Confidence Interval Actually Means

A 95% confidence interval is NOT:
    ✗  "There is a 95% probability that μ lies in this interval."

The population mean μ is a fixed (not random) unknown number.
It either is or isn't in the interval — there is no probability about it.

A 95% CI CORRECTLY means:
    ✓  "If we repeated this sampling procedure 100 times, about 95 of
        the resulting intervals would contain μ."

The randomness is in the interval itself (because x̄ is random),
not in the parameter.

    Diagram — 20 repeated experiments, each draws n=30, builds a 95% CI:

    True μ = 50 (vertical line)

    Sample  1:  ├──────────────●──────────────┤   covers μ ✓
    Sample  2:       ├──────────●───────────┤      covers μ ✓
    Sample  3:             ├──────────●──────────────────┤  covers μ ✓
    Sample  4:                 ├────●────┤               covers μ ✓
    Sample  5:   ├──────────────────────────●──────┤     misses μ ✗
    …
    ~95 out of 100 will cover the line.  ~5 will miss it.


──────────────────────────────────────────────────────────────────────────────
### Computing a Confidence Interval for the Mean

Case 1 — σ known (rare in practice) → use Z:

    CI = x̄ ± z* · (σ/√n)

    where z* is the critical value:  90% CI → z*=1.645
                                     95% CI → z*=1.960
                                     99% CI → z*=2.576

Case 2 — σ unknown (the normal case) → use t-distribution:

    CI = x̄ ± t*(df) · (s/√n)     df = n − 1

    The t-distribution has heavier tails than the normal — it accounts
    for the extra uncertainty of estimating σ from the data.
    As n → ∞, t → z (the tails fatten out to normal).

    Why heavier tails? Two sources of randomness:
        • x̄ is random (estimated from data)
        • s is random (also estimated from data)
    Stacking these uncertainties inflates the tails.

Width of a CI is controlled by three levers:
    ┌──────────────────────────────────────────────────────────────┐
    │  Width = 2 · t* · s/√n                                       │
    │                                                              │
    │  Want narrower interval?                                     │
    │    → Increase n  (width ∝ 1/√n — doubles n, shrinks by √2)   │
    │    → Accept lower confidence (99% is always wider than 95%)  │
    │    → Reduce population variance (better experimental design) │
    └──────────────────────────────────────────────────────────────┘



### PART 4 — HYPOTHESIS TESTING

### The Logic of Hypothesis Testing

Hypothesis testing answers: "Is this observed effect real, or explainable
by sampling variation alone?"

The framework:
    H₀ (null hypothesis):  the boring default — no effect, no difference
    H₁ (alternative):      what we want to demonstrate is true

Strategy: assume H₀ is true, then ask how surprising the observed
data would be. If the data are very unlikely under H₀, we reject it.

**The p-value** is the probability of observing data at least as extreme
as what was seen, assuming H₀ is true.

    p-value small  → data are surprising under H₀  → reject H₀
    p-value large  → data are consistent with H₀    → fail to reject H₀

    CRITICAL: p-value is NOT the probability that H₀ is true.
    It is P(data this extreme | H₀ is true) — a conditional probability
    about data, not about the hypothesis.

    Diagram — Two-sided t-test at α = 0.05:

    Distribution of test statistic under H₀:

             reject │          fail to reject           │ reject
    ────────────────┼───────────────────────────────────┼────────────
    … ─2.5  ─2.0  ─1.96         0         1.96  2.0  2.5 …
                    │◄────────── 95% ──────────────────►│
               ◄────│2.5%                           2.5%│────►
              (left tail)                      (right tail)
                   critical values               p < 0.05 → reject


──────────────────────────────────────────────────────────────────────────────
### Type I and Type II Errors

Reality has two states (H₀ true or false) and we make two decisions
(reject or not). This gives four outcomes:

    ┌──────────────────────┬─────────────────────┬────────────────────┐
    │                      │  Decision:          │  Decision:         │
    │                      │  Reject H₀          │  Fail to Reject H₀ │
    ├──────────────────────┼─────────────────────┼────────────────────┤
    │  Reality: H₀ TRUE    │  Type I Error (α)   │  Correct ✓         │
    │                      │  False Positive     │  True Negative     │
    ├──────────────────────┼─────────────────────┼────────────────────┤
    │  Reality: H₀ FALSE   │  Correct ✓          │  Type II Error (β) │
    │                      │  True Positive      │  False Negative    │
    └──────────────────────┴─────────────────────┴────────────────────┘

    α (alpha) = P(Type I Error) = significance level  [typically 0.05]
    β (beta)  = P(Type II Error)
    Power = 1 − β = P(correctly rejecting a false H₀)  [typically ≥ 0.80]

There is an inherent tension: making α smaller (stricter test) always
increases β (more false negatives), and vice versa. The only way to
reduce both simultaneously is to increase the sample size n.


──────────────────────────────────────────────────────────────────────────────
### Statistical Power & Sample Size Planning

Power is determined by four interlocked quantities:

    Power ↑  when:
        • Effect size (Cohen's d) is large — a bigger real difference is
          easier to detect
        • Sample size n is large — less noise, tighter sampling distribution
        • Significance level α is relaxed — easier to exceed the threshold
        • Population variance σ² is small — cleaner signal

    Cohen's d (standardised effect size for two means):

        d = (μ₁ − μ₂) / σ_pooled

        Small effect:  d ≈ 0.2
        Medium effect: d ≈ 0.5
        Large effect:  d ≈ 0.8

    Required sample size per group (two-sample t-test, α=0.05, power=0.80):

        n ≈ 2 · (z_α/2 + z_β)² / d²

        d = 0.2 (small):   n ≈ 394 per group
        d = 0.5 (medium):  n ≈  64 per group
        d = 0.8 (large):   n ≈  26 per group

    Always do power analysis BEFORE collecting data. Running a test and
    then adjusting n to get p < 0.05 is p-hacking — it inflates Type I
    error far above the nominal α.



### PART 5 — THE BOOTSTRAP: RESAMPLING AS A UNIVERSAL TOOL

### The Bootstrap Idea

The CLT gives us CIs for the mean. But what about the median, the 90th
percentile, a correlation, an AUC score, or any complex model metric?
No neat formula exists for their sampling distributions.

The bootstrap solves this by simulating the sampling process using the
observed sample as a stand-in for the unknown population.

**Algorithm (non-parametric bootstrap):**

    Original sample:  [x₁, x₂, x₃, …, xₙ]    (n observations)

    Repeat B times (B ≥ 1 000):
    ┌─────────────────────────────────────────────────────────┐
    │ 1. Draw n observations WITH REPLACEMENT from the sample │
    │    → bootstrap sample b                                 │
    │ 2. Compute the statistic θ̂* on bootstrap sample b       │
    │    → record θ̂*ᵦ                                         │
    └─────────────────────────────────────────────────────────┘

    The distribution of {θ̂*₁, θ̂*₂, …, θ̂*_B}
    approximates the sampling distribution of θ̂.

    Diagram — Bootstrap resampling (n=5, B=3 shown):

    Original:   [3, 7, 2, 9, 5]         → θ̂ = median = 5

    Boot-1:     [7, 7, 2, 3, 9]         → θ̂*₁ = median = 7
    Boot-2:     [3, 5, 9, 9, 2]         → θ̂*₂ = median = 5
    Boot-3:     [9, 2, 2, 5, 7]         → θ̂*₃ = median = 5
    …
    After B=1000 resamples, histogram the θ̂* values → that IS the
    approximate sampling distribution of the median.

**Percentile bootstrap CI** (simplest):
    Sort {θ̂*ᵦ}; the 2.5th and 97.5th percentiles form the 95% CI.

**Why it works:**
    The sample is our best available model of the population.
    Resampling from the sample mimics what it would look like to
    draw new samples from the real population.

**Limitations:**
    • Poor in very small samples (n < 20) — the sample is a bad
      model of the population
    • Does not fix a biased estimator — garbage in, garbage out
    • Computationally intensive but trivially parallelisable



### PART 6 — SAMPLING METHODS

### How You Draw the Sample Shapes Every Downstream Inference

The statistical guarantees of CLT and hypothesis tests assume a
specific sampling process: independent, identically distributed (i.i.d.)
draws from the population. Different sampling designs break or
preserve this in different ways.

**Simple Random Sampling (SRS)**

    Every unit has an equal probability of selection.
    Selections are independent. This is the textbook i.i.d. ideal.

    Population: [●●●●●●●●●●●●●●●●●●●●]  (N=20)
    SRS n=5:    [●     ●   ●       ●  ●]  (5 randomly chosen)

    Pro: simplest theory; no design adjustments needed.
    Con: may miss rare but important subgroups by chance.

**Stratified Random Sampling**

    Divide the population into strata (homogeneous subgroups),
    then draw a separate SRS within each stratum.

    Population stratified by class:
    ┌──────────────────┬────────────────┬────────────────┐
    │ Stratum A (60%)  │ Stratum B (30%)│ Stratum C (10%)│
    │ ●●●●●●●●●●●●     │ ●●●●●●         │ ●●             │
    └──────────────────┴────────────────┴────────────────┘
    Sample proportionally:  6 from A,   3 from B,   1 from C

    Pro: guaranteed representation of all strata; lower variance
         than SRS when strata are internally homogeneous.
    Con: requires knowing stratum membership up front.

    Key formula — stratified estimator of the population mean:
        x̄_st = Σ (Wₕ × x̄ₕ)
        where Wₕ = stratum h's proportion of population, x̄ₕ = stratum mean.

**Cluster Sampling**

    Divide the population into clusters (heterogeneous internally),
    randomly select some clusters, and sample all units within them.

    Example: survey of school students — randomly select 20 classrooms,
    survey everyone in those classrooms.

    Pro: logistically efficient when the population is geographically spread.
    Con: higher variance than SRS for same n (units within a cluster
         tend to be similar — effective sample size is smaller than n).

    Intra-cluster correlation ρ (ICC) quantifies this loss:
        Effective n = n / (1 + (m−1)ρ)
        where m = cluster size.  If ρ=0.2 and m=10: effective n = n/2.8.
        You need nearly 3× as many observations to match SRS precision.

**Systematic Sampling**

    Sort the population, pick every kth unit (k = N/n).
    Random start in [1, k], then deterministically pick every kth unit.

    Population of 20, want n=4:  k = 20/4 = 5
    Random start = 3:  pick units 3, 8, 13, 18

    Pro: operationally simple; approximately equivalent to SRS
         if ordering is unrelated to the variable of interest.
    Con: if the list has a periodic pattern of period k,
         systematic sampling can produce severely biased estimates.

    Diagram — comparison at a glance:
    ┌──────────────────┬────────────────┬─────────────┬─────────────────┐
    │ Method           │ Variance       │ Logistics   │ When to use     │
    ├──────────────────┼────────────────┼─────────────┼─────────────────┤
    │ SRS              │ Baseline       │ Moderate    │ Default choice  │
    │ Stratified       │ Lower than SRS │ Harder      │ Known subgroups │
    │ Cluster          │ Higher than SRS│ Easiest     │ Geographic/cost │
    │ Systematic       │ ≈ SRS          │ Very easy   │ Ordered lists   │
    └──────────────────┴────────────────┴─────────────┴─────────────────┘



### PART 7 — SAMPLING BIAS: WHEN YOUR SAMPLE LIES

### Sources of Bias

Statistical methods assume the sample is representative of the target
population. Sampling bias occurs when some units are systematically
more likely to be selected than others, distorting estimates.

**Selection Bias** — the sample is drawn from the wrong population.

    Example: testing a drug only on volunteers. Volunteers are
    systematically healthier, more motivated, and younger than the
    general patient population → results will not generalise.

    The crucial distinction:
        Target population  (who you want to draw conclusions about)
        Sampling frame     (who you actually have access to draw from)
    If these differ, no amount of clever statistics fixes the gap.

**Survivorship Bias** — only the "survivors" of a process are visible.

    Classic example: WWII aircraft analysis.
    Engineers analysed bullet holes on planes that RETURNED.
    They nearly reinforced the already-hit areas — the planes that
    were downed by hits to other areas were simply absent from data.

    ML parallel: evaluating a model only on customers still active in
    your database systematically excludes churned users who found your
    product bad — a biased performance estimate.

**Non-Response Bias** — the units that don't respond differ systematically
    from those that do.

    Example: an email survey of customer satisfaction. Extremely happy
    and extremely unhappy customers respond; the indifferent majority
    does not. Your sample mean is biased toward extremes.

    Response rate below ~60% should trigger serious concern about
    non-response bias regardless of raw sample size.

**Confirmation & Observer Bias** — data collection is influenced by
    expectations, particularly in subjective labelling tasks.

    Example: a radiologist who knows a patient's clinical history may
    subconsciously rate ambiguous scans differently. Blinding is the fix.

    Summary of biases and fixes:
    ┌───────────────────────────────────────────────────────────────┐
    │ Bias type        │ Core mechanism         │ Primary remedy    │
    ├──────────────────┼────────────────────────┼───────────────────┤
    │ Selection        │ Wrong sampling frame   │ Fix the frame     │
    │ Survivorship     │ Losers are invisible   │ Track & impute    │
    │ Non-response     │ Refusers differ        │ Follow-up / weight│
    │ Confirmation     │ Collector expectations │ Blind collection  │
    └───────────────────────────────────────────────────────────────┘

    No statistical analysis — however sophisticated — can recover
    a parameter from a biased sample. Bias cannot be averaged away
    with more data; it only becomes more precisely wrong.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Central Limit Theorem — Live Demonstration": {
        "description": (
            "Demonstrate the CLT empirically. Draw thousands of samples from "
            "three very different population shapes (uniform, exponential, "
            "bimodal). Show that sample-mean distributions converge to normal "
            "regardless of the parent population."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

N_SAMPLES = 5_000   # number of independent samples to draw
SAMPLE_SIZES = [2, 5, 30, 100]

# ── Three very different population generators ────────────────────────────
populations = {
    "Uniform [0,1]":       {"gen": lambda n: rng.uniform(0, 1, n),
                            "mu": 0.5,  "sigma": np.sqrt(1/12)},
    "Exponential (λ=1)":   {"gen": lambda n: rng.exponential(1.0, n),
                            "mu": 1.0,  "sigma": 1.0},
    "Bimodal (2 Gaussians)":{"gen": lambda n: np.where(
                                rng.random(n) < 0.5,
                                rng.normal(-3, 0.8, n),
                                rng.normal( 3, 0.8, n)),
                            "mu": 0.0,  "sigma": np.sqrt(9 + 0.64)},
}

fig, axes = plt.subplots(
    len(populations), len(SAMPLE_SIZES) + 1,
    figsize=(16, 10)
)
fig.suptitle(
    "Central Limit Theorem — Sample Mean Distributions Across Populations & Sample Sizes",
    fontsize=11, fontweight="bold"
)

print("=" * 70)
print("  CENTRAL LIMIT THEOREM DEMONSTRATION")
print("=" * 70)

for row, (pop_name, pop) in enumerate(populations.items()):
    gen   = pop["gen"]
    mu    = pop["mu"]
    sigma = pop["sigma"]

    # Column 0: plot the raw population distribution
    ax0 = axes[row, 0]
    pop_data = gen(10_000)
    ax0.hist(pop_data, bins=60, density=True, color="steelblue",
             alpha=0.7, edgecolor="none")
    ax0.set_title(f"Population\\n{pop_name}", fontsize=8)
    ax0.set_ylabel("Density" if row == 0 else "")
    ax0.axvline(mu, color="red", lw=1.5, linestyle="--", label=f"μ={mu:.2f}")
    ax0.legend(fontsize=7)

    print(f"\\n  Population: {pop_name}   (μ={mu:.3f}, σ={sigma:.3f})")
    print(f"  {'n':>5}  {'E[x̄]':>10}  {'SE theory':>12}  {'SE actual':>12}  "
          f"{'Shapiro p-val':>14}  Normal?")
    print(f"  {'─'*65}")

    for col, n in enumerate(SAMPLE_SIZES, start=1):
        # Draw N_SAMPLES independent samples of size n, compute each mean
        samples = np.array([gen(n).mean() for _ in range(N_SAMPLES)])

        se_theory = sigma / np.sqrt(n)
        se_actual = samples.std(ddof=1)

        # Shapiro-Wilk normality test on the sample means
        _, p_val = stats.shapiro(samples[:500])  # Shapiro needs n ≤ 5000
        is_normal = "✓ YES" if p_val > 0.05 else "✗ NO "

        print(f"  {n:>5}  {samples.mean():>10.4f}  {se_theory:>12.4f}  "
              f"{se_actual:>12.4f}  {p_val:>14.4f}  {is_normal}")

        # Overlay normal curve
        ax = axes[row, col]
        ax.hist(samples, bins=50, density=True, color="coral",
                alpha=0.6, edgecolor="none", label="x̄ distribution")
        x_range = np.linspace(samples.min(), samples.max(), 200)
        normal_pdf = stats.norm.pdf(x_range, mu, se_theory)
        ax.plot(x_range, normal_pdf, "k-", lw=1.8, label="N(μ, σ²/n)")
        ax.set_title(f"n = {n}", fontsize=8)
        if row == 0:
            ax.legend(fontsize=6)

print()
print("  KEY OBSERVATIONS:")
print("  • E[x̄] ≈ μ for ALL populations and ALL sample sizes (unbiasedness)")
print("  • SE actual ≈ σ/√n for all cases — exact CLT prediction")
print("  • Shapiro p-val crosses 0.05 as n grows → normality confirmed")
print("  • Exponential (most skewed) needs largest n to normalise")

plt.tight_layout()
plt.savefig("clt_demonstration.png", dpi=120)
print("\\n  Plot saved → clt_demonstration.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Confidence Intervals — Coverage Simulation": {
        "description": (
            "Prove what a 95% CI really means. Run 200 repeated experiments, "
            "build a CI from each, and show that ~95% of them contain the true mean. "
            "Compare Z-interval vs t-interval at small n."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)

TRUE_MU    = 50.0
TRUE_SIGMA = 10.0
N_REPS     = 200        # number of repeated experiments
CONF       = 0.95
ALPHA      = 1 - CONF

print("=" * 65)
print("  CONFIDENCE INTERVAL COVERAGE SIMULATION")
print("=" * 65)
print(f"  True μ = {TRUE_MU},  σ = {TRUE_SIGMA}")
print(f"  Confidence level: {CONF*100:.0f}%")
print(f"  Repetitions: {N_REPS}")
print()

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle(
    f"95% Confidence Intervals — {N_REPS} repeated experiments\\n"
    "(Red = misses true mean, Blue = covers it)",
    fontsize=11, fontweight="bold"
)

for ax_idx, n in enumerate([10, 50]):
    ax = axes[ax_idx]
    covers = 0
    z_star = stats.norm.ppf(1 - ALPHA / 2)
    t_star = stats.t.ppf(1 - ALPHA / 2, df=n - 1)

    widths_z = []
    widths_t = []

    for i in range(N_REPS):
        sample = rng.normal(TRUE_MU, TRUE_SIGMA, n)
        x_bar  = sample.mean()
        s      = sample.std(ddof=1)
        se     = s / np.sqrt(n)

        lo = x_bar - t_star * se
        hi = x_bar + t_star * se
        covered = lo <= TRUE_MU <= hi
        if covered:
            covers += 1

        color = "steelblue" if covered else "tomato"
        ax.plot([lo, hi], [i, i], color=color, lw=0.8, alpha=0.7)
        ax.plot(x_bar, i, ".", color=color, ms=3)

        widths_t.append(hi - lo)
        z_lo = x_bar - z_star * (TRUE_SIGMA / np.sqrt(n))
        z_hi = x_bar + z_star * (TRUE_SIGMA / np.sqrt(n))
        widths_z.append(z_hi - z_lo)

    ax.axvline(TRUE_MU, color="black", lw=1.5, linestyle="--",
               label=f"True μ = {TRUE_MU}")
    coverage_pct = covers / N_REPS * 100
    ax.set_title(f"n = {n}  |  Coverage: {coverage_pct:.1f}%  "
                 f"({'✓' if abs(coverage_pct-95)<3 else '✗'})",
                 fontsize=9)
    ax.set_xlabel("Value")
    ax.set_ylabel("Experiment index")
    ax.legend(fontsize=8)

    print(f"  n = {n}:")
    print(f"    Coverage:             {coverage_pct:.1f}%   "
          f"(expected ≈ 95%)")
    print(f"    Mean CI width  (t):   {np.mean(widths_t):.3f}")
    print(f"    Mean CI width  (z):   {np.mean(widths_z):.3f}")
    print(f"    t* = {t_star:.4f}   z* = {z_star:.4f}  "
          f"  t wider by: {(t_star/z_star - 1)*100:.1f}%")
    print()

print("  INTERPRETATION:")
print("  • ~95 of 200 intervals contain the true mean — that IS the")
print("    correct meaning of a 95% CI.")
print("  • At small n, the t-interval is wider than the z-interval")
print("    because we don't know σ and must estimate it — extra uncertainty.")
print("  • As n grows, t* → z* and the two intervals converge.")

plt.tight_layout()
plt.savefig("ci_coverage.png", dpi=120)
print("\\n  Plot saved → ci_coverage.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Hypothesis Testing — p-values, Power & Type I/II Errors": {
        "description": (
            "Run a two-sample t-test framework. Visualise the null and "
            "alternative distributions, mark critical regions, and show "
            "how power changes as effect size and sample size vary."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(7)

print("=" * 65)
print("  HYPOTHESIS TESTING — TWO-SAMPLE t-TEST")
print("=" * 65)

# ── Part A: one concrete experiment ──────────────────────────────────────
mu_A, mu_B = 50.0, 53.5    # true means of group A and B
sigma      = 10.0
n          = 40
alpha      = 0.05

groupA = rng.normal(mu_A, sigma, n)
groupB = rng.normal(mu_B, sigma, n)

t_stat, p_val = stats.ttest_ind(groupA, groupB)
df    = 2 * n - 2
t_crit = stats.t.ppf(1 - alpha/2, df=df)

print(f"\\n  GROUP A:  n={n},  x̄={groupA.mean():.3f},  s={groupA.std(ddof=1):.3f}")
print(f"  GROUP B:  n={n},  x̄={groupB.mean():.3f},  s={groupB.std(ddof=1):.3f}")
print(f"  True difference: μ_B − μ_A = {mu_B - mu_A}")
print()
print(f"  t-statistic:   {t_stat:.4f}")
print(f"  t-critical:  ±{t_crit:.4f}   (α={alpha}, two-sided, df={df})")
print(f"  p-value:       {p_val:.4f}")
decision = "REJECT H₀ ✓" if abs(t_stat) > t_crit else "FAIL TO REJECT H₀"
print(f"  Decision:      {decision}")
print()

# ── Part B: power as a function of n and effect size ─────────────────────
print(f"  {'Effect d':>10}  {'n':>6}  {'Power':>8}  Note")
print(f"  {'─'*45}")

effect_sizes = [0.2, 0.5, 0.8]
ns           = [10, 30, 50, 100, 200]

power_grid = {}
for d in effect_sizes:
    for n_test in ns:
        analysis = stats.ttest_ind_from_stats(
            mean1=0, std1=1, nobs1=n_test,
            mean2=d, std2=1, nobs2=n_test
        )
        # Compute power via non-central t distribution
        nc     = d * np.sqrt(n_test / 2)      # non-centrality parameter
        t_c    = stats.t.ppf(1 - alpha/2, df=2*n_test-2)
        power  = (1 - stats.nct.cdf(t_c, df=2*n_test-2, nc=nc)
                    + stats.nct.cdf(-t_c, df=2*n_test-2, nc=nc))
        power_grid[(d, n_test)] = power

        note = ""
        if power >= 0.80: note = "← adequate power"
        if power >= 0.95: note = "← excellent"
        label = f"  {d:>10.1f}  {n_test:>6}  {power:>8.3f}  {note}"
        if n_test in [10, 30, 100]:
            print(label)

print()

# ── Part C: visualise null vs alternative distributions ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: null vs alternative t-distributions
ax = axes[0]
df_plot = 78  # n=40 per group
t_range = np.linspace(-5, 8, 400)
null_pdf = stats.t.pdf(t_range, df=df_plot)
nc_val   = (mu_B - mu_A) / (sigma * np.sqrt(2 / 40))
alt_pdf  = stats.nct.pdf(t_range, df=df_plot, nc=nc_val)

ax.plot(t_range, null_pdf, "steelblue", lw=2, label="Null distribution (δ=0)")
ax.plot(t_range, alt_pdf,  "tomato",    lw=2, label=f"Alt distribution (δ={mu_B-mu_A})")
ax.axvline( t_crit, color="gray", linestyle="--", lw=1.2)
ax.axvline(-t_crit, color="gray", linestyle="--", lw=1.2)

# Shade Type I (alpha) in blue tails
x_right = t_range[t_range >= t_crit]
ax.fill_between(x_right, stats.t.pdf(x_right, df=df_plot),
                alpha=0.35, color="steelblue", label=f"Type I (α={alpha})")
x_left = t_range[t_range <= -t_crit]
ax.fill_between(x_left, stats.t.pdf(x_left, df=df_plot),
                alpha=0.35, color="steelblue")

# Shade Type II (beta) — area of alt dist left of t_crit
x_beta = t_range[t_range <= t_crit]
ax.fill_between(x_beta, stats.nct.pdf(x_beta, df=df_plot, nc=nc_val),
                alpha=0.3, color="orange", label="Type II (β)")

ax.set_xlabel("t-statistic")
ax.set_ylabel("Density")
ax.set_title("Null vs Alternative — Type I & II Errors", fontsize=9)
ax.legend(fontsize=7)
ax.set_xlim(-5, 8)
ax.grid(alpha=0.3)

# Right: power curves
ax2 = axes[1]
n_range = np.arange(5, 200, 2)
for d in effect_sizes:
    nc_arr   = d * np.sqrt(n_range / 2)
    t_c_arr  = stats.t.ppf(1 - alpha/2, df=2*n_range-2)
    power_arr = (1 - stats.nct.cdf(t_c_arr, df=2*n_range-2, nc=nc_arr)
                   + stats.nct.cdf(-t_c_arr, df=2*n_range-2, nc=nc_arr))
    ax2.plot(n_range, power_arr, lw=2, label=f"d = {d}")

ax2.axhline(0.80, color="gray", linestyle="--", lw=1.2, label="80% power")
ax2.axhline(0.05, color="lightgray", linestyle=":", lw=1.2, label="α = 5%")
ax2.set_xlabel("Sample size per group (n)")
ax2.set_ylabel("Power (1 − β)")
ax2.set_title("Power Curves — Effect Size vs Sample Size", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("hypothesis_power.png", dpi=120)
print("  KEY FINDINGS:")
print("  • Small effects (d=0.2) need ~394 per group for 80% power")
print("  • Large effects (d=0.8) need only ~26 per group")
print("  • Running underpowered tests wastes effort and inflates FDR")
print("\\n  Plot saved → hypothesis_power.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Bootstrap — Universal Confidence Intervals": {
        "description": (
            "Compute bootstrap CIs for three statistics with no closed-form "
            "sampling distribution: the median, the 90th percentile, and a "
            "correlation coefficient. Compare bootstrap CIs to analytical ones "
            "for the mean as a sanity check."
        ),
        "language": "python",
        "code": '''
import numpy as np
from scipy import stats

rng = np.random.default_rng(123)

N        = 80        # sample size
B        = 5_000     # bootstrap replicates
CONF     = 0.95
ALPHA    = 1 - CONF

# ── Generate a sample from a skewed population ────────────────────────────
# Lognormal: median = exp(mu_log), mean ≠ median (right-skewed)
MU_LOG, SIGMA_LOG = 3.5, 0.7    # lognormal parameters
sample = rng.lognormal(MU_LOG, SIGMA_LOG, N)

# True population quantities
true_median = np.exp(MU_LOG)
true_mean   = np.exp(MU_LOG + SIGMA_LOG**2 / 2)
true_p90    = stats.lognorm.ppf(0.90, s=SIGMA_LOG, scale=np.exp(MU_LOG))

print("=" * 65)
print("  BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 65)
print(f"  Population: LogNormal(μ={MU_LOG}, σ={SIGMA_LOG})")
print(f"  n = {N},  B = {B} bootstrap replicates,  {CONF*100:.0f}% CI")
print(f"  True mean   = {true_mean:.3f}")
print(f"  True median = {true_median:.3f}")
print(f"  True P90    = {true_p90:.3f}")
print()

# ── Bootstrap engine ──────────────────────────────────────────────────────
def bootstrap_ci(sample, stat_fn, B=5_000, alpha=0.05, rng=None):
    """Return (point_estimate, lo, hi) using the percentile bootstrap."""
    if rng is None:
        rng = np.random.default_rng()
    n          = len(sample)
    boot_stats = np.array([
        stat_fn(rng.choice(sample, size=n, replace=True))
        for _ in range(B)
    ])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return stat_fn(sample), lo, hi, boot_stats

# ── 1. Mean (compare to analytical t-CI as sanity check) ─────────────────
x_bar     = sample.mean()
se        = sample.std(ddof=1) / np.sqrt(N)
t_star    = stats.t.ppf(1 - ALPHA/2, df=N-1)
t_lo, t_hi = x_bar - t_star*se, x_bar + t_star*se

b_mean, b_lo_m, b_hi_m, _ = bootstrap_ci(sample, np.mean, B=B, rng=rng)

# ── 2. Median ─────────────────────────────────────────────────────────────
b_med, b_lo_med, b_hi_med, boot_med = bootstrap_ci(
    sample, np.median, B=B, rng=rng)

# ── 3. 90th Percentile ────────────────────────────────────────────────────
b_p90, b_lo_p90, b_hi_p90, boot_p90 = bootstrap_ci(
    sample, lambda x: np.percentile(x, 90), B=B, rng=rng)

# ── 4. Correlation (add a correlated variable) ────────────────────────────
y_sample = 0.6 * sample + rng.normal(0, 20, N)
pair     = np.column_stack([sample, y_sample])

def corr_fn(data):
    return np.corrcoef(data[:, 0], data[:, 1])[0, 1]

b_corr, b_lo_corr, b_hi_corr, boot_corr = bootstrap_ci(
    pair, corr_fn, B=B, rng=rng)

true_corr = 0.6 * np.std(sample) / np.sqrt(
    (0.6*np.std(sample))**2 + 20**2)

# ── Print summary ─────────────────────────────────────────────────────────
print(f"  {'Statistic':18s}  {'Estimate':>10}  "
      f"{'CI lower':>10}  {'CI upper':>10}  {'True value':>12}  Method")
print(f"  {'─'*76}")

rows = [
    ("Mean (bootstrap)", b_mean, b_lo_m, b_hi_m, true_mean, "percentile boot"),
    ("Mean (t-test)",    x_bar,  t_lo,   t_hi,   true_mean, "analytical t"),
    ("Median",           b_med,  b_lo_med, b_hi_med, true_median, "percentile boot"),
    ("90th Percentile",  b_p90,  b_lo_p90, b_hi_p90, true_p90, "percentile boot"),
    ("Correlation",      b_corr, b_lo_corr,b_hi_corr, true_corr, "percentile boot"),
]

for name, est, lo, hi, truth, method in rows:
    covered = "✓" if lo <= truth <= hi else "✗"
    print(f"  {name:18s}  {est:10.3f}  {lo:10.3f}  {hi:10.3f}  "
          f"{truth:12.3f}  {covered} {method}")

print()
print("  KEY OBSERVATIONS:")
print("  • Bootstrap CI for mean ≈ analytical t CI — good sanity check")
print("  • Bootstrap gives valid CIs for median and P90 where no")
print("    closed-form formula exists — its main practical power.")
print("  • Correlation CI is asymmetric (bounded by ±1), which the")
print("    bootstrap handles naturally without Fisher's z transform.")
print()

# ── Show bootstrap distribution shape ────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Bootstrap Sampling Distributions", fontsize=10, fontweight="bold")

for ax, (boot_vals, name, truth, lo, hi) in zip(axes, [
    (boot_med,  "Median",         true_median, b_lo_med, b_hi_med),
    (boot_p90,  "90th Percentile",true_p90,    b_lo_p90, b_hi_p90),
    (boot_corr, "Correlation",    true_corr,   b_lo_corr,b_hi_corr),
]):
    ax.hist(boot_vals, bins=60, density=True, color="steelblue",
            alpha=0.6, edgecolor="none")
    ax.axvline(truth, color="red",  lw=1.8, linestyle="--", label=f"True = {truth:.3f}")
    ax.axvline(lo,    color="gray", lw=1.2, linestyle=":",  label=f"95% CI")
    ax.axvline(hi,    color="gray", lw=1.2, linestyle=":")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Statistic value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("bootstrap_dists.png", dpi=120)
print("  Plot saved → bootstrap_dists.png")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Sampling Methods — Variance Comparison": {
        "description": (
            "Simulate a stratified population and compare the precision of "
            "four sampling designs: SRS, stratified, cluster, and systematic. "
            "Show how each design affects the standard error of the mean."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(99)

# ── Build a synthetic stratified population ───────────────────────────────
# Three strata with very different means (within-strata variance is low)
STRATA = {
    "Low income":    {"n": 600, "mu": 30_000, "sigma": 5_000},
    "Mid income":    {"n": 300, "mu": 70_000, "sigma": 8_000},
    "High income":   {"n": 100, "mu":180_000, "sigma":20_000},
}

TOTAL_N = sum(s["n"] for s in STRATA.values())
population = []
stratum_ids = []
for sid, (name, cfg) in enumerate(STRATA.items()):
    vals = rng.normal(cfg["mu"], cfg["sigma"], cfg["n"])
    population.extend(vals)
    stratum_ids.extend([sid] * cfg["n"])

population   = np.array(population)
stratum_ids  = np.array(stratum_ids)
TRUE_MEAN    = population.mean()

# Cluster the population into N_CLUSTERS groups (randomly assigned)
N_CLUSTERS   = 100
cluster_ids  = rng.integers(0, N_CLUSTERS, TOTAL_N)

print("=" * 65)
print("  SAMPLING METHODS — STANDARD ERROR COMPARISON")
print("=" * 65)
print(f"  Population N = {TOTAL_N},  True mean = {TRUE_MEAN:,.1f}")
print(f"  Strata: Low {STRATA['Low income']['n']}, "
      f"Mid {STRATA['Mid income']['n']}, High {STRATA['High income']['n']}")
print()

TARGET_N  = 100    # sample size for all methods
N_TRIALS  = 2_000  # repeated experiments per method

results = {method: [] for method in
           ["SRS", "Stratified", "Cluster", "Systematic"]}

for _ in range(N_TRIALS):

    # ── SRS: draw TARGET_N units uniformly ────────────────────────────────
    idx = rng.choice(TOTAL_N, TARGET_N, replace=False)
    results["SRS"].append(population[idx].mean())

    # ── Stratified: proportional allocation ───────────────────────────────
    strat_sample = []
    for sid, cfg in enumerate(STRATA.values()):
        nh  = round(TARGET_N * cfg["n"] / TOTAL_N)
        mask = stratum_ids == sid
        idx_h = rng.choice(np.where(mask)[0], nh, replace=False)
        strat_sample.extend(population[idx_h])
    results["Stratified"].append(np.mean(strat_sample))

    # ── Cluster: sample clusters_per_draw entire clusters ────────────────
    clusters_per_draw = 10    # select 10 clusters out of 100
    chosen_clusters = rng.choice(N_CLUSTERS, clusters_per_draw, replace=False)
    cluster_sample = population[np.isin(cluster_ids, chosen_clusters)]
    results["Cluster"].append(cluster_sample.mean())

    # ── Systematic: random start, every kth unit ─────────────────────────
    k = TOTAL_N // TARGET_N                    # interval
    start = rng.integers(0, k)
    sys_idx = np.arange(start, TOTAL_N, k)[:TARGET_N]
    results["Systematic"].append(population[sys_idx].mean())

# ── Print comparison table ────────────────────────────────────────────────
print(f"  {'Method':15s}  {'Mean(x̄)':>12}  {'Bias':>10}  "
      f"{'SE':>10}  {'Relative SE':>13}")
print(f"  {'─'*62}")

srs_se = np.std(results["SRS"], ddof=1)
for method, vals in results.items():
    arr  = np.array(vals)
    bias = arr.mean() - TRUE_MEAN
    se   = arr.std(ddof=1)
    rel  = se / srs_se
    flag = ""
    if se < srs_se * 0.8:  flag = " ← lower variance ✓"
    if se > srs_se * 1.2:  flag = " ← higher variance ✗"
    print(f"  {method:15s}  {arr.mean():>12,.1f}  {bias:>10.1f}  "
          f"{se:>10.1f}  {rel:>13.3f}{flag}")

print()
print(f"  Theoretical SRS SE = σ/√n = {population.std():.1f}/√{TARGET_N}"
      f" = {population.std()/np.sqrt(TARGET_N):.1f}")
print()
print("  INTERPRETATION:")
print("  • Stratified SE is LOWER than SRS — variance within strata is")
print("    much smaller than overall variance (strata are homogeneous).")
print("  • Cluster SE is HIGHER than SRS — units within a cluster share")
print("    similar incomes (intra-cluster correlation > 0).")
print("  • Systematic ≈ SRS when population order is unrelated to income.")

# ── Plot distributions of estimates ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Sampling Distribution of x̄ by Design (n={TARGET_N}, {N_TRIALS} trials)",
    fontsize=10, fontweight="bold"
)
colors = ["steelblue", "seagreen", "tomato", "goldenrod"]

for ax, (method, color) in zip(axes.ravel(),
                               zip(results.keys(), colors)):
    arr = np.array(results[method])
    ax.hist(arr, bins=60, density=True, color=color,
            alpha=0.7, edgecolor="none")
    ax.axvline(TRUE_MEAN, color="black", lw=1.8, linestyle="--",
               label=f"True μ")
    ax.axvline(arr.mean(), color=color, lw=1.5, linestyle="-",
               label=f"Mean x̄")
    ax.set_title(
        f"{method} — SE = {arr.std(ddof=1):,.1f}", fontsize=9)
    ax.set_xlabel("Estimated mean income")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("sampling_methods.png", dpi=120)
print("\\n  Plot saved → sampling_methods.png")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · Sampling Bias — Simulation & Detection": {
        "description": (
            "Simulate three canonical sampling biases: selection bias, "
            "survivorship bias, and non-response bias. Quantify how far "
            "biased estimates drift from the true parameter and show that "
            "more data does NOT fix bias."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(55)

N_POP = 100_000   # large population — treat as ground truth

print("=" * 68)
print("  SAMPLING BIAS — SIMULATION & DETECTION")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────────
# BIAS 1: SELECTION BIAS
# True population: customer satisfaction 0-10, mean ≈ 5.5
# Biased sample: only customers who voluntarily call support (skew high+low)
# ─────────────────────────────────────────────────────────────────────────
satisfaction = rng.normal(5.5, 2.0, N_POP).clip(0, 10)
true_mean_sat = satisfaction.mean()

# Callers are biased: angry (< 3) and delighted (> 8) are more likely to call
call_prob = np.where(satisfaction < 3,  0.40,
            np.where(satisfaction > 8,  0.45,
                                        0.05))
is_caller = rng.random(N_POP) < call_prob

srs_means    = []
biased_means = []
sample_sizes = [50, 100, 200, 500, 1000, 3000]

for n in sample_sizes:
    # SRS estimate
    srs_means.append(
        rng.choice(satisfaction, n, replace=False).mean())
    # Biased (callers only)
    callers = satisfaction[is_caller]
    biased_means.append(
        rng.choice(callers, min(n, len(callers)), replace=False).mean())

print(f"\\n  SELECTION BIAS — Customer Satisfaction Survey")
print(f"  True population mean:     {true_mean_sat:.3f}")
print(f"  Mean of callers (biased): {satisfaction[is_caller].mean():.3f}")
print(f"  Bias:                     {satisfaction[is_caller].mean() - true_mean_sat:+.3f}")
print()
print(f"  n vs bias (biased sample):")
print(f"  {'n':>6}  {'SRS est.':>10}  {'Biased est.':>12}  {'Bias':>8}")
print(f"  {'─'*42}")
for n, srs, bias in zip(sample_sizes, srs_means, biased_means):
    print(f"  {n:>6}  {srs:>10.3f}  {bias:>12.3f}  "
          f"{bias - true_mean_sat:>+8.3f}")

# ─────────────────────────────────────────────────────────────────────────
# BIAS 2: SURVIVORSHIP BIAS
# Simulate 2000 startups: each has an initial quality score ~ N(50,20)
# Startups "survive" (stay in database) with prob proportional to quality
# Estimating mean quality from only surviving companies will overestimate
# ─────────────────────────────────────────────────────────────────────────
quality     = rng.normal(50, 20, 2000).clip(0, 100)
survive_prob = (quality / 100) ** 1.5    # better companies survive more
survived    = rng.random(2000) < survive_prob

true_q   = quality.mean()
surv_q   = quality[survived].mean()
surv_bias = surv_q - true_q

print(f"\\n  SURVIVORSHIP BIAS — Startup Quality Estimation")
print(f"  All startups — true mean quality:     {true_q:.2f}")
print(f"  Surviving startups — measured mean:   {surv_q:.2f}")
print(f"  Survivorship bias:                    {surv_bias:+.2f}")
print(f"  Survival rate:                        "
      f"{survived.mean()*100:.1f}%  ({survived.sum()} of 2000)")

# ─────────────────────────────────────────────────────────────────────────
# BIAS 3: NON-RESPONSE BIAS
# Survey income; high-income respondents are less likely to respond
# ─────────────────────────────────────────────────────────────────────────
income       = rng.lognormal(10.5, 0.8, N_POP)
true_inc     = income.mean()
# Response rate declines with income (privacy concerns)
response_p   = np.exp(-income / 200_000) * 0.7 + 0.05
response_p   = response_p.clip(0.05, 0.80)
responded    = rng.random(N_POP) < response_p

resp_mean = income[responded].mean()
resp_bias = resp_mean - true_inc

print(f"\\n  NON-RESPONSE BIAS — Income Survey")
print(f"  True population mean income:    {true_inc:>10,.0f}")
print(f"  Respondents only — mean income: {resp_mean:>10,.0f}")
print(f"  Non-response bias:              {resp_bias:>+10,.0f}")
print(f"  Response rate:                  {responded.mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────
# Plot: "More data doesn't fix bias"
# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Sampling Bias — More Data Does NOT Fix a Biased Sample",
             fontsize=10, fontweight="bold")

# Left: selection bias vs SRS as n grows
n_axis = np.array(sample_sizes)
axes[0].plot(n_axis, srs_means,    "steelblue", lw=2, marker="o", ms=5,
             label="SRS estimate")
axes[0].plot(n_axis, biased_means, "tomato",    lw=2, marker="s", ms=5,
             label="Biased (callers only)")
axes[0].axhline(true_mean_sat, color="black", lw=1.5, linestyle="--",
                label=f"True mean = {true_mean_sat:.2f}")
axes[0].set_xscale("log")
axes[0].set_title("Selection Bias — Satisfaction Survey", fontsize=9)
axes[0].set_xlabel("Sample size (log scale)")
axes[0].set_ylabel("Estimated mean satisfaction")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)

# Middle: survivorship — surviving vs all
categories = ["All startups\\n(truth)", "Surviving\\nstartups only"]
vals       = [true_q, surv_q]
bars = axes[1].bar(categories, vals, color=["steelblue", "tomato"],
                   alpha=0.75, edgecolor="white", width=0.5)
axes[1].axhline(true_q, color="black", linestyle="--", lw=1.2)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.5,
                 f"{v:.1f}", ha="center", fontsize=9)
axes[1].set_title(
    f"Survivorship Bias\\nBias = {surv_bias:+.1f} quality points", fontsize=9)
axes[1].set_ylabel("Mean quality score")
axes[1].grid(alpha=0.3, axis="y")

# Right: non-response — income distribution of respondents vs all
axes[2].hist(income / 1000, bins=80, density=True, alpha=0.4,
             color="steelblue", edgecolor="none", label="All (truth)")
axes[2].hist(income[responded] / 1000, bins=80, density=True, alpha=0.5,
             color="tomato", edgecolor="none", label="Respondents only")
axes[2].axvline(true_inc/1000, color="steelblue", lw=1.8, linestyle="--")
axes[2].axvline(resp_mean/1000, color="tomato",    lw=1.8, linestyle="--")
axes[2].set_xlim(0, 500)
axes[2].set_title(
    f"Non-Response Bias — Income Survey\\n"
    f"Bias = {resp_bias/1000:+.1f}k (low earners under-represented)",
    fontsize=8)
axes[2].set_xlabel("Income (£000s)")
axes[2].set_ylabel("Density")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("sampling_bias.png", dpi=120)
print()
print("  SUMMARY — Core lesson:")
print("  Increasing n drives SE → 0, but bias is a fixed offset.")
print("  At large n, a biased estimate becomes PRECISELY wrong.")
print("  Fix the sampling design — no statistical correction can")
print("  recover a parameter from a systematically biased sample.")
print("\\n  Plot saved → sampling_bias.png")
''',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


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