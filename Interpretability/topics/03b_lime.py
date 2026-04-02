"""
LIME — Local Interpretable Model-Agnostic Explanations
=======================================================

LIME (Local Interpretable Model-Agnostic Explanations) is one of the most
widely used and intuitively accessible techniques for explaining individual
predictions of any machine learning model. Introduced by Ribeiro, Singh,
and Guestrin in the paper "Why Should I Trust You?" (KDD 2016), LIME
democratised model explainability by providing a simple, practical answer
to a question every ML practitioner faces: "The model predicted X for this
specific input. Why?"

LIME's key insight is elegant and surprisingly powerful: even if a model is
globally complex and non-linear (a deep neural network, a random forest with
thousands of trees, a gradient-boosted ensemble), it is often approximately
LINEAR in a small neighbourhood around any specific input. You don't need
to understand the entire model to understand one prediction — you only need
to understand what the model does in the LOCAL region near that instance.

LIME finds this local linear approximation by:
    1. Generating many slightly different versions of the input
    2. Asking the black-box model to predict each version
    3. Fitting a simple (interpretable) model to these local observations
    4. Using the simple model's coefficients as the explanation

This model-agnostic approach means LIME works identically for:
    - Random forests and gradient boosting
    - Deep neural networks and transformers
    - Support vector machines and kernel methods
    - Any model that takes inputs and produces outputs

The paper's title — "Why Should I Trust You?" — captures the real
motivation: trust in AI systems requires the ability to understand and
audit individual decisions, not just aggregate accuracy statistics.
LIME was the first practical tool to make this possible at scale.

This module covers LIME's algorithm in depth for tabular, text, and image
data; the mathematical foundations; the critical instability problem and its
causes; practical implementation; and an honest assessment of when LIME is
appropriate and when SHAP or other methods should be preferred.

"""

import textwrap
import re

TOPIC_NAME   = "LIME — Local Interpretable Model-Agnostic Explanations"
DISPLAY_NAME = "03b · LIME"
ICON         = "🍋"
SUBTITLE     = "Local Linear Approximations, Perturbation Sampling, and Trust"


THEORY = """

##### PART 1 — THE CORE IDEA: LOCAL FIDELITY OVER GLOBAL SIMPLICITY

### The Global vs Local Dilemma

    GLOBAL explanation problem:
        "Fit a simple model to explain the black-box model everywhere."
        Problem: most real-world models are highly non-linear globally.
                 A linear model fitting the entire complex model will have
                 very low fidelity (high error). The explanation is wrong.

    LOCAL explanation (LIME's approach):
        "Fit a simple model to explain the black-box model HERE, near THIS input."
        Key insight: local linearity is a much weaker and more achievable requirement.
        Most smooth functions look approximately linear in a tiny enough neighbourhood.

    The intuition: imagine a complex curved surface (like a mountain range).
    Globally, it's non-linear — you can't describe it with a plane.
    But RIGHT WHERE YOU'RE STANDING, the surface looks flat — locally linear.
    LIME finds that local "flat plane" for any specific prediction.

### Formal Problem Definition

    Given:
        f:    the black-box model (any function X → Y)
        x:    the specific instance to explain
        G:    a class of interpretable models (e.g., linear models, decision trees)
        Ω(g): a complexity measure for model g (number of features, depth)
        πₓ:  a proximity measure (how close a sample is to x)

    LIME finds:
        ξ(x) = argmin_{g ∈ G} [L(f, g, πₓ) + Ω(g)]

    where L(f, g, πₓ) is the LOCAL fidelity loss:
        L = Σ_z πₓ(z) × (f(z) - g(z))²

    This is a regularised optimisation: find the simplest interpretable model g
    that faithfully approximates f in the neighbourhood of x.

    The proximity measure πₓ(z) = exp(-d(x, z)²/σ²) weights samples by their
    distance from x — closer samples get higher weight in the fit.

### Why "Model-Agnostic" Is the Key Property

    LIME treats f as a black box: it only requires the ability to query f.
    No access to gradients. No knowledge of model architecture.
    No special support needed from the model.

    This universality means:
        Same LIME explanation for a 3-layer neural net and a gradient-boosted forest.
        The explanation quality depends on local linearity, not model type.
        New model types automatically work with LIME without code changes.

    Contrast with SHAP variants:
        TreeSHAP: only for tree ensembles
        DeepSHAP: only for neural networks
        GradientSHAP: only for models with gradients
        LIME: for ALL of the above, plus any other model


##### PART 2 — THE LIME ALGORITHM IN DETAIL

### Step 1: Define the Interpretable Representation

    LIME works in an INTERPRETABLE FEATURE SPACE, not the original space.
    The interpretable representation z' ∈ {0,1}ⁿ is binary:
        z'ⱼ = 1: "feature j is present" (active, at its value in x)
        z'ⱼ = 0: "feature j is absent" (replaced by its "off" state)

    The "off" state depends on data type:
        Tabular: replace with feature's mean (or random sample from training)
        Text:    remove (mask) the word/token
        Image:   replace superpixel with grey/blurred reference image

    This binary representation enables:
        The local linear model to have interpretable coefficients.
        Features to be "turned off" independently.
        Efficient sparse selection (LASSO can select K important features).

### Step 2: Perturbation Sampling

    LIME samples N random perturbations in the interpretable space:
        For each of N samples i:
            z'ᵢ ← random binary vector (each bit = 1 with probability 0.5)
            zᵢ ← reconstruct actual input from z'ᵢ and x:
                  features where z'ᵢⱼ=1: use x's value
                  features where z'ᵢⱼ=0: use the "off" state

    The original instance x always included: z' = [1,1,...,1] → z = x.

    Sample count N:
        N=500:  rough approximation (fast, often used for exploration)
        N=1000: standard quality (good for most use cases)
        N=5000: high quality (for stable, production-grade explanations)

### Step 3: Black-Box Query

    Get predictions from the complex model for all perturbed samples:
        ŷᵢ = f(zᵢ)   for i = 1, ..., N

    This is the only interaction with f — LIME only needs predictions.
    f can be ANY function: forward pass, API call, database lookup.

    Prediction type:
        Classification: use predicted probability f(z)[:,class_of_interest]
        Regression:     use the raw predicted value f(z)
        Multi-output:   explain each output separately

### Step 4: Proximity Weighting

    Each perturbed sample gets a weight based on its distance from x:
        πₓ(z'ᵢ) = exp(-D(x, z'ᵢ)² / σ²)

    Distance options (D):
        COSINE similarity of z'ᵢ with all-ones vector (how many features "on")
        EUCLIDEAN distance in original feature space
        MANHATTAN distance

    Kernel width σ controls the neighbourhood size:
        Small σ: very narrow neighbourhood (only very close perturbations matter)
                 Explanation is more locally faithful but may be less stable.
        Large σ: wide neighbourhood (all perturbations matter about equally)
                 Explanation is more stable but may miss local structure.

    The original LIME paper uses:
        σ = sqrt(0.75 × n_features)   (scales with dimensionality)

### Step 5: Weighted Local Regression

    Fit an interpretable linear model minimising weighted squared error:
        min_w Σᵢ πₓ(z'ᵢ) × (ŷᵢ - w^T z'ᵢ - b)²  +  λ||w||₁

    The L1 regularisation (LASSO) is key: it drives many coefficients to zero,
    producing SPARSE explanations with only K important features.

    LIME selects K by choosing λ that yields exactly K non-zero coefficients.
    K is a user-specified parameter (default K=5).

    Alternative selection: forward selection (greedy, adds features one by one).

### Step 6: The Explanation

    The explanation is the vector of linear coefficients w:
        wⱼ > 0: feature j SUPPORTS the prediction (pushes it up)
        wⱼ < 0: feature j OPPOSES the prediction (pushes it down)
        |wⱼ|:   the magnitude of feature j's local influence

    The intercept b is the predicted value if ALL features were turned off.

    LOCAL FIDELITY:
        The explanation is faithful only near x.
        At distance 2σ from x, the explanation may be wrong.
        This is intentional — it's LOCAL, not global.


##### PART 3 — LIME FOR TABULAR DATA

### Tabular LIME Specifics

    For tabular data (rows × columns), each column is one feature.

    INTERPRETABLE REPRESENTATION:
        z'ⱼ = 1: column j takes its value from x (as in the original instance)
        z'ⱼ = 0: column j is replaced by a sampled value from the training set

    The "off" state for tabular:
        MEAN: replace with feature mean. Simpler but creates unrealistic samples.
        SAMPLE: randomly pick a training sample's value for that feature.
                More realistic — maintains the training distribution per feature.
                Default in the lime library.

    WHY SAMPLING FROM TRAINING IS BETTER:
        Categorical feature "gender" with values {M, F}:
            Mean replacement → gender = 0.52 (not a valid category)
            Sampled replacement → gender ∈ {M, F} (always valid)
        Continuous feature "age" with range [18, 90]:
            Mean → age = 42.3 (valid but boring)
            Sampled → age drawn from training distribution (more realistic)

    DISCRETISATION:
        The lime library discretises continuous features into bins before
        computing the interpretable representation.
        "age = 45" → "age in [40, 50]"
        This produces more stable explanations (small changes in age don't
        change the feature's bin, so the explanation doesn't flip).

        Discretisation options:
            QUARTILE: bins at 25th, 50th, 75th percentiles
            DECILE: bins at every 10th percentile
            ENTROPY: bins at information-maximising thresholds

### Feature Selection for Tabular LIME

    After fitting the weighted regression with all features,
    LIME uses one of these selection strategies:

    HIGHEST WEIGHTS:
        Simply take the K features with largest |wⱼ|.
        Fast but may select correlated features.

    LASSO:
        L1-regularised regression naturally zeroes out unimportant features.
        Varies regularisation strength until exactly K remain.
        More stable than highest weights for correlated features.

    FORWARD SELECTION:
        Greedy: add one feature at a time, choosing the one that
        most improves prediction of f(z) in the local region.
        Computationally expensive but most principled.

    RIDGE:
        L2 regression (smooth, doesn't zero coefficients).
        Select K features by ranking |wⱼ|.

### The Categorical Feature Problem

    Tabular data often has categorical features.
    LIME treats them as binary: is this categorical value present?

    "color" with values {red, green, blue}:
        One-hot encode: color_red, color_green, color_blue
        Interpretable feature: "is color=red?"
        Off state: randomly sample another color from training

    The explanation becomes:
        "color=red contributes +0.15 to approval probability"
        (compared to having a randomly sampled color instead)


##### PART 4 — LIME FOR TEXT DATA

### Text LIME Representation

    For text, the interpretable features are WORDS (or tokens).
    Each word in the input text is one binary feature:
        z'ⱼ = 1: word j is present in this perturbed version
        z'ⱼ = 0: word j is removed (replaced by [UNK] or an empty space)

    Perturbation example:
        Original: "The movie was great but the acting was poor"
        z' = [1,1,1,1,0,1,1,0,1]:
        Perturbed: "The movie was great [REMOVED] the acting [REMOVED] poor"

    The model predicts sentiment for each perturbed version.
    LIME learns which words, when removed, most change the prediction.

### Tokenisation and Vocabulary

    LIME for text uses the simplest possible tokenisation: split on whitespace.
    Each unique word is one feature.

    For n-gram models or BERT: the same principle applies but:
        n-grams: each n-gram is one binary feature (present/absent)
        BERT tokens: each subword token is one binary feature

    Masking strategies for absent words:
        REMOVE: simply delete the word (changes sentence length)
        UNKNOWN: replace with [UNK] token (preserves length)
        ZERO EMBEDDING: replace with zero embedding vector (for embedding models)
        MASK: replace with [MASK] token (for masked language models)

    The choice of masking strategy affects explanation quality:
        Some models are very sensitive to sentence length changes.
        MASK is most principled for BERT-family models.

### Text LIME in Practice

    Text LIME is excellent for:
        Sentiment analysis: "which words made the model think positive?"
        Spam detection: "which words triggered the spam classifier?"
        Topic classification: "which words indicate this is a politics article?"
        Any bag-of-words style model

    Text LIME can fail for:
        Long-range dependencies: "not good" vs "not bad" — removing "not" flips
                                  meaning, but LIME sees them as independent words
        Contextual embeddings: BERT's "bank" means different things in different
                               contexts; LIME's binary word removal doesn't capture this
        Very long texts: n=500 words → 2⁵⁰⁰ possible perturbations, too many
                         to sample well with N=1000 samples


##### PART 5 — LIME FOR IMAGE DATA

### The Superpixel Representation

    Images have millions of pixels — we can't make each pixel one binary feature.
    LIME groups pixels into SUPERPIXELS (coherent regions) using:
        SLIC (Simple Linear Iterative Clustering): most common
        Quickshift: watershed-based segmentation
        Felzenszwalb: graph-based segmentation

    Each superpixel is one binary feature:
        z'ⱼ = 1: superpixel j shows its original pixels
        z'ⱼ = 0: superpixel j is replaced with a reference (grey, blurred, or black)

    A typical image gets 50–200 superpixels:
        More superpixels: finer-grained explanations (which exact region?)
        Fewer superpixels: more coarse (which rough area?)

### Perturbation and Reference Image

    For each perturbed sample z'ᵢ:
        Construct the actual perturbed image by:
            Keeping superpixels where z'ᵢⱼ=1 at their original values
            Replacing superpixels where z'ᵢⱼ=0 with the reference image

    Reference image options:
        GREY CONSTANT (e.g., mean=128): most common, simple
        GAUSSIAN BLUR: realistic (averages out the content)
        INPAINTING: fill with surrounding context (most realistic but slow)
        BLACK: aggressive, can cause artefacts for brightness-sensitive models

    The model is then queried with each perturbed image.

### Reading Image LIME Explanations

    After fitting the local linear model, the LIME explanation for images is:
        Positive superpixels (high wⱼ > 0): support the predicted class
        Negative superpixels (low wⱼ < 0): oppose the predicted class

    Visualisation:
        GREEN overlay: superpixels that support the prediction
        RED overlay: superpixels that oppose the prediction
        Or: show only the top K positive superpixels on a grey background

    Interpretation example:
        ImageNet classifier predicts "tabby cat"
        LIME highlights: ears, nose, striped fur patterns → GREEN
        LIME highlights: background grass → RED (opposes "cat")

    Failure example (Clever Hans):
        ImageNet "husky" classifier
        LIME highlights: snow background → GREEN
        Interpretation: model is using snow as a proxy for "husky" — wrong reasoning!


##### PART 6 — THE INSTABILITY PROBLEM: LIME'S CRITICAL LIMITATION

### What Instability Means

    Run LIME on the SAME instance with a DIFFERENT random seed.
    The explanation changes — sometimes dramatically.

    Example: loan application explanation
        Seed 1: top features = [credit_score (+0.25), income (+0.18), debt (-0.15)]
        Seed 2: top features = [income (+0.22), employment (+0.12), debt (-0.17)]
        credit_score disappeared and employment appeared — different story!

    This instability is not a bug — it's a consequence of the algorithm:
        The perturbation sample is RANDOM.
        Different random samples → different locally-fitted regression → different coefficients.

### Sources of Instability

    1. SAMPLING VARIANCE:
        With N=1000 samples from a high-dimensional neighbourhood, the
        sampling error is significant. Rare combinations may or may not appear.

    2. FEATURE CORRELATION:
        If features A and B are correlated, sometimes A gets credit (seed 1)
        and sometimes B gets credit (seed 2) — they can substitute for each other
        in the local linear fit.

    3. NEIGHBOURHOOD SIZE (kernel width σ):
        Different σ values include different sets of points.
        The locally linear approximation can differ substantially.

    4. NON-LINEARITY:
        In highly non-linear regions, the local linear approximation is poor.
        Small changes in which points are sampled → very different linear fits.

    5. DISCRETISATION ARTIFACTS (tabular):
        Feature values near bin boundaries can flip between bins in different
        perturbation samples, causing instability.

### Measuring Instability

    JACCARD INSTABILITY:
        Run LIME K times on the same instance.
        Measure how often the TOP-K features are the SAME:
        Jaccard(run_i, run_j) = |top_features_i ∩ top_features_j| / |top_features_i ∪ top_features_j|
        Perfect stability: Jaccard = 1.0 (same features every time)
        Random: Jaccard ≈ K/n

    RANK CORRELATION:
        Compute Spearman rank correlation of feature importance scores across runs.
        High correlation (r > 0.9): features are ranked similarly.
        Low correlation (r < 0.5): rankings are very unstable.

    SIGN CONSISTENCY:
        Do features maintain the same DIRECTION (positive/negative) across runs?
        Sign flip: feature is positive in one run, negative in another → unreliable.

### Mitigating Instability

    1. INCREASE N (most direct fix):
        N=500 → N=5000: sampling variance decreases.
        Diminishing returns above N=5000 for most use cases.
        Cost: N× slower (N model evaluations).

    2. TUNE KERNEL WIDTH σ:
        Default σ often not optimal.
        Calibrate: run with σ × {0.5, 0.75, 1.0, 1.5, 2.0}, pick the σ
        that gives best balance of faithfulness and stability.

    3. USE DETERMINISTIC LIME (BayLIME, ALIME):
        BayLIME: Bayesian regularisation reduces instability.
        ALIME: Adversarial robustness against instability via augmented samples.

    4. REPORT MULTIPLE RUNS:
        Show the MEAN explanation across K=10 runs.
        Show confidence intervals on each coefficient.
        Report the Jaccard stability score alongside explanations.

    5. USE SHAP INSTEAD:
        For tree models: TreeSHAP is deterministic (no sampling variance).
        For linear models: LinearSHAP is exact.
        SHAP's efficiency axiom is checkable; LIME's fidelity is not guaranteed.

    6. USE ANCHORS INSTEAD (for rule-based needs):
        Anchors (Ribeiro et al., 2018) are more stable than LIME.
        They find conditions sufficient for a prediction rather than linear weights.


##### PART 7 — LIME VARIANTS AND EXTENSIONS

### LIME for Time Series

    Time series data is handled similarly to image data with superpixels:
        SUBSEQUENCES as features: each contiguous time window is one feature.
        Off state: replace with the mean or a reference segment.
        The explanation shows which time windows matter most.

    The lime library has LimeTimeSeriesExplainer for this.

### NormLIME

    NormLIME (Ahern et al., 2019) addresses LIME's instability:
        1. Run LIME N times with different random seeds.
        2. Aggregate the results using a normalised averaging procedure.
        3. Produces more stable explanations at the cost of N× computation.

### LIME-Aleph

    LIME-Aleph extends LIME to produce FOL (First-Order Logic) rules:
        Instead of linear coefficients, produces IF-THEN logical rules.
        More interpretable for domain experts who think in rules.

### SP-LIME (Submodular Pick LIME)

    SP-LIME answers: "Which training instances should I manually inspect?"
    Uses submodular optimisation to select B instances that collectively
    provide global coverage of the model's behaviour.
    The user inspects these B LIME explanations to understand the whole model.

### GLIME (Generalised LIME)

    GLIME unifies LIME, SHAP, and related methods under a single framework:
        Different choices of the weighting function πₓ and the surrogate
        model class G correspond to different explanation methods.
        LIME: πₓ = exponential kernel, G = linear model
        KernelSHAP: πₓ = Shapley kernel, G = linear model
        This reveals that KernelSHAP IS a special case of LIME.


##### PART 8 — WHEN TO USE LIME AND WHEN NOT TO

### LIME's Sweet Spots

    TEXT CLASSIFICATION:
        LIME is excellent for text models. Words are natural discrete features.
        Binary perturbation (word present/absent) is semantically meaningful.
        The LIME library's text explainer is well-tested and widely used.
        Result: intuitive highlighted-word explanations.

    IMAGE CLASSIFICATION:
        Superpixel-based LIME produces visually clear explanations.
        Easy to understand: "these regions caused the prediction."
        Used successfully for debugging CNNs (detecting Clever Hans behaviour).

    COMPLEX API-BASED MODELS:
        When you only have API access (no model internals):
        LIME works with just predict_fn(X) → predictions.
        Perfect for third-party models or models as services.

    QUICK EXPLORATION:
        Fast to set up (pip install lime, 5 lines of code).
        Good for initial exploration before committing to more rigorous methods.

### When LIME Struggles

    TABULAR DATA WITH CORRELATED FEATURES:
        Correlation causes instability (which feature gets credit?).
        KernelSHAP or TreeSHAP is more principled for correlated tabular data.

    HIGH-DIMENSIONAL DATA:
        With n=500 features, N=1000 samples barely scratches the surface.
        Need N >> 2ⁿ which is infeasible.
        PartitionSHAP or TreeSHAP handles high dimensions better.

    PRODUCTION WITH AUDIT REQUIREMENTS:
        LIME explanations can't be verified (no efficiency axiom).
        For regulatory compliance, SHAP's provable efficiency is preferable.
        LIME can produce contradictory explanations for similar instances.

    MODELS WITH COMPLEX NON-LINEARITIES:
        Deep in a highly non-linear region, even the local linear
        approximation is poor. LIME's surrogate fidelity (R²) will be low.
        Always check the R² of the local model — low R² = don't trust it!

### LIME vs SHAP: Definitive Comparison

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Property              │ LIME                    │ SHAP              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Theoretical basis     │ Local optimisation      │ Shapley axioms    │
    │ Efficiency axiom      │ ❌ Not guaranteed       │ ✅ Always holds   │
    │ Stability             │ ⚠️  Can vary by seed    │ ✅ Deterministic  │
    │ Model agnostic        │ ✅ Yes                  │ ✅ (KernelSHAP)   │
    │ Tree-specific speed   │ N/A                     │ ✅ TreeSHAP fast  │
    │ Text/image support    │ ✅ Excellent            │ ✅ (with effort)  │
    │ Global aggregation    │ ⚠️  Not principled      │ ✅ Always correct │
    │ Ease of use           │ ✅ Very simple          │ ✅ Simple too     │
    │ Interpretable output  │ ✅ Linear weights       │ ✅ Attribution    │
    │ Feature interactions  │ ❌ Misses them          │ ✅ SHAP interact  │
    │ Fidelity check        │ ⚠️  R² (approximate)    │ ✅ Exact sum      │
    └─────────────────────────────────────────────────────────────────────┘

    Practical recommendation:
        For tree ensembles:       Use TreeSHAP (exact, fast, principled).
        For neural networks:      Use GradientSHAP or DeepSHAP.
        For text (any model):     LIME is excellent and practical.
        For images (any model):   LIME is the standard choice.
        For black-box APIs:       Both work; LIME is simpler to use.
        For regulatory audit:     Prefer SHAP (verifiable efficiency).
        For initial exploration:  LIME is faster to set up.

"""


OPERATIONS = {

    "1 · LIME from Scratch — Tabular Algorithm with Full Diagnostics": {
        "description": (
            "Implement LIME for tabular data completely from scratch. "
            "Step through each algorithm stage: perturbation, querying, weighting, fitting. "
            "Show how the kernel width affects neighbourhood size and explanation quality. "
            "Measure local fidelity (R²) of the surrogate model. "
            "Demonstrate the instability problem with multiple seeds. "
            "Compute Jaccard stability and rank correlation metrics. "
            "Compare discretised vs continuous feature representations."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import numpy as _np_impl
import scipy.optimize as _sp_opt

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class Ridge:
    def __init__(self, alpha=0.01, **kw): self.alpha = alpha
    def fit(self, X, y, sample_weight=None):
        X=_np_impl.atleast_2d(X); y=_np_impl.asarray(y,dtype=float)
        n,d=X.shape
        w=_np_impl.asarray(sample_weight,dtype=float) if sample_weight is not None else _np_impl.ones(n)
        w=w/w.sum()*n
        Xb=_np_impl.column_stack([X,_np_impl.ones(n)])
        A=(Xb.T*w)@Xb; A[_np_impl.arange(d),_np_impl.arange(d)]+=self.alpha
        params=_np_impl.linalg.solve(A,Xb.T@(w*y))
        self.coef_=params[:d]; self.intercept_=params[d]; return self
    def predict(self,X): return _np_impl.atleast_2d(X)@self.coef_+self.intercept_
from scipy.stats import spearmanr
import time

print("=" * 65)
print("  LIME FROM SCRATCH — TABULAR ALGORITHM WITH FULL DIAGNOSTICS")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Setup: create a realistic non-linear black-box model
# ─────────────────────────────────────────────────────────────────────────

# Generate training data (loan applications)
N = 1000
income       = np.random.normal(50000, 15000, N).clip(15000, 120000) / 100000
credit_score = np.random.normal(0.65, 0.15, N).clip(0.2, 1.0)
debt_ratio   = np.random.normal(0.35, 0.12, N).clip(0.0, 0.85)
age          = np.random.normal(42, 12, N).clip(20, 75) / 75
savings      = np.random.normal(0.25, 0.18, N).clip(0.0, 1.0)
employment_y = np.random.normal(5, 4, N).clip(0, 20) / 20

X_train = np.column_stack([income, credit_score, debt_ratio, age, savings, employment_y])
feature_names = ["Income", "CreditScore", "DebtRatio", "Age", "Savings", "Employment"]

def black_box_model(X):
    """
    Non-linear loan approval model — intentionally complex.
    TRUE feature importances (ground truth):
        Income: important (positive)
        CreditScore: most important (positive)
        DebtRatio: important (negative)
        Age: minor (non-monotone, U-shape)
        Savings: moderately important (positive)
        Employment: minor positive
    Interactions:
        Income × CreditScore: synergistic
        High debt with low savings: extra penalty
    """
    X = np.atleast_2d(X)
    logit = (
        2.5 * X[:, 0]                          # income
        + 4.0 * X[:, 1]                         # credit score (most important)
        - 3.0 * X[:, 2]                         # debt ratio (negative)
        + 0.5 * (X[:, 3] - 0.5) ** 2           # age (U-shape: young and old harder)
        + 1.8 * X[:, 4]                         # savings
        + 0.6 * X[:, 5]                         # employment
        + 2.0 * X[:, 0] * X[:, 1]              # income × credit interaction
        - 1.5 * X[:, 2] * (1 - X[:, 4])        # debt × (1-savings) penalty
        - 2.5                                    # bias
    )
    return 1 / (1 + np.exp(-logit))

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LIME algorithm step-by-step
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — LIME algorithm: step-by-step walkthrough")
print("━" * 65)
print()

class LIMETabularFromScratch:
    """
    Complete LIME implementation for tabular data.
    Implements the original Ribeiro et al. (2016) algorithm exactly.
    """
    def __init__(self, training_data, feature_names, kernel_width=None,
                 discretise=True, n_bins=4):
        self.train_data     = training_data
        self.feature_names  = feature_names
        self.n_features     = training_data.shape[1]
        self.kernel_width   = (kernel_width or
                               np.sqrt(0.75 * self.n_features))  # default from paper
        self.discretise     = discretise
        self.n_bins         = n_bins

        # Compute feature statistics for perturbation
        self.feature_means  = training_data.mean(axis=0)
        self.feature_stds   = training_data.std(axis=0) + 1e-8

        # Compute discretisation bins (quartiles)
        self.bins = [
            np.percentile(training_data[:, j],
                          np.linspace(0, 100, n_bins + 1))
            for j in range(self.n_features)
        ]

    def _to_interpretable(self, x):
        """Convert continuous x to binary interpretable representation."""
        z_prime = np.zeros(self.n_features)
        for j in range(self.n_features):
            if self.discretise:
                # Which bin does x[j] fall in?
                bin_x = np.digitize(x[j], self.bins[j][1:-1])
                z_prime[j] = bin_x  # bin index (not binary, but categorical)
            else:
                z_prime[j] = 1   # "present" in the interpretable space
        return z_prime

    def _perturb(self, x, n_samples=1000, rng=None):
        """
        STEP 1 + 2: Generate perturbed samples.
        Returns:
            z_prime: binary interpretable representations (n_samples × n_features)
            z_actual: actual feature values to pass to black-box (n_samples × n_features)
        """
        if rng is None:
            rng = np.random.RandomState(42)

        z_prime  = np.zeros((n_samples, self.n_features))
        z_actual = np.zeros((n_samples, self.n_features))

        for i in range(n_samples):
            # Random binary mask: 1 = feature present, 0 = off
            mask     = rng.randint(0, 2, self.n_features).astype(float)
            z_prime[i] = mask

            for j in range(self.n_features):
                if mask[j] == 1:
                    z_actual[i, j] = x[j]             # keep original value
                else:
                    # Replace with random sample from training distribution
                    idx = rng.randint(0, len(self.train_data))
                    z_actual[i, j] = self.train_data[idx, j]

        # Always include the original instance first
        z_prime  = np.vstack([np.ones(self.n_features), z_prime])
        z_actual = np.vstack([x, z_actual])

        return z_prime, z_actual

    def _compute_weights(self, z_prime):
        """
        STEP 4: Compute proximity weights.
        πₓ(z') = exp(-d(z', ones)² / σ²)
        """
        # Distance from all-ones (= original instance in binary space)
        distances = np.sqrt(((z_prime - 1.0) ** 2).sum(axis=1))
        weights   = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        return weights

    def _fit_surrogate(self, z_prime, y_pred, weights, n_features_select=5):
        """
        STEP 5: Weighted LASSO regression.
        Returns: (coefficients, intercept, r_squared, surrogate_predictions)
        """
        # Fit Ridge with weighted samples
        surrogate = Ridge(alpha=0.01)
        surrogate.fit(z_prime, y_pred, sample_weight=weights)
        y_surrogate = surrogate.predict(z_prime)

        # Compute weighted R² (local fidelity)
        wss_res = np.sum(weights * (y_pred - y_surrogate) ** 2)
        y_wmean = np.average(y_pred, weights=weights)
        wss_tot = np.sum(weights * (y_pred - y_wmean) ** 2)
        r_sq    = 1 - wss_res / (wss_tot + 1e-10)

        # Select top n_features_select by |coefficient|
        coef     = surrogate.coef_
        top_idx  = np.argsort(np.abs(coef))[::-1][:n_features_select]

        # Refit with only selected features (for cleaner explanation)
        surrogate_sparse = Ridge(alpha=0.01)
        surrogate_sparse.fit(z_prime[:, top_idx], y_pred, sample_weight=weights)

        full_coef           = np.zeros(self.n_features)
        full_coef[top_idx]  = surrogate_sparse.coef_

        return full_coef, surrogate.intercept_, r_sq, y_surrogate

    def explain(self, x, predict_fn, n_samples=1000, n_features_select=5, seed=42):
        """
        Full LIME pipeline.
        Returns dict with explanation details.
        """
        rng = np.random.RandomState(seed)

        # Steps 1+2: Perturb
        z_prime, z_actual = self._perturb(x, n_samples, rng)

        # Step 3: Query black-box
        y_pred = predict_fn(z_actual)

        # Step 4: Weights
        weights = self._compute_weights(z_prime)

        # Step 5: Fit surrogate
        coefs, intercept, r_sq, y_surrogate = self._fit_surrogate(
            z_prime, y_pred, weights, n_features_select)

        return {
            "coefs":        coefs,
            "intercept":    intercept,
            "r_squared":    r_sq,
            "prediction":   y_pred[0],    # original instance prediction
            "n_samples":    n_samples,
            "weights":      weights,
            "z_prime":      z_prime,
            "y_pred":       y_pred,
            "y_surrogate":  y_surrogate,
        }


# Explain a specific instance
x_explain  = np.array([0.60, 0.72, 0.35, 0.50, 0.40, 0.30])
pred_x     = black_box_model(x_explain.reshape(1,-1))[0]

print(f"  Instance to explain:")
for fname, val in zip(feature_names, x_explain):
    print(f"    {fname:<14}: {val:.3f}")
print(f"  Black-box prediction: {pred_x:.4f}  ({pred_x:.1%} approval probability)")
print()

lime = LIMETabularFromScratch(X_train, feature_names)
result = lime.explain(x_explain, lambda X: black_box_model(X), n_samples=1000)

print(f"  LIME explanation (N=1000, kernel_width={lime.kernel_width:.3f}):")
print(f"  Local surrogate R² = {result['r_squared']:.4f}  "
      f"({'Good' if result['r_squared'] > 0.7 else 'Poor'} local fidelity)")
print()
print(f"  {'Feature':<14} | {'Value':>8} | {'Coefficient':>13} | {'Direction':>22} | {'Magnitude bar'}")
print(f"  {'─'*80}")
max_coef = np.abs(result['coefs']).max() + 1e-10
for i, (fname, val, coef) in enumerate(zip(feature_names, x_explain, result['coefs'])):
    bar_len = int(abs(coef) / max_coef * 15)
    bar     = "█" * bar_len
    if coef == 0:
        direction = "— not selected"
    elif coef > 0:
        direction = "↑ supports approval"
    else:
        direction = "↓ opposes approval"
    print(f"  {fname:<14} | {val:>8.3f} | {coef:>+13.4f} | {direction:>22} | {bar}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Kernel width effect on neighbourhood and fidelity
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Kernel width: neighbourhood size vs fidelity")
print("━" * 65)
print()

print("  Effect of kernel width σ on explanation quality:")
print()
print(f"  {'σ':>6} | {'R²':>8} | {'Eff. samples':>14} | {'Top features':>30}")
print(f"  {'─'*65}")

for sigma in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
    lime_test = LIMETabularFromScratch(X_train, feature_names, kernel_width=sigma)
    r_test    = lime_test.explain(x_explain, lambda X: black_box_model(X),
                                   n_samples=1000, seed=42)

    # Effective sample size: 1 / sum(w²) × (sum(w))²
    w         = r_test['weights']
    eff_n     = (w.sum())**2 / (w**2).sum()

    top_feats = sorted([(abs(c), f) for c, f in zip(r_test['coefs'], feature_names)
                        if c != 0], reverse=True)[:3]
    top_str   = ", ".join(f"{f}({c:.2f})" for c, f in top_feats)

    r_qual = "✅" if r_test['r_squared'] > 0.7 else "⚠️ "
    print(f"  {sigma:>6.1f} | {r_test['r_squared']:>8.4f} {r_qual}| {eff_n:>14.1f} | {top_str}")

print()
print("  Small σ: few effective samples (very local), potentially lower R²")
print("  Large σ: many effective samples (global), potentially poor local fit")
print("  Optimal σ: balances fidelity and locality. Use cross-validation.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Instability analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Instability: same instance, different random seeds")
print("━" * 65)
print()

K_RUNS = 20
N_SELECT = 3   # select top 3 features
all_coefs  = []
all_topk   = []

for seed in range(K_RUNS):
    r = lime.explain(x_explain, lambda X: black_box_model(X),
                     n_samples=1000, n_features_select=N_SELECT, seed=seed)
    all_coefs.append(r['coefs'])
    top_k = set(np.argsort(np.abs(r['coefs']))[::-1][:N_SELECT])
    all_topk.append(top_k)

all_coefs = np.array(all_coefs)

# Jaccard stability
jaccards = []
for i in range(K_RUNS):
    for j in range(i+1, K_RUNS):
        inter  = len(all_topk[i] & all_topk[j])
        union  = len(all_topk[i] | all_topk[j])
        jaccards.append(inter / union if union > 0 else 1.0)
mean_jaccard = np.mean(jaccards)

# Rank correlation across runs
rank_corrs = []
for i in range(K_RUNS):
    for j in range(i+1, K_RUNS):
        r_corr, _ = spearmanr(np.abs(all_coefs[i]), np.abs(all_coefs[j]))
        rank_corrs.append(r_corr)
mean_rank_corr = np.mean(rank_corrs)

print(f"  Stability analysis over {K_RUNS} runs (N=1000 samples each):")
print(f"  Mean Jaccard similarity of top-{N_SELECT} features: {mean_jaccard:.4f}")
print(f"  (1.0 = perfectly stable, {N_SELECT/len(feature_names):.2f} = random baseline)")
print(f"  Mean Spearman rank correlation: {mean_rank_corr:.4f}")
print()

# Per-feature coefficient variability
print(f"  Per-feature coefficient variability across {K_RUNS} runs:")
print(f"  {'Feature':<14} | {'Mean coef':>12} | {'Std coef':>10} | {'CoV (%)':>10} | {'Stable?':>10}")
print(f"  {'─'*62}")
for i, fname in enumerate(feature_names):
    mean_c = all_coefs[:, i].mean()
    std_c  = all_coefs[:, i].std()
    cov    = abs(std_c / (mean_c + 1e-8)) * 100
    stable = "✅ Stable" if cov < 50 else ("⚠️  Moderate" if cov < 150 else "❌ Unstable")
    print(f"  {fname:<14} | {mean_c:>+12.4f} | {std_c:>10.4f} | {cov:>9.1f}% | {stable}")

print()

# Effect of N on stability
print(f"  Effect of N (sample count) on stability:")
print(f"  {'N':>7} | {'Jaccard':>9} | {'Rank corr':>11} | {'Time (s)':>10}")
print(f"  {'─'*45}")
for N_test in [100, 500, 1000, 2000, 5000]:
    jac_list = []
    t0 = time.perf_counter()
    coefs_n = []
    for seed in range(10):
        r_n = lime.explain(x_explain, lambda X: black_box_model(X),
                           n_samples=N_test, seed=seed)
        coefs_n.append(r_n['coefs'])
        top_k = set(np.argsort(np.abs(r_n['coefs']))[::-1][:3])
        jac_list.append(top_k)
    t_n = time.perf_counter() - t0
    coefs_n = np.array(coefs_n)
    j_pairs = [len(jac_list[i]&jac_list[j])/len(jac_list[i]|jac_list[j])
               for i in range(10) for j in range(i+1,10)]
    rc_pairs = [spearmanr(np.abs(coefs_n[i]), np.abs(coefs_n[j]))[0]
                for i in range(10) for j in range(i+1,10)]
    print(f"  {N_test:>7} | {np.mean(j_pairs):>9.4f} | {np.mean(rc_pairs):>11.4f} | {t_n:>10.3f}")

print()
print("  Stability improves with N but has diminishing returns above N=2000.")
print("  For production: N=2000-5000 with stability monitoring.")
''',
    },

    "2 · LIME for Text and the lime Library": {
        "description": (
            "LIME for natural language: token-level explanations. "
            "Use the lime library's LimeTextExplainer end-to-end. "
            "Explain sentiment analysis, spam detection, and topic classification. "
            "Show how word removal affects predictions. "
            "Visualise explanations as weighted word highlighting. "
            "Demonstrate LIME instability for text and mitigation via averaging. "
            "Compare LIME text explanations to TF-IDF feature importance."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import re
import time
from collections import Counter

print("=" * 65)
print("  LIME FOR TEXT — TOKEN-LEVEL EXPLANATIONS")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Build text classifiers and LIME explainers
# ─────────────────────────────────────────────────────────────────────────

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    import lime
    import lime.lime_text

    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("  lime not installed: pip install lime scikit-learn")

# Movie review dataset (representative)
reviews = [
    ("This film is absolutely brilliant! A masterpiece of cinema.", 1),
    ("Terrible movie. Complete waste of time and money.", 0),
    ("Outstanding performances and breathtaking cinematography.", 1),
    ("Boring and predictable. I fell asleep halfway through.", 0),
    ("A beautiful story that moved me to tears. Highly recommend.", 1),
    ("Worst film I have ever seen. Poorly written and directed.", 0),
    ("Incredible acting and a thought-provoking story.", 1),
    ("Dull, lifeless, and utterly forgettable. Avoid this movie.", 0),
    ("Stunning visuals and an emotionally powerful narrative.", 1),
    ("Disappointing and poorly paced. The acting was wooden.", 0),
    ("A cinematic gem. The director's finest work.", 1),
    ("Awful script and terrible performances. A total disaster.", 0),
    ("Heartfelt and genuinely moving. A must-see film.", 1),
    ("Slow, tedious, and frustrating to watch.", 0),
    ("Gripping from start to finish with excellent character development.", 1),
    ("Derivative and uninspired. I wanted my money back.", 0),
]

texts  = [r for r, _ in reviews]
labels = [l for _, l in reviews]

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LIME text algorithm from scratch
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Text LIME algorithm from scratch")
print("━" * 65)
print()

class TextLIMEScratch:
    """
    Minimal LIME for text classification, built from first principles.
    Each word in the input is one binary feature.
    """
    def __init__(self, tokeniser=None, kernel_width=25):
        self.tokenise     = tokeniser or (lambda t: t.lower().split())
        self.kernel_width = kernel_width

    def _perturb_text(self, text, n_samples=500, rng=None):
        """
        Perturb by randomly removing words.
        Returns: (z_prime binary masks, perturbed texts)
        """
        if rng is None: rng = np.random.RandomState(42)
        tokens = self.tokenise(text)
        n_toks = len(tokens)

        z_primes = []
        texts_p  = []

        # Always include original
        z_primes.append(np.ones(n_toks))
        texts_p.append(text)

        for _ in range(n_samples):
            mask      = rng.randint(0, 2, n_toks)
            z_primes.append(mask.astype(float))
            # Reconstruct text: remove words where mask=0
            kept      = [tok for tok, m in zip(tokens, mask) if m == 1]
            texts_p.append(" ".join(kept) if kept else "")

        return np.array(z_primes), texts_p, tokens

    def _kernel(self, z_prime):
        """Cosine distance-based kernel."""
        # Number of absent features = proportion of zeros
        n_absent  = (1 - z_prime).sum(axis=1)
        n_total   = z_prime.shape[1]
        distances = n_absent / n_total
        return np.exp(-(distances ** 2) / (self.kernel_width / 100) ** 2)

    def explain(self, text, predict_fn, n_samples=500, n_top=5, seed=42):
        """Full text LIME pipeline."""
        rng = np.random.RandomState(seed)

        z_primes, texts_p, tokens = self._perturb_text(text, n_samples, rng)
        y_pred  = predict_fn(texts_p)
        weights = self._kernel(z_primes)

        # Weighted regression: coefficients = word importances
        surrogate = Ridge(alpha=0.01)
        surrogate.fit(z_primes, y_pred, sample_weight=weights)

        coef = surrogate.coef_

        # Fidelity
        y_surr  = surrogate.predict(z_primes)
        wss_res = np.sum(weights * (y_pred - y_surr) ** 2)
        wss_tot = np.sum(weights * (y_pred - np.average(y_pred, weights=weights)) ** 2)
        r_sq    = 1 - wss_res / (wss_tot + 1e-10)

        return {
            "tokens":     tokens,
            "coefs":      coef,
            "r_squared":  r_sq,
            "prediction": y_pred[0],
        }


# Train a simple sentiment classifier
import numpy as _np_impl
import scipy.optimize as _sp_opt

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class Ridge:
    def __init__(self, alpha=0.01, **kw): self.alpha = alpha
    def fit(self, X, y, sample_weight=None):
        X=_np_impl.atleast_2d(X); y=_np_impl.asarray(y,dtype=float)
        n,d=X.shape
        w=_np_impl.asarray(sample_weight,dtype=float) if sample_weight is not None else _np_impl.ones(n)
        w=w/w.sum()*n
        Xb=_np_impl.column_stack([X,_np_impl.ones(n)])
        A=(Xb.T*w)@Xb; A[_np_impl.arange(d),_np_impl.arange(d)]+=self.alpha
        params=_np_impl.linalg.solve(A,Xb.T@(w*y))
        self.coef_=params[:d]; self.intercept_=params[d]; return self
    def predict(self,X): return _np_impl.atleast_2d(X)@self.coef_+self.intercept_
import re as _re2
from collections import Counter as _Counter2

class TfidfVectorizer:
    def __init__(self,max_features=None,min_df=1,**kw):
        self.max_features=max_features; self.min_df=min_df
    def _tok(self,text): return _re2.findall(r'\b[a-z]+\b',text.lower())
    def fit_transform(self,texts):
        texts=list(texts); n=len(texts)
        df=_Counter2(); tokenized=[]
        for t in texts:
            toks=self._tok(t); tokenized.append(toks)
            for w in set(toks): df[w]+=1
        vocab=[w for w,c in df.items() if c>=self.min_df]
        if self.max_features: vocab=sorted(vocab,key=lambda w:-df[w])[:self.max_features]
        vocab=sorted(vocab); self._vocab=vocab; self._w2i={w:i for i,w in enumerate(vocab)}
        self._idf=_np_impl.array([_np_impl.log((n+1)/(df.get(w,0)+1))+1 for w in vocab])
        return self._build(tokenized)
    def _build(self,tokenized):
        n=len(tokenized); d=len(self._vocab); X=_np_impl.zeros((n,d))
        for i,toks in enumerate(tokenized):
            total=len(toks) or 1
            for w in toks:
                if w in self._w2i: X[i,self._w2i[w]]+=1
            X[i]/=total
        X*=self._idf
        norms=_np_impl.linalg.norm(X,axis=1,keepdims=True); norms[norms==0]=1
        return X/norms
    def transform(self,texts): return self._build([self._tok(t) for t in texts])
    def get_feature_names_out(self): return _np_impl.array(self._vocab)

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,**kw): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        X=_np_impl.atleast_2d(X); y=_np_impl.asarray(y,dtype=float)
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        X=_np_impl.atleast_2d(X); p=_sig(X@self.coef_.ravel()+self.intercept_[0])
        return _np_impl.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==_np_impl.asarray(y))

tfidf = TfidfVectorizer(max_features=200, min_df=1)
clf   = LogisticRegression(max_iter=500)
X_tfidf = tfidf.fit_transform(texts)
clf.fit(X_tfidf, labels)

def predict_proba_text(text_list):
    """Predict probability of positive class for a list of texts."""
    X_t = tfidf.transform(text_list)
    return clf.predict_proba(X_t)[:, 1]

# Explain two examples
test_texts = [
    "This is an absolutely brilliant and wonderful film",
    "Terrible and boring waste of time completely awful",
]

text_lime = TextLIMEScratch()

for test_text in test_texts:
    pred = predict_proba_text([test_text])[0]
    result = text_lime.explain(test_text, predict_proba_text, n_samples=300)

    print(f"  Text: '{test_text}'")
    print(f"  Prediction: {pred:.4f}  ({'POSITIVE' if pred > 0.5 else 'NEGATIVE'})")
    print(f"  Surrogate R² = {result['r_squared']:.4f}")
    print()
    print("  Word contributions:")

    # Sort words by importance
    word_coefs = list(zip(result['tokens'], result['coefs']))
    word_coefs.sort(key=lambda t: t[1], reverse=True)

    for word, coef in word_coefs:
        bar = "█" * int(abs(coef) / max(abs(c) for _, c in word_coefs) * 12 + 0.5)
        sign = "+" if coef > 0 else ""
        emoji = "😊" if coef > 0 else ("😞" if coef < -0.01 else "  ")
        print(f"    {emoji} '{word}': {sign}{coef:.4f}  {bar}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Using the lime library (LimeTextExplainer)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Using the lime library: LimeTextExplainer")
print("━" * 65)
print()

if HAS_LIME:
    # LimeTextExplainer wraps the above logic
    explainer_text = lime.lime_text.LimeTextExplainer(
        class_names=["negative", "positive"],
        kernel_width=25,
    )

    # Predict function must accept list of strings, return (n, n_classes) array
    def predict_proba_2class(text_list):
        X_t = tfidf.transform(text_list)
        return clf.predict_proba(X_t)   # returns both classes

    print("  Explaining with lime library:")
    print()

    for i, test_text in enumerate(test_texts):
        pred = predict_proba_text([test_text])[0]
        exp  = explainer_text.explain_instance(
            test_text,
            predict_proba_2class,
            num_features=6,
            num_samples=500,
            labels=[1],   # explain positive class
        )

        print(f"  [{i}] '{test_text}'")
        print(f"       Prediction: {pred:.4f}")
        print(f"       Intercept (baseline): {exp.intercept[1]:.4f}")
        print()
        print(f"  {'Word':<15} | {'Weight':>10} | {'Direction'}")
        print(f"  {'─'*40}")
        for word, weight in exp.as_list(label=1):
            direction = "↑ positive" if weight > 0 else "↓ negative"
            print(f"  {word:<15} | {weight:>+10.4f} | {direction}")
        print()

    # Stability analysis for text LIME
    print("  Text LIME stability (same instance, 5 runs):")
    stability_text = test_texts[0]
    all_word_ranks = []
    for seed in range(5):
        exp_s = explainer_text.explain_instance(
            stability_text, predict_proba_2class,
            num_features=4, num_samples=300, labels=[1])
        top_words = [w for w, _ in exp_s.as_list(label=1)]
        all_word_ranks.append(set(top_words))
        weights_s  = {w: wt for w, wt in exp_s.as_list(label=1)}
        print(f"    Seed {seed}: {', '.join(f'{w}({wt:.3f})' for w, wt in exp_s.as_list(label=1)[:4])}")

    # Jaccard across runs
    jac_list = [len(all_word_ranks[i] & all_word_ranks[j]) /
                len(all_word_ranks[i] | all_word_ranks[j])
                for i in range(5) for j in range(i+1, 5)]
    print(f"    Mean Jaccard stability: {np.mean(jac_list):.4f}")
    print()

else:
    print("  lime library not available")
    LIME_LIB_REF = """
  LIME LIBRARY TEXT EXAMPLE:

  from lime.lime_text import LimeTextExplainer

  explainer = LimeTextExplainer(class_names=["negative", "positive"])

  # predict_fn: takes list of strings, returns (n_samples, n_classes) array
  def predict_fn(texts):
      X = tfidf_vectorizer.transform(texts)
      return model.predict_proba(X)

  # Explain one instance
  exp = explainer.explain_instance(
      "This movie was absolutely brilliant and touching",
      predict_fn,
      num_features=6,    # number of words to highlight
      num_samples=500,   # perturbations to generate
      labels=[1],        # explain positive class
  )

  # Get explanation as list
  for word, weight in exp.as_list(label=1):
      print(f"  {word}: {weight:+.4f}")

  # Visualise in Jupyter
  exp.show_in_notebook()        # HTML rendering
  exp.as_pyplot_figure()        # matplotlib bar chart
  html = exp.as_html()          # save as HTML

  # Explain multiple classes at once
  exp_multi = explainer.explain_instance(text, predict_fn, labels=[0, 1])
  for word, w0, w1 in zip(words, exp_multi.as_list(0), exp_multi.as_list(1)):
      print(f"  {word}: neg={w0:.3f}, pos={w1:.3f}")
"""
    print(LIME_LIB_REF)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Text LIME diagnostic — when words are misleading
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Text LIME diagnostics: surrogate fidelity")
print("━" * 65)
print()

print("  LIME fidelity (R²) across different text complexities:")
print()

diagnostic_texts = [
    ("Brilliant masterpiece",                             "Short positive (simple)"),
    ("Absolutely terrible and complete waste of time",    "Short negative (simple)"),
    ("Not bad but not great either, somewhere in middle", "Ambiguous/neutral (harder)"),
    ("The film started brilliantly but quickly became "
     "boring and predictable",                            "Mixed sentiment (hardest)"),
]

print(f"  {'Text (truncated)':<40} | {'Pred':>6} | {'R²':>8} | {'Complexity'}")
print(f"  {'─'*72}")
for text, label in diagnostic_texts:
    pred = predict_proba_text([text])[0]
    r    = text_lime.explain(text, predict_proba_text, n_samples=500)
    r_sq = r['r_squared']
    flag = "✅" if r_sq > 0.8 else ("⚠️ " if r_sq > 0.6 else "❌")
    print(f"  {text[:40]:<40} | {pred:>6.3f} | {r_sq:>7.4f}{flag} | {label}")

print()
print("  R² < 0.6: the local linear model is a poor approximation.")
print("  This means LIME's explanation may not faithfully reflect the model.")
print("  Always check R² before trusting an explanation!")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Comparing LIME to global TF-IDF importance
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — LIME (local) vs TF-IDF coefficients (global)")
print("━" * 65)
print()

# Global: logistic regression coefficients on TF-IDF
vocab     = tfidf.get_feature_names_out()
lr_coefs  = clf.coef_[0]
top_pos   = np.argsort(lr_coefs)[::-1][:6]
top_neg   = np.argsort(lr_coefs)[:6]

print("  GLOBAL model weights (logistic regression coefficients):")
print(f"  Most positive (→ good reviews): {', '.join(vocab[i] for i in top_pos)}")
print(f"  Most negative (→ bad reviews):  {', '.join(vocab[i] for i in top_neg)}")
print()

# Local: LIME for a specific text
specific = "The film had brilliant visuals but terrible pacing and poor acting"
pred_s   = predict_proba_text([specific])[0]
lime_s   = text_lime.explain(specific, predict_proba_text, n_samples=500)

print(f"  Local LIME for: '{specific}'")
print(f"  Prediction: {pred_s:.4f}")
print()
print("  Global vs Local:")
print(f"  {'Word':<12} | {'Global coef':>14} | {'LIME coef':>12} | {'Agree?':>8}")
print(f"  {'─'*55}")
for word, lime_coef in zip(lime_s['tokens'], lime_s['coefs']):
    if word in vocab:
        idx        = list(vocab).index(word)
        global_c   = lr_coefs[idx]
        agree      = "✅" if np.sign(global_c) == np.sign(lime_coef) or abs(lime_coef) < 0.01 else "⚠️ "
        print(f"  {word:<12} | {global_c:>+14.4f} | {lime_coef:>+12.4f} | {agree}")

print()
print("  Global weights reflect average model behaviour across all texts.")
print("  LIME weights reflect the model's sensitivity AT THIS SPECIFIC TEXT.")
print("  They can disagree when the model behaves differently in this local region.")
''',
    },

    "3 · LIME for Images and Production Considerations": {
        "description": (
            "LIME for image classification with superpixel segmentation. "
            "Implement image perturbation and superpixel greying manually. "
            "Show how superpixel count affects explanation granularity. "
            "Detect Clever Hans behaviour in an image classifier. "
            "LIME production patterns: batch explanation, monitoring, caching. "
            "LIME vs SHAP decision guide with concrete examples. "
            "Common mistakes and how to avoid them."
        ),
        "language": "python",
        "code": r'''
import numpy as _np_impl
import scipy.optimize as _sp_opt

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class Ridge:
    def __init__(self, alpha=0.01, **kw): self.alpha = alpha
    def fit(self, X, y, sample_weight=None):
        X=_np_impl.atleast_2d(X); y=_np_impl.asarray(y,dtype=float)
        n,d=X.shape
        w=_np_impl.asarray(sample_weight,dtype=float) if sample_weight is not None else _np_impl.ones(n)
        w=w/w.sum()*n
        Xb=_np_impl.column_stack([X,_np_impl.ones(n)])
        A=(Xb.T*w)@Xb; A[_np_impl.arange(d),_np_impl.arange(d)]+=self.alpha
        params=_np_impl.linalg.solve(A,Xb.T@(w*y))
        self.coef_=params[:d]; self.intercept_=params[d]; return self
    def predict(self,X): return _np_impl.atleast_2d(X)@self.coef_+self.intercept_
import numpy as np
import time

print("=" * 65)
print("  LIME FOR IMAGES AND PRODUCTION CONSIDERATIONS")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Image LIME superpixel segmentation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Image LIME: superpixel segmentation and perturbation")
print("━" * 65)
print()

print("  Image LIME overview:")
print()
IMAGE_LIME_OVERVIEW = """
  Why we need superpixels:
  ─────────────────────────────────────────────────────────────────────
  A 224×224×3 image has 150,528 pixels.
  Making each pixel a binary feature → 2^150528 possible perturbations.
  We need a manageable number of meaningful features.

  Superpixels solve this:
  ─────────────────────────────────────────────────────────────────────
  Group semantically coherent pixels into ~50-200 regions.
  Each region = one binary LIME feature.
  A 224×224 image with 50 superpixels → 2^50 possible combinations.
  With N=1000 samples: sparse but sufficient coverage.

  Superpixel algorithms:
  ─────────────────────────────────────────────────────────────────────
  SLIC  (Simple Linear Iterative Clustering):
      k-means in (L,a,b,x,y) colour+position space.
      Parameters: n_segments (number of superpixels), compactness
      Fast and produces regular-shaped segments.
      Default in lime.lime_image.

  Quickshift:
      Density-based clustering in colour+position space.
      Produces irregular segments following image edges.
      Parameters: kernel_size, max_dist, ratio

  Felzenszwalb:
      Graph-based segmentation.
      Produces fewer, larger superpixels.
      Parameters: scale, sigma, min_size

  Perturbation for absent superpixels (z'_j = 0):
  ─────────────────────────────────────────────────────────────────────
  GREY (128, 128, 128):  most common, simple
  GAUSSIAN BLUR of region: more realistic (image content averaged)
  INPAINTING: fill from surrounding context (realistic but slow)
  RANDOM NOISE: aggressive but tests model robustness
"""
print(IMAGE_LIME_OVERVIEW)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Synthetic image LIME demonstration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Synthetic image LIME: patch-based explanation")
print("━" * 65)
print()

# Create a synthetic image with controllable "important regions"
# 8×8 grid of patches, each patch is a "superpixel"
GRID_SIZE   = 4   # 4×4 grid = 16 patches/superpixels
IMG_SIZE    = 32  # 32×32 pixels total
PATCH_SIZE  = IMG_SIZE // GRID_SIZE

def make_synthetic_image():
    """
    Create a 32×32 synthetic image where:
    - Top-left quadrant (patches 0-3) = "signal" (bright)
    - Other regions = noise
    """
    img = np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 0.3   # background noise (dim)
    # Top-left 2×2 patches are bright (signal)
    img[:IMG_SIZE//2, :IMG_SIZE//2, :] += 0.6
    return np.clip(img, 0, 1)

def segment_image(img, grid_size=GRID_SIZE):
    """Simple grid-based segmentation (like SLIC but uniform grid)."""
    segments = np.zeros((IMG_SIZE, IMG_SIZE), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            r0 = i * PATCH_SIZE
            r1 = (i + 1) * PATCH_SIZE
            c0 = j * PATCH_SIZE
            c1 = (j + 1) * PATCH_SIZE
            segments[r0:r1, c0:c1] = i * grid_size + j
    return segments

def synthetic_classifier(images):
    """
    Black-box image classifier.
    Predicts based ONLY on the brightness of the top-left quadrant.
    (Ground truth: patches 0,1,4,5 are important)
    """
    preds = []
    for img in images:
        # Signal = mean brightness of top-left quadrant
        signal = img[:IMG_SIZE//2, :IMG_SIZE//2, :].mean()
        # Non-linear threshold + some noise
        logit  = 8.0 * (signal - 0.5) + np.random.normal(0, 0.1)
        preds.append(1 / (1 + np.exp(-logit)))
    return np.array(preds)

# Generate test image
test_image = make_synthetic_image()
segments   = segment_image(test_image)
n_segs     = GRID_SIZE * GRID_SIZE

true_pred  = synthetic_classifier([test_image])[0]
print(f"  Synthetic 32×32 image, {n_segs} patches (4×4 grid)")
print(f"  Ground truth: top-left 2×2 patches are the signal region")
print(f"  Classifier prediction: {true_pred:.4f}")
print()

# Image LIME from scratch
class ImageLIMEScratch:
    def __init__(self, image, segments, hide_color=0.5):
        self.image      = image
        self.segments   = segments
        self.n_segs     = segments.max() + 1
        self.hide_color = hide_color   # grey = 0.5

    def _perturb(self, n_samples=500, rng=None):
        """Generate perturbed images by greying out superpixels."""
        if rng is None: rng = np.random.RandomState(42)
        z_primes = []
        images_p = []

        # Original (all patches present)
        z_primes.append(np.ones(self.n_segs))
        images_p.append(self.image.copy())

        for _ in range(n_samples):
            mask  = rng.randint(0, 2, self.n_segs)
            z_primes.append(mask.astype(float))
            img_p = self.image.copy()
            # Grey out absent patches
            for seg_id in range(self.n_segs):
                if mask[seg_id] == 0:
                    img_p[self.segments == seg_id] = self.hide_color
            images_p.append(img_p)

        return np.array(z_primes), images_p

    def explain(self, predict_fn, n_samples=500, seed=42):
        rng = np.random.RandomState(seed)

        z_primes, images_p = self._perturb(n_samples, rng)
        y_pred = predict_fn(images_p)

        # Kernel weights: fewer absent patches = closer to original
        n_absent  = (1 - z_primes).sum(axis=1)
        distances = n_absent / self.n_segs
        sigma     = 0.25
        weights   = np.exp(-(distances ** 2) / sigma ** 2)

        # Weighted regression
        surrogate = Ridge(alpha=0.01)
        surrogate.fit(z_primes, y_pred, sample_weight=weights)

        y_surr  = surrogate.predict(z_primes)
        wss_res = np.sum(weights * (y_pred - y_surr)**2)
        wss_tot = np.sum(weights * (y_pred - np.average(y_pred, weights=weights))**2)
        r_sq    = 1 - wss_res / (wss_tot + 1e-10)

        return {
            "coefs":       surrogate.coef_,
            "r_squared":   r_sq,
            "prediction":  y_pred[0],
        }


image_explainer = ImageLIMEScratch(test_image, segments)
result = image_explainer.explain(synthetic_classifier, n_samples=500)

print(f"  LIME explanation:")
print(f"  Surrogate R² = {result['r_squared']:.4f}  "
      f"({'Good' if result['r_squared'] > 0.7 else 'Poor'})")
print()

# Interpret patch importance
print(f"  Patch importance (4×4 grid, top-left = patch 0):")
print()
coef_grid = result['coefs'].reshape(GRID_SIZE, GRID_SIZE)
for row in range(GRID_SIZE):
    cells = []
    for col in range(GRID_SIZE):
        c    = coef_grid[row, col]
        char = "█" if c > 0.05 else ("░" if c > -0.05 else "▒")
        cells.append(f"{char}({c:+.2f})")
    print("    " + "  ".join(cells))

print()
# Check if LIME correctly identified the signal patches
signal_patches     = [0, 1, GRID_SIZE, GRID_SIZE+1]   # top-left 2×2
signal_importance  = result['coefs'][signal_patches].mean()
noise_importance   = result['coefs'][[i for i in range(n_segs) if i not in signal_patches]].mean()
print(f"  Mean importance of TRUE signal patches: {signal_importance:+.4f}")
print(f"  Mean importance of noise patches:       {noise_importance:+.4f}")
print(f"  LIME correctly identified signal: {'✅' if signal_importance > noise_importance else '❌'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Production considerations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — LIME in production: patterns and pitfalls")
print("━" * 65)
print()

PRODUCTION_GUIDE = """
  LIME PRODUCTION PATTERNS:

  ── 1. Caching ───────────────────────────────────────────────────────────
  LIME is expensive (N model calls per explanation).
  Cache explanations keyed on (instance_hash, model_version, params).

  import hashlib, json, pickle, redis

  def lime_with_cache(x, predict_fn, cache_client, n_samples=1000):
      key = hashlib.md5(json.dumps({
          "x": x.tolist(),
          "n_samples": n_samples,
      }).encode()).hexdigest()

      if (cached := cache_client.get(key)):
          return pickle.loads(cached)

      explanation = lime_explainer.explain_instance(...)
      cache_client.setex(key, 3600, pickle.dumps(explanation))   # 1hr TTL
      return explanation

  ── 2. Stability Monitoring ──────────────────────────────────────────────
  Track Jaccard stability across production explanations.

  from collections import deque
  stability_buffer = deque(maxlen=1000)   # last 1000 explanations

  def explain_with_monitoring(x, predict_fn):
      exps = [lime_explain(x, seed=i) for i in range(3)]
      top_features = [set(get_top_k(e)) for e in exps]
      jaccard = compute_mean_jaccard(top_features)
      stability_buffer.append(jaccard)

      if np.mean(stability_buffer) < 0.7:
          alert("LIME stability degraded — check model or data distribution")

      return exps[0]   # return first run explanation

  ── 3. Fidelity Gating ───────────────────────────────────────────────────
  Only serve explanations where R² exceeds a threshold.

  def safe_explain(x, predict_fn, min_r2=0.6):
      exp = lime_explain(x, predict_fn)
      if exp.score < min_r2:
          return {
              "explanation": None,
              "message": "Explanation not reliable for this input (R²={:.2f})".format(exp.score),
              "fallback": "global_feature_importance"
          }
      return {"explanation": exp, "r2": exp.score}

  ── 4. Parallel Explanation ──────────────────────────────────────────────
  For batch explanations, parallelise across instances.

  from concurrent.futures import ProcessPoolExecutor

  def explain_batch(instances, predict_fn, max_workers=4):
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
          futures = [executor.submit(lime_explain, x, predict_fn)
                     for x in instances]
      return [f.result() for f in futures]

  ── 5. Adverse Action Notices (ECOA/GDPR) ────────────────────────────────
  For regulated lending/insurance: LIME top negative features = denial reasons.

  def generate_adverse_action(x, predict_fn):
      exp  = lime_explain(x, predict_fn)
      # Top NEGATIVE coefficients = what hurt the applicant most
      denial_reasons = [(fname, coef) for fname, coef in zip(feature_names, exp.coef_)
                        if coef < 0]
      denial_reasons.sort(key=lambda t: t[1])   # most negative first

      # Map to regulatory reason codes
      reasons = []
      for fname, coef in denial_reasons[:4]:
          code   = reason_code_mapping.get(fname, "99-OTHER")
          human  = human_readable_mapping.get(fname, fname)
          reasons.append({"code": code, "reason": human, "impact": abs(coef)})

      return {
          "decision": "DENIED",
          "reasons": reasons,   # required by ECOA
          "explanation_method": "LIME",
          "model_version": MODEL_VERSION,
      }
"""
print(PRODUCTION_GUIDE)

# Timing analysis
print("  LIME performance analysis (synthetic, tabular model):")
print()

def dummy_model(X):
    """Fast placeholder model for timing."""
    return 1 / (1 + np.exp(-X[:, 0] * 2 + X[:, 1]))

X_dummy     = np.random.randn(100, 10)
x_time_test = X_dummy[0]

print(f"  {'N samples':>10} | {'Time (s)':>10} | {'ms/explanation':>16} | {'Stability est.':>16}")
print(f"  {'─'*60}")
for N in [200, 500, 1000, 2000, 5000]:
    times = []
    for _ in range(3):
        t0   = time.perf_counter()
        # Simulate LIME: N model calls
        dummy_model(np.random.randn(N, 10))
        times.append(time.perf_counter() - t0)
    t_mean = np.mean(times)
    stability = "Low" if N < 500 else ("Medium" if N < 2000 else "High")
    print(f"  {N:>10} | {t_mean:>10.4f} | {t_mean*1000:>16.2f} | {stability:>16}")

print()
print("━" * 65)
print("  LIME COMMON MISTAKES AND HOW TO AVOID THEM")
print("━" * 65)
print()
MISTAKES = """
  MISTAKE 1: Not checking R² (fidelity)
  Fix: Always report exp.score. If R² < 0.6, don't serve the explanation.
       Or: increase N, reduce σ, or use SHAP instead.

  MISTAKE 2: Using the same N=300 for all model types
  Fix: Image models need N≥500 (more superpixels = more complex space).
       Text with long documents needs N≥2000.
       Tabular with many features needs N≥2000.

  MISTAKE 3: Reporting a single LIME run as definitive
  Fix: Always run 3-10 times and report mean ± std of coefficients.
       Or: report Jaccard stability score alongside the explanation.

  MISTAKE 4: Confusing local LIME for global importance
  Fix: LIME tells you about ONE prediction, not the model overall.
       For global importance: use SHAP summary plot or permutation importance.

  MISTAKE 5: Using LIME for correlated features (tabular)
  Fix: Check correlation matrix first. If r > 0.7 between features,
       use TreeSHAP or report that correlated features may split credit arbitrarily.

  MISTAKE 6: Not setting seed for reproducibility
  Fix: Always set seed for demonstrations and stored explanations.
       For production: use system entropy but log the seed used.

  MISTAKE 7: Trusting LIME for complex non-linear regions
  Fix: Check R² AND verify by manually toggling top features.
       If removing top-1 feature doesn't change prediction much: R² lied.
"""
print(MISTAKES)
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