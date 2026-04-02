"""
SHAP — SHapley Additive exPlanations
======================================

SHAP is the most theoretically rigorous and practically useful framework
for explaining machine learning model predictions. It unifies several
existing explanation methods under a single axiomatic framework rooted
in cooperative game theory, and provides the only attribution method that
simultaneously satisfies all four properties a fair explanation should have.

Developed by Scott Lundberg and Su-In Lee (2017), SHAP answers a deceptively
simple question: "How much did each feature contribute to this specific
prediction?" The answer it gives is not ad-hoc — it is the unique solution
guaranteed by mathematical theorems to be the only fair allocation of credit
among contributing factors.

In practice, SHAP has become the gold standard for production model
explainability. It is built into XGBoost, LightGBM, and CatBoost natively.
It powers model explainability at Microsoft, LinkedIn, Uber, JPMorgan, and
thousands of regulated industries. GDPR-compliant credit scoring systems use
SHAP to generate the legally required "meaningful information about the logic"
of automated decisions. Medical AI systems use SHAP to help clinicians
understand and override model recommendations.

This module covers SHAP from first principles — the cooperative game theory
foundation, the four axioms, the mathematical derivation of Shapley values,
every major algorithmic variant (KernelSHAP, TreeSHAP, DeepSHAP,
GradientSHAP, LinearSHAP, PartitionSHAP), every visualization type, and the
practical considerations that separate correct SHAP usage from common mistakes.

"""

import textwrap
import re

TOPIC_NAME = "SHAP — SHapley Additive exPlanations"
DISPLAY_NAME = "03a · SHAP"
ICON = "🎯"
SUBTITLE = "Shapley Values, Axioms, TreeSHAP, and Production Explainability"

THEORY = """

##### PART 1 — THE PROBLEM: ATTRIBUTING CREDIT IN A TEAM EFFORT

### The Feature Attribution Problem

    A model receives input x = [x₁, x₂, ..., xₙ] and produces prediction f(x).
    We want to assign a score φⱼ to each feature j such that:
        • φⱼ reflects how much feature j contributed to f(x)
        • The scores are "fair" in a well-defined mathematical sense
        • The scores sum to something meaningful

    This sounds simple but has deep difficulties:

    INTERACTION EFFECTS:
        If income=High AND credit=Good → approved (both features together matter)
        If income=High alone → borderline  (income alone less decisive)
        If credit=Good alone → borderline  (credit alone less decisive)
        How much credit does income deserve? How much does credit_score deserve?
        The answer depends on what we mean by "contribution."

    REDUNDANCY:
        If feature A and feature B are perfectly correlated, they carry the
        same information. Naively, both might get high attribution. But the
        model doesn't need both — how do we split credit fairly?

    BASELINE DEPENDENCE:
        "Feature j contributed X to the prediction" is meaningless without a
        reference point. Contributed X compared to WHAT?
        The baseline (the "nothing" state) must be defined.

### Why Existing Methods Failed

    Before SHAP, common approaches had fatal flaws:

    GRADIENT-BASED (∂f/∂xⱼ):
        Measures local sensitivity, not contribution.
        A feature can have zero gradient but high importance (saturation).
        Doesn't sum to the prediction.

    INPUT × GRADIENT (xⱼ × ∂f/∂xⱼ):
        Better, but violates symmetry — two identical features can get
        different attributions based on implementation order.

    PERMUTATION IMPORTANCE:
        Global, not local — can't explain individual predictions.
        Sensitive to feature correlations (double-counts correlated features).

    ATTENTION WEIGHTS:
        Not faithful — adversarial distributions produce same output but
        completely different attention patterns (Jain & Wallace, 2019).

    What was needed: a principled framework with mathematical guarantees.

### Cooperative Game Theory: The Foundation

    SHAP borrows from cooperative game theory (Lloyd Shapley, Nobel Prize 2012).

    THE COALITION GAME:
        n players form coalitions to earn payouts.
        Question: what is the fair share of the total payout for each player?

        Example: three investors (A, B, C) form a company.
            {A} alone earns:     $100K
            {B} alone earns:     $200K
            {A,B} together earn: $500K  (synergy!)
            {C} alone earns:     $50K
            {A,C} together:      $200K
            {B,C} together:      $400K
            {A,B,C} together:    $700K

        How do we fairly divide the $700K among A, B, and C?
        This is EXACTLY the feature attribution problem, where:
            players     = features
            coalitions  = subsets of features
            payout      = model prediction with only those features known


##### PART 2 — SHAPLEY VALUES: THE MATHEMATICAL DERIVATION

### The Shapley Value Formula

    For a model f and instance x with n features, the Shapley value for feature j is:

    φⱼ(f, x) = Σ_{S ⊆ N\\{j}} [ |S|!(n-|S|-1)!/n! ] × [f(x_{S∪{j}}) - f(x_S)]

    where:
        N = {1, 2, ..., n}  (set of all features)
        S = a subset of N that does NOT include feature j
        f(x_S) = model prediction when only features in S are "present"
                 (features not in S are replaced by a baseline/background)
        The weight |S|!(n-|S|-1)!/n! is the probability that a uniformly
        random permutation of features would have feature j joining coalition S

    PLAIN ENGLISH INTERPRETATION:
        Consider all possible orderings of features.
        In each ordering, when feature j "joins" the partial coalition formed
        by the features ordered before it, it contributes f(S∪{j}) - f(S).
        The Shapley value is the AVERAGE marginal contribution of feature j
        over ALL possible orderings.

### Worked Example: 3 Features

    Features: income (x₁), credit_score (x₂), debt_ratio (x₃)
    Instance: x = [0.7, 0.8, 0.3]
    Baseline: x̄ = [0.5, 0.5, 0.5]

    All orderings for feature j=income (x₁):
    There are 3! = 6 orderings of {x₁, x₂, x₃}:

    [x₁, x₂, x₃]: income joins empty set {}
                   Marginal: f(x₁, x̄₂, x̄₃) - f(x̄₁, x̄₂, x̄₃)

    [x₁, x₃, x₂]: income joins empty set {}  (same as above)

    [x₂, x₁, x₃]: income joins {x₂}
                   Marginal: f(x₁, x₂, x̄₃) - f(x̄₁, x₂, x̄₃)

    [x₃, x₁, x₂]: income joins {x₃}
                   Marginal: f(x₁, x̄₂, x₃) - f(x̄₁, x̄₂, x₃)

    [x₂, x₃, x₁]: income joins {x₂, x₃}
                   Marginal: f(x₁, x₂, x₃) - f(x̄₁, x₂, x₃)

    [x₃, x₂, x₁]: income joins {x₂, x₃}  (same coalition, same marginal)

    φ₁ = (1/6)[marginal_1 + marginal_1 + marginal_2 + marginal_3
                + marginal_4 + marginal_4]
       = (1/3)marginal_1 + (1/6)marginal_2 + (1/6)marginal_3 + (1/3)marginal_4

    This is equivalent to the subset formula with weights:
        |S|=0: weight = 0!×2!/3! = 2/6 = 1/3
        |S|=1: weight = 1!×1!/3! = 1/6
        |S|=2: weight = 2!×0!/3! = 2/6 = 1/3

### The Four Shapley Axioms

    These four properties UNIQUELY characterise the Shapley value.
    No other attribution function satisfies all four simultaneously.

    1. EFFICIENCY (Completeness / Summation):
       Σⱼ φⱼ = f(x) - f(x̄)
       The Shapley values sum to the difference between the prediction
       and the baseline prediction. All credit is accounted for exactly.
       → Verification: sum(shap_values) + expected_value = model_prediction

    2. SYMMETRY:
       If f(S ∪ {i}) = f(S ∪ {j}) for all S ⊆ N\\{i,j}
       then φᵢ = φⱼ
       Two features that contribute equally to every coalition receive equal values.
       Fairness: the attribution doesn't depend on feature naming or order.

    3. DUMMY (Null Player):
       If f(S ∪ {j}) = f(S) for all S ⊆ N\\{j}
       then φⱼ = 0
       A feature that never changes the prediction gets zero attribution.
       Doesn't matter if it's present or absent.

    4. LINEARITY (Additivity):
       φⱼ(f + g) = φⱼ(f) + φⱼ(g)
       If the model is a sum of two models, the Shapley values add up.
       φⱼ(αf) = α·φⱼ(f)  (scaling linearity also holds)

    UNIQUENESS THEOREM (Shapley, 1953):
    The Shapley value is the UNIQUE attribution function satisfying all four axioms.

### Understanding the Baseline

    f(x_S) requires defining what happens to features NOT in S.
    This is the BASELINE or BACKGROUND distribution.

    SINGLE BASELINE:
        Replace absent features with a fixed reference point.
        Example: all zeros, feature means, or a specific "normal" instance.
        φⱼ then measures: "contribution of feature j relative to the baseline."
        E[f(X)] baseline: "how much does feature j shift the prediction from average?"

    DISTRIBUTION BASELINE (KernelSHAP default):
        Replace absent features with random samples from training data.
        f(x_S) = E_{x̄∼D}[f(x_j present, x̄_j absent)]
        This marginalises over the empirical distribution.
        More expensive but avoids "unrealistic" feature combinations.

    INTERVENTIONAL vs OBSERVATIONAL SHAP:
        Interventional: f(x_S) = E[f(X) | X_S = x_S] (condition on S)
        Observational:  f(x_S) = E_{X∼P}[f(X) with features S fixed]
        TreeSHAP supports both; KernelSHAP uses interventional by default.

    CRITICAL: Always report which baseline you used.
    Different baselines give different (but each internally consistent) SHAP values.


##### PART 3 — KERNEL SHAP: MODEL-AGNOSTIC ESTIMATION

### KernelSHAP: Framing as Weighted Regression

    Computing exact Shapley values requires 2ⁿ model evaluations (all subsets).
    For n=20: 1M evaluations. For n=100: 10³⁰ evaluations. Infeasible.

    KernelSHAP (Lundberg & Lee, 2017) observes that the Shapley values are
    the coefficients of a specific weighted linear regression:

        g(z') = φ₀ + Σⱼ φⱼ z'ⱼ

    where z' ∈ {0,1}ⁿ is the binary "on/off" feature mask,
    and the regression is fit with the Shapley kernel weights:

        π(z') = (n-1) / [C(n, |z'|) × |z'| × (n - |z'|)]

    SPECIAL CASES:
        z' = 0 (no features): weight → ∞  (must predict f(x̄))
        z' = 1 (all features): weight → ∞ (must predict f(x))
        These boundary conditions enforce the efficiency axiom.

    ALGORITHM:
        1. Sample M subsets z'₁, ..., z'_M randomly (from the distribution π)
        2. Convert each z'ᵢ to actual features: z'ᵢ=1 → use xⱼ, z'ᵢ=0 → sample from D
        3. Get model predictions: yᵢ = f(zᵢ)
        4. Fit weighted regression: min_φ Σᵢ πᵢ (yᵢ - g(z'ᵢ))²
        5. The coefficients φ are the SHAP values

    COMPLEXITY: O(M × cost_of_model_evaluation)
        M=2048 is usually sufficient for accurate estimates.
        More features → need more samples for good approximation.

### KernelSHAP Convergence and Accuracy

    M samples controls the accuracy-speed tradeoff:
        M=100:  rough estimates, fast (good for interactive exploration)
        M=1000: decent estimates (sufficient for most use cases)
        M=5000: high accuracy (for final explanations in production)

    Convergence check: run KernelSHAP with M and 2M samples.
    If SHAP values change significantly: need more samples.

    For features with strong interactions: convergence is slower.
    For approximately additive models: few samples needed.

### KernelSHAP vs Permutation Sampling

    Original KernelSHAP: sample subsets from the Shapley kernel distribution.
    Permutation sampling (an alternative): sample random ORDERINGS of features,
    compute cumulative marginal contributions along each ordering.

    Both are unbiased estimators of the true Shapley values.
    Permutation sampling can be more efficient when features are many but
    interactions are sparse (most marginals are zero).


##### PART 4 — TREE SHAP: EXACT AND EFFICIENT FOR TREE ENSEMBLES

### Why Trees Are Special

    Decision trees have a structure that allows EXACT Shapley computation
    without exponential cost. TreeSHAP (Lundberg et al., 2018) runs in
    O(TLD²) time where T=trees, L=max leaves, D=max depth.

    For a Random Forest with 100 trees, each depth 20:
        KernelSHAP: ~2000 model evaluations per sample
        TreeSHAP:   O(100 × 2²⁰ × 20²) ≈ but in practice sub-second per sample
                    because the tree structure prunes the computation dramatically.

### How TreeSHAP Works

    For a SINGLE decision tree, consider explaining feature j's contribution.

    A sample x follows a specific PATH through the tree based on feature values.
    The Shapley computation asks: what would happen if feature j were "absent"?

    TreeSHAP computes f(x_S) by pushing x through the tree but at any node
    that splits on a feature NOT in S, splitting the data according to the
    fraction of training samples that went left and right at that node.

    ALGORITHM (simplified):
        recurse(node, path_weights):
            if leaf node:
                update SHAP values using accumulated path weights
            else:
                hot_child   = child on the path of x (x's feature value routes here)
                cold_child  = the other child

                # For features IN the coalition: go hot child (follow x)
                # For features NOT in coalition: split flow left/right proportionally

                Extend path_weights for this split
                recurse(hot_child, ...)
                recurse(cold_child, ...)

    The path_weights accumulate the Shapley kernel weighting automatically
    as the algorithm descends the tree.

### Interventional vs Observational TreeSHAP

    INTERVENTIONAL (default in shap library):
        f(x_S) = E_X[f(X) | X_S = x_S]
        Conditions only on features in S, marginalises over others.
        Computationally: uses the empirical distribution at each node
        to split the flow when a feature is absent.

    OBSERVATIONAL (path-dependent, tree_path_dependent=True):
        f(x_S) = E[f(X) | X_S = x_S, X follows the data distribution]
        Accounts for feature correlations — correlated features share credit.
        Generally: less blame on features that co-occur with used features.

    Difference in practice:
        For independent features: identical results.
        For highly correlated features: can differ significantly.
        OBSERVATIONAL is more appropriate for correlated real-world data.
        INTERVENTIONAL is the theoretically "correct" SHAP formulation.

### SHAP Interaction Values (TreeSHAP Extension)

    Beyond main effects, TreeSHAP computes SHAP interaction values:
        φᵢⱼ = contribution from the interaction between features i and j

    Properties:
        Diagonal: φᵢᵢ = main effect of feature i (after removing interactions)
        Off-diagonal: φᵢⱼ = φⱼᵢ  (symmetric)
        Sum: Σᵢ Σⱼ φᵢⱼ = φᵢ  (interactions sum to main SHAP value)

    Interpretation: φᵢⱼ measures how much the combination of features i and j
    contributes ABOVE the sum of their individual main effects.

    Positive interaction: i and j together contribute MORE than i alone + j alone.
    Negative interaction: i and j together contribute LESS (diminishing returns).

    Computed via: shap.TreeExplainer(model).shap_interaction_values(X)
    Shape: (n_samples, n_features, n_features)


##### PART 5 — SHAP FOR NEURAL NETWORKS: DEEP SHAP AND GRADIENT SHAP

### DeepSHAP (DeepLIFT + Shapley)

    DeepSHAP uses DeepLIFT's backpropagation rules to estimate SHAP values
    efficiently for deep networks.

    DeepLIFT (Shrikumar et al., 2017) propagates "contribution scores"
    backwards through the network:
        For each neuron: compute the difference-from-reference
        δxⱼ = xⱼ - x̄ⱼ  (actual minus baseline activation)
        Distribute this difference back to inputs proportionally

    DeepSHAP = DeepLIFT with Shapley kernel weighting.

    The connection to SHAP:
        When the baseline x̄ is sampled from the background distribution,
        and contributions are averaged over many baselines, DeepSHAP
        approximates the true Shapley values.

    Usage:
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(X_test)

    LIMITATIONS:
        Approximate — not guaranteed to equal true Shapley values.
        Doesn't handle all non-linearities perfectly (e.g., batch norm).
        Best for: feedforward networks with ReLU activations.

### GradientSHAP

    Combines Integrated Gradients with SHAP sampling:
        For each test instance x:
            1. Sample a background instance x̄ from the background distribution
            2. Sample a random α ∈ [0, 1]
            3. Compute gradient at x̄ + α(x - x̄)
            4. Attribution: (x - x̄) × gradient
            5. Average over many (x̄, α) pairs

    GradientSHAP ≈ Shapley values when:
        The function is approximately linear along each path from x̄ to x.
        Many (x̄, α) samples are used.

    MORE PRINCIPLED than raw gradients.
    FASTER than KernelSHAP for neural networks.
    NOT as exact as TreeSHAP for tree models.

### Linear SHAP

    For linear models (logistic regression, linear regression):
        φⱼ = wⱼ × (xⱼ - E[xⱼ])

    Where wⱼ is the model coefficient and E[xⱼ] is the mean of feature j.

    This is EXACT (not an approximation) when features are independent.
    When features are correlated: LinearSHAP accounts for correlations
    by adjusting for the feature covariance structure.

    Extremely fast — closed-form solution, no sampling needed.


##### PART 6 — SHAP VISUALISATIONS: READING EVERY PLOT TYPE

### Waterfall Plot (Single Prediction)

    Shows how features push the prediction from the baseline to the final output.

    Reading the waterfall:
        Bottom: baseline (E[f(X)] — average model output)
        Top:    final prediction f(x)
        Each bar: one feature's SHAP contribution
        Red bars:  positive contribution (push prediction UP)
        Blue bars: negative contribution (push prediction DOWN)
        Width:     magnitude of contribution
        Order:     features sorted by |SHAP value| (most influential first)

    Mathematical check: sum(all bars) = f(x) - E[f(X)]

    When to use: explaining ONE specific prediction to a stakeholder.
    Best for: loan officer explaining why an application was denied.

### Force Plot (Single or Multiple Predictions)

    A horizontal version of the waterfall. Shows forces "pushing" the
    prediction left (toward 0) or right (toward 1 for probability).

    Single prediction: baseline → final prediction
    Multiple predictions: can stack many force plots to show patterns.

    SHAP JavaScript widget: interactive — hover to see feature values.
    Used in production dashboards and Jupyter notebooks.

### SHAP Summary Plot (Global, All Features and Samples)

    The most informative global SHAP visualization.

    Each point represents one sample × one feature.
    X-axis: SHAP value (positive = pushes prediction up, negative = down)
    Y-axis: feature (ranked by mean |SHAP|, most important at top)
    Color: actual feature value (red=high, blue=low, using a gradient)

    What you can read from the summary plot:
        IMPORTANCE: features at top have highest global impact.
        DIRECTION: points on right = positive impact, left = negative impact.
        DISTRIBUTION: spread of points shows variability of impact.
        VALUE RELATIONSHIP: red points on right = high values → positive impact.

    Example interpretations:
        "Red points (high income) are on the right (positive SHAP): higher income
         pushes approval probability up."
        "Red points (high debt_ratio) are on the left: high debt pushes approval down."
        "Both red and blue for age_at_account: non-monotone, U-shaped relationship."

### Dependence Plot (One Feature, with Interaction)

    X-axis: actual value of feature j
    Y-axis: SHAP value for feature j
    Color: value of the feature that INTERACTS MOST with feature j

    Reading the dependence plot:
        Slope: positive slope = generally positive effect of feature j.
        Scatter/spread: the vertical spread is the interaction effect.
        Color pattern: if color aligns with vertical spread → the colored
                       feature explains the interaction.

    Example:
        X-axis: income
        Y-axis: SHAP(income)
        Color: credit_score
        Pattern: income has high positive SHAP but ONLY when credit is good (red).
                 Low credit (blue) flattens the income effect.
        Interpretation: income and credit have a synergistic interaction.

### Heatmap Plot (Multiple Features, Multiple Samples)

    Rows: features (ordered by clustering of SHAP patterns)
    Columns: samples (ordered by some criteria — prediction, feature value)
    Color: SHAP value at that (feature, sample) cell

    Reveals clusters of similar explanations:
        Group of samples with similar SHAP patterns = similar explanatory profile.
        Useful for: finding distinct "explanatory archetypes" in your data.

### Bar Plot (Global Feature Importance)

    Simple ranking: mean |SHAP value| per feature.
    Sometimes split into positive and negative contributions.
    Easiest to communicate to non-technical audiences.
    Loses distributional information (hides non-linearities).


##### PART 7 — SHAP IN PRODUCTION: BEST PRACTICES AND PITFALLS

### Choosing the Right SHAP Explainer

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Model type           │ Best SHAP method   │ Exact? │ Speed              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Linear models        │ LinearSHAP         │ Yes    │ Instant            │
    │ Tree ensembles (RF)  │ TreeSHAP           │ Yes    │ Very fast          │
    │ XGBoost/LightGBM     │ TreeSHAP (built-in)│ Yes    │ Very fast          │
    │ Neural nets (PyTorch)│ GradientSHAP       │ Approx │ Fast               │
    │ Neural nets (TF)     │ DeepSHAP           │ Approx │ Fast               │
    │ Any model (black-box)│ KernelSHAP         │ Approx │ Slow (M×inference) │
    │ Any model (fast)     │ PartitionSHAP      │ Approx │ Medium             │
    └─────────────────────────────────────────────────────────────────────────┘

### Background Data Selection

    The background dataset is critical for KernelSHAP and GradientSHAP.

    FULL TRAINING SET: most accurate but slow (E computed over all samples).
    k-MEANS SUMMARY (shap.kmeans(X_train, k=100)): fast approximation.
    RANDOM SAMPLE (shap.sample(X_train, 100)): simple fast approximation.
    SINGLE REFERENCE: fastest but most sensitive to baseline choice.

    Rule of thumb:
        k=50–200 background samples is usually sufficient.
        More background samples → slower but more stable SHAP values.

### Correlated Features and SHAP Instability

    When features are highly correlated (r > 0.8):
        INTERVENTIONAL SHAP: assigns credit to one and near-zero to the other
                             (which one gets credit can be arbitrary).
        OBSERVATIONAL SHAP: distributes credit across correlated features.

    Best practice for correlated features:
        1. Use TreeSHAP with path-dependent (observational) mode.
        2. Or: combine correlated features into a single feature first.
        3. Report both methods and note the difference.

### Efficiency Verification (Always Do This)

    After computing SHAP values, ALWAYS verify:
        shap_sum = shap_values.sum(axis=1) + explainer.expected_value
        assert np.allclose(shap_sum, model.predict_proba(X)[:, 1], atol=1e-4)

    If verification fails:
        Wrong explainer for the model type.
        Background distribution mismatch.
        Multi-class confusion (using wrong class index).

### Multi-Class Classification

    For k-class problems, SHAP returns k sets of values:
        shap_values = explainer.shap_values(X)
        # shap_values[c] has shape (n_samples, n_features) for class c

    φⱼ,c(x) = contribution of feature j to the probability of class c.
    Sum: Σc shap_values[c][sample] = 0 for each (sample, feature) pair.
         (credit for one class = debit for another — probabilities sum to 1)

    Common mistake: using shap_values[1] for a 2-class model but plotting it
    as if it represents absolute importance. Always check which class.

### SHAP for Regression

    shap_values has shape (n_samples, n_features) — no class dimension.
    φⱼ = contribution of feature j to the predicted output value.
    Sum: shap_values.sum() + expected_value = model.predict()

    Efficiency axiom: sum(SHAP values) = prediction - E[prediction]
    Easy to verify: works directly.

### SHAP for NLP: Token Importance

    For text models (BERT, GPT), SHAP can explain token importance:
        Baseline: mask token [MASK] or zero embedding.
        Perturbation: replace subsets of tokens with baseline.
        SHAP value for token j: how much does this token contribute to the prediction?

    shap.Explainer(pipeline)(text_samples)
    Produces: attribution scores per token → visualise as highlighted text.

    Challenge: n=tokens can be large → KernelSHAP needs many samples.
    PartitionSHAP: hierarchical partitioning of tokens (faster for long texts).


##### PART 8 — SHAP vs OTHER METHODS: WHEN TO USE WHAT

### SHAP vs LIME

    SHAP:
        Axiomatic foundation (the four Shapley axioms are guaranteed).
        Global consistency: can aggregate local explanations to global.
        Efficiency axiom: values sum to prediction gap (checkable).
        More expensive: O(2ⁿ) exact, O(M×f) approximate.
        Single baseline assumption.

    LIME:
        No axiomatic guarantees.
        Local linear approximation — may miss non-linearities.
        Faster than KernelSHAP in practice.
        Easier to understand intuitively (fits a simple model locally).
        Instability: same instance, different seeds → different explanations.

    When LIME is better:
        Very large n (hundreds of features) → LIME can be faster.
        You want a simple rule-based explanation for a lay audience.
        Exploratory use where exact axioms don't matter.

    When SHAP is better:
        You need the efficiency axiom (provable sum to prediction).
        You need global consistency (aggregate local → global).
        You're working with tree models (TreeSHAP is exact AND fast).
        Production explainability with audit requirements.

### SHAP vs Integrated Gradients

    Both satisfy the efficiency axiom (sum to prediction gap).
    Both handle feature interactions.

    Integrated Gradients:
        Works naturally for neural networks via gradients.
        Exact (given sufficient integration steps).
        Requires a single reference baseline.
        Model-specific: needs gradients.

    SHAP:
        Works for any model type.
        Uses a distribution of baselines (marginalises over data).
        Naturally handles feature correlations via background distribution.

    Best practice: use TreeSHAP for trees, IG for neural networks,
    and KernelSHAP only when other methods aren't applicable.

### SHAP for Fairness Auditing

    SHAP's global-to-local consistency makes it ideal for fairness analysis:

    DEMOGRAPHIC SUBGROUP ANALYSIS:
        Compute SHAP values separately for protected and non-protected groups.
        Compare: mean |SHAP| per feature per group.
        Large difference: model weighs features differently for different groups.

    PROXY DETECTION:
        A feature with high SHAP that correlates with a protected attribute
        may be acting as a proxy for discrimination.
        shap.dependence_plot("income", shap_values, X, interaction_index="race")
        Color = race: if race colors the income SHAP pattern → income is a proxy.

    COUNTERFACTUAL AUDIT:
        Use SHAP to find which features would need to change (and by how much)
        for the prediction to flip. Do these changes differ by protected group?

    ADVERSE ACTION NOTICES (ECOA compliance):
        For credit denials: the top negative SHAP features are the legally required
        "specific reasons" for the denial.
        "Reason 1: high debt-to-income ratio (SHAP = -0.23)"
        "Reason 2: insufficient credit history (SHAP = -0.15)"

"""

OPERATIONS = {

    "1 · Shapley Values from Scratch — Mathematics and the Four Axioms": {
        "description": (
            "Implement exact Shapley value computation from first principles. "
            "Enumerate all 2ⁿ subsets with correct weighting. "
            "Verify all four axioms (efficiency, symmetry, dummy, linearity). "
            "Demonstrate how the baseline changes interpretations. "
            "Show the convergence of KernelSHAP sampling approximation. "
            "Compare exact vs approximate SHAP accuracy."
        ),
        "language": "python",
        "code": r'''
import numpy as np
from itertools import combinations, permutations
import math

print("=" * 65)
print("  SHAPLEY VALUES — MATHEMATICS AND THE FOUR AXIOMS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Exact Shapley value computation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Exact Shapley values: all 2ⁿ subsets")
print("━" * 65)
print()

def exact_shapley_values(f, x, baseline, n_features):
    """
    Compute exact Shapley values by enumerating all 2^n subsets.

    f:          callable(np.array shape (1,n)) -> scalar
    x:          instance to explain, shape (n,)
    baseline:   reference point, shape (n,)
    n_features: n

    Returns: array of shape (n,) — one Shapley value per feature
    """
    phi = np.zeros(n_features)

    for j in range(n_features):
        others = [k for k in range(n_features) if k != j]

        for size in range(len(others) + 1):
            for S in combinations(others, size):
                S = list(S)

                # Compute weight: |S|! * (n-|S|-1)! / n!
                w = (math.factorial(len(S)) *
                     math.factorial(n_features - len(S) - 1) /
                     math.factorial(n_features))

                # Build x with S features present (others at baseline)
                x_with    = baseline.copy()
                for k in S: x_with[k] = x[k]
                x_with[j] = x[j]

                # Build x with only S features present
                x_without    = baseline.copy()
                for k in S: x_without[k] = x[k]

                # Marginal contribution of adding j to coalition S
                marginal = f(x_with.reshape(1,-1)).ravel()[0] - f(x_without.reshape(1,-1)).ravel()[0]
                phi[j]  += w * marginal

    return phi


def exact_shapley_via_permutations(f, x, baseline, n_features):
    """
    Alternative exact computation: average over all n! orderings.
    Pedagogically clearer — shows the ordering-based interpretation.
    """
    phi   = np.zeros(n_features)
    count = 0

    for ordering in permutations(range(n_features)):
        # Build coalition incrementally following this ordering
        coalition = []

        for j in ordering:
            # Current coalition S (features before j in this ordering)
            x_with    = baseline.copy()
            for k in coalition: x_with[k] = x[k]
            x_with[j] = x[j]

            x_without = baseline.copy()
            for k in coalition: x_without[k] = x[k]

            phi[j] += f(x_with.reshape(1,-1)).ravel()[0] - f(x_without.reshape(1,-1)).ravel()[0]
            coalition.append(j)

        count += 1

    return phi / count   # average over all orderings


# ── Define a model with known properties ─────────────────────────────
def model_additive(X):
    """Purely additive model: f = 2x0 + 1.5x1 - x2"""
    return 2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2]

def model_interactive(X):
    """Model with interaction: f = 2x0 + 1.5x1 + 1.0*x0*x1 - x2"""
    return (2.0 * X[:, 0] + 1.5 * X[:, 1]
            + 1.0 * X[:, 0] * X[:, 1]  # interaction!
            - 1.0 * X[:, 2])

n = 3
x_inst  = np.array([1.0, 1.0, 0.5])
baseline = np.array([0.0, 0.0, 0.0])

# Compute with both methods
phi_subset = exact_shapley_values(model_interactive, x_inst, baseline, n)
phi_perm   = exact_shapley_via_permutations(model_interactive, x_inst, baseline, n)

print("  Model: f(x) = 2x₀ + 1.5x₁ + x₀×x₁ - x₂  (with interaction)")
print(f"  Instance:  x = {x_inst}")
print(f"  Baseline: x̄ = {baseline}")
print()

pred      = model_interactive(x_inst.reshape(1,-1)).ravel()[0]
pred_base = model_interactive(baseline.reshape(1,-1)).ravel()[0]
print(f"  f(x)  = {pred:.4f}")
print(f"  f(x̄)  = {pred_base:.4f}")
print(f"  Gap   = {pred - pred_base:.4f}")
print()

print(f"  {'Method':<25} | {'φ₀ (x₀)':>12} | {'φ₁ (x₁)':>12} | {'φ₂ (x₂)':>12} | {'Sum':>10}")
print(f"  {'─'*73}")
for name, phi in [("Subset enumeration", phi_subset), ("Permutation average", phi_perm)]:
    print(f"  {name:<25} | {phi[0]:>12.6f} | {phi[1]:>12.6f} | {phi[2]:>12.6f} | {phi.sum():>10.6f}")

print(f"  {'Expected gap':<25} | {'':>12} | {'':>12} | {'':>12} | {pred-pred_base:>10.6f}")
print()
print("  Both methods give identical results — alternative derivations of same formula.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Verifying all four axioms
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Verifying the four Shapley axioms")
print("━" * 65)
print()

# AXIOM 1: EFFICIENCY
print("  AXIOM 1 — EFFICIENCY: Σⱼ φⱼ = f(x) - f(baseline)")
phi  = exact_shapley_values(model_interactive, x_inst, baseline, n)
gap  = model_interactive(x_inst.reshape(1,-1)).ravel()[0] - model_interactive(baseline.reshape(1,-1)).ravel()[0]
diff = abs(phi.sum() - gap)
print(f"    Σφⱼ = {phi.sum():.6f}")
print(f"    f(x) - f(x̄) = {gap:.6f}")
print(f"    |Σφⱼ - gap| = {diff:.2e}  {'✅ Satisfied' if diff < 1e-10 else '❌ Violated'}")
print()

# AXIOM 2: SYMMETRY
print("  AXIOM 2 — SYMMETRY: equal contributors get equal values")
# Create model where x0 and x1 contribute identically
def model_symmetric(X):
    """f(x) = x0 + x1 + x2  (all features contribute equally)"""
    return X[:, 0] + X[:, 1] + X[:, 2]

x_sym   = np.array([1.0, 1.0, 1.0])
phi_sym = exact_shapley_values(model_symmetric, x_sym, baseline, n)
print(f"    Model: f(x) = x₀ + x₁ + x₂  (symmetric)")
print(f"    Instance: all features = 1.0")
print(f"    φ₀ = {phi_sym[0]:.4f}, φ₁ = {phi_sym[1]:.4f}, φ₂ = {phi_sym[2]:.4f}")
sym_ok = abs(phi_sym[0] - phi_sym[1]) < 1e-10 and abs(phi_sym[1] - phi_sym[2]) < 1e-10
print(f"    All equal: {'✅ Satisfied' if sym_ok else '❌ Violated'}")
print()

# AXIOM 3: DUMMY
print("  AXIOM 3 — DUMMY: irrelevant feature gets φ = 0")
def model_dummy(X):
    """f(x) = 3x0 + 2x1  (feature x2 is completely ignored)"""
    return 3.0 * X[:, 0] + 2.0 * X[:, 1]

x_dummy   = np.array([0.8, 0.6, 0.9])   # x2 has a non-zero value
phi_dummy = exact_shapley_values(model_dummy, x_dummy, baseline, n)
print(f"    Model: f(x) = 3x₀ + 2x₁  (x₂ not used!)")
print(f"    x₂ value = {x_dummy[2]:.1f}  (non-zero, but irrelevant)")
print(f"    φ₂ = {phi_dummy[2]:.8f}  {'✅ Satisfied (≈0)' if abs(phi_dummy[2]) < 1e-10 else '❌ Violated'}")
print()

# AXIOM 4: LINEARITY
print("  AXIOM 4 — LINEARITY: φⱼ(f+g) = φⱼ(f) + φⱼ(g)")
def model_g(X):
    return 0.5 * X[:, 0] - 0.3 * X[:, 2]

def model_f_plus_g(X):
    return model_interactive(X) + model_g(X)

x_lin    = np.array([0.7, 0.8, 0.3])
phi_f    = exact_shapley_values(model_interactive, x_lin, baseline, n)
phi_g    = exact_shapley_values(model_g,           x_lin, baseline, n)
phi_fg   = exact_shapley_values(model_f_plus_g,    x_lin, baseline, n)

print(f"    φ(f):   {phi_f.round(4)}")
print(f"    φ(g):   {phi_g.round(4)}")
print(f"    φ(f+g): {phi_fg.round(4)}")
print(f"    φ(f)+φ(g): {(phi_f+phi_g).round(4)}")
lin_ok = np.allclose(phi_fg, phi_f + phi_g, atol=1e-10)
print(f"    φ(f+g) = φ(f)+φ(g): {'✅ Satisfied' if lin_ok else '❌ Violated'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Baseline impact
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Baseline matters: how reference point changes values")
print("━" * 65)
print()

x_exp = np.array([0.7, 0.8, 0.3])
baselines = {
    "Zero baseline [0,0,0]":        np.array([0.0, 0.0, 0.0]),
    "Mean baseline [0.5,0.5,0.5]":  np.array([0.5, 0.5, 0.5]),
    "Bad baseline [0.9,0.9,0.9]":   np.array([0.9, 0.9, 0.9]),
    "Neutral [0.3,0.3,0.6]":        np.array([0.3, 0.3, 0.6]),
}

feature_names = ["x₀ (income)", "x₁ (credit)", "x₂ (debt)"]

print(f"  Instance x = {x_exp}")
print(f"  Model: f(x) = 2x₀ + 1.5x₁ + x₀×x₁ - x₂")
print()
print(f"  {'Baseline':<32} | {'f(x̄)':>8} | {'φ₀':>8} | {'φ₁':>8} | {'φ₂':>8} | {'Sum=gap':>10}")
print(f"  {'─'*82}")

for bname, b in baselines.items():
    phi_b     = exact_shapley_values(model_interactive, x_exp, b, n)
    pred_b    = model_interactive(b.reshape(1,-1)).ravel()[0]
    pred_x    = model_interactive(x_exp.reshape(1,-1)).ravel()[0]
    gap_b     = pred_x - pred_b
    print(f"  {bname:<32} | {pred_b:>8.4f} | {phi_b[0]:>8.4f} | {phi_b[1]:>8.4f} | "
          f"{phi_b[2]:>8.4f} | {phi_b.sum():>10.4f}")

print()
print("  KEY INSIGHT: SHAP values CHANGE with the baseline!")
print("  φⱼ = 'contribution of feature j relative to the baseline'")
print("  Always report which baseline you used.")
print("  E[f(X)] baseline: 'how much does feature j shift from the average prediction?'")
print("  Zero baseline: 'what does feature j add starting from scratch?'")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: KernelSHAP convergence
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — KernelSHAP convergence: M samples vs accuracy")
print("━" * 65)
print()

def kernel_shap_approx(f, x, baseline, n_features, M=1000, seed=42):
    """
    KernelSHAP: approximate Shapley values via weighted linear regression.
    Implements the Shapley kernel weighting exactly.
    """
    rng = np.random.RandomState(seed)

    # Sample M binary masks (subsets)
    Z_prime = np.zeros((M + 2, n_features))  # +2 for boundary conditions

    # Boundary: all-zeros and all-ones (infinite weight → add directly)
    Z_prime[0, :] = 0   # baseline prediction
    Z_prime[1, :] = 1   # full prediction

    for i in range(2, M + 2):
        # Sample subset size from Shapley kernel
        size    = rng.randint(1, n_features)
        subset  = rng.choice(n_features, size=size, replace=False)
        Z_prime[i, subset] = 1

    # Get model predictions for each perturbed input
    Y = np.zeros(M + 2)
    for i in range(M + 2):
        x_perm = baseline.copy()
        for j in range(n_features):
            if Z_prime[i, j] == 1:
                x_perm[j] = x[j]
        Y[i] = f(x_perm.reshape(1, -1)).ravel()[0]

    # Compute Shapley kernel weights
    weights = np.zeros(M + 2)
    weights[0] = 1e10   # enforce boundary: Z'=0 → f(baseline)
    weights[1] = 1e10   # enforce boundary: Z'=1 → f(x)
    for i in range(2, M + 2):
        s = int(Z_prime[i].sum())
        if s > 0 and s < n_features:
            weights[i] = (n_features - 1) / (
                math.comb(n_features, s) * s * (n_features - s))

    # Weighted linear regression
    from numpy.linalg import lstsq
    # Include intercept (= baseline prediction)
    Z_with_intercept = np.column_stack([np.ones(M+2), Z_prime])
    W = np.diag(weights)
    # Solve: (Z^T W Z) φ = Z^T W Y
    ZWZ = Z_with_intercept.T @ W @ Z_with_intercept
    ZWY = Z_with_intercept.T @ (W @ Y)
    phi_with_intercept = lstsq(ZWZ, ZWY, rcond=None)[0]

    return phi_with_intercept[1:]   # drop intercept term

# True values
x_conv    = np.array([0.7, 0.8, 0.3])
baseline  = np.array([0.0, 0.0, 0.0])
phi_true  = exact_shapley_values(model_interactive, x_conv, baseline, n)

print(f"  True Shapley values (exact): {phi_true.round(6)}")
print()
print(f"  Convergence of KernelSHAP approximation:")
print(f"  {'M samples':>10} | {'φ₀ approx':>12} | {'φ₁ approx':>12} | {'φ₂ approx':>12} | {'Max error':>12}")
print(f"  {'─'*65}")

for M in [10, 50, 100, 500, 1000, 5000]:
    phi_approx = kernel_shap_approx(model_interactive, x_conv, baseline, n, M=M, seed=0)
    max_err    = np.max(np.abs(phi_approx - phi_true))
    print(f"  {M:>10} | {phi_approx[0]:>12.6f} | {phi_approx[1]:>12.6f} | "
          f"{phi_approx[2]:>12.6f} | {max_err:>12.6f}")

print()
print("  KernelSHAP converges to the true Shapley values as M → ∞.")
print("  M=1000–5000 is typically sufficient for production explanations.")
print("  For higher-dimensional data: convergence is slower.")
''',
    },

    "2 · TreeSHAP and the shap Library — Production Workflows": {
        "description": (
            "TreeSHAP: exact and fast SHAP for tree ensembles. "
            "Full workflow: train Random Forest and XGBoost, compute SHAP. "
            "Verify efficiency axiom quantitatively. "
            "SHAP interaction values: main effects vs interaction terms. "
            "Global SHAP analysis: summary plot interpretation in code. "
            "Local explanations: waterfall and force plot data. "
            "Compare TreeSHAP to permutation importance."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  TREESHAP AND THE SHAP LIBRARY — PRODUCTION WORKFLOWS")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Setup: realistic tabular dataset
# ─────────────────────────────────────────────────────────────────────────
try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    import shap

    # Load California housing dataset
    housing    = fetch_california_housing()
    X, y       = housing.data, housing.target
    feat_names = housing.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"  Dataset: California Housing")
    print(f"  Features: {len(feat_names)}, Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Target: median house value (in $100k)")
    print(f"  Features: {', '.join(feat_names)}")
    print()
    HAS_DEPS = True
except ImportError as e:
    print(f"  Missing dependency: {e}")
    print("  Install: pip install shap scikit-learn")
    HAS_DEPS = False

if not HAS_DEPS:
    REFERENCE = """
  TREESHAP PRODUCTION REFERENCE:

  import shap
  from sklearn.ensemble import RandomForestRegressor

  # Train model
  rf = RandomForestRegressor(n_estimators=100, random_state=42)
  rf.fit(X_train, y_train)

  # ── LOCAL explanation (one instance) ──────────────────────────────────
  explainer   = shap.TreeExplainer(rf)
  shap_values = explainer.shap_values(X_test)  # shape: (n_test, n_features)

  # Verify efficiency axiom:
  sum_shap = shap_values.sum(axis=1) + explainer.expected_value
  assert np.allclose(sum_shap, rf.predict(X_test), atol=1e-4)

  # Waterfall plot data for instance 0:
  sv0 = shap_values[0]
  for fname, val, sv in sorted(zip(feat_names, X_test[0], sv0),
                                key=lambda t: abs(t[2]), reverse=True):
      print(f"  {fname}: value={val:.3f}, SHAP={sv:+.4f}")

  # ── GLOBAL explanation (all test instances) ───────────────────────────
  mean_abs_shap = np.abs(shap_values).mean(axis=0)  # mean |SHAP| per feature
  ranked = np.argsort(mean_abs_shap)[::-1]

  # Summary plot (code interpretation):
  for idx in ranked:
      # Positive SHAP = high feature value pushes prediction up
      corr = np.corrcoef(X_test[:, idx], shap_values[:, idx])[0, 1]
      print(f"  {feat_names[idx]}: importance={mean_abs_shap[idx]:.4f}, "
            f"direction={'↑' if corr > 0 else '↓'}")

  # ── SHAP Interaction values ────────────────────────────────────────────
  shap_interactions = explainer.shap_interaction_values(X_test[:100])
  # Shape: (100, n_features, n_features)
  # shap_interactions[i, j, j] = main effect of feature j for sample i
  # shap_interactions[i, j, k] = interaction effect between j and k for i

  # Strongest interactions:
  interaction_matrix = np.abs(shap_interactions).mean(axis=0)
  np.fill_diagonal(interaction_matrix, 0)  # zero out main effects
  j, k = np.unravel_index(interaction_matrix.argmax(), interaction_matrix.shape)
  print(f"  Strongest interaction: {feat_names[j]} × {feat_names[k]}")

  # ── SHAP for different model types ────────────────────────────────────
  # XGBoost (built-in TreeSHAP support):
  import xgboost as xgb
  xgb_model = xgb.XGBRegressor().fit(X_train, y_train)
  explainer  = shap.TreeExplainer(xgb_model)
  # XGBoost can also compute SHAP internally:
  dtest = xgb.DMatrix(X_test)
  shap_xgb = xgb_model.get_booster().predict(dtest, pred_contribs=True)
  # shap_xgb[:, -1] = bias (intercept), shap_xgb[:, :-1] = SHAP values
"""
    print(REFERENCE)
    import sys; sys.exit(0) if False else None

else:
    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: Train model and compute TreeSHAP
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 1 — Train RandomForest and compute TreeSHAP")
    print("━" * 65)
    print()

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    r2 = rf.score(X_test, y_test)
    print(f"  Random Forest R² = {r2:.4f}")
    print()

    explainer = shap.TreeExplainer(rf)

    print("  Computing TreeSHAP for full test set...")
    t0 = time.perf_counter()
    shap_values = explainer.shap_values(X_test)
    t_shap = time.perf_counter() - t0

    print(f"  Computed {len(X_test)} × {len(feat_names)} SHAP values in {t_shap:.3f}s")
    print(f"  SHAP values shape: {shap_values.shape}")
    print()

    # ── Verify efficiency axiom ────────────────────────────────────────────
    ev       = explainer.expected_value
    sum_shap = shap_values.sum(axis=1) + ev
    preds    = rf.predict(X_test)
    max_err  = np.max(np.abs(sum_shap - preds))

    print(f"  Efficiency axiom verification:")
    print(f"    Expected value (baseline):     {ev:.4f}")
    print(f"    Example prediction:            {preds[0]:.4f}")
    print(f"    Sum(SHAP) + baseline:          {sum_shap[0]:.4f}")
    print(f"    Max |sum(SHAP) - prediction|:  {max_err:.2e}  {'✅' if max_err < 1e-4 else '❌'}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: Global analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 2 — Global SHAP analysis: feature importance and direction")
    print("━" * 65)
    print()

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked   = np.argsort(mean_abs)[::-1]

    print(f"  Global feature importance (mean |SHAP|) and directional effect:")
    print(f"  Baseline prediction: {ev:.4f} ($100k)")
    print()
    print(f"  {'Rank':>4} | {'Feature':<20} | {'Mean|SHAP|':>12} | "
          f"{'Direction':>11} | {'High val →':>12} | {'Low val →':>12}")
    print(f"  {'─'*80}")

    for rank, idx in enumerate(ranked):
        fname   = feat_names[idx]
        imp     = mean_abs[idx]
        # Spearman correlation: do high feature values tend to increase SHAP?
        corr    = np.corrcoef(X_test[:, idx], shap_values[:, idx])[0, 1]
        high_ef = "↑ Higher price" if corr > 0.1 else ("↓ Lower price" if corr < -0.1 else "Non-monotone")
        low_ef  = "↓ Lower price" if corr > 0.1 else ("↑ Higher price" if corr < -0.1 else "Non-monotone")
        direc   = f"r={corr:+.3f}"
        print(f"  {rank+1:>4} | {fname:<20} | {imp:>12.4f} | "
              f"{direc:>11} | {high_ef:>12} | {low_ef:>12}")

    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: Local explanation for one instance
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 3 — Local explanation: waterfall plot data")
    print("━" * 65)
    print()

    # Find an interesting instance: high prediction
    interesting_idx = np.argmax(preds)
    x_inst          = X_test[interesting_idx]
    sv_inst         = shap_values[interesting_idx]
    pred_inst       = preds[interesting_idx]

    print(f"  Instance #{interesting_idx}: highest predicted house price")
    print(f"  Prediction: {pred_inst:.4f} ($100k = ${pred_inst*100_000:,.0f})")
    print(f"  Baseline:   {ev:.4f} ($100k = ${ev*100_000:,.0f})")
    print(f"  Gap:        {pred_inst - ev:.4f} ($100k)")
    print()
    print("  Waterfall breakdown (how each feature contributes to the gap):")
    print()
    print(f"  {'Feature':<22} | {'Value':>10} | {'SHAP':>12} | {'Running total':>15} | {'Bar'}")
    print(f"  {'─'*78}")

    running = ev
    sv_ranked = np.argsort(np.abs(sv_inst))[::-1]
    print(f"  {'Baseline':<22} | {'':>10} | {'':>12} | {running:>15.4f} |")
    for idx in sv_ranked:
        fname   = feat_names[idx]
        val     = x_inst[idx]
        sv      = sv_inst[idx]
        running += sv
        bar_len = int(abs(sv) / np.abs(sv_inst).max() * 15)
        bar     = ("█" * bar_len) if sv > 0 else ("░" * bar_len)
        sign    = "+" if sv > 0 else ""
        print(f"  {fname:<22} | {val:>10.3f} | {sign}{sv:>11.4f} | {running:>15.4f} | {bar}")

    print()
    print(f"  Final prediction: {running:.4f}  (should equal {pred_inst:.4f})")
    print(f"  Verification: {'✅' if abs(running - pred_inst) < 1e-4 else '❌'}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: SHAP interaction values
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 4 — SHAP interaction values")
    print("━" * 65)
    print()

    N_INTER = 100
    print(f"  Computing SHAP interaction values for {N_INTER} test samples...")
    t0 = time.perf_counter()
    sv_inter = explainer.shap_interaction_values(X_test[:N_INTER])
    t_inter  = time.perf_counter() - t0
    print(f"  Done in {t_inter:.2f}s. Shape: {sv_inter.shape}")
    print(f"  (n_samples × n_features × n_features)")
    print()

    # Main effects = diagonal
    main_effects = sv_inter[:, np.arange(len(feat_names)), np.arange(len(feat_names))]
    mean_main    = np.abs(main_effects).mean(axis=0)

    # Interaction effects = off-diagonal
    interaction_matrix = np.abs(sv_inter).mean(axis=0)
    np.fill_diagonal(interaction_matrix, 0)  # zero out diagonal (main effects)
    mean_inter_per_feat = interaction_matrix.sum(axis=1)  # total interaction per feature

    print("  Main effects vs interaction contributions:")
    print(f"  {'Feature':<20} | {'Main effect':>14} | {'Total interaction':>18} | {'Frac interaction':>17}")
    print(f"  {'─'*75}")
    for idx in np.argsort(mean_main + mean_inter_per_feat)[::-1]:
        total = mean_main[idx] + mean_inter_per_feat[idx]
        frac  = mean_inter_per_feat[idx] / (total + 1e-10)
        print(f"  {feat_names[idx]:<20} | {mean_main[idx]:>14.4f} | "
              f"{mean_inter_per_feat[idx]:>18.4f} | {frac:>16.1%}")

    print()
    print("  Top pairwise interactions (off-diagonal of interaction matrix):")
    print(f"  {'Feature i':<20} | {'Feature j':<20} | {'Mean |interaction|':>20}")
    print(f"  {'─'*65}")
    # Get upper triangle (symmetric matrix)
    pairs = []
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            pairs.append((interaction_matrix[i, j], i, j))
    pairs.sort(reverse=True)
    for strength, i, j in pairs[:5]:
        print(f"  {feat_names[i]:<20} | {feat_names[j]:<20} | {strength:>20.4f}")

    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: SHAP vs permutation importance
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 5 — SHAP vs permutation importance comparison")
    print("━" * 65)
    print()

    def permutation_importance_regression(model, X, y, n_repeats=3):
        baseline_score = model.score(X, y)
        importances    = []
        for j in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                Xp    = X.copy()
                Xp[:, j] = np.random.permutation(Xp[:, j])
                scores.append(model.score(Xp, y))
            importances.append(baseline_score - np.mean(scores))
        return np.array(importances)

    print("  Computing permutation importance...")
    perm_imp = permutation_importance_regression(rf, X_test, y_test)
    gini_imp = rf.feature_importances_

    print(f"  {'Feature':<20} | {'SHAP rank':>10} | {'Perm rank':>10} | {'Gini rank':>10} | {'Agreement'}")
    print(f"  {'─'*68}")

    shap_rank = {feat_names[i]: r+1 for r, i in enumerate(np.argsort(mean_abs)[::-1])}
    perm_rank = {feat_names[i]: r+1 for r, i in enumerate(np.argsort(perm_imp)[::-1])}
    gini_rank = {feat_names[i]: r+1 for r, i in enumerate(np.argsort(gini_imp)[::-1])}

    for feat in feat_names:
        sr, pr, gr = shap_rank[feat], perm_rank[feat], gini_rank[feat]
        max_diff   = max(abs(sr-pr), abs(sr-gr), abs(pr-gr))
        agree      = "✅ Agree" if max_diff <= 2 else "⚠️  Differ"
        print(f"  {feat:<20} | {sr:>10} | {pr:>10} | {gr:>10} | {agree}")

    print()
    print("  When methods disagree on a feature's importance:")
    print("  → That feature likely has correlated competitors in the data.")
    print("  → Perm importance may underestimate if correlation allows substitution.")
    print("  → SHAP interaction values can reveal which correlated features are used.")
''',
    },

    "3 · KernelSHAP, GradientSHAP, and SHAP for NLP": {
        "description": (
            "SHAP beyond tree models. "
            "KernelSHAP on a neural network: background sampling strategies. "
            "GradientSHAP for PyTorch models: convergence and accuracy. "
            "SHAP for text classification: token-level attributions. "
            "Visualise SHAP as highlighted text and token importance bars. "
            "Compare SHAP explainers across model types."
        ),
        "language": "python",
        "code": r'''
import numpy as np
import time

print("=" * 65)
print("  KERNELSHAP, GRADIENTSHAP, AND SHAP FOR NLP")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: KernelSHAP on a neural network
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — KernelSHAP: black-box SHAP for any model")
print("━" * 65)
print()

try:
    import shap
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    # Create a tabular dataset
    N, D = 800, 6
    X_raw = np.random.randn(N, D).astype(np.float32)
    # True labels: non-linear function of features 0, 1, and their interaction
    y_raw = (
        2.0 * X_raw[:, 0]
        + 1.5 * X_raw[:, 1]
        + 1.2 * X_raw[:, 0] * X_raw[:, 1]   # interaction
        - 0.5 * X_raw[:, 2] ** 2             # quadratic
    )
    y_cls = (y_raw > y_raw.mean()).astype(np.float32)

    X_train, X_test = X_raw[:600], X_raw[600:]
    y_train, y_test = y_cls[:600], y_cls[600:]
    feat_names = [f"Feature_{i}" for i in range(D)]

    # Train a small neural network
    class TwoLayerNet(nn.Module):
        def __init__(self, d_in, d_h=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h), nn.ReLU(),
                nn.Linear(d_h, d_h),  nn.ReLU(),
                nn.Linear(d_h, 1),    nn.Sigmoid()
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = TwoLayerNet(D)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt    = torch.FloatTensor(X_train)
    yt    = torch.FloatTensor(y_train)
    for epoch in range(200):
        loss = nn.BCELoss()(model(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        acc = ((model(torch.FloatTensor(X_test)) > 0.5).float().numpy()
               == y_test).mean()
    print(f"  Neural net accuracy: {acc:.4f}")

    def predict_fn(x_np):
        """Wrapper for KernelSHAP: numpy in, numpy out."""
        with torch.no_grad():
            return model(torch.FloatTensor(x_np)).numpy()

    # ── Background data strategies ─────────────────────────────────────────
    print()
    print("  Background data selection impact (KernelSHAP):")
    print()

    x_explain  = X_test[:1]
    pred_x     = predict_fn(x_explain)[0]

    background_strategies = {
        "Full training (200)":  X_train[:200],
        "k-means (50)":        shap.kmeans(X_train, 50).data,
        "Random sample (50)":  shap.sample(X_train, 50).data,
        "Random sample (20)":  shap.sample(X_train, 20).data,
    }

    print(f"  Instance prediction: {pred_x:.4f}")
    print()
    print(f"  {'Background':>22} | {'φ₀':>8} | {'φ₁':>8} | {'φ₂':>8} | {'Sum+E[f]':>10} | {'Error vs pred':>14}")
    print(f"  {'─'*80}")

    ref_shap = None
    for bname, bg in background_strategies.items():
        t0       = time.perf_counter()
        exp_k    = shap.KernelExplainer(predict_fn, bg)
        sv_k     = exp_k.shap_values(x_explain, nsamples=500, silent=True)
        t_k      = time.perf_counter() - t0
        ev_k     = exp_k.expected_value
        sum_check = sv_k[0].sum() + ev_k
        err       = abs(sum_check - pred_x)

        if ref_shap is None: ref_shap = sv_k[0]
        print(f"  {bname:>22} | {sv_k[0][0]:>8.4f} | {sv_k[0][1]:>8.4f} | "
              f"{sv_k[0][2]:>8.4f} | {sum_check:>10.4f} | {err:>14.6f}")

    print()
    print("  k-means(50) is usually the best tradeoff: fast and representative.")
    print("  Full training set is most accurate but slow for large datasets.")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: GradientSHAP for neural networks
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 65)
    print("  SECTION 2 — GradientSHAP: gradient-based SHAP for deep nets")
    print("━" * 65)
    print()

    print("  GradientSHAP = (x - x̄) × E[∂f/∂x at random points between x̄ and x]")
    print("  Faster than KernelSHAP, more principled than raw gradients.")
    print()

    # GradientSHAP requires a PyTorch model directly
    background_t = torch.FloatTensor(shap.sample(X_train, 100).data)
    grad_explainer = shap.GradientExplainer(model, background_t)

    X_test_t = torch.FloatTensor(X_test[:50])
    print("  Computing GradientSHAP for 50 test samples...")
    t0       = time.perf_counter()
    sv_grad  = grad_explainer.shap_values(X_test_t)
    t_grad   = time.perf_counter() - t0
    print(f"  Done in {t_grad:.3f}s")
    print()

    # Verify efficiency (approximate for GradientSHAP)
    with torch.no_grad():
        preds_50 = model(X_test_t).numpy()
    ev_grad    = grad_explainer.expected_value
    sum_grad   = sv_grad.sum(axis=1) + ev_grad
    mean_err   = np.mean(np.abs(sum_grad - preds_50))
    print(f"  Efficiency check (approximate): mean |error| = {mean_err:.4f}")
    print(f"  (GradientSHAP is approximate — some efficiency error is expected)")
    print()

    # Global importance from GradientSHAP
    mean_abs_grad = np.abs(sv_grad).mean(axis=0)
    print("  Global feature importance (GradientSHAP):")
    print(f"  {'Feature':<12} | {'Mean |SHAP|':>14} | {'True important?':>18}")
    print(f"  {'─'*50}")
    for idx in np.argsort(mean_abs_grad)[::-1]:
        important = "✅ Yes" if idx in [0, 1] else ("⚠️  Interaction" if idx == 5 else "❌ Noise")
        print(f"  {feat_names[idx]:<12} | {mean_abs_grad[idx]:>14.4f} | {important:>18}")

    print()
    print("  GradientSHAP correctly identifies features 0 and 1 as most important.")
    print()

except Exception as e:
    print(f"  Missing or incompatible dependency: {e}")
    print("  Install: pip install shap torch scikit-learn")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: SHAP for text classification
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — SHAP for text: token-level attributions")
print("━" * 65)
print()

try:
    import shap
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Simple sentiment dataset
    texts = [
        "This product is absolutely amazing and wonderful",
        "Terrible experience, very disappointed and upset",
        "Great quality, highly recommend to everyone",
        "Worst purchase ever, complete waste of money",
        "Excellent service and fast delivery",
        "Poor quality and slow shipping",
        "Love this product, it works perfectly",
        "Broken on arrival, horrible customer service",
        "Outstanding product, exceeded expectations",
        "Do not buy this, completely useless",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]   # 1=positive, 0=negative

    # Train a logistic regression on TF-IDF features
    vectorizer = TfidfVectorizer(max_features=50)
    clf        = LogisticRegression()
    X_tfidf    = vectorizer.fit_transform(texts)
    clf.fit(X_tfidf, labels)

    # SHAP for text: use LinearSHAP with background = zero vector
    feature_names_text = vectorizer.get_feature_names_out()

    def predict_proba_wrapper(X_text):
        """Predict on dense arrays for SHAP compatibility."""
        return clf.predict_proba(X_text)[:, 1]

    # LinearSHAP on TF-IDF features
    background_tfidf = np.zeros((1, len(feature_names_text)))
    lin_explainer    = shap.LinearExplainer(
        clf,
        masker=shap.maskers.Independent(X_tfidf.toarray(), max_samples=50))
    sv_text = lin_explainer.shap_values(X_tfidf.toarray())

    print("  Text classification SHAP analysis:")
    print("  (Logistic Regression + TF-IDF)")
    print()

    for i, (text, label, sv) in enumerate(zip(texts[:4], labels[:4], sv_text[:4])):
        pred = clf.predict_proba(X_tfidf[i])[0, 1]
        print(f"  [{i}] '{text[:50]}...' " if len(text) > 50 else f"  [{i}] '{text}'")
        print(f"       True: {'positive ✅' if label==1 else 'negative ❌'}, "
              f"Predicted: {pred:.3f}")

        # Find active words (non-zero TF-IDF)
        active_features = np.where(X_tfidf[i].toarray()[0] > 0)[0]
        word_shap = [(feature_names_text[j], sv[j]) for j in active_features]
        word_shap.sort(key=lambda t: t[1], reverse=True)

        print("       Token contributions:")
        for word, shap_val in word_shap[:5]:
            bar   = "█" * int(abs(shap_val) / max(abs(s) for _, s in word_shap) * 8 + 0.5)
            sign  = "+" if shap_val > 0 else ""
            emoji = "😊" if shap_val > 0 else "😞"
            print(f"         {emoji} '{word}': {sign}{shap_val:.4f}  {bar}")
        print()

    # Global word-level importance
    mean_abs_text = np.abs(sv_text).mean(axis=0)
    top_words_idx = np.argsort(mean_abs_text)[::-1][:10]

    print("  Most globally important words (mean |SHAP|):")
    print(f"  {'Word':<20} | {'Mean|SHAP|':>12} | {'Positive?':>12}")
    print(f"  {'─'*50}")
    for idx in top_words_idx:
        word  = feature_names_text[idx]
        imp   = mean_abs_text[idx]
        # Positive SHAP → word pushes toward positive class
        mean_sv = sv_text[:, idx].mean()
        direc   = "Positive ↑" if mean_sv > 0 else "Negative ↓"
        print(f"  {word:<20} | {imp:>12.4f} | {direc:>12}")

except Exception as e:
    print(f"  Missing or incompatible dependency: {e}")
    print("  Install: pip install shap scikit-learn")
    print()

print()
print("━" * 65)
print("  SHAP EXPLAINER SELECTION GUIDE")
print("━" * 65)
print()
GUIDE = """
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Model type            │ Use this SHAP explainer     │ Exact? │ Speed │
  ├──────────────────────────────────────────────────────────────────────┤
  │ Linear regression     │ LinearExplainer             │ Yes    │ ***** │
  │ Logistic regression   │ LinearExplainer             │ Yes    │ ***** │
  │ Decision tree         │ TreeExplainer               │ Yes    │ ***** │
  │ Random Forest         │ TreeExplainer               │ Yes    │ ****  │
  │ XGBoost / LightGBM    │ TreeExplainer (built-in)    │ Yes    │ ***** │
  │ CatBoost              │ TreeExplainer               │ Yes    │ ****  │
  │ PyTorch MLP           │ GradientExplainer           │ Approx │ ****  │
  │ TensorFlow/Keras      │ DeepExplainer               │ Approx │ ***   │
  │ Any black-box model   │ KernelExplainer             │ Approx │ **    │
  │ Text (bag-of-words)   │ LinearExplainer / KernelExp │ *      │ ***   │
  │ Text (transformer)    │ PartitionExplainer          │ Approx │ **    │
  ├──────────────────────────────────────────────────────────────────────┤
  │ ALWAYS verify: shap_values.sum(axis=1) + explainer.expected_value    │
  │              ≈ model.predict(X_test)                                 │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(GUIDE)
''',
    },
}

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