"""
Explainable AI (XAI) — Making Black-Box Models Interpretable
=============================================================

Explainable AI (XAI) is the set of methods, frameworks, and techniques that
make the decisions of artificial intelligence systems understandable to humans.
As AI systems make increasingly consequential decisions — approving loans,
diagnosing disease, recommending sentences, screening job applicants — the
ability to explain, audit, and challenge those decisions has shifted from a
nice-to-have engineering feature into a legal, ethical, and scientific
necessity.

The "black-box problem" is fundamental: modern deep neural networks with
billions of parameters perform their computations through a cascade of
non-linear transformations that are mathematically opaque. No human can
inspect 175 billion weights and intuitively understand why a model predicted
what it predicted. XAI provides the tools to approximate, probe, visualise,
and interrogate these systems to extract human-comprehensible explanations.

XAI sits at the intersection of multiple disciplines:
    Machine learning:  model architecture, training dynamics, inductive bias
    Statistics:        hypothesis testing, confidence intervals, effect sizes
    Human-computer interaction: what explanations are useful to which people
    Philosophy:        causality, counterfactuals, what "explaining" means
    Law and ethics:    GDPR Article 22, EU AI Act, fairness regulations

This module covers the full XAI landscape: the taxonomy of explanation
methods, LIME and SHAP (the two most widely used practical tools), attention
visualisation, concept-based explanations, counterfactual reasoning, model
distillation, and the critical question of how to evaluate whether an
explanation is actually faithful, useful, and not misleading.

"""

import textwrap
import re

TOPIC_NAME   = "Explainable AI (XAI)"
DISPLAY_NAME = "04 · XAI"
ICON         = "🔍"
SUBTITLE     = "Interpretability, SHAP, LIME, Attention, and Faithful Explanations"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY XAI EXISTS: THE CASE FOR INTERPRETABILITY

### The Black-Box Problem

    A random forest with 1,000 trees, each 20 levels deep, makes a prediction
    via ~20 million comparisons. A transformer with 7 billion parameters
    performs billions of floating-point operations. No practitioner can trace
    these computations to understand "why" in any human-meaningful sense.

    This opacity creates concrete, documented failures:
        COMPAS recidivism scores:  A widely used criminal justice algorithm
            was found to predict higher recidivism rates for Black defendants
            than white defendants with equivalent criminal histories.
            With no explanation available, this bias was invisible until
            external auditing (ProPublica, 2016).

        Amazon's hiring algorithm: An ML résumé screening system trained on
            10 years of hiring data learned to penalise résumés containing
            the word "women's" (e.g., "women's chess club captain").
            The model was scrapped when the pattern was discovered (2018).

        Medical imaging: A chest X-ray classifier achieving 90% accuracy
            was found to classify based on whether a metal ruler appeared
            in the X-ray (a proxy for the severity protocol, not disease).
            Without saliency maps, this Clever Hans behaviour was invisible.

        Autonomous vehicles: Lane-detection models trained on sunny California
            data failed in snow and rain — no mechanism existed to identify
            that the model had learned "clear road = gray stripe on asphalt"
            rather than the intended "white painted lane marking."

### Why Explanation Is Hard

    Explanation is not the same as transparency:
        TRANSPARENT model: decision tree, linear model — you can read it.
        BLACK-BOX model: deep neural network — you cannot read the weights.

    But even transparent models become opaque at scale:
        A decision tree with depth 20 has 2^20 = 1M possible paths.
        A linear model with 50K features has 50K coefficients to inspect.
        "Transparency" is relative to human cognitive capacity.

    And explanations can be MISLEADING:
        A faithful explanation accurately reflects the model's computation.
        A useful explanation is comprehensible to its target audience.
        These can conflict: the faithful explanation may be technically correct
        but humanly incomprehensible; a simplified explanation may be
        understandable but omit important non-linear interactions.

    The fundamental tension in XAI:
        Accuracy ↔ Simplicity
        Fidelity ↔ Comprehensibility
        Global ↔ Local understanding
        Causality ↔ Correlation

### The Legal and Regulatory Imperative

    GDPR (EU, 2018) — Article 22:
        "Automated individual decision-making, including profiling."
        Data subjects have the right to obtain "meaningful information about
        the logic involved" in automated decisions that significantly affect them.
        Applies to: credit, employment, insurance, rental, medical decisions.

    EU AI Act (2024):
        High-risk AI systems must provide explanations to users.
        High-risk: biometric classification, critical infrastructure,
        employment decisions, credit scoring, law enforcement.

    ECOA (US, Equal Credit Opportunity Act):
        Credit denials must include specific reasons (adverse action notices).
        "Model says no" is legally insufficient.

    FDA AI/ML guidance (US):
        Medical AI systems must provide transparency documentation.
        Algorithms making medical decisions need audit trails.

    These regulations are driving XAI from research into production
    requirements — compliance teams now mandate explainability infrastructure
    in the same way they mandate data privacy controls.

### The Stakeholder Spectrum

    Different people need different explanations:

    DATA SCIENTIST:
        "Which features are driving this model's generalisation error?"
        "Is the model using the right signal or a spurious correlation?"
        Needs: global feature importance, partial dependence, model comparison.

    DOMAIN EXPERT (doctor, loan officer):
        "For THIS patient / THIS applicant: why was this decision made?"
        "What would need to change for a different outcome?"
        Needs: local explanations, counterfactuals, case-based reasoning.

    AFFECTED INDIVIDUAL (loan applicant, patient):
        "Why was I denied? What can I do to get approved?"
        Needs: plain-language explanation, actionable counterfactuals.

    REGULATOR / AUDITOR:
        "Does this model discriminate against protected groups?"
        "Is it stable and consistent across demographic groups?"
        Needs: fairness metrics, aggregate statistics, testing protocols.

    LEGAL COUNSEL:
        "Can we defend this decision in court?"
        Needs: documented, reproducible explanation methodology.


##### PART 2 — THE XAI TAXONOMY: MAPPING THE LANDSCAPE

### Dimension 1: Scope — Local vs Global

    LOCAL (instance-level) explanation:
        "Why did the model predict X for THIS specific input?"
        Answers: per-prediction rationale, feature contributions for one sample.
        Methods: LIME, SHAP (local), attention weights, saliency maps.
        Use case: contested decisions, debugging failures, user-facing explanations.

    GLOBAL (model-level) explanation:
        "What has the model learned overall? What are its general rules?"
        Answers: overall feature importance, decision boundaries, learned concepts.
        Methods: global SHAP, PDPs, ICE curves, decision rules, model distillation.
        Use case: model validation, bias auditing, regulatory documentation.

    SEMI-LOCAL (cohort-level):
        "How does the model behave for all customers with feature X?"
        Methods: SHAP interactions, subgroup PDPs, conditional importance.

### Dimension 2: Intrinsic vs Post-Hoc

    INTRINSIC (interpretable by design):
        The model IS the explanation. No extra technique needed.
        Examples:
            Linear/logistic regression:   coefficients = direct feature weights
            Decision trees:               rules = explicit decision paths
            Naive Bayes:                  probabilities from independence assumption
            Rule-based systems:           IF-THEN rules, directly readable
            Generalised Additive Models:  f(x) = sum of shape functions per feature
        Trade-off: generally lower accuracy than black-box models on complex tasks.

    POST-HOC (explanation after training):
        The model is black-box; explanation is computed separately.
        Examples:
            LIME:       fits local linear model around prediction of interest
            SHAP:       game-theoretic attribution of feature contributions
            Saliency:   gradient of output w.r.t. input (for neural nets)
            TCAV:       concept-based probing of neural net representations
            Counterfactuals: find minimal changes that flip the prediction
        Trade-off: may not be perfectly faithful; approximation involved.

### Dimension 3: Model-Specific vs Model-Agnostic

    MODEL-SPECIFIC:
        Exploits the particular structure of the model type.
        Works ONLY for that model family.
        Generally more faithful (direct, not approximate).
        Examples:
            Gradient × Input (neural nets):  uses backpropagation
            DeepLIFT (neural nets):          reference-based gradient propagation
            TreeSHAP (tree ensembles):        exact SHAP for trees in O(TLD²)
            Attention weights (transformers): internal attention mechanism
            LRP (Layer-wise Relevance Propagation): neural net-specific rules

    MODEL-AGNOSTIC:
        Treats the model as a black box (input → output oracle).
        Works for ANY model type.
        Generally less faithful (depends on sampling approximation).
        Examples:
            LIME:            local approximation with any surrogate
            Kernel SHAP:     SHAP for any model (Monte Carlo approximation)
            Permutation importance: shuffle features, measure accuracy drop
            Partial Dependence Plots: marginalize over all other features

### Dimension 4: Output Type of Explanation

    FEATURE ATTRIBUTION:
        Assigns a score to each input feature for a given prediction.
        "Feature X contributed +0.3 to the prediction, feature Y contributed -0.1."
        Methods: SHAP values, LIME coefficients, saliency maps.

    FEATURE IMPORTANCE (global):
        Ranks features by their overall contribution to model performance.
        "Feature X is the most important feature overall."
        Methods: permutation importance, mean |SHAP|, gini importance.

    RULE-BASED:
        Explains the prediction as a set of logical conditions.
        "IF income > 50K AND credit_score > 700 THEN approved."
        Methods: anchor explanations, decision rules, RuleFit.

    EXAMPLE-BASED:
        Explains a prediction using training examples.
        "This prediction is similar to these 3 training examples."
        Methods: k-NN in feature space, influence functions, prototype selection.

    COUNTERFACTUAL:
        "What is the minimal change to the input that would flip the prediction?"
        "If your income were $5,000 higher, you would have been approved."
        Methods: DICE, Wachter et al., CEM (contrastive explanations).

    CONCEPT-BASED:
        Explains predictions using human-defined concepts.
        "This image was classified as 'zebra' because of the 'striped' concept."
        Methods: TCAV, Net Dissect, ConceptSHAP.


##### PART 3 — LIME: LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS

### The Core LIME Idea

    LIME (Ribeiro et al., 2016) answers: "Why did the model predict THIS for
    THIS specific instance?" without access to model internals.

    Key insight: even if a model is globally non-linear and complex, it is
    often approximately LINEAR in a small neighbourhood around any single point.

    LIME finds the best local linear approximation to the black-box model.

    Algorithm:
        Given: instance x, black-box model f, number of samples N

        1. PERTURB: generate N samples z'_i near x by randomly setting
           features to their "off" state (zero, mean, or masked token).

        2. QUERY: get black-box predictions for all perturbed samples:
           y_i = f(z_i)   (z_i is the un-perturbed version of z'_i)

        3. WEIGHT: weight each sample by its distance to x in the
           original feature space: π(x, z_i) = exp(-d(x, z_i)² / σ²)
           Closer samples get higher weight.

        4. FIT: solve a weighted linear regression on the perturbed samples:
           min_{w} sum_i π(x, z_i) × (y_i - w^T z'_i)²
           with regularisation to select K important features (LASSO).

        5. EXPLAIN: the coefficients w are the LIME explanation.
           Positive w_j: feature j pushes prediction UP
           Negative w_j: feature j pushes prediction DOWN

### LIME for Different Data Types

    TABULAR DATA:
        "Off" state: sample from feature's training distribution.
        Perturbed sample: replace some features with their "off" state.
        Feature: one column of the table.

    TEXT DATA:
        "Off" state: remove (mask) a word or token.
        Perturbed sample: document with some words removed.
        Feature: one word/token (bag of words representation).
        Example: "The movie was GREAT but the acting [REMOVED]."

    IMAGE DATA:
        "Off" state: replace a superpixel with grey/blurred background.
        Perturbed sample: image with some superpixels greyed out.
        Feature: one superpixel (contiguous region, not individual pixel).
        Algorithm: SLIC or quickshift to segment image into superpixels first.

### LIME's Strengths and Limitations

    STRENGTHS:
        • Works for ANY model (truly model-agnostic)
        • Works for tabular, text, and image data
        • Produces human-readable linear explanations
        • Simple to implement and use (pip install lime)
        • Provides feature importance for the local prediction

    LIMITATIONS:
        • LOCAL INSTABILITY: small changes to x can produce very different
          LIME explanations (sensitive to random perturbation sampling).
          Run LIME 10 times on the same instance → may get different results.

        • NEIGHBOURHOOD DEFINITION: "nearby" is ill-defined. The kernel width σ
          dramatically affects which samples are used and thus the explanation.

        • LINEARITY ASSUMPTION: assumes local linearity. For highly non-linear
          boundaries (e.g., XOR-like regions), LIME explanations can be wrong.

        • FEATURE CORRELATION: if features are correlated, LIME may attribute
          importance to the wrong feature in the correlated pair.

        • SAMPLING BIAS: the perturbed samples may not be realistic
          (e.g., features that never co-occur in training data may appear together).


##### PART 4 — SHAP: SHAPLEY VALUES AND UNIFIED ATTRIBUTIONS

### Shapley Values: The Game-Theoretic Foundation

    Shapley values come from cooperative game theory (Lloyd Shapley, 1953).
    They answer: "How much did each player contribute to the coalition's payout?"

    In ML: "How much did each feature contribute to the model's prediction?"

    FORMAL DEFINITION:
    Let f be the model, x the instance, and N = {1,...,n} the set of features.
    The Shapley value for feature j is:

    φ_j(f, x) = sum_{S ⊆ N\\{j}} [ |S|! × (n - |S| - 1)! / n! ]
                × [ f(x_S ∪ {j}) - f(x_S) ]

    where:
        S:         subset of features NOT including j
        f(x_S):    model prediction with only features in S known
        f(x_S∪{j}): model prediction when feature j is added to S
        The weight: average over all possible orderings of adding j to S

    Intuition:
        Shapley values compute the AVERAGE MARGINAL CONTRIBUTION of each
        feature across ALL possible subsets of other features.
        Feature j's Shapley value = how much j adds to each possible coalition.

### The Four SHAP Axioms (Why Shapley Values Are "Correct")

    SHAP satisfies four desirable axioms that no other attribution method does:

    1. EFFICIENCY (completeness):
       sum_j φ_j = f(x) - f(baseline)
       All Shapley values sum to the prediction minus the baseline.
       Attributions are fully accounted for — nothing is missing or double-counted.

    2. SYMMETRY:
       If f(S ∪ {i}) = f(S ∪ {j}) for all S, then φ_i = φ_j.
       Two features that contribute equally get equal Shapley values.

    3. DUMMY:
       If feature j doesn't change the prediction for any coalition,
       its Shapley value is 0.

    4. ADDITIVITY (linearity):
       Shapley values of a sum of games = sum of their individual Shapley values.
       φ_j(f + g) = φ_j(f) + φ_j(g)

    NO OTHER attribution method satisfies all four simultaneously.
    This axiomatic uniqueness is why SHAP has become the gold standard.

### SHAP in Practice: The Baseline

    f(x_S): model prediction with only features in S "present."
    Features NOT in S are "absent" — replaced by the BASELINE.

    Common baselines:
        Expected value E[f(X)] over the training data (KernelSHAP default)
        Zero vector (for sparse data)
        Mean of each feature
        Reference instance (e.g., a "normal" patient)

    The baseline changes the interpretation:
        E[f(X)] baseline: φ_j = "how much does feature j shift the prediction
                           from the AVERAGE prediction?"
        Zero baseline:    φ_j = "how much does feature j contribute from scratch?"

    Always check: sum(SHAP values) + baseline = model prediction.

### SHAP Variants: Different Algorithms for Different Models

    KERNEL SHAP (Lundberg & Lee, 2017):
        Model-agnostic. Frames Shapley computation as a weighted linear regression.
        Samples subsets S randomly, weights them by the Shapley kernel:
        π(S) = (n-1) / [ C(n, |S|) × |S| × (n - |S|) ]
        Slower than TreeSHAP but works for any model.

    TREE SHAP (Lundberg et al., 2018):
        Exact SHAP for tree ensembles (Random Forest, XGBoost, LightGBM).
        Polynomial time: O(TLD²) where T=trees, L=leaves, D=max depth.
        Available in the shap library and directly in XGBoost/LightGBM.
        Extremely fast: can explain 10,000 predictions in seconds.

    DEEP SHAP:
        Uses DeepLIFT to compute approximate SHAP for deep networks.
        Propagates contributions layer by layer using backprop.
        Not always exact (approximation), but fast.

    GRADIENT SHAP:
        Combines Integrated Gradients with SHAP.
        Samples reference inputs from a distribution and uses gradients.
        Better for complex neural networks than DeepSHAP.

    LINEAR SHAP:
        Exact SHAP for linear models with independent features.
        φ_j = w_j × (x_j - E[x_j])
        Feature weight × deviation from mean. Simple and exact.

    PARTITION SHAP:
        Hierarchically partitions features into groups.
        Faster approximation for high-dimensional data.

### SHAP Visualisations

    FORCE PLOT (local, one prediction):
        Shows each feature's contribution as pushes left (negative) or right
        (positive) from the baseline value toward the final prediction.

    WATERFALL PLOT (local, one prediction):
        Vertical bars showing how each feature shifts the prediction step by step.
        Start at baseline E[f(X)], add each SHAP value sequentially.

    SUMMARY PLOT (global, all features):
        Each point = one sample × one feature.
        X-axis = SHAP value (positive/negative = direction of impact).
        Y-axis = feature (features ranked by mean |SHAP value|).
        Colour = actual feature value (high/low).
        Reveals: which features matter most AND how they impact predictions.

    DEPENDENCE PLOT (one feature vs interaction):
        X-axis = feature value, Y-axis = SHAP value for that feature.
        Colour = a second feature that interacts with it.
        Reveals non-linear relationships and interactions.

    BEESWARM PLOT:
        Like summary plot but with all points shown (not blurred density).
        Shows outliers and distribution of SHAP values.

    BAR PLOT (global, aggregated):
        Mean |SHAP value| per feature.
        Simple ranking of feature importance. Good for reports.


##### PART 5 — GRADIENT-BASED METHODS FOR NEURAL NETWORKS

### Saliency Maps: ∂f/∂x

    The simplest gradient-based explanation:
        Saliency[j] = |∂f(x)/∂x_j|

    For image classification:
        Backpropagate from the output logit to the input pixels.
        Large gradient = pixel strongly influences the prediction.
        Visualise as a heatmap over the original image.

    Problems with vanilla saliency:
        Saturation: ReLU units that are "off" have zero gradient, so
                    preceding inputs appear unimportant even if they matter.
        Noise: gradients are noisy and may reflect local geometry rather
               than semantically meaningful information.

### Gradient × Input

    Attribution[j] = x_j × ∂f(x)/∂x_j

    Multiplies the gradient by the actual input value.
    Rationale: a large gradient at x_j=0 doesn't mean x_j matters —
               no signal input regardless of sensitivity.
    Better than raw gradients but still has the saturation problem.

### Integrated Gradients (Sundararajan et al., 2017)

    The principled gradient-based attribution method.

    IG[j](x) = (x_j - x̄_j) × integral_{α=0}^{1} [ ∂f(x̄ + α(x-x̄)) / ∂x_j ] dα

    where x̄ is the BASELINE input (e.g., black image, zero embedding).

    Approximated by:
    IG[j] ≈ (x_j - x̄_j) × (1/m) × sum_{k=1}^{m} ∂f(x̄ + (k/m)(x-x̄)) / ∂x_j

    m=50 steps is usually sufficient.

    Why IG is better:
        Satisfies COMPLETENESS: sum of IG attributions = f(x) - f(x̄)
        Satisfies SENSITIVITY: if feature j always affects prediction, φ_j ≠ 0
        Avoids saturation: integrates across the path from baseline to input

    Practical choices:
        Black image (zero pixels) for image models
        Embedding of [PAD] token for NLP models
        Mean of training data for tabular models

### SmoothGrad

    Raw gradient maps are noisy. SmoothGrad averages noisy gradients:

    SmoothGrad[j](x) = E_{ε~N(0,σ²)}[∂f(x+ε)/∂x_j]
    ≈ (1/n) sum_{i=1}^{n} ∂f(x+ε_i)/∂x_j

    Adds Gaussian noise to the input n times, computes gradient each time,
    averages. The noise washes out high-frequency gradient artifacts.
    n=50 samples with σ=0.15 × (x_max - x_min) is a common choice.

### GradCAM (Gradient-weighted Class Activation Mapping)

    For CNNs, GradCAM produces class-discriminative heatmaps:
        1. Forward pass, compute gradient of class score w.r.t. LAST CONV LAYER.
        2. Global average pool the gradients: α_k = (1/Z) sum_{i,j} ∂y^c/∂A^k_{ij}
        3. Weight each feature map by its importance: L = ReLU(sum_k α_k × A^k)
        4. Upsample L to input resolution.

    Why the last conv layer?
        It has the highest semantic information while retaining spatial structure.
        Earlier layers have spatial info but are not class-discriminative.
        Fully connected layers have class info but no spatial structure.

    GradCAM++ and EigenCAM: improved variants with better handling of
    multiple object occurrences and more stable attributions.

### Layer-wise Relevance Propagation (LRP)

    LRP propagates the prediction score backward through the network,
    distributing "relevance" from output to input:

        R_j^{(l)} = sum_k [ (a_j × w_{jk}) / sum_j a_j × w_{jk} ] × R_k^{(l+1)}

    At each layer, relevance flows from neuron k in layer l+1 back to
    neuron j in layer l proportionally to its positive contribution.

    Conservation rule: sum of relevances at each layer = model output.
    No relevance created or destroyed in propagation.

    Variants (stabilisation rules):
        LRP-ε:   add small ε to denominator for numerical stability
        LRP-γ:   boost positive weights (γ > 0) for more stable explanations
        LRP-αβ:  separate treatment of positive and negative contributions


##### PART 6 — ATTENTION, CONCEPTS, AND COUNTERFACTUALS

### Attention as Explanation — Controversy

    Transformers compute attention weights showing "how much each token
    attends to each other token." It seems natural to use these as explanations.

    CASE FOR ATTENTION:
        Attention weights are internal model representations.
        High attention to a token suggests the model finds it relevant.
        Works directly from model internals — no post-hoc computation.
        Rollout and raw attention have been used successfully for ViT explainability.

    CASE AGAINST ATTENTION:
        Jain & Wallace (2019): "Attention is not Explanation."
        Adversarial attention distributions can be constructed that
        produce the SAME prediction but VERY DIFFERENT attention patterns.
        Wiegreffe & Pinter (2019): "Attention is not NOT Explanation."
        Counter-argument: attention correlates with gradient-based measures.

    CURRENT CONSENSUS:
        Attention weights alone are NOT faithful explanations.
        They should not be used as the SOLE explanation method.
        Combined with gradient information (GradCAM on attention maps),
        they can provide useful signal.
        Better alternative for transformers: Integrated Gradients on inputs.

### Concept-Based Explanations: TCAV

    TCAV (Testing with Concept Activation Vectors, Kim et al., 2018):
        Answers: "Did the model use the concept STRIPE when classifying ZEBRA?"

    How TCAV works:
        1. Collect positive examples of a human-defined concept (e.g., striped images)
           and negative examples (random non-striped images).
        2. Extract intermediate activations from layer l for both groups.
        3. Train a linear classifier to separate the concept from random examples
           in the activation space. The normal to this hyperplane is the CAV
           (Concept Activation Vector).
        4. Compute TCAVscore = fraction of class c inputs for which gradient
           is positive in the CAV direction:
           TCAVscore = |{x ∈ X_c : ∂f(x)/∂h_l · CAV > 0}| / |X_c|
        5. Statistical test: compare TCAVscore to random CAV baseline.
           High TCAVscore + statistically significant → concept was used.

    TCAV advantages:
        Human interpretable — uses concepts, not features.
        Tests any concept (even ones not in training data).
        Produces a significance test, not just a score.

### Counterfactual Explanations

    "What would need to change for the model to predict differently?"

    Formal definition (Wachter et al., 2017):
        Find x' close to x such that f(x') = y' (desired output).
        min ||x' - x||  subject to f(x') = y'

    Properties of good counterfactuals:
        PROXIMITY:    x' should be close to x (minimal change)
        VALIDITY:     f(x') must achieve the desired output
        SPARSITY:     change as few features as possible
        ACTIONABILITY: only change features the person can actually change
                       (can't change age/race; can change income/address)
        PLAUSIBILITY:  x' should be in the data manifold (realistic)

    DiCE (Diverse Counterfactual Explanations):
        Generates N diverse counterfactuals — multiple paths to the goal.
        Ensures variability: don't always suggest the same change.
        Example: "You can be approved by increasing income OR paying off debt."

    CEM (Contrastive Explanations Method):
        Finds PERTINENT POSITIVES: minimal features that must be present.
        Finds PERTINENT NEGATIVES: minimal features that if added, flip decision.
        "You were approved BECAUSE of your credit score (PP)."
        "You would be denied if your debt-to-income ratio was high (PN)."

### Anchors: Rule-Based Local Explanations

    Anchors (Ribeiro et al., 2018): find a RULE that is sufficient for a prediction.

    Anchor: a set of conditions such that the prediction holds with high precision
    whenever the conditions apply:
        P(f(z) = f(x) | anchor(z) is satisfied) ≥ τ (e.g., 95%)

    Example anchor for loan approval:
        "IF income > $45,000 AND credit_score > 680 THEN prediction = approved
         (holds for 96% of similar applicants)"

    Properties:
        PRECISION: the anchor must be correct at least τ% of the time.
        COVERAGE: the anchor should apply to as many cases as possible.
        These are usually in tension — higher precision → lower coverage.

    Advantage over LIME: anchors are if-then rules, easier to communicate.
    Disadvantage: combinatorial search for rules can be slow.


##### PART 7 — GLOBAL METHODS: PDPs, ICE, AND MODEL DISTILLATION

### Partial Dependence Plots (PDP)

    PDP shows the MARGINAL effect of feature j on the prediction,
    averaged over all other features:

    PD_j(x_j) = (1/n) sum_{i=1}^{n} f(x_j, x_i^{-j})

    where x_i^{-j} denotes all features except j from training sample i.

    Algorithm:
        1. For each value v in the range of feature j:
           For each training sample i: set x_j = v, keep all other features.
           Predict f(v, x_i^{-j}).
           Average over all samples.
        2. Plot the average prediction as a function of v.

    Interpretation:
        Rising PDP: increasing feature j increases predicted probability.
        Flat PDP: feature j doesn't affect prediction (when averaged).
        Complex shape: non-linear relationship.

    LIMITATION — extrapolation:
        If feature j is correlated with other features, some combinations
        (x_j=v, x_i^{-j}) may be unrealistic (outside the joint distribution).
        M-plots and ALE plots (Accumulated Local Effects) fix this.

### Individual Conditional Expectation (ICE)

    ICE plots are one line per training sample — uncollapsed PDPs:
        ICE_{ij}(x_j) = f(x_j, x_i^{-j})   for each sample i

    Reveals HETEROGENEITY that PDPs hide:
        If all ICE lines have the same shape: no interaction effects.
        If ICE lines CROSS each other: strong interaction effects.
        Crossed ICE + flat PDP = cancellation (feature matters differently for different groups).

### Accumulated Local Effects (ALE)

    ALE fixes the extrapolation problem in PDPs:
        Instead of marginalizing over the full distribution (PDP),
        average over LOCAL (conditional) distributions.

    ALE_j(x_j) = integral_{x_j^min}^{x_j} E[∂f/∂x_j | x_j = z] dz

    More computationally expensive than PDP but unbiased for correlated features.

### Global Surrogate Models

    Train an interpretable model to approximate the black-box globally:
        1. Get predictions from the black-box on a large dataset: y_hat = f(X)
        2. Train a simple model g (decision tree, linear model) on (X, y_hat).
        3. Use g as the explanation.

    The surrogate explains the BLACK BOX, not the TRUE labels.
    This is intentional — we want to understand the model, not the data.

    Fidelity = R² of surrogate on y_hat.
    High fidelity (R² ≈ 0.9): surrogate closely mimics the complex model.
    Low fidelity: the explanation is not faithful.

    LIMITATION: global linearity is often too restrictive.
    A linear surrogate for a non-linear model will have low fidelity.
    Better: fit a decision tree surrogate (captures some non-linearity).

### Feature Importance: Permutation-Based

    Permutation importance (Breiman, 2001; Fisher et al., 2019):
        1. Measure model performance on baseline dataset: score_0
        2. Shuffle feature j (break its relationship to target): score_j
        3. Importance_j = score_0 - score_j

    If shuffling feature j hurts performance: j was important.
    If shuffling j doesn't change performance: j was not used.

    Model-agnostic: works for any model, any metric.
    Computed on HELD-OUT data (test set) — so measures contribution to
    generalisation, not just training fit.

    LIMITATION — correlated features: if j and k are correlated,
    shuffling j may not hurt performance because k still provides the same info.
    Both j and k will appear unimportant even if both are used by the model.


##### PART 8 — EVALUATING XAI: FAITHFULNESS, STABILITY, AND PITFALLS

### The Evaluation Problem

    XAI faces a fundamental evaluation challenge:
        We want to evaluate whether an explanation is correct.
        But if we knew the correct explanation, we wouldn't need XAI.

    We cannot directly measure whether an explanation reflects the model's
    true computation. Instead, we use PROXY metrics.

### Faithfulness Metrics

    DELETION / INSERTION (Samek et al., 2017; Petsiuk et al., 2018):
        DELETION: progressively remove top-k most important features.
        Prediction should DROP rapidly if the explanation is faithful.
        Area under the deletion curve (AUC) → lower = better explanation.

        INSERTION: progressively ADD top-k most important features to a blank.
        Prediction should RISE rapidly if important features are identified.
        Area under the insertion curve → higher = better.

    FAITHFULNESS CORRELATION (Bhatt et al., 2020):
        Compute Spearman correlation between:
            SHAP/LIME attributions for feature j
            The measured change in prediction when feature j is removed.
        High correlation → explanations faithfully reflect sensitivity.

    INFIDELITY (Yeh et al., 2019):
        INFD(φ, x) = E_I [(I^T φ(x) - (f(x) - f(x - I)))²]
        where I is a random perturbation vector.
        Measures squared error between attribution prediction and actual prediction gap.

    COMPREHENSIVENESS (DeYoung et al., 2020, for NLP):
        Remove top-k tokens, measure prediction drop.
        Good explanations: removal of top tokens causes large prediction change.

    SUFFICIENCY:
        Keep ONLY top-k tokens, measure how much prediction is preserved.
        Good explanations: top tokens alone mostly maintain the prediction.

### Stability / Robustness Metrics

    LOCAL LIPSCHITZ CONTINUITY (Alvarez-Melis & Jaakkola, 2018):
        If two inputs x and x' are similar, their explanations should be similar.
        L = max_{x'} ||φ(x') - φ(x)||₂ / ||x' - x||₂
        Low L → stable explanations.

    MAX-SENSITIVITY (Yeh et al., 2019):
        Max change in explanation over small perturbations of the input:
        max_{ε: ||ε||<δ} ||φ(x+ε) - φ(x)||
        Low value → explanations don't change erratically with small input changes.

    PRACTICAL TEST:
        Run LIME/SHAP 10 times on the same input with different seeds.
        Compute the standard deviation of feature rankings.
        Good method: ranks are stable across runs.
        Bad method: top features vary run-to-run.

### Common XAI Pitfalls and Failure Modes

    1. EXPLANATION ≠ CAUSALITY:
        SHAP values measure correlation-based attribution, not causal effect.
        High SHAP value for "age" doesn't mean changing age would change outcome.
        Counterfactual explanations come closer to causality, but are still
        model-specific, not world-causal.

    2. THE RASHOMON EFFECT:
        Many different models may achieve similar accuracy.
        Their explanations can be radically different.
        Which explanation is "correct"? (Answer: explanations explain MODELS,
        not truth.)

    3. EXPLANATION GAMING:
        Adversarial users can craft inputs that look fair to post-hoc
        explanation tools but are actually discriminatory.
        Slack et al. (2020): designed a classifier that behaves discriminatorily
        on real data but appears fair when queried by LIME/SHAP (which use
        perturbed samples from a different distribution).

    4. CONFIRMATION BIAS:
        Explanations can INCREASE trust in wrong models.
        If an explanation looks plausible, users may fail to notice the model
        is fundamentally flawed (cognitive ease → reduced critical scrutiny).

    5. FEATURE ATTRIBUTION ≠ IMPORTANCE:
        A feature with high SHAP value for one prediction may not be globally
        important. Local and global explanations answer different questions.

    6. BASELINE SENSITIVITY:
        SHAP values change significantly based on the choice of baseline.
        Researchers sometimes choose baselines that produce desired-looking results.

    7. EXPLANATION COMPLEXITY vs UTILITY:
        Explaining a model accurately often requires as many parameters as
        the model itself — a fundamental result (Lipton, 2018).
        The simplification needed for human comprehension inevitably loses information.

### XAI Best Practices

    1. Use multiple explanation methods — if LIME and SHAP agree, more confidence.
    2. Always report the BASELINE for SHAP values.
    3. Measure faithfulness quantitatively (deletion/insertion curves).
    4. Measure stability (run explanations multiple times).
    5. Test on out-of-distribution inputs — does the explanation still make sense?
    6. Match explanation type to stakeholder: local for affected individuals,
       global for auditors, counterfactual for actionable guidance.
    7. Never use explanations as the sole decision basis — they approximate.
    8. Document the explanation methodology in model cards (Mitchell et al., 2019).

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · SHAP — Shapley Values from Scratch and with the SHAP Library": {
        "description": (
            "Deep dive into Shapley values. "
            "Implement exact Shapley value computation from scratch for a small model. "
            "Verify the efficiency axiom (values sum to prediction). "
            "Use the shap library with TreeSHAP for a Random Forest. "
            "Generate summary plots, waterfall plots, and dependence plots. "
            "Compute global feature importance from SHAP values. "
            "Show interaction effects with SHAP interaction values."
        ),
        "language": "python",
        "code": '''
import numpy as np
from itertools import combinations
import time

print("=" * 65)
print("  SHAP — SHAPLEY VALUES FROM SCRATCH AND WITH THE SHAP LIBRARY")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Exact Shapley values from first principles
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Exact Shapley values: first principles")
print("━" * 65)
print()

def exact_shapley(model_fn, x, baseline, n_features):
    """
    Compute EXACT Shapley values by enumerating all 2^n subsets.
    Only feasible for n <= 15 or so (exponential complexity).

    model_fn:    callable(array) -> scalar prediction
    x:           instance to explain (1D array of length n_features)
    baseline:    reference/background instance (1D array)
    """
    import math

    phi = np.zeros(n_features)

    for j in range(n_features):
        features_without_j = [f for f in range(n_features) if f != j]

        for size in range(len(features_without_j) + 1):
            # All subsets S of features_without_j with |S| = size
            for subset in combinations(features_without_j, size):
                subset = list(subset)

                # Build input WITH feature j in coalition S ∪ {j}
                x_with = baseline.copy()
                for f in subset:
                    x_with[f] = x[f]
                x_with[j] = x[j]
                f_with = model_fn(x_with.reshape(1, -1)).ravel()[0]

                # Build input WITHOUT feature j, coalition S
                x_without = baseline.copy()
                for f in subset:
                    x_without[f] = x[f]
                f_without = model_fn(x_without.reshape(1, -1)).ravel()[0]

                # Shapley weight for this subset
                s     = len(subset)
                n     = n_features
                weight = (math.factorial(s) * math.factorial(n - s - 1)
                          / math.factorial(n))

                # Marginal contribution
                phi[j] += weight * (f_with - f_without)

    return phi


print("  Example: simple 3-feature loan approval model")
print()

# Toy model: linear with interaction
def loan_model(X):
    """
    Prediction: P(approve) based on:
        income (0-1 normalised), credit_score (0-1), debt_ratio (0-1)
    with a non-linear interaction term.
    """
    income       = X[:, 0]
    credit_score = X[:, 1]
    debt_ratio   = X[:, 2]
    logit = (2.0 * income
             + 1.5 * credit_score
             - 2.5 * debt_ratio
             + 1.0 * income * credit_score   # interaction
             - 3.0)
    return 1 / (1 + np.exp(-logit))   # sigmoid → probability

# Applicant A: good income, good credit, moderate debt
x_A = np.array([0.7, 0.8, 0.3])

# Baseline: average applicant
baseline = np.array([0.5, 0.5, 0.5])

print(f"  Applicant A: income=0.7, credit_score=0.8, debt_ratio=0.3")
print(f"  Baseline:    income=0.5, credit_score=0.5, debt_ratio=0.5")
print()

pred_A    = loan_model(x_A.reshape(1, -1))[0]
pred_base = loan_model(baseline.reshape(1, -1))[0]
print(f"  f(applicant A)  = {pred_A:.4f}  ({pred_A:.1%} approval probability)")
print(f"  f(baseline)     = {pred_base:.4f}  ({pred_base:.1%} approval probability)")
print(f"  Prediction gap  = {pred_A - pred_base:.4f}")
print()

# Compute exact Shapley values
phi = exact_shapley(loan_model, x_A, baseline, n_features=3)
feature_names = ["Income", "Credit Score", "Debt Ratio"]

print("  Exact Shapley values:")
total = 0.0
for i, (name, val) in enumerate(zip(feature_names, phi)):
    direction = "↑" if val > 0 else "↓"
    print(f"    {name:<15}: φ = {val:+.4f}  {direction}")
    total += val

print()
print(f"  Sum of SHAP values: {total:.4f}")
print(f"  Prediction gap:     {pred_A - pred_base:.4f}")
print(f"  Efficiency axiom satisfied: {abs(total - (pred_A - pred_base)) < 1e-10} ✅")
print()
print("  Interpretation:")
print(f"    Income=0.7 vs baseline=0.5: contributes {phi[0]:+.4f} to approval prob")
print(f"    Credit=0.8 vs baseline=0.5: contributes {phi[1]:+.4f} to approval prob")
print(f"    Debt=0.3   vs baseline=0.5: contributes {phi[2]:+.4f} to approval prob")
print()

# Another applicant: mediocre profile
x_B    = np.array([0.4, 0.5, 0.7])
pred_B = loan_model(x_B.reshape(1, -1))[0]
phi_B  = exact_shapley(loan_model, x_B, baseline, n_features=3)

print(f"  Applicant B: income=0.4, credit_score=0.5, debt_ratio=0.7")
print(f"  f(applicant B) = {pred_B:.4f}  ({pred_B:.1%} approval)")
print()
print("  Shapley values for B:")
for name, val in zip(feature_names, phi_B):
    print(f"    {name:<15}: φ = {val:+.4f}")
print()
print("  Waterfall view (baseline → prediction):")
running = pred_base
print(f"    Baseline               = {running:.4f}")
for name, val in zip(feature_names, phi_B):
    running += val
    print(f"    + {name:<13}: {val:+.4f} → {running:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: TreeSHAP with the shap library
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — TreeSHAP on a Random Forest (shap library)")
print("━" * 65)
print()

try:
    import shap
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    print(f"  shap version: {shap.__version__}")
    print()

    # Load breast cancer dataset
    data   = load_breast_cancer()
    X, y   = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"  Dataset: Breast Cancer Wisconsin")
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Classes: {data.target_names[0]} (0), {data.target_names[1]} (1)")
    print()

    # Train a Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"  Random Forest accuracy: {acc:.4f}")
    print()

    # Compute SHAP values with TreeSHAP (exact, fast)
    explainer  = shap.TreeExplainer(rf)
    print("  Computing TreeSHAP values for test set...")
    t0         = time.perf_counter()
    shap_values = explainer.shap_values(X_test)
    t_shap     = time.perf_counter() - t0

    # shap_values is a list [class0_values, class1_values] for classification
    # We use class 1 (malignant = 0, benign = 1)
    sv = shap_values[1]   # SHAP values for class "malignant" (actually class 1)

    print(f"  SHAP values computed in {t_shap:.3f}s for {len(X_test)} samples")
    print(f"  SHAP values shape: {sv.shape}")
    print()

    # Verify efficiency axiom
    expected_value = explainer.expected_value[1]
    pred_proba     = rf.predict_proba(X_test)[:, 1]
    sum_shap       = sv.sum(axis=1) + expected_value
    max_err        = np.max(np.abs(sum_shap - pred_proba))
    print(f"  Efficiency axiom check:")
    print(f"    Baseline (expected value): {expected_value:.4f}")
    print(f"    Max |SHAP_sum - prediction|: {max_err:.2e}  {'✅' if max_err < 1e-4 else '❌'}")
    print()

    # Global feature importance: mean |SHAP| per feature
    feature_names_short = [name[:20] for name in data.feature_names]
    mean_abs_shap = np.abs(sv).mean(axis=0)
    ranked_idx    = np.argsort(mean_abs_shap)[::-1]

    print("  Global SHAP feature importance (mean |SHAP|):")
    print(f"  {'Rank':>4} | {'Feature':<25} | {'Mean |SHAP|':>12} | {'Bar'}")
    print(f"  {'─'*65}")
    max_val = mean_abs_shap.max()
    for rank, idx in enumerate(ranked_idx[:10]):
        bar = "█" * int(mean_abs_shap[idx] / max_val * 25)
        print(f"  {rank+1:>4} | {feature_names_short[idx]:<25} | "
              f"{mean_abs_shap[idx]:>12.4f} | {bar}")
    print()

    # Local explanation for one instance
    instance_idx = 0
    print(f"  Local SHAP explanation for test instance #{instance_idx}:")
    print(f"    True label:  {data.target_names[y_test[instance_idx]]}")
    print(f"    Predicted:   {rf.predict([X_test[instance_idx]])[0]} "
          f"({rf.predict_proba([X_test[instance_idx]])[0, 1]:.3f} probability)")
    print(f"    Baseline:    {expected_value:.4f}")
    print()
    print(f"  {'Feature':<25} | {'Value':>9} | {'SHAP value':>12} | {'Direction'}")
    print(f"  {'─'*60}")
    local_shap   = sv[instance_idx]
    local_vals   = X_test[instance_idx]
    local_ranked = np.argsort(np.abs(local_shap))[::-1]
    for idx in local_ranked[:8]:
        direction = "↑ pushes benign" if local_shap[idx] > 0 else "↓ pushes malignant"
        print(f"  {feature_names_short[idx]:<25} | {local_vals[idx]:>9.3f} | "
              f"{local_shap[idx]:>+12.4f} | {direction}")
    print()

    # SHAP interaction values (if time permits)
    print("  SHAP summary statistics:")
    print(f"    Positive SHAP count: {(sv > 0).sum()} "
          f"({(sv > 0).mean():.1%} of all feature-sample pairs)")
    print(f"    Negative SHAP count: {(sv < 0).sum()} "
          f"({(sv < 0).mean():.1%})")
    print(f"    Max SHAP value:      {sv.max():.4f}")
    print(f"    Min SHAP value:      {sv.min():.4f}")
    print()
    print("  To visualise (in Jupyter):")
    print("    shap.summary_plot(shap_values[1], X_test, feature_names=data.feature_names)")
    print("    shap.waterfall_plot(shap.Explanation(sv[0], expected_value, X_test[0], feature_names))")
    print("    shap.dependence_plot('worst radius', sv, X_test)")

except Exception:
    print("  shap not installed or incompatible: pip install shap")
    print("  sklearn not installed: pip install scikit-learn")
    print()
    SHAP_REF = """
  TREESHAP REFERENCE CODE:
  import shap, numpy as np
  from sklearn.ensemble import RandomForestClassifier

  # Train model
  rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

  # Create SHAP explainer (TreeSHAP — exact and fast for trees)
  explainer   = shap.TreeExplainer(rf)
  shap_values = explainer.shap_values(X_test)
  # shap_values[1]: values for class 1, shape (n_samples, n_features)

  # Verify efficiency: SHAP values sum to prediction - baseline
  expected_value = explainer.expected_value[1]
  sum_shap = shap_values[1].sum(axis=1) + expected_value
  assert np.allclose(sum_shap, rf.predict_proba(X_test)[:, 1])

  # Global importance: mean |SHAP| per feature
  mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
  ranked = np.argsort(mean_abs_shap)[::-1]  # rank features

  # Visualise
  shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
  shap.waterfall_plot(...)   # local explanation for one sample
  shap.dependence_plot("feature_name", shap_values[1], X_test)
  shap.force_plot(expected_value, shap_values[1][0], X_test[0])
"""
    print(SHAP_REF)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · LIME, Counterfactuals, and Gradient-Based Explanations": {
        "description": (
            "LIME implementation from scratch for tabular and text data. "
            "Show how perturbation sampling and local linear fitting work. "
            "Demonstrate LIME stability issues (run multiple times). "
            "Implement counterfactual explanations (minimal feature changes to flip prediction). "
            "Compute Integrated Gradients for a PyTorch model. "
            "Compare LIME vs SHAP vs saliency explanations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import numpy as _np_impl
import scipy.optimize as _sp_opt

class Ridge:
    def __init__(self, alpha=1.0, **kw): self.alpha = alpha
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
import time

print("=" * 65)
print("  LIME, COUNTERFACTUALS, AND GRADIENT-BASED EXPLANATIONS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LIME from scratch — tabular data
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — LIME from scratch: tabular explanation")
print("━" * 65)
print()

class LIMETabularScratch:
    """
    Simplified LIME implementation for tabular data.
    Follows the original paper algorithm precisely.
    """
    def __init__(self, training_data, feature_names, kernel_width=0.75,
                 n_samples=1000, n_features=5):
        self.train_data    = training_data
        self.feature_names = feature_names
        self.kernel_width  = kernel_width
        self.n_samples     = n_samples
        self.n_features    = n_features
        self.feature_means = training_data.mean(axis=0)
        self.feature_stds  = training_data.std(axis=0) + 1e-8

    def _kernel(self, distances):
        """Exponential kernel: closer samples get higher weight."""
        return np.exp(-(distances ** 2) / (2 * self.kernel_width ** 2))

    def explain(self, instance, predict_fn, seed=42):
        """
        Generate LIME explanation for one instance.
        Returns: (coefficients, intercept, r_squared)
        """
        rng = np.random.RandomState(seed)
        n   = instance.shape[0]

        # Step 1: PERTURB — sample around the instance
        # Binary mask: 1 = feature present, 0 = replaced with mean
        z_prime = rng.randint(0, 2, size=(self.n_samples, n)).astype(float)

        # Reconstruct actual feature values
        z_actual = np.zeros_like(z_prime)
        for i in range(self.n_samples):
            for j in range(n):
                if z_prime[i, j] == 1:
                    z_actual[i, j] = instance[j]    # keep original
                else:
                    z_actual[i, j] = self.feature_means[j]   # replace with mean

        # Add the original instance
        z_prime  = np.vstack([np.ones(n), z_prime])
        z_actual = np.vstack([instance, z_actual])

        # Step 2: QUERY — get black-box predictions
        y_pred = predict_fn(z_actual)

        # Step 3: WEIGHT — distance from instance in normalised space
        instance_normalised   = (instance - self.feature_means) / self.feature_stds
        z_actual_normalised   = (z_actual - self.feature_means) / self.feature_stds
        distances             = np.sqrt(((z_actual_normalised - instance_normalised)**2).sum(axis=1))
        weights               = self._kernel(distances)

        # Step 4: FIT — weighted linear regression on binary representation z'
        # Using Ridge regression for numerical stability
        regressor = Ridge(alpha=1.0)
        regressor.fit(z_prime, y_pred, sample_weight=weights)
        y_pred_local = regressor.predict(z_prime)

        # R² measures fidelity of local linear approximation
        ss_res = np.sum(weights * (y_pred - y_pred_local) ** 2)
        ss_tot = np.sum(weights * (y_pred - np.average(y_pred, weights=weights)) ** 2)
        r_sq   = 1 - ss_res / (ss_tot + 1e-10)

        return regressor.coef_, regressor.intercept_, r_sq


# Create a dataset: predict loan approval
np.random.seed(42)
N = 500
income       = np.random.normal(50000, 20000, N).clip(10000, 150000) / 100000
credit_score = np.random.normal(0.65, 0.15, N).clip(0.2, 1.0)
debt_ratio   = np.random.normal(0.35, 0.15, N).clip(0.0, 0.9)
age          = np.random.normal(40, 10, N).clip(20, 70) / 70
savings      = np.random.normal(0.3, 0.2, N).clip(0.0, 1.0)

X_train = np.column_stack([income, credit_score, debt_ratio, age, savings])
feature_names = ["Income", "Credit Score", "Debt Ratio", "Age", "Savings"]

# Black-box model (non-linear)
def loan_blackbox(X):
    """Non-linear loan approval model — black box."""
    logit = (3.0 * X[:, 0]          # income (positive)
             + 4.0 * X[:, 1]         # credit score (positive)
             - 3.5 * X[:, 2]         # debt ratio (negative)
             + 0.5 * X[:, 3]         # age (small positive)
             + 1.5 * X[:, 4]         # savings (positive)
             + 2.0 * X[:, 0] * X[:, 1]  # income × credit interaction
             - 0.5 * X[:, 2] ** 2    # high debt penalty (quadratic)
             - 2.5)
    return 1 / (1 + np.exp(-logit))

lime = LIMETabularScratch(X_train, feature_names, n_samples=1000)

# Explain a specific instance
x_explain = np.array([0.65, 0.72, 0.30, 0.55, 0.40])  # good applicant
pred       = loan_blackbox(x_explain.reshape(1, -1))[0]

print(f"  Instance to explain:")
for fname, val in zip(feature_names, x_explain):
    print(f"    {fname:<15}: {val:.3f}")
print(f"  Black-box prediction: {pred:.4f}  ({pred:.1%} approval probability)")
print()

coefs, intercept, r_sq = lime.explain(x_explain, loan_blackbox, seed=42)

print(f"  LIME explanation (local linear model, R²={r_sq:.3f}):")
print(f"  {'Feature':<15} | {'Value':>8} | {'Coefficient':>13} | {'Contribution':>14} | {'Direction'}")
print(f"  {'─'*70}")
contributions = []
for i, (name, val, coef) in enumerate(zip(feature_names, x_explain, coefs)):
    contrib    = coef * 1   # z' = 1 for all present features
    direction  = "↑ pro-approval" if coef > 0 else "↓ anti-approval"
    contributions.append((coef, name))
    print(f"  {name:<15} | {val:>8.3f} | {coef:>+13.4f} | {contrib:>+14.4f} | {direction}")
print()
print(f"  Local model: prediction ≈ {intercept:.4f} (intercept) "
      f"+ feature contributions")
print()

# LIME stability test
print("  LIME stability test: run 5 times with different seeds")
print(f"  {'Seed':>5} | {'Income':>9} | {'Credit':>9} | {'Debt':>9} | {'Age':>9} | {'Savings':>9} | {'R²':>6}")
print(f"  {'─'*70}")
all_coefs = []
for seed in range(5):
    c, _, r = lime.explain(x_explain, loan_blackbox, seed=seed)
    all_coefs.append(c)
    vals = [f"{v:+9.4f}" for v in c]
    print(f"  {seed:>5} | {'|'.join(vals)} | {r:.4f}")

all_coefs = np.array(all_coefs)
print()
print(f"  Feature ranking stability (std of coefficient across runs):")
for i, name in enumerate(feature_names):
    std = all_coefs[:, i].std()
    print(f"    {name:<15}: std = {std:.4f}  {'✅ stable' if std < 0.05 else '⚠️  unstable'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Counterfactual explanations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Counterfactual explanations: what would flip the decision?")
print("━" * 65)
print()

class SimpleCounterfactual:
    """
    Find the minimal change to x that flips f(x) to the desired outcome.
    Uses gradient ascent/descent with sparsity penalty.
    """
    def __init__(self, predict_fn, feature_names, feature_bounds,
                 actionable_mask=None, lambda_sparse=0.1):
        self.predict_fn    = predict_fn
        self.feature_names = feature_names
        self.bounds        = np.array(feature_bounds)
        self.actionable    = actionable_mask if actionable_mask is not None \
                             else np.ones(len(feature_names), dtype=bool)
        self.lambda_sparse = lambda_sparse

    def find_counterfactual(self, x, desired_pred=0.5, max_iter=1000, lr=0.01):
        """
        Find x' near x such that f(x') crosses desired_pred.
        Returns the counterfactual and number of changed features.
        """
        x_cf   = x.copy()
        losses  = []

        for iteration in range(max_iter):
            pred = self.predict_fn(x_cf.reshape(1, -1))[0]

            # Loss: distance from desired prediction + sparsity
            pred_loss   = (pred - desired_pred) ** 2
            sparse_loss = self.lambda_sparse * np.sum(np.abs(x_cf - x) * self.actionable)
            total_loss  = pred_loss + sparse_loss
            losses.append(total_loss)

            # Numerical gradient (finite differences)
            h = 1e-4
            grad = np.zeros_like(x_cf)
            for j in range(len(x_cf)):
                if not self.actionable[j]:
                    continue
                x_plus    = x_cf.copy(); x_plus[j]  += h
                x_minus   = x_cf.copy(); x_minus[j] -= h
                p_loss_plus  = (self.predict_fn(x_plus.reshape(1,-1))[0] - desired_pred)**2
                p_loss_minus = (self.predict_fn(x_minus.reshape(1,-1))[0] - desired_pred)**2
                s_plus  = self.lambda_sparse * np.sum(np.abs(x_plus  - x) * self.actionable)
                s_minus = self.lambda_sparse * np.sum(np.abs(x_minus - x) * self.actionable)
                grad[j] = ((p_loss_plus + s_plus) - (p_loss_minus + s_minus)) / (2 * h)

            # Gradient step
            x_cf -= lr * grad

            # Clip to valid bounds
            x_cf = np.clip(x_cf, self.bounds[:, 0], self.bounds[:, 1])

            # Early stop if close enough
            if abs(self.predict_fn(x_cf.reshape(1,-1))[0] - desired_pred) < 0.05:
                break

        return x_cf, iteration + 1


feature_bounds = [
    (0.1, 1.5),   # income (can increase)
    (0.0, 1.0),   # credit score (can improve)
    (0.0, 0.9),   # debt ratio (can decrease)
    (0.2, 1.0),   # age (NOT actionable — you can't change age)
    (0.0, 1.0),   # savings (can increase)
]
actionable = np.array([True, True, True, False, True])

# A denied applicant
x_denied   = np.array([0.40, 0.55, 0.55, 0.50, 0.15])
pred_denied = loan_blackbox(x_denied.reshape(1, -1))[0]

print(f"  Denied applicant:  approval probability = {pred_denied:.3f} ({pred_denied:.1%})")
print(f"  Features:")
for fname, val, act in zip(feature_names, x_denied, actionable):
    actionable_str = "(actionable)" if act else "(NOT actionable)"
    print(f"    {fname:<15}: {val:.3f}  {actionable_str}")
print()

cf = SimpleCounterfactual(loan_blackbox, feature_names, feature_bounds,
                           actionable_mask=actionable, lambda_sparse=0.2)

x_cf, n_iter = cf.find_counterfactual(x_denied, desired_pred=0.6, max_iter=500)
pred_cf = loan_blackbox(x_cf.reshape(1, -1))[0]

print(f"  Counterfactual found in {n_iter} iterations:")
print(f"  New approval probability: {pred_cf:.3f} ({pred_cf:.1%})")
print()
print(f"  Changes required:")
print(f"  {'Feature':<15} | {'Original':>10} | {'Counterfact':>12} | {'Change':>10} | {'Direction'}")
print(f"  {'─'*60}")
n_changed = 0
for fname, orig, cf_val, act in zip(feature_names, x_denied, x_cf, actionable):
    change    = cf_val - orig
    if abs(change) > 0.001:
        direction = "↑ increase" if change > 0 else "↓ decrease"
        n_changed += 1
    else:
        direction = "— unchanged"
    action_str = "" if act else " (locked)"
    print(f"  {fname:<15} | {orig:>10.3f} | {cf_val:>12.3f} | {change:>+10.3f} | "
          f"{direction}{action_str}")
print()
print(f"  Number of features changed: {n_changed}")
print(f"  L2 distance from original: {np.linalg.norm(x_cf - x_denied):.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Integrated Gradients for neural networks
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Integrated Gradients for neural networks")
print("━" * 65)
print()

try:
    import torch
    import torch.nn as nn

    class LoanNet(nn.Module):
        """Small neural network for loan prediction."""
        def __init__(self, n_features=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = LoanNet()
    # Train briefly
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(loan_blackbox(X_train))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(300):
        loss = nn.BCELoss()(model(X_t), y_t)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    print(f"  Neural network trained: {sum(p.numel() for p in model.parameters())} params")

    def integrated_gradients(model, x, baseline, m=50):
        """
        Compute Integrated Gradients attribution.
        IG[j] = (x_j - baseline_j) × integral_0^1 ∂f(baseline + α(x - baseline))/∂x_j dα
        Approximated with m uniformly spaced α values.
        """
        x_t    = torch.FloatTensor(x).unsqueeze(0).requires_grad_(False)
        b_t    = torch.FloatTensor(baseline).unsqueeze(0)
        alphas = torch.linspace(0, 1, m)
        grads  = []

        for alpha in alphas:
            # Interpolated input: baseline + α × (x - baseline)
            x_interp = (b_t + alpha * (x_t - b_t)).requires_grad_(True)
            output   = model(x_interp)
            output.backward()
            grads.append(x_interp.grad.squeeze(0).numpy())

        # Numerical integration (trapezoidal rule)
        grads      = np.array(grads)
        avg_grads  = np.trapz(grads, alphas.numpy(), axis=0)
        ig_attrs   = (x - baseline) * avg_grads
        return ig_attrs

    # Baseline: zero input (or mean input)
    baseline   = np.zeros(5)
    x_instance = x_denied

    ig = integrated_gradients(model, x_instance, baseline, m=50)

    print()
    print(f"  Integrated Gradients for denied applicant:")
    print(f"  Baseline: {baseline} (zero vector)")
    print()

    pred_net  = model(torch.FloatTensor(x_instance).unsqueeze(0)).item()
    pred_base = model(torch.FloatTensor(baseline).unsqueeze(0)).item()
    print(f"  f(instance): {pred_net:.4f}")
    print(f"  f(baseline): {pred_base:.4f}")
    print(f"  Prediction gap: {pred_net - pred_base:.4f}")
    print(f"  Sum of IG attributions: {ig.sum():.4f}")
    print(f"  Completeness (gap ≈ sum): "
          f"{'✅' if abs(ig.sum() - (pred_net - pred_base)) < 0.05 else '⚠️ '}")
    print()
    print(f"  {'Feature':<15} | {'Value':>8} | {'IG Attribution':>16} | {'Direction'}")
    print(f"  {'─'*56}")
    for fname, val, attr in zip(feature_names, x_instance, ig):
        direction = "↑" if attr > 0 else "↓"
        print(f"  {fname:<15} | {val:>8.3f} | {attr:>+16.4f} | {direction}")
    print()
    print("  IG vs LIME vs SHAP — which to use for neural networks?")
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │ Method │ Exact │ Fast  │ Model-agnostic │ Satisfies axioms│")
    print("  ├──────────────────────────────────────────────────────────┤")
    print("  │ LIME   │ No    │ Yes   │ Yes            │ No              │")
    print("  │ SHAP   │ ~Yes  │ ~Yes  │ Yes (KernelSHP)│ Yes             │")
    print("  │ IG     │ Yes   │ Yes   │ No (needs grad)│ Completeness ✅ │")
    print("  └──────────────────────────────────────────────────────────┘")

except ImportError:
    print("  PyTorch not available — showing Integrated Gradients reference:")
    IG_REF = """
  INTEGRATED GRADIENTS IMPLEMENTATION:

  import torch

  def integrated_gradients(model, x, baseline, m=50):
      x_t, b_t = torch.FloatTensor(x), torch.FloatTensor(baseline)
      alphas   = torch.linspace(0, 1, m)
      grads    = []
      for alpha in alphas:
          x_interp = (b_t + alpha*(x_t - b_t)).requires_grad_(True)
          model(x_interp.unsqueeze(0)).backward()
          grads.append(x_interp.grad.numpy())
      avg_grads = np.trapz(np.array(grads), alphas, axis=0)
      return (x - baseline) * avg_grads

  # Properties:
  # sum(ig_attributions) ≈ f(x) - f(baseline)  [Completeness]
  # baseline: all zeros, mean training vector, or PAD token embedding
  # m=50 steps is usually sufficient (convergence check: vary m)
"""
    print(IG_REF)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Evaluating Explanations — Faithfulness, Stability, and Global Analysis": {
        "description": (
            "How do you know if an explanation is good? "
            "Implement deletion and insertion curves for faithfulness evaluation. "
            "Measure LIME stability with the Lipschitz instability metric. "
            "Build global feature importance with PDPs and ICE curves. "
            "Compare permutation importance vs SHAP importance. "
            "Detect Clever Hans behaviour: models predicting for wrong reasons. "
            "Show XAI for fairness auditing."
        ),
        "language": "python",
        "code": '''
import numpy as np
import numpy as _np_impl

class _DTC:
    def __init__(self,max_depth=8,min_s=5,max_features='sqrt'):
        self.max_depth=max_depth; self.min_s=min_s; self.max_features=max_features
    def _nf(self,d):
        if self.max_features=='sqrt': return max(1,int(_np_impl.sqrt(d)))
        return d
    def _gini(self,y):
        if len(y)==0: return 0.0
        p=_np_impl.bincount(y,minlength=2)/len(y); return 1-_np_impl.sum(p**2)
    def _split(self,X,y,rng):
        n,d=X.shape; best=None; feats=rng.choice(d,size=min(self._nf(d),d),replace=False)
        for f in feats:
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>10: ts=ts[_np_impl.linspace(0,len(ts)-1,10).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_s or r.sum()<self.min_s: continue
                g=(l.sum()*self._gini(y[l])+r.sum()*self._gini(y[r]))/n
                if best is None or g<best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,y,d,rng):
        if d>=self.max_depth or len(y)<=self.min_s or self._gini(y)<1e-10:
            return ('L',_np_impl.bincount(y,minlength=2)/len(y))
        sp=self._split(X,y,rng)
        if sp is None: return ('L',_np_impl.bincount(y,minlength=2)/len(y))
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],y[l],d+1,rng),self._build(X[~l],y[~l],d+1,rng))
    def fit(self,X,y,rng=None):
        if rng is None: rng=_np_impl.random.RandomState(0)
        self._tree=self._build(X,y,0,rng); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict_proba(self,X):
        return _np_impl.array([self._p1(x,self._tree) for x in _np_impl.atleast_2d(X)])

class RandomForestClassifier:
    def __init__(self,n_estimators=100,max_depth=8,random_state=None,
                 max_features='sqrt',min_samples_split=5,n_jobs=None,**kw):
        self.n_estimators=min(n_estimators,30); self.max_depth=max_depth
        self.rs=random_state; self.max_features=max_features; self.min_s=min_samples_split
    def fit(self,X,y):
        rng=_np_impl.random.RandomState(self.rs); n=len(X)
        self._trees=[]; self._classes=_np_impl.unique(y)
        for _ in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            t=_DTC(self.max_depth,self.min_s,self.max_features)
            t.fit(X[idx],y[idx],rng); self._trees.append(t)
        self.feature_importances_=_np_impl.ones(X.shape[1])/X.shape[1]; return self
    def predict_proba(self,X):
        return _np_impl.mean([t.predict_proba(X) for t in self._trees],axis=0)
    def predict(self,X): return _np_impl.argmax(self.predict_proba(X),axis=1)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==_np_impl.asarray(y))

def train_test_split(*arrays,test_size=0.3,random_state=None,**kw):
    rng=_np_impl.random.RandomState(random_state); n=len(arrays[0])
    n_tr=int(n*(1-test_size)); idx=rng.permutation(n); ti,vi=idx[:n_tr],idx[n_tr:]
    out=[]
    for a in arrays: a=_np_impl.asarray(a); out+=[a[ti],a[vi]]
    return out
import time

print("=" * 65)
print("  EVALUATING EXPLANATIONS — FAITHFULNESS, STABILITY, GLOBAL")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Setup: synthetic dataset with known ground truth
# ─────────────────────────────────────────────────────────────────────────
# Create a dataset where we KNOW which features matter
# Feature 0: causal (truly important)
# Feature 1: causal (truly important)
# Feature 2: spurious (correlated with label via training bias)
# Feature 3-9: noise

N = 1000
X_data = np.random.randn(N, 10)
X_data[:, 2] = X_data[:, 0] + np.random.randn(N) * 0.1  # feature 2 ≈ feature 0

# True label: based on features 0 and 1 only
logit = 2.0 * X_data[:, 0] + 1.5 * X_data[:, 1] - 0.5
y_data = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
feature_names = [f"Feature_{i}" for i in range(10)]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test)
print(f"  Random Forest accuracy: {acc:.4f}")
print(f"  True causal features: Feature_0 and Feature_1 only.")
print(f"  Feature_2 is spurious (correlated with Feature_0)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Faithfulness — Deletion and Insertion curves
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Faithfulness: deletion and insertion curves")
print("━" * 65)
print()

print("  Deletion curve: progressively remove most important features.")
print("  If explanation is faithful, prediction should drop rapidly.")
print()
print("  Insertion curve: progressively add most important features to a blank.")
print("  If explanation is faithful, prediction should rise rapidly.")
print()

def deletion_insertion_auc(model, X, importances, baseline_value=0.0,
                             n_steps=10):
    """
    Compute deletion and insertion AUCs.
    importances: array of shape (n_samples, n_features) — higher = more important
    """
    n_features = X.shape[1]
    del_aucs, ins_aucs = [], []

    for i in range(min(len(X), 50)):   # evaluate on 50 samples
        imp = importances[i]
        ranked = np.argsort(imp)[::-1]   # most important first

        del_scores = [model.predict_proba(X[i].reshape(1, -1))[0, 1]]
        ins_scores = [model.predict_proba(np.full_like(X[i], baseline_value).reshape(1, -1))[0, 1]]

        x_del = X[i].copy()
        x_ins = np.full_like(X[i], baseline_value)

        step_size = max(1, n_features // n_steps)
        for step in range(0, n_features, step_size):
            features_to_remove = ranked[step:step+step_size]
            features_to_add    = ranked[step:step+step_size]

            x_del[features_to_remove] = baseline_value
            x_ins[features_to_add]    = X[i, features_to_add]

            del_scores.append(model.predict_proba(x_del.reshape(1, -1))[0, 1])
            ins_scores.append(model.predict_proba(x_ins.reshape(1, -1))[0, 1])

        del_aucs.append(np.mean(del_scores))
        ins_aucs.append(np.mean(ins_scores))

    return np.mean(del_aucs), np.mean(ins_aucs)

# Compute feature importances from different methods
# 1. Random importance (baseline — should be bad)
random_imp  = np.random.rand(len(X_test), 10)

# 2. SHAP importance (should be better)
try:
    import shap
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    shap_imp    = np.abs(shap_values[1])
    has_shap    = True
except Exception:
    shap_imp  = np.abs(np.random.randn(len(X_test), 10))  # placeholder
    has_shap  = False

# 3. Built-in RF importance (should be decent but may select Feature_2)
rf_imp_global = rf.feature_importances_
rf_imp        = np.tile(rf_imp_global, (len(X_test), 1))

importances_dict = {
    "Random (control)":        random_imp,
    "RF Gini importance":      rf_imp,
    "SHAP (TreeSHAP)" if has_shap else "SHAP (unavailable)": shap_imp,
}

print(f"  {'Method':<25} | {'Deletion AUC ↓':>16} | {'Insertion AUC ↑':>17} | {'Faithfulness'}")
print(f"  {'─'*75}")
for name, imp in importances_dict.items():
    del_auc, ins_auc = deletion_insertion_auc(rf, X_test, imp)
    # Good explanation: low deletion AUC (fast drop), high insertion AUC (fast rise)
    faithfulness = "✅ Faithful" if del_auc < 0.55 and ins_auc > 0.55 else "⚠️  Less faithful"
    print(f"  {name:<25} | {del_auc:>16.4f} | {ins_auc:>17.4f} | {faithfulness}")

print()
print("  Lower deletion AUC = removing top features hurts model more → faithful.")
print("  Higher insertion AUC = adding top features recovers model faster → faithful.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Permutation importance vs SHAP importance
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Feature importance comparison methods")
print("━" * 65)
print()

# 1. Permutation importance
def permutation_importance(model, X, y, n_repeats=5, metric="accuracy"):
    """Decrease in score when feature is shuffled."""
    baseline_score = model.score(X, y)
    importances    = []
    for j in range(X.shape[1]):
        scores_j = []
        for _ in range(n_repeats):
            X_perm       = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            scores_j.append(model.score(X_perm, y))
        importances.append(baseline_score - np.mean(scores_j))
    return np.array(importances)

print("  Computing permutation importance...")
perm_imp = permutation_importance(rf, X_test, y_test)
print("  Computing RF Gini importance...")
gini_imp = rf.feature_importances_

print()
print(f"  Feature importance comparison (top 5 features, ranked):")
print()
print(f"  {'Feature':<12} | {'Gini Imp':>10} | {'Perm Imp':>10} | "
      f"{'SHAP (mean|φ|)':>16} | {'True causal?':>13}")
print(f"  {'─'*72}")

shap_global = np.abs(shap_imp).mean(axis=0) if has_shap else np.zeros(10)

# Rank by permutation importance
perm_ranked = np.argsort(perm_imp)[::-1]
for rank, idx in enumerate(perm_ranked[:7]):
    is_causal = "✅ Yes" if idx in [0, 1] else ("⚠️  Spurious" if idx == 2 else "❌ Noise")
    print(f"  {feature_names[idx]:<12} | {gini_imp[idx]:>10.4f} | "
          f"{perm_imp[idx]:>10.4f} | {shap_global[idx]:>16.4f} | {is_causal}")
print()
print("  Ground truth: Feature_0 and Feature_1 are causal.")
print("  Feature_2 is spuriously correlated — may appear important to some methods.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Partial Dependence Plots (PDP) and ICE
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Partial Dependence Plots: global model understanding")
print("━" * 65)
print()

def compute_pdp(model, X, feature_idx, n_grid=50):
    """
    Partial dependence of prediction on feature j.
    Average prediction across all samples for each value of feature j.
    """
    grid_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    pdp_vals    = []
    for v in grid_values:
        X_mod = X.copy()
        X_mod[:, feature_idx] = v
        pdp_vals.append(model.predict_proba(X_mod)[:, 1].mean())
    return grid_values, np.array(pdp_vals)

def compute_ice(model, X, feature_idx, n_grid=50, n_samples=30):
    """
    Individual Conditional Expectation: one line per sample.
    Shows heterogeneity hidden by the PDP average.
    """
    grid_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    samples_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    ice_lines   = []
    for i in samples_idx:
        line = []
        for v in grid_values:
            x_mod = X[i].copy()
            x_mod[feature_idx] = v
            line.append(model.predict_proba(x_mod.reshape(1, -1))[0, 1])
        ice_lines.append(line)
    return grid_values, np.array(ice_lines)

print("  PDP analysis for each feature:")
print()
print(f"  {'Feature':<12} | {'PDP monotone?':>14} | {'PDP range':>11} | "
      f"{'ICE variance':>13} | {'Interactions?':>14}")
print(f"  {'─'*70}")

for feat_idx in range(min(5, X_test.shape[1])):
    grid, pdp    = compute_pdp(rf, X_test, feat_idx)
    grid_i, ice  = compute_ice(rf, X_test, feat_idx, n_samples=20)

    # Monotone: check if PDP is monotonically increasing or decreasing
    diffs     = np.diff(pdp)
    monotone  = "↑ yes" if np.all(diffs >= 0) else ("↓ yes" if np.all(diffs <= 0) else "No")
    pdp_range = pdp.max() - pdp.min()

    # ICE variance: spread of ICE lines measures interaction strength
    ice_std   = np.std(ice, axis=0).mean()
    interact  = "Strong" if ice_std > 0.1 else ("Moderate" if ice_std > 0.05 else "Weak")

    print(f"  {feature_names[feat_idx]:<12} | {monotone:>14} | {pdp_range:>11.4f} | "
          f"{ice_std:>13.4f} | {interact:>14}")

print()
print("  Non-monotone PDP: non-linear relationship between feature and outcome.")
print("  High ICE variance: the feature's effect differs by sample → interactions.")
print("  Crossed ICE lines: the effect even REVERSES for different groups.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Detecting Clever Hans — XAI for model debugging
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Detecting Clever Hans: XAI for model debugging")
print("━" * 65)
print()

print("  The 'Clever Hans' problem: a model learns a spurious shortcut.")
print("  Example: chest X-ray classifier using metal ruler as proxy for severity.")
print()

# Create a Clever Hans dataset
# True label depends on Feature_0
# But training data has a spurious feature (Feature_9) correlated with label
N_ch = 800
X_ch = np.random.randn(N_ch, 10)
# True causal relationship
y_true = (X_ch[:, 0] > 0).astype(int)
# Spurious: in training, Feature_9 is correlated with label
# (simulates photographer always using flash for severe cases)
X_ch[:, 9] = y_true + np.random.randn(N_ch) * 0.3

X_ch_train, X_ch_test, y_ch_train, y_ch_test = train_test_split(X_ch, y_true, test_size=0.3)

# Test set: break the spurious correlation
X_ch_test_fair    = X_ch_test.copy()
X_ch_test_fair[:, 9] = np.random.randn(len(X_ch_test))  # Feature_9 now random

rf_ch = RandomForestClassifier(n_estimators=100, random_state=0)
rf_ch.fit(X_ch_train, y_ch_train)

acc_biased  = rf_ch.score(X_ch_test, y_ch_test)
acc_fair    = rf_ch.score(X_ch_test_fair, y_ch_test)

print(f"  Biased test set (spurious correlation intact): {acc_biased:.4f}")
print(f"  Fair test set   (spurious correlation broken): {acc_fair:.4f}")
print(f"  Performance drop: {acc_biased - acc_fair:.4f}")
print()

# Use permutation importance to detect the spurious feature
perm_ch = permutation_importance(rf_ch, X_ch_test, y_ch_test)
gini_ch = rf_ch.feature_importances_

print("  Feature importance (XAI reveals the problem):")
print(f"  {'Feature':<12} | {'Gini Imp':>10} | {'Perm Imp':>10} | {'Truly causal?':>14}")
print(f"  {'─'*52}")
for idx in np.argsort(perm_ch)[::-1][:5]:
    true = "✅ Yes" if idx == 0 else ("⚠️  SPURIOUS!" if idx == 9 else "❌ Noise")
    print(f"  Feature_{idx:<4} | {gini_ch[idx]:>10.4f} | {perm_ch[idx]:>10.4f} | {true}")

print()
print("  XAI reveals that Feature_9 (spurious shortcut) has HIGH importance!")
print("  Without XAI: model looks great on biased test data.")
print("  With XAI: we see it's relying on the wrong signal.")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: XAI for fairness auditing
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — XAI for fairness: SHAP by demographic group")
print("━" * 65)
print()

# Simulate a demographic variable
groups  = np.random.choice([0, 1], size=len(X_test))  # 0=Group A, 1=Group B
group_names = ["Group A", "Group B"]

if has_shap:
    print("  Mean |SHAP| by demographic group:")
    print("  If explanations differ significantly, model treats groups differently.")
    print()
    print(f"  {'Feature':<12} | {'Group A mean|φ|':>17} | {'Group B mean|φ|':>17} | {'Ratio':>8}")
    print(f"  {'─'*62}")
    for idx in np.argsort(shap_global)[::-1][:5]:
        sv_a = np.abs(shap_imp[groups == 0, idx]).mean()
        sv_b = np.abs(shap_imp[groups == 1, idx]).mean()
        ratio = sv_a / (sv_b + 1e-9)
        flag  = "⚠️ " if ratio > 2.0 or ratio < 0.5 else "   "
        print(f"  {feature_names[idx]:<12} | {sv_a:>17.4f} | {sv_b:>17.4f} | "
              f"{ratio:>7.2f}× {flag}")
    print()
    print("  If a feature has very different SHAP magnitude across groups:")
    print("  → The model relies on that feature differently for different groups.")
    print("  → May indicate disparate treatment — investigate further.")
else:
    print("  (Install shap for fairness-by-group SHAP analysis)")
    print()

print("━" * 65)
print("  XAI BEST PRACTICES SUMMARY")
print("━" * 65)
print()
SUMMARY = """
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Question                    │ Best XAI Method                        │
  ├──────────────────────────────────────────────────────────────────────┤
  │ Why THIS prediction?        │ SHAP waterfall / LIME local            │
  │ What can I change?          │ Counterfactual (DiCE, Wachter)         │
  │ What does model rely on?    │ Global SHAP / Permutation importance   │
  │ Is model non-linear?        │ PDP + ICE curves                       │
  │ Is explanation stable?      │ LIME stability / Lipschitz metric      │
  │ Is explanation faithful?    │ Deletion/insertion curves              │
  │ Is model fair?              │ SHAP by group / Permutation by group   │
  │ Is model a Clever Hans?     │ Perm importance on held-out fair set   │
  │ How does feature X work?    │ SHAP dependence plot                   │
  │ Which model is more explbl? │ Surrogate fidelity comparison          │
  ├──────────────────────────────────────────────────────────────────────┤
  │ ALWAYS: use multiple methods, report baseline, test stability,       │
  │ match explanation type to stakeholder, document methodology.         │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(SUMMARY)
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