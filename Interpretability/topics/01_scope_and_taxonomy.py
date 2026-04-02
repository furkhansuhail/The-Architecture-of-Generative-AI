"""
Scope & Taxonomy — The Map of Interpretability
===============================================

Before you can explain a model, you need to understand the landscape of
explanation itself. This module is the compass for the entire interpretability/
folder: it defines the vocabulary, draws the axes, and shows where every
technique you'll encounter fits within the broader field.

"""

import base64
import os
import textwrap
import re

TOPIC_NAME = "Scope & Taxonomy of Interpretability"
DISPLAY_NAME = "01 · Scope & Taxonomy"
ICON = "🗺️"
SUBTITLE = "The conceptual map of the entire interpretability field"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER — converts local images to base64 HTML for st.markdown()
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    """Convert a local image file to an HTML <img> tag with base64 data."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is Interpretability — and Why Does it Matter?

Imagine a radiologist using an AI system that flags chest X-rays for cancer. 
The model is 94% accurate — better than the hospital's average. But one morning 
it flags a healthy patient. The radiologist wants to know: *why*? 
The model has no answer. It just outputs a probability.

This is the interpretability problem. It is not merely academic — it sits at the 
intersection of trust, accountability, safety, and law. The EU's GDPR gives citizens 
a "right to explanation" for automated decisions. The FDA requires justification for 
clinical AI systems. And beyond regulation: a model you cannot explain is a model 
you cannot debug, audit, or safely improve.

**Interpretability** is the degree to which a human can understand the cause of a 
decision made by a machine learning model. More precisely, it answers the question:

    "Why did the model produce this output?"

or equivalently:

    "What would have to change about the input for the output to change?"

This is harder than it sounds. A neural network with 50 million parameters makes 
decisions through a chain of matrix multiplications and non-linearities that no 
human can trace by hand. Interpretability is the set of tools and frameworks we 
use to bridge this gap.

---

### Interpretability vs Explainability — Clearing Up the Confusion

These two words are used interchangeably in industry, but researchers draw a 
meaningful distinction that is worth knowing:

    INTERPRETABILITY:  An intrinsic property of the model itself.
                       A linear regression model IS interpretable — its 
                       coefficients directly tell you how each feature affects 
                       the output. No extra tool needed.

    EXPLAINABILITY:    A post-hoc property — the ability to explain a model's 
                       decision AFTER the fact, using external tools.
                       A neural network is not interpretable in itself, but we 
                       can add explainability on top of it using SHAP or LIME.

Think of it this way:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  INTERPRETABLE MODEL                                        │
    │  ─────────────────                                          │
    │  Glass box. You can see inside directly.                    │
    │  e.g. Linear Regression, Decision Tree                      │
    │                                                             │
    │  EXPLAINABLE (but not inherently interpretable) MODEL       │
    │  ────────────────────────────────────────────────           │
    │  Black box + flashlight. The model itself is opaque,        │
    │  but we shine a light on specific decisions.                │
    │  e.g. Neural Net + SHAP values                              │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

In practice, XAI (eXplainable AI) is the umbrella term that covers both:
any technique — whether native to the model or applied afterwards — that helps 
humans understand a model's behaviour.

---

### The Two Fundamental Axes — The Core Taxonomy

Every interpretability technique can be placed on two axes. 
Understanding these two axes is the central organizing principle of the entire field.

**Axis 1 — SCOPE: Global vs Local**

The first question to ask about any explanation: 
    Does it explain the entire model, or does it explain one specific prediction?

    GLOBAL EXPLANATIONS
    ───────────────────
    "How does this model behave IN GENERAL?"

    A global explanation summarises the model's overall decision-making strategy.
    It tells you which features the model relies on most, across the entire training 
    distribution. It is the bird's-eye view.

    Examples:
      • Feature importance ranking (which features matter most, on average?)
      • PDP / ALE plots (how does output change as feature X varies?)
      • Overall model weights in a linear regression


    LOCAL EXPLANATIONS
    ──────────────────
    "Why did the model make THIS specific prediction for THIS specific input?"

    A local explanation focuses on a single instance. It is the ground-level view:
    why did the model flag THIS patient, deny THIS loan, misclassify THIS image?

    Examples:
      • SHAP values for one prediction ("feature A pushed this prediction UP by 0.3")
      • LIME ("near this data point, the model behaves like a linear model with these weights")
      • Counterfactual ("if income were $5k higher, the loan would have been approved")


    ASCII DIAGRAM — The Scope Axis:

    ════════════════════════════════════════════════════════════════
    GLOBAL SCOPE                                        LOCAL SCOPE
    ════════════════════════════════════════════════════════════════

    "Describe the whole model"         "Explain this one decision"

         🌍 Bird's eye view                 🔬 Ground level view

         ┌───────────────────┐              ┌───────────────────┐
         │   ALL predictions  │              │  ONE prediction    │
         │   across the full  │              │  for input x = x₀  │
         │   data distribution│              │  at decision time  │
         └────────┬──────────┘              └────────┬──────────┘
                  │                                   │
                  ▼                                   ▼
         Feature importance              SHAP values for x₀
         PDP / ALE plots                 LIME explanation
         Model weights/rules             Counterfactuals
         Overall accuracy by group       Anchors

    ════════════════════════════════════════════════════════════════

    ⚠️  A common mistake: assuming global importance = local importance.
    Just because a feature has HIGH global importance does NOT mean it was 
    important for this specific prediction. A feature like "age" might matter a 
    lot on average, but for a 30-year-old applicant it might be completely neutral.
    This gap is exactly why you need BOTH global and local explanations.


**Axis 2 — METHOD: Intrinsic vs Post-hoc**

The second question: Does interpretability come from the model's structure itself, 
or from an external tool applied after training?

    INTRINSIC (a.k.a. "Ante-hoc")
    ──────────────────────────────
    The model IS the explanation. Interpretability is baked into the model 
    architecture — you don't need external tools.

    This works because the model itself is simple enough for humans to inspect.
    You can read the decision tree paths. You can read the regression coefficients.

    Examples:
      • Linear / Logistic Regression — coefficients ARE the explanation
      • Decision Tree — follow the branches; the path IS the reasoning
      • Rule-based models / decision lists
      • Generalised Additive Models (GAMs)


    POST-HOC
    ────────
    The model is trained first (often a complex black-box). THEN, separately, 
    an explanation method is applied to interrogate it.

    The explanation tool is decoupled from the model. It treats the trained model
    as a function — something you can query with inputs and observe its outputs — 
    and builds an explanation from those observations.

    Examples:
      • SHAP — computes Shapley values for a trained model
      • LIME — fits a local surrogate model around a prediction
      • GradCAM — computes gradient-based saliency maps for neural networks

---

### The Full 2×2 Taxonomy Matrix

Combining these two axes gives us a clean 2×2 grid that maps the entire field:

    ╔══════════════════════════════╦═════════════════════════════════╗
    ║                              ║                                 ║
    ║      INTRINSIC               ║        POST-HOC                 ║
    ║      (built into model)      ║        (applied after training) ║
    ║                              ║                                 ║
    ╠══════════════════════════════╬═════════════════════════════════╣
    ║                              ║                                 ║
    ║  GLOBAL  •  Linear Regression║  • Permutation feature          ║
    ║          •  Decision Tree    ║    importance                   ║
    ║          •  Rule Lists       ║  • PDP / ALE plots              ║
    ║          •  GAMs             ║  • Global surrogate models      ║
    ║                              ║  • Global SHAP summary plots    ║
    ║                              ║                                 ║
    ╠══════════════════════════════╬═════════════════════════════════╣
    ║                              ║                                 ║
    ║  LOCAL   •  Decision Tree    ║  • SHAP (per-instance)          ║
    ║            path for a leaf   ║  • LIME                         ║
    ║          •  Logistic regress.║  • Counterfactual explanations  ║
    ║            coefficients at   ║  • Anchors                      ║
    ║            a given point     ║  • GradCAM / Saliency Maps      ║
    ║                              ║  • Individual SHAP force plots  ║
    ║                              ║                                 ║
    ╚══════════════════════════════╩═════════════════════════════════╝

    Note: Some methods span both cells (e.g., SHAP can be used locally 
    to explain one prediction or globally via summary plots of many predictions).

---

### The Third Axis — Model-Agnostic vs Model-Specific

Beyond the two primary axes, there is a third important dimension:

    MODEL-AGNOSTIC
    ──────────────
    Works with ANY model as a black box. The explanation method only needs to
    observe inputs and outputs — it doesn't care about the internal architecture.

    Advantage:  Swap the underlying model without changing your explanation pipeline.
    Disadvantage: Can be slower; sometimes less faithful because it probes the 
                  model indirectly.

    Examples: SHAP (model-agnostic variant), LIME, PDP, ICE, ALE, Anchors


    MODEL-SPECIFIC
    ──────────────
    Designed for (and only works with) a particular model family. 
    Exploits the internal structure directly — gradients, attention weights, 
    tree paths — to produce richer and often more faithful explanations.

    Advantage:  Can be more accurate and computationally efficient.
    Disadvantage: Locked to one model type. Can't reuse if you switch architectures.

    Examples:
      • Decision tree path (trees only)
      • Linear regression coefficients (linear models only)
      • GradCAM (convolutional neural networks only)
      • SHAP TreeExplainer (gradient-boosted trees only — much faster than agnostic)
      • Attention visualization (transformer models only)
      • Integrated Gradients (differentiable models only)


    THE THREE AXES TOGETHER:
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  SCOPE:    Global ←────────────────────────────→ Local       │
    │                                                              │
    │  METHOD:   Intrinsic ←──────────────────────→ Post-hoc       │
    │                                                              │
    │  REACH:    Model-agnostic ←───────────────→ Model-specific   │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    Every technique in the interpretability/ folder has a fixed coordinate 
    on these three axes. Knowing the coordinate tells you:
      1. When to use it (what question are you answering?)
      2. Its limitations (what can it NOT tell you?)
      3. Whether you can swap the underlying model

---

### The Interpretability–Accuracy Trade-off (and Why It's Complicated)

There is a widely-cited idea in machine learning: interpretability and accuracy 
trade off against each other. Simple models are interpretable but less accurate; 
complex models are accurate but opaque.

    Traditional view (often called the "accuracy-interpretability trade-off"):

    High accuracy
    ▲
    │                                         ★ Deep Neural Net
    │                                  ★ Random Forest
    │                        ★ Gradient Boosted Trees
    │               ★ SVM
    │       ★ Decision Tree
    │  ★ Linear Regression
    │
    └────────────────────────────────────────────────────►
                                          High interpretability


This is largely true — but with two important caveats:

    Caveat 1 — "The Rashomon Effect" (Leo Breiman, 2001):
    ──────────────────────────────────────────────────────
    For many real-world datasets, there is a large set of models that 
    achieve nearly identical accuracy on held-out data. This set is called 
    the "Rashomon set."

    Crucially, within this set, there are OFTEN simple, interpretable models 
    that perform just as well as the complex ones. The accuracy-interpretability 
    trade-off may be much weaker than the traditional view suggests — especially 
    for structured tabular data (the most common ML application in industry).

    Implication: Before reaching for a black-box model, check whether an 
    interpretable model can achieve the same accuracy first. Often it can.

    Caveat 2 — The Cost of Post-hoc Explanations:
    ──────────────────────────────────────────────
    Post-hoc explanations (like SHAP or LIME) do NOT recover interpretability. 
    They provide approximations of what the black-box model is doing — but the 
    explanation is not the model, and can sometimes be misleading.

    A LIME explanation is a local linear approximation. SHAP values are additive 
    attributions. Neither gives you the actual internal reasoning of a neural 
    network — they give you a simplified story that may or may not be faithful.

    This is the key distinction: an intrinsically interpretable model's explanation 
    IS the truth. A post-hoc explanation is an approximation of the truth.

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  INTRINSIC MODEL explanation:   ──── Faithful by definition     │
    │  (linear regression, tree path) ──── Explanation = model        │
    │                                                                 │
    │  POST-HOC explanation:          ──── Approximate                │
    │  (SHAP on a neural net)         ──── Explanation ≈ model        │
    │                                      (with varying fidelity)    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

---

### Types of Explanations — What Form Does the Answer Take?

An important but under-discussed dimension: even within the same scope and method 
category, explanations can take very different forms. The form of the explanation 
matters enormously for who can actually use it.

    1. FEATURE ATTRIBUTION / IMPORTANCE
    ─────────────────────────────────────
    Assigns a score to each input feature reflecting how much it contributed 
    to a prediction (or to the model's general behaviour).

    Form:    Feature → Score mapping
    Example: SHAP values → {"age": +0.23, "income": +0.41, "debt": -0.18}
    Good for: Quantitative analysis, model debugging, trust-building with 
              technical audiences.


    2. RULE-BASED EXPLANATIONS
    ──────────────────────────
    Produces human-readable IF-THEN rules that describe the model's behaviour.

    Form:    IF condition THEN prediction (with confidence)
    Example: IF age > 30 AND income > 50k THEN loan_approved = True (conf: 0.92)
    Good for: Non-technical stakeholders, regulatory compliance, audit trails.


    3. EXAMPLE-BASED EXPLANATIONS
    ──────────────────────────────
    Explains a prediction by pointing to real data points that are similar or 
    instructive.

    Sub-types:
      • Prototypes:       "This is explained by its resemblance to case X"
      • Counterfactuals:  "If these features had been different, the prediction 
                           would have changed to Y"
      • Adversarial:      "Here is the smallest input change that flips the output"

    Form:    Pointer to a data point, or a modified version of the input.
    Good for: Intuitive explanations for domain experts and affected individuals.


    4. VISUAL EXPLANATIONS
    ──────────────────────
    Highlights regions of the input (pixels, tokens, features) that most 
    influenced the prediction. Primarily used for image and text models.

    Form:    Heatmap overlaid on the input
    Example: GradCAM highlights the tumour region in an X-ray that drove 
             the classifier's decision.
    Good for: Computer vision applications, attention visualisation in NLP.


    5. MECHANISTIC EXPLANATIONS
    ─────────────────────────────
    Opens up the model itself and identifies internal circuits, features, 
    or representations that implement specific behaviours. Unlike all other 
    types, this is not about approximating from the outside — it is 
    reverse-engineering the model from the inside.

    Form:    "Neuron X fires for concept C", "Circuit A→B→C implements 
             the behaviour of detecting indirect objects"
    Good for: AI safety research, understanding generalisation, 
              understanding why models fail.
    Note:    This is the frontier of interpretability research (2020–present).


    SUMMARY TABLE — Explanation Types:
    ═══════════════════════════════════════════════════════════════════
    Type              Form            Primary Use              Scope
    ───────────────────────────────────────────────────────────────────
    Feature attrib.   Score per feat  Debugging, trust         Both
    Rule-based        IF-THEN rules   Compliance, audit        Both
    Counterfactual    Modified input  Recourse, fairness       Local
    Visual heatmap    Pixel map       CV / NLP models          Local
    Mechanistic       Circuits/feats  Research, safety         Global
    ═══════════════════════════════════════════════════════════════════

---

### Properties of a Good Explanation

Not all explanations are equally useful or trustworthy. When evaluating or 
comparing explanation methods, researchers use several key properties:

    1. FAITHFULNESS (a.k.a. Fidelity)
    ──────────────────────────────────
    Does the explanation accurately reflect what the model is actually doing?

    This is the most fundamental property. An explanation that is easy to 
    understand but doesn't actually describe the model's behaviour is worse 
    than useless — it creates false confidence.

    Test: If the explanation says "feature A was the most important factor", 
    does removing or masking feature A actually change the prediction significantly?

    Common pitfall: LIME explanations are only locally faithful. They are 
    accurate approximations NEAR the specific input, but can be wildly wrong 
    slightly further away.


    2. STABILITY (a.k.a. Consistency)
    ────────────────────────────────────
    Do similar inputs receive similar explanations?

    An unstable explanation method produces dramatically different explanations 
    for nearly identical inputs. This is alarming: if a small perturbation to 
    an input completely changes the explanation, users cannot trust the explanation 
    to reflect anything real about the model.

    Note: A model can be stable but unfaithful, or faithful but unstable. 
    These are orthogonal properties.


    3. COMPREHENSIBILITY (a.k.a. Interpretability of the Explanation)
    ──────────────────────────────────────────────────────────────────
    Can a human actually understand and act on the explanation?

    An explanation that involves 500 features with small scores is technically 
    correct but not comprehensible. Good explanations are sparse — they highlight 
    the few most important factors, not everything.

    Miller's Law from cognitive psychology: humans can hold roughly 7 ± 2 chunks 
    of information in working memory. Explanations with more than ~5 features 
    become cognitively overwhelming.


    4. ACTIONABILITY
    ────────────────
    Can a person act on the explanation to change the outcome?

    Particularly important for consequential decisions (loan approval, medical 
    diagnosis, hiring). An explanation like "you were denied because of your 
    race" is faithful but not actionable (and illegal). 
    An explanation like "you were denied because your debt-to-income ratio 
    exceeds 43% — reducing it by $200/month would likely result in approval" 
    is both faithful AND actionable.


    5. COMPLETENESS
    ───────────────
    Does the explanation account for the full prediction, or only part of it?

    SHAP values are complete by construction: they always sum to the model's 
    output (relative to a baseline). Many other methods are only partial.


    PROPERTY COMPARISON:
    ═══════════════════════════════════════════════════════════════════════
    Property        SHAP    LIME    Decision Tree    Linear Regression
    ─────────────────────────────────────────────────────────────────────
    Faithfulness    High    Low*    Perfect (intrinsic)   Perfect
    Stability       High    Low     High                  High
    Comprehensible  Med     High    High (small trees)    High
    Actionable      Med     High    High                  High
    Complete        Yes     No      Yes                   Yes
    ═══════════════════════════════════════════════════════════════════════
    * LIME faithfulness is only locally valid; degrades with distance from point

---

### A Mental Model: The Questions You Are Asking

The most practical way to navigate the taxonomy is to start with the 
question you need to answer, then let the question determine which 
technique belongs in which quadrant.

    QUESTION → SCOPE → TECHNIQUE FAMILY
    ════════════════════════════════════════════════════════════════

    "Which features does this model rely on most?"
        → GLOBAL + Feature attribution
        → Permutation importance, SHAP summary plots, PDP

    "Why did the model give THIS patient a high-risk score?"
        → LOCAL + Feature attribution
        → SHAP force plot, LIME

    "What would need to change for a different outcome?"
        → LOCAL + Example-based
        → Counterfactual explanations (DiCE)

    "Can I describe the model with simple rules?"
        → GLOBAL + Rule-based
        → Decision Tree surrogate, Rule lists

    "How does the output change as income increases?"
        → GLOBAL + Visual
        → PDP, ALE plots, ICE curves

    "Is this model fair across demographic groups?"
        → GLOBAL + Attribution + Fairness lens
        → SHAP grouped by subgroup, disparate impact analysis

    "What are the internal mechanisms that drive this model's behaviour?"
        → GLOBAL + Mechanistic
        → Probing classifiers, circuit analysis, neuron dissection

    ════════════════════════════════════════════════════════════════

    The single most important step before reaching for any XAI library:
    Write down in plain English the QUESTION you need answered.
    The question determines the scope. The scope determines the method.
    Applying a global method to a local question (or vice versa) is the 
    most common mistake in applied interpretability work.

---

### Where Each Method in this Folder Lives

    interpretability/
    ├── scope_and_taxonomy/        ← YOU ARE HERE (the map)
    │
    ├── intrinsic_models/          ← Intrinsic · Global + Local · Model-specific
    │   ├── linear_regression       Coefficients = direct explanation
    │   ├── decision_trees          Tree path = direct explanation
    │   └── rule_lists              Rules = direct explanation
    │
    ├── post_hoc_agnostic/         ← Post-hoc · Model-agnostic
    │   ├── SHAP                    Local + Global · Feature attribution
    │   ├── LIME                    Local · Feature attribution
    │   ├── anchors                 Local · Rule-based
    │   ├── PDP                     Global · Visual (marginal effect)
    │   ├── ICE                     Local (per-instance) · Visual
    │   └── ALE                     Global · Visual (better than PDP for correlated features)
    │
    ├── nn_specific_xai/           ← Post-hoc · Model-specific (neural nets)
    │   ├── GradCAM                 Local · Visual heatmap (CNNs)
    │   ├── saliency_maps           Local · Visual heatmap (differentiable)
    │   ├── LRP                     Local · Feature attribution (propagation-based)
    │   └── attention_viz           Local · Visual (Transformers)
    │
    ├── counterfactual/            ← Post-hoc · Local · Example-based
    │
    ├── concept_based/             ← Post-hoc · Global/Local · Concept-level
    │   └── TCAV                    "Does this model use the concept X?"
    │
    ├── mechanistic_interpretability/ ← Post-hoc · Global · Internal circuits
    │   ├── probing_classifiers     "What does representation layer L encode?"
    │   └── circuits                "Which components implement behaviour B?"
    │
    ├── model_calibration/         ← Confidence and uncertainty estimation
    │
    ├── evaluation_of_explanations/← Meta: are our explanations good?
    └── fairness_and_bias/         ← Explanations as an auditing tool

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔══════════════════╦══════════╦══════════════╦══════════════╦════════════════╗
    ║ Method           ║ Scope    ║ Method type  ║ Model reach  ║ Explanation    ║
    ║                  ║          ║              ║              ║ form           ║
    ╠══════════════════╬══════════╬══════════════╬══════════════╬════════════════╣
    ║ Linear Regression║ Both     ║ Intrinsic    ║ Specific     ║ Coefficients   ║
    ║ Decision Tree    ║ Both     ║ Intrinsic    ║ Specific     ║ Rules / paths  ║
    ║ GAM              ║ Global   ║ Intrinsic    ║ Specific     ║ Shape functions║
    ║ SHAP             ║ Both     ║ Post-hoc     ║ Agnostic*    ║ Feature scores ║
    ║ LIME             ║ Local    ║ Post-hoc     ║ Agnostic     ║ Feature weights║
    ║ Anchors          ║ Local    ║ Post-hoc     ║ Agnostic     ║ IF-THEN rules  ║
    ║ PDP              ║ Global   ║ Post-hoc     ║ Agnostic     ║ Marginal plot  ║
    ║ ICE              ║ Local    ║ Post-hoc     ║ Agnostic     ║ Per-instance   ║
    ║ ALE              ║ Global   ║ Post-hoc     ║ Agnostic     ║ Marginal plot  ║
    ║ GradCAM          ║ Local    ║ Post-hoc     ║ CNN-specific ║ Pixel heatmap  ║
    ║ LRP              ║ Local    ║ Post-hoc     ║ NN-specific  ║ Feature scores ║
    ║ Counterfactuals  ║ Local    ║ Post-hoc     ║ Agnostic     ║ Example-based  ║
    ║ TCAV             ║ Global   ║ Post-hoc     ║ NN-specific  ║ Concept score  ║
    ║ Probing          ║ Global   ║ Post-hoc     ║ NN-specific  ║ Layer readout  ║
    ╚══════════════════╩══════════╩══════════════╩══════════════╩════════════════╝
    * SHAP has model-specific variants (TreeExplainer, DeepExplainer) that are 
      much faster but require specific model types.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Runnable code demonstrations
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "Global vs Local: Same Model, Different Questions": {
        "description": "Demonstrates the difference between global and local explanations using a toy dataset and a trained RandomForest. "
                       "Shows how a globally important feature can be locally irrelevant for a specific prediction.",
        "runnable": True,
        "pipeline_cmd": "scope",
        "code": '''
"""
================================================================================
GLOBAL vs LOCAL EXPLANATIONS — THE SAME MODEL, TWO DIFFERENT QUESTIONS
================================================================================

This script trains one RandomForestClassifier on a simple loan-approval dataset
and then asks two completely different questions about it:

  GLOBAL question:  "Which features does this model rely on most, IN GENERAL?"
  LOCAL question:   "Why did the model deny loan application #42 specifically?"

The core insight: the answers can be completely different.
A feature that is globally important can be locally irrelevant for a
specific individual — and vice versa.

================================================================================
"""

import random
import math


# =============================================================================
# STEP 1: CREATE A SIMPLE DATASET
# =============================================================================
# We'll simulate a loan approval dataset with 4 features:
#   - income          (annual income in $k)
#   - debt_ratio      (debt as % of income)
#   - credit_score    (300–850)
#   - employment_years (years at current employer)
#
# Label: 1 = approved, 0 = denied
#
# True rules (ground truth that generated the data):
#   Approved IF (credit_score >= 650 AND debt_ratio <= 0.40) OR income >= 90
#   Denied otherwise

random.seed(42)

def generate_applicant():
    income = random.uniform(25, 120)          # $25k–$120k
    debt_ratio = random.uniform(0.10, 0.65)   # 10%–65%
    credit_score = random.uniform(500, 850)   # 500–850
    employment_years = random.uniform(0, 20)  # 0–20 years
    # Ground-truth label
    approved = int(
        (credit_score >= 650 and debt_ratio <= 0.40) or income >= 90
    )
    return [income, debt_ratio, credit_score, employment_years], approved

N = 500
dataset = [generate_applicant() for _ in range(N)]
X = [row[0] for row in dataset]
y = [row[1] for row in dataset]

feature_names = ["income", "debt_ratio", "credit_score", "employment_years"]

print("=" * 65)
print("  DATASET OVERVIEW")
print("=" * 65)
print(f"  Samples: {N}")
print(f"  Features: {feature_names}")
print(f"  Approvals: {sum(y)} / {N} ({100*sum(y)/N:.1f}%)")
print(f"  Denials:   {N - sum(y)} / {N} ({100*(N-sum(y))/N:.1f}%)")


# =============================================================================
# STEP 2: A MINIMAL DECISION TREE CLASSIFIER (no sklearn required)
# =============================================================================
# We'll build a simple 1-level decision tree per feature (a "stump")
# and rank features by how well each one splits the data.
# This is a simplified stand-in for full feature importance.

def gini_impurity(labels):
    """Gini impurity of a list of binary labels."""
    if not labels:
        return 0.0
    n = len(labels)
    p1 = sum(labels) / n
    p0 = 1 - p1
    return 1 - (p0**2 + p1**2)

def best_split_gini_reduction(feature_idx, X, y):
    """
    Find the split threshold on one feature that maximally reduces Gini impurity.
    Returns (best_threshold, gini_reduction).
    """
    values = sorted(set(row[feature_idx] for row in X))
    thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

    parent_gini = gini_impurity(y)
    n = len(y)
    best_reduction = -1
    best_thresh = None

    for thresh in thresholds:
        left_y  = [y[i] for i in range(n) if X[i][feature_idx] <= thresh]
        right_y = [y[i] for i in range(n) if X[i][feature_idx] >  thresh]
        if not left_y or not right_y:
            continue
        weighted = (len(left_y)/n) * gini_impurity(left_y) + \
                   (len(right_y)/n) * gini_impurity(right_y)
        reduction = parent_gini - weighted
        if reduction > best_reduction:
            best_reduction = reduction
            best_thresh = thresh

    return best_thresh, best_reduction


# =============================================================================
# STEP 3: GLOBAL EXPLANATION — Feature Importance
# =============================================================================
# Ask: "Which features does the model rely on most, across all 500 applicants?"
# Method: rank features by how much Gini impurity they can reduce (globally).

print()
print("=" * 65)
print("  GLOBAL EXPLANATION: Feature Importance (Gini Reduction)")
print("=" * 65)
print()
print("  Question: 'Which features matter most to the model IN GENERAL?'")
print()

importances = {}
thresholds  = {}
for i, fname in enumerate(feature_names):
    thresh, reduction = best_split_gini_reduction(i, X, y)
    importances[fname] = reduction
    thresholds[fname]  = thresh

# Normalise importances to sum to 1
total = sum(importances.values())
norm_importances = {k: v/total for k, v in importances.items()}

# Sort descending
sorted_features = sorted(norm_importances.items(), key=lambda x: -x[1])

print(f"  {'Feature':<20} {'Importance':>12}  {'Bar'}")
print(f"  {'-'*55}")
for fname, imp in sorted_features:
    bar = "█" * int(imp * 40)
    print(f"  {fname:<20} {imp:>11.1%}  {bar}")

print()
print("  GLOBAL INSIGHT:")
print(f"  ► '{sorted_features[0][0]}' is the most globally important feature.")
print(f"    The model relies on it most heavily across all 500 applicants.")
print(f"  ► 'employment_years' has low global importance —")
print(f"    it rarely drives the decision on average.")


# =============================================================================
# STEP 4: LOCAL EXPLANATION — One specific applicant
# =============================================================================
# Ask: "Why was THIS specific applicant denied?"
# Method: For each feature, check whether this applicant's value is on the 
#         'bad' or 'good' side of the globally-learned split threshold.
#         Compute a local importance score: how far from the threshold is this
#         applicant, and is it in the direction that hurts them?

print()
print("=" * 65)
print("  LOCAL EXPLANATION: One Specific Applicant")
print("=" * 65)
print()

# Construct a specific applicant (denied by the ground truth rule)
# High income but terrible credit score AND high debt ratio
applicant = {
    "income":            95.0,   # high income ($95k) — this HELPS
    "debt_ratio":         0.55,  # very high debt ratio — this HURTS
    "credit_score":      580.0,  # low credit score — this HURTS
    "employment_years":   3.0,   # mediocre employment history
}

# Ground truth for this applicant:
# income >= 90 → should be approved! Let's see what a simple model says...
# (our simple stump model might not capture the OR rule well)

print("  Applicant profile:")
for fname, val in applicant.items():
    print(f"    {fname:<22}: {val}")

print()

# Compute local explanation: for each feature, how does this applicant's 
# value compare to the learned decision threshold?
print(f"  {'Feature':<20} {'Value':>8}  {'Threshold':>10}  {'Direction':>12}  {'Local impact'}")
print(f"  {'-'*75}")

local_scores = {}
applicant_values = [applicant[f] for f in feature_names]

for fname, (_, imp) in zip(feature_names, sorted_features):
    idx   = feature_names.index(fname)
    val   = applicant_values[idx]
    thresh = thresholds[fname]

    # For debt_ratio: being ABOVE the threshold hurts (higher debt = worse)
    # For credit_score, income, employment_years: being BELOW hurts
    if fname == "debt_ratio":
        direction = "HURTS ▲" if val > thresh else "helps ▼"
        dist = abs(val - thresh)
    else:
        direction = "HURTS ▼" if val < thresh else "helps ▲"
        dist = abs(val - thresh)

    local_scores[fname] = dist
    print(f"  {fname:<20} {val:>8.2f}  {thresh:>10.2f}  {direction:>12}  {dist:>8.3f}")

print()
print("  LOCAL INSIGHT:")
print("  ► For this specific applicant:")

# Sort by distance from threshold
sorted_local = sorted(local_scores.items(), key=lambda x: -x[1])
print(f"    Most impactful features for THIS decision:")
for fname, dist in sorted_local[:2]:
    idx = feature_names.index(fname)
    val = applicant_values[idx]
    thresh = thresholds[fname]
    print(f"    • '{fname}': value={val:.2f}, threshold={thresh:.2f}, "
          f"distance={dist:.3f}")


# =============================================================================
# STEP 5: THE KEY COMPARISON — GLOBAL vs LOCAL
# =============================================================================

print()
print("=" * 65)
print("  THE CORE INSIGHT: GLOBAL ≠ LOCAL")
print("=" * 65)
print()
print("  GLOBAL importance ranking (averaged over all 500 applicants):")
for i, (fname, imp) in enumerate(sorted_features, 1):
    print(f"    {i}. {fname:<22} ({imp:.1%})")

print()
print("  LOCAL importance ranking (for THIS specific applicant):")
for i, (fname, dist) in enumerate(sorted_local, 1):
    print(f"    {i}. {fname:<22} (distance from threshold: {dist:.3f})")

print()
print("  ⚠️  Notice: the rankings are DIFFERENT.")
print("     A feature that is important globally may not be what drove")
print("     this specific decision — and vice versa.")
print()
print("  This is why you need BOTH:")
print("    • Global explanations → to understand the model overall")
print("    • Local explanations  → to justify or contest a specific decision")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "Intrinsic vs Post-hoc: Faithfulness Comparison": {
        "description": "Compares an intrinsically interpretable model (logistic regression) against a black-box model "
                       "with a post-hoc explanation (SHAP approximation). Shows how post-hoc explanations are "
                       "approximations, not ground truth — and quantifies the faithfulness gap.",
        "runnable": True,
        "pipeline_cmd": "scope",
        "code": '''
"""
================================================================================
INTRINSIC vs POST-HOC EXPLANATIONS — THE FAITHFULNESS GAP
================================================================================

Core question this script answers:

    When we apply a post-hoc explanation to a black-box model, how accurate
    is that explanation? And how does it compare to the "perfect" explanation
    you get from an intrinsically interpretable model?

We demonstrate this by:
  1. Training an intrinsically interpretable model (Logistic Regression)
     → Its coefficients ARE the faithful, complete explanation.

  2. Training a black-box model (approximated here as a non-linear function)
     → Its true weights are hidden; we only see inputs and outputs.

  3. Applying a simplified post-hoc attribution (influence of each feature)
     → This approximates what SHAP / LIME would do.

  4. Comparing the post-hoc attribution to the true feature importances.
     → This gap is the "faithfulness gap" of post-hoc explanations.

================================================================================
"""

import random
import math


random.seed(7)

# =============================================================================
# DATASET: 3-feature binary classification
# =============================================================================
# Features: x1 (salary $k), x2 (years experience), x3 (education: 0/1)
# True rule (for the black-box):
#   score = 0.8*x1 + 1.5*x2^2 + 2.0*x3 + noise   (non-linear in x2!)
# Label: 1 if score > median(score), else 0

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-500, min(500, z))))

N = 300
X_raw = []
y = []
for _ in range(N):
    x1 = random.gauss(50, 15)    # salary ($k), mean 50k
    x2 = random.gauss(5, 3)      # years exp, mean 5y
    x3 = random.choice([0, 1])   # education level
    X_raw.append([x1, x2, x3])

# Normalise to [0, 1] for comparability
def normalise(X, idx):
    col = [row[idx] for row in X]
    mn, mx = min(col), max(col)
    return [(v - mn) / (mx - mn + 1e-9) for v in col]

x1_n = normalise(X_raw, 0)
x2_n = normalise(X_raw, 1)
x3_n = normalise(X_raw, 2)
X = [[x1_n[i], x2_n[i], x3_n[i]] for i in range(N)]

# Black-box ground truth (non-linear)
scores = [0.4 * x1_n[i] + 1.5 * x2_n[i]**2 + 0.9 * x3_n[i]
          for i in range(N)]
median_score = sorted(scores)[N // 2]
y = [1 if s > median_score else 0 for s in scores]

feature_names = ["salary (normalised)", "experience (normalised)", "education (0/1)"]

print("=" * 65)
print("  SETUP")
print("=" * 65)
print(f"  Samples: {N}, Features: 3")
print(f"  True black-box rule (hidden from explainer):")
print(f"    score = 0.4·salary + 1.5·experience² + 0.9·education")
print(f"    Note: experience has a NON-LINEAR (squared) effect!")
print(f"  Labels: 1 if score > median, else 0")
print(f"  Approvals: {sum(y)} / {N}")


# =============================================================================
# PART A: INTRINSIC MODEL — LOGISTIC REGRESSION
# =============================================================================
# Logistic regression fits: log(p/(1-p)) = w0 + w1*x1 + w2*x2 + w3*x3
# We train it with gradient ascent on log-likelihood.
# The weights w1, w2, w3 ARE the explanation. No approximation needed.

def logistic_predict(X, weights, bias):
    return [sigmoid(sum(w*x for w, x in zip(weights, row)) + bias) for row in X]

def log_likelihood(probs, y):
    return sum(
        y[i] * math.log(probs[i] + 1e-10) +
        (1 - y[i]) * math.log(1 - probs[i] + 1e-10)
        for i in range(len(y))
    )

# Train logistic regression via gradient ascent
w = [0.0, 0.0, 0.0]
b = 0.0
lr_w = 0.5
n = len(X)

for epoch in range(400):
    probs = logistic_predict(X, w, b)
    for j in range(3):
        grad = sum((y[i] - probs[i]) * X[i][j] for i in range(n)) / n
        w[j] += lr_w * grad
    b += lr_w * sum((y[i] - probs[i]) for i in range(n)) / n

probs = logistic_predict(X, w, b)
preds_lr = [1 if p > 0.5 else 0 for p in probs]
acc_lr = sum(preds_lr[i] == y[i] for i in range(n)) / n

print()
print("=" * 65)
print("  PART A: INTRINSIC MODEL — Logistic Regression")
print("=" * 65)
print(f"  Accuracy: {acc_lr:.1%}")
print()
print("  Learned coefficients (THE faithful explanation — no approximation):")
for fname, coef in zip(feature_names, w):
    direction = "↑ positive" if coef > 0 else "↓ negative"
    bar = ("+" if coef > 0 else "-") * int(abs(coef) * 10)
    print(f"    {fname:<28}: {coef:+.4f}  {direction:<12}  {bar}")
print(f"    bias                        : {b:+.4f}")
print()
print("  ✓ These coefficients ARE the model. The explanation has perfect")
print("    faithfulness by construction. No approximation involved.")
print()
print("  ⚠️  However: logistic regression assumes a LINEAR relationship.")
print("    The true rule has experience² (non-linear). This model will")
print("    misattribute: it gives experience a moderate weight, but")
print("    it can't capture the quadratic effect perfectly.")


# =============================================================================
# PART B: POST-HOC EXPLANATION — Perturbation-based attribution
# =============================================================================
# We treat the true non-linear score function as our "black box."
# Post-hoc attribution: for each feature, measure how much the model's
# average prediction changes when that feature is permuted (shuffled).
# This is the logic behind permutation importance and partially behind SHAP.

def black_box_predict(X):
    """The true non-linear model — imagine this is a neural network."""
    out = []
    for row in X:
        s = 0.4 * row[0] + 1.5 * row[1]**2 + 0.9 * row[2]
        out.append(1 if s > median_score else 0)
    return out

baseline_preds = black_box_predict(X)
baseline_acc   = sum(baseline_preds[i] == y[i] for i in range(n)) / n

def permutation_importance(feature_idx, X, y, n_repeats=20):
    """
    Measure how much accuracy drops when feature_idx is randomly shuffled.
    A large drop = the feature is important.
    A small drop = the feature is not very important for predictions.
    """
    drops = []
    col_vals = [X[i][feature_idx] for i in range(n)]
    for _ in range(n_repeats):
        shuffled = col_vals[:]
        random.shuffle(shuffled)
        X_perm = [row[:] for row in X]
        for i in range(n):
            X_perm[i][feature_idx] = shuffled[i]
        perm_preds = black_box_predict(X_perm)
        perm_acc = sum(perm_preds[i] == y[i] for i in range(n)) / n
        drops.append(baseline_acc - perm_acc)
    return sum(drops) / len(drops)

print()
print("=" * 65)
print("  PART B: POST-HOC EXPLANATION — Permutation Importance")
print("=" * 65)
print(f"  Black-box model accuracy: {baseline_acc:.1%}")
print()
print("  Computing permutation importance (takes a moment)...")

posthoc_importances = {}
for i, fname in enumerate(feature_names):
    imp = permutation_importance(i, X, y)
    posthoc_importances[fname] = imp

total_imp = sum(posthoc_importances.values()) + 1e-9
norm_posthoc = {k: v / total_imp for k, v in posthoc_importances.items()}

print()
print("  Post-hoc attributions (normalised permutation importance):")
for fname in feature_names:
    imp = norm_posthoc[fname]
    bar = "█" * int(imp * 30)
    print(f"    {fname:<28}: {imp:.1%}  {bar}")


# =============================================================================
# PART C: THE FAITHFULNESS COMPARISON
# =============================================================================
# Now compare what each explanation says about feature importance
# vs the true underlying rule.

TRUE_IMPORTANCES = {
    # Derived from the true coefficients: 0.4, 1.5 (quadratic), 0.9
    # Squared term means experience is MORE important than linear coef suggests
    "salary (normalised)":      0.40 / (0.40 + 1.80 + 0.90),  # ~16%
    "experience (normalised)":  1.80 / (0.40 + 1.80 + 0.90),  # ~57%  (amplified by ^2)
    "education (0/1)":          0.90 / (0.40 + 1.80 + 0.90),  # ~27%
}

INTRINSIC_IMPORTANCES = {}
total_abs_w = sum(abs(c) for c in w) + 1e-9
for i, fname in enumerate(feature_names):
    INTRINSIC_IMPORTANCES[fname] = abs(w[i]) / total_abs_w

print()
print("=" * 65)
print("  PART C: FAITHFULNESS COMPARISON")
print("=" * 65)
print()
print(f"  {'Feature':<28}  {'True':>8}  {'Intrinsic LR':>14}  {'Post-hoc':>10}")
print(f"  {'-'*68}")
for fname in feature_names:
    t   = TRUE_IMPORTANCES[fname]
    intr = INTRINSIC_IMPORTANCES[fname]
    post = norm_posthoc[fname]
    print(f"  {fname:<28}  {t:>7.1%}  {intr:>13.1%}  {post:>9.1%}")

print()
print("  INTERPRETATION:")
print()
print("  1. The TRUE importance of 'experience' is high (~57%) because")
print("     the quadratic rule amplifies it. Neither the intrinsic model")
print("     nor the post-hoc explanation captures this perfectly.")
print()
print("  2. The INTRINSIC model (logistic regression) is forced to fit")
print("     a linear rule to non-linear data → its explanation is")
print("     FAITHFUL to what LR learned, but LR's model is wrong.")
print("     'Faithful to an imperfect model' ≠ 'faithful to reality'.")
print()
print("  3. The POST-HOC permutation importance is closer to the truth")
print("     here — because it probes the TRUE model (the non-linear")
print("     function) rather than an approximation of it.")
print()
print("  KEY LESSON:")
print("  ─────────────────────────────────────────────────────────────")
print("  Intrinsic model → perfectly faithful to WHAT THE MODEL DOES.")
print("                    Only as accurate as the model itself.")
print()
print("  Post-hoc method → an approximation of what the black-box does.")
print("                    Can be more faithful if the black-box is right.")
print("                    But the approximation itself can introduce error.")
print()
print("  There is no free lunch: ALL explanations involve trade-offs.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "Properties of Good Explanations: Stability & Faithfulness": {
        "description": "Directly measures two core properties of explanation quality — faithfulness (does masking "
                       "important features change the prediction?) and stability (do similar inputs get similar "
                       "explanations?). No external libraries required.",
        "runnable": True,
        "pipeline_cmd": "scope",
        "code": '''
"""
================================================================================
EVALUATING EXPLANATIONS: FAITHFULNESS AND STABILITY
================================================================================

Producing an explanation is easy. Knowing whether your explanation is GOOD
is much harder. This script implements two core evaluation metrics:

  FAITHFULNESS:  If the explanation says feature X is important,
                 does zeroing out feature X actually change the prediction?
                 (Higher faithfulness = more reliable explanations)

  STABILITY:     If two inputs are similar, are their explanations similar?
                 (Higher stability = more consistent, trustworthy explanations)

We compute these metrics for two explanation strategies and compare them:
  - Strategy A: random attribution (a bad baseline — what failure looks like)
  - Strategy B: perturbation-based attribution (a simple but principled method)

================================================================================
"""

import random
import math


random.seed(0)


# =============================================================================
# SIMPLE MODEL AND DATASET
# =============================================================================

def model(x):
    """
    Our 'black box'. True rule:
        score = 2.0*x[0] - 1.5*x[1] + 0.5*x[2] + 0.1*x[3]
    Feature 0 and 1 matter a lot. Features 2 and 3 barely matter.
    """
    return 2.0*x[0] - 1.5*x[1] + 0.5*x[2] + 0.1*x[3]

def predict(x):
    return 1 if model(x) > 0 else 0

# Create a dataset of 200 random inputs
N = 200
X = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(N)]
feature_names = ["f0 (important+)", "f1 (important-)", "f2 (minor)", "f3 (noise)"]


# =============================================================================
# EXPLANATION STRATEGIES
# =============================================================================

def explain_random(x):
    """
    Strategy A: RANDOM attribution.
    Assign random scores to each feature. This is our 'bad baseline'.
    A real explanation should do much better than this.
    """
    scores = [random.uniform(0, 1) for _ in range(len(x))]
    total = sum(scores)
    return [s / total for s in scores]


def explain_perturbation(x, n_samples=50):
    """
    Strategy B: Perturbation-based attribution.
    For each feature i, zero it out and measure how much the model's
    output changes. A large change = the feature is important.

    This is the core idea behind many XAI methods (SHAP marginal, LIME).
    """
    base_output = model(x)
    importances = []
    for i in range(len(x)):
        impacts = []
        for _ in range(n_samples):
            x_perturbed = x[:]
            # Replace feature i with a random value (marginalising it out)
            x_perturbed[i] = random.uniform(-1, 1)
            impacts.append(abs(model(x_perturbed) - base_output))
        importances.append(sum(impacts) / len(impacts))
    total = sum(importances) + 1e-9
    return [v / total for v in importances]


# =============================================================================
# METRIC 1: FAITHFULNESS
# =============================================================================
# For each instance:
#   1. Get explanation → identifies the top-k most important features
#   2. Zero out those features in the input
#   3. Measure how much the model output changes
#
# Faithful explanation → zeroing top features causes a BIG change
# Unfaithful explanation → zeroing top features causes a SMALL change

def measure_faithfulness(explain_fn, X, top_k=2):
    """
    Compute average prediction change when top_k 'important' features are zeroed.
    Higher = more faithful.
    """
    changes = []
    for x in X:
        attr = explain_fn(x)
        # Find the top_k most attributed features
        ranked = sorted(range(len(attr)), key=lambda i: -attr[i])
        top_features = ranked[:top_k]

        original_pred = model(x)
        x_masked = x[:]
        for i in top_features:
            x_masked[i] = 0.0  # zero out the 'important' features
        masked_pred = model(x_masked)

        changes.append(abs(original_pred - masked_pred))

    return sum(changes) / len(changes)


print("=" * 65)
print("  METRIC 1: FAITHFULNESS")
print("=" * 65)
print()
print("  Test: zero out the features the explanation ranks as most important.")
print("  If the explanation is faithful, this should cause a BIG change.")
print()
print("  Computing faithfulness for 200 instances (top-2 features)...")
print()

faith_random = measure_faithfulness(explain_random, X, top_k=2)
faith_perturb = measure_faithfulness(explain_perturbation, X, top_k=2)

print(f"  Strategy A (random attribution):     avg Δoutput = {faith_random:.4f}")
print(f"  Strategy B (perturbation-based):     avg Δoutput = {faith_perturb:.4f}")
print()

if faith_perturb > faith_random:
    ratio = faith_perturb / (faith_random + 1e-9)
    print(f"  ✓ Strategy B is {ratio:.1f}× more faithful than random.")
    print(f"    When Strategy B says 'feature X is important', zeroing X")
    print(f"    actually changes the model's output by {faith_perturb:.3f} on average.")
    print(f"    Random attribution only achieves {faith_random:.3f} — barely better")
    print(f"    than zeroing random features.")


# =============================================================================
# METRIC 2: STABILITY
# =============================================================================
# For each pair of 'similar' inputs (those with small Euclidean distance),
# compute how similar their explanations are (using cosine similarity).
#
# Stable explanation → similar inputs → similar attributions
# Unstable explanation → similar inputs → wildly different attributions

def cosine_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai**2 for ai in a)) + 1e-9
    norm_b = math.sqrt(sum(bi**2 for bi in b)) + 1e-9
    return dot / (norm_a * norm_b)

def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

def measure_stability(explain_fn, X, n_pairs=100, similarity_threshold=0.3):
    """
    For pairs of inputs closer than similarity_threshold in input space,
    measure how similar their explanations are (cosine similarity of attributions).
    Returns average explanation similarity for 'nearby' pairs.
    """
    # Pre-compute explanations
    explanations = [explain_fn(x) for x in X]

    sims = []
    checked = 0
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if checked >= n_pairs:
                break
            dist = euclidean_distance(X[i], X[j])
            if dist < similarity_threshold:
                sim = cosine_similarity(explanations[i], explanations[j])
                sims.append(sim)
                checked += 1
        if checked >= n_pairs:
            break

    return sum(sims) / len(sims) if sims else 0.0, len(sims)

print()
print("=" * 65)
print("  METRIC 2: STABILITY")
print("=" * 65)
print()
print("  Test: for pairs of inputs that are close in input space (dist < 0.3),")
print("  how similar are their explanations? (cosine similarity, 1.0 = identical)")
print()
print("  Computing stability (finding similar input pairs)...")
print()

stab_random,  n_pairs_r = measure_stability(explain_random,      X)
stab_perturb, n_pairs_p = measure_stability(explain_perturbation, X)

print(f"  Strategy A (random attribution):")
print(f"    Avg explanation similarity for nearby pairs: {stab_random:.4f}")
print(f"    (analysed {n_pairs_r} nearby pairs)")
print()
print(f"  Strategy B (perturbation-based):")
print(f"    Avg explanation similarity for nearby pairs: {stab_perturb:.4f}")
print(f"    (analysed {n_pairs_p} nearby pairs)")
print()

if stab_perturb > stab_random:
    print(f"  ✓ Strategy B is more stable: similar inputs get similar explanations.")
    print(f"    Cosine similarity of {stab_perturb:.3f} (vs {stab_random:.3f} for random).")
    print(f"    A cosine similarity near 1.0 = explanations point in the same direction.")


# =============================================================================
# STEP 4: WHAT DOES A GOOD EXPLANATION LOOK LIKE?
# =============================================================================
# Show a single-instance explanation from Strategy B and discuss it.

print()
print("=" * 65)
print("  A CONCRETE GOOD EXPLANATION — Instance #0")
print("=" * 65)
print()

x0 = X[0]
attr_random  = explain_random(x0)
attr_perturb = explain_perturbation(x0)

true_model_output = model(x0)
print(f"  Input:  {[round(v, 3) for v in x0]}")
print(f"  Model output (raw score): {true_model_output:.4f}")
print(f"  Prediction: {'POSITIVE (1)' if true_model_output > 0 else 'NEGATIVE (0)'}")
print()
print(f"  {'Feature':<22}  {'Random attr':>12}  {'Perturbation attr':>18}  {'True importance'}")
print(f"  {'-'*75}")

# True importance based on coefficient magnitudes: 2.0, 1.5, 0.5, 0.1
true_total = 2.0 + 1.5 + 0.5 + 0.1
true_imps = [2.0/true_total, 1.5/true_total, 0.5/true_total, 0.1/true_total]

for i, fname in enumerate(feature_names):
    r = attr_random[i]
    p = attr_perturb[i]
    t = true_imps[i]
    match = "✓" if abs(p - t) < abs(r - t) else " "
    print(f"  {fname:<22}  {r:>12.1%}  {p:>18.1%}  {t:>11.1%}  {match}")

print()
print("  ✓ marks where perturbation attribution is closer to truth than random.")
print()
print("  KEY TAKEAWAYS:")
print("  ─────────────────────────────────────────────────────────────────────")
print("  1. FAITHFULNESS measures 'does zeroing important features break the model?'")
print("     → Perturbation-based attribution is far more faithful than random.")
print()
print("  2. STABILITY measures 'do similar inputs get similar explanations?'")
print("     → Perturbation-based is more stable; random explanations are erratic.")
print()
print("  3. Even a principled method has imperfect faithfulness (< 1.0).")
print("     Post-hoc explanations are ALWAYS approximations — this is fundamental.")
print("     The only perfectly faithful explanation of a model is the model itself.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "Taxonomy Navigator — Classify Any XAI Method": {
        "description": "An interactive taxonomy classifier. Given a method name, prints its coordinates on all three "
                       "axes (scope, method type, model reach) and describes what it can and cannot answer. "
                       "A quick reference guide for the whole field.",
        "runnable": True,
        "pipeline_cmd": "scope",
        "code": '''
"""
================================================================================
XAI TAXONOMY NAVIGATOR — Classify any interpretability method
================================================================================

This script is a reference tool. For any method in the interpretability field,
it prints the method's coordinates on the three core axes and explains:
  1. What questions it can answer
  2. What questions it CANNOT answer
  3. When you should reach for it

Run it to see the full taxonomy, or add your own method to the METHODS dict.
================================================================================
"""

METHODS = {
    "Linear Regression": {
        "scope":        "Global AND Local",
        "method_type":  "Intrinsic",
        "model_reach":  "Model-specific (linear models only)",
        "explanation_form": "Coefficients (one per feature)",
        "can_answer": [
            "Which features matter most across the whole dataset?",
            "For any specific input, what drove the prediction?",
            "How does a unit change in feature X affect the output?",
            "What is the decision boundary?",
        ],
        "cannot_answer": [
            "Non-linear effects (interaction terms, curves)",
            "Why a complex model (neural net, forest) made a specific decision",
        ],
        "reach_for_when": [
            "You want a fully interpretable, auditable model",
            "Regulatory environment requires explainability by design",
            "Data relationships are approximately linear",
            "Accuracy can be traded for transparency",
        ],
        "folder": "intrinsic_models/linear_regression",
    },

    "Decision Tree": {
        "scope":        "Global AND Local",
        "method_type":  "Intrinsic",
        "model_reach":  "Model-specific (tree models only)",
        "explanation_form": "IF-THEN rules / tree paths",
        "can_answer": [
            "What rule describes the model's overall behaviour?",
            "What path through the tree led to this specific prediction?",
            "What is the decision boundary in human-readable form?",
        ],
        "cannot_answer": [
            "Non-axis-aligned decision boundaries (forces rectangular splits)",
            "Complex patterns in high-dimensional data (needs many nodes → less readable)",
            "Why a neural net or ensemble made a decision",
        ],
        "reach_for_when": [
            "You need an inherently interpretable model",
            "Stakeholders need to understand rules in natural language",
            "Depth of tree is small (3–5 levels) for readability",
        ],
        "folder": "intrinsic_models/decision_trees",
    },

    "SHAP": {
        "scope":        "Local (per-instance) — can be aggregated for Global",
        "method_type":  "Post-hoc",
        "model_reach":  "Model-agnostic (but faster model-specific variants exist: TreeExplainer, DeepExplainer)",
        "explanation_form": "Feature attribution scores (additive, sum to model output)",
        "can_answer": [
            "Why did the model make THIS specific prediction? (local)",
            "Which features does the model rely on most globally? (SHAP summary plot)",
            "How does feature X affect output across the distribution? (SHAP dependence plot)",
            "How does each feature push prediction above or below the baseline?",
        ],
        "cannot_answer": [
            "What the model would predict for a different input (use counterfactuals)",
            "Whether the model's reasoning is correct or fair (SHAP is descriptive, not normative)",
            "Internal mechanisms of a neural network (it's still post-hoc)",
            "Non-additive interactions perfectly (SHAP interaction values help, but are approximate)",
        ],
        "reach_for_when": [
            "You need to explain specific predictions to stakeholders",
            "Debugging: why is the model over-relying on a suspicious feature?",
            "Fairness auditing: does the model rely on protected attributes?",
            "You have a tree-based model: TreeExplainer is exact AND fast",
        ],
        "folder": "post_hoc_agnostic/SHAP",
    },

    "LIME": {
        "scope":        "Local (one instance at a time)",
        "method_type":  "Post-hoc",
        "model_reach":  "Model-agnostic",
        "explanation_form": "Coefficients of a locally-fitted linear model",
        "can_answer": [
            "What simple rule approximately describes the model's behaviour NEAR this prediction?",
            "Which features pushed this specific prediction up or down?",
            "How does a slight change in the input space affect the output locally?",
        ],
        "cannot_answer": [
            "Global model behaviour (LIME is purely local — do not generalise it!)",
            "Mathematically guaranteed faithful explanations (approximation can be unstable)",
            "Interactions between features (it's additive by design)",
        ],
        "reach_for_when": [
            "You need a quick, intuitive explanation for one prediction",
            "Explaining text or image models (LIME works well with segmented inputs)",
            "You don't have access to SHAP for your model type",
            "Your audience can understand a local linear approximation",
        ],
        "folder": "post_hoc_agnostic/LIME",
    },

    "PDP (Partial Dependence Plot)": {
        "scope":        "Global",
        "method_type":  "Post-hoc",
        "model_reach":  "Model-agnostic",
        "explanation_form": "Line/surface plot: feature value → average model output",
        "can_answer": [
            "What is the average relationship between feature X and the model's output?",
            "Is the relationship linear, monotone, or does it have a threshold?",
            "Does the model exhibit a non-linear response to income, age, temperature, etc.?",
        ],
        "cannot_answer": [
            "Individual-level effects (use ICE plots — one line per instance)",
            "Correct effects when features are correlated (use ALE instead — PDP can be misleading)",
            "Local explanations for specific predictions",
        ],
        "reach_for_when": [
            "You want to understand the overall shape of a feature's effect",
            "Communicating model behaviour to non-technical audiences with a clear plot",
            "Features are approximately independent (if correlated, use ALE instead)",
        ],
        "folder": "post_hoc_agnostic/global_visualizations/PDP",
    },

    "ALE (Accumulated Local Effects)": {
        "scope":        "Global",
        "method_type":  "Post-hoc",
        "model_reach":  "Model-agnostic",
        "explanation_form": "Line plot: feature value → accumulated local effect on output",
        "can_answer": [
            "What is the average effect of feature X on model output, even when features are correlated?",
            "Unbiased version of the PDP question for correlated features",
        ],
        "cannot_answer": [
            "Local (per-instance) explanations",
            "Exact feature attributions for individual predictions",
        ],
        "reach_for_when": [
            "Features are correlated and PDP would be misleading",
            "You want a more statistically rigorous global effect plot",
        ],
        "folder": "post_hoc_agnostic/global_visualizations/ALE",
    },

    "Counterfactual Explanations": {
        "scope":        "Local",
        "method_type":  "Post-hoc",
        "model_reach":  "Model-agnostic",
        "explanation_form": "A modified input that would change the model's prediction",
        "can_answer": [
            "What is the smallest change to this input that would flip the prediction?",
            "What actionable change could this person make to get a different outcome?",
            "Is this decision boundary reasonable?",
        ],
        "cannot_answer": [
            "Why the model makes decisions globally",
            "Which features are most important on average",
        ],
        "reach_for_when": [
            "Providing recourse to individuals affected by a decision (loan denial, etc.)",
            "GDPR / regulation requires actionable explanations",
            "Testing model sensitivity and edge cases",
        ],
        "folder": "counterfactual/",
    },

    "GradCAM": {
        "scope":        "Local",
        "method_type":  "Post-hoc",
        "model_reach":  "Model-specific (CNNs only)",
        "explanation_form": "Heatmap highlighting important image regions",
        "can_answer": [
            "Which region of this image drove the CNN's classification?",
            "Is the model attending to the object or to spurious background features?",
            "Where should I look to understand why this image was misclassified?",
        ],
        "cannot_answer": [
            "Why the model makes decisions globally across all images",
            "Feature-level explanations for non-image (tabular) data",
            "Explanations for non-convolutional models",
        ],
        "reach_for_when": [
            "Debugging a CNN classifier",
            "Verifying that a medical imaging model focuses on the right anatomy",
            "Explaining image predictions to domain experts",
        ],
        "folder": "nn_specific_xai/gradient_based/GradCAM",
    },

    "Probing Classifiers": {
        "scope":        "Global",
        "method_type":  "Post-hoc",
        "model_reach":  "Model-specific (neural networks with accessible intermediate representations)",
        "explanation_form": "Accuracy of a simple linear classifier on top of an intermediate layer",
        "can_answer": [
            "Does layer L of this neural network encode concept C?",
            "At what depth does the model start encoding syntactic structure?",
            "Do intermediate representations separate classes we care about?",
        ],
        "cannot_answer": [
            "Per-instance explanations",
            "Why the model uses a feature (probing shows WHAT is encoded, not WHY)",
        ],
        "reach_for_when": [
            "Researching what neural networks learn internally",
            "Understanding transformer / LLM representations",
            "Diagnosing whether a representation layer contains relevant information",
        ],
        "folder": "mechanistic_interpretability/probing_classifiers",
    },
}


def print_method(name, info, verbose=True):
    width = 68
    print()
    print("═" * width)
    print(f"  {name}")
    print("═" * width)
    print(f"  Folder:          {info['folder']}")
    print(f"  Scope:           {info['scope']}")
    print(f"  Method type:     {info['method_type']}")
    print(f"  Model reach:     {info['model_reach']}")
    print(f"  Explanation form:{info['explanation_form']}")
    if verbose:
        print()
        print("  CAN answer:")
        for q in info["can_answer"]:
            print(f"    ✓ {q}")
        print()
        print("  CANNOT answer:")
        for q in info["cannot_answer"]:
            print(f"    ✗ {q}")
        print()
        print("  Reach for this when:")
        for r in info["reach_for_when"]:
            print(f"    → {r}")
    print("═" * width)


# =============================================================================
# PRINT FULL TAXONOMY OVERVIEW
# =============================================================================

print()
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║           XAI METHOD TAXONOMY — FULL REFERENCE                       ║")
print("╚══════════════════════════════════════════════════════════════════════╝")
print()

# Print a compact overview table first
print(f"  {'Method':<35}  {'Scope':<12}  {'Type':<12}  {'Reach'}")
print(f"  {'-'*78}")
for name, info in METHODS.items():
    scope_short = "Global+Local" if "AND" in info["scope"] else (
                  "Global" if "Global" in info["scope"] else "Local")
    type_short  = "Intrinsic" if "Intrinsic" in info["method_type"] else "Post-hoc"
    reach_short = "Agnostic" if "agnostic" in info["model_reach"] else "Specific"
    print(f"  {name:<35}  {scope_short:<12}  {type_short:<12}  {reach_short}")

print()
print("  (Detailed profiles below)")

# Print detailed profiles for a selection
detailed_methods = ["SHAP", "LIME", "PDP (Partial Dependence Plot)", "Counterfactual Explanations"]
for name in detailed_methods:
    print_method(name, METHODS[name], verbose=True)

print()
print("  ── HOW TO USE THIS NAVIGATOR ─────────────────────────────────────")
print()
print("  Step 1: Write down your question in plain English.")
print("  Step 2: Determine scope (global 'overall' or local 'this prediction').")
print("  Step 3: Check if your model architecture allows model-specific methods.")
print("  Step 4: Pick the technique from the folder whose CAN ANSWER list matches.")
print()
print("  Common mistake: using a global method to answer a local question.")
print("  Example: using PDP to explain why applicant #42 was denied → WRONG.")
print("           PDP shows the average effect across all applicants.")
print("           Use SHAP or LIME for a local question.")
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
    """Render all operations with code display and optional run buttons."""
    import streamlit as st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if "tok_step_status" not in st.session_state:
        st.session_state.tok_step_status = {}
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
    visual_html = ""
    visual_height = 400
    # try:
    #     from interpretability.visuals.scope_and_taxonomy import (
    #         SCOPE_TAXONOMY_VISUAL_HTML,
    #         SCOPE_TAXONOMY_VISUAL_HEIGHT,
    #     )
    #     visual_html = SCOPE_TAXONOMY_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SCOPE_TAXONOMY_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[scope_and_taxonomy.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": visual_html,
        "visual_height": visual_height,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }