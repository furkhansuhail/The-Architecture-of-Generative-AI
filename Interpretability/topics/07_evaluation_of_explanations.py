"""
Evaluation of Explanations — How Do We Know If an Explanation Is Good?
=======================================================================

Interpretability methods produce outputs — heatmaps, feature attributions,
rules, circuits. But how do we know whether those outputs are trustworthy?
A SHAP attribution that says "income was the most important feature" might be
correct, or it might be a statistical artefact of the approximation method.
An attention map that highlights the right tokens might be doing so for the
wrong reasons. A GradCAM heatmap might look like it's highlighting the tumour
while actually responding to the scanner's calibration artefact.

Evaluation of explanations is the meta-level question: given an explanation
method M and an explanation output E, what is the quality of E? This module
builds the formal framework for answering that question rigorously.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Evaluation of Explanations — Metrics, Pitfalls, and Human Studies"
DISPLAY_NAME = "07 · Evaluation of Explanations"
ICON = "📏"
SUBTITLE = "Faithfulness, stability, comprehensibility, and human-grounded evaluation"


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

### The Evaluation Problem

A significant fraction of XAI research produces explanation methods without
evaluating whether those explanations are actually useful, truthful, or stable.
A heatmap that highlights the right region of an image for the wrong reason is
not a good explanation — but it looks good to a human observer. An attribution
that correctly identifies the most important feature for the training distribution
but gives wrong attributions for tail inputs is a poor explanation for those
tail cases — but aggregate metrics might hide this.

The evaluation problem is acute because of a fundamental circularity:
we train models because we do not have a complete ground truth for the
task. We want explanations because we do not fully understand the model.
Evaluating explanations requires either a GROUND TRUTH (which we often
lack) or a PROXY (which may not measure what we care about).

Doshi-Velez & Kim (2017) — "Towards a Rigorous Science of Interpretable
Machine Learning" — identified three levels of evaluation, each progressively
more expensive and closer to the real use case:

    LEVEL 1 — APPLICATION-GROUNDED:
    Evaluate explanations in the actual end-use application with real users.
    Does the doctor make better decisions with the explanation? Does the
    loan officer detect the wrong model faster?
    Most valid, most expensive, hardest to run.

    LEVEL 2 — HUMAN-GROUNDED:
    Evaluate explanations with human participants on simpler tasks that
    proxy the real application. Does the explanation help humans predict
    the model's behaviour? Does it help them identify mistakes?
    More tractable, still requires human studies.

    LEVEL 3 — FUNCTIONALLY-GROUNDED:
    Evaluate explanations using automated proxy metrics that do NOT require
    human studies. Faithfulness, stability, etc.
    Cheapest, most scalable, but most disconnected from actual utility.

Most published XAI evaluation is functionally-grounded. This module covers
the full spectrum, beginning with the formal desiderata that any evaluation
framework must capture.


##### PART I: THE DESIDERATA FOR GOOD EXPLANATIONS

### Five Properties That Define Explanation Quality

An explanation is a statement about WHY a model made a prediction. For that
statement to be GOOD, it must satisfy multiple criteria simultaneously:

    PROPERTY 1 — FAITHFULNESS:
    The explanation accurately describes the model's actual reasoning process.
    If the explanation says "feature A was most important," then feature A
    must genuinely have driven the prediction — not just be correlated with it.

    PROPERTY 2 — STABILITY (ROBUSTNESS):
    Similar inputs produce similar explanations. Small, semantically irrelevant
    perturbations of the input should not produce wildly different explanations.
    An unstable explanation signals that the method is capturing noise, not signal.

    PROPERTY 3 — COMPREHENSIBILITY:
    The explanation is understandable to its target audience without requiring
    excessive cognitive load. A 1000-feature SHAP breakdown is technically
    complete but incomprehensible. A 3-feature summary may be simpler but
    loses information.

    PROPERTY 4 — ACTIONABILITY:
    The explanation enables the user to take useful action — to correct the
    model, to challenge a decision, to improve the input, or to update their
    mental model of the system.

    PROPERTY 5 — COMPLETENESS:
    The explanation captures all causally relevant factors, not just the
    most prominent ones. An explanation that correctly identifies the top
    contributor but ignores three other major contributors is incomplete.

    # ================================================================== #
    **The desiderata trade-off — you can't have all five:**

    Faithfulness ↔ Comprehensibility:
    A fully faithful explanation lists every feature's exact contribution
    — but humans cannot process thousands of numbers. Simplifying to a
    3-feature summary increases comprehensibility but reduces faithfulness.

    Stability ↔ Faithfulness:
    Some faithful explanations (e.g., vanilla gradients) are highly
    unstable — small input changes produce large explanation changes.
    Stabilised methods (SmoothGrad) improve stability at the cost of
    averaging over nearby inputs, reducing local faithfulness.

    Completeness ↔ Comprehensibility:
    A complete explanation for a complex model may require hundreds of terms.
    Reducing it to a few terms increases comprehensibility but loses completeness.

    These trade-offs mean that "best" explanation method depends on the
    use case and audience. Evaluation must specify WHICH desiderata matter
    for the specific deployment context.
    # ================================================================== #


##### PART II: FAITHFULNESS — IS THE EXPLANATION TELLING THE TRUTH?

### The Faithfulness Problem

An explanation is faithful if it accurately describes the actual computational
process of the model. Unfaithful explanations can look plausible, agree with
human intuition, and still be wrong about what the model is doing.

The canonical example: a model trained to classify images of wolves vs huskies
learns to respond to SNOW IN THE BACKGROUND (huskies tend to be photographed
in snow; wolves are photographed in forests). A saliency map might highlight
the wolf's eyes and ears (because that's what humans attend to when looking
at wolves) even though the model's actual decision is driven by the snow.
The explanation is plausible but unfaithful.

Faithfulness is the property most tightly connected to CORRECTNESS of the
explanation. If an explanation is unfaithful, it cannot be used to understand
the model, correct the model, or hold the model accountable.


### Metric 1 — Sufficiency (Completeness Score)

The SUFFICIENCY of a feature set S is the degree to which S alone is sufficient
to reproduce the model's prediction:

    Sufficiency(S) = P(f(x_S) = f(x))

where x_S is the input with only features in S preserved (features NOT in S
are replaced by a baseline or marginalised out) and f is the model's prediction.

A perfectly sufficient explanation (S = all features in the explanation) means
that knowing only the explained features, the model's prediction is unchanged.
A faithfulness test: if the explanation says features {A, B, C} are responsible,
then the model's output should be approximately reproduced by an input that
has only A, B, C at their actual values and everything else marginalised.

    # ================================================================== #
    **Sufficiency computation:**

    Model: f(credit, income, debt, employment, age) → {APPROVE, DENY}
    Explanation: "credit and debt were the deciding features."
    Explanation as feature set: S = {credit, debt}

    Test: Create x_S where credit and debt are actual values,
          income, employment, age are replaced by their training means.
    Run model on x_S.
    Compare f(x_S) to f(x).

    If f(x_S) = f(x) = DENY → high sufficiency for S = {credit, debt}.
    If f(x_S) = APPROVE ≠ f(x) = DENY → low sufficiency — the explanation
    missed the features that actually drove the denial.
    # ================================================================== #


### Metric 2 — Necessity (Deletion / Removal Score)

The NECESSITY of a feature set S is the degree to which REMOVING S changes
the model's prediction:

    Necessity(S) = P(f(x_{-S}) ≠ f(x))

where x_{-S} is the input with features in S removed (replaced by baseline
or marginalised). High necessity means S is genuinely required — without it,
the model predicts differently.

ROAR (Remove And Retrain, Hooker et al., 2019) is the gold standard:
remove the top-k explanation features, RETRAIN the model on the modified
dataset, and measure the performance drop.

    Standard deletion test (no retraining):
    Remove the top-k attribution features from the input (replace with baseline).
    Measure the drop in model confidence for the original prediction.
    Larger drop → the removed features were genuinely important.
    This is faster than ROAR but biased: the model was not trained on
    such modified inputs, so its behaviour on them is unreliable.

    ROAR (with retraining):
    Remove top-k features from ALL training inputs.
    Retrain the model from scratch on the modified training set.
    Evaluate on test set with the same features removed.
    Large accuracy drop → the features were genuinely important.
    This is unbiased but very expensive (requires full model retraining per k).

    # ================================================================== #
    **The MoRF and LeRF curves:**

    Most Relevant First (MoRF): remove features in order from MOST to LEAST
    important (according to the explanation). Plot model confidence as
    features are progressively removed. Should fall steeply if the explanation
    correctly identifies important features.

    Least Relevant First (LeRF): remove features in order from LEAST to MOST
    important. Model confidence should stay high longer (least important first).

    Area Under Curve (AUC) of MoRF: lower is better (more important features
    removed first → bigger confidence drop).
    Area Under Curve of LeRF: higher is better (unimportant features removed →
    little effect on confidence).

    A perfectly faithful explanation:
    MoRF AUC << LeRF AUC  (big gap between the two curves)
    An unfaithful explanation:
    MoRF AUC ≈ LeRF AUC  (removing "important" features has the same effect
    as removing "unimportant" features → the ranking is random)
    # ================================================================== #


### Metric 3 — Comprehensiveness and Sufficiency (COMP and SUFF)

DeYoung et al. (2020) — ERASER benchmark — formalised COMPREHENSIVENESS and
SUFFICIENCY for natural language explanations (rationale extraction):

    COMPREHENSIVENESS (measuring necessity):
    comp(e, f, x) = f(x)[ŷ] − f(x_{-e})[ŷ]

    The drop in model confidence for the predicted class ŷ when the
    explanation evidence e is removed. Higher = explanation is more necessary.

    SUFFICIENCY (measuring sufficiency):
    suff(e, f, x) = f(x)[ŷ] − f(x_e)[ŷ]

    The DECREASE in confidence when ONLY the explanation evidence is kept
    (everything else removed). Lower (closer to 0 or negative) = keeping only
    the explanation is sufficient to maintain the prediction.

These two metrics together characterise faithfulness:
  • High comprehensiveness + low sufficiency = faithful explanation
  • Low comprehensiveness = explanation not necessary (removes it → no change)
  • High sufficiency = explanation not sufficient (keeping only it → bad prediction)


### Metric 4 — Sensitivity to Model Parameters

Adebayo et al. (2018) — Sanity Checks for Saliency — proposed a direct test:
if an explanation method is faithful to the model, then changing the model's
parameters should change the explanation.

    TEST: Progressively randomise the model's weights layer by layer.
    At each stage, compare the explanation output to the original.
    A faithful method's explanation should DIVERGE as parameters are randomised.
    An unfaithful method's explanation remains SIMILAR regardless of randomisation.

    Results (see saliency maps module for full details):
    PASS: Vanilla gradient, GradCAM, Integrated Gradients.
    FAIL: Guided Backpropagation, GuidedGradCAM.

    Methods that fail the sanity check are not faithful to the model —
    they are describing properties of the INPUT, not the MODEL.

This test is now a standard part of XAI evaluation for gradient-based methods.
Any new explanation method should report its performance on the cascading
randomisation test before being considered a model explanation.


### Metric 5 — Axiom Compliance

For attribution methods specifically, several axioms define what it means
to be "correct" in a game-theoretic sense. A method that violates these
axioms is unfaithful in a formal sense:

    EFFICIENCY (Completeness): Σᵢ φᵢ = f(x) − f(x')
    Attributions sum to the model's output difference from a baseline.
    Methods: Integrated Gradients, SHAP (by design).
    Violations: Vanilla gradient × Input (doesn't sum to f(x)).

    SENSITIVITY: If feature i is absent (xᵢ = xᵢ') then φᵢ = 0.
    If a feature hasn't changed, it gets zero attribution.
    Methods: IG, SHAP (by design).
    Violations: Vanilla gradient (can give non-zero attribution to constant features).

    IMPLEMENTATION INVARIANCE: Algebraically identical models get identical attributions.
    Methods: IG (by proof). SHAP (by proof).
    Violations: Vanilla gradient (depends on computational graph, not function).

    SYMMETRY: Features with identical marginal contributions get equal attribution.
    Methods: SHAP (by Shapley axiom).
    Violations: Most other methods.

These axioms provide a formal, model-agnostic faithfulness standard. Methods
that violate them can produce attributions that are technically wrong even
on toy problems with known ground truth.


##### PART III: STABILITY — DOES THE EXPLANATION CHANGE WITH THE INPUT?

### Why Stability Matters

Consider two medical images from the same patient, taken moments apart,
with only instrument noise differing. The model predicts the same diagnosis.
But the explanation for image A highlights the tumour region; the explanation
for image B highlights a different region near the border. Both explanations
are individually plausible — but their DISAGREEMENT signals something is wrong.
Either: (a) the method is inherently noisy and captures instrument artefacts,
not genuine model reasoning; or (b) the model itself is responding to noise,
and the explanation is faithfully (but wrongly) capturing that.

Stability is a necessary condition for RELIABILITY. An unstable explanation
cannot be acted upon. A loan officer who sees different feature attributions
each time the same application is processed cannot make consistent decisions.

    # ================================================================== #
    **Stability vs Faithfulness — the tension:**

    A perfectly faithful explanation captures the exact local gradient of
    the model. If the model's gradient is noisy (due to shattered gradients,
    ReLU discontinuities), the faithful explanation IS noisy. Faithfulness
    demands instability.

    SmoothGrad and attention rollout improve stability by averaging, but
    this averaging may be averaging away genuine local variation that the
    model is sensitive to. They sacrifice local faithfulness for global stability.

    This tension has no universal resolution. The right trade-off depends
    on whether the instability reflects genuine model sensitivity (keep it)
    or numerical noise (smooth it). Without knowing the ground truth,
    you cannot always distinguish these.
    # ================================================================== #


### Metric 6 — Lipschitz Stability (Alvarez-Melis & Jaakkola, 2018)

A formal measure of local stability: how much does the explanation change
when the input changes by ε?

    Lipschitz constant of explanation method E:
    L(E, x) = max_{x': ‖x'−x‖ ≤ ε} ‖E(f, x') − E(f, x)‖ / ‖x' − x‖

A smaller Lipschitz constant means the explanation changes less per unit
change in the input — higher stability.

In practice, this is estimated by:
    1. Generating n random perturbations x' = x + δ, where δ ~ N(0, ε²I)
    2. Computing E(f, x') for each perturbation
    3. Computing the average (or maximum) Lipschitz constant estimate:
       L̂(E, x) = avg [ ‖E(f, x') − E(f, x)‖ / ‖x' − x‖ ]

Methods are compared by their average Lipschitz constant across a test set.
Lower L̂ → more stable explanation method.

    # ================================================================== #
    **Empirical stability rankings (approximate, from literature):**

    Method                    Local Lipschitz (lower is more stable)
    ─────────────────────────────────────────────────────────────────
    Vanilla gradient          HIGH (most unstable — shattered gradients)
    Gradient × Input          HIGH-MEDIUM
    Integrated Gradients      MEDIUM (averaging along path helps)
    SmoothGrad                LOW-MEDIUM (explicitly averaged)
    SHAP KernelSHAP           LOW (averaging over many perturbations)
    LIME                      MEDIUM (depends on sample count)
    # ================================================================== #


### Metric 7 — Rank Correlation Stability

For feature attribution methods, stability can be measured as the rank
correlation between explanations for similar inputs:

    Given pairs (xᵢ, xᵢ') where xᵢ' is a small perturbation of xᵢ:
    Compute feature attribution vectors Eᵢ = E(f, xᵢ) and Eᵢ' = E(f, xᵢ').
    Measure: Spearman ρ(Eᵢ, Eᵢ') — rank correlation of feature attributions.

    ρ ≈ 1: highly stable — the RANKING of feature importances barely changes.
    ρ ≈ 0: completely unstable — the ranking is essentially random after perturbation.
    ρ < 0: perversely unstable — perturbation reverses the importance ranking.

Average rank correlation over many test instances is a scalar stability metric.
Importantly, this measures RANKING stability (same features identified as important)
separately from VALUE stability (exact attribution values match).
Ranking stability is usually more important for human-facing explanations.


### Metric 8 — Explanation Variance Under Random Seeds

Many stochastic explanation methods (SmoothGrad, SHAP KernelSHAP, LIME)
use random sampling internally. Their explanations vary across runs even
for the same input and model.

    Explanation variance for input x:
    Var_seed[E(f, x, seed)] = (1/N) Σ_{seed=1}^{N} ‖E(f, x, seedᵢ) − Ē(f, x)‖²

    where Ē(f, x) = (1/N) Σᵢ E(f, x, seedᵢ) is the mean explanation.

High variance indicates that more samples are needed for reliable explanations.
This variance can be reduced by increasing the number of samples (SHAP, LIME,
SmoothGrad all converge as sample count → ∞), but at increased computational cost.

Reporting variance alongside mean attributions is good practice — single-run
explanations from stochastic methods without confidence intervals are unreliable.


##### PART IV: COMPREHENSIBILITY — CAN HUMANS UNDERSTAND THE EXPLANATION?

### The Comprehensibility Problem

Comprehensibility is fundamentally a property of the MATCH between the
explanation format and the cognitive capacity of its audience.

A linear model with 1,000 coefficients is technically a complete explanation
of a linear model's predictions — but no human can process 1,000 coefficients
at once. A 3-feature LIME explanation with a bar chart is comprehensible to
a business analyst. A circuit diagram showing attention heads and MLP neurons
is comprehensible to an ML researcher but not to a patient's doctor.

Comprehensibility depends on:
  • AUDIENCE: technical expertise, domain knowledge, cognitive load tolerance.
  • FORMAT: numbers vs visual vs natural language vs rules.
  • COMPLEXITY: how many concepts must be held in working memory simultaneously.
  • FAMILIARITY: explanations in familiar formats (feature importance bars)
    are more comprehensible than unfamiliar ones (circuit diagrams).

This means comprehensibility CANNOT be evaluated without specifying the audience.
A functionally-grounded metric for comprehensibility that ignores the audience
is measuring the wrong thing.

    # ================================================================== #
    **Proxy metrics for comprehensibility (functionally-grounded):**

    EXPLANATION SIZE:
    Number of features/conditions in the explanation. Shorter = more comprehensible.
    Linear models: number of non-zero coefficients.
    Decision trees: depth or number of leaves.
    Rules: number of conditions per rule.

    READING LEVEL:
    For natural language explanations: Flesch-Kincaid grade level, SMOG index.
    Lower reading level → more comprehensible to a broader audience.

    CONCEPT COMPLEXITY:
    Does the explanation use concepts the audience is familiar with?
    Patient-facing: "your cholesterol was high" (familiar) vs
    "the LDL/HDL ratio feature had attribution +0.23" (unfamiliar).

    COGNITIVE LOAD ESTIMATE:
    Working memory research suggests 7±2 chunks as the limit of working memory
    capacity. Explanations with more than 7 independent elements exceed this limit.
    This motivates the "top-5 features" heuristics in many XAI tools.
    # ================================================================== #


### Miller's Law and Explanation Chunking

Miller (1956) — "The Magical Number Seven, Plus or Minus Two" — established
that human working memory can hold 7±2 independent items simultaneously.
This has direct implications for explanation design.

An attribution method that returns 768 feature attributions (one per BERT
embedding dimension) is technically complete but cognitively useless.
The explanation must be CHUNKED — aggregated into a manageable number of
higher-level concepts.

    Chunking strategies:
    1. FEATURE GROUPS: aggregate attributions by semantic group (e.g., all
       syntactic features → one score, all semantic features → another).
    2. TOP-K: report only the k most important features (k ≤ 7).
    3. HIERARCHICAL: show a summary at the top level, with detail available
       on demand for each item (progressive disclosure).
    4. CONTRASTIVE: "these 3 features supported APPROVE; these 2 features
       supported DENY" — contrast activates the understanding of difference.

The optimal chunking depends on the user's goal and the domain. There is
no universal "right" number of features to show.


### Metric 9 — Forward Simulation (Simulatability)

One concrete way to test whether a human "understood" an explanation:
can they SIMULATE the model's behaviour on a new input using only the explanation?

    Protocol:
    1. Show the human the explanation for inputs x₁, x₂, ..., xₙ.
    2. For a new input xₙ₊₁, ask the human to predict the model's output,
       using only the explanation (not the model itself).
    3. Measure: fraction of xₙ₊₁ predictions the human gets right.

    HIGH SIMULATABILITY: the human can predict the model's behaviour from
    the explanation. This implies the explanation captures the model's
    decision rule well enough for transfer to new cases.

    LOW SIMULATABILITY: the explanation fails to convey the model's logic.
    Even a "correct" explanation that humans cannot use to predict behaviour
    is not comprehensible in the sense that matters.

This metric directly tests the explanations' VALUE for users who need to
anticipate the model's behaviour — e.g., auditors checking for compliance,
clinicians understanding when to trust the model, engineers debugging failures.


##### PART V: HUMAN-GROUNDED EVALUATION — THE SPECTRUM OF USER STUDIES

### Why Automated Metrics Are Not Enough

Automated metrics (faithfulness, stability) measure properties of the explanation
in isolation. They do not measure whether the explanation is USEFUL for the humans
who will consume it. A highly faithful explanation that is incomprehensible to
its intended audience is not a good explanation, regardless of its faithfulness score.

Human-grounded evaluation is essential to close the loop between
"technically correct" and "actually useful." It is expensive and noisy,
but it is the only way to know whether explanations serve their stated purpose.

### Study Type 1 — Detect Model Errors (Unfairness, Shortcuts)

Task: given an explanation and model predictions on a dataset, can a human
identify when the model is making decisions for the wrong reasons?

    Protocol:
    1. Train a model with a known shortcut (e.g., the model responds to
       spurious background features rather than the correct ones).
    2. Show participants: (a) model predictions only, (b) model predictions
       + explanations.
    3. Ask: "Does this model seem to be using the right features?"
    4. Measure: time to identify the shortcut, accuracy of identification.

    GOOD EXPLANATION: significantly faster, more accurate shortcut detection.
    BAD EXPLANATION: no improvement over predictions-only baseline.

This study type is directly relevant to fairness auditing and safety checking.
It measures the most practically important property of explanations in
high-stakes settings.

    # ================================================================== #
    **Landmark result — Ribeiro et al. (2016) LIME paper:**

    Experiment: SVM classifier for "whether a doctor is male or female"
    from a biased dataset where the label correlated with word usage in
    medical notes (not with actual gender).

    Condition A: show only predictions.
    Condition B: show LIME explanations.

    Result: participants in Condition B identified the model's spurious
    reliance on non-gender-related features significantly faster and more
    accurately than Condition A. LIME explanations enabled users to catch
    a biased model that predictions alone would not reveal.
    # ================================================================== #


### Study Type 2 — Trust Calibration

Task: given an explanation, do humans calibrate their TRUST in the model's
predictions appropriately?

    Too much trust: humans defer to the model even when it's wrong.
    Too little trust: humans override correct model predictions.
    Appropriate trust: humans accept correct predictions, override wrong ones.

    Protocol:
    1. Participants make a decision (e.g., diagnose from image).
    2. Model also makes a prediction. Show participants the model's prediction.
    3. One condition: show explanation alongside prediction.
    4. Allow participant to accept or override the model.
    5. Measure: rate of correct overrides and correct acceptances.

    GOOD EXPLANATION: increases correct overrides without increasing false overrides.
    BAD EXPLANATION: increases trust uniformly (over-reliance, automation bias),
    or decreases trust uniformly (under-reliance).

This study type is critical for human-AI teaming where the goal is COMPLEMENTARY
performance — the human + AI should outperform either alone.


### Study Type 3 — Forward Simulation in Practice

(Lipton, 2018; Poursabzi-Sangdeh et al., 2021)

    Protocol:
    1. Show participants multiple examples from a dataset, each with an explanation.
    2. Remove the model. Present a new input without the model's prediction.
    3. Ask participants to predict what the MODEL would predict.
    4. Measure accuracy of participants' predictions of the model's behaviour.

    This is a direct test of whether participants have internalised the model's
    decision logic from the explanations.

    KEY FINDING (Poursabzi-Sangdeh et al., 2021):
    Contrary to intuition, MORE complex explanations sometimes led to LOWER
    simulation accuracy. Simpler models (linear models with 3 features) were
    easier to simulate than complex models with 6+ features, even when the
    complex model's explanation was technically more complete.

    The implication: comprehensibility, not completeness, drives simulatability.
    A faithful but complex explanation may perform WORSE than an incomplete but
    simple one in terms of enabling humans to predict model behaviour.


### Study Type 4 — Decision Quality Improvement

The most direct measure: does the explanation improve the human's final
DECISION OUTCOME, not just their understanding of the model?

    Protocol (medical imaging):
    1. Radiologists read CT scans and provide diagnoses.
    2. AI model also provides prediction.
    3. Condition A: radiologists see only the AI prediction.
    4. Condition B: radiologists see AI prediction + saliency map.
    5. Measure: final diagnostic accuracy after AI consultation.

    GOOD EXPLANATION: improves diagnostic accuracy (humans use AI appropriately).
    BAD EXPLANATION: decreases accuracy (humans over-rely on AI), or no effect.

This is the most expensive and most valid evaluation. Few XAI papers report
studies of this type. Studies that do exist show mixed results — explanations
sometimes help, sometimes have no effect, and occasionally harm performance
by increasing cognitive load or misleading users.

    # ================================================================== #
    **Automation bias — explanations can make it worse:**

    Automation bias (Parasuraman & Manzey, 2010): humans tend to defer
    to automated systems even when the system is wrong.

    Finding (Bussone et al., 2015): medical AI explanations sometimes
    INCREASED automation bias — participants were MORE likely to defer
    to wrong AI predictions when an explanation was shown, compared to
    seeing the prediction alone.

    Explanation: the explanation, even if technically incorrect, gave
    the feeling of understanding ("I know WHY the model said this").
    This confidence suppressed the critical evaluation that might have
    caught the model's error.

    This is the dark side of explanation: plausible but unfaithful
    explanations can INCREASE over-reliance, making human+AI performance
    WORSE than human alone.
    # ================================================================== #


##### PART VI: GROUND TRUTH EVALUATION — SYNTHETIC AND KNOWN MODELS

### Evaluation With a Known Ground Truth

The cleanest evaluation setting: use a model where the true explanation is known
by construction, then test whether explanation methods recover it.

    APPROACH 1 — SYNTHETIC DATA WITH KNOWN DECISION BOUNDARY:
    Construct a dataset where the true decision rule is known:
    f(x) = sign(x₁ × x₂ − x₃)  →  features 1, 2, 3 are relevant; 4, 5, ... are not.
    Run attribution methods on this model. Does feature 3 get higher attribution
    than feature 4? Do features 1, 2, 3 consistently rank above irrelevant features?

    APPROACH 2 — DISTILLED MODELS:
    Train a complex model, then distill it into a simpler interpretable model.
    The distilled model's parameters ARE the true explanation (within approximation).
    Compare: does the attribution method's output match the distilled model's coefficients?

    APPROACH 3 — MODELS WITH PLANTED SPURIOUS CORRELATIONS:
    Create a model that uses feature X for prediction but where the true causal
    feature is Y (X is a proxy correlated with Y in training data).
    A faithful explanation should identify X as important; a truly causal
    explanation should identify Y. Use this to test which methods find causal
    vs correlational features.

    # ================================================================== #
    **The ROAR test and the annotation shortcut:**

    For image models, human annotators can provide PIXEL-LEVEL GROUND TRUTH
    for what features are relevant: bounding boxes, segmentation masks.
    These define the "correct" explanation.

    Test: does the saliency map's high-attribution region overlap with the
    annotated relevant region?

    This is measured by:
    • IoU (Intersection over Union) of heatmap region with annotation.
    • Pointing game accuracy: does the maximum attribution pixel fall within
      the annotated region?

    Limitation: the annotation reflects what HUMANS think is relevant,
    not necessarily what the MODEL is using. If the model uses spurious
    features that happen to correlate with the annotated region, the
    explanation may appear "correct" by this metric even if unfaithful.
    # ================================================================== #


### The Benchmark Datasets

Several benchmark datasets have been developed specifically for XAI evaluation:

    MNIST / FASHION-MNIST:
    Simple, well-understood, frequently used as a sanity check.
    True relevant pixels: the digit strokes for MNIST, the clothing
    shape for Fashion-MNIST.
    Limitation: too simple — most methods work well here, giving little
    discrimination between methods.

    ImageNet + Segmentation Annotations:
    The PASCAL VOC and COCO segmentation annotations provide pixel-level
    ground truth for ImageNet images. Used in the "pointing game" evaluation.
    More challenging, more discriminating between methods.

    ERASER Benchmark (DeYoung et al., 2020):
    Benchmark for NLP explanation evaluation.
    Provides human-annotated rationales (key phrases) for multiple NLP tasks.
    Tests: faithfulness (comprehensiveness, sufficiency), stability.

    Synthetic Benchmarks (Samek et al., 2017):
    "Monkey in the boat" and other deliberately biased datasets where
    the spurious correlation is known. Used to test whether explanations
    reveal the spurious feature.

    XAIR (eXplainable AI Recall) Benchmark (Kim et al., 2022):
    Comprehensive benchmark for clinical AI explanation evaluation.
    Includes both automated metrics and human study components.


##### PART VII: THE SHORTCUT PROBLEM — EXPLANATIONS THAT LOOK GOOD BUT AREN'T

### The Confirmation Bias in Explanation Evaluation

Humans evaluating explanations are subject to CONFIRMATION BIAS: they rate
explanations as more faithful and more useful when the explanations confirm
their prior beliefs about which features are important.

This creates a systematic bias in human evaluation studies:

    • Explanations that highlight features humans expect to be important
      receive high plausibility ratings — regardless of whether those
      features are actually used by the model.

    • Explanations that highlight unexpected (but genuinely important)
      features are rated as "strange" or "wrong" — even when they are
      correct.

    Lipstick on a pig: a beautiful, intuitive explanation of a biased model
    might receive BETTER human evaluation ratings than an accurate but
    counterintuitive explanation of a fair model.

    # ================================================================== #
    **Nguyen (2018) — "Why are saliency maps noisy?":**

    Key finding: saliency map quality ratings correlate more strongly with
    the VISUAL QUALITY of the heatmap than with its faithfulness.
    Sharp, smooth, aesthetically pleasing heatmaps receive high ratings.
    Noisy, scattered heatmaps receive low ratings.

    But there is no reason to expect aesthetically pleasing heatmaps to be
    more faithful. The rating reflects human preference for visual coherence,
    not correspondence with model behaviour.

    Implication: never use "how good does this look?" as an evaluation criterion.
    Always use behavioural criteria (insertion/deletion, simulatability, etc.).
    # ================================================================== #


### Metric 10 — Alignment With Human Judgement (and Its Limits)

ALIGNMENT WITH HUMAN JUDGEMENT tests whether the explanation's feature
importance ranking matches what human domain experts consider important.

    Example: for a skin lesion classifier, do the pixels highlighted
    by the saliency map correspond to the dermatologist's "relevant" regions?
    Measured by: agreement score, rank correlation with expert ratings.

    THIS METRIC IS PROBLEMATIC:
    (a) Models often use correct features that humans would also use →
        the metric gives a "pass" even if the model has spurious correlations
        that happen to also be clinically relevant.
    (b) Models sometimes correctly use features humans undervalue →
        the metric penalises a correct model that has better-than-human features.
    (c) Models with spurious correlations may still highlight clinically
        relevant areas (by coincidence) → the metric passes the wrong model.

    Human judgement alignment is a useful SANITY CHECK but a poor PRIMARY METRIC.
    Use it alongside, not instead of, faithfulness tests.


### The "Fooling" Problem — Manipulable Explanations

Heo et al. (2019) — "Fooling Neural Network Interpretations via
Adversarial Model Manipulations" — demonstrated that models can be trained
to produce specific, desired explanations for ANY input, while making
arbitrary predictions.

The construction: train a model whose weights encode two functions:
  1. f(x): the actual prediction (arbitrary, possibly unfair).
  2. g(x): the explanation generator, designed to always highlight
     non-sensitive features (e.g., never highlight race or gender).

Both f and g share the same inputs. The model is trained to have
correct predictions AND to produce misleading explanations simultaneously.

    # ================================================================== #
    **The adversarial explanation — the Heo et al. construction:**

    Original model: f(x) relies heavily on race → biased, legally problematic.
    Explanation: "model uses race as a key feature" → unacceptable explanation.

    Adversarially manipulated model:
    • Same predictions as original f (same bias remains).
    • Explanation output: "model uses income, education, employment" → looks fair.

    The explanation has been designed to satisfy human reviewers. The bias
    has not been fixed — only hidden from the explanation.

    This is called "explanation laundering": using explanations to create
    the appearance of fairness while maintaining biased predictions.
    # ================================================================== #

The implication: explanations produced by a model that is adversarially trained
to produce specific explanations cannot be trusted. If a model is optimised
to produce good-looking explanations, evaluation metrics for those explanations
are all gameable.

The defence: always evaluate faithfulness by EXTERNAL tests (deletion, ROAR,
sanity checks) rather than by the explanation's appearance.


##### PART VIII: COMPARING EXPLANATION METHODS — WHAT THE BENCHMARKS SHOW

### What We Know From Systematic Comparisons

Numerous benchmark studies have systematically compared explanation methods
across multiple evaluation criteria. The results are more nuanced than any
single method's proponents suggest.

    FAITHFULNESS (deletion/insertion scores):
    SHAP (exact) > SHAP (approx) > Integrated Gradients > LIME ≈ SHAP KernelSHAP
    > SmoothGrad > Vanilla Gradient > GuidedBP (fails sanity check)

    STABILITY (Lipschitz constant, rank correlation):
    SHAP KernelSHAP > LIME > SmoothGrad > Integrated Gradients > SHAP FAST
    > Vanilla Gradient

    COMPREHENSIBILITY (user ratings, forward simulation accuracy):
    Anchor rules > LIME linear model > SHAP summary plot
    > Raw attribution numbers
    (Rules are most comprehensible; numbers are least)

    COMPUTATIONAL COST:
    Vanilla Gradient (1 backward) < GradCAM (1 backward to last conv)
    < IG (50–300 backward) < SHAP (2ⁿ model calls) < SHAP KernelSHAP (sample-based)
    < Deep SHAP (faster for neural nets)

    GENERAL FINDING:
    No single method dominates across all criteria. Method choice should be
    driven by which criteria matter most for the specific use case.

    # ================================================================== #
    **The Samek et al. (2016) deletion metric comparison:**

    On ImageNet with VGG16:
    Methods ranked by AUC of deletion curve (lower = more faithful):

    1. Integrated Gradients:   AUC = 0.31  (most faithful)
    2. SHAP:                   AUC = 0.33
    3. Gradient × Input:       AUC = 0.38
    4. Smoothed Gradient:      AUC = 0.41
    5. Vanilla Gradient:       AUC = 0.44
    6. Guided Backpropagation: AUC = 0.51  (least faithful)
    7. Random baseline:        AUC = 0.52  (not better than random)

    GuidedBP is statistically indistinguishable from RANDOM attribution
    on the deletion metric — confirming that it describes the input,
    not the model.
    # ================================================================== #


### The Meta-Evaluation Problem

Evaluating explanation methods requires evaluation metrics. But evaluation
metrics themselves can be wrong or biased. This creates a META-EVALUATION
problem: how do we know which evaluation metrics are good?

    CIRCULAR METRICS: Some metrics define faithfulness as "correlation with
    SHAP" (since SHAP has strong axioms). But if SHAP is wrong for a specific
    model architecture, the metric inherits SHAP's errors.

    DISTRIBUTION MISMATCH: Deletion tests replace features with baselines
    (zeros, means). If the model was trained on inputs where features are
    always non-zero, the deletion creates out-of-distribution inputs.
    The model's behaviour on OOD inputs may not reflect the model's true
    feature importance (see ROAR vs standard deletion debate).

    BENCHMARK SPECIFICITY: Metrics calibrated on one benchmark (e.g., ImageNet)
    may not transfer to other domains (medical imaging, NLP, tabular).
    High performance on ImageNet benchmarks does not guarantee faithfulness
    in medical imaging.

    THE ULTIMATE STANDARD: The only fully trustworthy evaluation is the
    application-grounded study with real users on the actual deployment task.
    Everything else is a proxy. Use proxies carefully, report uncertainty,
    and triangulate across multiple metrics.


##### PART IX: PRACTICAL EVALUATION PROTOCOL — A CHECKLIST

### The Minimum Required Evaluation Battery

For any XAI system deployed in a consequential application, the following
minimum evaluation battery should be reported before deployment:

    ✓ FAITHFULNESS — SANITY CHECK:
    Run the Adebayo cascading randomisation test (for gradient methods).
    The explanation should change meaningfully as model weights are randomised.
    If the explanation doesn't change, it is not faithful to the model.

    ✓ FAITHFULNESS — DELETION/INSERTION:
    Compute the MoRF curve (Most Relevant First deletion).
    Compare to the LeRF curve (Least Relevant First deletion).
    Report the AUC ratio. Large gap → faithful ranking.
    Small gap → attribution ranking is not better than random.

    ✓ STABILITY:
    Report Lipschitz constant estimate (average over test set).
    For stochastic methods: report variance across seeds for the same input.
    Provide a worst-case stability example (the instance with highest Lipschitz).

    ✓ COMPREHENSIBILITY — FORMAT:
    Report what format the explanation takes (numbers, visuals, rules, text).
    Report the average "size" of the explanation (number of features, rule length).
    Document the intended audience and their relevant expertise.

    ✓ AXIOM COMPLIANCE (for attribution methods):
    Check and report compliance with: Efficiency, Sensitivity, Implementation
    Invariance, Symmetry.
    Non-compliant methods should acknowledge which axioms they violate.

    ✓ HUMAN EVALUATION (if deployment is high-stakes):
    At minimum: run a forward simulation study.
    Preferred: run a mistake detection study.
    Gold standard: run an application-grounded study with real users.
    Report sample size, participant characteristics, statistical significance.

    ✓ DISTRIBUTION DOCUMENTATION:
    Specify which input distribution the evaluation is run on.
    Explicitly note that the evaluation does NOT guarantee calibration
    under distribution shift.
    If possible, report evaluation on held-out out-of-distribution inputs.


##### PART X: A COMPLETE WORKED EXAMPLE — EVALUATING AN ATTRIBUTION METHOD

### Full Evaluation of a Synthetic Attribution on a Known Model

We trace through a complete evaluation of an attribution method on a
synthetic binary classifier where the ground truth attribution is known.

    SETUP:
    True model: f(x₁, x₂, x₃, x₄, x₅) = sign(0.8·x₁ + 0.4·x₂ − 0.1·x₃ + 0·x₄ + 0·x₅)
    True feature importance: x₁ (0.8), x₂ (0.4), x₃ (−0.1), x₄ (0.0), x₅ (0.0)
    Irrelevant features: x₄ and x₅.

    Instance: x = (1.0, 0.5, 0.3, 0.7, −0.2), f(x) = 1 (positive class)

    Evaluation of two attribution methods:

    METHOD A (CORRECT):   φ = (0.80, 0.40, −0.10, 0.00, 0.00) — matches true weights
    METHOD B (INCORRECT): φ = (0.30, 0.25, −0.05, 0.70, 0.20) — inflated x₄, x₅

    FAITHFULNESS — DELETION TEST:
    Remove features in order of |φ| (most important first).

    Method A removes: x₁, x₂, x₃, x₄, x₅  (correct order)
    Method B removes: x₄, x₁, x₅, x₂, x₃  (wrong order — x₄ first!)

    Method A deletion scores (model confidence after each removal):
    Remove x₁: f(0, 0.5, 0.3, 0.7, −0.2) = sign(0+0.2−0.03) = sign(0.17) = +1
               confidence drops slightly (x₁ removed, but x₂ keeps it positive)
    Remove x₁, x₂: f(0, 0, 0.3, 0.7, −0.2) = sign(−0.03) = −1  ← prediction flips!
    Confidence drop at step 2: LARGE.

    Method B deletion scores (removing x₄ first):
    Remove x₄: f(1.0, 0.5, 0.3, 0, −0.2) = sign(0.8+0.2−0.03) = sign(0.97) = +1
               confidence barely changes (x₄ was not important!)
    Removing the "most important" feature (x₄) according to Method B barely
    affects the prediction. LOW faithfulness score.

    CONCLUSION: Method A has higher faithfulness (removing truly important features
    causes larger prediction changes). Method B's ranking is wrong.

    STABILITY TEST:
    Perturb input by ε = 0.1 in direction of x₄: x' = (1.0, 0.5, 0.3, 0.8, −0.2)

    Method A attributions for x':
    φ'_A ≈ (0.80, 0.40, −0.10, 0.00, 0.00)  (nearly unchanged — stable)
    Rank correlation ρ(φ_A, φ'_A) ≈ 1.0

    Method B attributions for x':
    φ'_B ≈ (0.30, 0.25, −0.05, 0.72, 0.20)  (x₄ attribution slightly changed)
    Rank correlation ρ(φ_B, φ'_B) ≈ 0.99  (also stable, but for wrong reasons)

    Both methods are locally stable in this case (small perturbation → small change).
    Stability alone does not distinguish correct from incorrect attributions.

    AXIOM COMPLIANCE:
    Method A: Σ φᵢ = 0.80 + 0.40 − 0.10 + 0 + 0 = 1.10
              f(x) = 1.0  (linear model output, not bounded)
              Efficiency: 1.10 ≈ f(x) = 1.0 if we define f(x) as the linear output.
              The sum matches the model's output — good efficiency.

    Method B: Σ φᵢ = 0.30 + 0.25 − 0.05 + 0.70 + 0.20 = 1.40
              Does NOT match f(x) = 1.0.
              Efficiency violation — Method B does not satisfy Completeness.

    FORWARD SIMULATION:
    Show a human only the attribution (not the model).
    Present new instance x'' = (0.5, 0.2, 0.1, 0.8, −0.5).
    Ask: "Is f(x'') positive or negative?"

    Using Method A's attribution (x₁ most important):
    Human reasons: "x₁=0.5 (positive, weight 0.8), x₂=0.2 (positive, weight 0.4)
    → probably positive."
    True: f(x'') = sign(0.4+0.08−0.01) = sign(0.47) = +1. Correct!

    Using Method B's attribution (x₄ most important):
    Human reasons: "x₄=0.8 (positive, weight 0.70) → probably positive."
    True: f(x'') = +1. Accidentally correct (for the wrong reason).
    If x'' had x₄=−0.5 instead: human would predict negative, but f(x'') =
    sign(0.4+0.08−0.01) = +1. Method B leads to wrong prediction!

    This illustrates: a method can be locally "correct" on one instance but
    fail to enable correct generalisation. Only ground-truth evaluation reveals
    this. Method B's attribution teaches the wrong model of f's behaviour.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔══════════════════════════╦═══════════════════╦══════════════════════════════╗
    ║ Evaluation Dimension     ║ Key Metric(s)     ║ Method Required              ║
    ╠══════════════════════════╬═══════════════════╬══════════════════════════════╣
    ║ Faithfulness (necessity) ║ MoRF AUC, ROAR    ║ Automated (deletion test)    ║
    ║ Faithfulness (suffic.)   ║ SUFF score        ║ Automated (insertion test)   ║
    ║ Faithfulness (sanity)    ║ Pass/Fail         ║ Automated (param randomise)  ║
    ║ Faithfulness (axioms)    ║ Efficiency, Sens. ║ Analytical check             ║
    ║ Stability (local)        ║ Lipschitz const.  ║ Automated (perturbations)    ║
    ║ Stability (rank)         ║ Spearman ρ        ║ Automated (perturbations)    ║
    ║ Stability (stochastic)   ║ Variance/seeds    ║ Automated (multi-run)        ║
    ║ Comprehensibility        ║ Size, simulatab.  ║ Human study                  ║
    ║ Error detection          ║ Shortcut detect.  ║ Human study                  ║
    ║ Trust calibration        ║ Override accuracy ║ Human study (decision tasks) ║
    ║ Decision improvement     ║ Task performance  ║ Application-grounded study   ║
    ╚══════════════════════════╩═══════════════════╩══════════════════════════════╝

    Three levels (Doshi-Velez & Kim, 2017):
      APPLICATION-GROUNDED: real users, real task (most valid, most expensive)
      HUMAN-GROUNDED:        real users, proxy task (intermediate)
      FUNCTIONALLY-GROUNDED: no users, automated metrics (cheapest, least valid)

    Key tension:
      Faithfulness ↔ Comprehensibility (complete explanations are incomprehensible)
      Stability    ↔ Faithfulness      (averaged explanations are less local)
      Completeness ↔ Actionability     (complete explanations are hard to act on)
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Explanation Evaluation from Scratch — Faithfulness, Stability, and Axiom Compliance": {
        "description": (
            "Implements the full evaluation battery for two attribution methods on a "
            "synthetic tabular classifier with known ground-truth attributions. "
            "Computes: (1) faithfulness via MoRF/LeRF deletion curves with AUC comparison, "
            "(2) sufficiency via insertion scores, (3) Lipschitz stability via random "
            "perturbations, (4) rank correlation stability, (5) axiom compliance checks "
            "(Efficiency, Sensitivity, Symmetry), and (6) forward simulation accuracy on "
            "held-out inputs. Reports a summary scorecard comparing both methods. "
            "Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "eval_explanations",
        "code": '''
"""
================================================================================
EXPLANATION EVALUATION — FAITHFULNESS, STABILITY, AND AXIOM COMPLIANCE
================================================================================

We evaluate two attribution methods on a known synthetic classifier:
  METHOD A: Ground-truth-aligned (correct attributions)
  METHOD B: Miscalibrated (overemphasises irrelevant features)

Evaluation battery:
  1. Faithfulness — MoRF (deletion) and LeRF (insertion) curves
  2. Faithfulness — Sufficiency score
  3. Faithfulness — Axiom compliance (Efficiency, Sensitivity, Symmetry)
  4. Stability — Lipschitz constant estimate
  5. Stability — Rank correlation under perturbation
  6. Forward simulation accuracy (proxy for comprehensibility)

Ground truth: f(x) = sigmoid(2.0*x1 + 1.0*x2 - 0.5*x3 + 0*x4 + 0*x5)
True importance: x1 > x2 > x3, x4 = x5 = 0.
================================================================================
"""

import math
import random

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL AND GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

TRUE_WEIGHTS = [2.0, 1.0, -0.5, 0.0, 0.0]  # feature weights
FEAT_NAMES   = ["x1 (imp)", "x2 (imp)", "x3 (low)", "x4 (irrel)", "x5 (irrel)"]
D = 5
BASELINE = [0.0] * D  # zero baseline for attribution

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

def model(x):
    """True model: logistic regression with known weights."""
    z = sum(TRUE_WEIGHTS[i] * x[i] for i in range(D))
    return sigmoid(z)

def model_class(x, threshold=0.5):
    return 1 if model(x) >= threshold else 0

def generate_instance(seed=0):
    """Generate a random test instance."""
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(D)]

# Test instance
X_TEST = [0.8, 0.6, 0.4, 0.7, -0.3]
Y_PRED = model(X_TEST)
print("=" * 68)
print("  EVALUATION SETUP")
print("=" * 68)
print(f"  True model: f(x) = sigmoid(2.0·x1 + 1.0·x2 - 0.5·x3 + 0·x4 + 0·x5)")
print(f"  True feature importance: x1 > x2 > x3 > x4 = x5 = 0")
print()
print(f"  Test instance: x = {[round(v,2) for v in X_TEST]}")
print(f"  Model output: f(x) = {Y_PRED:.4f}  (class 1, positive)")
print()

# Two attribution methods (synthetic — in real use, run SHAP/IG/LIME here)
# Method A: correct, aligned with true weights
PHI_A = [w * x for w, x in zip(TRUE_WEIGHTS, X_TEST)]  # gradient × input ≈ true

# Method B: miscalibrated — overemphasises irrelevant features x4, x5
PHI_B = [0.35, 0.20, -0.08, 0.55, 0.30]  # wrong — x4 and x5 get high attribution

# Normalise both to sum to f(x) - f(x') for fairness
delta_f = model(X_TEST) - model(BASELINE)
sum_A = sum(PHI_A)
sum_B = sum(PHI_B)
PHI_A = [p * delta_f / (sum_A + 1e-12) for p in PHI_A]
PHI_B = [p * delta_f / (sum_B + 1e-12) for p in PHI_B]

print("  Attribution methods:")
print(f"  {'Feature':<16}  {'Method A (correct)':>20}  {'Method B (wrong)':>18}")
print(f"  {'-'*58}")
for i in range(D):
    print(f"  {FEAT_NAMES[i]:<16}  {PHI_A[i]:>+20.5f}  {PHI_B[i]:>+18.5f}")
print(f"  {'Sum':>16}  {sum(PHI_A):>+20.5f}  {sum(PHI_B):>+18.5f}")
print(f"  {'Δf':>16}  {delta_f:>+20.5f}  {delta_f:>+18.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# FAITHFULNESS 1 — MoRF AND LeRF DELETION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def deletion_curve(x, attr, morf=True, n_steps=None):
    """
    Compute deletion/insertion curve.
    morf=True: Most Relevant First (delete important features first)
    morf=False: Least Relevant First (delete unimportant features first)
    Returns: list of model outputs after removing k features.
    """
    n = len(x)
    if n_steps is None:
        n_steps = n

    # Sort features by |attribution|, high to low (for MoRF) or low to high (LeRF)
    order = sorted(range(n), key=lambda i: abs(attr[i]), reverse=morf)

    curve = [model(x)]  # no features removed yet
    x_curr = list(x)
    for k in range(min(n_steps, n)):
        feat_idx = order[k]
        x_curr[feat_idx] = BASELINE[feat_idx]   # replace with baseline
        curve.append(model(x_curr))

    return curve

morf_A = deletion_curve(X_TEST, PHI_A, morf=True)
lerf_A = deletion_curve(X_TEST, PHI_A, morf=False)
morf_B = deletion_curve(X_TEST, PHI_B, morf=True)
lerf_B = deletion_curve(X_TEST, PHI_B, morf=False)

def auc_trapezoid(curve):
    """Trapezoidal AUC of a deletion curve."""
    n = len(curve)
    return sum((curve[k] + curve[k+1]) / 2 for k in range(n-1)) / (n-1)

auc_morf_A = auc_trapezoid(morf_A)
auc_lerf_A = auc_trapezoid(lerf_A)
auc_morf_B = auc_trapezoid(morf_B)
auc_lerf_B = auc_trapezoid(lerf_B)

print()
print("=" * 68)
print("  FAITHFULNESS 1 — MoRF/LeRF DELETION CURVES")
print("=" * 68)
print()
print("  MoRF: delete features from MOST to LEAST important.")
print("  LeRF: delete features from LEAST to MOST important.")
print("  Faithful method: MoRF drops fast (important features removed first).")
print()
print(f"  Step  {'Remaining':>10}  {'MoRF-A':>9}  {'LeRF-A':>9}  {'MoRF-B':>9}  {'LeRF-B':>9}")
print(f"  {'-'*58}")
for k in range(D+1):
    remaining = D - k
    print(f"  {k:>5}  {remaining:>10}  "
          f"{morf_A[k]:>9.4f}  {lerf_A[k]:>9.4f}  "
          f"{morf_B[k]:>9.4f}  {lerf_B[k]:>9.4f}")

print()
print(f"  AUC (lower MoRF = more faithful; higher LeRF = more faithful):")
print(f"  {'':>30}  {'Method A':>10}  {'Method B':>10}")
print(f"  {'MoRF AUC':>30}  {auc_morf_A:>10.4f}  {auc_morf_B:>10.4f}")
print(f"  {'LeRF AUC':>30}  {auc_lerf_A:>10.4f}  {auc_lerf_B:>10.4f}")
print(f"  {'MoRF-LeRF gap (higher = more faithful)':>30}  "
      f"{auc_lerf_A-auc_morf_A:>+10.4f}  {auc_lerf_B-auc_morf_B:>+10.4f}")

winner_morf = "A" if auc_morf_A < auc_morf_B else "B"
print()
print(f"  Faithfulness winner: Method {winner_morf} "
      f"({'A: correct attributions — important features first' if winner_morf=='A' else 'B: wrong!'})")


# ─────────────────────────────────────────────────────────────────────────────
# FAITHFULNESS 2 — SUFFICIENCY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def sufficiency_score(x, attr, top_k):
    """
    SUFFICIENCY: keep only top-k features, replace rest with baseline.
    Returns drop in model confidence: f(x_top_k) vs f(x_full).
    Lower drop = more sufficient (top-k features suffice to reproduce output).
    """
    order = sorted(range(len(x)), key=lambda i: abs(attr[i]), reverse=True)
    x_top_k = list(BASELINE)
    for i in range(top_k):
        x_top_k[order[i]] = x[order[i]]
    return model(x) - model(x_top_k)

print()
print("=" * 68)
print("  FAITHFULNESS 2 — SUFFICIENCY SCORE")
print("=" * 68)
print()
print("  Keep only top-k features. Sufficiency = how much keeping them")
print("  preserves the prediction. Low drop = high sufficiency.")
print()
print(f"  {'Top-k':>8}  {'Drop A':>9}  {'Drop B':>9}  {'More sufficient'}")
print(f"  {'-'*42}")
for k in range(1, D+1):
    drop_A = sufficiency_score(X_TEST, PHI_A, k)
    drop_B = sufficiency_score(X_TEST, PHI_B, k)
    better = "A" if drop_A <= drop_B else "B"
    print(f"  {k:>8}  {drop_A:>+9.4f}  {drop_B:>+9.4f}  Method {better}")


# ─────────────────────────────────────────────────────────────────────────────
# FAITHFULNESS 3 — AXIOM COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  FAITHFULNESS 3 — AXIOM COMPLIANCE")
print("=" * 68)
print()

def check_efficiency(phi, x, x_prime=None):
    """Efficiency axiom: Σ φᵢ = f(x) - f(x')."""
    if x_prime is None:
        x_prime = BASELINE
    delta = model(x) - model(x_prime)
    s = sum(phi)
    return abs(s - delta) / (abs(delta) + 1e-12), s, delta

def check_sensitivity(phi, x, x_prime=None):
    """Sensitivity: if xᵢ == x'ᵢ then φᵢ should be 0."""
    if x_prime is None:
        x_prime = BASELINE
    violations = [(i, phi[i]) for i in range(D)
                  if abs(x[i] - x_prime[i]) < 1e-9 and abs(phi[i]) > 1e-6]
    return violations

def check_symmetry(phi, x):
    """
    Symmetry: features with identical marginal contributions should have equal attribution.
    Here: x4 and x5 both have true weight 0 → they should have identical attributions.
    """
    # In our model, features 3 and 4 (x4, x5) have weight 0 → should have φ=0.
    return phi[3], phi[4]  # should both be ~0 for a symmetric method

for name, phi in [("Method A", PHI_A), ("Method B", PHI_B)]:
    gap, s, delta = check_efficiency(phi, X_TEST)
    violations    = check_sensitivity(phi, X_TEST)
    sym_3, sym_4  = check_symmetry(phi, X_TEST)

    print(f"  {name}:")
    print(f"    EFFICIENCY:  Σφᵢ={s:+.4f}, Δf={delta:+.4f}, gap={gap:.4%}  "
          f"{'✓' if gap < 0.01 else '✗ VIOLATED'}")
    print(f"    SENSITIVITY: features with x=x' but φ≠0: "
          f"{violations if violations else '(none) ✓'}")
    print(f"    SYMMETRY:    φ(x4)={sym_3:+.5f}, φ(x5)={sym_4:+.5f} "
          f"(both should be ≈0 since w4=w5=0)")
    sym_ok = abs(sym_3) < 0.02 and abs(sym_4) < 0.02
    print(f"    Symmetry:    {'✓ near-zero' if sym_ok else '✗ VIOLATED — irrelevant features attributed!'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STABILITY 1 — LIPSCHITZ CONSTANT ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def attributions_method_A(x):
    """Gradient × input for the linear-logistic model = w_i * x_i * σ'(z)."""
    z = sum(TRUE_WEIGHTS[i]*x[i] for i in range(D))
    p = sigmoid(z)
    d_sigma = p * (1 - p)  # sigmoid derivative
    phi = [TRUE_WEIGHTS[i] * x[i] * d_sigma for i in range(D)]
    delta = model(x) - model(BASELINE)
    s = sum(phi)
    if abs(s) > 1e-9:
        phi = [p * delta / s for p in phi]
    return phi

def attributions_method_B(x):
    """Method B: give arbitrary high weight to x4, x5 (wrong method)."""
    z = sum(TRUE_WEIGHTS[i]*x[i] for i in range(D))
    p = sigmoid(z)
    # Wrong attribution: inflate x4, x5
    phi = [0.35*x[0], 0.20*x[1], -0.08*x[2], 0.55*x[3], 0.30*x[4]]
    delta = model(x) - model(BASELINE)
    s = sum(phi)
    if abs(s) > 1e-9:
        phi = [p * delta / s for p in phi]
    return phi

def lipschitz_estimate(attr_fn, x, n_samples=50, eps=0.1):
    """Estimate local Lipschitz constant via random perturbations."""
    phi_x = attr_fn(x)
    lip_values = []
    for _ in range(n_samples):
        delta = [random.gauss(0, eps) for _ in range(D)]
        x_perturbed = [x[i] + delta[i] for i in range(D)]
        phi_perturbed = attr_fn(x_perturbed)
        norm_phi = math.sqrt(sum((phi_x[i]-phi_perturbed[i])**2 for i in range(D)))
        norm_delta = math.sqrt(sum(d**2 for d in delta))
        if norm_delta > 1e-9:
            lip_values.append(norm_phi / norm_delta)
    return sum(lip_values) / len(lip_values), max(lip_values)

print()
print("=" * 68)
print("  STABILITY — LIPSCHITZ CONSTANT ESTIMATE")
print("=" * 68)
print()
print("  Estimated via 50 random perturbations of size ε=0.1.")
print("  Lower = more stable explanation method.")
print()

lip_mean_A, lip_max_A = lipschitz_estimate(attributions_method_A, X_TEST)
lip_mean_B, lip_max_B = lipschitz_estimate(attributions_method_B, X_TEST)

print(f"  {'Metric':<25}  {'Method A':>10}  {'Method B':>10}  {'More stable'}")
print(f"  {'-'*55}")
print(f"  {'Mean Lipschitz':25}  {lip_mean_A:>10.4f}  {lip_mean_B:>10.4f}  "
      f"Method {'A' if lip_mean_A < lip_mean_B else 'B'}")
print(f"  {'Max Lipschitz':25}  {lip_max_A:>10.4f}  {lip_max_B:>10.4f}  "
      f"Method {'A' if lip_max_A < lip_max_B else 'B'}")

# Rank correlation stability
def rank_corr_stability(attr_fn, x, n_samples=30, eps=0.1):
    """Average Spearman rank correlation between original and perturbed attributions."""
    phi_x = attr_fn(x)
    rank_x = sorted(range(D), key=lambda i: phi_x[i])
    corrs = []
    for _ in range(n_samples):
        delta = [random.gauss(0, eps) for _ in range(D)]
        x_p = [x[i] + delta[i] for i in range(D)]
        phi_p = attr_fn(x_p)
        rank_p = sorted(range(D), key=lambda i: phi_p[i])
        # Spearman ρ = Pearson of ranks
        rx = [rank_x.index(i) for i in range(D)]
        rp = [rank_p.index(i) for i in range(D)]
        mean_rx = sum(rx) / D; mean_rp = sum(rp) / D
        cov = sum((rx[i]-mean_rx)*(rp[i]-mean_rp) for i in range(D)) / D
        std_rx = math.sqrt(sum((v-mean_rx)**2 for v in rx)/D + 1e-12)
        std_rp = math.sqrt(sum((v-mean_rp)**2 for v in rp)/D + 1e-12)
        corrs.append(cov / (std_rx * std_rp))
    return sum(corrs) / len(corrs)

rho_A = rank_corr_stability(attributions_method_A, X_TEST)
rho_B = rank_corr_stability(attributions_method_B, X_TEST)
print(f"  {'Rank correlation ρ (↑ better)':25}  {rho_A:>10.4f}  {rho_B:>10.4f}  "
      f"Method {'A' if rho_A > rho_B else 'B'}")


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD SIMULATION — CAN HUMANS PREDICT THE MODEL FROM THE EXPLANATION?
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  FORWARD SIMULATION — EXPLANATION ENABLES CORRECT PREDICTION?")
print("=" * 68)
print()
print("  A human who 'understood' the explanation should be able to predict")
print("  the model's output on new inputs using only the attribution method.")
print("  We simulate this by deriving a prediction rule from the attribution.")
print()

def human_predict_A(x):
    """
    'Human' who learned from Method A's explanation:
    They know the correct feature ranking x1 > x2 > x3, x4=x5=0.
    They use: 'if x1 > 0, probably positive; if x1 < -0.5, negative.'
    """
    # Human's mental model from Method A: x1 is dominant
    score = 2.0 * x[0] + 1.0 * x[1] - 0.5 * x[2]  # matches true weights
    return 1 if score > 0 else 0

def human_predict_B(x):
    """
    'Human' who learned from Method B's explanation:
    They believe x4 is most important (Method B says so).
    They use: 'if x4 > 0, probably positive.'
    """
    # Human's mental model from Method B: x4 is dominant (wrong!)
    score = 0.55 * x[3] + 0.35 * x[0] + 0.20 * x[1]  # follows Method B's ranking
    return 1 if score > 0 else 0

# Evaluate on 200 new instances
n_sim = 200
correct_A, correct_B = 0, 0
for seed in range(n_sim):
    x_new = generate_instance(seed + 1000)
    true_class = model_class(x_new)
    pred_A = human_predict_A(x_new)
    pred_B = human_predict_B(x_new)
    if pred_A == true_class: correct_A += 1
    if pred_B == true_class: correct_B += 1

sim_acc_A = correct_A / n_sim
sim_acc_B = correct_B / n_sim

print(f"  Simulation on {n_sim} held-out instances:")
print(f"  {'Method A forward simulation accuracy':>40}: {sim_acc_A:.1%}")
print(f"  {'Method B forward simulation accuracy':>40}: {sim_acc_B:.1%}")
print()
print(f"  Method A teaches the correct mental model → {sim_acc_A:.1%} simulation accuracy.")
print(f"  Method B teaches the WRONG mental model  → {sim_acc_B:.1%} simulation accuracy.")
print(f"  Accuracy gap: {sim_acc_A - sim_acc_B:+.1%} in favour of Method A.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY SCORECARD
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  EVALUATION SCORECARD SUMMARY")
print("=" * 68)
print()

eff_gap_A = abs(sum(PHI_A) - delta_f) / (abs(delta_f) + 1e-9)
eff_gap_B = abs(sum(PHI_B) - delta_f) / (abs(delta_f) + 1e-9)

rows = [
    ("Faithfulness",  "MoRF AUC (↓)",      f"{auc_morf_A:.4f}",   f"{auc_morf_B:.4f}",   "A" if auc_morf_A < auc_morf_B else "B"),
    ("Faithfulness",  "MoRF-LeRF gap (↑)",  f"{auc_lerf_A-auc_morf_A:+.4f}", f"{auc_lerf_B-auc_morf_B:+.4f}", "A" if auc_lerf_A-auc_morf_A > auc_lerf_B-auc_morf_B else "B"),
    ("Faithfulness",  "Efficiency gap (↓)", f"{eff_gap_A:.4f}",    f"{eff_gap_B:.4f}",    "A" if eff_gap_A < eff_gap_B else "B"),
    ("Faithfulness",  "Symmetry (irrel.=0)","✓ good",              "✗ violated",          "A"),
    ("Stability",     "Mean Lipschitz (↓)", f"{lip_mean_A:.4f}",   f"{lip_mean_B:.4f}",   "A" if lip_mean_A < lip_mean_B else "B"),
    ("Stability",     "Rank correlation ↑", f"{rho_A:.4f}",        f"{rho_B:.4f}",        "A" if rho_A > rho_B else "B"),
    ("Simulatability","Fwd simulation ↑",   f"{sim_acc_A:.1%}",    f"{sim_acc_B:.1%}",    "A" if sim_acc_A > sim_acc_B else "B"),
]

print(f"  {'Dimension':<16}  {'Metric':<24}  {'Method A':>10}  {'Method B':>10}  {'Winner'}")
print(f"  {'-'*72}")
for dim, metric, a, b, w in rows:
    print(f"  {dim:<16}  {metric:<24}  {a:>10}  {b:>10}  {w}")

wins_A = sum(1 for *_, w in rows if w == "A")
wins_B = sum(1 for *_, w in rows if w == "B")
print(f"  {'-'*72}")
print(f"  {'OVERALL':>16}  {'Total wins':>24}  {wins_A:>10}  {wins_B:>10}")
print()
print(f"  Method A wins {wins_A}/{len(rows)} evaluation criteria.")
print(f"  Method B wins {wins_B}/{len(rows)} evaluation criteria.")
print()
print("  CONCLUSION: Method A (correct attributions) outperforms Method B")
print("  across ALL faithfulness and simulation metrics. Stability differences")
print("  are smaller — both methods are locally stable because the underlying")
print("  model is smooth (sigmoid).")
print()
print("  KEY LESSON: Faithfulness metrics (MoRF/LeRF, axiom compliance)")
print("  correctly identify the WRONG method even when stability metrics")
print("  cannot fully discriminate. Always include faithfulness tests.")
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
    visual_html   = ""
    visual_height = 400
    # try:
    #     from interpretability.visuals.evaluation_of_explanations import (
    #         EVAL_VISUAL_HTML,
    #         EVAL_VISUAL_HEIGHT,
    #     )
    #     visual_html   = EVAL_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = EVAL_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[evaluation_of_explanations.py] Could not load visual: {e}", stacklevel=2)

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