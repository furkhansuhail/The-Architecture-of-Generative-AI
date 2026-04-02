"""
Fairness and Bias — Algorithmic Fairness in Machine Learning
=============================================================

Algorithmic fairness asks: when does a model treat different groups of
people differently, and when does that differential treatment constitute
an injustice? The question sounds simple; answering it rigorously requires
confronting deep tensions between statistical formalisations, moral philosophy,
legal frameworks, and domain-specific context.

This module builds algorithmic fairness theory from first principles —
the formal definitions of fairness criteria, the mathematical impossibility
theorems that govern when criteria can be jointly satisfied, the sources
of bias in the ML pipeline, and the techniques for measuring and mitigating
unfair outcomes.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Fairness and Bias — Algorithmic Fairness in ML"
DISPLAY_NAME = "08 · Fairness & Bias"
ICON = "⚖️"
SUBTITLE = "Fairness criteria, impossibility theorems, bias sources, and mitigation"


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

### Why Fairness Is Hard

In 2016, ProPublica published an investigation showing that COMPAS — a risk
assessment tool used by courts across the United States to predict recidivism
— falsely flagged Black defendants as future criminals at nearly twice the rate
it falsely flagged white defendants. Northpointe, the company that built
COMPAS, responded by showing that its scores were equally accurate across
racial groups: if you looked at everyone assigned a "high risk" score, the
actual recidivism rates were similar across races. Both claims were true.

This is not a contradiction in the data. It is a mathematical consequence of
the different base rates of recidivism between groups, combined with the fact
that it is IMPOSSIBLE to simultaneously satisfy all intuitively reasonable
notions of fairness when base rates differ. The COMPAS debate revealed that
the choice of fairness criterion is a moral and political decision — not a
technical one — and that different legitimate choices lead to incompatible
requirements.

Understanding why requires building the formal machinery from scratch.


##### PART I: THE SETUP — FORMAL NOTATION AND CONCEPTS

### The Basic Framework

Let:
  X ∈ ℝᵈ  — feature vector (observable attributes: income, age, credit history, ...)
  A ∈ {0, 1} — sensitive attribute (race, gender, age group, ...)
  Y ∈ {0, 1} — true label (will recidivate, creditworthy, qualified, ...)
  Ŷ ∈ {0, 1} — model prediction (predicted outcome)
  R ∈ [0, 1] — model score (predicted probability, e.g., risk score)

The sensitive attribute A divides the population into protected groups.
The simplest case: A=0 (majority group), A=1 (minority/protected group).
In practice, A may be multi-valued (multiple races, age brackets, etc.).

The fundamental challenge: X, A, and Y are all correlated. Historical data
reflects historical discrimination, social inequalities, and structural barriers.
A model trained on this data learns these correlations — and may perpetuate
or amplify the inequalities they reflect.

    # ================================================================== #
    **Why removing A from X is insufficient:**

    Naive approach: "just don't use the sensitive attribute as a feature."

    Problem 1 — PROXY FEATURES: X may contain proxies for A.
    Zip code correlates with race (due to residential segregation).
    Name correlates with race and gender (name-based discrimination).
    Credit history correlates with socioeconomic status (which correlates
    with race) due to historical lending discrimination.
    A model trained without A but WITH these proxies will still
    discriminate — often as strongly as if A were included.

    Problem 2 — IRREDUCIBLE CORRELATION: Even after removing all proxies,
    Y may be correlated with A because of structural inequalities that
    affect the outcome (not just the measurement).
    A model that perfectly predicts Y will therefore have A-correlated predictions,
    even with no causal path through the model.

    Problem 3 — REDUCED ACCURACY: Removing A and its proxies may reduce
    predictive accuracy across all groups. The accuracy-fairness tension is real.
    # ================================================================== #


### The Confusion Matrix by Group

Every quantitative fairness analysis starts with the confusion matrix,
computed SEPARATELY for each group A=0 and A=1:

    For group a ∈ {0, 1}:
    ┌─────────────────────────────────────────┐
    │          True Y=0      True Y=1         │
    │ Pred Ŷ=0    TN_a          FN_a          │
    │ Pred Ŷ=1    FP_a          TP_a          │
    └─────────────────────────────────────────┘

    TPR_a = TP_a / (TP_a + FN_a)  = P(Ŷ=1 | Y=1, A=a)  True positive rate
    FPR_a = FP_a / (FP_a + TN_a)  = P(Ŷ=1 | Y=0, A=a)  False positive rate
    FNR_a = FN_a / (FN_a + TP_a)  = P(Ŷ=0 | Y=1, A=a)  False negative rate
    TNR_a = TN_a / (TN_a + FP_a)  = P(Ŷ=0 | Y=0, A=a)  True negative rate
    PPV_a = TP_a / (TP_a + FP_a)  = P(Y=1 | Ŷ=1, A=a)  Positive predictive value
    NPV_a = TN_a / (TN_a + FN_a)  = P(Y=0 | Ŷ=0, A=a)  Negative predictive value

All fairness criteria are conditions relating these quantities across groups.


##### PART II: THE FORMAL FAIRNESS CRITERIA

### Criterion 1 — Demographic Parity (Statistical Parity)

The most intuitive fairness criterion: both groups should receive positive
predictions at the same rate.

    DEMOGRAPHIC PARITY:  P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)

In words: the probability of receiving a positive outcome (loan approved,
released on bail, hired) is the same regardless of group membership.

    Demographic parity difference (DPD):
    DPD = P(Ŷ=1 | A=1) − P(Ŷ=1 | A=0)  ← signed gap
    DPR = P(Ŷ=1 | A=1) / P(Ŷ=1 | A=0)  ← ratio (1.0 = perfect parity)

    Legal standard: the US "80% rule" (EEOC four-fifths rule) says
    a selection rate for any protected group less than 80% of the rate
    for the highest-selected group constitutes adverse impact.
    This is a demographic parity criterion with DPR ≥ 0.80.

    Critique: demographic parity ignores whether the different groups
    ACTUALLY HAVE DIFFERENT QUALIFICATION RATES. If Group 1 has a higher
    true rate of the positive outcome (P(Y=1|A=1) > P(Y=1|A=0)), then
    demographic parity requires the model to UNDERPREDICT group 1's positives
    or OVERPREDICT group 0's positives — trading accuracy for representation.
    This is sometimes called "fairness through unawareness" or "positive action."

    # ================================================================== #
    **Demographic parity in practice:**

    True recidivism rates: Group 0 = 25%, Group 1 = 45%.
    A perfectly accurate model predicts: 25% positive for A=0, 45% for A=1.
    Demographic parity = FALSE (45% ≠ 25%).

    To achieve demographic parity, the model must either:
    (a) Over-predict Group 0 recidivism (false positives for Group 0), or
    (b) Under-predict Group 1 recidivism (false negatives for Group 1).
    Both options introduce errors — accuracy must be sacrificed for parity.
    # ================================================================== #


### Criterion 2 — Equalised Odds (Hardt, Price & Srebro, 2016)

Equalised odds requires both TPR AND FPR to be equal across groups:

    EQUALISED ODDS:
    P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)   (equal TPR)
    AND
    P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)   (equal FPR)

    In words: among people who WILL recidivate, both groups are flagged at
    the same rate. Among people who WON'T recidivate, both groups are
    incorrectly flagged at the same rate.

    This is what ProPublica argued COMPAS was violating: COMPAS had a
    higher FPR for Black defendants (more innocent people flagged as high risk).

Equalised odds is a STRONGER condition than either TPR parity or FPR parity
alone. A model satisfies equalised odds only if it makes the same types
of errors at the same rates for both groups.

    Equalised opportunity (Hardt et al.): the weaker condition requiring
    only equal TPR (not necessarily equal FPR). The argument: if the
    decision is harmful to the person (being denied bail, denied a loan),
    the false negative rate matters more than the false positive rate —
    we care most about equal treatment among the truly deserving.


### Criterion 3 — Calibration (Score-Based Fairness)

Calibration requires that risk scores have the same meaning across groups:

    CALIBRATION:  P(Y=1 | R=r, A=0) = P(Y=1 | R=r, A=1)  for all r ∈ [0,1]

    In words: among all people assigned risk score r, the fraction who
    actually have the positive outcome is r — regardless of group membership.

    This is what Northpointe argued COMPAS satisfied: among all defendants
    assigned a given risk score, the actual recidivism rates were similar
    across racial groups. The score "meant the same thing" for both groups.

    Calibration ensures that the score can be INTERPRETED consistently
    across groups: "high risk" means equally high actual risk for everyone.
    This is particularly important when scores are used by downstream
    decision-makers who need to trust the score's meaning.

    Calibration is directly related to model calibration (see calibration module):
    a well-calibrated model in the global sense satisfies calibration within
    each group IF the model's error structure is the same across groups.
    If the model has different accuracy by group, it may be globally calibrated
    but not group-calibrated.


### Criterion 4 — Individual Fairness (Dwork et al., 2012)

Individual fairness requires that similar individuals receive similar outcomes:

    INDIVIDUAL FAIRNESS:
    If d_X(x, x') ≤ ε  then  d_Y(f(x), f(x')) ≤ δ

    where d_X is a task-appropriate similarity metric on inputs and
    d_Y is a metric on outputs. "Similar people should be treated similarly."

    This is the fairest-sounding criterion — it doesn't aggregate by group
    at all, instead requiring case-by-case consistency. But it requires
    specifying a SIMILARITY METRIC on inputs, which is a hard problem:
    • What makes two individuals similar for the purpose of this decision?
    • Should race be included in the similarity metric?
    • How do we compare different types of features (age vs income vs address)?

    Individual fairness subsumes anti-discrimination IF the similarity metric
    treats race as irrelevant (d_X(x, x') is small even when x and x' differ
    only in race). But designing such a metric is non-trivial.

    In practice: individual fairness is easier to state than to operationalise.
    It is most useful as a conceptual check: "would we give similar people
    different outcomes?" rather than as a computable metric.


### Criterion 5 — Counterfactual Fairness (Kusner et al., 2017)

Counterfactual fairness asks: would the individual receive the same decision
if they had been a member of a different group?

    COUNTERFACTUAL FAIRNESS:
    P(Ŷ_{A←a}(U) = y | X=x, A=a) = P(Ŷ_{A←a'}(U) = y | X=x, A=a)

    where Ŷ_{A←a}(U) is the prediction in a counterfactual world where
    A was set to a (via the do-calculus intervention) and U represents
    background noise variables.

    In plain language: "Would you have gotten the same outcome if you were
    a different race/gender, with everything else about you being the same?"

    Counterfactual fairness requires a CAUSAL MODEL of the data-generating
    process — a directed acyclic graph (DAG) specifying which variables
    causally influence which others. This is a significant requirement:
    causal models must be specified and validated, which is often difficult.

    The causally-resolved features X_{A←a'} are the features of individual x
    in the counterfactual world where A=a'. Some features of x causally
    depend on A (e.g., name, neighbourhood, historical access to credit)
    and would change in the counterfactual; others (e.g., height, eye colour)
    would not.

    # ================================================================== #
    **Counterfactual fairness — why causality is required:**

    Observed features of a Black applicant in the US:
    Credit score: 620 (lower than average for white applicants)
    Home address: zip code 48201 (majority-Black neighbourhood)
    Employment: 3 years at current job

    Counterfactual: the same person, but in a world where they were white.
    In that world:
    - Credit score might be 680 (historical lending bias would not have
      suppressed their access to credit)
    - Home address might be different (residential segregation would not
      have constrained their housing options)
    - Employment might be similar (this is less causally affected by race)

    Counterfactual fairness requires the decision to be the same for the
    Black applicant and their white counterfactual self. To compute this,
    we need to know: which features are causally downstream of race?
    This requires a causal model.
    # ================================================================== #


##### PART III: THE IMPOSSIBILITY THEOREMS

### Theorem 1 — Chouldechova (2017): Calibration and Equalised Odds Are Incompatible

Chouldechova (2017) proved that when base rates differ across groups (P(Y=1|A=0) ≠ P(Y=1|A=1)),
it is MATHEMATICALLY IMPOSSIBLE to simultaneously satisfy:
  1. Calibration (equal PPV across groups)
  2. Equal FPR across groups
  3. Equal FNR across groups

PROOF:
Let p_a = P(Y=1|A=a) be the base rate (prevalence) for group a.
Let FPR_a and FNR_a be the false positive and negative rates for group a.
The positive predictive value (PPV, precision) satisfies:

    PPV_a = p_a × (1 − FNR_a) / [p_a × (1 − FNR_a) + (1 − p_a) × FPR_a]

If we want PPV_0 = PPV_1 (calibration) AND FPR_0 = FPR_1 AND FNR_0 = FNR_1,
substituting equal FPR and FNR:

    p_0 × (1−FNR) / [p_0(1−FNR) + (1−p_0)FPR]
    = p_1 × (1−FNR) / [p_1(1−FNR) + (1−p_1)FPR]

This simplifies to p_0 = p_1. But we assumed p_0 ≠ p_1.
Contradiction. Therefore, all three conditions cannot hold simultaneously.

    # ================================================================== #
    **Chouldechova theorem — the COMPAS case:**

    True recidivism rates:
    Black defendants (group 1): p_1 = 0.51
    White defendants (group 0): p_0 = 0.39

    Since p_1 ≠ p_0, COMPAS cannot simultaneously have:
    • Equal PPV (score means the same thing for both groups) — what Northpointe claimed
    • Equal FPR (both groups falsely labelled at same rate) — what ProPublica wanted
    • Equal FNR (both groups fairly let through at same rate)

    The data FORCES a trade-off. Choosing calibration (equal PPV)
    leads to unequal FPR. Choosing equal FPR leads to uncalibrated scores.
    This is not Northpointe's failure — it is a mathematical impossibility.
    # ================================================================== #


### Theorem 2 — Kleinberg et al. (2016): Three-Way Incompatibility

Kleinberg, Mullainathan & Raghavan (2016) proved that, with differing base rates,
no non-trivial classifier can simultaneously satisfy:
  1. Calibration (within-class calibration)
  2. Balance for the positive class (equal TPR)
  3. Balance for the negative class (equal TNR)

"Non-trivial" means the classifier is not perfectly accurate (which would
trivially satisfy all conditions) and not trivially assigning everyone to
the same class.

This is equivalent to the Chouldechova result in different notation, but
makes the three-way structure even clearer. When base rates differ,
pick any two of the three fairness criteria — you cannot have all three.


### Theorem 3 — Hardt et al. (2016): Equalised Odds Constrains Achievable Accuracy

Hardt, Price & Srebro (2016) showed that if you have a model that is not
equalised-odds fair, the ONLY way to achieve equalised odds through post-
processing is to use randomised thresholding — and this always reduces
accuracy compared to the unconstrained optimal.

Concretely: in the Lagrangian formulation, enforcing equalised odds moves
you off the Pareto frontier of accuracy. There is a real cost to fairness.

The cost depends on:
  • How different the base rates are (larger gap → larger accuracy cost)
  • The correlation between the sensitive attribute and the optimal prediction
  • The degree of fairness constraint (strict equality vs tolerance)

    # ================================================================== #
    **The Pareto frontier of fairness vs accuracy:**

    Accuracy
    ↑
    │  ×           ← unconstrained optimal (unfair)
    │    ×
    │      ××××─── ← Pareto frontier (each point = a fairness-constrained
    │              ← maximum-accuracy model at that fairness level)
    │
    └──────────────────────────→ Fairness (e.g., DP difference)
                                  0.0 (perfectly fair) → ...

    Every point on the frontier is the BEST accuracy achievable
    for that fairness level. Moving toward perfect fairness (left)
    always costs accuracy when base rates differ.
    # ================================================================== #


### What the Impossibility Theorems Mean in Practice

The theorems do NOT say that fairness is impossible. They say:

    1. You must CHOOSE which fairness criterion to prioritise. Different
       legitimate choices lead to different, incompatible requirements.

    2. The choice of criterion is a MORAL AND POLITICAL decision, not a
       technical one. Statistics cannot tell you which criterion is right.
       Context, legal frameworks, the nature of the harm, and the
       preferences of affected communities all must inform the choice.

    3. When you choose one criterion, you are IMPLICITLY TRADING OFF others.
       A decision to enforce demographic parity is simultaneously a decision
       to accept unequal error rates. Transparency requires acknowledging
       this trade-off explicitly.

    4. When base rates are equal, many criteria are simultaneously satisfiable.
       The impossibility arises from group base rate differences, which
       themselves reflect historical injustice. There is no technical
       fix for this — only choices about how to allocate error.


##### PART IV: SOURCES OF BIAS IN THE ML PIPELINE

### The ML Pipeline and Bias Entry Points

Bias can enter at every stage of the machine learning pipeline. Understanding
WHERE bias enters determines which mitigation strategy is appropriate.

    DATA COLLECTION → FEATURE ENGINEERING → MODEL TRAINING → DEPLOYMENT

### Bias Source 1 — Historical Bias

Historical bias arises when the training data reflects historical discrimination
or social inequalities, and the model learns these patterns.

    Example: loan default prediction trained on 1990s lending data.
    In the 1990s, redlining and discriminatory lending practices meant
    that many qualified minority borrowers were denied loans. The training
    data shows lower LOAN APPROVAL RATES for minorities — because of
    discrimination, not because of lower creditworthiness. A model trained
    on this data learns to predict "default risk" partly by predicting
    "minority applicant" — even though the true relationship is a historical artefact.

    Historical bias is particularly pernicious because:
    • It is not a bug in the model — the model is doing exactly what it was trained to do.
    • It is not detectable from the data alone (the discrimination is baked in).
    • It tends to compound over time: biased decisions → biased outcomes → biased data
      → more biased models (a feedback loop).

    The Cambridge Analytica case is a non-lending example: targeting algorithms
    trained on historical political engagement data encoded the historical
    under-engagement of marginalised groups, leading to systematically lower
    political advertisement exposure for those communities.


### Bias Source 2 — Representation Bias

Representation bias occurs when the training data underrepresents certain groups,
causing the model to be less accurate for underrepresented groups.

    Example: face recognition systems trained predominantly on lighter-skinned faces.
    Buolamwini & Gebru (2018) — "Gender Shades" — tested three commercial face
    recognition APIs on faces stratified by skin tone and gender. Error rates:
      Lighter-skinned males:   0.8%
      Darker-skinned males:    3.4%
      Lighter-skinned females: 3.3%
      Darker-skinned females: 34.7%  ← 43× worse than lighter-skinned males!

    The models were not trained to discriminate — they were trained on datasets
    that had 75%+ light-skinned faces. The model simply had less training data
    for dark-skinned faces and thus lower accuracy for that group.

    Representation bias is usually the easiest to detect (measure per-group accuracy)
    but not always easy to fix (you cannot always collect more data for underrepresented
    groups, and oversampling can introduce other artefacts).

    # ================================================================== #
    **Representation bias measurement:**

    For each group a:
      Compute accuracy_a = (TP_a + TN_a) / (TP_a + TN_a + FP_a + FN_a)
      OR other per-group performance metrics.

    Report: accuracy by group, broken down by all sensitive attributes.
    Flag: any group with accuracy substantially below the overall average.
    The threshold for "substantially" depends on context.
    A common rule: flag if any group's accuracy is more than 5 percentage
    points below the overall accuracy.
    # ================================================================== #


### Bias Source 3 — Measurement Bias

Measurement bias occurs when the same construct is measured differently
for different groups, or when the target variable (Y) is itself a
biased proxy for the true quantity of interest.

    Example 1: predicting "criminal recidivism" using re-arrest as a proxy.
    Re-arrest rates reflect POLICING INTENSITY as well as actual criminal behaviour.
    In communities with heavier police presence (which correlates with race),
    the same underlying behaviour produces more arrests. Using re-arrest as Y
    builds policing disparity directly into the model's target variable.

    Example 2: predicting "job performance" using manager ratings.
    Manager ratings are the observed Y. But if managers rate minority
    employees lower due to unconscious bias (not actual performance),
    the model learns to predict biased ratings, not actual performance.
    The model is "accurate" on biased labels.

    Example 3: predicting "health needs" using healthcare costs as proxy.
    Obermeyer et al. (2019) showed a major healthcare algorithm used
    healthcare costs as a proxy for health needs. But minority patients
    had LOWER healthcare costs than equally sick white patients (due to
    unequal access to healthcare). The model predicted lower need for
    minority patients — because they couldn't afford care, not because
    they were healthier.


### Bias Source 4 — Aggregation Bias

Aggregation bias occurs when a single model is trained across multiple
groups whose relationships between features and outcomes differ, producing
a model that fits no group well.

    Example: predicting diabetes risk from HbA1c (blood sugar marker).
    HbA1c has different diagnostic implications for different ethnic groups
    — the threshold for diabetes diagnosis differs. A single model trained
    across all groups uses a one-size-fits-all HbA1c threshold, which may be
    too sensitive for some groups and too lenient for others.

    Mitigation: train separate models per group (when groups are large enough),
    or use group-conditioned models that explicitly allow different feature-
    outcome relationships by group.


### Bias Source 5 — Evaluation Bias

Evaluation bias occurs when the benchmark used to measure model performance
itself reflects historical bias or underrepresents certain groups.

    Example: NLP benchmarks developed predominantly on English text from
    Western contexts. Models trained and evaluated on these benchmarks
    may appear excellent while performing poorly on non-Western names,
    non-standard English dialects, or minority language contexts.

    Standard benchmark performance does NOT imply fairness. Always
    evaluate on group-stratified held-out sets, not just aggregate metrics.


### Bias Source 6 — Deployment and Feedback Bias

Even a well-designed model can generate bias through its deployment context:

    FEEDBACK LOOPS: A model's predictions influence future data. Predictive
    policing sends more officers to predicted high-crime areas → more arrests
    in those areas → more training data labelling those areas as high-crime
    → model becomes more confident about those areas. The model creates the
    "reality" it was measuring.

    USER BEHAVIOUR: Users who know a model's decision criteria will optimise
    for those criteria, changing the distribution the model was trained on.
    "Gaming" a hiring model by emphasising keywords changes which candidates
    apply and how they present themselves.

    POPULATION SHIFT: A model trained on historical applicants is deployed
    to a new applicant pool with different demographics. The model's fairness
    properties on the historical validation set may not transfer.


##### PART V: MEASURING BIAS — THE METRICS TOOLKIT

### The Audit Checklist

A responsible model audit should compute the following metrics, broken
down by each sensitive attribute (race, gender, age group, etc.):

    GROUP PERFORMANCE METRICS (for each group a):
      Accuracy_a:     Overall correct predictions for group a.
      TPR_a:          True positive rate (recall) for group a.
      FPR_a:          False positive rate for group a.
      PPV_a:          Precision for group a.
      F1_a:           F1 score for group a.
      AUC_a:          Area under ROC curve for group a.

    FAIRNESS METRICS (comparing groups a=0 and a=1):
      Demographic Parity Difference: P(Ŷ=1|A=1) − P(Ŷ=1|A=0)
      Equalised Odds Difference: max(|TPR_1 − TPR_0|, |FPR_1 − FPR_0|)
      Predictive Parity: |PPV_1 − PPV_0|
      Accuracy Equity: |Accuracy_1 − Accuracy_0|
      Calibration: test group-conditional calibration (see calibration module)

    DISPARITY RATIO METRICS (for compliance reporting):
      Selection rate ratio: P(Ŷ=1|A=1) / P(Ŷ=1|A=0)
      (EEOC 80% rule: flag if < 0.80)

    # ================================================================== #
    **Fairness metric reference table:**

    Metric                  Definition                  Range       Perfect value
    ──────────────────────────────────────────────────────────────────────────────
    DP difference           P(Ŷ=1|A=1)−P(Ŷ=1|A=0)       [−1,1]       0
    DP ratio                P(Ŷ=1|A=1)/P(Ŷ=1|A=0)       (0,∞)        1
    TPR gap                 TPR_1 − TPR_0               [−1,1]       0
    FPR gap                 FPR_1 − FPR_0               [−1,1]       0
    EO difference           max(|TPR gap|,|FPR gap|)    [0,1]        0
    Predictive parity diff  PPV_1 − PPV_0               [−1,1]       0
    Accuracy gap            Acc_1 − Acc_0               [−1,1]       0
    # =========================================================================== #


### Intersectionality — Multi-Attribute Fairness

Fairness must be evaluated at the INTERSECTION of sensitive attributes,
not just marginally for each attribute separately.

Crenshaw (1989) coined "intersectionality" to describe how multiple forms
of oppression interact: a Black woman's experience is not simply the sum
of her experience as Black and her experience as a woman — it is distinct
and often more severe.

    Example: a hiring model may be "fair by race" (similar TPR for Black and
    white candidates overall) and "fair by gender" (similar TPR for male
    and female candidates overall), yet systematically underperform for
    Black women specifically.

    Measuring intersectionality:
    Evaluate every COMBINATION of protected attributes, not just each attribute
    in isolation. For two binary attributes (race × gender): 4 groups.
    For three binary attributes: 8 groups. For k attributes with mₖ values
    each: Πₖ mₖ groups.

    In practice, many intersection cells will have too few examples for reliable
    estimation. Report the effective sample sizes and confidence intervals
    for all groups. Flag groups with fewer than ~30 examples as unreliable.


### The Impossibility of a Single Fairness Number

Every aggregate fairness metric (mean fairness across groups, worst-group metric,
Pareto-optimal trade-off) embeds a moral choice about how to aggregate group
disparities. There is no neutral single number.

    MEAN DISPARITY: sensitive to outliers; a good average can hide severe
    harm to a small group.

    WORST-GROUP METRIC (Sagawa et al., 2020): minimise the MAXIMUM error
    across all groups. Prioritises the least-favoured group. Tends to sacrifice
    overall accuracy for robustness. Appropriate when harming the worst-off
    group is unacceptable.

    PARETO OPTIMALITY: a model is Pareto-fair if you cannot improve any
    group's metric without worsening another's. Describes the frontier of
    achievable fairness-accuracy trade-offs. Does not tell you WHICH point
    on the frontier to choose.


##### PART VI: MITIGATION STRATEGIES

### The Three Intervention Stages

Bias mitigation interventions occur at three stages of the ML pipeline:

    PRE-PROCESSING (modify the data before training)
    IN-PROCESSING (modify the model or training process)
    POST-PROCESSING (modify the model's outputs after training)


### Pre-Processing — Reweighting and Resampling

REWEIGHTING assigns higher training loss weight to underrepresented or
disadvantaged groups:

    sample_weight_i = desired_representation(group_i) / actual_representation(group_i)

This upweights minority group examples, forcing the model to perform
well on them during training.

RESAMPLING synthetically creates more examples for underrepresented groups:
  • OVERSAMPLING: duplicate minority group examples (or create synthetic
    ones using SMOTE — Synthetic Minority Oversampling TEchnique).
  • UNDERSAMPLING: remove majority group examples.
  • Both approaches risk overfitting to minority group examples (oversampling)
    or wasting available data (undersampling).

FAIR REPRESENTATION LEARNING (Zemel et al., 2013): learn a latent
representation Z of the data that is maximally informative about Y but
minimally informative about A. Train a classifier on Z, not on X.
Conceptually: Z should be the "fairness-sanitised" version of X where
group membership information has been removed.

This is often formalised as minimising:
    L = L_classification(Z → Y) + λ × L_adversarial(Z → A)

The adversarial loss penalises representations from which A is predictable.


### In-Processing — Constrained Optimisation

The standard ML objective is:
    min_{f} L_task(f)  (minimise task loss)

With a fairness constraint:
    min_{f} L_task(f)   subject to  fairness_constraint(f) ≤ ε

This can be solved using:
  • LAGRANGIAN RELAXATION: add the fairness constraint as a penalty term
    min_{f} L_task(f) + λ × fairness_violation(f)
    The hyperparameter λ controls the fairness-accuracy trade-off.

  • ADVERSARIAL DEBIASING (Zhang et al., 2018): train the main classifier
    and an adversarial classifier simultaneously:
    - Classifier f tries to predict Y from X.
    - Adversary g tries to predict A from f's intermediate representations.
    - f is trained to fool g (make A unpredictable) while still predicting Y.

    # ================================================================== #
    **Adversarial debiasing — the min-max game:**

    Classifier loss: L_f = L_class(Y, f(X)) − λ × L_adv(A, g(f(X)))
    Adversary loss:  L_g = L_adv(A, g(f(X)))

    Gradient descent on L_g trains g to predict A.
    Gradient ascent on λ × L_adv(A, g(f(X))) trains f to be HARD for g to predict A.
    The equilibrium: f's representations encode Y but not A.

    This is a zero-sum game between f (wanting fair representations) and
    g (wanting to recover A). At Nash equilibrium, the representations are
    maximally predictive of Y subject to being minimally predictive of A.
    # ================================================================== #

  • REGULARISATION-BASED: Add fairness as a regularisation term:
    min_{f} L_task(f) + λ_1 × |TPR_0 − TPR_1| + λ_2 × |FPR_0 − FPR_1|
    This requires differentiable fairness metrics (TPR, FPR are not
    differentiable; need smooth surrogate losses).


### Post-Processing — Threshold Adjustment

The simplest and most widely deployed mitigation: adjust the DECISION THRESHOLD
separately for each group to achieve the desired fairness criterion.

For a binary classifier with score R(x) ∈ [0,1]:
    Predict Ŷ=1 if R(x) ≥ τ_a  where τ_a is a group-specific threshold.

Finding τ_0 and τ_1 to satisfy a fairness criterion:

    DEMOGRAPHIC PARITY: choose τ_0, τ_1 so that P(R(x)≥τ_0|A=0) = P(R(x)≥τ_1|A=1).
    The thresholds are set so that both groups have the same selection rate.

    EQUALISED ODDS: choose τ_0, τ_1 so that both TPR and FPR match.
    This requires solving a system of two equations in two unknowns.
    Solution: use the ROC curves for each group and find the intersection
    that equalises both rates simultaneously (Hardt et al., 2016).

    # ================================================================== #
    **Threshold adjustment — worked example:**

    Group 0 (base rate 30%): ROC curve passes through (FPR=0.2, TPR=0.7) at τ_0=0.4
    Group 1 (base rate 50%): ROC curve passes through (FPR=0.2, TPR=0.7) at τ_1=0.6

    To achieve equalised odds (equal TPR AND FPR for both groups):
    Choose τ_0 = 0.4 and τ_1 = 0.6.
    Both groups now have FPR=0.2 and TPR=0.7.

    Note: group 1 needs a HIGHER threshold to achieve the same error rates
    because its base rate is higher (more positive examples → easier to get
    high TPR). This is exactly the "different standards" that critics of
    threshold adjustment find objectionable.
    # ================================================================== #

Threshold adjustment is popular because:
  • Simple to implement without retraining.
  • Transparent and auditable.
  • Provides mathematical guarantees (the fairness constraint is enforced by construction).

The critiques:
  • "Different standards for different groups" — applying different thresholds
    by group has been argued to be a form of discrimination itself.
  • Only post-hoc: does not address the root cause of bias in the model or data.
  • Still requires access to the sensitive attribute at decision time.


### Calibrating the Fairness-Accuracy Trade-Off

For any fairness constraint, there is an accuracy cost. The decision-maker
must choose where on the Pareto frontier to operate.

    Tools for this decision:
    1. FAIRNESS-ACCURACY TRADE-OFF CURVES: plot accuracy (or AUC) vs fairness
       metric as the constraint is tightened. Visualise the cost of each unit
       of fairness improvement.

    2. REGRET ANALYSIS: compute the regret (accuracy loss) from enforcing fairness.
       Report: "achieving demographic parity costs 3.2% accuracy vs the
       unconstrained model."

    3. STAKEHOLDER ENGAGEMENT: the affected communities should have input into
       which fairness criterion is prioritised. Technical analysis cannot make
       this choice — it can only clarify the consequences of each choice.


##### PART VII: CAUSAL FAIRNESS — BEYOND STATISTICS

### Why Statistical Fairness Is Insufficient

Statistical fairness criteria (demographic parity, equalised odds, calibration)
are all defined in terms of OBSERVED CORRELATIONS between A, X, Y, and Ŷ.
They say nothing about CAUSALITY. This creates two related problems:

    PROBLEM 1 — SPURIOUS CORRELATION: A model can satisfy demographic parity
    while using A (or a proxy for A) as the primary predictor. If A correlates
    with Y for causal reasons (e.g., race causes different access to resources
    which causes different outcomes), demographic parity is achieved by
    balancing the predictions — but the underlying discrimination persists.

    PROBLEM 2 — RESOLVING VARIABLES: some features are causally downstream
    of A (they were caused partly by the sensitive attribute) and some are not.
    Statistical methods treat all features the same. A causal approach allows
    explicitly specifying which causal pathways from A to Ŷ are acceptable.

    # ================================================================== #
    **The causal framework — pathways from A to Ŷ:**

    In a job hiring model, race (A) may causally affect:
      PATH 1: A → education (discriminatory school funding) → skills → Y
      PATH 2: A → hiring manager bias → Y
      PATH 3: A → neighbourhood → commute time → attendance → Y

    Counterfactual fairness says: block ALL pathways from A to Ŷ.
    But some might argue PATH 1 should be allowed (education is legitimately
    predictive) while PATH 2 should definitely be blocked (it's direct discrimination).

    Causal fairness formalisms (Kilbertus et al., 2017 "Avoiding Discrimination
    through Causal Reasoning") allow specifying WHICH CAUSAL PATHWAYS are
    acceptable and which are not, giving finer-grained control than
    simple statistical criteria.
    # ================================================================== #


### Path-Specific Counterfactual Fairness

Kilbertus et al. (2017) introduced a framework for specifying fairness
through CAUSAL GRAPH ANALYSIS:

    Given a causal DAG G with nodes A (sensitive), X (features), Y (outcome):
    1. Identify "proxies" — variables causally downstream of A (caused by A).
    2. Identify "resolving variables" — proxies that are causally ACCEPTABLE
       mediators (e.g., education might be considered acceptable if we believe
       educational achievement should be rewarded regardless of how access was gained).
    3. Require the prediction Ŷ to be independent of A given the resolving variables.

This framework makes the causal assumptions EXPLICIT, which is both its
strength (transparent) and its weakness (requires agreeing on the causal model,
which is often contested).


##### PART VIII: LEGAL AND NORMATIVE FRAMEWORKS

### Disparate Treatment vs Disparate Impact

US anti-discrimination law recognises two distinct types of discrimination:

    DISPARATE TREATMENT (intentional discrimination):
    Treating someone differently BECAUSE of a protected characteristic.
    A model that explicitly uses race as a feature commits disparate treatment
    (assuming race is a protected characteristic in that context).
    Illegal in most US contexts regardless of intent.

    DISPARATE IMPACT (unintentional discrimination):
    Policies or practices that are facially neutral but have a discriminatory
    EFFECT on a protected group.
    A model that does not use race but produces systematically lower approval
    rates for minority applicants may constitute disparate impact.
    Legal standard: the affected group's selection rate must be at least 80%
    of the highest-selected group's rate (the EEOC four-fifths rule).
    The defendant can justify a disparate impact by showing "business necessity"
    (the practice is job-related and consistent with business necessity).

    Statistical vs causal: disparate treatment is a CAUSAL concept (the outcome
    is caused by the protected attribute). Disparate impact is a STATISTICAL
    concept (the outcome is correlated with the protected attribute, regardless
    of causation).

### The EU AI Act and Algorithmic Fairness Regulation

The EU AI Act (2024) categorises AI systems by risk level:

    HIGH-RISK AI SYSTEMS (requiring mandatory fairness audit):
    Employment screening, credit scoring, access to public services,
    law enforcement, education, border control.

    Requirements for high-risk systems:
    • Documented testing for bias across protected characteristics.
    • Demonstrated absence of discriminatory effects.
    • Human oversight and right to explanation for affected individuals.
    • Regular re-evaluation post-deployment.

    This creates legal obligations to implement the fairness evaluation
    practices described in this module. Companies deploying AI in EU
    high-risk categories face significant regulatory risk without
    documented fairness audits.

    # ================================================================== #
    **The "right to explanation" requirement:**

    Both the EU GDPR (Article 22) and the US Equal Credit Opportunity Act
    require that automated decisions:
    (a) Are explainable to affected individuals upon request.
    (b) Can be contested by affected individuals.
    (c) Can be overridden by humans in high-stakes contexts.

    This creates a direct link between fairness and interpretability:
    fair systems must also be explainable systems. A black-box model
    that is statistically fair but cannot explain its decisions to
    denied applicants may still be legally non-compliant.
    # ================================================================== #


##### PART IX: FAIRNESS IN LARGE LANGUAGE MODELS

### New Challenges for LLM Fairness

Large language models (GPT-4, Claude, Llama, etc.) present fairness
challenges that differ from classical classification settings:

    OPEN-ENDED OUTPUTS: Classification fairness metrics assume discrete outputs.
    LLMs produce open-ended text. Measuring whether an LLM is "fair" in its
    text generation is far harder than measuring TPR gaps.

    NO SINGLE "SENSITIVE ATTRIBUTE": An LLM processes text that may invoke
    any sensitive attribute implicitly. A prompt about "a doctor" may trigger
    gender stereotypes in the model's generation without any explicit gender
    mention.

    STEREOTYPE AMPLIFICATION: LLMs trained on internet text encode the
    stereotypes present in that text. Weidinger et al. (2021) identified
    stereotype amplification as a key harm: models may generate text that
    reinforces harmful stereotypes about marginalised groups, even when
    individual training examples were not intentionally stereotypical.


### Measuring LLM Fairness

    WORD EMBEDDING ASSOCIATION TEST (WEAT, Caliskan et al., 2017):
    Measures the association strength between target concepts and attribute words
    in the model's embedding space.
    Example: test whether "programmer" is more associated with "man" than "woman."
    Inspired by the Implicit Association Test from psychology.

    SEAT (Sentence Encoder Association Test):
    Extension of WEAT to sentence encoders — tests whether sentence embeddings
    encode demographic stereotypes.

    STEREOTYPICAL BIAS BENCHMARKS:
    WinoBias (Zhao et al., 2018): coreference resolution benchmark testing
    whether models resolve pronouns stereotypically (e.g., "The nurse called the
    doctor about his schedule" — does the model assume "his" refers to the doctor?).

    BBQ (Bias Benchmark for QA, Parrish et al., 2021): tests whether models
    answer ambiguous questions by defaulting to demographic stereotypes when
    insufficient information is given.

    OCCUPATION REPRESENTATION TESTS:
    "Write a story about a [doctor/nurse/engineer/teacher]" — measure the
    gendered distribution of pronouns in generated text.

    # ================================================================== #
    **LLM fairness — an illustrative example:**

    Prompt: "Write a story about a brilliant scientist."
    Ideal (fair): uses gender-neutral language or randomly assigns gender.
    Biased: defaults to male pronouns at significantly higher than 50% rate.

    Measurement: run the prompt 100 times. Count male/female/neutral pronouns.
    A fair model should produce approximately equal representation across genders
    in the absence of any gender information in the prompt.

    In practice, most LLMs show male bias for "scientist", "engineer", "CEO"
    and female bias for "nurse", "teacher", "receptionist" — reflecting
    the gender associations in internet training data.
    # ================================================================== #


### Mitigation for LLMs

    RLHF WITH FAIRNESS SIGNAL: Human feedback during RLHF can be structured
    to reward unbiased outputs. Annotators are specifically trained to flag
    stereotypical or biased generations.

    INSTRUCTION TUNING FOR FAIRNESS: Train the model to respond to instructions
    like "answer without gender assumptions" or "avoid stereotypes."

    RED-TEAMING: Systematically probe the model with prompts designed to
    elicit biased outputs. Use findings to improve training data and RLHF.

    CONSTITUTIONAL AI: Build fairness principles directly into the model's
    guiding constitution (the set of principles used during self-critiquing).
    Anthropic's Constitutional AI approach includes fairness as one of the
    principles the model applies when self-revising its outputs.

    REPRESENTATION BALANCING IN TRAINING DATA: Curate training data to
    ensure balanced representation of genders, races, and other demographic
    groups across different contexts and roles.


##### PART X: A COMPLETE WORKED EXAMPLE — COMPAS RECONSIDERED

### Computing All Fairness Metrics for a Synthetic COMPAS-Like Scenario

We use a synthetic dataset with the same structural properties as the
COMPAS data (different base rates by race, a miscalibrated score) to
trace through the full fairness audit.

    SYNTHETIC DATA (N=1000, two groups):
    Group 0 (White defendants):  P(Y=1) = 0.39 (39% recidivism)
    Group 1 (Black defendants):  P(Y=1) = 0.51 (51% recidivism)

    Risk score: R ~ Beta(shaped to be somewhat overconfident)
    Classification threshold: τ = 0.5 (score ≥ 0.5 → predict recidivate)

    CONFUSION MATRIX — GROUP 0:
    True label Y=0 (61%), predicted Ŷ=1: FPR_0 = 0.18  (18% false positive rate)
    True label Y=1 (39%), predicted Ŷ=1: TPR_0 = 0.74  (74% true positive rate)
    PPV_0 = 0.39 × 0.74 / (0.39 × 0.74 + 0.61 × 0.18) = 0.289 / (0.289 + 0.110) = 0.724

    CONFUSION MATRIX — GROUP 1:
    True label Y=0 (49%), predicted Ŷ=1: FPR_1 = 0.28  (28% false positive rate)
    True label Y=1 (51%), predicted Ŷ=1: TPR_1 = 0.74  (74% true positive rate — equal!)
    PPV_1 = 0.51 × 0.74 / (0.51 × 0.74 + 0.49 × 0.28) = 0.377 / (0.377 + 0.137) = 0.734

    FAIRNESS AUDIT:
    Demographic parity:
      P(Ŷ=1|A=0) = 0.39 × 0.74 + 0.61 × 0.18 = 0.289 + 0.110 = 0.399
      P(Ŷ=1|A=1) = 0.51 × 0.74 + 0.49 × 0.28 = 0.377 + 0.137 = 0.514
      DP difference = 0.514 − 0.399 = 0.115  ← VIOLATED (11.5 pp gap)

    Equalised odds:
      TPR gap = |0.74 − 0.74| = 0.00   ← satisfied!
      FPR gap = |0.28 − 0.18| = 0.10   ← VIOLATED (10 pp gap)
      This is exactly the ProPublica finding: FPR differs by 10 pp.

    Predictive parity (calibration of classifier):
      PPV gap = |0.734 − 0.724| = 0.010  ← approximately satisfied!
      This is Northpointe's claim: PPV is approximately equal.

    The impossibility theorem in action:
      - TPR parity: SATISFIED (both 74%)
      - FPR parity: VIOLATED (18% vs 28%)
      - PPV parity: APPROXIMATELY SATISFIED (72.4% vs 73.4%)

    Both ProPublica and Northpointe were reporting correct metrics.
    The difference is which metric they chose to highlight.
    Chouldechova's theorem guarantees that when p_0 ≠ p_1, you CANNOT
    simultaneously have equal TPR, equal FPR, AND equal PPV — unless the
    classifier is perfect. The data had different base rates → one of these
    must be violated. The choice of WHICH to violate is the moral question.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔══════════════════════════╦══════════════════════════════╦═══════════════════╗
    ║ Criterion                ║ Definition                   ║ Key Limitation    ║
    ╠══════════════════════════╬══════════════════════════════╬═══════════════════╣
    ║ Demographic parity       ║ P(Ŷ=1|A=0) = P(Ŷ=1|A=1)      ║ Ignores base rates║
    ║ Equalised odds           ║ Equal TPR AND FPR            ║ Incompatible w/   ║
    ║                          ║                              ║ calibration       ║
    ║ Equalised opportunity    ║ Equal TPR only               ║ Ignores FPR gap   ║
    ║ Calibration              ║ P(Y=1|R=r,A=0)=P(Y=1|R=r,A=1)║ Incompatible w/   ║
    ║                          ║                              ║ equal FPR         ║
    ║ Individual fairness      ║ Similar inputs → similar     ║ Needs similarity  ║
    ║                          ║ outputs                      ║ metric            ║
    ║ Counterfactual fairness  ║ Same outcome if A changed    ║ Needs causal DAG  ║
    ╚══════════════════════════╩══════════════════════════════╩═══════════════════╝

    The Impossibility Result (Chouldechova 2017, Kleinberg et al. 2016):
    When P(Y=1|A=0) ≠ P(Y=1|A=1), no non-trivial classifier can simultaneously satisfy:
      • Calibration (equal PPV)    AND
      • Equal TPR across groups    AND
      • Equal FPR across groups

    Mitigation stages:
      PRE-PROCESSING:    reweighting, resampling, fair representations
      IN-PROCESSING:     constrained optimisation, adversarial debiasing
      POST-PROCESSING:   group-specific threshold adjustment

    Key equation — Chouldechova's impossibility:
    PPV_a = p_a(1−FNR_a) / [p_a(1−FNR_a) + (1−p_a)FPR_a]
    If PPV_0=PPV_1 AND FPR_0=FPR_1 AND FNR_0=FNR_1, then p_0=p_1. Contradiction.
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Fairness from Scratch — Metrics, Impossibility, Threshold Adjustment": {
        "description": (
            "Generates a synthetic binary classification dataset with two groups "
            "having different base rates (mimicking the COMPAS structure). Computes "
            "all standard fairness metrics: demographic parity, equalised odds, "
            "predictive parity, accuracy equity, and calibration. Demonstrates the "
            "Chouldechova impossibility theorem numerically. Implements group-specific "
            "threshold adjustment to enforce equalised odds, demographic parity, and "
            "shows the accuracy cost of each. Plots an ASCII Pareto frontier of "
            "fairness vs accuracy. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "fairness",
        "code": '''
"""
================================================================================
FAIRNESS FROM SCRATCH — METRICS, IMPOSSIBILITY THEOREM, THRESHOLD ADJUSTMENT
================================================================================

We simulate a COMPAS-like binary risk score system with two groups:
  Group 0 (majority): base rate P(Y=1|A=0) = 0.35
  Group 1 (minority): base rate P(Y=1|A=1) = 0.55

Then:
  1. Compute all standard fairness metrics at threshold τ = 0.5
  2. Demonstrate Chouldechova's impossibility theorem numerically
  3. Apply threshold adjustment to achieve each fairness criterion
  4. Show the accuracy cost (fairness-accuracy trade-off)
  5. Plot ASCII Pareto frontier of DP difference vs overall accuracy
================================================================================
"""

import math
import random

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0/(1.0+math.exp(-z)) if z>=0 else math.exp(z)/(1.0+math.exp(z))

def beta_sample(alpha, beta_param):
    """Sample from Beta(alpha, beta) via rejection sampling."""
    while True:
        u = random.random()
        v = random.random()
        x = u ** (1/alpha)
        y = v ** (1/beta_param)
        if x + y <= 1:
            return x / (x + y)

def generate_group(n, base_rate, bias_scale=1.5, seed=0):
    """
    Generate n instances for a group.
    True label Y ~ Bernoulli(base_rate).
    Score R = sigmoid(2*(Y - 0.5) + noise) — correlated with Y but noisy.
    bias_scale makes scores more extreme (simulating overconfident system).
    """
    random.seed(seed)
    data = []
    for _ in range(n):
        y = 1 if random.random() < base_rate else 0
        # Score: higher when y=1, lower when y=0, with noise
        noise = random.gauss(0, 0.4)
        logit = bias_scale * (2*y - 1) + noise
        r = sigmoid(logit)
        data.append((r, y))
    return data

N0, N1   = 600, 400
BASE0    = 0.35  # majority group base rate
BASE1    = 0.55  # minority group base rate

GROUP0 = generate_group(N0, BASE0, bias_scale=1.5, seed=1)
GROUP1 = generate_group(N1, BASE1, bias_scale=1.5, seed=2)


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion(data, threshold=0.5):
    """Returns (TP, FP, TN, FN) for data = list of (score, true_label)."""
    tp = fp = tn = fn = 0
    for r, y in data:
        y_hat = 1 if r >= threshold else 0
        if y == 1 and y_hat == 1: tp += 1
        elif y == 0 and y_hat == 1: fp += 1
        elif y == 0 and y_hat == 0: tn += 1
        else: fn += 1
    return tp, fp, tn, fn

def metrics_from_confusion(tp, fp, tn, fn):
    eps = 1e-9
    tpr = tp / (tp + fn + eps)
    fpr = fp / (fp + tn + eps)
    tnr = tn / (tn + fp + eps)
    fnr = fn / (fn + tp + eps)
    ppv = tp / (tp + fp + eps)  # precision
    npv = tn / (tn + fn + eps)
    acc = (tp + tn) / (tp + fp + tn + fn + eps)
    sel = (tp + fp) / (tp + fp + tn + fn + eps)  # selection rate
    return dict(tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, ppv=ppv, npv=npv, acc=acc, sel=sel)

def compute_all_fairness(g0, g1, tau=0.5):
    c0 = compute_confusion(g0, tau)
    c1 = compute_confusion(g1, tau)
    m0 = metrics_from_confusion(*c0)
    m1 = metrics_from_confusion(*c1)

    # Overall
    n = len(g0) + len(g1)
    tp_all = c0[0] + c1[0]; fp_all = c0[1] + c1[1]
    tn_all = c0[2] + c1[2]; fn_all = c0[3] + c1[3]
    m_all  = metrics_from_confusion(tp_all, fp_all, tn_all, fn_all)

    return {
        "acc_0": m0["acc"], "acc_1": m1["acc"], "acc_all": m_all["acc"],
        "tpr_0": m0["tpr"], "tpr_1": m1["tpr"],
        "fpr_0": m0["fpr"], "fpr_1": m1["fpr"],
        "ppv_0": m0["ppv"], "ppv_1": m1["ppv"],
        "sel_0": m0["sel"], "sel_1": m1["sel"],
        "dp_diff": m1["sel"] - m0["sel"],
        "dp_ratio": m1["sel"] / (m0["sel"] + 1e-9),
        "tpr_gap": m1["tpr"] - m0["tpr"],
        "fpr_gap": m1["fpr"] - m0["fpr"],
        "eo_diff": max(abs(m1["tpr"]-m0["tpr"]), abs(m1["fpr"]-m0["fpr"])),
        "ppv_gap": m1["ppv"] - m0["ppv"],
        "acc_gap": m1["acc"] - m0["acc"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL AUDIT AT τ = 0.5
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("  INITIAL FAIRNESS AUDIT (threshold τ = 0.5)")
print("=" * 68)
print()

TAU = 0.5
fm = compute_all_fairness(GROUP0, GROUP1, TAU)

print(f"  Dataset: Group 0 (n={N0}, base rate={BASE0:.0%})  "
      f"Group 1 (n={N1}, base rate={BASE1:.0%})")
print()
print(f"  {'Metric':<30}  {'Group 0':>9}  {'Group 1':>9}  {'Gap/Ratio':>11}  {'Fair?'}")
print(f"  {'-'*68}")
rows = [
    ("Accuracy",         fm["acc_0"],  fm["acc_1"],  fm["acc_gap"],   abs(fm["acc_gap"])<0.05),
    ("Selection rate",   fm["sel_0"],  fm["sel_1"],  fm["dp_diff"],   abs(fm["dp_diff"])<0.05),
    ("TPR (recall)",     fm["tpr_0"],  fm["tpr_1"],  fm["tpr_gap"],   abs(fm["tpr_gap"])<0.05),
    ("FPR",              fm["fpr_0"],  fm["fpr_1"],  fm["fpr_gap"],   abs(fm["fpr_gap"])<0.05),
    ("PPV (precision)",  fm["ppv_0"],  fm["ppv_1"],  fm["ppv_gap"],   abs(fm["ppv_gap"])<0.05),
]
for name, v0, v1, gap, is_fair in rows:
    fair_str = "✓" if is_fair else f"✗ ({gap:+.3f})"
    print(f"  {name:<30}  {v0:>9.4f}  {v1:>9.4f}  {gap:>+11.4f}  {fair_str}")

print()
print(f"  Overall accuracy: {fm['acc_all']:.4f}")
print(f"  DP ratio (Group 1 / Group 0): {fm['dp_ratio']:.3f}  "
      f"{'✓ ≥ 0.80' if fm['dp_ratio'] >= 0.80 else '✗ < 0.80 (EEOC violation)'}")


# ─────────────────────────────────────────────────────────────────────────────
# CHOULDECHOVA IMPOSSIBILITY THEOREM — NUMERICAL DEMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  CHOULDECHOVA IMPOSSIBILITY THEOREM — NUMERICAL DEMONSTRATION")
print("=" * 68)
print()
print("  Theorem: when base rates differ, we cannot simultaneously have")
print("  equal TPR, equal FPR, AND equal PPV (calibrated predictions).")
print()
print("  We verify this by checking the formula:")
print("  PPV_a = p_a(1-FNR_a) / [p_a(1-FNR_a) + (1-p_a)FPR_a]")
print()

p0 = BASE0; p1 = BASE1
tpr_a = fm["tpr_0"]; fpr_a = fm["fpr_0"]
fnr_a = 1 - tpr_a

print(f"  Observed: TPR_0={fm['tpr_0']:.4f}, FPR_0={fm['fpr_0']:.4f}")
print(f"  Observed: TPR_1={fm['tpr_1']:.4f}, FPR_1={fm['fpr_1']:.4f}")
print()

# Compute PPV from the formula (using group 0's rates for both — what WOULD happen)
ppv_formula_with_g0_rates_for_g1 = (
    p1 * (1 - (1 - tpr_a)) /
    (p1*(1-(1-tpr_a)) + (1-p1)*fpr_a)
)
print(f"  Hypothetical: if Group 1 had SAME TPR and FPR as Group 0,")
print(f"  what PPV would Group 1 have?")
print(f"  PPV_1 (formula) = {p1:.2f}×{tpr_a:.4f} / "
      f"({p1:.2f}×{tpr_a:.4f} + {1-p1:.2f}×{fpr_a:.4f})")
print(f"                  = {ppv_formula_with_g0_rates_for_g1:.4f}")
print()
ppv_g0 = p0 * tpr_a / (p0*tpr_a + (1-p0)*fpr_a)
print(f"  PPV_0 (formula) = {ppv_g0:.4f}")
print(f"  PPV_1 (formula) = {ppv_formula_with_g0_rates_for_g1:.4f}")
print(f"  PPV gap WHEN equal TPR and FPR: {ppv_formula_with_g0_rates_for_g1 - ppv_g0:+.4f}")
print()
print(f"  → Even with EXACTLY EQUAL TPR and FPR, the PPVs differ by "
      f"{abs(ppv_formula_with_g0_rates_for_g1 - ppv_g0):.4f}.")
print(f"  This is FORCED by the different base rates ({p0:.0%} vs {p1:.0%}).")
print(f"  You cannot have all three equal simultaneously. ∎")


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD ADJUSTMENT FOR DIFFERENT FAIRNESS CRITERIA
# ─────────────────────────────────────────────────────────────────────────────

def find_threshold_for_tpr(data, target_tpr):
    """Find threshold that gives target TPR for a group."""
    positives = sorted([r for r, y in data if y == 1], reverse=True)
    n_pos = len(positives)
    if n_pos == 0: return 0.5
    target_idx = int(target_tpr * n_pos)
    if target_idx >= n_pos: return 0.0
    return positives[target_idx]

def find_threshold_for_fpr(data, target_fpr):
    """Find threshold that gives target FPR for a group."""
    negatives = sorted([r for r, y in data if y == 0], reverse=True)
    n_neg = len(negatives)
    if n_neg == 0: return 0.5
    target_idx = int(target_fpr * n_neg)
    if target_idx >= n_neg: return 0.0
    return negatives[target_idx]

def find_threshold_for_sel(data, target_sel):
    """Find threshold that gives target selection rate."""
    scores = sorted([r for r, _ in data], reverse=True)
    n = len(scores)
    target_idx = int(target_sel * n)
    if target_idx >= n: return 0.0
    return scores[target_idx]

print()
print("=" * 68)
print("  THRESHOLD ADJUSTMENT — ENFORCING FAIRNESS CRITERIA")
print("=" * 68)
print()

# Strategy 1: Equal TPR (equalised opportunity)
target_tpr = (fm["tpr_0"] + fm["tpr_1"]) / 2  # average TPR
tau0_eo = find_threshold_for_tpr(GROUP0, target_tpr)
tau1_eo = find_threshold_for_tpr(GROUP1, target_tpr)
fm_eo = compute_all_fairness(GROUP0, GROUP1, tau0_eo)
fm_eo1 = compute_all_fairness(GROUP0, GROUP1, tau1_eo)

# Strategy 2: Demographic parity
target_sel = (fm["sel_0"] + fm["sel_1"]) / 2
tau0_dp = find_threshold_for_sel(GROUP0, target_sel)
tau1_dp = find_threshold_for_sel(GROUP1, target_sel)

# Compute metrics with different thresholds for each group
def compute_fairness_split_tau(g0, g1, tau0, tau1):
    """Compute fairness with group-specific thresholds."""
    c0 = compute_confusion(g0, tau0)
    c1 = compute_confusion(g1, tau1)
    m0 = metrics_from_confusion(*c0)
    m1 = metrics_from_confusion(*c1)
    tp_all = c0[0]+c1[0]; fp_all = c0[1]+c1[1]
    tn_all = c0[2]+c1[2]; fn_all = c0[3]+c1[3]
    m_all = metrics_from_confusion(tp_all, fp_all, tn_all, fn_all)
    return m0, m1, m_all

m0_dp, m1_dp, mall_dp = compute_fairness_split_tau(GROUP0, GROUP1, tau0_dp, tau1_dp)
m0_eo, m1_eo, mall_eo = compute_fairness_split_tau(GROUP0, GROUP1, tau0_eo, tau1_eo)

# Baseline (both at 0.5)
m0_base, m1_base, mall_base = compute_fairness_split_tau(GROUP0, GROUP1, 0.5, 0.5)

strategies = [
    ("No fairness (τ=0.5 both)",
     0.5, 0.5, m0_base, m1_base, mall_base),
    (f"Demog. parity (τ0={tau0_dp:.3f}, τ1={tau1_dp:.3f})",
     tau0_dp, tau1_dp, m0_dp, m1_dp, mall_dp),
    (f"Equal opp. (τ0={tau0_eo:.3f}, τ1={tau1_eo:.3f})",
     tau0_eo, tau1_eo, m0_eo, m1_eo, mall_eo),
]

print(f"  {'Strategy':<40}  {'OvAcc':>6}  {'DPdiff':>7}  {'TPRgap':>7}  {'FPRgap':>7}  {'PPVgap':>7}")
print(f"  {'-'*76}")
for name, t0, t1, m0, m1, mall in strategies:
    dp_d = m1["sel"] - m0["sel"]
    tpr_g = m1["tpr"] - m0["tpr"]
    fpr_g = m1["fpr"] - m0["fpr"]
    ppv_g = m1["ppv"] - m0["ppv"]
    print(f"  {name:<40}  {mall['acc']:>6.4f}  {dp_d:>+7.4f}  "
          f"{tpr_g:>+7.4f}  {fpr_g:>+7.4f}  {ppv_g:>+7.4f}")

print()
base_acc = mall_base["acc"]
dp_acc   = mall_dp["acc"]
eo_acc   = mall_eo["acc"]

print(f"  ACCURACY COST OF FAIRNESS INTERVENTIONS:")
print(f"    Demographic parity: {(base_acc-dp_acc)*100:+.2f} pp vs no constraint")
print(f"    Equalised opportunity: {(base_acc-eo_acc)*100:+.2f} pp vs no constraint")
print()
print(f"  NOTE: Both strategies reduce overall accuracy. The Chouldechova")
print(f"  theorem guarantees this cost is unavoidable when base rates differ.")


# ─────────────────────────────────────────────────────────────────────────────
# ASCII PARETO FRONTIER — ACCURACY vs DP DIFFERENCE
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  PARETO FRONTIER — ACCURACY vs DEMOGRAPHIC PARITY DIFFERENCE")
print("=" * 68)
print()
print("  Each point = one pair of thresholds (τ_0, τ_1).")
print("  Moving left (more DP) costs accuracy (lower).")
print()

# Sweep τ_0 and τ_1 to trace the frontier
tau_grid = [i/20 for i in range(1, 20)]
points = []
for tau1 in tau_grid:
    m0, m1, mall = compute_fairness_split_tau(GROUP0, GROUP1, 0.5, tau1)
    dp_d = abs(m1["sel"] - m0["sel"])
    acc  = mall["acc"]
    points.append((dp_d, acc))

# Normalise for ASCII
min_dp = min(p[0] for p in points); max_dp = max(p[0] for p in points)
min_ac = min(p[1] for p in points); max_ac = max(p[1] for p in points)

HEIGHT, WIDTH = 8, 50
grid = [[" "]*WIDTH for _ in range(HEIGHT)]

for dp_d, acc in points:
    col = int((dp_d - min_dp) / (max_dp - min_dp + 1e-9) * (WIDTH-1))
    row = HEIGHT - 1 - int((acc - min_ac) / (max_ac - min_ac + 1e-9) * (HEIGHT-1))
    col = max(0, min(WIDTH-1, col))
    row = max(0, min(HEIGHT-1, row))
    grid[row][col] = "●"

print(f"  Acc ↑")
for i, row in enumerate(grid):
    label = f"  {min_ac + (HEIGHT-1-i)*(max_ac-min_ac)/(HEIGHT-1):.3f}" if i == 0 or i == HEIGHT-1 else "       "
    print(label + "│" + "".join(row))
print(f"       └{'─'*WIDTH}")
print(f"       {min_dp:.2f}" + " "*(WIDTH//2-4) + "← DP difference →" + " "*4 + f"{max_dp:.2f}")
print()
print(f"  Each ● = one threshold configuration.")
print(f"  Top-left: high accuracy, high DP difference (unfair)")
print(f"  Bottom-right: low accuracy, low DP difference (fair)")
print(f"  The Pareto frontier shows the cost of fairness:")
print(f"    Achieving DP difference ≈ 0 costs ~{(max_ac-min_ac)*100:.1f} pp of accuracy.")
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
    #     from interpretability.visuals.fairness_and_bias import (
    #         FAIRNESS_VISUAL_HTML,
    #         FAIRNESS_VISUAL_HEIGHT,
    #     )
    #     visual_html   = FAIRNESS_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = FAIRNESS_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[fairness_and_bias.py] Could not load visual: {e}", stacklevel=2)

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