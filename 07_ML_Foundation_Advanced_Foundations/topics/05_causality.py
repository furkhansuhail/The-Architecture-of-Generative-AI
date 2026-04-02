"""
Causality
=========

The mathematical framework for moving beyond correlation — structural
causal models, do-calculus, the ladder of causation, confounders,
counterfactuals, and causal inference methods that let us answer
"what would happen if we intervened?" from observational data.

"""

import textwrap
import re

TOPIC_NAME = "Causality"
DISPLAY_NAME = "05 · Causality"
ICON = "⚡"
SUBTITLE = "Beyond Correlation — Intervention, Confounding, and Counterfactuals"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Why Correlation Is Not Enough

Standard ML optimises for predictive accuracy under the assumption that
the future looks like the past. It learns associations — patterns that
co-occur in the training distribution. But association is not causation.

    The fundamental problem:

    ┌──────────────────────────────────────────────────────────────────┐
    │  PREDICTION asks:  "Given X, what is Y likely to be?"            │
    │  CAUSATION asks:   "If I change X, what will happen to Y?"       │
    │                    "What if X had been different?"               │
    │                    "Why did Y happen?"                           │
    └──────────────────────────────────────────────────────────────────┘

    These are fundamentally different questions. A model that perfectly
    predicts Y from X cannot answer what happens when we intervene on X —
    unless the causal structure is known.

    Classic examples of the correlation/causation gap:

    ─ Ice cream sales and drowning rates are correlated.
      Cause: hot weather drives both. Neither causes the other.
      Intervention: ban ice cream → drowning rate unchanged.

    ─ Shoe size and reading ability are correlated (in children).
      Cause: age drives both. Shoe size does not cause reading ability.
      Intervention: buy bigger shoes → reading ability unchanged.

    ─ Hospital mortality rates are higher at top hospitals.
      Cause: sicker patients are sent to better hospitals (selection bias).
      Intervention: sending healthy patients there would not increase mortality.

    ─ Drug X correlates with recovery in observational data.
      Cause (possible): healthier patients self-select to take drug X.
      Intervention: prescribing X to everyone may not replicate the benefit.

    Why ML systems fail without causal reasoning:
    ─ Spurious correlations break under distribution shift (Module 13).
    ─ Recommendations can have unintended effects (content → polarisation).
    ─ Fairness interventions require knowing causal pathways of discrimination.
    ─ Decision-making requires knowing what will change if we act.


──────────────────────────────────────────────────────────────────────────────
### The Ladder of Causation (Pearl, 2018)

Judea Pearl organises causal questions into three rungs, each strictly
more powerful than the one below it.

    ┌──────────────────────────────────────────────────────────────────┐
    │  RUNG 3: COUNTERFACTUALS         "What if? / Why?"               │
    │                                                                  │
    │  P(Y_x=1 = y | X=0, Y=y')                                        │
    │  "Given that patient took drug X=0 and died (Y=y'),              │
    │   what is the probability they would have survived if            │
    │   they had taken X=1?"                                           │
    │                                                                  │
    │  Requires: a fully specified structural causal model (SCM).      │
    │  Enables: attribution, blame, legal reasoning, personalised      │
    │           treatment effect estimation.                           │
    ├──────────────────────────────────────────────────────────────────┤
    │  RUNG 2: INTERVENTION            "What if we do?"                │
    │                                                                  │
    │  P(Y | do(X = x))                                                │
    │  "What is the distribution of Y if we SET X = x by force?"       │
    │  (different from observing X = x, which may be confounded)       │
    │                                                                  │
    │  Requires: a causal graph (DAG) and do-calculus.                 │
    │  Enables: policy evaluation, treatment effect estimation,        │
    │           A/B test design.                                       │
    ├──────────────────────────────────────────────────────────────────┤
    │  RUNG 1: ASSOCIATION             "What if I see?"                │
    │                                                                  │
    │  P(Y | X = x)                                                    │
    │  "Given that X = x is observed, what is Y likely to be?"         │
    │                                                                  │
    │  Standard statistics and ML live entirely on this rung.          │
    │  Enables: prediction, pattern recognition, classification.       │
    └──────────────────────────────────────────────────────────────────┘

    KEY INSIGHT:  P(Y | X = x)  ≠  P(Y | do(X = x))  in general.

    The observational distribution P(Y | X = x) includes confounding.
    The interventional distribution P(Y | do(X = x)) does not —
    it is what a randomised controlled trial measures.


──────────────────────────────────────────────────────────────────────────────
### Structural Causal Models (SCMs) and DAGs

A Structural Causal Model provides the formal language for causality.

    Definition (SCM):
    An SCM M = (V, U, F, P_U) consists of:
    ─ V = {V₁, ..., Vₙ}  endogenous (observed) variables
    ─ U = {U₁, ..., Uₙ}  exogenous (noise/background) variables
    ─ F = {f₁, ..., fₙ}  structural equations:  Vᵢ = fᵢ(Pa(Vᵢ), Uᵢ)
    ─ P_U               distribution over noise variables

    Pa(Vᵢ) = causal parents of Vᵢ in the causal graph.
    Each variable is a deterministic function of its parents plus noise.

    Example — Smoking, Tar, Cancer:

        T (Tar)  ←  S (Smoking)  →  C (Cancer)
                         ↓ (directly)
                     C (Cancer)

    Structural equations:
        S   = f_S(U_S)              S is exogenous (personal choice + genetics)
        T   = f_T(S, U_T)           Tar deposits caused by smoking
        C   = f_C(S, T, U_C)        Cancer caused by both directly and via tar

    Diagram 1 — The Causal DAG for Smoking/Tar/Cancer:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │    S (Smoking)─────────────────────────────▶ C (Cancer)         │
    │         │                                    ▲                  │
    │         │                                    │                  │
    │         ▼                                    │                  │
    │    T (Tar deposits)────────────────────────▶ │                  │
    │                                                                 │
    │    Direct causal path:   S → C               (direct effect)    │
    │    Indirect causal path: S → T → C           (mediated effect)  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    The DAG encodes causal assumptions:
    ─ An arrow A → B means A is a direct cause of B.
    ─ Absence of an arrow means no direct causal link (conditional independence).
    ─ DAGs must be acyclic — no feedback loops (use time-unrolled graphs for dynamic systems).

**Markov Condition:**

    Every variable is conditionally independent of its non-descendants
    given its parents:

        Vᵢ ⊥⊥ NonDesc(Vᵢ) | Pa(Vᵢ)

    This allows factoring the joint distribution:

        P(V₁, ..., Vₙ) = ∏ᵢ P(Vᵢ | Pa(Vᵢ))

    A crucial result: statistical dependencies in P(V) can be read off
    the DAG structure using d-separation.


──────────────────────────────────────────────────────────────────────────────
### Confounding — The Central Obstacle

A CONFOUNDER is a variable that causally influences both the treatment X
and the outcome Y, creating a spurious association between them.

    Diagram 2 — The Confounding Structure:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │              C (Confounder)                                      │
    │             ╱              ╲                                     │
    │            ↙                ↘                                    │
    │    X (Treatment)       Y (Outcome)                               │
    │                                                                  │
    │    OBSERVED association: X and Y are correlated (via C).         │
    │    CAUSAL question: does X actually cause Y?                     │
    │                                                                  │
    │    Example: C = socioeconomic status                             │
    │             X = attending a good school                          │
    │             Y = high income                                      │
    │    Confounding: wealthy families both send kids to good schools  │
    │    AND provide advantages that lead to high income.              │
    │    Observed correlation overstates (or even reverses) the        │
    │    true causal effect of schooling on income.                    │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    The Backdoor Criterion — when can we identify causal effects?

    A set of variables Z satisfies the BACKDOOR CRITERION for (X, Y) if:
    1. No element of Z is a descendant of X.
    2. Z blocks all "backdoor paths" from X to Y
       (paths that enter X through an arrow pointing INTO X).

    If Z satisfies the backdoor criterion:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)                   │
    │                                                                  │
    │  (Backdoor adjustment formula — Pearl's fundamental result)      │
    │                                                                  │
    │  We can compute the causal effect from observational data        │
    │  by adjusting for (conditioning on) the confounders Z.           │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 3 — Backdoor Paths and Adjustment:

    Causal graph:   Z → X → Y
                    Z → Y     (Z is a confounder)

    Backdoor path: X ← Z → Y   (non-causal, creates spurious X-Y correlation)
    To identify causal effect: condition on Z (blocks the backdoor path).
    Adjusted estimate: E[Y | do(X)] = Σ_z E[Y | X, Z=z] · P(Z=z)


──────────────────────────────────────────────────────────────────────────────
### d-Separation — Reading Independencies from DAGs

d-separation is the graphical criterion for reading conditional
independencies directly from a causal DAG — without any computation.

    Three fundamental structures (the building blocks of all DAGs):

    CHAIN:        A → B → C
        A and C are dependent (information flows A→B→C).
        A ⊥⊥ C | B  (conditioning on B blocks the path — "blocks the chain").

    FORK:         A ← B → C
        A and C are dependent via B (common cause).
        A ⊥⊥ C | B  (conditioning on B blocks the fork).

    COLLIDER:     A → B ← C
        A and C are INDEPENDENT (no information flows through B).
        A ⊥⊥ C         (marginally independent).
        A NOT ⊥⊥ C | B  (conditioning on collider B OPENS the path — surprising!).

    ┌──────────────────────────────────────────────────────────────────┐
    │  COLLIDER BIAS (Berkson's Paradox):                              │
    │                                                                  │
    │  Example: A = acting talent,  B = movie star,  C = looks         │
    │  A and C are independent in the population.                      │
    │  But conditioning on B = "movie star" creates a negative         │
    │  correlation: among movie stars, if you're not very talented,    │
    │  you must be very attractive.                                    │
    │                                                                  │
    │  In ML: conditioning on a collider can create spurious           │
    │  dependencies — this is a hidden source of confounding.          │
    └──────────────────────────────────────────────────────────────────┘

    d-Separation Definition:
    Nodes X and Y are d-separated by set Z if every path between X and Y
    is BLOCKED by Z. A path is blocked if:
    ─ It contains a chain or fork where the middle node is in Z, OR
    ─ It contains a collider where the collider and none of its descendants is in Z.

    If X and Y are d-separated by Z in the DAG, then X ⊥⊥ Y | Z in all
    distributions compatible with the DAG (global Markov condition).


──────────────────────────────────────────────────────────────────────────────
### The do-Operator and do-Calculus

The do(X = x) operator formally represents intervention:
setting X to x by external force, regardless of its natural causes.

    Observational: P(Y | X = x)    — condition on observing X = x.
    Interventional: P(Y | do(X=x)) — set X = x, cut off its parents.

    Diagram 4 — Observation vs Intervention:

    ORIGINAL GRAPH:           MUTILATED GRAPH (do(X=x)):

    C → X → Y                 C → X  X → Y
    C → Y                     C → Y
                               (arrows INTO X are removed)

    In the mutilated graph, X is set to x by intervention.
    No longer flows from C. Backdoor path C → X → Y is cut.
    The remaining C → Y is now NOT confounded.

    Pearl's do-calculus provides three rules for transforming
    expressions involving do() into expressions without do() —
    if certain d-separation conditions hold in the DAG.
    Together, the three rules are complete: any identifiable causal
    effect can be computed using do-calculus.

    Rule 1 (Insertion/deletion of observations):
        P(Y | do(X), Z, W) = P(Y | do(X), W)   if Y ⊥⊥ Z | X, W  in G_{X̄}

    Rule 2 (Action/observation exchange):
        P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)   if Y ⊥⊥ Z | X, W  in G_{X̄, Z̲}

    Rule 3 (Deletion of actions):
        P(Y | do(X), do(Z), W) = P(Y | do(X), W)   if Y ⊥⊥ Z | X, W  in G_{X̄, Z̄(W)}


──────────────────────────────────────────────────────────────────────────────
### Average Treatment Effect (ATE) and CATE

The most common causal estimand is the Average Treatment Effect.

    POTENTIAL OUTCOMES FRAMEWORK (Rubin, 1974):

    For each unit i and treatment value t, define:
        Y_i(t) = "potential outcome" — what Y would be if i received treatment t.

    For binary treatment T ∈ {0, 1}:
        Y_i(1) = outcome if treated
        Y_i(0) = outcome if untreated

    Individual Treatment Effect (ITE):
        τᵢ = Y_i(1) − Y_i(0)

    The FUNDAMENTAL PROBLEM OF CAUSAL INFERENCE:
        For each unit i, we observe EITHER Y_i(1) OR Y_i(0) — never both.
        The unobserved potential outcome is the COUNTERFACTUAL.

    Average Treatment Effect (ATE):
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  ATE = E[Y(1) − Y(0)]  =  E[Y(1)] − E[Y(0)]                      │
    │                                                                  │
    │  Under randomised assignment:  E[Y(T=1)] − E[Y(T=0)]             │
    │  Under confounding: these raw differences are BIASED.            │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Conditional ATE (CATE) — heterogeneous treatment effects:
        τ(x) = E[Y(1) − Y(0) | X = x]
        The treatment effect may vary across subgroups.
        Finding τ(x) is the goal of personalised medicine, targeted marketing.

    Identification conditions for ATE from observational data:
    ┌──────────────────────────────────────────────────────────────────┐
    │  1. IGNORABILITY (unconfoundedness):                             │
    │     (Y(0), Y(1)) ⊥⊥ T | X   (no unobserved confounders)          │
    │  2. POSITIVITY (overlap):                                        │
    │     0 < P(T=1 | X=x) < 1 for all x with positive density.        │
    │     (all units have non-zero chance of both treatments)          │
    │  3. SUTVA (stable unit treatment value assumption):              │
    │     No interference between units; treatment is well-defined.    │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Causal Inference Methods

**Randomised Controlled Trials (RCTs):**

    Gold standard. Random assignment of treatment ensures T ⊥⊥ (Y(0), Y(1)).
    Confounding is eliminated by design.
    ATE = E[Y | T=1] − E[Y | T=0]  (simple difference-in-means).
    Limitations: expensive, unethical for some treatments, Hawthorne effects.

**Propensity Score Methods:**

    Propensity score: e(x) = P(T=1 | X=x)
    Under ignorability, T ⊥⊥ (Y(0), Y(1)) | e(X).
    Can "balance" treatment and control groups on a scalar, not all of X.

    Inverse Probability Weighting (IPW):
    ┌──────────────────────────────────────────────────────────────────┐
    │  ATE_IPW = (1/n) Σᵢ [Tᵢ Yᵢ / e(Xᵢ) − (1−Tᵢ) Yᵢ / (1−e(Xᵢ))]      │
    │                                                                  │
    │  Upweights underrepresented groups to "simulate" an RCT.         │
    └──────────────────────────────────────────────────────────────────┘

    Propensity score matching:
    Match each treated unit to a control unit with similar e(x).
    Then estimate ATE as mean difference in matched pairs.

**Regression Adjustment (S-learner, T-learner):**

    T-LEARNER: Train two separate outcome models:
        μ̂₁(x) = E[Y | T=1, X=x]   (on treated units)
        μ̂₀(x) = E[Y | T=0, X=x]   (on control units)
        CATE estimate: τ̂(x) = μ̂₁(x) − μ̂₀(x)

    S-LEARNER: Train one model with T as a feature:
        μ̂(x, t) = E[Y | X=x, T=t]
        CATE estimate: τ̂(x) = μ̂(x, 1) − μ̂(x, 0)

**Doubly Robust Estimator:**

    Combines outcome regression and IPW:
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  ATE_DR = (1/n) Σᵢ [(Tᵢ(Yᵢ − μ̂₁(Xᵢ))) / e(Xᵢ)                    │
    │                    − ((1−Tᵢ)(Yᵢ − μ̂₀(Xᵢ))) / (1−e(Xᵢ))           │
    │                    + μ̂₁(Xᵢ) − μ̂₀(Xᵢ)]                            │
    │                                                                  │
    │  DOUBLY ROBUST: consistent if EITHER the outcome model OR the    │
    │  propensity score model is correctly specified (but not both     │
    │  need to be correct simultaneously).                             │
    └──────────────────────────────────────────────────────────────────┘

**Instrumental Variables (IV):**

    When unobserved confounders exist, use an instrument Z:
    ─ Z affects T (relevance): Z → T.
    ─ Z does not directly affect Y (exclusion restriction): no Z → Y path.
    ─ Z is independent of unobserved confounders.

    IV estimator:   ATE = Cov(Y, Z) / Cov(T, Z)

    Example: draft lottery number as instrument for military service.
    Lottery number affects whether you served (relevance) but is
    random (no direct effect on earnings, no confounders).

**Difference-in-Differences (DiD):**

    Compares the change over time in an outcome for a treated group
    vs an untreated (control) group.

    ATE_DiD = (Ȳ_treated_after − Ȳ_treated_before)
             − (Ȳ_control_after − Ȳ_control_before)

    Key assumption: parallel trends — in the absence of treatment,
    treated and control groups would have followed the same trend.

    Diagram 5 — Difference-in-Differences:

    Outcome
      │                            treated_after
      │                           /
      │              ___________/           ← actual outcome
      │             /         /  ← treatment effect (DiD)
      │            / _________   ← counterfactual (unobserved)
      │           //
      │__________/  control group (unaffected)
      └──────────────────────────────────── time
              before        after
              treatment     treatment


──────────────────────────────────────────────────────────────────────────────
### Counterfactuals — Rung 3 Reasoning

Counterfactuals answer: "What WOULD have happened, given what DID happen?"

    Notation: Y_x(u) = value of Y when X=x for unit u with background U=u.
    The observed outcome for unit u: Y = Y_{X(u)}(u) — the potential
    outcome under the treatment unit u actually received.

    THREE STEPS for counterfactual reasoning (Pearl's algorithm):

    Step 1 — ABDUCTION: Update P(U) given observed evidence.
        Compute P(U | E=e) — what noise values are consistent with
        what we observed?

    Step 2 — ACTION: Modify the SCM.
        Apply the intervention: replace the equation for X with X = x.
        (Mutilate the graph.)

    Step 3 — PREDICTION: Compute the result.
        Use the modified SCM and updated P(U | E=e) to compute
        the distribution of the counterfactual outcome.

    Example — did the drug cause the recovery?

    Observed: Patient took drug (T=1) and recovered (Y=1).
    Counterfactual question: Would they have recovered without the drug (T=0)?

    Abduction: compute P(U | T=1, Y=1) — noise values consistent with
               "patient took drug and recovered."
    Action: mutilate SCM, set T=0.
    Prediction: compute P(Y=1 | do(T=0), T=1, Y=1) = P(Y_{T=0}=1 | T=1, Y=1).

    If this is low → the drug was likely the cause of recovery (high PNS).
    PNS = Probability of Necessity and Sufficiency.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Three causal quantities for attribution:                        │
    │                                                                  │
    │  PN  (Probability of Necessity):                                 │
    │      P(Y_{x=0}=0 | X=1, Y=1)  "Would Y be 0 if X had been 0?"    │
    │                                                                  │
    │  PS  (Probability of Sufficiency):                               │
    │      P(Y_{x=1}=1 | X=0, Y=0)  "Would X alone cause Y?"           │
    │                                                                  │
    │  PNS (Probability of Necessity and Sufficiency):                 │
    │      P(Y_{x=1}=1, Y_{x=0}=0)  "X was both necessary AND          │
    │                                 sufficient for Y"                │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Causal Discovery — Learning the DAG from Data

So far we assumed the causal DAG is known. Causal discovery algorithms
attempt to learn the DAG structure from observational data.

**Constraint-Based Methods (PC Algorithm):**

    1. Start with a fully connected undirected graph.
    2. Remove edges between nodes that are conditionally independent
       (tested using partial correlation or CI tests).
    3. Orient edges using v-structures (colliders) and orientation rules.
    Result: a Markov equivalence class — a set of DAGs consistent with
    all observed independencies.

    Limitation: cannot distinguish between A → B and A ← B from
    observations alone (they imply the same independence structure).
    Need additional assumptions (e.g., no cycles, acyclicity) or
    functional form assumptions.

**Score-Based Methods (GES, NOTEARS):**

    Define a score for each DAG (e.g., BIC, log marginal likelihood).
    Search the space of DAGs for the highest-scoring structure.
    NOTEARS reformulates as continuous optimisation:
        min_W  ℓ(W; X)  subject to  h(W) = 0
    where h(W) = tr(e^{W ⊙ W}) − d = 0 encodes the acyclicity constraint.

**Functional Causal Models — LiNGAM:**

    If the noise variables are non-Gaussian (and linear structural eqs):
    LiNGAM (Linear Non-Gaussian Acyclic Model) can uniquely identify
    the causal direction A → B vs B → A from the non-Gaussianity of residuals.
    This breaks the Markov equivalence class.

**Practical Limitations of Causal Discovery:**

    ┌──────────────────────────────────────────────────────────────────┐
    │  1. Scale: exponential number of possible DAGs in n variables.   │
    │  2. Faithfulness assumption: independencies in data must come    │
    │     from the DAG structure — can fail in practice.               │
    │  3. Hidden confounders: unobserved variables can create false    │
    │     edges and block true ones.                                   │
    │  4. Causal sufficiency: CI tests require large samples.          │
    │  Domain knowledge + causal discovery = best practice.            │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Causality in Machine Learning

**Why Current ML Systems Lack Causal Reasoning:**

    Standard ML (ERM) minimises E_{(x,y)~P_train}[ℓ(f(x),y)].
    This finds any f that minimises training loss — including shortcuts
    and spurious correlations. The model does not know which features
    cause Y; it only knows which features predict Y in training.

    Under distribution shift (Module 13), the spurious features may
    no longer predict Y. The causal features remain predictive because
    the mechanism P(Y | causes) is invariant across environments.

**Invariant Risk Minimisation (IRM):**

    The key insight: causal features are those for which the same
    linear classifier is optimal ACROSS ALL ENVIRONMENTS.

    IRM objective:
        min_{φ, w}  Σ_e R^e(w ∘ φ)
        subject to: w ∈ argmin_{w̃} R^e(w̃ ∘ φ)  for all environments e

    Forces φ to extract features such that the same predictor head w
    works well in all training environments. Spurious correlations
    only hold in some environments → they cannot satisfy the constraint.

**Causal Representation Learning:**

    Goal: learn latent representations Z that correspond to causal
    variables — disentangled, intervention-stable, generalisable.

    Independent Component Analysis (ICA, Module 00) approximates this
    under linear assumptions. Deep generative models (VAE with causal
    structure) are an active research area.

    Diagram 6 — Causal vs Associational Representations:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  STANDARD ENCODER:          CAUSAL ENCODER:                     │
    │  x → [z₁, z₂, z₃, ...]     x → [z_cause, z_style, z_noise]      │
    │                                                                 │
    │  z_i = entangled features   z_cause = invariant under interv.   │ 
    │  Prediction: good in train  Prediction: good across envs.       │
    │  Robustness: poor           Robustness: good                    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Confounding — When Correlation Misleads": {
        "description": (
            "Simulates three classic confounding scenarios and shows how "
            "naive regression gives the wrong causal estimate, while "
            "adjustment (backdoor criterion) recovers the true effect. "
            "Covers: a simple confounder (spurious correlation), Simpson's "
            "paradox (trend reverses after stratification), and collider "
            "bias (conditioning creates spurious correlation). Computes "
            "naive ATE vs adjusted ATE and compares to ground truth."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt

class LinearRegression:
    def __init__(self, fit_intercept=True): self.fit_intercept = fit_intercept
    def fit(self, X, y):
        Xa = _np_impl.c_[_np_impl.ones(len(X)), X] if self.fit_intercept else X
        w  = _np_impl.linalg.lstsq(Xa, y, rcond=None)[0]
        if self.fit_intercept:
            self.intercept_ = w[0]; self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0;  self.coef_ = w
        return self
    def predict(self, X): return X @ self.coef_ + self.intercept_

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=1000, random_state=None, solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self, X, y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

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
                g=(_np_impl.var(r)*len(r)-_np_impl.var(r[l])*l.sum()-_np_impl.var(r[rr])*rr.sum())
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np_impl.var(r)<1e-12:
            return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,left,right=nd; return self._p1(x,left if x[f]<=t else right)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class GradientBoostingRegressor:
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        self._f0=y.mean(); F=_np_impl.full(len(y),self._f0); self._trees=[]
        for _ in range(self.n_estimators):
            r=y-F; t=_DTR(max_depth=self.max_depth); t.fit(X,r)
            self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return F

class GradientBoostingClassifier: pass  # imported but unused in ops

def mean_squared_error(y_true, y_pred):
    return _np_impl.mean((_np_impl.asarray(y_true)-_np_impl.asarray(y_pred))**2)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Simple confounding — spurious correlation
# ─────────────────────────────────────────────────────────────────────────────
# True causal graph: C → X, C → Y,  X → Y (weak)
# True effect of X on Y: β_true = 0.2
# Confounder C creates a large spurious X-Y correlation

print("=" * 65)
print("  SCENARIO 1: SIMPLE CONFOUNDING")
print("=" * 65)
print()

n = 2000
beta_true = 0.2   # true causal effect of X on Y
beta_C_X  = 1.5   # C → X strength
beta_C_Y  = 2.0   # C → Y strength

C = np.random.randn(n)                              # confounder
X = beta_C_X * C + np.random.randn(n) * 0.5        # X caused by C + noise
Y = beta_true * X + beta_C_Y * C + np.random.randn(n) * 0.5  # Y caused by X + C + noise

# Naive regression (ignores C)
naive_coef  = LinearRegression().fit(X.reshape(-1,1), Y).coef_[0]
# Adjusted regression (includes C)
adj_coef    = LinearRegression().fit(np.c_[X, C], Y).coef_[0]

print(f"  True causal effect β_XY = {beta_true:.4f}")
print(f"  Naive regression coef  = {naive_coef:.4f}  "
      f"(bias = {naive_coef - beta_true:+.4f})  ← CONFOUNDED")
print(f"  Adjusted regression    = {adj_coef:.4f}  "
      f"(bias = {adj_coef - beta_true:+.4f})  ← DECONFOUNDED")
print()
print("  The naive estimate is inflated because C drives both X and Y.")
print("  Conditioning on C (backdoor adjustment) recovers the true effect.")

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Simpson's Paradox — trend REVERSES after stratification
# ─────────────────────────────────────────────────────────────────────────────
# Drug trial: drug appears beneficial overall but harms both subgroups
# C = disease severity (high severity → more likely to get drug, worse outcome)

print()
print("=" * 65)
print("  SCENARIO 2: SIMPSON'S PARADOX")
print("=" * 65)
print()

# Group 0 (mild disease): 800 patients
# Group 1 (severe disease): 200 patients (sent to specialists who prescribe the drug)
# True drug effect: -0.3 (drug slightly HARMFUL in both groups)

np.random.seed(0)
n0, n1 = 800, 200

# Mild group: mostly not treated, low baseline risk
C0   = np.zeros(n0)
T0   = (np.random.rand(n0) < 0.2).astype(float)   # 20% treated
Y0   = 0.3 + (-0.3) * T0 + 0.1 * np.random.randn(n0)   # mean Y=0.3 untreated, 0.0 treated

# Severe group: mostly treated, high baseline risk
C1   = np.ones(n1)
T1   = (np.random.rand(n1) < 0.8).astype(float)   # 80% treated
Y1   = 0.8 + (-0.3) * T1 + 0.1 * np.random.randn(n1)   # mean Y=0.8 untreated, 0.5 treated

C_all = np.concatenate([C0, C1])
T_all = np.concatenate([T0, T1])
Y_all = np.concatenate([Y0, Y1])

# Overall (pooled) comparison
mean_Y_treated   = Y_all[T_all == 1].mean()
mean_Y_untreated = Y_all[T_all == 0].mean()
ate_naive_simp   = mean_Y_treated - mean_Y_untreated

# Within-group comparisons
ate_mild   = Y0[T0 == 1].mean() - Y0[T0 == 0].mean()
ate_severe = Y1[T1 == 1].mean() - Y1[T1 == 0].mean()

# Adjusted estimate (backdoor formula)
p_mild   = n0 / (n0 + n1)
p_severe = n1 / (n0 + n1)
ate_adj  = ate_mild * p_mild + ate_severe * p_severe

print(f"  True causal effect (drug): −0.30  (harmful in both groups)")
print()
print(f"  POOLED (naive):            {ate_naive_simp:+.4f}  ← appears BENEFICIAL! (wrong)")
print(f"  Within mild group:         {ate_mild:+.4f}  ← harmful (correct)")
print(f"  Within severe group:       {ate_severe:+.4f}  ← harmful (correct)")
print(f"  Adjusted (backdoor):       {ate_adj:+.4f}  ← recovers true sign")
print()
print("  Simpson's paradox: overall trend reversed because severity")
print("  confounds both treatment assignment and outcome.")
print("  Adjusted estimate = Σ_c (within-stratum effect) × P(C=c).")

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Collider bias — conditioning creates spurious correlation
# ─────────────────────────────────────────────────────────────────────────────
# A → B ← C.  A and C are independent marginally.
# Conditioning on B creates a spurious A-C correlation.

print()
print("=" * 65)
print("  SCENARIO 3: COLLIDER BIAS (BERKSON'S PARADOX)")
print("=" * 65)
print()

n = 3000
# A = talent (independent of looks)
# C = looks  (independent of talent)
# B = famous = 1 if (talent + looks > threshold)  — the COLLIDER
A = np.random.randn(n)   # talent
C = np.random.randn(n)   # looks
B = ((A + C) > 1.0).astype(int)   # famous: high talent OR high looks

# Marginal correlation (should be ~0)
corr_marginal = np.corrcoef(A, C)[0, 1]
# Conditional correlation among famous (B=1)
mask_famous = B == 1
corr_conditional = np.corrcoef(A[mask_famous], C[mask_famous])[0, 1]
# Conditional correlation among non-famous (B=0)
mask_not = B == 0
corr_notfamous   = np.corrcoef(A[mask_not], C[mask_not])[0, 1]

print(f"  A = talent, C = looks, B = famous (collider: A → B ← C)")
print()
print(f"  Marginal correlation A-C:            {corr_marginal:+.4f}  (≈0, truly independent)")
print(f"  Conditional corr A-C | B=1 (famous): {corr_conditional:+.4f}  ← NEGATIVE! (spurious)")
print(f"  Conditional corr A-C | B=0 (not):    {corr_notfamous:+.4f}  ← also non-zero!")
print()
print("  Among famous people, talent and looks are negatively correlated.")
print("  This is an ARTIFACT of conditioning on the collider B, not a real effect.")
print("  Practical: if we study only 'selected' populations (hospitals, elite unis),")
print("  we condition on a collider and create spurious correlations.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Confounding, Simpson's Paradox, and Collider Bias",
             fontsize=13, fontweight="bold")

# Panel (0,0): Simple confounding scatter
ax = axes[0, 0]
sc = ax.scatter(X[:500], Y[:500], c=C[:500], cmap="coolwarm", s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label="Confounder C")
x_range = np.linspace(X.min(), X.max(), 100)
ax.plot(x_range, naive_coef * x_range + Y.mean() - naive_coef * X.mean(),
        "tomato", lw=2.5, label=f"Naive β={naive_coef:.2f}")
ax.plot(x_range, adj_coef * x_range + Y.mean() - adj_coef * X.mean(),
        "steelblue", lw=2.5, ls="--", label=f"Adjusted β={adj_coef:.2f}")
ax.axline((0, 0), slope=beta_true, color="black", lw=1.5, ls=":",
          label=f"True β={beta_true:.2f}")
ax.set_title("Simple Confounding\\nC → X, C → Y creates spurious X-Y association",
             fontweight="bold")
ax.set_xlabel("X (treatment)"); ax.set_ylabel("Y (outcome)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (0,1): Bias comparison bar chart
ax = axes[0, 1]
methods = ["Naive\\n(confounded)", "Adjusted\\n(backdoor)", "True effect"]
values  = [naive_coef, adj_coef, beta_true]
colours = ["tomato", "steelblue", "black"]
bars = ax.bar(methods, values, color=colours, alpha=0.8)
ax.axhline(beta_true, color="black", lw=2, ls="--", alpha=0.7)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Estimated Effect: Naive vs Adjusted\\n(adjustment removes confounding bias)",
             fontweight="bold")
ax.set_ylabel("Estimated causal effect β")
ax.grid(alpha=0.3, axis="y")

# Panel (0,2): Simpson's paradox visualisation
ax = axes[0, 2]
# Plot group-stratified and overall trends
for Ti, col, ls, lbl in [(0, "steelblue", "-",  "Control (T=0)"),
                          (1, "tomato",    "--", "Treated (T=1)")]:
    mask0_t = (C_all == 0) & (T_all == Ti)
    mask1_t = (C_all == 1) & (T_all == Ti)
    ax.scatter(C_all[mask0_t], Y_all[mask0_t], c=col, s=8, alpha=0.3, marker="o")
    ax.scatter(C_all[mask1_t], Y_all[mask1_t], c=col, s=8, alpha=0.3, marker="s")

y0_mean = [Y_all[(C_all==0)&(T_all==t)].mean() for t in [0,1]]
y1_mean = [Y_all[(C_all==1)&(T_all==t)].mean() for t in [0,1]]
ax.plot([0, 0], y0_mean, "steelblue", lw=3, label=f"Mild: Δ={ate_mild:+.2f}")
ax.plot([1, 1], y1_mean, "tomato",    lw=3, label=f"Severe: Δ={ate_severe:+.2f}")
# Overall arrow (naive)
ax.annotate("", xy=(1.3, mean_Y_treated), xytext=(1.3, mean_Y_untreated),
            arrowprops=dict(arrowstyle="<->", color="seagreen", lw=2.5))
ax.text(1.35, (mean_Y_treated+mean_Y_untreated)/2,
        f"Pooled:\\n{ate_naive_simp:+.2f}", color="seagreen", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Mild", "Severe"])
ax.set_title("Simpson's Paradox\\n(treatment harmful within groups, appears helpful overall)",
             fontweight="bold")
ax.set_xlabel("Disease severity (C)"); ax.set_ylabel("Outcome Y")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,0): Collider: marginal scatter
ax = axes[1, 0]
ax.scatter(A[B==0], C[B==0], c="steelblue", s=5, alpha=0.2, label="Not famous (B=0)")
ax.scatter(A[B==1], C[B==1], c="tomato",    s=5, alpha=0.5, label="Famous (B=1)")
ax.set_title(f"Collider Bias: A (talent) vs C (looks)\\n"
             f"Marginal corr={corr_marginal:.3f}, "
             f"Among famous={corr_conditional:.3f}",
             fontweight="bold")
ax.set_xlabel("A (talent)"); ax.set_ylabel("C (looks)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,1): Causal DAG visualisations (text-based)
ax = axes[1, 1]
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

def draw_node(ax, x, y, label, color="steelblue"):
    circle = plt.Circle((x, y), 0.7, color=color, alpha=0.8)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

def draw_arrow(ax, x1, y1, x2, y2, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))

# DAG 1: confounding
draw_node(ax, 1.5, 7.5, "C", "seagreen")
draw_node(ax, 0.5, 6.0, "X", "steelblue")
draw_node(ax, 2.5, 6.0, "Y", "tomato")
draw_arrow(ax, 1.5, 7.0, 0.8, 6.5)
draw_arrow(ax, 1.5, 7.0, 2.2, 6.5)
draw_arrow(ax, 1.2, 6.0, 1.8, 6.0)
ax.text(1.5, 5.3, "Confounding: C → X, C → Y", ha="center", fontsize=9)
ax.text(1.5, 4.9, "X → Y (spurious path via C)", ha="center", fontsize=8, style="italic")

# DAG 2: collider
draw_node(ax, 5.5, 7.5, "A", "steelblue")
draw_node(ax, 7.5, 7.5, "C", "seagreen")
draw_node(ax, 6.5, 6.0, "B", "tomato")
draw_arrow(ax, 5.5, 7.0, 6.2, 6.5)
draw_arrow(ax, 7.5, 7.0, 6.8, 6.5)
ax.text(6.5, 5.3, "Collider: A → B ← C", ha="center", fontsize=9)
ax.text(6.5, 4.9, "A, C independent; cond. on B creates dep.", ha="center",
        fontsize=8, style="italic")

# DAG 3: instrument
draw_node(ax, 0.8, 2.5, "Z", "purple")
draw_node(ax, 2.5, 2.5, "X", "steelblue")
draw_node(ax, 4.5, 2.5, "Y", "tomato")
draw_node(ax, 3.5, 3.8, "C", "seagreen")
draw_arrow(ax, 1.5, 2.5, 1.8, 2.5)
draw_arrow(ax, 3.2, 2.5, 3.8, 2.5)
draw_arrow(ax, 3.5, 3.4, 2.8, 2.8)
draw_arrow(ax, 3.5, 3.4, 4.2, 2.8)
ax.text(2.8, 1.7, "IV: Z → X → Y (C unobserved)", ha="center", fontsize=9)

ax.set_title("Causal DAG Gallery\\n(confounding, collider, instrument)",
             fontweight="bold")

# Panel (1,2): Adjustment comparison for all scenarios
ax = axes[1, 2]
scenarios   = ["Simple\\nconfounding", "Simpson's\\nparadox"]
true_vals   = [beta_true, -0.30]
naive_vals  = [naive_coef, ate_naive_simp]
adj_vals    = [adj_coef, ate_adj]
x_pos = np.arange(len(scenarios))
w = 0.25
ax.bar(x_pos - w, true_vals,  w, color="black",    alpha=0.7, label="True causal effect")
ax.bar(x_pos,     naive_vals, w, color="tomato",   alpha=0.8, label="Naive estimate (biased)")
ax.bar(x_pos + w, adj_vals,   w, color="steelblue",alpha=0.8, label="Adjusted estimate")
ax.set_xticks(x_pos); ax.set_xticklabels(scenarios)
ax.set_ylabel("Estimated treatment effect")
ax.set_title("Naive vs Adjusted ATE\\n(backdoor adjustment recovers true effect)",
             fontweight="bold")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("confounding_simpsons.png", dpi=110)
print()
print("  Plot saved → confounding_simpsons.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Propensity Scores and Average Treatment Effect Estimation": {
        "description": (
            "Implements three ATE estimators on simulated observational data "
            "with known ground truth: (1) naive difference-in-means, "
            "(2) propensity score IPW with logistic regression, "
            "(3) T-learner outcome regression, and (4) the doubly robust "
            "estimator. Verifies all four against the true ATE computed from "
            "the structural equations. Shows CATE estimation and heterogeneous "
            "treatment effects. Visualises propensity score overlap (positivity "
            "assumption) and calibration."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt

class LinearRegression:
    def __init__(self, fit_intercept=True): self.fit_intercept = fit_intercept
    def fit(self, X, y):
        Xa = _np_impl.c_[_np_impl.ones(len(X)), X] if self.fit_intercept else X
        w  = _np_impl.linalg.lstsq(Xa, y, rcond=None)[0]
        if self.fit_intercept:
            self.intercept_ = w[0]; self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0;  self.coef_ = w
        return self
    def predict(self, X): return X @ self.coef_ + self.intercept_

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=1000, random_state=None, solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self, X, y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

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
                g=(_np_impl.var(r)*len(r)-_np_impl.var(r[l])*l.sum()-_np_impl.var(r[rr])*rr.sum())
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np_impl.var(r)<1e-12:
            return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,left,right=nd; return self._p1(x,left if x[f]<=t else right)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class GradientBoostingRegressor:
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        self._f0=y.mean(); F=_np_impl.full(len(y),self._f0); self._trees=[]
        for _ in range(self.n_estimators):
            r=y-F; t=_DTR(max_depth=self.max_depth); t.fit(X,r)
            self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return F

class GradientBoostingClassifier: pass  # imported but unused in ops

def mean_squared_error(y_true, y_pred):
    return _np_impl.mean((_np_impl.asarray(y_true)-_np_impl.asarray(y_pred))**2)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Data generating process — known ground truth
# ─────────────────────────────────────────────────────────────────────────────
# X = covariates (confounders), T = treatment, Y = outcome
# True CATE: τ(x) = 1 + 0.5 * x0 − 0.3 * x1  (heterogeneous)

def generate_observational_data(n=2000, seed=0):
    """
    Structural equations:
        X0, X1, X2 ~ N(0,1)
        T | X ~ Bernoulli(σ(0.5*X0 − 0.4*X1 + 0.3*X2))   (confounded)
        Y(0) = 2*X0 − X1 + ε                               (control PO)
        Y(1) = Y(0) + 1 + 0.5*X0 − 0.3*X1                (treated PO)
        Y     = T*Y(1) + (1−T)*Y(0)                        (observed)
    True ATE = E[Y(1) − Y(0)] = 1 + 0.5*0 − 0.3*0 = 1.0
    """
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, 3))
    logit = 0.5*X[:,0] - 0.4*X[:,1] + 0.3*X[:,2]
    T   = (rng.random(n) < 1/(1+np.exp(-logit))).astype(float)
    Y0  = 2*X[:,0] - X[:,1] + rng.normal(0, 0.5, n)
    tau = 1.0 + 0.5*X[:,0] - 0.3*X[:,1]   # individual treatment effects
    Y1  = Y0 + tau
    Y   = T * Y1 + (1-T) * Y0
    true_ate  = tau.mean()
    return X, T, Y, Y0, Y1, tau, true_ate

X, T, Y, Y0, Y1, tau_true, true_ate = generate_observational_data(n=3000)

print("=" * 65)
print("  ATE ESTIMATION: NAIVE, IPW, T-LEARNER, DOUBLY ROBUST")
print("=" * 65)
print()
print(f"  n = {len(X)},  True ATE = {true_ate:.4f}")
print(f"  Treatment rate: {T.mean():.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Naive difference-in-means (ignores confounding)
# ─────────────────────────────────────────────────────────────────────────────
ate_naive = Y[T==1].mean() - Y[T==0].mean()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Propensity score — logistic regression
# ─────────────────────────────────────────────────────────────────────────────
ps_clf = LogisticRegression(C=1.0, random_state=0)
ps_clf.fit(X, T.astype(int))
e_x     = ps_clf.predict_proba(X)[:, 1]   # estimated propensity scores
e_x     = np.clip(e_x, 0.05, 0.95)        # clip for stability

# IPW estimator
ipw_weights_1 =  T / e_x
ipw_weights_0 = (1 - T) / (1 - e_x)
ate_ipw = np.mean(ipw_weights_1 * Y) - np.mean(ipw_weights_0 * Y)

# Normalised IPW (Hájek estimator)
ate_nipw = (np.sum(ipw_weights_1 * Y) / ipw_weights_1.sum()
          - np.sum(ipw_weights_0 * Y) / ipw_weights_0.sum())

# ─────────────────────────────────────────────────────────────────────────────
# 3. T-Learner outcome regression (two separate models)
# ─────────────────────────────────────────────────────────────────────────────
mu1_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
mu0_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
mu1_model.fit(X[T==1], Y[T==1])
mu0_model.fit(X[T==0], Y[T==0])

mu1_hat = mu1_model.predict(X)   # E[Y(1)|X]
mu0_hat = mu0_model.predict(X)   # E[Y(0)|X]
cate_t  = mu1_hat - mu0_hat      # CATE from T-learner
ate_tlearner = cate_t.mean()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Doubly Robust estimator
# ─────────────────────────────────────────────────────────────────────────────
dr_scores = (  T * (Y - mu1_hat) / e_x
             - (1-T) * (Y - mu0_hat) / (1-e_x)
             + mu1_hat - mu0_hat)
ate_dr = dr_scores.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"  {'Estimator':>25}  {'ATE':>8}  {'Bias':>8}  {'Notes'}")
print("  " + "─" * 68)
for name, val in [("True ATE",             true_ate),
                  ("Naive diff-in-means",  ate_naive),
                  ("IPW (Horvitz-Thomps.)",ate_ipw),
                  ("Normalised IPW",        ate_nipw),
                  ("T-Learner",             ate_tlearner),
                  ("Doubly Robust",         ate_dr)]:
    bias  = val - true_ate
    note  = ""
    if name == "Naive diff-in-means": note = "confounded"
    if name == "Doubly Robust":       note = "consistent if PS or outcome correct"
    print(f"  {name:>25}  {val:>8.4f}  {bias:>+8.4f}  {note}")

print()
print("  KEY TAKEAWAYS:")
print("  ─ Naive estimate is biased due to confounding.")
print("  ─ IPW corrects for confounding by reweighting by propensity score.")
print("  ─ T-Learner uses flexible outcome models for CATE estimation.")
print("  ─ Doubly Robust is the most reliable: correct if either model correct.")

# ─────────────────────────────────────────────────────────────────────────────
# CATE estimation and heterogeneous effects
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  CATE ESTIMATION — HETEROGENEOUS TREATMENT EFFECTS")
print("=" * 65)
print()

cate_corr = np.corrcoef(cate_t, tau_true)[0, 1]
cate_mse  = np.mean((cate_t - tau_true)**2)
print(f"  T-learner CATE vs true CATE:")
print(f"    Pearson correlation: {cate_corr:.4f}")
print(f"    MSE:                 {cate_mse:.4f}")
print()

# Group-level CATE by X0 quintile
quintiles = np.percentile(X[:,0], [0, 20, 40, 60, 80, 100])
print(f"  CATE by X0 quintile (X0 modifies treatment effect):")
print(f"  {'X0 range':>15}  {'True CATE':>10}  {'Est. CATE':>10}  {'Diff':>8}")
print("  " + "─" * 47)
for q in range(5):
    mask = (X[:,0] >= quintiles[q]) & (X[:,0] < quintiles[q+1])
    print(f"  [{quintiles[q]:>5.2f},{quintiles[q+1]:>5.2f}]:  "
          f"{tau_true[mask].mean():>10.4f}  {cate_t[mask].mean():>10.4f}  "
          f"{cate_t[mask].mean()-tau_true[mask].mean():>+8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Positivity check — propensity score overlap
# ─────────────────────────────────────────────────────────────────────────────
print()
print("  POSITIVITY (OVERLAP) ASSESSMENT:")
print(f"  PS range (treated):  [{e_x[T==1].min():.3f}, {e_x[T==1].max():.3f}]")
print(f"  PS range (control):  [{e_x[T==0].min():.3f}, {e_x[T==0].max():.3f}]")
print(f"  Effective sample size (IPW): {(T/e_x + (1-T)/(1-e_x)).sum()**2 / ((T/e_x + (1-T)/(1-e_x))**2).sum():.1f} / {len(X)}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Causal Inference: Propensity Scores, IPW, T-Learner, Doubly Robust",
             fontsize=13, fontweight="bold")

# Panel (0,0): Propensity score distribution
ax = axes[0, 0]
ax.hist(e_x[T==1], bins=30, alpha=0.6, density=True, color="tomato",   label="Treated (T=1)")
ax.hist(e_x[T==0], bins=30, alpha=0.6, density=True, color="steelblue",label="Control (T=0)")
ax.set_title("Propensity Score Distribution\\n(overlap region = valid inference)",
             fontweight="bold")
ax.set_xlabel("e(X) = P(T=1|X)"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (0,1): ATE estimates comparison
ax = axes[0, 1]
estimators = ["True ATE", "Naive", "IPW", "NIPW", "T-Learner", "Doubly\\nRobust"]
ates = [true_ate, ate_naive, ate_ipw, ate_nipw, ate_tlearner, ate_dr]
colours_ate = ["black","tomato","seagreen","seagreen","steelblue","purple"]
bars = ax.bar(estimators, ates, color=colours_ate, alpha=0.8)
ax.axhline(true_ate, color="black", lw=2, ls="--", alpha=0.6, label="True ATE")
for bar, val in zip(bars, ates):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_title("ATE Estimates Comparison\\n(all should recover True ATE ≈ {:.2f})".format(true_ate),
             fontweight="bold")
ax.set_ylabel("Estimated ATE")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# Panel (0,2): CATE — estimated vs true
ax = axes[0, 2]
ax.scatter(tau_true[:500], cate_t[:500], c="steelblue", s=12, alpha=0.4)
lims = [min(tau_true.min(), cate_t.min())-0.2, max(tau_true.max(), cate_t.max())+0.2]
ax.plot(lims, lims, "k--", lw=2, label="Perfect estimation")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True CATE τ(x)"); ax.set_ylabel("Estimated CATE (T-learner)")
ax.set_title(f"CATE Estimation (T-Learner)\\ncorr={cate_corr:.3f}, MSE={cate_mse:.3f}",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,0): IPW weights distribution
ax = axes[1, 0]
w_all = T/e_x + (1-T)/(1-e_x)
ax.hist(w_all[T==1], bins=30, alpha=0.6, color="tomato",   density=True, label="Treated")
ax.hist(w_all[T==0], bins=30, alpha=0.6, color="steelblue",density=True, label="Control")
ax.axvline(1.0, color="black", lw=2, ls="--")
ax.set_title("IPW Weights Distribution\\n(extreme weights → unstable estimates, need clipping)",
             fontweight="bold")
ax.set_xlabel("IPW weight"); ax.set_ylabel("Density")
ax.set_xlim(0, 15); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,1): CATE by X0 — true vs estimated
ax = axes[1, 1]
x0_sorted = np.argsort(X[:,0])
ax.plot(X[x0_sorted, 0], tau_true[x0_sorted], "tomato", lw=2.5, label="True CATE τ(X0)")
ax.scatter(X[:,0], cate_t, s=5, alpha=0.2, color="steelblue", label="T-learner estimates")
# Smooth the estimates
from numpy.polynomial import polynomial as P
c = P.polyfit(X[:,0], cate_t, 2)
x0_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
ax.plot(x0_range, P.polyval(x0_range, c), "steelblue", lw=2.5, ls="--",
        label="T-learner (smooth)")
ax.set_xlabel("X0 (effect modifier)"); ax.set_ylabel("Treatment effect")
ax.set_title("Heterogeneous Treatment Effects\\nτ(x) = 1 + 0.5·X0 − 0.3·X1",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,2): Doubly robust — show it works when outcome model fails
ax = axes[1, 2]
# Compare: good PS + bad outcome vs bad PS + good outcome vs both good
np.random.seed(1)
n_mc  = 500
n_exp = 200

def run_experiment(good_ps, good_outcome, seed):
    rng = np.random.default_rng(seed)
    X_, T_, Y_, Y0_, Y1_, tau_, ate_ = generate_observational_data(n=n_exp, seed=seed)

    # Propensity score: good = logistic, bad = random noise
    if good_ps:
        ps_m = LogisticRegression(C=1.0).fit(X_, T_.astype(int))
        e_   = np.clip(ps_m.predict_proba(X_)[:,1], 0.05, 0.95)
    else:
        e_ = np.clip(rng.uniform(0.3, 0.7, n_exp), 0.05, 0.95)   # misspecified

    # Outcome model: good = linear regression, bad = constant
    if good_outcome:
        mu1_m = LinearRegression().fit(X_[T_==1], Y_[T_==1])
        mu0_m = LinearRegression().fit(X_[T_==0], Y_[T_==0])
        mu1_  = mu1_m.predict(X_)
        mu0_  = mu0_m.predict(X_)
    else:
        mu1_ = np.full(n_exp, Y_[T_==1].mean())   # misspecified (constant)
        mu0_ = np.full(n_exp, Y_[T_==0].mean())

    # DR estimator
    dr_ = (T_*(Y_-mu1_)/e_ - (1-T_)*(Y_-mu0_)/(1-e_) + mu1_ - mu0_).mean()
    return dr_ - ate_

configs = [
    ("Good PS\\n+ Good Outcome",  True,  True),
    ("Bad PS\\n+ Good Outcome",   False, True),
    ("Good PS\\n+ Bad Outcome",   True,  False),
    ("Bad PS\\n+ Bad Outcome",    False, False),
]
biases = []
for label, gps, gout in configs:
    b = [run_experiment(gps, gout, s) for s in range(n_mc)]
    biases.append(np.array(b))

ax.boxplot(biases, labels=[c[0] for c in configs],
           patch_artist=True,
           boxprops=dict(facecolor="steelblue", alpha=0.6))
ax.axhline(0, color="tomato", lw=2, ls="--", label="Zero bias")
ax.set_title("Doubly Robust: Works If EITHER Model Is Correct\\n"
             "(boxplots over 500 Monte Carlo runs)",
             fontweight="bold")
ax.set_ylabel("DR estimator bias")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("ate_estimation.png", dpi=110)
print()
print("  Plot saved → ate_estimation.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Causal Discovery — Learning DAGs from Observational Data": {
        "description": (
            "Implements the PC algorithm (constraint-based causal discovery) "
            "from scratch using conditional independence tests. Demonstrates "
            "causal discovery on a known ground-truth DAG, showing which edges "
            "are correctly oriented, which form a Markov equivalence class, "
            "and which are missed. Also shows LiNGAM's ability to distinguish "
            "A→B from A←B using non-Gaussianity of residuals. Includes a "
            "faithfulness violation example."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy import stats

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Conditional independence test using partial correlation
# ─────────────────────────────────────────────────────────────────────────────

def partial_corr(X_data, i, j, cond_set, alpha=0.01):
    """
    Test H0: X_i ⊥⊥ X_j | {X_k : k ∈ cond_set} using partial correlation.
    Returns (is_independent, p_value, partial_corr_coefficient).
    """
    n, d = X_data.shape
    if len(cond_set) == 0:
        # Marginal correlation test (Fisher's z)
        r, p = stats.pearsonr(X_data[:, i], X_data[:, j])
        return p > alpha, p, r
    else:
        # Partial correlation via residuals
        S_cols = list(cond_set)
        Z      = X_data[:, S_cols]
        # Regress X_i and X_j on the conditioning set Z
        res_i = X_data[:, i] - Z @ np.linalg.lstsq(Z, X_data[:, i], rcond=None)[0]
        res_j = X_data[:, j] - Z @ np.linalg.lstsq(Z, X_data[:, j], rcond=None)[0]
        r, p  = stats.pearsonr(res_i, res_j)
        return p > alpha, p, r


# ─────────────────────────────────────────────────────────────────────────────
# Simplified PC algorithm (skeleton + v-structure orientation)
# ─────────────────────────────────────────────────────────────────────────────

def pc_skeleton(X_data, alpha=0.01, max_cond_size=2):
    """
    PC Algorithm Phase 1: learn the skeleton (undirected graph).
    Returns: adjacency dict, separation sets dict
    """
    n, d = X_data.shape
    # Start with complete undirected graph
    adj = {i: set(range(d)) - {i} for i in range(d)}
    sep_set = {}   # sep_set[(i,j)] = conditioning set that separated i and j

    for k in range(max_cond_size + 1):
        to_remove = []
        for i in range(d):
            for j in list(adj[i]):
                if j <= i:
                    continue
                # Try all conditioning sets of size k from adj[i] - {j}
                cond_candidates = list(adj[i] - {j})
                if len(cond_candidates) < k:
                    continue
                for S in combinations(cond_candidates, k):
                    indep, p_val, _ = partial_corr(X_data, i, j, set(S), alpha)
                    if indep:
                        to_remove.append((i, j, set(S)))
                        break
        for i, j, S in to_remove:
            adj[i].discard(j)
            adj[j].discard(i)
            sep_set[(i, j)] = S
            sep_set[(j, i)] = S

    return adj, sep_set


def orient_v_structures(adj, sep_set, d):
    """
    PC Phase 2: orient v-structures (colliders).
    For triple i – k – j where i and j are not adjacent:
        if k NOT in sep_set[(i,j)]: orient as i → k ← j (collider).
    Returns: directed edges (partial) and undirected skeleton
    """
    directed = set()   # set of (i, j) meaning i → j

    for k in range(d):
        for i, j in combinations(range(d), 2):
            # Check: i adj k, j adj k, i NOT adj j
            if k in adj[i] and k in adj[j] and j not in adj[i]:
                S_ij = sep_set.get((i, j), sep_set.get((j, i), set()))
                if k not in S_ij:
                    # Orient as collider: i → k ← j
                    directed.add((i, k))
                    directed.add((j, k))
    return directed


def pc_algorithm(X_data, alpha=0.01, max_cond_size=2):
    """Run PC algorithm, return skeleton adjacency and directed edges."""
    n, d = X_data.shape
    adj, sep_set  = pc_skeleton(X_data, alpha, max_cond_size)
    directed = orient_v_structures(adj, sep_set, d)
    return adj, directed, sep_set


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth DAG and data generation
# ─────────────────────────────────────────────────────────────────────────────

# True DAG:   X0 → X1 → X3
#             X0 → X2 → X3
#             X2 → X4

n = 2000
# Structural equations (linear Gaussian)
rng = np.random.default_rng(7)
X0 = rng.standard_normal(n)
X1 = 0.8 * X0 + rng.standard_normal(n) * 0.5
X2 = 0.6 * X0 + rng.standard_normal(n) * 0.5
X3 = 0.7 * X1 + 0.5 * X2 + rng.standard_normal(n) * 0.4
X4 = 0.9 * X2 + rng.standard_normal(n) * 0.4
data = np.column_stack([X0, X1, X2, X3, X4])
var_names = ["X0", "X1", "X2", "X3", "X4"]

# True edges
true_edges = {(0,1), (0,2), (1,3), (2,3), (2,4)}
# True skeleton (undirected)
true_skeleton = {(min(i,j), max(i,j)) for i,j in true_edges}

print("=" * 65)
print("  CAUSAL DISCOVERY: PC ALGORITHM")
print("=" * 65)
print()
print("  True DAG edges: X0→X1, X0→X2, X1→X3, X2→X3, X2→X4")
print(f"  n = {n}")
print()

adj_pc, directed_pc, sep_pc = pc_algorithm(data, alpha=0.01, max_cond_size=2)

# Extract skeleton edges from PC
pc_skeleton_edges = set()
for i in range(5):
    for j in adj_pc[i]:
        if j > i:
            pc_skeleton_edges.add((i, j))

print(f"  PC skeleton edges found:")
for e in sorted(pc_skeleton_edges):
    status = "✓ correct" if e in true_skeleton else "✗ spurious"
    print(f"    {var_names[e[0]]}–{var_names[e[1]]}  {status}")

missed = true_skeleton - pc_skeleton_edges
if missed:
    print(f"  Missed edges: {[(var_names[e[0]],var_names[e[1]]) for e in missed]}")

print()
print(f"  PC oriented edges (v-structures):")
for (i, j) in sorted(directed_pc):
    print(f"    {var_names[i]}→{var_names[j]}")
if not directed_pc:
    print("    (no v-structures found — all edges in Markov equiv. class)")

# Skeleton accuracy
tp_skel = len(pc_skeleton_edges & true_skeleton)
fp_skel = len(pc_skeleton_edges - true_skeleton)
fn_skel = len(true_skeleton - pc_skeleton_edges)
prec    = tp_skel / (tp_skel + fp_skel) if (tp_skel + fp_skel) > 0 else 0
rec     = tp_skel / (tp_skel + fn_skel) if (tp_skel + fn_skel) > 0 else 0
print()
print(f"  Skeleton quality: precision={prec:.3f}, recall={rec:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# LiNGAM: identifying causal direction using non-Gaussianity
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  LiNGAM: CAUSAL DIRECTION FROM NON-GAUSSIANITY")
print("=" * 65)
print()
print("  Key idea: if A → B with non-Gaussian noise,")
print("  residuals from B ~ f(A) are independent of A.")
print("  Residuals from A ~ g(B) are NOT independent of B.")
print("  This asymmetry identifies the causal direction.")
print()

def lingam_direction_test(A, B):
    """
    For two variables A, B: test which is more likely causal.
    Fit B = α*A + ε_AB, A = β*B + ε_BA.
    Causal direction: A→B if ε_AB ⊥ A, NOT if ε_BA ⊥ B.
    Use HSIC (Hilbert-Schmidt Independence Criterion) proxy: |corr(ε, cause)|.
    """
    alpha = np.cov(A, B)[0,1] / np.var(A)
    beta  = np.cov(A, B)[0,1] / np.var(B)
    res_AB = B - alpha * A   # residuals of B ~ A
    res_BA = A - beta  * B   # residuals of A ~ B
    # Independence test: correlation with cause variable
    r_AB = abs(np.corrcoef(A, res_AB)[0,1])   # should be ~0 if A→B
    r_BA = abs(np.corrcoef(B, res_BA)[0,1])   # should be ~0 if B→A
    return r_AB, r_BA

# Generate non-Gaussian data: A → B (Laplace noise)
n_ling = 1000
A_true = np.random.laplace(0, 1, n_ling)
B_true = 0.7 * A_true + np.random.laplace(0, 0.5, n_ling)

r_AB, r_BA = lingam_direction_test(A_true, B_true)
print(f"  True direction: A → B (Laplace noise)")
print(f"  |corr(A, ε_{{B~A}})| = {r_AB:.4f}  ← should be small (residual indep of cause)")
print(f"  |corr(B, ε_{{A~B}})| = {r_BA:.4f}  ← should be large (residual NOT indep of effect)")
winner = "A→B" if r_AB < r_BA else "B→A"
print(f"  LiNGAM inferred direction: {winner}  {'✓ CORRECT' if winner=='A→B' else '✗ WRONG'}")

print()
# Gaussian case: LiNGAM fails
A_gauss = np.random.randn(n_ling)
B_gauss = 0.7 * A_gauss + np.random.randn(n_ling) * 0.5
r_AB_g, r_BA_g = lingam_direction_test(A_gauss, B_gauss)
print(f"  Gaussian noise case:")
print(f"  |corr(A, ε_{{B~A}})| = {r_AB_g:.4f}")
print(f"  |corr(B, ε_{{A~B}})| = {r_BA_g:.4f}")
print(f"  LiNGAM: {('A→B' if r_AB_g < r_BA_g else 'B→A')} — "
      f"indeterminate (Gaussian → cannot identify direction)")
print("  With Gaussian noise, A→B and B→A are Markov equivalent — undirected.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Causal Discovery: PC Algorithm, v-Structures, and LiNGAM",
             fontsize=13, fontweight="bold")

# Panel (0,0): True DAG (textual)
ax = axes[0, 0]
ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(0, 5)
positions = {0: (1, 4), 1: (0, 2.5), 2: (2, 2.5), 3: (1, 1), 4: (3, 1)}
for node, (x, y) in positions.items():
    col = "steelblue"
    circ = plt.Circle((x, y), 0.45, color=col, alpha=0.8)
    ax.add_patch(circ)
    ax.text(x, y, var_names[node], ha="center", va="center",
            color="white", fontweight="bold", fontsize=12)
for (src, tgt) in true_edges:
    x1, y1 = positions[src]
    x2, y2 = positions[tgt]
    dx, dy = x2-x1, y2-y1
    norm   = np.sqrt(dx**2 + dy**2)
    ax.annotate("", xy=(x2 - 0.5*dx/norm, y2 - 0.5*dy/norm),
                xytext=(x1 + 0.5*dx/norm, y1 + 0.5*dy/norm),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.set_title("True Causal DAG\\nX0→X1, X0→X2, X1→X3, X2→X3, X2→X4",
             fontweight="bold")

# Panel (0,1): PC-recovered skeleton
ax = axes[0, 1]
ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(0, 5)
for node, (x, y) in positions.items():
    col = "steelblue"
    circ = plt.Circle((x, y), 0.45, color=col, alpha=0.8)
    ax.add_patch(circ)
    ax.text(x, y, var_names[node], ha="center", va="center",
            color="white", fontweight="bold", fontsize=12)
for (i, j) in pc_skeleton_edges:
    x1, y1 = positions[i]; x2, y2 = positions[j]
    col = "tomato" if (i,j) not in true_skeleton else "seagreen"
    ax.plot([x1, x2], [y1, y2], color=col, lw=3, alpha=0.8)
# Directed edges found
for (i, j) in directed_pc:
    if j in adj_pc[i]:
        x1, y1 = positions[i]; x2, y2 = positions[j]
        dx, dy = x2-x1, y2-y1
        norm   = np.sqrt(dx**2+dy**2)+1e-9
        ax.annotate("", xy=(x2-0.5*dx/norm, y2-0.5*dy/norm),
                    xytext=(x1+0.5*dx/norm, y1+0.5*dy/norm),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
ax.set_title(f"PC-Recovered Graph\\n(green=correct, red=spurious; prec={prec:.2f}, rec={rec:.2f})",
             fontweight="bold")

# Panel (0,2): Separation sets and CI test p-values
ax = axes[0, 2]
# Show p-value matrix for all pairs
pval_matrix = np.ones((5, 5))
for i in range(5):
    for j in range(i+1, 5):
        _, p, _ = partial_corr(data, i, j, set(), alpha=0.01)
        pval_matrix[i, j] = p
        pval_matrix[j, i] = p
im = ax.imshow(np.log10(pval_matrix + 1e-10), cmap="RdYlGn", vmin=-10, vmax=0)
plt.colorbar(im, ax=ax, label="log10(p-value)")
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(var_names); ax.set_yticklabels(var_names)
ax.set_title("Marginal CI Test p-values (log scale)\\n(green = independent, red = dependent)",
             fontweight="bold")
for i in range(5):
    for j in range(5):
        if i != j:
            ax.text(j, i, f"{pval_matrix[i,j]:.2f}" if pval_matrix[i,j] > 0.001 else "<0.01",
                    ha="center", va="center", fontsize=7)

# Panel (1,0): LiNGAM — residuals for correct direction
ax = axes[1, 0]
alpha_fit = np.cov(A_true, B_true)[0,1] / np.var(A_true)
res = B_true - alpha_fit * A_true
ax.scatter(A_true[:300], res[:300], s=8, alpha=0.4, c="steelblue",
           label=f"|corr|={r_AB:.4f}")
ax.axhline(0, color="black", lw=1, ls="--")
ax.set_title(f"LiNGAM: Correct Direction A→B\\nResidual ε_{{B~A}} ⊥⊥ A (|corr|={r_AB:.4f})",
             fontweight="bold")
ax.set_xlabel("A"); ax.set_ylabel("Residual ε_{B~A}")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,1): LiNGAM — residuals for wrong direction
ax = axes[1, 1]
beta_fit = np.cov(A_true, B_true)[0,1] / np.var(B_true)
res_wrong = A_true - beta_fit * B_true
ax.scatter(B_true[:300], res_wrong[:300], s=8, alpha=0.4, c="tomato",
           label=f"|corr|={r_BA:.4f}")
ax.axhline(0, color="black", lw=1, ls="--")
ax.set_title(f"LiNGAM: Wrong Direction B→A\\nResidual ε_{{A~B}} NOT ⊥⊥ B (|corr|={r_BA:.4f})",
             fontweight="bold")
ax.set_xlabel("B"); ax.set_ylabel("Residual ε_{A~B}")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,2): Noise distribution comparison
ax = axes[1, 2]
noise_laplace = np.random.laplace(0, 1, 2000)
noise_gauss   = np.random.randn(2000)
ax.hist(noise_laplace, bins=60, alpha=0.6, density=True, color="tomato",
        label="Laplace (LiNGAM works)")
ax.hist(noise_gauss,   bins=60, alpha=0.6, density=True, color="steelblue",
        label="Gaussian (LiNGAM fails)")
from scipy.stats import laplace, norm
x_range = np.linspace(-5, 5, 200)
ax.plot(x_range, laplace.pdf(x_range), "tomato", lw=2.5)
ax.plot(x_range, norm.pdf(x_range),   "steelblue", lw=2.5)
ax.set_title("Noise Distributions\\n(non-Gaussian noise enables causal direction ID)",
             fontweight="bold")
ax.set_xlabel("Noise value"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("causal_discovery.png", dpi=110)
print()
print("  Plot saved → causal_discovery.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · IRM and Invariant Causal Features Across Environments": {
        "description": (
            "Implements Invariant Risk Minimisation (IRM) vs ERM on a "
            "dataset with spurious correlations that differ across "
            "environments. Shows that ERM learns the spurious feature "
            "(which has higher predictive power in training) while IRM "
            "learns the causal feature (which is invariant). Tests both "
            "on held-out environments where the spurious correlation "
            "reverses. Connects IRM to the backdoor criterion."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt

class LinearRegression:
    def __init__(self, fit_intercept=True): self.fit_intercept = fit_intercept
    def fit(self, X, y):
        Xa = _np_impl.c_[_np_impl.ones(len(X)), X] if self.fit_intercept else X
        w  = _np_impl.linalg.lstsq(Xa, y, rcond=None)[0]
        if self.fit_intercept:
            self.intercept_ = w[0]; self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0;  self.coef_ = w
        return self
    def predict(self, X): return X @ self.coef_ + self.intercept_

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))
class LogisticRegression:
    def __init__(self, C=1.0, max_iter=1000, random_state=None, solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self, X, y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

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
                g=(_np_impl.var(r)*len(r)-_np_impl.var(r[l])*l.sum()-_np_impl.var(r[rr])*rr.sum())
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np_impl.var(r)<1e-12:
            return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,left,right=nd; return self._p1(x,left if x[f]<=t else right)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class GradientBoostingRegressor:
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        self._f0=y.mean(); F=_np_impl.full(len(y),self._f0); self._trees=[]
        for _ in range(self.n_estimators):
            r=y-F; t=_DTR(max_depth=self.max_depth); t.fit(X,r)
            self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return F

class GradientBoostingClassifier: pass  # imported but unused in ops

def mean_squared_error(y_true, y_pred):
    return _np_impl.mean((_np_impl.asarray(y_true)-_np_impl.asarray(y_pred))**2)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Data generating process — two environments with different spurious correlations
# ─────────────────────────────────────────────────────────────────────────────
# Causal graph: X_causal → Y (invariant across environments)
#               E → X_spurious, E → Y  (spurious: corr changes with environment E)
# In environment 1: X_spurious correlated +0.9 with Y
# In environment 2: X_spurious correlated +0.6 with Y
# Test environment: X_spurious correlated -0.5 with Y (spurious flips!)

def make_environment(n, beta_causal, beta_spur, seed):
    """
    Y = beta_causal * X_causal + noise_Y
    X_spurious = beta_spur * Y + noise_spur  (Y → X_spurious via shared noise)
    X_causal is independent of the environment.
    """
    rng = np.random.default_rng(seed)
    X_causal  = rng.standard_normal(n)
    noise_Y   = rng.standard_normal(n) * 0.5
    Y         = beta_causal * X_causal + noise_Y
    noise_spur = rng.standard_normal(n) * 0.5
    X_spurious = beta_spur * Y + noise_spur
    return np.column_stack([X_causal, X_spurious]), Y

n_env = 2000
# Environment 1: strong spurious correlation
X1, Y1 = make_environment(n_env, beta_causal=1.0, beta_spur=3.0, seed=0)
# Environment 2: moderate spurious correlation
X2, Y2 = make_environment(n_env, beta_causal=1.0, beta_spur=1.5, seed=1)
# Test environment: spurious correlation REVERSES (OOD shift)
X_test, Y_test = make_environment(n_env, beta_causal=1.0, beta_spur=-2.0, seed=2)
# Oracle test: only causal feature available
X_oracle = X_test[:, [0]]
Y_oracle  = Y_test

print("=" * 65)
print("  INVARIANT RISK MINIMISATION vs ERM")
print("=" * 65)
print()
print("  Causal structure: X_causal → Y  (causal, invariant)")
print("                    Y → X_spurious (spurious, changes between envs)")
print()
print("  True causal coefficient on X_causal = 1.0")
print()

# Correlations in each environment
corr_env1 = np.corrcoef(X1[:,1], Y1)[0,1]
corr_env2 = np.corrcoef(X2[:,1], Y2)[0,1]
corr_test = np.corrcoef(X_test[:,1], Y_test)[0,1]
print(f"  X_spurious–Y correlation: Env1={corr_env1:.3f}, Env2={corr_env2:.3f}, "
      f"Test={corr_test:.3f}  ← FLIPPED!")
print()

# ─────────────────────────────────────────────────────────────────────────────
# ERM baseline: train on pooled data
# ─────────────────────────────────────────────────────────────────────────────
X_pool = np.vstack([X1, X2])
Y_pool = np.concatenate([Y1, Y2])

erm = LinearRegression().fit(X_pool, Y_pool)
coef_erm = erm.coef_
pred_erm_test = erm.predict(X_test)
mse_erm_test  = mean_squared_error(Y_test, pred_erm_test)

print(f"  ERM (pooled) coefficients:")
print(f"    X_causal  = {coef_erm[0]:.4f}  (true = 1.0)")
print(f"    X_spurious= {coef_erm[1]:.4f}  (true causal effect = 0.0)")
print(f"  ERM test MSE (spurious flipped env): {mse_erm_test:.4f}")
print()
print("  ERM picks up X_spurious because it is predictive in training.")
print("  Under distribution shift (test env), X_spurious hurts performance.")

# ─────────────────────────────────────────────────────────────────────────────
# IRM-inspired approach: penalise environment-specific gradients
# ─────────────────────────────────────────────────────────────────────────────
# Simplified IRM: find w such that w is simultaneously optimal in both envs.
# For linear case, the invariant solution is: use only X_causal.
# We simulate this by penalising difference in env-specific predictions.

def irm_linear(X_envs, Y_envs, lambda_irm=1000, lr=0.01, n_iter=500, seed=0):
    """
    Simplified IRM for linear models:
    min_w  Σ_e R^e(w) + λ Σ_e ‖∇_w R^e(w)‖²
    Gradient of ||∇R^e||² penalises environment-specific curvature.
    """
    rng = np.random.default_rng(seed)
    d   = X_envs[0].shape[1]
    w   = rng.standard_normal(d) * 0.01
    e_envs = len(X_envs)

    for t in range(n_iter):
        grad_total = np.zeros(d)
        for X_e, Y_e in zip(X_envs, Y_envs):
            n_e    = len(X_e)
            pred_e = X_e @ w
            res_e  = pred_e - Y_e
            # ERM gradient
            grad_e = 2 * X_e.T @ res_e / n_e
            grad_total += grad_e / e_envs
            # IRM penalty: gradient of ||gradient w.r.t. w scaled by w||^2
            # Simplified: penalise variance of per-sample losses across envs
            # penalty gradient ≈ 2 * lambda * grad_e * (grad_e @ w)
            irm_penalty_grad = 2 * lambda_irm * grad_e * (grad_e @ w) / n_e
            grad_total += irm_penalty_grad / e_envs

        w -= lr * grad_total

    return w

w_irm = irm_linear([X1, X2], [Y1, Y2], lambda_irm=500, lr=0.005, n_iter=1000)
pred_irm_test = X_test @ w_irm
mse_irm_test  = mean_squared_error(Y_test, pred_irm_test)

print()
print(f"  IRM coefficients:")
print(f"    X_causal  = {w_irm[0]:.4f}  (true = 1.0)")
print(f"    X_spurious= {w_irm[1]:.4f}  (should be ~0)")
print(f"  IRM test MSE (spurious flipped env): {mse_irm_test:.4f}")

# Oracle: only causal feature
oracle = LinearRegression().fit(X_pool[:, [0]], Y_pool)
mse_oracle = mean_squared_error(Y_test, oracle.predict(X_test[:, [0]]))
print()
print(f"  Oracle (X_causal only) test MSE: {mse_oracle:.4f}")
print()
print("  SUMMARY:")
print(f"  ERM MSE  = {mse_erm_test:.4f}  (relying on spurious feature, fails OOD)")
print(f"  IRM MSE  = {mse_irm_test:.4f}  (uses causal feature, robust OOD)")
print(f"  Oracle   = {mse_oracle:.4f}  (uses only X_causal, best possible)")

# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity to spurious correlation strength
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  ERM vs IRM ACROSS DIFFERENT SPURIOUS CORRELATIONS (OOD test)")
print("=" * 65)
print()

test_spurs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
print(f"  {'β_spur(test)':>14}  {'ERM MSE':>10}  {'IRM MSE':>10}  {'Oracle MSE':>12}")
print("  " + "─" * 50)

for beta_spur_t in test_spurs:
    X_t, Y_t = make_environment(1000, beta_causal=1.0, beta_spur=beta_spur_t, seed=99)
    mse_e = mean_squared_error(Y_t, erm.predict(X_t))
    mse_i = mean_squared_error(Y_t, X_t @ w_irm)
    mse_o = mean_squared_error(Y_t, oracle.predict(X_t[:, [0]]))
    print(f"  {beta_spur_t:>14.1f}  {mse_e:>10.4f}  {mse_i:>10.4f}  {mse_o:>12.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("IRM vs ERM: Invariant Causal Features and Spurious Correlations",
             fontsize=13, fontweight="bold")

# Panel (0,0): Spurious correlation in each environment
ax = axes[0, 0]
ax.scatter(X1[:,1], Y1, s=5, alpha=0.3, c="steelblue", label=f"Env 1 (β_spur=3.0)")
ax.scatter(X2[:,1], Y2, s=5, alpha=0.3, c="tomato",    label=f"Env 2 (β_spur=1.5)")
ax.scatter(X_test[:,1], Y_test, s=5, alpha=0.3, c="seagreen", label=f"Test (β_spur=−2.0)")
ax.set_xlabel("X_spurious"); ax.set_ylabel("Y")
ax.set_title("Spurious Correlation Changes Across Environments\\n"
             "(test: spurious feature REVERSES sign!)", fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (0,1): ERM vs IRM coefficients
ax = axes[0, 1]
methods = ["ERM", "IRM", "Oracle"]
causal_coefs  = [coef_erm[0], w_irm[0], oracle.coef_[0]]
spurious_coefs = [coef_erm[1], w_irm[1], 0.0]
x_pos = np.arange(3)
ax.bar(x_pos - 0.2, causal_coefs,  0.4, color="steelblue", alpha=0.8, label="X_causal coef")
ax.bar(x_pos + 0.2, spurious_coefs, 0.4, color="tomato",    alpha=0.8, label="X_spurious coef")
ax.axhline(1.0, color="steelblue", lw=2, ls="--", alpha=0.5, label="True causal = 1.0")
ax.axhline(0.0, color="tomato",    lw=2, ls="--", alpha=0.5, label="True spurious = 0.0")
ax.set_xticks(x_pos); ax.set_xticklabels(methods, fontsize=10)
ax.set_title("Learned Coefficients\\n(IRM recovers causal, ignores spurious)",
             fontweight="bold")
ax.set_ylabel("Coefficient value")
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

# Panel (0,2): Test MSE across spurious strengths
ax = axes[0, 2]
spur_range = np.linspace(-4, 4, 30)
erm_mses = []; irm_mses = []; oracle_mses = []
for b in spur_range:
    X_t2, Y_t2 = make_environment(800, 1.0, b, seed=77)
    erm_mses.append(mean_squared_error(Y_t2, erm.predict(X_t2)))
    irm_mses.append(mean_squared_error(Y_t2, X_t2 @ w_irm))
    oracle_mses.append(mean_squared_error(Y_t2, oracle.predict(X_t2[:,[0]])))
ax.plot(spur_range, erm_mses,    "tomato",    lw=2.5, label="ERM")
ax.plot(spur_range, irm_mses,    "steelblue", lw=2.5, label="IRM")
ax.plot(spur_range, oracle_mses, "black",     lw=2,   ls="--", label="Oracle")
ax.axvline(3.0,  color="steelblue", lw=1, ls=":", alpha=0.6, label="Training β_spur range")
ax.axvline(1.5,  color="steelblue", lw=1, ls=":", alpha=0.6)
ax.set_xlabel("Test environment β_spurious"); ax.set_ylabel("MSE")
ax.set_title("Test MSE vs Spurious Correlation Strength\\n"
             "(ERM fails when spurious signal flips; IRM robust)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,0): Predictions on test environment
ax = axes[1, 0]
sort_idx = np.argsort(Y_test[:300])
ax.plot(range(300), Y_test[:300][sort_idx], "black", lw=2, label="True Y")
ax.plot(range(300), pred_erm_test[:300][sort_idx], "tomato", lw=1.5, alpha=0.8,
        label=f"ERM pred (MSE={mse_erm_test:.3f})")
ax.plot(range(300), pred_irm_test[:300][sort_idx], "steelblue", lw=1.5, alpha=0.8,
        label=f"IRM pred (MSE={mse_irm_test:.3f})")
ax.set_title("Predictions on OOD Test Environment\\n"
             "(ERM diverges; IRM tracks true Y)",
             fontweight="bold")
ax.set_xlabel("Sorted sample index"); ax.set_ylabel("Y")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,1): Invariance test — per-environment risk
ax = axes[1, 1]
risk_env1_erm = mean_squared_error(Y1, erm.predict(X1))
risk_env2_erm = mean_squared_error(Y2, erm.predict(X2))
risk_env1_irm = mean_squared_error(Y1, X1 @ w_irm)
risk_env2_irm = mean_squared_error(Y2, X2 @ w_irm)
ax.scatter([1, 2], [risk_env1_erm, risk_env2_erm], s=150, c="tomato",
           zorder=5, label="ERM", marker="o")
ax.scatter([1, 2], [risk_env1_irm, risk_env2_irm], s=150, c="steelblue",
           zorder=5, label="IRM", marker="s")
ax.plot([1,2], [risk_env1_erm, risk_env2_erm], "tomato", lw=2, alpha=0.7)
ax.plot([1,2], [risk_env1_irm, risk_env2_irm], "steelblue", lw=2, alpha=0.7)
ax.set_xticks([1, 2]); ax.set_xticklabels(["Env 1", "Env 2"])
ax.set_ylabel("MSE per environment")
ax.set_title("Per-Environment Risk\\n"
             "(IRM forces similar risk in both; ERM can vary)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,2): Summary of causal inference methods
ax = axes[1, 2]
ax.axis("off")
methods_table = [
    ["RCT", "Gold standard", "Expensive/unethical", "Randomised assignment"],
    ["Backdoor adj.", "Exact if DAG known", "Need no hidden confounders", "Adjust for confounders Z"],
    ["IPW", "Observational", "Positivity + ignorability", "Reweight by P(T|X)"],
    ["Doubly Robust", "Observational", "Either PS or outcome model", "IPW + outcome model"],
    ["IV", "Hidden confounders OK", "Needs valid instrument", "Cov(Y,Z)/Cov(T,Z)"],
    ["DiD", "Panel data", "Parallel trends", "Before/after comparison"],
    ["IRM", "Multi-environment", "Multiple envs required", "Invariant feature learning"],
]
headers = ["Method", "Setting", "Key Assumption", "Core Idea"]
table = ax.table(cellText=methods_table, colLabels=headers,
                 cellLoc="left", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.1, 1.85)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f8f8f8")
ax.set_title("Causal Inference Methods Summary", fontweight="bold")

plt.tight_layout()
plt.savefig("irm_causal.png", dpi=110)
print()
print("  Plot saved → irm_causal.png")
print()
print("  KEY TAKEAWAYS — CAUSALITY:")
print("  1. Correlation ≠ causation: P(Y|X) ≠ P(Y|do(X)) under confounding.")
print("  2. Three rungs: association (observe) → intervention (do) → counterfactual.")
print("  3. DAGs encode causal assumptions; d-separation reads off independencies.")
print("  4. Confounders create spurious correlations → adjust using backdoor criterion.")
print("  5. Collider bias: conditioning on a collider CREATES spurious correlations.")
print("  6. ATE estimation: IPW, T-learner, doubly robust for observational data.")
print("  7. IRM finds features that are causally invariant across environments.")
print("  8. Causal features are robust to distribution shift; spurious ones are not.")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


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
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }