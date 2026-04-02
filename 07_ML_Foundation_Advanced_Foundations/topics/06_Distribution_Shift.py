"""
Distribution Shift & Robustness
================================

Covariate shift, label shift, concept drift, out-of-distribution
generalisation, and domain adaptation — the formal treatment of what
happens when the distribution at deployment differs from the distribution
at training, and how to detect, measure, and mitigate it.

"""

import textwrap
import re

TOPIC_NAME = "Distribution Shift & Robustness"
DISPLAY_NAME = "06 · Distribution Shift & Robustness"
ICON = "🌊"
SUBTITLE = "When the World Changes and Your Model Doesn't"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Core Problem: Train ≠ Deploy

Every ML guarantee — PAC bounds, cross-validation accuracy, test set
performance — assumes that training and test data are drawn from the
SAME distribution. Formally:

        Classical assumption:   P_train(X, Y) = P_test(X, Y)

This assumption is violated in almost every real deployment:
─ A spam classifier trained on 2020 emails faces 2024 email patterns.
─ A model trained on hospital A's patients is deployed at hospital B.
─ A self-driving model trained in California drives in Finland in winter.
─ A credit model trained in an economic boom is deployed during recession.

When the distribution shifts, all performance guarantees dissolve.
A model with 95% validation accuracy can perform at chance on deployment.

    Diagram 1 — The Training-Deployment Gap:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  TRAINING TIME:                                                 │
    │  Data ~ P_train(X, Y)                                           │
    │  Model learns f: X → Y optimised for P_train                    │
    │  Evaluation: validation accuracy = 0.94  ←── misleading!        │
    │                                                                 │
    │  DEPLOYMENT TIME:                                               │
    │  Data ~ P_deploy(X, Y)  ≠  P_train(X, Y)                        │
    │  Model f applied to P_deploy                                    │
    │  True accuracy = 0.61  ←── actual performance                   │
    │                                                                 │
    │  The gap (0.94 − 0.61 = 0.33) is the distribution shift cost.   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Why distribution shift is the central challenge of practical ML:
    ─ Train/val splits cannot detect it (same P_train for both).
    ─ Standard metrics (accuracy, F1) give no warning.
    ─ A model can be confidently wrong — high certainty, bad predictions.
    ─ Debugging is hard: the model worked perfectly in the lab.


──────────────────────────────────────────────────────────────────────────────
### A Taxonomy of Distribution Shift

Not all shifts are equal. Precise terminology allows precise solutions.

    Joint distribution:   P(X, Y) = P(X) · P(Y | X)
                                  = P(Y) · P(X | Y)

    ┌──────────────────────────────────────────────────────────────────┐
    │  TYPE                   WHAT CHANGES         WHAT STAYS SAME     │
    ├──────────────────────────────────────────────────────────────────┤
    │  Covariate shift        P(X)                 P(Y | X)            │
    │  Label shift            P(Y)                 P(X | Y)            │
    │  Concept drift          P(Y | X)             P(X)                │
    │  Dataset shift          P(X, Y)  fully       nothing             │
    │  Domain adaptation      P(X, Y)  between     task semantics      │
    │                         source/target        (transfer learning) │
    │  Subpopulation shift    P(X,Y|G) within-group nothing guaranteed │
    └──────────────────────────────────────────────────────────────────┘

**1. Covariate Shift — P(X) changes, P(Y|X) stays the same:**

    The mapping from features to labels is unchanged — if a doctor
    could still correctly diagnose from an X-ray, the rule holds.
    But the distribution of X-rays looks different at test time.

    Example:  Training data is 80% young patients (< 40 years).
              Test data is 60% elderly patients.
              P(Y|X) (diagnosis rule) unchanged.
              P(X) (age distribution) shifted.

    Why it hurts: the model over-optimises for young-patient features.
    Old-patient regions of X-space are poorly represented in training,
    so the model lacks confidence and accuracy there.

    Formal condition:   P_train(Y | X) = P_test(Y | X)
                        P_train(X)     ≠ P_test(X)

**2. Label Shift — P(Y) changes, P(X|Y) stays the same:**

    The prior probability of each class changes, but the appearance of
    each class (what it looks like) is unchanged.

    Example:  Training data: 10% fraud, 90% legitimate.
              Deployment: 30% fraud (new fraud wave).
              P(X | Y=fraud) — what fraudulent transactions look like — unchanged.
              P(Y) — the prior rate of fraud — has shifted.

    Formal condition:   P_train(X | Y) = P_test(X | Y)
                        P_train(Y)     ≠ P_test(Y)

    Correction: weight training examples by P_test(Y) / P_train(Y).
    These weights can be estimated from unlabelled target domain data.

**3. Concept Drift — P(Y|X) changes:**

    The very relationship between features and labels changes.
    This is the hardest shift to correct because the model's core
    assumption is violated — the same X now means a different Y.

    Example:  "Large" credit card transaction threshold in 2010 = $500.
              "Large" in 2024 = $5000 (inflation, lifestyle change).
              Same features → different labelling rule.

    Types of concept drift:
    ┌────────────────────────────────────────────────────────────────┐
    │  SUDDEN:    P(Y|X) changes abruptly at time t₀.                │
    │             Old model completely wrong after t₀.               │
    │  GRADUAL:   P(Y|X) transitions slowly over time.               │
    │             Old model degrades gradually.                      │
    │  RECURRING: P(Y|X) cycles (e.g., seasonal patterns).           │
    │             Old model from last year may still apply.          │
    │  INCREMENTAL: P(Y|X) drifts monotonically (inflation, etc.).   │
    └────────────────────────────────────────────────────────────────┘

    Diagram 2 — Types of Concept Drift Over Time:

    P(Y=1|x, t)
      │
    1 │       ___         ← sudden drift        ← recurring
      │      │   │
    0.5│─────   ──────────────────────────────────────────  ← gradual
      │                  ╲__________╱             ╱╲     ╱╲
    0 │
      └──────────────────────────────────────────────────── time t

**4. Subpopulation / Spurious Correlation Shift:**

    A model learns a spurious feature that correlates with Y in training
    but not in deployment. Classic example: cows detected by grass, not
    shape — model fails on cows in unusual environments.

    Formal:  P_train(Y | X_spurious) ≠ P_deploy(Y | X_spurious)
             because X_spurious is only correlated with Y in training.

    The model is right for the wrong reasons.


──────────────────────────────────────────────────────────────────────────────
### Importance Weighting — Correcting for Covariate Shift

Under covariate shift, the expected loss under P_test can be written as
a weighted expected loss under P_train:

    E_{P_test}[ℓ(f(x), y)]
    = ∫ ℓ(f(x), y) P_test(x, y) dx dy
    = ∫ ℓ(f(x), y) [P_test(x) / P_train(x)] P_train(x, y) dx dy
    = E_{P_train}[ w(x) · ℓ(f(x), y) ]

    where   w(x) = P_test(x) / P_train(x)   (importance weights)

    ┌──────────────────────────────────────────────────────────────────┐
    │  Covariate shift corrected estimator:                            │
    │                                                                  │
    │       min_f  Σᵢ w(xᵢ) · ℓ(f(xᵢ), yᵢ)                             │
    │                                                                  │
    │  Reweighting training examples by their likelihood ratio         │
    │  recovers an unbiased estimator of the test loss.                │
    └──────────────────────────────────────────────────────────────────┘

**Estimating Importance Weights:**

    Method 1 — Density ratio estimation:
        Train a classifier to distinguish train vs test samples.
        P(test | x) / P(train | x) ≈ P_test(x) / P_train(x).

        If classifier outputs p̂(test | x), then:
            w(x) = p̂(test | x) / p̂(train | x)
                 = p̂(test | x) / (1 − p̂(test | x))   (binary classifier)

    Method 2 — Kernel Mean Matching (KMM):
        Find weights w₁,...,wₙ such that the weighted training
        distribution matches the test distribution in a RKHS:

        min_w  ‖μ_test − Σᵢ wᵢ φ(xᵢ)‖²_Hₖ
        subject to: wᵢ ≥ 0,  (1/n) Σᵢ wᵢ = 1

        This is a quadratic program in the weights.

    Practical caveats:
    ┌──────────────────────────────────────────────────────────────────┐
    │  1.  Weights can be very large if P_test(x)/P_train(x) >> 1.     │
    │      Clip weights at some maximum (e.g., wᵢ ≤ 20) for stability. │
    │  2.  Requires unlabelled test data to estimate P_test(x).        │
    │  3.  If P_test has support outside P_train, correction fails.    │
    │      Cannot correct for regions with zero training density.      │
    │  4.  Effective sample size = (Σwᵢ)² / Σwᵢ² — can be << n.        │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Detecting Distribution Shift — Statistical Tests

Before correcting, we need to detect that a shift has occurred.

**Two-Sample Tests — testing if P_train = P_test:**

    Given samples X_train = {x₁,...,xₙ} ~ P_train
    and   X_test  = {x₁,...,xₘ} ~ P_test

    H₀: P_train = P_test
    H₁: P_train ≠ P_test

    MAXIMUM MEAN DISCREPANCY (MMD):

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  MMD²(P, Q) = ‖μ_P − μ_Q‖²_{Hₖ}                                  │
    │                                                                  │
    │  = E_{x,x'~P}[k(x,x')] − 2·E_{x~P,z~Q}[k(x,z)]                   │
    │  + E_{z,z'~Q}[k(z,z')]                                           │
    │                                                                  │
    │  Unbiased estimator:                                             │
    │  MMD²_u = (1/n²)ΣΣk(xᵢ,xⱼ) − (2/nm)ΣΣk(xᵢ,zⱼ)                     │
    │          + (1/m²)ΣΣk(zᵢ,zⱼ)                                       │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    MMD = 0 iff P = Q.  Large MMD → strong evidence of shift.
    Permutation test gives a p-value.

    KOLMOGOROV-SMIRNOV (KS) TEST  — for univariate distributions:
        KS statistic = sup_t |F_P(t) − F_Q(t)|
        Measures the maximum vertical gap between two CDFs.

    CLASSIFIER-BASED TEST:
        Train a classifier to distinguish train vs test samples.
        If AUC >> 0.5, the distributions are distinguishable → shift detected.
        AUC = 0.5 → cannot distinguish → no detectable shift.

**Feature-Level vs Prediction-Level Shift Detection:**

    ┌──────────────────────────────────────────────────────────────────┐
    │  FEATURE-LEVEL: test individual features for shift.              │
    │  → KS test on each feature, with multiple-testing correction.    │
    │  → Finds WHICH features are shifting.                            │
    │                                                                  │
    │  PREDICTION-LEVEL: monitor model output distribution.            │
    │  → Does the distribution of predicted scores shift?              │
    │  → Faster than feature-level but less informative.               │
    │                                                                  │
    │  PERFORMANCE MONITORING: track accuracy on labelled subsets.     │
    │  → Direct but requires labels in deployment — often unavailable. │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 3 — Shift Detection Pipeline:

    New data arrives
          │
          ▼
    Feature-level KS tests ──── any feature shifted? ─── NO → no alert
          │ YES
          ▼
    MMD test on joint distribution ──── statistically significant? ─── NO → monitor
          │ YES
          ▼
    Classify: covariate/label/concept drift?
          │
          ▼
    Apply correction (reweighting, recalibration, retrain)


──────────────────────────────────────────────────────────────────────────────
### Domain Adaptation — Learning Across Distributions

Domain adaptation is the general framework for adapting a model trained
on a source domain to perform well on a target domain.

    Setup:
        Source domain Ds: labelled data from P_source(X, Y)
        Target domain Dt: unlabelled (or sparsely labelled) data from P_target(X, Y)
        Goal: learn f that performs well on P_target using Ds and Dt.

    Taxonomy by label availability in target domain:

    ┌────────────────────────────────────────────────────────────────┐
    │  UNSUPERVISED DA:   no labels in target. Most common.          │
    │  SEMI-SUPERVISED DA: a few labels in target.                   │
    │  SUPERVISED DA:      many labels in target. (= fine-tuning)    │
    └────────────────────────────────────────────────────────────────┘

**Theoretical Bound (Ben-David et al., 2010):**

    The error on the target domain is bounded by:

    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  ε_T(f) ≤ ε_S(f) + d_H(S, T) + λ                                 │
    │                                                                  │
    │  where:                                                          │
    │  ε_S(f)      = source error (what we can measure)                │
    │  d_H(S, T)   = H-divergence between source and target            │
    │                (complexity of distinguishing the two domains)    │
    │  λ           = error of the ideal joint classifier on S ∪ T      │
    │                (irreducible if tasks are truly incompatible)     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    INTERPRETATION:  To minimise target error, we must:
    1. Have low source error ε_S(f)  — good performance on source.
    2. Minimise d_H(S,T)  — reduce distributional distance between
       source and target representations.

    Method 2 motivates DOMAIN-ADVERSARIAL training:
    Learn a feature extractor φ that minimises the divergence
    between φ(source) and φ(target) — make the representations
    indistinguishable between domains.

**Domain-Adversarial Neural Networks (DANN, Ganin et al., 2016):**

    Architecture:
    ─ Feature extractor G_f: X → Z   (shared representations)
    ─ Label predictor G_y: Z → Y     (classification head)
    ─ Domain discriminator G_d: Z → {source, target}

    Training objective (minimax):
        min_{G_f, G_y}  max_{G_d}   L_y(G_y(G_f(x)), y) − λ·L_d(G_d(G_f(x)), d)
                                    ↑ label classification   ↑ domain confusion

    The gradient reversal layer: multiply gradients by −λ during
    backprop through G_f from the domain discriminator.
    This forces G_f to extract domain-invariant features.

    Diagram 4 — DANN Architecture:

    Input x
      │
      ▼
    ┌─────────────────────────────┐
    │   Feature extractor G_f(x)  │
    └─────────────────────────────┘
           │                │
           │                │ (gradient reversal: ×−λ)
           ▼                ▼
    ┌──────────────┐  ┌───────────────────┐
    │  Label head  │  │  Domain classifier│
    │  G_y: Z → Y  │  │  G_d: Z → {s,t}   │
    └──────────────┘  └───────────────────┘
    Minimise class        Maximise domain
    prediction loss  ←──  confusion (FOOL the discriminator)


──────────────────────────────────────────────────────────────────────────────
### Out-of-Distribution (OOD) Detection

OOD detection asks: is this test example from the same distribution as
training? If not, the model should say "I don't know" rather than
silently give a wrong prediction with high confidence.

    The challenge: neural networks are confidently wrong on OOD inputs.
    A model trained on cats/dogs can classify a car as "dog" with 99%
    confidence. Softmax probabilities do NOT measure uncertainty.

    Diagram 5 — Confidence vs Familiarity:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  IN-DISTRIBUTION:     cat image → softmax → [0.02, 0.97, 0.01]  │
    │                                              correctly confident│
    │                                                                 │
    │  OOD (truck image):   truck    → softmax → [0.01, 0.95, 0.04]   │
    │                                              confidently wrong! │
    │                                                                 │
    │  The softmax output is a probability OVER CLASSES, not over     │
    │  "how familiar is this input". It cannot detect OOD by design.  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

**OOD Detection Methods:**

    BASELINE — Maximum Softmax Probability (MSP, Hendrycks 2017):
        Score(x) = max_c P(y=c | x)
        Threshold: flag as OOD if score < τ.
        Simple but unreliable — networks are overconfident on OOD.

    ODIN (Liang et al., 2018):
        Temperature scaling: soften the softmax with temperature T.
        Input perturbation: small adversarial perturbation in gradient direction.
        These two together widen the gap between in-distribution and OOD scores.

    DEEP ENSEMBLES:
        Train m models. OOD inputs show high disagreement between models.
        Score = variance of predictions across ensemble members.

    ENERGY-BASED DETECTION (Liu et al., 2020):
        Energy score:  E(x) = −T · log Σ_c exp(f_c(x)/T)
        In-distribution: lower energy.
        OOD: higher energy.
        More reliable than MSP — not bounded like softmax.

    MAHALANOBIS DISTANCE (Lee et al., 2018):
        Fit a Gaussian to each class in feature space:
        M(x) = min_c (φ(x) − μ_c)ᵀ Σ⁻¹ (φ(x) − μ_c)
        Low distance → in-distribution. High distance → OOD.

**Evaluation Metrics for OOD Detection:**

    AUROC:  Area under the ROC curve for the binary in/OOD task.
    AUPR:   Area under Precision-Recall curve.
    FPR@95TPR: False positive rate when 95% of in-distribution data
               is correctly classified as in-distribution.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Perfect OOD detector: AUROC=1.0, FPR@95TPR=0.0                  │
    │  Random detector:      AUROC=0.5, FPR@95TPR=0.95                 │
    │  MSP baseline:         AUROC≈0.85-0.90 on many benchmarks        │
    │  Deep Ensembles:        AUROC≈0.92-0.97                          │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Robustness — Resisting Adversarial and Natural Corruption

A model is ROBUST if its performance degrades gracefully under
perturbations of the input distribution.

**Natural Robustness — ImageNet-C and beyond:**

    ImageNet-C (Hendrycks & Dietterich, 2019): CIFAR/ImageNet images
    with 15 types of corruption at 5 severity levels (blur, noise,
    weather, digital). A robust model should maintain performance.

    mPC (mean Corruption Error) and relative mPC are standard metrics.

**Adversarial Robustness:**

    An adversarial example is a carefully crafted small perturbation
    δ to input x that fools the model:

        f(x + δ) ≠ f(x)   with   ‖δ‖ ≤ ε

    The perturbation is imperceptible to humans but defeats the model.

    FGSM (Fast Gradient Sign Method, Goodfellow et al., 2014):
        δ = ε · sign(∇_x ℓ(f(x), y))
        One gradient step in the direction that maximises the loss.

    PGD (Projected Gradient Descent, Madry et al., 2018):
        δᵢ₊₁ = Π_{‖δ‖≤ε}[ δᵢ + α · sign(∇_x ℓ(f(x+δᵢ), y)) ]
        Multi-step FGSM with projection back to the ε-ball.
        Stronger attack — better approximates the worst-case perturbation.

    Diagram 6 — Adversarial Perturbation:

    Original x:      ──── panda ────     Clean prediction:  panda (99%)
                          + ε·sign(∇ℓ)
                          ↓
    Adversarial x':  ──── panda ────     Adversarial pred:  gibbon (99%)
                   (looks same to humans, ‖δ‖∞ = 0.007)

    ADVERSARIAL TRAINING — the strongest known defence:
        Include adversarial examples in training:
        min_θ  E_{(x,y)~D} [ max_{‖δ‖≤ε} ℓ(f_θ(x+δ), y) ]
        Inner max: find worst-case perturbation (PGD attack).
        Outer min: update model to be robust to it.
        Cost: 3-10× training time, ~10-20% clean accuracy drop.

**Certified Robustness:**

    Adversarial training is empirical — a stronger attack might still
    break the model. Certified robustness provides mathematical guarantees.

    Randomised Smoothing (Cohen et al., 2019):
        Given f, construct a smoothed classifier g:
        g(x) = argmax_c P(f(x + N(0,σ²I)) = c)
        For any perturbation δ with ‖δ‖₂ ≤ σ·Φ⁻¹(p_A)/√2:
        g(x + δ) = g(x)  with probability 1 − α.
        Provides an ℓ₂ robustness certificate at scale.


──────────────────────────────────────────────────────────────────────────────
### Data Drift in Production — Monitoring and Adaptation

In production ML systems, distribution shift is continuous and inevitable.
A monitoring framework is essential.

    Diagram 7 — Production ML Monitoring Stack:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  RAW DATA MONITOR:          ● Feature statistics (mean, std)    │
    │                             ● Missing values, schema changes    │
    │                             ● KS test per feature               │
    │                                                                 │
    │  MODEL OUTPUT MONITOR:      ● Prediction score distribution     │
    │                             ● Class probability distribution    │
    │                             ● Confidence histograms             │
    │                                                                 │
    │  PERFORMANCE MONITOR:       ● Accuracy on labelled feedback     │
    │  (when labels available)    ● Precision/recall by segment       │
    │                                                                 │
    │  ALERT LAYER:               ● Statistical significance tests    │
    │                             ● Thresholds per metric             │
    │                             ● Escalation rules                  │
    │                                                                 │
    │  RESPONSE LAYER:            ● Retrain trigger                   │
    │                             ● Fallback to simpler model         │
    │                             ● Human-in-the-loop routing         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

**Sliding Window Methods for Concept Drift:**

    ADWIN (Adaptive Windowing, Bifet & Gavalda 2007):
        Maintain a window of recent predictions.
        If the mean error in any sub-window differs significantly
        from the rest → drift detected → shrink window (forget old data).

    DDM (Drift Detection Method):
        Track error rate pᵢ and standard deviation sᵢ.
        If pᵢ + sᵢ > p_min + 3·s_min → drift alarm.
        p_min, s_min are the best (lowest error) values seen so far.

    KSWIN (Kolmogorov-Smirnov Windowed):
        Apply KS test on two halves of a sliding window.
        If KS statistic exceeds threshold → concept drift detected.

**Continual Learning — Adapting Without Forgetting:**

    The challenge: when retraining on new data, the model "forgets"
    the old distribution (catastrophic forgetting).

    Solutions:
    ─ REPLAY BUFFERS: store a small subset of old data, include it
      in every retrain. Bounds forgetting at storage cost.
    ─ ELASTIC WEIGHT CONSOLIDATION (EWC): penalise changes to weights
      that were important for old tasks (measured by Fisher information).
    ─ PROGRESSIVE NEURAL NETWORKS: freeze old network columns, add new
      lateral connections for new data — no forgetting by design.


──────────────────────────────────────────────────────────────────────────────
### Spurious Correlations and Shortcut Learning

Models learn shortcuts — spurious correlations present in training
data that break under distribution shift.

    Classic examples:
    ─ CheXpert: chest X-ray model learned to predict positive diagnosis
      from hospital-specific text watermarks, not from pathology.
    ─ NLP inference: models learn "not" always leads to contradiction class.
    ─ ImageNet: grass background → cow; water background → whale.
    ─ COVID CT scans: model learned patient position (scanner artifact),
      not lung pathology.

    Why shortcuts are learned: they are predictive in training
    because of dataset construction biases. The ERM (empirical risk
    minimisation) principle finds ANY function that minimises training
    loss — including shortcuts.

    Diagram 8 — Shortcut Learning Under Distribution Shift:

    Training distribution:                  Shifted distribution:
    ┌─────────────────────────────┐          ┌──────────────────────────┐
    │  Cow + grass background     │          │  Cow + beach background  │
    │  P(grass | cow) = 0.95      │          │  P(grass | cow) = 0.10   │
    │                             │          │                          │
    │  Model learns:              │          │  Model predicts:         │
    │  "grass → cow" shortcut     │  ──────▶ │  NOT cow (no grass!)     │
    └─────────────────────────────┘          └──────────────────────────┘

**Mitigating Shortcut Learning:**

    ─ GROUP DRO (Distributionally Robust Optimisation):
        Minimise worst-case loss across known groups:
        min_f  max_g  E_{(x,y) ∈ group g}[ℓ(f(x), y)]
        Finds models that are good for all groups, not just majority.

    ─ DATA AUGMENTATION to break spurious correlations:
        Add examples with the spurious feature but wrong label association.
        E.g., cows on beach, boats on grass.

    ─ INVARIANT RISK MINIMISATION (IRM, Arjovsky et al., 2019):
        Learn features that are invariant across multiple environments.
        min_{φ,w}  Σ_e R^e(w ∘ φ)  s.t.  w = argmin_{w̃} R^e(w̃ ∘ φ)  ∀e
        Forces the same linear classifier to be optimal in all environments.


──────────────────────────────────────────────────────────────────────────────
### Practical Checklist: Deploying Robustly

    BEFORE TRAINING:
    ┌──────────────────────────────────────────────────────────────────┐
    │  1. Profile training data: which populations, time periods?      │
    │  2. Identify likely shifts at deployment.                        │
    │  3. Collect data from the target distribution if possible.       │
    │  4. Analyse spurious correlations in training data.              │
    └──────────────────────────────────────────────────────────────────┘

    DURING TRAINING:
    ┌──────────────────────────────────────────────────────────────────┐
    │  5. Evaluate on held-out data from the target domain.            │
    │  6. Use stratified splits to ensure all subpopulations covered.  │
    │  7. Apply data augmentation for known invariances.               │
    │  8. Consider distributionally robust objectives (Group DRO).     │
    └──────────────────────────────────────────────────────────────────┘

    AT DEPLOYMENT:
    ┌──────────────────────────────────────────────────────────────────┐
    │  9.  Monitor feature distributions (KS test, MMD).               │
    │  10. Monitor prediction score distribution.                      │
    │  11. Set automated alerts for statistically significant shifts.  │
    │  12. Define retrain triggers and fallback rules.                 │
    │  13. Track OOD detection scores (AUROC, FPR@95).                 │
    │  14. Log everything — post-hoc analysis requires full history.   │
    └──────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Covariate Shift Detection and Importance Weighting": {
        "description": (
            "Simulates covariate shift between a training distribution and a "
            "shifted test distribution. Detects the shift via the KS test "
            "per feature and the MMD two-sample test. Estimates importance "
            "weights via a domain classifier (train vs test discriminator). "
            "Shows that importance-weighted ERM recovers the unbiased test "
            "performance, while unweighted ERM degrades under the shift. "
            "Visualises weight distributions and effective sample size."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as _np_impl
import scipy.optimize as _sp_opt

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def roc_auc_score(y_true,y_score):
    yt=_np_impl.asarray(y_true); ys=_np_impl.asarray(y_score)
    desc=_np_impl.argsort(ys)[::-1]; yt=yt[desc]
    tp=_np_impl.cumsum(yt); fp=_np_impl.cumsum(1-yt)
    tp_r=tp/tp[-1]; fp_r=fp/fp[-1]
    return float(_np_impl.trapezoid(tp_r,fp_r))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; lam=1.0/(self.C*n)
        w=_np_impl.ones(n) if sample_weight is None else _np_impl.asarray(sample_weight)
        w=w/w.sum()*n
        def fg(params):
            p=_np_impl.clip(_sig(X@params[1:]+params[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(w*(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)))+0.5*lam*_np_impl.sum(params[1:]**2)
            e=w*(p-y)
            return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*params[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return accuracy_score(y,self.predict(X))

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
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict_proba(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        p=_sig(F); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

class SGDClassifier:
    def __init__(self,loss='log_loss',random_state=None,warm_start=False,alpha=0.0001,**kw):
        self.alpha=alpha; self._w=None; self._b=0.0; self._t=1
    def partial_fit(self,X,y,classes=None):
        X=_np_impl.atleast_2d(X).astype(float); y=_np_impl.asarray(y)
        if self._w is None: self._w=_np_impl.zeros(X.shape[1])
        lr=min(1.0/(self.alpha*self._t+1e-6),0.1)
        for xi,yi in zip(X,y):
            p=_sig(xi@self._w+self._b); e=p-yi
            self._w-=lr*(e*xi+self.alpha*self._w); self._b-=lr*e; self._t+=1
        return self
    def predict(self,X):
        X=_np_impl.atleast_2d(X).astype(float)
        return (_sig(X@self._w+self._b)>=0.5).astype(int)

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),max_iter=200,random_state=None,
                 activation='relu',alpha=0.0001,**kw):
        self.hidden_layer_sizes=hidden_layer_sizes; self.max_iter=max_iter
        self.random_state=random_state; self.alpha=alpha
    def fit(self,X,y):
        rng=_np_impl.random.default_rng(self.random_state)
        self._classes=_np_impl.unique(y); nc=len(self._classes)
        dims=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,_np_impl.sqrt(2.0/dims[i]),(dims[i],dims[i+1]))
                     for i in range(len(dims)-1)]
        self.intercepts_=[_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n=len(X); lr=1e-3
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for s in range(0,n,32):
                xb=X[idx[s:s+32]]; yb=y[idx[s:s+32]]; nb=len(xb)
                acts=[xb]
                for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+b
                    acts.append(_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z)
                logits=acts[-1]-acts[-1].max(1,keepdims=True)
                exp=_np_impl.exp(logits); probs=exp/exp.sum(1,keepdims=True)
                oh=_np_impl.zeros_like(probs)
                for ci,c in enumerate(self._classes): oh[yb==c,ci]=1
                delta=(probs-oh)/nb
                for i in range(len(self.coefs_)-1,-1,-1):
                    self.coefs_[i]-=lr*(acts[i].T@delta+self.alpha*self.coefs_[i])
                    self.intercepts_[i]-=lr*delta.sum(0)
                    if i>0: delta=(delta@self.coefs_[i].T)*(acts[i]>0)
        return self
    def predict_proba(self,X):
        a=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=a@W+b; a=_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z
        a=a-a.max(1,keepdims=True); e=_np_impl.exp(a); return e/e.sum(1,keepdims=True)
    def predict(self,X): return self._classes[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return accuracy_score(y,self.predict(X))

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=n_samples//2
    t=_np_impl.linspace(0,_np_impl.pi,n)
    X=_np_impl.vstack([_np_impl.c_[_np_impl.cos(t),_np_impl.sin(t)],
                       _np_impl.c_[1-_np_impl.cos(t),-_np_impl.sin(t)+0.5]])
    if noise>0: X+=rng.normal(0,noise,X.shape)
    return X,_np_impl.hstack([_np_impl.zeros(n),_np_impl.ones(n_samples-n)]).astype(int)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Simulate covariate shift
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(n, mean_X, seed=0):
    """
    P(Y|X) is fixed: Y = sign(X[:,0] + X[:,1] − 0.5 + noise).
    P(X) varies between train and test.
    """
    rng = np.random.default_rng(seed)
    X   = rng.multivariate_normal(mean_X, [[1.0, 0.5], [0.5, 1.5]], n)
    y   = (X[:, 0] + X[:, 1] - 0.5 + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y

# Training: centred at origin
X_tr, y_tr = generate_data(800, mean_X=[0.0, 0.0], seed=1)
# Test: shifted distribution (P(X) changed, P(Y|X) unchanged)
X_te, y_te = generate_data(400, mean_X=[1.2, -0.8], seed=2)

print("=" * 65)
print("  COVARIATE SHIFT DETECTION AND IMPORTANCE WEIGHTING")
print("=" * 65)
print()
print(f"  Training:   n={len(X_tr)}, mean X≈{X_tr.mean(0).round(2)}")
print(f"  Test (shifted): n={len(X_te)}, mean X≈{X_te.mean(0).round(2)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Detection: KS test per feature
# ─────────────────────────────────────────────────────────────────────────────

print("  DETECTION — KS TEST PER FEATURE:")
print(f"  {'Feature':>10}  {'KS stat':>10}  {'p-value':>12}  {'Shifted?':>10}")
print("  " + "─" * 46)
for j in range(X_tr.shape[1]):
    ks_stat, p_val = ks_2samp(X_tr[:, j], X_te[:, j])
    shifted = "YES ⚠" if p_val < 0.05 else "no"
    print(f"  {'X'+str(j):>10}  {ks_stat:>10.4f}  {p_val:>12.4e}  {shifted:>10}")

# ─────────────────────────────────────────────────────────────────────────────
# Detection: MMD two-sample test
# ─────────────────────────────────────────────────────────────────────────────

def mmd_squared(X, Y, gamma=1.0):
    """Unbiased MMD² estimate with RBF kernel."""
    def rbf(A, B):
        dists = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * A @ B.T
        return np.exp(-gamma * dists)
    n, m = len(X), len(Y)
    K_XX = rbf(X, X); K_YY = rbf(Y, Y); K_XY = rbf(X, Y)
    # Unbiased estimator (zero diagonal)
    np.fill_diagonal(K_XX, 0); np.fill_diagonal(K_YY, 0)
    return (K_XX.sum()/(n*(n-1)) + K_YY.sum()/(m*(m-1))
            - 2*K_XY.mean())

def mmd_permutation_test(X, Y, n_perms=200, gamma=1.0, seed=0):
    """Permutation test for H0: P_X = P_Y."""
    rng     = np.random.default_rng(seed)
    obs_mmd = mmd_squared(X, Y, gamma)
    XY      = np.vstack([X, Y])
    n       = len(X)
    null    = []
    for _ in range(n_perms):
        perm     = rng.permutation(len(XY))
        null_mmd = mmd_squared(XY[perm[:n]], XY[perm[n:]], gamma)
        null.append(null_mmd)
    p_val = (np.array(null) >= obs_mmd).mean()
    return obs_mmd, p_val, np.array(null)

print()
print("  DETECTION — MMD TWO-SAMPLE TEST:")
mmd_obs, mmd_p, mmd_null = mmd_permutation_test(
    X_tr[:200], X_te[:200], n_perms=500, gamma=0.5)
print(f"  MMD² (observed)  = {mmd_obs:.6f}")
print(f"  p-value (500 permutations) = {mmd_p:.4f}")
print(f"  Null distribution mean = {mmd_null.mean():.6f}")
print(f"  Shift detected: {'YES ⚠' if mmd_p < 0.05 else 'no'}")

# ─────────────────────────────────────────────────────────────────────────────
# Importance weight estimation via domain classifier
# ─────────────────────────────────────────────────────────────────────────────

print()
print("  IMPORTANCE WEIGHT ESTIMATION (domain classifier):")
# Combine train and test, label them 0 (train) and 1 (test)
X_domain = np.vstack([X_tr, X_te])
y_domain = np.array([0]*len(X_tr) + [1]*len(X_te))

sc_d  = StandardScaler().fit(X_domain)
X_d_s = sc_d.transform(X_domain)

domain_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         random_state=0)
domain_clf.fit(X_d_s, y_domain)
domain_auc = roc_auc_score(y_domain,
                            domain_clf.predict_proba(X_d_s)[:, 1])
print(f"  Domain classifier AUC = {domain_auc:.4f}  "
      f"(0.5=no shift, 1.0=perfect separation)")
print(f"  {'AUC > 0.6' if domain_auc > 0.6 else 'AUC ≤ 0.6'} → "
      f"{'significant' if domain_auc > 0.6 else 'no detectable'} shift")

# Compute importance weights for training set: w(x) = p(test|x)/p(train|x)
p_test_given_x = domain_clf.predict_proba(sc_d.transform(X_tr))[:, 1]
n_tr, n_te     = len(X_tr), len(X_te)
p_test_prior   = n_te / (n_tr + n_te)
p_train_prior  = n_tr / (n_tr + n_te)

# Bayes: p(test|x)/p(train|x) = [p(x|test)*p(test)] / [p(x|train)*p(train)]
#        Using classifier: p(test|x)/p(train|x) via Bayes
p_x_given_test  = p_test_given_x / p_test_prior
p_x_given_train = (1 - p_test_given_x) / p_train_prior
raw_weights = p_x_given_test / (p_x_given_train + 1e-10)
# Normalise and clip for stability
weights = np.clip(raw_weights, 0, 20)
weights = weights / weights.mean()   # normalise to mean 1

eff_n = (weights.sum()**2) / (weights**2).sum()
print()
print(f"  Importance weight statistics:")
print(f"    min={weights.min():.3f}  max={weights.max():.3f}  "
      f"mean={weights.mean():.3f}  std={weights.std():.3f}")
print(f"    Effective sample size (ESS) = {eff_n:.1f} / {len(X_tr)}"
      f"  ({100*eff_n/len(X_tr):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Compare: unweighted vs importance-weighted classifier
# ─────────────────────────────────────────────────────────────────────────────

print()
print("  EFFECT OF IMPORTANCE WEIGHTING ON TEST ACCURACY:")
sc_clf = StandardScaler().fit(X_tr)
X_tr_s = sc_clf.transform(X_tr)
X_te_s = sc_clf.transform(X_te)

# Unweighted (standard ERM)
clf_unw = LogisticRegression(C=1.0, random_state=0)
clf_unw.fit(X_tr_s, y_tr)

# Importance-weighted ERM
clf_iw = LogisticRegression(C=1.0, random_state=0)
clf_iw.fit(X_tr_s, y_tr, sample_weight=weights)

acc_tr_unw = accuracy_score(y_tr, clf_unw.predict(X_tr_s))
acc_te_unw = accuracy_score(y_te, clf_unw.predict(X_te_s))
acc_tr_iw  = accuracy_score(y_tr, clf_iw.predict(X_tr_s))
acc_te_iw  = accuracy_score(y_te, clf_iw.predict(X_te_s))

# Oracle: train on test distribution directly
clf_oracle = LogisticRegression(C=1.0, random_state=0)
clf_oracle.fit(X_te_s, y_te)
acc_oracle  = accuracy_score(y_te, clf_oracle.predict(X_te_s))

print()
print(f"  {'Method':>30}  {'Train acc':>10}  {'Test acc':>10}")
print("  " + "─" * 54)
print(f"  {'Unweighted ERM':>30}  {acc_tr_unw:>10.4f}  {acc_te_unw:>10.4f}")
print(f"  {'Importance-weighted ERM':>30}  {acc_tr_iw:>10.4f}  {acc_te_iw:>10.4f}")
print(f"  {'Oracle (train on test dist.)':>30}  {'N/A':>10}  {acc_oracle:>10.4f}")
print()
print("  ─ Unweighted ERM: optimised for training distribution.")
print("  ─ IW-ERM: corrects for shift → test accuracy closer to oracle.")
print(f"  ─ IW improvement: +{acc_te_iw - acc_te_unw:+.4f} on test set.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Covariate Shift: Detection, Importance Weighting, and Correction",
             fontsize=13, fontweight="bold")

# Panel (0,0): Train vs test marginal distributions
ax = axes[0, 0]
for j, col in enumerate(["steelblue", "tomato"]):
    ax.hist(X_tr[:, j], bins=30, alpha=0.5, density=True, color=col,
            label=f"Train X{j}")
    ax.hist(X_te[:, j], bins=30, alpha=0.5, density=True, color=col,
            linestyle="--", histtype="step", lw=2, label=f"Test X{j}")
ax.set_title("Feature Distributions: Train vs Test\\n(shifted test distribution)",
             fontweight="bold")
ax.set_xlabel("Feature value"); ax.set_ylabel("Density")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (0,1): Scatter — train vs test
ax = axes[0, 1]
ax.scatter(X_tr[:, 0], X_tr[:, 1], c="steelblue", s=10, alpha=0.4, label="Train")
ax.scatter(X_te[:, 0], X_te[:, 1], c="tomato",    s=10, alpha=0.5, label="Test")
ax.set_title("2D Scatter: Train vs Test\\n(test cloud shifted in feature space)",
             fontweight="bold")
ax.set_xlabel("X₀"); ax.set_ylabel("X₁")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (0,2): MMD null distribution
ax = axes[0, 2]
ax.hist(mmd_null, bins=30, color="steelblue", alpha=0.8, density=True,
        label="Null MMD² distribution")
ax.axvline(mmd_obs, color="tomato", lw=2.5, label=f"Observed MMD²={mmd_obs:.4f}")
ax.set_title(f"MMD Permutation Test\\n(p={mmd_p:.4f} — shift {'detected' if mmd_p<0.05 else 'not detected'})",
             fontweight="bold")
ax.set_xlabel("MMD²"); ax.set_ylabel("Density")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,0): Importance weight distribution
ax = axes[1, 0]
ax.hist(weights, bins=40, color="seagreen", alpha=0.8, edgecolor="white")
ax.axvline(1.0,  color="black", lw=2, ls="--", label="w=1 (no correction)")
ax.axvline(weights.mean(), color="tomato", lw=2, label=f"Mean={weights.mean():.2f}")
ax.set_title(f"Importance Weights Distribution\\n(ESS={eff_n:.0f}/{len(X_tr)} = {100*eff_n/len(X_tr):.0f}%)",
             fontweight="bold")
ax.set_xlabel("w(x) = p_test(x)/p_train(x)")
ax.set_ylabel("Count")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,1): Decision boundaries — unweighted vs weighted
ax = axes[1, 1]
h = 0.04
x0_min, x0_max = X_domain[:, 0].min()-0.5, X_domain[:, 0].max()+0.5
x1_min, x1_max = X_domain[:, 1].min()-0.5, X_domain[:, 1].max()+0.5
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                        np.arange(x1_min, x1_max, h))
X_grid_s = sc_clf.transform(np.c_[xx0.ravel(), xx1.ravel()])
Z_unw = clf_unw.predict(X_grid_s).reshape(xx0.shape)
Z_iw  = clf_iw.predict(X_grid_s).reshape(xx0.shape)
ax.contour(xx0, xx1, Z_unw, levels=[0.5], colors=["steelblue"],
           linestyles=["-"], linewidths=2.5)
ax.contour(xx0, xx1, Z_iw,  levels=[0.5], colors=["tomato"],
           linestyles=["--"], linewidths=2.5)
ax.scatter(X_te[y_te==1, 0], X_te[y_te==1, 1], c="seagreen", s=15, alpha=0.5)
ax.scatter(X_te[y_te==0, 0], X_te[y_te==0, 1], c="purple",   s=15, alpha=0.5)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0],color="steelblue",lw=2,label=f"Unweighted (test acc={acc_te_unw:.3f})"),
           Line2D([0],[0],color="tomato",lw=2,ls="--",label=f"IW-weighted (test acc={acc_te_iw:.3f})")]
ax.set_title("Decision Boundaries on Test Distribution\\n(IW shifts boundary toward test data)",
             fontweight="bold")
ax.legend(handles=handles, fontsize=8); ax.grid(alpha=0.3)

# Panel (1,2): Accuracy summary bar chart
ax = axes[1, 2]
methods  = ["Unweighted\\nERM", "IW-weighted\\nERM", "Oracle\\n(train on test)"]
tr_accs  = [acc_tr_unw, acc_tr_iw, None]
te_accs  = [acc_te_unw, acc_te_iw, acc_oracle]
x_pos    = np.arange(len(methods))
ax.bar(x_pos - 0.2, [t if t else 0 for t in tr_accs], 0.4,
       color="steelblue", alpha=0.8, label="Train accuracy")
ax.bar(x_pos + 0.2, te_accs, 0.4,
       color="tomato", alpha=0.8, label="Test accuracy")
ax.set_xticks(x_pos); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("Impact of Importance Weighting\\n(IW closes train-test accuracy gap)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for i, (ta, tea) in enumerate(zip(tr_accs, te_accs)):
    if ta: ax.text(i - 0.2, ta + 0.005, f"{ta:.3f}", ha="center", fontsize=8)
    ax.text(i + 0.2, tea + 0.005, f"{tea:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("covariate_shift.png", dpi=110)
print()
print("  Plot saved → covariate_shift.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Concept Drift Detection — ADWIN and DDM on Data Streams": {
        "description": (
            "Simulates three types of concept drift — sudden, gradual, and "
            "recurring — on a streaming binary classification task. Implements "
            "two drift detectors from scratch: DDM (Drift Detection Method) "
            "and a sliding-window KS test. Shows how both detectors raise "
            "alarms after true drift events and compares detection latency. "
            "Plots the error stream, drift alerts, and model performance "
            "before and after drift points."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as _np_impl
import scipy.optimize as _sp_opt

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def roc_auc_score(y_true,y_score):
    yt=_np_impl.asarray(y_true); ys=_np_impl.asarray(y_score)
    desc=_np_impl.argsort(ys)[::-1]; yt=yt[desc]
    tp=_np_impl.cumsum(yt); fp=_np_impl.cumsum(1-yt)
    tp_r=tp/tp[-1]; fp_r=fp/fp[-1]
    return float(_np_impl.trapezoid(tp_r,fp_r))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; lam=1.0/(self.C*n)
        w=_np_impl.ones(n) if sample_weight is None else _np_impl.asarray(sample_weight)
        w=w/w.sum()*n
        def fg(params):
            p=_np_impl.clip(_sig(X@params[1:]+params[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(w*(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)))+0.5*lam*_np_impl.sum(params[1:]**2)
            e=w*(p-y)
            return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*params[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return accuracy_score(y,self.predict(X))

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
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict_proba(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        p=_sig(F); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

class SGDClassifier:
    def __init__(self,loss='log_loss',random_state=None,warm_start=False,alpha=0.0001,**kw):
        self.alpha=alpha; self._w=None; self._b=0.0; self._t=1
    def partial_fit(self,X,y,classes=None):
        X=_np_impl.atleast_2d(X).astype(float); y=_np_impl.asarray(y)
        if self._w is None: self._w=_np_impl.zeros(X.shape[1])
        lr=min(1.0/(self.alpha*self._t+1e-6),0.1)
        for xi,yi in zip(X,y):
            p=_sig(xi@self._w+self._b); e=p-yi
            self._w-=lr*(e*xi+self.alpha*self._w); self._b-=lr*e; self._t+=1
        return self
    def predict(self,X):
        X=_np_impl.atleast_2d(X).astype(float)
        return (_sig(X@self._w+self._b)>=0.5).astype(int)

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),max_iter=200,random_state=None,
                 activation='relu',alpha=0.0001,**kw):
        self.hidden_layer_sizes=hidden_layer_sizes; self.max_iter=max_iter
        self.random_state=random_state; self.alpha=alpha
    def fit(self,X,y):
        rng=_np_impl.random.default_rng(self.random_state)
        self._classes=_np_impl.unique(y); nc=len(self._classes)
        dims=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,_np_impl.sqrt(2.0/dims[i]),(dims[i],dims[i+1]))
                     for i in range(len(dims)-1)]
        self.intercepts_=[_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n=len(X); lr=1e-3
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for s in range(0,n,32):
                xb=X[idx[s:s+32]]; yb=y[idx[s:s+32]]; nb=len(xb)
                acts=[xb]
                for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+b
                    acts.append(_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z)
                logits=acts[-1]-acts[-1].max(1,keepdims=True)
                exp=_np_impl.exp(logits); probs=exp/exp.sum(1,keepdims=True)
                oh=_np_impl.zeros_like(probs)
                for ci,c in enumerate(self._classes): oh[yb==c,ci]=1
                delta=(probs-oh)/nb
                for i in range(len(self.coefs_)-1,-1,-1):
                    self.coefs_[i]-=lr*(acts[i].T@delta+self.alpha*self.coefs_[i])
                    self.intercepts_[i]-=lr*delta.sum(0)
                    if i>0: delta=(delta@self.coefs_[i].T)*(acts[i]>0)
        return self
    def predict_proba(self,X):
        a=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=a@W+b; a=_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z
        a=a-a.max(1,keepdims=True); e=_np_impl.exp(a); return e/e.sum(1,keepdims=True)
    def predict(self,X): return self._classes[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return accuracy_score(y,self.predict(X))

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=n_samples//2
    t=_np_impl.linspace(0,_np_impl.pi,n)
    X=_np_impl.vstack([_np_impl.c_[_np_impl.cos(t),_np_impl.sin(t)],
                       _np_impl.c_[1-_np_impl.cos(t),-_np_impl.sin(t)+0.5]])
    if noise>0: X+=rng.normal(0,noise,X.shape)
    return X,_np_impl.hstack([_np_impl.zeros(n),_np_impl.ones(n_samples-n)]).astype(int)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Simulate streaming data with concept drift
# ─────────────────────────────────────────────────────────────────────────────

def make_stream(n_total=3000, drift_type="sudden", seed=0):
    """
    Generate a binary classification stream with concept drift.
    drift_type: "sudden", "gradual", "recurring"
    Returns: X (n×2), y (n,), drift_times (list of ground-truth drift indices)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_total, 2))
    y = np.zeros(n_total, dtype=int)

    if drift_type == "sudden":
        # Before t=1000: Y = sign(X0 + X1). After t=1000: Y = sign(X0 - X1).
        y[:1000]  = (X[:1000, 0] + X[:1000, 1] + rng.normal(0, 0.3, 1000) > 0).astype(int)
        y[1000:]  = (X[1000:, 0] - X[1000:, 1] + rng.normal(0, 0.3, 2000) > 0).astype(int)
        drift_times = [1000]

    elif drift_type == "gradual":
        # Mixing coefficient transitions from 0 to 1 between t=1000 and t=1500
        for i in range(n_total):
            alpha = min(max((i - 1000) / 500, 0), 1)
            logit = (1 - alpha) * (X[i, 0] + X[i, 1]) + alpha * (X[i, 0] - X[i, 1])
            y[i]  = int(logit + rng.normal(0, 0.3) > 0)
        drift_times = [1000, 1500]  # start and end of gradual drift

    elif drift_type == "recurring":
        # Drift at t=1000, reverses at t=2000 (seasonal-like)
        y[:1000]  = (X[:1000, 0] + X[:1000, 1] + rng.normal(0, 0.3, 1000) > 0).astype(int)
        y[1000:2000] = (X[1000:2000, 0] - X[1000:2000, 1] + rng.normal(0, 0.3, 1000) > 0).astype(int)
        y[2000:]  = (X[2000:, 0] + X[2000:, 1] + rng.normal(0, 0.3, 1000) > 0).astype(int)
        drift_times = [1000, 2000]

    return X, y, drift_times


# ─────────────────────────────────────────────────────────────────────────────
# DDM — Drift Detection Method (Gama et al., 2004)
# ─────────────────────────────────────────────────────────────────────────────

class DDM:
    """
    DDM: monitors running error rate and standard deviation.
    Alarm raised when pᵢ + sᵢ > p_min + k * s_min (k=3 for warning, k=4 for drift).
    """
    def __init__(self, k_warning=2.0, k_drift=3.0, min_instances=30):
        self.k_w    = k_warning
        self.k_d    = k_drift
        self.min_n  = min_instances
        self.reset()

    def reset(self):
        self.n      = 0
        self.p      = 0.0
        self.s      = 1.0
        self.p_min  = float("inf")
        self.s_min  = float("inf")
        self.warnings  = []
        self.drifts    = []

    def update(self, error):
        """Update with a new binary error (0 = correct, 1 = wrong)."""
        self.n += 1
        self.p  = self.p + (error - self.p) / self.n
        self.s  = np.sqrt(self.p * (1 - self.p) / self.n)
        if self.n < self.min_n:
            return "normal"
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        if self.p + self.s > self.p_min + self.k_d * self.s_min:
            self.drifts.append(self.n)
            self.reset()
            return "drift"
        if self.p + self.s > self.p_min + self.k_w * self.s_min:
            self.warnings.append(self.n)
            return "warning"
        return "normal"


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window KS Detector
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowKS:
    """
    Maintains two windows of error rates; raises alarm when KS test
    between the two halves is significant.
    """
    def __init__(self, window_size=100, alpha=0.01):
        self.ws    = window_size
        self.alpha = alpha
        self.buf   = []
        self.alarms = []

    def update(self, error, t):
        self.buf.append(error)
        if len(self.buf) > 2 * self.ws:
            self.buf = self.buf[-2*self.ws:]
        if len(self.buf) == 2 * self.ws:
            w1 = np.array(self.buf[:self.ws])
            w2 = np.array(self.buf[self.ws:])
            stat, pval = ks_2samp(w1, w2)
            if pval < self.alpha:
                self.alarms.append(t)
                self.buf = self.buf[self.ws:]   # reset reference window
                return "drift"
        return "normal"


# ─────────────────────────────────────────────────────────────────────────────
# Online learning with drift detection
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  CONCEPT DRIFT DETECTION ON DATA STREAMS")
print("=" * 65)

all_results = {}

for drift_type in ["sudden", "gradual", "recurring"]:
    X_stream, y_stream, true_drifts = make_stream(
        n_total=3000, drift_type=drift_type, seed=0)

    ddm  = DDM(k_warning=2.0, k_drift=3.0)
    ksw  = SlidingWindowKS(window_size=150, alpha=0.01)

    # Online learning: train on first 200 examples, then predict and update
    scaler     = StandardScaler()
    clf        = SGDClassifier(loss="log_loss", random_state=0, warm_start=True)
    burn_in    = 200
    errors     = np.zeros(len(y_stream))
    ddm_alarms = []
    ksw_alarms = []
    acc_window = []

    # Fit scaler on first batch
    scaler.fit(X_stream[:burn_in])
    clf.partial_fit(scaler.transform(X_stream[:burn_in]),
                    y_stream[:burn_in], classes=[0, 1])

    for t in range(burn_in, len(y_stream)):
        x_t = scaler.transform(X_stream[t:t+1])
        err = int(clf.predict(x_t)[0] != y_stream[t])
        errors[t] = err

        # Update detectors
        ddm_status = ddm.update(err)
        ksw_status = ksw.update(err, t)

        if ddm_status == "drift":
            ddm_alarms.append(t)
            # Retrain on recent window after drift
            recent = max(0, t - 200)
            scaler.fit(X_stream[recent:t])
            clf     = SGDClassifier(loss="log_loss", random_state=0)
            clf.partial_fit(scaler.transform(X_stream[recent:t]),
                            y_stream[recent:t], classes=[0, 1])
        else:
            clf.partial_fit(x_t, [y_stream[t]])

        if ksw_status == "drift":
            ksw_alarms.append(t)

        acc_window.append(1 - err)

    all_results[drift_type] = {
        "errors": errors,
        "ddm_alarms": ddm.drifts + ddm_alarms,
        "ksw_alarms": ksw_alarms,
        "true_drifts": true_drifts,
        "acc_window": acc_window,
    }

    # Summary
    rolling_err = np.convolve(errors[burn_in:], np.ones(100)/100, mode="valid")

    print()
    print(f"  DRIFT TYPE: {drift_type.upper()}")
    print(f"  True drift times:      {true_drifts}")
    print(f"  DDM alarms (drift):    {ddm.drifts[:5]}")
    print(f"  KS-window alarms:      {ksw_alarms[:5]}")
    print(f"  Mean error (first 800): {errors[burn_in:800].mean():.4f}")
    print(f"  Mean error (post-drift):{errors[true_drifts[0]:].mean():.4f}")

    # Detection latency
    for alarm_type, alarms in [("DDM", ddm.drifts), ("KSW", ksw_alarms)]:
        if alarms and true_drifts:
            first_alarm = min(alarms) if alarms else None
            if first_alarm and first_alarm > true_drifts[0]:
                latency = first_alarm - true_drifts[0]
                print(f"  {alarm_type} detection latency: {latency} samples")
            elif first_alarm:
                print(f"  {alarm_type} first alarm at t={first_alarm} "
                      f"(before true drift — false positive?)")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Concept Drift Detection: DDM and Sliding Window KS Test",
             fontsize=13, fontweight="bold")

drift_titles = {"sudden": "Sudden Drift (t=1000)",
                "gradual": "Gradual Drift (t=1000–1500)",
                "recurring": "Recurring Drift (t=1000, t=2000)"}

for row, drift_type in enumerate(["sudden", "gradual", "recurring"]):
    res = all_results[drift_type]
    t_range = np.arange(len(res["errors"]))

    # Rolling error rate
    window_s = 100
    roll_err = np.convolve(res["errors"], np.ones(window_s)/window_s,
                            mode="valid")
    t_roll   = np.arange(len(roll_err)) + (window_s - 1) // 2

    # Panel left: rolling error + drift alarms
    ax = axes[row, 0]
    ax.plot(t_roll, roll_err, "steelblue", lw=2, label="Rolling error (w=100)")
    for td in res["true_drifts"]:
        ax.axvline(td, color="black", lw=2, ls="--", alpha=0.8,
                   label="True drift" if td == res["true_drifts"][0] else "")
    for alarm_t in res["ddm_alarms"][:6]:
        ax.axvline(alarm_t, color="tomato", lw=1.5, alpha=0.7,
                   label="DDM alarm" if alarm_t == res["ddm_alarms"][0] else "")
    for alarm_t in res["ksw_alarms"][:6]:
        ax.axvline(alarm_t, color="seagreen", lw=1.5, ls=":",  alpha=0.7,
                   label="KSW alarm" if alarm_t == res["ksw_alarms"][0] else "")
    ax.set_ylabel("Rolling error rate")
    ax.set_title(f"{drift_titles[drift_type]}\\n(black=true drift, red=DDM, green=KSW)",
                 fontweight="bold")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 0.85)

    # Panel right: alarm timeline
    ax = axes[row, 1]
    ax.scatter(res["ddm_alarms"],
               [1.0] * len(res["ddm_alarms"]),
               marker="v", s=80, c="tomato", zorder=5, label="DDM drift alarm")
    ax.scatter(res["ksw_alarms"],
               [0.5] * len(res["ksw_alarms"]),
               marker="^", s=80, c="seagreen", zorder=5, label="KSW drift alarm")
    for td in res["true_drifts"]:
        ax.axvline(td, color="black", lw=2, ls="--",
                   label="True drift" if td == res["true_drifts"][0] else "")
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["KSW", "DDM"])
    ax.set_xlabel("Time step")
    ax.set_title(f"Alarm Timeline — {drift_type}\\n"
                 f"(closer to true drift line = lower latency)",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("concept_drift_detection.png", dpi=110)
print()
print("  Plot saved → concept_drift_detection.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · OOD Detection — Energy Score, MSP, and Mahalanobis Distance": {
        "description": (
            "Compares three OOD detection methods: Maximum Softmax Probability "
            "(MSP baseline), Energy score, and Mahalanobis distance in feature "
            "space. Uses a toy in-distribution dataset and three OOD datasets "
            "of increasing distance from the training distribution. Computes "
            "AUROC and FPR@95TPR for each method. Shows why MSP is overconfident "
            "and how energy/Mahalanobis distance are more reliable OOD indicators."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def roc_auc_score(y_true,y_score):
    yt=_np_impl.asarray(y_true); ys=_np_impl.asarray(y_score)
    desc=_np_impl.argsort(ys)[::-1]; yt=yt[desc]
    tp=_np_impl.cumsum(yt); fp=_np_impl.cumsum(1-yt)
    tp_r=tp/tp[-1]; fp_r=fp/fp[-1]
    return float(_np_impl.trapezoid(tp_r,fp_r))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; lam=1.0/(self.C*n)
        w=_np_impl.ones(n) if sample_weight is None else _np_impl.asarray(sample_weight)
        w=w/w.sum()*n
        def fg(params):
            p=_np_impl.clip(_sig(X@params[1:]+params[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(w*(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)))+0.5*lam*_np_impl.sum(params[1:]**2)
            e=w*(p-y)
            return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*params[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return accuracy_score(y,self.predict(X))

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
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict_proba(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        p=_sig(F); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

class SGDClassifier:
    def __init__(self,loss='log_loss',random_state=None,warm_start=False,alpha=0.0001,**kw):
        self.alpha=alpha; self._w=None; self._b=0.0; self._t=1
    def partial_fit(self,X,y,classes=None):
        X=_np_impl.atleast_2d(X).astype(float); y=_np_impl.asarray(y)
        if self._w is None: self._w=_np_impl.zeros(X.shape[1])
        lr=min(1.0/(self.alpha*self._t+1e-6),0.1)
        for xi,yi in zip(X,y):
            p=_sig(xi@self._w+self._b); e=p-yi
            self._w-=lr*(e*xi+self.alpha*self._w); self._b-=lr*e; self._t+=1
        return self
    def predict(self,X):
        X=_np_impl.atleast_2d(X).astype(float)
        return (_sig(X@self._w+self._b)>=0.5).astype(int)

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),max_iter=200,random_state=None,
                 activation='relu',alpha=0.0001,**kw):
        self.hidden_layer_sizes=hidden_layer_sizes; self.max_iter=max_iter
        self.random_state=random_state; self.alpha=alpha
    def fit(self,X,y):
        rng=_np_impl.random.default_rng(self.random_state)
        self._classes=_np_impl.unique(y); nc=len(self._classes)
        dims=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,_np_impl.sqrt(2.0/dims[i]),(dims[i],dims[i+1]))
                     for i in range(len(dims)-1)]
        self.intercepts_=[_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n=len(X); lr=1e-3
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for s in range(0,n,32):
                xb=X[idx[s:s+32]]; yb=y[idx[s:s+32]]; nb=len(xb)
                acts=[xb]
                for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+b
                    acts.append(_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z)
                logits=acts[-1]-acts[-1].max(1,keepdims=True)
                exp=_np_impl.exp(logits); probs=exp/exp.sum(1,keepdims=True)
                oh=_np_impl.zeros_like(probs)
                for ci,c in enumerate(self._classes): oh[yb==c,ci]=1
                delta=(probs-oh)/nb
                for i in range(len(self.coefs_)-1,-1,-1):
                    self.coefs_[i]-=lr*(acts[i].T@delta+self.alpha*self.coefs_[i])
                    self.intercepts_[i]-=lr*delta.sum(0)
                    if i>0: delta=(delta@self.coefs_[i].T)*(acts[i]>0)
        return self
    def predict_proba(self,X):
        a=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=a@W+b; a=_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z
        a=a-a.max(1,keepdims=True); e=_np_impl.exp(a); return e/e.sum(1,keepdims=True)
    def predict(self,X): return self._classes[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return accuracy_score(y,self.predict(X))

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=n_samples//2
    t=_np_impl.linspace(0,_np_impl.pi,n)
    X=_np_impl.vstack([_np_impl.c_[_np_impl.cos(t),_np_impl.sin(t)],
                       _np_impl.c_[1-_np_impl.cos(t),-_np_impl.sin(t)+0.5]])
    if noise>0: X+=rng.normal(0,noise,X.shape)
    return X,_np_impl.hstack([_np_impl.zeros(n),_np_impl.ones(n_samples-n)]).astype(int)

from scipy.special import softmax

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# OOD detection utilities
# ─────────────────────────────────────────────────────────────────────────────

def auroc(in_scores, ood_scores):
    """AUROC: in-distribution = positive (label 1)."""
    y_true  = np.concatenate([np.ones(len(in_scores)), np.zeros(len(ood_scores))])
    y_score = np.concatenate([in_scores, ood_scores])
    return roc_auc_score(y_true, y_score)

def fpr_at_tpr(in_scores, ood_scores, tpr_target=0.95):
    """FPR when TPR = tpr_target (using in-distribution = positive)."""
    threshold = np.percentile(in_scores, 100 * (1 - tpr_target))
    fpr = (ood_scores >= threshold).mean()
    return fpr

def msp_score(logits):
    """Maximum Softmax Probability — higher means more in-distribution."""
    probs = softmax(logits, axis=1)
    return probs.max(axis=1)

def energy_score(logits, T=1.0):
    """Energy score — lower (more negative) means more in-distribution."""
    return -T * np.log(np.exp(logits / T).sum(axis=1))

def mahalanobis_score(features, class_means, shared_cov_inv):
    """
    Mahalanobis distance to nearest class centre (negative = more in-dist).
    """
    dists = []
    for mu in class_means:
        diff = features - mu
        d    = np.einsum("ni,ij,nj->n", diff, shared_cov_inv, diff)
        dists.append(d)
    return -np.min(dists, axis=0)   # negative = closer = more in-dist

# ─────────────────────────────────────────────────────────────────────────────
# Generate in-distribution and OOD datasets
# ─────────────────────────────────────────────────────────────────────────────

n_per_class = 300
n_ood_each  = 300

# In-distribution: 4-class Gaussian blobs
class_centres = [np.array([-2., -2.]),
                 np.array([ 2., -2.]),
                 np.array([-2.,  2.]),
                 np.array([ 2.,  2.])]
X_in  = np.vstack([np.random.randn(n_per_class, 2) * 0.6 + c for c in class_centres])
y_in  = np.repeat([0, 1, 2, 3], n_per_class)

# OOD datasets: increasing distance from training distribution
X_ood_near = np.random.randn(n_ood_each, 2) * 0.6 + np.array([0., 0.])  # centre (between clusters)
X_ood_mid  = np.random.randn(n_ood_each, 2) * 0.8 + np.array([5., 0.])  # adjacent
X_ood_far  = np.random.randn(n_ood_each, 2) * 0.5 + np.array([10., 10.]) # distant

scaler = StandardScaler().fit(X_in)
X_in_s    = scaler.transform(X_in)
X_ood_ns  = scaler.transform(X_ood_near)
X_ood_ms  = scaler.transform(X_ood_mid)
X_ood_fs  = scaler.transform(X_ood_far)

# ─────────────────────────────────────────────────────────────────────────────
# Train MLP classifier
# ─────────────────────────────────────────────────────────────────────────────

clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=0,
                    activation="relu", alpha=0.01)
clf.fit(X_in_s, y_in)
train_acc = clf.score(X_in_s, y_in)
print("=" * 65)
print("  OOD DETECTION: MSP vs ENERGY vs MAHALANOBIS")
print("=" * 65)
print()
print(f"  In-distribution: 4-class Gaussians, n={len(X_in)}, train acc={train_acc:.4f}")
print(f"  OOD near (between clusters):  n={n_ood_each}")
print(f"  OOD mid  (adjacent region):   n={n_ood_each}")
print(f"  OOD far  (distant region):    n={n_ood_each}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Compute logits (pre-softmax) for each dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_logits_and_features(clf_model, X):
    """Extract final layer logits and penultimate layer features."""
    # MLP forward pass — extract hidden representation
    h = X.copy()
    for i, (W, b) in enumerate(zip(clf_model.coefs_[:-1], clf_model.intercepts_[:-1])):
        h = np.maximum(0, h @ W + b)   # ReLU hidden layers
    features = h.copy()
    logits   = h @ clf_model.coefs_[-1] + clf_model.intercepts_[-1]
    return logits, features

log_in, feat_in   = get_logits_and_features(clf, X_in_s)
log_ns, feat_ns   = get_logits_and_features(clf, X_ood_ns)
log_ms, feat_ms   = get_logits_and_features(clf, X_ood_ms)
log_fs, feat_fs   = get_logits_and_features(clf, X_ood_fs)

# Mahalanobis: fit class-conditional Gaussians in feature space
class_means = [feat_in[y_in == c].mean(axis=0) for c in range(4)]
# Shared (pooled) covariance
centered    = np.vstack([feat_in[y_in == c] - class_means[c] for c in range(4)])
shared_cov  = (centered.T @ centered) / len(centered) + 1e-5 * np.eye(feat_in.shape[1])
shared_cov_inv = np.linalg.inv(shared_cov)

# ─────────────────────────────────────────────────────────────────────────────
# Score computation and evaluation
# ─────────────────────────────────────────────────────────────────────────────

methods = {
    "MSP (baseline)": lambda log, feat: msp_score(log),
    "Energy (T=1.0)": lambda log, feat: -energy_score(log, T=1.0),  # neg: higher=more in-dist
    "Mahalanobis":    lambda log, feat: mahalanobis_score(feat, class_means, shared_cov_inv),
}

ood_datasets = [("Near OOD", X_ood_ns, log_ns, feat_ns),
                ("Mid OOD",  X_ood_ms, log_ms, feat_ms),
                ("Far OOD",  X_ood_fs, log_fs, feat_fs)]

print(f"  {'Method':>20}  {'OOD Dataset':>12}  {'AUROC':>8}  {'FPR@95TPR':>12}")
print("  " + "─" * 58)

results_ood = {}
for method_name, scorer in methods.items():
    in_scores = scorer(log_in, feat_in)
    results_ood[method_name] = {"in": in_scores, "ood_scores": []}
    for ood_name, _, log_ood, feat_ood in ood_datasets:
        ood_scores = scorer(log_ood, feat_ood)
        auc  = auroc(in_scores, ood_scores)
        fpr  = fpr_at_tpr(in_scores, ood_scores, tpr_target=0.95)
        results_ood[method_name]["ood_scores"].append((ood_name, ood_scores, auc, fpr))
        print(f"  {method_name:>20}  {ood_name:>12}  {auc:>8.4f}  {fpr:>12.4f}")

print()
print("  KEY OBSERVATIONS:")
print("  - MSP: neural networks are overconfident → low AUROC on near OOD.")
print("  - Energy score: better calibrated, wider separation from OOD.")
print("  - Mahalanobis: uses feature space geometry, best on structured OOD.")
print("  - All methods improve on far OOD — distance effect dominates.")
print()

# Softmax probability examples for OOD inputs
print("  MSP OVERCONFIDENCE DEMONSTRATION:")
print(f"  {'Dataset':>18}  {'Mean max softmax':>18}  {'Mean energy':>13}")
print("  " + "─" * 52)
for dname, _, log_d, _ in [("In-distribution", None, log_in, None)] + list(
        (n, None, l, None) for n, _, l, _ in ood_datasets):
    msp_vals = msp_score(log_d)
    e_vals   = energy_score(log_d)
    print(f"  {dname:>18}  {msp_vals.mean():>18.4f}  {e_vals.mean():>13.4f}")

print()
print("  MSP barely drops from in-dist to near OOD — overconfidence!")
print("  Energy score shows cleaner separation across all OOD sets.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("OOD Detection: MSP vs Energy Score vs Mahalanobis Distance",
             fontsize=13, fontweight="bold")

colours_data = ["steelblue", "tomato", "seagreen", "purple"]

# Panel (0,0): Data distributions
ax = axes[0, 0]
for c, col in zip(range(4), colours_data):
    mask = y_in == c
    ax.scatter(X_in[mask, 0], X_in[mask, 1], c=col, s=10, alpha=0.5, label=f"Class {c}")
ax.scatter(X_ood_near[:, 0], X_ood_near[:, 1], c="black",  s=10, alpha=0.4,
           marker="^", label="Near OOD")
ax.scatter(X_ood_mid[:, 0],  X_ood_mid[:, 1],  c="gray",   s=10, alpha=0.4,
           marker="s", label="Mid OOD")
ax.scatter(X_ood_far[:, 0],  X_ood_far[:, 1],  c="orange", s=10, alpha=0.4,
           marker="D", label="Far OOD")
ax.set_title("In-Distribution vs OOD Datasets\\n(4 classes + 3 OOD sets)",
             fontweight="bold")
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

# Panel (0,1): Score distributions — MSP
ax = axes[0, 1]
in_msp = results_ood["MSP (baseline)"]["in"]
ax.hist(in_msp, bins=40, alpha=0.6, density=True, color="steelblue", label="In-distribution")
for (oname, ods, _, _), col in zip(results_ood["MSP (baseline)"]["ood_scores"],
                                     ["tomato", "seagreen", "orange"]):
    ax.hist(ods, bins=40, alpha=0.5, density=True, color=col, label=oname)
ax.set_title("MSP Score Distribution\\n(in vs OOD — poor separation near OOD)",
             fontweight="bold")
ax.set_xlabel("MSP score"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (0,2): Score distributions — Energy
ax = axes[0, 2]
in_en = results_ood["Energy (T=1.0)"]["in"]
ax.hist(in_en, bins=40, alpha=0.6, density=True, color="steelblue", label="In-distribution")
for (oname, ods, _, _), col in zip(results_ood["Energy (T=1.0)"]["ood_scores"],
                                     ["tomato", "seagreen", "orange"]):
    ax.hist(ods, bins=40, alpha=0.5, density=True, color=col, label=oname)
ax.set_title("Energy Score Distribution\\n(better separation than MSP)",
             fontweight="bold")
ax.set_xlabel("−Energy (higher = more in-dist)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,0): Mahalanobis score distributions
ax = axes[1, 0]
in_maha = results_ood["Mahalanobis"]["in"]
ax.hist(in_maha, bins=40, alpha=0.6, density=True, color="steelblue", label="In-distribution")
for (oname, ods, _, _), col in zip(results_ood["Mahalanobis"]["ood_scores"],
                                     ["tomato", "seagreen", "orange"]):
    ax.hist(ods, bins=40, alpha=0.5, density=True, color=col, label=oname)
ax.set_title("Mahalanobis Distance Distribution\\n(uses feature space geometry)",
             fontweight="bold")
ax.set_xlabel("−Mahalanobis distance (higher = more in-dist)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,1): AUROC comparison bar chart
ax = axes[1, 1]
method_names = list(results_ood.keys())
ood_names    = [r[0] for r in results_ood[method_names[0]]["ood_scores"]]
x_pos        = np.arange(len(ood_names))
width        = 0.25
bar_colours  = ["steelblue", "tomato", "seagreen"]
for i, (mname, col) in enumerate(zip(method_names, bar_colours)):
    aucs = [r[2] for r in results_ood[mname]["ood_scores"]]
    ax.bar(x_pos + i*width, aucs, width, color=col, alpha=0.8, label=mname)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(ood_names, fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("AUROC")
ax.set_title("AUROC Comparison by OOD Distance\\n(higher=better OOD detection)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax.axhline(0.5, color="gray", lw=1, ls="--", label="Random")

# Panel (1,2): FPR@95TPR heatmap
ax = axes[1, 2]
fpr_matrix = np.array([[r[3] for r in results_ood[mname]["ood_scores"]]
                         for mname in method_names])
im = ax.imshow(fpr_matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="FPR@95TPR (lower=better)")
ax.set_xticks(range(len(ood_names))); ax.set_xticklabels(ood_names, fontsize=9)
ax.set_yticks(range(len(method_names))); ax.set_yticklabels(method_names, fontsize=8)
for i in range(len(method_names)):
    for j in range(len(ood_names)):
        ax.text(j, i, f"{fpr_matrix[i,j]:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if fpr_matrix[i,j] > 0.6 else "black")
ax.set_title("FPR @ 95% TPR Heatmap\\n(lower = better OOD detection)",
             fontweight="bold")

plt.tight_layout()
plt.savefig("ood_detection.png", dpi=110)
print()
print("  Plot saved → ood_detection.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Adversarial Robustness — FGSM, PGD, and Adversarial Training": {
        "description": (
            "Implements FGSM and PGD adversarial attacks from scratch on a "
            "toy classifier. Shows the attack success rate as a function of "
            "perturbation budget ε. Implements adversarial training (inner "
            "maximisation via PGD, outer minimisation via SGD). Compares "
            "clean accuracy, FGSM accuracy, and PGD accuracy for standard "
            "vs adversarially trained models. Demonstrates the clean accuracy "
            "vs robustness tradeoff."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def roc_auc_score(y_true,y_score):
    yt=_np_impl.asarray(y_true); ys=_np_impl.asarray(y_score)
    desc=_np_impl.argsort(ys)[::-1]; yt=yt[desc]
    tp=_np_impl.cumsum(yt); fp=_np_impl.cumsum(1-yt)
    tp_r=tp/tp[-1]; fp_r=fp/fp[-1]
    return float(_np_impl.trapezoid(tp_r,fp_r))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; lam=1.0/(self.C*n)
        w=_np_impl.ones(n) if sample_weight is None else _np_impl.asarray(sample_weight)
        w=w/w.sum()*n
        def fg(params):
            p=_np_impl.clip(_sig(X@params[1:]+params[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(w*(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p)))+0.5*lam*_np_impl.sum(params[1:]**2)
            e=w*(p-y)
            return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*params[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return accuracy_score(y,self.predict(X))

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
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict_proba(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        p=_sig(F); return _np_impl.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)

class SGDClassifier:
    def __init__(self,loss='log_loss',random_state=None,warm_start=False,alpha=0.0001,**kw):
        self.alpha=alpha; self._w=None; self._b=0.0; self._t=1
    def partial_fit(self,X,y,classes=None):
        X=_np_impl.atleast_2d(X).astype(float); y=_np_impl.asarray(y)
        if self._w is None: self._w=_np_impl.zeros(X.shape[1])
        lr=min(1.0/(self.alpha*self._t+1e-6),0.1)
        for xi,yi in zip(X,y):
            p=_sig(xi@self._w+self._b); e=p-yi
            self._w-=lr*(e*xi+self.alpha*self._w); self._b-=lr*e; self._t+=1
        return self
    def predict(self,X):
        X=_np_impl.atleast_2d(X).astype(float)
        return (_sig(X@self._w+self._b)>=0.5).astype(int)

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),max_iter=200,random_state=None,
                 activation='relu',alpha=0.0001,**kw):
        self.hidden_layer_sizes=hidden_layer_sizes; self.max_iter=max_iter
        self.random_state=random_state; self.alpha=alpha
    def fit(self,X,y):
        rng=_np_impl.random.default_rng(self.random_state)
        self._classes=_np_impl.unique(y); nc=len(self._classes)
        dims=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,_np_impl.sqrt(2.0/dims[i]),(dims[i],dims[i+1]))
                     for i in range(len(dims)-1)]
        self.intercepts_=[_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n=len(X); lr=1e-3
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for s in range(0,n,32):
                xb=X[idx[s:s+32]]; yb=y[idx[s:s+32]]; nb=len(xb)
                acts=[xb]
                for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+b
                    acts.append(_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z)
                logits=acts[-1]-acts[-1].max(1,keepdims=True)
                exp=_np_impl.exp(logits); probs=exp/exp.sum(1,keepdims=True)
                oh=_np_impl.zeros_like(probs)
                for ci,c in enumerate(self._classes): oh[yb==c,ci]=1
                delta=(probs-oh)/nb
                for i in range(len(self.coefs_)-1,-1,-1):
                    self.coefs_[i]-=lr*(acts[i].T@delta+self.alpha*self.coefs_[i])
                    self.intercepts_[i]-=lr*delta.sum(0)
                    if i>0: delta=(delta@self.coefs_[i].T)*(acts[i]>0)
        return self
    def predict_proba(self,X):
        a=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=a@W+b; a=_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z
        a=a-a.max(1,keepdims=True); e=_np_impl.exp(a); return e/e.sum(1,keepdims=True)
    def predict(self,X): return self._classes[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return accuracy_score(y,self.predict(X))

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=n_samples//2
    t=_np_impl.linspace(0,_np_impl.pi,n)
    X=_np_impl.vstack([_np_impl.c_[_np_impl.cos(t),_np_impl.sin(t)],
                       _np_impl.c_[1-_np_impl.cos(t),-_np_impl.sin(t)+0.5]])
    if noise>0: X+=rng.normal(0,noise,X.shape)
    return X,_np_impl.hstack([_np_impl.zeros(n),_np_impl.ones(n_samples-n)]).astype(int)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Numpy MLP — needed so we can compute gradients manually
# ─────────────────────────────────────────────────────────────────────────────

class TwoLayerNet:
    """2-hidden-layer binary classifier with manual gradient computation."""
    def __init__(self, d=2, h=64, seed=0):
        rng  = np.random.default_rng(seed)
        s1   = np.sqrt(2 / d); s2 = np.sqrt(2 / h)
        self.W1 = rng.normal(0, s1, (d, h)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, s2, (h, h)); self.b2 = np.zeros(h)
        self.W3 = rng.normal(0, s2, (h, 1)); self.b3 = np.zeros(1)

    def _relu(self, x): return np.maximum(0, x)
    def _drelu(self, x): return (x > 0).astype(float)
    def _sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, X):
        self.X   = X
        self.z1  = X @ self.W1 + self.b1
        self.h1  = self._relu(self.z1)
        self.z2  = self.h1 @ self.W2 + self.b2
        self.h2  = self._relu(self.z2)
        self.z3  = self.h2 @ self.W3 + self.b3
        self.out = self._sigmoid(self.z3).ravel()
        return self.out

    def loss(self, X, y):
        p = self.forward(X)
        return -np.mean(y * np.log(p + 1e-15) + (1-y) * np.log(1-p + 1e-15))

    def grad_input(self, X, y):
        """Gradient of loss w.r.t. input X — used for adversarial attacks."""
        p   = self.forward(X)
        dl  = (p - y).reshape(-1, 1) / len(y)
        dz3 = dl
        dh2 = dz3 @ self.W3.T
        dz2 = dh2 * self._drelu(self.z2)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * self._drelu(self.z1)
        dX  = dz1 @ self.W1.T
        return dX

    def grad_params(self, X, y):
        """Gradient of loss w.r.t. all parameters."""
        p   = self.forward(X)
        dl  = (p - y).reshape(-1, 1) / len(y)
        dz3 = dl
        dW3 = self.h2.T @ dz3;  db3 = dz3.sum(0)
        dh2 = dz3 @ self.W3.T
        dz2 = dh2 * self._drelu(self.z2)
        dW2 = self.h1.T @ dz2;  db2 = dz2.sum(0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * self._drelu(self.z1)
        dW1 = X.T @ dz1;        db1 = dz1.sum(0)
        return [(dW1,db1),(dW2,db2),(dW3,db3)]

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def train(self, X, y, epochs=300, lr=5e-3, batch=64, seed=0):
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                Xb = X[idx[s:s+batch]]; yb = y[idx[s:s+batch]]
                grads = self.grad_params(Xb, yb)
                for (W, b), (dW, db) in zip(
                        [(self.W1,self.b1),(self.W2,self.b2),(self.W3,self.b3)],
                        grads):
                    W -= lr * dW; b -= lr * db

# ─────────────────────────────────────────────────────────────────────────────
# Adversarial attacks
# ─────────────────────────────────────────────────────────────────────────────

def fgsm_attack(model, X, y, eps):
    """Fast Gradient Sign Method: δ = ε · sign(∇_x L)."""
    g = model.grad_input(X, y)
    return np.clip(X + eps * np.sign(g), -5, 5)

def pgd_attack(model, X, y, eps, alpha=None, n_steps=20, seed=0):
    """
    PGD attack: projected gradient descent in ∞-norm ball.
    δ_{t+1} = Π_{‖δ‖∞≤ε} [δ_t + α · sign(∇_x L(x+δ_t, y))]
    """
    if alpha is None:
        alpha = eps / 4
    rng  = np.random.default_rng(seed)
    X_adv = X + rng.uniform(-eps, eps, X.shape)   # random start
    X_adv = np.clip(X_adv, -5, 5)
    for _ in range(n_steps):
        g     = model.grad_input(X_adv, y)
        X_adv = X_adv + alpha * np.sign(g)
        X_adv = np.clip(X_adv, X - eps, X + eps)   # project to ε-ball
        X_adv = np.clip(X_adv, -5, 5)
    return X_adv

def adv_train_step(model, X, y, eps, lr=5e-3, n_pgd=7):
    """One step of adversarial training: inner PGD then outer SGD."""
    X_adv = pgd_attack(model, X, y, eps=eps, n_steps=n_pgd)
    grads = model.grad_params(X_adv, y)
    for (W, b), (dW, db) in zip(
            [(model.W1, model.b1), (model.W2, model.b2), (model.W3, model.b3)],
            grads):
        W -= lr * dW;  b -= lr * db

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Standard training + FGSM/PGD attacks
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  ADVERSARIAL ROBUSTNESS: FGSM, PGD, ADVERSARIAL TRAINING")
print("=" * 65)
print()

X_raw, y = make_moons(n_samples=600, noise=0.2, random_state=0)
sc  = StandardScaler().fit(X_raw)
X   = sc.transform(X_raw)
X_tr, X_te = X[:400], X[400:]
y_tr, y_te = y[:400], y[400:]

# Standard training
print("  Training standard (non-robust) model...")
net_std = TwoLayerNet(d=2, h=64, seed=0)
net_std.train(X_tr, y_tr, epochs=400, lr=5e-3)
acc_clean = accuracy_score(y_te, net_std.predict(X_te))
print(f"  Clean test accuracy: {acc_clean:.4f}")
print()

# Attack at various ε
print(f"  {'ε':>8}  {'Clean acc':>10}  {'FGSM acc':>10}  {'PGD-20 acc':>12}  {'PGD success':>13}")
print("  " + "─" * 58)

eps_vals = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
attack_results = []

for eps in eps_vals:
    if eps == 0:
        acc_f = acc_clean; acc_p = acc_clean
    else:
        X_fgsm = fgsm_attack(net_std, X_te, y_te, eps)
        X_pgd  = pgd_attack(net_std,  X_te, y_te, eps, n_steps=20)
        acc_f  = accuracy_score(y_te, net_std.predict(X_fgsm))
        acc_p  = accuracy_score(y_te, net_std.predict(X_pgd))
    pgd_sr = 1 - acc_p   # attack success rate
    attack_results.append((eps, acc_clean, acc_f, acc_p, pgd_sr))
    print(f"  {eps:>8.2f}  {acc_clean:>10.4f}  {acc_f:>10.4f}  "
          f"{acc_p:>12.4f}  {pgd_sr:>13.4f}")

print()
print("  - FGSM (1-step) degrades performance quickly with ε.")
print("  - PGD (20 steps) is a stronger attack — lower accuracy than FGSM.")
print("  - At ε=0.3, PGD reduces accuracy near chance level for standard model.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Adversarial training — inner PGD, outer SGD
# ─────────────────────────────────────────────────────────────────────────────

print()
print("  ADVERSARIAL TRAINING (PGD-7 inner attack, ε=0.3)...")
eps_train = 0.3
net_adv   = TwoLayerNet(d=2, h=64, seed=1)
rng_at    = np.random.default_rng(7)

for epoch in range(300):
    idx = rng_at.permutation(len(X_tr))
    for s in range(0, len(X_tr), 64):
        Xb = X_tr[idx[s:s+64]]; yb = y_tr[idx[s:s+64]]
        adv_train_step(net_adv, Xb, yb, eps=eps_train, lr=5e-3, n_pgd=5)
    if (epoch+1) % 100 == 0:
        acc_c_adv = accuracy_score(y_te, net_adv.predict(X_te))
        X_pgd_adv = pgd_attack(net_adv, X_te, y_te, eps=eps_train, n_steps=20)
        acc_p_adv = accuracy_score(y_te, net_adv.predict(X_pgd_adv))
        print(f"    Epoch {epoch+1}: clean={acc_c_adv:.4f}  "
              f"PGD-robust={acc_p_adv:.4f}")

# Final evaluation
print()
print("  FINAL COMPARISON: Standard vs Adversarially Trained Model:")
print(f"  {'Model':>25}  {'Clean acc':>10}  {'FGSM acc':>10}  {'PGD-20 acc':>12}")
print("  " + "─" * 62)

for eps_eval in [0.1, 0.2, 0.3, 0.5]:
    X_fgsm_s = fgsm_attack(net_std, X_te, y_te, eps_eval)
    X_pgd_s  = pgd_attack(net_std,  X_te, y_te, eps_eval, n_steps=20)
    X_fgsm_a = fgsm_attack(net_adv, X_te, y_te, eps_eval)
    X_pgd_a  = pgd_attack(net_adv,  X_te, y_te, eps_eval, n_steps=20)

    print(f"  ε={eps_eval}:")
    print(f"  {'Standard':>25}  {acc_clean:>10.4f}  "
          f"{accuracy_score(y_te, net_std.predict(X_fgsm_s)):>10.4f}  "
          f"{accuracy_score(y_te, net_std.predict(X_pgd_s)):>12.4f}")
    acc_clean_adv = accuracy_score(y_te, net_adv.predict(X_te))
    print(f"  {'Adversarially trained':>25}  {acc_clean_adv:>10.4f}  "
          f"{accuracy_score(y_te, net_adv.predict(X_fgsm_a)):>10.4f}  "
          f"{accuracy_score(y_te, net_adv.predict(X_pgd_a)):>12.4f}")
    print()

print("  - Adversarial training recovers much of the robustness under attack.")
print("  - Cost: some clean accuracy drop (~5-15%).")
print("  - This is the fundamental robustness-accuracy tradeoff.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Adversarial Robustness: FGSM, PGD Attacks, and Adversarial Training",
             fontsize=13, fontweight="bold")

# Panel (0,0): Clean decision boundary
def plot_boundary_simple(ax, model, X, y, title):
    h = 0.04
    x0r = X[:,0].min()-0.5, X[:,0].max()+0.5
    x1r = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx0, xx1 = np.meshgrid(np.arange(x0r[0],x0r[1],h), np.arange(x1r[0],x1r[1],h))
    Z = model.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)
    ax.contourf(xx0, xx1, Z, alpha=0.3, cmap="RdBu")
    ax.contour(xx0, xx1, Z, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(X[y==0,0], X[y==0,1], c="tomato",    s=15, alpha=0.6)
    ax.scatter(X[y==1,0], X[y==1,1], c="steelblue", s=15, alpha=0.6)
    ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.3)

plot_boundary_simple(axes[0,0], net_std, X_te, y_te,
                     f"Standard Model (clean acc={acc_clean:.3f})")

# Panel (0,1): Adversarial examples (ε=0.3)
ax = axes[0, 1]
eps_vis = 0.3
X_pgd_vis = pgd_attack(net_std, X_te[:80], y_te[:80], eps=eps_vis, n_steps=20)
ax.scatter(X_te[:80,0], X_te[:80,1], c="steelblue", s=30, alpha=0.5,
           label="Original", zorder=5)
ax.scatter(X_pgd_vis[:,0], X_pgd_vis[:,1], c="tomato", s=30, alpha=0.5,
           marker="^", label="PGD adversarial", zorder=5)
for i in range(0, 15, 3):
    ax.annotate("", xy=X_pgd_vis[i], xytext=X_te[i],
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
ax.set_title(f"PGD Adversarial Examples (ε={eps_vis})\\n"
             f"(arrows show perturbation direction)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (0,2): Attack strength vs accuracy
ax = axes[0, 2]
eps_list  = [r[0] for r in attack_results]
clean_a   = [r[1] for r in attack_results]
fgsm_a    = [r[2] for r in attack_results]
pgd_a     = [r[3] for r in attack_results]
ax.plot(eps_list, clean_a, "steelblue", lw=2.5, marker="o", ms=7, label="Clean")
ax.plot(eps_list, fgsm_a,  "seagreen",  lw=2.5, marker="s", ms=7, label="FGSM")
ax.plot(eps_list, pgd_a,   "tomato",    lw=2.5, marker="^", ms=7, label="PGD-20")
ax.axhline(0.5, color="gray", lw=1.5, ls="--", label="Chance")
ax.set_xlabel("Perturbation budget ε")
ax.set_ylabel("Test accuracy")
ax.set_title("Attack Success vs Perturbation Budget\\n(PGD > FGSM, both degrade rapidly)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,0): Adversarially trained decision boundary
plot_boundary_simple(axes[1,0], net_adv, X_te, y_te,
                     f"Adversarially Trained Model\\n"
                     f"(clean acc={accuracy_score(y_te, net_adv.predict(X_te)):.3f})")

# Panel (1,1): Robustness comparison across ε
ax = axes[1, 1]
eps_comp = [0.05, 0.1, 0.2, 0.3, 0.5]
pgd_std_accs = []
pgd_adv_accs = []
for e in eps_comp:
    Xp_s = pgd_attack(net_std, X_te, y_te, e, n_steps=20)
    Xp_a = pgd_attack(net_adv, X_te, y_te, e, n_steps=20)
    pgd_std_accs.append(accuracy_score(y_te, net_std.predict(Xp_s)))
    pgd_adv_accs.append(accuracy_score(y_te, net_adv.predict(Xp_a)))
ax.plot(eps_comp, pgd_std_accs, "tomato",    lw=2.5, marker="o", ms=7,
        label="Standard (PGD)")
ax.plot(eps_comp, pgd_adv_accs, "steelblue", lw=2.5, marker="s", ms=7,
        label="Adv. trained (PGD)")
ax.axhline(accuracy_score(y_te, net_std.predict(X_te)), color="tomato",
           lw=1, ls=":", alpha=0.7, label="Standard clean")
ax.axhline(accuracy_score(y_te, net_adv.predict(X_te)), color="steelblue",
           lw=1, ls=":", alpha=0.7, label="Adv. trained clean")
ax.set_xlabel("ε"); ax.set_ylabel("Accuracy under PGD")
ax.set_title("Robustness Comparison: Standard vs Adversarial Training\\n"
             "(dotted=clean accuracy; solid=under attack)", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,2): Clean vs robust accuracy tradeoff
ax = axes[1, 2]
ax.axis("off")
rows = [
    ["Standard training", f"{accuracy_score(y_te, net_std.predict(X_te)):.3f}",
     f"{accuracy_score(y_te, net_std.predict(pgd_attack(net_std,X_te,y_te,0.3,n_steps=20))):.3f}",
     "fast, no overhead"],
    ["Adv. training (PGD-7)", f"{accuracy_score(y_te, net_adv.predict(X_te)):.3f}",
     f"{accuracy_score(y_te, net_adv.predict(pgd_attack(net_adv,X_te,y_te,0.3,n_steps=20))):.3f}",
     "7× slower, robust"],
    ["Randomised Smoothing", "−5-15%", "cert. guarantee", "σ controls tradeoff"],
    ["ODIN / Energy score", "no change", "detect OOD only", "inference only"],
]
headers = ["Method", "Clean acc", "PGD acc (ε=0.3)", "Notes"]
table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 2.2)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif r == 2 and c == 2:
        cell.set_facecolor("#e8f5e8")
ax.set_title("Adversarial Defence Methods Summary", fontweight="bold")

plt.tight_layout()
plt.savefig("adversarial_robustness.png", dpi=110)
print()
print("  Plot saved → adversarial_robustness.png")
print()
print("  KEY TAKEAWAYS — DISTRIBUTION SHIFT & ROBUSTNESS:")
print("  1. Covariate shift: P(X) changes, P(Y|X) fixed → fix with importance weighting.")
print("  2. Concept drift: P(Y|X) changes → detect with DDM/KS, then retrain.")
print("  3. Label shift: P(Y) changes → correct class prior, not features.")
print("  4. Softmax probability is overconfident on OOD — use energy or Mahalanobis.")
print("  5. FGSM/PGD adversarial attacks degrade standard models rapidly.")
print("  6. Adversarial training is the strongest known defence — with an accuracy cost.")
print("  7. There is no free lunch: robustness and clean accuracy trade off.")
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