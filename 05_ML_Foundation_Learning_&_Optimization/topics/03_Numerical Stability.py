"""
Model Evaluation Metrics
=========================

How to measure whether a model is actually good — the full taxonomy of
metrics for classification, regression, and ranking; why accuracy alone
is almost always the wrong metric; and how to choose, interpret, and
avoid being misled by the numbers your evaluation pipeline produces.

"""

import textwrap
import re

TOPIC_NAME = "Model Evaluation Metrics"
DISPLAY_NAME = "01 · Model Evaluation Metrics"
ICON = "📊"
SUBTITLE = "Measuring What Actually Matters"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Question: What Does "Good" Mean?

A trained model produces predictions. An evaluation metric translates the
gap between those predictions and reality into a single number. Choosing
the wrong metric can mislead you into deploying a model that fails in
production, or discarding a model that would have worked perfectly.

The metric you optimise during training is the LOSS FUNCTION. The metric
you use to judge final quality is the EVALUATION METRIC. These are almost
always different things, and understanding why is the foundation of
rigorous ML evaluation.

    Diagram 1 — The Evaluation Hierarchy:

    ┌───────────────────────────────────────────────────────────────────┐
    │  BUSINESS OBJECTIVE           "Reduce fraud losses by 20%"        │
    │           ↑                                                       │
    │  EVALUATION METRIC            F2-score, AUC-PR, $ saved           │
    │           ↑                                                       │
    │  PROXY METRIC                 Precision, Recall, AUC-ROC          │
    │           ↑                                                       │
    │  TRAINING LOSS               Binary Cross-Entropy                 │
    │                                                                   │
    │  Each level approximates the one above.                           │
    │  Optimising the wrong level creates a misaligned model.           │
    └───────────────────────────────────────────────────────────────────┘

    The three questions every metric must answer:

        1. WHAT does it measure?      (precision, recall, error magnitude)
        2. WHEN is it appropriate?    (balanced classes, imbalanced, ranked)
        3. WHAT can it NOT see?       (its blind spots and failure modes)

    Metric taxonomy:
    ┌──────────────────────────────┬──────────────────────────────────────┐
    │ Task Type                    │ Key Metrics                          │
    ├──────────────────────────────┼──────────────────────────────────────┤
    │ Binary Classification        │ Accuracy, Precision, Recall, F1,     │
    │                              │ F-beta, AUC-ROC, AUC-PR, MCC         │
    │ Multiclass Classification    │ Macro/Micro/Weighted F1, Cohen's κ,  │
    │                              │ Confusion Matrix, Top-K Accuracy     │
    │ Regression                   │ MAE, MSE, RMSE, R², MAPE, Huber      │
    │ Ranking / Retrieval          │ NDCG, MAP, MRR, Precision@K          │
    │ Probabilistic                │ Log-loss, Brier Score, Calibration   │
    │ Object Detection             │ mAP, IoU, AP50, AP75                 │
    │ NLP Generation               │ BLEU, ROUGE, METEOR, BERTScore       │
    └──────────────────────────────┴──────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### The Confusion Matrix — The Foundation of Classification Metrics

All binary classification metrics are derived from four numbers that
describe every possible relationship between prediction and ground truth.

    Ground truth positive (y=1):
        Model predicts 1 → TRUE POSITIVE  (TP)  ← correct, caught it
        Model predicts 0 → FALSE NEGATIVE (FN)  ← missed it (Type II error)

    Ground truth negative (y=0):
        Model predicts 0 → TRUE NEGATIVE  (TN)  ← correct, ignored it
        Model predicts 1 → FALSE POSITIVE (FP)  ← false alarm (Type I error)

    Diagram 2 — The Confusion Matrix:

                            PREDICTED
                         Positive   Negative
                       ┌──────────┬──────────┐
    ACTUAL  Positive   │    TP    │    FN    │  ← All actual positives
                       ├──────────┼──────────┤
            Negative   │    FP    │    TN    │  ← All actual negatives
                       └──────────┴──────────┘

    Derived quantities:
        Total positives:     P = TP + FN
        Total negatives:     N = FP + TN
        Total predictions:   n = P + N = TP + FP + TN + FN

    The two fundamental error rates:
        False Positive Rate: FPR = FP / N = FP / (FP + TN)   ← among negatives
        False Negative Rate: FNR = FN / P = FN / (TP + FN)   ← among positives

    The practical meaning of each cell:
    ┌────────────────────────┬─────────────────────────────────────────────┐
    │ Cell                   │ Real-world meaning                          │
    ├────────────────────────┼─────────────────────────────────────────────┤
    │ TP (True Positive)     │ Fraud caught, cancer detected, spam blocked │
    │ FP (False Positive)    │ Legitimate email in spam, false arrest      │
    │ FN (False Negative)    │ Fraud missed, cancer undetected             │
    │ TN (True Negative)     │ Legitimate transaction approved correctly   │
    └────────────────────────┴─────────────────────────────────────────────┘

    The asymmetry of errors:
    ┌────────────────────────────────────────────────────────────────────┐
    │  In almost every real application, FP and FN have DIFFERENT        │
    │  costs. Medical diagnosis: FN (missed cancer) >> FP (false alarm). │
    │  Spam filter: FP (lost email) > FN (spam in inbox) for most users. │
    │  Fraud detection: FN (fraud missed) >> FP (blocked transaction).   │
    │                                                                    │
    │  Any metric that treats FP and FN symmetrically is likely wrong    │
    │  for your specific use case.                                       │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Core Classification Metrics

**Accuracy** — the fraction of all predictions that are correct:

        Accuracy = (TP + TN) / (TP + FP + TN + FN)

    When to use: ONLY when classes are roughly balanced AND
    both FP and FN carry equal cost.

    Why accuracy fails on imbalanced data:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Dataset: 99% negative, 1% positive (e.g., rare disease)          │
    │  A model that predicts "negative" for every sample:               │
    │      Accuracy = 99%   ← sounds excellent                          │
    │      Recall   = 0%    ← catches zero actual cases                 │
    │  Accuracy of 99% is completely misleading here.                   │
    └───────────────────────────────────────────────────────────────────┘

**Precision** — of everything the model labelled positive, how many truly were?

        Precision = TP / (TP + FP)   ← "When you cry wolf, are you right?"

    High precision → few false alarms.
    Use when FP cost is high: spam filter, recommendation, content moderation.

**Recall (Sensitivity, True Positive Rate)** — of all true positives, how
many did the model catch?

        Recall = TP / (TP + FN)   ← "Of all actual wolves, how many found?"

    High recall → few misses.
    Use when FN cost is high: cancer screening, fraud detection, safety systems.

**The Precision-Recall Tradeoff:**

    Every classifier has a decision threshold τ. Lower τ → predict positive
    more aggressively → more TP but also more FP → higher recall, lower precision.

    Diagram 3 — The Precision-Recall Tradeoff:

    Precision
    │
    1.0 │ ×
    0.9 │   ×
    0.8 │     ×
    0.7 │       ×
    0.6 │         ×   ← high τ (conservative, few predictions)
    0.5 │           ×
    0.4 │               ×
    0.3 │                   ×
    0.2 │                       ×
    0.1 │                           ×
        └────────────────────────────── Recall
              0.2  0.4  0.6  0.8  1.0

    As τ decreases: more positive predictions → recall increases,
                    but more FPs accumulate → precision decreases.

**Specificity (True Negative Rate):**

        Specificity = TN / (TN + FP) = 1 − FPR

    The recall for the negative class. Useful when correctly identifying
    negatives is also important (e.g., ruling out disease in healthy patients).


──────────────────────────────────────────────────────────────────────────────
### F-Scores — Combining Precision and Recall

**F1-Score** — the harmonic mean of precision and recall:

        F1 = 2 · (Precision × Recall) / (Precision + Recall)
           = 2·TP / (2·TP + FP + FN)

    Why the HARMONIC mean and not arithmetic?

        Arithmetic mean (0.9 + 0.1) / 2 = 0.5   ← tolerates one being 0.1
        Harmonic mean  2(0.9 × 0.1)/(0.9+0.1) = 0.18  ← penalises imbalance

    The harmonic mean is dominated by the smaller value.
    A model with Precision=0.9, Recall=0.1 is useless — F1 correctly reflects this.

    ┌───────────────────────────────────────────────────────────────────┐
    │  F1 = 1.0  → perfect precision AND perfect recall                 │
    │  F1 = 0.0  → either precision or recall is zero                   │
    │  F1 = 0.5  → example: P=0.33, R=1.0 (always predict positive)     │
    └───────────────────────────────────────────────────────────────────┘

**F-Beta Score** — weighted harmonic mean that allows controlling the tradeoff:

        F_β = (1 + β²) · (Precision × Recall) / (β² · Precision + Recall)

    β < 1:  precision-weighted (β=0.5: precision twice as important as recall)
    β = 1:  balanced (F1-score)
    β > 1:  recall-weighted (β=2: recall twice as important as precision)

    Choosing β based on cost asymmetry:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Medical diagnosis (FN >> FP):  use F2  (β=2, recall-weighted)       │
    │  Spam filter (FP >> FN):        use F0.5 (β=0.5, precision-weighted) │
    │  Balanced costs:                use F1  (β=1, equal weight)          │
    └──────────────────────────────────────────────────────────────────────┘

**Matthews Correlation Coefficient (MCC):**

        MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Range: −1 (completely wrong) to +1 (perfect) with 0 = random.

    MCC is considered the most informative single-number summary of a
    binary confusion matrix because it uses ALL four cells symmetrically
    and is robust to class imbalance.

    ┌────────────────────────────────────────────────────────────────────┐
    │  For imbalanced datasets, prefer MCC or F1 over accuracy.          │
    │  MCC is preferred over F1 when you also care about TN performance. │
    │  F1 is preferred when you only care about the positive class.      │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### ROC Curves and AUC-ROC

The Receiver Operating Characteristic (ROC) curve plots the True Positive
Rate (recall) against the False Positive Rate (1 − specificity) at every
possible decision threshold τ from 0 to 1.

    Diagram 4 — ROC Curve Anatomy:

    TPR (Recall)
    1.0 │                   ╭────────────────× (1,1)
        │             ╭─────╯
    0.8 │         ╭───╯   ← Model A (good, AUC≈0.85)
        │     ╭───╯
    0.6 │   ╭─╯
        │ ╭─╯
    0.4 │ │      ← Model B (AUC≈0.65, mediocre)
        │ │
    0.2 │ │
        │ │   ← Random classifier (AUC=0.50, diagonal)
    0.0 ×─────────────────────────────────────────── FPR
        0.0  0.2  0.4  0.6  0.8  1.0

    Key reference points:
        (0, 0): τ = 1.0  → predict nothing positive (no FP, no TP)
        (1, 1): τ = 0.0  → predict everything positive (all TP, all FP)
        (0, 1): τ = perfect → all TP, zero FP  (ideal point)
        diagonal: random classifier

    AUC-ROC — Area Under the ROC Curve:
        Equivalent interpretation: the probability that a randomly chosen
        positive example is ranked higher than a randomly chosen negative.

            AUC = P(score(x⁺) > score(x⁻))

    AUC-ROC interpretation:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  AUC = 1.00  Perfect ranking. No positive ranked below any negative  │
    │  AUC = 0.90  Excellent. 90% of pos-neg pairs correctly ranked        │
    │  AUC = 0.70  Moderate. Reasonable discrimination                     │
    │  AUC = 0.50  Random. No discrimination ability at all                │
    │  AUC < 0.50  Worse than random (predictions inverted)                │
    └──────────────────────────────────────────────────────────────────────┘

    When AUC-ROC misleads — the imbalanced class problem:

    AUC-ROC can look great even on severely imbalanced datasets because
    TN contributes to FPR. With 10,000 negatives and 100 positives, even
    a bad model that misclassifies all positives may show AUC=0.85 because
    it correctly ranks the large negative class.


──────────────────────────────────────────────────────────────────────────────
### Precision-Recall Curves and AUC-PR

The Precision-Recall curve plots Precision vs Recall at every threshold.
Unlike ROC, it ignores TN entirely — it is a metric ONLY about how well
the model handles the positive class.

    Diagram 5 — PR Curve vs ROC Curve on Imbalanced Data:

    AUC-ROC appears good:           AUC-PR reveals the truth:

    TPR                             Precision
    │                               │
    │    ╭────────────────          │ ×
    │  ╭─╯      AUC=0.85            │   ╲
    │ ╱                             │     ╲____
    │/                              │          ╲____
    └─────────────── FPR            └──────────────── Recall
    "Looks great"                  "Terrible at finding positives"

    AUC-PR interpretation:
        Baseline (random classifier): AUC-PR = prevalence = P/(P+N)
        For 1% positive rate: random AUC-PR = 0.01

    ┌──────────────────────────────────────────────────────────────────┐
    │  RULE: For imbalanced datasets (< 10% positive class):           │
    │  ALWAYS use AUC-PR as your primary metric, not AUC-ROC.          │
    │  AUC-ROC inflates performance when negatives dominate.           │
    └──────────────────────────────────────────────────────────────────┘

    Average Precision (AP) — the area under the PR curve, computed as
    the weighted mean of precisions at each threshold:

        AP = Σ_n (Recall_n − Recall_{n-1}) · Precision_n


──────────────────────────────────────────────────────────────────────────────
### Multiclass Classification Metrics

Extending binary metrics to K classes requires choosing how to aggregate
performance across classes. Three aggregation strategies exist:

    Diagram 6 — Multiclass Aggregation Strategies (K=3 classes):

    Per-class metrics computed independently:
        Class A: Precision_A = TP_A / (TP_A + FP_A),  Recall_A = ...
        Class B: Precision_B = TP_B / (TP_B + FP_B),  Recall_B = ...
        Class C: Precision_C = TP_C / (TP_C + FP_C),  Recall_C = ...

    MACRO averaging — unweighted mean across classes:
        F1_macro = (F1_A + F1_B + F1_C) / 3
        → Every class contributes EQUALLY regardless of size.
        → Use when all classes matter equally (rare disease detection).
        → Sensitive to poor performance on small classes.

    MICRO averaging — sum all TP, FP, FN then compute metric:
        F1_micro = 2·ΣTP / (2·ΣTP + ΣFP + ΣFN)
        → Dominated by the LARGEST class (weighted by sample count).
        → Use when overall correct classification rate matters.
        → Equivalent to accuracy on balanced datasets.

    WEIGHTED averaging — weighted mean, weight = class frequency:
        F1_weighted = Σ_k (nk/n) · F1_k
        → Accounts for class imbalance in the weighting.
        → Use for reporting on imbalanced datasets.

    When to use each:
    ┌──────────────┬─────────────────────────────────────────────────────┐
    │ Strategy     │ Use When                                            │
    ├──────────────┼─────────────────────────────────────────────────────┤
    │ Macro        │ Every class matters equally; you care about rare    │
    │              │ classes as much as common ones                      │
    │ Micro        │ Overall accuracy-like measure; large classes are    │
    │              │ more important by virtue of being more common       │
    │ Weighted     │ Imbalanced dataset, but importance scales with      │
    │              │ class frequency; standard reporting metric          │
    └──────────────┴─────────────────────────────────────────────────────┘

    Cohen's Kappa — measures agreement beyond chance:

        κ = (p_o − p_e) / (1 − p_e)

        where p_o = observed accuracy, p_e = expected accuracy by chance.

        κ = 1:  perfect agreement
        κ = 0:  agreement no better than chance
        κ < 0:  worse than chance

    Kappa is preferred over accuracy for imbalanced multiclass problems
    because it corrects for the agreement that would occur by chance.


──────────────────────────────────────────────────────────────────────────────
### Regression Metrics

In regression, predictions ŷ are continuous values compared to targets y.

    Diagram 7 — Regression Error Decomposition:

    Residual = y − ŷ  (actual minus predicted)

    y-axis
    │                    × ← y (true)
    │                 ↕ residual = y − ŷ
    │                    ○ ← ŷ (predicted)
    │       ×
    │           ×
    │    ×
    └────────────────── x-axis


**Mean Absolute Error (MAE):**

        MAE = (1/n) Σᵢ |yᵢ − ŷᵢ|

    Robust to outliers. Same units as target. Interpretable.
    Use when: outliers are common, you care about typical error equally.
    Gradient is constant → can be slow to optimise.

**Mean Squared Error (MSE):**

        MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²

    Penalises large errors quadratically — outliers dominate.
    Differentiable everywhere — smooth to optimise.
    Units are squared (hard to interpret).
    Use when: large errors are unacceptable (safety-critical systems).

**Root Mean Squared Error (RMSE):**

        RMSE = √MSE

    Same units as target — interpretable as MAE.
    Still outlier-sensitive (from the MSE inside).
    The most commonly reported regression metric.

**Comparing MAE vs RMSE:**
    ┌─────────────────────────────────────────────────────────────────────┐
    │  If RMSE >> MAE: outlier errors are driving RMSE up.                │
    │  The difference reveals whether errors are spread uniformly         │
    │  (RMSE ≈ MAE) or concentrated in a few large mistakes (RMSE >> MAE) │
    └─────────────────────────────────────────────────────────────────────┘

**R² (Coefficient of Determination):**

        R² = 1 − SS_res / SS_tot

        SS_res = Σ (yᵢ − ŷᵢ)²     ← residual sum of squares
        SS_tot = Σ (yᵢ − ȳ)²      ← total sum of squares (variance × n)

    Interpretation: fraction of variance in y explained by the model.
        R² = 1.0  → perfect predictions, model explains all variance
        R² = 0.0  → model is no better than predicting the mean ȳ
        R² < 0.0  → model is WORSE than predicting the mean (possible!)

    R² pitfalls:
    ┌───────────────────────────────────────────────────────────────────┐
    │  R² always increases as features are added, even irrelevant ones. │
    │  Use ADJUSTED R² which penalises for number of parameters:        │
    │                                                                   │
    │  R²_adj = 1 − (1−R²)(n−1)/(n−p−1)   where p = number of features  │
    │                                                                   │
    │  R² is not appropriate for comparing models across datasets.      │
    │  A model with R²=0.9 on easy data may be worse than R²=0.6 on     │
    │  hard data.                                                       │
    └───────────────────────────────────────────────────────────────────┘

**Mean Absolute Percentage Error (MAPE):**

        MAPE = (100/n) Σᵢ |yᵢ − ŷᵢ| / |yᵢ|

    Scale-free — can compare across datasets with different units.
    Fails when yᵢ = 0. Asymmetric — underestimates penalised more than
    overestimates of the same magnitude (because denominator = true y).

**Huber Loss (combines MAE and MSE):**

        H_δ(r) = r²/2           if |r| ≤ δ  (MSE regime)
               = δ(|r| − δ/2)  if |r| > δ  (MAE regime)

    Quadratic for small errors (smooth to optimise), linear for large
    errors (robust to outliers). δ controls the transition point.


──────────────────────────────────────────────────────────────────────────────
### Ranking and Retrieval Metrics

In information retrieval and recommendation systems, the model produces
a RANKED LIST of items. The quality of the ranking matters, not just
whether individual items are relevant.

    Setup: a query returns n items. Some are relevant (label=1), others not.

**Precision@K:**
        P@K = (relevant items in top K) / K
    How many of the first K results are actually relevant?

**Recall@K:**
        R@K = (relevant items in top K) / (total relevant items)
    Of all relevant items, how many appear in the top K?

**Average Precision (AP) for a single query:**

        AP = (1 / |relevant|) Σ_k P@k · rel(k)

    where rel(k) = 1 if item at rank k is relevant, else 0.
    This is the area under the precision-recall curve for one query.

**Mean Average Precision (MAP):**
        MAP = (1/Q) Σ_q AP(q)    (averaged over Q queries)

**Diagram 8 — AP Computation Example:**

    Ranked list for one query:  [✓, ✗, ✓, ✗, ✓, ✗, ✗, ✓]
    (✓ = relevant, ✗ = not relevant, 3 relevant items total)

    Rank:       1    2    3    4    5    6    7    8
    Relevant:   ✓    ✗    ✓    ✗    ✓    ✗    ✗    ✓
    P@k:       1/1   —   2/3   —   3/5   —    —   4/8
    Contribution: 1.0  +  0   + 0.67 +  0  + 0.60 +  0  +  0  + 0.50
    AP = (1.0 + 0.67 + 0.60 + 0.50) / 4 = 0.693  (4 = total relevant)

**Mean Reciprocal Rank (MRR):**

        MRR = (1/Q) Σ_q  1/rank(first relevant item for query q)

    Focuses entirely on where the FIRST relevant item appears.
    Used in: question answering (first correct answer rank), web search.

**Normalised Discounted Cumulative Gain (NDCG):**

    The most flexible ranking metric. Supports graded relevance (not just 0/1).

        DCG@K = Σ_{i=1}^{K}  (2^rel_i − 1) / log₂(i + 1)

        NDCG@K = DCG@K / IDCG@K

    IDCG = DCG of the ideal (perfect) ranking. NDCG ∈ [0, 1].

    The log₂(i+1) discount penalises relevant items appearing lower in rank.
    Positions near the top are exponentially more valuable.

    ┌───────────────────────────────────────────────────────────────────┐
    │  Ranking metric selection guide:                                  │
    │  Only care about 1st result?             → MRR                    │
    │  Only care about top K?                  → P@K, MAP               │
    │  Graded relevance (0-5 stars)?           → NDCG                   │
    │  Full list quality matters?              → AP, NDCG               │
    │  Recommendation systems?                 → NDCG, MAP              │
    └───────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Calibration — When Probabilities Must Be Trustworthy

A classifier that outputs 0.8 for a sample is claiming "80% probability
of being positive." CALIBRATION measures whether these stated probabilities
match real-world frequencies.

    A perfectly calibrated model satisfies:
        P(y=1 | model outputs p) = p     for all p ∈ [0, 1]

    Diagram 9 — Calibration Curves (Reliability Diagrams):

    Fraction of positives
    │
    1.0 │                           ●  ← overconfident (predicts high prob
        │                        ●     but fewer positives than claimed)
    0.8 │                     ●
        │                  ●         ╱  ← perfectly calibrated (diagonal)
    0.6 │               ●         ╱
        │            ●         ╱
    0.4 │         ●         ╱
        │      ●         ╱
    0.2 │   ●         ╱
        │  ●  ←  underconfident (model hedges, predicts probabilities
    0.0 └──────────────────────────────────── Mean predicted probability
         0.0  0.2  0.4  0.6  0.8  1.0       too close to 0.5)

    **Expected Calibration Error (ECE):**

        Partition predictions into M bins by probability.
        For each bin b:
            acc(b) = fraction of positive labels in the bin
            conf(b) = mean predicted probability in the bin

        ECE = Σ_b (|b|/n) · |acc(b) − conf(b)|

    **Brier Score:**

        BS = (1/n) Σᵢ (pᵢ − yᵢ)²

        A proper scoring rule that rewards both calibration and sharpness.
        BS = 0: perfect. BS = 1: perfectly wrong. BS = 0.25: random (50% base rate).

    **Log-Loss (Cross-Entropy Loss):**

        Log-Loss = −(1/n) Σᵢ [yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ)]

        Also a proper scoring rule. Penalises confident wrong predictions
        infinitely (log(0) = −∞). More sensitive to extreme errors than Brier.

    ┌───────────────────────────────────────────────────────────────────┐
    │  Calibration matters when probabilities drive decisions:          │
    │  - Medical: "70% chance of cancer" influences treatment choice    │
    │  - Finance: probability estimates drive portfolio allocation      │
    │  - Meteorology: "40% chance of rain" affects planning decisions   │
    │                                                                   │
    │  Tree ensembles and SVMs are often miscalibrated. Use             │
    │  Platt scaling (sigmoid) or isotonic regression to recalibrate.   │
    └───────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Metric Pathologies — When Numbers Lie

**1. The Accuracy Paradox (Class Imbalance):**
    As shown above — 99% accuracy on 1% positive-rate data is worthless.
    Fix: use F1, MCC, AUC-PR.

**2. Goodhart's Law:**
    "When a measure becomes a target, it ceases to be a good measure."
    Optimising directly for AUC-ROC can produce models that are great at
    ranking but useless at a fixed threshold. Always evaluate at the
    operating threshold you will use in production.

**3. Test Set Leakage:**
    If any information from test labels influences training (even indirectly
    through feature selection, hyperparameter tuning on test set, or
    normalisation using test statistics), reported metrics are inflated.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Correct pipeline order:                                         │
    │  1. Split data FIRST (before any preprocessing)                  │
    │  2. Fit ALL transformers on training set ONLY                    │
    │  3. Apply fitted transformers to validation and test sets        │
    │  4. Hyperparameter selection via cross-validation on train set   │
    │  5. Report FINAL metrics on test set ONCE, at the very end       │
    └──────────────────────────────────────────────────────────────────┘

**4. Distribution Shift between Evaluation and Production:**
    A model evaluated on a held-out slice of the training distribution
    may fail when deployed on a different distribution (different time
    period, different user demographic, different sensor).
    Fix: evaluate on out-of-time or out-of-distribution test splits.

**5. Metric-Target Mismatch:**
    Training with cross-entropy loss but reporting accuracy is common.
    The model is not optimising accuracy — it is optimising a proxy.
    This is not always wrong, but should be understood.

**6. Simpson's Paradox:**
    A model can appear better in every subgroup but worse overall, or
    vice versa. Always disaggregate metrics by subgroup.

    Example:
    ┌──────────────────────────┬──────────────┬──────────────┐
    │                          │ Model A acc  │ Model B acc  │
    ├──────────────────────────┼──────────────┼──────────────┤
    │ Subgroup 1 (n=100)       │ 90%          │ 85%          │
    │ Subgroup 2 (n=900)       │ 60%          │ 55%          │
    │ Overall (n=1000)         │ 63.0%        │ 56.5%        │
    ├──────────────────────────┼──────────────┼──────────────┤
    │ But if populations flip: │ (n=900)      │ (n=100)      │
    │ Model A: 900×0.9 + 100×0.6 = 870/1000 = 87%            │
    │ Model B: 100×0.85 + 900×0.55 = 580/1000 = 58%          │
    └──────────────────────────┴──────────────┴──────────────┘
    The better model depends on which subgroup is more prevalent.


──────────────────────────────────────────────────────────────────────────────
### Choosing the Right Metric — Decision Framework

    Diagram 10 — Metric Selection Decision Tree:

    Is it a classification problem?
    │
    ├── YES → Is it binary?
    │          ├── YES → Is the dataset balanced?
    │          │          ├── YES (>80-20 split) → Accuracy, F1
    │          │          └── NO  (imbalanced)   → AUC-PR, F1, MCC
    │          │          Do probabilities matter?
    │          │          └── YES → Brier Score, Log-Loss, AUC-ROC
    │          └── NO (multiclass) → Is class balance an issue?
    │                 ├── YES → Macro-F1, Weighted-F1, Cohen's κ
    │                 └── NO  → Accuracy, Micro-F1
    │
    └── NO → Is it regression?
              ├── YES → Are outliers a concern?
              │          ├── YES (outliers tolerated) → MAE, Huber
              │          └── NO  (outliers important) → RMSE, MSE
              │          Need to compare across scales?
              │          └── YES → MAPE, R²
              └── NO → Is it a ranking/retrieval task?
                        ├── Care about first result only → MRR
                        ├── Graded relevance (0-5 stars) → NDCG
                        └── Binary relevance, full list  → MAP, AP

    Final checklist before reporting any metric:
    ┌────────────────────────────────────────────────────────────────────┐
    │  □  Does the metric match the business cost structure?             │
    │  □  Is it computed on held-out data (no leakage)?                  │
    │  □  Is it disaggregated across important subgroups?                │
    │  □  Is the baseline (random, majority, naive) also reported?       │
    │  □  Is it evaluated at the operating threshold, not just AUC?      │
    │  □  Is statistical uncertainty (confidence intervals) included?    │
    └────────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Confusion Matrix & Classification Metrics Deep Dive": {
        "description": (
            "Build a complete classification metrics dashboard from scratch "
            "using only NumPy — no sklearn shortcuts. Compute TP, TN, FP, FN, "
            "accuracy, precision, recall, specificity, F1, F-beta, and MCC. "
            "Compare five different classifiers (including one that always "
            "predicts the majority class) on an imbalanced dataset and show "
            "exactly why accuracy is misleading when classes are skewed."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)

# ── Dataset: 1000 samples, 9% positive (imbalanced) ──────────────────────
N      = 1000
P_RATE = 0.09    # 9% positive — credit card fraud-like imbalance
y_true = (np.random.rand(N) < P_RATE).astype(int)
n_pos  = y_true.sum()
n_neg  = N - n_pos

print("=" * 65)
print("  CLASSIFICATION METRICS DEEP DIVE")
print(f"  Dataset: n={N}, positives={n_pos} ({n_pos/N*100:.1f}%), "
      f"negatives={n_neg} ({n_neg/N*100:.1f}%)")
print("=" * 65)
print()

# ── Five different "classifiers" (simulated decision scores) ──────────────
rng = np.random.default_rng(7)

# 1. Good classifier: AUC ≈ 0.88
scores_good = np.where(y_true == 1,
                        rng.beta(5, 2, N),
                        rng.beta(2, 5, N))

# 2. Mediocre classifier: AUC ≈ 0.67
scores_med  = np.where(y_true == 1,
                        rng.beta(3, 2.5, N),
                        rng.beta(2.5, 3, N))

# 3. Always-negative classifier (predict 0 for everything)
scores_neg  = np.zeros(N)

# 4. Always-positive classifier (predict 1 for everything)
scores_pos  = np.ones(N)

# 5. Random classifier
scores_rand = rng.random(N)

# ── Metric computation from scratch ──────────────────────────────────────
def compute_metrics(y_true, scores, threshold=0.5, beta=1.0):
    y_pred = (scores >= threshold).astype(int)
    TP = int(( y_pred &  y_true).sum())
    TN = int((~y_pred.astype(bool) & ~y_true.astype(bool)).sum())
    FP = int(( y_pred & ~y_true.astype(bool)).sum())
    FN = int((~y_pred.astype(bool) &  y_true.astype(bool)).sum())

    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall      = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    f1   = 2 * precision * recall / (precision + recall) \
           if (precision + recall) > 0 else 0.0
    fb   = (1 + beta**2) * precision * recall \
           / (beta**2 * precision + recall + 1e-12)

    denom_mcc = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = (TP*TN - FP*FN) / denom_mcc if denom_mcc > 0 else 0.0

    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                accuracy=accuracy, precision=precision,
                recall=recall, specificity=specificity,
                f1=f1, fb=fb, mcc=mcc)

def auc_roc(y_true, scores):
    """Trapezoidal AUC computed from scratch."""
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [0.], [0.]
    P = y_true.sum(); N_neg = len(y_true) - P
    for t in thresholds:
        yp = (scores >= t).astype(int)
        tp = (yp & y_true).sum()
        fp = (yp & ~y_true.astype(bool)).sum()
        tprs.append(tp / P if P > 0 else 0)
        fprs.append(fp / N_neg if N_neg > 0 else 0)
    tprs.append(1.); fprs.append(1.)
    tprs, fprs = np.array(tprs), np.array(fprs)
    return float(np.trapezoid(tprs, fprs))

classifiers = {
    "Good Classifier":      scores_good,
    "Mediocre Classifier":  scores_med,
    "Always-Negative":      scores_neg,
    "Always-Positive":      scores_pos,
    "Random Classifier":    scores_rand,
}

all_metrics = {}
for name, scores in classifiers.items():
    m = compute_metrics(y_true, scores, threshold=0.5, beta=2.0)
    m["auc"] = auc_roc(y_true, scores)
    all_metrics[name] = m

# ── Print comparison table ────────────────────────────────────────────────
print(f"  {'Classifier':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
      f"{'F1':>6} {'F2':>6} {'MCC':>7} {'AUC':>7}")
print("  " + "─" * 68)
for name, m in all_metrics.items():
    print(f"  {name:<22} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
          f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['fb']:>6.3f} "
          f"{m['mcc']:>7.3f} {m['auc']:>7.3f}")

print()
print("  INSIGHT: 'Always-Negative' achieves 91.0% accuracy (by predicting")
print("  nothing) — yet Recall=0, F1=0, MCC=0, AUC=0.5. Accuracy is")
print("  completely misleading on this imbalanced dataset.")
print()
print("  Good Classifier has lower accuracy than Always-Negative but:")
print(f"    F1 = {all_metrics['Good Classifier']['f1']:.3f}  (vs 0.000)")
print(f"    MCC= {all_metrics['Good Classifier']['mcc']:.3f}  (vs 0.000)")
print(f"    AUC= {all_metrics['Good Classifier']['auc']:.3f}  (vs 0.500)")

# ── Confusion matrix heatmaps ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Confusion Matrices and Classification Metrics\\n"
             "9% Positive Rate — Why Accuracy Alone is Misleading",
             fontsize=12, fontweight="bold")

cmap_cm = LinearSegmentedColormap.from_list("cm", ["#ffffff", "#2166ac"])
metric_bar_ax = axes[1, 2]

for idx, (name, m) in enumerate(all_metrics.items()):
    row, col = divmod(idx, 3)
    ax = axes[row, col]

    cm = np.array([[m['TN'], m['FP']],
                   [m['FN'], m['TP']]])
    im = ax.imshow(cm, cmap=cmap_cm, aspect="equal")

    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            colour = "white" if val > cm.max() * 0.5 else "black"
            label = ["TN", "FP", "FN", "TP"][[0,1,2,3][r*2+c]]
            ax.text(c, r, f"{label}\\n{val}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=colour)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["Actual Neg", "Actual Pos"])
    ax.set_title(f"{name}\\n"
                 f"Acc={m['accuracy']:.2f}  F1={m['f1']:.2f}  "
                 f"MCC={m['mcc']:.2f}  AUC={m['auc']:.2f}",
                 fontsize=9, fontweight="bold")

# Last panel: metric comparison bar chart
clf_names   = list(all_metrics.keys())
metric_keys = ["accuracy", "precision", "recall", "f1", "mcc"]
metric_lbls = ["Accuracy", "Precision", "Recall", "F1", "MCC"]
x = np.arange(len(metric_keys))
width = 0.15
colours_bar = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, (name, colour) in enumerate(zip(clf_names, colours_bar)):
    vals = [all_metrics[name][k] for k in metric_keys]
    metric_bar_ax.bar(x + i * width, vals, width, label=name,
                      color=colour, alpha=0.8)

metric_bar_ax.set_xticks(x + width * 2)
metric_bar_ax.set_xticklabels(metric_lbls, fontsize=9)
metric_bar_ax.set_title("All Metrics Side-by-Side\\n"
                         "(Accuracy inflates Always-Negative)", fontsize=9,
                         fontweight="bold")
metric_bar_ax.set_ylabel("Score")
metric_bar_ax.set_ylim(0, 1.15)
metric_bar_ax.legend(fontsize=7, loc="upper right", ncol=1)
metric_bar_ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("confusion_matrix_metrics.png", dpi=110)
print()
print("  Plot saved → confusion_matrix_metrics.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · ROC Curves, PR Curves, and AUC": {
        "description": (
            "Build ROC and Precision-Recall curves from scratch and plot them "
            "side by side for four classifiers on both a balanced and an "
            "imbalanced dataset. Demonstrates exactly why AUC-ROC is "
            "misleading on imbalanced data while AUC-PR reveals the truth. "
            "Includes threshold selection analysis and the F1-optimal threshold."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Helper: ROC and PR curves from scratch ────────────────────────────────
def roc_pr_curves(y_true, scores):
    thresholds = np.sort(np.unique(scores))[::-1]
    P   = y_true.sum()
    NEG = len(y_true) - P

    tprs, fprs, precs, recs = [0.], [0.], [1.], [0.]

    for t in thresholds:
        yp = (scores >= t).astype(int)
        tp = int((yp & y_true).sum())
        fp = int((yp & (1 - y_true)).sum())
        fn = P - tp
        tpr = tp / P    if P   > 0 else 0.0
        fpr = fp / NEG  if NEG > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec  = tp / P if P > 0 else 0.0
        tprs.append(tpr); fprs.append(fpr)
        precs.append(prec); recs.append(rec)

    tprs.append(1.); fprs.append(1.)
    precs.append(P/len(y_true)); recs.append(1.)

    auc_roc = float(np.trapezoid(tprs, fprs))
    auc_pr  = float(np.trapezoid(precs[::-1], recs[::-1]))

    return (np.array(tprs), np.array(fprs),
            np.array(precs), np.array(recs),
            auc_roc, auc_pr)

def f1_at_all_thresholds(y_true, scores):
    thresholds = np.sort(np.unique(scores))[::-1]
    f1s, ts = [], []
    for t in thresholds:
        yp = (scores >= t).astype(int)
        tp = int((yp & y_true).sum())
        fp = int((yp & (1-y_true)).sum())
        fn = int(y_true.sum()) - tp
        f1 = 2*tp / (2*tp + fp + fn + 1e-12)
        f1s.append(f1); ts.append(t)
    return np.array(ts), np.array(f1s)

rng = np.random.default_rng(42)

# ── Two datasets ─────────────────────────────────────────────────────────
datasets = {
    "Balanced (50/50)":    0.50,
    "Imbalanced (5/95)":   0.05,
}

classifiers_scores = {
    "Strong (AUC≈0.90)":   lambda y, n: np.where(y==1, rng.beta(7,2,n), rng.beta(2,7,n)),
    "Moderate (AUC≈0.70)": lambda y, n: np.where(y==1, rng.beta(3,2,n), rng.beta(2,3,n)),
    "Weak (AUC≈0.60)":     lambda y, n: np.where(y==1, rng.beta(2,1.8,n), rng.beta(1.8,2,n)),
    "Random":              lambda y, n: rng.random(n),
}
clf_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

N = 2000

print("=" * 72)
print("  ROC vs PRECISION-RECALL CURVES: BALANCED vs IMBALANCED")
print("=" * 72)

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle("ROC Curves vs Precision-Recall Curves\\n"
             "Left columns: Balanced dataset (50/50)  |  "
             "Right columns: Imbalanced dataset (5/95)\\n"
             "AUC-ROC inflates performance on imbalanced data — "
             "AUC-PR tells the real story",
             fontsize=11, fontweight="bold")

for ds_col, (ds_name, pos_rate) in enumerate(datasets.items()):
    y = (rng.random(N) < pos_rate).astype(int)
    P = y.sum()
    baseline_pr = P / N    # random classifier PR baseline

    print(f"\\n  Dataset: {ds_name}  |  n={N}, pos={P} ({P/N*100:.1f}%)")
    print(f"  {'Classifier':<24} {'AUC-ROC':>8}  {'AUC-PR':>8}  "
          f"{'Baseline PR':>12}  {'Gain over baseline':>18}")
    print("  " + "─" * 75)

    ax_roc = axes[0, ds_col * 2]
    ax_pr  = axes[0, ds_col * 2 + 1]
    ax_f1  = axes[1, ds_col * 2]
    ax_thr = axes[1, ds_col * 2 + 1]

    for (clf_name, score_fn), colour in zip(classifiers_scores.items(), clf_colours):
        scores = score_fn(y, N)
        tprs, fprs, precs, recs, auc_r, auc_p = roc_pr_curves(y, scores)
        gain = auc_p - baseline_pr

        print(f"  {clf_name:<24} {auc_r:>8.3f}  {auc_p:>8.3f}  "
              f"{baseline_pr:>12.3f}  {gain:>+17.3f}")

        ax_roc.plot(fprs, tprs, lw=2, color=colour,
                    label=f"{clf_name[:16]} (AUC={auc_r:.2f})")
        ax_pr.plot(recs, precs, lw=2, color=colour,
                   label=f"{clf_name[:16]} (AP={auc_p:.2f})")

        # F1 vs threshold
        ts, f1s = f1_at_all_thresholds(y, scores)
        ax_f1.plot(ts, f1s, lw=1.8, color=colour, alpha=0.85)
        best_idx = f1s.argmax()
        ax_f1.scatter([ts[best_idx]], [f1s[best_idx]],
                      s=60, color=colour, zorder=5)

    # Diagonal reference lines
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax_pr.axhline(baseline_pr, color="black", ls="--", lw=1,
                  label=f"Random baseline (AP={baseline_pr:.2f})")

    ax_roc.set_title(f"ROC Curve\\n{ds_name}", fontweight="bold", fontsize=9)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(fontsize=7); ax_roc.grid(alpha=0.3)
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])

    ax_pr.set_title(f"Precision-Recall Curve\\n{ds_name}",
                    fontweight="bold", fontsize=9)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=7); ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.02])

    ax_f1.set_title(f"F1 Score vs Threshold\\n{ds_name}",
                    fontweight="bold", fontsize=9)
    ax_f1.set_xlabel("Threshold τ"); ax_f1.set_ylabel("F1 Score")
    ax_f1.grid(alpha=0.3)

    ax_thr.text(0.5, 0.5,
        f"Dataset: {ds_name}\\n\\n"
        f"Positive rate: {pos_rate*100:.0f}%\\n"
        f"N positives: {P}\\n"
        f"N negatives: {N-P}\\n\\n"
        f"PR baseline (random): {baseline_pr:.3f}\\n\\n"
        f"KEY INSIGHT:\\n"
        f"On imbalanced data (5/95),\\n"
        f"AUC-ROC looks similar to\\n"
        f"balanced data, but AUC-PR\\n"
        f"is drastically lower.\\n\\n"
        f"AUC-PR reveals true quality\\n"
        f"for detecting rare events.",
        transform=ax_thr.transAxes, ha="center", va="center",
        fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fffde7", ec="#f9a825"))
    ax_thr.axis("off")

plt.tight_layout()
plt.savefig("roc_pr_curves.png", dpi=110)
print()
print("\\n  Plot saved → roc_pr_curves.png")
print()
print("  KEY TAKEAWAYS:")
print("  - On balanced data, AUC-ROC and AUC-PR tell similar stories.")
print("  - On imbalanced data (5%), AUC-ROC stays high while AUC-PR")
print("    collapses to near-baseline — revealing poor positive detection.")
print("  - The F1-optimal threshold (dot on F1 curve) ≠ 0.5 in general.")
print("  - ALWAYS report both AUC-ROC and AUC-PR on imbalanced data.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Regression Metrics — MAE, MSE, RMSE, R², and Huber": {
        "description": (
            "Compare all major regression metrics on three datasets that "
            "differ only in the presence and magnitude of outliers. Computes "
            "MAE, MSE, RMSE, R², MAPE, and Huber loss from scratch. Shows "
            "exactly how MSE/RMSE amplify outlier impact while MAE remains "
            "stable, and demonstrates how Huber loss bridges both worlds. "
            "Plots residual distributions and metric sensitivity curves."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Metric implementations ─────────────────────────────────────────────────
def mae(y, yh):
    return np.mean(np.abs(y - yh))

def mse(y, yh):
    return np.mean((y - yh) ** 2)

def rmse(y, yh):
    return np.sqrt(mse(y, yh))

def r2(y, yh):
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def mape(y, yh, eps=1e-8):
    return np.mean(np.abs((y - yh) / (np.abs(y) + eps))) * 100

def huber(y, yh, delta=1.0):
    r = y - yh
    return np.where(np.abs(r) <= delta,
                    0.5 * r**2,
                    delta * (np.abs(r) - 0.5 * delta)).mean()

# ── Generate datasets ──────────────────────────────────────────────────────
n = 300

# True relationship: y = 2x + 1 + noise
x      = np.linspace(0, 10, n)
y_true = 2 * x + 1

# Dataset A: clean Gaussian noise
yA = y_true + np.random.normal(0, 1.5, n)

# Dataset B: few large outliers (5%)
yB = yA.copy()
n_outliers = int(0.05 * n)
outlier_idx = np.random.choice(n, n_outliers, replace=False)
yB[outlier_idx] += np.random.choice([-1, 1], n_outliers) * \
                   np.random.uniform(10, 20, n_outliers)

# Dataset C: many moderate outliers (20%)
yC = yA.copy()
n_outliers_c = int(0.20 * n)
outlier_idx_c = np.random.choice(n, n_outliers_c, replace=False)
yC[outlier_idx_c] += np.random.choice([-1, 1], n_outliers_c) * \
                     np.random.uniform(3, 7, n_outliers_c)

# "Perfect" linear regression predictions
y_pred_A = 2 * x + 1     # ideal
y_pred_B = 2 * x + 1     # same ideal predictions, but data has outliers
y_pred_C = 2 * x + 1

datasets = {
    "A: Clean (Gaussian noise)":           (yA, y_pred_A),
    "B: 5% Large Outliers (±10–20)":       (yB, y_pred_B),
    "C: 20% Moderate Outliers (±3–7)":     (yC, y_pred_C),
}

print("=" * 72)
print("  REGRESSION METRICS: SENSITIVITY TO OUTLIERS")
print("  Model: true y = 2x+1, predictions = 2x+1 (ideal fit on clean data)")
print("=" * 72)
print()
print(f"  {'Dataset':<35} {'MAE':>6} {'RMSE':>6} {'R²':>6} "
      f"{'MAPE%':>7} {'Huber':>7}")
print("  " + "─" * 70)

all_results = {}
for ds_name, (y_true_ds, y_pred_ds) in datasets.items():
    results = {
        "mae":    mae(y_true_ds,  y_pred_ds),
        "mse":    mse(y_true_ds,  y_pred_ds),
        "rmse":   rmse(y_true_ds, y_pred_ds),
        "r2":     r2(y_true_ds,   y_pred_ds),
        "mape":   mape(y_true_ds, y_pred_ds),
        "huber":  huber(y_true_ds, y_pred_ds, delta=2.0),
    }
    all_results[ds_name] = results
    print(f"  {ds_name:<35} {results['mae']:>6.2f} {results['rmse']:>6.2f} "
          f"{results['r2']:>6.3f} {results['mape']:>7.2f} {results['huber']:>7.2f}")

print()
print("  INSIGHT: MAE increases modestly from A to B (outlier-robust).")
print("  RMSE jumps dramatically (outliers inflate squared error).")
print("  R² can go NEGATIVE when outliers dominate the SS_res term.")
print()
print("  RMSE / MAE ratio reveals outlier presence:")
for ds_name, results in all_results.items():
    ratio = results["rmse"] / results["mae"] if results["mae"] > 0 else 0
    note = ("→ clean"          if ratio < 1.3
            else "→ moderate outliers" if ratio < 2.0
            else "→ severe outliers")
    print(f"    {ds_name[:35]}: RMSE/MAE = {ratio:.2f}  {note}")

# ── Huber loss: δ sensitivity analysis ────────────────────────────────────
print()
print("  HUBER LOSS: Effect of delta parameter on dataset B (5% outliers):")
print(f"  {'δ':>6}  {'Huber':>8}  {'Regime (mostly MSE or MAE?)':>30}")
print("  " + "─" * 48)
for delta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
    h = huber(yB, y_pred_B, delta=delta)
    regime = ("≈ MAE (outlier-robust)" if delta < 1.0
              else "balanced" if delta < 5.0
              else "≈ MSE (sensitive)")
    print(f"  {delta:>6.1f}  {h:>8.3f}  {regime}")

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Regression Metrics: Sensitivity to Outliers\\n"
             "MAE is robust; MSE/RMSE are amplified by large residuals",
             fontsize=12, fontweight="bold")

ds_colours = ["steelblue", "tomato", "seagreen"]
x_plot = np.linspace(0, 10, 300)

# Row 1: scatter plots with residuals
for col, ((ds_name, (y_actual, y_pred)), colour) in enumerate(
        zip(datasets.items(), ds_colours)):
    ax = axes[0, col]
    residuals = y_actual - y_pred
    outlier_mask = np.abs(residuals) > 3 * np.std(yA - y_pred_A)

    ax.scatter(x, y_actual, s=8, alpha=0.5, color=colour, label="Data")
    ax.plot(x_plot, 2*x_plot + 1, "k-", lw=2, label="True / Pred")

    # Highlight outliers
    if outlier_mask.any():
        ax.scatter(x[outlier_mask], y_actual[outlier_mask],
                   s=40, color="red", zorder=5, label="Outliers")
        for xi, yi in zip(x[outlier_mask], y_actual[outlier_mask]):
            ax.plot([xi, xi], [2*xi+1, yi], "r-", alpha=0.4, lw=0.8)

    r = all_results[ds_name]
    ax.set_title(f"{ds_name}\\n"
                 f"MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  R²={r['r2']:.3f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Row 2: Residual histograms + Huber loss δ curve
for col, ((ds_name, (y_actual, y_pred)), colour) in enumerate(
        zip(datasets.items(), ds_colours)):
    if col < 2:
        ax = axes[1, col]
        residuals = y_actual - y_pred
        ax.hist(residuals, bins=40, color=colour, alpha=0.75, density=True,
                label="Residuals")
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.axvline(residuals.mean(), color="red", lw=1.5,
                   label=f"Mean={residuals.mean():.2f}")
        ax.set_title(f"Residual Distribution\\n{ds_name.split(':')[0]}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Residual"); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,2): Metric comparison bar chart across datasets
ax_bar = axes[1, 2]
metrics_to_show = ["mae", "rmse", "huber"]
labels_m = ["MAE", "RMSE", "Huber(δ=2)"]
x_bar = np.arange(len(metrics_to_show))
width = 0.25
for i, ((ds_name, _), colour) in enumerate(zip(datasets.items(), ds_colours)):
    vals = [all_results[ds_name][k] for k in metrics_to_show]
    ax_bar.bar(x_bar + i*width, vals, width, label=ds_name[:15],
               color=colour, alpha=0.8)
ax_bar.set_xticks(x_bar + width)
ax_bar.set_xticklabels(labels_m)
ax_bar.set_title("Metric Comparison Across Datasets\\n"
                  "(RMSE jumps, MAE/Huber stay lower)", fontsize=9,
                  fontweight="bold")
ax_bar.set_ylabel("Error Value")
ax_bar.legend(fontsize=8)
ax_bar.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("regression_metrics.png", dpi=110)
print()
print("  Plot saved → regression_metrics.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Calibration Analysis — When Probabilities Must Be Trusted": {
        "description": (
            "Demonstrate probability calibration using reliability diagrams, "
            "Expected Calibration Error (ECE), and Brier Score. Compares "
            "an uncalibrated model, an overconfident model, an underconfident "
            "model, and a well-calibrated model. Shows how Platt scaling "
            "and isotonic regression recalibrate a poorly calibrated classifier, "
            "and explains when calibration matters most in production."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1./self.C; w0=np.zeros(d+1)
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

class _DTree:
    def __init__(self,max_depth=8,min_samples=2):
        self.max_depth=max_depth; self.min_samples=min_samples
    def _gini(self,y):
        if not len(y): return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _split(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2
            thrs=all_t[np.linspace(0,len(all_t)-1,min(10,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples or r.sum()<self.min_samples: continue
                g=self._gini(y[l])*l.sum()+self._gini(y[r])*r.sum()
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        cls,cnt=np.unique(y,return_counts=True); pred=cls[cnt.argmax()]
        if depth>=self.max_depth or len(cls)==1 or len(y)<=self.min_samples:
            return ("L",pred,cnt/cnt.sum())
        sp=self._split(X,y)
        if sp is None: return ("L",pred,cnt/cnt.sum())
        _,f,t=sp; m=X[:,f]<=t
        return ("N",f,t,self._build(X[m],y[m],depth+1),self._build(X[~m],y[~m],depth+1))
    def fit(self,X,y):
        self.classes_=np.unique(y); self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd[0]=="L": return nd[2]
        return self._p1(x,nd[3] if x[nd[1]]<=nd[2] else nd[4])
    def predict_proba(self,X):
        nc=len(self.classes_)
        out=np.zeros((len(X),nc))
        for i,x in enumerate(X):
            probs=self._p1(x,self._tree); out[i,:len(probs)]=probs
        return out
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]

class RandomForestClassifier:
    def __init__(self,n_estimators=20,max_depth=8,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self._trees=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(np.sqrt(X.shape[1]))); fi=rng.choice(X.shape[1],nf,replace=False)
            t=_DTree(max_depth=self.max_depth); t.fit(X[np.ix_(idx,fi)],y[idx])
            t._feat_idx=fi; self._trees.append(t)
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); votes=np.zeros((len(X),nc))
        for t in self._trees:
            p=t.predict_proba(X[:,t._feat_idx])
            for li,gc in enumerate(t.classes_):
                gi=np.where(self.classes_==gc)[0]
                if len(gi): votes[:,gi[0]]+=p[:,li]
        return votes/len(self._trees)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

class IsotonicRegression:
    """Pool Adjacent Violators isotonic regression (non-decreasing)."""
    def __init__(self,out_of_bounds="clip"): self.out_of_bounds=out_of_bounds
    def fit(self,X,y):
        order=np.argsort(X); xs=X[order]; ys=y[order].astype(float)
        blocks=[[ys[0],1,xs[0]]]
        for i in range(1,len(ys)):
            blocks.append([ys[i],1,xs[i]])
            while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
                a=blocks.pop(); b=blocks.pop()
                total=a[1]+b[1]; merged=[(a[0]*a[1]+b[0]*b[1])/total,total,b[2]]
                blocks.append(merged)
        self._x=np.array([b[2] for b in blocks]); self._y=np.array([b[0] for b in blocks])
        return self
    def predict(self,X):
        out=np.interp(X,self._x,self._y)
        if self.out_of_bounds=="clip": out=np.clip(out,self._y.min(),self._y.max())
        return out

np.random.seed(42)

# ── Reliability diagram + ECE computation ────────────────────────────────
def reliability_diagram(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_accs.append(y_true[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_sizes.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append((lo + hi) / 2)
            bin_sizes.append(0)

    bin_accs   = np.array(bin_accs)
    bin_confs  = np.array(bin_confs)
    bin_sizes  = np.array(bin_sizes)

    valid = bin_sizes > 0
    ece = np.sum(bin_sizes[valid] * np.abs(bin_accs[valid] - bin_confs[valid])) \
          / bin_sizes[valid].sum()
    return bin_confs, bin_accs, bin_sizes, ece

def brier_score(y_true, probs):
    return np.mean((probs - y_true) ** 2)

def log_loss(y_true, probs, eps=1e-12):
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(y_true * np.log(probs) + (1-y_true) * np.log(1-probs))

# ── Dataset ────────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=3000, n_features=20, n_informative=8,
                            n_redundant=5, flip_y=0.05, random_state=0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                           random_state=1, stratify=y)

# ── Generate four calibration profiles ───────────────────────────────────
rng = np.random.default_rng(7)

# 1. Well-calibrated (logistic regression on enough data)
lr_clf = LogisticRegression(C=0.5, max_iter=500)
lr_clf.fit(X_tr, y_tr)
probs_good = lr_clf.predict_proba(X_te)[:, 1]

# 2. Overconfident: push probabilities toward 0 and 1
probs_over = np.clip(np.where(probs_good > 0.5,
                               np.minimum(1.0, probs_good * 1.8),
                               np.maximum(0.0, probs_good * 0.3)), 1e-6, 1-1e-6)

# 3. Underconfident: shrink toward 0.5
probs_under = 0.5 + (probs_good - 0.5) * 0.35

# 4. Random Forest (typically overconfident — a known calibration failure)
rf_clf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0)
rf_clf.fit(X_tr, y_tr)
probs_rf = rf_clf.predict_proba(X_te)[:, 1]

# ── Calibration methods ───────────────────────────────────────────────────
# Platt scaling (logistic regression on held-out fold of training set)
X_cal, X_val_cal, y_cal, y_val_cal = train_test_split(
    X_tr, y_tr, test_size=0.3, random_state=3)
rf_cal = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0)
rf_cal.fit(X_cal, y_cal)
probs_rf_cal = rf_cal.predict_proba(X_val_cal)[:, 1].reshape(-1, 1)

# Platt: logistic regression on calibration probs
platt = LogisticRegression(C=10, max_iter=200)
platt.fit(probs_rf_cal, y_val_cal)
probs_rf_platt = platt.predict_proba(
    rf_clf.predict_proba(X_te)[:, 1].reshape(-1, 1))[:, 1]

# Isotonic regression calibration
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(rf_clf.predict_proba(X_val_cal)[:, 1], y_val_cal)
probs_rf_iso = iso.predict(probs_rf)

all_probs = {
    "Well-Calibrated\\n(Logistic Reg)":  probs_good,
    "Overconfident\\n(Manual)":          probs_over,
    "Underconfident\\n(Manual)":         probs_under,
    "Random Forest\\n(Uncalibrated)":    probs_rf,
}
calib_probs = {
    "RF Uncalibrated":  probs_rf,
    "RF + Platt Scaling": probs_rf_platt,
    "RF + Isotonic":      probs_rf_iso,
}

print("=" * 65)
print("  CALIBRATION ANALYSIS: RELIABILITY DIAGRAMS AND ECE")
print("=" * 65)
print()
print("  Main comparison:")
print(f"  {'Model':<30}  {'ECE':>6}  {'Brier':>7}  {'Log-Loss':>9}  Description")
print("  " + "─" * 72)
for name, probs in all_probs.items():
    _, _, _, ece = reliability_diagram(y_te, probs)
    bs  = brier_score(y_te, probs)
    ll  = log_loss(y_te, probs)
    desc = ("≈ perfect" if ece < 0.03
            else "slight overconf" if name.startswith("Over")
            else "slight underconf" if name.startswith("Under")
            else "miscalibrated RF")
    nm = name.replace("\\n", " ")
    print(f"  {nm:<30}  {ece:>6.4f}  {bs:>7.4f}  {ll:>9.4f}  {desc}")

print()
print("  Calibration correction for Random Forest:")
print(f"  {'Method':<28}  {'ECE':>6}  {'Brier':>7}  {'Log-Loss':>9}")
print("  " + "─" * 55)
for name, probs in calib_probs.items():
    _, _, _, ece = reliability_diagram(y_te, probs)
    bs  = brier_score(y_te, probs)
    ll  = log_loss(y_te, probs)
    print(f"  {name:<28}  {ece:>6.4f}  {bs:>7.4f}  {ll:>9.4f}")

print()
print("  Brier score reference:")
print("  BS = 0.25  → random classifier at 50% base rate")
print("  BS < 0.10  → good probabilistic model")
print("  BS < 0.05  → excellent calibration AND discrimination")

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle("Probability Calibration: Reliability Diagrams\\n"
             "Perfect calibration = points on the diagonal. "
             "Gap above = underconfident. Gap below = overconfident.",
             fontsize=12, fontweight="bold")

colours_main = ["steelblue", "tomato", "darkorange", "mediumpurple"]
colours_cal  = ["mediumpurple", "seagreen", "#f97316"]

N_BINS = 10

def plot_reliability(ax, y_true, probs, title, colour, n_bins=N_BINS):
    bin_confs, bin_accs, bin_sizes, ece = reliability_diagram(
        y_true, probs, n_bins)
    valid = bin_sizes > 0

    # Gap bars (calibration error)
    for bc, ba in zip(bin_confs[valid], bin_accs[valid]):
        ax.bar(bc, ba - bc, bottom=bc, width=0.08,
               color="tomato" if ba < bc else "steelblue",
               alpha=0.35, zorder=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calib.")
    ax.scatter(bin_confs[valid], bin_accs[valid],
               s=bin_sizes[valid] * 0.3 + 20, color=colour,
               zorder=4, alpha=0.9, edgecolors="white", lw=0.5)
    ax.plot(bin_confs[valid], bin_accs[valid],
            "-", color=colour, lw=2, alpha=0.7)

    bs  = brier_score(y_true, probs)
    ax.set_title(f"{title}\\nECE={ece:.4f}  Brier={bs:.4f}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    return ece

# Row 1: four calibration profiles
for col, ((name, probs), colour) in enumerate(zip(all_probs.items(), colours_main)):
    plot_reliability(axes[0, col], y_te, probs,
                     name.replace("\\n", " "), colour)

# Row 2: calibration correction results + histogram
for col, ((name, probs), colour) in enumerate(zip(calib_probs.items(), colours_cal)):
    plot_reliability(axes[1, col], y_te, probs,
                     f"RF: {name}", colour)

# Last panel: probability histogram comparison
ax_hist = axes[1, 3]
for (name, probs), colour in zip(calib_probs.items(), colours_cal):
    ax_hist.hist(probs, bins=30, alpha=0.5, density=True,
                 label=name, color=colour, histtype="stepfilled")
ax_hist.set_title("Predicted Probability Distributions\\n"
                   "(Calibrated models spread more uniformly)",
                   fontsize=9, fontweight="bold")
ax_hist.set_xlabel("Predicted Probability")
ax_hist.set_ylabel("Density")
ax_hist.legend(fontsize=8)
ax_hist.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("calibration_analysis.png", dpi=110)
print()
print("  Plot saved → calibration_analysis.png")
print()
print("  KEY TAKEAWAYS:")
print("  - Well-calibrated: points sit on or near the diagonal.")
print("  - Overconfident: predicted prob > actual frequency (gap below diag).")
print("  - Underconfident: predicted prob < actual frequency (gap above diag).")
print("  - Random Forest is typically overconfident — uses majority votes.")
print("  - Platt scaling (logistic) and isotonic regression both reduce ECE.")
print("  - Isotonic regression is more flexible but needs more calibration data.")
print("  - Use calibrated models whenever probability thresholds matter.")
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