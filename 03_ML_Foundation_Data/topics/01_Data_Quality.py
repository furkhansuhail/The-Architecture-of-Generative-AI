"""
Data Quality: Missing Data, Scaling, Imbalance & Leakage
==========================================================

Four data problems that silently destroy model performance.
Each one can inflate reported metrics, produce models that fail in
production, or make an otherwise excellent algorithm useless.
Understanding how each damages training is as important as the fix.

"""

import textwrap
import re

TOPIC_NAME   = "Data Quality: Missing Data, Scaling, Imbalance & Leakage"
DISPLAY_NAME = "01 · Data Quality"
ICON         = "🧹"
SUBTITLE     = "The Four Silent Killers of Model Performance"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — MISSING DATA

### The Three Mechanisms of Missingness

The TYPE of missingness determines what you can and cannot safely do.
This is not cosmetic — using the wrong imputation for the wrong type
produces biased models that look fine on validation but fail in deployment.

**MCAR — Missing Completely At Random**
    P(missing | X, Y) = constant
    The probability of a value being missing has NO relationship to any
    variable, observed or unobserved.

    Real example: a sensor randomly fails with probability 0.05
    regardless of the reading it would have produced.

    Effect on model: if you drop MCAR rows, your dataset is smaller
    but still an unbiased sample. Simple imputation (mean/median) is safe.

    Test: Little's MCAR test. Compare missingness pattern to chance.


**MAR — Missing At Random**
    P(missing | X_observed) — depends only on observed variables
    The probability of a value being missing depends on OTHER observed
    features, but NOT on the missing value itself.

    Real example: older patients are less likely to report income.
    Income is missing, but we can predict missingness from age (observed).

    Effect on model: dropping rows creates systematic bias. You lose
    all older patients and the model never learns their patterns.
    Safe fix: impute using the relationship with observed variables
    (e.g., KNN imputation, MICE).


**MNAR — Missing Not At Random**
    P(missing | X_unobserved) — depends on the missing value itself
    The value is missing BECAUSE of what it would be.

    Real example: very high earners decline to report income.
    Very sick patients miss follow-up appointments (because they are sick).
    People with extreme weights avoid being weighed.

    Effect on model: this is the most dangerous type. ANY imputation
    creates bias because the missingness IS informative. The fact that
    a value is missing is itself a signal.

    Fix: add a binary indicator variable (is_missing) as a separate
    feature. This lets the model learn that "missing income" means
    "likely very high income" without pretending to know the value.

    Diagram 1 — Three Missingness Mechanisms:

    MCAR                  MAR                    MNAR
    ─────────────────     ─────────────────      ─────────────────
    Age  Income           Age  Income             Age   Income
    25   50k              25   50k                25    50k
    30   ???  ← random    65   ???  ← age>60      ???   ???  ← VERY high
    45   80k              30   70k                30    70k
    55   ???  ← random    70   ???  ← age>60      ???   ???  ← VERY high
    Missing ⊥ (Age,Inc)  Missing ⊥ Inc | Age     Missing ∼ Inc itself


### How Missing Data Harms Each Model Class

    ┌────────────────────────┬────────────────────────────────────────────┐
    │ Model                  │ How missingness hurts it                   │
    ├────────────────────────┼────────────────────────────────────────────┤
    │ Linear / Logistic Reg  │ NaN propagates through matrix multiply     │
    │                        │ → entire prediction becomes NaN            │
    ├────────────────────────┼────────────────────────────────────────────┤
    │ Neural Networks        │ Same NaN propagation; gradient undefined   │
    │                        │ for missing inputs                         │
    ├────────────────────────┼────────────────────────────────────────────┤
    │ Decision Trees (plain) │ Cannot split on NaN; most implementations  │
    │                        │ error or drop the row entirely             │
    ├────────────────────────┼────────────────────────────────────────────┤
    │ XGBoost / LightGBM     │ NATIVE missingness support: learns which   │
    │                        │ branch to take for NaN at each split.      │
    │                        │ Missingness becomes an implicit feature.   │
    ├────────────────────────┼────────────────────────────────────────────┤
    │ KNN                    │ Distance undefined when a feature is NaN.  │
    │                        │ Must impute before distance computation.   │
    ├────────────────────────┼────────────────────────────────────────────┤
    │ SVMs                   │ Kernel functions undefined with NaN.       │
    │                        │ Must impute before fitting.                │
    └────────────────────────┴────────────────────────────────────────────┘


### Imputation Strategies

**Mean / Median / Mode imputation** (univariate)
    Replace each missing value with the column mean (continuous),
    median (skewed/outliers), or mode (categorical).

    Pros: fast, simple, no data leakage risk.
    Cons: reduces variance artificially, destroys correlations between
    features, assumes MCAR. With 30%+ missingness, estimates become
    unreliable.

    Rule: use median (not mean) when the feature is skewed, to avoid
    the missing values being pulled toward outliers.

**KNN Imputation**
    Find the k nearest neighbours (using only observed features) and
    impute the missing value as the average of those neighbours' values.

    Pros: captures feature correlations, appropriate for MAR.
    Cons: O(n²) computation, sensitive to feature scale (must scale first),
    k is a hyperparameter.

**MICE / Iterative Imputation**
    Multiple Imputation by Chained Equations. For each feature with
    missing values:
        1. Impute missing values with a simple placeholder (e.g., mean)
        2. Regress that feature on all other features
        3. Impute missing values with regression predictions
        4. Repeat for all features → one "pass"
        5. Repeat all passes until convergence

    Pros: captures complex inter-feature relationships, appropriate for
    MAR, returns uncertainty estimates (multiple imputation).
    Cons: computationally expensive, many hyperparameters.

**Missing Indicator (binary flag)**
    Add a new boolean column: is_missing_feature_X = 1 if X is NaN.
    Then impute X as usual.

    Why: if the data is MNAR, the fact of missingness is informative.
    Letting the model see the flag allows it to learn "missing income
    → likely high earner" without corrupting the imputed value.
    Low cost; almost always worth adding for MNAR variables.

    When to drop rows: only when < 1-2% of rows are affected AND
    the missingness is MCAR. Otherwise impute.

    Critical leakage rule: fit the imputer on the TRAINING set only.
    Then transform BOTH train and test. Never fit on combined data.



### PART 2 — FEATURE SCALING

### Why Scaling Matters

Feature scaling is one of the most overlooked preprocessing steps.
It has NO effect on tree-based models but is CRITICAL for everything else.

**Effect 1 — Gradient Descent Convergence**

    Unscaled features create an elongated loss surface. Gradients
    point diagonally rather than toward the minimum, causing oscillation.

    Diagram 2 — Loss Surface Shape With vs Without Scaling:

    WITHOUT scaling                WITH scaling
    (features on different scales) (features normalised)

    w₂ │                            w₂ │
       │   ╭────────╮                  │      ╭──╮
       │ ╭──────────────╮              │    ╭──────╮
       │╭────────────────────╮         │   ╭────────╮
       │────────────────────────       │    ╰──────╯
       └──────────────────── w₁        │      ╰──╯
                                       └──────────── w₁
    Long narrow ellipses              Circular contours
    Gradient points sideways          Gradient points to minimum
    Oscillation, slow convergence     Fast convergence

    With unscaled features:
    - Feature "age" (range 18-90) has much smaller gradients than
      feature "salary" (range 20k-200k).
    - The learning rate that works for salary overshoots for age.
    - You need a tiny lr → very slow convergence.

**Effect 2 — Distance-Based Algorithms**

    KNN, K-means, SVMs with RBF kernel all compute distances.
    A feature with range [0, 10,000] dominates a feature with
    range [0, 1] even if the smaller-range feature is more informative.

    Without scaling: KNN decides "nearest neighbour" purely based on
    salary, completely ignoring education level.

**Effect 3 — Regularisation**

    L1 and L2 penalties apply equally to all weights. If salary is on
    a 100,000× larger scale than age, its coefficient will be 100,000×
    smaller in the raw data — so regularisation penalises it 100,000×
    less. The penalty is not fair across features without scaling.


### The Three Standard Scalers

**StandardScaler (z-score normalisation)**

    z = (x - μ) / σ

    Transforms each feature to have mean=0 and std=1.

    Pros: works well when features are approximately Gaussian.
    Handles negative values. Most common default.
    Cons: sensitive to outliers (an outlier shifts μ and inflates σ,
    compressing all other values). Not bounded to [0,1].

    Use when: linear/logistic regression, neural networks, SVM, PCA.

**MinMaxScaler**

    x_scaled = (x - x_min) / (x_max - x_min)    → range [0, 1]
    Or to [a, b]: a + (x_scaled) * (b - a)

    Pros: bounded output (useful when algorithm needs [0,1] range,
    e.g., sigmoid output, image pixels). Preserves zero.
    Cons: extremely sensitive to outliers. A single outlier compresses
    all other values into a tiny range near 0.

    Use when: image data (pixels already in known range), algorithms
    requiring strictly positive inputs.

**RobustScaler**

    x_scaled = (x - median) / IQR      IQR = Q3 - Q1

    Uses median and interquartile range instead of mean and std.
    Outliers have much less influence since they don't affect the median
    or IQR significantly.

    Pros: robust to outliers. Good for real-world tabular data which
    commonly contains extreme values.
    Cons: output is not bounded, not zero-mean in general.

    Use when: features have heavy tails or outliers (salaries, prices,
    medical measurements, fraud amounts).

    Diagram 3 — Effect of a Single Outlier on Each Scaler:

    Data: [1, 2, 3, 4, 5, 100]   ← 100 is an outlier

    StandardScaler:  all "normal" values → compressed near 0
                     [-0.72, -0.67, -0.62, -0.56, -0.51,  3.08]
                      ↑ everything squashed into [-0.72, -0.51]

    MinMaxScaler:    [-0.00,  0.01,  0.02,  0.03,  0.04,  1.00]
                      ↑ everything squashed into [0.00, 0.04]!

    RobustScaler:   [-1.25, -0.75, -0.25,  0.25,  0.75, 48.25]
                      ↑ normal values spread across [-1.25, 0.75] ✓


### The Golden Rule: Fit on Train, Transform Both

    WRONG (data leakage):
        scaler.fit(X_all)          ← uses test data statistics!
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

    CORRECT:
        scaler.fit(X_train)        ← only training statistics
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

    EVEN BETTER — use a Pipeline:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model",  LogisticRegression())])
        pipe.fit(X_train, y_train)      ← scaler fit on train only
        pipe.predict(X_test)            ← scaler applied to test ✓

    Fitting the scaler on all data leaks test set statistics (mean, std,
    min, max) into the training process — a subtle but real form of leakage.

    When NOT to scale: Decision Trees, Random Forests, Gradient
    Boosting (XGBoost, LightGBM). These algorithms split on thresholds
    — scaling changes the values but not the ORDER, so splits are
    unchanged. Scaling adds no benefit and wastes time.



### PART 3 — CLASS IMBALANCE


### The Accuracy Paradox

A dataset with 99% negative examples and 1% positive examples.
A classifier that ALWAYS predicts negative achieves 99% accuracy.
It catches ZERO actual positives. Accuracy is completely useless here.

    Diagram 4 — The Accuracy Paradox:

    100 examples: 99 negative, 1 positive

    "Predict everything negative" classifier:
    ┌─────────────────────────────────────────────────────────────┐
    │  Accuracy  = 99/100 = 99%    ← sounds great!                │
    │  Recall    = 0/1    = 0%     ← catches ZERO positives       │
    │  Precision = undefined (0/0)                                │
    │  F1        = 0%              ← correctly shows uselessness  │
    └─────────────────────────────────────────────────────────────┘

    ALWAYS use F1, PR-AUC, MCC, or ROC-AUC for imbalanced problems.
    NEVER report only accuracy.


### How Imbalance Hurts Model Training

**1. Decision boundary bias**

    The model learns that predicting the majority class is almost always
    "correct" under cross-entropy loss. The loss landscape is dominated
    by majority examples. The decision boundary shifts away from the
    minority class toward "predict majority."

    For a 99:1 dataset, the model's gradient is 99% driven by negative
    examples. The 1% positive examples have almost no influence on
    which direction the parameters move.

**2. Under-representation in batches**

    In mini-batch SGD with batch size 64 and 1% positive rate, a typical
    batch contains ~0.64 positive examples. Many batches see ZERO
    positives — those batches produce gradients that push the model
    exclusively toward the majority class.

**3. Threshold default**

    Most classifiers use 0.5 as the decision threshold. This is only
    appropriate when classes are balanced. For a 99:1 dataset, the
    optimal threshold may be as low as 0.1 — the model should only
    need moderate confidence to predict positive.


### Fix 1 — Threshold Adjustment

The cheapest fix. After training, sweep the threshold and pick the value
that maximises the metric you care about (F1, recall, etc.).
Use the validation set, never the test set, to pick the threshold.


### Fix 2 — Class Weights (Cost-Sensitive Learning)

Assign higher loss to minority class misclassifications:

    Loss = Σᵢ wᵢ · L(yᵢ, ŷᵢ)

    where wᵢ = N / (n_classes × n_samples_in_class_i)

    For 99:1 imbalance:
        w_negative = 100 / (2 × 99) ≈ 0.51
        w_positive = 100 / (2 ×  1) = 50.0

    Each positive misclassification now counts 100× more than a
    negative one. The model is forced to care about the minority class.

    sklearn: LogisticRegression(class_weight='balanced')
             RandomForestClassifier(class_weight='balanced')
             XGBoost: scale_pos_weight = n_negative / n_positive


### Fix 3 — Oversampling

Generate more minority class examples.

    Random oversampling: duplicate minority examples randomly.
    Simple but causes overfitting — the model memorises the duplicates.

    SMOTE: generate SYNTHETIC minority examples by interpolating
    between real minority examples in feature space.
    (Covered in Module 05 — Data Augmentation.)

    ADASYN: like SMOTE but focuses synthesis on harder-to-classify
    minority examples near the decision boundary.


### Fix 4 — Undersampling

Remove majority class examples to balance the dataset.

    Random undersampling: drop majority examples randomly.
    Risk: discard useful information from the majority class.

    Tomek links: remove majority examples that are the "nearest
    neighbour" of a minority example. Cleans the decision boundary
    without discarding as much information.

    NearMiss: keep majority examples that are CLOSEST to minority
    examples — preserves the most challenging majority examples.


### Fix 5 — Algorithmic Approaches

    Focal Loss (covered in Module 09):
        Down-weights easy examples (model is already confident on them)
        and focuses the gradient on hard, minority examples.

    BalancedBaggingClassifier:
        Trains each base estimator on a BALANCED bootstrap sample
        (equal numbers from each class). Combines bagging's variance
        reduction with undersampling's balance.

    Which fix to choose:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Imbalance ratio │ Recommended approach                          │
    ├─────────────────────────────────────────────────────────────────┤
    │ 2:1  to  5:1    │ Class weights — simple and effective          │
    │ 5:1  to  20:1   │ Class weights + threshold tuning              │
    │ 20:1 to  100:1  │ SMOTE + class weights                         │
    │ > 100:1         │ Focal loss + SMOTE + collect more data        │
    └─────────────────────────────────────────────────────────────────┘

    Matthews Correlation Coefficient (MCC) is the best single metric
    for severe imbalance:
        MCC = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    Range: −1 (inverse prediction) to 0 (random) to +1 (perfect).
    Unlike F1, MCC accounts for all four cells of the confusion matrix.



### PART 4 — DATA LEAKAGE

### What Is Data Leakage?

Data leakage occurs when information that would not be available at
prediction time is used during model training. This creates a model
that performs suspiciously well in evaluation but fails in production.

The core test: "Would I have access to this feature when making a
real prediction?" If no — it is a leaky feature.

Leakage is insidious because it is INVISIBLE in training and validation
metrics. The model looks great — then falls apart in production.


### Type 1 — Target Leakage

A feature directly encodes or strongly correlates with the label,
but would not exist before the label is known.

    Example: Predicting hospital readmission within 30 days.
    Including "discharge medication count" as a feature. Doctors
    prescribe more medications to sicker patients — patients who
    are likely to be readmitted. The feature is a CONSEQUENCE of
    the target, not a cause.

    Example: Predicting loan default. Including "debt collection
    calls received" as a feature. Collection calls happen AFTER
    default — you cannot know this before the loan is issued.

    Detection: suspiciously high feature importance for a variable
    that seems too directly related to the target. CV scores that
    are unrealistically good (>99% on a hard problem).


### Type 2 — Train-Test Contamination

Preprocessing steps that use statistics from the test set during
training. The most common form of accidental leakage.

    WRONG:
        scaler.fit(X_all)           ← test mean/std leaks into training
        imputer.fit(X_all)          ← test missingness pattern leaks
        pca.fit(X_all)              ← test variance structure leaks
        selected = select_k_best(X_all, y_all)  ← test labels leak!

    These steps must be fitted on the TRAINING set only, then
    applied to both. Use sklearn Pipelines to enforce this.


### Type 3 — Temporal Leakage

Time-ordered data where future information contaminates past predictions.

    Example: Stock price prediction. Using features computed from
    the next day's trading volume (unavailable at prediction time).

    Example: Fraud detection. A customer's total fraud history,
    including future fraud events, used to predict current transactions.

    Fix: use strict temporal splits. Train on t < cutoff.
    Test on t > cutoff. NEVER shuffle time-series data.
    Use rolling-window cross-validation, not k-fold.


### Type 4 — Duplicate Leakage

The same (or near-identical) examples appear in both train and test sets.

    Common cause: data collected at different times, then merged,
    producing duplicate rows before splitting.

    Example: patient records from the same hospital stay appearing
    as multiple rows. A split on rows (not patients) puts the same
    patient in both train and test.

    Fix: split on the ENTITY (patient ID, user ID, loan ID), not
    on individual rows. Group k-fold cross-validation.


### Type 5 — Pipeline Leakage (Feature Selection)

Using the full dataset (including test labels) to select features
before the train/test split.

    WRONG:
        important_features = select_by_correlation(X_all, y_all)
        X_selected = X_all[:, important_features]
        X_train, X_test = train_test_split(X_selected)   # too late!

    CORRECT:
        X_train, X_test = train_test_split(X_all)
        selector.fit(X_train, y_train)   # only train labels
        X_train_s = selector.transform(X_train)
        X_test_s  = selector.transform(X_test)

    BEST — use a Pipeline:
        pipe = Pipeline([
            ("select", SelectKBest(k=10)),
            ("scale",  StandardScaler()),
            ("model",  LogisticRegression()),
        ])
        cross_val_score(pipe, X_train, y_train, cv=5)
        # Each fold: selector fit on 4/5 of data, transforms 1/5 ✓


### Leakage Detection Checklist

    ┌─────────────────────────────────────────────────────────────────┐
    │  Red flag                      │ Possible leakage type          │
    ├─────────────────────────────────────────────────────────────────┤
    │  CV score seems too good       │ Any type                       │
    │  A feature has >50% importance │ Target leakage                 │
    │  Performance drops sharply     │                                │
    │    in production               │ Temporal or target leakage     │
    │  Feature name looks like the   │                                │
    │    target (e.g., "outcome_")   │ Target leakage                 │
    │  Preprocessing before split    │ Train-test contamination       │
    │  Time-ordered data shuffled    │ Temporal leakage               │
    │  Multiple rows per entity      │ Duplicate leakage              │
    └─────────────────────────────────────────────────────────────────┘



### PART 5 — OUTLIER DETECTION & TREATMENT

### Outlier ≠ Error

The most important principle: an outlier is an extreme value, not
necessarily a wrong one. Before removing any point you must ask:
"Is this an error, or is this the most important signal in my dataset?"

    A fraudulent transaction of $50,000 when the typical transaction
    is $40 is an outlier — and exactly what the fraud model must catch.
    Removing it destroys the signal you were hired to find.

    A temperature reading of 9,999°C from a broken sensor IS an error.
    These require completely different treatments.

Classification of outliers:

    Point outlier:   a single value far from the bulk of the distribution
    Contextual:      normal globally but anomalous in context
                     (30°C is normal in July, anomalous in January)
    Collective:      a group of values that together are anomalous even
                     though each individual value looks normal


### Detection Methods

**Univariate — Z-score**

    z = (x - μ) / σ    flag if |z| > 3

    Pros: simple and interpretable.
    Cons: assumes Gaussian distribution. Sensitive to the very outliers
    it tries to detect (the outlier inflates μ and σ, masking itself).
    Use modified Z-score with median/MAD instead for robustness:

        z_modified = 0.6745 × (x - median) / MAD
        MAD = median(|xᵢ - median(x)|)
        Flag if |z_modified| > 3.5

**Univariate — IQR Fence (Tukey)**

    Q1, Q3 = 25th and 75th percentile
    IQR    = Q3 − Q1
    Lower fence: Q1 − 1.5 × IQR
    Upper fence: Q3 + 1.5 × IQR     (standard boxplot whiskers)
    Severe:      Q1 − 3.0 × IQR  to  Q3 + 3.0 × IQR

    Pros: non-parametric, robust, matches boxplot intuition.
    Cons: univariate only — misses multivariate outliers where no single
    feature is extreme but the combination is impossible.

    Diagram 5 — IQR Fence:

    ────Q1─────────────Q3────
       |←── IQR ──→|
    ←1.5×IQR         1.5×IQR→
    [lower fence]  [upper fence]
    *  ← outlier       outlier → *

**Multivariate — Isolation Forest**

    Randomly partition the feature space with axis-aligned cuts.
    Anomalies are isolated by fewer cuts on average (they are in sparse
    regions, easier to separate from the rest).

    Anomaly score = average path length to isolation (shorter = more anomalous)
    contamination: expected fraction of outliers (hyperparameter)

    Pros: works in high dimensions, no distributional assumption,
    scales to large datasets, handles mixed outlier types.
    Cons: random — results vary between runs without a seed.

**Multivariate — Local Outlier Factor (LOF)**

    Compares a point's local density to its neighbours' local density.
    A point in a sparse neighbourhood surrounded by dense clusters is
    flagged as an outlier — even if it is not globally extreme.

    LOF ≈ 1:   normal (similar density to neighbours)
    LOF >> 1:  outlier (much lower density than neighbours)

    Pros: captures local structure, detects contextual outliers.
    Cons: O(n²) distance computation, requires scaling first.


### Treatment Strategies

    ┌─────────────────────────────────────────────────────────────────┐
    │ Treatment         │ When to use                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │ Remove the row    │ ONLY if confirmed measurement / entry error │
    │                   │ AND < 1% of data. Never remove real signal. │
    ├─────────────────────────────────────────────────────────────────┤
    │ Winsorize / Cap   │ Cap at a percentile (e.g., 1st–99th).       │
    │                   │ Keeps the row but limits outlier influence. │
    │                   │ Safe default for continuous features.       │
    ├─────────────────────────────────────────────────────────────────┤
    │ Log / sqrt        │ Right-skewed features (salaries, prices,    │
    │ transform         │ counts). Compresses the tail without        │
    │                   │ discarding data. log1p(x) handles zeros.    │
    ├─────────────────────────────────────────────────────────────────┤
    │ Robust scaler     │ Let RobustScaler reduce the outlier's       │
    │                   │ influence implicitly via IQR normalisation. │
    ├─────────────────────────────────────────────────────────────────┤
    │ Flag + keep       │ Add is_extreme_X = 1 and let the model      │
    │                   │ learn from both the value and the flag.     │
    ├─────────────────────────────────────────────────────────────────┤
    │ Separate model    │ If outliers form a coherent group (e.g.,    │
    │                   │ VIP customers), train a dedicated model.    │
    └─────────────────────────────────────────────────────────────────┘

    The pipeline rule: outlier treatment must be fitted on train only
    (the 99th percentile used for winsorizing must come from training).
    Use a custom sklearn Transformer inside a Pipeline.


### PART 6 — CATEGORICAL ENCODING

### Why Encoding Is a Data Quality Problem

Most ML algorithms require numeric inputs. Converting categories to
numbers is not cosmetic — the WRONG encoding silently introduces either
false ordinal relationships or target leakage.


### Encoding Methods

**Label Encoding (ordinal encoding)**

    Maps each category to an integer: Red→0, Blue→1, Green→2.

    DANGER: implies Red < Blue < Green — an ordering that does not
    exist. The model will use this as a numeric feature and learn
    that Green (2) is "twice" Blue (1).

    When to use: ONLY for genuinely ordinal features (Small, Medium,
    Large → 0, 1, 2). Never for nominal categories.

**One-Hot Encoding (OHE)**

    Creates k binary columns for k categories.
    One column is always 1, all others 0.

    Drop-first rule: drop one column (the "reference category") to
    avoid perfect multicollinearity in linear models. Tree models
    do not need this.

    Pros: no false ordinal relationship. Correct for linear models.
    Cons: creates k new columns — catastrophic for high-cardinality
    features (500 cities → 500 columns). High memory. Sparse matrices.

    Rule: safe for < 10–15 unique values. Avoid for > 50.

**Frequency / Count Encoding**

    Replace each category with its frequency (proportion) in the dataset.
    City→0.35 means that city appears in 35% of rows.

    Pros: no dimensionality explosion, no leakage risk.
    Cons: two categories with the same frequency are treated identically.

    Leakage safety: compute frequencies from training set only.

**Target Encoding (mean encoding)**

    Replace each category with the mean of the target Y for that category.
    Paris → mean(Y | city == "Paris")

    Pros: very powerful for high-cardinality features. Captures the
    relationship between category and target directly.
    Cons: SEVERE LEAKAGE RISK if done naively. If you compute the mean
    of Y for Paris using all data (including test), you have leaked the
    target into the features.

    Correct approach: use out-of-fold target encoding inside each
    cross-validation fold. sklearn's TargetEncoder handles this.
    Add smoothing: blend category mean with global mean to handle
    rare categories (few samples → unreliable mean):

        encoded = (count × category_mean + m × global_mean) / (count + m)
        m controls shrinkage toward global mean (m ≈ 10–50 typical)

**Hashing / Hash Encoding**

    Apply a hash function to map categories to a fixed number of buckets
    (e.g., 32 columns regardless of cardinality).

    Pros: fixed dimensionality regardless of cardinality. Handles
    unseen categories at test time gracefully.
    Cons: hash collisions (two categories map to the same bucket),
    not interpretable.

    Use when: extremely high cardinality (user IDs, product SKUs),
    online learning with unknown vocabulary.

**Embeddings (learned representations)**

    Learn a dense vector per category jointly with the model.
    Used in neural networks for categorical inputs (e.g., entity
    embeddings for tabular data, word2vec for text).

    Rule of thumb: embedding dimension ≈ min(50, (n_categories + 1) // 2)


### Encoding Decision Guide

    ┌─────────────────────────────────────────────────────────────────┐
    │ Feature type          │ Recommended encoding                    │
    ├─────────────────────────────────────────────────────────────────┤
    │ Binary (yes/no)       │ 0/1 label encoding                      │
    │ Ordinal (S/M/L/XL)    │ Ordinal label encoding (0/1/2/3)        │
    │ Nominal, k ≤ 15       │ One-hot encoding (drop_first=True)      │
    │ Nominal, 15 < k ≤ 50  │ Target encoding (with CV fold safety)   │
    │ Nominal, k > 50       │ Target encoding or hashing              │
    │ Neural network        │ Learned embeddings                      │
    │ Unseen categories     │ Hashing or frequency encoding           │
    └─────────────────────────────────────────────────────────────────┘

    The pipeline rule: ALL encoding must happen inside the pipeline,
    fitted on training data only. Target encoding is especially
    dangerous — use sklearn's TargetEncoder with cv parameter.



### PART 7 — DISTRIBUTION SHIFT & DATA DRIFT

### The Fifth Silent Killer

A model can be flawlessly trained, validated, and deployed — then
silently degrade in production as the world changes around it.
Distribution shift is the gap between the data you trained on and
the data arriving at inference time.

It is undetectable by standard train/test metrics. By definition,
your evaluation used historical data that matched training.


### Three Types of Shift

**Covariate Shift — P(X) changes, P(Y|X) stays the same**

    The input distribution changes but the relationship between
    inputs and outputs does not.

    Example: a credit model trained on 25–45 year olds is deployed
    to a new region where most applicants are 55–70. The feature
    "age" now has a completely different distribution. The model's
    learned decision boundary for age was calibrated on a different
    population.

    Detection: monitor feature distributions (mean, variance,
    percentiles) over time. PSI and KS test per feature.

    Fix: importance weighting — reweight training examples so that
    the training distribution matches the deployment distribution.
    Use density ratio estimation: w(x) = P_new(x) / P_old(x).

**Label Shift (Prior Shift) — P(Y) changes, P(X|Y) stays the same**

    The class proportions change over time, but the features of each
    class remain the same.

    Example: a fraud model trained on 0.5% fraud rate when fraud rate
    rises to 2.0%. The model's decision threshold (calibrated for 0.5%)
    will miss most fraud at the new base rate.

    Fix: recalibrate the decision threshold. Estimate P_new(Y) and
    use Bayes' theorem to adjust the posterior.

**Concept Drift — P(Y|X) changes**

    The fundamental relationship between inputs and outputs changes.
    This is the most dangerous form — no amount of reweighting fixes it.

    Example: "high income → low default risk" was true in 2019.
    After a recession, many high-income borrowers began defaulting
    at much higher rates. The concept has changed.

    Gradual drift: the relationship shifts slowly over months.
    Sudden drift: a shock event (pandemic, regulation change) causes
    an abrupt change.
    Recurring drift: seasonal patterns (Christmas fraud patterns differ
    from non-seasonal fraud patterns).

    Fix: retrain on recent data, use windowed training, maintain
    ensemble of models from different time periods.


### Detection Methods

**Population Stability Index (PSI)**

    PSI = Σₖ (P_new(k) − P_old(k)) × ln(P_new(k) / P_old(k))

    Binned version of KL divergence. Standard threshold:
        PSI < 0.1:   no significant shift — model still valid
        PSI 0.1–0.2: moderate shift — investigate
        PSI > 0.2:   major shift — retrain required

    Apply PSI to each input feature and to the model's output score.

**Kolmogorov-Smirnov (KS) Test**

    Two-sample KS test: does the new batch come from the same
    distribution as the training data?
    Returns a p-value — reject H₀ (same distribution) if p < 0.05.
    Apply per feature. Multiple testing correction needed.

**Chi-squared Test (categorical features)**

    Compares observed vs expected category frequencies between
    training and new data. Detects shifts in categorical distributions.

**Monitoring the Prediction Distribution**

    Even without ground truth labels (which may lag by days/months),
    monitor the distribution of the model's output scores.
    If the score distribution shifts, the model has encountered data
    it is not calibrated for.


### Mitigation Strategies

    ┌─────────────────────────────────────────────────────────────────┐
    │ Shift type      │ Fix                                           │
    ├─────────────────────────────────────────────────────────────────┤
    │ Covariate shift │ Importance weighting, domain adaptation       │
    │ Label shift     │ Recalibrate threshold, prior correction       │
    │ Concept drift   │ Retrain on recent data, windowed training     │
    │ All types       │ Monitor PSI + KS, set retraining triggers     │
    │                 │ Maintain model versioning and rollback        │
    └─────────────────────────────────────────────────────────────────┘


### PART 8 — FIX ORDERING & INTERACTIONS

### The Correct Preprocessing Order

Order matters. Many bugs arise from applying transformations out of
sequence. The canonical correct order:

    1. SPLIT first (train / validation / test) — before anything else
    2. Identify missingness mechanism (MCAR/MAR/MNAR) on training set
    3. Create missing indicator columns (for MNAR features)
    4. Impute missing values (fit imputer on train, transform all)
    5. Encode categoricals (fit encoder on train, transform all)
    6. Detect and treat outliers (fit winsorizer on train, transform all)
    7. Scale features (fit scaler on train, transform all)
    8. Resample for imbalance (ONLY on training set — never test)
    9. Feature selection (fit selector on train only)
    10. Train model

    Steps 3–9 must ALL be inside a sklearn Pipeline or equivalent.
    Steps 3–7 use fit on train, transform on train AND test.
    Step 8 uses transform on train ONLY (never oversample the test set).


### Critical Interaction: SMOTE Must Come After Splitting

    WRONG — common mistake:
        X_resampled, y_resampled = SMOTE().fit_resample(X_all, y_all)
        X_tr, X_te = train_test_split(X_resampled)
        # Synthetic samples may be near-duplicates of test examples.
        # Test set is polluted with "near-future" information.

    CORRECT:
        X_tr, X_te = train_test_split(X_all)
        X_tr_res, y_tr_res = SMOTE().fit_resample(X_tr, y_tr)
        # Synthetic samples only generated from training distribution.
        # Test set is untouched.

    This is one of the most common Kaggle "winning" bugs — SMOTE before
    split inflates scores artificially.


### Critical Interaction: Target Encoding Is a Leakage Trap

    WRONG:
        df["city_encoded"] = df.groupby("city")["target"].transform("mean")
        X_tr, X_te = train_test_split(df)
        # Test target values contributed to encoding ALL rows including train.

    CORRECT:
        X_tr, X_te = train_test_split(df)
        encoder = TargetEncoder(cv=5).fit(X_tr["city"], y_tr)
        X_tr["city_encoded"] = encoder.transform(X_tr["city"])
        X_te["city_encoded"] = encoder.transform(X_te["city"])
        # cv=5 means each training row's encoding used only the other 4 folds.


### Critical Interaction: Class Weights Change Probability Calibration

    When you use class_weight='balanced', the model learns shifted
    decision probabilities. The predicted probability P(y=1|x) is no
    longer the true posterior probability — it is inflated toward 0.5.

    Consequence: threshold = 0.5 is NO LONGER the correct threshold
    after using class weights. You must always tune the threshold on
    a held-out validation set after applying class weights.

    With imbalance ratio r = n_neg / n_pos and balanced class weights,
    the effective threshold shifts approximately to:
        optimal_threshold ≈ 1 / (1 + r)
    (e.g., for 10:1 imbalance: threshold ≈ 1/11 ≈ 0.09)


### Complete Interaction Map

    ┌──────────────────────────────────────────────────────────────────┐
    │ Step              │ Depends on                │ Feeds into       │
    ├──────────────────────────────────────────────────────────────────┤
    │ Missing indicator │ MNAR identification       │ Imputation step  │
    │ Imputation        │ Split (train stats only)  │ Encoding, scale  │
    │ Encoding          │ Split (train cats only)   │ Scaling, model   │
    │ Outlier treatment │ Split (train percentiles) │ Scaling, model   │
    │ Scaling           │ Post-imputation values    │ Model, SMOTE     │
    │ SMOTE             │ Post-split train only     │ Model training   │
    │ Feature selection │ Post-split train+labels   │ Model training   │
    │ Threshold tuning  │ Class weights used?       │ Final evaluation │
    └──────────────────────────────────────────────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Missing Data — MCAR/MAR/MNAR Detection & Imputation Comparison": {
        "description": (
            "Generate datasets with MCAR, MAR, and MNAR missingness. "
            "Show how each mechanism distorts the data distribution differently. "
            "Compare mean, median, KNN, and iterative imputation strategies."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)

np.random.seed(42)

print("=" * 65)
print("  MISSING DATA: MCAR vs MAR vs MNAR")
print("=" * 65)
print()

# ── Generate clean dataset: 3 features, binary target ─────────────────────
n = 1000
X_clean = np.column_stack([
    np.random.normal(0,  1, n),     # feature 0: income proxy
    np.random.normal(50, 15, n),    # feature 1: age
    np.random.normal(0,  1, n),     # feature 2: credit score proxy
])
y = (X_clean[:, 0] + 0.5*X_clean[:, 2] > 0).astype(int)

def introduce_missing(X, mechanism, missing_rate=0.20):
    Xm = X.copy().astype(float)
    n  = len(Xm)
    if mechanism == "MCAR":
        # Missing completely at random — no relationship to any variable
        mask = np.random.rand(n) < missing_rate
        Xm[mask, 0] = np.nan

    elif mechanism == "MAR":
        # Missing depends on observed variable (age): older -> more missing
        age_norm = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
        prob_missing = missing_rate * 2 * age_norm   # higher age -> more missing
        prob_missing = np.clip(prob_missing, 0, 0.95)
        mask = np.random.rand(n) < prob_missing
        Xm[mask, 0] = np.nan

    elif mechanism == "MNAR":
        # Missing depends on the VALUE ITSELF: high income -> more missing
        income_high = X[:, 0] > np.percentile(X[:, 0], 75)
        mask = income_high & (np.random.rand(n) < 0.70)
        Xm[mask, 0] = np.nan

    return Xm

# ── Show how each mechanism distorts the observed distribution ─────────────
print("  HOW MISSINGNESS DISTORTS OBSERVED DISTRIBUTION OF FEATURE 0:")
print(f"  {'Mechanism':10s} | {'% Missing':>10} | {'Observed Mean':>14} | "
      f"{'True Mean':>10} | {'Bias':>8}")
print(f"  {'─'*60}")

true_mean = X_clean[:, 0].mean()
for mech in ["MCAR", "MAR", "MNAR"]:
    Xm = introduce_missing(X_clean, mech, 0.20)
    observed = Xm[:, 0][~np.isnan(Xm[:, 0])]
    pct_miss = np.isnan(Xm[:, 0]).mean() * 100
    obs_mean = observed.mean()
    bias     = obs_mean - true_mean
    print(f"  {mech:10s} | {pct_miss:9.1f}% | {obs_mean:14.4f} | "
          f"{true_mean:10.4f} | {bias:+8.4f}")

print()
print("  MCAR: observed mean ≈ true mean (unbiased sample)")
print("  MAR:  slight bias (older patients missing → sample skewed younger)")
print("  MNAR: LARGE bias (high earners missing → observed mean too LOW)")
print()

# ── Compare imputation strategies on MNAR (hardest case) ──────────────────
print("  IMPUTATION STRATEGY COMPARISON (MNAR, 20% missing):")
print()
Xm_mnar = introduce_missing(X_clean, "MNAR", 0.20)

# Add missingness indicator feature
missing_indicator = np.isnan(Xm_mnar[:, 0]).astype(float).reshape(-1, 1)
Xm_with_flag = np.hstack([Xm_mnar, missing_indicator])

X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y,  test_size=0.3, random_state=0)
Xm_tr, Xm_te           = train_test_split(Xm_mnar,    test_size=0.3, random_state=0)
Xf_tr, Xf_te           = train_test_split(Xm_with_flag, test_size=0.3, random_state=0)

strategies = [
    ("No missing (oracle)",    X_tr,  X_te,
     None),
    ("Drop rows with NaN",     X_tr[~np.isnan(Xm_tr[:,0])],
     X_te,  None),
    ("Mean imputation",        Xm_tr, Xm_te,
     SimpleImputer(strategy="mean")),
    ("Median imputation",      Xm_tr, Xm_te,
     SimpleImputer(strategy="median")),
    ("KNN imputation (k=5)",   Xm_tr, Xm_te,
     KNNImputer(n_neighbors=5)),
    ("Iterative (MICE)",       Xm_tr, Xm_te,
     IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=0)),
    ("Mean + miss indicator",  Xf_tr, Xf_te,
     SimpleImputer(strategy="mean")),
]

scaler_ref = StandardScaler().fit(X_tr)

print(f"  {'Strategy':30s} | {'Train n':>8} | {'Test Acc':>10} | Notes")
print(f"  {'─'*70}")

for name, X_train_s, X_test_s, imputer in strategies:
    try:
        if imputer is not None:
            X_train_imp = imputer.fit_transform(X_train_s)
            X_test_imp  = imputer.transform(X_test_s)
        else:
            X_train_imp = X_train_s
            X_test_imp  = X_test_s

        if np.isnan(X_train_imp).any() or np.isnan(X_test_imp).any():
            print(f"  {name:30s} | {'NaN':>8} | {'NaN':>10} | still has NaN")
            continue

        sc = StandardScaler().fit(X_train_imp)
        clf = LogisticRegression(max_iter=500, random_state=0)
        clf.fit(sc.transform(X_train_imp), y_tr[:len(X_train_imp)])
        acc = accuracy_score(y_te, clf.predict(sc.transform(X_test_imp)))

        note = ""
        if "oracle" in name: note = "upper bound"
        if "Drop"   in name: note = "loses 70% of test rows if also applied to test"
        if "indicator" in name: note = "best for MNAR: flag encodes missingness as signal"

        print(f"  {name:30s} | {len(X_train_imp):8d} | {acc:10.4f} | {note}")
    except Exception as e:
        print(f"  {name:30s} | {'ERR':>8} | {'ERR':>10} | {str(e)[:30]}")

print()
print("  KEY LESSON:")
print("  For MNAR: mean/median imputation introduces systematic bias.")
print("  The 'missing indicator' flag lets the model learn that")
print("  NaN itself is an informative signal (high earner pattern).")

# ── Plot missingness patterns ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("How Missingness Mechanism Distorts Observed Distribution",
             fontsize=12, fontweight="bold")

for ax, mech, colour in zip(axes,
        ["MCAR", "MAR", "MNAR"],
        ["steelblue", "seagreen", "tomato"]):
    Xm = introduce_missing(X_clean, mech, 0.20)
    observed   = Xm[:, 0][~np.isnan(Xm[:, 0])]
    true_vals  = X_clean[:, 0]
    ax.hist(true_vals,  bins=40, alpha=0.35, color="gray",  label="True (full)")
    ax.hist(observed,   bins=40, alpha=0.65, color=colour,  label="Observed (non-missing)")
    ax.axvline(true_vals.mean(),  color="black",  lw=2, linestyle="--", label="True mean")
    ax.axvline(observed.mean(),   color=colour,   lw=2, linestyle="-",  label="Observed mean")
    pct = np.isnan(Xm[:, 0]).mean() * 100
    ax.set_title(mech + f" ({pct:.0f}% missing)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.set_xlabel("Feature 0 value"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("missing_data_mechanisms.png", dpi=120)
print()
print("  Plot saved -> missing_data_mechanisms.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Feature Scaling — Gradient Descent Convergence With vs Without": {
        "description": (
            "Show concretely how unscaled features create a distorted loss landscape "
            "and cause gradient descent to oscillate. Measure epochs to convergence "
            "with StandardScaler, MinMaxScaler, and no scaling."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)

np.random.seed(0)
print("=" * 65)
print("  FEATURE SCALING: GRADIENT DESCENT CONVERGENCE")
print("=" * 65)
print()

# ── Create a deliberately badly-scaled dataset ────────────────────────────
n = 800
X_base, y = make_classification(n_samples=n, n_features=4,
                                 n_informative=3, n_redundant=1, random_state=42)

# Deliberately scale features to wildly different ranges
X_unscaled = X_base.copy()
X_unscaled[:, 0] *= 100000   # salary: range ~ [-300k, 300k]
X_unscaled[:, 1] *= 1        # age: range ~ [-3, 3]
X_unscaled[:, 2] *= 50       # price: range ~ [-150, 150]
X_unscaled[:, 3] *= 0.001    # small feature: ~ [-0.003, 0.003]

X_tr, X_te, y_tr, y_te = train_test_split(X_unscaled, y,
                                            test_size=0.25, random_state=1)

print("  FEATURE RANGES IN UNSCALED DATA:")
print(f"  {'Feature':12s} | {'Min':>12} | {'Max':>12} | {'Std':>12}")
print(f"  {'─'*52}")
names = ["Salary(x1e5)", "Age(x1)", "Price(x50)", "Tiny(x1e-3)"]
for i, fname in enumerate(names):
    col = X_unscaled[:, i]
    print(f"  {fname:12s} | {col.min():12.3f} | {col.max():12.3f} | {col.std():12.3f}")
print()

# ── Train with manual gradient descent to show convergence ────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train_gd(X, y, lr=1e-6, n_epochs=200):
    """Mini logistic regression via gradient descent. Returns loss per epoch."""
    m, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(n_epochs):
        p   = sigmoid(X @ w + b)
        p   = np.clip(p, 1e-7, 1 - 1e-7)
        loss = -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
        losses.append(loss)
        grad_w = (p - y) @ X / m
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return losses, w, b

# Learning rate tuned per scaler (to make comparison fair)
configs = [
    ("No Scaling",       X_tr,                                X_te,  5e-9,  "tomato"),
    ("StandardScaler",   StandardScaler().fit_transform(X_tr),None,  0.1,   "steelblue"),
    ("MinMaxScaler",     MinMaxScaler().fit_transform(X_tr),  None,  0.1,   "seagreen"),
    ("RobustScaler",     RobustScaler().fit_transform(X_tr),  None,  0.1,   "purple"),
]

# Prepare test sets
std_sc  = StandardScaler().fit(X_tr)
mm_sc   = MinMaxScaler().fit(X_tr)
rob_sc  = RobustScaler().fit(X_tr)

configs[1] = ("StandardScaler", std_sc.fit_transform(X_tr), std_sc.transform(X_te),  0.1,  "steelblue")
configs[2] = ("MinMaxScaler",   mm_sc.fit_transform(X_tr),  mm_sc.transform(X_te),   0.1,  "seagreen")
configs[3] = ("RobustScaler",   rob_sc.fit_transform(X_tr), rob_sc.transform(X_te),  0.1,  "purple")

N_EPOCHS = 300
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of Feature Scaling on Gradient Descent Convergence",
             fontsize=12, fontweight="bold")

print(f"  {'Scaler':18s} | {'Final Loss':>11} | {'Converged?':>11} | {'Test Acc':>10}")
print(f"  {'─'*58}")

for name, Xtr_s, Xte_s, lr, colour in configs:
    losses, w, b = train_gd(Xtr_s, y_tr, lr=lr, n_epochs=N_EPOCHS)

    final_loss  = losses[-1]
    converged   = final_loss < 0.55   # below random baseline

    if Xte_s is not None:
        p_te = sigmoid(Xte_s @ w + b)
        acc  = ((p_te > 0.5).astype(int) == y_te).mean()
    else:
        # Use original test data with tiny lr
        p_te = sigmoid(X_te @ w + b)
        acc  = ((p_te > 0.5).astype(int) == y_te).mean()

    conv_str = "Yes" if converged else "No (stuck)"
    print(f"  {name:18s} | {final_loss:11.4f} | {conv_str:>11} | {acc:10.4f}")

    axes[0].plot(losses, colour, lw=2, label=name)
    axes[1].plot(losses[-50:], colour, lw=2)

axes[0].set_title("Loss Curve — All 300 Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1.5)

axes[1].set_title("Loss Curve — Last 50 Epochs (convergence detail)")
axes[1].set_xlabel("Epoch (last 50)"); axes[1].set_ylabel("BCE Loss")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("scaling_convergence.png", dpi=120)

print()
print("  KEY LESSON:")
print("  Without scaling, gradient descent requires an extremely tiny")
print("  learning rate (5e-9) to avoid exploding, and still barely moves.")
print("  With any scaling, the same lr=0.1 works fine and converges fast.")
print()
print("  Gradient at epoch 1 without scaling:")
X_demo = X_tr.copy()
p0 = sigmoid(X_demo @ np.zeros(4) + 0)
g  = (p0 - y_tr) @ X_demo / len(y_tr)
print(f"    Salary gradient:    {g[0]:+.2e}  (enormous — overshoots)")
print(f"    Age gradient:       {g[1]:+.2e}")
print(f"    Price gradient:     {g[2]:+.2e}")
print(f"    Tiny feat gradient: {g[3]:+.2e}  (negligible — never moves)")
print()
print("  After StandardScaler, all gradients are on the same scale,")
print("  so a single learning rate works for every parameter.")
print()
print("  Plot saved -> scaling_convergence.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Scaler Comparison — Outlier Sensitivity Demonstrated": {
        "description": (
            "Compare StandardScaler, MinMaxScaler, and RobustScaler "
            "on a dataset with outliers. Show how outliers compress the "
            "distribution with Standard/MinMax but not with Robust. "
            "Then compare downstream model performance."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)

np.random.seed(7)
print("=" * 65)
print("  SCALER COMPARISON: OUTLIER SENSITIVITY")
print("=" * 65)
print()

# ── Dataset with realistic outliers ───────────────────────────────────────
n = 600
X, y = make_classification(n_samples=n, n_features=4,
                            n_informative=3, n_redundant=1, random_state=0)

# Inject outliers: 3% of rows have extreme salary values
outlier_idx = np.random.choice(n, size=int(0.03*n), replace=False)
X[outlier_idx, 0] *= 50   # extreme outliers in feature 0

print(f"  Dataset: {n} samples, 4 features, 3% outliers in feature 0")
print()

# ── Show how each scaler handles outliers numerically ────────────────────
scalers = [
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler",   MinMaxScaler()),
    ("RobustScaler",   RobustScaler()),
]

print("  FEATURE 0 AFTER SCALING (showing outlier effect on normal values):")
print()

X_demo = X[:, 0].copy()
normal_mask   = np.ones(len(X_demo), dtype=bool)
normal_mask[outlier_idx] = False

print(f"  {'Scaler':18s} | {'Normal val range':>18} | {'Outlier val range':>18} | {'Compression?'}")
print(f"  {'─'*76}")

raw_normal_range = f"[{X_demo[normal_mask].min():.2f}, {X_demo[normal_mask].max():.2f}]"
raw_outlier_range = f"[{X_demo[outlier_idx].min():.1f}, {X_demo[outlier_idx].max():.1f}]"
print(f"  {'Raw (unscaled)':18s} | {raw_normal_range:>18} | {raw_outlier_range:>18} | baseline")

for name, scaler in scalers:
    X_s  = scaler.fit_transform(X[:, 0].reshape(-1, 1)).ravel()
    n_rng = f"[{X_s[normal_mask].min():.3f}, {X_s[normal_mask].max():.3f}]"
    o_rng = f"[{X_s[outlier_idx].min():.1f}, {X_s[outlier_idx].max():.1f}]"
    n_span = X_s[normal_mask].max() - X_s[normal_mask].min()
    compress = "SEVERE" if n_span < 0.1 else ("moderate" if n_span < 0.5 else "none")
    print(f"  {name:18s} | {n_rng:>18} | {o_rng:>18} | {compress}")

print()
print("  MinMaxScaler compresses all normal values into [0, 0.02]!")
print("  StandardScaler compresses them into [-0.3, 0.3].")
print("  RobustScaler keeps normal values in a wide, useful range.")
print()

# ── Compare KNN performance (distance-based — very sensitive to scale) ────
print("  KNN CLASSIFIER PERFORMANCE (5-fold CV, k=7):")
print("  KNN uses Euclidean distance — extremely sensitive to scaling.")
print()
print(f"  {'Preprocessing':25s} | {'CV Accuracy':>12} | {'Std':>8}")
print(f"  {'─'*50}")

knn = KNeighborsClassifier(n_neighbors=7)
configs = [
    ("No scaling",       X),
    ("StandardScaler",   StandardScaler().fit_transform(X)),
    ("MinMaxScaler",     MinMaxScaler().fit_transform(X)),
    ("RobustScaler",     RobustScaler().fit_transform(X)),
]

for name, X_proc in configs:
    scores = cross_val_score(knn, X_proc, y, cv=5, scoring="accuracy")
    flag = " <- recommended" if name == "RobustScaler" else ""
    print(f"  {name:25s} | {scores.mean():12.4f} | {scores.std():8.4f}{flag}")

print()

# ── Visualise compression ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Scaler Comparison: Effect of Outliers on Feature Distribution",
             fontsize=12, fontweight="bold")

plot_configs = [
    ("Raw (unscaled)", X[:, 0],
     "gray"),
    ("StandardScaler", StandardScaler().fit_transform(X[:, 0].reshape(-1,1)).ravel(),
     "steelblue"),
    ("MinMaxScaler",   MinMaxScaler().fit_transform(X[:, 0].reshape(-1,1)).ravel(),
     "tomato"),
    ("RobustScaler",   RobustScaler().fit_transform(X[:, 0].reshape(-1,1)).ravel(),
     "seagreen"),
]

for ax, (name, vals, colour) in zip(axes.ravel(), plot_configs):
    # Show only the non-outlier range for clarity (clip at 99.9th percentile)
    clip_high = np.percentile(vals, 99.5)
    clip_low  = np.percentile(vals, 0.5)
    clipped   = np.clip(vals, clip_low, clip_high)
    ax.hist(clipped, bins=40, color=colour, alpha=0.75, density=True)
    ax.hist(vals[normal_mask], bins=40, color="gray", alpha=0.35,
            density=False, label="Normal values")
    span = vals[normal_mask].max() - vals[normal_mask].min()
    ax.set_title(f"{name} — Normal span={span:.3f}", fontsize=10)
    ax.set_xlabel("Scaled value"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("scaler_comparison.png", dpi=120)
print("  Plot saved -> scaler_comparison.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Class Imbalance — Accuracy Paradox & 5 Fix Strategies": {
        "description": (
            "Demonstrate the accuracy paradox on a 98:2 imbalanced dataset. "
            "Compare 5 strategies: baseline, threshold tuning, class weights, "
            "SMOTE oversampling, and undersampling. "
            "Show why F1/MCC are better metrics than accuracy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)

np.random.seed(0)
print("=" * 65)
print("  CLASS IMBALANCE: ACCURACY PARADOX AND 5 FIX STRATEGIES")
print("=" * 65)
print()

# ── Severely imbalanced dataset: 98:2 ─────────────────────────────────────
X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=6,
    weights=[0.98, 0.02], flip_y=0.005, random_state=42)

X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1)

print(f"  Dataset: {len(y)} samples")
print(f"  Majority (class 0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Minority (class 1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
print()

def evaluate(name, y_true, y_pred, y_prob):
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred)
    prauc = average_precision_score(y_true, y_prob)
    return acc, f1, rec, prec, mcc, prauc

# ── BASELINE: plain logistic regression ──────────────────────────────────
clf_base = LogisticRegression(max_iter=500, random_state=0).fit(X_tr, y_tr)
y_pred_base = clf_base.predict(X_te)
y_prob_base = clf_base.predict_proba(X_te)[:, 1]

# ── FIX 1: Threshold tuning ───────────────────────────────────────────────
# Find threshold maximising F1 on validation
val_size  = int(0.2 * len(X_tr))
X_val, y_val = X_tr[:val_size], y_tr[:val_size]
X_tr2, y_tr2 = X_tr[val_size:], y_tr[val_size:]

clf_thr = LogisticRegression(max_iter=500, random_state=0).fit(X_tr2, y_tr2)
val_probs = clf_thr.predict_proba(X_val)[:, 1]

best_f1, best_thr = 0, 0.5
for thr in np.arange(0.05, 0.80, 0.01):
    pred_thr = (val_probs > thr).astype(int)
    f1_thr   = f1_score(y_val, pred_thr, zero_division=0)
    if f1_thr > best_f1:
        best_f1, best_thr = f1_thr, thr

y_pred_thr  = (clf_thr.predict_proba(X_te)[:, 1] > best_thr).astype(int)
y_prob_thr  = clf_thr.predict_proba(X_te)[:, 1]

# ── FIX 2: Class weights ──────────────────────────────────────────────────
clf_wt = LogisticRegression(max_iter=500, class_weight="balanced",
                             random_state=0).fit(X_tr, y_tr)
y_pred_wt  = clf_wt.predict(X_te)
y_prob_wt  = clf_wt.predict_proba(X_te)[:, 1]

# ── FIX 3: SMOTE oversampling (manual implementation) ────────────────────
def smote_simple(X_min, n_synthetic):
    """Generate synthetic minority samples via linear interpolation."""
    pass  # NearestNeighbors already defined at module level
    nn = NearestNeighbors(n_neighbors=6).fit(X_min)
    _, indices = nn.kneighbors(X_min)
    synthetic = []
    for _ in range(n_synthetic):
        i   = np.random.randint(0, len(X_min))
        j   = indices[i, np.random.randint(1, 6)]
        lam = np.random.uniform(0, 1)
        synthetic.append(X_min[i] + lam * (X_min[j] - X_min[i]))
    return np.array(synthetic)

X_min_tr = X_tr[y_tr == 1]
X_maj_tr = X_tr[y_tr == 0]
n_synth   = len(X_maj_tr) - len(X_min_tr)
X_synth   = smote_simple(X_min_tr, n_synth)
X_tr_smote = np.vstack([X_tr, X_synth])
y_tr_smote = np.hstack([y_tr, np.ones(n_synth)])

clf_smote = LogisticRegression(max_iter=500, random_state=0).fit(
    X_tr_smote, y_tr_smote)
y_pred_smote = clf_smote.predict(X_te)
y_prob_smote = clf_smote.predict_proba(X_te)[:, 1]

# ── FIX 4: Random undersampling ───────────────────────────────────────────
n_min = (y_tr == 1).sum()
X_maj_down = resample(X_maj_tr, n_samples=n_min*3, random_state=0, replace=False)
X_tr_under = np.vstack([X_maj_down, X_min_tr])
y_tr_under = np.hstack([np.zeros(len(X_maj_down)), np.ones(len(X_min_tr))])

clf_under = LogisticRegression(max_iter=500, random_state=0).fit(
    X_tr_under, y_tr_under)
y_pred_under = clf_under.predict(X_te)
y_prob_under = clf_under.predict_proba(X_te)[:, 1]

# ── Naive "always predict negative" baseline ──────────────────────────────
y_pred_naive = np.zeros(len(y_te), dtype=int)
y_prob_naive = np.zeros(len(y_te))

# ── Print results table ───────────────────────────────────────────────────
strategies = [
    ("Always predict 0 (naive)", y_pred_naive, y_prob_naive + 1e-9),
    ("Baseline LR (thr=0.5)",    y_pred_base,  y_prob_base),
    ("Fix 1: Threshold tuning",  y_pred_thr,   y_prob_thr),
    ("Fix 2: Class weights",     y_pred_wt,    y_prob_wt),
    ("Fix 3: SMOTE oversample",  y_pred_smote, y_prob_smote),
    ("Fix 4: Undersample 3:1",   y_pred_under, y_prob_under),
]

print(f"  {'Strategy':28s} | {'Acc':>6} | {'F1':>6} | {'Recall':>7} | "
      f"{'Prec':>6} | {'MCC':>7} | {'PR-AUC':>7}")
print(f"  {'─'*82}")

all_metrics = []
for name, y_pred, y_prob in strategies:
    acc, f1, rec, prec, mcc, prauc = evaluate(name, y_te, y_pred, y_prob)
    all_metrics.append((name, acc, f1, rec, prec, mcc, prauc))
    flag = " <- ACCURACY PARADOX!" if name.startswith("Always") else ""
    print(f"  {name:28s} | {acc:6.3f} | {f1:6.3f} | {rec:7.3f} | "
          f"{prec:6.3f} | {mcc:7.3f} | {prauc:7.3f}{flag}")

print()
print("  OBSERVATIONS:")
print("  - 'Always predict 0' gets 98% accuracy but F1=0, MCC=0")
print("  - Baseline LR also nearly useless (Recall~0 means zero positives caught)")
print("  - Class weights and SMOTE dramatically improve F1 and MCC")
print("  - Use F1/MCC/PR-AUC — never accuracy alone — for imbalanced data")
print(f"  - Best threshold found on validation: {best_thr:.2f} (not 0.5)")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Class Imbalance: Why Accuracy Misleads and How Fixes Help",
             fontsize=12, fontweight="bold")

metric_names = ["Accuracy", "F1", "Recall", "Precision", "MCC"]
colours = ["tomato", "steelblue", "seagreen", "purple", "orange"]

x_pos = np.arange(len(strategies))
width = 0.35

accs  = [m[1] for m in all_metrics]
f1s   = [m[2] for m in all_metrics]
mccs  = [(m[5] + 1) / 2 for m in all_metrics]   # rescale MCC to [0,1] for plot

axes[0].bar(x_pos, accs, color="steelblue", alpha=0.8)
axes[0].set_title("Accuracy (misleading!)")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([m[0][:18] for m in all_metrics], rotation=30, ha="right", fontsize=7)
axes[0].axhline(0.98, color="red", linestyle="--", lw=1.5,
                label="Naive baseline: 98%")
axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3, axis="y")

axes[1].bar(x_pos, f1s, color="seagreen", alpha=0.8)
axes[1].set_title("F1 Score (true performance)")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([m[0][:18] for m in all_metrics], rotation=30, ha="right", fontsize=7)
axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3, axis="y")

axes[2].bar(x_pos, [m[5] for m in all_metrics], color="purple", alpha=0.8)
axes[2].set_title("MCC (best single metric)")
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels([m[0][:18] for m in all_metrics], rotation=30, ha="right", fontsize=7)
axes[2].axhline(0, color="gray", linestyle="--", lw=1, label="Random = 0")
axes[2].legend(fontsize=8); axes[2].set_ylim(-0.1, 1.0); axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("class_imbalance_strategies.png", dpi=120)
print()
print("  Plot saved -> class_imbalance_strategies.png")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Data Leakage — Three Types Demonstrated With Inflated Metrics": {
        "description": (
            "Demonstrate three leakage types with concrete examples: "
            "target leakage (feature contains the label), "
            "train-test contamination (scaling before split), "
            "and pipeline leakage (feature selection before split). "
            "Measure the performance inflation each type creates."
        ),
        "language": "python",
        "code": '''
import numpy as np
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)

np.random.seed(42)
print("=" * 65)
print("  DATA LEAKAGE: THREE TYPES DEMONSTRATED")
print("=" * 65)
print()

# ── Base dataset ──────────────────────────────────────────────────────────
X_base, y = make_classification(
    n_samples=1000, n_features=20, n_informative=5,
    n_redundant=5, random_state=0)

print("  Base task: 20 features, only 5 truly informative")
print("  Honest upper bound (oracle knows which 5 features are real):")
X_good = X_base[:, :5]   # pretend we know the 5 informative features
sc = StandardScaler()
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(X_good, y, test_size=0.3,
                                                     random_state=1)
sc.fit(X_tr_g)
clf = LogisticRegression(max_iter=500, random_state=0)
clf.fit(sc.transform(X_tr_g), y_tr_g)
oracle_acc = accuracy_score(y_te_g, clf.predict(sc.transform(X_te_g)))
print(f"  Oracle test accuracy: {oracle_acc:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# LEAKAGE TYPE 1: TARGET LEAKAGE
# ─────────────────────────────────────────────────────────────────────────
print("  " + "─"*60)
print("  LEAKAGE TYPE 1 — TARGET LEAKAGE")
print("  " + "─"*60)
print("  Scenario: fraud detection. We accidentally include a feature")
print("  that is a CONSEQUENCE of the label (fraud happened).")
print()

# Create a leaky feature: highly correlated with y but would not exist
# before the label is known (e.g., 'account frozen' = consequence of fraud)
leaky_feat = y + np.random.normal(0, 0.3, len(y))   # ~90% correlated with y

X_leaked = np.column_stack([X_base, leaky_feat])     # add leaky column
X_clean  = X_base.copy()                              # no leaky column

results = {}
for name, X_use in [("With target leakage", X_leaked),
                     ("Without leakage",     X_clean)]:
    X_tr, X_te, y_tr, y_te = train_test_split(X_use, y, test_size=0.3,
                                                random_state=1)
    sc_ = StandardScaler().fit(X_tr)
    clf_ = LogisticRegression(max_iter=500, random_state=0)
    clf_.fit(sc_.transform(X_tr), y_tr)
    acc = accuracy_score(y_te, clf_.predict(sc_.transform(X_te)))
    results[name] = acc

leak1_inflation = results["With target leakage"] - results["Without leakage"]
print(f"  Without leakage:     {results['Without leakage']:.4f}")
print(f"  With target leakage: {results['With target leakage']:.4f}")
print(f"  Inflation:           +{leak1_inflation:.4f} ({leak1_inflation*100:.1f} percentage points)")
print()
print("  The leaky model APPEARS to be much better, but in production")
print("  the 'account frozen' feature does not exist at prediction time")
print("  (the account gets frozen AFTER fraud is detected, not before).")
print()

# ─────────────────────────────────────────────────────────────────────────
# LEAKAGE TYPE 2: TRAIN-TEST CONTAMINATION (scaling before split)
# ─────────────────────────────────────────────────────────────────────────
print("  " + "─"*60)
print("  LEAKAGE TYPE 2 — TRAIN-TEST CONTAMINATION")
print("  " + "─"*60)
print("  Scaling the FULL dataset before the train/test split leaks")
print("  test set statistics (mean, std) into the training process.")
print()

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_base, y, test_size=0.3, random_state=1)

def run_cv_split_timing(X, y, scale_before_split, k=5):
    """Run CV with scaling at the wrong time (before split) or correct time."""
    if scale_before_split:
        # WRONG: scale full data, then split
        X_s = StandardScaler().fit_transform(X)
        scores = cross_val_score(
            LogisticRegression(max_iter=500, random_state=0),
            X_s, y, cv=k, scoring="accuracy")
    else:
        # CORRECT: Pipeline ensures scaler is fit inside each fold
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=0))
        ])
        scores = cross_val_score(pipe, X, y, cv=k, scoring="accuracy")
    return scores.mean(), scores.std()

acc_leaked, std_leaked  = run_cv_split_timing(X_base, y, scale_before_split=True)
acc_correct, std_correct = run_cv_split_timing(X_base, y, scale_before_split=False)
leak2_inflation = acc_leaked - acc_correct

print(f"  Scale before split (WRONG):  {acc_leaked:.4f} +/- {std_leaked:.4f}")
print(f"  Pipeline (CORRECT):          {acc_correct:.4f} +/- {std_correct:.4f}")
print(f"  Inflation:                   +{leak2_inflation:.4f}")
print()
print("  Note: for StandardScaler the effect is subtle (mean and std of")
print("  the test set do not change dramatically). With small datasets,")
print("  normalising based on strong outliers in the test set, or using")
print("  the test label distribution in encoding, creates larger gaps.")
print()

# ─────────────────────────────────────────────────────────────────────────
# LEAKAGE TYPE 3: FEATURE SELECTION BEFORE SPLIT
# ─────────────────────────────────────────────────────────────────────────
print("  " + "─"*60)
print("  LEAKAGE TYPE 3 — FEATURE SELECTION BEFORE SPLIT")
print("  " + "─"*60)
print("  Using test set labels to select features BEFORE train/test split.")
print()

k_features = 5   # select top-5 features

# WRONG: select features using ALL data (including test labels)
selector_all = SelectKBest(f_classif, k=k_features).fit(X_base, y)
X_selected_all = selector_all.transform(X_base)
scores_leaked = cross_val_score(
    LogisticRegression(max_iter=500, random_state=0),
    X_selected_all, y, cv=5, scoring="accuracy")

# CORRECT: feature selection inside Pipeline (fit on each train fold only)
pipe_correct = Pipeline([
    ("select", SelectKBest(f_classif, k=k_features)),
    ("sc",     StandardScaler()),
    ("clf",    LogisticRegression(max_iter=500, random_state=0)),
])
scores_correct = cross_val_score(pipe_correct, X_base, y, cv=5,
                                  scoring="accuracy")

leak3_inflation = scores_leaked.mean() - scores_correct.mean()

print(f"  Select ALL data then CV (WRONG):  {scores_leaked.mean():.4f} +/- {scores_leaked.std():.4f}")
print(f"  Pipeline select inside CV (RIGHT): {scores_correct.mean():.4f} +/- {scores_correct.std():.4f}")
print(f"  Inflation:                         +{leak3_inflation:.4f} ({leak3_inflation*100:.1f} pp)")
print()
print("  This leak is subtle but real: when selecting features based on")
print("  all data, the selector has 'seen' the correlation between each")
print("  feature and the test labels — information that should not exist.")
print("  With many noisy features the inflation can be large (overfitting")
print("  to noise that looks predictive because we peeked at test labels).")
print()

# ── Summary ───────────────────────────────────────────────────────────────
print("  LEAKAGE SUMMARY:")
print(f"  {'Leakage Type':32s} | {'Inflation':>10} | {'Risk Level':>12}")
print(f"  {'─'*60}")
for ltype, inflation in [
    ("Target leakage",              leak1_inflation),
    ("Train-test contamination",    leak2_inflation),
    ("Feature selection before CV", leak3_inflation),
]:
    risk = "HIGH" if inflation > 0.05 else ("MEDIUM" if inflation > 0.01 else "LOW")
    print(f"  {ltype:32s} | {inflation:+10.4f} | {risk:>12}")
print()
print("  PREVENTION: Always use sklearn Pipeline.")
print("  Pipeline ensures every preprocessing step is fitted ONLY on")
print("  the training fold — never on validation or test data.")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · Complete Data Quality Pipeline — All Steps in Correct Order": {
        "description": (
            "Build a complete, correct data quality pipeline using sklearn Pipeline. "
            "Handle missing data, scaling, encoding and class imbalance in one object. "
            "Compare correct pipeline approach against a naive leaky approach."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  COMPLETE DATA QUALITY PIPELINE — CORRECT ORDER")
print("=" * 65)
print()

# ── Simulate a messy dataset with 3 issues: missing, scaling, imbalance ───
n = 1200

# Generate base features
X_num1 = np.random.normal(45, 15, n)          # age: ~18-90
X_num2 = np.random.exponential(60000, n)       # salary: heavily skewed
X_num3 = np.random.normal(650, 80, n)          # credit score: ~300-850
X_cat  = np.random.choice(["A","B","C","D"], n, p=[0.4,0.3,0.2,0.1])

# Introduce missing values
X_num1_m = X_num1.copy()
X_num2_m = X_num2.copy()
X_num1_m[np.random.rand(n) < 0.10] = np.nan     # MCAR: 10% missing
X_num2_m[X_num2 > np.percentile(X_num2, 80)] = np.nan  # MNAR: high earners

# Target: imbalanced (10% positive)
prob = 0.4*(X_num2 < 30000) + 0.3*(X_num3 < 580) + np.random.randn(n)*0.3
y = (prob > np.percentile(prob, 90)).astype(int)

# Assemble as numpy array (columns: age, salary, credit, cat_encoded)
le = LabelEncoder().fit(X_cat)
X_cat_enc = le.transform(X_cat).reshape(-1, 1).astype(float)
X_raw = np.column_stack([X_num1_m, X_num2_m, X_num3, X_cat_enc])

print(f"  Dataset: {n} samples, 4 features")
print(f"  Positive rate: {y.mean()*100:.1f}%")
print(f"  Missing in age:    {np.isnan(X_raw[:,0]).mean()*100:.1f}%")
print(f"  Missing in salary: {np.isnan(X_raw[:,1]).mean()*100:.1f}%")
print()

# ── CORRECT approach: split first, then Pipeline ───────────────────────────
print("  STEP 1: Split BEFORE any preprocessing")
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.25,
                                            stratify=y, random_state=1)

# Pipeline ensures every transformer is fitted on training data only
correct_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # fills NaN with train median
    ("scale",  StandardScaler()),                   # centres using train mean/std
    ("clf",    LogisticRegression(max_iter=1000,
                                   class_weight="balanced",
                                   random_state=0)),
])

print("  STEP 2: Fit Pipeline on X_train ONLY")
correct_pipe.fit(X_tr, y_tr)

print("  STEP 3: Evaluate on X_test (pipeline uses train statistics)")
y_pred_correct = correct_pipe.predict(X_te)
f1_c  = f1_score(y_te, y_pred_correct, zero_division=0)
rec_c = recall_score(y_te, y_pred_correct, zero_division=0)
acc_c = accuracy_score(y_te, y_pred_correct)
print()

# ── NAIVE (WRONG) approach: preprocess on all data before split ────────────
print("  NAIVE APPROACH (WRONG): preprocess ALL data, then split")

# Fill NaN using global statistics (leaks test median to training)
# SI2 and SS2 already defined above

imp_naive = SI2(strategy="median").fit(X_raw)      # sees test data!
X_imp_all = imp_naive.transform(X_raw)
sc_naive  = SS2().fit(X_imp_all)                   # sees test data!
X_scl_all = sc_naive.transform(X_imp_all)

X_n_tr, X_n_te, yn_tr, yn_te = train_test_split(
    X_scl_all, y, test_size=0.25, stratify=y, random_state=1)

clf_naive = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=0).fit(X_n_tr, yn_tr)
y_pred_naive = clf_naive.predict(X_n_te)
f1_n  = f1_score(y_te, y_pred_naive, zero_division=0)
rec_n = recall_score(y_te, y_pred_naive, zero_division=0)
acc_n = accuracy_score(y_te, y_pred_naive)

# ── Print comparison ──────────────────────────────────────────────────────
print()
print("  COMPARISON:")
print(f"  {'Approach':30s} | {'F1':>8} | {'Recall':>8} | {'Accuracy':>10}")
print(f"  {'─'*62}")
print(f"  {'Correct Pipeline (no leakage)':30s} | {f1_c:8.4f} | {rec_c:8.4f} | {acc_c:10.4f}")
print(f"  {'Naive (leaks test statistics)':30s} | {f1_n:8.4f} | {rec_n:8.4f} | {acc_n:10.4f}")
print()

# ── Show the Pipeline protects against leakage ────────────────────────────
print("  WHAT PIPELINE.FIT() DOES INTERNALLY:")
print()
print("  correct_pipe.fit(X_train, y_train) calls:")
print()
# Show the imputer's learned values (from training only)
imp  = correct_pipe.named_steps["impute"]
scl  = correct_pipe.named_steps["scale"]
print(f"    1. SimpleImputer fits on X_train:")
print(f"       Train medians: age={imp.statistics_[0]:.2f}, "
      f"salary={imp.statistics_[1]:.0f}, "
      f"credit={imp.statistics_[2]:.0f}")
print(f"    2. StandardScaler fits on imputed X_train:")
print(f"       Train means:   age={scl.mean_[0]:.2f}, "
      f"salary={scl.mean_[1]:.0f}")
print(f"       Train stds:    age={scl.scale_[0]:.2f}, "
      f"salary={scl.scale_[1]:.0f}")
print()
print("  When predict(X_test) is called:")
print("    - Uses the SAME training medians for imputation")
print("    - Uses the SAME training mean/std for scaling")
print("    - Test data never influences these statistics ✓")
print()
print("  THE GOLDEN RULE:")
print("  1. train_test_split() FIRST — before ANY transformation")
print("  2. ALL preprocessing inside a Pipeline")
print("  3. Pipeline.fit() only ever sees X_train")
print("  4. Pipeline.predict(X_test) uses training-fit transformers")

# ── Visual summary ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Correct Data Quality Pipeline vs Naive Leaky Approach",
             fontsize=12, fontweight="bold")

# Plot 1: Missing data before/after imputation
axes[0].bar(["With NaN (raw)", "After Pipeline Impute"],
            [np.isnan(X_tr).mean()*100, 0.0],
            color=["tomato", "seagreen"], alpha=0.8)
axes[0].set_ylabel("% Missing values"); axes[0].set_ylim(0, 15)
axes[0].set_title("Missing Values: Eliminated by Pipeline")
axes[0].grid(alpha=0.3, axis="y")

# Plot 2: Feature scale before/after
X_tr_imp = imp.transform(X_tr)
X_tr_scl = scl.transform(X_tr_imp)
ax = axes[1]
bp1 = ax.boxplot([X_tr[:, 0][~np.isnan(X_tr[:, 0])],
                   X_tr[:, 1][~np.isnan(X_tr[:, 1])],
                   X_tr[:, 2],
                   X_tr[:, 3]],
                  positions=[1,2,3,4], widths=0.35,
                  patch_artist=True,
                  boxprops=dict(facecolor="tomato", alpha=0.6))
bp2 = ax.boxplot([X_tr_scl[:, 0], X_tr_scl[:, 1],
                   X_tr_scl[:, 2], X_tr_scl[:, 3]],
                  positions=[1.4,2.4,3.4,4.4], widths=0.35,
                  patch_artist=True,
                  boxprops=dict(facecolor="steelblue", alpha=0.6))
ax.set_xticks([1.2, 2.2, 3.2, 4.2])
ax.set_xticklabels(["age", "salary", "credit", "cat"], fontsize=9)
ax.set_title("Feature Scale: Before (red) vs After (blue)")
ax.set_ylabel("Value"); ax.set_ylim(-5, 5); ax.grid(alpha=0.3)

# Plot 3: Performance comparison
metrics = ["F1", "Recall", "Accuracy"]
vals_c  = [f1_c, rec_c, acc_c]
vals_n  = [f1_n, rec_n, acc_n]
x_pos = np.arange(3)
width = 0.35
axes[2].bar(x_pos - width/2, vals_c, width, label="Correct Pipeline",
            color="steelblue", alpha=0.8)
axes[2].bar(x_pos + width/2, vals_n, width, label="Naive (leaky)",
            color="tomato", alpha=0.8)
axes[2].set_xticks(x_pos); axes[2].set_xticklabels(metrics)
axes[2].set_ylim(0, 1.1); axes[2].set_ylabel("Score")
axes[2].set_title("Model Performance Comparison")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("data_quality_pipeline.png", dpi=120)
print()
print("  Pipeline diagram saved -> data_quality_pipeline.png")
''',
    },
    # ── 7 ─────────────────────────────────────────────────────────────────────
    "7 · Outlier Detection & Treatment — IQR, Z-score, Isolation Forest, LOF": {
        "description": (
            "Compare four outlier detection methods on a realistic dataset. "
            "Show how winsorizing, log-transform, and removal affect distributions "
            "and downstream model performance. Demonstrate that outliers can be "
            "signal, not noise."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)

np.random.seed(42)

print("=" * 65)
print("  OUTLIER DETECTION & TREATMENT")
print("=" * 65)
print()

# ── Generate dataset with three types of outliers ─────────────────────────
n = 800
X0 = np.random.normal(0, 1, n)
X1 = np.random.normal(50, 10, n)
X2 = np.random.exponential(2, n)
y  = ((X0 + 0.5*X2) > np.percentile(X0 + 0.5*X2, 75)).astype(int)

idx_err   = np.random.choice(n, 15, replace=False)
idx_fraud = np.random.choice(n, 8,  replace=False)
idx_multi = np.random.choice(n, 10, replace=False)

X0[idx_err]   = np.random.uniform(50, 100, 15)
X2[idx_fraud] = np.random.uniform(30, 60, 8)
y[idx_fraud]  = 1
X0[idx_multi] = np.random.normal(-3, 0.2, 10)
X1[idx_multi] = np.random.normal(90, 0.5, 10)

X = np.column_stack([X0, X1, X2])

print(f"  Dataset: {n} samples, 3 features")
print(f"  Injected: {len(idx_err)} sensor errors, {len(idx_fraud)} real fraud signals,")
print(f"            {len(idx_multi)} multivariate outliers")
print(f"  Positive rate: {y.mean()*100:.1f}%")
print()

# ── Method 1: IQR Fence ────────────────────────────────────────────────────
print("  METHOD 1 — IQR FENCE (Tukey, univariate on feature 0)")
print()
Q1, Q3 = np.percentile(X0, 25), np.percentile(X0, 75)
IQR    = Q3 - Q1
lo,  hi  = Q1 - 1.5*IQR, Q3 + 1.5*IQR
lo3, hi3 = Q1 - 3.0*IQR, Q3 + 3.0*IQR

mask_iqr_mild   = (X0 < lo)  | (X0 > hi)
mask_iqr_severe = (X0 < lo3) | (X0 > hi3)

print(f"  Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
print(f"  Severe fence: [{lo3:.3f}, {hi3:.3f}]")
print(f"  Severe outliers flagged:      {mask_iqr_severe.sum()}")
print(f"  Sensor errors captured:       {mask_iqr_severe[idx_err].sum()}/{len(idx_err)}")
print(f"  Fraud signals flagged (bad!): {mask_iqr_severe[idx_fraud].sum()}/{len(idx_fraud)}")
print()

# ── Method 2: Modified Z-score ─────────────────────────────────────────────
print("  METHOD 2 — MODIFIED Z-SCORE (robust, uses median/MAD)")
print()
med   = np.median(X0)
MAD   = np.median(np.abs(X0 - med))
z_mod = 0.6745 * (X0 - med) / (MAD + 1e-9)
mask_zmod = np.abs(z_mod) > 3.5

print(f"  Median={med:.3f}, MAD={MAD:.3f}")
print(f"  Flagged with |z_modified|>3.5: {mask_zmod.sum()} points")
print(f"  Sensor errors captured:        {mask_zmod[idx_err].sum()}/{len(idx_err)}")
print(f"  Fraud signals flagged (bad!):  {mask_zmod[idx_fraud].sum()}/{len(idx_fraud)}")
print()

# ── Method 3: Isolation Forest ─────────────────────────────────────────────
print("  METHOD 3 — ISOLATION FOREST (multivariate)")
print()
iso = IsolationForest(contamination=0.05, random_state=42)
pred_iso = iso.fit_predict(X)
mask_iso = pred_iso == -1

print(f"  Flags {mask_iso.sum()} points ({mask_iso.mean()*100:.1f}%)")
print(f"  Sensor errors captured:        {mask_iso[idx_err].sum()}/{len(idx_err)}")
print(f"  Multivariate outliers found:   {mask_iso[idx_multi].sum()}/{len(idx_multi)}")
print(f"  Fraud signals flagged (bad!):  {mask_iso[idx_fraud].sum()}/{len(idx_fraud)}")
print()

# ── Method 4: LOF ──────────────────────────────────────────────────────────
print("  METHOD 4 — LOCAL OUTLIER FACTOR")
print()
X_sc_lof = StandardScaler().fit_transform(X)
lof      = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
pred_lof = lof.fit_predict(X_sc_lof)
mask_lof = pred_lof == -1

print(f"  Flags {mask_lof.sum()} points")
print(f"  Sensor errors captured:        {mask_lof[idx_err].sum()}/{len(idx_err)}")
print(f"  Multivariate outliers found:   {mask_lof[idx_multi].sum()}/{len(idx_multi)}")
print(f"  Fraud signals flagged (bad!):  {mask_lof[idx_fraud].sum()}/{len(idx_fraud)}")
print()

# ── Treatment comparison ───────────────────────────────────────────────────
print("  TREATMENT COMPARISON (downstream model F1):")
print()
print(f"  {'Treatment':34s} | {'CV F1 (mean)':>13} | {'CV F1 (std)':>12} | {'n_train':>8}")
print(f"  {'─'*75}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                            stratify=y, random_state=1)

def eval_treatment(name, Xtr, ytr):
    scaler = StandardScaler().fit(Xtr)
    Xs     = scaler.transform(Xtr)
    clf    = LogisticRegression(class_weight="balanced", max_iter=500, random_state=0)
    scores = cross_val_score(clf, Xs, ytr, cv=5, scoring="f1")
    print(f"  {name:34s} | {scores.mean():13.4f} | {scores.std():12.4f} | {len(ytr):8d}")

eval_treatment("No treatment (baseline)", X_tr, y_tr)

Q1_tr = np.percentile(X_tr[:,0], 25); Q3_tr = np.percentile(X_tr[:,0], 75)
IQR_tr = Q3_tr - Q1_tr
keep = (X_tr[:,0] >= Q1_tr - 3*IQR_tr) & (X_tr[:,0] <= Q3_tr + 3*IQR_tr)
eval_treatment("Remove IQR severe outliers", X_tr[keep], y_tr[keep])

X_tr_w = X_tr.copy()
for col in range(3):
    lo_w = np.percentile(X_tr[:,col], 1)
    hi_w = np.percentile(X_tr[:,col], 99)
    X_tr_w[:,col] = np.clip(X_tr[:,col], lo_w, hi_w)
eval_treatment("Winsorize at 1st-99th pct", X_tr_w, y_tr)

X_tr_log = X_tr.copy()
X_tr_log[:,2] = np.log1p(np.abs(X_tr[:,2]))
eval_treatment("Log-transform skewed feature", X_tr_log, y_tr)

iso_tr = IsolationForest(contamination=0.05, random_state=42).fit(X_tr)
keep_iso = iso_tr.predict(X_tr) == 1
eval_treatment("Remove Isolation Forest outliers", X_tr[keep_iso], y_tr[keep_iso])

print()
print("  KEY LESSON: Winsorizing and log-transform usually beat removal.")
print("  Removing outliers risks discarding fraud signals (minority class).")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Outlier Detection: Methods & Treatment Comparison",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.scatter(range(n), np.sort(X0), s=4, c="steelblue", alpha=0.5, label="Normal")
sorted_idx  = np.argsort(X0)
err_sorted   = np.isin(sorted_idx, idx_err)
fraud_sorted = np.isin(sorted_idx, idx_fraud)
ax.scatter(np.where(err_sorted)[0],   np.sort(X0)[err_sorted],
           s=50, c="tomato",  zorder=5, label="Sensor error")
ax.scatter(np.where(fraud_sorted)[0], np.sort(X0)[fraud_sorted],
           s=50, c="seagreen", zorder=5, marker="^", label="Real fraud (keep!)")
ax.axhline(hi3, color="tomato", linestyle="--", lw=1.5, label=f"Severe fence")
ax.axhline(lo3, color="tomato", linestyle="--", lw=1.5)
ax.set_title("IQR Fence — Feature 0 (sorted)")
ax.set_xlabel("Rank"); ax.set_ylabel("Value")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax2 = axes[1]
colours_iso = np.where(mask_iso, "tomato", "steelblue")
ax2.scatter(X[:,0], X[:,2], c=colours_iso, s=8, alpha=0.5)
ax2.scatter(X[idx_err,0], X[idx_err,2], s=100, c="darkred",
            marker="x", linewidths=2, label="Sensor error", zorder=5)
ax2.scatter(X[idx_fraud,0], X[idx_fraud,2], s=100, c="seagreen",
            marker="^", zorder=5, label="Real fraud (keep!)")
ax2.set_title("Isolation Forest Flags\\n(red = flagged as outlier)")
ax2.set_xlabel("Feature 0"); ax2.set_ylabel("Feature 2")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3 = axes[2]
ax3.hist(X[:,2], bins=50, alpha=0.6, color="tomato", density=True, label="Raw (skewed)")
ax3.hist(np.log1p(X[:,2]), bins=50, alpha=0.6, color="steelblue",
         density=True, label="log1p transformed")
ax3.set_title("Log-Transform: Compresses Skew\\nWithout Discarding Data")
ax3.set_xlabel("Value"); ax3.set_ylabel("Density")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outlier_detection.png", dpi=120)
print()
print("  Plot saved -> outlier_detection.png")
''',
    },

    # ── 8 ─────────────────────────────────────────────────────────────────────
    "8 · Categorical Encoding — OHE vs Target vs Frequency & Leakage Trap": {
        "description": (
            "Compare label, one-hot, frequency, and target encoding on a "
            "high-cardinality categorical feature. Demonstrate the target "
            "encoding leakage trap: naive encoding vs correct cross-fold encoding. "
            "Show how cardinality affects memory and model performance."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  CATEGORICAL ENCODING: STRATEGIES & LEAKAGE TRAPS")
print("=" * 65)
print()

# ── Dataset with high-cardinality city feature ────────────────────────────
n, n_cities = 2000, 80
city_ids    = np.random.randint(0, n_cities, n)
city_names  = [f"City_{i:02d}" for i in range(n_cities)]
city_base   = np.random.beta(2, 5, n_cities)

X_num = np.random.normal(0, 1, n)
true_probs = np.clip(0.6*city_base[city_ids] + 0.4*(X_num > 0.5), 0.02, 0.98)
y      = (np.random.rand(n) < true_probs).astype(int)
cities = np.array([city_names[i] for i in city_ids])

print(f"  Dataset: {n} samples, 1 numeric + 1 categorical ({n_cities} cities)")
print(f"  Positive rate: {y.mean()*100:.1f}%")
print()

# ── Show the ordinal fallacy ───────────────────────────────────────────────
print("  DANGER: LABEL ENCODING IMPLIES FALSE ORDINAL RELATIONSHIP")
print()
le = LabelEncoder().fit(cities)
print(f"  {'City':10s} | {'Label code':>11} | {'Mean(target)':>13}")
print(f"  {'─'*40}")
for c in sorted(city_names[:8]):
    code     = le.transform([c])[0]
    mean_y   = y[cities == c].mean()
    print(f"  {c:10s} | {code:11d} | {mean_y:13.4f}")
print("  No correlation between code and target — ordering is meaningless!")
print()

# ── Compare encoding strategies ───────────────────────────────────────────
print("  ENCODING STRATEGY COMPARISON:")
print()
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    np.column_stack([cities, X_num.astype(str)]),
    y, test_size=0.3, random_state=1)

cities_tr = X_tr_raw[:, 0]
cities_te = X_te_raw[:, 0]
num_tr    = X_tr_raw[:, 1].astype(float)
num_te    = X_te_raw[:, 1].astype(float)
global_mean = y_tr.mean()
m_smooth    = 20

def train_eval(name, Xtr, Xte, ytr, yte):
    sc  = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return accuracy_score(yte, pred), f1_score(yte, pred, zero_division=0), Xtr.shape[1]

results = []

# Numeric only
acc, f1, nc = train_eval("Numeric only (baseline)",
    num_tr.reshape(-1,1), num_te.reshape(-1,1), y_tr, y_te)
results.append(("Numeric only (baseline)", acc, f1, nc))

# Label encoding
le2 = LabelEncoder().fit(cities_tr)
safe_te = np.array([c if c in le2.classes_ else le2.classes_[0] for c in cities_te])
acc, f1, nc = train_eval("Label encoding (WRONG ordinal)",
    np.column_stack([le2.transform(cities_tr), num_tr]),
    np.column_stack([le2.transform(safe_te), num_te]), y_tr, y_te)
results.append(("Label encoding (WRONG ordinal)", acc, f1, nc))

# One-hot
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cities_tr.reshape(-1,1))
acc, f1, nc = train_eval("One-hot encoding",
    np.column_stack([ohe.transform(cities_tr.reshape(-1,1)), num_tr]),
    np.column_stack([ohe.transform(cities_te.reshape(-1,1)), num_te]), y_tr, y_te)
results.append(("One-hot encoding", acc, f1, nc))

# Frequency
freq_map  = {c: (cities_tr == c).mean() for c in np.unique(cities_tr)}
def freq_enc(cats): return np.array([freq_map.get(c, 1/n_cities) for c in cats])
acc, f1, nc = train_eval("Frequency encoding",
    np.column_stack([freq_enc(cities_tr), num_tr]),
    np.column_stack([freq_enc(cities_te), num_te]), y_tr, y_te)
results.append(("Frequency encoding", acc, f1, nc))

# Naive target encoding (LEAKY)
naive_map = {c: y[cities == c].mean() for c in np.unique(cities)}
def naive_enc(cats): return np.array([naive_map.get(c, global_mean) for c in cats])
acc, f1, nc = train_eval("Target enc. NAIVE (LEAKY)",
    np.column_stack([naive_enc(cities_tr), num_tr]),
    np.column_stack([naive_enc(cities_te), num_te]), y_tr, y_te)
results.append(("Target enc. NAIVE (LEAKY)", acc, f1, nc))

# Correct OOF target encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cats_tr_enc = np.zeros(len(cities_tr))
for fold_tr_idx, fold_val_idx in kf.split(cities_tr):
    for c in np.unique(cities_tr[fold_val_idx]):
        fm = cities_tr[fold_tr_idx] == c
        cnt = fm.sum()
        cm  = y_tr[fold_tr_idx][fm].mean() if cnt > 0 else global_mean
        sm  = (cnt*cm + m_smooth*global_mean) / (cnt + m_smooth)
        cats_tr_enc[fold_val_idx[cities_tr[fold_val_idx] == c]] = sm

correct_map = {}
for c in np.unique(cities_tr):
    fm = cities_tr == c; cnt = fm.sum()
    correct_map[c] = (cnt*y_tr[fm].mean() + m_smooth*global_mean)/(cnt+m_smooth)

def correct_enc_te(cats): return np.array([correct_map.get(c, global_mean) for c in cats])
acc, f1, nc = train_eval("Target enc. CORRECT (OOF)",
    np.column_stack([cats_tr_enc, num_tr]),
    np.column_stack([correct_enc_te(cities_te), num_te]), y_tr, y_te)
results.append(("Target enc. CORRECT (OOF)", acc, f1, nc))

print(f"  {'Method':35s} | {'Accuracy':>10} | {'F1':>8} | {'n_cols':>8}")
print(f"  {'─'*68}")
for name, acc, f1, nc in results:
    flag = " <- LEAKY" if "LEAKY" in name else (" <- WRONG" if "WRONG" in name else "")
    print(f"  {name:35s} | {acc:10.4f} | {f1:8.4f} | {nc:8d}{flag}")

print()
print("  Naive target encoding scores highest but is LEAKY (invalid).")
print("  Correct OOF target encoding: nearly as good, no leakage.")
print("  One-hot: n_cols explodes with cardinality (80 cities = 80 cols).")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Categorical Encoding: Strategies, Cardinality & Leakage",
             fontsize=12, fontweight="bold")

cards = [5, 10, 20, 50, 100, 200, 500]
axes[0].plot(cards, cards, "tomato", lw=2.5, label="One-hot (k cols)")
axes[0].plot(cards, [1]*len(cards), "steelblue", lw=2.5, label="Label/Freq/Target (1 col)")
axes[0].plot(cards, [min(32, k) for k in cards], "seagreen", lw=2.5,
             linestyle="--", label="Hashing (<=32 cols)")
axes[0].set_xlabel("Cardinality (# unique values)")
axes[0].set_ylabel("Columns created")
axes[0].set_title("Memory Cost vs Cardinality")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

f1_vals = [r[2] for r in results]
cols_enc = ["gray","tomato","steelblue","seagreen","red","purple"]
brs = axes[1].barh(range(len(results)), f1_vals, color=cols_enc, alpha=0.8)
axes[1].set_yticks(range(len(results)))
axes[1].set_yticklabels([r[0] for r in results], fontsize=8)
axes[1].set_xlabel("F1 Score")
axes[1].set_title("F1 Score by Encoding Method")
axes[1].grid(alpha=0.3, axis="x")
for bar, val in zip(brs, f1_vals):
    axes[1].text(val+0.002, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)

naive_vals   = [naive_map.get(c, global_mean) for c in cities_te]
correct_vals = [correct_map.get(c, global_mean) for c in cities_te]
axes[2].scatter(correct_vals, naive_vals, alpha=0.3, s=10, c="steelblue")
axes[2].plot([0,1],[0,1],"tomato",linestyle="--",lw=2,label="y=x (identical)")
axes[2].set_xlabel("Correct OOF encoding"); axes[2].set_ylabel("Naive encoding (LEAKY)")
axes[2].set_title("Naive vs Correct Target Encoding\\n(Naive is optimistically biased)")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("categorical_encoding.png", dpi=120)
print()
print("  Plot saved -> categorical_encoding.png")
''',
    },

    # ── 9 ─────────────────────────────────────────────────────────────────────
    "9 · Distribution Shift & Data Drift — PSI, KS Test & Concept Drift": {
        "description": (
            "Simulate all three types of drift: covariate, label, and concept drift. "
            "Detect covariate shift using PSI and KS test per feature. "
            "Show how a model silently degrades under each shift type. "
            "Demonstrate importance weighting as a fix for covariate shift."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)

np.random.seed(42)

print("=" * 65)
print("  DISTRIBUTION SHIFT & DATA DRIFT")
print("=" * 65)
print()

def psi(expected, actual, n_bins=10):
    bps = np.linspace(min(expected.min(), actual.min()),
                      max(expected.max(), actual.max()), n_bins + 1)
    exp_p = np.histogram(expected, bins=bps)[0].astype(float) + 1e-6
    act_p = np.histogram(actual,   bins=bps)[0].astype(float) + 1e-6
    exp_p /= exp_p.sum(); act_p /= act_p.sum()
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))

# ── Training data ──────────────────────────────────────────────────────────
n_train = 2000
X_tr = np.column_stack([
    np.random.normal(0, 1, n_train),
    np.random.normal(0, 1, n_train),
    np.random.normal(0, 1, n_train),
])
feat_names = ["Age", "Income", "Credit"]

def sigmoid(z): return 1 / (1 + np.exp(-z))
def true_prob(X): return sigmoid(0.8*X[:,0] + 0.5*X[:,1] - 0.3*X[:,2])
def true_prob_drift(X): return sigmoid(0.8*X[:,0] - 0.9*X[:,1] - 0.3*X[:,2])

y_tr = (np.random.rand(n_train) < true_prob(X_tr)).astype(int)
scaler = StandardScaler().fit(X_tr)
clf    = LogisticRegression(max_iter=500, random_state=0)
clf.fit(scaler.transform(X_tr), y_tr)

n_test = 1000
print(f"  Training: n={n_train}, positive rate={y_tr.mean()*100:.1f}%")
print()

# ── Four scenarios ─────────────────────────────────────────────────────────
def gen(age_mu=0, age_s=1, income_mu=0, income_s=1, credit_mu=0, credit_s=1,
        prob_fn=None, label_scale=1.0):
    X = np.column_stack([np.random.normal(age_mu, age_s, n_test),
                         np.random.normal(income_mu, income_s, n_test),
                         np.random.normal(credit_mu, credit_s, n_test)])
    fn = prob_fn if prob_fn else true_prob
    p  = np.clip(fn(X) * label_scale, 0.01, 0.99)
    y  = (np.random.rand(n_test) < p).astype(int)
    return X, y

scenarios = [
    ("No shift (baseline)",     gen()),
    ("Covariate shift (age)",   gen(age_mu=1.5, age_s=0.8)),
    ("Label shift (3x fraud)",  gen(prob_fn=lambda X: np.clip(true_prob(X)*3, 0, 1))),
    ("Concept drift",           gen(prob_fn=true_prob_drift)),
]

# ── Performance table ─────────────────────────────────────────────────────
print("  MODEL PERFORMANCE UNDER DIFFERENT SHIFT SCENARIOS:")
print()
print(f"  {'Scenario':30s} | {'AUC':>7} | {'F1':>7} | {'Pos rate':>10} | {'Degradation'}")
print(f"  {'─'*72}")

baseline_auc = None
auc_vals_plot = []
for name, (X_sc, y_sc) in scenarios:
    probs = clf.predict_proba(scaler.transform(X_sc))[:,1]
    preds = (probs > 0.5).astype(int)
    auc   = roc_auc_score(y_sc, probs) if y_sc.sum() > 0 and y_sc.sum() < len(y_sc) else 0.5
    f1    = f1_score(y_sc, preds, zero_division=0)
    auc_vals_plot.append(auc)
    if baseline_auc is None:
        baseline_auc = auc; degrade = "baseline"
    else:
        degrade = f"-{(baseline_auc-auc)*100:.1f} pts AUC"
    print(f"  {name:30s} | {auc:7.4f} | {f1:7.4f} | {y_sc.mean()*100:9.1f}% | {degrade}")

print()

# ── PSI + KS detection ────────────────────────────────────────────────────
print("  DRIFT DETECTION: PSI AND KS TEST")
print()
print("  PSI < 0.10: ok  |  0.10-0.20: investigate  |  > 0.20: retrain")
print()
print(f"  {'Scenario':26s} | {'Feature':8s} | {'PSI':>7} | {'KS p-val':>10} | {'Alarm?'}")
print(f"  {'─'*68}")

detect_scenarios = [
    ("Covariate (age shifts)",  scenarios[1][1][0]),
    ("Label shift",             scenarios[2][1][0]),
    ("Concept drift",           scenarios[3][1][0]),
]
for sname, X_new in detect_scenarios:
    for fi, fname in enumerate(feat_names):
        psi_val    = psi(X_tr[:,fi], X_new[:,fi])
        _, ks_p    = stats.ks_2samp(X_tr[:,fi], X_new[:,fi])
        alarm = "ALARM" if psi_val > 0.2 or ks_p < 0.01 else ("WARN" if psi_val > 0.1 else "ok")
        print(f"  {sname:26s} | {fname:8s} | {psi_val:7.4f} | {ks_p:10.4f} | {alarm}")
    print()

# ── Importance weighting fix ──────────────────────────────────────────────
print("  FIX: IMPORTANCE WEIGHTING FOR COVARIATE SHIFT")
print()
X_cov, y_cov = scenarios[1][1]
X_domain = np.vstack([X_tr, X_cov])
y_domain = np.array([0]*n_train + [1]*n_test)
scaler_d = StandardScaler().fit(X_domain)
clf_d    = LogisticRegression(max_iter=500, random_state=0)
clf_d.fit(scaler_d.transform(X_domain), y_domain)

tr_p  = clf_d.predict_proba(scaler_d.transform(X_tr))[:,1]
w     = np.clip((tr_p / (1 - tr_p + 1e-8)) / ((tr_p / (1 - tr_p + 1e-8)).mean()), 0.1, 10)

clf_iw = LogisticRegression(max_iter=500, random_state=0)
clf_iw.fit(scaler.transform(X_tr), y_tr, sample_weight=w)

auc_orig = roc_auc_score(y_cov, clf.predict_proba(scaler.transform(X_cov))[:,1])
auc_iw   = roc_auc_score(y_cov, clf_iw.predict_proba(scaler.transform(X_cov))[:,1])
print(f"  Covariate shift — Original AUC: {auc_orig:.4f}")
print(f"  Covariate shift — IW AUC:       {auc_iw:.4f}  ({(auc_iw-auc_orig)*100:+.2f} pts)")
print()

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribution Shift: Detection, Impact & Importance Weighting",
             fontsize=12, fontweight="bold")

axes[0].hist(X_tr[:,0], bins=40, alpha=0.5, color="steelblue",
             density=True, label="Training (age)")
axes[0].hist(X_cov[:,0], bins=40, alpha=0.5, color="tomato",
             density=True, label="Deployment (shifted)")
psi_age = psi(X_tr[:,0], X_cov[:,0])
axes[0].set_title(f"Covariate Shift: Age Distribution\\nPSI={psi_age:.3f} (>0.20 = major)")
axes[0].set_xlabel("Age (standardised)"); axes[0].set_ylabel("Density")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

snames = ["No shift","Covariate\\nshift","Label\\nshift","Concept\\ndrift"]
cols_d = ["seagreen","orange","tomato","red"]
bars = axes[1].bar(snames, auc_vals_plot, color=cols_d, alpha=0.8)
axes[1].axhline(0.5, color="gray", linestyle="--", lw=1.5, label="Random")
axes[1].set_ylim(0.4, 1.0); axes[1].set_ylabel("AUC")
axes[1].set_title("Model AUC Under Different Shift Types")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")
for bar, val in zip(bars, auc_vals_plot):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
                 ha="center", fontsize=9)

axes[2].hist(w, bins=40, color="purple", alpha=0.8)
axes[2].axvline(1.0, color="tomato", linestyle="--", lw=2, label="w=1 (no reweight)")
axes[2].set_xlabel("Importance weight w(x)")
axes[2].set_ylabel("Count")
axes[2].set_title(f"Importance Weights for Covariate Shift\\nAUC {auc_orig:.3f} -> {auc_iw:.3f}")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("distribution_shift.png", dpi=120)
print("  Plot saved -> distribution_shift.png")
print()
print("  KEY TAKEAWAYS:")
print("  PSI > 0.20 on any feature: investigate / retrain")
print("  Covariate shift: importance weighting can partially fix it")
print("  Label shift: recalibrate decision threshold")
print("  Concept drift: MUST retrain — no weighting can fix it")
''',
    },

    # ── 10 ────────────────────────────────────────────────────────────────────
    "10 · Fix Ordering & Interactions — SMOTE, Target Encoding & Threshold Bugs": {
        "description": (
            "Demonstrate the three most dangerous ordering mistakes: "
            "SMOTE before split, naive target encoding, and class weights "
            "without threshold re-tuning. Quantify the score inflation each "
            "produces and show the correct pipeline for each."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _opt
from scipy import stats as _scipy_stats

# ── Logistic Regression ───────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, max_iter=500, random_state=None, class_weight=None, C=1.0):
        self.max_iter=max_iter; self.random_state=random_state
        self.class_weight=class_weight; self.C=C
    def _sw(self, y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self, X, y, sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5/self.C*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(), X.T@e/n+w[1:]/self.C]
            return loss,g
        res=_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)
    def inverse_transform(self,X): return X*self.scale_+self.mean_

class MinMaxScaler:
    def fit(self,X):
        self.min_=X.min(axis=0); self.scale_=(X.max(axis=0)-self.min_)+1e-8; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X):
        self.center_=np.median(X,axis=0)
        q75,q25=np.percentile(X,[75,25],axis=0); self.scale_=(q75-q25)+1e-8; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Imputers ──────────────────────────────────────────────────────────────────
class SimpleImputer:
    def __init__(self,strategy="mean"): self.strategy=strategy
    def fit(self,X):
        if self.strategy=="mean": self.statistics_=np.nanmean(X,axis=0)
        elif self.strategy=="median": self.statistics_=np.nanmedian(X,axis=0)
        else: self.statistics_=np.nanmean(X,axis=0)
        return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(X.shape[1]):
            m=np.isnan(Xc[:,i]); Xc[m,i]=self.statistics_[i]
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)
# aliases used in op6
SI2=SimpleImputer; SS2=StandardScaler

class BayesianRidge:
    pass  # placeholder; IterativeImputer ignores estimator arg in this impl

class KNNImputer:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._Xf=X.copy(); return self
    def transform(self,X):
        Xc=X.copy()
        for i in range(len(X)):
            nan_c=np.where(np.isnan(X[i]))[0]
            if not len(nan_c): continue
            obs_c=np.where(~np.isnan(X[i]))[0]
            vr=np.where(~np.any(np.isnan(self._Xf[:,obs_c]),axis=1))[0] if len(obs_c) else np.array([])
            for c in nan_c:
                if not len(vr): Xc[i,c]=np.nanmean(self._Xf[:,c]); continue
                d=np.sqrt(np.sum((self._Xf[vr][:,obs_c]-X[i,obs_c])**2,axis=1))
                nn=vr[np.argsort(d)[:self.n_neighbors]]
                v=self._Xf[nn,c]; v=v[~np.isnan(v)]
                Xc[i,c]=np.mean(v) if len(v) else np.nanmean(self._Xf[:,c])
        return Xc
    def fit_transform(self,X): return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self,estimator=None,max_iter=5,random_state=None): self.max_iter=max_iter
    def fit_transform(self,X):
        Xc=X.copy()
        for j in range(Xc.shape[1]):
            m=np.isnan(Xc[:,j]); Xc[m,j]=np.nanmean(Xc[:,j])
        nm=np.isnan(X)
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                mr=np.where(nm[:,j])[0]
                if not len(mr): continue
                or_=np.where(~nm[:,j])[0]; oc=[c for c in range(X.shape[1]) if c!=j]
                try:
                    A=np.column_stack([np.ones(len(or_)),Xc[np.ix_(or_,oc)]])
                    b=Xc[or_,j]; w,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                    Xc[mr,j]=np.column_stack([np.ones(len(mr)),Xc[np.ix_(mr,oc)]])@w
                except: pass
        return Xc
    def transform(self,X): return self.fit_transform(X)

# ── Data generation ───────────────────────────────────────────────────────────
def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold ─────────────────────────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs]; tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs:]])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (yt==yp).mean()
def _tpfpfntn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=2*tp+fp+fn
    return (2*tp/d) if d>0 else zero_division
def recall_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fn)) if (tp+fn)>0 else zero_division
def precision_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return (tp/(tp+fp)) if (tp+fp)>0 else zero_division
def matthews_corrcoef(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/d if d>0 else 0.0
def roc_auc_score(yt,ys):
    if yt.sum()==0 or yt.sum()==len(yt): return 0.5
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum(); neg=len(ys2)-pos
    tpr=np.concatenate([[0],np.cumsum(ys2)/pos])
    fpr=np.concatenate([[0],np.cumsum(1-ys2)/neg])
    return float(np.trapezoid(tpr,fpr))
def average_precision_score(yt,ys):
    o=np.argsort(-ys); ys2=yt[o]; pos=ys2.sum()
    if pos==0: return 0.0
    tp=np.cumsum(ys2); prec=tp/np.arange(1,len(ys2)+1)
    return np.sum(prec*ys2)/pos
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tpfpfntn(yt,yp); return np.array([[tn,fp],[fn,tp]])

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(y); fs=n//cv; scores=[]
    for i in range(cv):
        te=np.arange(i*fs,(i+1)*fs if i<cv-1 else n)
        tr=np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs if i<cv-1 else n,n)])
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        preds=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            preds.append(np.bincount(nn.astype(int)).argmax())
        return np.array(preds)

# ── NearestNeighbors (for SMOTE) ──────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]
            D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── resample ──────────────────────────────────────────────────────────────────
def resample(*arrays,n_samples=None,random_state=None,replace=True):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if n_samples is None: n_samples=n
    idx=rng.choice(n,size=n_samples,replace=replace)
    return arrays[0][idx] if len(arrays)==1 else [a[idx] for a in arrays]

# ── Feature selection ─────────────────────────────────────────────────────────
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]
    Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_scipy_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        sc,_=self.score_func(X,y); self.selected_=np.argsort(sc)[::-1][:self.k]; return self
    def transform(self,X): return X[:,self.selected_]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X))
        return self.fit(X,y).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for _,step in self.steps[:-1]:
            if hasattr(step,"fit_transform"):
                try: Xt=step.fit_transform(Xt,y)
                except TypeError: Xt=step.fit_transform(Xt)
            else: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()
    def fit_transform(self,X,y=None): self.fit(X,y); return self.transform(X)

# ── Encoders ──────────────────────────────────────────────────────────────────
class LabelEncoder:
    def fit(self,y):
        self.classes_=np.unique(y); self._m={c:i for i,c in enumerate(self.classes_)}; return self
    def transform(self,y): return np.array([self._m.get(yi,0) for yi in y])
    def fit_transform(self,y): return self.fit(y).transform(y)
    def inverse_transform(self,y): return self.classes_[y]

class OneHotEncoder:
    def __init__(self,sparse_output=False,handle_unknown="ignore"): pass
    def fit(self,X): self.cats_=[np.unique(X[:,i]) for i in range(X.shape[1])]; return self
    def transform(self,X):
        cols=[]
        for i,cats in enumerate(self.cats_):
            for c in cats: cols.append((X[:,i]==c).astype(float))
        return np.column_stack(cols)
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── IsolationForest ───────────────────────────────────────────────────────────
class IsolationForest:
    def __init__(self,contamination=0.1,random_state=None,n_estimators=50):
        self.contamination=contamination; self.n_estimators=n_estimators
        self._rng=np.random.default_rng(random_state)
    @staticmethod
    def _c(n): return 2*(np.log(max(n-1,1))+0.5772156649)-2*(n-1)/(n+1e-9) if n>1 else 0.
    def fit(self,X): self._X=X.copy(); self._n,self._p=X.shape; return self
    def _tdepths(self,X,Xs,d=0,mx=8):
        n,ns=len(X),len(Xs); res=np.full(n,float(d)+self._c(ns))
        if ns<=1 or d>=mx: return res
        col=int(self._rng.integers(0,self._p))
        lo,hi=Xs[:,col].min(),Xs[:,col].max()
        if lo>=hi: return res
        sp=float(self._rng.uniform(lo,hi))
        lx=X[:,col]<sp; lxs=Xs[:,col]<sp; rx=~lx; rxs=~lxs
        if lxs.any() and lx.any(): res[lx]=self._tdepths(X[lx],Xs[lxs],d+1,mx)
        if rxs.any() and rx.any(): res[rx]=self._tdepths(X[rx],Xs[rxs],d+1,mx)
        return res
    def score_samples(self,X):
        sub=min(256,self._n); cn=self._c(sub); deps=np.zeros(len(X))
        for _ in range(self.n_estimators):
            idx=self._rng.choice(self._n,sub,replace=False)
            deps+=self._tdepths(X,self._X[idx])
        deps/=self.n_estimators
        return -2.**(- deps/(cn if cn else 1.))
    def fit_predict(self,X): self.fit(X); return self.predict(X)
    def predict(self,X):
        s=self.score_samples(X); t=np.percentile(s,100*self.contamination)
        return np.where(s<=t,-1,1)

# ── LocalOutlierFactor ────────────────────────────────────────────────────────
class LocalOutlierFactor:
    def __init__(self,n_neighbors=20,contamination=0.05):
        self.n_neighbors=n_neighbors; self.contamination=contamination
    def fit_predict(self,X):
        n=len(X); k=min(self.n_neighbors,n-1)
        D=np.sqrt(((X[:,None]-X[None])**2).sum(axis=2))
        kd=np.sort(D,axis=1)[:,k]; nn=np.argsort(D,axis=1)[:,1:k+1]
        rd=np.maximum(D,kd[None,:]); lrd=np.zeros(n)
        for i in range(n): lrd[i]=k/np.sum(rd[i,nn[i]])
        lof=np.array([np.mean(lrd[nn[i]])/( lrd[i]+1e-10) for i in range(n)])
        thr=np.percentile(lof,100*(1-self.contamination))
        return np.where(lof>thr,-1,1)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  FIX ORDERING & INTERACTIONS: THREE PIPELINE BUGS")
print("=" * 65)
print()

n = 3000
X_num, y = make_classification(
    n_samples=n, n_features=8, n_informative=5,
    weights=[0.92, 0.08], random_state=42)

n_cats   = 60
cat_ids  = np.random.randint(0, n_cats, n)
cat_str  = np.array([f"C{i:03d}" for i in cat_ids])
pos_rate = y.mean()

print(f"  Dataset: {n} samples, imbalance {(1-pos_rate)/pos_rate:.0f}:1")
print(f"  Positive rate: {pos_rate*100:.1f}%")
print()

# ── BUG 1: SMOTE BEFORE SPLIT ─────────────────────────────────────────────
print("  BUG 1 — SMOTE BEFORE SPLIT")
print()

try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)

    # WRONG: resample ALL data, then split
    X_res_w, y_res_w = sm.fit_resample(X_num, y)
    Xtr_w, Xte_w, ytr_w, yte_w = train_test_split(
        X_res_w, y_res_w, test_size=0.3, random_state=1)
    sc_w = StandardScaler().fit(Xtr_w)
    clf_w = LogisticRegression(max_iter=500, random_state=0)
    clf_w.fit(sc_w.transform(Xtr_w), ytr_w)
    p_w = clf_w.predict_proba(sc_w.transform(Xte_w))[:,1]
    f1_wrong  = f1_score(yte_w, (p_w > 0.5).astype(int), zero_division=0)
    auc_wrong = roc_auc_score(yte_w, p_w)

    # CORRECT: split first, then SMOTE only on training set
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        X_num, y, test_size=0.3, stratify=y, random_state=1)
    Xtr_res, ytr_res = sm.fit_resample(Xtr_c, ytr_c)
    sc_c = StandardScaler().fit(Xtr_res)
    clf_c = LogisticRegression(max_iter=500, random_state=0)
    clf_c.fit(sc_c.transform(Xtr_res), ytr_res)
    p_c = clf_c.predict_proba(sc_c.transform(Xte_c))[:,1]
    f1_correct  = f1_score(yte_c, (p_c > 0.5).astype(int), zero_division=0)
    auc_correct = roc_auc_score(yte_c, p_c)
    smote_inflation = (f1_wrong - f1_correct) * 100

    print(f"  {'Approach':38s} | {'F1':>8} | {'AUC':>8} | Note")
    print(f"  {'─'*72}")
    print(f"  {'WRONG: SMOTE before split':38s} | {f1_wrong:8.4f} | {auc_wrong:8.4f} | "
          "test contaminated")
    print(f"  {'CORRECT: split first, SMOTE train only':38s} | {f1_correct:8.4f} | "
          f"{auc_correct:8.4f} | test untouched")
    print(f"  Score inflation: {smote_inflation:+.2f} F1 pts")
    smote_available = True
except ImportError:
    print("  (imbalanced-learn not installed)")
    print("  Rule: ALWAYS split first, then apply SMOTE to train fold only.")
    f1_wrong = f1_correct = smote_inflation = 0.0
    smote_available = False
print()

# ── BUG 2: TARGET ENCODING WITHOUT OOF PROTECTION ─────────────────────────
print("  BUG 2 — NAIVE TARGET ENCODING (LEAKS TEST LABELS)")
print()

Xtr2, Xte2, ytr2, yte2 = train_test_split(
    X_num, y, test_size=0.3, stratify=y, random_state=1)
ctr2 = cat_str[:len(Xtr2)]
cte2 = cat_str[len(Xtr2):len(Xtr2)+len(Xte2)]
gm2  = ytr2.mean(); ms2 = 20

# WRONG: use full dataset y to compute means
naive_map2 = {c: y[cat_str == c].mean() for c in np.unique(cat_str)}

# CORRECT: out-of-fold
kf2 = KFold(n_splits=5, shuffle=True, random_state=0)
enc_tr2 = np.zeros(len(ctr2))
for fi, vi in kf2.split(ctr2):
    for c in np.unique(ctr2[vi]):
        fm = ctr2[fi] == c; cnt = fm.sum()
        cm = ytr2[fi][fm].mean() if cnt > 0 else gm2
        enc_tr2[vi[ctr2[vi] == c]] = (cnt*cm + ms2*gm2)/(cnt+ms2)

correct_map2 = {}
for c in np.unique(ctr2):
    fm = ctr2 == c; cnt = fm.sum()
    correct_map2[c] = (cnt*ytr2[fm].mean() + ms2*gm2)/(cnt+ms2)

def te(cats, mp, gm): return np.array([mp.get(c, gm) for c in cats])

def eval_te(Xtr_enc, Xte_enc, ytr, yte):
    sc  = StandardScaler().fit(Xtr_enc)
    clf = LogisticRegression(max_iter=500, random_state=0, class_weight="balanced")
    clf.fit(sc.transform(Xtr_enc), ytr)
    p   = clf.predict_proba(sc.transform(Xte_enc))[:,1]
    return f1_score(yte, (p > 0.5).astype(int), zero_division=0), roc_auc_score(yte, p)

Xtr_naive2 = np.column_stack([te(ctr2, naive_map2, gm2), Xtr2])
Xte_naive2 = np.column_stack([te(cte2, naive_map2, gm2), Xte2])
f1_naive2, auc_naive2 = eval_te(Xtr_naive2, Xte_naive2, ytr2, yte2)

Xtr_corr2 = np.column_stack([enc_tr2, Xtr2])
Xte_corr2 = np.column_stack([te(cte2, correct_map2, gm2), Xte2])
f1_corr2, auc_corr2 = eval_te(Xtr_corr2, Xte_corr2, ytr2, yte2)
te_inflation = (f1_naive2 - f1_corr2) * 100

print(f"  {'Approach':40s} | {'F1':>8} | {'AUC':>8}")
print(f"  {'─'*60}")
print(f"  {'WRONG: target enc on full data (LEAKY)':40s} | {f1_naive2:8.4f} | {auc_naive2:8.4f}")
print(f"  {'CORRECT: OOF target encoding':40s} | {f1_corr2:8.4f} | {auc_corr2:8.4f}")
print(f"  Score inflation: {te_inflation:+.2f} F1 pts")
print()

# ── BUG 3: CLASS WEIGHTS WITHOUT THRESHOLD RETUNING ───────────────────────
print("  BUG 3 — CLASS WEIGHTS SHIFT CALIBRATION: MUST RETUNE THRESHOLD")
print()

Xtr3, Xte3, ytr3, yte3 = train_test_split(
    X_num, y, test_size=0.3, stratify=y, random_state=1)
Xval3, Xte3b = Xte3[:len(Xte3)//2], Xte3[len(Xte3)//2:]
yval3, yte3b = yte3[:len(yte3)//2], yte3[len(yte3)//2:]

sc3  = StandardScaler().fit(Xtr3)
clf3 = LogisticRegression(max_iter=500, random_state=0, class_weight="balanced")
clf3.fit(sc3.transform(Xtr3), ytr3)

val_probs3 = clf3.predict_proba(sc3.transform(Xval3))[:,1]
te_probs3  = clf3.predict_proba(sc3.transform(Xte3b))[:,1]

f1_def = f1_score(yte3b, (te_probs3 > 0.5).astype(int), zero_division=0)
thrs   = np.linspace(0.01, 0.99, 200)
val_f1s = [f1_score(yval3, (val_probs3 > t).astype(int), zero_division=0) for t in thrs]
best_thr = thrs[np.argmax(val_f1s)]
f1_tuned = f1_score(yte3b, (te_probs3 > best_thr).astype(int), zero_division=0)
approx_thr = 1 / (1 + (1 - pos_rate)/pos_rate)

print(f"  Imbalance ratio: {(1-pos_rate)/pos_rate:.0f}:1  "
      f"=> approx optimal threshold ~ {approx_thr:.3f}")
print(f"  Best threshold found on validation: {best_thr:.3f}")
print()
print(f"  {'Approach':35s} | {'F1':>8} | {'Threshold':>10}")
print(f"  {'─'*58}")
print(f"  {'Default threshold=0.5':35s} | {f1_def:8.4f} | {'0.500':>10}")
print(f"  {'Tuned on validation set':35s} | {f1_tuned:8.4f} | {best_thr:10.3f}")
print(f"  Gain from threshold tuning: {(f1_tuned - f1_def)*100:+.2f} F1 pts")
print()
print("  RULE: After class_weight='balanced', ALWAYS tune threshold.")
print("  Never assume 0.5 is optimal for imbalanced problems.")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Three Common Pipeline Bugs and Their Score Inflation",
             fontsize=12, fontweight="bold")

if smote_available:
    axes[0].barh(["SMOTE before split\\n(WRONG)","Split first, SMOTE train\\n(CORRECT)"],
                 [f1_wrong, f1_correct], color=["tomato","seagreen"], alpha=0.8)
    axes[0].set_xlabel("F1 Score")
    axes[0].set_title(f"Bug 1: SMOTE Before Split\\nInflation = {smote_inflation:+.1f} F1 pts")
    for i, val in enumerate([f1_wrong, f1_correct]):
        axes[0].text(val+0.002, i, f"{val:.4f}", va="center", fontsize=10)
else:
    axes[0].text(0.5, 0.5, "imbalanced-learn\\nnot installed\\n\\nRule: split FIRST\\nthen SMOTE train only",
                 ha="center", va="center", transform=axes[0].transAxes, fontsize=11)
    axes[0].set_title("Bug 1: SMOTE Before Split")
axes[0].grid(alpha=0.3, axis="x")

axes[1].barh(["Naive target enc\\n(LEAKY)","OOF target enc\\n(CORRECT)"],
             [f1_naive2, f1_corr2], color=["tomato","seagreen"], alpha=0.8)
axes[1].set_xlabel("F1 Score")
axes[1].set_title(f"Bug 2: Target Encoding Leakage\\nInflation = {te_inflation:+.1f} F1 pts")
axes[1].grid(alpha=0.3, axis="x")
for i, val in enumerate([f1_naive2, f1_corr2]):
    axes[1].text(val+0.002, i, f"{val:.4f}", va="center", fontsize=10)

axes[2].plot(thrs, val_f1s, "steelblue", lw=2.5)
axes[2].axvline(0.5, color="tomato", linestyle="--", lw=2, label="Default=0.5")
axes[2].axvline(best_thr, color="seagreen", linestyle="--", lw=2,
                label=f"Optimal={best_thr:.3f}")
axes[2].axvline(approx_thr, color="purple", linestyle=":", lw=1.5,
                label=f"Formula={approx_thr:.3f}")
axes[2].set_xlabel("Decision threshold"); axes[2].set_ylabel("F1 on validation")
axes[2].set_title("Bug 3: Class Weights Shift Calibration\\nThreshold Must Be Re-tuned")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pipeline_ordering.png", dpi=120)
print()
print("  Plot saved -> pipeline_ordering.png")
print()
print("  CORRECT ORDER: split -> indicators -> impute -> encode")
print("                 -> winsorize -> scale -> SMOTE (train only)")
print("                 -> fit model -> tune threshold on val set")
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


# """
# Data Quality: Missing Data, Scaling, Imbalance & Leakage
# ==========================================================
#
# Four data problems that silently destroy model performance.
# Each one can inflate reported metrics, produce models that fail in
# production, or make an otherwise excellent algorithm useless.
# Understanding how each damages training is as important as the fix.
#
# """
#
# import textwrap
# import re
#
# TOPIC_NAME   = "Data Quality: Missing Data, Scaling, Imbalance & Leakage"
# DISPLAY_NAME = "01 · Data Quality"
# ICON         = "🧹"
# SUBTITLE     = "The Four Silent Killers of Model Performance"
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # THEORY
# # ─────────────────────────────────────────────────────────────────────────────
#
# THEORY = """
#
# ### PART 1 — MISSING DATA
#
# ### The Three Mechanisms of Missingness
#
# The TYPE of missingness determines what you can and cannot safely do.
# This is not cosmetic — using the wrong imputation for the wrong type
# produces biased models that look fine on validation but fail in deployment.
#
# **MCAR — Missing Completely At Random**
#     P(missing | X, Y) = constant
#     The probability of a value being missing has NO relationship to any
#     variable, observed or unobserved.
#
#     Real example: a sensor randomly fails with probability 0.05
#     regardless of the reading it would have produced.
#
#     Effect on model: if you drop MCAR rows, your dataset is smaller
#     but still an unbiased sample. Simple imputation (mean/median) is safe.
#
#     Test: Little's MCAR test. Compare missingness pattern to chance.
#
#
# **MAR — Missing At Random**
#     P(missing | X_observed) — depends only on observed variables
#     The probability of a value being missing depends on OTHER observed
#     features, but NOT on the missing value itself.
#
#     Real example: older patients are less likely to report income.
#     Income is missing, but we can predict missingness from age (observed).
#
#     Effect on model: dropping rows creates systematic bias. You lose
#     all older patients and the model never learns their patterns.
#     Safe fix: impute using the relationship with observed variables
#     (e.g., KNN imputation, MICE).
#
#
# **MNAR — Missing Not At Random**
#     P(missing | X_unobserved) — depends on the missing value itself
#     The value is missing BECAUSE of what it would be.
#
#     Real example: very high earners decline to report income.
#     Very sick patients miss follow-up appointments (because they are sick).
#     People with extreme weights avoid being weighed.
#
#     Effect on model: this is the most dangerous type. ANY imputation
#     creates bias because the missingness IS informative. The fact that
#     a value is missing is itself a signal.
#
#     Fix: add a binary indicator variable (is_missing) as a separate
#     feature. This lets the model learn that "missing income" means
#     "likely very high income" without pretending to know the value.
#
#     Diagram 1 — Three Missingness Mechanisms:
#
#     MCAR                  MAR                    MNAR
#     ─────────────────     ─────────────────      ─────────────────
#     Age  Income           Age  Income             Age   Income
#     25   50k              25   50k                25    50k
#     30   ???  ← random    65   ???  ← age>60      ???   ???  ← VERY high
#     45   80k              30   70k                30    70k
#     55   ???  ← random    70   ???  ← age>60      ???   ???  ← VERY high
#     Missing ⊥ (Age,Inc)  Missing ⊥ Inc | Age     Missing ∼ Inc itself
#
#
# ### How Missing Data Harms Each Model Class
#
#     ┌────────────────────────┬────────────────────────────────────────────┐
#     │ Model                  │ How missingness hurts it                   │
#     ├────────────────────────┼────────────────────────────────────────────┤
#     │ Linear / Logistic Reg  │ NaN propagates through matrix multiply     │
#     │                        │ → entire prediction becomes NaN            │
#     ├────────────────────────┼────────────────────────────────────────────┤
#     │ Neural Networks        │ Same NaN propagation; gradient undefined   │
#     │                        │ for missing inputs                         │
#     ├────────────────────────┼────────────────────────────────────────────┤
#     │ Decision Trees (plain) │ Cannot split on NaN; most implementations  │
#     │                        │ error or drop the row entirely             │
#     ├────────────────────────┼────────────────────────────────────────────┤
#     │ XGBoost / LightGBM     │ NATIVE missingness support: learns which   │
#     │                        │ branch to take for NaN at each split.      │
#     │                        │ Missingness becomes an implicit feature.   │
#     ├────────────────────────┼────────────────────────────────────────────┤
#     │ KNN                    │ Distance undefined when a feature is NaN.  │
#     │                        │ Must impute before distance computation.   │
#     ├────────────────────────┼────────────────────────────────────────────┤
#     │ SVMs                   │ Kernel functions undefined with NaN.       │
#     │                        │ Must impute before fitting.                │
#     └────────────────────────┴────────────────────────────────────────────┘
#
#
# ### Imputation Strategies
#
# **Mean / Median / Mode imputation** (univariate)
#     Replace each missing value with the column mean (continuous),
#     median (skewed/outliers), or mode (categorical).
#
#     Pros: fast, simple, no data leakage risk.
#     Cons: reduces variance artificially, destroys correlations between
#     features, assumes MCAR. With 30%+ missingness, estimates become
#     unreliable.
#
#     Rule: use median (not mean) when the feature is skewed, to avoid
#     the missing values being pulled toward outliers.
#
# **KNN Imputation**
#     Find the k nearest neighbours (using only observed features) and
#     impute the missing value as the average of those neighbours' values.
#
#     Pros: captures feature correlations, appropriate for MAR.
#     Cons: O(n²) computation, sensitive to feature scale (must scale first),
#     k is a hyperparameter.
#
# **MICE / Iterative Imputation**
#     Multiple Imputation by Chained Equations. For each feature with
#     missing values:
#         1. Impute missing values with a simple placeholder (e.g., mean)
#         2. Regress that feature on all other features
#         3. Impute missing values with regression predictions
#         4. Repeat for all features → one "pass"
#         5. Repeat all passes until convergence
#
#     Pros: captures complex inter-feature relationships, appropriate for
#     MAR, returns uncertainty estimates (multiple imputation).
#     Cons: computationally expensive, many hyperparameters.
#
# **Missing Indicator (binary flag)**
#     Add a new boolean column: is_missing_feature_X = 1 if X is NaN.
#     Then impute X as usual.
#
#     Why: if the data is MNAR, the fact of missingness is informative.
#     Letting the model see the flag allows it to learn "missing income
#     → likely high earner" without corrupting the imputed value.
#     Low cost; almost always worth adding for MNAR variables.
#
#     When to drop rows: only when < 1-2% of rows are affected AND
#     the missingness is MCAR. Otherwise impute.
#
#     Critical leakage rule: fit the imputer on the TRAINING set only.
#     Then transform BOTH train and test. Never fit on combined data.
#
#
#
# ### PART 2 — FEATURE SCALING
#
# ### Why Scaling Matters
#
# Feature scaling is one of the most overlooked preprocessing steps.
# It has NO effect on tree-based models but is CRITICAL for everything else.
#
# **Effect 1 — Gradient Descent Convergence**
#
#     Unscaled features create an elongated loss surface. Gradients
#     point diagonally rather than toward the minimum, causing oscillation.
#
#     Diagram 2 — Loss Surface Shape With vs Without Scaling:
#
#     WITHOUT scaling                WITH scaling
#     (features on different scales) (features normalised)
#
#     w₂ │                            w₂ │
#        │   ╭────────╮                  │      ╭──╮
#        │ ╭──────────────╮              │    ╭──────╮
#        │╭────────────────────╮         │   ╭────────╮
#        │────────────────────────       │    ╰──────╯
#        └──────────────────── w₁        │      ╰──╯
#                                        └──────────── w₁
#     Long narrow ellipses              Circular contours
#     Gradient points sideways          Gradient points to minimum
#     Oscillation, slow convergence     Fast convergence
#
#     With unscaled features:
#     - Feature "age" (range 18-90) has much smaller gradients than
#       feature "salary" (range 20k-200k).
#     - The learning rate that works for salary overshoots for age.
#     - You need a tiny lr → very slow convergence.
#
# **Effect 2 — Distance-Based Algorithms**
#
#     KNN, K-means, SVMs with RBF kernel all compute distances.
#     A feature with range [0, 10,000] dominates a feature with
#     range [0, 1] even if the smaller-range feature is more informative.
#
#     Without scaling: KNN decides "nearest neighbour" purely based on
#     salary, completely ignoring education level.
#
# **Effect 3 — Regularisation**
#
#     L1 and L2 penalties apply equally to all weights. If salary is on
#     a 100,000× larger scale than age, its coefficient will be 100,000×
#     smaller in the raw data — so regularisation penalises it 100,000×
#     less. The penalty is not fair across features without scaling.
#
#
# ### The Three Standard Scalers
#
# **StandardScaler (z-score normalisation)**
#
#     z = (x - μ) / σ
#
#     Transforms each feature to have mean=0 and std=1.
#
#     Pros: works well when features are approximately Gaussian.
#     Handles negative values. Most common default.
#     Cons: sensitive to outliers (an outlier shifts μ and inflates σ,
#     compressing all other values). Not bounded to [0,1].
#
#     Use when: linear/logistic regression, neural networks, SVM, PCA.
#
# **MinMaxScaler**
#
#     x_scaled = (x - x_min) / (x_max - x_min)    → range [0, 1]
#     Or to [a, b]: a + (x_scaled) * (b - a)
#
#     Pros: bounded output (useful when algorithm needs [0,1] range,
#     e.g., sigmoid output, image pixels). Preserves zero.
#     Cons: extremely sensitive to outliers. A single outlier compresses
#     all other values into a tiny range near 0.
#
#     Use when: image data (pixels already in known range), algorithms
#     requiring strictly positive inputs.
#
# **RobustScaler**
#
#     x_scaled = (x - median) / IQR      IQR = Q3 - Q1
#
#     Uses median and interquartile range instead of mean and std.
#     Outliers have much less influence since they don't affect the median
#     or IQR significantly.
#
#     Pros: robust to outliers. Good for real-world tabular data which
#     commonly contains extreme values.
#     Cons: output is not bounded, not zero-mean in general.
#
#     Use when: features have heavy tails or outliers (salaries, prices,
#     medical measurements, fraud amounts).
#
#     Diagram 3 — Effect of a Single Outlier on Each Scaler:
#
#     Data: [1, 2, 3, 4, 5, 100]   ← 100 is an outlier
#
#     StandardScaler:  all "normal" values → compressed near 0
#                      [-0.72, -0.67, -0.62, -0.56, -0.51,  3.08]
#                       ↑ everything squashed into [-0.72, -0.51]
#
#     MinMaxScaler:    [-0.00,  0.01,  0.02,  0.03,  0.04,  1.00]
#                       ↑ everything squashed into [0.00, 0.04]!
#
#     RobustScaler:   [-1.25, -0.75, -0.25,  0.25,  0.75, 48.25]
#                       ↑ normal values spread across [-1.25, 0.75] ✓
#
#
# ### The Golden Rule: Fit on Train, Transform Both
#
#     WRONG (data leakage):
#         scaler.fit(X_all)          ← uses test data statistics!
#         X_train_s = scaler.transform(X_train)
#         X_test_s  = scaler.transform(X_test)
#
#     CORRECT:
#         scaler.fit(X_train)        ← only training statistics
#         X_train_s = scaler.transform(X_train)
#         X_test_s  = scaler.transform(X_test)
#
#     EVEN BETTER — use a Pipeline:
#         pipe = Pipeline([("scaler", StandardScaler()),
#                          ("model",  LogisticRegression())])
#         pipe.fit(X_train, y_train)      ← scaler fit on train only
#         pipe.predict(X_test)            ← scaler applied to test ✓
#
#     Fitting the scaler on all data leaks test set statistics (mean, std,
#     min, max) into the training process — a subtle but real form of leakage.
#
#     When NOT to scale: Decision Trees, Random Forests, Gradient
#     Boosting (XGBoost, LightGBM). These algorithms split on thresholds
#     — scaling changes the values but not the ORDER, so splits are
#     unchanged. Scaling adds no benefit and wastes time.
#
#
#
# ### PART 3 — CLASS IMBALANCE
#
#
# ### The Accuracy Paradox
#
# A dataset with 99% negative examples and 1% positive examples.
# A classifier that ALWAYS predicts negative achieves 99% accuracy.
# It catches ZERO actual positives. Accuracy is completely useless here.
#
#     Diagram 4 — The Accuracy Paradox:
#
#     100 examples: 99 negative, 1 positive
#
#     "Predict everything negative" classifier:
#     ┌─────────────────────────────────────────────────────────────┐
#     │  Accuracy  = 99/100 = 99%    ← sounds great!                │
#     │  Recall    = 0/1    = 0%     ← catches ZERO positives       │
#     │  Precision = undefined (0/0)                                │
#     │  F1        = 0%              ← correctly shows uselessness  │
#     └─────────────────────────────────────────────────────────────┘
#
#     ALWAYS use F1, PR-AUC, MCC, or ROC-AUC for imbalanced problems.
#     NEVER report only accuracy.
#
#
# ### How Imbalance Hurts Model Training
#
# **1. Decision boundary bias**
#
#     The model learns that predicting the majority class is almost always
#     "correct" under cross-entropy loss. The loss landscape is dominated
#     by majority examples. The decision boundary shifts away from the
#     minority class toward "predict majority."
#
#     For a 99:1 dataset, the model's gradient is 99% driven by negative
#     examples. The 1% positive examples have almost no influence on
#     which direction the parameters move.
#
# **2. Under-representation in batches**
#
#     In mini-batch SGD with batch size 64 and 1% positive rate, a typical
#     batch contains ~0.64 positive examples. Many batches see ZERO
#     positives — those batches produce gradients that push the model
#     exclusively toward the majority class.
#
# **3. Threshold default**
#
#     Most classifiers use 0.5 as the decision threshold. This is only
#     appropriate when classes are balanced. For a 99:1 dataset, the
#     optimal threshold may be as low as 0.1 — the model should only
#     need moderate confidence to predict positive.
#
#
# ### Fix 1 — Threshold Adjustment
#
# The cheapest fix. After training, sweep the threshold and pick the value
# that maximises the metric you care about (F1, recall, etc.).
# Use the validation set, never the test set, to pick the threshold.
#
#
# ### Fix 2 — Class Weights (Cost-Sensitive Learning)
#
# Assign higher loss to minority class misclassifications:
#
#     Loss = Σᵢ wᵢ · L(yᵢ, ŷᵢ)
#
#     where wᵢ = N / (n_classes × n_samples_in_class_i)
#
#     For 99:1 imbalance:
#         w_negative = 100 / (2 × 99) ≈ 0.51
#         w_positive = 100 / (2 ×  1) = 50.0
#
#     Each positive misclassification now counts 100× more than a
#     negative one. The model is forced to care about the minority class.
#
#     sklearn: LogisticRegression(class_weight='balanced')
#              RandomForestClassifier(class_weight='balanced')
#              XGBoost: scale_pos_weight = n_negative / n_positive
#
#
# ### Fix 3 — Oversampling
#
# Generate more minority class examples.
#
#     Random oversampling: duplicate minority examples randomly.
#     Simple but causes overfitting — the model memorises the duplicates.
#
#     SMOTE: generate SYNTHETIC minority examples by interpolating
#     between real minority examples in feature space.
#     (Covered in Module 05 — Data Augmentation.)
#
#     ADASYN: like SMOTE but focuses synthesis on harder-to-classify
#     minority examples near the decision boundary.
#
#
# ### Fix 4 — Undersampling
#
# Remove majority class examples to balance the dataset.
#
#     Random undersampling: drop majority examples randomly.
#     Risk: discard useful information from the majority class.
#
#     Tomek links: remove majority examples that are the "nearest
#     neighbour" of a minority example. Cleans the decision boundary
#     without discarding as much information.
#
#     NearMiss: keep majority examples that are CLOSEST to minority
#     examples — preserves the most challenging majority examples.
#
#
# ### Fix 5 — Algorithmic Approaches
#
#     Focal Loss (covered in Module 09):
#         Down-weights easy examples (model is already confident on them)
#         and focuses the gradient on hard, minority examples.
#
#     BalancedBaggingClassifier:
#         Trains each base estimator on a BALANCED bootstrap sample
#         (equal numbers from each class). Combines bagging's variance
#         reduction with undersampling's balance.
#
#     Which fix to choose:
#     ┌─────────────────────────────────────────────────────────────────┐
#     │ Imbalance ratio │ Recommended approach                          │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ 2:1  to  5:1    │ Class weights — simple and effective          │
#     │ 5:1  to  20:1   │ Class weights + threshold tuning              │
#     │ 20:1 to  100:1  │ SMOTE + class weights                         │
#     │ > 100:1         │ Focal loss + SMOTE + collect more data        │
#     └─────────────────────────────────────────────────────────────────┘
#
#     Matthews Correlation Coefficient (MCC) is the best single metric
#     for severe imbalance:
#         MCC = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
#     Range: −1 (inverse prediction) to 0 (random) to +1 (perfect).
#     Unlike F1, MCC accounts for all four cells of the confusion matrix.
#
#
#
# ### PART 4 — DATA LEAKAGE
#
# ### What Is Data Leakage?
#
# Data leakage occurs when information that would not be available at
# prediction time is used during model training. This creates a model
# that performs suspiciously well in evaluation but fails in production.
#
# The core test: "Would I have access to this feature when making a
# real prediction?" If no — it is a leaky feature.
#
# Leakage is insidious because it is INVISIBLE in training and validation
# metrics. The model looks great — then falls apart in production.
#
#
# ### Type 1 — Target Leakage
#
# A feature directly encodes or strongly correlates with the label,
# but would not exist before the label is known.
#
#     Example: Predicting hospital readmission within 30 days.
#     Including "discharge medication count" as a feature. Doctors
#     prescribe more medications to sicker patients — patients who
#     are likely to be readmitted. The feature is a CONSEQUENCE of
#     the target, not a cause.
#
#     Example: Predicting loan default. Including "debt collection
#     calls received" as a feature. Collection calls happen AFTER
#     default — you cannot know this before the loan is issued.
#
#     Detection: suspiciously high feature importance for a variable
#     that seems too directly related to the target. CV scores that
#     are unrealistically good (>99% on a hard problem).
#
#
# ### Type 2 — Train-Test Contamination
#
# Preprocessing steps that use statistics from the test set during
# training. The most common form of accidental leakage.
#
#     WRONG:
#         scaler.fit(X_all)           ← test mean/std leaks into training
#         imputer.fit(X_all)          ← test missingness pattern leaks
#         pca.fit(X_all)              ← test variance structure leaks
#         selected = select_k_best(X_all, y_all)  ← test labels leak!
#
#     These steps must be fitted on the TRAINING set only, then
#     applied to both. Use sklearn Pipelines to enforce this.
#
#
# ### Type 3 — Temporal Leakage
#
# Time-ordered data where future information contaminates past predictions.
#
#     Example: Stock price prediction. Using features computed from
#     the next day's trading volume (unavailable at prediction time).
#
#     Example: Fraud detection. A customer's total fraud history,
#     including future fraud events, used to predict current transactions.
#
#     Fix: use strict temporal splits. Train on t < cutoff.
#     Test on t > cutoff. NEVER shuffle time-series data.
#     Use rolling-window cross-validation, not k-fold.
#
#
# ### Type 4 — Duplicate Leakage
#
# The same (or near-identical) examples appear in both train and test sets.
#
#     Common cause: data collected at different times, then merged,
#     producing duplicate rows before splitting.
#
#     Example: patient records from the same hospital stay appearing
#     as multiple rows. A split on rows (not patients) puts the same
#     patient in both train and test.
#
#     Fix: split on the ENTITY (patient ID, user ID, loan ID), not
#     on individual rows. Group k-fold cross-validation.
#
#
# ### Type 5 — Pipeline Leakage (Feature Selection)
#
# Using the full dataset (including test labels) to select features
# before the train/test split.
#
#     WRONG:
#         important_features = select_by_correlation(X_all, y_all)
#         X_selected = X_all[:, important_features]
#         X_train, X_test = train_test_split(X_selected)   # too late!
#
#     CORRECT:
#         X_train, X_test = train_test_split(X_all)
#         selector.fit(X_train, y_train)   # only train labels
#         X_train_s = selector.transform(X_train)
#         X_test_s  = selector.transform(X_test)
#
#     BEST — use a Pipeline:
#         pipe = Pipeline([
#             ("select", SelectKBest(k=10)),
#             ("scale",  StandardScaler()),
#             ("model",  LogisticRegression()),
#         ])
#         cross_val_score(pipe, X_train, y_train, cv=5)
#         # Each fold: selector fit on 4/5 of data, transforms 1/5 ✓
#
#
# ### Leakage Detection Checklist
#
#     ┌─────────────────────────────────────────────────────────────────┐
#     │  Red flag                      │ Possible leakage type          │
#     ├─────────────────────────────────────────────────────────────────┤
#     │  CV score seems too good       │ Any type                       │
#     │  A feature has >50% importance │ Target leakage                 │
#     │  Performance drops sharply     │                                │
#     │    in production               │ Temporal or target leakage     │
#     │  Feature name looks like the   │                                │
#     │    target (e.g., "outcome_")   │ Target leakage                 │
#     │  Preprocessing before split    │ Train-test contamination       │
#     │  Time-ordered data shuffled    │ Temporal leakage               │
#     │  Multiple rows per entity      │ Duplicate leakage              │
#     └─────────────────────────────────────────────────────────────────┘
#
#
#
# ### PART 5 — OUTLIER DETECTION & TREATMENT
#
# ### Outlier ≠ Error
#
# The most important principle: an outlier is an extreme value, not
# necessarily a wrong one. Before removing any point you must ask:
# "Is this an error, or is this the most important signal in my dataset?"
#
#     A fraudulent transaction of $50,000 when the typical transaction
#     is $40 is an outlier — and exactly what the fraud model must catch.
#     Removing it destroys the signal you were hired to find.
#
#     A temperature reading of 9,999°C from a broken sensor IS an error.
#     These require completely different treatments.
#
# Classification of outliers:
#
#     Point outlier:   a single value far from the bulk of the distribution
#     Contextual:      normal globally but anomalous in context
#                      (30°C is normal in July, anomalous in January)
#     Collective:      a group of values that together are anomalous even
#                      though each individual value looks normal
#
#
# ### Detection Methods
#
# **Univariate — Z-score**
#
#     z = (x - μ) / σ    flag if |z| > 3
#
#     Pros: simple and interpretable.
#     Cons: assumes Gaussian distribution. Sensitive to the very outliers
#     it tries to detect (the outlier inflates μ and σ, masking itself).
#     Use modified Z-score with median/MAD instead for robustness:
#
#         z_modified = 0.6745 × (x - median) / MAD
#         MAD = median(|xᵢ - median(x)|)
#         Flag if |z_modified| > 3.5
#
# **Univariate — IQR Fence (Tukey)**
#
#     Q1, Q3 = 25th and 75th percentile
#     IQR    = Q3 − Q1
#     Lower fence: Q1 − 1.5 × IQR
#     Upper fence: Q3 + 1.5 × IQR     (standard boxplot whiskers)
#     Severe:      Q1 − 3.0 × IQR  to  Q3 + 3.0 × IQR
#
#     Pros: non-parametric, robust, matches boxplot intuition.
#     Cons: univariate only — misses multivariate outliers where no single
#     feature is extreme but the combination is impossible.
#
#     Diagram 5 — IQR Fence:
#
#     ────Q1─────────────Q3────
#        |←── IQR ──→|
#     ←1.5×IQR         1.5×IQR→
#     [lower fence]  [upper fence]
#     *  ← outlier       outlier → *
#
# **Multivariate — Isolation Forest**
#
#     Randomly partition the feature space with axis-aligned cuts.
#     Anomalies are isolated by fewer cuts on average (they are in sparse
#     regions, easier to separate from the rest).
#
#     Anomaly score = average path length to isolation (shorter = more anomalous)
#     contamination: expected fraction of outliers (hyperparameter)
#
#     Pros: works in high dimensions, no distributional assumption,
#     scales to large datasets, handles mixed outlier types.
#     Cons: random — results vary between runs without a seed.
#
# **Multivariate — Local Outlier Factor (LOF)**
#
#     Compares a point's local density to its neighbours' local density.
#     A point in a sparse neighbourhood surrounded by dense clusters is
#     flagged as an outlier — even if it is not globally extreme.
#
#     LOF ≈ 1:   normal (similar density to neighbours)
#     LOF >> 1:  outlier (much lower density than neighbours)
#
#     Pros: captures local structure, detects contextual outliers.
#     Cons: O(n²) distance computation, requires scaling first.
#
#
# ### Treatment Strategies
#
#     ┌─────────────────────────────────────────────────────────────────┐
#     │ Treatment         │ When to use                                 │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Remove the row    │ ONLY if confirmed measurement / entry error │
#     │                   │ AND < 1% of data. Never remove real signal. │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Winsorize / Cap   │ Cap at a percentile (e.g., 1st–99th).       │
#     │                   │ Keeps the row but limits outlier influence. │
#     │                   │ Safe default for continuous features.       │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Log / sqrt        │ Right-skewed features (salaries, prices,    │
#     │ transform         │ counts). Compresses the tail without        │
#     │                   │ discarding data. log1p(x) handles zeros.    │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Robust scaler     │ Let RobustScaler reduce the outlier's       │
#     │                   │ influence implicitly via IQR normalisation. │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Flag + keep       │ Add is_extreme_X = 1 and let the model      │
#     │                   │ learn from both the value and the flag.     │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Separate model    │ If outliers form a coherent group (e.g.,    │
#     │                   │ VIP customers), train a dedicated model.    │
#     └─────────────────────────────────────────────────────────────────┘
#
#     The pipeline rule: outlier treatment must be fitted on train only
#     (the 99th percentile used for winsorizing must come from training).
#     Use a custom sklearn Transformer inside a Pipeline.
#
#
# ### PART 6 — CATEGORICAL ENCODING
#
# ### Why Encoding Is a Data Quality Problem
#
# Most ML algorithms require numeric inputs. Converting categories to
# numbers is not cosmetic — the WRONG encoding silently introduces either
# false ordinal relationships or target leakage.
#
#
# ### Encoding Methods
#
# **Label Encoding (ordinal encoding)**
#
#     Maps each category to an integer: Red→0, Blue→1, Green→2.
#
#     DANGER: implies Red < Blue < Green — an ordering that does not
#     exist. The model will use this as a numeric feature and learn
#     that Green (2) is "twice" Blue (1).
#
#     When to use: ONLY for genuinely ordinal features (Small, Medium,
#     Large → 0, 1, 2). Never for nominal categories.
#
# **One-Hot Encoding (OHE)**
#
#     Creates k binary columns for k categories.
#     One column is always 1, all others 0.
#
#     Drop-first rule: drop one column (the "reference category") to
#     avoid perfect multicollinearity in linear models. Tree models
#     do not need this.
#
#     Pros: no false ordinal relationship. Correct for linear models.
#     Cons: creates k new columns — catastrophic for high-cardinality
#     features (500 cities → 500 columns). High memory. Sparse matrices.
#
#     Rule: safe for < 10–15 unique values. Avoid for > 50.
#
# **Frequency / Count Encoding**
#
#     Replace each category with its frequency (proportion) in the dataset.
#     City→0.35 means that city appears in 35% of rows.
#
#     Pros: no dimensionality explosion, no leakage risk.
#     Cons: two categories with the same frequency are treated identically.
#
#     Leakage safety: compute frequencies from training set only.
#
# **Target Encoding (mean encoding)**
#
#     Replace each category with the mean of the target Y for that category.
#     Paris → mean(Y | city == "Paris")
#
#     Pros: very powerful for high-cardinality features. Captures the
#     relationship between category and target directly.
#     Cons: SEVERE LEAKAGE RISK if done naively. If you compute the mean
#     of Y for Paris using all data (including test), you have leaked the
#     target into the features.
#
#     Correct approach: use out-of-fold target encoding inside each
#     cross-validation fold. sklearn's TargetEncoder handles this.
#     Add smoothing: blend category mean with global mean to handle
#     rare categories (few samples → unreliable mean):
#
#         encoded = (count × category_mean + m × global_mean) / (count + m)
#         m controls shrinkage toward global mean (m ≈ 10–50 typical)
#
# **Hashing / Hash Encoding**
#
#     Apply a hash function to map categories to a fixed number of buckets
#     (e.g., 32 columns regardless of cardinality).
#
#     Pros: fixed dimensionality regardless of cardinality. Handles
#     unseen categories at test time gracefully.
#     Cons: hash collisions (two categories map to the same bucket),
#     not interpretable.
#
#     Use when: extremely high cardinality (user IDs, product SKUs),
#     online learning with unknown vocabulary.
#
# **Embeddings (learned representations)**
#
#     Learn a dense vector per category jointly with the model.
#     Used in neural networks for categorical inputs (e.g., entity
#     embeddings for tabular data, word2vec for text).
#
#     Rule of thumb: embedding dimension ≈ min(50, (n_categories + 1) // 2)
#
#
# ### Encoding Decision Guide
#
#     ┌─────────────────────────────────────────────────────────────────┐
#     │ Feature type          │ Recommended encoding                    │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Binary (yes/no)       │ 0/1 label encoding                      │
#     │ Ordinal (S/M/L/XL)    │ Ordinal label encoding (0/1/2/3)        │
#     │ Nominal, k ≤ 15       │ One-hot encoding (drop_first=True)      │
#     │ Nominal, 15 < k ≤ 50  │ Target encoding (with CV fold safety)   │
#     │ Nominal, k > 50       │ Target encoding or hashing              │
#     │ Neural network        │ Learned embeddings                      │
#     │ Unseen categories     │ Hashing or frequency encoding           │
#     └─────────────────────────────────────────────────────────────────┘
#
#     The pipeline rule: ALL encoding must happen inside the pipeline,
#     fitted on training data only. Target encoding is especially
#     dangerous — use sklearn's TargetEncoder with cv parameter.
#
#
#
# ### PART 7 — DISTRIBUTION SHIFT & DATA DRIFT
#
# ### The Fifth Silent Killer
#
# A model can be flawlessly trained, validated, and deployed — then
# silently degrade in production as the world changes around it.
# Distribution shift is the gap between the data you trained on and
# the data arriving at inference time.
#
# It is undetectable by standard train/test metrics. By definition,
# your evaluation used historical data that matched training.
#
#
# ### Three Types of Shift
#
# **Covariate Shift — P(X) changes, P(Y|X) stays the same**
#
#     The input distribution changes but the relationship between
#     inputs and outputs does not.
#
#     Example: a credit model trained on 25–45 year olds is deployed
#     to a new region where most applicants are 55–70. The feature
#     "age" now has a completely different distribution. The model's
#     learned decision boundary for age was calibrated on a different
#     population.
#
#     Detection: monitor feature distributions (mean, variance,
#     percentiles) over time. PSI and KS test per feature.
#
#     Fix: importance weighting — reweight training examples so that
#     the training distribution matches the deployment distribution.
#     Use density ratio estimation: w(x) = P_new(x) / P_old(x).
#
# **Label Shift (Prior Shift) — P(Y) changes, P(X|Y) stays the same**
#
#     The class proportions change over time, but the features of each
#     class remain the same.
#
#     Example: a fraud model trained on 0.5% fraud rate when fraud rate
#     rises to 2.0%. The model's decision threshold (calibrated for 0.5%)
#     will miss most fraud at the new base rate.
#
#     Fix: recalibrate the decision threshold. Estimate P_new(Y) and
#     use Bayes' theorem to adjust the posterior.
#
# **Concept Drift — P(Y|X) changes**
#
#     The fundamental relationship between inputs and outputs changes.
#     This is the most dangerous form — no amount of reweighting fixes it.
#
#     Example: "high income → low default risk" was true in 2019.
#     After a recession, many high-income borrowers began defaulting
#     at much higher rates. The concept has changed.
#
#     Gradual drift: the relationship shifts slowly over months.
#     Sudden drift: a shock event (pandemic, regulation change) causes
#     an abrupt change.
#     Recurring drift: seasonal patterns (Christmas fraud patterns differ
#     from non-seasonal fraud patterns).
#
#     Fix: retrain on recent data, use windowed training, maintain
#     ensemble of models from different time periods.
#
#
# ### Detection Methods
#
# **Population Stability Index (PSI)**
#
#     PSI = Σₖ (P_new(k) − P_old(k)) × ln(P_new(k) / P_old(k))
#
#     Binned version of KL divergence. Standard threshold:
#         PSI < 0.1:   no significant shift — model still valid
#         PSI 0.1–0.2: moderate shift — investigate
#         PSI > 0.2:   major shift — retrain required
#
#     Apply PSI to each input feature and to the model's output score.
#
# **Kolmogorov-Smirnov (KS) Test**
#
#     Two-sample KS test: does the new batch come from the same
#     distribution as the training data?
#     Returns a p-value — reject H₀ (same distribution) if p < 0.05.
#     Apply per feature. Multiple testing correction needed.
#
# **Chi-squared Test (categorical features)**
#
#     Compares observed vs expected category frequencies between
#     training and new data. Detects shifts in categorical distributions.
#
# **Monitoring the Prediction Distribution**
#
#     Even without ground truth labels (which may lag by days/months),
#     monitor the distribution of the model's output scores.
#     If the score distribution shifts, the model has encountered data
#     it is not calibrated for.
#
#
# ### Mitigation Strategies
#
#     ┌─────────────────────────────────────────────────────────────────┐
#     │ Shift type      │ Fix                                           │
#     ├─────────────────────────────────────────────────────────────────┤
#     │ Covariate shift │ Importance weighting, domain adaptation       │
#     │ Label shift     │ Recalibrate threshold, prior correction       │
#     │ Concept drift   │ Retrain on recent data, windowed training     │
#     │ All types       │ Monitor PSI + KS, set retraining triggers     │
#     │                 │ Maintain model versioning and rollback        │
#     └─────────────────────────────────────────────────────────────────┘
#
#
# ### PART 8 — FIX ORDERING & INTERACTIONS
#
# ### The Correct Preprocessing Order
#
# Order matters. Many bugs arise from applying transformations out of
# sequence. The canonical correct order:
#
#     1. SPLIT first (train / validation / test) — before anything else
#     2. Identify missingness mechanism (MCAR/MAR/MNAR) on training set
#     3. Create missing indicator columns (for MNAR features)
#     4. Impute missing values (fit imputer on train, transform all)
#     5. Encode categoricals (fit encoder on train, transform all)
#     6. Detect and treat outliers (fit winsorizer on train, transform all)
#     7. Scale features (fit scaler on train, transform all)
#     8. Resample for imbalance (ONLY on training set — never test)
#     9. Feature selection (fit selector on train only)
#     10. Train model
#
#     Steps 3–9 must ALL be inside a sklearn Pipeline or equivalent.
#     Steps 3–7 use fit on train, transform on train AND test.
#     Step 8 uses transform on train ONLY (never oversample the test set).
#
#
# ### Critical Interaction: SMOTE Must Come After Splitting
#
#     WRONG — common mistake:
#         X_resampled, y_resampled = SMOTE().fit_resample(X_all, y_all)
#         X_tr, X_te = train_test_split(X_resampled)
#         # Synthetic samples may be near-duplicates of test examples.
#         # Test set is polluted with "near-future" information.
#
#     CORRECT:
#         X_tr, X_te = train_test_split(X_all)
#         X_tr_res, y_tr_res = SMOTE().fit_resample(X_tr, y_tr)
#         # Synthetic samples only generated from training distribution.
#         # Test set is untouched.
#
#     This is one of the most common Kaggle "winning" bugs — SMOTE before
#     split inflates scores artificially.
#
#
# ### Critical Interaction: Target Encoding Is a Leakage Trap
#
#     WRONG:
#         df["city_encoded"] = df.groupby("city")["target"].transform("mean")
#         X_tr, X_te = train_test_split(df)
#         # Test target values contributed to encoding ALL rows including train.
#
#     CORRECT:
#         X_tr, X_te = train_test_split(df)
#         encoder = TargetEncoder(cv=5).fit(X_tr["city"], y_tr)
#         X_tr["city_encoded"] = encoder.transform(X_tr["city"])
#         X_te["city_encoded"] = encoder.transform(X_te["city"])
#         # cv=5 means each training row's encoding used only the other 4 folds.
#
#
# ### Critical Interaction: Class Weights Change Probability Calibration
#
#     When you use class_weight='balanced', the model learns shifted
#     decision probabilities. The predicted probability P(y=1|x) is no
#     longer the true posterior probability — it is inflated toward 0.5.
#
#     Consequence: threshold = 0.5 is NO LONGER the correct threshold
#     after using class weights. You must always tune the threshold on
#     a held-out validation set after applying class weights.
#
#     With imbalance ratio r = n_neg / n_pos and balanced class weights,
#     the effective threshold shifts approximately to:
#         optimal_threshold ≈ 1 / (1 + r)
#     (e.g., for 10:1 imbalance: threshold ≈ 1/11 ≈ 0.09)
#
#
# ### Complete Interaction Map
#
#     ┌──────────────────────────────────────────────────────────────────┐
#     │ Step              │ Depends on                │ Feeds into       │
#     ├──────────────────────────────────────────────────────────────────┤
#     │ Missing indicator │ MNAR identification       │ Imputation step  │
#     │ Imputation        │ Split (train stats only)  │ Encoding, scale  │
#     │ Encoding          │ Split (train cats only)   │ Scaling, model   │
#     │ Outlier treatment │ Split (train percentiles) │ Scaling, model   │
#     │ Scaling           │ Post-imputation values    │ Model, SMOTE     │
#     │ SMOTE             │ Post-split train only     │ Model training   │
#     │ Feature selection │ Post-split train+labels   │ Model training   │
#     │ Threshold tuning  │ Class weights used?       │ Final evaluation │
#     └──────────────────────────────────────────────────────────────────┘
#
# """
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # OPERATIONS
# # ─────────────────────────────────────────────────────────────────────────────
#
# OPERATIONS = {
#
#     # ── 1 ─────────────────────────────────────────────────────────────────────
#     "1 · Missing Data — MCAR/MAR/MNAR Detection & Imputation Comparison": {
#         "description": (
#             "Generate datasets with MCAR, MAR, and MNAR missingness. "
#             "Show how each mechanism distorts the data distribution differently. "
#             "Compare mean, median, KNN, and iterative imputation strategies."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.experimental import enable_iterative_imputer  # noqa
# from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
# from sklearn.linear_model import LogisticRegression, BayesianRidge
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# np.random.seed(42)
#
# print("=" * 65)
# print("  MISSING DATA: MCAR vs MAR vs MNAR")
# print("=" * 65)
# print()
#
# # ── Generate clean dataset: 3 features, binary target ─────────────────────
# n = 1000
# X_clean = np.column_stack([
#     np.random.normal(0,  1, n),     # feature 0: income proxy
#     np.random.normal(50, 15, n),    # feature 1: age
#     np.random.normal(0,  1, n),     # feature 2: credit score proxy
# ])
# y = (X_clean[:, 0] + 0.5*X_clean[:, 2] > 0).astype(int)
#
# def introduce_missing(X, mechanism, missing_rate=0.20):
#     Xm = X.copy().astype(float)
#     n  = len(Xm)
#     if mechanism == "MCAR":
#         # Missing completely at random — no relationship to any variable
#         mask = np.random.rand(n) < missing_rate
#         Xm[mask, 0] = np.nan
#
#     elif mechanism == "MAR":
#         # Missing depends on observed variable (age): older -> more missing
#         age_norm = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
#         prob_missing = missing_rate * 2 * age_norm   # higher age -> more missing
#         prob_missing = np.clip(prob_missing, 0, 0.95)
#         mask = np.random.rand(n) < prob_missing
#         Xm[mask, 0] = np.nan
#
#     elif mechanism == "MNAR":
#         # Missing depends on the VALUE ITSELF: high income -> more missing
#         income_high = X[:, 0] > np.percentile(X[:, 0], 75)
#         mask = income_high & (np.random.rand(n) < 0.70)
#         Xm[mask, 0] = np.nan
#
#     return Xm
#
# # ── Show how each mechanism distorts the observed distribution ─────────────
# print("  HOW MISSINGNESS DISTORTS OBSERVED DISTRIBUTION OF FEATURE 0:")
# print(f"  {'Mechanism':10s} | {'% Missing':>10} | {'Observed Mean':>14} | "
#       f"{'True Mean':>10} | {'Bias':>8}")
# print(f"  {'─'*60}")
#
# true_mean = X_clean[:, 0].mean()
# for mech in ["MCAR", "MAR", "MNAR"]:
#     Xm = introduce_missing(X_clean, mech, 0.20)
#     observed = Xm[:, 0][~np.isnan(Xm[:, 0])]
#     pct_miss = np.isnan(Xm[:, 0]).mean() * 100
#     obs_mean = observed.mean()
#     bias     = obs_mean - true_mean
#     print(f"  {mech:10s} | {pct_miss:9.1f}% | {obs_mean:14.4f} | "
#           f"{true_mean:10.4f} | {bias:+8.4f}")
#
# print()
# print("  MCAR: observed mean ≈ true mean (unbiased sample)")
# print("  MAR:  slight bias (older patients missing → sample skewed younger)")
# print("  MNAR: LARGE bias (high earners missing → observed mean too LOW)")
# print()
#
# # ── Compare imputation strategies on MNAR (hardest case) ──────────────────
# print("  IMPUTATION STRATEGY COMPARISON (MNAR, 20% missing):")
# print()
# Xm_mnar = introduce_missing(X_clean, "MNAR", 0.20)
#
# # Add missingness indicator feature
# missing_indicator = np.isnan(Xm_mnar[:, 0]).astype(float).reshape(-1, 1)
# Xm_with_flag = np.hstack([Xm_mnar, missing_indicator])
#
# X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y,  test_size=0.3, random_state=0)
# Xm_tr, Xm_te           = train_test_split(Xm_mnar,    test_size=0.3, random_state=0)
# Xf_tr, Xf_te           = train_test_split(Xm_with_flag, test_size=0.3, random_state=0)
#
# strategies = [
#     ("No missing (oracle)",    X_tr,  X_te,
#      None),
#     ("Drop rows with NaN",     X_tr[~np.isnan(Xm_tr[:,0])],
#      X_te,  None),
#     ("Mean imputation",        Xm_tr, Xm_te,
#      SimpleImputer(strategy="mean")),
#     ("Median imputation",      Xm_tr, Xm_te,
#      SimpleImputer(strategy="median")),
#     ("KNN imputation (k=5)",   Xm_tr, Xm_te,
#      KNNImputer(n_neighbors=5)),
#     ("Iterative (MICE)",       Xm_tr, Xm_te,
#      IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=0)),
#     ("Mean + miss indicator",  Xf_tr, Xf_te,
#      SimpleImputer(strategy="mean")),
# ]
#
# scaler_ref = StandardScaler().fit(X_tr)
#
# print(f"  {'Strategy':30s} | {'Train n':>8} | {'Test Acc':>10} | Notes")
# print(f"  {'─'*70}")
#
# for name, X_train_s, X_test_s, imputer in strategies:
#     try:
#         if imputer is not None:
#             X_train_imp = imputer.fit_transform(X_train_s)
#             X_test_imp  = imputer.transform(X_test_s)
#         else:
#             X_train_imp = X_train_s
#             X_test_imp  = X_test_s
#
#         if np.isnan(X_train_imp).any() or np.isnan(X_test_imp).any():
#             print(f"  {name:30s} | {'NaN':>8} | {'NaN':>10} | still has NaN")
#             continue
#
#         sc = StandardScaler().fit(X_train_imp)
#         clf = LogisticRegression(max_iter=500, random_state=0)
#         clf.fit(sc.transform(X_train_imp), y_tr[:len(X_train_imp)])
#         acc = accuracy_score(y_te, clf.predict(sc.transform(X_test_imp)))
#
#         note = ""
#         if "oracle" in name: note = "upper bound"
#         if "Drop"   in name: note = "loses 70% of test rows if also applied to test"
#         if "indicator" in name: note = "best for MNAR: flag encodes missingness as signal"
#
#         print(f"  {name:30s} | {len(X_train_imp):8d} | {acc:10.4f} | {note}")
#     except Exception as e:
#         print(f"  {name:30s} | {'ERR':>8} | {'ERR':>10} | {str(e)[:30]}")
#
# print()
# print("  KEY LESSON:")
# print("  For MNAR: mean/median imputation introduces systematic bias.")
# print("  The 'missing indicator' flag lets the model learn that")
# print("  NaN itself is an informative signal (high earner pattern).")
#
# # ── Plot missingness patterns ──────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle("How Missingness Mechanism Distorts Observed Distribution",
#              fontsize=12, fontweight="bold")
#
# for ax, mech, colour in zip(axes,
#         ["MCAR", "MAR", "MNAR"],
#         ["steelblue", "seagreen", "tomato"]):
#     Xm = introduce_missing(X_clean, mech, 0.20)
#     observed   = Xm[:, 0][~np.isnan(Xm[:, 0])]
#     true_vals  = X_clean[:, 0]
#     ax.hist(true_vals,  bins=40, alpha=0.35, color="gray",  label="True (full)")
#     ax.hist(observed,   bins=40, alpha=0.65, color=colour,  label="Observed (non-missing)")
#     ax.axvline(true_vals.mean(),  color="black",  lw=2, linestyle="--", label="True mean")
#     ax.axvline(observed.mean(),   color=colour,   lw=2, linestyle="-",  label="Observed mean")
#     pct = np.isnan(Xm[:, 0]).mean() * 100
#     ax.set_title(mech + f" ({pct:.0f}% missing)", fontsize=11, fontweight="bold")
#     ax.legend(fontsize=8); ax.set_xlabel("Feature 0 value"); ax.grid(alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("missing_data_mechanisms.png", dpi=120)
# print()
# print("  Plot saved -> missing_data_mechanisms.png")
# ''',
#     },
#
#     # ── 2 ─────────────────────────────────────────────────────────────────────
#     "2 · Feature Scaling — Gradient Descent Convergence With vs Without": {
#         "description": (
#             "Show concretely how unscaled features create a distorted loss landscape "
#             "and cause gradient descent to oscillate. Measure epochs to convergence "
#             "with StandardScaler, MinMaxScaler, and no scaling."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# np.random.seed(0)
# print("=" * 65)
# print("  FEATURE SCALING: GRADIENT DESCENT CONVERGENCE")
# print("=" * 65)
# print()
#
# # ── Create a deliberately badly-scaled dataset ────────────────────────────
# n = 800
# X_base, y = make_classification(n_samples=n, n_features=4,
#                                  n_informative=3, n_redundant=1, random_state=42)
#
# # Deliberately scale features to wildly different ranges
# X_unscaled = X_base.copy()
# X_unscaled[:, 0] *= 100000   # salary: range ~ [-300k, 300k]
# X_unscaled[:, 1] *= 1        # age: range ~ [-3, 3]
# X_unscaled[:, 2] *= 50       # price: range ~ [-150, 150]
# X_unscaled[:, 3] *= 0.001    # small feature: ~ [-0.003, 0.003]
#
# X_tr, X_te, y_tr, y_te = train_test_split(X_unscaled, y,
#                                             test_size=0.25, random_state=1)
#
# print("  FEATURE RANGES IN UNSCALED DATA:")
# print(f"  {'Feature':12s} | {'Min':>12} | {'Max':>12} | {'Std':>12}")
# print(f"  {'─'*52}")
# names = ["Salary(x1e5)", "Age(x1)", "Price(x50)", "Tiny(x1e-3)"]
# for i, fname in enumerate(names):
#     col = X_unscaled[:, i]
#     print(f"  {fname:12s} | {col.min():12.3f} | {col.max():12.3f} | {col.std():12.3f}")
# print()
#
# # ── Train with manual gradient descent to show convergence ────────────────
# def sigmoid(z):
#     return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
#
# def train_gd(X, y, lr=1e-6, n_epochs=200):
#     """Mini logistic regression via gradient descent. Returns loss per epoch."""
#     m, d = X.shape
#     w = np.zeros(d)
#     b = 0.0
#     losses = []
#     for _ in range(n_epochs):
#         p   = sigmoid(X @ w + b)
#         p   = np.clip(p, 1e-7, 1 - 1e-7)
#         loss = -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
#         losses.append(loss)
#         grad_w = (p - y) @ X / m
#         grad_b = (p - y).mean()
#         w -= lr * grad_w
#         b -= lr * grad_b
#     return losses, w, b
#
# # Learning rate tuned per scaler (to make comparison fair)
# configs = [
#     ("No Scaling",       X_tr,                                X_te,  5e-9,  "tomato"),
#     ("StandardScaler",   StandardScaler().fit_transform(X_tr),None,  0.1,   "steelblue"),
#     ("MinMaxScaler",     MinMaxScaler().fit_transform(X_tr),  None,  0.1,   "seagreen"),
#     ("RobustScaler",     RobustScaler().fit_transform(X_tr),  None,  0.1,   "purple"),
# ]
#
# # Prepare test sets
# std_sc  = StandardScaler().fit(X_tr)
# mm_sc   = MinMaxScaler().fit(X_tr)
# rob_sc  = RobustScaler().fit(X_tr)
#
# configs[1] = ("StandardScaler", std_sc.fit_transform(X_tr), std_sc.transform(X_te),  0.1,  "steelblue")
# configs[2] = ("MinMaxScaler",   mm_sc.fit_transform(X_tr),  mm_sc.transform(X_te),   0.1,  "seagreen")
# configs[3] = ("RobustScaler",   rob_sc.fit_transform(X_tr), rob_sc.transform(X_te),  0.1,  "purple")
#
# N_EPOCHS = 300
# fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# fig.suptitle("Effect of Feature Scaling on Gradient Descent Convergence",
#              fontsize=12, fontweight="bold")
#
# print(f"  {'Scaler':18s} | {'Final Loss':>11} | {'Converged?':>11} | {'Test Acc':>10}")
# print(f"  {'─'*58}")
#
# for name, Xtr_s, Xte_s, lr, colour in configs:
#     losses, w, b = train_gd(Xtr_s, y_tr, lr=lr, n_epochs=N_EPOCHS)
#
#     final_loss  = losses[-1]
#     converged   = final_loss < 0.55   # below random baseline
#
#     if Xte_s is not None:
#         p_te = sigmoid(Xte_s @ w + b)
#         acc  = ((p_te > 0.5).astype(int) == y_te).mean()
#     else:
#         # Use original test data with tiny lr
#         p_te = sigmoid(X_te @ w + b)
#         acc  = ((p_te > 0.5).astype(int) == y_te).mean()
#
#     conv_str = "Yes" if converged else "No (stuck)"
#     print(f"  {name:18s} | {final_loss:11.4f} | {conv_str:>11} | {acc:10.4f}")
#
#     axes[0].plot(losses, colour, lw=2, label=name)
#     axes[1].plot(losses[-50:], colour, lw=2)
#
# axes[0].set_title("Loss Curve — All 300 Epochs")
# axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
# axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
# axes[0].set_ylim(0, 1.5)
#
# axes[1].set_title("Loss Curve — Last 50 Epochs (convergence detail)")
# axes[1].set_xlabel("Epoch (last 50)"); axes[1].set_ylabel("BCE Loss")
# axes[1].grid(alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("scaling_convergence.png", dpi=120)
#
# print()
# print("  KEY LESSON:")
# print("  Without scaling, gradient descent requires an extremely tiny")
# print("  learning rate (5e-9) to avoid exploding, and still barely moves.")
# print("  With any scaling, the same lr=0.1 works fine and converges fast.")
# print()
# print("  Gradient at epoch 1 without scaling:")
# X_demo = X_tr.copy()
# p0 = sigmoid(X_demo @ np.zeros(4) + 0)
# g  = (p0 - y_tr) @ X_demo / len(y_tr)
# print(f"    Salary gradient:    {g[0]:+.2e}  (enormous — overshoots)")
# print(f"    Age gradient:       {g[1]:+.2e}")
# print(f"    Price gradient:     {g[2]:+.2e}")
# print(f"    Tiny feat gradient: {g[3]:+.2e}  (negligible — never moves)")
# print()
# print("  After StandardScaler, all gradients are on the same scale,")
# print("  so a single learning rate works for every parameter.")
# print()
# print("  Plot saved -> scaling_convergence.png")
# ''',
#     },
#
#     # ── 3 ─────────────────────────────────────────────────────────────────────
#     "3 · Scaler Comparison — Outlier Sensitivity Demonstrated": {
#         "description": (
#             "Compare StandardScaler, MinMaxScaler, and RobustScaler "
#             "on a dataset with outliers. Show how outliers compress the "
#             "distribution with Standard/MinMax but not with Robust. "
#             "Then compare downstream model performance."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import make_classification
#
# np.random.seed(7)
# print("=" * 65)
# print("  SCALER COMPARISON: OUTLIER SENSITIVITY")
# print("=" * 65)
# print()
#
# # ── Dataset with realistic outliers ───────────────────────────────────────
# n = 600
# X, y = make_classification(n_samples=n, n_features=4,
#                             n_informative=3, n_redundant=1, random_state=0)
#
# # Inject outliers: 3% of rows have extreme salary values
# outlier_idx = np.random.choice(n, size=int(0.03*n), replace=False)
# X[outlier_idx, 0] *= 50   # extreme outliers in feature 0
#
# print(f"  Dataset: {n} samples, 4 features, 3% outliers in feature 0")
# print()
#
# # ── Show how each scaler handles outliers numerically ────────────────────
# scalers = [
#     ("StandardScaler", StandardScaler()),
#     ("MinMaxScaler",   MinMaxScaler()),
#     ("RobustScaler",   RobustScaler()),
# ]
#
# print("  FEATURE 0 AFTER SCALING (showing outlier effect on normal values):")
# print()
#
# X_demo = X[:, 0].copy()
# normal_mask   = np.ones(len(X_demo), dtype=bool)
# normal_mask[outlier_idx] = False
#
# print(f"  {'Scaler':18s} | {'Normal val range':>18} | {'Outlier val range':>18} | {'Compression?'}")
# print(f"  {'─'*76}")
#
# raw_normal_range = f"[{X_demo[normal_mask].min():.2f}, {X_demo[normal_mask].max():.2f}]"
# raw_outlier_range = f"[{X_demo[outlier_idx].min():.1f}, {X_demo[outlier_idx].max():.1f}]"
# print(f"  {'Raw (unscaled)':18s} | {raw_normal_range:>18} | {raw_outlier_range:>18} | baseline")
#
# for name, scaler in scalers:
#     X_s  = scaler.fit_transform(X[:, 0].reshape(-1, 1)).ravel()
#     n_rng = f"[{X_s[normal_mask].min():.3f}, {X_s[normal_mask].max():.3f}]"
#     o_rng = f"[{X_s[outlier_idx].min():.1f}, {X_s[outlier_idx].max():.1f}]"
#     n_span = X_s[normal_mask].max() - X_s[normal_mask].min()
#     compress = "SEVERE" if n_span < 0.1 else ("moderate" if n_span < 0.5 else "none")
#     print(f"  {name:18s} | {n_rng:>18} | {o_rng:>18} | {compress}")
#
# print()
# print("  MinMaxScaler compresses all normal values into [0, 0.02]!")
# print("  StandardScaler compresses them into [-0.3, 0.3].")
# print("  RobustScaler keeps normal values in a wide, useful range.")
# print()
#
# # ── Compare KNN performance (distance-based — very sensitive to scale) ────
# print("  KNN CLASSIFIER PERFORMANCE (5-fold CV, k=7):")
# print("  KNN uses Euclidean distance — extremely sensitive to scaling.")
# print()
# print(f"  {'Preprocessing':25s} | {'CV Accuracy':>12} | {'Std':>8}")
# print(f"  {'─'*50}")
#
# knn = KNeighborsClassifier(n_neighbors=7)
# configs = [
#     ("No scaling",       X),
#     ("StandardScaler",   StandardScaler().fit_transform(X)),
#     ("MinMaxScaler",     MinMaxScaler().fit_transform(X)),
#     ("RobustScaler",     RobustScaler().fit_transform(X)),
# ]
#
# for name, X_proc in configs:
#     scores = cross_val_score(knn, X_proc, y, cv=5, scoring="accuracy")
#     flag = " <- recommended" if name == "RobustScaler" else ""
#     print(f"  {name:25s} | {scores.mean():12.4f} | {scores.std():8.4f}{flag}")
#
# print()
#
# # ── Visualise compression ─────────────────────────────────────────────────
# fig, axes = plt.subplots(2, 2, figsize=(13, 8))
# fig.suptitle("Scaler Comparison: Effect of Outliers on Feature Distribution",
#              fontsize=12, fontweight="bold")
#
# plot_configs = [
#     ("Raw (unscaled)", X[:, 0],
#      "gray"),
#     ("StandardScaler", StandardScaler().fit_transform(X[:, 0].reshape(-1,1)).ravel(),
#      "steelblue"),
#     ("MinMaxScaler",   MinMaxScaler().fit_transform(X[:, 0].reshape(-1,1)).ravel(),
#      "tomato"),
#     ("RobustScaler",   RobustScaler().fit_transform(X[:, 0].reshape(-1,1)).ravel(),
#      "seagreen"),
# ]
#
# for ax, (name, vals, colour) in zip(axes.ravel(), plot_configs):
#     # Show only the non-outlier range for clarity (clip at 99.9th percentile)
#     clip_high = np.percentile(vals, 99.5)
#     clip_low  = np.percentile(vals, 0.5)
#     clipped   = np.clip(vals, clip_low, clip_high)
#     ax.hist(clipped, bins=40, color=colour, alpha=0.75, density=True)
#     ax.hist(vals[normal_mask], bins=40, color="gray", alpha=0.35,
#             density=False, label="Normal values")
#     span = vals[normal_mask].max() - vals[normal_mask].min()
#     ax.set_title(f"{name} — Normal span={span:.3f}", fontsize=10)
#     ax.set_xlabel("Scaled value"); ax.grid(alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("scaler_comparison.png", dpi=120)
# print("  Plot saved -> scaler_comparison.png")
# ''',
#     },
#
#     # ── 4 ─────────────────────────────────────────────────────────────────────
#     "4 · Class Imbalance — Accuracy Paradox & 5 Fix Strategies": {
#         "description": (
#             "Demonstrate the accuracy paradox on a 98:2 imbalanced dataset. "
#             "Compare 5 strategies: baseline, threshold tuning, class weights, "
#             "SMOTE oversampling, and undersampling. "
#             "Show why F1/MCC are better metrics than accuracy."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (accuracy_score, f1_score, recall_score,
#                              precision_score, matthews_corrcoef,
#                              average_precision_score, roc_auc_score,
#                              confusion_matrix)
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import resample
#
# np.random.seed(0)
# print("=" * 65)
# print("  CLASS IMBALANCE: ACCURACY PARADOX AND 5 FIX STRATEGIES")
# print("=" * 65)
# print()
#
# # ── Severely imbalanced dataset: 98:2 ─────────────────────────────────────
# X, y = make_classification(
#     n_samples=5000, n_features=10, n_informative=6,
#     weights=[0.98, 0.02], flip_y=0.005, random_state=42)
#
# X = StandardScaler().fit_transform(X)
# X_tr, X_te, y_tr, y_te = train_test_split(
#     X, y, test_size=0.3, stratify=y, random_state=1)
#
# print(f"  Dataset: {len(y)} samples")
# print(f"  Majority (class 0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
# print(f"  Minority (class 1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
# print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
# print()
#
# def evaluate(name, y_true, y_pred, y_prob):
#     acc   = accuracy_score(y_true, y_pred)
#     f1    = f1_score(y_true, y_pred, zero_division=0)
#     rec   = recall_score(y_true, y_pred, zero_division=0)
#     prec  = precision_score(y_true, y_pred, zero_division=0)
#     mcc   = matthews_corrcoef(y_true, y_pred)
#     prauc = average_precision_score(y_true, y_prob)
#     return acc, f1, rec, prec, mcc, prauc
#
# # ── BASELINE: plain logistic regression ──────────────────────────────────
# clf_base = LogisticRegression(max_iter=500, random_state=0).fit(X_tr, y_tr)
# y_pred_base = clf_base.predict(X_te)
# y_prob_base = clf_base.predict_proba(X_te)[:, 1]
#
# # ── FIX 1: Threshold tuning ───────────────────────────────────────────────
# # Find threshold maximising F1 on validation
# val_size  = int(0.2 * len(X_tr))
# X_val, y_val = X_tr[:val_size], y_tr[:val_size]
# X_tr2, y_tr2 = X_tr[val_size:], y_tr[val_size:]
#
# clf_thr = LogisticRegression(max_iter=500, random_state=0).fit(X_tr2, y_tr2)
# val_probs = clf_thr.predict_proba(X_val)[:, 1]
#
# best_f1, best_thr = 0, 0.5
# for thr in np.arange(0.05, 0.80, 0.01):
#     pred_thr = (val_probs > thr).astype(int)
#     f1_thr   = f1_score(y_val, pred_thr, zero_division=0)
#     if f1_thr > best_f1:
#         best_f1, best_thr = f1_thr, thr
#
# y_pred_thr  = (clf_thr.predict_proba(X_te)[:, 1] > best_thr).astype(int)
# y_prob_thr  = clf_thr.predict_proba(X_te)[:, 1]
#
# # ── FIX 2: Class weights ──────────────────────────────────────────────────
# clf_wt = LogisticRegression(max_iter=500, class_weight="balanced",
#                              random_state=0).fit(X_tr, y_tr)
# y_pred_wt  = clf_wt.predict(X_te)
# y_prob_wt  = clf_wt.predict_proba(X_te)[:, 1]
#
# # ── FIX 3: SMOTE oversampling (manual implementation) ────────────────────
# def smote_simple(X_min, n_synthetic):
#     """Generate synthetic minority samples via linear interpolation."""
#     from sklearn.neighbors import NearestNeighbors
#     nn = NearestNeighbors(n_neighbors=6).fit(X_min)
#     _, indices = nn.kneighbors(X_min)
#     synthetic = []
#     for _ in range(n_synthetic):
#         i   = np.random.randint(0, len(X_min))
#         j   = indices[i, np.random.randint(1, 6)]
#         lam = np.random.uniform(0, 1)
#         synthetic.append(X_min[i] + lam * (X_min[j] - X_min[i]))
#     return np.array(synthetic)
#
# X_min_tr = X_tr[y_tr == 1]
# X_maj_tr = X_tr[y_tr == 0]
# n_synth   = len(X_maj_tr) - len(X_min_tr)
# X_synth   = smote_simple(X_min_tr, n_synth)
# X_tr_smote = np.vstack([X_tr, X_synth])
# y_tr_smote = np.hstack([y_tr, np.ones(n_synth)])
#
# clf_smote = LogisticRegression(max_iter=500, random_state=0).fit(
#     X_tr_smote, y_tr_smote)
# y_pred_smote = clf_smote.predict(X_te)
# y_prob_smote = clf_smote.predict_proba(X_te)[:, 1]
#
# # ── FIX 4: Random undersampling ───────────────────────────────────────────
# n_min = (y_tr == 1).sum()
# X_maj_down = resample(X_maj_tr, n_samples=n_min*3, random_state=0, replace=False)
# X_tr_under = np.vstack([X_maj_down, X_min_tr])
# y_tr_under = np.hstack([np.zeros(len(X_maj_down)), np.ones(len(X_min_tr))])
#
# clf_under = LogisticRegression(max_iter=500, random_state=0).fit(
#     X_tr_under, y_tr_under)
# y_pred_under = clf_under.predict(X_te)
# y_prob_under = clf_under.predict_proba(X_te)[:, 1]
#
# # ── Naive "always predict negative" baseline ──────────────────────────────
# y_pred_naive = np.zeros(len(y_te), dtype=int)
# y_prob_naive = np.zeros(len(y_te))
#
# # ── Print results table ───────────────────────────────────────────────────
# strategies = [
#     ("Always predict 0 (naive)", y_pred_naive, y_prob_naive + 1e-9),
#     ("Baseline LR (thr=0.5)",    y_pred_base,  y_prob_base),
#     ("Fix 1: Threshold tuning",  y_pred_thr,   y_prob_thr),
#     ("Fix 2: Class weights",     y_pred_wt,    y_prob_wt),
#     ("Fix 3: SMOTE oversample",  y_pred_smote, y_prob_smote),
#     ("Fix 4: Undersample 3:1",   y_pred_under, y_prob_under),
# ]
#
# print(f"  {'Strategy':28s} | {'Acc':>6} | {'F1':>6} | {'Recall':>7} | "
#       f"{'Prec':>6} | {'MCC':>7} | {'PR-AUC':>7}")
# print(f"  {'─'*82}")
#
# all_metrics = []
# for name, y_pred, y_prob in strategies:
#     acc, f1, rec, prec, mcc, prauc = evaluate(name, y_te, y_pred, y_prob)
#     all_metrics.append((name, acc, f1, rec, prec, mcc, prauc))
#     flag = " <- ACCURACY PARADOX!" if name.startswith("Always") else ""
#     print(f"  {name:28s} | {acc:6.3f} | {f1:6.3f} | {rec:7.3f} | "
#           f"{prec:6.3f} | {mcc:7.3f} | {prauc:7.3f}{flag}")
#
# print()
# print("  OBSERVATIONS:")
# print("  - 'Always predict 0' gets 98% accuracy but F1=0, MCC=0")
# print("  - Baseline LR also nearly useless (Recall~0 means zero positives caught)")
# print("  - Class weights and SMOTE dramatically improve F1 and MCC")
# print("  - Use F1/MCC/PR-AUC — never accuracy alone — for imbalanced data")
# print(f"  - Best threshold found on validation: {best_thr:.2f} (not 0.5)")
#
# # ── Plot ──────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle("Class Imbalance: Why Accuracy Misleads and How Fixes Help",
#              fontsize=12, fontweight="bold")
#
# metric_names = ["Accuracy", "F1", "Recall", "Precision", "MCC"]
# colours = ["tomato", "steelblue", "seagreen", "purple", "orange"]
#
# x_pos = np.arange(len(strategies))
# width = 0.35
#
# accs  = [m[1] for m in all_metrics]
# f1s   = [m[2] for m in all_metrics]
# mccs  = [(m[5] + 1) / 2 for m in all_metrics]   # rescale MCC to [0,1] for plot
#
# axes[0].bar(x_pos, accs, color="steelblue", alpha=0.8)
# axes[0].set_title("Accuracy (misleading!)")
# axes[0].set_xticks(x_pos)
# axes[0].set_xticklabels([m[0][:18] for m in all_metrics], rotation=30, ha="right", fontsize=7)
# axes[0].axhline(0.98, color="red", linestyle="--", lw=1.5,
#                 label="Naive baseline: 98%")
# axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3, axis="y")
#
# axes[1].bar(x_pos, f1s, color="seagreen", alpha=0.8)
# axes[1].set_title("F1 Score (true performance)")
# axes[1].set_xticks(x_pos)
# axes[1].set_xticklabels([m[0][:18] for m in all_metrics], rotation=30, ha="right", fontsize=7)
# axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3, axis="y")
#
# axes[2].bar(x_pos, [m[5] for m in all_metrics], color="purple", alpha=0.8)
# axes[2].set_title("MCC (best single metric)")
# axes[2].set_xticks(x_pos)
# axes[2].set_xticklabels([m[0][:18] for m in all_metrics], rotation=30, ha="right", fontsize=7)
# axes[2].axhline(0, color="gray", linestyle="--", lw=1, label="Random = 0")
# axes[2].legend(fontsize=8); axes[2].set_ylim(-0.1, 1.0); axes[2].grid(alpha=0.3, axis="y")
#
# plt.tight_layout()
# plt.savefig("class_imbalance_strategies.png", dpi=120)
# print()
# print("  Plot saved -> class_imbalance_strategies.png")
# ''',
#     },
#
#     # ── 5 ─────────────────────────────────────────────────────────────────────
#     "5 · Data Leakage — Three Types Demonstrated With Inflated Metrics": {
#         "description": (
#             "Demonstrate three leakage types with concrete examples: "
#             "target leakage (feature contains the label), "
#             "train-test contamination (scaling before split), "
#             "and pipeline leakage (feature selection before split). "
#             "Measure the performance inflation each type creates."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
#
# np.random.seed(42)
# print("=" * 65)
# print("  DATA LEAKAGE: THREE TYPES DEMONSTRATED")
# print("=" * 65)
# print()
#
# # ── Base dataset ──────────────────────────────────────────────────────────
# X_base, y = make_classification(
#     n_samples=1000, n_features=20, n_informative=5,
#     n_redundant=5, random_state=0)
#
# print("  Base task: 20 features, only 5 truly informative")
# print("  Honest upper bound (oracle knows which 5 features are real):")
# X_good = X_base[:, :5]   # pretend we know the 5 informative features
# sc = StandardScaler()
# X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(X_good, y, test_size=0.3,
#                                                      random_state=1)
# sc.fit(X_tr_g)
# clf = LogisticRegression(max_iter=500, random_state=0)
# clf.fit(sc.transform(X_tr_g), y_tr_g)
# oracle_acc = accuracy_score(y_te_g, clf.predict(sc.transform(X_te_g)))
# print(f"  Oracle test accuracy: {oracle_acc:.4f}")
# print()
#
# # ─────────────────────────────────────────────────────────────────────────
# # LEAKAGE TYPE 1: TARGET LEAKAGE
# # ─────────────────────────────────────────────────────────────────────────
# print("  " + "─"*60)
# print("  LEAKAGE TYPE 1 — TARGET LEAKAGE")
# print("  " + "─"*60)
# print("  Scenario: fraud detection. We accidentally include a feature")
# print("  that is a CONSEQUENCE of the label (fraud happened).")
# print()
#
# # Create a leaky feature: highly correlated with y but would not exist
# # before the label is known (e.g., 'account frozen' = consequence of fraud)
# leaky_feat = y + np.random.normal(0, 0.3, len(y))   # ~90% correlated with y
#
# X_leaked = np.column_stack([X_base, leaky_feat])     # add leaky column
# X_clean  = X_base.copy()                              # no leaky column
#
# results = {}
# for name, X_use in [("With target leakage", X_leaked),
#                      ("Without leakage",     X_clean)]:
#     X_tr, X_te, y_tr, y_te = train_test_split(X_use, y, test_size=0.3,
#                                                 random_state=1)
#     sc_ = StandardScaler().fit(X_tr)
#     clf_ = LogisticRegression(max_iter=500, random_state=0)
#     clf_.fit(sc_.transform(X_tr), y_tr)
#     acc = accuracy_score(y_te, clf_.predict(sc_.transform(X_te)))
#     results[name] = acc
#
# leak1_inflation = results["With target leakage"] - results["Without leakage"]
# print(f"  Without leakage:     {results['Without leakage']:.4f}")
# print(f"  With target leakage: {results['With target leakage']:.4f}")
# print(f"  Inflation:           +{leak1_inflation:.4f} ({leak1_inflation*100:.1f} percentage points)")
# print()
# print("  The leaky model APPEARS to be much better, but in production")
# print("  the 'account frozen' feature does not exist at prediction time")
# print("  (the account gets frozen AFTER fraud is detected, not before).")
# print()
#
# # ─────────────────────────────────────────────────────────────────────────
# # LEAKAGE TYPE 2: TRAIN-TEST CONTAMINATION (scaling before split)
# # ─────────────────────────────────────────────────────────────────────────
# print("  " + "─"*60)
# print("  LEAKAGE TYPE 2 — TRAIN-TEST CONTAMINATION")
# print("  " + "─"*60)
# print("  Scaling the FULL dataset before the train/test split leaks")
# print("  test set statistics (mean, std) into the training process.")
# print()
#
# X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
#     X_base, y, test_size=0.3, random_state=1)
#
# def run_cv_split_timing(X, y, scale_before_split, k=5):
#     """Run CV with scaling at the wrong time (before split) or correct time."""
#     if scale_before_split:
#         # WRONG: scale full data, then split
#         X_s = StandardScaler().fit_transform(X)
#         scores = cross_val_score(
#             LogisticRegression(max_iter=500, random_state=0),
#             X_s, y, cv=k, scoring="accuracy")
#     else:
#         # CORRECT: Pipeline ensures scaler is fit inside each fold
#         pipe = Pipeline([
#             ("sc",  StandardScaler()),
#             ("clf", LogisticRegression(max_iter=500, random_state=0))
#         ])
#         scores = cross_val_score(pipe, X, y, cv=k, scoring="accuracy")
#     return scores.mean(), scores.std()
#
# acc_leaked, std_leaked  = run_cv_split_timing(X_base, y, scale_before_split=True)
# acc_correct, std_correct = run_cv_split_timing(X_base, y, scale_before_split=False)
# leak2_inflation = acc_leaked - acc_correct
#
# print(f"  Scale before split (WRONG):  {acc_leaked:.4f} +/- {std_leaked:.4f}")
# print(f"  Pipeline (CORRECT):          {acc_correct:.4f} +/- {std_correct:.4f}")
# print(f"  Inflation:                   +{leak2_inflation:.4f}")
# print()
# print("  Note: for StandardScaler the effect is subtle (mean and std of")
# print("  the test set do not change dramatically). With small datasets,")
# print("  normalising based on strong outliers in the test set, or using")
# print("  the test label distribution in encoding, creates larger gaps.")
# print()
#
# # ─────────────────────────────────────────────────────────────────────────
# # LEAKAGE TYPE 3: FEATURE SELECTION BEFORE SPLIT
# # ─────────────────────────────────────────────────────────────────────────
# print("  " + "─"*60)
# print("  LEAKAGE TYPE 3 — FEATURE SELECTION BEFORE SPLIT")
# print("  " + "─"*60)
# print("  Using test set labels to select features BEFORE train/test split.")
# print()
#
# k_features = 5   # select top-5 features
#
# # WRONG: select features using ALL data (including test labels)
# selector_all = SelectKBest(f_classif, k=k_features).fit(X_base, y)
# X_selected_all = selector_all.transform(X_base)
# scores_leaked = cross_val_score(
#     LogisticRegression(max_iter=500, random_state=0),
#     X_selected_all, y, cv=5, scoring="accuracy")
#
# # CORRECT: feature selection inside Pipeline (fit on each train fold only)
# pipe_correct = Pipeline([
#     ("select", SelectKBest(f_classif, k=k_features)),
#     ("sc",     StandardScaler()),
#     ("clf",    LogisticRegression(max_iter=500, random_state=0)),
# ])
# scores_correct = cross_val_score(pipe_correct, X_base, y, cv=5,
#                                   scoring="accuracy")
#
# leak3_inflation = scores_leaked.mean() - scores_correct.mean()
#
# print(f"  Select ALL data then CV (WRONG):  {scores_leaked.mean():.4f} +/- {scores_leaked.std():.4f}")
# print(f"  Pipeline select inside CV (RIGHT): {scores_correct.mean():.4f} +/- {scores_correct.std():.4f}")
# print(f"  Inflation:                         +{leak3_inflation:.4f} ({leak3_inflation*100:.1f} pp)")
# print()
# print("  This leak is subtle but real: when selecting features based on")
# print("  all data, the selector has 'seen' the correlation between each")
# print("  feature and the test labels — information that should not exist.")
# print("  With many noisy features the inflation can be large (overfitting")
# print("  to noise that looks predictive because we peeked at test labels).")
# print()
#
# # ── Summary ───────────────────────────────────────────────────────────────
# print("  LEAKAGE SUMMARY:")
# print(f"  {'Leakage Type':32s} | {'Inflation':>10} | {'Risk Level':>12}")
# print(f"  {'─'*60}")
# for ltype, inflation in [
#     ("Target leakage",              leak1_inflation),
#     ("Train-test contamination",    leak2_inflation),
#     ("Feature selection before CV", leak3_inflation),
# ]:
#     risk = "HIGH" if inflation > 0.05 else ("MEDIUM" if inflation > 0.01 else "LOW")
#     print(f"  {ltype:32s} | {inflation:+10.4f} | {risk:>12}")
# print()
# print("  PREVENTION: Always use sklearn Pipeline.")
# print("  Pipeline ensures every preprocessing step is fitted ONLY on")
# print("  the training fold — never on validation or test data.")
# ''',
#     },
#
#     # ── 6 ─────────────────────────────────────────────────────────────────────
#     "6 · Complete Data Quality Pipeline — All Steps in Correct Order": {
#         "description": (
#             "Build a complete, correct data quality pipeline using sklearn Pipeline. "
#             "Handle missing data, scaling, encoding and class imbalance in one object. "
#             "Compare correct pipeline approach against a naive leaky approach."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score, recall_score
# from sklearn.datasets import make_classification
# import warnings
# warnings.filterwarnings("ignore")
#
# np.random.seed(42)
#
# print("=" * 65)
# print("  COMPLETE DATA QUALITY PIPELINE — CORRECT ORDER")
# print("=" * 65)
# print()
#
# # ── Simulate a messy dataset with 3 issues: missing, scaling, imbalance ───
# n = 1200
#
# # Generate base features
# X_num1 = np.random.normal(45, 15, n)          # age: ~18-90
# X_num2 = np.random.exponential(60000, n)       # salary: heavily skewed
# X_num3 = np.random.normal(650, 80, n)          # credit score: ~300-850
# X_cat  = np.random.choice(["A","B","C","D"], n, p=[0.4,0.3,0.2,0.1])
#
# # Introduce missing values
# X_num1_m = X_num1.copy()
# X_num2_m = X_num2.copy()
# X_num1_m[np.random.rand(n) < 0.10] = np.nan     # MCAR: 10% missing
# X_num2_m[X_num2 > np.percentile(X_num2, 80)] = np.nan  # MNAR: high earners
#
# # Target: imbalanced (10% positive)
# prob = 0.4*(X_num2 < 30000) + 0.3*(X_num3 < 580) + np.random.randn(n)*0.3
# y = (prob > np.percentile(prob, 90)).astype(int)
#
# # Assemble as numpy array (columns: age, salary, credit, cat_encoded)
# le = LabelEncoder().fit(X_cat)
# X_cat_enc = le.transform(X_cat).reshape(-1, 1).astype(float)
# X_raw = np.column_stack([X_num1_m, X_num2_m, X_num3, X_cat_enc])
#
# print(f"  Dataset: {n} samples, 4 features")
# print(f"  Positive rate: {y.mean()*100:.1f}%")
# print(f"  Missing in age:    {np.isnan(X_raw[:,0]).mean()*100:.1f}%")
# print(f"  Missing in salary: {np.isnan(X_raw[:,1]).mean()*100:.1f}%")
# print()
#
# # ── CORRECT approach: split first, then Pipeline ───────────────────────────
# print("  STEP 1: Split BEFORE any preprocessing")
# X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.25,
#                                             stratify=y, random_state=1)
#
# # Pipeline ensures every transformer is fitted on training data only
# correct_pipe = Pipeline([
#     ("impute", SimpleImputer(strategy="median")),   # fills NaN with train median
#     ("scale",  StandardScaler()),                   # centres using train mean/std
#     ("clf",    LogisticRegression(max_iter=1000,
#                                    class_weight="balanced",
#                                    random_state=0)),
# ])
#
# print("  STEP 2: Fit Pipeline on X_train ONLY")
# correct_pipe.fit(X_tr, y_tr)
#
# print("  STEP 3: Evaluate on X_test (pipeline uses train statistics)")
# y_pred_correct = correct_pipe.predict(X_te)
# f1_c  = f1_score(y_te, y_pred_correct, zero_division=0)
# rec_c = recall_score(y_te, y_pred_correct, zero_division=0)
# acc_c = accuracy_score(y_te, y_pred_correct)
# print()
#
# # ── NAIVE (WRONG) approach: preprocess on all data before split ────────────
# print("  NAIVE APPROACH (WRONG): preprocess ALL data, then split")
#
# # Fill NaN using global statistics (leaks test median to training)
# from sklearn.impute import SimpleImputer as SI2
# from sklearn.preprocessing import StandardScaler as SS2
#
# imp_naive = SI2(strategy="median").fit(X_raw)      # sees test data!
# X_imp_all = imp_naive.transform(X_raw)
# sc_naive  = SS2().fit(X_imp_all)                   # sees test data!
# X_scl_all = sc_naive.transform(X_imp_all)
#
# X_n_tr, X_n_te, yn_tr, yn_te = train_test_split(
#     X_scl_all, y, test_size=0.25, stratify=y, random_state=1)
#
# clf_naive = LogisticRegression(max_iter=1000, class_weight="balanced",
#                                 random_state=0).fit(X_n_tr, yn_tr)
# y_pred_naive = clf_naive.predict(X_n_te)
# f1_n  = f1_score(y_te, y_pred_naive, zero_division=0)
# rec_n = recall_score(y_te, y_pred_naive, zero_division=0)
# acc_n = accuracy_score(y_te, y_pred_naive)
#
# # ── Print comparison ──────────────────────────────────────────────────────
# print()
# print("  COMPARISON:")
# print(f"  {'Approach':30s} | {'F1':>8} | {'Recall':>8} | {'Accuracy':>10}")
# print(f"  {'─'*62}")
# print(f"  {'Correct Pipeline (no leakage)':30s} | {f1_c:8.4f} | {rec_c:8.4f} | {acc_c:10.4f}")
# print(f"  {'Naive (leaks test statistics)':30s} | {f1_n:8.4f} | {rec_n:8.4f} | {acc_n:10.4f}")
# print()
#
# # ── Show the Pipeline protects against leakage ────────────────────────────
# print("  WHAT PIPELINE.FIT() DOES INTERNALLY:")
# print()
# print("  correct_pipe.fit(X_train, y_train) calls:")
# print()
# # Show the imputer's learned values (from training only)
# imp  = correct_pipe.named_steps["impute"]
# scl  = correct_pipe.named_steps["scale"]
# print(f"    1. SimpleImputer fits on X_train:")
# print(f"       Train medians: age={imp.statistics_[0]:.2f}, "
#       f"salary={imp.statistics_[1]:.0f}, "
#       f"credit={imp.statistics_[2]:.0f}")
# print(f"    2. StandardScaler fits on imputed X_train:")
# print(f"       Train means:   age={scl.mean_[0]:.2f}, "
#       f"salary={scl.mean_[1]:.0f}")
# print(f"       Train stds:    age={scl.scale_[0]:.2f}, "
#       f"salary={scl.scale_[1]:.0f}")
# print()
# print("  When predict(X_test) is called:")
# print("    - Uses the SAME training medians for imputation")
# print("    - Uses the SAME training mean/std for scaling")
# print("    - Test data never influences these statistics ✓")
# print()
# print("  THE GOLDEN RULE:")
# print("  1. train_test_split() FIRST — before ANY transformation")
# print("  2. ALL preprocessing inside a Pipeline")
# print("  3. Pipeline.fit() only ever sees X_train")
# print("  4. Pipeline.predict(X_test) uses training-fit transformers")
#
# # ── Visual summary ────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(14, 5))
# fig.suptitle("Correct Data Quality Pipeline vs Naive Leaky Approach",
#              fontsize=12, fontweight="bold")
#
# # Plot 1: Missing data before/after imputation
# axes[0].bar(["With NaN (raw)", "After Pipeline Impute"],
#             [np.isnan(X_tr).mean()*100, 0.0],
#             color=["tomato", "seagreen"], alpha=0.8)
# axes[0].set_ylabel("% Missing values"); axes[0].set_ylim(0, 15)
# axes[0].set_title("Missing Values: Eliminated by Pipeline")
# axes[0].grid(alpha=0.3, axis="y")
#
# # Plot 2: Feature scale before/after
# X_tr_imp = imp.transform(X_tr)
# X_tr_scl = scl.transform(X_tr_imp)
# ax = axes[1]
# bp1 = ax.boxplot([X_tr[:, 0][~np.isnan(X_tr[:, 0])],
#                    X_tr[:, 1][~np.isnan(X_tr[:, 1])],
#                    X_tr[:, 2],
#                    X_tr[:, 3]],
#                   positions=[1,2,3,4], widths=0.35,
#                   patch_artist=True,
#                   boxprops=dict(facecolor="tomato", alpha=0.6))
# bp2 = ax.boxplot([X_tr_scl[:, 0], X_tr_scl[:, 1],
#                    X_tr_scl[:, 2], X_tr_scl[:, 3]],
#                   positions=[1.4,2.4,3.4,4.4], widths=0.35,
#                   patch_artist=True,
#                   boxprops=dict(facecolor="steelblue", alpha=0.6))
# ax.set_xticks([1.2, 2.2, 3.2, 4.2])
# ax.set_xticklabels(["age", "salary", "credit", "cat"], fontsize=9)
# ax.set_title("Feature Scale: Before (red) vs After (blue)")
# ax.set_ylabel("Value"); ax.set_ylim(-5, 5); ax.grid(alpha=0.3)
#
# # Plot 3: Performance comparison
# metrics = ["F1", "Recall", "Accuracy"]
# vals_c  = [f1_c, rec_c, acc_c]
# vals_n  = [f1_n, rec_n, acc_n]
# x_pos = np.arange(3)
# width = 0.35
# axes[2].bar(x_pos - width/2, vals_c, width, label="Correct Pipeline",
#             color="steelblue", alpha=0.8)
# axes[2].bar(x_pos + width/2, vals_n, width, label="Naive (leaky)",
#             color="tomato", alpha=0.8)
# axes[2].set_xticks(x_pos); axes[2].set_xticklabels(metrics)
# axes[2].set_ylim(0, 1.1); axes[2].set_ylabel("Score")
# axes[2].set_title("Model Performance Comparison")
# axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3, axis="y")
#
# plt.tight_layout()
# plt.savefig("data_quality_pipeline.png", dpi=120)
# print()
# print("  Pipeline diagram saved -> data_quality_pipeline.png")
# ''',
#     },
#     # ── 7 ─────────────────────────────────────────────────────────────────────
#     "7 · Outlier Detection & Treatment — IQR, Z-score, Isolation Forest, LOF": {
#         "description": (
#             "Compare four outlier detection methods on a realistic dataset. "
#             "Show how winsorizing, log-transform, and removal affect distributions "
#             "and downstream model performance. Demonstrate that outliers can be "
#             "signal, not noise."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score
#
# np.random.seed(42)
#
# print("=" * 65)
# print("  OUTLIER DETECTION & TREATMENT")
# print("=" * 65)
# print()
#
# # ── Generate dataset with three types of outliers ─────────────────────────
# n = 800
# X0 = np.random.normal(0, 1, n)
# X1 = np.random.normal(50, 10, n)
# X2 = np.random.exponential(2, n)
# y  = ((X0 + 0.5*X2) > np.percentile(X0 + 0.5*X2, 75)).astype(int)
#
# idx_err   = np.random.choice(n, 15, replace=False)
# idx_fraud = np.random.choice(n, 8,  replace=False)
# idx_multi = np.random.choice(n, 10, replace=False)
#
# X0[idx_err]   = np.random.uniform(50, 100, 15)
# X2[idx_fraud] = np.random.uniform(30, 60, 8)
# y[idx_fraud]  = 1
# X0[idx_multi] = np.random.normal(-3, 0.2, 10)
# X1[idx_multi] = np.random.normal(90, 0.5, 10)
#
# X = np.column_stack([X0, X1, X2])
#
# print(f"  Dataset: {n} samples, 3 features")
# print(f"  Injected: {len(idx_err)} sensor errors, {len(idx_fraud)} real fraud signals,")
# print(f"            {len(idx_multi)} multivariate outliers")
# print(f"  Positive rate: {y.mean()*100:.1f}%")
# print()
#
# # ── Method 1: IQR Fence ────────────────────────────────────────────────────
# print("  METHOD 1 — IQR FENCE (Tukey, univariate on feature 0)")
# print()
# Q1, Q3 = np.percentile(X0, 25), np.percentile(X0, 75)
# IQR    = Q3 - Q1
# lo,  hi  = Q1 - 1.5*IQR, Q3 + 1.5*IQR
# lo3, hi3 = Q1 - 3.0*IQR, Q3 + 3.0*IQR
#
# mask_iqr_mild   = (X0 < lo)  | (X0 > hi)
# mask_iqr_severe = (X0 < lo3) | (X0 > hi3)
#
# print(f"  Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
# print(f"  Severe fence: [{lo3:.3f}, {hi3:.3f}]")
# print(f"  Severe outliers flagged:      {mask_iqr_severe.sum()}")
# print(f"  Sensor errors captured:       {mask_iqr_severe[idx_err].sum()}/{len(idx_err)}")
# print(f"  Fraud signals flagged (bad!): {mask_iqr_severe[idx_fraud].sum()}/{len(idx_fraud)}")
# print()
#
# # ── Method 2: Modified Z-score ─────────────────────────────────────────────
# print("  METHOD 2 — MODIFIED Z-SCORE (robust, uses median/MAD)")
# print()
# med   = np.median(X0)
# MAD   = np.median(np.abs(X0 - med))
# z_mod = 0.6745 * (X0 - med) / (MAD + 1e-9)
# mask_zmod = np.abs(z_mod) > 3.5
#
# print(f"  Median={med:.3f}, MAD={MAD:.3f}")
# print(f"  Flagged with |z_modified|>3.5: {mask_zmod.sum()} points")
# print(f"  Sensor errors captured:        {mask_zmod[idx_err].sum()}/{len(idx_err)}")
# print(f"  Fraud signals flagged (bad!):  {mask_zmod[idx_fraud].sum()}/{len(idx_fraud)}")
# print()
#
# # ── Method 3: Isolation Forest ─────────────────────────────────────────────
# print("  METHOD 3 — ISOLATION FOREST (multivariate)")
# print()
# iso = IsolationForest(contamination=0.05, random_state=42)
# pred_iso = iso.fit_predict(X)
# mask_iso = pred_iso == -1
#
# print(f"  Flags {mask_iso.sum()} points ({mask_iso.mean()*100:.1f}%)")
# print(f"  Sensor errors captured:        {mask_iso[idx_err].sum()}/{len(idx_err)}")
# print(f"  Multivariate outliers found:   {mask_iso[idx_multi].sum()}/{len(idx_multi)}")
# print(f"  Fraud signals flagged (bad!):  {mask_iso[idx_fraud].sum()}/{len(idx_fraud)}")
# print()
#
# # ── Method 4: LOF ──────────────────────────────────────────────────────────
# print("  METHOD 4 — LOCAL OUTLIER FACTOR")
# print()
# X_sc_lof = StandardScaler().fit_transform(X)
# lof      = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
# pred_lof = lof.fit_predict(X_sc_lof)
# mask_lof = pred_lof == -1
#
# print(f"  Flags {mask_lof.sum()} points")
# print(f"  Sensor errors captured:        {mask_lof[idx_err].sum()}/{len(idx_err)}")
# print(f"  Multivariate outliers found:   {mask_lof[idx_multi].sum()}/{len(idx_multi)}")
# print(f"  Fraud signals flagged (bad!):  {mask_lof[idx_fraud].sum()}/{len(idx_fraud)}")
# print()
#
# # ── Treatment comparison ───────────────────────────────────────────────────
# print("  TREATMENT COMPARISON (downstream model F1):")
# print()
# print(f"  {'Treatment':34s} | {'CV F1 (mean)':>13} | {'CV F1 (std)':>12} | {'n_train':>8}")
# print(f"  {'─'*75}")
#
# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
#                                             stratify=y, random_state=1)
#
# def eval_treatment(name, Xtr, ytr):
#     scaler = StandardScaler().fit(Xtr)
#     Xs     = scaler.transform(Xtr)
#     clf    = LogisticRegression(class_weight="balanced", max_iter=500, random_state=0)
#     scores = cross_val_score(clf, Xs, ytr, cv=5, scoring="f1")
#     print(f"  {name:34s} | {scores.mean():13.4f} | {scores.std():12.4f} | {len(ytr):8d}")
#
# eval_treatment("No treatment (baseline)", X_tr, y_tr)
#
# Q1_tr = np.percentile(X_tr[:,0], 25); Q3_tr = np.percentile(X_tr[:,0], 75)
# IQR_tr = Q3_tr - Q1_tr
# keep = (X_tr[:,0] >= Q1_tr - 3*IQR_tr) & (X_tr[:,0] <= Q3_tr + 3*IQR_tr)
# eval_treatment("Remove IQR severe outliers", X_tr[keep], y_tr[keep])
#
# X_tr_w = X_tr.copy()
# for col in range(3):
#     lo_w = np.percentile(X_tr[:,col], 1)
#     hi_w = np.percentile(X_tr[:,col], 99)
#     X_tr_w[:,col] = np.clip(X_tr[:,col], lo_w, hi_w)
# eval_treatment("Winsorize at 1st-99th pct", X_tr_w, y_tr)
#
# X_tr_log = X_tr.copy()
# X_tr_log[:,2] = np.log1p(np.abs(X_tr[:,2]))
# eval_treatment("Log-transform skewed feature", X_tr_log, y_tr)
#
# iso_tr = IsolationForest(contamination=0.05, random_state=42).fit(X_tr)
# keep_iso = iso_tr.predict(X_tr) == 1
# eval_treatment("Remove Isolation Forest outliers", X_tr[keep_iso], y_tr[keep_iso])
#
# print()
# print("  KEY LESSON: Winsorizing and log-transform usually beat removal.")
# print("  Removing outliers risks discarding fraud signals (minority class).")
#
# # ── Plots ─────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# fig.suptitle("Outlier Detection: Methods & Treatment Comparison",
#              fontsize=12, fontweight="bold")
#
# ax = axes[0]
# ax.scatter(range(n), np.sort(X0), s=4, c="steelblue", alpha=0.5, label="Normal")
# sorted_idx  = np.argsort(X0)
# err_sorted   = np.isin(sorted_idx, idx_err)
# fraud_sorted = np.isin(sorted_idx, idx_fraud)
# ax.scatter(np.where(err_sorted)[0],   np.sort(X0)[err_sorted],
#            s=50, c="tomato",  zorder=5, label="Sensor error")
# ax.scatter(np.where(fraud_sorted)[0], np.sort(X0)[fraud_sorted],
#            s=50, c="seagreen", zorder=5, marker="^", label="Real fraud (keep!)")
# ax.axhline(hi3, color="tomato", linestyle="--", lw=1.5, label=f"Severe fence")
# ax.axhline(lo3, color="tomato", linestyle="--", lw=1.5)
# ax.set_title("IQR Fence — Feature 0 (sorted)")
# ax.set_xlabel("Rank"); ax.set_ylabel("Value")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)
#
# ax2 = axes[1]
# colours_iso = np.where(mask_iso, "tomato", "steelblue")
# ax2.scatter(X[:,0], X[:,2], c=colours_iso, s=8, alpha=0.5)
# ax2.scatter(X[idx_err,0], X[idx_err,2], s=100, c="darkred",
#             marker="x", linewidths=2, label="Sensor error", zorder=5)
# ax2.scatter(X[idx_fraud,0], X[idx_fraud,2], s=100, c="seagreen",
#             marker="^", zorder=5, label="Real fraud (keep!)")
# ax2.set_title("Isolation Forest Flags\n(red = flagged as outlier)")
# ax2.set_xlabel("Feature 0"); ax2.set_ylabel("Feature 2")
# ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
#
# ax3 = axes[2]
# ax3.hist(X[:,2], bins=50, alpha=0.6, color="tomato", density=True, label="Raw (skewed)")
# ax3.hist(np.log1p(X[:,2]), bins=50, alpha=0.6, color="steelblue",
#          density=True, label="log1p transformed")
# ax3.set_title("Log-Transform: Compresses Skew\nWithout Discarding Data")
# ax3.set_xlabel("Value"); ax3.set_ylabel("Density")
# ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("outlier_detection.png", dpi=120)
# print()
# print("  Plot saved -> outlier_detection.png")
# ''',
#     },
#
#     # ── 8 ─────────────────────────────────────────────────────────────────────
#     "8 · Categorical Encoding — OHE vs Target vs Frequency & Leakage Trap": {
#         "description": (
#             "Compare label, one-hot, frequency, and target encoding on a "
#             "high-cardinality categorical feature. Demonstrate the target "
#             "encoding leakage trap: naive encoding vs correct cross-fold encoding. "
#             "Show how cardinality affects memory and model performance."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")
#
# np.random.seed(42)
#
# print("=" * 65)
# print("  CATEGORICAL ENCODING: STRATEGIES & LEAKAGE TRAPS")
# print("=" * 65)
# print()
#
# # ── Dataset with high-cardinality city feature ────────────────────────────
# n, n_cities = 2000, 80
# city_ids    = np.random.randint(0, n_cities, n)
# city_names  = [f"City_{i:02d}" for i in range(n_cities)]
# city_base   = np.random.beta(2, 5, n_cities)
#
# X_num = np.random.normal(0, 1, n)
# true_probs = np.clip(0.6*city_base[city_ids] + 0.4*(X_num > 0.5), 0.02, 0.98)
# y      = (np.random.rand(n) < true_probs).astype(int)
# cities = np.array([city_names[i] for i in city_ids])
#
# print(f"  Dataset: {n} samples, 1 numeric + 1 categorical ({n_cities} cities)")
# print(f"  Positive rate: {y.mean()*100:.1f}%")
# print()
#
# # ── Show the ordinal fallacy ───────────────────────────────────────────────
# print("  DANGER: LABEL ENCODING IMPLIES FALSE ORDINAL RELATIONSHIP")
# print()
# le = LabelEncoder().fit(cities)
# print(f"  {'City':10s} | {'Label code':>11} | {'Mean(target)':>13}")
# print(f"  {'─'*40}")
# for c in sorted(city_names[:8]):
#     code     = le.transform([c])[0]
#     mean_y   = y[cities == c].mean()
#     print(f"  {c:10s} | {code:11d} | {mean_y:13.4f}")
# print("  No correlation between code and target — ordering is meaningless!")
# print()
#
# # ── Compare encoding strategies ───────────────────────────────────────────
# print("  ENCODING STRATEGY COMPARISON:")
# print()
# X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
#     np.column_stack([cities, X_num.astype(str)]),
#     y, test_size=0.3, random_state=1)
#
# cities_tr = X_tr_raw[:, 0]
# cities_te = X_te_raw[:, 0]
# num_tr    = X_tr_raw[:, 1].astype(float)
# num_te    = X_te_raw[:, 1].astype(float)
# global_mean = y_tr.mean()
# m_smooth    = 20
#
# def train_eval(name, Xtr, Xte, ytr, yte):
#     sc  = StandardScaler().fit(Xtr)
#     clf = LogisticRegression(max_iter=500, random_state=0)
#     clf.fit(sc.transform(Xtr), ytr)
#     pred = clf.predict(sc.transform(Xte))
#     return accuracy_score(yte, pred), f1_score(yte, pred, zero_division=0), Xtr.shape[1]
#
# results = []
#
# # Numeric only
# acc, f1, nc = train_eval("Numeric only (baseline)",
#     num_tr.reshape(-1,1), num_te.reshape(-1,1), y_tr, y_te)
# results.append(("Numeric only (baseline)", acc, f1, nc))
#
# # Label encoding
# le2 = LabelEncoder().fit(cities_tr)
# safe_te = np.array([c if c in le2.classes_ else le2.classes_[0] for c in cities_te])
# acc, f1, nc = train_eval("Label encoding (WRONG ordinal)",
#     np.column_stack([le2.transform(cities_tr), num_tr]),
#     np.column_stack([le2.transform(safe_te), num_te]), y_tr, y_te)
# results.append(("Label encoding (WRONG ordinal)", acc, f1, nc))
#
# # One-hot
# ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cities_tr.reshape(-1,1))
# acc, f1, nc = train_eval("One-hot encoding",
#     np.column_stack([ohe.transform(cities_tr.reshape(-1,1)), num_tr]),
#     np.column_stack([ohe.transform(cities_te.reshape(-1,1)), num_te]), y_tr, y_te)
# results.append(("One-hot encoding", acc, f1, nc))
#
# # Frequency
# freq_map  = {c: (cities_tr == c).mean() for c in np.unique(cities_tr)}
# def freq_enc(cats): return np.array([freq_map.get(c, 1/n_cities) for c in cats])
# acc, f1, nc = train_eval("Frequency encoding",
#     np.column_stack([freq_enc(cities_tr), num_tr]),
#     np.column_stack([freq_enc(cities_te), num_te]), y_tr, y_te)
# results.append(("Frequency encoding", acc, f1, nc))
#
# # Naive target encoding (LEAKY)
# naive_map = {c: y[cities == c].mean() for c in np.unique(cities)}
# def naive_enc(cats): return np.array([naive_map.get(c, global_mean) for c in cats])
# acc, f1, nc = train_eval("Target enc. NAIVE (LEAKY)",
#     np.column_stack([naive_enc(cities_tr), num_tr]),
#     np.column_stack([naive_enc(cities_te), num_te]), y_tr, y_te)
# results.append(("Target enc. NAIVE (LEAKY)", acc, f1, nc))
#
# # Correct OOF target encoding
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# cats_tr_enc = np.zeros(len(cities_tr))
# for fold_tr_idx, fold_val_idx in kf.split(cities_tr):
#     for c in np.unique(cities_tr[fold_val_idx]):
#         fm = cities_tr[fold_tr_idx] == c
#         cnt = fm.sum()
#         cm  = y_tr[fold_tr_idx][fm].mean() if cnt > 0 else global_mean
#         sm  = (cnt*cm + m_smooth*global_mean) / (cnt + m_smooth)
#         cats_tr_enc[fold_val_idx[cities_tr[fold_val_idx] == c]] = sm
#
# correct_map = {}
# for c in np.unique(cities_tr):
#     fm = cities_tr == c; cnt = fm.sum()
#     correct_map[c] = (cnt*y_tr[fm].mean() + m_smooth*global_mean)/(cnt+m_smooth)
#
# def correct_enc_te(cats): return np.array([correct_map.get(c, global_mean) for c in cats])
# acc, f1, nc = train_eval("Target enc. CORRECT (OOF)",
#     np.column_stack([cats_tr_enc, num_tr]),
#     np.column_stack([correct_enc_te(cities_te), num_te]), y_tr, y_te)
# results.append(("Target enc. CORRECT (OOF)", acc, f1, nc))
#
# print(f"  {'Method':35s} | {'Accuracy':>10} | {'F1':>8} | {'n_cols':>8}")
# print(f"  {'─'*68}")
# for name, acc, f1, nc in results:
#     flag = " <- LEAKY" if "LEAKY" in name else (" <- WRONG" if "WRONG" in name else "")
#     print(f"  {name:35s} | {acc:10.4f} | {f1:8.4f} | {nc:8d}{flag}")
#
# print()
# print("  Naive target encoding scores highest but is LEAKY (invalid).")
# print("  Correct OOF target encoding: nearly as good, no leakage.")
# print("  One-hot: n_cols explodes with cardinality (80 cities = 80 cols).")
#
# # ── Plots ─────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# fig.suptitle("Categorical Encoding: Strategies, Cardinality & Leakage",
#              fontsize=12, fontweight="bold")
#
# cards = [5, 10, 20, 50, 100, 200, 500]
# axes[0].plot(cards, cards, "tomato", lw=2.5, label="One-hot (k cols)")
# axes[0].plot(cards, [1]*len(cards), "steelblue", lw=2.5, label="Label/Freq/Target (1 col)")
# axes[0].plot(cards, [min(32, k) for k in cards], "seagreen", lw=2.5,
#              linestyle="--", label="Hashing (<=32 cols)")
# axes[0].set_xlabel("Cardinality (# unique values)")
# axes[0].set_ylabel("Columns created")
# axes[0].set_title("Memory Cost vs Cardinality")
# axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
#
# f1_vals = [r[2] for r in results]
# cols_enc = ["gray","tomato","steelblue","seagreen","red","purple"]
# brs = axes[1].barh(range(len(results)), f1_vals, color=cols_enc, alpha=0.8)
# axes[1].set_yticks(range(len(results)))
# axes[1].set_yticklabels([r[0] for r in results], fontsize=8)
# axes[1].set_xlabel("F1 Score")
# axes[1].set_title("F1 Score by Encoding Method")
# axes[1].grid(alpha=0.3, axis="x")
# for bar, val in zip(brs, f1_vals):
#     axes[1].text(val+0.002, bar.get_y()+bar.get_height()/2,
#                  f"{val:.4f}", va="center", fontsize=8)
#
# naive_vals   = [naive_map.get(c, global_mean) for c in cities_te]
# correct_vals = [correct_map.get(c, global_mean) for c in cities_te]
# axes[2].scatter(correct_vals, naive_vals, alpha=0.3, s=10, c="steelblue")
# axes[2].plot([0,1],[0,1],"tomato",linestyle="--",lw=2,label="y=x (identical)")
# axes[2].set_xlabel("Correct OOF encoding"); axes[2].set_ylabel("Naive encoding (LEAKY)")
# axes[2].set_title("Naive vs Correct Target Encoding\n(Naive is optimistically biased)")
# axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("categorical_encoding.png", dpi=120)
# print()
# print("  Plot saved -> categorical_encoding.png")
# ''',
#     },
#
#     # ── 9 ─────────────────────────────────────────────────────────────────────
#     "9 · Distribution Shift & Data Drift — PSI, KS Test & Concept Drift": {
#         "description": (
#             "Simulate all three types of drift: covariate, label, and concept drift. "
#             "Detect covariate shift using PSI and KS test per feature. "
#             "Show how a model silently degrades under each shift type. "
#             "Demonstrate importance weighting as a fix for covariate shift."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from scipy import stats
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.preprocessing import StandardScaler
#
# np.random.seed(42)
#
# print("=" * 65)
# print("  DISTRIBUTION SHIFT & DATA DRIFT")
# print("=" * 65)
# print()
#
# def psi(expected, actual, n_bins=10):
#     bps = np.linspace(min(expected.min(), actual.min()),
#                       max(expected.max(), actual.max()), n_bins + 1)
#     exp_p = np.histogram(expected, bins=bps)[0].astype(float) + 1e-6
#     act_p = np.histogram(actual,   bins=bps)[0].astype(float) + 1e-6
#     exp_p /= exp_p.sum(); act_p /= act_p.sum()
#     return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))
#
# # ── Training data ──────────────────────────────────────────────────────────
# n_train = 2000
# X_tr = np.column_stack([
#     np.random.normal(0, 1, n_train),
#     np.random.normal(0, 1, n_train),
#     np.random.normal(0, 1, n_train),
# ])
# feat_names = ["Age", "Income", "Credit"]
#
# def sigmoid(z): return 1 / (1 + np.exp(-z))
# def true_prob(X): return sigmoid(0.8*X[:,0] + 0.5*X[:,1] - 0.3*X[:,2])
# def true_prob_drift(X): return sigmoid(0.8*X[:,0] - 0.9*X[:,1] - 0.3*X[:,2])
#
# y_tr = (np.random.rand(n_train) < true_prob(X_tr)).astype(int)
# scaler = StandardScaler().fit(X_tr)
# clf    = LogisticRegression(max_iter=500, random_state=0)
# clf.fit(scaler.transform(X_tr), y_tr)
#
# n_test = 1000
# print(f"  Training: n={n_train}, positive rate={y_tr.mean()*100:.1f}%")
# print()
#
# # ── Four scenarios ─────────────────────────────────────────────────────────
# def gen(age_mu=0, age_s=1, income_mu=0, income_s=1, credit_mu=0, credit_s=1,
#         prob_fn=None, label_scale=1.0):
#     X = np.column_stack([np.random.normal(age_mu, age_s, n_test),
#                          np.random.normal(income_mu, income_s, n_test),
#                          np.random.normal(credit_mu, credit_s, n_test)])
#     fn = prob_fn if prob_fn else true_prob
#     p  = np.clip(fn(X) * label_scale, 0.01, 0.99)
#     y  = (np.random.rand(n_test) < p).astype(int)
#     return X, y
#
# scenarios = [
#     ("No shift (baseline)",     gen()),
#     ("Covariate shift (age)",   gen(age_mu=1.5, age_s=0.8)),
#     ("Label shift (3x fraud)",  gen(prob_fn=lambda X: np.clip(true_prob(X)*3, 0, 1))),
#     ("Concept drift",           gen(prob_fn=true_prob_drift)),
# ]
#
# # ── Performance table ─────────────────────────────────────────────────────
# print("  MODEL PERFORMANCE UNDER DIFFERENT SHIFT SCENARIOS:")
# print()
# print(f"  {'Scenario':30s} | {'AUC':>7} | {'F1':>7} | {'Pos rate':>10} | {'Degradation'}")
# print(f"  {'─'*72}")
#
# baseline_auc = None
# auc_vals_plot = []
# for name, (X_sc, y_sc) in scenarios:
#     probs = clf.predict_proba(scaler.transform(X_sc))[:,1]
#     preds = (probs > 0.5).astype(int)
#     auc   = roc_auc_score(y_sc, probs) if y_sc.sum() > 0 and y_sc.sum() < len(y_sc) else 0.5
#     f1    = f1_score(y_sc, preds, zero_division=0)
#     auc_vals_plot.append(auc)
#     if baseline_auc is None:
#         baseline_auc = auc; degrade = "baseline"
#     else:
#         degrade = f"-{(baseline_auc-auc)*100:.1f} pts AUC"
#     print(f"  {name:30s} | {auc:7.4f} | {f1:7.4f} | {y_sc.mean()*100:9.1f}% | {degrade}")
#
# print()
#
# # ── PSI + KS detection ────────────────────────────────────────────────────
# print("  DRIFT DETECTION: PSI AND KS TEST")
# print()
# print("  PSI < 0.10: ok  |  0.10-0.20: investigate  |  > 0.20: retrain")
# print()
# print(f"  {'Scenario':26s} | {'Feature':8s} | {'PSI':>7} | {'KS p-val':>10} | {'Alarm?'}")
# print(f"  {'─'*68}")
#
# detect_scenarios = [
#     ("Covariate (age shifts)",  scenarios[1][1][0]),
#     ("Label shift",             scenarios[2][1][0]),
#     ("Concept drift",           scenarios[3][1][0]),
# ]
# for sname, X_new in detect_scenarios:
#     for fi, fname in enumerate(feat_names):
#         psi_val    = psi(X_tr[:,fi], X_new[:,fi])
#         _, ks_p    = stats.ks_2samp(X_tr[:,fi], X_new[:,fi])
#         alarm = "ALARM" if psi_val > 0.2 or ks_p < 0.01 else ("WARN" if psi_val > 0.1 else "ok")
#         print(f"  {sname:26s} | {fname:8s} | {psi_val:7.4f} | {ks_p:10.4f} | {alarm}")
#     print()
#
# # ── Importance weighting fix ──────────────────────────────────────────────
# print("  FIX: IMPORTANCE WEIGHTING FOR COVARIATE SHIFT")
# print()
# X_cov, y_cov = scenarios[1][1]
# X_domain = np.vstack([X_tr, X_cov])
# y_domain = np.array([0]*n_train + [1]*n_test)
# scaler_d = StandardScaler().fit(X_domain)
# clf_d    = LogisticRegression(max_iter=500, random_state=0)
# clf_d.fit(scaler_d.transform(X_domain), y_domain)
#
# tr_p  = clf_d.predict_proba(scaler_d.transform(X_tr))[:,1]
# w     = np.clip((tr_p / (1 - tr_p + 1e-8)) / ((tr_p / (1 - tr_p + 1e-8)).mean()), 0.1, 10)
#
# clf_iw = LogisticRegression(max_iter=500, random_state=0)
# clf_iw.fit(scaler.transform(X_tr), y_tr, sample_weight=w)
#
# auc_orig = roc_auc_score(y_cov, clf.predict_proba(scaler.transform(X_cov))[:,1])
# auc_iw   = roc_auc_score(y_cov, clf_iw.predict_proba(scaler.transform(X_cov))[:,1])
# print(f"  Covariate shift — Original AUC: {auc_orig:.4f}")
# print(f"  Covariate shift — IW AUC:       {auc_iw:.4f}  ({(auc_iw-auc_orig)*100:+.2f} pts)")
# print()
#
# # ── Plots ─────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# fig.suptitle("Distribution Shift: Detection, Impact & Importance Weighting",
#              fontsize=12, fontweight="bold")
#
# axes[0].hist(X_tr[:,0], bins=40, alpha=0.5, color="steelblue",
#              density=True, label="Training (age)")
# axes[0].hist(X_cov[:,0], bins=40, alpha=0.5, color="tomato",
#              density=True, label="Deployment (shifted)")
# psi_age = psi(X_tr[:,0], X_cov[:,0])
# axes[0].set_title(f"Covariate Shift: Age Distribution\nPSI={psi_age:.3f} (>0.20 = major)")
# axes[0].set_xlabel("Age (standardised)"); axes[0].set_ylabel("Density")
# axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
#
# snames = ["No shift","Covariate\nshift","Label\nshift","Concept\ndrift"]
# cols_d = ["seagreen","orange","tomato","red"]
# bars = axes[1].bar(snames, auc_vals_plot, color=cols_d, alpha=0.8)
# axes[1].axhline(0.5, color="gray", linestyle="--", lw=1.5, label="Random")
# axes[1].set_ylim(0.4, 1.0); axes[1].set_ylabel("AUC")
# axes[1].set_title("Model AUC Under Different Shift Types")
# axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")
# for bar, val in zip(bars, auc_vals_plot):
#     axes[1].text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
#                  ha="center", fontsize=9)
#
# axes[2].hist(w, bins=40, color="purple", alpha=0.8)
# axes[2].axvline(1.0, color="tomato", linestyle="--", lw=2, label="w=1 (no reweight)")
# axes[2].set_xlabel("Importance weight w(x)")
# axes[2].set_ylabel("Count")
# axes[2].set_title(f"Importance Weights for Covariate Shift\nAUC {auc_orig:.3f} -> {auc_iw:.3f}")
# axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("distribution_shift.png", dpi=120)
# print("  Plot saved -> distribution_shift.png")
# print()
# print("  KEY TAKEAWAYS:")
# print("  PSI > 0.20 on any feature: investigate / retrain")
# print("  Covariate shift: importance weighting can partially fix it")
# print("  Label shift: recalibrate decision threshold")
# print("  Concept drift: MUST retrain — no weighting can fix it")
# ''',
#     },
#
#     # ── 10 ────────────────────────────────────────────────────────────────────
#     "10 · Fix Ordering & Interactions — SMOTE, Target Encoding & Threshold Bugs": {
#         "description": (
#             "Demonstrate the three most dangerous ordering mistakes: "
#             "SMOTE before split, naive target encoding, and class weights "
#             "without threshold re-tuning. Quantify the score inflation each "
#             "produces and show the correct pipeline for each."
#         ),
#         "language": "python",
#         "code": '''
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")
#
# np.random.seed(42)
#
# print("=" * 65)
# print("  FIX ORDERING & INTERACTIONS: THREE PIPELINE BUGS")
# print("=" * 65)
# print()
#
# n = 3000
# X_num, y = make_classification(
#     n_samples=n, n_features=8, n_informative=5,
#     weights=[0.92, 0.08], random_state=42)
#
# n_cats   = 60
# cat_ids  = np.random.randint(0, n_cats, n)
# cat_str  = np.array([f"C{i:03d}" for i in cat_ids])
# pos_rate = y.mean()
#
# print(f"  Dataset: {n} samples, imbalance {(1-pos_rate)/pos_rate:.0f}:1")
# print(f"  Positive rate: {pos_rate*100:.1f}%")
# print()
#
# # ── BUG 1: SMOTE BEFORE SPLIT ─────────────────────────────────────────────
# print("  BUG 1 — SMOTE BEFORE SPLIT")
# print()
#
# try:
#     from imblearn.over_sampling import SMOTE
#     sm = SMOTE(random_state=42)
#
#     # WRONG: resample ALL data, then split
#     X_res_w, y_res_w = sm.fit_resample(X_num, y)
#     Xtr_w, Xte_w, ytr_w, yte_w = train_test_split(
#         X_res_w, y_res_w, test_size=0.3, random_state=1)
#     sc_w = StandardScaler().fit(Xtr_w)
#     clf_w = LogisticRegression(max_iter=500, random_state=0)
#     clf_w.fit(sc_w.transform(Xtr_w), ytr_w)
#     p_w = clf_w.predict_proba(sc_w.transform(Xte_w))[:,1]
#     f1_wrong  = f1_score(yte_w, (p_w > 0.5).astype(int), zero_division=0)
#     auc_wrong = roc_auc_score(yte_w, p_w)
#
#     # CORRECT: split first, then SMOTE only on training set
#     Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
#         X_num, y, test_size=0.3, stratify=y, random_state=1)
#     Xtr_res, ytr_res = sm.fit_resample(Xtr_c, ytr_c)
#     sc_c = StandardScaler().fit(Xtr_res)
#     clf_c = LogisticRegression(max_iter=500, random_state=0)
#     clf_c.fit(sc_c.transform(Xtr_res), ytr_res)
#     p_c = clf_c.predict_proba(sc_c.transform(Xte_c))[:,1]
#     f1_correct  = f1_score(yte_c, (p_c > 0.5).astype(int), zero_division=0)
#     auc_correct = roc_auc_score(yte_c, p_c)
#     smote_inflation = (f1_wrong - f1_correct) * 100
#
#     print(f"  {'Approach':38s} | {'F1':>8} | {'AUC':>8} | Note")
#     print(f"  {'─'*72}")
#     print(f"  {'WRONG: SMOTE before split':38s} | {f1_wrong:8.4f} | {auc_wrong:8.4f} | "
#           "test contaminated")
#     print(f"  {'CORRECT: split first, SMOTE train only':38s} | {f1_correct:8.4f} | "
#           f"{auc_correct:8.4f} | test untouched")
#     print(f"  Score inflation: {smote_inflation:+.2f} F1 pts")
#     smote_available = True
# except ImportError:
#     print("  (imbalanced-learn not installed)")
#     print("  Rule: ALWAYS split first, then apply SMOTE to train fold only.")
#     f1_wrong = f1_correct = smote_inflation = 0.0
#     smote_available = False
# print()
#
# # ── BUG 2: TARGET ENCODING WITHOUT OOF PROTECTION ─────────────────────────
# print("  BUG 2 — NAIVE TARGET ENCODING (LEAKS TEST LABELS)")
# print()
#
# Xtr2, Xte2, ytr2, yte2 = train_test_split(
#     X_num, y, test_size=0.3, stratify=y, random_state=1)
# ctr2 = cat_str[:len(Xtr2)]
# cte2 = cat_str[len(Xtr2):len(Xtr2)+len(Xte2)]
# gm2  = ytr2.mean(); ms2 = 20
#
# # WRONG: use full dataset y to compute means
# naive_map2 = {c: y[cat_str == c].mean() for c in np.unique(cat_str)}
#
# # CORRECT: out-of-fold
# kf2 = KFold(n_splits=5, shuffle=True, random_state=0)
# enc_tr2 = np.zeros(len(ctr2))
# for fi, vi in kf2.split(ctr2):
#     for c in np.unique(ctr2[vi]):
#         fm = ctr2[fi] == c; cnt = fm.sum()
#         cm = ytr2[fi][fm].mean() if cnt > 0 else gm2
#         enc_tr2[vi[ctr2[vi] == c]] = (cnt*cm + ms2*gm2)/(cnt+ms2)
#
# correct_map2 = {}
# for c in np.unique(ctr2):
#     fm = ctr2 == c; cnt = fm.sum()
#     correct_map2[c] = (cnt*ytr2[fm].mean() + ms2*gm2)/(cnt+ms2)
#
# def te(cats, mp, gm): return np.array([mp.get(c, gm) for c in cats])
#
# def eval_te(Xtr_enc, Xte_enc, ytr, yte):
#     sc  = StandardScaler().fit(Xtr_enc)
#     clf = LogisticRegression(max_iter=500, random_state=0, class_weight="balanced")
#     clf.fit(sc.transform(Xtr_enc), ytr)
#     p   = clf.predict_proba(sc.transform(Xte_enc))[:,1]
#     return f1_score(yte, (p > 0.5).astype(int), zero_division=0), roc_auc_score(yte, p)
#
# Xtr_naive2 = np.column_stack([te(ctr2, naive_map2, gm2), Xtr2])
# Xte_naive2 = np.column_stack([te(cte2, naive_map2, gm2), Xte2])
# f1_naive2, auc_naive2 = eval_te(Xtr_naive2, Xte_naive2, ytr2, yte2)
#
# Xtr_corr2 = np.column_stack([enc_tr2, Xtr2])
# Xte_corr2 = np.column_stack([te(cte2, correct_map2, gm2), Xte2])
# f1_corr2, auc_corr2 = eval_te(Xtr_corr2, Xte_corr2, ytr2, yte2)
# te_inflation = (f1_naive2 - f1_corr2) * 100
#
# print(f"  {'Approach':40s} | {'F1':>8} | {'AUC':>8}")
# print(f"  {'─'*60}")
# print(f"  {'WRONG: target enc on full data (LEAKY)':40s} | {f1_naive2:8.4f} | {auc_naive2:8.4f}")
# print(f"  {'CORRECT: OOF target encoding':40s} | {f1_corr2:8.4f} | {auc_corr2:8.4f}")
# print(f"  Score inflation: {te_inflation:+.2f} F1 pts")
# print()
#
# # ── BUG 3: CLASS WEIGHTS WITHOUT THRESHOLD RETUNING ───────────────────────
# print("  BUG 3 — CLASS WEIGHTS SHIFT CALIBRATION: MUST RETUNE THRESHOLD")
# print()
#
# Xtr3, Xte3, ytr3, yte3 = train_test_split(
#     X_num, y, test_size=0.3, stratify=y, random_state=1)
# Xval3, Xte3b = Xte3[:len(Xte3)//2], Xte3[len(Xte3)//2:]
# yval3, yte3b = yte3[:len(yte3)//2], yte3[len(yte3)//2:]
#
# sc3  = StandardScaler().fit(Xtr3)
# clf3 = LogisticRegression(max_iter=500, random_state=0, class_weight="balanced")
# clf3.fit(sc3.transform(Xtr3), ytr3)
#
# val_probs3 = clf3.predict_proba(sc3.transform(Xval3))[:,1]
# te_probs3  = clf3.predict_proba(sc3.transform(Xte3b))[:,1]
#
# f1_def = f1_score(yte3b, (te_probs3 > 0.5).astype(int), zero_division=0)
# thrs   = np.linspace(0.01, 0.99, 200)
# val_f1s = [f1_score(yval3, (val_probs3 > t).astype(int), zero_division=0) for t in thrs]
# best_thr = thrs[np.argmax(val_f1s)]
# f1_tuned = f1_score(yte3b, (te_probs3 > best_thr).astype(int), zero_division=0)
# approx_thr = 1 / (1 + (1 - pos_rate)/pos_rate)
#
# print(f"  Imbalance ratio: {(1-pos_rate)/pos_rate:.0f}:1  "
#       f"=> approx optimal threshold ~ {approx_thr:.3f}")
# print(f"  Best threshold found on validation: {best_thr:.3f}")
# print()
# print(f"  {'Approach':35s} | {'F1':>8} | {'Threshold':>10}")
# print(f"  {'─'*58}")
# print(f"  {'Default threshold=0.5':35s} | {f1_def:8.4f} | {'0.500':>10}")
# print(f"  {'Tuned on validation set':35s} | {f1_tuned:8.4f} | {best_thr:10.3f}")
# print(f"  Gain from threshold tuning: {(f1_tuned - f1_def)*100:+.2f} F1 pts")
# print()
# print("  RULE: After class_weight='balanced', ALWAYS tune threshold.")
# print("  Never assume 0.5 is optimal for imbalanced problems.")
#
# # ── Plots ─────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# fig.suptitle("Three Common Pipeline Bugs and Their Score Inflation",
#              fontsize=12, fontweight="bold")
#
# if smote_available:
#     axes[0].barh(["SMOTE before split\n(WRONG)","Split first, SMOTE train\n(CORRECT)"],
#                  [f1_wrong, f1_correct], color=["tomato","seagreen"], alpha=0.8)
#     axes[0].set_xlabel("F1 Score")
#     axes[0].set_title(f"Bug 1: SMOTE Before Split\nInflation = {smote_inflation:+.1f} F1 pts")
#     for i, val in enumerate([f1_wrong, f1_correct]):
#         axes[0].text(val+0.002, i, f"{val:.4f}", va="center", fontsize=10)
# else:
#     axes[0].text(0.5, 0.5, "imbalanced-learn\nnot installed\n\nRule: split FIRST\nthen SMOTE train only",
#                  ha="center", va="center", transform=axes[0].transAxes, fontsize=11)
#     axes[0].set_title("Bug 1: SMOTE Before Split")
# axes[0].grid(alpha=0.3, axis="x")
#
# axes[1].barh(["Naive target enc\n(LEAKY)","OOF target enc\n(CORRECT)"],
#              [f1_naive2, f1_corr2], color=["tomato","seagreen"], alpha=0.8)
# axes[1].set_xlabel("F1 Score")
# axes[1].set_title(f"Bug 2: Target Encoding Leakage\nInflation = {te_inflation:+.1f} F1 pts")
# axes[1].grid(alpha=0.3, axis="x")
# for i, val in enumerate([f1_naive2, f1_corr2]):
#     axes[1].text(val+0.002, i, f"{val:.4f}", va="center", fontsize=10)
#
# axes[2].plot(thrs, val_f1s, "steelblue", lw=2.5)
# axes[2].axvline(0.5, color="tomato", linestyle="--", lw=2, label="Default=0.5")
# axes[2].axvline(best_thr, color="seagreen", linestyle="--", lw=2,
#                 label=f"Optimal={best_thr:.3f}")
# axes[2].axvline(approx_thr, color="purple", linestyle=":", lw=1.5,
#                 label=f"Formula={approx_thr:.3f}")
# axes[2].set_xlabel("Decision threshold"); axes[2].set_ylabel("F1 on validation")
# axes[2].set_title("Bug 3: Class Weights Shift Calibration\nThreshold Must Be Re-tuned")
# axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("pipeline_ordering.png", dpi=120)
# print()
# print("  Plot saved -> pipeline_ordering.png")
# print()
# print("  CORRECT ORDER: split -> indicators -> impute -> encode")
# print("                 -> winsorize -> scale -> SMOTE (train only)")
# print("                 -> fit model -> tune threshold on val set")
# ''',
#     },
#
# }
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Dedent operation code strings
# # ─────────────────────────────────────────────────────────────────────────────
# for _op in OPERATIONS.values():
#     _op["code"] = textwrap.dedent(_op["code"]).strip()
#
#
# def _strip_ansi(text):
#     return re.compile(r'\x1b\[[0-9;]*m').sub('', text)
#
#
# def get_content():
#     return {
#         "display_name":  DISPLAY_NAME,
#         "icon":          ICON,
#         "subtitle":      SUBTITLE,
#         "theory":        THEORY,
#         "visual_html":   "",
#         "visual_height": 400,
#         "complexity":    None,
#         "operations":    OPERATIONS,
#     }