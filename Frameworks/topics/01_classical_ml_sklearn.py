"""
Scikit-Learn — The Universal ML Framework
==========================================

Scikit-learn is the most widely used classical machine learning library in Python.
It provides a clean, consistent API for dozens of algorithms — from linear models
to ensemble methods — along with tools for preprocessing, evaluation, and pipelines.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Scikit-Learn: The Universal ML Framework"
DISPLAY_NAME = "01 . sklearn · Scikit-Learn"
ICON         = "🔬"
SUBTITLE     = "Unified API for Classical Machine Learning"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER — converts local images to base64 HTML for st.markdown()
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    """Convert a local image file to an HTML <img> tag with base64 data.
    This allows images to render inside st.markdown() with unsafe_allow_html=True.
    """
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

### What is Scikit-Learn?

Scikit-learn (sklearn) is the foundational Python library for classical machine learning.
It was born out of a Google Summer of Code project in 2007, and today it is the single
most depended-upon ML library in the Python ecosystem — used by researchers, data
scientists, and engineers alike.

Its defining characteristic is not any particular algorithm, but its **design philosophy**:
a single, consistent API that works the same way regardless of which algorithm you use.
Once you understand the sklearn API, you can swap a Logistic Regression for a Random Forest
for a Gradient Boosting Machine with a two-word change. That uniformity is its superpower.

Think of scikit-learn as the **"standard toolkit"** of machine learning — the first place
you reach for before considering more specialized frameworks (PyTorch, TensorFlow, etc.).
It handles the 80% of real-world ML that doesn't require a neural network: tabular data,
feature engineering, model evaluation, and production pipelines.

---

### The Three Pillars of the sklearn API

Scikit-learn organizes every object around three types of operations. Master these three
interfaces and you can use any algorithm in the library.

    **Pillar 1 — Estimators (Learn from data)**

        An Estimator is any object that learns parameters from data.
        It has exactly ONE method that does the learning:

            estimator.fit(X, y)         ← supervised learning (X = features, y = labels)
            estimator.fit(X)            ← unsupervised learning (no labels needed)

        After fit(), the learned parameters are stored on the object (e.g., model.coef_).

    **Pillar 2 — Predictors (Make predictions)**

        A Predictor is an Estimator that can produce outputs for new data:

            estimator.predict(X)        ← returns class labels or numeric values
            estimator.predict_proba(X)  ← returns probability distributions (classifiers)
            estimator.score(X, y)       ← returns accuracy or R² depending on the task

    **Pillar 3 — Transformers (Reshape data)**

        A Transformer reshapes, scales, or encodes data without producing a final prediction:

            transformer.transform(X)       ← apply the transformation
            transformer.fit_transform(X)   ← fit and transform in one step (more efficient)

        Examples: StandardScaler, PCA, OneHotEncoder, SimpleImputer.

---

    **Diagram 1 — The Three Interfaces Visualized:**

    SCIKIT-LEARN OBJECT HIERARCHY
    ══════════════════════════════════════════════════════════════════

         BaseEstimator
              │
              ├─── Estimator  ──►  .fit(X, y)  or  .fit(X)
              │         │
              │         ├─── Predictor  ──►  .predict(X)
              │         │                    .predict_proba(X)
              │         │                    .score(X, y)
              │         │
              │         └─── Transformer  ──►  .transform(X)
              │                                .fit_transform(X)
              │
              └─── Pipeline  ──►  chains Transformers + final Estimator

    ──────────────────────────────────────────────────────────────────
    TYPE              LEARNS FROM DATA?   PRODUCES OUTPUT?   RESHAPES?
    ──────────────────────────────────────────────────────────────────
    Classifier        Yes                 Class labels        No
    Regressor         Yes                 Numeric values      No
    Clusterer         Yes (unsupervised)  Cluster IDs         No
    Preprocessor      Yes                 —                   Yes
    Dimensionality    Yes                 —                   Yes
    Reducer
    ──────────────────────────────────────────────────────────────────

---

### The Complete ML Workflow in sklearn

Every machine learning project follows the same high-level arc. Understanding this
arc — and which sklearn tool handles each step — is more valuable than memorizing
any individual algorithm.


    **Diagram 2 — End-to-End ML Pipeline:**

    RAW DATA ──► SPLIT ──► PREPROCESS ──► MODEL ──► EVALUATE ──► DEPLOY
         │           │           │            │           │
         │       train_test   StandardScaler  fit()    cross_val
         │         split()    OneHotEncoder   predict()  score()
         │                    SimpleImputer   score()    GridSearch
         │
         └── Real world: wrap ALL of this in a Pipeline object

    Step-by-step:

    1. COLLECT & LOAD DATA
       └── pandas DataFrames, numpy arrays, or sklearn's built-in toy datasets
           (load_iris, load_digits, make_classification, etc.)

    2. EXPLORE (EDA)
       └── Check shapes, missing values, class imbalance, feature distributions

    3. SPLIT DATA
       └── train_test_split(X, y, test_size=0.2, random_state=42)
           → Never touch test data until final evaluation!

    4. PREPROCESS
       └── Scale numeric features   → StandardScaler / MinMaxScaler
           Encode categoricals      → OneHotEncoder / OrdinalEncoder
           Handle missing values    → SimpleImputer
           Reduce dimensions        → PCA / SelectKBest

    5. CHOOSE & TRAIN A MODEL
       └── model = RandomForestClassifier(n_estimators=100)
           model.fit(X_train, y_train)

    6. EVALUATE
       └── model.score(X_test, y_test)
           classification_report(y_test, y_pred)
           cross_val_score(model, X, y, cv=5)

    7. TUNE HYPERPARAMETERS
       └── GridSearchCV / RandomizedSearchCV

    8. WRAP IN A PIPELINE (production-ready)
       └── Pipeline([('scaler', StandardScaler()), ('clf', SVC())])


---

### Part 1: Supervised Learning

Supervised learning is the most common ML task: given labeled examples (X, y), learn
a mapping from inputs to outputs that generalizes to new, unseen data.

Sklearn divides supervised learning into two subcategories:

    **Classification** — predict a discrete category (class label)
        Examples: spam vs. not spam, digit recognition, disease diagnosis

    **Regression** — predict a continuous numeric value
        Examples: house price, temperature forecast, stock return


    **Diagram 3 — Classification vs. Regression:**

    CLASSIFICATION                      REGRESSION
    ══════════════════════════          ══════════════════════════
    Input X → Discrete label y          Input X → Continuous value ŷ

        x₂                                  y (output)
        ↑                                   ↑
        │   ○ ○                             │         ●
        │     ○  ╲                          │      ●●
        │         ╲  ★ ★                    │   ●●
        │          ╲  ★                     │  ●●
        │○          ╲★                      │●
        └──────────────→ x₁                 └───────────→ x (input)

    Decision boundary separates           Best-fit line (or curve)
    classes in feature space              minimizes prediction error

    ─────────────────────────────         ─────────────────────────────
    Output: class label (0, 1, "cat")     Output: float (100.0, 3.14)
    Metric: accuracy, F1, AUC-ROC         Metric: MSE, RMSE, R²
    ─────────────────────────────         ─────────────────────────────

**Key Classification Algorithms in sklearn:**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Algorithm               Class                       Best For            │
    │──────────────────────────────────────────────────────────────────────────│
    │  Logistic Regression     LogisticRegression()        Linearly separable  │
    │                                                      Binary baseline     │
    │  K-Nearest Neighbors     KNeighborsClassifier()      Small datasets      │
    │                                                      Non-parametric      │
    │  Support Vector Machine  SVC()                       High-dim data       │
    │                                                      Small/medium data   │
    │  Decision Tree           DecisionTreeClassifier()    Interpretable       │
    │                                                      Mixed feature types │
    │  Random Forest           RandomForestClassifier()    General purpose     │
    │                                                      Handles overfitting │
    │  Gradient Boosting       GradientBoostingClassifier  Tabular data        │
    │                          HistGradientBoosting...     Competition winner  │
    │  Naive Bayes             GaussianNB()                NLP, text data      │
    │                                                      Very fast           │
    └──────────────────────────────────────────────────────────────────────────┘

**Key Regression Algorithms:**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Algorithm               Class                       Best For            │
    │──────────────────────────────────────────────────────────────────────────│
    │  Linear Regression       LinearRegression()          Baseline, simple    │
    │  Ridge Regression        Ridge()                     L2 regularization   │
    │  Lasso Regression        Lasso()                     Feature selection   │
    │  ElasticNet              ElasticNet()                L1 + L2 combined    │
    │  Decision Tree           DecisionTreeRegressor()     Non-linear          │
    │  Random Forest           RandomForestRegressor()     Robust, general     │
    │  SVR                     SVR()                       Non-linear, small   │
    │  Gradient Boosting       GradientBoostingRegressor() Best tabular perf   │
    └──────────────────────────────────────────────────────────────────────────┘


---

### Part 2: Unsupervised Learning

Unsupervised learning works with unlabeled data. Instead of predicting a target,
the model finds hidden structure in the data — patterns, groupings, or compressed
representations that weren't given explicitly.

The two major branches are:

    **Clustering** — group similar data points together
        No labels needed. The algorithm discovers natural groupings.

    **Dimensionality Reduction** — compress high-dimensional data
        Retain the most information using fewer features.


    **Diagram 4 — Clustering and Dimensionality Reduction:**

    CLUSTERING (K-Means example)          DIM REDUCTION (PCA example)
    ══════════════════════════════        ══════════════════════════════════

    Before:                               Before (3D):
        ● ● ●      ▲ ▲                       ●  ●  ● (x, y, z)
         ● ●      ▲   ▲
          ●        ▲▲                     After (2D — principal components):
          ■ ■ ■                              ● ● ● (PC1, PC2)
           ■  ■                              Most variance preserved!

    After k-means assigns cluster IDs:    PCA rotates the axes to align with
        ●=cluster 0, ▲=cluster 1,         the directions of maximum variance,
        ■=cluster 2                       then projects down to fewer dims.

    ─────────────────────────────────    ────────────────────────────────────
    Output: integer cluster ID            Output: transformed feature matrix
    No y labels needed                    Useful for visualization (n_comp=2)
                                          and preprocessing before modeling

**Key Unsupervised Algorithms in sklearn:**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Algorithm          Class                Best For                        │
    │──────────────────────────────────────────────────────────────────────────│
    │  K-Means            KMeans()             Globular clusters, fast         │
    │  DBSCAN             DBSCAN()             Arbitrary shapes, outlier det.  │
    │  Agglomerative      AgglomerativeClustr  Hierarchical clustering         │
    │  Gaussian Mixture   GaussianMixture()    Soft cluster assignments        │
    │  PCA                PCA()                Linear dim reduction            │
    │  t-SNE              TSNE()               2D/3D visualization only        │
    │  Truncated SVD      TruncatedSVD()       Sparse data, NLP                │
    │  IsolationForest    IsolationForest()    Anomaly/outlier detection       │
    └──────────────────────────────────────────────────────────────────────────┘


---

### Part 3: Preprocessing — The Most Underrated Part of ML

In practice, the quality of your preprocessing often matters more than your choice
of algorithm. Sklearn provides a rich set of transformers that handle the messy
realities of real-world data.

**Why Scaling Matters — An Intuitive Example:**

Imagine you're classifying people using two features:
    - age (range: 20–80)
    - annual salary in dollars (range: 20,000–200,000)

Without scaling, a distance-based algorithm like KNN will almost completely
ignore age because salary differences (in raw units) are 1000x larger.
The salary feature will dominate simply because of its units, not its importance.

StandardScaler fixes this by transforming each feature to have mean=0 and std=1:

                z = (x - μ) / σ

Now both features have equal scale, and the algorithm can evaluate them fairly.

    **Diagram 5 — Effect of Scaling on a Decision Boundary:**

    WITHOUT SCALING                     WITH STANDARDSCALER
    ══════════════════════════          ══════════════════════════

    salary ($)                          salary (standardized)
        ↑                                   ↑
    200k│          ★ ★★                    2│    ★ ★★
    150k│         ★★                       1│   ★★
    100k│  ○ ○○                            0│○ ○○
     50k│○                                -1│○
        └──────────→ age                    └──────────→ age (std)
          20  50  80                         -1  0  1

    The salary axis dominates.          Both axes have equal importance.
    Decision boundary is near-           Decision boundary is fair.
    horizontal (age barely matters).


**Common Preprocessing Operations:**

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Task                      sklearn Class          Notes                  │
    │──────────────────────────────────────────────────────────────────────────│
    │  Feature Scaling           StandardScaler()       Mean=0, Std=1          │
    │  Min-Max Scaling           MinMaxScaler()         Scales to [0,1]        │
    │  Missing Value Imputation  SimpleImputer()        mean/median/constant   │
    │  One-Hot Encode Categories OneHotEncoder()        Nominal → binary cols  │
    │  Ordinal Encode Categories OrdinalEncoder()       Ordered categories     │
    │  Polynomial Features       PolynomialFeatures()   Add x², x₁x₂, etc.     │
    │  Feature Selection         SelectKBest()          Keep top-K features    │
    │  Variance Threshold        VarianceThreshold()    Remove near-constant   │
    └──────────────────────────────────────────────────────────────────────────┘

**The Data Leakage Trap:**

    ⚠️  One of the most common and costly mistakes in ML:

    WRONG (data leakage):                CORRECT (no leakage):
    ──────────────────────               ─────────────────────────────
    scaler.fit(X)          ← uses        X_train, X_test = split(X)
    X_scaled = transform(X)  ALL data    scaler.fit(X_train)      ← only train
    X_train, X_test = split(X_scaled)    X_train = transform(X_train)
                                         X_test = transform(X_test)  ← apply only

    Why it matters: fitting the scaler on the full dataset lets test-set
    statistics "leak" into training. Your validation metrics will look better
    than they really are, leading to false confidence in your model.

    The solution: always fit preprocessors ONLY on training data, then
    apply (transform) to validation/test data. sklearn Pipelines enforce
    this automatically — which is another reason to always use them.


---

### Part 4: Model Evaluation — Knowing When to Trust Your Model

A model is only as good as your ability to measure its true performance.
Evaluation is not an afterthought — it is the core of the scientific method
applied to machine learning.

**The Bias-Variance Tradeoff:**

All ML models make errors. These errors come from two sources:

    **Bias** — error from wrong assumptions in the model.
        A high-bias model is too simple. It cannot capture the true pattern.
        It underfits the data — it performs poorly on BOTH training and test sets.

    **Variance** — error from sensitivity to small fluctuations in training data.
        A high-variance model is too complex. It memorizes training data.
        It overfits — performs great on training set, terrible on test set.


    **Diagram 6 — Bias-Variance Tradeoff:**

    UNDERFITTING          JUST RIGHT              OVERFITTING
    (High Bias)           (Balanced)              (High Variance)
    ════════════          ════════════            ════════════════

        ● ●                  ● ●                    ● ●
       ●   ●               ●   ●                  ●/  \●
      ●     ●             ●     ●                ●/    \●
      ─────────           ~~~~~~~~~              ~\/\/\/~
    Straight line         Smooth curve          Wiggly curve
    can't fit the         follows the           memorizes every
    real pattern          true shape            training point

    Train error: HIGH     Train error: LOW       Train error: VERY LOW
    Test error: HIGH      Test error: LOW        Test error: HIGH
                          ← THIS IS THE GOAL →

**Cross-Validation — The Gold Standard of Evaluation:**

A single train/test split can be misleading — you might have gotten lucky (or
unlucky) with how the data was split. Cross-validation gives a more reliable
estimate by testing on multiple different held-out subsets.


    **Diagram 7 — k-Fold Cross-Validation (k=5):**

    Full dataset: [■■■■■■■■■■■■■■■■■■■■]  (20 examples)

    Fold 1: [TEST ][■■■■][■■■■][■■■■][■■■■]  → score₁
    Fold 2: [■■■■][TEST ][■■■■][■■■■][■■■■]  → score₂
    Fold 3: [■■■■][■■■■][TEST ][■■■■][■■■■]  → score₃
    Fold 4: [■■■■][■■■■][■■■■][TEST ][■■■■]  → score₄
    Fold 5: [■■■■][■■■■][■■■■][■■■■][TEST ]  → score₅

    Final score = mean(score₁, score₂, score₃, score₄, score₅)
                ± std_dev  ← tells you how stable the model is

    Usage: cross_val_score(model, X, y, cv=5, scoring='accuracy')

    WHY IT'S BETTER THAN A SINGLE SPLIT:
        - Every example is used for testing exactly once
        - Every example is used for training k-1 times
        - You get a distribution of scores, not just one number
        - The std deviation reveals model stability


**Classification Metrics — Beyond Accuracy:**

Accuracy (correct / total) seems intuitive but is often misleading.
Consider a dataset with 95% class 0 and 5% class 1.
A model that always predicts class 0 achieves 95% accuracy — but is completely useless.

    Confusion Matrix — the foundation of all classification metrics:

    ┌──────────────────────────────────────────────────────────────┐
    │                      PREDICTED                               │
    │                  Negative     Positive                       │
    │   ACTUAL  ─────────────────────────────                      │
    │   Negative │  TN (True  │  FP (False │  ← Specificity        │
    │            │  Negative) │  Positive) │    = TN/(TN+FP)       │
    │   Positive │  FN (False │  TP (True  │  ← Recall/Sensitivity │
    │            │  Negative) │  Positive) │    = TP/(TP+FN)       │
    │                  ↑              ↑                            │
    │              Precision      Precision                        │
    │              (not useful)   = TP/(TP+FP)                     │
    └──────────────────────────────────────────────────────────────┘

    Key derived metrics:
        Precision  = TP / (TP + FP)    → "Of my positive predictions, how many were right?"
        Recall     = TP / (TP + FN)    → "Of all actual positives, how many did I catch?"
        F1 Score   = 2 × (P × R)/(P+R) → Harmonic mean of precision and recall
        AUC-ROC    = Area under ROC curve → Model's ability to discriminate classes

    When to use which:
        - High precision needed: spam filter (don't wrongly mark legit email as spam)
        - High recall needed: cancer screening (don't miss any real cases)
        - Balanced: F1 score

**Regression Metrics:**

    MAE  (Mean Absolute Error)    = mean(|y - ŷ|)       ← robust to outliers
    MSE  (Mean Squared Error)     = mean((y - ŷ)²)      ← penalizes large errors
    RMSE (Root MSE)               = √MSE                ← same units as y
    R²   (Coefficient of Det.)    = 1 - SS_res/SS_tot   ← 1.0 = perfect, 0 = baseline


---

### Part 5: Pipelines — The Professional Way to Build Models

A Pipeline chains multiple steps (transformers + a final estimator) into a single
sklearn-compatible object. This is the single most important feature for
taking models from notebooks to production.

    **Why Pipelines are Essential:**

        1. Prevents data leakage (fit only on training data, automatically)
        2. Makes cross-validation correct (each fold preprocesses independently)
        3. One object to save, load, and deploy (joblib.dump / joblib.load)
        4. Hyperparameter tuning spans ALL steps (tune scaler AND model together)
        5. Clean, readable code


    **Diagram 8 — Pipeline Data Flow:**

    X_train ──► [Step 1: Imputer] ──► [Step 2: Scaler] ──► [Step 3: PCA] ──► [Step 4: Classifier]
                  SimpleImputer        StandardScaler          PCA(10)           SVC(C=1.0)
                    .fit_transform()    .fit_transform()         .fit_transform()   .fit()

                                         ↑ TRAINING PHASE ↑

    X_test  ──► [Step 1: Imputer] ──► [Step 2: Scaler] ──► [Step 3: PCA] ──► [Step 4: Classifier]
                  .transform()          .transform()            .transform()       .predict()

                                         ↑ INFERENCE PHASE ↑

    KEY INSIGHT:
        fit()   is called ONLY during training — on training data
        transform() is called on BOTH train and test — using training statistics
        The Pipeline enforces this automatically. You cannot accidentally leak.


    **ColumnTransformer — Handle Mixed Data Types:**

    Real datasets have numeric AND categorical features. ColumnTransformer
    applies different transformations to different columns simultaneously:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Raw DataFrame                                                  │
    │  ┌──────┬────────┬──────────────┬───────────┐                   │
    │  │ age  │ salary │ department   │ city      │  → target: churn  │
    │  │ 34   │ 75000  │ Engineering  │ New York  │                   │
    │  │ 28   │ 52000  │ Marketing    │ Austin    │                   │
    │  └──────┴────────┴──────────────┴───────────┘                   │
    │        │               │                                        │
    │        ▼               ▼                                        │
    │  StandardScaler     OneHotEncoder                               │
    │  (numeric cols)     (categorical cols)                          │
    │        │                │                                       │
    │        └───────┬────────┘                                       │
    │                ▼                                                │
    │  Combined transformed feature matrix → Model                    │
    └─────────────────────────────────────────────────────────────────┘


---

### Part 6: Hyperparameter Tuning — Finding the Best Configuration

Every sklearn estimator has **parameters** (learned from data, like weights)
and **hyperparameters** (set before training, like `n_estimators` or `C`).
Hyperparameter tuning systematically searches for the best configuration.

    **Grid Search** — exhaustively try every combination:

        param_grid = {
            'classifier__C':        [0.1, 1.0, 10.0],
            'classifier__kernel':   ['linear', 'rbf'],
            'scaler__with_mean':    [True, False]
        }
        → Tries 3 × 2 × 2 = 12 combinations
        → Best for small grids where you know the right range

    **Randomized Search** — sample random combinations:

        param_dist = {
            'n_estimators':    [50, 100, 200, 500],
            'max_depth':       [None, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        }
        → Tries n_iter random combinations (e.g., 20 out of 48 total)
        → Better for large search spaces

    **Diagram 9 — Grid Search vs. Randomized Search:**

    GRID SEARCH                         RANDOMIZED SEARCH
    ════════════════════════            ════════════════════════
    C=0.1   C=1   C=10                  C is continuous [0.01, 100]
    ──────────────────                  ────────────────────────────
    ■       ■      ■   kernel='rbf'      ×       ×          ×
    ■       ■      ■   kernel='lin'         ×          ×
                                           ×               ×
    All 6 combinations tried.           20 random points sampled.

    Good when grid is small.            Better when space is large.
    Misses between grid points.         Covers space more uniformly.

    Both methods use cross-validation internally to evaluate each combination.
    The best_params_ and best_score_ are available on the fitted search object.


---

### Part 7: The sklearn Ecosystem — What Else It Connects To

Scikit-learn doesn't exist in isolation. It is the hub of a rich ecosystem:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                      THE SKLEARN ECOSYSTEM                           │
    │                                                                      │
    │   Data In      Preprocessing    Modeling       Evaluation/Tracking   │
    │   ─────────    ──────────────   ─────────      ──────────────────    │
    │   pandas  ──►  sklearn        ──► sklearn  ──►  sklearn metrics      │
    │   numpy       transformers       estimators     cross_val_score      │
    │                    │                 │                │              │
    │              imbalanced-learn    XGBoost         MLflow              │
    │              (SMOTE, etc.)       LightGBM        Weights & Biases    │
    │              feature-engine      CatBoost        joblib (saving)     │
    │              category_encoders   skorch                              │
    │                                 (PyTorch in sklearn API)             │
    └──────────────────────────────────────────────────────────────────────┘

    **sklearn-compatible wrappers** (same fit/predict API):
        - XGBoost:   xgb.XGBClassifier()        ← plug directly into Pipelines
        - LightGBM:  lgb.LGBMClassifier()        ← same as native sklearn
        - CatBoost:  CatBoostClassifier()         ← handles categoricals natively
        - skorch:    NeuralNetClassifier()        ← wraps PyTorch in sklearn API


---

### Part 8: When to Use sklearn vs. Deep Learning

Sklearn is not the right tool for every problem. Knowing when to use it —
and when to reach for PyTorch or TensorFlow — is a core professional skill.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  USE SKLEARN WHEN:            USE DEEP LEARNING (PyTorch/TF) WHEN:  │
    │  ────────────────────         ────────────────────────────────────  │
    │  • Tabular/structured data    • Images, audio, video, raw text      │
    │  • Small/medium datasets      • Very large datasets (millions+)     │
    │    (< ~1M rows)               • Transfer learning (fine-tuning)     │
    │  • Need interpretability      • Sequence/time-series modeling       │
    │  • Limited compute (CPU)      • Need custom architectures           │
    │  • Fast iteration needed      • Embeddings, representation learning │
    │  • Baseline model first       • State-of-the-art performance        │
    │  • Feature engineering focus  • GPU required for reasonable speed   │
    │                                                                     │
    │  Rule of thumb:                                                     │
    │  Try a Gradient Boosting model (sklearn/XGBoost) on tabular data    │
    │  BEFORE reaching for a neural network. It will often win.           │
    └─────────────────────────────────────────────────────────────────────┘

    The Gradient Boosting baseline:
        GBMs (GradientBoostingClassifier, HistGradientBoostingClassifier,
        XGBoost, LightGBM, CatBoost) are the DOMINANT algorithm for tabular data.
        In ML competitions on structured data, the winner is a GBM 80%+ of the time.
        A well-tuned GBM is often indistinguishable from a neural network on tables.

---

### Summary: The sklearn Mental Model

    1. Everything in sklearn is an Estimator: fit() / predict() / transform()

    2. The workflow is always: Split → Preprocess → Fit → Evaluate → Tune

    3. Preprocessing must be fit on train data only (use Pipelines to enforce this)

    4. Cross-validation beats a single train/test split for honest evaluation

    5. Choose metrics that match your problem (accuracy is often wrong; use F1/AUC)

    6. Pipelines are not optional in production — they prevent leakage and simplify deployment

    7. Start with a GBM baseline before considering neural networks on tabular data

    8. sklearn is interoperable with XGBoost, LightGBM, Pandas, and the entire PyData stack
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
None
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ──────────────────────────────────────────────────────────────────────────
    "1. The sklearn API: fit / predict / transform": {
        "description": "Core API demo — train a classifier and a transformer side by side",
        "runnable": True,
        "pipeline_cmd": "sklearn",
        "code": '''
"""
================================================================================
SKLEARN CORE API — fit / predict / transform
================================================================================

The entire sklearn library follows three interfaces:

    Estimator.fit(X, y)         ← learn parameters from data
    Predictor.predict(X)        ← produce outputs for new data
    Transformer.transform(X)    ← reshape / scale the data

Once you know these three, you can use any algorithm in sklearn.
================================================================================
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load a dataset
# ─────────────────────────────────────────────────────────────────────────────
# sklearn comes with built-in toy datasets for learning and testing.
# load_iris() returns a Bunch object with .data (features) and .target (labels).
#
# Iris dataset:
#   - 150 samples, 4 features, 3 classes (setosa, versicolor, virginica)
#   - Features: sepal length, sepal width, petal length, petal width
# ─────────────────────────────────────────────────────────────────────────────

iris = load_iris()
X = iris.data        # shape: (150, 4) — feature matrix
y = iris.target      # shape: (150,)   — class labels [0, 1, 2]

print("=" * 60)
print("  SKLEARN CORE API DEMONSTRATION")
print("=" * 60)
print(f"""
Dataset: Iris
  Samples:  {X.shape[0]}
  Features: {X.shape[1]}  ({', '.join(iris.feature_names)})
  Classes:  {len(iris.target_names)}  ({', '.join(iris.target_names)})
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Split data into training and test sets
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL RULE: NEVER fit any model or preprocessor on test data.
# The test set simulates unseen data. Touching it before evaluation
# gives you an optimistic (misleading) estimate of real-world performance.
#
# test_size=0.2 → 80% training, 20% test
# random_state=42 → reproducible split (same split every run)
# stratify=y → preserves class proportions in both splits
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # ensures each class is proportionally represented
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")
print(f"Class distribution (train): {np.bincount(y_train)}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Preprocessing — StandardScaler (a Transformer)
# ─────────────────────────────────────────────────────────────────────────────
# StandardScaler is a Transformer: it scales features to mean=0, std=1.
#
# CORRECT order (no data leakage):
#   1. fit() ONLY on training data → computes mean and std from train set
#   2. transform() on train → apply scaling using training statistics
#   3. transform() on test  → apply the SAME scaling (not refit!)
#
# If you fit on the full dataset first, test-set statistics contaminate
# the training process → inflated validation scores → broken evaluation.
# ─────────────────────────────────────────────────────────────────────────────

scaler = StandardScaler()

# fit() computes: self.mean_ = mean(X_train) and self.scale_ = std(X_train)
scaler.fit(X_train)

# transform() applies: X_scaled = (X - mean_) / scale_
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # use TRAINING mean/std, not test!

print("Scaler learned from training data:")
print(f"  Feature means:  {scaler.mean_.round(3)}")
print(f"  Feature stds:   {scaler.scale_.round(3)}")
print(f"  Train mean after scaling: {X_train_scaled.mean(axis=0).round(6)}")
print(f"  Train std  after scaling: {X_train_scaled.std(axis=0).round(6)}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Train a model (an Estimator + Predictor)
# ─────────────────────────────────────────────────────────────────────────────
# KNeighborsClassifier classifies a new point by the majority label among its
# k nearest neighbors in the feature space.
#
# n_neighbors=5 is the key hyperparameter:
#   Small k (1-3) → complex boundary, prone to overfitting noisy data
#   Large k (50+) → very smooth boundary, may underfit
# ─────────────────────────────────────────────────────────────────────────────

model = KNeighborsClassifier(n_neighbors=5)

# fit() stores the training data — KNN is a lazy learner (no actual training)
model.fit(X_train_scaled, y_train)

print("Model trained: KNeighborsClassifier(n_neighbors=5)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Evaluate
# ─────────────────────────────────────────────────────────────────────────────

y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test,  y_pred_test)

print(f"Train Accuracy: {train_acc:.4f}  ({train_acc*100:.1f}%)")
print(f"Test  Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
print()

gap = train_acc - test_acc
if gap > 0.05:
    print(f"  ⚠️  Gap of {gap:.3f} → possible OVERFITTING (model memorized training data)")
elif gap < -0.01:
    print(f"  ⚠️  Negative gap → possible data leakage or luck")
else:
    print(f"  ✅ Small gap ({gap:.3f}) → model generalizes well")

print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Manual prediction walkthrough
# ─────────────────────────────────────────────────────────────────────────────
# Let's manually predict a single flower to see exactly what's happening.

sample = X_test_scaled[0].reshape(1, -1)   # shape must be (1, n_features)
prediction = model.predict(sample)
probas = model.predict_proba(sample)

print("Manual prediction for test sample #0:")
print(f"  Raw features (original):  {X_test[0].round(2)}")
print(f"  Scaled features:          {X_test_scaled[0].round(3)}")
print(f"  Predicted class:          {prediction[0]} ({iris.target_names[prediction[0]]})")
print(f"  True label:               {y_test[0]} ({iris.target_names[y_test[0]]})")
print(f"  Class probabilities:")
for i, (cls, p) in enumerate(zip(iris.target_names, probas[0])):
    bar = "█" * int(p * 20)
    print(f"    {cls:15s}: {p:.2f}  {bar}")

print()
print("=" * 60)
print("  Key Takeaways")
print("=" * 60)
print("""
  1. fit()       → learns parameters from training data only
  2. transform() → applies learned transformation (scaler)
  3. predict()   → applies learned model to new data
  4. ALWAYS fit preprocessors on train, transform on both train & test
  5. The test set is sacred — never use it until final evaluation
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "2. Pipelines — The Professional Way": {
        "description": "Build a production-grade Pipeline that chains preprocessing and modeling",
        "runnable": True,
        "pipeline_cmd": "sklearn",
        "code": '''
"""
================================================================================
SKLEARN PIPELINES
================================================================================

A Pipeline chains multiple steps (transformers + a final estimator) into a
single sklearn-compatible object.

Benefits:
    1. Prevents data leakage automatically
    2. One object to fit, predict, save, and deploy
    3. Cross-validation works correctly across all steps
    4. Hyperparameter tuning can span ALL steps simultaneously
================================================================================
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Create a synthetic dataset with some missing values
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

# Simulate real-world messiness: randomly insert NaN values (~10% of data)
mask = np.random.rand(*X.shape) < 0.10
X[mask] = np.nan

print("=" * 60)
print("  SKLEARN PIPELINE DEMONSTRATION")
print("=" * 60)
print(f"""
Dataset: Synthetic Binary Classification
  Samples:  {X.shape[0]}
  Features: {X.shape[1]}
  Missing values: {np.isnan(X).sum()} ({np.isnan(X).mean()*100:.1f}% of cells)
  Class balance: {np.bincount(y)}
""")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ─────────────────────────────────────────────────────────────────────────────
# WRONG WAY (what NOT to do — data leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("WRONG approach (data leakage illustration):")
print("─" * 60)
print("""
  imputer = SimpleImputer()
  imputer.fit(X)                ← ⚠️ uses test set statistics!
  X_all = imputer.transform(X)
  X_train, X_test = split(X_all)  ← test info already leaked into train

  This inflates evaluation scores — you'll be overconfident about
  real-world performance. Never do this in practice.
""")


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT WAY: Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Each step is a (name, estimator) tuple.
# Steps are applied in order: imputer → scaler → classifier
#
# During fit():
#   imputer.fit_transform(X_train) → scaler.fit_transform(X_train_imp) → clf.fit(X_train_imp_scaled, y_train)
#
# During predict():
#   imputer.transform(X) → scaler.transform(X_imp) → clf.predict(X_imp_scaled)
#
# The Pipeline NEVER calls fit() on test data. Leakage impossible.
# ─────────────────────────────────────────────────────────────────────────────

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # fill NaN with column median
    ('scaler',  StandardScaler()),                    # center and scale
    ('clf',     RandomForestClassifier(
                    n_estimators=100,
                    random_state=42))                 # final estimator
])

# ONE call to fit — handles ALL steps automatically
pipeline.fit(X_train, y_train)

print("─" * 60)
print("CORRECT approach: Pipeline")
print("─" * 60)
print("Pipeline steps:")
for name, step in pipeline.steps:
    print(f"  [{name}]  {step.__class__.__name__}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate the pipeline
# ─────────────────────────────────────────────────────────────────────────────

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Evaluation on test set:")
print(classification_report(y_test, y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation on the full pipeline
# ─────────────────────────────────────────────────────────────────────────────
# cross_val_score splits the data, then fits the ENTIRE PIPELINE from scratch
# on each fold's training data. This is the only correct way to cross-validate
# a pipeline — it ensures no leakage across folds.
# ─────────────────────────────────────────────────────────────────────────────

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print(f"Cross-Validation (5-fold):")
for i, score in enumerate(cv_scores):
    bar = "█" * int(score * 30)
    print(f"  Fold {i+1}: {score:.4f}  {bar}")
print(f"  Mean:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Compare multiple pipelines (the real power)
# ─────────────────────────────────────────────────────────────────────────────
# Because all classifiers follow the same API, swapping them is trivial.
# Just change the last step — everything else stays identical.
# ─────────────────────────────────────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF kernel)":    SVC(probability=True, random_state=42),
}

print("─" * 60)
print("Comparing multiple classifiers (same preprocessing):")
print("─" * 60)
print(f"  {'Model':<25} {'CV Mean':>8}  {'CV Std':>7}")
print(f"  {'-' * 45}")

results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     clf)
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"  {name:<25} {scores.mean():>8.4f}  ±{scores.std():>6.4f}")

print()
best = max(results, key=lambda k: results[k].mean())
print(f"  Best model: {best}")
print()
print("=" * 60)
print("  Key Takeaways")
print("=" * 60)
print("""
  1. Pipelines prevent data leakage — fit() only on training data
  2. cross_val_score on a Pipeline is automatically leak-free per fold
  3. Swapping algorithms is a one-liner — same API for everything
  4. In production: pipeline.fit(X_train), joblib.dump(pipeline, 'model.pkl')
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "3. Model Evaluation — Beyond Accuracy": {
        "description": "Confusion matrix, classification report, cross-validation, and regression metrics",
        "runnable": True,
        "pipeline_cmd": "sklearn",
        "code": '''
"""
================================================================================
SKLEARN MODEL EVALUATION
================================================================================

Accuracy is often the wrong metric.
This module shows the full evaluation toolkit:
    - Confusion matrix + derived metrics
    - Classification report
    - Cross-validation with multiple scoring metrics
    - Regression metrics (MAE, MSE, RMSE, R²)
    - The bias-variance tradeoff visualized through learning curves
================================================================================
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PART A: CLASSIFICATION EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

# Create an imbalanced dataset (90% class 0, 10% class 1)
# This is where accuracy becomes dangerously misleading
X_clf, y_clf = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.90, 0.10],   # 90% class 0, 10% class 1
    random_state=42,
    flip_y=0.05
)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
)

print("=" * 65)
print("  CLASSIFICATION EVALUATION — IMBALANCED DATASET (90/10)")
print("=" * 65)
print(f"""
Class distribution in test set:
  Class 0 (majority): {(y_test == 0).sum()} samples ({(y_test == 0).mean()*100:.0f}%)
  Class 1 (minority): {(y_test == 1).sum()} samples ({(y_test == 1).mean()*100:.0f}%)
""")


# Dummy classifier: always predicts the majority class
class AlwaysMajorityClassifier:
    def fit(self, X, y): self.majority = 0; return self
    def predict(self, X): return np.zeros(len(X), dtype=int)
    def predict_proba(self, X): return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

dummy = AlwaysMajorityClassifier()
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)


# Real classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])
pipe.fit(X_train, y_train)
y_pred_real = pipe.predict(X_test)
y_proba_real = pipe.predict_proba(X_test)[:, 1]


# ── Print comparison ──
print("─" * 65)
print("Metric Comparison: Always-Majority vs. Real Classifier")
print("─" * 65)

metrics = {
    "Accuracy":  [accuracy_score(y_test, y_pred_dummy),
                  accuracy_score(y_test, y_pred_real)],
    "Precision": [precision_score(y_test, y_pred_dummy, zero_division=0),
                  precision_score(y_test, y_pred_real)],
    "Recall":    [recall_score(y_test, y_pred_dummy),
                  recall_score(y_test, y_pred_real)],
    "F1 Score":  [f1_score(y_test, y_pred_dummy),
                  f1_score(y_test, y_pred_real)],
    "AUC-ROC":   [0.5,  # random = 0.5 AUC
                  roc_auc_score(y_test, y_proba_real)],
}

print(f"  {'Metric':<12}  {'Always-0':>10}  {'RandomForest':>14}  {'Better?'}")
print(f"  {'-' * 55}")
for name, (v_dummy, v_real) in metrics.items():
    winner = "✅ Real" if v_real > v_dummy else "❌ Same/Worse"
    print(f"  {name:<12}  {v_dummy:>10.4f}  {v_real:>14.4f}  {winner}")

print(f"""
  ⚠️  The always-majority model gets 90% ACCURACY despite being useless.
  ✅  F1 Score and AUC-ROC correctly reveal the real classifier is better.
  → In imbalanced problems, NEVER report only accuracy.
""")


# ── Confusion Matrix ──
print("─" * 65)
print("Confusion Matrix (Real Classifier):")
print("─" * 65)
cm = confusion_matrix(y_test, y_pred_real)
tn, fp, fn, tp = cm.ravel()
print(f"""
                        PREDICTED
                    Negative   Positive
    ACTUAL Negative   {tn:>5}      {fp:>5}    ← {tn} correct rejections, {fp} false alarms
           Positive   {fn:>5}      {tp:>5}    ← {fn} missed positives, {tp} caught positives

    True Negatives  (TN): {tn}  — correctly said "no" for negative examples
    False Positives (FP): {fp}  — falsely said "yes" (Type I error)
    False Negatives (FN): {fn}  — missed a positive (Type II error)
    True Positives  (TP): {tp}  — correctly said "yes" for positive examples

    Precision = TP/(TP+FP) = {tp}/{tp+fp} = {tp/(tp+fp):.3f}  (of predicted positives, how many were right?)
    Recall    = TP/(TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.3f}  (of all actual positives, how many did we catch?)
""")


# ── Full Classification Report ──
print("─" * 65)
print("Full Classification Report:")
print("─" * 65)
print(classification_report(y_test, y_pred_real, target_names=['Class 0', 'Class 1']))


# ─────────────────────────────────────────────────────────────────────────────
# PART B: REGRESSION EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=20, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(Xr_train, yr_train)
yr_pred = lin_reg.predict(Xr_test)

mae  = mean_absolute_error(yr_test, yr_pred)
mse  = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
r2   = r2_score(yr_test, yr_pred)

print("=" * 65)
print("  REGRESSION EVALUATION")
print("=" * 65)
print(f"""
Model: LinearRegression on synthetic regression dataset

  MAE  = {mae:.3f}   ← average |error| (same units as y, robust to outliers)
  MSE  = {mse:.3f}  ← average squared error (penalizes large errors more)
  RMSE = {rmse:.3f}   ← √MSE (same units as y, comparable to std of error)
  R²   = {r2:.4f}    ← fraction of variance explained (1.0 = perfect, 0 = baseline)

  R² interpretation:
    R² = 1.0  → model explains ALL variance — perfect prediction
    R² = 0.0  → model is no better than predicting the mean of y
    R² < 0.0  → model is WORSE than the mean baseline (very bad)
    R² = {r2:.4f}  → this model explains {r2*100:.1f}% of the variance in y
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART C: CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 65)
print("Cross-Validation (5-fold, StratifiedKFold for classification):")
print("─" * 65)

pipe_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc    = cross_val_score(pipe_cv, X_clf, y_clf, cv=cv, scoring='accuracy')
cv_f1     = cross_val_score(pipe_cv, X_clf, y_clf, cv=cv, scoring='f1')
cv_auc    = cross_val_score(pipe_cv, X_clf, y_clf, cv=cv, scoring='roc_auc')

print(f"  {'Fold':<8} {'Accuracy':>10} {'F1 Score':>10} {'AUC-ROC':>10}")
print(f"  {'-' * 42}")
for i, (acc, f1, auc) in enumerate(zip(cv_acc, cv_f1, cv_auc)):
    print(f"  Fold {i+1:<3}  {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
print(f"  {'-' * 42}")
print(f"  {'Mean':<8} {cv_acc.mean():>10.4f} {cv_f1.mean():>10.4f} {cv_auc.mean():>10.4f}")
print(f"  {'±Std':<8} {cv_acc.std():>10.4f} {cv_f1.std():>10.4f} {cv_auc.std():>10.4f}")
print(f"""
  Reading the table:
    Mean  → expected performance on unseen data
    ± Std → stability of the model; high std = sensitive to which data was used

  A model with Mean=0.85 ± 0.02 is more trustworthy than
  a model with Mean=0.88 ± 0.12 — the second is far less stable.
""")

print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  CLASSIFICATION:
    1. Accuracy is misleading on imbalanced data → use F1, AUC-ROC
    2. Confusion matrix reveals WHERE your model fails
    3. Precision vs. recall tradeoff: tune the threshold for your use case

  REGRESSION:
    4. RMSE is in the same units as your target → easy to interpret
    5. R² tells you how much better you are than always predicting the mean

  GENERAL:
    6. Cross-validation is more reliable than a single train/test split
    7. Report mean ± std — a stable model is as important as a high-scoring one
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "4. Hyperparameter Tuning — GridSearch & RandomizedSearch": {
        "description": "Find the best hyperparameters using cross-validated grid and random search",
        "runnable": True,
        "pipeline_cmd": "sklearn",
        "code": '''
"""
================================================================================
HYPERPARAMETER TUNING WITH GRIDSEARCHCV AND RANDOMIZEDSEARCHCV
================================================================================

Model parameters are learned during training (e.g., weights).
Hyperparameters are set BEFORE training (e.g., n_estimators, C, max_depth).

sklearn provides two automated search strategies:
    GridSearchCV        — exhaustive: tries every combination
    RandomizedSearchCV  — stochastic: samples n_iter random combinations

Both use cross-validation internally to evaluate each configuration,
ensuring honest estimates of generalization performance.
================================================================================
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


np.random.seed(42)
X, y = make_classification(
    n_samples=600, n_features=10, n_informative=7,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("=" * 65)
print("  HYPERPARAMETER TUNING")
print("=" * 65)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print()


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: GridSearchCV — exhaustive search over a parameter grid
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: In a Pipeline, hyperparameter names use double-underscore notation:
#       'step_name__param_name'
#   So 'clf__C' means the 'C' parameter of the step named 'clf'.
# ─────────────────────────────────────────────────────────────────────────────

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True))
])

param_grid_svm = {
    'clf__C':      [0.01, 0.1, 1.0, 10.0, 100.0],  # regularization strength
    'clf__kernel': ['linear', 'rbf'],                # decision boundary shape
    'clf__gamma':  ['scale', 'auto'],                # RBF kernel bandwidth
}

total_combos = (len(param_grid_svm['clf__C']) *
                len(param_grid_svm['clf__kernel']) *
                len(param_grid_svm['clf__gamma']))

print(f"─" * 65)
print(f"METHOD 1: GridSearchCV (SVM)")
print(f"─" * 65)
print(f"  Parameter grid:")
for name, values in param_grid_svm.items():
    print(f"    {name}: {values}")
print(f"  Total combinations: {total_combos}")
print(f"  CV folds: 5")
print(f"  Total model fits: {total_combos * 5}")
print()

grid_search = GridSearchCV(
    estimator=svm_pipe,
    param_grid=param_grid_svm,
    cv=5,
    scoring='f1',          # optimize for F1 score
    n_jobs=-1,             # use all available CPU cores
    verbose=0,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"  Best Parameters: {grid_search.best_params_}")
print(f"  Best CV F1 Score: {grid_search.best_score_:.4f}")
print()

# Show top 5 parameter combinations
import pandas as pd
results_df = pd.DataFrame(grid_search.cv_results_)
top5 = results_df.nlargest(5, 'mean_test_score')[
    ['param_clf__C', 'param_clf__kernel', 'mean_test_score', 'std_test_score']
]
print("  Top 5 parameter combinations (by mean CV F1):")
print(f"  {'C':>8}  {'Kernel':>8}  {'Mean F1':>9}  {'Std F1':>8}")
print(f"  {'-' * 42}")
for _, row in top5.iterrows():
    print(f"  {str(row['param_clf__C']):>8}  {str(row['param_clf__kernel']):>8}  "
          f"{row['mean_test_score']:>9.4f}  ±{row['std_test_score']:>6.4f}")
print()

# Evaluate best SVM on test set
y_pred_svm = grid_search.predict(X_test)
svm_test_f1 = grid_search.score(X_test, y_test)
print(f"  Test F1 Score (best SVM): {svm_test_f1:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: RandomizedSearchCV — sample random combinations
# ─────────────────────────────────────────────────────────────────────────────
# Better when:
#   - Search space is large (exponential number of combinations)
#   - You don't know the right range for each parameter
#   - You want to explore many parameters quickly
#
# Key insight from Bergstra & Bengio (2012): random search outperforms grid
# search when only a few parameters matter — random search samples more
# unique values for each parameter, while grid search fixes some dimensions.
# ─────────────────────────────────────────────────────────────────────────────

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_dist_rf = {
    'clf__n_estimators':     [50, 100, 200, 300, 500],
    'clf__max_depth':        [None, 5, 10, 15, 20, 30],
    'clf__min_samples_split': [2, 5, 10, 15],
    'clf__min_samples_leaf':  [1, 2, 4, 8],
    'clf__max_features':     ['sqrt', 'log2', None],
    'clf__bootstrap':        [True, False],
}

max_combos = (5 * 6 * 4 * 4 * 3 * 2)  # total possible combinations
n_iter = 30

print(f"─" * 65)
print(f"METHOD 2: RandomizedSearchCV (Random Forest)")
print(f"─" * 65)
print(f"  Total possible combinations: {max_combos}")
print(f"  Combinations sampled (n_iter): {n_iter}")
print(f"  Efficiency: {n_iter}/{max_combos} = {n_iter/max_combos*100:.1f}% of the full grid")
print(f"  Total fits: {n_iter * 5} (vs {max_combos * 5} for full grid)")
print()

rand_search = RandomizedSearchCV(
    estimator=rf_pipe,
    param_distributions=param_dist_rf,
    n_iter=n_iter,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=0,
    return_train_score=True
)

rand_search.fit(X_train, y_train)

print(f"  Best Parameters:")
for k, v in rand_search.best_params_.items():
    print(f"    {k}: {v}")
print(f"  Best CV F1 Score: {rand_search.best_score_:.4f}")

y_pred_rf = rand_search.predict(X_test)
rf_test_f1 = rand_search.score(X_test, y_test)
print(f"  Test F1 Score (best RF): {rf_test_f1:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Compare default vs tuned models
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.metrics import f1_score

default_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])
default_rf.fit(X_train, y_train)
default_f1 = f1_score(y_test, default_rf.predict(X_test))

print(f"─" * 65)
print(f"Impact of Hyperparameter Tuning:")
print(f"─" * 65)
print(f"  Default RF (all defaults):  Test F1 = {default_f1:.4f}")
print(f"  Tuned RF (RandomizedSearch): Test F1 = {rf_test_f1:.4f}")
gain = rf_test_f1 - default_f1
print(f"  Improvement:                {'+' if gain >= 0 else ''}{gain:.4f} ({'+' if gain >= 0 else ''}{gain*100:.2f}%)")
print()

print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  1. Always tune inside a Pipeline — hyperparameters span ALL steps
  2. Use double-underscore notation: 'step_name__param_name'
  3. GridSearch: complete but slow — best for small, well-defined grids
  4. RandomizedSearch: efficient — better for large spaces or unknowns
  5. The tuning process uses cross-validation internally — no test leakage
  6. final_estimator = search.best_estimator_ → a ready-to-deploy Pipeline
  7. Never use the test set to decide hyperparameters — use CV scores only
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "5. Preprocessing — ColumnTransformer for Real Datasets": {
        "description": "Handle mixed numeric and categorical features with ColumnTransformer inside a Pipeline",
        "runnable": True,
        "pipeline_cmd": "sklearn",
        "code": '''
"""
================================================================================
SKLEARN PREPROCESSING WITH COLUMNTRANSFORMER
================================================================================

Real-world datasets mix numeric and categorical features.
ColumnTransformer applies different transformations to different columns,
then concatenates the results into a single feature matrix.

This is the standard pattern for production ML on tabular data.
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Create a realistic mixed-type dataset (employee churn prediction)
# ─────────────────────────────────────────────────────────────────────────────

n = 400
data = pd.DataFrame({
    # Numeric features
    'age':              np.random.randint(22, 62, n),
    'salary':           np.random.normal(70000, 20000, n).round(),
    'years_at_company': np.random.randint(0, 20, n),
    'satisfaction':     np.random.uniform(1, 10, n).round(1),
    'hours_per_week':   np.random.normal(45, 8, n).round(1),

    # Categorical features
    'department':       np.random.choice(['Engineering', 'Sales', 'HR', 'Finance'], n),
    'education':        np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n),
    'remote_policy':    np.random.choice(['Full Remote', 'Hybrid', 'On-Site'], n),
})

# Introduce missing values (~8%)
for col in ['salary', 'satisfaction', 'department']:
    mask = np.random.rand(n) < 0.08
    data.loc[mask, col] = np.nan

# Target: churn (binary)
# Higher satisfaction and salary → less likely to churn
churn_prob = (
    0.4
    - 0.03 * data['satisfaction'].fillna(5)
    + 0.005 * (data['hours_per_week'].fillna(45) - 45)
    - 0.000002 * data['salary'].fillna(70000)
)
y = (np.random.rand(n) < churn_prob.clip(0.05, 0.95)).astype(int)
X = data.copy()

print("=" * 65)
print("  COLUMNTRANSFORMER — MIXED FEATURE TYPES")
print("=" * 65)
print()
print(f"Dataset: Employee Churn Prediction")
print(f"  Samples: {n},  Churn rate: {y.mean()*100:.1f}%")
print()
print(f"Feature types:")
print(X.dtypes.to_string())
print()
print(f"Missing values:")
print(X.isnull().sum()[X.isnull().sum() > 0].to_string())
print()


# ─────────────────────────────────────────────────────────────────────────────
# Define feature groups
# ─────────────────────────────────────────────────────────────────────────────

numeric_features     = ['age', 'salary', 'years_at_company',
                         'satisfaction', 'hours_per_week']
nominal_features     = ['department', 'remote_policy']   # no natural order
ordinal_features     = ['education']                     # has natural order


# ─────────────────────────────────────────────────────────────────────────────
# Build the ColumnTransformer
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry: (name, transformer, columns)
#
# Numeric pipeline:
#   SimpleImputer(median)  → fill NaN with column median (robust to outliers)
#   StandardScaler         → center + scale to mean=0, std=1
#
# Nominal pipeline:
#   SimpleImputer(most_frequent) → fill NaN with most common category
#   OneHotEncoder               → create binary indicator columns
#     handle_unknown='ignore'   → unseen categories at inference → all zeros
#
# Ordinal pipeline:
#   OrdinalEncoder with explicit category order → preserves rank information
# ─────────────────────────────────────────────────────────────────────────────

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

ordinal_transformer = Pipeline([
    ('encoder', OrdinalEncoder(
        categories=[['High School', 'Bachelor', 'Master', 'PhD']],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric',  numeric_transformer,  numeric_features),
        ('nominal',  nominal_transformer,  nominal_features),
        ('ordinal',  ordinal_transformer,  ordinal_features),
    ],
    remainder='drop'   # drop any columns not explicitly listed
)


# ─────────────────────────────────────────────────────────────────────────────
# Build the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   GradientBoostingClassifier(
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=3,
                         random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

full_pipeline.fit(X_train, y_train)

print("─" * 65)
print("Pipeline Structure:")
print("─" * 65)
print(f"""
  ColumnTransformer
  ├── [numeric]  → SimpleImputer(median) → StandardScaler
  │   Columns: {numeric_features}
  │
  ├── [nominal]  → SimpleImputer(most_freq) → OneHotEncoder
  │   Columns: {nominal_features}
  │
  └── [ordinal]  → OrdinalEncoder(High School < Bachelor < Master < PhD)
      Columns: {ordinal_features}

  ↓
  GradientBoostingClassifier
""")


# ─────────────────────────────────────────────────────────────────────────────
# Show what the transformed data looks like
# ─────────────────────────────────────────────────────────────────────────────

X_transformed = preprocessor.fit_transform(X_train)
ohe_feature_names = (preprocessor
                     .named_transformers_['nominal']
                     .named_steps['encoder']
                     .get_feature_names_out(nominal_features).tolist())

all_feature_names = numeric_features + ohe_feature_names + ordinal_features

print(f"─" * 65)
print(f"Original features: {X.shape[1]}")
print(f"After transformation: {X_transformed.shape[1]} features")
print()
print(f"Final feature list:")
for i, name in enumerate(all_feature_names):
    print(f"  [{i:>2}] {name}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

y_pred = full_pipeline.predict(X_test)
print("─" * 65)
print("Model Performance:")
print("─" * 65)
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring='f1')
print(f"5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()

# Feature importance from the GBM (on transformed features)
importances = full_pipeline.named_steps['classifier'].feature_importances_
print("Top 5 Most Important Features:")
sorted_idx = importances.argsort()[::-1][:5]
for rank, idx in enumerate(sorted_idx):
    bar = "█" * int(importances[idx] * 100)
    print(f"  {rank+1}. {all_feature_names[idx]:<30} {importances[idx]:.4f}  {bar}")

print()
print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  1. ColumnTransformer handles mixed data types in one step
  2. Different columns need different transformers — no one-size-fits-all
  3. Ordinal features need explicit category order (not random integers)
  4. OneHotEncoder creates binary columns — avoids false numeric ordering
  5. handle_unknown='ignore' makes inference robust to new categories
  6. Wrapping in a Pipeline ensures correct fit-only-on-train behavior
  7. get_feature_names_out() extracts interpretable feature names after OHE
""")
'''
    },

    # ──────────────────────────────────────────────────────────────────────────
    "6. Unsupervised Learning — Clustering & Dimensionality Reduction": {
        "description": "K-Means clustering, DBSCAN, and PCA dimensionality reduction from scratch",
        "runnable": True,
        "pipeline_cmd": "sklearn",
        "code": '''
"""
================================================================================
UNSUPERVISED LEARNING IN SKLEARN
================================================================================

Unsupervised learning finds structure in data WITHOUT labels.

This module covers:
    1. K-Means Clustering  — partition data into k globular clusters
    2. DBSCAN              — density-based clustering (handles arbitrary shapes)
    3. PCA                 — linear dimensionality reduction

Key contrast with supervised learning:
    Supervised:   learn f(X) → y   (need labels y)
    Unsupervised: find structure in X   (no labels needed)
================================================================================
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  UNSUPERVISED LEARNING: CLUSTERING & DIM REDUCTION")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
# K-Means partitions n samples into k clusters.
#
# Algorithm:
#   1. Initialize k centroids randomly (or with k-means++ for better starts)
#   2. Assign each point to its nearest centroid (Euclidean distance)
#   3. Recompute each centroid as the mean of its assigned points
#   4. Repeat steps 2-3 until centroids stop moving (convergence)
#
# Hyperparameters:
#   k (n_clusters) — must be specified upfront (key limitation)
#   init='k-means++' — smart initialization that spreads centroids out
#   n_init=10 — run the algorithm 10 times, keep the best result
# ─────────────────────────────────────────────────────────────────────────────

# Create 4 well-separated blobs
X_blobs, y_true = make_blobs(
    n_samples=300, centers=4, cluster_std=0.8, random_state=42
)

print()
print("─" * 65)
print("PART 1: K-Means Clustering")
print("─" * 65)
print(f"Dataset: {X_blobs.shape[0]} samples, 4 true clusters")
print()

# The Elbow Method — how to choose k
print("Elbow Method (choosing optimal k):")
print(f"  k     Inertia     Description")
print(f"  ─────────────────────────────────────────────────────────────")
for k in range(2, 8):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_blobs)
    bar = "█" * int(km.inertia_ / 500)
    flag = " ← elbow here" if k == 4 else ""
    print(f"  k={k}   {km.inertia_:>9.1f}   {bar[:40]}{flag}")

print(f"""
  The elbow method: plot inertia vs k.
  Inertia drops steeply, then levels off.
  The "elbow" point (where the curve bends) is the optimal k.
  At k=4 the inertia stops dropping sharply — we've found the 4 natural clusters.
""")

# Fit the final K-Means model
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_blobs)
labels_km = kmeans.labels_

sil  = silhouette_score(X_blobs, labels_km)
ari  = adjusted_rand_score(y_true, labels_km)

print(f"K-Means (k=4) Results:")
print(f"  Cluster sizes:     {np.bincount(labels_km)}")
print(f"  Inertia:           {kmeans.inertia_:.2f}")
print(f"  Silhouette Score:  {sil:.4f}  (range: -1 to 1, higher = better defined clusters)")
print(f"  Adjusted Rand Idx: {ari:.4f}  (1.0 = perfect match with true labels)")
print(f"""
  Silhouette Score interpretation:
    Near  1.0 → points are well inside their cluster, far from others
    Near  0.0 → points are near the boundary between clusters
    Near -1.0 → points may be in the wrong cluster
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: DBSCAN — Density-Based Clustering
# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
#
# Core idea: a cluster is a dense region of points separated from other clusters
# by regions of lower density. Points in sparse regions are classified as noise.
#
# Key advantages over K-Means:
#   - Does NOT require specifying k upfront
#   - Finds clusters of arbitrary shape (not just globular)
#   - Automatically identifies outliers/noise points (label = -1)
#
# Parameters:
#   eps       — maximum distance between two samples to be considered neighbors
#   min_samples — minimum number of samples in a neighborhood for a core point
# ─────────────────────────────────────────────────────────────────────────────

X_moons, y_moons = make_moons(n_samples=200, noise=0.08, random_state=42)

print()
print("─" * 65)
print("PART 2: DBSCAN Clustering (Non-Globular Shapes)")
print("─" * 65)
print(f"Dataset: Two interleaved half-moons (K-Means cannot separate these!)")
print()

# K-Means fails on non-globular shapes
km_moons = KMeans(n_clusters=2, random_state=42)
km_moons.fit(X_moons)
ari_km_moons = adjusted_rand_score(y_moons, km_moons.labels_)

# DBSCAN handles arbitrary shapes
db = DBSCAN(eps=0.25, min_samples=5)
db.fit(X_moons)
labels_db = db.labels_
n_clusters_found = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()
ari_db = adjusted_rand_score(y_moons, labels_db)

print(f"  Algorithm        Clusters Found   Noise Points   ARI Score")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  K-Means (k=2)    2                0              {ari_km_moons:.4f}  ← fails!")
print(f"  DBSCAN           {n_clusters_found}                {n_noise}              {ari_db:.4f}  ← works!")
print(f"""
  K-Means assumes clusters are globular (spherical) — it fails badly
  on crescent/moon shapes because it tries to fit spherical boundaries.

  DBSCAN walks through dense regions, growing clusters organically.
  It correctly identifies the two crescent-shaped clusters.
  Isolated points (low density) get label -1 = noise/outlier.

  DBSCAN label -1 = noise → use for outlier detection!
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: PCA — Principal Component Analysis
# ─────────────────────────────────────────────────────────────────────────────
# PCA finds the directions of maximum variance in the data (principal components)
# and projects the data onto a lower-dimensional subspace.
#
# Each principal component (PC) is:
#   - Orthogonal (perpendicular) to all other PCs
#   - The direction that captures maximum REMAINING variance
#
# Applications:
#   1. Visualization: project to 2D for human inspection
#   2. Preprocessing: remove redundant/correlated features
#   3. Noise reduction: discard components with low variance
# ─────────────────────────────────────────────────────────────────────────────

digits = load_digits()
X_digits = digits.data        # shape: (1797, 64) — each image = 8x8 = 64 pixels
y_digits = digits.target

scaler = StandardScaler()
X_digits_scaled = scaler.fit_transform(X_digits)

print()
print("─" * 65)
print("PART 3: PCA — Dimensionality Reduction")
print("─" * 65)
print(f"Dataset: Handwritten Digits (sklearn)")
print(f"  Samples: {X_digits.shape[0]}")
print(f"  Original features: {X_digits.shape[1]} (64 pixel intensities per 8×8 image)")
print()

# Full PCA to find how many components capture most variance
pca_full = PCA()
pca_full.fit(X_digits_scaled)

# Cumulative explained variance
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

print("Explained Variance vs. Number of Components:")
print(f"  {'Components':>12}  {'Cumul. Variance':>16}  {'Bar'}")
print(f"  {'-' * 55}")
for n in [1, 5, 10, 20, 30, 40, 50, 64]:
    pct = cumvar[n-1] * 100
    bar = "█" * int(pct / 3)
    print(f"  {n:>12}  {pct:>14.1f}%  {bar}")

# Find 95% and 99% thresholds
n_95 = np.argmax(cumvar >= 0.95) + 1
n_99 = np.argmax(cumvar >= 0.99) + 1
print(f"""
  n_components for 95% variance: {n_95}  (compresses 64 → {n_95} features: {100-n_95/64*100:.0f}% size reduction)
  n_components for 99% variance: {n_99}  (compresses 64 → {n_99} features: {100-n_99/64*100:.0f}% size reduction)
""")

# Apply PCA for 2D visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_digits_scaled)

print(f"PCA 2D Projection:")
print(f"  PC1 explains: {pca_2d.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"  PC2 explains: {pca_2d.explained_variance_ratio_[1]*100:.1f}% of variance")
print(f"  Total:        {sum(pca_2d.explained_variance_ratio_)*100:.1f}% of variance in 2 dimensions")
print()

# Show 2D cluster structure (ASCII visualization)
print("  2D PCA projection — class separation visible:")
print(f"  (Showing centroids per digit class in PC1-PC2 space)")
print()
for digit in range(10):
    mask = y_digits == digit
    cx = X_2d[mask, 0].mean()
    cy = X_2d[mask, 1].mean()
    print(f"  Digit {digit}: PC1={cx:+.2f}, PC2={cy:+.2f}")

print()
print("=" * 65)
print("  Key Takeaways")
print("=" * 65)
print("""
  K-MEANS:
    1. Fast and scalable — good default for globular clusters
    2. Requires k upfront — use Elbow Method or Silhouette Score to choose
    3. Fails on non-globular shapes and unequal cluster densities

  DBSCAN:
    4. No k required — finds clusters of any shape
    5. Label -1 = noise/outlier — built-in anomaly detection
    6. Sensitive to eps and min_samples — tune with domain knowledge

  PCA:
    7. Always StandardScale before PCA — PCA is sensitive to scale
    8. explained_variance_ratio_ tells you how much info each PC contains
    9. Common rule: keep enough PCs for 95% cumulative variance
   10. PCA is linear — for non-linear structure, consider t-SNE or UMAP
""")
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None

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
    visual_html   = ""
    visual_height = 400
    # try:
    #     from frameworks.visuals.sklearn_visual import (
    #         SKLEARN_VISUAL_HTML,
    #         SKLEARN_VISUAL_HEIGHT,
    #     )
    #     visual_html   = SKLEARN_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SKLEARN_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[sklearn_framework.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None,
        "operations":    OPERATIONS,
    }