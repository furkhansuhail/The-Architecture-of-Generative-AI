"""
Scikit-Learn — Machine Learning in Python
==========================================

Scikit-Learn (sklearn) is Python's premier machine learning library, providing
a consistent API for the entire supervised and unsupervised ML workflow.

"""

import re
import textwrap


TOPIC_NAME   = "Scikit-Learn: Machine Learning in Python"
DISPLAY_NAME = "01 · Scikit-Learn"
ICON         = "🤖"
SUBTITLE     = "The End-to-End Machine Learning Workflow"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is Scikit-Learn?

Scikit-Learn (imported as `sklearn`) is Python's standard library for classical
machine learning. It is built on top of NumPy, SciPy, and Matplotlib and covers
virtually every supervised and unsupervised learning algorithm you'll encounter
before entering the deep learning world.

Every major step in a machine learning project has a corresponding Scikit-Learn
tool: splitting data, filling missing values, encoding categories, fitting models,
evaluating performance, tuning hyperparameters, saving models, and chaining
everything into reproducible pipelines.


### Why Scikit-Learn?

There are two core reasons Scikit-Learn dominates practical ML work:

**1. A consistent, unified API.**
Every model in Scikit-Learn — whether a linear regression, a random forest, or
a support vector machine — exposes exactly the same interface:

    estimator.fit(X_train, y_train)      # learn patterns
    estimator.predict(X_test)            # apply learned patterns
    estimator.score(X_test, y_test)      # measure performance

This means that once you know how to work with one model, you can immediately
work with any other. Experimenting with different algorithms costs almost nothing.

**2. Production-grade implementations.**
Every algorithm is implemented with numerical stability, efficiency, and edge
cases in mind. You don't need to implement gradient descent, cross-validation,
or imputation yourself — Scikit-Learn provides battle-tested versions.


### The Standard Workflow

Every Scikit-Learn project follows roughly the same sequence of steps:

    ┌────────────────────────────────────────────────────────────────────┐
    │   THE SKLEARN WORKFLOW                                             │
    │                                                                    │
    │  0. Get the data → load CSV or built-in dataset                    │
    │                                                                    │
    │  1. Prepare the data:                                              │
    │     • Split into features X and labels y                           │
    │     • Fill or remove missing values (imputation)                   │
    │     • Encode categorical columns as numbers                        │
    │     • Split into train and test sets                               │
    │                                                                    │
    │  2. Choose a model (estimator) for your problem type:              │
    │     • Regression   → predict a number                              │
    │     • Classification → predict a category                          │
    │     • Clustering   → find groups with no labels                    │
    │                                                                    │
    │  3. Fit the model: estimator.fit(X_train, y_train)                 │
    │                                                                    │
    │  4. Evaluate: estimator.score() / cross_val_score() / metrics      │
    │                                                                    │
    │  5. Improve: hyperparameter tuning (RandomizedSearchCV/GridSearchCV)│
    │                                                                    │
    │  6. Save: pickle or joblib                                         │
    │                                                                    │
    │  7. Deploy: wrap in a Pipeline for reproducibility                 │
    └────────────────────────────────────────────────────────────────────┘


### Problem Types

The first question you must answer is: what kind of problem is this?

    Problem Type        Goal                        Sklearn Module
    ─────────────────────────────────────────────────────────────────────
    Classification      Predict a category           sklearn.ensemble
                        (e.g. heart disease: yes/no) sklearn.svm, etc.
    Regression          Predict a number             sklearn.ensemble
                        (e.g. house price)           sklearn.linear_model
    Clustering          Group unlabelled data        sklearn.cluster
    Dimensionality      Reduce features              sklearn.decomposition
    Reduction

A useful heuristic that holds surprisingly often:
    • Structured data (tables, DataFrames)  → ensemble methods (Random Forest, XGBoost)
    • Unstructured data (images, text)      → deep learning or transfer learning


### Getting Data Ready — Three Key Concerns

Before any model can learn, the data must satisfy three requirements that most
real-world datasets violate:

**1. Numerics only.** Machine learning models work with numbers. Text, categories,
   and booleans must be converted — a process called *feature encoding*.
   Scikit-Learn's `OneHotEncoder` turns each unique category value into its own
   binary column. `pd.get_dummies()` achieves the same result via pandas.

**2. No missing values.** Most estimators will raise an error if `NaN` values are
   present. The solution is *imputation*: replacing missing values with something
   sensible (column mean, median, a constant, or a learned value).
   Scikit-Learn's `SimpleImputer` handles this cleanly.

**3. Consistent train/test treatment.** The rule that cannot be broken:
   fit imputers and encoders ONLY on training data, then transform test data
   using those already-fitted objects. Fitting on test data causes *data leakage*
   — the model indirectly "sees" the test set before evaluation, inflating scores.

   fit_transform(X_train)   ← learn statistics from train, then transform it
   transform(X_test)        ← apply those same statistics to test (no re-fitting)


### Estimators — Sklearn's Core Abstraction

Every model, transformer, and preprocessor in Scikit-Learn is called an
*estimator*. All estimators share a common interface:

    fit(X, y)            Learn from data. For transformers, stores learned
                         statistics. For models, stores learned parameters.

    predict(X)           Use learned patterns to produce output.
                         Returns labels for classifiers, values for regressors.

    predict_proba(X)     Classifiers only. Returns class probabilities instead
                         of hard labels. Useful when a confidence score matters.

    transform(X)         Transformers only (imputers, scalers, encoders).
                         Apply the fitted transformation to new data.

    fit_transform(X)     Shortcut for fit() then transform() in one call.
                         Use ONLY on training data.

    score(X, y)          Built-in evaluation. Default metric depends on the
                         estimator type (accuracy for classifiers, R² for regressors).

    get_params()         Returns the current hyperparameter settings as a dict.


### Model Evaluation — Three Layers

Scikit-Learn provides evaluation at three levels of sophistication:

**Layer 1 — score()**: The fastest check. Returns a single number.
    clf.score(X_test, y_test)   # accuracy for classifiers
    model.score(X_test, y_test) # R² for regressors

**Layer 2 — cross_val_score()**: More reliable. Uses K-fold cross-validation,
    which evaluates the model on K different train/test splits and returns K scores.
    The mean of those scores is far less sensitive to lucky/unlucky splits than
    a single evaluation.

    cross_val_score(clf, X, y, cv=5)   → array of 5 scores

**Layer 3 — Problem-specific functions**: Maximum control.
    For classification:
        accuracy_score, precision_score, recall_score, f1_score
        roc_auc_score, confusion_matrix, classification_report
    For regression:
        r2_score, mean_absolute_error, mean_squared_error


### Classification Metrics Deep Dive

Understanding classification metrics requires understanding the four possible
outcomes for any prediction on a binary problem:

    ┌───────────────┬────────────────────┬────────────────────┐
    │               │  Predicted:  1     │  Predicted:  0     │
    ├───────────────┼────────────────────┼────────────────────┤
    │  Actual:  1   │ True Positive (TP) │ False Negative (FN)│
    │  Actual:  0   │ False Positive (FP)│ True Negative (TN) │
    └───────────────┴────────────────────┴────────────────────┘

From these four cells, all classification metrics are derived:

    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
                The fraction of all predictions that were correct.
                Misleading when classes are imbalanced.

    Precision = TP / (TP + FP)
                Of everything I labelled positive, what fraction truly was?
                High precision → few false alarms.

    Recall    = TP / (TP + FN)
                Of everything that truly was positive, what fraction did I catch?
                High recall → few missed positives.

    F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
                Harmonic mean of precision and recall. Use when both matter equally.

When to use which:
    • Accuracy        → classes are balanced (roughly equal number of each label)
    • Precision       → false positives are costly (e.g. spam filter: don't block real email)
    • Recall          → false negatives are costly (e.g. disease screening: don't miss sick patients)
    • F1-score        → both FP and FN matter, and classes may be imbalanced

The ROC-AUC score measures the model's ability to rank positive samples higher
than negative ones, across all possible decision thresholds. AUC = 1.0 is perfect;
AUC = 0.5 is random guessing.


### Regression Metrics Deep Dive

For regression (predicting continuous values), the key metrics are:

    R² (Coefficient of Determination):
        Measures what fraction of the variance in y is explained by the model.
        R² = 1.0  → perfect predictions
        R² = 0.0  → model is no better than predicting the mean every time
        R² < 0    → model is worse than predicting the mean (very bad model)

    MAE (Mean Absolute Error):
        Average of |y_pred - y_true| across all samples.
        Treats all errors equally. Easy to interpret — same units as y.

    MSE (Mean Squared Error):
        Average of (y_pred - y_true)² across all samples.
        Squaring amplifies large errors. Sensitive to outliers.
        Use when large errors are disproportionately worse than small ones.

    When to prefer MAE vs MSE:
        Being $10,000 off is TWICE as bad as being $5,000 off  → use MAE
        Being $10,000 off is MORE than twice as bad            → use MSE


### Hyperparameter Tuning — The Three Approaches

*Parameters* are learned by the model during training (weights, biases, split thresholds).
*Hyperparameters* are settings YOU choose before training (number of trees, max depth, etc.).

Tuning strategy 1 — **By hand**:
    Manually try different values, evaluate on a validation set. Intuitive but slow.
    Requires a three-way split: train / validation / test.

Tuning strategy 2 — **RandomizedSearchCV**:
    Define a grid of possible hyperparameter values. Sklearn randomly samples
    n_iter combinations, evaluates each with cross-validation, and returns the best.
    Best for large search spaces where exhaustive search is too slow.

Tuning strategy 3 — **GridSearchCV**:
    Exhaustively tests every combination in a grid. More thorough than random search
    but computationally expensive. Best for small, refined grids after a random search
    has narrowed down the neighborhood.

Both search methods store the best hyperparameters in `.best_params_` and the best
estimator in `.best_estimator_`. Calling `.predict()` on them automatically uses the best.


### Pipelines — Reproducible End-to-End Workflows

A `Pipeline` chains multiple steps (transformers + a final estimator) into a single
object. Benefits:

    1. Code is compact and readable — one object does everything.
    2. No leakage risk — Pipeline applies transformers correctly during cross-validation.
    3. Hyperparameter tuning works seamlessly — prefix hyperparameter names with
       the step name: "model__n_estimators", "preprocessor__num__imputer__strategy".

    Pipeline(steps=[
        ("preprocessor", preprocessor),   # any ColumnTransformer
        ("model",         model),          # any sklearn estimator
    ])

    # The whole pipeline behaves like a single estimator:
    pipe.fit(X_train, y_train)
    pipe.predict(X_test)
    pipe.score(X_test, y_test)


### Saving and Loading Models

Once trained, a model can be serialised to disk so it can be reused without retraining.
Scikit-Learn recommends two approaches:

    pickle — Python's built-in object serialisation
        import pickle
        pickle.dump(model, open("model.pkl", "wb"))    # save
        model = pickle.load(open("model.pkl", "rb"))   # load

    joblib — More efficient for models containing large NumPy arrays
        from joblib import dump, load
        dump(model, "model.joblib")     # save
        model = load("model.joblib")    # load

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 0 ──────────────────────────────────────────────────────────────────
    "00 · End-to-End Sklearn Workflow": {
        "description": (
            "A complete Scikit-Learn workflow in a single cell — from raw CSV data "
            "to trained model, predictions, and evaluation metrics. This is the "
            "pattern every subsequent operation expands on."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

            # ── 1. Get the data ──────────────────────────────────────────────
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df   = pd.DataFrame(data["data"], columns=data["feature_names"])
            df["target"] = data["target"]
            print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
            print(f"Classes: {dict(zip(['malignant','benign'], np.bincount(df['target'])))}")

            # ── 2. Create X (features) and y (labels) ────────────────────────
            X = df.drop("target", axis=1)
            y = df["target"]

            # ── 3. Split into train / test sets ─────────────────────────────
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                  test_size=0.2,
                                                                  random_state=42)
            print(f"\\nTrain: {X_train.shape}  Test: {X_test.shape}")

            # ── 4. Choose a model and instantiate it ─────────────────────────
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

            # ── 5. Fit the model on training data ────────────────────────────
            clf.fit(X_train, y_train)

            # ── 6. Make predictions ──────────────────────────────────────────
            y_preds = clf.predict(X_test)

            # ── 7. Evaluate ──────────────────────────────────────────────────
            print(f"\\nTrain accuracy : {clf.score(X_train, y_train)*100:.2f}%")
            print(f"Test  accuracy : {clf.score(X_test,  y_test )*100:.2f}%")
            print(f"accuracy_score : {accuracy_score(y_test, y_preds)*100:.2f}%")
            print("\\nClassification report:")
            print(classification_report(y_test, y_preds,
                                        target_names=["malignant", "benign"]))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, y_preds))
        ''',
    },

    # ── 1 ──────────────────────────────────────────────────────────────────
    "01 · Splitting Data — X, y, train_test_split": {
        "description": (
            "The universal first step: separate features (X) from labels (y), "
            "then split both into training and test sets. The test set is held out "
            "entirely until final evaluation — the model never touches it during training."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import load_breast_cancer

            data = load_breast_cancer()
            df = pd.DataFrame(data["data"], columns=data["feature_names"])
            df["target"] = data["target"]
            print("Full dataset shape:", df.shape)

            # ── Separate features and labels ─────────────────────────────────
            X = df.drop("target", axis=1)   # features — every column except target
            y = df["target"]                # labels  — the column to predict

            print("X shape:", X.shape)      # (569, 30) — 569 samples, 30 features
            print("y shape:", y.shape)      # (569,)

            # ── Split into train and test ────────────────────────────────────
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,      # 20% for testing, 80% for training
                random_state=42     # fix the shuffle seed for reproducibility
            )

            print(f"\\nX_train: {X_train.shape}  X_test:  {X_test.shape}")
            print(f"y_train: {y_train.shape}  y_test:  {y_test.shape}")
            print(f"Training samples  : {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
            print(f"Test samples      : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

            # ── Class distribution check ─────────────────────────────────────
            print(f"\\ny_train class balance: {dict(y_train.value_counts().sort_index())}")
            print(f"y_test  class balance: {dict(y_test.value_counts().sort_index())}")

            # stratify=y ensures proportional class distribution in both sets:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                       random_state=42, stratify=y)
            print(f"\\nWith stratify — train: {dict(y_tr.value_counts().sort_index())}")
            print(f"With stratify — test : {dict(y_te.value_counts().sort_index())}")
        ''',
    },

    # ── 2 ──────────────────────────────────────────────────────────────────
    "02 · Feature Encoding — OneHotEncoder & get_dummies": {
        "description": (
            "Machine learning models require numerical input. Categorical columns "
            "(strings like 'Toyota', 'Red', '4 doors') must be converted to numbers "
            "first. OneHotEncoder (sklearn) and pd.get_dummies (pandas) both convert "
            "each unique category value into its own binary column."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer

            # ── Simulate the car sales dataset ──────────────────────────────
            np.random.seed(42)
            car_sales = pd.DataFrame({
                "Make":         np.random.choice(["Toyota", "Honda", "BMW", "Ford"], 10),
                "Colour":       np.random.choice(["Red", "Blue", "White"], 10),
                "Doors":        np.random.choice([2, 4], 10),
                "Odometer (KM)": np.random.randint(10_000, 200_000, 10),
                "Price":        np.random.randint(5_000, 40_000, 10),
            })
            print("Car sales data:")
            print(car_sales.head())
            print("\\nData types:")
            print(car_sales.dtypes)

            # ── Problem: model cannot handle strings ─────────────────────────
            X = car_sales.drop("Price", axis=1)
            y = car_sales["Price"]

            # ── Solution 1: OneHotEncoder via ColumnTransformer (sklearn) ────
            categorical_features = ["Make", "Colour", "Doors"]
            one_hot     = OneHotEncoder(sparse_output=False)
            transformer = ColumnTransformer(
                [("one_hot", one_hot, categorical_features)],
                remainder="passthrough"   # keep non-listed columns as-is
            )

            transformed_X = transformer.fit_transform(X)
            print(f"\\nOriginal X shape : {X.shape}")
            print(f"Transformed shape: {transformed_X.shape}")
            print("First row (encoded + passthrough):", transformed_X[0])

            # ── Solution 2: pd.get_dummies (pandas) ──────────────────────────
            dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]])
            print("\\npd.get_dummies result:")
            print(dummies.head())

            # Key difference:
            # OneHotEncoder integrates into sklearn Pipelines and can be fitted
            # separately on train/test (safer). pd.get_dummies is faster to write
            # but fits on the full dataset.
        ''',
    },

    # ── 3 ──────────────────────────────────────────────────────────────────
    "03 · Handling Missing Data with Pandas": {
        "description": (
            "The simplest approach to missing values: fill them in-place using "
            "pandas .fillna() then drop rows where the target label is missing. "
            "A good first step before applying any machine learning model."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd

            # ── Simulate a car sales dataset with missing values ─────────────
            np.random.seed(42)
            n = 20
            car_sales_missing = pd.DataFrame({
                "Make":         np.random.choice(["Toyota", "Honda", None, "Ford"], n),
                "Colour":       np.random.choice(["Red", None, "White"], n),
                "Doors":        [np.random.choice([2, 4, None]) for _ in range(n)],
                "Odometer (KM)": [np.random.randint(10_000, 200_000)
                                   if np.random.rand() > 0.1 else None for _ in range(n)],
                "Price":        [np.random.randint(5_000, 40_000)
                                  if np.random.rand() > 0.15 else None for _ in range(n)],
            })
            print("Missing values before filling:")
            print(car_sales_missing.isna().sum())
            print()

            # ── Step 1: Fill categorical columns with "missing" ──────────────
            car_sales_missing = car_sales_missing.assign(
                Make   = car_sales_missing["Make"].fillna("missing"),
                Colour = car_sales_missing["Colour"].fillna("missing"),
            )

            # ── Step 2: Fill numerical columns with mean / a constant ─────────
            mean_odo = car_sales_missing["Odometer (KM)"].mean()
            car_sales_missing = car_sales_missing.assign(
                **{"Odometer (KM)": car_sales_missing["Odometer (KM)"].fillna(mean_odo)},
                Doors = car_sales_missing["Doors"].fillna(4),
            )

            print("After filling features:")
            print(car_sales_missing.isna().sum())

            # ── Step 3: Drop rows where target (Price) is missing ────────────
            # We don't want to invent fake labels — drop them instead.
            before = len(car_sales_missing)
            car_sales_missing.dropna(inplace=True)
            after  = len(car_sales_missing)
            print(f"\\nAfter dropping rows with missing Price:")
            print(car_sales_missing.isna().sum())
            print(f"Rows removed: {before - after}  Remaining: {after}")
        ''',
    },

    # ── 4 ──────────────────────────────────────────────────────────────────
    "04 · Handling Missing Data with SimpleImputer": {
        "description": (
            "Scikit-Learn's SimpleImputer fills missing values using a strategy "
            "(mean, median, constant). Crucially, the imputer is fitted ONLY on "
            "training data, then applied to both train and test — preventing data leakage."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.impute import SimpleImputer
            from sklearn.compose import ColumnTransformer
            from sklearn.model_selection import train_test_split

            # ── Simulate missing-data dataset ────────────────────────────────
            np.random.seed(42)
            n = 100
            df = pd.DataFrame({
                "Make":         np.random.choice(["Toyota", "Honda", None, "Ford"], n),
                "Colour":       np.random.choice(["Red", None, "White"], n),
                "Doors":        [np.random.choice([2, 4, None]) for _ in range(n)],
                "Odometer (KM)": [np.random.randint(10_000, 200_000)
                                   if np.random.rand() > 0.1 else None for _ in range(n)],
                "Price":        np.random.randint(5_000, 40_000, n),
            })

            # ── Drop rows with missing target, then split ────────────────────
            X = df.drop("Price", axis=1)
            y = df["Price"]
            np.random.seed(42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            print("Missing values in X_train:")
            print(X_train.isna().sum())

            # ── Build imputer with ColumnTransformer ─────────────────────────
            cat_imputer  = SimpleImputer(strategy="constant",  fill_value="missing")
            door_imputer = SimpleImputer(strategy="constant",  fill_value=4)
            num_imputer  = SimpleImputer(strategy="mean")

            imputer = ColumnTransformer([
                ("cat_imputer",  cat_imputer,  ["Make", "Colour"]),
                ("door_imputer", door_imputer, ["Doors"]),
                ("num_imputer",  num_imputer,  ["Odometer (KM)"]),
            ])

            # CRITICAL: fit_transform on TRAIN, transform-only on TEST
            filled_X_train = imputer.fit_transform(X_train)   # learn + apply
            filled_X_test  = imputer.transform(X_test)        # apply only (no re-fit)

            # Convert back to DataFrame for inspection
            cols = ["Make", "Colour", "Doors", "Odometer (KM)"]
            train_filled = pd.DataFrame(filled_X_train, columns=cols)
            test_filled  = pd.DataFrame(filled_X_test,  columns=cols)

            print("\\nAfter imputation — missing in train:")
            print(train_filled.isna().sum())
            print("After imputation — missing in test:")
            print(test_filled.isna().sum())
            print("\\nFirst 3 imputed rows:")
            print(train_filled.head(3))
        ''',
    },

    # ── 5 ──────────────────────────────────────────────────────────────────
    "05 · Choosing a Regression Model — Ridge vs RandomForestRegressor": {
        "description": (
            "Once you know the problem is regression (predicting a number), "
            "the Scikit-Learn algorithm map suggests starting with Ridge then "
            "trying ensemble methods. This operation demonstrates both on the "
            "Diabetes dataset — a standard built-in regression benchmark."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_diabetes
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import RandomForestRegressor

            # ── Load the Diabetes dataset (built-in, no download needed) ─────
            # Goal: predict diabetes progression score one year after baseline
            data = load_diabetes()
            df = pd.DataFrame(data["data"], columns=data["feature_names"])
            df["target"] = data["target"]   # disease progression score
            print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
            print(df.head(3))
            print(f"\\nTarget range: {df['target'].min():.0f} – {df['target'].max():.0f}")

            np.random.seed(42)
            X = df.drop("target", axis=1)
            y = df["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                  test_size=0.2,
                                                                  random_state=42)

            # ── Model 1: Ridge Regression (linear) ──────────────────────────
            # Good first attempt — fast, interpretable, suitable when data is linear
            ridge = Ridge()
            ridge.fit(X_train, y_train)
            ridge_score = ridge.score(X_test, y_test)
            print(f"\\nRidge R² score       : {ridge_score:.4f}")

            # ── Model 2: RandomForestRegressor (ensemble) ────────────────────
            # Ensemble of decision trees — powerful for non-linear patterns
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_score = rf.score(X_test, y_test)
            print(f"RandomForest R² score: {rf_score:.4f}")

            diff = rf_score - ridge_score
            print(f"\\nDifference: {'+' if diff>=0 else ''}{diff:.4f}")
            print("\\nKey insight: same fit/score API — only the estimator class changes.")
            print("Experiment with models freely; the workflow stays identical.")

            # Feature importance (Random Forest only)
            importances = pd.Series(rf.feature_importances_,
                                    index=X.columns).sort_values(ascending=False)
            print("\\nFeature importances (Random Forest):")
            print(importances.round(4))
        ''',
    },

    # ── 6 ──────────────────────────────────────────────────────────────────
    "06 · Choosing a Classification Model — LinearSVC vs RandomForestClassifier": {
        "description": (
            "For classification (predicting a category), the algorithm map suggests "
            "LinearSVC as a first attempt, then ensemble methods if scores are low. "
            "The workflow is identical to regression — only the estimator class changes."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.svm import LinearSVC
            from sklearn.ensemble import RandomForestClassifier

            def class_counts(y):
                """Return (count_class_0, count_class_1) for a binary label array."""
                vals, cnts = np.unique(y, return_counts=True)
                return dict(zip(vals.tolist(), cnts.tolist()))

            data = load_breast_cancer()
            X = data["data"]
            y = data["target"]
            print(f"Dataset: {len(X)} samples | Classes: 0=malignant, 1=benign")
            print(f"Class balance: {class_counts(y)}")

            np.random.seed(42)
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                  test_size=0.2,
                                                                  random_state=42)

            # ── Model 1: LinearSVC ────────────────────────────────────────────
            # Often the algorithm map's first suggestion for classification
            svc = LinearSVC(max_iter=10000, random_state=42)
            svc.fit(X_train, y_train)
            svc_score = svc.score(X_test, y_test)
            print(f"\\nLinearSVC accuracy      : {svc_score*100:.2f}%")

            # ── Model 2: RandomForestClassifier ──────────────────────────────
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            clf_score = clf.score(X_test, y_test)
            print(f"RandomForest accuracy   : {clf_score*100:.2f}%")

            diff = (clf_score - svc_score) * 100
            print(f"\\nImprovement: {'+' if diff >= 0 else ''}{diff:.2f}%")
            print("\\nRule of thumb: for structured/tabular data, try RandomForest first.")
            print("For unstructured data (images, text), try deep learning.")
        ''',
    },

    # ── 7 ──────────────────────────────────────────────────────────────────
    "07 · Fitting a Model and Making Predictions": {
        "description": (
            "The three key prediction methods: predict() for hard labels, "
            "predict_proba() for class probabilities, and the shape rules "
            "that must be followed — data passed to predict() must have the "
            "same number of features as data the model was trained on."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            # ── predict() — hard label output ───────────────────────────────
            y_preds = clf.predict(X_test)
            print("predict() on test set (first 10):")
            print("Predicted:", y_preds[:10])
            print("Actual   :", y_test[:10])

            # Manual accuracy calculation (equivalent to accuracy_score)
            manual_acc = np.mean(y_preds == y_test)
            print(f"\\nManual accuracy  : {manual_acc*100:.2f}%")
            print(f"accuracy_score() : {accuracy_score(y_test, y_preds)*100:.2f}%")

            # ── predict_proba() — probability output ─────────────────────────
            y_probs = clf.predict_proba(X_test[:5])
            print("\\npredict_proba() for first 5 samples:")
            print("Format: [P(class=0), P(class=1)]")
            for i, probs in enumerate(y_probs):
                label = clf.predict(X_test[i:i+1])[0]
                print(f"  Sample {i}: {probs}  → predicted label: {label}")

            # ── Shape requirement ────────────────────────────────────────────
            print(f"\\nModel trained on {X_train.shape[1]} features per sample.")
            print("predict() requires the same number of features:")
            try:
                clf.predict(np.array([[1, 2, 3]]))   # wrong number of features
            except ValueError as e:
                print(f"  Wrong shape error: {e}")

            print(f"  Correct shape: X_test[0:1].shape = {X_test[0:1].shape}")
            print(f"  Prediction   : {clf.predict(X_test[0:1])}")
        ''',
    },

    # ── 8 ──────────────────────────────────────────────────────────────────
    "08 · Evaluating Models — score() Method": {
        "description": (
            "The score() method is the quickest evaluation tool. It returns "
            "accuracy for classifiers and R² (coefficient of determination) "
            "for regressors. Importantly, always check BOTH train and test scores "
            "to diagnose overfitting."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # ── Classification: score() = mean accuracy ──────────────────────
            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            train_score = clf.score(X_train, y_train)
            test_score  = clf.score(X_test,  y_test)
            print("CLASSIFICATION (breast cancer)")
            print(f"  Train accuracy: {train_score*100:.2f}%")
            print(f"  Test  accuracy: {test_score*100:.2f}%")
            gap = train_score - test_score
            print(f"  Train-test gap: {gap*100:.2f}%  ", end="")
            print("(high gap → overfitting)" if gap > 0.1 else "(acceptable)")

            # ── Regression: score() = R² ──────────────────────────────────────
            from sklearn.datasets import load_diabetes
            d_reg = load_diabetes()
            X_h, y_h = d_reg["data"], d_reg["target"]
            X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
                X_h, y_h, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_h, y_train_h)

            train_r2 = model.score(X_train_h, y_train_h)
            test_r2  = model.score(X_test_h,  y_test_h)
            print("\\nREGRESSION (Diabetes dataset)")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test  R²: {test_r2:.4f}")
            print("  R²=1.0 → perfect. R²=0.0 → predicting the mean.")

            # ── What score() does under the hood ─────────────────────────────
            from sklearn.metrics import accuracy_score, r2_score
            print("\\n--- score() is equivalent to ---")
            print(f"  accuracy_score(y_test, clf.predict(X_test))  = "
                  f"{accuracy_score(y_test, clf.predict(X_test))*100:.2f}%")
            print(f"  r2_score(y_test_h, model.predict(X_test_h))  = "
                  f"{r2_score(y_test_h, model.predict(X_test_h)):.4f}")
        ''',
    },

    # ── 9 ──────────────────────────────────────────────────────────────────
    "09 · Evaluating Models — Cross-Validation": {
        "description": (
            "A single train/test split can be lucky or unlucky. Cross-validation "
            "evaluates the model K times on different splits and averages the results, "
            "giving a far more reliable estimate of real-world performance."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.ensemble import RandomForestClassifier

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            # ── Single split score ─────────────────────────────────────────
            single_score = clf.score(X_test, y_test)
            print(f"Single split score    : {single_score*100:.2f}%")

            # ── 5-fold cross-validation ───────────────────────────────────
            # Sklearn splits X into 5 folds; trains on 4, evaluates on 1, rotates.
            cv5_scores = cross_val_score(clf, X, y, cv=5)
            print(f"\\n5-fold CV scores      : {np.round(cv5_scores, 4)}")
            print(f"5-fold CV mean        : {np.mean(cv5_scores)*100:.2f}%")
            print(f"5-fold CV std         : ±{np.std(cv5_scores)*100:.2f}%")

            # ── 10-fold cross-validation ──────────────────────────────────
            cv10_scores = cross_val_score(clf, X, y, cv=10)
            print(f"\\n10-fold CV mean       : {np.mean(cv10_scores)*100:.2f}%")

            # ── Why cross-validation is preferred ────────────────────────
            print("\\n--- Comparison ---")
            print(f"  Single split         : {single_score*100:.2f}%  (may be lucky/unlucky)")
            print(f"  5-fold CV mean       : {np.mean(cv5_scores)*100:.2f}%  (average over 5 splits)")
            print(f"  10-fold CV mean      : {np.mean(cv10_scores)*100:.2f}%  (average over 10 splits)")
            print("\\nWhen reporting model performance, prefer the cross-validated score.")
            print("It is less sensitive to how the data was split.")
        ''',
    },

    # ── 10 ─────────────────────────────────────────────────────────────────
    "10 · Classification Metrics — Accuracy, ROC & AUC": {
        "description": (
            "Accuracy alone can be misleading on imbalanced datasets. The ROC curve "
            "and AUC score evaluate a classifier across ALL decision thresholds, "
            "giving a threshold-agnostic measure of how well the model separates classes."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_curve, roc_auc_score

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            # ── Accuracy ──────────────────────────────────────────────────
            print(f"Accuracy (test set): {clf.score(X_test, y_test)*100:.2f}%")

            # ── ROC curve requires probabilities for the positive class ────
            y_probs = clf.predict_proba(X_test)[:, 1]   # column 1 = P(benign)

            fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            auc = roc_auc_score(y_test, y_probs)
            print(f"ROC-AUC score       : {auc:.4f}")
            print("(AUC=1.0 is perfect, AUC=0.5 is random guessing)")

            # ── Perfect AUC (comparing labels to themselves) ──────────────
            perfect_auc = roc_auc_score(y_test, y_test)
            print(f"Perfect AUC         : {perfect_auc:.4f}")

            # ── Plot ──────────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Model ROC
            axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"Model (AUC={auc:.3f})")
            axes[0].plot([0,1], [0,1], color="navy", lw=1, linestyle="--", label="Random (AUC=0.5)")
            axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
            axes[0].set_title("ROC Curve"); axes[0].legend()

            # Perfect ROC
            fpr_p, tpr_p, _ = roc_curve(y_test, y_test)
            axes[1].plot(fpr_p, tpr_p, color="green", lw=2, label="Perfect (AUC=1.0)")
            axes[1].plot([0,1], [0,1], color="navy", lw=1, linestyle="--", label="Random")
            axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
            axes[1].set_title("Perfect ROC Curve"); axes[1].legend()

            plt.tight_layout()
            plt.savefig("roc_curves.png", dpi=100)
            plt.show()
            print("\\nROC curves saved to roc_curves.png")
            print("\\nInterpretation: the further the orange curve bows toward top-left,")
            print("the better the model at separating classes at any threshold.")
        ''',
    },

    # ── 11 ─────────────────────────────────────────────────────────────────
    "11 · Classification Metrics — Confusion Matrix": {
        "description": (
            "A confusion matrix shows exactly WHERE a model makes mistakes — "
            "which actual classes it predicts correctly and which it confuses. "
            "This operation demonstrates three ways to plot one, including "
            "Scikit-Learn's modern ConfusionMatrixDisplay API."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_preds = clf.predict(X_test)

            # ── Raw confusion matrix (numpy array) ───────────────────────
            cm = confusion_matrix(y_test, y_preds)
            print("Confusion matrix (raw):")
            print(cm)
            print("Format: [[TN FP]  ← predicted class 0")
            print("         [FN TP]] ← predicted class 1")

            # ── As a readable DataFrame ───────────────────────────────────
            print("\\nAs pandas cross-tabulation:")
            print(pd.crosstab(y_test, y_preds,
                              rownames=["Actual"], colnames=["Predicted"]))

            # ── ConfusionMatrixDisplay (sklearn ≥ 1.0) ───────────────────
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # From estimator (model makes predictions internally)
            ConfusionMatrixDisplay.from_estimator(
                estimator=clf, X=X_test, y=y_test,
                display_labels=["Malignant", "Benign"],
                ax=axes[0], colorbar=False)
            axes[0].set_title("from_estimator()")

            # From predictions (use pre-computed y_preds)
            ConfusionMatrixDisplay.from_predictions(
                y_true=y_test, y_pred=y_preds,
                display_labels=["Malignant", "Benign"],
                ax=axes[1], colorbar=False)
            axes[1].set_title("from_predictions()")

            plt.tight_layout()
            plt.savefig("confusion_matrices.png", dpi=100)
            plt.show()
            print("\\nConfusion matrices saved to confusion_matrices.png")
            print("\\nIdeal matrix: large numbers on the diagonal (TN, TP)")
            print("Off-diagonal cells are mistakes: FP (top-right), FN (bottom-left)")
        ''',
    },

    # ── 12 ─────────────────────────────────────────────────────────────────
    "12 · Classification Metrics — Classification Report & Class Imbalance": {
        "description": (
            "The classification report summarises precision, recall, and F1 for "
            "every class. This operation also demonstrates WHY accuracy is misleading "
            "for imbalanced classes — a model can be 99.99% 'accurate' while being "
            "completely useless."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (classification_report, accuracy_score,
                                          precision_score, recall_score, f1_score)

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_preds = clf.predict(X_test)

            # ── Full classification report ────────────────────────────────
            print("Classification Report:")
            print(classification_report(y_test, y_preds,
                                        target_names=["Malignant", "Benign"]))

            # ── Individual metric functions ───────────────────────────────
            print("Individual metric functions:")
            print(f"  Accuracy : {accuracy_score(y_test, y_preds)*100:.2f}%")
            print(f"  Precision: {precision_score(y_test, y_preds):.4f}")
            print(f"  Recall   : {recall_score(y_test, y_preds):.4f}")
            print(f"  F1 Score : {f1_score(y_test, y_preds):.4f}")

            # ── Class imbalance problem ───────────────────────────────────
            print("\\n" + "="*60)
            print("CLASS IMBALANCE DEMONSTRATION")
            print("="*60)
            print("Imagine 10,000 people. Only 1 has a rare disease.")
            print("A model that predicts 'no disease' for everyone is 99.99% accurate.")
            print("But is it useful?\\n")

            disease_true  = np.zeros(10_000, dtype=int)
            disease_true[0] = 1                          # only 1 positive case
            disease_preds = np.zeros(10_000, dtype=int)  # model always predicts 0

            report = classification_report(disease_true, disease_preds,
                                           output_dict=True, zero_division=0)
            print(pd.DataFrame(report).T.round(3))
            print(f"\\nAccuracy: {accuracy_score(disease_true, disease_preds)*100:.2f}%")
            print("\\n→ Perfect accuracy, but recall for class 1 = 0.0")
            print("→ The model missed the ONLY sick patient.")
            print("\\nWhen to use each metric:")
            print("  • Accuracy  → classes are balanced")
            print("  • Precision → false positives are costly (spam filter)")
            print("  • Recall    → false negatives are costly (disease screening)")
            print("  • F1        → both FP and FN matter, imbalanced classes")
        ''',
    },

    # ── 13 ─────────────────────────────────────────────────────────────────
    "13 · Regression Metrics — R², MAE, MSE": {
        "description": (
            "Three complementary regression metrics: R² for an overall quality score, "
            "MAE for interpretable average error in the same units as y, and MSE "
            "for penalising large errors more heavily. Demonstrated on California Housing."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.datasets import load_diabetes
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

            # ── Load the Diabetes dataset (built-in, no download) ─────────
            data_reg = load_diabetes()
            X = data_reg["data"]
            y = data_reg["target"]   # disease progression score (numeric)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_preds = model.predict(X_test)

            # ── R² score ─────────────────────────────────────────────────
            r2 = r2_score(y_test, y_preds)
            # What would a model that just predicts the mean score?
            y_mean_pred = np.full(len(y_test), y_test.mean())
            r2_mean_model = r2_score(y_test, y_mean_pred)
            r2_perfect    = r2_score(y_test, y_test)
            print(f"R² (our model)   : {r2:.4f}")
            print(f"R² (mean model)  : {r2_mean_model:.4f}  ← always 0")
            print(f"R² (perfect)     : {r2_perfect:.4f}  ← always 1")

            # ── MAE ──────────────────────────────────────────────────────
            mae = mean_absolute_error(y_test, y_preds)
            print(f"\\nMAE : {mae:.4f}  (average absolute error in score units)")

            # ── MSE ──────────────────────────────────────────────────────
            mse = mean_squared_error(y_test, y_preds)
            print(f"MSE : {mse:.4f}  (amplifies large errors via squaring)")
            print(f"RMSE: {np.sqrt(mse):.4f} (MSE in original units)")

            # ── Comparison table ──────────────────────────────────────────
            y_test_arr = np.array(y_test)
            df_preds = pd.DataFrame({
                "actual":     y_test_arr[:10],
                "predicted":  y_preds[:10],
                "difference": y_preds[:10] - y_test_arr[:10],
                "abs_diff":   np.abs(y_preds[:10] - y_test_arr[:10]),
                "sq_diff":    (y_preds[:10] - y_test_arr[:10]) ** 2,
            }).round(2)
            print("\\nFirst 10 predictions vs actuals:")
            print(df_preds.to_string())

            # ── Scatter plot ──────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 4))
            x_idx = np.arange(len(df_preds))
            ax.scatter(x_idx, df_preds["actual"],    label="Actual",    color="steelblue")
            ax.scatter(x_idx, df_preds["predicted"], label="Predicted", color="darkorange", marker="x", s=80)
            ax.set_xlabel("Sample index"); ax.set_ylabel("Disease Progression Score")
            ax.set_title("Actual vs Predicted (first 10 test samples)"); ax.legend()
            plt.tight_layout()
            plt.savefig("regression_predictions.png", dpi=100)
            plt.show()
            print("\\nPlot saved to regression_predictions.png")
        ''',
    },

    # ── 14 ─────────────────────────────────────────────────────────────────
    "14 · Using the scoring Parameter with cross_val_score": {
        "description": (
            "cross_val_score's scoring parameter lets you specify any metric name "
            "instead of using the estimator's default. This allows direct comparison "
            "of multiple metrics in a single cross-validated experiment."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            from sklearn.datasets import load_breast_cancer, load_diabetes
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # ── Classification scoring ────────────────────────────────────
            print("CLASSIFICATION SCORING (Breast Cancer)")
            print("=" * 45)
            data = load_breast_cancer()
            X_clf, y_clf = data["data"], data["target"]
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                np.random.seed(42)
                scores = cross_val_score(clf, X_clf, y_clf, cv=5, scoring=metric)
                print(f"  {metric:<12}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

            # ── Regression scoring ────────────────────────────────────────
            print("\\nREGRESSION SCORING (Diabetes dataset)")
            print("=" * 45)
            d_reg = load_diabetes()
            X_reg, y_reg = d_reg["data"], d_reg["target"]
            model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Note the "neg_" prefix: sklearn always maximises, so errors are negated
            for metric in ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]:
                np.random.seed(42)
                scores = cross_val_score(model, X_reg, y_reg, cv=3, scoring=metric)
                mean = np.mean(scores)
                print(f"  {metric:<30}: {mean:.4f}")
                if "neg_" in metric:
                    print(f"    → actual value (negated back): {-mean:.4f}")

            print("\\nNote: 'neg_mean_absolute_error' returns negative values.")
            print("Closer to 0 = better (sklearn convention: higher is always better).")
        ''',
    },

    # ── 15 ─────────────────────────────────────────────────────────────────
    "15 · Hyperparameter Tuning by Hand": {
        "description": (
            "Manual tuning: create a three-way train/validation/test split, "
            "establish a baseline model, then try different hyperparameters on "
            "the validation set. An evaluate_preds() helper makes repeated "
            "comparison of metrics easy."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_breast_cancer
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            def evaluate_preds(y_true, y_preds, label=""):
                """Print and return a dict of classification metrics."""
                acc  = accuracy_score(y_true, y_preds)
                prec = precision_score(y_true, y_preds, zero_division=0)
                rec  = recall_score(y_true, y_preds, zero_division=0)
                f1   = f1_score(y_true, y_preds, zero_division=0)
                if label:
                    print(f"--- {label} ---")
                print(f"  Accuracy : {acc*100:.2f}%")
                print(f"  Precision: {prec:.4f}")
                print(f"  Recall   : {rec:.4f}")
                print(f"  F1 Score : {f1:.4f}")
                return {"accuracy": round(acc, 4), "precision": round(prec, 4),
                        "recall": round(rec, 4), "f1": round(f1, 4)}

            # ── Three-way split: train / validation / test ────────────────
            data = load_breast_cancer()
            np.random.seed(42)
            idx = np.random.permutation(len(data["data"]))
            X, y = data["data"][idx], data["target"][idx]

            n       = len(X)
            n_train = int(0.70 * n)   # 70%
            n_valid = int(0.15 * n)   # 15%
            # remaining 15% → test

            X_train, y_train = X[:n_train], y[:n_train]
            X_valid, y_valid = X[n_train:n_train+n_valid], y[n_train:n_train+n_valid]
            X_test,  y_test  = X[n_train+n_valid:], y[n_train+n_valid:]
            print(f"Split sizes — Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

            # ── Baseline: default hyperparameters ────────────────────────
            clf_base = RandomForestClassifier(random_state=42)
            clf_base.fit(X_train, y_train)
            baseline = evaluate_preds(y_valid, clf_base.predict(X_valid), "Baseline (defaults)")

            # ── Attempt 2: increase n_estimators ─────────────────────────
            clf_2 = RandomForestClassifier(n_estimators=200, random_state=42)
            clf_2.fit(X_train, y_train)
            metrics_2 = evaluate_preds(y_valid, clf_2.predict(X_valid), "n_estimators=200")

            # ── Attempt 3: limit depth ────────────────────────────────────
            clf_3 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            clf_3.fit(X_train, y_train)
            metrics_3 = evaluate_preds(y_valid, clf_3.predict(X_valid), "n_estimators=200, max_depth=10")

            # ── Compare all ───────────────────────────────────────────────
            print("\\n--- Final comparison on validation set ---")
            compare = pd.DataFrame({
                "baseline":   baseline,
                "n_est=200":  metrics_2,
                "depth=10":   metrics_3,
            })
            print(compare)
            print("\\n(Do NOT evaluate on test set yet — save it for final model only)")
        ''',
    },

    # ── 16 ─────────────────────────────────────────────────────────────────
    "16 · Hyperparameter Tuning — RandomizedSearchCV": {
        "description": (
            "RandomizedSearchCV randomly samples combinations from a hyperparameter "
            "grid, evaluates each with cross-validation, and returns the best. "
            "It is much faster than exhaustive search and often finds near-optimal "
            "settings in a fraction of the time."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split, RandomizedSearchCV
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            def evaluate_preds(y_true, y_preds):
                return {
                    "accuracy":  round(accuracy_score(y_true, y_preds), 4),
                    "precision": round(precision_score(y_true, y_preds, zero_division=0), 4),
                    "recall":    round(recall_score(y_true, y_preds, zero_division=0), 4),
                    "f1":        round(f1_score(y_true, y_preds, zero_division=0), 4),
                }

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            np.random.seed(42)
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                  test_size=0.2,
                                                                  random_state=42)

            # ── Define the hyperparameter search space ─────────────────────
            # These are the values RandomizedSearchCV will sample from
            param_grid = {
                "n_estimators":    [10, 50, 100, 200, 500],
                "max_depth":       [None, 5, 10, 20],
                "max_features":    ["sqrt", "log2"],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf":  [1, 2, 4],
            }
            # Total combinations: 5×4×2×3×3 = 360 — we'll try 20 randomly

            clf = RandomForestClassifier(random_state=42, n_jobs=-1)

            rs_clf = RandomizedSearchCV(
                estimator=clf,
                param_distributions=param_grid,
                n_iter=20,    # number of random combinations to try
                cv=5,         # 5-fold cross-validation for each
                verbose=1,
                random_state=42,
                n_jobs=-1,
            )

            print(f"Searching {20} of {5*4*2*3*3} possible combinations...")
            rs_clf.fit(X_train, y_train)

            print(f"\\nBest parameters found: {rs_clf.best_params_}")
            print(f"Best CV score        : {rs_clf.best_score_*100:.2f}%")

            # Calling predict() on rs_clf uses the best estimator automatically
            rs_preds = rs_clf.predict(X_test)
            rs_metrics = evaluate_preds(y_test, rs_preds)

            # Baseline for comparison
            baseline = RandomForestClassifier(random_state=42)
            baseline.fit(X_train, y_train)
            base_metrics = evaluate_preds(y_test, baseline.predict(X_test))

            print("\\nComparison:")
            import pandas as pd
            print(pd.DataFrame({"baseline": base_metrics, "RandomizedSearchCV": rs_metrics}))
        ''',
    },

    # ── 17 ─────────────────────────────────────────────────────────────────
    "17 · Hyperparameter Tuning — GridSearchCV": {
        "description": (
            "GridSearchCV exhaustively tries every combination in a grid. Use it "
            "after RandomizedSearchCV has narrowed the search space — the refined "
            "grid has far fewer combinations but thoroughly covers the promising region. "
            "Includes a bar chart comparing all tuning approaches."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            def evaluate_preds(y_true, y_preds, label=""):
                m = {
                    "accuracy":  round(accuracy_score(y_true, y_preds), 4),
                    "precision": round(precision_score(y_true, y_preds, zero_division=0), 4),
                    "recall":    round(recall_score(y_true, y_preds, zero_division=0), 4),
                    "f1":        round(f1_score(y_true, y_preds, zero_division=0), 4),
                }
                if label:
                    print(f"{label}: accuracy={m['accuracy']*100:.2f}%  f1={m['f1']:.4f}")
                return m

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            np.random.seed(42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # ── Baseline ───────────────────────────────────────────────────
            base = RandomForestClassifier(random_state=42).fit(X_train, y_train)
            base_m = evaluate_preds(y_test, base.predict(X_test), "Baseline")

            # ── RandomizedSearch (for reference) ──────────────────────────
            rs = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                    {"n_estimators": [50, 100, 200],
                                     "max_depth": [None, 10, 20],
                                     "min_samples_leaf": [1, 2]},
                                    n_iter=10, cv=5, random_state=42)
            rs.fit(X_train, y_train)
            rs_m = evaluate_preds(y_test, rs.predict(X_test), "RandomizedSearchCV")
            print(f"  Best params: {rs.best_params_}")

            # ── GridSearch: refined grid near the random search result ─────
            # 3×2×2×2 = 24 combinations — manageable
            grid_2 = {
                "n_estimators":    [100, 200, 500],
                "max_depth":       [None, 10],
                "max_features":    ["sqrt", "log2"],
                "min_samples_leaf": [1, 2],
            }
            gs = GridSearchCV(RandomForestClassifier(random_state=42),
                              grid_2, cv=5, verbose=1)
            gs.fit(X_train, y_train)
            gs_m = evaluate_preds(y_test, gs.predict(X_test), "GridSearchCV")
            print(f"  Best params: {gs.best_params_}")

            # ── Compare all approaches ────────────────────────────────────
            compare = pd.DataFrame({
                "Baseline": base_m,
                "RandomizedCV": rs_m,
                "GridSearchCV": gs_m,
            })
            print("\\nFull comparison:")
            print(compare)

            compare.plot.bar(figsize=(10, 5), rot=0)
            plt.title("Hyperparameter Tuning — Metric Comparison")
            plt.ylabel("Score"); plt.ylim(0.8, 1.01)
            plt.tight_layout()
            plt.savefig("hyperparameter_comparison.png", dpi=100)
            plt.show()
            print("\\nComparison chart saved to hyperparameter_comparison.png")
        ''',
    },

    # ── 18 ─────────────────────────────────────────────────────────────────
    "18 · Saving and Loading Models — pickle & joblib": {
        "description": (
            "Once a model is trained, serialise it to disk so it can be reloaded "
            "without retraining. Python's pickle module and Scikit-Learn's preferred "
            "joblib both achieve this — joblib is more efficient for large NumPy arrays."
        ),
        "language": "python",
        "code": '''
            import pickle
            import numpy as np
            import tempfile, os
            from joblib import dump, load
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            data = load_breast_cancer()
            X, y = data["data"], data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Train the model we want to save
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            original_score = accuracy_score(y_test, clf.predict(X_test))
            print(f"Trained model accuracy: {original_score*100:.2f}%")

            tmpdir = tempfile.mkdtemp()

            # ── Method 1: pickle ─────────────────────────────────────────
            pkl_path = os.path.join(tmpdir, "model.pkl")
            pickle.dump(clf, open(pkl_path, "wb"))          # wb = write binary
            print(f"\\nSaved with pickle to: {pkl_path}")
            print(f"File size: {os.path.getsize(pkl_path) / 1024:.1f} KB")

            loaded_pickle = pickle.load(open(pkl_path, "rb"))  # rb = read binary
            pkl_score = accuracy_score(y_test, loaded_pickle.predict(X_test))
            print(f"Loaded pickle model accuracy: {pkl_score*100:.2f}%  (same as original: {pkl_score == original_score})")

            # ── Method 2: joblib ─────────────────────────────────────────
            jbl_path = os.path.join(tmpdir, "model.joblib")
            dump(clf, jbl_path)
            print(f"\\nSaved with joblib to: {jbl_path}")
            print(f"File size: {os.path.getsize(jbl_path) / 1024:.1f} KB")

            loaded_joblib = load(jbl_path)
            jbl_score = accuracy_score(y_test, loaded_joblib.predict(X_test))
            print(f"Loaded joblib model accuracy: {jbl_score*100:.2f}%  (same as original: {jbl_score == original_score})")

            # ── Which to use? ─────────────────────────────────────────────
            print("\\n--- pickle vs joblib ---")
            print("pickle  : built-in, works for any Python object")
            print("joblib  : more efficient for large NumPy arrays (sklearn models)")
            print("         Scikit-Learn documentation recommends joblib")
            print("Either  : never load models from untrusted sources (security risk)")
        ''',
    },

    # ── 19 ─────────────────────────────────────────────────────────────────
    "19 · Putting It All Together — sklearn Pipeline": {
        "description": (
            "A Pipeline chains preprocessing (imputation, encoding) and modelling "
            "into a single estimator. The code is compact, leakage-proof, and "
            "compatible with GridSearchCV. This is the production-ready pattern "
            "for any sklearn project."
        ),
        "language": "python",
        "code": '''
            import numpy as np
            import pandas as pd
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split, GridSearchCV

            np.random.seed(42)

            # ── Simulate the car sales dataset with missing values ────────
            n = 200
            df = pd.DataFrame({
                "Make":   np.random.choice(["Toyota", "Honda", None, "Ford"], n, p=[0.3,0.3,0.1,0.3]),
                "Colour": np.random.choice(["Red", None, "White", "Blue"], n, p=[0.3,0.1,0.3,0.3]),
                "Doors":  [np.random.choice([2, 4, None], p=[0.3,0.6,0.1]) for _ in range(n)],
                "Odometer (KM)": [np.random.randint(10_000, 200_000) if np.random.rand() > 0.08
                                   else None for _ in range(n)],
                "Price":  np.random.randint(5_000, 50_000, n),
            })

            # Drop rows with missing target (can't invent labels)
            df.dropna(subset=["Price"], inplace=True)

            X = df.drop("Price", axis=1)
            y = df["Price"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            print(f"Training samples: {len(X_train)}  Test samples: {len(X_test)}")
            print(f"Missing values in X_train:\\n{X_train.isna().sum().to_dict()}")

            # ── Build sub-pipelines for each feature group ────────────────
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot",  OneHotEncoder(handle_unknown="ignore")),
            ])
            door_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=4)),
            ])
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
            ])

            # ── Combine into a ColumnTransformer preprocessor ────────────
            preprocessor = ColumnTransformer(transformers=[
                ("cat",  categorical_transformer, ["Make", "Colour"]),
                ("door", door_transformer,        ["Doors"]),
                ("num",  numeric_transformer,     ["Odometer (KM)"]),
            ])

            # ── Full Pipeline: preprocessing + model ──────────────────────
            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model",        RandomForestRegressor(random_state=42)),
            ])

            model.fit(X_train, y_train)
            print(f"\\nPipeline R² score (test): {model.score(X_test, y_test):.4f}")

            # ── GridSearchCV with Pipeline ────────────────────────────────
            # Prefix hyperparameter names with the step name + double underscore
            pipe_grid = {
                "preprocessor__num__imputer__strategy": ["mean", "median"],
                "model__n_estimators":   [50, 100],
                "model__max_depth":      [None, 10],
                "model__min_samples_split": [2, 4],
            }

            gs_model = GridSearchCV(model, pipe_grid, cv=3, verbose=1)
            gs_model.fit(X_train, y_train)

            print(f"\\nGridSearchCV best params: {gs_model.best_params_}")
            print(f"GridSearchCV best CV score: {gs_model.best_score_:.4f}")
            print(f"GridSearchCV test  score  : {gs_model.score(X_test, y_test):.4f}")
            print("\\nThe entire Pipeline — preprocessing + model — was tuned together.")
            print("No data leakage. No manual bookkeeping. Fully reproducible.")
        ''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
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
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 600,
        "complexity":    None,
        "operations":    OPERATIONS,
    }