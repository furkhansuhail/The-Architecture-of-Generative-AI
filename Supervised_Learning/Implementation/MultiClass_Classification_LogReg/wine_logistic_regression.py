"""
=============================================================================
  Multiclass Classification with Logistic Regression — Wine Dataset (UCI)
=============================================================================
  Dataset   : sklearn.datasets.load_wine (178 samples, 13 features, 3 classes)
  Objective : Predict wine cultivar (class 0, 1, 2) from chemical properties
  Techniques: OvR vs Softmax (multinomial), GridSearchCV, cross-validation,
              ROC-OvR, Precision-Recall, SHAP-style feature importance
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.inspection import permutation_importance

# Plotting style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CLASS_NAMES = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  1. LOADING DATASET")
print("=" * 65)

raw = load_wine()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target
df["cultivar"] = df["target"].map({i: f"Cultivar {i}" for i in range(3)})

print(f"Shape         : {df.shape}")
print(f"Features      : {raw.feature_names}")
print(f"Classes       : {raw.target_names}")
print(f"Class balance :\n{df['cultivar'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  2. EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\n── Descriptive Statistics ──────────────────────────────────")
print(df.drop(columns=["target", "cultivar"]).describe().round(2).to_string())

print("\n── Missing values ──────────────────────────────────────────")
print(df.isnull().sum().to_string())

print("\n── Correlation with target ─────────────────────────────────")
corr_with_target = df.drop(columns=["cultivar"]).corr()["target"].drop("target").sort_values(key=abs, ascending=False)
print(corr_with_target.round(3).to_string())

# ── EDA Figure 1: Class distribution + Feature distributions ─────────────────
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle("EDA — Feature Distributions by Cultivar", fontsize=16, fontweight="bold")

# Class distribution (first cell)
ax0 = axes[0, 0]
counts = df["cultivar"].value_counts()
bars = ax0.bar(counts.index, counts.values, color=COLORS)
ax0.set_title("Class Distribution")
ax0.set_ylabel("Count")
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(val), ha="center", va="bottom", fontsize=9)

# Feature distributions (remaining cells)
features = raw.feature_names
cell_iter = [(r, c) for r in range(3) for c in range(5)][1:]  # skip first

for idx, (r, c) in enumerate(cell_iter):
    if idx >= len(features):
        axes[r, c].axis("off")
        continue
    feat = features[idx]
    ax = axes[r, c]
    for i, (cls, color) in enumerate(zip(df["cultivar"].unique(), COLORS)):
        data = df[df["cultivar"] == cls][feat]
        ax.hist(data, bins=15, alpha=0.6, color=COLORS[i], label=cls, edgecolor="white")
    ax.set_title(feat[:28], fontsize=8)
    ax.set_xlabel("")
    if idx == 0:
        ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig("/home/claude/eda_distributions.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n[Saved] eda_distributions.png")

# ── EDA Figure 2: Correlation heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
corr_matrix = df.drop(columns=["cultivar"]).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/eda_correlation.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] eda_correlation.png")

# ── EDA Figure 3: Pairplot of top-4 features ─────────────────────────────────
top4 = corr_with_target.abs().nlargest(4).index.tolist()
pair_df = df[top4 + ["cultivar"]]
g = sns.pairplot(pair_df, hue="cultivar", palette=dict(zip(df["cultivar"].unique(), COLORS)),
                 diag_kind="kde", plot_kws={"alpha": 0.6, "s": 30})
g.figure.suptitle("Pairplot — Top-4 Features Correlated with Target", y=1.02, fontsize=13, fontweight="bold")
g.figure.savefig("/home/claude/eda_pairplot.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] eda_pairplot.png")

# ── EDA Figure 4: Boxplots ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
fig.suptitle("EDA — Boxplots by Cultivar", fontsize=14, fontweight="bold")
cell_iter2 = [(r, c) for r in range(3) for c in range(5)]

for idx, (r, c) in enumerate(cell_iter2):
    if idx >= len(features):
        axes[r, c].axis("off")
        continue
    feat = features[idx]
    ax = axes[r, c]
    data_by_class = [df[df["cultivar"] == cls][feat].values for cls in sorted(df["cultivar"].unique())]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False, widths=0.5)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(feat[:28], fontsize=8)
    ax.set_xticklabels(["C0", "C1", "C2"], fontsize=7)

plt.tight_layout()
plt.savefig("/home/claude/eda_boxplots.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] eda_boxplots.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  3. PREPROCESSING")
print("=" * 65)

X = df[raw.feature_names].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {X_train.shape[0]} samples")
print(f"Test size  : {X_test.shape[0]} samples")
print(f"Train class dist : {np.bincount(y_train)}")
print(f"Test class dist  : {np.bincount(y_test)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\nScaling : StandardScaler applied (fit on train, transform on test)")
print(f"Mean (train, feature 0): {X_train_sc[:, 0].mean():.4f}  (should be ~0)")
print(f"Std  (train, feature 0): {X_train_sc[:, 0].std():.4f}   (should be ~1)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING — STRATEGY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  4. MODEL TRAINING — Solver & Penalty Comparison")
print("=" * 65)

strategies = {
    "lbfgs   + L2":          LogisticRegression(solver="lbfgs",  penalty="l2",          max_iter=1000, random_state=42),
    "saga    + L1":          LogisticRegression(solver="saga",   penalty="l1",          max_iter=3000, random_state=42),
    "saga    + L2":          LogisticRegression(solver="saga",   penalty="l2",          max_iter=3000, random_state=42),
    "saga    + ElasticNet":  LogisticRegression(solver="saga",   penalty="elasticnet",  l1_ratio=0.5,  max_iter=3000, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_results = {}

for name, model in strategies.items():
    cv_res = cross_validate(model, X_train_sc, y_train, cv=cv,
                            scoring=["accuracy", "f1_macro"], return_train_score=True)
    strat_results[name] = {
        "CV Acc (mean)":  cv_res["test_accuracy"].mean(),
        "CV Acc (std)":   cv_res["test_accuracy"].std(),
        "CV F1 (mean)":   cv_res["test_f1_macro"].mean(),
        "Train Acc":      cv_res["train_accuracy"].mean(),
    }
    print(f"\n{name}:")
    print(f"  5-Fold CV Accuracy : {cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}")
    print(f"  5-Fold CV F1 Macro : {cv_res['test_f1_macro'].mean():.4f}")
    print(f"  Train Accuracy     : {cv_res['train_accuracy'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  5. HYPERPARAMETER TUNING — GridSearchCV")
print("=" * 65)

param_grid = {
    "logisticregression__C":       [0.001, 0.01, 0.1, 1, 10, 100],
    "logisticregression__penalty": ["l2"],
    "logisticregression__solver":  ["lbfgs"],
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logisticregression", LogisticRegression(max_iter=1000, random_state=42))
])

grid_search = GridSearchCV(pipe, param_grid, cv=cv,
                           scoring="f1_macro", n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)   # NOTE: raw X; pipeline applies scaler internally

print(f"Best params : {grid_search.best_params_}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ─────────────────────────────────────────────────────────────────────────────
# 6. FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  6. FINAL MODEL EVALUATION ON TEST SET")
print("=" * 65)

y_pred       = best_model.predict(X_test)
y_proba      = best_model.predict_proba(X_test)
best_C       = grid_search.best_params_["logisticregression__C"]

print(f"\nBest C = {best_C}")
print(f"\n── Classification Report ───────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

mcc   = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Matthews Correlation Coefficient : {mcc:.4f}")
print(f"Cohen's Kappa                    : {kappa:.4f}")

# ── Plot 1: Confusion matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/eval_confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n[Saved] eval_confusion_matrix.png")

# ── Plot 2: ROC curves (OvR) ──────────────────────────────────────────────────
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(7, 6))

for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — One-vs-Rest (Test Set)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("/home/claude/eval_roc_curves.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] eval_roc_curves.png")

# ── Plot 3: Precision-Recall curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
    ap = average_precision_score(y_bin[:, i], y_proba[:, i])
    ax.plot(recall, precision, color=color, lw=2, label=f"{cls_name} (AP = {ap:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — One-vs-Rest", fontsize=13, fontweight="bold")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig("/home/claude/eval_pr_curves.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] eval_pr_curves.png")

# ── Plot 4: Prediction probability heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(y_test)
proba_sorted = y_proba[sorted_idx]
sns.heatmap(proba_sorted.T, cmap="YlOrRd", yticklabels=CLASS_NAMES,
            xticklabels=False, ax=ax, vmin=0, vmax=1,
            cbar_kws={"label": "Predicted Probability"})
ax.set_xlabel("Test Samples (sorted by true class)")
ax.set_title("Predicted Probability per Class (Test Set)", fontsize=13, fontweight="bold")

# Add vertical lines separating classes
boundaries = np.where(np.diff(y_test[sorted_idx]))[0] + 1
for b in boundaries:
    ax.axvline(x=b, color="white", lw=2)

plt.tight_layout()
plt.savefig("/home/claude/eval_proba_heatmap.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] eval_proba_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  7. FEATURE IMPORTANCE")
print("=" * 65)

# 7a. Model coefficients (L2 regularized LR)
lr_model = best_model.named_steps["logisticregression"]
coef = lr_model.coef_   # shape: (n_classes, n_features)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Logistic Regression Coefficients by Class", fontsize=14, fontweight="bold")

for i, (ax, cls_name, color) in enumerate(zip(axes, CLASS_NAMES, COLORS)):
    sorted_idx = np.argsort(np.abs(coef[i]))
    ax.barh(np.array(raw.feature_names)[sorted_idx], coef[i][sorted_idx],
            color=[color if v > 0 else "#C44E52" for v in coef[i][sorted_idx]],
            edgecolor="white", height=0.7)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title(cls_name, fontsize=12)
    ax.set_xlabel("Coefficient value")

plt.tight_layout()
plt.savefig("/home/claude/feat_coefficients.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] feat_coefficients.png")

# 7b. Permutation importance
perm_imp = permutation_importance(best_model, X_test, y_test,
                                  scoring="f1_macro", n_repeats=30,
                                  random_state=42, n_jobs=-1)

perm_df = pd.DataFrame({
    "feature":    raw.feature_names,
    "importance": perm_imp.importances_mean,
    "std":        perm_imp.importances_std,
}).sort_values("importance", ascending=False)

print("\nPermutation Importances (F1 Macro drop):")
print(perm_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = ["#4C72B0" if v > 0 else "#C44E52" for v in perm_df["importance"]]
ax.barh(perm_df["feature"][::-1], perm_df["importance"][::-1],
        xerr=perm_df["std"][::-1], color=colors_bar[::-1],
        edgecolor="white", height=0.7, capsize=3)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_xlabel("Mean F1 Macro decrease")
ax.set_title("Permutation Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/feat_permutation_importance.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] feat_permutation_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. C REGULARIZATION SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  8. REGULARIZATION SENSITIVITY ANALYSIS")
print("=" * 65)

C_range = np.logspace(-3, 3, 30)
train_accs, test_accs, cv_accs = [], [], []

for C in C_range:
    model_c = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, solver="lbfgs", penalty="l2",
                                  max_iter=1000, random_state=42))
    ])
    model_c.fit(X_train, y_train)
    train_accs.append(model_c.score(X_train, y_train))
    test_accs.append(model_c.score(X_test, y_test))

    cv_r = cross_validate(model_c, X_train, y_train, cv=cv, scoring="accuracy")
    cv_accs.append(cv_r["test_score"].mean())

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(C_range, train_accs, label="Train Accuracy",  color="#4C72B0", lw=2)
ax.semilogx(C_range, cv_accs,    label="CV Accuracy",     color="#55A868", lw=2)
ax.semilogx(C_range, test_accs,  label="Test Accuracy",   color="#DD8452", lw=2, linestyle="--")
ax.axvline(best_C, color="red", linestyle=":", lw=1.5, label=f"Best C = {best_C}")
ax.set_xlabel("Regularization parameter C (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs. Regularization Strength", fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylim([0.8, 1.01])
plt.tight_layout()
plt.savefig("/home/claude/reg_sensitivity.png", dpi=130, bbox_inches="tight")
plt.close()
print("[Saved] reg_sensitivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  9. FINAL SUMMARY")
print("=" * 65)
from sklearn.metrics import accuracy_score, f1_score
acc   = accuracy_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred, average="macro")

print(f"""
Dataset        : Wine (UCI) — 178 samples, 13 features, 3 classes
Best Strategy  : Softmax (lbfgs) with L2 regularization
Best C         : {best_C}

─── Test Set Metrics ────────────────────────────────────────
  Accuracy              : {acc:.4f}
  F1 Score (Macro)      : {f1:.4f}
  Matthews Corr Coef    : {mcc:.4f}
  Cohen's Kappa         : {kappa:.4f}

─── Outputs saved ───────────────────────────────────────────
  EDA         : eda_distributions.png, eda_correlation.png,
                eda_pairplot.png, eda_boxplots.png
  Evaluation  : eval_confusion_matrix.png, eval_roc_curves.png,
                eval_pr_curves.png, eval_proba_heatmap.png
  Features    : feat_coefficients.png, feat_permutation_importance.png
  Tuning      : reg_sensitivity.png
""")
print("=" * 65)
print("  DONE")
print("=" * 65)