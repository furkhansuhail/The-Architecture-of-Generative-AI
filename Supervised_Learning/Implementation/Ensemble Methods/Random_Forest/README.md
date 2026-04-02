# Random Forest — Bagging with Decision Trees

**Dataset**: Titanic (GitHub/Kaggle) · fallback → Breast Cancer Wisconsin (sklearn built-in)  
**Task**: Binary Classification

## Project Structure

```
rf_project/
├── main.py                        ← Entry point (orchestrates all steps)
├── config.py                      ← Central hyper-parameters & paths
├── requirements.txt
├── data/
│   └── data_loader.py             ← Step 1: Load, impute, encode, split
├── eda/
│   └── exploratory_analysis.py   ← Step 2: EDA plots & summary stats
├── model/
│   └── random_forest_model.py    ← Step 3: RF + Bagging, CV, grid search
├── results/
│   └── visualizer.py             ← Step 4: CM, ROC, feature importance, tree
└── utils/
    └── helpers.py                 ← Decorators, banners, bagging explainer
```

## How to Run

```bash
pip install -r requirements.txt

# Full pipeline (includes grid search)
python main.py

# Fast mode — skip grid search
python main.py --fast
```

## Output Files (saved to `outputs/`)

| File | Description |
|------|-------------|
| `01_class_balance.png`        | Target class distribution |
| `02_numeric_distributions.png`| Histograms for all features |
| `03_correlation_heatmap.png`  | Pearson correlation heatmap |
| `04_target_by_feature.png`    | Class breakdown by key features |
| `05_confusion_matrices.png`   | RF vs Bagging confusion matrices |
| `06_roc_curve.png`            | AUC-ROC comparison |
| `07_feature_importance.png`   | Mean Decrease in Impurity ranking |
| `08_oob_error_curve.png`      | OOB error vs number of trees |
| `09_single_tree.png`          | One decision tree from the forest |
| `model_report.txt`            | Full metrics text report |

## How Random Forest = Bagging + Decision Trees

```
For each of the B trees:
  1. Bootstrap sample (sample WITH replacement)  ← Bagging
  2. Grow a Decision Tree, but at each split
     consider only √p random features            ← RF innovation
  3. No pruning → low bias, high variance

Prediction: Majority vote across all B trees
OOB Score:  ~36.8% of rows per tree are unseen → free validation
```

## Results (Breast Cancer Wisconsin, 200 trees)

| Metric    | Random Forest | Bagging  |
|-----------|--------------|----------|
| Accuracy  | 95.6%        | 95.6%    |
| Precision | 95.9%        | 95.9%    |
| Recall    | 97.2%        | 97.2%    |
| F1-Score  | 96.6%        | 96.6%    |
| AUC-ROC   | 0.993        | 0.993    |
| CV Acc    | 96.0 ± 1.9%  | —        |
