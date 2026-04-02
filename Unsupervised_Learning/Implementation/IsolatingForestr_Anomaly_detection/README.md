# 🔍 Isolation Forest — Anomaly Detection Project

> **Difficulty:** Intermediate → Advanced  
> **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle) — synthetic fallback included

---

## 📁 Project Structure

```
isolation_forest_project/
│
├── config.py           ← All hyper-params, paths, flags (start here)
├── data_loader.py      ← Stage 1 · Load Kaggle CSV or generate synthetic data
├── eda.py              ← Stage 2 · Exploratory Data Analysis (4 plot sets)
├── preprocessor.py     ← Stage 3 · RobustScaler + stratified train/test split
├── model.py            ← Stage 4 · IsolationForest + grid-search tuning
├── evaluator.py        ← Stage 5 · Full metric suite (AUC, F1, MCC, P@K …)
├── visualizer.py       ← Stage 6 · 6 rich result plots incl. t-SNE
├── reporter.py         ← Stage 7 · HTML + JSON run report
├── main.py             ← Pipeline orchestrator
│
├── requirements.txt
├── data/               ← Place creditcard.csv here (optional)
├── models/             ← Saved model + scaler (auto-created)
├── outputs/            ← All plots (auto-created)
└── reports/            ← HTML report + JSON summary + run.log (auto-created)
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run (synthetic data — no download needed)
python main.py
```

### Using the real Kaggle dataset

```bash
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in  data/

# Then set in config.py:
USE_SYNTHETIC = False

python main.py
```

---

## 🧠 How Isolation Forest Works

```
Normal point  →  takes MANY splits to isolate  →  long path  →  LOW score
Anomaly point →  takes FEW  splits to isolate  →  short path →  HIGH score
```

| Step | Detail |
|------|--------|
| Build ensemble of random iTrees | Each tree uses random feature + random split |
| Average path length | Shorter path ⟹ easier to isolate ⟹ anomaly |
| `decision_function` | Outputs normalised anomaly score |
| `contamination` | Controls the score threshold for `predict` |

---

## 📊 Pipeline Stages

### Stage 1 — Data Loader (`data_loader.py`)
- Loads Kaggle CSV **or** auto-generates a realistic synthetic fraud dataset
- Logs class balance and dataset shape

### Stage 2 — EDA (`eda.py`)
| Plot | File |
|------|------|
| Class distribution bar + pie | `01_class_distribution.png` |
| Amount & Time by class | `02_amount_time.png` |
| Feature distributions (V1–V28) | `03_feature_distributions.png` |
| Correlation heat-map (top 15) | `04_correlation_heatmap.png` |

### Stage 3 — Pre-processing (`preprocessor.py`)
- Drops near-zero-variance columns
- **RobustScaler** on `Amount` and `Time` (outlier-resistant)
- Stratified 80/20 train-test split
- Persists scaler to `models/scaler.pkl`

### Stage 4 — Model Training (`model.py`)
- Default: `IsolationForest(n_estimators=200, contamination=0.02)`
- **Grid Search** over `n_estimators`, `max_samples`, `contamination`, `max_features`
- Selects best combo by **ROC-AUC** on validation set
- Saves to `models/isolation_forest.pkl`

### Stage 5 — Evaluation (`evaluator.py`)
Metric categories:
- **Classification:** Accuracy, Precision, Recall, F1, MCC, Confusion Matrix
- **Ranking:** ROC-AUC, PR-AUC
- **Top-K:** Precision@50, @100, @200, @n_anomalies

### Stage 6 — Results Display (`visualizer.py`)
| Plot | File |
|------|------|
| Score distribution by class | `05_anomaly_score_distribution.png` |
| ROC + PR curves | `06_roc_pr_curves.png` |
| Confusion matrix | `07_confusion_matrix.png` |
| Amount vs anomaly score scatter | `08_top_anomalies_scatter.png` |
| Feature importance (permutation proxy) | `09_feature_importance_proxy.png` |
| t-SNE projection (label + score) | `10_tsne_projection.png` |

### Stage 7 — Report (`reporter.py`)
- `reports/report.html` — interactive HTML with all plots embedded
- `reports/summary.json` — machine-readable metrics + params
- `reports/run.log` — full pipeline log

---

## ⚙️ Configuration (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `USE_SYNTHETIC` | `True` | Use synthetic data (no download needed) |
| `N_SAMPLES` | 10 000 | Synthetic dataset size |
| `CONTAMINATION` | 0.02 | Expected anomaly fraction |
| `TUNE_MODEL` | `True` | Enable grid-search |
| `SCALE_FEATURES` | `True` | RobustScaler on Amount + Time |
| `TEST_SIZE` | 0.20 | Train/test split ratio |

---

## 🔬 Key Design Decisions

1. **RobustScaler over StandardScaler** — financial amounts have heavy tails; RobustScaler uses median/IQR and is not thrown off by the very anomalies we're trying to find.

2. **Negated `decision_function`** — `IsolationForest.decision_function` returns a score where *lower = more anomalous*. We negate it everywhere so *higher = more anomalous* (more intuitive for threshold setting and plotting).

3. **Semi-supervised grid search** — the model trains *unsupervised* but we rank hyper-param combos using the labelled validation set's ROC-AUC. This is a valid and common practice for anomaly detection benchmarking.

4. **Precision@K** — in real fraud systems, a human analyst reviews only the top-K flagged transactions. P@K directly measures operational value.

5. **Permutation feature importance** — IsolationForest has no native feature importances. We shuffle each feature independently and measure the change in mean anomaly score as a proxy.

---

## 📚 Further Reading

- [Isolation Forest paper (Liu et al., 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [sklearn IsolationForest docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
