"""
model.py - DBSCAN Training & Hyperparameter Tuning Module.

Features:
  - Train a DBSCAN model
  - Grid-search over (eps, min_samples) using silhouette score + anomaly recall
  - Predict anomalies (noise points → fraud candidates)
  - Full evaluation report vs ground-truth labels
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import warnings

import config


# ──────────────────────────────────────────────────────────────────────────────
class DBSCANModel:
    def __init__(self, eps: float = None, min_samples: int = None):
        self.eps         = eps         or config.DBSCAN_EPS
        self.min_samples = min_samples or config.DBSCAN_MIN_SAMPLES
        self.model: DBSCAN | None     = None
        self.cluster_labels: np.ndarray | None = None
        self.anomaly_preds : np.ndarray | None = None   # binary: 1=anomaly

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, X: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN and return raw cluster labels.
        Noise points are labelled -1 by DBSCAN.
        """
        print(f"[Model] Training DBSCAN  eps={self.eps}  "
              f"min_samples={self.min_samples}  metric={config.DBSCAN_METRIC}")
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=config.DBSCAN_METRIC,
            n_jobs=-1,
        )
        self.cluster_labels = self.model.fit_predict(X)
        self.anomaly_preds  = (self.cluster_labels == -1).astype(int)

        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise    = (self.cluster_labels == -1).sum()
        n_total    = len(self.cluster_labels)

        print(f"[Model] Clusters found  : {n_clusters}")
        print(f"[Model] Noise points    : {n_noise:,}  ({n_noise/n_total*100:.2f}%)")
        return self.cluster_labels

    # ── evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Compare DBSCAN noise points (anomaly predictions) against ground-truth
        fraud labels.
        """
        if self.anomaly_preds is None:
            raise RuntimeError("Call train() first.")

        y_pred = self.anomaly_preds

        # Silhouette score (on non-noise points)
        mask = self.cluster_labels != -1
        sil  = None
        if mask.sum() > 1 and len(np.unique(self.cluster_labels[mask])) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sil = silhouette_score(X[mask], self.cluster_labels[mask],
                                       sample_size=min(5000, mask.sum()),
                                       random_state=config.RANDOM_STATE)

        # Classification metrics (treating noise as fraud prediction)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score   (y_true, y_pred, zero_division=0)
        f1        = f1_score       (y_true, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except Exception:
            roc_auc = None

        cm = confusion_matrix(y_true, y_pred)

        results = {
            "eps"           : self.eps,
            "min_samples"   : self.min_samples,
            "n_clusters"    : len(set(self.cluster_labels)) -
                              (1 if -1 in self.cluster_labels else 0),
            "n_noise"       : int((self.cluster_labels == -1).sum()),
            "silhouette"    : round(sil, 4) if sil is not None else None,
            "precision"     : round(precision, 4),
            "recall"        : round(recall, 4),
            "f1_score"      : round(f1, 4),
            "roc_auc"       : round(roc_auc, 4) if roc_auc else None,
            "confusion_matrix": cm,
        }

        self._print_report(results, y_true)
        return results

    # ── hyperparameter tuning ─────────────────────────────────────────────────
    def tune(self, X: np.ndarray, y_true: np.ndarray,
             eps_range: tuple = None, min_samples_range: tuple = None,
             n_eps: int = 6, n_min: int = 4) -> pd.DataFrame:
        """
        Grid-search over eps × min_samples combinations.
        Ranks by F1 score (fraud recall weighted).
        """
        eps_low, eps_high   = eps_range or config.EPS_RANGE
        mns_low, mns_high   = min_samples_range or config.MIN_SAMPLES_RANGE

        eps_grid = np.linspace(eps_low, eps_high, n_eps).round(2)
        mns_grid = np.linspace(mns_low, mns_high, n_min, dtype=int)

        print(f"\n[Model] Hyperparameter Search:")
        print(f"  eps grid        : {eps_grid.tolist()}")
        print(f"  min_samples grid: {mns_grid.tolist()}")
        print(f"  Total runs      : {len(eps_grid) * len(mns_grid)}\n")

        records = []
        for eps in eps_grid:
            for mns in mns_grid:
                try:
                    m = DBSCAN(eps=float(eps), min_samples=int(mns),
                               metric=config.DBSCAN_METRIC, n_jobs=-1)
                    lbls = m.fit_predict(X)
                    preds = (lbls == -1).astype(int)

                    n_cl = len(set(lbls)) - (1 if -1 in lbls else 0)
                    n_ns = int((lbls == -1).sum())

                    f1  = f1_score(y_true, preds, zero_division=0)
                    rec = recall_score(y_true, preds, zero_division=0)
                    pre = precision_score(y_true, preds, zero_division=0)

                    records.append({
                        "eps": eps, "min_samples": mns,
                        "n_clusters": n_cl, "n_noise": n_ns,
                        "precision": round(pre, 4),
                        "recall"   : round(rec, 4),
                        "f1_score" : round(f1,  4),
                    })
                    print(f"  eps={eps:.2f}  min_s={mns:>3}  "
                          f"clusters={n_cl:>3}  noise={n_ns:>5}  "
                          f"F1={f1:.4f}  R={rec:.4f}  P={pre:.4f}")
                except Exception as e:
                    print(f"  eps={eps:.2f}  min_s={mns} → ERROR: {e}")

        tuning_df = (pd.DataFrame(records)
                     .sort_values("f1_score", ascending=False)
                     .reset_index(drop=True))

        best = tuning_df.iloc[0]
        print(f"\n[Model] ★  Best params → "
              f"eps={best['eps']}  min_samples={best['min_samples']}  "
              f"F1={best['f1_score']}")
        self.eps         = float(best["eps"])
        self.min_samples = int(best["min_samples"])
        return tuning_df

    # ── private ───────────────────────────────────────────────────────────────
    def _print_report(self, r: dict, y_true: np.ndarray):
        print("\n" + "=" * 55)
        print("  DBSCAN EVALUATION REPORT")
        print("=" * 55)
        print(f"  eps             : {r['eps']}")
        print(f"  min_samples     : {r['min_samples']}")
        print(f"  Clusters found  : {r['n_clusters']}")
        print(f"  Noise points    : {r['n_noise']:,}")
        print(f"  Silhouette Score: {r['silhouette']}")
        print("-" * 55)
        print("  Anomaly Detection Metrics (Noise = Fraud Prediction)")
        print(f"  Precision  : {r['precision']:.4f}")
        print(f"  Recall     : {r['recall']:.4f}")
        print(f"  F1-Score   : {r['f1_score']:.4f}")
        print(f"  ROC-AUC    : {r['roc_auc']}")
        print("-" * 55)
        print("  Classification Report:")
        print(classification_report(y_true, self.anomaly_preds,
                                    target_names=["Normal", "Fraud"],
                                    zero_division=0))
        print("  Confusion Matrix:")
        cm = r["confusion_matrix"]
        print(f"  {'':>12} Pred Normal  Pred Fraud")
        print(f"  True Normal  {cm[0,0]:>11,}  {cm[0,1]:>10,}")
        print(f"  True Fraud   {cm[1,0]:>11,}  {cm[1,1]:>10,}")
        print("=" * 55 + "\n")
