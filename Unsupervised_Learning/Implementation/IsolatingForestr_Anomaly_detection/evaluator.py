"""
evaluator.py - Comprehensive evaluation of the Isolation Forest predictions.

Metrics computed
────────────────
  Classification (using IsolationForest.predict threshold):
    • Confusion matrix
    • Precision, Recall, F1, Accuracy
    • Matthews Correlation Coefficient (MCC)

  Ranking / Scoring (using raw anomaly scores):
    • ROC-AUC
    • Average Precision (AP / PR-AUC)
    • Precision@K  (for K = 50, 100, 200 top anomalies)

  Outputs a nicely formatted report dict and logs it.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    accuracy_score, matthews_corrcoef,
)
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# IsolationForest returns  +1 (normal) / -1 (anomaly)
# We remap to  0 / 1  for sklearn metrics
_REMAP = {1: 0, -1: 1}


def evaluate(model: IsolationForest,
             X_test: pd.DataFrame,
             y_test: pd.Series,
             test_scores: np.ndarray) -> dict:
    """
    Full evaluation suite.

    Parameters
    ----------
    model        : fitted IsolationForest
    X_test       : test features
    y_test       : true binary labels (0=normal, 1=anomaly)
    test_scores  : negated decision_function scores (higher = more anomalous)

    Returns
    -------
    report : dict  (all metrics)
    """
    logger.info("──── Evaluation ─────────────────────────────────")

    raw_preds  = model.predict(X_test)                           # +1 / -1
    y_pred     = pd.Series(raw_preds).map(_REMAP).values        # 0 / 1
    y_true     = y_test.values

    # ── Classification metrics ───────────────────────────────────────────────
    cm         = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = {
        # Confusion matrix
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        # Standard metrics
        "Accuracy"  : round(accuracy_score(y_true, y_pred),        4),
        "Precision" : round(precision_score(y_true, y_pred,
                            zero_division=0),                       4),
        "Recall"    : round(recall_score(y_true, y_pred,
                            zero_division=0),                       4),
        "F1"        : round(f1_score(y_true, y_pred,
                            zero_division=0),                       4),
        "MCC"       : round(matthews_corrcoef(y_true, y_pred),     4),
        # Ranking metrics
        "ROC_AUC"   : round(roc_auc_score(y_true, test_scores),   4),
        "PR_AUC"    : round(average_precision_score(y_true,
                            test_scores),                           4),
    }

    # ── Precision@K ──────────────────────────────────────────────────────────
    sorted_idx   = np.argsort(-test_scores)          # descending anomaly score
    n_anomalies  = int(y_true.sum())
    for k in [50, 100, 200, n_anomalies]:
        if k > len(y_true):
            continue
        top_k        = sorted_idx[:k]
        prec_k       = y_true[top_k].sum() / k
        report[f"Precision@{k}"] = round(float(prec_k), 4)

    # ── Print report ─────────────────────────────────────────────────────────
    _log_report(report, y_true, y_pred)

    return report


# ──────────────────────────────────────────────────────────────────────────────

def _log_report(report: dict, y_true, y_pred) -> None:
    sep = "─" * 50
    logger.info("\n%s", sep)
    logger.info("  Isolation Forest — Evaluation Report")
    logger.info(sep)
    logger.info("  Confusion Matrix  (actual \\ predicted)")
    logger.info("              Normal   Anomaly")
    logger.info("  Normal    %6d  %6d", report["TN"], report["FP"])
    logger.info("  Anomaly   %6d  %6d", report["FN"], report["TP"])
    logger.info(sep)
    for key in ["Accuracy", "Precision", "Recall", "F1", "MCC",
                "ROC_AUC", "PR_AUC"]:
        logger.info("  %-18s : %.4f", key, report[key])
    pk_keys = [k for k in report if k.startswith("Precision@")]
    for k in pk_keys:
        logger.info("  %-18s : %.4f", k, report[k])
    logger.info(sep)
    logger.info("\n  Classification Report (sklearn):\n%s",
                classification_report(y_true, y_pred,
                                      target_names=["Normal", "Anomaly"],
                                      zero_division=0))
