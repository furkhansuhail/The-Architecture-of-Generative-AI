"""
reporter.py - Generates a final text/JSON report summarising the entire run.
"""

import json
import os
from datetime import datetime

import config


# ──────────────────────────────────────────────────────────────────────────────
class Reporter:
    def __init__(self):
        self.report: dict = {}

    def build(self, data_info: dict, pca_report: dict,
              eval_results: dict, tuning_df=None):
        cm = eval_results.pop("confusion_matrix", None)
        self.report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset"      : data_info,
            "preprocessing": {
                "dropped_columns": config.DROP_COLUMNS,
                "pca_enabled"    : config.USE_PCA,
                "pca_components" : config.PCA_COMPONENTS,
                **({k: v for k, v in pca_report.items()
                    if k != "explained_variance_ratio"} if pca_report else {}),
            },
            "dbscan_config": {
                "eps"           : config.DBSCAN_EPS,
                "min_samples"   : config.DBSCAN_MIN_SAMPLES,
                "metric"        : config.DBSCAN_METRIC,
            },
            "evaluation"   : eval_results,
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "top_tuning_results": (
                tuning_df.head(5).to_dict(orient="records")
                if tuning_df is not None else []
            ),
        }
        eval_results["confusion_matrix"] = cm  # restore

    def save(self):
        path = os.path.join(config.REPORT_DIR, "final_report.json")
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        print(f"[Reporter] Report saved → {path}")
        return path

    def print_summary(self):
        ev = self.report.get("evaluation", {})
        print("\n" + "╔" + "═" * 52 + "╗")
        print("║  FINAL RUN SUMMARY" + " " * 33 + "║")
        print("╠" + "═" * 52 + "╣")
        print(f"║  Precision  : {ev.get('precision', '-'):<37}║")
        print(f"║  Recall     : {ev.get('recall',    '-'):<37}║")
        print(f"║  F1-Score   : {ev.get('f1_score',  '-'):<37}║")
        print(f"║  ROC-AUC    : {ev.get('roc_auc',   '-'):<37}║")
        sil_val = ev.get('silhouette') or '-'
        print(f"║  Silhouette : {str(sil_val):<37}║")
        print("╚" + "═" * 52 + "╝\n")
