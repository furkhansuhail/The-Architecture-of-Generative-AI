"""
visualizer.py - Results Visualisation Module.

Generates:
  1. PCA 2D cluster scatter plot
  2. t-SNE 2D cluster scatter plot  
  3. Confusion matrix heatmap
  4. Hyperparameter tuning heatmaps (F1, Recall, Precision)
  5. Cluster size distribution bar chart
  6. PCA explained variance plot
  7. Final summary dashboard (composite figure)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config


# ──────────────────────────────────────────────────────────────────────────────
class Visualizer:
    def __init__(self):
        self._palette = {
            "normal" : "#2A9D8F",
            "fraud"  : "#E76F51",
            "noise"  : "#E63946",
            "cluster": "#457B9D",
        }
        sns.set_style("whitegrid")

    # ── public API ────────────────────────────────────────────────────────────
    def plot_clusters_2d(self, X: np.ndarray, cluster_labels: np.ndarray,
                          y_true: np.ndarray = None, method: str = "pca"):
        """2D scatter of DBSCAN clusters using PCA or t-SNE projection."""
        print(f"[Visualizer] Generating 2D cluster plot ({method.upper()}) …")

        X_2d = self._reduce_2d(X, method=method)

        fig, axes = plt.subplots(1, 2 if y_true is not None else 1,
                                 figsize=(16 if y_true is not None else 8, 6))
        if y_true is None:
            axes = [axes]

        # ── DBSCAN cluster view ───────────────────────────────────────────
        ax = axes[0]
        unique_lbls = sorted(set(cluster_labels))
        cmap = plt.get_cmap(config.COLORMAP, max(len(unique_lbls), 2))
        for i, lbl in enumerate(unique_lbls):
            mask = cluster_labels == lbl
            if lbl == -1:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=self._palette["noise"], s=8, alpha=0.6,
                           label=f"Noise (anomaly) [{mask.sum():,}]", zorder=5)
            else:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[cmap(i)], s=5, alpha=0.4,
                           label=f"Cluster {lbl} [{mask.sum():,}]")

        ax.set_title(f"DBSCAN Clusters  ({method.upper()} projection)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel(f"{method.upper()} Dim 1")
        ax.set_ylabel(f"{method.upper()} Dim 2")
        ax.legend(markerscale=2, fontsize=8, loc="best",
                  framealpha=0.9, ncol=2)

        # ── Ground truth view ────────────────────────────────────────────
        if y_true is not None:
            ax2 = axes[1]
            mask_n = y_true == 0
            mask_f = y_true == 1
            ax2.scatter(X_2d[mask_n, 0], X_2d[mask_n, 1],
                        c=self._palette["normal"], s=5, alpha=0.3,
                        label=f"Normal [{mask_n.sum():,}]")
            ax2.scatter(X_2d[mask_f, 0], X_2d[mask_f, 1],
                        c=self._palette["fraud"], s=20, alpha=0.8,
                        marker="x", linewidths=1.0,
                        label=f"Fraud (true) [{mask_f.sum():,}]", zorder=5)
            ax2.set_title(f"Ground Truth Labels  ({method.upper()} projection)",
                          fontsize=13, fontweight="bold")
            ax2.set_xlabel(f"{method.upper()} Dim 1")
            ax2.set_ylabel(f"{method.upper()} Dim 2")
            ax2.legend(markerscale=2, fontsize=9)

        plt.tight_layout()
        path = f"{config.PLOT_DIR}/results_clusters_{method}.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)
        print(f"[Visualizer] Saved → {path}")

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Annotated confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["Normal (0)", "Fraud (1)"]

        # Normalise per row for colour, show raw counts as text
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)"
                           for j in range(2)] for i in range(2)])

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=["Pred Normal", "Pred Fraud"],
                    yticklabels=["True Normal", "True Fraud"],
                    ax=ax, linewidths=1, cbar_kws={"label": "Row-normalised rate"},
                    annot_kws={"size": 12})
        ax.set_title("Confusion Matrix  (DBSCAN Noise = Fraud Prediction)",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual Class", fontsize=11)
        ax.set_xlabel("Predicted Class", fontsize=11)

        plt.tight_layout()
        path = f"{config.PLOT_DIR}/results_confusion_matrix.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)
        print(f"[Visualizer] Saved → {path}")

    def plot_tuning_heatmaps(self, tuning_df: pd.DataFrame):
        """F1 / Recall / Precision heatmaps from hyperparameter search."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Hyperparameter Tuning Results  (eps × min_samples)",
                     fontsize=14, fontweight="bold")

        metrics = ["f1_score", "recall", "precision"]
        cmaps   = ["YlOrRd", "YlGn", "Blues"]
        titles  = ["F1-Score", "Recall (Fraud)", "Precision (Fraud)"]

        for ax, metric, cmap, title in zip(axes, metrics, cmaps, titles):
            pivot = tuning_df.pivot_table(
                index="min_samples", columns="eps", values=metric, aggfunc="mean"
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                        ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
                        annot_kws={"size": 9})
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("eps", fontsize=10)
            ax.set_ylabel("min_samples", fontsize=10)

        plt.tight_layout()
        path = f"{config.PLOT_DIR}/results_tuning_heatmaps.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)
        print(f"[Visualizer] Saved → {path}")

    def plot_cluster_sizes(self, cluster_labels: np.ndarray):
        """Bar chart of cluster sizes (noise shown separately)."""
        unique, counts = np.unique(cluster_labels, return_counts=True)
        df = pd.DataFrame({"cluster": unique, "count": counts})
        df["label"] = df["cluster"].apply(
            lambda x: "Noise (-1)" if x == -1 else f"Cluster {x}"
        )
        df["color"] = df["cluster"].apply(
            lambda x: self._palette["noise"] if x == -1 else self._palette["cluster"]
        )
        df = df.sort_values("count", ascending=False)

        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.8 + 2), 5))
        bars = ax.bar(range(len(df)), df["count"], color=df["color"].values,
                      edgecolor="black", linewidth=0.7)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
        ax.set_title("Cluster Size Distribution", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Points")

        for bar, cnt in zip(bars, df["count"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5, f"{cnt:,}",
                    ha="center", va="bottom", fontsize=8)

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self._palette["cluster"], label="Cluster"),
            Patch(facecolor=self._palette["noise"],   label="Noise (anomaly)"),
        ]
        ax.legend(handles=legend_elements, fontsize=10)
        plt.tight_layout()
        path = f"{config.PLOT_DIR}/results_cluster_sizes.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)
        print(f"[Visualizer] Saved → {path}")

    def plot_pca_variance(self, pca_report: dict):
        """Cumulative explained variance curve from PCA."""
        if not pca_report:
            return
        cum_var = pca_report["cumulative_variance"]
        ind_var = pca_report["explained_variance_ratio"]
        n = len(cum_var)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(1, n + 1), ind_var, alpha=0.6, color="#457B9D",
               label="Individual variance")
        ax2 = ax.twinx()
        ax2.plot(range(1, n + 1), cum_var, "o-", color="#E63946",
                 linewidth=2, markersize=5, label="Cumulative variance")
        ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.7, label="90% threshold")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Cumulative Explained Variance", fontsize=11, color="#E63946")

        ax.set_xlabel("Principal Component", fontsize=11)
        ax.set_ylabel("Individual Explained Variance", fontsize=11, color="#457B9D")
        ax.set_title("PCA Explained Variance", fontsize=13, fontweight="bold")
        ax.set_xticks(range(1, n + 1))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

        plt.tight_layout()
        path = f"{config.PLOT_DIR}/results_pca_variance.png"
        fig.savefig(path, dpi=config.FIGURE_DPI)
        plt.close(fig)
        print(f"[Visualizer] Saved → {path}")

    def plot_summary_dashboard(self, results: dict, y_true: np.ndarray,
                                anomaly_preds: np.ndarray):
        """A single summary dashboard figure with key metrics."""
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#F8F9FA")
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle("DBSCAN Anomaly Detection — Summary Dashboard",
                     fontsize=16, fontweight="bold", y=0.98)

        colors_metric = ["#2A9D8F", "#E76F51", "#457B9D", "#E9C46A"]
        metric_labels = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
        metric_values = [
            results.get("precision", 0),
            results.get("recall",    0),
            results.get("f1_score",  0),
            results.get("roc_auc",   0) or 0,
        ]

        # ── Metric bars ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.barh(metric_labels, metric_values, color=colors_metric,
                        edgecolor="white", height=0.5)
        ax1.set_xlim(0, 1.0)
        ax1.set_title("Detection Metrics", fontweight="bold")
        for bar, v in zip(bars, metric_values):
            ax1.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                     f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
        ax1.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Score")
        ax1.set_facecolor("#FFFFFF")

        # ── Confusion matrix mini ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        cm  = results["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax2.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)",
                         ha="center", va="center", fontsize=10, fontweight="bold",
                         color="white" if cm_norm[i,j] > 0.5 else "black")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred Normal", "Pred Fraud"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["True Normal", "True Fraud"])
        ax2.set_title("Confusion Matrix", fontweight="bold")

        # ── Stats summary text ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis("off")
        total      = len(y_true)
        true_fraud = y_true.sum()
        pred_fraud = anomaly_preds.sum()
        tp = int(((y_true == 1) & (anomaly_preds == 1)).sum())
        fp = int(((y_true == 0) & (anomaly_preds == 1)).sum())
        fn = int(((y_true == 1) & (anomaly_preds == 0)).sum())
        tn = int(((y_true == 0) & (anomaly_preds == 0)).sum())

        lines = [
            ("Dataset", ""),
            (f"  Total samples",  f"{total:,}"),
            (f"  True fraud",     f"{true_fraud:,}  ({true_fraud/total*100:.3f}%)"),
            (f"  Pred anomalies", f"{pred_fraud:,}  ({pred_fraud/total*100:.3f}%)"),
            ("", ""),
            ("Model", ""),
            (f"  Clusters",       f"{results['n_clusters']}"),
            (f"  Noise pts",      f"{results['n_noise']:,}"),
            (f"  Silhouette",     f"{results['silhouette']}"),
            ("", ""),
            ("Outcomes", ""),
            (f"  True Positives",  f"{tp:,}"),
            (f"  False Positives", f"{fp:,}"),
            (f"  False Negatives", f"{fn:,}"),
            (f"  True Negatives",  f"{tn:,}"),
        ]
        y_pos = 0.98
        for key, val in lines:
            if key and not val:   # section header
                ax3.text(0.02, y_pos, key, fontsize=11, fontweight="bold",
                         transform=ax3.transAxes, color="#264653")
            else:
                ax3.text(0.02, y_pos, key,  fontsize=9.5, transform=ax3.transAxes)
                ax3.text(0.62, y_pos, val,  fontsize=9.5, transform=ax3.transAxes,
                         ha="right", color="#E76F51")
            y_pos -= 0.065

        ax3.set_title("Quick Summary", fontweight="bold")
        ax3.set_facecolor("#FAFAFA")
        ax3.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color="#FAFAFA",
                                     transform=ax3.transAxes, zorder=-1))

        # ── Prediction distribution ───────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, :2])
        categories = ["True Normal\n(Correct)", "False Alarm\n(FP)",
                      "Missed Fraud\n(FN)", "Caught Fraud\n(TP)"]
        values     = [tn, fp, fn, tp]
        bar_colors = ["#2A9D8F", "#E9C46A", "#E76F51", "#264653"]
        bars2 = ax4.bar(categories, values, color=bar_colors, edgecolor="black",
                        linewidth=0.8, width=0.5)
        for bar, v in zip(bars2, values):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total * 0.003,
                     f"{v:,}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
        ax4.set_title("Prediction Breakdown", fontweight="bold", fontsize=12)
        ax4.set_ylabel("Count")
        ax4.set_facecolor("#FFFFFF")

        # ── DBSCAN config summary ─────────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        config_text = (
            f"DBSCAN Configuration\n"
            f"{'─'*28}\n"
            f"ε (eps)         : {results['eps']}\n"
            f"min_samples     : {results['min_samples']}\n"
            f"Metric          : {config.DBSCAN_METRIC}\n"
            f"PCA components  : {config.PCA_COMPONENTS if config.USE_PCA else 'N/A'}\n"
            f"Sample size     : {total:,}\n"
            f"{'─'*28}\n"
            f"Noise pts → Fraud predictions\n"
            f"Core pts  → Normal transactions"
        )
        ax5.text(0.1, 0.88, config_text, fontsize=10, transform=ax5.transAxes,
                 va="top", family="monospace",
                 bbox=dict(facecolor="#EDF2F4", edgecolor="#ADB5BD",
                           boxstyle="round,pad=0.6"))
        ax5.set_title("Configuration", fontweight="bold")

        plt.savefig(f"{config.PLOT_DIR}/results_dashboard.png",
                    dpi=config.FIGURE_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Dashboard saved → {config.PLOT_DIR}/results_dashboard.png")

    # ── private ───────────────────────────────────────────────────────────────
    def _reduce_2d(self, X: np.ndarray, method: str = "pca") -> np.ndarray:
        if X.shape[1] == 2:
            return X
        if method == "tsne":
            print("  Running t-SNE … (this may take ~30s)")
            from sklearn import __version__ as _skv
            _iter_key = 'max_iter' if tuple(int(x) for x in _skv.split('.')[:2]) >= (1, 4) else 'n_iter'
            reducer = TSNE(n_components=2, perplexity=config.TSNE_PERPLEXITY,
                           **{_iter_key: config.TSNE_N_ITER},
                           random_state=config.RANDOM_STATE)
            return reducer.fit_transform(X)
        else:  # pca
            reducer = PCA(n_components=2, random_state=config.RANDOM_STATE)
            return reducer.fit_transform(X)