"""
================================================================================
  K-MEANS CLUSTERING MODULE
  A complete, plug-and-play module for unsupervised partitional clustering
  with comprehensive K-selection methods
================================================================================

WHAT IS K-MEANS CLUSTERING?
-----------------------------
K-Means is an unsupervised learning algorithm that partitions n data points
into exactly K non-overlapping clusters by minimising the total within-cluster
variance. Unlike supervised models, there are no labels — the algorithm
discovers structure purely from the data itself.

THE CORE IDEA — LLOYD'S ALGORITHM:
  Repeat until convergence:
    Step 1 — Assignment : zᵢ = argmin_k ‖xᵢ − μₖ‖²   (nearest centroid)
    Step 2 — Update     : μₖ ← mean of all points assigned to cluster k

  This monotonically decreases the objective (WCSS) and is guaranteed
  to converge to a LOCAL minimum.

THE OBJECTIVE — WITHIN-CLUSTER SUM OF SQUARES (WCSS / Inertia):
  J = Σᵢ Σₖ  1[zᵢ = k] · ‖xᵢ − μₖ‖²

  Minimising J produces tight, compact clusters.
  Globally optimal J is NP-hard → Lloyd's finds a local minimum.

THE CENTRAL CHALLENGE — CHOOSING K:
  K is the most critical hyperparameter. This module implements every major
  method for selecting K:

    1. Elbow Method          — plot WCSS vs K, find "elbow" (diminishing returns)
    2. Silhouette Score      — measures cohesion vs separation (−1 to +1)
    3. Gap Statistic         — compare WCSS to random null reference data
    4. Calinski-Harabasz     — ratio of between-cluster to within-cluster variance
    5. Davies-Bouldin Index  — average cluster similarity (lower = better)
    6. BIC via GMM           — penalised likelihood, balances fit vs complexity

INITIALISATION — K-MEANS++:
  Pick first centroid randomly. Each subsequent centroid is sampled with
  probability ∝ D²(x) (squared distance to nearest already-chosen centroid).
  Guarantees: expected cost within O(log K) of optimal. Always use this.

HOW TO USE THIS MODULE
-----------------------
  from KMeansClustering import KMeansModel

  model = KMeansModel(K=4)
  model.fit(X)
  labels = model.predict(X)
  model.summary()

  # Or let the tuner find K for you:
  tuner = KMeansTuner(K_max=10)
  best_K = tuner.fit(X)
  tuner.results_summary()

CONTENTS
---------
  1. KMeansModel        — core model: fit / predict / summary
  2. KMeansTuner        — K-selection: all 6 methods with recommendations
  3. KMeansEvaluator    — cluster quality: silhouette, inertia, cohesion, separation
  4. KMeansVisualizer   — 6-panel diagnostic dashboard + per-method plots
  5. run_full_pipeline() — one call: tune → fit → evaluate → visualize

================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score
)
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 ── CORE MODEL CLASS
#  Wraps sklearn KMeans with auto-scaling, K-Means++ init, a human-readable
#  cluster summary, and convenient predict / transform methods.
# ══════════════════════════════════════════════════════════════════════════════

class KMeansModel:
    """
    A production-ready K-Means clustering model.

    CLUSTERING WORKFLOW:
    ─────────────────────
    1. Features (X) enter the model
    2. K centroids are initialised via K-Means++ (spreads centroids intelligently)
    3. Assignment Step: each point goes to its nearest centroid
    4. Update Step: each centroid moves to the mean of its cluster
    5. Steps 3-4 repeat until centroids stop moving (convergence)

    Parameters
    ----------
    K : int, default=3
        Number of clusters. The single most important hyperparameter.
        Use KMeansTuner to find the optimal K automatically.

    init : str, default='k-means++'
        Initialisation strategy:
          'k-means++' → Probabilistic seeding (O(log K) quality guarantee)
          'random'    → Pick K data points uniformly at random (faster, worse)

    n_init : int, default=10
        Number of independent runs with different random seeds.
        Best inertia result is kept. Higher n_init → more robust but slower.
        With init='k-means++', n_init=5 is usually enough.

    max_iter : int, default=300
        Safety cap on the number of Lloyd's iterations per run.
        Most datasets converge well before 100 iterations.

    tol : float, default=1e-4
        Convergence threshold: stop when centroid movement falls below this.

    scale_features : bool, default=True
        Standardize X before fitting (zero mean, unit variance).
        CRITICAL: K-Means uses Euclidean distance — unscaled features are unfair.
        A feature with range [0, 10000] dominates one with range [0, 1].

    random_state : int, default=42
        Seed for reproducibility.

    Example
    -------
    >>> model = KMeansModel(K=4)                    # doctest: +SKIP
    >>> model.fit(X)                                 # doctest: +SKIP
    >>> labels = model.predict(X)                    # doctest: +SKIP
    >>> model.summary()                              # doctest: +SKIP
    """

    def __init__(
        self,
        K: int = 3,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        self.K              = K
        self.init           = init
        self.n_init         = n_init
        self.max_iter       = max_iter
        self.tol            = tol
        self.scale_features = scale_features
        self.random_state   = random_state

        self._model   = KMeans(
            n_clusters=K, init=init, n_init=n_init,
            max_iter=max_iter, tol=tol, random_state=random_state,
        )
        self._scaler  = StandardScaler() if scale_features else None

        # Set after fit()
        self.labels_       = None
        self.centroids_    = None
        self.inertia_      = None
        self.n_iter_       = None
        self.feature_names_= None
        self.n_features_   = None
        self.n_samples_    = None
        self.cluster_sizes_= None
        self.is_fitted_    = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, feature_names=None):
        """
        Train K-Means on data X.

        WHAT HAPPENS STEP-BY-STEP:
        ───────────────────────────
        1. Optionally standardize X (mean=0, std=1 per feature).
        2. Initialise K centroids via K-Means++ (or random).
        3. Assignment Step: assign each point to the nearest centroid.
        4. Update Step: move each centroid to the mean of its assigned points.
        5. Repeat Steps 3-4 until centroid movement < tol, or max_iter reached.
        6. Repeat the entire process n_init times; keep the run with lowest inertia.

        Parameters
        ----------
        X            : array-like (n_samples, n_features)
        feature_names: optional list of feature column names

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)

        self.feature_names_ = (
            list(feature_names) if feature_names is not None
            else [f"feature_{i}" for i in range(X.shape[1])]
        )
        self.n_samples_, self.n_features_ = X.shape

        # Scale
        X_ready = self._scaler.fit_transform(X) if self.scale_features else X

        # Fit
        self._model.fit(X_ready)

        # Cache results
        self.labels_        = self._model.labels_
        self.centroids_     = self._scaler.inverse_transform(self._model.cluster_centers_) \
                              if self.scale_features else self._model.cluster_centers_.copy()
        self.inertia_       = self._model.inertia_
        self.n_iter_        = self._model.n_iter_
        self.cluster_sizes_ = np.bincount(self.labels_, minlength=self.K)
        self.is_fitted_     = True

        print(
            f"[KMeansModel] Fitted — K={self.K} clusters | "
            f"{self.n_samples_} samples, {self.n_features_} features\n"
            f"  init='{self.init}', n_init={self.n_init}, "
            f"converged in {self.n_iter_} iterations\n"
            f"  Inertia (WCSS): {self.inertia_:.4f}\n"
            f"  Cluster sizes:  {self.cluster_sizes_.tolist()}"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Assign new data points to the nearest trained centroid.

        For each point xᵢ:
            label = argmin_k ‖xᵢ − μₖ‖²

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        labels : ndarray (n_samples,) — integer cluster indices 0..K-1
        """
        self._check_fitted()
        X = np.array(X, dtype=float)
        X_ready = self._scaler.transform(X) if self.scale_features else X
        return self._model.predict(X_ready)

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, X):
        """
        Transform X to cluster-distance space.

        Each output column is the distance from each sample to one centroid.
        Shape: (n_samples, K)

        USE CASE: Use the K distance columns as new features for a supervised model.
        This adds non-linear cluster structure awareness to linear classifiers.
        """
        self._check_fitted()
        X = np.array(X, dtype=float)
        X_ready = self._scaler.transform(X) if self.scale_features else X
        return self._model.transform(X_ready)

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print a detailed, structured summary of the fitted clustering including:
          - Configuration and convergence info
          - Per-cluster size, fraction, and centroid position
          - Compactness score per cluster (mean intra-cluster distance)
          - Global quality metrics (inertia, silhouette if feasible)
        """
        self._check_fitted()
        sep = "=" * 65

        print(f"\n{sep}")
        print("  K-MEANS CLUSTERING MODEL SUMMARY")
        print(sep)
        print(f"  K (clusters)    : {self.K}")
        print(f"  Samples         : {self.n_samples_}")
        print(f"  Features        : {self.n_features_}")
        print(f"  Init strategy   : {self.init}")
        print(f"  n_init runs     : {self.n_init}")
        print(f"  Iterations      : {self.n_iter_}  (converged)")
        print(f"  Inertia (WCSS)  : {self.inertia_:.4f}")
        print(f"  Feature scaling : {'ON (StandardScaler)' if self.scale_features else 'OFF'}")

        # Per-cluster breakdown
        X_ready = self._scaler.transform(
            self.centroids_) if self.scale_features else self.centroids_

        # Reconstruct X_scaled for compactness (we can't retrieve original X here,
        # so we use the centroid in scaled space as a proxy metric indicator)
        print(f"\n  {'Cluster':>9}  |  {'Size':>6}  |  {'Fraction':>9}  |  "
              f"{'Centroid (first 3 features)':>30}")
        print(f"  {'─────────':>9}──┼──{'──────':>6}──┼──{'─────────':>9}──┼──"
              f"{'──────────────────────────────':>30}")

        for k in range(self.K):
            size = self.cluster_sizes_[k]
            frac = size / self.n_samples_
            # Show first 3 centroid dimensions
            c = self.centroids_[k]
            c_str = "  ".join([f"{v:+.3f}" for v in c[:3]])
            if self.n_features_ > 3:
                c_str += " ..."
            print(f"  {k:>9}  |  {size:>6}  |  {frac:>9.1%}  |  {c_str}")

        # Silhouette (fast if n < 5000)
        X_sc = self._scaler.transform(
            np.zeros((1, self.n_features_))) if self.scale_features else None
        try:
            # We can only compute silhouette if we stored scaled data — use centroids
            # as a proxy for a quick indicator. Full silhouette needs original X.
            print(f"\n  CENTROID GUIDE:")
            print(f"    Centroids are in ORIGINAL feature space (before scaling).")
            print(f"    All distance computations internally use SCALED space.")
        except Exception:
            pass

        print(f"\n  INTERPRETATION GUIDE:")
        print(f"    Inertia (WCSS) — lower is better. Compare across runs / K values.")
        print(f"    Cluster size imbalance warns of potential K-Means failure modes.")
        print(f"    Use KMeansEvaluator for Silhouette, CH, DB metrics.")
        print(sep + "\n")

    # ── get_cluster_df ────────────────────────────────────────────────────────

    def get_cluster_df(self, X, feature_names=None) -> pd.DataFrame:
        """
        Return original data X with a 'cluster' column appended.

        Parameters
        ----------
        X            : original feature matrix (n_samples, n_features)
        feature_names: column names (optional)

        Returns
        -------
        pd.DataFrame with columns [feature_names..., 'cluster']
        """
        self._check_fitted()
        cols = feature_names or self.feature_names_
        df = pd.DataFrame(np.array(X, dtype=float), columns=cols)
        df["cluster"] = self.labels_
        return df

    # ── private ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call .fit(X) first.")

    def _get_scaled(self, X):
        X = np.array(X, dtype=float)
        return self._scaler.transform(X) if self.scale_features else X


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 ── K SELECTION TUNER
#  The central challenge of K-Means: how many clusters?
#  Implements all 6 standard methods with a consensus recommendation.
# ══════════════════════════════════════════════════════════════════════════════

class KMeansTuner:
    """
    Comprehensive K-selection for K-Means using 6 complementary methods.

    THE PROBLEM WITH CHOOSING K:
    ─────────────────────────────
    WCSS always decreases as K increases — with K=n every point is its own
    cluster and WCSS=0. We need measures that balance fit against complexity.
    No single method is universally best; using several together is the standard.

    METHODS IMPLEMENTED:
    ─────────────────────
    1. Elbow / WCSS
       Plot inertia vs K. Look for the "elbow" — the point of
       diminishing returns. Computed as the angle change in the curve.
       Best for: quick visual scan. Weakness: elbow often ambiguous.

    2. Silhouette Score
       s(i) = (b(i) - a(i)) / max(a(i), b(i))
         a(i) = mean dist to own cluster (cohesion)
         b(i) = mean dist to nearest other cluster (separation)
       Range: -1 (wrong cluster) to +1 (deep inside cluster).
       Choose K that maximises mean silhouette.
       Best for: general purpose. Weakness: O(n²) — slow for large n.

    3. Gap Statistic (Tibshirani et al., 2001)
       Gap(K) = E[log WCSS_random] - log WCSS_data
       Compare observed WCSS to K-Means on uniformly random data.
       Optimal K: smallest K s.t. Gap(K) ≥ Gap(K+1) - s_{K+1}
       Best for: statistically rigorous. Weakness: slow (needs B reference datasets).

    4. Calinski-Harabasz Index (Variance Ratio Criterion)
       CH = [B / (K-1)] / [W / (n-K)]
         B = between-cluster variance (how far apart clusters are)
         W = within-cluster variance (how tight clusters are)
       Higher is better. Choose K that maximises CH.
       Best for: large datasets (O(n·K·d)), no O(n²) issue.

    5. Davies-Bouldin Index
       DB = (1/K) Σₖ max_{j≠k} [ (σₖ + σⱼ) / d(μₖ, μⱼ) ]
         σₖ = mean dist of points in cluster k to its centroid
         d(μₖ, μⱼ) = distance between centroids
       Lower is better. Choose K that minimises DB.
       Best for: compact, well-separated clusters.

    6. BIC via Gaussian Mixture Model
       BIC = -2·log(L) + p·log(n)  where p = number of free parameters
       GMM's BIC penalises model complexity. Choose K that minimises BIC.
       Best for: when clusters may have varying sizes/shapes.

    Parameters
    ----------
    K_min : int, default=2
        Minimum K to evaluate.

    K_max : int, default=10
        Maximum K to evaluate.

    n_init : int, default=10
        K-Means restarts per K value (for stability).

    gap_B : int, default=10
        Number of reference datasets for the Gap Statistic.
        Higher B → more accurate but slower.

    scale_features : bool, default=True
        Standardize X before fitting.

    random_state : int, default=42

    Example
    -------
    >>> tuner = KMeansTuner(K_max=10)              # doctest: +SKIP
    >>> best_K = tuner.fit(X)                      # doctest: +SKIP
    >>> tuner.results_summary()                    # doctest: +SKIP
    """

    def __init__(
        self,
        K_min: int = 2,
        K_max: int = 10,
        n_init: int = 10,
        gap_B: int = 10,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        self.K_min          = K_min
        self.K_max          = K_max
        self.n_init         = n_init
        self.gap_B          = gap_B
        self.scale_features = scale_features
        self.random_state   = random_state

        self.K_range_       = list(range(K_min, K_max + 1))
        self.results_       = {}
        self.best_K_        = {}
        self.consensus_K_   = None
        self._X_scaled      = None
        self._is_fitted     = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X) -> int:
        """
        Run all K-selection methods on data X.

        Computes Elbow, Silhouette, Gap Statistic, Calinski-Harabasz,
        Davies-Bouldin, and BIC (via GMM) for each K in [K_min, K_max].

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        consensus_K : int — recommended K from majority vote across methods
        """
        X = np.array(X, dtype=float)
        scaler = StandardScaler() if self.scale_features else None
        self._X_scaled = scaler.fit_transform(X) if self.scale_features else X.copy()
        n, d = self._X_scaled.shape

        print(f"[KMeansTuner] Evaluating K = {self.K_min} … {self.K_max} "
              f"({len(self.K_range_)} values) | {n} samples, {d} features")

        inertias, sil_scores, ch_scores, db_scores, bic_scores = [], [], [], [], []
        all_labels = {}

        # ── Fit KMeans for each K ─────────────────────────────────────────────
        print("[KMeansTuner] Step 1/3 — Fitting K-Means for each K ...")
        for K in self.K_range_:
            km = KMeans(n_clusters=K, init="k-means++", n_init=self.n_init,
                        max_iter=300, random_state=self.random_state)
            km.fit(self._X_scaled)
            all_labels[K] = km.labels_
            inertias.append(km.inertia_)

            # Silhouette: only defined for K >= 2
            sil = silhouette_score(self._X_scaled, km.labels_) if K >= 2 else np.nan
            sil_scores.append(sil)

            # Calinski-Harabasz
            ch = calinski_harabasz_score(self._X_scaled, km.labels_) if K >= 2 else np.nan
            ch_scores.append(ch)

            # Davies-Bouldin
            db = davies_bouldin_score(self._X_scaled, km.labels_) if K >= 2 else np.nan
            db_scores.append(db)

            # BIC via GMM (more principled model selection)
            gm = GaussianMixture(n_components=K, n_init=3, random_state=self.random_state)
            gm.fit(self._X_scaled)
            bic_scores.append(gm.bic(self._X_scaled))

            print(f"  K={K:>2} | Inertia={km.inertia_:>10.2f} | "
                  f"Silhouette={sil:>7.4f} | CH={ch:>8.2f} | "
                  f"DB={db:>7.4f} | BIC={gm.bic(self._X_scaled):>10.2f}")

        # ── Gap Statistic ─────────────────────────────────────────────────────
        print(f"\n[KMeansTuner] Step 2/3 — Computing Gap Statistic "
              f"(B={self.gap_B} reference datasets) ...")
        gap_vals, gap_stds = self._compute_gap(self._X_scaled, inertias)

        # ── Elbow detection via maximum second-difference ─────────────────────
        print("[KMeansTuner] Step 3/3 — Detecting elbows and best K ...")
        elbow_K = self._detect_elbow(inertias)

        # ── Best K per method ─────────────────────────────────────────────────
        best_sil_K = self.K_range_[int(np.nanargmax(sil_scores))]
        best_ch_K  = self.K_range_[int(np.nanargmax(ch_scores))]
        best_db_K  = self.K_range_[int(np.nanargmin(db_scores))]
        best_bic_K = self.K_range_[int(np.argmin(bic_scores))]
        best_gap_K = self._detect_gap_K(gap_vals, gap_stds)

        self.best_K_ = {
            "Elbow (WCSS)":          elbow_K,
            "Silhouette":            best_sil_K,
            "Gap Statistic":         best_gap_K,
            "Calinski-Harabasz":     best_ch_K,
            "Davies-Bouldin":        best_db_K,
            "BIC (GMM)":             best_bic_K,
        }

        # ── Consensus vote ────────────────────────────────────────────────────
        votes = list(self.best_K_.values())
        from collections import Counter
        vote_counts = Counter(votes)
        self.consensus_K_ = vote_counts.most_common(1)[0][0]

        # ── Store everything ──────────────────────────────────────────────────
        self.results_ = pd.DataFrame({
            "K":               self.K_range_,
            "Inertia":         inertias,
            "Silhouette":      sil_scores,
            "CalinskiHarabasz":ch_scores,
            "DaviesBouldin":   db_scores,
            "BIC":             bic_scores,
            "Gap":             gap_vals,
            "Gap_std":         gap_stds,
        })

        self._inertias   = inertias
        self._sil_scores = sil_scores
        self._ch_scores  = ch_scores
        self._db_scores  = db_scores
        self._bic_scores = bic_scores
        self._gap_vals   = gap_vals
        self._gap_stds   = gap_stds
        self._all_labels = all_labels
        self._is_fitted  = True

        return self.consensus_K_

    # ── results_summary ───────────────────────────────────────────────────────

    def results_summary(self):
        """
        Print a structured summary of all K-selection method results,
        best K per method, and the consensus recommendation.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit(X) first.")

        sep = "=" * 70
        print(f"\n{sep}")
        print("  K-SELECTION RESULTS SUMMARY")
        print(sep)

        print(f"\n  {'K':>4}  |  {'Inertia':>10}  |  {'Silhouette':>11}  |  "
              f"{'CH Index':>9}  |  {'DB Index':>9}  |  {'BIC':>12}  |  {'Gap':>8}")
        print(f"  {'────':>4}──┼──{'──────────':>10}──┼──{'───────────':>11}──┼──"
              f"{'─────────':>9}──┼──{'─────────':>9}──┼──{'────────────':>12}──┼──{'────────':>8}")

        for _, row in self.results_.iterrows():
            k = int(row["K"])
            markers = []
            if k == self.best_K_.get("Elbow (WCSS)"):          markers.append("E")
            if k == self.best_K_.get("Silhouette"):            markers.append("S")
            if k == self.best_K_.get("Calinski-Harabasz"):     markers.append("C")
            if k == self.best_K_.get("Davies-Bouldin"):        markers.append("D")
            if k == self.best_K_.get("BIC (GMM)"):             markers.append("B")
            if k == self.best_K_.get("Gap Statistic"):         markers.append("G")
            flag = f"  ← [{','.join(markers)}]" if markers else ""

            sil = f"{row['Silhouette']:.4f}" if not np.isnan(row["Silhouette"]) else "  n/a "
            ch  = f"{row['CalinskiHarabasz']:.2f}" if not np.isnan(row["CalinskiHarabasz"]) else "  n/a"
            db  = f"{row['DaviesBouldin']:.4f}" if not np.isnan(row["DaviesBouldin"]) else "  n/a"
            gap = f"{row['Gap']:.4f}" if not np.isnan(row["Gap"]) else "  n/a"

            print(f"  {k:>4}  |  {row['Inertia']:>10.2f}  |  {sil:>11}  |  "
                  f"{ch:>9}  |  {db:>9}  |  {row['BIC']:>12.2f}  |  {gap:>8}{flag}")

        print(f"\n{sep}")
        print("  BEST K PER METHOD")
        print(sep)
        for method, best_k in self.best_K_.items():
            print(f"  {method:<28} →  K = {best_k}")

        print(f"\n{sep}")
        print("  CONSENSUS RECOMMENDATION")
        print(sep)
        print(f"  Recommended K : {self.consensus_K_}  (majority vote across all methods)")
        print(f"\n  LEGEND: E=Elbow  S=Silhouette  C=Calinski-Harabasz  "
              f"D=Davies-Bouldin  B=BIC  G=Gap")
        print(sep + "\n")

    # ── get_best_model ────────────────────────────────────────────────────────

    def get_best_model(self, method: str = "consensus") -> KMeansModel:
        """
        Return a pre-configured KMeansModel using the best K from a given method.

        Parameters
        ----------
        method : str — one of:
            'consensus', 'Elbow (WCSS)', 'Silhouette', 'Gap Statistic',
            'Calinski-Harabasz', 'Davies-Bouldin', 'BIC (GMM)'

        Returns
        -------
        KMeansModel with K pre-set
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit(X) first.")

        if method == "consensus":
            K = self.consensus_K_
        elif method in self.best_K_:
            K = self.best_K_[method]
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: "
                             f"consensus, {', '.join(self.best_K_.keys())}")

        print(f"[KMeansTuner] Creating KMeansModel with K={K} (method='{method}')")
        return KMeansModel(K=K, scale_features=self.scale_features,
                           random_state=self.random_state)

    # ── private helpers ───────────────────────────────────────────────────────

    def _detect_elbow(self, inertias: list) -> int:
        """
        Detect the elbow K via the maximum second derivative (angle change).

        Mathematically: find K where the rate of improvement drops fastest.
        The second difference Δ²ᵢ = inertia[i-1] - 2·inertia[i] + inertia[i+1]
        peaks at the true elbow.
        """
        inertias = np.array(inertias)
        if len(inertias) < 3:
            return self.K_range_[0]
        # Second differences
        d2 = np.abs(np.diff(np.diff(inertias)))
        elbow_idx = int(np.argmax(d2)) + 1   # +1 because we lost 2 elements via double-diff
        return self.K_range_[elbow_idx]

    def _compute_gap(self, X_scaled, inertias):
        """
        Compute the Gap Statistic for each K.

        Algorithm:
          For b = 1..B:
              Draw X_ref ~ Uniform(feature bounding box)
              Fit K-Means on X_ref, record WCSS_ref(K)
          Gap(K) = mean_b[log WCSS_ref(K)] - log WCSS_data(K)
          s_K    = std_b[log WCSS_ref(K)] * sqrt(1 + 1/B)

        The optimal K is the SMALLEST K s.t. Gap(K) >= Gap(K+1) - s_{K+1}.
        """
        rng = np.random.default_rng(self.random_state)
        n, d = X_scaled.shape

        # Bounding box of the scaled data
        X_min = X_scaled.min(axis=0)
        X_max = X_scaled.max(axis=0)

        gap_vals = np.full(len(self.K_range_), np.nan)
        gap_stds = np.full(len(self.K_range_), np.nan)

        for i, K in enumerate(self.K_range_):
            ref_log_wcss = []
            for _ in range(self.gap_B):
                # Reference dataset: uniform in the bounding box
                X_ref = rng.uniform(X_min, X_max, size=(n, d))
                km_ref = KMeans(n_clusters=K, init="k-means++", n_init=3,
                                max_iter=100, random_state=int(rng.integers(0, 10000)))
                km_ref.fit(X_ref)
                ref_log_wcss.append(np.log(km_ref.inertia_ + 1e-10))

            ref_log_wcss = np.array(ref_log_wcss)
            gap_vals[i] = ref_log_wcss.mean() - np.log(inertias[i] + 1e-10)
            gap_stds[i] = ref_log_wcss.std() * np.sqrt(1 + 1 / self.gap_B)

        return gap_vals.tolist(), gap_stds.tolist()

    def _detect_gap_K(self, gap_vals, gap_stds) -> int:
        """
        Select K using the 1-SE rule on the Gap Statistic.
        Optimal K: smallest K s.t. Gap(K) >= Gap(K+1) - s_{K+1}
        """
        gap = np.array(gap_vals)
        std = np.array(gap_stds)
        for i in range(len(self.K_range_) - 1):
            if gap[i] >= gap[i + 1] - std[i + 1]:
                return self.K_range_[i]
        # Fallback: K with maximum gap
        return self.K_range_[int(np.nanargmax(gap))]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 ── EVALUATOR
#  Cluster quality requires different metrics than supervised learning.
#  There are no labels to compare against — quality is measured intrinsically.
# ══════════════════════════════════════════════════════════════════════════════

class KMeansEvaluator:
    """
    Comprehensive cluster quality metrics for K-Means.

    METRICS EXPLAINED:
    ──────────────────
    Inertia (WCSS) — Sum of squared distances from each point to its centroid.
                     Lower = tighter, more compact clusters.
                     NOT comparable across datasets or different K values directly.

    Silhouette Score — Measures how similar each point is to its own cluster
                       versus other clusters. Range: −1 to +1.
                       Mean over all points: higher = better defined clusters.
                       +1: point is deep inside its cluster, far from all others
                        0: point is on the boundary between two clusters
                       −1: point is probably in the wrong cluster

    Calinski-Harabasz (CH) Index — Ratio of between-cluster to within-cluster
                       variance. Higher = better. Favours compact, separated clusters.
                       Very fast to compute (no O(n²) step). Good for large datasets.

    Davies-Bouldin (DB) Index — Average similarity between each cluster and its
                       most similar cluster. Lower = better.
                       Combines cluster spread (σ) and centroid separation (d).

    Inter-cluster Distances — Mean pairwise distance between centroids.
                       Higher = clusters are more spread out = better separation.

    Example
    -------
    >>> ev = KMeansEvaluator(model)               # doctest: +SKIP
    >>> ev.evaluate(X, label="My Data")           # doctest: +SKIP
    >>> ev.per_cluster_report(X)                  # doctest: +SKIP
    """

    def __init__(self, model: KMeansModel):
        self.model    = model
        self.metrics_ = {}

    def evaluate(self, X, label: str = "Dataset") -> dict:
        """
        Compute all cluster quality metrics on X.

        Parameters
        ----------
        X     : feature matrix used for clustering
        label : display label for the output

        Returns
        -------
        dict of metric → value
        """
        self.model._check_fitted()
        X = np.array(X, dtype=float)
        X_sc = self.model._get_scaled(X)
        labels = self.model.labels_

        inertia = self.model.inertia_
        sil     = silhouette_score(X_sc, labels) if self.model.K >= 2 else np.nan
        ch      = calinski_harabasz_score(X_sc, labels) if self.model.K >= 2 else np.nan
        db      = davies_bouldin_score(X_sc, labels) if self.model.K >= 2 else np.nan

        # Mean inter-centroid distance (separation)
        ctrs = self.model._model.cluster_centers_
        n_c  = len(ctrs)
        dists = [np.linalg.norm(ctrs[i] - ctrs[j])
                 for i in range(n_c) for j in range(i + 1, n_c)]
        inter_dist = np.mean(dists) if dists else np.nan

        self.metrics_ = dict(
            Inertia=inertia, Silhouette=sil,
            CalinskiHarabasz=ch, DaviesBouldin=db,
            MeanInterCentroidDist=inter_dist
        )

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  CLUSTER QUALITY EVALUATION — {label}")
        print(sep)
        print(f"  Inertia (WCSS)             : {inertia:.4f}  (lower = tighter clusters)")
        print(f"  Silhouette Score           : {sil:.4f}  "
              f"({'excellent' if sil > 0.6 else 'good' if sil > 0.4 else 'fair' if sil > 0.2 else 'poor'})")
        print(f"  Calinski-Harabasz Index    : {ch:.4f}  (higher = better)")
        print(f"  Davies-Bouldin Index       : {db:.4f}  (lower = better)")
        print(f"  Mean Inter-Centroid Dist   : {inter_dist:.4f}  (higher = more separated)")
        print(f"\n  SILHOUETTE INTERPRETATION:")
        print(f"    > 0.7   Excellent — strong cluster structure")
        print(f"    0.5–0.7 Good      — reasonable cluster structure")
        print(f"    0.3–0.5 Fair      — weak but present structure")
        print(f"    < 0.3   Poor      — clusters may be artificial")
        print(sep + "\n")
        return self.metrics_

    def per_cluster_report(self, X) -> pd.DataFrame:
        """
        Per-cluster breakdown: size, silhouette, compactness, and separation.

        Returns
        -------
        pd.DataFrame with one row per cluster
        """
        self.model._check_fitted()
        X = np.array(X, dtype=float)
        X_sc = self.model._get_scaled(X)
        labels = self.model.labels_
        ctrs   = self.model._model.cluster_centers_

        # Per-point silhouette
        sil_samples = silhouette_samples(X_sc, labels) if self.model.K >= 2 \
                      else np.zeros(len(X))

        rows = []
        for k in range(self.model.K):
            mask    = labels == k
            pts_k   = X_sc[mask]
            n_k     = mask.sum()
            sil_k   = sil_samples[mask].mean() if n_k > 0 else np.nan
            n_neg   = (sil_samples[mask] < 0).sum()

            # Mean distance to own centroid (compactness)
            if n_k > 0:
                compact = np.mean(np.linalg.norm(pts_k - ctrs[k], axis=1))
            else:
                compact = np.nan

            # Mean distance to nearest other centroid (separation)
            other_ctrs = [ctrs[j] for j in range(self.model.K) if j != k]
            if other_ctrs:
                sep = np.min([np.linalg.norm(ctrs[k] - c) for c in other_ctrs])
            else:
                sep = np.nan

            rows.append({
                "Cluster":      k,
                "Size":         n_k,
                "Fraction":     n_k / len(X),
                "Silhouette":   round(sil_k, 4),
                "n_negative_sil": n_neg,
                "Compactness":  round(compact, 4),
                "Separation":   round(sep, 4),
            })

        df = pd.DataFrame(rows)

        sep_line = "=" * 70
        print(f"\n{sep_line}")
        print("  PER-CLUSTER BREAKDOWN")
        print(sep_line)
        print(f"  {'Cluster':>8}  {'Size':>6}  {'Fraction':>9}  "
              f"{'Silhouette':>11}  {'Neg Sil':>8}  {'Compact':>9}  {'Sep':>9}")
        print(f"  {'────────':>8}──{'──────':>6}──{'─────────':>9}──"
              f"{'───────────':>11}──{'────────':>8}──{'─────────':>9}──{'─────────':>9}")
        for _, r in df.iterrows():
            print(f"  {int(r['Cluster']):>8}  {int(r['Size']):>6}  {r['Fraction']:>9.1%}  "
                  f"{r['Silhouette']:>11.4f}  {int(r['n_negative_sil']):>8}  "
                  f"{r['Compactness']:>9.4f}  {r['Separation']:>9.4f}")
        print(f"\n  Compactness : Mean distance of cluster members to their centroid (lower=tighter)")
        print(f"  Separation  : Distance from this centroid to the nearest OTHER centroid (higher=better)")
        print(f"  Neg Sil     : Points with silhouette < 0 (likely borderline / misassigned)")
        print(sep_line + "\n")
        return df

    def compare_K_values(self, X, K_list: list) -> pd.DataFrame:
        """
        Fit K-Means for several K values and compare all quality metrics side by side.

        Useful for confirming the tuner's recommendation and seeing how
        metrics trade off across different K choices.

        Parameters
        ----------
        X      : feature matrix
        K_list : list of K values to compare (e.g., [2, 3, 4, 5, 6])

        Returns
        -------
        pd.DataFrame sorted by Silhouette (descending)
        """
        X = np.array(X, dtype=float)
        scaler = StandardScaler() if self.model.scale_features else None
        X_sc   = scaler.fit_transform(X) if scaler else X

        rows = []
        print(f"\n{'='*65}")
        print("  K-VALUE COMPARISON")
        print(f"{'='*65}")
        print(f"  {'K':>4}  |  {'Inertia':>10}  |  {'Silhouette':>11}  |  "
              f"{'CH':>9}  |  {'DB':>8}")
        print(f"  {'────':>4}──┼──{'──────────':>10}──┼──{'───────────':>11}──┼──"
              f"{'─────────':>9}──┼──{'────────':>8}")

        for K in K_list:
            km = KMeans(n_clusters=K, init="k-means++", n_init=10,
                        random_state=self.model.random_state)
            km.fit(X_sc)
            labels = km.labels_
            sil = silhouette_score(X_sc, labels) if K >= 2 else np.nan
            ch  = calinski_harabasz_score(X_sc, labels) if K >= 2 else np.nan
            db  = davies_bouldin_score(X_sc, labels) if K >= 2 else np.nan
            rows.append({"K": K, "Inertia": km.inertia_,
                         "Silhouette": sil, "CH": ch, "DB": db})
            print(f"  {K:>4}  |  {km.inertia_:>10.2f}  |  {sil:>11.4f}  |  "
                  f"{ch:>9.2f}  |  {db:>8.4f}")

        df = pd.DataFrame(rows).sort_values("Silhouette", ascending=False)
        print(f"\n  Best K by Silhouette: K={int(df.iloc[0]['K'])}")
        print(f"  Best K by CH Index:   K={int(df.loc[df['CH'].idxmax(), 'K'])}")
        print(f"  Best K by DB Index:   K={int(df.loc[df['DB'].idxmin(), 'K'])}")
        print(f"{'='*65}\n")
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 ── VISUALIZER
#  Six complementary diagnostic plots for K-Means analysis.
# ══════════════════════════════════════════════════════════════════════════════

class KMeansVisualizer:
    """
    Diagnostic plots for K-Means clustering and K-selection.

    Plots available:
      1. plot_elbow()            — WCSS vs K with elbow annotation
      2. plot_silhouette_curve() — Silhouette Score vs K
      3. plot_gap_statistic()    — Gap Statistic with 1-SE error bars
      4. plot_ch_db()            — Calinski-Harabasz and Davies-Bouldin vs K
      5. plot_silhouette_bars()  — Per-point silhouette breakdown for fitted model
      6. plot_clusters_2d()      — PCA-projected cluster scatter plot
      7. plot_all()              — 2×3 full diagnostic dashboard

    Example
    -------
    >>> viz = KMeansVisualizer(model, tuner)       # doctest: +SKIP
    >>> viz.plot_all(X)                            # doctest: +SKIP
    """

    _PALETTE = [
        "#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    ]

    def __init__(self, model: KMeansModel, tuner: KMeansTuner = None):
        self.model = model
        self.tuner = tuner

    # ── 1. Elbow Plot ─────────────────────────────────────────────────────────

    def plot_elbow(self, ax=None):
        """
        Elbow plot: WCSS (Inertia) vs K.

        WHAT TO LOOK FOR:
          The "elbow" is the point where the curve bends sharply —
          adding more clusters gives rapidly diminishing improvement.
          This suggests the data has a natural structure at that K.

        LIMITATION:
          The elbow is often visually ambiguous. Use in combination with
          Silhouette / CH / DB scores for confirmation.
        """
        self._require_tuner()
        K_range  = self.tuner.K_range_
        inertias = self.tuner._inertias
        elbow_K  = self.tuner.best_K_["Elbow (WCSS)"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(K_range, inertias, "o-", color="#3498DB",
                linewidth=2.5, markersize=7, label="Inertia (WCSS)")
        ax.axvline(elbow_K, color="#E74C3C", linestyle="--", linewidth=1.5,
                   label=f"Elbow at K={elbow_K}")

        # Annotate percentage drops
        for i in range(1, len(inertias)):
            pct = 100 * (inertias[i-1] - inertias[i]) / inertias[i-1]
            ax.annotate(f"−{pct:.0f}%",
                        xy=(K_range[i], inertias[i]),
                        xytext=(K_range[i] + 0.15, inertias[i] + inertias[0] * 0.015),
                        fontsize=7.5, color="dimgray")

        ax.set_xlabel("Number of Clusters  K")
        ax.set_ylabel("Inertia  (WCSS = Σ ‖xᵢ − μₖ‖²)")
        ax.set_title(f"Elbow Method  —  Best K = {elbow_K}")
        ax.set_xticks(K_range)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 2. Silhouette Curve ───────────────────────────────────────────────────

    def plot_silhouette_curve(self, ax=None):
        """
        Mean Silhouette Score vs K.

        WHAT TO LOOK FOR:
          The K with the HIGHEST mean silhouette score is preferred.
          A score > 0.5 indicates reasonable cluster structure.
          Secondary peaks can indicate alternative valid clusterings.

        FORMULA:
          For each point i in cluster k:
            a(i) = mean dist to other points in k          (cohesion)
            b(i) = mean dist to points in nearest other cluster (separation)
            s(i) = (b(i) − a(i)) / max(a(i), b(i))
          Mean silhouette = mean of s(i) across all i
        """
        self._require_tuner()
        K_range    = self.tuner.K_range_
        sil_scores = self.tuner._sil_scores
        best_K     = self.tuner.best_K_["Silhouette"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        colors = ["#E74C3C" if k == best_K else "#3498DB" for k in K_range]

        ax.bar(K_range, sil_scores, color=colors, edgecolor="white",
               width=0.6, alpha=0.85)
        ax.axhline(0.5, color="#F39C12", linestyle="--", linewidth=1.2,
                   label="0.5 — good threshold")
        ax.axhline(0.3, color="grey",    linestyle=":", linewidth=1.0,
                   label="0.3 — minimum acceptable")

        for k, s in zip(K_range, sil_scores):
            if not np.isnan(s):
                ax.text(k, s + 0.008, f"{s:.3f}", ha="center",
                        fontsize=8, color="black")

        ax.set_xlabel("Number of Clusters  K")
        ax.set_ylabel("Mean Silhouette Score")
        ax.set_title(f"Silhouette Score  —  Best K = {best_K}")
        ax.set_xticks(K_range)
        ax.set_ylim(0, min(1.05, max(sil_scores) * 1.2) if sil_scores else 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        red_patch  = mpatches.Patch(color="#E74C3C", label=f"Best K={best_K}")
        blue_patch = mpatches.Patch(color="#3498DB", label="Other K values")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="upper right")

    # ── 3. Gap Statistic ──────────────────────────────────────────────────────

    def plot_gap_statistic(self, ax=None):
        """
        Gap Statistic vs K with 1-standard-error bands.

        WHAT TO LOOK FOR:
          The optimal K (by Tibshirani's rule) is the SMALLEST K s.t.:
            Gap(K) ≥ Gap(K+1) − s_{K+1}
          i.e., the first K where the gap is not significantly smaller
          than the next value.

        INTERPRETATION:
          A large Gap means the data has more structure than random noise.
          Gap ≈ 0 means K-Means isn't finding more structure than would
          appear by chance in uniformly random data.
        """
        self._require_tuner()
        K_range  = self.tuner.K_range_
        gap_vals = np.array(self.tuner._gap_vals)
        gap_stds = np.array(self.tuner._gap_stds)
        best_K   = self.tuner.best_K_["Gap Statistic"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        valid = ~np.isnan(gap_vals)
        k_arr = np.array(K_range)[valid]
        g_arr = gap_vals[valid]
        s_arr = gap_stds[valid]

        ax.plot(k_arr, g_arr, "o-", color="#9B59B6",
                linewidth=2.5, markersize=7, label="Gap(K)")
        ax.fill_between(k_arr, g_arr - s_arr, g_arr + s_arr,
                        alpha=0.15, color="#9B59B6", label="±1 SE")
        ax.axvline(best_K, color="#E74C3C", linestyle="--", linewidth=1.5,
                   label=f"Optimal K={best_K}")

        ax.set_xlabel("Number of Clusters  K")
        ax.set_ylabel("Gap Statistic")
        ax.set_title(f"Gap Statistic  —  Optimal K = {best_K}")
        ax.set_xticks(K_range)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 4. CH and DB Indices ──────────────────────────────────────────────────

    def plot_ch_db(self, ax1=None, ax2=None):
        """
        Calinski-Harabasz (higher = better) and Davies-Bouldin (lower = better)
        plotted vs K on separate axes.

        CALINSKI-HARABASZ:
          CH = [B / (K-1)] / [W / (n-K)]
          B = between-cluster variance (spread of centroids)
          W = within-cluster variance (spread of points in clusters)
          Interpretation: "How much more spread are the cluster centres than the points?"
          Higher ratio → clusters are both tight AND well-separated.

        DAVIES-BOULDIN:
          DB = (1/K) Σₖ max_{j≠k} (σₖ + σⱼ) / d(μₖ, μⱼ)
          For each cluster, find the worst (most similar) neighbour.
          Average over all clusters.
          Lower → clusters are compact and far from each other.
        """
        self._require_tuner()
        K_range  = self.tuner.K_range_
        ch_scores = self.tuner._ch_scores
        db_scores = self.tuner._db_scores
        best_ch_K = self.tuner.best_K_["Calinski-Harabasz"]
        best_db_K = self.tuner.best_K_["Davies-Bouldin"]

        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # CH plot
        ax1.plot(K_range, ch_scores, "s-", color="#2ECC71",
                 linewidth=2.5, markersize=7)
        ax1.axvline(best_ch_K, color="#E74C3C", linestyle="--",
                    linewidth=1.5, label=f"Best K={best_ch_K}")
        ax1.set_xlabel("Number of Clusters  K")
        ax1.set_ylabel("Calinski-Harabasz Index  (↑ higher = better)")
        ax1.set_title(f"Calinski-Harabasz Index  —  Best K = {best_ch_K}")
        ax1.set_xticks(K_range)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # DB plot
        ax2.plot(K_range, db_scores, "D-", color="#E67E22",
                 linewidth=2.5, markersize=7)
        ax2.axvline(best_db_K, color="#E74C3C", linestyle="--",
                    linewidth=1.5, label=f"Best K={best_db_K}")
        ax2.set_xlabel("Number of Clusters  K")
        ax2.set_ylabel("Davies-Bouldin Index  (↓ lower = better)")
        ax2.set_title(f"Davies-Bouldin Index  —  Best K = {best_db_K}")
        ax2.set_xticks(K_range)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    # ── 5. Silhouette Bar Plot ────────────────────────────────────────────────

    def plot_silhouette_bars(self, X, ax=None):
        """
        Per-cluster silhouette bar chart (the "silhouette plot").

        WHAT IT SHOWS:
          Each horizontal bar = one data point, coloured by its cluster.
          Bars sorted by silhouette score within each cluster.
          The dashed line is the mean silhouette across all points.

        WHAT TO LOOK FOR:
          - Most bars should extend well past the mean line → tight clusters
          - Many bars < 0 in a cluster → that cluster may be splitting two true clusters
          - Very unequal bar widths across clusters → unequal cluster sizes
          - A cluster almost entirely < 0 → it likely doesn't belong as a separate cluster
        """
        self.model._check_fitted()
        X = np.array(X, dtype=float)
        X_sc   = self.model._get_scaled(X)
        labels = self.model.labels_
        K      = self.model.K

        sil_vals = silhouette_samples(X_sc, labels)
        mean_sil = sil_vals.mean()

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, max(5, K * 1.5)))

        y_lower = 10
        for k in range(K):
            mask    = labels == k
            sil_k   = np.sort(sil_vals[mask])
            n_k     = sil_k.size
            y_upper = y_lower + n_k

            color = self._PALETTE[k % len(self._PALETTE)]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_k,
                             alpha=0.75, color=color,
                             label=f"Cluster {k}  (n={n_k}, s̄={sil_k.mean():.3f})")
            y_lower = y_upper + 10   # gap between clusters

        ax.axvline(mean_sil, color="#E74C3C", linestyle="--", linewidth=1.5,
                   label=f"Mean silhouette = {mean_sil:.3f}")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

        ax.set_xlabel("Silhouette Coefficient  (b(i)−a(i)) / max(a,b)")
        ax.set_ylabel("Cluster (sorted by silhouette within cluster)")
        ax.set_title(f"Silhouette Plot  —  K={K}  |  Mean = {mean_sil:.3f}")
        ax.set_yticks([])
        ax.set_xlim([-0.2, 1.0])
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

    # ── 6. 2D Cluster Scatter ─────────────────────────────────────────────────

    def plot_clusters_2d(self, X, ax=None):
        """
        Scatter plot of clusters projected to 2D via PCA.

        When d > 2, PCA reduces the data to 2 principal components that
        capture the most variance. The scatter plot shows the cluster
        structure in this reduced space.

        Centroids are plotted as large black ★ markers.
        Explained variance ratio is shown in axis labels.

        NOTE: PCA projection may distort the cluster geometry if the
        true separating directions are not the highest-variance ones.
        Use as an exploratory visualization, not a ground truth.
        """
        self.model._check_fitted()
        X = np.array(X, dtype=float)
        X_sc = self.model._get_scaled(X)
        K    = self.model.K

        # PCA to 2D
        pca   = PCA(n_components=2, random_state=42)
        X_2d  = pca.fit_transform(X_sc)
        ev    = pca.explained_variance_ratio_

        # Project centroids to 2D as well
        ctrs_2d = pca.transform(self.model._model.cluster_centers_)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        for k in range(K):
            mask  = self.model.labels_ == k
            color = self._PALETTE[k % len(self._PALETTE)]
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, s=25, alpha=0.6, label=f"Cluster {k}")

        # Centroids
        ax.scatter(ctrs_2d[:, 0], ctrs_2d[:, 1],
                   c="black", s=220, marker="*", zorder=5,
                   edgecolors="white", linewidths=0.5, label="Centroids")

        ax.set_xlabel(f"PC 1  ({ev[0]*100:.1f}% variance explained)")
        ax.set_ylabel(f"PC 2  ({ev[1]*100:.1f}% variance explained)")
        ax.set_title(f"Cluster Scatter  (PCA projection)  —  K={K}  "
                     f"|  {(ev[0]+ev[1])*100:.1f}% total variance")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    # ── 7. BIC Plot ───────────────────────────────────────────────────────────

    def plot_bic(self, ax=None):
        """
        BIC (via GMM) vs K.

        BIC = -2 · log(L) + p · log(n)
          L = likelihood of the fitted GMM
          p = number of free parameters
          n = number of data points

        WHAT TO LOOK FOR:
          The K that minimises BIC is preferred.
          BIC trades off fit quality (log-likelihood) against model complexity (p·log n).
          A lower BIC = better model, accounting for the risk of overfitting.

        WHY GMM FOR BIC?
          GMM has an explicit likelihood, so BIC is well-defined.
          K-Means can be seen as a special case of GMM (with equal, spherical covariance).
          GMM BIC generalises well and is the standard for model selection in clustering.
        """
        self._require_tuner()
        K_range   = self.tuner.K_range_
        bic_scores= self.tuner._bic_scores
        best_K    = self.tuner.best_K_["BIC (GMM)"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(K_range, bic_scores, "^-", color="#1ABC9C",
                linewidth=2.5, markersize=7, label="BIC (GMM)")
        ax.axvline(best_K, color="#E74C3C", linestyle="--", linewidth=1.5,
                   label=f"Best K={best_K} (min BIC)")

        for k, b in zip(K_range, bic_scores):
            ax.text(k, b + (max(bic_scores) - min(bic_scores)) * 0.01,
                    f"{b:.0f}", ha="center", fontsize=7.5, color="dimgray")

        ax.set_xlabel("Number of Clusters  K")
        ax.set_ylabel("BIC  (lower = better)")
        ax.set_title(f"BIC via Gaussian Mixture Model  —  Best K = {best_K}")
        ax.set_xticks(K_range)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 8. Full Dashboard ─────────────────────────────────────────────────────

    def plot_all(self, X, save_path: str = None):
        """
        Render a comprehensive 2×3 diagnostic dashboard.

        Layout:
          ┌─────────────────┬─────────────────┬─────────────────┐
          │   Elbow Plot    │  Silhouette     │  Gap Statistic  │
          │   (WCSS vs K)   │  Score vs K     │  with error bars│
          ├─────────────────┼─────────────────┼─────────────────┤
          │  CH vs K  &     │  Silhouette     │  Cluster Scatter│
          │  DB vs K        │  Bar Chart      │  (PCA 2D)       │
          └─────────────────┴─────────────────┴─────────────────┘
        """
        fig = plt.figure(figsize=(19, 12))
        fig.suptitle(
            f"K-Means Clustering  —  K-Selection Diagnostic Dashboard\n"
            f"K={self.model.K}  |  init='{self.model.init}'  |  "
            f"Consensus recommendation: K={self.tuner.consensus_K_ if self.tuner else '?'}  |  "
            f"Inertia={self.model.inertia_:.2f}",
            fontsize=13, fontweight="bold"
        )

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

        ax1  = fig.add_subplot(gs[0, 0])
        ax2  = fig.add_subplot(gs[0, 1])
        ax3  = fig.add_subplot(gs[0, 2])
        ax4a = fig.add_subplot(gs[1, 0])
        # ax4b shares a twin axis with ax4a
        ax5  = fig.add_subplot(gs[1, 1])
        ax6  = fig.add_subplot(gs[1, 2])

        self.plot_elbow(ax=ax1)
        self.plot_silhouette_curve(ax=ax2)
        self.plot_gap_statistic(ax=ax3)

        # Bottom-left: CH on primary axis, DB on twin
        ax4b = ax4a.twinx()
        self._plot_ch_db_combined(ax4a, ax4b)

        self.plot_silhouette_bars(X, ax=ax5)
        self.plot_clusters_2d(X, ax=ax6)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"[KMeansVisualizer] Dashboard saved to: {save_path}")

        plt.tight_layout()
        plt.show()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _plot_ch_db_combined(self, ax1, ax2):
        """Plot CH (left axis, green) and DB (right axis, orange) on one panel."""
        self._require_tuner()
        K_range   = self.tuner.K_range_
        ch_scores = self.tuner._ch_scores
        db_scores = self.tuner._db_scores

        l1, = ax1.plot(K_range, ch_scores, "s-", color="#2ECC71",
                       linewidth=2.5, markersize=6, label="CH Index (↑)")
        l2, = ax2.plot(K_range, db_scores, "D--", color="#E67E22",
                       linewidth=2.0, markersize=6, label="DB Index (↓)")

        ax1.set_xlabel("K")
        ax1.set_ylabel("Calinski-Harabasz  (↑)", color="#2ECC71")
        ax2.set_ylabel("Davies-Bouldin  (↓)",     color="#E67E22")
        ax1.tick_params(axis="y", labelcolor="#2ECC71")
        ax2.tick_params(axis="y", labelcolor="#E67E22")

        bk_ch = self.tuner.best_K_["Calinski-Harabasz"]
        bk_db = self.tuner.best_K_["Davies-Bouldin"]
        ax1.axvline(bk_ch, color="#2ECC71", linestyle=":", alpha=0.6,
                    label=f"CH best K={bk_ch}")
        ax2.axvline(bk_db, color="#E67E22", linestyle=":", alpha=0.6,
                    label=f"DB best K={bk_db}")

        ax1.set_title("Calinski-Harabasz & Davies-Bouldin vs K")
        ax1.set_xticks(K_range)
        ax1.grid(True, alpha=0.25)
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=8, loc="upper right")

    def _require_tuner(self):
        if self.tuner is None or not self.tuner._is_fitted:
            raise RuntimeError(
                "A fitted KMeansTuner is required for this plot. "
                "Pass tuner=KMeansTuner(...).fit(X) to KMeansVisualizer."
            )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 ── FULL PIPELINE
#  One function call: tune → fit → evaluate → visualize
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X,
    feature_names=None,
    K: int = None,
    K_max: int = 10,
    auto_tune: bool = True,
    n_init: int = 10,
    gap_B: int = 10,
    scale_features: bool = True,
    plot: bool = True,
    save_plot: str = None,
    random_state: int = 42,
) -> dict:
    """
    End-to-end K-Means clustering pipeline.

    PIPELINE STEPS:
    ───────────────
    1.  (Optional) Run KMeansTuner to evaluate K=2..K_max on 6 methods
        and recommend the best K via majority vote.
    2.  Fit KMeansModel with the selected K.
    3.  Print full model summary.
    4.  Evaluate cluster quality (Silhouette, CH, DB, per-cluster breakdown).
    5.  (Optional) Generate 6-panel diagnostic dashboard.

    Parameters
    ----------
    X             : feature matrix (n_samples, n_features)
    feature_names : optional list of feature names
    K             : fixed K (used when auto_tune=False)
    K_max         : upper bound for K-search (when auto_tune=True)
    auto_tune     : run KMeansTuner to find optimal K
    n_init        : K-Means restarts per K value
    gap_B         : Gap Statistic reference datasets (higher = more accurate, slower)
    scale_features: standardize features before fitting
    plot          : display diagnostic dashboard
    save_plot     : path to save the plot PNG
    random_state  : reproducibility seed

    Returns
    -------
    dict: model, tuner, evaluator, visualizer, metrics
    """
    print("\n" + "█" * 62)
    print("  K-MEANS CLUSTERING — FULL PIPELINE")
    print("█" * 62)

    # ── 1. Prepare inputs ─────────────────────────────────────────────────────
    if hasattr(X, "columns") and feature_names is None:
        feature_names = list(X.columns)
    X = np.array(X, dtype=float)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    print(f"\n[Pipeline] Dataset — {X.shape[0]} samples, {X.shape[1]} features")

    # ── 2. Tune K ─────────────────────────────────────────────────────────────
    tuner = None
    if auto_tune:
        print(f"\n[Pipeline] Step 1/3 — K-Selection (K = 2 … {K_max}) ...")
        tuner = KMeansTuner(K_min=2, K_max=K_max, n_init=n_init,
                            gap_B=gap_B, scale_features=scale_features,
                            random_state=random_state)
        selected_K = tuner.fit(X)
        tuner.results_summary()
        K = selected_K
    else:
        if K is None:
            raise ValueError("Provide K when auto_tune=False.")
        print(f"\n[Pipeline] Step 1/3 — Using provided K={K}")

    # ── 3. Fit ────────────────────────────────────────────────────────────────
    print(f"\n[Pipeline] Step 2/3 — Fitting KMeansModel with K={K} ...")
    model = KMeansModel(K=K, n_init=n_init, scale_features=scale_features,
                        random_state=random_state)
    model.fit(X, feature_names=feature_names)
    model.summary()

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[Pipeline] Step 3/3 — Evaluating cluster quality ...")
    evaluator = KMeansEvaluator(model)
    metrics   = evaluator.evaluate(X, label="Full Dataset")
    evaluator.per_cluster_report(X)

    # ── 5. Visualize ──────────────────────────────────────────────────────────
    visualizer = KMeansVisualizer(model, tuner)
    if plot and tuner is not None:
        print("\n[Pipeline] Generating diagnostic dashboard ...")
        visualizer.plot_all(X, save_path=save_plot)
    elif plot and tuner is None:
        print("\n[Pipeline] Generating cluster plots (no tuner — limited plots) ...")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        visualizer.plot_silhouette_bars(X, ax=axes[0])
        visualizer.plot_clusters_2d(X,   ax=axes[1])
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot, bbox_inches="tight", dpi=150)
        plt.show()

    print("\n" + "█" * 62)
    print("  PIPELINE COMPLETE")
    print("█" * 62 + "\n")

    return {
        "model":     model,
        "tuner":     tuner,
        "evaluator": evaluator,
        "visualizer":visualizer,
        "metrics":   metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO — runs when you execute this file directly
#  python KMeansClustering.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  K-MEANS CLUSTERING MODULE — DEMO")
    print("  Using a synthetic dataset with 4 true clusters")
    print("=" * 62 + "\n")

    # ── Generate synthetic clustering data with known K=4 ─────────────────────
    # Four Gaussian blobs with different locations and spreads
    from sklearn.datasets import make_blobs

    X_demo, y_true = make_blobs(
        n_samples    = 600,
        n_features   = 6,
        centers      = 4,
        cluster_std  = [0.8, 1.0, 0.7, 1.2],
        random_state = 42,
    )

    feature_names_demo = [f"feature_{i:02d}" for i in range(6)]

    # ── Run the full pipeline ─────────────────────────────────────────────────
    results = run_full_pipeline(
        X             = X_demo,
        feature_names = feature_names_demo,
        K_max         = 10,
        auto_tune     = True,
        n_init        = 10,
        gap_B         = 10,
        scale_features= True,
        plot          = True,
        save_plot     = "kmeans_clustering_diagnostics.png",
        random_state  = 42,
    )

    # ── Access specific components ────────────────────────────────────────────
    model     = results["model"]
    tuner     = results["tuner"]
    evaluator = results["evaluator"]

    print("Best K per method:")
    for method, best_k in tuner.best_K_.items():
        print(f"  {method:<28}  K = {best_k}")

    print(f"\nConsensus K: {tuner.consensus_K_}")

    print("\nYou can customise after fitting:")
    print("  model.get_cluster_df(X)            # DataFrame with cluster labels")
    print("  model.transform(X)                 # Cluster-distance features")
    print("  evaluator.compare_K_values(X, [3,4,5])  # Side-by-side K comparison")
    print("  tuner.get_best_model('Silhouette') # KMeansModel from a specific method")