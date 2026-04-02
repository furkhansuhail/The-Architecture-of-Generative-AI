OPERATIONS   = {}
VISUAL_HTML  = ""

"""Module: 01 · K-Means Clustering"""

"""
K-Means Clustering — The Foundation of Unsupervised Learning
=============================================================

K-Means is the most widely used clustering algorithm in machine learning.
Before autoencoders, before DBSCAN, before Gaussian Mixture Models —
there was partitioning data into K groups by minimising intra-cluster variance.
Every more sophisticated clustering method you will study is an extension or
generalisation of the ideas introduced here.
"""

import math
import os

DISPLAY_NAME = "01 · K-Means Clustering"
ICON         = "🔵"
SUBTITLE     = "Partition data into K groups by minimising within-cluster variance"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
What is Clustering in Machine Learning?

Clustering is the task of grouping a set of objects so that objects in the same
group (called a cluster) are more similar to each other than to objects in other
groups. Unlike supervised learning, there are no labels — the algorithm must
discover structure in the data entirely on its own.

This places clustering firmly in the domain of unsupervised learning: we have
input data X but no target labels y. The model must find its own notion of what
counts as "similar" and organise the data accordingly.

Real-world clustering problems:
    * Group customers by purchasing behaviour for targeted marketing
    * Segment medical images into tissue types without labelled scans
    * Identify topics in a large corpus of text documents
    * Detect anomalies by finding points that belong to no cluster
    * Compress images by representing each pixel with its cluster centroid colour

K-Means is the most important starting point for all of these.


## 01 · K-Means Clustering

K-Means partitions n data points into exactly K non-overlapping clusters by
minimising the total within-cluster variance. It is one of the oldest and most
studied algorithms in all of computer science, first described by Stuart Lloyd in
1957 (and independently by MacQueen in 1967, who coined the term "K-Means").

Despite its age, K-Means remains the dominant choice for clustering due to its
simplicity, speed, and surprising effectiveness across a vast range of problems.


### What is K-Means Clustering?

K-Means is an iterative algorithm that alternates between two steps:
    1. Assign each data point to the nearest centroid
    2. Move each centroid to the mean of its assigned points

These two steps are repeated until the assignments no longer change — at which
point the algorithm has converged to a local minimum of the objective function.

The Real-World Analogy:

Imagine you drop K pins onto a map of customer locations. Each customer goes to
their nearest pin. Then each pin moves to the centre of its customers. Repeat.
The pins will drift toward natural hubs in your customer base and stop moving
when they have settled into the natural "centres" of each group.

    Things that exist inside the model (learnable parameters):
        - Centroids (μ₁, μ₂, ..., μₖ) — one position vector per cluster
        - Assignments (zᵢ) — which cluster each point belongs to

    Things that exist only at setup time (hyperparameters / configuration):
        - K — the number of clusters (the critical hyperparameter)
        - Initialisation strategy — random, K-Means++, or manual
        - Maximum iterations — a safety cap on the loop
        - Convergence tolerance — how small a centroid shift to accept as "done"
        - Distance metric — usually Euclidean, but others exist


### K-Means as Optimisation (ERM for Unsupervised Learning)

K-Means is best understood as an optimisation problem with a specific objective:

    Objective function to MINIMISE (called Within-Cluster Sum of Squares, WCSS):

        J = Σᵢ₌₁ⁿ  Σₖ₌₁ᴷ  1[zᵢ = k] · ‖xᵢ − μₖ‖²

    where:
        n       — number of data points
        K       — number of clusters
        xᵢ      — the i-th data point (a vector in ℝᵈ)
        μₖ      — the centroid of cluster k (also in ℝᵈ)
        zᵢ      — the cluster assignment of point i  (zᵢ ∈ {1, ..., K})
        1[·]    — indicator function (1 if true, 0 otherwise)
        ‖·‖²    — squared Euclidean distance

    In plain English: for each data point, measure the squared distance to its
    assigned centroid and sum up all those distances. Find assignments and
    centroids that make this total as small as possible.

This is also called the "inertia" of the clustering. Lower inertia = tighter,
more compact clusters.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │  K-Means as ERM:                                                         │
    │                                                                          │
    │  Hypothesis class:  H = { all possible K-partitions of ℝᵈ }              │
    │                     (each centroid defines a Voronoi cell)               │
    │                                                                          │
    │  Loss function:     L(xᵢ, μₖ) = ‖xᵢ − μₖ‖²   (squared Euclidean)          │
    │                                                                          │
    │  Training objective:                                                     │
    │      min_{μ, z}   Σᵢ ‖xᵢ − μ_{zᵢ}‖²                                      │
    │                   ↑                                                      │
    │     This is jointly over centroid positions AND assignments              │
    │                                                                          │
    │  Optimiser:   Lloyd's Algorithm (alternating minimisation)               │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

This problem is NP-hard to solve exactly (finding the globally optimal partition
for general K and n). Lloyd's algorithm finds a local optimum efficiently — and
in practice, the local optima found are often very close to the global one,
especially with good initialisation.


---

### Part 1: The K-Means Algorithm — Lloyd's Method

The algorithm operates in three phases:

Phase 1 — Initialisation:
    Choose K initial centroids. (How we choose them matters a lot — see Part 4.)

Phase 2 — Iteration (repeat until convergence):

    Step A — Assignment Step:
        For each data point xᵢ, find the nearest centroid:
            zᵢ = argmin_k  ‖xᵢ − μₖ‖²

        This partitions the space into K Voronoi cells — regions where every
        point is closer to centroid μₖ than to any other centroid.

    Step B — Update Step:
        For each cluster k, move the centroid to the mean of all assigned points:
            μₖ ← (1 / |Cₖ|) · Σᵢ:zᵢ=k  xᵢ

        where Cₖ is the set of points currently assigned to cluster k.

Phase 3 — Convergence Check:
    Stop when either:
        - No assignment has changed between iterations, OR
        - Centroid movement is below a tolerance threshold ε, OR
        - Maximum iterations reached

    # =======================================================================================
    Diagram 1 — Lloyd's Algorithm on 2D Data (K=3):

    ITERATION 0 (Initialisation):        ITERATION 1 (After Assignment):
    ·  · ·  ·  · ·     ×                ·  · ·  ·  · ·
         ·  · ·  ·          ×             ↑cluster 1       × (centroid)
    ·       ·      · ×                  ·       ·      ·
    ↑random centroids (×)               assignments drawn by proximity

    ITERATION 2 (After Update):          ITERATION 3 (Converged):
    ·  · ●  ·  · ·                      ·  · ●  ·  · ·
         ·  · ·  ·                           ·  · ·  ·
    ·       ·  ●  ·                      ·       ·  ●  ·
               ●                                   ●
    ↑centroids moved to cluster means    ↑no further movement — DONE

    Each × is a centroid. Each cluster's points share a colour/symbol.
    Voronoi boundaries define which region "belongs" to each centroid.
    # =======================================================================================


Key mathematical property: Each step of Lloyd's algorithm monotonically decreases
(or maintains) the objective J. This is because:
    - The assignment step finds the best assignments given fixed centroids
    - The update step finds the best centroids given fixed assignments
    - Both steps can only decrease or maintain J — never increase it
    - J is bounded below by 0
    → Therefore Lloyd's algorithm is guaranteed to converge.

However, it converges to a LOCAL minimum of J, not necessarily the global one.


---

### Part 2: The Distance Metric — Euclidean and Its Implications

The standard K-Means algorithm uses squared Euclidean distance:

        d²(a, b) = Σⱼ (aⱼ − bⱼ)²  =  ‖a − b‖²

Why squared Euclidean specifically?
    • The centroid update (mean) is the closed-form minimiser of squared Euclidean distance
    • This makes the Update Step analytically tractable — no optimisation needed
    • If you use a different distance metric, the Update Step changes (or has no closed form)

The mean minimises squared distance:
    μ* = argmin_μ  Σᵢ ‖xᵢ − μ‖²    →    μ* = (1/n) Σᵢ xᵢ   ← the sample mean

Proof:
    ∂/∂μ  Σᵢ ‖xᵢ − μ‖²  =  -2 Σᵢ (xᵢ − μ)  =  0
    →  Σᵢ xᵢ  =  n · μ
    →  μ  =  (1/n) Σᵢ xᵢ     QED

This is why the Update Step is simply taking the mean. Any other distance metric
would require a different (and often iterative) centroid update step.

Implications for data preprocessing:
    • K-Means uses raw Euclidean distance — feature scaling is CRITICAL
    • A feature with range [0, 10000] will dominate a feature with range [0, 1]
    • Always StandardScale (zero mean, unit variance) or MinMaxScale before K-Means
    • Failure to scale is one of the most common K-Means mistakes


Alternatives to Euclidean distance:
    Metric          Update Step         Algorithm Variant
    ───────────────────────────────────────────────────────
    Squared L2      Mean                K-Means (standard)
    L1 (Manhattan)  Geometric Median    K-Medians
    Any metric      Medoid (data point) K-Medoids / PAM
    Cosine          Normalised mean     Spherical K-Means (for text/NLP)


---

### Part 3: Convergence — Guarantees and Geometry

K-Means always converges, but not necessarily to the global optimum.

Convergence theorem:
    Since there are a finite number of possible assignment configurations (at most
    K^n, though in practice far fewer are visited), and each iteration strictly
    decreases (or maintains) J, the algorithm must eventually revisit a configuration.
    Since J is non-increasing and finite configurations exist, convergence is guaranteed.

In practice:
    - Most datasets converge in 10–300 iterations
    - Convergence is often much faster than the maximum iterations cap
    - The bound K^n is extremely loose — practical convergence is polynomial

Local vs Global Minima:
    The K-Means objective J has many local minima. The number of local minima
    grows rapidly with K and n. This is why:
        1. Initialisation matters enormously (see Part 4)
        2. Best practice is to run K-Means multiple times with different initialisation
        3. Select the run with the lowest final J value

    # =======================================================================================
    Diagram 2 — Local Minima: Two Different Initialisations, Two Different Results

    Bad initialisation result:           Good initialisation result:
    ●● ●●●  ○○○ ○                        ●●●●●  ○○○○   △△△
    ●●  ●●●  ○ ○○○    △△△              ●  ●●●  ○○○○   △△△△
    ●●   ●●  ○○○  ○  △△                 ●●●●   ○○○○   △△△

    ← WCSS = 485 (suboptimal)           ← WCSS = 203 (much better)

    The left result happened because two centroids started near the same cluster.
    The right result had one centroid per natural group — found the global optimum.
    # =======================================================================================

Practical advice:
    - scikit-learn's KMeans uses n_init=10 by default (10 random restarts)
    - With KMeans++ initialisation, even n_init=1 is often sufficient
    - For very large datasets, MiniBatchKMeans can be used with more restarts efficiently


---

### Part 4: Initialisation Strategies — The Critical First Step

How you initialise the centroids has an enormous effect on the quality of the
final solution. Poor initialisation leads to slow convergence and bad local optima.

────────────────────────────────────────────────────────────
Strategy 1: Random (Forgy Method)
────────────────────────────────────────────────────────────
    Pick K data points uniformly at random as the initial centroids.

    Pros:   Simple, fast
    Cons:   High chance of two centroids starting in the same cluster
            Can converge to very bad local optima
            High variance across runs — results differ substantially

    # =======================================================================================
    Diagram 3 — Forgy vs K-Means++:

    Forgy (random) — BAD case:           K-Means++ — systematically spread:
    ● ● ● ● ● ●                          ● ● ● ● ● ●
    ×  ×                                 ×
           ● ● ● ●                            ● ● ● ●
               ● ● ● ●                            × ● ● ●
                                                        ×
    Two centroids × in same cluster.    Centroids spread across all clusters.
    Third cluster gets no centroid.     Each natural group gets a centroid.
    # =======================================================================================

────────────────────────────────────────────────────────────
Strategy 2: K-Means++ (Arthur & Vassilvitskii, 2007)
────────────────────────────────────────────────────────────
    A smarter probabilistic initialisation that spreads centroids across the data.

    Algorithm:
        1. Choose the first centroid μ₁ uniformly at random from the data points
        2. For k = 2, 3, ..., K:
               a. For each data point xᵢ, compute D(xᵢ) = min_j ‖xᵢ - μⱼ‖²
                  (the squared distance to the nearest already-chosen centroid)
               b. Choose the next centroid μₖ by sampling from the data with
                  probability proportional to D(xᵢ):
                        P(xᵢ chosen) = D(xᵢ) / Σⱼ D(xⱼ)

    Why this works:
        - Points far from all existing centroids are more likely to be chosen next
        - This ensures centroids are spread across the data space
        - Each natural cluster is likely to receive at least one centroid
        - Results in 2–10x fewer iterations to converge vs random initialisation

    Theoretical guarantee:
        K-Means++ with subsequent Lloyd's algorithm achieves an expected cost
        within O(log K) of the optimal cost. This is the only known initialisation
        with a provable approximation guarantee.

    Pros:   Much better convergence, lower final WCSS, provable guarantee
    Cons:   Slightly slower initialisation (O(nKd) instead of O(Kd))
            Still has variance — but much less than random

    In practice, K-Means++ is the default choice in all major libraries.
    scikit-learn uses init='k-means++' by default.

────────────────────────────────────────────────────────────
Strategy 3: Manual / Domain-Guided Initialisation
────────────────────────────────────────────────────────────
    When you have strong prior knowledge about where clusters should be, you can
    initialise centroids manually. This is useful when:
        - Running K-Means incrementally on streaming data
        - Fine-tuning a previously trained clustering
        - Seeding from a coarser algorithm (e.g., hierarchical clustering)


---

### Part 5: Choosing K — The Central Challenge

K is the single most important hyperparameter in K-Means. There is no universally
correct method to choose it, but several principled techniques exist.

────────────────────────────────────────────────────────────
Method 1: The Elbow Method
────────────────────────────────────────────────────────────
    Run K-Means for K = 1, 2, 3, ..., K_max. Plot WCSS (inertia) vs K.

    As K increases, WCSS always decreases — with K=n, every point is its own
    centroid and WCSS = 0. What we look for is the "elbow" — the point where
    adding one more cluster gives a rapidly diminishing return.

    # =======================================================================================
    Diagram 4 — The Elbow Plot:

    WCSS
    (Inertia)
       │
    800│ ×
       │
    600│    ×
       │
    400│       ×  ← ELBOW (diminishing returns after K=3)
       │          ×
    200│              ×
       │                  × × ×
     0 └────────────────────────────── K
       1    2    3    4    5    6    7

    The "elbow" at K=3 suggests 3 clusters is the right choice.
    # =======================================================================================

    Limitation: The elbow is often ambiguous — sometimes it's a smooth curve with
    no clear bend. Never rely solely on the elbow method.

────────────────────────────────────────────────────────────
Method 2: Silhouette Score
────────────────────────────────────────────────────────────
    The Silhouette Score quantifies how well each point fits its assigned cluster
    versus the next-nearest cluster. It combines cohesion and separation.

    For point i:
        a(i) = mean distance from i to all other points in its cluster (cohesion)
        b(i) = mean distance from i to all points in the nearest OTHER cluster (separation)

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Interpretation:
        s(i) ≈ +1  →  Point is well inside its own cluster, far from others
        s(i) ≈  0  →  Point is on the boundary between two clusters
        s(i) ≈ -1  →  Point is probably in the wrong cluster

    The overall Silhouette Score is the mean s(i) across all points.
    Choose K that maximises the average Silhouette Score.

    Pros:   Interpretable, works for any clustering algorithm, not just K-Means
    Cons:   O(n²) computation — slow for large datasets; use sampling approximation

────────────────────────────────────────────────────────────
Method 3: Gap Statistic (Tibshirani, Walther, Hastie, 2001)
────────────────────────────────────────────────────────────
    Compare the observed WCSS to the expected WCSS under a null reference
    distribution (data with no cluster structure — uniform random data).

        Gap(K) = E[log WCSS_random(K)] - log WCSS_data(K)

    The optimal K is the smallest K such that Gap(K) ≥ Gap(K+1) - s_{K+1}
    where s_{K+1} is the standard error of Gap(K+1) estimates.

    Interpretation: A large Gap means the data has more structure than random —
    the clusters are real and meaningful.

    Pros:   Most statistically rigorous method
    Cons:   Computationally expensive (requires many reference datasets), complex

────────────────────────────────────────────────────────────
Method 4: Domain Knowledge and Practical Constraints
────────────────────────────────────────────────────────────
    In many real applications, K is determined by business context, not math:
        - Segment customers into exactly 4 tiers (premium, regular, at-risk, inactive)
        - Assign documents to exactly 10 topics (a product requirement)
        - Cluster genes into K groups where K is determined by a biologist

    Don't over-engineer K selection when domain knowledge provides a clear answer.

────────────────────────────────────────────────────────────
Summary: K Selection Strategy
────────────────────────────────────────────────────────────

    ┌────────────────────────────────────────────────────────────────────┐
    │  Best practice:                                                    │
    │    1. Check domain knowledge first — if K is obvious, use it       │
    │    2. Plot the Elbow curve to get a rough range                    │
    │    3. Compute Silhouette Scores across that range                  │
    │    4. Select K that balances interpretability and score            │
    │    5. Validate clusters make sense in context                      │
    └────────────────────────────────────────────────────────────────────┘


---

### Part 6: Complexity Analysis

Time Complexity:
    Each iteration of Lloyd's algorithm requires:
        • Assignment Step: O(n · K · d)
          — For each of n points, compute distance to each of K centroids in d dimensions
        • Update Step: O(n · d)
          — Compute the mean of each cluster

    Total per iteration: O(n · K · d)
    For T iterations total: O(T · n · K · d)

    In practice:
        - T is typically 10–300 iterations
        - Convergence is often much faster than the cap
        - The algorithm is essentially linear in n — it scales well

Space Complexity:
    O(n · d)  — store all data points
    O(K · d)  — store all centroids
    O(n)      — store all assignments

Comparison:

    ─────────────────────────────────────────────────────────────────────────────
    Algorithm           Time per Iter     Space        Scales to Large Data?
    ─────────────────────────────────────────────────────────────────────────────
    K-Means (Lloyd)     O(nKd)            O(nd + Kd)   Yes (but all data in RAM)
    MiniBatchKMeans     O(bKd) b=batch    O(bd + Kd)   Yes (streaming capable)
    K-Medoids (PAM)     O(K(n-K)²)        O(n²)        No — quadratic in n
    DBSCAN              O(n log n)         O(n)         Yes (with spatial index)
    Agglomerative       O(n² log n)        O(n²)        No — for small datasets
    ─────────────────────────────────────────────────────────────────────────────

For n > 10,000 samples and tight latency requirements, use MiniBatchKMeans.


---

### Part 7: Limitations and Failure Modes

K-Means is powerful but has well-documented failure modes. Knowing these will
save you enormous amounts of debugging time.

────────────────────────────────────────────────────────────
Failure Mode 1: Assumes Spherical Clusters
────────────────────────────────────────────────────────────
    K-Means uses Euclidean distance and finds spherical (hyperspherical in high-d)
    clusters. It cannot find elongated, curved, or irregular shapes.

    # =======================================================================================
    Diagram 5 — Cluster Shape Failures:

    Data (two half-moons):          K-Means result (wrong!):    Spectral/DBSCAN (correct):
         ● ● ● ●                        ○○○○ ●●●●                  ●●●● ○○○○
       ●       ●                      ○○○ ● ● ●●●                ●●●● ○○○○○
         ● ● ●                        ○○○ ●●●●                   ●●●  ○○○○

    K-Means cuts straight through the moons — it cannot follow the curved boundary.
    # =======================================================================================

    When to use alternatives:
        - Half-moons, rings, spirals → Spectral Clustering
        - Arbitrary density shapes → DBSCAN
        - Elongated ellipses → Gaussian Mixture Models

────────────────────────────────────────────────────────────
Failure Mode 2: Assumes Equal-Sized Clusters
────────────────────────────────────────────────────────────
    K-Means minimises total WCSS, which tends to produce clusters of similar size.
    If your true clusters are very unequal (one cluster has 1000 points, another
    has 5), K-Means will split the large cluster to compensate.

    Alternative: Gaussian Mixture Models with full covariance matrices, or DBSCAN.

────────────────────────────────────────────────────────────
Failure Mode 3: Assumes Equal Cluster Variance
────────────────────────────────────────────────────────────
    Tight, compact clusters and loose, spread-out clusters are treated equally.
    K-Means has no notion of "this cluster's points are more spread out."

    Alternative: Gaussian Mixture Models with per-cluster covariance.

────────────────────────────────────────────────────────────
Failure Mode 4: Sensitive to Outliers
────────────────────────────────────────────────────────────
    Outliers are assigned to their nearest centroid and then participate in the
    centroid update step. A single extreme outlier can drag a centroid far from
    the natural cluster centre.

    Fix: Remove outliers before clustering, or use K-Medoids (medoid = actual
    data point → more robust) or DBSCAN (which labels outliers as noise).

────────────────────────────────────────────────────────────
Failure Mode 5: Curse of Dimensionality
────────────────────────────────────────────────────────────
    In high-dimensional spaces (d >> 10), Euclidean distances become meaningless.
    The ratio of the maximum to minimum pairwise distance approaches 1 as d → ∞.
    This means all points appear equally (dis)similar — clusters lose meaning.

    Fix: Apply dimensionality reduction (PCA, UMAP, t-SNE) before K-Means.
    Reduce to d ≈ 2–50 before clustering. This is standard practice in genomics,
    NLP (cluster word/sentence embeddings after reducing dimension), and CV.

────────────────────────────────────────────────────────────
Failure Mode 6: Empty Clusters
────────────────────────────────────────────────────────────
    Occasionally, a centroid ends up far from all data points and no point is
    assigned to it. The cluster becomes empty. This causes a division-by-zero
    in the update step.

    Handling strategies:
        - Reinitialise the empty centroid to a random data point
        - Replace it with the point furthest from any centroid
        - Simply reduce K by 1 and continue

    scikit-learn handles this automatically.


---

### Part 8: Variants and Extensions

────────────────────────────────────────────────────────────
MiniBatch K-Means
────────────────────────────────────────────────────────────
    Instead of computing assignments for all n points each iteration, sample
    a random mini-batch of size b and update centroids based on only that batch.

    Update rule for centroid μₖ given new batch point xᵢ assigned to k:
        μₖ ← μₖ + (1/nₖ) · (xᵢ − μₖ)

    where nₖ is the count of points assigned to cluster k so far.

    Pros:   O(b · K · d) per iteration (b << n) — much faster for large datasets
            Enables online/streaming clustering
    Cons:   Slightly worse final clustering quality vs full K-Means
            Convergence is noisier — loss oscillates more

    When to use: n > 100,000 samples where full K-Means is too slow.

────────────────────────────────────────────────────────────
Fuzzy K-Means (Fuzzy C-Means)
────────────────────────────────────────────────────────────
    Standard K-Means assigns each point to exactly one cluster (hard assignment).
    Fuzzy K-Means assigns each point to ALL clusters with a degree of membership:

        uᵢₖ ∈ [0, 1]   such that   Σₖ uᵢₖ = 1

    The objective becomes:
        J = Σᵢ Σₖ uᵢₖᵐ · ‖xᵢ − μₖ‖²    where m > 1 is the fuzziness parameter

    Centroid update:    μₖ = (Σᵢ uᵢₖᵐ xᵢ) / (Σᵢ uᵢₖᵐ)
    Membership update:  uᵢₖ = 1 / Σⱼ (‖xᵢ-μₖ‖/‖xᵢ-μⱼ‖)^(2/(m-1))

    m=1 → approaches hard K-Means; m→∞ → all clusters equally probable

    When to use: When points genuinely belong to multiple categories (e.g.,
    a document that discusses both politics AND economics, a gene that
    participates in multiple pathways).

────────────────────────────────────────────────────────────
K-Medoids (PAM — Partitioning Around Medoids)
────────────────────────────────────────────────────────────
    Instead of using the mean as the centroid (which may not be a real data point),
    K-Medoids selects an ACTUAL data point as the "medoid" — the most central member.

    Objective: Σᵢ d(xᵢ, mₖ)    where mₖ is the medoid (must be a real point)

    Pros:   Works with any distance metric (even non-Euclidean ones like DTW,
            edit distance, etc.)
            Medoids are interpretable — "this particular customer represents the group"
            More robust to outliers
    Cons:   O(K(n-K)²) per iteration — much slower than K-Means

    When to use: Categorical data, time-series clustering, graph-structured data,
    or any setting where the mean is not well-defined.

────────────────────────────────────────────────────────────
Bisecting K-Means
────────────────────────────────────────────────────────────
    A hierarchical variant that builds the K-partition top-down:
        1. Start with all data in one cluster
        2. Select the cluster with the highest WCSS
        3. Bisect it (apply K-Means with K=2)
        4. Repeat until K clusters total

    Pros:   Produces a dendrogram-like hierarchy; avoids the worst local optima
    Cons:   Greedy — bad early splits cannot be fixed later

────────────────────────────────────────────────────────────
Kernel K-Means
────────────────────────────────────────────────────────────
    Apply the kernel trick: implicitly map data to a high-dimensional feature space
    φ(x), then run K-Means in that space.

    Objective: Σᵢ ‖φ(xᵢ) − μₖ‖²    (computed via kernel matrix K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ))

    This allows K-Means to find non-linear cluster boundaries in the original space.
    Equivalent to Spectral Clustering under certain conditions.


---

### Part 9: K-Means as a Foundation for Other Algorithms

K-Means is not just a standalone algorithm — it is a building block for many
other important machine learning techniques.

1. Vector Quantisation (VQ):
    K-Means centroid positions act as a codebook. Each data point is represented
    by the index of its nearest centroid instead of its full d-dimensional vector.
    This achieves lossy compression proportional to log₂(K) bits per point.
    Used in: JPEG image compression, audio codec design, neural network weight quantisation.

2. Feature Engineering (Cluster Features):
    Run K-Means and use cluster assignment or distance-to-centroid as new features
    for a downstream supervised model. This adds non-linear structure awareness
    to linear models.

3. Data Summarisation for Large Datasets:
    Replace n data points with K centroids (each weighted by cluster size).
    Useful as a preprocessing step before algorithms that don't scale to large n.

4. Initialisation for Gaussian Mixture Models (EM Algorithm):
    EM for GMMs is sensitive to initialisation. Running K-Means first and using
    the K centroids as initial Gaussian means provides an excellent warm start.

5. Image Segmentation:
    Treat each pixel's RGB values as a 3D point. Run K-Means with K=8 or K=16
    to segment the image into regions. Each segment gets the colour of its centroid.
    This is the simplest practical image segmentation algorithm.

6. Document Clustering and Topic Discovery:
    Embed documents as TF-IDF vectors or dense word embeddings, apply PCA to
    reduce dimension, then run K-Means to discover topics.

7. Neural Network Weight Initialisation:
    Run K-Means on the activations from an earlier layer to determine initial
    weights for the next layer (used in some self-organised map variants).


---

### Part 10: Practical Best Practices

A complete practical checklist before running K-Means:

    ┌────────────────────────────────────────────────────────────────────────┐
    │  Before Running K-Means:                                               │
    │                                                                        │
    │  ✓ 1. Scale your features (StandardScaler or MinMaxScaler)             │
    │      → K-Means is distance-based; unscaled features break it           │
    │                                                                        │
    │  ✓ 2. Handle outliers (IQR filtering, IsolationForest, etc.)           │
    │      → Outliers distort centroid positions                             │
    │                                                                        │
    │  ✓ 3. Reduce dimensionality if d > 20 (PCA first)                      │
    │      → Euclidean distance becomes meaningless in high-d                │
    │                                                                        │
    │  ✓ 4. Choose K using Elbow + Silhouette (not just one method)          │
    │                                                                        │
    │  ✓ 5. Use KMeans++ initialisation (default in sklearn)                 │
    │                                                                        │
    │  ✓ 6. Set n_init ≥ 10 (run multiple times, keep best)                  │
    │                                                                        │
    │  ✓ 7. Verify clusters are interpretable in context                     │
    │      → Low WCSS doesn't mean the clusters are meaningful               │
    │                                                                        │
    │  ✓ 8. For n > 100k samples, use MiniBatchKMeans                        │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘


### Summary Table

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  K-Means: At a Glance                                                        │
    │                                                                              │
    │  Type:          Unsupervised — Partitional Clustering                        │
    │  Objective:     Minimise WCSS = Σᵢ ‖xᵢ − μ_{zᵢ}‖²                            │
    │  Algorithm:     Lloyd's (alternating assignment + update)                    │
    │  Initialisation: K-Means++ (best) or random (Forgy)                          │
    │  Convergence:   Guaranteed to local optimum; NP-hard globally                │
    │  Complexity:    O(T · n · K · d) time; O(n·d + K·d) space                    │
    │  Hyperparameters: K (critical), init, n_init, max_iter, tol                  │
    │                                                                              │
    │  Strengths:     Fast, simple, scales to large n, interpretable centroids     │
    │  Weaknesses:    Spherical clusters only, needs K upfront, outlier-sensitive  │
    │                                                                              │
    │  When to use:   Compact, roughly equal-sized, globular clusters              │
    │  Alternatives:  DBSCAN (arbitrary shape), GMM (probabilistic/elliptical),    │
    │                 Spectral (curved), K-Medoids (non-Euclidean metrics)         │
    └──────────────────────────────────────────────────────────────────────────────┘
"""


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL HTML
# ─────────────────────────────────────────────────────────────────────────────

VISUAL_HTML = ""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "K-Means from Scratch": {
        "description": "Full Lloyd's Algorithm implementation with step-by-step commentary — no sklearn",
        "runnable": True,
        "code": '''
"""
================================================================================
K-MEANS CLUSTERING FROM SCRATCH — LLOYD\'S ALGORITHM
================================================================================

We build everything from first principles using only NumPy.
No sklearn, no black boxes. Every step is explained.

Architecture:
    Repeat until convergence:
        ┌─────────────────────────────────────────────────────────────────┐
        │  STEP A — Assignment: zᵢ = argmin_k ‖xᵢ − μₖ‖²                   │
        │  STEP B — Update:     μₖ ← (1/|Cₖ|) Σᵢ:zᵢ=k xᵢ                   │
        └─────────────────────────────────────────────────────────────────┘

Objective: J = Σᵢ ‖xᵢ − μ_{zᵢ}‖²    (Within-Cluster Sum of Squares)

================================================================================
"""

import numpy as np

np.random.seed(42)


# =============================================================================
# K-MEANS IMPLEMENTATION
# =============================================================================

class KMeans:
    """
    K-Means clustering via Lloyd\'s Algorithm.

    Parameters (learnable):
        self.centroids : np.array of shape (K, d)
                         One centroid per cluster, in d-dimensional space.
        self.labels_   : np.array of shape (n,)
                         Cluster assignment for each data point.

    Hyperparameters (set by user):
        K             : number of clusters
        max_iter      : maximum iterations before stopping
        tol           : convergence threshold (centroid movement)
        init          : initialisation strategy — "random" or "kmeans++"
    """

    def __init__(self, K=3, max_iter=300, tol=1e-4, init="kmeans++"):
        self.K        = K
        self.max_iter = max_iter
        self.tol      = tol
        self.init     = init

        # These are set during fit()
        self.centroids     = None
        self.labels_       = None
        self.inertia_      = None
        self.n_iter_       = 0
        self.history       = []   # track (centroids, labels, wcss) per iteration

    # ─── Euclidean distance ───────────────────────────────────────────────────
    def _euclidean_distances(self, X, centroids):
        """
        Compute squared Euclidean distance from every point to every centroid.

        Args:
            X         : (n, d) — data matrix
            centroids : (K, d) — current centroid positions

        Returns:
            dist²     : (n, K) — dist²[i, k] = ‖xᵢ − μₖ‖²

        Shape analysis:
            X[:, np.newaxis, :]      — (n, 1, d)   broadcast over K
            centroids[np.newaxis, :] — (1, K, d)   broadcast over n
            difference               — (n, K, d)
            .sum(axis=2)             — (n, K)       squared distance per (i, k)
        """
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, K, d)
        return (diff ** 2).sum(axis=2)                              # (n, K)

    # ─── Initialisation ───────────────────────────────────────────────────────
    def _init_centroids(self, X):
        """
        Initialise K centroids using the chosen strategy.
        """
        n, d = X.shape

        if self.init == "random":
            # Forgy: pick K data points uniformly at random
            indices = np.random.choice(n, self.K, replace=False)
            return X[indices].copy()

        elif self.init == "kmeans++":
            # K-Means++ probabilistic spread initialisation
            centroids = []

            # Step 1: First centroid chosen uniformly at random
            idx = np.random.randint(0, n)
            centroids.append(X[idx].copy())

            # Steps 2..K: Each new centroid sampled ∝ D²(x)
            for _ in range(self.K - 1):
                c_array = np.array(centroids)
                dists   = self._euclidean_distances(X, c_array)  # (n, k_chosen)
                D2      = dists.min(axis=1)                       # (n,) min dist to any chosen centroid

                # Sample next centroid with probability proportional to D²
                probs = D2 / D2.sum()
                idx   = np.random.choice(n, p=probs)
                centroids.append(X[idx].copy())

            return np.array(centroids)   # (K, d)

        else:
            raise ValueError(f"Unknown init strategy: {self.init!r}. Use \'random\' or \'kmeans++\'.")

    # ─── Assignment Step ──────────────────────────────────────────────────────
    def _assign(self, X):
        """
        Assign each point to the nearest centroid.

        Returns:
            labels : (n,) integer array — label[i] = index of nearest centroid
        """
        dist2  = self._euclidean_distances(X, self.centroids)  # (n, K)
        return dist2.argmin(axis=1)                             # (n,)

    # ─── Update Step ─────────────────────────────────────────────────────────
    def _update(self, X, labels):
        """
        Move each centroid to the mean of its assigned points.

        If a cluster is empty (no assigned points), reinitialise it to a random
        data point to avoid NaN centroids.

        Returns:
            new_centroids : (K, d)
        """
        n, d = X.shape
        new_centroids = np.zeros((self.K, d))

        for k in range(self.K):
            mask = (labels == k)
            if mask.sum() == 0:
                # Empty cluster: reinitialise to a random data point
                new_centroids[k] = X[np.random.randint(n)]
            else:
                new_centroids[k] = X[mask].mean(axis=0)

        return new_centroids

    # ─── WCSS Calculation ─────────────────────────────────────────────────────
    def _wcss(self, X, labels):
        """Compute Within-Cluster Sum of Squares (the objective J)."""
        total = 0.0
        for k in range(self.K):
            mask = (labels == k)
            if mask.sum() > 0:
                total += ((X[mask] - self.centroids[k]) ** 2).sum()
        return total

    # ─── Main fit() ──────────────────────────────────────────────────────────
    def fit(self, X, verbose=True):
        """
        Train K-Means on data X.

        Args:
            X       : np.array of shape (n, d)
            verbose : print per-iteration diagnostics

        After fit():
            self.centroids  — final centroid positions (K, d)
            self.labels_    — final cluster assignments (n,)
            self.inertia_   — final WCSS (scalar)
            self.n_iter_    — number of iterations until convergence
        """
        # ─── Initialise ──────────────────────────────────────────────────────
        self.centroids = self._init_centroids(X)

        if verbose:
            print(f"  Initialisation ({self.init}): {self.K} centroids placed")
            print(f"  First centroid: {self.centroids[0].round(3)}")
            print()

        for i in range(self.max_iter):

            # STEP A — Assignment: assign each point to nearest centroid
            labels = self._assign(X)

            # STEP B — Update: move each centroid to mean of its cluster
            new_centroids = self._update(X, labels)

            # Compute WCSS for diagnostics
            wcss = self._wcss(X, labels)
            self.history.append({
                "iteration": i,
                "wcss": wcss,
                "centroid_shift": np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            })

            if verbose and (i < 5 or i % 10 == 0):
                shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
                print(f"  Iter {i:>3} | WCSS: {wcss:>10.2f} | Max centroid shift: {shift:.6f}")

            # Check convergence: did centroids move at all?
            shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            self.n_iter_   = i + 1

            if shift < self.tol:
                if verbose:
                    print(f"\\n  Converged at iteration {i+1} (shift {shift:.2e} < tol {self.tol})")
                break

        # Final labels and inertia
        self.labels_  = self._assign(X)
        self.inertia_ = self._wcss(X, self.labels_)

    def predict(self, X_new):
        """Assign new data points to the nearest trained centroid."""
        dist2 = self._euclidean_distances(X_new, self.centroids)
        return dist2.argmin(axis=1)


# =============================================================================
# GENERATE DATA: 3 WELL-SEPARATED GAUSSIAN CLUSTERS
# =============================================================================

print("=" * 65)
print("  K-MEANS CLUSTERING FROM SCRATCH — LLOYD\'S ALGORITHM")
print("=" * 65)

n_per_cluster = 100
true_K = 3

# Three cluster centres
centres = np.array([[2.0, 2.0],
                    [8.0, 3.0],
                    [5.0, 8.0]])

# Generate data
X_list = []
for c in centres:
    X_list.append(np.random.randn(n_per_cluster, 2) * 0.8 + c)
X = np.vstack(X_list)   # (300, 2)
true_labels = np.repeat([0, 1, 2], n_per_cluster)

n, d = X.shape
print(f"\\n  Dataset: {n} points in ℝ^{d} | {true_K} true clusters (100 points each)")
print(f"  Cluster centres: {centres.tolist()}")


# =============================================================================
# FIT K-MEANS
# =============================================================================

print("\\n" + "=" * 65)
print("  TRAINING (K=3, init=kmeans++)")
print("=" * 65 + "\\n")

km = KMeans(K=3, max_iter=100, tol=1e-6, init="kmeans++")
km.fit(X, verbose=True)

print("\\n" + "=" * 65)
print("  RESULTS")
print("=" * 65)
print(f"\\n  Iterations to converge : {km.n_iter_}")
print(f"  Final WCSS (inertia)   : {km.inertia_:.4f}")
print(f"\\n  Learned centroid positions (K=3):")
for k, c in enumerate(km.centroids):
    print(f"    Cluster {k}: ({c[0]:.3f}, {c[1]:.3f})   "
          f"[True: ({centres[k][0]:.1f}, {centres[k][1]:.1f})]")

print(f"\\n  Cluster sizes: {[int((km.labels_==k).sum()) for k in range(3)]} "
      f"(true: [100, 100, 100])")


# =============================================================================
# STEP-BY-STEP TRACE ON A TINY DATASET
# =============================================================================

print("\\n" + "=" * 65)
print("  STEP-BY-STEP TRACE: 6 POINTS, K=2")
print("=" * 65)

# Tiny 6-point dataset where we can trace by hand
X_tiny = np.array([[1.0, 1.0],
                   [1.5, 2.0],
                   [3.0, 4.0],
                   [5.0, 7.0],
                   [3.5, 5.0],
                   [4.5, 5.0]])

print(f"""
  Data:
    x₀ = (1.0, 1.0)
    x₁ = (1.5, 2.0)
    x₂ = (3.0, 4.0)
    x₃ = (5.0, 7.0)
    x₄ = (3.5, 5.0)
    x₅ = (4.5, 5.0)
""")

np.random.seed(7)
km_tiny = KMeans(K=2, max_iter=20, tol=1e-10, init="random")
km_tiny.fit(X_tiny, verbose=False)

print("  Iteration-by-iteration WCSS:")
for step in km_tiny.history:
    print(f"    Iter {step[\'iteration\']:>2} | WCSS = {step[\'wcss\']:.4f} | "
          f"Max shift = {step[\'centroid_shift\']:.6f}")

print(f"\\n  Final centroids:")
for k, c in enumerate(km_tiny.centroids):
    pts = X_tiny[km_tiny.labels_ == k]
    print(f"    Cluster {k}: centroid = ({c[0]:.3f}, {c[1]:.3f})")
    print(f"               points   = {pts.tolist()}")


# =============================================================================
# SENSITIVITY TO INITIALISATION
# =============================================================================

print("\\n" + "=" * 65)
print("  INITIALISATION SENSITIVITY COMPARISON")
print("=" * 65)

results_by_init = []
for seed in range(10):
    np.random.seed(seed)
    km_r = KMeans(K=3, init="random", max_iter=200, tol=1e-8)
    km_r.fit(X, verbose=False)
    results_by_init.append(km_r.inertia_)

print(f"\\n  10 runs with RANDOM initialisation:")
print(f"    Best  WCSS : {min(results_by_init):.2f}")
print(f"    Worst WCSS : {max(results_by_init):.2f}")
print(f"    Mean  WCSS : {sum(results_by_init)/len(results_by_init):.2f}")
print(f"    Std   WCSS : {(sum((r - sum(results_by_init)/10)**2 for r in results_by_init)/10)**0.5:.2f}")

results_pp = []
for seed in range(10):
    np.random.seed(seed)
    km_pp = KMeans(K=3, init="kmeans++", max_iter=200, tol=1e-8)
    km_pp.fit(X, verbose=False)
    results_pp.append(km_pp.inertia_)

print(f"\\n  10 runs with K-MEANS++ initialisation:")
print(f"    Best  WCSS : {min(results_pp):.2f}")
print(f"    Worst WCSS : {max(results_pp):.2f}")
print(f"    Mean  WCSS : {sum(results_pp)/len(results_pp):.2f}")
print(f"    Std   WCSS : {(sum((r - sum(results_pp)/10)**2 for r in results_pp)/10)**0.5:.2f}")

print(f"""
  Takeaway:
    K-Means++ consistently finds better solutions and has much lower variance.
    The best random run may match K-Means++, but the worst runs are far behind.
    Always use K-Means++ (default in sklearn) for production.
""")
''',
    },


    "Choosing K — Elbow & Silhouette": {
        "description": "Determine the optimal number of clusters using the Elbow Method and Silhouette Score",
        "runnable": True,
        "code": '''
"""
================================================================================
CHOOSING K: ELBOW METHOD AND SILHOUETTE SCORE
================================================================================

The hardest question in K-Means: how many clusters are there in the data?
We systematically evaluate K = 1 through 10 using two complementary methods.

    Elbow Method:   Plot WCSS vs K — find the "elbow" (diminishing returns)
    Silhouette:     Measure how well each point fits its cluster (−1 to +1)

The right K often satisfies both: WCSS elbow AND peak Silhouette Score.
================================================================================
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# =============================================================================
# GENERATE DATA WITH KNOWN STRUCTURE (TRUE K = 4)
# =============================================================================

# Four well-separated clusters in 2D
centres_true = np.array([[-4, -4], [4, -4], [-4, 4], [4, 4]], dtype=float)
n_per = 80

X_parts = [np.random.randn(n_per, 2) * 0.9 + c for c in centres_true]
X = np.vstack(X_parts)

# Scale features (always do this before K-Means)
scaler = StandardScaler()
X = scaler.fit_transform(X)

n = len(X)
print("=" * 70)
print("  CHOOSING K: ELBOW METHOD AND SILHOUETTE SCORE")
print("=" * 70)
print(f"\\n  Dataset: {n} points in 2D with TRUE K=4 clusters (80 points each)")


# =============================================================================
# ELBOW METHOD: WCSS vs K
# =============================================================================

print("\\n" + "=" * 70)
print("  ELBOW METHOD — WCSS (Within-Cluster Sum of Squares) vs K")
print("=" * 70)

K_range   = range(1, 11)
wcss_list = []

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X)
    wcss_list.append(km.inertia_)

print(f"\\n  {'K':>4}  |  {'WCSS':>12}  |  {'ΔWCSS':>12}  |  {'% Drop':>8}")
print(f"  {'────':>4}──┼──{'────────────':>12}──┼──{'────────────':>12}──┼──{'────────':>8}")

for i, (k, wcss) in enumerate(zip(K_range, wcss_list)):
    delta = wcss_list[i-1] - wcss if i > 0 else 0
    pct   = 100 * delta / wcss_list[i-1] if i > 0 else 0
    arrow = "  ← ELBOW" if k == 4 else ""
    print(f"  {k:>4}  |  {wcss:>12.2f}  |  {delta:>12.2f}  |  {pct:>7.1f}%{arrow}")

print(f"""
  Reading the Elbow:
    K=1 → K=2: Massive drop — clearly better to have 2 clusters than 1
    K=3 → K=4: Still large drop — 4 clusters meaningfully reduces WCSS
    K=4 → K=5: Much smaller drop — diminishing returns begin here
    → The ELBOW is at K=4, matching the true structure.
""")


# =============================================================================
# SILHOUETTE SCORE: HOW WELL DO POINTS FIT THEIR CLUSTERS?
# =============================================================================

print("=" * 70)
print("  SILHOUETTE SCORE — Cohesion vs Separation (higher = better)")
print("=" * 70)
print(f"""
  For each point i:
    a(i) = mean distance to other points IN the same cluster    (cohesion)
    b(i) = mean distance to points in the NEAREST other cluster  (separation)
    s(i) = (b(i) − a(i)) / max(a(i), b(i))

  Interpretation:
    s ≈ +1 → Point is deep inside its cluster, far from all others
    s ≈  0 → Point sits on the boundary between two clusters
    s ≈ -1 → Point is closer to a different cluster (likely misassigned)
""")

print(f"  {'K':>4}  |  {'Silhouette':>12}  |  {'Interpretation'}")
print(f"  {'────':>4}──┼──{'────────────':>12}──┼──{'─────────────────────────────'}")

sil_scores = []
for k in range(2, 11):   # Silhouette undefined for K=1
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels)
    sil_scores.append(score)
    arrow  = "  ← BEST" if k == 4 else ""
    interp = "Excellent" if score > 0.6 else ("Good" if score > 0.4 else ("Fair" if score > 0.2 else "Poor"))
    print(f"  {k:>4}  |  {score:>12.4f}  |  {interp}{arrow}")

best_k_sil = sil_scores.index(max(sil_scores)) + 2
print(f"\\n  Best K by Silhouette Score: K={best_k_sil} (score={max(sil_scores):.4f})")


# =============================================================================
# PER-POINT SILHOUETTE ANALYSIS AT OPTIMAL K
# =============================================================================

print("\\n" + "=" * 70)
print("  PER-CLUSTER SILHOUETTE BREAKDOWN AT K=4")
print("=" * 70)

km_best = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
labels_best = km_best.fit_predict(X)
sample_sils = silhouette_samples(X, labels_best)

print(f"\\n  Overall silhouette: {sample_sils.mean():.4f}\\n")
print(f"  {'Cluster':>9}  |  {'Size':>6}  |  {'Mean s(i)':>10}  |  {'Min s(i)':>10}  |  {'# Negative':>10}")
print(f"  {'─────────':>9}──┼──{'──────':>6}──┼──{'──────────':>10}──┼──{'──────────':>10}──┼──{'──────────':>10}")

for k in range(4):
    mask    = labels_best == k
    sils_k  = sample_sils[mask]
    n_neg   = (sils_k < 0).sum()
    print(f"  {k:>9}  |  {mask.sum():>6}  |  {sils_k.mean():>10.4f}  |  {sils_k.min():>10.4f}  |  {n_neg:>10}")

print(f"""
  Interpretation:
    Negative silhouette scores indicate misassigned points — likely sitting
    on the boundary between two clusters. A high fraction of negative scores
    in any cluster suggests K may be too high (over-segmentation).
""")


# =============================================================================
# COMBINING BOTH METHODS — DECISION
# =============================================================================

print("=" * 70)
print("  DECISION FRAMEWORK: COMBINING ELBOW + SILHOUETTE")
print("=" * 70)
print(f"""
  Elbow suggests:     K = 4  (largest drop in WCSS, then diminishing returns)
  Silhouette best:    K = 4  (peak mean silhouette score)
  True K:             K = 4  ✓

  When they agree → high confidence. Both metrics point to K=4.

  When they disagree:
    → Elbow says K=3, Silhouette says K=5:
       Inspect the data. Run with both and pick the more interpretable result.
       Silhouette is generally more trustworthy as it measures cluster quality directly.

  Rule of thumb:
    1. Use the Elbow plot to narrow the range (e.g., K is between 3 and 6)
    2. Use Silhouette to pick the best K within that range
    3. Always validate that the clusters make sense in context
""")
''',
    },


    "K-Means Failure Modes & When to Use Alternatives": {
        "description": "Demonstrate exactly when and why K-Means fails, with alternative algorithm recommendations",
        "runnable": True,
        "code": '''
"""
================================================================================
K-MEANS FAILURE MODES — WHEN K-MEANS BREAKS AND WHAT TO USE INSTEAD
================================================================================

K-Means has well-documented failure modes. Recognising them is critical to
choosing the right clustering algorithm for your problem.

Failure modes demonstrated:
    1. Non-spherical clusters (half-moons, concentric rings)
    2. Highly unequal cluster sizes
    3. Clusters of very different densities
    4. Sensitivity to outliers
    5. High-dimensional data (Euclidean distance breaks down)

================================================================================
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

np.random.seed(42)

print("=" * 70)
print("  K-MEANS FAILURE MODES: WHEN TO USE ALTERNATIVES")
print("=" * 70)


# =============================================================================
# FAILURE MODE 1: NON-SPHERICAL / NON-CONVEX CLUSTERS
# =============================================================================

print("\\n" + "=" * 70)
print("  FAILURE MODE 1: Non-Spherical Clusters (Two Moons)")
print("=" * 70)

X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

# K-Means (expected to fail)
km = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_km = km.fit_predict(X_moons)
ari_km = adjusted_rand_score(y_moons, labels_km)

# DBSCAN (expected to succeed)
db = DBSCAN(eps=0.3, min_samples=5)
labels_db = db.fit_predict(X_moons)
ari_db = adjusted_rand_score(y_moons, labels_db)

print(f"""
  Two-Moons Dataset (300 points, 2 crescent-shaped clusters):

  Algorithm       | ARI Score  | Interpretation
  ──────────────────────────────────────────────────────
  K-Means (K=2)   | {ari_km:>9.4f} | {"POOR — cuts straight through the moons" if ari_km < 0.8 else "Surprisingly good"}
  DBSCAN          | {ari_db:>9.4f} | {"EXCELLENT — follows density boundaries correctly" if ari_db > 0.8 else "Decent"}

  Adjusted Rand Index (ARI):
    ARI = 1.0  → Perfect match with true labels
    ARI = 0.0  → No better than random assignment
    ARI < 0.0  → Worse than random

  Why K-Means fails:
    K-Means minimises WCSS using Euclidean distance.
    The optimal Euclidean partition of two half-moons is a vertical split
    through the middle — completely ignoring the curved structure.

  Why DBSCAN succeeds:
    DBSCAN groups points that are density-connected — it follows the shape of
    the data rather than assuming spherical clusters.
    It naturally handles arbitrary shapes and identifies noise points.
""")


# =============================================================================
# FAILURE MODE 2: HIGHLY UNEQUAL CLUSTER SIZES
# =============================================================================

print("=" * 70)
print("  FAILURE MODE 2: Highly Unequal Cluster Sizes")
print("=" * 70)

# One large cluster (900 pts), one small cluster (100 pts)
X_large = np.random.randn(900, 2) * 1.0 + np.array([0, 0])
X_small = np.random.randn(100, 2) * 0.3 + np.array([6, 0])
X_unequal = np.vstack([X_large, X_small])
y_unequal  = np.array([0]*900 + [1]*100)

X_unequal = StandardScaler().fit_transform(X_unequal)

km_unequal = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_unequal = km_unequal.fit_predict(X_unequal)
ari_unequal = adjusted_rand_score(y_unequal, labels_unequal)

# Count cluster sizes found by K-Means
found_sizes = [(labels_unequal == k).sum() for k in range(2)]

print(f"""
  Dataset: 1 large cluster (900 pts) + 1 tiny cluster (100 pts)
  True sizes:  [900, 100]
  K-Means found sizes: {sorted(found_sizes, reverse=True)}
  ARI Score: {ari_unequal:.4f}

  {"K-Means correctly separated the clusters." if ari_unequal > 0.9 else
   "K-Means SPLIT the large cluster instead of finding the small one."}

  Why this happens:
    WCSS minimisation tends to produce roughly equal-sized clusters.
    When one true cluster is very large, K-Means often splits it in two
    rather than grouping all 100 small-cluster points together.

  Better alternatives:
    → Gaussian Mixture Models with full covariance — handles varying sizes
    → DBSCAN with min_samples tuned — the small cluster can be detected
      as a dense region (requires appropriate eps and min_samples)
""")


# =============================================================================
# FAILURE MODE 3: UNEQUAL DENSITIES
# =============================================================================

print("=" * 70)
print("  FAILURE MODE 3: Very Different Cluster Densities")
print("=" * 70)

# Tight cluster (std=0.2) and loose cluster (std=2.0)
X_tight = np.random.randn(200, 2) * 0.2 + np.array([0, 0])
X_loose = np.random.randn(200, 2) * 2.0 + np.array([5, 0])
X_density = np.vstack([X_tight, X_loose])
y_density  = np.array([0]*200 + [1]*200)

X_density = StandardScaler().fit_transform(X_density)

km_density = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_density = km_density.fit_predict(X_density)
ari_density = adjusted_rand_score(y_density, labels_density)

gm_density = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
labels_gm_density = gm_density.fit_predict(X_density)
ari_gm_density = adjusted_rand_score(y_density, labels_gm_density)

print(f"""
  Dataset: 1 tight cluster (σ=0.2) + 1 loose cluster (σ=2.0)

  Algorithm            | ARI Score              | Interpretation
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  K-Means (K=2)        | {ari_density:>9.4f}    | {"Struggles with density mismatch" if ari_density < 0.9 else "Handles OK"}
  Gaussian MM (full)   | {ari_gm_density:>9.4f} | {"Correctly models each cluster\'s spread" if ari_gm_density > 0.9 else "Imperfect"}

  Why K-Means struggles:
    K-Means has no notion of cluster spread/variance. It optimises WCSS
    with the implicit assumption that all clusters have equal variance.
    When one cluster is 10× looser, some of its distant points get "stolen"
    by the neighbouring tight cluster to reduce WCSS.

  Why GMM succeeds:
    Gaussian Mixture Models explicitly model each cluster with its own
    covariance matrix Σₖ. The EM algorithm finds the right spread for
    each cluster independently.
""")


# =============================================================================
# FAILURE MODE 4: SENSITIVITY TO OUTLIERS
# =============================================================================

print("=" * 70)
print("  FAILURE MODE 4: Sensitivity to Outliers")
print("=" * 70)

# Clean data: 2 well-separated clusters
X_clean = np.vstack([
    np.random.randn(100, 2) * 0.5 + np.array([0, 0]),
    np.random.randn(100, 2) * 0.5 + np.array([5, 0])
])

# Add 5 extreme outliers
outliers = np.array([[15, 0], [-10, 0], [2.5, 15], [2.5, -10], [20, 20]])
X_outliers = np.vstack([X_clean, outliers])

km_clean    = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_clean)
km_outliers = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_outliers)

c_clean    = np.sort(km_clean.cluster_centers_[:, 0])
c_outliers = np.sort(km_outliers.cluster_centers_[:, 0])

print(f"""
  Experiment: 2 clusters at x=0 and x=5 (200 clean points)
  Then add 5 extreme outliers at positions (15,0) (−10,0) (2.5,15) etc.

  Without outliers:
    Centroid 1 x-coordinate: {c_clean[0]:.3f}  (true: 0.000)
    Centroid 2 x-coordinate: {c_clean[1]:.3f}  (true: 5.000)

  With 5 outliers:
    Centroid 1 x-coordinate: {c_outliers[0]:.3f}  (shifted by {abs(c_outliers[0]-c_clean[0]):.3f})
    Centroid 2 x-coordinate: {c_outliers[1]:.3f}  (shifted by {abs(c_outliers[1]-c_clean[1]):.3f})

  Outliers DRAG centroids away from the true cluster centres.
  With only 5 outliers out of 205 points (~2.4%), centroids can shift noticeably.

  Fixes:
    1. Remove outliers before clustering (IQR, IsolationForest, LOF)
    2. Use K-Medoids — medoids are actual data points, so one extreme point
       cannot drag the representative arbitrarily far
    3. Use DBSCAN — outliers are labelled as noise (label = -1), never used
       to define clusters
""")


# =============================================================================
# SUMMARY: WHEN TO USE WHAT
# =============================================================================

print("=" * 70)
print("  DECISION GUIDE: K-MEANS vs ALTERNATIVES")
print("=" * 70)
print(f"""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Situation                             │  Recommended Algorithm         │
  ├────────────────────────────────────────┼────────────────────────────────┤
  │  Spherical, equal-sized clusters       │  K-Means  ✓ (use this!)        │
  │  Large datasets (n > 100k)             │  MiniBatchKMeans               │
  │  Non-spherical shapes (moons, rings)   │  DBSCAN, Spectral Clustering   │
  │  Unknown number of clusters            │  DBSCAN (finds K automatically)│
  │  Soft/probabilistic membership         │  Gaussian Mixture Models       │
  │  Unequal cluster sizes/variances       │  GMM with full covariance      │
  │  Non-Euclidean distance metric         │  K-Medoids (PAM)               │
  │  Data with known outliers              │  DBSCAN or K-Medoids           │
  │  High-dimensional data (d > 50)        │  PCA/UMAP first, then K-Means  │
  │  Hierarchical structure needed         │  Agglomerative Clustering      │
  │  Online / streaming data               │  MiniBatchKMeans               │
  └────────────────────────────────────────┴────────────────────────────────┘

  K-Means is the right default for compact, roughly spherical clusters.
  The moment your data deviates from that assumption, consider the alternatives.
""")
''',
    },


    "K-Means Applications: Image Compression & Segmentation": {
        "description": "Apply K-Means to real problems: image colour quantisation and pixel segmentation",
        "runnable": True,
        "code": '''
"""
================================================================================
K-MEANS APPLICATIONS: IMAGE COLOUR COMPRESSION & SEGMENTATION
================================================================================

K-Means is the foundation of several important practical algorithms.
Here we implement two image processing applications from scratch:

    1. Colour Quantisation (lossy image compression)
       → Treat each pixel\'s RGB as a 3D point
       → K-Means centroids = the K representative colours
       → Replace each pixel with its nearest centroid colour
       → Compression ratio = log₂(256³) / log₂(K) bits per pixel

    2. Image Segmentation
       → Same technique, different interpretation
       → Each centroid = one "segment" of the image
       → Used in medical imaging, satellite imagery, object detection

================================================================================
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("=" * 70)
print("  K-MEANS APPLICATIONS: IMAGE COMPRESSION & SEGMENTATION")
print("=" * 70)


# =============================================================================
# SYNTHETIC "IMAGE" — 50x50 RGB with 4 dominant colour regions
# =============================================================================

# Create a synthetic 50x50 "image" with 4 colour blocks + noise
H, W = 50, 50
img  = np.zeros((H, W, 3), dtype=np.float64)

# Top-left quadrant: mostly red
img[:25, :25] = [220, 50, 50]
# Top-right quadrant: mostly green
img[:25, 25:] = [50, 200, 60]
# Bottom-left quadrant: mostly blue
img[25:, :25] = [50, 60, 210]
# Bottom-right quadrant: mostly yellow
img[25:, 25:] = [220, 210, 50]

# Add noise to simulate a real photograph
noise = np.random.randn(H, W, 3) * 20
img   = np.clip(img + noise, 0, 255)

# Reshape: (H*W, 3) — each row is one pixel\'s RGB
pixels = img.reshape(-1, 3)   # (2500, 3)
n_pixels = len(pixels)

print(f"\\n  Synthetic Image: {H}×{W} = {n_pixels} pixels, 3 channels (RGB)")
print(f"  Raw storage (24-bit colour): {n_pixels * 3} bytes = {n_pixels * 3 / 1024:.2f} KB")


# =============================================================================
# COLOUR QUANTISATION: Compress by reducing to K colours
# =============================================================================

print("\\n" + "=" * 70)
print("  COLOUR QUANTISATION (Lossy Compression)")
print("=" * 70)
print(f"""
  Idea:
    Store only K centroid colours + one index per pixel (log₂K bits).
    Full 24-bit colour needs 3 bytes/pixel.
    With K colours, you need log₂(K) bits/pixel (plus the K×3 codebook).

    Compression ratio ≈ 24 / log₂(K)     (ignoring codebook overhead)
""")

print(f"  {'K':>5} | {'WCSS':>12} | {'Bits/pixel':>11} | {'Compression':>12} | "
      f"{'Codebook':>10} | Time")
print(f"  {'─────':>5}─┼─{'────────────':>12}─┼─{'───────────':>11}─┼─"
      f"{'────────────':>12}─┼─{'──────────':>10}─┼─{'────'}")

for K in [2, 4, 8, 16, 32, 64, 128]:
    km_img = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42)
    km_img.fit(pixels)

    labels     = km_img.labels_
    centroids  = km_img.cluster_centers_

    # Reconstruct image with quantised colours
    img_compressed = centroids[labels].reshape(H, W, 3)

    # Compute reconstruction error (PSNR proxy)
    wcss       = km_img.inertia_
    mse        = ((pixels - centroids[labels]) ** 2).mean()
    bits_per_px = np.log2(K) if K > 1 else 1.0
    compression = 24.0 / bits_per_px
    codebook_bytes = K * 3   # K centroids × 3 bytes each

    print(f"  {K:>5} | {wcss:>12.1f} | {bits_per_px:>11.1f} | "
          f"       {compression:>6.1f}:1 | {codebook_bytes:>8}  B | "
          f"MiniBatch")

print(f"""
  Reading the table:
    K=2   → 1 bit/pixel  → 24:1 compression  → Very lossy (only 2 colours)
    K=8   → 3 bits/pixel → 8:1  compression  → Visible quality loss
    K=32  → 5 bits/pixel → 4.8:1 compression → Noticeable to trained eye
    K=128 → 7 bits/pixel → 3.4:1 compression → Near-original quality

  This is essentially how GIF compression works (256-colour palette).
  JPEG uses DCT + Huffman coding instead of K-Means, but K-Means colour
  quantisation is the conceptual predecessor.
""")


# =============================================================================
# RECONSTRUCTION QUALITY: PIXEL-LEVEL ANALYSIS
# =============================================================================

print("=" * 70)
print("  RECONSTRUCTION QUALITY ANALYSIS AT K=4 (matching our 4-colour image)")
print("=" * 70)

km_4 = KMeans(n_clusters=4, n_init=10, random_state=42)
km_4.fit(pixels)

labels_4    = km_4.labels_
centroids_4 = km_4.cluster_centers_
pixels_reconstructed = centroids_4[labels_4]

per_pixel_error = np.sqrt(((pixels - pixels_reconstructed) ** 2).sum(axis=1))
mse_4           = ((pixels - pixels_reconstructed) ** 2).mean()
psnr_proxy      = 20 * np.log10(255 / np.sqrt(mse_4))

print(f"""
  K=4 Reconstruction (true image has 4 dominant colours → perfect K):

  Learned Centroids (cluster representative colours):
""")
colour_names = ["Red region", "Green region", "Blue region", "Yellow region"]
for k, (c, name) in enumerate(zip(centroids_4, colour_names)):
    size = (labels_4 == k).sum()
    print(f"    Cluster {k}  R={c[0]:>5.1f}  G={c[1]:>5.1f}  B={c[2]:>5.1f}  "
          f"→ {size:>5} pixels  ({100*size/n_pixels:.1f}%)  [{name}]")

print(f"""
  Error Metrics:
    Mean Squared Error    : {mse_4:.4f}
    Root MSE              : {np.sqrt(mse_4):.4f}  (RGB units out of 255)
    PSNR proxy            : {psnr_proxy:.2f} dB  (>30 dB = visually good quality)
    Mean per-pixel error  : {per_pixel_error.mean():.4f} RGB units
    Max per-pixel error   : {per_pixel_error.max():.4f} RGB units

  With K=4 (matching the true image structure), reconstruction is near-perfect.
  The only error comes from the noise we added at generation time.
""")


# =============================================================================
# VECTOR QUANTISATION AS DATA COMPRESSION
# =============================================================================

print("=" * 70)
print("  VECTOR QUANTISATION — THE FORMAL FRAMEWORK")
print("=" * 70)
print(f"""
  Vector Quantisation (VQ) is the generalisation of K-Means to data compression:

    Step 1: Training
        Given a dataset of vectors, run K-Means to find K centroids.
        These K centroids form the "codebook" C = {{c₁, c₂, ..., cₖ}}.

    Step 2: Encoding (Compression)
        For each data vector xᵢ, find the nearest centroid:
            code(xᵢ) = argmin_k ‖xᵢ − cₖ‖²  → store only this index (log₂K bits)

    Step 3: Decoding (Decompression)
        To reconstruct, look up the centroid at the stored index:
            # x̂ᵢ = c_{code(xᵢ)}
            x̂ᵢ = c_{{code(xᵢ)}}
  Storage comparison for our {H}×{W} image:
    Original (raw)       : {H*W*3} bytes  ({H*W*3/1024:.1f} KB)
    K=16 VQ (indices)    : {H*W*4//8} bytes  (4 bits × {H*W} pixels)  +  {16*3} bytes codebook
    K=256 VQ (indices)   : {H*W} bytes  (1 byte × {H*W} pixels)       +  {256*3} bytes codebook

  The fundamental tradeoff:
    More centroids K → better quality → more bits per pixel (log₂K)
    Fewer centroids K → worse quality → fewer bits per pixel
    Optimal K balances distortion (WCSS) against rate (bits) — the Rate-Distortion curve.

  Applications of K-Means / VQ in production:
    • JPEG 2000 embedded codebooks
    • Speech codecs (CELP, AMR)
    • Product quantisation in FAISS (Facebook\'s billion-scale nearest neighbour search)
    • Neural network weight quantisation (K-Means on 32-bit floats → 4-bit indices)
    • Embedding compression in recommendation systems
""")
''',
    },


    "MiniBatch K-Means — Scaling to Large Datasets": {
        "description": "Compare standard K-Means vs MiniBatch K-Means on quality, speed, and convergence behaviour",
        "runnable": True,
        "code": '''
"""
================================================================================
MINIBATCH K-MEANS — SCALING K-MEANS TO LARGE DATASETS
================================================================================

Standard K-Means requires all n data points in memory every iteration.
MiniBatch K-Means processes random mini-batches, dramatically reducing:
    - Memory footprint (only batch of size b needs to fit in RAM)
    - Time per iteration (O(bKd) instead of O(nKd))
    - Enables online/streaming clustering

Tradeoff: Slightly worse final cluster quality than full K-Means.

Update rule for each mini-batch centroid update:
    nₖ  ← nₖ + 1            (increment count of points assigned to k)
    μₖ  ← μₖ + (1/nₖ)(x − μₖ)    (online mean update)
================================================================================
"""

import numpy as np
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

np.random.seed(42)


# =============================================================================
# GENERATE LARGE DATASET
# =============================================================================

K_true    = 5
n_samples = 50_000
n_feats   = 10

# True cluster centres in 10D space
centres = np.random.randn(K_true, n_feats) * 5

X_parts = [np.random.randn(n_samples // K_true, n_feats) * 0.8 + centres[k]
           for k in range(K_true)]
X_big = np.vstack(X_parts)
y_big = np.repeat(range(K_true), n_samples // K_true)

X_big = StandardScaler().fit_transform(X_big)

print("=" * 70)
print("  MINIBATCH K-MEANS: SCALING TO LARGE DATASETS")
print("=" * 70)
print(f"\\n  Dataset: {len(X_big):,} samples × {n_feats} features | True K={K_true}")


# =============================================================================
# TIMING COMPARISON: FULL K-MEANS vs MINIBATCH K-MEANS
# =============================================================================

print("\\n" + "=" * 70)
print("  TIMING COMPARISON: Full K-Means vs MiniBatch K-Means")
print("=" * 70)

# Full K-Means
t0 = time.perf_counter()
km_full = KMeans(n_clusters=5, n_init=5, max_iter=300, random_state=42)
km_full.fit(X_big)
t_full = time.perf_counter() - t0
ari_full = adjusted_rand_score(y_big, km_full.labels_)

print(f"""
  Full K-Means:
    Time          : {t_full:.3f} s
    Inertia       : {km_full.inertia_:,.2f}
    Iterations    : {km_full.n_iter_}
    ARI Score     : {ari_full:.4f}
    n_init used   : 5
""")

# MiniBatch K-Means at different batch sizes
batch_sizes = [100, 500, 1024, 5000]

print(f"  {'Batch Size':>12} | {'Time (s)':>9} | {'Inertia':>12} | {'ARI':>7} | {'Speedup':>8}")
print(f"  {'────────────':>12}─┼─{'─────────':>9}─┼─{'────────────':>12}─┼─{'───────':>7}─┼─{'────────':>8}")

for b in batch_sizes:
    t0 = time.perf_counter()
    mb_km = MiniBatchKMeans(n_clusters=5, batch_size=b, n_init=5,
                            max_iter=300, random_state=42)
    mb_km.fit(X_big)
    t_mb = time.perf_counter() - t0
    ari_mb = adjusted_rand_score(y_big, mb_km.labels_)
    speedup = t_full / t_mb
    print(f"  {b:>12,} | {t_mb:>9.3f} | {mb_km.inertia_:>12,.2f} | "
          f"{ari_mb:>7.4f} | {speedup:>7.1f}×")

print(f"""
  Observations:
    → Smaller batches: faster per iteration, noisier convergence, worse quality
    → Larger batches: slower per iteration, smoother convergence, quality ≈ full K-Means
    → batch_size=1024 is a common practical default (sklearn\'s default)

  When to use MiniBatch K-Means:
    • n > 100,000 samples and speed matters
    • Streaming / online setting (data arrives continuously)
    • Memory constrained environments (batch fits in RAM, full dataset doesn\'t)
    • Acceptable small quality loss in exchange for major speed gain
""")


# =============================================================================
# THE ONLINE UPDATE RULE — HOW MINIBATCH WORKS
# =============================================================================

print("=" * 70)
print("  THE MINIBATCH UPDATE RULE (Online Mean Update)")
print("=" * 70)
print(f"""
  Standard K-Means centroid update (full batch):
        μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ          ← recompute from scratch each iteration

  MiniBatch K-Means centroid update (online):
    For each data point x in the current mini-batch assigned to cluster k:
        nₖ  ← nₖ + 1                        ← increment running count
        μₖ  ← μₖ + (1/nₖ)(x − μₖ)          ← online mean update

  Why the online update works:
    The running mean formula (1/nₖ)(x − μₖ) is the incremental mean update.
    It is mathematically equivalent to recomputing the mean from scratch if
    all points were incorporated in order — but much cheaper per step.

  Worked example (K=1, updating a centroid):
    μ starts at 5.0, n=0
    Observe x₁=7: n=1, μ ← 5.0 + (1/1)(7 - 5.0) = 7.0
    Observe x₂=3: n=2, μ ← 7.0 + (1/2)(3 - 7.0) = 5.0
    Observe x₃=9: n=3, μ ← 5.0 + (1/3)(9 - 5.0) = 6.33
    True mean of [7,3,9] = 19/3 = 6.33 ✓

  This is the Welford online algorithm for numerically stable mean computation.
  It avoids storing all data points and prevents floating-point accumulation errors.

  Learning rate decay:
    In MiniBatchKMeans, the effective learning rate (1/nₖ) decreases as more
    points are assigned to cluster k. Early in training, centroids move quickly.
    Later, they become more stable — an automatic annealing schedule.
""")


# =============================================================================
# CONVERGENCE TRACKING
# =============================================================================

print("=" * 70)
print("  CONVERGENCE COMPARISON: Inertia vs Iterations")
print("=" * 70)

# Smaller dataset for clearer tracking
np.random.seed(99)
X_small = np.vstack([np.random.randn(200, 2) * 0.5 + c
                     for c in [[0,0],[5,0],[0,5],[5,5],[2.5,2.5]]])
X_small = StandardScaler().fit_transform(X_small)

print(f"\\n  Tracking convergence on a 1000-point 2D dataset (K=5):")
print(f"\\n  {'Iteration':>10} | {'Full KM Inertia':>16} | {'MiniBatch Inertia':>18}")
print(f"  {'──────────':>10}─┼─{'────────────────':>16}─┼─{'──────────────────':>18}")

# Manually track convergence per iteration
prev_inertia_full = None
for max_it in [1, 2, 3, 5, 10, 20, 50, 100]:
    km_iter = KMeans(n_clusters=5, n_init=1, max_iter=max_it,
                     init="k-means++", random_state=42)
    km_iter.fit(X_small)

    mb_iter = MiniBatchKMeans(n_clusters=5, n_init=1, max_iter=max_it,
                              batch_size=100, random_state=42)
    mb_iter.fit(X_small)

    print(f"  {max_it:>10} | {km_iter.inertia_:>16.2f} | {mb_iter.inertia_:>18.2f}")

print(f"""
  Key observation:
    Full K-Means converges monotonically — inertia always decreases.
    MiniBatch K-Means converges noisily — some iterations may increase inertia
    because the update is based only on a random sample, not all data.

    Despite the noise, MiniBatch reaches a good solution quickly and the
    quality gap vs full K-Means narrows with more iterations.
""")
''',
    },


    "▶ Run: K-Means Clustering": {
        "description": (
            "Runs 01_k_means.py from the Implementation folder. "
            "Full K-Means clustering implementation with visualisation of "
            "centroids, cluster assignments, elbow method, and silhouette analysis."
        ),
        "runnable": True,
        "code": '''
import os, sys
from pathlib import Path

_impl = (
    Path(os.getcwd())
    / "Unsupervised_Learning"
    / "Implementation"
    / "Clustering"
    / "01_k_means.py"
)

if not _impl.exists():
    print(f"[ERROR] File not found: {_impl}")
    print(f"  CWD is: {os.getcwd()}")
    print("  Expected structure:  <project_root>/Unsupervised_Learning/Implementation/Clustering/")
    sys.exit(1)

print(f"Running: {_impl}")
print("=" * 65)
exec(
    compile(_impl.read_text(encoding="utf-8"), str(_impl), "exec"),
    {"__name__": "__main__", "__file__": str(_impl)}
)
''',
    },

}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Standalone demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np

    print("\n" + "=" * 65)
    print("  K-MEANS CLUSTERING: THE FOUNDATION OF UNSUPERVISED LEARNING")
    print("=" * 65)
    print("""
  This script demonstrates K-Means clustering from the ground up.

  Key Concepts:
    • Objective: minimise WCSS = Σᵢ ‖xᵢ − μ_{zᵢ}‖²
    • Lloyd's Algorithm: alternating Assignment + Update steps
    • Guaranteed convergence to a local optimum (NP-hard globally)
    • K-Means++ initialisation: O(log K) approximation guarantee
    • Elbow Method + Silhouette Score for choosing K
    • Failure modes: non-spherical clusters, outliers, unequal sizes
    """)

    np.random.seed(42)

    # ─── Tiny 2D dataset: 3 obvious clusters ─────────────────────────────────
    centres = np.array([[1.0, 1.0], [5.0, 1.0], [3.0, 5.0]])
    X_parts = [np.random.randn(30, 2) * 0.4 + c for c in centres]
    X = np.vstack(X_parts)

    print("=" * 65)
    print("  DATASET: 90 points in ℝ² with 3 true clusters (30 each)")
    print("=" * 65)
    print(f"\n  True cluster centres: {centres.tolist()}")

    # ─── Manual Lloyd's Algorithm ─────────────────────────────────────────────
    K = 3
    np.random.seed(7)
    idx = np.random.choice(len(X), K, replace=False)
    centroids = X[idx].copy()

    print(f"\n  Initial centroids (random):")
    for k, c in enumerate(centroids):
        print(f"    μ{k} = ({c[0]:.3f}, {c[1]:.3f})")

    print(f"\n  {'Iter':>5} | {'WCSS':>12} | {'Max shift':>12}")
    print(f"  {'─────':>5}─┼─{'────────────':>12}─┼─{'────────────':>12}")

    for iteration in range(50):
        # Assignment step
        dists   = np.array([[np.sum((x - c)**2) for c in centroids] for x in X])
        labels  = dists.argmin(axis=1)

        # WCSS
        wcss = sum(np.sum((X[labels == k] - centroids[k])**2) for k in range(K))

        # Update step
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))

        print(f"  {iteration:>5} | {wcss:>12.4f} | {shift:>12.8f}")

        centroids = new_centroids
        if shift < 1e-8:
            print(f"\n  Converged at iteration {iteration+1}!")
            break

    print(f"\n  Final centroids:")
    for k, c in enumerate(centroids):
        true = centres[k]
        print(f"    μ{k} = ({c[0]:.3f}, {c[1]:.3f})   [True: ({true[0]:.1f}, {true[1]:.1f})]")

    print(f"\n  Cluster sizes: {[(labels==k).sum() for k in range(K)]} (true: [30, 30, 30])")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1. K-Means minimises WCSS: Σᵢ ‖xᵢ − μ_{zᵢ}‖²  (within-cluster variance)
  2. Lloyd's algorithm: alternating Assignment + Update until convergence
  3. Convergence is GUARANTEED to a local optimum (not necessarily global)
  4. K-Means++ gives O(log K) approximation — always use it over random init
  5. Choosing K: Elbow Method gives a range; Silhouette Score picks the best
  6. Feature scaling is MANDATORY — Euclidean distance is scale-sensitive
  7. Fails on non-spherical clusters, unequal sizes, and outliers
  8. MiniBatchKMeans scales to millions of samples with minor quality loss
  9. Applications: image compression, customer segmentation, document clustering
  10. Building block for GMMs (warm start), VQ codebooks, and spectral methods
    """)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ──────────────────────────────────────────────────────────────────────────
    Component / Variant             Complexity              Notes
    ──────────────────────────────────────────────────────────────────────────
    Assignment Step (1 iter)        O(n · K · d)            Dist from each point to each centroid
    Update Step (1 iter)            O(n · d)                Mean of each cluster
    Full Lloyd's Algorithm          O(T · n · K · d)        T ≈ 10–300 iterations in practice
    K-Means++ Initialisation        O(K · n · d)            Linear in n, done once
    Silhouette Score (exact)        O(n² · d)               Slow for large n — use sampling
    MiniBatch K-Means (1 iter)      O(b · K · d)            b = batch size (b << n)
    K-Medoids / PAM                 O(K · (n−K)² · d)       Quadratic — only for small n
    Prediction (inference)          O(K · d)                Single distance comparison per point
    ──────────────────────────────────────────────────────────────────────────
    Space: O(n · d + K · d) for data + centroids
    ──────────────────────────────────────────────────────────────────────────
"""


def get_content():
    """Return all content for this topic module."""
    return {
        "theory":                 THEORY,
        "theory_raw":             THEORY,
        "complexity":             COMPLEXITY,
        "operations":             OPERATIONS,
        "interactive_components": [],
    }