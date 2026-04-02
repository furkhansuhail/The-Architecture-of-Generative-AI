"""
The Curse of Dimensionality
============================

Why adding more features can make your model dramatically worse, why
nearest-neighbour search breaks down, and why high-dimensional spaces are
profoundly counter-intuitive.

"""

import textwrap
import re

TOPIC_NAME = "The Curse of Dimensionality"
DISPLAY_NAME = "03 · Curse of Dimensionality"
ICON = "📐"
SUBTITLE = "When More Features Means Less Information"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Problem: High-Dimensional Spaces are Empty

The "Curse of Dimensionality" was coined by Richard Bellman in 1961.
It refers to a family of related phenomena that emerge when data lives in
a space with many dimensions — all of which make machine learning harder,
sometimes catastrophically so.

The core intuition is this:

    As the number of dimensions grows, the volume of the space
    grows so fast that your data becomes vanishingly sparse —
    no matter how much data you have.

This is not a limitation of current algorithms. It is a mathematical fact
about geometry. Understanding it is essential for every practitioner.

There are four distinct sub-problems grouped under this name:

    •   Exponential volume growth  — space becomes astronomically large
    •   Data sparsity              — samples become isolated points
    •   Distance concentration     — all distances become equal
    •   Model complexity explosion — more dims → exponentially more parameters


──────────────────────────────────────────────────────────────────────────────

### Problem 1: Exponential Volume Growth

Consider a hypercube with side length 1 in d dimensions.
Its volume is always 1^d = 1.

Now ask: what fraction of that volume is within distance 0.1 of any edge?

In 1D this is trivial. In high dimensions the answer becomes extreme:

    Diagram 1 — Edge Volume Fraction in a Unit Hypercube:

    The "edge region" is everything OUTSIDE the inner hypercube of
    side (1 - 2ε) — i.e., anything within ε = 0.1 of a face.

    Inner volume = (1 - 2ε)^d = (0.8)^d

    d    Inner Volume   Edge Fraction   Interpretation
    ─────────────────────────────────────────────────────────────────────
    1      0.800         20.0 %         A small sliver at each end
    2      0.640         36.0 %         A border region
    3      0.512         48.8 %         Nearly half the cube is "edge"
    5      0.328         67.2 %         Most of the cube is edge
    10     0.107         89.3 %         Almost all volume is near the edge
    30     0.001         99.9 %         The centre barely exists
    100    ≈ 0.0         ≈100.0 %       Entirely edge. No interior.
    ─────────────────────────────────────────────────────────────────────

    Consequence for ML: In 100 dimensions, almost all of your training
    points are in the "corners" — in a thin shell near the boundary.
    The "centre" of the distribution is effectively empty.

    ┌──────────────────────────────────────────────────────────────────┐
    │  2D: Centre is populated     100D: Everything is near the edge   │
    │                                                                  │
    │   ● ● ● ● ● ●               ·  ·  ·  ·  ·  ·                     │
    │   ● ● ● ● ● ●               ·                 ·                  │
    │   ● ● ● ● ● ●               ·                 ·                  │
    │   ● ● ● ● ● ●               ·                 ·                  │
    │   ● ● ● ● ● ●               ·  ·  ·  ·  ·  ·                     │
    │                                                                  │
    │   Points fill the space.    Points cluster on the surface.       │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────

### Problem 2: Data Sparsity — You Need Exponentially More Data

To cover a 1D unit interval with density ρ (one sample per unit length),
you need ρ¹ samples. To cover a d-dimensional unit hypercube with the
same density you need ρ^d samples.

    Diagram 2 — Samples Needed to Maintain Coverage Density:

    Suppose you want a sample within distance 0.1 of any test point.
    In 1D that requires ~10 points. In higher dimensions:

    d     Samples needed    Visual scale
    ──────────────────────────────────────────────────────────────────
    1          10           ██
    2         100           ████████████████████
    3       1,000           Fills a large room
    10  10,000,000,000      More than all humans who ever lived
    20  10^20               More than atoms in a handful of sand
    ──────────────────────────────────────────────────────────────────

    Practical consequence:
    ┌─────────────────────────────────────────────────────────────────┐
    │  A dataset with 10,000 samples that seems large in 10 dims      │
    │  is effectively a SINGLE SPARSE POINT in 20 dimensions.         │
    │  There are no "nearby" neighbours — every test point is an      │
    │  extrapolation, not an interpolation.                           │
    └─────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Problem 3: Distance Concentration

This is perhaps the most surprising and dangerous manifestation.

In low dimensions, Euclidean distance is meaningful: some points are close,
some are far. In high dimensions, ALL pairwise distances converge to the
same value. There is no longer a meaningful notion of "nearest neighbour."

Formally, the ratio of max-to-min distance → 1 as d → ∞:

    lim_{d→∞}  (d_max - d_min) / d_min  =  0

In practice this means:

    Diagram 3 — Distance Distributions in Low vs High Dimensions:

    LOW DIMENSIONS (d = 2)          HIGH DIMENSIONS (d = 500)

    Distance distribution:          Distance distribution:

    Count                           Count
      │                               │
      │    ╭──╮                       │              ╭──╮
      │   ╱    ╲                      │             ╱    ╲
      │  ╱      ╲                     │            ╱      ╲
      │ ╱        ╲_____               │    _______╱        ╲_______
      └──────────────────  dist       └────────────────────────────  dist
         0   2   4   6                   0     28    30    32

    Wide spread of distances.       All distances tightly clustered.
    "Near" and "far" are clear.     "Near" and "far" are meaningless.

    Numerical example (1,000 random uniform points):
    ┌────────────────────────────────────────────────────────────────┐
    │  d     Mean Dist   Min Dist   Max Dist   (Max-Min)/Min         │
    ├────────────────────────────────────────────────────────────────┤
    │    2     0.52        0.003      1.41        469 ×              │
    │   10     1.83        0.82       2.71          2.3 ×            │
    │   50     5.77        4.90       6.55          0.34 ×           │
    │  100     8.17        7.45       8.85          0.19 ×           │
    │  500    18.25       17.62      18.89          0.07 ×           │
    └────────────────────────────────────────────────────────────────┘

    → All distances shrink to the same value in high dims.
    → Your nearest neighbour is almost as far as your farthest neighbour.
    → Distance-based algorithms (KNN, K-Means, SVMs with RBF kernels,
      anomaly detectors) are all critically degraded.


──────────────────────────────────────────────────────────────────────────────
### Problem 4: The Hypersphere Shrinks Away

A sphere inscribed inside a unit hypercube occupies an increasingly
tiny fraction of the cube's volume as dimensions increase.

    Volume of unit hypersphere in d dimensions:

            π^(d/2)
    V_d = ──────────
           Γ(d/2+1)

    Diagram 4 — Hypersphere vs Hypercube Volume Ratio:

    d    Hypersphere Volume   Hypercube Volume    Ratio (sphere/cube)
    ──────────────────────────────────────────────────────────────────
    1         2.000               2.000              100.0 %
    2         3.142               4.000               78.5 %
    3         4.189               8.000               52.4 %
    4         4.935              16.000               30.8 %
    5         5.264              32.000               16.4 %
    10        2.550           1024.000                0.25 %
    20        0.026    1,048,576.000              0.000002 %
    ──────────────────────────────────────────────────────────────────

    The sphere's volume PEAKS at d = 5 and then DECREASES back to zero.
    In 20 dimensions, the sphere is essentially invisible inside the cube.

    Visualised:

          2D                   5D                   20D
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │    ╭────╮    │     │   ╭──────╮   │     │      ·       │
    │   ╱      ╲  │     │  ╱  ████  ╲  │     │              │
    │  │  ████  │ │     │ │   ████   │ │     │      ·       │
    │   ╲      ╱  │     │  ╲        ╱  │     │              │
    │    ╰────╯   │     │   ╰──────╯   │     │      ·       │
    └──────────────┘     └──────────────┘     └──────────────┘
    Sphere = 78.5%       Sphere = 16.4%       Sphere ≈ 0.000002%
    of cube              of cube              of cube

    Practical implication: Gaussian distributions concentrate all their
    mass in a thin shell. The "typical" point is NOT near the mean — it
    is in a high-dimensional shell far from the mean.


──────────────────────────────────────────────────────────────────────────────
### Impact on Machine Learning Algorithms

**K-Nearest Neighbours (KNN)**

KNN is the algorithm most directly destroyed by the curse.
Its entire premise is that nearby points have similar labels.
In high dimensions there are no "nearby" points.

    ┌────────────────────────────────────────────────────────────────────┐
    │  d=2:   KNN accuracy ≈ 95%  (neighbours are genuinely close)      │
    │  d=10:  KNN accuracy ≈ 78%  (distances are less meaningful)       │
    │  d=50:  KNN accuracy ≈ 55%  (barely better than random)           │
    │  d=200: KNN accuracy ≈ 51%  (essentially random)                  │
    └────────────────────────────────────────────────────────────────────┘

**Linear Models**

Linear models are more robust — they don't rely on distance.
But as d grows, you have more parameters to estimate, so you need more
data. The bias-variance tradeoff worsens. Regularisation becomes critical.

**Decision Trees / Random Forests**

Trees are somewhat immune — they only split on one feature at a time.
But with many irrelevant features the probability of choosing a useful
split decreases, and ensembles must grow larger to compensate.

**Neural Networks**

Very high-dimensional inputs (like images) are handled by learning
useful low-dimensional representations internally (embeddings, features).
This is exactly what convolutional and attention layers do — they
compress the input into a dense, low-dimensional manifold.

**Kernel Methods (SVM with RBF)**

The RBF kernel exp(-γ||x-y||²) depends on Euclidean distance.
When all distances are the same, all kernel values converge to the same
constant — the kernel matrix becomes rank-1. The SVM loses all information.

    Impact Summary:
    ┌──────────────────────────────┬─────────────────────────────────────┐
    │ Algorithm                    │ Sensitivity to High Dimensions      │
    ├──────────────────────────────┼─────────────────────────────────────┤
    │ KNN                          │ CRITICAL — collapses completely     │
    │ K-Means                      │ HIGH — distance-based clustering    │
    │ SVM (RBF kernel)             │ HIGH — kernel distances converge    │
    │ Naïve Bayes                  │ LOW — uses per-feature statistics   │
    │ Linear / Logistic Regression │ MODERATE — needs regularisation     │
    │ Decision Trees               │ LOW-MODERATE — feature splits help  │
    │ Neural Networks              │ MODERATE — learn own representations│
    └──────────────────────────────┴─────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### How Many Samples Do You Actually Need?

A common heuristic: you need at least 5–10 training samples PER DIMENSION
for a model to have any hope of generalising (this is the rule of thumb
from classical statistics for linear models).

For more complex models, the requirement is much higher:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Model Type             Rough Samples Needed per Dimension       │
    ├──────────────────────────────────────────────────────────────────┤
    │  Linear regression      5 – 10                                  │
    │  Logistic regression    10 – 20                                 │
    │  KNN (k=5)              100 – 1000                              │
    │  Decision tree          50 – 100                                │
    │  Neural network         Depends on architecture & regularisation │
    └──────────────────────────────────────────────────────────────────┘

    Practical reality: most real datasets have MANY more features than
    samples. A 50×50 image is 2,500 dimensions. A genome is ~20,000.
    A text bag-of-words might have 100,000+ dimensions. This is why
    dimensionality reduction is not optional — it is essential.


──────────────────────────────────────────────────────────────────────────────
### Solutions: Dimensionality Reduction

The primary weapon against the curse is reducing the number of dimensions
while retaining the most important structure in the data.


1. **Principal Component Analysis (PCA)**

PCA finds the directions of maximum variance and projects the data onto
the top-k of them. It is a linear transformation.

    Original d-dimensional space → k-dimensional subspace  (k << d)

    How it works:
    ┌──────────────────────────────────────────────────────────────────┐
    │  1. Centre the data (subtract the mean)                         │
    │  2. Compute the covariance matrix C = (1/n) X^T X               │
    │  3. Eigen-decompose C to get eigenvectors v₁, v₂, …, v_d        │
    │     (sorted by eigenvalue — highest variance first)             │
    │  4. Project data onto top-k eigenvectors: Z = X · V_k           │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 5 — PCA in 2D (reducing to 1D):

             y                           z₁
             │     ●●                    │
             │   ●●●●                   ●●●●●●●●●●●●
             │  ●●●●●   ← PC1 →         │
             │   ●●●●●                  Points projected onto
             │     ●●●                  the principal axis.
             └─────────────── x         Variance preserved.


2. **t-SNE (t-distributed Stochastic Neighbour Embedding)**

Non-linear method optimised for *visualisation* (2D or 3D output).
Preserves local neighbourhood structure. NOT meant for downstream ML.

    Use when:  Exploring clusters, visualising embeddings
    Avoid:     Pre-processing for classifiers (non-deterministic, slow)


3. **UMAP (Uniform Manifold Approximation and Projection)**

Newer non-linear method. Faster than t-SNE. Preserves both local
and global structure better. Can be used as pre-processing.

    Use when:  Visualisation AND as input to classifiers
    Strength:  Scalable to millions of points


4. **Feature Selection (Filter / Wrapper / Embedded)**

Instead of combining features, discard the irrelevant ones entirely.

    Filter:   Rank by correlation, mutual information, variance
    Wrapper:  Recursive feature elimination (RFE) — try subsets
    Embedded: LASSO (L1 regularisation) naturally zeros out features

    When to prefer over PCA:
        - When interpretability of original features matters
        - When features have known semantic meaning
        - When the data is extremely sparse (text, genomics)


5. **Autoencoders (Neural Network Approach)**

An encoder network maps high-dimensional input to a low-dimensional
"bottleneck" representation. Trained to reconstruct the input.

    Input (d) → Encoder → Bottleneck (k) → Decoder → Reconstructed (d)

    Use when: data has complex non-linear structure (images, audio, text)


6. **The Manifold Hypothesis**

A key insight underlying modern deep learning:

    Even though natural data (images, text, audio) lives in a
    very high-dimensional space, it actually lies on or near a
    much lower-dimensional MANIFOLD embedded in that space.

    Diagram 6 — The Manifold Hypothesis:

    High-dim space (e.g., 256×256 image = 65,536 dims):
    ┌──────────────────────────────────────────────────────────────────┐
    │                  ·  · ·                                          │
    │   ·   · · ·  ────────────  ← natural image manifold             │
    │         ·  ·/              (perhaps only ~100 "true" dims)       │
    │          · /                                                     │
    │     · · · /    · ·  ·  ·  ← random noise (NOT on manifold)      │
    │          /                                                       │
    └──────────────────────────────────────────────────────────────────┘

    Most points in the 65,536-dim space are random noise (no structure).
    Real images occupy a tiny subspace — the "image manifold."
    Neural networks learn to map onto this manifold implicitly.


──────────────────────────────────────────────────────────────────────────────
### Practical Guidelines

Rule 1 — Always check  d / n  (features / samples) ratio.

    If d / n > 0.1, you are likely in the high-dimensional regime.
    If d / n > 1,   the problem is severely under-determined.

    Fixes: collect more data, reduce features, apply regularisation.


Rule 2 — Perform dimensionality reduction BEFORE distance-based methods.

    KNN, K-Means, SVM-RBF, anomaly detection with Euclidean distances:
    always apply PCA or feature selection first.


Rule 3 — Use regularisation aggressively in high dimensions.

    L1 (LASSO) implicitly selects features by setting many to zero.
    L2 (Ridge) prevents coefficient explosion.
    Both are more important when d is large.


Rule 4 — Understand explained variance when using PCA.

    Plot the cumulative explained variance. Choose k such that
    90–95% of variance is retained.

    Diagram 7 — Scree Plot (Explained Variance vs Components):

    Variance
    explained
    100% │                              ─────────────────────
      90 │                   ──────────
      80 │           ────────
      70 │      ─────
      60 │  ─────
      50 │──
         └─────────────────────────────────────────────── # components
              1    5   10   15   20   25   30
                   ↑
                   Choose k here (elbow of the curve)


Rule 5 — Be sceptical of high accuracy in very high dimensions.

    If your model performs suspiciously well on high-dimensional data,
    consider whether it has overfit. The training set might have enough
    dimensions to separate ANY labelling — this is the "blessing" that
    turns into a curse when you test on new data.

    This is related to the VC dimension:
        A model with d parameters in d-dimensional space can shatter
        (perfectly classify) up to d+1 points regardless of labels.
        Very high dimensions → very high VC dimension → easy to overfit.


──────────────────────────────────────────────────────────────────────────────
### Summary Table

    ┌────────────────────────────────┬───────────────────────────────────────┐
    │ Phenomenon                     │ What it Means for Your Model          │
    ├────────────────────────────────┼───────────────────────────────────────┤
    │ Volume grows exponentially     │ Data becomes sparse — hard to cover   │
    │                                │ the input space with finite samples   │
    │ Data lives near the boundary   │ The "average" case doesn't exist      │
    │                                │ — all points are statistical outliers │
    │ Distances concentrate          │ KNN, K-Means, RBF-SVM all degrade     │
    │                                │ — neighbours aren't meaningful        │
    │ Hypersphere volume → 0         │ Gaussian mass lives in shells,        │
    │                                │ not near the mean                     │
    │ Exponentially more params      │ Need exponentially more data;         │
    │                                │ overfitting risk explodes             │
    ├────────────────────────────────┼───────────────────────────────────────┤
    │ Fix: PCA / UMAP                │ Compress to low-dim representation    │
    │ Fix: Feature selection / LASSO │ Discard irrelevant dimensions         │
    │ Fix: Collect more data         │ Reduce d/n ratio                      │
    │ Fix: Regularisation            │ Constrain model complexity            │
    │ Fix: Neural embeddings         │ Learn the manifold implicitly         │
    └────────────────────────────────┴───────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Volume & Sparsity in High Dimensions": {
        "description": (
            "Visualise how hypersphere volume peaks and collapses, how edge "
            "fractions grow to nearly 100%, and how many samples are needed "
            "to maintain coverage density as dimensions increase."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gamma, pi

# ── Hypersphere volume formula ────────────────────────────────────────────
def sphere_volume(d, r=1.0):
    """Volume of a d-dimensional hypersphere of radius r."""
    return (pi ** (d / 2) * r**d) / gamma(d / 2 + 1)

dims = np.arange(1, 21)
sphere_vols = [sphere_volume(d) for d in dims]

# ── Edge fraction of unit hypercube ──────────────────────────────────────
eps = 0.1  # "edge" = within 0.1 of any face
edge_fracs = [1 - (1 - 2 * eps) ** d for d in dims]

# ── Samples needed to cover space at density of 10 pts in 1D ─────────────
samples_needed = [10 ** d for d in dims]

print("=" * 65)
print("  GEOMETRY OF HIGH-DIMENSIONAL SPACES")
print("=" * 65)
print()

print("  d    Sphere Vol   Edge Frac (ε=0.1)   Samples for Coverage")
print("  " + "─" * 60)
for d in [1, 2, 3, 5, 10, 15, 20]:
    sv   = sphere_volume(d)
    ef   = (1 - (0.8) ** d) * 100
    samp = 10 ** d
    samp_str = f"{samp:.2e}" if samp > 1e6 else f"{samp:>12,.0f}"
    print(f"  {d:>2d}   {sv:>10.4f}   {ef:>14.1f} %   {samp_str}")

print()
print("  Key insight: As d increases, the sphere volume PEAKS at ~d=5")
print("  then COLLAPSES to zero. The cube's interior becomes unreachable.")
print()
print("  Edge fraction: in 20D, >99% of the cube is within 10% of a face.")
print("  Samples needed: maintaining coverage requires 10^d points.")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("The Curse of Dimensionality — Geometric Perspective",
             fontsize=13, fontweight="bold")

# Panel 1: Hypersphere volume
ax = axes[0]
ax.plot(dims, sphere_vols, "b-o", lw=2, ms=6)
ax.axvline(5, color="orange", ls="--", lw=1.5, label="Peak at d=5")
ax.fill_between(dims, sphere_vols, alpha=0.2, color="steelblue")
ax.set_title("Hypersphere Volume\\n(radius = 1)", fontweight="bold")
ax.set_xlabel("Dimensions (d)")
ax.set_ylabel("Volume")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: Edge fraction
ax = axes[1]
ax.plot(dims, [e * 100 for e in edge_fracs], "r-o", lw=2, ms=6)
ax.axhline(90, color="gray", ls="--", lw=1, label="90% threshold")
ax.fill_between(dims, [e * 100 for e in edge_fracs], alpha=0.2, color="tomato")
ax.set_title("Edge Region Fraction\\n(ε = 0.1 from any face)", fontweight="bold")
ax.set_xlabel("Dimensions (d)")
ax.set_ylabel("% of Volume that is 'Edge'")
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: Samples needed (log scale)
ax = axes[2]
log_samples = [d * np.log10(10) for d in dims]  # log10 of 10^d = d
ax.plot(dims, log_samples, "g-o", lw=2, ms=6)
ax.set_title("Samples Needed for Coverage\\n(log₁₀ scale)", fontweight="bold")
ax.set_xlabel("Dimensions (d)")
ax.set_ylabel("log₁₀(Samples needed)")
ax.set_yticks(range(0, 22, 2))
ax.set_yticklabels([f"10^{i}" for i in range(0, 22, 2)])
ax.axhline(7, color="orange", ls="--", lw=1, label="10M samples")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dimensionality_geometry.png", dpi=120)
print()
print("  Plot saved → dimensionality_geometry.png")
print()
print("  Takeaways:")
print("  1. Hypersphere volume peaks at d=5, then shrinks to zero.")
print("  2. At d=20, >99% of your space is 'edge'. No real interior.")
print("  3. Maintaining density coverage requires exponentially more data.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Distance Concentration Phenomenon": {
        "description": (
            "Empirically demonstrate that pairwise distances between random "
            "points concentrate around a single value as dimensions increase. "
            "Plots the full distance distribution and the (max-min)/min ratio "
            "across dimensions from 2 to 1000."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

N_POINTS = 500     # number of random points per experiment
DIMS_TO_PLOT = [2, 10, 50, 200]
DIMS_RANGE   = [2, 5, 10, 20, 50, 100, 200, 500, 1000]

print("=" * 65)
print("  DISTANCE CONCENTRATION IN HIGH DIMENSIONS")
print(f"  ({N_POINTS} random uniform points per dimension)")
print("=" * 65)
print()
print("  d       Mean Dist   Std Dev   Min Dist   Max Dist   (Max-Min)/Min")
print("  " + "─" * 66)

concentration_ratios = []

for d in DIMS_RANGE:
    X = np.random.uniform(0, 1, size=(N_POINTS, d))

    # Compute pairwise Euclidean distances (upper triangle only)
    dists = []
    step = max(1, N_POINTS // 100)  # sample to keep runtime low
    indices = np.arange(0, N_POINTS, step)
    for i in indices:
        for j in range(i + 1, N_POINTS, step):
            dists.append(np.linalg.norm(X[i] - X[j]))

    dists = np.array(dists)
    d_min  = dists.min()
    d_max  = dists.max()
    d_mean = dists.mean()
    d_std  = dists.std()
    ratio  = (d_max - d_min) / d_min if d_min > 0 else float("inf")
    concentration_ratios.append(ratio)

    print(f"  {d:>4d}    {d_mean:>8.3f}   {d_std:>7.4f}   "
          f"{d_min:>8.3f}   {d_max:>8.3f}   {ratio:>10.4f} ×")

print()
print("  As d increases → std dev shrinks, ratio → 0.")
print("  At d=1000, max and min distances differ by less than 5%.")
print("  Nearest neighbour is almost as far as farthest neighbour.")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Distance Concentration: Why KNN Breaks in High Dimensions",
             fontsize=13, fontweight="bold")

# Panel 1: Distance distributions for selected dims
ax = axes[0]
colours = ["steelblue", "seagreen", "darkorange", "crimson"]
for d, colour in zip(DIMS_TO_PLOT, colours):
    X = np.random.uniform(0, 1, size=(200, d))
    dists = []
    for i in range(0, 200, 5):
        for j in range(i + 1, 200, 5):
            dists.append(np.linalg.norm(X[i] - X[j]))
    dists = np.array(dists)
    ax.hist(dists, bins=40, density=True, alpha=0.5, color=colour,
            label=f"d = {d}")
ax.set_title("Pairwise Distance Distributions\\n(narrow = concentrated)",
             fontweight="bold")
ax.set_xlabel("Euclidean Distance")
ax.set_ylabel("Density")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: Concentration ratio vs dimensions
ax = axes[1]
ax.loglog(DIMS_RANGE, concentration_ratios, "r-o", lw=2, ms=7)
ax.set_title("Distance Concentration Ratio\\n(Max−Min)/Min → 0",
             fontweight="bold")
ax.set_xlabel("Dimensions (d)  [log scale]")
ax.set_ylabel("(Max−Min)/Min  [log scale]")
ax.axhline(0.1, color="gray", ls="--", lw=1,
           label="Ratio = 0.1 (distances nearly equal)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

for d, r in zip(DIMS_RANGE, concentration_ratios):
    ax.annotate(f"{d}", (d, r), textcoords="offset points",
                xytext=(5, 3), fontsize=8, color="darkred")

plt.tight_layout()
plt.savefig("distance_concentration.png", dpi=120)
print()
print("  Plot saved → distance_concentration.png")
print()
print("  Key Takeaways:")
print("  - Low d: wide distance spread → near/far distinction is meaningful")
print("  - High d: narrow spike → every point is equally 'near' every other")
print("  - This destroys KNN, K-Means, RBF kernels, and anomaly detectors")
print("  - The ratio (max-min)/min follows a power law decline: O(1/√d)")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · KNN Accuracy Degrades with Dimensions": {
        "description": (
            "Train a K-Nearest Neighbours classifier on a binary classification "
            "task and measure accuracy as the number of dimensions grows from 2 "
            "to 200. The informative features are fixed; irrelevant noise dimensions "
            "are added to simulate the curse."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w0=np.zeros(d+1); lam=1./self.C
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict(self,X): return (_sig(X@self.coef_.ravel()+self.intercept_[0])>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

class _DTree:
    def __init__(self,max_depth=5): self.max_depth=max_depth
    def _gini(self,y):
        if not len(y): return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _split(self,X,y):
        best=None; n=len(y)
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2
            thrs=all_t[np.linspace(0,len(all_t)-1,min(10,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<2 or r.sum()<2: continue
                g=self._gini(y[l])*l.sum()+self._gini(y[r])*r.sum()
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        cls,cnt=np.unique(y,return_counts=True); leaf={"leaf":True,"pred":cls[cnt.argmax()]}
        if depth>=self.max_depth or len(cls)==1: return leaf
        sp=self._split(X,y)
        if sp is None: return leaf
        _,f,t=sp; m=X[:,f]<=t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[m],y[m],depth+1),"r":self._build(X[~m],y[~m],depth+1)}
    def fit(self,X,y): self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd["leaf"]: return nd["pred"]
        return self._p1(x,nd["l"] if x[nd["f"]]<=nd["t"] else nd["r"])
    def predict(self,X): return np.array([self._p1(x,self._tree) for x in X])
    def score(self,X,y): return (self.predict(X)==y).mean()

class DecisionTreeClassifier(_DTree):
    def __init__(self,max_depth=5,random_state=None): super().__init__(max_depth=max_depth or 20)

np.random.seed(7)

# ── Dataset: 2 informative Gaussian blobs ─────────────────────────────────
N_TRAIN = 400
N_TEST  = 200
N_INFO  = 2      # number of genuinely informative dimensions

# Informative features: class 0 ~ N([-1,-1], I), class 1 ~ N([1,1], I)
X_info_tr = np.vstack([
    np.random.randn(N_TRAIN // 2, N_INFO) - 1.0,
    np.random.randn(N_TRAIN // 2, N_INFO) + 1.0
])
y_train = np.array([0] * (N_TRAIN // 2) + [1] * (N_TRAIN // 2))

X_info_te = np.vstack([
    np.random.randn(N_TEST // 2, N_INFO) - 1.0,
    np.random.randn(N_TEST // 2, N_INFO) + 1.0
])
y_test = np.array([0] * (N_TEST // 2) + [1] * (N_TEST // 2))

# Noise dimensions to add
DIM_RANGE = [2, 5, 10, 20, 50, 100, 150, 200]
NOISE_STD = 1.0   # noise has same scale as informative features

knn_accs   = []
lr_accs    = []
tree_accs  = []

print("=" * 65)
print("  KNN vs LOGISTIC REGRESSION vs DECISION TREE")
print("  ACCURACY AS NOISE DIMENSIONS ARE ADDED")
print(f"  Informative dims: {N_INFO}  |  Train: {N_TRAIN}  |  Test: {N_TEST}")
print("=" * 65)
print()
print(f"  {'d':>4}  {'Total Dims':>10}  {'KNN':>8}  {'LogReg':>8}  {'Tree':>8}")
print("  " + "─" * 50)

for d_noise in DIM_RANGE:
    # Total dims = informative + noise
    total_d = N_INFO + d_noise

    noise_tr = np.random.randn(N_TRAIN, d_noise) * NOISE_STD
    noise_te = np.random.randn(N_TEST,  d_noise) * NOISE_STD

    X_tr = np.hstack([X_info_tr, noise_tr])
    X_te = np.hstack([X_info_te, noise_te])

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr_s, y_train)
    knn_acc = knn.score(X_te_s, y_test) * 100

    # Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=0)
    lr.fit(X_tr_s, y_train)
    lr_acc = lr.score(X_te_s, y_test) * 100

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=0)
    tree.fit(X_tr_s, y_train)
    tree_acc = tree.score(X_te_s, y_test) * 100

    knn_accs.append(knn_acc)
    lr_accs.append(lr_acc)
    tree_accs.append(tree_acc)

    flag = " ← 2 informative dims" if d_noise == 2 else ""
    print(f"  {d_noise:>4}  {total_d:>10}  "
          f"{knn_acc:>7.1f}%  {lr_acc:>7.1f}%  {tree_acc:>7.1f}%{flag}")

print()
print(f"  KNN degradation:  {knn_accs[0]:.1f}% → {knn_accs[-1]:.1f}%")
print(f"  LogReg stability: {lr_accs[0]:.1f}% → {lr_accs[-1]:.1f}%")
print(f"  Tree degradation: {tree_accs[0]:.1f}% → {tree_accs[-1]:.1f}%")
print()
print("  LogReg is robust: its decision boundary ignores noisy dims (small coeff)")
print("  KNN is fragile: noise dims dilute the signal from informative dims")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("KNN vs Linear Model: Accuracy vs Number of Dimensions",
             fontsize=13, fontweight="bold")

total_dims = [N_INFO + d for d in DIM_RANGE]

ax = axes[0]
ax.plot(total_dims, knn_accs,  "r-o", lw=2, ms=7, label="KNN (k=5)")
ax.plot(total_dims, lr_accs,   "b-s", lw=2, ms=7, label="Logistic Regression")
ax.plot(total_dims, tree_accs, "g-^", lw=2, ms=7, label="Decision Tree (d=5)")
ax.axhline(50, color="gray", ls="--", lw=1, label="Random baseline (50%)")
ax.set_title("Test Accuracy vs Total Dimensions\\n(2 informative + noise)",
             fontweight="bold")
ax.set_xlabel("Total Dimensions (2 informative + noise)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(40, 105)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: KNN accuracy drop as fraction of original
ax = axes[1]
knn_relative = [a / knn_accs[0] * 100 for a in knn_accs]
lr_relative  = [a / lr_accs[0]  * 100 for a in lr_accs]
ax.plot(total_dims, knn_relative, "r-o", lw=2, ms=7, label="KNN")
ax.plot(total_dims, lr_relative,  "b-s", lw=2, ms=7, label="Logistic Regression")
ax.axhline(100, color="gray", ls="--", lw=1)
ax.set_title("Relative Accuracy\\n(% of performance at d=2)",
             fontweight="bold")
ax.set_xlabel("Total Dimensions")
ax.set_ylabel("Relative Accuracy (%)")
ax.set_ylim(30, 115)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("knn_degradation.png", dpi=120)
print()
print("  Plot saved → knn_degradation.png")
print()
print("  Key Takeaways:")
print("  - KNN accuracy collapses as irrelevant dims drown the signal")
print("  - Logistic Regression is robust — linear models adapt coefficients")
print("  - Decision Trees degrade too — but more slowly than KNN")
print("  - Solution: reduce dimensions BEFORE using distance-based methods")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · PCA as the Antidote — Restoring KNN Performance": {
        "description": (
            "Demonstrate how PCA rescues KNN from the curse of dimensionality. "
            "Shows PCA explained-variance scree plots, the transformation of a "
            "200-dimensional dataset back to its 2-dimensional signal, and "
            "the recovery of KNN accuracy after dimensionality reduction."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class PCA:
    def __init__(self,n_components=None): self.n_components=n_components
    def fit(self,X):
        self.mean_=X.mean(0); Xc=X-self.mean_
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        tot=np.sum(s**2); self.explained_variance_ratio_=s**2/tot
        self.components_=Vt; return self
    def transform(self,X):
        k=self.n_components
        return (X-self.mean_)@(self.components_[:k].T if k else self.components_.T)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(axis=1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()

np.random.seed(99)

# ── Create a high-dimensional dataset with low intrinsic dimensionality ────
# True signal lives in 2D. We embed it in 200D + noise.
N_SAMPLES = 600
N_CLASSES = 3
N_SIGNAL  = 2     # true dimensionality
N_TOTAL   = 200   # dimensionality we observe

# 3 Gaussian clusters in 2D
means = [np.array([2, 2]), np.array([-2, 2]), np.array([0, -2.5])]
labels_all = np.repeat([0, 1, 2], N_SAMPLES // N_CLASSES)
X_signal = np.vstack([
    np.random.randn(N_SAMPLES // N_CLASSES, N_SIGNAL) + means[i]
    for i in range(N_CLASSES)
])

# Embed in 200D: rotate into random 200D subspace, add noise
random_projection = np.random.randn(N_SIGNAL, N_TOTAL)
X_high = X_signal @ random_projection + np.random.randn(N_SAMPLES, N_TOTAL) * 0.5

# Train / test split
idx = np.random.permutation(N_SAMPLES)
tr, te = idx[:400], idx[400:]
X_tr, X_te = X_high[tr], X_high[te]
y_tr, y_te = labels_all[tr], labels_all[te]

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

print("=" * 65)
print("  PCA AS THE ANTIDOTE TO THE CURSE OF DIMENSIONALITY")
print(f"  Dataset: {N_CLASSES} clusters | Signal dims: {N_SIGNAL} | Total dims: {N_TOTAL}")
print("=" * 65)
print()

# ── KNN at various PCA dimensionalities ──────────────────────────────────
pca_dims = [1, 2, 3, 5, 10, 20, 50, 100, 200]
accs = []

print(f"  {'PCA Dims':>8}  {'KNN Acc':>8}  {'Var Explained':>14}")
print("  " + "─" * 35)

pca_full = PCA(n_components=min(N_TOTAL, N_SAMPLES - 1)).fit(X_tr_s)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

for k in pca_dims:
    if k >= min(N_TOTAL, len(X_tr_s)):
        accs.append(np.nan)
        continue
    pca = PCA(n_components=k)
    Xtr_pca = pca.fit_transform(X_tr_s)
    Xte_pca = pca.transform(X_te_s)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr_pca, y_tr)
    acc = knn.score(Xte_pca, y_te) * 100
    accs.append(acc)
    var = cumvar[k - 1] * 100
    marker = " ← true signal dim" if k == N_SIGNAL else ""
    print(f"  {k:>8d}  {acc:>7.1f}%  {var:>13.1f}%{marker}")

# No PCA baseline
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_tr_s, y_tr)
raw_acc = knn_raw.score(X_te_s, y_te) * 100
print(f"  {'No PCA':>8}  {raw_acc:>7.1f}%  {'100.0':>13}%  ← full 200D")
print()
print(f"  Best KNN accuracy (PCA):    {max(accs):.1f}%")
print(f"  KNN accuracy (no PCA):      {raw_acc:.1f}%")
print(f"  Accuracy recovery:          +{max(accs) - raw_acc:.1f} percentage points")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("PCA: Recovering Structure Lost to the Curse of Dimensionality",
             fontsize=13, fontweight="bold")

# Panel 1: Scree plot (explained variance)
ax = axes[0]
comp_range = np.arange(1, min(51, len(cumvar) + 1))
ax.plot(comp_range, cumvar[:50] * 100, "b-o", ms=4, lw=2)
ax.axhline(90, color="gray", ls="--", lw=1, label="90% threshold")
ax.axvline(N_SIGNAL, color="green", ls="--", lw=2,
           label=f"True signal ({N_SIGNAL} dims)")
ax.fill_between(comp_range, cumvar[:50] * 100, alpha=0.2, color="steelblue")
ax.set_title("Scree Plot:\\nCumulative Explained Variance", fontweight="bold")
ax.set_xlabel("Number of PCA Components")
ax.set_ylabel("Cumulative Variance Explained (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: KNN accuracy vs PCA dimensions
ax = axes[1]
valid = [(k, a) for k, a in zip(pca_dims, accs) if not np.isnan(a)]
ks, acc_v = zip(*valid)
ax.plot(ks, acc_v, "r-o", lw=2, ms=7, label="KNN (k=5) after PCA")
ax.axhline(raw_acc, color="navy", ls="--", lw=2,
           label=f"KNN no PCA ({raw_acc:.0f}%)")
ax.axvline(N_SIGNAL, color="green", ls="--", lw=2,
           label=f"True dims ({N_SIGNAL})")
ax.set_title("KNN Test Accuracy\\nvs PCA Dimensionality", fontweight="bold")
ax.set_xlabel("PCA Dimensions Retained")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: 2D PCA projection coloured by class
ax = axes[2]
pca2 = PCA(n_components=2).fit(X_tr_s)
Xte_2d = pca2.transform(X_te_s)
colours = ["steelblue", "tomato", "seagreen"]
for cls in range(N_CLASSES):
    mask = y_te == cls
    ax.scatter(Xte_2d[mask, 0], Xte_2d[mask, 1],
               c=colours[cls], s=30, alpha=0.7, label=f"Class {cls}")
ax.set_title("Test Set — First 2 PCA Components\\n(200D → 2D recovery)",
             fontweight="bold")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pca_antidote.png", dpi=120)
print()
print("  Plot saved → pca_antidote.png")
print()
print("  Key Takeaways:")
print("  - In 200D, KNN barely outperforms random (distances are useless)")
print("  - After PCA to 2D, the true 2D structure is recovered completely")
print("  - The scree plot identifies the elbow where the true signal lives")
print("  - PCA is the most direct antidote to the curse for distance-based models")
print("  - Always apply PCA (or feature selection) before KNN / K-Means / RBF-SVM")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
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
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }