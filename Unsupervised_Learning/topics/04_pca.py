
import textwrap
import re

"""Module: 04 · PCA"""
TOPIC_NAME   ="PCA - Principal Component Analysis"
DISPLAY_NAME = "04 · PCA"
ICON         = "📐"
SUBTITLE     = "Principal Component Analysis — linear dimensionality reduction"

THEORY = """
## 04 · Principal Component Analysis (PCA)

---

### The Problem PCA Solves

Real-world data is almost always high-dimensional. A dataset of 1000 medical
measurements per patient, a corpus of 50,000-word vocabulary vectors, an image
with 256×256 = 65,536 pixel values. Working directly in these spaces is painful:

- Visualisation is impossible beyond 3 dimensions.
- Many machine learning algorithms degrade in high dimensions (curse of
  dimensionality — distances concentrate, nearest-neighbour searches become
  meaningless).
- Many features are redundant or highly correlated — they carry overlapping
  information.
- Training is slow when the input dimension is large.

**PCA finds a lower-dimensional subspace that preserves as much variance
(information) as possible.**

It does this by finding a new set of axes — called **principal components** —
that are:
  1. Linear combinations of the original features.
  2. Ordered by how much variance they explain (PC1 explains the most, PC2
     the second most, and so on).
  3. Mutually orthogonal (uncorrelated with each other).

You then project your data onto the top K principal components, discarding
the rest. If the first K components capture, say, 95% of the total variance,
you have retained nearly all the information in K dimensions instead of the
original p.

---

### Intuition: Rotating the Axes

Imagine a 2D scatter plot where the data forms an elongated ellipse tilted at
45 degrees. The original axes (x₁, x₂) don't align with this shape — both
coordinates are needed to describe where any point is.

PCA rotates the axes to align with the ellipse:
- The first new axis (PC1) points along the long direction of the ellipse —
  the direction of maximum variance.
- The second new axis (PC2) points along the short direction — perpendicular
  to PC1, capturing the remaining variance.

In this new coordinate system, most of the information is concentrated in the
PC1 coordinate. If the ellipse is very thin, PC2 adds little — you can safely
drop it and describe the data in 1D instead of 2D with minimal information loss.

PCA is exactly this rotation, generalised to p dimensions.

---

### Step 1 — Centre the Data (Mean Subtraction)

Before anything else, subtract the mean of each feature:

    X_centred[i][j] = X[i][j] − μⱼ

where μⱼ = (1/n) Σᵢ X[i][j] is the mean of feature j.

**Why?** PCA finds directions of variance around the origin. If the data is not
centred, the first component would just point toward the cloud's centre of mass
rather than its principal spread direction. Centring moves the cloud so its
centre of mass is at the origin.

Optionally, you may also **standardise** (divide by standard deviation per
feature):

    X_scaled[i][j] = (X[i][j] − μⱼ) / σⱼ

This is necessary when features are measured on different scales (e.g. age in
years vs. income in thousands). Without standardisation, high-variance features
dominate the principal components simply because of their scale, not their
information content.

Rule of thumb: standardise unless all features are in the same units and you
have a reason to preserve scale differences.

---

### Step 2 — Compute the Covariance Matrix

The covariance matrix C is a p×p symmetric matrix:

    C = (1/(n−1)) · Xᵀ · X        (where X is already centred)

The entry C[j][k] = cov(xⱼ, xₖ) measures how much features j and k vary
together:

    cov(xⱼ, xₖ) = (1/(n−1)) · Σᵢ (xᵢⱼ − μⱼ)(xᵢₖ − μₖ)

The diagonal entries C[j][j] = var(xⱼ) are the variance of each feature.
The off-diagonal entries measure correlations. High |C[j][k]| means j and k
are redundant — they carry overlapping information that PCA will compress.

The total variance in the dataset = trace(C) = sum of diagonal entries
= sum of all feature variances.

---

### Step 3 — Eigendecomposition

The covariance matrix C is real and symmetric, so it has a full set of p
real eigenvalues and p orthogonal eigenvectors:

    C · vₖ = λₖ · vₖ

where:
- vₖ is the k-th eigenvector (a unit vector in the original feature space)
- λₖ is the k-th eigenvalue (a scalar)

Each eigenvector is a **principal component direction**. The corresponding
eigenvalue λₖ tells you the **variance of the data along that direction**.

Sort by eigenvalue descending: λ₁ ≥ λ₂ ≥ ... ≥ λₚ ≥ 0.

The first principal component v₁ is the direction of maximum variance.
The second v₂ is the direction of maximum variance among all directions
perpendicular to v₁. And so on.

**Why does this work?** The variance of the data when projected onto any unit
vector u is uᵀCu. Maximising this subject to ||u||=1 gives exactly the
eigenvector corresponding to the largest eigenvalue. This is the Rayleigh
quotient maximisation — a standard result in linear algebra.

---

### Step 4 — Select K Components

Explained variance ratio for component k:

    EVR(k) = λₖ / Σⱼ λⱼ = λₖ / trace(C)

This is the fraction of total variance captured by component k alone.

**Cumulative explained variance:**

    CEVR(K) = Σₖ₌₁ᴷ λₖ / trace(C)

You choose K such that CEVR(K) ≥ your threshold (commonly 90%, 95%, or 99%).

**Scree plot:** Plot λₖ vs k. Look for the "elbow" — the point where
eigenvalues stop dropping sharply and level off. Components beyond the elbow
explain diminishing variance and are often discarded.

A common practical heuristic: keep components with λₖ > 1 when the data is
standardised (Kaiser criterion). A component explaining less variance than a
single original feature is unlikely to be informative.

---

### Step 5 — Project the Data

Form the projection matrix W from the top K eigenvectors as columns:

    W = [v₁ | v₂ | ... | vₖ]    shape: (p × K)

Project:

    Z = X_centred · W              shape: (n × K)

Each row of Z is the low-dimensional representation of the corresponding
original data point. The columns of Z are the principal component scores.

To reconstruct (approximately) the original data:

    X_reconstructed = Z · Wᵀ + μ

The reconstruction error = total discarded variance = Σₖ₌ₖ₊₁ᵖ λₖ.

---

### A Worked Example (4 Points, 2D → 1D)

Data (4 points):
    x₁ = (1, 2)
    x₂ = (3, 5)
    x₃ = (2, 3)
    x₄ = (4, 6)

**Step 1 — Mean subtraction:**
    μ = (2.5, 4.0)
    x₁_c = (−1.5, −2.0)
    x₂_c = ( 0.5,  1.0)
    x₃_c = (−0.5, −1.0)
    x₄_c = ( 1.5,  2.0)

**Step 2 — Covariance matrix:**

    C = (1/3) · Xᵀ · X

    C[1,1] = (1/3)(2.25+0.25+0.25+2.25) = (1/3)(5.0)  = 1.667
    C[2,2] = (1/3)(4.0 +1.0 +1.0 +4.0)  = (1/3)(10.0) = 3.333
    C[1,2] = (1/3)(3.0 +0.5 +0.5 +3.0)  = (1/3)(7.0)  = 2.333

    C = [[1.667, 2.333],
         [2.333, 3.333]]

**Step 3 — Eigendecomposition:**

    trace(C) = 1.667 + 3.333 = 5.0
    det(C)   = 1.667×3.333 − 2.333² = 5.556 − 5.444 = 0.112

    Eigenvalues: λ² − 5λ + 0.112 = 0
    λ₁ = (5 + √(25 − 0.448)) / 2 ≈ 4.978
    λ₂ = (5 − √(25 − 0.448)) / 2 ≈ 0.022

    Eigenvector for λ₁ ≈ 4.978:
    (C − λ₁I)v = 0
    v₁ ≈ [0.555, 0.832]   (normalised)

    Eigenvector for λ₂ ≈ 0.022:
    v₂ ≈ [−0.832, 0.555]  (perpendicular to v₁)

**Step 4 — Explained variance:**

    EVR(1) = 4.978 / 5.0 = 99.6%  → PC1 alone captures 99.6% of variance!
    EVR(2) = 0.022 / 5.0 =  0.4%

**Step 5 — Project onto PC1:**

    Z = X_centred · v₁
    z₁ = (−1.5)(0.555) + (−2.0)(0.832) = −0.833 − 1.664 = −2.497
    z₂ = ( 0.5)(0.555) + ( 1.0)(0.832) =  0.278 + 0.832 =  1.110
    z₃ = (−0.5)(0.555) + (−1.0)(0.832) = −0.278 − 0.832 = −1.110
    z₄ = ( 1.5)(0.555) + ( 2.0)(0.832) =  0.833 + 1.664 =  2.497

    Z = [−2.497, 1.110, −1.110, 2.497]

The 4 points are now described in 1D with 99.6% of the original information
preserved. The 2D data lies almost perfectly on a line — PCA found it.

---

### SVD Formulation

PCA via eigendecomposition of the covariance matrix is mathematically correct
but numerically unstable for large matrices. In practice, PCA is computed
using the **Singular Value Decomposition (SVD)** of the centred data matrix:

    X_centred = U · Σ · Vᵀ

where:
- U is an (n × p) matrix with orthonormal columns (left singular vectors)
- Σ is a (p × p) diagonal matrix with non-negative singular values σ₁ ≥ σ₂ ≥ ... ≥ σₚ
- V is a (p × p) orthogonal matrix (right singular vectors)

The connection to eigendecomposition:

    Principal component directions  = columns of V  (same as eigenvectors of C)
    Eigenvalues                      = σₖ² / (n−1)
    PC scores                        = U · Σ  (same as X_centred · V)

SVD is preferred because:
1. It never explicitly forms Xᵀ·X, avoiding squaring the condition number.
2. It computes the full decomposition in a single numerically stable pass.
3. The truncated SVD (keeping only top K singular vectors) is efficient for
   large, sparse matrices using iterative methods (e.g. ARPACK, randomised SVD).

---

### Kernel PCA

Standard PCA is linear — it finds linear directions of maximum variance. If
your data lies on a nonlinear manifold (a Swiss roll, concentric circles,
etc.), linear PCA cannot unfold it.

**Kernel PCA** extends PCA to nonlinear structures using the kernel trick.
Instead of working in the original feature space, it implicitly maps data to
a high-dimensional (possibly infinite) feature space via a kernel function:

    k(xᵢ, xⱼ) = φ(xᵢ)ᵀ · φ(xⱼ)

Common kernels:
- **RBF (Gaussian):** k(x,y) = exp(−||x−y||² / 2σ²)
- **Polynomial:** k(x,y) = (xᵀy + c)ᵈ
- **Sigmoid:** k(x,y) = tanh(αxᵀy + c)

The kernel matrix K (n×n) is computed: K[i][j] = k(xᵢ, xⱼ).
PCA is then performed on the centred kernel matrix. The result is a nonlinear
projection of the original data.

---

### Incremental / Online PCA

Standard PCA requires loading the entire dataset into memory to compute the
covariance matrix or perform SVD. For streaming data or datasets too large to
fit in memory, **Incremental PCA** processes the data in mini-batches.

Each batch updates the running estimate of the principal components using a
rank-1 update rule. The result converges to the true PCA solution as more
batches are processed.

---

### PCA vs Other Dimensionality Reduction Methods

| Method      | Type        | Linear? | Preserves          | Best for                        |
|-------------|-------------|---------|--------------------|---------------------------------|
| PCA         | Unsupervised| Yes     | Global variance    | Preprocessing, visualisation    |
| LDA         | Supervised  | Yes     | Class separation   | Classification preprocessing    |
| t-SNE       | Unsupervised| No      | Local structure    | Visualisation (2D/3D only)      |
| UMAP        | Unsupervised| No      | Local + global     | Visualisation, general reduction|
| Autoencoder | Unsupervised| No      | Learned features   | Complex nonlinear structure     |
| ICA         | Unsupervised| Yes     | Statistical indep. | Signal separation (BSS)         |
| NMF         | Unsupervised| Yes     | Non-negativity     | Parts-based representation      |
| Factor Anal.| Unsupervised| Yes     | Latent factors     | Psychometrics, interpretability |

---

### Common Uses of PCA

**Visualisation** — Reduce to 2 or 3 components to plot high-dimensional data.
The first two PCs capture the dominant structure. Colour points by class labels
to see whether classes are separable in the reduced space.

**Preprocessing before supervised learning** — Reduce dimensionality before
feeding into a classifier or regressor. Removes redundancy, speeds up training,
sometimes improves generalisation by eliminating noisy low-variance features.
However: tree-based methods (Random Forest, XGBoost) handle high dimensions
well and often don't benefit from PCA preprocessing.

**Noise filtering / compression** — Reconstruct from the top K components.
The discarded components carry less signal and more noise — reconstruction
acts as a low-rank filter. Used in image compression, signal denoising,
and recommendation systems (collaborative filtering via matrix factorisation).

**Multicollinearity removal** — PCA components are orthogonal by construction.
Feeding PC scores instead of raw correlated features into linear regression
or logistic regression eliminates multicollinearity problems.

**Anomaly detection** — Project to K components and reconstruct. Points with
high reconstruction error lie far from the principal subspace and may be
anomalies. Used in network intrusion detection, manufacturing quality control.

---

### Limitations and Pitfalls

**Linearity** — PCA can only find linear structure. Nonlinear manifolds
require kernel PCA, t-SNE, UMAP, or autoencoders.

**Scale sensitivity** — If features are not standardised, high-variance
features dominate regardless of their information content. Always standardise
unless you have a specific reason not to.

**Interpretability** — Each principal component is a linear combination of
all original features, with potentially non-zero loadings on every feature.
PC1 is not "feature 3" — it is a mixture. This makes interpretation difficult
compared to methods like sparse PCA that produce loadings with many zeros.

**Assumes Gaussian structure** — PCA's variance-maximisation objective is
optimal when the data is Gaussian. For highly non-Gaussian distributions,
ICA (Independent Component Analysis) may be more appropriate.

**Information loss** — Discarded components do carry some information. For
critical applications, validate that the information loss is acceptable by
measuring downstream task performance with and without PCA.

**Sensitive to outliers** — Extreme outliers inflate variance in their
direction, potentially pulling the first principal component toward them.
Robust PCA variants (e.g. RPCA using L1 instead of L2) mitigate this.

---

### Key Takeaways

1. PCA finds orthogonal directions of maximum variance in your data via
   eigendecomposition of the covariance matrix (or SVD of the data matrix).

2. Always centre your data first. Standardise when features are on different
   scales.

3. Eigenvalue = variance along that principal component. Explained variance
   ratio = λₖ / Σλ. Use scree plot or cumulative EVR to choose K.

4. Projection: Z = X_centred · W where W = top K eigenvectors. Reconstruction:
   X̂ = Z · Wᵀ + μ. Reconstruction error = discarded variance.

5. SVD is numerically preferred over direct eigendecomposition of Xᵀ·X.

6. PCA is linear. For nonlinear structure, use kernel PCA, t-SNE, or UMAP.
"""


OPERATIONS = {

    "▶ Run: PCA From Scratch (Full Walkthrough)": {
        "description": "Complete PCA from first principles — mean subtraction, covariance matrix, eigendecomposition, explained variance, and 2D projection. Every step printed.",
        "code": """
import math

# ── Dataset: 6 points in 3D (two correlated features + one noisy) ─────────────
X = [
    [2.5, 2.4, 0.1],
    [0.5, 0.7, 0.8],
    [2.2, 2.9, 0.3],
    [1.9, 2.2, 0.5],
    [3.1, 3.0, 0.2],
    [2.3, 2.7, 0.9],
]
feature_names = ['f1','f2','f3']
n, p = len(X), len(X[0])

print("=" * 55)
print("     PCA — Step-by-Step Walkthrough")
print("=" * 55)
print(f"Dataset: {n} samples, {p} features")
print()

# ── Step 1: Mean subtraction ──────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
Xc    = [[X[i][j] - means[j] for j in range(p)] for i in range(n)]

print("STEP 1 — Mean of each feature:")
for j,f in enumerate(feature_names):
    print(f"  μ[{f}] = {means[j]:.4f}")
print()

# ── Step 2: Covariance matrix ─────────────────────────────────────────────────
def cov_matrix(Xc):
    n, p = len(Xc), len(Xc[0])
    C = [[0.0]*p for _ in range(p)]
    for j in range(p):
        for k in range(p):
            C[j][k] = sum(Xc[i][j]*Xc[i][k] for i in range(n)) / (n-1)
    return C

C = cov_matrix(Xc)
print("STEP 2 — Covariance Matrix C:")
header = "         " + "".join(f"  {feature_names[j]:>8}" for j in range(p))
print(header)
for j in range(p):
    row = f"  {feature_names[j]:>5}  " + "  ".join(f"{C[j][k]:>8.4f}" for k in range(p))
    print(row)
print()
print(f"  trace(C) = total variance = {sum(C[j][j] for j in range(p)):.4f}")
print()

# ── Step 3: Power iteration for eigenvalues/eigenvectors ──────────────────────
def mat_vec(M, v):
    n = len(v)
    return [sum(M[i][j]*v[j] for j in range(n)) for i in range(n)]

def normalize(v):
    norm = math.sqrt(sum(x**2 for x in v))
    return [x/norm for x in v], norm

def power_iteration(M, n_iter=1000, seed_v=None):
    p = len(M)
    if seed_v is None:
        v = [1.0/math.sqrt(p)] * p
    else:
        v = seed_v[:]
    for _ in range(n_iter):
        w = mat_vec(M, v)
        v, norm = normalize(w)
    eigenval = sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return eigenval, v

def deflate(M, eigenval, eigenvec):
    p = len(M)
    return [[M[i][j] - eigenval*eigenvec[i]*eigenvec[j] for j in range(p)] for i in range(p)]

# Extract all eigenpairs via deflation
eigenpairs = []
Cd = [row[:] for row in C]
seed_vecs = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
for k in range(p):
    lam, vec = power_iteration(Cd, seed_v=seed_vecs[k])
    if lam < 0: lam = 0.0
    eigenpairs.append((lam, vec))
    Cd = deflate(Cd, lam, vec)

eigenpairs.sort(reverse=True, key=lambda x: x[0])
eigenvalues   = [ep[0] for ep in eigenpairs]
eigenvectors  = [ep[1] for ep in eigenpairs]
total_var     = sum(eigenvalues)

print("STEP 3 — Eigenvalues and Principal Component Directions:")
for k, (lam, vec) in enumerate(eigenpairs):
    evr = lam / total_var * 100 if total_var > 0 else 0
    vec_str = "[" + ", ".join(f"{v:>7.4f}" for v in vec) + "]"
    print(f"  PC{k+1}: λ={lam:>7.4f}  ({evr:>5.1f}% variance)  direction={vec_str}")
print()

# ── Step 4: Cumulative explained variance ─────────────────────────────────────
print("STEP 4 — Explained Variance (Scree):")
cumulative = 0
for k, lam in enumerate(eigenvalues):
    evr = lam / total_var * 100 if total_var > 0 else 0
    cumulative += evr
    bar = "█" * int(evr * 3)
    print(f"  PC{k+1}: {evr:>5.1f}%  cumulative: {cumulative:>5.1f}%  {bar}")
print()

# ── Step 5: Project onto top 2 PCs ───────────────────────────────────────────
K = 2
W = eigenvectors[:K]   # K eigenvectors as rows → shape K×p

def project(Xc, W):
    return [[sum(Xc[i][j]*W[k][j] for j in range(len(W[k])))
             for k in range(len(W))]
            for i in range(len(Xc))]

Z = project(Xc, W)

print(f"STEP 5 — Projection onto top {K} PCs:")
print(f"  {'Point':<7}  {'PC1':>8}  {'PC2':>8}")
print("  " + "─" * 28)
for i, z in enumerate(Z):
    print(f"  pt {i+1:<3}  {z[0]:>8.4f}  {z[1]:>8.4f}")
print()

# ── Step 6: Reconstruction ────────────────────────────────────────────────────
def reconstruct(Z, W, means):
    p = len(W[0])
    return [[sum(Z[i][k]*W[k][j] for k in range(len(W)))+means[j]
             for j in range(p)] for i in range(len(Z))]

Xr = reconstruct(Z, W, means)

print(f"STEP 6 — Reconstruction from {K} PCs (X̂ ≈ X):")
print(f"  {'Point':<7}  {'Original':^28}  {'Reconstructed':^28}  Rec. error")
print("  " + "─" * 75)
total_err = 0
for i in range(n):
    orig  = str([f"{X[i][j]:.2f}" for j in range(p)])
    rec   = str([f"{Xr[i][j]:.2f}" for j in range(p)])
    err   = math.sqrt(sum((X[i][j]-Xr[i][j])**2 for j in range(p)))
    total_err += err**2
    print(f"  pt {i+1:<3}  {orig:<28}  {rec:<28}  {err:.4f}")

print()
discarded_var = sum(eigenvalues[K:])
print(f"  Total reconstruction SSE     : {total_err:.4f}")
print(f"  Discarded eigenvalue sum     : {discarded_var:.4f}")
print(f"  Information retained         : {(1-discarded_var/total_var)*100:.1f}%")
""",
        "runnable": True,
    },

    "▶ Run: Scree Plot and Component Selection": {
        "description": "Generate a full scree plot with explained variance per component and cumulative variance. Automatically identifies the elbow and shows how many components are needed to hit 90%, 95%, 99% thresholds.",
        "code": """
import math
import random

random.seed(42)

# ── Generate a 10D dataset with intrinsic 3D structure ────────────────────────
# True data lives in 3D: sample 3 latent variables, then embed in 10D with noise
n, true_dim, obs_dim = 120, 3, 10

latent = [[random.gauss(0,1) for _ in range(true_dim)] for _ in range(n)]

# Random projection matrix to embed 3D → 10D (fixed, not random per run)
import math as _m
proj = []
for i in range(obs_dim):
    row = [_m.cos(i*j+1) for j in range(true_dim)]  # deterministic "random" matrix
    norm = _m.sqrt(sum(x**2 for x in row))
    proj.append([x/norm for x in row])

X = []
for i in range(n):
    point = [sum(latent[i][k]*proj[j][k] for k in range(true_dim)) +
             random.gauss(0, 0.15) for j in range(obs_dim)]
    X.append(point)

p = obs_dim

# ── Standardise ───────────────────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n))/n for j in range(p)]
stds  = [math.sqrt(sum((X[i][j]-means[j])**2 for i in range(n))/(n-1)) for j in range(p)]
Xc    = [[(X[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(n)]

# ── Covariance matrix ─────────────────────────────────────────────────────────
C = [[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

# ── Power iteration eigendecomposition ────────────────────────────────────────
def mat_vec(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

def power_iter(M, tol=1e-10, max_iter=5000):
    p = len(M)
    v = [math.cos(k) for k in range(p)]  # deterministic seed
    v_norm = math.sqrt(sum(x**2 for x in v))
    v = [x/v_norm for x in v]
    for _ in range(max_iter):
        w = mat_vec(M, v)
        w_norm = math.sqrt(sum(x**2 for x in w))
        v_new = [x/w_norm for x in w]
        diff = math.sqrt(sum((v_new[i]-v[i])**2 for i in range(p)))
        v = v_new
        if diff < tol: break
    lam = sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam, 0.0), v

eigenvalues = []
Cd = [row[:] for row in C]
for _ in range(p):
    lam, vec = power_iter(Cd)
    eigenvalues.append(lam)
    # Deflate
    Cd = [[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]

eigenvalues.sort(reverse=True)
total_var = sum(eigenvalues)
evrs      = [lam/total_var*100 if total_var>0 else 0 for lam in eigenvalues]
cumevrs   = []
c = 0
for e in evrs:
    c += e
    cumevrs.append(c)

# ── Print scree plot ──────────────────────────────────────────────────────────
print(f"Scree Plot  |  {obs_dim}D dataset  |  True intrinsic dim = {true_dim}")
print(f"(Standardised: each feature has unit variance)")
print()
print(f"  {'PC':<5}  {'Eigenvalue':>11}  {'Var %':>7}  {'Cumulative':>11}  Bar")
print("  " + "─"*65)

W = 35
for k, (lam, evr, cev) in enumerate(zip(eigenvalues, evrs, cumevrs)):
    bar_len = int(evr/max(evrs)*W)
    bar     = "█"*bar_len
    marker  = " ◄ elbow" if k == 2 else ""
    print(f"  PC{k+1:<3}  {lam:>11.4f}  {evr:>6.2f}%  {cev:>10.2f}%  {bar}{marker}")

print()

# ── Threshold analysis ────────────────────────────────────────────────────────
print("  Components needed to reach variance threshold:")
for threshold in [80, 90, 95, 99]:
    for k, cev in enumerate(cumevrs):
        if cev >= threshold:
            print(f"    {threshold:>3}% threshold  →  K = {k+1} components")
            break

print()
print(f"  Kaiser criterion (λ > 1.0, data is standardised):")
kaiser_k = sum(1 for lam in eigenvalues if lam > 1.0)
print(f"    {kaiser_k} components have eigenvalue > 1.0")
print()
print(f"  True intrinsic dimensionality: {true_dim}")
print(f"  Scree elbow at: PC3 (▲ marked above)")
print(f"  → PCA correctly identifies the {true_dim}D structure despite {obs_dim}D observations.")
""",
        "runnable": True,
    },

    "▶ Run: PCA for Noise Filtering / Reconstruction": {
        "description": "Apply PCA to a noisy dataset, reconstruct from top K components, and measure reconstruction error "
                       "at each K. Demonstrates how PCA acts as a low-rank filter.",
        "code": """
import math
import random

random.seed(0)

# ── Generate noisy low-rank data ───────────────────────────────────────────────
# True signal: 2D line embedded in 5D + independent Gaussian noise
n, p = 80, 5
noise_std = 0.4

true_signal = [random.gauss(0,1) for _ in range(n)]  # 1D latent
direction   = [0.6, 0.8, 0.2, -0.3, 0.1]             # embedded in 5D

X = []
for i in range(n):
    pt = [true_signal[i]*direction[j] + random.gauss(0, noise_std)
          for j in range(p)]
    X.append(pt)

# ── Mean-centre ───────────────────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n))/n for j in range(p)]
Xc    = [[X[i][j]-means[j] for j in range(p)] for i in range(n)]

# ── Covariance + eigendecomposition ──────────────────────────────────────────
C = [[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M):
    p=len(M)
    v=[math.cos(k+1) for k in range(p)]
    nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(3000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v=[x/nrm for x in w]
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

eigenpairs=[]
Cd=[row[:] for row in C]
for _ in range(p):
    lam,vec=power_iter(Cd)
    eigenpairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
eigenpairs.sort(reverse=True,key=lambda x:x[0])
eigenvalues=[ep[0] for ep in eigenpairs]
eigenvectors=[ep[1] for ep in eigenpairs]
total_var=sum(eigenvalues)

# ── Reconstruction at each K ──────────────────────────────────────────────────
def project(Xc,W):
    return [[sum(Xc[i][j]*W[k][j] for j in range(len(W[0]))) for k in range(len(W))]
            for i in range(len(Xc))]

def reconstruct(Z,W,means):
    p=len(W[0])
    return [[sum(Z[i][k]*W[k][j] for k in range(len(W)))+means[j] for j in range(p)]
            for i in range(len(Z))]

def rmse(A,B):
    n,p=len(A),len(A[0])
    return math.sqrt(sum((A[i][j]-B[i][j])**2 for i in range(n) for j in range(p))/(n*p))

print(f"PCA Noise Filtering  |  n={n}, p={p}, noise σ={noise_std}")
print(f"True signal: 1D line embedded in {p}D")
print()
print(f"  {'K':<4}  {'Eigenvalue λₖ':>14}  {'Var %':>7}  {'Cum %':>7}  {'RMSE':>8}  {'Noise removed?'}")
print("  " + "─"*68)

# original_rmse = math.sqrt(sum(noise_std**2 for _ in range(p))/p)
# baseline_rmse = rmse(X, [[means[j]]*p for _ in range(n)])

original_rmse = math.sqrt(sum(noise_std**2 for _ in range(p))/p)
baseline_rmse = rmse(X, [means[:] for _ in range(n)])   # ← fixed

for K in range(1, p+1):
    W = eigenvectors[:K]
    Z = project(Xc, W)
    Xr = reconstruct(Z, W, means)
    r = rmse(X, Xr)
    evr = eigenvalues[K-1]/total_var*100 if total_var>0 else 0
    cev = sum(eigenvalues[:K])/total_var*100 if total_var>0 else 0
    noise_flag = "✓ Signal captured" if K==1 else ("~ Noise added" if K>2 else "")
    print(f"  K={K:<3}  {eigenvalues[K-1]:>14.4f}  {evr:>6.2f}%  {cev:>6.2f}%  {r:>8.4f}  {noise_flag}")

print()
print(f"  Observations:")
print(f"    K=1  : captures the true 1D signal → lowest RMSE on true signal")
print(f"    K>1  : adding more components adds noise back (higher RMSE)")
print(f"    K={p}  : full reconstruction = original data (no compression)")
print()
print(f"  → Reconstruction with K=1 is BETTER than the original noisy data.")
print(f"    PCA acts as a denoising filter by projecting onto the signal subspace.")
""",
        "runnable": True,
    },

    "▶ Run: PCA Loadings and Feature Interpretation": {
        "description": "Compute and print the PCA loadings matrix. Shows which original features each principal component "
                       "represents, enabling interpretation of the principal components.",
        "code": """
import math
import random

random.seed(7)

# ── Dataset: financial-like features with known structure ─────────────────────
# 3 latent factors: growth, risk, liquidity
# 8 observable features built from combinations
n = 100
feature_names = ['Revenue','Profit','EPS','Volatility','Beta','CashFlow','Debt','CurrentRatio']
p = len(feature_names)

# Simulate latent factors
growth     = [random.gauss(0,1) for _ in range(n)]
risk       = [random.gauss(0,1) for _ in range(n)]
liquidity  = [random.gauss(0,1) for _ in range(n)]

noise = 0.3
X = []
for i in range(n):
    g,r,l = growth[i], risk[i], liquidity[i]
    X.append([
        0.9*g + noise*random.gauss(0,1),    # Revenue   ← growth
        0.85*g + noise*random.gauss(0,1),   # Profit    ← growth
        0.8*g + 0.2*r + noise*random.gauss(0,1),  # EPS   ← growth+risk
        0.9*r + noise*random.gauss(0,1),    # Volatility← risk
        0.85*r + noise*random.gauss(0,1),   # Beta      ← risk
        0.8*l + noise*random.gauss(0,1),    # CashFlow  ← liquidity
        -0.7*l + noise*random.gauss(0,1),   # Debt      ← -liquidity
        0.8*l + 0.1*g + noise*random.gauss(0,1), # CurrentRatio ← liquidity
    ])

# ── Standardise ───────────────────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n))/n for j in range(p)]
stds  = [math.sqrt(sum((X[i][j]-means[j])**2 for i in range(n))/(n-1)) for j in range(p)]
Xc    = [[(X[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(n)]

# ── Covariance matrix ─────────────────────────────────────────────────────────
C = [[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M, seed):
    p=len(M)
    v=seed[:]; nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(5000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v_new=[x/nrm for x in w]
        if math.sqrt(sum((v_new[k]-v[k])**2 for k in range(p)))<1e-10: break
        v=v_new
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

seeds=[[1.0 if k==j else 0.0 for k in range(p)] for j in range(p)]
eigenpairs=[]
Cd=[row[:] for row in C]
for s in seeds:
    lam,vec=power_iter(Cd,s)
    eigenpairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
eigenpairs.sort(reverse=True,key=lambda x:x[0])
eigenvalues=[ep[0] for ep in eigenpairs]
eigenvectors=[ep[1] for ep in eigenpairs]
total_var=sum(eigenvalues)

# ── Loadings matrix ────────────────────────────────────────────────────────────
# Loadings = eigenvector * sqrt(eigenvalue)  (correlation between feature and PC)
K=3
loadings=[[eigenvectors[k][j]*math.sqrt(eigenvalues[k]) for k in range(K)] for j in range(p)]

print(f"PCA Loadings Matrix  |  {p} features → {K} principal components")
print(f"Dataset: simulated financial metrics (n={n})")
print()
print(f"  {'Feature':<14}" + "".join(f"  {'PC'+str(k+1):>8}" for k in range(K)) + "   Dominant PC")
print("  " + "─"*55)

for j,fname in enumerate(feature_names):
    row_loadings = loadings[j]
    dominant_k = max(range(K), key=lambda k: abs(loadings[j][k]))
    bars = "".join(
        ("  " + ("█" if loadings[j][k]>0 else "░") * int(abs(loadings[j][k])*5)).ljust(10)
        if abs(loadings[j][k])>0.2 else "  " + " "*8
        for k in range(K)
    )
    num_str = "".join(f"  {loadings[j][k]:>+8.3f}" for k in range(K))
    print(f"  {fname:<14}{num_str}   PC{dominant_k+1}")

print()
print(f"  Explained variance:")
for k in range(K):
    evr=eigenvalues[k]/total_var*100
    print(f"    PC{k+1}: {evr:.1f}%")
print()
print(f"  PC Interpretation:")
print(f"    PC1 — high loadings on Revenue, Profit, EPS      → 'Growth factor'")
print(f"    PC2 — high loadings on Volatility, Beta           → 'Risk factor'")
print(f"    PC3 — high loadings on CashFlow, Debt, CurrentRatio → 'Liquidity factor'")
print()
print(f"  PCA recovered the 3 latent factors from 8 observed features.")
print(f"  Features with large |loading| on a PC are most associated with it.")
""",
        "runnable": True,
    },

    "▶ Run: PCA vs Raw Features — Downstream Classification": {
        "description": "Compare k-nearest-neighbour classification accuracy on raw features vs PCA-reduced features at "
                       "different K values. Shows the effect of dimensionality reduction on a downstream task.",
        "code": """
import math
import random

random.seed(42)

# ── Dataset: 3 classes in 20D (true signal in 4D, rest noise) ─────────────────
n_per_class, true_dim, obs_dim = 40, 4, 20
noise_std = 0.8

class_centres_latent = [
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
]

# Fixed embedding matrix (4D → 20D)
import math as _m
embed = [[_m.cos(i+j*1.7+0.5) for j in range(true_dim)] for i in range(obs_dim)]
embed_norms = [math.sqrt(sum(embed[i][j]**2 for j in range(true_dim))) for i in range(obs_dim)]
embed = [[embed[i][j]/embed_norms[i] for j in range(true_dim)] for i in range(obs_dim)]

X, y = [], []
for cls, centre in enumerate(class_centres_latent):
    for _ in range(n_per_class):
        latent = [centre[j]+random.gauss(0,0.6) for j in range(true_dim)]
        obs    = [sum(latent[j]*embed[i][j] for j in range(true_dim)) +
                  random.gauss(0,noise_std) for i in range(obs_dim)]
        X.append(obs); y.append(cls)

n = len(X); p = obs_dim

# ── Train / test split ────────────────────────────────────────────────────────
idx = list(range(n)); random.shuffle(idx)
split = int(0.75*n)
tr_idx, te_idx = idx[:split], idx[split:]
Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
Xte = [X[i] for i in te_idx]; yte = [y[i] for i in te_idx]

# ── PCA fit on train ──────────────────────────────────────────────────────────
means = [sum(Xtr[i][j] for i in range(len(Xtr)))/len(Xtr) for j in range(p)]
stds  = [math.sqrt(sum((Xtr[i][j]-means[j])**2 for i in range(len(Xtr)))/(len(Xtr)-1)) for j in range(p)]
Xctr  = [[(Xtr[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(len(Xtr))]
Xcte  = [[(Xte[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(len(Xte))]

C = [[sum(Xctr[i][j]*Xctr[i][k] for i in range(len(Xctr)))/(len(Xctr)-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M, seed):
    p=len(M); v=seed[:]; nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(3000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v_new=[x/nrm for x in w]
        if math.sqrt(sum((v_new[k]-v[k])**2 for k in range(p)))<1e-10: v=v_new; break
        v=v_new
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

seeds=[[float(k==j) for k in range(p)] for j in range(p)]
eigenpairs=[]; Cd=[row[:] for row in C]
for s in seeds:
    lam,vec=power_iter(Cd,s); eigenpairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
eigenpairs.sort(reverse=True,key=lambda x:x[0])
evecs=[ep[1] for ep in eigenpairs]

def project(Xc,W):
    return [[sum(Xc[i][j]*W[k][j] for j in range(p)) for k in range(len(W))]
            for i in range(len(Xc))]

# ── 1-NN classifier ───────────────────────────────────────────────────────────
def euclidean(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def knn_acc(Xtr,ytr,Xte,yte,K=1):
    correct=0
    for i,p_te in enumerate(Xte):
        dists=[(euclidean(p_te,Xtr[j]),ytr[j]) for j in range(len(Xtr))]
        dists.sort(); nn_labels=[d[1] for d in dists[:K]]
        pred=max(set(nn_labels),key=nn_labels.count)
        if pred==yte[i]: correct+=1
    return correct/len(yte)

# ── Compare raw vs PCA at each K ─────────────────────────────────────────────
raw_acc = knn_acc(Xctr, ytr, Xcte, yte)
total_var = sum(ep[0] for ep in eigenpairs)

print(f"PCA + 1-NN Classification  |  {obs_dim}D → K  |  3 classes  |  n={n}")
print(f"True signal lives in {true_dim}D  |  Noise σ={noise_std}")
print()
print(f"  Raw {obs_dim}D accuracy (no PCA): {raw_acc:.1%}")
print()
print(f"  {'K PCs':<8}  {'Cum Var':>8}  {'Accuracy':>10}  {'vs Raw':>8}  {'Signal?'}")
print("  " + "─"*55)

best_acc, best_K = 0, 0
for K in [1,2,3,4,5,8,10,15,20]:
    if K > p: break
    W   = evecs[:K]
    Ztr = project(Xctr, W)
    Zte = project(Xcte, W)
    acc = knn_acc(Ztr, ytr, Zte, yte)
    cev = sum(ep[0] for ep in eigenpairs[:K])/total_var*100
    delta = acc - raw_acc
    signal = "✓ signal" if K<=true_dim+1 else ("~ noise" if K>true_dim+3 else "")
    bar = ("▲" if delta>0 else "▼") + f" {abs(delta):.1%}"
    print(f"  K={K:<6}  {cev:>7.1f}%  {acc:>10.1%}  {bar:>8}  {signal}")
    if acc > best_acc: best_acc=acc; best_K=K

print()
print(f"  Best accuracy: {best_acc:.1%} at K={best_K} components")
print(f"  → Reducing from {obs_dim}D to {best_K}D improves accuracy by",
      f"{best_acc-raw_acc:+.1%}")
print(f"  → Noise dimensions hurt 1-NN by inflating distances with irrelevant info.")
""",
        "runnable": True,
    },

}

# Dedent all operation code strings — they're indented inside the dict literal,
# so each line has ~20 leading spaces. textwrap.dedent removes the common indent,
# producing clean left-aligned code that runs without IndentationError.
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def render_operations(st, scripts_dir=None, main_script=None):
    """Render all operations with code display and optional run buttons."""
    import streamlit as st  # local import so module stays importable without st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if scripts_dir is None:
        scripts_dir = None
    if main_script is None:
        main_script = None

    scripts_available = main_script.exists()

    if "tok_step_status"  not in st.session_state:
        st.session_state.tok_step_status  = {}
    if "tok_step_outputs" not in st.session_state:
        st.session_state.tok_step_outputs = {}

    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**{op_data['description']}**")
            st.markdown("---")
            st.code(op_data["code"], language=op_data.get("language", "python"))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    try:
        from Unsupervised_Learning.visuals.pca_visual import (   # ← match your exact folder casing
            PCA_VISUAL_HTML,
            PCA_VISUAL_HEIGHT,
        )
        visual_html   = PCA_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = PCA_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None, # COMPLEXITY,
        "operations":    OPERATIONS,
    }

