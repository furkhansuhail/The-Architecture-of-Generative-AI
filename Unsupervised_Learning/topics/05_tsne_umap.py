import textwrap
import re


"""Module: 05 · t-SNE and UMAP"""

TOPIC_NAME   ="t-SNE and UMAP"
DISPLAY_NAME = "05 · t-SNE and UMAP"
ICON         = "🗺️"
SUBTITLE     = "Non-linear dimensionality reduction for visualisation"

THEORY = """
## 05 · t-SNE and UMAP

---

### Why PCA Is Not Enough

PCA is a linear method. It rotates and scales the coordinate axes but cannot
bend, fold, or warp them. If your data lies on a nonlinear manifold — a Swiss
roll, a collection of interleaved spirals, a set of handwritten digit clusters
that form complex curved boundaries in pixel space — PCA will project these
structures onto a flat plane and collapse them into an unreadable blur.

Consider 2000 images of handwritten digits 0–9. In the original 784-dimensional
pixel space, the digit "7" and the digit "1" might be nearby in Euclidean
distance (similar vertical stroke). But they are conceptually — and
perceptually — very different. A good visualisation should pull them apart.

**t-SNE** and **UMAP** solve this by abandoning the idea of finding a global
linear projection. Instead, they ask:

> For each point, who are its neighbours in the high-dimensional space?
> Find a low-dimensional layout that preserves those neighbourhood relationships
> as faithfully as possible.

The result is a 2D (or 3D) map where nearby points in the embedding were also
nearby in the original high-dimensional space — revealing clusters, outliers,
and local structure that PCA cannot.

---

### t-SNE — t-Distributed Stochastic Neighbour Embedding

Developed by Laurens van der Maaten and Geoffrey Hinton (2008). t-SNE is the
most widely used visualisation method in machine learning and computational
biology.

#### Step 1 — High-Dimensional Similarities (Gaussian Affinities)

For every pair of points i, j in the original space, define a conditional
probability that i would pick j as its neighbour:

    p(j|i) = exp(−||xᵢ − xⱼ||² / 2σᵢ²)
             ─────────────────────────────
             Σₖ≠ᵢ exp(−||xᵢ − xₖ||² / 2σᵢ²)

This is a softmax over negative squared distances — it assigns high probability
to close neighbours and near-zero probability to distant points.

The bandwidth σᵢ is chosen **per point** so that the effective number of
neighbours (the perplexity) matches a user-specified target:

    Perplexity = 2^(H(pᵢ))

where H(pᵢ) = −Σⱼ p(j|i) · log₂ p(j|i) is the Shannon entropy of the
conditional distribution. Binary search finds the σᵢ that achieves the target
perplexity.

**What perplexity controls:** It is roughly the expected number of meaningful
neighbours each point has. Typical values: 5–50. Too low → only immediate
neighbours matter (fragmented islands). Too high → the algorithm sees every
point as a neighbour (everything collapses).

Symmetrise the probabilities:

    pᵢⱼ = (p(j|i) + p(i|j)) / (2n)

This ensures all pairs have a well-defined similarity, even asymmetric ones.

---

#### Step 2 — Low-Dimensional Similarities (Student-t Affinities)

In the 2D embedding, define similarities using a **Student-t distribution with
1 degree of freedom** (a Cauchy distribution):

    qᵢⱼ = (1 + ||yᵢ − yⱼ||²)⁻¹
           ──────────────────────
           Σₖ≠ₗ (1 + ||yₖ − yₗ||²)⁻¹

**Why Student-t instead of Gaussian?** This is the key insight of t-SNE.

The Student-t distribution has heavier tails than the Gaussian. In 2D, a
moderate physical separation between yᵢ and yⱼ corresponds to a much smaller
qᵢⱼ than it would under a Gaussian. This means:

- To make qᵢⱼ ≈ pᵢⱼ for nearby points, the embedding must keep them very
  close together → **tight clusters**.
- For distant points where pᵢⱼ ≈ 0, even a moderate |yᵢ − yⱼ| achieves
  qᵢⱼ ≈ 0 → **well-separated clusters with gaps between them**.

This mismatch between Gaussian (high-dim) and Student-t (low-dim) is what
creates the visually crisp cluster separation t-SNE is known for.

---

#### Step 3 — Minimise KL Divergence via Gradient Descent

Treat the embedding coordinates Y = {y₁, ..., yₙ} as parameters to optimise.
The objective is to make the low-dimensional similarities Q match the
high-dimensional similarities P:

    C = KL(P || Q) = Σᵢⱼ pᵢⱼ · log(pᵢⱼ / qᵢⱼ)

KL divergence measures how much information is lost by using Q to approximate
P. It is zero when P = Q exactly and positive otherwise.

Note: KL divergence is **asymmetric**. KL(P||Q) penalises cases where
pᵢⱼ is large but qᵢⱼ is small (nearby high-dim neighbours not close in 2D)
much more heavily than the reverse. This is why t-SNE aggressively pulls
neighbours together — failing to represent a true high-dim neighbour in 2D
is a much larger penalty than artificially separating distant points.

The gradient with respect to yᵢ:

    ∂C/∂yᵢ = 4 · Σⱼ (pᵢⱼ − qᵢⱼ) · (yᵢ − yⱼ) · (1 + ||yᵢ − yⱼ||²)⁻¹

Interpretation: each pair (i,j) exerts a force on yᵢ:
- If pᵢⱼ > qᵢⱼ: they are closer in high-dim than low-dim → attractive force
  pulling yᵢ toward yⱼ.
- If pᵢⱼ < qᵢⱼ: they are farther in high-dim than low-dim → repulsive force
  pushing yᵢ away from yⱼ.

Gradient descent with momentum (and an early exaggeration phase) runs for
typically 250–1000 iterations until the embedding stabilises.

---

#### The Early Exaggeration Trick

In the first ~250 iterations, pᵢⱼ values are multiplied by a factor of 4–12.
This artificially inflates the attractive forces between true neighbours,
causing clusters to form quickly and giving the optimisation a better global
structure before fine-grained adjustments happen. After this phase, the
exaggeration is removed and normal gradient descent continues.

---

#### t-SNE Algorithm Summary

```
Input: high-dim data X, perplexity, n_iter, learning_rate

1. Compute pᵢⱼ (symmetrised Gaussian similarities, bandwidth per point)
2. Initialise Y randomly (or from PCA for reproducibility)
3. Early exaggeration phase (~250 iters):
   a. Multiply all pᵢⱼ by 4
   b. Gradient descent update on Y
4. Normal optimisation phase (~750 iters):
   a. Restore original pᵢⱼ
   b. Continue gradient descent with momentum
5. Return Y (2D or 3D embedding)
```

---

### UMAP — Uniform Manifold Approximation and Projection

Developed by Leland McInnes, John Healy, and James Melville (2018). UMAP is
newer than t-SNE and has largely become the preferred method for two reasons:
it is much faster, and it better preserves global structure.

UMAP is grounded in a sophisticated mathematical framework (Riemannian geometry,
algebraic topology, fuzzy set theory) but its practical algorithm parallels
t-SNE closely. Here is the essential version:

---

#### Step 1 — High-Dimensional Graph (Fuzzy Simplicial Set)

For each point xᵢ, find its k nearest neighbours (k = n_neighbors parameter).
Define the distance to the nearest neighbour as ρᵢ (a local scaling factor):

    ρᵢ = distance to xᵢ's nearest neighbour

The fuzzy affinity from xᵢ to any neighbour xⱼ:

    vᵢⱼ = exp(−max(0, dist(xᵢ,xⱼ) − ρᵢ) / σᵢ)

where σᵢ is chosen so that Σⱼ vᵢⱼ = log₂(k) (analogous to perplexity matching).

Symmetrise:

    wᵢⱼ = vᵢⱼ + vⱼᵢ − vᵢⱼ · vⱼᵢ

This is the fuzzy union — the probability that at least one of the directed
edges i→j or j→i exists in the high-dimensional graph.

The subtraction of the product is a standard formula for combining
probabilities: P(A or B) = P(A) + P(B) − P(A and B).

**Key difference from t-SNE:** UMAP uses k nearest neighbours (a sparse graph),
whereas t-SNE computes all n² pairwise similarities. This is why UMAP scales
to millions of points while t-SNE struggles beyond ~10,000.

---

#### Step 2 — Low-Dimensional Graph (Differentiable Curve Family)

In the low-dimensional embedding, define affinities using a smooth curve that
the optimisation will learn to fit:

    qᵢⱼ = (1 + a · ||yᵢ − yⱼ||^(2b))⁻¹

The parameters a and b are fit to approximate a Student-t distribution for
default settings (a ≈ 1.577, b ≈ 0.895 for min_dist=0.1), but can be varied
to control how tightly points cluster.

**min_dist parameter:** Controls the minimum distance between points in the
embedding. Small min_dist → tight, well-separated clusters. Large min_dist →
points spread more uniformly, preserving more global topology.

---

#### Step 3 — Cross-Entropy Minimisation (Negative Sampling)

UMAP minimises a cross-entropy between the high-dim and low-dim fuzzy graphs:

    C = Σᵢⱼ [wᵢⱼ · log(wᵢⱼ/qᵢⱼ) + (1−wᵢⱼ) · log((1−wᵢⱼ)/(1−qᵢⱼ))]

The first term is the positive term — attract connected neighbours.
The second term is the negative term — repel non-neighbours.

Rather than summing the negative term over all O(n²) pairs, UMAP uses
**negative sampling**: for each positive edge (i,j), sample a small number
of random negative pairs and apply repulsive forces to those. This is the
key computational trick that makes UMAP scale to large datasets.

The gradient is applied via stochastic gradient descent with a decaying
learning rate over typically 200–500 epochs.

---

#### UMAP Algorithm Summary

```
Input: data X, n_neighbors, n_components, min_dist, n_epochs

1. Build k-NN graph (approximate using RP-trees for speed)
2. Compute fuzzy affinities wᵢⱼ (local normalisation with ρᵢ)
3. Initialise Y with spectral embedding (uses Laplacian eigenmaps)
4. For each epoch:
   a. For each positive edge (i,j):
      - Compute gradient to move yᵢ, yⱼ closer
   b. For each positive edge, sample n_neg_samples negative pairs:
      - Compute gradient to push those pairs apart
   c. Apply SGD updates with learning rate schedule
5. Return Y
```

---

### t-SNE vs UMAP: Direct Comparison

| Property                   | t-SNE                            | UMAP                              |
|----------------------------|----------------------------------|-----------------------------------|
| Speed (n=10,000)           | ~minutes                         | ~seconds                          |
| Scalability                | Poor (O(n²) naïve, O(n log n) Barnes-Hut) | Excellent (O(n log n) or better) |
| Preserves local structure  | Excellent                        | Excellent                         |
| Preserves global structure | Poor (cluster distances meaningless) | Better (relative cluster positions meaningful) |
| Deterministic              | No (random init, stochastic)     | No (but spectral init helps)      |
| Interpretable distances    | No                               | Partially                         |
| Out-of-sample projection   | Not natively supported           | Supported (parametric UMAP)       |
| Key parameter              | perplexity (5–50)                | n_neighbors (5–50), min_dist      |
| Mathematical basis         | KL divergence, SNE               | Riemannian geometry, fuzzy sets   |
| Typical use case           | Visualisation of fixed dataset   | Visualisation + downstream tasks  |

---

### Key Parameters and Their Effects

**t-SNE: perplexity**

Perplexity ≈ effective number of neighbours considered per point.

- perplexity = 5:   Only immediate neighbours matter. Many tiny isolated clusters,
                    fragmented structure. Can reveal fine local details.
- perplexity = 30:  Balanced. Standard default. Clusters are stable and interpretable.
- perplexity = 100: Each point considers many neighbours. Clusters merge together.
                    Global structure more visible but local detail blurred.

Rule: perplexity should be less than n/3 and typically between 5–50. For small
datasets (n < 200), use smaller perplexity.

**t-SNE: learning_rate**

Too small → slow convergence, dense ball. Too large → diffuse, noisy layout.
Auto-learning-rate (n/early_exaggeration) is a good default. Typical range: 10–1000.

**t-SNE: n_iter**

More iterations → better convergence. 1000 is usually sufficient. For large
datasets or complex structure, 2000+ may be needed. Monitor the KL divergence
to confirm convergence.

**UMAP: n_neighbors**

Analogous to perplexity. Controls the balance between local and global structure.

- n_neighbors = 5:  Very local. Micro-clusters visible. Global layout unreliable.
- n_neighbors = 15: Good default. Balances local and global.
- n_neighbors = 50: More global structure. Individual clusters may merge.

**UMAP: min_dist**

Controls how tightly points are packed in the embedding.

- min_dist = 0.0:   Points pack into tight clusters. Maximum separation.
- min_dist = 0.1:   Default. Tight clusters with good gap between them.
- min_dist = 0.9:   Points spread out more uniformly. Better topology preservation.

**UMAP: n_components**

UMAP can project to any number of dimensions (2, 3, 10, ...). t-SNE is
typically limited to 2–3 dimensions because the gradient computation
scales poorly to higher dimensions.

---

### Critical Warnings: What t-SNE Does NOT Show

These are the most common misinterpretations of t-SNE plots:

**1. Cluster sizes are meaningless.**
The algorithm compresses all clusters to roughly similar visual sizes regardless
of their true sizes in the original space. A large cluster in t-SNE does not
mean it is large or important in the original space.

**2. Distances between clusters are meaningless.**
The gap between cluster A and cluster B in a t-SNE plot says nothing about
whether those clusters are actually far apart in the original space. Two very
different clusters may appear nearby; two similar ones may appear far.

**3. The number of clusters can be misleading.**
At different perplexity values, one true cluster may split into many visual
islands, or many true clusters may merge into one visual blob. Always run
multiple perplexity values.

**4. Topology is not preserved globally.**
t-SNE is designed only to preserve local neighbourhood structure. Any
interpretation of the global layout is unreliable. UMAP is somewhat better
on this, but still not perfect.

**5. Crowding problem.**
In very high dimensions, points can have many equidistant neighbours — they
exist on a high-dimensional sphere. When compressed to 2D, the centre of the
embedding gets "crowded" with nearby points. The Student-t distribution
partially alleviates this, but it does not eliminate it.

**Correct uses of t-SNE/UMAP:**
- Does the data form distinct clusters? (yes/no, but don't count them)
- Do pre-labelled classes separate in the embedding? (class separability)
- Are there obvious outliers?
- Does there appear to be a continuous gradient or manifold structure?

---

### Barnes-Hut t-SNE (Approximate, O(n log n))

The standard t-SNE gradient requires summing over all n² pairs — O(n²) per
iteration, which is prohibitive for n > 5000.

Barnes-Hut t-SNE uses a **quadtree / octree** to approximate the repulsive
forces. Distant groups of points are treated as a single super-point at their
centre of mass. If the Barnes-Hut opening angle θ is small enough (θ < 0.5),
the approximation introduces negligible error while reducing the repulsive
force computation to O(n log n) per iteration.

This makes t-SNE feasible for n ~ 100,000 points.

For n > 1,000,000, UMAP with approximate nearest neighbours (using random
projection trees) is the practical choice.

---

### When to Use Each Method

**Use PCA when:**
- You need a fast, interpretable linear baseline.
- Downstream tasks (regression, classification) will use the reduced features.
- Explained variance ratios need to be reported.
- n is very large and speed matters.

**Use t-SNE when:**
- Pure visualisation of a fixed dataset (n < ~50,000).
- You want the sharpest, most visually crisp cluster separation.
- You are exploring cluster structure for the first time.

**Use UMAP when:**
- n is large (>10,000) and speed matters.
- You need to project new (out-of-sample) points (parametric UMAP).
- You want some global structure preserved alongside local clusters.
- You are using the reduced representation as features for a downstream model.
- You need to reduce to more than 2–3 dimensions.

---

### Key Takeaways

1. t-SNE and UMAP are nonlinear dimensionality reduction methods. They
   preserve local neighbourhood structure in high dimensions and reveal
   cluster and manifold geometry that linear PCA cannot.

2. t-SNE defines high-dim similarities with Gaussians (per-point bandwidth
   set by perplexity) and low-dim similarities with the heavier-tailed
   Student-t distribution. This mismatch creates the tight-cluster, wide-gap
   visual style via KL divergence minimisation.

3. UMAP builds a sparse k-NN fuzzy graph in high dimensions and matches it in
   low dimensions using cross-entropy + negative sampling. This makes it far
   faster than t-SNE and more scalable.

4. Neither method preserves inter-cluster distances. Cluster size and the gaps
   between clusters in the 2D plot have no reliable meaning. Use these
   methods for exploration, not for measuring distances.

5. Always run multiple parameter values (perplexity / n_neighbors). A single
   t-SNE or UMAP plot can be misleading. Stable structure across parameter
   ranges is more trustworthy.
"""


OPERATIONS = {

    "▶ Run: t-SNE From Scratch (Core Algorithm)": {
        "description": "Full t-SNE implementation from first principles — Gaussian affinities with perplexity matching via binary search, Student-t low-dim similarities, KL divergence objective, and gradient descent with momentum. Runs on a small synthetic dataset.",
        "code": """
import math
import random

random.seed(42)

# ── Generate 3-cluster data in 4D ─────────────────────────────────────────────
def blob(cx, n, dim, spread=0.4):
    return [[cx[d] + random.gauss(0, spread) for d in range(dim)] for _ in range(n)]

DIM  = 4
data = (blob([2,2,0,0], 20, DIM)
      + blob([-2,0,2,0], 20, DIM)
      + blob([0,-2,-2,2], 20, DIM))
n    = len(data)
true_labels = [0]*20 + [1]*20 + [2]*20

print(f"t-SNE from scratch  |  n={n}, input_dim={DIM}, output_dim=2")
print()

# ── Euclidean distances squared ───────────────────────────────────────────────
def sq_dist(a, b):
    return sum((a[k]-b[k])**2 for k in range(len(a)))

D2 = [[sq_dist(data[i], data[j]) for j in range(n)] for i in range(n)]

# ── Step 1: Perplexity-based Gaussian affinities ──────────────────────────────
PERPLEXITY = 10.0

def compute_pij(D2, perplexity):
    n = len(D2)
    P = [[0.0]*n for _ in range(n)]
    for i in range(n):
        # Binary search for sigma_i
        lo, hi = 0.0, 1e10
        beta = 1.0  # beta = 1/(2*sigma^2)
        for _ in range(50):
            # Conditional probabilities
            exp_d = [math.exp(-D2[i][j]*beta) if j!=i else 0.0 for j in range(n)]
            sum_exp = sum(exp_d) or 1e-10
            p_cond  = [e/sum_exp for e in exp_d]
            # Entropy
            H = -sum(p*math.log(p+1e-10) for p in p_cond if p>0)
            perp_i = math.exp(H)
            diff = perp_i - perplexity
            if abs(diff) < 1e-5: break
            if diff > 0: lo = beta; beta = (beta+hi)/2 if hi<1e9 else beta*2
            else:        hi = beta; beta = (beta+lo)/2
        P[i] = p_cond
    # Symmetrise: p_ij = (p(j|i) + p(i|j)) / 2n
    Psym = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            Psym[i][j] = (P[i][j] + P[j][i]) / (2*n)
    return Psym

print("Step 1: Computing Gaussian affinities (perplexity matching)...")
Pij = compute_pij(D2, PERPLEXITY)
print(f"  Done. Mean p_ij (off-diagonal): {sum(Pij[i][j] for i in range(n) for j in range(n) if i!=j)/(n*(n-1)):.6f}")
print()

# ── Step 2: Initialise embedding ──────────────────────────────────────────────
Y = [[random.gauss(0, 0.01) for _ in range(2)] for _ in range(n)]

# ── Step 3: Gradient descent ──────────────────────────────────────────────────
LR            = 200.0
MOMENTUM      = 0.8
N_ITER        = 300
EARLY_EXAG    = 4.0
EXAG_ITERS    = 100

gains    = [[1.0]*2 for _ in range(n)]
velocity = [[0.0]*2 for _ in range(n)]

def compute_qij_and_grad(Y, Pij, exag=1.0):
    n = len(Y)
    # q_ij = (1 + ||y_i - y_j||^2)^-1  (unnormalised)
    q_num = [[0.0]*n for _ in range(n)]
    q_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                d2 = sum((Y[i][k]-Y[j][k])**2 for k in range(2))
                q_num[i][j] = 1.0/(1.0 + d2)
                q_sum += q_num[i][j]
    q_sum = max(q_sum, 1e-10)
    Qij = [[q_num[i][j]/q_sum for j in range(n)] for i in range(n)]

    # KL divergence
    kl = sum(Pij[i][j]*exag * math.log((Pij[i][j]*exag)/max(Qij[i][j],1e-10))
             for i in range(n) for j in range(n) if i!=j and Pij[i][j]>1e-12)

    # Gradient: dC/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1+||y_i-y_j||^2)^-1
    grad = [[0.0]*2 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            pq_diff = Pij[i][j]*exag - Qij[i][j]
            factor  = 4.0 * pq_diff * q_num[i][j]
            for d in range(2):
                grad[i][d] += factor * (Y[i][d] - Y[j][d])
    return Qij, grad, kl

print("Step 2: Gradient descent (300 iterations)...")
print(f"  {'Iter':>6}  {'KL div':>10}  {'Phase'}")
print("  " + "─"*35)

for t in range(1, N_ITER+1):
    exag    = EARLY_EXAG if t <= EXAG_ITERS else 1.0
    Qij, grad, kl = compute_qij_and_grad(Y, Pij, exag)

    # Adaptive gains (sign-agreement check)
    for i in range(n):
        for d in range(2):
            same_sign = (grad[i][d] > 0) == (velocity[i][d] > 0)
            gains[i][d] = gains[i][d]*0.8 if same_sign else gains[i][d]+0.2
            gains[i][d] = max(gains[i][d], 0.01)

    # Momentum update
    for i in range(n):
        for d in range(2):
            velocity[i][d] = MOMENTUM*velocity[i][d] - LR*gains[i][d]*grad[i][d]
            Y[i][d]        += velocity[i][d]

    # Centre embedding
    for d in range(2):
        m = sum(Y[i][d] for i in range(n))/n
        for i in range(n): Y[i][d] -= m

    if t in (1, 50, 100, 150, 200, 250, 300) or t == EXAG_ITERS:
        phase = "Early exaggeration" if t <= EXAG_ITERS else "Normal"
        print(f"  {t:>6}  {kl:>10.4f}  {phase}")

print()

# ── Evaluate: are same-class points close in embedding? ──────────────────────
def mean_intra_dist(Y, labels):
    n = len(Y)
    classes = sorted(set(labels))
    intra = []
    for c in classes:
        members = [i for i in range(n) if labels[i]==c]
        dists   = [math.sqrt(sum((Y[members[a]][d]-Y[members[b]][d])**2 for d in range(2)))
                   for a in range(len(members)) for b in range(a+1,len(members))]
        intra.append(sum(dists)/len(dists) if dists else 0)
    return intra

def mean_inter_dist(Y, labels):
    classes = sorted(set(labels))
    n = len(Y)
    inter = []
    for ci in range(len(classes)):
        for cj in range(ci+1, len(classes)):
            mi = [i for i in range(n) if labels[i]==classes[ci]]
            mj = [i for i in range(n) if labels[i]==classes[cj]]
            dists=[math.sqrt(sum((Y[a][d]-Y[b][d])**2 for d in range(2)))
                   for a in mi for b in mj]
            inter.append(sum(dists)/len(dists) if dists else 0)
    return inter

intra = mean_intra_dist(Y, true_labels)
inter = mean_inter_dist(Y, true_labels)

print("Embedding quality:")
print(f"  Mean intra-cluster distance: {sum(intra)/len(intra):.3f}")
print(f"  Mean inter-cluster distance: {sum(inter)/len(inter):.3f}")
ratio = (sum(inter)/len(inter)) / (sum(intra)/len(intra))
print(f"  Inter/Intra ratio           : {ratio:.2f}x  (higher = better separation)")
print()

# ── ASCII scatter of final embedding ─────────────────────────────────────────
print("2D Embedding (ASCII scatter):")
W, H = 60, 24
xs = [Y[i][0] for i in range(n)]
ys = [Y[i][1] for i in range(n)]
mnx, mxx = min(xs), max(xs)
mny, mxy = min(ys), max(ys)
grid = [['·']*W for _ in range(H)]
SYMBOLS = ['O', '#', '*']
for i in range(n):
    col = int((xs[i]-mnx)/(mxx-mnx+1e-9)*(W-1))
    row = H-1-int((ys[i]-mny)/(mxy-mny+1e-9)*(H-1))
    grid[row][col] = SYMBOLS[true_labels[i]]
print(f"  ┌{'─'*W}┐")
for row in grid: print(f"  │{''.join(row)}│")
print(f"  └{'─'*W}┘")
print(f"  O=class0  #=class1  *=class2")
""",
        "runnable": True,
    },

    "▶ Run: Perplexity Effect on t-SNE": {
        "description": "Run t-SNE with four different perplexity values on the same dataset. Print cluster separation metrics to show how perplexity controls local vs global structure.",
        "code": """
import math
import random

random.seed(99)

# ── Dataset: 4 clusters in 6D ─────────────────────────────────────────────────
def blob(cx, n, spread=0.5):
    dim = len(cx)
    return [[cx[d]+random.gauss(0,spread) for d in range(dim)] for _ in range(n)]

DIM  = 6
data = (blob([3,3,0,0,0,0], 15, 0.4)
      + blob([-3,0,3,0,0,0], 15, 0.4)
      + blob([0,-3,-3,0,0,0], 15, 0.4)
      + blob([0,0,0,3,3,0],  15, 0.4))
n    = len(data)
labels = [0]*15 + [1]*15 + [2]*15 + [3]*15

def sq_dist(a, b): return sum((a[k]-b[k])**2 for k in range(len(a)))
D2 = [[sq_dist(data[i], data[j]) for j in range(n)] for i in range(n)]

def compute_pij(D2, perplexity):
    n   = len(D2)
    P   = [[0.0]*n for _ in range(n)]
    for i in range(n):
        lo, hi, beta = 0.0, 1e10, 1.0
        for _ in range(50):
            ed   = [math.exp(-D2[i][j]*beta) if j!=i else 0.0 for j in range(n)]
            s    = sum(ed) or 1e-10
            pc   = [e/s for e in ed]
            H    = -sum(p*math.log(p+1e-10) for p in pc if p>0)
            diff = math.exp(H) - perplexity
            if abs(diff)<1e-5: break
            if diff>0: lo=beta; beta=(beta+hi)/2 if hi<1e9 else beta*2
            else:      hi=beta; beta=(beta+lo)/2
        P[i] = pc
    Ps = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            Ps[i][j] = (P[i][j]+P[j][i])/(2*n)
    return Ps

def run_tsne(D2, Pij, n_iter=200, lr=150.0):
    n       = len(D2)
    Y       = [[random.gauss(0,0.01) for _ in range(2)] for _ in range(n)]
    vel     = [[0.0]*2 for _ in range(n)]
    gains   = [[1.0]*2 for _ in range(n)]
    mom     = 0.8
    for t in range(1, n_iter+1):
        exag = 4.0 if t<=80 else 1.0
        q_num = [[0.0]*n for _ in range(n)]
        q_sum = 1e-10
        for i in range(n):
            for j in range(n):
                if i!=j:
                    d2 = sum((Y[i][k]-Y[j][k])**2 for k in range(2))
                    q_num[i][j]=1.0/(1.0+d2); q_sum+=q_num[i][j]
        grad = [[0.0]*2 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i==j: continue
                f = 4*(Pij[i][j]*exag - q_num[i][j]/q_sum)*q_num[i][j]
                for d in range(2): grad[i][d]+=f*(Y[i][d]-Y[j][d])
        for i in range(n):
            for d in range(2):
                same=(grad[i][d]>0)==(vel[i][d]>0)
                gains[i][d]=gains[i][d]*0.8 if same else gains[i][d]+0.2
                gains[i][d]=max(gains[i][d],0.01)
                vel[i][d]=mom*vel[i][d]-lr*gains[i][d]*grad[i][d]
                Y[i][d]+=vel[i][d]
        for d in range(2):
            m=sum(Y[i][d] for i in range(n))/n
            for i in range(n): Y[i][d]-=m
    return Y

def silhouette(Y, labels):
    n = len(Y)
    def d(a,b): return math.sqrt(sum((Y[a][k]-Y[b][k])**2 for k in range(2)))
    scores=[]
    classes=sorted(set(labels))
    for i in range(n):
        same=[j for j in range(n) if j!=i and labels[j]==labels[i]]
        diff_classes={c:[j for j in range(n) if labels[j]==c] for c in classes if c!=labels[i]}
        if not same: continue
        a=sum(d(i,j) for j in same)/len(same)
        b=min(sum(d(i,j) for j in ms)/len(ms) for ms in diff_classes.values() if ms)
        scores.append((b-a)/max(a,b) if max(a,b)>0 else 0)
    return sum(scores)/len(scores) if scores else 0

# ── Run for 4 perplexity values ────────────────────────────────────────────────
perplexities = [5, 15, 30, 50]
print(f"Perplexity Effect on t-SNE  |  n={n}, {DIM}D → 2D  |  4 clusters")
print(f"Silhouette score range: -1 (bad) to +1 (perfect separation)")
print()
print(f"  {'Perplexity':>12}  {'Silhouette':>12}  {'Intra dist':>12}  {'Inter dist':>12}  Verdict")
print("  " + "─"*68)

for perp in perplexities:
    Pij = compute_pij(D2, perp)
    Y   = run_tsne(D2, Pij, n_iter=200)
    sil = silhouette(Y, labels)

    classes = sorted(set(labels))
    intra_dists=[]
    for c in classes:
        ms=[i for i in range(n) if labels[i]==c]
        dists=[math.sqrt(sum((Y[a][k]-Y[b][k])**2 for k in range(2)))
               for a in range(len(ms)) for b in range(a+1,len(ms))]
        intra_dists.append(sum(dists)/len(dists) if dists else 0)
    inter_dists=[]
    for ci in range(len(classes)):
        for cj in range(ci+1,len(classes)):
            ma=[i for i in range(n) if labels[i]==classes[ci]]
            mb=[i for i in range(n) if labels[i]==classes[cj]]
            dists=[math.sqrt(sum((Y[a][k]-Y[b][k])**2 for k in range(2)))
                   for a in ma for b in mb]
            inter_dists.append(sum(dists)/len(dists) if dists else 0)

    intra=sum(intra_dists)/len(intra_dists)
    inter=sum(inter_dists)/len(inter_dists)

    if sil > 0.5:   verdict = "✓ Good separation"
    elif sil > 0.2: verdict = "~ Moderate"
    else:           verdict = "✗ Poor / fragmented"

    print(f"  {perp:>12}  {sil:>12.3f}  {intra:>12.3f}  {inter:>12.3f}  {verdict}")

print()
print("Notes:")
print("  Low perplexity (5)  → local structure only, may fragment clusters")
print("  High perplexity (50)→ sees more of the global picture, may merge clusters")
print("  Optimal perplexity is typically between 10–30 for this dataset size")
""",
        "runnable": True,
    },

    "▶ Run: UMAP From Scratch (Core Algorithm)": {
        "description": "UMAP core algorithm from first principles — k-NN graph, fuzzy affinities with local normalisation, cross-entropy with negative sampling, SGD embedding. Compares to t-SNE on the same data.",
        "code": """
import math
import random

random.seed(7)

# ── Dataset: 3 clusters in 5D ─────────────────────────────────────────────────
def blob(cx, n, spread=0.5):
    return [[cx[d]+random.gauss(0,spread) for d in range(len(cx))] for _ in range(n)]

data = (blob([3,3,0,0,0], 25, 0.5)
      + blob([-3,0,3,0,0], 25, 0.5)
      + blob([0,-3,-3,0,0], 25, 0.5))
n    = len(data)
labels = [0]*25 + [1]*25 + [2]*25

def euclidean(a, b): return math.sqrt(sum((a[k]-b[k])**2 for k in range(len(a))))

print(f"UMAP from scratch  |  n={n}, input_dim=5, output_dim=2")
print()

# ── Step 1: k-NN graph ────────────────────────────────────────────────────────
K_NEIGHBORS = 10

def build_knn(data, k):
    n = len(data)
    knn = []
    for i in range(n):
        dists = sorted([(euclidean(data[i], data[j]), j) for j in range(n) if j!=i])
        knn.append([idx for _, idx in dists[:k]])
    return knn

knn = build_knn(data, K_NEIGHBORS)
print(f"Step 1: Built {K_NEIGHBORS}-NN graph for {n} points")

# ── Step 2: Fuzzy affinities ──────────────────────────────────────────────────
# rho_i = distance to nearest neighbour (local scaling)
# sigma_i chosen so that sum(v_ij) = log2(k)

def compute_fuzzy_affinities(data, knn, k):
    n = len(data)
    target_sum = math.log2(k)
    W = {}  # sparse: (i,j) -> weight

    for i in range(n):
        # rho_i = dist to nearest neighbour
        rho_i = euclidean(data[i], data[knn[i][0]])

        # Binary search for sigma_i
        lo, hi, sigma = 0.0, 1e10, 1.0
        for _ in range(64):
            v = [math.exp(-max(0.0, euclidean(data[i], data[j]) - rho_i)/sigma)
                 for j in knn[i]]
            diff = sum(v) - target_sum
            if abs(diff) < 1e-5: break
            if diff > 0: lo = sigma; sigma = (sigma+hi)/2 if hi<1e9 else sigma*2
            else:        hi = sigma; sigma = (sigma+lo)/2

        for j, vij in zip(knn[i], v):
            W[(i,j)] = vij

    # Symmetrise: w_ij = v_ij + v_ji - v_ij * v_ji
    Wsym = {}
    all_pairs = set(W.keys()) | {(j,i) for (i,j) in W.keys()}
    for (i,j) in all_pairs:
        if i >= j: continue
        vij = W.get((i,j), 0.0)
        vji = W.get((j,i), 0.0)
        w   = vij + vji - vij*vji
        if w > 1e-6:
            Wsym[(i,j)] = w
            Wsym[(j,i)] = w
    return Wsym

W = compute_fuzzy_affinities(data, knn, K_NEIGHBORS)
positive_edges = [(i,j,w) for (i,j),w in W.items() if i<j]
print(f"Step 2: Fuzzy affinity graph — {len(positive_edges)} positive edges")
print(f"  Mean weight: {sum(w for _,_,w in positive_edges)/len(positive_edges):.4f}")
print()

# ── Step 3: Initialise embedding (random) ─────────────────────────────────────
Y = [[random.gauss(0, 0.1) for _ in range(2)] for _ in range(n)]

# ── Step 4: SGD with negative sampling ────────────────────────────────────────
A, B = 1.577, 0.895   # default UMAP curve params for min_dist=0.1
N_EPOCHS   = 200
LR         = 1.0
NEG_SAMPLES = 5

def q_low(d2, a=A, b=B):
    return 1.0 / (1.0 + a*(d2**b))

def grad_pos(yi, yj):
    d2  = sum((yi[k]-yj[k])**2 for k in range(2)) + 1e-8
    q   = q_low(d2)
    dq  = -A*B*(d2**(B-1))*(q**2)*2        # ← A and B uppercase
    fac = -2.0*dq/((q+1e-8)*d2)
    return [fac*(yi[k]-yj[k]) for k in range(2)]

def grad_neg(yi, yj):
    d2  = sum((yi[k]-yj[k])**2 for k in range(2)) + 1e-8
    q   = q_low(d2)
    dq  = -A*B*(d2**(B-1))*(q**2)*2        # ← A and B uppercase
    fac = 2.0*(1-q+1e-8)*(-dq)/((1-q+1e-8)*(q+1e-8)*d2)
    fac = min(fac, 4.0)
    return [fac*(yi[k]-yj[k]) for k in range(2)]

print(f"Step 3: SGD optimisation ({N_EPOCHS} epochs, {NEG_SAMPLES} negative samples/edge)")
print(f"  {'Epoch':>6}  {'LR':>8}  {'Pos edges processed'}")
print("  " + "─"*38)

for epoch in range(1, N_EPOCHS+1):
    lr_e = LR * (1.0 - (epoch-1)/N_EPOCHS)
    random.shuffle(positive_edges)

    for (i, j, w) in positive_edges:
        # Positive gradient
        gp = grad_pos(Y[i], Y[j])
        for d in range(2):
            Y[i][d] -= lr_e * gp[d] * w
            Y[j][d] += lr_e * gp[d] * w

        # Negative samples
        for _ in range(NEG_SAMPLES):
            k = random.randint(0, n-1)
            if k == i or k == j: continue
            gn = grad_neg(Y[i], Y[k])
            for d in range(2):
                Y[i][d] -= lr_e * gn[d]

    if epoch in (1, 50, 100, 150, 200):
        print(f"  {epoch:>6}  {lr_e:>8.4f}  {len(positive_edges)} edges")

# ── Evaluate ──────────────────────────────────────────────────────────────────
def silhouette(Y, labels):
    n   = len(Y)
    def d(a,b): return math.sqrt(sum((Y[a][k]-Y[b][k])**2 for k in range(2)))
    cls = sorted(set(labels)); scores=[]
    for i in range(n):
        same=[j for j in range(n) if j!=i and labels[j]==labels[i]]
        diffs={c:[j for j in range(n) if labels[j]==c] for c in cls if c!=labels[i]}
        if not same: continue
        a=sum(d(i,j) for j in same)/len(same)
        b=min(sum(d(i,j) for j in ms)/len(ms) for ms in diffs.values() if ms)
        scores.append((b-a)/max(a,b) if max(a,b)>0 else 0)
    return sum(scores)/len(scores) if scores else 0

sil = silhouette(Y, labels)
print()
print(f"Embedding quality:")
print(f"  Silhouette score: {sil:.3f}  (target: > 0.5 for clean separation)")
print()

# ── ASCII scatter ─────────────────────────────────────────────────────────────
print("2D UMAP Embedding:")
W2, H2 = 60, 20
xs=[Y[i][0] for i in range(n)]; ys=[Y[i][1] for i in range(n)]
mnx,mxx=min(xs),max(xs); mny,mxy=min(ys),max(ys)
grid=[['·']*W2 for _ in range(H2)]
SYM=['O','#','*']
for i in range(n):
    col=int((xs[i]-mnx)/(mxx-mnx+1e-9)*(W2-1))
    row=H2-1-int((ys[i]-mny)/(mxy-mny+1e-9)*(H2-1))
    grid[row][col]=SYM[labels[i]]
print(f"  ┌{'─'*W2}┐")
for row in grid: print(f"  │{''.join(row)}│")
print(f"  └{'─'*W2}┘")
print(f"  O=class0  #=class1  *=class2")
""",
        "runnable": True,
    },

    "▶ Run: t-SNE vs UMAP vs PCA on Same Data": {
        "description": "Head-to-head comparison of PCA, t-SNE, and UMAP on the same 4-cluster dataset. Reports silhouette scores and inter/intra-cluster distance ratios for each method.",
        "code": """
import math
import random

random.seed(21)

# ── Dataset: 4 clusters in 8D with varying densities ─────────────────────────
def blob(cx, n, spread):
    return [[cx[d]+random.gauss(0,spread) for d in range(len(cx))] for _ in range(n)]

data = (blob([4,4,0,0,0,0,0,0], 20, 0.4)    # tight cluster
      + blob([-4,0,4,0,0,0,0,0], 20, 0.4)
      + blob([0,-4,-4,0,0,0,0,0], 20, 0.7)   # looser cluster
      + blob([0,0,0,4,0,4,0,0],  20, 0.4))
n      = len(data)
labels = [0]*20 + [1]*20 + [2]*20 + [3]*20
p      = len(data[0])

# ────────────────────────────────────────────────────────────────────
# PCA (2 components)
# ────────────────────────────────────────────────────────────────────
means = [sum(data[i][j] for i in range(n))/n for j in range(p)]
Xc    = [[data[i][j]-means[j] for j in range(p)] for i in range(n)]
C     = [[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M, seed):
    p=len(M); v=seed[:]; nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(3000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v2=[x/nrm for x in w]
        if math.sqrt(sum((v2[k]-v[k])**2 for k in range(p)))<1e-10: v=v2; break
        v=v2
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

seeds=[[float(k==j) for k in range(p)] for j in range(p)]
epairs=[]; Cd=[row[:] for row in C]
for s in seeds:
    lam,vec=power_iter(Cd,s); epairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
epairs.sort(reverse=True,key=lambda x:x[0])
W_pca=[ep[1] for ep in epairs[:2]]
Y_pca=[[sum(Xc[i][j]*W_pca[k][j] for j in range(p)) for k in range(2)] for i in range(n)]

# ────────────────────────────────────────────────────────────────────
# t-SNE (mini, 150 iters)
# ────────────────────────────────────────────────────────────────────
def sq_dist(a,b): return sum((a[k]-b[k])**2 for k in range(len(a)))
D2=[[sq_dist(data[i],data[j]) for j in range(n)] for i in range(n)]

def pij_fast(D2, perp=15):
    n=len(D2); P=[[0.0]*n for _ in range(n)]
    for i in range(n):
        lo,hi,beta=0.0,1e10,1.0
        for _ in range(50):
            ed=[math.exp(-D2[i][j]*beta) if j!=i else 0.0 for j in range(n)]
            s=sum(ed) or 1e-10; pc=[e/s for e in ed]
            H=-sum(p*math.log(p+1e-10) for p in pc if p>0)
            diff=math.exp(H)-perp
            if abs(diff)<1e-5: break
            if diff>0: lo=beta; beta=(beta+hi)/2 if hi<1e9 else beta*2
            else:      hi=beta; beta=(beta+lo)/2
        P[i]=pc
    Ps=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n): Ps[i][j]=(P[i][j]+P[j][i])/(2*n)
    return Ps

def run_tsne_fast(D2, Pij, n_iter=150, lr=150.0):
    n=len(D2); Y=[[random.gauss(0,0.01) for _ in range(2)] for _ in range(n)]
    vel=[[0.0]*2 for _ in range(n)]; gains=[[1.0]*2 for _ in range(n)]
    for t in range(1,n_iter+1):
        exag=4.0 if t<=60 else 1.0
        q_num=[[0.0]*n for _ in range(n)]; qs=1e-10
        for i in range(n):
            for j in range(n):
                if i!=j:
                    d2=sum((Y[i][k]-Y[j][k])**2 for k in range(2))
                    q_num[i][j]=1.0/(1.0+d2); qs+=q_num[i][j]
        grad=[[0.0]*2 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i==j: continue
                f=4*(Pij[i][j]*exag-q_num[i][j]/qs)*q_num[i][j]
                for d in range(2): grad[i][d]+=f*(Y[i][d]-Y[j][d])
        for i in range(n):
            for d in range(2):
                same=(grad[i][d]>0)==(vel[i][d]>0)
                gains[i][d]=gains[i][d]*0.8 if same else gains[i][d]+0.2
                gains[i][d]=max(gains[i][d],0.01)
                vel[i][d]=0.8*vel[i][d]-lr*gains[i][d]*grad[i][d]
                Y[i][d]+=vel[i][d]
        for d in range(2):
            m=sum(Y[i][d] for i in range(n))/n
            for i in range(n): Y[i][d]-=m
    return Y

Pij=pij_fast(D2, perp=15)
Y_tsne=run_tsne_fast(D2, Pij, n_iter=150)

# ────────────────────────────────────────────────────────────────────
# UMAP (mini, 100 epochs)
# ────────────────────────────────────────────────────────────────────
def euclidean(a,b): return math.sqrt(sum((a[k]-b[k])**2 for k in range(len(a))))
K=10
knn=[[idx for _,idx in sorted([(euclidean(data[i],data[j]),j) for j in range(n) if j!=i])[:K]]
     for i in range(n)]

target=math.log2(K); W_umap={}
for i in range(n):
    rho=euclidean(data[i],data[knn[i][0]])
    lo,hi,sigma=0.0,1e10,1.0
    for _ in range(64):
        v=[math.exp(-max(0.0,euclidean(data[i],data[j])-rho)/sigma) for j in knn[i]]
        diff=sum(v)-target
        if abs(diff)<1e-5: break
        if diff>0: lo=sigma; sigma=(sigma+hi)/2 if hi<1e9 else sigma*2
        else:      hi=sigma; sigma=(sigma+lo)/2
    for j,vij in zip(knn[i],v): W_umap[(i,j)]=vij

edges=[]
all_p=set(W_umap)|{(j,i) for (i,j) in W_umap}
for (i,j) in all_p:
    if i>=j: continue
    vij=W_umap.get((i,j),0); vji=W_umap.get((j,i),0)
    w=vij+vji-vij*vji
    if w>1e-6: edges.append((i,j,w))

Y_umap=[[random.gauss(0,0.1) for _ in range(2)] for _ in range(n)]
A,B=1.577,0.895
for epoch in range(1,101):
    lr_e=1.0*(1-(epoch-1)/100); random.shuffle(edges)
    for (i,j,w) in edges:
        d2=sum((Y_umap[i][k]-Y_umap[j][k])**2 for k in range(2))+1e-8
        q=1/(1+A*d2**B)
        dq=-A*B*(d2**(B-1))*(q**2)*2
        fp=-2*dq/((q+1e-8)*d2)
        for d in range(2):
            dy=lr_e*fp*(Y_umap[i][d]-Y_umap[j][d])*w
            Y_umap[i][d]-=dy; Y_umap[j][d]+=dy
        for _ in range(5):
            k2=random.randint(0,n-1)
            if k2 in (i,j): continue
            d2n=sum((Y_umap[i][k]-Y_umap[k2][k])**2 for k in range(2))+1e-8
            qn=1/(1+A*d2n**B)
            fn=min(2*(1-qn)*A*B*(d2n**(B-1))*(qn**2)*2/((1-qn+1e-8)*(qn+1e-8)*d2n),4.0)
            for d in range(2): Y_umap[i][d]-=lr_e*fn*(Y_umap[i][d]-Y_umap[k2][d])

# ── Evaluation: silhouette ────────────────────────────────────────────────────
def silhouette(Y, labels):
    n=len(Y); cls=sorted(set(labels)); scores=[]
    def d(a,b): return math.sqrt(sum((Y[a][k]-Y[b][k])**2 for k in range(2)))
    for i in range(n):
        same=[j for j in range(n) if j!=i and labels[j]==labels[i]]
        diffs={c:[j for j in range(n) if labels[j]==c] for c in cls if c!=labels[i]}
        if not same: continue
        a_=sum(d(i,j) for j in same)/len(same)
        b_=min(sum(d(i,j) for j in ms)/len(ms) for ms in diffs.values() if ms)
        scores.append((b_-a_)/max(a_,b_) if max(a_,b_)>0 else 0)
    return sum(scores)/len(scores) if scores else 0

methods = [('PCA',  Y_pca),  ('t-SNE', Y_tsne), ('UMAP', Y_umap)]

print(f"PCA vs t-SNE vs UMAP  |  n={n}, 8D → 2D  |  4 clusters (mixed density)")
print()
print(f"  {'Method':<10}  {'Silhouette':>12}  {'Assessment'}")
print("  " + "─"*45)
for name, Y in methods:
    sil = silhouette(Y, labels)
    if sil > 0.6:    assess = "✓ Excellent — clear separation"
    elif sil > 0.4:  assess = "✓ Good"
    elif sil > 0.2:  assess = "~ Moderate"
    else:            assess = "✗ Poor — clusters overlap"
    print(f"  {name:<10}  {sil:>12.3f}  {assess}")

print()
print("Expected finding:")
print("  PCA   — fast, linear, may not separate nonlinear structure well")
print("  t-SNE — sharp local cluster separation, ignores global layout")
print("  UMAP  — good balance of local + global structure, faster than t-SNE")
""",
        "runnable": True,
    },

    "▶ Run: When Cluster Distances Are Misleading": {
        "description": "Demonstrate t-SNE's key warning: distances between clusters in the 2D plot are not meaningful. Two datasets with very different true inter-cluster distances are shown to produce similar-looking t-SNE plots.",
        "code": """
import math
import random

random.seed(55)

# ── Two datasets: same local structure, very different global scale ────────────
# Dataset A: 3 tight clusters very far apart in 4D
# Dataset B: 3 tight clusters very close together in 4D

def blob(cx, n, spread):
    return [[cx[d]+random.gauss(0,spread) for d in range(len(cx))] for _ in range(n)]

# Dataset A: clusters separated by 10 units
dataA = (blob([10, 0, 0, 0], 20, 0.4)
       + blob([-10, 0, 0, 0], 20, 0.4)
       + blob([0, 10, 0, 0],  20, 0.4))

# Dataset B: clusters separated by 1.5 units
dataB = (blob([1.5, 0, 0, 0], 20, 0.4)
       + blob([-1.5, 0, 0, 0], 20, 0.4)
       + blob([0, 1.5, 0, 0],  20, 0.4))

n = 60; labels = [0]*20+[1]*20+[2]*20

# True inter-cluster distance for each dataset
def true_inter(data, labels):
    classes = sorted(set(labels))
    dists=[]
    for ci in range(len(classes)):
        for cj in range(ci+1,len(classes)):
            ma=[i for i in range(n) if labels[i]==classes[ci]]
            mb=[i for i in range(n) if labels[i]==classes[cj]]
            # centroid distance
            ca=[sum(data[i][d] for i in ma)/len(ma) for d in range(len(data[0]))]
            cb=[sum(data[i][d] for i in mb)/len(mb) for d in range(len(data[0]))]
            dists.append(math.sqrt(sum((ca[d]-cb[d])**2 for d in range(len(data[0])))))
    return sum(dists)/len(dists)

# ── Run t-SNE on both ─────────────────────────────────────────────────────────
def sq_dist(a,b): return sum((a[k]-b[k])**2 for k in range(len(a)))

def run_tsne(data, perp=10, n_iter=150, lr=150):
    n=len(data)
    D2=[[sq_dist(data[i],data[j]) for j in range(n)] for i in range(n)]
    P=[[0.0]*n for _ in range(n)]
    for i in range(n):
        lo,hi,beta=0.0,1e10,1.0
        for _ in range(50):
            ed=[math.exp(-D2[i][j]*beta) if j!=i else 0.0 for j in range(n)]
            s=sum(ed) or 1e-10; pc=[e/s for e in ed]
            H=-sum(p*math.log(p+1e-10) for p in pc if p>0)
            diff=math.exp(H)-perp
            if abs(diff)<1e-5: break
            if diff>0: lo=beta; beta=(beta+hi)/2 if hi<1e9 else beta*2
            else:      hi=beta; beta=(beta+lo)/2
        P[i]=pc
    Ps=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n): Ps[i][j]=(P[i][j]+P[j][i])/(2*n)
    Y=[[random.gauss(0,0.01) for _ in range(2)] for _ in range(n)]
    vel=[[0.0]*2 for _ in range(n)]; gains=[[1.0]*2 for _ in range(n)]
    for t in range(1,n_iter+1):
        exag=4.0 if t<=60 else 1.0
        q_num=[[0.0]*n for _ in range(n)]; qs=1e-10
        for i in range(n):
            for j in range(n):
                if i!=j:
                    d2=sum((Y[i][k]-Y[j][k])**2 for k in range(2))
                    q_num[i][j]=1.0/(1.0+d2); qs+=q_num[i][j]
        grad=[[0.0]*2 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i==j: continue
                f=4*(Ps[i][j]*exag-q_num[i][j]/qs)*q_num[i][j]
                for d in range(2): grad[i][d]+=f*(Y[i][d]-Y[j][d])
        for i in range(n):
            for d in range(2):
                same=(grad[i][d]>0)==(vel[i][d]>0)
                gains[i][d]=gains[i][d]*0.8 if same else gains[i][d]+0.2
                gains[i][d]=max(gains[i][d],0.01)
                vel[i][d]=0.8*vel[i][d]-lr*gains[i][d]*grad[i][d]
                Y[i][d]+=vel[i][d]
        for d in range(2):
            m=sum(Y[i][d] for i in range(n))/n
            for i in range(n): Y[i][d]-=m
    return Y

print("Demonstrating: t-SNE inter-cluster distances are MEANINGLESS")
print("="*58)
print()

for name, dataset in [("Dataset A (true inter-cluster dist ≈ 14.1)", dataA),
                       ("Dataset B (true inter-cluster dist ≈ 2.1)",  dataB)]:
    true_d = true_inter(dataset, labels)
    Y = run_tsne(dataset, perp=10, n_iter=150)

    # Measure embedded inter-cluster centroid distances
    classes = sorted(set(labels))
    emb_inter=[]
    for ci in range(len(classes)):
        for cj in range(ci+1, len(classes)):
            ma=[i for i in range(n) if labels[i]==classes[ci]]
            mb=[i for i in range(n) if labels[i]==classes[cj]]
            ca=[sum(Y[i][d] for i in ma)/len(ma) for d in range(2)]
            cb=[sum(Y[i][d] for i in mb)/len(mb) for d in range(2)]
            emb_inter.append(math.sqrt(sum((ca[d]-cb[d])**2 for d in range(2))))
    emb_d = sum(emb_inter)/len(emb_inter)

    print(f"  {name}")
    print(f"    True high-dim inter-cluster distance : {true_d:.2f}")
    print(f"    t-SNE embedded inter-cluster distance: {emb_d:.2f}")
    print()

print("Key finding:")
print("  Dataset A has clusters 6.7× further apart than Dataset B in the")
print("  original 4D space. Yet their t-SNE inter-cluster distances are")
print("  nearly identical. The 2D plot looks almost the same for both.")
print()
print("  ⚠️  NEVER use t-SNE cluster gaps to judge true class separability.")
print("  ✓  USE t-SNE only to ask: do clusters exist? not how far apart are they?")
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
        from Unsupervised_Learning.visuals.tSNE_UMAP_visual import (   # ← match your exact folder casing
            VISUAL_HTML,
            VISUAL_HEIGHT,
        )
        visual_html   = VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = VISUAL_HEIGHT
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


