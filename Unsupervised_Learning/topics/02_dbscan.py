import textwrap
import re

OPERATIONS   = {}

"""Module: 02 · DBSCAN"""
TOPIC_NAME   ="DBSCAN - Density based clustering"
DISPLAY_NAME = "02 · DBSCAN"
ICON         = "🌀"
SUBTITLE     = "Density-based clustering — finds arbitrary shapes and outliers"

THEORY = """
## 02 · DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**

---

### The Problem with K-Means

Before learning DBSCAN, it helps to understand exactly where K-Means fails.

K-Means partitions data by minimising the sum of squared distances from each
point to its assigned centroid. This means every cluster is implicitly spherical
— it assumes each cluster is roughly circular (in 2D) or ellipsoidal (in higher
dimensions). When your real data isn't shaped that way, K-Means forces the
wrong structure onto it.

Three specific failure modes:

**1. Non-convex shapes** — Two interleaved half-moons, a ring inside a circle,
a spiral. K-Means will carve these into two blobs regardless of K.

**2. Outliers contaminate centroids** — K-Means assigns every single point to
a cluster. There is no concept of "noise." One extreme outlier shifts its
centroid away from the true cluster centre.

**3. You must choose K in advance** — You need to know (or guess) how many
clusters exist. In many real applications — anomaly detection, geographic
hot-spots, biology — the number of clusters is part of what you're trying
to discover.

DBSCAN solves all three.

---

### The Core Idea: Density

Instead of asking "which centroid is this point closest to?", DBSCAN asks a
completely different question:

> **Is this region of space dense?**

If a region of space has enough points packed into it, that region forms a
cluster. If a point sits in a sparse region, far from any dense area, it is
labelled **noise** — not assigned to any cluster.

This is a fundamental shift in thinking. K-Means is geometry-based (distance
to a centre). DBSCAN is density-based (local neighbourhood density).

---

### Two Hyperparameters

DBSCAN has exactly two parameters:

**ε (epsilon)** — the neighbourhood radius. For each point, DBSCAN draws a
circle of radius ε around it and asks: how many other points fall inside?

**MinPts** — the minimum number of points required inside the ε-neighbourhood
for a region to be considered dense. This includes the point itself.

These two parameters define what "dense" means in your data.

---

### Three Types of Points

Once ε and MinPts are set, every point in the dataset is classified as one of
exactly three types:

**Core Point** — A point p is a core point if at least MinPts points lie within
distance ε of p (counting p itself).

    |N_ε(p)| ≥ MinPts

where N_ε(p) is the ε-neighbourhood of p: the set of all points within distance
ε of p.

Core points are the heart of clusters. They live in dense regions.

---

**Border Point** — A point q is a border point if:
  - It is NOT a core point (fewer than MinPts neighbours), BUT
  - It lies within the ε-neighbourhood of at least one core point.

Border points are on the edge of a cluster. They belong to a cluster, but they
don't have enough neighbours to be its core.

---

**Noise Point** — A point is a noise point (outlier) if it is neither a core
point nor a border point. It does not lie within ε of any core point.

Noise points are assigned the label **−1**. They belong to no cluster.

---

### Reachability and Connectivity

DBSCAN builds clusters through two key concepts:

**Directly Density-Reachable** — Point q is directly density-reachable from
core point p if:
  - q ∈ N_ε(p)  (q is within ε of p)
  - p is a core point

Note: this relationship is NOT symmetric. q can be directly reachable from p
without p being directly reachable from q (if q is a border point).

---

**Density-Reachable** — Point q is density-reachable from p if there exists a
chain of points p₁, p₂, ..., pₙ where:
  - p₁ = p and pₙ = q
  - Each pᵢ₊₁ is directly density-reachable from pᵢ
  - All intermediate points p₁ through pₙ₋₁ are core points

This is the transitive closure of direct density-reachability. Two clusters can
be chained together through a sequence of core points, even if the endpoints are
far apart.

---

**Density-Connected** — Points p and q are density-connected if there exists a
point o such that both p and q are density-reachable from o.

Density-connectivity IS symmetric. A cluster in DBSCAN is a maximal set of
density-connected points.

---

### The Algorithm (Step by Step)

Here is DBSCAN from first principles:

```
Input: Dataset X, parameters ε and MinPts

1. Mark all points as unvisited.

2. For each unvisited point p:

   a. Mark p as visited.

   b. Compute N_ε(p) = {all points within distance ε of p}.

   c. If |N_ε(p)| < MinPts:
        Mark p as noise (label = -1).
        (It may be re-labelled as a border point later.)

   d. If |N_ε(p)| ≥ MinPts:
        p is a core point. Create a new cluster C.
        Add p to C.
        Let Seeds = N_ε(p) minus already-processed points.

        For each point q in Seeds:
            If q is unvisited:
                Mark q as visited.
                Compute N_ε(q).
                If |N_ε(q)| ≥ MinPts:
                    Seeds = Seeds ∪ N_ε(q)   (expand the frontier)
            If q is not yet a member of any cluster:
                Add q to cluster C.

3. Return cluster labels for all points.
```

The key mechanism is the Seeds expansion in step 2d. Starting from a core point,
DBSCAN expands outward through all reachable core points, absorbing their
neighbourhoods into the same cluster. This naturally follows the shape of the
data — it stops expanding only when it hits a sparse region.

---

### A Worked Example

Data (5 points, ε = 1.5, MinPts = 3):

    A = (1, 1)
    B = (1.3, 1.2)
    C = (0.9, 1.5)
    D = (5, 5)   ← far away
    E = (1.1, 1.1)

Step 1 — Compute ε-neighbourhoods:

    N_ε(A) = {A, B, C, E}   → |N_ε(A)| = 4 ≥ 3  → A is a Core Point ✓
    N_ε(B) = {A, B, C, E}   → |N_ε(B)| = 4 ≥ 3  → B is a Core Point ✓
    N_ε(C) = {A, B, C, E}   → |N_ε(C)| = 4 ≥ 3  → C is a Core Point ✓
    N_ε(E) = {A, B, C, E}   → |N_ε(E)| = 4 ≥ 3  → E is a Core Point ✓
    N_ε(D) = {D}             → |N_ε(D)| = 1 < 3  → D is Noise ✗

Step 2 — Build clusters:

    Start with A (core point). Create Cluster 1.
    Seeds = {B, C, E}.

    Process B (core) → N_ε(B) = {A,B,C,E} → all already in Seeds or visited.
    Process C (core) → same.
    Process E (core) → same.
    Seeds exhausted.

    Cluster 1 = {A, B, C, E}
    Noise     = {D}

Final labels: A→1, B→1, C→1, E→1, D→−1

---

### Distance Metric

The standard DBSCAN uses **Euclidean distance**:

    dist(p, q) = √(Σᵢ (pᵢ − qᵢ)²)

But DBSCAN can use any distance metric. For geographic coordinates, you might
use Haversine distance. For text, cosine distance. For images, learned
embedding distance. The algorithm itself is agnostic to the metric.

The choice of distance metric interacts with ε: the same ε value has very
different meaning under different metrics. Always scale your features before
applying DBSCAN with Euclidean distance, otherwise high-variance features will
dominate the neighbourhood calculation.

---

### Time and Space Complexity

**Naïve implementation:**

    Time:  O(n²)  — for each of n points, compute distance to all other n points
    Space: O(n²)  — if storing the full distance matrix

**With a spatial index (k-d tree or ball tree):**

    Time:  O(n log n) average case for low-dimensional data
    Space: O(n)

For high-dimensional data (d > ~20), spatial indices lose their advantage and
the algorithm degrades back toward O(n²). This is sometimes called the "curse
of dimensionality" — a well-known limitation of DBSCAN.

---

### Choosing ε and MinPts

This is the most important practical question in applying DBSCAN.

**Rule of thumb for MinPts:**
  - For 2D data: MinPts = 4
  - For d-dimensional data: MinPts ≥ d + 1
  - Larger MinPts → smoother, more robust clusters; more points become noise
  - MinPts = 1 degenerates to a nearest-neighbour graph (no clustering)
  - MinPts = 2 degenerates to single-linkage hierarchical clustering

**The k-Distance Graph for ε:**

The standard method for choosing ε:

1. Fix MinPts = k.
2. For each point, compute its distance to its k-th nearest neighbour. Call this
   the **k-distance**.
3. Sort all n k-distances in descending order and plot them.
4. Look for the **"elbow"** — the point of maximum curvature in the plot.
5. The ε value at the elbow is your estimate.

The intuition: points in dense regions have small k-distances. Points in sparse
regions have large k-distances. The elbow separates these two populations. Any
ε below the elbow treats dense-region spacing as the threshold; noise points
(which have larger k-distances) fall outside it.

---

### DBSCAN vs K-Means: Direct Comparison

| Property                    | K-Means                          | DBSCAN                           |
|-----------------------------|----------------------------------|----------------------------------|
| Cluster shape               | Convex (spherical)               | Arbitrary (follows density)      |
| Number of clusters          | Must specify K in advance        | Determined automatically         |
| Outlier handling            | All points assigned to a cluster | Noise points labelled −1         |
| Cluster size sensitivity    | Assumes similar-sized clusters   | Handles variable-sized clusters  |
| Deterministic               | No (depends on initialisation)   | Yes (given fixed point ordering) |
| Scales to large n           | Yes (O(n·K·iterations))          | Yes with spatial index O(n log n)|
| High-dimensional data       | Works well                       | Degrades (curse of dimensionality)|
| Hyperparameters             | K                                | ε and MinPts                     |
| Underlying assumption       | Minimise within-cluster variance | Density is uniform within clusters|

---

### When Does DBSCAN Struggle?

**Varying density clusters** — DBSCAN uses a single global ε. If one cluster is
very dense and another is sparse, no single ε correctly separates both from
noise. This is the most fundamental limitation. The algorithm HDBSCAN (Hierarchical
DBSCAN) addresses this by building a hierarchy of clusterings across all possible
ε values and extracting the most stable clusters.

**High-dimensional data** — In many dimensions, Euclidean distance concentrates:
all points become approximately equidistant from each other. The notion of a
"dense neighbourhood" breaks down. Dimensionality reduction (PCA, UMAP) before
DBSCAN is often necessary.

**Large ε can merge distinct clusters** — If ε is too large, density bridges
form between clusters that should be separate.

---

### Relationship to Other Algorithms

**OPTICS (Ordering Points To Identify the Clustering Structure)** — A
generalisation of DBSCAN that produces a reachability plot instead of a flat
clustering, allowing you to extract clusters at multiple ε levels. Does not
require specifying ε in advance.

**HDBSCAN (Hierarchical DBSCAN)** — Builds a hierarchy of DBSCAN clusterings
by varying ε from ∞ to 0. Extracts the most persistent (stable) flat clustering
from this hierarchy. Handles varying-density clusters. Generally the preferred
choice over DBSCAN in practice.

**Mean Shift** — Also density-based but finds cluster centres by iteratively
shifting each point toward the mean of its neighbourhood. Automatically determines
K but is O(n²) and sensitive to bandwidth (analogous to ε).

---

### Key Takeaways

1. DBSCAN defines clusters as dense regions of space, connected through
   density-reachability. Shape does not matter — only local density does.

2. Every point is exactly one of: core (dense region interior), border (dense
   region edge), or noise (isolated). Noise points get label −1.

3. Two parameters: ε sets the neighbourhood radius; MinPts sets the density
   threshold. Use the k-distance elbow plot to choose ε.

4. DBSCAN is deterministic, does not require specifying K, and naturally
   identifies outliers — three things K-Means cannot do.

5. Its main weakness is a single global density threshold. For datasets where
   cluster densities vary, prefer HDBSCAN.
"""


OPERATIONS = {

    "▶ Run: Core vs Border vs Noise Classification": {
        "description": "Classify every point in a 2D dataset as Core, Border, or Noise using a manual DBSCAN scan. Prints a full point-by-point classification table.",
        "code": """
import math

# ── Dataset ───────────────────────────────────────────────────────────────────
points = {
    'A': (1.0, 1.0),
    'B': (1.3, 1.2),
    'C': (0.9, 1.5),
    'E': (1.1, 1.1),
    'F': (4.0, 0.5),  # border candidate
    'D': (5.0, 5.0),  # noise
}

EPS     = 1.5
MINPTS  = 3

# ── Euclidean distance ────────────────────────────────────────────────────────
def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ── Compute ε-neighbourhood for each point ────────────────────────────────────
names = list(points.keys())
coords = list(points.values())

neighbourhoods = {}
for name, pt in points.items():
    neighbours = [
        other for other, ocoord in points.items()
        if dist(pt, ocoord) <= EPS
    ]
    neighbourhoods[name] = neighbours

# ── Classify ─────────────────────────────────────────────────────────────────
core_points = {p for p, nbrs in neighbourhoods.items() if len(nbrs) >= MINPTS}

labels = {}
for name in points:
    if name in core_points:
        labels[name] = 'CORE'
    elif any(nbr in core_points for nbr in neighbourhoods[name] if nbr != name):
        labels[name] = 'BORDER'
    else:
        labels[name] = 'NOISE'

# ── Print results ─────────────────────────────────────────────────────────────
print(f"Parameters: ε = {EPS}, MinPts = {MINPTS}")
print()
print(f"{'Point':<8} {'Coords':<18} {'|N_ε|':<8} {'Neighbours':<30} {'Type'}")
print("─" * 80)
for name in names:
    coord  = points[name]
    nbrs   = neighbourhoods[name]
    label  = labels[name]
    print(f"{name:<8} {str(coord):<18} {len(nbrs):<8} {str(nbrs):<30} {label}")

print()
print(f"Core Points  : {sorted(p for p,l in labels.items() if l=='CORE')}")
print(f"Border Points: {sorted(p for p,l in labels.items() if l=='BORDER')}")
print(f"Noise Points : {sorted(p for p,l in labels.items() if l=='NOISE')}")
""",
        "runnable": True,
    },

    "▶ Run: DBSCAN From Scratch": {
        "description": "Full DBSCAN implementation from first principles — no sklearn. Builds clusters via the seeds-expansion algorithm on a synthetic moon-shaped dataset.",
        "code": """
import math
import random

random.seed(42)

# ── Generate two half-moon clusters manually ──────────────────────────────────
def make_moons(n=60, noise=0.08):
    data, true_labels = [], []
    for i in range(n):
        angle = math.pi * i / (n // 2)
        if i < n // 2:
            x = math.cos(angle)
            y = math.sin(angle)
            label = 0
        else:
            x = 1 - math.cos(angle)
            y = 1 - math.sin(angle) - 0.5
            label = 1
        x += random.gauss(0, noise)
        y += random.gauss(0, noise)
        data.append((x, y))
        true_labels.append(label)
    # Add 4 noise outliers
    for _ in range(4):
        data.append((random.uniform(-1.5, 2.5), random.uniform(-1.5, 1.5)))
        true_labels.append(-1)
    return data, true_labels

X, true_labels = make_moons(n=60, noise=0.1)

# ── DBSCAN from scratch ───────────────────────────────────────────────────────
def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def region_query(X, idx, eps):
    return [j for j, pt in enumerate(X) if euclidean(X[idx], pt) <= eps]

def dbscan(X, eps, minpts):
    n       = len(X)
    labels  = [-999] * n   # -999 = unvisited
    cluster_id = 0

    for i in range(n):
        if labels[i] != -999:
            continue
        neighbours = region_query(X, i, eps)
        if len(neighbours) < minpts:
            labels[i] = -1   # noise
            continue
        # Core point — start a new cluster
        cluster_id += 1
        labels[i]  = cluster_id
        seeds = list(neighbours)
        s = 0
        while s < len(seeds):
            q = seeds[s]
            if labels[q] == -1:
                labels[q] = cluster_id          # noise → border
            if labels[q] == -999:
                labels[q] = cluster_id
                q_neighbours = region_query(X, q, eps)
                if len(q_neighbours) >= minpts:
                    seeds.extend(q_neighbours)  # expand frontier
            s += 1

    return labels

EPS    = 0.3
MINPTS = 4

labels = dbscan(X, EPS, MINPTS)

# ── Results ───────────────────────────────────────────────────────────────────
from collections import Counter

unique_clusters = set(labels) - {-1}
n_clusters = len(unique_clusters)
n_noise    = labels.count(-1)
counts     = Counter(l for l in labels if l != -1)

print("=" * 50)
print("        DBSCAN Results")
print("=" * 50)
print(f"  Dataset size  : {len(X)} points")
print(f"  Parameters    : ε={EPS}, MinPts={MINPTS}")
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points  : {n_noise}")
print()
print("  Cluster sizes:")
for cid in sorted(counts):
    bar = "█" * (counts[cid] // 2)
    print(f"    Cluster {cid}: {counts[cid]:3d} points  {bar}")
print()

# Purity check (compare to true labels)
correct = 0
for i in range(len(X)):
    if labels[i] != -1:
        # Find majority true label in this cluster
        pass

# Quick purity by checking cluster composition
for cid in sorted(unique_clusters):
    members = [i for i,l in enumerate(labels) if l == cid]
    true_in_cluster = Counter(true_labels[i] for i in members)
    majority = true_in_cluster.most_common(1)[0]
    print(f"  Cluster {cid}: {len(members)} pts | true-label breakdown: {dict(true_in_cluster)}")

print()
print(f"  Noise points assigned label -1: {n_noise}")
print("  (True outliers we injected: 4)")
""",
        "runnable": True,
    },

    "▶ Run: k-Distance Elbow Plot (ε Selection)": {
        "description": "Compute the k-distance graph to find the optimal ε using the elbow method. Prints the sorted k-distances and identifies the elbow automatically.",
        "code": """
import math
import random

random.seed(0)

# ── Generate data: two dense blobs + noise ────────────────────────────────────
def gen_blob(cx, cy, r, n):
    pts = []
    while len(pts) < n:
        x = random.gauss(cx, r)
        y = random.gauss(cy, r)
        pts.append((x, y))
    return pts

X = (gen_blob(0, 0, 0.4, 40)
   + gen_blob(3, 3, 0.4, 40)
   + [(random.uniform(-2, 5), random.uniform(-2, 5)) for _ in range(10)])

random.shuffle(X)
n = len(X)

# ── Compute k-distances ───────────────────────────────────────────────────────
K = 4  # MinPts - 1 (standard rule: k = MinPts)

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

k_distances = []
for i, pt in enumerate(X):
    dists = sorted(euclidean(pt, X[j]) for j in range(n) if j != i)
    k_distances.append(dists[K - 1])  # distance to k-th nearest neighbour

k_distances_sorted = sorted(k_distances, reverse=True)

# ── Find elbow via maximum second difference ──────────────────────────────────
second_diffs = []
for i in range(1, len(k_distances_sorted) - 1):
    d2 = k_distances_sorted[i-1] - 2*k_distances_sorted[i] + k_distances_sorted[i+1]
    second_diffs.append((i, abs(d2), k_distances_sorted[i]))

elbow_idx, _, elbow_eps = max(second_diffs, key=lambda x: x[1])

# ── Print the k-distance plot (ASCII) ────────────────────────────────────────
print(f"k-Distance Graph  (k={K}, n={n} points)")
print(f"Sorted k-distances (descending) — look for the elbow")
print()

max_d = max(k_distances_sorted)
width = 50

print(f"{'idx':>5}  {'k-dist':>7}  Plot")
print("─" * 70)
for i, d in enumerate(k_distances_sorted):
    bar_len = int(d / max_d * width)
    bar     = "█" * bar_len
    marker  = " ◄ ELBOW" if i == elbow_idx else ""
    if i % 5 == 0 or i == elbow_idx:
        print(f"  {i:>3}    {d:>6.3f}  {bar}{marker}")

print()
print("=" * 50)
print(f"  Elbow detected at index : {elbow_idx}")
print(f"  Suggested ε             : {elbow_eps:.3f}")
print(f"  Interpretation:")
print(f"    Points left of elbow  → in dense clusters (small k-dist)")
print(f"    Points right of elbow → sparse / noise    (large k-dist)")
print(f"    Use ε ≈ {elbow_eps:.2f} with MinPts = {K}")
""",
        "runnable": True,
    },

    "▶ Run: DBSCAN vs K-Means on Non-Convex Data": {
        "description": "Direct comparison of DBSCAN and K-Means on concentric ring data where K-Means fundamentally fails and DBSCAN succeeds.",
        "code": """
import math
import random

random.seed(7)

# ── Generate concentric rings ─────────────────────────────────────────────────
def make_ring(r, noise, n):
    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n + random.gauss(0, 0.05)
        x = r * math.cos(angle) + random.gauss(0, noise)
        y = r * math.sin(angle) + random.gauss(0, noise)
        pts.append((x, y))
    return pts

inner = make_ring(1.0, 0.08, 50)
outer = make_ring(2.5, 0.08, 80)
X     = inner + outer
true  = [0]*50 + [1]*80

n = len(X)

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ── K-Means (K=2) ─────────────────────────────────────────────────────────────
def kmeans(X, K=2, iters=100):
    # Init: pick K random points as centroids
    centroids = random.sample(X, K)
    labels    = [0] * len(X)
    for _ in range(iters):
        # Assign
        for i, pt in enumerate(X):
            labels[i] = min(range(K), key=lambda k: euclidean(pt, centroids[k]))
        # Update
        new_centroids = []
        for k in range(K):
            members = [X[i] for i in range(len(X)) if labels[i] == k]
            if members:
                cx = sum(p[0] for p in members) / len(members)
                cy = sum(p[1] for p in members) / len(members)
                new_centroids.append((cx, cy))
            else:
                new_centroids.append(centroids[k])
        centroids = new_centroids
    return labels

# ── DBSCAN ────────────────────────────────────────────────────────────────────
def region_query(X, idx, eps):
    return [j for j in range(len(X)) if euclidean(X[idx], X[j]) <= eps]

def dbscan(X, eps, minpts):
    labels = [-999] * len(X)
    cid    = 0
    for i in range(len(X)):
        if labels[i] != -999: continue
        nb = region_query(X, i, eps)
        if len(nb) < minpts:
            labels[i] = -1; continue
        cid += 1
        labels[i] = cid
        seeds = list(nb)
        s = 0
        while s < len(seeds):
            q = seeds[s]
            if labels[q] == -1: labels[q] = cid
            if labels[q] == -999:
                labels[q] = cid
                qnb = region_query(X, q, eps)
                if len(qnb) >= minpts: seeds.extend(qnb)
            s += 1
    return labels

# ── Evaluate: purity ──────────────────────────────────────────────────────────
def purity(pred_labels, true_labels):
    from collections import Counter
    clusters = set(l for l in pred_labels if l != -1)
    total_correct = 0
    for cid in clusters:
        members = [true_labels[i] for i,l in enumerate(pred_labels) if l == cid]
        if members:
            total_correct += Counter(members).most_common(1)[0][1]
    assigned = sum(1 for l in pred_labels if l != -1)
    return total_correct / assigned if assigned > 0 else 0

km_labels   = kmeans(X, K=2)
dbscan_lbl  = dbscan(X, eps=0.35, minpts=4)

km_purity  = purity(km_labels,  true)
db_purity  = purity(dbscan_lbl, true)

db_clusters = len(set(l for l in dbscan_lbl if l != -1))
db_noise    = dbscan_lbl.count(-1)

# ── ASCII scatter ─────────────────────────────────────────────────────────────
def ascii_scatter(labels, title, W=50, H=20):
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    mn_x, mx_x = min(xs)-0.2, max(xs)+0.2
    mn_y, mx_y = min(ys)-0.2, max(ys)+0.2
    grid = [['·']*W for _ in range(H)]
    symbols = ['○', '●', '□', '■', '△', '▲']
    for i, (pt, lbl) in enumerate(zip(X, labels)):
        col = int((pt[0]-mn_x)/(mx_x-mn_x) * (W-1))
        row = int((pt[1]-mn_y)/(mx_y-mn_y) * (H-1))
        row = H - 1 - row
        sym = 'x' if lbl == -1 else symbols[lbl % len(symbols)]
        grid[row][col] = sym
    print(f"  ┌{'─'*W}┐  {title}")
    for row in grid:
        print(f"  │{''.join(row)}│")
    print(f"  └{'─'*W}┘")

print("Concentric Rings — DBSCAN vs K-Means")
print("="*55)
print()
ascii_scatter(km_labels,  f"K-Means (K=2)   Purity={km_purity:.2f}")
print()
ascii_scatter(dbscan_lbl, f"DBSCAN (ε=0.35) Purity={db_purity:.2f}  Noise={db_noise}")
print()
print(f"  K-Means purity  : {km_purity:.2%}  (forced to split ring — fails)")
print(f"  DBSCAN purity   : {db_purity:.2%}  (follows ring shape — succeeds)")
print(f"  DBSCAN clusters : {db_clusters}  |  Noise: {db_noise} pts")
""",
        "runnable": True,
    },

    "▶ Run: Effect of ε and MinPts on Clustering": {
        "description": "Grid search over ε and MinPts values. Shows how the number of clusters, noise count, and classification of points changes across the parameter space.",
        "code": """
import math
import random

random.seed(3)

# ── Dataset: 3 clusters + sparse noise ───────────────────────────────────────
def blob(cx, cy, r, n):
    return [(random.gauss(cx,r), random.gauss(cy,r)) for _ in range(n)]

X = (blob(0, 0, 0.3, 30)
   + blob(3, 0, 0.3, 30)
   + blob(1.5, 2.5, 0.3, 25)
   + [(random.uniform(-1,4), random.uniform(-1,3.5)) for _ in range(8)])

n = len(X)

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def dbscan(X, eps, minpts):
    labels = [-999]*len(X)
    cid = 0
    for i in range(len(X)):
        if labels[i] != -999: continue
        nb = [j for j in range(len(X)) if euclidean(X[i],X[j]) <= eps]
        if len(nb) < minpts:
            labels[i] = -1; continue
        cid += 1; labels[i] = cid
        seeds = list(nb)
        s = 0
        while s < len(seeds):
            q = seeds[s]
            if labels[q] == -1: labels[q] = cid
            if labels[q] == -999:
                labels[q] = cid
                qnb = [j for j in range(len(X)) if euclidean(X[q],X[j]) <= eps]
                if len(qnb) >= minpts: seeds.extend(qnb)
            s += 1
    return labels

# ── Parameter grid ────────────────────────────────────────────────────────────
eps_values    = [0.2, 0.4, 0.6, 0.9, 1.5]
minpts_values = [2, 4, 7, 10]

print(f"Dataset: {n} points  (3 true clusters + 8 noise)")
print()
print(f"{'ε':>6} | {'MinPts':>7} | {'Clusters':>9} | {'Noise pts':>9} | {'Core pts':>9} | Interpretation")
print("─" * 80)

for eps in eps_values:
    for mp in minpts_values:
        labels = dbscan(X, eps, mp)
        n_clusters = len(set(l for l in labels if l != -1))
        n_noise    = labels.count(-1)
        # count core points
        n_core = 0
        for i in range(len(X)):
            nb = sum(1 for j in range(len(X)) if euclidean(X[i],X[j]) <= eps)
            if nb >= mp:
                n_core += 1

        if n_clusters == 0:
            interp = "⚠ All noise — ε too small or MinPts too large"
        elif n_clusters == 1:
            interp = "⚠ Everything merged — ε too large"
        elif n_clusters == 3 and n_noise <= 12:
            interp = "✓ Correct — 3 clusters recovered"
        elif n_clusters > 5:
            interp = "⚠ Over-fragmented — ε too small"
        else:
            interp = f"~ {n_clusters} clusters"

        print(f"  {eps:>4.1f} | {mp:>7} | {n_clusters:>9} | {n_noise:>9} | {n_core:>9} | {interp}")
    print()

print("Key observations:")
print("  • Increasing ε merges clusters (fewer, larger clusters, less noise)")
print("  • Increasing MinPts raises density bar (fewer core pts, more noise)")
print("  • Sweet spot: ε=0.4-0.6, MinPts=4 recovers the 3 true clusters")
""",
        "runnable": True,
    },

    "▶ Run: Density-Reachability Chain Trace": {
        "description": "Trace the full density-reachability chain between two specified points. Shows whether they are directly reachable, density-reachable, or density-connected.",
        "code": """
import math
from collections import deque

# ── Small labelled dataset ────────────────────────────────────────────────────
#  Three clusters: A-cluster, B-cluster, isolated noise
points = {
    'p1': (0.0, 0.0),
    'p2': (0.4, 0.1),
    'p3': (0.2, 0.5),
    'p4': (0.5, 0.4),
    'p5': (0.1, 0.3),
    'p6': (3.0, 3.0),   # second cluster
    'p7': (3.4, 3.1),
    'p8': (3.2, 3.5),
    'p9': (3.5, 3.3),
    'pN': (6.0, 0.0),   # noise
}

EPS    = 0.6
MINPTS = 3

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

names  = list(points.keys())
coords = list(points.values())

# Neighbourhoods
nb = {p: [q for q in names if dist(points[p], points[q]) <= EPS] for p in names}

# Core points
core = {p for p in names if len(nb[p]) >= MINPTS}

print(f"Parameters: ε={EPS}, MinPts={MINPTS}")
print()
print("Point classifications:")
for p in names:
    if p in core:
        kind = f"CORE   |N_ε|={len(nb[p])}"
    elif any(c in core for c in nb[p] if c != p):
        kind = f"BORDER |N_ε|={len(nb[p])}"
    else:
        kind = f"NOISE  |N_ε|={len(nb[p])}"
    print(f"  {p}: {kind}  neighbours={nb[p]}")

print()

# ── Density-reachability BFS ──────────────────────────────────────────────────
def density_reachable_path(src, tgt):
    \"\"\"Find a density-reachable chain from src to tgt through core points.\"\"\"
    if src not in core:
        return None, f"{src} is not a core point — density-reachability requires starting from a core point"
    queue   = deque([[src]])
    visited = {src}
    while queue:
        path = queue.popleft()
        last = path[-1]
        for nbr in nb[last]:
            if nbr == tgt:
                return path + [nbr], "PATH FOUND"
            if nbr not in visited and nbr in core:
                visited.add(nbr)
                queue.append(path + [nbr])
    return None, f"No density-reachable path from {src} to {tgt}"

def density_connected(p, q):
    \"\"\"Check if p and q are density-connected (common origin point o).\"\"\"
    for o in core:
        p_reach, _ = density_reachable_path(o, p)
        q_reach, _ = density_reachable_path(o, q)
        if p_reach and q_reach:
            return True, o
    return False, None

# Test pairs
test_pairs = [('p1', 'p4'), ('p1', 'p6'), ('p1', 'pN'), ('p3', 'p2')]

for (a, b) in test_pairs:
    print(f"─── Reachability: {a} → {b} ───")
    # Direct?
    direct = b in nb[a] and a in core
    print(f"  Directly density-reachable : {'YES' if direct else 'NO'}")
    # Density-reachable?
    path, msg = density_reachable_path(a, b)
    if path:
        print(f"  Density-reachable path     : {' → '.join(path)}")
    else:
        print(f"  Density-reachable          : NO  ({msg})")
    # Density-connected?
    connected, origin = density_connected(a, b)
    if connected:
        print(f"  Density-connected          : YES (via origin={origin})")
    else:
        print(f"  Density-connected          : NO")
    print()
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
        from Unsupervised_Learning.visuals.dbscan_visual import (   # ← match your exact folder casing
            DBSCAN_VISUAL_HTML,
            DBSCAN_VISUAL_HEIGHT,
        )
        visual_html   = DBSCAN_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = DBSCAN_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[01_linear_regression.py] Could not load visual: {e}", stacklevel=2)

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
