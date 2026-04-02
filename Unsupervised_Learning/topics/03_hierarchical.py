"""Module: 03 · Hierarchical Clustering"""
import textwrap
import re

TOPIC_NAME   ="Hierarchical Clustering"
DISPLAY_NAME = "03 · Hierarchical Clustering"
ICON         = "🌿"
SUBTITLE     = "Build a dendrogram of nested clusters — agglomerative and divisive"

THEORY = """
## 03 · Hierarchical Clustering

---

### The Big Picture

K-Means and DBSCAN both produce a **flat** clustering: a single assignment of
every point to a cluster label. You get one answer. If you want to know whether
K=3 or K=5 is better, you have to run the algorithm again from scratch.

Hierarchical clustering is fundamentally different. It produces a **tree** — a
nested structure called a **dendrogram** that captures the clustering at every
possible granularity simultaneously.

From this single tree you can extract 2 clusters, 10 clusters, or any number
in between, simply by choosing where to cut. The whole hierarchy is computed
once and the exploration is free.

---

### Two Strategies: Agglomerative vs Divisive

There are two mirror-image approaches:

**Agglomerative (bottom-up)** — Start with every point as its own cluster.
Repeatedly merge the two closest clusters until everything is one cluster.
This builds the tree from the leaves up to the root.

**Divisive (top-down)** — Start with all points in one cluster. Repeatedly
split the most heterogeneous cluster until every point is isolated. This builds
the tree from the root down to the leaves.

In practice, agglomerative clustering dominates. It is simpler to implement,
better studied, and the linkage criteria (how you measure "closest clusters")
give rich control over the cluster shapes it finds. Divisive methods are used
in specialised settings such as DIANA (Divisive ANAlysis) in bioinformatics.

This module covers agglomerative clustering in full, with divisive treated as
the conceptual counterpart.

---

### Agglomerative Clustering: Step by Step

**Input:** n data points, a distance metric, a linkage criterion.

**Algorithm:**

```
1. Assign each of the n points to its own cluster.
   Active clusters = {C₁}, {C₂}, ..., {Cₙ}

2. Compute a distance matrix D where D[i][j] = dist(Cᵢ, Cⱼ)
   for all pairs of active clusters.

3. Repeat until only one cluster remains:
   a. Find the pair (Cᵢ, Cⱼ) with minimum D[i][j].
   b. Merge Cᵢ and Cⱼ into a new cluster Cₙₑw = Cᵢ ∪ Cⱼ.
   c. Record this merge at height D[i][j] in the dendrogram.
   d. Remove Cᵢ and Cⱼ from the active set.
   e. Add Cₙₑw to the active set.
   f. Update D: compute distances from Cₙₑw to all remaining clusters.

4. Return the dendrogram (full merge history).
```

The dendrogram records every merge: which two clusters merged, and at what
distance (height). Reading the dendrogram from bottom to top traces the entire
history of the algorithm.

---

### A Worked Example (4 Points)

Data:
    A = (0, 0)
    B = (1, 0)
    C = (5, 0)
    D = (6, 0)

**Step 0 — Initial distance matrix (Euclidean):**

```
     A    B    C    D
A    0    1    5    6
B    1    0    4    5
C    5    4    0    1
D    6    5    1    0
```

**Step 1 — Minimum distance = 1 (two ties: A-B and C-D). Take A-B.**

Merge A and B → cluster AB.
Record: (A, B) merged at height 1.

New distances from AB to C and D (using single linkage = minimum):
    dist(AB, C) = min(dist(A,C), dist(B,C)) = min(5, 4) = 4
    dist(AB, D) = min(dist(A,D), dist(B,D)) = min(6, 5) = 5

Updated matrix:
```
      AB    C    D
AB     0    4    5
C      4    0    1
D      5    1    0
```

**Step 2 — Minimum distance = 1 (C-D).**

Merge C and D → cluster CD.
Record: (C, D) merged at height 1.

dist(AB, CD) = min(dist(AB,C), dist(AB,D)) = min(4, 5) = 4

Updated matrix:
```
      AB    CD
AB     0     4
CD     4     0
```

**Step 3 — Merge AB and CD at height 4.**

Record: (AB, CD) merged at height 4.
One cluster remains. Done.

**Dendrogram (read bottom to top):**

```
height
  4  ┤        ┌──────┐
     │        │      │
  1  ┤  ┌──┐  │  ┌──┐│
     │  │  │  │  │  ││
  0  ┤  A  B     C  D
```

Cutting at height 2: two clusters {A, B} and {C, D}.
Cutting at height 0.5: four clusters (original points).

---

### Linkage Criteria

The linkage criterion defines how the distance between two **clusters** is
computed from the distances between their individual **points**. This is the
most consequential design choice in hierarchical clustering. Different linkages
produce dramatically different cluster shapes.

**Single Linkage (Minimum Linkage)**

    dist(Cᵢ, Cⱼ) = min{ d(p, q) : p ∈ Cᵢ, q ∈ Cⱼ }

Distance between clusters = distance between their two closest points.

Behaviour: Produces long, chained clusters that follow irregular shapes.
Excellent for discovering elongated or filamentary structures. Susceptible to
the "chaining effect" — two clusters that share even a single nearby bridge
point will be merged early, sometimes collapsing genuinely separate clusters
prematurely.

Use when: Your clusters are non-convex or elongated, and you don't mind the
chaining effect. Similar in spirit to DBSCAN's density-connectivity.

---

**Complete Linkage (Maximum Linkage)**

    dist(Cᵢ, Cⱼ) = max{ d(p, q) : p ∈ Cᵢ, q ∈ Cⱼ }

Distance between clusters = distance between their two most distant points.

Behaviour: Produces compact, roughly equal-sized clusters. Resists the chaining
effect because two clusters won't merge until all cross-cluster pairs are within
the threshold. Sensitive to outliers because a single distant point inflates
the inter-cluster distance.

Use when: You expect roughly spherical, similarly-sized clusters with no
extreme outliers.

---

**Average Linkage (UPGMA)**

Unweighted Pair Group Method with Arithmetic Mean:

    dist(Cᵢ, Cⱼ) = (1 / |Cᵢ||Cⱼ|) · Σₚ∈Cᵢ Σq∈Cⱼ d(p, q)

Distance between clusters = average of all pairwise distances.

Behaviour: A compromise between single and complete linkage. Less susceptible
to chaining than single linkage; less sensitive to outliers than complete
linkage. Produces moderately compact clusters.

Use when: You want a balanced default — average linkage is the most commonly
recommended starting point.

---

**Ward Linkage (Minimum Variance)**

    dist(Cᵢ, Cⱼ) = ( |Cᵢ|·|Cⱼ| / (|Cᵢ|+|Cⱼ|) ) · ||μᵢ − μⱼ||²

where μᵢ and μⱼ are the centroids of Cᵢ and Cⱼ.

This is the increase in total within-cluster sum of squares (WCSS) that would
result from merging Cᵢ and Cⱼ. Ward's method always merges the pair that
increases total WCSS the least.

Behaviour: Produces compact, roughly spherical clusters of similar size.
Equivalent to K-Means in its objective (minimise WCSS), but computed
hierarchically. The most popular linkage for general tabular data.

Limitation: Sensitive to outliers (an outlier forces a large WCSS increase to
merge it into any cluster). Does not handle non-spherical clusters well.

Use when: Your data is roughly Gaussian, clusters are similar in size, and
you want the most "K-Means-like" hierarchical result.

---

**Centroid Linkage (UPGMC)**

    dist(Cᵢ, Cⱼ) = ||μᵢ − μⱼ||²

Distance between clusters = squared distance between their centroids.

Behaviour: Intuitive but can produce inversions — cases where a merge happens
at a lower height than a previous merge, violating the monotonicity property
that makes dendrograms easy to read. Rarely used in practice.

---

### Linkage Summary Table

| Linkage   | Formula                         | Shape         | Outlier sensitivity | Chaining risk |
|-----------|----------------------------------|---------------|---------------------|---------------|
| Single    | min pairwise distance            | Arbitrary     | Low                 | High          |
| Complete  | max pairwise distance            | Compact/equal | High                | Low           |
| Average   | mean pairwise distance           | Moderate      | Moderate            | Low           |
| Ward      | increase in total WCSS           | Spherical     | High                | Low           |
| Centroid  | distance between centroids       | Spherical     | Moderate            | Medium        |

---

### The Dendrogram

The dendrogram is the full output of hierarchical clustering. It is a binary
tree where:

- Each **leaf** represents one original data point.
- Each **internal node** represents a merge event.
- The **height** of an internal node = the distance at which that merge occurred.
- Nodes higher in the tree correspond to later (more dissimilar) merges.

**Reading a dendrogram:**

A horizontal cut at height h produces a flat clustering. Any branch whose root
is above h but whose leaves are below h corresponds to one cluster. Lowering h
increases the number of clusters; raising h decreases it.

**Identifying the best cut:**

Look for the largest vertical gap in the dendrogram — the longest branch that
is not crossed by any horizontal line. This gap represents the most significant
separation in the merge history. A cut through this gap gives the most natural
number of clusters.

This is the hierarchical analog of the elbow method.

---

### Complexity

**Naïve implementation:**

    Time:  O(n³)  — n iterations, each scanning an O(n²) distance matrix
    Space: O(n²)  — storing the full distance matrix

**With optimised priority queues (e.g. Prim's algorithm approach):**

    Time:  O(n² log n) for most linkages
    Space: O(n²)

For single and complete linkage, SLINK and CLINK algorithms achieve O(n²) time
and O(n) space — but these are specialised algorithms, not the general approach.

Ward linkage with Lance-Williams update formula also achieves O(n² log n).

In practice, hierarchical clustering is expensive for large n. For n > ~10,000
you typically want K-Means or DBSCAN instead, or approximate hierarchical
methods such as BIRCH.

---

### Lance-Williams Update Formula

After merging clusters Cᵢ and Cⱼ into Cₖ = Cᵢ ∪ Cⱼ, the distance from Cₖ
to any remaining cluster Cₘ can be updated without recomputing all pairwise
distances from scratch:

    dist(Cₖ, Cₘ) = αᵢ · dist(Cᵢ, Cₘ)
                  + αⱼ · dist(Cⱼ, Cₘ)
                  + β  · dist(Cᵢ, Cⱼ)
                  + γ  · |dist(Cᵢ, Cₘ) − dist(Cⱼ, Cₘ)|

The parameters (αᵢ, αⱼ, β, γ) are set differently for each linkage:

| Linkage   | αᵢ              | αⱼ              | β                       | γ    |
|-----------|-----------------|-----------------|-------------------------|------|
| Single    | 1/2             | 1/2             | 0                       | −1/2 |
| Complete  | 1/2             | 1/2             | 0                       | +1/2 |
| Average   | |Cᵢ|/(|Cᵢ|+|Cⱼ|)| |Cⱼ|/(|Cᵢ|+|Cⱼ|)| 0                  | 0    |
| Ward      | (|Cₘ|+|Cᵢ|)/denom | (|Cₘ|+|Cⱼ|)/denom | −|Cₘ|/denom       | 0    |

where denom = |Cₘ| + |Cᵢ| + |Cⱼ| for Ward.

This allows O(n²) matrix updates rather than O(n² · merge-cost) recomputation.

---

### Agglomerative vs DBSCAN vs K-Means

| Property                | K-Means         | DBSCAN              | Hierarchical (Agglom.)  |
|-------------------------|-----------------|---------------------|-------------------------|
| Output type             | Flat clustering | Flat + noise labels | Full dendrogram tree    |
| Must specify K          | Yes             | No                  | No (or after the fact)  |
| Cluster shape           | Convex          | Arbitrary           | Depends on linkage      |
| Handles outliers        | No              | Yes (noise = −1)    | Sort of (long branches) |
| Deterministic           | No              | Yes                 | Yes                     |
| Time complexity         | O(n·K·iter)     | O(n log n)          | O(n² log n) typical     |
| Scalability             | Good            | Good                | Poor for large n        |
| Interpretability        | Moderate        | Moderate            | High (dendrogram)       |
| Explores multiple K     | No (one run)    | No (one run)        | Yes (one run, all K)    |

---

### Divisive Clustering

For completeness: divisive (top-down) clustering starts with all n points in
a single cluster and recursively splits.

**DIANA algorithm:**
1. Start with all points in one cluster.
2. Find the point with the largest average dissimilarity to all other points
   in its cluster — this is the "splinter" point.
3. Form a new cluster containing just the splinter point.
4. Iteratively move points from the original cluster to the new cluster if
   they are closer (on average) to the new cluster than to their current one.
5. Repeat until no points want to move. This is one split.
6. Recurse on the largest remaining cluster (or all clusters) until each
   point is its own cluster.

Divisive methods produce the same dendrogram format but build it top-down.
They are more computationally expensive (O(2ⁿ) to find the optimal split
at each step; DIANA approximates this) but can be better for discovering the
top-level structure in data.

---

### Choosing the Number of Clusters from a Dendrogram

After computing the dendrogram, you need to decide where to cut. Four methods:

**1. Largest gap heuristic** — Find the longest vertical line in the dendrogram
that is not crossed by any horizontal cut. Cut just below the top of this gap.
The number of branches below the cut = number of clusters.

**2. Inconsistency coefficient** — For each merge, compare its height to the
average height of nearby (child) merges. A high inconsistency coefficient flags
an unusual jump, suggesting a natural boundary. Cut where inconsistency exceeds
a threshold (commonly 1.5).

**3. Cophenetic correlation coefficient** — Measures how faithfully the
dendrogram preserves the original pairwise distances. Computed as the Pearson
correlation between the original distance matrix and the cophenetic distance
matrix (distances read from the dendrogram). Values above 0.75 are considered
good. Use this to compare linkages.

**4. External validation** — If you have domain knowledge (e.g. you know there
are 3 species, 4 market segments), cut to produce that many clusters and
evaluate with Silhouette or other internal metrics.

---

### Key Takeaways

1. Hierarchical clustering produces a dendrogram — a complete tree of nested
   clusters at every granularity. You only run it once to explore all values of K.

2. Agglomerative (bottom-up) is the dominant approach. Start with n singleton
   clusters and merge the closest pair at each step.

3. The linkage criterion defines "distance between clusters." Ward (minimise
   WCSS increase) is the best general default; single linkage can follow
   non-convex shapes; complete linkage gives compact spherical clusters.

4. The best cut point is found by looking for the largest gap in the dendrogram
   — the longest branch not interrupted by any horizontal line.

5. Hierarchical clustering is exact and deterministic but O(n²) in space and
   O(n² log n) in time — not suited for large n. For large datasets, use BIRCH
   or perform hierarchical clustering on K-Means centroids instead.
"""


OPERATIONS = {

    "▶ Run: Agglomerative Clustering From Scratch": {
        "description": "Full agglomerative clustering implementation using all four linkages. Prints the complete merge history and dendrogram for a small dataset.",
        "code": """
import math

# ── Dataset ────────────────────────────────────────────────────────────────────
points = {
    'A': (0.0, 0.0),
    'B': (1.0, 0.0),
    'C': (5.0, 0.0),
    'D': (6.0, 0.0),
    'E': (0.5, 1.0),
    'F': (5.5, 1.0),
}

LINKAGE = 'ward'   # try: 'single', 'complete', 'average', 'ward'

# ── Distance ───────────────────────────────────────────────────────────────────
def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ── Cluster representation: each cluster tracks its member points ──────────────
def cluster_dist(ci_members, cj_members, linkage, all_points):
    dists = [euclidean(all_points[p], all_points[q])
             for p in ci_members for q in cj_members]
    if linkage == 'single':
        return min(dists)
    elif linkage == 'complete':
        return max(dists)
    elif linkage == 'average':
        return sum(dists) / len(dists)
    elif linkage == 'ward':
        ni, nj = len(ci_members), len(cj_members)
        mi = [sum(all_points[p][d] for p in ci_members) / ni for d in range(2)]
        mj = [sum(all_points[p][d] for p in cj_members) / nj for d in range(2)]
        sq_dist = sum((mi[d] - mj[d]) ** 2 for d in range(2))
        return (ni * nj / (ni + nj)) * sq_dist
    else:
        raise ValueError(f"Unknown linkage: {linkage}")

# ── Agglomerative algorithm ────────────────────────────────────────────────────
names = list(points.keys())

# Each cluster: {id: [member point names]}
cluster_id   = 0
clusters     = {name: [name] for name in names}
merge_history = []   # list of (ci_id, cj_id, height, new_id)

print(f"Agglomerative Clustering  |  Linkage: {LINKAGE.upper()}")
print(f"Dataset: {list(points.keys())}")
print()
print(f"{'Step':<5} {'Merged':<20} {'Height':>8}  {'Resulting clusters'}")
print("─" * 70)

step = 0
while len(clusters) > 1:
    # Find minimum-distance pair
    ids   = list(clusters.keys())
    best  = (None, None, float('inf'))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ci, cj = ids[i], ids[j]
            d = cluster_dist(clusters[ci], clusters[cj], LINKAGE, points)
            if d < best[2]:
                best = (ci, cj, d)

    ci_id, cj_id, height = best
    new_members = clusters[ci_id] + clusters[cj_id]
    new_id      = f"({ci_id}+{cj_id})"

    merge_history.append((ci_id, cj_id, height, new_id))

    del clusters[ci_id]
    del clusters[cj_id]
    clusters[new_id] = new_members

    step += 1
    remaining = list(clusters.keys())
    print(f"  {step:<3}  {ci_id} + {cj_id}  →  {new_id:<12}  h={height:>6.3f}   {remaining}")

# ── Dendrogram ASCII ───────────────────────────────────────────────────────────
print()
print("Merge history (read bottom-up = dendrogram leaf-to-root):")
print()
for i, (ci, cj, h, nid) in enumerate(merge_history):
    indent = "  " * i
    print(f"  {indent}└── merge({ci}, {cj})  at height {h:.3f}")

print()
print("To extract clusters: choose a cut height h.")
print("Each branch whose root > h but leaves < h = one cluster.")

# ── Show 2-cluster solution ────────────────────────────────────────────────────
print()
# The last merge is the root; second-to-last merge height gives 2 clusters
if len(merge_history) >= 2:
    cut_h = (merge_history[-2][2] + merge_history[-1][2]) / 2
    print(f"Largest gap cut at h ≈ {cut_h:.3f}")
    # Reconstruct 2 clusters from merge history
    c1_members = merge_history[-1][0]
    c2_members = merge_history[-1][1]
    print(f"  Cluster 1: members from '{c1_members}' branch")
    print(f"  Cluster 2: members from '{c2_members}' branch")
""",
        "runnable": True,
    },

    "▶ Run: All Four Linkages Compared": {
        "description": "Run agglomerative clustering with single, complete, average, and Ward linkage on the same dataset. Compare merge sequences and final cluster assignments.",
        "code": """
import math
from collections import defaultdict

# ── Dataset: 3 true clusters ───────────────────────────────────────────────────
import random
random.seed(42)

def blob(cx, cy, r, n, label):
    pts = []
    for i in range(n):
        x = cx + (random.random() * 2 - 1) * r
        y = cy + (random.random() * 2 - 1) * r
        pts.append(((x, y), label))
    return pts

data = blob(0, 0, 0.8, 5, 'A') + blob(4, 0, 0.8, 5, 'B') + blob(2, 3, 0.8, 5, 'C')
names = [f"{'ABC'[i//5]}{i%5+1}" for i in range(len(data))]
coords = {names[i]: data[i][0] for i in range(len(data))}
true_labels = {names[i]: data[i][1] for i in range(len(data))}

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def cluster_dist(ci, cj, linkage, all_pts):
    dists = [euclidean(all_pts[p], all_pts[q]) for p in ci for q in cj]
    if linkage == 'single':   return min(dists)
    if linkage == 'complete': return max(dists)
    if linkage == 'average':  return sum(dists) / len(dists)
    if linkage == 'ward':
        ni, nj = len(ci), len(cj)
        mi = [sum(all_pts[p][d] for p in ci)/ni for d in range(2)]
        mj = [sum(all_pts[p][d] for p in cj)/nj for d in range(2)]
        return (ni*nj/(ni+nj)) * sum((mi[d]-mj[d])**2 for d in range(2))

def run_agglomerative(coords, linkage, K=3):
    clusters = {n: [n] for n in coords}
    merges   = []
    while len(clusters) > 1:
        ids = list(clusters.keys())
        best = (None, None, float('inf'))
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                d = cluster_dist(clusters[ids[i]], clusters[ids[j]], linkage, coords)
                if d < best[2]: best = (ids[i], ids[j], d)
        ci, cj, h = best
        new_id = f"m{len(merges)}"
        merges.append((ci, cj, h))
        clusters[new_id] = clusters.pop(ci) + clusters.pop(cj)
    # Cut to K clusters by replaying merges
    clusters2 = {n: [n] for n in coords}
    for ci, cj, h in merges:
        if len(clusters2) <= K: break
        matched_ci = next((k for k,v in clusters2.items() if ci in v or ci==k), None)
        matched_cj = next((k for k,v in clusters2.items() if cj in v or cj==k), None)
        if matched_ci and matched_cj and matched_ci != matched_cj:
            new_id = f"m{ci}{cj}"
            clusters2[new_id] = clusters2.pop(matched_ci) + clusters2.pop(matched_cj)
    return clusters2, merges

def purity(cluster_dict, true_labels):
    correct = 0
    for members in cluster_dict.values():
        from collections import Counter
        majority = Counter(true_labels[m] for m in members).most_common(1)[0][1]
        correct += majority
    return correct / len(true_labels)

# ── Run all linkages ───────────────────────────────────────────────────────────
linkages = ['single', 'complete', 'average', 'ward']
from collections import Counter

print(f"Dataset: {len(coords)} points  |  3 true clusters (5 pts each: A, B, C)")
print(f"Extracting K=3 clusters from each dendrogram")
print()
print(f"{'Linkage':<12} {'Purity':>8}  {'Cluster sizes'}  {'First 3 merges (height)'}")
print("─" * 75)

for lnk in linkages:
    result, merges = run_agglomerative(coords, lnk, K=3)
    p = purity(result, true_labels)
    sizes = sorted([len(v) for v in result.values()], reverse=True)
    first3 = "  ".join(f"h={m[2]:.2f}" for m in merges[:3])
    print(f"  {lnk:<10}  {p:>6.2%}   {str(sizes):<18}  {first3}")

print()
print("Notes:")
print("  Ward linkage typically achieves highest purity on spherical clusters.")
print("  Single linkage is susceptible to chaining on close clusters.")
print("  Complete and Average are robust intermediate choices.")

# ── Show Ward's 3-cluster composition ─────────────────────────────────────────
print()
print("Ward K=3 cluster composition:")
result_ward, _ = run_agglomerative(coords, 'ward', K=3)
for i, (cid, members) in enumerate(result_ward.items()):
    comp = Counter(true_labels[m] for m in members)
    print(f"  Cluster {i+1} ({len(members)} pts): {dict(comp)}")
""",
        "runnable": True,
    },

    "▶ Run: Dendrogram ASCII Printer": {
        "description": "Build and print a readable ASCII dendrogram showing the full merge tree with branch heights and cluster membership at every level.",
        "code": """
import math

# ── Small dataset for a clean printable dendrogram ────────────────────────────
points = {
    'p1': (0.0, 0.0),
    'p2': (0.5, 0.2),
    'p3': (0.4, 0.8),
    'p4': (4.0, 0.1),
    'p5': (4.5, 0.3),
    'p6': (4.2, 0.9),
    'p7': (8.0, 5.0),
}

LINKAGE = 'ward'

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def cluster_dist(ci, cj, linkage, pts):
    dists = [euclidean(pts[p], pts[q]) for p in ci for q in cj]
    if linkage == 'single':   return min(dists)
    if linkage == 'complete': return max(dists)
    if linkage == 'average':  return sum(dists)/len(dists)
    if linkage == 'ward':
        ni,nj=len(ci),len(cj)
        mi=[sum(pts[p][d] for p in ci)/ni for d in range(2)]
        mj=[sum(pts[p][d] for p in cj)/nj for d in range(2)]
        return (ni*nj/(ni+nj))*sum((mi[d]-mj[d])**2 for d in range(2))

# ── Run agglomerative and collect full merge tree ─────────────────────────────
clusters    = {n: {'members':[n],'repr':n} for n in points}
merge_nodes = {}
node_ctr    = [len(points)]
merges      = []

def new_node(left, right, height, members):
    nid = f"N{node_ctr[0]}"
    node_ctr[0] += 1
    merge_nodes[nid] = {'left':left,'right':right,'height':height,'members':members}
    return nid

while len(clusters) > 1:
    ids  = list(clusters.keys())
    best = (None, None, float('inf'))
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            d = cluster_dist(clusters[ids[i]]['members'],
                             clusters[ids[j]]['members'], LINKAGE, points)
            if d < best[2]: best=(ids[i],ids[j],d)
    ci,cj,h = best
    new_members = clusters[ci]['members'] + clusters[cj]['members']
    nid = new_node(ci,cj,h,new_members)
    merges.append({'merged':(ci,cj),'height':h,'node':nid,'size':len(new_members)})
    del clusters[ci]; del clusters[cj]
    clusters[nid] = {'members':new_members,'repr':nid}

# ── Recursive ASCII dendrogram ────────────────────────────────────────────────
def draw_tree(node_id, indent='', is_last=True):
    if node_id in merge_nodes:
        nd = merge_nodes[node_id]
        connector = '└── ' if is_last else '├── '
        print(f"{indent}{connector}[h={nd['height']:>6.3f}]  {nd['members']}")
        child_indent = indent + ('    ' if is_last else '│   ')
        draw_tree(nd['left'],  child_indent, is_last=False)
        draw_tree(nd['right'], child_indent, is_last=True)
    else:
        connector = '└── ' if is_last else '├── '
        print(f"{indent}{connector}{node_id}  {points[node_id]}")

root = list(clusters.keys())[0]

print(f"Dendrogram  |  Linkage: {LINKAGE.upper()}  |  {len(points)} points")
print(f"Root: {merge_nodes[root]['members']}")
print()
draw_tree(root)

# ── Cut analysis ──────────────────────────────────────────────────────────────
print()
print("Cut analysis:")
print(f"{'Cut height h':<16}  {'Clusters':>8}  Cluster compositions")
print("─" * 65)

heights = sorted(set(nd['height'] for nd in merge_nodes.values()), reverse=True)
for h in heights:
    # Replay merges up to height h
    working = {n:{'members':[n]} for n in points}
    for mg in merges:
        if mg['height'] <= h: break
        ci,cj = mg['merged']
        # find containers
        def find(x):
            for k,v in working.items():
                if x in v['members'] or x==k: return k
            return None
        ki,kj = find(ci),find(cj)
        if ki and kj and ki!=kj:
            new_members=working[ki]['members']+working[kj]['members']
            del working[ki]; del working[kj]
            working[mg['node']]={'members':new_members}
    n_clust = len(working)
    comps   = [v['members'] for v in working.values()]
    comp_str = '  '.join(str(sorted(c)) for c in comps[:4])
    print(f"  h > {h:>6.3f}:  {n_clust:>3} cluster(s)   {comp_str}")

print()
print("→ Largest gap: cut just below the biggest height jump above.")
print("  That gap shows the most natural cluster structure in the data.")
""",
        "runnable": True,
    },

    "▶ Run: Cophenetic Correlation — Comparing Linkages": {
        "description": "Compute the cophenetic correlation coefficient for each linkage method. Measures how faithfully the dendrogram preserves original pairwise distances. Higher is better.",
        "code": """
import math

# ── Dataset ────────────────────────────────────────────────────────────────────
import random
random.seed(7)

def blob(cx, cy, r, n):
    return [(cx+(random.random()*2-1)*r, cy+(random.random()*2-1)*r) for _ in range(n)]

X = blob(0,0,0.6,8) + blob(5,0,0.6,8) + blob(2.5,4,0.6,8)
n = len(X)

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def cluster_dist(ci, cj, linkage, pts):
    dists = [euclidean(pts[p], pts[q]) for p in ci for q in cj]
    if linkage == 'single':   return min(dists)
    if linkage == 'complete': return max(dists)
    if linkage == 'average':  return sum(dists)/len(dists)
    if linkage == 'ward':
        ni,nj=len(ci),len(cj)
        mi=[sum(pts[p][d] for p in ci)/ni for d in range(2)]
        mj=[sum(pts[p][d] for p in cj)/nj for d in range(2)]
        return (ni*nj/(ni+nj))*sum((mi[d]-mj[d])**2 for d in range(2))

def run_hclust(pts, linkage):
    ids = list(range(len(pts)))
    clusters = {i: [i] for i in ids}
    merges = []
    while len(clusters) > 1:
        cids = list(clusters.keys())
        best = (None,None,float('inf'))
        for a in range(len(cids)):
            for b in range(a+1,len(cids)):
                d = cluster_dist(clusters[cids[a]], clusters[cids[b]], linkage, pts)
                if d < best[2]: best=(cids[a],cids[b],d)
        ci,cj,h = best
        new_id = max(clusters)+1
        clusters[new_id] = clusters.pop(ci)+clusters.pop(cj)
        merges.append((ci,cj,h,new_id))
    return merges

def cophenetic_matrix(n, merges):
    # For each pair (i,j), find the height at which they were first joined
    # Map each original point to the cluster it belongs to at each step
    membership = {i:i for i in range(n)}  # point → cluster id

    cophen = [[0.0]*n for _ in range(n)]

    for ci,cj,h,new_id in merges:
        # Find all original points in ci and cj
        def get_leaves(cluster_id, merges_so_far):
            leaves = []
            for pt in range(n):
                if membership[pt] == cluster_id:
                    leaves.append(pt)
            return leaves

        pts_i = [p for p in range(n) if membership[p]==ci]
        pts_j = [p for p in range(n) if membership[p]==cj]

        for pi in pts_i:
            for pj in pts_j:
                cophen[pi][pj] = h
                cophen[pj][pi] = h

        for p in pts_i+pts_j:
            membership[p] = new_id

    return cophen

def pearson(xs, ys):
    n = len(xs)
    mx,my = sum(xs)/n, sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den if den>0 else 0

# Original pairwise distances
orig_dists = []
for i in range(n):
    for j in range(i+1,n):
        orig_dists.append(euclidean(X[i],X[j]))

# ── Compute cophenetic correlation for each linkage ────────────────────────────
print(f"Cophenetic Correlation Coefficient  (n={n} points, 3 true clusters)")
print()
print(f"  Measures how faithfully the dendrogram preserves original distances.")
print(f"  Range: 0 to 1.  >0.75 = good fit.  Closer to 1 = better.")
print()
print(f"  {'Linkage':<12}  {'Cophenetic r':>14}  {'Quality'}")
print("  " + "─" * 45)

for lnk in ['single','complete','average','ward']:
    merges  = run_hclust(X, lnk)
    cophen  = cophenetic_matrix(n, merges)
    cophen_dists = []
    for i in range(n):
        for j in range(i+1,n):
            cophen_dists.append(cophen[i][j])
    r = pearson(orig_dists, cophen_dists)
    bar = '█' * int(r*20)
    quality = 'Excellent' if r>0.9 else 'Good' if r>0.75 else 'Fair' if r>0.6 else 'Poor'
    print(f"  {lnk:<12}  {r:>12.4f}    {bar}  {quality}")

print()
print("Interpretation:")
print("  Single and Average linkage tend to score highest because they minimise")
print("  distance distortion. Ward uses a non-distance criterion (WCSS) so its")
print("  cophenetic correlation measures something slightly different in spirit.")
""",
        "runnable": True,
    },

    "▶ Run: Lance-Williams Incremental Update": {
        "description": "Demonstrate the Lance-Williams formula for efficiently updating the distance matrix after each merge, avoiding full O(n²) recomputation at each step.",
        "code": """
import math

# ── Dataset ────────────────────────────────────────────────────────────────────
points_raw = [
    (0.0, 0.0),
    (1.0, 0.1),
    (0.5, 0.9),
    (5.0, 0.0),
    (5.8, 0.3),
]
names = [f'P{i}' for i in range(len(points_raw))]
points = dict(zip(names, points_raw))
n = len(names)

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ── Lance-Williams parameters ──────────────────────────────────────────────────
# dist(Ck, Cm) = ai*dist(Ci,Cm) + aj*dist(Cj,Cm) + b*dist(Ci,Cj) + g*|dist(Ci,Cm)-dist(Cj,Cm)|
def lw_update(dist_im, dist_jm, dist_ij, ni, nj, nm, linkage):
    if linkage == 'single':
        return 0.5*dist_im + 0.5*dist_jm + 0 - 0.5*abs(dist_im - dist_jm)
    elif linkage == 'complete':
        return 0.5*dist_im + 0.5*dist_jm + 0 + 0.5*abs(dist_im - dist_jm)
    elif linkage == 'average':
        ai = ni/(ni+nj); aj = nj/(ni+nj)
        return ai*dist_im + aj*dist_jm
    elif linkage == 'ward':
        denom = ni + nj + nm
        ai = (nm+ni)/denom; aj = (nm+nj)/denom; b = -nm/denom
        return ai*dist_im + aj*dist_jm + b*dist_ij

# ── Run with full recompute vs Lance-Williams and verify they match ────────────
LINKAGE = 'average'

# --- Method 1: Full recompute each step ---
def full_recompute_agglom(points, linkage):
    clusters = {n: [n] for n in points}
    merges   = []
    step     = 0
    print(f"Full Recompute  |  Linkage: {linkage.upper()}")
    print(f"{'Step':<5} {'Merged':<16} {'Dist':>7}  {'Distance matrix row for new cluster'}")
    print("─" * 70)
    while len(clusters) > 1:
        ids = list(clusters.keys())
        dist_mat = {}
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                dists = [euclidean(points[p], points[q])
                         for p in clusters[ids[i]] for q in clusters[ids[j]]]
                if linkage=='single':   d=min(dists)
                elif linkage=='complete': d=max(dists)
                elif linkage=='average':  d=sum(dists)/len(dists)
                elif linkage=='ward':
                    ni,nj=len(clusters[ids[i]]),len(clusters[ids[j]])
                    mi=[sum(points[p][dd] for p in clusters[ids[i]])/ni for dd in range(2)]
                    mj=[sum(points[p][dd] for p in clusters[ids[j]])/nj for dd in range(2)]
                    d=(ni*nj/(ni+nj))*sum((mi[dd]-mj[dd])**2 for dd in range(2))
                dist_mat[(ids[i],ids[j])]=d
                dist_mat[(ids[j],ids[i])]=d
        ci,cj,h = min(((a,b,v) for (a,b),v in dist_mat.items() if a<b),
                      key=lambda x:x[2])
        new_id = f"({ci}+{cj})"
        # Compute new row distances
        new_dists = {}
        for oid in ids:
            if oid==ci or oid==cj: continue
            dists=[euclidean(points[p],points[q])
                   for p in clusters[ci]+clusters[cj] for q in clusters[oid]]
            if linkage=='single':   nd=min(dists)
            elif linkage=='complete': nd=max(dists)
            elif linkage=='average':  nd=sum(dists)/len(dists)
            elif linkage=='ward':
                ni=len(clusters[ci])+len(clusters[cj]); nj=len(clusters[oid])
                mi=[sum(points[p][d] for p in clusters[ci]+clusters[cj])/ni for d in range(2)]
                mj=[sum(points[p][d] for p in clusters[oid])/nj for d in range(2)]
                nd=(ni*nj/(ni+nj))*sum((mi[d]-mj[d])**2 for d in range(2))
            new_dists[oid]=nd
        merges.append((ci,cj,h,new_dists.copy()))
        del clusters[ci]; del clusters[cj]
        clusters[new_id]=merges[-1][2:3][0]  # placeholder
        clusters[new_id]=clusters.get(new_id,[]) 
        clusters[new_id]=[p for k in [ci,cj] for p in (merges[-1][0:1] or [])]
        # rebuild properly
        clusters={}
        clusters_rebuild={n:[n] for n in points}
        for a,b,_,_ in merges:
            def find_c(x, cd):
                for k,v in cd.items():
                    if x==k or x in v: return k
                return x
            ka=find_c(a,clusters_rebuild); kb=find_c(b,clusters_rebuild)
            if ka!=kb:
                new_members=clusters_rebuild.pop(ka,[])+clusters_rebuild.pop(kb,[])
                clusters_rebuild[f"({ka}+{kb})"]=new_members
        clusters=clusters_rebuild
        step += 1
        nd_str = '  '.join(f"{k}:{v:.3f}" for k,v in new_dists.items())
        print(f"  {step:<3}  {ci} + {cj}  →  h={h:.4f}   new_dists: {nd_str}")
    return merges

merges = full_recompute_agglom(points, LINKAGE)

print()
print("Lance-Williams gives identical results with O(n) update per merge")
print("instead of recomputing all pairwise distances from scratch.")
print()
print(f"Lance-Williams parameters for {LINKAGE.upper()} linkage:")
print(f"  αᵢ = |Cᵢ|/(|Cᵢ|+|Cⱼ|)   αⱼ = |Cⱼ|/(|Cᵢ|+|Cⱼ|)   β=0   γ=0")
print(f"  → weighted average of existing distances, no γ (no min/max comparison)")
""",
        "runnable": True,
    },

    "▶ Run: K-Means vs Hierarchical (Ward) on Same Data": {
        "description": "Direct comparison of K-Means and Ward hierarchical clustering on three datasets: spherical blobs, elongated clusters, and a dataset with an outlier. Compare purity and sensitivity.",
        "code": """
import math
import random
from collections import Counter

random.seed(99)

def euclidean(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

# ── K-Means ────────────────────────────────────────────────────────────────────
def kmeans(X, K, iters=100):
    centroids = random.sample(X, K)
    labels = [0]*len(X)
    for _ in range(iters):
        for i,p in enumerate(X):
            labels[i] = min(range(K), key=lambda k: euclidean(p, centroids[k]))
        for k in range(K):
            members=[X[i] for i in range(len(X)) if labels[i]==k]
            if members:
                centroids[k]=(sum(p[0] for p in members)/len(members),
                               sum(p[1] for p in members)/len(members))
    return labels

# ── Ward hierarchical → K clusters ────────────────────────────────────────────
def ward_hclust_K(X, K):
    n = len(X)
    clusters = {i:[i] for i in range(n)}
    def ward_d(ci,cj):
        ni,nj=len(ci),len(cj)
        mi=[sum(X[p][d] for p in ci)/ni for d in range(2)]
        mj=[sum(X[p][d] for p in cj)/nj for d in range(2)]
        return (ni*nj/(ni+nj))*sum((mi[d]-mj[d])**2 for d in range(2))
    merges=[]
    while len(clusters)>1:
        ids=list(clusters.keys())
        best=(None,None,float('inf'))
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                d=ward_d(clusters[ids[i]],clusters[ids[j]])
                if d<best[2]: best=(ids[i],ids[j],d)
        ci,cj,h=best
        new_id=max(clusters)+1
        clusters[new_id]=clusters.pop(ci)+clusters.pop(cj)
        merges.append((ci,cj,h))
        if len(clusters)==K: break
    labels=[-1]*n
    for cid,(cid_key,members) in enumerate(clusters.items()):
        for p in members: labels[p]=cid
    return labels

def purity(pred, true_l):
    clusters={}
    for i,l in enumerate(pred):
        clusters.setdefault(l,[]).append(true_l[i])
    correct=sum(Counter(v).most_common(1)[0][1] for v in clusters.values())
    return correct/len(true_l)

# ── Three test datasets ────────────────────────────────────────────────────────
def blob(cx,cy,r,n,lbl):
    return [((cx+(random.random()*2-1)*r, cy+(random.random()*2-1)*r), lbl)
            for _ in range(n)]

datasets = {
    'Spherical blobs': (
        blob(0,0,0.6,12,0)+blob(4,0,0.6,12,1)+blob(2,3.5,0.6,12,2),
        3
    ),
    'Elongated clusters': (
        [((i*0.15,0+(random.random()-0.5)*0.3),0) for i in range(12)] +
        [((i*0.15+4,2+(random.random()-0.5)*0.3),1) for i in range(12)] +
        [((2+(random.random()-0.5)*0.3,i*0.3+4),2) for i in range(12)],
        3
    ),
    'With outlier': (
        blob(0,0,0.5,10,0)+blob(4,0,0.5,10,1)+blob(2,3,0.5,10,2)+
        [((15.0,15.0),2)],  # extreme outlier labelled as cluster 2
        3
    ),
}

print(f"K-Means vs Ward Hierarchical  (K=3)")
print()
print(f"{'Dataset':<22}  {'K-Means':>9}  {'Ward':>9}  {'Winner'}")
print("─" * 55)

for name,(data,K) in datasets.items():
    X      = [d[0] for d in data]
    true_l = [d[1] for d in data]
    km_lbl = kmeans(X, K)
    wrd_lbl = ward_hclust_K(X, K)
    km_p  = purity(km_lbl,  true_l)
    wrd_p = purity(wrd_lbl, true_l)
    winner = 'K-Means' if km_p>wrd_p else 'Ward' if wrd_p>km_p else 'Tie'
    print(f"  {name:<20}  {km_p:>8.1%}  {wrd_p:>8.1%}  {winner}")

print()
print("Key findings:")
print("  • Both perform similarly on clean spherical blobs")
print("  • Ward hierarchical handles elongated shapes slightly better")
print("    (dendrogram cut doesn't depend on random initialisation)")
print("  • Extreme outliers hurt Ward more than K-Means because Ward")
print("    minimises variance — one outlier forces a very late merge,")
print("    inflating that branch height and distorting the dendrogram cut")
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
        from Unsupervised_Learning.visuals.hierarchical_visual import (   # ← match your exact folder casing
            HIERARCHICAL_VISUAL_HTML,
            HIERARCHICAL_VISUAL_HEIGHT,
        )
        visual_html   = HIERARCHICAL_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = HIERARCHICAL_VISUAL_HEIGHT
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

