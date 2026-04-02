import textwrap
import re

"""Module: 08 · Anomaly Detection"""

TOPIC_NAME   ="Anomaly Detection"
DISPLAY_NAME = "08 · Anomaly Detection"
ICON         = "🚨"
SUBTITLE     = "Isolation Forest, One-Class SVM — finding the outliers"

THEORY = r"""
## 08 · Anomaly Detection

---

### What Is Anomaly Detection?

**Anomaly detection** (also called outlier detection or novelty detection) is the
task of identifying data points that are significantly different from the majority
of the data — points that do not conform to an expected pattern.

Anomalies go by several names: outliers, novelties, exceptions, aberrations,
contaminants. The distinction between them matters:

**Outlier detection** — the training data itself is assumed to contain some
anomalies. The goal is to separate the majority (inliers) from the minority
(outliers) within the same dataset. Suitable when you don't have clean training
data.

**Novelty detection** — the training data is assumed to be clean (all normal).
The goal is to flag new test points that are different from the training
distribution. Suitable when you can collect a clean normal dataset first.

**Why unsupervised?** In most real applications there are no labels. Normal
data is abundant; anomalies are rare, unexpected, and by definition hard to
enumerate in advance (you cannot label what you have not yet seen).

---

### Types of Anomalies

**Point anomaly** — a single data point is anomalous compared to the rest.
Example: a credit card transaction of $50,000 when all others are under $500.

**Contextual anomaly** — a point is anomalous only within a specific context
(e.g. time window, user, location). Example: 30°C is normal in summer but
anomalous in winter for the same location.

**Collective anomaly** — a group of related points is anomalous collectively,
even if each individual point might appear normal. Example: a specific sequence
of network packets that individually look fine but together constitute an attack.

Most algorithms address point anomalies. Contextual and collective anomalies
require domain-specific feature engineering or sequence models.

---

### Evaluation Without Labels

When labels are available (rare), standard metrics apply: precision, recall,
F1, AUC-ROC, AUC-PR. Because anomalies are rare, class imbalance makes
**precision-recall curves more informative than ROC curves**.

**Contamination rate** — the estimated fraction of anomalies in the data.
Most algorithms use this as a hyperparameter to set the decision threshold.
Typical values: 0.01 to 0.10.

When no labels exist, validation is qualitative: domain experts examine
flagged points and assess plausibility.

---

### Isolation Forest

Isolation Forest (Liu, Ting, Zhou 2008) is one of the most widely used
anomaly detection algorithms. It is fast, scalable, and works well in
high dimensions — properties most other methods lack.

#### Core Intuition

Normal points lie in dense regions and are hard to isolate — you need many
random cuts to separate one from its neighbours. Anomalous points lie in
sparse, outlying regions and are easy to isolate — just a few random cuts
suffice.

Isolation Forest exploits this by measuring **how few cuts it takes to
isolate each point**. Anomalies get isolated in fewer steps.

#### Building an Isolation Tree

    1. Randomly select a feature j from {1, ..., p}
    2. Randomly select a split value v uniformly between
       [min(feature j), max(feature j)]
    3. Split the data: points with x[j] < v go left, rest go right
    4. Recurse on each half until:
       - only 1 point remains (fully isolated), or
       - the tree reaches a maximum depth h_max = ceil(log2(n))

This produces a binary tree where each leaf contains one or a few points.
The **path length** of a point x is the number of edges traversed from the
root to the leaf where x ends up. Short path = easy to isolate = likely anomaly.

#### The Anomaly Score

Build a forest of T isolation trees, each on a random subsample of size psi
(typically psi = 256, T = 100). The average path length over all trees is:

    E[h(x)] = average path length of x across T trees

Longer average paths mean the point is hard to isolate (normal).
Shorter paths mean easy to isolate (anomalous).

Normalise into an anomaly score in [0, 1]:

    s(x, psi) = 2^{ -E[h(x)] / c(psi) }

where c(psi) is the expected path length of an unsuccessful binary search tree
over psi points (the average path length assuming random partitioning):

    c(n) = 2 * H(n-1) - 2*(n-1)/n

    H(k) = ln(k) + 0.5772 (Euler-Mascheroni constant)  [harmonic number approximation]

Interpretation of s(x):
- s ≈ 1  → very short path → highly anomalous
- s ≈ 0  → very long path  → clearly normal
- s ≈ 0.5 → indistinguishable from random (no anomaly signal)

If ALL points have s ≈ 0.5, there are likely no anomalies in the dataset.

#### Why Subsampling Helps

Using a small subsample (psi=256) rather than the full dataset:
1. **Speed**: building and querying a tree on 256 points is O(256 log 256)
   regardless of n — scales to millions of points.
2. **Accuracy**: with psi=256, the path length difference between normal
   and anomalous points is maximised. Larger subsamples actually hurt by
   allowing normal points' trees to become deeper, obscuring the signal.
   This is the "swamping" and "masking" problem that subsampling avoids.

#### Algorithm Summary

    Input: data X (n x p), n_trees T, subsample_size psi, contamination rate

    Build phase:
      For t = 1 to T:
        Sample psi points from X without replacement
        Build isolation tree on the subsample (max_depth = ceil(log2(psi)))

    Score phase:
      For each point x in X:
        For t = 1 to T:
          Traverse tree t and record path length h_t(x)
        E[h(x)] = mean(h_t(x) over t)
        s(x) = 2^{ -E[h(x)] / c(psi) }

    Threshold: set contamination fraction * n as the number of anomalies
    Anomalies: the top contamination_rate * n points by score s(x)

---

### One-Class SVM (OCSVM)

One-Class SVM (Schölkopf et al. 2001) takes a different approach: it learns
a **decision boundary** that encloses the normal data in feature space.
Points outside this boundary are anomalies.

#### The Core Idea

Standard SVM separates two classes by finding a maximum-margin hyperplane.
One-Class SVM has only one class (normal data) and must find a hypersphere
(or hyperplane in the kernel-induced feature space) that:

- Contains as many normal points as possible
- Has the smallest volume possible (tightest fit)

#### Formulation

In the kernelised feature space F defined by kernel k(x, x'), find a
hyperplane w · Φ(x) = ρ that separates the data from the origin with maximum
margin, while allowing a fraction ν of points to fall below the hyperplane
(support vectors on the wrong side = outlier budget):

    Minimise:   (1/2)||w||² - ρ + (1/νn) Σᵢ ξᵢ
    Subject to: w · Φ(xᵢ) ≥ ρ - ξᵢ,   ξᵢ ≥ 0

where ν ∈ (0, 1] is the key hyperparameter:
- ν is an upper bound on the fraction of outliers
- ν is a lower bound on the fraction of support vectors

The decision function for a new point x:

    f(x) = sign(w · Φ(x) - ρ)
         = sign( Σᵢ αᵢ k(xᵢ, x) - ρ )

    f(x) = +1 → normal (inside the boundary)
    f(x) = -1 → anomaly (outside the boundary)

#### The Role of the Kernel

The choice of kernel determines the shape of the decision boundary in the
original input space:

**RBF (Radial Basis Function) kernel:**

    k(x, z) = exp(-γ ||x-z||²)

Most common choice. Maps data to an infinite-dimensional space; the decision
boundary can be arbitrarily non-linear and can wrap tightly around complex
data clusters.

**Linear kernel:** k(x, z) = x^T z
Hyperplane in original space. Works for linearly separable normal data.

**Polynomial kernel:** k(x, z) = (x^T z + c)^d
Polynomial decision boundary of degree d.

#### Hyperparameter γ (RBF kernel)

γ controls how far the influence of each training point reaches:
- Small γ: each point influences a large neighbourhood → smooth, loose boundary
- Large γ: each point influences only very nearby points → tight, jagged boundary
  that may overfit

#### OCSVM vs Isolation Forest

| Property                | Isolation Forest           | One-Class SVM (RBF)              |
|-------------------------|----------------------------|----------------------------------|
| Speed (training)        | O(T * psi * log(psi))      | O(n² to n³)                      |
| Speed (inference)       | O(T * log(psi))            | O(n_sv * p)                      |
| Scales to large n       | Excellent (psi=256)        | Poor (kernel matrix n x n)       |
| High-dimensional data   | Good                       | Poor (curse of dimensionality)   |
| Boundary shape          | Implicit (path lengths)    | Explicit hypersphere/surface     |
| Hyperparameters         | contamination, n_trees     | nu, gamma, kernel                |
| Interpretability        | Path length, feature split | Support vectors, margin          |
| Anomaly score           | Continuous [0, 1]          | Signed distance to boundary      |
| Best for                | Large datasets, tabular    | Small n, low-dim, dense clusters |

---

### Statistical Methods: Z-Score and IQR

For univariate data, classical statistical methods are often sufficient:

**Z-Score method:**

    z(x) = (x - μ) / σ

A point is anomalous if |z(x)| > threshold (typically 2.5–3.5).
Assumes normally distributed data. Sensitive to the outliers distorting μ and σ.
Use robust variants: replace μ with median, σ with MAD (median absolute deviation).

**IQR (Interquartile Range) method:**

    IQR = Q3 - Q1  (75th percentile minus 25th percentile)
    Lower fence = Q1 - k * IQR
    Upper fence = Q3 + k * IQR

Points outside [lower fence, upper fence] are flagged as outliers.
Standard k = 1.5 (Tukey's fence). More conservative: k = 3.0.
Robust to non-normal distributions. Used in boxplot whiskers.

**Mahalanobis distance** — multivariate generalisation of Z-score:

    D_M(x) = sqrt( (x - μ)^T Σ^{-1} (x - μ) )

Accounts for correlations between features. A point is anomalous if
D_M(x) > chi² threshold. Assumes multivariate Gaussian distribution.
Breaks down for non-Gaussian or high-dimensional data.

---

### Local Outlier Factor (LOF)

LOF (Breunig et al. 2000) detects anomalies based on the **local density**
of each point relative to its neighbours — it is a density-based method
that handles clusters of varying density.

For each point x:

1. Find its k nearest neighbours N_k(x).

2. Compute the **reachability distance**:

    reach_dist(x, o) = max( k-dist(o), dist(x, o) )

    where k-dist(o) is the distance from o to its k-th nearest neighbour.
    This smooths out local fluctuations.

3. Compute the **local reachability density** (LRD):

    lrd(x) = 1 / ( sum_{o in N_k(x)} reach_dist(x, o) / |N_k(x)| )

    High LRD = x is in a dense neighbourhood. Low LRD = x is sparse.

4. Compute the **LOF score**:

    LOF(x) = ( sum_{o in N_k(x)} lrd(o) / lrd(x) ) / |N_k(x)|

    LOF ≈ 1: similar density to neighbours → normal
    LOF >> 1: x is in a much sparser region than its neighbours → anomalous

LOF is effective when anomalies exist in regions of lower density than
their surrounding normal clusters. It handles multi-density datasets
where a global threshold would incorrectly flag low-density clusters.

---

### Practical Workflow

**Step 1: Preprocessing**
- Normalise features (StandardScaler or MinMaxScaler).
- Handle missing values (impute before anomaly detection).
- For high-dimensional data, consider PCA or autoencoders for dimensionality
  reduction before applying LOF or OCSVM.

**Step 2: Algorithm selection**
- Large n (>10,000): Isolation Forest.
- Small n, low dim, one tight cluster: OCSVM with RBF.
- Multiple density clusters: LOF.
- Univariate streams: Z-score / IQR with rolling windows.

**Step 3: Set contamination rate**
- Use domain knowledge: "we expect ~2% fraud".
- If unknown, start with 0.05 and tune by examining flagged points.

**Step 4: Threshold calibration**
- With labels: optimise F1 or precision@k on a validation set.
- Without labels: examine the score distribution for a natural gap between
  normal and anomalous scores, or use a fixed percentile.

**Step 5: Post-processing**
- Ensemble multiple detectors and use majority vote or average score.
- Present flagged points to domain experts for confirmation.
- Track false-positive rate if deployed in a real-time system.

---

### Key Takeaways

1. Anomaly detection is unsupervised because anomalies are rare, unknown in
   advance, and often unlabelled. The goal is to flag points that deviate
   from the normal data distribution.

2. Isolation Forest isolates points with random axis-aligned cuts. Anomalies
   are isolated in fewer cuts (short path length) → high anomaly score.
   Fast, scalable, works in high dimensions.

3. One-Class SVM finds a kernel-induced hypersphere enclosing the normal
   data. Points outside = anomalies. Effective for small, low-dimensional
   datasets with a well-defined normal cluster.

4. LOF measures local density relative to neighbours. Effective for
   multi-density datasets where global thresholds would fail.

5. Always validate flagged anomalies with domain experts. The contamination
   rate is a critical hyperparameter — tune it with domain knowledge.
"""


OPERATIONS = {

    "▶ Run: Isolation Forest From Scratch": {
        "description": "Full Isolation Forest implementation from first principles. Builds isolation trees with random "
                       "feature splits, computes average path lengths, normalises into anomaly scores. Tests on injected "
                       "anomalies.",
        "code": r'''
import math, random
random.seed(42)

# Isolation tree node
class INode:
    def __init__(self, split_feat=None, split_val=None,
                 left=None, right=None, size=0, depth=0):
        self.split_feat = split_feat
        self.split_val  = split_val
        self.left       = left
        self.right      = right
        self.size       = size      # leaf: number of points
        self.depth      = depth

def build_tree(X, current_depth, max_depth):
    n, p = len(X), len(X[0])
    # Termination: too few points or max depth reached
    if n <= 1 or current_depth >= max_depth:
        return INode(size=n, depth=current_depth)
    # Random feature and split value
    feat = random.randint(0, p-1)
    col  = [X[i][feat] for i in range(n)]
    mn, mx = min(col), max(col)
    if mn == mx:
        return INode(size=n, depth=current_depth)
    split = random.uniform(mn, mx)
    left_X  = [X[i] for i in range(n) if X[i][feat] < split]
    right_X = [X[i] for i in range(n) if X[i][feat] >= split]
    if not left_X or not right_X:
        return INode(size=n, depth=current_depth)
    return INode(
        split_feat=feat, split_val=split,
        left=build_tree(left_X,  current_depth+1, max_depth),
        right=build_tree(right_X, current_depth+1, max_depth),
        depth=current_depth
    )

# Expected path length for BST of n points
def c(n):
    if n <= 1: return 0.0
    if n == 2: return 1.0
    H = math.log(n-1) + 0.5772156649  # harmonic number approximation
    return 2*H - 2*(n-1)/n

def path_length(x, node, current_length):
    # Leaf node
    if node.split_feat is None:
        return current_length + c(node.size)
    if x[node.split_feat] < node.split_val:
        return path_length(x, node.left,  current_length+1)
    else:
        return path_length(x, node.right, current_length+1)

def anomaly_score(x, trees, psi):
    avg_path = sum(path_length(x, t, 0) for t in trees) / len(trees)
    return 2.0 ** (-avg_path / c(psi))

# Dataset: 2D, normal cluster + injected anomalies
N_NORMAL   = 80
N_ANOMALY  = 8
PSI        = 64     # subsample size per tree
N_TREES    = 50
MAX_DEPTH  = math.ceil(math.log2(PSI))

# Normal: tight cluster around origin
X_normal  = [[random.gauss(0, 0.8), random.gauss(0, 0.8)] for _ in range(N_NORMAL)]
# Anomalies: scattered far away
X_anomaly = [[random.uniform(4, 7)*random.choice([-1,1]),
              random.uniform(4, 7)*random.choice([-1,1])]
             for _ in range(N_ANOMALY)]
X_all    = X_normal + X_anomaly
true_anom = [False]*N_NORMAL + [True]*N_ANOMALY
n_total   = len(X_all)

print(f"Isolation Forest From Scratch")
print(f"n={n_total} ({N_NORMAL} normal, {N_ANOMALY} anomaly)  "
      f"T={N_TREES} trees  psi={PSI}  max_depth={MAX_DEPTH}")
print()

# Build forest
trees = []
for _ in range(N_TREES):
    subsample = random.sample(X_all, PSI)
    trees.append(build_tree(subsample, 0, MAX_DEPTH))

# Score all points
scores = [anomaly_score(x, trees, PSI) for x in X_all]

# Threshold at top 10% by score
threshold_idx = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
contamination = N_ANOMALY / n_total
n_flag = max(1, int(contamination * n_total))
flagged = set(threshold_idx[:n_flag*2])  # flag top 2× for recall

# Evaluate
tp = sum(1 for i in range(n_total) if i in flagged and true_anom[i])
fp = sum(1 for i in range(n_total) if i in flagged and not true_anom[i])
fn = sum(1 for i in range(n_total) if i not in flagged and true_anom[i])
tn = sum(1 for i in range(n_total) if i not in flagged and not true_anom[i])
prec = tp/(tp+fp) if (tp+fp) else 0
rec  = tp/(tp+fn) if (tp+fn) else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0

print(f"Results (top {len(flagged)} points flagged as anomalies):")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
print()

# Score distribution
normal_scores  = [scores[i] for i in range(N_NORMAL)]
anomaly_scores = [scores[i] for i in range(N_NORMAL, n_total)]
print(f"Anomaly score distribution:")
print(f"  Normal  points: mean={sum(normal_scores)/len(normal_scores):.4f}  "
      f"max={max(normal_scores):.4f}  min={min(normal_scores):.4f}")
print(f"  Anomaly points: mean={sum(anomaly_scores)/len(anomaly_scores):.4f}  "
      f"max={max(anomaly_scores):.4f}  min={min(anomaly_scores):.4f}")
print()

# Top flagged points
print("Top 10 highest-scoring points (most anomalous):")
print(f"  {'Rank':>5}  {'x':>8}  {'y':>8}  {'Score':>8}  {'True label':>12}")
print("  " + "-"*48)
for rank, i in enumerate(threshold_idx[:10]):
    lbl = "ANOMALY" if true_anom[i] else "normal"
    print(f"  {rank+1:>5}  {X_all[i][0]:>8.3f}  {X_all[i][1]:>8.3f}  "
          f"{scores[i]:>8.4f}  {lbl:>12}")
''',
        "runnable": True,
    },

    "▶ Run: Z-Score and IQR Outlier Detection": {
        "description": "Implement Z-score, robust Z-score (median/MAD), and IQR methods from scratch on univariate data. "
                       "Compare sensitivity to non-Gaussian distributions and the effect of the outliers on each method.",
        "code": r'''
import math, random
random.seed(1)

# Dataset: mostly normal, a few outliers
data_normal  = [random.gauss(50, 5) for _ in range(95)]
data_outlier = [1.0, 3.0, 120.0, 130.0, 150.0]   # obvious outliers
data = data_normal + data_outlier
n    = len(data)

def mean(x):    return sum(x)/len(x)
def std(x):
    m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def median(x):
    s=sorted(x); mid=len(s)//2
    return (s[mid-1]+s[mid])/2 if len(s)%2==0 else s[mid]
def mad(x):
    m=median(x); return median([abs(v-m) for v in x])
def percentile(x, p):
    s=sorted(x); idx=p/100*(len(s)-1)
    lo,hi=int(idx),min(int(idx)+1,len(s)-1)
    return s[lo]+(idx-lo)*(s[hi]-s[lo])

print("Outlier Detection: Z-Score vs Robust Z-Score vs IQR")
print(f"n={n} ({len(data_normal)} normal Gaussian(50,5), {len(data_outlier)} injected outliers)")
print()

# Method 1: Standard Z-score
mu, sigma = mean(data), std(data)
z_threshold = 2.5
z_outliers = [i for i,v in enumerate(data) if abs((v-mu)/sigma) > z_threshold]
print(f"Method 1: Z-score  |  mu={mu:.2f}, sigma={sigma:.2f}  |  threshold |z|>{z_threshold}")
print(f"  Flagged {len(z_outliers)} points: "
      + ", ".join(f"{data[i]:.1f}" for i in sorted(z_outliers)[:10]))
print(f"  True outliers recovered: "
      + str(sum(1 for i in z_outliers if i >= len(data_normal)))
      + f"/{len(data_outlier)}")
print()

# Method 2: Robust Z-score (median/MAD)
med    = median(data)
mad_v  = mad(data)
# MAD scale factor: 1.4826 converts MAD to ~sigma for Gaussian
scale  = 1.4826 * mad_v
rz_threshold = 2.5
rz_outliers  = [i for i,v in enumerate(data) if abs((v-med)/scale) > rz_threshold]
print(f"Method 2: Robust Z-score  |  median={med:.2f}, MAD={mad_v:.2f}  |  threshold>{rz_threshold}")
print(f"  Flagged {len(rz_outliers)} points: "
      + ", ".join(f"{data[i]:.1f}" for i in sorted(rz_outliers)[:10]))
print(f"  True outliers recovered: "
      + str(sum(1 for i in rz_outliers if i >= len(data_normal)))
      + f"/{len(data_outlier)}")
print()

# Method 3: IQR
q1   = percentile(data, 25)
q3   = percentile(data, 75)
iqr  = q3 - q1
k    = 1.5
lo   = q1 - k*iqr
hi   = q3 + k*iqr
iqr_outliers = [i for i,v in enumerate(data) if v < lo or v > hi]
print(f"Method 3: IQR  |  Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}  |  k={k}")
print(f"  Fences: [{lo:.2f}, {hi:.2f}]")
print(f"  Flagged {len(iqr_outliers)} points: "
      + ", ".join(f"{data[i]:.1f}" for i in sorted(iqr_outliers)[:10]))
print(f"  True outliers recovered: "
      + str(sum(1 for i in iqr_outliers if i >= len(data_normal)))
      + f"/{len(data_outlier)}")
print()

# Impact of outliers on mean/std
data_clean = data_normal[:]
print("Impact of outliers on descriptive statistics:")
print(f"  {'Stat':<10}  {'With outliers':>16}  {'Without outliers':>18}  {'Change':>8}")
print("  " + "-"*58)
for stat, fn_ in [("Mean", mean), ("Std", std),
                   ("Median", median), ("MAD", mad)]:
    with_o = fn_(data)
    without = fn_(data_clean)
    pct = (with_o-without)/abs(without)*100 if without else 0
    print(f"  {stat:<10}  {with_o:>16.4f}  {without:>18.4f}  {pct:>7.1f}%")
print()
print("Key finding:")
print("  Standard Z-score uses mean/std which are DISTORTED by the outliers")
print("  themselves — flagging may miss outliers or flag normals.")
print("  Robust Z-score (median/MAD) and IQR are not affected by outliers.")
''',
        "runnable": True,
    },

    "▶ Run: Local Outlier Factor (LOF)": {
        "description": "Implement LOF from scratch. Demonstrates how LOF detects outliers in datasets with clusters of "
                       "different densities — cases where global methods like Z-score fail.",
        "code": r'''
import math, random
random.seed(7)

def dist(a, b): return math.sqrt(sum((a[j]-b[j])**2 for j in range(len(a))))

def knn(data, i, k):
    ds = sorted((dist(data[i],data[j]),j) for j in range(len(data)) if j!=i)
    return [j for _,j in ds[:k]], ds[k-1][0]  # neighbours, k-dist

def lof_scores(data, k=5):
    n = len(data)
    # k-NN and k-distance for all points
    nbrs  = [None]*n; kdists = [0.0]*n
    for i in range(n):
        nbrs[i], kdists[i] = knn(data, i, k)

    # Reachability distance: reach_dist(i, o) = max(k-dist(o), dist(i,o))
    def reach_dist(i, o):
        return max(kdists[o], dist(data[i], data[o]))

    # Local reachability density
    lrd = [0.0]*n
    for i in range(n):
        avg_rd = sum(reach_dist(i,o) for o in nbrs[i]) / k
        lrd[i] = 1.0 / max(avg_rd, 1e-10)

    # LOF score
    lof = [0.0]*n
    for i in range(n):
        lof[i] = sum(lrd[o] for o in nbrs[i]) / (k * lrd[i])
    return lof

# Dataset: two clusters of different density + outliers
# Dense cluster A
clusterA = [[random.gauss(0, 0.3), random.gauss(0, 0.3)] for _ in range(25)]
# Loose cluster B (same number of points, sparser)
clusterB = [[random.gauss(6, 1.5), random.gauss(0, 1.5)] for _ in range(25)]
# True outliers: isolated points
outliers  = [[12,5],[13,-4],[-5,5],[-4,-5]]
data = clusterA + clusterB + outliers
n    = len(data)
true_outlier = [False]*50 + [True]*4

print("Local Outlier Factor (LOF)  |  k=5 neighbours")
print(f"n={n}: 25 dense cluster A + 25 sparse cluster B + 4 true outliers")
print()

scores = lof_scores(data, k=5)

# Separate scores by group
scA   = scores[:25]
scB   = scores[25:50]
scOut = scores[50:]

print(f"  {'Group':<22}  {'Mean LOF':>10}  {'Max LOF':>10}  Notes")
print("  " + "-"*58)
print(f"  {'Dense cluster A':<22}  {sum(scA)/len(scA):>10.3f}  "
      f"{max(scA):>10.3f}  LOF near 1 = normal")
print(f"  {'Sparse cluster B':<22}  {sum(scB)/len(scB):>10.3f}  "
      f"{max(scB):>10.3f}  LOF still near 1 despite sparsity")
print(f"  {'True outliers':<22}  {sum(scOut)/len(scOut):>10.3f}  "
      f"{max(scOut):>10.3f}  LOF >> 1 = detected")
print()

# Show that Z-score FAILS on this dataset (cluster B looks like outliers)
xs = [p[0] for p in data]
mu = sum(xs)/n; sd = math.sqrt(sum((x-mu)**2 for x in xs)/n)
z_outliers = [i for i,p in enumerate(data) if abs((p[0]-mu)/sd)>2.5 or abs((p[1]-mu)/sd)>2.5]
print("Comparison — global Z-score on same data:")
print(f"  Z-score flags {len(z_outliers)} points")
z_true = sum(1 for i in z_outliers if true_outlier[i])
z_false = len(z_outliers) - z_true
print(f"  True outliers caught: {z_true}/4  |  False positives: {z_false}")
print(f"  (Z-score mistakes sparse-cluster-B points for outliers)")
print()

# LOF threshold at 1.5
lof_flagged = [i for i,s in enumerate(scores) if s > 1.5]
lof_true  = sum(1 for i in lof_flagged if true_outlier[i])
lof_false = len(lof_flagged) - lof_true
print(f"LOF (threshold > 1.5) flags {len(lof_flagged)} points")
print(f"  True outliers caught: {lof_true}/4  |  False positives: {lof_false}")
print()
print("LOF key insight:")
print("  Cluster B is sparse globally but LOCALLY NORMAL (its members are")
print("  similar to each other). LOF compares each point to ITS OWN neighbours'")
print("  density, not the global density. This lets LOF ignore the cluster's")
print("  overall sparsity and focus on truly isolated points.")
''',
        "runnable": True,
    },

    "▶ Run: Comparing Anomaly Detectors": {
        "description": "Head-to-head comparison of Isolation Forest, Z-score, IQR, and LOF on three different anomaly "
                       "scenarios: point outliers, cluster outliers, and multi-density data.",
        "code": r'''
import math, random
random.seed(42)

def dist(a,b): return math.sqrt(sum((a[j]-b[j])**2 for j in range(len(a))))
def mean(x): return sum(x)/len(x)
def std(x): m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def median(x): s=sorted(x); mid=len(s)//2; return (s[mid-1]+s[mid])/2 if len(s)%2==0 else s[mid]
def mad(x): m=median(x); return median([abs(v-m) for v in x])
def percentile(x,p):
    s=sorted(x); idx=p/100*(len(s)-1); lo,hi=int(idx),min(int(idx)+1,len(s)-1)
    return s[lo]+(idx-lo)*(s[hi]-s[lo])

# Isolation tree
def c_n(n):
    if n<=1: return 0.0
    if n==2: return 1.0
    return 2*(math.log(n-1)+0.5772156649)-2*(n-1)/n

def build_itree(X, d, max_d):
    n,p=len(X),len(X[0])
    if n<=1 or d>=max_d: return {'leaf':True,'size':n}
    f=random.randint(0,p-1); col=[X[i][f] for i in range(n)]
    mn,mx=min(col),max(col)
    if mn==mx: return {'leaf':True,'size':n}
    s=random.uniform(mn,mx)
    L=[X[i] for i in range(n) if X[i][f]<s]; R=[X[i] for i in range(n) if X[i][f]>=s]
    if not L or not R: return {'leaf':True,'size':n}
    return {'leaf':False,'f':f,'s':s,'L':build_itree(L,d+1,max_d),'R':build_itree(R,d+1,max_d)}

def path_len(x, node, l):
    if node['leaf']: return l+c_n(node['size'])
    return path_len(x,node['L'],l+1) if x[node['f']]<node['s'] else path_len(x,node['R'],l+1)

def iforest_scores(X, n_trees=40, psi=32):
    max_d=math.ceil(math.log2(max(psi,2)))
    trees=[build_itree(random.sample(X,min(psi,len(X))),0,max_d) for _ in range(n_trees)]
    return [2**(-mean([path_len(x,t,0) for t in trees])/c_n(psi)) for x in X]

def lof_scores(data, k=5):
    n=len(data)
    nbrs=[]; kdists=[]
    for i in range(n):
        ds=sorted((dist(data[i],data[j]),j) for j in range(n) if j!=i)
        nbrs.append([j for _,j in ds[:k]]); kdists.append(ds[k-1][0])
    def rd(i,o): return max(kdists[o],dist(data[i],data[o]))
    lrd=[0.0]*n
    for i in range(n): lrd[i]=1/max(sum(rd(i,o) for o in nbrs[i])/k,1e-10)
    return [sum(lrd[o] for o in nbrs[i])/(k*lrd[i]) for i in range(n)]

def evaluate(pred_scores, true_anom, n_flag):
    top=sorted(range(len(pred_scores)),key=lambda i:pred_scores[i],reverse=True)
    flagged=set(top[:n_flag])
    tp=sum(1 for i in flagged if true_anom[i])
    fp=sum(1 for i in flagged if not true_anom[i])
    fn=sum(1 for i in range(len(true_anom)) if i not in flagged and true_anom[i])
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    return prec, rec, f1

# Three scenarios
SCENARIOS = []

# Scenario 1: Point outliers in 1D Gaussian
s1_normal  = [random.gauss(50,5) for _ in range(90)]
s1_anomaly = [1.0, 2.0, 120.0, 125.0, 130.0]
s1_data    = [[v] for v in s1_normal+s1_anomaly]
s1_true    = [False]*90+[True]*5
SCENARIOS.append(("Point outliers (1D Gaussian)", s1_data, s1_true, 5))

# Scenario 2: Two 2D clusters + outliers in between
s2_A = [[random.gauss(-3,0.5),random.gauss(0,0.5)] for _ in range(40)]
s2_B = [[random.gauss( 3,0.5),random.gauss(0,0.5)] for _ in range(40)]
s2_out = [[random.uniform(-1,1),random.uniform(-1,1)] for _ in range(6)]
s2_data = s2_A+s2_B+s2_out
s2_true = [False]*80+[True]*6
SCENARIOS.append(("Between-cluster outliers (2D)", s2_data, s2_true, 6))

# Scenario 3: Dense + sparse cluster + isolated outliers
s3_A = [[random.gauss(0,0.3),random.gauss(0,0.3)] for _ in range(30)]
s3_B = [[random.gauss(6,1.2),random.gauss(0,1.2)] for _ in range(30)]
s3_out = [[13,6],[14,-5],[-7,7]]
s3_data= s3_A+s3_B+s3_out
s3_true= [False]*60+[True]*3
SCENARIOS.append(("Multi-density clusters (2D)", s3_data, s3_true, 3))

print("Anomaly Detector Comparison  |  F1 scores")
print()
print(f"  {'Scenario':<32}  {'IF':>8}  {'Z-score':>8}  {'IQR':>8}  {'LOF':>8}")
print("  " + "-"*68)

for name, data, true_anom, n_flag in SCENARIOS:
    # Isolation Forest
    if_sc = iforest_scores(data)

    # Z-score (using first feature)
    vals = [p[0] for p in data]
    m,s  = mean(vals),std(vals)
    z_sc = [abs((v-m)/max(s,1e-9)) for v in vals]

    # IQR
    q1,q3 = percentile(vals,25),percentile(vals,75)
    iqr_v = q3-q1
    iqr_sc = [max(0,(v-q3)/max(iqr_v,1e-9)) if v>q3 else
               max(0,(q1-v)/max(iqr_v,1e-9)) if v<q1 else 0 for v in vals]

    # LOF
    lof_sc = lof_scores(data, k=min(5,len(data)-1))

    pr_if  = evaluate(if_sc,  true_anom, n_flag*2)[2]
    pr_z   = evaluate(z_sc,   true_anom, n_flag*2)[2]
    pr_iqr = evaluate(iqr_sc, true_anom, n_flag*2)[2]
    pr_lof = evaluate(lof_sc, true_anom, n_flag*2)[2]

    best = max(pr_if,pr_z,pr_iqr,pr_lof)
    def fmt(v): return f"*{v:.3f}*" if v==best else f" {v:.3f} "
    print(f"  {name:<32}  {fmt(pr_if)}  {fmt(pr_z)}  {fmt(pr_iqr)}  {fmt(pr_lof)}")

print()
print("  * = best F1 for that scenario")
print()
print("Expected findings:")
print("  Isolation Forest: robust across all scenarios, especially high-dim")
print("  Z-score/IQR: excellent for simple 1D Gaussian cases")
print("  LOF: best for multi-density data where global methods fail")
''',
        "runnable": True,
    },

    "▶ Run: Anomaly Score Threshold Calibration": {
        "description": "Show how the contamination rate parameter controls the detection threshold. Plot precision-recall "
                       "tradeoff as the threshold varies, and find the optimal threshold with and without labels.",
        "code": r'''
import math, random
random.seed(5)

def c_n(n):
    if n<=1: return 0.0
    if n==2: return 1.0
    return 2*(math.log(n-1)+0.5772156649)-2*(n-1)/n

def build_itree(X,d,md):
    n,p=len(X),len(X[0])
    if n<=1 or d>=md: return {'l':True,'s':n}
    f=random.randint(0,p-1); col=[x[f] for x in X]
    mn,mx=min(col),max(col)
    if mn==mx: return {'l':True,'s':n}
    sv=random.uniform(mn,mx)
    L=[x for x in X if x[f]<sv]; R=[x for x in X if x[f]>=sv]
    if not L or not R: return {'l':True,'s':n}
    return {'l':False,'f':f,'sv':sv,'L':build_itree(L,d+1,md),'R':build_itree(R,d+1,md)}

def pl(x,node,l):
    if node['l']: return l+c_n(node['s'])
    return pl(x,node['L'],l+1) if x[node['f']]<node['sv'] else pl(x,node['R'],l+1)

PSI=64; N_T=60; MD=math.ceil(math.log2(PSI))
def iforest(X):
    trees=[build_itree(random.sample(X,min(PSI,len(X))),0,MD) for _ in range(N_T)]
    return [2**(-sum(pl(x,t,0) for t in trees)/N_T/c_n(PSI)) for x in X]

# Dataset with known contamination
N_NOR=100; N_ANO=10
X_nor = [[random.gauss(0,0.8),random.gauss(0,0.8)] for _ in range(N_NOR)]
X_ano = [[random.uniform(3,6)*random.choice([-1,1]),
          random.uniform(3,6)*random.choice([-1,1])] for _ in range(N_ANO)]
X_all = X_nor+X_ano
n     = len(X_all)
true  = [0]*N_NOR + [1]*N_ANO
scores = iforest(X_all)

# Sort descending by score
order  = sorted(range(n), key=lambda i: scores[i], reverse=True)

print(f"Threshold Calibration  |  n={n} ({N_NOR} normal, {N_ANO} anomaly)")
print(f"True contamination rate: {N_ANO/n:.2f}")
print()

# Precision-recall at each possible threshold
print("Precision-Recall curve (sweeping threshold from tight to loose):")
print(f"  {'Cont. rate':>12}  {'Flagged':>8}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
print("  " + "-"*50)

prev_f1 = 0; best_thresh = 0; best_f1_row = None
for n_flag in [3, 5, 7, 10, 12, 15, 18, 20, 25, 30]:
    flagged = set(order[:n_flag])
    tp = sum(1 for i in flagged if true[i]==1)
    fp = len(flagged)-tp
    fn = N_ANO-tp
    prec=tp/(tp+fp) if tp+fp else 0
    rec =tp/(tp+fn) if tp+fn else 0
    f1  =2*prec*rec/(prec+rec) if prec+rec else 0
    cont=n_flag/n
    marker=" <-- best F1" if f1>prev_f1 else ""
    if f1>prev_f1: prev_f1=f1; best_f1_row=(n_flag,prec,rec,f1)
    print(f"  {cont:>12.3f}  {n_flag:>8}  {prec:>8.3f}  {rec:>8.3f}  {f1:>8.3f}{marker}")

print()
print(f"Best F1 at contamination = {best_f1_row[0]/n:.2f}  "
      f"(Prec={best_f1_row[1]:.3f}  Rec={best_f1_row[2]:.3f}  F1={best_f1_row[3]:.3f})")
print()

# Score distribution gap
sorted_scores = sorted(scores, reverse=True)
print("Score distribution (top 30 scores descending):")
for i, s in enumerate(sorted_scores[:30]):
    bar = "█"*int(s*40)
    lbl = "ANO" if true[order[i]]==1 else "nor"
    print(f"  {i+1:>3} [{lbl}] {s:.4f} {bar}")

print()
print("Calibration strategy without labels:")
print("  1. Look for a natural GAP in the score distribution (scores drop sharply)")
print("  2. Use domain knowledge: 'we expect ~X% anomalies in this system'")
print("  3. If deployed, monitor false-positive rate from human review feedback")
''',
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
        from Unsupervised_Learning.visuals.anomaly_visual import (   # ← match your exact folder casing
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

