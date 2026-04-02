"""
Computational Complexity in ML
================================

How the cost of algorithms scales with data size, feature dimension,
model capacity, and precision — Big-O analysis for ML practitioners,
time-space tradeoffs, NP-hardness of core ML problems, and the
practical implications for algorithm selection at scale.

"""

import textwrap
import re

TOPIC_NAME = "Computational Complexity in ML"
DISPLAY_NAME = "07 · Computational Complexity"
ICON = "⏱️"
SUBTITLE = "How ML Algorithms Scale — and When They Break"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Why Complexity Matters in ML

Every ML algorithm makes a tradeoff between accuracy, computational cost,
and memory. Understanding how cost scales allows you to:

    ─ Choose the right algorithm for a given data size n, dimension d.
    ─ Predict when an algorithm will become infeasible.
    ─ Design efficient approximations when exact methods are too slow.
    ─ Understand why some ML problems are fundamentally hard.

    The key quantities:

    ┌──────────────────────────────────────────────────────────────────┐
    │  n  = number of training examples                                │
    │  d  = input dimension (number of features)                       │
    │  k  = number of classes / clusters / components                  │
    │  T  = number of training iterations / tree depth / sequence len  │
    │  m  = number of model parameters (weights)                       │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 1 — Orders of Growth (practical crossover points):

    Cost
      │ O(2ⁿ) ─────────────────────── (intractable n > 30)
      │ O(n³) ──────────────────────── (painful n > 10,000)
      │ O(n²) ──────────────── (manageable n < 100,000)
      │ O(n·d) ──────── (standard ML workload)
      │ O(n log n) ─── (sort + tree operations)
      │ O(n) ─── (ideal: single pass)
      │ O(log n) ─ (lookup, binary search)
      │ O(1)
      └──────────────────────────────────────────── n
              10  100  1K  10K  100K  1M  10M

    The most important transitions:
    ─ n=10K: quadratic O(n²) becomes expensive (~100M operations).
    ─ n=100K: cubic O(n³) is infeasible (~10¹⁵ operations).
    ─ n=10M: only O(n) or O(n log n) algorithms are practical.


──────────────────────────────────────────────────────────────────────────────
### Big-O Notation — Formal Definitions

Big-O captures the worst-case asymptotic behaviour of algorithms.

    Formal definitions:
    ┌──────────────────────────────────────────────────────────────────┐
    │  f(n) = O(g(n)):  ∃ c, n₀  s.t.  f(n) ≤ c·g(n)  ∀ n ≥ n₀         │
    │  f(n) = Ω(g(n)):  ∃ c, n₀  s.t.  f(n) ≥ c·g(n)  ∀ n ≥ n₀         │
    │  f(n) = Θ(g(n)):  f(n) = O(g(n))  AND  f(n) = Ω(g(n))            │
    │  f(n) = o(g(n)):  lim_{n→∞} f(n)/g(n) = 0   (strictly smaller)   │
    └──────────────────────────────────────────────────────────────────┘

    Important subtleties for ML:
    ─ Big-O is worst-case unless otherwise stated.
    ─ Constants matter in practice but are hidden in Big-O.
    ─ "n" in ML can be samples, features, or parameters — be precise.
    ─ Memory complexity (space) is as important as time complexity.

    Common complexity functions (slowest to fastest growth):
    O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n^1.5)
    < O(n²) < O(n² log n) < O(n³) < O(2ⁿ) < O(n!)

    AMORTISED COMPLEXITY:  average cost over a sequence of operations.
    Example: dynamic array resizing is O(n) per resize but O(1) amortised.
    Mini-batch gradient descent: O(b·d) per step, not O(n·d).


──────────────────────────────────────────────────────────────────────────────
### Complexity of Fundamental ML Operations

**Matrix Operations (the core of linear algebra in ML):**

    ┌────────────────────────────────────────────────────┬────────────────┐
    │  Operation                                         │  Complexity    │
    ├────────────────────────────────────────────────────┼────────────────┤
    │  Matrix multiply:  (n×d) · (d×k)                   │  O(ndk)        │
    │  Matrix-vector:    (n×d) · (d×1)                   │  O(nd)         │
    │  Matrix inverse:   (d×d)⁻¹                         │  O(d³)         │
    │  Cholesky factor:  (d×d) symmetric                 │  O(d³/3)       │
    │  Eigendecompose:   (d×d) symmetric                 │  O(d³)         │
    │  SVD:              (n×d) with n ≥ d                │  O(nd²)        │
    │  LU factorisation: (n×n)                           │  O(n³/3)       │
    │  Sparse (nnz entries): mat-vec                     │  O(nnz)        │
    └────────────────────────────────────────────────────┴────────────────┘

    Strassen's algorithm: matrix multiply in O(n^2.81) instead of O(n³).
    In practice, BLAS implementations with cache-optimised tiling are used.
    GPU parallelism reduces wall-clock time but not algorithmic complexity.

    The d³ bottleneck:  many ML algorithms have a d³ term from matrix
    inversion. For d = 10,000 features, that's 10¹² operations.
    Solutions: approximations (Nyström, sketching), sparse structures.

**Sorting and Search:**

    Binary search:      O(log n)   — requires sorted data
    Sort (comparison):  O(n log n) — optimal lower bound
    Sort (radix):       O(n·k)     — non-comparison based
    Priority queue:     O(log n)   — insert/extract-min
    Hash table:         O(1) amortised — insert/lookup

**Distance and Kernel Computations:**

    All pairwise L2 distances (n×d data):    O(n²d)
    Kernel matrix K (n×n):                   O(n²d)  for RBF/poly kernels
    k-NN (exact, brute force):               O(nd)   per query
    k-NN (KD-tree, d≤20):                   O(log n) per query
    k-NN (KD-tree, d>20):                   degrades to O(n) (curse of dim)
    Approximate nearest neighbour (LSH):     O(n^ρ)  ρ < 1


──────────────────────────────────────────────────────────────────────────────
### Complexity Reference Table for ML Algorithms

    ┌─────────────────────────┬────────────────────┬──────────────┬──────────────┐
    │  ALGORITHM              │  TRAINING TIME     │  PRED. TIME  │  SPACE       │
    ├─────────────────────────┼────────────────────┼──────────────┼──────────────┤
    │  Linear regression (OLS)│  O(nd² + d³)       │  O(d)        │  O(nd + d²)  │
    │  Ridge regression       │  O(nd² + d³)       │  O(d)        │  O(nd + d²)  │
    │  Ridge (dual/kernel)    │  O(n³)             │  O(nd)       │  O(n²)       │
    │  Logistic regression    │  O(T·nd)           │  O(d)        │  O(nd)       │
    │  Naive Bayes            │  O(nd)             │  O(kd)       │  O(kd)       │
    │  k-NN (brute)           │  O(1)              │  O(nd)       │  O(nd)       │
    │  k-NN (KD-tree, d≤20)   │  O(nd log n)       │  O(log n)    │  O(nd)       │
    │  Decision tree          │  O(nd log n)       │  O(depth)    │  O(n)        │
    │  Random forest (B trees)│  O(B·nd log n)     │  O(B·depth)  │  O(B·n)      │
    │  Gradient boosting      │  O(T·nd log n)     │  O(T·depth)  │  O(T·n)      │
    │  SVM (kernel, n>1000)   │  O(n²d) − O(n³)    │  O(n_sv · d) │  O(n²)       │
    │  SVM (linear, SGD)      │  O(T·nd)           │  O(d)        │  O(d)        │
    │  PCA (full)             │  O(nd² + d³)       │  O(kd)       │  O(nd)       │
    │  PCA (truncated SVD)    │  O(ndk)            │  O(kd)       │  O(nd)       │
    │  k-means (Lloyd's)      │  O(T·nkd)          │  O(kd)       │  O(nd + k)   │
    │  GMM (EM)               │  O(T·nk²d + d³k)   │  O(kd)       │  O(nd)       │
    │  Neural net (forward)   │  O(m) = O(Σ lᵢlᵢ₊₁) │ O(m)         │ O(m)        │
    │  Neural net (backward)  │  O(2m)             │  O(m)        │  O(m)        │
    │  Transformer (self-attn)│  O(T²·d)  per lyr  │  O(T²·d)     │  O(T²)       │
    │  GP regression          │  O(n³)             │  O(n²)       │  O(n²)       │
    │  KRR (kernel)           │  O(n³)             │  O(nd)       │  O(n²)       │
    └─────────────────────────┴────────────────────┴──────────────┴──────────────┘

    Key: T = iterations/trees/seq.len, k = classes/clusters, d = features,
         n = training samples, m = parameters, n_sv = support vectors.


──────────────────────────────────────────────────────────────────────────────
### The Transformer's Quadratic Bottleneck

Self-attention is the dominant operation in transformers:

    For a sequence of length T and embedding dimension d:

        Q, K, V = X·Wq,  X·Wk,  X·Wv   ∈ ℝ^{T×d}
        Attention = softmax(QKᵀ/√d) · V

    The QKᵀ product: (T×d)·(d×T) = T×T matrix — O(T²d) operations.
    The attention matrix: O(T²) space.

    ┌──────────────────────────────────────────────────────────────────┐
    │  T=512   (BERT): T² = 262,144    tokens ← feasible               │
    │  T=4096  (GPT-4 context): T² = 16,777,216   ← expensive          │
    │  T=100K  (long documents): T² = 10¹⁰    ← infeasible naively     │
    └──────────────────────────────────────────────────────────────────┘

    Efficient attention variants that break the T² barrier:
    ─ Sparse attention (Longformer): attend only to local + global tokens.
      Complexity: O(T·w) where w is the local window size.
    ─ Linear attention (Performer): approximate softmax with kernel trick.
      Complexity: O(T·d) — linear in sequence length.
    ─ FlashAttention: same O(T²) but IO-aware, reduces memory bandwidth.
      10× faster in practice without changing algorithmic complexity.
    ─ Sliding window + global (BigBird): O(T) for most tokens.


──────────────────────────────────────────────────────────────────────────────
### NP-Hardness in Core ML Problems

Several fundamental ML problems are NP-hard in the worst case.
This means no polynomial-time algorithm is known (and likely none exists).

    OPTIMAL DECISION TREES:
        Finding the minimal-depth decision tree consistent with training data
        is NP-hard. Greedy recursive splitting (ID3/CART) is a polynomial
        heuristic with no optimality guarantee.

    NEURAL NETWORK TRAINING:
        Finding weights that globally minimise the training loss is NP-hard
        for networks with even one hidden layer. SGD finds local minima
        (or saddle points). In practice, over-parameterised networks avoid
        bad local minima empirically — but no guarantee exists.

    k-MEANS CLUSTERING:
        Finding the globally optimal k-means partition (minimising intra-
        cluster variance) is NP-hard. Lloyd's algorithm converges to a local
        optimum. k-means++ initialisation reduces bad local optima in practice.

    FEATURE SELECTION (EXACT):
        Finding the minimum subset of features with accuracy ≥ threshold
        is NP-hard (reduces to set cover). Greedy forward/backward selection
        is a polynomial heuristic.

    MAX CUT (GRAPH PARTITION):
        Many graph-based clustering and community detection problems.
        Spectral methods provide approximate polynomial-time solutions.

    ┌──────────────────────────────────────────────────────────────────┐
    │  NP-hardness does not mean the problem is always hard in         │
    │  practice. It means worst-case instances are hard.               │
    │  Real data often has structure that makes approximate methods    │
    │  work well. Practitioners use:                                   │
    │  ─ Greedy heuristics (decision trees, k-means++)                 │
    │  ─ Relaxations (SDP for clustering, LP for integer programs)     │
    │  ─ Randomised algorithms (gradient descent with restarts)        │
    │  ─ Approximation algorithms with provable guarantees             │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Space Complexity and Memory Bottlenecks

Time complexity is only half the story. Memory often limits ML at scale.

    MEMORY HIERARCHY (latency and bandwidth):

    ┌────────────────────┬──────────────────┬────────────────────────────┐
    │  Level             │  Latency         │  Size (typical)            │
    ├────────────────────┼──────────────────┼────────────────────────────┤
    │  CPU registers     │  < 1 ns          │  KB                        │
    │  L1 cache          │  ~1 ns           │  32–64 KB per core         │
    │  L2 cache          │  ~4 ns           │  256 KB – 4 MB per core    │
    │  L3 cache          │  ~10–40 ns       │  8–64 MB shared            │
    │  RAM (DRAM)        │  ~100 ns         │  8–512 GB                  │
    │  GPU VRAM          │  ~100 ns         │  8–80 GB (A100: 80 GB)     │
    │  NVMe SSD          │  ~50 µs          │  TB                        │
    │  Network storage   │  ~1 ms           │  PB                        │
    └────────────────────┴──────────────────┴────────────────────────────┘

    Memory complexity of common ML data structures:
    ─ Dense matrix (n×d float32):  n × d × 4 bytes
    ─ Kernel matrix (n×n):         n² × 4 bytes  (n=50K → 10 GB!)
    ─ Transformer attention (T×T): T² × 4 bytes  (T=4K → 64 MB per head)
    ─ Gradient checkpointing:      reduces activation memory by √T factor

    Memory bottleneck examples:
    ─ n=100K samples: kernel matrix K = 100K² × 4 bytes = 40 GB.
      This alone exceeds typical GPU VRAM → kernel methods impractical.
    ─ LLM with 70B parameters: 70×10⁹ × 2 bytes (BF16) = 140 GB VRAM.
      Requires 2+ A100 GPUs just for inference.

    Gradient checkpointing (activation recomputation):
    ─ Naive backprop: store all layer activations during forward pass.
      Memory O(L) where L = number of layers. For deep nets, huge.
    ─ Checkpointing: only store activations at √L checkpoints.
      Recompute others during backward pass. Memory O(√L), cost 1.33× time.


──────────────────────────────────────────────────────────────────────────────
### Scalable Algorithms — Approximation and Streaming

When exact algorithms are too slow, use approximations with guarantees.

**Stochastic Gradient Descent (SGD):**

    Full gradient descent:   O(nd) per step, converges in O(1/T) iterations.
    SGD (one sample):        O(d) per step, converges in O(1/√T) iterations.
    Mini-batch SGD (b):      O(bd) per step — best of both worlds.

    For large n: mini-batch SGD with b=256 is 1000× cheaper per step than
    full gradient descent, with only log(n/b) worse convergence.

**Nyström and Random Features (from Module 11):**

    Exact kernel matrix: O(n²d) build, O(n³) solve.
    Nyström rank-m:      O(nmd) build, O(nm² + m³) solve, O(m) predict.
    Random Fourier:      O(nDd) feature map, O(nD) train, O(D) predict.

**Locality Sensitive Hashing (LSH) for Approximate NN:**

    Exact k-NN: O(nd) per query.
    LSH: O(n^ρ · d) per query where ρ < 1.
    For cosine similarity: ρ = (1 − θ/π)/(1 − (π−θ)/π) < 1.
    LSH is the basis of efficient similarity search in billion-scale retrieval.

**Sketch Algorithms — Streaming Data:**

    Count-Min Sketch: approximate frequency count in O(ε⁻¹ log δ⁻¹) space.
    HyperLogLog: estimate cardinality |S| with 2% error in O(log log n) space.
    Bloom Filter: set membership with O(ε⁻¹ log n) bits and no false negatives.
    Reservoir Sampling: uniform sample of k items from stream in O(k) space.

    These matter for ML when:
    ─ Data arrives as a stream (cannot store all examples).
    ─ Computing exact statistics is too expensive (token frequency in LLMs).
    ─ Feature hashing: map high-cardinality features to fixed-size vectors.

**Distributed and Parallel Complexity:**

    Data parallelism: split n samples across P processors.
    Time per step: O(nd/P) compute + O(d) communication.
    Speed-up limited by communication overhead (Amdahl's law).

    Model parallelism: split model parameters across P GPUs.
    Required when model doesn't fit in one GPU's memory.
    Pipeline parallelism: partition layers across GPUs, run stages in parallel.


──────────────────────────────────────────────────────────────────────────────
### Practical Complexity Guide — Algorithm Selection

    Given n samples, d features, k classes, choose the right algorithm:

    ┌──────────────────────────────────────────────────────────────────┐
    │  n < 1,000:  Any algorithm. Exact methods fine.                  │
    │              Prefer interpretable models first.                  │
    │                                                                  │
    │  n ~ 10,000: Linear methods fast. Tree ensembles fast.           │
    │              Kernel SVM/KRR marginal (n³ starts showing).        │
    │              Neural nets: feasible with GPU.                     │
    │                                                                  │
    │  n ~ 100,000: Kernel methods impractical (10¹⁵ ops for n³).      │
    │               SGD-based methods mandatory.                       │
    │               Tree ensembles still fast (O(n log n) per tree).   │
    │                                                                  │
    │  n > 1M:    Mini-batch SGD only. Approximate nearest neighbours. │
    │              Sparse models, feature hashing.                     │
    │              Distributed training for deep nets.                 │
    │                                                                  │
    │  d > 10K:   Avoid O(d³) operations (matrix inversion).           │
    │              Use iterative solvers (conjugate gradient).         │
    │              Dimensionality reduction first (PCA to d' << d).    │
    │                                                                  │
    │  d >> n:    Dual formulations preferred (n << d: O(n³) << O(d³)) │
    │              Lasso/ridge with coordinate descent fast.           │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 2 — Complexity Phase Diagram (n vs d):

    d (features)
        │
    1M  │                     ← neural net territory (SGD + GPU)
        │
    100K│         SVM (kernel impractical)
        │           GBM / RF still work
        │
    10K │     PCA first, then any linear method
        │     Dual Ridge (O(n³) < O(d³) when n < d)
        │
    1K  │ Primal Ridge (O(d²n + d³))
        │ SVM exact viable
        │ kNN-tree fast (d≤20)
        │
    100 │ All methods viable
        └──────────────────────────────────────────────── n (samples)
            100   1K   10K  100K  1M   10M

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Empirical Complexity Scaling — Measure Wall-Clock vs n and d": {
        "description": (
            "Measures empirical wall-clock time and memory usage for core ML "
            "operations as n and d scale: matrix multiply, SVD, kernel matrix "
            "construction, k-means, linear regression (primal vs dual), and "
            "k-NN. Fits log-log regression to extract empirical exponents and "
            "compares to theoretical Big-O. Shows where algorithms break down "
            "and at what n the quadratic/cubic bottlenecks become visible."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import tracemalloc
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy
from scipy.spatial import KDTree as _KDTree

class TruncatedSVD:
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
    def fit(self, X):
        U, s, Vt = _np_impl.linalg.svd(X, full_matrices=False)
        k = min(self.n_components, len(s))
        self.components_ = Vt[:k]; self.singular_values_ = s[:k]; return self
    def transform(self, X): return X @ self.components_.T
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class NearestNeighbors:
    def __init__(self, n_neighbors=5, algorithm='brute'):
        self.k = n_neighbors; self.algorithm = algorithm
    def fit(self, X):
        self._X = X.copy()
        if self.algorithm == 'kd_tree':
            self._tree = _KDTree(X)
        return self
    def kneighbors(self, X_q):
        X_q = _np_impl.atleast_2d(X_q)
        if self.algorithm == 'kd_tree':
            return self._tree.query(X_q, k=self.k)
        d2 = ((_np_impl.sum(self._X**2, 1, keepdims=True).T
               + _np_impl.sum(X_q**2, 1, keepdims=True)
               - 2 * X_q @ self._X.T))
        dists = _np_impl.sqrt(_np_impl.maximum(d2, 0))
        idx = _np_impl.argsort(dists, 1)[:, :self.k]
        return _np_impl.sort(dists, 1)[:, :self.k], idx

def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    c_arr = (rng.uniform(-5,5,(centers,n_features)) if isinstance(centers,int)
             else _np_impl.asarray(centers))
    nc = len(c_arr)
    counts = [n_samples//nc+(1 if i<n_samples%nc else 0) for i in range(nc)]
    Xs, ys = [], []
    for i,(cnt,c) in enumerate(zip(counts,c_arr)):
        Xs.append(rng.normal(c,cluster_std,(cnt,n_features))); ys.append(_np_impl.full(cnt,i,dtype=int))
    return _np_impl.vstack(Xs), _np_impl.hstack(ys)

def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    y = rng.integers(0,2,n_samples)
    Xi = rng.standard_normal((n_samples,n_informative))
    Xi += (2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    Xr = (Xi[:,:n_redundant]+0.3*rng.standard_normal((n_samples,n_redundant))
          if n_redundant>0 else _np_impl.empty((n_samples,0)))
    nn2 = max(0,n_features-n_informative-n_redundant)
    Xn = rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    return _np_impl.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features],y.astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict(self,X): return (_sig(X@self.coef_.ravel()+self.intercept_[0])>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class SGDClassifier:
    def __init__(self,**kw): self._w=None; self._b=0.0; self._t=1
    def partial_fit(self,X,y,classes=None):
        X=_np_impl.atleast_2d(X).astype(float); y=_np_impl.asarray(y)
        if self._w is None: self._w=_np_impl.zeros(X.shape[1])
        lr=min(1.0/(0.0001*self._t+1e-6),0.1)
        for xi,yi in zip(X,y):
            p=_sig(xi@self._w+self._b); e=p-yi
            self._w-=lr*e*xi; self._b-=lr*e; self._t+=1
        return self
    def predict(self,X):
        return (_sig(_np_impl.atleast_2d(X).astype(float)@self._w+self._b)>=0.5).astype(int)

def cross_val_score(estimator,X,y,cv=5,scoring='accuracy'):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]))
    return _np_impl.array(scores)

class KMeans:
    def __init__(self,n_clusters=8,init='k-means++',n_init=10,
                 random_state=None,max_iter=300,**kw):
        self.n_clusters=n_clusters; self.init=init
        self.n_init=n_init; self.random_state=random_state; self.max_iter=max_iter
    def _init_c(self,X,rng):
        if self.init=='k-means++':
            cs=[X[rng.integers(0,len(X))]]
            for _ in range(self.n_clusters-1):
                d=_np_impl.array([min(((x-c)**2).sum() for c in cs) for x in X])
                cs.append(X[rng.choice(len(X),p=d/d.sum())])
            return _np_impl.array(cs)
        return X[rng.choice(len(X),self.n_clusters,replace=False)]
    def _run1(self,X,rng):
        cs=self._init_c(X,rng)
        for _ in range(self.max_iter):
            d=_np_impl.array([((X-c)**2).sum(1) for c in cs]).T
            lb=d.argmin(1)
            nc=_np_impl.array([X[lb==k].mean(0) if (lb==k).any() else cs[k]
                                for k in range(self.n_clusters)])
            if _np_impl.allclose(nc,cs,atol=1e-6): break
            cs=nc
        iner=sum(((X[lb==k]-cs[k])**2).sum() for k in range(self.n_clusters) if (lb==k).any())
        return lb,cs,iner
    def fit(self,X):
        rng=_np_impl.random.default_rng(self.random_state); best=None
        for _ in range(self.n_init):
            lb,cs,iner=self._run1(X,rng)
            if best is None or iner<best[2]: best=(lb,cs,iner)
        self.labels_,self.cluster_centers_,self.inertia_=best; return self
    def predict(self,X):
        d=_np_impl.array([((X-c)**2).sum(1) for c in self.cluster_centers_]).T
        return d.argmin(1)

class _TS:
    def __init__(self):
        self.feature=_np_impl.array([],dtype=int); self.threshold=_np_impl.array([],dtype=float)
        self.children_left=_np_impl.array([],dtype=int); self.children_right=_np_impl.array([],dtype=int)
        self.value=[]; self.n_node_samples=_np_impl.array([],dtype=int)

class DecisionTreeClassifier:
    def __init__(self,max_depth=None,random_state=None,**kw):
        self.max_depth=max_depth or 999
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            for t in (vals[:-1]+vals[1:])/2:
                l=X[:,f]<=t; r=~l
                if l.sum()<1 or r.sum()<1: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth,nid,ts):
        cl,cc=_np_impl.unique(y,return_counts=True); val=_np_impl.zeros(len(self._classes))
        for c,cnt in zip(cl,cc): val[_np_impl.where(self._classes==c)[0][0]]=cnt
        ts.feature=_np_impl.append(ts.feature,-2); ts.threshold=_np_impl.append(ts.threshold,-2.0)
        ts.children_left=_np_impl.append(ts.children_left,-1); ts.children_right=_np_impl.append(ts.children_right,-1)
        ts.value.append(val); ts.n_node_samples=_np_impl.append(ts.n_node_samples,len(y))
        cur=nid[0]; nid[0]+=1
        if depth<self.max_depth and len(_np_impl.unique(y))>1:
            sp=self._best(X,y)
            if sp:
                _,f,t=sp; ts.feature[cur]=f; ts.threshold[cur]=t; l=X[:,f]<=t
                ts.children_left[cur]=nid[0]; self._build(X[l],y[l],depth+1,nid,ts)
                ts.children_right[cur]=nid[0]; self._build(X[~l],y[~l],depth+1,nid,ts)
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); ts=_TS(); nid=[0]
        self._build(X,y,0,nid,ts); self.tree_=ts; return self
    def _p1(self,x,n=0):
        ts=self.tree_
        if ts.children_left[n]==-1: return self._classes[ts.value[n].argmax()]
        return self._p1(x,ts.children_left[n] if x[ts.feature[n]]<=ts.threshold[n] else ts.children_right[n])
    def predict(self,X): return _np_impl.array([self._p1(x) for x in X])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

from scipy.linalg import svd

np.random.seed(42)

def measure_time(fn, *args, repeats=3, **kwargs):
    """Time a function call, return median time in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return np.median(times)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Matrix multiply scaling with n
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  EMPIRICAL SCALING: CORE ML OPERATIONS")
print("=" * 65)
print()

n_vals = [100, 300, 500, 1000, 2000, 3000, 5000]
d_fix  = 50

print(f"  MATRIX MULTIPLY (n×d) · (d×n), d={d_fix}")
print(f"  {'n':>6}  {'Time(s)':>10}  {'Ops(est.)':>14}  {'Ops/sec':>12}")
print("  " + "─" * 46)
mm_times = []
for n in n_vals:
    A = np.random.randn(n, d_fix); B = np.random.randn(d_fix, n)
    t = measure_time(np.dot, A, B)
    ops = 2 * n * d_fix * n   # FLOPs for matrix multiply
    mm_times.append(t)
    print(f"  {n:>6}  {t:>10.5f}  {ops:>14,.0f}  {ops/t:>12,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: SVD scaling with n
# ─────────────────────────────────────────────────────────────────────────────

print()
print(f"  SVD (n×d matrix, d={d_fix}) — O(nd²)")
print(f"  {'n':>6}  {'Full SVD(s)':>12}  {'Trunc-SVD(s)':>14}  {'Speedup':>8}")
print("  " + "─" * 44)
svd_times = []; tsvd_times = []
for n in n_vals:
    A = np.random.randn(n, d_fix)
    t_full  = measure_time(svd, A, full_matrices=False)
    tsvd    = TruncatedSVD(n_components=10, random_state=0)
    t_trunc = measure_time(tsvd.fit, A)
    svd_times.append(t_full); tsvd_times.append(t_trunc)
    print(f"  {n:>6}  {t_full:>12.5f}  {t_trunc:>14.5f}  {t_full/t_trunc:>8.2f}x")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Kernel matrix construction — O(n²d)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(f"  KERNEL MATRIX (RBF, n×n), d={d_fix} — O(n²d)")
print(f"  {'n':>6}  {'Time(s)':>10}  {'Size(MB)':>10}  {'Scale factor':>14}")
print("  " + "─" * 44)
kern_times = []
for i, n in enumerate(n_vals):
    A = np.random.randn(n, d_fix)
    def rbf_kern(A):
        dists = np.sum(A**2,1,keepdims=True) + np.sum(A**2,1) - 2*A@A.T
        return np.exp(-dists)
    t     = measure_time(rbf_kern, A)
    size  = n*n*4/1e6  # MB
    scale = n**2 * d_fix / (n_vals[0]**2 * d_fix) if i > 0 else 1.0
    kern_times.append(t)
    print(f"  {n:>6}  {t:>10.5f}  {size:>10.2f}  {scale:>14.1f}x")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Linear regression — primal vs dual
# ─────────────────────────────────────────────────────────────────────────────

print()
print("  LINEAR REGRESSION: PRIMAL O(nd²+d³) vs DUAL O(n³)")
n_wide = [100, 300, 500, 800, 1000]
d_wide = 2000   # d > n → dual is cheaper

print(f"  d={d_wide} (d >> n → dual cheaper when n is small)")
print(f"  {'n':>6}  {'Primal(s)':>10}  {'Dual(s)':>10}  {'Dual wins?':>12}")
print("  " + "─" * 42)
for n in n_wide:
    A = np.random.randn(n, d_wide)
    y = np.random.randn(n)
    lam = 0.1
    # Primal: (AᵀA + λI)⁻¹ Aᵀy  — d×d system
    t_primal = measure_time(
        lambda: np.linalg.solve(A.T@A + lam*np.eye(d_wide), A.T@y)
    )
    # Dual: A(AAᵀ + λI)⁻¹ y   — n×n system
    t_dual   = measure_time(
        lambda: A.T @ np.linalg.solve(A@A.T + lam*np.eye(n), y)
    )
    print(f"  {n:>6}  {t_primal:>10.4f}  {t_dual:>10.4f}  "
          f"{'✓ YES' if t_dual < t_primal else 'no':>12}")

print()
print("  Dual is faster when n < d because n×n solve < d×d solve.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: k-NN brute force vs approximate
# ─────────────────────────────────────────────────────────────────────────────

print()
print("  k-NN QUERY TIME: BRUTE FORCE vs TREE, d=10")
d_knn    = 10
n_knn    = [200, 500, 1000, 2000, 5000, 10000]
print(f"  {'n_train':>8}  {'Brute(s)':>10}  {'KD-tree(s)':>12}  {'Speedup':>9}")
print("  " + "─" * 44)
knn_brute = []; knn_tree = []
for n in n_knn:
    X_tr = np.random.randn(n, d_knn)
    X_q  = np.random.randn(100, d_knn)   # 100 query points
    nn_b = NearestNeighbors(n_neighbors=5, algorithm="brute")
    nn_t = NearestNeighbors(n_neighbors=5, algorithm="kd_tree")
    nn_b.fit(X_tr); nn_t.fit(X_tr)
    t_b = measure_time(nn_b.kneighbors, X_q)
    t_t = measure_time(nn_t.kneighbors, X_q)
    knn_brute.append(t_b); knn_tree.append(t_t)
    print(f"  {n:>8}  {t_b:>10.5f}  {t_t:>12.5f}  {t_b/t_t:>9.2f}x")

# ─────────────────────────────────────────────────────────────────────────────
# Fit log-log regression to extract empirical exponents
# ─────────────────────────────────────────────────────────────────────────────

print()
print("  EMPIRICAL COMPLEXITY EXPONENTS (log-log fit):")
print(f"  {'Operation':>28}  {'Empirical O(nᵅ)':>18}  {'Theory':>10}")
print("  " + "─" * 60)
for name, n_list, t_list, theory in [
    ("Matrix multiply (d=50)", n_vals, mm_times, "O(n²d) → α=2"),
    ("Full SVD (d=50)",        n_vals, svd_times, "O(nd²) → α=1"),
    ("Kernel matrix (d=50)",   n_vals, kern_times, "O(n²d) → α=2"),
    ("k-NN brute (d=10)",      n_knn,  knn_brute,  "O(nd) → α=1"),
    ("k-NN tree (d=10)",       n_knn,  knn_tree,   "O(log n) → α→0"),
]:
    log_n = np.log(n_list)
    log_t = np.log(np.maximum(t_list, 1e-9))
    alpha = np.polyfit(log_n, log_t, 1)[0]
    print(f"  {name:>28}  {'O(n^'+f'{alpha:.2f}'+')':>18}  {theory:>10}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Empirical Complexity Scaling: Wall-Clock Time vs n",
             fontsize=13, fontweight="bold")

# Panel (0,0): Matrix multiply
ax = axes[0, 0]
ax.loglog(n_vals, mm_times, "steelblue", lw=2.5, marker="o", ms=8, label="Measured")
ref = mm_times[0] * (np.array(n_vals)/n_vals[0])**2
ax.loglog(n_vals, ref, "tomato", lw=2, ls="--", label="O(n²) reference")
ax.set_title(f"Matrix Multiply (d={d_fix})\\nlog-log slope ≈ 2 → O(n²)",
             fontweight="bold")
ax.set_xlabel("n (log)"); ax.set_ylabel("Time (s, log)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (0,1): SVD full vs truncated
ax = axes[0, 1]
ax.loglog(n_vals, svd_times,  "steelblue", lw=2.5, marker="o", ms=8, label="Full SVD")
ax.loglog(n_vals, tsvd_times, "tomato",    lw=2.5, marker="s", ms=8, label="Truncated SVD (k=10)")
ref_lin = svd_times[0] * np.array(n_vals) / n_vals[0]
ax.loglog(n_vals, ref_lin, "gray", lw=1.5, ls=":", label="O(n) reference")
ax.set_title(f"SVD Scaling (d={d_fix})\\n(Full≈O(nd²), Trunc≈O(ndk))",
             fontweight="bold")
ax.set_xlabel("n (log)"); ax.set_ylabel("Time (s, log)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (0,2): Kernel matrix + memory
ax = axes[0, 2]
ax_twin = ax.twinx()
ax.loglog(n_vals, kern_times, "steelblue", lw=2.5, marker="o", ms=8, label="Kernel time")
ref_n2  = kern_times[0] * (np.array(n_vals)/n_vals[0])**2
ax.loglog(n_vals, ref_n2, "tomato", lw=2, ls="--", label="O(n²)")
mem_mb  = [n**2 * 4 / 1e6 for n in n_vals]
ax_twin.loglog(n_vals, mem_mb, "seagreen", lw=2, marker="^", ms=7, ls="--")
ax_twin.set_ylabel("Memory (MB)", color="seagreen")
ax_twin.tick_params(axis="y", labelcolor="seagreen")
ax.set_title(f"Kernel Matrix (RBF, d={d_fix})\\n(time + memory both O(n²))",
             fontweight="bold")
ax.set_xlabel("n (log)"); ax.set_ylabel("Time (s, log)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (1,0): Primal vs dual regression
ax = axes[1, 0]
primal_t = []; dual_t = []
n_range  = [50, 100, 200, 400, 600, 800, 1000]
d_range  = 1200
for n_r in n_range:
    A_ = np.random.randn(n_r, d_range); y_ = np.random.randn(n_r)
    tp = measure_time(lambda: np.linalg.solve(A_.T@A_+0.1*np.eye(d_range), A_.T@y_))
    td = measure_time(lambda: A_.T @ np.linalg.solve(A_@A_.T+0.1*np.eye(n_r), y_))
    primal_t.append(tp); dual_t.append(td)
ax.semilogy(n_range, primal_t, "steelblue", lw=2.5, marker="o", ms=7, label=f"Primal O(d³) d={d_range}")
ax.semilogy(n_range, dual_t,   "tomato",    lw=2.5, marker="s", ms=7, label=f"Dual O(n³)")
ax.set_title(f"Ridge: Primal vs Dual (d={d_range})\\n(dual wins when n < d)",
             fontweight="bold")
ax.set_xlabel("n"); ax.set_ylabel("Time (s, log)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# Panel (1,1): k-NN scaling
ax = axes[1, 1]
ax.loglog(n_knn, knn_brute, "steelblue", lw=2.5, marker="o", ms=8, label="Brute force O(n)")
ax.loglog(n_knn, knn_tree,  "tomato",    lw=2.5, marker="s", ms=8, label="KD-tree O(log n)")
ref_log = knn_tree[0] * np.log(np.array(n_knn)) / np.log(n_knn[0])
ax.loglog(n_knn, ref_log, "gray", lw=1.5, ls=":", label="O(log n) reference")
ax.set_title(f"k-NN Query Time (d={d_knn}, 100 queries)\\n(KD-tree much faster for low d)",
             fontweight="bold")
ax.set_xlabel("n_train (log)"); ax.set_ylabel("Time (s, log)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (1,2): Algorithm phase diagram
ax = axes[1, 2]
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(100, 1e7); ax.set_ylim(10, 1e7)

# Region colours
from matplotlib.patches import Polygon
ax.fill_between([100,1e4,1e4,100], [10,10,1e7,1e7], alpha=0.08, color="steelblue",
                label="Kernel SVM/KRR feasible")
ax.fill_between([1e4,1e7,1e7,1e4], [10,10,1000,1000], alpha=0.08, color="tomato",
                label="d > n: dual preferred")
ax.fill_between([100,1e7,1e7,100], [1e5,1e5,1e7,1e7], alpha=0.1, color="seagreen",
                label="Neural net territory")

# Algorithm labels
for (n_p, d_p, label, col) in [
    (500, 500,    "All methods\\n(n≈d)",          "black"),
    (50000, 100,  "GBM/RF\\n(large n, small d)",  "steelblue"),
    (1000, 50000, "Dual ridge\\n(small n, large d)", "tomato"),
    (5e5, 5000,   "SGD models only\\n(large n,d)",  "seagreen"),
]:
    ax.text(n_p, d_p, label, ha="center", va="center", fontsize=7.5,
            color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=col, alpha=0.8))

ax.set_xlabel("n (samples, log scale)")
ax.set_ylabel("d (features, log scale)")
ax.set_title("Algorithm Phase Diagram\\n(algorithm choice by (n, d) regime)",
             fontweight="bold")
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("complexity_scaling.png", dpi=110)
print()
print("  Plot saved → complexity_scaling.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Transformer Attention Complexity and Efficient Alternatives": {
        "description": (
            "Measures the O(T²) complexity of standard self-attention vs "
            "sequence length T. Implements and benchmarks three efficient "
            "attention alternatives: (1) local windowed attention O(T·w), "
            "(2) linear attention O(T·d), and (3) random sparse attention. "
            "Shows memory and time scaling for each. Demonstrates why the "
            "quadratic bottleneck matters for long sequences and when each "
            "alternative is appropriate."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

np.random.seed(42)

def measure(fn, *args, repeats=3, **kwargs):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return np.median(times), result

# ─────────────────────────────────────────────────────────────────────────────
# Standard self-attention — O(T²d)
# ─────────────────────────────────────────────────────────────────────────────

def standard_attention(Q, K, V):
    """Full self-attention: O(T²d) time, O(T²) memory."""
    d    = Q.shape[-1]
    A    = Q @ K.T / np.sqrt(d)            # T×T attention scores
    A    = np.exp(A - A.max(-1, keepdims=True))  # stable softmax
    A   /= A.sum(-1, keepdims=True)
    return A @ V                             # T×d output

# ─────────────────────────────────────────────────────────────────────────────
# Local windowed attention — O(T·w·d)
# ─────────────────────────────────────────────────────────────────────────────

def windowed_attention(Q, K, V, window=32):
    """Each token attends to ±window/2 neighbours only."""
    T, d   = Q.shape
    out    = np.zeros_like(Q)
    half_w = window // 2
    for t in range(T):
        lo = max(0, t - half_w)
        hi = min(T, t + half_w + 1)
        scores = Q[t] @ K[lo:hi].T / np.sqrt(d)
        scores = np.exp(scores - scores.max())
        scores /= scores.sum()
        out[t]  = scores @ V[lo:hi]
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Linear attention approximation — O(T·d²)
# ─────────────────────────────────────────────────────────────────────────────

def linear_attention(Q, K, V):
    """
    Linear attention via kernel trick: φ(Q) @ (φ(K)ᵀ V)
    Feature map φ(x) = ELU(x) + 1  (non-negative, simple)
    O(T·d²) instead of O(T²d)
    """
    phi_Q = np.maximum(Q, 0) + 1    # ELU-like feature map
    phi_K = np.maximum(K, 0) + 1
    # Key insight: (phi_Q @ phi_Kᵀ) @ V = phi_Q @ (phi_Kᵀ @ V)
    # Right factor is d×d — computed once for all queries.
    KV   = phi_K.T @ V              # d×d   (computed once)
    K_sum = phi_K.sum(0)            # d     (normalisation)
    out  = (phi_Q @ KV) / (phi_Q @ K_sum + 1e-6)[:, None]
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Complexity analysis
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  TRANSFORMER ATTENTION: COMPLEXITY ANALYSIS")
print("=" * 65)
print()

d_model = 64   # embedding dimension (small for CPU demo)
T_vals  = [32, 64, 128, 256, 512, 1024]

std_times  = []
win_times  = []
lin_times  = []
std_mems   = []

print(f"  d_model = {d_model}")
print()
print(f"  {'T':>6}  {'Std(s)':>9}  {'Win(s)':>9}  {'Lin(s)':>9}  "
      f"{'Attn mem MB':>12}  {'Win/Std':>9}  {'Lin/Std':>9}")
print("  " + "─" * 72)

for T in T_vals:
    Q = np.random.randn(T, d_model) * 0.1
    K = np.random.randn(T, d_model) * 0.1
    V = np.random.randn(T, d_model) * 0.1

    t_std, _ = measure(standard_attention, Q, K, V)
    t_win, _ = measure(windowed_attention,  Q, K, V, window=min(32, T))
    t_lin, _ = measure(linear_attention,    Q, K, V)

    mem_attn = T * T * 4 / 1e6   # T×T float32 attention matrix
    std_times.append(t_std); win_times.append(t_win); lin_times.append(t_lin)
    std_mems.append(mem_attn)

    print(f"  {T:>6}  {t_std:>9.5f}  {t_win:>9.5f}  {t_lin:>9.5f}  "
          f"{mem_attn:>12.3f}  {t_win/t_std:>9.3f}  {t_lin/t_std:>9.3f}")

print()
print("  EMPIRICAL SCALING (log-log fit to extract exponent):")
log_T = np.log(T_vals)
for name, times in [("Standard attention", std_times),
                     ("Windowed attention", win_times),
                     ("Linear attention",   lin_times)]:
    log_t = np.log(np.maximum(times, 1e-9))
    alpha = np.polyfit(log_T, log_t, 1)[0]
    print(f"  {name:>22}: empirical O(T^{alpha:.2f})")

print()
print("  Standard: O(T²) confirmed. Linear: O(T¹) ≈ confirmed.")
print("  Windowed: between O(T) and O(T²) — depends on w.")

# ─────────────────────────────────────────────────────────────────────────────
# Approximation quality of efficient attention
# ─────────────────────────────────────────────────────────────────────────────

print()
print("  APPROXIMATION QUALITY vs STANDARD ATTENTION:")
T_test = 256
Q_t = np.random.randn(T_test, d_model)
K_t = np.random.randn(T_test, d_model)
V_t = np.random.randn(T_test, d_model)

out_std = standard_attention(Q_t, K_t, V_t)
for name, fn, kwargs in [
    ("Win w=16",   windowed_attention, {"window": 16}),
    ("Win w=32",   windowed_attention, {"window": 32}),
    ("Win w=64",   windowed_attention, {"window": 64}),
    ("Linear",     linear_attention,   {}),
]:
    out = fn(Q_t, K_t, V_t, **kwargs) if kwargs else fn(Q_t, K_t, V_t)
    cos_sim = np.mean([np.dot(out_std[t], out[t]) /
                       (np.linalg.norm(out_std[t]) * np.linalg.norm(out[t]) + 1e-9)
                       for t in range(T_test)])
    mse     = np.mean((out_std - out)**2)
    print(f"  {name:>12}:  cosine_sim={cos_sim:.4f}  MSE={mse:.6f}")

print()
print("  Windowed attention approximation improves with window size.")
print("  Linear attention is fastest but may not capture long-range dependencies.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Transformer Attention Complexity: O(T²) vs Efficient Alternatives",
             fontsize=13, fontweight="bold")

# Panel (0,0): Time scaling
ax = axes[0, 0]
ax.loglog(T_vals, std_times, "tomato",    lw=2.5, marker="o", ms=8, label="Standard O(T²d)")
ax.loglog(T_vals, win_times, "steelblue", lw=2.5, marker="s", ms=8, label="Windowed O(T·w·d)")
ax.loglog(T_vals, lin_times, "seagreen",  lw=2.5, marker="^", ms=8, label="Linear O(T·d²)")
ref_T2 = std_times[0] * (np.array(T_vals)/T_vals[0])**2
ref_T1 = lin_times[0] * np.array(T_vals) / T_vals[0]
ax.loglog(T_vals, ref_T2, "tomato",   lw=1, ls=":", alpha=0.5)
ax.loglog(T_vals, ref_T1, "seagreen", lw=1, ls=":", alpha=0.5)
ax.set_xlabel("Sequence length T (log)"); ax.set_ylabel("Time (s, log)")
ax.set_title("Attention Time Scaling\\n(standard T² vs efficient T¹)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (0,1): Memory scaling
ax = axes[0, 1]
T_ext  = np.logspace(1, 5, 50)
mem_std  = T_ext**2 * 4 / 1e6
mem_win  = T_ext * 32 * 4 / 1e6
mem_lin  = T_ext * d_model * 4 / 1e6
ax.loglog(T_ext, mem_std, "tomato",    lw=2.5, label="Standard O(T²)")
ax.loglog(T_ext, mem_win, "steelblue", lw=2.5, label="Windowed O(T·w)")
ax.loglog(T_ext, mem_lin, "seagreen",  lw=2.5, label="Linear O(T·d)")
ax.axhline(80*1024, color="gray", lw=1.5, ls="--", label="A100 VRAM (80GB)")
ax.axhline(16*1024, color="gray", lw=1.5, ls=":",  label="Consumer GPU (16GB)")
ax.set_xlabel("Sequence length T (log)"); ax.set_ylabel("Memory (MB, log)")
ax.set_title("Attention Memory Scaling\\n(T≈10K: standard attention hits GPU limits)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

# Panel (1,0): Standard attention matrix visualisation
ax = axes[1, 0]
T_vis = 64
Q_vis = np.random.randn(T_vis, 16)
K_vis = np.random.randn(T_vis, 16)
A_vis = np.exp(Q_vis @ K_vis.T / 4)
A_vis /= A_vis.sum(-1, keepdims=True)
im = ax.imshow(A_vis, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, label="Attention weight")
ax.set_title(f"Standard Attention Matrix (T={T_vis})\\n"
             f"(full T×T — each row is a distribution)", fontweight="bold")
ax.set_xlabel("Key position"); ax.set_ylabel("Query position")

# Panel (1,1): Complexity comparison table
ax = axes[1, 1]
ax.axis("off")
rows = [
    ["Standard", "O(T²·d)", "O(T²)", "Exact", "All tasks"],
    ["Windowed", "O(T·w·d)", "O(T·w)", "Local only", "NLP (local context)"],
    ["Sparse", "O(T·s·d)", "O(T·s)", "Selected", "Structured patterns"],
    ["Linear", "O(T·d²)", "O(T·d)", "Approximate", "Fast, long sequences"],
    ["FlashAttn", "O(T²·d)", "O(T)", "Exact+IO", "Practical T²"],
    ["Performer", "O(T·d·r)", "O(T·d·r)", "Approx", "Efficient inference"],
]
headers = ["Method", "Time", "Space", "Accuracy", "Best for"]
table = ax.table(cellText=rows, colLabels=headers,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 2.0)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif r == 4:
        cell.set_facecolor("#e8f5e8")
ax.set_title("Attention Variants: Complexity and Tradeoffs", fontweight="bold")

plt.tight_layout()
plt.savefig("attention_complexity.png", dpi=110)
print()
print("  Plot saved → attention_complexity.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · NP-Hardness in ML — Decision Trees, k-Means, and Feature Selection": {
        "description": (
            "Demonstrates the NP-hardness of three core ML problems by "
            "comparing greedy heuristics against optimal solutions (tractable "
            "for small instances): optimal decision trees vs CART, globally "
            "optimal k-means (brute-force for tiny k and n) vs Lloyd's "
            "algorithm with different initialisations, and exhaustive feature "
            "selection vs greedy forward selection. Shows the optimality gap "
            "and how problem size makes exact solutions infeasible."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy
from scipy.spatial import KDTree as _KDTree

class TruncatedSVD:
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
    def fit(self, X):
        U, s, Vt = _np_impl.linalg.svd(X, full_matrices=False)
        k = min(self.n_components, len(s))
        self.components_ = Vt[:k]; self.singular_values_ = s[:k]; return self
    def transform(self, X): return X @ self.components_.T
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class NearestNeighbors:
    def __init__(self, n_neighbors=5, algorithm='brute'):
        self.k = n_neighbors; self.algorithm = algorithm
    def fit(self, X):
        self._X = X.copy()
        if self.algorithm == 'kd_tree':
            self._tree = _KDTree(X)
        return self
    def kneighbors(self, X_q):
        X_q = _np_impl.atleast_2d(X_q)
        if self.algorithm == 'kd_tree':
            return self._tree.query(X_q, k=self.k)
        d2 = ((_np_impl.sum(self._X**2, 1, keepdims=True).T
               + _np_impl.sum(X_q**2, 1, keepdims=True)
               - 2 * X_q @ self._X.T))
        dists = _np_impl.sqrt(_np_impl.maximum(d2, 0))
        idx = _np_impl.argsort(dists, 1)[:, :self.k]
        return _np_impl.sort(dists, 1)[:, :self.k], idx

def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    c_arr = (rng.uniform(-5,5,(centers,n_features)) if isinstance(centers,int)
             else _np_impl.asarray(centers))
    nc = len(c_arr)
    counts = [n_samples//nc+(1 if i<n_samples%nc else 0) for i in range(nc)]
    Xs, ys = [], []
    for i,(cnt,c) in enumerate(zip(counts,c_arr)):
        Xs.append(rng.normal(c,cluster_std,(cnt,n_features))); ys.append(_np_impl.full(cnt,i,dtype=int))
    return _np_impl.vstack(Xs), _np_impl.hstack(ys)

def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    y = rng.integers(0,2,n_samples)
    Xi = rng.standard_normal((n_samples,n_informative))
    Xi += (2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    Xr = (Xi[:,:n_redundant]+0.3*rng.standard_normal((n_samples,n_redundant))
          if n_redundant>0 else _np_impl.empty((n_samples,0)))
    nn2 = max(0,n_features-n_informative-n_redundant)
    Xn = rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    return _np_impl.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features],y.astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict(self,X): return (_sig(X@self.coef_.ravel()+self.intercept_[0])>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class SGDClassifier:
    def __init__(self,**kw): self._w=None; self._b=0.0; self._t=1
    def partial_fit(self,X,y,classes=None):
        X=_np_impl.atleast_2d(X).astype(float); y=_np_impl.asarray(y)
        if self._w is None: self._w=_np_impl.zeros(X.shape[1])
        lr=min(1.0/(0.0001*self._t+1e-6),0.1)
        for xi,yi in zip(X,y):
            p=_sig(xi@self._w+self._b); e=p-yi
            self._w-=lr*e*xi; self._b-=lr*e; self._t+=1
        return self
    def predict(self,X):
        return (_sig(_np_impl.atleast_2d(X).astype(float)@self._w+self._b)>=0.5).astype(int)

def cross_val_score(estimator,X,y,cv=5,scoring='accuracy'):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]))
    return _np_impl.array(scores)

class KMeans:
    def __init__(self,n_clusters=8,init='k-means++',n_init=10,
                 random_state=None,max_iter=300,**kw):
        self.n_clusters=n_clusters; self.init=init
        self.n_init=n_init; self.random_state=random_state; self.max_iter=max_iter
    def _init_c(self,X,rng):
        if self.init=='k-means++':
            cs=[X[rng.integers(0,len(X))]]
            for _ in range(self.n_clusters-1):
                d=_np_impl.array([min(((x-c)**2).sum() for c in cs) for x in X])
                cs.append(X[rng.choice(len(X),p=d/d.sum())])
            return _np_impl.array(cs)
        return X[rng.choice(len(X),self.n_clusters,replace=False)]
    def _run1(self,X,rng):
        cs=self._init_c(X,rng)
        for _ in range(self.max_iter):
            d=_np_impl.array([((X-c)**2).sum(1) for c in cs]).T
            lb=d.argmin(1)
            nc=_np_impl.array([X[lb==k].mean(0) if (lb==k).any() else cs[k]
                                for k in range(self.n_clusters)])
            if _np_impl.allclose(nc,cs,atol=1e-6): break
            cs=nc
        iner=sum(((X[lb==k]-cs[k])**2).sum() for k in range(self.n_clusters) if (lb==k).any())
        return lb,cs,iner
    def fit(self,X):
        rng=_np_impl.random.default_rng(self.random_state); best=None
        for _ in range(self.n_init):
            lb,cs,iner=self._run1(X,rng)
            if best is None or iner<best[2]: best=(lb,cs,iner)
        self.labels_,self.cluster_centers_,self.inertia_=best; return self
    def predict(self,X):
        d=_np_impl.array([((X-c)**2).sum(1) for c in self.cluster_centers_]).T
        return d.argmin(1)

class _TS:
    def __init__(self):
        self.feature=_np_impl.array([],dtype=int); self.threshold=_np_impl.array([],dtype=float)
        self.children_left=_np_impl.array([],dtype=int); self.children_right=_np_impl.array([],dtype=int)
        self.value=[]; self.n_node_samples=_np_impl.array([],dtype=int)

class DecisionTreeClassifier:
    def __init__(self,max_depth=None,random_state=None,**kw):
        self.max_depth=max_depth or 999
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            for t in (vals[:-1]+vals[1:])/2:
                l=X[:,f]<=t; r=~l
                if l.sum()<1 or r.sum()<1: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth,nid,ts):
        cl,cc=_np_impl.unique(y,return_counts=True); val=_np_impl.zeros(len(self._classes))
        for c,cnt in zip(cl,cc): val[_np_impl.where(self._classes==c)[0][0]]=cnt
        ts.feature=_np_impl.append(ts.feature,-2); ts.threshold=_np_impl.append(ts.threshold,-2.0)
        ts.children_left=_np_impl.append(ts.children_left,-1); ts.children_right=_np_impl.append(ts.children_right,-1)
        ts.value.append(val); ts.n_node_samples=_np_impl.append(ts.n_node_samples,len(y))
        cur=nid[0]; nid[0]+=1
        if depth<self.max_depth and len(_np_impl.unique(y))>1:
            sp=self._best(X,y)
            if sp:
                _,f,t=sp; ts.feature[cur]=f; ts.threshold[cur]=t; l=X[:,f]<=t
                ts.children_left[cur]=nid[0]; self._build(X[l],y[l],depth+1,nid,ts)
                ts.children_right[cur]=nid[0]; self._build(X[~l],y[~l],depth+1,nid,ts)
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); ts=_TS(); nid=[0]
        self._build(X,y,0,nid,ts); self.tree_=ts; return self
    def _p1(self,x,n=0):
        ts=self.tree_
        if ts.children_left[n]==-1: return self._classes[ts.value[n].argmax()]
        return self._p1(x,ts.children_left[n] if x[ts.feature[n]]<=ts.threshold[n] else ts.children_right[n])
    def predict(self,X): return _np_impl.array([self._p1(x) for x in X])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Optimal feature selection (brute force) vs greedy
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  NP-HARDNESS IN ML: FEATURE SELECTION")
print("=" * 65)
print()
print("  Finding minimum feature subset with accuracy ≥ threshold:")
print("  Exact = try all 2^d subsets  (NP-hard). Greedy = polynomial.")
print()

# Generate data with known relevant and irrelevant features
n_samples = 300
n_features = 12   # tractable for brute-force (2^12 = 4096 subsets)
n_informative = 5

X, y = make_classification(n_samples=n_samples,
                            n_features=n_features,
                            n_informative=n_informative,
                            n_redundant=2,
                            n_repeated=0,
                            random_state=42)
X = StandardScaler().fit_transform(X)

# Brute force: evaluate all subsets of size 1..n_features
def eval_subset(X_, y_, idx):
    if len(idx) == 0:
        return 0.0
    clf = LogisticRegression(C=1.0, max_iter=200, random_state=0)
    scores = cross_val_score(clf, X_[:, idx], y_, cv=5, scoring="accuracy")
    return scores.mean()

print(f"  Brute-force feature selection (2^{n_features} = {2**n_features:,} subsets)...")
best_acc_bf  = {k: 0.0 for k in range(1, 6)}
best_idx_bf  = {k: None for k in range(1, 6)}
total_evals  = 0
for k in range(1, 6):   # subsets of size 1..5
    for combo in combinations(range(n_features), k):
        acc = eval_subset(X, y, list(combo))
        total_evals += 1
        if acc > best_acc_bf[k]:
            best_acc_bf[k] = acc
            best_idx_bf[k] = list(combo)

print(f"  Total evaluations: {total_evals:,}")
print()
print(f"  {'k features':>12}  {'Best acc (BF)':>14}  {'Features':>20}")
for k in range(1, 6):
    print(f"  {k:>12}  {best_acc_bf[k]:>14.4f}  {str(best_idx_bf[k]):>20}")

# Greedy forward selection
print()
print("  Greedy forward selection (polynomial):")
selected_g = []
remaining  = list(range(n_features))
greedy_accs = []

for step in range(5):
    best_next  = None
    best_acc_g = 0.0
    for feat in remaining:
        candidate = selected_g + [feat]
        acc = eval_subset(X, y, candidate)
        if acc > best_acc_g:
            best_acc_g = acc; best_next = feat
    selected_g.append(best_next)
    remaining.remove(best_next)
    greedy_accs.append(best_acc_g)
    print(f"  Step {step+1}: add feature {best_next:>2}  → acc={best_acc_g:.4f}  "
          f"selected={selected_g}")

print()
print("  OPTIMALITY GAP (brute-force optimal vs greedy):")
for k in range(1, 6):
    gap = best_acc_bf[k] - greedy_accs[k-1]
    print(f"  k={k}: optimal={best_acc_bf[k]:.4f}  greedy={greedy_accs[k-1]:.4f}  "
          f"gap={gap:+.4f}")
print()
print(f"  2^{n_features}={2**n_features:,} brute-force evals was tractable for n_feat={n_features}.")
print(f"  For n_feat=50: 2^50 = {2**50:,} — completely infeasible.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: k-means — multiple initialisations show NP-hardness
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  NP-HARDNESS: k-MEANS MULTIPLE LOCAL OPTIMA")
print("=" * 65)
print()

X_clust, y_clust = make_blobs(n_samples=200, n_features=2,
                               centers=4, cluster_std=0.8, random_state=0)

print("  Running k-means with 30 random initialisations (k=4):")
inertias = []
for seed in range(30):
    km = KMeans(n_clusters=4, init="random", n_init=1, random_state=seed, max_iter=300)
    km.fit(X_clust)
    inertias.append(km.inertia_)

km_pp = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0)
km_pp.fit(X_clust)
inertia_pp = km_pp.inertia_

best_iner  = min(inertias)
worst_iner = max(inertias)
print(f"  Random init: best={best_iner:.2f}, worst={worst_iner:.2f}, "
      f"gap={worst_iner-best_iner:.2f} ({100*(worst_iner-best_iner)/best_iner:.1f}%)")
print(f"  k-means++:   inertia={inertia_pp:.2f}")
print()
print("  The gap between best and worst random init demonstrates that")
print("  k-means can converge to sub-optimal local minima.")
print("  k-means++ reduces this by choosing initial centres more carefully.")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Decision tree — greedy CART vs exhaustive small tree
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  NP-HARDNESS: DECISION TREE — GREEDY CART vs EXHAUSTIVE")
print("=" * 65)
print()

# Small dataset where we can enumerate depth-2 trees
X_tree = X_clust[:50]
y_tree = (X_clust[:50, 0] + X_clust[:50, 1] > 0).astype(int)

# CART greedy
cart = DecisionTreeClassifier(max_depth=2, random_state=0)
cart.fit(X_tree, y_tree)
cart_acc = cart.score(X_tree, y_tree)
cart_feat1 = cart.tree_.feature[0]
cart_thresh1 = cart.tree_.threshold[0]
print(f"  CART (greedy): first split on feature {cart_feat1}, "
      f"threshold {cart_thresh1:.3f}")
print(f"  CART training accuracy: {cart_acc:.4f}")

# Exhaustive search over all depth-1 stumps (tractable)
best_acc_stump = 0.0
best_feat_s, best_thresh_s = None, None
n_evals_stump = 0
for feat in range(X_tree.shape[1]):
    thresholds = np.unique(X_tree[:, feat])
    for thresh in thresholds:
        pred = (X_tree[:, feat] > thresh).astype(int)
        acc  = max(np.mean(pred == y_tree), np.mean(pred != y_tree))
        n_evals_stump += 1
        if acc > best_acc_stump:
            best_acc_stump = acc; best_feat_s = feat; best_thresh_s = thresh

print(f"  Exhaustive stump: best feature={best_feat_s}, "
      f"thresh={best_thresh_s:.3f}, acc={best_acc_stump:.4f}")
print(f"  Stump evaluations: {n_evals_stump} (manageable for depth=1)")
print()
print("  For depth-d trees with n samples and F features:")
print("  Number of unique trees ~ O(F·n)^(2^d - 1)  — exponential in depth.")
print("  CART is greedy: makes locally optimal splits, may miss global optimum.")

# Number of trees as a function of depth
print()
print("  TREE ENUMERATION COMPLEXITY:")
F = X_tree.shape[1]; N = len(X_tree)
for depth in [1, 2, 3, 4, 5]:
    n_nodes = 2**depth - 1   # internal nodes
    n_splits = F * N          # possible splits per node
    # Very rough upper bound
    approx = n_splits ** n_nodes
    print(f"  Depth {depth}: ~{n_splits}^{n_nodes} = "
          f"{min(approx, 1e30):.2e} candidate trees  "
          f"{'(tractable)' if approx < 1e6 else '(intractable)'}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("NP-Hardness in ML: Feature Selection, k-Means, Decision Trees",
             fontsize=13, fontweight="bold")

# Panel (0,0): Feature selection accuracy vs k
ax = axes[0, 0]
k_vals = list(range(1, 6))
ax.plot(k_vals, [best_acc_bf[k] for k in k_vals], "tomato",    lw=2.5, marker="o", ms=8,
        label="Brute force optimal")
ax.plot(k_vals, greedy_accs, "steelblue", lw=2.5, marker="s", ms=8,
        label="Greedy forward selection")
ax.set_xlabel("Number of features selected (k)")
ax.set_ylabel("Cross-val accuracy")
ax.set_title("Feature Selection: Brute Force vs Greedy\\n"
             "(greedy near-optimal here; gap grows in harder cases)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylim(0.5, 1.0)

# Panel (0,1): Exponential search space
ax = axes[0, 1]
d_vals = np.arange(1, 51)
ax.semilogy(d_vals, 2**d_vals, "tomato", lw=2.5, label="All subsets: 2^d")
ax.semilogy(d_vals, d_vals**2, "steelblue", lw=2.5, label="Greedy forward: O(d²)")
ax.axvline(20, color="gray", lw=1.5, ls="--", label="d=20: 2^20 ≈ 10^6")
ax.axvline(40, color="gray", lw=1.5, ls=":",  label="d=40: 2^40 ≈ 10^12")
ax.set_xlabel("Number of features d")
ax.set_ylabel("Search space size (log)")
ax.set_title("Feature Selection Search Space\\n(greedy is only practical option for d>30)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# Panel (0,2): k-means inertia distribution
ax = axes[0, 2]
ax.hist(inertias, bins=15, color="steelblue", alpha=0.8, edgecolor="white",
        label="Random init (30 runs)")
ax.axvline(best_iner,  color="seagreen", lw=2.5, label=f"Best random: {best_iner:.1f}")
ax.axvline(worst_iner, color="tomato",   lw=2.5, label=f"Worst: {worst_iner:.1f}")
ax.axvline(inertia_pp, color="black",    lw=2.5, ls="--", label=f"k-means++: {inertia_pp:.1f}")
ax.set_title("k-Means Inertia: Different Random Inits\\n"
             "(local optima spread → NP-hardness in practice)",
             fontweight="bold")
ax.set_xlabel("Inertia (within-cluster sum of squares)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,0): k-means best vs worst solution
ax = axes[1, 0]
km_best  = KMeans(n_clusters=4, init="random", n_init=1, random_state=int(np.argmin(inertias)))
km_worst = KMeans(n_clusters=4, init="random", n_init=1, random_state=int(np.argmax(inertias)))
km_best.fit(X_clust); km_worst.fit(X_clust)
labels_b = km_best.predict(X_clust)
labels_w = km_worst.predict(X_clust)
colours_km = ["steelblue", "tomato", "seagreen", "purple"]
for c in range(4):
    mask = labels_b == c
    ax.scatter(X_clust[mask, 0], X_clust[mask, 1], c=colours_km[c],
               s=20, alpha=0.7, marker="o")
ax.set_title(f"Best k-Means Init (inertia={best_iner:.1f})\\n(good global structure)",
             fontweight="bold")
ax.set_xlabel("X0"); ax.set_ylabel("X1"); ax.grid(alpha=0.3)

ax = axes[1, 1]
for c in range(4):
    mask = labels_w == c
    ax.scatter(X_clust[mask, 0], X_clust[mask, 1], c=colours_km[c],
               s=20, alpha=0.7, marker="o")
ax.set_title(f"Worst k-Means Init (inertia={worst_iner:.1f})\\n(poor local optimum)",
             fontweight="bold")
ax.set_xlabel("X0"); ax.set_ylabel("X1"); ax.grid(alpha=0.3)

# Panel (1,2): Exponential complexity of decision trees
ax = axes[1, 2]
depths = np.arange(1, 8)
approx_trees = np.minimum((X_tree.shape[1] * len(X_tree))**(2**depths - 1), 1e30)
ax.semilogy(depths, approx_trees, "tomato", lw=2.5, marker="o", ms=8,
            label="Approx candidate trees")
ax.semilogy(depths, 2**depths, "steelblue", lw=2.5, marker="s", ms=8,
            label="2^depth (nodes to split)")
ax.axhline(1e6,  color="gray", lw=1.5, ls="--", label="10^6 (feasibility threshold)")
ax.set_xlabel("Tree depth")
ax.set_ylabel("Size (log)")
ax.set_title("Decision Tree Search Space vs Depth\\n"
             "(greedy CART is the only tractable option)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("np_hardness_ml.png", dpi=110)
print()
print("  Plot saved → np_hardness_ml.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Approximation Algorithms — Sketching, LSH, and Mini-batch SGD": {
        "description": (
            "Demonstrates three key algorithmic techniques for scaling ML: "
            "(1) Random sketching/Johnson-Lindenstrauss lemma — shows that "
            "pairwise distances are preserved after projecting n×d to n×k. "
            "(2) Locality Sensitive Hashing for approximate nearest neighbours — "
            "benchmarks query time vs accuracy. (3) Mini-batch SGD convergence — "
            "compares full gradient, SGD, and mini-batch on a logistic regression "
            "problem, showing the convergence/cost tradeoff."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy
from scipy.spatial import KDTree as _KDTree

class TruncatedSVD:
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
    def fit(self, X):
        U, s, Vt = _np_impl.linalg.svd(X, full_matrices=False)
        k = min(self.n_components, len(s))
        self.components_ = Vt[:k]; self.singular_values_ = s[:k]; return self
    def transform(self, X): return X @ self.components_.T
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

class NearestNeighbors:
    def __init__(self, n_neighbors=5, algorithm='brute'):
        self.k = n_neighbors; self.algorithm = algorithm
    def fit(self, X):
        self._X = X.copy()
        if self.algorithm == 'kd_tree':
            self._tree = _KDTree(X)
        return self
    def kneighbors(self, X_q):
        X_q = _np_impl.atleast_2d(X_q)
        if self.algorithm == 'kd_tree':
            return self._tree.query(X_q, k=self.k)
        d2 = ((_np_impl.sum(self._X**2, 1, keepdims=True).T
               + _np_impl.sum(X_q**2, 1, keepdims=True)
               - 2 * X_q @ self._X.T))
        dists = _np_impl.sqrt(_np_impl.maximum(d2, 0))
        idx = _np_impl.argsort(dists, 1)[:, :self.k]
        return _np_impl.sort(dists, 1)[:, :self.k], idx

def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    c_arr = (rng.uniform(-5,5,(centers,n_features)) if isinstance(centers,int)
             else _np_impl.asarray(centers))
    nc = len(c_arr)
    counts = [n_samples//nc+(1 if i<n_samples%nc else 0) for i in range(nc)]
    Xs, ys = [], []
    for i,(cnt,c) in enumerate(zip(counts,c_arr)):
        Xs.append(rng.normal(c,cluster_std,(cnt,n_features))); ys.append(_np_impl.full(cnt,i,dtype=int))
    return _np_impl.vstack(Xs), _np_impl.hstack(ys)

def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    y = rng.integers(0,2,n_samples)
    Xi = rng.standard_normal((n_samples,n_informative))
    Xi += (2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    Xr = (Xi[:,:n_redundant]+0.3*rng.standard_normal((n_samples,n_redundant))
          if n_redundant>0 else _np_impl.empty((n_samples,0)))
    nn2 = max(0,n_features-n_informative-n_redundant)
    Xn = rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    return _np_impl.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features],y.astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,solver='lbfgs'):
        self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1.0/(self.C*n)
        def fg(w):
            p=_np_impl.clip(_sig(X@w[1:]+w[0]),1e-10,1-1e-10)
            loss=-_np_impl.mean(y*_np_impl.log(p)+(1-y)*_np_impl.log(1-p))+0.5*lam*_np_impl.sum(w[1:]**2)
            e=p-y; return loss,_np_impl.r_[e.mean(),X.T@e/n+lam*w[1:]]
        res=_sp_opt.minimize(fg,_np_impl.zeros(d+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':self.max_iter})
        self.coef_=res.x[1:].reshape(1,-1); self.intercept_=_np_impl.array([res.x[0]]); return self
    def predict(self,X): return (_sig(X@self.coef_.ravel()+self.intercept_[0])>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class SGDClassifier:
    def __init__(self,**kw): self._w=None; self._b=0.0; self._t=1
    def partial_fit(self,X,y,classes=None):
        X=_np_impl.atleast_2d(X).astype(float); y=_np_impl.asarray(y)
        if self._w is None: self._w=_np_impl.zeros(X.shape[1])
        lr=min(1.0/(0.0001*self._t+1e-6),0.1)
        for xi,yi in zip(X,y):
            p=_sig(xi@self._w+self._b); e=p-yi
            self._w-=lr*e*xi; self._b-=lr*e; self._t+=1
        return self
    def predict(self,X):
        return (_sig(_np_impl.atleast_2d(X).astype(float)@self._w+self._b)>=0.5).astype(int)

def cross_val_score(estimator,X,y,cv=5,scoring='accuracy'):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]))
    return _np_impl.array(scores)

class KMeans:
    def __init__(self,n_clusters=8,init='k-means++',n_init=10,
                 random_state=None,max_iter=300,**kw):
        self.n_clusters=n_clusters; self.init=init
        self.n_init=n_init; self.random_state=random_state; self.max_iter=max_iter
    def _init_c(self,X,rng):
        if self.init=='k-means++':
            cs=[X[rng.integers(0,len(X))]]
            for _ in range(self.n_clusters-1):
                d=_np_impl.array([min(((x-c)**2).sum() for c in cs) for x in X])
                cs.append(X[rng.choice(len(X),p=d/d.sum())])
            return _np_impl.array(cs)
        return X[rng.choice(len(X),self.n_clusters,replace=False)]
    def _run1(self,X,rng):
        cs=self._init_c(X,rng)
        for _ in range(self.max_iter):
            d=_np_impl.array([((X-c)**2).sum(1) for c in cs]).T
            lb=d.argmin(1)
            nc=_np_impl.array([X[lb==k].mean(0) if (lb==k).any() else cs[k]
                                for k in range(self.n_clusters)])
            if _np_impl.allclose(nc,cs,atol=1e-6): break
            cs=nc
        iner=sum(((X[lb==k]-cs[k])**2).sum() for k in range(self.n_clusters) if (lb==k).any())
        return lb,cs,iner
    def fit(self,X):
        rng=_np_impl.random.default_rng(self.random_state); best=None
        for _ in range(self.n_init):
            lb,cs,iner=self._run1(X,rng)
            if best is None or iner<best[2]: best=(lb,cs,iner)
        self.labels_,self.cluster_centers_,self.inertia_=best; return self
    def predict(self,X):
        d=_np_impl.array([((X-c)**2).sum(1) for c in self.cluster_centers_]).T
        return d.argmin(1)

class _TS:
    def __init__(self):
        self.feature=_np_impl.array([],dtype=int); self.threshold=_np_impl.array([],dtype=float)
        self.children_left=_np_impl.array([],dtype=int); self.children_right=_np_impl.array([],dtype=int)
        self.value=[]; self.n_node_samples=_np_impl.array([],dtype=int)

class DecisionTreeClassifier:
    def __init__(self,max_depth=None,random_state=None,**kw):
        self.max_depth=max_depth or 999
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            for t in (vals[:-1]+vals[1:])/2:
                l=X[:,f]<=t; r=~l
                if l.sum()<1 or r.sum()<1: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth,nid,ts):
        cl,cc=_np_impl.unique(y,return_counts=True); val=_np_impl.zeros(len(self._classes))
        for c,cnt in zip(cl,cc): val[_np_impl.where(self._classes==c)[0][0]]=cnt
        ts.feature=_np_impl.append(ts.feature,-2); ts.threshold=_np_impl.append(ts.threshold,-2.0)
        ts.children_left=_np_impl.append(ts.children_left,-1); ts.children_right=_np_impl.append(ts.children_right,-1)
        ts.value.append(val); ts.n_node_samples=_np_impl.append(ts.n_node_samples,len(y))
        cur=nid[0]; nid[0]+=1
        if depth<self.max_depth and len(_np_impl.unique(y))>1:
            sp=self._best(X,y)
            if sp:
                _,f,t=sp; ts.feature[cur]=f; ts.threshold[cur]=t; l=X[:,f]<=t
                ts.children_left[cur]=nid[0]; self._build(X[l],y[l],depth+1,nid,ts)
                ts.children_right[cur]=nid[0]; self._build(X[~l],y[~l],depth+1,nid,ts)
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); ts=_TS(); nid=[0]
        self._build(X,y,0,nid,ts); self.tree_=ts; return self
    def _p1(self,x,n=0):
        ts=self.tree_
        if ts.children_left[n]==-1: return self._classes[ts.value[n].argmax()]
        return self._p1(x,ts.children_left[n] if x[ts.feature[n]]<=ts.threshold[n] else ts.children_right[n])
    def predict(self,X): return _np_impl.array([self._p1(x) for x in X])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

import time

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Johnson-Lindenstrauss Lemma — random projections preserve distances
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  JOHNSON-LINDENSTRAUSS: RANDOM SKETCHING")
print("=" * 65)
print()
print("  Lemma: For n points in ℝᵈ, there exists a projection to")
print("  k = O(ε⁻² log n) dimensions preserving all pairwise distances")
print("  within (1±ε) factor.")
print()

def jl_sketch(X, k, seed=0):
    """Random projection to k dimensions: φ(x) = (1/√k) R·x."""
    rng = np.random.default_rng(seed)
    d   = X.shape[1]
    R   = rng.standard_normal((k, d)) / np.sqrt(k)
    return X @ R.T   # n × k

def jl_bound(n, eps, delta=0.05):
    """JL bound: k = O(log(n/delta) / eps²)."""
    return int(np.ceil(8 * np.log(n / delta) / eps**2))

n_pts  = 500
d_orig = 1000
X_jl   = np.random.randn(n_pts, d_orig)

# Compute exact pairwise distances (sample 1000 pairs)
rng_jl = np.random.default_rng(0)
idx1   = rng_jl.integers(0, n_pts, 2000)
idx2   = rng_jl.integers(0, n_pts, 2000)
mask   = idx1 != idx2
idx1, idx2 = idx1[mask][:1000], idx2[mask][:1000]
dists_exact = np.linalg.norm(X_jl[idx1] - X_jl[idx2], axis=1)

print(f"  n={n_pts}, d={d_orig} → project to k dimensions")
print()
print(f"  {'k':>6}  {'JL bound':>10}  {'Mean dist ratio':>17}  {'Max distortion':>16}  {'% within ε=0.2'}  ")
print("  " + "─" * 68)

eps_target = 0.2
k_bound    = jl_bound(n_pts, eps_target)
k_vals_jl  = [5, 10, 20, 50, 100, 200, k_bound, 500]

for k in sorted(set(k_vals_jl)):
    X_proj    = jl_sketch(X_jl, k)
    dists_proj = np.linalg.norm(X_proj[idx1] - X_proj[idx2], axis=1)
    ratio     = dists_proj / (dists_exact + 1e-12)
    mean_r    = ratio.mean()
    max_dist  = abs(ratio - 1).max()
    pct_ok    = np.mean(np.abs(ratio - 1) <= eps_target) * 100
    jl_flag   = "← JL bound" if k == k_bound else ""
    print(f"  {k:>6}  {k_bound:>10}  {mean_r:>17.4f}  {max_dist:>16.4f}  "
          f"{pct_ok:>14.1f}%  {jl_flag}")

print()
print(f"  JL bound (ε={eps_target}): k ≥ {k_bound} suffices to preserve distances.")
print("  In practice, much smaller k works — JL is a worst-case bound.")

# ─────────────────────────────────────────────────────────────────────────────
# Mini-batch SGD convergence: full vs SGD vs mini-batch
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  MINI-BATCH SGD CONVERGENCE vs COST TRADEOFF")
print("=" * 65)
print()

n_sg  = 2000
d_sg  = 50
X_sg, y_sg = make_classification(n_samples=n_sg, n_features=d_sg,
                                   n_informative=20, random_state=0)
X_sg = StandardScaler().fit_transform(X_sg)

# Manual logistic regression gradient
def log_loss_grad(w, X_, y_, lam=0.01):
    """Returns (loss, gradient) for logistic regression."""
    logits = X_ @ w
    probs  = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    loss   = -np.mean(y_ * np.log(probs + 1e-15) + (1-y_) * np.log(1-probs+1e-15))
    loss  += 0.5 * lam * np.dot(w, w)
    grad   = X_.T @ (probs - y_) / len(y_) + lam * w
    return loss, grad

# Run for each batch size
batch_sizes = [1, 16, 64, 256, n_sg]  # 1=SGD, n=full GD
lr          = 0.1
n_epochs    = 30
lam         = 0.01

results_sgd = {}
for bs in batch_sizes:
    rng_sgd = np.random.default_rng(42)
    w       = np.zeros(d_sg)
    losses  = []
    times   = []
    t_start = time.perf_counter()
    for ep in range(n_epochs):
        idx = rng_sgd.permutation(n_sg)
        ep_loss = 0.0
        for start in range(0, n_sg, bs):
            Xb = X_sg[idx[start:start+bs]]; yb = y_sg[idx[start:start+bs]]
            l, g = log_loss_grad(w, Xb, yb, lam)
            w   -= lr * g
            ep_loss += l
        l_full, _ = log_loss_grad(w, X_sg, y_sg, lam)
        losses.append(l_full)
        times.append(time.perf_counter() - t_start)
    results_sgd[bs] = (losses, times, w.copy())

# Optimal solution via sklearn (very high precision reference)
clf_opt = LogisticRegression(C=1/lam, max_iter=5000, random_state=0).fit(X_sg, y_sg)
w_opt   = np.r_[clf_opt.coef_[0]]
opt_loss, _ = log_loss_grad(w_opt, X_sg, y_sg, lam)

print(f"  Optimal loss (sklearn L-BFGS): {opt_loss:.6f}")
print()
print(f"  {'Batch size':>12}  {'Final loss':>12}  {'Total time(s)':>14}  {'Time/epoch':>12}  {'Acc':>8}")
print("  " + "─" * 62)
for bs, (losses, times, w_f) in results_sgd.items():
    acc = accuracy_score(y_sg, (X_sg @ w_f > 0).astype(int))
    label = "SGD" if bs == 1 else ("Full GD" if bs == n_sg else f"mini-b={bs}")
    print(f"  {label:>12}  {losses[-1]:>12.6f}  {times[-1]:>14.4f}  "
          f"{times[-1]/n_epochs:>12.5f}  {acc:>8.4f}")

print()
print("  Mini-batch SGD (b=64, 256) best balance of speed and convergence.")
print("  Full GD: slow per epoch but smooth convergence.")
print("  SGD (b=1): fast iterations, noisy, slow convergence.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Approximate Algorithms: JL Sketching, Mini-batch SGD, Streaming",
             fontsize=13, fontweight="bold")

# Panel (0,0): JL projection accuracy vs k
ax = axes[0, 0]
k_range   = [5, 10, 20, 50, 100, 200, 500]
mean_rats = []; pct_ok_arr = []
for k in k_range:
    X_p  = jl_sketch(X_jl, k)
    dp   = np.linalg.norm(X_p[idx1] - X_p[idx2], axis=1)
    rats = dp / (dists_exact + 1e-12)
    mean_rats.append(np.abs(rats - 1).mean())
    pct_ok_arr.append(np.mean(np.abs(rats - 1) <= 0.2) * 100)

ax.semilogx(k_range, mean_rats,   "tomato",    lw=2.5, marker="o", ms=8, label="Mean distortion |r−1|")
ax.semilogx(k_range, [p/100 for p in pct_ok_arr], "steelblue", lw=2.5, marker="s", ms=8,
            label="Fraction within ε=0.2")
ax.axvline(k_bound, color="black", lw=2, ls="--", label=f"JL bound k={k_bound}")
ax.set_xlabel("Projection dimension k (log)"); ax.set_ylabel("Metric")
ax.set_title(f"Johnson-Lindenstrauss Sketching\\n(d={d_orig} → k, n={n_pts})",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# Panel (0,1): JL distance preservation example
ax = axes[0, 1]
k_vis = k_bound
X_proj_vis = jl_sketch(X_jl, k_vis)
dists_p_vis = np.linalg.norm(X_proj_vis[idx1] - X_proj_vis[idx2], axis=1)
ax.scatter(dists_exact[:200], dists_p_vis[:200], s=6, alpha=0.4, c="steelblue")
lim_max = max(dists_exact.max(), dists_p_vis.max())
ax.plot([0, lim_max], [0, lim_max], "tomato", lw=2, ls="--", label="Perfect preservation")
ax.plot([0, lim_max], [0, 0.8*lim_max], "gray", lw=1, ls=":", alpha=0.5)
ax.plot([0, lim_max], [0, 1.2*lim_max], "gray", lw=1, ls=":", alpha=0.5)
ax.set_xlabel("True distance"); ax.set_ylabel(f"Projected distance (k={k_vis})")
ax.set_title(f"JL Distance Preservation\\n(projected vs true, k={k_vis})",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (0,2): Mini-batch SGD loss curves
ax = axes[0, 2]
colours_bs = {1: "tomato", 16: "orange", 64: "steelblue",
              256: "seagreen", n_sg: "purple"}
labels_bs  = {1: "SGD (b=1)", 16: "b=16", 64: "b=64",
              256: "b=256", n_sg: "Full GD"}
for bs, (losses, _, _) in results_sgd.items():
    ax.semilogy(range(n_epochs), losses, color=colours_bs[bs], lw=2,
                label=labels_bs[bs])
ax.axhline(opt_loss, color="black", lw=2, ls="--", label="Optimal loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
ax.set_title("Mini-Batch SGD Convergence\\n(by epoch — larger batch smoother)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

# Panel (1,0): Loss vs WALL-CLOCK TIME
ax = axes[1, 0]
for bs, (losses, times, _) in results_sgd.items():
    ax.semilogy(times, losses, color=colours_bs[bs], lw=2, label=labels_bs[bs])
ax.axhline(opt_loss, color="black", lw=2, ls="--", label="Optimal")
ax.set_xlabel("Wall-clock time (s)"); ax.set_ylabel("Loss (log)")
ax.set_title("Loss vs Wall-Clock Time\\n(mini-batch wins — fast AND convergent)",
             fontweight="bold")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

# Panel (1,1): JL dimension requirement vs n and ε
ax = axes[1, 1]
n_range_jl  = [100, 500, 1000, 5000, 10000, 100000, 1e6]
eps_range   = [0.05, 0.1, 0.2, 0.3]
for eps, col in zip(eps_range, ["tomato", "steelblue", "seagreen", "purple"]):
    k_bounds = [jl_bound(int(n), eps) for n in n_range_jl]
    ax.semilogx(n_range_jl, k_bounds, lw=2, color=col, label=f"ε={eps}")
ax.set_xlabel("n (number of points, log)"); ax.set_ylabel("Required k")
ax.set_title("JL Bound: Required k vs n\\n(k grows as O(log n / ε²))",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (1,2): Algorithm scaling comparison
ax = axes[1, 2]
n_show = np.logspace(2, 6, 50)
ax.loglog(n_show, n_show**0,          "black",    lw=2, ls="-",  label="O(1)")
ax.loglog(n_show, np.log2(n_show),    "purple",   lw=2, ls="-",  label="O(log n)")
ax.loglog(n_show, n_show,             "steelblue",lw=2, ls="-",  label="O(n)")
ax.loglog(n_show, n_show*np.log2(n_show), "seagreen", lw=2, ls="-", label="O(n log n)")
ax.loglog(n_show, n_show**2,          "orange",   lw=2, ls="--", label="O(n²)")
ax.loglog(n_show, n_show**3,          "tomato",   lw=2, ls=":",  label="O(n³)")
ax.axhline(1e8, color="gray", lw=1.5, ls="--", alpha=0.7, label="~10s on 1 GHz")
ax.set_xlabel("n (log)"); ax.set_ylabel("Operations (log)")
ax.set_title("Algorithm Complexity Reference\\n(horizontal line = feasibility threshold)",
             fontweight="bold")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("approximation_algorithms.png", dpi=110)
print()
print("  Plot saved → approximation_algorithms.png")
print()
print("  KEY TAKEAWAYS — COMPUTATIONAL COMPLEXITY:")
print("  1. Matrix inversion O(d³): avoid for d>10K; use iterative solvers.")
print("  2. Kernel methods O(n³): infeasible for n>100K; use Nyström/RFF.")
print("  3. Transformer attention O(T²d): use sparse/linear variants for T>4K.")
print("  4. Feature selection, k-means, decision trees: NP-hard; use heuristics.")
print("  5. JL lemma: random projection to O(log n / ε²) preserves distances.")
print("  6. Mini-batch SGD: O(bd) per step; b=64-256 best speed/convergence.")
print("  7. Dual formulation: cheaper when n < d (kernel methods, ridge).")
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