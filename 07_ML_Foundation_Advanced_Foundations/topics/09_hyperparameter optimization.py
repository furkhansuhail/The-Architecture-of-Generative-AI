"""
Hyperparameter Optimisation
============================

Grid search, random search, Bayesian optimisation, and multi-fidelity
methods — the systematic framework for finding the best model configuration
without overfitting to the validation set.

"""

import textwrap
import re

TOPIC_NAME = "Hyperparameter Optimisation"
DISPLAY_NAME = "09 · Hyperparameter Optimisation"
ICON = "🔧"
SUBTITLE = "Finding the Best Model Configuration Systematically"

THEORY = """

### What Are Hyperparameters?

Model parameters (weights w) are learned from data by minimising a loss.
Hyperparameters (HPs) are configuration choices that are set BEFORE training —
they control the learning process itself.

    ┌──────────────────────────────────────────────────────────────────┐
    │  PARAMETERS (learned):   network weights, SVM support vectors,   │
    │                           tree split thresholds.                 │
    │                                                                  │
    │  HYPERPARAMETERS (set before training):                          │
    │  ─ Regularisation:  λ, dropout rate, weight decay                │
    │  ─ Architecture:    depth, width, kernel size, n_layers          │
    │  ─ Optimisation:    learning rate, batch size, momentum, β       │
    │  ─ Data:            augmentation type, normalisation strategy    │
    │  ─ Model type:      kernel choice, n_estimators, max_depth       │
    └──────────────────────────────────────────────────────────────────┘

    The hyperparameter optimisation (HPO) problem:

        θ* = argmin_{θ ∈ Θ} L_val(f_θ(D_train), D_val)

    where θ ∈ Θ is the hyperparameter configuration,
    f_θ(D_train) = model trained with HPs θ on training data,
    L_val = validation loss evaluated on the validation set.

    This is a NESTED optimisation — the inner loop trains the model,
    the outer loop searches for the best HP configuration.


──────────────────────────────────────────────────────────────────────────────
### Grid Search — Exhaustive but Expensive

Grid search evaluates every combination of HPs on a predefined grid.

    Algorithm:
    ─ Define a grid: λ ∈ {0.001, 0.01, 0.1, 1.0}, depth ∈ {3, 5, 7}.
    ─ Evaluate all 4 × 3 = 12 combinations by cross-validation.
    ─ Return the combination with lowest validation error.

    Cost: O(K₁ × K₂ × ... × Kₙ × T_train)
    where Kᵢ = number of grid points for HP i, T_train = training cost.

    Diagram 1 — Grid Search Coverage:

    λ
    ┤  ×  ×  ×  ×
    ┤  ×  ×  ×  ×
    ┤  ×  ×  ×  ×    Each × = one training run
    ┤  ×  ×  ×  ×
    └─────────────── depth

    Weakness: exponential in the number of HPs.
    With 10 HPs each with 10 values: 10^10 evaluations — infeasible.

    Strengths: simple to implement, reproducible, parallelisable.
    Appropriate for: ≤ 2-3 HPs with a small number of discrete values.


──────────────────────────────────────────────────────────────────────────────
### Random Search — Surprisingly More Efficient

Random search (Bergstra & Bengio, 2012) samples HP configurations uniformly
at random from the search space rather than from a fixed grid.

    Diagram 2 — Random Search vs Grid Search (2D HP space):

    Grid search (9 evals):      Random search (9 evals):
    λ                            λ
    ┤  ×  ×  ×                   ┤  ×     ×      ×
    ┤  ×  ×  ×                   ┤     ×    ×
    ┤  ×  ×  ×                   ┤  ×       ×  ×
    └──────────── depth          └──────────────── depth

    Key insight: in high-dimensional HP spaces, most HPs are irrelevant
    for a particular dataset. The important HPs might be just 1-2 out of 10.

    ┌──────────────────────────────────────────────────────────────────┐
    │  If only 2 out of 10 HPs matter, grid search wastes 10^8 evals   │
    │  on all combinations of the irrelevant 8 HPs.                    │
    │  Random search: every evaluation explores a different value of   │
    │  the important HPs, regardless of the irrelevant ones.           │
    │                                                                  │
    │  For a budget of B evaluations:                                  │
    │  Grid search covers √B unique values per HP (2D case).           │
    │  Random search covers B unique values for every HP.              │
    └──────────────────────────────────────────────────────────────────┘

    Strengths: simple, parallelisable, efficient in high dimensions.
    Weakness: ignores information from past evaluations — treats each
    evaluation as independent, makes no use of the "landscape".


──────────────────────────────────────────────────────────────────────────────
### Bayesian Optimisation — Learning the Landscape

Bayesian optimisation (BO) uses past evaluations to build a probabilistic
model of the objective function, then uses that model to decide where to
evaluate next.

    Key insight: evaluating f(θ) is expensive (train a model).
    A probabilistic surrogate model is cheap to evaluate.
    Use the surrogate to select the most promising θ* next.

    ┌──────────────────────────────────────────────────────────────────┐
    │  BAYESIAN OPTIMISATION LOOP:                                     │
    │                                                                  │
    │  1. Evaluate f(θ) at a few random initial points.                │
    │  2. Fit a surrogate model (GP or Tree Parzen Estimator) to       │
    │     all observed (θ, f(θ)) pairs.                                │
    │  3. Maximise an acquisition function α(θ) using the surrogate.   │
    │  4. Evaluate f at θ* = argmax α(θ).                              │
    │  5. Update the surrogate with the new observation.               │
    │  6. Go to step 3.                                                │
    └──────────────────────────────────────────────────────────────────┘

**Surrogate Model — Gaussian Process:**

    Model the objective as: f(θ) ~ GP(μ(θ), k(θ, θ'))
    Posterior after observing (θ₁,y₁),...,(θₙ,yₙ):
        μₙ(θ) = kₙᵀ(Kₙ + σ²I)⁻¹y       (posterior mean)
        σₙ²(θ) = k(θ,θ) − kₙᵀ(Kₙ+σ²I)⁻¹kₙ  (posterior variance)

    The GP provides both a PREDICTION of performance and an UNCERTAINTY.
    This uncertainty is used by the acquisition function.

**Acquisition Functions — Exploration vs Exploitation:**

    EXPECTED IMPROVEMENT (EI):
    ┌──────────────────────────────────────────────────────────────────┐
    │  EI(θ) = E[max(f(θ) − f*, 0)]                                    │
    │                                                                  │
    │  = (μₙ(θ) − f*) Φ(Z) + σₙ(θ) φ(Z)                                 │
    │  where Z = (μₙ(θ) − f*) / σₙ(θ),  f* = best observed so far       │
    │  Φ = normal CDF, φ = normal PDF                                  │
    └──────────────────────────────────────────────────────────────────┘

    UPPER CONFIDENCE BOUND (UCB):
        UCB(θ) = μₙ(θ) + β · σₙ(θ)
        β controls exploration vs exploitation tradeoff.

    PROBABILITY OF IMPROVEMENT (PI):
        PI(θ) = P(f(θ) > f* + ε) = Φ((μₙ(θ) − f* − ε) / σₙ(θ))

    Diagram 3 — GP Surrogate and EI Acquisition:

    f(θ)
      │         True f (unknown)
      │    ·   · · ·  ·  ·
      │─────────────────────────── GP posterior mean
      │   [   uncertainty band   ]
      └──────────────────────────── θ

    EI(θ)
      │                  ↑
      │               ↑     ↑      EI is high where:
      │             ↑          ↑   (1) μₙ(θ) > f* (exploit)
      └──────────────────────────── θ  (2) σₙ(θ) large (explore)
                    ★  next query point = max EI


──────────────────────────────────────────────────────────────────────────────
### Tree Parzen Estimator (TPE)

TPE (Bergstra et al., 2011) is the HP optimisation algorithm used in
Hyperopt and Optuna — one of the most practical BO variants.

    Instead of modelling f(θ) directly, TPE models:
    ─ p(θ | y < y*):  distribution of θ that gave GOOD results.
    ─ p(θ | y ≥ y*):  distribution of θ that gave BAD results.
    where y* is a quantile threshold (e.g., best 25% of evaluations).

    Acquisition: EI ∝ ℓ(θ) / g(θ)
    where ℓ(θ) = p(θ | good), g(θ) = p(θ | bad).
    Sample candidates from ℓ(θ), select the one with highest ℓ/g ratio.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Advantages of TPE:                                              │
    │  ─ Handles conditional HP spaces (e.g., kernel type in SVM:      │
    │    if kernel="rbf", then use γ; if kernel="poly", use degree).   │
    │  ─ Scales better than GP to 10+ HPs.                             │
    │  ─ Each model fit is O(n log n) instead of O(n³) for GP.         │
    │  ─ Handles categorical, continuous, and integer HPs natively.    │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Multi-Fidelity Methods — Hyperband and ASHA

For large models, evaluating each HP configuration fully is expensive.
Multi-fidelity methods use cheap approximations to discard bad configs early.

**Successive Halving:**

    1. Start with n configurations, evaluated for r₀ resources each.
    2. Keep the top 1/η, double resources: evaluate for η·r₀ each.
    3. Repeat until 1 configuration remains (fully evaluated).

    Budget: n · r₀ · (1 + 1/η + 1/η² + ...) ≈ n·r₀·η/(η-1)  (geometric series)

    ┌──────────────────────────────────────────────────────────────────┐
    │  SUCCESSIVE HALVING (n=8, η=2):                                  │
    │                                                                  │
    │  Round 1: ████ ████ ████ ████ ████ ████ ████ ████  (8 configs)   │
    │           r=1  each                                              │
    │                                                                  │
    │  Round 2: ████████ ████████ ████████ ████████  (top 4)           │
    │           r=2  each                                              │
    │                                                                  │
    │  Round 3: ████████████████ ████████████████    (top 2)           │
    │           r=4  each                                              │
    │                                                                  │
    │  Round 4: ████████████████████████████████     (winner)          │
    │           r=8  each                                              │
    └──────────────────────────────────────────────────────────────────┘

**Hyperband (Li et al., 2018):**

    Challenge with successive halving: how to set n and r₀?
    If n is large and r₀ is small: aggressive early elimination.
    If n is small and r₀ is large: careful but few candidates.

    Hyperband runs successive halving for MULTIPLE BRACKETS with different
    n/r₀ tradeoffs, allocating a total budget B.
    This removes the need to choose n vs r₀ — both are tried.

**ASHA (Asynchronous Successive Halving):**

    Parallel version — workers report results asynchronously.
    Eliminates the synchronisation barrier in Hyperband.
    Used in Ray Tune — the practical standard for large-scale HPO.


──────────────────────────────────────────────────────────────────────────────
### Avoiding HP Overfitting

    ┌──────────────────────────────────────────────────────────────────┐
    │  CRITICAL: Every HP evaluation that uses the validation set      │
    │  "uses up" some validation data. With enough HP trials, you      │
    │  will find a configuration that happens to work well on the      │
    │  validation set by chance — even if it doesn't generalise.       │
    │                                                                  │
    │  Solutions:                                                      │
    │  1. Hold out a FINAL TEST SET never touched during HPO.          │
    │  2. Use k-fold cross-validation for each HP evaluation.          │
    │  3. Apply Bonferroni correction for statistical testing.         │
    │  4. Report variance across multiple HP seeds.                    │
    └──────────────────────────────────────────────────────────────────┘

    Rule of thumb: with B HP evaluations on a validation set of size m,
    the effective overfitting is O(√(B log B / m)).
    Large B and small validation set → significant HP overfitting risk.

"""

OPERATIONS = {

    "1 · Grid vs Random vs Bayesian Search — HPO Comparison": {
        "description": (
            "Implements and compares three HPO strategies on a GBM classifier: "
            "grid search, random search, and a from-scratch Gaussian Process "
            "Bayesian optimisation with Expected Improvement acquisition. "
            "Shows convergence curves (best validation accuracy vs number "
            "of evaluations), the GP surrogate model after each step, and "
            "the EI landscape. Demonstrates that BO finds good configurations "
            "in far fewer evaluations than grid or random search."
        ),
        "timeout":300,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl
import scipy.optimize as _sp_opt
import copy as _copy

def make_classification(n_samples=100,n_features=20,n_informative=2,
                        n_redundant=2,n_repeated=0,random_state=None,**kw):
    rng=_np_impl.random.default_rng(random_state)
    y=rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=n_redundant
    Xr=(Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr))
        if nr>0 else _np_impl.empty((n_samples,0)))
    nn2=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    parts=[a for a in [Xi,Xr,Xn] if a.shape[1]>0]
    return _np_impl.hstack(parts)[:,:n_features],y.astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class _DTR:
    def __init__(self,max_depth=3,min_s=5): self.max_depth=max_depth; self.min_s=min_s
    def _split(self,X,r):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>10: ts=ts[_np_impl.linspace(0,len(ts)-1,10).astype(int)]
            for t in ts:
                l=X[:,f]<=t; rr=~l
                if l.sum()<self.min_s or rr.sum()<self.min_s: continue
                g=_np_impl.var(r)*len(r)-_np_impl.var(r[l])*l.sum()-_np_impl.var(r[rr])*rr.sum()
                if best is None or g>best[0]: best=(g,f,t,l)
        return best
    def _build(self,X,r,d):
        if d>=self.max_depth or len(r)<=self.min_s or _np_impl.var(r)<1e-12: return ('L',r.mean())
        sp=self._split(X,r)
        if sp is None: return ('L',r.mean())
        _,f,t,l=sp
        return ('N',f,t,self._build(X[l],r[l],d+1),self._build(X[~l],r[~l],d+1))
    def fit(self,X,r): self._tree=self._build(X,r,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])

class GradientBoostingClassifier:
    def __init__(self,n_estimators=15,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        self._classes=_np_impl.unique(y)
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_DTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return self._classes[(_sig(F)>=0.5).astype(int)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

def cross_val_score(estimator,X,y,cv=3,scoring='accuracy'):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]))
    return _np_impl.array(scores)

from scipy.stats import norm as scipy_norm

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                            random_state=0)
X = StandardScaler().fit_transform(X)

def evaluate_hp(log_lr, max_depth):
    """Train GBM and return 3-fold CV accuracy. The expensive function."""
    lr  = 10**log_lr
    md  = int(np.clip(np.round(max_depth), 1, 8))
    gbm = GradientBoostingClassifier(learning_rate=lr, max_depth=md,
                                      n_estimators=15, random_state=0)
    return cross_val_score(gbm, X, y, cv=3, scoring="accuracy").mean()

# True 2D landscape (precomputed for plotting)
log_lr_range = np.linspace(-4, -0.3, 8)
depth_range  = np.linspace(1, 8, 8)
print("  Computing true HPO landscape (8×8 grid)... ", end="", flush=True)
landscape = np.array([[evaluate_hp(lr, d) for lr in log_lr_range]
                       for d in depth_range])
best_true  = landscape.max()
best_lr_d  = np.unravel_index(landscape.argmax(), landscape.shape)
print(f"done. Best={best_true:.4f} at log_lr={log_lr_range[best_lr_d[1]]:.2f}, "
      f"depth={depth_range[best_lr_d[0]]:.1f}")

print()
print("=" * 65)
print("  HPO COMPARISON: GRID vs RANDOM vs BAYESIAN OPTIMISATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────────────────────────────────────

lr_grid_vals  = np.linspace(-4, -0.3, 4)
dep_grid_vals = np.array([1, 3, 7])
grid_results  = []
for lr_v in lr_grid_vals:
    for dep_v in dep_grid_vals:
        score = evaluate_hp(lr_v, dep_v)
        grid_results.append((lr_v, dep_v, score))

grid_best = max(grid_results, key=lambda x: x[2])
grid_best_vals = np.maximum.accumulate([r[2] for r in grid_results])
print(f"  Grid search ({len(grid_results)} evals): best={grid_best[2]:.4f}  "
      f"(log_lr={grid_best[0]:.2f}, depth={grid_best[1]:.0f})")

# ─────────────────────────────────────────────────────────────────────────────
# Random Search
# ─────────────────────────────────────────────────────────────────────────────

rng_rs      = np.random.default_rng(1)
n_random    = len(grid_results)
random_lrs  = rng_rs.uniform(-4, -0.3, n_random)
random_deps = rng_rs.uniform(1, 8, n_random)
random_results = []
for lr_v, dep_v in zip(random_lrs, random_deps):
    score = evaluate_hp(lr_v, dep_v)
    random_results.append((lr_v, dep_v, score))

rand_best = max(random_results, key=lambda x: x[2])
rand_best_vals = np.maximum.accumulate([r[2] for r in random_results])
print(f"  Random search ({len(random_results)} evals): best={rand_best[2]:.4f}  "
      f"(log_lr={rand_best[0]:.2f}, depth={rand_best[1]:.1f})")

# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Optimisation with GP (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def gp_posterior(X_obs, y_obs, X_pred, length_scales, noise=1e-3):
    """GP regression with RBF kernel."""
    def rbf(A, B, ls):
        diffs = (A[:, None, :] - B[None, :, :]) / ls
        return np.exp(-0.5 * np.sum(diffs**2, axis=-1))
    K   = rbf(X_obs, X_obs, length_scales) + noise*np.eye(len(X_obs))
    Ks  = rbf(X_obs, X_pred, length_scales)
    Kss = rbf(X_pred, X_pred, length_scales)
    L   = np.linalg.cholesky(K + 1e-6*np.eye(len(K)))
    alpha  = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    mu     = Ks.T @ alpha
    v      = np.linalg.solve(L, Ks)
    var    = np.diag(Kss) - np.einsum("ij,ij->j", v, v)
    return mu, np.maximum(var, 0)

def expected_improvement(mu, sigma, best_y):
    Z  = (mu - best_y) / (sigma + 1e-9)
    ei = (mu - best_y) * scipy_norm.cdf(Z) + sigma * scipy_norm.pdf(Z)
    return np.maximum(ei, 0)

# HP search space: [log_lr, depth] normalised to [0,1]
def normalise(lr, dep):
    return np.array([(lr - (-4)) / (-0.3 - (-4)), (dep - 1) / 7.0])

def denormalise(x):
    return x[0] * (-0.3 - (-4)) + (-4), x[1] * 7.0 + 1

# BO grid for candidate evaluation
candidate_grid = np.array([[normalise(lr, d) for lr in np.linspace(-4,-0.3,15)]
                             for d in np.linspace(1,8,15)]).reshape(-1, 2)

# Initial random evaluations (3 warmup points)
rng_bo  = np.random.default_rng(7)
n_init  = 3
n_bo    = len(grid_results) - n_init
bo_configs = rng_bo.uniform(0, 1, (n_init, 2))
bo_scores  = [evaluate_hp(*denormalise(c)) for c in bo_configs]

bo_best_vals = list(np.maximum.accumulate(bo_scores))
ls = np.array([0.3, 0.3])

for trial in range(n_bo):
    X_obs = bo_configs; y_obs = np.array(bo_scores)
    mu, var = gp_posterior(X_obs, y_obs, candidate_grid, ls)
    sigma   = np.sqrt(var)
    ei      = expected_improvement(mu, sigma, max(y_obs))
    best_cand = candidate_grid[np.argmax(ei)]
    lr_next, dep_next = denormalise(best_cand)
    score_next = evaluate_hp(lr_next, dep_next)
    bo_configs = np.vstack([bo_configs, best_cand])
    bo_scores.append(score_next)
    bo_best_vals.append(max(bo_best_vals[-1], score_next))

bo_best = max(bo_scores)
print(f"  Bayesian opt ({len(bo_scores)} evals): best={bo_best:.4f}")
print()
print(f"  True best (20×20 exhaustive): {best_true:.4f}")
print()
print(f"  Gap from true best:")
print(f"    Grid search: {best_true - grid_best[2]:+.4f}")
print(f"    Random:      {best_true - rand_best[2]:+.4f}")
print(f"    BO (GP):     {best_true - bo_best:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("HPO: Grid Search vs Random Search vs Bayesian Optimisation",
             fontsize=13, fontweight="bold")

# Panel (0,0): True landscape
ax = axes[0, 0]
im = ax.contourf(log_lr_range, depth_range, landscape, levels=20, cmap="viridis")
plt.colorbar(im, ax=ax, label="CV accuracy")
ax.scatter(*[log_lr_range[best_lr_d[1]], depth_range[best_lr_d[0]]],
           c="red", s=200, marker="*", zorder=10, label="True best")
ax.set_xlabel("log₁₀(learning rate)"); ax.set_ylabel("max_depth")
ax.set_title("True HPO Landscape (20×20 exhaustive)\\n(★=global optimum)",
             fontweight="bold")
ax.legend(fontsize=9)

# Panel (0,1): Grid and random search coverage
ax = axes[0, 1]
ax.contourf(log_lr_range, depth_range, landscape, levels=20, cmap="viridis", alpha=0.4)
grid_pts = np.array([(r[0],r[1]) for r in grid_results])
rand_pts = np.array([(r[0],r[1]) for r in random_results])
ax.scatter(grid_pts[:,0], grid_pts[:,1], c="steelblue", s=40, zorder=5, label="Grid")
ax.scatter(rand_pts[:,0], rand_pts[:,1], c="tomato",    s=40, marker="^", zorder=5, label="Random")
ax.set_xlabel("log₁₀(lr)"); ax.set_ylabel("depth")
ax.set_title(f"Grid vs Random Coverage\\n(each {len(grid_results)} evaluations)",
             fontweight="bold")
ax.legend(fontsize=9)

# Panel (0,2): BO evaluated points + GP posterior
ax = axes[0, 2]
ax.contourf(log_lr_range, depth_range, landscape, levels=20, cmap="viridis", alpha=0.4)
# Denormalise BO points
bo_pts_orig = np.array([denormalise(c) for c in bo_configs])
colours_bo  = plt.cm.Reds(np.linspace(0.3, 1.0, len(bo_configs)))
for i, (pt, sc) in enumerate(zip(bo_pts_orig, bo_scores)):
    ax.scatter(pt[0], pt[1], c=[colours_bo[i]], s=60, zorder=5)
    if i < 3:
        ax.annotate(f"init", xy=pt, fontsize=6, color="gray")
ax.set_xlabel("log₁₀(lr)"); ax.set_ylabel("depth")
ax.set_title(f"BO Evaluated Points (colour=order)\\n({len(bo_configs)} total evaluations)",
             fontweight="bold")

# Panel (1,0): Convergence curves
ax = axes[1, 0]
n_evals = range(1, len(grid_results)+1)
ax.plot(n_evals, grid_best_vals, "steelblue", lw=2.5, marker="o", ms=4, label="Grid search")
ax.plot(n_evals, rand_best_vals, "tomato",    lw=2.5, marker="s", ms=4, label="Random search")
ax.plot(range(1, len(bo_best_vals)+1), bo_best_vals, "seagreen", lw=2.5,
        marker="^", ms=4, label="Bayesian opt (GP)")
ax.axhline(best_true, color="black", lw=2, ls="--", label=f"True best={best_true:.4f}")
ax.set_xlabel("Number of evaluations"); ax.set_ylabel("Best CV accuracy so far")
ax.set_title("Convergence Curves\\n(BO finds good configs faster)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel (1,1): GP surrogate after N evaluations
ax = axes[1, 1]
# Show GP mean over the 2D space
gp_grid_2d = np.array([[normalise(lr, d) for lr in log_lr_range]
                         for d in depth_range]).reshape(-1, 2)
mu_gp, var_gp = gp_posterior(bo_configs, np.array(bo_scores), gp_grid_2d, ls)
mu_gp_2d = mu_gp.reshape(len(depth_range), len(log_lr_range))
im2 = ax.contourf(log_lr_range, depth_range, mu_gp_2d, levels=20, cmap="viridis")
plt.colorbar(im2, ax=ax, label="GP posterior mean")
ax.scatter(bo_pts_orig[:,0], bo_pts_orig[:,1], c="white", s=40, zorder=5, edgecolors="black")
ax.set_xlabel("log₁₀(lr)"); ax.set_ylabel("depth")
ax.set_title(f"GP Surrogate (posterior mean, {len(bo_configs)} obs)\\n"
             f"(white dots = evaluations so far)", fontweight="bold")

# Panel (1,2): Strategy comparison table
ax = axes[1, 2]
ax.axis("off")
rows = [
    ["Grid search",   "O(∏Kᵢ)",  "None",     "Simple, exhaustive",  "Low-d, discrete HPs"],
    ["Random search", "O(B)",      "None",     "High-d, irrelevant HPs", "Good default"],
    ["GP Bayes opt",  "O(B·n³)",   "High",     "Low-B, expensive f",  "≤10 HPs, ≤200 trials"],
    ["TPE (Optuna)",  "O(B·nlogn)","Medium",   "Conditional spaces",  "Practical standard"],
    ["Hyperband",     "O(B)",      "Low",      "Neural net HP search", "Large-scale, parallel"],
    ["ASHA",          "O(B)",      "Very low", "Async parallel",      "Ray Tune, very large"],
]
headers = ["Method", "Cost/iter", "Overhead", "Best when", "Use case"]
table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", loc="center")
table.auto_set_font_size(False); table.set_fontsize(8)
table.scale(1.1, 1.85)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif r == 4:
        cell.set_facecolor("#e8f5e8")
ax.set_title("HPO Methods Comparison", fontweight="bold")

plt.tight_layout()
plt.savefig("hpo_comparison.png", dpi=110)
print()
print("  Plot saved → hpo_comparison.png")
print()
print("  KEY TAKEAWAYS — HYPERPARAMETER OPTIMISATION:")
print("  1. Grid search is exponential in the number of HPs — use for ≤3 HPs.")
print("  2. Random search is better than grid when few HPs actually matter.")
print("  3. Bayesian optimisation uses past results to guide the search.")
print("  4. EI acquisition balances exploitation (go to best) and exploration.")
print("  5. TPE (Optuna) handles conditional HP spaces and scales better than GP.")
print("  6. Hyperband/ASHA: use cheap approximations to discard bad configs early.")
print("  7. Always keep a held-out test set — HPO overfits the validation set.")
''',
    },
}

for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def get_content():
    return {
        "display_name": DISPLAY_NAME, "icon": ICON, "subtitle": SUBTITLE,
        "theory": THEORY, "visual_html": "", "visual_height": 400,
        "complexity": None, "operations": OPERATIONS,
    }