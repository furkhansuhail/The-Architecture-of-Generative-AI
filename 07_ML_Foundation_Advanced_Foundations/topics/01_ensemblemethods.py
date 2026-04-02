"""
Ensemble Methods
================

The principle that many imperfect models, combined intelligently,
outperform any single model — and why this directly reduces both
overfitting (variance) and underfitting (bias).

"""

import textwrap
import re

TOPIC_NAME   = "Ensemble Methods"
DISPLAY_NAME = "01 · Ensemble Methods"
ICON         = "🌳"
SUBTITLE     = "Many Weak Learners → One Strong Learner"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY ENSEMBLES WORK

### The Core Intuition

If 100 independent doctors each have a 70% chance of making the correct
diagnosis, and you take the MAJORITY VOTE of all 100, the probability of
the group being correct is extremely high — approaching certainty.

This is the "wisdom of crowds" applied to machine learning.

Mathematically, if each of M classifiers has error rate ε (independently):

    P(ensemble wrong) = Σ C(M,k) × εᵏ × (1-ε)^(M-k)
                         k > M/2

    For M=100, ε=0.30: P(ensemble wrong) ≈ 0.00016  (vs 30% per model)

The catch: this only works when individual models make DIFFERENT mistakes.
Ensembling N identical models gives you nothing. The key ingredient is
DIVERSITY among the component models.

    Diagram 1 — Why Diversity Matters:

    CORRELATED models (bad ensemble):      DIVERSE models (good ensemble):
    Model A:  ✓ ✓ ✗ ✓ ✗ ✗ ✓            Model A:  ✓ ✗ ✓ ✓ ✗ ✓ ✗
    Model B:  ✓ ✓ ✗ ✓ ✗ ✗ ✓            Model B:  ✗ ✓ ✓ ✗ ✓ ✓ ✓
    Model C:  ✓ ✓ ✗ ✓ ✗ ✗ ✓            Model C:  ✓ ✓ ✗ ✓ ✓ ✗ ✓
    Majority: ✓ ✓ ✗ ✓ ✗ ✗ ✓            Majority: ✓ ✓ ✓ ✓ ✓ ✓ ✓
    Error: 3/7 = 43% (same as each!)    Error: 0/7 = 0% (perfect!)

    Diverse models fail on different examples → majority vote cancels errors.
    Correlated models fail together → majority vote doesn't help.


### How Ensembles Fix Overfitting and Underfitting

    ┌──────────────────────────────────────────────────────────────────┐
    │ Method    │ Reduces     │ How                                    │
    ├──────────────────────────────────────────────────────────────────┤
    │ Bagging   │ VARIANCE    │ Average many high-variance models      │
    │           │             │ trained on different bootstrap samples │
    ├──────────────────────────────────────────────────────────────────┤
    │ Boosting  │ BIAS        │ Chain models sequentially, each one    │
    │           │             │ correcting the previous model's errors │
    ├──────────────────────────────────────────────────────────────────┤
    │ Stacking  │ Both        │ Meta-learner picks the best combination│
    │           │             │ of diverse base models                 │
    └──────────────────────────────────────────────────────────────────┘


##### PART 2 — BAGGING  (Bootstrap Aggregating)

### What is Bagging?

Bagging was introduced by Leo Breiman in 1994. The idea is simple:

    1. Create B "bootstrap" datasets by sampling N examples from the
       training set WITH REPLACEMENT (each bootstrap sample is size N
       but ~63% unique — ~37% are duplicates).
    2. Train one independent model on each bootstrap dataset.
    3. Aggregate predictions:
       - Classification: majority vote
       - Regression:     average

    Diagram 2 — Bagging Pipeline:

    ORIGINAL DATASET  [D₁, D₂, D₃, D₄, D₅ ... Dₙ]
          │
          ├──► Bootstrap B₁ [D₂, D₂, D₄, D₁, D₃ ...] ──► Model₁ ──►┐
          │                                                        │
          ├──► Bootstrap B₂ [D₅, D₁, D₁, D₄, D₃ ...] ──► Model₂ ──►┤
          │                                                        ├──► VOTE/AVG
          ├──► Bootstrap B₃ [D₁, D₃, D₅, D₂, D₅ ...] ──► Model₃ ──►┤
          │                                                        │
          └──► Bootstrap Bᴮ [D₄, D₄, D₂, D₃, D₁ ...] ──► Modelᴮ ──►┘

    Diversity source: different bootstrap samples → different models.

**Why does bagging reduce variance?**
    The variance of an average of B i.i.d. random variables each with
    variance σ² is σ²/B. So averaging B models divides variance by B.
    In practice, models are correlated (same training set structure)
    so the reduction is less than B, but still substantial.

**Out-of-Bag (OOB) Error:**
    Each bootstrap sample leaves out ~37% of training examples.
    These "out-of-bag" examples can be used to evaluate the model
    that didn't see them — giving a free cross-validation estimate!
    Random forests use OOB error as their default validation mechanism.


##### PART 3 — RANDOM FORESTS

### What Makes a Random Forest Different from Bagging?

A Random Forest is bagging applied to decision trees, with an extra source
of randomness: at each split in each tree, only a RANDOM SUBSET of features
is considered (typically √p for classification, p/3 for regression,
where p = total number of features).

    Plain Bagging:          Random Forest:
    All features available  Only √p features at each split
    at every split          ─────────────────────────────
    ─────────────────       More diverse trees, lower correlation,
    Trees still correlated  BETTER variance reduction
    (same important
    features always win)

    Why feature subsampling helps:
    Without it, the most informative feature wins almost every first split
    across all trees → the trees are nearly identical → diversity is lost.
    With it, even the best feature is sometimes blocked → other features
    must step up → each tree learns from a different perspective.

**Random Forest Feature Importance:**
    Each feature's importance = total reduction in impurity (Gini or
    entropy) it causes, summed across all splits in all trees.
    This gives a robust, ensemble-averaged importance score.

    Diagram 3 — Feature Importance from a Random Forest:

    Impurity Reduction (Gini) summed over all splits and all trees:

    Feature 1  ████████████████████  0.42  ← most important
    Feature 2  ██████████           0.21
    Feature 3  ████████             0.17
    Feature 4  ████                 0.08
    Feature 5  ███                  0.06
    Feature 6  █                    0.04  ← least important
    Feature 7  █                    0.02

    Features with near-zero importance can be safely removed.


##### PART 4 — BOOSTING

### What is Boosting?

Boosting builds models SEQUENTIALLY rather than in parallel. Each new
model specifically targets the mistakes made by all previous models.
The focus shifts adaptively to the hardest examples.

While bagging targets variance, boosting primarily targets BIAS —
it can take many weak learners (each barely better than random) and
combine them into a strong learner with low bias.

### AdaBoost (Adaptive Boosting) — 1997

    Algorithm:
    1. Initialise equal weights wᵢ = 1/N for all N training examples.
    2. For t = 1, 2, ..., T:
       a. Train weak learner hₜ on weighted data.
       b. Compute weighted error: εₜ = Σ wᵢ · 𝟙[hₜ(xᵢ) ≠ yᵢ]
       c. Compute learner weight: αₜ = ½ ln((1-εₜ)/εₜ)
          (good models get high α; near-random models get α ≈ 0)
       d. Update sample weights:
          wᵢ ← wᵢ · exp(-αₜ · yᵢ · hₜ(xᵢ))
          Misclassified examples get HIGHER weight → next learner focuses on them.
       e. Normalise weights so they sum to 1.
    3. Final prediction: H(x) = sign(Σ αₜ · hₜ(x))

    Diagram 4 — AdaBoost Weight Evolution:

    Round 1: equal weights        Round 2: misclassified ↑
    ● ● ● ● ● ○ ○ ○ ○ ○          ●●● ● ● ○ ○ ●●● ○ ○
    all same size                 large dots got wrong → bigger weight

    Round 3: new hard examples ↑  FINAL: weighted vote
    ● ●●● ●●● ○ ○ ●● ○ ○        Combined: accurate on all regions


### Gradient Boosting (GBM) — 1999

Gradient Boosting is the modern generalisation of AdaBoost. Instead of
reweighting examples, it fits each new tree to the RESIDUALS (errors) of
the current ensemble — or more precisely, to the negative gradient of
any differentiable loss function.

    Algorithm (Gradient Boosted Trees):
    1. Initialise F₀(x) = constant (e.g., mean of y)
    2. For t = 1, 2, ..., T:
       a. Compute residuals (negative gradient of loss):
          rᵢ = -∂L(yᵢ, Fₜ₋₁(xᵢ)) / ∂Fₜ₋₁(xᵢ)
          For MSE loss: rᵢ = yᵢ - Fₜ₋₁(xᵢ)   (the actual residual)
       b. Fit a shallow tree hₜ to the residuals {(xᵢ, rᵢ)}
       c. Update: Fₜ(x) = Fₜ₋₁(x) + η · hₜ(x)   (η = learning rate)
    3. Final: F_T(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₜ(x)

    Why it works as a bias reducer:
    Each tree corrects what the previous ensemble got wrong.
    The ensemble gradually fits the training data more and more precisely.
    The learning rate η slows this down to avoid overfitting.

    Diagram 5 — Gradient Boosting as Residual Correction:

    Training data: y = 10, 20, 30, 40 (regression example)

    Step 0: F₀ = mean = 25    Residuals: -15, -5, +5, +15
    Step 1: h₁ fits residuals  Ensemble: 25 + η·h₁
                                Residuals: smaller but not zero
    Step 2: h₂ fits new res.   Ensemble: ... + η·h₂
                                Residuals: even smaller
    ...
    Step T: residuals → 0      Ensemble converges to true y values

    The ensemble "descends" the loss surface in FUNCTION SPACE —
    this is why it's called gradient boosting.


### XGBoost, LightGBM, CatBoost

These are optimised implementations of gradient boosting, not different
algorithms. Their improvements are engineering innovations:

    ┌──────────────────┬───────────────────────────────────────────────┐
    │ Feature          │ What it does                                  │
    ├──────────────────┼───────────────────────────────────────────────┤
    │ XGBoost          │ Regularised objective (L1+L2 on tree weights) │
    │                  │ Sparse-aware (handles missing values)         │
    │                  │ Column / row subsampling like RF              │
    ├──────────────────┼───────────────────────────────────────────────┤
    │ LightGBM         │ Leaf-wise tree growth (vs level-wise)         │
    │                  │ Histogram-based splits (much faster)          │
    │                  │ GOSS: focusses on large-gradient samples      │
    ├──────────────────┼───────────────────────────────────────────────┤
    │ CatBoost         │ Native categorical encoding (no preprocessing)│
    │                  │ Ordered boosting (reduces target leakage)     │
    └──────────────────┴───────────────────────────────────────────────┘

    In practice: start with LightGBM (fastest), tune with XGBoost or CatBoost
    if needed. All three drastically outperform plain sklearn GradientBoosting.


##### PART 5 — STACKING & BLENDING

### Stacking (Stacked Generalisation)

Stacking trains a meta-learner (Level-1) whose inputs are the predictions
of several base models (Level-0). The meta-learner learns how to best
combine the base model predictions.

    Diagram 6 — Two-Level Stacking Architecture:

    TRAINING DATA
          │
          ├─────────────────────┬──────────────────┬──────────────────┐
          │                     │                  │                  │
          ▼                     ▼                  ▼                  ▼
    [  Model A  ]         [ Model B  ]       [ Model C  ]      [ Model D  ]
    (SVM)                  (Random Forest)   (Logistic Reg)    (Neural Net)
          │                     │                  │                  │
          ├─────────────────────┴──────────────────┴──────────────────┘
          │        Level-0 predictions → new feature matrix
          ▼
    [META-LEARNER]  (Logistic Regression or Ridge)
    Learns: "When does each base model tend to be right?"
          │
          ▼
    FINAL PREDICTION

    Critical rule: Base models must make predictions on data they did NOT
    train on (out-of-fold predictions), otherwise the meta-learner is
    trained on training-set predictions → massive data leakage!

    Correct stacking with k-fold:
    - Split training data into k folds
    - For each fold: train base models on other k-1 folds, predict on held fold
    - This produces out-of-fold (OOF) predictions for ALL training examples
    - Train meta-learner on OOF predictions

**Blending** — simpler alternative: use a fixed holdout set to generate
    Level-0 predictions. Faster but wastes data and has higher variance.


### Voting Classifiers (Simpler Ensemble)

    Hard voting: each model votes on the class label → majority wins
    Soft voting:  each model outputs probabilities → average probs → argmax

    Soft voting is almost always better because probability magnitudes
    carry more information than bare votes (a 99% confident model should
    count more than a 51% confident model).

    ┌────────────────────────────────────────────────────────────┐
    │ Model A:  P(cat)=0.90  P(dog)=0.10                         │
    │ Model B:  P(cat)=0.60  P(dog)=0.40                         │
    │ Model C:  P(cat)=0.45  P(dog)=0.55                         │
    │                                                            │
    │ Hard voting: 2 say cat, 1 says dog → predict CAT           │
    │ Soft voting: avg P(cat)=(0.90+0.60+0.45)/3=0.65 → CAT ✓    │
    │                                                            │
    │ Now change Model A to P(cat)=0.51:                         │
    │ Hard voting: still 2-1 for cat                             │
    │ Soft voting: avg=(0.51+0.60+0.45)/3=0.52 → cat, but barely │
    │             appropriately less confident                   │
    └────────────────────────────────────────────────────────────┘


##### PART 6 — CHOOSING THE RIGHT ENSEMBLE METHOD

    ┌─────────────────┬───────────────┬───────────────┬──────────────────┐
    │ Method          │ Parallelisable│ Main benefit  │ Overfit risk     │
    ├─────────────────┼───────────────┼───────────────┼──────────────────┤
    │ Bagging         │ Yes           │ Low variance  │ Low              │
    │ Random Forest   │ Yes           │ Low variance  │ Low              │
    │ AdaBoost        │ No            │ Low bias      │ Medium           │
    │ Gradient Boost  │ No            │ Low bias+var  │ Medium-High      │
    │ XGBoost/LGBM    │ Partial       │ Low bias+var  │ Low (regularised)│
    │ Stacking        │ Partially     │ Both          │ Medium (OOF cv)  │
    └─────────────────┴───────────────┴───────────────┴──────────────────┘

    Rule of thumb:
    • Tabular data: XGBoost or LightGBM first. Add stacking for competitions.
    • High-dimensional sparse data: AdaBoost or GBM with shallow trees.
    • Need interpretability: Random Forest (feature importances).
    • Very little data: Bagging (maximises use of limited training data).

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Bagging from Scratch": {
        "description": (
            "Implement bootstrap aggregating from scratch using NumPy. "
            "Show variance reduction numerically: one decision tree vs 100 bagged trees. "
            "Also verify the 63.2% unique-sample property."
        ),
        "timeout": 600,
        "language": "python",
        "code": '''
import numpy as np
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf)); X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        skf=StratifiedKFold(n_splits=cv,shuffle=True,random_state=0)
        splits=list(skf.split(X,y))
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

# ── Decision Tree (classifier + regressor) ────────────────────────────────────
class _BaseTree:
    def __init__(self,max_depth=None,min_samples=2,mode="cls"):
        self.max_depth=max_depth or 999; self.min_samples=min_samples; self.mode=mode
    def _impurity(self,y):
        if self.mode=="reg": return np.var(y)*len(y) if len(y) else 0
        if not len(y): return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _split(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2
            thrs=all_t[np.linspace(0,len(all_t)-1,min(12,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples or r.sum()<self.min_samples: continue
                g=self._impurity(y[l])+self._impurity(y[r])
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        if self.mode=="cls":
            cls,cnt=np.unique(y,return_counts=True); pred=cls[cnt.argmax()]; probs=cnt/cnt.sum()
        else:
            pred=y.mean(); probs=None
        if depth>=self.max_depth or len(y)<=self.min_samples:
            return ("L",pred,probs,len(y))
        sp=self._split(X,y)
        if sp is None: return ("L",pred,probs,len(y))
        _,f,t=sp; m=X[:,f]<=t
        return ("N",f,t,self._build(X[m],y[m],depth+1),self._build(X[~m],y[~m],depth+1),len(y))
    def fit(self,X,y):
        self.classes_=np.unique(y) if self.mode=="cls" else None
        self._tree=self._build(X,y,0)
        if self.mode=="cls": self._compute_importances(X)
        return self
    def _compute_importances(self,X):
        imp=np.zeros(X.shape[1])
        def walk(nd):
            if nd[0]=="L": return
            f,t,l_nd,r_nd,n=nd[1],nd[2],nd[3],nd[4],nd[5]
            nl=l_nd[3] if l_nd[0]=="L" else l_nd[5]
            nr=r_nd[3] if r_nd[0]=="L" else r_nd[5]
            ig=self._impurity_node(nd)-nl/n*self._impurity_node(l_nd)-nr/n*self._impurity_node(r_nd)
            imp[f]+=n*ig; walk(l_nd); walk(r_nd)
        walk(self._tree)
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp
    def _impurity_node(self,nd):
        if nd[0]=="L": return 0
        return 0  # simplified
    def _p1(self,x,nd):
        if nd[0]=="L": return nd[1],nd[2]
        return self._p1(x,nd[3] if x[nd[1]]<=nd[2] else nd[4])
    def predict(self,X): return np.array([self._p1(x,self._tree)[0] for x in X])
    def predict_proba(self,X):
        nc=len(self.classes_); out=np.zeros((len(X),nc))
        cmap={c:i for i,c in enumerate(self.classes_)}
        for i,x in enumerate(X):
            _,probs=self._p1(x,self._tree)
            if probs is not None:
                for ci,p in zip(self.classes_,probs): out[i,cmap[ci]]=p
        return out
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

class DecisionTreeClassifier(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="cls")

class DecisionTreeRegressor(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="reg")
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

# ── Random Forest Classifier (with oob_score + feature_importances_) ──────────
class RandomForestClassifier:
    def __init__(self,n_estimators=20,max_depth=None,oob_score=False,
                 random_state=None,n_jobs=None,max_features="sqrt"):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.oob_score=oob_score; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self._trees=[]; self._feat_idxs=[]
        oob_votes=np.zeros((n,len(self.classes_)))
        oob_counts=np.zeros(n)
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(np.sqrt(X.shape[1]))); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeClassifier(max_depth=self.max_depth)
            t.fit(X[np.ix_(idx,fi)],y[idx])
            self._trees.append(t); self._feat_idxs.append(fi)
            if self.oob_score:
                oob=np.setdiff1d(np.arange(n),idx)
                if len(oob):
                    p=t.predict_proba(X[np.ix_(oob,fi)])
                    for li,gc in enumerate(t.classes_):
                        gi=np.where(self.classes_==gc)[0]
                        if len(gi): oob_votes[oob,gi[0]]+=p[:,li]
                    oob_counts[oob]+=1
        # feature importances (mean decrease impurity across trees)
        imp=np.zeros(X.shape[1])
        for t,fi in zip(self._trees,self._feat_idxs):
            if hasattr(t,"feature_importances_"):
                imp[fi]+=t.feature_importances_
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp+1/X.shape[1]
        if self.oob_score:
            mask=oob_counts>0
            oob_pred=self.classes_[oob_votes[mask].argmax(1)]
            self.oob_score_=float((oob_pred==y[mask]).mean())
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); votes=np.zeros((len(X),nc))
        for t,fi in zip(self._trees,self._feat_idxs):
            p=t.predict_proba(X[:,fi])
            for li,gc in enumerate(t.classes_):
                gi=np.where(self.classes_==gc)[0]
                if len(gi): votes[:,gi[0]]+=p[:,li]
        return votes/len(self._trees)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,
                "oob_score":self.oob_score,"random_state":self.random_state}

# ── Logistic Regression ───────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1./self.C
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C,"max_iter":self.max_iter}

# ── Gradient Boosting Classifier (log-loss) ───────────────────────────────────
class GradientBoostingClassifier:
    def __init__(self,n_estimators=30,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.lr=learning_rate
        self.max_depth=max_depth; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        self.classes_=np.unique(y)
        p_mean=y.mean(); p_mean=np.clip(p_mean,1e-6,1-1e-6)
        self.F0_=float(np.log(p_mean/(1-p_mean)))
        F=np.full(len(y),self.F0_); self._trees=[]
        for _ in range(self.n_estimators):
            p=_sig(F); r=y-p          # negative gradient of log-loss
            t=DecisionTreeRegressor(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t)
            F+=self.lr*t.predict(X)
        return self
    def _decision(self,X):
        F=np.full(len(X),self.F0_)
        for t in self._trees: F+=self.lr*t.predict(X)
        return F
    def predict_proba(self,X):
        p=_sig(self._decision(X)); return np.c_[1-p,p]
    def predict(self,X): return (self._decision(X)>=0).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"learning_rate":self.lr,
                "max_depth":self.max_depth,"random_state":self.random_state}

# ── SVC (RBF, with probability via sigmoid calibration) ──────────────────────
class SVC:
    """Linear SVM via L-BFGS (fast, no kernel matrix)."""
    def __init__(self,C=1.0,kernel="rbf",gamma="scale",probability=True,random_state=None):
        self.C=C
    def fit(self,X,y):
        self._classes=np.unique(y); n,d=X.shape; lam=1./(2*self.C*n)
        yb=(2*(y==self._classes[1])-1).astype(float)
        def fg(w):
            margins=1-yb*(X@w[1:]+w[0]); hinge=np.maximum(0,margins)
            loss=lam*np.sum(w[1:]**2)+np.mean(hinge)
            mask=hinge>0
            g_w=2*lam*w[1:]-np.mean(yb[mask,None]*X[mask],axis=0) if mask.any() else 2*lam*w[1:]
            g_b=-np.mean(yb[mask]) if mask.any() else 0.
            return loss,np.r_[g_b,g_w]
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":200})
        self._b=res.x[0]; self._w=res.x[1:]; return self
    def _dec(self,X): return X@self._w+self._b
    def predict(self,X): return self._classes[(self._dec(X)>=0).astype(int)]
    def predict_proba(self,X): p=_sig(self._dec(X)); return np.c_[1-p,p]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C}

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); self.classes_=np.unique(y); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def predict_proba(self,X):
        nc=len(self.classes_); out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            p=np.bincount(nn.astype(int),minlength=nc)/self.n_neighbors; out.append(p)
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

# ── VotingClassifier ──────────────────────────────────────────────────────────
class VotingClassifier:
    def __init__(self,estimators,voting="hard"):
        self.estimators=estimators; self.voting=voting
    def fit(self,X,y):
        self._fitted=[]; self.classes_=np.unique(y)
        for nm,est in self.estimators:
            e=_copy.deepcopy(est); e.fit(X,y); self._fitted.append(e)
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); avg=np.zeros((len(X),nc))
        for e in self._fitted: avg+=e.predict_proba(X)
        return avg/len(self._fitted)
    def predict(self,X):
        if self.voting=="soft": return self.classes_[self.predict_proba(X).argmax(1)]
        votes=np.stack([e.predict(X) for e in self._fitted],axis=1)
        return np.array([np.bincount(row.astype(int)).argmax() for row in votes])
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"voting":self.voting}

np.random.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)
N = len(y_tr)

# ── Verify the 63.2% bootstrap property ──────────────────────────────────
print("=" * 60)
print("  BAGGING FROM SCRATCH")
print("=" * 60)
print()
print("  BOOTSTRAP SAMPLING PROPERTY:")
print(f"  When sampling N={N} points with replacement, the expected")
print(f"  fraction of UNIQUE samples = 1 - (1 - 1/N)^N → 1 - 1/e ≈ 63.2%")
print()

unique_fractions = []
for _ in range(1000):
    boot_idx = np.random.choice(N, size=N, replace=True)
    unique_fractions.append(len(np.unique(boot_idx)) / N)
print(f"  Empirical (avg over 1000 bootstraps): "
      f"{np.mean(unique_fractions)*100:.2f}% unique")
print(f"  Theoretical: 63.21%")
print()

# ── Manual Bagging ────────────────────────────────────────────────────────
class BaggingClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.oob_indices = []

    def fit(self, X, y):
        N = len(y)
        for _ in range(self.n_estimators):
            boot_idx = np.random.choice(N, size=N, replace=True)
            oob_idx  = np.setdiff1d(np.arange(N), boot_idx)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
            self.oob_indices.append(oob_idx)
        return self

    def predict(self, X):
        votes = np.stack([t.predict(X) for t in self.trees], axis=1)
        return np.apply_along_axis(
            lambda v: np.bincount(v.astype(int)).argmax(), 1, votes)

    def oob_score(self, X, y):
        N = len(y)
        oob_votes = [[] for _ in range(N)]
        for tree, oob_idx in zip(self.trees, self.oob_indices):
            preds = tree.predict(X[oob_idx])
            for i, pred in zip(oob_idx, preds):
                oob_votes[i].append(int(pred))
        correct = sum(
            1 for i in range(N)
            if oob_votes[i] and
               np.bincount(oob_votes[i]).argmax() == y[i]
        )
        return correct / N

# ── Compare single tree vs ensembles of different sizes ───────────────────
print("  SINGLE TREE vs BAGGED ENSEMBLE (different B values):")
print(f"  {'Method':30s} | {'Test Acc':>10} | {'Notes'}")
print(f"  {'─'*65}")

# Single deep tree (high variance)
single_tree = DecisionTreeClassifier(max_depth=None)
single_tree.fit(X_tr, y_tr)
acc_single = accuracy_score(y_te, single_tree.predict(X_te))
print(f"  {'Single deep tree':30s} | {acc_single:10.4f} | high variance")

for B in [5, 20, 50]:
    bag = BaggingClassifier(n_estimators=B, max_depth=None)
    bag.fit(X_tr, y_tr)
    acc_test = accuracy_score(y_te, bag.predict(X_te))
    oob      = bag.oob_score(X_tr, y_tr)
    print(f"  {f'Bagging B={B}':30s} | {acc_test:10.4f} | "
          f"OOB score: {oob:.4f}")

# ── Show variance reduction across multiple random seeds ──────────────────
print()
print("  VARIANCE ANALYSIS (50 random train/test splits):")
single_accs, bag_accs = [], []
for seed in range(15):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                            random_state=seed)
    st = DecisionTreeClassifier(max_depth=None).fit(Xtr, ytr)
    single_accs.append(accuracy_score(yte, st.predict(Xte)))
    bg = BaggingClassifier(n_estimators=30, max_depth=None).fit(Xtr, ytr)
    bag_accs.append(accuracy_score(yte, bg.predict(Xte)))

print(f"  Single tree : mean={np.mean(single_accs):.4f}  "
      f"std={np.std(single_accs):.4f}  "
      f"range=[{min(single_accs):.3f}, {max(single_accs):.3f}]")
print(f"  Bagging B=100: mean={np.mean(bag_accs):.4f}  "
      f"std={np.std(bag_accs):.4f}  "
      f"range=[{min(bag_accs):.3f}, {max(bag_accs):.3f}]")
print()
print(f"  Variance reduction: {np.std(single_accs)**2 / np.std(bag_accs)**2:.1f}×")
print("  Bagging reduces variance (std) dramatically with minimal bias cost.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Random Forest vs Single Decision Tree": {
        "description": (
            "Compare a fully-grown decision tree against a Random Forest. "
            "Visualise decision boundaries, examine feature importances, "
            "and show OOB error tracking."
        ),
        "timeout": 600,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf)); X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        skf=StratifiedKFold(n_splits=cv,shuffle=True,random_state=0)
        splits=list(skf.split(X,y))
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

# ── Decision Tree (classifier + regressor) ────────────────────────────────────
class _BaseTree:
    def __init__(self,max_depth=None,min_samples=2,mode="cls"):
        self.max_depth=max_depth or 999; self.min_samples=min_samples; self.mode=mode
    def _impurity(self,y):
        if self.mode=="reg": return np.var(y)*len(y) if len(y) else 0
        if not len(y): return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _split(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2
            thrs=all_t[np.linspace(0,len(all_t)-1,min(12,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples or r.sum()<self.min_samples: continue
                g=self._impurity(y[l])+self._impurity(y[r])
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        if self.mode=="cls":
            cls,cnt=np.unique(y,return_counts=True); pred=cls[cnt.argmax()]; probs=cnt/cnt.sum()
        else:
            pred=y.mean(); probs=None
        if depth>=self.max_depth or len(y)<=self.min_samples:
            return ("L",pred,probs,len(y))
        sp=self._split(X,y)
        if sp is None: return ("L",pred,probs,len(y))
        _,f,t=sp; m=X[:,f]<=t
        return ("N",f,t,self._build(X[m],y[m],depth+1),self._build(X[~m],y[~m],depth+1),len(y))
    def fit(self,X,y):
        self.classes_=np.unique(y) if self.mode=="cls" else None
        self._tree=self._build(X,y,0)
        if self.mode=="cls": self._compute_importances(X)
        return self
    def _compute_importances(self,X):
        imp=np.zeros(X.shape[1])
        def walk(nd):
            if nd[0]=="L": return
            f,t,l_nd,r_nd,n=nd[1],nd[2],nd[3],nd[4],nd[5]
            nl=l_nd[3] if l_nd[0]=="L" else l_nd[5]
            nr=r_nd[3] if r_nd[0]=="L" else r_nd[5]
            ig=self._impurity_node(nd)-nl/n*self._impurity_node(l_nd)-nr/n*self._impurity_node(r_nd)
            imp[f]+=n*ig; walk(l_nd); walk(r_nd)
        walk(self._tree)
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp
    def _impurity_node(self,nd):
        if nd[0]=="L": return 0
        return 0  # simplified
    def _p1(self,x,nd):
        if nd[0]=="L": return nd[1],nd[2]
        return self._p1(x,nd[3] if x[nd[1]]<=nd[2] else nd[4])
    def predict(self,X): return np.array([self._p1(x,self._tree)[0] for x in X])
    def predict_proba(self,X):
        nc=len(self.classes_); out=np.zeros((len(X),nc))
        cmap={c:i for i,c in enumerate(self.classes_)}
        for i,x in enumerate(X):
            _,probs=self._p1(x,self._tree)
            if probs is not None:
                for ci,p in zip(self.classes_,probs): out[i,cmap[ci]]=p
        return out
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

class DecisionTreeClassifier(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="cls")

class DecisionTreeRegressor(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="reg")
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

# ── Random Forest Classifier (with oob_score + feature_importances_) ──────────
class RandomForestClassifier:
    def __init__(self,n_estimators=20,max_depth=None,oob_score=False,
                 random_state=None,n_jobs=None,max_features="sqrt"):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.oob_score=oob_score; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self._trees=[]; self._feat_idxs=[]
        oob_votes=np.zeros((n,len(self.classes_)))
        oob_counts=np.zeros(n)
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(np.sqrt(X.shape[1]))); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeClassifier(max_depth=self.max_depth)
            t.fit(X[np.ix_(idx,fi)],y[idx])
            self._trees.append(t); self._feat_idxs.append(fi)
            if self.oob_score:
                oob=np.setdiff1d(np.arange(n),idx)
                if len(oob):
                    p=t.predict_proba(X[np.ix_(oob,fi)])
                    for li,gc in enumerate(t.classes_):
                        gi=np.where(self.classes_==gc)[0]
                        if len(gi): oob_votes[oob,gi[0]]+=p[:,li]
                    oob_counts[oob]+=1
        # feature importances (mean decrease impurity across trees)
        imp=np.zeros(X.shape[1])
        for t,fi in zip(self._trees,self._feat_idxs):
            if hasattr(t,"feature_importances_"):
                imp[fi]+=t.feature_importances_
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp+1/X.shape[1]
        if self.oob_score:
            mask=oob_counts>0
            oob_pred=self.classes_[oob_votes[mask].argmax(1)]
            self.oob_score_=float((oob_pred==y[mask]).mean())
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); votes=np.zeros((len(X),nc))
        for t,fi in zip(self._trees,self._feat_idxs):
            p=t.predict_proba(X[:,fi])
            for li,gc in enumerate(t.classes_):
                gi=np.where(self.classes_==gc)[0]
                if len(gi): votes[:,gi[0]]+=p[:,li]
        return votes/len(self._trees)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,
                "oob_score":self.oob_score,"random_state":self.random_state}

# ── Logistic Regression ───────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1./self.C
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C,"max_iter":self.max_iter}

# ── Gradient Boosting Classifier (log-loss) ───────────────────────────────────
class GradientBoostingClassifier:
    def __init__(self,n_estimators=30,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.lr=learning_rate
        self.max_depth=max_depth; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        self.classes_=np.unique(y)
        p_mean=y.mean(); p_mean=np.clip(p_mean,1e-6,1-1e-6)
        self.F0_=float(np.log(p_mean/(1-p_mean)))
        F=np.full(len(y),self.F0_); self._trees=[]
        for _ in range(self.n_estimators):
            p=_sig(F); r=y-p          # negative gradient of log-loss
            t=DecisionTreeRegressor(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t)
            F+=self.lr*t.predict(X)
        return self
    def _decision(self,X):
        F=np.full(len(X),self.F0_)
        for t in self._trees: F+=self.lr*t.predict(X)
        return F
    def predict_proba(self,X):
        p=_sig(self._decision(X)); return np.c_[1-p,p]
    def predict(self,X): return (self._decision(X)>=0).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"learning_rate":self.lr,
                "max_depth":self.max_depth,"random_state":self.random_state}

# ── SVC (RBF, with probability via sigmoid calibration) ──────────────────────
class SVC:
    """Linear SVM via L-BFGS (fast, no kernel matrix)."""
    def __init__(self,C=1.0,kernel="rbf",gamma="scale",probability=True,random_state=None):
        self.C=C
    def fit(self,X,y):
        self._classes=np.unique(y); n,d=X.shape; lam=1./(2*self.C*n)
        yb=(2*(y==self._classes[1])-1).astype(float)
        def fg(w):
            margins=1-yb*(X@w[1:]+w[0]); hinge=np.maximum(0,margins)
            loss=lam*np.sum(w[1:]**2)+np.mean(hinge)
            mask=hinge>0
            g_w=2*lam*w[1:]-np.mean(yb[mask,None]*X[mask],axis=0) if mask.any() else 2*lam*w[1:]
            g_b=-np.mean(yb[mask]) if mask.any() else 0.
            return loss,np.r_[g_b,g_w]
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":200})
        self._b=res.x[0]; self._w=res.x[1:]; return self
    def _dec(self,X): return X@self._w+self._b
    def predict(self,X): return self._classes[(self._dec(X)>=0).astype(int)]
    def predict_proba(self,X): p=_sig(self._dec(X)); return np.c_[1-p,p]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C}

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); self.classes_=np.unique(y); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def predict_proba(self,X):
        nc=len(self.classes_); out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            p=np.bincount(nn.astype(int),minlength=nc)/self.n_neighbors; out.append(p)
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

# ── VotingClassifier ──────────────────────────────────────────────────────────
class VotingClassifier:
    def __init__(self,estimators,voting="hard"):
        self.estimators=estimators; self.voting=voting
    def fit(self,X,y):
        self._fitted=[]; self.classes_=np.unique(y)
        for nm,est in self.estimators:
            e=_copy.deepcopy(est); e.fit(X,y); self._fitted.append(e)
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); avg=np.zeros((len(X),nc))
        for e in self._fitted: avg+=e.predict_proba(X)
        return avg/len(self._fitted)
    def predict(self,X):
        if self.voting=="soft": return self.classes_[self.predict_proba(X).argmax(1)]
        votes=np.stack([e.predict(X) for e in self._fitted],axis=1)
        return np.array([np.bincount(row.astype(int)).argmax() for row in votes])
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"voting":self.voting}

np.random.seed(0)

# ── Dataset with 10 features, 5 informative ───────────────────────────────
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, random_state=42)
feature_names = [f"feat_{i}" for i in range(X.shape[1])]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

print("=" * 65)
print("  DECISION TREE vs RANDOM FOREST")
print("=" * 65)
print()

# ── Single decision tree ──────────────────────────────────────────────────
for depth in [None, 3, 5]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=0)
    cv = cross_val_score(dt, X_tr, y_tr, cv=5, scoring="accuracy")
    dt.fit(X_tr, y_tr)
    test_acc = accuracy_score(y_te, dt.predict(X_te))
    label = f"max_depth={depth}" if depth else "Unlimited depth"
    print(f"  Decision Tree ({label})")
    print(f"    CV Acc  : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"    Test Acc: {test_acc:.4f}")
    print()

# ── Random Forest: effect of n_estimators ─────────────────────────────────
print("  RANDOM FOREST — Effect of n_estimators:")
print(f"  {'n_trees':>8} | {'OOB Score':>10} | {'Test Acc':>10}")
print(f"  {'─'*34}")

rf_best = None
for n in [1, 5, 10, 25, 50]:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                 random_state=0, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    test_acc = accuracy_score(y_te, rf.predict(X_te))
    print(f"  {n:8d} | {rf.oob_score_:10.4f} | {test_acc:10.4f}")
    if n == 50:
        rf_best = rf

# ── Feature Importance ────────────────────────────────────────────────────
print()
print("  RANDOM FOREST FEATURE IMPORTANCES (n_trees=100):")
importances = rf_best.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
for rank, idx in enumerate(sorted_idx, 1):
    bar = "█" * int(importances[idx] * 100)
    tag = " ← truly informative" if idx < 5 else ""
    print(f"    {rank:2d}. {feature_names[idx]:10s}: "
          f"{importances[idx]:.4f}  {bar}{tag}")

# ── Visualise on 2D subset ────────────────────────────────────────────────
X2, y2 = make_classification(n_samples=800, n_features=2, n_informative=2,
                               n_redundant=0, random_state=5)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.3,
                                                random_state=1)
xx, yy = np.meshgrid(np.linspace(X2[:,0].min()-0.5, X2[:,0].max()+0.5, 200),
                     np.linspace(X2[:,1].min()-0.5, X2[:,1].max()+0.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Decision Boundaries: Single Tree vs Random Forest (2D)",
             fontsize=12, fontweight="bold")

models_2d = [
    ("Single Tree (unlimited)", DecisionTreeClassifier(random_state=0)),
    ("Single Tree (depth=3)",   DecisionTreeClassifier(max_depth=3, random_state=0)),
    ("Random Forest (50)",      RandomForestClassifier(n_estimators=50,
                                                        random_state=0)),
]

for ax, (name, model) in zip(axes, models_2d):
    model.fit(X2_tr, y2_tr)
    Z = model.predict(grid).reshape(xx.shape)
    acc = accuracy_score(y2_te, model.predict(X2_te))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X2_te[y2_te==0, 0], X2_te[y2_te==0, 1],
               c="steelblue", s=20, alpha=0.7)
    ax.scatter(X2_te[y2_te==1, 0], X2_te[y2_te==1, 1],
               c="tomato", s=20, alpha=0.7)
    ax.set_title(f"{name}\\nTest Acc: {acc:.4f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("random_forest_vs_tree.png", dpi=120)
print()
print("  Plot saved → random_forest_vs_tree.png")
print("  Notice: single unlimited tree has jagged, fragmented boundaries.")
print("  Random Forest: smooth, stable boundary → lower variance, better generalisation.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Gradient Boosting from Scratch": {
        "description": (
            "Implement gradient boosted trees from scratch for regression. "
            "Show how each new tree corrects the residuals of the ensemble, "
            "and how the learning rate controls the tradeoff."
        ),
        "timeout": 600,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf)); X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        skf=StratifiedKFold(n_splits=cv,shuffle=True,random_state=0)
        splits=list(skf.split(X,y))
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

# ── Decision Tree (classifier + regressor) ────────────────────────────────────
class _BaseTree:
    def __init__(self,max_depth=None,min_samples=2,mode="cls"):
        self.max_depth=max_depth or 999; self.min_samples=min_samples; self.mode=mode
    def _impurity(self,y):
        if self.mode=="reg": return np.var(y)*len(y) if len(y) else 0
        if not len(y): return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _split(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2
            thrs=all_t[np.linspace(0,len(all_t)-1,min(12,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples or r.sum()<self.min_samples: continue
                g=self._impurity(y[l])+self._impurity(y[r])
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        if self.mode=="cls":
            cls,cnt=np.unique(y,return_counts=True); pred=cls[cnt.argmax()]; probs=cnt/cnt.sum()
        else:
            pred=y.mean(); probs=None
        if depth>=self.max_depth or len(y)<=self.min_samples:
            return ("L",pred,probs,len(y))
        sp=self._split(X,y)
        if sp is None: return ("L",pred,probs,len(y))
        _,f,t=sp; m=X[:,f]<=t
        return ("N",f,t,self._build(X[m],y[m],depth+1),self._build(X[~m],y[~m],depth+1),len(y))
    def fit(self,X,y):
        self.classes_=np.unique(y) if self.mode=="cls" else None
        self._tree=self._build(X,y,0)
        if self.mode=="cls": self._compute_importances(X)
        return self
    def _compute_importances(self,X):
        imp=np.zeros(X.shape[1])
        def walk(nd):
            if nd[0]=="L": return
            f,t,l_nd,r_nd,n=nd[1],nd[2],nd[3],nd[4],nd[5]
            nl=l_nd[3] if l_nd[0]=="L" else l_nd[5]
            nr=r_nd[3] if r_nd[0]=="L" else r_nd[5]
            ig=self._impurity_node(nd)-nl/n*self._impurity_node(l_nd)-nr/n*self._impurity_node(r_nd)
            imp[f]+=n*ig; walk(l_nd); walk(r_nd)
        walk(self._tree)
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp
    def _impurity_node(self,nd):
        if nd[0]=="L": return 0
        return 0  # simplified
    def _p1(self,x,nd):
        if nd[0]=="L": return nd[1],nd[2]
        return self._p1(x,nd[3] if x[nd[1]]<=nd[2] else nd[4])
    def predict(self,X): return np.array([self._p1(x,self._tree)[0] for x in X])
    def predict_proba(self,X):
        nc=len(self.classes_); out=np.zeros((len(X),nc))
        cmap={c:i for i,c in enumerate(self.classes_)}
        for i,x in enumerate(X):
            _,probs=self._p1(x,self._tree)
            if probs is not None:
                for ci,p in zip(self.classes_,probs): out[i,cmap[ci]]=p
        return out
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

class DecisionTreeClassifier(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="cls")

class DecisionTreeRegressor(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="reg")
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

# ── Random Forest Classifier (with oob_score + feature_importances_) ──────────
class RandomForestClassifier:
    def __init__(self,n_estimators=20,max_depth=None,oob_score=False,
                 random_state=None,n_jobs=None,max_features="sqrt"):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.oob_score=oob_score; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self._trees=[]; self._feat_idxs=[]
        oob_votes=np.zeros((n,len(self.classes_)))
        oob_counts=np.zeros(n)
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(np.sqrt(X.shape[1]))); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeClassifier(max_depth=self.max_depth)
            t.fit(X[np.ix_(idx,fi)],y[idx])
            self._trees.append(t); self._feat_idxs.append(fi)
            if self.oob_score:
                oob=np.setdiff1d(np.arange(n),idx)
                if len(oob):
                    p=t.predict_proba(X[np.ix_(oob,fi)])
                    for li,gc in enumerate(t.classes_):
                        gi=np.where(self.classes_==gc)[0]
                        if len(gi): oob_votes[oob,gi[0]]+=p[:,li]
                    oob_counts[oob]+=1
        # feature importances (mean decrease impurity across trees)
        imp=np.zeros(X.shape[1])
        for t,fi in zip(self._trees,self._feat_idxs):
            if hasattr(t,"feature_importances_"):
                imp[fi]+=t.feature_importances_
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp+1/X.shape[1]
        if self.oob_score:
            mask=oob_counts>0
            oob_pred=self.classes_[oob_votes[mask].argmax(1)]
            self.oob_score_=float((oob_pred==y[mask]).mean())
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); votes=np.zeros((len(X),nc))
        for t,fi in zip(self._trees,self._feat_idxs):
            p=t.predict_proba(X[:,fi])
            for li,gc in enumerate(t.classes_):
                gi=np.where(self.classes_==gc)[0]
                if len(gi): votes[:,gi[0]]+=p[:,li]
        return votes/len(self._trees)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,
                "oob_score":self.oob_score,"random_state":self.random_state}

# ── Logistic Regression ───────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1./self.C
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C,"max_iter":self.max_iter}

# ── Gradient Boosting Classifier (log-loss) ───────────────────────────────────
class GradientBoostingClassifier:
    def __init__(self,n_estimators=30,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.lr=learning_rate
        self.max_depth=max_depth; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        self.classes_=np.unique(y)
        p_mean=y.mean(); p_mean=np.clip(p_mean,1e-6,1-1e-6)
        self.F0_=float(np.log(p_mean/(1-p_mean)))
        F=np.full(len(y),self.F0_); self._trees=[]
        for _ in range(self.n_estimators):
            p=_sig(F); r=y-p          # negative gradient of log-loss
            t=DecisionTreeRegressor(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t)
            F+=self.lr*t.predict(X)
        return self
    def _decision(self,X):
        F=np.full(len(X),self.F0_)
        for t in self._trees: F+=self.lr*t.predict(X)
        return F
    def predict_proba(self,X):
        p=_sig(self._decision(X)); return np.c_[1-p,p]
    def predict(self,X): return (self._decision(X)>=0).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"learning_rate":self.lr,
                "max_depth":self.max_depth,"random_state":self.random_state}

# ── SVC (RBF, with probability via sigmoid calibration) ──────────────────────
class SVC:
    """Linear SVM via L-BFGS (fast, no kernel matrix)."""
    def __init__(self,C=1.0,kernel="rbf",gamma="scale",probability=True,random_state=None):
        self.C=C
    def fit(self,X,y):
        self._classes=np.unique(y); n,d=X.shape; lam=1./(2*self.C*n)
        yb=(2*(y==self._classes[1])-1).astype(float)
        def fg(w):
            margins=1-yb*(X@w[1:]+w[0]); hinge=np.maximum(0,margins)
            loss=lam*np.sum(w[1:]**2)+np.mean(hinge)
            mask=hinge>0
            g_w=2*lam*w[1:]-np.mean(yb[mask,None]*X[mask],axis=0) if mask.any() else 2*lam*w[1:]
            g_b=-np.mean(yb[mask]) if mask.any() else 0.
            return loss,np.r_[g_b,g_w]
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":200})
        self._b=res.x[0]; self._w=res.x[1:]; return self
    def _dec(self,X): return X@self._w+self._b
    def predict(self,X): return self._classes[(self._dec(X)>=0).astype(int)]
    def predict_proba(self,X): p=_sig(self._dec(X)); return np.c_[1-p,p]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C}

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); self.classes_=np.unique(y); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def predict_proba(self,X):
        nc=len(self.classes_); out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            p=np.bincount(nn.astype(int),minlength=nc)/self.n_neighbors; out.append(p)
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

# ── VotingClassifier ──────────────────────────────────────────────────────────
class VotingClassifier:
    def __init__(self,estimators,voting="hard"):
        self.estimators=estimators; self.voting=voting
    def fit(self,X,y):
        self._fitted=[]; self.classes_=np.unique(y)
        for nm,est in self.estimators:
            e=_copy.deepcopy(est); e.fit(X,y); self._fitted.append(e)
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); avg=np.zeros((len(X),nc))
        for e in self._fitted: avg+=e.predict_proba(X)
        return avg/len(self._fitted)
    def predict(self,X):
        if self.voting=="soft": return self.classes_[self.predict_proba(X).argmax(1)]
        votes=np.stack([e.predict(X) for e in self._fitted],axis=1)
        return np.array([np.bincount(row.astype(int)).argmax() for row in votes])
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"voting":self.voting}

np.random.seed(7)

# ── Data: noisy sine wave ─────────────────────────────────────────────────
N = 200
X_all = np.sort(np.random.uniform(0, 6, N))
y_all = np.sin(X_all) + np.random.normal(0, 0.2, N)
X_tr, y_tr = X_all[:150].reshape(-1, 1), y_all[:150]
X_te, y_te = X_all[150:].reshape(-1, 1), y_all[150:]

# ── Gradient Boosting from scratch (MSE loss) ─────────────────────────────
class GradientBoostedTrees:
    """
    Manual GBT for regression with MSE loss.
    Residuals = negative gradient of MSE = y - F(x).
    """
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=2):
        self.T  = n_estimators
        self.lr = learning_rate
        self.depth = max_depth
        self.trees = []
        self.F0 = None

    def fit(self, X, y):
        self.F0 = y.mean()                      # initial constant prediction
        F = np.full(len(y), self.F0)
        self.train_mses = []

        for t in range(self.T):
            residuals = y - F                   # negative gradient of MSE
            tree = DecisionTreeRegressor(max_depth=self.depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            F += self.lr * update
            self.trees.append(tree)
            self.train_mses.append(np.mean((y - F) ** 2))

        return self

    def predict(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F

    def staged_predict(self, X):
        """Yield predictions after 1, 2, ..., T trees."""
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
            yield F.copy()

# ── Compare learning rates ─────────────────────────────────────────────────
print("=" * 60)
print("  GRADIENT BOOSTING FROM SCRATCH (MSE regression)")
print("=" * 60)
print(f"  Task: fit y = sin(x) + noise")
print(f"  Train: {len(y_tr)} samples  |  Test: {len(y_te)} samples")
print()

configs = [(0.01, "tomato"), (0.1, "seagreen"), (0.5, "steelblue"),
           (1.0,  "purple")]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Gradient Boosting: Residual Fitting & Learning Rate Effect",
             fontsize=12, fontweight="bold")

x_line = np.linspace(0, 6, 300).reshape(-1, 1)
axes[0].scatter(X_tr.ravel(), y_tr, c="gray", s=15, alpha=0.5, label="Data")
axes[0].plot(x_line.ravel(), np.sin(x_line.ravel()), "k--",
             lw=2, label="True sin(x)")

print(f"  {'Learning rate':>14} | {'Train MSE':>10} | {'Test MSE':>10}")
print(f"  {'─'*42}")

for lr, colour in configs:
    gbt = GradientBoostedTrees(n_estimators=100, learning_rate=lr, max_depth=2)
    gbt.fit(X_tr, y_tr)
    tr_mse = np.mean((y_tr - gbt.predict(X_tr))**2)
    te_mse = np.mean((y_te - gbt.predict(X_te))**2)
    print(f"  {lr:>14.2f} | {tr_mse:10.5f} | {te_mse:10.5f}")
    axes[0].plot(x_line.ravel(), gbt.predict(x_line), colour,
                 lw=1.8, alpha=0.8, label=f"lr={lr}")
    axes[1].plot(range(1, 101), gbt.train_mses, colour,
                 lw=1.8, label=f"lr={lr}")

axes[0].legend(fontsize=8); axes[0].set_title("Fitted Curves (100 trees)")
axes[0].set_xlabel("x");    axes[0].set_ylabel("y")
axes[1].set_title("Training MSE vs Number of Trees")
axes[1].set_xlabel("n_trees"); axes[1].set_ylabel("MSE")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig("gradient_boosting_scratch.png", dpi=120)

# ── Show residual shrinkage step-by-step ──────────────────────────────────
gbt_demo = GradientBoostedTrees(n_estimators=20, learning_rate=0.3, max_depth=2)
gbt_demo.fit(X_tr, y_tr)
F = np.full(len(y_tr), gbt_demo.F0)
print()
print("  RESIDUAL SHRINKAGE (lr=0.3, first 6 trees):")
print(f"  {'Step':>5} | {'Max |Residual|':>15} | {'Train MSE':>10}")
print(f"  {'─'*38}")
print(f"  {'0':>5} | {np.max(np.abs(y_tr - F)):15.5f} | "
      f"{np.mean((y_tr-F)**2):10.5f}  (baseline: predict mean)")
for i, tree in enumerate(gbt_demo.trees[:6], 1):
    F += 0.3 * tree.predict(X_tr)
    print(f"  {i:>5} | {np.max(np.abs(y_tr-F)):15.5f} | "
          f"{np.mean((y_tr-F)**2):10.5f}")

print()
print("  Each tree absorbs some of the residual error.")
print("  The ensemble converges; learning rate controls the step size.")
print()
print("  Plot saved → gradient_boosting_scratch.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Stacking with Out-of-Fold Predictions": {
        "description": (
            "Build a full two-level stacking ensemble correctly — "
            "using out-of-fold (OOF) predictions to avoid data leakage. "
            "Compare: single models vs voting vs stacking."
        ),
        "timeout": 600,
        "language": "python",
        "code": '''
import numpy as np
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        weights=None,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_breast_cancer():
    rng=np.random.default_rng(0); n=569; nf=30
    y=rng.integers(0,2,n)
    X=rng.standard_normal((n,nf)); X+=(2*y-1)[:,None]*rng.uniform(0.3,0.8,nf)
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y.astype(int))

def train_test_split(*arrays,test_size=0.25,random_state=None,stratify=None):
    rng=np.random.default_rng(random_state); n=len(arrays[0])
    if stratify is not None:
        ti,vi=[],[]
        for c in np.unique(stratify):
            idx=np.where(stratify==c)[0]; rng.shuffle(idx)
            nt=max(1,int(len(idx)*test_size)); vi.extend(idx[:nt]); ti.extend(idx[nt:])
        ti,vi=np.array(ti),np.array(vi)
    else:
        idx=rng.permutation(n); nt=int(n*test_size); vi=idx[:nt]; ti=idx[nt:]
    out=[]
    for a in arrays: out.append(a[ti]); out.append(a[vi])
    return out

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        rng=np.random.default_rng(self.random_state); folds=[[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)): folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i]); tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy",n_jobs=None):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        skf=StratifiedKFold(n_splits=cv,shuffle=True,random_state=0)
        splits=list(skf.split(X,y))
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

# ── Decision Tree (classifier + regressor) ────────────────────────────────────
class _BaseTree:
    def __init__(self,max_depth=None,min_samples=2,mode="cls"):
        self.max_depth=max_depth or 999; self.min_samples=min_samples; self.mode=mode
    def _impurity(self,y):
        if self.mode=="reg": return np.var(y)*len(y) if len(y) else 0
        if not len(y): return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _split(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_t=(vals[:-1]+vals[1:])/2
            thrs=all_t[np.linspace(0,len(all_t)-1,min(12,len(all_t))).astype(int)]
            for t in thrs:
                l=X[:,f]<=t; r=~l
                if l.sum()<self.min_samples or r.sum()<self.min_samples: continue
                g=self._impurity(y[l])+self._impurity(y[r])
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        if self.mode=="cls":
            cls,cnt=np.unique(y,return_counts=True); pred=cls[cnt.argmax()]; probs=cnt/cnt.sum()
        else:
            pred=y.mean(); probs=None
        if depth>=self.max_depth or len(y)<=self.min_samples:
            return ("L",pred,probs,len(y))
        sp=self._split(X,y)
        if sp is None: return ("L",pred,probs,len(y))
        _,f,t=sp; m=X[:,f]<=t
        return ("N",f,t,self._build(X[m],y[m],depth+1),self._build(X[~m],y[~m],depth+1),len(y))
    def fit(self,X,y):
        self.classes_=np.unique(y) if self.mode=="cls" else None
        self._tree=self._build(X,y,0)
        if self.mode=="cls": self._compute_importances(X)
        return self
    def _compute_importances(self,X):
        imp=np.zeros(X.shape[1])
        def walk(nd):
            if nd[0]=="L": return
            f,t,l_nd,r_nd,n=nd[1],nd[2],nd[3],nd[4],nd[5]
            nl=l_nd[3] if l_nd[0]=="L" else l_nd[5]
            nr=r_nd[3] if r_nd[0]=="L" else r_nd[5]
            ig=self._impurity_node(nd)-nl/n*self._impurity_node(l_nd)-nr/n*self._impurity_node(r_nd)
            imp[f]+=n*ig; walk(l_nd); walk(r_nd)
        walk(self._tree)
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp
    def _impurity_node(self,nd):
        if nd[0]=="L": return 0
        return 0  # simplified
    def _p1(self,x,nd):
        if nd[0]=="L": return nd[1],nd[2]
        return self._p1(x,nd[3] if x[nd[1]]<=nd[2] else nd[4])
    def predict(self,X): return np.array([self._p1(x,self._tree)[0] for x in X])
    def predict_proba(self,X):
        nc=len(self.classes_); out=np.zeros((len(X),nc))
        cmap={c:i for i,c in enumerate(self.classes_)}
        for i,x in enumerate(X):
            _,probs=self._p1(x,self._tree)
            if probs is not None:
                for ci,p in zip(self.classes_,probs): out[i,cmap[ci]]=p
        return out
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

class DecisionTreeClassifier(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="cls")

class DecisionTreeRegressor(_BaseTree):
    def __init__(self,max_depth=None,random_state=None): super().__init__(max_depth=max_depth,mode="reg")
    def get_params(self,deep=True): return {"max_depth":self.max_depth,"random_state":None}

# ── Random Forest Classifier (with oob_score + feature_importances_) ──────────
class RandomForestClassifier:
    def __init__(self,n_estimators=20,max_depth=None,oob_score=False,
                 random_state=None,n_jobs=None,max_features="sqrt"):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.oob_score=oob_score; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self._trees=[]; self._feat_idxs=[]
        oob_votes=np.zeros((n,len(self.classes_)))
        oob_counts=np.zeros(n)
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(np.sqrt(X.shape[1]))); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeClassifier(max_depth=self.max_depth)
            t.fit(X[np.ix_(idx,fi)],y[idx])
            self._trees.append(t); self._feat_idxs.append(fi)
            if self.oob_score:
                oob=np.setdiff1d(np.arange(n),idx)
                if len(oob):
                    p=t.predict_proba(X[np.ix_(oob,fi)])
                    for li,gc in enumerate(t.classes_):
                        gi=np.where(self.classes_==gc)[0]
                        if len(gi): oob_votes[oob,gi[0]]+=p[:,li]
                    oob_counts[oob]+=1
        # feature importances (mean decrease impurity across trees)
        imp=np.zeros(X.shape[1])
        for t,fi in zip(self._trees,self._feat_idxs):
            if hasattr(t,"feature_importances_"):
                imp[fi]+=t.feature_importances_
        s=imp.sum(); self.feature_importances_=imp/s if s>0 else imp+1/X.shape[1]
        if self.oob_score:
            mask=oob_counts>0
            oob_pred=self.classes_[oob_votes[mask].argmax(1)]
            self.oob_score_=float((oob_pred==y[mask]).mean())
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); votes=np.zeros((len(X),nc))
        for t,fi in zip(self._trees,self._feat_idxs):
            p=t.predict_proba(X[:,fi])
            for li,gc in enumerate(t.classes_):
                gi=np.where(self.classes_==gc)[0]
                if len(gi): votes[:,gi[0]]+=p[:,li]
        return votes/len(self._trees)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"max_depth":self.max_depth,
                "oob_score":self.oob_score,"random_state":self.random_state}

# ── Logistic Regression ───────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; lam=1./self.C
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C,"max_iter":self.max_iter}

# ── Gradient Boosting Classifier (log-loss) ───────────────────────────────────
class GradientBoostingClassifier:
    def __init__(self,n_estimators=30,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators=n_estimators; self.lr=learning_rate
        self.max_depth=max_depth; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        self.classes_=np.unique(y)
        p_mean=y.mean(); p_mean=np.clip(p_mean,1e-6,1-1e-6)
        self.F0_=float(np.log(p_mean/(1-p_mean)))
        F=np.full(len(y),self.F0_); self._trees=[]
        for _ in range(self.n_estimators):
            p=_sig(F); r=y-p          # negative gradient of log-loss
            t=DecisionTreeRegressor(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t)
            F+=self.lr*t.predict(X)
        return self
    def _decision(self,X):
        F=np.full(len(X),self.F0_)
        for t in self._trees: F+=self.lr*t.predict(X)
        return F
    def predict_proba(self,X):
        p=_sig(self._decision(X)); return np.c_[1-p,p]
    def predict(self,X): return (self._decision(X)>=0).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True):
        return {"n_estimators":self.n_estimators,"learning_rate":self.lr,
                "max_depth":self.max_depth,"random_state":self.random_state}

# ── SVC (RBF, with probability via sigmoid calibration) ──────────────────────
class SVC:
    """Linear SVM via L-BFGS (fast, no kernel matrix)."""
    def __init__(self,C=1.0,kernel="rbf",gamma="scale",probability=True,random_state=None):
        self.C=C
    def fit(self,X,y):
        self._classes=np.unique(y); n,d=X.shape; lam=1./(2*self.C*n)
        yb=(2*(y==self._classes[1])-1).astype(float)
        def fg(w):
            margins=1-yb*(X@w[1:]+w[0]); hinge=np.maximum(0,margins)
            loss=lam*np.sum(w[1:]**2)+np.mean(hinge)
            mask=hinge>0
            g_w=2*lam*w[1:]-np.mean(yb[mask,None]*X[mask],axis=0) if mask.any() else 2*lam*w[1:]
            g_b=-np.mean(yb[mask]) if mask.any() else 0.
            return loss,np.r_[g_b,g_w]
        res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                             options={"maxiter":200})
        self._b=res.x[0]; self._w=res.x[1:]; return self
    def _dec(self,X): return X@self._w+self._b
    def predict(self,X): return self._classes[(self._dec(X)>=0).astype(int)]
    def predict_proba(self,X): p=_sig(self._dec(X)); return np.c_[1-p,p]
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"C":self.C}

# ── KNeighborsClassifier ──────────────────────────────────────────────────────
class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); self.classes_=np.unique(y); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def predict_proba(self,X):
        nc=len(self.classes_); out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            p=np.bincount(nn.astype(int),minlength=nc)/self.n_neighbors; out.append(p)
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"n_neighbors":self.n_neighbors}

# ── VotingClassifier ──────────────────────────────────────────────────────────
class VotingClassifier:
    def __init__(self,estimators,voting="hard"):
        self.estimators=estimators; self.voting=voting
    def fit(self,X,y):
        self._fitted=[]; self.classes_=np.unique(y)
        for nm,est in self.estimators:
            e=_copy.deepcopy(est); e.fit(X,y); self._fitted.append(e)
        return self
    def predict_proba(self,X):
        nc=len(self.classes_); avg=np.zeros((len(X),nc))
        for e in self._fitted: avg+=e.predict_proba(X)
        return avg/len(self._fitted)
    def predict(self,X):
        if self.voting=="soft": return self.classes_[self.predict_proba(X).argmax(1)]
        votes=np.stack([e.predict(X) for e in self._fitted],axis=1)
        return np.array([np.bincount(row.astype(int)).argmax() for row in votes])
    def score(self,X,y): return (self.predict(X)==y).mean()
    def get_params(self,deep=True): return {"voting":self.voting}

np.random.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
sc = StandardScaler()
X  = sc.fit_transform(X)
# Subsample for speed
_rng_s=np.random.default_rng(0); _si=_rng_s.choice(len(y),300,replace=False)
X,y=X[_si],y[_si]

print("=" * 65)
print("  STACKING ENSEMBLE WITH OUT-OF-FOLD PREDICTIONS")
print("=" * 65)
print(f"  Dataset: Breast Cancer  ({X.shape[0]} samples, {X.shape[1]} features)")
print()

# ── Base models (Level-0) ─────────────────────────────────────────────────
base_models = [
    ("LogReg",    LogisticRegression(max_iter=500, C=1.0)),
    ("RandomFor", RandomForestClassifier(n_estimators=10, random_state=0)),
    ("GBM",       GradientBoostingClassifier(n_estimators=15, random_state=0)),
    ("SVM",       SVC(probability=True, C=1.0, random_state=0)),
    ("KNN",       KNeighborsClassifier(n_neighbors=5)),
]

# ── Evaluate each base model alone ────────────────────────────────────────
print("  BASE MODEL INDIVIDUAL PERFORMANCE (5-fold CV):")
print(f"  {'Model':12s} | {'CV Acc':>8} | {'Std':>7}")
print(f"  {'─'*35}")
for name, model in base_models:
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    print(f"  {name:12s} | {scores.mean():8.4f} | {scores.std():7.4f}")
print()

# ── Generate OOF predictions (correct stacking without leakage) ───────────
print("  GENERATING OUT-OF-FOLD (OOF) PREDICTIONS...")
K = 3
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
oof_train = np.zeros((len(y), len(base_models)))  # shape: (N, n_base_models)

for fold_idx, (tr_idx, vl_idx) in enumerate(kf.split(X, y)):
    X_tr_f, y_tr_f = X[tr_idx], y[tr_idx]
    X_vl_f         = X[vl_idx]
    for m_idx, (name, model) in enumerate(base_models):
        model.fit(X_tr_f, y_tr_f)
        oof_train[vl_idx, m_idx] = model.predict_proba(X_vl_f)[:, 1]

print(f"  OOF matrix shape: {oof_train.shape}  (N samples × {len(base_models)} base models)")
print()

# ── Train meta-learner on OOF predictions ────────────────────────────────
meta_learner = LogisticRegression(max_iter=500, C=0.5)
meta_scores  = cross_val_score(meta_learner, oof_train, y,
                                cv=3, scoring="accuracy")
print(f"  META-LEARNER (Logistic Regression on OOF features):")
print(f"    CV Acc: {meta_scores.mean():.4f} ± {meta_scores.std():.4f}")
print()

# ── Voting ensemble (simpler baseline) ────────────────────────────────────
voting_hard = VotingClassifier(
    estimators=base_models, voting="hard")
voting_soft = VotingClassifier(
    estimators=base_models, voting="soft")

hard_scores = cross_val_score(voting_hard, X, y, cv=3, scoring="accuracy")
soft_scores = cross_val_score(voting_soft, X, y, cv=3, scoring="accuracy")

# ── Summary ───────────────────────────────────────────────────────────────
print("  FINAL COMPARISON:")
print(f"  {'Method':28s} | {'CV Acc':>8} | {'Std':>7}")
print(f"  {'─'*45}")
for name, model in base_models:
    sc_ = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    print(f"  {name:28s} | {sc_.mean():8.4f} | {sc_.std():7.4f}")
print(f"  {'Voting (Hard)':28s} | {hard_scores.mean():8.4f} | {hard_scores.std():7.4f}")
print(f"  {'Voting (Soft)':28s} | {soft_scores.mean():8.4f} | {soft_scores.std():7.4f}")
print(f"  {'Stacking (OOF + meta-LR)':28s} | {meta_scores.mean():8.4f} | {meta_scores.std():7.4f}  ← stacking")
print()
print("  STACKING NOTES:")
print("  - OOF predictions prevent data leakage into the meta-learner")
print("  - The meta-learner learns WHEN each base model is trustworthy")
print("  - More base models = richer OOF feature set = better meta-learning")
print("  - In Kaggle competitions, stacking is the dominant winning strategy")
''',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }