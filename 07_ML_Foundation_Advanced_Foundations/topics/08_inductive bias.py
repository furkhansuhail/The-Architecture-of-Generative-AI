"""
Inductive Bias
==============

Every ML model bets on structure. The assumptions baked into an
architecture — smoothness, locality, translation invariance, linearity —
are its inductive bias. Understanding these determines which problems
a model solves well and where it will fail.

"""

import textwrap
import re

TOPIC_NAME = "Inductive Bias"
DISPLAY_NAME = "08 · Inductive Bias"
ICON = "🎯"
SUBTITLE = "Every Model Bets on Structure — What Does Yours Assume?"

THEORY = """

### What Is Inductive Bias?

Any learning algorithm that generalises must go beyond the training data.
Given finite observations, infinitely many hypotheses are consistent with
those observations. The learner must prefer some over others.

    ┌──────────────────────────────────────────────────────────────────┐
    │  INDUCTIVE BIAS = the set of assumptions, preferences, and       │
    │  constraints that a learning algorithm uses to generalise        │
    │  from training examples to unseen examples.                      │
    │                                                                  │
    │  Without inductive bias, generalisation is impossible.           │
    │  (No Free Lunch Theorem: no algorithm is better than random on   │
    │   all possible tasks — Module 10.)                               │
    │                                                                  │
    │  With the RIGHT inductive bias for your task, learning is        │
    │  sample-efficient and generalisable.                             │
    │  With the WRONG inductive bias, no amount of data helps.         │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 1 — Inductive Bias as a Prior on Functions:

    The hypothesis space H contains all possible functions.
    Inductive bias restricts H to a subset H' (explicit bias)
    or assigns a preference order over H (implicit bias).

    All possible functions f: X → Y
    ┌────────────────────────────────────────────────────────────────┐
    │    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○       │
    │    ○ ○ [linear models: H' = {f: f(x) = wᵀx+b}] ○ ○ ○ ○ ○       │
    │    ○   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○         │
    │    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○         │
    └────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Inductive Biases of Common Architectures

**Linear Models — smoothness and global structure:**

    Assumption: f(x) = wᵀx + b  (linearity in features).
    Implicit bias of gradient descent: converges to the minimum-norm
    solution among all linear functions fitting the data.
    Good for: tabular data with independent features, high-d sparse problems.
    Fails on: non-linear relationships, interactions between features.

**Decision Trees — axis-aligned partitions:**

    Assumption: the decision boundary consists of axis-aligned hyperplanes.
    Prediction = a lookup in a table of conditions on individual features.
    Good for: categorical features, ordinal thresholds, no normalisation needed.
    Fails on: diagonal boundaries, smooth manifolds, rotation-variant problems.

    Diagram 2 — Decision Tree Inductive Bias:

    ┌─────────────────────┐    ┌────────────────────────┐
    │  LEARNED (axis-     │    │  ACTUAL boundary       │
    │  aligned steps):    │    │  (diagonal, smooth):   │
    │                     │    │                        │
    │  ─────────┐         │    │  ///////               │
    │           │         │    │  //////                │
    │  ─────────┘         │    │  /////                 │
    └─────────────────────┘    └────────────────────────┘
     Many axis-aligned steps     Smooth diagonal line
     needed to approximate

**k-NN — manifold smoothness / local similarity:**

    Assumption: nearby inputs should have the same label.
    Implicit metric: Euclidean distance (or other chosen metric).
    Good for: locally smooth functions, clear cluster structure.
    Fails on: high dimensions (curse of dimensionality), irrelevant features
    dominate distance, when "nearby" in feature space ≠ nearby semantically.

**Support Vector Machines — maximum margin:**

    Assumption: the largest-margin separator is the most generalisable.
    Kernel trick allows non-linear boundaries (but must choose kernel).
    Good for: binary classification with clear margins, text (linear SVM).
    Fails on: n >> 1000 (O(n³) training), multi-class, overlapping classes.

**Convolutional Neural Networks (CNNs) — locality and translation invariance:**

    Two explicit inductive biases:
    1. LOCALITY: each filter looks at a small receptive field (spatial locality).
       "Nearby pixels are more related than far pixels."
    2. TRANSLATION EQUIVARIANCE: the same filter is applied everywhere.
       "A cat in the top-left and a cat in the bottom-right should activate
       the same feature detectors."
    Good for: image data, any structured spatial/temporal data.
    Fails on: problems where location matters absolutely (not relatively),
    long-range dependencies (addressed by transformers).

**Recurrent Neural Networks — temporal order:**

    Assumption: sequential structure matters; earlier inputs affect later ones.
    Hidden state = compressed summary of history.
    Good for: time series, language with short dependencies.
    Fails on: long-range dependencies (vanishing gradient), parallelisation.

**Transformers — global pairwise attention:**

    Assumption: any pair of tokens can be directly relevant to each other.
    No locality bias — all positions can attend to all positions.
    Good for: language (long-range deps), images (global context, ViT).
    Fails on: small data (strong bias required; transformers need large data),
    strict spatial locality (CNN is more efficient for that).

    Diagram 3 — CNN vs Transformer Receptive Field:

    CNN layer 1:         CNN layer 3:         Transformer:
    ┌────────────┐      ┌─────────────┐      ┌─────────────┐
    │ × ○ ○ ○ ○  │      │ × × × ○ ○   │      │ × × × × ×   │
    │ ○ ○ ○ ○ ○  │      │ × × × ○ ○   │      │ × × × × ×   │
    │ ○ ○ ○ ○ ○  │      │ × × × × ○   │      │ × × × × ×   │
    │ ○ ○ ○ ○ ○  │      │ ○ ○ ○ ○ ○   │      │ × × × × ×   │
    └────────────┘      └─────────────┘      └─────────────┘
     Small local field    Growing field         Full field
     (3×3 kernel)         (stacked layers)      (one layer!)

**Graph Neural Networks — permutation invariance, neighbourhood:**

    Assumption: node features + local graph structure determine labels.
    Message passing: each node aggregates information from its neighbours.
    Permutation invariant: relabelling nodes doesn't change the output.
    Good for: social networks, molecules, knowledge graphs.
    Fails on: long-range structural information (many hops needed), heterophily.

**Gaussian Processes — smoothness via kernel:**

    Assumption: f ~ GP(0, k(·,·)) — the kernel encodes smoothness assumptions.
    RBF kernel: "nearby inputs have similar outputs" (exponential decay).
    The choice of kernel IS the choice of inductive bias.
    Good for: small data, uncertainty quantification, regression with known priors.


──────────────────────────────────────────────────────────────────────────────
### Explicit vs Implicit Inductive Bias

**Explicit bias** — hard constraints on the hypothesis class:
    ─ "The function must be linear" → linear model.
    ─ "The boundary must be a hyperplane" → SVM.
    ─ "The function must be monotone in feature 3" → isotonic regression.
    These can be expressed as constraints on the model's parameters.

**Implicit bias of optimisation** — which solution gradient descent finds:
    Even without explicit regularisation, SGD/gradient descent has a
    preference for certain types of solutions.

    ┌──────────────────────────────────────────────────────────────────┐
    │  For linear models:                                              │
    │  GD from zero init converges to the MINIMUM L2 NORM solution     │
    │  (implicit L2 regularisation).                                   │
    │                                                                  │
    │  For matrix factorisation:                                       │
    │  GD converges to the MINIMUM NUCLEAR NORM solution               │
    │  (implicit low-rank regularisation).                             │
    │                                                                  │
    │  For neural networks:                                            │
    │  SGD with small learning rate prefers FLAT MINIMA                │
    │  (connected to the PAC-Bayes / sharpness-aware ideas).           │
    │  Small weight initialisation + SGD → edge-of-stability dynamics. │
    └──────────────────────────────────────────────────────────────────┘

    Why implicit bias matters for deep learning:
    Over-parameterised neural networks (m >> n) can memorise training data.
    Many solutions with zero training loss exist.
    SGD picks a specific one — the implicit bias determines which,
    and empirically that solution generalises well.


──────────────────────────────────────────────────────────────────────────────
### The Bias-Variance-Inductive Bias Triangle

Inductive bias, model bias, and regularisation are three names for the
same phenomenon at different levels of description.

    ┌──────────────────────────────────────────────────────────────────┐
    │  INDUCTIVE BIAS    = the prior assumptions about the task.       │
    │                                                                  │
    │  MODEL BIAS        = systematic error from wrong assumptions     │
    │                      (underfitting when bias is too strong).     │
    │                                                                  │
    │  REGULARISATION    = explicit penalty to enforce inductive bias  │
    │                      during optimisation.                        │
    │                                                                  │
    │  Relationship:                                                   │
    │  Strong inductive bias → low variance, high bias.                │
    │  Weak inductive bias   → high variance, low bias.                │
    │  Correct inductive bias → best of both worlds.                   │
    └──────────────────────────────────────────────────────────────────┘

    Diagram 4 — Matching Inductive Bias to Problem Structure:

    Problem: f(x) is a smooth function of 2D input.

    CORRECT BIAS (RBF kernel, neural net):
        Low training error + good generalisation.

    TOO STRONG (linear model):
        High training error (underfitting). Bias >> variance.

    TOO WEAK (high-degree polynomial, no regularisation):
        Low training error + poor generalisation (overfitting). Variance >> bias.


──────────────────────────────────────────────────────────────────────────────
### Symmetry as Inductive Bias

Modern deep learning theory connects inductive bias to symmetry groups.

    TRANSLATION EQUIVARIANCE (CNNs):
        f(shift(x)) = shift(f(x))
        The output shifts when the input shifts.
        This is built into CNNs via weight sharing (same filters everywhere).

    ROTATION EQUIVARIANCE (equivariant networks):
        f(rotate(x)) = rotate(f(x))
        For molecular property prediction: rotating a molecule should
        not change its predicted energy. E(3)-equivariant networks.

    PERMUTATION INVARIANCE (GNNs, Set Transformers):
        f({x₁,...,xₙ}) = f({xₙ,...,x₁})
        The set prediction should not change if we reorder its elements.
        DeepSets: f(S) = ρ(Σₓ∈S φ(x))  — sum over independent embeddings.

    ┌──────────────────────────────────────────────────────────────────┐
    │  GAUGE: if your problem has a known symmetry, encode it in the   │
    │  architecture. This reduces the effective number of parameters,  │
    │  improves sample efficiency, and prevents the model from         │
    │  wasting capacity learning to be invariant.                      │
    └──────────────────────────────────────────────────────────────────┘

    Example: AlphaFold2 uses equivariance to 3D rotations and reflections
    as a core inductive bias. This is a key reason it achieves such
    accurate protein structure prediction.


──────────────────────────────────────────────────────────────────────────────
### Choosing the Right Inductive Bias

    A practical guide:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Problem structure          → Inductive bias           → Model   │
    ├──────────────────────────────────────────────────────────────────┤
    │  Spatial locality            → Local filters             → CNN   │
    │  Long-range dependencies     → Global attention    → Transformer │
    │  Graph/relational structure  → Neighbourhood aggregation → GNN   │
    │  Temporal order              → Recurrence/attention  → RNN/Attn  │
    │  Linear relationships        → Linearity             → Lin. Reg  │
    │  Monotone relationships      → Monotone constraints  → Isotonic  │
    │  Small data, known smooth.   → GP / Bayesian neural net          │
    │  Physical laws (conserv.)    → Physics-informed NN (PINN)        │
    │  Permutation invariant       → DeepSets / GNN                    │
    │  Rotation invariant          → Equivariant network               │
    │  Tabular, mixed types        → Gradient boosting (weak bias)     │
    └──────────────────────────────────────────────────────────────────┘

    When in doubt: start with gradient boosting (weak bias, flexible).
    If you know structure: encode it explicitly for sample efficiency.
    If data is large: weak bias (deep net) works; the data provides the bias.

"""

OPERATIONS = {

    "1 · Inductive Bias Comparison — Same Data, Different Models": {
        "description": (
            "Fits six model families on the same synthetic dataset with "
            "varying underlying function structures (linear, sine, step, "
            "spiral). Shows which model's inductive bias matches which "
            "problem by comparing train/test performance. Visualises "
            "decision boundaries to make the bias differences concrete. "
            "Demonstrates how wrong inductive bias causes systematic "
            "underfitting that cannot be fixed with more data."
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

def _sig(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=1000,random_state=None,**kw):
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

class DecisionTreeClassifier:
    def __init__(self,max_depth=None,random_state=None,**kw): self.max_depth=max_depth or 999
    def _imp(self,y):
        if not len(y): return 0
        _,c=_np_impl.unique(y,return_counts=True); p=c/c.sum(); return 1-_np_impl.sum(p**2)
    def _best(self,X,y):
        best=None
        for f in range(X.shape[1]):
            vals=_np_impl.unique(X[:,f])
            if len(vals)<2: continue
            ts=(vals[:-1]+vals[1:])/2
            if len(ts)>20: ts=ts[_np_impl.linspace(0,len(ts)-1,20).astype(int)]
            for t in ts:
                l=X[:,f]<=t; r=~l
                if l.sum()<2 or r.sum()<2: continue
                g=self._imp(y)-l.sum()/len(y)*self._imp(y[l])-r.sum()/len(y)*self._imp(y[r])
                if best is None or g>best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,d):
        cls,cnt=_np_impl.unique(y,return_counts=True); pred=cls[cnt.argmax()]
        if d>=self.max_depth or len(_np_impl.unique(y))==1 or len(y)<=2: return ('L',pred)
        sp=self._best(X,y)
        if sp is None: return ('L',pred)
        _,f,t=sp; l=X[:,f]<=t
        return ('N',f,t,self._build(X[l],y[l],d+1),self._build(X[~l],y[~l],d+1))
    def fit(self,X,y): self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd[0]=='L': return nd[1]
        _,f,t,L,R=nd; return self._p1(x,L if x[f]<=t else R)
    def predict(self,X): return _np_impl.array([self._p1(x,self._tree) for x in X])
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5,**kw): self.k=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        out=[]
        for x in X:
            d=_np_impl.sqrt(((self._X-x)**2).sum(1))
            nn=self._y[_np_impl.argsort(d)[:self.k]]
            vals,cnt=_np_impl.unique(nn,return_counts=True); out.append(vals[cnt.argmax()])
        return _np_impl.array(out)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class SVC:
    def __init__(self,kernel="rbf",C=1.0,gamma="scale",**kw):
        self.C=C; self.kernel=kernel; self._g=gamma
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); yb=_np_impl.where(y==self._classes[1],1.0,-1.0)
        n,d=X.shape
        g=1.0/(d*X.var()) if self._g=="scale" else float(self._g)
        if self.kernel=="rbf":
            rng=_np_impl.random.default_rng(0); D=min(200,4*d)
            self._W=rng.normal(0,_np_impl.sqrt(2*g),(d,D))
            self._b_rff=rng.uniform(0,2*_np_impl.pi,D)
            Xp=_np_impl.sqrt(2/D)*_np_impl.cos(X@self._W+self._b_rff)
        else: Xp=X
        lam=1.0/(self.C*n)
        def fg(w):
            m=yb*(Xp@w[1:]+w[0]); hinge=_np_impl.maximum(0,1-m)
            loss=lam*_np_impl.sum(w[1:]**2)+_np_impl.mean(hinge)
            mask=hinge>0
            gw=2*lam*w[1:]-(_np_impl.mean(yb[mask,None]*Xp[mask],axis=0) if mask.any() else 0)
            gb=-_np_impl.mean(yb[mask]) if mask.any() else 0.0
            return loss,_np_impl.r_[gb,gw]
        res=_sp_opt.minimize(fg,_np_impl.zeros(Xp.shape[1]+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':500})
        self._b0=res.x[0]; self._w=res.x[1:]; return self
    def _transform(self,X):
        if self.kernel=="rbf":
            D=self._W.shape[1]
            return _np_impl.sqrt(2/D)*_np_impl.cos(X@self._W+self._b_rff)
        return X
    def predict(self,X): return self._classes[(self._transform(X)@self._w+self._b0>=0).astype(int)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),max_iter=200,random_state=None,
                 activation='relu',alpha=0.0001,**kw):
        self.hidden_layer_sizes=hidden_layer_sizes; self.max_iter=max_iter
        self.random_state=random_state; self.alpha=alpha
    def fit(self,X,y):
        rng=_np_impl.random.default_rng(self.random_state)
        self._classes=_np_impl.unique(y); nc=len(self._classes)
        dims=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,_np_impl.sqrt(2.0/dims[i]),(dims[i],dims[i+1]))
                     for i in range(len(dims)-1)]
        self.intercepts_=[_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n=len(X); lr=1e-3
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for s in range(0,n,32):
                xb=X[idx[s:s+32]]; yb=y[idx[s:s+32]]; nb=len(xb)
                acts=[xb]
                for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+b
                    acts.append(_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z)
                logits=acts[-1]-acts[-1].max(1,keepdims=True)
                exp=_np_impl.exp(logits); probs=exp/exp.sum(1,keepdims=True)
                oh=_np_impl.zeros_like(probs)
                for ci,c in enumerate(self._classes): oh[yb==c,ci]=1
                delta=(probs-oh)/nb
                for i in range(len(self.coefs_)-1,-1,-1):
                    self.coefs_[i]-=lr*(acts[i].T@delta+self.alpha*self.coefs_[i])
                    self.intercepts_[i]-=lr*delta.sum(0)
                    if i>0: delta=(delta@self.coefs_[i].T)*(acts[i]>0)
        return self
    def predict(self,X):
        a=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=a@W+b; a=_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z
        a=a-a.max(1,keepdims=True); e=_np_impl.exp(a); probs=e/e.sum(1,keepdims=True)
        return self._classes[probs.argmax(1)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

class _GBDTR:
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
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.lr=learning_rate
    def fit(self,X,y):
        yf=y.astype(float); p0=_np_impl.clip(yf.mean(),1e-6,1-1e-6)
        self._f0=_np_impl.log(p0/(1-p0)); F=_np_impl.full(len(y),self._f0); self._trees=[]
        self._classes=_np_impl.unique(y)
        for _ in range(self.n_estimators):
            r=yf-_sig(F); t=_GBDTR(max_depth=self.max_depth)
            t.fit(X,r); self._trees.append(t); F+=self.lr*t.predict(X)
        return self
    def predict(self,X):
        F=_np_impl.full(len(X),self._f0)
        for t in self._trees: F+=self.lr*t.predict(X)
        return self._classes[(_sig(F)>=0.5).astype(int)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]))
    return _np_impl.array(scores)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Datasets with different true structure
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(kind, n=400, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "linear":
        X = rng.standard_normal((n, 2))
        y = (X[:,0] + 0.5*X[:,1] + rng.normal(0, noise, n) > 0).astype(int)
    elif kind == "circles":
        r = rng.uniform(0, 1, n); theta = rng.uniform(0, 2*np.pi, n)
        X = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        y = (r + rng.normal(0, noise, n) < 0.5).astype(int)
    elif kind == "xor":
        X = rng.standard_normal((n, 2))
        y = ((X[:,0] > 0) == (X[:,1] > 0)).astype(int)
        X += rng.normal(0, noise, X.shape)
    elif kind == "spiral":
        t = rng.uniform(0, 3*np.pi, n//2)
        r = t / (3*np.pi)
        X0 = np.column_stack([r*np.cos(t), r*np.sin(t)])
        X1 = np.column_stack([-r*np.cos(t), -r*np.sin(t)])
        X  = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
        y  = np.concatenate([np.zeros(n//2), np.ones(n//2)]).astype(int)
    return X, y

datasets = {
    "Linear":  make_dataset("linear"),
    "Circles": make_dataset("circles"),
    "XOR":     make_dataset("xor"),
    "Spiral":  make_dataset("spiral"),
}

models = {
    "Logistic\\n(linear)":       LogisticRegression(C=1.0),
    "Decision\\nTree":           DecisionTreeClassifier(max_depth=5),
    "k-NN\\n(k=5)":              KNeighborsClassifier(n_neighbors=5),
    "SVM\\n(RBF)":               SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Neural\\nNet":              MLPClassifier(hidden_layer_sizes=(64,32),
                                              max_iter=500, random_state=0),
    "Gradient\\nBoosting":       GradientBoostingClassifier(n_estimators=100,
                                                            random_state=0),
}

print("=" * 65)
print("  INDUCTIVE BIAS: WHICH MODEL FITS WHICH PROBLEM?")
print("=" * 65)
print()
print("  Cross-validation accuracy (5-fold):")
print()

# Header
hdr = f"  {'':>20}" + "".join(f"{dn:>12}" for dn in datasets.keys())
print(hdr)
print("  " + "─" * (20 + 12*len(datasets)))

results = {}
for mname, model in models.items():
    row = f"  {mname.replace(chr(10),' '):>20}"
    results[mname] = {}
    for dname, (X, y) in datasets.items():
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean_cv = cv.mean()
        results[mname][dname] = mean_cv
        flag = "★" if mean_cv > 0.90 else (" " if mean_cv > 0.75 else "×")
        row += f"  {mean_cv:.3f}{flag:>1} "
    print(row)

print()
print("  ★ = excellent fit (>90%), × = poor fit (<75%)")
print()
print("  KEY OBSERVATIONS:")
print("  ─ Logistic regression: excellent on linear, poor on XOR/spiral.")
print("  ─ k-NN: good on circles (local metric), can struggle with spiral.")
print("  ─ SVM RBF: good on circles and XOR (smooth non-linear boundary).")
print("  ─ Neural net + GBM: flexible, work well across all tasks.")
print()
print("  WRONG inductive bias = permanent underfitting, even with infinite data.")
print("  LINEAR model on XOR: no amount of data reaches 100% accuracy.")

# ─────────────────────────────────────────────────────────────────────────────
# Decision boundary plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(models), len(datasets),
                          figsize=(4*len(datasets), 3*len(models)))
fig.suptitle("Inductive Bias: Decision Boundaries — Each Model × Each Dataset",
             fontsize=13, fontweight="bold")

h = 0.05
for row_i, (mname, model) in enumerate(models.items()):
    for col_j, (dname, (X, y)) in enumerate(datasets.items()):
        ax = axes[row_i, col_j]
        model.fit(X, y)
        x0_min, x0_max = X[:,0].min()-0.3, X[:,0].max()+0.3
        x1_min, x1_max = X[:,1].min()-0.3, X[:,1].max()+0.3
        xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                                np.arange(x1_min, x1_max, h))
        Z = model.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)
        acc = accuracy_score(y, model.predict(X))
        ax.contourf(xx0, xx1, Z, alpha=0.25, cmap="RdBu")
        ax.scatter(X[y==0,0], X[y==0,1], c="steelblue", s=6, alpha=0.5)
        ax.scatter(X[y==1,0], X[y==1,1], c="tomato",    s=6, alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        cv_val = results[mname][dname]
        ax.set_title(f"{cv_val:.2f}", fontsize=9,
                     color="seagreen" if cv_val>0.90 else ("black" if cv_val>0.75 else "tomato"),
                     fontweight="bold")
        if row_i == 0:
            ax.set_xlabel(dname, fontweight="bold", fontsize=10)
        if col_j == 0:
            ax.set_ylabel(mname.replace("\\n"," "), fontsize=8)

plt.tight_layout()
plt.savefig("inductive_bias_boundaries.png", dpi=95)
print("  Plot saved → inductive_bias_boundaries.png")
''',
    },

    "2 · Implicit Bias of Gradient Descent — Minimum Norm Solutions": {
        "description": (
            "Demonstrates the implicit bias of gradient descent for linear "
            "models: shows that GD from zero initialisation converges to the "
            "minimum L2-norm solution among all interpolating solutions. "
            "Compares GD solution to explicit Ridge (which is minimum norm "
            "for λ→0). Also shows the implicit bias for matrix factorisation "
            "(nuclear norm minimisation) and how learning rate / batch size "
            "affects which minimum is found."
        ),
        "timeout":300,
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Implicit bias of gradient descent: minimum L2-norm interpolation
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  IMPLICIT BIAS OF GRADIENT DESCENT: MINIMUM L2-NORM SOLUTION")
print("=" * 65)
print()
print("  For over-parameterised linear models (d > n), infinitely many")
print("  solutions interpolate the training data (zero training loss).")
print("  GD from zero init converges to the minimum-norm solution.")
print()

n, d = 50, 200   # d >> n: underdetermined system
X_imp = np.random.randn(n, d)
y_imp = np.random.randn(n)

# Minimum-norm solution: w_MN = Xᵀ (XXᵀ)⁻¹ y
w_min_norm = X_imp.T @ np.linalg.solve(X_imp @ X_imp.T, y_imp)
residual_mn = np.linalg.norm(X_imp @ w_min_norm - y_imp)

# Gradient descent from zero init
w_gd = np.zeros(d)
lr   = 0.01
n_steps = 10000
losses_gd = []
norms_gd  = []

for step in range(n_steps):
    grad = 2 * X_imp.T @ (X_imp @ w_gd - y_imp) / n
    w_gd = w_gd - lr * grad
    if step % 100 == 0:
        losses_gd.append(np.mean((X_imp @ w_gd - y_imp)**2))
        norms_gd.append(np.linalg.norm(w_gd))

print(f"  Minimum-norm solution: ‖w_MN‖₂ = {np.linalg.norm(w_min_norm):.6f}")
print(f"  GD solution (10K steps): ‖w_GD‖₂ = {np.linalg.norm(w_gd):.6f}")
print(f"  Difference ‖w_GD - w_MN‖ = {np.linalg.norm(w_gd - w_min_norm):.6f}")
print(f"  Training loss (GD): {losses_gd[-1]:.8f}  (approaching 0)")
print()
print("  GD converges to minimum-norm interpolating solution.")
print()

# Compare Ridge (λ→0) vs GD for varying λ
print("  RIDGE vs GD: norm of solution as λ → 0")
print(f"  {'λ (Ridge)':>14}  {'‖w_Ridge‖':>12}  {'Train MSE':>11}")
print("  " + "─" * 40)
for lam in [1.0, 0.1, 0.01, 0.001, 0.0001, 0.0]:
    if lam == 0:
        w_r = w_min_norm
    else:
        # Ridge: (XᵀX + λI)w = Xᵀy  (dual: w = Xᵀ(XXᵀ + λI)⁻¹y)
        w_r = X_imp.T @ np.linalg.solve(X_imp @ X_imp.T + lam * np.eye(n), y_imp)
    mse = np.mean((X_imp @ w_r - y_imp)**2)
    print(f"  {lam:>14.4f}  {np.linalg.norm(w_r):>12.6f}  {mse:>11.6f}")
print(f"  {'GD from 0':>14}  {np.linalg.norm(w_gd):>12.6f}  "
      f"{np.mean((X_imp@w_gd-y_imp)**2):>11.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Implicit bias in matrix factorisation: nuclear norm minimisation
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  IMPLICIT BIAS: MATRIX FACTORISATION → NUCLEAR NORM")
print("=" * 65)
print()
print("  Factorising M = UV with r < rank(M_true): GD finds low-rank solution.")
print("  This is the implicit bias that powers collaborative filtering / SVD.")
print()

n_rows, n_cols = 20, 15
rank_true      = 3

# True low-rank matrix
U_true = np.random.randn(n_rows, rank_true)
V_true = np.random.randn(rank_true, n_cols)
M_true = U_true @ V_true

# Observe only 50% of entries (like recommendation system)
obs_mask = np.random.rand(n_rows, n_cols) < 0.5
M_obs    = M_true * obs_mask

# Factorisation with r=3 via gradient descent
r_model = 4
U_f = np.random.randn(n_rows, r_model) * 0.01
V_f = np.random.randn(r_model, n_cols) * 0.01
lr_mf = 0.01
for step in range(3000):
    M_pred  = U_f @ V_f
    resid   = (M_pred - M_obs) * obs_mask
    dU = 2 * resid @ V_f.T
    dV = 2 * U_f.T @ resid
    U_f -= lr_mf * dU; V_f -= lr_mf * dV

M_recovered = U_f @ V_f
obs_mse = np.mean(((M_recovered - M_true) * obs_mask)**2)
unobs_mse = np.mean(((M_recovered - M_true) * (1-obs_mask))**2)
nuc_norm  = np.linalg.norm(M_recovered, ord="nuc")
nuc_true  = np.linalg.norm(M_true, ord="nuc")

print(f"  True matrix: rank={rank_true}, nuclear norm={nuc_true:.3f}")
print(f"  Factorisation: r={r_model}")
print(f"  Observed MSE:   {obs_mse:.6f}")
print(f"  Unobserved MSE: {unobs_mse:.6f}  (generalisation via implicit low-rank bias)")
print(f"  Recovered nuclear norm: {nuc_norm:.3f}  (close to true: {nuc_true:.3f})")
print()
print("  Without any explicit regularisation, GD finds the minimum nuclear")
print("  norm matrix consistent with the observed entries.")
print("  This is the implicit bias that enables matrix completion.")

# ─────────────────────────────────────────────────────────────────────────────
# Flat vs sharp minima — sharpness and generalisation
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  FLAT vs SHARP MINIMA: IMPLICIT BIAS OF BATCH SIZE")
print("=" * 65)
print()
print("  Large-batch SGD finds sharper minima (poor generalisation).")
print("  Small-batch SGD finds flatter minima (better generalisation).")
print("  This is the implicit bias of stochastic gradient descent.")
print()

import numpy as _np2
import scipy.optimize as _sp2

def _sig2(z): return 1.0/(1.0+_np2.exp(-_np2.clip(z,-500,500)))

def make_moons(n_samples=100,noise=0.1,random_state=None):
    rng=_np2.random.default_rng(random_state); n=n_samples//2
    t=_np2.linspace(0,_np2.pi,n)
    X=_np2.vstack([_np2.c_[_np2.cos(t),_np2.sin(t)],
                   _np2.c_[1-_np2.cos(t),-_np2.sin(t)+0.5]])
    if noise>0: X+=rng.normal(0,noise,X.shape)
    return X,_np2.hstack([_np2.zeros(n),_np2.ones(n_samples-n)]).astype(int)

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),max_iter=200,random_state=None,
                 alpha=0.0001,learning_rate_init=0.001,batch_size='auto',**kw):
        self.hidden_layer_sizes=hidden_layer_sizes; self.max_iter=max_iter
        self.random_state=random_state; self.alpha=alpha
        self.lr=learning_rate_init; self.batch_size=batch_size
    def fit(self,X,y):
        rng=_np2.random.default_rng(self.random_state)
        self._classes=_np2.unique(y)
        dims=[X.shape[1]]+list(self.hidden_layer_sizes)+[1]
        self.W=[rng.normal(0,_np2.sqrt(2/dims[i]),(dims[i],dims[i+1]))
                for i in range(len(dims)-1)]
        self.b=[_np2.zeros(dims[i+1]) for i in range(len(dims)-1)]
        n=len(X)
        bs=(min(200,n) if self.batch_size=='auto' else
            (n if int(self.batch_size)>=n else int(self.batch_size)))
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for s in range(0,n,bs):
                xb=X[idx[s:s+bs]]; yb=y[idx[s:s+bs]]; nb=len(xb)
                acts=[xb]
                for i,(W,b) in enumerate(zip(self.W,self.b)):
                    z=acts[-1]@W+b
                    acts.append(_np2.maximum(0,z) if i<len(self.W)-1 else z)
                out=_sig2(acts[-1].ravel())
                yf=_np2.where(yb==self._classes[1],1.0,0.0)
                delta=((out-yf)/nb).reshape(-1,1)
                for i in range(len(self.W)-1,-1,-1):
                    dW=acts[i].T@delta+self.alpha*self.W[i]; db=delta.sum(0)
                    self.W[i]-=self.lr*dW; self.b[i]-=self.lr*db
                    if i>0: delta=(delta@self.W[i].T)*(acts[i]>0)
        return self
    def predict(self,X):
        a=X
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            z=a@W+b; a=_np2.maximum(0,z) if i<len(self.W)-1 else z
        return self._classes[(_sig2(a.ravel())>=0.5).astype(int)]
    def score(self,X,y): return _np2.mean(self.predict(X)==y)


X_m, y_m = make_moons(n_samples=400, noise=0.2, random_state=0)
sc_m = StandardScaler().fit(X_m)
X_ms = sc_m.transform(X_m)
X_tr_m, X_te_m = X_ms[:300], X_ms[300:]
y_tr_m, y_te_m = y_m[:300],  y_m[300:]

print(f"  {'Batch size':>12}  {'Train acc':>11}  {'Test acc':>10}  {'Notes':>20}")
print("  " + "─" * 56)
for bs, note in [(1, "SGD (noise → flat)"),
                 (16, "mini-batch"),
                 (64, "mini-batch"),
                 (300, "full GD (sharp)")]:
    mlp_bs = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500,
                            batch_size=bs, random_state=0, alpha=0.0,
                            learning_rate_init=0.01)
    mlp_bs.fit(X_tr_m, y_tr_m)
    tr_a = mlp_bs.score(X_tr_m, y_tr_m)
    te_a = mlp_bs.score(X_te_m, y_te_m)
    print(f"  {bs:>12}  {tr_a:>11.4f}  {te_a:>10.4f}  {note:>20}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Implicit Bias of Gradient Descent: Minimum Norm and Flat Minima",
             fontsize=13, fontweight="bold")

# Panel (0,0): GD convergence to min-norm
ax = axes[0, 0]
steps_range = np.arange(0, len(losses_gd)) * 100
ax.semilogy(steps_range, losses_gd, "steelblue", lw=2.5, label="Training loss")
ax2 = ax.twinx()
ax2.plot(steps_range, norms_gd, "tomato", lw=2.5, label="‖w‖₂")
ax2.axhline(np.linalg.norm(w_min_norm), color="tomato", lw=1.5, ls="--",
            alpha=0.7, label=f"Min-norm ‖w_MN‖={np.linalg.norm(w_min_norm):.2f}")
ax2.set_ylabel("‖w‖₂", color="tomato")
ax.set_xlabel("Gradient descent steps"); ax.set_ylabel("Training loss (log)")
ax.set_title("GD Converges to Minimum-Norm Solution\\n(d>n: underdetermined system)",
             fontweight="bold")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8); ax.grid(alpha=0.3, which="both")

# Panel (0,1): Solution space (2D analogy)
ax = axes[0, 1]
theta = np.linspace(0, 2*np.pi, 300)
# Show feasible set and minimum-norm solution (2D illustration)
ax.fill_between([-3, 3], [-2, -2], [2, 2], alpha=0.05, color="steelblue")
ax.axhline(0.7, color="steelblue", lw=2, label="Feasible set {w: Xw=y}")
for scale in [0.3, 0.7, 1.2, 1.8]:
    ax.plot(scale*np.cos(theta), scale*np.sin(theta), "gray", lw=1, alpha=0.4)
ax.scatter([0, 0.4, -0.3], [0.7, 0.7, 0.7], c=["tomato","steelblue","steelblue"],
           s=[120,60,60], zorder=10)
ax.annotate("Min-norm\\nw* = (0, 0.7)", xy=(0, 0.7), xytext=(0.5, 0.2),
            arrowprops=dict(arrowstyle="->", color="tomato"), fontsize=9, color="tomato")
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.set_title("Minimum-Norm Solution\\n(GD selects min-norm from feasible set)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect("equal")

# Panel (0,2): Matrix recovery
ax = axes[0, 2]
ax.imshow(M_true, cmap="RdBu", aspect="auto", vmin=-3, vmax=3)
ax2_im = fig.add_axes([0, 0, 0, 0])   # dummy
ax.set_title(f"True Matrix (rank={rank_true})\\n"
             f"Recovered unobs. MSE = {unobs_mse:.4f}",
             fontweight="bold")
ax.set_xlabel("Columns"); ax.set_ylabel("Rows")

# Panel (1,0): Recovered vs true singular values
ax = axes[1, 0]
sv_true = np.linalg.svd(M_true, compute_uv=False)
sv_rec  = np.linalg.svd(M_recovered, compute_uv=False)
x_sv    = np.arange(1, min(n_rows, n_cols)+1)
ax.semilogy(x_sv, sv_true, "tomato",    lw=2.5, marker="o", ms=7, label="True SVs")
ax.semilogy(x_sv[:len(sv_rec)], sv_rec, "steelblue", lw=2.5, marker="s", ms=7,
            label="Recovered SVs")
ax.axvline(rank_true, color="gray", lw=1.5, ls="--", label=f"True rank={rank_true}")
ax.set_xlabel("Singular value index"); ax.set_ylabel("Singular value (log)")
ax.set_title("True vs Recovered Singular Values\\n(GD finds near-minimum nuclear norm)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (1,1): Flat vs sharp minima illustration
ax = axes[1, 1]
w_range = np.linspace(-2, 2, 300)
# Sharp minimum: deep narrow valley
sharp = 10 * (w_range**2) + np.exp(-10*w_range**2)*0.5
flat  = 0.5 * (w_range**2) + 0.3
ax.plot(w_range, sharp, "tomato",    lw=2.5, label="Sharp minimum (large batch)")
ax.plot(w_range, flat,  "steelblue", lw=2.5, label="Flat minimum (small batch)")
ax.axvline(0, color="black", lw=1, ls="--")
ax.annotate("If dist. shifts slightly,\\nflat min stays good", xy=(0.3, 0.4),
            xytext=(0.8, 1.5), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax.set_xlim(-2, 2); ax.set_ylim(-0.1, 3)
ax.set_xlabel("Weight parameter w"); ax.set_ylabel("Loss")
ax.set_title("Flat vs Sharp Minima\\n(flat → better generalisation)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel (1,2): Inductive bias vs data size (underfitting/overfitting zones)
ax = axes[1, 2]
n_range = np.logspace(1, 4, 50)
bias_strong  = 0.25 + 0.5 * np.exp(-n_range/200)     # underfitting floor
bias_weak    = 0.05 + 5.0 / n_range                    # overfitting, needs data
bias_correct = 0.05 + 0.3 * np.exp(-n_range/50)        # quick convergence
ax.semilogx(n_range, bias_strong,  "tomato",    lw=2.5, label="Too-strong bias (underfits)")
ax.semilogx(n_range, bias_weak,    "steelblue", lw=2.5, label="Too-weak bias (overfits)")
ax.semilogx(n_range, bias_correct, "seagreen",  lw=2.5, label="Correct bias (ideal)")
ax.set_xlabel("Training data size n (log)")
ax.set_ylabel("Test error")
ax.set_title("Inductive Bias vs Data Size\\n(correct bias = fast convergence, low error)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("implicit_bias_gd.png", dpi=110)
print()
print("  Plot saved → implicit_bias_gd.png")
print()
print("  KEY TAKEAWAYS — INDUCTIVE BIAS:")
print("  1. Every model assumes structure; wrong bias = unfixable underfitting.")
print("  2. CNNs: locality + translation equivariance → sample-efficient on images.")
print("  3. GD from zero init finds minimum-norm interpolating solution (linear models).")
print("  4. Matrix factorisation GD finds minimum nuclear norm (implicit low-rank).")
print("  5. Small batch size SGD prefers flat minima → better generalisation.")
print("  6. Match inductive bias to known problem symmetries for efficiency.")
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