"""
Data Preprocessing
==================

Normalisation, standardisation, encoding, imputation, outlier handling,
and feature engineering — the essential transformations that turn raw data
into model-ready tensors, and why each choice matters for learning.

"""

import textwrap
import re

TOPIC_NAME = "Data Preprocessing"
DISPLAY_NAME = "10 · Data Preprocessing"
ICON = "🧹"
SUBTITLE = "Garbage In, Garbage Out — The Data Determines the Ceiling"

THEORY = """

### Why Preprocessing Matters

Raw data is rarely ready for ML models. Models have assumptions about
the form of their input: gradient-based models need numerically stable
activations; distance-based models need features on comparable scales;
tree models need numeric encodings of categorical variables.

    ┌──────────────────────────────────────────────────────────────────┐
    │  RULE: preprocessing decisions can change performance by         │
    │  10-30% — often more than model architecture choices.            │
    │                                                                  │
    │  The preprocessing pipeline must be:                             │
    │  ─ FIT on training data only.                                    │
    │  ─ APPLIED to train, validation, and test sets.                  │
    │  ─ Part of the model pipeline for deployment.                    │
    │                                                                  │
    │  DATA LEAKAGE: fitting preprocessing on test data contaminates   │
    │  the evaluation — the model sees test distribution at train time.│
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Feature Scaling

Scaling affects gradient-based models (linear, neural nets) and
distance-based models (k-NN, SVM, k-means) significantly.
Tree-based models are scale-invariant (splits are rank-based).

**Standardisation (Z-score normalisation):**

    x' = (x − μ) / σ     where μ = mean, σ = std of training data.

    ─ Output: mean 0, std 1 (but NOT bounded).
    ─ Preserves outliers — just shifts/scales them.
    ─ Best for: Gaussian or near-Gaussian features, neural networks,
      SVM, PCA (requires zero mean for correct eigenvectors).

**Min-Max Normalisation:**

    x' = (x − min) / (max − min)   → output in [0, 1]

    ─ Bounded output — important for sigmoid/tanh activations.
    ─ Sensitive to outliers (one outlier squashes all other values).
    ─ Best for: bounded inputs, image pixel values (÷255).

**Robust Scaling:**

    x' = (x − median) / IQR   where IQR = Q75 − Q25

    ─ Robust to outliers — median and IQR are not affected by extremes.
    ─ Best for: data with outliers that should not be removed.

**Max-Abs Scaling:**

    x' = x / max|x|   → output in [-1, 1]

    ─ Preserves zero entries — does not shift the data.
    ─ Best for: sparse data (does not destroy sparsity).

**Power Transforms (Box-Cox, Yeo-Johnson):**

    Transforms a skewed distribution toward Gaussian.
    Box-Cox: requires positive data. Yeo-Johnson: handles zeros/negatives.

    Box-Cox: x' = (xᵟ − 1) / δ  if δ ≠ 0;  log(x) if δ = 0.
    δ is chosen to maximise normality (MLE or grid search).

    ─ Best for: right-skewed features (incomes, durations, counts).
    ─ Makes linear model assumptions (Gaussian errors) more valid.

    Diagram 1 — Effect of Scaling on Gradient Descent:

    UNSCALED (features on different scales):
    Loss surface: elongated ellipse → slow zig-zagging convergence.

    STANDARDISED:
    Loss surface: more circular → fast direct convergence.

    ┌────────────────┐      ┌────────────────┐
    │  /----\\       │      │   (○○○)        │
    │ /      \\      │  →   │   (○○○)        │  Standardised:
    │ \\      /      │      │   (○○○)        │  near-circular
    │  \\----/       │      └────────────────┘
    │ Elongated      │
    └────────────────┘
     Gradient zig-zags       Fast convergence


──────────────────────────────────────────────────────────────────────────────
### Encoding Categorical Variables

Neural nets and linear models require numeric inputs.
Categorical variables must be converted.

**Ordinal encoding:**

    Maps categories to integers: {low, medium, high} → {0, 1, 2}.
    Implies an ordering and equal spacing between categories.
    Use only when the ordering is genuine and roughly equidistant.

**One-Hot Encoding (OHE):**

    Creates a binary indicator column per category.
    {red, green, blue} → [1,0,0], [0,1,0], [0,0,1].

    ─ Correct for nominal categories (no natural ordering).
    ─ Creates multicollinearity: drop one column for linear models.
    ─ High cardinality (10K categories): 10K new features — sparse, expensive.

**Target Encoding:**

    Replace category c with E[Y | X = c] (mean of target for that category).
    ─ Reduces cardinality to 1 column regardless of n_categories.
    ─ RISK: target leakage if computed on training set directly.
    ─ Use out-of-fold target encoding to prevent leakage.
    ─ Smooth with prior: encoded_c = (n_c · ȳ_c + m · ȳ) / (n_c + m)
      where n_c = count of category c, m = smoothing strength.

**Embedding (for neural networks):**

    Map each category to a dense vector of dimension k.
    Learned during training — captures semantic similarity.
    Rule of thumb: k ≈ min(50, n_categories / 2).
    Used in recommendation systems, NLP (word2vec is an embedding).

    ┌──────────────────────────────────────────────────────────────────┐
    │  WHEN TO USE EACH:                                               │
    │  Ordinal:        ordinal categories, tree models                 │
    │  One-hot:        nominal, few categories (< 15), linear models   │
    │  Target enc.:    high-cardinality, GBM/tree models               │
    │  Hash encoding:  very high cardinality (>1K), feature hashing    │
    │  Embedding:      neural nets, any cardinality                    │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Missing Value Imputation

Real-world data has missing values. How they are handled determines
whether the model learns the right patterns.

    Types of missingness (Rubin's typology):
    ─ MCAR (Missing Completely At Random): missingness is random.
      Simple imputation usually safe.
    ─ MAR (Missing At Random): missingness depends on observed variables.
      More careful imputation needed.
    ─ MNAR (Missing Not At Random): missingness depends on the missing value.
      E.g., high-income individuals don't report income. Most dangerous.

    Imputation methods:

    SIMPLE:
    ─ Mean/median imputation: replace with column mean/median.
      Fast but can distort variance and correlations.
    ─ Mode imputation: for categorical variables.
    ─ Constant fill: fill with 0, "Unknown", or domain-specific value.

    MODEL-BASED:
    ─ k-NN imputation: fill with weighted average of k nearest neighbours.
    ─ Iterative imputation (MICE): fit a model to predict each feature
      from the others, iterate until convergence. Best statistical properties.
    ─ Multiple imputation: create M imputed datasets, train M models,
      pool predictions. Gold standard for statistical inference.

    ┌──────────────────────────────────────────────────────────────────┐
    │  ALWAYS add a binary "was_missing" indicator feature when        │
    │  imputing. The fact that a value was missing can itself be       │
    │  predictive (MNAR case). Throwing away this information is       │
    │  often worse than the imputation method choice.                  │
    └──────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Outlier Detection and Handling

**Univariate methods:**
    ─ Z-score: |z| > 3 → outlier. Sensitive to mean/std itself.
    ─ IQR rule: x < Q25 − 1.5·IQR or x > Q75 + 1.5·IQR → outlier.
    ─ Modified Z-score: uses median and MAD (median absolute deviation).

**Multivariate methods:**
    ─ Mahalanobis distance: accounts for correlations between features.
    ─ Isolation Forest: outliers are isolated with fewer splits.
    ─ Local Outlier Factor (LOF): density-based, finds local anomalies.

**Handling strategies:**
    ─ REMOVE: only if certain the outlier is a measurement error.
    ─ CAP / WINSORISE: clip values at (P5, P95) quantiles.
    ─ LOG TRANSFORM: compresses the range of skewed distributions.
    ─ KEEP: if the outlier is real data. Robust models handle them.
    ─ SEPARATE MODEL: fit one model on normal data, one on outliers.


──────────────────────────────────────────────────────────────────────────────
### Feature Engineering

Beyond cleaning raw features, feature engineering creates new informative
representations.

    ─ POLYNOMIAL FEATURES: x₁², x₁x₂ — captures non-linearities.
      Use with regularisation to avoid explosion.
    ─ LOG TRANSFORMS: log(x), log(1+x) — compresses skewed distributions.
    ─ RATIOS: x₁/x₂ — common in finance (P/E ratio), biology.
    ─ BINNING / QUANTILE TRANSFORM: converts continuous to ordinal.
    ─ DATE/TIME DECOMPOSITION: extract hour, day-of-week, month, is_holiday.
    ─ INTERACTION FEATURES: x₁ × x₂ — explicitly captures interactions.
    ─ ROLLING STATISTICS: rolling mean, std, max over a time window.
    ─ AGGREGATE FEATURES: user-level stats, product-level stats.

    Feature selection methods (after engineering):
    ─ Variance threshold: remove near-constant features.
    ─ Correlation filter: remove one of two correlated features (> 0.95).
    ─ Univariate tests: χ², ANOVA F-test, mutual information.
    ─ RFECV: Recursive Feature Elimination with Cross-Validation.
    ─ SHAP-based: remove features with near-zero mean |SHAP|.


──────────────────────────────────────────────────────────────────────────────
### The Preprocessing Pipeline: Order Matters

    ┌──────────────────────────────────────────────────────────────────┐
    │  RECOMMENDED ORDER:                                              │
    │                                                                  │
    │  1. Duplicates removal                                           │
    │  2. Type casting (numeric, categorical, datetime)                │
    │  3. Outlier detection / capping                                  │
    │  4. Missing value imputation                                     │
    │  5. Feature engineering (ratios, log transforms, interactions)   │
    │  6. Categorical encoding (OHE, target encoding, ordinal)         │
    │  7. Feature scaling (standardise/normalise)                      │
    │  8. Feature selection (variance, correlation, importance)        │
    │  9. Dimensionality reduction (optional: PCA, t-SNE for viz)      │
    │                                                                  │
    │  FIT steps 3-9 on TRAINING DATA ONLY.                            │
    │  APPLY to validation and test sets using fitted parameters.      │
    └──────────────────────────────────────────────────────────────────┘

"""

OPERATIONS = {
    "1 · Scaling, Encoding, and Imputation — Effects on Model Performance": {
        "description": (
            "Demonstrates the effect of different scaling strategies on a "
            "logistic regression and SVM classifier: no scaling vs standard "
            "vs min-max vs robust scaling. Shows how scaling accelerates "
            "gradient descent convergence. Implements and compares three "
            "categorical encoding strategies: one-hot, ordinal, and target "
            "encoding. Demonstrates missing value imputation strategies "
            "and their effect on model accuracy."
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
from scipy.stats import yeojohnson as _yj

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class MinMaxScaler:
    def fit(self,X,y=None):
        self.min_=X.min(0); self.scale_=X.max(0)-X.min(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.min_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class RobustScaler:
    def fit(self,X,y=None):
        self.center_=_np_impl.median(X,0)
        self.scale_=_np_impl.percentile(X,75,0)-_np_impl.percentile(X,25,0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.center_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class PowerTransformer:
    def __init__(self,method="yeo-johnson",**kw): self.method=method
    def fit_transform(self,X,y=None):
        result=_np_impl.zeros_like(X,dtype=float); self._lambdas=[]
        for j in range(X.shape[1]):
            col,lam=_yj(X[:,j]); result[:,j]=col; self._lambdas.append(lam)
        return result

class OneHotEncoder:
    def __init__(self,sparse_output=True,drop=None,**kw): self.drop=drop
    def fit(self,X,y=None):
        self.categories_=[_np_impl.unique(X[:,j]) for j in range(X.shape[1])]; return self
    def transform(self,X):
        parts=[]
        for j,cats in enumerate(self.categories_):
            ohe=(X[:,j:j+1]==cats[None,:]).astype(float)
            if self.drop=="first": ohe=ohe[:,1:]
            parts.append(ohe)
        return _np_impl.hstack(parts)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class SimpleImputer:
    def __init__(self,strategy="mean",fill_value=0,**kw):
        self.strategy=strategy; self.fill_value=fill_value
    def fit(self,X,y=None):
        X=_np_impl.array(X,dtype=float)
        if self.strategy=="mean":       self.stats_=_np_impl.nanmean(X,0)
        elif self.strategy=="median":   self.stats_=_np_impl.nanmedian(X,0)
        elif self.strategy=="constant": self.stats_=_np_impl.full(X.shape[1],float(self.fill_value))
        return self
    def transform(self,X):
        X=_np_impl.array(X,dtype=float); result=X.copy()
        for j in range(X.shape[1]):
            mask=_np_impl.isnan(result[:,j]); result[mask,j]=self.stats_[j]
        return result
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

class KNNImputer:
    def __init__(self,n_neighbors=5,**kw): self.k=n_neighbors
    def fit(self,X,y=None):
        self._X=_np_impl.array(X,dtype=float)
        self._col_means=_np_impl.nanmean(self._X,0); return self
    def transform(self,X):
        X=_np_impl.array(X,dtype=float); result=X.copy(); ref=self._X
        for i in range(len(X)):
            miss=_np_impl.where(_np_impl.isnan(X[i]))[0]
            if len(miss)==0: continue
            present=_np_impl.where(~_np_impl.isnan(X[i]))[0]
            if len(present)==0:
                result[i,miss]=self._col_means[miss]; continue
            dists=_np_impl.full(len(ref),_np_impl.inf)
            for j in range(len(ref)):
                if _np_impl.any(_np_impl.isnan(ref[j,present])): continue
                dists[j]=_np_impl.sqrt(_np_impl.sum((X[i,present]-ref[j,present])**2))
            for col in miss:
                valid=~_np_impl.isnan(ref[:,col])&_np_impl.isfinite(dists)
                if not valid.any(): result[i,col]=self._col_means[col]; continue
                vd=dists.copy(); vd[~valid]=_np_impl.inf
                nn=_np_impl.argsort(vd)[:self.k]; nn=nn[_np_impl.isfinite(vd[nn])]
                result[i,col]=ref[nn,col].mean() if len(nn)>0 else self._col_means[col]
        return result
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

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

class SVC:
    def __init__(self,kernel="rbf",C=1.0,gamma="scale",**kw):
        self.C=C; self.kernel=kernel; self._g=gamma
    def fit(self,X,y):
        self._classes=_np_impl.unique(y); yb=_np_impl.where(y==self._classes[1],1.0,-1.0)
        n,d=X.shape
        g=1.0/(d*X.var()) if self._g=="scale" else float(self._g)
        rng=_np_impl.random.default_rng(0); D=min(300,4*d)
        self._W=rng.normal(0,_np_impl.sqrt(2*g),(d,D))
        self._b_rff=rng.uniform(0,2*_np_impl.pi,D)
        Xp=_np_impl.sqrt(2/D)*_np_impl.cos(X@self._W+self._b_rff)
        lam=1.0/(self.C*n)
        def fg(w):
            m=yb*(Xp@w[1:]+w[0]); hinge=_np_impl.maximum(0,1-m)
            loss=lam*_np_impl.sum(w[1:]**2)+_np_impl.mean(hinge); mask=hinge>0
            gw=2*lam*w[1:]-(_np_impl.mean(yb[mask,None]*Xp[mask],0) if mask.any() else 0)
            gb=-_np_impl.mean(yb[mask]) if mask.any() else 0.0
            return loss,_np_impl.r_[gb,gw]
        res=_sp_opt.minimize(fg,_np_impl.zeros(Xp.shape[1]+1),method='L-BFGS-B',jac=True,
                             options={'maxiter':500})
        self._b0=res.x[0]; self._w=res.x[1:]; return self
    def predict(self,X):
        D=self._W.shape[1]; Xp=_np_impl.sqrt(2/D)*_np_impl.cos(X@self._W+self._b_rff)
        return self._classes[(Xp@self._w+self._b0>=0).astype(int)]
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

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
    def __init__(self,n_estimators=100,max_depth=3,learning_rate=0.1,random_state=None,**kw):
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

class Pipeline:
    def __init__(self,steps): self.steps=list(steps)
    def fit(self,X,y):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.fit(Xt,y).transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def predict(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return self.steps[-1][1].predict(Xt)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==y)

def cross_val_score(estimator,X,y,cv=5,scoring='accuracy'):
    n=len(X); idx=_np_impl.arange(n); fs=n//cv; scores=[]
    for k in range(cv):
        vi=idx[k*fs:(k+1)*fs]; ti=_np_impl.concatenate([idx[:k*fs],idx[(k+1)*fs:]])
        est=_copy.deepcopy(estimator); est.fit(X[ti],y[ti])
        scores.append(_np_impl.mean(est.predict(X[vi])==y[vi]))
    return _np_impl.array(scores)

def train_test_split(*arrays,test_size=0.3,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=len(arrays[0])
    n_tr=int(n*(1-test_size)); idx=rng.permutation(n); ti,vi=idx[:n_tr],idx[n_tr:]
    out=[]
    for a in arrays: out+=[a[ti],a[vi]]
    return out

def make_classification(n_samples=100,n_features=20,n_informative=2,
                        n_redundant=2,n_repeated=0,random_state=None,**kw):
    rng=_np_impl.random.default_rng(random_state)
    y=rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=n_redundant
    Xr=(Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else _np_impl.empty((n_samples,0)))
    nn2=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn2)) if nn2>0 else _np_impl.empty((n_samples,0))
    parts=[a for a in [Xi,Xr,Xn] if a.shape[1]>0]
    return _np_impl.hstack(parts)[:,:n_features],y.astype(int)


np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Effect of scaling on model performance
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  FEATURE SCALING: EFFECT ON MODEL PERFORMANCE")
print("=" * 65)
print()

# Dataset with deliberately different feature scales
n, d = 200, 6
X_raw, y = make_classification(n_samples=n, n_features=d, n_informative=4, random_state=0)
# Artificially create scale differences
X_raw[:, 0] *= 1000    # feature 0: scale 1000×
X_raw[:, 1] *= 0.001   # feature 1: scale 0.001×
X_raw[:, 2] += 500     # feature 2: large mean
X_raw[:, 3] = np.abs(X_raw[:, 3])  # feature 3: make right-skewed

scalers = {
    "None (raw)":       None,
    "Standard scaler":  StandardScaler(),
    "Min-Max":          MinMaxScaler(),
    "Robust scaler":    RobustScaler(),
}

models = {
    "Logistic Reg.": LogisticRegression(max_iter=1000, C=1.0, random_state=0),
    "SVM (RBF)":     SVC(kernel="rbf", C=1.0, gamma="scale"),
}

print(f"  {'Scaler':>20}  {'LogReg CV':>12}  {'SVM CV':>10}")
print("  " + "─" * 46)
scaling_results = {}
for sname, scaler in scalers.items():
    row = {}
    for mname, model in models.items():
        if scaler is not None:
            pipe = Pipeline([("scaler", scaler), ("model", model)])
        else:
            pipe = Pipeline([("model", model)])
        cv_acc = cross_val_score(pipe, X_raw, y, cv=3, scoring="accuracy").mean()
        row[mname] = cv_acc
    scaling_results[sname] = row
    print(f"  {sname:>20}  {row['Logistic Reg.']:>12.4f}  {row['SVM (RBF)']:>10.4f}")

print()
print("  KEY: SVM (RBF) uses distances — sensitive to scale.")
print("  Logistic regression: gradient convergence depends on scale.")
print("  StandardScaler often best for gradient-based and distance models.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SGD convergence with and without standardisation
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  GRADIENT DESCENT CONVERGENCE: SCALED vs UNSCALED")
print("=" * 65)
print()

X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.3, random_state=0)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

losses_unscaled, losses_scaled = [], []
w_un = np.zeros(d); w_sc = np.zeros(d)

from scipy.special import expit
def log_loss_val(w, Xv, yv):
    p = expit(Xv @ w)
    return -np.mean(yv*np.log(p+1e-15) + (1-yv)*np.log(1-p+1e-15))

for ep in range(200):
    for X_ep, y_ep, w_ref, loss_list in [
        (X_tr, y_tr, None, losses_unscaled),
        (X_tr_s, y_tr, None, losses_scaled)
    ]:
        if loss_list is losses_unscaled:
            w = w_un
        else:
            w = w_sc
        p  = expit(X_ep @ w)
        g  = X_ep.T @ (p - y_ep) / len(y_ep)
        if loss_list is losses_unscaled:
            w_un -= 0.00001 * g   # tiny lr needed for unscaled
            losses_unscaled.append(log_loss_val(w_un, X_ep, y_ep))
        else:
            w_sc -= 0.1 * g       # standard lr for scaled
            losses_scaled.append(log_loss_val(w_sc, X_ep, y_ep))

print(f"  After 200 epochs:")
print(f"  Unscaled GD final loss:  {losses_unscaled[-1]:.6f}  "
      f"(lr=0.00001 — needed tiny lr to avoid divergence)")
print(f"  Standardised GD final:   {losses_scaled[-1]:.6f}  (lr=0.1)")
print(f"  Faster convergence: standardised reaches {losses_unscaled[-1]:.4f} "
      f"in {np.searchsorted(np.array(losses_scaled)[::-1], losses_unscaled[-1]):d} epochs")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Categorical encoding comparison
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  CATEGORICAL ENCODING: OHE vs ORDINAL vs TARGET ENCODING")
print("=" * 65)
print()

# Dataset with categorical features
rng = np.random.default_rng(0)
n_s = 200
# Feature: "city" with 5 categories, strong signal
city_probs   = {"A": 0.7, "B": 0.4, "C": 0.6, "D": 0.3, "E": 0.55}
cities       = rng.choice(list(city_probs.keys()), n_s)
y_cat        = (rng.random(n_s) < np.array([city_probs[c] for c in cities])).astype(int)
X_num        = rng.standard_normal((n_s, 3))   # additional numeric features

# One-hot encoding
ohe = OneHotEncoder(sparse_output=False, drop="first")
X_ohe   = np.hstack([X_num, ohe.fit_transform(cities.reshape(-1,1))])

# Ordinal encoding
city_ordinal = {c: i for i, c in enumerate(sorted(city_probs.keys()))}
X_ord   = np.hstack([X_num, np.array([city_ordinal[c] for c in cities]).reshape(-1,1)])

# Target encoding with leave-one-out (prevent leakage)
def target_encode_loo(X_cat, y, smoothing=10):
    global_mean = y.mean()
    encoded = np.zeros(len(X_cat))
    for i, cat in enumerate(X_cat):
        mask     = (X_cat == cat)
        mask[i]  = False  # leave one out
        n_cat    = mask.sum()
        mean_cat = y[mask].mean() if n_cat > 0 else global_mean
        # Smoothed estimate
        encoded[i] = (n_cat * mean_cat + smoothing * global_mean) / (n_cat + smoothing)
    return encoded

X_tgt   = np.hstack([X_num, target_encode_loo(cities, y_cat).reshape(-1,1)])

print(f"  {'Encoding':>20}  {'n_features':>12}  {'GBM CV acc':>12}")
print("  " + "─" * 48)
for enc_name, X_enc in [("One-Hot (drop 1st)", X_ohe),
                         ("Ordinal",            X_ord),
                         ("Target (LOO-smooth)",X_tgt)]:
    gbm = GradientBoostingClassifier(n_estimators=15, random_state=0)
    cv  = cross_val_score(gbm, X_enc, y_cat, cv=3, scoring="accuracy")
    print(f"  {enc_name:>20}  {X_enc.shape[1]:>12}  {cv.mean():>12.4f} ± {cv.std():.4f}")

print()
print("  Target encoding: reduces to 1 column, captures category signal directly.")
print("  OHE: safe but creates many columns for high-cardinality features.")
print("  Ordinal: only valid if true ordering exists between categories.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Missing value imputation
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  MISSING VALUE IMPUTATION: STRATEGIES COMPARED")
print("=" * 65)
print()

# Create dataset with simulated missing values
X_complete, y_imp = make_classification(n_samples=150, n_features=8,
                                         n_informative=5, random_state=1)

missing_fracs = [0.05, 0.15, 0.30]
imputation_methods = {
    "Mean imputation":     SimpleImputer(strategy="mean"),
    "Median imputation":   SimpleImputer(strategy="median"),
    "k-NN imputation":     KNNImputer(n_neighbors=5),
    "Constant (0)":        SimpleImputer(strategy="constant", fill_value=0),
}

print(f"  {'Imputer':>25}  " + "  ".join(f"miss={f:.0%}" for f in missing_fracs))
print("  " + "─" * 60)

imputation_results = {}
for imp_name, imputer in imputation_methods.items():
    row = f"  {imp_name:>25}  "
    imputation_results[imp_name] = {}
    for frac in missing_fracs:
        rng_miss = np.random.default_rng(42)
        X_miss   = X_complete.copy().astype(float)
        mask_miss = rng_miss.random(X_miss.shape) < frac
        X_miss[mask_miss] = np.nan
        pipe = Pipeline([("imputer", imputer),
                         ("scaler",  StandardScaler()),
                         ("model",   LogisticRegression(max_iter=300, C=1.0))])
        cv = cross_val_score(pipe, X_miss, y_imp, cv=3, scoring="accuracy")
        imputation_results[imp_name][frac] = cv.mean()
        row += f"  {cv.mean():.4f} "
    print(row)

print()
print("  k-NN imputation typically best (uses feature correlations).")
print("  Mean/median: fast, reasonable for MCAR. Add missingness indicator!")
print("  Constant: can introduce spurious patterns, use carefully.")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Data Preprocessing: Scaling, Encoding, and Imputation",
             fontsize=13, fontweight="bold")

# Panel (0,0): Feature distributions before/after scaling
ax = axes[0, 0]
feat_idx = 0
vals_raw = X_raw[:, feat_idx]
vals_std = StandardScaler().fit_transform(vals_raw.reshape(-1,1)).ravel()
vals_mm  = MinMaxScaler().fit_transform(vals_raw.reshape(-1,1)).ravel()
vals_rob = RobustScaler().fit_transform(vals_raw.reshape(-1,1)).ravel()
for vals, label, col in [(vals_raw/vals_raw.std(), "Raw (std-norm)", "gray"),
                          (vals_std, "StandardScaler", "steelblue"),
                          (vals_mm,  "MinMaxScaler",   "tomato"),
                          (vals_rob, "RobustScaler",   "seagreen")]:
    ax.hist(vals, bins=30, alpha=0.5, density=True, label=label)
ax.set_title("Feature Distributions After Scaling\\n(Feature 0, artificially scaled 1000×)",
             fontweight="bold")
ax.set_xlabel("Feature value"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (0,1): Scaling effect on model accuracy
ax = axes[0, 1]
scaler_names = list(scaling_results.keys())
log_accs = [scaling_results[s]["Logistic Reg."] for s in scaler_names]
svm_accs = [scaling_results[s]["SVM (RBF)"]     for s in scaler_names]
x_pos = np.arange(len(scaler_names))
ax.bar(x_pos - 0.2, log_accs, 0.4, color="steelblue", alpha=0.8, label="Logistic Reg.")
ax.bar(x_pos + 0.2, svm_accs, 0.4, color="tomato",    alpha=0.8, label="SVM (RBF)")
ax.set_xticks(x_pos)
ax.set_xticklabels([s.replace(" ", "\\n") for s in scaler_names], fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.set_title("Scaling vs CV Accuracy\\n(SVM very sensitive to scale)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# Panel (0,2): SGD convergence curves
ax = axes[0, 2]
ax.semilogy(range(len(losses_unscaled)), losses_unscaled, "tomato", lw=2,
            label="Unscaled (lr=0.00001)")
ax.semilogy(range(len(losses_scaled)),   losses_scaled,   "steelblue", lw=2,
            label="StandardScaler (lr=0.1)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Log-loss (log)")
ax.set_title("Gradient Descent Convergence\\n(scaled data converges 100× faster)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel (1,0): Power transform — before/after
ax = axes[1, 0]
# Right-skewed feature
X_skew = np.random.exponential(2, 500)
X_log  = np.log1p(X_skew)
pt = PowerTransformer(method="yeo-johnson")
X_pt = pt.fit_transform(X_skew.reshape(-1,1)).ravel()
from scipy.stats import norm as snorm
ax.hist(X_skew, bins=40, alpha=0.5, density=True, color="tomato", label="Raw (skewed)")
ax.hist(X_log,  bins=40, alpha=0.5, density=True, color="steelblue", label="log(1+x)")
ax.hist(X_pt,   bins=40, alpha=0.5, density=True, color="seagreen", label="Yeo-Johnson")
x_range = np.linspace(-3, 3, 100)
ax.plot(x_range, snorm.pdf(x_range), "black", lw=2, ls="--", label="Gaussian ref")
ax.set_title("Power Transform for Skewed Features\\n(power transform → near-Gaussian)",
             fontweight="bold")
ax.set_xlabel("Feature value"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel (1,1): Target encoding visualisation
ax = axes[1, 1]
city_list  = sorted(city_probs.keys())
city_means = [city_probs[c] for c in city_list]
city_counts = [np.sum(cities == c) for c in city_list]
# Smoothed target encoding
sm = 10
global_m = y_cat.mean()
tgt_enc_vals = [(cnt * city_probs[c] + sm * global_m)/(cnt + sm)
                for c, cnt in zip(city_list, city_counts)]
x_pos2 = np.arange(len(city_list))
ax.bar(x_pos2 - 0.2, city_means,   0.4, color="tomato",    alpha=0.8, label="True P(Y=1|city)")
ax.bar(x_pos2 + 0.2, tgt_enc_vals, 0.4, color="steelblue", alpha=0.8, label="Target encoding (smooth)")
ax.axhline(global_m, color="gray", lw=1.5, ls="--", label=f"Global mean={global_m:.2f}")
ax.set_xticks(x_pos2); ax.set_xticklabels(city_list)
ax.set_title("Target Encoding vs True Signal\\n(smoothing shrinks toward global mean)",
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# Panel (1,2): Imputation comparison heatmap
ax = axes[1, 2]
imp_names = list(imputation_methods.keys())
imp_accs  = np.zeros((len(imp_names), len(missing_fracs)))
for i, imp_name in enumerate(imp_names):
    for j, frac in enumerate(missing_fracs):
        imp_accs[i, j] = imputation_results[imp_name][frac]

im2 = ax.imshow(imp_accs, cmap="RdYlGn", vmin=0.6, vmax=1.0, aspect="auto")
plt.colorbar(im2, ax=ax, label="CV accuracy")
ax.set_xticks(range(len(missing_fracs)))
ax.set_xticklabels([f"{f:.0%}" for f in missing_fracs])
ax.set_yticks(range(len(imp_names)))
ax.set_yticklabels([n.replace(" imputation","") for n in imp_names], fontsize=9)
for i in range(len(imp_names)):
    for j in range(len(missing_fracs)):
        ax.text(j, i, f"{imp_accs[i,j]:.3f}", ha="center", va="center", fontsize=9)
ax.set_title("Imputation Strategy vs Accuracy\\n(rows=imputer, cols=missing fraction)",
             fontweight="bold")

plt.tight_layout()
plt.savefig("preprocessing_effects.png", dpi=110)
print()
print("  Plot saved → preprocessing_effects.png")
print()
print("  KEY TAKEAWAYS — DATA PREPROCESSING:")
print("  1. Fit preprocessing on train only — never on test/validation.")
print("  2. StandardScaler: best for gradient-based models and distance models.")
print("  3. Tree models (GBM, RF) are scale-invariant; scaling doesn't help.")
print("  4. One-hot encoding: safe for low cardinality; target enc. for high.")
print("  5. Add missing indicator features alongside imputed values.")
print("  6. k-NN imputation uses feature correlations — better than mean/median.")
print("  7. Power transforms (log, Yeo-Johnson) correct skewed distributions.")
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