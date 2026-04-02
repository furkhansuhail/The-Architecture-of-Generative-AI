"""
Data Augmentation & Feature Engineering
=========================================

Two complementary strategies for beating overfitting and underfitting at
the data level — before touching a single model hyperparameter.

Overfitting fix  → augmentation grows the effective dataset size
Underfitting fix → feature engineering gives the model richer signals

"""

import textwrap
import re

TOPIC_NAME   = "Data Augmentation & Feature Engineering"
DISPLAY_NAME = "02 · Data Augmentation & Feature Engineering"
ICON         = "🔬"
SUBTITLE     = "Beating Overfitting & Underfitting at the Data Level"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """


### PART 1 — DATA AUGMENTATION  (fix for overfitting)

### Why Data Augmentation Works

Overfitting occurs when a model has more parameters than the data can
constrain — it memorises the training set instead of learning general
patterns. The most direct cure is more data. Data augmentation generates
new, realistic training examples by applying label-preserving
transformations to existing ones.

    Original Dataset:   N real examples
    Augmented Dataset:  N × k virtual examples  (k = augmentations per sample)

The key requirement: a transformation is valid if and only if it does NOT
change the true label. A horizontally flipped cat photo is still a cat.
A sentence with synonyms replaced still carries the same meaning.

    Without augmentation                 With augmentation
    ┌──────────────────┐                ┌──────────────────────────────────┐
    │ 1 000 cat photos │                │ 1 cat photo → flip → rotate →    │
    │ model memorises  │                │ crop → colour shift → ...        │
    │ exact pixel vals │                │ = effectively 8 000+ examples    │
    └──────────────────┘                └──────────────────────────────────┘
    Large val/train gap                  Gap narrows, generalisation improves


──────────────────────────────────────────────────────────────────────────────
### Image Augmentation

The most mature augmentation domain. Every technique below is a
label-preserving transformation of pixel values.

    Diagram 1 — Common Image Augmentation Pipeline:

    ORIGINAL IMAGE  →  [Horizontal Flip]  →  [Random Crop]  →
    [Colour Jitter]  →  [Random Rotation]  →  [Gaussian Blur]  →
    [CutOut / Erasing]  →  AUGMENTED SAMPLE (fed to model)

**Geometric transforms** — move or reshape pixels, labels unchanged:
    • Horizontal / vertical flip
    • Random crop  (take a random sub-region → forces position invariance)
    • Random rotation ±θ degrees
    • Affine transforms (shear, scale, translate)
    • Perspective warp

**Colour / pixel-value transforms** — change appearance, not structure:
    • Brightness / contrast / saturation jitter
    • Hue shift
    • Gaussian / salt-and-pepper noise
    • Gaussian blur / sharpening

**Occlusion / mixing transforms** — advanced, very effective:
    • CutOut: zero out a random rectangular patch → forces the model to
      use multiple cues rather than relying on one discriminative region.
    • CutMix: replace a patch with the corresponding patch from another
      image; label becomes a weighted mix (e.g., 70% cat / 30% dog).
    • MixUp: linear blend of two images AND their labels at the pixel level.
      x_new = λ·x₁ + (1-λ)·x₂,  y_new = λ·y₁ + (1-λ)·y₂
      Encourages the model to learn smoother decision boundaries.

    Diagram 2 — MixUp vs CutMix:

    MixUp (entire image blended):          CutMix (region replaced):
    ┌──────────────────────────┐          ┌─────────────────────────┐
    │░░░░░░░░░░░░░░░░░░░░░░░░░░│          │▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    │░░[cat + dog blended]░░░░░│          │▓▓▓[cat]▓▓▒▒▒[dog]▒▒▒▒▒▒▒│
    │░░░░░░░░░░░░░░░░░░░░░░░░░░│          │▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
    └──────────────────────────┘          └─────────────────────────┘
    Label: 0.7·cat + 0.3·dog              Label: 0.6·cat + 0.4·dog
    (λ=0.7)                               (based on patch area ratio)


──────────────────────────────────────────────────────────────────────────────
### Text Augmentation

Text augmentation is harder because small changes can flip the meaning.

**Lexical substitution** — replace words with synonyms (WordNet, word
    embeddings). Example: "The movie was great" → "The film was fantastic"
    Label: positive sentiment — unchanged.

**Back-translation** — translate to French, then back to English.
    The paraphrase is grammatically fresh but semantically equivalent.
    Example: "I feel amazing" → (French) "Je me sens incroyable"
             → (back to English) "I feel incredible"

**EDA (Easy Data Augmentation)** — Wei & Zou 2019 — four cheap operations:
    • Synonym Replacement (SR): replace n random non-stop-words with synonyms
    • Random Insertion (RI): insert a random synonym of a random word
    • Random Swap (RS): swap two random words k times
    • Random Deletion (RD): delete each word with probability p

**Contextual augmentation** — use a masked language model (e.g., BERT) to
    generate contextually appropriate replacements for masked tokens.
    More powerful than simple synonym replacement.

**Token noise** — randomly insert, delete, or swap characters.
    Useful for making models robust to typos.


──────────────────────────────────────────────────────────────────────────────
### Tabular Data Augmentation

Tabular data is the most common in industry but the hardest to augment —
there is no obvious notion of "flip" or "crop" for a row of features.

**Gaussian noise injection** — add small Gaussian noise to continuous
    features. Effective if the noise scale is much smaller than the
    feature's natural variance. Equivalent to Tikhonov regularisation.

**SMOTE (Synthetic Minority Over-sampling Technique)** — for class
    imbalance, generate synthetic minority samples by interpolating
    between real minority examples in feature space:

    Diagram 3 — SMOTE Interpolation:

    Feature space (2D projection):

         x₂  ↑
             |     ● ← minority class (rare)
             |   ●
             |     ✦  ← synthetic point = ● + λ·(●-●), λ ∈ [0,1]
             |   ●
             | ○ ○ ○ ○ ← majority class
             └─────────────────────── x₁

    For each minority point, find its k nearest minority neighbours.
    Synthesise new points along the line between them.
    Result: better-balanced class distribution without real new data.

**Feature dropout** — randomly zero out tabular features during training
    (analogous to image CutOut). Forces the model to learn robust
    representations that don't rely on any single feature being present.

**Gaussian copula synthesis** — sample new rows from a learned multivariate
    distribution that preserves the joint dependencies between features.
    Used in frameworks like SDV (Synthetic Data Vault).


──────────────────────────────────────────────────────────────────────────────
### When Augmentation Hurts

Not all augmentations are label-preserving. Common mistakes:

    ┌───────────────────────────────────────────────────────────────┐
    │ Task                  │ INVALID augmentation (label changes)  │
    ├───────────────────────┼───────────────────────────────────────┤
    │ Medical X-ray         │ Vertical flip (anatomy is asymmetric) │
    │ Digit recognition     │ 180° rotation (6 becomes 9)           │
    │ Sentiment analysis    │ Antonym replacement ("good"→"bad")    │
    │ Time-series (ordered) │ Random temporal shuffling             │
    └───────────────────────┴───────────────────────────────────────┘

    Always ask: "Does this transformation preserve the true label?"
    If not, it is harmful, not helpful.



### PART 2 — FEATURE ENGINEERING  (fix for underfitting)

### What is Feature Engineering?

Feature engineering is the process of using domain knowledge to create,
transform, or select input features that make the true pattern in the data
more accessible to the model. It is the primary cure for underfitting when
you don't want to increase model complexity.

    The fundamental insight:
    A complex model on poor features ≪ A simple model on good features

    Raw data often hides the signal:
    ┌─────────────────────────────────────────────────────────────┐
    │ Raw feature    │ Engineered feature   │ Why better          │
    ├─────────────────────────────────────────────────────────────┤
    │ timestamp      │ hour, day_of_week,   │ cyclical patterns   │
    │                │ is_weekend, month    │ now linearly clear  │
    │ lat, lon       │ distance_to_centre   │ collapses 2D to 1D  │
    │                │ geohash bucket       │ relevant dimension  │
    │ price          │ log(price)           │ skewed → normal     │
    │ birth_date     │ age                  │ model sees meaning  │
    │ text           │ tf-idf, embeddings   │ semantic signal     │
    └─────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Types of Feature Engineering

**1. Mathematical transforms** — fix distributional issues:
    • log(x): compress right-skewed distributions (prices, counts)
    • √x: moderate compression
    • x²: emphasise large values, capture quadratic relationships
    • 1/x: capture inverse relationships
    • Box-Cox / Yeo-Johnson: general power transform that finds the best
      exponent to make a feature approximately normal

**2. Interaction features** — capture joint effects:
    • x₁ × x₂: multiplicative interaction (e.g., height × weight → BMI)
    • x₁ / x₂: ratio (e.g., debt / income → debt-to-income ratio)
    • x₁ - x₂: difference (e.g., arrival_time - departure_time → duration)

**3. Polynomial features** — let a linear model learn non-linear patterns:
    • Degree-2 expansion: {x₁, x₂} → {x₁, x₂, x₁², x₂², x₁x₂}
    • This is how you can give a linear model the power to fit a parabola
      without switching to a neural network.

    Diagram 4 — Polynomial Features Fix Underfitting:

    Data has a curved boundary:
    ○ ○ ○  ● ● ●
    ○ ○ ○  ● ● ●
    ○ ●●●  ○ ○ ●         ← can't separate with a line

    Raw features: [x₁, x₂]   → linear model fails (underfits)

    With x₁², x₂², x₁x₂ added:
    The same linear model can now fit a quadratic boundary → success.

**4. Binning / discretisation** — turn continuous into categorical:
    • Equal-width: divide range into k bins of equal size
    • Equal-frequency (quantile): each bin has equal number of samples
    • Useful when the relationship is step-like, not smooth

**5. Encoding categorical features** — make categories numeric:
    • One-hot encoding: {red, blue, green} → [1,0,0], [0,1,0], [0,0,1]
      Warning: high cardinality (1000s of categories) → use embeddings
    • Label encoding: red=0, blue=1, green=2 (only for ordinal features)
    • Target encoding: replace category with mean of target in that group
      (powerful but requires careful cross-validation to avoid leakage)
    • Embeddings: learn a dense vector per category (used in deep learning)

**6. Date / time features**:
    • Extract: hour, day_of_week, week_of_year, month, quarter, year
    • Cyclical encoding for periodic features (hour 23 ≈ hour 0):
        sin(2π × hour / 24)   and   cos(2π × hour / 24)
      → represents cyclical time as a point on the unit circle
    • Lags: feature at time t-1, t-7, t-30 (crucial for time-series)
    • Rolling statistics: 7-day mean, 30-day std (capture trends)

    Diagram 5 — Cyclical Encoding of Hour-of-Day:

    Naive encoding:  hour=0, ..., hour=23   (0 and 23 appear far apart)
                     BUT 11pm and midnight ARE nearly identical in behaviour.

    Cyclical:        hour → (sin(2π·hour/24), cos(2π·hour/24))

                    cos
                     ↑
                     │  0h = (0, 1)          6h = (1, 0)
                     │        ★──────────────────────●
                     │       ╱                        │
                 12h ●──────────────────────────── ───│
                     │   ╲                        │
                     │    ★───────────────────────●
                     │ 18h = (-1, 0)       ≈ (0,-1) = 12h? no, 12h
                     └───────────────────────────────── sin

                    Points close in time are close in this 2D space.
                    Hour 23 and hour 0 are neighbours. ✓


──────────────────────────────────────────────────────────────────────────────
### Feature Selection  (fix for overfitting)

When you have too many features, irrelevant or redundant ones introduce
noise that causes overfitting. Feature selection reduces the input
dimensionality, keeping only the most informative features.

Three families of methods:

**Filter methods** — score features independently of any model:
    • Pearson correlation with the target (for continuous features)
    • Mutual information (captures non-linear relationships)
    • Chi-squared test (for categorical features)
    • ANOVA F-test (for continuous features with categorical target)
    Pro: very fast. Con: ignores feature interactions.

**Wrapper methods** — use model performance to evaluate feature subsets:
    • Recursive Feature Elimination (RFE): train model, remove the weakest
      feature, retrain, repeat until k features remain.
    • Forward selection: start with no features, greedily add the best one.
    • Backward elimination: start with all features, remove the worst one.
    Pro: captures interactions. Con: computationally expensive.

**Embedded methods** — feature selection happens during model training:
    • L1 regularisation (Lasso): weights of irrelevant features → 0
    • Tree feature importance: random forest ranks features by how much
      they reduce impurity across all trees.
    • Gradient boosting feature importance
    Pro: efficient, integrated with training. Con: model-specific.

    Comparison:
    ┌──────────────┬────────────────┬──────────────┬──────────────────┐
    │ Method       │ Speed          │ Interactions │ Model-agnostic?  │
    ├──────────────┼────────────────┼──────────────┼──────────────────┤
    │ Filter       │ Very fast      │ No           │ Yes              │
    │ Wrapper (RFE)│ Slow           │ Yes          │ Yes              │
    │ Embedded (L1)│ Fast (1 fit)   │ Partial      │ No (model-tied)  │
    └──────────────┴────────────────┴──────────────┴──────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Dimensionality Reduction

When features are highly correlated, dimensionality reduction compresses
them into fewer, uncorrelated components — reducing overfitting while
preserving the most informative signal.

**PCA (Principal Component Analysis)** — projects data onto the directions
    of maximum variance. Each principal component is a linear combination
    of original features, orthogonal (uncorrelated) to all others.

    Diagram 6 — PCA Finds the Directions of Maximum Variance:

    Original 2D data (highly correlated features x₁, x₂):

         x₂ ↑
            │      ● ●
            │    ● ● ●                  PC1 ↗ (direction of max variance)
            │  ● ● ● ●           ╱────────────────────────────────────────
            │ ● ● ● ●         ╱ ↙ PC2 (perpendicular to PC1, less variance)
            │● ● ●           ╱
            └────────────────────────── x₁

    Projecting onto PC1 alone (1D) captures most of the variance.
    The 2D problem becomes 1D with minimal information loss.

**LDA (Linear Discriminant Analysis)** — for classification. Finds
    directions that maximise class separation (not just variance).
    Often better than PCA as a pre-processing step for classifiers.

**t-SNE / UMAP** — non-linear dimensionality reduction for visualisation.
    Reveals cluster structure invisible in PCA. NOT used for training —
    only for understanding and exploration.

    When to use PCA before training:
    ✓ Many correlated features (e.g., image pixels, sensor readings)
    ✓ Linear model that can't capture correlations natively
    ✓ Want to reduce training time
    ✗ Tree-based models (they handle correlated features natively)
    ✗ When interpretability of original features matters

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Image Augmentation Pipeline from Scratch": {
        "description": (
            "Build a full image augmentation pipeline using only NumPy. "
            "Apply flipping, rotation, brightness jitter, cropping, and CutOut. "
            "Show how each transform keeps the label valid."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Create a synthetic 32×32 "image" with structure (not random noise) ────
def make_synthetic_image():
    """A simple pattern: bright centre circle on dark background."""
    img = np.zeros((32, 32, 3), dtype=np.float32)
    cx, cy, r = 16, 16, 8
    Y, X = np.ogrid[:32, :32]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    img[mask, 0] = 0.9   # red channel: circle
    img[~mask, 2] = 0.4  # blue channel: background
    img += np.random.uniform(0, 0.05, img.shape)  # tiny noise
    return np.clip(img, 0, 1)

# ── Augmentation functions (all label-preserving) ─────────────────────────
def horizontal_flip(img):
    return img[:, ::-1, :]

def random_rotation(img, max_angle=30):
    """Rotate by a random angle via bilinear interpolation."""
    angle = np.random.uniform(-max_angle, max_angle)
    rad   = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    H, W = img.shape[:2]
    cx, cy = W / 2, H / 2
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            # Inverse rotation to find source pixel
            src_x = cos_a*(x-cx) + sin_a*(y-cy) + cx
            src_y = -sin_a*(x-cx) + cos_a*(y-cy) + cy
            xi, yi = int(src_x), int(src_y)
            if 0 <= xi < W and 0 <= yi < H:
                out[y, x] = img[yi, xi]
    return out

def brightness_jitter(img, factor_range=(0.6, 1.4)):
    factor = np.random.uniform(*factor_range)
    return np.clip(img * factor, 0, 1)

def random_crop_and_resize(img, crop_frac=0.8):
    """Crop a random sub-region then resize back (nearest neighbour)."""
    H, W = img.shape[:2]
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    top  = np.random.randint(0, H - ch + 1)
    left = np.random.randint(0, W - cw + 1)
    cropped = img[top:top+ch, left:left+cw, :]
    # Nearest-neighbour resize back to H×W
    row_idx = (np.arange(H) * ch / H).astype(int)
    col_idx = (np.arange(W) * cw / W).astype(int)
    return cropped[np.ix_(row_idx, col_idx)]

def cutout(img, patch_size=8):
    """Zero out a random square patch."""
    out = img.copy()
    H, W = img.shape[:2]
    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)
    y1, y2 = max(0, cy - patch_size//2), min(H, cy + patch_size//2)
    x1, x2 = max(0, cx - patch_size//2), min(W, cx + patch_size//2)
    out[y1:y2, x1:x2, :] = 0.0
    return out

def gaussian_noise(img, std=0.05):
    return np.clip(img + np.random.normal(0, std, img.shape), 0, 1)

# ── Build pipeline ────────────────────────────────────────────────────────
original = make_synthetic_image()

transforms = [
    ("Original",          original),
    ("Horizontal Flip",   horizontal_flip(original)),
    ("Brightness Jitter", brightness_jitter(original)),
    ("Random Crop",       random_crop_and_resize(original)),
    ("CutOut",            cutout(original, patch_size=10)),
    ("Gaussian Noise",    gaussian_noise(original, std=0.08)),
    ("Rotation ~25°",     random_rotation(original, max_angle=25)),
    ("Full Pipeline",     cutout(brightness_jitter(
                          random_crop_and_resize(original)), patch_size=8)),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Image Augmentation Pipeline — All Transforms Are Label-Preserving",
             fontsize=12, fontweight="bold")

for ax, (name, img) in zip(axes.ravel(), transforms):
    ax.imshow(img)
    ax.set_title(name, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("augmentation_pipeline.png", dpi=120)

print("=" * 60)
print("  IMAGE AUGMENTATION PIPELINE")
print("=" * 60)
print()
print("  8 transforms demonstrated:")
for name, _ in transforms:
    valid = "✓ label-preserving" if name != "Original" else "  baseline"
    print(f"    {name:<22}  {valid}")
print()
print("  In practice you chain 2-4 of these randomly per sample.")
print("  Each training epoch sees a DIFFERENT augmented version,")
print("  effectively multiplying your dataset size.")
print()
print("  Plot saved → augmentation_pipeline.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · MixUp Augmentation Demo": {
        "description": (
            "Implement MixUp from scratch and show how it creates soft labels "
            "that prevent the model from being overconfident, "
            "smoothing the decision boundary."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt

# ── Sigmoid ───────────────────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,max_iter=500,C=1.0,penalty="l2",solver="lbfgs",random_state=None,class_weight=None):
        self.max_iter=max_iter; self.C=C; self.penalty=penalty; self.class_weight=class_weight
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        lam=1.0/self.C if self.C>0 else 0
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso alias ────────────────────────────────────────────────────────────────
Lasso = LogisticRegression   # used only as feature selector; coef_ attribute is shared

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X0=np.c_[np.cos(t),np.sin(t)]; X1=np.c_[1-np.cos(t),-np.sin(t)+0.5]
    X=np.vstack([X0,X1])+rng.normal(0,noise,(n_samples,2))
    y=np.hstack([np.zeros(n),np.ones(n)]).astype(int)
    return X,y

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,n_clusters_per_class=1,weights=None,flip_y=0.01,
                        random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_digits():
    """Synthetic digits-like dataset: 1797 samples, 64 features, 10 classes."""
    rng=np.random.default_rng(0); n_per=180; n_cls=10
    X_list=[]; y_list=[]
    for c in range(n_cls):
        proto=rng.uniform(0,16,(64,)); proto=np.clip(proto,0,16)
        samp=proto+rng.normal(0,3,(n_per,64)); samp=np.clip(samp,0,16)
        X_list.append(samp); y_list.append(np.full(n_per,c))
    X=np.vstack(X_list); y=np.hstack(y_list).astype(int)
    idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold / StratifiedKFold ───────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs if i<self.n_splits-1 else n]
            tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs if i<self.n_splits-1 else n:]])
            yield tr,te

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); classes=np.unique(y)
        rng=np.random.default_rng(self.random_state)
        folds=[[] for _ in range(self.n_splits)]
        for c in classes:
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)):
                folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i])
            tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()
def _tp_fp_fn_tn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); d=2*tp+fp+fn
    return float(2*tp/d) if d>0 else float(zero_division)
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); return np.array([[tn,fp],[fn,tp]])
def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c:3d}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv
        splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),
                 np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── NearestNeighbors ──────────────────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]; D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── PolynomialFeatures ────────────────────────────────────────────────────────
class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True):
        self.degree=degree; self.include_bias=include_bias
    def _combos(self,n_feat):
        from itertools import combinations_with_replacement
        combos=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: combos.append(())
            else: combos.extend(combinations_with_replacement(range(n_feat),d))
        return combos
    def fit(self,X,y=None):
        self._combos_=self._combos(X.shape[1])
        self.n_output_features_=len(self._combos_)
        return self
    def transform(self,X):
        cols=[]
        for combo in self._combos_:
            if len(combo)==0: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────
class PCA:
    def __init__(self,n_components=None): self.n_components=n_components
    def fit(self,X):
        self.mean_=X.mean(axis=0); Xc=X-self.mean_
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        tot=np.sum(s**2); self.explained_variance_ratio_=(s**2)/tot
        self.components_=Vt; self.singular_values_=s
        if self.n_components: self.components_=Vt[:self.n_components]
        return self
    def transform(self,X): return (X-self.mean_)@self.components_.T
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── MI feature selection ──────────────────────────────────────────────────────
def _discretise(arr,n_bins=12):
    p=np.unique(np.percentile(arr,np.linspace(0,100,n_bins+1)))
    return np.digitize(arr,p[1:-1])
def _mi(xd,yd):
    n=len(xd); mi=0.0
    for xv in np.unique(xd):
        for yv in np.unique(yd):
            nxy=np.sum((xd==xv)&(yd==yv))
            if nxy==0: continue
            mi+=(nxy/n)*np.log((nxy*n)/(np.sum(xd==xv)*np.sum(yd==yv)))
    return max(mi,0.0)
def mutual_info_classif(X,y,random_state=None):
    yd=y.astype(int)
    return np.array([_mi(_discretise(X[:,i]),yd) for i in range(X.shape[1])])
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]; Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_sp_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        result=self.score_func(X,y)
        sc=result if not isinstance(result,tuple) else result[0]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(sc)[::-1][:self.k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── Simple Decision Tree (for RF) ─────────────────────────────────────────────
class _DTree:
    def __init__(self,max_depth=6,min_samples=2,rng=None):
        self.max_depth=max_depth; self.min_samples=min_samples; self.rng=rng or np.random
        self.tree_=None
    def _gini(self,y):
        if len(y)==0: return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _best_split(self,X,y,feats):
        best={"gain":-1,"feat":None,"thr":None}
        g0=self._gini(y); n=len(y)
        for f in feats:
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_thrs=(vals[:-1]+vals[1:])/2
            # sample at most 8 thresholds to keep pure-Python trees fast
            thrs=all_thrs[np.linspace(0,len(all_thrs)-1,min(8,len(all_thrs))).astype(int)]
            for t in thrs:
                lm=X[:,f]<=t; rm=~lm
                if lm.sum()<self.min_samples or rm.sum()<self.min_samples: continue
                gain=g0-(lm.sum()/n*self._gini(y[lm])+rm.sum()/n*self._gini(y[rm]))
                if gain>best["gain"]: best={"gain":gain,"feat":f,"thr":t}
        return best
    def _build(self,X,y,depth):
        classes,counts=np.unique(y,return_counts=True)
        leaf={"leaf":True,"pred":classes[np.argmax(counts)]}
        if depth>=self.max_depth or len(y)<=self.min_samples or len(classes)==1:
            return leaf
        n_feats=max(1,int(np.sqrt(X.shape[1])))
        feats=self.rng.choice(X.shape[1],n_feats,replace=False)
        sp=self._best_split(X,y,feats)
        if sp["gain"]<=0: return leaf
        lm=X[:,sp["feat"]]<=sp["thr"]
        return {"leaf":False,"feat":sp["feat"],"thr":sp["thr"],
                "left":self._build(X[lm],y[lm],depth+1),
                "right":self._build(X[~lm],y[~lm],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _pred1(self,x,node):
        if node["leaf"]: return node["pred"]
        return self._pred1(x,node["left"] if x[node["feat"]]<=node["thr"] else node["right"])
    def predict(self,X): return np.array([self._pred1(x,self.tree_) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=15,max_depth=5,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self.feature_importances_=np.zeros(X.shape[1])
        self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); tree_rng=np.random.default_rng(int(rng.integers(0,1e9)))
            t=_DTree(max_depth=self.max_depth,rng=tree_rng).fit(X[idx],y[idx])
            self.estimators_.append(t)
        # Approximate feature importances via split frequency
        def _imp(node,imp):
            if node["leaf"]: return
            imp[node["feat"]]+=1; _imp(node["left"],imp); _imp(node["right"],imp)
        for t in self.estimators_: _imp(t.tree_,self.feature_importances_)
        s=self.feature_importances_.sum()
        if s>0: self.feature_importances_/=s
        return self
    def predict(self,X):
        votes=np.array([t.predict(X) for t in self.estimators_])
        return np.array([np.bincount(votes[:,i].astype(int),minlength=len(self.classes_)).argmax()
                         for i in range(len(X))])
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── RFE ───────────────────────────────────────────────────────────────────────
class RFE:
    def __init__(self,estimator,n_features_to_select=10,step=1):
        self.estimator=estimator; self.n_features_to_select=n_features_to_select; self.step=step
    def fit(self,X,y):
        n_feat=X.shape[1]; mask=np.ones(n_feat,dtype=bool); feats=np.arange(n_feat)
        while mask.sum()>self.n_features_to_select:
            est=_copy.deepcopy(self.estimator); est.fit(X[:,mask],y)
            importance=np.abs(est.coef_.ravel())
            remove=max(1,min(self.step,mask.sum()-self.n_features_to_select))
            worst=np.argsort(importance)[:remove]
            active=feats[mask]; mask[active[worst]]=False
        self._mask=mask; return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── SelectFromModel ────────────────────────────────────────────────────────────
class SelectFromModel:
    def __init__(self,estimator,max_features=None,threshold=None):
        self.estimator=estimator; self.max_features=max_features; self.threshold=threshold
    def fit(self,X,y):
        est=_copy.deepcopy(self.estimator); est.fit(X,y)
        if hasattr(est,"feature_importances_"): imp=est.feature_importances_
        elif hasattr(est,"coef_"): imp=np.abs(est.coef_.ravel())
        else: imp=np.ones(X.shape[1])
        k=self.max_features if self.max_features else X.shape[1]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(imp)[::-1][:k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

np.random.seed(7)

# ── Dataset ───────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
X = StandardScaler().fit_transform(X)

def mixup_batch(X, y, alpha=0.4):
    """
    MixUp: for each sample, blend it with a random other sample.
    lambda ~ Beta(alpha, alpha)
    x_mix = lam*x_i + (1-lam)*x_j
    y_mix = lam*y_i + (1-lam)*y_j   ← SOFT label, not hard 0/1
    """
    n = len(X)
    lam = np.random.beta(alpha, alpha, size=n)

    # Random pairing
    idx_shuffle = np.random.permutation(n)
    X_shuffle   = X[idx_shuffle]
    y_shuffle   = y[idx_shuffle]

    # Blend
    lam_col = lam.reshape(-1, 1)
    X_mix = lam_col * X + (1 - lam_col) * X_shuffle
    y_mix = lam * y + (1 - lam) * y_shuffle    # soft labels in [0, 1]

    return X_mix, y_mix, lam

# ── Generate several MixUp examples at different alpha values ─────────────
alphas = [0.1, 0.4, 1.0, 2.0]

print("=" * 60)
print("  MIXUP AUGMENTATION")
print("=" * 60)
print()
print("  MixUp formula:")
print("    x_mix = λ · x_i + (1-λ) · x_j")
print("    y_mix = λ · y_i + (1-λ) · y_j")
print("    where λ ~ Beta(α, α)")
print()
print("  Effect of α on lambda distribution:")
print(f"  {'α':>6} | {'mean λ':>8} | {'std λ':>8} | {'% near boundaries':>18}")
print(f"  {'─'*55}")
for alpha in alphas:
    lam_samples = np.random.beta(alpha, alpha, 10000)
    near_boundary = np.mean((lam_samples < 0.1) | (lam_samples > 0.9))
    print(f"  {alpha:6.1f} | {lam_samples.mean():8.3f} | "
          f"{lam_samples.std():8.3f} | {near_boundary*100:17.1f}%")

print()
print("  Small α → λ concentrated near 0 and 1 (weak blending)")
print("  Large α → λ concentrated near 0.5 (strong blending)")
print()

# ── Show MixUp examples visually ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("MixUp: Effect of Alpha on Blending Strength",
             fontsize=12, fontweight="bold")

colours = {0: "steelblue", 1: "tomato"}

for ax, alpha in zip(axes, alphas):
    X_mix, y_mix, lam = mixup_batch(X, y, alpha=alpha)

    # Colour points by their soft label (shade = mix intensity)
    sc = ax.scatter(X_mix[:, 0], X_mix[:, 1],
                    c=y_mix, cmap="RdBu", vmin=0, vmax=1,
                    s=20, alpha=0.7, edgecolors="none")
    ax.set_title(f"Alpha = {alpha}\\nSoft labels in [{y_mix.min():.2f}, "
                 f"{y_mix.max():.2f}]", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.04)

plt.tight_layout()
plt.savefig("mixup_demo.png", dpi=120)
print("  Plot saved → mixup_demo.png")
print()
print("  WHY MIXUP PREVENTS OVERFITTING:")
print("  1. Model never sees the same example twice → less memorisation")
print("  2. Soft labels discourage overconfidence (avoids p(y|x)→1.0)")
print("  3. Decision boundaries become smoother and more linear between")
print("     training points — better generalisation in gaps")
print("  4. Acts as a vicinal risk minimisation: model must interpolate")
print("     correctly between data points, not just at them")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · SMOTE for Class Imbalance": {
        "description": (
            "Implement SMOTE from scratch for a severely imbalanced dataset. "
            "Compare a baseline classifier vs one trained on SMOTE-augmented data. "
            "Show that accuracy is misleading but F1 exposes the real difference."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt

# ── Sigmoid ───────────────────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,max_iter=500,C=1.0,penalty="l2",solver="lbfgs",random_state=None,class_weight=None):
        self.max_iter=max_iter; self.C=C; self.penalty=penalty; self.class_weight=class_weight
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        lam=1.0/self.C if self.C>0 else 0
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso alias ────────────────────────────────────────────────────────────────
Lasso = LogisticRegression   # used only as feature selector; coef_ attribute is shared

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X0=np.c_[np.cos(t),np.sin(t)]; X1=np.c_[1-np.cos(t),-np.sin(t)+0.5]
    X=np.vstack([X0,X1])+rng.normal(0,noise,(n_samples,2))
    y=np.hstack([np.zeros(n),np.ones(n)]).astype(int)
    return X,y

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,n_clusters_per_class=1,weights=None,flip_y=0.01,
                        random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_digits():
    """Synthetic digits-like dataset: 1797 samples, 64 features, 10 classes."""
    rng=np.random.default_rng(0); n_per=180; n_cls=10
    X_list=[]; y_list=[]
    for c in range(n_cls):
        proto=rng.uniform(0,16,(64,)); proto=np.clip(proto,0,16)
        samp=proto+rng.normal(0,3,(n_per,64)); samp=np.clip(samp,0,16)
        X_list.append(samp); y_list.append(np.full(n_per,c))
    X=np.vstack(X_list); y=np.hstack(y_list).astype(int)
    idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold / StratifiedKFold ───────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs if i<self.n_splits-1 else n]
            tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs if i<self.n_splits-1 else n:]])
            yield tr,te

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); classes=np.unique(y)
        rng=np.random.default_rng(self.random_state)
        folds=[[] for _ in range(self.n_splits)]
        for c in classes:
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)):
                folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i])
            tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()
def _tp_fp_fn_tn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); d=2*tp+fp+fn
    return float(2*tp/d) if d>0 else float(zero_division)
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); return np.array([[tn,fp],[fn,tp]])
def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c:3d}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv
        splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),
                 np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── NearestNeighbors ──────────────────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]; D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── PolynomialFeatures ────────────────────────────────────────────────────────
class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True):
        self.degree=degree; self.include_bias=include_bias
    def _combos(self,n_feat):
        from itertools import combinations_with_replacement
        combos=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: combos.append(())
            else: combos.extend(combinations_with_replacement(range(n_feat),d))
        return combos
    def fit(self,X,y=None):
        self._combos_=self._combos(X.shape[1])
        self.n_output_features_=len(self._combos_)
        return self
    def transform(self,X):
        cols=[]
        for combo in self._combos_:
            if len(combo)==0: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────
class PCA:
    def __init__(self,n_components=None): self.n_components=n_components
    def fit(self,X):
        self.mean_=X.mean(axis=0); Xc=X-self.mean_
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        tot=np.sum(s**2); self.explained_variance_ratio_=(s**2)/tot
        self.components_=Vt; self.singular_values_=s
        if self.n_components: self.components_=Vt[:self.n_components]
        return self
    def transform(self,X): return (X-self.mean_)@self.components_.T
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── MI feature selection ──────────────────────────────────────────────────────
def _discretise(arr,n_bins=12):
    p=np.unique(np.percentile(arr,np.linspace(0,100,n_bins+1)))
    return np.digitize(arr,p[1:-1])
def _mi(xd,yd):
    n=len(xd); mi=0.0
    for xv in np.unique(xd):
        for yv in np.unique(yd):
            nxy=np.sum((xd==xv)&(yd==yv))
            if nxy==0: continue
            mi+=(nxy/n)*np.log((nxy*n)/(np.sum(xd==xv)*np.sum(yd==yv)))
    return max(mi,0.0)
def mutual_info_classif(X,y,random_state=None):
    yd=y.astype(int)
    return np.array([_mi(_discretise(X[:,i]),yd) for i in range(X.shape[1])])
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]; Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_sp_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        result=self.score_func(X,y)
        sc=result if not isinstance(result,tuple) else result[0]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(sc)[::-1][:self.k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── Simple Decision Tree (for RF) ─────────────────────────────────────────────
class _DTree:
    def __init__(self,max_depth=6,min_samples=2,rng=None):
        self.max_depth=max_depth; self.min_samples=min_samples; self.rng=rng or np.random
        self.tree_=None
    def _gini(self,y):
        if len(y)==0: return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _best_split(self,X,y,feats):
        best={"gain":-1,"feat":None,"thr":None}
        g0=self._gini(y); n=len(y)
        for f in feats:
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_thrs=(vals[:-1]+vals[1:])/2
            # sample at most 8 thresholds to keep pure-Python trees fast
            thrs=all_thrs[np.linspace(0,len(all_thrs)-1,min(8,len(all_thrs))).astype(int)]
            for t in thrs:
                lm=X[:,f]<=t; rm=~lm
                if lm.sum()<self.min_samples or rm.sum()<self.min_samples: continue
                gain=g0-(lm.sum()/n*self._gini(y[lm])+rm.sum()/n*self._gini(y[rm]))
                if gain>best["gain"]: best={"gain":gain,"feat":f,"thr":t}
        return best
    def _build(self,X,y,depth):
        classes,counts=np.unique(y,return_counts=True)
        leaf={"leaf":True,"pred":classes[np.argmax(counts)]}
        if depth>=self.max_depth or len(y)<=self.min_samples or len(classes)==1:
            return leaf
        n_feats=max(1,int(np.sqrt(X.shape[1])))
        feats=self.rng.choice(X.shape[1],n_feats,replace=False)
        sp=self._best_split(X,y,feats)
        if sp["gain"]<=0: return leaf
        lm=X[:,sp["feat"]]<=sp["thr"]
        return {"leaf":False,"feat":sp["feat"],"thr":sp["thr"],
                "left":self._build(X[lm],y[lm],depth+1),
                "right":self._build(X[~lm],y[~lm],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _pred1(self,x,node):
        if node["leaf"]: return node["pred"]
        return self._pred1(x,node["left"] if x[node["feat"]]<=node["thr"] else node["right"])
    def predict(self,X): return np.array([self._pred1(x,self.tree_) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=15,max_depth=5,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self.feature_importances_=np.zeros(X.shape[1])
        self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); tree_rng=np.random.default_rng(int(rng.integers(0,1e9)))
            t=_DTree(max_depth=self.max_depth,rng=tree_rng).fit(X[idx],y[idx])
            self.estimators_.append(t)
        # Approximate feature importances via split frequency
        def _imp(node,imp):
            if node["leaf"]: return
            imp[node["feat"]]+=1; _imp(node["left"],imp); _imp(node["right"],imp)
        for t in self.estimators_: _imp(t.tree_,self.feature_importances_)
        s=self.feature_importances_.sum()
        if s>0: self.feature_importances_/=s
        return self
    def predict(self,X):
        votes=np.array([t.predict(X) for t in self.estimators_])
        return np.array([np.bincount(votes[:,i].astype(int),minlength=len(self.classes_)).argmax()
                         for i in range(len(X))])
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── RFE ───────────────────────────────────────────────────────────────────────
class RFE:
    def __init__(self,estimator,n_features_to_select=10,step=1):
        self.estimator=estimator; self.n_features_to_select=n_features_to_select; self.step=step
    def fit(self,X,y):
        n_feat=X.shape[1]; mask=np.ones(n_feat,dtype=bool); feats=np.arange(n_feat)
        while mask.sum()>self.n_features_to_select:
            est=_copy.deepcopy(self.estimator); est.fit(X[:,mask],y)
            importance=np.abs(est.coef_.ravel())
            remove=max(1,min(self.step,mask.sum()-self.n_features_to_select))
            worst=np.argsort(importance)[:remove]
            active=feats[mask]; mask[active[worst]]=False
        self._mask=mask; return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── SelectFromModel ────────────────────────────────────────────────────────────
class SelectFromModel:
    def __init__(self,estimator,max_features=None,threshold=None):
        self.estimator=estimator; self.max_features=max_features; self.threshold=threshold
    def fit(self,X,y):
        est=_copy.deepcopy(self.estimator); est.fit(X,y)
        if hasattr(est,"feature_importances_"): imp=est.feature_importances_
        elif hasattr(est,"coef_"): imp=np.abs(est.coef_.ravel())
        else: imp=np.ones(X.shape[1])
        k=self.max_features if self.max_features else X.shape[1]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(imp)[::-1][:k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

np.random.seed(0)

# ── Severely imbalanced dataset: 97% majority, 3% minority ───────────────
X, y = make_classification(
    n_samples=2000, n_features=5, n_informative=4,
    weights=[0.97, 0.03], flip_y=0.01, random_state=42
)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1)

print("=" * 60)
print("  SMOTE: SYNTHETIC MINORITY OVER-SAMPLING TECHNIQUE")
print("=" * 60)
print(f"  Total samples  : {len(y)}")
print(f"  Majority (0)   : {(y==0).sum()}  ({(y==0).mean()*100:.1f}%)")
print(f"  Minority (1)   : {(y==1).sum()}  ({(y==1).mean()*100:.1f}%)")
print()

# ── SMOTE from scratch ────────────────────────────────────────────────────
def smote(X_min, k=5, n_synthetic=None):
    """
    Generate synthetic minority samples by interpolating between
    real minority points and their k nearest neighbours.
    """
    pass  # NearestNeighbors defined at module level
    n = len(X_min)
    if n_synthetic is None:
        n_synthetic = n

    nn = NearestNeighbors(n_neighbors=k+1).fit(X_min)
    _, indices = nn.kneighbors(X_min)  # first col is self

    synthetic = []
    for _ in range(n_synthetic):
        i   = np.random.randint(0, n)
        nn_i = indices[i, 1:]                    # exclude self
        j   = nn_i[np.random.randint(0, k)]     # random neighbour
        lam = np.random.uniform(0, 1)
        syn = X_min[i] + lam * (X_min[j] - X_min[i])
        synthetic.append(syn)

    return np.array(synthetic)

# Apply SMOTE to training set
X_min_tr = X_tr[y_tr == 1]
X_maj_tr = X_tr[y_tr == 0]
n_to_gen = len(X_maj_tr) - len(X_min_tr)   # enough to balance perfectly

print(f"  Training set before SMOTE:")
print(f"    Majority: {len(X_maj_tr)}  Minority: {len(X_min_tr)}")

X_synthetic = smote(X_min_tr, k=5, n_synthetic=n_to_gen)
X_tr_smote  = np.vstack([X_tr, X_synthetic])
y_tr_smote  = np.hstack([y_tr, np.ones(n_to_gen)])

print(f"  Training set after SMOTE:")
print(f"    Majority: {(y_tr_smote==0).sum()}  "
      f"Minority: {(y_tr_smote==1).sum()}")
print()

# ── Train two models ──────────────────────────────────────────────────────
model_plain = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
model_smote = LogisticRegression(max_iter=500).fit(X_tr_smote, y_tr_smote)

for name, model in [("BASELINE (no augmentation)", model_plain),
                     ("SMOTE augmented",            model_smote)]:
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)
    print(f"  {name}")
    print(f"    Accuracy : {acc:.4f}  ← MISLEADING on imbalanced data")
    print(f"    F1 Score : {f1:.4f}  ← TRUE performance indicator")
    print(f"    Confusion Matrix:")
    print(f"               Pred 0   Pred 1")
    print(f"      True 0:   {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"      True 1:   {cm[1,0]:5d}    {cm[1,1]:5d}")
    tp_rate = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0
    print(f"    Minority recall (% caught): {tp_rate*100:.1f}%")
    print()

# ── Visualise in 2D (first 2 features) ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SMOTE: Synthetic Minority Oversampling", fontsize=12,
             fontweight="bold")

ax1, ax2 = axes
ax1.scatter(X_tr[y_tr==0, 0], X_tr[y_tr==0, 1],
            c="steelblue", alpha=0.4, s=15, label="Majority (real)")
ax1.scatter(X_tr[y_tr==1, 0], X_tr[y_tr==1, 1],
            c="tomato", s=40, label=f"Minority (real, n={len(X_min_tr)})")
ax1.set_title("Before SMOTE", fontsize=10)
ax1.legend(fontsize=8)

ax2.scatter(X_tr[y_tr==0, 0], X_tr[y_tr==0, 1],
            c="steelblue", alpha=0.3, s=15, label="Majority (real)")
ax2.scatter(X_tr[y_tr==1, 0], X_tr[y_tr==1, 1],
            c="tomato", s=40, label="Minority (real)")
ax2.scatter(X_synthetic[:, 0], X_synthetic[:, 1],
            c="orange", marker="x", s=20, alpha=0.6,
            label=f"Synthetic (SMOTE, n={len(X_synthetic)})")
ax2.set_title("After SMOTE (balanced)", fontsize=10)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("smote_demo.png", dpi=120)
print("  Plot saved → smote_demo.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Feature Engineering Fixes Underfitting": {
        "description": (
            "Show a linear model completely failing on non-linear data (underfitting), "
            "then fix it purely with polynomial and interaction feature engineering "
            "— without changing the model class."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt

# ── Sigmoid ───────────────────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,max_iter=500,C=1.0,penalty="l2",solver="lbfgs",random_state=None,class_weight=None):
        self.max_iter=max_iter; self.C=C; self.penalty=penalty; self.class_weight=class_weight
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        lam=1.0/self.C if self.C>0 else 0
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso alias ────────────────────────────────────────────────────────────────
Lasso = LogisticRegression   # used only as feature selector; coef_ attribute is shared

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X0=np.c_[np.cos(t),np.sin(t)]; X1=np.c_[1-np.cos(t),-np.sin(t)+0.5]
    X=np.vstack([X0,X1])+rng.normal(0,noise,(n_samples,2))
    y=np.hstack([np.zeros(n),np.ones(n)]).astype(int)
    return X,y

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,n_clusters_per_class=1,weights=None,flip_y=0.01,
                        random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_digits():
    """Synthetic digits-like dataset: 1797 samples, 64 features, 10 classes."""
    rng=np.random.default_rng(0); n_per=180; n_cls=10
    X_list=[]; y_list=[]
    for c in range(n_cls):
        proto=rng.uniform(0,16,(64,)); proto=np.clip(proto,0,16)
        samp=proto+rng.normal(0,3,(n_per,64)); samp=np.clip(samp,0,16)
        X_list.append(samp); y_list.append(np.full(n_per,c))
    X=np.vstack(X_list); y=np.hstack(y_list).astype(int)
    idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold / StratifiedKFold ───────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs if i<self.n_splits-1 else n]
            tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs if i<self.n_splits-1 else n:]])
            yield tr,te

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); classes=np.unique(y)
        rng=np.random.default_rng(self.random_state)
        folds=[[] for _ in range(self.n_splits)]
        for c in classes:
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)):
                folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i])
            tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()
def _tp_fp_fn_tn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); d=2*tp+fp+fn
    return float(2*tp/d) if d>0 else float(zero_division)
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); return np.array([[tn,fp],[fn,tp]])
def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c:3d}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv
        splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),
                 np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── NearestNeighbors ──────────────────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]; D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── PolynomialFeatures ────────────────────────────────────────────────────────
class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True):
        self.degree=degree; self.include_bias=include_bias
    def _combos(self,n_feat):
        from itertools import combinations_with_replacement
        combos=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: combos.append(())
            else: combos.extend(combinations_with_replacement(range(n_feat),d))
        return combos
    def fit(self,X,y=None):
        self._combos_=self._combos(X.shape[1])
        self.n_output_features_=len(self._combos_)
        return self
    def transform(self,X):
        cols=[]
        for combo in self._combos_:
            if len(combo)==0: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────
class PCA:
    def __init__(self,n_components=None): self.n_components=n_components
    def fit(self,X):
        self.mean_=X.mean(axis=0); Xc=X-self.mean_
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        tot=np.sum(s**2); self.explained_variance_ratio_=(s**2)/tot
        self.components_=Vt; self.singular_values_=s
        if self.n_components: self.components_=Vt[:self.n_components]
        return self
    def transform(self,X): return (X-self.mean_)@self.components_.T
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── MI feature selection ──────────────────────────────────────────────────────
def _discretise(arr,n_bins=12):
    p=np.unique(np.percentile(arr,np.linspace(0,100,n_bins+1)))
    return np.digitize(arr,p[1:-1])
def _mi(xd,yd):
    n=len(xd); mi=0.0
    for xv in np.unique(xd):
        for yv in np.unique(yd):
            nxy=np.sum((xd==xv)&(yd==yv))
            if nxy==0: continue
            mi+=(nxy/n)*np.log((nxy*n)/(np.sum(xd==xv)*np.sum(yd==yv)))
    return max(mi,0.0)
def mutual_info_classif(X,y,random_state=None):
    yd=y.astype(int)
    return np.array([_mi(_discretise(X[:,i]),yd) for i in range(X.shape[1])])
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]; Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_sp_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        result=self.score_func(X,y)
        sc=result if not isinstance(result,tuple) else result[0]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(sc)[::-1][:self.k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── Simple Decision Tree (for RF) ─────────────────────────────────────────────
class _DTree:
    def __init__(self,max_depth=6,min_samples=2,rng=None):
        self.max_depth=max_depth; self.min_samples=min_samples; self.rng=rng or np.random
        self.tree_=None
    def _gini(self,y):
        if len(y)==0: return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _best_split(self,X,y,feats):
        best={"gain":-1,"feat":None,"thr":None}
        g0=self._gini(y); n=len(y)
        for f in feats:
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_thrs=(vals[:-1]+vals[1:])/2
            # sample at most 8 thresholds to keep pure-Python trees fast
            thrs=all_thrs[np.linspace(0,len(all_thrs)-1,min(8,len(all_thrs))).astype(int)]
            for t in thrs:
                lm=X[:,f]<=t; rm=~lm
                if lm.sum()<self.min_samples or rm.sum()<self.min_samples: continue
                gain=g0-(lm.sum()/n*self._gini(y[lm])+rm.sum()/n*self._gini(y[rm]))
                if gain>best["gain"]: best={"gain":gain,"feat":f,"thr":t}
        return best
    def _build(self,X,y,depth):
        classes,counts=np.unique(y,return_counts=True)
        leaf={"leaf":True,"pred":classes[np.argmax(counts)]}
        if depth>=self.max_depth or len(y)<=self.min_samples or len(classes)==1:
            return leaf
        n_feats=max(1,int(np.sqrt(X.shape[1])))
        feats=self.rng.choice(X.shape[1],n_feats,replace=False)
        sp=self._best_split(X,y,feats)
        if sp["gain"]<=0: return leaf
        lm=X[:,sp["feat"]]<=sp["thr"]
        return {"leaf":False,"feat":sp["feat"],"thr":sp["thr"],
                "left":self._build(X[lm],y[lm],depth+1),
                "right":self._build(X[~lm],y[~lm],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _pred1(self,x,node):
        if node["leaf"]: return node["pred"]
        return self._pred1(x,node["left"] if x[node["feat"]]<=node["thr"] else node["right"])
    def predict(self,X): return np.array([self._pred1(x,self.tree_) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=15,max_depth=5,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self.feature_importances_=np.zeros(X.shape[1])
        self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); tree_rng=np.random.default_rng(int(rng.integers(0,1e9)))
            t=_DTree(max_depth=self.max_depth,rng=tree_rng).fit(X[idx],y[idx])
            self.estimators_.append(t)
        # Approximate feature importances via split frequency
        def _imp(node,imp):
            if node["leaf"]: return
            imp[node["feat"]]+=1; _imp(node["left"],imp); _imp(node["right"],imp)
        for t in self.estimators_: _imp(t.tree_,self.feature_importances_)
        s=self.feature_importances_.sum()
        if s>0: self.feature_importances_/=s
        return self
    def predict(self,X):
        votes=np.array([t.predict(X) for t in self.estimators_])
        return np.array([np.bincount(votes[:,i].astype(int),minlength=len(self.classes_)).argmax()
                         for i in range(len(X))])
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── RFE ───────────────────────────────────────────────────────────────────────
class RFE:
    def __init__(self,estimator,n_features_to_select=10,step=1):
        self.estimator=estimator; self.n_features_to_select=n_features_to_select; self.step=step
    def fit(self,X,y):
        n_feat=X.shape[1]; mask=np.ones(n_feat,dtype=bool); feats=np.arange(n_feat)
        while mask.sum()>self.n_features_to_select:
            est=_copy.deepcopy(self.estimator); est.fit(X[:,mask],y)
            importance=np.abs(est.coef_.ravel())
            remove=max(1,min(self.step,mask.sum()-self.n_features_to_select))
            worst=np.argsort(importance)[:remove]
            active=feats[mask]; mask[active[worst]]=False
        self._mask=mask; return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── SelectFromModel ────────────────────────────────────────────────────────────
class SelectFromModel:
    def __init__(self,estimator,max_features=None,threshold=None):
        self.estimator=estimator; self.max_features=max_features; self.threshold=threshold
    def fit(self,X,y):
        est=_copy.deepcopy(self.estimator); est.fit(X,y)
        if hasattr(est,"feature_importances_"): imp=est.feature_importances_
        elif hasattr(est,"coef_"): imp=np.abs(est.coef_.ravel())
        else: imp=np.ones(X.shape[1])
        k=self.max_features if self.max_features else X.shape[1]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(imp)[::-1][:k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

np.random.seed(0)

# ── Data: circular decision boundary — impossible for a linear model ───────
n = 500
r_inner = np.random.uniform(0, 1.0, n // 2)
r_outer = np.random.uniform(1.8, 3.0, n // 2)
theta_i = np.random.uniform(0, 2*np.pi, n // 2)
theta_o = np.random.uniform(0, 2*np.pi, n // 2)

X_inner = np.c_[r_inner*np.cos(theta_i), r_inner*np.sin(theta_i)]
X_outer = np.c_[r_outer*np.cos(theta_o), r_outer*np.sin(theta_o)]
X = np.vstack([X_inner, X_outer])
y = np.hstack([np.zeros(n//2), np.ones(n//2)])

# Add noise
X += np.random.normal(0, 0.1, X.shape)

# train_test_split defined above
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

print("=" * 65)
print("  FEATURE ENGINEERING: FIXING UNDERFITTING IN A LINEAR MODEL")
print("=" * 65)
print(f"  Problem: circular boundary — class 0 (inner) vs class 1 (outer)")
print(f"  A linear model CANNOT draw a circle. Watch feature engineering fix it.")
print()

configs = [
    ("Raw features [x₁, x₂]",              1, False),
    ("Poly degree 2  (adds x₁²,x₂²,x₁x₂)", 2, False),
    ("Poly degree 3",                         3, False),
    ("Poly degree 4",                         4, False),
]

results = []
xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-3.5, 3.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Feature Engineering: Polynomial Features Fix a Linear Model's Underfitting",
             fontsize=11, fontweight="bold")

for ax, (label, degree, _) in zip(axes, configs):
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(X_tr, y_tr)
    acc_tr = accuracy_score(y_tr, pipe.predict(X_tr))
    acc_te = accuracy_score(y_te, pipe.predict(X_te))
    n_feat = pipe.named_steps["poly"].n_output_features_

    # Decision boundary
    Z = pipe.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap="RdBu")
    ax.scatter(X_te[y_te==0, 0], X_te[y_te==0, 1],
               c="steelblue", s=15, alpha=0.6, label="Class 0")
    ax.scatter(X_te[y_te==1, 0], X_te[y_te==1, 1],
               c="tomato",    s=15, alpha=0.6, label="Class 1")
    ax.set_title(f"{label}\\n{n_feat} features | Test acc: {acc_te:.3f}",
                 fontsize=8.5)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([]); ax.set_yticks([])

    results.append((label, degree, n_feat, acc_tr, acc_te))

plt.tight_layout()
plt.savefig("feature_engineering_boundary.png", dpi=120)

print(f"  {'Model':40s} {'Features':>8} {'Train':>8} {'Test':>8}")
print(f"  {'─'*68}")
for label, deg, n_feat, acc_tr, acc_te in results:
    status = ""
    if acc_te < 0.70:    status = "← UNDERFIT"
    elif acc_te > 0.95:  status = "← Great!"
    else:                status = "← Improving"
    print(f"  {label:40s} {n_feat:8d} {acc_tr:8.3f} {acc_te:8.3f}  {status}")

print()
print("  KEY LESSON:")
print("  The model class (Logistic Regression) never changed.")
print("  Only the FEATURES changed — and the model went from ~50%")
print("  (random) accuracy to near-perfect.")
print("  Feature engineering gives a linear model non-linear power.")
print()
print("  Plot saved → feature_engineering_boundary.png")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Feature Selection: Filter vs Wrapper vs Embedded": {
        "description": (
            "Compare all three families of feature selection on a dataset with "
            "many irrelevant features. Show that embedded (L1) and wrapper (RFE) "
            "outperform filter methods."
        ),
        "language": "python",
        "code": '''
import numpy as np
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt

# ── Sigmoid ───────────────────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,max_iter=500,C=1.0,penalty="l2",solver="lbfgs",random_state=None,class_weight=None):
        self.max_iter=max_iter; self.C=C; self.penalty=penalty; self.class_weight=class_weight
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        lam=1.0/self.C if self.C>0 else 0
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso alias ────────────────────────────────────────────────────────────────
Lasso = LogisticRegression   # used only as feature selector; coef_ attribute is shared

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X0=np.c_[np.cos(t),np.sin(t)]; X1=np.c_[1-np.cos(t),-np.sin(t)+0.5]
    X=np.vstack([X0,X1])+rng.normal(0,noise,(n_samples,2))
    y=np.hstack([np.zeros(n),np.ones(n)]).astype(int)
    return X,y

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,n_clusters_per_class=1,weights=None,flip_y=0.01,
                        random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_digits():
    """Synthetic digits-like dataset: 1797 samples, 64 features, 10 classes."""
    rng=np.random.default_rng(0); n_per=180; n_cls=10
    X_list=[]; y_list=[]
    for c in range(n_cls):
        proto=rng.uniform(0,16,(64,)); proto=np.clip(proto,0,16)
        samp=proto+rng.normal(0,3,(n_per,64)); samp=np.clip(samp,0,16)
        X_list.append(samp); y_list.append(np.full(n_per,c))
    X=np.vstack(X_list); y=np.hstack(y_list).astype(int)
    idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold / StratifiedKFold ───────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs if i<self.n_splits-1 else n]
            tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs if i<self.n_splits-1 else n:]])
            yield tr,te

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); classes=np.unique(y)
        rng=np.random.default_rng(self.random_state)
        folds=[[] for _ in range(self.n_splits)]
        for c in classes:
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)):
                folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i])
            tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()
def _tp_fp_fn_tn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); d=2*tp+fp+fn
    return float(2*tp/d) if d>0 else float(zero_division)
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); return np.array([[tn,fp],[fn,tp]])
def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c:3d}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv
        splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),
                 np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── NearestNeighbors ──────────────────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]; D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── PolynomialFeatures ────────────────────────────────────────────────────────
class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True):
        self.degree=degree; self.include_bias=include_bias
    def _combos(self,n_feat):
        from itertools import combinations_with_replacement
        combos=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: combos.append(())
            else: combos.extend(combinations_with_replacement(range(n_feat),d))
        return combos
    def fit(self,X,y=None):
        self._combos_=self._combos(X.shape[1])
        self.n_output_features_=len(self._combos_)
        return self
    def transform(self,X):
        cols=[]
        for combo in self._combos_:
            if len(combo)==0: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────
class PCA:
    def __init__(self,n_components=None): self.n_components=n_components
    def fit(self,X):
        self.mean_=X.mean(axis=0); Xc=X-self.mean_
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        tot=np.sum(s**2); self.explained_variance_ratio_=(s**2)/tot
        self.components_=Vt; self.singular_values_=s
        if self.n_components: self.components_=Vt[:self.n_components]
        return self
    def transform(self,X): return (X-self.mean_)@self.components_.T
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── MI feature selection ──────────────────────────────────────────────────────
def _discretise(arr,n_bins=12):
    p=np.unique(np.percentile(arr,np.linspace(0,100,n_bins+1)))
    return np.digitize(arr,p[1:-1])
def _mi(xd,yd):
    n=len(xd); mi=0.0
    for xv in np.unique(xd):
        for yv in np.unique(yd):
            nxy=np.sum((xd==xv)&(yd==yv))
            if nxy==0: continue
            mi+=(nxy/n)*np.log((nxy*n)/(np.sum(xd==xv)*np.sum(yd==yv)))
    return max(mi,0.0)
def mutual_info_classif(X,y,random_state=None):
    yd=y.astype(int)
    return np.array([_mi(_discretise(X[:,i]),yd) for i in range(X.shape[1])])
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]; Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_sp_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        result=self.score_func(X,y)
        sc=result if not isinstance(result,tuple) else result[0]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(sc)[::-1][:self.k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── Simple Decision Tree (for RF) ─────────────────────────────────────────────
class _DTree:
    def __init__(self,max_depth=6,min_samples=2,rng=None):
        self.max_depth=max_depth; self.min_samples=min_samples; self.rng=rng or np.random
        self.tree_=None
    def _gini(self,y):
        if len(y)==0: return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _best_split(self,X,y,feats):
        best={"gain":-1,"feat":None,"thr":None}
        g0=self._gini(y); n=len(y)
        for f in feats:
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_thrs=(vals[:-1]+vals[1:])/2
            # sample at most 8 thresholds to keep pure-Python trees fast
            thrs=all_thrs[np.linspace(0,len(all_thrs)-1,min(8,len(all_thrs))).astype(int)]
            for t in thrs:
                lm=X[:,f]<=t; rm=~lm
                if lm.sum()<self.min_samples or rm.sum()<self.min_samples: continue
                gain=g0-(lm.sum()/n*self._gini(y[lm])+rm.sum()/n*self._gini(y[rm]))
                if gain>best["gain"]: best={"gain":gain,"feat":f,"thr":t}
        return best
    def _build(self,X,y,depth):
        classes,counts=np.unique(y,return_counts=True)
        leaf={"leaf":True,"pred":classes[np.argmax(counts)]}
        if depth>=self.max_depth or len(y)<=self.min_samples or len(classes)==1:
            return leaf
        n_feats=max(1,int(np.sqrt(X.shape[1])))
        feats=self.rng.choice(X.shape[1],n_feats,replace=False)
        sp=self._best_split(X,y,feats)
        if sp["gain"]<=0: return leaf
        lm=X[:,sp["feat"]]<=sp["thr"]
        return {"leaf":False,"feat":sp["feat"],"thr":sp["thr"],
                "left":self._build(X[lm],y[lm],depth+1),
                "right":self._build(X[~lm],y[~lm],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _pred1(self,x,node):
        if node["leaf"]: return node["pred"]
        return self._pred1(x,node["left"] if x[node["feat"]]<=node["thr"] else node["right"])
    def predict(self,X): return np.array([self._pred1(x,self.tree_) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=15,max_depth=5,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self.feature_importances_=np.zeros(X.shape[1])
        self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); tree_rng=np.random.default_rng(int(rng.integers(0,1e9)))
            t=_DTree(max_depth=self.max_depth,rng=tree_rng).fit(X[idx],y[idx])
            self.estimators_.append(t)
        # Approximate feature importances via split frequency
        def _imp(node,imp):
            if node["leaf"]: return
            imp[node["feat"]]+=1; _imp(node["left"],imp); _imp(node["right"],imp)
        for t in self.estimators_: _imp(t.tree_,self.feature_importances_)
        s=self.feature_importances_.sum()
        if s>0: self.feature_importances_/=s
        return self
    def predict(self,X):
        votes=np.array([t.predict(X) for t in self.estimators_])
        return np.array([np.bincount(votes[:,i].astype(int),minlength=len(self.classes_)).argmax()
                         for i in range(len(X))])
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── RFE ───────────────────────────────────────────────────────────────────────
class RFE:
    def __init__(self,estimator,n_features_to_select=10,step=1):
        self.estimator=estimator; self.n_features_to_select=n_features_to_select; self.step=step
    def fit(self,X,y):
        n_feat=X.shape[1]; mask=np.ones(n_feat,dtype=bool); feats=np.arange(n_feat)
        while mask.sum()>self.n_features_to_select:
            est=_copy.deepcopy(self.estimator); est.fit(X[:,mask],y)
            importance=np.abs(est.coef_.ravel())
            remove=max(1,min(self.step,mask.sum()-self.n_features_to_select))
            worst=np.argsort(importance)[:remove]
            active=feats[mask]; mask[active[worst]]=False
        self._mask=mask; return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── SelectFromModel ────────────────────────────────────────────────────────────
class SelectFromModel:
    def __init__(self,estimator,max_features=None,threshold=None):
        self.estimator=estimator; self.max_features=max_features; self.threshold=threshold
    def fit(self,X,y):
        est=_copy.deepcopy(self.estimator); est.fit(X,y)
        if hasattr(est,"feature_importances_"): imp=est.feature_importances_
        elif hasattr(est,"coef_"): imp=np.abs(est.coef_.ravel())
        else: imp=np.ones(X.shape[1])
        k=self.max_features if self.max_features else X.shape[1]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(imp)[::-1][:k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

np.random.seed(42)

# ── Dataset: 5 truly informative features out of 50 ──────────────────────
X, y = make_classification(
    n_samples=400, n_features=50, n_informative=5,
    n_redundant=5, n_repeated=0, n_clusters_per_class=1,
    random_state=42
)
X = StandardScaler().fit_transform(X)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
k = 10    # select top-10 features for comparison

print("=" * 65)
print("  FEATURE SELECTION: FILTER vs WRAPPER vs EMBEDDED")
print("=" * 65)
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Truly informative features: 5 out of {X.shape[1]}")
print(f"  Selecting top k={k} features in each method")
print()

# ── Baseline: all 50 features ─────────────────────────────────────────────
base_scores = cross_val_score(
    LogisticRegression(max_iter=500), X, y, cv=cv, scoring="accuracy")
print(f"  BASELINE (all {X.shape[1]} features)")
print(f"    CV Accuracy: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
print()

# ── Filter: Mutual Information ────────────────────────────────────────────
pipe_filter = Pipeline([
    ("select", SelectKBest(mutual_info_classif, k=k)),
    ("clf",    LogisticRegression(max_iter=500)),
])
filter_scores = cross_val_score(pipe_filter, X, y, cv=cv, scoring="accuracy")
# Get selected features (fit once on full data for analysis)
selector_mi = SelectKBest(mutual_info_classif, k=k).fit(X, y)
mi_features  = np.where(selector_mi.get_support())[0]
print(f"  FILTER — Mutual Information (top {k})")
print(f"    Selected features (indices): {sorted(mi_features)}")
print(f"    CV Accuracy: {filter_scores.mean():.4f} ± {filter_scores.std():.4f}")
print()

# ── Wrapper: RFE ──────────────────────────────────────────────────────────
rfe_base = LogisticRegression(max_iter=500, C=1.0)
pipe_rfe  = Pipeline([
    ("select", RFE(rfe_base, n_features_to_select=k, step=5)),
    ("clf",    LogisticRegression(max_iter=500)),
])
rfe_scores = cross_val_score(pipe_rfe, X, y, cv=cv, scoring="accuracy")
rfe_sel = RFE(rfe_base, n_features_to_select=k, step=5).fit(X, y)
rfe_features = np.where(rfe_sel.get_support())[0]
print(f"  WRAPPER — Recursive Feature Elimination (top {k})")
print(f"    Selected features (indices): {sorted(rfe_features)}")
print(f"    CV Accuracy: {rfe_scores.mean():.4f} ± {rfe_scores.std():.4f}")
print()

# ── Embedded: L1 / Lasso ──────────────────────────────────────────────────
lasso_sel = SelectFromModel(
    LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=2000),
    max_features=k
)
pipe_l1 = Pipeline([
    ("select", lasso_sel),
    ("clf",    LogisticRegression(max_iter=500)),
])
l1_scores = cross_val_score(pipe_l1, X, y, cv=cv, scoring="accuracy")
lasso_sel.fit(X, y)
l1_features = np.where(lasso_sel.get_support())[0]
print(f"  EMBEDDED — L1 Logistic Regression (top {k})")
print(f"    Selected features (indices): {sorted(l1_features)}")
print(f"    CV Accuracy: {l1_scores.mean():.4f} ± {l1_scores.std():.4f}")
print()

# ── Embedded: Random Forest Importance ────────────────────────────────────
rf_sel = SelectFromModel(RandomForestClassifier(n_estimators=20,
                          random_state=0), max_features=k)
pipe_rf = Pipeline([
    ("select", rf_sel),
    ("clf",    LogisticRegression(max_iter=500)),
])
rf_scores = cross_val_score(pipe_rf, X, y, cv=cv, scoring="accuracy")
rf_sel.fit(X, y)
rf_features = np.where(rf_sel.get_support())[0]
print(f"  EMBEDDED — Random Forest Importance (top {k})")
print(f"    Selected features (indices): {sorted(rf_features)}")
print(f"    CV Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print()

# ── Summary table ─────────────────────────────────────────────────────────
methods = [
    ("All features (baseline)", base_scores),
    ("Filter (Mutual Info)",    filter_scores),
    ("Wrapper (RFE)",           rfe_scores),
    ("Embedded (L1)",           l1_scores),
    ("Embedded (RF Importance)",rf_scores),
]
print(f"  {'Method':30s} {'CV Acc':>8} {'Std':>8} {'vs Baseline':>12}")
print(f"  {'─'*62}")
base_mean = base_scores.mean()
for name, scores in methods:
    delta = scores.mean() - base_mean
    marker = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
    print(f"  {name:30s} {scores.mean():8.4f} {scores.std():8.4f} "
          f"  {delta:+.4f} {marker}")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · PCA Dimensionality Reduction": {
        "description": (
            "Apply PCA to high-dimensional data. Show the explained variance curve "
            "to find the optimal number of components, then compare model performance "
            "with and without PCA preprocessing."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
from scipy import stats as _sp_stats
import scipy.optimize as _sp_opt

# ── Sigmoid ───────────────────────────────────────────────────────────────────
def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# ── LogisticRegression ────────────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self,max_iter=500,C=1.0,penalty="l2",solver="lbfgs",random_state=None,class_weight=None):
        self.max_iter=max_iter; self.C=C; self.penalty=penalty; self.class_weight=class_weight
    def _sw(self,y):
        if self.class_weight=="balanced":
            n=len(y); cls,cnt=np.unique(y,return_counts=True)
            w={c:n/(len(cls)*k) for c,k in zip(cls,cnt)}
            return np.array([w[yi] for yi in y])
        return np.ones(len(y))
    def fit(self,X,y,sample_weight=None):
        n,d=X.shape; w0=np.zeros(d+1); sw=self._sw(y)
        if sample_weight is not None: sw=sw*sample_weight; sw/=sw.mean()
        lam=1.0/self.C if self.C>0 else 0
        def fg(w):
            p=np.clip(_sigmoid(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5*lam*np.sum(w[1:]**2)
            e=sw*(p-y); g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1)
        return self
    def predict_proba(self,X):
        p=_sigmoid(X@self.coef_.ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── Lasso alias ────────────────────────────────────────────────────────────────
Lasso = LogisticRegression   # used only as feature selector; coef_ attribute is shared

# ── Scalers ───────────────────────────────────────────────────────────────────
class StandardScaler:
    def fit(self,X):
        self.mean_=np.nanmean(X,axis=0); self.scale_=np.nanstd(X,axis=0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Data generators ───────────────────────────────────────────────────────────
def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X0=np.c_[np.cos(t),np.sin(t)]; X1=np.c_[1-np.cos(t),-np.sin(t)+0.5]
    X=np.vstack([X0,X1])+rng.normal(0,noise,(n_samples,2))
    y=np.hstack([np.zeros(n),np.ones(n)]).astype(int)
    return X,y

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        n_repeated=0,n_clusters_per_class=1,weights=None,flip_y=0.01,
                        random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(2,n_samples,p=weights) if weights else rng.integers(0,2,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*rng.uniform(0.8,1.5,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-n_redundant)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def load_digits():
    """Synthetic digits-like dataset: 1797 samples, 64 features, 10 classes."""
    rng=np.random.default_rng(0); n_per=180; n_cls=10
    X_list=[]; y_list=[]
    for c in range(n_cls):
        proto=rng.uniform(0,16,(64,)); proto=np.clip(proto,0,16)
        samp=proto+rng.normal(0,3,(n_per,64)); samp=np.clip(samp,0,16)
        X_list.append(samp); y_list.append(np.full(n_per,c))
    X=np.vstack(X_list); y=np.hstack(y_list).astype(int)
    idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
    class _D:
        def __init__(self,X,y): self.data=X; self.target=y
    return _D(X,y)

# ── train_test_split ──────────────────────────────────────────────────────────
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

# ── KFold / StratifiedKFold ───────────────────────────────────────────────────
class KFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); idx=np.arange(n)
        if self.shuffle: np.random.default_rng(self.random_state).shuffle(idx)
        fs=n//self.n_splits
        for i in range(self.n_splits):
            te=idx[i*fs:(i+1)*fs if i<self.n_splits-1 else n]
            tr=np.concatenate([idx[:i*fs],idx[(i+1)*fs if i<self.n_splits-1 else n:]])
            yield tr,te

class StratifiedKFold:
    def __init__(self,n_splits=5,shuffle=False,random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self,X,y=None):
        n=len(X); classes=np.unique(y)
        rng=np.random.default_rng(self.random_state)
        folds=[[] for _ in range(self.n_splits)]
        for c in classes:
            idx=np.where(y==c)[0]
            if self.shuffle: rng.shuffle(idx)
            for i,chunk in enumerate(np.array_split(idx,self.n_splits)):
                folds[i].extend(chunk.tolist())
        for i in range(self.n_splits):
            te=np.array(folds[i])
            tr=np.array([x for j,f in enumerate(folds) if j!=i for x in f])
            yield tr,te

# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()
def _tp_fp_fn_tn(yt,yp):
    tp=((yp==1)&(yt==1)).sum(); fp=((yp==1)&(yt==0)).sum()
    fn=((yp==0)&(yt==1)).sum(); tn=((yp==0)&(yt==0)).sum()
    return tp,fp,fn,tn
def f1_score(yt,yp,zero_division=0):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); d=2*tp+fp+fn
    return float(2*tp/d) if d>0 else float(zero_division)
def confusion_matrix(yt,yp):
    tp,fp,fn,tn=_tp_fp_fn_tn(yt,yp); return np.array([[tn,fp],[fn,tp]])
def classification_report(yt,yp,**kw):
    classes=np.unique(np.concatenate([yt,yp])); lines=[]
    for c in classes:
        tp=((yp==c)&(yt==c)).sum(); fp=((yp==c)&(yt!=c)).sum(); fn=((yp!=c)&(yt==c)).sum()
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*p*r/(p+r) if p+r>0 else 0; sup=(yt==c).sum()
        lines.append(f"  class {c:3d}: precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  support={sup}")
    return chr(10).join(lines)

# ── cross_val_score ───────────────────────────────────────────────────────────
def cross_val_score(estimator,X,y,cv=5,scoring="accuracy"):
    if hasattr(cv,"split"): splits=list(cv.split(X,y))
    else:
        n=len(y); fs=n//cv
        splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),
                 np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr]); yp=est.predict(X[te])
        if scoring=="accuracy": scores.append((yp==y[te]).mean())
        elif scoring=="f1": scores.append(f1_score(y[te],yp,zero_division=0))
    return np.array(scores)

# ── NearestNeighbors ──────────────────────────────────────────────────────────
class NearestNeighbors:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X): self._X=X.copy(); return self
    def kneighbors(self,X=None):
        if X is None: X=self._X
        D=[]; I=[]
        for xi in X:
            d=np.sqrt(np.sum((self._X-xi)**2,axis=1))
            idx=np.argsort(d)[:self.n_neighbors]; D.append(d[idx]); I.append(idx)
        return np.array(D),np.array(I)

# ── PolynomialFeatures ────────────────────────────────────────────────────────
class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=True):
        self.degree=degree; self.include_bias=include_bias
    def _combos(self,n_feat):
        from itertools import combinations_with_replacement
        combos=[]
        for d in range(0 if self.include_bias else 1,self.degree+1):
            if d==0: combos.append(())
            else: combos.extend(combinations_with_replacement(range(n_feat),d))
        return combos
    def fit(self,X,y=None):
        self._combos_=self._combos(X.shape[1])
        self.n_output_features_=len(self._combos_)
        return self
    def transform(self,X):
        cols=[]
        for combo in self._combos_:
            if len(combo)==0: cols.append(np.ones(len(X)))
            else:
                col=np.ones(len(X))
                for i in combo: col=col*X[:,i]
                cols.append(col)
        return np.column_stack(cols)
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────
class PCA:
    def __init__(self,n_components=None): self.n_components=n_components
    def fit(self,X):
        self.mean_=X.mean(axis=0); Xc=X-self.mean_
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        tot=np.sum(s**2); self.explained_variance_ratio_=(s**2)/tot
        self.components_=Vt; self.singular_values_=s
        if self.n_components: self.components_=Vt[:self.n_components]
        return self
    def transform(self,X): return (X-self.mean_)@self.components_.T
    def fit_transform(self,X): return self.fit(X).transform(X)

# ── Pipeline ──────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self,steps): self.steps=steps; self.named_steps=dict(steps)
    def fit(self,X,y=None):
        Xt=X
        for nm,step in self.steps[:-1]:
            try: Xt=step.fit_transform(Xt,y)
            except TypeError: step.fit(Xt); Xt=step.transform(Xt)
        self.steps[-1][1].fit(Xt,y); return self
    def transform(self,X):
        Xt=X
        for _,step in self.steps[:-1]: Xt=step.transform(Xt)
        return Xt
    def predict(self,X): return self.steps[-1][1].predict(self.transform(X))
    def predict_proba(self,X): return self.steps[-1][1].predict_proba(self.transform(X))
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── MI feature selection ──────────────────────────────────────────────────────
def _discretise(arr,n_bins=12):
    p=np.unique(np.percentile(arr,np.linspace(0,100,n_bins+1)))
    return np.digitize(arr,p[1:-1])
def _mi(xd,yd):
    n=len(xd); mi=0.0
    for xv in np.unique(xd):
        for yv in np.unique(yd):
            nxy=np.sum((xd==xv)&(yd==yv))
            if nxy==0: continue
            mi+=(nxy/n)*np.log((nxy*n)/(np.sum(xd==xv)*np.sum(yd==yv)))
    return max(mi,0.0)
def mutual_info_classif(X,y,random_state=None):
    yd=y.astype(int)
    return np.array([_mi(_discretise(X[:,i]),yd) for i in range(X.shape[1])])
def f_classif(X,y):
    grps=[X[y==c] for c in np.unique(y)]; Fv,pv=[],[]
    for j in range(X.shape[1]):
        F,p=_sp_stats.f_oneway(*[g[:,j] for g in grps]); Fv.append(F); pv.append(p)
    return np.array(Fv),np.array(pv)

class SelectKBest:
    def __init__(self,score_func=None,k=10): self.score_func=score_func or f_classif; self.k=k
    def fit(self,X,y):
        result=self.score_func(X,y)
        sc=result if not isinstance(result,tuple) else result[0]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(sc)[::-1][:self.k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── Simple Decision Tree (for RF) ─────────────────────────────────────────────
class _DTree:
    def __init__(self,max_depth=6,min_samples=2,rng=None):
        self.max_depth=max_depth; self.min_samples=min_samples; self.rng=rng or np.random
        self.tree_=None
    def _gini(self,y):
        if len(y)==0: return 0
        _,c=np.unique(y,return_counts=True); p=c/c.sum(); return 1-np.sum(p**2)
    def _best_split(self,X,y,feats):
        best={"gain":-1,"feat":None,"thr":None}
        g0=self._gini(y); n=len(y)
        for f in feats:
            vals=np.unique(X[:,f])
            if len(vals)<2: continue
            all_thrs=(vals[:-1]+vals[1:])/2
            # sample at most 8 thresholds to keep pure-Python trees fast
            thrs=all_thrs[np.linspace(0,len(all_thrs)-1,min(8,len(all_thrs))).astype(int)]
            for t in thrs:
                lm=X[:,f]<=t; rm=~lm
                if lm.sum()<self.min_samples or rm.sum()<self.min_samples: continue
                gain=g0-(lm.sum()/n*self._gini(y[lm])+rm.sum()/n*self._gini(y[rm]))
                if gain>best["gain"]: best={"gain":gain,"feat":f,"thr":t}
        return best
    def _build(self,X,y,depth):
        classes,counts=np.unique(y,return_counts=True)
        leaf={"leaf":True,"pred":classes[np.argmax(counts)]}
        if depth>=self.max_depth or len(y)<=self.min_samples or len(classes)==1:
            return leaf
        n_feats=max(1,int(np.sqrt(X.shape[1])))
        feats=self.rng.choice(X.shape[1],n_feats,replace=False)
        sp=self._best_split(X,y,feats)
        if sp["gain"]<=0: return leaf
        lm=X[:,sp["feat"]]<=sp["thr"]
        return {"leaf":False,"feat":sp["feat"],"thr":sp["thr"],
                "left":self._build(X[lm],y[lm],depth+1),
                "right":self._build(X[~lm],y[~lm],depth+1)}
    def fit(self,X,y): self.tree_=self._build(X,y,0); return self
    def _pred1(self,x,node):
        if node["leaf"]: return node["pred"]
        return self._pred1(x,node["left"] if x[node["feat"]]<=node["thr"] else node["right"])
    def predict(self,X): return np.array([self._pred1(x,self.tree_) for x in X])

class RandomForestClassifier:
    def __init__(self,n_estimators=15,max_depth=5,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.random_state=random_state; self.estimators_=[]
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X)
        self.classes_=np.unique(y); self.feature_importances_=np.zeros(X.shape[1])
        self.estimators_=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True); tree_rng=np.random.default_rng(int(rng.integers(0,1e9)))
            t=_DTree(max_depth=self.max_depth,rng=tree_rng).fit(X[idx],y[idx])
            self.estimators_.append(t)
        # Approximate feature importances via split frequency
        def _imp(node,imp):
            if node["leaf"]: return
            imp[node["feat"]]+=1; _imp(node["left"],imp); _imp(node["right"],imp)
        for t in self.estimators_: _imp(t.tree_,self.feature_importances_)
        s=self.feature_importances_.sum()
        if s>0: self.feature_importances_/=s
        return self
    def predict(self,X):
        votes=np.array([t.predict(X) for t in self.estimators_])
        return np.array([np.bincount(votes[:,i].astype(int),minlength=len(self.classes_)).argmax()
                         for i in range(len(X))])
    def score(self,X,y): return (self.predict(X)==y).mean()

# ── RFE ───────────────────────────────────────────────────────────────────────
class RFE:
    def __init__(self,estimator,n_features_to_select=10,step=1):
        self.estimator=estimator; self.n_features_to_select=n_features_to_select; self.step=step
    def fit(self,X,y):
        n_feat=X.shape[1]; mask=np.ones(n_feat,dtype=bool); feats=np.arange(n_feat)
        while mask.sum()>self.n_features_to_select:
            est=_copy.deepcopy(self.estimator); est.fit(X[:,mask],y)
            importance=np.abs(est.coef_.ravel())
            remove=max(1,min(self.step,mask.sum()-self.n_features_to_select))
            worst=np.argsort(importance)[:remove]
            active=feats[mask]; mask[active[worst]]=False
        self._mask=mask; return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

# ── SelectFromModel ────────────────────────────────────────────────────────────
class SelectFromModel:
    def __init__(self,estimator,max_features=None,threshold=None):
        self.estimator=estimator; self.max_features=max_features; self.threshold=threshold
    def fit(self,X,y):
        est=_copy.deepcopy(self.estimator); est.fit(X,y)
        if hasattr(est,"feature_importances_"): imp=est.feature_importances_
        elif hasattr(est,"coef_"): imp=np.abs(est.coef_.ravel())
        else: imp=np.ones(X.shape[1])
        k=self.max_features if self.max_features else X.shape[1]
        self._mask=np.zeros(X.shape[1],dtype=bool)
        self._mask[np.argsort(imp)[::-1][:k]]=True
        return self
    def transform(self,X): return X[:,self._mask]
    def fit_transform(self,X,y=None):
        if y is None: y=np.zeros(len(X),dtype=int)
        return self.fit(X,y).transform(X)
    def get_support(self): return self._mask

np.random.seed(0)

# ── Load digits dataset (64 features = 8×8 pixels) ────────────────────────
digits = load_digits()
X, y   = digits.data, digits.target
X = StandardScaler().fit_transform(X)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print("=" * 60)
print("  PCA DIMENSIONALITY REDUCTION")
print("=" * 60)
print(f"  Dataset : Digits  ({X.shape[0]} samples, {X.shape[1]} features)")
print(f"  Classes : 0-9 (10 classes)")
print()

# ── Fit PCA and analyse explained variance ────────────────────────────────
pca_full = PCA().fit(X)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)

# Find how many components for 90%, 95%, 99%
thresholds = [0.80, 0.90, 0.95, 0.99]
print("  Explained Variance Thresholds:")
print(f"  {'Threshold':>12} | {'Components needed':>18} | "
      f"{'Compression':>12}")
print(f"  {'─'*50}")
for thr in thresholds:
    n_comp = np.searchsorted(cum_var, thr) + 1
    compr  = 1 - n_comp / X.shape[1]
    print(f"  {thr*100:11.0f}% | {n_comp:18d} | {compr*100:11.1f}% reduction")
print()

# ── Compare accuracy vs number of PCA components ─────────────────────────
n_comp_list = [2, 5, 10, 20, 30, 40, 50, 64]
acc_list    = []

print("  Accuracy vs Number of PCA Components:")
print(f"  {'Components':>12} | {'% Var':>8} | {'CV Accuracy':>12} | Notes")
print(f"  {'─'*60}")

for n_comp in n_comp_list:
    if n_comp > X.shape[1]:
        continue
    pipe = Pipeline([
        ("pca", PCA(n_components=n_comp)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    var_explained = cum_var[n_comp - 1]
    acc_list.append((n_comp, var_explained, scores.mean()))

    note = ""
    if n_comp == 2:    note = "← good for visualisation only"
    if n_comp == 20:   note = "← ~95% variance captured"
    if n_comp == 64:   note = "← all features (baseline)"
    print(f"  {n_comp:12d} | {var_explained*100:7.1f}% | "
          f"{scores.mean():12.4f} | {note}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PCA: Explained Variance & Classification Accuracy",
             fontsize=12, fontweight="bold")

# Left: cumulative explained variance
ax1 = axes[0]
ax1.plot(range(1, len(cum_var)+1), cum_var * 100, "steelblue", lw=2)
for thr in [80, 90, 95, 99]:
    n = np.searchsorted(cum_var, thr/100) + 1
    ax1.axhline(thr, color="gray", linestyle="--", alpha=0.6)
    ax1.axvline(n,   color="gray", linestyle="--", alpha=0.6)
    ax1.annotate(f"{thr}%\\n(n={n})", xy=(n, thr), xytext=(n+2, thr-7),
                 fontsize=8, color="gray")
ax1.set_xlabel("Number of Principal Components")
ax1.set_ylabel("Cumulative Explained Variance (%)")
ax1.set_title("Scree Plot: Variance vs Components")
ax1.grid(alpha=0.3)

# Right: accuracy vs components
ax2 = axes[1]
comps = [x[0] for x in acc_list]
accs  = [x[2] for x in acc_list]
ax2.plot(comps, accs, "tomato", lw=2, marker="o", ms=6)
ax2.axhline(accs[-1], color="gray", linestyle="--",
            label=f"Baseline (all {X.shape[1]} features): {accs[-1]:.4f}")
ax2.set_xlabel("Number of Principal Components")
ax2.set_ylabel("CV Accuracy")
ax2.set_title("Classification Accuracy vs Components")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pca_analysis.png", dpi=120)
print()
print("  Plot saved → pca_analysis.png")
print()
print("  KEY TAKEAWAYS:")
print("  - Most variance is captured by very few components")
print("  - Performance often matches or EXCEEDS the full-feature baseline")
print("    with far fewer dimensions (because noise is discarded)")
print("  - The 'elbow' in the scree plot guides the component choice")
''',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    """Return all content for this topic module — single source of truth."""
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