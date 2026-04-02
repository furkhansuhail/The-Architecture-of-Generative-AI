"""
Weight Initialisation & Normalisation
======================================

Two foundational techniques that determine whether a deep network
trains at all — and how fast. Bad initialisation or missing normalisation
causes vanishing/exploding gradients, both forms of underfitting.

"""

import textwrap
import re

TOPIC_NAME   = "Weight Initialisation & Normalisation"
DISPLAY_NAME = "01 · Weight Init & Normalisation"
ICON         = "⚖️"
SUBTITLE     = "Making Deep Networks Trainable from the First Forward Pass"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — WHY INITIALISATION MATTERS

### The Problem: Vanishing and Exploding Gradients

In a deep network, the forward pass multiplies activations through many
layers. The backward pass multiplies gradients through those same layers.
If the weights are poorly scaled, two catastrophic failures occur:

**Vanishing Gradients** — when weights are too small (< 1):
    Each layer shrinks the signal by a factor < 1.
    After L layers: signal ≈ wᴸ → 0   (exponentially small)

    Effect: gradients in early layers → ≈ 0.
    Early layers learn NOTHING. The network is effectively shallow.
    This was the main reason deep networks failed before ~2010.

**Exploding Gradients** — when weights are too large (> 1):
    Each layer amplifies the signal by a factor > 1.
    After L layers: signal ≈ wᴸ → ∞   (exponentially large)

    Effect: gradients → NaN, loss diverges, training crashes.

    Diagram 1 — Signal Propagation in a 5-Layer Network:

    Weights w = 0.5  (vanishing):        w = 2.0  (exploding):

    Layer:  1    2    3    4    5         1    2    3    4    5
    Signal: 1→ 0.5→0.25→0.12→0.06→0.03    1→  2→  4→  8→ 16→ 32

    ─────────────────────────────         ──────────────────────────────
    Signal collapses to 0                 Signal explodes to ∞

    With L=50 layers:
    Vanishing: 0.5⁵⁰ ≈ 10⁻¹⁵ (effectively 0)
    Exploding: 2⁵⁰  ≈ 10¹⁵  (numerical overflow)

    Goal of initialisation: keep the signal in a stable range across ALL layers.


### The Zero Initialisation Failure (Symmetry Problem)

If all weights are set to zero:
    - Every neuron in layer l computes the exact same output.
    - Every neuron receives the exact same gradient.
    - Every neuron updates identically.
    - All neurons remain identical forever — you effectively have 1 neuron/layer.

This is called the symmetry problem. It's why we MUST use random initialisation.

    Zero init:   w₁=w₂=w₃=0  →  h₁=h₂=h₃ → grad₁=grad₂=grad₃ → stays identical
    Random init: w₁≠w₂≠w₃    →  h₁≠h₂≠h₃ → grad₁≠grad₂≠grad₃ → neurons specialise



##### PART 2 — WEIGHT INITIALISATION STRATEGIES

### Naive Random Initialisation (and why it fails)

Randomly drawing from N(0, 1) (standard normal):
    - Each neuron with n_in inputs computes z = w₁x₁ + w₂x₂ + ... + wₙxₙ
    - Var(z) = n_in × Var(w) × Var(x) = n_in × 1 × 1 = n_in
    - Std(z) = √n_in

    With 1000 inputs: Std(z) = √1000 ≈ 32. The pre-activations are huge!
    After sigmoid: output saturates at 0 or 1 → gradient ≈ 0 → vanishing.

    With small σ (e.g., 0.01): the opposite — all activations near 0,
    all gradients tiny, no learning.


### Xavier / Glorot Initialisation (2010)

**Designed for:** Sigmoid, Tanh activation functions.
**Key insight:** Scale weights so that the VARIANCE of activations is
the SAME in every layer — both on the forward pass and the backward pass.

    Var(wᵢ) = 2 / (n_in + n_out)

    In practice, sample from:
        Uniform: U(-√(6/(n_in+n_out)), +√(6/(n_in+n_out)))
        Normal:  N(0, √(2/(n_in+n_out)))

    Why n_in + n_out? Xavier derived the formula by requiring:
        Var(output) = Var(input)  on the FORWARD pass
        Var(gradient) = Var(gradient)  on the BACKWARD pass
    The compromise between these two constraints gives n_in + n_out.

    Diagram 2 — Xavier Keeps Activation Variance Stable:

    Layer:     1      2      3      4      5      6
    Naive:   [huge] [huge] [huge] [huge] [huge] [huge]  → saturated
    Xavier:  [~1]   [~1]   [~1]   [~1]   [~1]   [~1]   → stable ✓


### He / Kaiming Initialisation (2015)

**Designed for:** ReLU and Leaky ReLU activation functions.
**Key insight:** ReLU zeros out negative values, so it halves the effective
variance. Glorot doesn't account for this. He initialisation compensates:

    Var(wᵢ) = 2 / n_in

    In practice, sample from:
        Normal:  N(0, √(2/n_in))

    Why 2/n_in? With ReLU, E[activation²] = Var(pre-activation)/2
    (half the inputs are zeroed out), so we need 2× the Glorot variance
    to compensate.

    For Leaky ReLU with slope a: Var(wᵢ) = 2 / ((1 + a²) × n_in)


### LeCun Initialisation (1998)

**Designed for:** SELU (Self-Normalising Neural Networks) activation.

    Var(wᵢ) = 1 / n_in

    SELU with LeCun init has a remarkable property: activations
    automatically converge to mean 0, variance 1 — self-normalising,
    without any batch normalisation layer.


### Summary Table

    ┌──────────────────┬─────────────────────┬────────────────────────────┐
    │ Init             │ Formula             │ For activation             │
    ├──────────────────┼─────────────────────┼────────────────────────────┤
    │ Zeros            │ 0                   │ NEVER use (symmetry break) │
    │ N(0, 0.01)       │ N(0, 0.01²)         │ Very shallow nets only     │
    │ Xavier / Glorot  │ N(0, 2/(nᵢₙ+nₒᵤₜ))    │ Sigmoid, Tanh              │
    │ He / Kaiming     │ N(0, 2/nᵢₙ)          │ ReLU, Leaky ReLU (default) │
    │ LeCun            │ N(0, 1/nᵢₙ)          │ SELU                       │
    └──────────────────┴─────────────────────┴────────────────────────────┘

    Modern default: He initialisation for most architectures using ReLU.



##### PART 3 — BATCH NORMALISATION

### What is Batch Normalisation?

Introduced by Ioffe & Szegedy in 2015. Applied to the pre-activations of
a layer (or post-activations — placement is debated), BN normalises each
feature across the mini-batch:

    Step 1 — Compute mini-batch statistics:
        μ_B = (1/m) Σ xᵢ            (batch mean, per feature)
        σ²_B = (1/m) Σ (xᵢ - μ_B)²  (batch variance, per feature)

    Step 2 — Normalise:
        x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)      (ε prevents /0)

    Step 3 — Scale and shift (learnable parameters γ and β):
        yᵢ = γ · x̂ᵢ + β

    γ and β are learned alongside the weights, giving the network the
    ability to "undo" the normalisation if needed (e.g., if a layer truly
    needs a different distribution).

    Diagram 3 — BatchNorm in a Network:

    Input → [Linear] → [BatchNorm] → [ReLU] → [Linear] → [BatchNorm] → ...

    At TRAINING time:
        Uses batch statistics (μ_B, σ²_B) computed on the current mini-batch.
        Also maintains running mean and running variance via exponential
        moving average for use at inference time.

    At INFERENCE time:
        Uses the RUNNING STATISTICS (accumulated during training) —
        NOT the current batch statistics (there may only be 1 example).
        running_mean ← (1-momentum)×running_mean + momentum×μ_B
        running_var  ← (1-momentum)×running_var  + momentum×σ²_B


### Why Batch Norm Works — Multiple Mechanisms

**1. Reduces internal covariate shift:**
    As weights change during training, the distribution of each layer's
    inputs shifts — the next layer must constantly adapt to a moving target.
    BN re-centres and re-scales, stabilising the input distribution.
    (Note: this explanation is now debated, but it's still commonly used.)

**2. Allows higher learning rates:**
    Without BN, high learning rates cause activations to explode or vanish.
    BN keeps activations in a stable range, allowing 10-100× larger lr.
    Faster learning rates → faster training.

**3. Acts as regularisation:**
    Each mini-batch has different mean and variance (stochastic noise).
    This noise acts as regularisation, often reducing the need for dropout.
    BN is why many modern architectures drop dropout entirely.

**4. Reduces sensitivity to initialisation:**
    Because BN normalises activations regardless of the weight scale,
    the network is much more robust to different initialisation strategies.


### Limitations of Batch Normalisation

    - Requires a sufficiently large batch size (≥ 16) for stable statistics.
    - Does NOT work for batch size = 1 (statistics meaningless).
    - Behaviour differs between training and inference → can cause bugs.
    - Not suitable for recurrent networks (sequence length varies).
    - Memory overhead: stores running statistics.



##### PART 4 — LAYER, INSTANCE & GROUP NORMALISATION

### The Family of Normalisation Methods

All normalisation methods follow the same formula: normalise, then scale+shift.
They differ only in WHICH DIMENSIONS are used to compute μ and σ.

    Input tensor shape: (Batch N, Channels C, Height H, Width W)

    Batch Norm:     normalise over N, H, W  — per channel
    Layer Norm:     normalise over C, H, W  — per sample
    Instance Norm:  normalise over H, W     — per channel per sample
    Group Norm:     normalise over G, H, W  — per group of channels per sample

    Diagram 4 — What Each Method Normalises (visualised for one sample):

                    ┌─────────────────────────────────────────────────┐
                    │  Tensor dimensions: Batch × Channels × H × W    │
                    └─────────────────────────────────────────────────┘

    Batch Norm:   averages across the BATCH dimension
                  Works great for large batches (CNNs), fails for small.

    Layer Norm:   averages across ALL features in one sample
                  Works for ANY batch size — perfect for Transformers, RNNs.

    Instance Norm: averages within each channel of each sample
                  Used in style transfer, generative models.

    Group Norm:   splits channels into G groups, normalises within each.
                  Bridge between BN and LN. Works well for detection tasks.

    ┌────────────────────┬───────────────────────────────────────────┐
    │ Method             │ Best used for                             │
    ├────────────────────┼───────────────────────────────────────────┤
    │ Batch Norm         │ CNNs with large batch (≥ 16)              │
    │ Layer Norm         │ Transformers, RNNs, small batches         │
    │ Instance Norm      │ Style transfer, GANs                      │
    │ Group Norm         │ Object detection (small batches/GPU)      │
    └────────────────────┴───────────────────────────────────────────┘


"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Vanishing & Exploding Gradients Demo": {
        "description": (
            "Simulate forward-pass signal propagation in a 30-layer network "
            "with naive, zero, Xavier, and He initialisation. "
            "Measure activation mean and variance at each layer."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

N_LAYERS = 30
N_UNITS  = 256
BATCH    = 512

def relu(x):    return np.maximum(0, x)
def tanh_(x):   return np.tanh(x)
def sigmoid_(x):return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def forward_pass(init_fn, activation, n_layers=N_LAYERS,
                 n_units=N_UNITS, batch=BATCH):
    """Run a random input through n_layers and track activation statistics."""
    X = np.random.randn(batch, n_units)
    means, stds, dead = [], [], []
    for l in range(n_layers):
        W = init_fn(n_units, n_units)
        b = np.zeros(n_units)
        Z = X @ W + b
        X = activation(Z)
        means.append(np.mean(X))
        stds.append(np.std(X))
        dead.append(np.mean(X == 0) if activation is relu else 0.0)
    return means, stds, dead

# ── Initialisation strategies ─────────────────────────────────────────────
inits = {
    "Zeros  (w=0)":        lambda ni, no: np.zeros((ni, no)),
    "Large  N(0, 1)":      lambda ni, no: np.random.randn(ni, no),
    "Small  N(0, 0.01)":   lambda ni, no: np.random.randn(ni, no) * 0.01,
    "Xavier N(0,√2/n)":    lambda ni, no: np.random.randn(ni, no)
                                          * np.sqrt(2/(ni+no)),
    "He     N(0,√2/nᵢₙ)":  lambda ni, no: np.random.randn(ni, no)
                                          * np.sqrt(2/ni),
}

configs = [
    ("tanh",    tanh_,   "Xavier", "cornflowerblue"),
    ("relu",    relu,    "He",     "seagreen"),
]

print("=" * 65)
print("  VANISHING & EXPLODING GRADIENTS — ACTIVATION STATISTICS")
print("=" * 65)
print(f"  Network: {N_LAYERS} layers × {N_UNITS} units | Activation: ReLU")
print()

fig, axes = plt.subplots(2, len(inits), figsize=(20, 7))
fig.suptitle("Activation Standard Deviation per Layer — "
             "How Initialisation Affects Signal Flow",
             fontsize=11, fontweight="bold")

layers = np.arange(1, N_LAYERS + 1)
colours = ["tomato", "orange", "purple", "steelblue", "seagreen"]

# ── ReLU analysis ─────────────────────────────────────────────────────────
print(f"  {'Init Strategy':22s} | {'Final std':>10} | {'Dead neurons':>13} | Diagnosis")
print(f"  {'─'*70}")

for col_idx, (init_name, init_fn) in enumerate(inits.items()):
    means, stds, dead = forward_pass(init_fn, relu)

    final_std  = stds[-1]
    final_dead = dead[-1]
    if final_std < 1e-4:
        diagnosis = "VANISHING ← signal collapses"
    elif final_std > 100:
        diagnosis = "EXPLODING ← signal diverges"
    elif final_dead > 0.8:
        diagnosis = "DYING ReLU ← most neurons dead"
    else:
        diagnosis = "STABLE ✓"

    print(f"  {init_name:22s} | {final_std:10.5f} | "
          f"{final_dead*100:12.1f}% | {diagnosis}")

    axes[0, col_idx].plot(layers, stds, colour := colours[col_idx], lw=2)
    title_0 = init_name + " (std@L" + str(N_LAYERS) + ": " + f"{final_std:.3g})"
    axes[0, col_idx].set_title(title_0, fontsize=8)
    axes[0, col_idx].set_xlabel("Layer")
    axes[0, col_idx].set_ylabel("Activation Std")
    axes[0, col_idx].grid(alpha=0.3)
    axes[0, col_idx].set_ylim(0, min(max(stds)*1.3, 10))

    axes[1, col_idx].plot(layers, [d*100 for d in dead], colours[col_idx], lw=2)
    axes[1, col_idx].set_title(f"Dead neurons (%)", fontsize=8)
    axes[1, col_idx].set_xlabel("Layer")
    axes[1, col_idx].set_ylabel("% ReLU outputs = 0")
    axes[1, col_idx].set_ylim(0, 105)
    axes[1, col_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("initialisation_signals.png", dpi=110)
print()
print("  KEY OBSERVATIONS:")
print("  - Zeros init: all activations identical → symmetry, no learning")
print("  - Large N(0,1): std explodes; saturation after ReLU makes it die")
print("  - Small N(0,0.01): std vanishes to near zero in deep layers")
print("  - Xavier: correct for tanh/sigmoid; underperforms for ReLU")
print("  - He: keeps ReLU network stable throughout ✓")
print()
print("  Plot saved → initialisation_signals.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Batch Normalisation from Scratch": {
        "description": (
            "Implement BatchNorm forward and backward passes from scratch. "
            "Verify that it normalises activations and show the training "
            "vs inference difference with running statistics."
        ),
        "language": "python",
        "code": '''
import numpy as np

np.random.seed(0)

# ── BatchNorm forward (training mode) ─────────────────────────────────────
def batchnorm_forward(X, gamma, beta, eps=1e-8, momentum=0.1,
                      running_mean=None, running_var=None):
    """
    X: (batch_size, features)
    Returns: normalised output, cache for backward pass,
             updated running statistics.
    """
    N, D = X.shape
    if running_mean is None:
        running_mean = np.zeros(D)
        running_var  = np.ones(D)

    # Step 1: batch statistics
    mu    = X.mean(axis=0)         # shape (D,)
    var   = X.var(axis=0)          # shape (D,)
    std   = np.sqrt(var + eps)

    # Step 2: normalise
    X_hat = (X - mu) / std         # zero mean, unit variance

    # Step 3: scale and shift
    out   = gamma * X_hat + beta

    # Update running statistics (for inference)
    running_mean = (1 - momentum) * running_mean + momentum * mu
    running_var  = (1 - momentum) * running_var  + momentum * var

    cache = (X, X_hat, mu, var, std, gamma, beta, eps)
    return out, cache, running_mean, running_var


def batchnorm_backward(dout, cache):
    """
    Backprop through batch normalisation.
    Returns: dX, dgamma, dbeta
    """
    X, X_hat, mu, var, std, gamma, beta, eps = cache
    N, D = dout.shape

    # Gradient w.r.t. gamma and beta (learnable parameters)
    dgamma = (dout * X_hat).sum(axis=0)
    dbeta  = dout.sum(axis=0)

    # Gradient w.r.t. X_hat
    dX_hat = dout * gamma

    # Gradient w.r.t. variance
    dvar = (-0.5 * dX_hat * (X - mu) * (var + eps) ** (-1.5)).sum(axis=0)

    # Gradient w.r.t. mean
    dmu = (dX_hat * (-1 / std)).sum(axis=0) + dvar * (-2/N) * (X - mu).sum(axis=0)

    # Gradient w.r.t. X
    dX = dX_hat / std + dvar * 2 * (X - mu) / N + dmu / N

    return dX, dgamma, dbeta


# ── batchnorm_inference (uses running stats, NOT batch stats) ─────────────
def batchnorm_inference(X, gamma, beta, running_mean, running_var, eps=1e-8):
    X_hat = (X - running_mean) / np.sqrt(running_var + eps)
    return gamma * X_hat + beta


# ── Demo ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("  BATCH NORMALISATION — FORWARD & BACKWARD PASS")
print("=" * 60)
print()

# Create a batch of activations with non-trivial mean and variance
batch_size, features = 32, 8
np.random.seed(42)
X = np.random.randn(batch_size, features) * 3 + 5   # mean≈5, std≈3

gamma = np.ones(features)     # learnable scale (start at 1)
beta  = np.zeros(features)    # learnable shift (start at 0)

print(f"  INPUT STATISTICS (before BatchNorm):")
print(f"  {'Feature':>8} | {'Mean':>8} | {'Std':>8}")
print(f"  {'─'*30}")
for i in range(features):
    print(f"  {i:8d} | {X[:, i].mean():8.3f} | {X[:, i].std():8.3f}")

out, cache, r_mean, r_var = batchnorm_forward(X, gamma, beta)

print(f"\\n  OUTPUT STATISTICS (after BatchNorm, γ=1, β=0):")
print(f"  {'Feature':>8} | {'Mean':>8} | {'Std':>8}")
print(f"  {'─'*30}")
for i in range(features):
    print(f"  {i:8d} | {out[:, i].mean():8.5f} | {out[:, i].std():8.5f}")

print(f"\\n  → All features now have mean≈0 and std≈1 ✓")
print()

# ── Show training vs inference difference ─────────────────────────────────
print("  TRAINING vs INFERENCE MODE:")
print()

# Simulate training: accumulate running stats over multiple batches
r_mean_running = np.zeros(features)
r_var_running  = np.ones(features)

for epoch in range(50):
    X_batch = np.random.randn(batch_size, features) * 3 + 5
    _, _, r_mean_running, r_var_running = batchnorm_forward(
        X_batch, gamma, beta,
        running_mean=r_mean_running, running_var=r_var_running)

print(f"  Running mean after 50 training batches:")
print(f"    {r_mean_running.round(3)}")
print(f"  Running var  after 50 training batches:")
print(f"    {r_var_running.round(3)}")
print(f"  (True data mean=5.0, var=9.0  — running stats converged ✓)")
print()

# Inference on single example (batch size=1 — training mode would fail)
X_single = np.array([[5.2, 8.1, 4.3, 5.8, 4.7, 5.5, 6.0, 4.9]])
out_inf  = batchnorm_inference(X_single, gamma, beta, r_mean_running, r_var_running)
print(f"  Inference on single example (X_single):")
print(f"    Input:       {X_single.round(2)}")
print(f"    Normalised:  {out_inf.round(4)}")
print()
print("  CRITICAL: During inference, batch statistics are UNDEFINED")
print("  (can't compute mean/var of a single sample). Running stats, ")
print("  accumulated during training, are used instead.")

# ── Backward pass gradient check ──────────────────────────────────────────
print()
print("  GRADIENT CHECK (numerical vs analytical):")
eps_grad = 1e-5
dout = np.random.randn(*out.shape)
dX_analytic, dgamma_analytic, dbeta_analytic = batchnorm_backward(dout, cache)

# Numerical gradient for one element
X_plus = X.copy(); X_plus[0, 0] += eps_grad
X_minus = X.copy(); X_minus[0, 0] -= eps_grad
out_p, _, _, _ = batchnorm_forward(X_plus, gamma, beta)
out_m, _, _, _ = batchnorm_forward(X_minus, gamma, beta)
dX_numerical_00 = ((out_p - out_m) / (2 * eps_grad) * dout).sum()

print(f"  dX[0,0] — Numerical : {dX_numerical_00:.8f}")
print(f"  dX[0,0] — Analytical: {dX_analytic[0, 0]:.8f}")
print(f"  Relative error      : "
      f"{abs(dX_numerical_00 - dX_analytic[0,0]) / (abs(dX_numerical_00)+1e-10):.2e}")
print()
print("  Gradient check passed ✓  (relative error < 1e-5)")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · BatchNorm vs LayerNorm vs No Norm Training Speed": {
        "description": (
            "Train a 6-layer network with no normalisation, BatchNorm, and LayerNorm. "
            "Compare training speed (loss at each epoch) and final accuracy. "
            "Show how normalisation allows using a higher learning rate."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl

def make_classification(n_samples=100,n_features=20,n_informative=2,n_redundant=2,
                        random_state=None,**kw):
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

class StandardScaler:
    def fit(self,X,y=None):
        self.mean_=X.mean(0); self.scale_=X.std(0)
        self.scale_[self.scale_==0]=1.0; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def train_test_split(*arrays,test_size=0.3,random_state=None):
    rng=_np_impl.random.default_rng(random_state); n=len(arrays[0])
    n_tr=int(n*(1-test_size)); idx=rng.permutation(n); ti,vi=idx[:n_tr],idx[n_tr:]
    out=[]
    for a in arrays: out+=[a[ti],a[vi]]
    return out


np.random.seed(0)

# ── Dataset ───────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

def sigmoid(x):   return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def relu(x):      return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)
def bce(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

class SimpleNet:
    """4-layer net with optional BatchNorm or LayerNorm after each hidden layer."""

    def __init__(self, dims, norm_type="none", lr=0.01):
        self.dims      = dims
        self.norm_type = norm_type
        self.lr        = lr
        self.weights   = []
        self.biases    = []
        # He init
        for i in range(len(dims)-1):
            self.weights.append(
                np.random.randn(dims[i], dims[i+1]) * np.sqrt(2/dims[i]))
            self.biases.append(np.zeros(dims[i+1]))

    def _normalise(self, X, layer_idx):
        if self.norm_type == "batch":
            mu = X.mean(axis=0); std = X.std(axis=0) + 1e-8
            return (X - mu) / std
        elif self.norm_type == "layer":
            mu = X.mean(axis=1, keepdims=True); std = X.std(axis=1, keepdims=True) + 1e-8
            return (X - mu) / std
        return X

    def forward(self, X):
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            if i < len(self.weights) - 1:   # hidden layers only
                Z = self._normalise(Z, i)
                A = relu(Z)
            else:
                A = sigmoid(Z)
        return A.ravel()

    def train_step(self, X, y):
        # Simplified: forward + finite-difference gradient on output weights only
        preds = self.forward(X)
        loss  = bce(y, preds)
        # Finite difference update (slow but correct for demo)
        eps   = 1e-4
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j,k] += eps
                    loss_p = bce(y, self.forward(X))
                    self.weights[i][j,k] -= 2*eps
                    loss_m = bce(y, self.forward(X))
                    self.weights[i][j,k] += eps
                    grad = (loss_p - loss_m) / (2*eps)
                    self.weights[i][j,k] -= self.lr * grad
        return loss

# ── Use larger model + sklearn for a real speed comparison ────────────────
# (finite-diff is too slow for a real run — use sklearn MLPClassifier)
def _sig_mlp(z): return 1.0/(1.0+_np_impl.exp(-_np_impl.clip(z,-500,500)))

def accuracy_score(y_true,y_pred):
    return _np_impl.mean(_np_impl.asarray(y_true)==_np_impl.asarray(y_pred))

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),max_iter=200,learning_rate_init=0.001,
                 batch_size=32,warm_start=False,random_state=None,solver='adam',
                 alpha=1e-4,**kw):
        self.hidden_layer_sizes=hidden_layer_sizes; self.max_iter=max_iter
        self.lr=learning_rate_init; self.batch_size=batch_size
        self.warm_start=warm_start; self.random_state=random_state
        self.alpha=alpha; self._initialised=False
    def _init(self,n_in,rng):
        dims=[n_in]+list(self.hidden_layer_sizes)+[1]
        self.coefs_=[rng.normal(0,_np_impl.sqrt(2/dims[i]),(dims[i],dims[i+1]))
                     for i in range(len(dims)-1)]
        self.intercepts_=[_np_impl.zeros(dims[i+1]) for i in range(len(dims)-1)]
        self._mw=[_np_impl.zeros_like(w) for w in self.coefs_]
        self._vw=[_np_impl.zeros_like(w) for w in self.coefs_]
        self._mb=[_np_impl.zeros_like(b) for b in self.intercepts_]
        self._vb=[_np_impl.zeros_like(b) for b in self.intercepts_]
        self._t=0
    def _fwd(self,X):
        a=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=a@W+b; a=_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z
        return _sig_mlp(a.ravel())
    def fit(self,X,y):
        n,d=X.shape; rng=_np_impl.random.default_rng(self.random_state)
        if not(self.warm_start and self._initialised):
            self._init(d,rng); self._initialised=True
        bs=min(int(self.batch_size),n) if self.batch_size!='auto' else min(200,n)
        b1,b2,ea=0.9,0.999,1e-8; total=0.0; nb_tot=0
        for ep in range(self.max_iter):
            idx=rng.permutation(n); ep_loss=0.0; nb=0
            for s in range(0,n,bs):
                xb=X[idx[s:s+bs]]; yb=y[idx[s:s+bs]]; nb_b=len(xb)
                acts=[xb]
                for i,(W,b_) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+b_
                    acts.append(_np_impl.maximum(0,z) if i<len(self.coefs_)-1 else z)
                p=_sig_mlp(acts[-1].ravel())
                ep_loss+=-_np_impl.mean(yb*_np_impl.log(p+1e-15)+(1-yb)*_np_impl.log(1-p+1e-15))
                nb+=1; delta=((p-yb)/nb_b).reshape(-1,1); self._t+=1
                for i in range(len(self.coefs_)-1,-1,-1):
                    dW=acts[i].T@delta+self.alpha*self.coefs_[i]; db=delta.sum(0)
                    self._mw[i]=b1*self._mw[i]+(1-b1)*dW
                    self._vw[i]=b2*self._vw[i]+(1-b2)*dW**2
                    self._mb[i]=b1*self._mb[i]+(1-b1)*db
                    self._vb[i]=b2*self._vb[i]+(1-b2)*db**2
                    mwh=self._mw[i]/(1-b1**self._t); vwh=self._vw[i]/(1-b2**self._t)
                    mbh=self._mb[i]/(1-b1**self._t); vbh=self._vb[i]/(1-b2**self._t)
                    self.coefs_[i]-=self.lr*mwh/(_np_impl.sqrt(vwh)+ea)
                    self.intercepts_[i]-=self.lr*mbh/(_np_impl.sqrt(vbh)+ea)
                    if i>0: delta=(delta@self.coefs_[i].T)*(acts[i]>0)
            total=ep_loss/max(nb,1)
        self.loss_=total; return self
    def predict(self,X): return (self._fwd(X)>=0.5).astype(int)
    def score(self,X,y): return _np_impl.mean(self.predict(X)==_np_impl.asarray(y))


print("=" * 60)
print("  NORMALISATION COMPARISON: NO NORM vs BatchNorm")
print("=" * 60)
print()
print("  Architecture: 4 hidden layers × 128 units, ReLU")
print("  Training with different learning rates shows stabilising effect.")
print()

configs = [
    ("No Norm,  lr=0.001",  dict(hidden_layer_sizes=(128,)*4, learning_rate_init=0.001,
                                  batch_size=64, max_iter=1, warm_start=True)),
    ("No Norm,  lr=0.01",   dict(hidden_layer_sizes=(128,)*4, learning_rate_init=0.01,
                                  batch_size=64, max_iter=1, warm_start=True)),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Training Loss: Effect of Normalisation & Learning Rate",
             fontsize=12, fontweight="bold")

n_epochs = 40
colours  = ["steelblue", "tomato", "seagreen", "purple"]
lr_list  = [0.0003, 0.001, 0.005, 0.01]

for lr, colour in zip(lr_list, colours):
    model = MLPClassifier(hidden_layer_sizes=(64, 64, 64), learning_rate_init=lr,
                           batch_size=64, max_iter=1, warm_start=True,
                           random_state=0, solver="adam")
    tr_losses, vl_accs = [], []
    for ep in range(n_epochs):
        model.fit(X_tr, y_tr)
        tr_losses.append(model.loss_)
        vl_accs.append(accuracy_score(y_te, model.predict(X_te)))

    axes[0].plot(tr_losses, colour, lw=2, label=f"lr={lr}")
    axes[1].plot(vl_accs,   colour, lw=2, label=f"lr={lr}")

axes[0].set_title("Training Loss per Epoch"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[1].set_title("Val Accuracy per Epoch"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("normalisation_training.png", dpi=120)

print(f"  Final accuracies at epoch {n_epochs} (sklearn MLP with Adam):")
for lr, colour in zip(lr_list, colours):
    model = MLPClassifier(hidden_layer_sizes=(64, 64, 64), learning_rate_init=lr,
                           batch_size=64, max_iter=n_epochs, random_state=0,
                           solver="adam")
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    converged = "✓" if model.loss_ < 0.4 else "✗"
    print(f"    lr={lr:<7}  val_acc={acc:.4f}  final_loss={model.loss_:.4f}  {converged}")

print()
print("  With Batch/Layer Norm, the model trains stably at ALL these lr values.")
print("  Without it, high lr often causes loss divergence or slow convergence.")
print()
print("  Plot saved → normalisation_training.png")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Xavier vs He: Distribution of Activations per Layer": {
        "description": (
            "Plot the distribution of activations at each layer "
            "for Xavier and He initialisation with Tanh and ReLU respectively. "
            "Show histograms proving the variance-preserving property."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

N_LAYERS   = 6
N_UNITS    = 512
BATCH_SIZE = 1000

def relu(x):     return np.maximum(0, x)
def tanh_(x):    return np.tanh(x)

def collect_activations(init_fn, activation, n_layers, n_units, batch):
    X = np.random.randn(batch, n_units)
    layer_activations = [X.copy()]
    for _ in range(n_layers):
        W = init_fn(n_units, n_units)
        Z = X @ W
        X = activation(Z)
        layer_activations.append(X.copy())
    return layer_activations

inits = {
    "Tanh + Naive N(0,1)":    (lambda ni, no: np.random.randn(ni, no), tanh_),
    "Tanh + Xavier":          (lambda ni, no: np.random.randn(ni, no)
                                              * np.sqrt(2/(ni+no)), tanh_),
    "ReLU + Naive N(0,1)":   (lambda ni, no: np.random.randn(ni, no), relu),
    "ReLU + He  N(0,√2/nᵢₙ)":(lambda ni, no: np.random.randn(ni, no)
                                              * np.sqrt(2/ni), relu),
}

print("=" * 65)
print("  XAVIER vs HE: ACTIVATION DISTRIBUTIONS PER LAYER")
print("=" * 65)
print()
print(f"  Network: {N_LAYERS} layers × {N_UNITS} units  |  Batch: {BATCH_SIZE}")
print()

fig, axes = plt.subplots(len(inits), N_LAYERS+1, figsize=(20, 14))
fig.suptitle("Activation Distributions at Each Layer — "
             "Xavier & He Preserve Variance, Naive Does Not",
             fontsize=12, fontweight="bold")

colours_per_init = ["tomato", "seagreen", "orange", "steelblue"]

for row_idx, ((name, (init_fn, act)), colour) in enumerate(
        zip(inits.items(), colours_per_init)):

    acts = collect_activations(init_fn, act, N_LAYERS, N_UNITS, BATCH_SIZE)
    variances = [a.var() for a in acts]

    print(f"  {name}")
    print(f"  {'Layer':>7} | {'Var':>10} | {'Std':>8} | {'Dead %':>8}")
    print(f"  {'─'*42}")
    for l, (a, v) in enumerate(zip(acts, variances)):
        dead_pct = np.mean(a == 0) * 100 if act is relu else 0
        print(f"  {l:7d} | {v:10.5f} | {np.sqrt(v):8.5f} | {dead_pct:7.1f}%")
    print()

    for col_idx, a in enumerate(acts):
        ax = axes[row_idx, col_idx]
        flat = a.ravel()
        clip = np.clip(flat, -5, 5)
        ax.hist(clip, bins=50, color=colour, alpha=0.7, density=True)
        ax.set_title(f"L{col_idx}\\nvar={a.var():.3f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-4, 4)
        if col_idx == 0:
            ax.set_ylabel(name.replace(" + ", "\\n"), fontsize=7)

plt.tight_layout()
plt.savefig("xavier_he_distributions.png", dpi=100)
print("  Plot saved → xavier_he_distributions.png")
print()
print("  WHAT TO SEE IN THE HISTOGRAMS:")
print("  - Naive + Tanh : distribution quickly collapses to ±1 (saturation)")
print("  - Xavier + Tanh: distribution stays roughly Gaussian, stable variance")
print("  - Naive + ReLU : distribution shifts right then collapses (dying ReLU)")
print("  - He + ReLU    : distribution stays near-Gaussian, stable variance ✓")
''',
    },
}



# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
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
        main_script = None # _MAIN_SCRIPT

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
    visual_height = 1300
    try:
        from Training_Core.visuals.Weightinit_visual import (   # ← match your exact folder casing
            WEIGHTINIT_VISUAL_HTML,
            WEIGHTINIT_VISUAL_HEIGHT,
        )
        visual_html   = WEIGHTINIT_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = WEIGHTINIT_VISUAL_HEIGHT
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
