"""
Optimisers & Learning Rate Strategies
=======================================

How gradient descent actually works in practice — and why the naive version
leads to underfitting, oscillation, and slow convergence.
Adaptive optimisers, momentum, and learning rate schedules are all tools
that make the difference between a model that trains and one that doesn't.

"""

import textwrap
import re

TOPIC_NAME   = "Optimisers & Learning Rate Strategies"
DISPLAY_NAME = "02 · Optimisers & LR Strategies"
ICON         = "🚀"
SUBTITLE     = "From SGD to Adam — Making Gradient Descent Work"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """


##### PART 1 — THE LOSS LANDSCAPE & WHY OPTIMISATION IS HARD

### The Loss Surface

Training a neural network means finding the weights W that minimise the
loss L(W). In a network with millions of parameters, L(W) is a
high-dimensional surface with many challenging features:

    Diagram 1 — Loss Surface Obstacles:

    Loss │
         │  saddle
         │  point ↓         plateau (gradients ≈ 0, lr too small → stuck)
         │   ─╮─────────────────────────────────────────
         │  ╱   ╲          local           global
         │ ╱     ╲─────────minimum─────────minimum
         │╱        ╲─────╯       ╲────────╱
         └──────────────────────────────────────────── Weights →

    Obstacles gradient descent must navigate:
    ┌────────────────────────────────────────────────────────────┐
    │  Local minima    — stuck in suboptimal solution            │
    │  Saddle points   — gradient=0 but not a minimum            │
    │  Plateaus        — gradient ≈ 0, learning stalls           │
    │  Ravines         — narrow valleys that cause oscillation   │
    │  Cliffs          — sudden large gradients → divergence     │
    └────────────────────────────────────────────────────────────┘

    Note: for deep networks, local minima are less problematic than
    saddle points and plateaus. Most "bad" local minima are in fact
    saddle points that look like minima from certain directions.


### Vanilla Gradient Descent (Batch GD)

    w ← w − η · ∇L(w)       where η = learning rate

    • Uses the gradient of the FULL dataset at each step.
    • Guaranteed to converge to a local minimum (if η is small enough).
    • Problem: computing ∇L over the full dataset is expensive.
    • Problem: the gradient is exact but slow — one update per pass.


### Stochastic Gradient Descent (SGD)

    Sample one random example (xᵢ, yᵢ), compute its gradient, update:
    w ← w − η · ∇Lᵢ(w)

    • One update per example instead of per epoch → much faster.
    • The gradient is a NOISY estimate of the true gradient.
    • Noise is actually BENEFICIAL: it helps escape shallow local minima.
    • Problem: very noisy → oscillates, may not converge to exact minimum.


### Mini-Batch SGD (the universal default)

    Sample a mini-batch B of m examples, average their gradients:
    w ← w − η · (1/m) Σᵢ∈B ∇Lᵢ(w)

    • Best of both worlds: noisier than full GD (escape local minima),
      more stable than single-sample SGD.
    • GPU-parallelisable: batches of 32-512 are standard.
    • The noise is now controllable via batch size:
        - Smaller batch → more noise → better generalisation but slower
        - Larger batch → less noise → faster per-epoch but may converge
          to sharper, less-generalising minima

    Diagram 2 — Loss Surface Trajectory by Batch Size:

    Full GD         Mini-batch SGD      Single-sample SGD
    (B=N)           (B=64)              (B=1)
    ────────────    ────────────────    ────────────────────
    →→→→→→→→→→      ↗↘↗↗↘↗↗↘↗↘          ↗↘↙↗↘↗↙↘↗↗↙↘↗↙↗
    Smooth path     Noisy but focused   Very noisy, zig-zag
    Slow updates    Good balance        Fast updates, erratic



##### PART 2 — MOMENTUM-BASED OPTIMISERS

### SGD with Momentum

The key problem with vanilla SGD: in a ravine (where the loss curves
sharply in one dimension and gently in another), SGD oscillates wildly
across the ravine and moves slowly along it.

Momentum fixes this by accumulating a velocity vector in the direction
of persistent gradients:

    vₜ = β · vₜ₋₁ + (1 − β) · ∇L(w)    (exponential moving average)
    w  ← w − η · vₜ

    β (momentum coefficient) is typically 0.9.
    "Velocity" v builds up in consistent directions and dampens oscillations.

    Diagram 3 — SGD vs Momentum in a Ravine:

    Top view of a narrow elliptical loss surface:

    SGD (oscillates):              SGD + Momentum (smooth):
    ┌─────────────────────┐        ┌───────────────────────┐
    │  ×  ×  ×  ×  ×  ×   │        │                       │
    │ × × × × × × × ×     │        │  ──────────────────→  │
    │  ×  ×  ×  ×  ×  ×   │        │                       │
    │ × × × × × × ×       │        │                       │
    └─────────────────────┘        └───────────────────────┘
    Zig-zag across narrow axis      Smooth path along valley


### Nesterov Momentum (NAG)

An improvement: look ahead in the direction of momentum before computing
the gradient:

    vₜ = β · vₜ₋₁ + ∇L(w − β · vₜ₋₁)   (gradient at "lookahead" position)
    w  ← w − η · vₜ

    Why better? Regular momentum computes the gradient at the current
    position, then takes a step. NAG first takes the momentum step, THEN
    computes the gradient at the new position — a smarter correction.
    NAG converges faster in theory and often in practice.



##### PART 3 — ADAPTIVE LEARNING RATE OPTIMISERS

### The Motivation: Not All Parameters Learn at the Same Rate

In vanilla SGD, every parameter uses the same global learning rate η.
But parameters differ enormously:
    - Some appear in nearly every example (common features) → large, reliable gradients
    - Some appear rarely (rare words, sparse inputs) → infrequent, small gradients

Rare-feature parameters should take LARGER steps when they do get gradients.
Common-feature parameters should take SMALLER steps to avoid overshooting.
Adaptive optimisers learn a per-parameter learning rate automatically.


### AdaGrad (2011)

Adapts η per parameter by dividing by the square root of the sum of all
past squared gradients:

    Gₜ = Gₜ₋₁ + gₜ²       (accumulate squared gradients)
    w  ← w − (η / √(Gₜ + ε)) · gₜ

    • Rare features: Gₜ small → large effective lr → fast learning
    • Common features: Gₜ large → small effective lr → stable learning
    • Problem: Gₜ only GROWS → effective lr → 0 → learning stops.
      AdaGrad dies for long training runs.


### RMSProp (2012)

Fix AdaGrad's dying lr by using an exponential moving average of
squared gradients (forgetting old gradients):

    sₜ = ρ · sₜ₋₁ + (1 − ρ) · gₜ²    (decay rate ρ ≈ 0.9)
    w  ← w − (η / √(sₜ + ε)) · gₜ

    The effective lr no longer decays to zero — it adapts to the
    recent gradient magnitude. Works well for non-stationary problems.


### Adam (Adaptive Moment Estimation) — 2015

The most popular optimiser. Combines momentum (1st moment) with RMSProp
(2nd moment):

    mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ     (1st moment: momentum)
    vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²    (2nd moment: squared grads)

    Bias correction (compensates for initialisation at 0):
    m̂ₜ = mₜ / (1 − β₁ᵗ)
    v̂ₜ = vₜ / (1 − β₂ᵗ)

    Update:
    w ← w − η · m̂ₜ / (√v̂ₜ + ε)

    Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8, η=0.001

    Adam =  Momentum (direction)  +  RMSProp (step size)
               ↑                           ↑
           Accelerates along           Normalises by
           consistent direction        gradient magnitude


### AdamW (Adam with Decoupled Weight Decay) — 2019

A critical fix for Adam. Standard Adam applies L2 regularisation
INSIDE the adaptive scaling, which distorts the regularisation effect:

    Adam + L2:   w ← w − η · (m̂ₜ/(√v̂ₜ+ε) + λw)    ← WRONG
                 The adaptive scaling affects the weight decay too.

    AdamW:       w ← w − η · m̂ₜ/(√v̂ₜ+ε)  − η·λ·w  ← CORRECT
                 Weight decay applied SEPARATELY from the gradient step.

    AdamW is now the default for transformer training (BERT, GPT, ViT).
    When in doubt about regularisation with Adam, use AdamW.

    Optimiser Comparison:
    ┌────────────────┬───────────────────────┬────────────────────────────┐
    │ Optimiser      │ Key idea              │ When to use                │
    ├────────────────┼───────────────────────┼────────────────────────────┤
    │ SGD            │ Pure gradient descent │ Simple, convex problems    │
    │ SGD+Momentum   │ Exponential avg of g  │ CNNs, well-tuned lr        │
    │ Nesterov       │ Look-ahead momentum   │ Fine-tuning with momentum  │
    │ AdaGrad        │ Divide by Σg²         │ Sparse data (NLP embeddings│
    │ RMSProp        │ Divide by EMA(g²)     │ RNNs, non-stationary       │
    │ Adam           │ Momentum + RMSProp    │ General default (fast)     │
    │ AdamW          │ Adam + proper wd      │ Transformers, regularised  │
    └────────────────┴───────────────────────┴────────────────────────────┘



##### PART 4 — LEARNING RATE SCHEDULES

### Why Decay the Learning Rate?

A large learning rate helps explore the loss surface quickly early in
training. But as the model approaches a minimum, a large lr causes
oscillation around it instead of converging into it.

    Phase 1 (early training):   large lr → coarse exploration
    Phase 2 (fine-tuning):      small lr → precise convergence

    Learning rate schedules automate this transition.

### Common Schedules

**Step Decay:**
    Multiply lr by γ every k epochs.
    ηₜ = η₀ × γ^floor(t/k)      (e.g., multiply by 0.1 every 30 epochs)

    Diagram 4 — Step Decay:

    η  │
       │█████
       │    ████
       │        ███
       │           ███
       └─────────────────── Epoch

**Exponential Decay:**
    ηₜ = η₀ × γᵗ               (continuous smooth decay)

**Cosine Annealing:**
    ηₜ = η_min + ½(η_max − η_min)(1 + cos(πt/T))

    Starts high, smoothly descends to η_min, then can RESTART.
    Popular for computer vision and NLP. Allows "warm restarts" which
    can escape local minima by briefly increasing lr again.

    Diagram 5 — Cosine Annealing (with warm restarts):

    η  │╲             ╲             ╲
       │  ╲            ╲            ╲
       │    ╲           ╲           ╲
       │      ──╲         ──╲         ──╲
       │          ╲           ╲          ╲
       └──────────────────────────────────── Epoch
              restart       restart

**Warmup:**
    Start with a very small lr, linearly increase to the target lr
    over a warmup period (10-20% of training), then decay.

    WHY WARMUP? At initialisation, parameters and gradients are not yet
    meaningful. A large initial lr would make huge destructive updates.
    Warmup lets the optimizer accumulate reliable gradient estimates before
    taking large steps. Essential for transformer training.

    Diagram 6 — Warmup then Cosine Decay (Transformer recipe):

    η  │         ╱╲
       │        ╱  ╲
       │       ╱    ╲
       │      ╱      ╲
       │     ╱        ──────────╲
       │    ╱                    ──────╲
       │───╱                           ──────
       └─────────────────────────────────── Step
           Warmup  Peak  Cosine decay to min

**1-cycle Policy (super-convergence):**
    Phase 1: lr grows from base to max (and momentum decreases)
    Phase 2: lr decays from max to min (and momentum increases)
    Can train 5-10× faster than fixed lr schedules.



##### PART 5 — GRADIENT CLIPPING

### What is Gradient Clipping?

In RNNs and deep networks, gradients can occasionally explode — a single
batch produces abnormally large gradients (e.g., at a "cliff" in the loss
surface). This causes a catastrophic parameter update that undoes previous
learning.

**Gradient clipping** caps the gradient magnitude before the update:

    Method 1 — Clip by value:
        g ← clip(g, -c, c)    (each gradient element independently clipped)
        Problem: changes the direction of the gradient vector.

    Method 2 — Clip by global norm (better):
        ‖g‖ = √(Σ gᵢ²)
        if ‖g‖ > c:   g ← g × c / ‖g‖
        → Scales the entire gradient vector to have norm c,
          preserving its direction.

    Diagram 7 — Gradient Clipping Visualised:

    Loss
      │
      │         ← gradient cliff ─┐
      │                            ↓
      │          ___/|──────────────────────  ← normal loss
      │        /     |  ↑ huge gradient
      │       /      | this step would overshoot
      │      /       │                              x
      └──────────────────────────────────────────────

    Without clipping: w jumps to a terrible region, training crashes.
    With clipping:    gradient capped → update bounded → training continues.

    Where to use: RNNs (almost always), very deep networks, any
    architecture with loss spikes. Typical clip value: 1.0 to 5.0.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Optimisers from Scratch: SGD → Momentum → RMSProp → Adam": {
        "description": (
            "Implement SGD, SGD+Momentum, RMSProp, and Adam from scratch "
            "and race them on the same quadratic loss surface. "
            "Track convergence trajectories and speed."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Test objective: Rosenbrock function (classic optimisation benchmark) ──
# f(x, y) = (1-x)² + 100(y - x²)²
# Global minimum at (1, 1). Narrow curved valley — hard for plain SGD.

def rosenbrock(w):
    x, y = w
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(w):
    x, y = w
    dfdx = -2*(1 - x) - 400*x*(y - x**2)
    dfdy = 200*(y - x**2)
    return np.array([dfdx, dfdy])

# ── Optimiser implementations ──────────────────────────────────────────────
class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr
    def step(self, w, g, **state):
        return w - self.lr * g, state

class SGDMomentum:
    def __init__(self, lr=0.001, beta=0.9):
        self.lr, self.beta = lr, beta
    def step(self, w, g, v=None, **state):
        if v is None: v = np.zeros_like(w)
        v = self.beta * v + (1 - self.beta) * g
        return w - self.lr * v, {"v": v}

class RMSProp:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps = lr, rho, eps
    def step(self, w, g, s=None, **state):
        if s is None: s = np.zeros_like(w)
        s = self.rho * s + (1 - self.rho) * g**2
        return w - self.lr * g / (np.sqrt(s) + self.eps), {"s": s}

class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
    def step(self, w, g, m=None, v=None, t=0, **state):
        if m is None: m = np.zeros_like(w)
        if v is None: v = np.zeros_like(w)
        t  += 1
        m   = self.b1 * m + (1 - self.b1) * g
        v   = self.b2 * v + (1 - self.b2) * g**2
        m_h = m / (1 - self.b1**t)
        v_h = v / (1 - self.b2**t)
        return w - self.lr * m_h / (np.sqrt(v_h) + self.eps), {"m": m, "v": v, "t": t}

# ── Run optimisers ────────────────────────────────────────────────────────
optimisers = [
    ("SGD       lr=0.001",    SGD(lr=0.001),           "tomato"),
    ("Momentum  lr=0.001",    SGDMomentum(lr=0.001),   "orange"),
    ("RMSProp   lr=0.01",     RMSProp(lr=0.01),        "steelblue"),
    ("Adam      lr=0.01",     Adam(lr=0.01),            "seagreen"),
]

N_STEPS = 2000
w0 = np.array([-1.5, 1.5])   # starting point far from (1,1)

print("=" * 65)
print("  OPTIMISER COMPARISON ON ROSENBROCK FUNCTION")
print("=" * 65)
print(f"  Objective: f(x,y) = (1-x)² + 100(y-x²)²")
print(f"  Global min: (1, 1)  |  Start: {w0}")
print(f"  Max steps: {N_STEPS}")
print()

trajectories = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Optimiser Comparison on Rosenbrock Function",
             fontsize=13, fontweight="bold")

for name, opt, colour in optimisers:
    w = w0.copy()
    state = {}
    losses = []
    path = [w.copy()]

    for step in range(N_STEPS):
        g = rosenbrock_grad(w)
        w, state = opt.step(w, g, **state)
        w = np.clip(w, -3, 3)    # safety clip
        losses.append(rosenbrock(w))
        path.append(w.copy())
        if rosenbrock(w) < 1e-6:
            print(f"  {name}: CONVERGED at step {step+1}! "
                  f"w=({w[0]:.5f}, {w[1]:.5f})")
            break

    final_loss = losses[-1]
    final_w    = w
    n_steps    = len(losses)
    converged  = "✓" if final_loss < 0.01 else "✗"
    print(f"  {name:25s} | final_loss={final_loss:.6f} | "
          f"w=({final_w[0]:.3f},{final_w[1]:.3f}) | {converged}")

    trajectories[name] = {"losses": losses, "path": np.array(path),
                           "colour": colour}

print()

# ── Plot 1: loss curves ────────────────────────────────────────────────────
for name, data in trajectories.items():
    axes[0].semilogy(data["losses"], data["colour"], lw=2,
                     label=name, alpha=0.85)
axes[0].set_title("Loss vs Step (log scale)")
axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss (log)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, N_STEPS)

# ── Plot 2: 2D trajectories on contour ────────────────────────────────────
xg = np.linspace(-2, 1.5, 200)
yg = np.linspace(-0.5, 2.5, 200)
Xg, Yg = np.meshgrid(xg, yg)
Zg = (1 - Xg)**2 + 100*(Yg - Xg**2)**2

axes[1].contour(Xg, Yg, np.log1p(Zg), levels=30, cmap="viridis", alpha=0.5)
axes[1].contourf(Xg, Yg, np.log1p(Zg), levels=30, cmap="viridis", alpha=0.2)
axes[1].plot(1, 1, "k*", ms=15, label="Global min (1,1)")

for name, data in trajectories.items():
    path = data["path"]
    axes[1].plot(path[:, 0], path[:, 1], data["colour"], lw=1.5,
                 alpha=0.8, label=name)
    axes[1].plot(path[0, 0], path[0, 1], "o", color=data["colour"], ms=8)

axes[1].set_title("Trajectory on Loss Landscape")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
axes[1].legend(fontsize=8); axes[1].set_xlim(-2, 1.5); axes[1].set_ylim(-0.5, 2.5)

plt.tight_layout()
plt.savefig("optimiser_comparison.png", dpi=120)
print("  Plot saved → optimiser_comparison.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Learning Rate Range Test (Find the Best LR)": {
        "description": (
            "Implement the learning rate range test (LR finder). "
            "Sweep lr exponentially from tiny to large, plot loss vs lr, "
            "and identify the sweet spot automatically."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np_impl

def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, random_state=None, **kw):
    rng = _np_impl.random.default_rng(random_state)
    y   = rng.integers(0, 2, n_samples)
    Xi  = rng.standard_normal((n_samples, n_informative))
    Xi += (2*y - 1)[:, None] * rng.uniform(0.8, 1.5, n_informative)
    nr  = n_redundant
    Xr  = (Xi[:, :nr] + 0.3*rng.standard_normal((n_samples, nr))
           if nr > 0 else _np_impl.empty((n_samples, 0)))
    nn2 = max(0, n_features - n_informative - nr)
    Xn  = rng.standard_normal((n_samples, nn2)) if nn2 > 0 else _np_impl.empty((n_samples, 0))
    parts = [a for a in [Xi, Xr, Xn] if a.shape[1] > 0]
    return _np_impl.hstack(parts)[:, :n_features], y.astype(int)

class StandardScaler:
    def fit(self, X, y=None):
        self.mean_  = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):          return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

def train_test_split(*arrays, test_size=0.2, random_state=None):
    rng  = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = int(n * (1 - test_size)); idx = rng.permutation(n)
    ti, vi = idx[:n_tr], idx[n_tr:]; out = []
    for a in arrays: out += [a[ti], a[vi]]
    return out


np.random.seed(0)

# ── Data ──────────────────────────────────────────────────────────────────
X, y = make_classification(n_samples=2000, n_features=15, n_informative=8,
                            random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=1)

# ── Simple neural net ─────────────────────────────────────────────────────
def sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def relu(x):     return np.maximum(0, x)

class MiniNet:
    def __init__(self, lr=0.01):
        n_in = X_tr.shape[1]
        self.W1 = np.random.randn(n_in, 64) * np.sqrt(2/n_in)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 32) * np.sqrt(2/64)
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, 1)  * np.sqrt(2/32)
        self.b3 = np.zeros(1)
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return sigmoid(self.z3).ravel()

    def step(self, X, y):
        p = self.forward(X)
        p = np.clip(p, 1e-7, 1-1e-7)
        loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        # Backward
        dz3 = (p - y).reshape(-1, 1) / len(y)
        dW3 = self.a2.T @ dz3
        dz2 = dz3 @ self.W3.T * (self.z2 > 0)
        dW2 = self.a1.T @ dz2
        dz1 = dz2 @ self.W2.T * (self.z1 > 0)
        dW1 = X.T @ dz1
        # Update
        for W, dW in [(self.W1,dW1),(self.W2,dW2),(self.W3,dW3)]:
            W -= self.lr * dW
        return loss

# ── LR Range Test ─────────────────────────────────────────────────────────
def lr_range_test(X_tr, y_tr, lr_start=1e-5, lr_end=10.0,
                  n_steps=100, batch_size=64, smoothing=0.98):
    """
    Exponentially increase lr from lr_start to lr_end.
    Record smoothed loss at each step.
    The best lr is just before the loss starts exploding.
    """
    model = MiniNet(lr=lr_start)
    lr_mult = (lr_end / lr_start) ** (1 / n_steps)
    lrs, losses_raw, losses_smooth = [], [], []
    best_loss = np.inf
    avg_loss  = 0.0
    n = len(y_tr)

    for step in range(n_steps):
        # Mini-batch
        idx = np.random.choice(n, batch_size, replace=False)
        loss = model.step(X_tr[idx], y_tr[idx])

        # Exponential smoothing of loss
        avg_loss = smoothing * avg_loss + (1 - smoothing) * loss
        smooth   = avg_loss / (1 - smoothing**(step + 1))  # bias correction

        lrs.append(model.lr)
        losses_raw.append(loss)
        losses_smooth.append(smooth)

        if smooth < best_loss:
            best_loss = smooth

        # Stop if loss explodes
        if step > 5 and smooth > 5 * best_loss:
            print(f"    Loss diverged at step {step} — stopping scan.")
            break

        model.lr *= lr_mult

    return lrs, losses_smooth

print("=" * 60)
print("  LEARNING RATE RANGE TEST")
print("=" * 60)
print()
print("  Sweeping lr from 1e-5 to 10.0 over 200 mini-batch steps...")
print()

lrs, smooth_losses = lr_range_test(X_tr, y_tr, n_steps=200)

# ── Find optimal lr (steepest negative gradient of loss) ─────────────────
loss_arr = np.array(smooth_losses)
lr_arr   = np.array(lrs)

# Gradient of loss w.r.t. log(lr)
log_lr = np.log10(lr_arr)
loss_grad = np.gradient(loss_arr, log_lr)
min_grad_idx = np.argmin(loss_grad[:len(loss_grad)//2 + int(len(loss_grad)*0.1)])
optimal_lr = lr_arr[min_grad_idx]

print(f"  Identified optimal learning rate: {optimal_lr:.6f}")
print(f"  (This is where the loss decreases most steeply)")
print()
print("  Interpretation guide:")
print("  ┌─────────────────────────────────────────────────────────┐")
print("  │  lr too small  → loss barely moves (flat on left)      │")
print("  │  lr optimal    → steepest descent in loss curve         │")
print("  │  lr too large  → loss explodes (sharp rise on right)    │")
print("  │                                                         │")
print("  │  Recommended training lr: optimal_lr / 10 to           │")
print("  │  optimal_lr  (use the bottom of the curve)             │")
print("  └─────────────────────────────────────────────────────────┘")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Learning Rate Range Test (LR Finder)", fontsize=12, fontweight="bold")

axes[0].plot(lr_arr, smooth_losses, "steelblue", lw=2)
axes[0].axvline(optimal_lr, color="tomato", linestyle="--", lw=2,
                label=f"Optimal lr ≈ {optimal_lr:.5f}")
axes[0].set_xscale("log")
axes[0].set_xlabel("Learning Rate (log scale)")
axes[0].set_ylabel("Smoothed Loss")
axes[0].set_title("Loss vs Learning Rate")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].plot(lr_arr, loss_grad, "seagreen", lw=2)
axes[1].axvline(optimal_lr, color="tomato", linestyle="--", lw=2,
                label=f"Min gradient at lr={optimal_lr:.5f}")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xscale("log")
axes[1].set_xlabel("Learning Rate (log scale)")
axes[1].set_ylabel("d(Loss)/d(log lr)")
axes[1].set_title("Loss Gradient (finds steepest descent)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("lr_range_test.png", dpi=120)
print()
print("  Plot saved → lr_range_test.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Learning Rate Schedules Comparison": {
        "description": (
            "Implement and visualise Step Decay, Exponential Decay, "
            "Cosine Annealing, and Warmup+Cosine schedules. "
            "Then train a network with each and compare convergence speed."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Schedule formulas ─────────────────────────────────────────────────────
def step_decay(epoch, lr0=0.1, drop=0.5, every=20):
    """Multiply lr by `drop` every `every` epochs."""
    return lr0 * (drop ** (epoch // every))

def exponential_decay(epoch, lr0=0.1, gamma=0.97):
    """Multiply lr by gamma each epoch."""
    return lr0 * (gamma ** epoch)

def cosine_annealing(epoch, lr_min=1e-5, lr_max=0.1, T=100):
    """Cosine schedule from lr_max to lr_min over T epochs."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))

def cosine_with_warmup(epoch, lr_min=1e-5, lr_max=0.1, T=100, warmup=10):
    """Linear warmup for `warmup` epochs, then cosine decay."""
    if epoch < warmup:
        return lr_min + (lr_max - lr_min) * epoch / warmup
    t = epoch - warmup
    T_adj = T - warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_adj))

def cosine_with_restarts(epoch, lr_min=1e-5, lr_max=0.1, T_0=25):
    """Cosine annealing with warm restarts (SGDR)."""
    t = epoch % T_0
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_0))

def one_cycle(epoch, lr_min=1e-5, lr_max=0.1, T=100):
    """1-cycle policy: ramp up, ramp down, brief low."""
    peak = int(0.45 * T)
    if epoch < peak:
        return lr_min + (lr_max - lr_min) * epoch / peak
    elif epoch < 2 * peak:
        t = epoch - peak
        return lr_max - (lr_max - lr_min) * t / peak
    else:
        return lr_min / 10

# ── Plot schedules ────────────────────────────────────────────────────────
epochs = np.arange(100)
schedules = [
    ("Step Decay\\n(drop=0.5 / 20ep)",     [step_decay(e)           for e in epochs], "tomato"),
    ("Exponential\\n(γ=0.97)",              [exponential_decay(e)    for e in epochs], "orange"),
    ("Cosine Annealing",                    [cosine_annealing(e)     for e in epochs], "steelblue"),
    ("Warmup+Cosine\\n(warmup=10ep)",       [cosine_with_warmup(e)   for e in epochs], "seagreen"),
    ("Cosine Restarts\\n(T₀=25ep)",         [cosine_with_restarts(e) for e in epochs], "purple"),
    ("1-Cycle Policy",                      [one_cycle(e)            for e in epochs], "brown"),
]

print("=" * 60)
print("  LEARNING RATE SCHEDULES")
print("=" * 60)
print(f"  {'Schedule':30s} | {'Start lr':>9} | {'Final lr':>9} | {'Min lr':>9}")
print(f"  {'─'*65}")
for name, lrs, _ in schedules:
    clean_name = name.replace("\\n", " ")
    print(f"  {clean_name:30s} | {lrs[0]:9.6f} | {lrs[-1]:9.6f} | "
          f"{min(lrs):9.6f}")
print()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Learning Rate Schedules — Visual Comparison",
             fontsize=13, fontweight="bold")

for ax, (name, lrs, colour) in zip(axes.ravel(), schedules):
    ax.plot(epochs, lrs, colour, lw=2.5)
    ax.fill_between(epochs, lrs, alpha=0.15, color=colour)
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_ylim(0, 0.115)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("lr_schedules.png", dpi=120)
print("  Plot saved → lr_schedules.png")
print()
print("  SCHEDULE SELECTION GUIDE:")
print("  ┌──────────────────────────────────────────────────────────┐")
print("  │ Schedule        │ Best for                              │")
print("  ├──────────────────────────────────────────────────────────┤")
print("  │ Step decay      │ Simple CNN, classification            │")
print("  │ Exponential     │ Any task, smooth decay                │")
print("  │ Cosine          │ CV, NLP — smooth convergence          │")
print("  │ Warmup+Cosine   │ Transformers — BERT, GPT standard     │")
print("  │ Cosine+Restart  │ Ensemble creation (restarts diversify)│")
print("  │ 1-cycle         │ Fast training, super-convergence      │")
print("  └──────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Gradient Clipping Demo": {
        "description": (
            "Simulate exploding gradients in an RNN-like scenario. "
            "Show how training diverges without clipping and stays stable with it. "
            "Compare clip-by-value vs clip-by-norm."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Simulate a loss landscape with gradient cliffs ─────────────────────────
# Toy problem: optimise w to minimise f(w) = sin(w) + occasional cliff spikes
# The "cliffs" simulate exploding gradient events in RNNs.

def compute_gradient(w, step):
    """Smooth gradient with periodic spikes (simulating RNN gradient cliffs)."""
    grad = np.cos(w) + 0.1 * np.random.randn()   # baseline gradient

    # Occasional large spike (gradient cliff) every ~25 steps
    if step % 25 == 0 and step > 0:
        grad += np.random.choice([-1, 1]) * np.random.uniform(8, 15)

    return grad

def run_optimiser(n_steps=200, lr=0.1, clip_method=None, clip_value=1.0):
    w     = 3.0      # start far from minimum at w=π/2
    ws, gs, losses = [w], [], []

    for step in range(n_steps):
        g = compute_gradient(w, step)
        original_g = g

        # Apply clipping
        if clip_method == "value":
            g = np.clip(g, -clip_value, clip_value)
        elif clip_method == "norm":
            norm = abs(g)
            if norm > clip_value:
                g = g * clip_value / norm

        w = w - lr * g
        ws.append(w)
        gs.append(abs(original_g))
        losses.append(abs(np.sin(w)))   # proxy loss

    return np.array(ws), np.array(gs), np.array(losses)

# ── Run experiments ────────────────────────────────────────────────────────
print("=" * 60)
print("  GRADIENT CLIPPING DEMO")
print("=" * 60)
print()
print("  Simulating a loss landscape with gradient cliffs (every 25 steps).")
print("  Baseline gradient: cos(w) + noise")
print("  Cliff spikes: ±8 to ±15 (randomly)")
print()

configs = [
    ("No Clipping",             None,    1.0,  "tomato"),
    ("Clip by Value  (c=1.0)", "value",  1.0,  "orange"),
    ("Clip by Norm   (c=1.0)", "norm",   1.0,  "steelblue"),
    ("Clip by Norm   (c=5.0)", "norm",   5.0,  "seagreen"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Gradient Clipping: Effect on Training Stability",
             fontsize=12, fontweight="bold")

print(f"  {'Method':30s} | {'Final w':>10} | {'Final loss':>11} | Stable?")
print(f"  {'─'*62}")

steps = np.arange(201)
for name, method, clip_val, colour in configs:
    ws, gs, losses = run_optimiser(n_steps=200, lr=0.08,
                                   clip_method=method, clip_value=clip_val)

    final_w    = ws[-1]
    final_loss = losses[-1]
    diverged   = np.any(np.abs(ws) > 50) or np.isnan(ws[-1])
    stable     = "✓ Stable" if not diverged else "✗ DIVERGED"

    print(f"  {name:30s} | {final_w:10.4f} | {final_loss:11.6f} | {stable}")

    axes[0].plot(steps, ws,            colour, lw=1.8, label=name, alpha=0.85)
    axes[1].plot(steps[1:], gs,        colour, lw=1.8, alpha=0.85)
    axes[2].plot(steps[1:], losses,    colour, lw=1.8, alpha=0.85)

for ax, title, ylabel in zip(
        axes,
        ["Parameter w over Time", "Gradient Magnitude", "Loss |sin(w)|"],
        ["w", "|gradient|", "Loss"]):
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

axes[0].legend(fontsize=8)
axes[0].axhline(np.pi/2, color="black", linestyle="--", lw=1,
                label="Optimum (π/2)")

# Mark cliff events
for cliff_step in range(25, 200, 25):
    axes[1].axvline(cliff_step, color="gray", alpha=0.4, linestyle=":")

axes[1].axhline(1.0, color="black", linestyle="--", lw=1,
                label="Clip threshold (c=1)")
axes[1].legend(fontsize=8); axes[1].set_ylim(0, 18)

plt.tight_layout()
plt.savefig("gradient_clipping.png", dpi=120)
print()
print("  Plot saved → gradient_clipping.png")
print()
print("  KEY INSIGHTS:")
print("  - No Clipping: w jumps wildly at every cliff → often diverges")
print("  - Clip by Value: safer but distorts gradient DIRECTION")
print("  - Clip by Norm: preserves direction, only scales magnitude ✓")
print("  - Larger clip_value (c=5): less aggressive, still prevents disaster")
print("  - For RNNs: clip_norm=1.0 to 5.0 is standard practice")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Adam vs AdamW: Weight Decay Correctness": {
        "description": (
            "Demonstrate why L2 regularisation in Adam is broken and how "
            "AdamW fixes it. Show that AdamW's weight decay is truly decoupled "
            "from the adaptive scaling."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Implement Adam (with L2 inside gradient) vs AdamW (decoupled wd) ──────
class AdamL2:
    """Standard Adam where L2 is added to the gradient (incorrect coupling)."""
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd

    def step(self, w, g, m, v, t):
        g_reg = g + self.wd * w       # L2 gradient INSIDE the update
        m = self.b1*m + (1-self.b1)*g_reg
        v = self.b2*v + (1-self.b2)*g_reg**2
        m_h = m / (1 - self.b1**t)
        v_h = v / (1 - self.b2**t)
        w = w - self.lr * m_h / (np.sqrt(v_h) + self.eps)
        return w, m, v

class AdamW:
    """AdamW: weight decay DECOUPLED from adaptive scaling."""
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd

    def step(self, w, g, m, v, t):
        m = self.b1*m + (1-self.b1)*g   # clean gradient only
        v = self.b2*v + (1-self.b2)*g**2
        m_h = m / (1 - self.b1**t)
        v_h = v / (1 - self.b2**t)
        # Weight decay applied SEPARATELY, after adaptive step
        w = w - self.lr * m_h / (np.sqrt(v_h) + self.eps) \
              - self.lr * self.wd * w
        return w, m, v

# ── Demonstration: track effective weight decay per parameter ──────────────
# With Adam+L2, the effective decay rate is wd/sqrt(v_hat) — not constant.
# With AdamW, the effective decay rate is exactly wd — always.

print("=" * 65)
print("  ADAM vs ADAMW: WEIGHT DECAY CORRECTNESS")
print("=" * 65)
print()
print("  In Adam+L2, the weight decay is scaled by 1/√v̂ (adaptive).")
print("  Parameters with large gradients get LESS decay than intended.")
print("  Parameters with small gradients get MORE decay than intended.")
print("  AdamW applies CONSTANT weight decay, independent of gradient magnitude.")
print()

# Simulate two parameters: one with large gradients (high v̂) and one small
n_steps = 500
wd = 0.01

# Param 1: frequent large gradients (simulates embedding of common word)
g1_seq = np.random.randn(n_steps) * 1.5

# Param 2: infrequent small gradients (simulates embedding of rare word)
g2_seq = np.random.randn(n_steps) * 0.05
g2_seq[::10] = 0   # often zero (sparse)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Adam+L2 vs AdamW: Effective Weight Decay is Different",
             fontsize=12, fontweight="bold")

for param_idx, (g_seq, param_name) in enumerate([
        (g1_seq, "Param 1: Large frequent gradients\\n(e.g., common word embedding)"),
        (g2_seq, "Param 2: Small sparse gradients\\n(e.g., rare word embedding)")]):

    for opt_idx, (OptClass, opt_name, colour) in enumerate([
            (AdamL2, "Adam+L2", "tomato"),
            (AdamW,  "AdamW",   "steelblue")]):

        opt = OptClass(lr=0.001, wd=wd)
        w = 1.0   # start at 1.0 — track how decay drives it toward 0
        m, v = 0.0, 0.0
        ws, eff_decays = [w], []

        for t, g in enumerate(g_seq, 1):
            w_prev = w
            w, m, v = opt.step(w, g, m, v, t)
            ws.append(w)

            # Effective decay = how much weight changed due to decay alone
            # For AdamW: -lr * wd * w_prev
            # For Adam+L2: embedded in adaptive step, hard to isolate
            if isinstance(opt, AdamW):
                eff_decays.append(opt.lr * opt.wd)
            else:
                v_hat = v / (1 - opt.b2**t)
                eff_decays.append(opt.lr * opt.wd / (np.sqrt(v_hat) + opt.eps))

        ax = axes[param_idx, opt_idx]
        ax2 = ax.twinx()
        ax.plot(ws, colour, lw=1.5, alpha=0.8, label="w")
        ax2.plot(eff_decays, colour, lw=1, alpha=0.4, linestyle="--",
                 label="eff. wd rate")
        ax2.axhline(wd, color="black", linestyle=":", lw=1.5,
                    label=f"Target wd={wd}")
        ax.set_title(f"{param_name}\\n{opt_name}", fontsize=9)
        ax.set_xlabel("Step"); ax.set_ylabel("w value", color=colour)
        ax2.set_ylabel("Effective decay rate")
        ax2.set_ylim(0, wd * 5)

print(f"  NUMERICAL COMPARISON (wd={wd}, lr=0.001):")
print()
print(f"  Effective weight decay rate:")
print(f"  ┌──────────────────────────────────────────────────────────┐")
print(f"  │ Optimiser  │ Large grad param   │ Small grad param      │")
print(f"  ├──────────────────────────────────────────────────────────┤")

for ParamGrads, pname in [(g1_seq, "large"), (g2_seq, "small")]:
    b2, eps = 0.999, 1e-8
    v = 0.0
    final_v_hats = []
    for t, g in enumerate(ParamGrads[:100], 1):
        v = 0.999 * v + 0.001 * g**2
        v_hat = v / (1 - b2**t)
        final_v_hats.append(v_hat)
    final_v_hat = np.mean(final_v_hats[-10:])
    eff_wd_adam = 0.001 * wd / (np.sqrt(final_v_hat) + eps)
    print(f"  │ Adam+L2    │ {eff_wd_adam:.8f} ({pname:5s}) │", end="")

print()
print(f"  │ AdamW      │ {wd:.8f} (always) │ {wd:.8f} (always) │")
print(f"  └──────────────────────────────────────────────────────────┘")
print()
print("  AdamW applies EXACTLY wd=0.01 regardless of gradient magnitude.")
print("  Adam+L2 over-decays small-gradient params and under-decays large ones.")

plt.tight_layout()
plt.savefig("adam_vs_adamw.png", dpi=120)
print()
print("  Plot saved → adam_vs_adamw.png")
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
    visual_height = 1150
    try:
        from Training_Core.visuals.optimisers_visual import (
            OPTIMISERS_VISUAL_HTML,
            OPTIMISERS_VISUAL_HEIGHT,
        )
        visual_html   = OPTIMISERS_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = OPTIMISERS_VISUAL_HEIGHT
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
        "complexity":    None,
        "operations":    OPERATIONS,
    }
