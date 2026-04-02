"""
Backpropagation & the Chain Rule
==================================

The algorithm that made deep learning possible. Backpropagation is not
magic — it is a systematic, efficient application of the chain rule of
calculus to a computation graph. Understanding it deeply means you can
debug training, reason about gradient flow, and design new architectures
without hitting invisible walls.

"""

import textwrap
import re

TOPIC_NAME   = "Backpropagation & the Chain Rule"
DISPLAY_NAME = "04 · Backpropagation"
ICON         = "🔁"
SUBTITLE     = "How Neural Networks Learn — The Chain Rule Applied to Graphs"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """


##### PART 1 — THE CREDIT ASSIGNMENT PROBLEM

### Why Backpropagation Had to Be Invented

Training a neural network means adjusting millions of weights so that the
network's loss decreases. To do this we need to know: "how much does each
weight contribute to the current error?" This is the credit assignment problem.

The naive answer — try every small perturbation of every weight and measure
the change in loss — is called numerical differentiation. For a network with
1 million weights, it requires 1 million forward passes per update step.
That is computationally impossible at scale.

Backpropagation solves this in TWO forward passes + ONE backward pass,
regardless of the number of weights. It does so by exploiting the structure
of the computation: the network is a composition of simple functions, and the
chain rule tells us exactly how to propagate error signals back through them.

    Cost of computing ALL gradients:
    ┌─────────────────────────────────────────────────────────────┐
    │  Numerical differentiation:  O(N) forward passes            │
    │  Backpropagation:            O(1) forward + O(1) backward   │
    │  Speedup:                    N× faster  (N = # parameters)  │
    │                                                             │
    │  For GPT-3 (175B parameters): backprop is 175 BILLION×      │
    │  faster than numerical differentiation.                     │
    └─────────────────────────────────────────────────────────────┘


##### PART 2 — THE CHAIN RULE

### Scalar Chain Rule

If y = f(u)  and  u = g(x), then:

    dy/dx  =  dy/du  ×  du/dx

This extends to any chain of compositions:

    y = f₄(f₃(f₂(f₁(x))))

    dy/dx = (dy/df₄) × (df₄/df₃) × (df₃/df₂) × (df₂/df₁) × (df₁/dx)

    Each factor is the LOCAL gradient of one function evaluated at the
    input it received during the forward pass.

### Vector Chain Rule (Jacobian form)

When inputs and outputs are vectors, the chain rule uses Jacobian matrices:

    If y = f(u)  and  u = g(x):
    ∂y/∂x  =  (∂y/∂u) × (∂u/∂x)   [matrix multiplication of Jacobians]

In practice we almost never materialise the full Jacobian (it would be
n_out × n_in for each layer — enormous). Instead we use the
"vector-Jacobian product" (VJP):

    Given upstream gradient dL/dy (a row vector),
    the local gradient is:  dL/dx = (dL/dy) · (∂y/∂x)

This VJP is exactly what backpropagation computes at each node.


##### PART 3 — COMPUTATION GRAPHS

### What is a Computation Graph?

A computation graph represents a mathematical expression as a directed
acyclic graph (DAG). Each node is either:
    • a LEAF node: an input variable (x, W, b) or constant
    • an OPERATION node: computes a function of its inputs (+, ×, exp, etc.)

Edges represent data flow (the output of one node is input to the next).

    Diagram 1 — Computation Graph for z = (x + y) × (x − 3):

    x ──┬──────────────────────┐
        │                      ↓
        │              ┌───────┤
        ↓              │  ( × )├─── z
    [  + ] ─────────► │        │
        ↑              └───────┤
    y ──┘                      ↑
                   ┌──────────┘
    x ──── [−3] ──┘

    Forward pass:  evaluate each node left to right
    Backward pass: propagate dL/dz right to left using chain rule

    Diagram 2 — A One-Layer Network as a Computation Graph:

    x ──────────── [× W] ──── [+ b] ──── z ──── [ReLU] ──── a ──── [Loss]
                      ↑          ↑                                      │
                      W          b                                      │
                                                                        │ dL/da
                     dL/dW ←  dL/db ←  dL/dz  ←  dL/da × da/dz ←────┘
                    (backward pass flows right to left)

    Every arrow in the backward pass carries a gradient.
    Each node receives the gradient from the right (upstream) and
    multiplies by its local gradient to produce the gradient flowing left.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4 — FORWARD AND BACKWARD PASSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### The Forward Pass

The forward pass computes the loss from the input:

    For each layer l = 1, 2, ..., L:
        z[l] = W[l] × a[l-1] + b[l]      (linear transform)
        a[l] = activation(z[l])           (non-linearity)
    Loss = L(a[L], y)                     (compare final output to label)

    Everything computed during the forward pass must be CACHED (stored)
    because the backward pass needs it to compute local gradients.
    This is why memory usage scales with network depth.


### The Backward Pass

The backward pass propagates gradients from loss back to every weight:

    Initialise: dL/da[L] = gradient of loss w.r.t. final output
    For each layer l = L, L-1, ..., 1:

        STEP 1 — Gradient through activation:
            dL/dz[l] = dL/da[l] × da[l]/dz[l]
                                   ↑
                               local gradient of activation
                               (depends on z[l] cached from forward pass)

        STEP 2 — Gradient w.r.t. weights and bias:
            dL/dW[l] = dL/dz[l] × a[l-1]ᵀ     ← used to UPDATE W[l]
            dL/db[l] = dL/dz[l].sum(axis=0)    ← used to UPDATE b[l]

        STEP 3 — Gradient to pass to previous layer:
            dL/da[l-1] = W[l]ᵀ × dL/dz[l]     ← becomes upstream for layer l-1

    The gradient flows backwards, one layer at a time.


### Why the Forward Pass Must Be Cached

    Diagram 3 — What Gets Cached and Why:

    Forward pass stores:              Backward pass needs it for:
    ┌──────────────────────────────────────────────────────────────┐
    │  a[l-1]  (activation of prev layer)   → compute dL/dW[l]     │
    │  z[l]    (pre-activation)             → compute da/dz local  │
    │                                          gradient of ReLU,   │
    │                                          sigmoid, etc.       │
    └──────────────────────────────────────────────────────────────┘

    Memory cost of backprop = O(L × batch_size × layer_width)
    This is why gradient checkpointing exists: recompute activations
    on the fly during the backward pass instead of storing them all.
    Halves memory at the cost of ~33% extra compute.


##### PART 5 — BACKPROP THROUGH COMMON OPERATIONS

### Linear Layer: z = Wx + b

    Forward:   z = Wx + b

    Backward (given upstream gradient dL/dz):
        dL/dW = dL/dz · xᵀ           (outer product, shape = W's shape)
        dL/db = dL/dz.sum(over batch) (sum over the batch dimension)
        dL/dx = Wᵀ · dL/dz           (pass gradient to previous layer)

    Intuition:
        dL/dW tells us how much each weight contributed to the error.
        Since z = Wx, each element of W appears in z multiplied by x,
        so the chain rule gives dL/dW = (dL/dz) × x.

        dL/dx = Wᵀ dL/dz passes the blame to the inputs x
        (which are the activations of the previous layer).


### ReLU: a = max(0, z)

    Forward:   a = max(0, z)

    Backward (given dL/da):
        dL/dz = dL/da × (z > 0)    ← element-wise multiplication

    The gradient of ReLU is a binary gate:
        If z > 0: gradient passes through unchanged (gate open)
        If z ≤ 0: gradient is killed (gate closed, neuron dead)

    This binary gate is what makes ReLU both powerful and dangerous.
    It allows exact gradients where the neuron is active, but kills
    learning for permanently negative neurons (dying ReLU).


### Sigmoid: σ(z) = 1/(1+e⁻ᶻ)

    Forward:   a = σ(z)

    Backward:
        dL/dz = dL/da × σ(z) × (1 − σ(z))
              = dL/da × a × (1 − a)     ← uses cached a, not z!

    The σ(z)(1−σ(z)) term is maximised at 0.25 (when z=0).
    For |z| > 4, this term is < 0.018 → gradient nearly vanishes.
    This is the quantitative source of the sigmoid vanishing gradient.


### Softmax + Cross-Entropy (combined for stability)

    Forward:   ŷ = softmax(z),   Loss = -Σ y log(ŷ)

    Backward (combined gradient):
        dL/dz = ŷ − y    ← same elegant form as sigmoid + BCE!

    This beautiful result is why softmax and cross-entropy are always
    used together. The gradient is just the prediction error.

    Deriving this manually requires the softmax Jacobian:
        ∂softmax(zᵢ)/∂zⱼ = softmax(zᵢ)(δᵢⱼ − softmax(zⱼ))
    When composed with cross-entropy, the Jacobian terms cancel and
    we are left with the prediction error only.


### Addition: z = x + y

    dL/dx = dL/dz × 1 = dL/dz    (gradient passes through unchanged)
    dL/dy = dL/dz × 1 = dL/dz    (gradient DISTRIBUTES to both branches)

    Diagram 4 — Addition Gate (gradient distributor):

    x ───────────────────────────────────────────────────────────
                         ↓                      ↑  dL/dx = dL/dz 
                      [ + ] ──→ z      dL/dz ──┤
                         ↑                      ↑  dL/dy = dL/dz
    y ───────────────────────────────────────────────────────────

    Addition gates are "gradient distributors" — they send the same
    gradient upstream to every input branch. This is why residual
    connections (skip connections in ResNets) are so powerful: they
    provide a direct gradient highway from the loss all the way back
    to early layers, bypassing the multiplicative attenuation of the
    intermediate layer gradients.


### Multiplication: z = x × y

    dL/dx = dL/dz × y    (scale by the OTHER operand)
    dL/dy = dL/dz × x    (scale by the OTHER operand)

    Multiplication gates are "gradient scalers" — each input gets
    the gradient scaled by the other input's value.
    If y is very small, dL/dx is small → vanishing gradient through ×.
    This is exactly the mechanism behind vanishing gradients in RNNs
    where the same W matrix is multiplied through many time steps.


##### PART 6 — NUMERICAL GRADIENT CHECKING

### How to Verify Your Backprop Implementation

Gradient checking uses the definition of the derivative:

    df/dx ≈ [f(x + ε) − f(x − ε)] / (2ε)    for small ε (e.g., 1e-5)

This is the CENTRED finite difference — far more accurate than the
one-sided (f(x+ε)−f(x))/ε because the O(ε) error terms cancel.

Algorithm:
    1. Compute analytical gradient g_analytic via backprop
    2. For each parameter θᵢ:
       a. θᵢ ← θᵢ + ε;  loss_plus  = forward_pass(θ)
       b. θᵢ ← θᵢ − ε;  loss_minus = forward_pass(θ)
       c. θᵢ ← θᵢ + ε;  (restore)
       d. g_numeric[i] = (loss_plus − loss_minus) / (2ε)
    3. Relative error = ||g_analytic − g_numeric|| / ||g_analytic + g_numeric||

    Interpretation:
    < 1e-7:  perfect — almost certainly correct
    < 1e-5:  probably fine
    < 1e-3:  suspicious — check your implementation
    > 1e-2:  wrong — there is a bug in your gradient

    Important: ONLY use gradient checking for debugging. NEVER use it
    during actual training — it requires O(N) forward passes.


##### PART 7 — GRADIENT FLOW ANALYSIS

### Why Deep Networks Are Hard to Train

In a depth-L network, the gradient of the loss w.r.t. weights in
layer 1 involves L multiplications of local Jacobians:

    dL/dW[1] = (dL/da[L]) × (∏ᵢ₌₂ᴸ da[i]/dz[i] × dz[i]/da[i-1]) × dL/dW[1]

Each factor da[i]/dz[i] is the local gradient of the activation:
    Sigmoid:  max value 0.25 → after 10 layers: 0.25¹⁰ ≈ 10⁻⁶ (vanished!)
    ReLU:     value is 0 or 1 → gradient either preserved or killed exactly

Each factor dz[i]/da[i-1] = W[i]ᵀ — if the weights are:
    |eigenvalue| < 1 → gradients SHRINK exponentially → VANISHING
    |eigenvalue| > 1 → gradients GROW  exponentially → EXPLODING

    Diagram 5 — Gradient Magnitude Through a Deep Network:

    Layer:      L   L-1   L-2   L-3   L-4   ...   1
                │
    Sigmoid:    1 → 0.25 → 0.06 → 0.015 → 0.004 → ...→ ≈ 0
                │
    ReLU:       1 →  1   →  1   →  1    →  1    → ...→ ≈ 1
    (active)    │
                │
    ReLU:       1 →  0   →  0   →  0    →  0    → ...→ 0
    (dead)      │

### How Modern Architectures Fix Gradient Flow

    ┌────────────────────────┬──────────────────────────────────────────┐
    │ Technique              │ How it helps gradient flow               │
    ├────────────────────────┼──────────────────────────────────────────┤
    │ ReLU (not sigmoid)     │ Gradient ∈ {0, 1} — no attenuation       │
    │ He initialisation      │ Keeps gradient variance stable per layer │
    │ Batch Normalisation    │ Normalises pre-activations; stabilises   │
    │                        │ gradient magnitudes across layers        │
    │ Residual connections   │ Skip connections provide gradient highway│
    │ (ResNet, Transformer)  │ dL/dx = dL/d(x+F) = dL/dx + dL/dF        │
    │                        │ Gradient always has a direct path to x   │
    │ Gradient clipping      │ Caps gradient norm to prevent explosion  │
    │ Layer Normalisation    │ Stabilises gradients in Transformers     │
    └────────────────────────┴──────────────────────────────────────────┘

    The ResNet skip connection gradient:
        d/dx (x + F(x)) = 1 + dF/dx
    The "+1" means gradient is ALWAYS at least 1 — never vanishes
    regardless of what dF/dx does. This is why ResNets can have
    1000+ layers while plain networks fail beyond ~20.

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Chain Rule on a Scalar Computation Graph": {
        "description": (
            "Walk through backpropagation step-by-step on a small scalar "
            "computation graph. Show every local gradient, upstream gradient, "
            "and how they multiply to give the final answer. "
            "Verify with numerical differentiation."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  BACKPROPAGATION ON A SCALAR COMPUTATION GRAPH")
print("=" * 65)
print()
print("  Function: L = ((x * w + b) ** 2) / 2")
print("  (a single-neuron MSE loss — simplest possible network)")
print()

# ── Step 1: Choose values ──────────────────────────────────────────────────
x = 2.0    # input
w = 0.5    # weight (this is what we want to update)
b = -1.0   # bias
y = 1.0    # true label

print(f"  Values:   x={x}  w={w}  b={b}  y_true={y}")
print()

# ── Step 2: Forward pass — compute every intermediate node ────────────────
#   Node 1:  u  = x * w          (multiply)
#   Node 2:  z  = u + b          (add bias)
#   Node 3:  e  = z - y          (error: prediction minus truth)
#   Node 4:  L  = 0.5 * e^2      (MSE loss)

u = x * w
z = u + b
e = z - y
L = 0.5 * e ** 2

print("  FORWARD PASS:")
print(f"    u = x * w         = {x} * {w}   = {u}")
print(f"    z = u + b         = {u} + ({b})  = {z}")
print(f"    e = z - y_true    = {z} - {y}   = {e}")
print(f"    L = 0.5 * e^2     = 0.5 * {e}^2 = {L}")
print()

# ── Step 3: Backward pass — apply chain rule at each node ─────────────────
print("  BACKWARD PASS (right to left):")
print()

# dL/de: gradient of L w.r.t. e
dL_de = e
print(f"    dL/de = e = {dL_de}                    [d(0.5*e^2)/de = e]")

# dL/dz: gradient through the subtraction node (e = z - y, so de/dz = 1)
dL_dz = dL_de * 1.0
print(f"    dL/dz = dL/de * de/dz = {dL_de} * 1 = {dL_dz}   [e=z-y, so de/dz=1]")

# dL/du: gradient through addition node (z = u + b, so dz/du = 1)
dL_du = dL_dz * 1.0
print(f"    dL/du = dL/dz * dz/du = {dL_dz} * 1 = {dL_du}   [z=u+b, so dz/du=1]")

# dL/db: gradient through addition node (z = u + b, so dz/db = 1)
dL_db = dL_dz * 1.0
print(f"    dL/db = dL/dz * dz/db = {dL_dz} * 1 = {dL_db}   [z=u+b, so dz/db=1]")

# dL/dw: gradient through multiplication node (u = x * w, so du/dw = x)
dL_dw = dL_du * x
print(f"    dL/dw = dL/du * du/dw = {dL_du} * x = {dL_du} * {x} = {dL_dw}")
print(f"           [u=x*w, so du/dw=x — the input scales the gradient]")

# dL/dx: gradient through multiplication node (u = x * w, so du/dx = w)
dL_dx = dL_du * w
print(f"    dL/dx = dL/du * du/dx = {dL_du} * w = {dL_du} * {w} = {dL_dx}")
print()

# ── Step 4: Verify with numerical differentiation ─────────────────────────
def L_fn(w_val, b_val):
    u_ = x * w_val
    z_ = u_ + b_val
    e_ = z_ - y
    return 0.5 * e_ ** 2

eps = 1e-5
dL_dw_num = (L_fn(w + eps, b) - L_fn(w - eps, b)) / (2 * eps)
dL_db_num = (L_fn(w, b + eps) - L_fn(w, b - eps)) / (2 * eps)

print("  GRADIENT CHECK (numerical vs analytical):")
print(f"  {'Gradient':10s} | {'Analytical':>14} | {'Numerical':>14} | {'Error':>12}")
print(f"  {'─'*56}")
for name, analytic, numeric in [
        ("dL/dw", dL_dw, dL_dw_num),
        ("dL/db", dL_db, dL_db_num)]:
    err = abs(analytic - numeric)
    ok  = "✓" if err < 1e-8 else "✗"
    print(f"  {name:10s} | {analytic:14.8f} | {numeric:14.8f} | {err:12.2e} {ok}")

print()
print("  WEIGHT UPDATE (gradient descent with lr=0.1):")
lr = 0.1
w_new = w - lr * dL_dw
b_new = b - lr * dL_db
print(f"    w_new = w - lr * dL/dw = {w} - {lr}*{dL_dw} = {w_new:.4f}")
print(f"    b_new = b - lr * dL/db = {b} - {lr}*{dL_db} = {b_new:.4f}")

L_new = L_fn(w_new, b_new)
print()
print(f"  Loss before update: {L:.6f}")
print(f"  Loss after update:  {L_new:.6f}  (decrease = {L - L_new:.6f})")
print()
print("  Full chain rule expansion:")
print("  dL/dw = e * (de/dz) * (dz/du) * (du/dw)")
total = e * 1.0 * 1.0 * x
print(f"        = {e} * 1 * 1 * {x} = {total}")
print("  Every local gradient multiplied together = the final gradient.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Full MLP Backprop from Scratch": {
        "description": (
            "Implement a complete 3-layer MLP with explicit forward and backward "
            "passes from scratch — no PyTorch, no autograd. "
            "Every weight gradient derived manually and verified."
        ),
        "language": "python",
        "code": '''
import numpy as np

np.random.seed(42)

print("=" * 65)
print("  COMPLETE MLP BACKPROPAGATION FROM SCRATCH")
print("=" * 65)
print()
print("  Architecture: input(4) -> hidden1(8) -> hidden2(6) -> output(1)")
print("  Activation:   ReLU in hidden layers, Sigmoid at output")
print("  Loss:         Binary Cross-Entropy")
print()

# ── Activation functions ──────────────────────────────────────────────────
def relu(z):          return np.maximum(0, z)
def relu_back(dA, z): return dA * (z > 0)          # dL/dz = dL/da * (z>0)

def sigmoid(z):       return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def bce_loss(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# ── Network: store weights, biases, and all cached values ─────────────────
class MLP:
    """
    3-layer MLP with explicit forward and backward methods.
    All intermediate values cached for the backward pass.
    """
    def __init__(self, layer_dims, lr=0.01):
        self.lr = lr
        self.params = {}
        # He initialisation
        for l, (n_in, n_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:]), 1):
            self.params[f"W{l}"] = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
            self.params[f"b{l}"] = np.zeros((1, n_out))
        self.cache = {}    # stores forward-pass values needed for backprop

    def forward(self, X):
        """
        Forward pass — compute output and cache intermediate values.

        Layer 1: z1 = X  @ W1 + b1,   a1 = ReLU(z1)
        Layer 2: z2 = a1 @ W2 + b2,   a2 = ReLU(z2)
        Layer 3: z3 = a2 @ W3 + b3,   a3 = Sigmoid(z3)  [output]
        """
        # Layer 1
        z1 = X  @ self.params["W1"] + self.params["b1"]
        a1 = relu(z1)
        # Layer 2
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = relu(z2)
        # Layer 3 (output)
        z3 = a2 @ self.params["W3"] + self.params["b3"]
        a3 = sigmoid(z3)

        # Cache everything needed by backward pass
        self.cache = {"X": X, "z1": z1, "a1": a1,
                               "z2": z2, "a2": a2,
                               "z3": z3, "a3": a3}
        return a3.ravel()

    def backward(self, y):
        """
        Backward pass — compute gradients for all weights and biases.

        Starting from dL/da3 and working backwards to dL/dW1.
        """
        m    = len(y)
        grads = {}

        # ── Unpack cache ────────────────────────────────────────────────
        X,  z1, a1 = self.cache["X"],  self.cache["z1"], self.cache["a1"]
        z2, a2     = self.cache["z2"], self.cache["a2"]
        z3, a3     = self.cache["z3"], self.cache["a3"]

        # ── Layer 3 backward (Sigmoid + BCE) ────────────────────────────
        # dL/dz3 = a3 - y   (combined BCE + sigmoid gradient)
        dz3 = (a3 - y.reshape(-1, 1)) / m         # (batch, 1)
        grads["dW3"] = a2.T @ dz3                  # (n2, 1)
        grads["db3"] = dz3.sum(axis=0, keepdims=True)
        dA2          = dz3 @ self.params["W3"].T   # pass to prev layer

        # ── Layer 2 backward (ReLU) ──────────────────────────────────────
        # dL/dz2 = dL/da2 * (z2 > 0)
        dz2          = relu_back(dA2, z2)           # (batch, n2)
        grads["dW2"] = a1.T @ dz2                   # (n1, n2)
        grads["db2"] = dz2.sum(axis=0, keepdims=True)
        dA1          = dz2 @ self.params["W2"].T    # pass to prev layer

        # ── Layer 1 backward (ReLU) ──────────────────────────────────────
        dz1          = relu_back(dA1, z1)           # (batch, n1)
        grads["dW1"] = X.T @ dz1                    # (n_in, n1)
        grads["db1"] = dz1.sum(axis=0, keepdims=True)

        return grads

    def update(self, grads):
        """Gradient descent update for all parameters."""
        for l in range(1, 4):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def train_step(self, X, y):
        preds = self.forward(X)
        loss  = bce_loss(y, preds)
        grads = self.backward(y)
        self.update(grads)
        return loss, preds, grads

# ── Small demo ────────────────────────────────────────────────────────────
layer_dims = [4, 8, 6, 1]
net = MLP(layer_dims, lr=0.05)

# Single batch for detailed inspection
X_demo = np.random.randn(5, 4)
y_demo = np.array([1, 0, 1, 0, 1], dtype=float)

print("  DETAILED SINGLE FORWARD + BACKWARD STEP:")
print()
preds = net.forward(X_demo)
loss  = bce_loss(y_demo, preds)
grads = net.backward(y_demo)

print(f"  Input shape:        {X_demo.shape}")
print(f"  Prediction (raw):   {preds.round(4)}")
print(f"  True labels:        {y_demo.astype(int)}")
print(f"  BCE Loss:           {loss:.6f}")
print()
print("  GRADIENT SHAPES AND NORMS (each weight's gradient):")
print(f"  {'Parameter':8s} | {'Shape':>12} | {'Grad Norm':>12} | {'Max |grad|':>12}")
print(f"  {'─'*52}")
for l in range(1, 4):
    for pname in [f"W{l}", f"b{l}"]:
        g    = grads[f"d{pname}"]
        norm = np.linalg.norm(g)
        maxg = np.max(np.abs(g))
        print(f"  {pname:8s} | {str(g.shape):>12} | {norm:12.6f} | {maxg:12.6f}")

# ── Full training run ─────────────────────────────────────────────────────
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
        self.mean_ = X.mean(0); self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0; return self
    def transform(self, X):             return (X - self.mean_) / self.scale_
    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

def train_test_split(*arrays, test_size=0.2, random_state=None, **kw):
    rng  = _np_impl.random.default_rng(random_state); n = len(arrays[0])
    n_tr = int(n * (1 - test_size)); idx = rng.permutation(n)
    ti, vi = idx[:n_tr], idx[n_tr:]; out = []
    for a in arrays: out += [a[ti], a[vi]]
    return out


X_full, y_full = make_classification(n_samples=1000, n_features=4,
                                      n_informative=3, n_redundant=1,
                                      random_state=0)
X_full = StandardScaler().fit_transform(X_full)
X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2)

net2 = MLP([4, 16, 8, 1], lr=0.05)
print()
print("  TRAINING RUN (1000 samples, 4 features, 80 epochs):")
print(f"  {'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10}")
print(f"  {'─'*33}")

for epoch in range(80):
    idx   = np.random.permutation(len(y_tr))
    total_loss = 0
    for start in range(0, len(y_tr), 32):
        batch = idx[start:start+32]
        loss, _, _ = net2.train_step(X_tr[batch], y_tr[batch])
        total_loss += loss

    if (epoch + 1) % 10 == 0:
        preds_tr = net2.forward(X_tr)
        acc_tr   = ((preds_tr > 0.5).astype(int) == y_tr).mean()
        print(f"  {epoch+1:6d} | {total_loss/(len(y_tr)//32):10.5f} | {acc_tr:10.4f}")

preds_te = net2.forward(X_te)
acc_te   = ((preds_te > 0.5).astype(int) == y_te).mean()
print()
print(f"  Final Test Accuracy: {acc_te:.4f}")
print()
print("  Every gradient was computed by hand — no autograd used.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Numerical Gradient Checking": {
        "description": (
            "Implement a complete gradient checker that verifies analytical "
            "backprop gradients against numerical finite differences. "
            "Deliberately introduce a bug and show the checker catches it."
        ),
        "language": "python",
        "code": '''
import numpy as np

np.random.seed(0)

print("=" * 65)
print("  NUMERICAL GRADIENT CHECKING")
print("=" * 65)
print()
print("  Method: centred finite difference")
print("  df/dx ~ [f(x+eps) - f(x-eps)] / (2*eps)    eps = 1e-5")
print()

# ── Network to check ──────────────────────────────────────────────────────
def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward_and_loss(params, X, y):
    """Single forward pass returning the scalar loss."""
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    p  = sigmoid(z2).ravel()
    p  = np.clip(p, 1e-7, 1-1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def backward_gradients(params, X, y):
    """Analytical gradients via backprop."""
    W1, b1, W2, b2 = params
    m = len(y)

    # Forward
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    p  = sigmoid(z2).ravel()

    # Backward
    dz2 = (p - y).reshape(-1, 1) / m
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    dA1 = dz2 @ W2.T
    dz1 = dA1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return [dW1, db1, dW2, db2]

def gradient_check(params, X, y, eps=1e-5):
    """
    Compare analytical vs numerical gradients for every parameter.
    Returns relative error for each parameter tensor.
    """
    analytic_grads = backward_gradients(params, X, y)
    results = []

    for i, (p, ag) in enumerate(zip(params, analytic_grads)):
        numeric_grad = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]

            p[idx] = orig + eps
            loss_p = forward_and_loss(params, X, y)

            p[idx] = orig - eps
            loss_m = forward_and_loss(params, X, y)

            numeric_grad[idx] = (loss_p - loss_m) / (2 * eps)
            p[idx] = orig          # restore
            it.iternext()

        # Relative error
        num   = np.linalg.norm(ag - numeric_grad)
        denom = np.linalg.norm(ag) + np.linalg.norm(numeric_grad)
        rel_err = num / (denom + 1e-10)
        results.append((i, rel_err, ag, numeric_grad))

    return results

# ── Set up small network and data ─────────────────────────────────────────
n_in, n_h, n_out = 5, 6, 1
X_check = np.random.randn(8, n_in)
y_check = np.random.randint(0, 2, 8).astype(float)

W1 = np.random.randn(n_in, n_h) * 0.3
b1 = np.random.randn(1, n_h)   * 0.1
W2 = np.random.randn(n_h, n_out) * 0.3
b2 = np.random.randn(1, n_out) * 0.1
params = [W1, b1, W2, b2]

param_names = ["W1", "b1", "W2", "b2"]

# ── TEST 1: Correct implementation ────────────────────────────────────────
print("  TEST 1: CORRECT IMPLEMENTATION")
check_results = gradient_check(params, X_check, y_check)
print(f"  {'Parameter':>8} | {'Relative Error':>16} | {'Status':>10}")
print(f"  {'─'*42}")
all_ok = True
for i, (idx, rel_err, ag, ng) in enumerate(check_results):
    status = "✓ PASS" if rel_err < 1e-5 else "✗ FAIL"
    if rel_err >= 1e-5: all_ok = False
    print(f"  {param_names[idx]:>8} | {rel_err:16.2e} | {status}")

print(f"  Overall: {'ALL PASS ✓' if all_ok else 'SOME FAILED ✗'}")
print()

# ── TEST 2: Introduce a bug — wrong gradient for W2 ───────────────────────
def backward_with_bug(params, X, y):
    """Same as above but with a sign error in dW2."""
    W1, b1, W2, b2 = params
    m = len(y)
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    p  = sigmoid(z2).ravel()

    dz2 = (p - y).reshape(-1, 1) / m
    dW2 = -a1.T @ dz2          # BUG: wrong sign!
    db2 = dz2.sum(axis=0)
    dA1 = dz2 @ W2.T
    dz1 = dA1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return [dW1, db1, dW2, db2]

print("  TEST 2: BUGGY IMPLEMENTATION (wrong sign on dW2)")
analytic_buggy = backward_with_bug(params, X_check, y_check)
check_results_buggy = gradient_check(params, X_check, y_check)

# Override analytic with buggy version for comparison
print(f"  {'Parameter':>8} | {'Relative Error':>16} | {'Status':>10}")
print(f"  {'─'*42}")
analytic_correct = backward_gradients(params, X_check, y_check)
bug_found = False
for i, name in enumerate(param_names):
    ag_c = analytic_correct[i]
    ag_b = analytic_buggy[i]
    ng   = check_results_buggy[i][3]   # numeric grad from correct checker

    # Compare buggy analytic to numeric
    num   = np.linalg.norm(ag_b - ng)
    denom = np.linalg.norm(ag_b) + np.linalg.norm(ng)
    rel   = num / (denom + 1e-10)
    status = "✗ BUG FOUND!" if rel > 1e-3 else "✓ pass"
    if rel > 1e-3: bug_found = True
    print(f"  {name:>8} | {rel:16.2e} | {status}")

print(f"  Bug detected: {'YES ✓' if bug_found else 'NO'}")
print()
print("  GRADIENT CHECK THRESHOLDS:")
print("  < 1e-7 : Perfect — implementation almost certainly correct")
print("  < 1e-5 : Fine — acceptable floating point error")
print("  < 1e-3 : Suspicious — review your backward pass")
print("  > 1e-2 : Wrong — there is definitely a bug")
print()
print("  IMPORTANT: Only use gradient checking for DEBUGGING.")
print("  Never use it during training (requires N forward passes).")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Gradient Flow Visualisation Across Network Depth": {
        "description": (
            "Train networks of depth 5, 10, and 20 with Sigmoid vs ReLU. "
            "Plot the gradient magnitude at each layer to visualise "
            "vanishing gradients in sigmoid networks and stable flow in ReLU."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Measure gradient norms in each layer after one backward pass ───────────
def relu(z):     return np.maximum(0, z)
def sigmoid(z):  return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def tanh_(z):    return np.tanh(z)

def compute_grad_norms(n_layers, n_units, activation_fn,
                       act_name, batch=128, n_in=32):
    """
    Build a deep network, do one forward + backward pass,
    and return the gradient L2-norm at each layer.
    """
    np.random.seed(1)
    # He init for all activations (fair comparison)
    Ws = [np.random.randn(n_in if l==0 else n_units, n_units)
          * np.sqrt(2 / (n_in if l == 0 else n_units))
          for l in range(n_layers)]
    bs = [np.zeros(n_units) for _ in range(n_layers)]
    Ws.append(np.random.randn(n_units, 1) * 0.01)  # output weight
    bs.append(np.zeros(1))

    X = np.random.randn(batch, n_in)
    y = (np.random.rand(batch) > 0.5).astype(float)

    # Forward pass — cache pre-activations and activations
    caches = []
    A = X
    for l in range(n_layers):
        Z = A @ Ws[l] + bs[l]
        A_new = activation_fn(Z)
        caches.append((A, Z, A_new))
        A = A_new
    # Output layer
    logit = (A @ Ws[-1] + bs[-1]).ravel()
    p     = sigmoid(logit)

    # Backward
    dA = ((p - y) / batch).reshape(-1, 1) @ Ws[-1].T

    grad_norms = []
    for l in reversed(range(n_layers)):
        A_prev, Z, A_curr = caches[l]
        if act_name == "Sigmoid":
            local_grad = sigmoid(Z) * (1 - sigmoid(Z))
        elif act_name == "Tanh":
            local_grad = 1 - tanh_(Z) ** 2
        else:  # ReLU
            local_grad = (Z > 0).astype(float)

        dZ  = dA * local_grad
        dW  = A_prev.T @ dZ
        dA  = dZ @ Ws[l].T
        grad_norms.insert(0, np.linalg.norm(dW))

    return grad_norms

# ── Run experiments ────────────────────────────────────────────────────────
print("=" * 65)
print("  GRADIENT FLOW: VANISHING GRADIENTS ACROSS NETWORK DEPTH")
print("=" * 65)
print()

configs = [
    ("Sigmoid", sigmoid, "tomato",    "--"),
    ("Tanh",    tanh_,   "orange",    "-."),
    ("ReLU",    relu,    "steelblue", "-"),
]

depth_list = [5, 10, 20]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Gradient Magnitude per Layer: Sigmoid vs Tanh vs ReLU",
             fontsize=12, fontweight="bold")

for ax, n_layers in zip(axes, depth_list):
    print(f"  Depth = {n_layers} layers:")
    print(f"  {'Activation':10s} | {'Grad@L1':>12} | {'Grad@last':>12} | {'Ratio':>10}")
    print(f"  {'─'*52}")

    for act_name, act_fn, colour, ls in configs:
        norms = compute_grad_norms(n_layers, 64, act_fn, act_name)
        ax.semilogy(range(1, n_layers+1), norms,
                    color=colour, linestyle=ls, lw=2.5, label=act_name)

        ratio = norms[0] / (norms[-1] + 1e-30)
        flag  = "← VANISHED" if norms[0] < 1e-6 else ""
        print(f"  {act_name:10s} | {norms[0]:12.3e} | {norms[-1]:12.3e} | "
              f"{ratio:10.1f}×  {flag}")
    print()

    ax.set_xlabel("Layer (1 = earliest)"); ax.set_ylabel("Gradient L2-norm")
    ax.set_title(f"Depth = {n_layers} layers")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.axhline(1e-7, color="red", linestyle=":", lw=1, alpha=0.6,
               label="Vanishing threshold")

plt.tight_layout()
plt.savefig("gradient_flow.png", dpi=120)
print("  Plot saved → gradient_flow.png")
print()

# ── Quantify vanishing effect analytically ────────────────────────────────
print("  ANALYTICAL VANISHING GRADIENT CALCULATION:")
print()
print("  Sigmoid: each layer multiplies gradient by σ(z)(1-σ(z)) <= 0.25")
for L in [5, 10, 20]:
    worst = 0.25 ** L
    print(f"    Depth {L:2d}: worst-case multiplier = 0.25^{L} = {worst:.2e}")

print()
print("  ReLU: gradient is either 0 or 1 — no attenuation for active neurons")
print("  ResNet skip: gradient = 1 + dF/dx >= 1 always — no vanishing possible")
print()
print("  This is why residual networks (ResNets, Transformers) can be")
print("  100-1000 layers deep while plain sigmoid networks fail beyond ~5.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Mini Autograd Engine": {
        "description": (
            "Build a miniature automatic differentiation engine from scratch. "
            "Each operation records a backward function. "
            "Call .backward() and get all gradients automatically — "
            "exactly how PyTorch and JAX work internally."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  MINI AUTOGRAD ENGINE — HOW PYTORCH WORKS INTERNALLY")
print("=" * 65)
print()

class Tensor:
    """
    A scalar-valued tensor that tracks its computation history
    and can compute gradients via backpropagation.

    Mirrors the core of PyTorch's autograd in ~100 lines.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data     = float(data)
        self.grad     = 0.0          # gradient accumulates here
        self._backward = lambda: None  # function to compute local gradients
        self._prev    = set(_children)
        self._op      = _op          # for debugging: what created this node
        self.label    = label

    def __repr__(self):
        return f"Tensor(data={self.data:.4f}, grad={self.grad:.4f}, op='{self._op}')"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out   = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # Addition: gradient distributes to both inputs unchanged
            self.grad  += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out   = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # Multiplication: gradient scaled by the other operand
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += out.grad * (exponent * self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __neg__(self):   return self * -1
    def __sub__(self, other): return self + (-other if isinstance(other, Tensor)
                                             else Tensor(-other))
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Tensor(other) - self
    def __truediv__(self, other): return self * other**-1

    def relu(self):
        out = Tensor(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def exp(self):
        e   = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += out.grad * e     # d/dx(e^x) = e^x

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-10), (self,), "log")

        def _backward():
            self.grad += out.grad * (1.0 / (self.data + 1e-10))

        out._backward = _backward
        return out

    def backward(self):
        """
        Topological sort of the computation graph, then call
        _backward() in reverse order (from output to inputs).
        """
        topo  = []
        visited = set()
        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0      # dL/dL = 1 (seed gradient)
        for node in reversed(topo):
            node._backward()

# ── Demo 1: Scalar computation graph ─────────────────────────────────────
print("  DEMO 1 — Scalar: L = 0.5 * (x*w + b - y)^2")
print()

x = Tensor(2.0,  label="x")
w = Tensor(0.5,  label="w")
b = Tensor(-1.0, label="b")
y = Tensor(1.0,  label="y")

u   = x * w           ; u.label   = "u=x*w"
z   = u + b           ; z.label   = "z=u+b"
e   = z - y           ; e.label   = "e=z-y"
e2  = e ** 2          ; e2.label  = "e^2"
L   = Tensor(0.5) * e2; L.label   = "L"

L.backward()

print(f"  Forward:  u={u.data:.3f}  z={z.data:.3f}  e={e.data:.3f}  L={L.data:.3f}")
print(f"  Backward: dL/dw={w.grad:.3f}  dL/db={b.grad:.3f}  dL/dx={x.grad:.3f}")
print()

# Verify numerically
eps = 1e-5
def L_fn(wv, bv):
    u_ = 2.0*wv; z_ = u_+bv; e_ = z_-1.0; return 0.5*e_**2
dw_num = (L_fn(0.5+eps, -1.0) - L_fn(0.5-eps, -1.0)) / (2*eps)
db_num = (L_fn(0.5, -1.0+eps) - L_fn(0.5, -1.0-eps)) / (2*eps)
print(f"  Numerical check:  dL/dw={dw_num:.3f}  dL/db={db_num:.3f}")
print(f"  Match: dw={'✓' if abs(w.grad - dw_num) < 1e-8 else '✗'}  "
      f"db={'✓' if abs(b.grad - db_num) < 1e-8 else '✗'}")
print()

# ── Demo 2: Single neuron with ReLU ──────────────────────────────────────
print("  DEMO 2 — ReLU neuron: a = ReLU(w1*x1 + w2*x2 + b)")
print()

w1 = Tensor(0.4,  label="w1")
w2 = Tensor(-0.3, label="w2")
b2 = Tensor(0.1,  label="b")
x1 = Tensor(1.5,  label="x1")
x2 = Tensor(-0.5, label="x2")

pre_act = w1*x1 + w2*x2 + b2
a       = pre_act.relu()
loss    = a ** 2        # trivial loss for demo

loss.backward()

print(f"  pre_activation = {pre_act.data:.4f}")
print(f"  a = ReLU({pre_act.data:.4f}) = {a.data:.4f}")
print(f"  loss = a^2 = {loss.data:.4f}")
print()
print(f"  dL/dw1 = {w1.grad:.4f}   (x1={x1.data} scaled gradient)")
print(f"  dL/dw2 = {w2.grad:.4f}   (x2={x2.data} scaled gradient)")
print(f"  dL/db  = {b2.grad:.4f}")
print()

# ── Demo 3: What happens when ReLU is dead ────────────────────────────────
print("  DEMO 3 — Dead ReLU neuron (pre-activation < 0)")

w_dead = Tensor(0.1,  label="w_dead")
x_neg  = Tensor(-5.0, label="x_neg")
b_neg  = Tensor(-2.0, label="b_neg")

pre_dead = w_dead * x_neg + b_neg    # = 0.1*(-5) + (-2) = -2.5
a_dead   = pre_dead.relu()           # = ReLU(-2.5) = 0
loss_dead = a_dead ** 2

loss_dead.backward()

print(f"  pre_activation = {pre_dead.data:.4f}  (negative)")
print(f"  a = ReLU({pre_dead.data:.4f}) = {a_dead.data:.4f}  (dead!)")
print(f"  dL/dw = {w_dead.grad:.4f}  (ZERO — no gradient flows!)")
print(f"  The weight w_dead={w_dead.data} will NEVER update. Neuron is dead.")
print()
print("  This is exactly the Dying ReLU problem — the autograd engine")
print("  shows clearly why it occurs: the backward gate is 0.")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · Residual Connections & Gradient Highway": {
        "description": (
            "Show numerically why skip connections solve vanishing gradients. "
            "Compare gradient magnitude at layer 1 in a plain deep network "
            "vs a residual network of the same depth. "
            "Plot the gradient flow for both across depths 5, 10, 20, 50."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

def relu(z):     return np.maximum(0, z)
def sigmoid(z):  return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def grad_norm_plain(n_layers, n_units=64, batch=64, n_in=32):
    """
    Plain deep network: no skip connections.
    Returns gradient norm at each layer.
    """
    np.random.seed(1)
    Ws = [np.random.randn(n_in if l == 0 else n_units, n_units)
          * np.sqrt(2 / (n_in if l == 0 else n_units))
          for l in range(n_layers)]
    Ws.append(np.random.randn(n_units, 1) * 0.01)

    X = np.random.randn(batch, n_in)
    y = (np.random.rand(batch) > 0.5).astype(float)

    # Forward
    caches, A = [], X
    for l in range(n_layers):
        Z = A @ Ws[l]
        A_new = relu(Z)
        caches.append((A, Z))
        A = A_new

    logit = (A @ Ws[-1]).ravel()
    p = sigmoid(logit)
    dA = ((p - y) / batch).reshape(-1, 1) @ Ws[-1].T

    norms = []
    for l in reversed(range(n_layers)):
        A_prev, Z = caches[l]
        dZ = dA * (Z > 0)
        norms.insert(0, np.linalg.norm(A_prev.T @ dZ))
        dA = dZ @ Ws[l].T
    return norms

def grad_norm_residual(n_layers, n_units=64, batch=64, n_in=32):
    """
    Residual network: output = F(x) + x (skip connection).
    Returns gradient norm at each layer.

    Backward through skip:
        d/dx [F(x) + x] = dF/dx + 1
        The '+1' means gradient is always >= 1 → no vanishing.
    """
    np.random.seed(1)
    # Project input to n_units if needed
    W_proj = np.random.randn(n_in, n_units) * np.sqrt(2 / n_in)
    Ws = [np.random.randn(n_units, n_units) * np.sqrt(2 / n_units)
          for _ in range(n_layers)]
    Ws.append(np.random.randn(n_units, 1) * 0.01)

    X = np.random.randn(batch, n_in)
    y = (np.random.rand(batch) > 0.5).astype(float)

    # Forward
    A = X @ W_proj    # project input
    caches = []
    for l in range(n_layers):
        Z     = A @ Ws[l]
        F     = relu(Z)
        A_new = F + A          # SKIP CONNECTION: output = F(x) + x
        caches.append((A, Z, F))
        A = A_new

    logit = (A @ Ws[-1]).ravel()
    p = sigmoid(logit)
    dA = ((p - y) / batch).reshape(-1, 1) @ Ws[-1].T

    norms = []
    for l in reversed(range(n_layers)):
        A_prev, Z, F = caches[l]
        # dA flows back through BOTH the skip path AND the ReLU path
        dF = dA * (Z > 0)                  # through the ReLU branch
        dW = A_prev.T @ dF
        norms.insert(0, np.linalg.norm(dW))
        # Gradient through skip: dA_prev = dF @ W.T  +  dA (identity path)
        dA = dF @ Ws[l].T + dA             # '+dA' = gradient highway
    return norms

# ── Plot and compare ───────────────────────────────────────────────────────
depth_list = [5, 10, 20, 50]

print("=" * 65)
print("  RESIDUAL CONNECTIONS: GRADIENT HIGHWAY EFFECT")
print("=" * 65)
print()
print("  Plain network gradient at layer 1 vs Residual network:")
print(f"  {'Depth':>7} | {'Plain L1 grad':>15} | {'Residual L1 grad':>17} | {'Speedup':>10}")
print(f"  {'─'*57}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Gradient Flow: Plain Deep Network vs Residual Network",
             fontsize=12, fontweight="bold")

plain_l1_norms, res_l1_norms = [], []
for depth in depth_list:
    n_plain  = grad_norm_plain(depth)
    n_res    = grad_norm_residual(depth)
    plain_l1_norms.append(n_plain[0])
    res_l1_norms.append(n_res[0])
    speedup  = n_res[0] / (n_plain[0] + 1e-30)
    print(f"  {depth:7d} | {n_plain[0]:15.3e} | {n_res[0]:17.3e} | {speedup:10.1f}x")

    layers = list(range(1, depth + 1))
    axes[0].semilogy(layers, n_plain, lw=2, alpha=0.8,
                     label=f"Plain   depth={depth}")
    axes[1].semilogy(layers, n_res, lw=2, linestyle="--", alpha=0.8,
                     label=f"Residual depth={depth}")

print()
print("  In a depth-50 plain network, the gradient at layer 1 may be")
print("  10^15 times SMALLER than in a residual network of the same depth.")
print()

for ax, title in zip(axes, ["Plain Network (no skip)", "Residual Network (with skip)"]):
    ax.set_xlabel("Layer number (1 = closest to input)")
    ax.set_ylabel("Gradient L2-norm (log scale)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axhline(1e-7, color="red", linestyle=":", alpha=0.5,
               label="Vanishing threshold")

# ── Side plot: gradient at layer 1 vs depth ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogy(depth_list, plain_l1_norms, "tomato", lw=2.5, marker="o",
             ms=8, label="Plain network")
ax2.semilogy(depth_list, res_l1_norms, "steelblue", lw=2.5,
             marker="s", ms=8, label="Residual network")
ax2.set_xlabel("Network Depth (number of layers)")
ax2.set_ylabel("Gradient norm at layer 1 (log scale)")
ax2.set_title("Gradient at Earliest Layer vs Network Depth")
ax2.legend(fontsize=10); ax2.grid(alpha=0.3)
ax2.annotate("Gradient vanishes", xy=(depth_list[-1], plain_l1_norms[-1]),
             xytext=(30, plain_l1_norms[-1]*1000),
             arrowprops=dict(arrowstyle="->", color="tomato"), color="tomato",
             fontsize=9)
ax2.annotate("Gradient survives", xy=(depth_list[-1], res_l1_norms[-1]),
             xytext=(30, res_l1_norms[-1]*0.01),
             arrowprops=dict(arrowstyle="->", color="steelblue"),
             color="steelblue", fontsize=9)

plt.tight_layout()
fig.savefig("gradient_flow_comparison.png", dpi=120)
fig2.savefig("gradient_vs_depth.png", dpi=120)
print("  Plots saved → gradient_flow_comparison.png, gradient_vs_depth.png")
print()
print("  KEY INSIGHT — skip connection backward pass:")
print("  Plain:    dA_prev  = dF @ W.T            (multiplicative chain)")
print("  Residual: dA_prev  = dF @ W.T  +  dA     (always >= dA)")
print("  The +dA term is a DIRECT COPY of the upstream gradient — a highway")
print("  that bypasses all the multiplicative attenuation.")
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
        from Training_Core.visuals.backpropagation_visual import (
            BACKPROP_VISUAL_HTML,
            BACKPROP_VISUAL_HEIGHT,
        )
        visual_html   = BACKPROP_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = BACKPROP_VISUAL_HEIGHT
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