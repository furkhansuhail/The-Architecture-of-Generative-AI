"""
Information Theory
==================

Shannon's framework for quantifying uncertainty, information, and
the fundamental limits of communication and compression.
In ML: every loss function, every feature selection method, every
VAE objective, and every RLHF penalty connects back to these ideas.

"""

import textwrap
import re

TOPIC_NAME   = "Information Theory"
DISPLAY_NAME = "02 · Information Theory"
ICON         = "📡"
SUBTITLE     = "Entropy, Mutual Information & KL Divergence — The Language of Uncertainty"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### PART 1 — SELF-INFORMATION & ENTROPY

### Self-Information (Surprisal)

The information content of a single outcome x with probability P(x):

    I(x) = -log P(x)

Intuition: rare events are more informative (surprising).
    • A fair coin shows heads (P=0.5): I = -log₂(0.5) = 1 bit of info
    • A biased coin (P=0.99) shows heads: I = -log₂(0.99) ≈ 0.014 bits (boring)
    • A black swan event (P=0.001): I = -log₂(0.001) ≈ 10 bits (very surprising)

Units depend on the base of log:
    • log₂ → bits    (natural for binary communication)
    • logₑ → nats    (natural for calculus, used in PyTorch/NumPy losses)
    • log₁₀ → bans   (rare in practice)
    Conversion: 1 nat = 1/ln(2) ≈ 1.443 bits


### Shannon Entropy H(X)

Entropy is the EXPECTED self-information — the average surprise
(or average uncertainty) of a random variable X:

    H(X) = -Σₓ P(x) · log P(x)    [discrete]
    H(X) = -∫ f(x) · log f(x) dx  [continuous — called differential entropy]

    Convention: 0 · log(0) = 0  (by continuity, since x·log(x)→0 as x→0)

Properties and bounds:

    • H(X) ≥ 0 always (for discrete; continuous can be negative)
    • H(X) = 0: deterministic variable (no uncertainty)
    • H(X) is maximised by the UNIFORM distribution:
        H_max = log₂(k) bits for a k-outcome discrete variable
      (A fair k-sided die has maximum entropy)
    • H(X) is CONCAVE in P

    Diagram 1 — Entropy of a Binary Variable vs Bias p:

    H(p) │
    1.0  │         ╭──────╮
         │       ╭─╯      ╰─╮
         │      ╱            ╲
    0.5  │     ╱              ╲
         │    ╱                ╲
    0.0  │───╱                  ╲───
         └────────────────────────── p (probability of heads)
              0   0.2  0.4  0.5  0.8  1.0

    H(0) = 0  (always tails — certain)
    H(0.5) = 1 bit  (fair coin — maximum uncertainty)
    H(1) = 0  (always heads — certain)
    H is symmetric: H(p) = H(1-p)


### PART 2 — JOINT, CONDITIONAL ENTROPY & THE CHAIN RULE

### Joint Entropy H(X, Y)

    H(X, Y) = -Σₓ Σᵧ P(x,y) · log P(x,y)

    H(X, Y) measures the total uncertainty in knowing BOTH X and Y together.

    Bounds:
        max(H(X), H(Y))  ≤  H(X, Y)  ≤  H(X) + H(Y)
        Lower bound achieved when one variable determines the other.
        Upper bound achieved when X and Y are independent.


### Conditional Entropy H(Y | X)

    H(Y | X) = -Σₓ P(x) Σᵧ P(y|x) log P(y|x)
             = H(X, Y) - H(X)

    Interpretation: average remaining uncertainty in Y after observing X.

    Key property (conditioning reduces entropy):
        H(Y | X)  ≤  H(Y)
        Knowing X never increases uncertainty about Y on average.
        Equality iff X and Y are independent.


### Chain Rule for Entropy

    H(X, Y) = H(X) + H(Y | X)
             = H(Y) + H(X | Y)

    "Total uncertainty = marginal uncertainty + conditional uncertainty"

    Diagram 2 — Entropy Venn Diagram:

         ┌────────────────────────────────────┐
         │         H(X, Y)                    │
         │   ┌──────────────────┐             │
         │   │      H(X)        │             │
         │   │   ┌──────────────┼──────────┐  │
         │   │   │  I(X;Y)      │  H(Y)    │  │
         │   │   │ (shared)     │          │  │
         │   │   │              │          │  │
         │   │H(X|Y)            │H(Y|X)    │  │
         │   └──────────────────┘          │  │
         │                                 │  │
         └─────────────────────────────────┘  │
                                              │
    H(X|Y) = uncertainty in X given Y (left exclusive region)
    H(Y|X) = uncertainty in Y given X (right exclusive region)
    I(X;Y) = mutual information (overlapping region)
    H(X,Y) = entire box area



### PART 3 — MUTUAL INFORMATION

### Definition and Interpretations

    I(X; Y) = H(X) + H(Y) − H(X, Y)
            = H(X) − H(X | Y)       ← reduction in X's uncertainty given Y
            = H(Y) − H(Y | X)       ← reduction in Y's uncertainty given X
            = KL(P(X,Y) ‖ P(X)P(Y)) ← divergence from independence

    I(X; Y) ≥ 0 always
    I(X; Y) = 0 iff X and Y are independent (no shared information)
    I(X; X) = H(X)  (a variable is maximally informative about itself)
    I(X; Y) = I(Y; X)  (symmetric — unlike KL divergence)

### Why Mutual Information Is Better Than Correlation

    Pearson correlation ρ(X,Y) only captures LINEAR relationships.
    Mutual information I(X;Y) captures ALL statistical dependencies:
    linear, non-linear, and multi-modal.

    Example:
        Y = X²:   ρ(X,Y) ≈ 0  (not linearly related)
                  I(X;Y) > 0  (but they ARE related!)

    For feature selection in ML: MI-based methods find features
    that are informative about the target regardless of linearity.


### PART 4 — INFORMATION GAIN & DECISION TREES

### How Decision Trees Use Entropy

A decision tree greedily picks the feature and split value that maximises
Information Gain (IG) at each node:

    IG(Y; feature A) = H(Y) − H(Y | A)

    H(Y):     entropy of the labels before the split (parent node)
    H(Y | A): weighted average entropy of labels in child nodes after split

    Equivalently: IG = reduction in label uncertainty from knowing A.

    Diagram 3 — Decision Tree Split (binary classification):

    BEFORE SPLIT:             AFTER SPLIT on feature A ≤ 3.5:
    ┌─────────────────┐      ┌────────────┐    ┌────────────┐
    │ + + + − − − + + │      │ + + + − −  │    │ + +        │
    │ H = 0.954 bits  │  →   │ H = 0.971  │    │ H = 0.0    │
    │ (8 samples)     │      │ (5 samples)│    │ (3 samples)│
    └─────────────────┘      └────────────┘    └────────────┘

    IG = 0.954 - (5/8 × 0.971 + 3/8 × 0.0) = 0.954 - 0.607 = 0.347 bits

    Larger IG → better split → chosen first.

### Gini Impurity vs Entropy

    Gini = 1 - Σₖ P(k)²         (sklearn default for trees)
    Entropy = -Σₖ P(k) log P(k)  (ID3, C4.5 algorithm)

    Both measure node impurity. Results are usually nearly identical.
    Gini is slightly faster (no log computation).
    Entropy via IG has a stronger information-theoretic justification.


### PART 5 — KL DIVERGENCE (DEEP DIVE)

### The Full Picture

    KL(P ‖ Q) = Σₓ P(x) log(P(x)/Q(x))
              = H(P, Q) − H(P)              ← cross-entropy minus entropy
              = E_P [log P(x) − log Q(x)]  ← expected log-likelihood ratio

    Properties:
        • KL(P ‖ Q) ≥ 0  always  (Gibbs' inequality)
        • KL(P ‖ Q) = 0 iff P = Q almost everywhere
        • NOT symmetric: KL(P‖Q) ≠ KL(Q‖P) in general
        • Jensen-Shannon divergence: JSD = ½KL(P‖M) + ½KL(Q‖M), M=(P+Q)/2
          → bounded in [0,1], symmetric — often preferred in practice

### Forward vs Reverse KL: Zero-Forcing vs Mass-Covering

    Forward KL (P ‖ Q): "mean-seeking" — Q tries to cover ALL of P
        minimise E_P[log P/Q] → Q spreads to match wherever P has mass
        If P has two modes, Q will spread between them (mean-seeking).
        Used in: supervised learning, MLE (P=data, Q=model)

    Reverse KL (Q ‖ P): "mode-seeking" — Q concentrates on ONE mode of P
        minimise E_Q[log Q/P] → Q avoids regions where P=0 (zero-forcing)
        If P has two modes, Q will snap to one (ignores the other).
        Used in: variational inference, VAEs

    This asymmetry is WHY variational autoencoders tend to produce
    blurry images (the approximate posterior Q misses modes of P).


### PART 6 — CROSS-ENTROPY & CONNECTIONS TO ML LOSSES

### Cross-Entropy H(P, Q)

    H(P, Q) = -Σₓ P(x) log Q(x)
            = H(P) + KL(P ‖ Q)

    When P is the true data distribution and Q is the model:
    Minimising H(P, Q) w.r.t. Q ≡ minimising KL(P ‖ Q)
    (since H(P) is constant w.r.t. Q)

    This is the theoretical justification for cross-entropy loss:
    training minimises the divergence between the model and reality.


### Information Theory in ML — Connection Map

    ┌────────────────────────────────────────────────────────────┐
    │ ML Concept                    Information Theory Basis     │
    ├────────────────────────────────────────────────────────────┤
    │ Cross-entropy loss           H(P, Q) = H(P) + KL(P‖Q)      │
    │ MLE for classifiers          Minimise KL(data ‖ model)     │
    │ Decision tree splitting      Information Gain = H−H(Y|A)   │
    │ Feature selection            Mutual Information I(X;Y)     │
    │ VAE ELBO objective           Reconstruction − KL(q‖p)      │
    │ RLHF KL penalty              KL(model ‖ reference model)   │
    │ Dropout (info bottleneck)    Compress info through network │
    │ Model calibration            Minimise H(P_true, P_model)   │
    └────────────────────────────────────────────────────────────┘


### PART 7 — SOURCE CODING & DATA COMPRESSION

### Shannon's Source Coding Theorem (Noiseless Coding Theorem)

Entropy H(X) is not just an abstraction — it is the theoretical minimum
average number of bits required to losslessly encode outcomes of X.

    Shannon's First Theorem:
        For any lossless code over a source X:

        H(X) ≤ L_avg < H(X) + 1

    where L_avg = Σₓ P(x) · length(codeword for x)

    Corollary: no lossless code can on average use fewer bits than H(X).
    This is the fundamental limit of compression.

    Intuition: frequent symbols deserve short codewords (few bits),
    rare symbols can afford long codewords (many bits). Entropy tells
    us exactly how well we can exploit this.


### Fixed-Length vs Variable-Length Codes

    Fixed-length: all codewords same length → wasteful for skewed distributions
    Variable-length: shorter codes for frequent symbols → approaches H(X)

    Example — encoding a 4-symbol source:

    Symbol  P(x)   Fixed (2 bits)   Huffman code   length
    ──────  ─────  ─────────────    ────────────   ──────
    A       0.50   00               0              1
    B       0.25   01               10             2
    C       0.125  10               110            3
    D       0.125  11               111            3

    Fixed avg:   2.000 bits
    Huffman avg: 0.50×1 + 0.25×2 + 0.125×3 + 0.125×3 = 1.750 bits
    H(X):        0.50×1 + 0.25×2 + 0.125×3 + 0.125×3 = 1.750 bits ← exact!

    This source has H(X) = 1.75 bits, and Huffman achieves it exactly.


### Huffman Coding Algorithm

Build a binary tree greedily, bottom-up:
    1. Sort symbols by probability (ascending)
    2. Merge the two lowest-probability symbols into a parent node
       with probability = sum of children
    3. Repeat until one root node remains
    4. Traverse the tree: left edge = 0, right edge = 1

    Each leaf's depth = its codeword length.
    Huffman is provably OPTIMAL among all prefix-free variable-length codes.


### Perplexity

Perplexity is the exponentiated cross-entropy — the standard evaluation
metric for language models:

    Perplexity(P, Q) = 2^H(P, Q) = 2^{-Σ P(x) log₂ Q(x)}   [bits]
                     = e^H(P, Q)                               [nats]

    Interpretation: the "effective vocabulary size" the model is confused
    over at each step. Lower is better.

    Perfect model: PP = 2^H(P) (matching source entropy)
    Random model (uniform over V words): PP = V

    A language model with PP = 50 is roughly as uncertain as a uniform
    distribution over 50 equally likely next words.

    Connection to cross-entropy loss:
        Minimising mean cross-entropy loss ≡ minimising perplexity
        → the standard training objective for GPT, BERT, etc.


### PART 8 — MAXIMUM ENTROPY PRINCIPLE

### The Principle

"Among all distributions consistent with your known constraints,
 choose the one with maximum entropy."

Maximum entropy (MaxEnt) is the least assumptive choice — it makes no
additional assumptions beyond what is known. Injecting extra structure
(lower entropy) would be claiming knowledge you do not have.

    Formally: maximise H(X) subject to:
        Σ P(x) = 1                              (normalisation)
        E[fₖ(X)] = μₖ for k = 1, ..., m        (moment constraints)

    The solution always takes the exponential family form:
        P*(x) ∝ exp(Σₖ λₖ fₖ(x))

    where λₖ are Lagrange multipliers tuned to match the constraints.


### MaxEnt Recovers Every Common Distribution

    ┌──────────────────────────────────────────────────────────────┐
    │ Constraints                         Max-Entropy Solution     │
    ├──────────────────────────────────────────────────────────────┤
    │ None (finite support {1..k})        Uniform(1/k, ..., 1/k)   │
    │ Fixed mean μ (x ≥ 0)                Exponential(λ = 1/μ)     │
    │ Fixed mean μ and variance σ²        Gaussian N(μ, σ²)        │
    │ Fixed mean on [0,1]                 Beta (special cases)     │
    │ Fixed mean count on integers ≥ 0    Poisson(λ)               │
    └──────────────────────────────────────────────────────────────┘

    The Gaussian is not just convenient — it is the uniquely justified
    distribution when only the mean and variance are known.


### MaxEnt and the Softmax Function

    In multiclass classification, the MaxEnt classifier under linear
    feature constraints is exactly logistic regression / softmax:

        P(y = k | x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)

    Softmax emerges naturally from MaxEnt, not just from intuition.
    This is why it is the principled output layer for classifiers.


### MaxEnt and Regularisation

    MaxEnt with a Gaussian prior over parameters ↔ L2 regularisation
    MaxEnt with a Laplace prior               ↔ L1 regularisation

    Regularising is equivalent to adding a prior constraint on the
    model parameters — MaxEnt tells you which prior to use.


### PART 9 — INFORMATION BOTTLENECK

### The Problem of Compression with Relevance

Given input X and label Y, we want to find a compressed representation Z
(e.g. a hidden layer activation) that:
    1. Compresses X as much as possible  →  minimise I(X; Z)
    2. Retains task-relevant information →  maximise I(Z; Y)

These two goals are in fundamental tension: compression destroys information,
but some of that information may be necessary to predict Y.


### The Information Bottleneck Lagrangian

    min  I(X; Z) − β · I(Z; Y)
     Z

    β ≥ 0 controls the compression-relevance trade-off:
        β → 0:  compress maximally (Z carries almost nothing about X or Y)
        β → ∞:  preserve Y-relevant info at all cost (Z ≈ X)

    The "IB curve" traces I(Z; Y) vs I(X; Z) as β varies:

    I(Z;Y)  │              ╭──── optimal frontier
    (task   │           ╭──╯
     info)  │        ╭──╯
            │     ╭──╯
            │ ╭───╯
            └────────────────── I(X;Z) (compression)
                              →  less compression, more X info

    Points ON the curve are Pareto-optimal (you cannot get more task info
    without also keeping more input info).


### Deep Learning and the Information Bottleneck

Tishby & Schwartz-Ziv (2017) proposed that during SGD training, deep networks
pass through two phases:
    1. Error minimisation (fitting):   I(Z; Y) rises rapidly
    2. Compression (forgetting):       I(X; Z) decreases slowly

This "compression phase" may explain why deep networks generalise — they
learn to discard input details irrelevant to the label.

    Note: this interpretation remains debated. The compression phase
    depends on the activation function (clear with tanh, less so with ReLU).


### Connections to Other Concepts

    ┌────────────────────────────────────────────────────────────┐
    │ IB concept               Related ML idea                   │
    ├────────────────────────────────────────────────────────────┤
    │ Minimise I(X; Z)         Dropout, noise injection          │
    │ Maximise I(Z; Y)         Supervised representation loss    │
    │ IB Lagrangian            β-VAE (β controls KL weight)      │
    │ IB curve                 Pareto frontier for compression   │
    │ Task-relevant bits       Feature selection via MI          │
    └────────────────────────────────────────────────────────────┘


### PART 10 — FISHER INFORMATION & DIFFERENTIAL ENTROPY NUANCES

### Fisher Information I(θ)

While Shannon entropy measures the uncertainty of a random variable,
Fisher information measures how much a sample tells us about a parameter θ.

    I(θ) = E_x[(∂/∂θ log p(x|θ))²]         [scalar parameter]
          = −E_x[∂²/∂θ² log p(x|θ)]        [equivalent via integration by parts]

    ∂/∂θ log p(x|θ) is called the SCORE FUNCTION.
    Its expected value is zero: E[score] = 0.

    Intuition: if the log-likelihood is sharply curved around θ̂,
    the data strongly constrains θ → high Fisher information.


### Cramér-Rao Bound

Fisher information sets the fundamental limit on estimation precision:

    Var(θ̂) ≥ 1 / I(θ)     for any unbiased estimator θ̂

    The MLE achieves this bound asymptotically (it is efficient).
    1/I(θ) is the minimum achievable variance — no unbiased estimator
    can do better regardless of its form.

    Example — Gaussian with known σ², estimating μ:
        I(μ) = n/σ²
        Cramér-Rao: Var(μ̂) ≥ σ²/n
        MLE (sample mean): Var(x̄) = σ²/n  ← achieves the bound exactly!


### Fisher Information Matrix (FIM)

For vector parameters θ ∈ ℝᵈ:

    F(θ) = E[(∇_θ log p(x|θ))(∇_θ log p(x|θ))ᵀ]

    Properties:
        F is symmetric and positive semi-definite
        F⁻¹ is the Cramér-Rao lower bound on the covariance of any unbiased estimator


### Natural Gradient

Ordinary gradient descent treats all parameter directions equally.
The natural gradient accounts for the geometry of the parameter space
(the Riemannian manifold of probability distributions):

    θ̃ = F(θ)⁻¹ ∇_θ L(θ)     [natural gradient]

    This is invariant to reparametrisation and converges faster near
    saddle points. It underlies:
        • KFAC (Kronecker-Factored Approximate Curvature)
        • Trust-region policy optimisation (TRPO) in RL
        • Second-order optimisation methods in general


### Differential Entropy — Key Nuances

Differential entropy h(X) = −∫ f(x) log f(x) dx has important differences
from discrete entropy that are easy to miss:

    1. Can be NEGATIVE:
       Uniform(0, 0.1): h = log(0.1) ≈ −2.30 nats (< 0!)
       This is not a bug — differential entropy is not a probability.

    2. NOT invariant to reparametrisation:
       h(aX) = h(X) + log|a|
       Scaling X by 2 adds log(2) ≈ 0.693 nats regardless of distribution.
       Discrete entropy H(X) is invariant to bijections.

    3. Gaussian maximises differential entropy for fixed variance:
       Among all distributions with fixed variance σ²:
           h(X) ≤ ½ log(2πeσ²)   with equality iff X ~ N(μ, σ²)

       This is the theoretical justification for the Gaussian as
       the maximum-entropy noise model — and why MSE loss (Gaussian
       likelihood) is the least assumptive choice for regression.

    4. KL divergence IS invariant to reparametrisation:
       KL(P ‖ Q) is the same regardless of how you parameterise X.
       This is why KL is preferred over differential entropy for
       measuring distributional differences in practice.

    Summary table:
    ┌──────────────────────────┬────────────────┬────────────────────┐
    │ Property                 │ Discrete H(X)  │ Differential h(X)  │
    ├──────────────────────────┼────────────────┼────────────────────┤
    │ Always ≥ 0               │ ✓              │ ✗ (can be < 0)     │
    │ Invariant to bijections  │ ✓              │ ✗                  │
    │ KL divergence consistent │ ✓              │ ✓                  │
    │ Maximum for uniform      │ ✓              │ ✗ (Gaussian max)   │
    └──────────────────────────┴────────────────┴────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Shannon Entropy — Properties, Bounds & Distribution Comparison": {
        "description": (
            "Compute Shannon entropy for every distribution from Module 11. "
            "Visualise the binary entropy H(p) curve. "
            "Show entropy for different distributions and compare to theoretical maximum."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 65)
print("  SHANNON ENTROPY — PROPERTIES AND BOUNDS")
print("=" * 65)
print()

# ── Core entropy functions ─────────────────────────────────────────────────
def entropy_discrete(probs):
    """H(X) = -Σ p·log₂(p) for a discrete distribution."""
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]   # ignore zero probabilities
    return -np.sum(probs * np.log2(probs))

def entropy_nats(probs):
    """H(X) in nats (natural log)."""
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

# ── Part 1: Binary entropy curve ──────────────────────────────────────────
print("  PART 1 — BINARY ENTROPY CURVE  H(p, 1-p)")
print()
p_vals = np.linspace(0.001, 0.999, 1000)
H_binary = -(p_vals * np.log2(p_vals) + (1-p_vals) * np.log2(1-p_vals))

print(f"  {'p':>8} | {'H(p) bits':>12} | {'Interpretation'}")
print(f"  {'─'*48}")
for p_val, label in [(0.01, "almost deterministic"),
                      (0.1,  "mostly tails"),
                      (0.3,  "biased"),
                      (0.5,  "maximum entropy (fair coin)"),
                      (0.7,  "biased other way"),
                      (0.99, "almost deterministic")]:
    h = entropy_discrete([p_val, 1-p_val])
    print(f"  {p_val:8.2f} | {h:12.6f} | {label}")

print()
print(f"  Maximum entropy = log₂(2) = 1.0 bit (at p=0.5)")
print()

# ── Part 2: Entropy of various discrete distributions ─────────────────────
print("  PART 2 — ENTROPY OF COMMON DISTRIBUTIONS")
print()
print(f"  {'Distribution':28s} | {'H (bits)':>10} | {'H_max (bits)':>13} | {'% of max':>10}")
print(f"  {'─'*68}")

dist_examples = [
    ("Bernoulli(p=0.5)",  [0.5, 0.5]),
    ("Bernoulli(p=0.9)",  [0.9, 0.1]),
    ("Bernoulli(p=0.99)", [0.99, 0.01]),
    ("Uniform (k=4)",     [0.25]*4),
    ("Uniform (k=8)",     [0.125]*8),
    ("Skewed (k=4)",      [0.7, 0.15, 0.1, 0.05]),
    ("Peaky (k=4)",       [0.97, 0.01, 0.01, 0.01]),
    ("Uniform (k=256)",   [1/256]*256),
]
for name, probs in dist_examples:
    h     = entropy_discrete(probs)
    k     = len(probs)
    h_max = np.log2(k)
    pct   = 100 * h / h_max if h_max > 0 else 0
    print(f"  {name:28s} | {h:10.4f} | {h_max:13.4f} | {pct:9.1f}%")

print()
print("  KEY FACTS:")
print("  - Uniform distribution achieves maximum entropy log₂(k)")
print("  - Adding more equally-likely outcomes always increases entropy")
print("  - Concentrated distributions have low entropy (high certainty)")
print()

# ── Part 3: Entropy of continuous distributions (differential entropy) ─────
print("  PART 3 — DIFFERENTIAL ENTROPY OF CONTINUOUS DISTRIBUTIONS")
print()
print("  (Differential entropy can be negative — unlike discrete!)")
print()

n_pts = 10000

cont_dists = [
    ("Uniform(0,1)",      np.random.uniform(0, 1, n_pts),   "analytical: log(1)=0 nats"),
    ("Uniform(0,2)",      np.random.uniform(0, 2, n_pts),   "log(2)=0.693 nats"),
    ("Gaussian N(0,1)",   np.random.normal(0, 1, n_pts),    "0.5·log(2πe·1²)=1.419 nats"),
    ("Gaussian N(0,2)",   np.random.normal(0, 2, n_pts),    "0.5·log(2πe·4)=2.112 nats"),
    ("Exponential(λ=1)",  np.random.exponential(1, n_pts),  "1-log(1)=1 nat"),
]

print(f"  {'Distribution':22s} | {'Est H (nats)':>13} | {'Theoretical'}")
print(f"  {'─'*65}")
for name, samples, theoretical in cont_dists:
    # Kernel density entropy estimation
    kde    = stats.gaussian_kde(samples)
    x_eval = np.linspace(samples.min(), samples.max(), 500)
    p_eval = kde(x_eval)
    p_eval = np.maximum(p_eval, 1e-10)
    dx     = x_eval[1] - x_eval[0]
    h_est  = -np.sum(p_eval * np.log(p_eval) * dx)
    print(f"  {name:22s} | {h_est:13.4f} | {theoretical}")

print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Shannon Entropy: Properties and Distributions",
             fontsize=12, fontweight="bold")

# Plot 1: Binary entropy curve
axes[0].plot(p_vals, H_binary, "steelblue", lw=2.5)
axes[0].axhline(1.0, color="gray", linestyle="--", lw=1,
                label="Maximum: 1 bit")
axes[0].axvline(0.5, color="tomato", linestyle="--", lw=1.5, alpha=0.7)
axes[0].fill_between(p_vals, H_binary, alpha=0.1, color="steelblue")
axes[0].set_xlabel("p (probability of outcome 1)")
axes[0].set_ylabel("H(p) in bits")
axes[0].set_title("Binary Entropy H(p, 1-p)")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].annotate("H=1 bit (fair coin)", xy=(0.5, 1.0),
                 xytext=(0.6, 0.85),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)

# Plot 2: Entropy vs number of uniform outcomes
k_vals   = np.arange(2, 50)
h_unif   = np.log2(k_vals)
axes[1].plot(k_vals, h_unif, "seagreen", lw=2.5, label="Uniform dist")
axes[1].set_xlabel("Number of outcomes k")
axes[1].set_ylabel("Max entropy log₂(k) bits")
axes[1].set_title("Maximum Entropy vs Outcome Count")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].fill_between(k_vals, h_unif, alpha=0.15, color="seagreen")

# Plot 3: Entropy of Gaussian as function of σ
sigmas = np.linspace(0.1, 5, 200)
h_gaussian = 0.5 * np.log2(2 * np.pi * np.e * sigmas**2)
axes[2].plot(sigmas, h_gaussian, "tomato", lw=2.5)
axes[2].set_xlabel("Standard deviation σ")
axes[2].set_ylabel("Differential entropy (bits)")
axes[2].set_title("Gaussian Entropy: H(N(0,σ²)) = ½log₂(2πeσ²)")
axes[2].axhline(0, color="gray", linestyle="--", lw=0.8)
axes[2].grid(alpha=0.3)
axes[2].annotate("H can be negative for sigma < 1/sqrt(2pie)",
                 xy=(0.24, -0.5), xytext=(1.0, -0.8), fontsize=8,
                 arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig("entropy_properties.png", dpi=120)
print("  Plot saved → entropy_properties.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Joint & Conditional Entropy — Chain Rule Verified Numerically": {
        "description": (
            "Compute joint entropy H(X,Y), conditional entropies H(X|Y) and H(Y|X), "
            "and verify the chain rule H(X,Y) = H(X) + H(Y|X) numerically. "
            "Show how correlations change conditional entropy."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

print("=" * 65)
print("  JOINT & CONDITIONAL ENTROPY — CHAIN RULE VERIFICATION")
print("=" * 65)
print()

# ── Core entropy functions ─────────────────────────────────────────────────
def h(probs):
    """H(X) in nats."""
    p = np.array(probs, dtype=float).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def joint_entropy(joint_pmf):
    """H(X,Y) from a 2D joint PMF."""
    return h(joint_pmf.ravel())

def marginal_x(joint_pmf):
    return joint_pmf.sum(axis=1)   # sum over Y

def marginal_y(joint_pmf):
    return joint_pmf.sum(axis=0)   # sum over X

def conditional_h_y_given_x(joint_pmf):
    """H(Y|X) = H(X,Y) - H(X)."""
    return joint_entropy(joint_pmf) - h(marginal_x(joint_pmf))

def conditional_h_x_given_y(joint_pmf):
    """H(X|Y) = H(X,Y) - H(Y)."""
    return joint_entropy(joint_pmf) - h(marginal_y(joint_pmf))

def mutual_information(joint_pmf):
    """I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    return (h(marginal_x(joint_pmf))
            + h(marginal_y(joint_pmf))
            - joint_entropy(joint_pmf))

# ── Example 1: Independent X, Y ───────────────────────────────────────────
print("  EXAMPLE 1 — INDEPENDENT X AND Y")
print()
px  = np.array([0.5, 0.3, 0.2])       # marginal P(X)
py  = np.array([0.4, 0.4, 0.2])       # marginal P(Y)
joint_indep = np.outer(px, py)        # P(X,Y) = P(X)·P(Y)  (independence)

hx   = h(px);    hy  = h(py)
hxy  = joint_entropy(joint_indep)
hyx  = conditional_h_y_given_x(joint_indep)
hxy_ = conditional_h_x_given_y(joint_indep)
mi   = mutual_information(joint_indep)

print(f"  Joint PMF (3x3, outer product of marginals):")
print(f"  {joint_indep.round(4)}")
print()
print(f"  H(X)        = {hx:.5f} nats")
print(f"  H(Y)        = {hy:.5f} nats")
print(f"  H(X,Y)      = {hxy:.5f} nats")
print()
print(f"  Chain rule: H(X,Y) = H(X) + H(Y|X)")
print(f"              {hxy:.5f} = {hx:.5f} + {hyx:.5f}")
chain1_ok = abs(hxy - (hx + hyx)) < 1e-10
print(f"              Match: {'✓' if chain1_ok else '✗'}")
print()
print(f"  H(Y|X)      = {hyx:.5f} nats")
print(f"  H(Y) - H(Y|X) = {hy:.5f} - {hyx:.5f} = {hy-hyx:.5f}")
print(f"  I(X;Y)      = {mi:.5f} nats  (≈0 since independent!)")
print()

# ── Example 2: Correlated X, Y ────────────────────────────────────────────
print("  EXAMPLE 2 — CORRELATED X AND Y")
print()
# Y tends to equal X (diagonal heavy)
joint_corr = np.array([
    [0.30, 0.05, 0.05],
    [0.05, 0.25, 0.05],
    [0.05, 0.05, 0.15],
])
joint_corr /= joint_corr.sum()   # normalise

hx2  = h(marginal_x(joint_corr))
hy2  = h(marginal_y(joint_corr))
hxy2 = joint_entropy(joint_corr)
hyx2 = conditional_h_y_given_x(joint_corr)
hxy2_= conditional_h_x_given_y(joint_corr)
mi2  = mutual_information(joint_corr)

print(f"  Joint PMF (diagonal heavy = correlated):")
print(f"  {joint_corr.round(4)}")
print()
print(f"  H(X)        = {hx2:.5f} nats")
print(f"  H(Y)        = {hy2:.5f} nats")
print(f"  H(X,Y)      = {hxy2:.5f} nats")
print(f"  H(Y|X)      = {hyx2:.5f} nats   (LOWER than H(Y)={hy2:.4f})")
print(f"  I(X;Y)      = {mi2:.5f} nats   (>0 — they share information)")
print()

# ── Example 3: Perfect dependence ─────────────────────────────────────────
print("  EXAMPLE 3 — PERFECT DEPENDENCE (Y = X)")
print()
joint_perfect = np.diag([0.5, 0.3, 0.2])

hx3  = h(marginal_x(joint_perfect))
hy3  = h(marginal_y(joint_perfect))
hxy3 = joint_entropy(joint_perfect)
hyx3 = conditional_h_y_given_x(joint_perfect)
mi3  = mutual_information(joint_perfect)

print(f"  H(X)        = {hx3:.5f} nats")
print(f"  H(Y|X)      = {hyx3:.5f} nats   (≈0 — knowing X tells everything about Y)")
print(f"  I(X;Y)      = {mi3:.5f} nats   (= H(X) = H(Y))")
print()

# ── Summary comparison ─────────────────────────────────────────────────────
print("  SUMMARY COMPARISON:")
print(f"  {'Scenario':20s} | {'H(X)':>8} | {'H(Y)':>8} | {'H(Y|X)':>8} | "
      f"{'I(X;Y)':>8} | {'Conditioning'}")
print(f"  {'─'*78}")
for scenario, hx_v, hy_v, hyx_v, mi_v in [
        ("Independent",    hx, hy, hyx,  mi),
        ("Correlated",     hx2, hy2, hyx2, mi2),
        ("Perfect Y=X",    hx3, hy3, hyx3, mi3)]:
    reduces = "reduces H" if hyx_v < hy_v - 0.001 else "no effect (indep)"
    print(f"  {scenario:20s} | {hx_v:8.4f} | {hy_v:8.4f} | {hyx_v:8.4f} | "
          f"{mi_v:8.4f} | {reduces}")

print()
print("  KEY INSIGHT:")
print("  H(Y|X) <= H(Y) always — conditioning never increases entropy")
print("  The reduction H(Y) - H(Y|X) = I(X;Y) is the mutual information")

# ── Visualise ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Joint Entropy and Mutual Information",
             fontsize=12, fontweight="bold")

scenarios = [
    ("Independent (MI~0)", joint_indep, mi),
    ("Correlated (MI>0)",  joint_corr,  mi2),
    ("Perfect (MI=H(X))",  joint_perfect, mi3),
]
for ax, (title, jpdf, mi_val) in zip(axes, scenarios):
    im = ax.imshow(jpdf, cmap="Blues", vmin=0, vmax=0.35)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(title + " I(X;Y)=" + f"{mi_val:.4f} nats", fontsize=10)
    ax.set_xlabel("Y"); ax.set_ylabel("X")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{jpdf[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if jpdf[i,j] > 0.15 else "black")

plt.tight_layout()
plt.savefig("joint_conditional_entropy.png", dpi=120)
print()
print("  Plot saved → joint_conditional_entropy.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Mutual Information vs Pearson Correlation as Feature Selector": {
        "description": (
            "Compare MI-based feature selection against Pearson correlation "
            "on datasets with linear AND non-linear relationships. "
            "Show where correlation fails and MI succeeds."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# sklearn replaced with pure NumPy equivalents

def _mi_from_bins(x_disc, y_disc):
    """MI from integer-coded variables via joint histogram."""
    n = len(x_disc)
    mi = 0.0
    for xv in np.unique(x_disc):
        for yv in np.unique(y_disc):
            n_xy = np.sum((x_disc == xv) & (y_disc == yv))
            if n_xy == 0:
                continue
            n_x = np.sum(x_disc == xv)
            n_y = np.sum(y_disc == yv)
            mi += (n_xy / n) * np.log((n_xy * n) / (n_x * n_y))
    return max(mi, 0.0)

def _discretise(arr, n_bins=12):
    pcts = np.unique(np.percentile(arr, np.linspace(0, 100, n_bins + 1)))
    return np.digitize(arr, pcts[1:-1])

def mutual_info_regression(X, y, random_state=None, n_bins=12):
    y_d = _discretise(y, n_bins)
    return np.array([_mi_from_bins(_discretise(X[:, i], n_bins), y_d)
                     for i in range(X.shape[1])])

def mutual_info_classif(X, y, random_state=None, n_bins=12):
    y_int = y.astype(int)
    return np.array([_mi_from_bins(_discretise(X[:, i], n_bins), y_int)
                     for i in range(X.shape[1])])

def make_classification(n_samples=500, n_features=10, n_informative=5,
                        n_redundant=0, n_repeated=0, random_state=None):
    rng = np.random.default_rng(random_state)
    y   = rng.integers(0, 2, n_samples)
    X   = rng.standard_normal((n_samples, n_features))
    shift = rng.uniform(0.8, 1.5, n_informative)
    X[:, :n_informative] += (2 * y - 1)[:, None] * shift
    return X, y

class StandardScaler:
    def fit_transform(self, X):
        mu  = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        return (X - mu) / std

np.random.seed(42)

print("=" * 65)
print("  MUTUAL INFORMATION vs PEARSON CORRELATION AS FEATURE SELECTOR")
print("=" * 65)
print()

n = 2000
x = np.random.uniform(-3, 3, n)

# Create features with different types of relationships to target y
np.random.seed(0)
noise = lambda: np.random.normal(0, 0.3, n)

f1 = x + noise()                          # Linear         — corr & MI both detect
f2 = x**2 + noise()                       # Quadratic      — MI detects, corr misses
f3 = np.sin(2*x) + noise()                # Sinusoidal     — MI detects, corr may miss
f4 = np.abs(x) + noise()                  # Abs value      — MI detects, corr misses
f5 = np.random.normal(0, 1, n)            # Pure noise     — neither detects
f6 = x * (x > 0) + noise()               # Half-linear    — both partial

target = x + noise()    # Target is linear in x

X_feats = np.column_stack([f1, f2, f3, f4, f5, f6])
X_std   = StandardScaler().fit_transform(X_feats)
feat_names = ["Linear", "Quadratic", "Sinusoidal", "Abs value", "Pure noise", "Half-linear"]

# ── Pearson correlation ────────────────────────────────────────────────────
pearson_corr = np.array([np.corrcoef(X_feats[:, i], target)[0, 1]
                         for i in range(X_feats.shape[1])])

# ── Mutual information (regression) ───────────────────────────────────────
mi_scores = mutual_info_regression(X_std, target, random_state=0)

print(f"  {'Feature':14s} | {'|Pearson ρ|':>12} | {'MI score':>10} | {'Correlation rank':>17} | {'MI rank':>8}")
print(f"  {'─'*72}")

pearson_ranks = np.argsort(np.abs(pearson_corr))[::-1] + 1
mi_ranks      = np.argsort(mi_scores)[::-1] + 1

for i, name in enumerate(feat_names):
    disc = ""
    if abs(pearson_corr[i]) < 0.1 and mi_scores[i] > 0.1:
        disc = " ← MI wins!"
    print(f"  {name:14s} | {abs(pearson_corr[i]):12.4f} | {mi_scores[i]:10.4f} | "
          f"{pearson_ranks[i]:17d} | {mi_ranks[i]:8d}{disc}")

print()
print("  KEY FINDINGS:")
print("  - Quadratic, Sinusoidal, Abs value: ρ≈0 but MI>0")
print("    → Pearson correlation MISSES these, MI correctly detects them")
print("  - Linear: both detect it (ρ is large, MI is large)")
print("  - Pure noise: both correctly assign low scores")
print()

# ── Visual comparison ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Mutual Information vs Pearson Correlation: When MI Wins",
             fontsize=12, fontweight="bold")

colours_scatter = ["steelblue", "tomato", "seagreen", "purple", "gray", "orange"]

for i, (name, colour) in enumerate(zip(feat_names, colours_scatter)):
    ax = axes[i // 3, i % 3]
    ax.scatter(X_feats[:, i], target, alpha=0.1, s=6, color=colour)
    ax.set_xlabel(f"Feature: {name}")
    ax.set_ylabel("Target")

    rho_val = pearson_corr[i]
    mi_val  = mi_scores[i]
    title = name + "  |rho|=" + f"{abs(rho_val):.3f}" + "  MI=" + f"{mi_val:.3f}"
    if abs(rho_val) < 0.1 and mi_val > 0.1:
        title += "  ← MI wins!"
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mi_vs_correlation.png", dpi=110)
print("  Plot saved → mi_vs_correlation.png")
print()

# ── Feature selection comparison on classification ─────────────────────────
print("  FEATURE SELECTION ON A CLASSIFICATION TASK (10 features, 5 noisy):")
print()

X_clf, y_clf = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=0, n_repeated=0, random_state=0)

mi_clf    = mutual_info_classif(X_clf, y_clf, random_state=0)
pearson_clf = np.abs([np.corrcoef(X_clf[:, i], y_clf)[0, 1]
                       for i in range(10)])

print(f"  {'Feature':10s} | {'MI score':>10} | {'|Pearson|':>10} | {'MI rank':>8} | {'Corr rank':>10}")
print(f"  {'─'*57}")
mi_r   = np.argsort(mi_clf)[::-1]    + 1
corr_r = np.argsort(pearson_clf)[::-1] + 1
for i in range(10):
    print(f"  feat_{i:4d}   | {mi_clf[i]:10.4f} | {pearson_clf[i]:10.4f} | "
          f"{mi_r[i]:8d} | {corr_r[i]:10d}")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Information Gain for Decision Trees — From Scratch": {
        "description": (
            "Implement entropy-based Information Gain from scratch. "
            "Show how a decision tree picks splits greedily. "
            "Compare IG vs Gini impurity on the same dataset."
        ),
        "language": "python",
        "code": '''
import numpy as np
from collections import Counter

print("=" * 65)
print("  INFORMATION GAIN FOR DECISION TREES — FROM SCRATCH")
print("=" * 65)
print()

# ── Core functions ─────────────────────────────────────────────────────────
def entropy(labels):
    """H(Y) for a list of class labels."""
    if len(labels) == 0:
        return 0.0
    counts = Counter(labels)
    n      = len(labels)
    probs  = np.array(list(counts.values())) / n
    probs  = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def gini(labels):
    """Gini impurity = 1 - Σ pₖ² ."""
    if len(labels) == 0:
        return 0.0
    counts = Counter(labels)
    n      = len(labels)
    return 1.0 - sum((c/n)**2 for c in counts.values())

def information_gain(parent_labels, split_groups):
    """
    IG(Y; split) = H(parent) - weighted_avg_H(children)
    split_groups: list of label arrays for each child node
    """
    h_parent = entropy(parent_labels)
    n_total  = len(parent_labels)
    weighted_h = sum(
        (len(grp) / n_total) * entropy(grp)
        for grp in split_groups if len(grp) > 0
    )
    return h_parent - weighted_h

def best_split(X, y, feature_idx):
    """Find the best threshold for a single feature (exhaustive search)."""
    values      = np.unique(X[:, feature_idx])
    thresholds  = (values[:-1] + values[1:]) / 2   # midpoints
    best_ig, best_thresh = -np.inf, None

    for thresh in thresholds:
        left  = y[X[:, feature_idx] <= thresh]
        right = y[X[:, feature_idx] >  thresh]
        ig    = information_gain(y, [left, right])
        if ig > best_ig:
            best_ig, best_thresh = ig, thresh

    return best_ig, best_thresh

# ── Demo dataset: iris-like (simplified 2-class, 3 features) ──────────────
np.random.seed(0)
n_per_class = 50

# Class 0: small sepal, small petal
X0 = np.column_stack([
    np.random.normal(4.9, 0.4, n_per_class),   # feature 0
    np.random.normal(3.1, 0.4, n_per_class),   # feature 1
    np.random.normal(1.5, 0.3, n_per_class),   # feature 2 (most discriminative)
])
# Class 1: larger sepal, larger petal
X1 = np.column_stack([
    np.random.normal(6.5, 0.5, n_per_class),
    np.random.normal(3.0, 0.4, n_per_class),
    np.random.normal(5.0, 0.5, n_per_class),   # feature 2 strongly different
])

X_tree = np.vstack([X0, X1])
y_tree = np.array([0]*n_per_class + [1]*n_per_class)
feature_names = ["Sepal length", "Sepal width", "Petal length"]

print(f"  Dataset: {len(y_tree)} samples  |  2 classes  |  {X_tree.shape[1]} features")
print(f"  Class 0: {(y_tree==0).sum()}  Class 1: {(y_tree==1).sum()}")
print()

# ── Root node entropy ──────────────────────────────────────────────────────
h_root  = entropy(y_tree)
g_root  = gini(y_tree)
print(f"  ROOT NODE:")
print(f"    Entropy H(Y)  = {h_root:.4f} bits  "
      f"(max is log₂(2)=1.0 for balanced binary)")
print(f"    Gini impurity = {g_root:.4f}       "
      f"(max is 0.5 for balanced binary)")
print()

# ── Find best split for each feature ──────────────────────────────────────
print("  BEST SPLIT PER FEATURE:")
print(f"  {'Feature':14s} | {'Best threshold':>16} | {'Info Gain':>11} | {'Rank':>6}")
print(f"  {'─'*55}")

ig_results = []
for fi, fname in enumerate(feature_names):
    ig, thresh = best_split(X_tree, y_tree, fi)
    ig_results.append((ig, thresh, fname))

ig_results_sorted = sorted(ig_results, reverse=True)
for rank, (ig_val, thresh_val, fname) in enumerate(ig_results_sorted, 1):
    flag = " ← CHOSEN (highest IG)" if rank == 1 else ""
    print(f"  {fname:14s} | {thresh_val:16.4f} | {ig_val:11.4f} | {rank:6d}{flag}")

print()

# ── Simulate the first split ──────────────────────────────────────────────
best_fi   = feature_names.index(ig_results_sorted[0][2])
best_thr  = ig_results_sorted[0][1]
best_ig   = ig_results_sorted[0][0]

left_mask  = X_tree[:, best_fi] <= best_thr
right_mask = ~left_mask
y_left  = y_tree[left_mask]
y_right = y_tree[right_mask]

print(f"  SPLIT ON '{feature_names[best_fi]}' <= {best_thr:.4f}:")
print(f"  {'':4s} Left node:   {left_mask.sum():3d} samples  "
      f"[class 0: {(y_left==0).sum()}, class 1: {(y_left==1).sum()}]  "
      f"H={entropy(y_left):.4f} bits")
print(f"  {'':4s} Right node:  {right_mask.sum():3d} samples  "
      f"[class 0: {(y_right==0).sum()}, class 1: {(y_right==1).sum()}]  "
      f"H={entropy(y_right):.4f} bits")
print()
print(f"  IG = H(root) - [({left_mask.sum()}/{len(y_tree)})·H(left) "
      f"+ ({right_mask.sum()}/{len(y_tree)})·H(right)]")
print(f"     = {h_root:.4f} - [{left_mask.sum()/len(y_tree):.3f}·"
      f"{entropy(y_left):.4f} + {right_mask.sum()/len(y_tree):.3f}·"
      f"{entropy(y_right):.4f}]")
print(f"     = {best_ig:.4f} bits  (large IG = good split!)")
print()

# ── Compare IG vs Gini side-by-side ───────────────────────────────────────
print("  IG vs GINI IMPURITY COMPARISON:")
print(f"  {'p (fraction class 1)':22s} | {'H(p) bits':>12} | {'Gini':>8} | Similar?")
print(f"  {'─'*55}")
for p_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    labels_test = [1]*int(p_val*10) + [0]*(10 - int(p_val*10))
    h_val = entropy(labels_test)
    g_val = gini(labels_test)
    sim = "yes" if abs(h_val/1.0 - g_val/0.5) < 0.05 else "slight diff"
    print(f"  {p_val:22.1f} | {h_val:12.4f} | {g_val:8.4f} | {sim}")
print()
print("  Entropy and Gini give nearly identical rankings.")
print("  Entropy: stronger information-theoretic justification.")
print("  Gini:    faster (no log), default in sklearn's DecisionTreeClassifier.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · KL Divergence — Asymmetry, Zero-Forcing & Mass-Covering": {
        "description": (
            "Demonstrate KL divergence asymmetry in detail. "
            "Show zero-forcing (reverse KL) vs mass-covering (forward KL) "
            "on a bimodal distribution. Compare KL vs Jensen-Shannon divergence."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize

np.random.seed(0)

print("=" * 65)
print("  KL DIVERGENCE — ASYMMETRY & ZERO-FORCING vs MASS-COVERING")
print("=" * 65)
print()

# ── Core functions ─────────────────────────────────────────────────────────
def kl_divergence(p, q, eps=1e-12):
    """KL(P || Q) = Σ p·log(p/q). Expects normalised arrays."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    mask = p > eps
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

def js_divergence(p, q):
    """Jensen-Shannon: symmetric, bounded in [0,1]."""
    p = np.array(p, dtype=float); p /= p.sum()
    q = np.array(q, dtype=float); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# ── Part 1: KL asymmetry on simple discrete distributions ─────────────────
print("  PART 1 — KL ASYMMETRY ON DISCRETE DISTRIBUTIONS")
print()

pairs = [
    ("Nearly identical",     [0.5, 0.3, 0.2], [0.45, 0.35, 0.2]),
    ("Different shapes",     [0.7, 0.2, 0.1], [0.2, 0.5, 0.3]),
    ("P has extra mass",     [0.5, 0.5, 0.0], [0.33, 0.33, 0.34]),
    ("Q has zero mass!",     [0.33, 0.33, 0.34], [0.5, 0.5, 0.0]),
]

print(f"  {'Scenario':20s} | {'KL(P||Q)':>10} | {'KL(Q||P)':>10} | "
      f"{'JSD':>8} | {'Note'}")
print(f"  {'─'*80}")
for name, p_dist, q_dist in pairs:
    kl_pq = kl_divergence(p_dist, q_dist)
    kl_qp = kl_divergence(q_dist, p_dist)
    jsd   = js_divergence(p_dist, q_dist)
    note  = "∞ if P has mass where Q=0!" if kl_pq > 100 else ""
    if kl_pq > 100:
        kl_pq_str = "∞"
    else:
        kl_pq_str = f"{kl_pq:10.4f}"
    print(f"  {name:20s} | {kl_pq_str:>10} | {kl_qp:10.4f} | {jsd:8.4f} | {note}")

print()
print("  KL(P||Q) = ∞ whenever Q assigns zero probability to a region where P>0.")
print("  This is the 'zero-avoidance' property of KL divergence.")
print("  JSD is always finite and bounded in [0, ln2].")
print()

# ── Part 2: Forward vs Reverse KL on bimodal distribution ─────────────────
print("  PART 2 — FORWARD vs REVERSE KL ON BIMODAL DISTRIBUTION")
print()
print("  P: bimodal Gaussian (two modes at -2 and +2)")
print("  Q: unimodal Gaussian N(μ, σ²) — we find best μ, σ to approximate P")
print()

x = np.linspace(-6, 6, 1000)
dx = x[1] - x[0]

# True P: bimodal mixture
p_bimodal = (0.5 * stats.norm.pdf(x, -2, 0.8) +
             0.5 * stats.norm.pdf(x,  2, 0.8))
p_bimodal /= p_bimodal.sum() * dx

def forward_kl_loss(params):
    """Minimise KL(P || Q) — forward KL, mean-seeking."""
    mu, log_sig = params
    sig = np.exp(log_sig)
    if sig < 0.1 or sig > 10:
        return 1e10
    q = stats.norm.pdf(x, mu, sig)
    q = np.maximum(q, 1e-12)
    q /= q.sum() * dx
    kl = np.sum(p_bimodal * np.log(np.maximum(p_bimodal, 1e-12) / q) * dx)
    return float(kl)

def reverse_kl_loss(params):
    """Minimise KL(Q || P) — reverse KL, mode-seeking."""
    mu, log_sig = params
    sig = np.exp(log_sig)
    if sig < 0.1 or sig > 10:
        return 1e10
    q = stats.norm.pdf(x, mu, sig)
    q = np.maximum(q, 1e-12)
    q /= q.sum() * dx
    p = np.maximum(p_bimodal, 1e-12)
    kl = np.sum(q * np.log(q / p) * dx)
    return float(kl)

# Forward KL: minimise KL(P||Q)
res_fwd = optimize.minimize(forward_kl_loss, [0.0, 0.0], method="Nelder-Mead")
mu_fwd, sig_fwd = res_fwd.x[0], np.exp(res_fwd.x[1])

# Reverse KL: minimise KL(Q||P) — two modes, try both starting points
best_rev, best_mu_rev, best_sig_rev = np.inf, None, None
for mu_init in [-2.0, 0.0, 2.0]:
    res_rev = optimize.minimize(reverse_kl_loss, [mu_init, 0.0], method="Nelder-Mead")
    if res_rev.fun < best_rev:
        best_rev     = res_rev.fun
        best_mu_rev  = res_rev.x[0]
        best_sig_rev = np.exp(res_rev.x[1])

q_fwd = stats.norm.pdf(x, mu_fwd, sig_fwd)
q_rev = stats.norm.pdf(x, best_mu_rev, best_sig_rev)

print(f"  FORWARD KL (mean-seeking):")
print(f"    Best Q: N(μ={mu_fwd:.3f}, σ={sig_fwd:.3f})")
print(f"    Mean of Q sits BETWEEN the two modes — covers both (blurry)")
print(f"    KL(P||Q) = {forward_kl_loss([mu_fwd, np.log(sig_fwd)]):.4f}")
print()
print(f"  REVERSE KL (mode-seeking):")
print(f"    Best Q: N(μ={best_mu_rev:.3f}, σ={best_sig_rev:.3f})")
print(f"    Q snaps to ONE mode and IGNORES the other")
print(f"    KL(Q||P) = {reverse_kl_loss([best_mu_rev, np.log(best_sig_rev)]):.4f}")
print()
print("  Why this matters in ML:")
print("  - Variational inference uses REVERSE KL → mode-seeking → misses modes")
print("    → VAE approximate posteriors tend to collapse to one mode")
print("  - GAN training uses a divergence closer to FORWARD KL → mode-covering")
print("    → but can suffer from mode collapse with unstable training")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("KL Divergence: Asymmetry and Forward vs Reverse",
             fontsize=12, fontweight="bold")

# Left: forward KL
axes[0].fill_between(x, p_bimodal, alpha=0.3, color="gray", label="P (bimodal)")
axes[0].plot(x, p_bimodal, "k-", lw=2, label="P (bimodal)")
axes[0].plot(x, q_fwd/q_fwd.sum()/dx, "tomato", lw=2.5, linestyle="--",
             label=f"Q* fwd N({mu_fwd:.2f},{sig_fwd:.2f})")
axes[0].set_title("Forward KL(P||Q) — Q covers BOTH modes (spreads)", fontsize=10)
axes[0].legend(fontsize=8); axes[0].set_xlabel("x"); axes[0].grid(alpha=0.3)
axes[0].set_ylim(0)

# Middle: reverse KL
axes[1].fill_between(x, p_bimodal, alpha=0.3, color="gray", label="P (bimodal)")
axes[1].plot(x, p_bimodal, "k-", lw=2, label="P (bimodal)")
axes[1].plot(x, q_rev/q_rev.sum()/dx, "steelblue", lw=2.5, linestyle="--",
             label=f"Q* rev N({best_mu_rev:.2f},{best_sig_rev:.2f})")
axes[1].set_title("Reverse KL(Q||P) — Q snaps to ONE mode (ignores other)", fontsize=10)
axes[1].legend(fontsize=8); axes[1].set_xlabel("x"); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0)

# Right: KL asymmetry demo — vary P
p_grid  = np.linspace(0.01, 0.99, 300)
kl_pq_g = np.array([kl_divergence([p_val, 1-p_val], [0.5, 0.5]) for p_val in p_grid])
kl_qp_g = np.array([kl_divergence([0.5, 0.5], [p_val, 1-p_val]) for p_val in p_grid])
jsd_g   = np.array([js_divergence([p_val, 1-p_val], [0.5, 0.5]) for p_val in p_grid])

axes[2].plot(p_grid, kl_pq_g, "tomato",    lw=2.5, label="KL(P||Q)")
axes[2].plot(p_grid, kl_qp_g, "steelblue", lw=2.5, label="KL(Q||P)")
axes[2].plot(p_grid, jsd_g,   "seagreen",  lw=2.5, label="JSD (symmetric)")
axes[2].set_xlabel("p (P = Bernoulli(p), Q = Bernoulli(0.5))")
axes[2].set_ylabel("Divergence (nats)")
axes[2].set_title("KL Asymmetry vs JSD — Q = Bernoulli(0.5) fixed", fontsize=10)
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
axes[2].set_ylim(0)

plt.tight_layout()
plt.savefig("kl_divergence_deep.png", dpi=120)
print()
print("  Plot saved → kl_divergence_deep.png")
''',
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "6 · ELBO Decomposition — VAE Objective from First Principles": {
        "description": (
            "Derive and numerically verify the Evidence Lower BOund (ELBO). "
            "Show ELBO = reconstruction - KL(q||p). "
            "Visualise how each term evolves during training of a toy VAE."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

print("=" * 65)
print("  ELBO DECOMPOSITION — VAE OBJECTIVE FROM FIRST PRINCIPLES")
print("=" * 65)
print()
print("  DERIVATION:")
print()
print("  Goal: model P(x) by marginalising over latent variable z:")
print("        P(x) = ∫ P(x|z) P(z) dz     (intractable integral)")
print()
print("  Introduce approximate posterior q(z|x) and write:")
print()
print("  log P(x) = log ∫ P(x|z) P(z) dz")
print("           = log E_q[P(x|z) P(z) / q(z|x)]   (multiply/divide by q)")
print("           ≥ E_q[log(P(x|z) P(z) / q(z|x))]  (Jensen's inequality)")
print("           = E_q[log P(x|z)] - KL(q(z|x) || P(z))")
print()
print("  ELBO = E_q[log P(x|z)] - KL(q(z|x) || P(z))")
print("       = Reconstruction term - KL regularisation")
print("          [how well z decodes to x]  [how close q is to prior p(z)=N(0,I)]")
print()
print("  log P(x) ≥ ELBO  (ELBO is a LOWER BOUND on log-evidence)")
print("  Maximising ELBO simultaneously:")
print("    1. Improves reconstruction quality")
print("    2. Keeps posterior close to the prior")
print()

# ── Numerical verification of ELBO ────────────────────────────────────────
print("  NUMERICAL VERIFICATION:")
print()

# Simple 1D Gaussian VAE setup
# P(z) = N(0,1)  prior
# q(z|x) = N(mu_q, sig_q²)  approximate posterior (output of encoder)
# P(x|z) = N(z, sig_x²)  decoder (simple identity mapping for illustration)

def kl_gaussian(mu_q, sig_q, mu_p=0.0, sig_p=1.0):
    """KL(N(mu_q,sig_q²) || N(mu_p,sig_p²)) — closed form."""
    return (np.log(sig_p/sig_q)
            + (sig_q**2 + (mu_q - mu_p)**2) / (2*sig_p**2)
            - 0.5)

def reconstruction_loss(x, mu_q, sig_q, n_mc=5000):
    """E_q[log P(x|z)] where P(x|z)=N(z, 0.1²), estimated by MC sampling."""
    z_samples    = np.random.normal(mu_q, sig_q, n_mc)
    log_p_x_z    = -0.5 * ((x - z_samples) / 0.1)**2 - np.log(0.1) - 0.5*np.log(2*np.pi)
    return log_p_x_z.mean()

# Ground truth log P(x) via importance sampling
def log_p_x_true(x, n_mc=20000):
    """Estimate log P(x) = log ∫ P(x|z)P(z)dz via Monte Carlo."""
    z_prior = np.random.normal(0, 1, n_mc)
    log_p_x_z = -0.5*((x - z_prior)/0.1)**2 - np.log(0.1) - 0.5*np.log(2*np.pi)
    log_w      = log_p_x_z   # importance weights (prior is proposal)
    log_p      = np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max()
    return float(log_p)

x_obs = 1.5   # an observed data point

print(f"  x = {x_obs}  (observed data point)")
print(f"  Prior:   P(z) = N(0, 1)")
print(f"  Decoder: P(x|z) = N(z, 0.1²)")
print()
print(f"  True log P(x) ≈ {log_p_x_true(x_obs):.4f} nats  (Monte Carlo estimate)")
print()
print(f"  {'q(z|x) params':20s} | {'Recon term':>12} | {'KL term':>10} | "
      f"{'ELBO':>10} | {'gap to logP(x)':>14}")
print(f"  {'─'*74}")

q_configs = [
    (0.0, 1.0,  "Prior (no encoding)"),
    (0.5, 0.8,  "Rough posterior"),
    (1.0, 0.5,  "Better posterior"),
    (1.5, 0.1,  "Sharp posterior"),
    (x_obs, 0.1, "Near-true posterior"),
]

true_log_p = log_p_x_true(x_obs)
for mu_q, sig_q, label in q_configs:
    recon = reconstruction_loss(x_obs, mu_q, sig_q)
    kl    = kl_gaussian(mu_q, sig_q)
    elbo  = recon - kl
    gap   = true_log_p - elbo   # should be >= 0
    print(f"  N({mu_q:.1f},{sig_q:.1f}) [{label:20s}] | "
          f"{recon:12.4f} | {kl:10.4f} | {elbo:10.4f} | {gap:14.4f}")

print()
print("  ELBO ≤ log P(x) always (gap = KL(q||P(z|x)) ≥ 0)")
print("  Perfect q → gap=0 → ELBO = log P(x) (intractable truth recovered)")
print()

# ── Simulate training progression ─────────────────────────────────────────
print("  TRAINING DYNAMICS SIMULATION:")
print()

# Simulate gradient ascent on ELBO w.r.t. mu_q, sig_q
mu_q_init, sig_q_init = 0.0, 1.0   # start at prior
lr_sim  = 0.05
n_steps = 80
mu_q_t, sig_q_t = mu_q_init, sig_q_init

recons_sim, kls_sim, elbos_sim = [], [], []

for step in range(n_steps):
    recon_v = reconstruction_loss(x_obs, mu_q_t, sig_q_t, n_mc=500)
    kl_v    = kl_gaussian(mu_q_t, sig_q_t)
    elbo_v  = recon_v - kl_v
    recons_sim.append(recon_v)
    kls_sim.append(kl_v)
    elbos_sim.append(elbo_v)

    # Analytical gradient of ELBO w.r.t. mu_q:
    # d(Recon)/d(mu_q) ≈ (x - mu_q) / 0.1²
    # d(KL)/d(mu_q)    = mu_q / sig_p²
    d_elbo_mu  =  (x_obs - mu_q_t) / 0.01 - mu_q_t
    d_elbo_sig = (-1/sig_q_t + sig_q_t/1.0) * (-1)   # approx
    mu_q_t  += lr_sim * d_elbo_mu * 0.01  # small step
    sig_q_t  = max(0.05, sig_q_t - lr_sim * 0.02)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("VAE ELBO: Reconstruction vs KL Trade-off During Training",
             fontsize=12, fontweight="bold")

steps = np.arange(1, n_steps+1)
axes[0].plot(steps, elbos_sim, "steelblue", lw=2.5, label="ELBO")
axes[0].axhline(true_log_p, color="green", linestyle="--", lw=2,
                label=f"True log P(x)={true_log_p:.2f}")
axes[0].fill_between(steps, elbos_sim, true_log_p,
                     alpha=0.2, color="tomato", label="Gap = KL(q||P(z|x))")
axes[0].set_xlabel("Training step"); axes[0].set_ylabel("Nats")
axes[0].set_title("ELBO converges to log P(x)")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].plot(steps, recons_sim, "seagreen", lw=2.5, label="Reconstruction")
axes[1].plot(steps, [-k for k in kls_sim], "tomato", lw=2.5,
             label="-KL (regularisation)")
axes[1].set_xlabel("Training step"); axes[1].set_ylabel("Nats")
axes[1].set_title("Reconstruction vs KL Trade-off")
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

# Bar chart: final decomposition
final_recon = recons_sim[-1]
final_kl    = kls_sim[-1]
final_elbo  = elbos_sim[-1]
bars = axes[2].bar(["Reconstruction", "-KL", "ELBO"],
                   [final_recon, -final_kl, final_elbo],
                   color=["seagreen", "tomato", "steelblue"], alpha=0.8)
axes[2].axhline(true_log_p, color="green", linestyle="--", lw=2,
                label=f"True log P(x)={true_log_p:.2f}")
axes[2].set_ylabel("Nats"); axes[2].legend(fontsize=9)
axes[2].set_title("Final ELBO Decomposition")
for bar in bars:
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", fontsize=9)
axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("elbo_decomposition.png", dpi=120)
print("  Plot saved → elbo_decomposition.png")
print()
print("  ELBO = Reconstruction − KL = log P(x) − KL(q(z|x) || P(z|x))")
print("  The 'beta-VAE' multiplies the KL term by β>1 to force")
print("  stronger disentanglement at the cost of reconstruction quality.")
''',
    },
    # ── 7 ─────────────────────────────────────────────────────────────────────
    "7 · Huffman Coding & Source Compression — Shannon's First Theorem": {
        "description": (
            "Build a Huffman tree from scratch and verify that average code length "
            "approaches H(X). Sweep distributions from uniform to peaky and show "
            "the compression gap. Compute perplexity and connect it to LM training."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from heapq import heappush, heappop

print("=" * 65)
print("  HUFFMAN CODING & SOURCE COMPRESSION")
print("=" * 65)
print()

# ── Huffman tree implementation ───────────────────────────────────────────
class HuffmanNode:
    def __init__(self, symbol, prob, left=None, right=None):
        self.symbol = symbol
        self.prob   = prob
        self.left   = left
        self.right  = right
    def __lt__(self, other):
        return self.prob < other.prob

def build_huffman_tree(symbols, probs):
    heap = []
    for s, p in zip(symbols, probs):
        heappush(heap, HuffmanNode(s, p))
    while len(heap) > 1:
        left  = heappop(heap)
        right = heappop(heap)
        parent = HuffmanNode(None, left.prob + right.prob, left, right)
        heappush(heap, parent)
    return heappop(heap)

def get_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node.symbol is not None:
        codes[node.symbol] = prefix if prefix else "0"
    else:
        get_codes(node.left,  prefix + "0", codes)
        get_codes(node.right, prefix + "1", codes)
    return codes

def entropy_bits(probs):
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

# ── Part 1: Classic 4-symbol example ──────────────────────────────────────
print("  PART 1 — CLASSIC 4-SYMBOL EXAMPLE")
print()

symbols_4 = ["A", "B", "C", "D"]
probs_4   = [0.50, 0.25, 0.125, 0.125]

tree_4  = build_huffman_tree(symbols_4, probs_4)
codes_4 = get_codes(tree_4)
H_4     = entropy_bits(probs_4)

print(f"  {'Symbol':8s} | {'P(x)':>8} | {'Code':>10} | {'Length':>8} | {'P×len':>8}")
print(f"  {'─'*52}")
avg_len_4 = 0.0
for sym, prob in zip(symbols_4, probs_4):
    code = codes_4[sym]
    l    = len(code)
    avg_len_4 += prob * l
    print(f"  {sym:8s} | {prob:8.4f} | {code:>10} | {l:8d} | {prob*l:8.4f}")
print(f"  {'─'*52}")
print(f"  {'Average':8s} | {'':8s} | {'':10} | {avg_len_4:8.4f} | ")
print()
print(f"  H(X)         = {H_4:.4f} bits  (theoretical minimum)")
print(f"  Huffman L_avg = {avg_len_4:.4f} bits")
print(f"  Redundancy    = {avg_len_4 - H_4:.4f} bits  (Shannon guarantees < 1 bit)")
print()

# ── Part 2: Sweep compression across many distributions ───────────────────
print("  PART 2 — COMPRESSION EFFICIENCY ACROSS DISTRIBUTIONS")
print()

k = 8   # 8-symbol alphabet
print(f"  {k}-symbol alphabet. Fixed-length needs log₂({k}) = {np.log2(k):.1f} bits.")
print()
print(f"  {'Distribution':28s} | {'H(X)':>8} | {'L_avg':>8} | {'Fixed':>7} | {'Savings vs fixed':>17}")
print(f"  {'─'*75}")

dist_configs = [
    ("Uniform",           np.ones(k) / k),
    ("Slightly skewed",   np.array([0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04])),
    ("Power law α=1.0",   np.array([1/r for r in range(1, k+1)])),
    ("Power law α=2.0",   np.array([1/r**2 for r in range(1, k+1)])),
    ("Very skewed",       np.array([0.70, 0.10, 0.06, 0.04, 0.04, 0.03, 0.02, 0.01])),
    ("Near-degenerate",   np.array([0.97, 0.01, 0.005, 0.005, 0.003, 0.003, 0.002, 0.002])),
]

fixed_len = np.log2(k)
huffman_avgs = []
entropies    = []

for name, probs_raw in dist_configs:
    probs_norm = np.array(probs_raw, dtype=float)
    probs_norm /= probs_norm.sum()
    syms  = [str(i) for i in range(k)]
    tree  = build_huffman_tree(syms, probs_norm)
    codes = get_codes(tree)
    H_val = entropy_bits(probs_norm)
    L_avg = sum(probs_norm[i] * len(codes[str(i)]) for i in range(k))
    savings = fixed_len - L_avg
    entropies.append(H_val)
    huffman_avgs.append(L_avg)
    print(f"  {name:28s} | {H_val:8.4f} | {L_avg:8.4f} | {fixed_len:7.1f} | {savings:17.4f} bits saved")

print()
print(f"  Shannon guarantee: H(X) ≤ L_avg < H(X) + 1  (always holds!)")
gaps = [L - H for L, H in zip(huffman_avgs, entropies)]
print(f"  Observed gaps (L_avg - H): min={min(gaps):.4f}, max={max(gaps):.4f}")
print()

# ── Part 3: Perplexity ──────────────────────────────────────────────────
print("  PART 3 — PERPLEXITY (language model evaluation metric)")
print()
print("  Perplexity = 2^H(P,Q)  where H(P,Q) = -Σ P(x) log₂ Q(x)")
print("  = 'effective vocabulary size the model is confused over'")
print()

vocab_size = 50000
# True next-word distribution (concentrated on 5 plausible words)
true_probs = np.zeros(vocab_size)
true_probs[:5] = [0.40, 0.25, 0.15, 0.12, 0.08]

# Various model quality levels
model_configs = [
    ("Perfect model (Q=P)",         true_probs.copy()),
    ("Good LM (near P)",            np.clip(true_probs + np.random.normal(0, 0.01, vocab_size), 0, None)),
    ("Mediocre LM",                 np.clip(true_probs + np.random.normal(0, 0.05, vocab_size), 0, None)),
    ("Random (uniform)",            np.ones(vocab_size) / vocab_size),
]

print(f"  {'Model':28s} | {'CE loss (nats)':>15} | {'Perplexity':>12}")
print(f"  {'─'*60}")
for mname, q_raw in model_configs:
    q_norm = np.array(q_raw, dtype=float)
    q_norm = np.maximum(q_norm, 1e-12)
    q_norm /= q_norm.sum()
    ce_nats = -np.sum(true_probs * np.log(q_norm + 1e-300))
    ppl_nats = np.exp(ce_nats)
    print(f"  {mname:28s} | {ce_nats:15.4f} | {ppl_nats:12.1f}")

print()
print(f"  Uniform (worst case) PP = vocab_size = {vocab_size:,}")
print(f"  State-of-the-art LLMs: PP ≈ 5–20 on standard benchmarks")
print()

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Huffman Coding: Shannon's Source Coding Theorem",
             fontsize=12, fontweight="bold")

# Plot 1: code lengths vs prob for 4-symbol example
syms_plot = list(codes_4.keys())
probs_plot = [probs_4[symbols_4.index(s)] for s in syms_plot]
lens_plot  = [len(codes_4[s]) for s in syms_plot]
opt_lens   = [-np.log2(p) for p in probs_plot]   # optimal = -log₂(p)

x_pos = np.arange(len(syms_plot))
axes[0].bar(x_pos - 0.2, lens_plot, width=0.35, color="steelblue",
            alpha=0.8, label="Huffman length")
axes[0].bar(x_pos + 0.2, opt_lens,  width=0.35, color="tomato",
            alpha=0.8, label="-log₂(P) [optimal]")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f"{s}\\np={p:.3f}" for s, p in zip(syms_plot, probs_plot)],
                         fontsize=9)
axes[0].set_ylabel("Code length (bits)")
axes[0].set_title("Huffman vs Optimal Code Lengths")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3, axis="y")

# Plot 2: H vs L_avg across distributions
dist_labels = [c[0] for c in dist_configs]
x2 = np.arange(len(dist_labels))
axes[1].bar(x2 - 0.2, entropies,    width=0.35, color="steelblue", alpha=0.8, label="H(X)")
axes[1].bar(x2 + 0.2, huffman_avgs, width=0.35, color="seagreen",  alpha=0.8, label="L_avg (Huffman)")
axes[1].axhline(fixed_len, color="tomato", linestyle="--", lw=2,
                label=f"Fixed-length ({fixed_len:.1f} bits)")
axes[1].set_xticks(x2)
axes[1].set_xticklabels([c[0] for c in dist_configs], rotation=30, ha="right", fontsize=7)
axes[1].set_ylabel("Bits per symbol")
axes[1].set_title("H(X) vs Huffman L_avg vs Fixed-Length")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, axis="y")

# Plot 3: redundancy = L_avg - H
axes[2].bar(x2, gaps, color="purple", alpha=0.8)
axes[2].axhline(1.0, color="tomato", linestyle="--", lw=2,
                label="Shannon bound: gap < 1")
axes[2].set_xticks(x2)
axes[2].set_xticklabels([c[0] for c in dist_configs], rotation=30, ha="right", fontsize=7)
axes[2].set_ylabel("L_avg - H(X) [bits of redundancy]")
axes[2].set_title("Huffman Redundancy (always < 1 bit)")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("huffman_compression.png", dpi=120)
print("  Plot saved → huffman_compression.png")
print()
print("  KEY TAKEAWAYS:")
print("  Shannon's theorem: H(X) ≤ L_avg < H(X)+1 — always, for any source")
print("  Huffman assigns short codes to frequent symbols, long to rare ones")
print("  More skewed distribution → more compression headroom over fixed-length")
print("  Perplexity = 2^CE: the standard LM benchmark; lower = better model")
''',
    },

    # ── 8 ─────────────────────────────────────────────────────────────────────
    "8 · Maximum Entropy Principle — Distributions from Constraints": {
        "description": (
            "Show that MaxEnt recovers familiar distributions from moment constraints. "
            "Demonstrate Gaussian as the max-entropy distribution for fixed variance. "
            "Connect MaxEnt to the softmax output layer and to regularisation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize

np.random.seed(0)

print("=" * 65)
print("  MAXIMUM ENTROPY PRINCIPLE")
print("=" * 65)
print()
print("  'Given known constraints, the least-assumptive distribution")
print("   is the one with MAXIMUM entropy.'")
print()

# ── Helper ──────────────────────────────────────────────────────────────
def differential_entropy_gaussian(sigma):
    """Analytical: h(N(mu, sigma²)) = 0.5 * log(2*pi*e*sigma²) nats."""
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

def entropy_discrete_nats(probs):
    p = np.array(probs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

# ── Part 1: MaxEnt over finite alphabet ──────────────────────────────────
print("  PART 1 — FINITE ALPHABET: NO CONSTRAINTS → UNIFORM IS MAX-ENTROPY")
print()
k_vals = [2, 4, 8, 16, 32]
print(f"  {'k (outcomes)':>14} | {'H_uniform (nats)':>18} | {'Any other dist H':>18} | {'Gap':>8}")
print(f"  {'─'*65}")
for k_val in k_vals:
    uniform_probs = np.ones(k_val) / k_val
    h_uniform = entropy_discrete_nats(uniform_probs)
    # skewed alternative: geometric-like
    alt = np.array([0.5**i for i in range(1, k_val+1)], dtype=float)
    alt /= alt.sum()
    h_alt = entropy_discrete_nats(alt)
    print(f"  {k_val:14d} | {h_uniform:18.4f} | {h_alt:18.4f} | {h_uniform-h_alt:8.4f}")
print()

# ── Part 2: Gaussian as max-entropy for fixed variance ───────────────────
print("  PART 2 — GAUSSIAN MAXIMISES ENTROPY FOR FIXED VARIANCE")
print()
print("  For any distribution with variance σ², h(X) ≤ ½ log(2πeσ²) = h(N(0,σ²))")
print()

sigma = 1.0
h_gaussian = differential_entropy_gaussian(sigma)

n_pts = 5000
distributions_vs_gaussian = [
    ("Gaussian N(0,1)",     np.random.normal(0, sigma, n_pts)),
    ("Uniform ±√3",         np.random.uniform(-np.sqrt(3), np.sqrt(3), n_pts)),
    ("Laplace(0, 1/√2)",    np.random.laplace(0, 1/np.sqrt(2), n_pts)),
    ("t-distribution df=5", np.random.standard_t(5, n_pts) * np.sqrt(3/5)),
    ("Mixture of Gaussians", np.concatenate([
        np.random.normal(-1.5, 0.5, n_pts//2),
        np.random.normal( 1.5, 0.5, n_pts//2)])),
]

print(f"  {'Distribution':26s} | {'Var(X)':>9} | {'h (nats, KDE est.)':>20} | {'h_Gaussian':>12} | {'Gap':>8}")
print(f"  {'─'*85}")
for dname, samples in distributions_vs_gaussian:
    var_s  = np.var(samples)
    # KDE-based entropy estimate
    kde    = stats.gaussian_kde(samples)
    x_eval = np.linspace(samples.min(), samples.max(), 800)
    p_eval = np.maximum(kde(x_eval), 1e-12)
    dx     = x_eval[1] - x_eval[0]
    h_est  = float(-np.sum(p_eval * np.log(p_eval) * dx))
    h_gauss_equiv = differential_entropy_gaussian(np.sqrt(var_s))
    gap    = h_gauss_equiv - h_est
    print(f"  {dname:26s} | {var_s:9.4f} | {h_est:20.4f} | {h_gauss_equiv:12.4f} | {gap:8.4f}")

print()
print("  Gaussian always achieves the highest entropy among distributions")
print("  with the same variance. Gap ≥ 0 for all other distributions.")
print()

# ── Part 3: Softmax as MaxEnt classifier ─────────────────────────────────
print("  PART 3 — SOFTMAX IS THE MAX-ENTROPY CLASSIFIER")
print()
print("  Softmax: P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)")
print("  This is the MaxEnt solution with constraints E[φ(x,y)] = empirical avg.")
print()
print("  Key insight: for K classes, the entropy of the softmax output")
print("  measures the model's uncertainty about its prediction:")
print()

K = 5
logit_configs = [
    ("Very confident  (logit gap=10)", np.array([10.0, 0.0, 0.0, 0.0, 0.0])),
    ("Confident       (logit gap=3)",  np.array([3.0,  0.0, 0.0, 0.0, 0.0])),
    ("Uncertain       (logit gap=0.5)",np.array([0.5,  0.0, 0.0, 0.0, 0.0])),
    ("Maximum entropy (all equal)",    np.zeros(K)),
]

print(f"  {'Scenario':32s} | {'Softmax probs':>32} | {'H (bits)':>9}")
print(f"  {'─'*82}")
for sname, logits in logit_configs:
    logits_shifted = logits - logits.max()
    probs_sm = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
    h_sm = entropy_discrete_nats(probs_sm) / np.log(2)
    probs_str = "[" + ", ".join(f"{p:.3f}" for p in probs_sm) + "]"
    print(f"  {sname:32s} | {probs_str:>32} | {h_sm:9.4f}")

print()
print(f"  Max entropy = log₂({K}) = {np.log2(K):.4f} bits (uniform over all classes)")
print()

# ── Part 4: MaxEnt with mean constraint → Exponential ────────────────────
print("  PART 4 — MEAN CONSTRAINT ON x≥0 → EXPONENTIAL DISTRIBUTION")
print()
print("  Constraint: E[X] = μ  (fixed mean, x ≥ 0)")
print("  Max-entropy solution: Exponential(λ = 1/μ)")
print()

mu_target = 2.0
lam = 1.0 / mu_target
x_exp = np.linspace(0, 15, 1000)
f_exp = lam * np.exp(-lam * x_exp)

candidates = [
    ("Exponential(λ=1/μ) [MaxEnt]",    np.random.exponential(mu_target, 20000)),
    ("Half-Gaussian (σ=μ√(π/2))",      np.abs(np.random.normal(0, mu_target*np.sqrt(np.pi/2)/2, 20000))),
    ("Uniform(0, 2μ)",                  np.random.uniform(0, 2*mu_target, 20000)),
]

print(f"  {'Distribution':36s} | {'Mean':>8} | {'h (nats)':>10} | {'Is MaxEnt':>10}")
print(f"  {'─'*72}")
h_exp_analytical = 1 - np.log(lam)   # analytical h(Exp(λ)) = 1 - log(λ)
for cname, samps in candidates:
    mean_s = samps.mean()
    kde_c  = stats.gaussian_kde(samps)
    xe     = np.linspace(0, samps.max(), 600)
    pe     = np.maximum(kde_c(xe), 1e-12)
    dx_c   = xe[1] - xe[0]
    h_c    = float(-np.sum(pe * np.log(pe) * dx_c))
    is_max = "✓ YES" if cname.startswith("Exponential") else "✗ no"
    print(f"  {cname:36s} | {mean_s:8.4f} | {h_c:10.4f} | {is_max:>10}")

print(f"  Analytical h(Exp(λ={lam:.2f})): {h_exp_analytical:.4f} nats")
print()

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Maximum Entropy Principle — Distributions from Constraints",
             fontsize=12, fontweight="bold")

# Plot 1: H vs k for uniform distribution
k_range = np.arange(2, 65)
h_unif  = np.log2(k_range)
axes[0].plot(k_range, h_unif, "steelblue", lw=2.5)
axes[0].fill_between(k_range, h_unif, alpha=0.15, color="steelblue")
for k_mark in [2, 4, 8, 16, 32, 64]:
    if k_mark <= k_range[-1]:
        axes[0].axvline(k_mark, color="gray", linestyle=":", lw=0.8)
axes[0].set_xlabel("Number of outcomes k")
axes[0].set_ylabel("Max entropy H_max = log₂(k) [bits]")
axes[0].set_title("Max Entropy Grows with Alphabet Size\\n(Uniform achieves it)")
axes[0].grid(alpha=0.3)

# Plot 2: distributions vs Gaussian (same variance), showing Gaussian wins
dist_names_plot = [d[0] for d in distributions_vs_gaussian]
h_estimates_plot = []
h_gaussian_equivs_plot = []
for dname, samples in distributions_vs_gaussian:
    var_s  = np.var(samples)
    kde_   = stats.gaussian_kde(samples)
    xe_    = np.linspace(samples.min(), samples.max(), 600)
    pe_    = np.maximum(kde_(xe_), 1e-12)
    dx_    = xe_[1] - xe_[0]
    h_estimates_plot.append(float(-np.sum(pe_ * np.log(pe_) * dx_)))
    h_gaussian_equivs_plot.append(differential_entropy_gaussian(np.sqrt(var_s)))

x3 = np.arange(len(dist_names_plot))
axes[1].bar(x3 - 0.2, h_gaussian_equivs_plot, width=0.35, color="tomato",
            alpha=0.8, label="h(Gaussian, same var)")
axes[1].bar(x3 + 0.2, h_estimates_plot,        width=0.35, color="steelblue",
            alpha=0.8, label="h(distribution)")
axes[1].set_xticks(x3)
axes[1].set_xticklabels([d.split(" ")[0] for d in dist_names_plot],
                         rotation=20, ha="right", fontsize=8)
axes[1].set_ylabel("Differential entropy (nats)")
axes[1].set_title("Gaussian Has Max Entropy\\nfor Fixed Variance")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, axis="y")

# Plot 3: Softmax entropy vs confidence
gap_range = np.linspace(0, 8, 200)
h_softmax = []
for gap in gap_range:
    logits = np.array([gap] + [0.0]*(K-1))
    logits -= logits.max()
    probs_sm = np.exp(logits) / np.exp(logits).sum()
    h_softmax.append(entropy_discrete_nats(probs_sm) / np.log(2))

axes[2].plot(gap_range, h_softmax, "seagreen", lw=2.5)
axes[2].axhline(np.log2(K), color="tomato", linestyle="--", lw=2,
                label=f"Max H = log₂({K}) = {np.log2(K):.2f} bits")
axes[2].set_xlabel("Logit gap (top class − others)")
axes[2].set_ylabel("Softmax output entropy H [bits]")
axes[2].set_title("Softmax Entropy vs Model Confidence\\n(MaxEnt = uniform = most uncertain)")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("max_entropy.png", dpi=120)
print("  Plot saved → max_entropy.png")
print()
print("  KEY TAKEAWAYS:")
print("  MaxEnt = least-assumptive distribution given what you know")
print("  No constraints   → Uniform (maximum entropy on finite alphabet)")
print("  Fixed variance   → Gaussian (max differential entropy)")
print("  Fixed mean, x≥0  → Exponential")
print("  Linear features  → Softmax / logistic regression")
print("  MaxEnt justifies: Gaussian noise, softmax output, and regularisation")
''',
    },

    # ── 9 ─────────────────────────────────────────────────────────────────────
    "9 · Information Bottleneck — Compression vs Task Relevance": {
        "description": (
            "Simulate the information bottleneck trade-off. "
            "Build a toy dataset and train representations with varying β. "
            "Plot the IB curve: I(X;Z) on x-axis, I(Z;Y) on y-axis. "
            "Show how the β parameter controls compression vs task relevance."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("=" * 65)
print("  INFORMATION BOTTLENECK — COMPRESSION vs TASK RELEVANCE")
print("=" * 65)
print()
print("  IB Lagrangian: min I(X;Z) - β·I(Z;Y)")
print("  β=0: compress maximally; β→∞: preserve all task-relevant info")
print()

# ── Discrete IB on a toy joint distribution ───────────────────────────────
# X: 8 input symbols  |  Y: 2 classes  |  Z: compressed representation

# True P(X, Y) — first 4 symbols mostly class 0, last 4 mostly class 1
px = np.ones(8) / 8    # uniform input
# P(Y|X): noisy class labels
py_given_x = np.array([
    [0.90, 0.10],   # x=0 → strongly class 0
    [0.85, 0.15],   # x=1 → strongly class 0
    [0.80, 0.20],   # x=2 → mostly class 0
    [0.70, 0.30],   # x=3 → weakly class 0
    [0.30, 0.70],   # x=4 → weakly class 1
    [0.20, 0.80],   # x=5 → mostly class 1
    [0.15, 0.85],   # x=6 → strongly class 1
    [0.10, 0.90],   # x=7 → strongly class 1
])

# Joint P(X, Y)
pxy = px[:, np.newaxis] * py_given_x
py  = pxy.sum(axis=0)

def entropy_nats(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-12]
    p /= p.sum()
    return -np.sum(p * np.log(p))

def mi_xy(p_joint):
    """Mutual information from a joint distribution matrix."""
    px_  = p_joint.sum(axis=1, keepdims=True)
    py_  = p_joint.sum(axis=0, keepdims=True)
    prod = px_ * py_
    mask = (p_joint > 1e-12) & (prod > 1e-12)
    return float(np.sum(p_joint[mask] * np.log(p_joint[mask] / prod[mask])))

I_XY = mi_xy(pxy)
H_X  = entropy_nats(px)
H_Y  = entropy_nats(py)

print(f"  Joint P(X,Y) setup:")
print(f"    |X| = 8 symbols,  |Y| = 2 classes")
print(f"    H(X) = {H_X:.4f} nats   H(Y) = {H_Y:.4f} nats")
print(f"    I(X;Y) = {I_XY:.4f} nats  (max task-relevant info)")
print()

# ── Iterate over clusterings Z with different resolutions ─────────────────
# Model Z as a soft clustering of X: p(z|x) — encoder distribution
# We use progressively coarser clusterings to sweep the IB curve

print("  SWEEPING β — DIFFERENT COMPRESSION LEVELS:")
print()
print(f"  {'Description':30s} | {'I(X;Z) nats':>12} | {'I(Z;Y) nats':>12} | {'η = I(Z;Y)/I(X;Y)':>20}")
print(f"  {'─'*82}")

ib_points = []   # (I_XZ, I_ZY) pairs for the IB curve

# Define a set of encoder matrices p(z|x) from fine to coarse
encoders = [
    # Identity (no compression) — each x maps to its own z
    ("Identity (β→∞)",        np.eye(8)),
    # 4-cluster: {0,1} {2,3} {4,5} {6,7}
    ("4 clusters (fine)",     np.array([
        [1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],
        [0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]], dtype=float)),
    # 3-cluster: {0,1,2} {3,4} {5,6,7}
    ("3 clusters",            np.array([
        [1,0,0],[1,0,0],[1,0,0],[0,1,0],
        [0,1,0],[0,0,1],[0,0,1],[0,0,1]], dtype=float)),
    # 2-cluster: {0,1,2,3} {4,5,6,7} (class boundary)
    ("2 clusters (boundary)", np.array([
        [1,0],[1,0],[1,0],[1,0],
        [0,1],[0,1],[0,1],[0,1]], dtype=float)),
    # 2-cluster: bad boundary {0,1,2,3,4} {5,6,7}
    ("2 clusters (bad split)", np.array([
        [1,0],[1,0],[1,0],[1,0],[1,0],
        [0,1],[0,1],[0,1]], dtype=float)),
    # 1 cluster (maximum compression)
    ("1 cluster (β=0)",       np.ones((8,1))),
]

for desc, encoder in encoders:
    enc = encoder.copy().astype(float)
    enc /= enc.sum(axis=1, keepdims=True)   # normalise rows → proper p(z|x)

    n_z = enc.shape[1]

    # p(z) = Σ_x p(x) p(z|x)
    pz = px @ enc   # shape (n_z,)

    # p(x, z) = p(x) * p(z|x)
    pxz = px[:, np.newaxis] * enc   # shape (8, n_z)

    # p(z, y) = Σ_x p(x, y) p(z|x)
    pzy = pxy.T @ enc   # shape (2, n_z) → transpose to (n_z, 2)
    pzy = pzy.T         # shape (n_z, 2) each row is a z, cols are y classes

    I_XZ = mi_xy(pxz)   # I(X;Z)
    I_ZY = mi_xy(pzy)   # I(Z;Y)
    eta  = I_ZY / I_XY if I_XY > 1e-10 else 0.0

    ib_points.append((I_XZ, I_ZY, desc))
    print(f"  {desc:30s} | {I_XZ:12.4f} | {I_ZY:12.4f} | {eta:20.4f}")

print()
print(f"  I(X;Y) = {I_XY:.4f} nats — the ceiling for I(Z;Y)")
print(f"  Best compression without losing task info: 2-cluster at class boundary")
print()

# ── Continuous IB simulation: noisy channel model ─────────────────────────
print("  CONTINUOUS IB SIMULATION:")
print()
print("  X ~ N(0,1),  Y = sign(X),  Z = X + β-dependent noise")
print("  More noise = more compression = lower I(X;Z), lower I(Z;Y)")
print()

n_ib = 50000
X_cont = np.random.normal(0, 1, n_ib)
Y_cont = (X_cont > 0).astype(float)

noise_levels = np.logspace(-2, 2, 30)   # σ_noise from 0.01 to 100
ixz_cont, izy_cont = [], []

for sigma_noise in noise_levels:
    Z_cont = X_cont + np.random.normal(0, sigma_noise, n_ib)
    # I(X;Z) ≈ 0.5 * log(1 + σ_X² / σ_noise²)   [Gaussian channel formula]
    # I(Z;Y): mutual info between noisy channel output and binary label
    ixz_val = 0.5 * np.log(1 + 1.0 / sigma_noise**2)
    # Estimate I(Z;Y) numerically
    z_pos = Z_cont[Y_cont == 1]
    z_neg = Z_cont[Y_cont == 0]
    # Use MI formula via conditional entropy
    h_z   = float(entropy_nats(np.histogram(Z_cont, bins=100, density=True)[0] + 1e-12))
    h_z_y0 = float(entropy_nats(np.histogram(z_neg, bins=80, density=True)[0] + 1e-12))
    h_z_y1 = float(entropy_nats(np.histogram(z_pos, bins=80, density=True)[0] + 1e-12))
    izy_val = max(0, h_z - 0.5*h_z_y0 - 0.5*h_z_y1)
    ixz_cont.append(ixz_val)
    izy_cont.append(izy_val)

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Information Bottleneck: Compression vs Task Relevance",
             fontsize=12, fontweight="bold")

# Plot 1: Discrete IB curve
ixz_vals = [pt[0] for pt in ib_points]
izy_vals = [pt[1] for pt in ib_points]
descs    = [pt[2] for pt in ib_points]
colours_ib = plt.cm.plasma(np.linspace(0.1, 0.9, len(ib_points)))

axes[0].scatter(ixz_vals, izy_vals, c=colours_ib, s=120, zorder=5)
for i, (ix, iy, desc) in enumerate(ib_points):
    axes[0].annotate(desc.split("(")[0].strip(), (ix, iy),
                     textcoords="offset points", xytext=(5, 4), fontsize=7)
axes[0].axhline(I_XY, color="tomato", linestyle="--", lw=1.5,
                label=f"Max I(Z;Y) = I(X;Y) = {I_XY:.3f}")
axes[0].set_xlabel("I(X;Z) — compression (more → less compressed)")
axes[0].set_ylabel("I(Z;Y) — task relevance")
axes[0].set_title("Discrete IB Curve\\n(encoders of X → Z)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Plot 2: Continuous IB curve
axes[1].plot(ixz_cont, izy_cont, "steelblue", lw=2.5)
axes[1].scatter(ixz_cont[0],  izy_cont[0],  color="tomato",   s=100, zorder=5,
                label="High noise (max compress)")
axes[1].scatter(ixz_cont[-1], izy_cont[-1], color="seagreen", s=100, zorder=5,
                label="Low noise (min compress)")
axes[1].set_xlabel("I(X;Z) — compression")
axes[1].set_ylabel("I(Z;Y) — task relevance")
axes[1].set_title("Continuous IB Curve\\nZ = X + Gaussian noise")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Plot 3: β trade-off — show I(X;Z) and I(Z;Y) as noise varies
axes[2].semilogx(noise_levels, ixz_cont, "steelblue", lw=2.5, label="I(X;Z) — compression")
axes[2].semilogx(noise_levels, izy_cont, "seagreen",  lw=2.5, label="I(Z;Y) — task info")
axes[2].set_xlabel("Noise σ (proxy for 1/β)")
axes[2].set_ylabel("Mutual information (nats)")
axes[2].set_title("β Trade-off: More Compression\\n→ Less Task Information")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)
axes[2].annotate("← high β\\n(less compress)", xy=(noise_levels[3], ixz_cont[3]),
                 xytext=(noise_levels[3]*2, ixz_cont[3]+0.3), fontsize=8,
                 arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig("information_bottleneck.png", dpi=120)
print("  Plot saved → information_bottleneck.png")
print()
print("  KEY TAKEAWAYS:")
print("  IB trades off I(X;Z) (compression) vs I(Z;Y) (task relevance)")
print("  β controls this: β=0 → compress everything; β→∞ → keep all info")
print("  A good representation lies on the IB frontier (Pareto-optimal)")
print("  Deep networks may compress X while retaining Y-relevant features")
print("  β-VAE directly implements IB: β scales the KL(q||p) penalty")
''',
    },

    # ── 10 ────────────────────────────────────────────────────────────────────
    "10 · Fisher Information & Differential Entropy Nuances": {
        "description": (
            "Compute Fisher information for common distributions and verify "
            "the Cramér-Rao bound. Show the natural gradient vs vanilla gradient. "
            "Demonstrate differential entropy nuances: negativity, "
            "non-invariance to scaling, and Gaussian as the maximum."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("=" * 65)
print("  FISHER INFORMATION & DIFFERENTIAL ENTROPY NUANCES")
print("=" * 65)
print()

# ── Part 1: Fisher Information for common distributions ───────────────────
print("  PART 1 — FISHER INFORMATION I(θ) = E[(∂ log p / ∂θ)²]")
print()
print("  Cramér-Rao bound: Var(θ̂) ≥ 1 / (n · I(θ))  for any unbiased estimator")
print()

print(f"  {'Distribution':28s} | {'Parameter θ':>12} | {'I(θ) (1 obs)':>14} | "
      f"{'CR bound n=30':>15} | {'MLE Var(n=30)':>14}")
print(f"  {'─'*90}")

N = 30
N_MC = 50000

fisher_examples = [
    # (name, true_theta, sampler(theta, n), mle(samples), analytical_fisher)
    ("Bernoulli(p)",      0.4,
     lambda p, n: np.random.binomial(1, p, n),
     lambda s: s.mean(),
     lambda p: 1 / (p * (1-p))),

    ("Poisson(λ)",        3.0,
     lambda lam, n: np.random.poisson(lam, n),
     lambda s: s.mean(),
     lambda lam: 1 / lam),

    ("Gaussian(μ, σ=1)",  2.0,
     lambda mu, n: np.random.normal(mu, 1.0, n),
     lambda s: s.mean(),
     lambda mu: 1.0),           # I(μ) = 1/σ² = 1 when σ=1

    ("Exponential(λ)",    0.5,
     lambda lam, n: np.random.exponential(1/lam, n),
     lambda s: 1/s.mean(),
     lambda lam: 1 / lam**2),
]

for name, theta, sampler, mle_fn, fisher_fn in fisher_examples:
    fisher_analytical = fisher_fn(theta)
    cr_bound = 1 / (N * fisher_analytical)

    # Monte Carlo estimate of MLE variance
    mle_estimates = [mle_fn(sampler(theta, N)) for _ in range(N_MC)]
    mle_var = np.var(mle_estimates)

    print(f"  {name:28s} | {theta:12.4f} | {fisher_analytical:14.4f} | "
          f"{cr_bound:15.6f} | {mle_var:14.6f}")

print()
print("  MLE Var ≈ CR Bound in each row — MLE is efficient (achieves the bound)!")
print()

# ── Part 2: Fisher info vs sample size — tightening the CR bound ──────────
print("  PART 2 — FISHER INFO SCALES LINEARLY WITH n")
print()
print("  I_n(θ) = n · I₁(θ)  (i.i.d. samples are additive in information)")
print()

theta_bern = 0.4
I_1 = 1 / (theta_bern * (1 - theta_bern))

print(f"  Bernoulli(p={theta_bern}):  I₁(p) = 1/(p(1-p)) = {I_1:.4f}")
print()
print(f"  {'n':>6} | {'I_n = n·I₁':>14} | {'CR bound 1/I_n':>16} | "
      f"{'MC MLE Var':>12} | {'ratio MLE/CR':>13}")
print(f"  {'─'*70}")

n_vals = [5, 10, 30, 100, 300, 1000]
for n_val in n_vals:
    I_n     = n_val * I_1
    cr      = 1 / I_n
    mle_ests = np.array([
        np.random.binomial(1, theta_bern, n_val).mean()
        for _ in range(N_MC)])
    mc_var  = np.var(mle_ests)
    ratio   = mc_var / cr
    print(f"  {n_val:6d} | {I_n:14.4f} | {cr:16.6f} | {mc_var:12.6f} | {ratio:13.4f}")

print()
print("  Ratio ≈ 1 throughout — MLE achieves the Cramér-Rao bound.")
print()

# ── Part 3: Differential entropy nuances ─────────────────────────────────
print("  PART 3 — DIFFERENTIAL ENTROPY NUANCES")
print()

def h_kde(samples):
    """KDE-based differential entropy estimate in nats."""
    kde_   = stats.gaussian_kde(samples)
    x_eval = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 600)
    p_eval = np.maximum(kde_(x_eval), 1e-12)
    dx_    = x_eval[1] - x_eval[0]
    return float(-np.sum(p_eval * np.log(p_eval) * dx_))

print("  3a — Differential entropy CAN be negative:")
print()
n_samp = 50000
negative_examples = [
    ("Uniform(0, 0.1)",   np.random.uniform(0, 0.1, n_samp),   np.log(0.1)),
    ("Uniform(0, 0.5)",   np.random.uniform(0, 0.5, n_samp),   np.log(0.5)),
    ("Uniform(0, 1)",     np.random.uniform(0, 1.0, n_samp),   np.log(1.0)),
    ("Uniform(0, 2)",     np.random.uniform(0, 2.0, n_samp),   np.log(2.0)),
    ("Gaussian N(0,0.1)", np.random.normal(0, 0.1, n_samp),    0.5*np.log(2*np.pi*np.e*0.01)),
]

print(f"  {'Distribution':24s} | {'h (KDE est)':>13} | {'h (analytic)':>14} | {'Negative?':>10}")
print(f"  {'─'*68}")
for dname, samps, h_anal in negative_examples:
    h_est = h_kde(samps)
    neg = "YES ← !" if h_anal < 0 else "no"
    print(f"  {dname:24s} | {h_est:13.4f} | {h_anal:14.4f} | {neg:>10}")

print()
print("  3b — h(aX) = h(X) + log|a|  (NOT invariant to scaling!):")
print()
X_base = np.random.normal(0, 1, n_samp)
h_base = h_kde(X_base)
print(f"  {'Scaling':18s} | {'h(aX) estimated':>17} | {'h(X)+log|a|':>14} | {'Match':>7}")
print(f"  {'─'*62}")
for a_scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
    h_scaled = h_kde(a_scale * X_base)
    h_theory = h_base + np.log(a_scale)
    match = "✓" if abs(h_scaled - h_theory) < 0.05 else "✗"
    print(f"  a={a_scale:14.1f} | {h_scaled:17.4f} | {h_theory:14.4f} | {match:>7}")

print()
print("  3c — Gaussian maximises h for fixed variance:")
print()
sigma_fixed = 1.5
h_gauss_max = 0.5 * np.log(2 * np.pi * np.e * sigma_fixed**2)
print(f"  Fixed σ² = {sigma_fixed**2:.2f}  →  h_max = ½log(2πeσ²) = {h_gauss_max:.4f} nats")
print()
print(f"  {'Distribution (same var)':28s} | {'h (nats)':>10} | {'Gap to Gaussian':>16}")
print(f"  {'─'*60}")
for dname_c, samps_c in [
    ("Gaussian N(0,σ)",         np.random.normal(0, sigma_fixed, n_samp)),
    ("Laplace(0,σ/√2)",         np.random.laplace(0, sigma_fixed/np.sqrt(2), n_samp)),
    ("Uniform(±σ√3)",           np.random.uniform(-sigma_fixed*np.sqrt(3),
                                                    sigma_fixed*np.sqrt(3), n_samp)),
    ("t-dist (df=5, scaled)",   stats.t.rvs(5, scale=sigma_fixed*np.sqrt(3/5), size=n_samp)),
]:
    h_c   = h_kde(samps_c)
    gap_c = h_gauss_max - h_c
    print(f"  {dname_c:28s} | {h_c:10.4f} | {gap_c:16.4f}")

print()

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Fisher Information & Differential Entropy Nuances",
             fontsize=12, fontweight="bold")

# Plot 1: CR bound vs n for Bernoulli
n_range = np.arange(1, 201)
cr_range = 1 / (n_range * I_1)
axes[0].semilogy(n_range, cr_range, "steelblue", lw=2.5, label="CR bound 1/(n·I₁)")
axes[0].semilogy(n_range, theta_bern*(1-theta_bern)/n_range, "tomato",
                 linestyle="--", lw=2, label="MLE Var = p(1-p)/n")
axes[0].set_xlabel("Sample size n")
axes[0].set_ylabel("Variance (log scale)")
axes[0].set_title(f"Cramér-Rao Bound — Bernoulli(p={theta_bern})\\nMLE achieves the bound")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Plot 2: h(aX) vs scaling factor a
a_range = np.linspace(0.1, 10, 200)
h_theory_range = h_base + np.log(a_range)
axes[1].plot(a_range, h_theory_range, "steelblue", lw=2.5, label="h(X) + log(a) [theory]")
axes[1].axhline(h_base, color="gray", linestyle="--", lw=1.5, label=f"h(X) = {h_base:.3f}")
axes[1].axvline(1.0,    color="tomato", linestyle=":", lw=1.5, label="a=1 (no scaling)")
axes[1].set_xlabel("Scaling factor a")
axes[1].set_ylabel("h(aX) nats")
axes[1].set_title("h(aX) = h(X) + log|a|\\nDifferential entropy is NOT scale-invariant")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Plot 3: Gaussian vs others (same variance), showing Gaussian wins
dist_names_p3 = ["Gaussian\\nN(0,σ)", "Laplace\\n(0,σ/√2)", "Uniform\\n±σ√3", "t-dist\\n(df=5)"]
h_vals_p3 = []
for dname_p3, samps_p3 in [
    ("Gaussian", np.random.normal(0, sigma_fixed, n_samp)),
    ("Laplace",  np.random.laplace(0, sigma_fixed/np.sqrt(2), n_samp)),
    ("Uniform",  np.random.uniform(-sigma_fixed*np.sqrt(3), sigma_fixed*np.sqrt(3), n_samp)),
    ("t-dist",   stats.t.rvs(5, scale=sigma_fixed*np.sqrt(3/5), size=n_samp)),
]:
    h_vals_p3.append(h_kde(samps_p3))

colours_p3 = ["tomato", "steelblue", "seagreen", "purple"]
bars_p3 = axes[2].bar(dist_names_p3, h_vals_p3, color=colours_p3, alpha=0.8)
axes[2].axhline(h_gauss_max, color="tomato", linestyle="--", lw=2,
                label=f"Gaussian max h = {h_gauss_max:.3f}")
axes[2].set_ylabel("Differential entropy h (nats)")
axes[2].set_title(f"Gaussian Maximises h for Fixed σ²={sigma_fixed**2:.2f}\\n(same variance, different shapes)")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3, axis="y")
for bar, val in zip(bars_p3, h_vals_p3):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("fisher_differential_entropy.png", dpi=120)
print("  Plot saved → fisher_differential_entropy.png")
print()
print("  KEY TAKEAWAYS:")
print("  Fisher I(θ) = 'info one sample carries about θ' → sets estimation limit")
print("  Cramér-Rao: Var(θ̂) ≥ 1/(n·I(θ)) — MLE achieves this (efficient)")
print("  I_n = n·I_1: more i.i.d. samples → proportionally more information")
print("  Differential entropy CAN be negative (narrow Gaussians, small Uniforms)")
print("  h(aX) = h(X) + log|a|: NOT invariant to reparametrisation (unlike KL)")
print("  Gaussian uniquely maximises h for fixed variance → justifies MSE loss")
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