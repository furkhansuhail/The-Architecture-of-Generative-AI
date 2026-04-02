"""
The No Free Lunch Theorem
==========================

Why no single machine learning algorithm dominates all others, what this
means for how we choose and evaluate models, and why the theorem is both
a profound constraint and a practical guide for every ML practitioner.

"""

import textwrap
import re

TOPIC_NAME = "The No Free Lunch Theorem"
DISPLAY_NAME = "04 · No Free Lunch Theorem"
ICON = "🍽️"
SUBTITLE = "Why No Algorithm Rules Them All"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Insight: All Algorithms are Equal in Ignorance

The No Free Lunch (NFL) Theorem, proved by David Wolpert and William
Macready in 1997, states a fundamental and humbling truth about machine
learning:

    When averaged over ALL possible problem distributions, every
    learning algorithm performs identically — no better than random
    search.

There is no universally superior algorithm. Any algorithm that performs
better than random on one class of problems must perform worse than random
on the complementary class of problems. Every gain is paid for somewhere.

This does not mean "all algorithms are equally good in practice." It means
that algorithm superiority is always *conditional* — it only holds given
specific assumptions about the problem structure.

The theorem has two major variants with different scopes:

    •   NFL for Search and Optimisation  — Wolpert & Macready 1995/1997
    •   NFL for Supervised Learning      — Wolpert 1996


──────────────────────────────────────────────────────────────────────────────
### Formal Statement (Supervised Learning Version)

Let:
    X    = input space (all possible feature vectors)
    Y    = output space (all possible labels)
    f    = the true target function  f: X → Y
    A, B = two learning algorithms
    d    = training dataset sampled from some distribution P(f)
    E(A) = expected off-training-set error of algorithm A

The NFL theorem states:

    ∑_f  E(A | f, d)  =  ∑_f  E(B | f, d)

    Summed over ALL possible target functions f, every algorithm A and B
    achieves exactly the same expected generalisation error.

Equivalently:

    𝔼_f [ E(A | f) ]  =  𝔼_f [ E(B | f) ]  for all A, B

    If A outperforms B for some functions, B outperforms A for others —
    and the excesses cancel exactly.

    Proof Sketch:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  There are |Y|^|X| possible target functions (all mappings X → Y).  │
    │  For any training set d, the unobserved portion of f is completely   │
    │  unconstrained — any completion is equally likely under the uniform  │
    │  prior over all f.                                                   │
    │                                                                      │
    │  Since A's predictions on unseen data are based on regularities in d │
    │  — and regularities in d are uncorrelated with regularities in f on  │
    │  unseen data under a uniform prior — no algorithm can do better than │
    │  guessing.                                                            │
    │                                                                      │
    │  The total number of f's where A beats B exactly equals the number  │
    │  where B beats A. Summed across all f, the errors are equal.         │
    └──────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Intuition: The Function Space is Vast and Flat

Imagine all possible functions from inputs to outputs laid out as a landscape.

    Diagram 1 — All Possible Functions (Simplified: X={0,1,2,3}, Y={0,1}):

    There are 2^4 = 16 possible binary functions on 4 inputs:

    f₁:  0,0,0,0    f₅:  0,1,0,0    f₉:  1,0,0,0    f₁₃: 1,1,0,0
    f₂:  0,0,0,1    f₆:  0,1,0,1    f₁₀: 1,0,0,1    f₁₄: 1,1,0,1
    f₃:  0,0,1,0    f₇:  0,1,1,0    f₁₁: 1,0,1,0    f₁₅: 1,1,1,0
    f₄:  0,0,1,1    f₈:  0,1,1,1    f₁₂: 1,0,1,1    f₁₆: 1,1,1,1

    Suppose you observe training data: {(0,0), (1,0)} — inputs 0 and 1
    both map to output 0.

    Remaining predictions needed: inputs 2 and 3.
    Possible completions still consistent with training data:

        f: 0,0,_,_   →   4 completions (f₁, f₂, f₃, f₄)

    Algorithm A predicts (0,0) for inputs 2 and 3 (e.g., "predict 0")
    Algorithm B predicts (1,1) for inputs 2 and 3 (e.g., "predict 1")

    Over the 4 consistent completions:
        A is correct for f₁.              B is correct for f₄.
        A is partially correct for f₂,f₃. B is partially correct for f₂,f₃.
        Total correct predictions: A=5,   B=5.   EXACTLY equal.

    No matter how clever A's "inductive bias" seems — predict what you
    saw — it gains nothing over the opposite bias on this function space.


──────────────────────────────────────────────────────────────────────────────
### The Optimisation Version (Search)

The NFL theorem was originally proved for search and optimisation, not just
supervised learning. It applies to any algorithm that evaluates an objective
function and tries to find its minimum (or maximum).

    Let f: X → Y be a real-valued cost function.
    Let a₁, a₂, ..., aₜ be the sequence of points evaluated so far.
    Let dₜ = {(a₁, f(a₁)), ..., (aₜ, f(aₜ))} be the history.

    NFL Theorem for Optimisation:

        ∑_f  P(dₜ | f, m)  =  ∑_f  P(dₜ | f, m')

    For any two optimisation algorithms m and m', the distributions over
    possible performance histories are identical when summed over all f.

    Diagram 2 — What "Averaged Over All f" Means for Optimisation:

    Consider two optimisation algorithms on the same search space:

    GRADIENT DESCENT              RANDOM SEARCH
    ──────────────────────        ─────────────────────
    Step towards lower slope.     Evaluate random points.
    Excellent on smooth,          Excellent on chaotic,
    convex functions.             non-smooth functions.

    Smooth convex f:              Chaotic f:
    GD: ✓✓✓✓✓✓✓✓✓✓              GD: ✗✗✗✗✗✗✗✗✗✗
    RS: ✗✗✗✗✗✗✗✗✗✗              RS: ✓✓✓✓✓✓✓✓✓✓

    Summed across ALL f (smooth + chaotic equally likely):
    GD total score = RS total score   → exactly tied.

    The advantage of gradient descent on smooth functions is exactly
    cancelled by its disadvantage on every other type of function.


──────────────────────────────────────────────────────────────────────────────
### What the Theorem Does NOT Say

The NFL theorem is widely misunderstood. Clarifying its limits is as
important as understanding the theorem itself.

**NFL does NOT say: "all algorithms are equally good in practice."**

    Real problems are NOT drawn uniformly at random from all possible
    functions. Physical, biological, and social processes have structure —
    smoothness, locality, continuity, hierarchy, sparsity. Algorithms that
    exploit this structure outperform those that don't.

    ┌───────────────────────────────────────────────────────────────────┐
    │  NFL holds under a UNIFORM PRIOR over all possible functions.     │
    │  The real world is NOT uniform. It has structure.                 │
    │  Algorithms that match that structure win in practice.            │
    └───────────────────────────────────────────────────────────────────┘

**NFL does NOT say: "you can't beat random search on your specific task."**

    You absolutely can — and should — beat random search on any specific
    well-defined task. NFL only prevents guaranteed wins across all tasks
    simultaneously. Domain-specific knowledge breaks the symmetry.

**NFL does NOT say: "algorithm design is pointless."**

    On the contrary — it clarifies exactly WHERE design should be focused:
    on encoding the right inductive biases for the problem class at hand.
    The theorem explains why this is both necessary and sufficient.

**NFL does NOT apply when the prior is non-uniform.**

    If we know the problem is smooth, continuous, and low-dimensional,
    gradient descent is genuinely superior to random search — not just on
    some functions but on the entire relevant subspace. The uniform prior
    assumption is what makes all algorithms equal. Realistic priors destroy
    that equality.

    Summary of What NFL Says vs Doesn't Say:
    ┌────────────────────────────────────┬──────────────────────────────────┐
    │ NFL DOES say                       │ NFL does NOT say                 │
    ├────────────────────────────────────┼──────────────────────────────────┤
    │ No algorithm beats all others on   │ All algorithms perform equally   │
    │ all problems simultaneously        │ on any specific problem          │
    ├────────────────────────────────────┼──────────────────────────────────┤
    │ Gains on one class of problems     │ You can't beat random search on  │
    │ are paid for on other classes      │ a concrete real-world task       │
    ├────────────────────────────────────┼──────────────────────────────────┤
    │ Without prior knowledge, all       │ Prior knowledge is impossible    │
    │ algorithms are equally valid       │ to obtain or use                 │
    ├────────────────────────────────────┼──────────────────────────────────┤
    │ Algorithm choice encodes           │ All inductive biases are equally │
    │ assumptions about problem          │ valid or invalid                 │
    │ structure                          │                                  │
    └────────────────────────────────────┴──────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Inductive Bias: The Key Concept Unlocked by NFL

If NFL makes all algorithms equal under ignorance, what breaks the symmetry
and makes one algorithm better than another on a real problem?

The answer is **inductive bias** — the set of assumptions an algorithm
makes about the structure of the target function, beyond what the training
data explicitly says.

    Inductive bias is NOT a flaw. It is a necessity.
    Without assumptions, you cannot generalise from finite data at all.
    The question is never "should we have bias?" but "which bias is right?"

    Diagram 3 — Inductive Bias of Common Algorithms:

    ┌────────────────────────────┬──────────────────────────────────────────┐
    │ Algorithm                  │ Inductive Bias (Implicit Assumptions)    │
    ├────────────────────────────┼──────────────────────────────────────────┤
    │ Linear Regression          │ Target is a linear function of inputs    │
    │ KNN (k=1)                  │ Similar inputs have similar outputs;     │
    │                            │ local smoothness everywhere              │
    │ Decision Trees             │ Target can be approximated by axis-      │
    │                            │ aligned rectangular regions              │
    │ Naïve Bayes                │ Features are conditionally independent   │
    │                            │ given the class label                    │
    │ Neural Networks (deep)     │ Target is a composition of many simple   │
    │                            │ non-linear transformations; hierarchical │
    │ SVMs (RBF kernel)          │ Nearby points (in kernel space) share    │
    │                            │ labels; smooth decision boundaries       │
    │ Random Forests             │ Ensemble of trees covers diverse         │
    │                            │ rectangular decompositions               │
    │ Gaussian Processes         │ Target is a sample from a smooth         │
    │                            │ Gaussian random process                  │
    └────────────────────────────┴──────────────────────────────────────────┘

    The STRONGER and MORE CORRECT the inductive bias for your problem,
    the better the algorithm will perform — and the LESS data it needs.


──────────────────────────────────────────────────────────────────────────────
### The Bias-Variance-NFL Triangle

The NFL theorem connects deeply with the bias-variance tradeoff from
Module 02. Together they form a three-part framework:

    Diagram 4 — The NFL-Bias-Variance Triangle:

                            NO FREE LUNCH
                           ╱              ╲
                    All algorithms       Gains on
                    equal under          some tasks
                    ignorance            → losses elsewhere
                         ╲              ╱
                           INDUCTIVE BIAS
                          ╱              ╲
                    High bias           Low bias
                    (strong             (flexible
                    assumptions)        assumptions)
                    → Low variance      → High variance
                    → May underfit      → May overfit
                         ╲              ╱
                        BIAS-VARIANCE TRADEOFF

    The NFL theorem tells you THAT you need inductive bias.
    The bias-variance tradeoff tells you HOW MUCH bias to use.
    Together: choose the right TYPE of bias for your data structure.


──────────────────────────────────────────────────────────────────────────────
### The Ugly Duckling Theorem — A Close Relative

Related to NFL is the Ugly Duckling Theorem (Watanabe, 1969), which shows:

    Without a prior over feature relevance, ALL objects are equally
    similar to each other. There is no "natural" clustering or
    categorisation without assumptions.

    Example:

    Consider three objects: a white swan, a black swan, and an ugly duckling.
    In the feature space of all possible boolean predicates:

        P₁: "is a bird"          P₂: "is white"
        P₃: "can swim"           P₄: "has feathers"
        P₅: "is ugly"            ... (infinitely many predicates)

    White Swan  and  Black Swan  share:  P₁, P₃, P₄, ...  (most predicates)
    White Swan  and  Ugly Duckling share: P₃, P₄, ...  (slightly fewer)

    BUT if we count ALL possible predicates (including "is white OR ugly"),
    every pair of objects shares exactly the same number of predicates.

    Implication for ML:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Feature engineering is not just helpful — it is theoretically      │
    │  required. Choosing which features to include encodes a prior       │
    │  about what similarities matter. Without that choice, all           │
    │  similarities are equally meaningless.                              │
    └─────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### NFL and Model Selection in Practice

The NFL theorem has direct, practical implications for how we select and
evaluate models:

**Implication 1: There is no default "best algorithm."**

    Claims like "neural networks always win" or "gradient boosting is
    always best" are empirically true within certain narrow problem classes
    (natural images, tabular data with mixed types), but are never
    universally true.

    This is not pessimism — it is a call to understand your problem deeply.

**Implication 2: Cross-validation is philosophically justified by NFL.**

    Because no algorithm is a priori better, we cannot choose models
    theoretically — we must evaluate them empirically on the specific
    problem. Cross-validation is the principled way to do this.

**Implication 3: Ensembles work because they cover more inductive biases.**

    A random forest uses many different decision trees with different
    random feature subsets. A stacking ensemble combines classifiers with
    completely different inductive biases. NFL explains WHY this helps:
    different biases are good for different sub-regions of the function
    space — covering more bases reduces the "unpaid debt" of any single bias.

    Diagram 5 — Single Model vs Ensemble under NFL:

    Function Space                   Function Space
    ┌─────────────────────────────┐  ┌──────────────────────────────┐
    │ ████████░░░░░░░░░░░░░░░░░░  │  │ ████████░░░░░░░░░░░░░░░░░░░  │
    │ ░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │ ░░░░░░░░████████░░░░░░░░░░░  │
    │ ░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │ ░░░░░░░░░░░░░░░░████████░░░  │
    └─────────────────────────────┘  └──────────────────────────────┘
    Single algorithm covers           Ensemble covers more of the
    one region well.                  function space jointly.
    Fails everywhere else.            Each member covers a different
    ████ = performs well              region. Total coverage increases.

**Implication 4: Domain knowledge is the "free lunch" that NFL allows.**

    NFL assumes no prior knowledge about the problem distribution.
    The moment you inject domain knowledge — smoothness, periodicity,
    scale invariance, sparsity — you break the uniform prior assumption
    and create genuine superiority for the right algorithm.

    Convolutional networks don't beat fully-connected nets because of
    magic — they beat them because translation invariance and local
    spatial structure are real properties of natural images. The
    inductive bias matches the true data distribution.


──────────────────────────────────────────────────────────────────────────────
### NFL and Hyperparameter Optimisation

The NFL theorem applies recursively to hyperparameter tuning. Consider
the "algorithm" of selecting hyperparameters:

    Diagram 6 — NFL Applied to HPO Algorithms:

    Hyperparameter Optimisation Methods:
        Random Search       — random exploration of HPO space
        Grid Search         — exhaustive enumeration
        Bayesian Optimisation — models the performance landscape
        Hyperband / ASHA    — early stopping of poor configs

    NFL says: averaged over all HPO landscapes, all these methods
    are equal.

    In practice: Bayesian optimisation assumes the performance landscape
    is smooth and correlated. For problems where it is, BO wins clearly.
    For discontinuous, noisy HPO landscapes, random search is competitive.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  This is why AutoML is fundamentally hard — and why it works.       │
    │  AutoML encodes a prior over "which configurations tend to work     │
    │  on this type of problem." That prior is learned from prior tasks   │
    │  (meta-learning). NFL is circumvented by prior task knowledge.      │
    └─────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### NFL and Deep Learning — A Modern Perspective

Deep learning appears to violate the spirit of NFL — deep networks seem
to "just work" on a remarkable range of tasks. How do we reconcile this?

The answer is multi-layered:

1. **Architecture = Inductive Bias**

    The architecture of a neural network encodes its inductive bias:
    - CNNs: local connectivity + weight sharing → translation invariance
    - RNNs/Transformers: sequence structure → temporal/token dependencies
    - GNNs: graph structure → relational reasoning

    These are not neutral. They encode strong assumptions that happen
    to match the structure of the data domains they're applied to.

2. **SGD has implicit regularisation**

    Stochastic gradient descent is biased towards flat minima (Keskar
    et al., 2017). Flat minima generalise better. This is an inductive
    bias built into the OPTIMISER, not just the model.

3. **The Lottery Ticket Hypothesis**

    Large networks contain many smaller subnetworks ("tickets"). Training
    finds the subnetwork whose inductive bias matches the problem. A large
    enough network is not one biased model — it is a hypothesis space
    rich enough to contain the right inductive bias.

4. **Scale and the scaling hypothesis**

    As model scale increases, emergent capabilities appear. This is
    consistent with NFL: the bias of a very large model is so broad that
    it effectively covers the true data distribution for most human-relevant
    tasks (which are not uniformly drawn from all possible functions, but
    concentrated in a narrow region of function space shaped by physics,
    biology, and human cognition).

    ┌────────────────────────────────────────────────────────────────────┐
    │  The "unreasonable effectiveness" of deep learning is not a        │
    │  refutation of NFL. It is evidence that natural data lives in a    │
    │  tiny, structured region of function space — and that deep         │
    │  architectures, with the right inductive biases, cover that region │
    │  remarkably well.                                                  │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Practical Takeaways for ML Practitioners

1. **Understand your problem's structure before choosing an algorithm.**

    Ask: Is the relationship smooth? Is it sparse? Is it hierarchical?
    Are features independent? Is there spatial or temporal structure?
    Your answers determine the right inductive bias — and therefore
    the right algorithm family.

2. **Never trust benchmark results blindly.**

    A benchmark is a specific distribution over problems. A method that
    wins a benchmark has a bias that matches that distribution. It may
    fail completely on a different distribution. Always evaluate on your
    own data.

3. **Multiple algorithms are better than one.**

    Because no single algorithm dominates, ensemble methods — which
    combine diverse inductive biases — are consistently among the
    strongest performers across many problem types.

4. **Domain knowledge is your competitive advantage.**

    NFL guarantees that the algorithm with the right prior for your
    problem will outperform all-purpose methods. Encoding domain
    knowledge (through feature engineering, custom architectures, or
    informed priors) is always worth the effort.

5. **Regularisation is the act of choosing a bias.**

    L1 regularisation encodes "the true function is sparse."
    L2 regularisation encodes "the true function has small weights."
    Dropout encodes "the true function doesn't need all features jointly."
    Every regularisation strategy is a prior — choose it consciously.

    Final Summary:
    ┌──────────────────────────────────────────────────────────────────┐
    │  The NFL theorem does not condemn us to ignorance. It teaches    │
    │  us that knowledge is the source of all advantage in ML.         │
    │                                                                  │
    │  The free lunch is not "a universally optimal algorithm."        │
    │  The free lunch IS "correctly encoding what you know about       │
    │  your problem into the structure of your model."                 │
    │                                                                  │
    │  Domain knowledge + right inductive bias + right evaluation      │
    │                    = the only reliable path to good models.      │
    └──────────────────────────────────────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · NFL on a Finite Function Space": {
        "description": (
            "Exhaustively verify the NFL theorem on a small discrete problem. "
            "Enumerate ALL possible binary functions on 4 inputs, train four "
            "different algorithms (majority vote, nearest neighbour, constant "
            "predictors) on every possible training set, and confirm that their "
            "total errors across all functions are exactly equal."
        ),
        "language": "python",
        "code": '''
import numpy as np
from itertools import product, combinations

# ── Setup: X = {0,1,2,3}, Y = {0,1} ─────────────────────────────────────
X_ALL  = [0, 1, 2, 3]           # all possible inputs
N_X    = len(X_ALL)
N_FUNC = 2 ** N_X               # 2^4 = 16 possible functions
TRAIN_SIZE = 2                  # train on 2 points, test on the remaining 2

print("=" * 65)
print("  NO FREE LUNCH THEOREM — EXHAUSTIVE VERIFICATION")
print(f"  Input space: {X_ALL}")
print(f"  Output space: {{0, 1}}")
print(f"  Total possible functions: {N_FUNC}")
print(f"  Training set size: {TRAIN_SIZE}  |  Test set size: {N_X - TRAIN_SIZE}")
print("=" * 65)
print()

# ── Enumerate all 16 functions as truth tables ────────────────────────────
all_functions = list(product([0, 1], repeat=N_X))
# all_functions[i] is a tuple (f(0), f(1), f(2), f(3))

# ── Define four "algorithms" (prediction strategies) ─────────────────────
def predict_majority(train_x, train_y, test_x):
    """Predict the most common label in training data."""
    majority = 1 if sum(train_y) >= len(train_y) / 2 else 0
    return [majority] * len(test_x)

def predict_nearest(train_x, train_y, test_x):
    """1-NN: predict label of nearest training input (by integer distance)."""
    preds = []
    for tx in test_x:
        dists = [abs(tx - tr) for tr in train_x]
        nn_idx = dists.index(min(dists))
        preds.append(train_y[nn_idx])
    return preds

def predict_always_zero(train_x, train_y, test_x):
    """Always predict 0 regardless of training data."""
    return [0] * len(test_x)

def predict_always_one(train_x, train_y, test_x):
    """Always predict 1 regardless of training data."""
    return [1] * len(test_x)

algorithms = {
    "Majority Vote":   predict_majority,
    "1-NN":            predict_nearest,
    "Always-0":        predict_always_zero,
    "Always-1":        predict_always_one,
}

# ── Enumerate all possible training sets of size TRAIN_SIZE ──────────────
train_subsets = list(combinations(range(N_X), TRAIN_SIZE))
# Each subset is a tuple of indices into X_ALL for the training inputs

# ── Compute total test error for each algorithm across all (f, train_set) ─
total_errors = {name: 0 for name in algorithms}
n_evaluations = 0

for f_tuple in all_functions:
    for train_idx in train_subsets:
        test_idx = [i for i in range(N_X) if i not in train_idx]

        train_x = [X_ALL[i] for i in train_idx]
        train_y = [f_tuple[i] for i in train_idx]
        test_x  = [X_ALL[i] for i in test_idx]
        test_y  = [f_tuple[i] for i in test_idx]

        for name, algo in algorithms.items():
            preds = algo(train_x, train_y, test_x)
            errors = sum(p != t for p, t in zip(preds, test_y))
            total_errors[name] += errors

        n_evaluations += 1

# ── Report ────────────────────────────────────────────────────────────────
print(f"  Evaluated over:")
print(f"    {N_FUNC} functions  ×  {len(train_subsets)} training sets"
      f"  =  {n_evaluations} total (function, training-set) pairs")
print()
print("  Algorithm          Total Test Errors   Errors per Evaluation")
print("  " + "─" * 55)
for name, total in total_errors.items():
    per_eval = total / n_evaluations
    print(f"  {name:<20} {total:>16}   {per_eval:>18.4f}")

print()
errors_list = list(total_errors.values())
all_equal = all(e == errors_list[0] for e in errors_list)
print(f"  All total errors equal? → {'YES ✓ — NFL Theorem Verified!' if all_equal else 'NO (unexpected)'}")
print()
print("  Interpretation:")
print("  - Majority Vote and 1-NN feel 'smart' — they use the training data.")
print("  - Always-0 and Always-1 feel 'dumb' — they ignore the data entirely.")
print("  - Yet all four accumulate EXACTLY the same total error.")
print("  - On functions where Majority Vote wins, Always-1 or Always-0 wins more.")
print("  - The NFL theorem holds: gains on some functions cancel losses on others.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Algorithm Performance Across Problem Types": {
        "description": (
            "Demonstrate NFL empirically on real sklearn classifiers by testing "
            "each on five structurally different synthetic datasets. Show that no "
            "single algorithm wins across all problem types, and compute a "
            "normalised performance matrix to visualise the tradeoffs."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def make_classification(n_samples=200,n_features=2,n_informative=2,n_redundant=0,
                        n_clusters_per_class=1,class_sep=1.0,flip_y=0.0,random_state=None):
    rng=np.random.default_rng(random_state); nc=2
    y=rng.integers(0,nc,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*class_sep*rng.uniform(0.8,1.2,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.2*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

def make_circles(n_samples=200,noise=0.1,factor=0.5,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    Xo=np.c_[np.cos(t),np.sin(t)]; Xi=np.c_[factor*np.cos(t),factor*np.sin(t)]
    X=np.vstack([Xo,Xi])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

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
        n=len(y); fs=n//cv
        splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w0=np.zeros(d+1); lam=1./self.C
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def predict_proba(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            p=np.bincount(nn.astype(int),minlength=2)/self.n_neighbors; out.append(p)
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()

class DecisionTreeClassifier:
    def __init__(self,max_depth=5,random_state=None): self.max_depth=max_depth or 30
    def _gini(self,y):
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
                if l.sum()<2 or r.sum()<2: continue
                g=self._gini(y[l])*l.sum()+self._gini(y[r])*r.sum()
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        cls,cnt=np.unique(y,return_counts=True); pred=cls[cnt.argmax()]
        if depth>=self.max_depth or len(cls)==1: return ("L",pred)
        sp=self._split(X,y)
        if sp is None: return ("L",pred)
        _,f,t=sp; m=X[:,f]<=t
        return ("N",f,t,self._build(X[m],y[m],depth+1),self._build(X[~m],y[~m],depth+1))
    def fit(self,X,y): self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd[0]=="L": return nd[1]
        return self._p1(x,nd[3] if x[nd[1]]<=nd[2] else nd[4])
    def predict(self,X): return np.array([self._p1(x,self._tree) for x in X])
    def predict_proba(self,X):
        p=self.predict(X); out=np.zeros((len(X),2)); out[p==0,0]=1; out[p==1,1]=1; return out
    def score(self,X,y): return (self.predict(X)==y).mean()

class GaussianNB:
    def fit(self,X,y):
        self.classes_=np.unique(y); self._priors=[]; self._means=[]; self._vars=[]
        for c in self.classes_:
            Xc=X[y==c]; self._priors.append(len(Xc)/len(X))
            self._means.append(Xc.mean(0)); self._vars.append(Xc.var(0)+1e-9)
        return self
    def _ll(self,X,i):
        m,v=self._means[i],self._vars[i]
        return -0.5*np.sum(np.log(2*np.pi*v)+(X-m)**2/v,1)+np.log(self._priors[i])
    def predict_proba(self,X):
        lp=np.c_[[self._ll(X,i) for i in range(len(self.classes_))]].T
        lp-=lp.max(1,keepdims=True); p=np.exp(lp); return p/p.sum(1,keepdims=True)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

class SVC:
    def __init__(self,kernel="rbf",C=1.0,gamma="scale",probability=True):
        self.C=C; self.gamma=gamma
    def _rbf(self,X1,X2):
        d2=((X1[:,None]-X2[None])**2).sum(2); return np.exp(-self._g*d2)
    def fit(self,X,y):
        self._X=X.copy(); self._classes=np.unique(y)
        yb=(2*(y==self._classes[1])-1).astype(float)
        n=len(X); self._g=(1/(X.shape[1]*X.var()) if self.gamma=="scale" else float(self.gamma))
        K=self._rbf(X,X); Q=np.outer(yb,yb)*K
        def obj(a): return 0.5*a@Q@a-a.sum()
        def jac(a): return Q@a-1
        res=_sp_opt.minimize(obj,np.zeros(n),method="L-BFGS-B",jac=jac,
                             bounds=[(0,self.C)]*n,options={"maxiter":150})
        a=np.clip(res.x,0,self.C); sv=a>1e-5
        self._sv_X=X[sv]; self._sv_y=yb[sv]; self._sv_a=a[sv]
        w=self._sv_a*self._sv_y
        self._b=np.mean(self._sv_y-w@self._rbf(self._sv_X,X[sv]))
        return self
    def _dec(self,X): return (self._sv_a*self._sv_y)@self._rbf(self._sv_X,X)+self._b
    def predict(self,X): return self._classes[(self._dec(X)>=0).astype(int)]
    def predict_proba(self,X): d=self._dec(X); p=_sig(d); return np.c_[1-p,p]
    def score(self,X,y): return (self.predict(X)==y).mean()

class RandomForestClassifier:
    def __init__(self,n_estimators=15,max_depth=None,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self._trees=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(np.sqrt(X.shape[1]))); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeClassifier(max_depth=self.max_depth or 8)
            t.fit(X[np.ix_(idx,fi)],y[idx]); self._trees.append((t,fi))
        return self
    def predict_proba(self,X):
        votes=np.zeros((len(X),2))
        for t,fi in self._trees: votes+=t.predict_proba(X[:,fi])
        return votes/len(self._trees)
    def predict(self,X): return self.predict_proba(X).argmax(1)
    def score(self,X,y): return (self.predict(X)==y).mean()

np.random.seed(42)

# ── Define five structurally distinct classification problems ─────────────
datasets = {}

# 1. Linearly separable (wide margin)
X, y = make_classification(n_samples=400, n_features=2, n_redundant=0,
                            n_informative=2, n_clusters_per_class=1,
                            class_sep=2.5, random_state=1)
datasets["Linear (separable)"] = (X, y)

# 2. Two moons — non-linear, smooth boundary
X, y = make_moons(n_samples=400, noise=0.2, random_state=2)
datasets["Two Moons (non-linear)"] = (X, y)

# 3. Concentric circles — highly non-linear
X, y = make_circles(n_samples=400, noise=0.1, factor=0.4, random_state=3)
datasets["Circles (radial)"] = (X, y)

# 4. XOR pattern — decision boundary is axis-aligned rectangles
X = np.random.uniform(-1, 1, (400, 2))
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
datasets["XOR (rectangular)"] = (X, y)

# 5. High-noise, many features — hard generalisation
X, y = make_classification(n_samples=400, n_features=20, n_informative=3,
                            n_redundant=10, flip_y=0.15, random_state=5)
datasets["High-Noise (20 features)"] = (X, y)

# ── Define classifiers ────────────────────────────────────────────────────
classifiers = {
    "Logistic Reg.":   LogisticRegression(C=1.0, max_iter=500),
    "KNN (k=5)":       KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":   DecisionTreeClassifier(max_depth=5, random_state=0),
    "Naïve Bayes":     GaussianNB(),
    "SVM (RBF)":       SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Random Forest":   RandomForestClassifier(n_estimators=15, random_state=0),
}

# ── Cross-validated accuracy for each (classifier, dataset) pair ─────────
print("=" * 72)
print("  NO FREE LUNCH — ALGORITHM PERFORMANCE ACROSS PROBLEM TYPES")
print("  (5-fold cross-validated accuracy, mean ± std)")
print("=" * 72)
print()

# Header
header = f"  {'Dataset':<26}"
for clf_name in classifiers:
    header += f"  {clf_name[:12]:>12}"
print(header)
print("  " + "─" * 100)

results = {}   # results[dataset_name][clf_name] = mean_acc

for ds_name, (X, y) in datasets.items():
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    row = {}
    row_str = f"  {ds_name:<26}"
    for clf_name, clf in classifiers.items():
        scores = cross_val_score(clf, X_s, y, cv=cv, scoring="accuracy")
        row[clf_name] = scores.mean() * 100
        row_str += f"  {scores.mean()*100:>10.1f}%"
    results[ds_name] = row
    print(row_str)

# ── Which algorithm wins each dataset? ───────────────────────────────────
print()
print("  WINNER PER DATASET:")
print("  " + "─" * 55)
wins = {clf: 0 for clf in classifiers}
for ds_name, row in results.items():
    winner = max(row, key=row.get)
    wins[winner] += 1
    print(f"  {ds_name:<28} → {winner}  ({row[winner]:.1f}%)")

print()
print("  TOTAL WINS:")
print("  " + "─" * 35)
for clf_name, w in wins.items():
    print(f"  {clf_name:<20} {w} win{'s' if w != 1 else ''}")

print()
print("  → No single algorithm wins across all 5 problem types.")
print("  → Linear Regression wins on linear problems.")
print("  → KNN / SVM win on non-linear smooth boundaries.")
print("  → Decision Trees win on XOR (axis-aligned rectangles).")
print("  → This IS the No Free Lunch theorem in action.")

# ── Heatmap visualisation ─────────────────────────────────────────────────
clf_names = list(classifiers.keys())
ds_names  = list(datasets.keys())

matrix = np.array([[results[ds][clf] for clf in clf_names] for ds in ds_names])

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")

ax.set_xticks(range(len(clf_names)))
ax.set_xticklabels(clf_names, fontsize=10, rotation=20, ha="right")
ax.set_yticks(range(len(ds_names)))
ax.set_yticklabels(ds_names, fontsize=10)

for i in range(len(ds_names)):
    for j in range(len(clf_names)):
        best = clf_names[np.argmax(matrix[i])]
        weight = "bold" if clf_names[j] == best else "normal"
        text = ax.text(j, i, f"{matrix[i, j]:.1f}%",
                       ha="center", va="center", fontsize=9,
                       fontweight=weight,
                       color="black" if matrix[i, j] > 65 else "white")

plt.colorbar(im, ax=ax, label="CV Accuracy (%)")
ax.set_title(
    "No Free Lunch: Classifier Performance Across Structurally Different Problems\\n"
    "(Bold = winner for each dataset. No single algorithm dominates.)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("nfl_performance_matrix.png", dpi=120)
print()
print("  Plot saved → nfl_performance_matrix.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Inductive Bias Visualised": {
        "description": (
            "Visualise the decision boundaries of six classifiers on three "
            "datasets where their inductive biases match or clash with the "
            "true data structure. A 3×6 grid shows exactly where each "
            "algorithm's assumptions help or hurt it."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy as _copy
import scipy.optimize as _sp_opt

def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class StandardScaler:
    def fit(self,X): self.mean_=X.mean(0); self.scale_=X.std(0)+1e-8; return self
    def transform(self,X): return (X-self.mean_)/self.scale_
    def fit_transform(self,X,y=None): return self.fit(X).transform(X)

def make_classification(n_samples=200,n_features=2,n_informative=2,n_redundant=0,
                        n_clusters_per_class=1,class_sep=1.0,flip_y=0.0,random_state=None):
    rng=np.random.default_rng(random_state); nc=2
    y=rng.integers(0,nc,n_samples)
    Xi=rng.standard_normal((n_samples,n_informative))
    Xi+=(2*y-1)[:,None]*class_sep*rng.uniform(0.8,1.2,n_informative)
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.2*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=1-y[fm]
    return X,y.astype(int)

def make_moons(n_samples=200,noise=0.1,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,np.pi,n)
    X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[1-np.cos(t),-np.sin(t)+0.5]])
    X+=rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

def make_circles(n_samples=200,noise=0.1,factor=0.5,random_state=None):
    rng=np.random.default_rng(random_state); n=n_samples//2
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    Xo=np.c_[np.cos(t),np.sin(t)]; Xi=np.c_[factor*np.cos(t),factor*np.sin(t)]
    X=np.vstack([Xo,Xi])+rng.normal(0,noise,(n_samples,2))
    return X,np.hstack([np.zeros(n),np.ones(n)]).astype(int)

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
        n=len(y); fs=n//cv
        splits=[(np.concatenate([np.arange(0,i*fs),np.arange((i+1)*fs,n)]),np.arange(i*fs,(i+1)*fs)) for i in range(cv)]
    scores=[]
    for tr,te in splits:
        est=_copy.deepcopy(estimator); est.fit(X[tr],y[tr])
        scores.append((est.predict(X[te])==y[te]).mean())
    return np.array(scores)

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        n,d=X.shape; w0=np.zeros(d+1); lam=1./self.C
        def fg(w):
            p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
            loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
            e=p-y; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
            return loss,g
        res=_sp_opt.minimize(fg,w0,method="L-BFGS-B",jac=True,options={"maxiter":self.max_iter})
        self.intercept_=np.array([res.x[0]]); self.coef_=res.x[1:].reshape(1,-1); return self
    def predict_proba(self,X):
        p=_sig(X@self.coef_.ravel()+self.intercept_[0]); return np.c_[1-p,p]
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
    def score(self,X,y): return (self.predict(X)==y).mean()

class KNeighborsClassifier:
    def __init__(self,n_neighbors=5): self.n_neighbors=n_neighbors
    def fit(self,X,y): self._X=X.copy(); self._y=y.copy(); return self
    def predict(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            out.append(np.bincount(nn.astype(int)).argmax())
        return np.array(out)
    def predict_proba(self,X):
        out=[]
        for xi in X:
            d=np.sqrt(((self._X-xi)**2).sum(1))
            nn=self._y[np.argsort(d)[:self.n_neighbors]]
            p=np.bincount(nn.astype(int),minlength=2)/self.n_neighbors; out.append(p)
        return np.array(out)
    def score(self,X,y): return (self.predict(X)==y).mean()

class DecisionTreeClassifier:
    def __init__(self,max_depth=5,random_state=None): self.max_depth=max_depth or 30
    def _gini(self,y):
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
                if l.sum()<2 or r.sum()<2: continue
                g=self._gini(y[l])*l.sum()+self._gini(y[r])*r.sum()
                if best is None or g<best[0]: best=(g,f,t)
        return best
    def _build(self,X,y,depth):
        cls,cnt=np.unique(y,return_counts=True); pred=cls[cnt.argmax()]
        if depth>=self.max_depth or len(cls)==1: return ("L",pred)
        sp=self._split(X,y)
        if sp is None: return ("L",pred)
        _,f,t=sp; m=X[:,f]<=t
        return ("N",f,t,self._build(X[m],y[m],depth+1),self._build(X[~m],y[~m],depth+1))
    def fit(self,X,y): self._tree=self._build(X,y,0); return self
    def _p1(self,x,nd):
        if nd[0]=="L": return nd[1]
        return self._p1(x,nd[3] if x[nd[1]]<=nd[2] else nd[4])
    def predict(self,X): return np.array([self._p1(x,self._tree) for x in X])
    def predict_proba(self,X):
        p=self.predict(X); out=np.zeros((len(X),2)); out[p==0,0]=1; out[p==1,1]=1; return out
    def score(self,X,y): return (self.predict(X)==y).mean()

class GaussianNB:
    def fit(self,X,y):
        self.classes_=np.unique(y); self._priors=[]; self._means=[]; self._vars=[]
        for c in self.classes_:
            Xc=X[y==c]; self._priors.append(len(Xc)/len(X))
            self._means.append(Xc.mean(0)); self._vars.append(Xc.var(0)+1e-9)
        return self
    def _ll(self,X,i):
        m,v=self._means[i],self._vars[i]
        return -0.5*np.sum(np.log(2*np.pi*v)+(X-m)**2/v,1)+np.log(self._priors[i])
    def predict_proba(self,X):
        lp=np.c_[[self._ll(X,i) for i in range(len(self.classes_))]].T
        lp-=lp.max(1,keepdims=True); p=np.exp(lp); return p/p.sum(1,keepdims=True)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

class SVC:
    def __init__(self,kernel="rbf",C=1.0,gamma="scale",probability=True):
        self.C=C; self.gamma=gamma
    def _rbf(self,X1,X2):
        d2=((X1[:,None]-X2[None])**2).sum(2); return np.exp(-self._g*d2)
    def fit(self,X,y):
        self._X=X.copy(); self._classes=np.unique(y)
        yb=(2*(y==self._classes[1])-1).astype(float)
        n=len(X); self._g=(1/(X.shape[1]*X.var()) if self.gamma=="scale" else float(self.gamma))
        K=self._rbf(X,X); Q=np.outer(yb,yb)*K
        def obj(a): return 0.5*a@Q@a-a.sum()
        def jac(a): return Q@a-1
        res=_sp_opt.minimize(obj,np.zeros(n),method="L-BFGS-B",jac=jac,
                             bounds=[(0,self.C)]*n,options={"maxiter":150})
        a=np.clip(res.x,0,self.C); sv=a>1e-5
        self._sv_X=X[sv]; self._sv_y=yb[sv]; self._sv_a=a[sv]
        w=self._sv_a*self._sv_y
        self._b=np.mean(self._sv_y-w@self._rbf(self._sv_X,X[sv]))
        return self
    def _dec(self,X): return (self._sv_a*self._sv_y)@self._rbf(self._sv_X,X)+self._b
    def predict(self,X): return self._classes[(self._dec(X)>=0).astype(int)]
    def predict_proba(self,X): d=self._dec(X); p=_sig(d); return np.c_[1-p,p]
    def score(self,X,y): return (self.predict(X)==y).mean()

class RandomForestClassifier:
    def __init__(self,n_estimators=15,max_depth=None,random_state=None):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.random_state=random_state
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state); n=len(X); self._trees=[]
        for i in range(self.n_estimators):
            idx=rng.choice(n,n,replace=True)
            nf=max(1,int(np.sqrt(X.shape[1]))); fi=rng.choice(X.shape[1],nf,replace=False)
            t=DecisionTreeClassifier(max_depth=self.max_depth or 8)
            t.fit(X[np.ix_(idx,fi)],y[idx]); self._trees.append((t,fi))
        return self
    def predict_proba(self,X):
        votes=np.zeros((len(X),2))
        for t,fi in self._trees: votes+=t.predict_proba(X[:,fi])
        return votes/len(self._trees)
    def predict(self,X): return self.predict_proba(X).argmax(1)
    def score(self,X,y): return (self.predict(X)==y).mean()

np.random.seed(42)

# ── Datasets ──────────────────────────────────────────────────────────────
datasets_plot = [
    ("Linear", *make_classification(n_samples=300, n_features=2,
                                     n_redundant=0, n_informative=2,
                                     n_clusters_per_class=1, class_sep=1.8,
                                     random_state=1)),
    ("Two Moons", *make_moons(n_samples=300, noise=0.25, random_state=2)),
    ("Circles",   *make_circles(n_samples=300, noise=0.12,
                                 factor=0.4, random_state=3)),
]

classifiers_plot = [
    ("Logistic Reg.\\n(linear boundary)",    LogisticRegression(C=1.0, max_iter=300)),
    ("KNN k=5\\n(local smoothness)",         KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree\\n(axis-aligned)",       DecisionTreeClassifier(max_depth=4, random_state=0)),
    ("Naïve Bayes\\n(feature independence)", GaussianNB()),
    ("SVM RBF\\n(smooth kernel)",            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
    ("Random Forest\\n(ensemble of trees)",  RandomForestClassifier(n_estimators=15, random_state=0)),
]

n_ds   = len(datasets_plot)
n_clf  = len(classifiers_plot)
h = 0.15   # mesh resolution (reduced for performance)

cmap_bg  = ListedColormap(["#ffeaea", "#eaeaff"])
cmap_pts = ListedColormap(["#cc0000", "#0000cc"])

fig, axes = plt.subplots(n_ds, n_clf, figsize=(22, 11))
fig.suptitle(
    "Inductive Bias Visualised: Decision Boundaries Across Problem Types\\n"
    "(Green border = best accuracy for this dataset; columns show inductive bias)",
    fontsize=12, fontweight="bold"
)

for row, (ds_name, X, y) in enumerate(datasets_plot):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.3,
                                               random_state=0, stratify=y)

    # Find best classifier accuracy for this dataset (to highlight winner)
    accs = []
    for _, clf in classifiers_plot:
        clf.fit(X_tr, y_tr)
        accs.append(clf.score(X_te, y_te))
    best_acc = max(accs)

    x_min, x_max = X_s[:, 0].min() - .5, X_s[:, 0].max() + .5
    y_min, y_max = X_s[:, 1].min() - .5, X_s[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    for col, (clf_name, clf) in enumerate(classifiers_plot):
        ax = axes[row, col]
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te) * 100

        # Decision boundary
        try:
            if hasattr(clf, "predict_proba"):
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            else:
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg, levels=20)
            ax.contour(xx, yy, Z > 0.5, colors=["gray"], linewidths=[0.8],
                       linestyles=["--"])
        except Exception:
            pass

        ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap=cmap_pts,
                   s=15, alpha=0.8, edgecolors="none")

        border_color = "green" if abs(acc / 100 - best_acc) < 0.001 else "lightgray"
        border_width = 2.5 if border_color == "green" else 0.8
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{acc:.0f}%", fontsize=9,
                     color="green" if border_color == "green" else "black",
                     fontweight="bold" if border_color == "green" else "normal")

        if row == 0:
            ax.set_xlabel(clf_name, fontsize=8.5, labelpad=3)
            ax.xaxis.set_label_position("top")
        if col == 0:
            ax.set_ylabel(ds_name, fontsize=10, fontweight="bold", labelpad=5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("inductive_bias_boundaries.png", dpi=110)

print("=" * 65)
print("  INDUCTIVE BIAS — DECISION BOUNDARIES")
print("=" * 65)
print()
print("  Plot saved → inductive_bias_boundaries.png")
print()
print("  Key observations:")
print("  - Logistic Regression excels on the linear dataset.")
print("    Its linear boundary is a perfect match for the problem structure.")
print("    It fails on circles/moons — hard linear boundary cannot curve.")
print()
print("  - KNN and SVM (RBF) capture the curved moons/circles boundaries")
print("    because their implicit bias is local smoothness.")
print()
print("  - Decision Tree draws axis-aligned rectangles. This fits XOR-style")
print("    data perfectly but produces jagged boundaries on smooth problems.")
print()
print("  - Naïve Bayes assumes feature independence. For linearly separable")
print("    Gaussians this is fine. For circles it produces ellipsoidal")
print("    boundaries that cannot separate the rings.")
print()
print("  - Random Forest (ensemble of trees) is the most robust, but")
print("    still fails on problems that require smooth curved boundaries.")
print()
print("  → The inductive bias that matches the true data structure WINS.")
print("  → No single bias wins everywhere. This is the NFL theorem.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · NFL in Optimisation — Random Search vs Gradient Descent": {
        "description": (
            "Compare random search and gradient descent on three structurally "
            "different 1D objective functions: smooth convex, multimodal, and "
            "discontinuous. Confirm that the algorithm which wins on smooth "
            "functions loses on discontinuous ones — and compute aggregate "
            "performance to illustrate the NFL balance."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Define three structurally different objective functions ───────────────
def f_smooth(x):
    """Smooth convex bowl. Gradient descent is ideal."""
    return (x - 1.5) ** 2 + 0.5

def f_multimodal(x):
    """Many local minima. GD gets stuck; random search can find global min."""
    return np.sin(3 * x) * np.exp(-0.1 * x**2) + 0.3 * x**2

def f_discontinuous(x):
    """Piecewise flat with spikes. Gradient carries no information."""
    base = np.floor(x * 3) * 0.2
    spikes = np.where(np.abs(x - np.round(x * 2) / 2) < 0.08,
                      -1.5 + (x - 1.5)**2 * 0.3, 0)
    return base + spikes + 0.1

funcs = {
    "Smooth Convex\\n(GD ideal)":          f_smooth,
    "Multimodal\\n(GD gets stuck)":        f_multimodal,
    "Discontinuous\\n(GD useless)":        f_discontinuous,
}

x_range = np.linspace(-3, 3, 2000)
N_RUNS  = 200    # independent runs per algorithm per function

# ── Gradient Descent ──────────────────────────────────────────────────────
def gradient_descent(f, x_range, lr=0.05, n_steps=80):
    """Simple numerical gradient descent starting from a random point."""
    x = np.random.uniform(x_range[0], x_range[-1])
    eps = 1e-5
    for _ in range(n_steps):
        grad = (f(x + eps) - f(x - eps)) / (2 * eps)
        x = x - lr * grad
        x = np.clip(x, x_range[0], x_range[-1])
    return f(x)

# ── Random Search ─────────────────────────────────────────────────────────
def random_search(f, x_range, n_evals=80):
    """Evaluate function at n_evals random points; return minimum found."""
    xs = np.random.uniform(x_range[0], x_range[-1], n_evals)
    return min(f(xi) for xi in xs)

print("=" * 65)
print("  NFL IN OPTIMISATION: GD vs RANDOM SEARCH")
print(f"  {N_RUNS} independent runs per (algorithm, function) pair")
print(f"  Budget: 80 function evaluations each")
print("=" * 65)
print()

results = {}
for f_name, f in funcs.items():
    true_min = f(x_range).min()
    gd_gaps, rs_gaps = [], []

    for _ in range(N_RUNS):
        gd_val = gradient_descent(f, x_range)
        rs_val = random_search(f, x_range)
        gd_gaps.append(gd_val - true_min)
        rs_gaps.append(rs_val - true_min)

    results[f_name] = {
        "GD_gaps": np.array(gd_gaps),
        "RS_gaps": np.array(rs_gaps),
        "true_min": true_min,
    }
    label = f_name.replace("\\n", " ")
    print(f"  {label}")
    print(f"    True minimum:            {true_min:.4f}")
    print(f"    GD   mean gap to min:    {np.mean(gd_gaps):.4f}  "
          f"(std={np.std(gd_gaps):.4f})")
    print(f"    RS   mean gap to min:    {np.mean(rs_gaps):.4f}  "
          f"(std={np.std(rs_gaps):.4f})")
    winner = "GD" if np.mean(gd_gaps) < np.mean(rs_gaps) else "RS"
    print(f"    Winner: {winner}")
    print()

# ── Total aggregate performance ───────────────────────────────────────────
total_gd = sum(r["GD_gaps"].mean() for r in results.values())
total_rs = sum(r["RS_gaps"].mean() for r in results.values())
print("  AGGREGATE PERFORMANCE (sum of mean gaps across all 3 functions):")
print(f"    GD total: {total_gd:.4f}")
print(f"    RS total: {total_rs:.4f}")
print()
print("  → GD wins on smooth function, RS wins on discontinuous/multimodal.")
print("  → Aggregate scores are close — gains on one type cancel on another.")
print("  → This is the optimisation form of the No Free Lunch theorem.")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    "NFL in Optimisation: GD vs Random Search Across Function Types\\n"
    "(GD excels on smooth; fails on discontinuous — total performance is balanced)",
    fontsize=12, fontweight="bold"
)

colours = {"Smooth Convex\\n(GD ideal)":     ("steelblue", "tomato"),
           "Multimodal\\n(GD gets stuck)":   ("steelblue", "tomato"),
           "Discontinuous\\n(GD useless)":   ("steelblue", "tomato")}

for col, (f_name, f) in enumerate(funcs.items()):
    res = results[f_name]
    y_vals = f(x_range)

    # Top row: function shape with GD and RS trajectories
    ax_top = axes[0, col]
    ax_top.plot(x_range, y_vals, "k-", lw=2, label="Objective f(x)")
    ax_top.axhline(res["true_min"], color="gray", ls="--", lw=1,
                   label=f"True min={res['true_min']:.2f}")
    ax_top.set_title(f_name.replace("\\n", " — "), fontsize=10, fontweight="bold")
    ax_top.set_xlabel("x")
    ax_top.set_ylabel("f(x)")
    ax_top.legend(fontsize=8)
    ax_top.grid(alpha=0.3)

    # Annotate winner
    gd_mean = res["GD_gaps"].mean()
    rs_mean = res["RS_gaps"].mean()
    winner_str = f"Winner: {'GD' if gd_mean < rs_mean else 'RS'}"
    ax_top.text(0.98, 0.95, winner_str, transform=ax_top.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold",
                color="green",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="green"))

    # Bottom row: distribution of gaps
    ax_bot = axes[1, col]
    ax_bot.hist(res["GD_gaps"], bins=30, alpha=0.6, color="steelblue",
                label=f"GD  (mean={gd_mean:.3f})", density=True)
    ax_bot.hist(res["RS_gaps"], bins=30, alpha=0.6, color="tomato",
                label=f"RS  (mean={rs_mean:.3f})", density=True)
    ax_bot.axvline(gd_mean, color="steelblue", lw=2, ls="--")
    ax_bot.axvline(rs_mean, color="tomato",    lw=2, ls="--")
    ax_bot.set_title(f"Gap to True Minimum Distribution\\n({N_RUNS} runs)",
                     fontsize=9)
    ax_bot.set_xlabel("f(x*) − f(true min)")
    ax_bot.set_ylabel("Density")
    ax_bot.legend(fontsize=8)
    ax_bot.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("nfl_optimisation.png", dpi=110)
print()
print("  Plot saved → nfl_optimisation.png")
print()
print("  Key Takeaways:")
print("  - GD dominates on smooth convex functions (gradient is informative)")
print("  - RS is competitive or superior on multimodal & discontinuous functions")
print("  - Total aggregated performance is balanced — NFL in action")
print("  - Knowing your function's structure determines which algorithm to use")
print("  - This is why hyperparameter optimisation methods must be matched")
print("    to the smoothness of the loss landscape they're optimising over")
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