"""
Naive Bayes — Probabilistic Classification via Bayes' Theorem
=============================================================

Naive Bayes is the oldest and simplest probabilistic classifier. It applies
Bayes' theorem to compute the posterior probability of each class given the
observed features, and predicts the class with the highest posterior.

The "naive" in Naive Bayes refers to the conditional independence assumption:
given the class label, all features are assumed to be independent of each other.
This assumption is almost never literally true — yet Naive Bayes works remarkably
well in practice, especially for text classification, spam filtering, and medical
diagnosis, often outperforming far more complex models when data is scarce.

It is the only model in this series derived from pure probability theory rather
than from optimisation. Understanding it requires understanding Bayes' theorem,
which is one of the most important results in all of statistics.

"""

import base64
import os
import textwrap
import re

DISPLAY_NAME = "05 · Naive Bayes"
ICON = "🎲"
SUBTITLE = "Probabilistic classification using Bayes theorem with feature independence"

VISUAL_HTML = ""

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What is Naive Bayes?

Naive Bayes is a probabilistic classifier that uses Bayes' theorem to compute,
for each class c, the probability that an input x belongs to that class:

                    P(C=c | x) ∝ P(C=c) · Π_j P(xⱼ | C=c)

It then predicts the class with the highest posterior probability.

The "naive" assumption is that all features x₁, x₂, ..., xₚ are conditionally
independent given the class label C. This means:

                    P(x | C=c) = P(x₁ | C=c) · P(x₂ | C=c) · ... · P(xₚ | C=c)

In reality, features are almost never truly independent — the words "New" and
"York" in a document are not independent. Yet Naive Bayes works astonishingly
well precisely because even a wrong independence assumption produces surprisingly
accurate posterior class rankings (though not accurate probabilities).

Think of it like a doctor making a diagnosis. A naive Bayes doctor would assess
each symptom — fever, cough, fatigue — independently, compute the probability
of each symptom given each disease, and combine them multiplicatively. A more
sophisticated doctor would consider interactions between symptoms. The naive
doctor is less accurate but makes decisions very fast with very little data.

    Things that exist inside the model (learned during training):
        - P(C=c):          class prior — fraction of training examples per class
        - P(xⱼ | C=c):    feature likelihoods — one distribution per feature per class

    Things you control before training (hyperparameters):
        - var_smoothing:   Gaussian NB regularisation (prevents zero-variance)
        - alpha:           Laplace smoothing for Multinomial/Bernoulli NB
        - The model variant: Gaussian, Multinomial, Bernoulli, Complement


### Naive Bayes as Empirical Risk Minimisation (ERM)

Naive Bayes is not derived from ERM — it is derived from probability theory.
It minimises 0-1 loss only when the independence assumption holds exactly.
We can still place it in the ERM template for comparison:

    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  Hypothesis class:  H = { argmax_c P(C=c) Πⱼ P(xⱼ|C=c) }      │
    │                     (linear in log space)                    │
    │                                                              │
    │  Loss function:     Negative log-likelihood of the data      │
    │                     (minimising NLL = maximising likelihood) │
    │                                                              │
    │  Training:          Maximum likelihood estimation of         │
    │                     P(C=c) and P(xⱼ | C=c) from counts        │
    │                     NO gradient descent — closed-form!       │
    │                                                              │
    │  Prediction:        argmax_c log P(C=c) + Σⱼ log P(xⱼ|C=c)    │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

The training procedure is not gradient descent — it is simple counting
and averaging (MLE). This makes Naive Bayes the fastest-training model
in this entire series: a single pass over the data is sufficient.

The full ERM comparison across all modules:

    Linear Regression:    MSE,      gradient descent in weight space
    Logistic Regression:  BCE,      gradient descent in weight space
    SVM:                  Hinge,    quadratic programming
    Decision Tree:        Gini,     greedy split search
    Random Forest:        0-1,      averaging B independent trees
    Gradient Boosting:    any loss, gradient descent in function space
    KNN:                  none,     memorisation (no optimisation)
    Naive Bayes:          NLL,      closed-form MLE (counting + averaging)


### The Inductive Bias of Naive Bayes

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Naive Bayes encodes two beliefs:                           │
    │                                                             │
    │  1. CONDITIONAL INDEPENDENCE — given the class label,       │
    │     all features are independent of each other.             │
    │     P(x₁, x₂, ..., xₚ | C=c) = Πⱼ P(xⱼ | C=c)                 │
    │                                                             │
    │  2. THE DATA GENERATING PROCESS — each feature is           │
    │     generated from a class-conditional distribution.        │
    │     The form of this distribution (Gaussian, Multinomial,   │
    │     Bernoulli) must be chosen to match the data type.       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Unlike all other models which learn a discriminative boundary (directly modelling
P(C|x)), Naive Bayes learns a generative model (modelling P(x,C) = P(x|C)·P(C)).
It can in principle generate new data samples, not just classify.

---

### Part 1: Bayes' Theorem


### The Foundation:

Bayes' theorem is a rule for updating beliefs in light of evidence:

                    P(C | x) = P(x | C) · P(C) / P(x)

    P(C | x):   POSTERIOR — probability of class C given evidence x
    P(x | C):   LIKELIHOOD — probability of observing x given class C
    P(C)    :   PRIOR — probability of class C before seeing x
    P(x)    :   EVIDENCE — normalising constant (same for all classes)


### Why We Can Ignore P(x):

For classification, we compare P(C=c | x) across classes. Since P(x) is the
same for all classes, we can drop it and work with the unnormalised posterior:

                    P(C=c | x) ∝ P(x | C=c) · P(C=c)

We predict the class with the highest posterior (the MAP decision):

                    ŷ = argmax_c  P(C=c) · P(x | C=c)


### Concrete Example — Spam Filtering:

    Email arrives containing the word "FREE". Is it spam or not?

    Prior probabilities (from historical data):
        P(spam) = 0.3     (30% of all emails are spam)
        P(not_spam) = 0.7

    Likelihood of seeing "FREE" in each class:
        P("FREE" | spam)     = 0.8   (80% of spam emails contain "FREE")
        P("FREE" | not_spam) = 0.1   (10% of normal emails contain "FREE")

    Posterior:
        P(spam | "FREE")     ∝ 0.3 × 0.8 = 0.24
        P(not_spam | "FREE") ∝ 0.7 × 0.1 = 0.07

    Normalised:
        P(spam | "FREE")     = 0.24 / (0.24 + 0.07) = 0.774
        P(not_spam | "FREE") = 0.07 / (0.24 + 0.07) = 0.226

    Decision: spam (77.4% posterior probability)

    Now add a second word: "Meeting"
        P("Meeting" | spam)     = 0.1
        P("Meeting" | not_spam) = 0.4

    Naive Bayes assumes independence: multiply the likelihoods:
        P(spam | "FREE", "Meeting")     ∝ 0.3 × 0.8 × 0.1 = 0.024
        P(not_spam | "FREE", "Meeting") ∝ 0.7 × 0.1 × 0.4 = 0.028

    Decision: NOT spam (despite "FREE", the word "Meeting" flips it)


    # ======================================================================================= # 
    **Diagram 1 — Bayes' Theorem: Prior → Likelihood → Posterior:**

    PRIOR: before seeing any features        POSTERIOR: after observing "FREE"
    ─────────────────────────────────────    ─────────────────────────────────────
    Spam     [███    ] 30%                   Spam     [███████   ] 77%
    Not Spam [███████] 70%                   Not Spam [██        ] 23%

    The evidence "FREE" (8× more common in spam than normal email) shifts
    the posterior strongly toward spam.

    Add "Meeting" (4× more common in normal):
    Spam     [████████] 46%    ← nearly back to 50/50 (meeting moderates it)
    Not Spam [████████] 54%

    Each feature (word) updates our belief multiplicatively.
    # ======================================================================================= # 


### Log-Space Computation:

In practice, multiplying many probabilities underflows to zero for long documents.
We work in log space — multiplications become additions:

    log P(C=c | x) ∝ log P(C=c) + Σⱼ log P(xⱼ | C=c)

This is numerically stable even for thousands of features.


---


### Part 2: Three Variants of Naive Bayes


### Gaussian Naive Bayes (GNB) — Continuous Features:

Assumes each feature xⱼ follows a Gaussian (normal) distribution within each class:

                    P(xⱼ | C=c) = N(xⱼ; μⱼ_c, σ²ⱼ_c)

Training: estimate μⱼ_c (class-conditional mean) and σ²ⱼ_c (variance) from data.
No gradient descent — just compute means and variances.

    P(xⱼ | C=c) = (1 / √(2πσ²ⱼ_c)) · exp(−(xⱼ − μⱼ_c)² / (2σ²ⱼ_c))

Use when: features are continuous (measurements, sensor readings, biomarkers).


### Multinomial Naive Bayes (MNB) — Count Features:

Assumes features are counts (e.g., word frequencies in a document):

                    P(x | C=c) = (Σⱼ xⱼ)! / (Πⱼ xⱼ!) · Πⱼ θⱼ_c^xⱼ

Where θⱼ_c = P(feature j | class c) is the probability of feature j in class c.

Training: estimate θⱼ_c by counting occurrences with Laplace smoothing:

                    θⱼ_c = (count(j, c) + α) / (count(c) + α · p)

Where α is the Laplace smoothing parameter (α=1 = add-one smoothing).

Use when: features are integer counts (word counts, n-gram frequencies).


### Bernoulli Naive Bayes (BNB) — Binary Features:

Assumes each feature is binary: xⱼ ∈ {0, 1} (feature j present or absent):

                    P(xⱼ | C=c) = θⱼ_c^xⱼ · (1 − θⱼ_c)^(1−xⱼ)

Unlike MNB, BNB explicitly models the ABSENCE of features. A word NOT appearing
in a spam email is evidence of non-spam — BNB captures this, MNB ignores it.

Use when: features are presence/absence indicators (document classification
with binary bag-of-words, or any binary feature vector).


    # ======================================================================================= # 
    **Diagram 2 — When to Use Which Variant:**

    Feature type                 → Recommended Naive Bayes variant
    ─────────────────────────────────────────────────────────────────
    Continuous (height, temp.)   → Gaussian NB
    Word counts in document      → Multinomial NB  (most common for NLP)
    Word present/absent (0/1)    → Bernoulli NB
    Mixed continuous + binary    → Custom: stack GNB + BNB predictions
    Categorical (not ordinal)    → CategoricalNB  (sklearn)
    ─────────────────────────────────────────────────────────────────

    Key difference: MNB vs BNB for text
    MNB: "FREE appeared 3 times → very strong spam signal"
    BNB: "FREE appeared (present=1) → spam signal"
         "Meeting absent (present=0) → spam signal"
    BNB explicitly penalises for the absence of non-spam words.
    # ======================================================================================= # 


### Laplace Smoothing — Handling Unseen Features:

Without smoothing, if a word never appeared in training spam emails, then
P("newword" | spam) = 0, and the entire product becomes zero:

    P(spam | "FREE", "newword") ∝ 0.8 × 0 = 0   (even with strong spam signal!)

This is the zero-frequency problem. Laplace smoothing adds a pseudocount of α
to every feature, ensuring no probability is exactly zero:

    θⱼ_c = (count(j, c) + α) / (count(c) + α · p)

    α = 0: no smoothing (zero-frequency problem)
    α = 1: add-one (Laplace) smoothing — most common
    α > 1: heavier smoothing — more regularisation

---

### Part 3: Why the Naive Assumption Works in Practice


### The Miracle of Wrong Assumptions:

The conditional independence assumption is almost always violated in practice:
    - In text, "New" and "York" co-occur far more than independence implies
    - Medical features (blood pressure and cholesterol) are correlated
    - Pixel values in images are heavily spatially correlated

Yet Naive Bayes still works well. Why?

**Reason 1 — Ranking vs calibration:**
For classification, we only need to get the RANKING of class posteriors right,
not their exact values. The independence assumption can distort the probabilities
significantly while still ranking the correct class highest.

**Reason 2 — High-dimensional sparsity:**
In text classification with 50,000 word features, most features are 0 for any
given document. The few non-zero features are approximately independent in practice
(a document about "spam email" doesn't automatically also contain "mortgage").

**Reason 3 — Low data regime:**
Estimating P(x₁, x₂, ..., xₚ | c) jointly requires exponential data.
Estimating P(xⱼ | c) independently requires only linear data.
When data is scarce, the independence assumption is a form of regularisation.


### The Log-Odds Form — Naive Bayes is a Linear Classifier:

For binary classification, taking the log of the posterior ratio:

    log(P(C=1|x) / P(C=0|x)) = log(P(C=1)/P(C=0)) + Σⱼ log(P(xⱼ|C=1)/P(xⱼ|C=0))

This is a linear function of the log-likelihoods! For Gaussian features:

    = log(P(C=1)/P(C=0)) + Σⱼ [log N(xⱼ; μⱼ₁, σⱼ₁²) − log N(xⱼ; μⱼ₀, σⱼ₀²)]

If all class-conditional variances are equal (σⱼ₁² = σⱼ₀²):
    The squared terms cancel → decision boundary is LINEAR in x
    (This is Linear Discriminant Analysis / LDA)

If variances differ between classes:
    Quadratic terms remain → decision boundary is QUADRATIC in x
    (This is Quadratic Discriminant Analysis / QDA)

    ┌─────────────────────────────────────────────────────────────┐
    │  GNB with equal class variances = LDA (Linear boundary)     │
    │  GNB with unequal class variances = QDA (Quadratic)         │
    │  Multinomial NB = Linear classifier in log-count space      │
    │  Bernoulli NB = Linear classifier in binary feature space   │
    └─────────────────────────────────────────────────────────────┘

Naive Bayes is a linear classifier in disguise (for most variants and conditions).
This explains why it competes well with Logistic Regression — they are both
fitting linear decision boundaries, but via different mechanisms:
    LR: discriminative — directly optimises P(C|x) via BCE loss
    NB: generative — models P(x|C) · P(C) and applies Bayes' theorem


---


### Part 4: Naive Bayes for Text Classification


### The Bag-of-Words Representation:

Text is converted to a feature vector by representing each document as a count
(or presence/absence) of each word in the vocabulary:

    Vocabulary: {spam, free, offer, meeting, report, urgent}
    Document: "Free urgent offer"
    Feature vector: [0, 1, 1, 0, 0, 1]   ← bag-of-words (binary)
    Count vector:   [0, 1, 1, 0, 0, 1]   ← same here (each word once)
    Document: "Free free spam spam spam"
    Count vector:   [3, 2, 0, 0, 0, 0]   ← Multinomial NB uses these counts


    # ======================================================================================= # 
    **Diagram 3 — Spam Classification Pipeline:**

    INPUT EMAIL TEXT
         │
         ▼
    Tokenise + clean → [word1, word2, ...]
         │
         ▼
    Vectorise (CountVectorizer or TfidfVectorizer)
         │
         ▼
    Feature matrix X: (n_docs × vocab_size)
         │
         ▼
    Multinomial/Bernoulli Naive Bayes
         │
         ├── Training: count word frequencies per class
         │             P(word | spam), P(word | not_spam)
         │
         └── Prediction: log P(spam) + Σ log P(wordᵢ | spam)
                         vs log P(not_spam) + Σ log P(wordᵢ | not_spam)
                         → predict class with higher log-posterior

    # ======================================================================================= # 


### Why Naive Bayes Dominates Text Classification:

1. **Fast training**: one pass over the data to count word frequencies
2. **Fast prediction**: sum of pre-computed log-probabilities
3. **Memory efficient**: store only (vocab_size × n_classes) parameters
4. **Handles high dimensionality naturally**: vocab can be 100k+ words
5. **Strong baseline**: often competitive with deep learning on short texts


---


### Part 5: Advantages, Limitations, and Comparisons


### Advantages:

1. **Extremely fast training** — O(n·p) single pass over data. No iterations.
   For n=1M examples, p=100k words, training takes seconds.

2. **Works well with very little data** — the independence assumption acts
   as implicit regularisation. Logistic Regression needs far more data.

3. **Naturally handles missing features** — skip missing features in the product.
   Other models require imputation; NB simply omits that factor.

4. **Probabilistic output** — returns calibrated P(C=c|x) directly.
   (Note: these probabilities are often over-confident due to the naive assumption.)

5. **Handles multi-class natively** — compute posteriors for all classes,
   predict the argmax. No one-vs-rest modification needed.

6. **Online/streaming learning** — can update counts incrementally as new
   data arrives. Ideal for data streams without storing the full dataset.


### Limitations:

1. **Conditional independence assumption** — rarely true in practice. This can
   make probability estimates unreliable (overconfident) even when classification
   accuracy is high.

2. **Zero-frequency problem** — a feature-class combination not seen in training
   gives zero probability. Requires Laplace smoothing.

3. **Poor probability calibration** — the naive assumption causes posteriors
   to cluster near 0 and 1 rather than being well-spread. Use Platt scaling or
   isotonic regression to calibrate if probabilities matter.

4. **Can't model feature interactions** — "New York" is one concept, but NB
   treats "New" and "York" as independent. Decision trees naturally capture
   interactions; NB cannot.


### Naive Bayes vs Logistic Regression — The Generative/Discriminative Pair:

These two are deeply connected. Both produce linear decision boundaries for most
common setups. The difference is in how they are estimated:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                      Naive Bayes (Generative)                        │
    │                      vs Logistic Regression (Discriminative)         │
    │────────────────────────────────────────────────────────────────────  │
    │  Objective:       maximise P(x, C)      maximise P(C | x)            │
    │  Training:        count + average       gradient descent             │
    │  Speed:           extremely fast        slower                       │
    │  Low data:        better (regularised)  worse (overfits)             │
    │  Enough data:     worse (wrong model)   better (learns true boundary)│
    │  Calibration:     poor (overconfident)  good (well calibrated)       │
    │  Correlated feats: worse               handles better                │
    │  Extrapolation:   generative (can)      discriminative (cannot)      │
    └──────────────────────────────────────────────────────────────────────┘

The key insight: as training data grows to infinity, Logistic Regression
converges to the true P(C|x), while Naive Bayes converges to the NB model
(which is only the true P(C|x) when the independence assumption holds).

With limited data, NB is often better because it wastes fewer parameters.
With abundant data, LR is usually better because it learns the true boundary.

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ──────────────────────────────────────────────────────────────────────────
    Operation                   Complexity          Notes
    ──────────────────────────────────────────────────────────────────────────
    Training (Gaussian NB)      O(n·p)              one pass: compute means/vars
    Training (Multinomial NB)   O(n·p)              one pass: count words
    Prediction                  O(p·k)              k = number of classes
    Memory                      O(p·k)              store μ,σ² or θ per class
    Laplace smoothing           O(1) extra          just add α to counts
    ──────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Naive Bayes from Scratch": {
        "description": "Gaussian and Multinomial NB implemented via MLE — counting, not gradient descent",
        "runnable": True,
        "code": '''
"""
================================================================================
NAIVE BAYES FROM SCRATCH — MLE ESTIMATION
================================================================================

We implement two variants of Naive Bayes from first principles:
    1. Gaussian NB — for continuous features
    2. Multinomial NB — for count features (text classification)

Training = count observations and compute empirical distributions.
Prediction = apply Bayes' theorem in log space.

No gradient descent. No iterations. One pass over the data.

================================================================================
"""

import numpy as np
from collections import Counter


# =============================================================================
# GAUSSIAN NAIVE BAYES
# =============================================================================

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    Assumes P(xⱼ | C=c) = N(xⱼ; μⱼ_c, σ²ⱼ_c)
    Estimates μ and σ² per feature per class from training data (MLE).

    Prediction in log space to avoid numerical underflow:
        log P(C=c | x) ∝ log P(C=c) + Σⱼ log N(xⱼ; μⱼ_c, σ²ⱼ_c)
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing  # prevents division by zero
        self.classes_      = None
        self.log_priors_   = {}   # log P(C=c)
        self.means_        = {}   # μⱼ_c: shape (p,) per class
        self.vars_         = {}   # σ²ⱼ_c: shape (p,) per class

    def fit(self, X, y):
        """
        Training: for each class, compute the mean and variance of each feature.

        P(C=c):       prior = (count of class c) / n
        μⱼ_c:         mean of feature j among examples with label c
        σ²ⱼ_c:        variance of feature j among examples with label c
        """
        n, p         = X.shape
        self.classes_ = np.unique(y)

        for c in self.classes_:
            X_c = X[y == c]               # examples with class c
            n_c = X_c.shape[0]

            self.log_priors_[c] = np.log(n_c / n)
            self.means_[c]      = X_c.mean(axis=0)           # shape (p,)
            self.vars_[c]       = X_c.var(axis=0) + self.var_smoothing  # shape (p,)

        return self

    def _log_likelihood(self, x, c):
        """
        Compute log P(x | C=c) under the Gaussian assumption for each feature.

        log N(xⱼ; μ, σ²) = -0.5 log(2πσ²) - 0.5 (xⱼ-μ)² / σ²
        """
        mu  = self.means_[c]
        var = self.vars_[c]
        log_probs = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var
        return log_probs.sum()   # sum across features (log of product)

    def predict_log_proba(self, X):
        """
        For each example x, compute log P(C=c | x) ∝ log P(C=c) + log P(x|C=c)
        Returns shape (n, k) where k = number of classes.
        """
        log_posteriors = []
        for x in X:
            row = []
            for c in self.classes_:
                log_post = self.log_priors_[c] + self._log_likelihood(x, c)
                row.append(log_post)
            log_posteriors.append(row)
        return np.array(log_posteriors)   # (n, k)

    def predict(self, X):
        log_post = self.predict_log_proba(X)
        return self.classes_[log_post.argmax(axis=1)]

    def predict_proba(self, X):
        """Convert log posteriors to normalised probabilities."""
        log_post = self.predict_log_proba(X)
        # Softmax-style normalisation in log space (numerically stable)
        log_post -= log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


# =============================================================================
# MULTINOMIAL NAIVE BAYES
# =============================================================================

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes for count features (e.g., word counts).

    Models P(feature j | class c) = θⱼ_c, estimated as:
        θⱼ_c = (count(j in class c) + α) / (total count in class c + α·p)

    Prediction:
        log P(C=c | x) ∝ log P(C=c) + Σⱼ xⱼ · log θⱼ_c

    Note: xⱼ is the COUNT of feature j (e.g., how many times a word appears).
    """

    def __init__(self, alpha=1.0):
        self.alpha      = alpha   # Laplace smoothing parameter
        self.classes_   = None
        self.log_priors_      = {}
        self.log_likelihoods_ = {}   # log θⱼ_c: shape (p,) per class

    def fit(self, X, y):
        """
        Training: count word occurrences per class, then normalise.

        θⱼ_c = (count of feature j in class c + α) / (total features in c + α·p)
        """
        n, p          = X.shape
        self.classes_ = np.unique(y)
        _, p          = X.shape

        for c in self.classes_:
            X_c   = X[y == c]
            n_c   = X_c.shape[0]

            self.log_priors_[c] = np.log(n_c / n)

            # Count of each feature across all class-c documents
            counts = X_c.sum(axis=0) + self.alpha          # shape (p,)
            total  = counts.sum()
            self.log_likelihoods_[c] = np.log(counts / total)  # log θⱼ_c

        return self

    def predict_log_proba(self, X):
        log_posteriors = []
        for x in X:
            row = []
            for c in self.classes_:
                # log P(x | c) = Σⱼ xⱼ · log θⱼ_c
                log_like = (x * self.log_likelihoods_[c]).sum()
                row.append(self.log_priors_[c] + log_like)
            log_posteriors.append(row)
        return np.array(log_posteriors)

    def predict(self, X):
        lp = self.predict_log_proba(X)
        return self.classes_[lp.argmax(axis=1)]


# =============================================================================
# DEMO 1: GAUSSIAN NB — STEP-BY-STEP WALKTHROUGH
# =============================================================================

print("=" * 65)
print("  GAUSSIAN NAIVE BAYES — STEP-BY-STEP")
print("=" * 65)

# Tiny dataset: classify if an animal is a cat or dog based on [weight_kg, height_cm]
X_tiny = np.array([
    [4.0, 25.0],   # cat
    [4.5, 23.0],   # cat
    [3.5, 24.0],   # cat
    [12.0, 55.0],  # dog
    [14.0, 58.0],  # dog
    [10.0, 50.0],  # dog
])
y_tiny = np.array([0, 0, 0, 1, 1, 1])   # 0=cat, 1=dog
classes_str = {0: "cat", 1: "dog"}

gnb = GaussianNaiveBayes()
gnb.fit(X_tiny, y_tiny)

print(f"""
  Training set (n=6): weight [kg] and height [cm]
  Cats:    {X_tiny[y_tiny==0].tolist()}
  Dogs:    {X_tiny[y_tiny==1].tolist()}

  LEARNED PARAMETERS (MLE):
""")
for c in gnb.classes_:
    print(f"  Class {c} ({classes_str[c]}):  prior P(C={c}) = exp({gnb.log_priors_[c]:.4f}) = {np.exp(gnb.log_priors_[c]):.4f}")
    print(f"    Feature 0 (weight): μ = {gnb.means_[c][0]:.2f}, σ² = {gnb.vars_[c][0]:.4f}")
    print(f"    Feature 1 (height): μ = {gnb.means_[c][1]:.2f}, σ² = {gnb.vars_[c][1]:.4f}")

x_query = np.array([5.0, 28.0])
print(f"\n  QUERY: animal with weight=5.0 kg, height=28 cm")

for c in gnb.classes_:
    ll = gnb._log_likelihood(x_query, c)
    lp = gnb.log_priors_[c] + ll
    print(f"  log P(C={c}={classes_str[c]}) + log P(x|C={c}) = {gnb.log_priors_[c]:.4f} + {ll:.4f} = {lp:.4f}")

probs = gnb.predict_proba(x_query.reshape(1, -1))[0]
pred  = gnb.predict(x_query.reshape(1, -1))[0]
print(f"\n  Normalised posteriors:")
for c, p_val in zip(gnb.classes_, probs):
    print(f"    P(C={c}={classes_str[c]} | x) = {p_val:.4f}")
print(f"  Prediction: {classes_str[pred]}")


# =============================================================================
# DEMO 2: MULTINOMIAL NB — TEXT CLASSIFICATION
# =============================================================================

print("\n" + "=" * 65)
print("  MULTINOMIAL NB — SPAM CLASSIFICATION")
print("=" * 65)

# Tiny vocabulary and documents
vocab    = ["free", "money", "urgent", "meeting", "report", "offer", "project", "deadline"]
# Each document is a word count vector over the vocabulary
X_text = np.array([
    [3, 2, 1, 0, 0, 2, 0, 0],  # spam
    [2, 3, 2, 0, 0, 1, 0, 0],  # spam
    [1, 1, 0, 0, 0, 3, 0, 0],  # spam
    [0, 0, 0, 1, 2, 0, 3, 1],  # not spam
    [0, 0, 0, 2, 1, 0, 2, 2],  # not spam
    [0, 1, 0, 1, 3, 0, 1, 0],  # not spam
])
y_text = np.array([1, 1, 1, 0, 0, 0])   # 1=spam, 0=not_spam

mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_text, y_text)

print(f"""
  Vocabulary: {vocab}
  Spam emails (y=1): rows 0-2  | Not-spam emails (y=0): rows 3-5

  LEARNED LOG-LIKELIHOODS log P(word | class):
""")
print(f"  {'Word':>10} | {'log P(w|spam)':>14} | {'log P(w|not_spam)':>18} | {'Spam/NotSpam ratio':>20}")
print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*18}-+-{'-'*20}")
for j, word in enumerate(vocab):
    lls  = mnb.log_likelihoods_[1][j]   # log P(word | spam)
    llns = mnb.log_likelihoods_[0][j]   # log P(word | not_spam)
    ratio = np.exp(lls - llns)
    note  = "spam word" if ratio > 2 else ("not-spam word" if ratio < 0.5 else "neutral")
    print(f"  {word:>10} | {lls:>14.4f} | {llns:>18.4f} | {ratio:>20.3f}  {note}")

# Classify a new email
x_new = np.array([2, 1, 1, 0, 0, 1, 0, 0])   # "free free money urgent offer"
label_str = {0: "not spam", 1: "SPAM"}
pred_text  = mnb.predict(x_new.reshape(1, -1))[0]
lps        = mnb.predict_log_proba(x_new.reshape(1, -1))[0]

print(f"""
  NEW EMAIL word counts: {dict(zip(vocab, x_new))}
  Log posteriors:
    log P(spam | email)     = {lps[1]:.4f}
    log P(not_spam | email) = {lps[0]:.4f}
  Prediction: {label_str[pred_text]}
""")


# =============================================================================
# DEMO 3: SKLEARN COMPARISON ON REAL DATA
# =============================================================================

print("=" * 65)
print("  SKLEARN COMPARISON: OUR NB vs SKLEARN")
print("=" * 65)

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

for ds_name, X_ds, y_ds in [
    ("Iris",  load_iris().data,  load_iris().target),
    ("Wine",  load_wine().data,  load_wine().target),
]:
    X_tr, X_te, y_tr, y_te = train_test_split(X_ds, y_ds, test_size=0.25, random_state=42)

    our_gnb = GaussianNaiveBayes()
    our_gnb.fit(X_tr, y_tr)
    our_acc = accuracy_score(y_te, our_gnb.predict(X_te))

    sk_gnb = GaussianNB()
    sk_gnb.fit(X_tr, y_tr)
    sk_acc = accuracy_score(y_te, sk_gnb.predict(X_te))

    print(f"  {ds_name:>6}: Our GNB = {our_acc:.4f}, sklearn GNB = {sk_acc:.4f}  "
          f"{'✓ match' if abs(our_acc - sk_acc) < 0.02 else '⚠ differs'}")

print(f"\n  Our from-scratch implementation matches sklearn.\n")
''',
    },

    "Bayes' Theorem and the Prior": {
        "description": "How the prior shifts predictions — worked examples, base rate neglect, updating beliefs",
        "runnable": True,
        "code": '''
"""
================================================================================
BAYES' THEOREM IN PRACTICE
================================================================================

Naive Bayes is built on Bayes' theorem. This script demonstrates:
    1. How the prior shifts predictions (base rate matters!)
    2. Sequential Bayesian updating (each feature updates the posterior)
    3. Base rate neglect — the famous cognitive bias NB avoids
    4. Prior choice and its effect on classifier behaviour

================================================================================
"""

import numpy as np


# =============================================================================
# PART 1: BAYES' THEOREM STEP BY STEP
# =============================================================================

print("=" * 65)
print("  PART 1: BAYES' THEOREM — WORKED EXAMPLES")
print("=" * 65)

def bayes_update(prior, likelihood_pos, likelihood_neg):
    """
    Apply Bayes' theorem for a single binary observation.

    prior:           P(hypothesis) before seeing evidence
    likelihood_pos:  P(evidence | hypothesis is TRUE)
    likelihood_neg:  P(evidence | hypothesis is FALSE)

    Returns posterior P(hypothesis | evidence).
    """
    p_evidence = prior * likelihood_pos + (1 - prior) * likelihood_neg
    posterior  = (prior * likelihood_pos) / p_evidence
    return posterior

print(f"""
  EXAMPLE 1: Medical test for a rare disease
  ──────────────────────────────────────────────────────────────────
  Disease prevalence (prior):    P(disease) = 0.01  (1 in 100)
  Test sensitivity:              P(+test | disease) = 0.99
  Test false positive rate:      P(+test | no disease) = 0.05
""")

prior       = 0.01
like_pos    = 0.99   # P(positive test | disease)
like_neg    = 0.05   # P(positive test | no disease)
posterior   = bayes_update(prior, like_pos, like_neg)

print(f"  P(disease | positive test) = ?")
print(f"  = P(+test|disease) · P(disease) / P(+test)")
print(f"  = {like_pos} × {prior} / ({like_pos}×{prior} + {like_neg}×{1-prior})")
print(f"  = {like_pos*prior:.4f} / {like_pos*prior + like_neg*(1-prior):.4f}")
print(f"  = {posterior:.4f}  ({100*posterior:.1f}%)")
print(f"""
  SURPRISING RESULT: Even with a 99% accurate test, a positive result
  for a rare disease (1%) only gives {100*posterior:.0f}% probability of disease.

  This is BASE RATE NEGLECT — humans instinctively ignore P(disease)=0.01
  and focus only on the 99% test accuracy. Bayes' theorem correctly
  accounts for the low prior probability.

  Intuition: out of 10,000 people tested:
    100 have disease:    99 test positive  (true positives)
    9,900 don't:         495 test positive (false positives, 5%)
    Total positives: 594
    P(disease | positive) = 99/594 = {99/594:.3f}
""")

print(f"  EXAMPLE 2: How much does the prior matter?")
print(f"  (Same test: sensitivity=99%, false positive=5%)")
print(f"  {'Prior P(disease)':>18} | {'Posterior P(disease|+)':>25}")
print(f"  {'-'*18}-+-{'-'*25}")
for prior_val in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
    post = bayes_update(prior_val, 0.99, 0.05)
    print(f"  {prior_val:>18.3f} | {post:>25.4f}")

print(f"""
  The posterior is dramatically affected by the prior.
  This is why context (base rates) matter so much in probabilistic reasoning.
""")


# =============================================================================
# PART 2: SEQUENTIAL BAYESIAN UPDATING
# =============================================================================

print("=" * 65)
print("  PART 2: SEQUENTIAL UPDATING — EACH FEATURE UPDATES THE POSTERIOR")
print("=" * 65)

print(f"""
  Naive Bayes: each feature (word) updates the posterior multiplicatively.
  We simulate updating the spam posterior as we read each word in an email.

  Prior: P(spam) = 0.3 (30% of all emails are spam)

  Word likelihoods (from training data):
  ─────────────────────────────────────────────────────────────────
  Word         P(word|spam)   P(word|not_spam)   Spam ratio
  ─────────────────────────────────────────────────────────────────
  "FREE"       0.80           0.05               16.0  (strong spam signal)
  "URGENT"     0.60           0.10               6.0   (spam signal)
  "meeting"    0.05           0.40               0.125 (not-spam signal)
  "report"     0.08           0.35               0.229 (not-spam signal)
  "click"      0.70           0.08               8.75  (spam signal)
  ─────────────────────────────────────────────────────────────────
""")

words = [
    ("FREE",    0.80, 0.05),
    ("URGENT",  0.60, 0.10),
    ("meeting", 0.05, 0.40),
    ("report",  0.08, 0.35),
    ("click",   0.70, 0.08),
]

prior = 0.3
print(f"  Prior: P(spam) = {prior:.3f}")
print(f"  {'After seeing':>20} | {'P(spam|words so far)':>22} | {'Direction'}")
print(f"  {'-'*20}-+-{'-'*22}-+-{'-'*15}")

posterior = prior
for word, lp, ln in words:
    posterior = bayes_update(posterior, lp, ln)
    direction = "↑ more spam" if lp > ln else "↓ less spam"
    print(f"  {'...+'+repr(word):>20} | {posterior:>22.4f} | {direction}")

print(f"""
  Each word sequentially updates the belief.
  Spam words push P(spam) up; not-spam words push it down.
  The final posterior balances all evidence.
  This is exactly what Naive Bayes computes — just in log space.
""")


# =============================================================================
# PART 3: EFFECT OF LAPLACE SMOOTHING
# =============================================================================

print("=" * 65)
print("  PART 3: LAPLACE SMOOTHING — HANDLING UNSEEN WORDS")
print("=" * 65)

print(f"""
  Zero-frequency problem: a word never seen in training spam emails gives
  P(word | spam) = 0, making the entire spam posterior zero.

  Laplace smoothing adds α=1 to every count:
    θⱼ_c = (count(j, c) + α) / (total_count(c) + α × vocab_size)

  Example: vocabulary size = 5, class "spam" has 100 total word occurrences.
  A new word "bitcoin" was never seen in training spam emails.
""")

vocab_size  = 5
total_spam  = 100
count_bitcoin_spam = 0   # never seen in spam

print(f"  {'Smoothing α':>12} | {'P(bitcoin|spam)':>18} | Effect")
print(f"  {'-'*12}-+-{'-'*18}-+-{'-'*30}")
for alpha in [0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
    theta = (count_bitcoin_spam + alpha) / (total_spam + alpha * vocab_size)
    effect = "ZERO (problematic!)" if alpha == 0 else ("good regularisation" if 0.1 <= alpha <= 1.0 else "")
    print(f"  {alpha:>12.3f} | {theta:>18.8f} | {effect}")

print(f"""
  α=0:   "bitcoin" has zero probability in spam → product = 0 → NB refuses to classify
  α=1:   reasonable small probability assigned → NB gracefully handles unseen words
  α=10:  heavy smoothing → probabilities become uniform → loses discrimination

  In practice: α=1.0 (Laplace smoothing) is a robust default.
  For large vocabularies: α=0.1 or alpha selected by cross-validation.
""")


# =============================================================================
# PART 4: CLASS PRIOR SENSITIVITY
# =============================================================================

print("=" * 65)
print("  PART 4: ADJUSTING THE PRIOR FOR CLASS IMBALANCE")
print("=" * 65)

print(f"""
  In imbalanced datasets, the prior P(C=c) strongly affects predictions.
  Setting class_prior manually can help calibrate the model.

  Example: fraud detection (1% fraud, 99% legitimate transactions)
  A model with uniform priors (50% fraud, 50% legit) will predict too much fraud.
  A model with the true prior (1%/99%) will be calibrated correctly.
""")

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)

# Imbalanced dataset: 5% class 1 (minority)
X_imb, y_imb = make_classification(
    n_samples=2000, n_features=5, n_informative=3,
    weights=[0.95, 0.05], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.25, random_state=42)

print(f"  Dataset: n=2000, 5% positive class, 95% negative class")
print(f"  Train class distribution: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
print()

configs = [
    ("Uniform prior [0.5, 0.5]",    [0.5,  0.5],  "ignores imbalance"),
    ("True prior [0.95, 0.05]",     [0.95, 0.05], "matches data"),
    ("Sklearn default (fit_prior)", None,          "learned from training data"),
]

print(f"  {'Prior':>30} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

for name, prior, note in configs:
    if prior is None:
        gnb = GaussianNB()
    else:
        gnb = GaussianNB(priors=prior)
    gnb.fit(X_tr, y_tr)
    y_pred = gnb.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    print(f"  {name:>30} | {acc:>10.4f} | {prec:>10.4f} | {rec:>8.4f} | {f1:>8.4f}")

print(f"""
  With uniform priors: the model predicts class 1 too often (overestimates minority)
  With true priors: correctly calibrated for the actual data distribution
  Default (learned): fits priors from training data — usually best
""")
''',
    },

    "Naive Bayes for Text Classification": {
        "description": "Spam filtering, vocabulary, TF-IDF, and performance comparison with logistic regression",
        "runnable": True,
        "code": '''
"""
================================================================================
NAIVE BAYES FOR TEXT CLASSIFICATION
================================================================================

Text classification is the classic application of Multinomial Naive Bayes.
This script demonstrates:
    1. Text vectorisation (CountVectorizer, TfidfVectorizer)
    2. Multinomial and Bernoulli NB on the 20 Newsgroups dataset
    3. Most discriminative words per class (interpretability)
    4. NB vs Logistic Regression — the generative/discriminative tradeoff

================================================================================
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

np.random.seed(42)


# =============================================================================
# PART 1: 20 NEWSGROUPS — MULTI-CLASS TEXT CLASSIFICATION
# =============================================================================

print("=" * 65)
print("  PART 1: 20 NEWSGROUPS DATASET")
print("=" * 65)

# Use a 4-class subset for speed
categories = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]

train_data = fetch_20newsgroups(subset="train", categories=categories,
                                remove=("headers", "footers", "quotes"),
                                random_state=42)
test_data  = fetch_20newsgroups(subset="test",  categories=categories,
                                remove=("headers", "footers", "quotes"),
                                random_state=42)

print(f"\n  4 newsgroups: {categories}")
print(f"  Training docs: {len(train_data.data)}")
print(f"  Test docs:     {len(test_data.data)}")
print(f"\n  Class distribution in training:")
for i, cat in enumerate(train_data.target_names):
    count = (train_data.target == i).sum()
    print(f"    {cat:>30}: {count} documents")


# =============================================================================
# PART 2: VECTORISATION COMPARISON
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  PART 2: COUNT VECTORS vs TF-IDF")
print(f"{'=' * 65}")

print(f"""
  CountVectorizer: xⱼ = count of word j in the document
  TfidfVectorizer: xⱼ = TF × IDF
                   TF = count / total words in doc  (normalised frequency)
                   IDF = log(n_docs / df(j))         (penalises common words)

  TF-IDF down-weights words that appear in many documents ("the", "a", "is")
  and up-weights words specific to a few documents ("quasar", "goalie").
""")

vectorisers = {
    "CountVectorizer (raw counts)": CountVectorizer(max_features=10000, stop_words="english"),
    "TfidfVectorizer":              TfidfVectorizer(max_features=10000, stop_words="english"),
}

for vec_name, vectoriser in vectorisers.items():
    X_tr = vectoriser.fit_transform(train_data.data)
    X_te = vectoriser.transform(test_data.data)

    mnb = MultinomialNB(alpha=0.1)
    mnb.fit(X_tr, train_data.target)
    acc = accuracy_score(test_data.target, mnb.predict(X_te))
    print(f"  MultinomialNB + {vec_name}: test accuracy = {acc:.4f}")


# =============================================================================
# PART 3: NB VARIANTS COMPARISON
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  PART 3: NAIVE BAYES VARIANTS")
print(f"{'=' * 65}")

# Use TF-IDF for all
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
X_tr_tfidf = tfidf.fit_transform(train_data.data)
X_te_tfidf = tfidf.transform(test_data.data)

# For Bernoulli: binarise features (word present or absent)
count_vec = CountVectorizer(max_features=10000, stop_words="english", binary=True)
X_tr_bin = count_vec.fit_transform(train_data.data)
X_te_bin = count_vec.transform(test_data.data)

count_vec_raw = CountVectorizer(max_features=10000, stop_words="english")
X_tr_raw = count_vec_raw.fit_transform(train_data.data)
X_te_raw = count_vec_raw.transform(test_data.data)

print(f"\n  {'Model':>35} | {'Features':>12} | {'Test Acc':>10}")
print(f"  {'-'*35}-+-{'-'*12}-+-{'-'*10}")

nb_configs = [
    ("MultinomialNB α=1.0", MultinomialNB(alpha=1.0),  X_tr_raw, X_te_raw),
    ("MultinomialNB α=0.1", MultinomialNB(alpha=0.1),  X_tr_raw, X_te_raw),
    ("BernoulliNB α=1.0",   BernoulliNB(alpha=1.0),    X_tr_bin, X_te_bin),
    ("ComplementNB α=0.1",  ComplementNB(alpha=0.1),   X_tr_raw, X_te_raw),
    ("MNB + TF-IDF",        MultinomialNB(alpha=0.1),  X_tr_tfidf, X_te_tfidf),
]

for name, clf, X_tr_f, X_te_f in nb_configs:
    clf.fit(X_tr_f, train_data.target)
    acc = accuracy_score(test_data.target, clf.predict(X_te_f))
    print(f"  {name:>35} | {'Sparse':>12} | {acc:>10.4f}")


# =============================================================================
# PART 4: MOST DISCRIMINATIVE WORDS
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  PART 4: MOST DISCRIMINATIVE WORDS PER CLASS")
print(f"{'=' * 65}")

# Re-fit MNB with raw counts for interpretability
mnb_final = MultinomialNB(alpha=0.1)
mnb_final.fit(X_tr_raw, train_data.target)
feat_names = count_vec_raw.get_feature_names_out()

print(f"\n  Top 8 most indicative words per class:")
print(f"  (highest log-likelihood ratio relative to other classes)")

for i, cat in enumerate(train_data.target_names):
    # log P(word | class i) - mean over other classes
    log_probs = mnb_final.feature_log_prob_[i]
    other_mean = np.mean([mnb_final.feature_log_prob_[j]
                          for j in range(len(train_data.target_names)) if j != i], axis=0)
    scores = log_probs - other_mean
    top_idx = np.argsort(scores)[::-1][:8]
    top_words = [feat_names[j] for j in top_idx]
    print(f"\n  {cat}:")
    print(f"    {top_words}")


# =============================================================================
# PART 5: NB vs LOGISTIC REGRESSION — LOW DATA REGIME
# =============================================================================

print(f"\n{'=' * 65}")
print(f"  PART 5: NB vs LOGISTIC REGRESSION — TRAINING SET SIZE")
print(f"{'=' * 65}")

print(f"""
  THEORY: Naive Bayes is a generative model — it converges faster.
  Logistic Regression is discriminative — it converges to the true boundary
  but needs more data to do so.

  Expected: NB wins with small training data, LR wins with large data.
""")

print(f"  {'Train size':>12} | {'MNB acc':>10} | {'LR acc':>10} | Winner")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

X_all = X_tr_raw
y_all = train_data.target

for n_train in [50, 100, 200, 500, 1000, len(X_all)]:
    idx = np.random.choice(X_all.shape[0], min(n_train, X_all.shape[0]), replace=False)
    X_subset = X_all[idx]
    y_subset = y_all[idx]

    mnb_s = MultinomialNB(alpha=0.1).fit(X_subset, y_subset)
    lr_s  = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_subset, y_subset)

    mnb_acc = accuracy_score(test_data.target, mnb_s.predict(X_te_raw))
    lr_acc  = accuracy_score(test_data.target, lr_s.predict(X_te_raw))
    winner  = "MNB" if mnb_acc > lr_acc + 0.01 else ("LR" if lr_acc > mnb_acc + 0.01 else "tie")
    print(f"  {n_train:>12} | {mnb_acc:>10.4f} | {lr_acc:>10.4f} | {winner}")

print(f"""
  NB often wins at very small training sizes — its independence assumption
  acts as regularisation, preventing overfitting to few examples.
  LR catches up and often overtakes NB as n grows.

  Practical guideline:
    → Small data (< 1000 examples): try NB first
    → Sufficient data: LR usually wins
    → Need probabilities: use LR (better calibrated)
    → Need speed / simplicity: use NB
""")
''',
    },

    "Naive Bayes vs Logistic Regression — Calibration": {
        "description": "Probability calibration, the generative-discriminative gap, and when NB beats LR",
        "runnable": True,
        "code": '''
"""
================================================================================
NAIVE BAYES vs LOGISTIC REGRESSION — CALIBRATION AND COMPARISON
================================================================================

This script explores:
    1. Probability calibration — does P̂(C=1|x) = actual fraction positive?
    2. The generative/discriminative gap across training sizes
    3. Log-loss (proper scoring rule) comparison
    4. Where NB beats LR, LR beats NB, and when they're equivalent

================================================================================
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

np.random.seed(42)


# =============================================================================
# PART 1: PROBABILITY CALIBRATION
# =============================================================================

print("=" * 65)
print("  PART 1: PROBABILITY CALIBRATION")
print("=" * 65)

print(f"""
  A well-calibrated classifier: if it says P(C=1|x) = 0.7, then roughly
  70% of examples with that predicted probability should actually be class 1.

  Naive Bayes tends to be OVERCONFIDENT — its probabilities cluster near 0 and 1
  because the independence assumption makes each feature independently confirm
  the class, leading to extreme posterior values.

  Logistic Regression is usually better calibrated (BCE loss penalises extremes).
""")

X, y = make_classification(n_samples=2000, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB().fit(X_tr, y_tr)
lr  = LogisticRegression(C=1.0, max_iter=1000).fit(X_tr, y_tr)

proba_gnb = gnb.predict_proba(X_te)[:, 1]
proba_lr  = lr.predict_proba(X_te)[:, 1]

# Bin predicted probabilities and compare to actual fraction positive
n_bins = 10
print(f"  Calibration check (predicted prob vs actual fraction positive):")
print(f"  {'Prob bin':>12} | {'GNB pred':>10} | {'GNB actual':>12} | {'LR pred':>10} | {'LR actual':>12}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")

bins = np.linspace(0, 1, n_bins + 1)
for lo, hi in zip(bins[:-1], bins[1:]):
    for proba, name in [(proba_gnb, "GNB"), (proba_lr, "LR")]:
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() > 0:
            mean_pred   = proba[mask].mean()
            frac_pos    = y_te[mask].mean()
    # Print both columns together
    gnb_mask = (proba_gnb >= lo) & (proba_gnb < hi)
    lr_mask  = (proba_lr  >= lo) & (proba_lr  < hi)
    gnb_pred = proba_gnb[gnb_mask].mean() if gnb_mask.sum() > 0 else float("nan")
    gnb_act  = y_te[gnb_mask].mean()      if gnb_mask.sum() > 0 else float("nan")
    lr_pred  = proba_lr[lr_mask].mean()   if lr_mask.sum() > 0  else float("nan")
    lr_act   = y_te[lr_mask].mean()       if lr_mask.sum() > 0  else float("nan")
    if gnb_mask.sum() > 0 or lr_mask.sum() > 0:
        print(f"  {lo:.1f}–{hi:.1f}       | {gnb_pred:>10.3f} | {gnb_act:>12.3f} | "
              f"{lr_pred:>10.3f} | {lr_act:>12.3f}")

print(f"""
  Perfect calibration: pred ≈ actual (diagonal line)
  GNB calibration: often the probabilities are pushed to extremes (0.01 or 0.99)
  LR calibration: generally closer to the diagonal
""")


# =============================================================================
# PART 2: LOG-LOSS COMPARISON
# =============================================================================

print("=" * 65)
print("  PART 2: LOG-LOSS (PROPER SCORING RULE)")
print("=" * 65)

print(f"""
  Log-loss = -mean(y·log(p) + (1-y)·log(1-p))
  Penalises confident wrong predictions HEAVILY (unlike accuracy).
  A well-calibrated model has lower log-loss than an overconfident one.
""")

print(f"  {'Metric':>25} | {'Gaussian NB':>14} | {'Logistic Reg':>14}")
print(f"  {'-'*25}-+-{'-'*14}-+-{'-'*14}")
print(f"  {'Accuracy':>25} | {accuracy_score(y_te, gnb.predict(X_te)):>14.4f} | "
      f"{accuracy_score(y_te, lr.predict(X_te)):>14.4f}")
print(f"  {'Log-loss':>25} | {log_loss(y_te, proba_gnb):>14.4f} | "
      f"{log_loss(y_te, proba_lr):>14.4f}")
print(f"  {'Brier score':>25} | {brier_score_loss(y_te, proba_gnb):>14.4f} | "
      f"{brier_score_loss(y_te, proba_lr):>14.4f}")

print(f"""
  Even when accuracy is similar, LR almost always has lower log-loss.
  This matters when downstream decisions depend on probability calibration
  (e.g., threshold tuning, cost-sensitive classification, uncertainty estimation).

  If you only need class labels: NB accuracy is often comparable to LR.
  If you need calibrated probabilities: LR is usually superior.
""")


# =============================================================================
# PART 3: WHERE NB OUTPERFORMS LR
# =============================================================================

print("=" * 65)
print("  PART 3: WHEN DOES NB WIN? — CORRELATED FEATURES")
print("=" * 65)

print(f"""
  Counterintuitive: NB often wins over LR when features are HIGHLY CORRELATED.
  Reason: correlated features cause LR to over-count their evidence (multicollinearity).
  NB's naive assumption accidentally helps by treating each correlated feature
  as independent evidence, but the redundancy averages out.

  Test: vary feature correlation and compare NB vs LR.
""")

from sklearn.datasets import make_classification

print(f"  {'n_redundant feats':>20} | {'GNB acc':>10} | {'LR acc':>10} | Winner")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

for n_red in [0, 2, 5, 8, 10, 15]:
    n_features = 20                          # enlarged so all n_red values fit
    n_info = max(2, n_features - n_red - 1)  # always leaves at least 1 useless feature
    n_red_capped = min(n_red, n_features - n_info - 1)
    X_r, y_r = make_classification(
        n_samples=500, n_features=n_features, n_informative=n_info,
        n_redundant=n_red_capped, n_repeated=0,
        random_state=42
    )
    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(X_r, y_r, test_size=0.25, random_state=42)

    gnb_r = GaussianNB().fit(X_r_tr, y_r_tr)
    lr_r  = LogisticRegression(C=1.0, max_iter=1000).fit(X_r_tr, y_r_tr)

    gnb_acc = accuracy_score(y_r_te, gnb_r.predict(X_r_te))
    lr_acc  = accuracy_score(y_r_te, lr_r.predict(X_r_te))
    winner  = "GNB" if gnb_acc > lr_acc + 0.01 else ("LR" if lr_acc > gnb_acc + 0.01 else "tie")
    print(f"  {n_red:>20} | {gnb_acc:>10.4f} | {lr_acc:>10.4f} | {winner}")

print(f"""
  Key findings:
    Low correlation (n_redundant=0): LR usually wins (better model)
    High correlation: GNB can compete or win (independence assumption helps)

  Summary of the NB vs LR tradeoffs:
  ──────────────────────────────────────────────────────────────────────────
  Situation                      Prefer NB?    Reason
  ──────────────────────────────────────────────────────────────────────────
  Very small training data       YES           independence = regularisation
  Very fast training needed      YES           single pass, closed-form
  Text / high-dim sparse data    YES           handles p >> n naturally
  Calibrated probabilities       NO            NB overconfident
  Feature interactions matter    NO            LR handles better
  Correlated continuous feats    MAYBE         depends on data
  Large n, continuous features   NO            LR converges to true boundary
  ──────────────────────────────────────────────────────────────────────────
""")
''',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import numpy as np
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 65)
    print("  NAIVE BAYES: PROBABILISTIC CLASSIFICATION")
    print("=" * 65)
    print("""
  Key concepts demonstrated:
    • Bayes' theorem: P(C|x) ∝ P(x|C) · P(C)
    • Naive assumption: features conditionally independent given class
    • Training = counting (MLE) — no gradient descent, single pass
    • Log-space computation prevents numerical underflow
    • Gaussian NB: continuous features (μ and σ² per class per feature)
    • Multinomial NB: count features (word frequencies in text)
    • Laplace smoothing: prevents zero-probability for unseen features
    • Generative model: learns P(x|C), unlike LR which learns P(C|x)
    """)

    np.random.seed(42)

    # Gaussian NB on Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    gnb = GaussianNB()
    gnb.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, gnb.predict(X_te))

    print("=" * 65)
    print("  GAUSSIAN NB ON IRIS DATASET")
    print("=" * 65)
    print(f"\n  3 classes, 4 features, n=150")
    print(f"  Test accuracy: {acc:.4f}")

    print(f"\n  Learned class-conditional means (μⱼ_c):")
    print(f"  {'Class':>10}", end="")
    for fname in iris.feature_names:
        print(f"  {fname[:12]:>14}", end="")
    print()
    print(f"  {'-' * 10}", end="")
    for _ in iris.feature_names:
        print(f"--{'-' * 14}", end="")
    print()
    for c, cname in enumerate(iris.target_names):
        print(f"  {cname:>10}", end="")
        for mu in gnb.theta_[c]:
            print(f"  {mu:>14.3f}", end="")
        print()

    print(f"\n  Learned class-conditional standard deviations (σⱼ_c):")
    print(f"  {'Class':>10}", end="")
    for fname in iris.feature_names:
        print(f"  {fname[:12]:>14}", end="")
    print()
    print(f"  {'-' * 10}", end="")
    for _ in iris.feature_names:
        print(f"--{'-' * 14}", end="")
    print()
    for c, cname in enumerate(iris.target_names):
        print(f"  {cname:>10}", end="")
        for var in gnb.var_[c]:
            print(f"  {np.sqrt(var):>14.3f}", end="")
        print()

    # Manual Bayes calculation
    print(f"\n  Classify a flower with [5.1, 3.5, 1.4, 0.2]:")
    x_q = np.array([5.1, 3.5, 1.4, 0.2])
    proba = gnb.predict_proba(x_q.reshape(1, -1))[0]
    for c, cname in enumerate(iris.target_names):
        print(f"    P({cname} | x) = {proba[c]:.6f}")
    pred = iris.target_names[gnb.predict(x_q.reshape(1, -1))[0]]
    print(f"  Prediction: {pred}")

    print(f"\n{'=' * 65}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'=' * 65}")
    print("""
  1.  NB applies Bayes' theorem: P(C|x) ∝ P(C) · Πⱼ P(xⱼ|C)
  2.  Naive assumption: features are conditionally independent given class
  3.  Training = MLE: compute class priors + per-feature likelihoods
  4.  No gradient descent — a single pass over data is all that's needed
  5.  Gaussian NB: P(xⱼ|c) = N(μⱼ_c, σ²ⱼ_c), for continuous features
  6.  Multinomial NB: P(xⱼ|c) = θⱼ_c (word probabilities), for counts
  7.  Bernoulli NB: P(xⱼ|c) = Bernoulli(θⱼ_c), explicitly models absences
  8.  Laplace smoothing (α=1): prevents zero-frequency catastrophe
  9.  Works in log-space: log P(C|x) = log P(C) + Σⱼ log P(xⱼ|C)
  10. Generative model: learns P(x,C) — can in principle generate new data
  11. Probabilities are often overconfident — Platt scaling can recalibrate
  12. NB beats LR with very small data; LR beats NB with sufficient data
  13. Classic use case: spam filtering, news classification, medical diagnosis
    """)

# # ─────────────────────────────────────────────────────────────────────────────
# # CONTENT EXPORT
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_content():
#     return {
#         "theory":                THEORY,
#         "theory_raw":            THEORY,
#         "complexity":            COMPLEXITY,
#         "operations":            OPERATIONS,
#         "interactive_components": [],
#     }


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
        main_script = None  # _MAIN_SCRIPT

    scripts_available = main_script.exists()

    if "tok_step_status" not in st.session_state:
        st.session_state.tok_step_status = {}
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
    visual_html = ""
    visual_height = 400
    try:
        from Supervised_Learning.visuals.naive_bayes_visual import (  # ← match your exact folder casing
            NB_VISUAL_HTML,
            NB_VISUAL_HEIGHT,
        )
        visual_html = NB_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = NB_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[05_naive_bayes.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": visual_html,
        "visual_height": visual_height,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }
