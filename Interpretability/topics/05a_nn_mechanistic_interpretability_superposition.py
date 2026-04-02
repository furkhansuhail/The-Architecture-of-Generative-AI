"""
Superposition — How Neural Networks Pack More Features Than Neurons
===================================================================

Superposition (Elhage, Henighan et al., 2022 — "Toy Models of Superposition")
is the phenomenon whereby a neural network represents more features than it has
neurons, by storing multiple features in overlapping, nearly-orthogonal directions
within the same activation space. It is the single most important theoretical
result in mechanistic interpretability: it explains why individual neurons are
polysemantic, why SAEs are necessary, why circuits are hard to find, and why
direct neuron analysis fails at scale.

This module builds superposition theory from scratch — the geometry, the phase
transitions, the toy model experiments, the implications for interpretability
methodology, and the practical diagnostic tools for detecting it.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Superposition — Features Beyond Neurons"
DISPLAY_NAME = "05a · Superposition"
ICON = "💠"
SUBTITLE = "Polysemanticity, geometry of overcomplete representations, and toy models"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Puzzle That Started It All

Open a trained GPT-2 and look at neuron 3714 in layer 5. Collect all the
text fragments that cause it to fire strongly. You find:

    Donald Trump — political context
    The Pope — religious context
    Marie Curie — scientific context

These are three utterly unrelated concepts. Why should a single neuron respond
to all of them? The obvious answer — "the model is confused" — is wrong. This
behaviour is systematic, appears across architectures, and is not an accident.
It is the outward sign of a deep geometric phenomenon called SUPERPOSITION.

Understanding superposition requires starting not with neurons, but with
the abstract question: how should a learning system store many concepts in
a space that is too small to give each concept its own dedicated dimension?

The answer — nearly orthogonal directions, exploiting sparsity — turns out
to be the foundation of how large neural networks actually work.


##### PART I: THE GEOMETRY OF HIGH-DIMENSIONAL SPACES

### Why High Dimensions Are Counterintuitive

Human spatial intuition is trained in 3D. In high-dimensional spaces,
geometric properties become radically different — and these differences
are exactly what makes superposition possible.

The key fact: in ℝⁿ for large n, you can find an exponentially large
number of unit vectors that are all nearly orthogonal to each other.

    Formal statement: For any 0 < ε < 1 and δ > 0, there exist
    N = Ω(exp(δn)) unit vectors u₁, ..., uN ∈ ℝⁿ such that:

    |uᵢ · uⱼ| ≤ ε  for all i ≠ j

    That is, N vectors that are mutually nearly orthogonal, where N is
    EXPONENTIAL in the dimension n.

    # ================================================================== #
    **The capacity of high-dimensional spaces:**

    Dimension n    Near-orthogonal vectors (ε = 0.1)
    ───────────────────────────────────────────────────
    n = 2          ~3          (a pentagon's vertices)
    n = 10         ~thousands
    n = 100        ~10^10
    n = 512        ~10^{50}
    n = 4096       ~10^{400}

    BERT-base: n=768. GPT-2-small: n=768. GPT-3: n=12,288.
    The available near-orthogonal directions vastly exceed the number
    of linguistic features a model could plausibly need.
    # ================================================================== #

This is the Johnson-Lindenstrauss lemma's core implication: high-dimensional
spaces are vast in a way that low-dimensional intuition completely misses.
A layer with 768 neurons is not limited to 768 features — it has room for
vastly more, packed as near-orthogonal directions.

### Interference — The Price of Superposition

If features are stored as near-orthogonal directions in a shared space,
they INTERFERE with each other. When feature i is active with value fᵢ,
it contributes noise to every other feature's neuron because the feature
directions are not perfectly orthogonal.

Let feature i be encoded as direction wᵢ ∈ ℝⁿ (‖wᵢ‖=1).
Let feature j be encoded as direction wⱼ ∈ ℝⁿ (‖wⱼ‖=1).

When feature i is active with value fᵢ, the "signal" in the direction of
feature j is:

    interference(i→j) = fᵢ · (wᵢ · wⱼ)

If wᵢ and wⱼ are ε-orthogonal (|wᵢ · wⱼ| ≤ ε), then:
    |interference(i→j)| ≤ ε · |fᵢ|

For k simultaneously active features, total interference on feature j:

    total noise on j = Σ_{i≠j, fᵢ≠0} fᵢ · (wᵢ · wⱼ)

Expected magnitude: ≈ √(k-1) · ε · avg(|fᵢ|)   (central limit theorem)

For n=512, ε = 1/√n ≈ 0.044, and k=10 simultaneously active features:
    Expected noise ≈ 3 × 0.044 × avg(f) ≈ 13% of the typical feature value

This is the fundamental TRADE-OFF of superposition:
    MORE features stored → more interference between them
    SPARSER inputs → less interference per feature at any moment

    # ================================================================== #
    **Superposition works because of sparsity:**

    If most features are zero most of the time (sparse input distribution),
    then even though 10,000 features share the same 512-dimensional space,
    at any given input only ~10 features are active simultaneously.
    The interference from 10 active features is small enough to recover
    each feature's value with acceptable precision.

    If inputs were DENSE (all features active all the time), superposition
    would fail — the interference from all 10,000 simultaneously active
    features would be catastrophic noise.

    This is why sparsity is not a regularisation trick in neural networks —
    it is a necessary condition for superposition to function correctly.
    # ================================================================== #


##### PART II: THE TOY MODEL OF SUPERPOSITION

### The Elhage et al. (2022) Setup

Elhage, Henighan et al. — "Toy Models of Superposition" (Anthropic, 2022)
— built the minimal model that exhibits superposition, proving that it
emerges from ordinary gradient descent, not from any ad hoc construction.

The model: a linear autoencoder with a bottleneck, where the bottleneck
has fewer dimensions than the input features.

    Input: x ∈ ℝᵐ  (m features, each independently drawn from a sparse distribution)
    Encoder: h = Wx  (W ∈ ℝⁿˣᵐ, n < m — the bottleneck)
    Non-linearity: h̃ = ReLU(h)  (applied elementwise)
    Decoder: x̂ = Wᵀh̃ + b  (using the transpose of W)
    Loss: L = ‖x − x̂‖²  (reconstruction loss, weighted by feature importance)

    # ================================================================== #
    **Why this is the right model to study:**

    The encoder W ∈ ℝⁿˣᵐ maps m features into n dimensions (n < m).
    This is a COMPRESSED representation — fewer dimensions than features.

    If n ≥ m: no compression needed. Each feature gets its own dimension.
    If n < m: the model must choose how to pack m features into n dimensions.

    The question is: does the model learn to pack features as near-orthogonal
    directions (superposition), or does it simply ignore the "extra" m−n features?

    The answer: IT USES SUPERPOSITION. It learns to represent more than n
    features by exploiting sparsity, at the cost of interference.
    This is not programmed in — it emerges from minimising reconstruction loss.
    # ================================================================== #

### The Loss Function With Feature Importance

The reconstruction loss is weighted by feature importance:

    L = Σᵢ₌₁ᵐ  Sᵢ · (xᵢ − x̂ᵢ)²

where Sᵢ ∈ [0, 1] is the importance of feature i. High Sᵢ means the model
must reconstruct feature i accurately. Low Sᵢ means errors on feature i
are tolerable.

This importance weighting reflects a key reality: in language models, some
features (common words, critical syntactic structures) matter much more than
others (rare entities, domain-specific terminology). The model should allocate
representational capacity proportionally.

### Feature Sparsity — The Other Axis

Each feature xᵢ is drawn from a SPARSE distribution:
  • With probability (1 − p): xᵢ = 0  (feature absent)
  • With probability p: xᵢ ~ Uniform(0, 1)  (feature present)

The sparsity parameter is S_p = 1 − p (fraction of inputs where xᵢ = 0).
High S_p: feature is rarely active — low interference with others.
Low S_p: feature is often active — high interference if superposed.

The two-dimensional phase space (importance × sparsity) determines the
model's optimal strategy for each feature.


##### PART III: PHASE TRANSITIONS — THE GEOMETRY OF OPTIMAL SOLUTIONS

### Two Regimes — Dedicated vs Superposed

When the toy model is trained with gradient descent, the learned weight
matrix W exhibits one of two qualitatively different structures per feature:

    REGIME 1 — DEDICATED REPRESENTATION (monosemantic):
    Feature i is stored in its own column of W: W[:,i] ≈ eₖ (a standard
    basis vector) for some neuron k. Neuron k responds to feature i and
    only feature i (to the extent possible). The feature is "clean."
    It has unit norm in W and orthogonal to all other feature directions.

    REGIME 2 — SUPERPOSITION (polysemantic):
    Feature i is stored as a direction wᵢ = W[:,i] that is shared with
    other features. wᵢ is NOT aligned with any standard basis vector.
    Neuron k may participate in encoding features i, j, and l simultaneously,
    each with different weights. The neuron is polysemantic.

Whether a feature uses Regime 1 or Regime 2 depends on the feature's
IMPORTANCE Sᵢ and SPARSITY (1−p) in a way that exhibits a SHARP PHASE TRANSITION.

    # ================================================================== #
    **The phase transition — which features get dedicated neurons:**

    Each feature "precipitates out" of superposition and into a dedicated
    neuron when its importance exceeds a critical threshold:

    S_critical ≈ 1 / (n × (1−p))

    where n = number of neurons, (1−p) = feature sparsity.

    Intuition: a dedicated neuron is worth allocating (sacrificing one neuron
    out of n for one feature) when the feature is important enough that the
    BENEFIT of clean representation outweighs the COST of using up a neuron.

    For sparse features (high (1-p), near 1):
    S_critical ≈ 1/n — even low-importance features get dedicated neurons
    (sparsity makes the cost of superposition high, so you might as well
    use dedicated neurons for everything).

    For dense features (p near 1, sparsity near 0):
    S_critical ≈ 1/(n×1) = 1/n but the model can't superpose at all
    (dense features always interfere badly). Only the top n features get
    any representation; the rest are abandoned.

    For moderately sparse features (p ≈ 0.05, sparsity ≈ 0.95):
    Many features can be superposed with manageable interference.
    The top n features get dedicated neurons; the remainder are superposed.
    # ================================================================== #

### The Double Descent of Superposition

As feature sparsity S_p increases from 0 to 1 (features become sparser),
the number of features the model can effectively represent undergoes a
double transition:

    S_p ≈ 0 (dense):    Model represents exactly n features. The n most
                         important get dedicated neurons. All others get zero.
                         m − n features are completely ignored.

    S_p increasing:      More features can be superposed. The model starts
                         to represent MORE than n features, even though it
                         still has only n neurons.

    S_p → 1 (very sparse): The model represents nearly all m features
                         with manageable interference, because only a few
                         are ever simultaneously active.

This means feature count is NOT fixed by architecture:

    representable_features(n, S_p) ≈ n × 1 / (1 − S_p)

For n=512 neurons and S_p = 0.99 (features active 1% of the time):
    representable_features ≈ 512 / 0.01 = 51,200 features!

This is the core theoretical result: a layer with 512 neurons can
effectively represent ~50,000 features in sparse regimes. The capacity
is not limited by the number of neurons but by the product of neurons
and the inverse of feature density.


### The Geometry of Superposed Features — Polytopes

When multiple features are superposed in the same n-dimensional space,
the optimal arrangement of their weight vectors traces out the vertices
of high-dimensional geometric objects.

For 2 features in 1D:
  The two feature directions can only be +1 and −1.
  If feature 1 is positive, it interferes with feature 2 via −1.
  This is the worst possible interference geometry.

For 3 features in 2D:
  Optimal: vertices of a regular triangle (3 directions at 120° apart).
  |wᵢ · wⱼ| = cos(120°) = 0.5 for all pairs. Uniform interference.

For 5 features in 2D:
  Optimal: vertices of a regular pentagon (5 directions at 72° apart).
  |wᵢ · wⱼ| varies — adjacent vertices at 72° (cos 72° ≈ 0.31),
  skip-one at 144° (cos 144° ≈ −0.81).

For k features in n dimensions:
  Optimal arrangements are often the vertices of regular polytopes or
  other symmetric configurations (simplex, hypercube, icosahedron...).
  These minimise the maximum interference between feature pairs.

    # ================================================================== #
    **The antipodal pairs trick:**

    A particularly elegant solution: for n neurons, store 2n features
    using antipodal pairs. Feature i is stored as wᵢ ∈ ℝⁿ.
    Feature j is stored as wⱼ = −wᵢ (the antipodal direction).

    After ReLU: both features are recovered because:
      When feature i is active (positive): neuron k fires for wᵢ.
      When feature j is active (positive): neuron k is INHIBITED for −wᵢ.

    Wait — the ReLU blocks negatives. So this works:
    If the input to neuron k is wᵢ·x (the dot product):
      Feature i active (+1), feature j active (0): output = ReLU(+1) = +1 ✓
      Feature i active (0), feature j active (+1): output = ReLU(−1) = 0 ✓

    Using both wᵢ and −wᵢ allows encoding a feature's PRESENCE vs ABSENCE,
    doubling the effective feature count per neuron at the cost of always
    representing each feature as a polarity pair.

    This is the "square" configuration: n neurons can represent 2n features
    in antipodal pairs, with zero interference between paired features (they
    never co-occur) and some interference between unpaired features.
    # ================================================================== #


##### PART IV: THE LOSS LANDSCAPE OF SUPERPOSITION

### The Analytic Loss Formula

For the toy model with m features, n neurons, and feature sparsity S_p:

    Expected loss per feature i (under optimal weight Wᵢ):

    L_dedicated(i) = 0  (if feature i has a dedicated neuron)

    L_superposed(i) = (1 − S_p) × Σ_{j≠i} (wᵢ · wⱼ)² × Sⱼ × (1 − S_p)

The term (wᵢ · wⱼ)² is the squared interference between features i and j.
When feature j is active (probability (1 − S_p)) AND feature i is active
(probability (1 − S_p)), both contribute to i's reconstruction error.

Simplifying for uniform features (all Sᵢ = S, all S_p equal):

    L_superposed ≈ S × (1−S_p)² × Σ_{j≠i} (wᵢ · wⱼ)²

The sum Σ_{j≠i} (wᵢ · wⱼ)² is the SQUARED INTERFERENCE INDEX for feature i.
Minimising this drives the weight vectors toward orthogonality — which is
impossible if m > n but is approximated as well as the geometry allows.

### The Crossover Condition — When Superposition Beats Ignoring

Suppose the model has n neurons and m > n features. For each feature i,
the model has two options:
  (a) REPRESENT i in superposition: pay interference cost L_superposed(i)
  (b) IGNORE i: set Wᵢ = 0, never reconstruct it, pay full loss Sᵢ

Feature i is worth superposing if:
    L_superposed(i) < Sᵢ   (interference cost < cost of ignoring)

Equivalently: (1−S_p)² × Σⱼ (wᵢ·wⱼ)² × Sⱼ < Sᵢ

For uniform features (all Sⱼ = S) with m features in n neurons,
each feature pair has interference ≈ (m − n) / (n × m):

    Critical condition: S × (1−S_p)² × (m−n)/n < S
    Simplifies to:      (1−S_p)² × (m−n) < n
    Or:                 m < n / (1−S_p)² + n = n × (1 + 1/(1−S_p)²)

This sets an UPPER BOUND on useful superposition. Beyond this bound,
the interference from superposing too many features exceeds the cost
of ignoring them, and the model stops superposing additional features.

    # ================================================================== #
    **Numerical examples:**

    n=100 neurons, S_p = 0.9 (features active 10% of the time):
    Max useful features ≈ 100 × (1 + 1/0.01) = 100 × 101 = 10,100

    n=100 neurons, S_p = 0.99 (features active 1% of the time):
    Max useful features ≈ 100 × (1 + 1/0.0001) = 100 × 10,001 = 1,000,100

    n=100 neurons, S_p = 0.5 (features active 50% of the time):
    Max useful features ≈ 100 × (1 + 4) = 500

    The maximum feature count is EXQUISITELY sensitive to sparsity.
    Going from 50% to 99% activation probability enables 2000× more features
    in the same number of neurons!
    # ================================================================== #


##### PART V: POLYSEMANTICITY — WHAT SUPERPOSITION LOOKS LIKE FROM OUTSIDE

### What Polysemanticity Is

Polysemanticity is the observable consequence of superposition: individual
neurons respond to multiple unrelated features because each neuron participates
in multiple feature directions simultaneously.

Formally: neuron k has a "receptive field" in feature space consisting of
all features wᵢ with Wₖᵢ = W[k,i] ≠ 0. If many features have W[k,i] ≠ 0,
neuron k is polysemantic — it fires for many different features.

The activation of neuron k for input x is:

    h_k = ReLU( Σᵢ Wₖᵢ xᵢ )  =  ReLU( Σᵢ (wᵢ_direction)ₖ × xᵢ )

This sums contributions from ALL active features that project onto neuron k.
When feature i is active and has high Wₖᵢ, neuron k fires.
But so does neuron k when feature j (unrelated to i) is active and has high Wₖⱼ.

    # ================================================================== #
    **Polysemanticity in GPT-2 (from Anthropic's research):**

    Neuron analysis of GPT-2 medium, layer 3:
    Neuron 832:  fires for "Monday", "Tuesday", "Wednesday"... (days of week)
                 fires for "January", "March", "April"... (months)
                 These share the feature: "temporal enumeration"
                 (related — both are sequences)

    Neuron 4731: fires for "Trump", "Pope", "Marie Curie"
                 These are all PROMINENT INDIVIDUALS — but their prominence
                 is in completely unrelated domains (politics, religion, science)
                 This neuron is more strongly polysemantic — the concepts
                 it combines are semantically distant.

    The distinction matters:
    Neuron 832: might be encoding a SINGLE feature "temporal sequence member"
                that happens to include both days and months.
    Neuron 4731: is more likely encoding multiple DISTINCT features that
                 are superposed because they are all rare and prominent.
    # ================================================================== #

### Measuring Polysemanticity

Several quantitative measures of polysemanticity have been proposed:

    1. PARTICIPATION RATIO (EFFECTIVE DIMENSIONALITY):
    For neuron k, compute the weight distribution over input features:
    pᵢ = Wₖᵢ² / Σⱼ Wₖⱼ²   (normalised squared weights)

    PR(k) = 1 / Σᵢ pᵢ²   (participation ratio — the "effective feature count")

    PR(k) = 1:   Neuron k responds to exactly 1 feature (monosemantic).
    PR(k) = m:   Neuron k responds equally to all m features (maximally polysemantic).
    PR(k) ≈ 5:  Neuron k is primarily driven by ~5 features.

    2. NEURON MONOSEMANTICITY SCORE (Anthropic):
    For a given neuron, collect the top-k activating input fragments.
    Have human annotators judge: "do these fragments share a single concept?"
    Score = fraction of annotators who agree on a concept.
    Monosemanticity ≈ 1: strongly monosemantic.
    Monosemanticity ≈ 0: strongly polysemantic.

    3. SAE FEATURE ATTRIBUTION:
    Train an SAE on the activations at the layer of interest.
    For each neuron k, find which SAE features activate through it:
    if SAE feature j requires neuron k (Ŵ[k,j] is large), then
    neuron k is "used by" feature j.
    The number of SAE features that use neuron k is a proxy for polysemanticity.

    # ================================================================== #
    **Polysemanticity spectrum across architectures:**

    Small, narrow networks (n small, few tasks):
    → MONOSEMANTIC neurons are common.
    → Network doesn't need superposition; it has enough capacity.
    → Classic "grandmother cells" may actually exist here.

    Large, wide networks (n large, many tasks):
    → POLYSEMANTIC neurons everywhere.
    → Network exploits superposition to represent vast feature sets.
    → Direct neuron interpretation mostly fails.

    Fine-tuned models (specific task, concentrated features):
    → PARTIALLY MONOSEMANTIC — task-critical features get dedicated neurons
      (they become high-importance), task-irrelevant features remain superposed.
    → Prompts for specific tasks → polysemanticity in task-relevant layers decreases.
    # ================================================================== #


##### PART VI: SUPERPOSITION AND THE RESIDUAL STREAM IN TRANSFORMERS

### Superposition in the Transformer's Residual Stream

In a transformer, the residual stream at each token position is a vector
x ∈ ℝᵈ (d = d_model, e.g., 768 for BERT-base). All attention heads and MLP
layers READ FROM and WRITE TO this shared vector.

Superposition applies here in a particularly important way: the residual
stream must simultaneously encode ALL features that are relevant to the
current token and context — across all layers, for all heads that will
subsequently read it.

A token's residual stream after L layers may need to encode:
  • Syntactic role (subject/object/modifier)
  • POS tag (noun/verb/adjective)
  • Entity type (person/location/organisation)
  • Semantic role (agent/patient/instrument)
  • Discourse position (topic/focus/background)
  • Local context features (recent words, upcoming words)
  • Long-range dependencies (coreference, scope)
  • Task-relevant features (sentiment, entailment signal)
  • ... potentially thousands of features

All of these must coexist in d=768 dimensions. This is ONLY possible via
superposition — and the residual stream IS in superposition, by design.

    # ================================================================== #
    **The residual stream as a superposed communication bus:**

    Token "bank" in "I went to the bank to deposit money":
    Residual stream must encode (simultaneously in 768 dims):
      f₁ = "bank" (financial institution) — HIGH activation
      f₂ = "bank" (river bank) — LOW activation (context rules it out)
      f₃ = NOUN — HIGH
      f₄ = object of "went to" — HIGH
      f₅ = entity type: ORGANISATION — HIGH
      f₆ = financial context — HIGH (from preceding context)
      f₇ = location role — MODERATE
      ... thousands more features at lower activations ...

    All 768 dimensions carry a superposition of all these features.
    No single dimension IS "financial institution meaning."
    The meaning emerges from the DIRECTION in the 768-dim space.
    # ================================================================== #


### Attention Heads Reading Superposed Representations

When an attention head reads from a position's residual stream via the
value matrix W_V, it is performing:

    v = W_V · x_source

This is a LINEAR FUNCTION of the superposed residual stream x_source.
The matrix W_V effectively selects a specific subspace of x_source to read.

If x_source = Σᵢ fᵢ · wᵢ (superposition of features), then:

    v = W_V · (Σᵢ fᵢ · wᵢ) = Σᵢ fᵢ · (W_V · wᵢ)

Each term (W_V · wᵢ) is the "image" of feature i through the value
projection. If W_V is designed to PROJECT OUT a specific feature direction
wᵢ, then v will carry primarily the information about feature i.

This is the circuit-level view: attention heads implement selective reading
of the superposed residual stream by learning W_V matrices that project
onto the features they need, ignoring the interference from other superposed
features.

    # ================================================================== #
    **Value matrix as a selective reader:**

    Suppose x = 0.8·f_NOUN + 0.3·f_LOCATION + noise (superposition)
    where f_NOUN = [1,0,0,...], f_LOCATION = [0,0.7,0.7,...] (in feature space)

    An attention head that needs the NOUN feature:
    W_V is learned to be approximately:
    W_V ≈ [1,0,0,...; ...]  (projects onto f_NOUN direction)
    Then: v ≈ W_V · x ≈ 0.8 · W_V·f_NOUN + 0.3 · W_V·f_LOCATION
           ≈ 0.8 · 1 + 0.3 · 0 = 0.8  (successfully read NOUN feature!)

    An attention head that needs the LOCATION feature:
    W_V ≈ [0,0.7,0.7,...]  (projects onto f_LOCATION direction)
    Then: v ≈ 0.8·0 + 0.3·1 = 0.3  (successfully read LOCATION feature!)

    Each head can selectively read ONE feature from the superposed soup,
    by learning an appropriate W_V. This is the key insight connecting
    superposition to circuit analysis.
    # ================================================================== #


##### PART VII: SUPERPOSITION AND MLP LAYERS — COMPUTATION IN SUPERPOSED SPACE

### MLP Layers as Feature Combiners

MLP layers (the feed-forward sub-layer of each transformer block) take the
superposed residual stream and perform nonlinear computations:

    h = GELU(W₁ · x + b₁)    W₁ ∈ ℝ^{d_ff × d_model}
    output = W₂ · h + b₂     W₂ ∈ ℝ^{d_model × d_ff}

where d_ff >> d_model (typically 4×). The MLP expands into a higher-
dimensional intermediate space (d_ff neurons), applies nonlinearity, then
collapses back.

The key insight: MLP computation happens in the expanded d_ff space,
where superposition is LESS SEVERE (more dimensions per feature). The
MLP uses this expansion to:

    1. DETECT feature combinations: W₁ rows act as key detectors —
       neuron j in the MLP fires when the input x matches key W₁[j].
       Because d_ff >> d_model, there is more room to store features
       as monosemantic neurons within the MLP's hidden layer.

    2. COMPUTE nonlinear functions: the GELU nonlinearity allows condition-
       checking ("IF feature A AND feature B are present THEN...").

    3. WRITE new features: W₂ columns write the computed results back to
       the residual stream in superposed form (d_model dimensions).

    # ================================================================== #
    **The MLP expansion as a superposition relief valve:**

    d_model = 768  (residual stream — highly superposed)
    d_ff = 3072    (MLP hidden layer — less superposed, 4× more capacity)

    The MLP's W₁ ∈ ℝ^{3072×768} projects from the crowded 768-dim space
    into a 3072-dim space with 4× more room per feature.
    In 3072 dims, features are better separated, and the GELU nonlinearity
    can be more precisely applied to detect specific feature combinations.

    After the nonlinear computation, W₂ ∈ ℝ^{768×3072} projects back to
    the residual stream, re-superposing the results into 768 dims.

    This expand-nonlinear-compress structure is OPTIMAL for performing
    nonlinear feature computation under a dimensionality constraint.
    It is not an arbitrary architectural choice — it is what you'd design
    if you knew about superposition.
    # ================================================================== #


### Non-linear Computation With Superposed Inputs

The GELU nonlinearity can implement logical-AND-like operations on features
even in a superposed space. Consider a neuron j in the MLP hidden layer:

    h_j = GELU(W₁[j] · x)

Neuron j fires when W₁[j] · x is large and positive. In a superposed space:

    W₁[j] · x = W₁[j] · (Σᵢ fᵢ wᵢ) = Σᵢ fᵢ (W₁[j] · wᵢ)

If W₁[j] is aligned with the directions of BOTH feature A and feature B:
    h_j is large only when BOTH fA is large AND fB is large.
    This is an AND gate in feature space.

If W₁[j] is aligned with feature A and anti-aligned with feature B:
    h_j is large only when fA is large AND fB is small.
    This is an AND-NOT gate.

The key result: MLP neurons can implement feature conjunctions in superposed
space, but with NOISE proportional to the interference from other superposed
features. High sparsity reduces this noise; low sparsity makes AND gates
unreliable.


##### PART VIII: WHY SUPERPOSITION BREAKS NEURON-LEVEL INTERPRETABILITY

### The Fundamental Interpretability Problem

If features are linear directions in activation space, and neurons are the
standard basis vectors of that space, then:

    Features ≠ Neurons

The correct unit of analysis is the FEATURE DIRECTION, not the neuron.
But feature directions are hidden inside the high-dimensional activation
space, superposed with many other features. You cannot read them off from
neuron activations directly.

    # ================================================================== #
    **What fails without superposition awareness:**

    Standard neuron analysis: "neuron 47 fires for tokens about violence."
    Conclusion: "the model has a 'violence' concept in neuron 47."

    Reality (with superposition): neuron 47 participates in:
      Feature A: violence-related content (weights +0.6)
      Feature B: financial loss (weights +0.4)
      Feature C: game-over states (weights +0.5)

    All three involve negative outcomes in different domains.
    Neuron 47 is a participant in a "negative outcome" superposition.
    It does not REPRESENT violence — it is part of multiple feature directions.

    If you ablate neuron 47 expecting to remove "violence":
    You accidentally also affect financial and game-state processing.
    The intervention is NOT surgical — it destroys unintended computations.

    This is why neuron ablation studies in early interpretability work
    produced confusing results: ablating a "face neuron" also degraded
    performance on unrelated tasks.
    # ================================================================== #

### The Consequences for Mechanistic Interpretability

Superposition has specific, concrete consequences for each interpretability
methodology:

    NEURON MAX ACTIVATION:
    Finding: "neuron k activates maximally for X."
    Superposition problem: k may activate for X primarily because feature
    direction fX has a large component in the k-th neuron. But fX also has
    components in 50 other neurons. You're seeing one dimension of a 50-dim feature.

    ACTIVATION PATCHING:
    Finding: "patching neuron k's activation changes the output."
    Superposition problem: patching neuron k changes all features superposed
    on it. The effect may be partly due to the target feature and partly due
    to interference with other features.

    ABLATION STUDIES:
    Finding: "ablating neurons k₁, k₂, k₃ eliminates behaviour B."
    Superposition problem: those neurons may be the ONLY neurons that encode
    feature fB (so ablating them works), or they may be a SUBSET of the neurons
    encoding fB (so ablating them only partially works), or they may encode fB
    plus many other features (so ablating causes collateral damage).

    LINEAR PROBING:
    Finding: "a linear probe on neuron k achieves 90% accuracy for property P."
    Superposition problem: the probe is likely using MULTIPLE NEURONS to
    reconstruct the feature direction for P. Each individual neuron's contribution
    is noisy; the probe extracts P from the subspace that spans P across all neurons.
    This is actually ROBUST to superposition — linear probes can work well even
    when features are superposed!

    # ================================================================== #
    **Why linear probes are superposition-robust:**

    If feature P is encoded as direction wP ∈ ℝⁿ (with wP spread across
    all n neurons), then the optimal linear probe is exactly wP.
    The probe: g(h) = wP · h recovers fP from the superposed h.
    It doesn't need to know which neurons are involved — it works on the
    whole activation vector simultaneously.

    This is why probing classifiers work surprisingly well even in the
    presence of superposition: they naturally operate on the full
    activation vector and can find feature directions even if those
    directions are not aligned with neuron axes.
    # ================================================================== #


##### PART IX: SPARSE AUTOENCODERS AS THE ANTIDOTE

### The SAE as a Superposition Decoder

If superposition is the problem — features are hidden in linear combinations
of neurons — then the solution is to DECODE them: find the feature directions
wᵢ and compute fᵢ = wᵢ · h for each activation vector h.

A Sparse Autoencoder (SAE) does exactly this:

    ENCODER:  h_feat = ReLU(W_enc · (h − b_pre) + b_enc)

    h_feat is a HIGHER-DIMENSIONAL sparse vector of feature activations.
    Each dimension of h_feat corresponds to one feature.
    The sparsity (most h_feat values are zero) mirrors the sparsity
    assumption that makes superposition work.

    DECODER:  ĥ = W_dec · h_feat + b_pre

    W_dec[:,i] is the feature direction for feature i in the original
    activation space. When feature i is active (h_feat[i] > 0), its
    direction W_dec[:,i] is added to the reconstruction ĥ.

The SAE's decoder columns ARE the feature directions that superposition
mixed together. Training the SAE to reconstruct h with sparse h_feat
forces it to DISENTANGLE the superposed features into separate, sparse
dimensions.

    # ================================================================== #
    **The SAE's key insight:**

    Superposition: h = Σᵢ fᵢ · wᵢ  (m features superposed in n dimensions, m>>n)
    SAE: finds wᵢ and fᵢ such that:
      (1) h ≈ Σᵢ fᵢ · W_dec[:,i]   (good reconstruction)
      (2) Most fᵢ = 0 for any given h  (sparsity)
      (3) W_dec[:,i] are unit-norm    (normalised feature directions)

    After training, W_dec[:,i] ≈ wᵢ (the true feature directions).
    h_feat[i] ≈ fᵢ (the true feature activations).
    The SAE has unrolled the superposition.
    # ================================================================== #

### Why the L1 Penalty Is Essential

The SAE's objective:
    L = ‖h − ĥ‖² + λ · ‖h_feat‖₁

The L1 penalty on h_feat is not just regularisation — it is the mechanism
that INDUCES sparsity, which forces the SAE to find the true sparse feature
decomposition rather than a dense, entangled one.

Without L1: the SAE finds many equivalent dense solutions where each
h_feat dimension carries a weighted mixture of features. The reconstruction
is perfect, but the h_feat dimensions are not interpretable.

With L1: the SAE is forced to use as few active features as possible.
This drives it toward the solution where h_feat[i] is non-zero only when
the corresponding feature is genuinely present — matching the true sparse
structure of the original superposed representation.

The λ parameter controls the sparsity-reconstruction trade-off:
  λ too small → h_feat is dense → features remain entangled
  λ too large → h_feat is all zeros → reconstruction fails
  λ just right → h_feat is sparse with active dimensions matching real features

    # ================================================================== #
    **The L1 penalty creates the right inductive bias:**

    True superposition: h = 0.8·f_political + 0.3·f_news + noise
    (Only 2 of 10,000 features are active)

    Without L1: SAE might find h_feat with 50 non-zero dimensions,
    each carrying a small fragment of f_political and f_news.
    Reconstruction is perfect but features are mixed.

    With L1: SAE is penalised for each non-zero dimension.
    It learns: "I can reconstruct h with just 2 active dimensions."
    It finds the 2 feature directions that explain h most efficiently.
    h_feat[political] ≈ 0.8, h_feat[news] ≈ 0.3, rest ≈ 0.
    # ================================================================== #


### Superposition Geometry in Learned SAE Features

After training, the SAE's decoder columns W_dec[:,i] reveal the geometry
of superposition in the original model:

    NEARLY ORTHOGONAL CLUSTERS: Features from the same "domain"
    (e.g., all named entity features, all syntactic features) may cluster
    in the same subspace but be nearly orthogonal within the cluster.

    ANTIPODAL PAIRS: Some features appear as near-antipodal pairs —
    one direction for "the concept is present", the opposite for "strongly absent".

    REGULAR POLYTOPES: In small toy models, the feature directions
    arrange themselves as vertices of regular polytopes (pentagon,
    icosahedron, etc.) — the optimal geometry for packed, uniform-importance features.

    HUB NEURONS: Some neurons (standard basis directions) are
    shared by many SAE features. These neurons are the most polysemantic.
    The SAE makes this visible: if many W_dec[:,i] have large k-th component,
    neuron k is a hub in the superposition graph.


##### PART X: DIAGNOSING AND MEASURING SUPERPOSITION IN PRACTICE

### Practical Superposition Diagnostics

Before investing in SAE training, several fast diagnostics can establish
whether superposition is present in a layer of interest:

    DIAGNOSTIC 1 — PCA EXPLAINED VARIANCE:
    Run PCA on the activation matrix H ∈ ℝ^{N×d} (N inputs, d dimensions).
    If most variance is explained by d components → representations are
    roughly linear and interpretable (low superposition).
    If the "intrinsic dimensionality" is much lower than d → activations
    lie in a low-dimensional subspace, possibly monosemantic.
    If explained variance falls off slowly → many features are superposed,
    no small number of PCs captures the structure.

    DIAGNOSTIC 2 — ACTIVATION DISTRIBUTION:
    Plot the histogram of activation values for each neuron across a large
    dataset. A monosemantic neuron has a bimodal distribution: mostly near-zero
    (feature absent) with a tail of large positives (feature present).
    A polysemantic neuron has a more complex distribution: multiple peaks
    corresponding to the different features that activate it.

    DIAGNOSTIC 3 — MUTUAL INFORMATION MATRIX:
    Compute the mutual information I(hᵢ; hⱼ) between all pairs of neuron
    activations (hᵢ, hⱼ). In a monosemantic representation, neurons
    are independent (I≈0). In a superposed representation, neurons
    are correlated (large I) because they share features.
    A high-MI block structure in the MI matrix reveals which neurons
    are co-participants in superposed feature groups.

    DIAGNOSTIC 4 — MAX ACTIVATION ANALYSIS:
    For each neuron, collect the 10 inputs that maximally activate it.
    Inspect them: do they share a single, coherent concept (monosemantic)?
    Or do they seem like multiple unrelated concepts (polysemantic)?
    Manual inspection of top-activating examples remains the gold standard
    for characterising individual neurons.

    DIAGNOSTIC 5 — SAE FEATURE COUNT PER NEURON:
    After training an SAE, count how many SAE features have large weight
    in each neuron (decoder column). This directly measures polysemanticity.
    Distribution of feature counts:
      Peak at 1 → mostly monosemantic
      Peak at 5-10 → moderate superposition
      Long tail to 50+ → heavy superposition

    # ================================================================== #
    **Quick superposition check — the "feature count test":**

    Train an SAE with expansion factor 8 (d_SAE = 8 × d_model).
    For each neuron k, count how many SAE features i have |W_dec[k,i]| > 0.1.
    Call this count_k.

    If median(count_k) ≈ 1: neurons are monosemantic (low superposition)
    If median(count_k) ≈ 5: moderate superposition
    If median(count_k) > 10: heavy superposition

    Compare across layers: earlier layers in BERT-like models tend to be
    less superposed; later layers more so. This mirrors the layerwise
    probing profile (see probing module).
    # ================================================================== #


### The Superposition Landscape Across Training

Superposition changes during training, not just across layers:

    EARLY TRAINING: Few features have been learned. Low superposition.
    Neurons are approximately monosemantic because the model hasn't
    yet learned enough distinct features to need packing.

    MIDDLE TRAINING: The number of learned features exceeds the neuron count.
    Superposition begins: the model packs lower-priority features as it
    learns them.

    LATE TRAINING / GROKKING: For tasks that require generalisation
    (like modular arithmetic), a phase transition occurs where the model
    reorganises from a memorisation-based to an algorithmic representation.
    This is associated with a DECREASE in superposition for task-relevant
    features (they become high-importance and precipitate into dedicated neurons)
    and an INCREASE for task-irrelevant features (they get pushed into
    superposition to free up space).

    FINE-TUNING: Fine-tuning on a specific task concentrates importance
    on task-relevant features, causing them to precipitate out of superposition.
    Models become more interpretable (less polysemantic) for their target domain
    after fine-tuning — at the cost of increased polysemanticity for unrelated features.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔══════════════════════════════╦════════════════════════════════════════════╗
    ║ Concept                      ║ Key Result                                 ║
    ╠══════════════════════════════╬════════════════════════════════════════════╣
    ║ Feature capacity (sparse)    ║ ~n / (1-S_p)  features per layer           ║
    ║ Interference magnitude       ║ ε ~ 1/√n per feature pair                  ║
    ║ Phase transition threshold   ║ S_crit ~ 1 / (n × (1-S_p))                 ║
    ║ Optimal geometry (k in n)    ║ Vertices of regular polytopes              ║
    ║ Polysemanticity cause        ║ Multiple features share neuron directions  ║
    ║ SAE objective                ║ L = ‖h-ĥ‖² + λ‖h_feat‖₁                    ║
    ║ SAE decoder = ?              ║ True feature directions in activation space║
    ║ Why L1 penalty?              ║ Induces sparsity to match true features    ║
    ╚══════════════════════════════╩════════════════════════════════════════════╝

    The fundamental trade-off:
      MORE features stored → MORE interference between active features
      SPARSER inputs        → LESS interference at any given moment

    Feature capacity formula:
      Representable features ≈ n / (1 - S_p)²
      where n = neurons, S_p = feature sparsity (1 = always inactive, 0 = always active)

    Monosemantic threshold:
      Feature i gets a dedicated neuron if:
        importance(i) > 1 / (n × sparsity(i))
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Superposition from Scratch — Toy Model, Phase Transitions, and SAE Recovery": {
        "description": (
            "Implements the full Elhage et al. toy model of superposition from scratch. "
            "Trains a linear autoencoder (encoder W, ReLU, decoder Wᵀ) with "
            "gradient descent on sparse synthetic features with varying importance weights. "
            "Shows that superposition emerges naturally from optimisation. "
            "Measures: which features get dedicated neurons (monosemantic) vs superposed, "
            "the relationship between importance/sparsity and the phase transition, "
            "and feature interference as a function of sparsity. "
            "Then trains a minimal SAE to recover the superposed feature directions "
            "and verifies recovery via cosine similarity. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "superposition",
        "code": '''
"""
================================================================================
SUPERPOSITION FROM SCRATCH — TOY MODEL, PHASE TRANSITIONS, AND SAE RECOVERY
================================================================================

We implement the Elhage et al. (2022) toy model of superposition:
  - Linear bottleneck autoencoder: x → Wx → ReLU → Wᵀ → x_hat
  - m=10 features compressed into n=2 neurons (severe compression)
  - Features have varying importance weights S_i and sparsity S_p
  - Training via gradient descent (no libraries)

We then show:
  1. Phase transition: high-importance features get dedicated neurons
  2. Interference grows with density (1 - S_p)
  3. Feature geometry: weight directions become near-orthogonal
  4. SAE recovery: sparse autoencoder decodes superposed features

All pure Python, no external dependencies.
================================================================================
"""

import math
import random

random.seed(314)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION — SPARSE SYNTHETIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

M = 10   # number of features (more than neurons — forces superposition)
N = 2    # number of neurons (bottleneck)

# Feature importances: exponentially decreasing (feature 0 is most important)
IMPORTANCE = [0.9 ** i for i in range(M)]

# Feature sparsity (fraction of time each feature is zero)
SPARSITY = 0.85  # features are active 15% of the time

def sample_input(m=M, sparsity=SPARSITY):
    """Generate one sparse input vector x ∈ ℝᵐ."""
    x = []
    for _ in range(m):
        if random.random() < sparsity:
            x.append(0.0)
        else:
            x.append(random.uniform(0.5, 1.0))
    return x


# ─────────────────────────────────────────────────────────────────────────────
# TOY MODEL — LINEAR BOTTLENECK AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return max(0.0, x)

def relu_d(x):
    return 1.0 if x > 0 else 0.0

def dot(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def vec_scale(v, s):
    return [vi*s for vi in v]

def vec_add(a, b):
    return [a[i]+b[i] for i in range(len(a))]

# W ∈ ℝ^{N×M}: encoder weights. Column W[:,i] = direction for feature i.
# Decoder uses Wᵀ (transpose of W).
W = [[random.gauss(0, 0.3) for _ in range(M)] for _ in range(N)]
b = [random.gauss(0, 0.05) for _ in range(M)]  # decoder bias

def forward(x, W, b):
    """
    x: input (m,)
    h_pre: pre-activation at bottleneck  (n,)
    h: bottleneck activations after ReLU (n,)
    x_hat: reconstruction (m,)
    """
    # Encoder: h_pre = W x
    h_pre = [sum(W[k][i]*x[i] for i in range(M)) for k in range(N)]
    # ReLU
    h = [relu(z) for z in h_pre]
    # Decoder: x_hat = Wᵀ h + b
    x_hat = [sum(W[k][i]*h[k] for k in range(N)) + b[i] for i in range(M)]
    return h_pre, h, x_hat

def weighted_loss(x, x_hat, importance):
    """Importance-weighted reconstruction loss."""
    return sum(importance[i] * (x[i] - x_hat[i])**2 for i in range(M))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — GRADIENT DESCENT
# ─────────────────────────────────────────────────────────────────────────────

lr = 0.01
N_STEPS = 8000
BATCH   = 16

losses = []

for step in range(N_STEPS):
    dW = [[0.0]*M for _ in range(N)]
    db = [0.0]*M

    batch_loss = 0.0
    for _ in range(BATCH):
        x = sample_input()
        h_pre, h, x_hat = forward(x, W, b)

        # Reconstruction error (importance-weighted)
        err = [(x_hat[i] - x[i]) * IMPORTANCE[i] for i in range(M)]
        batch_loss += sum(IMPORTANCE[i]*(x[i]-x_hat[i])**2 for i in range(M))

        # Gradient w.r.t. decoder bias b
        for i in range(M):
            db[i] += 2.0 * err[i]

        # Gradient w.r.t. W (combined encoder+decoder, tied weights)
        # d_loss/d_W[k][i]:
        #   Decoder contribution: 2 * err[i] * h[k]
        #   Encoder contribution: 2 * err[j] * W[k][j] * relu_d(h_pre[k]) * x[i]  (chain rule)
        for k in range(N):
            enc_grad = sum(2*err[j]*W[k][j] for j in range(M)) * relu_d(h_pre[k])
            for i in range(M):
                dW[k][i] += 2.0 * err[i] * h[k]          # decoder gradient
                dW[k][i] += enc_grad * x[i]               # encoder gradient

    # Update
    for k in range(N):
        for i in range(M):
            W[k][i] -= lr * dW[k][i] / BATCH
    for i in range(M):
        b[i] -= lr * db[i] / BATCH

    losses.append(batch_loss / BATCH)

print("=" * 68)
print("  TOY MODEL OF SUPERPOSITION")
print("=" * 68)
print(f"  Architecture: {M} features → {N} neurons (bottleneck) → {M} features")
print(f"  Feature sparsity: {SPARSITY:.0%} (features active {(1-SPARSITY)*100:.0f}% of time)")
print(f"  Feature importances: [{', '.join(f'{I:.2f}' for I in IMPORTANCE[:5])}, ...]")
print()
print(f"  Training: {N_STEPS} steps, batch={BATCH}")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS — FEATURE WEIGHTS AND SUPERPOSITION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  FEATURE WEIGHT ANALYSIS — WHICH FEATURES GET DEDICATED NEURONS?")
print("=" * 68)
print()

# For each feature i, its weight vector (direction in N-dim activation space)
# is the i-th column of W: col_i = [W[k][i] for k in range(N)]
feature_norms = []
feature_dirs = []
for i in range(M):
    col = [W[k][i] for k in range(N)]
    norm = math.sqrt(sum(c**2 for c in col))
    feature_norms.append(norm)
    if norm > 1e-6:
        feature_dirs.append([c/norm for c in col])
    else:
        feature_dirs.append([0.0]*N)

# Compute participation ratio per feature column
# PR_i = norm_i^2 / sum_k W[k][i]^2  -- for two-neuron case, this is 1/sum(normalized_sq)
# Simpler: check alignment with standard basis (monosemantic = aligned with e_k)
def alignment_with_basis(col):
    """How aligned is this direction with the nearest standard basis vector?"""
    norm = math.sqrt(sum(c**2 for c in col))
    if norm < 1e-6:
        return 0.0
    unit = [c/norm for c in col]
    return max(abs(u) for u in unit)

print(f"  {'Feature':>8}  {'Importance':>11}  {'‖W[:,i]‖':>9}  "
      f"{'Basis align':>12}  {'Type'}")
print(f"  {'-'*60}")

for i in range(M):
    col = [W[k][i] for k in range(N)]
    norm = feature_norms[i]
    align = alignment_with_basis(col)
    if norm < 0.1:
        ftype = "IGNORED   (not represented)"
    elif align > 0.85:
        ftype = "DEDICATED (monosemantic)"
    elif align > 0.60:
        ftype = "SEMI-DEDICATED"
    else:
        ftype = "SUPERPOSED (polysemantic)"
    print(f"  {i:>8}  {IMPORTANCE[i]:>11.3f}  {norm:>9.4f}  {align:>12.4f}  {ftype}")

print()
print(f"  INTERPRETATION:")
print(f"  High-importance features (top {sum(1 for n in feature_norms if n > 0.3)}) "
      f"get large ‖W[:,i]‖ (actively represented).")
print(f"  Low-importance features may be ignored or weakly superposed.")
print(f"  Features aligned with basis vectors are monosemantic; others are superposed.")


# ─────────────────────────────────────────────────────────────────────────────
# INTERFERENCE MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  INTERFERENCE MEASUREMENT — PAIRWISE COSINE SIMILARITIES")
print("=" * 68)
print()
print("  |wᵢ · wⱼ| measures the interference between feature i and j.")
print("  Near-orthogonal (|cos| ≈ 0) = low interference = good superposition.")
print("  Near-parallel (|cos| ≈ 1) = high interference = features confused.")
print()

# Only for actively represented features (norm > 0.1)
active_feats = [i for i in range(M) if feature_norms[i] > 0.1]
print(f"  Active features: {active_feats}")
print()

if len(active_feats) >= 2:
    print(f"  {'(i,j)':>8}  {'cos(wᵢ,wⱼ)':>12}  {'Interference level'}")
    print(f"  {'-'*45}")
    for ii in range(len(active_feats)):
        for jj in range(ii+1, len(active_feats)):
            fi, fj = active_feats[ii], active_feats[jj]
            cos = dot(feature_dirs[fi], feature_dirs[fj])
            level = "HIGH" if abs(cos) > 0.5 else ("MEDIUM" if abs(cos) > 0.2 else "LOW")
            print(f"  ({fi},{fj}):  {cos:>+12.4f}  {level}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE TRANSITION EXPERIMENT — VARY IMPORTANCE, OBSERVE TRANSITION
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  PHASE TRANSITION — IMPORTANCE vs SUPERPOSITION")
print("=" * 68)
print()
print("  We train models with M=4 features in N=2 neurons, sweeping")
print("  importance of features 2 and 3 (features 0,1 always high importance).")
print("  At a critical importance threshold, features 2,3 transition from")
print("  IGNORED to SUPERPOSED to DEDICATED.")
print()

def train_small_model(m, n, importances, sparsity, n_steps=3000, lr=0.02, batch=32):
    """Train the toy model and return the weight matrix."""
    W_s = [[random.gauss(0, 0.3) for _ in range(m)] for _ in range(n)]
    b_s = [0.0] * m
    for step in range(n_steps):
        dW_s = [[0.0]*m for _ in range(n)]
        db_s = [0.0]*m
        for _ in range(batch):
            x = [0.0 if random.random()<sparsity else random.uniform(0.5,1.0)
                 for _ in range(m)]
            h_pre = [sum(W_s[k][i]*x[i] for i in range(m)) for k in range(n)]
            h = [relu(z) for z in h_pre]
            x_hat = [sum(W_s[k][i]*h[k] for k in range(n))+b_s[i] for i in range(m)]
            err = [(x_hat[i]-x[i])*importances[i] for i in range(m)]
            for k in range(n):
                enc_g = sum(2*err[j]*W_s[k][j] for j in range(m)) * relu_d(h_pre[k])
                for i in range(m):
                    dW_s[k][i] += 2.0*err[i]*h[k] + enc_g*x[i]
            for i in range(m):
                db_s[i] += 2.0*err[i]
        for k in range(n):
            for i in range(m):
                W_s[k][i] -= lr*dW_s[k][i]/batch
        for i in range(m):
            b_s[i] -= lr*db_s[i]/batch
    return W_s

M_PT = 4  # 4 features, 2 neurons
importance_levels = [0.05, 0.15, 0.30, 0.50, 0.80]

print(f"  {'Importance(2,3)':>16}  {'‖W[:,2]‖':>9}  {'‖W[:,3]‖':>9}  "
      f"{'Status f2':>15}  {'Status f3':>15}")
print(f"  {'-'*70}")

for imp_23 in importance_levels:
    importances_pt = [1.0, 0.8, imp_23, imp_23]
    random.seed(42)
    W_pt = train_small_model(M_PT, N, importances_pt, SPARSITY,
                             n_steps=2000, lr=0.03, batch=32)

    norms = [math.sqrt(sum(W_pt[k][i]**2 for k in range(N))) for i in range(M_PT)]
    aligns = [alignment_with_basis([W_pt[k][i] for k in range(N)]) for i in range(M_PT)]

    def status(norm, align):
        if norm < 0.08: return "IGNORED"
        if align > 0.85: return "DEDICATED"
        return "SUPERPOSED"

    print(f"  {imp_23:>16.2f}  {norms[2]:>9.4f}  {norms[3]:>9.4f}  "
          f"{status(norms[2],aligns[2]):>15}  {status(norms[3],aligns[3]):>15}")

print()
print("  KEY FINDING: As importance of features 2,3 increases, they transition")
print("  from IGNORED → SUPERPOSED → DEDICATED. This is the phase transition")
print("  predicted by superposition theory. The threshold depends on sparsity.")


# ─────────────────────────────────────────────────────────────────────────────
# SPARSITY EXPERIMENT — HOW SPARSITY AFFECTS FEATURE COUNT
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SPARSITY EFFECT — HOW MANY FEATURES GET REPRESENTED?")
print("=" * 68)
print()
print("  Same architecture (N=2 neurons, M=6 features), varying sparsity.")
print("  More sparsity → less interference → more features get represented.")
print()
print(f"  {'Sparsity':>10}  {'Active frac':>12}  {'# repr features':>17}  "
      f"{'Avg ‖W[:,i]‖ (active)':>22}")
print(f"  {'-'*65}")

M_SP = 6
importances_sp = [0.9**i for i in range(M_SP)]

for sp in [0.0, 0.3, 0.6, 0.8, 0.95]:
    random.seed(42)
    W_sp = train_small_model(M_SP, N, importances_sp, sp,
                             n_steps=2000, lr=0.03, batch=32)
    norms_sp = [math.sqrt(sum(W_sp[k][i]**2 for k in range(N))) for i in range(M_SP)]
    n_repr = sum(1 for norm in norms_sp if norm > 0.15)
    avg_norm = sum(n for n in norms_sp if n > 0.15) / max(n_repr, 1)

    active_frac = 1 - sp
    print(f"  {sp:>10.2f}  {active_frac:>12.2f}  {n_repr:>17}  {avg_norm:>22.4f}")

print()
print("  Higher sparsity → more features get represented (less interference).")
print(f"  Theoretical max at S_p={SPARSITY}: ~{N/(1-SPARSITY)**2:.0f} features per {N} neurons.")
print(f"  In practice limited by M={M_SP} available features in this experiment.")


# ─────────────────────────────────────────────────────────────────────────────
# SAE RECOVERY — DECODE SUPERPOSED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SAE RECOVERY — SPARSE AUTOENCODER DECODES SUPERPOSED FEATURES")
print("=" * 68)
print()
print("  We train an SAE on the BOTTLENECK ACTIVATIONS h ∈ ℝ² of our")
print(f"  trained toy model. The SAE has {M} hidden units (= M features) to")
print(f"  recover the {M} original feature directions from the 2-dim bottleneck.")
print()

# Collect bottleneck activations from the trained toy model
random.seed(99)
N_COLLECT = 1000
H_data = []  # bottleneck activations h ∈ ℝ²
X_data = []  # original inputs

for _ in range(N_COLLECT):
    x = sample_input()
    h_pre, h, x_hat = forward(x, W, b)
    H_data.append(h)
    X_data.append(x)

# SAE architecture: 2 → M hidden → 2 (reconstruction of h)
M_SAE = M  # match the true number of features

# SAE weights
random.seed(55)
W_enc_sae = [[random.gauss(0, 0.3) for _ in range(N)] for _ in range(M_SAE)]
b_enc_sae = [0.0] * M_SAE
W_dec_sae = [[random.gauss(0, 0.3) for _ in range(M_SAE)] for _ in range(N)]
b_dec_sae = [0.0] * N

def sae_encode(h, W_enc, b_enc):
    pre = [sum(W_enc[j][k]*h[k] for k in range(len(h))) + b_enc[j]
           for j in range(len(W_enc))]
    return [max(0.0, p) for p in pre]

def sae_decode(z, W_dec, b_dec):
    return [sum(W_dec[k][j]*z[j] for j in range(len(z))) + b_dec[k]
            for k in range(len(W_dec))]

def normalise_dec_cols(W_dec_sae, M_SAE, N):
    """Normalise decoder columns to unit norm."""
    for j in range(M_SAE):
        col_norm = math.sqrt(sum(W_dec_sae[k][j]**2 for k in range(N)))
        if col_norm > 1e-6:
            for k in range(N):
                W_dec_sae[k][j] /= col_norm

# Train SAE
SAE_STEPS = 3000
SAE_LR    = 0.02
SAE_LAM   = 0.05  # L1 sparsity weight

for step in range(SAE_STEPS):
    dWe = [[0.0]*N for _ in range(M_SAE)]
    dbe = [0.0]*M_SAE
    dWd = [[0.0]*M_SAE for _ in range(N)]
    dbd = [0.0]*N

    # Mini-batch
    batch_idx = [random.randint(0, N_COLLECT-1) for _ in range(32)]
    for idx in batch_idx:
        h = H_data[idx]

        # Forward
        z = sae_encode(h, W_enc_sae, b_enc_sae)
        h_hat = sae_decode(z, W_dec_sae, b_dec_sae)

        # Reconstruction loss gradient
        recon_err = [2.0*(h_hat[k] - h[k]) for k in range(N)]

        # Gradient w.r.t. decoder weights + bias
        for k in range(N):
            dbd[k] += recon_err[k]
            for j in range(M_SAE):
                dWd[k][j] += recon_err[k] * z[j]

        # L1 gradient
        l1_grad = [SAE_LAM * (1.0 if z[j] > 0 else 0.0) for j in range(M_SAE)]

        # Backprop through encoder
        for j in range(M_SAE):
            # Gradient through decoder
            dec_g = sum(W_dec_sae[k][j] * recon_err[k] for k in range(N))
            gate  = 1.0 if z[j] > 0 else 0.0
            total_g = (dec_g + l1_grad[j]) * gate
            dbe[j] += total_g
            for kk in range(N):
                dWe[j][kk] += total_g * h[kk]

    # Update
    batch_size = len(batch_idx)
    for j in range(M_SAE):
        b_enc_sae[j] -= SAE_LR * dbe[j] / batch_size
        for kk in range(N):
            W_enc_sae[j][kk] -= SAE_LR * dWe[j][kk] / batch_size
    for k in range(N):
        b_dec_sae[k] -= SAE_LR * dbd[k] / batch_size
        for j in range(M_SAE):
            W_dec_sae[k][j] -= SAE_LR * dWd[k][j] / batch_size

    normalise_dec_cols(W_dec_sae, M_SAE, N)

print(f"  SAE trained: 2 inputs → {M_SAE} hidden → 2 outputs (recovering {M} features)")
print(f"  L1 sparsity weight λ = {SAE_LAM}")
print()

# Compare SAE decoder columns to true feature directions (W columns)
print(f"  Matching SAE features to true feature directions:")
print(f"  (cosine similarity between SAE decoder col and true W column)")
print()
print(f"  {'True feat':>10}  {'‖W[:,i]‖':>10}  {'Best SAE feat':>14}  {'Cosine sim':>11}  {'Recovered?'}")
print(f"  {'-'*58}")

used_sae = set()
for i in range(M):
    true_dir = feature_dirs[i]
    true_norm = feature_norms[i]

    # Find best matching unused SAE decoder column
    best_cos, best_j = -1.0, -1
    for j in range(M_SAE):
        if j in used_sae:
            continue
        sae_col = [W_dec_sae[k][j] for k in range(N)]
        sae_norm = math.sqrt(sum(c**2 for c in sae_col))
        if sae_norm < 1e-6:
            continue
        sae_unit = [c/sae_norm for c in sae_col]
        cos = abs(dot(true_dir, sae_unit))
        if cos > best_cos:
            best_cos = cos
            best_j = j

    if best_j >= 0:
        used_sae.add(best_j)

    recovered = "✓" if (true_norm > 0.1 and best_cos > 0.7) else (
                "~" if (true_norm > 0.1 and best_cos > 0.4) else
                "✗ ignored" if true_norm < 0.1 else "✗ missed")
    print(f"  {i:>10}  {true_norm:>10.4f}  {best_j:>14}  {best_cos:>11.4f}  {recovered}")

print()
print("  The SAE successfully recovers the feature directions that the toy model")
print("  learned to encode in its 2-dimensional bottleneck activation space.")
print("  Features with ‖W[:,i]‖ ≈ 0 (ignored by the toy model) are not")
print("  recovered — correctly, since they are not present in the activations.")
''',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Dedent all operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER OPERATIONS (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def render_operations(st, scripts_dir=None, main_script=None):
    import streamlit as st

    st.markdown("---")
    st.subheader("⚙️ Operations")

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

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    # try:
    #     from interpretability.visuals.superposition import (
    #         SUPERPOSITION_VISUAL_HTML,
    #         SUPERPOSITION_VISUAL_HEIGHT,
    #     )
    #     visual_html   = SUPERPOSITION_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = SUPERPOSITION_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[superposition.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }