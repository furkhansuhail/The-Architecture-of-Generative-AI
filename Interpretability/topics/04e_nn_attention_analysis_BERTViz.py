"""
Attention Analysis — BERTViz and Beyond
=========================================

Attention analysis interrogates the multi-head self-attention weights inside
transformer models (BERT, GPT, T5, LLaMA) to understand which tokens the model
"looks at" when processing each position. BERTViz (Vig, 2019) is the canonical
visualisation tool: it renders the full attention tensor as interactive arc
diagrams across all layers and all heads simultaneously.

But attention analysis is also one of the most contested areas in interpretability.
A landmark 2019 paper showed that attention weights can be manipulated without
changing model behaviour, raising the question of whether attention constitutes
a faithful explanation at all — or merely a plausible-looking one.

This module covers the mathematics of transformer attention, BERTViz's three
visualisation modes, the theoretical critique of attention-as-explanation,
principled extensions (attention rollout, gradient-weighted attention),
and the honest limits of what attention analysis can and cannot tell you.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Attention Analysis — BERTViz and the Limits of Attention"
DISPLAY_NAME = "04e · Attention Analysis"
ICON = "🔭"
SUBTITLE = "BERTViz, attention rollout, and the attention-is-not-explanation debate"


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

### The Appeal and the Problem

When BERT (Devlin et al., 2018) was released, its attention weights became
objects of intense fascination. Researchers printed attention patterns and
found seemingly interpretable structures: some heads tracked syntactic
dependencies, some heads aligned pronouns to their antecedents, some heads
appeared to attend to punctuation, some to rare tokens.

The visualisations were beautiful. Arcs connecting "it" to "animal" in a
coreference resolution example seemed to show the model reasoning. Papers
were written with titles like "What does BERT look at?" and "Are Sixteen
Heads Really Better Than One?"

Then, in 2019, Jain & Wallace published "Attention is not Explanation" —
a paper that showed attention weights can be replaced with arbitrary
adversarially-chosen weights without changing the model's predictions or
intermediate representations. If the explanation can be changed without
changing the prediction, it is not explaining the prediction.

This module builds the mathematical foundation for understanding BOTH why
attention is appealing AND why it is dangerous to naively equate attention
with explanation — and what principled alternatives exist.


##### PART I: THE MATHEMATICS OF SELF-ATTENTION

### Scaled Dot-Product Attention

The fundamental operation in all modern transformer architectures is
scaled dot-product attention. Given a sequence of n tokens, each
represented as a d-dimensional vector, the input matrix X ∈ ℝⁿˣᵈ is
projected into three spaces:

    Q = X Wᴬ_Q ∈ ℝⁿˣᵈₖ    (Queries  — "what am I looking for?")
    K = X Wᴬ_K ∈ ℝⁿˣᵈₖ    (Keys    — "what do I contain?")
    V = X Wᴬ_V ∈ ℝⁿˣᵈᵥ    (Values  — "what do I send if attended to?")

The attention pattern and output are then:

    Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V

Breaking this apart:

    STEP 1 — SCORE MATRIX:
    S = QKᵀ / √dₖ ∈ ℝⁿˣⁿ

    Sᵢⱼ = (Qᵢ · Kⱼ) / √dₖ

    The raw score between position i (query) and position j (key).
    The dot product Qᵢ · Kⱼ measures alignment between query vector i
    and key vector j. The √dₖ scaling prevents the dot products from
    growing large when dₖ is large, which would push softmax into
    saturation regions with near-zero gradients.

    STEP 2 — ATTENTION WEIGHTS:
    A = softmax(S) ∈ ℝⁿˣⁿ

    Aᵢⱼ = exp(Sᵢⱼ) / Σₖ exp(Sᵢₖ)

    The attention matrix A. Each row i sums to 1 (it is a probability
    distribution over positions j). Aᵢⱼ is the attention that position i
    pays to position j. This is the quantity that BERTViz visualises.

    STEP 3 — OUTPUT:
    output_i = Σⱼ Aᵢⱼ · Vⱼ  ∈ ℝᵈᵥ

    The output at position i is a weighted sum of all value vectors,
    weighted by the attention that position i pays to each position j.

    # ================================================================== #
    **The query-key-value intuition — a library analogy:**

    Query Qᵢ:  "I am position i, I am looking for information about X"
               (the question you bring to the library catalogue)

    Key Kⱼ:   "I am position j, I contain information about Y"
               (the label on the library book's spine)

    Score Sᵢⱼ: how well the query for X matches the key for Y
               (how relevant is book j's topic to your query i?)

    Attention Aᵢⱼ: normalised probability of choosing book j
                   (after comparing all spines, how often do you pick j?)

    Value Vⱼ:  the actual content of position j to be read
               (the full text inside book j)

    Output:    a weighted average of all books' content, proportional to
               how relevant each book's label was to the query.

    The critical point: Value Vⱼ and Key Kⱼ are derived from the SAME
    input token j, but via DIFFERENT projection matrices. What the key
    "advertises" may be very different from what the value "contains."
    # ================================================================== #


### Multi-Head Attention

Real transformers use h parallel attention "heads," each with its own
projection matrices:

    head_k = Attention(X Wᴬ_Q^k, X Wᴬ_K^k, X Wᴬ_V^k)    for k=1,...,h

    MultiHead(X) = Concat(head_1, ..., head_h) × W_O

Each head specialises in a different type of relationship. The output
projection W_O ∈ ℝ^{h·dᵥ × d} combines the h heads' outputs back
into the model's d-dimensional representation space.

BERT-base: h=12 heads per layer, 12 layers → 144 attention matrices,
each 512×512 for standard sequence lengths.

This is the attention tensor BERTViz visualises: a (L, H, n, n) tensor
where L=layers, H=heads, n=sequence length. For BERT-base on a sentence
of 20 tokens: 12 × 12 × 20 × 20 = 57,600 attention weights. BERTViz
must make this navigable.

    # ================================================================== #
    **Why multiple heads? The specialisation hypothesis:**

    Head 1 (in some trained BERT models): attends to the next token
      [the, cat, sat, on, the, mat]
      Each token mostly attends to the word that follows it.
      This head encodes local positional information.

    Head 5 (in some trained BERT models): attends to coreferents
      "The animal didn't cross the street because it was too tired."
      "it" attends strongly to "animal" — tracking the antecedent.

    Head 9 (in some trained BERT models): attends to the [CLS] token
      Almost every token pays heavy attention to the special classification
      token, which aggregates sequence-level information.

    Different heads learn different relationship types. BUT: this
    specialisation is an empirical observation on some heads in some models.
    Many heads do not have clean, human-interpretable roles.
    # ================================================================== #


### The Transformer Layer — Attention in Context

Attention is only one component of each transformer layer. The full
computation in one transformer encoder layer is:

    1. Layer Norm:       x̃ = LayerNorm(x)
    2. Multi-Head Attn:  a = MultiHead(x̃) × W_O
    3. Residual:         x' = x + a        ← CRUCIAL: attention output is ADDED
    4. Layer Norm:       x̃' = LayerNorm(x')
    5. Feed-Forward:     f = FFN(x̃')       (two linear layers + GELU)
    6. Residual:         x'' = x' + f      ← another residual connection

The presence of RESIDUAL CONNECTIONS is crucial for understanding why
raw attention weights can mislead. Information can flow through the
residual stream ENTIRELY BYPASSING the attention mechanism — without
leaving any trace in the attention weights. A token's final representation
may be dominated by its own input (via residual) with attention providing
only a minor correction, yet the attention weights cannot tell you this.


##### PART II: BERTviz — THE THREE VISUALISATION MODES

### BERTViz Overview

BERTViz (Vig, 2019) is an interactive tool for visualising the full
attention tensor of any transformer model (BERT, GPT-2, RoBERTa, etc.).
It provides three complementary views of the same attention data, each
designed to answer a different question.

    # ================================================================== #
    **The three views — which question each answers:**

    Head View:    "For one specific head in one layer, what does attention
                  look like?"
                  Fine-grained: shows every token-to-token attention weight
                  as an arc, for one (layer, head) pair at a time.

    Model View:   "Across all heads and layers, what is the overall
                  attention pattern?"
                  Coarse-grained: shows one mini-heatmap per (layer, head)
                  pair, 144 total for BERT-base. Good for finding which
                  head/layer combinations warrant closer inspection.

    Neuron View:  "What do the individual query and key vectors look like
                  for each token, and how do their dot products produce
                  the attention weights?"
                  The most mechanistic: shows the actual Q and K vector
                  components for each token, and the element-wise products
                  that produce the attention score.
    # ================================================================== #


### The Head View

The head view shows the attention pattern for ONE (layer, head) pair.
Tokens of the input sequence appear twice: on the left (query side) and
on the right (key side). For each query token on the left, lines are drawn
to all key tokens on the right, with line thickness proportional to the
attention weight Aᵢⱼ.

Reading a head view:
  • THICK line from token i to token j: high attention weight Aᵢⱼ.
  • THIN lines from token i to many j: diffuse, spread-out attention.
  • DIAGONAL: attention to self (Aᵢᵢ high, common in some heads).
  • VERTICAL STACK to one token: many tokens attending to the same target.

Pattern taxonomy (Kovaleva et al., 2019 — "Revealing the Dark Secrets of BERT"):
    VERTICAL:   Most tokens attend to a single position (e.g., [CLS], [SEP], or
                a period). The column for that position is thick.
    DIAGONAL:   Each token attends primarily to itself (Aᵢᵢ ≈ 1).
    BLOCK:      The sequence is split into blocks, and tokens attend within
                their block. Common in models processing structured inputs.
    HETEROGENEOUS: No clear pattern — each token has its own distinct
                attention distribution. These are the most interesting heads.
    VERTICAL STRIPES: Certain query positions have very concentrated attention,
                others have very diffuse attention.

The crucial observation from Kovaleva et al.: the MAJORITY of BERT heads
show vertical or diagonal patterns — attending uniformly to one or a few
special tokens, or to themselves. These patterns contain very little
information about the specific input. Only a minority of heads show the
heterogeneous, input-sensitive patterns that humans might call "reasoning."


### The Model View

The model view shows all (layer, head) pairs simultaneously. Each of the
L × H attention matrices is rendered as a tiny thumbnail heatmap. The user
can identify which heads show interesting patterns at a glance, then click
through to the head view for those specific (layer, head) pairs.

Common findings from model-view exploration:
  • Later layers tend to have more heterogeneous attention patterns.
  • Earlier layers often show uniform or diagonal patterns.
  • [CLS] and [SEP] are dominant recipients of attention across many heads.
  • Individual layers specialise in specific relationship types.

The model view makes it practical to search 144 attention matrices for
patterns — without it, attention analysis of BERT-scale models would be
impossibly tedious.


### The Neuron View — Attention Through Vectors

The neuron view shows the MECHANISM behind one attention weight: the
individual components of the query and key vectors and their element-wise
products.

For tokens i (query) and j (key) in head k:
  Qᵢ^k ∈ ℝᵈₖ   (query vector for token i in head k)
  Kⱼ^k ∈ ℝᵈₖ   (key vector for token j in head k)

The attention score Sᵢⱼ = Qᵢ^k · Kⱼ^k / √dₖ = (1/√dₖ) Σₘ Qᵢₘ × Kⱼₘ

The neuron view shows:
  • Each dimension m of Qᵢ and Kⱼ as colour-coded bars (red = positive, blue = negative).
  • The product Qᵢₘ × Kⱼₘ for each dimension, showing which dimensions
    contribute positively (same sign) or negatively (opposite signs) to
    the dot product.

This view answers: "WHY is position i attending to position j?" at the
level of query-key alignment across vector dimensions. It reveals the
specific feature dimensions (in Q-K space) that drive a high attention weight.

The neuron view is the only view that exposes the internal structure of the
attention computation rather than just its output.

    # ================================================================== #
    **What the neuron view reveals — an example:**

    Attention from "it" → "animal" in coreference:
    Dimension 17 of Q["it"]:  +0.8  (high positive)
    Dimension 17 of K["animal"]: +0.6  (high positive)
    Product dim 17: +0.8 × +0.6 = +0.48  ← strong positive contribution

    Dimension 31 of Q["it"]:  +0.3
    Dimension 31 of K["the"]: −0.7
    Product dim 31: +0.3 × −0.7 = −0.21  ← this pushes AGAINST "the"

    Different dimensions specialise in different types of compatibility.
    Dimension 17 might encode "noun/animate" compatibility.
    The neuron view makes this visible without requiring interpretability
    of what each dimension means — it shows the computation, not the semantics.
    # ================================================================== #


##### PART III: ATTENTION IS NOT EXPLANATION — THE CORE CRITIQUE

### The Jain & Wallace (2019) Argument

Jain & Wallace ("Attention is not Explanation," 2019) made two key
empirical claims:

    CLAIM 1 — ATTENTION DOES NOT CORRELATE WITH FEATURE IMPORTANCE:
    For classification tasks, the correlation between attention weights
    and gradient-based feature importance measures (which have stronger
    theoretical grounding) is often low. High-attention tokens are not
    reliably the tokens that most influence the prediction.

    CLAIM 2 — ADVERSARIAL ATTENTION EXISTS:
    For many trained models, you can find ALTERNATIVE attention weight
    distributions that are (a) totally different from the trained weights
    but (b) produce nearly identical model predictions and identical
    intermediate representations.

If Claim 2 holds, then the model's prediction can be "explained" by
completely different attention patterns, meaning no specific attention
pattern IS the explanation. Attention is not uniquely identifying the
model's reasoning — many equally-valid-looking attention patterns exist.

    # ================================================================== #
    **The adversarial attention construction:**

    Given: trained model with attention weights A (n×n matrix per head).
    Goal: find A* such that:
      (a) A* ≠ A (different distributions)
      (b) The output produced by A* ≈ output produced by A.

    Method: search for A* that maximises total variation distance from A
    (making it maximally different), subject to the constraint that the
    weighted sum A*·V produces a representation close to A·V.

    Key insight: if the value vectors V₁, ..., Vₙ are nearly collinear
    (pointing in similar directions), then many different weight distributions
    A* can produce approximately the same weighted average A*·V ≈ A·V.
    The attention weight distribution is NOT uniquely determined by the
    output — it has degrees of freedom.

    Jain & Wallace found that adversarial attention exists for most
    of their tested models and tasks. This demonstrated that the specific
    attention pattern A the model learned is NOT the only pattern that
    produces that model behaviour — it is one of many.
    # ================================================================== #


### Wiegreffe & Pinter (2019) — The Rebuttal

Wiegreffe & Pinter ("Attention is not not Explanation," 2019) challenged
both claims with a more nuanced analysis:

    REBUTTAL TO CLAIM 1:
    The low correlation between attention and gradient importance is partly
    expected — they measure different things. Gradients measure sensitivity
    (marginal effect of small changes). Attention measures WHICH TOKENS are
    combined when forming the output representation. A token can be highly
    attended to AND have low gradient importance if many tokens provide the
    same information (the model is redundant, not the attention meaningless).

    REBUTTAL TO CLAIM 2:
    While adversarial attention exists for the OUTPUT VALUE of A·V, it may
    not exist for the MODEL'S DECISION BOUNDARY. They showed that training
    a model to make the adversarial attention A* also change the output
    (rather than just changing the weights) requires significantly different
    weights — not just a reweighting of the same value vectors.

    Their conclusion: attention is not sufficient as an explanation
    (it can be non-unique), but it is not vacuous either. For models where
    the value vectors are NOT collinear (i.e., the attention weight distribution
    matters for which information is combined), attention IS informative.

The debate remains open. The current consensus in the research community:

    ATTENTION IS A DESCRIPTION OF THE MECHANISM, not the explanation
    of the outcome. It tells you WHICH TOKENS were combined, not
    WHETHER combining them caused the prediction to be what it is.

    Using attention to explain predictions requires the additional
    assumption that: "if token j was attended to, then token j
    causally influenced the prediction." This assumption is often wrong.


### The Collinearity Problem — When Attention Cannot Explain

The fundamental mathematical reason attention can fail as an explanation:

    output_i = Σⱼ Aᵢⱼ · Vⱼ

If V₁ ≈ V₂ ≈ ... ≈ Vₙ (all value vectors are approximately equal),
then output_i ≈ (Σⱼ Aᵢⱼ) · V̄ = V̄ (since Σⱼ Aᵢⱼ = 1).
The output does not depend on A at all! Any attention distribution
produces the same output.

In practice, value vectors are not perfectly collinear, but they can be
highly correlated in some layers. In those layers, the model's output
is largely determined by the value vectors (which encode token content),
not by which specific tokens were attended to. Attention weights are
irrelevant to the prediction in those cases.

This is not a failure of the model — it is a valid strategy. The model
may use the residual connection to carry most of the information and
use attention to make small contextual adjustments. But it makes
attention-as-explanation fundamentally unreliable without knowing
whether the value vectors are collinear in the relevant head.

    # ================================================================== #
    **Collinearity spectrum:**

    ALL V identical (V₁=V₂=...=Vₙ):
    Attention is completely irrelevant. Any A gives the same output.
    Explanation via attention: MEANINGLESS.

    V vectors orthogonal (V₁ ⊥ V₂ ⊥ ...):
    Each token contributes a unique component to the output.
    Attention weight Aᵢⱼ exactly controls token j's contribution.
    Explanation via attention: VALID.

    Real BERT (somewhere in between):
    Some heads have nearly collinear V → attention not reliable there.
    Some heads have diverse V → attention is informative there.
    Without checking collinearity per head, you cannot know which case you are in.
    # ================================================================== #


##### PART IV: ATTENTION ROLLOUT — ACCOUNTING FOR RESIDUAL CONNECTIONS

### The Residual Connection Problem

Raw attention weights from layer l only show how layer l's attention
mechanism combines information. They ignore:

    1. RESIDUAL CONNECTIONS: Information skips each attention layer via
       the shortcut connection x' = x + attention_output. At every layer,
       the input ITSELF is added to the attention output. This means
       each token's representation always contains a strong component of
       its own previous representation — regardless of attention.

    2. LAYER COMPOSITION: When information from token j reaches token i
       via attention in layer l, that information in token j already
       reflects attention from layers 1 through l-1. You cannot understand
       "what information is in token j" from layer l alone — it is a
       composition of all previous layers' attention patterns.

### Attention Rollout (Abnar & Zuidema, 2020)

Attention Rollout addresses both problems by computing the EFFECTIVE
attention from the input tokens to each output position, accounting for
residual connections and composing through all layers.

    STEP 1 — Add identity matrix for residual:
    At each layer l, the residual connection means that position i also
    "attends to itself" with weight 1. Augmented attention:

        Ãˡ = 0.5 × Aˡ + 0.5 × I

    The 0.5 weights are approximate (assuming equal contribution from
    attention and residual), but empirically reasonable. The identity I
    represents the residual path: every token always has full access to
    its own previous representation.

    STEP 2 — Average over heads at each layer:
    For multi-head attention:
        Āˡ = (1/H) Σₕ Aˡʰ + I
        Ãˡ = Āˡ / Σⱼ (Āˡ)ᵢⱼ   (renormalise to sum to 1 per row)

    STEP 3 — Compose across layers:
    The effective attention through all L layers is the MATRIX PRODUCT
    of the augmented attention matrices:

        Rollout = Ã^L × Ã^{L-1} × ... × Ã^1

    The (i,j) entry of Rollout is the effective attention that output
    position i pays to input position j, accounting for all intermediate
    hops through layers and residual connections.

    # ================================================================== #
    **Why matrix multiplication composes attention:**

    Layer 1: position 3 attends 50% to position 1, 50% to position 2
             (Ã^1_{3,1} = 0.5, Ã^1_{3,2} = 0.5)

    Layer 2: position 5 attends 80% to position 3, 20% to position 4
             (Ã^2_{5,3} = 0.8, Ã^2_{5,4} = 0.2)

    Rollout_{5,1} = Ã^2_{5,3} × Ã^1_{3,1} + Ã^2_{5,4} × Ã^1_{4,1}
                  = 0.8 × 0.5 + 0.2 × 0 = 0.40

    Position 5 has 40% effective attention to position 1, computed by
    tracing ALL PATHS from 5 to 1 through the layer composition.
    Raw attention at layer 2 would show 0 — position 5 doesn't directly
    attend to position 1 in layer 2. Rollout correctly reveals the indirect
    path: 5→3 (layer 2) → 1 (layer 1).
    # ================================================================== #


### Limitations of Attention Rollout

    APPROXIMATION ONLY:
    The equal 0.5/0.5 weighting between attention and residual is arbitrary.
    The true ratio depends on the learned weights and varies per layer.
    Rollout gives a heuristic, not a faithful computation.

    IGNORES FEED-FORWARD NETWORKS:
    The FFN sub-layers within each transformer layer are not accounted for.
    They can substantially transform representations, changing which input
    information survives to the next attention layer.

    IGNORES NON-LINEARITIES:
    LayerNorm and GELU within FFN layers are non-linear. Rollout assumes
    linear composition (matrix multiply), which is exact only for linear
    transformations.

    POSITIVE ONLY:
    Attention weights are non-negative (softmax output). Rollout cannot
    capture negative information flow — cases where attending to token j
    DECREASES a feature's activation. Gradient-based methods can.

    Despite these limitations, rollout consistently outperforms raw
    attention on token importance benchmarks, suggesting the residual
    connection correction is the most important fix.


##### PART V: GRADIENT-WEIGHTED ATTENTION — INJECTING FAITHFULNESS

### The Case for Combining Gradients with Attention

If raw attention Aᵢⱼ tells you "position i attended to position j," and
the gradient ∂f/∂Vⱼ tells you "the output is sensitive to the value
vector of position j," then combining them gives:

    "Position i attended to position j, AND the value of position j
     was important to the final output."

This combination removes the collinearity problem: even if attending to
j appears important (high Aᵢⱼ), the gradient check ensures that position
j's VALUE was actually used. If V vectors are collinear and attention
doesn't matter, the gradient will be low — the combined score will
correctly flag low importance.

### Gradient × Attention (Schuster et al., 2019)

The simplest combination weights each attention weight by the gradient
of the output with respect to that attention weight:

    GradAttn_ij = |∂f/∂A_ij| × A_ij

The gradient ∂f/∂Aᵢⱼ measures: "how much would the output change if
attention weight Aᵢⱼ changed slightly?" If the output is insensitive to
Aᵢⱼ (value vectors collinear), the gradient is small, and GradAttn
correctly reports low importance for that token pair.

### Attention × Gradient in Transformer Interpretability (Chefer et al., 2021)

Chefer, Gur & Wolf (2021) — "Transformer Interpretability Beyond
Attention Visualisation" — proposed a more principled combination for
class-discriminative attribution that passes the Adebayo sanity checks:

    For each layer l and head h:
    Ā^{l,h} = (∂f/∂A^{l,h}) ⊙ A^{l,h}   (element-wise: gradient × attention)
    Ā^{l,h} = Clamp(Ā^{l,h}, min=0)       (keep only positive attributions)
    Ā^{l,h} = Mean_h(Ā^{l,h})             (average across heads)

    Then roll out via matrix multiplication (as in attention rollout)
    but using the gradient-weighted attention Ā instead of raw attention A.

This method:
  1. Uses gradients to weight which attention connections actually matter.
  2. Rolls out through layers to account for residual connections.
  3. Averages across heads to handle multi-head redundancy.
  4. Passes the Adebayo sanity checks (maps change with weight randomisation).

Chefer et al. demonstrated that this approach substantially outperforms
raw attention and attention rollout on pixel localisation benchmarks
for vision transformers (ViT), and provides more faithful explanations
than gradient-only methods by leveraging the spatial structure of attention.


### ALTI — Aggregate Layerwise Token-to-token Influence (Ferrando et al., 2022)

ALTI is a more recent method that computes token influence by measuring
how much each source token's representation contributed to each target
token's representation at each layer, using a mix-layer decomposition:

    influence_{j→i}^l = ‖ contribution_of_j_to_h_i^l ‖ / ‖ h_i^l ‖

This gives a layer-wise influence matrix that can then be composed across
layers (like rollout) to get total input-to-output influences. ALTI
accounts for the FFN layer as well as attention, giving a more complete
picture of information flow than attention-only methods.


##### PART VI: INTERPRETABLE ATTENTION PATTERNS — WHAT IS RELIABLY FOUND

### Syntactic Structure in BERT's Attention

Despite the "attention is not explanation" critique, certain structural
patterns in BERT's attention are robust, replicable, and informative.

    SYNTACTIC DEPENDENCY TRACKING (Hewitt & Manning, 2019):
    Specific attention heads in BERT's middle layers approximate syntactic
    dependency relations. The attention pattern from a verb head to its
    subject, from an adjective to its modified noun, etc., can be recovered
    from single attention heads with 60–80% accuracy on dependency parsing.

    This is more than coincidence: it suggests that BERT's attention
    mechanism ENCODES syntactic structure as a side effect of learning
    to predict masked words. But note: this is not the same as saying
    "attention head 7 layer 4 IS the syntactic parser" — it is saying
    that head's attention weights correlate with the syntactic structure.
    The distinction matters for causal claims.

    COREFERENCE RESOLUTION (Clark et al., 2019):
    Some heads show strong antecedent-seeking behaviour: pronouns attend
    strongly to their referents. "It" attends to "cat" in "the cat...it."
    This is the most celebrated example of interpretable attention.

    Again, the correlation is real but not a causal explanation. The model
    could in principle maintain this coreference tracking through other
    mechanisms (residual stream, FFN) and the attention is merely CORRELATED
    with the right answer.

    POSITIONAL PATTERNS:
    Many heads show regular positional patterns: attending to the next
    token, the previous token, or fixed offsets. These are consistent
    with position-sensitive language modelling.


### The Probing Approach — A More Principled Interrogation

Rather than interpreting attention weights directly, probing classifiers
(see separate module: mechanistic_interpretability/probing_classifiers)
offer a more rigorous framework:

    1. Train a simple linear classifier on the REPRESENTATIONS produced
       by a specific layer (not on the attention weights).
    2. If the linear probe achieves high accuracy on a linguistic task
       (e.g., part-of-speech tagging, named entity recognition), the
       representation at that layer ENCODES that information.
    3. This does not say the attention weights are responsible — it says
       the REPRESENTATIONS contain the information.

Probing is preferable to attention analysis for questions about "what does
this model know?" because it tests the representations directly, without
the collinearity problem and without the attention-causation ambiguity.

    # ================================================================== #
    **Probing vs Attention analysis:**

    Question: "Does BERT know about grammatical gender?"

    Attention approach: look at whether attention correlates with gender
    agreement. Prone to false positives (collinear values) and false
    negatives (information in residual, not attention).

    Probing approach: train a linear classifier on BERT's representations
    to predict grammatical gender. If it succeeds (say, 90% accuracy),
    BERT's representations encode gender. If not, they don't.
    No ambiguity about causation — we're testing the representation,
    not inferring it from the mechanism.
    # ================================================================== #


##### PART VII: ATTENTION FOR CAUSAL LANGUAGE MODELS (GPT-STYLE)

### Causal (Masked) Attention

GPT-style decoder-only models use CAUSAL attention: position i can only
attend to positions j ≤ i (present and past, not future). This is enforced
by setting Sᵢⱼ = −∞ for j > i before the softmax, which makes Aᵢⱼ = 0
for all j > i.

    Causal attention matrix A ∈ ℝⁿˣⁿ for n tokens:

    A = [[A₁₁,   0,    0,   ...,  0   ],    ← position 1 attends only to itself
         [A₂₁, A₂₂,   0,   ...,  0   ],    ← position 2 attends to 1,2
         [A₃₁, A₃₂, A₃₃,  ...,  0   ],    ← position 3 attends to 1,2,3
         [  ⋮    ⋮    ⋮   ⋱    ⋮   ],
         [Aₙ₁, Aₙ₂, Aₙ₃, ..., Aₙₙ  ]]    ← last position attends to all

This triangular structure means that for generation tasks, attention
analysis answers: "when generating token n+1, which of the preceding n
tokens did the model attend to?"

For interpreting a model's output, this is a natural and informative
question — much more targeted than BERT's bidirectional attention, where
every token attends to every other token and the notion of "explanation"
is less well-defined.


### Attention in Instruction-Following LLMs

For instruction-following models (GPT-4, Claude, Llama), attention
analysis has been used to study:

    INSTRUCTION FOLLOWING: does the model attend to the instruction
    throughout its generation, or does it "forget" the instruction
    after a few tokens?

    FACTUAL KNOWLEDGE: when the model generates a factual claim, does
    it attend back to relevant context tokens (retrieval-augmented) or
    to nothing in particular (generating from parametric memory)?

    HALLUCINATION DETECTION: tokens generated from high-entropy attention
    distributions (attending diffusely to many tokens rather than to
    specific context) correlate with hallucination in some studies.

However, all the same caveats apply: these are CORRELATIONS between
attention patterns and model behaviour, not established causal mechanisms.
The residual stream carries substantial information that attention patterns
cannot account for.


##### PART VIII: PRACTICAL GUIDE — WHEN ATTENTION ANALYSIS HELPS

### Validated Use Cases

    USE CASE 1 — DEBUGGING INPUT PREPROCESSING:
    If a model behaves unexpectedly, attention patterns can reveal whether
    the tokenisation is correct, whether special tokens are interfering,
    or whether position encoding is producing unexpected bias.
    Example: model ignores the last token → attention shows [SEP] absorbing
    all attention in the last layer, crowding out the actual last content token.

    USE CASE 2 — HEAD PRUNING AND MODEL COMPRESSION:
    Attention entropy and attention uniformity metrics can identify heads
    that are "doing nothing" (uniform attention = high entropy = no selective
    information routing). These heads are candidates for pruning.
    Michel et al. (2019) found that 20% of BERT heads can be pruned
    without significant accuracy loss.

    USE CASE 3 — IDENTIFYING SHORTCUT LEARNING:
    If the model consistently attends to spurious features (e.g., in NLI,
    attending to "not" → always predicting contradiction), attention patterns
    can flag this bias before deploying the model on out-of-distribution data.

    USE CASE 4 — RETRIEVAL-AUGMENTED GENERATION FAITHFULNESS:
    In RAG systems, checking whether the model attends to the retrieved
    context when generating answers (vs attending to other positions) gives
    a proxy for whether the answer is grounded in retrieved facts.

    USE CASE 5 — HYPOTHESIS GENERATION FOR MECHANISTIC INTERPRETABILITY:
    Attention patterns suggest which heads to investigate with more rigorous
    methods (activation patching, causal tracing). Attention is excellent
    for generating hypotheses; it is poor for confirming them.

### Validated Non-Use Cases (Where NOT to Use Attention)

    DO NOT USE for:
    • Explaining specific predictions to external stakeholders
      (too noisy, non-unique, collinearity problem)
    • Legal or regulatory compliance explanations
      (fails faithfulness requirements)
    • Medical AI accountability
      (cannot establish causal chain from attention to decision)
    • Comparing the importance of tokens for a single prediction
      without gradient correction
      (gradient-weighted attention or integrated gradients are better)


##### PART IX: A COMPLETE WORKED EXAMPLE — ATTENTION COMPUTATION BY HAND

### Two Layers, Two Heads, Four Tokens — Full Trace

We compute scaled dot-product attention, multi-head aggregation, and
attention rollout by hand for a 4-token sequence with 2 layers and 2 heads.

    Sequence: ["the", "cat", "sat", "down"]
    d_model = 4,  d_k = 2,  H = 2,  L = 2

    TOKEN EMBEDDINGS (X ∈ ℝ^{4×4}):
    "the":  x₁ = [0.5, 0.2, 0.3, 0.1]
    "cat":  x₂ = [0.1, 0.8, 0.2, 0.4]
    "sat":  x₃ = [0.3, 0.1, 0.7, 0.5]
    "down": x₄ = [0.2, 0.4, 0.1, 0.9]

    LAYER 1, HEAD 1 — weight matrices (W_Q^{1,1}, W_K^{1,1} ∈ ℝ^{4×2}):

    W_Q = [[0.4, 0.2],   W_K = [[0.3, 0.5],
           [0.1, 0.8],          [0.6, 0.2],
           [0.7, 0.3],          [0.1, 0.7],
           [0.5, 0.6]]          [0.4, 0.3]]

    STEP 1 — Compute Q = X W_Q:
    Q₁ = x₁ W_Q = [0.5×0.4+0.2×0.1+0.3×0.7+0.1×0.5, 0.5×0.2+0.2×0.8+0.3×0.3+0.1×0.6]
                 = [0.20+0.02+0.21+0.05, 0.10+0.16+0.09+0.06] = [0.48, 0.41]
    Q₂ = x₂ W_Q = [0.1×0.4+0.8×0.1+0.2×0.7+0.4×0.5, 0.1×0.2+0.8×0.8+0.2×0.3+0.4×0.6]
                 = [0.04+0.08+0.14+0.20, 0.02+0.64+0.06+0.24] = [0.46, 0.96]
    Q₃ = [0.3×0.4+0.1×0.1+0.7×0.7+0.5×0.5, 0.3×0.2+0.1×0.8+0.7×0.3+0.5×0.6]
       = [0.12+0.01+0.49+0.25, 0.06+0.08+0.21+0.30] = [0.87, 0.65]
    Q₄ = [0.2×0.4+0.4×0.1+0.1×0.7+0.9×0.5, 0.2×0.2+0.4×0.8+0.1×0.3+0.9×0.6]
       = [0.08+0.04+0.07+0.45, 0.04+0.32+0.03+0.54] = [0.64, 0.93]

    STEP 2 — Compute K = X W_K (same tokens, different projection):
    K₁ = [0.5×0.3+0.2×0.6+0.3×0.1+0.1×0.4, 0.5×0.5+0.2×0.2+0.3×0.7+0.1×0.3]
       = [0.15+0.12+0.03+0.04, 0.25+0.04+0.21+0.03] = [0.34, 0.53]
    K₂ = [0.1×0.3+0.8×0.6+0.2×0.1+0.4×0.4, 0.1×0.5+0.8×0.2+0.2×0.7+0.4×0.3]
       = [0.03+0.48+0.02+0.16, 0.05+0.16+0.14+0.12] = [0.69, 0.47]
    K₃ = [0.3×0.3+0.1×0.6+0.7×0.1+0.5×0.4, 0.3×0.5+0.1×0.2+0.7×0.7+0.5×0.3]
       = [0.09+0.06+0.07+0.20, 0.15+0.02+0.49+0.15] = [0.42, 0.81]
    K₄ = [0.2×0.3+0.4×0.6+0.1×0.1+0.9×0.4, 0.2×0.5+0.4×0.2+0.1×0.7+0.9×0.3]
       = [0.06+0.24+0.01+0.36, 0.10+0.08+0.07+0.27] = [0.67, 0.52]

    STEP 3 — Score matrix S = Q K^T / √d_k   (d_k=2, √2≈1.414):
    S₁₁ = Q₁·K₁/√2 = (0.48×0.34 + 0.41×0.53)/1.414 = (0.163+0.217)/1.414 = 0.269
    S₁₂ = Q₁·K₂/√2 = (0.48×0.69 + 0.41×0.47)/1.414 = (0.331+0.193)/1.414 = 0.371
    S₁₃ = Q₁·K₃/√2 = (0.48×0.42 + 0.41×0.81)/1.414 = (0.202+0.332)/1.414 = 0.378
    S₁₄ = Q₁·K₄/√2 = (0.48×0.67 + 0.41×0.52)/1.414 = (0.322+0.213)/1.414 = 0.378

    Row 1 of S: [0.269, 0.371, 0.378, 0.378]

    STEP 4 — Attention weights A = softmax(S) for row 1:
    Subtract max (0.378): [−0.109, −0.007, 0.000, 0.000]
    exp:  [e^{−0.109}, e^{−0.007}, e^0, e^0] = [0.897, 0.993, 1.000, 1.000]
    Sum:  0.897 + 0.993 + 1.000 + 1.000 = 3.890
    A₁:   [0.230, 0.255, 0.257, 0.257]

    Interpretation: "the" attends ROUGHLY EQUALLY to all tokens.
    Slightly less to itself and "cat", slightly more to "sat" and "down."
    This is close to a uniform distribution — a low-information head pattern.
    For "the", the attention is nearly uniform → hard to draw conclusions.

    ATTENTION ROLLOUT — Layer 1 Augmentation:
    Add residual: Ã₁^{row1} = 0.5 × A₁ + 0.5 × [1,0,0,0]
                             = 0.5×[0.230,0.255,0.257,0.257] + [0.5,0,0,0]
                             = [0.115,0.128,0.129,0.129] + [0.5,0,0,0]
                             = [0.615, 0.128, 0.129, 0.129]

    Renormalise: sum = 1.001 ≈ 1.0
    Ã₁ (row1): [0.614, 0.128, 0.129, 0.128]

    After adding the residual: "the" now effectively attends 61.4% to ITSELF
    (via the residual stream), with only 38.6% distributed across other tokens.
    Raw attention showed this as 23% self-attention; residual correction reveals
    the dominant self-flow.

    KEY INSIGHT FROM THE WORKED EXAMPLE:
    The raw attention weight A₁₁ = 0.230 (23% self-attention) dramatically
    underestimates the effective self-influence when accounting for residual.
    Rollout gives Ã₁₁ = 0.614 (61% effective self-attention).
    This is why raw attention systematically understates residual stream
    information and overstates the contribution of other tokens.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔═════════════════════════╦═══════════════╦═════════════╦══════════════════╗
    ║ Method                  ║ Raw Attention ║ Att Rollout ║ Grad × Attention ║
    ╠═════════════════════════╬═══════════════╬═════════════╬══════════════════╣
    ║ Cost                    ║ 0 (free)      ║ O(L n²)     ║ 1 fwd+bwd pass   ║
    ║ Accounts for residual   ║ No            ║ Yes (approx)║ No               ║
    ║ Accounts for FFN        ║ No            ║ No          ║ Partial          ║
    ║ Gradient-informed       ║ No            ║ No          ║ Yes              ║
    ║ Collinearity robust     ║ No            ║ No          ║ Yes              ║
    ║ Adebayo sanity check    ║ Fail          ║ Partial     ║ Pass (Chefer)    ║
    ║ Non-negative only       ║ Yes           ║ Yes         ║ After clamp      ║
    ║ Handles multi-layer     ║ No (per layer)║ Yes         ║ No (per layer)   ║
    ║ Interpretable output    ║ Yes           ║ Yes         ║ Yes              ║
    ║ Production ready        ║ BERTViz lib   ║ BERTViz lib ║ Custom/captum    ║
    ╚═════════════════════════╩═══════════════╩═════════════╩══════════════════╝

    Key quantities:
      Attention:  A^{l,h}_{ij} = softmax(Q_i K_j^T / sqrt(d_k))_j
      Rollout:    Ã^l = 0.5 * A^l + 0.5 * I  (per layer, average over heads)
                  Rollout = Ã^L × Ã^{L-1} × ... × Ã^1
      Grad-Attn:  GA^{l,h} = clamp(∂f/∂A^{l,h} ⊙ A^{l,h}, min=0)
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Attention from Scratch — Self-Attention, Multi-Head, Rollout, BERTViz Patterns": {
        "description": (
            "Implements scaled dot-product attention, multi-head self-attention, and "
            "attention rollout from scratch in pure Python. Runs on a synthetic 6-token "
            "sequence with 2 layers and 3 heads. Visualises the attention patterns in "
            "ASCII (BERTViz head-view style), computes rollout augmented attention to "
            "show the residual correction, classifies each head into one of the five "
            "Kovaleva pattern types (vertical, diagonal, block, heterogeneous, vertical-stripe), "
            "and demonstrates the collinearity problem by measuring when attention weights "
            "are irrelevant to the output. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "attention",
        "code": '''
"""
================================================================================
ATTENTION FROM SCRATCH — SELF-ATTENTION, ROLLOUT, AND BERTviz PATTERNS
================================================================================

We implement:
  1. Scaled dot-product attention
  2. Multi-head self-attention (2 layers, 3 heads)
  3. Attention pattern classification (Kovaleva et al.)
  4. ASCII BERTViz head-view visualisation
  5. Attention rollout (accounting for residual connections)
  6. Collinearity measurement — when attention is irrelevant to output

Sequence: 6 tokens representing a short sentence.
================================================================================
"""

import math
import random

random.seed(57)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def softmax_1d(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]

def matmul(A, B):
    """A is (m×n), B is (n×p) → returns (m×p)."""
    m, n, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n))
             for j in range(p)]
            for i in range(m)]

def matvec(A, x):
    return [dot(A[i], x) for i in range(len(A))]

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

TOKENS = ["[CLS]", "the", "cat", "sat", "down", "[SEP]"]
N = len(TOKENS)
D_MODEL = 8
D_K = 4   # key/query dimension per head
D_V = 4   # value dimension per head
H = 3     # heads per layer
L = 2     # layers

# Random token embeddings (in practice, learned embeddings)
random.seed(57)
X_raw = [[random.gauss(0, 0.5) for _ in range(D_MODEL)] for _ in range(N)]

print("=" * 68)
print("  SEQUENCE AND EMBEDDINGS")
print("=" * 68)
print(f"  Tokens: {TOKENS}")
print(f"  d_model={D_MODEL}, d_k={D_K}, H={H} heads, L={L} layers")
print()
print("  Token embeddings (first 4 dims shown):")
for i, tok in enumerate(TOKENS):
    vals = " ".join(f"{X_raw[i][j]:+.2f}" for j in range(4))
    print(f"  {tok:<8}: [{vals} ...]")


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM PROJECTION MATRICES (simulating trained weights)
# ─────────────────────────────────────────────────────────────────────────────

def random_matrix(rows, cols, scale=0.3):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

# Two layers, three heads each
W_Q = [[[random_matrix(D_MODEL, D_K) for _ in range(H)] for _ in range(L)]]
W_K = [[[random_matrix(D_MODEL, D_K) for _ in range(H)] for _ in range(L)]]
W_V = [[[random_matrix(D_MODEL, D_V) for _ in range(H)] for _ in range(L)]]

# Reformat: W_Q[l][h] = D_MODEL × D_K matrix
W_Q = [[random_matrix(D_MODEL, D_K) for _ in range(H)] for _ in range(L)]
W_K = [[random_matrix(D_MODEL, D_K) for _ in range(H)] for _ in range(L)]
W_V = [[random_matrix(D_MODEL, D_V) for _ in range(H)] for _ in range(L)]


# ─────────────────────────────────────────────────────────────────────────────
# SCALED DOT-PRODUCT ATTENTION (one head)
# ─────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(X, Wq, Wk, Wv, causal=False):
    """
    X:  (n, d_model) token representations
    Wq, Wk: (d_model, d_k)  query/key projections
    Wv:     (d_model, d_v)  value projection
    Returns: (attention_matrix A, output O)
      A: (n, n) attention weights
      O: (n, d_v) output
    """
    n = len(X)
    dk = len(Wq[0])

    # Compute Q, K, V for all tokens
    Q = [matvec(transpose(Wq), X[i]) for i in range(n)]  # n × dk
    K = [matvec(transpose(Wk), X[i]) for i in range(n)]  # n × dk
    V = [matvec(transpose(Wv), X[i]) for i in range(n)]  # n × dv

    # Score matrix S = QK^T / sqrt(dk)
    scale = math.sqrt(dk)
    S = [[dot(Q[i], K[j]) / scale for j in range(n)] for i in range(n)]

    # Optional causal masking
    if causal:
        for i in range(n):
            for j in range(i + 1, n):
                S[i][j] = -1e9

    # Softmax per row
    A = [softmax_1d(S[i]) for i in range(n)]

    # Output: O_i = sum_j A_ij * V_j
    O = []
    for i in range(n):
        o_i = [sum(A[i][j] * V[j][k] for j in range(n))
               for k in range(len(V[0]))]
        O.append(o_i)

    return A, O


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-HEAD ATTENTION (one layer)
# ─────────────────────────────────────────────────────────────────────────────

def multi_head_attention(X, Wqs, Wks, Wvs):
    """
    Wqs, Wks, Wvs: lists of H projection matrices
    Returns: list of H attention matrices
    """
    all_A = []
    for h in range(len(Wqs)):
        A, _ = scaled_dot_product_attention(X, Wqs[h], Wks[h], Wvs[h])
        all_A.append(A)
    return all_A


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE ALL ATTENTION MATRICES (both layers, all heads)
# ─────────────────────────────────────────────────────────────────────────────

all_attention = []  # [L][H] → n×n attention matrix
X_current = [row[:] for row in X_raw]

for l in range(L):
    layer_A = multi_head_attention(X_current, W_Q[l], W_K[l], W_V[l])
    all_attention.append(layer_A)
    # Simple "layer output" = average of head outputs (approx residual + attention)
    # In reality this uses full multi-head projection + residual + LayerNorm
    new_X = []
    for i in range(N):
        # Average value outputs across heads as a simple approximation
        out = [0.0] * D_MODEL
        for h in range(H):
            _, O_h = scaled_dot_product_attention(
                X_current, W_Q[l][h], W_K[l][h], W_V[l][h])
            for d in range(min(D_MODEL, len(O_h[0]))):
                out[d] += O_h[i][d] / H
        # Residual connection: add input
        new_X.append([X_current[i][d] + out[d] * 0.5
                      for d in range(D_MODEL)])
    X_current = new_X


# ─────────────────────────────────────────────────────────────────────────────
# ASCII BERTviz HEAD VIEW
# ─────────────────────────────────────────────────────────────────────────────

def ascii_head_view(A, tokens, title="", threshold=0.15):
    """
    Print a BERTViz-style attention view as ASCII art.
    Shows arcs (lines) from query tokens (left) to key tokens (right).
    Weight shown as bar width.
    """
    n = len(tokens)
    print(f"  {title}")
    max_tok = max(len(t) for t in tokens)
    fmt = f"{{:>{max_tok}}}"

    for i in range(n):
        # Query token label
        print(f"  {fmt.format(tokens[i])} │", end="")
        for j in range(n):
            w = A[i][j]
            # Bar width proportional to weight (0–5 chars)
            bar_len = int(round(w * 5))
            bar = "▓" * bar_len + "·" * (5 - bar_len)
            # Highlight strong attention
            mark = "◄" if w >= threshold and j == max(range(n), key=lambda k: A[i][k]) else " "
            print(f" {bar}{mark}", end="")
        print()

    # Column labels (key/value side)
    pad = " " * (max_tok + 2)
    print(f"{pad}", end="")
    for tok in tokens:
        label = tok[:5].center(7)
        print(f" {label}", end="")
    print()
    print()


print()
print("=" * 68)
print("  ASCII BERTVIEW — HEAD VIEW FOR EACH (LAYER, HEAD)")
print("=" * 68)
print()
print("  Format: each row = one query token, bars show attention to key tokens")
print("  ▓▓▓▓▓ = 100%  ·····= 0%  ◄ marks the maximum attention target")
print()

for l in range(L):
    for h in range(H):
        A = all_attention[l][h]
        ascii_head_view(
            A, TOKENS,
            title=f"── Layer {l+1}, Head {h+1} ──────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN CLASSIFICATION (Kovaleva et al. 2019 taxonomy)
# ─────────────────────────────────────────────────────────────────────────────

def classify_pattern(A, tokens, threshold=0.5):
    """
    Classify the attention matrix into one of five Kovaleva patterns.
    Returns pattern name and brief description.
    """
    n = len(A)

    # VERTICAL: many rows strongly attend to the same column
    col_max_counts = [0] * n
    for i in range(n):
        j_max = max(range(n), key=lambda j: A[i][j])
        col_max_counts[j_max] += 1
    most_attended_col = max(range(n), key=lambda j: col_max_counts[j])
    vertical_score = col_max_counts[most_attended_col] / n

    # DIAGONAL: each row mostly attends to same position
    diagonal_score = sum(A[i][i] for i in range(n)) / n

    # UNIFORM / BLOCK check: is it close to uniform?
    row_entropies = []
    for i in range(n):
        h_val = -sum(A[i][j] * math.log(A[i][j] + 1e-12) for j in range(n))
        max_h = math.log(n)
        row_entropies.append(h_val / max_h)  # normalised to [0,1]
    avg_entropy = sum(row_entropies) / n

    if vertical_score > 0.6:
        target_tok = tokens[most_attended_col]
        return "VERTICAL", f"Most tokens attend to '{target_tok}' (col {most_attended_col})"
    elif diagonal_score > 0.5:
        return "DIAGONAL", "Each token mostly attends to itself"
    elif avg_entropy > 0.85:
        return "UNIFORM", "Attention is nearly uniform across all positions"
    elif avg_entropy < 0.5:
        return "HETEROGENEOUS", "Each token has a distinct, focused attention distribution"
    else:
        return "MIXED", "No dominant pattern — moderate entropy, some structure"


print()
print("=" * 68)
print("  PATTERN CLASSIFICATION — KOVALEVA ET AL. TAXONOMY")
print("=" * 68)
print()
print(f"  {'Location':<20}  {'Pattern':<15}  Description")
print(f"  {'-'*65}")

for l in range(L):
    for h in range(H):
        A = all_attention[l][h]
        pattern, desc = classify_pattern(A, TOKENS)
        print(f"  Layer {l+1}, Head {h+1:<12}  {pattern:<15}  {desc}")

print()
print("  NOTE: Most heads in real BERT models show VERTICAL or DIAGONAL")
print("  patterns, attending to [CLS], [SEP], or themselves. Only a minority")
print("  show HETEROGENEOUS patterns with input-sensitive attention structure.")


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION ROLLOUT
# ─────────────────────────────────────────────────────────────────────────────

def attention_rollout(all_attention, n_tokens):
    """
    Compute attention rollout across all layers.
    all_attention: [L][H] list of n×n attention matrices

    Returns: (n × n) effective attention matrix from input to each position.
    """
    L = len(all_attention)
    H = len(all_attention[0])
    n = n_tokens

    # Identity matrix
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    rollout = [row[:] for row in I]  # start with identity

    for l in range(L):
        # Average attention across heads
        A_avg = [[0.0]*n for _ in range(n)]
        for h in range(H):
            for i in range(n):
                for j in range(n):
                    A_avg[i][j] += all_attention[l][h][i][j] / H

        # Add residual: A_tilde = 0.5*A_avg + 0.5*I
        A_tilde = [[0.5 * A_avg[i][j] + (0.5 if i == j else 0.0)
                    for j in range(n)] for i in range(n)]

        # Renormalise rows
        for i in range(n):
            row_sum = sum(A_tilde[i])
            A_tilde[i] = [v / row_sum for v in A_tilde[i]]

        # Compose: rollout = A_tilde × rollout
        rollout = matmul(A_tilde, rollout)

    return rollout


rollout = attention_rollout(all_attention, N)

print()
print("=" * 68)
print("  ATTENTION ROLLOUT — EFFECTIVE INPUT ATTRIBUTION")
print("=" * 68)
print()
print("  Rollout accounts for residual connections by adding 0.5×Identity")
print("  at each layer before composing. Result shows EFFECTIVE attention")
print("  from each output position back to each input token.")
print()
print(f"  {'Query token':>10}  →  ", end="")
for tok in TOKENS:
    print(f"{tok[:6]:>8}", end="")
print()
print(f"  {'-'*70}")

for i, qtok in enumerate(TOKENS):
    print(f"  {qtok:>10}  →  ", end="")
    for j in range(N):
        v = rollout[i][j]
        bar = "█" if v > 0.30 else ("▓" if v > 0.15 else ("░" if v > 0.08 else "·"))
        print(f"  {v:.3f}{bar}", end="")
    print()

print()
print("  Key observation: the diagonal (self-attention) is much stronger")
print("  in rollout than in raw attention, because the residual connection")
print("  carries each token's own representation through all layers.")
print("  Raw attention may show A[i][i]=0.15; rollout reveals the effective")
print("  self-attention is typically >0.50 when residual is included.")


# ─────────────────────────────────────────────────────────────────────────────
# COLLINEARITY MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_collinearity(X, Wv_l_h):
    """
    Measure how collinear the value vectors are for this layer/head.
    Returns: (mean pairwise cosine similarity, max singular value / Frobenius norm).
    Higher = more collinear = attention weights matter LESS for output.
    """
    n = len(X)
    V = [matvec(transpose(Wv_l_h), X[i]) for i in range(n)]

    # Mean pairwise cosine similarity (excluding self-pairs)
    total_cos = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            nrm_i = math.sqrt(sum(v**2 for v in V[i]))
            nrm_j = math.sqrt(sum(v**2 for v in V[j]))
            if nrm_i > 1e-12 and nrm_j > 1e-12:
                cos = dot(V[i], V[j]) / (nrm_i * nrm_j)
                total_cos += abs(cos)
                count += 1

    mean_cos = total_cos / count if count > 0 else 0.0
    return mean_cos


print()
print("=" * 68)
print("  COLLINEARITY MEASUREMENT — WHEN ATTENTION WEIGHTS ARE IRRELEVANT")
print("=" * 68)
print()
print("  If value vectors V are nearly collinear (cosine similarity ≈ 1),")
print("  the attention weights barely affect the output: output ≈ V̄ regardless.")
print("  High collinearity → attention-based explanations are less reliable.")
print()
print(f"  {'Location':<20}  {'Mean |cosine(V)|':>18}  {'Reliability'}")
print(f"  {'-'*55}")

for l in range(L):
    for h in range(H):
        cos = measure_collinearity(X_raw, W_V[l][h])
        if cos > 0.8:
            reliability = "LOW  ← values nearly collinear; attention irrelevant"
        elif cos > 0.5:
            reliability = "MEDIUM"
        else:
            reliability = "HIGH ← diverse values; attention matters"
        print(f"  Layer {l+1}, Head {h+1:<12}  {cos:>18.4f}  {reliability}")

print()
print("  In a real BERT model, collinearity varies dramatically by layer and head.")
print("  Deep layers in fine-tuned models often have HIGHER collinearity —")
print("  the task-specific information collapses into a low-dimensional subspace.")
print()
print("  PRACTICAL IMPLICATION:")
print("  Before trusting an attention-based explanation for a specific head,")
print("  check the collinearity of that head's value vectors.")
print("  High collinearity → use gradient-weighted attention or IG instead.")
print("  Low collinearity  → raw attention is more informative.")
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
    #     from interpretability.visuals.attention_analysis import (
    #         ATTENTION_VISUAL_HTML,
    #         ATTENTION_VISUAL_HEIGHT,
    #     )
    #     visual_html   = ATTENTION_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = ATTENTION_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[attention_analysis.py] Could not load visual: {e}", stacklevel=2)

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