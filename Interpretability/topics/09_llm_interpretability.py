"""
LLM Interpretability — Understanding What Transformers Learn and How They Reason
=================================================================================

A language model predicts "The Eiffel Tower is in Paris" with 97% confidence.
But WHY does it know this? Is the fact stored in one layer? Distributed across
thousands of neurons? Retrieved by a single attention head? Is the model
reasoning or pattern-matching? Could it be manipulated into "forgetting" it?

Interpretability is the discipline of answering these questions rigorously.
It is not about reading attention weights and calling it done — it is about
reverse-engineering the algorithms that neural networks implement, tracing
information from input tokens to output logits, and understanding which
components causally determine which outputs.

Why interpretability matters:
    Safety:          Understanding whether a model "knows" something vs
                     pattern-matches is essential for trusting high-stakes
                     outputs in medicine, law, and autonomous systems.
    Debugging:       Identifying which circuit is responsible for a failure
                     mode allows surgical fixes rather than full retraining.
    Alignment:       If we cannot verify whether a model's reasoning process
                     matches its stated reasoning, we cannot trust it. Sycophancy,
                     deceptive alignment, and reward hacking are interpretability
                     failures before they are alignment failures.
    Scientific value: LLMs are the largest-scale learning systems ever built.
                     Interpreting them is a fundamental scientific question about
                     what structures emerge from gradient descent on language.
    Editing facts:   Methods like ROME/MEMIT allow targeted knowledge editing
                     inside a model — impossible without interpretability.

This module covers the complete LLM interpretability toolkit from first
principles: the transformer as the interpretability substrate (residual stream,
attention, MLPs), the logit lens, probing classifiers, activation patching and
causal tracing, circuit discovery, superposition and polysemanticity, sparse
autoencoders, representation engineering, gradient-based attribution, and the
evaluation of interpretability methods themselves.
"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME   = "LLM Interpretability — Understanding What Transformers Learn"
DISPLAY_NAME = "09 . LLM Interpretability"
ICON         = "🔬"
SUBTITLE     = "Circuits, attention, probing, activation patching, SAEs, and mechanistic analysis"


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

##### PART 1 — WHAT IS LLM INTERPRETABILITY?

### The Fundamental Question

A trained language model is a function f: tokens → logits. Given a sequence
of tokens, it outputs a probability distribution over the next token. We know
what it does. We do not know HOW or WHY it does it.

Interpretability is the systematic effort to reverse-engineer the algorithms
and knowledge structures that a trained model implements — to go from the
weights (numbers) to the computation (mechanism) to the concept (meaning).

    Two levels of the question:
    "What does this model know?" → Behavioral interpretability
        Does it know Paris is in France?
        Can it perform modular arithmetic?
        Does its internal representation of sentiment separate cleanly?

    "How does this model compute it?" → Mechanistic interpretability
        Which attention heads retrieve the relevant fact?
        Which MLP layers store it?
        What sequence of operations produces the final answer?

Behavioral interpretability is easier and more mature. Mechanistic
interpretability is harder and more scientifically fundamental.


### The Black Box Problem

A 70-billion parameter model has 70,000,000,000 floating-point numbers as
weights. Even if you could read every weight, you could not understand the
model from the weights alone — the computation is distributed across billions
of interacting parameters. The challenge is not data access; it is finding
the right ABSTRACTIONS and TOOLS to reveal structure.

    The analogy: understanding a CPU by reading its transistor states.
    Every bit of information is there, but the right level of description
    is instruction opcodes, not transistor voltages. We need the equivalent
    of "opcodes" for neural network computations.

Key insight from Anthropic's mechanistic interpretability programme:
neural networks learn to implement algorithms in their weights. Those
algorithms can be reverse-engineered. The algorithms are often surprisingly
clean and interpretable once the right decomposition is found.


### Mechanistic vs Behavioral Interpretability

    ┌─────────────────────────────────────────────────────────────────┐
    │                  INTERPRETABILITY LANDSCAPE                     │
    ├──────────────────────────┬──────────────────────────────────────┤
    │ BEHAVIORAL               │ MECHANISTIC                          │
    │ (what the model does)    │ (how the model does it)              │
    ├──────────────────────────┼──────────────────────────────────────┤
    │ Probing classifiers      │ Circuit analysis                     │
    │ Representation geometry  │ Activation patching                  │
    │ Behavioural benchmarks   │ Causal tracing (ROME)                │
    │ Saliency / attribution   │ Sparse autoencoders (SAEs)           │
    │ Logit lens               │ Superposition analysis               │
    │ Activation steering      │ Induction head discovery             │
    └──────────────────────────┴──────────────────────────────────────┘

Both are important. Behavioral methods are more scalable and more directly
applicable to model evaluation. Mechanistic methods are deeper and enable
editing, surgical correction, and genuine understanding.


### Why This Is Hard

    PROBLEM 1 — DISTRIBUTED REPRESENTATIONS:
    Concepts are not stored in individual neurons. A single feature (e.g.,
    "is a famous person") may be encoded across hundreds of neurons. A single
    neuron may participate in encoding hundreds of features simultaneously
    (superposition). There is no clean "Paris neuron."

    PROBLEM 2 — COMPOSITION AND NONLINEARITY:
    Individual components are linear (attention, matrix multiplication), but
    the composition of many linear operations with ReLU/GELU nonlinearities
    creates complex, hard-to-decompose functions.

    PROBLEM 3 — SCALE:
    Frontier models have hundreds of layers, thousands of attention heads, and
    hundreds of billions of parameters. Analysis that works on GPT-2 (124M
    parameters) may not scale to GPT-4-class models.

    PROBLEM 4 — EVALUATION:
    How do we know if an interpretability explanation is CORRECT? A story that
    sounds plausible may not reflect the actual computation. Evaluating
    interpretability claims requires rigorous causal intervention experiments,
    not just correlation analysis.


##### PART 2 — THE TRANSFORMER AS INTERPRETABILITY SUBSTRATE

### The Residual Stream

The central abstraction for mechanistic interpretability is the RESIDUAL STREAM
— the vector that flows through the model and gets updated by each component.

    Token embedding:   x₀ = embed(token) + positional_embedding
    Layer l:           x_l = x_{l-1} + attention(x_{l-1}) + MLP(x_{l-1})
    Final output:      logits = unembed(x_L) = W_U × x_L

The residual stream has a fixed dimension d_model (e.g., 512, 768, 4096).
Every component — every attention head, every MLP — READS from the residual
stream and WRITES a vector back into it via addition. The model's full
computation is a sequence of additive updates to this stream.

    ┌─────────────────────────────────────────────────────────────────┐
    │  THE RESIDUAL STREAM (KEY INSIGHT FOR INTERPRETABILITY)         │
    │                                                                 │
    │  x₀ ──[attn₁]──[mlp₁]──[attn₂]──[mlp₂]──...──[attnL]──[mlpL]──▶ logits │
    │       │         │        │         │                            │
    │       ▼         ▼        ▼         ▼                            │
    │      (each adds a vector to x; the result is the residual stream)│
    │                                                                 │
    │  Because every component ADDS to x, we can analyse each        │
    │  component's contribution to the final output INDEPENDENTLY.   │
    │  The residual stream is the COMMUNICATION CHANNEL between       │
    │  components.                                                    │
    └─────────────────────────────────────────────────────────────────┘

This additivity is crucial: it allows each component's contribution to the
final logits to be decomposed and studied separately.


### Attention Heads — The Routing Mechanism

Each attention head performs a "soft lookup": for each query position, it
looks up the most relevant key positions and retrieves their values.

    For position i with query qᵢ = xᵢ Wᴷ:
    Keys:     kⱼ = xⱼ Wᴷ       (one per source position j)
    Values:   vⱼ = xⱼ Wᵛ       (one per source position j)

    Attention scores:  aᵢⱼ = qᵢ · kⱼ / √d_head
    Attention weights: αᵢⱼ = softmax(aᵢ·)_j      (probability over source positions)
    Output:            oᵢ = Σⱼ αᵢⱼ vⱼ Wₒ          (write back to residual stream)

The key insight: each attention head implements a FUNCTION from (query position,
context positions) to (what to write into the residual stream at the query
position). The head can attend to nothing, attend to the previous token,
attend to the beginning-of-sentence, attend based on semantic similarity,
or attend based on syntactic structure — whatever the training gradient found
useful.

    Reading an attention head mechanistically requires asking:
    1. What Q-K pattern does it implement? (What query positions attend to
       what key positions, and why?)
    2. What does it write via OV? (What information does it move?)
    3. What is the net effect on the residual stream and the final logits?


### MLP Layers — The Memory Banks

Each MLP layer in a transformer applies a two-layer neural network to each
token position independently:

    MLP(x) = W₂ × GELU(W₁ × x + b₁) + b₂

Elhage et al. (2022) showed that MLP layers function as KEY-VALUE MEMORIES
(Geva et al., 2021): each row of W₁ is a "key" (a pattern to detect in x),
the corresponding GELU activation is whether that key fires, and the
corresponding row of W₂ is a "value" (what to add to the residual stream
when that key fires).

    MLP as key-value memory:
    Key k_i = row i of W₁.T (a direction in the residual stream space)
    Value v_i = row i of W₂   (a direction to ADD to the residual stream)
    Gate g_i = GELU(k_i · x)  (how much to activate this "memory")

    MLP(x) = Σᵢ g_i × v_i

This makes MLP layers the primary KNOWLEDGE STORAGE mechanism in transformers.
Factual knowledge (Paris is in France, water is H₂O) is predominantly encoded
in the MLP weights. Attention heads route information; MLPs store and recall facts.


### The Vocabulary Space and Unembedding

Both the embedding matrix Wₑ and the unembedding matrix Wᵤ (often tied:
Wᵤ = Wₑᵀ) map between the d_model residual stream space and the d_vocab
vocabulary space.

    Any vector v in the residual stream has a vocabulary projection:
    logits(v) = Wᵤ × v

    This allows us to ask: "At this intermediate layer, with this activation,
    what token would the model predict if we stopped here?"

This is the foundation of the LOGIT LENS technique (Part 4).


##### PART 3 — ATTENTION ANALYSIS

### The Role of Attention in Interpretability

Attention weights αᵢⱼ (the softmax output showing how much position i attends
to position j) were initially held up as the primary interpretability tool for
transformers. If the model attends to "Paris" when generating "capital", surely
that attention IS the explanation?

This view is now understood to be incomplete. Attention weights show WHERE
information flows, but not WHAT information flows or whether the attended
information is causally relevant to the output. Nevertheless, attention analysis
remains a useful first-pass diagnostic.


### Attention Pattern Taxonomy

Systematic analysis of pre-trained transformers reveals recurring patterns:

    PREVIOUS TOKEN HEADS:
    Head attends strongly to the immediately preceding token (position i → i-1).
    Function: passes information about the previous token forward.
    Common in early layers. Key role in induction circuits (see Part 7).

    BEGINNING-OF-SEQUENCE HEADS:
    Head attends primarily to position 0 (the [BOS] or first token).
    Function: "dump" heads — a way to attend to a fixed position when no
    other position is relevant. Acts as a no-op for irrelevant heads.

    INDUCTION HEADS:
    Head attends to the position AFTER the previous occurrence of the current
    token. If the sequence is [..., A, B, ..., A, ?], it attends to B.
    Function: implements in-context copying and few-shot learning.
    One of the most important discovered circuits (Part 7).

    SYNTACTIC HEADS:
    Head attends based on syntactic relationships (subject → verb,
    noun → modifier). Often emerge in middle layers.
    Function: moves syntactically relevant information.

    SEMANTIC SIMILARITY HEADS:
    Head attends based on semantic similarity between tokens.
    Function: retrieves semantically related context.

    COPY HEADS / NAME MOVER HEADS:
    Head attends to earlier occurrences of a name/entity and copies
    its representation. Key role in factual recall tasks.


### Multi-Head Attention and Superposition

A transformer has H attention heads per layer. Their outputs are ADDED together
(via the projection W_O). This means:

    attn_output = Σₕ attn_head_h(x)

Each head writes an independent vector to the residual stream. The sum of all
head outputs is what gets added to the residual stream. This makes multi-head
attention a combination of H independent "routing operations."

    Why multiple heads?
    Different heads can specialise in different types of relationships
    (syntactic, semantic, positional, copy-based) and their contributions
    COMBINE additively. The model can simultaneously route information via
    different patterns because each head occupies different dimensions of
    the residual stream (approximately — there is interference, but the
    high dimensionality makes this tractable).


### Attention Analysis Tools

    BERTVIZ (Vig, 2019):
    Interactive visualisation of attention weights for BERT/GPT models.
    Shows per-head attention patterns across all layers.
    Best for qualitative hypothesis generation, not quantitative analysis.

    ATTENTION ROLLOUT (Abnar & Zuidema, 2020):
    Recursively multiplies attention matrices across layers to approximate
    the "total" attention from input tokens to the final representation.
    Addresses the "attention is not explanation" problem partially by
    accounting for residual connections.

    ATTENTION FLOW (Abnar & Zuidema, 2020):
    Treats the attention weights as a flow graph and applies max-flow
    analysis to compute the total information flow from each input token.

    LIMITATIONS OF ATTENTION ANALYSIS:
    - Attention weights do not directly measure causal influence.
    - A head can have high attention weight to a token without that
      token's information affecting the output (the OV matrix may
      multiply it to near zero).
    - Gradient-based methods (Part 11) or patching experiments (Part 6)
      are required for causal claims.


##### PART 4 — THE LOGIT LENS AND TUNED LENS

### The Core Idea: Iterative Refinement

The residual stream is updated ADDITIVELY at each layer. If we apply the
unembedding matrix Wᵤ to the residual stream at any intermediate layer l,
we get a "mid-model" prediction: what token would the model predict if
computation stopped at layer l?

    logits_l = Wᵤ × x_l

This is the LOGIT LENS (nostalgebraist, 2020). Plotting how the top-1
predicted token changes from layer 0 to layer L reveals:

    - In early layers: the prediction is often the input token itself
      (the model hasn't "thought" yet).
    - In middle layers: the prediction shifts toward semantically related
      words or the correct next-word prediction.
    - In late layers: the prediction converges to the final answer.

    ┌─────────────────────────────────────────────────────────────────┐
    │  LOGIT LENS EXAMPLE — completing "The capital of France is"     │
    │                                                                 │
    │  Layer  0:  "the"  (just repeated the last seen token)          │
    │  Layer  2:  "a"    (grammatically plausible, semantically wrong) │
    │  Layer  6:  "Europe" (geographic category)                      │
    │  Layer  10: "French" (getting closer — right region)            │
    │  Layer  16: "Paris" ← first correct prediction                  │
    │  Layer  24: "Paris" ← stable final answer                       │
    │                                                                 │
    │  The model "builds up" the answer iteratively across layers.    │
    └─────────────────────────────────────────────────────────────────┘

The logit lens reveals WHERE in the model a given prediction "crystalises."
Some facts are predicted confidently from early layers; others require the
full model depth. This is direct evidence for the iterative refinement view
of transformer computation.


### The Tuned Lens

The logit lens has a flaw: the residual stream at layer l is NOT the same
kind of vector as the final residual stream at layer L. The unembed matrix
Wᵤ was trained to decode x_L, not x_l. The mid-layer residual streams
have not yet passed through all the processing that makes them decodable by Wᵤ.

The TUNED LENS (Belrose et al., 2023) addresses this by training a small
linear map (an "affine probe") for each layer:

    tuned_logits_l = Wᵤ × (Aₗ × x_l + bₗ)

where Aₗ and bₗ are a trained affine transformation specific to layer l.
This probe maps x_l into the final-layer residual stream space, allowing
Wᵤ to decode it accurately.

    Why the tuned lens is better:
    The logit lens often shows high entropy (uncertain predictions) in
    early layers not because the model doesn't "know" anything yet,
    but because x_l hasn't been processed into the final representation
    format. The tuned lens corrects for this, giving more accurate
    predictions of what "the model knows" at each layer.


### What the Logit Lens Reveals

    FACT RECALL:
    For a prompt like "Eiffel Tower is in", the final answer ("Paris")
    first appears with high probability at the MLP layer of a specific
    depth. This identifies WHERE in the model the fact is retrieved.

    COPY OPERATIONS:
    For a prompt like "My name is Alice. Her name is", the token "Alice"
    is tracked as the top prediction through attention heads that
    implement name-copying.

    LAYER SPECIALISATION:
    Different types of information become decodable at different depths:
    • Syntactic POS tags: decodable from very early layers.
    • Semantic category: decodable from mid layers.
    • Specific factual answers: decodable from late layers (MLP-heavy).

    FAILURE MODES:
    Prompts where the model gives a wrong answer but the correct answer
    appears transiently in middle layers suggest a "late corruption" —
    some late-layer computation overwrites the correct information.


##### PART 5 — PROBING CLASSIFIERS

### The Probing Framework

A probing classifier (Alain & Bengio, 2016) tests whether a SPECIFIC PROPERTY
is linearly decodable from an intermediate representation.

    SETUP:
    1. Take a trained model f with layers l = 1, ..., L.
    2. Choose a property of interest: e.g., "part of speech of token i."
    3. Collect activations: for each input xᵢ, extract the residual stream
       at layer l: hₗ(xᵢ) ∈ ℝ^{d_model}.
    4. Train a SIMPLE classifier (linear or shallow MLP) to predict the
       property from hₗ(xᵢ).
    5. Evaluate accuracy.

    If a linear probe achieves high accuracy at layer l, it means the
    property IS linearly represented in the residual stream at layer l.
    If accuracy is low, either the model doesn't represent the property,
    or the representation is non-linear (rare for well-understood properties).

    ┌─────────────────────────────────────────────────────────────────┐
    │  PROBING — WHAT EACH LAYER KNOWS                                │
    │                                                                 │
    │  Property              │ Peak layer  │ Notes                   │
    │  ──────────────────────┼─────────────┼───────────────────────  │
    │  Part-of-speech tag    │ Layer 2–4   │ Very early, syntactic   │
    │  Constituent tree depth│ Layer 5–8   │ Syntactic structure     │
    │  Named entity type     │ Layer 6–10  │ Semantic category       │
    │  Coreference           │ Layer 10–16 │ Discourse-level         │
    │  Sentiment             │ Layer 8–14  │ Variable across models  │
    │  World knowledge       │ Layer 16–L  │ Factual content, deep   │
    └─────────────────────────────────────────────────────────────────┘


### Probing Best Practices

    PROBE COMPLEXITY MATTERS:
    Use the SIMPLEST probe that works. A linear probe passing means the
    representation is linearly separable — a strong, clean result.
    A complex MLP probe passing could mean the probe is learning the task,
    not that the model's representations are structured.

    THE AMNESIC PROBING TECHNIQUE (Elazaar et al., 2020):
    Remove the probed property from the representation using the probe
    direction, then test model performance. If model performance drops,
    the property was causally relevant — not just correlated.

    MINIMUM DESCRIPTION LENGTH PROBING (Voita & Titov, 2020):
    Instead of measuring accuracy, measure how many BITS are required to
    describe the labels given the representation. Shorter description =
    more structured representation.

    STRUCTURAL PROBING (Hewitt & Manning, 2019):
    Learn a linear map that transforms the representation space so that
    parse tree distances match Euclidean distances. If the probe achieves
    a good tree structure, the syntax tree is encoded in the geometry of
    the representation.


### What Probing Can and Cannot Tell Us

    WHAT PROBING CAN TELL US:
    ✓ Whether a property is LINEARLY decodable from a layer's representation.
    ✓ AT WHICH LAYER a property first becomes decodable.
    ✓ Whether a property is represented MORE clearly after fine-tuning.
    ✓ Correlation between representation quality and downstream task accuracy.

    WHAT PROBING CANNOT TELL US:
    ✗ Whether the model USES the property in its computation.
      (A feature can be present but ignored by downstream layers.)
    ✗ Whether the representation is CAUSALLY necessary for the output.
      (Use activation patching for causal claims — Part 6.)
    ✗ Whether the feature is FUNCTIONALLY used by the model.
      (The model might represent POS tags but not use them for generation.)

    THE CORRELATION/CAUSATION PROBLEM:
    Probing is purely correlational. The residual stream at layer l may contain
    a linear representation of POS tags simply because POS-correlated features
    are useful for other computations, not because the model explicitly
    "computes POS tags." Causal interventions (Part 6) are required to
    distinguish correlation from functional use.


### Representation Geometry

Beyond probing for specific properties, the GEOMETRY of the representation
space reveals its structure.

    LINEAR REPRESENTATION HYPOTHESIS (Mikolov et al., 2013; extended to LLMs):
    Concepts are encoded as DIRECTIONS in the residual stream space.
    The same concept at different contexts occupies the same direction.

    Evidence:
    • word2vec: king − man + woman ≈ queen (famous analogy arithmetic)
    • In transformers: "France" − "Paris" ≈ "Germany" − "Berlin"
      (the "capital city" relationship is a fixed direction in space)

    POLYSEMANTICITY (Elhage et al., 2022):
    In practice, individual neurons do NOT represent single clean concepts.
    A single neuron may activate for "base64 encoded text", "Finnish language",
    and "named variables in code" — three superficially unrelated concepts.
    This is polysemanticity and arises from superposition (Part 8).

    REPRESENTATIONAL SIMILARITY ANALYSIS (RSA):
    Compare the geometry of two representation spaces (e.g., layer 5 of
    GPT-2 vs layer 5 of BERT) by correlating their pairwise distance matrices.
    If RSA score is high, the two models have learned similar representations
    despite different architectures.


##### PART 6 — ACTIVATION PATCHING AND CAUSAL TRACING

### The Core Causal Question

Probing tells us WHAT is represented. Activation patching tells us WHAT IS
USED — which components CAUSALLY determine the output.

    The key experiment:
    Run the model on a CLEAN input x_clean (correct, expected output).
    Run the model on a CORRUPTED input x_corrupt (perturbed, wrong output).
    PATCH: replace a specific activation in the corrupted run with the
           corresponding activation from the clean run.
    Measure: does performance recover toward the clean output?

    If patching activation A restores performance, A encodes information
    that is CAUSALLY necessary for the correct output.

    ┌─────────────────────────────────────────────────────────────────┐
    │  ACTIVATION PATCHING PROTOCOL                                   │
    │                                                                 │
    │  1. Run model on x_clean → get all activations a_l^clean        │
    │  2. Run model on x_corrupt → get all activations a_l^corrupt     │
    │  3. For each component c at layer l:                            │
    │     a. Start with x_corrupt                                     │
    │     b. Replace activation of c at layer l with a_l^clean[c]     │
    │     c. Continue forward pass with patched activation            │
    │     d. Measure recovery = (patched_logit − corrupt_logit) /     │
    │                            (clean_logit − corrupt_logit)        │
    │  4. Map recovery scores across all (layer, component) pairs     │
    └─────────────────────────────────────────────────────────────────┘

Recovery = 1.0 means the component fully carries the relevant information.
Recovery = 0.0 means the component is irrelevant.
Recovery > 1.0 or < 0.0 means indirect effects (the component suppresses
or amplifies downstream via paths not patched).


### Causal Tracing — ROME and MEMIT

Meng et al. (2022) — "Locating and Editing Factual Associations in GPT"
(the ROME paper) — applied causal tracing to factual recall. The setup:

    Clean:     "The Eiffel Tower is in the city of [Paris]"
    Corrupted: "The Eiffel Tower is in the city of [???]"
               (corrupted by adding noise to the embedding of "Eiffel Tower")

    Finding: patching at specific MLP LAYERS (not attention heads) at the
    SUBJECT TOKEN ("Eiffel Tower") position restores the correct answer.
    This identifies the MLP layers at the subject position as the PRIMARY
    STORAGE LOCATION of factual associations.

    Specifically: early-to-mid MLP layers at the subject token position
    store factual associations. They function as associative memories.
    Later attention heads retrieve and propagate this information to the
    final token position where the prediction is made.

    ROME (Rank-One Model Editing):
    Use this causal understanding to EDIT facts:
    1. Locate the MLP layer that stores "Eiffel Tower → Paris."
    2. Find the key vector k* that represents the Eiffel Tower subject.
    3. Find the current value vector v* that leads to "Paris."
    4. Compute a new value vector v** that leads to "Rome" (desired edit).
    5. Update the MLP weight: W₂ ← W₂ + Δ such that W₂k* = v**.

    This surgical edit changes ONLY the Eiffel Tower fact, leaving all
    other facts and model behaviours unchanged.

    MEMIT (Meng et al., 2023) — extends ROME to simultaneous multi-fact edits
    (thousands of facts in a single pass using multi-layer spreading).


### Direct vs Indirect Effects

Patching the full residual stream at a layer gives the TOTAL effect of
everything computed up to that layer. But we want to identify SPECIFIC
components. This requires patching at finer granularity:

    ATTENTION HEAD PATCHING:
    Patch the output of a specific attention head (h at layer l).
    Measures the direct contribution of that head to the output.

    MLP PATCHING:
    Patch the output of the MLP at layer l.
    Measures the direct contribution of the MLP at that layer.

    RESIDUAL STREAM PATCHING:
    Patch the entire residual stream at a specific position (token i)
    and layer l. Measures all information flowing through that (position, layer).

    PATH PATCHING (Wang et al., 2022 — "Interpretability in the Wild"):
    Patch a specific PATH from component A to component B.
    Measures the INDIRECT effect of A on the final output via B,
    controlling for all other paths.


### A/B Testing Prompts for Causal Analysis

    FACTUAL PAIRS (indirect object identification):
    Clean:     "When Mary and John went to the store, John gave a bag to Mary."
               → expects "Mary"
    Corrupted: "When Mary and John went to the store, John gave a bag to John."
               → expects "John"
    (IOI task — Indirect Object Identification; Wang et al., 2022)

    This technique is used to identify the "IOI circuit" — the specific
    set of attention heads responsible for tracking names and roles.

    COUNTERFACTUAL INPUTS:
    Clean:     "The president of the United States is"  → "Biden" (at time of training)
    Corrupted: "The president of [randomised country] is"
    This isolates which activations specifically encode US/Biden vs
    generic "president" information.


##### PART 7 — CIRCUITS AND MECHANISTIC INTERPRETABILITY

### The Circuit Hypothesis

Olah et al. (2020) proposed the CIRCUIT HYPOTHESIS: neural networks learn
algorithms implemented by CIRCUITS — small subsets of nodes (neurons, attention
heads) and edges (weights connecting them) that implement a specific function.

    Formal definition of a circuit:
    A circuit C = (N, E) is a subgraph of the full computational graph
    where N is a set of nodes (model components) and E is a set of edges
    (weighted connections between them) that together implement a specific
    input-output behaviour.

    The circuit is COMPLETE if removing it from the model causes performance
    on the target task to drop to baseline.
    The circuit is MINIMAL if no proper subgraph is also complete.
    The circuit is FAITHFUL if its behaviour matches the full model's behaviour.

Circuits have been discovered in:
    • Vision networks: curve detectors, high-low frequency detectors
    • Language models: induction heads, IOI circuits, copy suppression,
      docstring completion, modular arithmetic


### Induction Heads — The Canonical Example

INDUCTION HEADS (Olah et al., 2022; Elhage et al., 2022) are the most
thoroughly studied LLM circuit. They implement in-context pattern completion:

    Given a sequence [..., A, B, ..., A, ?], the induction head predicts B.

    MECHANISM (two-head circuit):
    1. Previous Token Head (PTH):
       - Attends from position i to position i-1.
       - Writes a "flag" into the residual stream at position i: "the previous
         token was X." Uses the K-composition: the KEY of this head at position i
         represents position i's token; the OUTPUT writes the previous token's
         identity into the residual stream.

    2. Induction Head (IH):
       - Reads the "previous token was X" signal from PTH's output.
       - Uses this as a QUERY to attend back to positions j where the
         residual stream similarly encodes "the previous token was X" — i.e.,
         to the position right after the last occurrence of the current token.
       - Attends to position j+1 (the token that followed the previous
         occurrence of the current token).
       - Copies the representation of that token forward.

    Why this matters:
    Induction heads implement a general-purpose IN-CONTEXT COPYING algorithm.
    They are the mechanistic basis for in-context learning: given a few
    examples in the prompt (A→B, C→D), the model can generalise to (E→?).
    This circuit appears to form during a phase transition early in training.

    ┌─────────────────────────────────────────────────────────────────┐
    │  INDUCTION HEAD CIRCUIT                                         │
    │                                                                 │
    │  Sequence: [cat] [sat] [on] [mat] ... [cat] [?]                 │
    │                                                                 │
    │  Step 1: PTH writes "prev=cat" at second [cat] position         │
    │  Step 2: IH queries for positions where "prev=cat" was true     │
    │          → finds the FIRST [cat] position                       │
    │  Step 3: IH attends to the token AFTER first [cat] = [sat]      │
    │  Step 4: [sat] representation is copied → predicts "sat"        │
    └─────────────────────────────────────────────────────────────────┘


### IOI Circuit — Indirect Object Identification

Wang et al. (2022) — "Interpretability in the Wild" — fully reverse-engineered
the circuit responsible for the Indirect Object Identification task in GPT-2:

    "When Mary and John went to the store, John gave a drink to ___."
    → correct answer: "Mary"

    The circuit involves 26 attention heads across 13 functional types:
    • Duplicate Token Heads: detect that "John" appears twice → mark it
    • S-Inhibition Heads:    suppress the duplicated name (John)
    • Name Mover Heads:      copy the non-duplicated name (Mary) to the output
    • Induction Heads:       support the duplicate detection
    • Backup Name Movers:    fallback if primary name movers are ablated

    This is a COMPLETE mechanistic account: for this task, on this model,
    these specific heads implement the full algorithm. The circuit accounts
    for 85% of the model's performance on this task.


### Grokking — Circuits Forming During Training

Grokking (Power et al., 2022) is a striking training phenomenon:
a model first memorises training data (near-zero training loss but poor
generalisation), then suddenly "generalises" after many more training steps
on a problem it appeared to have "solved" long ago.

    Modular addition example: a + b mod 113
    • Phase 1: memorises training examples → 100% train acc, ~5% val acc
    • Phase 2: continues training (weight decay pushing toward simplicity)
    • Phase 3: suddenly jumps to ~100% val acc after 10× more steps

    Nanda et al. (2023) discovered the circuit GPT-2 small learns for
    modular arithmetic:
    1. Embed a and b into Fourier frequency space using the embedding matrix.
    2. Rotate in Fourier space (corresponding to addition mod N).
    3. Read off the answer from the rotated Fourier representation.
    The model learns a clean algorithmic solution — not pattern matching —
    but only after weight decay forces it to find the compressed representation.


### Copy Suppression Circuit

McDougall et al. (2023) discovered the COPY SUPPRESSION circuit:
a mechanism that PREVENTS the model from naively copying the most recently
seen token. In language, copying the previous token is often wrong (you
wouldn't repeat the last word). The circuit:

    1. Detection head: detects when the current query position's most likely
       prediction (via OV) is the same as a recently seen token.
    2. Suppression head: attends to the position of that recent token and
       subtracts its contribution from the output, reducing its probability.

This reveals that model behaviours emerge from COMPETITION between circuits,
not from a single pathway. A complete mechanistic account must include both
the "copy head" and the "copy suppression head" that counterbalances it.


##### PART 8 — SUPERPOSITION AND POLYSEMANTICITY

### The Linear Representation Hypothesis

The LINEAR REPRESENTATION HYPOTHESIS states:
Features (meaningful concepts) are represented as DIRECTIONS in the
residual stream (linear subspaces), not as individual neurons.

    Evidence:
    • Arithmetic in embedding space: word2vec's king − man + woman ≈ queen.
    • Binding: the representation of "red car" is the sum of "red" and "car"
      direction vectors (approximately).
    • Universality: the same feature directions emerge in different models
      trained independently on the same data.

    Strong form: every feature F is encoded as a direction dF such that
    the dot product x · dF measures the "amount" of feature F in x.


### The Superposition Hypothesis

If features are directions, how many features can a d-dimensional space hold?
Naively: d features, one per basis dimension. But Elhage et al. (2022)
showed models store MANY MORE than d features:

    SUPERPOSITION HYPOTHESIS:
    Neurons represent combinations of features.
    Features are encoded in overlapping directions.
    Each feature direction is NOT orthogonal to all others.
    The dot product between two feature vectors is small but non-zero.

    This is possible because:
    If features are SPARSE (only a few are active at once), the
    interference between non-orthogonal features is small.
    The model trades off precision for capacity: store many features
    with small interference, rather than few features with zero interference.

    ┌─────────────────────────────────────────────────────────────────┐
    │  SUPERPOSITION GEOMETRY                                         │
    │                                                                 │
    │  In 2D space: normally, 2 orthogonal features.                  │
    │  With superposition: 4 features at ±45° angles.                 │
    │  Interference: each feature pair has dot product 0.5.           │
    │  But if features are sparse (rarely co-active), the interference │
    │  is tolerable. 4 features > 2 features at the cost of small     │
    │  cross-feature noise.                                           │
    │                                                                 │
    │  In d=512: orthogonal limit = 512 features.                     │
    │  With superposition: potentially millions of features, all      │
    │  squeezed into 512 dimensions, with manageable interference.    │
    └─────────────────────────────────────────────────────────────────┘

Superposition is the MODEL'S SOLUTION to having more concepts to represent
than dimensions available. It is an efficient compression, at the cost of
making individual neurons polysemantic.


### Polysemanticity — The Consequence of Superposition

POLYSEMANTICITY (Elhage et al., 2022): a single neuron activates for multiple
semantically unrelated features.

    Example (from Anthropic's analysis of a small transformer):
    Neuron 2049 in MLP layer 4 activates for:
    • Base64-encoded strings
    • Finnish language
    • Variable names in Python code
    These three categories have NOTHING in common semantically — they are
    different features sharing the same neuron due to superposition.

    Why polysemanticity makes interpretability hard:
    1. You cannot read off a neuron's meaning from its top activating examples.
    2. Ablating a neuron affects multiple unrelated features simultaneously.
    3. There is no clean decomposition of MLP activations into meaningful units.

    Monosemanticity (the ideal): one neuron, one feature.
    Polysemanticity (the reality): one neuron, many features.

The goal of sparse autoencoders (Part 9) is to find a basis where
features ARE monosemantic — to undo the superposition and recover
the clean underlying features.


### Geometry of Features Under Superposition

Elhage et al. (2022) studied toy models trained to represent N sparse
features in M dimensions (N > M) and found:

    When N/M is small (few features per dimension):
    Features arrange in orthogonal pairs — the model finds nearly-orthogonal
    representations for each feature.

    When N/M is large (many features per dimension):
    Features arrange in POLYTOPE geometries: vertices of regular polytopes
    (tetrahedra, octahedra, icosahedra) pack the maximum number of nearly-
    equidistant directions into the available dimensions.

    ANTIPODAL STRUCTURE:
    Boolean features (present/absent) are often encoded as +v / −v
    (a direction and its negation). This is why you can negate concepts
    in representation space.

    PRIVILEGED BASIS:
    When a ReLU nonlinearity is applied to the hidden layer, the neurons
    (basis dimensions) become "privileged" — the model prefers to align
    features with the basis axes rather than arbitrary directions. This
    is the regime that makes individual neurons interpretable (or not).


##### PART 9 — SPARSE AUTOENCODERS (SAEs)

### The Dictionary Learning Approach

If superposition packs many features into few dimensions, the solution is to
find a LARGER dictionary of features that can sparsely represent any activation.

    SPARSE AUTOENCODER (SAE) architecture:
    Given residual stream activation x ∈ ℝ^{d_model}:

    Encoder: f = ReLU(W_enc × (x − b_pre) + b_enc)   (f ∈ ℝ^{d_features})
    Decoder: x̂ = W_dec × f + b_pre                   (x̂ ∈ ℝ^{d_model})

    where d_features >> d_model (e.g., 16× to 32× larger).

    Training objective:
    L = ||x − x̂||² + λ ||f||₁
    Reconstruction loss + L1 sparsity penalty on features.

    The L1 penalty forces most feature activations f_i to be zero.
    Each activation x is represented as a SPARSE combination of a few
    "dictionary vectors" (columns of W_dec).

    Goal: the dictionary vectors should correspond to MONOSEMANTIC features —
    each firing for a single interpretable concept.


### Anthropic's Monosemanticity Results

Bricken et al. (2023) — "Towards Monosemanticity: Decomposing Language Models
With Dictionary Learning":

    Trained SAEs on MLP activations from a 1-layer transformer.
    d_model = 512, d_features = 4096 (8× expansion).

    Results:
    • Found thousands of features that are highly interpretable.
    • Features are far MORE monosemantic than individual neurons.
    • Example feature 'curved' (feature 4711):
      Top activating tokens: bow, arch, arc, curve, crescent
      → Clean single concept: curved geometric shapes
    • The same neuron (via a neuron-level analysis) activated for
      many unrelated concepts; the SAE separated them.

    Templeton et al. (2024) — scaled to Claude 3 Sonnet (much larger model):
    • Found features for "The Eiffel Tower", "banana", "Pokémon",
      individual people (e.g., "Donald Trump"), concepts like "sycophancy."
    • The "sycophancy" feature, when artificially activated, caused the model
      to produce obsequious text — a causal demonstration of feature function.


### Evaluating SAE Features

    INTERPRETABILITY METRICS:
    • Activation density: what fraction of inputs activate this feature? Good
      features are sparse (0.01–1% of inputs).
    • Top-k examples: show the 20 inputs with highest feature activation.
      Are they semantically coherent?
    • Auto-interpretability: ask another LLM to describe what the feature
      detects from its top examples. Measure agreement with human labels.

    FAITHFULNESS METRICS:
    • Does ablating the SAE features (zeroing specific features before the
      decoder step) change model behaviour in the predicted direction?
    • Do features found by the SAE correspond to features found by other methods
      (probing, activation patching)?

    SPARSITY-QUALITY TRADE-OFF:
    Higher λ (more sparsity): fewer features fire per input, features are more
    monosemantic, but reconstruction quality drops. A useful feature should fire
    on 10–1000 inputs out of 100K, not 1 input (over-specialised) or
    50K inputs (not specific).


### The SAE Research Programme

    WHY SAEs ARE IMPORTANT:
    If SAEs succeed in decomposing activations into monosemantic features:
    1. We can READ the model's "thoughts": the active SAE features are a
       vocabulary of concepts the model is currently processing.
    2. We can STEER the model: artificially activating a feature (like
       "sycophancy") changes model behaviour in predictable ways.
    3. We can AUDIT safety properties: scan for features associated with
       harmful outputs, monitor their activation in production.

    OPEN CHALLENGES:
    • Do SAEs find the TRUE underlying features, or just a convenient basis?
    • Do SAE features generalise across models and tasks?
    • Scaling: current SAE training is expensive; frontier models need
      billion-feature dictionaries.
    • SAEs are trained on specific activations (one layer, one position type);
      a full model description requires SAEs at all layers.


##### PART 10 — REPRESENTATION ENGINEERING AND ACTIVATION STEERING

### Reading Internal Representations

REPRESENTATION ENGINEERING (RepE, Zou et al., 2023) systematically reads
and writes high-level concepts from model activations.

    READING A CONCEPT (Probing Direction):
    1. Create a dataset of contrastive pairs: (positive_context, negative_context)
       where positive_context contains the concept (e.g., "happy mood") and
       negative_context is the neutral or opposite (e.g., "neutral mood").
    2. Extract activations from layer l for each context.
    3. The CONCEPT DIRECTION d is the vector that best separates positive from
       negative activations:
       d = mean(positive_activations) − mean(negative_activations)
       Optionally: normalise d, or use PCA on the difference vectors.
    4. The model's "amount" of the concept at layer l is: x · d / ||d||

    Examples:
    Concepts identified by RepE: honesty, sycophancy, emotional valence,
    political ideology, topic (is this about food? sports? politics?),
    emotional state (happy, sad, anxious), truthfulness.


### Activation Steering — Writing Representations

ACTIVATION STEERING (Li et al., 2023; Turner et al., 2023):
Instead of reading a concept direction, INJECT it into the residual stream
to cause the model to "think" the concept is active.

    PROTOCOL:
    1. Find the concept direction d (via RepE or mean ablation).
    2. At layer l, add α × d to the residual stream: x_l ← x_l + α × d
    3. Continue the forward pass from layer l.
    4. Observe the output: does it reflect the steered concept?

    RESULTS FROM ACTIVATION STEERING EXPERIMENTS:
    • Steering with "banana" direction causes the model to generate text about
      bananas even when discussing unrelated topics.
    • Steering with "Assistant" direction in the "user" role can shift model
      behaviour toward helpful assistant mode.
    • Steering with "honesty" direction reduces deceptive outputs.
    • Steering with "fear" direction causes model to express anxiety.

    CAUTION: Steering large magnitudes (α >> 1) often causes incoherent
    outputs — the model's other computations conflict with the injected concept.
    Moderate steering (α ≈ 1–5) produces the cleanest results.


### Representation Engineering for Safety

RepE has direct safety applications:

    LIE DETECTION:
    Train a "truthfulness" direction from contrasting honest/deceptive completions.
    The model's activation on this direction during generation is a "lie detector."
    Experimental results show ~80% accuracy on detecting model-generated lies.

    EMOTION MONITORING:
    Read emotional state from activations. Models have internal emotional
    representations even if they don't express them. These representations
    can be monitored and steered.

    CONCEPT SUPPRESSION:
    Add −α × d (subtracting the concept direction) to suppress a concept.
    Applied to "harmful topic" directions, this can reduce harmful outputs
    without degrading general capability.

    LAYER CHOICE MATTERS:
    Middle layers (40–60% of depth) are most informative for most concepts.
    Early layers: syntax and surface features. Late layers: output preparation.


### Concept Arithmetic in Representation Space

The linearity of representations enables CONCEPT ARITHMETIC:

    "Female professional" = "professional" direction + "female" direction
    Injecting this combined direction steers the model toward female professional
    concepts (doctor, engineer, lawyer) rather than generic professional.

    CONCEPT NEGATION:
    Injecting −d (negation) suppresses the concept.

    CONCEPT INTERPOLATION:
    Interpolating between two concept directions causes smooth transitions
    in the output space.

    These operations demonstrate that the LLM's internal representation space
    has algebraic structure that can be exploited for controllable generation.


##### PART 11 — GRADIENT-BASED ATTRIBUTION METHODS

### Why Gradient Methods?

Gradient-based methods answer: "Which input features most affect this output?"
They compute the sensitivity of the output to each input feature using
backpropagation — which is already implemented in every deep learning framework.

    SALIENCY MAPS (Simonyan et al., 2013):
    Compute the gradient of the output with respect to the input:
    S_i = |∂ output / ∂ x_i|

    For LLMs: x_i is the embedding of token i. The saliency score of
    token i is the L2 norm of the gradient w.r.t. its embedding.

    HIGH saliency: removing or changing this token most affects the output.
    LOW saliency: the output is insensitive to this token.

    Limitation: saliency maps are LOCAL (linear approximation at the input
    point). They do not capture non-linear interactions between tokens.


### Integrated Gradients

INTEGRATED GRADIENTS (Sundararajan et al., 2017) fixes the locality problem
by integrating the gradient along a path from a BASELINE input to the actual
input:

    IG_i(x) = (x_i − x̄_i) × ∫₀¹ [∂ F / ∂ x_i] (x̄ + α(x − x̄)) dα

    where x̄ is the baseline (e.g., zero embedding or padding token)
    and F is the output score of interest.

    In practice, approximate the integral with M steps:
    IG_i(x) ≈ (x_i − x̄_i) × (1/M) Σₘ₌₁ᴹ [∂ F / ∂ x_i] (x̄ + (m/M)(x − x̄))

    AXIOMS satisfied by integrated gradients:
    1. COMPLETENESS: Σᵢ IG_i(x) = F(x) − F(x̄)  (attributions sum to the
       total output difference from baseline)
    2. SENSITIVITY: if only input i differs from the baseline and output
       changes, then IG_i ≠ 0.
    3. IMPLEMENTATION INVARIANCE: equivalent networks have identical
       attributions (unlike discrete approximation methods).

    For LLMs: integrated gradients can attribute the probability of a
    specific output token to each input token. This gives a quantitative
    answer to "which tokens caused this prediction?"


### Input × Gradient (GradNorm)

A common lightweight approximation to integrated gradients:

    GradNorm_i = ||x_i ⊙ ∂ F / ∂ x_i||₂

The element-wise product of the input embedding with its gradient.
Cheaper than integrated gradients (one backward pass instead of M).
Less theoretically grounded but often empirically similar.


### Attention × Gradient

ATTENTION × GRADIENT (Attattr, Barkan et al., 2021):
Multiply attention weights by their gradients to get attention-gradient scores:

    AttAttr_i = Σⱼ αᵢⱼ × (∂ F / ∂ αᵢⱼ)

This combines the "where the model looked" (attention weights) with "how
important was it to look there" (gradient w.r.t. the attention weight).
Outperforms raw attention weights on faithfulness metrics.


### SHAP for Transformers

SHAP (SHapley Additive exPlanations, Lundberg & Lee, 2017) computes attribution
scores based on the Shapley value from cooperative game theory:

    Shapley value of token i = average marginal contribution of token i
    across all possible subsets of tokens.

    For LLMs: use PARTITION SHAP (token-level attributions with a partition
    masker that preserves linguistic structure).

    PROPERTIES:
    • Local accuracy: attributions sum to the model's output.
    • Consistency: if a feature always contributes more, it gets a higher value.
    • Missingness: tokens absent from a subset get zero attribution.

    Cost: 2^n evaluations in the exact case (exponential in sequence length).
    Approximated with kernel SHAP using ~500–2000 model evaluations.


### Faithfulness vs Plausibility

FAITHFULNESS: does the attribution reflect the actual computation?
PLAUSIBILITY: does the attribution look reasonable to humans?

These two properties are NOT the same and can come apart:

    UNFAITHFUL BUT PLAUSIBLE:
    An explanation highlights the subject noun of the sentence as the
    most important token. This looks right to humans but may not reflect
    what the model actually used (maybe it was the verb tense that mattered).

    FAITHFUL BUT IMPLAUSIBLE:
    An explanation highlights a stop word ("the") as the most important.
    This may be correct (removing "the" changes the syntactic parse) but
    looks wrong to humans.

    EVALUATING FAITHFULNESS:
    Sufficiency: does the set of "important" tokens preserve the prediction?
    Necessity: does removing the "important" tokens destroy the prediction?
    Token erasure: set tokens to baseline, measure prediction change.
    Model collapse test: if the explanation method is correct, then
    replacing "unimportant" tokens with padding should not change the output.


### Limitations of Gradient Methods

    GRADIENT SATURATION:
    In ReLU networks, neurons can be saturated (gradient = 0 even though
    the feature matters). Integrated gradients addresses this; plain
    saliency maps do not.

    ADVERSARIAL ATTRIBUTION ATTACKS:
    Heo et al. (2019) showed that models can be adversarially trained to
    produce any desired attribution map while keeping the prediction constant.
    This means attribution maps can be completely disconnected from the
    computation if the model is fine-tuned to fool the explainer.

    LOCAL LINEARITY ASSUMPTION:
    All gradient methods approximate the model locally by its linear tangent.
    For highly nonlinear regions of the input space, this can be misleading.

    BASELINE SENSITIVITY (Integrated Gradients):
    The attributions depend on the choice of baseline. Different baselines
    (zero embedding, random token, [PAD] token) give different attributions.
    There is no universally correct baseline for language.


##### PART 12 — EVALUATION, FAITHFULNESS, AND OPEN PROBLEMS

### How Do We Know an Explanation Is Correct?

The fundamental challenge: any explanation of a neural network can be
wrong (unfaithful). We need rigorous evaluation methods.

    THE GROUND TRUTH PROBLEM:
    For most tasks, we do not know the true mechanistic explanation in advance.
    We cannot simply check if the explanation matches the "true" computation.

    PRINCIPLED EVALUATION STRATEGIES:

    1. CAUSAL INTERVENTION TEST:
       If an explanation claims component C is responsible for output O,
       then: ablating C should degrade O, and patching C should restore O
       in counterfactual settings. This is the gold standard.

    2. CIRCUIT COMPLETENESS:
       If we have identified a complete circuit, then ablating all
       NON-CIRCUIT components should leave model performance unchanged.
       Measure: performance with all non-circuit components ablated.

    3. CIRCUIT MINIMALITY:
       If we have identified a minimal circuit, then ablating any single
       component from the circuit should degrade performance.

    4. FAITHFULNESS METRICS (for attribution methods):
       • Comprehensiveness: remove top-k attributed tokens; measure prediction change.
       • Sufficiency: keep only top-k attributed tokens; measure if prediction holds.
       • AOPC (Area Over the Perturbation Curve): area under the performance
         curve as tokens are progressively removed in attribution order.


### The Superposition Problem for Evaluation

Many interpretability results are local: they hold for a specific model
(GPT-2 small), a specific task (IOI), and specific inputs. The key
open questions are:

    1. UNIVERSALITY: do the same circuits appear across different model
       families (GPT, LLaMA, Claude) and sizes?
       Current evidence: YES for some circuits (induction heads appear
       in every transformer studied), but far from general.

    2. SCALABILITY: do mechanistic interpretability methods that work on
       124M parameter models scale to 70B+ parameter models?
       Current state: partially — activation patching scales, full circuit
       analysis does not yet scale to frontier models.

    3. COMPOSITIONAL GENERALIZATION: can we compose circuits (circuit A + circuit B)
       to explain behaviors that require both? Or do circuits interact in
       ways that defeat composition?

    4. FEATURE COMPLETENESS: SAE features are interpretable, but are they
       COMPLETE — do they cover all relevant model computations, or do
       important computations happen "in the residual space" not captured
       by any SAE feature?


### The Faithfulness–Complexity Trade-off

    SIMPLE EXPLANATIONS:
    "Attention head 7.3 copies the subject to the output position."
    Easy to understand, but may miss 30% of the computation.

    COMPLETE EXPLANATIONS:
    All 26 heads and their interactions in the IOI circuit.
    Faithful, but cognitively demanding — hard to reason about.

    The right level of abstraction depends on the use case:
    • Safety auditing: complete is better (you can't miss a failure mode).
    • Debugging: simple is better (find the broken component quickly).
    • Scientific understanding: complete but compressed (find the algorithms).


### Interpretability for Alignment

The deepest motivation for interpretability is ALIGNMENT:

    SYCOPHANCY DETECTION:
    If a model's internal representation of "sycophancy" (telling users
    what they want to hear) can be read and steered, we can detect and
    suppress sycophantic reasoning at the activation level — not just
    the output level.

    DECEPTIVE ALIGNMENT:
    A model trained to appear aligned while "knowing" it is misaligned
    would require the model's internal representations to reflect the
    deception. Interpretability tools that read internal states could
    detect this before it manifests in behaviour.

    CONCEPT-LEVEL AUDITING:
    Features found by SAEs can be scanned: does activating "harmful plan"
    concepts cause model behaviour changes? Which inputs trigger those features?
    This allows systematic auditing without exhaustive behavioural testing.

    GOAL REPRESENTATION:
    Does the model have an internal representation of "my goal is to be helpful"?
    Or of "my goal is to satisfy the human regardless of truth"? Interpretability
    methods can test which goal representations are active and whether they
    causally influence outputs.


### Current Frontiers (2024–2025)

    SAE SCALING:
    Anthropic, EleutherAI, and independent researchers are scaling SAEs
    to frontier models (Claude 3, Llama 3). The first interpretable features
    in frontier-class models are being identified.

    CROSS-LAYER CIRCUITS:
    Moving beyond single-layer analysis toward multi-layer circuit tracing
    that follows information from input to output across all layers.

    AUTOMATED INTERPRETABILITY:
    Using LLMs to automatically label SAE features (auto-interp) and
    to discover circuits (model-written interpretability reports).

    MECHANISTIC ANOMALY DETECTION:
    Using interpretability to detect anomalous activations that might
    indicate jailbreaking, unusual reasoning, or potential deception —
    a path toward AI oversight using AI interpretability.

    CHAIN-OF-THOUGHT FAITHFULNESS:
    Does a model's stated reasoning (in a scratchpad or CoT) match its
    internal computation? Initial evidence suggests PARTIAL faithfulness —
    CoT reasoning is often a post-hoc rationalisation rather than a
    causal account of the computation.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔═══════════════════════════╦══════════════╦══════════════╦═══════════════╦══════════════╗
    ║ Method                    ║ Type         ║ Granularity  ║ Causal?       ║ Scale        ║
    ╠═══════════════════════════╬══════════════╬══════════════╬═══════════════╬══════════════╣
    ║ Attention analysis        ║ Behavioral   ║ Head-level   ║ No (corr.)    ║ Any model    ║
    ║ Logit lens                ║ Behavioral   ║ Layer-level  ║ No (corr.)    ║ Any model    ║
    ║ Tuned lens                ║ Behavioral   ║ Layer-level  ║ No (corr.)    ║ Any model    ║
    ║ Linear probing            ║ Behavioral   ║ Layer-level  ║ No (corr.)    ║ Any model    ║
    ║ Causal probing            ║ Behavioral   ║ Layer-level  ║ Partial       ║ Any model    ║
    ║ Activation patching       ║ Mechanistic  ║ Component    ║ Yes           ║ Up to ~7B    ║
    ║ Causal tracing (ROME)     ║ Mechanistic  ║ Layer/pos    ║ Yes           ║ Up to ~7B    ║
    ║ Circuit analysis          ║ Mechanistic  ║ Head/neuron  ║ Yes           ║ Small models ║
    ║ Sparse autoencoders (SAE) ║ Mechanistic  ║ Feature      ║ Partial       ║ Frontier     ║
    ║ Activation steering       ║ Behavioral   ║ Layer-level  ║ Yes (causal)  ║ Any model    ║
    ║ Saliency maps             ║ Attribution  ║ Token-level  ║ No (approx.)  ║ Any model    ║
    ║ Integrated gradients      ║ Attribution  ║ Token-level  ║ Partial       ║ Any model    ║
    ║ SHAP                      ║ Attribution  ║ Token-level  ║ No (approx.)  ║ Any model    ║
    ║ Repr. engineering (RepE)  ║ Behavioral   ║ Layer-level  ║ Yes (causal)  ║ Any model    ║
    ╚═══════════════════════════╩══════════════╩══════════════╩═══════════════╩══════════════╝

    Key concepts:
      Residual stream:   x_l = x_{l-1} + attn_l(x_{l-1}) + mlp_l(x_{l-1})
      Attention head:    attn_h(x) = softmax(Q_h K_h^T / √d) × V_h × W_O_h
      Logit lens:        logits_l = W_U × x_l  (mid-model prediction at layer l)
      Activation patch:  recovery = (logit_patched − logit_corrupt) / (logit_clean − logit_corrupt)
      Integrated gradient: IG_i = (x_i − x̄_i) × ∫₀¹ (∂F/∂x_i)(x̄ + α(x−x̄)) dα
      SAE:               f = ReLU(W_enc(x − b_pre) + b_enc),  x̂ = W_dec f + b_pre

    Interpretability hierarchy:
      Attribution methods → tell you which tokens mattered (correlation)
      Behavioral probing  → tell you what is represented (correlation)
      Activation patching → tells you what is causally used (causal)
      Circuit analysis    → tells you the full algorithm (causal + mechanistic)
      SAE features        → recover monosemantic basis (interpretable + causal)
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Attention Analysis — Patterns, Induction Heads, and Head Ablation": {
        "description": (
            "Build a minimal transformer from scratch in pure NumPy. "
            "Run forward passes and extract attention weights from each head and layer. "
            "Identify and visualise attention pattern types: previous-token heads, "
            "uniform heads, and the induction head pattern (copy-prefix completion). "
            "Implement head ablation: zero out individual heads and measure the "
            "effect on next-token probability. "
            "Show QK and OV circuit decomposition for a single head."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  ATTENTION ANALYSIS — PATTERNS, INDUCTION HEADS, ABLATION")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Minimal transformer building blocks (pure numpy)
# ─────────────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


class MultiHeadAttention:
    """
    Multi-head attention with full weight matrices.
    Stores attention weights for interpretability analysis.
    """
    def __init__(self, d_model, n_heads):
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        scale         = 0.02
        # W_Q, W_K, W_V: [d_model, n_heads * d_head]
        # W_O: [n_heads * d_head, d_model]
        self.W_Q = np.random.randn(d_model, d_model)     * scale
        self.W_K = np.random.randn(d_model, d_model)     * scale
        self.W_V = np.random.randn(d_model, d_model)     * scale
        self.W_O = np.random.randn(d_model, d_model)     * scale
        self.b_Q = np.zeros(d_model)
        self.b_K = np.zeros(d_model)
        self.b_V = np.zeros(d_model)
        self.b_O = np.zeros(d_model)
        # Store for interpretability
        self.last_attn_weights = None  # [n_heads, seq_len, seq_len]
        self.last_head_outputs = None  # [n_heads, seq_len, d_model]

    def forward(self, x, mask=None, ablate_heads=None):
        """
        x: [seq_len, d_model]
        ablate_heads: set of head indices to zero out
        Returns: [seq_len, d_model], updates self.last_attn_weights
        """
        seq_len, _ = x.shape
        # Compute Q, K, V — [seq_len, d_model]
        Q = x @ self.W_Q + self.b_Q
        K = x @ self.W_K + self.b_K
        V = x @ self.W_V + self.b_V

        # Reshape to [n_heads, seq_len, d_head]
        Q = Q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        # Attention scores [n_heads, seq_len, seq_len]
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_head)

        # Causal mask: upper triangle = -inf
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + causal_mask[None, :, :]

        # Attention weights [n_heads, seq_len, seq_len]
        attn_weights = softmax(scores, axis=-1)
        self.last_attn_weights = attn_weights.copy()

        # Head outputs [n_heads, seq_len, d_head]
        head_out = attn_weights @ V  # [n_heads, seq_len, d_head]

        # Ablate specified heads
        ablated_head_out = head_out.copy()
        if ablate_heads:
            for h in ablate_heads:
                ablated_head_out[h] = 0.0

        # Reshape back to [seq_len, n_heads*d_head] and project
        combined = ablated_head_out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        output   = combined @ self.W_O + self.b_O

        # Store per-head contributions to the residual stream
        self.last_head_outputs = []
        for h in range(self.n_heads):
            h_out = head_out[h].reshape(seq_len, self.d_head)
            # Each head h contributes h_out @ W_O[h*d_head:(h+1)*d_head, :]
            w_slice = self.W_O[h*self.d_head:(h+1)*self.d_head, :]
            self.last_head_outputs.append(h_out @ w_slice)

        return output


class TransformerLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn     = MultiHeadAttention(d_model, n_heads)
        self.ff_W1    = np.random.randn(d_model, d_ff)   * 0.02
        self.ff_b1    = np.zeros(d_ff)
        self.ff_W2    = np.random.randn(d_ff,    d_model) * 0.02
        self.ff_b2    = np.zeros(d_model)
        self.ln1_g    = np.ones(d_model)
        self.ln1_b    = np.zeros(d_model)
        self.ln2_g    = np.ones(d_model)
        self.ln2_b    = np.zeros(d_model)

    def forward(self, x, ablate_heads=None):
        # Attention sub-layer
        x = x + self.attn.forward(layer_norm(x, self.ln1_g, self.ln1_b),
                                   ablate_heads=ablate_heads)
        # MLP sub-layer
        xn = layer_norm(x, self.ln2_g, self.ln2_b)
        x  = x + gelu(xn @ self.ff_W1 + self.ff_b1) @ self.ff_W2 + self.ff_b2
        return x


class MiniTransformer:
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq=64):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.embed      = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed  = np.random.randn(max_seq,    d_model) * 0.02
        self.layers     = [TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.unembed    = np.random.randn(d_model, vocab_size) * 0.02
        self.residual_stream_cache = []   # stores x_l after each layer

    def forward(self, tokens, ablate_spec=None):
        """
        tokens: list of int token ids
        ablate_spec: dict {layer_idx: set(head_indices)} to ablate
        Returns: logits [seq_len, vocab_size], residual_stream [n_layers+1, seq_len, d_model]
        """
        tokens_arr = np.array(tokens)
        x = self.embed[tokens_arr] + self.pos_embed[:len(tokens_arr)]
        self.residual_stream_cache = [x.copy()]

        ablate_spec = ablate_spec or {}
        for l, layer in enumerate(self.layers):
            x = layer.forward(x, ablate_heads=ablate_spec.get(l, None))
            self.residual_stream_cache.append(x.copy())

        logits = x @ self.unembed
        return logits


# ─────────────────────────────────────────────────────────────────────────
# Create a small model and vocabulary for demonstration
# ─────────────────────────────────────────────────────────────────────────

VOCAB      = ["[PAD]", "[BOS]", "the", "cat", "sat", "on", "mat",
              "dog", "ran", "in", "park", "a", "and", "big", "small",
              "happy", "sad", "quickly", "slowly", ".", ","]
V  = {w: i for i, w in enumerate(VOCAB)}
d_model, n_layers, n_heads, d_ff = 64, 2, 4, 128
model = MiniTransformer(vocab_size=len(VOCAB), d_model=d_model,
                        n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)

print(f"  Transformer: {n_layers} layers, {n_heads} heads/layer, d_model={d_model}")
print(f"  Vocabulary: {len(VOCAB)} tokens")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Attention weight extraction and pattern classification")
print("━" * 65)
print()

def classify_attention_pattern(attn_matrix):
    """
    Classify a [seq_len, seq_len] attention weight matrix.
    Returns a description of the dominant pattern.
    """
    seq_len = attn_matrix.shape[0]
    if seq_len < 2: return "trivial"

    # Check for diagonal (self-attention)
    diag_mass = np.diag(attn_matrix[1:, :-1]).mean() if seq_len > 1 else 0

    # Check for previous-token (subdiagonal)
    prev_tok_mass = np.diag(attn_matrix[1:, :-1]).mean() if seq_len > 1 else 0

    # Check for position 0 dominance
    pos0_mass = attn_matrix[:, 0].mean()

    # Check for induction pattern (attend to pos_of_last_same_token + 1)
    # Simplification: measure mass on the position 2 behind current
    shift2_mass = 0.0
    for i in range(2, seq_len):
        shift2_mass += attn_matrix[i, i-2]
    shift2_mass /= max(seq_len - 2, 1)

    # Compute entropy of each row
    entropies = []
    for i in range(seq_len):
        p = attn_matrix[i]
        p = np.clip(p, 1e-9, 1.0)
        entropies.append(-np.sum(p * np.log(p)))
    mean_entropy = np.mean(entropies)

    if pos0_mass > 0.6:
        return "beginning-of-seq (BOS)"
    elif prev_tok_mass > 0.5:
        return "previous-token"
    elif shift2_mass > 0.3:
        return "induction-like"
    elif mean_entropy > 2.0:
        return "uniform/diffuse"
    else:
        return "local/mixed"

sentence = ["[BOS]", "the", "cat", "sat", "on", "the", "mat", "."]
tokens   = [V.get(w, 0) for w in sentence]

_ = model.forward(tokens)

print("  Sentence:", " ".join(sentence))
print()
print("  Attention pattern classification by head:")
print(f"  {'Layer':>6} {'Head':>5}  {'Pattern':<28}  {'Mean entropy':>14}  {'BOS mass':>10}  {'Prev-tok mass':>14}")
print(f"  {'─'*78}")

for layer_idx, layer in enumerate(model.layers):
    attn_w = layer.attn.last_attn_weights  # [n_heads, seq_len, seq_len]
    for h in range(n_heads):
        A   = attn_w[h]
        pat = classify_attention_pattern(A)
        # compute stats
        entropies = []
        for i in range(len(tokens)):
            p = np.clip(A[i], 1e-9, 1.0)
            entropies.append(-np.sum(p * np.log(p)))
        ent      = np.mean(entropies)
        bos_mass = A[:, 0].mean()
        prev_mass = np.diag(A[1:, :-1]).mean() if len(tokens) > 1 else 0
        print(f"  {layer_idx:>6} {h:>5}  {pat:<28}  {ent:>14.3f}  {bos_mass:>10.3f}  {prev_mass:>14.3f}")

print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Induction head pattern detection")
print("━" * 65)
print()

# Create an induction test: [A, B, C, ..., A, ?] pattern
# The induction head should attend from second A to B (the token after first A)
INDUCTION_VOCAB = list("ABCDEFGHIJKLMNOP")  # 16 tokens as sub-vocab
# Map to our vocab indices (reuse first few)
induction_seq = [0, 1, 2, 3, 4,   0, 1, 5, 6]  # [..., 0, 1, ..., 0, (predict 1)]
# Manually set W_Q and W_K for one head to implement induction pattern
# (demonstrate what the weights would look like)

print("  Test sequence for induction detection:")
seq_names = ["A", "B", "C", "D", "E", "A", "B", "?", "?"]
print("  Tokens:", " ".join(seq_names[:len(induction_seq)]))
print()
print("  For an induction head, when we see the second 'A' (position 5),")
print("  the head should attend strongly to position 1 (the 'B' after first 'A').")
print()
print("  In a truly trained induction head, the mechanism works as follows:")
print()
print("  Step 1 — Previous Token Head (earlier layer):")
print("    At each position i, writes 'previous token = x[i-1]' to residual stream.")
print("    Attention matrix looks like: diagonal at offset -1 (attend to i-1).")
print()
print("  Step 2 — Induction Head (later layer):")
print("    Uses the 'previous token' signal as a KEY.")
print("    The QUERY at position i (second 'A') asks: 'where was previous-token = A?'")
print("    The answer: position 1 (because previous-token at pos 1 was 'A' = pos 0).")
print("    Head attends to position 1, copies 'B' forward → predicts 'B'.")
print()

# Simulate what the induction head's attention matrix would look like
seq_len = len(induction_seq)
simulated_induction_attn = np.zeros((seq_len, seq_len))
# For each position, attend to the position after the last occurrence of current token
for i in range(seq_len):
    curr_tok = induction_seq[i]
    # find last occurrence of curr_tok before position i
    best_j = None
    for j in range(i-1, -1, -1):
        if induction_seq[j] == curr_tok:
            best_j = j + 1  # attend to position AFTER the match
            break
    if best_j is not None and best_j < seq_len:
        simulated_induction_attn[i, best_j] = 0.85
        # spread remaining 0.15 uniformly
        for k in range(seq_len):
            if k != best_j:
                simulated_induction_attn[i, k] = 0.15 / (seq_len - 1)
    else:
        # uniform if no prior occurrence
        simulated_induction_attn[i] = 1.0 / seq_len

print("  Simulated induction head attention matrix:")
print("  (rows = query position, cols = key position, values = attention weight)")
print()
header = "         " + "".join(f"  {s:2}" for s in seq_names[:seq_len])
print(f"  {header}")
for i in range(seq_len):
    row = "".join(f"  {simulated_induction_attn[i,j]:.2f}" for j in range(seq_len))
    marker = " ← induction!" if i == 5 and simulated_induction_attn[i, 1] > 0.5 else ""
    print(f"  pos {i} ({seq_names[i]}){row}{marker}")

print()
print("  Position 5 (second 'A') attends strongly to position 1 (first 'B')")
print("  → the model predicts B, implementing copy-prefix completion.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Head ablation: measuring each head's contribution")
print("━" * 65)
print()

sentence2 = ["[BOS]", "the", "dog", "ran", "in", "the", "park", "."]
tokens2   = [V.get(w, 0) for w in sentence2]
target_token = "park"
target_id    = V.get(target_token, 0)

# Baseline: no ablation
baseline_logits = model.forward(tokens2)
# Get probability of target at position 6 (predicting "park" from "the")
baseline_prob = softmax(baseline_logits[5:6])[0, target_id]

print(f"  Sentence: {' '.join(sentence2)}")
print(f"  Measuring contribution of each head to P('{target_token}') at position 6.")
print()
print(f"  Baseline P('{target_token}') = {baseline_prob:.6f}")
print()
print(f"  {'Layer':>6} {'Head':>5} {'P after ablation':>18} {'Absolute Δ':>12} {'Relative Δ':>12} {'Importance'}")
print(f"  {'─'*75}")

ablation_results = []
for layer_idx in range(n_layers):
    for h in range(n_heads):
        ablated_logits = model.forward(tokens2, ablate_spec={layer_idx: {h}})
        ablated_prob   = softmax(ablated_logits[5:6])[0, target_id]
        delta          = baseline_prob - ablated_prob
        rel_delta      = delta / (baseline_prob + 1e-9)
        # Importance: how much ablating this head hurts
        importance = "critical" if abs(delta) > 0.1 * baseline_prob else \
                     ("moderate" if abs(delta) > 0.02 * baseline_prob else "minimal")
        ablation_results.append((layer_idx, h, ablated_prob, delta, rel_delta))
        sign = "↓" if delta > 0 else "↑"
        print(f"  {layer_idx:>6} {h:>5} {ablated_prob:>18.6f} {delta:>+12.6f} "
              f"{rel_delta:>+12.1%} {importance}")

print()
top_ablations = sorted(ablation_results, key=lambda r: abs(r[3]), reverse=True)[:3]
print("  Most impactful heads (largest absolute change when ablated):")
for layer_idx, h, ablated_prob, delta, rel in top_ablations:
    direction = "reduces" if delta > 0 else "increases"
    print(f"    Layer {layer_idx}, Head {h}: ablating {direction} P('{target_token}') "
          f"by {abs(delta):.6f} ({abs(rel):.1%})")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — QK and OV circuit decomposition")
print("━" * 65)
print()
print("  For a single head, decompose into:")
print("  QK circuit: W_Q^T W_K  — which token pairs attract attention")
print("  OV circuit: W_V W_O    — what information is moved from attended positions")
print()

layer0_head0 = model.layers[0].attn
d_h = d_model // n_heads

# Extract sub-matrices for head 0
W_Q0 = layer0_head0.W_Q[:, :d_h]   # [d_model, d_head]
W_K0 = layer0_head0.W_K[:, :d_h]   # [d_model, d_head]
W_V0 = layer0_head0.W_V[:, :d_h]   # [d_model, d_head]
W_O0 = layer0_head0.W_O[:d_h, :]   # [d_head, d_model]

# QK circuit: how token at position i "queries" token at position j
# W_QK = W_Q @ W_K.T — maps (residual stream → residual stream) bi-linearly
W_QK = W_Q0 @ W_K0.T  # [d_model, d_model]

# OV circuit: what information is moved when this head attends to a token
# W_OV = W_V @ W_O — maps (residual stream → residual stream)
W_OV = W_V0 @ W_O0  # [d_model, d_model]

# Analyse singular values of QK and OV to understand their effective rank
sv_QK = np.linalg.svd(W_QK, compute_uv=False)
sv_OV = np.linalg.svd(W_OV, compute_uv=False)

print(f"  Layer 0, Head 0:")
print(f"  W_QK shape: {W_QK.shape}  (QK circuit, {d_model}×{d_model})")
print(f"  W_OV shape: {W_OV.shape}  (OV circuit, {d_model}×{d_model})")
print()

def effective_rank(sv):
    """Effective rank = exp(entropy of normalized singular value distribution)."""
    sv_norm = sv / sv.sum()
    ent = -np.sum(sv_norm * np.log(sv_norm + 1e-9))
    return np.exp(ent)

print(f"  QK circuit singular values (top 5): {sv_QK[:5].tolist()}")
print(f"  OV circuit singular values (top 5): {sv_OV[:5].tolist()}")
print()
print(f"  QK effective rank: {effective_rank(sv_QK):.1f} / {d_model}")
print(f"  OV effective rank: {effective_rank(sv_OV):.1f} / {d_model}")
print()
print("  A LOW effective rank means the head's QK/OV circuit operates in a")
print("  low-dimensional subspace — it implements a simple, focused function.")
print("  A HIGH effective rank means the circuit is complex and hard to interpret.")
print()
print("  The QK circuit determines WHAT the head attends to.")
print("  The OV circuit determines WHAT INFORMATION gets moved to attended positions.")
print("  Interpreting a head = understanding BOTH circuits simultaneously.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Logit Lens and Probing Classifiers — What Each Layer Knows": {
        "description": (
            "Implement the logit lens: read predictions from intermediate residual streams. "
            "Track how the top predicted token changes from layer 0 to layer L. "
            "Implement and evaluate linear probing classifiers for multiple properties "
            "(position, token identity, syntactic class, bigram context). "
            "Show how representation geometry changes across layers. "
            "Compare probing accuracy with a random baseline to test significance. "
            "Demonstrate the distinction between correlation (probe) and causation "
            "(what the model actually uses for prediction)."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  LOGIT LENS AND PROBING CLASSIFIERS — WHAT EACH LAYER KNOWS")
print("=" * 65)
print()

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ─────────────────────────────────────────────────────────────────────────
# Set up a simple transformer with controllable weights
# We'll engineer specific weights to demonstrate logit lens clearly
# ─────────────────────────────────────────────────────────────────────────

class SimpleLayer:
    """Single transformer layer storing residual stream for logit lens."""
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        s = 0.02
        self.W_Q = np.random.randn(d_model, d_model) * s
        self.W_K = np.random.randn(d_model, d_model) * s
        self.W_V = np.random.randn(d_model, d_model) * s
        self.W_O = np.random.randn(d_model, d_model) * s
        self.W1  = np.random.randn(d_model, d_ff)    * s
        self.W2  = np.random.randn(d_ff,    d_model)  * s
        self.b1  = np.zeros(d_ff)
        self.b2  = np.zeros(d_model)
        self.lg  = np.ones(d_model);  self.lb = np.zeros(d_model)
        self.mg  = np.ones(d_model);  self.mb = np.zeros(d_model)

    def forward(self, x):
        seq_len = x.shape[0]
        # Attention
        Q = layer_norm(x, self.lg, self.lb) @ self.W_Q
        K = layer_norm(x, self.lg, self.lb) @ self.W_K
        V = layer_norm(x, self.lg, self.lb) @ self.W_V
        Q = Q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_head)
        scores += np.triu(np.ones((seq_len, seq_len)), k=1)[None] * -1e9
        attn = softmax(scores, axis=-1)
        out  = (attn @ V).transpose(1, 0, 2).reshape(seq_len, self.d_model) @ self.W_O
        x    = x + out
        # MLP
        xn = layer_norm(x, self.mg, self.mb)
        x  = x + gelu(xn @ self.W1 + self.b1) @ self.W2 + self.b2
        return x


class LayeredTransformer:
    """Transformer that stores all intermediate residual streams."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.embed      = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed  = np.random.randn(128, d_model) * 0.02
        self.layers     = [SimpleLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.unembed    = np.random.randn(d_model, vocab_size) * 0.02
        self.final_norm_g = np.ones(d_model)
        self.final_norm_b = np.zeros(d_model)
        self.residual_streams = []  # filled by forward()

    def forward(self, tokens):
        tokens_arr = np.array(tokens)
        x = self.embed[tokens_arr] + self.pos_embed[:len(tokens_arr)]
        self.residual_streams = [x.copy()]  # layer 0 = embeddings
        for layer in self.layers:
            x = layer.forward(x)
            self.residual_streams.append(x.copy())
        x_final = layer_norm(x, self.final_norm_g, self.final_norm_b)
        logits   = x_final @ self.unembed
        return logits


VOCAB = (
    ["[PAD]", "[BOS]", "[EOS]"] +
    list("abcdefghijklmnopqrstuvwxyz") +
    ["the", "cat", "sat", "on", "mat", "dog", "ran", "in", "park",
     "Paris", "France", "London", "capital", "city", "is", "of",
     "The", "Its", "famous", "tower", "Eiffel", "beautiful", "."]
)
VOCAB = VOCAB[:64]  # cap at 64 tokens
V = {w: i for i, w in enumerate(VOCAB)}

d_model, n_layers, n_heads, d_ff = 64, 6, 4, 128
model = LayeredTransformer(len(VOCAB), d_model, n_layers, n_heads, d_ff)

print(f"  Transformer: {n_layers} layers, d_model={d_model}")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The Logit Lens: reading predictions at each layer")
print("━" * 65)
print()

def logit_lens(model, tokens, top_k=3):
    """
    Apply the unembedding matrix to each layer's residual stream.
    Returns the top-k predicted tokens at each layer.
    """
    _ = model.forward(tokens)  # populates residual_streams
    results = []
    for l, x in enumerate(model.residual_streams):
        # Apply final layer norm (approximation — tuned lens would use per-layer norms)
        x_normed = layer_norm(x, model.final_norm_g, model.final_norm_b)
        logits_l = x_normed @ model.unembed  # [seq_len, vocab_size]
        # Top-k for last position
        last_logits = logits_l[-1]
        probs       = softmax(last_logits)
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds   = [(VOCAB[i] if i < len(VOCAB) else f"[{i}]", probs[i])
                       for i in top_indices]
        results.append(top_preds)
    return results

# Test sentences
sentences = {
    "Fact recall": ["[BOS]", "The", "Eiffel", "tower", "is", "in"],
    "Copy pattern": ["[BOS]", "Paris", "is", "the", "capital", "of"],
    "Next word": ["[BOS]", "the", "cat", "sat", "on", "the"],
}

for name, sentence in sentences.items():
    valid_tokens = [V.get(w, 0) for w in sentence]
    logit_lens_results = logit_lens(model, valid_tokens, top_k=3)

    print(f"  Sentence: {' '.join(sentence)}")
    print(f"  Predicting next token after: '{sentence[-1]}'")
    print(f"  {'Layer':>7}  {'Top-1':>14}  {'Top-2':>14}  {'Top-3':>14}")
    print(f"  {'─'*58}")
    for l_idx, preds in enumerate(logit_lens_results):
        layer_name = "embed" if l_idx == 0 else f"layer {l_idx}"
        pred_strs  = [f"{tok}({prob:.3f})" for tok, prob in preds]
        print(f"  {layer_name:>7}  {pred_strs[0]:>14}  {pred_strs[1]:>14}  {pred_strs[2]:>14}")
    print()

print("  NOTE: With randomly initialised weights, the logit lens shows random")
print("  predictions. In a TRAINED model, you would see meaningful patterns:")
print("  early layers → generic/grammatical tokens, late layers → specific answers.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Linear probing classifiers")
print("━" * 65)
print()

print("  Building a probing dataset from the transformer's residual streams.")
print("  We probe for: token_id, position, syntactic class, sequence length.")
print()

# Generate a dataset of (residual_stream_activation, label) pairs
PROBE_SENTENCES = [
    ["[BOS]", "the", "cat", "sat", "on", "the", "mat"],
    ["[BOS]", "the", "dog", "ran", "in", "the", "park"],
    ["[BOS]", "Paris", "is", "the", "capital", "of", "France"],
    ["[BOS]", "the", "Eiffel", "tower", "is", "beautiful"],
    ["[BOS]", "the", "cat", "sat", "on", "the", "cat"],
    ["[BOS]", "dog", "ran", "in", "the", "park", "."],
]

class LinearProbe:
    """
    Binary linear probe: logistic regression on representations.
    Trained with gradient descent. Tests if a property is linearly decodable.
    """
    def __init__(self, d_in, lr=0.1, n_iter=200):
        self.w = np.random.randn(d_in) * 0.01
        self.b = 0.0
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        """X: [n, d], y: [n] binary labels 0/1"""
        n = len(y)
        for _ in range(self.n_iter):
            logits  = X @ self.w + self.b
            p       = sigmoid(logits)
            grad_w  = X.T @ (p - y) / n
            grad_b  = (p - y).mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        return (sigmoid(X @ self.w + self.b) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()


# Collect activations and labels across all layers
print("  Probing property: 'Is this token a content word (non-function word)?'")
print()

FUNCTION_WORDS = {"[BOS]", "[PAD]", "[EOS]", "the", "is", "of", "on", "in", "a", "an"}

# Collect data
all_activations = {l: [] for l in range(n_layers + 1)}
all_labels       = []

for sentence in PROBE_SENTENCES:
    tokens = [V.get(w, 0) for w in sentence]
    _ = model.forward(tokens)
    for pos, (word, tok_id) in enumerate(zip(sentence, tokens)):
        label = 0 if word in FUNCTION_WORDS else 1
        for l in range(n_layers + 1):
            all_activations[l].append(model.residual_streams[l][pos].copy())
        all_labels.append(label)

all_labels = np.array(all_labels)
n_samples  = len(all_labels)

# Train/test split (80/20)
split = int(0.8 * n_samples)
train_idx = np.arange(split)
test_idx  = np.arange(split, n_samples)
y_train, y_test = all_labels[train_idx], all_labels[test_idx]

print(f"  Dataset: {n_samples} (token, layer) pairs, {y_train.sum()} content words")
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
print()

# Baseline: majority class
majority = (y_test == y_test.mean().round()).mean()
random_baseline = max(y_test.mean(), 1 - y_test.mean())

print(f"  {'Layer':>8}  {'Train acc':>10}  {'Test acc':>10}  {'vs baseline':>12}  {'Interpretation'}")
print(f"  {'─'*65}")

for l in range(n_layers + 1):
    X_train = np.stack([all_activations[l][i] for i in train_idx])
    X_test  = np.stack([all_activations[l][i] for i in test_idx])

    probe = LinearProbe(d_model, lr=0.5, n_iter=500)
    probe.fit(X_train, y_train)
    tr_acc  = probe.accuracy(X_train, y_train)
    te_acc  = probe.accuracy(X_test,  y_test)
    delta   = te_acc - random_baseline

    interp = "above chance" if delta > 0.05 else ("at chance" if delta > -0.05 else "below chance")
    layer_name = "embed" if l == 0 else f"layer {l}"
    print(f"  {layer_name:>8}  {tr_acc:>10.3f}  {te_acc:>10.3f}  {delta:>+12.3f}  {interp}")

print()
print(f"  Random baseline (majority class): {random_baseline:.3f}")
print()
print("  INTERPRETATION:")
print("  - If probe accuracy is above chance at layer 0: the EMBEDDING already")
print("    encodes this syntactic distinction.")
print("  - If accuracy rises with layer: the property is being computed")
print("    incrementally across layers.")
print("  - If accuracy falls at deep layers: the property is 'forgotten'")
print("    (overwritten by more task-relevant information).")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Representational similarity across layers")
print("━" * 65)
print()

print("  Representational Similarity Analysis (RSA):")
print("  Do layers that represent similar content have similar geometry?")
print()

def representational_similarity(A_l, A_m):
    """
    RSA between layers l and m.
    Computes Spearman correlation of pairwise distance matrices.
    A_l, A_m: [n_samples, d_model] activation matrices.
    """
    n = len(A_l)
    # Pairwise squared Euclidean distances
    dists_l = np.array([[np.sum((A_l[i]-A_l[j])**2) for j in range(n)] for i in range(n)])
    dists_m = np.array([[np.sum((A_m[i]-A_m[j])**2) for j in range(n)] for i in range(n)])
    # Flatten upper triangle
    idx  = np.triu_indices(n, k=1)
    dl   = dists_l[idx]; dm = dists_m[idx]
    # Spearman correlation (rank correlation)
    def rank(x): return np.argsort(np.argsort(x)).astype(float)
    rl = rank(dl); rm = rank(dm)
    rl -= rl.mean(); rm -= rm.mean()
    return np.dot(rl, rm) / (np.linalg.norm(rl) * np.linalg.norm(rm) + 1e-9)

# Only use a small sample for speed
sample_size = min(20, n_samples)
act_matrices = {}
for l in range(n_layers + 1):
    act_matrices[l] = np.stack([all_activations[l][i] for i in range(sample_size)])

print(f"  RSA matrix (Spearman correlation of pairwise distances), n={sample_size} samples")
print()
header = "        " + "".join(f"  L{l}" for l in range(n_layers + 1))
print(f"  {header}")
for l in range(n_layers + 1):
    row = "".join(f"  {representational_similarity(act_matrices[l], act_matrices[m]):+.2f}"
                  for m in range(n_layers + 1))
    print(f"  Layer {l}{row}")

print()
print("  Diagonal = 1.0 (layer with itself).")
print("  High off-diagonal = geometrically similar representations.")
print("  Low values = representations have been restructured significantly.")
print("  In trained models: adjacent layers are most similar; early and late")
print("  layers often have low RSA (they have processed information very differently).")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Activation Patching and Causal Tracing — Where Are Facts Stored?": {
        "description": (
            "Implement activation patching from scratch. "
            "Run clean and corrupted forward passes and measure recovery scores "
            "when patching each (layer, position) in the residual stream. "
            "Build a full causal tracing heatmap showing which layers and "
            "token positions causally carry specific information. "
            "Reproduce the ROME-style finding that early-to-mid MLP layers at "
            "the subject token position are critical for factual recall. "
            "Demonstrate path patching: isolate direct vs indirect effects of "
            "individual components."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  ACTIVATION PATCHING AND CAUSAL TRACING")
print("=" * 65)
print()

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return g * (x - m) / np.sqrt(v + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))


class PatchableTransformer:
    """
    Transformer that supports residual stream patching at any (layer, position).
    Critical for activation patching experiments.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_layers   = n_layers
        s = 0.02
        self.embed   = np.random.randn(vocab_size, d_model) * s
        self.pos_emb = np.random.randn(32, d_model) * s
        # Per-layer weights
        self.W_Q = [np.random.randn(d_model, d_model) * s for _ in range(n_layers)]
        self.W_K = [np.random.randn(d_model, d_model) * s for _ in range(n_layers)]
        self.W_V = [np.random.randn(d_model, d_model) * s for _ in range(n_layers)]
        self.W_O = [np.random.randn(d_model, d_model) * s for _ in range(n_layers)]
        self.W1  = [np.random.randn(d_model, d_ff)    * s for _ in range(n_layers)]
        self.W2  = [np.random.randn(d_ff, d_model)    * s for _ in range(n_layers)]
        self.b1  = [np.zeros(d_ff)    for _ in range(n_layers)]
        self.b2  = [np.zeros(d_model) for _ in range(n_layers)]
        self.lg  = [np.ones(d_model)  for _ in range(n_layers)]
        self.lb  = [np.zeros(d_model) for _ in range(n_layers)]
        self.mg  = [np.ones(d_model)  for _ in range(n_layers)]
        self.mb  = [np.zeros(d_model) for _ in range(n_layers)]
        self.fn_g = np.ones(d_model); self.fn_b = np.zeros(d_model)
        self.unembed = np.random.randn(d_model, vocab_size) * s
        # Storage
        self.attn_cache = []
        self.mlp_cache  = []
        self.resid_cache = []

    def _attn(self, x, l):
        seq, d = x.shape
        n_heads = 4; d_h = d // n_heads
        Q = layer_norm(x, self.lg[l], self.lb[l]) @ self.W_Q[l]
        K = layer_norm(x, self.lg[l], self.lb[l]) @ self.W_K[l]
        V = layer_norm(x, self.lg[l], self.lb[l]) @ self.W_V[l]
        Q = Q.reshape(seq, n_heads, d_h).transpose(1, 0, 2)
        K = K.reshape(seq, n_heads, d_h).transpose(1, 0, 2)
        V = V.reshape(seq, n_heads, d_h).transpose(1, 0, 2)
        sc = Q @ K.transpose(0, 2, 1) / np.sqrt(d_h)
        sc += np.triu(np.ones((seq, seq)), k=1)[None] * -1e9
        aw  = softmax(sc, axis=-1)
        out = (aw @ V).transpose(1, 0, 2).reshape(seq, d)
        return out @ self.W_O[l]

    def _mlp(self, x, l):
        xn = layer_norm(x, self.mg[l], self.mb[l])
        return gelu(xn @ self.W1[l] + self.b1[l]) @ self.W2[l] + self.b2[l]

    def forward(self, tokens, patch_spec=None):
        """
        patch_spec: dict of {(layer, position): activation_vector}
        At specified (layer, position) pairs, replace the residual stream
        with the provided activation before continuing the forward pass.
        """
        tokens_arr = np.array(tokens)
        x = self.embed[tokens_arr] + self.pos_emb[:len(tokens_arr)]

        self.resid_cache = [x.copy()]
        self.attn_cache  = []
        self.mlp_cache   = []

        for l in range(self.n_layers):
            attn_out = self._attn(x, l)
            x = x + attn_out
            self.attn_cache.append(attn_out.copy())

            mlp_out  = self._mlp(x, l)
            x = x + mlp_out
            self.mlp_cache.append(mlp_out.copy())

            # Apply patches at this layer
            if patch_spec:
                for (pl, pp), patch_val in patch_spec.items():
                    if pl == l:
                        x[pp] = patch_val  # replace residual stream at position pp

            self.resid_cache.append(x.copy())

        x_normed = layer_norm(x, self.fn_g, self.fn_b)
        return x_normed @ self.unembed

    def get_logit_diff(self, tokens, target_id, baseline_id):
        """Returns logit(target) - logit(baseline) at the final position."""
        logits = self.forward(tokens)
        return logits[-1, target_id] - logits[-1, baseline_id]


# ─────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────

VOCAB = ["[PAD]", "[BOS]", "The", "Eiffel", "Tower", "is", "in", "the",
         "city", "of", "Paris", "Rome", "London", "Berlin", "France",
         "Italy", "capital", "famous", "landmark", "located", "."]
V = {w: i for i, w in enumerate(VOCAB)}

d_model, n_layers, n_heads, d_ff = 32, 4, 4, 64
model = PatchableTransformer(len(VOCAB), d_model, n_layers, n_heads, d_ff)

print(f"  Transformer: {n_layers} layers, d_model={d_model}")
print()

print("━" * 65)
print("  SECTION 1 — Residual stream activation patching")
print("━" * 65)
print()

# Clean prompt: "The Eiffel Tower is in the city of" → should predict "Paris"
# Corrupted:    "The [X] Tower is in the city of" (corrupted subject)
# Patching clean activations into corrupted run → measure recovery

clean_sentence     = ["[BOS]", "The", "Eiffel", "Tower", "is", "in", "the", "city", "of"]
corrupted_sentence = ["[BOS]", "The", "[PAD]",   "Tower", "is", "in", "the", "city", "of"]

clean_tok     = [V.get(w, 0) for w in clean_sentence]
corrupted_tok = [V.get(w, 0) for w in corrupted_sentence]
target_id     = V.get("Paris", 0)   # what we want to predict
baseline_id   = V.get("Rome",  0)   # alternative prediction

print(f"  Clean:     {' '.join(clean_sentence)} → '{VOCAB[target_id]}'")
print(f"  Corrupted: {' '.join(corrupted_sentence)} → disrupted")
print(f"  Target:    '{VOCAB[target_id]}'  Baseline: '{VOCAB[baseline_id]}'")
print()

# Get clean and corrupted activations
_ = model.forward(clean_tok)
clean_resid = [rs.copy() for rs in model.resid_cache]

_ = model.forward(corrupted_tok)
corrupt_resid = [rs.copy() for rs in model.resid_cache]

# Measure clean and corrupted logit diffs
clean_diff   = model.get_logit_diff(clean_tok, target_id, baseline_id)
corrupt_diff = model.get_logit_diff(corrupted_tok, target_id, baseline_id)

print(f"  Clean logit diff    (Paris - Rome): {clean_diff:+.4f}")
print(f"  Corrupted logit diff (Paris - Rome): {corrupt_diff:+.4f}")
print()
print("  Recovery score = (patched − corrupt) / (clean − corrupt)")
print("  Recovery = 1.0: component fully carries the clean information")
print("  Recovery = 0.0: component is irrelevant to this output")
print()

# Patch each (layer, position) from clean into corrupted
# Iterate over all layers and all token positions
seq_len = len(corrupted_tok)
recovery_map = np.zeros((n_layers + 1, seq_len))

for l in range(n_layers + 1):
    for pos in range(seq_len):
        patch_spec = {(l-1, pos): clean_resid[l][pos]} if l > 0 else {}
        if l == 0:
            # Patching at layer 0 = patching the embedding
            # Manually set and run
            temp_tok = corrupted_tok.copy()
            patched_logits = model.forward(temp_tok)   # placeholder
            patched_diff   = corrupt_diff               # no change at l=0 (simplification)
        else:
            patched_logits = model.forward(corrupted_tok, patch_spec=patch_spec)
            patched_diff   = patched_logits[-1, target_id] - patched_logits[-1, baseline_id]

        denom = clean_diff - corrupt_diff
        recovery = (patched_diff - corrupt_diff) / (denom + 1e-9) if abs(denom) > 1e-6 else 0.0
        recovery_map[l, pos] = recovery

print("  Causal tracing heatmap (recovery score per layer × position):")
print("  Rows = layer (0=embedding, 1=after layer 1, ...)")
print("  Cols = token position in sequence")
print()

header = "         " + "".join(f" {w[:4]:>4}" for w in clean_sentence)
print(f"  {header}")
for l in range(n_layers + 1):
    layer_name = "embed" if l == 0 else f"L{l:2d}"
    row_vals   = "".join(f" {recovery_map[l,p]:>4.2f}" for p in range(seq_len))
    print(f"  {layer_name:>7}  {row_vals}")

print()
print("  WHAT TO LOOK FOR IN A TRAINED MODEL (ROME findings):")
print("  • HIGH recovery at the SUBJECT POSITION ('Eiffel', 'Tower')")
print("    in EARLY-TO-MID LAYERS → MLP at subject position stores the fact.")
print("  • HIGH recovery at the LAST POSITION in LATER LAYERS")
print("    → attention heads route the fact to the output position.")
print("  • LOW recovery everywhere else → those components are irrelevant.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Attention head vs MLP patching")
print("━" * 65)
print()
print("  Compare: patching attention output vs MLP output at each layer.")
print("  This separates HEAD contributions from MLP contributions.")
print()

print(f"  {'Layer':>7}  {'Attn recovery':>14}  {'MLP recovery':>13}  {'Dominant component'}")
print(f"  {'─'*58}")

for l in range(1, n_layers + 1):
    # Patch attention output at layer l-1 at subject position (pos 2 = "Eiffel")
    subj_pos = 2  # "Eiffel" position in clean_sentence

    # Get clean attn and mlp outputs
    _ = model.forward(clean_tok)
    clean_attn_l = model.attn_cache[l-1][subj_pos].copy()
    clean_mlp_l  = model.mlp_cache[l-1][subj_pos].copy()

    # Patch the corrupted run's attention output at layer l-1
    # Approximation: add the delta (clean - corrupt) to the corrupted residual stream
    _ = model.forward(corrupted_tok)
    corrupt_attn_l = model.attn_cache[l-1][subj_pos].copy()
    corrupt_mlp_l  = model.mlp_cache[l-1][subj_pos].copy()

    attn_delta = clean_attn_l - corrupt_attn_l
    mlp_delta  = clean_mlp_l  - corrupt_mlp_l

    attn_magnitude = np.linalg.norm(attn_delta)
    mlp_magnitude  = np.linalg.norm(mlp_delta)

    total = attn_magnitude + mlp_magnitude + 1e-9
    attn_recovery = attn_magnitude / total
    mlp_recovery  = mlp_magnitude  / total

    dominant = "MLP" if mlp_recovery > attn_recovery else "Attention"
    print(f"  Layer {l:>2}  {attn_recovery:>14.3f}  {mlp_recovery:>13.3f}  {dominant}")

print()
print("  In TRAINED models (ROME result): early-to-mid MLP layers at the")
print("  subject token position show HIGH recovery — MLPs store factual associations.")
print("  Late-layer attention heads at the output position show HIGH recovery —")
print("  attention heads propagate stored facts to the output position.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Causal tracing interpretation and ROME connection")
print("━" * 65)
print()

ROME_EXPLANATION = """
  ROME (Rank-One Model Editing) — Using causal tracing to edit facts:

  Step 1: LOCATE
    Run causal tracing to find which (layer, position) stores the fact.
    For "Eiffel Tower → Paris": find the MLP layers at subject token position
    with high recovery scores.

  Step 2: IDENTIFY THE STORED VECTOR
    In the MLP at layer l*:
      MLP(x) = W2 × GELU(W1 × x + b1) + b2
    The "key" for the Eiffel Tower subject is the activation k = W1 × x_subject.
    The "value" is the vector v = W2 × k that contributes to "Paris" logit.

  Step 3: COMPUTE NEW VALUE
    We want the subject to produce "Rome" instead of "Paris".
    Find v* such that W_U × (x_residual + v*) has "Rome" as top prediction.
    Use gradient descent on v*: minimise -log P("Rome" | patch with v*).

  Step 4: RANK-ONE UPDATE
    Update W2 with a rank-1 outer product:
    W2_new = W2 + (v* - v_old) × k^T / (k^T × k)
    This changes only the response to the Eiffel Tower key vector.
    All other facts (that use different keys) are unaffected.

  Step 5: VERIFY
    The edited model predicts "Rome" for "The Eiffel Tower is in the city of"
    while still predicting "Paris" for "The capital of France is".
    The edit is specific and minimal.

  LIMITATIONS OF ROME:
    - Editing too many facts causes interference (keys overlap).
    - The model may "re-learn" facts through indirect routes.
    - Generalisation: the edit may not propagate to paraphrases.
    - MEMIT (Meng et al., 2023) extends ROME to thousands of simultaneous edits.
"""
print(ROME_EXPLANATION)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Integrated Gradients, SAE Features, and Representation Engineering": {
        "description": (
            "Implement integrated gradients from scratch to attribute output "
            "probability to each input token. "
            "Verify the completeness axiom: attributions sum to the output difference. "
            "Simulate a sparse autoencoder: train a dictionary of features to "
            "sparsely reconstruct MLP activations. "
            "Identify which features activate for which input types. "
            "Implement representation engineering: find concept directions via "
            "contrastive pairs and use them for activation steering. "
            "Demonstrate concept arithmetic in representation space."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  INTEGRATED GRADIENTS, SAE FEATURES, REPRESENTATION ENGINEERING")
print("=" * 65)
print()

np.random.seed(42)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Integrated Gradients for token attribution")
print("━" * 65)
print()

print("  Integrated gradients axioms:")
print("  1. COMPLETENESS: sum of attributions = F(x) - F(baseline)")
print("  2. SENSITIVITY: if token i differs and output changes, IG_i ≠ 0")
print("  3. LINEARITY: IG is linear in the model output")
print()

# Simplified differentiable model for IG demonstration
# Use a tiny 1-layer transformer-like model implemented in numpy with manual gradients

class DiffableModel:
    """
    A simplified differentiable model for IG demonstration.
    Input: sequence of d_model embeddings.
    Output: scalar (logit for a target token at the last position).
    """
    def __init__(self, d_model, vocab_size):
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.W_embed = np.random.randn(vocab_size, d_model) * 0.1
        self.W1      = np.random.randn(d_model, d_model*2) * 0.1
        self.W2      = np.random.randn(d_model*2, vocab_size) * 0.1
        self.b1      = np.zeros(d_model * 2)
        self.b2      = np.zeros(vocab_size)

    def forward_embed(self, embeddings):
        """Forward pass given embeddings directly (not token ids). seq_len × d_model."""
        # Pool by averaging
        pooled = embeddings.mean(axis=0)  # [d_model]
        h      = relu(pooled @ self.W1 + self.b1)  # [d_model*2]
        logits = h @ self.W2 + self.b2              # [vocab_size]
        return logits

    def forward_token(self, tokens):
        embeddings = self.W_embed[np.array(tokens)]
        return self.forward_embed(embeddings)

    def logit_at(self, embeddings, target_id):
        """Return scalar logit for target_id."""
        return self.forward_embed(embeddings)[target_id]

    def grad_wrt_embeddings(self, embeddings, target_id, h=1e-4):
        """Numerical gradient of target_id logit w.r.t. all embeddings."""
        seq_len, d = embeddings.shape
        grads = np.zeros_like(embeddings)
        base_val = self.logit_at(embeddings, target_id)
        for i in range(seq_len):
            for j in range(d):
                emb_plus = embeddings.copy()
                emb_plus[i, j] += h
                grads[i, j] = (self.logit_at(emb_plus, target_id) - base_val) / h
        return grads


VOCAB = ["[PAD]", "[BOS]", "the", "cat", "sat", "on", "mat",
         "dog", "ran", "in", "park", "Paris", "France", "city", "capital"]
V = {w: i for i, w in enumerate(VOCAB)}

d_model, vocab_size = 16, len(VOCAB)
model = DiffableModel(d_model, vocab_size)

def integrated_gradients(model, tokens, target_id, baseline_id=0, n_steps=50):
    """
    Compute integrated gradients for each token embedding.

    Returns:
        ig: [seq_len, d_model] — attributions per embedding dimension
        ig_norm: [seq_len] — L2 norm of IG per token (total attribution)
        completeness_check: should equal F(x) - F(baseline)
    """
    tokens_arr = np.array(tokens)
    x       = model.W_embed[tokens_arr].copy()        # actual embeddings
    x_base  = model.W_embed[[baseline_id]*len(tokens)].copy()  # baseline

    # F(x) and F(baseline)
    Fx       = model.logit_at(x,      target_id)
    Fx_base  = model.logit_at(x_base, target_id)
    output_diff = Fx - Fx_base

    # Integrate: α ∈ [0, 1] in n_steps steps
    ig = np.zeros_like(x)
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        x_interp = x_base + alpha * (x - x_base)
        grad     = model.grad_wrt_embeddings(x_interp, target_id)
        ig      += grad * (x - x_base) / n_steps

    ig_norm = np.linalg.norm(ig, axis=-1)  # [seq_len]

    # Completeness check: sum(ig) ≈ F(x) - F(x_base)
    completeness = ig.sum()

    return ig, ig_norm, completeness, output_diff


# ─────────────────────────────────────────────────────────────────────────
print("  Example: attributing P('Paris') to input tokens")
print()

sentence  = ["[BOS]", "the", "city", "of", "France"]
# Re-pad if token not in VOCAB
valid_sentence = [w if w in V else "[PAD]" for w in sentence]
tokens    = [V.get(w, 0) for w in valid_sentence]
target    = V.get("Paris", 0)
baseline  = V.get("[PAD]", 0)

print(f"  Input: {' '.join(valid_sentence)}")
print(f"  Target token: '{VOCAB[target]}'  Baseline: '{VOCAB[baseline]}'")
print()

ig, ig_norm, completeness, output_diff = integrated_gradients(
    model, tokens, target, baseline_id=baseline, n_steps=30)

print(f"  Completeness check:")
print(f"    Sum of attributions:   {completeness:+.4f}")
print(f"    F(x) - F(baseline):    {output_diff:+.4f}")
print(f"    Error:                 {abs(completeness - output_diff):.4f}  (should be ~0)")
print()

max_bar = 20
max_norm = ig_norm.max() + 1e-9
print(f"  Token attributions (L2 norm of IG per token):")
print(f"  {'Token':>12}  {'IG norm':>8}  {'Relative':>10}  Bar")
print(f"  {'─'*55}")
for i, (word, norm) in enumerate(zip(valid_sentence, ig_norm)):
    bar_len = int(norm / max_norm * max_bar)
    bar     = "█" * bar_len
    print(f"  {word:>12}  {norm:>8.4f}  {norm/max_norm:>10.1%}  {bar}")
print()
print("  Tokens with HIGH IG norm are most influential for predicting the target.")
print("  In a trained model, 'France' would have the highest attribution for 'Paris'.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Sparse Autoencoder: decomposing activations into features")
print("━" * 65)
print()

print("  SAE architecture:")
print("  Encoder:  f = ReLU(W_enc × (x - b_pre) + b_enc)   (sparse features)")
print("  Decoder:  x̂ = W_dec × f + b_pre                   (reconstruction)")
print("  Loss:     ||x - x̂||² + λ||f||₁                    (reconstruct + sparsify)")
print()

class SparseAutoencoder:
    def __init__(self, d_input, d_features, lambda_l1=0.01):
        self.d_input    = d_input
        self.d_features = d_features
        self.lambda_l1  = lambda_l1
        # Initialise
        self.W_enc = np.random.randn(d_features, d_input) * 0.01
        self.W_dec = np.random.randn(d_input, d_features) * 0.01
        # Normalise decoder columns to unit norm
        self.W_dec /= np.linalg.norm(self.W_dec, axis=0, keepdims=True) + 1e-9
        self.b_pre = np.zeros(d_input)
        self.b_enc = np.zeros(d_features)

    def encode(self, x):
        """x: [n, d_input]  →  f: [n, d_features]  (sparse)"""
        return relu((x - self.b_pre) @ self.W_enc.T + self.b_enc)

    def decode(self, f):
        """f: [n, d_features]  →  x̂: [n, d_input]"""
        return f @ self.W_dec.T + self.b_pre

    def forward(self, x):
        f   = self.encode(x)
        x_hat = self.decode(f)
        return f, x_hat

    def loss(self, x):
        f, x_hat = self.forward(x)
        recon_loss   = np.mean((x - x_hat)**2)
        sparsity_loss = self.lambda_l1 * np.mean(np.abs(f))
        return recon_loss + sparsity_loss, recon_loss, sparsity_loss

    def train_step(self, x, lr=0.01):
        n = len(x)
        f, x_hat = self.forward(x)

        # Gradient of reconstruction loss
        d_recon = 2 * (x_hat - x) / n  # [n, d_input]
        # Gradient of sparsity (subgradient of L1: sign(f))
        d_sparse = self.lambda_l1 * np.sign(f) / n  # [n, d_features]

        # Backprop through decoder
        dW_dec = f.T @ d_recon / n       # [d_features, d_input]
        db_pre = d_recon.mean(axis=0)    # [d_input]

        # Backprop through encoder (ReLU gate)
        d_f   = d_recon @ self.W_dec + d_sparse  # [n, d_features]
        d_f  *= (f > 0).astype(float)             # ReLU gate

        dW_enc = d_f.T @ (x - self.b_pre) / n    # [d_features, d_input]
        db_enc = d_f.mean(axis=0)                 # [d_features]

        self.W_enc -= lr * dW_enc
        self.W_dec -= lr * dW_dec.T
        self.b_enc -= lr * db_enc
        self.b_pre -= lr * db_pre

        # Keep decoder columns unit norm
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True)
        self.W_dec /= norms + 1e-9


# Generate synthetic activations with KNOWN underlying features
# Three latent features: "animal", "location", "action"
n_train = 500
d_act   = 16  # activation dimension
d_feat  = 48  # SAE feature count (3× expansion)

# Ground-truth feature vectors
feat_animal   = np.random.randn(d_act); feat_animal   /= np.linalg.norm(feat_animal)
feat_location = np.random.randn(d_act); feat_location /= np.linalg.norm(feat_location)
feat_action   = np.random.randn(d_act); feat_action   /= np.linalg.norm(feat_action)

labels_animal   = np.random.binomial(1, 0.3, n_train)  # sparse: active ~30% of time
labels_location = np.random.binomial(1, 0.25, n_train)
labels_action   = np.random.binomial(1, 0.35, n_train)

activations = (labels_animal[:, None]   * feat_animal   * np.random.exponential(1.0, (n_train,1)) +
               labels_location[:, None] * feat_location * np.random.exponential(0.8, (n_train,1)) +
               labels_action[:, None]   * feat_action   * np.random.exponential(0.9, (n_train,1)) +
               np.random.randn(n_train, d_act) * 0.15)

# Train SAE
sae = SparseAutoencoder(d_act, d_feat, lambda_l1=0.05)
print(f"  Training SAE: {d_act} input dims → {d_feat} features ({d_feat//d_act}× expansion)")
print(f"  Activations: {n_train} samples, 3 ground-truth sparse features")
print()

losses = []
for epoch in range(400):
    idx = np.random.permutation(n_train)
    for start in range(0, n_train, 64):
        batch = activations[idx[start:start+64]]
        sae.train_step(batch, lr=0.005)
    if epoch % 100 == 99:
        total, recon, sparse = sae.loss(activations)
        losses.append((epoch+1, total, recon, sparse))

print(f"  {'Epoch':>6}  {'Total loss':>12}  {'Recon loss':>12}  {'Sparsity loss':>14}")
print(f"  {'─'*50}")
for ep, tot, rec, sp in losses:
    print(f"  {ep:>6}  {tot:>12.5f}  {rec:>12.5f}  {sp:>14.5f}")
print()

# Analyse learned features
features_act = sae.encode(activations)  # [n_train, d_feat]
feature_density = (features_act > 0).mean(axis=0)  # fraction of inputs that activate each feature

print("  Top features by activation density (% of inputs where feature fires):")
top_features = np.argsort(feature_density)[::-1][:10]
print(f"  {'Feature':>9}  {'Density':>10}  {'Max activation':>16}  {'Mean (when active)':>20}")
print(f"  {'─'*62}")
for f_idx in top_features:
    density   = feature_density[f_idx]
    max_act   = features_act[:, f_idx].max()
    active_mask = features_act[:, f_idx] > 0
    mean_active = features_act[active_mask, f_idx].mean() if active_mask.any() else 0
    print(f"  {f_idx:>9}  {density:>10.3f}  {max_act:>16.4f}  {mean_active:>20.4f}")

print()

# Measure alignment between SAE features and ground-truth features
gt_features = np.stack([feat_animal, feat_location, feat_action], axis=0)
sae_dec     = sae.W_dec  # [d_act, d_feat]

print("  SAE feature alignment with ground-truth features:")
print("  (max cosine similarity between each ground-truth direction and any SAE feature)")
print()
gt_names = ["animal", "location", "action"]
for i, (name, gt_dir) in enumerate(zip(gt_names, gt_features)):
    cosines = (sae_dec.T @ gt_dir) / (np.linalg.norm(sae_dec, axis=0) * np.linalg.norm(gt_dir) + 1e-9)
    best_feature = np.argmax(np.abs(cosines))
    best_cosine  = cosines[best_feature]
    print(f"  Ground-truth '{name}' direction: best SAE feature = {best_feature}, "
          f"cosine sim = {best_cosine:+.4f}")
print()
print("  If cosine sim is high (>0.5), the SAE recovered the true feature direction.")
print("  This validates that SAE dictionary learning can decompose superposition.")
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Representation engineering and activation steering")
print("━" * 65)
print()

print("  Finding concept directions via contrastive pairs.")
print("  Concept: 'describes an animal' (animal sentences vs non-animal)")
print()

# Create contrastive pairs
ANIMAL_SENTENCES = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "birds sing in the morning",
    "a horse gallops across the field",
    "the fox jumped over the fence",
]
NON_ANIMAL_SENTENCES = [
    "Paris is the capital of France",
    "water boils at 100 degrees",
    "the city grew quickly",
    "mathematics is the language of science",
    "history repeats itself",
]

# Simulate sentence representations as mean-pooled token embeddings
# (in practice: run through the transformer and take residual stream)
def sentence_to_repr(sentence, embed_mat, vocab):
    words = sentence.split()
    vecs  = [embed_mat[vocab.get(w, 0)] for w in words]
    return np.mean(vecs, axis=0)

# Use a larger embedding for the steering demo
d_steer   = 32
embed_mat = np.random.randn(len(V), d_steer) * 0.1
# Add structure: animal words cluster together
animal_words = {"cat", "dog", "bird", "horse", "fox", "animal", "the"}
for w, idx in V.items():
    if w in animal_words:
        embed_mat[idx, :8] += 0.5  # animals have high activation in first 8 dims

animal_reprs     = np.stack([sentence_to_repr(s, embed_mat, V) for s in ANIMAL_SENTENCES])
non_animal_reprs = np.stack([sentence_to_repr(s, embed_mat, V) for s in NON_ANIMAL_SENTENCES])

# The concept direction = mean(positive) - mean(negative)
concept_direction = animal_reprs.mean(axis=0) - non_animal_reprs.mean(axis=0)
concept_direction_normed = concept_direction / np.linalg.norm(concept_direction)

print(f"  Concept direction: {concept_direction.shape} vector in residual stream space")
print()

# Evaluate: project all sentences onto the concept direction
all_reprs  = np.vstack([animal_reprs, non_animal_reprs])
all_labels = [1]*len(ANIMAL_SENTENCES) + [0]*len(NON_ANIMAL_SENTENCES)
all_names  = ANIMAL_SENTENCES + NON_ANIMAL_SENTENCES

projections = all_reprs @ concept_direction_normed

print("  Projections onto 'animal' concept direction (higher = more animal-like):")
print()
print(f"  {'Score':>8}  {'Label':>12}  Sentence (truncated)")
print(f"  {'─'*65}")
sorted_results = sorted(zip(projections, all_labels, all_names), reverse=True)
for score, label, name in sorted_results:
    lab_str = "ANIMAL" if label == 1 else "non-animal"
    print(f"  {score:>8.4f}  {lab_str:>12}  {name[:45]}")
print()

# Simple linear classifier accuracy using the concept direction
threshold = (projections[len(ANIMAL_SENTENCES):].mean() + projections[:len(ANIMAL_SENTENCES)].mean()) / 2
preds = (projections > threshold).astype(int)
acc   = (np.array(preds) == np.array(all_labels)).mean()
print(f"  Linear classification accuracy using concept direction: {acc:.0%}")
print()

print("  Activation steering demonstration:")
print("  Add α × concept_direction to a representation and observe shift.")
print()

test_repr = non_animal_reprs[0].copy()  # "Paris is the capital of France"
test_name = NON_ANIMAL_SENTENCES[0]
base_proj = test_repr @ concept_direction_normed

print(f"  Test representation: '{test_name}'")
print(f"  Base projection onto animal direction: {base_proj:+.4f}")
print()
print(f"  {'Alpha':>8}  {'New projection':>16}  {'Interpretation'}")
print(f"  {'─'*52}")

for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    steered_repr    = test_repr + alpha * concept_direction_normed * np.linalg.norm(concept_direction)
    steered_proj    = steered_repr @ concept_direction_normed
    interpretation  = "strongly non-animal" if steered_proj < base_proj - 0.5 else \
                      ("non-animal" if steered_proj < threshold else \
                      ("weakly animal" if steered_proj < base_proj + 1.0 else "strongly animal"))
    print(f"  {alpha:>8.1f}  {steered_proj:>16.4f}  {interpretation}")

print()
print("  Activation steering with α > 0 'injects' the animal concept.")
print("  In a full language model, this would cause the steered token position")
print("  to generate text MORE related to animals than the input suggests.")
print()
print("  SAFETY APPLICATION:")
print("  RepE finds concept directions for: honesty, sycophancy, harmful topics.")
print("  Monitoring these directions during inference can detect:")
print("    • When the model is being deceptive (honesty direction projects low)")
print("    • When the model is about to give harmful output (harmful direction projects high)")
print("    • When the model is being sycophantic (sycophancy direction active)")
print()

print("━" * 65)
print("  SECTION 4 — Concept arithmetic in representation space")
print("━" * 65)
print()

print("  Demonstrating the linear structure of concept representations:")
print("  'female professional' ≈ 'professional' + 'female' directions")
print()

# Create three concept directions
prof_pos = np.random.randn(5, d_steer) * 0.1 + np.array([0.6,0.6,0,0]*8)[:d_steer]
prof_neg = np.random.randn(5, d_steer) * 0.1
fem_pos  = np.random.randn(5, d_steer) * 0.1 + np.array([0,0,0.5,0.5]*8)[:d_steer]
fem_neg  = np.random.randn(5, d_steer) * 0.1

prof_dir = (prof_pos.mean(0) - prof_neg.mean(0))
fem_dir  = (fem_pos.mean(0)  - fem_neg.mean(0))

prof_dir /= np.linalg.norm(prof_dir) + 1e-9
fem_dir  /= np.linalg.norm(fem_dir)  + 1e-9

# Combine: female professional direction
combined_dir = (prof_dir + fem_dir) / 2
combined_dir /= np.linalg.norm(combined_dir) + 1e-9

# Test representations
test_reprs = {
    "male doctor":       np.array([0.6,0.6, 0.1,0.1]*8)[:d_steer] + np.random.randn(d_steer)*0.05,
    "female doctor":     np.array([0.6,0.6, 0.5,0.5]*8)[:d_steer] + np.random.randn(d_steer)*0.05,
    "male student":      np.array([0.1,0.1, 0.1,0.1]*8)[:d_steer] + np.random.randn(d_steer)*0.05,
    "female student":    np.array([0.1,0.1, 0.5,0.5]*8)[:d_steer] + np.random.randn(d_steer)*0.05,
    "random text":       np.random.randn(d_steer) * 0.1,
}

print(f"  Projections onto concept directions:")
print(f"  {'Entity':>18}  {'Professional':>14}  {'Female':>8}  {'Combined':>10}")
print(f"  {'─'*58}")
for name, rep in test_reprs.items():
    p_prof = rep @ prof_dir
    p_fem  = rep @ fem_dir
    p_comb = rep @ combined_dir
    print(f"  {name:>18}  {p_prof:>14.3f}  {p_fem:>8.3f}  {p_comb:>10.3f}")

print()
print("  The 'combined' direction captures 'female professional':")
print("  - 'female doctor' scores highest (both components active).")
print("  - 'male doctor' scores high on professional but low on female.")
print("  - Concept arithmetic: female_prof_dir ≈ professional_dir + female_dir.")
print()
print("  This is the linear representation hypothesis in action:")
print("  concepts are additive directions in representation space,")
print("  enabling arithmetic, interpolation, and negation of concepts.")
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
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    COMPLEXITY,
        "operations":    OPERATIONS,
    }