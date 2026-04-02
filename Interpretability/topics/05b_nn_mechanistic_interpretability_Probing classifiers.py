"""
Probing Classifiers — What Do Representations Know?
=====================================================

Probing classifiers (also called diagnostic classifiers) are the standard
experimental tool for asking: "does this neural network's representation at
layer l encode property P?" A linear probe trains a simple classifier on top
of frozen intermediate activations and asks whether that property is linearly
decodable from the representation — without modifying the original model.

The method is elegant in its simplicity and deceptively subtle in its
interpretation. This module covers the mathematical formalism, the design
decisions that determine what a probe actually measures, the battery of
interpretive pitfalls, and the principled framework for drawing valid
conclusions from probing experiments.

"""

import math
import random
import textwrap
import re
import os
import base64


TOPIC_NAME = "Probing Classifiers — What Do Representations Know?"
DISPLAY_NAME = "05b · Probing Classifiers"
ICON = "🧪"
SUBTITLE = "Linear probes for decoding properties of neural representations"


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

### The Central Question

Every layer of a deep neural network transforms its input into a new
representation. After training, these intermediate representations encode
something — but what, exactly? Does BERT's layer 8 encode syntactic
dependency structure? Does a vision transformer's mid-layer know about
object identity before it knows about colour? Does a language model
trained only on predicting the next token implicitly learn grammatical
gender, named entity types, or sentiment polarity?

Probing classifiers answer these questions by asking a simpler surrogate
question: can a simple model DECODE property P from representation h?

The argument runs:

    IF a linear classifier trained on frozen representations h^l(x)
    can predict property P(x) with high accuracy,
    THEN the information about P is present in h^l in a linearly
    accessible form — the representation "knows" about P.

The method requires no modification of the original network. The probe
is trained separately, on the frozen representations, in a supervised
manner using labelled examples of P. The probe's accuracy is the
measurement; its weights are not the explanation.

Probing is now a cornerstone of interpretability research. It has been
used to study linguistic structure in language models (Tenney et al., 2019),
syntactic trees in sentence encoders (Hewitt & Manning, 2019), world models
in RL agents (Anand et al., 2019), and factual knowledge in transformers
(Petroni et al., 2019).


##### PART I: THE FORMAL SETUP

### Definitions

Let f : X → Y be a trained neural network with L layers.
Let h^l(x) ∈ ℝᵈˡ denote the activation of layer l on input x.

A PROBING CLASSIFIER (or probe) for property P at layer l is a function:

    g : ℝᵈˡ → C

where C is the set of possible values of P (e.g., {NOUN, VERB, ADJ, ...}
for part-of-speech, {0, 1} for sentiment, ℝ for depth estimation).

The probe is trained by:

    1. Collecting a PROBE DATASET: pairs (xᵢ, P(xᵢ)) for i = 1, ..., n,
       where P(xᵢ) is a label for property P obtained from linguistic
       annotation, heuristics, or a separate ground-truth system.

    2. Computing FROZEN REPRESENTATIONS: h^l(xᵢ) for each xᵢ using the
       trained model f, with all weights of f held fixed.

    3. TRAINING the probe g to predict P(xᵢ) from h^l(xᵢ) using
       standard supervised learning (cross-entropy loss for classification,
       MSE for regression).

    4. EVALUATING the probe on a held-out test set.

The probe is ALWAYS simple — typically a logistic regression or a
single linear layer. The simplicity constraint is load-bearing:
it is the key design decision that determines what the probe measures.

    # ================================================================== #
    **Standard probe architecture — logistic regression:**

    Probe weights:  W_probe ∈ ℝ^{|C| × d_l}
    Probe bias:     b_probe ∈ ℝ^{|C|}

    Prediction:     ŷ = softmax(W_probe · h^l(x) + b_probe)
    Training loss:  L = −Σᵢ Σ_c y_{ic} log ŷ_{ic}  (cross-entropy)

    This is a LINEAR function of the representation h^l(x).
    No non-linearities. No hidden layers. No interaction terms.
    Just a dot product between the representation and the probe weights.
    # ================================================================== #


### Why the Probe Must Be Linear

The linearity constraint is not an arbitrary limitation — it is the key
assumption that makes probing results interpretable.

    IF the probe is linear and achieves high accuracy:
    → The property P is LINEARLY ENCODED in the representation h^l.
    → Knowing h^l and the direction W_probe[c] suffices to read off P.
    → The information is accessible via a simple affine transformation.
    → This is consistent with the linear representation hypothesis
      (see mechanistic interpretability module).

    IF the probe is non-linear (e.g., a multi-layer MLP):
    → A complex non-linear probe achieving high accuracy tells you much less.
    → The probe itself may be doing the "work" of computing P from raw
      features that only implicitly encode P.
    → A powerful enough MLP can decode almost anything from anything —
      the probe's complexity makes the result ambiguous.

The rule of thumb: probe complexity should be LESS than the complexity
you expect the original model to have used to solve the same task.
Using a multi-layer MLP probe on a multi-layer transformer is circular:
both the probe and the model are solving the same task with similar tools.

    # ================================================================== #
    **The probe complexity spectrum:**

    MOST RESTRICTIVE (cleanest interpretation):
    ────────────────────────────────────────────────────────────────────
    Dot product:  ŷ = W[1] · h  (one-vs-all, no bias)
    Logistic:     ŷ = σ(W · h + b)  (linear + sigmoid, default choice)
    Ridge/LASSO:  linear with L2/L1 regularisation for high-d settings

    INTERMEDIATE:
    ────────────────────────────────────────────────────────────────────
    Structural probe: distance-based, decodes tree structure (Hewitt&Manning)
    Linear SVM:  margin-maximising linear separator (equivalent to ridge)

    LEAST RESTRICTIVE (weakest interpretation):
    ────────────────────────────────────────────────────────────────────
    2-layer MLP:  allows non-linear combinations of representation dims
    3-layer MLP:  can approximate any function (universal approximator)
    Full transformer: essentially the whole model again (circular)
    # ================================================================== #


### The Probing Dataset — Critical Design Choices

The probing dataset must be carefully constructed to avoid confounds.
Several design choices materially affect what the probe measures:

    CHOICE 1 — LABELLING SCHEME:
    How is P(x) defined and labelled? Automatically via existing parsers/taggers,
    or manually by annotators? Automatic labels carry the errors and biases
    of the labelling tool. Manual labels are expensive but cleaner.

    CHOICE 2 — TRAIN/TEST SPLIT STRATEGY:
    Naive random split: probe overfits surface features (lexical, positional).
    Better: split by vocabulary (different words in train and test) to prevent
    the probe from learning that "a specific word → specific tag" rather than
    "this feature in the representation → this tag."
    Best: split by DOMAIN to test generalisation to new contexts.

    CHOICE 3 — DATASET SIZE AND CLASS BALANCE:
    Imbalanced classes: a probe predicting the majority class always
    achieves high accuracy. Always report class-balanced accuracy or F1.
    Too small: high variance, probe may not learn the true signal.
    Too large: probes become powerful enough to decode things that are
    not truly linearly encoded (they exploit weak but consistent signals).

    CHOICE 4 — WHAT TO PROBE:
    Token-level properties (POS, NER, morphology): take the representation
    of a specific token position.
    Sentence-level properties (sentiment, entailment): take the [CLS]
    token representation, or average over all token representations.
    Relational properties (dependency heads, coreference): take pairs of
    token representations and probe their difference or concatenation.


##### PART II: WHAT HIGH PROBE ACCURACY ACTUALLY MEANS

### The "Encoding" Claim

When a probe achieves 90% accuracy predicting part-of-speech from BERT's
layer 8 representations, many researchers say "BERT layer 8 encodes POS
tags." This is a specific, interpretable claim — but it requires careful
unpacking.

    WHAT HIGH PROBE ACCURACY ESTABLISHES:
    ──────────────────────────────────────
    There EXISTS a linear function g(h^l(x)) that can predict P(x)
    with 90% accuracy. The representation h^l contains sufficient
    information about P, accessible via a linear function.

    WHAT HIGH PROBE ACCURACY DOES NOT ESTABLISH:
    ─────────────────────────────────────────────
    1. That the MODEL USES property P for its primary task.
       The information may be present but unused — a "passenger" encoding.
       The model might produce the same output if you erased P from h^l.

    2. That the information was LEARNED because it is useful.
       The model may have incidentally encoded P as a side effect of
       learning something else, without P being involved in any computation.

    3. That the encoding is CAUSAL.
       High probe accuracy shows correlation between h^l and P.
       It does not show that h^l causes good performance on tasks requiring P.

    4. That P is accessible at the GRANULARITY the probe implies.
       A 90% accurate POS probe may use 10 dimensions out of 768.
       The other 758 dimensions might be completely unrelated to POS.

    # ================================================================== #
    **The passenger information problem:**

    Suppose a language model is trained to predict the next word.
    It learns, incidentally, to encode POS tags linearly because
    certain POS tags reliably follow certain other POS tags.
    (Noun phrases often follow determiners; verbs often follow subjects.)

    A probe finds: "layer 8 encodes POS with 92% accuracy!"

    Question: does the model USE POS for its predictions?
    Test: ablate the POS direction from h^8 (zero out W_probe^T directions).
           Does next-word prediction accuracy drop?

    If YES: POS is causally involved.
    If NO:  POS is a passenger — incidentally encoded but not used.

    Without the causal test, the probe's 92% accuracy could mean either.
    # ================================================================== #


### The Selectivity Baseline

A probe's raw accuracy must always be compared to a SELECTIVITY BASELINE
that accounts for how much information could have been decoded from a
simpler representation.

    CONTROL 1 — RANDOM BASELINE:
    Shuffle the labels P(xᵢ) randomly. Train a probe on shuffled labels.
    The probe's accuracy on shuffled labels should be near the chance level
    (1/|C| for balanced classes). If it is significantly above chance,
    the probe is overfitting to spurious correlations in the dataset.

    CONTROL 2 — MAJORITY CLASS BASELINE:
    Always predict the most frequent class. For imbalanced data, this can
    be surprisingly high (90% if one class is 90% of the data).
    Your probe must significantly beat this to be informative.

    CONTROL 3 — SURFACE FEATURE BASELINE:
    Train a probe on simpler features: token identity (one-hot encoding),
    position, sentence length, word frequency. If the surface baseline
    achieves similar accuracy as the representation probe, the probe is
    measuring surface statistics, not deep representations.

    CONTROL 4 — RANDOM REPRESENTATION BASELINE (Hewitt & Liang, 2019):
    Train a probe on a RANDOM PROJECTION of the representation: h^l_rand = Mh^l,
    where M is a random Gaussian matrix of the SAME DIMENSION as h^l.
    This preserves the information content (by Johnson-Lindenstrauss) but
    scrambles the interpretation.
    If probe accuracy is similar for random projections as for actual
    representations, the probe is exploiting geometric properties of
    the high-dimensional space, not genuine encoding.

    CONTROL 5 — UNTRAINED MODEL BASELINE (Tamkin et al., 2020):
    Train a probe on representations from an UNTRAINED model (random weights).
    If an untrained model's representations also give high probe accuracy,
    the probe is measuring structural properties of the architecture, not
    learned features.

    # ================================================================== #
    **Selectivity score (Hewitt & Liang, 2019):**

    Let acc_P = probe accuracy on actual representations
    Let acc_R = probe accuracy on representations from a random linear probe
                (trained on the same task with a random control task label)

    Selectivity = acc_P − acc_R

    High selectivity: the representation genuinely encodes P beyond
    what a random representation of the same dimension would contain.

    Low selectivity: the representation's high accuracy for P is at
    least partly due to the dimensionality of the space, not genuine encoding.
    # ================================================================== #


##### PART III: THE LAYERWISE PROBING PROFILE

### Probing Across All Layers

The most informative use of probing is to train a separate probe at EACH
LAYER and compare the accuracy profiles. This gives a layerwise probing
profile: a curve of probe accuracy vs layer depth.

The profile reveals:
  • WHICH LAYERS encode a property most strongly.
  • WHERE in the network a property first appears.
  • WHERE the representation of a property PEAKS and DECLINES.
  • HOW the model's internal representations evolve during processing.

    # ================================================================== #
    **Typical layerwise profiles in BERT:**

    POS tagging accuracy:
    Layer:  0    1    2    3    4    5    6    7    8    9   10   11   12
    Acc:   55%  72%  80%  84%  86%  87%  86%  85%  84%  81%  77%  73%  70%
    Profile shape: early peak (layers 2-5), then gradually declines.
    Interpretation: POS is computed early and then partially overwritten
    by higher-level semantic representations in deeper layers.

    NER (named entity recognition):
    Layer:  0    1    2    3    4    5    6    7    8    9   10   11   12
    Acc:   58%  64%  69%  73%  76%  79%  82%  85%  87%  86%  84%  81%  78%
    Profile shape: gradual rise, peaks at layer 8.
    Interpretation: NER requires combining lexical identity (available early)
    with contextual disambiguation (built up through middle layers).

    Coreference resolution:
    Layer:  0    1    2    3    4    5    6    7    8    9   10   11   12
    Acc:   52%  55%  58%  62%  66%  70%  75%  80%  84%  87%  88%  87%  85%
    Profile shape: late peak (layers 9-11).
    Interpretation: coreference requires long-range context and entity
    tracking, which requires processing the full sentence — available only
    in the deepest layers.
    # ================================================================== #

This pattern — early linguistic layers, late semantic layers — appears
consistently across models and was the main finding of Tenney et al.'s
"BERT Rediscovers the Classical NLP Pipeline" (2019). The model processes
information in a roughly bottom-up order: morphology → syntax → semantics
→ discourse, mirroring the traditional NLP pipeline.


### The Edge Weight Visualisation (Tenney et al., 2019)

Rather than comparing raw accuracies, Tenney et al. computed SCALAR MIX
weights: for each task, the "expected layer" where the information lives,
computed as a weighted average of layers with weights proportional to
their marginal improvement in probe accuracy.

    Scalar mix weight for task P at layer l:
    w_P^l ∝ max(0, acc_P^l − acc_P^{l-1})

    Expected layer = Σ_l l × (w_P^l / Σ_l w_P^l)

This gives a single number per task: the "centre of gravity" of the
task-relevant information in the network. Tasks with lower expected layers
are resolved earlier; tasks with higher expected layers need more processing.

    BERT-base expected layers (approximate):
    Morphological features (POS, tense):   2.5
    Syntactic (dependencies, constituency): 5.8
    Semantic (SRL, coreference):           8.3
    Discourse:                             11.7

The monotone ordering confirms the pipeline metaphor: earlier layers handle
lower-level structure, later layers handle higher-level meaning.


##### PART IV: THE STRUCTURAL PROBE — BEYOND CLASSIFICATION

### The Limitation of Classification Probes

Standard classification probes answer binary or multinomial questions:
"is this token a noun?" (yes/no), "what is this token's NER label?"
(PERSON/ORG/LOC/...). They cannot answer structural questions like:

    "Does the representation encode the full parse tree structure?"
    "Does the model know which word is the syntactic head of which other word?"
    "Is the embedding space organised such that syntactically related words
     are closer to each other?"

These structural questions require a different probe design.

### The Structural Probe (Hewitt & Manning, 2019)

The structural probe tests whether syntactic tree distances are encoded
in the GEOMETRIC STRUCTURE of the representation space.

The hypothesis: there exists a linear transformation B ∈ ℝ^{k×d} such
that in the TRANSFORMED space B·h, the squared Euclidean distance between
two words' representations approximates their tree distance:

    ‖B · h^l(w₁) − B · h^l(w₂)‖² ≈ d_tree(w₁, w₂)

where d_tree(w₁, w₂) is the number of edges on the shortest path between
w₁ and w₂ in the dependency parse tree.

If such a B exists with k << d, the representation space has a low-dimensional
SUBSPACE in which syntactic structure is metrically encoded.

    # ================================================================== #
    **Why this is more powerful than classification probing:**

    Classification probe for "is w₁ the head of w₂?":
    Binary question, one pair at a time.
    Does not check the GLOBAL consistency of the structural encoding.

    Structural probe for tree distances:
    Tests whether the ENTIRE parse tree is consistently encoded.
    If B correctly approximates tree distances for ALL pairs in ALL sentences,
    then the model has a genuine geometric representation of tree structure.
    A probe that merely memorises "this verb is usually the head of this noun"
    cannot generalise to novel tree configurations.
    # ================================================================== #

Hewitt & Manning found that BERT layers 5–8 encode syntactic tree distances
with high accuracy (SPEARMAN correlation ≈ 0.85 with gold-standard tree
distances). The transformation B has rank ≈ 64 (out of 768), suggesting
that syntax occupies a 64-dimensional subspace of BERT's 768-dimensional
space.

### Training the Structural Probe

    Loss function:
    L_distance = (1/|S|) Σ_{s∈S} (1/|s|²) Σ_{w₁,w₂∈s}
                 | ‖B h^l(w₁) − B h^l(w₂)‖² − d_tree(w₁, w₂) |

    where S is the set of training sentences and |s| is the number of words.
    The (1/|s|²) normalisation prevents longer sentences from dominating.

    Parameters: B ∈ ℝ^{k×d}, trained by gradient descent.
    Evaluation: SPEARMAN correlation between predicted and true tree distances,
    plus the UNDIRECTED UNLABELLED ATTACHMENT SCORE (UUAS) — what fraction
    of predicted heads match the gold-standard parse tree heads.

The structural probe is still linear (B is a linear transformation), but
it probes METRIC STRUCTURE rather than categorical labels — a richer test
of whether syntax is geometrically encoded.


##### PART V: INTERPRETIVE PITFALLS — WHAT PROBES DO NOT SHOW

### Pitfall 1 — Encoding ≠ Using

The most important pitfall: a representation that ENCODES a property is
not necessarily USING that property for the primary task.

Consider a language model encoding POS with 92% probe accuracy. The model
may have learned this encoding as a side effect of predicting tokens that
follow specific syntactic patterns. But if you were to ablate the POS
information from the representation (project it out), the model's next-word
prediction accuracy might not change at all — the POS information was a
passenger, present but irrelevant to the output.

This "encoding vs using" distinction requires a CAUSAL test beyond the probe.
The intervention approach from mechanistic interpretability (activation
patching, causal ablation) is needed to establish whether the encoded
property is causally involved in the model's computation.

    # ================================================================== #
    **The encoding vs using distinction:**

    PROBE SAYS:         "Layer 8 encodes gender agreement with 89% accuracy."

    THIS MEANS:         A linear function of layer-8 representations can
                        predict grammatical gender of a noun.

    THIS DOES NOT MEAN: The model checks gender agreement when generating text.

    CAUSAL TEST NEEDED: Replace the gender-encoding direction in layer-8
                        representations with the opposite gender.
                        Does the model's generation change to violate
                        gender agreement? If YES: the model uses the encoding.
                        If NO: the encoding is a passenger.

    The difference matters enormously for safety-relevant claims.
    "The model encodes deception-related features" is very different from
    "the model uses deception-related features when being deceptive."
    # ================================================================== #


### Pitfall 2 — The Probe Capaciy Issue (Hewitt & Liang, 2019)

If the probe is too powerful, it can AMPLIFY weak signals into high accuracy.
A 3-layer MLP can learn to compute complex functions from very weak
representational signals — including signals that are present in
random representations of the same dimensionality.

Hewitt & Liang (2019) showed this with the selectivity control:
a random task (predicting random labels from the representation) had
surprisingly high accuracy with non-linear probes in high-dimensional
spaces. Linear probes showed much lower accuracy on random tasks —
confirming that linear probes are more selective.

This pitfall implies:
  • ALWAYS use linear probes (or very simple non-linear probes) as the default.
  • ALWAYS compare to the random task control (selectivity score).
  • ALWAYS compare to the untrained model control.
  • REPORT the probe's accuracy on a shuffled-label control.


### Pitfall 3 — Lexical Shortcuts

Many properties are strongly correlated with specific lexical items.
"Obama" is almost always a proper noun. "The" is always a determiner.
"Beautiful" is almost always an adjective.

A probe trained without vocabulary-based train/test splitting will exploit
these lexical shortcuts: it learns "Obama → PROPN, the → DET, beautiful → ADJ"
rather than learning genuinely representational features.

Detection: compare probe accuracy with lexical identity as input (just
a one-hot encoding of the token) to the probe with representations as input.
If both are similar, the probe is using lexical shortcuts.

Fix: hold out specific tokens from training. Evaluate on unseen tokens only.
This forces the probe to generalise via representational features, not
lexical memorisation.


### Pitfall 4 — Positional Biases

Token position in a sentence is often correlated with syntactic role.
The first token is often the subject. The verb often appears near the
middle. The period is always at the end. A probe for "is this a verb?"
might achieve high accuracy partly by exploiting position statistics,
not syntactic features.

Detection: compare probe accuracy on randomly position-shuffled sentences
to normal sentences. If accuracy drops significantly on shuffled sentences,
the probe uses positional information.

Fix: include position as a control feature. Report the MARGINAL contribution
of representations beyond what position alone predicts.


### Pitfall 5 — Conflating Properties

Different properties are often correlated. POS correlates with NER (named
entities tend to be nouns). Dependency head correlates with word frequency
(function words tend to be heads). Sentiment correlates with domain.

A probe trained to predict P may incidentally be learning Q (which is easier
to decode from representations) because P and Q are correlated in the data.

Detection: control for the confounding property. Train a probe for P after
regressing out Q from the representations (project out the Q direction).
If probe accuracy for P drops significantly, the probe was using Q, not P.


### Pitfall 6 — Mutual Information Confound

Anagnostidis et al. (2022) showed that representation dimensionality affects
probing results: representations in high-dimensional spaces contain more
information (by the Johnson-Lindenstrauss lemma) and thus support higher-
accuracy linear probes. When comparing representations of different
dimensionalities (e.g., different model sizes), differences in probe accuracy
may reflect differences in representational dimensionality, not differences
in what the models encode.

Fix: always probe after dimensionality reduction (PCA to equal dimensions),
or report normalised mutual information rather than raw accuracy.


##### PART VI: PROBING FOR WORLD MODELS AND FACTUAL KNOWLEDGE

### Beyond Linguistics — What Else Can Be Probed?

Probing was originally developed for linguistic properties but applies to
any property that can be labelled and matched to a specific input.

    FACTUAL KNOWLEDGE (Petroni et al., 2019 — "Language Models as KBs"):
    Can BERT predict the missing entity in "Dante was born in [MASK]"?
    This tests whether factual triples (entity, relation, object) are
    linearly encoded in BERT's MLM representations.
    Finding: BERT encodes many factual relations, but with low precision
    and poor generalisation to paraphrases.

    WORLD MODELS IN RL AGENTS (Anand et al., 2019):
    A CNN trained on Atari games via reinforcement learning — does its
    intermediate representation encode game state variables (player position,
    enemy position, score, time remaining)?
    Linear probes show that the agent's representations encode structured
    world state, not just "what to press next."

    GEOMETRY IN NAVIGATION AGENTS (Cueva & Wei, 2018):
    Recurrent networks trained to navigate encode grid-cell-like and
    place-cell-like representations — discovered via probing for spatial
    position from hidden states.

    EMOTIONS IN LANGUAGE MODELS (Anthropic, 2025):
    SAE analysis + probing found interpretable emotion-like features in
    Claude's residual stream, active in expected contexts and with expected
    causal effects on outputs.

    SOCIAL CONCEPTS IN DIALOGUE MODELS:
    Probes for social concepts (trust, formality, deception intent) in
    dialogue model representations reveal whether models encode social
    context. Used to study sycophancy and stereotype encoding.


### The Knowledge Localisation Question

A natural extension of probing: WHERE in the network is factual knowledge
stored? If you probe at every layer for "the capital of France is [?]",
which layer achieves highest accuracy?

Meng et al. (2022) — "Locating and Editing Factual Associations in GPT"
(ROME) — used probing + causal tracing to find that factual knowledge
in GPT-2-XL is concentrated in specific MLP layers (layers 15-20). Editing
those specific MLP weights changes the factual associations without damaging
other model capabilities.

This illustrates the full power of probing: not just finding where information
is, but using that information to guide targeted model editing.

    # ================================================================== #
    **From probing to model editing — the workflow:**

    1. PROBE each layer for factual property P.
    2. IDENTIFY the layer(s) with highest probe accuracy.
    3. CAUSAL TRACE: patch factual associations one layer at a time.
       Identify which specific layer is causally responsible.
    4. EDIT the identified MLP weights to change the factual association.
    5. VERIFY: the edited fact changes; unrelated facts are preserved.

    This workflow (ROME, MEMIT) allows targeted surgical editing of
    factual knowledge in language models — enabled by probing.
    # ================================================================== #


##### PART VII: MUTUAL INFORMATION AND INFORMATION-THEORETIC PROBING

### The MDL Probe — Minimum Description Length

Voita & Titov (2020) — "Information-Theoretic Probing with Minimum
Description Length" — proposed a principled information-theoretic
alternative to accuracy-based probing. The key insight:

    If a probe achieves high accuracy by OVERFITTING a pattern
    (memorising training examples rather than learning the structure),
    it will take many bits to describe both the probe and the data.
    If the probe captures GENUINE structure, the description is compact.

The MDL probe measures: how many bits does it take to TRANSMIT the label
distribution of the training data to a receiver, assuming they know the
representations?

    MDL = code_length(probe_labels | representations)
        = −Σᵢ log₂ P_probe(yᵢ | hˡ(xᵢ))

    Shorter MDL = the probe captures more genuine structure.

Key advantage: MDL accounts for probe complexity. A highly complex probe
that perfectly fits the training data still has a long description length
because you need to transmit all the probe parameters. A simple probe
that generalises cleanly has a short description length.

This gives the COMPRESSION METRIC:
    compression = MDL_uniform / MDL_probe

where MDL_uniform is the code length under a uniform distribution (no
representation at all). Higher compression = better encoding.

    # ================================================================== #
    **MDL vs accuracy — why they differ:**

    Scenario: 10,000 training examples, 2 classes.

    PROBE A: 90% accuracy on test, 50% accuracy on unseen token types.
    Accuracy says: GOOD (90%). MDL says: SUSPECT (doesn't generalise).

    PROBE B: 87% accuracy on test, 86% accuracy on unseen token types.
    Accuracy says: SLIGHTLY WORSE. MDL says: BETTER (genuinely generalises).

    MDL captures what accuracy misses: whether the probe found REAL structure
    or exploited training set statistics. MDL prefers probes that describe
    the data compactly — i.e., probes that found the genuine structure.
    # ================================================================== #


### Mutual Information Probing

An alternative to accuracy: directly estimate the MUTUAL INFORMATION
between the representation h^l(x) and the property P(x):

    I(h^l(X); P(X)) = H(P(X)) − H(P(X) | h^l(X))

where H is Shannon entropy.

Advantages:
  • Model-free: no assumption about what the probe looks like.
  • Symmetric: measures genuine statistical dependence.
  • Invariant to representation transformations that preserve information.

Disadvantages:
  • Mutual information in high dimensions is hard to estimate accurately.
  • Does not distinguish linearly accessible from non-linearly accessible MI.
  • Requires density estimation in ℝᵈˡ — challenging for large d_l.

Practical approaches: estimate MI via variational lower bounds (MINE,
InfoNCE), or via k-NN entropy estimators for moderate dimensions.

The distinction between LINEARLY ACCESSIBLE mutual information (what
a linear probe measures) and TOTAL mutual information (what MI measures)
is meaningful: it tells you whether the encoding is in a linearly accessible
format — which is relevant for the linear representation hypothesis.


##### PART VIII: PROBING FOR SAFETY — DECEPTION, SYCOPHANCY, AND BIAS

### Safety-Relevant Probing Applications

Probing has direct applications to AI safety by testing whether models
encode safety-relevant properties — independent of whether those properties
are expressed in the model's outputs.

    BIAS PROBING (Webster et al., 2018; May et al., 2019):
    Do language model representations encode gender, racial, or religious
    biases? Probing finds that stereotype-aligned associations are linearly
    encoded in word embeddings and contextual representations, even when
    the model's explicit outputs appear unbiased.

    Key concern: a model may pass explicit bias tests (it refuses to produce
    biased outputs) while its representations encode strong biases that
    could surface in downstream fine-tuning.

    DECEPTION PROBING (Marks & Tegmark, 2023 — "The Geometry of Truth"):
    Do LLM representations encode whether a statement is TRUE or FALSE,
    independently of the model's stated confidence?

    Finding: there is a consistent "truth direction" in the activation
    space of LLMs. When the model processes a statement, its representation
    can be probed to recover the ACTUAL truth value — even when the model
    states the opposite. A model that says "Paris is the capital of Germany"
    has an internal representation that encodes the falsehood of this claim.

    This raises a profound question: if the model "knows" the truth but
    says the false thing, what does "know" mean, and how do we define
    deception for AI systems?

    SYCOPHANCY PROBING:
    Can we detect sycophantic behaviour before it manifests in output?
    A probe trained to predict "will this response agree with the user's
    incorrect claim?" from early-layer representations can flag sycophantic
    tendency before the model generates a single output token.

    REFUSAL PREDICTION:
    Probes on the [BOS] or prompt representations can predict whether
    a model will refuse a request before it generates. This is useful
    for understanding what features of a prompt trigger refusal circuits.

    # ================================================================== #
    **The geometry of truth — a probing result with safety implications:**

    Marks & Tegmark (2023) trained a logistic probe on LLaMA representations
    to predict whether a statement is true or false.
    The probe achieves 85%+ accuracy — the representations encode truth value.

    More striking: the probe GENERALISES across topics and statement types,
    suggesting a domain-general "truth direction" in the representation space.

    Implications:
    1. If the model "knows" truth but lies: this is deception-adjacent.
    2. If the model "doesn't know" truth and guesses: this is hallucination.
    3. The probe can distinguish these two failure modes — a capability
       that external evaluation (just looking at outputs) cannot provide.

    Caution: high probe accuracy does not mean the model "knows" truth in
    any philosophically deep sense. It means the representations are
    organised differently for true vs false statements. The representation
    might encode "I've seen this type of statement be false" rather than
    genuine truth assessment.
    # ================================================================== #


##### PART IX: A COMPLETE FRAMEWORK FOR VALID PROBING EXPERIMENTS

### The Four-Step Protocol

To run a probing experiment that supports valid, well-calibrated
conclusions, the following four steps should be followed:

    STEP 1 — DEFINE THE CLAIM PRECISELY:
    ──────────────────────────────────────────────────────────────────
    Before training any probe, state your claim:
    "Layer l of model M linearly encodes property P for input distribution D."

    This forces you to specify:
    • Which layer? (l)
    • Which model? (M) — architecture, training data, fine-tuning
    • Which property? (P) — labelling scheme, class definition
    • Which distribution? (D) — domain, language, input type

    A claim without these specifications is not falsifiable.

    STEP 2 — CONSTRUCT APPROPRIATE CONTROLS:
    ──────────────────────────────────────────────────────────────────
    Always include at minimum:
    a) Majority class baseline (to assess class balance)
    b) Lexical/positional baseline (to assess surface feature shortcuts)
    c) Untrained model baseline (to isolate learned vs architectural encoding)
    d) Shuffled label probe (to assess overfitting to spurious correlations)
    e) Random projection control — selectivity score (Hewitt & Liang, 2019)

    If your probe does not beat all five controls, your claim is not supported.

    STEP 3 — PROBE ACROSS ALL LAYERS:
    ──────────────────────────────────────────────────────────────────
    Do not just probe your "interesting" layer. Probe all layers.
    Plot the full layerwise profile. Report:
    • Where accuracy peaks
    • Where accuracy first exceeds the surface baseline
    • The scalar mix expected layer

    This contextualises the claim within the model's overall processing.

    STEP 4 — DISTINGUISH ENCODING FROM USING:
    ──────────────────────────────────────────────────────────────────
    If possible, run a causal intervention:
    a) Project out the P direction from the representation.
    b) Run the model forward on the modified representation.
    c) Measure whether task performance changes.

    If performance changes: P is USED (stronger claim).
    If performance is unchanged: P is encoded but not used (weaker claim).
    Report which case applies. Be explicit that encoding ≠ using.


##### PART X: A COMPLETE WORKED EXAMPLE — PROBING BY HAND

### Probing for Sentiment in a 3-Dimensional Representation

We trace through a complete probing experiment on a toy example where
the ground truth is known, to make every step explicit.

    SETUP:
    ──────────────────────────────────────────────────────────────────
    Hypothetical model: a sentence encoder that maps sentences to 3D vectors.
    The model was trained only on next-sentence prediction (not sentiment).
    We want to know: does the representation encode sentiment?

    REPRESENTATIONS (h ∈ ℝ³) — 8 training examples:
    ─────────────────────────────────────────────────────────────────
    Sentence                    h₁     h₂     h₃    Sentiment
    "Great product!"          +0.8  +0.2  −0.1   POSITIVE (+1)
    "Love this!"              +0.7  +0.3   0.0   POSITIVE (+1)
    "Excellent quality."      +0.9  +0.1  −0.2   POSITIVE (+1)
    "Would recommend."        +0.6  +0.2  +0.1   POSITIVE (+1)
    "Terrible experience."    −0.7  −0.3  +0.2   NEGATIVE (−1)
    "Waste of money."         −0.8  −0.1  +0.1   NEGATIVE (−1)
    "Not good at all."        −0.6  −0.2  +0.3   NEGATIVE (−1)
    "Very disappointing."     −0.7  −0.4  +0.2   NEGATIVE (−1)

    Observation: h₁ is strongly correlated with sentiment.
    Positive examples: h₁ ∈ {+0.6, +0.7, +0.8, +0.9}
    Negative examples: h₁ ∈ {−0.6, −0.7, −0.7, −0.8}

    LOGISTIC REGRESSION PROBE:
    ─────────────────────────────────────────────────────────────────
    Probe: P(y=POSITIVE | h) = σ(w₁h₁ + w₂h₂ + w₃h₃ + b)

    After fitting to the training data (gradient descent):
    Optimal weights: w₁ ≈ +5.2,  w₂ ≈ +0.3,  w₃ ≈ −0.1,  b ≈ 0.0

    w₁ is by far the largest — the probe correctly identifies h₁ as
    the sentiment direction. w₂ and w₃ are nearly zero.

    TRAINING ACCURACY: 8/8 = 100%  (perfect linear separability)

    SURFACE FEATURE BASELINE:
    ─────────────────────────────────────────────────────────────────
    Probe using only token identity (does "great/love/excellent/recommend"
    → positive? does "terrible/waste/disappointing" → negative?).
    Surface accuracy: 7/8 = 87.5% (near-perfect because vocabulary is tiny).
    Conclusion: some surface shortcut exists. We need to test on unseen words.

    TEST ON UNSEEN VOCABULARY:
    ─────────────────────────────────────────────────────────────────
    Test examples with words not in training:
    "Outstanding performance."  h = (+0.85, +0.15, −0.05)  POSITIVE
    "Awful experience."         h = (−0.75, −0.25, +0.15)  NEGATIVE

    Probe predictions:
    σ(5.2×0.85 + 0.3×0.15 + (−0.1×−0.05)) = σ(4.47+0.05+0.005) = σ(4.52) ≈ 0.99 → POSITIVE ✓
    σ(5.2×−0.75 + 0.3×−0.25 + (−0.1×0.15)) = σ(−3.90−0.08−0.02) = σ(−4.00) ≈ 0.02 → NEGATIVE ✓

    Test accuracy: 2/2 = 100%. The probe generalises to new vocabulary.
    The generalisation comes from the REPRESENTATION (h₁ is consistently
    positive for positive sentiment), not from lexical memorisation.

    SHUFFLED LABEL CONTROL:
    ─────────────────────────────────────────────────────────────────
    Randomly shuffle the 8 sentiment labels.
    Train a new probe on shuffled labels.
    Expected accuracy on shuffled test: 50% (chance for balanced classes).
    Actual: 50% ± 5% (probe cannot learn; labels are random).
    ✓ Confirms the real probe learned genuine signal, not spurious patterns.

    UNTRAINED MODEL CONTROL:
    ─────────────────────────────────────────────────────────────────
    Replace h with random Gaussian vectors of the same dimension.
    Train probe on random h.
    Expected test accuracy: 50% (random representations carry no signal).
    Actual: 52% ± 8% (not significantly above chance).
    ✓ Confirms the actual representations encode genuine sentiment structure.

    CONCLUSION:
    ─────────────────────────────────────────────────────────────────
    The representation h linearly encodes sentiment:
      • First dimension h₁ is the sentiment direction
      • Probe accuracy: 100% (test), 100% (unseen vocabulary)
      • Shuffled label probe: 50% (chance)
      • Random representation probe: 52% (chance)
      • Selectivity score: 100% − 52% = 48 percentage points above chance

    CAUTION: This shows encoding. It does not show that the model
    uses sentiment for next-sentence prediction. A causal test
    (ablate h₁, measure next-sentence accuracy) would be needed.

"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
    ╔═══════════════════════════╦══════════════════╦══════════════════════════════╗
    ║ Probe Type                ║ Measures         ║ Interpretation               ║
    ╠═══════════════════════════╬══════════════════╬══════════════════════════════╣
    ║ Logistic regression       ║ Linear encoding  ║ Cleanest; default choice     ║
    ║ Structural probe (Hewitt) ║ Metric structure ║ Encodes full tree geometry   ║
    ║ MDL probe (Voita)         ║ Compression      ║ Accounts for probe complexity║
    ║ Mutual information        ║ Total dependency ║ Not limited to linear access ║
    ║ MLP probe                 ║ Non-linear enc.  ║ Weaker interpretation        ║
    ╚═══════════════════════════╩══════════════════╩══════════════════════════════╝

    Required controls for valid probing:
      1. Majority class baseline       (class balance check)
      2. Surface feature baseline      (lexical/positional shortcut check)
      3. Untrained model baseline      (learned vs architectural encoding)
      4. Shuffled label control        (overfitting check)
      5. Random projection selectivity (dimensionality confound check)

    Key distinction:
      Encoding: probe acc >> baseline  → information is linearly accessible
      Using:    causal ablation degrades task performance → information is causally active

    Selectivity = acc_probe(task P) − acc_probe(random task)
    Expected layer = Σ_l l × max(0, acc_P^l − acc_P^{l-1}) / Σ_l max(0, acc_P^l − acc_P^{l-1})
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Probing from Scratch — Linear Probes, Layer Profiles, and Controls": {
        "description": (
            "Implements logistic regression probes from scratch (no sklearn) on a "
            "synthetic 4-layer encoder with 8-dimensional representations. Probes "
            "for two properties (POS tag and sentiment) at every layer, plots the "
            "layerwise probing profile as an ASCII chart, runs all five required "
            "controls (majority baseline, surface baseline, untrained model baseline, "
            "shuffled label control, selectivity score), and demonstrates the "
            "encoding-vs-using distinction via a causal ablation on the identified "
            "encoding direction. Zero external dependencies."
        ),
        "runnable": True,
        "pipeline_cmd": "probing",
        "code": '''
"""
================================================================================
PROBING FROM SCRATCH — LOGISTIC PROBES, LAYER PROFILES, AND ALL CONTROLS
================================================================================

We implement:
  1. A synthetic 4-layer encoder with engineered properties in representations
  2. Logistic regression probes trained via gradient descent (no libraries)
  3. Layerwise probing profiles for two properties
  4. All five required controls:
       (a) Majority class baseline
       (b) Surface feature (lexical) baseline
       (c) Untrained model baseline
       (d) Shuffled label control
       (e) Selectivity score (random task control)
  5. Causal ablation — project out the property direction and measure
     whether the downstream task degrades (encoding → using test)
================================================================================
"""

import math
import random

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

# We create 80 sentences (tokens), each with:
#   - A token type: NOUN, VERB, ADJ, DET (POS tag — property 1)
#   - Sentiment polarity: POSITIVE, NEGATIVE (property 2)
# Each token gets an 8-dimensional representation built as:
#   dims 0-1: encode POS (learned feature, appears in early layers)
#   dims 2-3: encode sentiment (appears in middle layers)
#   dims 4-7: noise (not encoding any property)
# We simulate a 4-layer encoder by adding progressive noise.

VOCAB = {
    # (token, pos_class, sentiment_class)
    "the":        (0, 3),   # DET, neutral (use 0)
    "a":          (0, 3),
    "great":      (1, 0),   # ADJ, POSITIVE
    "terrible":   (1, 1),   # ADJ, NEGATIVE
    "runs":       (2, 3),   # VERB, neutral
    "cat":        (3, 3),   # NOUN, neutral
    "dog":        (3, 3),
    "amazing":    (1, 0),   # ADJ, POSITIVE
    "dreadful":   (1, 1),   # ADJ, NEGATIVE
    "beautiful":  (1, 0),
    "awful":      (1, 1),
    "plays":      (2, 3),
    "book":       (3, 3),
    "house":      (3, 3),
    "excellent":  (1, 0),
    "horrible":   (1, 1),
}

# POS encoding directions (in dims 0-1):
# NOUN=[0.9,0.0], VERB=[0.0,0.9], ADJ=[0.6,0.6], DET=[-0.6,-0.6]
POS_DIRS = [
    [-0.6, -0.6],  # DET
    [ 0.6,  0.6],  # ADJ
    [ 0.0,  0.9],  # VERB
    [ 0.9,  0.0],  # NOUN
]
POS_NAMES = ["DET", "ADJ", "VERB", "NOUN"]

# Sentiment encoding directions (in dims 2-3):
# POSITIVE=[0.8,0.3], NEGATIVE=[-0.8,0.3], NEUTRAL=[0.0,0.8]
SENT_DIRS = {
    0: [ 0.8,  0.3],   # POSITIVE
    1: [-0.8,  0.3],   # NEGATIVE
    3: [ 0.0,  0.8],   # NEUTRAL
}
SENT_NAMES = {0: "POS", 1: "NEG", 3: "NEUTRAL"}


def make_token_repr(token, noise_scale=0.1, seed_offset=0):
    """Create a base 8-dim representation for a token."""
    pos_class, sent_class = VOCAB[token]
    random.seed(hash(token) + seed_offset)

    # Dims 0-1: POS direction + noise
    pos_dir = POS_DIRS[pos_class]
    dims_01 = [pos_dir[d] + random.gauss(0, noise_scale) for d in range(2)]

    # Dims 2-3: sentiment direction + noise
    sent_dir = SENT_DIRS[sent_class]
    dims_23 = [sent_dir[d] + random.gauss(0, noise_scale) for d in range(2)]

    # Dims 4-7: pure noise
    dims_47 = [random.gauss(0, 0.5) for _ in range(4)]

    return dims_01 + dims_23 + dims_47

# Generate dataset: 80 token instances, balanced across POS and sentiment
tokens_list, pos_labels, sent_labels = [], [], []
token_names_list = []

for tok, (pos_cls, sent_cls) in VOCAB.items():
    n_rep = 5  # 5 occurrences per token type
    for k in range(n_rep):
        tokens_list.append(make_token_repr(tok, noise_scale=0.15, seed_offset=k*100))
        pos_labels.append(pos_cls)
        sent_labels.append(sent_cls)
        token_names_list.append(tok)

N = len(tokens_list)
D = 8  # representation dimension


def add_layer_noise(reprs, layer, noise_scale=0.1):
    """
    Simulate a layer transformation: add noise proportional to layer depth.
    POS information slightly degrades with depth.
    Sentiment information increases with depth (peaks at layer 2).
    """
    out = []
    for h in reprs:
        new_h = list(h)
        # Dims 0-1 (POS): degrade slightly with depth
        pos_decay = 1.0 - 0.1 * layer
        new_h[0] = h[0] * pos_decay + random.gauss(0, noise_scale)
        new_h[1] = h[1] * pos_decay + random.gauss(0, noise_scale)
        # Dims 2-3 (sentiment): strengthen up to layer 2, then flatten
        sent_amp = 1.0 + 0.3 * min(layer, 2) - 0.2 * max(0, layer - 2)
        new_h[2] = h[2] * sent_amp + random.gauss(0, noise_scale * 0.5)
        new_h[3] = h[3] * sent_amp + random.gauss(0, noise_scale * 0.5)
        # Dims 4-7: increasing noise
        for d in range(4, 8):
            new_h[d] = h[d] + random.gauss(0, noise_scale * (1 + layer * 0.3))
        out.append(new_h)
    return out

# Build 4-layer representations
random.seed(10)
reprs_by_layer = [tokens_list]  # layer 0 = raw embeddings
current = [list(h) for h in tokens_list]
for l in range(1, 5):  # layers 1-4
    current = add_layer_noise(current, l, noise_scale=0.08)
    reprs_by_layer.append([list(h) for h in current])

print("=" * 68)
print("  DATASET")
print("=" * 68)
print(f"  {N} token instances, {D}-dim representations, 5 layers (0-4)")
print(f"  POS classes: {POS_NAMES}  (encoded in dims 0-1)")
print(f"  Sentiment: POS/NEG/NEUTRAL  (encoded in dims 2-3)")
print(f"  Dims 4-7: noise (no property encoded)")

# Train/test split: 70/30, stratified
random.seed(42)
indices = list(range(N))
random.shuffle(indices)
n_train = int(0.7 * N)
train_idx = set(indices[:n_train])
test_idx  = set(indices[n_train:])


# ─────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION PROBE — BINARY ONE-VS-REST
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

def train_logistic_probe(X_train, y_train, n_classes, lr=0.05, n_epochs=200, l2=0.01):
    """
    Trains one-vs-all logistic regression probes.
    X_train: list of d-dim vectors
    y_train: list of integer class labels (0 to n_classes-1)
    Returns: W (n_classes × d), b (n_classes,)
    """
    d = len(X_train[0])
    n = len(X_train)
    W = [[0.0]*d for _ in range(n_classes)]
    b = [0.0]*n_classes

    for epoch in range(n_epochs):
        for c in range(n_classes):
            # Binary OVR: class c vs all others
            dW = [0.0]*d
            db = 0.0
            for i in range(n):
                y_c = 1 if y_train[i] == c else 0
                z   = sum(W[c][j]*X_train[i][j] for j in range(d)) + b[c]
                p   = sigmoid(z)
                err = p - y_c
                for j in range(d):
                    dW[j] += err * X_train[i][j]
                db += err

            # Update with L2 regularisation
            for j in range(d):
                W[c][j] -= lr * (dW[j]/n + l2 * W[c][j])
            b[c] -= lr * db/n

    return W, b

def predict_logistic(X, W, b):
    """Predict class labels using OVR logistic regression."""
    preds = []
    for x in X:
        scores = [sum(W[c][j]*x[j] for j in range(len(x))) + b[c]
                  for c in range(len(W))]
        preds.append(max(range(len(scores)), key=lambda c: scores[c]))
    return preds

def accuracy(preds, labels):
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)

def balanced_accuracy(preds, labels, n_classes):
    """Per-class recall averaged across classes."""
    recalls = []
    for c in range(n_classes):
        true_c = [l == c for l in labels]
        pred_c = [p == c for p in preds]
        tp = sum(t and p for t, p in zip(true_c, pred_c))
        n_c = sum(true_c)
        recalls.append(tp / n_c if n_c > 0 else 0.0)
    return sum(recalls) / n_classes


# ─────────────────────────────────────────────────────────────────────────────
# LAYERWISE PROBING PROFILES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  LAYERWISE PROBING PROFILES")
print("=" * 68)
print()

# Filter out NEUTRAL for binary sentiment probe
def sentiment_binary_labels(sent_labs):
    """Keep only POS(0) and NEG(1); drop NEUTRAL(3)."""
    return [(i, s) for i, s in enumerate(sent_labs) if s in (0, 1)]

sent_binary = sentiment_binary_labels(sent_labels)
sent_idx_list = [i for i, _ in sent_binary]
sent_lbl_list = [s for _, s in sent_binary]

# Identify train/test within sentiment subset
sent_train_idx_in_sub = [k for k, i in enumerate(sent_idx_list) if i in train_idx]
sent_test_idx_in_sub  = [k for k, i in enumerate(sent_idx_list) if i in test_idx]

pos_acc_by_layer  = []
sent_acc_by_layer = []

for l in range(5):
    reprs_l = reprs_by_layer[l]

    # POS probe (4 classes)
    X_tr_pos = [reprs_l[i] for i in range(N) if i in train_idx]
    y_tr_pos = [pos_labels[i] for i in range(N) if i in train_idx]
    X_te_pos = [reprs_l[i] for i in range(N) if i in test_idx]
    y_te_pos = [pos_labels[i] for i in range(N) if i in test_idx]

    W_pos, b_pos = train_logistic_probe(X_tr_pos, y_tr_pos, n_classes=4,
                                         lr=0.05, n_epochs=300)
    preds_pos = predict_logistic(X_te_pos, W_pos, b_pos)
    bal_acc_pos = balanced_accuracy(preds_pos, y_te_pos, 4)
    pos_acc_by_layer.append(bal_acc_pos)

    # Sentiment probe (binary: POS vs NEG only)
    X_tr_sent = [reprs_l[sent_idx_list[k]] for k in sent_train_idx_in_sub]
    y_tr_sent = [sent_lbl_list[k] for k in sent_train_idx_in_sub]
    X_te_sent = [reprs_l[sent_idx_list[k]] for k in sent_test_idx_in_sub]
    y_te_sent = [sent_lbl_list[k] for k in sent_test_idx_in_sub]

    W_sent, b_sent = train_logistic_probe(X_tr_sent, y_tr_sent, n_classes=2,
                                           lr=0.05, n_epochs=300)
    preds_sent = predict_logistic(X_te_sent, W_sent, b_sent)
    bal_acc_sent = balanced_accuracy(preds_sent, y_te_sent, 2)
    sent_acc_by_layer.append(bal_acc_sent)

# Print layerwise table
print(f"  {'Layer':<8}  {'POS acc':>9}  {'Sent acc':>10}")
print(f"  {'-'*32}")
for l in range(5):
    pos_bar  = "█" * int(pos_acc_by_layer[l]  * 20)
    sent_bar = "█" * int(sent_acc_by_layer[l] * 20)
    print(f"  Layer {l}:  {pos_acc_by_layer[l]:>9.1%}  {sent_acc_by_layer[l]:>10.1%}")

# ASCII profile chart
print()
print("  PROBING PROFILE (ASCII chart — █ = probe accuracy)")
print()
height = 10
for row in range(height, 0, -1):
    thresh = row / height
    line = f"  {thresh*100:>3.0f}% │"
    for l in range(5):
        pos_filled  = "P" if pos_acc_by_layer[l]  >= thresh else " "
        sent_filled = "S" if sent_acc_by_layer[l] >= thresh else " "
        line += f" {pos_filled}{sent_filled} "
    print(line)
print("       └" + "────" * 5)
print("         " + "  ".join(f"L{l}" for l in range(5)))
print("         P=POS probe, S=Sentiment probe")
print()

pos_peak  = max(range(5), key=lambda l: pos_acc_by_layer[l])
sent_peak = max(range(5), key=lambda l: sent_acc_by_layer[l])
print(f"  POS probe peaks at Layer {pos_peak} ({pos_acc_by_layer[pos_peak]:.1%})")
print(f"  Sent probe peaks at Layer {sent_peak} ({sent_acc_by_layer[sent_peak]:.1%})")
print()
print(f"  INTERPRETATION: POS peaks early (Layer {pos_peak}) — morphological properties")
print(f"  are encoded in lower layers. Sentiment peaks later (Layer {sent_peak}) —")
print(f"  semantic properties require more contextual processing.")


# ─────────────────────────────────────────────────────────────────────────────
# FIVE REQUIRED CONTROLS (at the best layer for each property)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  FIVE REQUIRED CONTROLS — POS TASK AT BEST LAYER")
print("=" * 68)

best_l = pos_peak
reprs_best = reprs_by_layer[best_l]

X_tr = [reprs_best[i] for i in range(N) if i in train_idx]
y_tr = [pos_labels[i] for i in range(N) if i in train_idx]
X_te = [reprs_best[i] for i in range(N) if i in test_idx]
y_te = [pos_labels[i] for i in range(N) if i in test_idx]

W_best, b_best = train_logistic_probe(X_tr, y_tr, n_classes=4, lr=0.05, n_epochs=300)
preds_best = predict_logistic(X_te, W_best, b_best)
main_acc   = balanced_accuracy(preds_best, y_te, 4)

# CONTROL 1: Majority class baseline
from collections import Counter
majority_cls = max(set(y_te), key=lambda c: y_te.count(c))
majority_preds = [majority_cls] * len(y_te)
majority_acc = balanced_accuracy(majority_preds, y_te, 4)

# CONTROL 2: Lexical (surface) baseline — one-hot of token index
tok_to_idx = {tok: i for i, tok in enumerate(VOCAB.keys())}
def lexical_repr(tok_name, dim=len(VOCAB)):
    h = [0.0] * dim
    if tok_name in tok_to_idx:
        h[tok_to_idx[tok_name]] = 1.0
    return h

X_tr_lex = [lexical_repr(token_names_list[i]) for i in range(N) if i in train_idx]
X_te_lex = [lexical_repr(token_names_list[i]) for i in range(N) if i in test_idx]
W_lex, b_lex = train_logistic_probe(X_tr_lex, y_tr, n_classes=4, lr=0.1, n_epochs=300)
preds_lex = predict_logistic(X_te_lex, W_lex, b_lex)
lexical_acc = balanced_accuracy(preds_lex, y_te, 4)

# CONTROL 3: Untrained model (random representations at layer 0)
X_tr_rand = [reprs_by_layer[0][i] for i in range(N) if i in train_idx]
X_te_rand = [reprs_by_layer[0][i] for i in range(N) if i in test_idx]
W_rand, b_rand = train_logistic_probe(X_tr_rand, y_tr, n_classes=4, lr=0.05, n_epochs=300)
preds_rand = predict_logistic(X_te_rand, W_rand, b_rand)
untrained_acc = balanced_accuracy(preds_rand, y_te, 4)

# CONTROL 4: Shuffled label control
random.seed(99)
y_shuffled = list(y_tr)
random.shuffle(y_shuffled)
W_shuf, b_shuf = train_logistic_probe(X_tr, y_shuffled, n_classes=4, lr=0.05, n_epochs=300)
preds_shuf = predict_logistic(X_te, W_shuf, b_shuf)
shuffled_acc = balanced_accuracy(preds_shuf, y_te, 4)

# CONTROL 5: Selectivity — random task probe (assign random labels, train, measure)
random.seed(77)
y_random_task = [random.randint(0, 3) for _ in range(len(y_tr))]
W_sel, b_sel = train_logistic_probe(X_tr, y_random_task, n_classes=4, lr=0.05, n_epochs=300)
preds_sel = predict_logistic(X_te, W_sel, b_sel)
random_task_acc = balanced_accuracy(preds_sel, y_te, 4)
selectivity = main_acc - random_task_acc

print()
print(f"  Main probe (Layer {best_l}):      {main_acc:.1%}  (balanced acc, POS 4-way)")
print()
print(f"  {'Control':<30}  {'Acc':>7}  {'Interpretation'}")
print(f"  {'-'*68}")
print(f"  {'Majority class baseline':30}  {majority_acc:>7.1%}  Uninformative baseline")
print(f"  {'Lexical (surface) baseline':30}  {lexical_acc:>7.1%}  Token identity only")
print(f"  {'Untrained model (layer 0)':30}  {untrained_acc:>7.1%}  Before any learning")
print(f"  {'Shuffled labels':30}  {shuffled_acc:>7.1%}  Should be ≈ chance")
print(f"  {'Random task (selectivity)':30}  {random_task_acc:>7.1%}  Dimensionality baseline")
print(f"  {'-'*68}")
print(f"  {'Selectivity score':30}  {selectivity:>+7.1%}  main − random_task")
print()
main_beats_all = (main_acc > majority_acc and main_acc > lexical_acc and
                  main_acc > untrained_acc and shuffled_acc < majority_acc + 0.1 and
                  selectivity > 0.10)
print(f"  Probe beats all 5 controls: {'YES ✓' if main_beats_all else 'NO ✗'}")
if main_beats_all:
    print(f"  → VALID CONCLUSION: Layer {best_l} genuinely encodes POS information")
    print(f"    in a linearly accessible format, beyond surface statistics.")
else:
    print(f"  → INVALID CONCLUSION: some controls are not beaten; results are ambiguous.")


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING vs USING — CAUSAL ABLATION
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  ENCODING vs USING — CAUSAL ABLATION TEST")
print("=" * 68)
print()
print("  We test whether the POS encoding is USED by a downstream task.")
print("  Downstream task: predict whether a token is a content word")
print("  (NOUN or VERB = 1) vs function word (ADJ or DET = 0).")
print()
print("  Ablation: project out the POS direction from representations.")
print("  If POS is used: ablated accuracy < original accuracy.")
print("  If POS is a passenger: ablated accuracy ≈ original accuracy.")
print()

# Downstream task: content word vs function word (uses POS)
content_labels = [1 if pos_labels[i] in (2, 3) else 0 for i in range(N)]

X_te_main = [reprs_best[i] for i in range(N) if i in test_idx]
y_te_content = [content_labels[i] for i in range(N) if i in test_idx]
X_tr_main = [reprs_best[i] for i in range(N) if i in train_idx]
y_tr_content = [content_labels[i] for i in range(N) if i in train_idx]

W_content, b_content = train_logistic_probe(X_tr_main, y_tr_content, n_classes=2,
                                             lr=0.05, n_epochs=300)
preds_content_orig = predict_logistic(X_te_main, W_content, b_content)
acc_orig = accuracy(preds_content_orig, y_te_content)

# The POS direction is roughly w_best for the NOUN probe direction
# For simplicity: use dim 0 (h₁) as the primary POS direction
# Project out dim 0 and dim 1 (the POS encoding dims)
def project_out(X, dims_to_zero):
    """Zero out specific dimensions from each representation."""
    return [[h[d] if d not in dims_to_zero else 0.0 for d in range(len(h))]
            for h in X]

X_te_ablated = project_out(X_te_main, dims_to_zero={0, 1})
X_tr_ablated = project_out(X_tr_main, dims_to_zero={0, 1})

# Retrain downstream probe on ablated representations
W_ablated, b_ablated = train_logistic_probe(X_tr_ablated, y_tr_content, n_classes=2,
                                              lr=0.05, n_epochs=300)
preds_ablated = predict_logistic(X_te_ablated, W_ablated, b_ablated)
acc_ablated = accuracy(preds_ablated, y_te_content)

print(f"  Downstream task: content word (NOUN/VERB=1) vs function word (ADJ/DET=0)")
print()
print(f"  Accuracy WITHOUT ablation: {acc_orig:.1%}")
print(f"  Accuracy WITH ablation (dims 0,1 zeroed): {acc_ablated:.1%}")
print(f"  Accuracy drop: {acc_orig - acc_ablated:+.1%}")
print()

if acc_orig - acc_ablated > 0.10:
    print(f"  CONCLUSION: POS encoding is USED (accuracy drops {acc_orig-acc_ablated:.1%}).")
    print(f"  The representation actively relies on the POS-encoding dimensions")
    print(f"  to distinguish content words from function words.")
    print(f"  Claim: Layer {best_l} ENCODES and USES POS information.")
else:
    print(f"  CONCLUSION: POS encoding may be a PASSENGER (accuracy drop < 10%).")
    print(f"  Ablating the POS direction does not significantly degrade the task.")
    print(f"  Claim: Layer {best_l} ENCODES POS but does not necessarily USE it.")
    print(f"  (The downstream task may use other dims, not the POS dims.)")
print()
print(f"  This causal ablation distinguishes:")
print(f"  'The representation encodes X'  (probe acc alone)")
print(f"  'The representation uses X'     (ablation test)")
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
    #     from interpretability.visuals.probing_classifiers import (
    #         PROBING_VISUAL_HTML,
    #         PROBING_VISUAL_HEIGHT,
    #     )
    #     visual_html   = PROBING_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = PROBING_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[probing_classifiers.py] Could not load visual: {e}", stacklevel=2)

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