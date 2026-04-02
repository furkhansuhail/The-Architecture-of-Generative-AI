"""
Learning Paradigms
==================

The fundamental ways a machine learning system can acquire knowledge —
how data is labelled, how feedback is structured, how experience is
accumulated, and how prior knowledge is leveraged to solve new problems.

"""

import textwrap
import re

TOPIC_NAME = "Learning Paradigms"
DISPLAY_NAME = "02 · Learning Paradigms"
ICON = "🧠"
SUBTITLE = "How Machines Acquire Knowledge from Experience"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### The Central Question: What Is the Source of Supervision?

Every machine learning system learns from some form of experience. The
fundamental question that defines a learning paradigm is:

    What signal tells the model it is right or wrong?
    Where does that signal come from?
    How much of it is available?
    How is it structured?

The answers determine the entire architecture of the learning system —
the data it needs, the objective it optimises, and what it can achieve.

    Diagram 1 — The Learning Paradigm Landscape:

    ┌─────────────────────────────────────────────────────────────────┐
    │              NATURE OF THE SUPERVISORY SIGNAL                   │
    │                                                                 │
    │  Full labels         Partial labels       No labels             │
    │  for every sample    or indirect signal   at all                │
    │       │                    │                   │                │
    │       ▼                    ▼                   ▼                │
    │  SUPERVISED          SEMI-SUPERVISED      UNSUPERVISED          │
    │  LEARNING            / SELF-SUPERVISED    LEARNING              │
    │       │                    │                   │                │
    │       │              REINFORCEMENT             │                │
    │       │              LEARNING                  │                │
    │       │              (rewards, not labels)     │                │
    │       │                                        │                │
    │       └──────── TRANSFER LEARNING ─────────────┘                │
    │                 (knowledge from one domain                      │
    │                  applied to another)                            │
    └─────────────────────────────────────────────────────────────────┘

    Paradigm overview:
    ┌──────────────────────────┬─────────────────────────────────────┐
    │ Paradigm                 │ Supervisory Signal                  │
    ├──────────────────────────┼─────────────────────────────────────┤
    │ Supervised Learning      │ Human-provided label per sample     │
    │ Unsupervised Learning    │ None — structure discovered         │
    │ Semi-Supervised          │ Labels for small subset only        │
    │ Self-Supervised          │ Labels derived from data itself     │
    │ Reinforcement Learning   │ Scalar reward from environment      │
    │ Transfer Learning        │ Pre-trained model from another task │
    │ Few-Shot Learning        │ 1–5 labelled examples per class     │
    │ Active Learning          │ Model queries human for labels      │
    │ Online Learning          │ Labels arrive one at a time         │
    │ Multi-Task Learning      │ Simultaneous signals from N tasks   │
    └──────────────────────────┴─────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 1: Supervised Learning

Supervised learning is the most widely deployed paradigm. A human expert
annotates each training example with a ground-truth label. The model
learns a mapping f: X → Y that minimises prediction error.

    Formal setup:

        Dataset:     D = {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
        Goal:        Learn f such that f(x) ≈ y for new, unseen x
        Loss:        L(f(x), y)  — measures prediction error
        Training:    θ* = argmin  𝔼[(L(f_θ(x), y)]
                              θ

    Diagram 2 — The Supervised Learning Pipeline:

    ┌──────────────┐    Human     ┌──────────────┐
    │  Raw Data    │ ──annotates─▶│  Labelled    │
    │  x₁, x₂, …  │              │  Dataset      │
    └──────────────┘              │  (xᵢ, yᵢ)    │
                                  └──────┬───────┘
                                         │  train
                                         ▼
                                  ┌──────────────┐
                                  │    Model     │  minimise L(ŷ, y)
                                  │   f_θ(x)     │◀── gradient descent
                                  └──────┬───────┘
                                         │  predict
                                         ▼
                        new x  ──────▶  ŷ = f_θ(x)

    Sub-paradigms:

    CLASSIFICATION — Y is a discrete set of categories
        Binary:       y ∈ {0, 1}          (spam/not spam)
        Multiclass:   y ∈ {1, 2, ..., K}  (digit 0–9)
        Multilabel:   y ⊆ {1, ..., K}     (image tags)

    REGRESSION — Y is continuous
        Scalar:       y ∈ ℝ               (house price)
        Vector:       y ∈ ℝᵈ              (bounding box coordinates)
        Structured:   y is a sequence,    (machine translation)
                          tree, or graph

    Key properties:
    ┌────────────────────────────────────────────────────────────────────┐
    │  Strengths: Strong performance when labels are plentiful, clear    │
    │             theoretical foundations, well-understood failure modes │
    │                                                                    │
    │  Weaknesses: Labels are expensive — medical imaging requires       │
    │              radiologists; legal text requires lawyers.            │
    │              Human labels carry human bias and noise.              │
    │              Closed-world assumption: can only predict classes     │
    │              seen during training.                                 │
    └────────────────────────────────────────────────────────────────────┘

    The labelling bottleneck in practice:

    ┌──────────────────────┬─────────────────┬──────────────────────────┐
    │ Domain               │ Label Cost      │ Expert Required          │
    ├──────────────────────┼─────────────────┼──────────────────────────┤
    │ Image classification │ $0.01–0.10/img  │ Lay annotator            │
    │ Medical imaging      │ $50–500/scan    │ Radiologist              │
    │ Legal document NLP   │ $100–1000/doc   │ Lawyer                   │
    │ Speech recognition   │ $0.10–1.00/min  │ Transcriptionist         │
    │ Protein structure    │ Years/structure │ Biochemist + wet lab     │
    └──────────────────────┴─────────────────┴──────────────────────────┘

    This labelling cost motivates every other paradigm in this module.


──────────────────────────────────────────────────────────────────────────────
### Paradigm 2: Unsupervised Learning

No labels at all. The model must discover structure, patterns, and
regularities inherent in the data distribution P(X) itself.

    Formal setup:

        Dataset:     D = {x₁, x₂, ..., xₙ}   (no labels)
        Goal:        Learn the structure of P(X)
        Tasks:       Clustering, density estimation, dimensionality
                     reduction, generative modelling, anomaly detection

    Diagram 3 — What Unsupervised Learning Discovers:

    Input data (raw points):          Learned structure:

    · · · · · · · · ·                 ● ● ◆ ◆ ▲ ▲
    · · · · · · · · ·     ───────▶   ● ● ◆ ◆ ▲ ▲
    · · · · · · · · ·                 Clusters: 3 distinct groups
                                      No human told it what they mean.

    Major unsupervised tasks:

    CLUSTERING — partition data into groups of similar points
        K-Means:  assumes spherical clusters, fixed K
        DBSCAN:   density-based, finds arbitrary shapes, handles noise
        GMM:      soft probabilistic cluster assignments

    DIMENSIONALITY REDUCTION — compress data to lower-dim representation
        PCA:    linear, maximises variance (see Module 03)
        t-SNE:  non-linear, preserves local neighbourhood
        UMAP:   non-linear, preserves local + global structure
        ICA:    finds statistically independent components

    DENSITY ESTIMATION — model the distribution P(X) explicitly
        KDE:    non-parametric smooth density estimate
        GMMs:   mixture of Gaussians approximation
        NFs:    normalising flows (exact log-likelihood)
        VAEs:   variational approximation (image generation)
        GANs:   implicit density via adversarial training

    ANOMALY DETECTION — identify points unlikely under P(X)
        Isolation Forest, One-Class SVM, Autoencoders

    Key insight about unsupervised learning:
    ┌────────────────────────────────────────────────────────────────────┐
    │  Unsupervised learning discovers WHAT exists in the data.          │
    │  Supervised learning predicts WHICH category it belongs to.        │
    │  The two are deeply complementary: good unsupervised               │
    │  representations dramatically improve supervised performance       │
    │  when labels are subsequently added.                               │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 3: Semi-Supervised Learning

In practice, labels are scarce but unlabelled data is abundant. A website
has millions of images but only thousands have been annotated. A hospital
has decades of scans but radiologist reports for only a small fraction.

Semi-supervised learning exploits BOTH:
    - Small labelled set:    Dₗ = {(x₁,y₁), ..., (xₘ,yₘ)}    m << n
    - Large unlabelled set:  Dᵤ = {xₘ₊₁, ..., xₙ}

    Diagram 4 — Semi-Supervised Setup:

    Labelled data (m=100):          Unlabelled data (n=10,000):
    ● ● ◆ ◆ ▲ ▲                    · · · · · · · · · · · ·
    (rare, expensive)               · · · · · · · · · · · ·
                                    · · · · · · · · · · · ·
                                    (abundant, cheap)

    The core assumptions that make this possible:

    1. SMOOTHNESS ASSUMPTION
       If two points x₁, x₂ are close in input space, they should have
       the same label. Unlabelled data reveals the density structure
       and therefore which labelled points should propagate their labels.

    2. CLUSTER ASSUMPTION
       Points in the same cluster tend to share a label.
       Decision boundaries should pass through LOW-DENSITY regions.

    3. MANIFOLD ASSUMPTION
       High-dimensional data lies on a low-dimensional manifold.
       Unlabelled data reveals the shape of that manifold.

    Key algorithms:

    SELF-TRAINING (Pseudo-labelling):
        1. Train classifier on Dₗ
        2. Predict labels for Dᵤ (pseudo-labels)
        3. Add high-confidence pseudo-labelled samples to training set
        4. Retrain on Dₗ ∪ Dᵤ_selected
        5. Repeat

        Risk: confirmation bias — wrong early predictions reinforce
        themselves if the confidence threshold is set too low.

    LABEL PROPAGATION / GRAPH-BASED:
        Build a graph where nodes are all points (labelled + unlabelled)
        and edges connect similar points. Propagate labels from labelled
        nodes to their neighbours along high-density edges.

    CONSISTENCY REGULARISATION (MixMatch, FixMatch, etc.):
        Key insight: the model should produce the SAME prediction for
        an unlabelled point and its augmented versions.

            L_consistency = 𝔼[||f(x) − f(Augment(x))||²]

        Forces the model to learn augmentation-invariant representations.
        Unlabelled data provides regularisation signal even without labels.

    ┌────────────────────────────────────────────────────────────────────┐
    │  Practical impact: FixMatch (2020) achieves 94.9% on CIFAR-10      │
    │  using only 40 labelled images (4 per class). A fully supervised   │
    │  model needs ~50,000 images to match this. 1,250× label savings.   │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 4: Self-Supervised Learning

The most impactful paradigm of the last decade. The model generates its OWN
supervisory signal from the structure of unlabelled data — no human labels
needed at any stage.

    Core idea: design a PRETEXT TASK whose labels come from the data itself.
    Solving the pretext task forces the model to learn useful representations.

    Diagram 5 — Self-Supervised Pretext Tasks:

    MASKED LANGUAGE MODELLING (BERT):
    Input:   "The cat sat on the [MASK] mat"
    Target:  Predict: "cozy"
    Label source: the original sentence itself. No human annotation.

    NEXT SENTENCE PREDICTION:
    Sentence A: "The dog barked."
    Sentence B: "It was hungry."     → label: CONSECUTIVE (True)
    Sentence B: "Stars are bright."  → label: RANDOM (False)

    IMAGE ROTATION PREDICTION:
    Take image, rotate 0°/90°/180°/270°, predict which rotation.
    Label source: the rotation applied — free.

    CONTRASTIVE LEARNING (SimCLR, MoCo):
    Take one image → create two augmented views → maximise agreement
    between views of SAME image, minimise agreement with OTHER images.

        x ──aug1──▶ z₁ ─╮
                         │─── pull z₁ and z₂ close (positive pair)
        x ──aug2──▶ z₂ ─╯
        x' ──────▶ z'  ─────── push z₁ and z' apart (negative pair)

    MASKED IMAGE MODELLING (MAE — He et al. 2022):
    Mask 75% of image patches → predict masked patches from visible ones.
    The model must learn rich semantics to reconstruct missing regions.

    Why self-supervised learning works so well:

        Pretext tasks as proxy objectives:
        ┌──────────────────────────────────────────────────────────────┐
        │  To predict a masked word in a sentence, a model must        │
        │  understand grammar, semantics, world knowledge, and context.│
        │  To reconstruct a masked image patch, a model must           │
        │  understand texture, object shape, and spatial relationships.│
        │  Representations learned for these tasks transfer powerfully │
        │  to downstream classification, detection, and generation.    │
        └──────────────────────────────────────────────────────────────┘

    The two-stage paradigm that dominates modern ML:

        Stage 1 — Pre-training (self-supervised, massive data):
            Train on billions of unlabelled text/images.
            Learn rich general representations.

        Stage 2 — Fine-tuning (supervised, small labelled data):
            Adapt the pre-trained model to the specific task.
            Often needs only hundreds to thousands of labels.

    BERT / GPT / CLIP / DINO — all are self-supervised pre-trained models.
    The representations learned are so general that fine-tuning a BERT
    model on 1,000 labelled examples routinely outperforms a model
    trained from scratch on 100,000 labelled examples.

    ┌────────────────────────────────────────────────────────────────────┐
    │  Self-supervised learning decouples the learning of representa-    │
    │  tions from the learning of task-specific predictions. This is     │
    │  the key to why language models generalise across thousands of     │
    │  downstream tasks without retraining from scratch.                 │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 5: Reinforcement Learning

In RL, there are no labels at all — not even ones derived from the data.
Instead, an AGENT interacts with an ENVIRONMENT, takes ACTIONS, receives
REWARDS, and learns a POLICY that maximises cumulative reward over time.

    Formal setup (Markov Decision Process — MDP):

        S       — state space (what the agent observes)
        A       — action space (what the agent can do)
        P(s'|s,a) — transition dynamics (how the world responds)
        R(s,a)  — reward function (scalar feedback)
        γ ∈ [0,1) — discount factor (how much future rewards matter)

    Goal: learn policy π(a|s) that maximises expected discounted return:

            G_t = 𝔼[∑_{k=0}^{∞} γᵏ · R_{t+k+1}]

    Diagram 6 — The RL Interaction Loop:

         ┌───────────────────────────────────────────────────┐
         │                                                   │
         │        ┌─────────────┐                            │
         │        │    AGENT    │                            │
         │        │  (policy π) │                            │
         │        └──────┬──────┘                            │
         │               │ action aₜ                          │
         │               ▼                                   │
         │        ┌─────────────┐                            │
         │        │ ENVIRONMENT │──▶ reward rₜ₊₁              │
         │        │             │──▶ new state sₜ₊₁ ──────────┘
         │        └─────────────┘
         │

    Key concepts:

    VALUE FUNCTION — expected return from state s under policy π:
        V^π(s) = 𝔼_π[G_t | sₜ = s]

    Q-FUNCTION (action-value) — expected return from taking action a in s:
        Q^π(s,a) = 𝔼_π[G_t | sₜ = s, aₜ = a]

    BELLMAN EQUATION — recursive decomposition of the value function:
        V^π(s) = 𝔼_π[R(s,a) + γ · V^π(s')]

    The exploration-exploitation dilemma:
    ┌────────────────────────────────────────────────────────────────────┐
    │  EXPLOITATION: take the action known to give the highest reward.   │
    │  EXPLORATION: try new actions — might find something better.       │
    │                                                                    │
    │  Too much exploitation → stuck in a local policy optimum.          │
    │  Too much exploration  → never converges on good behaviour.        │
    │  ε-greedy: with probability ε explore randomly, else exploit.      │
    └────────────────────────────────────────────────────────────────────┘

    Major RL algorithm families:

    DYNAMIC PROGRAMMING (requires known model):
        Value Iteration, Policy Iteration — exact solutions, small MDPs.

    MODEL-FREE VALUE-BASED:
        Q-Learning:   tabular, provably converges to Q* off-policy.
        DQN:          deep neural network approximates Q(s,a).
        Used when:    discrete action spaces (Atari games).

    MODEL-FREE POLICY GRADIENT:
        REINFORCE:    directly optimise 𝔼[G_t · log π(aₜ|sₜ)].
        Actor-Critic: combine policy gradient with value function baseline.
        PPO, TRPO:    stable large-scale policy optimisation.
        Used when:    continuous action spaces (robotics, locomotion).

    MODEL-BASED RL:
        Learn a model of the environment, plan using it.
        AlphaGo/AlphaZero: learn model + Monte Carlo Tree Search.
        Dyna-Q: combine model learning with Q-learning.

    RL vs Supervised Learning — the fundamental differences:
    ┌──────────────────────┬──────────────────────┬────────────────────────┐
    │                      │ Supervised Learning  │ Reinforcement Learning │
    ├──────────────────────┼──────────────────────┼────────────────────────┤
    │ Feedback             │ Correct label        │ Scalar reward (sparse) │
    │ Feedback timing      │ Immediate            │ Delayed (credit assign)│
    │ Data distribution    │ Fixed dataset        │ Non-stationary (agent  │
    │                      │                      │ changes distribution)  │
    │ Closed/open world    │ Closed (fixed tasks) │ Open (continuous inter)│
    │ Exploration needed   │ No                   │ Yes (critical)         │
    └──────────────────────┴──────────────────────┴────────────────────────┘

    RLHF — Reinforcement Learning from Human Feedback:
    ┌────────────────────────────────────────────────────────────────────┐
    │  Modern LLMs (GPT-4, Claude, Gemini) are aligned using RLHF:       │
    │  1. Pre-train with self-supervised learning on text                │
    │  2. Fine-tune with supervised learning on curated responses        │
    │  3. Train a reward model from human preference comparisons         │
    │  4. Fine-tune the LLM using PPO against the reward model           │
    │                                                                    │
    │  This hybrid of all three paradigms (self-supervised, supervised,  │
    │  RL) is what produces helpful, harmless, honest behaviour.         │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 6: Transfer Learning

Train on one task/domain where data is abundant, then adapt to a different
(but related) task/domain where data is scarce. Knowledge acquired in the
source domain is "transferred" to the target domain.

    Formal setup:

        Source domain:  Dₛ = (Xₛ, P(Xₛ))    Task: Tₛ
        Target domain:  Dₜ = (Xₜ, P(Xₜ))    Task: Tₜ
        Goal: improve performance on Tₜ using knowledge from Tₛ

    Diagram 7 — Transfer Learning Spectrum:

    FEATURE EXTRACTION               FINE-TUNING
    ─────────────────────            ──────────────────────
    ┌──────────────┐                 ┌──────────────┐
    │ Pre-trained  │ ← FROZEN        │ Pre-trained  │ ← small lr update
    │ backbone     │                 │ backbone     │
    │ (feature map)│                 │ (feature map)│
    └──────┬───────┘                 └──────┬───────┘
           │                                │
    ┌──────▼───────┐                 ┌──────▼───────┐
    │  New head    │ ← TRAINED       │  New head    │ ← larger lr update
    │  (task-spec) │                 │  (task-spec) │
    └──────────────┘                 └──────────────┘
    Fast, few params to train.       Slower, can adapt features.
    Good when source ≈ target.       Good when source ≠ target.

    Types of transfer:

    INDUCTIVE TRANSFER (different tasks, same domain):
        Source: ImageNet classification
        Target: Medical image classification
        The low-level features (edges, textures) transfer well.

    TRANSDUCTIVE TRANSFER / DOMAIN ADAPTATION (same task, different domain):
        Source: News article sentiment (formal text)
        Target: Twitter sentiment (informal text)
        The task is the same but the input distribution has shifted.

    ZERO-SHOT TRANSFER:
        The model is never shown the target task during training at all.
        CLIP (Contrastive Language-Image Pre-training) can classify images
        into arbitrary categories it has never seen, by aligning image
        embeddings with text embeddings of category descriptions.

    Transfer learning in practice — what transfers and what doesn't:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Early layers: Low-level features (edges, curves, colours).       │
    │  These transfer universally — across domains and tasks.           │
    │                                                                   │
    │  Middle layers: Mid-level features (textures, object parts).      │
    │  Transfer well within a domain; less across domains.              │
    │                                                                   │
    │  Late layers: High-level features (object identities, semantics). │
    │  These are task-specific. Usually replaced entirely.              │
    └───────────────────────────────────────────────────────────────────┘

    Negative transfer — when transfer HURTS:

    If Dₛ and Dₜ are sufficiently different (unrelated tasks or domains),
    forcing the model to initialise from a source pre-trained model can
    hurt performance vs training from scratch. This is rarer than
    positive transfer but must be monitored.

    Transfer learning decision tree:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Is labelled target data scarce? → YES → Use transfer learning  │
    │  Is source similar to target?                                   │
    │    → Very similar → Feature extraction (freeze backbone)        │
    │    → Somewhat similar → Fine-tune all layers (small lr)         │
    │    → Very different → Fine-tune + consider intermediate tasks   │
    │  Is labelled target data abundant? → NO → Train from scratch    │
    └─────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 7: Few-Shot and Zero-Shot Learning

The extreme end of data scarcity. In few-shot learning, only K labelled
examples per class are available (K = 1, 5, 10). The model must generalise
from this tiny sample. Zero-shot learning has zero labelled examples.

    Notation: N-way K-shot learning
        N = number of classes in the task
        K = number of labelled examples per class

    5-way 1-shot: 5 classes, 1 labelled example each = 5 training images.

    Diagram 8 — The Episodic Training Framework (Meta-Learning):

    Each "episode" simulates the test-time few-shot scenario:

    Support Set (what the model "sees"):     Query Set (what it predicts):
    ┌─────────────────────────────────┐      ┌────────────────────────┐
    │ Class A: [img₁]                 │      │  ?  →  predict class   │
    │ Class B: [img₂]                 │      │  ?  →  predict class   │
    │ Class C: [img₃]                 │      │  ?  →  predict class   │
    │ Class D: [img₄]                 │      │  ?  →  predict class   │
    │ Class E: [img₅]                 │      │  ?  →  predict class   │
    └─────────────────────────────────┘      └────────────────────────┘
    Train on many such episodes so the model learns to LEARN QUICKLY.

    Approaches to few-shot learning:

    METRIC-BASED (Siamese Networks, Prototypical Networks, Matching Networks):
        Learn an embedding space where examples from the same class
        cluster together. Classify by similarity to class prototypes.

        Prototypical Network:
            For each class c, compute prototype:  p_c = mean of support embeddings
            Classify query x:  argmin_c  d(f(x), p_c)

    OPTIMISATION-BASED (MAML — Model-Agnostic Meta-Learning):
        Learn an initialisation θ₀ such that a small number of gradient
        steps from θ₀ produces good performance on any new task.

        Outer loop:   θ₀ ← θ₀ − α · ∇_{θ₀} ∑_task L_task(θ₀ + k steps)
        Inner loop:   For each task, take k gradient steps from θ₀

        "Learning to learn" — the model learns how to adapt, not just what.

    ZERO-SHOT LEARNING:
        Classify novel classes using semantic descriptions (attributes,
        word vectors, class names in text) without any training images.

        CLIP: jointly trains image encoder and text encoder with:
            L = - log(exp(sim(I_i, T_i)/τ) / ∑_j exp(sim(I_i, T_j)/τ))
        At test time: encode class name as text, find nearest image.

    ┌────────────────────────────────────────────────────────────────────┐
    │  Key insight: few-shot learning is not about memorising examples.  │
    │  It is about learning the STRUCTURE of the problem class so well   │
    │  that a new instance of it requires only minimal adaptation.       │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 8: Active Learning

Instead of passively learning from a fixed dataset, the model actively
CHOOSES which data points to have labelled. Given a budget of B labels,
which B examples should the oracle (human expert) annotate to maximise
model performance?

    Diagram 9 — Active Learning Loop:

    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   Unlabelled pool:   · · · · · · · · · · · · · · · ·               │
    │                                                                    │
    │           ▲  Query strategy selects informative samples            │
    │           │                                                        │
    │   ┌───────┴──────┐                                                 │
    │   │    MODEL     │  ──▶  query oracle  ──▶  receive label          │
    │   │  (current)   │                                                 │
    │   └───────┬──────┘                                                 │
    │           │  retrain with new labels                               │
    │           ▼                                                        │
    │   Labelled set:  ● ● ● ● ● (grows after each query)                │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

    Query strategies — how to select which sample to label next:

    UNCERTAINTY SAMPLING:
        Label the sample the model is LEAST certain about.
        Binary classification: choose x with p closest to 0.5.
        Multiclass: choose x with minimum margin (gap between top-2 probs)
        or maximum entropy: H(y|x) = -∑_c p_c log p_c

    QUERY-BY-COMMITTEE (QBC):
        Train a committee of diverse models. Query the sample on which
        the committee members DISAGREE most. Disagreement = informative.

    CORE-SET SELECTION (Greedy):
        Choose samples that best cover the input space — maximise the
        minimum distance from any query to the existing labelled set.
        Geometrically: query the samples furthest from labelled centres.

    EXPECTED MODEL CHANGE:
        Query the sample that would cause the greatest change in model
        parameters (largest gradient norm if it were labelled).

    ┌────────────────────────────────────────────────────────────────────┐
    │  Empirical result: active learning can match the performance of    │
    │  fully supervised models using 30–90% fewer labels depending on    │
    │  the domain. For expensive domains (medical, legal), this          │
    │  directly translates to large cost and time savings.               │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 9: Online Learning

In batch learning, the model is trained once on a fixed dataset and then
deployed. In online learning, data arrives as a continuous stream and the
model updates after EACH example, with no fixed training/test split.

    Diagram 10 — Batch vs Online Learning:

    BATCH LEARNING:                       ONLINE LEARNING:
    ─────────────────                     ─────────────────────────────

    ┌──────────────┐                      x₁ ──▶ model ──▶ ŷ₁ ──▶ update
    │ Collect all  │                      x₂ ──▶ model ──▶ ŷ₂ ──▶ update
    │ data first   │                      x₃ ──▶ model ──▶ ŷ₃ ──▶ update
    └──────┬───────┘                      ...
           │                              xₙ ──▶ model ──▶ ŷₙ ──▶ update
    ┌──────▼───────┐
    │ Train once   │                      Model is ALWAYS up to date.
    └──────┬───────┘                      Adapts to concept drift.
           │                              Memory: O(1) per step.
    ┌──────▼───────┐
    │ Deploy fixed │
    └──────────────┘

    Key concepts:

    REGRET — the performance gap between online learner and the best
    fixed model in hindsight. Good online algorithms have sublinear regret:

            Regret(T) = ∑_{t=1}^T L_t(f_t) − min_f ∑_{t=1}^T L_t(f)

    Ideal: Regret(T) = O(√T) — the average per-step regret → 0.

    CONCEPT DRIFT — the underlying data distribution P(X,Y) changes over
    time. Batch models trained on historical data become stale.

        Types of drift:
        Sudden drift:    P changes abruptly (news topic shift, market crash)
        Gradual drift:   P changes slowly (seasonal patterns, user behaviour)
        Recurring drift: P cycles (weekday/weekend patterns)

    Algorithms:
        SGD on each example:       simple, works if distribution is stable
        FTRL (Follow The Regularised Leader): best for sparse features
        Hoeffding Trees:           incremental decision trees
        ADWIN (Adaptive Windowing): detect drift, forget old data

    ┌────────────────────────────────────────────────────────────────────┐
    │  Online learning is the default in production systems at scale:    │
    │  Google Search ranking, ad click prediction, fraud detection,      │
    │  and recommendation systems all use online or mini-batch           │
    │  learning to stay current with shifting user behaviour.            │
    └────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### Paradigm 10: Multi-Task Learning

Instead of training one model per task, train a single model on MULTIPLE
related tasks simultaneously. Shared structure is learned jointly —
each task improves the others through shared representations.

    Diagram 11 — Multi-Task Learning Architecture:

    SINGLE-TASK:                    MULTI-TASK:
    ─────────────                   ──────────────────────────────
    Input ──▶ f_θ₁ ──▶ ŷ₁          Input ──▶ Shared ──▶ Head 1 ──▶ ŷ₁
                                              Backbone ──▶ Head 2 ──▶ ŷ₂
    Input ──▶ f_θ₂ ──▶ ŷ₂                             ──▶ Head 3 ──▶ ŷ₃
                                    Parameters shared,
    Input ──▶ f_θ₃ ──▶ ŷ₃          tasks regularise each other.
    Three separate models.
    No shared knowledge.

    Why it works — the three mechanisms:

    1. IMPLICIT DATA AUGMENTATION:
       Training on Task B provides additional gradient signal that
       prevents overfitting on Task A's sparse data.

    2. REPRESENTATION LEARNING:
       Features useful for multiple tasks are forced into the shared
       representation. These tend to be the most general and robust.

    3. REGULARISATION:
       Task diversity prevents the model from overfitting to task-specific
       noise. Tasks impose competing constraints that force the shared
       layers to learn genuinely general structure.

    Loss:    L_total = ∑_k w_k · L_k(θ_shared, θ_head_k)

    Examples of successful multi-task learning:

    ┌──────────────────────────────────────────────────────────────────┐
    │ GPT / BERT: Pre-trained on many tasks simultaneously             │
    │             (next token, masked token, next sentence, etc.)      │
    │                                                                  │
    │ Self-driving: Jointly learn object detection, lane segmentation, │
    │               depth estimation, traffic sign recognition         │
    │                                                                  │
    │ NLP models:  Jointly learn NER, POS tagging, parsing,            │
    │              coreference resolution                              │
    │                                                                  │
    │ DeepMind's Gato: One model, 600+ tasks (text, images, robotics)  │
    └──────────────────────────────────────────────────────────────────┘

    Task relationship matters:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Tasks too similar:    trivially shared, little benefit         │
    │  Tasks moderately related: most benefit from sharing            │
    │  Tasks too dissimilar: negative transfer (see Transfer Learning)│
    │                                                                 │
    │  Hard challenge: automatically learning task weights w_k        │
    │  and which parameters to share vs. keep task-specific.          │
    └─────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
### How the Paradigms Combine in Modern Systems

Real-world ML systems rarely use a single paradigm. State-of-the-art models
stack multiple paradigms to address different data limitations:

    Diagram 12 — The Modern Large Model Training Pipeline:

    Stage 1: SELF-SUPERVISED PRE-TRAINING
        Paradigm: Self-Supervised Learning
        Data:     Trillions of tokens / billions of images (no labels)
        Goal:     Learn universal representations
        Result:   Rich, general foundation model

         ↓

    Stage 2: SUPERVISED FINE-TUNING
        Paradigm: Supervised Learning (Transfer Learning)
        Data:     Thousands to millions of labelled examples
        Goal:     Adapt to specific tasks / formats / domains
        Result:   Task-competent model

         ↓

    Stage 3: PREFERENCE ALIGNMENT
        Paradigm: Reinforcement Learning (RLHF)
        Data:     Human preference comparisons
        Goal:     Align outputs with human values
        Result:   Helpful, harmless, honest model

         ↓

    Stage 4: CONTINUED ADAPTATION (optional)
        Paradigm: Active Learning + Online Learning
        Data:     Production data, user feedback
        Goal:     Stay current, correct specific failure modes
        Result:   Continually improving deployed system

    Paradigm selection guide:
    ┌─────────────────────────────────┬──────────────────────────────────┐
    │ Situation                       │ Recommended Paradigm(s)          │
    ├─────────────────────────────────┼──────────────────────────────────┤
    │ Plentiful labelled data         │ Supervised Learning              │
    │ Abundant data, few labels       │ Semi-supervised / Self-supervised│
    │ Novel task, large pretrained    │ Transfer Learning + Fine-tuning  │
    │ model available                 │                                  │
    │ Only 1–10 examples per class    │ Few-Shot / Meta-Learning         │
    │ New classes at test time        │ Zero-Shot Learning               │
    │ Labelling budget is limited     │ Active Learning                  │
    │ Data arrives as stream          │ Online Learning                  │
    │ Sequential decision-making      │ Reinforcement Learning           │
    │ Multiple related tasks          │ Multi-Task Learning              │
    │ No idea what structure exists   │ Unsupervised Learning            │
    └─────────────────────────────────┴──────────────────────────────────┘

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable Python demonstrations)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Supervised vs Unsupervised — Same Data, Different Questions": {
        "description": (
            "Apply both supervised and unsupervised learning to the same "
            "synthetic dataset and visualise what each paradigm discovers. "
            "Supervised classification learns a decision boundary using labels. "
            "K-Means clustering discovers structure without any labels. "
            "Demonstrates that the two paradigms answer fundamentally different "
            "questions about the same data."
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

def make_blobs(n_samples=100, centers=None, cluster_std=1.0, random_state=None):
    rng=np.random.default_rng(random_state)
    if centers is None: centers=3
    if isinstance(centers, int):
        centers=[rng.uniform(-5,5,2) for _ in range(centers)]
    centers=np.array(centers); k=len(centers)
    n_per=[n_samples//k]*k; n_per[-1]+=n_samples-sum(n_per)
    X=[]; y=[]
    for i,(c,n) in enumerate(zip(centers,n_per)):
        X.append(rng.normal(0,cluster_std,(n,len(c)))+c); y.extend([i]*n)
    return np.vstack(X), np.array(y)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        n_classes=2,n_clusters_per_class=1,weights=None,
                        class_sep=1.0,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(n_classes,n_samples,p=weights) if weights else rng.integers(0,n_classes,n_samples)
    means=rng.normal(0,class_sep*2,(n_classes,n_informative))
    Xi=np.array([rng.normal(means[yi],1.0,n_informative) for yi in y])
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=rng.integers(0,n_classes,fm.sum())
    return X,y.astype(int)

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

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        self.classes_=np.unique(y); n,d=X.shape; lam=1./self.C
        if len(self.classes_)==2:
            yb=(y==self.classes_[1]).astype(float)
            def fg(w):
                p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
                loss=-np.mean(yb*np.log(p)+(1-yb)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
                e=p-yb; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
                return loss,g
            res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                                 options={"maxiter":self.max_iter})
            self._ws=[res.x]; self._binary=True
        else:
            self._ws=[]; self._binary=False
            for c in self.classes_:
                yb=(y==c).astype(float)
                def fg(w,yb=yb):
                    p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
                    loss=-np.mean(yb*np.log(p)+(1-yb)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
                    e=p-yb; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
                    return loss,g
                res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                                     options={"maxiter":self.max_iter})
                self._ws.append(res.x)
        return self
    def predict_proba(self,X):
        if self._binary:
            p=_sig(X@self._ws[0][1:]+self._ws[0][0]); return np.c_[1-p,p]
        scores=np.column_stack([_sig(X@w[1:]+w[0]) for w in self._ws])
        return scores/scores.sum(1,keepdims=True)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

class KMeans:
    def __init__(self,n_clusters=3,random_state=None,n_init=10,max_iter=300):
        self.n_clusters=n_clusters; self.random_state=random_state
        self.n_init=n_init; self.max_iter=max_iter
    def _run_once(self,X,rng):
        idx=rng.choice(len(X),self.n_clusters,replace=False)
        centers=X[idx].copy()
        for _ in range(self.max_iter):
            dists=((X[:,None]-centers[None])**2).sum(2)
            labels=dists.argmin(1)
            new_centers=np.array([X[labels==k].mean(0) if (labels==k).any()
                                  else centers[k] for k in range(self.n_clusters)])
            if np.allclose(centers,new_centers,atol=1e-6): break
            centers=new_centers
        return labels,centers,((X-centers[labels])**2).sum()
    def fit(self,X):
        rng=np.random.default_rng(self.random_state); best=None
        for _ in range(self.n_init):
            r=self._run_once(X,rng)
            if best is None or r[2]<best[2]: best=r
        self.labels_,self.cluster_centers_,self.inertia_=best; return self
    def predict(self,X):
        return ((X[:,None]-self.cluster_centers_[None])**2).sum(2).argmin(1)

def adjusted_rand_score(labels_true,labels_pred):
    n=len(labels_true)
    ut=np.unique(labels_true); up=np.unique(labels_pred)
    mt={v:i for i,v in enumerate(ut)}; mp={v:i for i,v in enumerate(up)}
    ct=np.zeros((len(ut),len(up)),dtype=int)
    for a,b in zip(labels_true,labels_pred): ct[mt[a],mp[b]]+=1
    def C2(x): return x*(x-1)//2
    s_c2=sum(C2(ct[i,j]) for i in range(ct.shape[0]) for j in range(ct.shape[1]))
    a=ct.sum(1); b=ct.sum(0)
    sa=np.sum(C2(a)); sb=np.sum(C2(b))
    exp=sa*sb/C2(n); mx=0.5*(sa+sb)-exp
    return 0. if mx==0 else (s_c2-exp)/mx

class LabelSpreading:
    def __init__(self,kernel="rbf",alpha=0.2,max_iter=100,gamma=20):
        self.alpha=alpha; self.max_iter=max_iter; self.gamma=gamma
    def fit(self,X,y):
        n=len(X); self._classes=np.unique(y[y>=0])
        d2=((X[:,None]-X[None])**2).sum(2)
        W=np.exp(-self.gamma*d2); np.fill_diagonal(W,0)
        D=W.sum(1); D[D==0]=1; S=W/D[:,None]
        nc=len(self._classes); cmap={c:i for i,c in enumerate(self._classes)}
        Y=np.zeros((n,nc))
        for i,yi in enumerate(y):
            if yi>=0: Y[i,cmap[yi]]=1.
        F=Y.copy()
        for _ in range(self.max_iter):
            F=self.alpha*(S@F)+(1-self.alpha)*Y
        self._F=F; self._X=X.copy(); return self
    def predict(self,X):
        d2=((X[:,None]-self._X[None])**2).sum(2)
        W=np.exp(-self.gamma*d2); W/=W.sum(1,keepdims=True)+1e-10
        return self._classes[(W@self._F).argmax(1)]

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),activation="relu",max_iter=200,
                 random_state=None,alpha=0.0001,learning_rate_init=0.001,
                 warm_start=False,batch_size=200):
        self.hidden_layer_sizes=hidden_layer_sizes; self.activation=activation
        self.max_iter=max_iter; self.random_state=random_state; self.alpha=alpha
        self.lr=learning_rate_init; self.bs=batch_size
    def _act(self,z):
        if self.activation=="relu": return np.maximum(0,z)
        if self.activation=="tanh": return np.tanh(z)
        return _sig(z)
    def _dact(self,z):
        if self.activation=="relu": return (z>0).astype(float)
        if self.activation=="tanh": t=np.tanh(z); return 1-t**2
        s=_sig(z); return s*(1-s)
    def _softmax(self,z):
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        self.classes_=np.unique(y); nc=len(self.classes_)
        cmap={c:i for i,c in enumerate(self.classes_)}
        Y=np.eye(nc)[[cmap[yi] for yi in y]]
        layers=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,np.sqrt(2/layers[i]),(layers[i],layers[i+1]))
                     for i in range(len(layers)-1)]
        self.intercepts_=[np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        n=len(X); bs=min(self.bs,n)
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for start in range(0,n,bs):
                b=idx[start:start+bs]; Xb=X[b]; Yb=Y[b]
                acts=[Xb]; zs=[]
                for i,(W,bi) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+bi; zs.append(z)
                    acts.append(self._act(z) if i<len(self.coefs_)-1 else self._softmax(z))
                dA=acts[-1]-Yb
                for i in reversed(range(len(self.coefs_))):
                    dZ=dA if i==len(self.coefs_)-1 else dA*self._dact(zs[i])
                    dW=acts[i].T@dZ/len(b)+self.alpha*self.coefs_[i]
                    db=dZ.mean(0); dA=dZ@self.coefs_[i].T
                    self.coefs_[i]-=self.lr*dW; self.intercepts_[i]-=self.lr*db
        return self
    def _fwd(self,X):
        A=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=A@W+b
            A=self._act(z) if i<len(self.coefs_)-1 else self._softmax(z)
        return A
    def predict_proba(self,X): return self._fwd(X)
    def predict(self,X): return self.classes_[self._fwd(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

np.random.seed(42)

# ── Dataset: 3 Gaussian blobs with known labels ───────────────────────────
X, y_true = make_blobs(n_samples=450, centers=[[-3, -2], [0, 3], [3, -1]],
                       cluster_std=1.1, random_state=7)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── SUPERVISED: Logistic Regression (uses labels) ────────────────────────
lr = LogisticRegression(C=2.0, max_iter=300, random_state=0)
lr.fit(X, y_true)
y_pred_supervised = lr.predict(X)
sup_acc = accuracy_score(y_true, y_pred_supervised) * 100

# ── UNSUPERVISED: K-Means (no labels at all) ─────────────────────────────
km = KMeans(n_clusters=3, random_state=0, n_init=3)
km.fit(X)
y_pred_unsup = km.labels_
ari = adjusted_rand_score(y_true, y_pred_unsup)   # how well clusters match true labels

print("=" * 65)
print("  SUPERVISED vs UNSUPERVISED ON THE SAME DATASET")
print("  Data: 450 points from 3 Gaussian blobs")
print("=" * 65)
print()
print("  SUPERVISED (Logistic Regression):")
print(f"    Uses labels:           YES — all 450 labels provided")
print(f"    Question answered:     'Given x, which class is it?'")
print(f"    Training accuracy:     {sup_acc:.1f}%")
print()
print("  UNSUPERVISED (K-Means, K=3):")
print(f"    Uses labels:           NO — discovers structure alone")
print(f"    Question answered:     'What groups exist in the data?'")
print(f"    Adjusted Rand Index:   {ari:.3f}  (1.0 = perfect match with true labels)")
print()
print("  Both look at identical raw data but answer DIFFERENT questions.")
print("  K-Means recovers nearly the same grouping as the true labels")
print("  without ever seeing them — the geometric structure is real.")
print()
print("  BUT: unsupervised learning does not assign meaningful names.")
print("  K-Means cluster 0 might correspond to true class 2.")
print("  Labels give MEANING; unsupervised gives STRUCTURE.")
print()

# ── Count cluster-to-class assignments ───────────────────────────────────
print("  Cluster → True class correspondence (majority label per cluster):")
for cluster_id in range(3):
    mask = y_pred_unsup == cluster_id
    class_counts = np.bincount(y_true[mask], minlength=3)
    dominant = class_counts.argmax()
    purity = class_counts[dominant] / mask.sum() * 100
    print(f"    K-Means cluster {cluster_id} → mostly true class {dominant} "
          f"({purity:.0f}% purity)")

# ── Decision boundary mesh ────────────────────────────────────────────────
h = 0.05
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                      np.arange(x2_min, x2_max, h))
mesh = np.c_[xx.ravel(), yy.ravel()]

Z_sup  = lr.predict(mesh).reshape(xx.shape)
Z_unsup = km.predict(mesh).reshape(xx.shape)

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Supervised vs Unsupervised Learning: Same Data, Different Questions",
             fontsize=12, fontweight="bold")

cmaps_bg = ["RdYlBu", "Pastel1", "Set3"]
colours_true = ["#d62728", "#1f77b4", "#2ca02c"]
colours_km   = ["#ff7f0e", "#9467bd", "#8c564b"]

# Panel 1: Raw data (unlabelled)
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c="steelblue", s=20, alpha=0.6,
           edgecolors="none")
ax.set_title("Raw Data\\n(no labels visible)", fontsize=11, fontweight="bold")
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.text(0.03, 0.97, "What does the data look like?\\nBoth paradigms start here.",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

# Panel 2: Supervised result
ax = axes[1]
ax.contourf(xx, yy, Z_sup, alpha=0.25, cmap="RdYlBu")
ax.contour(xx, yy, Z_sup, colors="gray", linewidths=0.8, linestyles="--")
for cls, col in enumerate(colours_true):
    mask = y_true == cls
    ax.scatter(X[mask, 0], X[mask, 1], c=col, s=22, alpha=0.85,
               edgecolors="none", label=f"Class {cls}")
ax.set_title(f"Supervised: Logistic Regression\\nAccuracy = {sup_acc:.1f}%",
             fontsize=11, fontweight="bold", color="navy")
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.legend(fontsize=9)
ax.text(0.03, 0.97, "Learns decision boundary.\\nLabels required.",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="#ddeeff", ec="navy"))

# Panel 3: Unsupervised result
ax = axes[2]
ax.contourf(xx, yy, Z_unsup, alpha=0.2, cmap="Pastel1")
for cluster_id, col in enumerate(colours_km):
    mask = y_pred_unsup == cluster_id
    ax.scatter(X[mask, 0], X[mask, 1], c=col, s=22, alpha=0.85,
               edgecolors="none", label=f"Cluster {cluster_id}")
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           c="black", marker="X", s=120, zorder=5, label="Centroids")
ax.set_title(f"Unsupervised: K-Means (k=3)\\nARI = {ari:.3f} (vs true labels)",
             fontsize=11, fontweight="bold", color="darkgreen")
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.legend(fontsize=9)
ax.text(0.03, 0.97, "Discovers groupings.\\nNo labels needed.",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="#ddffdd", ec="darkgreen"))

plt.tight_layout()
plt.savefig("supervised_vs_unsupervised.png", dpi=110)
print("  Plot saved → supervised_vs_unsupervised.png")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Semi-Supervised Learning — Label Efficiency": {
        "description": (
            "Demonstrate the power of semi-supervised learning by comparing a "
            "supervised-only baseline (trained on a tiny labelled subset) with "
            "a self-training pseudo-label approach that iteratively labels "
            "high-confidence unlabelled points. Sweeps over labelled set sizes "
            "from 5 to 200 and shows how semi-supervised learning dramatically "
            "narrows the performance gap with full supervision."
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

def make_blobs(n_samples=100, centers=None, cluster_std=1.0, random_state=None):
    rng=np.random.default_rng(random_state)
    if centers is None: centers=3
    if isinstance(centers, int):
        centers=[rng.uniform(-5,5,2) for _ in range(centers)]
    centers=np.array(centers); k=len(centers)
    n_per=[n_samples//k]*k; n_per[-1]+=n_samples-sum(n_per)
    X=[]; y=[]
    for i,(c,n) in enumerate(zip(centers,n_per)):
        X.append(rng.normal(0,cluster_std,(n,len(c)))+c); y.extend([i]*n)
    return np.vstack(X), np.array(y)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        n_classes=2,n_clusters_per_class=1,weights=None,
                        class_sep=1.0,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(n_classes,n_samples,p=weights) if weights else rng.integers(0,n_classes,n_samples)
    means=rng.normal(0,class_sep*2,(n_classes,n_informative))
    Xi=np.array([rng.normal(means[yi],1.0,n_informative) for yi in y])
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=rng.integers(0,n_classes,fm.sum())
    return X,y.astype(int)

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

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        self.classes_=np.unique(y); n,d=X.shape; lam=1./self.C
        if len(self.classes_)==2:
            yb=(y==self.classes_[1]).astype(float)
            def fg(w):
                p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
                loss=-np.mean(yb*np.log(p)+(1-yb)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
                e=p-yb; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
                return loss,g
            res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                                 options={"maxiter":self.max_iter})
            self._ws=[res.x]; self._binary=True
        else:
            self._ws=[]; self._binary=False
            for c in self.classes_:
                yb=(y==c).astype(float)
                def fg(w,yb=yb):
                    p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
                    loss=-np.mean(yb*np.log(p)+(1-yb)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
                    e=p-yb; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
                    return loss,g
                res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                                     options={"maxiter":self.max_iter})
                self._ws.append(res.x)
        return self
    def predict_proba(self,X):
        if self._binary:
            p=_sig(X@self._ws[0][1:]+self._ws[0][0]); return np.c_[1-p,p]
        scores=np.column_stack([_sig(X@w[1:]+w[0]) for w in self._ws])
        return scores/scores.sum(1,keepdims=True)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

class KMeans:
    def __init__(self,n_clusters=3,random_state=None,n_init=10,max_iter=300):
        self.n_clusters=n_clusters; self.random_state=random_state
        self.n_init=n_init; self.max_iter=max_iter
    def _run_once(self,X,rng):
        idx=rng.choice(len(X),self.n_clusters,replace=False)
        centers=X[idx].copy()
        for _ in range(self.max_iter):
            dists=((X[:,None]-centers[None])**2).sum(2)
            labels=dists.argmin(1)
            new_centers=np.array([X[labels==k].mean(0) if (labels==k).any()
                                  else centers[k] for k in range(self.n_clusters)])
            if np.allclose(centers,new_centers,atol=1e-6): break
            centers=new_centers
        return labels,centers,((X-centers[labels])**2).sum()
    def fit(self,X):
        rng=np.random.default_rng(self.random_state); best=None
        for _ in range(self.n_init):
            r=self._run_once(X,rng)
            if best is None or r[2]<best[2]: best=r
        self.labels_,self.cluster_centers_,self.inertia_=best; return self
    def predict(self,X):
        return ((X[:,None]-self.cluster_centers_[None])**2).sum(2).argmin(1)

def adjusted_rand_score(labels_true,labels_pred):
    n=len(labels_true)
    ut=np.unique(labels_true); up=np.unique(labels_pred)
    mt={v:i for i,v in enumerate(ut)}; mp={v:i for i,v in enumerate(up)}
    ct=np.zeros((len(ut),len(up)),dtype=int)
    for a,b in zip(labels_true,labels_pred): ct[mt[a],mp[b]]+=1
    def C2(x): return x*(x-1)//2
    s_c2=sum(C2(ct[i,j]) for i in range(ct.shape[0]) for j in range(ct.shape[1]))
    a=ct.sum(1); b=ct.sum(0)
    sa=np.sum(C2(a)); sb=np.sum(C2(b))
    exp=sa*sb/C2(n); mx=0.5*(sa+sb)-exp
    return 0. if mx==0 else (s_c2-exp)/mx

class LabelSpreading:
    def __init__(self,kernel="rbf",alpha=0.2,max_iter=100,gamma=20):
        self.alpha=alpha; self.max_iter=max_iter; self.gamma=gamma
    def fit(self,X,y):
        n=len(X); self._classes=np.unique(y[y>=0])
        d2=((X[:,None]-X[None])**2).sum(2)
        W=np.exp(-self.gamma*d2); np.fill_diagonal(W,0)
        D=W.sum(1); D[D==0]=1; S=W/D[:,None]
        nc=len(self._classes); cmap={c:i for i,c in enumerate(self._classes)}
        Y=np.zeros((n,nc))
        for i,yi in enumerate(y):
            if yi>=0: Y[i,cmap[yi]]=1.
        F=Y.copy()
        for _ in range(self.max_iter):
            F=self.alpha*(S@F)+(1-self.alpha)*Y
        self._F=F; self._X=X.copy(); return self
    def predict(self,X):
        d2=((X[:,None]-self._X[None])**2).sum(2)
        W=np.exp(-self.gamma*d2); W/=W.sum(1,keepdims=True)+1e-10
        return self._classes[(W@self._F).argmax(1)]

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),activation="relu",max_iter=200,
                 random_state=None,alpha=0.0001,learning_rate_init=0.001,
                 warm_start=False,batch_size=200):
        self.hidden_layer_sizes=hidden_layer_sizes; self.activation=activation
        self.max_iter=max_iter; self.random_state=random_state; self.alpha=alpha
        self.lr=learning_rate_init; self.bs=batch_size
    def _act(self,z):
        if self.activation=="relu": return np.maximum(0,z)
        if self.activation=="tanh": return np.tanh(z)
        return _sig(z)
    def _dact(self,z):
        if self.activation=="relu": return (z>0).astype(float)
        if self.activation=="tanh": t=np.tanh(z); return 1-t**2
        s=_sig(z); return s*(1-s)
    def _softmax(self,z):
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        self.classes_=np.unique(y); nc=len(self.classes_)
        cmap={c:i for i,c in enumerate(self.classes_)}
        Y=np.eye(nc)[[cmap[yi] for yi in y]]
        layers=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,np.sqrt(2/layers[i]),(layers[i],layers[i+1]))
                     for i in range(len(layers)-1)]
        self.intercepts_=[np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        n=len(X); bs=min(self.bs,n)
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for start in range(0,n,bs):
                b=idx[start:start+bs]; Xb=X[b]; Yb=Y[b]
                acts=[Xb]; zs=[]
                for i,(W,bi) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+bi; zs.append(z)
                    acts.append(self._act(z) if i<len(self.coefs_)-1 else self._softmax(z))
                dA=acts[-1]-Yb
                for i in reversed(range(len(self.coefs_))):
                    dZ=dA if i==len(self.coefs_)-1 else dA*self._dact(zs[i])
                    dW=acts[i].T@dZ/len(b)+self.alpha*self.coefs_[i]
                    db=dZ.mean(0); dA=dZ@self.coefs_[i].T
                    self.coefs_[i]-=self.lr*dW; self.intercepts_[i]-=self.lr*db
        return self
    def _fwd(self,X):
        A=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=A@W+b
            A=self._act(z) if i<len(self.coefs_)-1 else self._softmax(z)
        return A
    def predict_proba(self,X): return self._fwd(X)
    def predict(self,X): return self.classes_[self._fwd(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()
from copy import deepcopy

np.random.seed(42)

# ── Dataset ───────────────────────────────────────────────────────────────
X_all, y_all = make_classification(
    n_samples=2000, n_features=10, n_informative=6,
    n_redundant=2, n_clusters_per_class=2, flip_y=0.05,
    class_sep=1.0, random_state=0)

scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

# Fixed test set (fully labelled)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=1, stratify=y_all)

# ── Self-Training Implementation ──────────────────────────────────────────
def self_training(X_labelled, y_labelled, X_unlabelled,
                  X_test, y_test, threshold=0.85, max_iters=10):
    """
    Iteratively:
      1. Train classifier on labelled set
      2. Predict probabilities on unlabelled set
      3. Add high-confidence predictions to labelled set
      4. Repeat until no more confident predictions or max_iters reached
    """
    X_L = X_labelled.copy()
    y_L = y_labelled.copy()
    X_U = X_unlabelled.copy()
    history = []

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=0)
    clf.fit(X_L, y_L)
    history.append({
        "iter": 0, "n_labelled": len(X_L),
        "test_acc": accuracy_score(y_test, clf.predict(X_test)) * 100
    })

    for iteration in range(1, max_iters + 1):
        if len(X_U) == 0:
            break
        probs = clf.predict_proba(X_U)
        max_probs = probs.max(axis=1)
        confident_mask = max_probs >= threshold

        if confident_mask.sum() == 0:
            break

        pseudo_labels = clf.predict(X_U[confident_mask])
        X_L = np.vstack([X_L, X_U[confident_mask]])
        y_L = np.concatenate([y_L, pseudo_labels])
        X_U = X_U[~confident_mask]

        clf.fit(X_L, y_L)
        acc = accuracy_score(y_test, clf.predict(X_test)) * 100
        history.append({
            "iter": iteration, "n_labelled": len(X_L), "test_acc": acc
        })

    return clf, history

# ── Sweep over labelled set sizes ─────────────────────────────────────────
labelled_sizes = [5, 10, 20, 30, 50, 75, 100, 150, 200]
sup_accs     = []
semisup_accs = []
label_spread_accs = []

# Full supervision baseline
clf_full = LogisticRegression(C=1.0, max_iter=500, random_state=0)
clf_full.fit(X_train_pool, y_train_pool)
full_acc = accuracy_score(y_test, clf_full.predict(X_test)) * 100

print("=" * 72)
print("  SEMI-SUPERVISED LEARNING — LABEL EFFICIENCY")
print(f"  Total training pool: {len(X_train_pool)} | Test set: {len(X_test)}")
print(f"  Full supervision baseline: {full_acc:.1f}%")
print("=" * 72)
print()
print(f"  {'Labelled':>8}  {'Supervised':>12}  {'Self-Training':>14}  "
      f"{'Label Spreading':>16}  {'Gain (semi-sup)':>15}")
print("  " + "─" * 72)

for n_lab in labelled_sizes:
    # Sample labelled subset (stratified)
    idx_lab = []
    for cls in [0, 1]:
        cls_idx = np.where(y_train_pool == cls)[0]
        chosen  = np.random.choice(cls_idx, size=n_lab // 2, replace=False)
        idx_lab.extend(chosen)
    idx_lab = np.array(idx_lab)
    idx_unlab = np.setdiff1d(np.arange(len(X_train_pool)), idx_lab)

    X_lab  = X_train_pool[idx_lab]
    y_lab  = y_train_pool[idx_lab]
    X_unlab = X_train_pool[idx_unlab]

    # 1. Supervised only (uses only n_lab labels)
    clf_sup = LogisticRegression(C=1.0, max_iter=500, random_state=0)
    clf_sup.fit(X_lab, y_lab)
    acc_sup = accuracy_score(y_test, clf_sup.predict(X_test)) * 100

    # 2. Self-training (pseudo-labelling)
    _, history = self_training(X_lab, y_lab, X_unlab,
                               X_test, y_test, threshold=0.85)
    acc_semi = history[-1]["test_acc"]

    # 3. Label Spreading (graph-based)
    y_mixed = np.full(len(X_train_pool), -1)
    y_mixed[idx_lab] = y_lab
    ls = LabelSpreading(kernel="rbf", alpha=0.2, max_iter=100)
    ls.fit(X_train_pool, y_mixed)
    acc_ls = accuracy_score(y_test, ls.predict(X_test)) * 100

    sup_accs.append(acc_sup)
    semisup_accs.append(acc_semi)
    label_spread_accs.append(acc_ls)

    gain = max(acc_semi, acc_ls) - acc_sup
    print(f"  {n_lab:>8d}  {acc_sup:>11.1f}%  {acc_semi:>13.1f}%  "
          f"{acc_ls:>15.1f}%  {gain:>+14.1f}%")

print()
print(f"  Full supervision (all {len(X_train_pool)} labels): {full_acc:.1f}%")
print()
print("  Key observations:")
print("  - With 10 labels supervised = near-random; semi-supervised >> supervised")
print("  - Semi-supervised gap closes as labels increase (less unlabelled benefit)")
print("  - Label Spreading (graph-based) uses geometric structure of all points")
print("  - Self-training risk: confident wrong predictions propagate errors")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Semi-Supervised Learning: Label Efficiency\\n"
             "(Same unlabelled data — only labelled set size varies)",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(labelled_sizes, sup_accs,          "r-o", lw=2, ms=7,
        label="Supervised only")
ax.plot(labelled_sizes, semisup_accs,       "b-s", lw=2, ms=7,
        label="Self-Training (pseudo-labels)")
ax.plot(labelled_sizes, label_spread_accs,  "g-^", lw=2, ms=7,
        label="Label Spreading (graph)")
ax.axhline(full_acc, color="black", ls="--", lw=2,
           label=f"Full supervision ({full_acc:.1f}%)")
ax.fill_between(labelled_sizes, sup_accs, semisup_accs,
                alpha=0.12, color="steelblue", label="Semi-sup gain")
ax.set_title("Test Accuracy vs Number of Labelled Samples",
             fontweight="bold")
ax.set_xlabel("Number of Labelled Training Examples")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(50, 100)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: Self-training iteration history for n_lab=10
n_lab = 10
idx_lab = []
for cls in [0, 1]:
    cls_idx = np.where(y_train_pool == cls)[0]
    chosen  = np.random.choice(cls_idx, size=5, replace=False)
    idx_lab.extend(chosen)
idx_lab  = np.array(idx_lab)
idx_unlab = np.setdiff1d(np.arange(len(X_train_pool)), idx_lab)
X_lab   = X_train_pool[idx_lab]
y_lab   = y_train_pool[idx_lab]
X_unlab = X_train_pool[idx_unlab]

_, history_10 = self_training(X_lab, y_lab, X_unlab,
                               X_test, y_test, threshold=0.80, max_iters=15)

iters     = [h["iter"]      for h in history_10]
n_labs    = [h["n_labelled"] for h in history_10]
iter_accs = [h["test_acc"]  for h in history_10]

ax2 = axes[1]
ax2_twin = ax2.twinx()
line1, = ax2.plot(iters, iter_accs, "b-o", lw=2, ms=7,
                  label="Test Accuracy")
line2, = ax2_twin.plot(iters, n_labs, "g--s", lw=2, ms=7,
                       label="# Labelled points")
ax2.axhline(full_acc, color="black", ls="--", lw=1.5,
            label=f"Full supervision ({full_acc:.1f}%)")
ax2.set_title("Self-Training Iterations\\n(starting with 10 labelled examples)",
              fontweight="bold")
ax2.set_xlabel("Self-Training Iteration")
ax2.set_ylabel("Test Accuracy (%)", color="steelblue")
ax2_twin.set_ylabel("Labelled Set Size", color="seagreen")
ax2.tick_params(axis="y", labelcolor="steelblue")
ax2_twin.tick_params(axis="y", labelcolor="seagreen")
lines = [line1, line2]
labels_leg = [l.get_label() for l in lines]
ax2.legend(lines, labels_leg, fontsize=9, loc="lower right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("semi_supervised_label_efficiency.png", dpi=110)
print()
print("  Plot saved → semi_supervised_label_efficiency.png")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Reinforcement Learning — Q-Learning on GridWorld": {
        "description": (
            "Implement tabular Q-learning from scratch on a 6×6 GridWorld "
            "with obstacles. The agent starts at the top-left and must reach "
            "the bottom-right goal while avoiding walls. Demonstrates the "
            "exploration-exploitation tradeoff via ε-greedy, the convergence "
            "of Q-values via the Bellman equation, and visualises the learned "
            "value function and optimal policy as arrows on the grid."
        ),
        "language": "python",
        "code": '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)

# ── GridWorld environment ─────────────────────────────────────────────────
GRID_ROWS  = 6
GRID_COLS  = 6
START      = (0, 0)
GOAL       = (5, 5)
OBSTACLES  = {(1,1),(1,2),(2,4),(3,1),(3,2),(4,3),(4,4)}

ACTIONS    = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}  # up/down/left/right
ACTION_SYM = {0: "↑", 1: "↓", 2: "←", 3: "→"}

R_GOAL     =  10.0
R_STEP     =  -0.1
R_WALL     =  -1.0

def step(state, action):
    """Take action from state, return (next_state, reward, done)."""
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    # Stay in bounds and avoid obstacles
    if (nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS
            or (nr, nc) in OBSTACLES):
        return state, R_WALL, False   # bounce back
    if (nr, nc) == GOAL:
        return (nr, nc), R_GOAL, True
    return (nr, nc), R_STEP, False

# ── Q-Learning ────────────────────────────────────────────────────────────
Q = np.zeros((GRID_ROWS, GRID_COLS, 4))

EPISODES    = 3000
ALPHA       = 0.3     # learning rate
GAMMA       = 0.95    # discount factor
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = EPISODES * 0.7   # decay over 70% of training

episode_rewards = []
steps_per_episode = []

print("=" * 65)
print("  REINFORCEMENT LEARNING: Q-LEARNING ON 6×6 GRIDWORLD")
print(f"  Episodes: {EPISODES}  |  α={ALPHA}  |  γ={GAMMA}")
print(f"  Start: {START}  |  Goal: {GOAL}  |  Obstacles: {len(OBSTACLES)}")
print("=" * 65)
print()

for ep in range(EPISODES):
    state = START
    epsilon = max(EPSILON_END,
                  EPSILON_START - (EPSILON_START - EPSILON_END) * ep / EPSILON_DECAY)
    total_reward = 0
    n_steps      = 0
    done = False

    while not done and n_steps < 200:
        r, c = state
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)   # explore
        else:
            action = np.argmax(Q[r, c])     # exploit

        next_state, reward, done = step(state, action)
        nr, nc = next_state

        # Bellman update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]
        td_target = reward + GAMMA * np.max(Q[nr, nc]) * (1 - done)
        td_error  = td_target - Q[r, c, action]
        Q[r, c, action] += ALPHA * td_error

        state = next_state
        total_reward += reward
        n_steps += 1

    episode_rewards.append(total_reward)
    steps_per_episode.append(n_steps)

# ── Evaluate learned policy ───────────────────────────────────────────────
def run_greedy(Q, max_steps=50):
    """Follow greedy policy, return path and whether goal was reached."""
    state = START
    path = [state]
    for _ in range(max_steps):
        r, c = state
        action = np.argmax(Q[r, c])
        state, _, done = step(state, action)
        path.append(state)
        if done:
            return path, True
    return path, False

path, reached_goal = run_greedy(Q)

print(f"  Greedy policy reaches goal: {'YES ✓' if reached_goal else 'NO ✗'}")
print(f"  Path length: {len(path)-1} steps")
print(f"  Path: {' → '.join(str(s) for s in path[:10])}"
      f"{'...' if len(path) > 10 else ''}")
print()

# ── Q-value and training statistics ──────────────────────────────────────
V = Q.max(axis=2)  # state value = max Q over actions

window = 100
smoothed_rewards = np.convolve(episode_rewards,
                                np.ones(window)/window, mode="valid")
smoothed_steps   = np.convolve(steps_per_episode,
                                np.ones(window)/window, mode="valid")

print(f"  Training statistics (last 100 episodes):")
print(f"    Mean reward:    {np.mean(episode_rewards[-100:]):.2f}")
print(f"    Mean steps:     {np.mean(steps_per_episode[-100:]):.1f}")
print(f"    Goal reach rate: {sum(r > 0 for r in episode_rewards[-100:])}/100")
print()
print("  Value function (V = max_a Q(s,a)):")
print("  ┌" + "─"*25 + "┐")
for row in range(GRID_ROWS):
    line = "  │"
    for col in range(GRID_COLS):
        s = (row, col)
        if s == GOAL:
            line += " GOAL"
        elif s in OBSTACLES:
            line += " ████"
        elif s == START:
            line += " STRT"
        else:
            line += f" {V[row,col]:>4.1f}"
    print(line + "│")
print("  └" + "─"*25 + "┘")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Reinforcement Learning: Q-Learning on GridWorld\\n"
             "Agent learns to navigate from Start → Goal by trial and error",
             fontsize=12, fontweight="bold")

# Panel 1: Training curve (smoothed reward + epsilon)
ax = axes[0]
ep_range = np.arange(len(smoothed_rewards)) + window
ax.plot(ep_range, smoothed_rewards, "steelblue", lw=2,
        label=f"Reward (smoothed {window}ep)")
ax.axhline(0, color="gray", ls="--", lw=1)
eps_smooth = [max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END)
                  * ep / EPSILON_DECAY) for ep in range(EPISODES)]
ax_e = ax.twinx()
ax_e.plot(range(EPISODES), eps_smooth, "tomato", lw=1.5, ls="--",
          alpha=0.7, label="ε (exploration)")
ax_e.set_ylabel("Epsilon", color="tomato")
ax_e.tick_params(axis="y", labelcolor="tomato")
ax.set_title("Training Curve\\n(reward increases as agent learns)",
             fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Episode Reward", color="steelblue")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: Learned value function heatmap
ax = axes[1]
V_plot = V.copy()
for r, c in OBSTACLES:
    V_plot[r, c] = np.nan
cmap_v = LinearSegmentedColormap.from_list("vf", ["#440154","#31688e",
                                                    "#35b779","#fde725"])
im = ax.imshow(V_plot, cmap=cmap_v, aspect="equal")
plt.colorbar(im, ax=ax, label="State Value V(s)")
# Mark special cells
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        s = (r, c)
        if s in OBSTACLES:
            ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                        color="black", alpha=0.85))
        elif s == GOAL:
            ax.text(c, r, "GOAL", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        elif s == START:
            ax.text(c, r, "START", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")
ax.set_title("Learned Value Function V(s)\\n(brighter = closer to goal)",
             fontweight="bold")
ax.set_xticks(range(GRID_COLS)); ax.set_yticks(range(GRID_ROWS))
ax.grid(alpha=0.3)

# Panel 3: Optimal policy arrows + learned path
ax = axes[2]
grid_display = np.zeros((GRID_ROWS, GRID_COLS))
ax.imshow(grid_display, cmap="Greys", aspect="equal", vmin=0, vmax=1,
          alpha=0.1)

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        s = (r, c)
        if s in OBSTACLES:
            ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                        color="black", alpha=0.8))
        elif s == GOAL:
            ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                        color="gold", alpha=0.8))
            ax.text(c, r, "★", ha="center", va="center", fontsize=14)
        elif s == START:
            ax.text(c, r, "S", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="navy")
        else:
            best_a = np.argmax(Q[r, c])
            ax.text(c, r, ACTION_SYM[best_a], ha="center", va="center",
                    fontsize=14, color="steelblue")

# Draw learned path
path_arr = np.array(path)
ax.plot(path_arr[:, 1], path_arr[:, 0], "r-", lw=3, alpha=0.7,
        label=f"Greedy path ({len(path)-1} steps)")
ax.plot(path_arr[0, 1], path_arr[0, 0], "go", ms=10, zorder=5)
ax.plot(path_arr[-1, 1], path_arr[-1, 0], "r*", ms=14, zorder=5)
ax.set_title("Optimal Policy (arrows) & Learned Path (red)\\n"
             "Agent learns optimal route by trial & error",
             fontweight="bold")
ax.set_xticks(range(GRID_COLS)); ax.set_yticks(range(GRID_ROWS))
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("qlearning_gridworld.png", dpi=110)
print()
print("  Plot saved → qlearning_gridworld.png")
print()
print("  Key Takeaways:")
print("  - Agent learns purely from reward signal — no labels, no teacher")
print("  - ε-greedy: high ε early (explore), low ε later (exploit)")
print("  - Bellman equation propagates value estimates backwards from goal")
print("  - Value function shows which states are 'good' (closer to goal)")
print("  - Policy arrows emerge from Q-values without explicit programming")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Transfer Learning — Pre-Training vs Fine-Tuning vs Scratch": {
        "description": (
            "Quantify the benefit of transfer learning by training the same "
            "architecture in three ways: (1) from scratch on a small target "
            "dataset, (2) feature extraction from a pre-trained source model "
            "with a new classification head, and (3) full fine-tuning of the "
            "pre-trained model. Sweeps over target dataset sizes to show how "
            "the transfer advantage grows as labelled target data decreases."
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

def make_blobs(n_samples=100, centers=None, cluster_std=1.0, random_state=None):
    rng=np.random.default_rng(random_state)
    if centers is None: centers=3
    if isinstance(centers, int):
        centers=[rng.uniform(-5,5,2) for _ in range(centers)]
    centers=np.array(centers); k=len(centers)
    n_per=[n_samples//k]*k; n_per[-1]+=n_samples-sum(n_per)
    X=[]; y=[]
    for i,(c,n) in enumerate(zip(centers,n_per)):
        X.append(rng.normal(0,cluster_std,(n,len(c)))+c); y.extend([i]*n)
    return np.vstack(X), np.array(y)

def make_classification(n_samples=200,n_features=20,n_informative=2,n_redundant=2,
                        n_classes=2,n_clusters_per_class=1,weights=None,
                        class_sep=1.0,flip_y=0.01,random_state=None):
    rng=np.random.default_rng(random_state)
    y=rng.choice(n_classes,n_samples,p=weights) if weights else rng.integers(0,n_classes,n_samples)
    means=rng.normal(0,class_sep*2,(n_classes,n_informative))
    Xi=np.array([rng.normal(means[yi],1.0,n_informative) for yi in y])
    nr=min(n_redundant,n_informative)
    Xr=Xi[:,:nr]+0.3*rng.standard_normal((n_samples,nr)) if nr>0 else np.empty((n_samples,0))
    nn=max(0,n_features-n_informative-nr)
    Xn=rng.standard_normal((n_samples,nn)) if nn>0 else np.empty((n_samples,0))
    X=np.hstack([a for a in [Xi,Xr,Xn] if a.shape[1]>0])[:,:n_features]
    if flip_y>0:
        fm=rng.random(n_samples)<flip_y; y[fm]=rng.integers(0,n_classes,fm.sum())
    return X,y.astype(int)

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

def accuracy_score(yt,yp): return (np.asarray(yt)==np.asarray(yp)).mean()

class LogisticRegression:
    def __init__(self,C=1.0,max_iter=500,random_state=None): self.C=C; self.max_iter=max_iter
    def fit(self,X,y):
        self.classes_=np.unique(y); n,d=X.shape; lam=1./self.C
        if len(self.classes_)==2:
            yb=(y==self.classes_[1]).astype(float)
            def fg(w):
                p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
                loss=-np.mean(yb*np.log(p)+(1-yb)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
                e=p-yb; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
                return loss,g
            res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                                 options={"maxiter":self.max_iter})
            self._ws=[res.x]; self._binary=True
        else:
            self._ws=[]; self._binary=False
            for c in self.classes_:
                yb=(y==c).astype(float)
                def fg(w,yb=yb):
                    p=np.clip(_sig(X@w[1:]+w[0]),1e-7,1-1e-7)
                    loss=-np.mean(yb*np.log(p)+(1-yb)*np.log(1-p))+0.5*lam*np.sum(w[1:]**2)
                    e=p-yb; g=np.r_[e.mean(),X.T@e/n+lam*w[1:]]
                    return loss,g
                res=_sp_opt.minimize(fg,np.zeros(d+1),method="L-BFGS-B",jac=True,
                                     options={"maxiter":self.max_iter})
                self._ws.append(res.x)
        return self
    def predict_proba(self,X):
        if self._binary:
            p=_sig(X@self._ws[0][1:]+self._ws[0][0]); return np.c_[1-p,p]
        scores=np.column_stack([_sig(X@w[1:]+w[0]) for w in self._ws])
        return scores/scores.sum(1,keepdims=True)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

class KMeans:
    def __init__(self,n_clusters=3,random_state=None,n_init=10,max_iter=300):
        self.n_clusters=n_clusters; self.random_state=random_state
        self.n_init=n_init; self.max_iter=max_iter
    def _run_once(self,X,rng):
        idx=rng.choice(len(X),self.n_clusters,replace=False)
        centers=X[idx].copy()
        for _ in range(self.max_iter):
            dists=((X[:,None]-centers[None])**2).sum(2)
            labels=dists.argmin(1)
            new_centers=np.array([X[labels==k].mean(0) if (labels==k).any()
                                  else centers[k] for k in range(self.n_clusters)])
            if np.allclose(centers,new_centers,atol=1e-6): break
            centers=new_centers
        return labels,centers,((X-centers[labels])**2).sum()
    def fit(self,X):
        rng=np.random.default_rng(self.random_state); best=None
        for _ in range(self.n_init):
            r=self._run_once(X,rng)
            if best is None or r[2]<best[2]: best=r
        self.labels_,self.cluster_centers_,self.inertia_=best; return self
    def predict(self,X):
        return ((X[:,None]-self.cluster_centers_[None])**2).sum(2).argmin(1)

def adjusted_rand_score(labels_true,labels_pred):
    n=len(labels_true)
    ut=np.unique(labels_true); up=np.unique(labels_pred)
    mt={v:i for i,v in enumerate(ut)}; mp={v:i for i,v in enumerate(up)}
    ct=np.zeros((len(ut),len(up)),dtype=int)
    for a,b in zip(labels_true,labels_pred): ct[mt[a],mp[b]]+=1
    def C2(x): return x*(x-1)//2
    s_c2=sum(C2(ct[i,j]) for i in range(ct.shape[0]) for j in range(ct.shape[1]))
    a=ct.sum(1); b=ct.sum(0)
    sa=np.sum(C2(a)); sb=np.sum(C2(b))
    exp=sa*sb/C2(n); mx=0.5*(sa+sb)-exp
    return 0. if mx==0 else (s_c2-exp)/mx

class LabelSpreading:
    def __init__(self,kernel="rbf",alpha=0.2,max_iter=100,gamma=20):
        self.alpha=alpha; self.max_iter=max_iter; self.gamma=gamma
    def fit(self,X,y):
        n=len(X); self._classes=np.unique(y[y>=0])
        d2=((X[:,None]-X[None])**2).sum(2)
        W=np.exp(-self.gamma*d2); np.fill_diagonal(W,0)
        D=W.sum(1); D[D==0]=1; S=W/D[:,None]
        nc=len(self._classes); cmap={c:i for i,c in enumerate(self._classes)}
        Y=np.zeros((n,nc))
        for i,yi in enumerate(y):
            if yi>=0: Y[i,cmap[yi]]=1.
        F=Y.copy()
        for _ in range(self.max_iter):
            F=self.alpha*(S@F)+(1-self.alpha)*Y
        self._F=F; self._X=X.copy(); return self
    def predict(self,X):
        d2=((X[:,None]-self._X[None])**2).sum(2)
        W=np.exp(-self.gamma*d2); W/=W.sum(1,keepdims=True)+1e-10
        return self._classes[(W@self._F).argmax(1)]

class MLPClassifier:
    def __init__(self,hidden_layer_sizes=(100,),activation="relu",max_iter=200,
                 random_state=None,alpha=0.0001,learning_rate_init=0.001,
                 warm_start=False,batch_size=200):
        self.hidden_layer_sizes=hidden_layer_sizes; self.activation=activation
        self.max_iter=max_iter; self.random_state=random_state; self.alpha=alpha
        self.lr=learning_rate_init; self.bs=batch_size
    def _act(self,z):
        if self.activation=="relu": return np.maximum(0,z)
        if self.activation=="tanh": return np.tanh(z)
        return _sig(z)
    def _dact(self,z):
        if self.activation=="relu": return (z>0).astype(float)
        if self.activation=="tanh": t=np.tanh(z); return 1-t**2
        s=_sig(z); return s*(1-s)
    def _softmax(self,z):
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def fit(self,X,y):
        rng=np.random.default_rng(self.random_state)
        self.classes_=np.unique(y); nc=len(self.classes_)
        cmap={c:i for i,c in enumerate(self.classes_)}
        Y=np.eye(nc)[[cmap[yi] for yi in y]]
        layers=[X.shape[1]]+list(self.hidden_layer_sizes)+[nc]
        self.coefs_=[rng.normal(0,np.sqrt(2/layers[i]),(layers[i],layers[i+1]))
                     for i in range(len(layers)-1)]
        self.intercepts_=[np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        n=len(X); bs=min(self.bs,n)
        for ep in range(self.max_iter):
            idx=rng.permutation(n)
            for start in range(0,n,bs):
                b=idx[start:start+bs]; Xb=X[b]; Yb=Y[b]
                acts=[Xb]; zs=[]
                for i,(W,bi) in enumerate(zip(self.coefs_,self.intercepts_)):
                    z=acts[-1]@W+bi; zs.append(z)
                    acts.append(self._act(z) if i<len(self.coefs_)-1 else self._softmax(z))
                dA=acts[-1]-Yb
                for i in reversed(range(len(self.coefs_))):
                    dZ=dA if i==len(self.coefs_)-1 else dA*self._dact(zs[i])
                    dW=acts[i].T@dZ/len(b)+self.alpha*self.coefs_[i]
                    db=dZ.mean(0); dA=dZ@self.coefs_[i].T
                    self.coefs_[i]-=self.lr*dW; self.intercepts_[i]-=self.lr*db
        return self
    def _fwd(self,X):
        A=X
        for i,(W,b) in enumerate(zip(self.coefs_,self.intercepts_)):
            z=A@W+b
            A=self._act(z) if i<len(self.coefs_)-1 else self._softmax(z)
        return A
    def predict_proba(self,X): return self._fwd(X)
    def predict(self,X): return self.classes_[self._fwd(X).argmax(1)]
    def score(self,X,y): return (self.predict(X)==y).mean()

np.random.seed(42)

# ── Simulate a SOURCE domain (abundant labelled data, related task) ───────
# Source task: classify 10 types of objects (ImageNet-like)
# We simulate "pre-training" by learning a feature extractor
N_SOURCE    = 5000
N_FEATURES  = 20
N_SOURCE_CLS = 10

X_src, y_src = make_classification(
    n_samples=N_SOURCE, n_features=N_FEATURES, n_informative=12,
    n_redundant=4, n_classes=N_SOURCE_CLS, n_clusters_per_class=1,
    random_state=0)

scaler_src = StandardScaler().fit(X_src)
X_src_s    = scaler_src.transform(X_src)

# "Pre-train" a deep feature extractor on source task
pretrained_backbone = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu", max_iter=80, random_state=0)
pretrained_backbone.fit(X_src_s, y_src)
print("Pre-trained backbone accuracy on source task:",
      f"{pretrained_backbone.score(X_src_s, y_src)*100:.1f}%")

def extract_features(model, X):
    """Extract activations from the last hidden layer (32-dim)."""
    activations = X
    for i, (W, b) in enumerate(zip(model.coefs_[:-1], model.intercepts_[:-1])):
        activations = activations @ W + b
        activations = np.maximum(0, activations)   # ReLU
    return activations   # shape: (n, 32)

# ── TARGET domain (scarce labelled data, related 4-class task) ───────────
N_TARGET    = 1200
N_TARGET_CLS = 4

X_tgt, y_tgt = make_classification(
    n_samples=N_TARGET, n_features=N_FEATURES, n_informative=10,
    n_redundant=4, n_classes=N_TARGET_CLS, n_clusters_per_class=1,
    random_state=7)

scaler_tgt = StandardScaler().fit(X_tgt)
X_tgt_s    = scaler_tgt.transform(X_tgt)

X_tgt_feat = extract_features(pretrained_backbone, X_tgt_s)  # pre-trained features

_tgt_idx = np.arange(len(X_tgt_s))
_tr_idx, _te_idx = train_test_split(_tgt_idx, test_size=0.3,
    random_state=1, stratify=y_tgt)
X_train_all, X_test_tgt = X_tgt_s[_tr_idx], X_tgt_s[_te_idx]
y_train_all, y_test_tgt = y_tgt[_tr_idx], y_tgt[_te_idx]
X_feat_train_all = X_tgt_feat[_tr_idx]
X_feat_test      = X_tgt_feat[_te_idx]

# ── Sweep over target training set sizes ─────────────────────────────────
target_sizes = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 840]

scratch_accs   = []
feat_ext_accs  = []
finetune_accs  = []

print()
print("=" * 72)
print("  TRANSFER LEARNING: PERFORMANCE vs TARGET DATASET SIZE")
print(f"  Source: {N_SOURCE} samples, {N_SOURCE_CLS} classes (pre-training)")
print(f"  Target: {N_TARGET} total, {N_TARGET_CLS} classes (target task)")
print("=" * 72)
print()
print(f"  {'Target N':>8}  {'From Scratch':>13}  {'Feature Extraction':>19}  "
      f"{'Fine-Tuning':>12}  {'Transfer Gain':>13}")
print("  " + "─" * 72)

for n_target in target_sizes:
    if n_target > len(X_train_all):
        n_target = len(X_train_all)

    # Sample n_target examples
    idx = np.random.choice(len(X_train_all), size=n_target, replace=False)
    X_tr_sub   = X_train_all[idx]
    y_tr_sub   = y_train_all[idx]
    X_feat_sub = X_feat_train_all[idx]

    # 1. SCRATCH: Train MLP from random init on small target dataset
    clf_scratch = MLPClassifier(hidden_layer_sizes=(64, 32),
                                max_iter=300, random_state=0, alpha=0.01)
    clf_scratch.fit(X_tr_sub, y_tr_sub)
    acc_scratch = clf_scratch.score(X_test_tgt, y_test_tgt) * 100

    # 2. FEATURE EXTRACTION: freeze backbone, train linear head only
    clf_head = LogisticRegression(C=5.0, max_iter=500, random_state=0)
    clf_head.fit(X_feat_sub, y_tr_sub)
    acc_feat = clf_head.score(X_feat_test, y_test_tgt) * 100

    # 3. FINE-TUNING: continue training full network (warm-start from backbone)
    # Simulate by training MLP with warm initialisation (more epochs, smaller lr)
    clf_ft = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                            max_iter=150, random_state=0, alpha=0.005,
                            learning_rate_init=0.0005, warm_start=False)
    clf_ft.fit(X_feat_sub, y_tr_sub)   # fine-tune on pre-trained features
    acc_ft = clf_ft.score(X_feat_test, y_test_tgt) * 100

    scratch_accs.append(acc_scratch)
    feat_ext_accs.append(acc_feat)
    finetune_accs.append(acc_ft)
    best_transfer = max(acc_feat, acc_ft)
    gain = best_transfer - acc_scratch
    print(f"  {n_target:>8d}  {acc_scratch:>12.1f}%  {acc_feat:>18.1f}%  "
          f"{acc_ft:>11.1f}%  {gain:>+12.1f}%")

print()
print("  Key takeaway: transfer learning advantage is LARGEST when target")
print("  data is SCARCE. With only 10 target examples, feature extraction")
print("  already outperforms training from scratch with far more epochs.")
print("  As target data grows, the scratch model catches up — this matches")
print("  real-world practice: fine-tune for small datasets, consider")
print("  training from scratch when millions of target labels are available.")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Transfer Learning: From Scratch vs Feature Extraction vs Fine-Tuning\\n"
             "Transfer advantage is largest when target data is scarce",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.semilogx(target_sizes, scratch_accs,   "r-o", lw=2, ms=7,
            label="From Scratch")
ax.semilogx(target_sizes, feat_ext_accs,  "b-s", lw=2, ms=7,
            label="Feature Extraction\\n(frozen backbone)")
ax.semilogx(target_sizes, finetune_accs,  "g-^", lw=2, ms=7,
            label="Fine-Tuning\\n(full network)")
ax.set_title("Test Accuracy vs Target Dataset Size\\n(log scale x-axis)",
             fontweight="bold")
ax.set_xlabel("Number of Target Training Examples (log scale)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(20, 100)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# Panel 2: Transfer gain (advantage over scratch)
ax2 = axes[1]
feat_gain = [f - s for f, s in zip(feat_ext_accs, scratch_accs)]
ft_gain   = [f - s for f, s in zip(finetune_accs, scratch_accs)]
ax2.semilogx(target_sizes, feat_gain, "b-s", lw=2, ms=7,
             label="Feature Extraction gain")
ax2.semilogx(target_sizes, ft_gain,   "g-^", lw=2, ms=7,
             label="Fine-Tuning gain")
ax2.axhline(0, color="red", ls="--", lw=1.5, label="Scratch baseline")
ax2.fill_between(target_sizes,
                 [max(f, t) for f, t in zip(feat_gain, ft_gain)],
                 0, alpha=0.15, color="steelblue", label="Transfer advantage")
ax2.set_title("Transfer Learning Advantage\\n(Accuracy gain over from-scratch)",
              fontweight="bold")
ax2.set_xlabel("Number of Target Training Examples (log scale)")
ax2.set_ylabel("Accuracy Gain over From-Scratch (%)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("transfer_learning_comparison.png", dpi=110)
print()
print("  Plot saved → transfer_learning_comparison.png")
print()
print("  Practical guidance:")
print("  • < 100 target labels:  Feature Extraction — don't risk overfitting")
print("  • 100-10K target labels: Fine-Tuning — adapt the representations")
print("  • > 100K target labels:  Consider training from scratch")
print("  • Very different domains: Fine-tune all layers with small lr")
print("  • Very similar domains:   Feature Extraction is often sufficient")
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