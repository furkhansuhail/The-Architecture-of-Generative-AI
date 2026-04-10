"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MODULE 09 · Training Concepts                        ║
║                         A Deep-Dive Reference Guide                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Topics Covered                                                              ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║   1.  Pre-Training                    7.  LoRA & PEFT                        ║
║   2.  Fine-Tuning                     8.  Data Quality over Quantity         ║
║   3.  RLHF                            9.  Training Compute Scaling           ║
║   4.  Reward Models                  10.  Constitutional AI  [bonus]         ║
║   5.  SFT vs RLHF                    11.  Evaluation & Benchmarking [bonus]  ║
║   6.  Catastrophic Forgetting            Glossary                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Introduction
────────────
Training a large language model is a multi-stage pipeline. Each stage builds
on the previous one, adding capability, alignment, or efficiency. This module
provides a thorough explanation of every major concept in that pipeline — from
raw pre-training on internet text all the way through parameter-efficient
fine-tuning and scaling laws.

Understanding these concepts together is essential: they are deeply
interdependent. How a model is pre-trained determines what fine-tuning can
unlock; how data quality is managed determines whether compute scaling
delivers returns; how RLHF is applied determines alignment quality and safety
characteristics.

┌─────────────────────────────────────────────────────────────────────────────┐
│ HOW TO READ THIS MODULE                                                     │
│ Each section opens with a plain-English definition, then dives into the     │
│ mechanism, the intuition behind it, known failure modes, and where          │
│ applicable — mathematical framing. Key terms are highlighted on first use.  │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# SECTION 1 · PRE-TRAINING
# =============================================================================

PRE_TRAINING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  1 · PRE-TRAINING                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IS PRE-TRAINING?
─────────────────────
Pre-training is the foundational learning phase of a large language model
(LLM). The model is exposed to enormous quantities of text — typically
hundreds of billions to several trillion tokens drawn from web crawls, books,
code repositories, Wikipedia, scientific papers, and more — and trained to
perform a single, deceptively simple task: predict the next token.


THE TRAINING OBJECTIVE: NEXT-TOKEN PREDICTION
─────────────────────────────────────────────
Given a sequence of tokens [t1, t2, ..., tN], the model learns to predict
tN+1. This is called a causal language modelling (CLM) objective or
autoregressive modelling. The model outputs a probability distribution over
the entire vocabulary (commonly 32,000–128,000 tokens) and is penalised with
cross-entropy loss for distributing probability mass away from the correct
next token.

┌─────────────────────────────────────────────────────────────────────────────┐
│ INTUITION                                                                   │
│ "Predict the next word" sounds trivial, but to consistently predict well    │
│ across all contexts, the model must implicitly learn grammar, facts,        │
│ reasoning patterns, code syntax, social conventions, and more — because     │
│ all of these determine what comes next in real text.                        │
└─────────────────────────────────────────────────────────────────────────────┘


WHAT GETS LEARNED?
──────────────────
Pre-training is entirely self-supervised — no human labels are required. The
signal comes from the data itself. Through exposure to enough examples, the
model internalises:

  • Syntax and grammar across multiple languages
  • World knowledge encoded in factual prose
  • Reasoning chains present in textbooks and forums
  • Code structure and conventions across programming languages
  • Stylistic registers from formal academic writing to casual chat


ARCHITECTURE AND SCALE
──────────────────────
Modern pre-trained LLMs almost universally use the Transformer architecture —
specifically the decoder-only variant (e.g., GPT, LLaMA, Mistral). Key
hyperparameters include: number of layers, attention heads, embedding
dimension, context window length, and vocabulary size. Pre-training runs on
clusters of thousands of accelerators (GPUs or TPUs) for weeks to months.


DATA COMPOSITION
────────────────
Data mixture matters enormously. The ratio of web text to books to code to
scientific text shapes model capability profiles. CommonCrawl, The Pile, C4,
RedPajama, and FineWeb are common pre-training corpora. Deduplication, quality
filtering, and toxicity removal are standard preprocessing steps.


KEY FAILURE MODES IN PRE-TRAINING
──────────────────────────────────
  • Memorisation of training data (privacy and copyright risks)
  • Bias and toxicity inherited from web-scraped text
  • Inconsistent knowledge across domains due to imbalanced data
  • Training instabilities at scale (loss spikes, gradient explosions)
"""


# =============================================================================
# SECTION 2 · FINE-TUNING
# =============================================================================

FINE_TUNING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  2 · FINE-TUNING                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IS FINE-TUNING?
────────────────────
Fine-tuning takes a pre-trained model — already rich with general language
knowledge — and continues training it on a smaller, curated dataset to
specialise its behaviour. The goal is to shift the model from being a
general-purpose next-token predictor into a model that behaves helpfully,
safely, or competently within a specific domain or task.


TYPES OF FINE-TUNING
────────────────────

  TASK-SPECIFIC FINE-TUNING
  ─────────────────────────
  The model is trained on labelled examples of a specific task: sentiment
  classification, named entity recognition, summarisation, question answering,
  etc. The pre-trained weights provide a strong initialisation; the fine-tuning
  step adapts the model to the new task's distribution.

  INSTRUCTION FINE-TUNING
  ───────────────────────
  The model is trained on (instruction, response) pairs, teaching it to follow
  natural language instructions. This is what transforms a raw completion model
  into an assistant. Datasets like FLAN, Alpaca, and Dolly are examples. The
  quality and diversity of instructions drive generalisation.

  DOMAIN FINE-TUNING
  ──────────────────
  Continued pre-training on domain-specific corpora (medical, legal, financial,
  code). This refreshes and deepens domain knowledge without changing the task
  format.


HOW FINE-TUNING WORKS MECHANICALLY
────────────────────────────────────
The pre-trained model's weights are used as a starting point. A smaller
learning rate is used than in pre-training to avoid overwriting learned
representations. The model is trained to minimise loss on the curated
fine-tuning examples, updating all or a subset of parameters.


THE RISK OF OVER-FITTING
────────────────────────
Fine-tuning on a small dataset can cause the model to over-fit: it memorises
the training examples rather than generalising. Early stopping, regularisation,
and careful dataset curation mitigate this.
"""


# =============================================================================
# SECTION 3 · RLHF
# =============================================================================

RLHF = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  3 · REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

THE ALIGNMENT PROBLEM RLHF SOLVES
──────────────────────────────────
After supervised fine-tuning, models can follow instructions but they do not
reliably behave the way humans prefer. They may be verbose, sycophantic,
unhelpful, or unsafe in subtle ways. The fundamental issue is that 'generate
text that looks like a good answer' (which SFT optimises) is not the same as
'generate text that actually is a good answer' (which we care about). RLHF
bridges this gap.


THE THREE-STAGE RLHF PIPELINE
──────────────────────────────

  STAGE 1 — SUPERVISED FINE-TUNING (SFT)
  ───────────────────────────────────────
  A base model is fine-tuned on high-quality human-written demonstrations to
  produce an initial policy model capable of following instructions. This gives
  the RL training a reasonable starting point.

  STAGE 2 — REWARD MODEL TRAINING
  ────────────────────────────────
  Human raters are shown pairs of model outputs for the same prompt and asked
  to choose which is better. These comparisons train a separate reward model
  (RM) to predict human preferences. The RM maps a (prompt, response) pair to
  a scalar score.

  STAGE 3 — RL POLICY OPTIMISATION
  ─────────────────────────────────
  The SFT model (now called the policy) is further trained using Proximal
  Policy Optimisation (PPO) or similar RL algorithms. For each generated
  response, the reward model scores it. The policy is updated to maximise the
  reward — i.e., to generate responses the RM (as a proxy for humans) would
  rate highly.

┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY RL AND NOT JUST MORE SFT?                                               │
│ SFT requires ground-truth correct responses for every example — expensive   │
│ to produce at scale. RLHF only needs preference comparisons (A or B?),      │
│ which are much cheaper to collect. More importantly, RL can optimise for    │
│ reward beyond what was demonstrated in the training data.                   │
└─────────────────────────────────────────────────────────────────────────────┘


PPO IN THE RLHF CONTEXT
────────────────────────
PPO (Proximal Policy Optimisation) is used because it constrains how much the
policy changes per update step via a clipping mechanism, preventing
catastrophic policy collapse. A KL divergence penalty between the current
policy and the SFT reference model prevents the policy from drifting too far
from sensible language generation.


KEY CHALLENGES AND FAILURE MODES
──────────────────────────────────
  • Reward hacking: the policy finds ways to maximise reward score without
    truly improving quality (e.g., producing verbose but empty responses)
  • Mode collapse: the policy narrows to a small set of 'safe' high-scoring
    outputs
  • Distributional shift: the reward model trained on limited data may not
    generalise to novel policy outputs
  • Human rater inconsistency: noisy preference labels degrade the reward model
"""


# =============================================================================
# SECTION 4 · REWARD MODEL
# =============================================================================

REWARD_MODEL = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  4 · THE REWARD MODEL                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

ROLE IN THE PIPELINE
────────────────────
The reward model (RM) is a separate neural network, typically initialised from
the same pre-trained base as the policy, with a regression head replacing the
final vocabulary projection layer. Its sole job is to score (prompt, response)
pairs with a scalar value representing predicted human preference.


TRAINING THE REWARD MODEL
──────────────────────────
Human annotators rate pairs of responses for the same prompt. Given outputs A
and B, they indicate which is better (or if they are roughly equivalent). The
reward model is trained with a contrastive loss (Bradley-Terry model) such
that the RM assigns a higher score to the preferred response than to the
rejected one.

┌─────────────────────────────────────────────────────────────────────────────┐
│ MATHEMATICAL FRAMING                                                        │
│                                                                             │
│   Loss = -log( sigmoid( RM(prompt, chosen) - RM(prompt, rejected) ) )      │
│                                                                             │
│ This pushes the score gap between chosen and rejected responses to be as    │
│ large as possible.                                                          │
└─────────────────────────────────────────────────────────────────────────────┘


WHAT MAKES A GOOD REWARD MODEL?
────────────────────────────────
  • Diverse preference data covering many topics, tones, and failure modes
  • Calibrated scores — not just binary rankings but graded quality differences
  • Generalisation to out-of-distribution prompts and responses
  • Robustness to adversarial responses that 'game' surface features


GOODHART'S LAW AND REWARD HACKING
───────────────────────────────────
The reward model is a proxy for human preferences, not the real thing. When
the policy is optimised aggressively against the RM, it finds responses that
score highly on the proxy but do not actually satisfy human intent. This is an
instance of Goodhart's Law: "When a measure becomes a target, it ceases to be
a good measure." Regular human evaluation alongside RM scores is essential to
detect drift.


CONSTITUTIONAL AI AND PROCESS REWARD MODELS
─────────────────────────────────────────────
Extensions of the reward model idea include: Constitutional AI (Anthropic),
where the model self-critiques responses against a set of principles before
rating them; and Process Reward Models (PRMs), which score individual
reasoning steps rather than the final answer — particularly important for
mathematical and logical reasoning tasks.
"""


# =============================================================================
# SECTION 5 · SFT vs RLHF
# =============================================================================

SFT_VS_RLHF = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  5 · SFT vs RLHF — A COMPARISON                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINING THE TWO APPROACHES
────────────────────────────
Supervised Fine-Tuning (SFT) trains the model via maximum likelihood on
labelled (input, target_output) pairs. RLHF trains the model to maximise a
reward signal derived from human preference comparisons, via reinforcement
learning.


COMPARISON TABLE
────────────────

┌──────────────────────┬────────────────────────────────┬──────────────────────────────────────┐
│ Dimension            │ SFT                            │ RLHF                                 │
├──────────────────────┼────────────────────────────────┼──────────────────────────────────────┤
│ Training signal      │ Labelled input-output pairs    │ Scalar reward from human preference  │
│                      │                                │ comparisons                          │
├──────────────────────┼────────────────────────────────┼──────────────────────────────────────┤
│ Data requirement     │ High-quality demonstrations    │ Preference pairs (cheaper to collect)│
├──────────────────────┼────────────────────────────────┼──────────────────────────────────────┤
│ Optimisation         │ Maximum likelihood /           │ Policy gradient (PPO);               │
│                      │ cross-entropy                  │ KL-regularised                       │
├──────────────────────┼────────────────────────────────┼──────────────────────────────────────┤
│ Risk                 │ Over-fitting, imitation        │ Reward hacking, mode collapse        │
│                      │ ceiling                        │                                      │
├──────────────────────┼────────────────────────────────┼──────────────────────────────────────┤
│ Use case             │ Teach format and task          │ Align behaviour with human values    │
│                      │ behaviour                      │                                      │
├──────────────────────┼────────────────────────────────┼──────────────────────────────────────┤
│ Sample efficiency    │ High — direct supervision      │ Lower — requires many rollouts       │
└──────────────────────┴────────────────────────────────┴──────────────────────────────────────┘


DPO: A SIMPLER ALTERNATIVE
───────────────────────────
Direct Preference Optimisation (DPO) is a recent approach that achieves
RLHF-like alignment without a separate reward model or RL loop. It reformulates
the preference learning objective so the policy is optimised directly from
comparison data. DPO is simpler to implement and often competitive with
PPO-based RLHF in practice.
"""


# =============================================================================
# SECTION 6 · CATASTROPHIC FORGETTING
# =============================================================================

CATASTROPHIC_FORGETTING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  6 · CATASTROPHIC FORGETTING                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IS IT?
───────────
Catastrophic forgetting (also called catastrophic interference) is the
phenomenon where training a neural network on a new task causes it to abruptly
lose performance on previously learned tasks. In the LLM context, fine-tuning
on a narrow task can degrade the model's general language capabilities, world
knowledge, or reasoning skills.


WHY IT HAPPENS
──────────────
Neural networks store knowledge in distributed weight patterns across all
layers. When gradient updates shift weights to reduce loss on the fine-tuning
data, those same weight changes can interfere with the patterns that encode
prior knowledge. Unlike human memory (which is largely additive and
associative), gradient descent rewrites shared weights without an explicit
mechanism to protect old information.


PRACTICAL CONSEQUENCES FOR LLMs
────────────────────────────────
  • A model fine-tuned aggressively on customer service data may forget how to
    write code
  • Domain fine-tuning may reduce factual accuracy in unrelated domains
  • Instruction fine-tuning on a narrow dataset can reduce stylistic diversity
  • RLHF can cause an 'alignment tax': RL updates sometimes reduce benchmark
    performance


MITIGATION STRATEGIES
──────────────────────

  REGULARISATION TECHNIQUES
  ─────────────────────────
  Elastic Weight Consolidation (EWC) adds a penalty proportional to the
  importance of each weight for previous tasks, slowing the rate at which
  those weights change during fine-tuning. Synaptic Intelligence (SI) is a
  related approach.

  REPLAY / DATA MIXING
  ────────────────────
  Mixing fine-tuning data with samples from the original pre-training
  distribution (or a proxy thereof) prevents the model from drifting too far
  from its general capabilities. This is widely used in practice.

  PARAMETER-EFFICIENT FINE-TUNING (PEFT)
  ───────────────────────────────────────
  By freezing most pre-trained weights and only updating a small subset (or a
  low-rank adapter), PEFT methods dramatically reduce the surface area for
  catastrophic forgetting. See Section 7.

  MODULAR / MIXTURE-OF-EXPERTS ARCHITECTURES
  ───────────────────────────────────────────
  Routing computation through different expert sub-networks for different
  inputs can allow task-specific adaptation without disrupting general weights.
"""


# =============================================================================
# SECTION 7 · LoRA AND PEFT
# =============================================================================

LORA_AND_PEFT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  7 · LoRA AND PARAMETER-EFFICIENT FINE-TUNING (PEFT)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

THE PROBLEM PEFT SOLVES
────────────────────────
Full fine-tuning of a 70-billion-parameter model requires updating all 70B
parameters, which demands enormous GPU memory and compute. It also risks
catastrophic forgetting. Parameter-Efficient Fine-Tuning (PEFT) methods update
only a small fraction of parameters while keeping the rest frozen, achieving
comparable results at a fraction of the cost.


LOW-RANK ADAPTATION (LoRA)
──────────────────────────

  CORE IDEA
  ─────────
  LoRA (Hu et al., 2021) is the most widely adopted PEFT method. It observes
  that the weight updates during fine-tuning have a low intrinsic
  dimensionality — meaning the actual degrees of freedom needed to adapt a
  model are far fewer than the number of parameters.

  MECHANISM
  ─────────
  For a weight matrix W (of shape d×k), instead of updating W directly, LoRA
  inserts two small trainable matrices A (d×r) and B (r×k) where r << min(d,k).
  The adapted weight is W + AB. During training, only A and B are updated; W
  remains frozen. At inference, AB can be merged back into W for zero overhead.

┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY LOW RANK WORKS                                                          │
│ The hypothesis is that the 'task adaptation' signal lies in a               │
│ low-dimensional subspace of the full weight space. If the pre-trained       │
│ model already encodes most of the required knowledge, fine-tuning only      │
│ needs to adjust a small directional component, which can be captured by a   │
│ rank-r factorisation.                                                       │
└─────────────────────────────────────────────────────────────────────────────┘


LoRA HYPERPARAMETERS
─────────────────────

┌───────────────────┬──────────────────────────────────────────────────────────┐
│ Hyperparameter    │ Description                                              │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ r  (rank)         │ Dimensionality of the adapter matrices. Typical values:  │
│                   │ 4, 8, 16, 32, 64. Higher r = more parameters, more       │
│                   │ expressivity, more risk of forgetting.                   │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ alpha  (scaling)  │ Scaling factor applied to the LoRA output: (alpha/r)*AB. │
│                   │ Controls the effective learning rate of the adapter.     │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ Target modules    │ Which weight matrices to attach LoRA to. Attention       │
│                   │ projections (Q, K, V, O) are most common; MLP layers     │
│                   │ sometimes included.                                      │
├───────────────────┼──────────────────────────────────────────────────────────┤
│ Dropout           │ Applied to adapter matrices as regularisation.           │
└───────────────────┴──────────────────────────────────────────────────────────┘


OTHER PEFT METHODS
──────────────────

  QLoRA
  ─────
  Combines LoRA with 4-bit quantisation of the frozen base model weights. This
  makes fine-tuning a 65B model feasible on a single consumer GPU. QLoRA
  introduced several innovations: 4-bit NormalFloat (NF4) quantisation, double
  quantisation, and paged optimisers.

  PREFIX TUNING & PROMPT TUNING
  ──────────────────────────────
  Rather than modifying weights, these methods prepend trainable 'soft prompt'
  vectors to the input (prompt tuning) or to the hidden states at each layer
  (prefix tuning). Only these vectors are trained. They are parameter-efficient
  but less expressive than LoRA for large task shifts.

  ADAPTER LAYERS
  ──────────────
  Small bottleneck feed-forward networks inserted between existing layers. The
  original layers are frozen; only the adapters are trained. Adapters are
  architecturally clean but add inference latency (unlike LoRA, they cannot be
  merged away).

  (IA)³
  ─────
  Learns learned vectors that rescale keys, values, and feed-forward
  activations. Even fewer parameters than LoRA with competitive performance on
  many benchmarks.
"""


# =============================================================================
# SECTION 8 · DATA QUALITY OVER QUANTITY
# =============================================================================

DATA_QUALITY_OVER_QUANTITY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  8 · DATA QUALITY OVER QUANTITY                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

THE CORE PRINCIPLE
──────────────────
One million carefully curated, high-quality training examples consistently
outperforms one billion noisy, redundant, or low-quality ones. This principle
has been validated empirically across pre-training, instruction tuning, and
RLHF. Raw scale of data cannot compensate for systematic quality deficiencies.


WHAT MAKES DATA 'HIGH QUALITY'?
─────────────────────────────────

  ACCURACY AND FACTUAL CORRECTNESS
  ─────────────────────────────────
  Training data containing factual errors teaches the model those errors.
  Web-scraped data contains significant misinformation, outdated facts, and
  hallucinated content from other language models (a problem called 'model
  collapse' or 'data pollution').

  DIVERSITY AND COVERAGE
  ──────────────────────
  High diversity — across topics, styles, domains, difficulty levels,
  languages, and formats — produces models that generalise better. Datasets
  with high redundancy inflate apparent size without adding information.

  INSTRUCTION-FOLLOWING FIDELITY
  ───────────────────────────────
  For instruction fine-tuning, examples where the response actually follows
  the instruction well are far more valuable than examples where the response
  is superficially related but non-compliant.

  ABSENCE OF HARMFUL CONTENT
  ───────────────────────────
  Harmful, toxic, or biased content in training data is learned by the model.
  Quality pipelines filter aggressively for such content, though perfect
  filtering is unsolved.


EVIDENCE FOR QUALITY OVER QUANTITY
────────────────────────────────────
  • Phi-1 (Microsoft, 2023): a 1.3B model trained only on 'textbook quality'
    synthetic data outperformed 7B models trained on diverse web data on
    coding benchmarks.

  • Alpaca vs. LIMA: LIMA (Zhou et al., 2023) showed that 1,000 carefully
    chosen instruction-response pairs can achieve instruction-following quality
    comparable to models trained on 52,000 Alpaca examples.

  • Chinchilla analysis: using more high-quality tokens per parameter is better
    than scaling parameters alone on a fixed compute budget — quality of the
    token distribution matters.


DATA CURATION TECHNIQUES
─────────────────────────

  DEDUPLICATION
  ─────────────
  Exact and near-duplicate documents inflate apparent dataset size and cause
  models to memorise rather than generalise. MinHash LSH and suffix arrays are
  used for efficient deduplication at scale.

  QUALITY SCORING AND FILTERING
  ──────────────────────────────
  Classifier-based filtering (using a 'quality classifier' trained on
  known-good documents vs. random web text), perplexity filtering (removing
  documents that are either too easy or too hard for a reference model), and
  heuristic filters (removing very short documents, those with high symbol
  ratios, etc.).

  SYNTHETIC DATA GENERATION
  ──────────────────────────
  Using a capable model to generate training data (Alpaca, WizardLM, Orca)
  allows scale and control. Quality control on synthetic data is critical:
  model-generated data can inherit model biases and hallucinations.
"""


# =============================================================================
# SECTION 9 · TRAINING COMPUTE SCALING
# =============================================================================

TRAINING_COMPUTE_SCALING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  9 · TRAINING COMPUTE SCALING AND THE CHINCHILLA LAWS                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

THE SCALING HYPOTHESIS
──────────────────────
A central empirical finding in deep learning is that model performance scales
predictably with three factors: model parameters (N), training tokens (D), and
compute budget (C). The relationship follows approximate power laws: loss
decreases as a power of each factor when the others are held fixed.


PRE-CHINCHILLA ERA: PARAMETER-CENTRIC SCALING
──────────────────────────────────────────────
GPT-3 (Brown et al., 2020) and contemporaries were trained under the
assumption that bigger models are better. GPT-3 used 175B parameters trained
on approximately 300B tokens — roughly 1.7 tokens per parameter. This ratio
was thought to be reasonable but was later shown to be severely undertrained.


THE CHINCHILLA LAWS (Hoffmann et al., 2022)
────────────────────────────────────────────
DeepMind's 'Training Compute-Optimal Large Language Models' paper (colloquially
'the Chinchilla paper') studied over 400 models ranging from 70M to 16B
parameters. Their key finding: for a given compute budget C, the optimal
strategy allocates roughly equal resources to parameters and tokens.

┌─────────────────────────────────────────────────────────────────────────────┐
│ THE CHINCHILLA RULE OF THUMB                                                │
│                                                                             │
│   Optimal token count  D ≈ 20 × N                                           │
│   (where N = number of parameters)                                          │
│                                                                             │
│ A 7B parameter model should be trained on approximately 140B tokens for     │
│ compute-optimal training. The Chinchilla model (70B params, 1.4T tokens)    │
│ dramatically outperformed Gopher (280B params, 300B tokens) despite using   │
│ the same compute budget.                                                    │
└─────────────────────────────────────────────────────────────────────────────┘


IMPLICATIONS OF CHINCHILLA SCALING
────────────────────────────────────
  • Many of the largest models circa 2021–22 were massively undertrained
    relative to their parameter count
  • Smaller models trained on more data can match or beat larger undertrained
    models at the same compute budget
  • LLaMA (Touvron et al., 2023) deliberately trained smaller models (7B–65B)
    on much larger token counts (1T–1.4T) to produce inference-efficient models
    that still achieve high capability
  • For inference-constrained applications (where the model runs many times),
    it is often worth spending extra training compute to shrink the model


COMPUTE BUDGET AND SCALING LAWS
─────────────────────────────────

  FLOPs ESTIMATION
  ────────────────
  Approximate total training FLOPs:

      C ≈ 6 × N × D

  Where the factor of 6 accounts for the forward pass (2ND), backward pass
  (4ND), and gradient accumulation. This formula holds for transformer models
  trained with the standard Adam optimiser.

  BEYOND CHINCHILLA
  ─────────────────
  Chinchilla optimality is defined for a fixed compute budget when you train
  and discard the model. In practice, if a model will serve millions of
  inference requests, it may be worth training longer (more tokens) even if
  the marginal loss reduction is small, because the per-query inference cost of
  a smaller model is lower. LLaMA 2, Mistral, and Gemma all reflect this
  philosophy.


DATA SCALING WALLS
──────────────────
A practical concern emerging as token counts reach into the trillions: we may
be approaching the limit of high-quality human-generated text available on the
internet. Responses to this include: synthetic data generation (using frontier
models), multi-epoch training (with careful de-duplication), and continued
data quality investment.
"""


# =============================================================================
# SECTION 10 · CONSTITUTIONAL AI  [bonus topic]
# =============================================================================

CONSTITUTIONAL_AI = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  10 · CONSTITUTIONAL AI  [Bonus Topic]                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

MOTIVATION
──────────
Constitutional AI (CAI), introduced by Anthropic, addresses a limitation of
standard RLHF: it requires large volumes of human feedback to train the reward
model. CAI proposes using a set of principles (a 'constitution') and the model
itself to generate preference data — reducing reliance on human labellers for
safety-relevant feedback.


HOW IT WORKS
────────────

  SUPERVISED LEARNING PHASE — CRITIQUE AND REVISION
  ──────────────────────────────────────────────────
  A 'helpful-only' SFT model generates an initial response. The model is then
  prompted to critique its own response against a principle from the
  constitution (e.g., 'identify any harmful content in this response'). It then
  revises the response to fix the identified issue. This critique-revise loop
  is applied multiple times with different principles. The revised responses
  become supervised fine-tuning data.

  RL PHASE — AI FEEDBACK
  ──────────────────────
  Instead of humans comparing responses, a feedback model is prompted with a
  constitutional principle to choose the better of two responses. This
  AI-generated preference data trains a Preference Model (PM) — equivalent to
  a reward model. The policy is then optimised against this PM using RL, as in
  standard RLHF.


ADVANTAGES
──────────
  • Scales safety feedback without proportionally scaling human annotator time
  • Makes safety criteria explicit and auditable (the constitution is a
    readable document)
  • Reduces inconsistency introduced by human rater variability on edge cases
"""


# =============================================================================
# SECTION 11 · EVALUATION AND BENCHMARKING  [bonus topic]
# =============================================================================

EVALUATION_AND_BENCHMARKING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  11 · EVALUATION AND BENCHMARKING  [Bonus Topic]                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHY EVALUATION IS HARD
────────────────────────
Evaluating LLMs is substantially harder than evaluating narrow ML systems. The
model's output space is vast (natural language) and quality is
multi-dimensional: factual accuracy, helpfulness, safety, reasoning ability,
instruction following, creativity, and calibration are all separately important
and partially in tension.


CATEGORIES OF EVALUATION
─────────────────────────

  AUTOMATED BENCHMARKS
  ────────────────────
  Fixed question sets with reference answers: MMLU (world knowledge),
  HellaSwag (commonsense), HumanEval (code generation), GSM8K (arithmetic
  reasoning), MATH, BIG-Bench. These enable reproducible comparison but can be
  gamed by benchmark contamination (training data that overlaps with the
  benchmark).

  LLM-AS-JUDGE
  ────────────
  A frontier model (GPT-4, Claude) rates pairwise comparisons of responses,
  approximating human preferences at scale. MT-Bench and AlpacaEval use this
  approach. Susceptible to positional bias and self-promotion bias (a model
  rating itself favourably).

  HUMAN EVALUATION
  ────────────────
  The gold standard, but expensive and slow. Essential for detecting subtle
  misalignment, unsafe outputs, or capability regressions not captured by
  automated metrics.


BENCHMARK CONTAMINATION
────────────────────────
If benchmark questions appear in pre-training data, reported benchmark
performance overstates true generalisation. This is a growing problem as
benchmarks age and their content spreads across the internet. Mitigation
includes: held-out test sets, decontamination procedures, and newer, harder
benchmarks created after model training cutoffs.
"""


# =============================================================================
# GLOSSARY
# =============================================================================

GLOSSARY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUICK REFERENCE GLOSSARY                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────┬───────────────────────────────────────────────┐
│ Term                         │ Definition                                    │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Autoregressive model         │ Model that generates tokens one at a time,    │
│                              │ conditioning on all previous tokens.          │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Benchmark contamination      │ When test set questions appear in training     │
│                              │ data, inflating reported scores.              │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ CAI (Constitutional AI)      │ Anthropic's method for using a written        │
│                              │ constitution and self-critique to improve     │
│                              │ alignment.                                    │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Chinchilla laws              │ Empirical scaling rules showing that          │
│                              │ compute-optimal training requires             │
│                              │ proportional scaling of both parameters and   │
│                              │ tokens.                                       │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Cross-entropy loss           │ Standard training objective for language      │
│                              │ models; measures divergence between predicted │
│                              │ and true token distributions.                 │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ DPO (Direct Preference Opt.) │ RLHF variant that optimises the policy        │
│                              │ directly from preference data without a       │
│                              │ separate RM.                                  │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ EWC (Elastic Weight Consol.) │ Regularisation technique that slows updates   │
│                              │ to weights important for previous tasks.      │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ FLOPs                        │ Floating-point operations — the standard unit │
│                              │ for measuring training and inference compute. │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Goodhart's Law               │ When a proxy measure becomes the target, it   │
│                              │ stops being a good measure.                   │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ KL divergence penalty        │ Constraint in RLHF that prevents the RL       │
│                              │ policy from drifting far from the SFT         │
│                              │ reference model.                              │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ LoRA (Low-Rank Adaptation)   │ PEFT method that inserts trainable low-rank   │
│                              │ matrices alongside frozen weight matrices.    │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ PEFT                         │ Parameter-Efficient Fine-Tuning — umbrella    │
│                              │ term for methods that update only a fraction  │
│                              │ of model parameters.                          │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ PPO                          │ Proximal Policy Optimisation — RL algorithm   │
│                              │ used in RLHF to update the language model     │
│                              │ policy.                                       │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Process Reward Model         │ Reward model that scores individual reasoning │
│                              │ steps rather than the final output.           │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ QLoRA                        │ LoRA applied to a 4-bit quantised base model  │
│                              │ for memory-efficient fine-tuning.             │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Reward model                 │ Separate model trained to score               │
│                              │ (prompt, response) pairs based on human       │
│                              │ preferences.                                  │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ RLHF                         │ Reinforcement Learning from Human Feedback —  │
│                              │ aligning LLM behaviour via preference-based   │
│                              │ reward signals.                               │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ SFT (Supervised Fine-Tuning) │ Training on labelled input-output pairs with  │
│                              │ standard cross-entropy loss.                  │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Token                        │ The atomic unit of text for LLMs; roughly     │
│                              │ 0.75 words in English.                        │
├──────────────────────────────┼───────────────────────────────────────────────┤
│ Transformer                  │ Dominant neural architecture for LLMs, based  │
│                              │ on self-attention mechanisms.                 │
└──────────────────────────────┴───────────────────────────────────────────────┘
"""


# =============================================================================
# MODULE INDEX
# =============================================================================

SECTIONS: dict[str, str] = {
    "intro":                    __doc__,
    "pre_training":             PRE_TRAINING,
    "fine_tuning":              FINE_TUNING,
    "rlhf":                     RLHF,
    "reward_model":             REWARD_MODEL,
    "sft_vs_rlhf":              SFT_VS_RLHF,
    "catastrophic_forgetting":  CATASTROPHIC_FORGETTING,
    "lora_and_peft":            LORA_AND_PEFT,
    "data_quality_over_quantity": DATA_QUALITY_OVER_QUANTITY,
    "training_compute_scaling": TRAINING_COMPUTE_SCALING,
    "constitutional_ai":        CONSTITUTIONAL_AI,
    "evaluation_benchmarking":  EVALUATION_AND_BENCHMARKING,
    "glossary":                 GLOSSARY,
}


def list_sections() -> None:
    """Print all available section keys."""
    print("\nMODULE 09 · Available Sections")
    print("─" * 40)
    for key in SECTIONS:
        print(f"  {key}")
    print()


def read(section: str) -> None:
    """
    Print a section by key name.

    Parameters
    ----------
    section : str
        One of the keys in SECTIONS (e.g. 'rlhf', 'glossary').

    Examples
    --------
    >>> read("rlhf")
    >>> read("glossary")
    """
    content = SECTIONS.get(section)
    if content is None:
        print(f"Section '{section}' not found. Call list_sections() to see all keys.")
        return
    print(content)


def read_all() -> None:
    """Print every section in order."""
    for content in SECTIONS.values():
        print(content)
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No argument — print the full module
        read_all()
    elif sys.argv[1] == "--list":
        list_sections()
    else:
        # Print the requested section
        read(sys.argv[1])