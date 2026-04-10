"""
Attention Mechanism — Complete Concept Deep Dive
=================================================

The core operation that replaced recurrence and made modern LLMs possible.

9 Core Topics • 6 Extended Topics • ~9,800 words
"""

import textwrap
import re

TOPIC_NAME   = "Attention Mechanism"
DISPLAY_NAME = "08 · Attention Mechanism"
ICON         = "🔍"
SUBTITLE     = "Complete Concept Deep Dive"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Attention

┌─────────────────────────────────┬─────────────────────────────────────────────┐
│ Topic                           │ Summary                                     │
├─────────────────────────────────┼─────────────────────────────────────────────┤
│ 🔍 Attention Intuition          │ Why global token visibility is essential    │
│ 🗝️  Query · Key · Value         │ The Q, K, V projection matrices explained   │
│ ⚙️  Score Computation           │ Dot product → Scale → Softmax → Weighted Sum│
│ 🔀 Multi-Head Attention         │ Parallel attention across subspaces         │
│ 🚫 Causal Masking               │ Why future tokens are masked during training│
│ 📈 Attention Complexity         │ O(n²) — why long contexts are expensive     │
│ ⚡ Flash Attention               │ Memory-efficient attention via tiling        │
│ 🗜️  Grouped Query Attention     │ GQA — reducing KV cache memory usage        │
│ ↔️  Cross-Attention             │ Attending across two different sequences    │
│ 🧱 Positional Encoding          │ Sinusoidal, RoPE, ALiBi — injecting order   │
│ 💾 KV Cache                     │ Caching K,V at inference for fast generation│
│ 🌐 Sparse & Local Attention     │ BigBird, Longformer, Sliding Window         │
│ 📏 Scaled Dot-Product Detail    │ Why divide by √dk?                          │
│ 🏗️  Full Transformer Layer      │ Residuals, LayerNorm, FFN — the full picture│
│ 🎤 Interview Q&A                │ Key questions and complete answers          │
└─────────────────────────────────┴─────────────────────────────────────────────┘


##### 1. — ATTENTION INTUITION

Before diving into the mathematics, it is essential to understand why attention
was invented and what problem it solves. This section builds the conceptual
foundation upon which everything else rests.

### The Fixed-Window Problem

Traditional neural architectures for sequences — RNNs, LSTMs, and CNNs —
process tokens through a fixed-size window. Information from distant positions
must travel through every intermediate step. In an LSTM, a dependency between
a pronoun at token 200 and its antecedent at token 5 requires the model to
successfully carry a signal through 195 recurrent steps. Gradient signal decays
exponentially, and the information is progressively overwritten by intervening
content.

Language is full of constructions that span long distances. Consider the
Winograd schema:

    'The trophy didn't fit in the suitcase because it was too large.'

To resolve what 'it' refers to, the model must reason over the entire sentence.
RNNs routinely fail at this; attention handles it naturally.

### The Attention Solution

Attention eliminates the step-by-step bottleneck by allowing every token to
directly query every other token in a single forward pass. The path length
between any two tokens becomes O(1) instead of O(n). This architectural choice
is what makes Transformers so powerful for language.

┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY INSIGHT                                                                 │
│                                                                             │
│ Attention is a differentiable, parameterised soft-lookup: instead of        │
│ retrieving one item from a table by exact key match, it retrieves a         │
│ weighted blend of all values, where weights are learned similarities        │
│ between the query and each key.                                             │
└─────────────────────────────────────────────────────────────────────────────┘

### Three Mental Models

    Model 1 — Search Engine
        The Query is what you are searching for. Keys are the index entries
        each token broadcasts. Values are the content retrieved. Instead of
        returning one result, attention returns a soft weighted blend of all
        results.

    Model 2 — Dinner Table
        Every word at the table whispers to every other word simultaneously.
        Each word's final meaning is the weighted average of all the messages
        it received.

    Model 3 — Differentiable Dictionary
        A hash map does exact key lookup. Attention does soft nearest-neighbour
        search and returns an interpolated value — fully differentiable, so
        weights are learnable.

### Historical Roots

┌────────┬──────────────────┬───────────────────────────────────────────────┐
│ Year   │ Work             │ Contribution                                  │
├────────┼──────────────────┼───────────────────────────────────────────────┤
│ 2015   │ Bahdanau et al.  │ First attention for NMT. Decoder attends to   │
│        │                  │ encoder states rather than just the final     │
│        │                  │ vector. Additive/concat form.                 │
├────────┼──────────────────┼───────────────────────────────────────────────┤
│ 2015   │ Luong et al.     │ Simplified dot-product and general attention  │
│        │                  │ forms. Faster than Bahdanau.                  │
├────────┼──────────────────┼───────────────────────────────────────────────┤
│ 2017   │ Vaswani et al.   │ 'Attention Is All You Need' — removes         │
│        │                  │ recurrence entirely. Self-attention + FFN only│
├────────┼──────────────────┼───────────────────────────────────────────────┤
│ 2018+  │ BERT, GPT, T5…   │ Multi-head self-attention becomes the         │
│        │                  │ universal building block for language models. │
└────────┴──────────────────┴───────────────────────────────────────────────┘


##### 2 — QUERY, KEY, AND VALUE

Every attention layer projects its input into three separate vector spaces.
Understanding the distinct role of each projection is critical to understanding
what attention is actually computing.

### Conceptual Roles

┌──────────────┬──────────────────────────────────────────────────────────────┐
│ Query (Q)    │ The 'seeking' vector. Encodes what information the current   │
│              │ token is looking for from the rest of the sequence.          │
│              │ Q = X · Wᵀ  where Wᵀ ∈ ℝ^{d × d_k}                           │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Key (K)      │ The 'advertising' vector. Each token broadcasts what kind of │
│              │ information it contains. Similarity between Q and K          │
│              │ determines attention weight.  K = X · W^K                    │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Value (V)    │ The 'content' vector. Once attention weights are computed    │
│              │ from Q·Kᵀ, the output is a weighted sum of V vectors —       │
│              │ the actual information aggregated.  V = X · W^V              │
└──────────────┴──────────────────────────────────────────────────────────────┘

### Why Three Projections, Not Two?

Separating Key from Value gives the model an additional degree of freedom.
A token can advertise high relevance (via K) while delivering a nuanced,
transformed representation (via V). If K and V were the same, relevance and
content would be coupled — you could not independently control 'how much
attention I get' from 'what information I provide.'

### The Core Equation

    Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V

### Projection Parameters

┌─────────────────┬─────────────────┬──────────────────────────┬──────────────┐
│ Matrix          │ Shape           │ Role                     │ Param Count  │
├─────────────────┼─────────────────┼──────────────────────────┼──────────────┤
│ Wᵀ  (Query)     │ ℝ^{d × d_k}     │ Projects token embedding │ d × d_k      │
│                 │                 │ to query space           │              │
├─────────────────┼─────────────────┼──────────────────────────┼──────────────┤
│ W^K (Key)       │ ℝ^{d × d_k}     │ Projects token embedding │ d × d_k      │
│                 │                 │ to key space             │              │
├─────────────────┼─────────────────┼──────────────────────────┼──────────────┤
│ W^V (Value)     │ ℝ^{d × d_v}     │ Projects token embedding │ d × d_v      │
│                 │                 │ to value space           │              │
├─────────────────┼─────────────────┼──────────────────────────┼──────────────┤
│ W^O (Output)    │ ℝ^{hd_v × d}    │ Mixes multi-head outputs │ hd_v × d     │
│                 │                 │ back to model dim        │              │
└─────────────────┴─────────────────┴──────────────────────────┴──────────────┘

Total parameters for multi-head attention layer: 4d²  (when d_k = d_v = d/h).
This is constant regardless of the number of heads.

### Worked Example

Input: 'The cat sat on the mat' — 6 tokens, d=512, h=8, d_k=64.

    Step 1: Project.
        Q = X·Wᵀ  → shape (6, 64)
        K = X·W^K → shape (6, 64)
        V = X·W^V → shape (6, 64)

    Step 2: For 'sat' (row 3 of Q), compute dot products with all 6 Keys.
        High scores for 'cat' (subject) and 'mat' (object) — the verb's args.

    Step 3: Softmax over 6 scores gives attention weights. Weighted sum of V
        yields a representation of 'sat' that encodes who sat and where.


##### 3 — ATTENTION SCORE COMPUTATION

The computation proceeds in four deterministic steps: raw dot products, scaling,
softmax normalisation, and weighted aggregation. Each step has a specific
mathematical justification.

### Step 1: Dot Product (Raw Scores)

    S = Q · Kᵀ  ∈ ℝ^{n × n}

Matrix multiply Q ∈ ℝ^{n×d_k} by Kᵀ ∈ ℝ^{d_k×n} to get score matrix S ∈ ℝ^{n×n}.
Entry S[i,j] is the dot product between the query of token i and the key of
token j — a measure of directional similarity in d_k-dimensional space.

### Step 2: Scale by 1/√d_k

    S_scaled = S / √d_k

Without scaling, when d_k is large, dot products grow in magnitude — their
variance equals d_k for unit-normal Q and K vectors. Large values push softmax
into saturation (near one-hot distributions), causing near-zero gradients.
Dividing by √d_k normalises variance to 1 regardless of dimension, stabilising
training.

┌─────────────────────────────────────────────────────────────────────────────┐
│ Statistical derivation:                                                     │
│   If q_i, k_i ~ N(0,1) independently, then q·k = Σ q_i·k_i has variance     │
│   d_k. Standard deviation = √d_k. Dividing by √d_k gives std dev = 1.       │
└─────────────────────────────────────────────────────────────────────────────┘

### Step 3: Softmax Normalisation

    A[i,j] = exp(S_scaled[i,j]) / Σ_j exp(S_scaled[i,j])

Softmax is applied row-wise, producing a valid probability distribution over
positions for each query. Each row of A sums to 1. This creates competitive
normalisation: increasing attention on one token necessarily decreases it on
others, producing focused attention patterns where the model concentrates on
its most relevant context.

### Step 4: Weighted Sum

    O = A · V  where O ∈ ℝ^{n × d_v}

The output O[i] = Σ_j A[i,j] · V[j] is a convex combination of all Value
vectors, weighted by the attention distribution. The result is a contextualised
representation: each token's output encodes information from the most relevant
parts of the sequence.

### Common Misconceptions

┌───────────┬────────────────────────────────────────────────────────────────┐
│ Myth 1    │ Attention = correlation.                                       │
│           │ False. Attention weights are a learned, parameterised          │
│           │ similarity in a projected space — entirely different from      │
│           │ Pearson correlation.                                           │
├───────────┼────────────────────────────────────────────────────────────────┤
│ Myth 2    │ Higher raw score = more important.                             │
│           │ False. Importance is relative within a row after softmax.      │
│           │ A score of 3.0 gets near-zero attention if others are 5.0+.    │
├───────────┼────────────────────────────────────────────────────────────────┤
│ Myth 3    │ Attention explains the model.                                  │
│           │ Attention weights are a by-product of computation, not causal  │
│           │ explainers. Gradient-based attribution is needed for genuine   │
│           │ interpretability (Jain & Wallace, 2019).                       │
└───────────┴────────────────────────────────────────────────────────────────┘


##### 4 — MULTI-HEAD ATTENTION

A single attention head collapses all relational signals into one pattern.
Multi-head attention runs h independent attention heads in parallel, each in a
different d_k-dimensional subspace, then combines their outputs.

### Motivation

Language simultaneously encodes syntax (which verb governs which noun),
semantics (topical relatedness), discourse (coreference), and positional
structure. A single head must trade off between all of these. Multiple heads
allow each to specialise independently, capturing different aspects of the
input in parallel.

### Equations

    head_i = Attention(Q·W_iᵀ, K·W_i^K, V·W_i^V)

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W^O

Where W_iᵀ, W_i^K ∈ ℝ^{d×d_k} and W_i^V ∈ ℝ^{d×d_v} are per-head projection
matrices. W^O ∈ ℝ^{hd_v×d} mixes head outputs back to model dimension.

### Dimension Convention

┌────────────────────────────┬──────────────────────┬───────────────────────┐
│ Quantity                   │ Value (d=512, h=8)   │ Formula               │
├────────────────────────────┼──────────────────────┼───────────────────────┤
│ d_k per head               │ 64                   │ d / h                 │
├────────────────────────────┼──────────────────────┼───────────────────────┤
│ d_v per head               │ 64                   │ d / h  (usually)      │
├────────────────────────────┼──────────────────────┼───────────────────────┤
│ Concat output dim          │ 512                  │ h × d_v               │
├────────────────────────────┼──────────────────────┼───────────────────────┤
│ Total attn parameters      │ 4 × 512² = 1,048,576 │ 4d²                   │
└────────────────────────────┴──────────────────────┴───────────────────────┘

### What Heads Learn (Empirical Findings)

┌──────────────────┬──────────────────────────────────┬──────────────────────┐
│ Head Type        │ Pattern                          │ Evidence             │
├──────────────────┼──────────────────────────────────┼──────────────────────┤
│ Syntactic        │ Attends to dependency-linked     │ Clark et al. 2019    │
│                  │ tokens (verb→subject)            │ (BERT)               │
├──────────────────┼──────────────────────────────────┼──────────────────────┤
│ Positional       │ Always attends to previous or    │ Emergent offset      │
│                  │ next token                       │ patterns             │
├──────────────────┼──────────────────────────────────┼──────────────────────┤
│ Coreference      │ Links pronouns to antecedents    │ Discourse coherence  │
│                  │                                  │ heads                │
├──────────────────┼──────────────────────────────────┼──────────────────────┤
│ Rare-token       │ Attends to low-frequency,        │ Named entity focus   │
│                  │ semantically heavy tokens        │                      │
├──────────────────┼──────────────────────────────────┼──────────────────────┤
│ Delimiter        │ Attends to [CLS], [SEP], or BOS  │ Aggregation heads    │
└──────────────────┴──────────────────────────────────┴──────────────────────┘

### Implementation: Batched Parallel Computation

In practice, all h heads are computed simultaneously by reshaping tensors.
The Query matrix is reshaped from (B, n, d) to (B, h, n, d_k), enabling a
single batched matrix multiplication for all heads simultaneously — no
sequential loop is needed. This is what makes multi-head attention GPU-efficient.

##### 5 — CAUSAL MASKING

Autoregressive language models generate tokens sequentially:
P(x) = Π_i P(x_i | x_1…x_{i-1}).

Causal masking enforces this constraint during training, preventing token i
from attending to any token j > i.

### The Problem Without Masking

During training with teacher forcing, the entire ground-truth sequence is fed
as input simultaneously. Without a mask, position i would directly see position
i+1 in the same forward pass — the answer is visible in the input. The model
would learn to copy rather than predict, and would fail entirely at inference
time when future tokens are unavailable.

### The Mask

    S_masked[i,j] = S[i,j]   if j ≤ i
    S_masked[i,j] = −∞        if j > i

A lower-triangular binary mask where M[i,j]=1 for j≤i and 0 otherwise.
Before softmax, scores at masked positions are set to −∞.
After softmax: exp(−∞)=0, so future tokens get exactly zero attention weight.

    Visualised for a 4-token sequence:
        ┌───┬───┬───┬───┐
        │ 1 │ 0 │ 0 │ 0 │  ← token 1 sees only itself
        │ 1 │ 1 │ 0 │ 0 │  ← token 2 sees tokens 1–2
        │ 1 │ 1 │ 1 │ 0 │  ← token 3 sees tokens 1–3
        │ 1 │ 1 │ 1 │ 1 │  ← token 4 sees all tokens
        └───┴───┴───┴───┘
         t1  t2  t3  t4

┌─────────────────┬────────────────────────────────────────────────────────┐
│ Why −∞ not 0?   │ Setting to 0 before softmax gives softmax(0) = 1/n ≠ 0 │
│                 │ — the position still receives nonzero attention.       │
│                 │ Setting to −∞ ensures exp(−∞) = 0 exactly, completely  │
│                 │ excluding the position.                                │
└─────────────────┴────────────────────────────────────────────────────────┘

### Training vs Inference

┌────────────────────┬──────────────────────────────────┬───────────────────┐
│ Phase              │ Masking                          │ Benefit           │
├────────────────────┼──────────────────────────────────┼───────────────────┤
│ Training           │ Full causal mask applied; all    │ Parallel: one fwd │
│                    │ positions computed in parallel   │ pass → n preds    │
├────────────────────┼──────────────────────────────────┼───────────────────┤
│ Inference (naive)  │ Only past tokens in input;       │ Token-by-token    │
│                    │ mask not strictly needed         │ generation        │
├────────────────────┼──────────────────────────────────┼───────────────────┤
│ Inference (KV$)    │ Past K,V cached; new token's Q   │ Efficient: only   │
│                    │ attends to cache. Mask implicit. │ new token/step    │
└────────────────────┴──────────────────────────────────┴───────────────────┘

### Bidirectional vs Causal: Architecture Comparison

┌──────────────┬───────────────────────┬──────────────────────┬─────────────┐
│ Model        │ Attention Type        │ Training Objective   │ Best For    │
├──────────────┼───────────────────────┼──────────────────────┼─────────────┤
│ BERT         │ Full bidirectional    │ Masked LM (MLM)      │ Classific., │
│              │ (no mask)             │                      │ NER, QA     │
├──────────────┼───────────────────────┼──────────────────────┼─────────────┤
│ GPT-family   │ Causal masked         │ Next-token pred.     │ Text gen.   │
│              │ self-attention        │                      │             │
├──────────────┼───────────────────────┼──────────────────────┼─────────────┤
│ T5 / BART    │ Encoder: bidir.;      │ Seq2seq denoising    │ Translation,│
│              │ Decoder: causal +     │                      │ summarise   │
│              │ cross-attn            │                      │             │
└──────────────┴───────────────────────┴──────────────────────┴─────────────┘


##### 6 — ATTENTION COMPLEXITY

The quadratic scaling of standard attention in sequence length is the central
bottleneck for long-context language models. Understanding exactly what scales
quadratically — and why — is essential for appreciating Flash Attention, GQA,
and sparse methods.

### Where O(n²) Comes From

┌───────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Operation         │ FLOPs            │ Memory           │ Why              │
├───────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Q·Kᵀ matmul       │ O(n² · d_k)      │ O(n²)            │ n×n score matrix │
├───────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Softmax (row)     │ O(n²)            │ O(n²)            │ Requires full    │
│                   │                  │                  │ n×n matrix       │
├───────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ A·V matmul        │ O(n² · d_v)      │ O(n²)            │ Same order       │
├───────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Total per layer   │ O(n² · d)        │ O(n² · h)        │ Summing all      │
└───────────────────┴──────────────────┴──────────────────┴──────────────────┘

### Concrete Memory Numbers

┌──────────────┬───────────────────────┬──────────────────────┬───────────────┐
│ Context Len  │ Model                 │ Attn Matrix (FP32,   │ Verdict       │
│              │                       │ h=12)                │               │
├──────────────┼───────────────────────┼──────────────────────┼───────────────┤
│ 512          │ BERT-base             │ ~12 MB               │ Trivial       │
├──────────────┼───────────────────────┼──────────────────────┼───────────────┤
│ 2,048        │ GPT-2 Large           │ ~600 MB              │ Manageable    │
├──────────────┼───────────────────────┼──────────────────────┼───────────────┤
│ 8,192        │ LLaMA-2 context       │ ~8.5 GB              │ Expensive     │
├──────────────┼───────────────────────┼──────────────────────┼───────────────┤
│ 128,000      │ Claude 3 / GPT-4T     │ ~2 TB (naive)        │ Impossible    │
│              │                       │                      │ w/o FlashAttn │
├──────────────┼───────────────────────┼──────────────────────┼───────────────┤
│ 1,000,000    │ Gemini 1.5 Pro        │ Petabyte-scale       │ Requires      │
│              │                       │                      │ ring attention│
└──────────────┴───────────────────────┴──────────────────────┴───────────────┘

### The Memory Bandwidth Problem

Even if FLOPs were free, the n×n attention matrix must be read and written to
GPU HBM (High Bandwidth Memory). HBM bandwidth is 10–50× slower than on-chip
SRAM. For long sequences, the bottleneck is not computation but data movement —
this is precisely what Flash Attention targets.

### Approaches to Reduce Complexity

┌──────────────────┬───────────────┬─────────────┬────────────┬─────────────┐
│ Approach         │ FLOPs         │ Memory      │ Quality    │ Example     │
├──────────────────┼───────────────┼─────────────┼────────────┼─────────────┤
│ Flash Attention  │ O(n²d)        │ O(n)        │ Exact      │ All modern  │
│                  │               │             │            │ LLMs        │
├──────────────────┼───────────────┼─────────────┼────────────┼─────────────┤
│ Sparse / Local   │ O(n·w·d)      │ O(n·w)      │ Approx.    │ Mistral,    │
│                  │               │             │            │ BigBird     │
├──────────────────┼───────────────┼─────────────┼────────────┼─────────────┤
│ Linear Attention │ O(n·d²)       │ O(d²)       │ Approx.    │ Performer   │
├──────────────────┼───────────────┼─────────────┼────────────┼─────────────┤
│ GQA / MQA        │ O(n²d)        │ O(n·G·d_k)  │ Exact      │ LLaMA-2/3   │
│                  │               │             │ (less KV)  │             │
└──────────────────┴───────────────┴─────────────┴────────────┴─────────────┘

##### 7 — FLASH ATTENTION

Flash Attention (Tri Dao et al., 2022) computes exact attention without ever
materialising the full n×n attention matrix in GPU HBM. It achieves the same
mathematical result as standard attention but with O(n) HBM memory and
significantly fewer HBM reads.

### The Core Problem

Standard attention writes the n×n score matrix S and attention weight matrix A
to HBM. For n=8192 with 32 heads, this is ~8.5 GB of HBM traffic per layer per
forward pass. HBM bandwidth on an A100 is ~2 TB/s — at batch size 1, this
dominates latency. The n×n matrix also consumes GPU memory that could hold
longer sequences or larger batches.

### The Tiling Strategy

┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY IDEA                                                                    │
│                                                                             │
│ Divide Q into row-blocks of size B_r and K, V into column-blocks of size    │
│ B_c. Process one tile (Q_i, K_j) at a time, keeping it entirely in on-chip  │
│ SRAM. The n×n matrix is never written to HBM.                               │
└─────────────────────────────────────────────────────────────────────────────┘

    GPU HBM                          On-chip SRAM (fast)
    ─────────────────────            ──────────────────────────────
    Q  [B, n, d]  ──tile──►          Q_i  [B_r, d_k]
    K  [B, n, d]  ──tile──►          K_j  [B_c, d_k]   ← computed here
    V  [B, n, d]  ──tile──►          V_j  [B_c, d_v]
                                          ↓
    O  [B, n, d]  ◄──write──         O_i  (partial output, rescaled)

### Online Softmax: The Mathematical Trick

The difficulty with tiling is that softmax requires the global row maximum for
numerical stability and the global sum for normalisation. Flash Attention
resolves this with an online algorithm that maintains running statistics:

    •   m_i  — running maximum of scores seen so far for row i
    •   ℓ_i  — running sum of exp(scores − m_i) for row i
    •   O_i  — running weighted sum of Values for row i

When a new tile is processed, m_i and ℓ_i are updated and the previous partial
output O_i is rescaled to account for the updated normalisation. After all tiles
are processed, the result is numerically equivalent to standard attention.

    O_i ← diag(ℓ_old / ℓ_new) · O_i + new_tile_contribution

### Memory & IO Analysis

┌────────────────────────────┬────────────────────────┬───────────────────────┐
│ Property                   │ Standard Attention     │ Flash Attention v2    │
├────────────────────────────┼────────────────────────┼───────────────────────┤
│ HBM memory for attn matrix │ O(n²)                  │ O(1) — never stored   │
├────────────────────────────┼────────────────────────┼───────────────────────┤
│ Total HBM memory           │ O(n²)                  │ O(n·d)                │
├────────────────────────────┼────────────────────────┼───────────────────────┤
│ HBM reads/writes           │ O(n²)                  │ O(n²/B) ≈ O(n²/d)     │
├────────────────────────────┼────────────────────────┼───────────────────────┤
│ FLOPs                      │ O(n²·d)                │ O(n²·d) — exact same  │
├────────────────────────────┼────────────────────────┼───────────────────────┤
│ A100 utilisation           │ ~35% of peak           │ ~70% of peak          │
├────────────────────────────┼────────────────────────┼───────────────────────┤
│ Numerically equivalent?    │ Yes (baseline)         │ Yes (exact same)      │
└────────────────────────────┴────────────────────────┴───────────────────────┘

### Backward Pass: Recomputation

Flash Attention does not store the n×n attention matrix for backpropagation.
Instead, it recomputes the attention scores from stored Q, K, V during the
backward pass. This trades a moderate increase in FLOPs (~33% extra) for a
massive reduction in activation memory, enabling much larger batch sizes and
longer sequences.

### Version History & Impact

┌───────────────────────┬──────┬───────────────────────────────────┬──────────┐
│ Version               │ Year │ Key Improvement                   │ Speedup  │
├───────────────────────┼──────┼───────────────────────────────────┼──────────┤
│ Flash Attention v1    │ 2022 │ Original tiled kernel; 2–4×       │ Baseline │
│                       │      │ speedup; 10–20× memory reduction  │          │
├───────────────────────┼──────┼───────────────────────────────────┼──────────┤
│ Flash Attention v2    │ 2023 │ Better warp work distribution;    │ ~2× v1   │
│                       │      │ parallelism over seq dim          │          │
├───────────────────────┼──────┼───────────────────────────────────┼──────────┤
│ Flash Attention v3    │ 2024 │ H100-specific: FP8, warp          │ ~2× v2   │
│                       │      │ specialisation, async pipeline    │          │
├───────────────────────┼──────┼───────────────────────────────────┼──────────┤
│ FlexAttention (torch) │ 2024 │ Programmable via torch.compile;   │ ≈ v2     │
│                       │      │ custom masks/biases               │          │
└───────────────────────┴──────┴───────────────────────────────────┴──────────┘

Flash Attention is now standard in virtually all production LLMs: GPT-4, Claude,
LLaMA, Mistral, Gemini. It is the enabling technology for practical
long-context inference.


##### 8 — GROUPED QUERY ATTENTION (GQA)

GQA reduces the KV cache memory required at inference by sharing Key and Value
heads across groups of Query heads. It interpolates between Multi-Head Attention
(full quality) and Multi-Query Attention (minimum memory).

### The KV Cache Problem

During autoregressive generation, Key and Value tensors for all past tokens must
be retained in GPU memory so they are not recomputed at every step. This KV
cache grows linearly with sequence length and is often the primary memory
bottleneck at inference, limiting batch size and throughput.

    KV cache size = 2 × n_layers × n_kv_heads × d_head × seq_len × bytes/param

For LLaMA-3 70B in BF16 with n_kv=8, d_head=128, 80 layers:
    •   ~0.33 MB per token
    •   At 4,096 tokens:  ~1.3 GB
    •   At 128k tokens:   ~41 GB  (close to full GPU memory budget)

### MHA vs MQA vs GQA

┌──────────────────┬──────────────┬──────────────┬────────────┬─────────────┐
│ Variant          │ n_kv_heads   │ KV Cache     │ Quality    │ Used By     │
├──────────────────┼──────────────┼──────────────┼────────────┼─────────────┤
│ Multi-Head (MHA) │ = n_q (e.g.  │ Full         │ ★★★★★    │ BERT, GPT-2,│
│                  │ 32)          │ (baseline)   │            │ orig. Trans.│
├──────────────────┼──────────────┼──────────────┼────────────┼─────────────┤
│ Multi-Query(MQA) │ 1            │ 1/n_q of MHA │ ★★★☆☆    │ Falcon 7B,  │
│                  │              │              │ (degraded) │ early Gemma │
├──────────────────┼──────────────┼──────────────┼────────────┼─────────────┤
│ Grouped Q. (GQA) │ G (e.g. 8)   │ G/n_q of MHA │ ★★★★★    │ LLaMA-2/3   │
│                  │              │              │ ≈MHA       │ 70B, Mistral│
└──────────────────┴──────────────┴──────────────┴────────────┴─────────────┘

### GQA Equations

    kv_index(i) = floor( i / (n_q_heads / n_kv_heads) )

For query head i, the corresponding KV head index is simply integer division
by the group size. All query heads within a group share identical Key and Value
projections, reducing KV cache by a factor of n_q / G.

    KV cache reduction factor = n_q_heads / G

### Empirical Results (Ainslie et al. 2023)

GQA with G=8 (n_q=32 or n_q=64) matches Multi-Head Attention quality on
virtually all benchmarks while providing near-Multi-Query memory efficiency.
The sweet spot: G between n_q/4 and n_q/8. GQA has since become the standard
choice for large models.

┌──────────────────┬────────────┬─────────────┬────────────┬────────────────┐
│ Model            │ n_q_heads  │ n_kv_heads  │ Group Size │ KV Reduction   │
├──────────────────┼────────────┼─────────────┼────────────┼────────────────┤
│ LLaMA-2 70B      │ 64         │ 8           │ 8          │ 8×             │
├──────────────────┼────────────┼─────────────┼────────────┼────────────────┤
│ LLaMA-3 8B       │ 32         │ 8           │ 4          │ 4×             │
├──────────────────┼────────────┼─────────────┼────────────┼────────────────┤
│ LLaMA-3 70B      │ 64         │ 8           │ 8          │ 8×             │
├──────────────────┼────────────┼─────────────┼────────────┼────────────────┤
│ Mistral 7B       │ 32         │ 8           │ 4          │ 4×             │
├──────────────────┼────────────┼─────────────┼────────────┼────────────────┤
│ Gemma 9B         │ 16         │ 8           │ 2          │ 2×             │
└──────────────────┴────────────┴─────────────┴────────────┴────────────────┘

##### 9 — CROSS-ATTENTION

Cross-attention is the mechanism by which one sequence attends to another. It
uses the same mathematical operation as self-attention but draws Queries from
one sequence and Keys/Values from a different sequence.

### Self-Attention vs Cross-Attention

┌───────────────────────┬──────────────────────────┬─────────────────────────┐
│ Property              │ Self-Attention           │ Cross-Attention         │
├───────────────────────┼──────────────────────────┼─────────────────────────┤
│ Q source              │ Same sequence X          │ Sequence A (e.g. dec.)  │
├───────────────────────┼──────────────────────────┼─────────────────────────┤
│ K, V source           │ Same sequence X          │ Sequence B (e.g. enc.)  │
├───────────────────────┼──────────────────────────┼─────────────────────────┤
│ Score matrix shape    │ n × n  (square)          │ n_tgt × n_src (rect.)   │
├───────────────────────┼──────────────────────────┼─────────────────────────┤
│ Causal mask?          │ Yes (decoder) / No (enc.)│ Typically no            │
├───────────────────────┼──────────────────────────┼─────────────────────────┤
│ Primary use           │ Within-sequence context  │ Seq2seq conditioning    │
└───────────────────────┴──────────────────────────┴─────────────────────────┘

### Equations

    Q = decoder_hidden · Wᵀ    (shape: n_tgt × d_k)
    K = encoder_output · W^K   (shape: n_src × d_k)
    V = encoder_output · W^V   (shape: n_src × d_v)

    O = softmax( Q·Kᵀ / √d_k ) · V    (shape: n_tgt × d_v)

### Use Cases

┌────────────────────┬──────────────────┬──────────────────┬────────────────┐
│ Domain             │ Q Source         │ K/V Source       │ What It Learns │
├────────────────────┼──────────────────┼──────────────────┼────────────────┤
│ Machine Trans.     │ Decoder (tgt     │ Encoder (src     │ Which source   │
│                    │ lang token)      │ lang)            │ words to trans.│
├────────────────────┼──────────────────┼──────────────────┼────────────────┤
│ Image Captioning   │ Text decoder tok │ Image patch embs │ Which regions  │
│                    │                  │ (ViT)            │ to describe    │
├────────────────────┼──────────────────┼──────────────────┼────────────────┤
│ Speech Recognition │ Text decoder tok │ Audio frame embs │ Which frames   │
│                    │                  │                  │ to transcribe  │
├────────────────────┼──────────────────┼──────────────────┼────────────────┤
│ Diffusion (txt2img)│ U-Net spatial    │ Text token embs  │ Spatial cond.  │
│                    │ features         │ (CLIP)           │ on text        │
├────────────────────┼──────────────────┼──────────────────┼────────────────┤
│ Video QA           │ Question token   │ Video frame feats│ Which frames   │
│                    │                  │                  │ answer the Q   │
└────────────────────┴──────────────────┴──────────────────┴────────────────┘

### Implementation Notes

    •   No causal mask: The decoder can freely attend to all encoder positions.
        Masking the encoder would discard information unnecessarily.

    •   Static KV cache: The encoder runs once per input. Its K, V tensors are
        computed once and reused at every decoder layer and every decoding step,
        unlike the growing self-attention KV cache.

    •   Decoder-only alternative: Modern decoder-only LLMs (GPT, LLaMA) avoid
        cross-attention by prepending the source as a prefix in the same
        sequence, using causal self-attention throughout. Simpler architecture
        but less memory-efficient for very long sources.


##### 10 — POSITIONAL ENCODING  [Extended]

Attention is permutation-equivariant: shuffling tokens produces the same
attention scores (just reordered). Positional encoding injects order information
so the model can distinguish 'dog bites man' from 'man bites dog'.

### Sinusoidal (Original Transformer)

    PE(pos, 2i)   = sin( pos / 10000^{2i/d} )
    PE(pos, 2i+1) = cos( pos / 10000^{2i/d} )

Added to token embeddings before the first layer. Fixed (not learned). Produces
a unique position fingerprint at each frequency. Can generalise to unseen
sequence lengths, though quality degrades.

### Learned Absolute PE

Replace fixed sinusoids with a learned embedding table of shape (max_len, d).
Used in BERT and GPT-1/2. Cannot extrapolate beyond max_len seen during training
— a fundamental limitation for long-context extension.

### RoPE — Rotary Position Embeddings

Encodes position by rotating Q and K vectors in 2D subspaces:
    q_rotated = R(pos) · q

The dot product Q·K then naturally encodes relative position.

Advantages: relative position is implicit, no position vectors added to
embeddings, better length extrapolation. Used in LLaMA, PaLM, Gemma, Mistral.

    (Q_rot · K_rot)[i,j] depends only on (i - j) — relative distance

### ALiBi — Attention with Linear Biases

Subtracts a linear penalty proportional to distance directly from attention
scores before softmax. No position vectors or rotations — purely a bias term:

    S_biased[i,j] = Q[i]·K[j] / √d_k − m · (i − j)

Strong length extrapolation: trained on short sequences, evaluates on longer
ones with graceful degradation.

##### 11 — KV CACHE  [Extended]

The KV cache is the memory structure that makes autoregressive generation
computationally practical. Without it, generating each new token would require
reprocessing the entire context from scratch.

### Mechanism

At inference step t, the model computes K_t and V_t for the new token and
appends them to the cache:
    K_cache = [K_1, …, K_t]
    V_cache = [V_1, …, V_t]

The new token's Query attends over the entire cache. Only one new token is
processed per step.

### Memory & Bandwidth Impact

┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Property                     │ Detail                                       │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ Size formula                 │ 2 × n_layers × n_kv_heads × d_head × seq_len │
│                              │ × bytes                                      │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ LLaMA-3 70B (BF16)           │ ~0.33 MB/token; 41 GB at 128k tokens         │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ Throughput impact            │ Each decode step reads entire KV cache from  │
│                              │ HBM — bandwidth bottleneck                   │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ Mitigation: GQA              │ Reduce n_kv_heads (e.g. 8 vs 64) → 8× cache  │
│                              │ reduction                                    │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ Mitigation: quantisation     │ INT8/INT4 KV cache: 2–4× reduction with      │
│                              │ minor quality loss                           │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ Mitigation: PagedAttention   │ vLLM manages KV cache in virtual pages;      │
│                              │ eliminates fragmentation                     │
└──────────────────────────────┴──────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
12 — SPARSE & LOCAL ATTENTION  [Extended]
──────────────────────────────────────────────────────────────────────────────

For sequences too long for full attention even with Flash Attention, sparse
attention patterns reduce FLOPs from O(n²) to O(n·w) or O(n√n) by computing
only a subset of pairwise interactions.

┌──────────────────┬──────────────────┬──────────────────────┬───────────────┐
│ Method           │ Complexity       │ Pattern              │ Used By       │
├──────────────────┼──────────────────┼──────────────────────┼───────────────┤
│ Sliding Window   │ O(n·w·d)         │ Each token attends   │ Mistral 7B    │
│                  │                  │ to [i-w, i] only     │ (w=4096)      │
├──────────────────┼──────────────────┼──────────────────────┼───────────────┤
│ Dilated/Strided  │ O(n·w·d)         │ Attend to every k-th │ Sparse Trans. │
│                  │                  │ token; incr. dilation│               │
├──────────────────┼──────────────────┼──────────────────────┼───────────────┤
│ BigBird /        │ O(n·d)           │ Local + global tokens│ Document NLP  │
│ Longformer       │                  │ + random             │ tasks         │
├──────────────────┼──────────────────┼──────────────────────┼───────────────┤
│ Axial Attention  │ O(n^1.5 · d)     │ Attend along rows    │ Image gen.    │
│                  │                  │ then cols separately │               │
├──────────────────┼──────────────────┼──────────────────────┼───────────────┤
│ MoA (Mixture)    │ Mixed            │ Some heads full,     │ Hybrid appr.  │
│                  │                  │ some sparse          │               │
└──────────────────┴──────────────────┴──────────────────────┴───────────────┘

Key trade-off: sparse methods reduce FLOPs but may miss long-range dependencies
that fall outside the sparse pattern. Stacking multiple sparse-attention layers
can recover some long-range capacity, as information propagates through
intermediate hops.

##### 13 — FULL TRANSFORMER LAYER ARCHITECTURE

Attention is one sub-layer within each Transformer block. Understanding how it
sits within the full architecture is essential for implementation and debugging.

### Layer Structure (Pre-LN, modern)

    x ← x + MultiHeadAttn( LayerNorm(x) )
    x ← x + FFN( LayerNorm(x) )

Each Transformer layer contains two sub-layers: Multi-Head Self-Attention and
a Position-wise Feed-Forward Network (FFN). Both are wrapped with a residual
connection and Layer Normalisation.

    ┌───────────────────────────────────────────────────────┐
    │  Input x                                              │
    │    │                                                  │
    │    ├─────────────────────────┐                        │
    │    ▼                         │  (residual)            │
    │  LayerNorm(x)                │                        │
    │    ▼                         │                        │
    │  MultiHeadAttention          │                        │
    │    ▼                         │                        │
    │  + ◄─────────────────────────┘                        │
    │    │                                                  │
    │    ├─────────────────────────┐                        │
    │    ▼                         │  (residual)            │
    │  LayerNorm                   │                        │
    │    ▼                         │                        │
    │  FFN (MLP)                   │                        │
    │    ▼                         │                        │
    │  + ◄─────────────────────────┘                        │
    │    ▼                                                  │
    │  Output x                                             │
    └───────────────────────────────────────────────────────┘

### Pre-LN vs Post-LN

┌────────────────┬───────────────────────┬──────────────────┬───────────────┐
│ Variant        │ Order                 │ Stability        │ Used By       │
├────────────────┼───────────────────────┼──────────────────┼───────────────┤
│ Post-LN        │ Sublayer → Add → LN   │ Unstable for     │ Original      │
│ (original)     │                       │ deep models;     │ Transformer   │
│                │                       │ requires warm-up │ (2017)        │
├────────────────┼───────────────────────┼──────────────────┼───────────────┤
│ Pre-LN         │ LN → Sublayer → Add   │ Stable; no       │ GPT, LLaMA,   │
│ (modern)       │                       │ special LR sched │ most modern   │
│                │                       │ needed           │ LLMs          │
└────────────────┴───────────────────────┴──────────────────┴───────────────┘

### Feed-Forward Network Sub-layer

    FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2        [ReLU variant]

    FFN(x) = (x·W_1 ⊙ SiLU(x·W_g)) · W_2             [SwiGLU, LLaMA]

FFN hidden dimension is typically 4d (or ~8/3 × d for SwiGLU). The FFN contains
~2/3 of all parameters in a Transformer and is believed to store factual
knowledge (Geva et al. 2021: 'Transformer FFN Layers Are Key-Value Memories').

    Roles:
        Attention = token mixing     (communication between positions)
        FFN       = token processing (per-position transformation)

This division is formalised in the 'Mathematical Framework for Transformer
Circuits' (Elhage et al. 2021).


###### 14 — INTERVIEW QUESTIONS & MODEL ANSWERS

┌───────────────────────────────────────────────────────────────────────────┐
│ Q1: Why do we scale attention scores by 1/√d_k?                           │
├───────────────────────────────────────────────────────────────────────────┤
│ If Q and K are drawn from N(0,1), their dot product has variance d_k, so  │
│ standard deviation √d_k. Without scaling, large d_k causes dot products   │
│ to grow, pushing softmax into saturation (near one-hot), which produces   │
│ near-zero gradients and stalls training. Dividing by √d_k normalises      │
│ variance to 1 regardless of dimension.                                    │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q2: What is O(n²) complexity in attention and how is it addressed?        │
├───────────────────────────────────────────────────────────────────────────┤
│ The Q·Kᵀ matrix multiply and softmax both require O(n²) FLOPs and O(n²)   │
│ memory because every pair of tokens interacts. Addressed by:              │
│   (1) Flash Attention — same FLOPs but O(n) HBM memory via tiling         │
│   (2) Sparse attention — fewer FLOPs but approximate                      │
│   (3) GQA/MQA — reduces KV cache at inference, not training FLOPs         │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q3: What is the difference between MHA, MQA, and GQA?                     │
├───────────────────────────────────────────────────────────────────────────┤
│ MHA: each query head has its own K and V heads (full quality, full memory)│
│ MQA: all query heads share a single K and V head (max savings, quality↓)  │
│ GQA: query heads divided into G groups, each sharing one K/V pair —       │
│   interpolates between MQA and MHA. G=8 achieves near-MHA quality with    │
│   4–8× KV cache reduction.                                                │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q4: What is the difference between self-attention and cross-attention?    │
├───────────────────────────────────────────────────────────────────────────┤
│ Self-attention: Q, K, and V all derived from the same sequence.           │
│ Cross-attention: Q from one sequence (e.g. decoder), K and V from a       │
│ different sequence (e.g. encoder output). Same mathematical operation,    │
│ different data sources. Self-attention captures within-sequence           │
│ dependencies; cross-attention enables conditioning on a separate sequence.│
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q5: Why does causal masking use −∞ instead of 0?                          │
├───────────────────────────────────────────────────────────────────────────┤
│ Setting a score to 0 before softmax gives softmax(0) = 1/n ≠ 0 — the      │
│ position still receives nonzero attention. Setting to −∞ gives            │
│ exp(−∞) = 0, so after softmax the position gets exactly zero weight.      │
│ It is completely excluded from the weighted sum of Values.                │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q6: How does Flash Attention avoid materialising the n×n matrix?          │
├───────────────────────────────────────────────────────────────────────────┤
│ Flash Attention tiles Q, K, V into blocks that fit in on-chip SRAM. It    │
│ processes one (Q_i, K_j) tile at a time, computing partial attention      │
│ outputs and maintaining running statistics (max and sum) for the online   │
│ softmax. The final output is mathematically equivalent to standard        │
│ attention but the full n×n matrix never exists in HBM — only SRAM-sized   │
│ tiles at any given time.                                                  │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q7: What do different attention heads learn?                              │
├───────────────────────────────────────────────────────────────────────────┤
│ Empirically, heads specialise: syntactic heads attend to                  │
│ dependency-linked tokens (verb→subject), positional heads attend to fixed │
│ offsets (previous/next token), coreference heads link pronouns to         │
│ antecedents, and rare-token heads focus on semantically salient tokens.   │
│ This specialisation is emergent from training, not designed. Many heads   │
│ can be pruned with minimal quality loss (Michel et al. 2019).             │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Q8: Why is RoPE preferred over learned absolute positional embeddings?    │
├───────────────────────────────────────────────────────────────────────────┤
│ Learned absolute PE cannot extrapolate beyond the max sequence length     │
│ seen during training. RoPE encodes relative position implicitly via       │
│ rotation of Q and K vectors, so the dot product Q·K depends only on the   │
│ relative distance (i-j). This allows better length generalisation at      │
│ inference time. RoPE is now standard in LLaMA, Gemma, Mistral, and most   │
│ recent large models.                                                      │
└───────────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
QUICK REFERENCE: COMPLEXITY CHEATSHEET
──────────────────────────────────────────────────────────────────────────────

┌─────────────────────┬───────────────┬───────────────┬─────────┬────────────┐
│ Attention Type      │ Time          │ KV Cache Mem  │ Quality │ Key Paper  │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ Standard MHA        │ O(n²·d)       │ O(n·d)        │ ★★★★★ │ Vaswani'17 │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ Multi-Query (MQA)   │ O(n²·d)       │ O(n·d_k)      │ ★★★☆☆ │ Shazeer'19 │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ Grouped Query (GQA) │ O(n²·d)       │ O(n·G·d_k)    │ ★★★★★ │ Ainslie'23 │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ Flash Attention v2  │ O(n²·d) FLOPs │ O(n·d) HBM    │ ★★★★★ │ Tri Dao'23 │
│                     │               │               │ (exact) │            │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ Sliding Window      │ O(n·w·d)      │ O(w·d)        │ ★★★☆☆ │ Mistral'23 │
│                     │               │               │ (local) │            │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ BigBird             │ O(n·d)        │ O(n·d) approx │ ★★★★☆ │ Zaheer'20  │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ Linear Attention    │ O(n·d²)       │ O(d²)         │ ★★★☆☆ │ Choro.'21  │
├─────────────────────┼───────────────┼───────────────┼─────────┼────────────┤
│ Cross-Attention     │ O(n_s·n_t·d)  │ O(n_src·d)    │ ★★★★★ │ Vaswani'17 │
│                     │               │ static        │         │            │
└─────────────────────┴───────────────┴───────────────┴─────────┴────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS  (runnable code snippets)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "Scaled Dot-Product Attention (NumPy)": """
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    \"\"\"
    Args:
        Q: (n, d_k)  — Query matrix
        K: (n, d_k)  — Key matrix
        V: (n, d_v)  — Value matrix
        mask: (n, n) boolean, True = masked (future token), optional
    Returns:
        output: (n, d_v)
        weights: (n, n) attention weights
    \"\"\"
    d_k = Q.shape[-1]

    # Step 1: Raw scores
    scores = Q @ K.T                          # (n, n)

    # Step 2: Scale
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply causal mask (set future positions to -inf)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)

    # Step 4: Softmax (row-wise)
    scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)  # (n, n)

    # Step 5: Weighted sum of Values
    output = weights @ V                      # (n, d_v)
    return output, weights

# Example: 4 tokens, d_k=8, d_v=8
n, d_k, d_v = 4, 8, 8
np.random.seed(42)
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
V = np.random.randn(n, d_v)

# Causal mask: mask[i,j]=True means token i cannot attend to j
mask = np.triu(np.ones((n, n), dtype=bool), k=1)
output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

print("Attention weights (causal):")
print(np.round(weights, 3))
print("Each row should sum to 1:", np.allclose(weights.sum(axis=-1), 1))
print("Output shape:", output.shape)
""",

    "Multi-Head Attention (PyTorch)": """
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        # (B, n, d_model) → (B, h, n, d_k)
        B, n, _ = x.shape
        return x.view(B, n, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        B, n, _ = x.shape
        Q = self.split_heads(self.W_q(x))   # (B, h, n, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Scaled dot-product attention (all heads in parallel)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B,h,n,n)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)                 # (B,h,n,n)

        context = attn_weights @ V                               # (B,h,n,d_k)
        context = context.transpose(1, 2).contiguous()          # (B,n,h,d_k)
        context = context.view(B, n, self.d_model)              # (B,n,d_model)

        return self.W_o(context), attn_weights

# Usage
d_model, num_heads, seq_len, batch = 512, 8, 20, 2
mha = MultiHeadAttention(d_model, num_heads)
x   = torch.randn(batch, seq_len, d_model)

# Causal mask
mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

out, weights = mha(x, mask=mask)
print("Output shape:  ", out.shape)       # (2, 20, 512)
print("Weights shape: ", weights.shape)   # (2, 8, 20, 20)
print("Param count:   ", sum(p.numel() for p in mha.parameters()))  # 4 × d²
""",

    "Grouped Query Attention (GQA)": """
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    \"\"\"
    GQA: n_q_heads query heads share n_kv_heads key/value heads.
    Set n_kv_heads=1 for MQA; n_kv_heads=n_q_heads for standard MHA.
    \"\"\"
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q  = n_q_heads
        self.n_kv = n_kv_heads
        self.d_k  = d_model // n_q_heads
        self.g    = n_q_heads // n_kv_heads  # group size

        self.W_q = nn.Linear(d_model, n_q_heads  * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model,                bias=False)

    def forward(self, x):
        B, n, _ = x.shape
        Q = self.W_q(x).view(B, n, self.n_q,  self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, n, self.n_kv, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, n, self.n_kv, self.d_k).transpose(1, 2)

        # Expand K, V so each query head has a corresponding KV head
        K = K.repeat_interleave(self.g, dim=1)  # (B, n_q, n, d_k)
        V = V.repeat_interleave(self.g, dim=1)

        scores  = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out     = (weights @ V).transpose(1, 2).contiguous()
        out     = out.view(B, n, -1)
        return self.W_o(out)

# Compare KV cache sizes
d_model, n_q, n_kv, seq_len = 512, 32, 8, 2048
gqa = GroupedQueryAttention(d_model, n_q, n_kv)
x   = torch.randn(1, seq_len, d_model)
out = gqa(x)
print("Output shape:", out.shape)

d_k    = d_model // n_q
kv_mha = 2 * seq_len * n_q  * d_k
kv_gqa = 2 * seq_len * n_kv * d_k
print(f"MHA  KV cache: {kv_mha:,} elements  ({kv_mha*2/1e6:.2f} MB BF16)")
print(f"GQA  KV cache: {kv_gqa:,} elements  ({kv_gqa*2/1e6:.2f} MB BF16)")
print(f"Reduction: {n_q // n_kv}×")
""",

    "Causal Mask Visualisation": """
import numpy as np

def make_causal_mask(seq_len):
    \"\"\"Lower-triangular mask — 1 = attend, 0 = blocked\"\"\"
    return np.tril(np.ones((seq_len, seq_len), dtype=int))

def visualise_mask(mask, tokens=None):
    n = mask.shape[0]
    header = "      " + "  ".join(f"t{j+1}" for j in range(n))
    print(header)
    print("    " + "─" * (n * 4))
    for i in range(n):
        label = f"t{i+1}" if tokens is None else tokens[i][:3]
        row   = "  ".join("✓" if mask[i,j] else "✗" for j in range(n))
        print(f"  {label:3s} │ {row}")

tokens = ["The", "cat", "sat", "on", "the", "mat"]
mask   = make_causal_mask(len(tokens))

print("Causal Attention Mask  (✓ = can attend, ✗ = blocked)")
print("=" * 45)
visualise_mask(mask, tokens)
print()
print("Token 'sat' can attend to:", [tokens[j] for j in range(6) if mask[2,j]])
print("Token 'mat' can attend to:", [tokens[j] for j in range(6) if mask[5,j]])
""",

    "Flash Attention: Online Softmax Demo": """
import numpy as np

def standard_attention(Q, K, V):
    \"\"\"Materialises full n×n matrix.\"\"\"
    d_k    = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V

def flash_attention_1d(Q, K, V, block_size=2):
    \"\"\"
    Simplified Flash Attention for a SINGLE query row using online softmax.
    Demonstrates the tiling concept — never stores full score vector.
    \"\"\"
    d_k = Q.shape[-1]
    n   = K.shape[0]

    m_i  = -np.inf      # running max
    l_i  = 0.0          # running sum of exp(scores - m_i)
    O_i  = np.zeros(V.shape[-1])  # running output

    for j in range(0, n, block_size):
        K_j = K[j:j+block_size]
        V_j = V[j:j+block_size]

        s_j  = Q @ K_j.T / np.sqrt(d_k)          # tile scores
        m_j  = s_j.max()
        e_j  = np.exp(s_j - m_j)                  # tile exps

        m_new = max(m_i, m_j)
        l_new = np.exp(m_i - m_new) * l_i + np.exp(m_j - m_new) * e_j.sum()

        # Rescale previous output and add new contribution
        O_i   = (np.exp(m_i - m_new) * l_i * O_i +
                 np.exp(m_j - m_new) * (e_j @ V_j)) / l_new

        m_i, l_i = m_new, l_new

    return O_i

np.random.seed(0)
n, d_k, d_v = 8, 4, 4
Q_row = np.random.randn(d_k)
K     = np.random.randn(n, d_k)
V     = np.random.randn(n, d_v)

std_out   = standard_attention(Q_row[None], K, V)[0]
flash_out = flash_attention_1d(Q_row, K, V, block_size=2)

print("Standard Attention output:", np.round(std_out, 5))
print("Flash Attention output:   ", np.round(flash_out, 5))
print("Numerically identical:    ", np.allclose(std_out, flash_out, atol=1e-10))
""",

    "Positional Encodings Comparison": """
import numpy as np
import math

# ── Sinusoidal PE (Vaswani et al. 2017) ──────────────────────────────────────
def sinusoidal_pe(max_len, d_model):
    pe  = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe

# ── RoPE rotation matrix for a single position ───────────────────────────────
def rope_rotation(pos, d_k, theta=10000.0):
    \"\"\"Returns the rotation matrix R(pos) for a d_k-dim vector.\"\"\"
    freqs = 1.0 / (theta ** (np.arange(0, d_k, 2) / d_k))
    angles = pos * freqs
    cos_v  = np.repeat(np.cos(angles), 2)
    sin_v  = np.repeat(np.sin(angles), 2)
    # Interleave signs for [x1, x2, x3, x4] → [-x2, x1, -x4, x3]
    return cos_v, sin_v

def apply_rope(q, pos, theta=10000.0):
    d_k        = q.shape[-1]
    cos_v, sin_v = rope_rotation(pos, d_k, theta)
    q_rot = np.empty_like(q)
    q_rot[0::2] = q[0::2] * cos_v[0::2] - q[1::2] * sin_v[1::2]
    q_rot[1::2] = q[1::2] * cos_v[1::2] + q[0::2] * sin_v[0::2]
    return q_rot

# ── ALiBi bias ────────────────────────────────────────────────────────────────
def alibi_bias(n, num_heads):
    \"\"\"Returns ALiBi bias matrix (n×n) for a single head at given slope m.\"\"\"
    slopes = 2 ** (-8 * np.arange(1, num_heads + 1) / num_heads)
    # For head 0, distance matrix * slope
    dists  = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return -slopes[0] * dists  # shape (n, n) for head 0

# ── Demo ──────────────────────────────────────────────────────────────────────
pe = sinusoidal_pe(max_len=10, d_model=8)
print("Sinusoidal PE for positions 0–3 (first 4 dims):")
print(np.round(pe[:4, :4], 3))

q = np.random.randn(8)
print("\\nOriginal q[:4]:      ", np.round(q[:4], 3))
print("RoPE-rotated pos=0:  ", np.round(apply_rope(q, pos=0)[:4], 3))
print("RoPE-rotated pos=5:  ", np.round(apply_rope(q, pos=5)[:4], 3))

bias = alibi_bias(n=6, num_heads=4)
print("\\nALiBi bias matrix (head 0, 6 tokens):")
print(np.round(bias, 3))
""",

}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def render_operations() -> dict:
    """
    Execute every snippet in OPERATIONS and return a dict of
    { name: {"code": str, "output": str} }.
    """
    import io, contextlib
    results = {}
    for name, code in OPERATIONS.items():
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<string>", "exec"), {})
            output = _strip_ansi(buf.getvalue())
        except Exception as exc:
            output = f"[Error] {exc}"
        results[name] = {"code": code.strip(), "output": output.strip()}
    return results


def get_content() -> dict:
    return {
        "topic":        TOPIC_NAME,
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "operations":   OPERATIONS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE PREVIEW
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 78)
    print(f"  {ICON}  {DISPLAY_NAME}")
    print(f"     {SUBTITLE}")
    print("=" * 78)
    print(textwrap.dedent(THEORY[:3000]))
    print("\n... [truncated for preview] ...\n")

    print("\nRunning code operations...\n" + "-" * 40)
    results = render_operations()
    for name, res in results.items():
        print(f"\n▶ {name}")
        print(res["output"][:400])
        print()