"""Module: 02 · Language Modeling"""

import os
import re
import sys
import textwrap
from pathlib import Path
import base64

TOPIC_NAME   = "Language Modeling — Detailed Breakdown"
DISPLAY_NAME = "02 · Language Modeling"
ICON         = "📊"
SUBTITLE     = "Autoregressive prediction — the core task all LLMs are trained on"

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / "Implementation" / "LanguageModeling_Implementation" / "scripts"
_MAIN_SCRIPT  = _SCRIPTS_DIR / "lm_main.py"

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif",  "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">️ Image not found: {path}</p>'

THEORY = """
## 02 · Language Modeling

## Overview
A **language model** assigns a probability to any sequence of tokens.
During training, the model learns to predict the next token given all previous ones.
At inference, you sample from those predictions to generate text — one token at a time.
This single objective — *predict the next token* — is responsible for GPT-4, LLaMA, Claude,
Gemini, and every other modern large language model.

## Language Modeling Landscape — Full View

                            LANGUAGE MODELING LANDSCAPE — FULL VIEW

    ══════════════════════════════════════════════════════════════════════════════════════════
    
                            ┌───────────────────────────────────────┐
                            │  P(token₁, token₂, …, tokenₙ) = ?      │
                            │  How likely is this sequence of text? │
                            └──────────────────┬────────────────────┘
                                               │ Chain Rule of Probability
                                               │
                    ┌──────────────────────────▼──────────────────────────┐
                    │   P(w₁, w₂, …, wₙ) =  ∏  P(wₜ | w₁, …, wₜ₋₁)          │
                    │                        t=1                          │
                    │   Predict each token given ALL preceding context    │
                    └──────────────────────────┬──────────────────────────┘
                                               │
              ┌────────────────────────────────┼────────────────────────────────┐
              │                                │                                │
              ▼                                ▼                                ▼
    ┌──────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
    │   N-GRAM MODELS      │    │  NEURAL LANGUAGE MODELS  │    │  TRAINING OBJECTIVE     │
    │  (Statistical)       │    │  (Modern Standard)       │    │  (How we optimise)      │
    │                      │    │                          │    │                         │
    │ P(wₜ|wₜ₋ₙ₊₁…wₜ₋₁)        │    │ RNN → Transformer        │    │ Cross-Entropy Loss      │
    │ Count & smooth       │    │ Full context window      │    │ Perplexity as metric    │
    │ Markov assumption    │    │ Learned representations  │    │ Teacher forcing trick   │
    └──────────────────────┘    └──────────────────────────┘    └─────────────────────────┘
              │                                │                                │
              │                                │                                │
              ▼                                ▼                                ▼
    ┌──────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
    │ Unigram              │    │ Autoregressive (Decoder) │    │ Sampling Strategies     │
    │ Bigram               │    │   GPT, LLaMA, Claude     │    │                         │
    │ Trigram / N-gram     │    │                          │    │ Greedy  → deterministic │
    │ Kneser-Ney smoothing │    │ Masked (Encoder)         │    │ Beam    → approximate   │
    │                      │    │   BERT, RoBERTa          │    │ Temp    → creative ctrl │
    │ Sparsity problem:    │    │                          │    │ Top-K   → vocabulary    │
    │ unseen n-grams = 0   │    │ Encoder-Decoder          │    │           truncation    │
    │                      │    │   T5, BART               │    │ Top-P   → nucleus       │
    └──────────────────────┘    └──────────────────────────┘    └─────────────────────────┘

    ══════════════════════════════════════════════════════════════════════════════════════════

---

## 1. The Probability Chain Rule — Why "Predict Next Token" Works

The fundamental idea is to decompose the joint probability of a full sequence using
the **chain rule of probability**. No approximation is made here — it is exact:

    P(w₁, w₂, w₃, w₄)
      = P(w₁)
      × P(w₂ | w₁)
      × P(w₃ | w₁, w₂)
      × P(w₄ | w₁, w₂, w₃)

In general:
    P(w₁, …, wₙ) = ∏ P(wₜ | w₁, …, wₜ₋₁)
                   t=1

**Training goal:** Learn a function f(w₁, …, wₜ₋₁) that outputs a probability
distribution over every possible next token wₜ ∈ Vocabulary.

**Each training example is:** Every position in every document.
If a document has 1,000 tokens, that is 999 individual prediction tasks.
A trillion-token training corpus creates trillions of supervision signals —
at zero labeling cost. This is called **self-supervised learning**.

---

## 2. N-Gram Language Models — The Statistical Baseline

### The Markov Assumption
Computing P(wₜ | w₁, …, wₜ₋₁) is hard — the full history can be arbitrarily long.
N-gram models make a simplifying **Markov assumption**: only the last (n−1) tokens matter.

    P(wₜ | w₁, …, wₜ₋₁) ≈ P(wₜ | wₜ₋ₙ₊₁, …, wₜ₋₁)

N-gram models:

    N=1 (Unigram):   P(wₜ) — word frequency only, ignores all context
    N=2 (Bigram):    P(wₜ | wₜ₋₁) — conditions on 1 prior word
    N=3 (Trigram):   P(wₜ | wₜ₋₂, wₜ₋₁) — conditions on 2 prior words
    N=5:             Highest N commonly used in practice

### Computing N-gram Probabilities

From a corpus, simply count and divide:

    P(wₜ | wₜ₋₁) = count(wₜ₋₁, wₜ) / count(wₜ₋₁)

**Concrete bigram example:**

    Corpus: "the cat sat on the mat the cat sat"

    Bigram counts:
    (the, cat)  → 2
    (cat, sat)  → 2
    (sat, on)   → 1
    (on, the)   → 1
    (the, mat)  → 1

    P(cat | the)  =  count(the, cat) / count(the)  =  2/3  ≈ 0.667
    P(mat | the)  =  count(the, mat) / count(the)  =  1/3  ≈ 0.333
    P(sat | cat)  =  count(cat, sat) / count(cat)  =  2/2  =  1.000

### The Sparsity Problem

Longer n-grams are more expressive but suffer from **data sparsity**:
any n-gram not seen in training gets probability 0. This is catastrophic.

**Solutions — Smoothing:**

    Laplace (Add-1):       Add 1 to every count
                           P(wₜ | wₜ₋₁) = (count(wₜ₋₁, wₜ) + 1) / (count(wₜ₋₁) + |V|)

    Add-k Smoothing:       Add a small constant k (tuned on validation set)

    Kneser-Ney:            State-of-the-art smoothing — uses the "continuation count"
                           (how many unique contexts a word appears in), not raw frequency.
                           The word "Francisco" is common, but only after "San" →
                           Kneser-Ney assigns it a low probability in other contexts.

    Interpolation:         Blend multiple n-gram orders:
                           P(wₜ|wₜ₋₂,wₜ₋₁) = λ₃·P₃ + λ₂·P₂ + λ₁·P₁  where ∑λ = 1

### Limitations of N-gram Models

    1. No long-range dependencies beyond n−1 words
    2. Memory grows exponentially with n (store all n-gram counts)
    3. Identical n-gram context, wildly different meaning → no generalisation
    4. Each word is a surface form — no semantic similarity encoded

**Example of limitation 3:**
    "The dog bit the man" → P(man | bit, the) is estimated from count data.
    "The animal gnawed the person" → different tokens, zero learned transfer.
    A neural model that has read both sentences generalises. N-grams cannot.

---

## 3. Neural Language Models — The Modern Standard

### Key Idea
Replace the count table with a neural network that maps token history to a
probability distribution. The network learns **continuous representations** that
generalise across semantically similar contexts.

### Feedforward Neural Language Model (Bengio et al., 2003)

The first successful neural LM — a simple precursor to modern LLMs:

    1. Look up embeddings for each of the n−1 context tokens
    2. Concatenate embeddings into a single fixed-size vector
    3. Pass through one or more hidden layers (tanh/ReLU)
    4. Output layer: linear projection → softmax over vocabulary

    x = [e(wₜ₋ₙ₊₁) || … || e(wₜ₋₁)]       # concatenated embeddings
    h = tanh(W_h x + b_h)                    # hidden layer
    o = softmax(W_o h + b_o)                 # probability over |V| tokens

Still limited to a fixed context window. RNNs and Transformers removed this limit.

### Recurrent Neural Networks (RNNs / LSTMs)

RNNs maintain a hidden state hₜ passed from step to step:

    hₜ = tanh(W_hh · hₜ₋₁ + W_xh · xₜ + b)
    P(wₜ₊₁ | w₁…wₜ) = softmax(W_hy · hₜ + b_y)

In principle, hₜ can capture arbitrary context. In practice:
- Vanishing gradient problem: gradients shrink over long sequences
- LSTMs and GRUs mitigate (but don't fully solve) this
- Sequential computation is slow — cannot parallelise across time steps

### Transformer Language Models (2017–present)

The Transformer replaced RNNs entirely for language modeling:
- **Self-attention** lets every token attend directly to every past token
- **Parallelisation**: all positions computed simultaneously during training
- **Scales effortlessly**: more layers, more heads, more parameters → better

See Module 03 (Attention & Transformers) for the full architecture breakdown.

### Three Training Paradigms

    ┌─────────────────────────────────┬────────────────────────────────────┬───────────────────────────────┐
    │  Autoregressive (Causal LM)     │  Masked Language Model (MLM)       │  Encoder-Decoder (Seq2Seq)    │
    ├─────────────────────────────────┼────────────────────────────────────┼───────────────────────────────┤
    │ Predict NEXT token              │ Predict MASKED tokens              │ Encode input, decode output   │
    │                                 │                                    │                               │
    │ Causal (triangular) mask:       │ Random 15% tokens masked           │ Encoder: bidirectional attn   │
    │ token i can only see 1…i-1      │ Bidirectional context used         │ Decoder: causal attn          │
    │                                 │                                    │                               │
    │ Loss: NLL of next token at      │ Loss: NLL of masked tokens only    │ Loss: NLL of decoder tokens   │
    │ every position                  │                                    │                               │
    │                                 │ Strong contextual representations  │ Excels at translation,        │
    │ Naturally generative:           │ Not naturally generative           │ summarization, Q&A            │
    │ sample to produce text          │                                    │                               │
    │                                 │                                    │                               │
    │ GPT-2, GPT-3, GPT-4,            │ BERT, RoBERTa, DistilBERT,         │ T5, BART, mT5                 │
    │ LLaMA, Mistral, Claude          │ ALBERT, DeBERTa                    │                               │
    └─────────────────────────────────┴────────────────────────────────────┴───────────────────────────────┘

---

## 4. The Training Objective — Cross-Entropy Loss

The model outputs a vector of logits z ∈ ℝ|V| — one raw score per vocabulary token.
Softmax converts logits to a probability distribution:

    P(wₜ = k | context) = exp(zₖ) / ∑ exp(zⱼ)
                                      j

**Cross-entropy loss** for a single position:

    ℒₜ = −log P(wₜ = y | context)

where y is the true next token ID.

**Loss over a sequence of T tokens:**

    ℒ = −(1/T) ∑ log P(wₜ | w₁, …, wₜ₋₁)
               t=1

This is equivalent to **maximising the log-likelihood** of the training data.

### Teacher Forcing

During training, the model always receives the *true* previous tokens as input —
even if it would have predicted something different. This is called **teacher forcing**.

    Training input:   [<BOS>, "The",  "cat",  "sat"]
    Training target:  ["The", "cat",  "sat",  "on" ]
    At each step, true token is used as next input regardless of model's prediction.

    Advantage:  Stable, fast training.
    Disadvantage: Exposure bias — at inference time the model feeds its own (potentially
                  wrong) predictions back in. Small errors compound over long sequences.

### Why NLL and Cross-Entropy Are the Same Thing

    NLL (Negative Log-Likelihood) of the ground truth token y:
        −log P(y)

    Cross-Entropy H(p, q) when p is a one-hot distribution:
        −∑ p(k) log q(k) = −1 · log q(y) = −log P(y)

    They are identical when the target distribution is one-hot.

---

## 5. Perplexity — The Standard Evaluation Metric

Perplexity (PPL) measures how "surprised" a model is by unseen text.
Lower perplexity = better model (less surprised = better predictions).

    PPL = exp( ℒ )
        = exp(−(1/T) ∑ log P(wₜ | w₁,…,wₜ₋₁))

Intuition: **PPL ≈ "effective vocabulary size the model is choosing from at each step."**

    PPL = 1     → Model is perfectly certain every time (impossible in practice)
    PPL = |V|   → Model is completely uniform — no learning at all
    PPL = 10    → Model acts like it's choosing from 10 equally likely options

**Benchmark perplexities on Penn Treebank (PTB) dataset:**

    Model                   PPL
    ────────────────────────────────
    Trigram (Kneser-Ney)    141
    LSTM (2017 SOTA)         58
    Transformer-XL (2019)    21
    GPT-2 (117M)             35       ← different test set, not directly comparable
    GPT-3 (175B)             20       ← on Penn Treebank
    GPT-4 (est.)             <5       ← on standard benchmarks

**Important caveat:** Perplexity is dataset-specific. You cannot compare PPL across
different test sets or tokenizers. Always verify what corpus and tokenizer were used.

---

## 6. Autoregressive Generation — The Inference Loop

At inference, the model generates tokens one at a time. At each step:

    1. Feed the current sequence into the model
    2. Compute logits for the next position
    3. Apply a sampling strategy to select the next token
    4. Append the selected token and repeat

    Prompt:   "The capital of France is"
    Step 1:   → logits over |V| → select "Paris"     → "The capital of France is Paris"
    Step 2:   → logits over |V| → select "."         → "The capital of France is Paris."
    Step 3:   → logits over |V| → select "<EOS>"     → stop

This loop is the same for every decoder-only LLM, from GPT-2 to GPT-4.

### KV-Cache (Efficient Inference)

Naively, at each new step you would recompute attention over the entire sequence.
KV-caching saves the key and value matrices from all past positions —
only the new token's Q, K, V need to be computed. This is why inference
throughput depends primarily on memory bandwidth, not FLOPS.

---

## 7. Sampling Strategies — Controlling the Output Distribution

At each step the model produces P(next_token | context) — a distribution over |V| tokens.
How you sample from this distribution controls creativity, coherence, and diversity.

### A) Greedy Decoding

    Always pick the token with the highest probability.
    
    ŵ = argmax P(wₜ | context)

    Pros: Fast, deterministic, consistent.
    Cons: Often repetitive and boring. Can get stuck in loops.
          Not optimal — the globally best sequence ≠ always choosing local maximum.

### B) Beam Search

    Maintain a "beam" of the top-B partial sequences simultaneously.
    At each step, expand each beam by all possible next tokens, keep top-B.
    
    B=1 → greedy decoding.
    B=4 or B=5 → commonly used for translation, summarisation.
    
    Pros: Better than greedy at finding high-probability sequences.
    Cons: Still often generates generic, repetitive text for open-ended generation.
          Computationally B× more expensive than greedy.

### C) Temperature Sampling

    Divide all logits by temperature T before softmax:
    
    P_T(wₜ = k) = exp(zₖ / T) / ∑ exp(zⱼ / T)

    T < 1.0 → sharpen distribution → more peaked → more deterministic, focused
    T = 1.0 → unchanged → sample from true model distribution  
    T > 1.0 → flatten distribution → more uniform → more random, creative

    T → 0   → approaches greedy decoding
    T → ∞   → approaches uniform random (pure noise)

    Typical usage:
    - Code generation:          T ≈ 0.2  (need accuracy)
    - Creative writing:         T ≈ 0.9  (want variety)
    - Factual Q&A:              T ≈ 0.0  (need determinism)

### D) Top-K Sampling

    Keep only the K most probable tokens; sample uniformly among them.
    
    Filter: keep top-K tokens by probability, renormalise, sample.
    
    K = 1   → greedy decoding
    K = 50  → commonly used default
    
    Problem: K=50 can include very unlikely tokens when the distribution is peaked,
             or too few good options when the distribution is flat.

### E) Top-P (Nucleus) Sampling

    Proposed in "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2020).
    
    Dynamically select the smallest set of tokens whose cumulative probability ≥ P.
    
    Sort tokens by probability (descending).
    Add tokens one by one until cumulative probability ≥ P. Sample from this set.
    
    p = 0.9 → Keep just enough tokens to cover 90% of probability mass.
    
    When distribution is peaked:  nucleus might be just 1–5 tokens.
    When distribution is flat:    nucleus might be 100+ tokens.
    
    This adapts to the model's actual confidence at each step.
    Top-P is generally preferred over Top-K for open-ended generation.

### F) Combining Strategies

Real systems combine multiple strategies:

    1. Apply temperature scaling to logits
    2. Apply Top-K filter (remove long tail)
    3. Apply Top-P filter (nucleus)
    4. Sample from remaining renormalised distribution

    Typical production settings:  T=0.8, K=40, P=0.9

---

## 8. Repetition & Degeneration — The Failure Mode

Without intervention, autoregressive LLMs tend to repeat themselves.
The model falls into loops because a repeated phrase raises the probability of
repeating it again — a feedback cycle.

**Repetition penalty (simple fix):**

    Divide the logit of any previously seen token by a penalty factor r > 1.
    This makes the model less likely to repeat tokens it has already generated.
    
    Typical r = 1.1 to 1.3

**Why greedy / beam search are worst:**

    These deterministic strategies always pick high-probability tokens.
    If the sequence so far makes "and" the most likely next word every step,
    they generate "and and and and..." indefinitely.

---

## 9. Mathematical Summary

    Language Model:          P(w₁, …, wₙ) = ∏ P(wₜ | w₁, …, wₜ₋₁)

    NLL Training Loss:       ℒ = −(1/T) ∑ log P(wₜ | w₁,…,wₜ₋₁)

    Perplexity:              PPL = exp(ℒ)

    Temperature Softmax:     P_T(k) = exp(zₖ/T) / ∑ exp(zⱼ/T)

    Top-P Nucleus:           S = smallest set s.t. ∑ P(wₜ=k) ≥ p
                                                   k∈S

    Bigram MLE:              P(wₜ | wₜ₋₁) = count(wₜ₋₁, wₜ) / count(wₜ₋₁)

    Cross-Entropy = NLL when target is one-hot: −log P(y)

---

## Key Takeaways

- Every modern LLM is trained on the same objective: **predict the next token**.
- This objective arises directly from the chain rule of probability — no approximation.
- N-gram models count occurrences; neural models learn continuous representations.
- **Perplexity = exp(cross-entropy loss)** — lower is better; dataset-specific.
- **Teacher forcing** enables fast training but introduces exposure bias at inference.
- **Temperature** controls sharpness; **Top-P** dynamically adapts the candidate set.
- **KV-caching** makes autoregressive inference efficient by not recomputing past tokens.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ──────────────────────────────────────────────────────────────────
    "1 · N-Gram Language Model": {
        "description": (
            "Build a bigram (and trigram) language model from scratch using pure Python. "
            "Counts token co-occurrences, computes MLE probabilities with Add-k smoothing, "
            "and generates text by sampling the learned distribution. "
            "Shows exactly what classical NLP used before neural networks."
        ),
        "language": "python",
        "code": """
            import math
            import random
            from collections import defaultdict, Counter

            # ── Corpus ──────────────────────────────────────────────────────────
            CORPUS = (
                "the cat sat on the mat . the cat ate the rat . "
                "the rat ran from the cat . the dog chased the cat . "
                "a dog and a cat are friends . the mat is on the floor . "
                "the cat sat by the door . a rat and a cat cannot be friends . "
                "the dog sat on the mat too ."
            ).lower().split()

            print("Corpus tokens:", len(CORPUS))
            print("Unique tokens:", len(set(CORPUS)))
            print()

            # ── Build N-gram counts ──────────────────────────────────────────────
            BOS, EOS = "<BOS>", "<EOS>"
            tokens = [BOS] + CORPUS + [EOS]

            unigram_counts = Counter(tokens)
            bigram_counts  = defaultdict(Counter)
            trigram_counts = defaultdict(Counter)

            for i in range(len(tokens) - 1):
                bigram_counts[tokens[i]][tokens[i+1]] += 1

            for i in range(len(tokens) - 2):
                trigram_counts[(tokens[i], tokens[i+1])][tokens[i+2]] += 1

            vocab = list(set(tokens))
            V     = len(vocab)

            print(f"Vocabulary size |V| = {V}")
            print()

            # ── MLE + Add-k Bigram Probability ─────────────────────────────────
            k = 0.01   # smoothing constant

            def bigram_prob(prev, word):
                num   = bigram_counts[prev][word] + k
                denom = unigram_counts[prev] + k * V
                return num / denom

            # Show some probabilities
            queries = [
                ("the", "cat"),
                ("the", "dog"),
                ("the", "rat"),
                ("cat", "sat"),
                ("cat", "ate"),
                ("the", "zygote"),   # never seen — smoothing handles it
            ]

            print(f"{'P(word | prev)':30s}   {'count(prev,word)':18s}  {'P':>10s}")
            print("-" * 65)
            for prev, word in queries:
                c = bigram_counts[prev][word]
                p = bigram_prob(prev, word)
                print(f"  P({word:10s} | {prev:6s})         count={c:3d}          {p:.6f}")

            print()

            # ── Text Generation via Bigram Sampling ───────────────────────────
            def generate_bigram(max_tokens=20, seed=42):
                random.seed(seed)
                current = BOS
                generated = []
                for _ in range(max_tokens):
                    candidates = vocab
                    weights    = [bigram_prob(current, w) for w in candidates]
                    current    = random.choices(candidates, weights=weights, k=1)[0]
                    if current == EOS:
                        break
                    generated.append(current)
                return " ".join(generated)

            print("── Generated Text (Bigram Sampling) ──")
            for seed in [0, 1, 2, 3, 4]:
                print(f"  Seed {seed}: {generate_bigram(seed=seed)}")

            print()

            # ── Trigram MLE Probabilities ─────────────────────────────────────
            def trigram_prob(w1, w2, word):
                tri_count = trigram_counts[(w1, w2)][word]
                bi_count  = bigram_counts[w1][w2]
                if bi_count == 0:
                    return bigram_prob(w2, word)          # back-off to bigram
                num   = tri_count + k
                denom = bi_count  + k * V
                return num / denom

            print("── Trigram Probabilities ──")
            trigram_queries = [
                ("the", "cat", "sat"),
                ("the", "cat", "ate"),
                ("the", "cat", "ran"),
            ]
            for w1, w2, word in trigram_queries:
                p = trigram_prob(w1, w2, word)
                c = trigram_counts[(w1,w2)][word]
                print(f"  P({word:6s} | {w1},{w2})  count={c}  P={p:.6f}")
        """,
    },

    # ── 2 ──────────────────────────────────────────────────────────────────
    "2 · Cross-Entropy Loss & Perplexity": {
        "description": (
            "Implement cross-entropy loss and perplexity from scratch using NumPy. "
            "Shows how the training objective is computed, the numerical trick that prevents "
            "log(0) instability, and how perplexity relates to loss. Compares a well-trained "
            "vs random model to build intuition for what the numbers mean."
        ),
        "language": "python",
        "code": """
            import numpy as np

            # ── Softmax & Cross-Entropy ──────────────────────────────────────────
            def softmax(logits):
                \"\"\"Numerically stable softmax.\"\"\"
                logits = logits - logits.max()          # subtract max → prevents overflow
                exp    = np.exp(logits)
                return exp / exp.sum()

            def cross_entropy(logits, target_id):
                \"\"\"
                Compute cross-entropy loss for a single position.
                logits:    raw model outputs, shape (vocab_size,)
                target_id: integer index of the true next token
                \"\"\"
                probs = softmax(logits)
                # Clip to avoid log(0) — shouldn't happen after softmax but safe practice
                loss  = -np.log(probs[target_id] + 1e-12)
                return loss, probs[target_id]

            def sequence_loss_and_ppl(logits_list, target_ids):
                \"\"\"
                logits_list:  list of T arrays each of shape (vocab_size,)
                target_ids:   list of T integer token IDs (ground truth)
                Returns: (mean_loss, perplexity)
                \"\"\"
                losses = []
                for logits, tid in zip(logits_list, target_ids):
                    loss, _ = cross_entropy(logits, tid)
                    losses.append(loss)
                mean_loss  = np.mean(losses)
                perplexity = np.exp(mean_loss)
                return mean_loss, perplexity

            vocab_size = 50_000
            np.random.seed(42)

            print("═" * 60)
            print("  Cross-Entropy Loss & Perplexity — Worked Examples")
            print("═" * 60)
            print()

            # ── Example 1: Model is PERFECTLY certain ─────────────────────────
            print("── Example 1: Perfect model (logit on true token = +∞ effectively) ──")
            logits_perfect = np.full(vocab_size, -1e9)
            logits_perfect[7] = 100.0                     # true token = 7
            loss, prob = cross_entropy(logits_perfect, target_id=7)
            print(f"  P(true token) ≈ {prob:.6f}")
            print(f"  Loss          = {loss:.6f}   (→ 0, perfect)")
            print(f"  Perplexity    = {np.exp(loss):.4f}   (→ 1, no surprise)")
            print()

            # ── Example 2: Model assigns uniform distribution (untrained) ──────
            print("── Example 2: Untrained model (uniform distribution) ──")
            logits_uniform = np.zeros(vocab_size)         # all logits equal → uniform
            loss_u, prob_u = cross_entropy(logits_uniform, target_id=7)
            ppl_u          = np.exp(loss_u)
            print(f"  P(true token) ≈ 1/{vocab_size} = {prob_u:.8f}")
            print(f"  Loss          = {loss_u:.4f}  ≈ log({vocab_size}) = {np.log(vocab_size):.4f}")
            print(f"  Perplexity    = {ppl_u:.1f}   ≈ |V| = {vocab_size}")
            print()

            # ── Example 3: Realistic trained model — simulate a sequence ──────
            print("── Example 3: Realistic trained model on a 12-token sequence ──")
            T = 12
            true_tokens = np.random.randint(0, vocab_size, size=T)
            sequence_logits = []
            for i, tid in enumerate(true_tokens):
                # Simulate a model that gives the true token a boosted logit
                logits = np.random.randn(vocab_size) * 1.0
                logits[tid] += 6.0       # well-trained model boosts true token
                sequence_logits.append(logits)

            mean_loss, ppl = sequence_loss_and_ppl(sequence_logits, true_tokens)
            print(f"  Sequence length:  T = {T}")
            print(f"  Mean loss:        {mean_loss:.4f}  nats")
            print(f"  Perplexity:       {ppl:.2f}")
            print()

            # ── Example 4: Compare a good vs bad model side by side ──────────
            print("── Example 4: Good model vs Bad model on same sequence ──")
            boosts = {"Good model": 8.0, "Average model": 4.0, "Weak model": 1.0, "Random": 0.0}
            print(f"  {'Model':20s}  {'Mean Loss':12s}  {'Perplexity':12s}")
            print("  " + "-" * 48)
            for label, boost in boosts.items():
                sl = []
                for tid in true_tokens:
                    lg = np.random.randn(vocab_size) * 1.0
                    lg[tid] += boost
                    sl.append(lg)
                ml, pp = sequence_loss_and_ppl(sl, true_tokens)
                print(f"  {label:20s}  {ml:12.4f}  {pp:12.2f}")

            print()
            print("Key insight: Perplexity = exp(loss). When loss ≈ 0, PPL ≈ 1.")
            print(f"When loss = log|V| = {np.log(vocab_size):.2f}, PPL = |V| = {vocab_size}.")
        """,
    },

    # ── 3 ──────────────────────────────────────────────────────────────────
    "3 · Autoregressive Text Generation Loop": {
        "description": (
            "Simulate the full autoregressive generation loop used by GPT, LLaMA, and Claude "
            "at inference time. A tiny mock transformer produces logits for each position; "
            "tokens are sampled one at a time, fed back in, and the process repeats. "
            "Illustrates teacher forcing (training) vs free-running (inference) modes."
        ),
        "language": "python",
        "code": """
            import numpy as np

            # ── Tiny Vocabulary & Model Mock ────────────────────────────────────
            VOCAB = ["<BOS>", "<EOS>", "the", "cat", "sat", "on", "mat",
                     "dog", "ran", "and", "a",  "rat", ".", "by", "door"]
            V   = len(VOCAB)
            w2i = {w: i for i, w in enumerate(VOCAB)}
            i2w = {i: w for i, w in enumerate(VOCAB)}

            # Learned bigram-style logit table — simulates a trained tiny model
            # logit_table[context_id][next_id] = logit value
            np.random.seed(7)
            LOGIT_TABLE = np.random.randn(V, V) * 0.3      # base noise

            # Inject "knowledge" — boost plausible continuations
            boosts = [
                ("<BOS>", "the"), ("<BOS>", "a"),
                ("the",   "cat"), ("the",  "dog"), ("the", "mat"), ("the", "rat"),
                ("cat",   "sat"), ("cat",  "ran"),
                ("dog",   "sat"), ("dog",  "ran"),
                ("sat",   "on"),  ("sat",  "."),
                ("ran",   "and"), ("ran",  "by"),
                ("on",    "the"), ("on",   "a"),
                (".",     "<EOS>"),
                ("a",     "cat"), ("a",    "dog"),
            ]
            for prev, nxt in boosts:
                LOGIT_TABLE[w2i[prev]][w2i[nxt]] += 4.0

            def model_logits(token_id):
                \"\"\"Return logits for next token given the current context token.\"\"\"
                return LOGIT_TABLE[token_id].copy()

            def softmax(z):
                z = z - z.max()
                e = np.exp(z)
                return e / e.sum()

            # ── Autoregressive Generation ────────────────────────────────────────
            def generate(prompt_ids, max_new_tokens=10, temperature=1.0, seed=None):
                \"\"\"
                Free-running autoregressive generation.
                At each step:
                  1. Take the last token
                  2. Ask the model for logits
                  3. Sample next token from softmax(logits / T)
                  4. Append to sequence; stop on <EOS>
                \"\"\"
                if seed is not None:
                    np.random.seed(seed)

                ids   = list(prompt_ids)
                steps = []

                for step in range(max_new_tokens):
                    current_id = ids[-1]
                    logits     = model_logits(current_id)
                    probs      = softmax(logits / temperature)

                    # Sample
                    next_id    = np.random.choice(V, p=probs)
                    p_chosen   = probs[next_id]

                    steps.append({
                        "step":       step + 1,
                        "input_tok":  i2w[current_id],
                        "output_tok": i2w[next_id],
                        "prob":       p_chosen,
                    })
                    ids.append(next_id)

                    if next_id == w2i["<EOS>"]:
                        break

                return ids, steps

            # ── Run Examples ───────────────────────────────────────────────────
            prompt = [w2i["<BOS>"]]

            print("═" * 65)
            print("  Autoregressive Generation — Step-by-Step Trace")
            print("═" * 65)
            print(f"  Vocabulary: {VOCAB}")
            print(f"  Prompt:     [<BOS>]")
            print()

            for temp, seed in [(1.0, 0), (1.0, 1), (0.3, 0), (2.0, 0)]:
                ids, steps = generate(prompt, max_new_tokens=12, temperature=temp, seed=seed)
                text = " ".join(i2w[i] for i in ids[1:])   # skip BOS

                print(f"  Temperature={temp:.1f}  Seed={seed}")
                print(f"  ┌{'─'*7}┬{'─'*12}┬{'─'*14}┬{'─'*10}┐")
                print(f"  │ {'Step':5s} │ {'Input':10s} │ {'→ Predicted':12s} │ {'P(token)':8s} │")
                print(f"  ├{'─'*7}┼{'─'*12}┼{'─'*14}┼{'─'*10}┤")
                for s in steps:
                    print(f"  │ {s['step']:5d} │ {s['input_tok']:10s} │   {s['output_tok']:12s}│ {s['prob']:.4f}   │")
                print(f"  └{'─'*7}┴{'─'*12}┴{'─'*14}┴{'─'*10}┘")
                print(f'  Generated: "{text}"')
                print()

            # ── Teacher Forcing Illustration ───────────────────────────────────
            print("── Teacher Forcing (Training Mode) vs Free-Running (Inference) ──")
            print()
            TRUE_SEQ = ["<BOS>", "the", "cat", "sat", "on", "the", "mat", "."]
            true_ids = [w2i[w] for w in TRUE_SEQ]

            print("  Training (Teacher Forcing): always feed true token as next input")
            total_loss = 0.0
            for t in range(len(TRUE_SEQ) - 1):
                inp_id  = true_ids[t]
                tgt_id  = true_ids[t + 1]
                logits  = model_logits(inp_id)
                probs   = softmax(logits)
                loss    = -np.log(probs[tgt_id] + 1e-12)
                total_loss += loss
                print(f"    t={t+1}  input={i2w[inp_id]:6s}  target={i2w[tgt_id]:6s}"
                      f"  P={probs[tgt_id]:.4f}  loss={loss:.4f}")

            mean_loss  = total_loss / (len(TRUE_SEQ) - 1)
            perplexity = np.exp(mean_loss)
            print(f"\\n  Mean loss: {mean_loss:.4f}   Perplexity: {perplexity:.2f}")
        """,
    },

    # ── 4 ──────────────────────────────────────────────────────────────────
    "4 · Temperature, Top-K & Top-P Sampling": {
        "description": (
            "Implement and visualise all three major sampling strategies used in production LLMs: "
            "temperature scaling, Top-K filtering, and Top-P (nucleus) sampling. "
            "Shows step-by-step how each transforms the raw probability distribution "
            "and demonstrates how they interact when combined."
        ),
        "language": "python",
        "code": """
            import numpy as np

            # ── Helper: softmax ─────────────────────────────────────────────────
            def softmax(z):
                z = z - z.max()
                e = np.exp(z)
                return e / e.sum()

            # ── Example logits — simulate a peaked distribution ─────────────────
            VOCAB = ["Paris", "London", "Berlin", "Rome", "Madrid", "Tokyo",
                     "Sydney", "Cairo",  "Seoul",  "Oslo",  "Lima",   "the",
                     "is",    "a",      "very",   "city",  ".",      "<EOS>"]
            V = len(VOCAB)

            np.random.seed(3)
            raw_logits = np.random.randn(V) * 1.5
            # Boost "Paris" strongly — model knows the answer
            raw_logits[0] += 5.0    # Paris
            raw_logits[1] += 2.0    # London (secondary)
            raw_logits[2] += 1.5    # Berlin
            raw_logits[3] += 1.0    # Rome

            def print_dist(probs, label, top_n=8):
                order = np.argsort(probs)[::-1]
                print(f"  {label}")
                print(f"  {'Token':12s}  {'Prob':8s}  Bar")
                print("  " + "-" * 48)
                cumulative = 0.0
                for rank, idx in enumerate(order[:top_n]):
                    bar = "█" * int(probs[idx] * 60)
                    cumulative += probs[idx]
                    print(f"  {VOCAB[idx]:12s}  {probs[idx]:.5f}  {bar}")
                print(f"  ... ({V-top_n} more tokens, cumulative top-{top_n} = {cumulative:.4f})")
                print()

            # ── 1. Baseline Distribution ─────────────────────────────────────────
            baseline = softmax(raw_logits)
            print("═" * 60)
            print("  Sampling Strategy Comparison")
            print("═" * 60)
            print()
            print_dist(baseline, "BASELINE — softmax(logits) — T=1.0")

            # ── 2. Temperature Scaling ───────────────────────────────────────────
            print("── Temperature Scaling ──")
            for T in [0.3, 0.7, 1.0, 1.5, 2.5]:
                probs = softmax(raw_logits / T)
                top1  = VOCAB[np.argmax(probs)]
                top1p = probs.max()
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                print(f"  T={T:.1f}  P({top1}) = {top1p:.4f}   entropy = {entropy:.3f} nats")
            print()
            print_dist(softmax(raw_logits / 0.3), "Temperature T=0.3 (sharp)")
            print_dist(softmax(raw_logits / 2.0), "Temperature T=2.0 (flat)")

            # ── 3. Top-K Sampling ────────────────────────────────────────────────
            def top_k_probs(logits, k):
                \"\"\"Zero out all but top-K logits, then softmax.\"\"\"
                threshold = np.sort(logits)[::-1][k-1]
                filtered  = np.where(logits >= threshold, logits, -1e10)
                return softmax(filtered)

            print("── Top-K Sampling ──")
            for k in [1, 3, 5, 10, 18]:
                probs = top_k_probs(raw_logits, k)
                non_zero = (probs > 1e-9).sum()
                top1p    = probs.max()
                print(f"  K={k:2d}  active tokens = {non_zero:2d}  P(Paris) = {top1p:.4f}")
            print()
            print_dist(top_k_probs(raw_logits, 5), "Top-K=5 filtered distribution")

            # ── 4. Top-P (Nucleus) Sampling ─────────────────────────────────────
            def top_p_probs(logits, p):
                \"\"\"Keep smallest set of tokens whose cumulative probability >= p.\"\"\"
                probs     = softmax(logits)
                order     = np.argsort(probs)[::-1]
                cumsum    = np.cumsum(probs[order])
                # Find cutoff: include all tokens up to where cumsum first reaches p
                cutoff_n  = np.searchsorted(cumsum, p) + 1
                nucleus   = order[:cutoff_n]
                filtered  = np.full(V, -1e10)
                filtered[nucleus] = logits[nucleus]
                return softmax(filtered), cutoff_n

            print("── Top-P (Nucleus) Sampling ──")
            for p_val in [0.5, 0.7, 0.9, 0.95, 0.99]:
                probs, n_tokens = top_p_probs(raw_logits, p_val)
                top1p = probs.max()
                print(f"  P={p_val:.2f}  nucleus size = {n_tokens:2d} tokens  P(Paris) = {top1p:.4f}")
            print()
            print_dist(top_p_probs(raw_logits, 0.9)[0], "Top-P=0.90 nucleus distribution")

            # ── 5. Combined Pipeline ─────────────────────────────────────────────
            print("── Combined Pipeline: T=0.8, K=10, P=0.9 ──")
            T_scaled  = raw_logits / 0.8
            topk_logits = np.where(
                T_scaled >= np.sort(T_scaled)[::-1][9], T_scaled, -1e10
            )
            nucleus, n = top_p_probs(topk_logits, 0.9)
            print(f"  Active tokens after K: 10   After P=0.9: {n}")
            print_dist(nucleus, "Final combined distribution")

            # ── 6. Sample Comparison ─────────────────────────────────────────────
            np.random.seed(42)
            print("── 50-sample draws from each strategy ──")
            strategies = {
                "Greedy":    lambda: VOCAB[np.argmax(baseline)],
                "T=1.0":     lambda: VOCAB[np.random.choice(V, p=baseline)],
                "T=0.3":     lambda: VOCAB[np.random.choice(V, p=softmax(raw_logits/0.3))],
                "T=2.0":     lambda: VOCAB[np.random.choice(V, p=softmax(raw_logits/2.0))],
                "Top-K=5":   lambda: VOCAB[np.random.choice(V, p=top_k_probs(raw_logits,5))],
                "Top-P=0.9": lambda: VOCAB[np.random.choice(V, p=top_p_probs(raw_logits,0.9)[0])],
            }
            from collections import Counter
            for label, sampler in strategies.items():
                samples = [sampler() for _ in range(50)]
                counts  = Counter(samples).most_common(4)
                summary = "  ".join(f"{w}×{c}" for w, c in counts)
                print(f"  {label:12s}: {summary}")
        """,
    },

    # ── 5 ──────────────────────────────────────────────────────────────────
    "5 · Perplexity Benchmark & Model Comparison": {
        "description": (
            "Calculate and compare perplexity across simulated models of varying quality. "
            "Demonstrates how perplexity acts as a single number that summarises model quality, "
            "shows the effect of tokenizer choice on perplexity values, "
            "and reproduces benchmark-style PPL comparisons you'd see in research papers."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            def softmax(z):
                z = z - z.max()
                e = np.exp(z)
                return e / e.sum()

            def compute_ppl(logits_seq, targets):
                \"\"\"
                logits_seq: list of (vocab_size,) arrays — one per position
                targets:    list of int — true token index at each position
                Returns: (mean_NLL, perplexity)
                \"\"\"
                nll_total = 0.0
                T = len(targets)
                for logits, tid in zip(logits_seq, targets):
                    probs = softmax(logits)
                    nll_total += -math.log(probs[tid] + 1e-12)
                mean_nll  = nll_total / T
                ppl       = math.exp(mean_nll)
                return mean_nll, ppl

            # ── Simulate Test Corpus ──────────────────────────────────────────────
            np.random.seed(0)
            VOCAB_SIZE  = 50_000
            T           = 200          # sequence length for evaluation

            TRUE_TOKENS = np.random.randint(0, VOCAB_SIZE, size=T)

            def make_logits(true_tokens, boost, noise_std=1.0):
                \"\"\"
                Generate logits for each token in a sequence.
                boost:     how much to add to the true token's logit.
                Higher boost → model is more confident and correct → lower PPL.
                \"\"\"
                seq = []
                for tid in true_tokens:
                    lg = np.random.randn(VOCAB_SIZE) * noise_std
                    lg[tid] += boost
                    seq.append(lg)
                return seq

            # ── Model Tiers ──────────────────────────────────────────────────────
            model_tiers = [
                ("Random baseline",          0.0,  1.0),
                ("Unigram LM",               0.5,  1.0),
                ("Trigram (Kneser-Ney sim)", 2.0,  1.0),
                ("LSTM (2017 era)",          4.0,  0.8),
                ("Transformer (2019)",       6.0,  0.6),
                ("GPT-2 scale",              8.0,  0.5),
                ("GPT-3 / LLaMA scale",     10.0,  0.4),
                ("Near-perfect (oracle)",   15.0,  0.3),
            ]

            print("═" * 68)
            print("  Language Model Perplexity Benchmarks")
            print(f"  Test set: {T} tokens   |   Vocabulary size: {VOCAB_SIZE:,}")
            print("═" * 68)
            print(f"  {'Model':30s}  {'Mean NLL':10s}  {'PPL':10s}  {'Quality'}")
            print("  " + "─" * 64)

            for name, boost, noise in model_tiers:
                logits = make_logits(TRUE_TOKENS, boost, noise)
                nll, ppl = compute_ppl(logits, TRUE_TOKENS)
                quality = (
                    "★★★★★" if ppl < 20 else
                    "★★★★" if ppl < 100 else
                    "★★★"  if ppl < 1_000 else
                    "★★"   if ppl < 10_000 else
                    "★"
                )
                print(f"  {name:30s}  {nll:10.4f}  {ppl:10.2f}  {quality}")

            print()
            print(f"  Random baseline PPL ≈ |V| = {VOCAB_SIZE:,}")
            print(f"  Perfect oracle PPL  → 1.0")
            print()

            # ── Perplexity Sensitivity: Small loss change → big PPL change ───────
            print("── PPL Sensitivity to Loss Changes ──")
            print(f"  {'Mean NLL':10s}  {'PPL':12s}  Change vs NLL=3.0")
            print("  " + "─" * 42)
            reference_nll = 3.0
            reference_ppl = math.exp(reference_nll)
            for nll_val in [3.5, 3.2, 3.0, 2.8, 2.5, 2.0, 1.5, 1.0]:
                ppl  = math.exp(nll_val)
                diff = ppl - reference_ppl
                sign = "+" if diff >= 0 else ""
                print(f"  {nll_val:10.2f}  {ppl:12.2f}  {sign}{diff:+.2f}")
            print()
            print("  A 0.5-nat drop in NLL can halve perplexity — small loss gains matter a lot.")
            print()

            # ── Bits-per-character (BPC) — Alternative Metric ─────────────────
            print("── Bits-Per-Character (BPC) ──")
            print("  BPC = NLL / log(2)  (convert nats to bits)")
            print()
            for name, boost, noise in model_tiers[::2]:
                logits = make_logits(TRUE_TOKENS, boost, noise)
                nll, ppl = compute_ppl(logits, TRUE_TOKENS)
                bpc = nll / math.log(2)
                print(f"  {name:30s}  PPL={ppl:8.2f}  BPC={bpc:.4f}")
            print()
            print("  Good character-level models achieve BPC < 1.0 on English text.")
            print("  (Each character costs less than 1 bit to encode on average.)")
        """,
    },

    # ── 6 ──────────────────────────────────────────────────────────────────
    "6 · Repetition Penalty & Degeneration": {
        "description": (
            "Demonstrate the repetition/degeneration failure mode of autoregressive LLMs "
            "and implement the standard repetition penalty fix. Shows why greedy decoding "
            "collapses into loops and how a simple logit penalty breaks them."
        ),
        "language": "python",
        "code": """
            import numpy as np
            from collections import Counter

            VOCAB = ["the", "cat", "sat", "on", "mat", "and", "a", "dog",
                     "ran", ".",  "is",  "very", "good", "<EOS>", "big",
                     "small", "old", "new", "fast", "slow"]
            V   = len(VOCAB)
            w2i = {w: i for i, w in enumerate(VOCAB)}
            i2w = {i: w for w, i in w2i.items()}

            np.random.seed(0)
            # Logit table: simulates a model that learns simple co-occurrence
            LOGIT_TABLE = np.random.randn(V, V) * 0.5

            # Boost "the → cat" and "cat → sat" etc.
            LOGIT_TABLE[w2i["the"]][w2i["cat"]] += 5
            LOGIT_TABLE[w2i["the"]][w2i["mat"]] += 4
            LOGIT_TABLE[w2i["the"]][w2i["dog"]] += 3
            LOGIT_TABLE[w2i["cat"]][w2i["sat"]] += 5
            LOGIT_TABLE[w2i["cat"]][w2i["is"]]  += 3
            LOGIT_TABLE[w2i["sat"]][w2i["on"]]  += 5
            LOGIT_TABLE[w2i["on"]] [w2i["the"]] += 6   # ← creates a loop: the→cat→sat→on→the
            LOGIT_TABLE[w2i["mat"]][w2i["."]]   += 4
            LOGIT_TABLE[w2i["."]  ][w2i["<EOS>"]] += 4

            def softmax(z):
                z = z - z.max()
                return np.exp(z) / np.exp(z).sum()

            def generate(start_token, max_tokens=25, temperature=1.0,
                         rep_penalty=1.0, top_k=None, seed=42):
                \"\"\"
                rep_penalty: divide logit of already-seen tokens by this factor (>1).
                top_k:       keep only top-K tokens before sampling.
                \"\"\"
                np.random.seed(seed)
                ids      = [w2i[start_token]]
                seen_ids = Counter()

                for _ in range(max_tokens):
                    cur_id = ids[-1]
                    logits = LOGIT_TABLE[cur_id].copy()

                    # Apply repetition penalty
                    if rep_penalty > 1.0:
                        for seen_id, cnt in seen_ids.items():
                            # Penalty grows with repeat count
                            logits[seen_id] /= (rep_penalty ** cnt)

                    # Apply Top-K
                    if top_k is not None and top_k < V:
                        threshold = np.sort(logits)[::-1][top_k - 1]
                        logits    = np.where(logits >= threshold, logits, -1e10)

                    probs  = softmax(logits / temperature)
                    nxt_id = np.random.choice(V, p=probs)

                    if i2w[nxt_id] == "<EOS>":
                        break
                    seen_ids[nxt_id] += 1
                    ids.append(nxt_id)

                return " ".join(i2w[i] for i in ids)

            print("═" * 65)
            print("  Repetition Penalty — Demonstration")
            print("═" * 65)
            print()

            configs = [
                ("Greedy (T=0.1, no penalty)",  0.1,  1.0, None),
                ("T=1.0, no penalty",           1.0,  1.0, None),
                ("T=1.0, penalty=1.1",          1.0,  1.1, None),
                ("T=1.0, penalty=1.3",          1.0,  1.3, None),
                ("T=1.0, penalty=1.5",          1.0,  1.5, None),
                ("T=0.8, penalty=1.3, K=8",     0.8,  1.3, 8),
            ]

            for label, temp, rep, k in configs:
                text = generate("the", max_tokens=20, temperature=temp,
                                rep_penalty=rep, top_k=k, seed=42)
                token_list = text.split()
                unique     = len(set(token_list))
                total      = len(token_list)
                diversity  = unique / total if total > 0 else 0
                print(f"  Config: {label}")
                print(f"    Output:    {text}")
                print(f"    Tokens: {total}  Unique: {unique}  Diversity: {diversity:.2f}")
                print()

            # ── Why Greedy Loops ─────────────────────────────────────────────────
            print("── Why Greedy Decoding Loops: The Probability Cycle ──")
            print()
            cycle = ["the", "cat", "sat", "on"]
            for i in range(len(cycle)):
                cur  = cycle[i]
                probs = softmax(LOGIT_TABLE[w2i[cur]])
                top3  = np.argsort(probs)[::-1][:3]
                print(f"  After '{cur:5s}' →  top 3 next tokens:  " +
                      "  ".join(f"{i2w[t]}({probs[t]:.2f})" for t in top3))
            print()
            print("  Notice 'on' → 'the' closes the loop. Greedy follows this every time.")
            print("  Repetition penalty reduces 'the' logit each time it appears,")
            print("  eventually making a different continuation more likely.")
        """,
    },

}

# Dedent all code strings
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
        scripts_dir = _SCRIPTS_DIR
    if main_script is None:
        main_script = _MAIN_SCRIPT

    if "lm_step_status"  not in st.session_state:
        st.session_state.lm_step_status  = {}
    if "lm_step_outputs" not in st.session_state:
        st.session_state.lm_step_outputs = {}

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

def get_topic_data():
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "operations":   OPERATIONS,
    }

def get_content():
    """Return all content for this topic module."""
    return {
        "display_name":      DISPLAY_NAME,
        "icon":              ICON,
        "subtitle":          SUBTITLE,
        "theory":            THEORY,
        "theory_raw":        THEORY,
        "visual_html":       "",
        "operations":        OPERATIONS,
        "render_operations": render_operations,
    }
