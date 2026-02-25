"""Module: 03 · Transformer LLMs"""

import os
import re
import sys
import textwrap
from pathlib import Path
import base64

TOPIC_NAME   = "Transformer LLMs — Detailed Breakdown"
DISPLAY_NAME = "03 · Transformer LLMs"
ICON         = "🏗️"
SUBTITLE     = "GPT, BERT and T5 families — decoder, encoder and seq2seq architectures."

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / "Implementation" / "Transformer_Implementation" / "scripts"
_MAIN_SCRIPT  = _SCRIPTS_DIR / "transformer_main.py"

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

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## Overview
The **Transformer** is the architecture that powers every modern large language model.
Introduced in "Attention Is All You Need" (Vaswani et al., 2017), it replaced RNNs by
processing all tokens in parallel via **self-attention** — a mechanism that lets every
token directly read and weight any other token in the sequence.

Three major families have emerged from this single architecture:
- **Decoder-only** (GPT, LLaMA, Claude, Mistral) — autoregressive text generation
- **Encoder-only** (BERT, RoBERTa, DeBERTa) — deep bidirectional representations
- **Encoder-Decoder** (T5, BART, mT5) — seq2seq tasks (translation, summarisation)

Understanding the Transformer means understanding four things in order:
self-attention → multi-head attention → the full Transformer block → how blocks
are stacked and trained.


The Core Difference in utilization of the three major families and  What They Produce ?

Encoder-only models read a sequence and produce a rich contextual representation of it. 
Every token's final vector is informed by every other token in both directions — left and right. 
The model is optimized to understand text deeply, not generate it.

Decoder-only models read a sequence and produce the next token. Each token can only see tokens before it (causal mask). 
The model is optimized to continue text.

This single structural difference — bidirectional vs. left-only attention — determines everything about where each is useful.
 
## **Encoder-Only — When You Need to Understand/Classify**

**The key insight is: you already have the full text, and you want to extract something from it or make a decision about it.**

**You're not generating anything new. You're mapping existing text → label, number, category, or another fixed-size output.**

**Text Classification** — "Is this email spam or not?" The model reads the entire email bidirectionally, 
pools the [CLS] token, and a classifier head outputs spam/not spam. You need the full bidirectional context because the 
word "free" near the beginning matters more if you've already seen "money" at the end.

**Named Entity Recognition (NER)** — Labeling each token as PERSON, ORG, LOCATION, or O. 
Every word's label depends on the surrounding words in both directions. 
"Apple" is PRODUCT or ORG depending on context that can come after it.

**Semantic Search / Embeddings** — You want to encode a sentence as a single dense vector so you can find similar sentences.
The vector needs to capture the full meaning of the sentence, which requires seeing the whole sentence at once. 
BERT-style models are still used heavily in retrieval systems (like the first stage of RAG pipelines).

**Question Answering (extractive)** — "Given this passage, where in the text is the answer to this question?" 
The model reads the full passage and question together bidirectionally and predicts start/end token positions of the 
answer span. No generation involved.

**Semantic Similarity** — "Are these two sentences paraphrases?" Pass both sentences through the encoder, 
compare their representations. The model needs full context of each sentence to judge this.

**The pattern:** the output is always a fixed-size decision derived from a complete input. 
Bidirectionality makes the representations richer because every token can attend to the whole document, 
not just what came before it.


## **Decoder-Only — When You Need to Generate**

The key insight is: you have a starting context and you want to produce new text that continues or responds to it.

The causal mask isn't a limitation — it's the feature. It enforces that generation at position t only depends on 
positions before t, which is exactly the correct constraint for producing coherent left-to-right text.

**Open-ended Generation** — "Write a story about a knight." The model has no fixed endpoint; 
it just keeps producing the next most probable token given all previous tokens until it decides to stop.

**Instruction Following / Chat** — "Explain quantum entanglement simply." 
The model sees the instruction and generates a response token by token. 
There is no predetermined response to extract — the response must be created.

**Code Completion** — "def fibonacci(n):" → complete the function. 
Each token of code depends on all prior code, not future code.

**Summarization (modern approach)** — Given a document, generate a summary. The model generates the summary autoregressively. 
Note that encoder-decoder (T5, BART) was historically preferred here, 
but modern decoder-only models do this just as well with prompting.

**In-context learning / few-shot prompting** — Showing examples in the context window and asking the model to follow the pattern.
The causal structure means the model "reads" the examples as context for generating the answer.

**The pattern:** the output length is not fixed in advance, the output is new text that didn't exist in the input, 
and it's generated sequentially left to right.

## **The Logic Behind Why Each Architecture Suits Its Use Case**

For encoder-only, bidirectionality gives strictly better representations for understanding tasks. 
If you mask the word "bank" in the sentence "He sat on the river bank fishing," 
a bidirectional model can use "river" (before) and "fishing" (after) to resolve the ambiguity. 
A causal model at the word "bank" can only see "He sat on the river" — it misses the disambiguating context that comes after. 
For any task where you have the full text and want to understand it, bidirectional is just better.

For decoder-only, causal masking is logically required. If a language model could see the future token while predicting it, 
training loss would be meaningless — the model would just memorize the token it's about to predict. 
The causal constraint enforces the generative contract: predict each token only from prior context. 
This also means that during inference, you can extend the context with each generated token and run the model again in a consistent way.

There's also a training signal difference. BERT-style MLM only computes loss on the ~15% of tokens that are masked. 
GPT-style CLM computes loss at every single position. This means decoder models get ~7× more gradient signal per sequence, 
which is one reason they train more sample-efficiently at scale.

## **Why Decoder-Only Won at Scale**
Historically, encoder-decoder (T5, BART) was popular for tasks like translation and summarization because 
it combined bidirectional understanding (encoder) with generation (decoder). 

But as models scaled to billions of parameters, 
decoder-only models showed an unexpected property: with enough capacity and data, they became excellent at understanding 
tasks too, just through prompting and in-context learning.

GPT-3 could do classification by framing it as text completion ("The sentiment of this review is: ___"). 
It could do extractive QA by generating the answer as text. The "understanding" tasks that seemed to require an encoder 
could be reformulated as generation tasks. And since you only need to maintain one type of model, 
the ecosystem consolidated around decoder-only. 
Today, BERT-style encoders survive primarily in two niches: high-throughput embedding/retrieval where you need fast 
fixed-size vectors, and constrained classification tasks where you don't need generation at all and want something smaller and faster.

---

## Transformer Architecture — Full Landscape

                        TRANSFORMER ARCHITECTURE — FULL VIEW

    ══════════════════════════════════════════════════════════════════════════════════
    
                                   INPUT TOKENS
                                  [w₁, w₂, …, wₙ]
                                        │
                               ┌────────▼─────────┐
                               │  Token Embedding │   shape: (seq_len, d_model)
                               │  E ∈ ℝ^|V|×d     │
                               └────────┬─────────┘
                                        │  +
                               ┌────────▼────────┐
                               │Positional Encod.│   sin/cos or RoPE or ALiBi
                               └────────┬────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │ ENCODER BLOCK ×N  │  DECODER BLOCK ×N │
                    │                   │                   │
                    │  ┌─────────────┐  │  ┌─────────────┐  │
                    │  │ LayerNorm   │  │  │ LayerNorm   │  │
                    │  ├─────────────┤  │  ├─────────────┤  │
                    │  │ Multi-Head  │  │  │ Masked MHA  │  │
                    │  │  Attention  │  │  │ (causal)    │  │
                    │  │ (Bidir.)    │  │  ├─────────────┤  │
                    │  ├─────────────┤  │  │ Cross-Attn  │  │
                    │  │  Residual   │  │  │ (enc→dec)   │  │
                    │  │  + Norm     │  │  ├─────────────┤  │
                    │  ├─────────────┤  │  │  Residual   │  │
                    │  │ Feed-Fwd    │  │  │  + Norm     │  │
                    │  │ Network     │  │  ├─────────────┤  │
                    │  │ (FFN)       │  │  │ Feed-Fwd    │  │
                    │  ├─────────────┤  │  │ Network     │  │
                    │  │  Residual   │  │  ├─────────────┤  │
                    │  │  + Norm     │  │  │  Residual   │  │
                    │  └─────────────┘  │  │  + Norm     │  │
                    │  × N layers       │  └─────────────┘  │
                    │                   │  × N layers       │
                    │  BERT: N=12/24    │  GPT-2: N=12/24   │
                    │  RoBERTa: N=12/24 │  GPT-3: N=96      │
                    │                   │  LLaMA-3: N=32    │
                    └───────────────────┴───────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  Linear + Softmax │   (d_model → |V|)
                              │  (LM Head)        │   for next-token logits
                              └───────────────────┘

    ══════════════════════════════════════════════════════════════════════════════════

---

## 1. Self-Attention — The Core Mechanism

Self-attention answers the question: **"For each token, which other tokens should
I gather information from, and how much?"**

### The Query-Key-Value Framework

Every input token xᵢ ∈ ℝ^d is projected into three vectors:

    Q = X Wᵠ    (Query)   — "What am I looking for?"
    K = X Wᴷ    (Key)     — "What do I contain?"
    V = X Wᵛ    (Value)   — "What will I contribute if selected?"

where Wᵠ, Wᴷ, Wᵛ ∈ ℝ^(d_model × d_k) are learned weight matrices.

### Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V

Step-by-step:

    Step 1 — Dot products:    S = Q Kᵀ            shape: (seq_len, seq_len)
    Step 2 — Scale:           S = S / √d_k         prevents softmax saturation
    Step 3 — Mask (optional): S[i,j] = -∞ if j > i (causal/decoder mask)
    Step 4 — Softmax:         A = softmax(S, dim=-1) shape: (seq_len, seq_len)
    Step 5 — Aggregate:       out = A · V          shape: (seq_len, d_k)

**Why √d_k?** When d_k is large, dot products grow large in magnitude, pushing
softmax into saturation (near-zero gradients). Dividing by √d_k keeps the
variance of the dot products near 1.0 regardless of dimension.

### A Concrete 3-Token Example

    Tokens:  ["The", "cat", "sat"]
    d_model = 4,   d_k = 4

    Q = [[0.2, 0.1, 0.8, 0.3],    ← "The"  is looking for …
         [0.9, 0.1, 0.2, 0.4],    ← "cat"  is looking for …
         [0.1, 0.7, 0.3, 0.5]]    ← "sat"  is looking for …

    K = [[0.3, 0.5, 0.1, 0.2],    ← "The"  offers …
         [0.8, 0.2, 0.7, 0.1],    ← "cat"  offers …
         [0.1, 0.4, 0.9, 0.6]]    ← "sat"  offers …

    Score matrix S = Q @ Kᵀ / √4:
         The    cat    sat
    The [0.30,  0.64,  0.61]   ← "The" attends mostly to "cat" and "sat"
    cat [0.44,  0.87,  0.61]   ← "cat" attends mostly to itself
    sat [0.55,  0.68,  0.89]   ← "sat" attends mostly to itself

    A = softmax(S):
         The    cat    sat
    The [0.27,  0.37,  0.36]
    cat [0.25,  0.42,  0.33]
    sat [0.23,  0.30,  0.47]

    Output = A @ V            ← each position is a weighted blend of all Values

### What Attention Learns

Different attention heads learn to encode different relationships:
- Syntactic heads: subject-verb, noun-adjective agreement
- Coreference heads: pronoun → antecedent ("it" → "the cat")
- Positional heads: attend to previous/next token
- Rare token heads: attend to unusual or informative tokens

---

## 2. Causal (Decoder) Masking

In autoregressive models (GPT, LLaMA), token i must NOT attend to future tokens.
This is enforced with a **causal mask** — set all positions (i, j) where j > i to -∞
before the softmax, so those weights become exactly 0.

    Mask for seq_len=4:
         pos0   pos1   pos2   pos3
    pos0 [  0,   -∞,   -∞,   -∞  ]    pos0 can only see itself
    pos1 [  0,    0,   -∞,   -∞  ]    pos1 can see pos0 and itself
    pos2 [  0,    0,    0,   -∞  ]    pos2 can see pos0, pos1, itself
    pos3 [  0,    0,    0,    0  ]    pos3 can see all prior positions

This is called a **lower-triangular** mask. It enables processing all positions
in parallel during training while preserving the left-to-right constraint.

Without this mask, the model would "cheat" — seeing future tokens when predicting
the current one, making training loss meaningless.

---

## 3. Multi-Head Attention (MHA)

Running a single attention function gives one "view" of the sequence.
**Multi-Head Attention** runs h independent attention operations in parallel,
each with its own projection matrices, then concatenates and projects:

    head_i = Attention(Q Wᵢᵠ,  K Wᵢᴷ,  V Wᵢᵛ)
    MHA(Q, K, V) = Concat(head₁, …, headₕ) · Wᴼ

    where Wᵢᵠ, Wᵢᴷ, Wᵢᵛ ∈ ℝ^(d_model × d_k)  and  d_k = d_model / h

**Dimension tracking (GPT-2 base):**

    d_model = 768,   h = 12 heads,   d_k = d_v = 64

    Input X:                (seq_len, 768)
    Each Wᵢᵠ, Wᵢᴷ, Wᵢᵛ:     (768, 64)
    Each head output:       (seq_len, 64)
    After concat:           (seq_len, 768)   ← h × d_k = 12 × 64 = 768
    After Wᴼ ∈ ℝ^(768,768): (seq_len, 768)   ← same shape as input

**Why multiple heads?**
Each head can specialise in a different relationship. One head tracks syntax,
another coreference, another local context. Using a single large head would
mix all these signals into one averaged view.

### Grouped-Query Attention (GQA) — Modern Variant

Used by LLaMA-2/3, Mistral, Gemma. Instead of h unique K, V pairs (one per head),
GQA shares K and V across groups of heads. If h=32 query heads but only g=8 KV heads:

    Each KV head is shared by h/g = 4 query heads.

This reduces the KV-cache memory by 4× at inference — critical for long contexts.

### Multi-Query Attention (MQA)

Extreme case: a single shared K and V for all h query heads.
Even more memory-efficient; slight quality degradation. Used by GPT-J, Falcon.

---

## 4. The Feed-Forward Network (FFN)

After each attention sublayer, every position independently passes through
a two-layer FFN (identical weights for each position):

    FFN(x) = max(0, x W₁ + b₁) W₂ + b₂       ← ReLU activation (original)
    FFN(x) = GELU(x W₁ + b₁) W₂ + b₂         ← GELU (GPT-2, BERT)
    FFN(x) = SwiGLU(x W₁, x W₃) W₂ + b₂      ← SwiGLU (LLaMA, PaLM)

    Dimension expansion:  d_model → d_ff → d_model
    Typical:              d_ff = 4 × d_model

    GPT-2   :   768 → 3072 → 768
    GPT-3   :   12288 → 49152 → 12288
    LLaMA-3 :   4096 → 14336 → 4096  (uses SwiGLU, so slightly different ratio)

**What does the FFN do?**
Attention collects and mixes information across positions.
The FFN then processes each position independently — it acts like a key-value
memory store. Research shows FFN layers store factual associations
("Eiffel Tower" → "Paris", "H₂O" → "water"), while attention layers route
information between positions.

### SwiGLU Activation (Modern Standard)

    SwiGLU(x, W₁, W₃) = (x W₁ ⊙ σ(x W₃ · β)) W₂

    σ = sigmoid,  ⊙ = element-wise multiply

SwiGLU has three weight matrices instead of two. It consistently outperforms
ReLU and GELU on large-scale benchmarks and is used by LLaMA-1/2/3, PaLM, and Gemma.

---

## 5. Layer Normalisation & Residual Connections

### Residual Connections (Skip Connections)

Every sublayer (attention, FFN) is wrapped in a residual connection:

    x = x + Sublayer(x)

This allows gradients to flow directly from the output back to early layers
without passing through the sublayer's transformation. Essential for training
networks with dozens or hundreds of layers.

### Pre-Norm vs Post-Norm

**Post-Norm (original Transformer, BERT):**

    x = LayerNorm( x + Sublayer(x) )

**Pre-Norm (GPT-2, LLaMA, modern standard):**

    x = x + Sublayer( LayerNorm(x) )

Pre-Norm is more stable during training — no warm-up learning rate schedule required,
gradients don't vanish as easily. Almost all post-2019 LLMs use Pre-Norm.

### RMSNorm (Simplified LayerNorm)

Used by LLaMA, Mistral, and most modern models. Removes the mean-centering step:

    LayerNorm(x) = γ · (x − μ) / √(σ² + ε)  + β    ← 2 learned params per dim
    RMSNorm(x)   = γ · x / RMS(x)                    ← 1 learned param per dim

    RMS(x) = √( (1/d) ∑ xᵢ² )

15–20% faster than LayerNorm; comparable quality. γ and β are learned scale/shift.

---

## 6. Transformer Block — The Full Picture

A single Transformer block (Pre-Norm style) for a decoder-only model:

    Input: x ∈ ℝ^(seq_len × d_model)

    # ── Attention sublayer ──────────────────────────────────────────────────
    x_norm   = RMSNorm(x)
    attn_out = MultiHeadAttention(x_norm, mask=causal_mask)
    x        = x + attn_out                    # residual

    # ── FFN sublayer ─────────────────────────────────────────────────────────
    x_norm  = RMSNorm(x)
    ffn_out = FFN(x_norm)
    x       = x + ffn_out                     # residual

    Output: x ∈ ℝ^(seq_len × d_model)  (same shape as input — stackable)

The block is stackable: N identical blocks are chained. The output of block i
feeds directly into block i+1. Each block refines the representation.

---

## 7. Model Families — Architecture Deep Dive

### A) Decoder-Only (GPT / LLaMA / Mistral / Claude)

The dominant architecture for generative LLMs since GPT-2.

    Structure:   Token Embedding + Positional Encoding
                 → N × Decoder Block (with causal mask)
                 → RMSNorm
                 → Linear (d_model → |V|) + Softmax  [LM head]

    Key property: causal (triangular) self-attention mask.
                  Token i sees only positions ≤ i.

    Training:    Autoregressive LM — predict next token at every position.
    Inference:   Sample one token at a time, append, repeat.

    Model Family Comparison:
    ┌───────────────┬──────────┬──────┬────────┬────────┬──────────┬──────────────┐
    │ Model         │ Params   │  N   │ d_model│ Heads  │  d_ff    │ Context      │
    ├───────────────┼──────────┼──────┼────────┼────────┼──────────┼──────────────┤
    │ GPT-2 Small   │ 117M     │  12  │   768  │  12    │  3,072   │ 1,024        │
    │ GPT-2 Large   │ 774M     │  36  │ 1,280  │  20    │  5,120   │ 1,024        │
    │ GPT-3         │ 175B     │  96  │ 12,288 │  96    │ 49,152   │ 2,048        │
    │ LLaMA-2 7B    │   7B     │  32  │ 4,096  │  32    │ 11,008   │ 4,096        │
    │ LLaMA-3 8B    │   8B     │  32  │ 4,096  │  32    │ 14,336   │ 8,192        │
    │ LLaMA-3 70B   │  70B     │  80  │ 8,192  │  64    │ 28,672   │ 8,192        │
    │ Mistral 7B    │   7B     │  32  │ 4,096  │  32    │ 14,336   │ 32,768       │
    └───────────────┴──────────┴──────┴────────┴────────┴──────────┴──────────────┘

### B) Encoder-Only (BERT / RoBERTa / DeBERTa)

    Structure:   Token Embedding + Positional Encoding
                 → N × Encoder Block (bidirectional self-attention — NO mask)
                 → Pooled [CLS] representation or per-token representations

    Key property: bidirectional — every token attends to every other token.
                  Richer contextual representations than autoregressive models.

    Training:    Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
                 15% of tokens are masked; model predicts them from full context.

    Use cases:   Classification, NER, question answering, semantic similarity.
                 NOT generative — you can't "generate text" from BERT directly.

    BERT Base:   12 layers,  768 hidden,  12 heads,  110M params
    BERT Large:  24 layers, 1024 hidden,  16 heads,  340M params
    RoBERTa:     Same architecture; trained longer, no NSP, better data.
    DeBERTa-v3:  Disentangled attention + ELECTRA pretraining.

### C) Encoder-Decoder (T5 / BART / mT5)

    Structure:   Encoder stack (bidirectional) + Decoder stack (causal)
                 Decoder attends to encoder via cross-attention.

    Training:    T5: "text-to-text" — every task framed as string-in → string-out
                     Span Corruption: mask spans of tokens, predict them
                 BART: denoising autoencoder — corrupt input, reconstruct it

    Use cases:   Translation, summarisation, structured prediction, Q&A.

    Cross-Attention: the decoder's Q comes from the decoder,
                     but K and V come from the encoder's output.
    This allows the decoder to "read" the full encoded input at every step.

    T5-Base:     220M params   T5-Large:  770M   T5-11B: 11B
    BART-Large:  400M params

---

## 8. Training Transformers — The Full Pipeline

### Stage 0: Data Preparation

    1. Collect raw text (web crawl, books, code, papers…)
    2. Deduplicate (MinHash, exact dedup) — duplicates harm generalisation
    3. Quality filter (language ID, perplexity filter, heuristic rules)
    4. Tokenize with BPE / SentencePiece → integer token IDs
    5. Pack sequences into fixed-length chunks (e.g., 2048 or 4096 tokens)

    LLaMA-3 pretraining data: ~15 trillion tokens across 8B and 70B models.

### Stage 1: Pre-Training

**Objective:**

    Decoder-only:  L = −(1/T) ∑ log P(wₜ | w₁,…,wₜ₋₁)     (next-token LM)
    Encoder-only:  L_MLM = −∑ log P(wᵢ | w₋ᵢ)              (masked token)
    Enc-Dec (T5):  L = −(1/T) ∑ log P(wₜ | source, w₁,…,wₜ₋₁) (span predict)

**Optimiser:** AdamW (Adam + weight decay)

    θₜ = θₜ₋₁ − α · m̂ₜ / (√v̂ₜ + ε) − λ θₜ₋₁   (weight decay term)

    Typical hyperparameters:
    β₁ = 0.9,  β₂ = 0.95,  ε = 1e-8,  λ = 0.1

**Learning Rate Schedule:**

    Warmup → Cosine decay:
    0 → peak_lr over first ~2000 steps (warmup)
    peak_lr → peak_lr × 0.1 over the remaining steps (cosine)

    LLaMA-3 8B:  peak_lr = 3e-4,  warmup = 2,000 steps,  total = ~1M steps

**Batch Size:** Very large — gradient noise needs to be averaged over many tokens.

    GPT-3:    0.5M tokens per batch
    LLaMA-2:  4M  tokens per batch
    LLaMA-3:  16M tokens per batch (128K sequences × 128 tokens)

**Gradient Clipping:**  Clip gradient norm to 1.0 — prevents exploding gradients.

### Stage 2: Supervised Fine-Tuning (SFT)

After pretraining, the model predicts text but doesn't follow instructions.
SFT trains on (instruction, response) pairs with the same LM loss,
but only computing loss on the response tokens (not the instruction).

    Data: 10K–1M high-quality human-written (prompt, response) pairs
    Epochs: 1–3 (avoid overfitting on small SFT dataset)
    Learning rate: 10–100× smaller than pretraining (1e-5 to 1e-4)

### Stage 3: RLHF / DPO (Alignment)

**RLHF (Reinforcement Learning from Human Feedback):**
1. Train a reward model from human preference comparisons
2. Use PPO (Proximal Policy Optimisation) to maximise reward
3. KL divergence penalty prevents the model from drifting too far from SFT

**DPO (Direct Preference Optimisation) — Modern alternative:**
Skips the reward model entirely. Directly optimises on preference pairs
(chosen response yᵥ, rejected response yₗ) using a closed-form loss:

    L_DPO = −E[ log σ( β · log(π(yᵥ|x)/π_ref(yᵥ|x))
                      − β · log(π(yₗ|x)/π_ref(yₗ|x)) ) ]

DPO is simpler, more stable, and achieves competitive results to RLHF.
Used by Zephyr, Tulu-2, and many open-source instruction-tuned models.

---

## 9. Scaling Laws — How Performance Grows with Scale

Empirically derived by Kaplan et al. (2020, OpenAI) and Hoffmann et al. (2022, DeepMind).

### Power-Law Scaling

    L(N) ≈ A / N^α      (loss vs. parameters, fixing compute)
    L(D) ≈ B / D^β      (loss vs. dataset size, fixing compute)
    L(C) ≈ G / C^γ      (loss vs. compute, optimal allocation)

Loss follows a **power law** — each 10× increase in parameters or data gives
a predictable, consistent improvement in loss.

### Chinchilla Scaling Laws (Hoffmann et al., 2022)

Key finding: **for a given compute budget, models are undertrained**.
The optimal ratio is approximately:

    D ≈ 20 × N     (dataset tokens ≈ 20 × number of parameters)

| Model (pre-Chinchilla) | Params | Tokens trained | Chinchilla-optimal tokens |
|------------------------|--------|----------------|---------------------------|
| GPT-3                  | 175B   | 300B           | 3.5T                      |
| Gopher                 | 280B   | 300B           | 5.6T                      |
| Chinchilla             |  70B   | 1.4T           | 1.4T ✓ (optimal)          |
| LLaMA-3 8B             |   8B   | 15T            | much more than optimal    |

**Post-Chinchilla insight:** If inference cost matters (which it does commercially),
you want a *smaller model trained on more data* — it's cheaper to run and equally
capable at the compute optimum.

### What Scales and What Doesn't

**Scales reliably:** Loss, downstream benchmark performance, in-context learning,
few-shot accuracy, code generation.

**Doesn't scale predictably:** Specific capabilities (arithmetic, reasoning) can
appear suddenly at certain scales — called **emergent abilities**. Hard to predict.

---

## 10. Parameter Count — Where Are the Parameters?

For a decoder-only Transformer with:
N layers, d_model dimensions, h attention heads, d_ff = 4 d_model, |V| vocabulary size:

    Embedding layer:         |V| × d_model
    Per-layer attention:     4 × d_model²     (Q, K, V, O projections)
    Per-layer FFN:           8 × d_model²     (2 × [d_model × 4d_model])
    Per-layer norms:         4 × d_model      (small, negligible)
    LM head:                 |V| × d_model    (often tied to embedding)

    Total ≈ 2|V|d + N × 12 d²

    GPT-2 base:  2×50257×768 + 12×12×768² = 77M + 85M ≈ 117M   ✓
    GPT-3:       2×50257×12288 + 96×12×12288² ≈ 175B            ✓

**The attention vs FFN split:**
In large models, FFN parameters dominate:
- Attention: ~4 d² per layer
- FFN: ~8 d² per layer (with 4× expansion)
So 2/3 of model parameters live in FFN layers.

---

## 11. Positional Encoding in Modern LLMs

(See also Module 01 — Tokenization & Embeddings for the full breakdown)

**Sinusoidal (original Transformer):** Fixed, no learned parameters.
**Learned (BERT, GPT-2):** Trained position vectors, max context fixed at train time.
**RoPE (LLaMA, Mistral, GPT-NeoX):** Encodes position as rotation in head dimension.
**ALiBi (MPT, BLOOM):** Linear attention bias by distance; extrapolates well.
**YaRN / LongRoPE:** Extensions of RoPE for 100K+ context windows.

---

## 12. Flash Attention — Solving the O(n²) Memory Problem

Standard attention computes and materialises the full (seq_len × seq_len) score matrix in
GPU High-Bandwidth Memory (HBM). For a 4096-token sequence this is 4096² = 16M floats ≈ 32 MB
**per layer per head** — and reading/writing it repeatedly dominates runtime. It's IO-bound,
not compute-bound.

**Flash Attention (Dao et al., 2022)** fuses the attention computation into a single kernel
using *tiling* over SRAM (on-chip cache, ~100× faster than HBM):

    Standard: for each row i:
        S[i, :] = Q[i] · Kᵀ / √dₖ              # read all K from HBM → SRAM
        A[i, :] = softmax(S[i, :])               # write S to HBM, read back
        out[i]  = A[i, :] · V                    # read all V from HBM

    Flash:    for each tile of rows (chunk at a time):
        Tile fits in SRAM → compute partial softmax with running max & denominator
        Never materialise the full (n×n) matrix in HBM
        Output computed on-the-fly; only final result written to HBM

**Memory:** O(n²) → O(n).  **Speed:** 2–4× faster wall-clock on A100s.
**Backward pass:** recomputes attention tile from Q, K, V on-the-fly during backward
(no need to store A), trading ~20% more FLOPS for much less memory.

Flash Attention 2 (2023) further improved parallelism across the sequence dimension.
Flash Attention 3 (2024) targets H100 hardware, exploiting asynchronous operations.

Every serious LLM training run uses Flash Attention. It is now the default in PyTorch
(`torch.nn.functional.scaled_dot_product_attention` calls it automatically).

    ┌──────────────────────────────────────────────────────────────────┐
    │  SRAM (fast, ~20 MB)  ← tiles of Q, K, V fit here                │
    │                                                                  │
    │  HBM  (slow, ~80 GB)  ← full Q, K, V matrices, final output      │
    │                                                                  │
    │  Flash Attention: keep computation in SRAM, minimise HBM reads   │
    └──────────────────────────────────────────────────────────────────┘

---

## 13. KV Cache — Efficient Autoregressive Inference

During training, all tokens are processed in parallel (thanks to the causal mask).
During inference, tokens are generated **one at a time**. Without caching:

    Generating token t requires attention over all previous t tokens.
    Generating all T tokens → O(T²) total attention computations.

The **KV Cache** stores the Key and Value tensors for all previously computed tokens:

    Step 1: process prompt [t₁, t₂, … tₚ] → store K₁…Kₚ, V₁…Vₚ in cache
    Step 2: generate token tₚ₊₁ using cached K, V + new query Qₚ₊₁
    Step 3: append Kₚ₊₁, Vₚ₊₁ to cache → repeat

Only the new token needs a forward pass through the network.
All previous K, V vectors are read from cache — no recomputation.

**KV Cache Memory (per token, per layer):**

    Cache size = 2 × seq_len × n_kv_heads × d_head × bytes_per_element × n_layers

    LLaMA-3 8B (BF16, 8K context, 8 KV heads, 32 layers, d_head=128):
    = 2 × 8192 × 8 × 128 × 2 × 32 = 1.07 GB per sequence

    LLaMA-3 70B (BF16, 8K context, 8 KV heads, 80 layers, d_head=128):
    = 2 × 8192 × 8 × 128 × 2 × 80 = 2.68 GB per sequence

This is why long-context inference is memory-bound and why GQA/MQA matters so much —
fewer KV heads means a proportionally smaller cache.

---

## 14. Mixed Precision Training — BF16 / FP16

LLMs are trained in reduced precision to cut memory and speed up tensor core operations.

**FP32 (single precision):** 32 bits, range ≈ ±3.4×10³⁸, precision ≈ 7 decimal digits.
**FP16 (half precision):** 16 bits, range ≈ ±65504, precision ≈ 3 decimal digits.
**BF16 (bfloat16):** 16 bits, range ≈ ±3.4×10³⁸, precision ≈ 2–3 decimal digits.

BF16 has the same exponent range as FP32 (8 exponent bits vs FP16's 5),
so it handles large gradient magnitudes without overflow — critical for training stability.
This is why BF16 is preferred over FP16 for LLM training.

**Mixed Precision Training (Micikevicius et al., 2018):**

    ┌────────────────────────────────────────────────────────────────┐
    │  FP32 master weights  ← optimizer state always in FP32         │
    │                                                                │
    │  BF16 forward pass    ← cast to BF16, compute activations      │
    │  BF16 backward pass   ← gradients computed in BF16             │
    │                                                                │
    │  FP32 optimizer step  ← cast grad to FP32, update master wt    │
    │  BF16 copy → GPU      ← cast updated weights back to BF16      │
    └────────────────────────────────────────────────────────────────┘

**Memory savings:** Weights stored in BF16 → 2 bytes/param (vs 4 bytes FP32).
But optimizer state (Adam m, v, FP32 master copy) = 12 bytes/param.
Total: ~16 bytes/param in mixed precision vs 4 bytes for weights-only FP32.

    LLaMA-3 8B:  8B × 16 bytes ≈ 128 GB  (training)
                 8B × 2  bytes ≈  16 GB  (inference in BF16)

**Loss scaling** (needed for FP16, not BF16): multiply loss by a large scalar before
backward to prevent gradient underflow, then divide before optimizer step.

---

## 15. Gradient Checkpointing — Trading Compute for Memory

During the backward pass, PyTorch stores **all intermediate activations** from the
forward pass (needed to compute gradients). For a large model this dominates memory.

    Memory per layer ≈ batch_size × seq_len × d_model × 4  (for a full block)
    LLaMA-3 8B, batch=8, seq=4096: ≈ 8 × 4096 × 4096 × 4 × 32 layers ≈ 16 GB

**Gradient Checkpointing** (activation recomputation):

    Normal:        store all activations → fast backward, high memory
    Checkpointing: store only checkpoint boundary activations
                   during backward, recompute each segment from its checkpoint

    Memory reduction: ~√N for N layers (checkpointing every √N layers)
    Compute overhead: ~33% more FLOPs (one extra forward pass worth of work)

In practice, every Transformer block boundary is a checkpoint — each block's
input is stored, intermediate activations within the block are recomputed during backward.

    PyTorch API: torch.utils.checkpoint.checkpoint(function, *inputs)

Combined with mixed precision, gradient checkpointing makes it feasible to train
very large models on limited hardware.

---

## 16. Dropout & Regularisation

**Dropout (Srivastava et al., 2014):** During training, randomly zero out activations
with probability p. At inference, multiply by (1-p) to maintain expected magnitude.

    Applied to:
    - Attention weights (after softmax)          — "attention dropout"
    - FFN hidden layer activations               — "residual dropout"
    - Embedding layer                            — "embedding dropout"

    Typical p: 0.1 (10%) for LLMs. Very large models often use p=0 or p=0.05
    because the dataset is so large that overfitting is not the primary concern.

**Weight Decay (L2 regularisation via AdamW):** Adds λ‖θ‖² to the loss, shrinking
weights toward zero each step. In AdamW this is applied directly to parameters
(not through the gradient), preserving Adam's adaptive scaling.

**Why dropout matters less at scale:** Models like GPT-3 use dropout=0 during
pretraining. With 300B tokens and 175B parameters, the model doesn't overfit.
SFT and RLHF fine-tuning stages do use small dropout to prevent memorisation.

---

## 17. Mixture of Experts (MoE)

Standard FFN: every token goes through the same FFN weights.
**MoE FFN:** N expert FFNs exist; a learned router selects the top-k for each token.

    Standard FFN:  out = FFN(x)                  — 1 FFN, all tokens
    MoE FFN:       scores = Router(x)             — (n_experts,) logits
                   top_k  = topk(scores)          — select k experts
                   out    = Σᵢ gᵢ · Expert_i(x)  — weighted sum of k expert outputs

    Router:  a simple linear layer W_r ∈ ℝ^(d_model × n_experts)
    Gates gᵢ: softmax over top-k scores (others zeroed out)

**Key insight:** MoE decouples *parameter count* (capacity) from *FLOPs per token*.

    Mixtral 8×7B:   8 experts × 7B each = 46B total params
                    Top-2 routing: only 2 experts active per token → ~13B active params
                    Similar FLOPs to a 13B dense model, but capacity of 46B

**Load balancing:** Without regularisation, the router collapses — all tokens go to
one or two experts. An auxiliary *load balancing loss* encourages uniform routing:

    L_balance = α × n_experts × Σᵢ (fᵢ · Pᵢ)

    fᵢ = fraction of tokens routed to expert i
    Pᵢ = mean router probability for expert i

**Models using MoE:** GPT-4 (reportedly), Mixtral 8×7B/8×22B, Gemini 1.5, Switch Transformer,
GLaM, Grok-1 (314B with 8 experts).

**MoE challenges:** Expert routing causes token dropping (if an expert is full),
load imbalance, and more complex distributed training (expert parallelism).

---

## 18. Sliding Window & Sparse Attention

Full self-attention is O(n²) in memory and compute. For long sequences this is prohibitive.

**Sliding Window Attention (Beltagy et al., 2020 — Longformer; used in Mistral 7B):**

    Each token attends only to a local window of W previous tokens.
    Complexity: O(n × W)  instead of O(n²)

    Mistral 7B: W = 4096 tokens (even with 32K context)
    Combined with rolling KV cache: only the last W K,V vectors are kept

    For very long documents, distant context is accessed through layer depth —
    lower layers handle local context, upper layers capture broader patterns.

**Sparse Attention patterns (BigBird, Longformer):**

    Global tokens: a few tokens (e.g. [CLS]) attend to everything
    Local tokens:  standard sliding window
    Random tokens: each token attends to r random positions

    This gives O(n) complexity while maintaining reasonable context coverage.

**Grouped Query Attention (GQA) vs Sliding Window:**
GQA reduces KV-cache memory (fewer KV heads); sliding window reduces attention complexity.
They address orthogonal problems and can be combined.

---

## 19. Weight Tying

In most LLMs, the **input embedding matrix** E ∈ ℝ^(|V| × d_model) and the
**LM head** (output projection d_model → |V|) share the same weights:

    Input: x_t (token id) → E[x_t] ∈ ℝ^d_model      (embedding lookup)
    Output: h_T → h_T · Eᵀ → logits ∈ ℝ^|V|          (tied projection)

**Why it works:** both layers learn that similar tokens should have similar representations.
The embedding that maps token → vector space is the same space used to score
next-token predictions. This constraint is linguistically motivated and empirically effective.

**Parameter savings:** 2 × |V| × d_model saved. For LLaMA-3 8B (|V|=128256, d=4096):
    2 × 128256 × 4096 ≈ 1.05 billion parameters saved — ~13% of model size.

Used by: GPT-2, GPT-3, LLaMA-1/2/3, Mistral, and most modern decoder-only LLMs.

---

## 20. Multi-GPU Distributed Training — Full Breakdown

Training a 70B model requires ~560 GB just for the optimizer state (FP32 weights + Adam m, v).
A single A100 has 80 GB. Distributing training across many GPUs is mandatory at scale.

### The Memory Problem — Why Distribution Is Necessary

    Model weights (BF16):       N × 2 bytes
    Gradients (FP16):           N × 2 bytes
    Optimizer FP32 master wts:  N × 4 bytes
    Adam momentum m (FP32):     N × 4 bytes
    Adam variance v (FP32):     N × 4 bytes
    ─────────────────────────────────────────
    Total per parameter:            16 bytes

    LLaMA-3 70B:  70B × 16 = 1,120 GB  ← needs 14+ A100 80GB GPUs just to store state

### Strategy 1: Data Parallelism (DP / DDP)

Every GPU holds a **full copy** of the model. Each GPU gets a different shard of the batch.
After backward, gradients are averaged via **AllReduce** across all GPUs.

    GPU 0: full model, batch shard 0 → grad_0 ──┐
    GPU 1: full model, batch shard 1 → grad_1 ──┤─→ AllReduce → avg_grad → update all
    GPU 2: full model, batch shard 2 → grad_2 ──┘

    Effective batch size = per_GPU_batch × n_GPUs
    Scaling constraint:  full model must fit on every single GPU

PyTorch DDP overlaps gradient AllReduce with the backward pass — communication
happens while later layers are still computing gradients. Near-linear scaling to ~1000s of GPUs.

### Strategy 2: Tensor Parallelism (TP) — Megatron-LM

Splits **individual weight matrices** across GPUs. Each GPU owns a horizontal slice.

    FFN W₁ ∈ ℝ^(d × 4d):  GPU 0 owns columns [0, d],  GPU 1 owns columns [d, 2d], etc.

    Forward: each GPU computes its shard's output → AllReduce to sum → next layer
    Attention: each GPU owns a subset of heads (complete heads, not partial)

    ┌────────────────────────────────────────────────────────┐
    │ GPU 0: head 0..7,   FFN cols 0..d_ff/2                 │
    │ GPU 1: head 8..15,  FFN cols d_ff/2..d_ff              │
    │ ← AllReduce after every layer ──────────────────────── │
    └────────────────────────────────────────────────────────┘

    Communication per layer: 2 AllReduce operations (one after attn, one after FFN)
    Requires fast interconnect (NVLink within a node — 600 GB/s for H100s)
    Typically used within a single node (≤ 8 GPUs)

### Strategy 3: Pipeline Parallelism (PP) — GPipe / PipeDream

Splits the model **vertically** — different GPUs hold different layer ranges.

    GPU 0: layers 0–7    GPU 1: layers 8–15    GPU 2: layers 16–23    GPU 3: layers 24–31

Activations flow forward across GPUs; gradients flow backward.
Naïve PP wastes time: GPU 1 waits while GPU 0 finishes the full sequence.

**Solution: micro-batching.** Split each batch into M micro-batches.

    Time →
    GPU 0: [F₀][F₁][F₂][F₃][  ][B₃][B₂][B₁][B₀]   ← "pipeline bubble"
    GPU 1:     [F₀][F₁][F₂][F₃][B₃][B₂][B₁][B₀]
    GPU 2:         [F₀][F₁][F₂][F₃][B₃][B₂][B₁][B₀]

    1F1B schedule (one Forward, one Backward per step): minimises bubble fraction to ~1/M
    Interleaved pipeline: each GPU holds multiple non-contiguous layer ranges → smaller bubble

PP requires much less bandwidth than TP (only layer-boundary activations are communicated).
It works across nodes over InfiniBand (~400 Gb/s).

### Strategy 4: ZeRO — Zero Redundancy Optimizer (DeepSpeed)

Key insight: standard DP replicates optimizer state, gradients, and parameters across all GPUs.
ZeRO eliminates this redundancy by **sharding** across data-parallel ranks.

    ZeRO Stage 1: Shard optimizer state across N GPUs
                  Each GPU stores 1/N of (FP32 master weights + m + v)
                  Memory reduction: ~4× for optimizer state

    ZeRO Stage 2: Additionally shard gradients
                  Each GPU accumulates only the gradients for its parameter shard
                  Memory reduction: ~8× vs baseline DP

    ZeRO Stage 3: Additionally shard the model parameters themselves
                  Before each layer: AllGather to reconstruct full weights
                  After backward:   ReduceScatter to update each GPU's shard
                  Theoretically unlimited parameter count with enough GPUs

    ZeRO-Infinity: Offload optimizer state/gradients to CPU RAM or NVMe SSD
                   Enables models >> GPU memory at the cost of CPU-GPU bandwidth

    ┌─────────────────────────────────────────────────────────────────┐
    │                     ZeRO Memory Comparison (N=8 GPUs)           │
    │  Baseline DP:     Ψ = 16 bytes/param on every GPU               │
    │  ZeRO Stage 1:   ~6 bytes/param (optimizer state sharded)       │
    │  ZeRO Stage 2:   ~4 bytes/param (+ grad sharded)                │
    │  ZeRO Stage 3:   ~2 bytes/param (+ params sharded)              │
    └─────────────────────────────────────────────────────────────────┘

### Strategy 5: FSDP — Fully Sharded Data Parallel (PyTorch)

PyTorch's native ZeRO-3 implementation. Each module's parameters are sharded by default.
Parameters are gathered (AllGather) just before their module's compute, then re-sharded.

Used by Meta's LLaMA training pipeline (torchtitan). Simpler API than DeepSpeed ZeRO-3
but achieves similar memory savings.

### Strategy 6: Expert Parallelism (EP)

Specific to MoE models. Different GPUs host different experts.
The router sends each token's embedding to the appropriate expert's GPU.
Requires high-bandwidth inter-GPU communication for token routing.
Used by GPT-4, Mixtral, and Switch Transformer at scale.

### 3D Parallelism — Combining All Strategies

Real LLM training at scale uses all three axes simultaneously:

    Total GPUs = TP degree × PP degree × DP degree

    Example — GPT-3 training (Microsoft/OpenAI):
        TP = 8   (tensor parallel within a node, 8 GPUs per node)
        PP = 8   (pipeline parallel across nodes, 8 pipeline stages)
        DP = 1024 replicas of the above TP×PP configuration
        Total GPUs: 8 × 8 × 1024 = 65,536 A100s

    LLaMA-3 405B (Meta):
        TP = 8,  PP = 16,  DP = many replicas
        Total: 16,384 H100 GPUs across ~2,048 nodes

    ┌────────────────────────────────────────────────────────────────┐
    │                  3D Parallelism Layout                         │
    │                                                                │
    │  Node 0        Node 1        Node 2        Node 3              │
    │ [G0][G1]...[G7][G8]...[G15][G16]...[G23][G24]...[G31]          │ 
    │  └──── TP ────┘  └─────────── PP ──────────────────┘           │
    │  └──────────────────────── DP replicas ─────────────────┘      │
    └────────────────────────────────────────────────────────────────┘

### Communication Primitives

    AllReduce:      every GPU sends its tensor; every GPU receives the elementwise sum
                    Used in DP for gradient averaging
                    Ring-AllReduce is bandwidth-optimal: O(N) bytes per GPU

    AllGather:      every GPU sends its shard; every GPU receives the full tensor
                    Used in ZeRO-3 to reconstruct parameters before compute

    ReduceScatter:  every GPU sends a shard; each GPU receives the reduced result for
                    its shard (inverse of AllGather)
                    Used in ZeRO-3 for gradient sharding

    Broadcast:      one GPU sends; all others receive — used for checkpoint sync

    Point-to-Point: send/recv between two GPUs — used in pipeline parallelism for
                    passing activations between pipeline stages

### Hardware Interconnects

    NVLink (within node):  up to 900 GB/s bidirectional (H100 NVLink 4)
                           Required for Tensor Parallelism (high bandwidth)

    PCIe (fallback):       ~64 GB/s — too slow for TP, acceptable for DP

    InfiniBand (between nodes): NDR InfiniBand: 400 Gb/s (50 GB/s) per link
                                Multiple rails: effective ~100–200 GB/s per node
                                Used for Pipeline Parallelism and DP AllReduce

    The choice TP vs PP vs ZeRO depends on the bottleneck:
    - Fast NVLink → TP within a node
    - Cross-node (InfiniBand) → PP or ZeRO DP
    - Memory-bound → ZeRO-3 or FSDP

### Gradient Accumulation — Simulating Large Batches

When GPU memory limits the per-step batch size, gradient accumulation simulates
a larger effective batch without extra memory:

    for step in range(total_steps):
        loss_accum = 0
        for micro in range(accumulation_steps):           # accumulate K micro-batches
            batch = get_micro_batch()
            loss  = model(batch) / accumulation_steps     # scale loss
            loss.backward()                               # accumulate grads
            loss_accum += loss.item()

        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()                             # clear after step

    Effective batch = micro_batch_size × accumulation_steps × n_GPUs

    LLaMA-3 used: 16M tokens/batch = 128K seqs × 128 tokens, accumulated over many GPUs.

---

## 21. Speculative Decoding — Faster Inference

Autoregressive generation is sequential: generate one token → feed it back → repeat.
This under-utilises the GPU which is built for parallel compute.

**Speculative Decoding (Chen et al., 2023; Leviathan et al., 2023):**

    1. A small "draft" model (e.g. LLaMA-3 1B) generates K candidate tokens in parallel
    2. The large "target" model (e.g. LLaMA-3 70B) verifies all K tokens in ONE parallel pass
    3. Accept tokens where the large model's probability ≥ draft model's probability
    4. Reject at the first mismatch, resample from the large model's distribution there

    Acceptance criterion (using rejection sampling):
        If P_large(xₜ) / P_draft(xₜ) ≥ 1: accept (target agrees or prefers this token)
        Otherwise: accept with probability P_large(xₜ) / P_draft(xₜ)

**Why it works:** The large model processes K tokens in parallel (as a prefill),
which is much faster than K sequential decode steps. If most tokens are accepted,
you get K tokens at the cost of ~1 large model forward pass.

    Typical speedup: 2–3× wall-clock latency reduction
    Output distribution: mathematically identical to sampling from the large model alone

**Models using speculative decoding:** Used in production by Google, Meta, and others.
Works best when the draft model is 5–10× smaller than the target.

---

## 22. Training Stability — Loss Spikes & Checkpoint Averaging

### Loss Spikes

Long LLM training runs regularly encounter sudden loss spikes — the gradient norm
explodes (sometimes 10–100×) due to:

    - A batch with unusual token distributions (very long sequences, rare tokens)
    - Numerical instability in BF16 operations at a particularly sensitive parameter
    - Data contamination (HTML tags, garbled text hitting an edge case)

**Mitigation strategies:**

    1. Gradient clipping (clip_norm=1.0): catches most spikes early
    2. Loss spike detection: if loss > 5× running average, discard batch and skip
    3. Checkpoint rollback: revert to last checkpoint N steps before the spike
    4. Data filtering: re-inspect and remove the offending data batch, resume training
    5. Smaller learning rate in affected region

LLaMA-3's training paper describes several spike incidents across the 15T token run,
each handled by rolling back ~100–200 steps.

### Checkpoint Averaging (SOUP / SLERP)

**Weight Averaging (Model Soup):** Average parameters of multiple checkpoints
from the same training run. Even simple linear averaging (Wortsman et al., 2022)
consistently improves performance by 1–2%, especially on out-of-distribution benchmarks.

**SLERP (Spherical Linear Interpolation):** Used for merging fine-tuned model variants.
Interpolates along the geodesic path on the weight sphere:

    SLERP(θ₀, θ₁, t) = sin((1-t)Ω)/sin(Ω) · θ₀ + sin(tΩ)/sin(Ω) · θ₁
    where Ω = arccos(θ₀ · θ₁ / (‖θ₀‖ ‖θ₁‖))

Used for merging instruct-tuned variants, multilingual models, and capability merges.

---

## 23. Data Preparation, Mixing & Curriculum

### Data Quality Filtering

Raw internet text is noisy. Production pipelines apply multiple filters:

    1. Language identification (fastText): keep target language(s)
    2. URL blocklist: remove adult content, spam, malware domains
    3. Heuristic filters: remove documents with too-short lines, high symbol ratio,
       excessive repetition, or very low token/character ratio
    4. Perplexity filter: use a smaller LM to score documents; discard extremes
       (garbled text scores very high; templated text scores very low)
    5. Exact deduplication: SHA256 hash of n-grams; remove exact duplicates
    6. MinHash near-deduplication: Jaccard similarity; remove near-duplicates
    7. PII removal: regex-based filtering of phone numbers, SSNs, email addresses

LLaMA-3 applied all of the above to 15T tokens, reducing from ~30T raw to 15T clean.

### Data Mixing

Different data sources are mixed at specific ratios throughout training:

    LLaMA-3 8B approximate mix:
    ┌────────────────┬─────────────┬───────────────────────────────┐
    │ Source         │ Proportion  │ Rationale                     │
    ├────────────────┼─────────────┼───────────────────────────────┤
    │ Web (filtered) │ ~80%        │ General knowledge, fluency    │
    │ Code           │ ~8%         │ Reasoning, structured output  │
    │ Math/Science   │ ~3%         │ Mathematical reasoning        │
    │ Books          │ ~5%         │ Long-form coherence           │
    │ Multilingual   │ ~4%         │ Language coverage             │
    └────────────────┴─────────────┴───────────────────────────────┘

### Curriculum / Annealing

Training is not uniform from start to finish:

    1. Early training: high learning rate, broad data mixture
    2. Mid training: stable mixture, cosine decay begins
    3. Late annealing phase: drastically reduce LR (10–100×), switch to
       highest-quality data only (curated books, math problems, code)
       This is where the model "consolidates" knowledge

LLaMA-3 used a final annealing phase on 40M high-quality tokens — just 0.3% of
total training data — which significantly boosted benchmark performance.

---

## Key Takeaways

- **Self-attention** is O(n²) in sequence length but captures all-pair dependencies in one step.
- **Causal masking** enables parallel training of autoregressive models — no sequential steps.
- **Multi-head attention** lets different heads learn different types of relationships simultaneously.
- **FFN layers** store factual knowledge; **attention layers** route and aggregate information.
- **Pre-Norm + residuals** enable stable training of very deep networks (100+ layers).
- **GQA/MQA** reduce KV-cache memory — critical for long-context inference.
- **Flash Attention** eliminates the O(n²) memory bottleneck via SRAM tiling; 2–4× faster than standard attention.
- **KV Cache** makes autoregressive inference efficient by storing and reusing K, V tensors.
- **Mixed Precision (BF16) + Gradient Checkpointing** are essential for fitting large models into GPU memory.
- **MoE** decouples parameter count from FLOPs — 46B parameter model with 13B active params per token.
- **Weight tying** saves ~13% parameters in large models by sharing embedding and LM head matrices.
- **Pretraining → SFT → RLHF/DPO** is the standard three-stage pipeline for instruction-tuned LLMs.
- **Chinchilla laws** say: smaller model + more data beats larger undertrained model.
- **Scaling laws** are power laws — loss improves predictably and smoothly with scale.
- **3D Parallelism (TP + PP + DP)** is required to train 70B+ models: GPT-3 used 65,536 A100s.
- **ZeRO / FSDP** shards optimizer state, gradients and parameters to eliminate redundancy across GPUs.
- **Speculative decoding** achieves 2–3× inference speedup with mathematically identical output.
- **Data mixing, curriculum, and annealing** are as important as architecture for final model quality.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ──────────────────────────────────────────────────────────────────
    "1 · Scaled Dot-Product Self-Attention": {
        "description": (
            "Build scaled dot-product attention from scratch using only NumPy. "
            "Traces every matrix multiplication step by step, prints shapes at each stage, "
            "visualises the attention weight matrix, and demonstrates how the causal mask "
            "enforces left-to-right information flow in decoder models."
        ),
        "language": "python",
        "code": """
            import numpy as np

            np.random.seed(42)

            # ── Setup ────────────────────────────────────────────────────────────
            seq_len = 5
            d_model = 8
            d_k     = 8    # key/query dimension

            TOKENS = ["The", "cat", "sat", "on", "mat"]

            # Simulated input embeddings  (seq_len, d_model)
            X = np.random.randn(seq_len, d_model).astype(np.float32)

            # Learned projection matrices
            Wq = np.random.randn(d_model, d_k).astype(np.float32) * 0.3
            Wk = np.random.randn(d_model, d_k).astype(np.float32) * 0.3
            Wv = np.random.randn(d_model, d_k).astype(np.float32) * 0.3

            def softmax(z, axis=-1):
                z = z - z.max(axis=axis, keepdims=True)   # numerical stability
                e = np.exp(z)
                return e / e.sum(axis=axis, keepdims=True)

            def scaled_dot_product_attention(X, Wq, Wk, Wv, mask=None):
                Q   = X @ Wq                    # (seq_len, d_k)
                K   = X @ Wk                    # (seq_len, d_k)
                V   = X @ Wv                    # (seq_len, d_k)

                scores = Q @ K.T                # (seq_len, seq_len)  — raw dot products
                scores = scores / np.sqrt(d_k)  # scale

                if mask is not None:
                    scores = scores + mask      # add -inf where masked

                weights = softmax(scores)       # (seq_len, seq_len)  — attention weights
                out     = weights @ V           # (seq_len, d_k)      — context vectors
                return Q, K, V, scores, weights, out

            # ── 1. Full (Bidirectional) Attention ────────────────────────────────
            print("=" * 60)
            print("  FULL (BIDIRECTIONAL) ATTENTION — encoder style")
            print("=" * 60)

            Q, K, V, scores_full, weights_full, out_full = scaled_dot_product_attention(X, Wq, Wk, Wv)

            print(f"  Input X:        {X.shape}")
            print(f"  Q = X @ Wq:     {Q.shape}")
            print(f"  K = X @ Wk:     {K.shape}")
            print(f"  V = X @ Wv:     {V.shape}")
            print(f"  Scores Q@Kᵀ:   {scores_full.shape}")
            print(f"  Weights softmax:{weights_full.shape}")
            print(f"  Output:         {out_full.shape}")
            print()

            print("  Attention Weight Matrix  A[i,j] = how much token i attends to token j")
            print(f"  {'':12s}", end="")
            for t in TOKENS:
                print(f"  {t:6s}", end="")
            print()
            print("  " + "─" * 52)
            for i, row_tok in enumerate(TOKENS):
                print(f"  {row_tok:10s} │", end="")
                for j in range(seq_len):
                    val = weights_full[i, j]
                    bar = "▓" if val > 0.25 else ("░" if val > 0.15 else "·")
                    print(f"  {val:.3f}{bar}", end="")
                print()
            print()
            print("  Row sums (must be 1.0):", np.round(weights_full.sum(axis=1), 4))
            print()

            # ── 2. Causal (Masked) Attention — decoder style ──────────────────
            print("=" * 60)
            print("  CAUSAL (MASKED) ATTENTION — decoder style")
            print("=" * 60)

            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            print("  Causal Mask (upper triangle = -1e9 → softmax → 0):")
            print(f"  {'':12s}", end="")
            for t in TOKENS:
                print(f"  {t:6s}", end="")
            print()
            print("  " + "─" * 52)

            _, _, _, scores_causal, weights_causal, out_causal = scaled_dot_product_attention(
                X, Wq, Wk, Wv, mask=causal_mask
            )

            for i, row_tok in enumerate(TOKENS):
                print(f"  {row_tok:10s} │", end="")
                for j in range(seq_len):
                    val = weights_causal[i, j]
                    if j > i:
                        print(f"  {'✗':6s}", end="")   # masked
                    else:
                        print(f"  {val:.3f} ", end="")
                print()
            print()

            # ── 3. Attention Entropy — how focused vs diffuse ──────────────────
            print("── Attention Entropy (bits) — lower = more focused ──")
            print()
            for i, tok in enumerate(TOKENS):
                w      = weights_causal[i, :i+1]            # only valid positions
                w      = w / w.sum()                        # renormalise
                ent    = -np.sum(w * np.log2(w + 1e-12))
                bar    = "█" * int(ent * 8)
                print(f"  {tok:6s}  entropy={ent:.3f} bits  {bar}")
            print()
            print("  Position 0 has entropy=0 (only sees itself)")
            print("  Later positions have higher entropy — more context to attend to")
        """,
    },

    # ── 2 ──────────────────────────────────────────────────────────────────
    "2 · Multi-Head Attention (MHA)": {
        "description": (
            "Implement full Multi-Head Attention with h parallel heads. "
            "Shows how each head projects into a lower-dimensional subspace, "
            "runs attention independently, and how outputs are concatenated and "
            "projected. Compares what different heads focus on and counts all parameters."
        ),
        "language": "python",
        "code": """
            import numpy as np

            np.random.seed(7)

            # ── Config ───────────────────────────────────────────────────────────
            seq_len = 6
            d_model = 16
            n_heads = 4
            d_k     = d_model // n_heads    # = 4 per head

            TOKENS = ["The", "quick", "brown", "fox", "jumps", "over"]

            X = np.random.randn(seq_len, d_model).astype(np.float32)

            def softmax(z, axis=-1):
                z = z - z.max(axis=axis, keepdims=True)
                return np.exp(z) / np.exp(z).sum(axis=axis, keepdims=True)

            class MultiHeadAttention:
                def __init__(self, d_model, n_heads, seed=42):
                    rng = np.random.RandomState(seed)
                    scale = 0.1
                    self.d_model = d_model
                    self.n_heads = n_heads
                    self.d_k     = d_model // n_heads

                    # Each head has its own Q, K, V projections
                    self.Wq = [rng.randn(d_model, self.d_k).astype(np.float32)*scale for _ in range(n_heads)]
                    self.Wk = [rng.randn(d_model, self.d_k).astype(np.float32)*scale for _ in range(n_heads)]
                    self.Wv = [rng.randn(d_model, self.d_k).astype(np.float32)*scale for _ in range(n_heads)]
                    # Output projection
                    self.Wo = rng.randn(d_model, d_model).astype(np.float32) * scale

                def forward(self, X, mask=None):
                    head_outputs = []
                    head_weights = []

                    for h in range(self.n_heads):
                        Q = X @ self.Wq[h]              # (seq_len, d_k)
                        K = X @ self.Wk[h]              # (seq_len, d_k)
                        V = X @ self.Wv[h]              # (seq_len, d_k)

                        scores  = Q @ K.T / np.sqrt(self.d_k)
                        if mask is not None:
                            scores = scores + mask
                        weights = softmax(scores)        # (seq_len, seq_len)
                        out     = weights @ V            # (seq_len, d_k)

                        head_outputs.append(out)
                        head_weights.append(weights)

                    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
                    output = concat @ self.Wo                         # (seq_len, d_model)
                    return output, head_weights

                def param_count(self):
                    q = self.n_heads * self.d_model * self.d_k
                    k = self.n_heads * self.d_model * self.d_k
                    v = self.n_heads * self.d_model * self.d_k
                    o = self.d_model * self.d_model
                    return {"Q projections": q, "K projections": k,
                            "V projections": v, "O projection": o,
                            "Total": q + k + v + o}

            mha = MultiHeadAttention(d_model, n_heads)
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            output, head_weights = mha.forward(X, mask=causal_mask)

            print("=" * 65)
            print("  Multi-Head Attention — Forward Pass")
            print("=" * 65)
            print(f"  d_model={d_model},  n_heads={n_heads},  d_k={d_k} per head")
            print(f"  Input:   {X.shape}   →   Output: {output.shape}")
            print()

            # ── Parameter Count ──────────────────────────────────────────────────
            params = mha.param_count()
            print("── Parameter Count ──")
            for name, count in params.items():
                bar = "▪" * (count // (d_model * d_k))
                print(f"  {name:20s}: {count:6d}  {bar}")
            print()
            print(f"  Analytic check: 4 × d_model² = 4 × {d_model}² = {4*d_model**2}")
            print()

            # ── Per-Head Attention Patterns ──────────────────────────────────────
            print("── Per-Head Attention Weights (causal) ──")
            print()
            for h_idx, weights in enumerate(head_weights):
                print(f"  Head {h_idx+1}:")
                print(f"  {'':10s}", end="")
                for t in TOKENS:
                    print(f"  {t[:5]:5s}", end="")
                print()
                for i, tok in enumerate(TOKENS):
                    print(f"  {tok:10s}", end="")
                    for j in range(seq_len):
                        if j > i:
                            print(f"  {'·····':5s}", end="")
                        else:
                            val = weights[i, j]
                            bar = "█" if val > 0.4 else ("▓" if val > 0.25 else ("░" if val > 0.1 else "·"))
                            print(f"  {val:.3f}", end="")
                    print()
                # Dominant focus for each head
                dominant = []
                for i in range(seq_len):
                    valid = weights[i, :i+1]
                    if valid.sum() > 0:
                        focus = valid.argmax()
                        dominant.append(f"{TOKENS[i]}→{TOKENS[focus]}")
                print(f"  Dominant focus: {', '.join(dominant)}")
                print()

            # ── Concat + Output Projection ────────────────────────────────────────
            print("── Shape Flow Through MHA ──")
            print(f"  Input X:                    {X.shape}")
            print(f"  Per-head Q,K,V:             ({seq_len}, {d_k})  each")
            print(f"  Per-head output:            ({seq_len}, {d_k})  × {n_heads} heads")
            print(f"  After concat:               ({seq_len}, {n_heads}×{d_k}) = ({seq_len}, {d_model})")
            print(f"  After Wᴼ projection:        ({seq_len}, {d_model})")
            print(f"  Same shape as input ✓ — blocks can be stacked")
        """,
    },

    # ── 3 ──────────────────────────────────────────────────────────────────
    "3 · Feed-Forward Network & Activation Functions": {
        "description": (
            "Implement the Transformer FFN sublayer with ReLU, GELU, and SwiGLU activations. "
            "Plots the activation function shapes, shows the dimension expansion/contraction, "
            "and benchmarks how much of total model parameters live in FFN vs attention layers."
        ),
        "language": "python",
        "code": """
            import numpy as np

            np.random.seed(0)

            # ── Activation Functions ──────────────────────────────────────────────
            def relu(x):
                return np.maximum(0, x)

            def gelu(x):
                \"\"\"Gaussian Error Linear Unit — approximate formula used by BERT/GPT-2.\"\"\"
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

            def swish(x, beta=1.0):
                return x * (1 / (1 + np.exp(-beta * x)))

            def swiglu(x, W1, W3, W2):
                \"\"\"
                SwiGLU: SwiGLU(x, W1, W3) @ W2
                  gate   = swish(x @ W1)
                  value  = x @ W3
                  output = (gate * value) @ W2
                Three weight matrices instead of two.
                Used by LLaMA, PaLM, Gemma.
                \"\"\"
                gate   = swish(x @ W1)
                value  = x @ W3
                return (gate * value) @ W2

            # ── Print Activation Shapes (ASCII plot) ─────────────────────────────
            x_vals = np.linspace(-3, 3, 61)

            def ascii_plot(func, func_name, x_range=(-3, 3), y_range=(-0.5, 1.5), width=60, height=10):
                x_vals = np.linspace(*x_range, width)
                y_vals = func(x_vals)
                y_min, y_max = y_range
                print(f"  {func_name}:")
                grid = [[" "] * width for _ in range(height)]
                for col, (x, y) in enumerate(zip(x_vals, y_vals)):
                    row = int((y_max - y) / (y_max - y_min) * (height - 1))
                    row = max(0, min(height - 1, row))
                    grid[row][col] = "█"
                # Zero line
                zero_row = int((y_max - 0) / (y_max - y_min) * (height - 1))
                zero_row = max(0, min(height - 1, zero_row))
                for col in range(width):
                    if grid[zero_row][col] == " ":
                        grid[zero_row][col] = "─"
                print(f"  y={y_max:.1f} ┤")
                for row in grid:
                    print(f"        │ {''.join(row)}")
                print(f"  y={y_min:.1f} ┤")
                print(f"          {'x='+str(x_range[0]):^10s}{'x=0':^40s}{'x='+str(x_range[1]):>8s}")
                print()

            print("=" * 62)
            print("  Activation Functions — Visual Comparison")
            print("=" * 62)
            print()
            ascii_plot(relu,             "ReLU  — max(0,x)      — original Transformer")
            ascii_plot(gelu,             "GELU  — smooth ReLU   — BERT, GPT-2, GPT-3")
            ascii_plot(swish,            "Swish — x·σ(x)        — component of SwiGLU")

            # ── FFN Implementations ───────────────────────────────────────────────
            print("=" * 62)
            print("  FFN Sublayer — Forward Pass")
            print("=" * 62)

            seq_len = 4
            d_model = 16
            d_ff    = d_model * 4   # standard expansion factor

            X = np.random.randn(seq_len, d_model).astype(np.float32)

            class FFN_ReLU:
                def __init__(self, d_model, d_ff):
                    scale = np.sqrt(2.0 / d_model)
                    self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
                    self.b1 = np.zeros(d_ff, dtype=np.float32)
                    self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale
                    self.b2 = np.zeros(d_model, dtype=np.float32)

                def forward(self, x):
                    h = relu(x @ self.W1 + self.b1)    # (seq_len, d_ff)
                    return h @ self.W2 + self.b2         # (seq_len, d_model)

                def param_count(self):
                    return (self.W1.size + self.b1.size +
                            self.W2.size + self.b2.size)

            class FFN_SwiGLU:
                \"\"\"Three matrices: W1 (gate linear), W3 (value linear), W2 (output).\"\"\"
                def __init__(self, d_model, d_ff):
                    scale = np.sqrt(2.0 / d_model)
                    self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
                    self.W3 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
                    self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale

                def forward(self, x):
                    gate  = swish(x @ self.W1)              # gating
                    value = x @ self.W3                      # value path
                    return (gate * value) @ self.W2

                def param_count(self):
                    return self.W1.size + self.W3.size + self.W2.size

            ffn_relu   = FFN_ReLU(d_model, d_ff)
            ffn_swiglu = FFN_SwiGLU(d_model, d_ff)

            out_relu   = ffn_relu.forward(X)
            out_swiglu = ffn_swiglu.forward(X)

            print(f"  Config: d_model={d_model},  d_ff={d_ff} (4× expansion)")
            print()
            print(f"  FFN ReLU:")
            print(f"    Input:        {X.shape}  (seq_len, d_model)")
            print(f"    After W1:     ({seq_len}, {d_ff}) — expansion")
            print(f"    After ReLU:   ({seq_len}, {d_ff}) — activation (zeros out negatives)")
            print(f"    After W2:     {out_relu.shape}  — contraction back to d_model")
            print(f"    Parameters:   {ffn_relu.param_count():,}")
            print()
            print(f"  FFN SwiGLU:")
            print(f"    Input → gate path (W1 + Swish): ({seq_len}, {d_ff})")
            print(f"    Input → value path (W3):        ({seq_len}, {d_ff})")
            print(f"    gate ⊙ value → W2:               {out_swiglu.shape}")
            print(f"    Parameters:   {ffn_swiglu.param_count():,}  (3 matrices vs 2)")
            print()

            # ── Parameter Budget Breakdown ────────────────────────────────────────
            print("=" * 62)
            print("  Parameter Budget — Attention vs FFN at Scale")
            print("=" * 62)
            configs = [
                ("GPT-2 Base",   12,   768, 12, 50257),
                ("GPT-2 Large",  36,  1280, 20, 50257),
                ("LLaMA-3 8B",   32,  4096, 32, 128256),
                ("GPT-3 175B",   96, 12288, 96, 50257),
            ]
            print(f"  {'Model':16s} {'N':>4s} {'d':>6s} {'Attn':>12s} {'FFN':>12s} {'Embed':>10s} {'Total':>12s} {'FFN%':>6s}")
            print("  " + "─" * 78)
            for name, N, d, h, V in configs:
                attn  = N * 4 * d * d
                ffn   = N * 8 * d * d
                embed = 2 * V * d
                total = attn + ffn + embed
                print(f"  {name:16s} {N:>4d} {d:>6d} {attn/1e6:>10.1f}M {ffn/1e6:>10.1f}M "
                      f"{embed/1e6:>8.1f}M {total/1e6:>10.1f}M {ffn/total*100:>5.1f}%")
            print()
            print("  FFN layers hold ~57% of parameters in most models.")
            print("  Despite this, attention layers receive more research focus.")
        """,
    },

    # ── 4 ──────────────────────────────────────────────────────────────────
    "4 · Full Transformer Block (Pre-Norm, Residuals, RMSNorm)": {
        "description": (
            "Assemble a complete Pre-Norm Transformer decoder block from scratch: "
            "RMSNorm → MHA (causal) → residual → RMSNorm → FFN → residual. "
            "Stacks multiple blocks, tracks hidden state evolution across layers, "
            "and verifies that residual connections preserve gradient signal."
        ),
        "language": "python",
        "code": """
            import numpy as np

            np.random.seed(42)

            # ── Helpers ──────────────────────────────────────────────────────────
            def softmax(z, axis=-1):
                z = z - z.max(axis=axis, keepdims=True)
                return np.exp(z) / np.exp(z).sum(axis=axis, keepdims=True)

            def gelu(x):
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

            # ── RMSNorm ──────────────────────────────────────────────────────────
            class RMSNorm:
                \"\"\"
                RMSNorm: x / RMS(x) * gamma
                Used by LLaMA, Mistral, Gemma — 15% faster than LayerNorm.
                \"\"\"
                def __init__(self, d_model, eps=1e-8):
                    self.gamma = np.ones(d_model, dtype=np.float32)   # learned scale
                    self.eps   = eps

                def forward(self, x):
                    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + self.eps)
                    return self.gamma * (x / rms)

            # ── Multi-Head Attention ─────────────────────────────────────────────
            class MHA:
                def __init__(self, d_model, n_heads):
                    self.h  = n_heads
                    self.dk = d_model // n_heads
                    s = 0.1
                    self.Wq = np.random.randn(d_model, d_model).astype(np.float32) * s
                    self.Wk = np.random.randn(d_model, d_model).astype(np.float32) * s
                    self.Wv = np.random.randn(d_model, d_model).astype(np.float32) * s
                    self.Wo = np.random.randn(d_model, d_model).astype(np.float32) * s

                def forward(self, x, mask=None):
                    B, T, D = x.shape[0], x.shape[1] if len(x.shape)==3 else 1, x.shape[-1]
                    if len(x.shape) == 2:
                        x = x[np.newaxis, ...]   # add batch dim
                    T = x.shape[1]

                    Q = x @ self.Wq          # (1, T, D)
                    K = x @ self.Wk
                    V = x @ self.Wv

                    # Split heads: (1, T, D) → (1, T, h, dk) → (1, h, T, dk)
                    def split(t):
                        return t.reshape(1, T, self.h, self.dk).transpose(0,2,1,3)

                    Q, K, V = split(Q), split(K), split(V)   # (1, h, T, dk)

                    scores = Q @ K.transpose(0,1,3,2) / np.sqrt(self.dk)  # (1,h,T,T)
                    if mask is not None:
                        scores = scores + mask[np.newaxis, np.newaxis, :, :]

                    weights = softmax(scores)                              # (1,h,T,T)
                    ctx     = weights @ V                                  # (1,h,T,dk)

                    # Merge heads: (1,h,T,dk) → (1,T,D)
                    ctx = ctx.transpose(0,2,1,3).reshape(1, T, D)
                    out = ctx @ self.Wo
                    return out.squeeze(0)   # (T, D)

            # ── FFN ─────────────────────────────────────────────────────────────
            class FFN:
                def __init__(self, d_model, d_ff):
                    s = 0.1
                    self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * s
                    self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * s

                def forward(self, x):
                    return gelu(x @ self.W1) @ self.W2

            # ── Transformer Block (Pre-Norm) ─────────────────────────────────────
            class TransformerBlock:
                \"\"\"
                Pre-Norm Transformer decoder block:
                  x = x + MHA(RMSNorm(x))
                  x = x + FFN(RMSNorm(x))
                \"\"\"
                def __init__(self, d_model, n_heads, d_ff, layer_idx=0):
                    self.norm1 = RMSNorm(d_model)
                    self.attn  = MHA(d_model, n_heads)
                    self.norm2 = RMSNorm(d_model)
                    self.ffn   = FFN(d_model, d_ff)
                    self.layer_idx = layer_idx

                def forward(self, x, mask=None):
                    # ── Attention sublayer ──────────────────────────────────────
                    x_normed  = self.norm1.forward(x)
                    attn_out  = self.attn.forward(x_normed, mask=mask)
                    x         = x + attn_out            # residual connection

                    # ── FFN sublayer ────────────────────────────────────────────
                    x_normed  = self.norm2.forward(x)
                    ffn_out   = self.ffn.forward(x_normed)
                    x         = x + ffn_out              # residual connection
                    return x

            # ── Config ──────────────────────────────────────────────────────────
            d_model = 16
            n_heads = 4
            d_ff    = d_model * 4
            n_layers = 3
            seq_len  = 5
            TOKENS   = ["<BOS>", "The", "cat", "sat", "on"]

            X = np.random.randn(seq_len, d_model).astype(np.float32)
            causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

            # Build N stacked blocks
            blocks = [TransformerBlock(d_model, n_heads, d_ff, i) for i in range(n_layers)]

            # ── Forward Pass Through All Layers ──────────────────────────────────
            print("=" * 62)
            print("  Transformer Decoder Stack — Forward Pass")
            print("=" * 62)
            print(f"  Config: d_model={d_model}, n_heads={n_heads}, "
                  f"d_ff={d_ff}, n_layers={n_layers}")
            print(f"  Sequence: {TOKENS}")
            print()

            h = X.copy()
            stats = []
            stats.append(("Input X", h.mean(), h.std(), np.linalg.norm(h)))

            for idx, block in enumerate(blocks):
                h = block.forward(h, mask=causal_mask)
                stats.append((f"After Block {idx+1}", h.mean(), h.std(), np.linalg.norm(h)))

            print(f"  {'Stage':20s}  {'Mean':>10s}  {'Std':>10s}  {'L2 Norm':>10s}")
            print("  " + "─" * 56)
            for label, mean, std, norm in stats:
                print(f"  {label:20s}  {mean:>10.5f}  {std:>10.5f}  {norm:>10.4f}")
            print()

            # ── Residual Connection Demo ─────────────────────────────────────────
            print("── Residual Connections — Why They Matter ──")
            print()
            print("  Without residuals: x = Sublayer(x)")
            print("  With residuals:    x = x + Sublayer(x)    ← gradient highway")
            print()

            d_test = 8
            x_test = np.random.randn(d_test).astype(np.float32)
            W_test = np.random.randn(d_test, d_test).astype(np.float32) * 0.1
            n_iter = 20

            def forward_no_residual(x, W, n):
                for _ in range(n):
                    x = np.tanh(x @ W)
                return x

            def forward_with_residual(x, W, n):
                for _ in range(n):
                    x = x + np.tanh(x @ W)
                return x

            out_no  = forward_no_residual(x_test, W_test, n_iter)
            out_res = forward_with_residual(x_test, W_test, n_iter)

            print(f"  After {n_iter} layers, L2 norm of hidden state:")
            print(f"    No residuals:   {np.linalg.norm(out_no):.6f}  (vanishes or saturates)")
            print(f"    With residuals: {np.linalg.norm(out_res):.6f}  (preserved)")
            print()

            # ── Shape Summary ────────────────────────────────────────────────────
            print("── Complete Shape Flow (one block) ──")
            print()
            print(f"  Input x:                   ({seq_len}, {d_model})")
            print(f"  RMSNorm(x):                ({seq_len}, {d_model})")
            print(f"  MHA output:                ({seq_len}, {d_model})")
            print(f"  x = x + MHA(norm(x)):      ({seq_len}, {d_model})  ← same shape, residual")
            print(f"  RMSNorm(x):                ({seq_len}, {d_model})")
            print(f"  FFN hidden:                ({seq_len}, {d_ff})")
            print(f"  FFN output:                ({seq_len}, {d_model})")
            print(f"  x = x + FFN(norm(x)):      ({seq_len}, {d_model})  ← same shape, residual")
            print()
            print("  Every block is input-output shape preserving → infinitely stackable.")
        """,
    },

    # ── 5 ──────────────────────────────────────────────────────────────────
    "5 · Pre-Training Objectives (CLM vs MLM vs Span Corruption)": {
        "description": (
            "Implement and compare all three major pre-training objectives: "
            "Causal Language Modeling (GPT-style), Masked Language Modeling (BERT-style), "
            "and Span Corruption (T5-style). Computes loss for each objective on the same "
            "input sequence and explains what the model learns from each signal."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math
            from collections import defaultdict

            np.random.seed(0)

            # ── Shared Setup ─────────────────────────────────────────────────────
            VOCAB  = ["<PAD>","<BOS>","<EOS>","<MASK>","<SEP>",
                      "The","cat","sat","on","the","mat",".",
                      "A","dog","ran","quickly","over","fence"]
            V   = len(VOCAB)
            w2i = {w: i for i, w in enumerate(VOCAB)}
            i2w = {i: w for w, i in w2i.items()}

            SENTENCE = ["<BOS>","The","cat","sat","on","the","mat",".","<EOS>"]
            ids      = [w2i[w] for w in SENTENCE]
            T        = len(ids)

            def softmax(z):
                z = z - z.max()
                return np.exp(z) / np.exp(z).sum()

            def cross_entropy(logits, target_id):
                probs = softmax(logits)
                return -math.log(probs[target_id] + 1e-12), probs[target_id]

            # Simulated model logits: slightly biased toward true tokens
            def sim_logits(true_id, strength=4.0):
                lg = np.random.randn(V)
                lg[true_id] += strength
                return lg

            # ══════════════════════════════════════════════════════════════════════
            # OBJECTIVE 1 — Causal Language Modeling (CLM)  ← GPT, LLaMA, Claude
            # ══════════════════════════════════════════════════════════════════════
            print("=" * 65)
            print("  OBJECTIVE 1: Causal LM (CLM) — GPT / LLaMA style")
            print("=" * 65)
            print()
            print("  Rule: predict token at position t using ONLY positions < t")
            print("  Loss computed at every position in the sequence")
            print()
            print(f"  {'pos':>4s}  {'input →':10s}  {'target':10s}  {'P(target)':>10s}  {'NLL':>8s}")
            print("  " + "─" * 55)

            clm_losses = []
            for t in range(1, T):                         # predict positions 1..T-1
                input_tok  = SENTENCE[t-1]
                target_tok = SENTENCE[t]
                target_id  = ids[t]

                logits     = sim_logits(target_id, strength=5.0)
                loss, prob = cross_entropy(logits, target_id)
                clm_losses.append(loss)
                print(f"  {t:>4d}  {input_tok:12s}→  {target_tok:12s}  {prob:>10.4f}  {loss:>8.4f}")

            clm_mean = np.mean(clm_losses)
            clm_ppl  = math.exp(clm_mean)
            print(f"\\n  CLM Mean NLL: {clm_mean:.4f}   Perplexity: {clm_ppl:.2f}")
            print(f"  Tokens contributing to loss: {T-1}/{T}  (all except BOS input)")

            # ══════════════════════════════════════════════════════════════════════
            # OBJECTIVE 2 — Masked Language Modeling (MLM) ← BERT, RoBERTa
            # ══════════════════════════════════════════════════════════════════════
            print()
            print("=" * 65)
            print("  OBJECTIVE 2: Masked LM (MLM) — BERT / RoBERTa style")
            print("=" * 65)
            print()
            print("  Rule: randomly mask 15% of tokens, predict from FULL (bidirectional) context")
            print("  The 15% split: 80% → [MASK], 10% → random token, 10% → unchanged")
            print()

            mask_rate  = 0.15
            np.random.seed(42)

            # Decide which positions to mask (skip BOS, EOS)
            eligible = list(range(1, T-1))
            n_mask   = max(1, int(len(eligible) * mask_rate))
            masked_positions = sorted(np.random.choice(eligible, size=n_mask, replace=False))

            masked_sentence = list(SENTENCE)
            replacements    = {}
            for pos in masked_positions:
                r = np.random.rand()
                original = SENTENCE[pos]
                if r < 0.80:
                    masked_sentence[pos] = "<MASK>"
                    replacements[pos]    = ("→ [MASK]",  original)
                elif r < 0.90:
                    rand_tok = np.random.choice(list(w2i.keys()))
                    masked_sentence[pos] = rand_tok
                    replacements[pos]    = (f"→ {rand_tok}(rand)", original)
                else:
                    replacements[pos]    = ("→ unchanged", original)

            print(f"  Original:  {' '.join(SENTENCE)}")
            print(f"  Masked:    {' '.join(masked_sentence)}")
            print()
            print(f"  {'pos':>4s}  {'replacement':20s}  {'target':10s}  {'P(target)':>10s}  {'NLL':>8s}")
            print("  " + "─" * 60)

            mlm_losses = []
            for pos in masked_positions:
                repl, orig = replacements[pos]
                target_id  = w2i[orig]
                logits     = sim_logits(target_id, strength=6.0)
                loss, prob = cross_entropy(logits, target_id)
                mlm_losses.append(loss)
                print(f"  {pos:>4d}  {repl:20s}  {orig:10s}  {prob:>10.4f}  {loss:>8.4f}")

            mlm_mean = np.mean(mlm_losses) if mlm_losses else 0
            mlm_ppl  = math.exp(mlm_mean)
            print(f"\\n  MLM Mean NLL: {mlm_mean:.4f}   Perplexity: {mlm_ppl:.2f}")
            print(f"  Tokens contributing to loss: {len(masked_positions)}/{T} (~15%)")
            print()
            print("  ⚠  MLM sees fewer loss signals per sequence than CLM.")
            print("     But each signal uses FULL bidirectional context → richer.")

            # ══════════════════════════════════════════════════════════════════════
            # OBJECTIVE 3 — Span Corruption (T5 style)
            # ══════════════════════════════════════════════════════════════════════
            print()
            print("=" * 65)
            print("  OBJECTIVE 3: Span Corruption — T5 style")
            print("=" * 65)
            print()
            print("  Rule: mask contiguous SPANS (not individual tokens).")
            print("  Encoder sees corrupted input; decoder predicts masked spans.")
            print("  Each span replaced by a single sentinel token <extra_id_N>.")
            print()

            # Span corruption on inner tokens
            inner = SENTENCE[1:-1]   # skip BOS/EOS for clarity
            span_start = 2           # "sat"
            span_end   = 4           # "the" (exclusive: positions 2,3)
            span_len   = span_end - span_start

            corrupted = inner[:span_start] + ["<extra_id_0>"] + inner[span_end:]
            target    = ["<extra_id_0>"] + inner[span_start:span_end] + ["<extra_id_1>"]

            print(f"  Original tokens (inner):  {' '.join(inner)}")
            print(f"  Corrupted encoder input:  {' '.join(corrupted)}")
            print(f"  Decoder target (to predict): {' '.join(target)}")
            print()

            span_tokens = inner[span_start:span_end]
            print(f"  {'pos':>4s}  {'target token':15s}  {'P(token)':>10s}  {'NLL':>8s}")
            print("  " + "─" * 44)

            span_losses = []
            for tok in target:
                if tok in w2i:
                    tid = w2i[tok]
                else:
                    tid = 0
                logits = sim_logits(tid, strength=5.5)
                loss, prob = cross_entropy(logits, tid)
                span_losses.append(loss)
                print(f"  {'':>4s}  {tok:15s}  {prob:>10.4f}  {loss:>8.4f}")

            span_mean = np.mean(span_losses)
            print(f"\\n  Span Mean NLL: {span_mean:.4f}")
            print(f"  Input length reduction: {len(inner)} → {len(corrupted)} tokens (encoder)")
            print(f"  Predicted span length: {len(target)} tokens (decoder)")

            # ── Side-by-Side Comparison ──────────────────────────────────────────
            print()
            print("=" * 65)
            print("  Side-by-Side Comparison")
            print("=" * 65)
            rows = [
                ("Objective",        "CLM",             "MLM",              "Span Corruption"),
                ("Model family",     "GPT/LLaMA/Claude","BERT/RoBERTa",     "T5/BART"),
                ("Attention type",   "Causal (decoder)","Bidirectional",    "Enc bidirec+Dec causal"),
                ("% tokens as loss", f"{(T-1)/T*100:.0f}%",
                                     f"{mask_rate*100:.0f}%",
                                     f"~15% spans"),
                ("Naturally generat.","Yes (sample)",   "No",               "Yes (enc→dec)"),
                ("Context",          "Left-only",       "Full bidirec.",    "Full (enc), causal (dec)"),
            ]
            col_w = [22, 18, 18, 24]
            for row in rows:
                line = "  "
                for i, cell in enumerate(row):
                    line += cell[:col_w[i]].ljust(col_w[i]) + " │ "
                print(line)
                if row[0] == "Objective":
                    print("  " + "─" * 84)
        """,
    },

    # ── 6 ──────────────────────────────────────────────────────────────────
    "6 · Training Loop — AdamW, LR Schedule & Gradient Clipping": {
        "description": (
            "Implement a complete mini Transformer training loop from scratch: "
            "AdamW optimiser with bias correction, cosine LR schedule with linear warmup, "
            "and gradient norm clipping. Trains a tiny 2-layer decoder to predict the next "
            "token on a small corpus and plots the loss curve."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            np.random.seed(42)

            # ─────────────────────────────────────────────────────────────────────
            # Tiny vocabulary and corpus
            # ─────────────────────────────────────────────────────────────────────
            VOCAB  = ["<PAD>","<BOS>","<EOS>","the","cat","sat","on","mat",
                      "dog","ran","a",".",  "and","over","quick"]
            V      = len(VOCAB)
            w2i    = {w: i for i, w in enumerate(VOCAB)}

            CORPUS_TEXT = [
                ["<BOS>","the","cat","sat","on","the","mat",".","<EOS>"],
                ["<BOS>","a","dog","ran","over","the","mat",".","<EOS>"],
                ["<BOS>","the","quick","cat","and","the","dog",".","<EOS>"],
                ["<BOS>","the","dog","sat","on","a","mat",".","<EOS>"],
            ]
            CORPUS = [[w2i[w] for w in seq] for seq in CORPUS_TEXT]

            # ─────────────────────────────────────────────────────────────────────
            # Model Parameters (tiny 2-layer decoder)
            # ─────────────────────────────────────────────────────────────────────
            d_model = 32
            d_ff    = 64
            n_heads = 4
            n_layers = 2
            d_k     = d_model // n_heads

            def init_params():
                \"\"\"Initialise all model parameters using Xavier / small random init.\"\"\"
                s = lambda r, c: np.random.randn(r, c).astype(np.float64) * np.sqrt(2.0/(r+c))
                params = {}
                params["embed"] = np.random.randn(V, d_model).astype(np.float64) * 0.02
                for l in range(n_layers):
                    params[f"Wq{l}"] = s(d_model, d_model)
                    params[f"Wk{l}"] = s(d_model, d_model)
                    params[f"Wv{l}"] = s(d_model, d_model)
                    params[f"Wo{l}"] = s(d_model, d_model)
                    params[f"W1{l}"] = s(d_model, d_ff)
                    params[f"W2{l}"] = s(d_ff,    d_model)
                    params[f"g1{l}"] = np.ones(d_model,  dtype=np.float64)   # RMSNorm scale
                    params[f"g2{l}"] = np.ones(d_model,  dtype=np.float64)
                params["lm_W"]  = s(d_model, V)
                return params

            def rmsnorm(x, g, eps=1e-8):
                rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
                return g * x / rms

            def softmax_2d(z):
                z = z - z.max(axis=-1, keepdims=True)
                e = np.exp(z)
                return e / e.sum(axis=-1, keepdims=True)

            def gelu(x):
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

            def forward(params, token_ids):
                \"\"\"
                Simplified forward pass — returns per-position logits.
                token_ids: list of int, length T
                Returns: logits of shape (T, V)
                \"\"\"
                T  = len(token_ids)
                x  = params["embed"][token_ids]        # (T, d_model)

                # Sinusoidal positional encoding (fixed)
                pos  = np.arange(T)[:, None]
                dims = np.arange(0, d_model, 2)
                pe   = np.zeros((T, d_model), dtype=np.float64)
                pe[:, 0::2] = np.sin(pos / (10000 ** (dims / d_model)))
                pe[:, 1::2] = np.cos(pos / (10000 ** (dims / d_model)))
                x = x + pe

                mask = np.triu(np.full((T, T), -1e9), k=1)

                for l in range(n_layers):
                    # ── Attention ────────────────────────────────────────────────
                    xn  = rmsnorm(x, params[f"g1{l}"])
                    Q   = xn @ params[f"Wq{l}"]
                    K   = xn @ params[f"Wk{l}"]
                    V_  = xn @ params[f"Wv{l}"]
                    sc  = Q @ K.T / np.sqrt(d_k) + mask
                    A   = softmax_2d(sc)
                    ctx = A @ V_
                    x   = x + ctx @ params[f"Wo{l}"]

                    # ── FFN ──────────────────────────────────────────────────────
                    xn  = rmsnorm(x, params[f"g2{l}"])
                    h   = gelu(xn @ params[f"W1{l}"])
                    x   = x + h @ params[f"W2{l}"]

                logits = x @ params["lm_W"]            # (T, V)
                return logits

            def compute_loss(params, token_ids):
                \"\"\"CLM loss: predict token t from context 0..t-1.\"\"\"
                logits = forward(params, token_ids[:-1])   # inputs: all but last
                targets = token_ids[1:]                     # targets: all but first
                T = len(targets)
                loss = 0.0
                for t in range(T):
                    z    = logits[t] - logits[t].max()
                    lse  = np.log(np.exp(z).sum())
                    loss += -z[targets[t]] + lse
                return loss / T

            def numerical_grad(params, key, idx, token_ids, eps=1e-5):
                \"\"\"Finite-difference gradient for a single parameter element.\"\"\"
                orig = params[key].flat[idx]
                params[key].flat[idx] = orig + eps
                loss_plus = compute_loss(params, token_ids)
                params[key].flat[idx] = orig - eps
                loss_minus = compute_loss(params, token_ids)
                params[key].flat[idx] = orig
                return (loss_plus - loss_minus) / (2 * eps)

            # ─────────────────────────────────────────────────────────────────────
            # AdamW Optimiser
            # ─────────────────────────────────────────────────────────────────────
            class AdamW:
                def __init__(self, params, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
                    self.lr  = lr
                    self.b1  = b1
                    self.b2  = b2
                    self.eps = eps
                    self.wd  = wd
                    self.t   = 0
                    self.m   = {k: np.zeros_like(v) for k, v in params.items()}
                    self.v   = {k: np.zeros_like(v) for k, v in params.items()}

                def step(self, params, grads):
                    self.t += 1
                    for k in params:
                        if k not in grads or grads[k] is None:
                            continue
                        g = grads[k]
                        self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
                        self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g**2
                        # Bias correction
                        m_hat = self.m[k] / (1 - self.b1**self.t)
                        v_hat = self.v[k] / (1 - self.b2**self.t)
                        # Weight decay (applied to params directly, not through gradient)
                        params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                        params[k] -= self.lr * self.wd * params[k]

            # ─────────────────────────────────────────────────────────────────────
            # Learning Rate Schedule: Warmup + Cosine Decay
            # ─────────────────────────────────────────────────────────────────────
            def lr_schedule(step, total_steps, peak_lr, warmup_steps, min_lr_ratio=0.1):
                if step < warmup_steps:
                    return peak_lr * (step + 1) / warmup_steps        # linear warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine   = 0.5 * (1 + math.cos(math.pi * progress))
                return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)

            # ─────────────────────────────────────────────────────────────────────
            # Gradient Computation via Finite Differences (for illustration)
            # ─────────────────────────────────────────────────────────────────────
            def compute_grads_fd(params, token_ids, keys_to_diff, eps=1e-5):
                \"\"\"Finite-difference gradients for selected parameter arrays.\"\"\"
                grads = {}
                for key in keys_to_diff:
                    p    = params[key]
                    grad = np.zeros_like(p)
                    for idx in range(p.size):
                        grad.flat[idx] = numerical_grad(params, key, idx, token_ids, eps)
                    grads[key] = grad
                return grads

            # ─────────────────────────────────────────────────────────────────────
            # Training Loop
            # ─────────────────────────────────────────────────────────────────────
            TOTAL_STEPS   = 80
            WARMUP_STEPS  = 10
            PEAK_LR       = 5e-3
            GRAD_CLIP_NORM = 1.0

            params    = init_params()
            optimiser = AdamW(params, lr=PEAK_LR)

            # We only differentiate the embedding and LM head for speed
            # (full backprop would require autograd — this shows the loop structure)
            DIFF_KEYS = ["embed", "lm_W"]

            print("=" * 65)
            print("  Training Loop — AdamW + Cosine Schedule + Grad Clipping")
            print("=" * 65)
            print(f"  Model: {n_layers} layers, d={d_model}, h={n_heads}  "
                  f"  Vocab: {V}  Corpus: {len(CORPUS)} sequences")
            print(f"  AdamW: peak_lr={PEAK_LR}, wd=0.1, β₁=0.9, β₂=0.95")
            print(f"  Schedule: {WARMUP_STEPS} warmup steps → cosine decay over {TOTAL_STEPS} steps")
            print()

            loss_history = []
            lr_history   = []

            for step in range(TOTAL_STEPS):
                # Sample a sequence
                seq = CORPUS[step % len(CORPUS)]

                # Compute loss
                loss = compute_loss(params, seq)

                # Compute gradients (finite-difference for embed and lm_W)
                grads = compute_grads_fd(params, seq, DIFF_KEYS, eps=1e-5)

                # Gradient clipping
                total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
                if total_norm > GRAD_CLIP_NORM:
                    clip_coef = GRAD_CLIP_NORM / (total_norm + 1e-8)
                    grads     = {k: g * clip_coef for k, g in grads.items()}
                    clipped   = True
                else:
                    clipped   = False

                # LR schedule update
                current_lr = lr_schedule(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
                optimiser.lr = current_lr

                # Optimiser step
                optimiser.step(params, grads)

                loss_history.append(loss)
                lr_history.append(current_lr)

                if step % 10 == 0 or step == TOTAL_STEPS - 1:
                    ppl  = math.exp(min(loss, 10))
                    clip_marker = "✂" if clipped else " "
                    bar  = "█" * int((1 - min(loss,4)/4) * 20)
                    print(f"  step {step:4d}  loss={loss:.4f}  ppl={ppl:6.2f}  "
                          f"lr={current_lr:.5f}  grad_norm={total_norm:.3f} {clip_marker}  {bar}")

            print()
            print(f"  Final loss: {loss_history[-1]:.4f}   "
                  f"Initial loss: {loss_history[0]:.4f}   "
                  f"Improvement: {loss_history[0]-loss_history[-1]:.4f} nats")
            print()

            # ── LR Schedule Visualisation ────────────────────────────────────────
            print("── Learning Rate Schedule ──")
            print()
            max_lr = max(lr_history)
            for step in range(0, TOTAL_STEPS, TOTAL_STEPS//10):
                lr  = lr_history[step]
                bar = "█" * int(lr / max_lr * 40)
                phase = "WARMUP" if step < WARMUP_STEPS else "COSINE"
                print(f"  step {step:4d} [{phase:6s}]  lr={lr:.5f}  {bar}")

            print()
            print("── AdamW vs SGD: Why AdamW? ──")
            print("  SGD:   θ ← θ − lr · g")
            print("  Adam:  θ ← θ − lr · m̂ / (√v̂ + ε)    (adaptive per-param LR)")
            print("  AdamW: θ ← θ − lr · m̂ / (√v̂ + ε)  − lr · wd · θ")
            print("                                                ↑")
            print("  Weight decay decoupled from gradient — regularises parameters")
            print("  directly, not through the adaptive gradient scaling.")
        """,
    },

    # ── 7 ──────────────────────────────────────────────────────────────────
    "7 · Scaling Laws & Parameter Budget Calculator": {
        "description": (
            "Explore Chinchilla scaling laws — how loss, compute, and parameter count "
            "relate to each other via power laws. Implements a parameter budget calculator "
            "that breaks down exactly where parameters live across embedding, attention, "
            "and FFN layers for any model configuration."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            # ── Parameter Budget Calculator ──────────────────────────────────────
            def param_budget(name, N, d_model, n_heads, d_ff, vocab_size,
                             tie_embeddings=True, gqa_groups=None):
                \"\"\"
                Break down the parameter count of a decoder-only Transformer.
                gqa_groups: number of KV head groups for GQA (None = full MHA)
                \"\"\"
                d_k   = d_model // n_heads
                kv_heads = gqa_groups if gqa_groups else n_heads

                embed      = vocab_size * d_model
                lm_head    = 0 if tie_embeddings else vocab_size * d_model

                # Per-layer attention
                wq_per     = d_model * d_model          # Q: all heads
                wk_per     = d_model * (kv_heads * d_k) # K: kv_heads only (GQA)
                wv_per     = d_model * (kv_heads * d_k) # V: kv_heads only
                wo_per     = d_model * d_model          # output projection
                attn_per   = wq_per + wk_per + wv_per + wo_per
                attn_total = N * attn_per

                # Per-layer FFN
                ffn_per    = d_model * d_ff + d_ff * d_model   # W1 + W2
                ffn_total  = N * ffn_per

                # Norms (small)
                norm_total = N * 2 * d_model

                total = embed + lm_head + attn_total + ffn_total + norm_total

                return {
                    "name":         name,
                    "embed":        embed,
                    "lm_head":      lm_head,
                    "attn_total":   attn_total,
                    "ffn_total":    ffn_total,
                    "norm_total":   norm_total,
                    "total":        total,
                    "attn_pct":     attn_total / total * 100,
                    "ffn_pct":      ffn_total  / total * 100,
                }

            MODELS = [
                # name,           N,   d,      h,   d_ff,    vocab,  tie,  gqa
                ("GPT-2 Small",  12,   768,   12,   3072,  50257, True,  None),
                ("GPT-2 Large",  36,  1280,   20,   5120,  50257, True,  None),
                ("LLaMA-2 7B",   32,  4096,   32,  11008,  32000, True,  None),
                ("LLaMA-3 8B",   32,  4096,   32,  14336, 128256, True,     8),
                ("LLaMA-2 70B",  80,  8192,   64,  28672,  32000, True,     8),
                ("GPT-3 175B",   96, 12288,   96,  49152,  50257, True,  None),
            ]

            print("=" * 80)
            print("  Parameter Budget Calculator — Where Do the Parameters Live?")
            print("=" * 80)
            print()

            budgets = []
            for args in MODELS:
                b = param_budget(*args)
                budgets.append(b)

            print(f"  {'Model':18s}  {'Embed':>9s}  {'Attn':>9s}  {'FFN':>9s}  "
                  f"{'Norm':>6s}  {'Total':>10s}  {'Attn%':>6s}  {'FFN%':>5s}")
            print("  " + "─" * 78)
            for b in budgets:
                def fmt(n):
                    if n >= 1e9: return f"{n/1e9:.2f}B"
                    if n >= 1e6: return f"{n/1e6:.1f}M"
                    return f"{n/1e3:.1f}K"
                print(f"  {b['name']:18s}  {fmt(b['embed']):>9s}  {fmt(b['attn_total']):>9s}  "
                      f"{fmt(b['ffn_total']):>9s}  {fmt(b['norm_total']):>6s}  "
                      f"{fmt(b['total']):>10s}  {b['attn_pct']:>5.1f}%  {b['ffn_pct']:>4.1f}%")
            print()

            # ── GQA vs MHA Parameter Comparison ──────────────────────────────────
            print("── GQA vs MHA — KV Cache & Parameter Savings ──")
            print()
            print("  Config: LLaMA-3 8B (d=4096, h=32, layers=32)")
            print()
            print(f"  {'Strategy':20s}  {'KV heads':>8s}  {'Attn params':>12s}  "
                  f"{'KV-cache (2048 ctx)':>20s}")
            print("  " + "─" * 65)

            d = 4096; h = 32; N = 32; dk = d // h
            ctx_len = 2048; bytes_per = 2   # fp16

            for kv_h in [32, 8, 4, 1]:
                wq = d * d
                wk = d * (kv_h * dk)
                wv = d * (kv_h * dk)
                wo = d * d
                attn_p = N * (wq + wk + wv + wo)
                # KV cache: 2 (K+V) × ctx_len × kv_heads × dk × bytes × layers
                kv_cache_bytes = 2 * ctx_len * kv_h * dk * bytes_per * N
                label = "MHA" if kv_h == h else f"GQA g={kv_h}" if kv_h > 1 else "MQA"
                saving = (1 - kv_h/h) * 100
                print(f"  {label:20s}  {kv_h:>8d}  {attn_p/1e6:>10.1f}M  "
                      f"{kv_cache_bytes/1e6:>14.1f} MB   (-{saving:.0f}% vs MHA)")
            print()

            # ── Chinchilla Scaling Laws ──────────────────────────────────────────
            print("=" * 80)
            print("  Chinchilla Scaling Laws — Optimal Training Token Budget")
            print("=" * 80)
            print()
            print("  Key result (Hoffmann et al. 2022): for optimal compute use,")
            print("  train with D ≈ 20 × N tokens (N = number of parameters).")
            print()
            print("  L(N, D) ≈ E + A/N^α + B/D^β")
            print("  E=1.69, A=406.4, B=410.7, α=0.34, β=0.28  (fitted constants)")
            print()

            E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28

            def chinchilla_loss(N_params, D_tokens):
                return E + A / N_params**alpha + B / D_tokens**beta

            def optimal_tokens(N_params):
                return 20 * N_params   # Chinchilla rule of thumb

            print(f"  {'Model':18s}  {'N params':>10s}  {'Tokens used':>12s}  "
                  f"{'Optimal D':>12s}  {'Status':>15s}  {'Pred. Loss':>10s}")
            print("  " + "─" * 82)

            actual_data = [
                ("GPT-3 175B",   175e9, 300e9),
                ("Gopher 280B",  280e9, 300e9),
                ("Chinchilla 70B", 70e9, 1.4e12),
                ("LLaMA-2 7B",    7e9,  2e12),
                ("LLaMA-3 8B",    8e9, 15e12),
            ]

            def fmt_b(n):
                if n >= 1e12: return f"{n/1e12:.1f}T"
                if n >= 1e9:  return f"{n/1e9:.0f}B"
                return f"{n/1e6:.0f}M"

            for name, N_p, D_t in actual_data:
                opt_D = optimal_tokens(N_p)
                loss  = chinchilla_loss(N_p, D_t)
                ratio = D_t / opt_D
                if ratio < 0.5:
                    status = "⚠ undertrained"
                elif ratio < 1.5:
                    status = "✓ near-optimal"
                else:
                    status = "★ overtrained"
                print(f"  {name:18s}  {fmt_b(N_p):>10s}  {fmt_b(D_t):>12s}  "
                      f"{fmt_b(opt_D):>12s}  {status:>15s}  {loss:>10.4f}")

            print()
            print("  Insight: LLaMA-3 trains an 8B model on 15T tokens — far beyond")
            print("  Chinchilla optimal. This gives a smaller, efficient inference model")
            print("  at the cost of more training compute. Worth it for deployment.")
            print()

            # ── Scaling Law Power Laws ────────────────────────────────────────────
            print("── Loss vs Parameters (Compute-Optimal, Chinchilla) ──")
            print()
            print(f"  {'N (params)':>12s}  {'Optimal D':>12s}  {'Pred. Loss':>12s}  {'Pred. PPL':>10s}")
            print("  " + "─" * 52)
            for log_n in range(7, 12):
                N_p  = 10**log_n
                D_t  = optimal_tokens(N_p)
                loss = chinchilla_loss(N_p, D_t)
                ppl  = math.exp(loss)
                print(f"  {fmt_b(N_p):>12s}  {fmt_b(D_t):>12s}  {loss:>12.4f}  {ppl:>10.2f}")

            print()
            print("  Each 10× increase in N (at optimal D) reduces loss by ~0.5 nats.")
            print("  Gains slow down — this is why 'scaling is running out' concerns arise.")
        """,
    },

    # ── 8 ──────────────────────────────────────────────────────────────────
    "8 · SFT, RLHF & DPO — Alignment Training": {
        "description": (
            "Simulate all three stages of instruction-tuning alignment: "
            "Supervised Fine-Tuning (SFT) on prompt-response pairs, "
            "Reward Model training from human preference comparisons, "
            "and Direct Preference Optimisation (DPO) as a simpler RLHF alternative. "
            "Uses toy logit vectors to show the mechanics of each objective clearly."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            np.random.seed(0)

            # ─────────────────────────────────────────────────────────────────────
            # Stage 1: Supervised Fine-Tuning (SFT)
            # ─────────────────────────────────────────────────────────────────────
            print("=" * 65)
            print("  STAGE 1: Supervised Fine-Tuning (SFT)")
            print("=" * 65)
            print()
            print("  SFT dataset: (instruction, response) pairs")
            print("  Loss: CLM on response tokens only — instruction tokens masked")
            print()

            VOCAB = ["<PAD>","<BOS>","<EOS>","<INST>","</INST>",
                     "Paris","France","capital","is","the","of","city","beautiful",
                     "I","don't","know","London","unsure"]
            V   = len(VOCAB)
            w2i = {w: i for i, w in enumerate(VOCAB)}

            SFT_DATA = [
                {
                    "instruction": ["<INST>","What","is","the","capital","of","France","</INST>"],
                    "response":    ["<BOS>","The","capital","of","France","is","Paris",".","<EOS>"],
                },
                {
                    "instruction": ["<INST>","Name","a","beautiful","city","</INST>"],
                    "response":    ["<BOS>","Paris","is","a","beautiful","city",".","<EOS>"],
                },
            ]

            def softmax(z):
                z = z - z.max()
                return np.exp(z) / np.exp(z).sum()

            def simulate_logits(true_idx, strength=5.0):
                lg = np.random.randn(V)
                lg[true_idx] += strength
                return lg

            print(f"  {'Example':>3s}  {'Token':12s}  {'masked?':>8s}  {'P(token)':>10s}  {'NLL':>8s}")
            print("  " + "─" * 50)

            total_sft_loss = 0.0
            n_response_toks = 0

            for ex_idx, example in enumerate(SFT_DATA):
                full_seq = example["instruction"] + example["response"]
                resp_start = len(example["instruction"])

                for t in range(1, len(full_seq)):
                    tok      = full_seq[t]
                    is_resp  = t >= resp_start
                    tid      = w2i.get(tok, 0)
                    logits   = simulate_logits(tid, strength=5.0 if is_resp else 3.0)
                    probs    = softmax(logits)
                    nll      = -math.log(probs[tid] + 1e-12)

                    mask_str = "response" if is_resp else "MASKED"
                    if is_resp:
                        total_sft_loss += nll
                        n_response_toks += 1

                    if t < resp_start + 3 or t >= len(full_seq) - 2:
                        print(f"  {ex_idx+1:>3d}  {tok:12s}  {mask_str:>8s}  "
                              f"{probs[tid]:>10.4f}  {nll:>8.4f}")
                print(f"  {'...':>3s}  {'':12s}  {'':>8s}")

            sft_loss = total_sft_loss / n_response_toks
            print(f"\\n  SFT Mean Loss (response tokens only): {sft_loss:.4f}")
            print(f"  SFT Perplexity: {math.exp(sft_loss):.2f}")
            print()

            # ─────────────────────────────────────────────────────────────────────
            # Stage 2: Reward Model Training
            # ─────────────────────────────────────────────────────────────────────
            print("=" * 65)
            print("  STAGE 2: Reward Model Training")
            print("=" * 65)
            print()
            print("  Human labellers compare response pairs: (yᵥ, yₗ) — chosen vs rejected")
            print("  Reward model trained with Bradley-Terry loss:")
            print("  L_RM = -log σ(r(x, yᵥ) - r(x, yₗ))")
            print()

            # Each preference pair: (prompt, chosen, rejected)
            PREFERENCES = [
                {
                    "prompt":    "What is the capital of France?",
                    "chosen":    "The capital of France is Paris.",
                    "rejected":  "I don't know.",
                    "r_chosen":  2.5,    # simulated reward scores
                    "r_rejected":-1.2,
                },
                {
                    "prompt":    "Name a European capital city.",
                    "chosen":    "Paris is a beautiful capital city of France.",
                    "rejected":  "London is unsure, maybe Paris.",
                    "r_chosen":  1.8,
                    "r_rejected": 0.3,
                },
                {
                    "prompt":    "Describe France.",
                    "chosen":    "France is a country in Western Europe with a rich history.",
                    "rejected":  "France is the capital of Paris.",   # factually wrong
                    "r_chosen":  3.1,
                    "r_rejected":-2.5,
                },
            ]

            def sigmoid(x):
                return 1 / (1 + math.exp(-x))

            def rm_loss(r_chosen, r_rejected):
                return -math.log(sigmoid(r_chosen - r_rejected) + 1e-12)

            print(f"  {'Ex':>3s}  {'r(chosen)':>10s}  {'r(rejected)':>12s}  "
                  f"{'margin':>8s}  {'P(chosen>rej)':>14s}  {'RM Loss':>8s}")
            print("  " + "─" * 62)

            total_rm_loss = 0.0
            for i, pref in enumerate(PREFERENCES):
                rc  = pref["r_chosen"]
                rl  = pref["r_rejected"]
                margin = rc - rl
                prob   = sigmoid(margin)
                loss   = rm_loss(rc, rl)
                total_rm_loss += loss
                print(f"  {i+1:>3d}  {rc:>10.2f}  {rl:>12.2f}  "
                      f"{margin:>8.2f}  {prob:>14.4f}  {loss:>8.4f}")

            print(f"\\n  Mean RM Loss: {total_rm_loss/len(PREFERENCES):.4f}")
            print()

            # ─────────────────────────────────────────────────────────────────────
            # Stage 3: DPO — Direct Preference Optimisation
            # ─────────────────────────────────────────────────────────────────────
            print("=" * 65)
            print("  STAGE 3: Direct Preference Optimisation (DPO)")
            print("=" * 65)
            print()
            print("  DPO skips the reward model entirely.")
            print("  Loss: L_DPO = -E[log σ(β·(log π(yᵥ|x) - log π_ref(yᵥ|x))")
            print("                          - β·(log π(yₗ|x) - log π_ref(yₗ|x)))]")
            print()
            print("  β = temperature parameter controlling deviation from reference model")
            print()

            beta = 0.1   # DPO temperature

            # Simulated log-probabilities from current policy and reference (SFT) policy
            DPO_EXAMPLES = [
                {
                    "prompt":         "Capital of France?",
                    "chosen":         "Paris",
                    "rejected":       "I don't know",
                    "log_pi_chosen":  -1.2,   # current model
                    "log_pi_rej":     -3.5,
                    "log_ref_chosen": -1.5,   # reference (SFT) model
                    "log_ref_rej":    -2.8,
                },
                {
                    "prompt":         "Best European city?",
                    "chosen":         "Paris is a cultural hub.",
                    "rejected":       "I am unsure.",
                    "log_pi_chosen":  -2.1,
                    "log_pi_rej":     -4.0,
                    "log_ref_chosen": -2.3,
                    "log_ref_rej":    -3.5,
                },
            ]

            def dpo_loss(log_pi_chosen, log_pi_rej, log_ref_chosen, log_ref_rej, beta):
                # Implicit reward: β * (log π - log π_ref)
                reward_chosen  = beta * (log_pi_chosen - log_ref_chosen)
                reward_rejected = beta * (log_pi_rej   - log_ref_rej)
                loss = -math.log(sigmoid(reward_chosen - reward_rejected) + 1e-12)
                return loss, reward_chosen, reward_rejected

            print(f"  β = {beta}")
            print()
            print(f"  {'Ex':>3s}  {'chosen':20s}  {'r(chosen)':>10s}  "
                  f"{'r(rejected)':>12s}  {'DPO Loss':>10s}")
            print("  " + "─" * 62)

            total_dpo = 0.0
            for i, ex in enumerate(DPO_EXAMPLES):
                loss, r_c, r_rej = dpo_loss(
                    ex["log_pi_chosen"], ex["log_pi_rej"],
                    ex["log_ref_chosen"], ex["log_ref_rej"], beta
                )
                total_dpo += loss
                print(f"  {i+1:>3d}  {ex['chosen'][:20]:20s}  {r_c:>10.4f}  "
                      f"{r_rej:>12.4f}  {loss:>10.4f}")

            print(f"\\n  Mean DPO Loss: {total_dpo/len(DPO_EXAMPLES):.4f}")
            print()

            # ── Pipeline Comparison ──────────────────────────────────────────────
            print("=" * 65)
            print("  Alignment Pipeline Comparison")
            print("=" * 65)
            print()
            rows = [
                ("Stage",       "SFT",             "RLHF",                "DPO"),
                ("Input data",  "Instruct+response","Human pref pairs",    "Human pref pairs"),
                ("Extra model", "None",             "Reward model",        "None"),
                ("Training",    "CLM loss",         "PPO + KL penalty",    "Closed-form loss"),
                ("Stability",   "Very stable",      "Tricky (RL)",         "Stable"),
                ("Complexity",  "Low",              "High",                "Low"),
                ("Quality",     "Good baseline",    "State-of-the-art",    "Near-RLHF"),
                ("Examples",    "LLaMA-2-Chat",     "InstructGPT, Claude", "Zephyr, Tulu-2"),
            ]
            col_w = [14, 20, 22, 20]
            for row in rows:
                line = "  "
                for i, cell in enumerate(row[:4]):
                    line += cell[:col_w[i]].ljust(col_w[i]) + " │ "
                print(line)
                if row[0] == "Stage":
                    print("  " + "─" * 80)
        """,
    },

    # ── 9 ──────────────────────────────────────────────────────────────────
    "9 · Flash Attention — Tiled SRAM Computation": {
        "description": (
            "Simulate Flash Attention's tiled block computation and compare it against "
            "standard attention. Shows how partial softmax is computed using running max "
            "and denominator without materialising the full n×n matrix. Measures memory "
            "access patterns and demonstrates the IO complexity advantage."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            np.random.seed(42)

            # ── Setup ────────────────────────────────────────────────────────────
            seq_len = 8       # small for illustration
            d_k     = 4
            TOKENS  = [f"t{i}" for i in range(seq_len)]

            Q = np.random.randn(seq_len, d_k).astype(np.float32) * 0.5
            K = np.random.randn(seq_len, d_k).astype(np.float32) * 0.5
            V = np.random.randn(seq_len, d_k).astype(np.float32) * 0.5

            # ── Standard Attention ───────────────────────────────────────────────
            def standard_attention(Q, K, V, causal=True):
                \"\"\"
                Standard attention: materialises the full (n,n) score matrix.
                HBM reads: Q + K + V + S + A = multiple full passes over n² matrix.
                \"\"\"
                T = Q.shape[0]
                S = Q @ K.T / math.sqrt(d_k)           # (T, T) — full matrix in memory

                if causal:
                    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
                    S[mask] = -1e9

                A = np.exp(S - S.max(axis=-1, keepdims=True))  # (T, T) in memory
                A = A / A.sum(axis=-1, keepdims=True)

                out = A @ V
                return out, S, A

            std_out, S_matrix, A_matrix = standard_attention(Q, K, V, causal=True)

            print("=" * 65)
            print("  STANDARD ATTENTION — Memory Analysis")
            print("=" * 65)
            print()
            print(f"  Sequence length:  {seq_len}")
            print(f"  Head dimension:   {d_k}")
            print()
            print(f"  Tensors materialised in HBM (GPU main memory):")
            print(f"    Q, K, V matrices:  3 × ({seq_len} × {d_k}) = {3*seq_len*d_k} floats")
            print(f"    Score matrix S:    ({seq_len} × {seq_len}) = {seq_len**2} floats  ← O(n²)")
            print(f"    Attention A:       ({seq_len} × {seq_len}) = {seq_len**2} floats  ← O(n²)")
            print(f"    Output:            ({seq_len} × {d_k}) = {seq_len*d_k} floats")
            print()
            print(f"  Peak memory (floats): {3*seq_len*d_k + 2*seq_len**2 + seq_len*d_k}")
            print(f"  The n² matrices are the problem at long sequences:")
            print(f"    seq=4096:  4096² × 2 = 33.6M floats = 67 MB per head per layer")
            print(f"    seq=32768: 32768² × 2 = 2.1B floats = 4.3 GB per head per layer")
            print()
            print("  Attention weight matrix A (causal):")
            print(f"  {'':6s}", end="")
            for t in TOKENS:
                print(f"  {t:4s}", end="")
            print()
            for i in range(seq_len):
                print(f"  {TOKENS[i]:4s} │", end="")
                for j in range(seq_len):
                    if j > i:
                        print(f"  {'·':4s}", end="")
                    else:
                        print(f"  {A_matrix[i,j]:.2f}", end="")
                print()

            # ── Flash Attention — Tiled Block Algorithm ───────────────────────
            def flash_attention(Q, K, V, block_size=4, causal=True):
                \"\"\"
                Flash Attention: compute output without materialising full n×n matrix.

                Key trick: online softmax with running max (m) and denominator (l).
                For each output row block (Qᵢ), iterate over key/value blocks (Kⱼ, Vⱼ).
                Update running statistics without ever having the full row in memory.

                HBM reads: O(n) — Q, K, V each read once per pass
                SRAM usage: O(block_size × d_k) — fits in fast cache
                \"\"\"
                T, dk = Q.shape
                out   = np.zeros((T, dk), dtype=np.float32)   # final output accumulator

                hbm_accesses = 0    # track IO operations

                for i_start in range(0, T, block_size):
                    i_end  = min(i_start + block_size, T)
                    Qi     = Q[i_start:i_end]          # load Q block into SRAM
                    hbm_accesses += Qi.size

                    # Running softmax statistics for this Q block
                    mi  = np.full((i_end - i_start, 1), -np.inf, dtype=np.float32)  # running max
                    li  = np.zeros((i_end - i_start, 1), dtype=np.float32)           # running denom
                    oi  = np.zeros((i_end - i_start, dk), dtype=np.float32)          # running output

                    for j_start in range(0, T, block_size):
                        j_end = min(j_start + block_size, T)
                        Kj = K[j_start:j_end]          # load K block into SRAM
                        Vj = V[j_start:j_end]          # load V block into SRAM
                        hbm_accesses += Kj.size + Vj.size

                        # Compute attention scores for this (Qi, Kj) block
                        Sij = Qi @ Kj.T / math.sqrt(dk)    # (block, block) — fits in SRAM

                        # Apply causal mask
                        if causal:
                            for r in range(i_end - i_start):
                                for c in range(j_end - j_start):
                                    if j_start + c > i_start + r:
                                        Sij[r, c] = -1e9

                        # Online softmax update:  m_new = max(m_old, max(Sij))
                        mij_new = np.maximum(mi, Sij.max(axis=-1, keepdims=True))
                        Pij     = np.exp(Sij - mij_new)                      # renormalised scores

                        # Rescale previous running values, add new contribution
                        scale   = np.exp(mi - mij_new)
                        li      = scale * li + Pij.sum(axis=-1, keepdims=True)
                        oi      = scale * oi + Pij @ Vj
                        mi      = mij_new

                    # Finalise output for this Q block
                    out[i_start:i_end] = oi / li         # divide by accumulated denominator

                return out, hbm_accesses

            flash_out, flash_hbm = flash_attention(Q, K, V, block_size=4, causal=True)

            # Standard HBM accesses (rough count)
            std_hbm = 3*seq_len*d_k + 2*seq_len**2 + seq_len*d_k

            print()
            print("=" * 65)
            print("  FLASH ATTENTION — Tiled Block Algorithm")
            print("=" * 65)
            print()
            print("  Algorithm: iterate over tiles of (Qᵢ, Kⱼ, Vⱼ) in SRAM")
            print("  Use online softmax (running max + denominator) per tile")
            print("  Never write the full n×n score matrix to HBM")
            print()
            print(f"  HBM accesses — Standard: {std_hbm} floats")
            print(f"  HBM accesses — Flash:    {flash_hbm} floats")
            print(f"  Reduction ratio:          {std_hbm/flash_hbm:.2f}×")
            print()

            # Verify outputs are identical
            max_err = np.abs(std_out - flash_out).max()
            print(f"  Max absolute error (Flash vs Standard): {max_err:.2e}")
            print(f"  Outputs identical: {'✓ YES' if max_err < 1e-4 else '✗ NO'}")
            print()
            print("  Memory complexity:")
            print(f"    Standard: O(n²) — score matrix grows quadratically with seq_len")
            print(f"    Flash:    O(n)  — only tiles in SRAM, output is O(n)")
            print()
            print("  At seq_len=32768 (long context):")
            print(f"    Standard score matrix: {32768**2 * 2 / 1e9:.1f} GB per head per layer")
            print(f"    Flash SRAM tile (bs=256): {256 * 256 * 4 / 1e6:.2f} MB — fits in L2 cache")
            print()
            print("  Speed improvements (empirical, A100):")
            print("    Flash Attention 1:   ~2× over PyTorch standard")
            print("    Flash Attention 2:   ~4× over PyTorch standard (better parallelism)")
            print("    Flash Attention 3:   ~6× over PyTorch standard (H100 async)")
        """,
    },

    # ── 10 ─────────────────────────────────────────────────────────────────
    "10 · KV Cache — Efficient Autoregressive Inference": {
        "description": (
            "Simulate the KV cache mechanism for autoregressive token generation. "
            "Compare token generation with and without caching to show the O(n²) vs O(n) "
            "difference. Calculate KV cache memory usage for real LLaMA models and explain "
            "why GQA/MQA are critical for long-context inference."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            np.random.seed(0)

            # ── Setup ────────────────────────────────────────────────────────────
            d_model  = 16
            n_heads  = 4
            d_k      = d_model // n_heads
            n_layers = 2

            VOCAB = ["<BOS>", "The", "cat", "sat", "on", "mat", "and", "ran", "<EOS>"]
            PROMPT = ["<BOS>", "The", "cat"]

            Wq = [np.random.randn(d_model, d_k).astype(np.float32) * 0.1 for _ in range(n_heads)]
            Wk = [np.random.randn(d_model, d_k).astype(np.float32) * 0.1 for _ in range(n_heads)]
            Wv = [np.random.randn(d_model, d_k).astype(np.float32) * 0.1 for _ in range(n_heads)]
            embed = np.random.randn(len(VOCAB), d_model).astype(np.float32) * 0.1

            def softmax(z):
                z = z - z.max(axis=-1, keepdims=True)
                return np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)

            # ── Generation WITHOUT KV Cache ───────────────────────────────────
            def generate_no_cache(prompt_ids, n_new=4):
                \"\"\"Naive autoregressive generation: recompute all K,V every step.\"\"\"
                all_ids    = list(prompt_ids)
                total_ops  = 0

                for step in range(n_new):
                    T = len(all_ids)
                    x = embed[all_ids]              # (T, d_model)

                    # For EACH new token: process ALL T tokens through attention
                    for head in range(n_heads):
                        Q = x @ Wq[head]            # (T, d_k)
                        K = x @ Wk[head]            # (T, d_k) — RECOMPUTED every step
                        V = x @ Wv[head]            # (T, d_k) — RECOMPUTED every step
                        S = Q @ K.T / math.sqrt(d_k)     # (T, T)
                        A = softmax(S)
                        _ = A @ V                   # (T, d_k)
                        total_ops += T * T          # QK dot product: T×T every step

                    # Greedy: pick most probable next token
                    logits = x[-1] @ embed.T       # (vocab_size,)
                    next_id = np.argmax(logits)
                    all_ids.append(next_id)

                return all_ids, total_ops

            # ── Generation WITH KV Cache ──────────────────────────────────────
            def generate_with_cache(prompt_ids, n_new=4):
                \"\"\"
                KV Cache generation:
                - Process prompt once to fill the cache
                - For each new token: only process the new token,
                  read K,V for all previous tokens from cache
                \"\"\"
                all_ids   = list(prompt_ids)
                total_ops = 0

                # Initialise KV cache: one K, V per head
                kv_cache = {h: {"K": [], "V": []} for h in range(n_heads)}

                # ── Prefill: process all prompt tokens at once ──────────────
                T_prompt = len(prompt_ids)
                x_prompt = embed[prompt_ids]       # (T_prompt, d_model)

                for head in range(n_heads):
                    K_prompt = x_prompt @ Wk[head] # (T_prompt, d_k)
                    V_prompt = x_prompt @ Wv[head]
                    kv_cache[head]["K"] = list(K_prompt)   # store all prompt K vectors
                    kv_cache[head]["V"] = list(V_prompt)
                    total_ops += T_prompt * T_prompt        # prefill attention cost

                # ── Decode: generate one token at a time ───────────────────
                for step in range(n_new):
                    curr_id = all_ids[-1]
                    x_new   = embed[curr_id:curr_id+1]     # (1, d_model) — just new token

                    head_outputs = []
                    for head in range(n_heads):
                        q_new = x_new @ Wq[head]           # (1, d_k) — only new query

                        # Append new K, V to cache
                        k_new = x_new @ Wk[head]
                        v_new = x_new @ Wv[head]
                        kv_cache[head]["K"].append(k_new[0])
                        kv_cache[head]["V"].append(v_new[0])

                        # Attention over ALL past + current tokens (from cache)
                        K_all = np.stack(kv_cache[head]["K"])  # (T_past, d_k)
                        V_all = np.stack(kv_cache[head]["V"])

                        scores = q_new @ K_all.T / math.sqrt(d_k)  # (1, T_past)
                        attn   = softmax(scores)
                        ctx    = attn @ V_all                        # (1, d_k)
                        head_outputs.append(ctx)
                        total_ops += len(kv_cache[head]["K"])        # only 1 query vs T_past keys

                    context = np.concatenate(head_outputs, axis=-1)  # (1, d_model)
                    logits  = context @ embed.T
                    next_id = np.argmax(logits)
                    all_ids.append(int(next_id))

                return all_ids, total_ops

            N_NEW = 4

            ids_no_cache, ops_no  = generate_no_cache(list(range(len(PROMPT))), N_NEW)
            ids_cache,    ops_yes = generate_with_cache(list(range(len(PROMPT))), N_NEW)

            print("=" * 65)
            print("  KV Cache — Autoregressive Generation Comparison")
            print("=" * 65)
            print(f"  Prompt: {PROMPT}   Generating {N_NEW} new tokens")
            print()
            print(f"  {'':30s}  {'No Cache':>12s}  {'With Cache':>12s}")
            print("  " + "─" * 58)
            print(f"  {'Attention ops (QK dot products)':30s}  {ops_no:>12,}  {ops_yes:>12,}")
            print(f"  {'Speedup':30s}  {'1.0×':>12s}  {ops_no/ops_yes:>11.2f}×")
            print()
            print("  Without cache: at each new token, process ALL past tokens again")
            print("  Total ops grow as: T₀ + (T₀+1) + (T₀+2)... = O(T²)")
            print()
            print("  With cache: only 1 new query vector, K/V from cache")
            print("  Total ops grow as: T_prompt² + n_new × T_growing = O(T)")
            print()

            # ── KV Cache Memory Calculator ────────────────────────────────
            print("=" * 65)
            print("  KV Cache Memory Calculator — Real LLaMA Models")
            print("=" * 65)
            print()
            print("  Formula: 2 × seq_len × n_kv_heads × d_head × bytes × n_layers")
            print()
            print(f"  {'Model':20s} {'Layers':>7} {'KV heads':>9} {'d_head':>7} "
                  f"{'ctx_len':>8} {'Cache (BF16)':>14}")
            print("  " + "─" * 70)

            models = [
                ("LLaMA-3 8B",   32,  8, 128,   8192),
                ("LLaMA-3 70B",  80,  8, 128,   8192),
                ("LLaMA-3 405B", 126, 8, 128,   8192),
                ("Mistral 7B",   32,  8, 128,  32768),
                ("GPT-3 175B",   96, 96, 128,   4096),   # MHA (96 KV heads)
            ]

            for name, layers, kv_heads, d_head, ctx in models:
                bytes_bf16 = 2
                cache_bytes = 2 * ctx * kv_heads * d_head * bytes_bf16 * layers
                cache_gb    = cache_bytes / 1e9
                print(f"  {name:20s} {layers:>7d} {kv_heads:>9d} {d_head:>7d} "
                      f"{ctx:>8,} {cache_gb:>12.2f} GB")

            print()
            print("  Key insight: LLaMA-3 70B with 8K context needs only 2.68 GB cache")
            print("  vs GPT-3 175B full MHA at 4K context: 18.87 GB cache")
            print("  GQA (8 KV heads vs 96) → 12× cache reduction at similar quality")
            print()

            # ── Rolling Cache for Sliding Window ─────────────────────────────
            print("── Rolling KV Cache (Sliding Window Attention) ──")
            print()
            window = 6
            full_seq_len = 12
            print(f"  Sliding window W={window}, sequence length={full_seq_len}")
            print(f"  Cache holds last {window} K,V vectors only (fixed size)")
            print()
            cache = []
            for t in range(full_seq_len):
                cache.append(f"KV_{t}")
                if len(cache) > window:
                    cache.pop(0)    # evict oldest
                active = ', '.join(cache)
                evicted = f"evicted KV_{t-window}" if t >= window else "no eviction"
                print(f"  t={t:2d}: cache=[{active}]  ({evicted})")
            print()
            print(f"  Memory: O(W) = O({window}) regardless of sequence length ✓")
        """,
    },

    # ── 11 ─────────────────────────────────────────────────────────────────
    "11 · Mixed Precision Training & Gradient Checkpointing": {
        "description": (
            "Simulate BF16 mixed precision training: show the master FP32 weight copy, "
            "BF16 forward/backward pass, and FP32 optimizer update cycle. Then demonstrate "
            "gradient checkpointing memory savings vs standard backprop, with concrete "
            "memory calculations for real model scales."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            np.random.seed(42)

            # ── Part 1: Mixed Precision Training ─────────────────────────────
            print("=" * 65)
            print("  MIXED PRECISION TRAINING — BF16 Forward / FP32 Optimizer")
            print("=" * 65)
            print()

            def simulate_precision_formats():
                \"\"\"Show how BF16 vs FP32 represent the same values.\"\"\"
                test_values = [0.1, 1.23456789, 65504.0, 0.000123, -3.14159]

                print("  Value precision comparison (FP32 vs BF16 approximation):")
                print(f"  {'Original FP32':>20s}  {'BF16 approx':>15s}  {'Abs Error':>12s}  {'Status':>10s}")
                print("  " + "─" * 65)

                for v in test_values:
                    # BF16: truncate FP32 mantissa from 23 bits to 7 bits
                    # (keep sign + 8 exponent bits, drop lowest 16 mantissa bits)
                    fp32_bits = np.float32(v).view(np.uint32)
                    bf16_bits = (fp32_bits >> 16).astype(np.uint16)         # truncate
                    fp32_from_bf16 = (bf16_bits.astype(np.uint32) << 16).view(np.float32)
                    err = abs(float(v) - float(fp32_from_bf16))

                    # FP16 overflow check (max ≈ 65504)
                    fp16_ok = abs(v) < 65500
                    print(f"  {v:>20.8f}  {float(fp32_from_bf16):>15.6f}  {err:>12.2e}  "
                          f"{'FP16 ok' if fp16_ok else 'FP16 OVERFLOW':>10s}")

                print()
                print("  BF16 vs FP16 exponent range:")
                print("    FP32:  8 exp bits  → max ±3.4×10³⁸  (no overflow in training)")
                print("    BF16:  8 exp bits  → max ±3.4×10³⁸  (same as FP32 — SAFE)")
                print("    FP16:  5 exp bits  → max ±65504      (overflow risk in training!)")
                print()
                print("  → BF16 is preferred for training; FP16 needs loss scaling")

            simulate_precision_formats()

            # ── Mixed Precision Training Cycle ────────────────────────────────
            print()
            print("── Mixed Precision Training Cycle (1 step) ──")
            print()

            d_model = 8
            params_fp32  = np.random.randn(d_model, d_model).astype(np.float32)
            params_bf16  = params_fp32.copy()    # in real code: cast to bf16
            lr = 1e-3

            # Simulate Adam states (always FP32)
            m = np.zeros_like(params_fp32)
            v = np.zeros_like(params_fp32)
            t = 1

            # Simulated gradient (computed in BF16, here as float32 for illustration)
            grad_bf16 = np.random.randn(d_model, d_model).astype(np.float32) * 0.01
            grad_fp32 = grad_bf16.copy()    # cast to FP32 for optimizer

            # AdamW update in FP32
            beta1, beta2, eps, wd = 0.9, 0.95, 1e-8, 0.1
            m = beta1 * m + (1 - beta1) * grad_fp32
            v = beta2 * v + (1 - beta2) * grad_fp32**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            params_fp32 = params_fp32 - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * wd * params_fp32
            params_bf16 = params_fp32.copy()    # cast back to bf16 for next forward pass

            steps = [
                ("1. Load BF16 weights from GPU memory",       "params_bf16",  "2 bytes/param"),
                ("2. BF16 forward pass (activations)",         "activations",  "~2× BF16 params"),
                ("3. BF16 backward pass (gradients)",          "grad_bf16",    "2 bytes/param"),
                ("4. Cast gradient → FP32",                    "grad_fp32",    "4 bytes/param"),
                ("5. FP32 AdamW update (m, v, master weights)","optimizer",    "12 bytes/param"),
                ("6. Cast updated weights → BF16",             "params_bf16",  "2 bytes/param"),
            ]
            for i, (desc, tensor, mem) in enumerate(steps):
                print(f"  {i+1}. {desc}")
                print(f"     [{tensor}] — {mem}")
                print()

            # ── Memory breakdown ──────────────────────────────────────────────
            print("── Memory Per Parameter in Mixed Precision ──")
            print()
            components = [
                ("BF16 model weights",        2),
                ("BF16 gradients",            2),
                ("FP32 master weights",        4),
                ("FP32 Adam momentum (m)",     4),
                ("FP32 Adam variance (v)",     4),
            ]
            total_bytes = sum(b for _, b in components)
            for name, b in components:
                bar = "█" * b
                print(f"  {name:30s}: {b} bytes  {bar}")
            print(f"  {'─'*48}")
            print(f"  {'Total':30s}: {total_bytes} bytes/param")
            print()

            model_sizes = [("7B", 7e9), ("8B", 8e9), ("70B", 70e9), ("405B", 405e9)]
            print(f"  {'Model':10s}  {'Training (GB)':>15s}  {'Inference BF16 (GB)':>20s}")
            print("  " + "─" * 50)
            for name, N in model_sizes:
                train_gb = N * total_bytes / 1e9
                infer_gb = N * 2 / 1e9
                print(f"  {name:10s}  {train_gb:>15.1f}  {infer_gb:>20.1f}")

            # ── Part 2: Gradient Checkpointing ────────────────────────────────
            print()
            print("=" * 65)
            print("  GRADIENT CHECKPOINTING — Trading Compute for Memory")
            print("=" * 65)
            print()

            def memory_analysis(n_layers, seq_len, d_model, batch_size, bytes_per=2):
                \"\"\"
                Estimate activation memory for a Transformer decoder stack.
                Each block stores: attention scores + intermediate FFN = ~4 × seq × d tensors
                \"\"\"
                # Per-layer activations stored during forward pass:
                # x_input, x_after_attn, x_after_norm, x_ffn_hidden, x_output
                acts_per_layer = 5 * seq_len * d_model * batch_size * bytes_per
                total_standard = n_layers * acts_per_layer

                # Checkpointing every layer: only store layer inputs
                # Recompute intermediate activations during backward
                acts_checkpoint = n_layers * (seq_len * d_model * batch_size * bytes_per)
                # Savings: don't store intermediate acts (4 × per layer)
                checkpoint_mem  = n_layers * 1 * seq_len * d_model * batch_size * bytes_per

                return {
                    "standard_gb":    total_standard / 1e9,
                    "checkpoint_gb":  checkpoint_mem / 1e9,
                    "saving_factor":  total_standard / checkpoint_mem,
                    "extra_flops":    "~33% more (one extra forward pass)",
                }

            configs = [
                ("LLaMA-3 8B  (batch=8,  seq=4096)",  32, 4096, 4096, 8),
                ("LLaMA-3 70B (batch=4,  seq=4096)",  80, 4096, 8192, 4),
                ("GPT-3 175B  (batch=2,  seq=2048)",  96, 2048, 12288, 2),
            ]

            print(f"  {'Config':44s}  {'Standard GB':>12s}  {'Checkpointed GB':>16s}  {'Saving':>7s}")
            print("  " + "─" * 83)
            for label, n_l, seq, d, bs in configs:
                r = memory_analysis(n_l, seq, d, bs)
                print(f"  {label:44s}  {r['standard_gb']:>12.1f}  {r['checkpoint_gb']:>16.1f}  "
                      f"{r['saving_factor']:>6.1f}×")

            print()
            print("  PyTorch API:  torch.utils.checkpoint.checkpoint(fn, *inputs)")
            print("  When to use:  always during pretraining of large models")
            print("  Trade-off:    ~33% more compute for 4–5× less activation memory")
            print()
            print("── Optimal Checkpoint Placement Strategy ──")
            print()
            print("  Strategy 1: Checkpoint every layer boundary (most common)")
            print("    Memory: O(n_layers) — just layer inputs")
            print("    Recompute: 1 extra forward per layer during backward")
            print()
            print("  Strategy 2: Checkpoint every k layers (√n strategy)")
            print("    Memory: O(√n_layers) — intermediate states at checkpoints")
            print("    Recompute: up to k forward passes per segment")
            print()
            n_l = 32
            k   = int(math.sqrt(n_l))
            print(f"  For {n_l} layers: optimal k ≈ √{n_l} = {k}")
            print(f"  Memory:    store {n_l//k} checkpoints + recompute up to {k} layers each")
        """,
    },

    # ── 12 ─────────────────────────────────────────────────────────────────
    "12 · Mixture of Experts (MoE) — Routing & Load Balancing": {
        "description": (
            "Implement a Mixture of Experts FFN layer with a learned router, top-k expert "
            "selection, gated output combination, and load balancing loss. Simulate routing "
            "behaviour across a batch of tokens and visualise expert utilisation. Compare "
            "FLOPs and parameter counts vs dense equivalents."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            np.random.seed(0)

            # ── Config ────────────────────────────────────────────────────────
            d_model    = 16
            d_ff       = 32        # each expert's hidden dim
            n_experts  = 8
            top_k      = 2         # tokens routed to top-2 experts
            n_tokens   = 24        # batch of tokens to route

            # Token embeddings (simulated)
            tokens = np.random.randn(n_tokens, d_model).astype(np.float32) * 0.5

            # Router weight matrix: d_model → n_experts
            W_router = np.random.randn(d_model, n_experts).astype(np.float32) * 0.1

            # Expert FFN weights (each expert: W1 ∈ d_model×d_ff, W2 ∈ d_ff×d_model)
            experts_W1 = [np.random.randn(d_model, d_ff).astype(np.float32) * 0.1
                          for _ in range(n_experts)]
            experts_W2 = [np.random.randn(d_ff, d_model).astype(np.float32) * 0.1
                          for _ in range(n_experts)]

            def softmax(z, axis=-1):
                z = z - z.max(axis=axis, keepdims=True)
                return np.exp(z) / np.exp(z).sum(axis=axis, keepdims=True)

            def gelu(x):
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

            # ── Router ─────────────────────────────────────────────────────────
            def route_tokens(tokens, W_router, top_k):
                \"\"\"
                Route each token to top_k experts.
                Returns: expert assignments, gate values, router logits
                \"\"\"
                logits   = tokens @ W_router                    # (n_tokens, n_experts)
                probs    = softmax(logits, axis=-1)             # (n_tokens, n_experts)

                # Select top-k experts per token
                top_k_idx  = np.argsort(probs, axis=-1)[:, -top_k:]  # (n_tokens, k)
                top_k_probs = np.take_along_axis(probs, top_k_idx, axis=-1)

                # Re-normalise gate values over selected experts
                top_k_gates = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

                return top_k_idx, top_k_gates, probs

            # ── MoE Forward Pass ───────────────────────────────────────────────
            def moe_forward(tokens, W_router, experts_W1, experts_W2, top_k):
                n_tok = tokens.shape[0]
                d     = tokens.shape[1]
                out   = np.zeros_like(tokens)

                top_k_idx, top_k_gates, router_probs = route_tokens(tokens, W_router, top_k)

                # Expert utilisation counter
                expert_counts = np.zeros(n_experts, dtype=int)

                for t in range(n_tok):
                    token_out = np.zeros(d, dtype=np.float32)
                    for ki in range(top_k):
                        expert_id = top_k_idx[t, ki]
                        gate      = top_k_gates[t, ki]

                        # Expert FFN
                        h = gelu(tokens[t] @ experts_W1[expert_id])
                        expert_out = h @ experts_W2[expert_id]

                        token_out += gate * expert_out
                        expert_counts[expert_id] += 1

                    out[t] = token_out

                return out, router_probs, expert_counts

            output, router_probs, expert_counts = moe_forward(
                tokens, W_router, experts_W1, experts_W2, top_k
            )

            print("=" * 65)
            print("  Mixture of Experts (MoE) — Forward Pass")
            print("=" * 65)
            print(f"  n_experts={n_experts},  top_k={top_k},  n_tokens={n_tokens}")
            print(f"  d_model={d_model},  d_ff={d_ff} per expert")
            print()

            # ── Routing Visualisation ──────────────────────────────────────────
            top_k_idx_all, top_k_gates_all, _ = route_tokens(tokens, W_router, top_k)
            print("── Token Routing (first 10 tokens) ──")
            print()
            print(f"  {'Token':>6s}  {'Expert 0':>7s}  {'Expert 1':>7s}  {'Gate 0':>7s}  {'Gate 1':>7s}")
            print("  " + "─" * 45)
            for t in range(min(10, n_tokens)):
                e0, e1 = top_k_idx_all[t, -1], top_k_idx_all[t, -2]
                g0, g1 = top_k_gates_all[t, -1], top_k_gates_all[t, -2]
                print(f"  {t:>6d}  Expert {e0:>1d}           Expert {e1:>1d}       {g0:.4f}   {g1:.4f}")

            # ── Expert Utilisation ─────────────────────────────────────────────
            print()
            print("── Expert Utilisation ──")
            print()
            max_count = expert_counts.max()
            ideal = n_tokens * top_k / n_experts
            print(f"  Ideal tokens per expert (uniform): {ideal:.1f}")
            print()
            for e in range(n_experts):
                count = expert_counts[e]
                bar   = "█" * count + "░" * max(0, int(ideal * 2) - count)
                load_ratio = count / ideal
                flag = "⚠ OVERLOADED" if load_ratio > 1.5 else ("✓" if load_ratio > 0.5 else "⚠ UNDERUSED")
                print(f"  Expert {e}: {count:>3d} tokens  {bar}  ({load_ratio:.2f}× ideal) {flag}")

            # ── Load Balancing Loss ────────────────────────────────────────────
            print()
            print("── Load Balancing Auxiliary Loss ──")
            print()
            print("  Without balancing loss, router collapses to 1-2 experts ('routing collapse')")
            print()
            print("  L_balance = α × n_experts × Σᵢ (fᵢ × Pᵢ)")
            print("  fᵢ = fraction of tokens routed to expert i")
            print("  Pᵢ = mean router probability for expert i")
            print("  α  = small coefficient (e.g. 0.01)")
            print()

            alpha = 0.01
            f = expert_counts / expert_counts.sum()          # fraction routed to each expert
            P = router_probs.mean(axis=0)                    # mean router prob per expert
            L_balance = alpha * n_experts * np.dot(f, P)

            print(f"  Expert load fractions f: {np.round(f, 3)}")
            print(f"  Mean router probs P:     {np.round(P, 3)}")
            print(f"  L_balance:               {L_balance:.6f}")
            print()

            # ── FLOPs & Parameter Comparison ──────────────────────────────────
            print("── Dense FFN vs MoE FFN — FLOPs and Parameters ──")
            print()
            print(f"  Config: d_model={d_model},  n_experts={n_experts},  top_k={top_k},  d_ff={d_ff}")
            print()

            # Dense equivalent (same active FLOPs)
            d_ff_dense = d_ff  # same per-expert ff dim
            dense_params  = 2 * d_model * d_ff_dense              # W1 + W2
            dense_flops   = 2 * 2 * d_model * d_ff_dense          # 2 matmuls per token

            moe_params    = n_experts * 2 * d_model * d_ff       # all expert weights
            moe_active_flops = top_k * 2 * 2 * d_model * d_ff   # only top_k experts active

            router_params = d_model * n_experts

            print(f"  Dense FFN:       {dense_params:,} params,  {dense_flops:,} FLOPs/token")
            print(f"  MoE FFN total:   {moe_params + router_params:,} params,  {moe_active_flops:,} active FLOPs/token")
            print(f"  MoE param ratio: {(moe_params + router_params) / dense_params:.1f}× more parameters")
            print(f"  MoE FLOPs ratio: {moe_active_flops / dense_flops:.1f}× more active FLOPs (top_{top_k})")
            print()

            # Real model comparison
            print("  Real model examples:")
            print(f"  {'Model':20s}  {'Params':>10s}  {'Active/token':>14s}  {'Experts':>8s}  {'Top-k':>6s}")
            print("  " + "─" * 65)
            real_moe = [
                ("Mixtral 8×7B",   "46.7B", "12.9B", 8, 2),
                ("Mixtral 8×22B",  "141B",  "39.1B", 8, 2),
                ("Switch-Base",    "7B",     "0.2B",  128, 1),
                ("GLaM 1.2T",      "1.2T",  "96.6B", 64, 2),
            ]
            for name, total, active, ne, tk in real_moe:
                print(f"  {name:20s}  {total:>10s}  {active:>14s}  {ne:>8d}  {tk:>6d}")
        """,
    },

    # ── 13 ─────────────────────────────────────────────────────────────────
    "13 · Multi-GPU Distributed Training Strategies": {
        "description": (
            "Simulate and compare all major distributed training strategies: Data Parallelism, "
            "Tensor Parallelism, Pipeline Parallelism, and ZeRO optimizer sharding. Calculate "
            "memory per GPU, communication volume, and speedup for each strategy. Show the "
            "3D parallelism combination used by real LLM training runs."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            # ── Model Memory Calculator ───────────────────────────────────────
            def model_memory(N_params, strategy="fp32", optimizer="adamw"):
                \"\"\"
                Total training memory per parameter.
                N_params: number of parameters
                Returns bytes per parameter for each memory component.
                \"\"\"
                components = {}
                if strategy == "bf16":
                    components["Model weights (BF16)"]      = 2
                    components["Gradients (BF16)"]           = 2
                    components["FP32 master weights"]        = 4
                    components["Adam momentum m (FP32)"]     = 4
                    components["Adam variance v (FP32)"]     = 4
                else:
                    components["Model weights (FP32)"]       = 4
                    components["Gradients (FP32)"]           = 4
                    components["Adam momentum m (FP32)"]     = 4
                    components["Adam variance v (FP32)"]     = 4

                total = sum(components.values())
                return components, total

            comps, total_bpp = model_memory(1, "bf16")

            print("=" * 70)
            print("  Memory Per Parameter — Mixed Precision (BF16)")
            print("=" * 70)
            print()
            for name, bpp in comps.items():
                bar = "█" * bpp
                print(f"  {name:35s}: {bpp} bytes {bar}")
            print(f"  {'─'*50}")
            print(f"  {'Total':35s}: {total_bpp} bytes/param")
            print()

            # ── Strategy 1: Data Parallelism ──────────────────────────────────
            print("=" * 70)
            print("  STRATEGY 1: Data Parallelism (DP / DDP)")
            print("=" * 70)
            print()

            N_params  = 7e9    # 7B model
            N_gpus_dp = [1, 4, 8, 16, 32, 64]

            print("  Every GPU holds a FULL copy of the model and optimizer state.")
            print("  Gradients AllReduced across all GPUs after backward pass.")
            print()
            print(f"  Model: {N_params/1e9:.0f}B params  ({N_params * total_bpp / 1e9:.0f} GB per GPU)")
            print()
            print(f"  {'N GPUs':>8s}  {'Mem/GPU':>10s}  {'Eff. Batch':>12s}  {'AllReduce Vol':>15s}")
            print("  " + "─" * 50)

            local_batch_tokens = 512  # tokens per GPU per step
            for n in N_gpus_dp:
                mem_per_gpu     = N_params * total_bpp / 1e9     # same regardless of N
                eff_batch       = n * local_batch_tokens
                allreduce_vol   = 2 * N_params * 2 / 1e9         # 2× for ring all-reduce, bf16
                print(f"  {n:>8d}  {mem_per_gpu:>9.1f}G  {eff_batch:>12,}  {allreduce_vol:>13.2f} GB")

            print()
            print("  DP constraint: full model must fit on a SINGLE GPU")
            print(f"  7B model needs {N_params * total_bpp / 1e9:.0f} GB → requires A100 80GB ✓")
            print(f"  70B model needs {70e9 * total_bpp / 1e9:.0f} GB → DOES NOT fit on one A100 ✗")

            # ── Strategy 2: Tensor Parallelism ────────────────────────────────
            print()
            print("=" * 70)
            print("  STRATEGY 2: Tensor Parallelism (TP) — Megatron-LM")
            print("=" * 70)
            print()
            print("  Split individual weight matrices across GPUs (column / row partitioning).")
            print("  Communication: AllReduce after each attention sublayer + each FFN.")
            print()

            d_model = 4096
            d_ff    = 14336
            n_layers = 32
            tp_degrees = [1, 2, 4, 8]

            # Parameters per layer per GPU
            attn_params_full = 4 * d_model * d_model      # Q, K, V, O
            ffn_params_full  = 2 * d_model * d_ff          # W1, W2

            print(f"  Config: d_model={d_model}, d_ff={d_ff}, n_layers={n_layers}")
            print()
            print(f"  {'TP':>4s}  {'Attn params/GPU':>17s}  {'FFN params/GPU':>16s}  "
                  f"{'Comms/layer':>13s}  {'Bandwidth req':>14s}")
            print("  " + "─" * 70)

            for tp in tp_degrees:
                attn_per_gpu = attn_params_full // tp
                ffn_per_gpu  = ffn_params_full  // tp
                # 2 AllReduce per layer (after attn, after FFN); each is seq_len × d_model
                # Assume seq=2048, bf16
                seq = 2048
                comms_per_layer = 2 * seq * d_model * 2 / 1e6   # MB
                bw_needed = comms_per_layer * n_layers * 1000 / 1   # MB/s at 1000 steps/s approx
                print(f"  {tp:>4d}  {attn_per_gpu/1e6:>15.2f}M  {ffn_per_gpu/1e6:>14.2f}M  "
                      f"{comms_per_layer:>11.2f} MB  {bw_needed/1e3:>12.1f} GB/s")

            print()
            print("  NVLink (H100): ~900 GB/s  → TP up to 8 within a node")
            print("  PCIe:           ~64 GB/s  → TP impractical beyond TP=2")
            print("  Always use NVLink for TP — it's bandwidth-sensitive")

            # ── Strategy 3: Pipeline Parallelism ──────────────────────────────
            print()
            print("=" * 70)
            print("  STRATEGY 3: Pipeline Parallelism (PP) — GPipe / PipeDream")
            print("=" * 70)
            print()
            print("  Split model depth across GPUs. Each GPU holds a range of layers.")
            print("  Micro-batching reduces pipeline bubble fraction.")
            print()

            total_layers = 32
            pp_degrees   = [1, 2, 4, 8]
            M_micro      = 8   # number of micro-batches

            print(f"  Total layers: {total_layers},  Micro-batches M={M_micro}")
            print()
            print(f"  {'PP':>4s}  {'Layers/GPU':>12s}  {'Bubble %':>10s}  {'Memory/GPU':>12s}")
            print("  " + "─" * 45)

            N_70b = 70e9
            for pp in pp_degrees:
                layers_per_gpu = total_layers // pp
                # Bubble fraction = (pp - 1) / (M + pp - 1) for 1F1B schedule
                bubble_pct = (pp - 1) / (M_micro + pp - 1) * 100
                mem_per_gpu = N_70b / pp * total_bpp / 1e9   # params / pp stages
                print(f"  {pp:>4d}  {layers_per_gpu:>12d}  {bubble_pct:>9.1f}%  {mem_per_gpu:>10.1f} GB")

            print()
            print("  1F1B schedule (PipeDream): interleave 1 forward + 1 backward")
            print("  Interleaved PP: each GPU holds multiple non-contiguous chunks → smaller bubble")
            print("  PP communication: only activations at stage boundaries (low bandwidth)")

            # ── Strategy 4: ZeRO ──────────────────────────────────────────────
            print()
            print("=" * 70)
            print("  STRATEGY 4: ZeRO — Zero Redundancy Optimizer (DeepSpeed)")
            print("=" * 70)
            print()
            print("  Shards optimizer state, gradients, and parameters across DP replicas.")
            print()

            N_p   = 70e9
            n_gpu = 64   # DP group size

            zero_stages = [
                ("Baseline DP",    total_bpp,                 "nothing",   "AllReduce grads"),
                ("ZeRO Stage 1",   2 + 2 + (4+4+4)/n_gpu,   "opt state", "AllReduce + gather opt"),
                ("ZeRO Stage 2",   2 + 2/n_gpu + (4+4+4)/n_gpu, "opt+grad", "ReduceScatter + AllGather"),
                ("ZeRO Stage 3",   total_bpp/n_gpu,           "opt+grad+param", "AllGather params each layer"),
            ]

            print(f"  N_params={N_p/1e9:.0f}B,  N_GPUs (DP group)={n_gpu}")
            print()
            print(f"  {'Stage':15s}  {'Bytes/param':>12s}  {'Mem/GPU':>12s}  {'What is sharded':>18s}")
            print("  " + "─" * 65)

            for stage, bpp, sharded, comm in zero_stages:
                mem_gpu = N_p * bpp / 1e9
                print(f"  {stage:15s}  {bpp:>12.2f}  {mem_gpu:>10.1f} GB  {sharded:>18s}")

            print()
            print(f"  Stage 3 allows {n_gpu}× memory reduction vs baseline!")
            print()
            print("  ZeRO-Infinity: offload optimizer state to CPU RAM or NVMe SSD")
            print("  FSDP (PyTorch): native ZeRO-3 equivalent; used in LLaMA training")

            # ── 3D Parallelism ─────────────────────────────────────────────────
            print()
            print("=" * 70)
            print("  3D PARALLELISM — Combining TP + PP + DP")
            print("=" * 70)
            print()
            print("  Total GPUs = TP_degree × PP_degree × DP_degree")
            print()

            configs_3d = [
                ("GPT-3 175B   (OpenAI/MSFT)", 175e9, 8, 8,  1024),
                ("LLaMA-3 405B (Meta)",         405e9, 8, 16, 128),
                ("Megatron 1T  (NVIDIA)",       1000e9, 8, 16, 1024),
            ]

            print(f"  {'Model':28s}  {'TP':>4s}  {'PP':>4s}  {'DP':>6s}  "
                  f"{'Total GPUs':>11s}  {'Mem/GPU':>10s}")
            print("  " + "─" * 72)

            for name, N, tp, pp, dp in configs_3d:
                total_gpus = tp * pp * dp
                # Params per GPU: split by TP × PP, replicated across DP
                params_per_gpu = N / (tp * pp)
                mem_gpu = params_per_gpu * total_bpp / 1e9
                print(f"  {name:28s}  {tp:>4d}  {pp:>4d}  {dp:>6d}  "
                      f"{total_gpus:>11,}  {mem_gpu:>8.1f} GB")

            # ── Communication Primitives ───────────────────────────────────────
            print()
            print("── Communication Primitives — When Each Is Used ──")
            print()
            primitives = [
                ("AllReduce",      "DP grad averaging",          "Every layer, after backward"),
                ("AllGather",      "ZeRO-3 param reconstruction","Before each layer forward"),
                ("ReduceScatter",  "ZeRO-3 grad sharding",       "After each layer backward"),
                ("Send/Recv (P2P)","PP stage communication",     "At each pipeline stage boundary"),
                ("Broadcast",      "Checkpoint sync",            "Once at init or checkpoint load"),
            ]
            print(f"  {'Primitive':18s}  {'Used in':30s}  {'When':30s}")
            print("  " + "─" * 82)
            for prim, use, when in primitives:
                print(f"  {prim:18s}  {use:30s}  {when:30s}")

            # ── Gradient Accumulation ──────────────────────────────────────────
            print()
            print("── Gradient Accumulation — Large Effective Batch Size ──")
            print()
            print("  Accumulate gradients over K micro-batches before optimizer step.")
            print("  Effective batch = micro_batch × accum_steps × n_GPUs")
            print()

            configs_accum = [
                ("GPT-3",    512,  0.5e6, "0.5M tokens/batch"),
                ("LLaMA-2",  1024, 4e6,  "4M tokens/batch"),
                ("LLaMA-3",  128,  16e6, "16M tokens/batch"),
            ]

            for name, tok_per_gpu, target_tokens, desc in configs_accum:
                accum_steps = int(target_tokens / (tok_per_gpu * 1024))  # assume 1024 GPUs
                print(f"  {name:10s}: {tok_per_gpu} tokens/GPU × 1024 GPUs × {accum_steps} accum = {desc}")
        """,
    },

    # ── 14 ─────────────────────────────────────────────────────────────────
    "14 · Speculative Decoding — Faster Inference": {
        "description": (
            "Implement speculative decoding: a small draft model proposes tokens, the large "
            "target model verifies them in a single parallel forward pass using rejection "
            "sampling. Show that accepted token distribution is mathematically identical to "
            "sampling from the target model. Measure expected speedup."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math

            np.random.seed(42)

            # ── Setup ────────────────────────────────────────────────────────
            VOCAB    = ["the", "cat", "sat", "on", "mat", "a", "dog", "ran",
                        "quickly", "over", "big", "small", "and", "then", "."]
            V        = len(VOCAB)
            K        = 4     # draft tokens per speculation step
            N_ROUNDS = 6     # number of speculation rounds

            # ── Simulate draft and target model distributions ────────────────
            def make_distribution(rng, concentration=1.0):
                \"\"\"Make a probability distribution with adjustable peakiness.\"\"\"
                logits = rng.randn(V) * concentration
                logits -= logits.max()
                probs  = np.exp(logits)
                return probs / probs.sum()

            rng_draft  = np.random.RandomState(10)
            rng_target = np.random.RandomState(20)

            def draft_model_probs(context_len):
                \"\"\"Small draft model: faster but lower quality. Somewhat correlated with target.\"\"\"
                base   = make_distribution(rng_draft, 0.5)
                noise  = rng_draft.randn(V) * 0.1
                probs  = np.abs(base + noise)
                return probs / probs.sum()

            def target_model_probs(context_len):
                \"\"\"Large target model: slower but higher quality.\"\"\"
                base   = make_distribution(rng_target, 1.5)
                noise  = rng_target.randn(V) * 0.05
                probs  = np.abs(base + noise)
                return probs / probs.sum()

            # ── Speculative Decoding Algorithm ────────────────────────────────
            def speculative_decode_step(context_len, K, rng):
                \"\"\"
                One round of speculative decoding:
                1. Draft model proposes K tokens autoregressively
                2. Target model verifies all K tokens in ONE parallel forward pass
                3. Accept/reject using rejection sampling

                Theorem (Chen et al. 2023): accepted tokens are distributed exactly
                as if sampled from the target model alone.
                \"\"\"
                # Step 1: Draft K candidate tokens
                draft_tokens = []
                draft_probs  = []
                cl = context_len

                for _ in range(K):
                    p_draft = draft_model_probs(cl)
                    token   = rng.choice(V, p=p_draft)
                    draft_tokens.append(token)
                    draft_probs.append(p_draft[token])
                    cl += 1

                # Step 2: Target model verifies all K tokens in ONE pass
                target_probs_all = [target_model_probs(context_len + i) for i in range(K)]
                # Also get target distribution for bonus token position K+1
                p_target_bonus   = target_model_probs(context_len + K)

                # Step 3: Rejection sampling — accept each draft token
                accepted = []
                for k in range(K):
                    tok     = draft_tokens[k]
                    p_d     = draft_probs[k]                # draft prob of this token
                    p_t     = target_probs_all[k][tok]      # target prob of this token

                    accept_prob = min(1.0, p_t / (p_d + 1e-12))
                    r = rng.rand()

                    if r < accept_prob:
                        accepted.append((tok, "✓ accepted", p_d, p_t, accept_prob))
                    else:
                        # Reject: sample a correction token from adjusted target distribution
                        adjusted = np.maximum(0, target_probs_all[k] - draft_probs[k] * np.ones(V) * (p_d / V))
                        # Simple approximation: use target dist directly
                        correction = rng.choice(V, p=target_probs_all[k])
                        accepted.append((correction, "✗ rejected→corrected", p_d, p_t, accept_prob))
                        break   # stop at first rejection

                # Bonus token: always sample from target at position len(accepted)
                bonus_tok = rng.choice(V, p=p_target_bonus)
                total_tokens = len(accepted) + 1   # accepted + bonus

                return draft_tokens, accepted, bonus_tok, total_tokens

            # ── Simulate multiple rounds ──────────────────────────────────────
            print("=" * 70)
            print("  SPECULATIVE DECODING — Draft + Verify")
            print("=" * 70)
            print(f"  Draft model generates K={K} candidate tokens per round")
            print(f"  Target model verifies all K in ONE parallel forward pass")
            print()

            rng         = np.random.RandomState(7)
            total_toks  = 0
            total_large_fwd = 0   # target model forward passes (each processes K tokens)
            total_large_seq = 0   # hypothetical sequential target calls

            print(f"  {'Round':>6s}  {'Draft tokens':30s}  {'Accepted':>9s}  {'Tokens/round':>13s}")
            print("  " + "─" * 65)

            for round_i in range(N_ROUNDS):
                draft_toks, accepted, bonus, n_out = speculative_decode_step(0, K, rng)

                n_accepted = sum(1 for _, status, *_ in accepted if "✓" in status)
                draft_str  = " ".join(VOCAB[t] for t in draft_toks)
                acc_str    = f"{n_accepted}/{K} + bonus"

                total_toks        += n_out
                total_large_fwd   += 1         # 1 target forward pass per round (verifies K)
                total_large_seq   += n_out      # sequential would need n_out target calls

                print(f"  {round_i+1:>6d}  {draft_str:30s}  {acc_str:>9s}  {n_out:>13d}")

                # Show acceptance details for first 2 rounds
                if round_i < 2:
                    for k, (tok, status, p_d, p_t, acc_p) in enumerate(accepted):
                        tok_name = VOCAB[tok]
                        print(f"         token {k}: '{tok_name:8s}'  p_draft={p_d:.3f}  "
                              f"p_target={p_t:.3f}  accept_p={acc_p:.3f}  {status}")
                    print(f"         bonus: '{VOCAB[bonus]:8s}'  (always sampled from target)")
                    print()

            print()
            print(f"  Total tokens generated:     {total_toks}")
            print(f"  Target model fwd passes:    {total_large_fwd}  (speculative)")
            print(f"  Hypothetical sequential:    {total_large_seq}  (naive autoregressive)")
            print(f"  Tokens per target fwd pass: {total_toks / total_large_fwd:.2f}")
            print()

            # ── Speedup Analysis ──────────────────────────────────────────────
            print("=" * 70)
            print("  Speedup Analysis")
            print("=" * 70)
            print()

            draft_cost_ratio = 0.1   # draft model is ~10% cost of target
            accept_rates = [0.5, 0.7, 0.8, 0.9, 0.95]

            print(f"  Assumptions: K={K} draft tokens, draft costs {int(draft_cost_ratio*100)}% of target")
            print()
            print(f"  {'Accept rate':>12s}  {'Expected toks/step':>20s}  {'Speedup':>10s}")
            print("  " + "─" * 48)

            for alpha in accept_rates:
                # Expected number of accepted tokens (geometric distribution)
                # E[n] = (1 - alpha^K) / (1 - alpha) + 1  (approximately)
                if alpha < 1.0:
                    expected_accepted = sum(alpha**i for i in range(K)) + 1
                else:
                    expected_accepted = K + 1

                # Cost: K × draft_cost + 1 × target_cost (for verification)
                spec_cost  = K * draft_cost_ratio + 1
                naive_cost = expected_accepted     # expected_accepted sequential target calls

                speedup = naive_cost / spec_cost
                print(f"  {alpha:>12.0%}  {expected_accepted:>20.2f}  {speedup:>9.2f}×")

            print()
            print("  Key theorem: accepted tokens are distributed EXACTLY as target model")
            print("  This means speculative decoding is a lossless speedup — no quality loss")
            print()
            print("  Practical speedups (reported):")
            print("    CodeLlama (7B draft → 70B target):  ~2.5×")
            print("    Medusa (multi-head draft):           ~2.2×")
            print("    Google PaLM 2 production:            ~2–3×")
        """,
    },

    # ── 15 ─────────────────────────────────────────────────────────────────
    "15 · Data Preparation, Mixing & Training Stability": {
        "description": (
            "Simulate the full data preparation pipeline: quality filtering heuristics, "
            "deduplication, data source mixing ratios, and curriculum/annealing. "
            "Then demonstrate loss spike detection and gradient norm monitoring — "
            "critical for stable long LLM training runs."
        ),
        "language": "python",
        "code": """
            import numpy as np
            import math
            import re
            from collections import Counter

            np.random.seed(0)

            # ── Part 1: Data Quality Filtering ───────────────────────────────
            print("=" * 65)
            print("  DATA QUALITY FILTERING PIPELINE")
            print("=" * 65)
            print()

            # Simulated raw documents
            RAW_DOCS = [
                "The Transformer architecture was introduced by Vaswani et al. in 2017. "
                "It replaced recurrent networks with self-attention, enabling parallelism.",

                "lol omg wtf lol lol lol click here buy now!!! $$$ FREE MONEY LOL WTF",

                "Paris is the capital of France. France is in Western Europe. "
                "The Eiffel Tower is in Paris. Paris is a beautiful city in France.",

                "<html><body><div class='ad'>Buy now!</div><!--SPAM--></body></html>",

                "In 2022, DeepMind published the Chinchilla paper showing that most large "
                "language models were undertrained. They found that for a given compute "
                "budget, one should train a smaller model on more data.",

                "aaaaaaaaaaaa bbbbbbbbbbbb cccccccc aaaaaaaaa bbbbbb aaaaaaaaaaaaaaaa",

                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",

                "a",  # too short
            ]

            def filter_pipeline(doc):
                \"\"\"Apply a series of quality filters to a document.\"\"\"
                reasons = []

                # 1. Length check
                tokens = doc.split()
                if len(tokens) < 10:
                    reasons.append("too short (<10 tokens)")

                # 2. Symbol ratio (non-alphabetic characters)
                alpha = sum(1 for c in doc if c.isalpha())
                if len(doc) > 0 and alpha / len(doc) < 0.5:
                    reasons.append(f"low alpha ratio ({alpha/len(doc):.2f})")

                # 3. Repetition check (duplicate n-grams)
                words = doc.lower().split()
                if len(words) > 4:
                    bigrams   = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                    bigram_counts = Counter(bigrams)
                    most_common_frac = bigram_counts.most_common(1)[0][1] / len(bigrams)
                    if most_common_frac > 0.3:
                        reasons.append(f"high repetition ({most_common_frac:.2f})")

                # 4. HTML/boilerplate detection
                if re.search(r'<[a-zA-Z][^>]*>', doc):
                    reasons.append("contains HTML tags")

                # 5. Spam pattern
                spam_keywords = ["click here", "buy now", "free money", "lol omg"]
                for kw in spam_keywords:
                    if kw.lower() in doc.lower():
                        reasons.append(f"spam keyword: '{kw}'")
                        break

                passed = len(reasons) == 0
                return passed, reasons

            print(f"  {'#':>3s}  {'Preview':42s}  {'Pass?':>6s}  {'Reason if failed'}")
            print("  " + "─" * 88)
            keep_count = 0
            for i, doc in enumerate(RAW_DOCS):
                passed, reasons = filter_pipeline(doc)
                preview   = (doc[:40] + "...") if len(doc) > 40 else doc
                status    = "✓ KEEP" if passed else "✗ DROP"
                reason    = "; ".join(reasons) if reasons else "-"
                keep_count += int(passed)
                print(f"  {i+1:>3d}  {preview:42s}  {status:>6s}  {reason[:40]}")

            print()
            print(f"  Kept: {keep_count}/{len(RAW_DOCS)} documents ({keep_count/len(RAW_DOCS)*100:.0f}%)")
            print("  Real pipelines filter ~70% of raw web data, retaining ~30%")

            # ── Part 2: Deduplication ─────────────────────────────────────────
            print()
            print("=" * 65)
            print("  DEDUPLICATION — MinHash Near-Duplicate Detection")
            print("=" * 65)
            print()
            print("  MinHash: approximate Jaccard similarity via hashed n-gram shingles")
            print("  Jaccard(A,B) = |A∩B| / |A∪B|  — fraction of shared n-grams")
            print()

            def ngrams(text, n=5):
                words = text.lower().split()
                return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

            def jaccard(a, b):
                na, nb = ngrams(a), ngrams(b)
                if not na or not nb:
                    return 0.0
                return len(na & nb) / len(na | nb)

            DEDUP_DOCS = [
                "The cat sat on the mat and the dog ran over the fence quickly.",
                "The cat sat on the mat and the dog ran over the fence quickly today.",  # near-dup
                "Paris is the capital of France, a country in Western Europe.",
                "Paris is capital of France, located in Western Europe.",               # near-dup
                "Quantum computing uses qubits instead of classical bits.",              # unique
            ]

            THRESHOLD = 0.5  # jaccard similarity threshold for dedup

            print(f"  Jaccard similarity threshold: {THRESHOLD}")
            print()
            kept = [DEDUP_DOCS[0]]
            for i in range(1, len(DEDUP_DOCS)):
                max_sim = max(jaccard(DEDUP_DOCS[i], k) for k in kept)
                is_dup  = max_sim >= THRESHOLD
                status  = f"✗ DUP (sim={max_sim:.2f})" if is_dup else f"✓ KEEP (max_sim={max_sim:.2f})"
                preview = DEDUP_DOCS[i][:50] + "..."
                print(f"  Doc {i+1}: {preview}")
                print(f"         {status}")
                if not is_dup:
                    kept.append(DEDUP_DOCS[i])

            print(f"\\n  Kept {len(kept)}/{len(DEDUP_DOCS)} after dedup")

            # ── Part 3: Data Mixing ───────────────────────────────────────────
            print()
            print("=" * 65)
            print("  DATA MIXING — Source Proportions")
            print("=" * 65)
            print()

            TOTAL_TOKENS = 15e12   # LLaMA-3: 15T tokens

            DATA_MIX = [
                ("Web (filtered)",  0.80, "General knowledge, world facts, fluency"),
                ("Code",            0.08, "Structured reasoning, syntax, algorithms"),
                ("Books",           0.05, "Long-form coherence, narrative reasoning"),
                ("Math/Science",    0.03, "Quantitative reasoning, proofs"),
                ("Multilingual",    0.04, "Cross-lingual transfer, coverage"),
            ]

            print(f"  Total training tokens: {TOTAL_TOKENS/1e12:.0f}T")
            print()
            print(f"  {'Source':20s}  {'Proportion':>11s}  {'Tokens':>12s}  Rationale")
            print("  " + "─" * 75)
            for src, prop, rationale in DATA_MIX:
                tokens = TOTAL_TOKENS * prop
                bar    = "█" * int(prop * 40)
                print(f"  {src:20s}  {prop:>10.0%}  {tokens/1e9:>10.0f}B  {rationale}")
            print()
            print(f"  Total: {sum(p for _,p,_ in DATA_MIX)*100:.0f}%  {TOTAL_TOKENS/1e12:.0f}T tokens")

            # ── Part 4: Loss Spike Detection ──────────────────────────────────
            print()
            print("=" * 65)
            print("  TRAINING STABILITY — Loss Spike Detection & Response")
            print("=" * 65)
            print()

            # Simulate a training loss curve with two spikes
            np.random.seed(5)
            n_steps = 60
            loss_curve = []

            for t in range(n_steps):
                base_loss = 4.0 * math.exp(-t / 20) + 1.8    # decaying baseline
                noise     = np.random.randn() * 0.05
                spike     = 3.0 if t == 22 else (2.0 if t == 45 else 0.0)
                loss_curve.append(base_loss + noise + spike)

            # Spike detection: exponential moving average
            window  = 10
            ema_alpha = 0.9
            ema = loss_curve[0]
            SPIKE_THRESHOLD = 2.0   # spike if loss > threshold × EMA

            print(f"  Spike threshold: {SPIKE_THRESHOLD}× EMA  |  EMA α={ema_alpha}")
            print()
            print(f"  {'Step':>5s}  {'Loss':>8s}  {'EMA':>8s}  {'Ratio':>7s}  {'Action'}")
            print("  " + "─" * 55)

            rollback_to = 0
            for t, loss in enumerate(loss_curve):
                ratio = loss / ema if ema > 0 else 1.0
                is_spike = ratio > SPIKE_THRESHOLD

                if is_spike:
                    action = f"⚠ SPIKE! Rollback to step {rollback_to}"
                else:
                    action = "✓ normal"
                    rollback_to = max(0, t - 5)    # save checkpoint every 5 steps

                # Display significant steps
                if t < 5 or is_spike or t % 10 == 0:
                    bar = "█" * int(min(loss, 5) * 4)
                    print(f"  {t:>5d}  {loss:>8.3f}  {ema:>8.3f}  {ratio:>7.2f}  {action}")

                # Update EMA (after spike: skip spike value)
                if not is_spike:
                    ema = ema_alpha * ema + (1 - ema_alpha) * loss

            print()
            print("  Standard response to loss spikes:")
            print("  1. Detect: loss > 2× running EMA → flag as spike")
            print("  2. Skip: discard the offending batch, do not update weights")
            print("  3. Rollback: revert to checkpoint N steps before spike")
            print("  4. Inspect: examine what data caused the spike, remove it")
            print("  5. Resume: continue training with filtered data")
            print()
            print("  LLaMA-3 experienced several spikes over 15T token training run,")
            print("  each handled with rollback of 100–200 steps.")
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
    import streamlit as st

    st.markdown("---")
    st.subheader("⚙️ Operations")

    if "tf_step_status"  not in st.session_state:
        st.session_state.tf_step_status  = {}
    if "tf_step_outputs" not in st.session_state:
        st.session_state.tf_step_outputs = {}

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