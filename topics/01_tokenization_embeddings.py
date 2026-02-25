"""Module 01: Tokenization & Embeddings"""
from Required_Images.tokenization_visual import get_visual_html

import os
import re
import sys
import subprocess
from pathlib import Path
import base64
import textwrap

TOPIC_NAME   = "Tokenization & Embeddings — Detailed Breakdown"
DISPLAY_NAME = "01 · Tokenization & Embeddings"
ICON         = "🔤"
SUBTITLE     = "How raw text becomes numbers — the bedrock of every language model."

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / "Implementation" / "Tokenization_Implementation" / "scripts"
_MAIN_SCRIPT  = _SCRIPTS_DIR / "tokenization_main.py"

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
Before any neural network can process language, text must be converted into numbers.
This module covers the complete pipeline: from splitting raw text into tokens, to
representing those tokens as high-dimensional vectors that encode semantic meaning.

## 1. Tokenization
Tokenization is the process of splitting raw text into discrete units called *tokens*.

### Character Tokenization
Split text into individual characters. Simple but produces very long sequences.

### Word Tokenization
Split on whitespace/punctuation. Vocabulary explosion problem — millions of unique words.

### Subword Tokenization *(The Modern Standard)*
The sweet spot — handles unknown words without a massive vocabulary.

#### Byte Pair Encoding (BPE)
- Start with a character-level vocabulary
- Iteratively merge the most frequent adjacent pair of tokens
- Used by: GPT-2, GPT-3, GPT-4, LLaMA, Mistral

#### WordPiece
- Similar to BPE but merges based on likelihood instead of frequency
- Used by: BERT, DistilBERT, ALBERT

#### Unigram Language Model
- Starts with a large vocabulary, prunes tokens that least affect likelihood
- Used by: T5, ALBERT, SentencePiece

#### SentencePiece
- Language-agnostic tokenizer that treats raw text as a stream of Unicode chars
- No whitespace assumption — works well for Chinese, Japanese, etc.

## 2. Special Tokens
Every tokenizer reserves special tokens for model control:
- `[CLS]` — Classification token (BERT)
- `[SEP]` — Separator (BERT)
- `<s>` / `</s>` — Start/End of sequence
- `[PAD]` — Padding to equal length in a batch
- `[UNK]` — Unknown token (rare in subword tokenizers)
- `[MASK]` — Masked token for MLM training

## 3. Static Word Embeddings

### Word2Vec (2013)
- **Skip-gram**: Given a center word, predict surrounding context words
- **CBOW**: Given context words, predict the center word
- Result: 300-dimensional dense vectors where similar words cluster together
- Famous property: `king - man + woman ≈ queen`

### GloVe (Global Vectors)
- Combines global co-occurrence statistics with local context
- Train on the co-occurrence matrix of the whole corpus
- Captures both local context (like Word2Vec) and global statistics

### FastText
- Extends Word2Vec by representing words as bags of character n-grams
- Handles OOV (out-of-vocabulary) words naturally
- Better for morphologically rich languages

## 4. Contextual Embeddings
Static embeddings give the same vector for "bank" (financial) and "bank" (river).
Contextual embeddings produce different vectors based on surrounding context.

This requires a full model (Transformer) — covered in Module 03.

## 5. Positional Encoding
Transformers process tokens in parallel (no inherent order), so position must be
injected explicitly.

### Sinusoidal (Original Transformer)
$$PE_{(pos, 2i)}   =  \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
$$PE_{(pos, 2i+1)} =  \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$

### Learned Positional Embeddings
Position vectors trained alongside the model. Used by BERT, GPT-2.

### Rotary Positional Encoding (RoPE)
- Encodes position as a rotation in embedding space
- Relative positions naturally captured
- Used by: LLaMA, Mistral, GPT-NeoX

### ALiBi (Attention with Linear Biases)
- Add a linear bias to attention scores based on token distance
- Better extrapolation to longer sequences than trained on
- Used by: MPT, BLOOM


### Tokenization & Embeddings — Detailed Breakdown


                                        TOKENIZATION & EMBEDDING LANDSCAPE — FULL VIEW
    
    ══════════════════════════════════════════════════════════════════════════════════════════════════════════
    
                                        ┌──────────────────────────────────┐
                                        │     RAW TEXT  →  NUMBERS         │
                                        │  (The complete pipeline every    │
                                        │   LLM uses before it can think)  │
                                        └──────────────────┬───────────────┘
                                                           │
                      ┌────────────────────────────────────┼────────────────────────────────────┐
                      │                                    │                                    │
                      ▼                                    ▼                                    ▼
        ┌─────────────────────────────┐    ┌──────────────────────────────┐    ┌──────────────────────────────┐
        │       TOKENIZATION          │    │        EMBEDDINGS            │    │    POSITIONAL ENCODING       │
        │  (text → integer IDs)       │    │  (IDs → dense vectors)       │    │  (inject sequence order)     │
        │                             │    │                              │    │                              │
        │  Split text into units and  │    │  Map each token to a point   │    │  Transformers have no        │
        │  map each to an integer ID  │    │  in high-dimensional space   │    │  inherent word order —       │
        │  from a fixed vocabulary    │    │  where meaning is geometry   │    │  position must be added      │
        └──────────────┬──────────────┘    └──────────────┬───────────────┘    └──────────────┬───────────────┘
                       │                                  │                                   │
         ┌─────────────┼─────────────┐          ┌─────────┼──────────┐             ┌──────────┼──────────┐
         ▼             ▼             ▼          ▼         ▼          ▼             ▼          ▼          ▼
    ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌──────┐  ┌──────┐   ┌─────────┐ ┌───────┐ ┌────────┐
    │Character│  │  Word    │  │ Subword  │  │Static│  │Contex│  │ BOS/ │   │Sinusoid-│ │Learned│ │ RoPE   │
    │         │  │          │  │          │  │      │  │ tual │  │ EOS  │   │  al     │ │       │ │        │
    │Simple,  │  │Whitespace│  │The modern│  │Word2-│  │(BERT,│  │ and  │   │Fixed    │ │Trained│ │Rotation│
    │long seqs│  │/punct.   │  │standard  │  │Vec,  │  │GPT,  │  │model │   │math     │ │with   │ │in emb  │
    │no OOV   │  │vocab     │  │(BPE,WP,  │  │GloVe,│  │LLaMA)│  │spec. │   │formula  │ │model  │ │space   │
    │issue    │  │explosion │  │Unigram)  │  │FText │  │      │  │tokens│   │         │ │       │ │        │
    └─────────┘  └──────────┘  └────┬─────┘  └──────┘  └──────┘  └──────┘   └─────────┘ └───────┘ └───┬────┘
                                    │                                                                 │
              ┌─────────────────────┼───────────────────────┐                                  ALiBi  │
              ▼                     ▼                       ▼                                ┌────────┘
        ┌──────────┐          ┌──────────┐          ┌──────────────┐
        │   BPE    │          │WordPiece │          │   Unigram /  │
        │          │          │          │          │ SentencePiece│
        │Merge most│          │Merge by  │          │              │
        │frequent  │          │likelihood│          │Start large,  │
        │char pairs│          │(not freq)│          │prune by      │
        │          │          │          │          │likelihood    │
        │GPT-2/3/4 │          │BERT,     │          │T5, ALBERT,   │
        │LLaMA,    │          │DistilBERT│          │multilingual  │
        │Mistral   │          │          │          │              │
        └──────────┘          └──────────┘          └──────────────┘

    ══════════════════════════════════════════════════════════════════════════════════════════════════════════

---

### Why This Pipeline Exists — The Core Problem

Neural networks can only process numbers, not text.
But you can't just assign word → number arbitrarily (that would impose a false numeric ordering).
The solution is a two-stage approach that every modern LLM uses:

    Stage 1: Tokenization  — text → integer IDs (a lookup index, not a meaningful number)
    Stage 2: Embedding     — integer ID → dense vector (a meaningful point in space)

These are fundamentally different operations done at different times for different reasons.

---

## 1. Tokenization — Text to Integer IDs

### The Core Question: What Is a "Token"?

A token is the atomic unit your model processes. The choice of *how* to tokenize
determines vocabulary size, sequence length, and how the model handles rare or unknown words.

### What Is a Vocabulary?

Before diving in, it helps to have a firm definition of "vocabulary" — a word used
constantly in this module.

A **vocabulary** is a fixed lookup dictionary built once during tokenizer training.
It maps every possible token (character, word, or subword piece) to a unique integer ID:

    Vocabulary (simplified example):
    ─────────────────────────────────
    [PAD]       →  0
    [UNK]       →  1
    "the"       →  2
    "cat"       →  3
    "sat"       →  4
    "un"        →  5
    "##happy"   →  6   ← a subword piece
    ...
    "zygote"    →  49,000

The vocabulary is fixed and never changes at inference time. If a piece of text isn't
in the vocabulary, the tokenizer either maps it to [UNK] (word tokenizers) or breaks
it into smaller pieces that are in the vocabulary (subword tokenizers).
Vocabulary *size* (30K, 50K, 128K) is a core engineering tradeoff covered below.

Three tokenization strategies have been tried, in historical order:

---

### A) Character Tokenization

Split text into individual characters. Every possible character is a token.

    "Hello world"  →  ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
                   →  [ 72,  69,  76,  76,  79,  32,  87,  79,  82,  76,  68]

    Pros:
    - Tiny vocabulary (~256 UTF-8 characters, ~70 for English)
    - Zero OOV (out-of-vocabulary) problem — every character is known
    
    Cons:
    - Very long sequences (a sentence of 100 characters = 100 tokens)
    - The model must learn to compose meaning from individual characters
    - Attention scales quadratically with sequence length — this is expensive
    
    Used by: early character-level RNNs. Mostly abandoned for modern LLMs.

---

### B) Word Tokenization

Split on whitespace and punctuation. Each word is a token.

    "The quick brown fox jumps"  →  ["The", "quick", "brown", "fox", "jumps"]
                                 →  [  427,     890,     304,    88,    1042]

    Pros:
    - Short sequences
    - Each token is a complete meaningful unit
    
    Cons:
    - Vocabulary explosion: English has ~170,000 words, but with names, technical
      terms, conjugations — easily 1M+ unique tokens in a real corpus
    - OOV problem: "unfathomable" not in vocab → [UNK] (you lose all information)
    - "run", "runs", "running", "runner" are completely unrelated tokens
    
    Used by: early NLP systems. Also largely abandoned.

---

### C) Subword Tokenization — The Modern Standard

The key insight: common words get their own token; rare words are broken into
meaningful pieces. You never need [UNK]. Vocabulary stays manageable.

    "tokenization"  →  ["token", "ization"]           ← two pieces, both meaningful
    "anthropic"     →  ["anthrop", "ic"]
    "unfathomable"  →  ["un", "fath", "omable"]       ← broken into recognizable parts
    "cat"           →  ["cat"]                        ← common word, own token
    "cats"          →  ["cat", "s"]                   ← morpheme-aware

Typical vocabulary sizes in production models:

    Model           Tokenizer           Vocab Size
    ──────────────────────────────────────────────
    GPT-2           BPE                     50,257
    GPT-4           BPE (cl100k)           100,256
    LLaMA-2         BPE (SentPiece)         32,000
    LLaMA-3         BPE (tiktoken)         128,256
    BERT            WordPiece               30,522
    T5              Unigram (SP)            32,100
    Mistral-7B      BPE (SentPiece)         32,000

---

#### BPE (Byte Pair Encoding) — Step-by-Step

BPE is the algorithm behind GPT-2, GPT-4, LLaMA, Mistral, and most modern LLMs.
It builds a vocabulary from the ground up by learning which character combinations
appear most frequently in your training corpus.

**Algorithm:**

    Step 0: Start — every character is its own token
    Step 1: Count — count every adjacent pair of tokens in the corpus
    Step 2: Merge — merge the most frequent pair into a single new token
    Step 3: Repeat — keep merging until you reach your target vocabulary size

**Concrete traced example:**

Corpus:  ["low", "low", "lower", "newer", "newest", "widest"]
After adding end-of-word marker </w>:

    Initial token counts:
    l o w </w>       → 2    (from "low" × 2)
    l o w e r </w>   → 1    (from "lower")
    n e w e r </w>   → 1    (from "newer")
    n e w e s t </w> → 1    (from "newest")
    w i d e s t </w> → 1    (from "widest")

    Merge 1: ('e', 's') → 'es'   (most frequent adjacent pair)
    l o w </w>       → 2
    l o w e r </w>   → 1
    n e w e r </w>   → 1
    n e w es t </w>  → 1       ← merged
    w i d es t </w>  → 1       ← merged

    Merge 2: ('es', 't') → 'est'
    n e w est </w>   → 1
    w i d est </w>   → 1

    Merge 3: ('l', 'o') → 'lo'
    lo w </w>        → 2
    lo w e r </w>    → 1

    Merge 4: ('lo', 'w') → 'low'
    low </w>         → 2
    low e r </w>     → 1

    Merge 5: ('n', 'e') → 'ne'
    ne w e r </w>    → 1
    ne w est </w>    → 1

    Merge 6: ('ne', 'w') → 'new'
    new e r </w>     → 1
    new est </w>     → 1

    Merge 7: ('new', 'est') → 'newest'   ← whole word merged!
    newest </w>      → 1

    Merge 8: ('e', 'r') → 'er'
    low er </w>      → 1
    new er </w>      → 1

After 8 merges, the final vocabulary includes:
    Original chars + {'es', 'est', 'lo', 'low', 'ne', 'new', 'newest', 'er', ...}

Real models run thousands of merges to build a vocab of 32K–128K tokens.

---

#### WordPiece (BERT's Algorithm)

WordPiece is conceptually similar to BPE but uses a different merge criterion.
Instead of merging the most *frequent* pair, it merges the pair that maximizes
the training data likelihood — i.e., the pair whose joint probability divided
by the product of individual probabilities is highest.

    Score(A, B) = frequency(AB) / (frequency(A) × frequency(B))

This means WordPiece prefers merges that are *surprising* — pairs that appear
together far more often than you'd expect if they were independent.

A key visual difference: WordPiece marks the *beginning* of a word with no prefix,
and marks continuation pieces with `##`:

    "tokenization"  →  ["token", "##ization"]
    "playing"       →  ["play",  "##ing"]

Used by: BERT, DistilBERT, ALBERT.

---

## 2. Tokenization Quirks — Surprises for Beginners

Real tokenizers behave in ways that surprise almost everyone when they first encounter them.
These aren't bugs — they're logical consequences of how the algorithms work.

**Leading spaces matter:**
In GPT-style BPE, a word at the start of a sentence and the same word mid-sentence
often tokenize to *different* IDs, because the leading space is part of the token:

    "dog"          →  [18031]         ← no leading space
    " dog"          →  [5679]          ← space included, different ID entirely
    "The dog sat"   →  [791, 5679, 7482]   ← "dog" here has the space version

**Numbers are unpredictable:**
Numbers rarely get single tokens. The tokenizer has no concept of arithmetic:

    "100"    →  [1041]              ← one token
    "999"    →  [29929, 29929, 29929]  ← three tokens (one per digit!)
    "2024"   →  [29906, 29900, 29906, 29946]  ← four tokens

This is why LLMs notoriously struggle with arithmetic — the number isn't one unit to the model,
it's a sequence of digit tokens with no inherent numeric meaning.

**Capitalization creates different tokens:**
    "Hello"  →  [15043]
    "hello"  →  [29882, 1032, 417]  ← three tokens instead of one!

**Whitespace and punctuation count:**
    "end."   →  ["end", "."]        ← two tokens
    "end ."  →  ["end", " ."]       ← still two, but the period token is different

The practical takeaway: token count is never simply "word count". A good rule of thumb
is ~1.3–1.5 tokens per word in English. For code, math, or non-English text, token
counts can be much higher, directly affecting cost and context window usage.

---

#### Unigram Language Model (SentencePiece)

Instead of starting small and merging up, Unigram starts with a very large
vocabulary (e.g., all substrings up to length 16) and then *prunes* it.

At each step it removes the tokens whose removal causes the least drop in the
probability of the training corpus, until the target vocabulary size is reached.

SentencePiece is a framework that implements both BPE and Unigram and is
language-agnostic — it treats raw text as a byte stream with no whitespace
assumptions. This makes it ideal for Chinese, Japanese, Arabic, etc.

Used by: LLaMA-1/2, T5, ALBERT, mT5.

---

## 3. Special Tokens — What They Actually Are and When They're Added

Every tokenizer reserves some IDs for control signals, not words.
These vary by model family — this is a common source of confusion:

    Token       ID (example)    Purpose                                 Model Family
    ─────────────────────────────────────────────────────────────────────────────────
    [CLS]       101             Classification token —                  BERT, RoBERTa,
                                prepended to every input                encoder models
                                its final state = sentence rep
                                
    [SEP]       102             Separator between segments              BERT, RoBERTa
                                (question [SEP] context)               encoder models
                                
    [PAD]       0               Padding to equal batch length             Most models
                                meaningless filler              
                                
    [UNK]       100             Unknown token (rare in subword)           Legacy/BERT
    
    [MASK]      103             Masked-out token for MLM                BERT training
                                training objective              
    
    <s>         1               Beginning of sequence                  LLaMA, Mistral
    </s>        2               End of sequence                        LLaMA, Mistral
    
    <|endoftext|> 50256         End of text marker                              GPT-2
    
    <|im_start|>  —             Chat turn start marker                  ChatML format
    <|im_end|>    —             Chat turn end marker              (OpenChat, Mistral)
    
    [INST]      —               Instruction start                        LLaMA-2 chat
    [/INST]     —               Instruction end                          LLaMA-2 chat

Critical distinction for modern LLMs:
[CLS] and [SEP] are BERT-era tokens for encoder models.
If you're working with a decoder-only LLM (GPT, LLaMA, Mistral), these tokens
don't exist and aren't used. Modern decoder LLMs use BOS (beginning of sequence),
EOS (end of sequence), and model-specific instruction markers instead.

---

## 4. The Embedding Table — Turning IDs Into Vectors

After tokenization you have a sequence of integer IDs — just indices.
The embedding layer converts each index into a dense vector.

### What the Embedding Table Looks Like

An embedding table is a matrix of shape [vocab_size × d_model]:

    Vocabulary size:   50,257  (GPT-2)
    Embedding dimension: 768   (GPT-2 base)

### What Does "Embedding Dimension" Mean?

The embedding dimension (d_model) is how many numbers are used to represent each token.
Think of it like coordinates: a location on a map needs 2 numbers (latitude, longitude).
A token in embedding space needs 768, 1024, or 4096 numbers.

You can loosely think of each dimension as capturing some property of the token —
things like "how formal is this word?", "is this a verb or noun?", "what domain does it belong to?".
In practice these dimensions aren't interpretable by humans; the model discovers
the most useful properties automatically during training. The higher the dimension,
the more nuance the model can encode — but the more memory and compute it costs.
    
    Common d_model values:
    GPT-2 base   →   768      (small, fast)
    GPT-2 XL     →  1,600
    LLaMA-2 7B   →  4,096
    LLaMA-3 70B  →  8,192     (large, expressive)
    
    
    Table shape:  [50,257 × 768]  =  ~38.6M floats   =   ~154 MB in float32
                                                     =   ~77 MB in bfloat16

    The table (simplified to vocab=6, d_model=4 for illustration):
    
    Token ID │  dim_0    dim_1    dim_2    dim_3
    ─────────┼──────────────────────────────────
         0   │  0.000    0.000    0.000    0.000    ← [PAD] — all zeros by convention
         1   │  0.234   -0.871    0.512    0.009    ← [CLS] or  <s>
         2   │ -0.102    0.634   -0.891    0.771    ← [SEP] or </s>
         3   │  0.543   -0.211    0.089   -0.432    ← "the"
         4   │ -0.671    0.123    0.556    0.338    ← "cat"
         5   │  0.021    0.788   -0.234   -0.109    ← "sat"

When token ID 4 ("cat") enters the model, the embedding layer looks up row 4
and returns: [-0.671, 0.123, 0.556, 0.338] — a 4-dimensional vector.

In real models this is 768, 1024, 2048, or 4096 dimensions.

### How Embeddings Encode Meaning

After training, semantically similar words occupy nearby regions in this vector space.
The geometry of the space captures relationships:

    Similar words cluster together:
    king   → [0.81, -0.22,  0.57, ...]
    queen  → [0.80, -0.23,  0.54, ...]   ← very close to king
    cat    → [-0.34, 0.71, -0.12, ...]   ← far from king, close to other animals
    
    Analogies emerge as vector arithmetic:
    king - man + woman  ≈  queen
    paris - france + germany  ≈  berlin
    
    This works because the model learns to encode gender, royalty, geography, etc.
    as consistent *directions* in the embedding space.

The embedding table is NOT fixed — it's part of the model and gets updated
during training via backpropagation, just like every other weight matrix.

**Why does this happen — is it programmed in?**
No — it emerges purely from the training objective. The model is never told
"king and queen should be similar." It learns this because king and queen appear
in similar contexts: "the ___ ruled the kingdom", "the ___ wore a crown", etc.
The training loss rewards the model for predicting context words correctly, and
the only way to do that efficiently is to place words that share context near each other
in vector space. Similarity is a *side effect* of learning to predict well, not a goal.

The embedding table is NOT fixed — it's part of the model and gets updated
during training via backpropagation, just like every other weight matrix.

---

## 5. Static vs Contextual Embeddings — A Critical Distinction

### Static Embeddings (Word2Vec, GloVe, FastText)

Each word has exactly ONE vector, regardless of context.

    "I went to the river bank to fish."
    "I deposited money at the bank."
    
    Static embeddings: "bank" → same vector in both sentences.
    The model cannot distinguish financial bank from river bank.

### Word2Vec (2013) — How It Actually Works

Word2Vec trains a shallow neural network with one hidden layer.
The key insight: words that appear in similar contexts have similar meanings.

**Skip-gram objective:**
Given a center word, predict the surrounding context words.

    Sentence: "The quick brown fox jumps"
    Center word: "brown" (position 2)
    Window size: 2
    
    Training pairs:
    ("brown" → "The")     ← 2 positions left
    ("brown" → "quick")   ← 1 position left
    ("brown" → "fox")     ← 1 position right
    ("brown" → "jumps")   ← 2 positions right
    
    The network learns: given the input vector for "brown",
    maximize the probability of outputting these context words.

    After training, the input-side weight matrix becomes the embedding table.
    
    Architecture:
    
    Input (one-hot)    Hidden layer         Output (softmax)
    ┌─────────────┐    ┌───────────────┐    ┌──────────────────┐
    │ [0,0,1,0,...│    │ 300 neurons   │    │ prob over 50,000 │
    │  vocab=50K] │ ─▶ │ (no activation│ ─▶ │ words            │
    │             │    │ function)     │    │                  │
    └─────────────┘    └───────────────┘    └──────────────────┘
      1 × 50,000          1 × 300             1 × 50,000
    
    The hidden layer weight matrix (50,000 × 300) IS the embedding table.

**CBOW (Continuous Bag of Words):**
The inverse — given the context words, predict the center word.
Averages the context vectors, then predicts. Faster to train, slightly worse quality.

### GloVe (Global Vectors, 2014)

Word2Vec only uses local context (a sliding window).
GloVe uses the *global* co-occurrence matrix of the entire corpus.

    Build a matrix X where X[i][j] = how often word j appears near word i
    in the entire training corpus.
    
    Objective: learn vectors u_i and v_j such that:
    u_i · v_j + b_i + b_j  ≈  log(X[i][j])
    
    This means the dot product of two word vectors approximates
    the log of how often those words co-occur globally.

GloVe captures both local context (like Word2Vec) and global corpus statistics,
often producing slightly better analogical reasoning results.

### FastText (2016)

Extension of Word2Vec that represents words as bags of character n-grams.

    "where"  →  ["<wh", "whe", "her", "ere", "re>", "<where>"]
                (n-grams of size 3-6, plus the full word)
    
    The word vector = sum of all its n-gram vectors.

Advantages:
- Handles OOV words: "unbelievable" not in vocab? Sum its n-gram vectors anyway.
- Morphologically aware: "running", "runner", "runs" share many n-grams → similar vectors
- Better for languages with rich morphology (Finnish, Turkish, German)

### Contextual Embeddings — What They Add

Contextual embeddings (BERT, GPT, LLaMA) produce a *different* vector
for the same word depending on its surrounding context.

    "I went to the river bank to fish."
    "bank" → [-0.32, 0.71, -0.44, ...]   ← river sense

    "I deposited money at the bank."
    "bank" → [0.88, -0.21, 0.59, ...]    ← financial sense

This requires running the full Transformer model — covered in Module 03.
The embedding table produces the initial static vector; the transformer layers
transform it into a context-aware representation.

---

## 6. Positional Encoding — Injecting Order Into Attention

Transformers process all tokens in parallel (unlike RNNs, which process left to right).
This parallel processing is what makes them fast and parallelizable — but it means
the model has no inherent sense of which token came first.

Without positional encoding, "The cat sat on the mat" and "The mat sat on the cat"
would produce identical representations (same tokens, same attention patterns).

Positional encoding adds position information by adding a position vector
to the embedding of each token:

    final_vector[pos] = embedding[token_id] + positional_encoding[pos]

Both vectors have the same dimension, so they can be added directly.

---

### A) Sinusoidal Positional Encoding (Original Transformer, 2017)

Uses fixed sine and cosine waves at different frequencies:

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Where:
- pos = position of the token in the sequence (0, 1, 2, ...)
- i   = dimension index (0, 1, 2, ... d_model/2)
- d_model = embedding dimension

The intuition: each dimension oscillates at a different frequency.
Low-frequency dimensions capture coarse (early/late) position.
High-frequency dimensions capture fine-grained (adjacent) position.

    Position 0:  [sin(0/1), cos(0/1), sin(0/100), cos(0/100), ...]
                = [0.000,   1.000,    0.000,       1.000,     ...]
    
    Position 1:  [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
                = [0.841,   0.540,    0.010,       1.000,     ...]
    
    Position 10: [sin(10), cos(10), sin(0.1), cos(0.1), ...]
                = [-0.544, -0.839,   0.0998,   0.995,   ...]

Key property: cos(pos1 - pos2) can be computed from PE(pos1) and PE(pos2),
so the model can learn to attend based on *relative* distance, not just absolute position.

No parameters to learn. Works for any sequence length.
Extrapolates beyond training lengths (though quality degrades).
Used by: original Transformer, some encoder models.

---

### B) Learned Positional Embeddings

Instead of a fixed formula, the positional encoding is a learnable embedding table
with shape [max_seq_length × d_model] — trained alongside the model.

    Position 0 → a learned vector [0.023, -0.891, ...]
    Position 1 → a learned vector [0.512,  0.334, ...]
    ...
    Position 511 → a learned vector [-0.671, 0.123, ...]

Advantages: the model can learn the most useful positional representation for the task.
Disadvantages: fixed maximum sequence length (no extrapolation beyond max_seq_length).

Used by: BERT (max 512 tokens), GPT-2 (max 1024 tokens).

---

### C) RoPE — Rotary Positional Encoding

RoPE encodes position as a *rotation* in 2D subspaces of the embedding.
For each pair of dimensions (dim_2i, dim_2i+1), the vector is rotated by an
angle proportional to the position.

    For a vector v at position m, dimension pair (2i, 2i+1):
    
    [v_2i',    ]   [cos(m·θ_i)   -sin(m·θ_i)] [v_2i  ]
    [v_2i+1'   ] = [sin(m·θ_i)    cos(m·θ_i)] [v_2i+1]
    
    where θ_i = 10000^(-2i / d_model)   (same base as sinusoidal)

The key mathematical property: the dot product between two RoPE-encoded vectors
at positions m and n depends only on their *relative distance* (m - n), not their
absolute positions. This naturally makes attention relative.

This is applied to the Query and Key vectors inside each attention head,
not to the raw embeddings.

RoPE with "NTK-aware scaling" allows extending the context window far beyond
the length trained on (e.g., extending LLaMA-2 from 4K to 32K+ context).

Used by: LLaMA (all versions), Mistral, GPT-NeoX, Falcon, Yi, Qwen, and
most modern open-source LLMs.

---

### D) ALiBi — Attention with Linear Biases

ALiBi takes a completely different approach: instead of adding position information
to the embeddings, it adds a linear penalty to the attention scores based on
the distance between tokens.

    Modified attention score:
    score(q_i, k_j)  =  (q_i · k_j) / √d  −  m · |i − j|

    where m is a head-specific slope (different slope per attention head,
    set by a geometric sequence based on number of heads).

Intuition: tokens far apart are penalized — attention decays with distance.
The slope m controls how steeply attention drops off.

Key advantage: trained on short sequences (1K), extrapolates well to long sequences (4K+)
without quality degradation. No learnable parameters needed.

Used by: BLOOM (176B), MPT, OpenLLM.

---

### Comparison: Positional Encoding Methods

    │Method         │ Learnable │ Relative  │ Extrapolates    │ Applied To        │ Used By              │
    │───────────────┼───────────┼───────────┼─────────────────┼───────────────────┼──────────────────────│
    │Sinusoidal     │ No        │ Partial   │ Yes (degrades)  │ Embeddings        │ Original Transformer │
    │Learned        │ Yes       │ No        │ No (hard limit) │ Embeddings        │ BERT, GPT-2          │
    │RoPE           │ No        │ Yes       │ Yes (w/ scaling)│ Q, K in attention │ LLaMA, Mistral       │ 
    │ALiBi          │ No        │ Yes       │ Yes (strong)    │ Attention scores  │ BLOOM, MPT           │

---

## 7. The Complete Pipeline — From Raw Text to Model Input

Let's trace a single example all the way through:

    Input text:   "The cat sat"
    Model:        LLaMA-3 (d_model=4096, vocab=128,256)

    ─────────────────────────────────────────────────────────────────────────────
    STEP 1: Tokenization (text → token IDs)
    ─────────────────────────────────────────────────────────────────────────────
    "The cat sat"
        ↓  tokenizer (BPE, vocab=128,256)
    [791, 8415, 7482]    ← just 3 integer IDs
    
    The tokenizer is deterministic — same input always gives same output.
    It runs on CPU, before the model ever sees anything.
    
    ─────────────────────────────────────────────────────────────────────────────
    STEP 2: Add Special Tokens (BOS marker)
    ─────────────────────────────────────────────────────────────────────────────
    [1, 791, 8415, 7482]
     ↑
     BOS token (ID=1 for LLaMA) added at the front.
     For instruction-tuned models, instruction markers also wrap the input:
     [1, 518, 25580, 29962, 791, 8415, 7482, 518, 29914, 25580, 29962]
         ←─── [INST] ────→                  ←────── [/INST] ──────→
    
    ─────────────────────────────────────────────────────────────────────────────
    STEP 3: Batching & Padding (if multiple sequences)
    ─────────────────────────────────────────────────────────────────────────────
    Single sequence (no padding needed in this example).
    In a batch of mixed lengths, shorter sequences are right-padded with 0 (PAD ID)
    and an attention mask (1=real, 0=pad) is created alongside.
    
    What is an attention mask and why does it matter?
    GPUs process all sequences in a batch simultaneously, which requires them to be
    the same length — hence padding. But we don't want the model to treat PAD tokens
    as real information. The attention mask tells the model which positions to ignore:
    
        Sequence A (length 4):  [1, 791, 8415, 7482]    mask: [1, 1, 1, 1]
        Sequence B (length 2):  [1, 2832,    0,    0]   mask: [1, 1, 0, 0]
                                                ↑ ↑                  ↑  ↑
                                         PAD tokens               ignored in attention
    
    Without the mask, PAD tokens would "bleed" into the model's understanding of
    real tokens — polluting the output. The mask ensures padding is invisible.
    
    ─────────────────────────────────────────────────────────────────────────────
    STEP 4: Move to GPU as Tensor
    ─────────────────────────────────────────────────────────────────────────────
    input_ids     = tensor([[1, 791, 8415, 7482]])   shape: [1, 4]   (integers)
    attention_mask = tensor([[1,   1,    1,    1]])   shape: [1, 4]   (1s, no padding)
    
    ─────────────────────────────────────────────────────────────────────────────
    STEP 5: Embedding Lookup (IDs → Dense Vectors)
    ─────────────────────────────────────────────────────────────────────────────
    The model's embedding layer (shape [128,256 × 4096]) looks up each ID:
    
    ID    1  → [0.023, -0.891, 0.445, ..., 0.112]     ← BOS vector,   4096 floats
    ID  791  → [0.512,  0.334,-0.667, ..., 0.889]     ← "The" vector, 4096 floats
    ID 8415  → [-0.234, 0.776, 0.190, ...,-0.423]     ← "cat" vector, 4096 floats
    ID 7482  → [0.667, -0.441, 0.321, ..., 0.017]     ← "sat" vector, 4096 floats
    
    Result: hidden_states of shape [1, 4, 4096]      ← batch=1, tokens=4, dims=4096
    
    ─────────────────────────────────────────────────────────────────────────────
    STEP 6: Add Positional Encoding
    ─────────────────────────────────────────────────────────────────────────────
    For RoPE (LLaMA's approach): position is NOT added to embeddings here.
    Instead, RoPE rotations are applied to Q and K inside each attention head.
    
    For sinusoidal/learned PE: a position vector is added to each token's embedding:
    hidden_states[0, pos, :] += positional_encoding[pos, :]   for each pos
    
    Shape unchanged: [1, 4, 4096]
    
    ─────────────────────────────────────────────────────────────────────────────
    STEP 7: Flow Through Transformer Layers
    ─────────────────────────────────────────────────────────────────────────────
    The [1, 4, 4096] tensor flows through all 32 transformer layers
    (for LLaMA-3 8B), being transformed at each step.
    
    This is where static embeddings become contextual representations.

    ─────────────────────────────────────────────────────────────────────────────
    FULL PICTURE:
    ─────────────────────────────────────────────────────────────────────────────

    RAW TEXT     STEP 1      STEP 2       STEP 3       STEP 4       STEP 5          STEP 6         STEP 7
    "The cat" → Tokenize → Add BOS   → Batch +    → Move to   → Embedding     → Positional   → Transformer
                to IDs      + markers   Padding       GPU          Lookup          Encoding       Layers
    
    "The cat"   [791, 8415]  [1, 791,    [1, 4]       GPU         [1, 4, 4096]    [1, 4, 4096]   contextual
    (string)    (integers)   8415]       tensor        tensor      (float)         (+ position)   reps
                             (integers)  on CPU        on GPU      3D tensor       3D tensor      [1, 4, 4096]

---

## 8. Memory Cost of Embeddings

The embedding table is not free. For large vocab + large d_model:
   
    Model               Vocab      d_model    Table Shape          Size (BF16)
    ──────────────────────────────────────────────────────────────────────────
    GPT-2 base          50,257     768         [50K × 768]              ~77 MB
    GPT-2 XL            50,257    1,600        [50K × 1,600]           ~161 MB
    LLaMA-2 7B          32,000    4,096        [32K × 4,096]           ~262 MB
    LLaMA-3 8B         128,256    4,096        [128K × 4,096]         ~1.05 GB
    GPT-4 (estimated)  100,256    ~12,288      [100K × 12K]           ~2.46 GB

For LLaMA-3 8B: just the embedding table is ~1 GB — over 12% of total model size.
This is why vocabulary size is a real engineering tradeoff, not a free choice.

Additionally, the embedding table is often *tied* with the output (LM head) weight matrix.
This means the same matrix is used to embed input tokens AND to project back to logits
at the end. This halves the memory cost and improves training stability.

---

### Summary Mental Model

    RAW TEXT                   TOKENIZATION               EMBEDDING
    ┌────────────────┐          ┌────────────────┐          ┌────────────────────────────────┐
    │ "The cat sat"  │  ──────▶ │ [791, 8415,    │  ──────▶ │ [[...4096 floats for "The"],   │
    │                │          │  7482]         │          │  [...4096 floats for "cat"],   │
    │                │          │                │          │  [...4096 floats for "sat"]]   │
    │  (string)      │          │  (integers)    │          │                                │
    │                │          │                │          │  Shape: [1, 3, 4096]           │
    │                │          │  on CPU        │          │  on GPU, in bfloat16           │
    └────────────────┘          └────────────────┘          └────────────────────────────────┘
    No neural network yet       Tokenizer is fixed,         Embedding table = model weight,
                                deterministic, not          learned during training,
                                part of the model           updated by backprop

    POSITIONAL ENCODING         TRANSFORMER LAYERS           OUTPUT
    ┌──────────────────────┐    ┌───────────────────────┐    ┌──────────────────────────────┐
    │ Add position info to │    │ 32 layers of self-    │    │ Final [1, 3, 4096] tensor    │
    │ each token vector    │    │ attention + FFN       │    │ = context-aware embedding    │
    │                      │    │ transform the static  │    │ for each token               │
    │ "cat" at position 1  │    │ embeddings into       │    │                              │
    │ ≠ "cat" at position  │    │ contextual ones       │    │ → LM head projects to logits │
    │ 5 (different PE)     │    │                       │    │ → Next token prediction      │
    └──────────────────────┘    └───────────────────────┘    └──────────────────────────────┘

{{TOKENIZATION_IMAGE}}

---

## What's Next — Module 03: Attention

You now understand the complete input pipeline:
raw text → tokens → integer IDs → embedding vectors → positional encoding added.

The vectors entering the Transformer are still *static* — "cat" at position 1 has no
idea what "sat" at position 2 means. The Transformer's job is to make every token
*aware* of every other token in the sequence. The mechanism that does this is
called **self-attention**.

Module 03 answers: how do 4,096-dimensional vectors "talk" to each other,
and how does that produce the contextual understanding that makes LLMs powerful?
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
|--------------------------|------------------|-------------------|---------------------|
| Aspect                   | Character        | Word              | Subword (BPE/WP)    |
|--------------------------|------------------|-------------------|---------------------|
| Vocab Size               | ~256             | ~100K–1M+         | 30K–128K            |
| OOV Handling             | None             | [UNK] token       | Always decomposable |
| Sequence Length          | Very long        | Short             | Moderate            |
| Morphology Awareness     | Yes (trivially)  | No                | Yes (learned)       |
| Multilingual Support     | Good             | Poor              | Good                |
| Training Data Needed     | Minimal          | Large             | Moderate            |
| Used By                  | Early char-RNNs  | Legacy NLP        | All modern LLMs     |
|--------------------------|------------------|-------------------|---------------------|
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = \
    {

        "1 · Tokenize text with HuggingFace":
            {
                "description": "Compare how different tokenizers split the same sentence.",
                "language": "python",
                "code": """
                        from transformers import AutoTokenizer

                        sentence = "The quick brown fox jumps over the lazy dog. Tokenization matters!"

                        for model_name in ["gpt2", "bert-base-uncased", "t5-small"]:
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                            tokens = tokenizer.tokenize(sentence)
                            ids    = tokenizer.encode(sentence)
                            print(f"\\n{'='*50}")
                            print(f"Model    : {model_name}")
                            print(f"Tokens   : {tokens}")
                            print(f"Token IDs: {ids}")
                            print(f"Vocab size: {tokenizer.vocab_size:,}")
                    """,
            },

        "2 · BPE from scratch":
            {
                "description": "Implement Byte Pair Encoding from scratch to understand the merge algorithm.",
                "language": "python",
                "code": """
                        from collections import Counter

                        def get_vocab(corpus: list[str]) -> dict:
                            vocab = Counter()
                            for word in corpus:
                                vocab[' '.join(list(word)) + ' </w>'] += 1
                            return dict(vocab)

                        def get_stats(vocab: dict) -> Counter:
                            pairs = Counter()
                            for word, freq in vocab.items():
                                symbols = word.split()
                                for i in range(len(symbols) - 1):
                                    pairs[(symbols[i], symbols[i+1])] += freq
                            return pairs

                        def merge_vocab(pair: tuple, vocab: dict) -> dict:
                            new_vocab = {}
                            bigram = ' '.join(pair)
                            replacement = ''.join(pair)
                            for word in vocab:
                                new_word = word.replace(bigram, replacement)
                                new_vocab[new_word] = vocab[word]
                            return new_vocab

                        # ── Demo ──
                        corpus = ["low", "low", "lower", "newer", "newest", "widest"]
                        vocab  = get_vocab(corpus)
                        print("Initial vocab:", vocab)

                        NUM_MERGES = 8
                        for i in range(NUM_MERGES):
                            stats  = get_stats(vocab)
                            best   = max(stats, key=stats.get)
                            vocab  = merge_vocab(best, vocab)
                            print(f"Merge {i+1}: {best!r}  →  {''.join(best)!r}")

                        print("\\nFinal vocab:", vocab)
                        """,
            },

        "3 · Word2Vec with Gensim":
            {
                "description": "Train a Word2Vec model and explore the embedding space.",
                "language": "python",
                "code": """
                        from gensim.models import Word2Vec

                        sentences = [
                            ["the", "king", "rules", "the", "kingdom"],
                            ["the", "queen", "rules", "the", "land"],
                            ["man", "is", "strong", "and", "powerful"],
                            ["woman", "is", "strong", "and", "powerful"],
                            ["the", "prince", "will", "become", "king"],
                            ["the", "princess", "will", "become", "queen"],
                            ["paris", "is", "the", "capital", "of", "france"],
                            ["berlin", "is", "the", "capital", "of", "germany"],
                        ]

                        model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, epochs=100, seed=42)

                        print("Most similar to 'king':")
                        print(model.wv.most_similar("king", topn=5))

                        print("\\nMost similar to 'queen':")
                        print(model.wv.most_similar("queen", topn=5))

                        # Analogy: king - man + woman ≈ ?
                        result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
                        print("\\nking - man + woman ≈", result)
                        """,
            },

        "4 · Positional Encoding — Sinusoidal":
            {
                "description": "Implement and visualize sinusoidal positional encoding.",
                "language": "python",
                "code": """
                        import numpy as np

                        def sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
                            pe  = np.zeros((max_len, d_model))
                            pos = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
                            i   = np.arange(d_model)[np.newaxis, :]           # (1, d_model)
                            div = np.power(10000, (2 * (i // 2)) / d_model)

                            pe[:, 0::2] = np.sin(pos / div[:, 0::2])          # even dims
                            pe[:, 1::2] = np.cos(pos / div[:, 1::2])          # odd  dims
                            return pe

                        PE = sinusoidal_encoding(max_len=50, d_model=16)

                        print(f"Positional encoding shape: {PE.shape}")
                        print(f"\\nFirst 5 positions, first 8 dims:")
                        print(np.round(PE[:5, :8], 3))

                        # Verify: cosine similarity between nearby positions should be high
                        from numpy.linalg import norm
                        def cosine_sim(a, b):
                            return np.dot(a, b) / (norm(a) * norm(b))

                        print("\\nCosine similarities (position 0 vs others):")
                        for p in [1, 5, 10, 20, 49]:
                            sim = cosine_sim(PE[0], PE[p])
                            print(f"  pos 0 vs pos {p:2d}: {sim:.4f}")
                        """,
            },

        "5 · Tokenize text — compare across models":
            {
                "description": (
                    "Run the same sentence through GPT-2 (BPE), BERT (WordPiece), and T5 (Unigram/SentencePiece) "
                    "and compare how each model splits and IDs the tokens."
                ),
                "language": "python",
                "code": """
                        from transformers import AutoTokenizer

                        sentence = "The quick brown fox jumps over the lazy dog. Tokenization matters!"

                        for model_name in ["gpt2", "bert-base-uncased", "t5-small"]:
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                            tokens    = tokenizer.tokenize(sentence)
                            ids       = tokenizer.encode(sentence)

                            print(f"\\n{'='*60}")
                            print(f"Model     : {model_name}")
                            print(f"Tokens    : {tokens}")
                            print(f"Token IDs : {ids}")
                            print(f"# tokens  : {len(tokens)}")
                            print(f"Vocab size: {tokenizer.vocab_size:,}")

                        # What to notice:
                        # • GPT-2 uses BPE — "Ġ" prefix marks tokens that follow a space
                        # • BERT WordPiece — "##" prefix marks continuation subwords
                        # • T5 SentencePiece — "▁" marks word beginnings (space encoded in token)
                        # • Same sentence → different number of tokens across models
                        """,
            },

        "6· BPE from scratch — traced merge steps":
            {
                "description": (
                    "Implement the full BPE algorithm from scratch and trace every merge step "
                    "to see exactly how the vocabulary is built."
                ),
                "language": "python",
                "code": """
                        from collections import Counter

                        def get_vocab(corpus):
                            \"\"\"Represent each word as space-separated characters with </w> end marker.\"\"\"
                            vocab = Counter()
                            for word in corpus:
                                vocab[' '.join(list(word)) + ' </w>'] += 1
                            return dict(vocab)

                        def get_stats(vocab):
                            \"\"\"Count frequency of every adjacent symbol pair across all words.\"\"\"
                            pairs = Counter()
                            for word, freq in vocab.items():
                                symbols = word.split()
                                for i in range(len(symbols) - 1):
                                    pairs[(symbols[i], symbols[i + 1])] += freq
                            return pairs

                        def merge_vocab(pair, vocab):
                            \"\"\"Replace all occurrences of 'pair' with the merged symbol.\"\"\"
                            bigram = ' '.join(pair)
                            merged = ''.join(pair)
                            return {word.replace(bigram, merged): freq for word, freq in vocab.items()}

                        # ── Demo corpus ──
                        corpus = ["low", "low", "lower", "newer", "newest", "widest"]
                        vocab  = get_vocab(corpus)

                        print("Initial character-level vocab:")
                        for word, freq in sorted(vocab.items(), key=lambda x: -x[1]):
                            print(f"  {freq}×  {word}")

                        print(f"\\nRunning BPE merges...")
                        print("-" * 50)

                        NUM_MERGES   = 10
                        learned_merges = []

                        for i in range(NUM_MERGES):
                            stats = get_stats(vocab)
                            if not stats:
                                break
                            best  = max(stats, key=stats.get)
                            vocab = merge_vocab(best, vocab)
                            learned_merges.append(best)
                            print(f"Merge {i+1:2d}: {best[0]!r:12s} + {best[1]!r:12s} → {''.join(best)!r}  "
                                  f"(appeared {stats[best]} times)")

                        print("\\nFinal vocabulary tokens:")
                        all_tokens = set()
                        for word in vocab:
                            all_tokens.update(word.split())
                        print(sorted(all_tokens))

                        print(f"\\nLearned {len(learned_merges)} merge rules — these become the tokenizer's merge table.")
                        print("New text is tokenized by: start character-level, apply merges in order.")
                        """,
            },

        "7 · Word2Vec — train and explore embedding geometry":
            {
                "description": (
                    "Train a Word2Vec model with Gensim and explore how meaning is encoded "
                    "as geometry — vector arithmetic, analogies, and nearest neighbours."
                ),
                "language": "python",
                "code": """
                        from gensim.models import Word2Vec
                        import numpy as np

                        sentences = [
                            ["the", "king", "rules", "the", "kingdom"],
                            ["the", "queen", "rules", "the", "land"],
                            ["the", "king", "and", "queen", "govern", "together"],
                            ["man", "is", "strong", "and", "powerful"],
                            ["woman", "is", "strong", "and", "powerful"],
                            ["the", "prince", "will", "become", "king"],
                            ["the", "princess", "will", "become", "queen"],
                            ["paris", "is", "the", "capital", "of", "france"],
                            ["berlin", "is", "the", "capital", "of", "germany"],
                            ["london", "is", "the", "capital", "of", "england"],
                            ["france", "germany", "england", "are", "countries"],
                            ["man", "and", "woman", "are", "people"],
                            ["king", "man", "rules", "kingdom"],
                            ["queen", "woman", "rules", "land"],
                        ]

                        model = Word2Vec(
                            sentences,
                            vector_size=64,
                            window=3,
                            min_count=1,
                            epochs=300,
                            sg=1,        # 1 = skip-gram, 0 = CBOW
                            seed=42,
                        )

                        print("Embedding dimension:", model.vector_size)
                        print("Vocabulary size    :", len(model.wv))
                        print()

                        print("Vector for 'king' (first 8 dims):")
                        print(np.round(model.wv['king'][:8], 4))
                        print()

                        print("Most similar to 'king':")
                        for word, score in model.wv.most_similar("king", topn=5):
                            print(f"  {word:15s}  similarity={score:.4f}")

                        print()
                        print("Most similar to 'france':")
                        for word, score in model.wv.most_similar("france", topn=4):
                            print(f"  {word:15s}  similarity={score:.4f}")

                        print()
                        # Vector arithmetic: king - man + woman ≈ queen
                        result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
                        print("king − man + woman ≈")
                        for word, score in result:
                            print(f"  {word:15s}  similarity={score:.4f}")

                        print()
                        # paris - france + germany ≈ berlin
                        result2 = model.wv.most_similar(positive=["paris", "germany"], negative=["france"], topn=3)
                        print("paris − france + germany ≈")
                        for word, score in result2:
                            print(f"  {word:15s}  similarity={score:.4f}")

                        print()
                        cosine = model.wv.similarity
                        print("Cosine similarities:")
                        print(f"  king  vs queen  : {cosine('king', 'queen'):.4f}")
                        print(f"  king  vs man    : {cosine('king', 'man'):.4f}")
                        print(f"  king  vs paris  : {cosine('king', 'paris'):.4f}")
                        print(f"  france vs germany: {cosine('france', 'germany'):.4f}")
                        """,
            },

        "8 · Embedding table — size and lookup mechanics":
            {
                "description": (
                    "Inspect the shape and memory cost of a real embedding table using HuggingFace. "
                    "See exactly what a lookup returns and how tied embeddings work."
                ),
                "language": "python",
                "code": """
                        import torch
                        from transformers import AutoTokenizer, AutoModel

                        model_name = "gpt2"
                        tokenizer  = AutoTokenizer.from_pretrained(model_name)
                        model      = AutoModel.from_pretrained(model_name)

                        emb_table  = model.wte  # Word Token Embedding table
                        pos_table  = model.wpe  # Word Position Embedding table (GPT-2 uses learned PE)

                        vocab_size, d_model = emb_table.weight.shape
                        max_pos             = pos_table.weight.shape[0]

                        print(f"Model: {model_name}")
                        print(f"Vocab size       : {vocab_size:,}")
                        print(f"Embedding dim    : {d_model}")
                        print(f"Max positions    : {max_pos}")
                        print()

                        param_count = vocab_size * d_model
                        bytes_fp32  = param_count * 4
                        bytes_bf16  = param_count * 2
                        print(f"Embedding table params: {param_count:,}")
                        print(f"Memory (float32)      : {bytes_fp32 / 1e6:.1f} MB")
                        print(f"Memory (bfloat16)     : {bytes_bf16 / 1e6:.1f} MB")
                        print()

                        # Perform a manual embedding lookup
                        sentence = "The cat sat"
                        ids      = tokenizer.encode(sentence)
                        id_tensor = torch.tensor([ids])

                        print(f"Tokens: {tokenizer.convert_ids_to_tokens(ids)}")
                        print(f"IDs   : {ids}")
                        print()

                        with torch.no_grad():
                            token_embeddings = emb_table(id_tensor)   # shape [1, seq_len, d_model]
                            pos_ids          = torch.arange(len(ids)).unsqueeze(0)
                            pos_embeddings   = pos_table(pos_ids)
                            final_embeddings = token_embeddings + pos_embeddings  # add position

                        print(f"Token embedding shape : {token_embeddings.shape}")
                        print(f"Position embedding shape: {pos_embeddings.shape}")
                        print(f"Final (token + pos) shape: {final_embeddings.shape}")
                        print()
                        print(f"'The' embedding (first 8 dims)  :", token_embeddings[0, 0, :8].numpy().round(4))
                        print(f"'cat' embedding (first 8 dims)  :", token_embeddings[0, 1, :8].numpy().round(4))
                        print(f"'sat' embedding (first 8 dims)  :", token_embeddings[0, 2, :8].numpy().round(4))
                        print()

                        # Cosine similarity between word embeddings (before context/attention)
                        def cosine(a, b):
                            a, b = a.numpy(), b.numpy()
                            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

                        import numpy as np
                        the_vec = token_embeddings[0, 0]
                        cat_vec = token_embeddings[0, 1]
                        sat_vec = token_embeddings[0, 2]
                        print("Static cosine similarities (before transformer layers):")
                        print(f"  'The' vs 'cat' : {cosine(the_vec, cat_vec):.4f}")
                        print(f"  'The' vs 'sat' : {cosine(the_vec, sat_vec):.4f}")
                        print(f"  'cat' vs 'sat' : {cosine(cat_vec, sat_vec):.4f}")
                        """,
            },

        "9 · Positional Encoding — sinusoidal, visualised":
            {
                "description": (
                    "Implement sinusoidal positional encoding from scratch. "
                    "Print the encoding matrix and verify its properties — "
                    "nearby positions are more similar than distant ones."
                ),
                "language": "python",
                "code": """
                        import numpy as np

                        def sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
                            PE  = np.zeros((max_len, d_model))
                            pos = np.arange(max_len)[:, None]       # (max_len, 1)
                            i   = np.arange(d_model)[None, :]       # (1, d_model)
                            div = np.power(10000.0, (2 * (i // 2)) / d_model)

                            PE[:, 0::2] = np.sin(pos / div[:, 0::2])   # even dims → sin
                            PE[:, 1::2] = np.cos(pos / div[:, 1::2])   # odd dims  → cos
                            return PE

                        PE = sinusoidal_pe(max_len=100, d_model=64)

                        print(f"Positional encoding shape: {PE.shape}  (max_len=100, d_model=64)")
                        print()
                        print("First 5 positions, first 8 dimensions:")
                        header = " pos │ " + "  ".join([f"dim_{d}" for d in range(8)])
                        print(header)
                        print("-" * len(header))
                        for p in range(5):
                            row = f"  {p}  │ " + "  ".join([f"{PE[p, d]:+.3f}" for d in range(8)])
                            print(row)

                        print()
                        print("Observation: dim 0 (sin, fast freq) changes quickly between positions.")
                        print("             dim 62 (cos, slow freq) barely changes — encodes coarse position.")

                        # Cosine similarity: nearby positions should score higher
                        def cosine_sim(a, b):
                            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

                        print()
                        print("Cosine similarity between position 0 and other positions:")
                        print("(closer positions should be more similar)")
                        for p in [1, 2, 5, 10, 20, 50, 99]:
                            sim = cosine_sim(PE[0], PE[p])
                            bar = "█" * int((sim + 1) * 10)
                            print(f"  pos 0 vs pos {p:2d}: {sim:+.4f}  {bar}")

                        print()
                        print("Key property: similarity decays with distance → model can learn relative position.")
                        print()
                        print("Verify: all position vectors have the same norm (fixed magnitude):")
                        norms = np.linalg.norm(PE, axis=1)
                        print(f"  min norm={norms.min():.3f}, max norm={norms.max():.3f}, std={norms.std():.4f}")
                        """,
            },

        "10 · RoPE — Rotary Positional Encoding":
            {
                "description": (
                    "Implement RoPE from scratch and demonstrate how it encodes relative position "
                    "into the Query/Key dot product — the property that makes it so effective."
                ),
                "language": "python",
                "code": """
                    import numpy as np

                    def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
                        \"\"\"Precompute RoPE cos/sin tables for a given sequence length and head dimension.\"\"\"
                        # θ_i = base^(-2i / head_dim)  for i in [0, head_dim/2)
                        half = head_dim // 2
                        theta = base ** (-np.arange(half) * 2 / head_dim)   # (half,)
                        positions = np.arange(seq_len)                        # (seq_len,)
                        freqs     = np.outer(positions, theta)                # (seq_len, half)
                        cos_cache = np.cos(freqs)                             # (seq_len, half)
                        sin_cache = np.sin(freqs)                             # (seq_len, half)
                        return cos_cache, sin_cache

                    def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
                        \"\"\"Apply RoPE rotation to a query or key vector x of shape (seq_len, head_dim).\"\"\"
                        half = x.shape[-1] // 2
                        x1, x2 = x[..., :half], x[..., half:]          # split dims
                        # Rotate: [x1, x2] → [x1·cos - x2·sin, x1·sin + x2·cos]
                        return np.concatenate([x1 * cos - x2 * sin,
                                               x1 * sin + x2 * cos], axis=-1)

                    # Parameters matching a small attention head
                    seq_len  = 8
                    head_dim = 16    # must be even
                    base     = 10000.0

                    cos_cache, sin_cache = build_rope_cache(seq_len, head_dim, base)

                    print(f"RoPE cache shape: {cos_cache.shape}  (seq_len={seq_len}, head_dim/2={head_dim//2})")
                    print()

                    # Simulate a query and key vector (as if extracted from attention)
                    np.random.seed(42)
                    Q = np.random.randn(seq_len, head_dim).astype(np.float32)   # queries at each position
                    K = np.random.randn(seq_len, head_dim).astype(np.float32)   # keys   at each position

                    Q_rope = apply_rope(Q, cos_cache, sin_cache)
                    K_rope = apply_rope(K, cos_cache, sin_cache)

                    # The dot product Q_rope[m] · K_rope[n] depends only on (m - n), not absolute positions
                    # Demonstrate: score between same relative distance pairs should behave consistently
                    print("Attention scores  Q[m] · K[n]  (without RoPE vs with RoPE):")
                    print()
                    print(f"{'(m, n)':12s}  {'relative dist':14s}  {'score (no PE)':14s}  {'score (RoPE)':12s}")
                    print("-" * 60)

                    pairs = [(0,1), (1,2), (2,3), (3,4),   # distance = 1
                             (0,2), (1,3), (2,4),           # distance = 2
                             (0,4), (1,5)]                  # distance = 4

                    for m, n in pairs:
                        score_raw  = float(Q[m] @ K[n])
                        score_rope = float(Q_rope[m] @ K_rope[n])
                        print(f"  ({m},{n})         dist={abs(m-n)}              {score_raw:+.4f}          {score_rope:+.4f}")

                    print()
                    print("Key property of RoPE: the score Q_rope[m] · K_rope[n] is a function of")
                    print("only (m-n), not absolute m or n. Same relative distance → same encoding.")
                    print("This is why RoPE produces naturally relative attention.")
                    """,
            },

    }

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
        scripts_dir = _SCRIPTS_DIR
    if main_script is None:
        main_script = _MAIN_SCRIPT

    scripts_available = main_script.exists()

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

# def get_topic_data():
#     """Return all content for this topic module (legacy interface)."""
#     return get_content()

def get_topic_data():
    try:
        visual_html = get_visual_html()
    except Exception:
        visual_html = ""
    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": visual_html,
        "operations": OPERATIONS,
    }

def get_content():
    """Return all content for this topic module."""
    try:
        from Required_Images.tokenization_visual import get_visual_html, VISUAL_HEIGHT
        visual_html   = get_visual_html()
        visual_height = VISUAL_HEIGHT
    except Exception:
        visual_html   = ""
        visual_height = 400

    tok_img = _image_to_html(
        "Required_Images/Tokenization_Breakdown.png",
        alt="Tokenization & Embedding Pipeline",
        width="80%",
    )
    theory_with_images = THEORY.replace("{{TOKENIZATION_IMAGE}}", tok_img)

    interactive_components = [
        {
            "placeholder": "{{TOKENIZATION_IMAGE}}",
            "html":        visual_html,
            "height":      visual_height,
        }
    ]

    return {
        "display_name":          DISPLAY_NAME,
        "icon":                  ICON,
        "subtitle":              SUBTITLE,
        "theory":                theory_with_images,
        "theory_raw":            THEORY,
        "visual_html":           visual_html,
        "complexity":            COMPLEXITY,
        "operations":            OPERATIONS,
        "render_operations":     render_operations,
        "interactive_components": interactive_components,
    }