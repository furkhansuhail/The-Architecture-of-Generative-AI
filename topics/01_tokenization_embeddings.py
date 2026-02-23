"""Module 01: Tokenization & Embeddings"""
from Required_Images.tokenization_visual import get_visual_html

DISPLAY_NAME = "01 · Tokenization & Embeddings"
ICON         = "🔤"
SUBTITLE     = "How raw text becomes numbers — the bedrock of every language model"

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
$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
$$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$

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
"""

OPERATIONS = {
    "1 · Tokenize text with HuggingFace": {
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
""".strip(),
    },
    "2 · BPE from scratch": {
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
""".strip(),
    },
    "3 · Word2Vec with Gensim": {
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
""".strip(),
    },
    "4 · Positional Encoding — Sinusoidal": {
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
""".strip(),
    },
}

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
