"""
Tokenization, Vocabulary & Byte-Pair Encoding
===============================================

Before a language model sees a single weight update, raw text must be
converted into discrete integer IDs. Tokenization is that conversion.
The choice of tokenizer shapes vocabulary size, context efficiency,
multilingual capability, and even model arithmetic ability.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Tokenization, Vocabulary & BPE"
DISPLAY_NAME = "02 · Tokenization & BPE"
ICON         = "✂️"
SUBTITLE     = "How Text Becomes Numbers"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return (f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
                f'style="width:{width}; border-radius:8px; margin:12px 0;">')
    return f'<p style="color:red;">⚠️ Image not found: {path}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### What Is Tokenization?

Language models do not operate on characters, words, or sentences directly.
They operate on **tokens** — integer IDs drawn from a fixed vocabulary.
Tokenization is the process of converting a raw string into a sequence of
these integers, and converting a sequence of integers back into a string.

This seems like a mundane preprocessing step, but it has deep consequences:

    •   A larger vocabulary  → fewer tokens per sentence → longer effective
        context, but a larger embedding table and logit head.
    •   A smaller vocabulary → more tokens per sentence → shorter context per
        dollar, but more compact model weight matrices.
    •   Poor tokenization of numbers → arithmetic errors (each digit may be a
        separate token, destroying positional structure).
    •   Poor multilingual tokenization → non-English text uses 3–10× more
        tokens than English for the same content, making those languages
        effectively more expensive to model.

Tokenization is also the source of some of the strangest LLM failure modes —
like models that struggle to count letters in a word, because the word may be
tokenized as a single opaque token with no internal character structure.


### A Brief History: Three Generations of Tokenizers

**Generation 1 — Word-level tokenization**

    Split on whitespace and punctuation. Every unique word is a token.
    Vocabulary is huge (millions of entries in large corpora), unknown words
    cannot be represented (UNK token), and morphological variants ("run",
    "runs", "running") are entirely separate entries with no shared
    representation.

    Problem: open vocabulary. Any rare or misspelled word is OOV (out of
    vocabulary).

**Generation 2 — Character-level tokenization**

    Every character is a token. Vocabulary is tiny (~256 for ASCII, ~1114 for
    Unicode categories). Never has OOV problems. But sequences are very long
    (a 1000-character paragraph = 1000 tokens), and the model must learn to
    form words from scratch — expensive and slow.

**Generation 3 — Subword tokenization (current standard)**

    The sweet spot: vocabulary of ~30k–128k subword units. Common words are
    single tokens; rare words are split into meaningful pieces. No OOV — any
    string can be encoded as a sequence of known subwords plus fallback to
    individual bytes.

    The dominant algorithms: **BPE** (GPT family), **WordPiece** (BERT family),
    **SentencePiece/Unigram** (LLaMA, T5, Gemma).


### Byte-Pair Encoding (BPE) — The Algorithm

BPE was originally a data compression algorithm (Gage, 1994) adapted for NLP
by Sennrich et al. (2016) and then used by the GPT series.

**Core idea:** Start with individual characters (or bytes). Repeatedly find
the most frequent adjacent pair in the training corpus and merge them into a
new single token. Repeat until the vocabulary reaches the desired size.

**Step-by-step walkthrough:**

    Initial vocabulary: every unique character in the corpus.
    Corpus (simplified): "low low low lowest newest"

    Start by splitting each word into characters + end-of-word marker:
        l o w </w>  ×3
        l o w e s t </w>  ×1
        n e w e s t </w>  ×1

    Step 1 — Count all adjacent pairs:
        (l, o) : 4       ← appears in all "low*" words
        (o, w) : 4
        (e, s) : 2
        (e, w) : 1
        ...

    The most frequent pair is (l, o) with count 4. Merge them → "lo".

    Step 2 — Corpus after merge:
        lo w </w>  ×3
        lo w e s t </w>  ×1
        n e w e s t </w>  ×1

    Now most frequent pair is (lo, w) with count 4. Merge → "low".

    Step 3 — Corpus:
        low </w>  ×3
        low e s t </w>  ×1
        n e w e s t </w>  ×1

    Continue until vocabulary size is reached.

The key insight: **frequent substrings get merged first**, so common words
("the", "is", "of") end up as single tokens, while rare words decompose into
familiar subword pieces.


    **Diagram 1 — BPE Merge Tree:**

    BPE MERGE SEQUENCE (simplified)
    ════════════════════════════════════════════════════════════════

    Vocabulary starts as individual characters:
    {l, o, w, e, s, t, n, </w>}   (8 tokens)

    Merge 1:  (l + o)   → lo       vocab += {lo}
    Merge 2:  (lo + w)  → low      vocab += {low}
    Merge 3:  (e + s)   → es       vocab += {es}
    Merge 4:  (es + t)  → est      vocab += {est}
    Merge 5:  (low + est) → lowest vocab += {lowest}
    ...

    After V merges, vocabulary size = initial_chars + V

    Final encoding of "lowest":
        Before BPE: l  o  w  e  s  t        (6 tokens)
        After  BPE: low  est                 (2 tokens) ✓


    **Diagram 2 — The BPE Encoding Pipeline:**

    ENCODING A NEW STRING WITH A TRAINED BPE TOKENIZER
    ════════════════════════════════════════════════════════════════

    Input string:  "tokenization"

    Step 1 — Pre-tokenization (split on whitespace/punctuation):
        ["tokenization"]

    Step 2 — Start with character split:
        t  o  k  e  n  i  z  a  t  i  o  n

    Step 3 — Apply learned merge rules in order:
        Rule 1 (t+o → to):      to  k  e  n  i  z  a  t  i  o  n
        Rule 2 (to+k → tok):    tok  e  n  i  z  a  t  i  o  n
        Rule 3 (tok+en → token): token  i  z  a  t  i  o  n
        Rule 4 (i+z → iz):      token  iz  a  t  i  o  n
        Rule 5 (a+t → at):      token  iz  at  i  o  n
        ...

    Final tokens:  ["token", "iz", "ation"]
    Token IDs:     [  3044,  1096,   1634 ]   ← indices into vocab table


### Byte-Level BPE (GPT-2 and beyond)

Vanilla BPE operates on Unicode characters. But Unicode has 143,000+ code
points — many corpora will contain exotic characters that appear too rarely to
survive as vocabulary entries.

GPT-2 introduced **byte-level BPE**: start from the 256 raw byte values
instead of Unicode characters. This guarantees:

    •   The base vocabulary of 256 bytes covers every possible string.
    •   No UNK token is ever needed — any input can be encoded.
    •   The vocabulary remains compact and well-defined regardless of language.

GPT-4's tokenizer (cl100k_base) uses byte-level BPE with vocabulary size
100,277. LLaMA uses SentencePiece (a different implementation) with 32,000 or
128,000 tokens depending on the version.


### SentencePiece and the Unigram Language Model

SentencePiece (Kudo & Richardson, 2018) is a language-independent tokenization
library used by LLaMA, T5, Gemma, and many multilingual models.

Key differences from BPE:
    •   Trains directly on raw text, no pre-tokenization required.
    •   Supports both BPE and the **Unigram Language Model** algorithm.
    •   Treats whitespace as a regular character (uses a ▁ prefix symbol),
        making it truly language-agnostic (Chinese, Arabic, etc. work natively).

**Unigram LM:** Instead of greedily merging pairs bottom-up, Unigram starts
with a large candidate vocabulary and prunes tokens whose removal causes the
least increase in corpus encoding cost (measured by a unigram language model
over subword sequences). The result is often slightly better compression than
BPE and handles morphologically rich languages better.


### WordPiece (BERT Family)

WordPiece (Schuster & Nakamura, 2012; used by BERT, DistilBERT, etc.) is
similar to BPE but chooses the merge pair that maximises the likelihood of the
training data under a language model, rather than raw frequency:

    score(A, B) = freq(AB) / (freq(A) × freq(B))

This tends to merge pairs that co-occur more than chance would predict —
capturing morphological units better in practice. BERT's vocabulary uses ##
prefixes for continuation tokens (##ing, ##tion) to distinguish them from
word-initial tokens.


### Special Tokens

Every tokenizer defines a set of special tokens that are never produced by the
BPE/Unigram rules:

    Token       Purpose
    ─────────────────────────────────────────────────────────
    <|endoftext|>   GPT series: marks end of a document
    <s>, </s>       LLaMA: beginning/end of sequence
    [BOS], [EOS]    Generic: beginning/end of sequence
    [PAD]           Padding to equal length in a batch
    [UNK]           Out-of-vocabulary fallback (rare in byte-level)
    [MASK]          BERT: masked token for masked LM objective
    <|im_start|>    ChatML: marks the start of a conversation turn
    <|im_end|>      ChatML: marks the end of a conversation turn
    ─────────────────────────────────────────────────────────

For instruction-tuned / chat models, the **chat template** specifies how
conversation turns are wrapped with special tokens. Getting this right during
fine-tuning is critical — a mismatch between the tokenizer's chat template
and the training data format causes silent performance degradation.

Example — LLaMA-3 chat format:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful assistant.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    What is BPE?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>


### Token Fertility — A Measure of Tokenizer Quality

**Fertility** = average number of tokens per word. Lower is more efficient
(for a given vocabulary size). Cross-lingual fertility analysis reveals
tokenizer bias:

    Language        GPT-2 fertility     LLaMA-3 fertility
    ────────────────────────────────────────────────────
    English         ~1.3                ~1.2
    German          ~1.6                ~1.4
    Spanish         ~1.5                ~1.3
    Arabic          ~4.2                ~2.1
    Chinese         ~2.8                ~1.9
    Hindi           ~5.1                ~2.3
    ────────────────────────────────────────────────────

Models trained on GPT-2's tokenizer are implicitly more expensive for
non-English languages — every Arabic word consumes 4× the context budget of
an English word, meaning the model sees much less Arabic "content" per
training step.

This is a key reason why modern multilingual models train dedicated tokenizers
on balanced multilingual corpora before pre-training.


### Tokenization Gotchas That Affect Model Behaviour

**1. Number tokenization and arithmetic**

    "1234567" might tokenize as ["123", "456", "7"]
    "1234567" in another tokenizer: ["1", "2", "3", "4", "5", "6", "7"]

    Column-aligned arithmetic requires digits to be positionally consistent.
    Inconsistent number tokenization is a primary cause of LLM arithmetic
    failures. Newer tokenizers often keep single digits as tokens.

**2. Capitalisation and leading spaces**

    "Hello" and " Hello" (note the space) are often different tokens.
    "hello" ≠ "Hello" ≠ " hello" in most BPE tokenizers.
    This means the model learns separate embeddings for each — subtle but
    important for prompting: always match the format seen during training.

**3. Trailing whitespace and newlines**

    "\\n\\n" is often a single token. "\\n" + "\\n" gives the same string but
    may tokenize differently at document boundaries. When preparing training
    data, normalize whitespace consistently.

**4. Tokenizer-model mismatch**

    Loading a model's weights with the wrong tokenizer (e.g., GPT-2 weights
    with LLaMA tokenizer) is silent and catastrophic — the embedding table
    and logit head will be entirely misaligned.

**5. The "SolidGoldMagikarp" problem**

    Tokens can exist in the vocabulary that were present in the tokenizer
    training corpus but absent from the model's pre-training text. These
    "ghost tokens" have essentially random embeddings and can cause bizarre
    model outputs when used.


    **Diagram 3 — How Vocabulary Size Affects Context Efficiency:**

    VOCABULARY SIZE TRADE-OFFS
    ════════════════════════════════════════════════════════════════

    Same sentence: "The transformer architecture is highly parallelisable."

    Tokenizer         Vocab     Tokens    Tokens used
    ──────────────────────────────────────────────────────────────
    Character-level     128        50     T-h-e- -t-r-a-n-s-f-...
    BPE 8k           8,000        12     The  trans  former  arch...
    BPE 32k         32,000         9     The  transformer  arch...
    BPE 100k       100,000         8     The  transformer  architecture...
    ──────────────────────────────────────────────────────────────

    Each token consumes one position in the context window.
    With a 4096-token context:
      - Character-level: fits ~82 words
      - BPE 100k:        fits ~512 words (6.2× more content per context)

    But larger vocab → larger embedding table:
      - 32k  × 4096d = 131M params just for embeddings
      - 100k × 4096d = 410M params just for embeddings

    Trade-off: balance context efficiency vs. parameter cost.


### The Tokenization-Pretraining Feedback Loop

A subtle but important point: the tokenizer is trained on a corpus, then
frozen before model training begins. The tokenizer's vocabulary distribution
reflects the pre-tokenizer corpus, not the model training corpus. If you
train a tokenizer on English Wikipedia and then train your LLM on code,
you'll find that common code tokens (indent sequences, common variable names,
operators) are split into many subwords, wasting context.

This is why modern LLMs with strong coding ability (Codex, DeepSeek-Coder,
StarCoder) train tokenizers on balanced text+code corpora — so that both
domains are compressed efficiently.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Tokenizer Algorithm Comparison

| Property              | BPE (byte-level)     | WordPiece (BERT)       | Unigram LM (SP)        |
|-----------------------|----------------------|------------------------|------------------------|
| Training objective    | Merge by frequency   | Merge by LM likelihood | Prune by LM cost       |
| Direction             | Bottom-up merges     | Bottom-up merges       | Top-down pruning       |
| OOV handling          | Never (byte fallback)| [UNK] token            | Never (char fallback)  |
| Language agnostic?    | With byte-level      | No (pre-tokenization)  | Yes (no pre-tok needed)|
| Used by               | GPT-2/3/4, LLaMA-3   | BERT, DistilBERT       | LLaMA-1/2, T5, Gemma   |
| Vocab continuations   | No special marker    | ## prefix              | ▁ prefix for word-start|
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "BPE From Scratch": {
        "description": "A complete, readable BPE tokenizer built from scratch — training (merge rules) and encoding, no external libraries.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
BYTE-PAIR ENCODING (BPE) TOKENIZER — BUILT FROM SCRATCH
================================================================================

Implements the full BPE algorithm:
    1. Train:  learn merge rules from a corpus
    2. Encode: apply rules to tokenize new text
    3. Decode: convert token IDs back to text

No external libraries. Pure Python for maximum readability.
================================================================================
"""

from collections import defaultdict, Counter
import re


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Pre-tokenization helpers
# ─────────────────────────────────────────────────────────────────────────────

def pretokenize(text: str) -> list[str]:
    """
    Split text into words and attach an end-of-word marker </w>.
    This makes it possible to reconstruct the original text after BPE.

    "hello world" → ["hello</w>", "world</w>"]
    """
    words = re.findall(r'\S+', text)
    return [w + "</w>" for w in words]


def word_to_chars(word: str) -> tuple[str, ...]:
    """Turn a pre-tokenized word into a tuple of individual characters."""
    return tuple(word)   # ("h","e","l","l","o","<","/","w",">")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BPE Training
# ─────────────────────────────────────────────────────────────────────────────

def get_vocab(corpus: list[str]) -> dict[tuple, int]:
    """
    Build initial vocabulary from the corpus.
    Returns {char_tuple: frequency} for every word.
    """
    vocab: dict[tuple, int] = defaultdict(int)
    for word in corpus:
        vocab[word_to_chars(word)] += 1
    return dict(vocab)


def get_pair_freqs(vocab: dict[tuple, int]) -> Counter:
    """Count how often each adjacent pair appears across the corpus."""
    pairs: Counter = Counter()
    for word_chars, freq in vocab.items():
        for i in range(len(word_chars) - 1):
            pairs[(word_chars[i], word_chars[i + 1])] += freq
    return pairs


def merge_vocab(pair: tuple[str, str], vocab: dict[tuple, int]) -> dict[tuple, int]:
    """
    Replace every occurrence of (a, b) in the vocabulary with "ab".
    Returns a new vocabulary with the merge applied.
    """
    a, b = pair
    merged = a + b
    new_vocab: dict[tuple, int] = {}
    for word_chars, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        while i < len(word_chars):
            if i < len(word_chars) - 1 and word_chars[i] == a and word_chars[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word_chars[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def train_bpe(corpus: list[str], num_merges: int, verbose: bool = True):
    """
    Train BPE on a corpus for num_merges iterations.

    Returns:
        merges:     list of (pair) tuples in the order they were applied
        final_vocab: the resulting token set (all unique subwords)
    """
    # Build initial char-level vocabulary
    vocab = get_vocab(corpus)

    # Collect initial character-level tokens
    all_tokens: set[str] = set()
    for word_chars in vocab:
        all_tokens.update(word_chars)

    merges: list[tuple[str, str]] = []

    for step in range(num_merges):
        pairs = get_pair_freqs(vocab)
        if not pairs:
            break

        # Pick the most frequent pair (ties broken alphabetically for stability)
        best = max(pairs, key=lambda p: (pairs[p], p))
        merged_token = best[0] + best[1]

        merges.append(best)
        vocab = merge_vocab(best, vocab)
        all_tokens.add(merged_token)

        if verbose:
            print(f"  Merge {step+1:3d}: {best[0]!r:10} + {best[1]!r:10} → "
                  f"{merged_token!r:15}  (freq={pairs[best]})")

    return merges, sorted(all_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: BPE Encoding (applying learned merges to new text)
# ─────────────────────────────────────────────────────────────────────────────

def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """
    Encode a single pre-tokenized word by applying the learned merge rules
    in the order they were learned.
    """
    tokens = list(word)  # start with individual characters

    for (a, b) in merges:
        i = 0
        new_tokens: list[str] = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a full string into BPE subword tokens."""
    words = pretokenize(text)
    result: list[str] = []
    for word in words:
        result.extend(encode_word(word, merges))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Vocabulary → ID mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_token_to_id(final_vocab: list[str]) -> dict[str, int]:
    """Assign a unique integer ID to every token in the final vocabulary."""
    return {tok: i for i, tok in enumerate(sorted(final_vocab))}


def tokens_to_ids(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    return [token_to_id[t] for t in tokens]


def ids_to_text(ids: list[int], id_to_token: dict[int, str]) -> str:
    tokens = [id_to_token[i] for i in ids]
    return "".join(tokens).replace("</w>", " ").strip()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Training corpus — deliberately repetitive so merges are visible
    corpus_text = (
        "low low low low low "
        "lower lower lower "
        "lowest lowest "
        "newer newer newer "
        "newest newest "
        "wider wide "
    )
    corpus = pretokenize(corpus_text)

    print("=" * 60)
    print("  TRAINING BPE")
    print("=" * 60)
    print(f"  Corpus words: {sorted(set(corpus))}")
    print(f"  Initial vocabulary: {sorted(set(c for w in corpus for c in w))}")
    print()

    merges, final_vocab = train_bpe(corpus, num_merges=15, verbose=True)

    print(f"\n  Final vocabulary ({len(final_vocab)} tokens):")
    print(f"  {final_vocab}")

    # Build ID mappings
    tok2id = build_token_to_id(final_vocab)
    id2tok = {v: k for k, v in tok2id.items()}

    print("\n" + "=" * 60)
    print("  ENCODING NEW STRINGS")
    print("=" * 60)

    test_strings = ["low", "lowest", "newer", "widest", "lowering"]
    for s in test_strings:
        tokens = encode(s, merges)
        ids    = tokens_to_ids(tokens, tok2id)
        print(f"  {s!r:15} → tokens: {tokens}  →  ids: {ids}")

    print("\n" + "=" * 60)
    print("  ROUND-TRIP: IDs → TEXT")
    print("=" * 60)
    sample = "low newest wider"
    tokens = encode(sample, merges)
    ids    = tokens_to_ids(tokens, tok2id)
    back   = ids_to_text(ids, id2tok)
    print(f"  Original:  {sample!r}")
    print(f"  Tokens:    {tokens}")
    print(f"  IDs:       {ids}")
    print(f"  Decoded:   {back!r}")
    print(f"  Match:     {sample.strip() == back.strip()}")
''',
    },

    "Tokenizer Comparison (tiktoken vs sentencepiece)": {
        "description": "Compare GPT-4's tiktoken tokenizer vs LLaMA's SentencePiece tokenizer on the same text — token count, fertility, and edge cases.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
TOKENIZER COMPARISON: tiktoken (GPT-4) vs sentencepiece (LLaMA)
================================================================================

Requires:
    pip install tiktoken sentencepiece

SentencePiece requires a model file. This script uses the LLaMA tokenizer
if available, otherwise falls back to a synthetic example for illustration.
================================================================================
"""

# ── Try to import real tokenizers ─────────────────────────────────────────────
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    print("⚠️  tiktoken not installed. Run: pip install tiktoken")

try:
    import sentencepiece as spm
    sp_available = True
except ImportError:
    sp_available = False
    print("⚠️  sentencepiece not installed. Run: pip install sentencepiece")


# ── Fallback: illustrate concepts without real tokenizers ────────────────────

def illustrate_tokenization_concepts():
    """
    When real tokenizers are unavailable, illustrate key concepts using
    a simple hand-crafted BPE-like tokenizer.
    """
    print("\n" + "=" * 65)
    print("  TOKENIZATION CONCEPTS — ILLUSTRATION")
    print("=" * 65)

    examples = [
        # (description, word-level tokens, subword tokens, char tokens)
        ("Common English word",   ["the"],               ["the"],                  list("the")),
        ("Compound word",         ["tokenization"],      ["token","ization"],       list("tokenization")),
        ("Rare technical term",   ["autoregressive"],    ["auto","regressive"],     list("autoregressive")),
        ("Number",                ["12345"],             ["123","45"],              list("12345")),
        ("Code snippet",          ["def","forward"],     ["def","forward"],         list("defforward")),
        ("German compound",       ["Bundesrepublik"],    ["Bundes","republik"],     list("Bundesrepublik")),
    ]

    print(f"\n  {'Example':<22} {'Word':>5}  {'Subword':>8}  {'Char':>6}")
    print(f"  {'':─<22} {'tokens':>5}  {'tokens':>8}  {'tokens':>6}")
    for desc, word_toks, sub_toks, char_toks in examples:
        print(f"  {desc:<22} {len(word_toks):>5}  {len(sub_toks):>8}  {len(char_toks):>6}")

    print("\n  Key observations:")
    print("  • Subword tokenization is consistently more compact than char-level")
    print("  • Rare/long words split into recognisable morphological pieces")
    print("  • Numbers may split inconsistently — a known arithmetic weakness")

    print("\n" + "=" * 65)
    print("  KNOWN EDGE CASES")
    print("=" * 65)
    edge_cases = [
        ("Leading space matters",
         "' Hello' vs 'Hello' are different tokens in most BPE tokenizers"),
        ("Numbers and arithmetic",
         "'100000' may be ['100', '000'] — positional structure is lost"),
        ("Capitalisation",
         "'the', 'The', ' the', ' The' are often 4 distinct tokens"),
        ("Whitespace runs",
         "'\\n\\n' is often one token; used as paragraph separator signal"),
        ("Code indentation",
         "4 spaces might be one token '    ' or four ' ' ' ' ' ' ' '"),
    ]
    for title, explanation in edge_cases:
        print(f"\n  ⚠️  {title}")
        print(f"     {explanation}")


# ── Real tokenizer comparison (if available) ─────────────────────────────────

def compare_tokenizers():
    if not tiktoken_available:
        illustrate_tokenization_concepts()
        return

    enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 tokenizer

    test_texts = [
        "The transformer architecture is highly parallelisable.",
        "def attention(q, k, v): return softmax(q @ k.T / sqrt(d)) @ v",
        "Bundesverfassungsgericht",                          # German
        "مرحبا بالعالم",                                   # Arabic
        "1234567890",                                        # Numbers
        "   \n\n   ",                                        # Whitespace
        "SolidGoldMagikarp",                                 # Ghost token demo
    ]

    print("=" * 70)
    print("  TIKTOKEN (cl100k_base / GPT-4) TOKENIZATION ANALYSIS")
    print("=" * 70)

    print(f"\n  {'Text':<40} {'# Tokens':>9}  Tokens")
    print(f"  {'':─<40} {'':─>9}  ─────────────────────")
    for text in test_texts:
        ids    = enc.encode(text)
        tokens = [enc.decode([i]) for i in ids]
        display_text = repr(text)[:38]
        display_tokens = str(tokens)[:60]
        print(f"  {display_text:<40} {len(ids):>9}  {display_tokens}")

    # Fertility analysis
    print("\n" + "=" * 70)
    print("  CROSS-LINGUAL FERTILITY  (tokens per word)")
    print("=" * 70)

    sentences = {
        "English":  "The quick brown fox jumps over the lazy dog",
        "German":   "Der schnelle braune Fuchs springt über den faulen Hund",
        "Spanish":  "El rápido zorro marrón salta sobre el perro perezoso",
        "Arabic":   "الثعلب البني السريع يقفز فوق الكلب الكسول",
        "Chinese":  "敏捷的棕色狐狸跳过懒惰的狗",
        "Hindi":    "तेज भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है",
    }

    print(f"\n  {'Language':<12} {'Words':>7}  {'Tokens':>7}  {'Fertility':>10}")
    print(f"  {'':─<12} {'':─>7}  {'':─>7}  {'':─>10}")
    for lang, sentence in sentences.items():
        words  = len(sentence.split())
        tokens = len(enc.encode(sentence))
        fert   = tokens / words if words else 0
        bar    = "█" * int(fert * 4)
        print(f"  {lang:<12} {words:>7}  {tokens:>7}  {fert:>8.2f}×  {bar}")

    print("\n  ℹ️  English fertility ≈ 1.2–1.4×; non-Latin scripts can be 3–5×")
    print("     Higher fertility = fewer 'real words' fit in the context window")


if __name__ == "__main__":
    compare_tokenizers()
''',
    },

    "Chat Template and Special Token Demo": {
        "description": "Show how chat templates wrap conversation turns with special tokens — the format that instruction-tuned models expect.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
CHAT TEMPLATES AND SPECIAL TOKENS
================================================================================

Instruction-tuned LLMs expect conversations to be formatted with special
tokens that mark role boundaries (system / user / assistant).

A mismatch between the training chat template and the inference format is a
silent performance killer — the model has never seen this token pattern during
training.

This script shows the templates used by GPT (ChatML), LLaMA-3, and Mistral,
and implements a minimal template renderer from scratch.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Literal


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role:    Role
    content: str


# ── Template definitions ───────────────────────────────────────────────────────

TEMPLATES = {

    "ChatML (GPT-4 / Mistral Instruct)": {
        "description": (
            "OpenAI's ChatML format. Each turn is wrapped by <|im_start|> / <|im_end|>. "
            "Widely adopted as a de-facto standard."
        ),
        "render": lambda msgs: _render_chatml(msgs),
    },

    "LLaMA-3": {
        "description": (
            "Meta's LLaMA-3 format. Uses <|begin_of_text|>, header tags, "
            "and <|eot_id|> (end-of-turn) separators."
        ),
        "render": lambda msgs: _render_llama3(msgs),
    },

    "LLaMA-2 / Mistral v0.1": {
        "description": (
            "LLaMA-2 and early Mistral format. System prompt is injected inside "
            "the first user [INST] block. No explicit role tags for assistant."
        ),
        "render": lambda msgs: _render_llama2(msgs),
    },

    "Alpaca": {
        "description": (
            "Simple instruction-following format from Stanford Alpaca. "
            "No special tokens — just natural language markers."
        ),
        "render": lambda msgs: _render_alpaca(msgs),
    },
}


def _render_chatml(messages: list[Message]) -> str:
    out = []
    for msg in messages:
        out.append(f"<|im_start|>{msg.role}\\n{msg.content}<|im_end|>")
    out.append("<|im_start|>assistant\\n")   # generation prompt
    return "\\n".join(out)


def _render_llama3(messages: list[Message]) -> str:
    out = ["<|begin_of_text|>"]
    for msg in messages:
        out.append(
            f"<|start_header_id|>{msg.role}<|end_header_id|>\\n\\n"
            f"{msg.content}<|eot_id|>"
        )
    out.append("<|start_header_id|>assistant<|end_header_id|>\\n\\n")
    return "\\n".join(out)


def _render_llama2(messages: list[Message]) -> str:
    """
    LLaMA-2: [INST] ... [/INST] for user turns.
    System prompt is prepended inside the first [INST] block as <<SYS>>...
    """
    out   = []
    sys   = next((m.content for m in messages if m.role == "system"), None)
    turns = [m for m in messages if m.role != "system"]

    for i, msg in enumerate(turns):
        if msg.role == "user":
            prefix = ""
            if i == 0 and sys:
                prefix = f"<<SYS>>\\n{sys}\\n<</SYS>>\\n\\n"
            out.append(f"<s>[INST] {prefix}{msg.content} [/INST]")
        elif msg.role == "assistant":
            out.append(f"{msg.content} </s>")
    return " ".join(out)


def _render_alpaca(messages: list[Message]) -> str:
    sys   = next((m.content for m in messages if m.role == "system"), "")
    turns = [m for m in messages if m.role != "system"]
    out   = []
    if sys:
        out.append(sys + "\\n")
    for msg in turns:
        if msg.role == "user":
            out.append(f"### Instruction:\\n{msg.content}\\n")
        elif msg.role == "assistant":
            out.append(f"### Response:\\n{msg.content}\\n")
    out.append("### Response:\\n")  # generation prompt
    return "\\n".join(out)


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conversation = [
        Message("system",    "You are a helpful AI assistant specialising in machine learning."),
        Message("user",      "What is byte-pair encoding?"),
        Message("assistant", "BPE is a subword tokenization algorithm that starts with individual characters and iteratively merges the most frequent adjacent pairs."),
        Message("user",      "Why is it preferred over word-level tokenization?"),
    ]

    for template_name, template in TEMPLATES.items():
        print("=" * 65)
        print(f"  {template_name}")
        print(f"  {template['description']}")
        print("=" * 65)
        rendered = template["render"](conversation)
        print(rendered)
        print()

    # Show special token inventory
    print("=" * 65)
    print("  SPECIAL TOKEN INVENTORY")
    print("=" * 65)
    special_tokens = [
        ("<|endoftext|>",          "GPT-2/3",    "End of document"),
        ("<|im_start|>",           "ChatML",     "Start of a conversation turn"),
        ("<|im_end|>",             "ChatML",     "End of a conversation turn"),
        ("<|begin_of_text|>",      "LLaMA-3",    "Start of the full sequence"),
        ("<|eot_id|>",             "LLaMA-3",    "End of turn"),
        ("<|start_header_id|>",    "LLaMA-3",    "Start of role header"),
        ("<|end_header_id|>",      "LLaMA-3",    "End of role header"),
        ("<s>",                    "LLaMA-2",    "Beginning of sequence"),
        ("</s>",                   "LLaMA-2",    "End of sequence"),
        ("[INST]",                 "LLaMA-2",    "Start of user instruction"),
        ("[/INST]",                "LLaMA-2",    "End of user instruction"),
        ("<<SYS>>",                "LLaMA-2",    "Start of system prompt (inside [INST])"),
        ("[PAD]",                  "General",    "Padding to equal sequence lengths"),
        ("[MASK]",                 "BERT",       "Masked position for MLM objective"),
    ]

    print(f"  {'Token':<28} {'Model':<12} Description")
    print(f"  {'':─<28} {'':─<12} ─────────────────────────")
    for tok, model, desc in special_tokens:
        print(f"  {tok:<28} {model:<12} {desc}")
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
    visual_html   = ""
    visual_height = 600
    # try:
    #     from llm_training.visuals.tokenization import (
    #         TOKENIZATION_VISUAL_HTML,
    #         TOKENIZATION_VISUAL_HEIGHT,
    #     )
    #     visual_html   = TOKENIZATION_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = TOKENIZATION_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(f"[02_tokenization_vocabulary_bpe.py] Could not load visual: {e}", stacklevel=2)

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