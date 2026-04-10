"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""

TOPIC_NAME = "LLM - Tokenisation"
DISPLAY_NAME = "LLM - Tokenisation"
ICON         = "🔤"
SUBTITLE     = "How text becomes numbers — and why it matters for everything"
CATEGORY = "LLM Architecture"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""

Order:

    Module_Concepts · Tokenisation
          ├── what_is_a_token            Characters vs. words vs. subwords
          ├── bpe_tokenisation           Byte Pair Encoding — how vocabulary is built
          ├── tokeniser_vocabulary       Vocab size trade-offs (GPT-2: 50k, LLaMA: 32k)
          ├── token_counting             Why "1 word ≈ 1.3 tokens" and when it breaks
          ├── tokenisation_edge_cases    Numbers, code, non-English, emojis
          ├── special_tokens             <BOS>, <EOS>, <PAD>, [INST] — roles and positions
          └── tokenisation_and_cost      Token count = compute cost = API billing unit


## 1. WHAT IS A TOKEN — Characters vs. Words vs. Subwords

Before any neural network can process text, every part of that text must become
a number. A neural network cannot operate on the string "hello" — it operates on
integers, which are looked up in an embedding table to produce the dense floating-
point vectors that actually flow through the layers.

A TOKEN is that basic unit of conversion: a chunk of text that maps to exactly
one integer ID in a fixed lookup table called the vocabulary.

The fundamental design question is: what should a "chunk" be?
Three natural answers exist, each with genuine trade-offs.


## 1.1  The Three Approaches to Tokenisation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  APPROACH 1 — CHARACTER-LEVEL TOKENISATION                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      Every single printable character gets its own token ID.
      Vocabulary: roughly 100–300 entries.

      "Hello, world!" tokenised:
       H   e   l   l   o   ,       w   o   r   l   d   !
      [H] [e] [l] [l] [o] [,] [ ] [w] [o] [r] [l] [d] [!]
       ↑
       13 tokens for 13 characters — a 1-to-1 mapping

      "antidisestablishmentarianism" → 28 tokens  (one per character)

      ┌────────────────────────────────────────────────────────────┐
      │  PROS                          CONS                        │
      │  ──────────────────────────    ──────────────────────────  │
      │  Tiny vocabulary (~300)        Sequences very long         │
      │  Universal coverage            Quadratic attention cost    │
      │  No unknown-word problem       Model learns spelling       │
      │  Works for any language        from scratch                │
      └────────────────────────────────────────────────────────────┘


    ╔══════════════════════════════════════════════════════════════════╗
    ║  APPROACH 2 — WORD-LEVEL TOKENISATION                            ║
    ╚══════════════════════════════════════════════════════════════════╝

      Every distinct word (and punctuation mark) gets its own token ID.
      Vocabulary: 100,000–500,000+ entries for open-domain text.

      "Hello, world!" tokenised:
      [Hello] [,] [world] [!]
       4 tokens.

      "antidisestablishmentarianism" → [antidisestablishmentarianism]
       1 token. But this word might appear only ~10 times in the entire
       training corpus, giving its embedding almost no training signal.

      ┌────────────────────────────────────────────────────────────┐
      │  PROS                          CONS                        │
      │  ──────────────────────────    ──────────────────────────  │
      │  Short sequences               Vocabulary explodes         │
      │  Each token = one word         Rare words poorly trained   │
      │  Easy to reason about          <UNK> collapses all         │
      │                                unknown words to one ID     │
      │                                "run"/"runs"/"running"/     │
      │                                "ran" = 4 separate slots    │
      └────────────────────────────────────────────────────────────┘


    ╔══════════════════════════════════════════════════════════════════╗
    ║  APPROACH 3 — SUBWORD TOKENISATION (THE PRODUCTION STANDARD)     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Common words get one token.
      Rare words split into recognisable pieces.
      Vocabulary: 30,000–150,000 entries.

      "Hello, world!" tokenised (GPT-4 cl100k):
      [Hello] [,] [ world] [!]
       4 tokens.

      "antidisestablishmentarianism" tokenised:
      [anti] [dis] [establish] [ment] [arian] [ism]
       6 tokens — each a known morpheme with its own training signal.

      "GPU" → [GPU]   (frequent enough to earn a single slot)

      "unhappiness" → [un] [happiness]   or   [unhappy] [ness]
       (depends on what the training corpus made frequent)

      ┌────────────────────────────────────────────────────────────┐
      │  PROS                          CONS                        │
      │  ──────────────────────────    ──────────────────────────  │
      │  Balanced vocab + seq length   Boundaries feel arbitrary   │
      │  No <UNK> possible             " hello" ≠ "hello"          │
      │  Morphology preserved          (space is part of token)    │
      │  Covers all languages          Numbers and code are        │
      │  via byte fallback             often expensive             │
      └────────────────────────────────────────────────────────────┘


## 1.2  Why Sequence Length Determines Everything

Transformer attention requires every token to compute a score against every
other token in the sequence. This produces an N × N matrix where N is the
number of tokens. Cost is quadratic in sequence length.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ATTENTION COST — THE SAME SENTENCE, THREE TOKENISATIONS         ║
    ╚══════════════════════════════════════════════════════════════════╝

      Sentence: "The transformer architecture changed NLP forever."
      (52 characters, 7 words)

      ┌────────────────────────────────────────────────────────────────┐
      │  Method          Tokens   Attention matrix    Relative cost    │
      ├────────────────────────────────────────────────────────────────┤
      │  Character-level   52      52  × 52  = 2,704  baseline         │
      │  Subword (BPE)      8       8  ×  8  =    64   42× cheaper     │
      │  Word-level         7       7  ×  7  =    49   55× cheaper     │
      └────────────────────────────────────────────────────────────────┘

      Scaling to a 10,000-character document (~2,000 words):

        Character-level:  ~10,000 tokens  →  100,000,000 attention cells
        Subword BPE:       ~2,500 tokens  →    6,250,000 attention cells
                                              ↑ 16× cheaper per head

      This is why character-level tokenisation is never used in production
      LLMs. The attention cost at even moderate context lengths becomes
      computationally and financially untenable.


## 1.3  The Vocabulary–Sequence-Length Trade-Off Curve

As vocabulary size increases, each token covers more text, which reduces
the number of tokens needed — but the embedding table and LM head grow
proportionally with vocabulary size.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TOKEN COUNT FOR A 100-CHARACTER ENGLISH SENTENCE BY VOCAB SIZE  ║
    ╚══════════════════════════════════════════════════════════════════╝

      Vocab 256    (byte-level)  ████████████████████████████████░░░  ~100 tokens
      Vocab 1k                   ████████████████████████░░░░░░░░░░░   ~65 tokens
      Vocab 10k                  █████████████████░░░░░░░░░░░░░░░░░░   ~30 tokens
      Vocab 32k    (LLaMA 2)     ████████████░░░░░░░░░░░░░░░░░░░░░░░   ~22 tokens
      Vocab 50k    (GPT-2)       ██████████░░░░░░░░░░░░░░░░░░░░░░░░░   ~19 tokens
      Vocab 100k   (GPT-4)       █████████░░░░░░░░░░░░░░░░░░░░░░░░░░   ~17 tokens
      Vocab 256k   (Gemma)       ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~14 tokens
                                          ↑
                                  diminishing returns above ~100k

      The memory cost of the embedding table:
      vocab_size × d_model parameters, stored in fp16 (2 bytes each)

        50,257  × 4096 dims × 2 bytes ≈  400 MB    (GPT-2 style)
       100,256  × 4096 dims × 2 bytes ≈  800 MB    (GPT-4 style)
       128,256  × 4096 dims × 2 bytes ≈  1.0 GB    (LLaMA 3)

      The LM head projection uses the same matrix in reverse.
      For tied embeddings this is free; for separate weights it doubles the cost.

      Production sweet spot: 32k–128k tokens.
      Below 32k: multilingual coverage degrades; non-Latin text fragments badly.
      Above 256k: embedding tables consume significant memory with little gain.

**KEY CONCEPT:** Tokens are not words, not characters, and not sentences. They
are the atom of LLM computation. Everything — context window limits, API
pricing, attention complexity, embedding memory — is denominated in tokens.


---


## 2. BPE TOKENISATION — How Vocabulary Is Built

The dominant tokenisation algorithm used in GPT-style models is Byte Pair
Encoding (BPE). The name comes from a 1994 data compression algorithm, but
the modern LLM version is meaningfully different.

BPE gives us a deterministic, learned vocabulary that emerges from the
statistical structure of the training corpus — no linguistic knowledge required.


## 2.1  The UTF-8 Foundation

Modern LLM tokenisers operate on UTF-8 bytes, not characters.

Why bytes?

  Every possible text input — English, Chinese, Arabic, emoji, code,
  binary-encoded data, intentionally malformed sequences — can be
  expressed as a sequence of bytes (values 0–255).

  By starting with all 256 possible byte values as the initial vocabulary,
  the tokeniser has total coverage. There is no "unknown" input. Any text
  that can exist can be tokenised — and detokenised back losslessly.

Before BPE can run, the text undergoes Unicode normalisation (typically NFC
or NFKC) to canonicalise composed vs. decomposed character forms. The
normalised string is then encoded to raw UTF-8 bytes.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  UTF-8 ENCODING — WHAT "café" LOOKS LIKE IN BYTES                ║
    ╚══════════════════════════════════════════════════════════════════╝

      c   →  0x63  (1 byte  — standard ASCII)
      a   →  0x61  (1 byte)
      f   →  0x66  (1 byte)
      é   →  0xC3 0xA9  (2 bytes — outside ASCII range)

      "café" as a byte sequence: [0x63, 0x61, 0x66, 0xC3, 0xA9]
                                   c     a     f    é (byte 1) é (byte 2)

      The initial BPE vocabulary assigns a unique ID to each of the 256
      possible byte values, so "café" starts as 5 token IDs before any
      merge rules are applied.

      After merges, common sequences like "ca" and "fe" may merge into
      single tokens, compressing the representation.


## 2.2  BPE Training — Building the Vocabulary

BPE training takes a large text corpus and iteratively builds a vocabulary
by merging the most frequent adjacent symbol pairs.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FULL WORKED EXAMPLE — BPE TRAINING FROM SCRATCH                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      CORPUS (simplified):
      ┌─────────────────────────────────────────────────────────────┐
      │  "low"    appears 5 times                                   │
      │  "lower"  appears 2 times                                   │
      │  "newest" appears 6 times                                   │
      │  "wider"  appears 3 times                                   │
      └─────────────────────────────────────────────────────────────┘

      STEP 0 — Initial vocabulary: all unique characters (bytes)
      ─────────────────────────────────────────────────────────────
      { l, o, w, e, r, n, s, t, i, d }   (10 symbols from this corpus)

      Words split into characters (</w> marks end-of-word boundary):

        "low"    ×5  →  [l] [o] [w] </w>
        "lower"  ×2  →  [l] [o] [w] [e] [r] </w>
        "newest" ×6  →  [n] [e] [w] [e] [s] [t] </w>
        "wider"  ×3  →  [w] [i] [d] [e] [r] </w>

      Count ALL adjacent pairs (weighted by word frequency):

        l·o   →  5 + 2 = 7     ← winner — most frequent pair
        o·w   →  5 + 2 = 7     ← tied
        w·</w>→  5              
        w·e   →  2 + 6 = 8     ← also a strong candidate
        e·r   →  2 + 3 = 5
        e·w   →  6
        ...

      (when tied, the pair that appears first in the vocabulary order wins)

      ITERATION 1 — merge most frequent pair: (w, e) → we
      ─────────────────────────────────────────────────────────────
        "low"    ×5  →  [l] [o] [w] </w>        (no 'we' present)
        "lower"  ×2  →  [l] [o] [we] [r] </w>   ← w+e merged
        "newest" ×6  →  [n] [e] [we] [s] [t] </w> ← w+e merged
        "wider"  ×3  →  [w] [i] [d] [e] [r] </w>  (no 'we' present)

      Vocabulary now: { l, o, w, e, r, n, s, t, i, d, we }
      Merge rule #1: (w, e) → we

      ITERATION 2 — recount pairs, find next winner
      ─────────────────────────────────────────────────────────────
      Count pairs again:

        l·o     →  5 + 2  = 7  ← winner
        o·w     →  5
        n·e     →  6
        e·we    →  6
        we·s    →  6
        o·we    →  2
        ...

      Merge pair: (l, o) → lo
      Merge rule #2: (l, o) → lo

      ITERATION 3
      ─────────────────────────────────────────────────────────────
        "low"    ×5  →  [lo] [w] </w>
        "lower"  ×2  →  [lo] [we] [r] </w>
        "newest" ×6  →  [n] [e] [we] [s] [t] </w>
        "wider"  ×3  →  [w] [i] [d] [e] [r] </w>

      Count pairs. Winner: (lo, w) → low  (count = 5 + 2 = 7)
      Merge rule #3: (lo, w) → low

      Vocabulary now: { l, o, w, e, r, n, s, t, i, d, we, lo, low }

      ...continue until vocabulary reaches target size (e.g. 50,257)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHAT THE TRAINING PRODUCES                                      ║
    ╚══════════════════════════════════════════════════════════════════╝

      After all iterations complete, training produces two artefacts:

      1. VOCABULARY FILE (vocab.json)
         A dictionary mapping every known symbol to an integer ID.

         {
           "!": 0,
           "\"": 1,
           "#": 2,
           ...
           "the": 262,
           " the": 318,
           ...
           "low": 9831,
           "lower": 9832,
           "lowest": 11343,
           ...
         }

      2. MERGE FILE (merges.txt)
         An ordered list of all merge rules, one per line.
         Order matters — rules are applied in the order they were learned.

         w e           ← rule #1
         l o           ← rule #2
         lo w          ← rule #3
         n e           ← rule #4
         ...

      These two files together fully define the tokeniser.
      Given any new text, the tokeniser applies the merge rules in
      order to produce integer IDs. No other information is needed.


## 2.3  Pre-Tokenisation — The Often-Overlooked Step

Before BPE runs, text is split by a pre-tokenisation regex that prevents
merges from crossing certain boundaries.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY PRE-TOKENISATION EXISTS                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Without pre-tokenisation, BPE might merge across word boundaries:
      "the cat" → the pair (e, ·) (where · is a space) might merge,
      producing tokens that span word boundaries like "the " or " c".

      While not technically wrong, this has an undesirable consequence:
      the model cannot distinguish "end of word" from "start of word"
      in the resulting tokens.

      The solution: run a regex first that splits text into "pre-tokens"
      (chunks). BPE can only merge WITHIN a pre-token, never across one.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  GPT-4's CL100K REGEX (SIMPLIFIED)                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      GPT-4's tokeniser (cl100k_base) uses a regex that splits on:

        • Contractions ("'s", "'t", "'re", "'ve", "'ll", "'d")
        • Optional space + word characters
        • Optional space + non-word, non-space characters (punctuation)
        • Whitespace
        • Non-whitespace sequences that don't fit above

      EFFECT ON "Hello, world!":

        Pre-tokeniser splits: ["Hello", ",", " world", "!"]
                                                ↑
                                         space is PART of " world"
                                         not a separate token

      EFFECT ON "Hello  world" (double space):

        Pre-tokeniser splits: ["Hello", "  ", "world"]
                               The two spaces become their own pre-token.

      KEY CONSEQUENCE — SPACE IS PART OF THE TOKEN:

        " world" (with leading space)  →  different token from  "world"
        " the"   (with leading space)  →  different token from  "the"

        ┌────────────────────────────────────────────────────────────┐
        │  Token     ID (cl100k)    Appears in                       │
        │  "the"     1820           at start of text or after space  │
        │  " the"    279            in mid-sentence (after a word)   │
        │  "The"     791            capitalised, start of word       │
        │  " The"    578            capitalised, mid-sentence        │
        └────────────────────────────────────────────────────────────┘

        These are FOUR DIFFERENT TOKENS representing the same word in
        four different positional contexts. This is by design — the
        leading space encodes positional information.


## 2.4  BPE Encoding — Tokenising New Text at Inference

At inference time, new text must be encoded using the frozen vocabulary
and merge rules from training.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ENCODING ALGORITHM — STEP BY STEP                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      Input: "lower"

      STEP 1 — Normalise and UTF-8 encode
        "lower" → bytes [0x6C, 0x6F, 0x77, 0x65, 0x72]
        As characters: l, o, w, e, r

      STEP 2 — Pre-tokenisation regex
        "lower" contains no apostrophes, no spaces → stays as one chunk.
        Pre-token: ["lower"]

      STEP 3 — Start with byte-level tokens
        [l] [o] [w] [e] [r]

      STEP 4 — Apply merge rules IN ORDER from the merge file

        Rule #1: (w, e) → we
          [l] [o] [we] [r]         ← w and e merged

        Rule #2: (l, o) → lo
          [lo] [we] [r]            ← l and o merged

        Rule #3: (lo, w) → low
          No 'w' adjacent to 'lo' in current sequence → skip

        Rule #4: (n, e) → ne
          No 'n' present → skip

        Rule #5: (lo, we) → lowe   (hypothetical, if this rule exists)
          [lowe] [r]

        Rule #6: (lowe, r) → lower  (hypothetical)
          [lower]

        If "lower" was in the training corpus frequently enough,
        it becomes a single token in the final vocabulary.

      STEP 5 — Look up final symbols in vocabulary dictionary
        [lower] → ID 2793   (for example)

      Output: [2793]
      The string "lower" encoded as a single integer.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ENCODING "lowest" (WORD NOT IN VOCABULARY AS A UNIT)            ║
    ╚══════════════════════════════════════════════════════════════════╝

      If "lowest" was rare in training, it might not have its own entry.

      STEP 3: [l] [o] [w] [e] [s] [t]

      Apply merges:
        (w, e) → we:       [l] [o] [we] [s] [t]
        (l, o) → lo:       [lo] [we] [s] [t]
        (lo, we) → lowe:   [lowe] [s] [t]
        (s, t) → st:       [lowe] [st]        (if this merge exists)

      Final: ["lowe", "st"]  →  [ID_lowe, ID_st]
      2 tokens for "lowest".

      This is the sub-word decomposition in action — the word didn't
      need to be in the training corpus as a unit to be tokenised.


## 2.5  SentencePiece and the LLaMA Family

LLaMA-family models (and many multilingual models) use SentencePiece, a
tokenisation library that supports both BPE and a different algorithm called
Unigram Language Model.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  KEY DIFFERENCES: SENTENCEPIECE vs. TIKTOKEN (GPT-STYLE)         ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌──────────────────────┬────────────────────────────────────────┐
      │  Feature             │  SentencePiece         Tiktoken (BPE)  │
      ├──────────────────────┼──────────────────────────────────────────┤
      │  Used in             │  LLaMA, T5, ALBERT     GPT-2/3/4        │
      │  Space handling      │  ▁ prefix in token     Leading space in │
      │                      │  ("▁hello" = " hello") token ("_hello") │
      │  Pre-tokenisation    │  Treats raw text as    Regex splits     │
      │                      │  stream of chars       before BPE       │
      │  Byte fallback       │  <0xNN> special tokens  Byte tokens     │
      │                      │  for unknown bytes     in vocabulary    │
      │  Algorithm options   │  BPE or Unigram LM     BPE only         │
      └──────────────────────┴──────────────────────────────────────────┘

      SENTENCEPIECE SPACE CONVENTION:

        "Hello world" → [▁Hello] [▁world]
                         ↑
                    ▁ marks "this token begins after a space"
                    The ▁ is inside the token, not a separator.

      TIKTOKEN SPACE CONVENTION:

        "Hello world" → [Hello] [ world]
                                  ↑
                    The space is the first character of " world"

      Both encode the same positional information, just differently.


---


## 3. TOKENISER VOCABULARY — Size Trade-Offs Across Models

The vocabulary is the complete mapping from every known string to an integer ID.
Its size is one of the most consequential design decisions in LLM architecture.


## 3.1  What a Vocabulary Entry Actually Is

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ANATOMY OF A VOCABULARY                                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      The vocabulary is two parallel structures:

      1. TOKEN → ID  (the encoder — used when processing input)
         A hash table mapping every known byte-sequence to an integer.

         " the"    →   279
         "Hello"   →  15496
         " world"  →   995
         " Python" →  11361
         " GPU"    →  19904
         ...

      2. ID → TOKEN  (the decoder — used when generating output)
         An array where index i contains the byte-sequence for token i.

         index 279   → bytes [0x20, 0x74, 0x68, 0x65] = " the"
         index 15496 → bytes [0x48, 0x65, 0x6C, 0x6C, 0x6F] = "Hello"
         ...

      The vocabulary size is the length of this array.
      Token IDs are integers from 0 to (vocab_size - 1).

      The vocabulary file for GPT-4 (cl100k_base) is ~1.7 MB on disk.
      The merge rules file is ~1.1 MB.
      Together they define the complete tokeniser.


## 3.2  Major Tokenisers and Their Vocabulary Sizes

    +─────────────────────────────────────────────────────────────────────────+
    │  Model Family      Tokeniser        Vocab Size  Algorithm   Released    │
    +─────────────────────────────────────────────────────────────────────────+
    │  GPT-2             gpt2             50,257      BPE         2019        │
    │  GPT-3, GPT-3.5    p50k_base        50,281      BPE         2020        │
    │  GPT-4, GPT-3.5T   cl100k_base     100,256      BPE         2023        │
    │  GPT-4o            o200k_base      200,019      BPE         2024        │
    │  LLaMA 1 & 2       llama           32,000       SentPiece   2023        │
    │  LLaMA 3           llama3         128,256       tiktoken    2024        │
    │  Mistral 7B        mistral         32,000       SentPiece   2023        │
    │  Gemma             gemma          256,000       SentPiece   2024        │
    │  Claude (Anthropic)  —            ~100k range   BPE         —           │
    +─────────────────────────────────────────────────────────────────────────+

    Note: GPT-2's vocab is 50,257 rather than a round number because the
    original 50,000 BPE merges + 256 base byte tokens + 1 special <|endoftext|>
    token add up to 50,257.


## 3.3  Why GPT-4 Doubled the Vocabulary vs. GPT-2

Moving from 50k to 100k was not arbitrary. Each improvement was deliberate:

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE GAINS FROM A LARGER VOCABULARY                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. BETTER CODE COVERAGE
         ─────────────────────
         Programming identifiers, common function names, and boilerplate
         syntax are much more likely to get single tokens in cl100k.

         GPT-2 (p50k):  "def "     →  ["def", " "]     2 tokens
         GPT-4 (cl100k): "def "    →  ["def "]          1 token

         GPT-2:   "self."    →  ["self", "."]           2 tokens
         GPT-4:   "self."    →  ["self."]               1 token

         Common Python patterns like "import ", "return ", "class "
         are all single tokens in cl100k. This reduces sequence length
         for code by roughly 15–25%.

      2. BETTER NUMBER HANDLING
         ─────────────────────
         cl100k tokenises multi-digit integers more efficiently.
         Digits 0–9 are single tokens. Two-digit numbers have tokens.
         Common 3- and 4-digit numbers may have dedicated tokens.

      3. BETTER MULTILINGUAL COVERAGE
         ──────────────────────────────
         With 100k slots, more non-Latin characters earn single tokens.
         Japanese hiragana, katakana, and common kanji each get tokens.
         Common Chinese characters get single-token entries.
         This reduces sequence length for non-Latin text significantly.

      4. WHITESPACE AND INDENTATION
         ─────────────────────────
         cl100k has tokens for common repeated-space patterns:
         "    " (4 spaces), "        " (8 spaces), etc.
         Critical for Python code where indentation is syntax.


## 3.4  Memory Cost of the Embedding Table

The vocabulary size directly determines one of the model's largest parameter
blocks — the token embedding table.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  EMBEDDING TABLE MEMORY — WORKED CALCULATION                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Formula:
        parameters = vocab_size × d_model
        memory (fp16) = parameters × 2 bytes

      For a 7B LLaMA 3 model (d_model = 4096, vocab = 128,256):

        parameters = 128,256 × 4,096 = 525,336,576  (~525M parameters)
        memory     = 525M × 2 bytes  = 1.05 GB

      This single matrix is ~7.5% of the entire 7B model's parameter count.
      For the 70B model (d_model = 8192, vocab = 128,256):

        parameters = 128,256 × 8,192 = 1,050,673,152  (~1.05B parameters)
        memory     = 1.05B × 2 bytes = 2.1 GB

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE LM HEAD IS ALSO VOCABULARY-SIZED                            ║
    ╚══════════════════════════════════════════════════════════════════╝

      The LM head (final output projection) maps from d_model → vocab_size.
      Its shape is the TRANSPOSE of the embedding table.

        Shape: d_model × vocab_size  =  4096 × 128,256

      In many architectures this matrix is TIED (weight sharing):
      the same matrix is used for both input embedding and output projection.
      This saves 1 GB of VRAM and encourages coherent input/output representations.

      When weights are NOT tied (some larger models), both matrices exist
      in VRAM simultaneously, doubling the vocabulary-related memory cost.


## 3.5  Vocabulary Coverage Across Languages

A fixed vocabulary trained predominantly on English will be more efficient
for English text than for other languages.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TOKENS PER WORD — ENGLISH vs. OTHER LANGUAGES (cl100k)          ║
    ╚══════════════════════════════════════════════════════════════════╝

      "Hello, how are you?"  (English, 5 words):
      [Hello] [,] [ how] [ are] [ you] [?]
       6 tokens — ~1.2 tokens per word

      "Hallo, wie geht es Ihnen?"  (German, 5 words):
      [Hall] [o] [,] [ wie] [ ge] [ht] [ es] [ Ih] [nen] [?]
       10 tokens — ~2.0 tokens per word
       (German compound words and less frequent forms fragment more)

      "こんにちは、お元気ですか？"  (Japanese):
       15 tokens for a 6-word equivalent
       ~2.5 tokens per "word" unit

      Arabic script (morphologically rich, right-to-left):
       Often 3–5× more tokens than equivalent English text
       in models with English-dominant vocabularies

      This has practical consequences:
        • Non-English users get less context per context window
        • Non-English API calls cost more per word of content
        • Models trained on English-heavy data may reason less
          efficiently about other languages (shorter effective context)

      LLaMA 3's 128k vocabulary and Gemma's 256k vocabulary both
      explicitly improve multilingual coverage relative to 32k/50k tokenisers.


---


## 4. TOKEN COUNTING — Why "1 Word ≈ 1.3 Tokens" and When It Breaks

In well-written English prose, a useful rule of thumb is:
  1 word  ≈  1.3 tokens
  1 token ≈  0.75 words  (approximately 3/4 of a word)

Understanding where this comes from — and where it fails — is essential for
estimating context window usage and API costs before making API calls.


## 4.1  Deriving the Ratio

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE CHAIN OF APPROXIMATIONS                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Step 1 — Average English word length:
        Most used English words: 4–5 characters
        (function words "the", "and", "is" → 2-3 chars)
        (content words "language", "model" → 5-8 chars)
        Weighted average for running prose: ~4.7 characters per word

      Step 2 — Average characters per token (English prose, cl100k):
        Measured empirically: 3.5–4.5 characters per token
        Using midpoint: ~3.8 characters per token

      Step 3 — Tokens per word:
        tokens_per_word = chars_per_word ÷ chars_per_token
                        = 4.7 ÷ 3.8
                        ≈ 1.24 ≈ 1.3

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WORKED EXAMPLES — DIFFERENT TEXT TYPES                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      TEXT 1: Plain English prose
      ────────────────────────────
      "The large language model generates text by predicting the most
       likely next token given all preceding tokens in the context window."
       (24 words, 133 characters)

      Actual tokenisation (cl100k):
      [The][ large][ language][ model][ generates][ text][ by][ predicting]
      [ the][ most][ likely][ next][ token][ given][ all][ preceding]
      [ tokens][ in][ the][ context][ window][.]
       22 tokens for 24 words  →  0.92 tokens/word
       Chars per token: 133 ÷ 22 = 6.0  (higher than average — simple words)

      TEXT 2: Technical prose with jargon
      ─────────────────────────────────────
      "Rotary positional embeddings encode sequence position as a phase
       rotation applied to query and key vectors before dot-product attention."
       (22 words, 145 characters)

      Actual tokenisation:
      [Rot][ary][ positional][ embed][dings][ encode][ sequence][ position]
      [ as][ a][ phase][ rotation][ applied][ to][ query][ and][ key]
      [ vectors][ before][ dot][–][product][ attention][.]
       24 tokens for 22 words  →  1.09 tokens/word

      TEXT 3: Academic/rare vocabulary
      ──────────────────────────────────
      "Antidisestablishmentarianism encompasses pneumonoultramicroscopicsilicovolcanoconiosis."
       (2 "words", 85 characters)

      Tokenisation:
      [Anti][dis][establishment][arian][ism][ encompass][es]
      [ pne][um][on][oul][tr][am][icro][sc][opic][sil][ico][vol][can][oc][on][ios][is][.]
       28 tokens for 2 "words"  →  14 tokens/word
       Rare words fragment severely.


## 4.2  The 1.3 Rule — When It Holds and When It Doesn't

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ACCURACY OF THE 1.3 TOKENS/WORD APPROXIMATION BY TEXT TYPE      ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌─────────────────────────────────┬────────────┬──────────────────────┐
      │  Text Type                      │ Tok / Word │  Notes               │
      ├─────────────────────────────────┼────────────┼──────────────────────┤
      │  Common English prose           │  1.2–1.4   │  Rule holds well     │
      │  Business/formal English        │  1.2–1.5   │  Rule holds          │
      │  Technical docs (common vocab)  │  1.3–1.6   │  Close               │
      │  Legal / medical text           │  1.5–2.0   │  Long Latin terms    │
      │  Code (Python, with spaces)     │  2.0–4.0   │  Rule BREAKS         │
      │  JSON / structured data         │  2.0–5.0   │  Rule BREAKS         │
      │  Numbers and math               │  1.5–6.0   │  Rule BREAKS badly   │
      │  Non-Latin scripts              │  3.0–8.0   │  Rule BREAKS badly   │
      │  Emojis                         │  1.0–3.0   │  Per emoji           │
      └─────────────────────────────────┴────────────┴──────────────────────┘

      The rule works specifically for common English prose.
      For anything else, estimate conservatively or measure directly.


## 4.3  Context Window Planning in Characters

Knowing the chars-per-token ratio lets you estimate whether content will
fit in a context window without actually calling the tokeniser.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CAPACITY ESTIMATES FOR COMMON CONTEXT WINDOWS                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      Chars per token for English prose: ~3.8 (use 4 for easy maths)

      ┌───────────────────────────────────────────────────────────────────┐
      │  Context window  Tokens    Chars (÷4)    Pages (÷3000 chars/pg)  │
      ├───────────────────────────────────────────────────────────────────┤
      │  GPT-2             1,024       ~4k       ~1.3 pages              │
      │  GPT-4-turbo     128,000     ~512k       ~171 pages              │
      │  Claude 3.5      200,000     ~800k       ~267 pages              │
      │  Gemini 1.5 Pro 1,000,000   ~4,000k     ~1,333 pages            │
      └───────────────────────────────────────────────────────────────────┘

      Practical conversion:
        English word count  ÷  750  ≈  token count in thousands
        "A 15,000-word document is roughly 20,000 tokens."

      For non-English or technical content: divide by 500 instead of 750
      to be conservative.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ALWAYS LEAVE ROOM FOR THE RESPONSE                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      The context window is shared between INPUT and OUTPUT.

      If the model has a 128k context window and your prompt is 100k tokens,
      only 28k tokens remain for the response.
      If max_tokens is set to 4096, the model can generate up to 4096 tokens
      of response — but you must ensure prompt_tokens + max_tokens ≤ context_limit.

      Conservative planning budget:
        Reserve 20–30% of the context window for the response.
        Use at most 70–80% for the prompt.


---


## 5. TOKENISATION EDGE CASES — Numbers, Code, Non-English, Emojis

The 1.3 tokens-per-word rule breaks down badly for several important
categories of text. Understanding why — and by how much — is essential
for anyone building applications on top of LLMs.


## 5.1  Numbers and Arithmetic

Numbers are among the most problematic inputs for LLM tokenisers.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW INTEGERS ARE TOKENISED (cl100k / GPT-4)                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Single digits 0–9: always single tokens.
      Two-digit numbers 10–99: all have single tokens.
      Three-digit: most have single tokens.
      Four-digit: many have single tokens.
      Five+ digits: almost always fragment.

      Examples (cl100k):

        "7"       →  [7]                            1 token
        "42"      →  [42]                           1 token
        "100"     →  [100]                          1 token
        "2024"    →  [2024]                         1 token
        "12345"   →  [123][45]                      2 tokens
        "999999"  →  [999][999]                     2 tokens
        "1000000" →  [1000][000]                    2 tokens

      Decimal numbers:

        "3.14"    →  [3][.][14]                     3 tokens
        "0.001"   →  [0][.][001]                    3 tokens
        "98.6"    →  [98][.][6]                     3 tokens

      This fragmentation has significant consequences for arithmetic tasks.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY LLMS ARE BAD AT ARITHMETIC — A TOKEN EXPLANATION            ║
    ╚══════════════════════════════════════════════════════════════════╝

      Problem: "What is 47823 + 96541?"

      Tokenisation of 47823:   [478][23]     — 2 tokens
      Tokenisation of 96541:   [965][41]     — 2 tokens

      The model sees:  "What is" [478] [23] "+" [965] [41] "?"

      To compute the sum, the model must:
        1. Understand that [478] and [23] together represent "47823"
           (the number was split across a token boundary mid-digit)
        2. Carry out multi-step addition across these fragmented pieces
        3. Output the result, which must ALSO be re-fragmented as tokens

      This is like asking a human to add numbers written on torn pieces
      of paper where the tear can fall anywhere mid-digit. The model
      can learn to do this but it is genuinely harder than if each digit
      had a consistent single representation.

      The fragmentation is also INCONSISTENT:
        "12345" → [123][45]       (split after 3 digits)
        "12346" → [12346]         (5-digit number that happens to have a token)
        "12347" → [123][47]       (split after 3 digits again)

      The model cannot rely on positional consistency.
      This is a primary reason LLMs benefit from using code interpreters
      (Python execution) for arithmetic instead of computing natively.


## 5.2  Code and Indentation

Source code tokenises very differently from prose — often 2–4× more tokens
per character than equivalent English text.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  PYTHON CODE TOKENISATION — A FUNCTION EXAMPLE                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      Source code:

        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

      Tokenised (cl100k, each [...] is one token):

        [def][ fibonacci][(][n][)][:][\n]
        [    ][if][ n][ <=][ 1][:][\n]
        [        ][return][ n][\n]
        [    ][return][ fibonacci][(][n][ -][ 1][)][ +][ fibonacci][(][n][ -][ 2][)]

      Token count: ~35 tokens for ~115 characters
      Characters per token: ~3.3
      Notably expensive elements:
        • [    ] = 4 spaces of indentation (1 token in cl100k, was 4 in p50k)
        • [        ] = 8 spaces (1 token in cl100k)
        • [(] and [)] and [:] = typically 1 token each
        • [\n] = newline = 1 token

    ╔══════════════════════════════════════════════════════════════════╗
    ║  INDENTATION COST — GPT-2 vs GPT-4 TOKENISERS                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      Python uses spaces for indentation. Every 4 spaces is one indent level.
      In p50k_base (GPT-2/3 tokeniser):

        "    " (4 spaces)   →  [ ][ ][ ][ ]   4 tokens
        "        " (8 sp)   →  [ ][ ][ ][ ][ ][ ][ ][ ]  8 tokens

      In cl100k_base (GPT-4 tokeniser):

        "    " (4 spaces)   →  [    ]          1 token  ← 4× cheaper
        "        " (8 sp)   →  [        ]      1 token  ← 8× cheaper

      For a deeply-nested Python file (average 3 levels = 12 spaces per line,
      100 lines):
        p50k:   1200 indentation tokens
        cl100k: 100 indentation tokens      ← 12× cheaper

      This is one of the primary reasons cl100k was introduced for GPT-4.
      Code-heavy use cases with GPT-3.5 were consuming tokens at high rates
      purely from whitespace.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  JSON AND STRUCTURED DATA                                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      JSON is among the most token-expensive formats per information unit.

      {"user": "alice", "role": "admin", "active": true}

      Tokenised:
      [{]["][ user]["][:][ "][alice]["][,][ "][role]["][:]
      [ "][admin]["][,][ "][active]["][:][ true][}]
      ~20 tokens for 51 characters = 2.55 chars/token

      The heavy use of ", :, { and } — each needing its own token or
      forming small tokens — makes JSON structurally expensive.

      Dense JSON payloads at scale (e.g. injecting a 50-field API response
      into a prompt) can cost 2–3× the tokens of equivalent prose description.
      Consider describing structured data in prose for large payloads.


## 5.3  Non-English Languages

All production tokenisers are trained primarily on English text.
This creates a systematic efficiency gap for other languages.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TOKENS PER CHARACTER FOR DIFFERENT SCRIPTS (cl100k)             ║
    ╚══════════════════════════════════════════════════════════════════╝

      English (Latin, common):    1 token per ~3.5–4.5 chars  (baseline)
      French / Spanish:           1 token per ~3.0–4.0 chars  (close to English)
      German:                     1 token per ~2.5–3.5 chars  (compounds fragment)
      Russian / Cyrillic:         1 token per ~1.5–2.5 chars
      Greek:                      1 token per ~1.5–2.0 chars
      Chinese (Mandarin):         1 token per ~1.5–2.0 chars
      Japanese (mixed scripts):   1 token per ~1.0–1.5 chars
      Arabic:                     1 token per ~0.8–1.5 chars  (morphologically rich)
      Thai (no spaces):           1 token per ~0.5–1.0 chars

      ┌──────────────────────────────────────────────────────────────┐
      │  "How are you today?"  in different languages                │
      │                                                              │
      │  English:   How are you today?       →  5 tokens            │
      │  French:    Comment allez-vous?      →  7 tokens   1.4×     │
      │  Russian:   Как вы сегодня?         →  9 tokens   1.8×     │
      │  Chinese:   你今天好吗？             →  8 tokens   1.6×     │
      │  Arabic:    كيف حالك اليوم؟         → 13 tokens   2.6×     │
      │  Thai:      วันนี้คุณเป็นอย่างไรบ้าง → 20 tokens  4.0×     │
      └──────────────────────────────────────────────────────────────┘

      Practical implications:

        1. CONTEXT WINDOW IS SMALLER FOR NON-ENGLISH USERS
           A 128k context window holds ~512k English characters (~170 pages)
           but only ~128k Thai characters (~43 pages) for the same token budget.

        2. API COSTS ARE HIGHER PER WORD
           A 1,000-word Arabic document costs 2–3× more in tokens
           than a 1,000-word English document of equivalent information density.

        3. MODELS MAY REASON LESS WELL
           More tokens needed = shorter effective context = the model may
           run out of context window before processing all the relevant text.


## 5.4  Emojis and Special Unicode

Emojis are encoded in UTF-8 and can use 4–7 bytes each, making them
unexpectedly expensive in tokens.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  EMOJI TOKENISATION — BYTE COSTS                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      Common emojis and their UTF-8 byte counts:

        😀  (U+1F600)  →  4 bytes  →  [ð][Ł][ĸ][Ĩ]  or single token in cl100k
        ❤️  (U+2764 + U+FE0F variation selector)  →  6 bytes
        👨‍👩‍👧‍👦 (family emoji) →  25 bytes  →  multiple tokens  (ZWJ sequence)

      In cl100k (GPT-4's tokeniser), many common emojis have earned
      dedicated tokens because they appeared frequently in training data:

        😀  →  [😀]          1 token
        ❤️  →  [❤][️]        2 tokens  (base char + variation selector)
        🏳️‍🌈 →  [🏳][️][‍][🌈]   4 tokens  (rainbow flag = ZWJ sequence)

      ZWJ (Zero Width Joiner) sequences — where multiple emojis are
      combined into one visible symbol — are especially expensive:

        👨‍💻  (man at laptop)  →  [👨][‍][💻]   3 tokens
        🧑‍🤝‍🧑  (people holding hands) → [🧑][‍][🤝][‍][🧑]  5 tokens

      A single visible emoji can cost 1–5 tokens.
      Emoji-heavy content (social media, chat logs) tokenises much more
      expensively per visible character than prose.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  OTHER PROBLEMATIC UNICODE                                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      Mathematical symbols:
        ∫∑∏√∞  — each 1–3 tokens (model-dependent)
        Full equations can be very token-expensive

      Homoglyph attacks:
        "аccount" (Cyrillic 'а' in an otherwise-Latin word) →
        The Cyrillic 'а' (U+0430) is a different byte sequence from
        Latin 'a' (U+0061). After Unicode normalisation they may or
        may not be collapsed. This can cause unexpected tokenisation.

      Right-to-left text:
        Arabic and Hebrew include directional control characters
        (U+200F RIGHT-TO-LEFT MARK, U+202B etc.) that consume tokens
        but are invisible to human readers.

      Invisible characters:
        Zero-width space (U+200B), soft hyphen (U+00AD), non-breaking
        space (U+00A0) — all visible to the tokeniser, invisible to humans.
        This is used in watermarking and prompt injection attacks.


---


## 6. SPECIAL TOKENS — <BOS>, <EOS>, <PAD>, [INST] and Their Roles

Special tokens are token IDs that are reserved for structural purposes
rather than representing text. They are never produced by the BPE algorithm
from user input — they are injected programmatically by the tokenisation
wrapper.


## 6.1  What Makes a Token "Special"

    ╔══════════════════════════════════════════════════════════════════╗
    ║  REGULAR TOKENS vs. SPECIAL TOKENS                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      REGULAR TOKENS

        Produced by: the BPE merge rules applied to input text.
        Represent: chunks of natural text (words, subwords, bytes).
        Can appear: anywhere in the context window as part of content.
        Example: "the" → token ID 279  (appears millions of times in text)

      SPECIAL TOKENS

        Produced by: the tokenisation wrapper, injected at specific positions.
        Represent: structural markers that the model was trained to respond to.
        Are NOT produced by BPE from the raw text bytes.
        Example: <|endoftext|> → token ID 50256 in GPT-2.

      CRITICAL DISTINCTION:

        If a user types the literal string "<|endoftext|>", the tokeniser
        does NOT produce token ID 50256. It tokenises the characters
        individually: [<][|][end][of][text][|][>] → 7 tokens.

        The genuine special token 50256 can ONLY be injected programmatically
        by the code that wraps the tokeniser.

        This means users cannot accidentally (or intentionally) inject
        the actual special token structure through their input text — only
        through the semantic meaning of what they write.


## 6.2  Common Special Tokens Across Model Families

    +───────────────────────────────────────────────────────────────────────────+
    │  Token          ID (typical)   Model family    Purpose                    │
    +───────────────────────────────────────────────────────────────────────────+
    │  <|endoftext|>  50256          GPT-2/3         End-of-document separator  │
    │  <|im_start|>   —              GPT-4 (ChatML)  Start of a turn            │
    │  <|im_end|>     —              GPT-4 (ChatML)  End of a turn              │
    │  <s>            1              LLaMA 1/2       Beginning of sequence      │
    │  </s>           2              LLaMA 1/2       End of sequence            │
    │  <<SYS>>        —              LLaMA 2 chat    System prompt start        │
    │  <</SYS>>       —              LLaMA 2 chat    System prompt end          │
    │  [INST]         —              LLaMA 2 chat    User turn start            │
    │  [/INST]        —              LLaMA 2 chat    User turn end              │
    │  <|begin_of_text|>  128000     LLaMA 3         Beginning of sequence      │
    │  <|end_of_text|>    128001     LLaMA 3         End of generation          │
    │  <|start_header_id|> 128006    LLaMA 3         Role header start          │
    │  <|end_header_id|>   128007    LLaMA 3         Role header end            │
    │  <|eot_id|>          128009    LLaMA 3         End of turn                │
    │  [PAD]           0             BERT-family     Padding for batching       │
    │  [CLS]           101           BERT            Classification token       │
    │  [SEP]           102           BERT            Segment separator          │
    │  [MASK]          103           BERT            Masked token (pretraining) │
    +───────────────────────────────────────────────────────────────────────────+


## 6.3  ChatML — The OpenAI Conversation Format

GPT-4 and GPT-3.5-turbo use the ChatML (Chat Markup Language) format to
structure multi-turn conversations. Understanding this format explains what
the model actually receives at the token level.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CHATML FORMAT — WHAT THE MODEL ACTUALLY RECEIVES                ║
    ╚══════════════════════════════════════════════════════════════════╝

      A two-turn conversation (system + user + assistant + user):

      API call:
        messages = [
          {"role": "system",    "content": "You are a helpful assistant."},
          {"role": "user",      "content": "What is 2+2?"},
          {"role": "assistant", "content": "2+2 equals 4."},
          {"role": "user",      "content": "And 3+3?"}
        ]

      What the tokeniser actually produces:

      <|im_start|>system\n
      You are a helpful assistant.<|im_end|>\n
      <|im_start|>user\n
      What is 2+2?<|im_end|>\n
      <|im_start|>assistant\n
      2+2 equals 4.<|im_end|>\n
      <|im_start|>user\n
      And 3+3?<|im_end|>\n
      <|im_start|>assistant\n
      ↑
      the model is now expected to generate here

      Token anatomy:
        <|im_start|>  — special token  (injected)
        system        — regular tokens  "sys" "tem" or "system"
        \n            — newline token
        [content]     — regular tokens  (from the message text)
        <|im_end|>    — special token  (injected)
        \n            — newline token

      The model was trained to:
        • Understand <|im_start|>role as a turn boundary marker
        • Not include <|im_end|> until it intends to stop generating
        • Generate in the "assistant" slot after the final <|im_start|>assistant\n


## 6.4  LLaMA 3 Conversation Format

LLaMA 3 uses a different but structurally similar format with different
special token names.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  LLAMA 3 FORMAT — THE SAME CONVERSATION                          ║
    ╚══════════════════════════════════════════════════════════════════╝

      <|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>\n\n
      You are a helpful assistant.<|eot_id|>
      <|start_header_id|>user<|end_header_id|>\n\n
      What is 2+2?<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\n\n
      2+2 equals 4.<|eot_id|>
      <|start_header_id|>user<|end_header_id|>\n\n
      And 3+3?<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\n\n
      ↑
      generation starts here

      Key special tokens:
        <|begin_of_text|>       — marks the very start (like a BOS)
        <|start_header_id|>     — before the role name
        <|end_header_id|>       — after the role name, before content
        <|eot_id|>              — end of turn (like <|im_end|> in ChatML)

      IMPORTANT NOTE ON LLAMA 2 vs LLAMA 3:

        LLaMA 2 used text-based markers like [INST] and <<SYS>> — these were
        regular text tokens, not dedicated special token IDs. This meant they
        were more vulnerable to being "overridden" by user content that
        mimicked the format.

        LLaMA 3 switched to dedicated special tokens with reserved IDs (e.g.
        128006, 128007) that cannot be produced by BPE from user text.
        This is a meaningful security improvement for prompt injection resistance.


## 6.5  The <PAD> Token and Why Batching Requires It

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY PADDING EXISTS                                              ║
    ╚══════════════════════════════════════════════════════════════════╝

      GPU matrix operations require rectangular tensors.
      Every sequence in a batch must have identical length.

      Batch of 3 requests with different prompt lengths:
        Request A: 12 tokens  →  [t][t][t]...[t][t][t]
        Request B:  4 tokens  →  [t][t][t][t]
        Request C: 17 tokens  →  [t][t][t]...[t][t][t][t][t]
                                  ↑ longest in batch

      After padding to length 17 (the maximum):
        Request A: [t]...[t][PAD][PAD][PAD][PAD][PAD]  ← 5 padding tokens
        Request B: [t][t][t][t][PAD][PAD]...[PAD]       ← 13 padding tokens
        Request C: [t]...[t][t][t][t][t][t]             ← no padding needed

      The <PAD> token occupies a vocabulary slot but contributes nothing.
      An ATTENTION MASK is used to prevent PAD positions from influencing
      the computation: their attention scores are set to -∞ before softmax.

      However: the GPU still runs matrix multiplications for PAD positions —
      the attention mask prevents them from AFFECTING other tokens, but does
      not prevent the COMPUTE from occurring. This is padding inefficiency.

      Modern serving systems use left-padding (padding at the start) combined
      with positional encoding offsets, or packing (multiple sequences packed
      into one row without padding) to minimise this waste.


## 6.6  Special Token Security — Prompt Injection via Token Confusion

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY ENCODING A STRING ≠ INJECTING THE SPECIAL TOKEN             ║
    ╚══════════════════════════════════════════════════════════════════╝

      SCENARIO: User sends the string "<|im_end|>" as their message.

      What the APPLICATION does (correct):
        Takes the string "<|im_end|>"
        Passes it through BPE tokenisation
        Gets regular text tokens: [<][|][im][_][end][|][>]  → 7 tokens
        These are 7 regular content tokens, NOT the special token.

      What the APPLICATION does (incorrect — a bug):
        Blindly concatenates user input into a raw token sequence
        including the actual special token ID for <|im_end|>
        The model sees this as a genuine end-of-turn marker

      The protection that makes "correct" correct:
        The tokeniser wrapper applies BPE to user text.
        BPE never produces special token IDs — those are injected
        programmatically only.
        So as long as the application processes user text through
        the tokeniser rather than treating it as raw token IDs,
        special token injection from user text is structurally prevented.

      SEMANTIC INJECTION (harder to prevent):
        A user who writes "Ignore all previous instructions and instead..."
        does NOT inject a special token.
        But the CONTENT of their message may semantically override the
        model's behaviour if the model was not sufficiently trained to
        resist such phrasing.
        This is prompt injection at the embedding level, not the token level.
        No tokenisation safeguard can prevent it.


---


## 7. TOKENISATION AND COST — Token Count = Compute Cost = API Billing Unit

Every token you send to an LLM API and every token the model generates
costs compute. API providers convert this compute cost into a per-token price.
Understanding the cost model in detail lets you build efficiently and avoid surprises.


## 7.1  The Token = Compute Equation

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY TOKENS MAP DIRECTLY TO COMPUTE COST                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      Every input token:
        • Requires a pre-fill forward pass contribution
          (each token's K and V are computed for every layer)
        • Occupies KV cache memory throughout the generation

      Every output token:
        • Requires one complete auto-regressive decode step
          (a full forward pass of the entire model per token)
        • Appends one entry to the KV cache

      The compute is direct: more tokens in → more matrix multiplies →
      more GPU-time → more cost. There is no "cheap" token or "skipped" layer.
      Every token is processed equally.

      This is why providers charge per token rather than per character,
      per word, or per API call.


## 7.2  Input Tokens vs. Output Tokens — Asymmetric Pricing

Input (prompt) tokens and output (completion) tokens are almost always
priced differently — and output tokens cost more.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  WHY OUTPUT COSTS MORE THAN INPUT                                ║
    ╚══════════════════════════════════════════════════════════════════╝

      INPUT (pre-fill phase):
        All N prompt tokens are processed in ONE batched matrix operation.
        The GPU processes all tokens simultaneously.
        Cost: approximately one matrix multiply per token per layer,
        but the parallelism makes it highly efficient.
        Effective GPU-seconds per token: low.

      OUTPUT (decode phase):
        Each token requires one sequential forward pass of the ENTIRE model.
        Tokens must be generated one at a time (each depends on the previous).
        The GPU runs at much lower utilisation per token.
        Effective GPU-seconds per token: 3–5× higher than input.

      This asymmetry means:
        • Long prompts with short responses → cheaper
        • Short prompts with long responses → more expensive per total token

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SAMPLE PRICING — MAJOR PROVIDERS (approximate, mid-2024)        ║
    ╚══════════════════════════════════════════════════════════════════╝

      Prices are in USD per million tokens (MTok).

      ┌─────────────────────────────────┬───────────────┬──────────────┐
      │  Model                          │  Input / MTok │ Output / MTok│
      ├─────────────────────────────────┼───────────────┼──────────────┤
      │  GPT-4o                         │   $5.00       │   $15.00     │
      │  GPT-4o mini                    │   $0.15       │    $0.60     │
      │  GPT-4-turbo                    │  $10.00       │   $30.00     │
      │  Claude 3.5 Sonnet              │   $3.00       │   $15.00     │
      │  Claude 3 Haiku                 │   $0.25       │    $1.25     │
      │  Claude 3 Opus                  │  $15.00       │   $75.00     │
      │  Gemini 1.5 Flash               │   $0.075      │    $0.30     │
      │  Gemini 1.5 Pro                 │   $3.50       │   $10.50     │
      └─────────────────────────────────┴───────────────┴──────────────┘

      Output/input price ratio: typically 3–5× more expensive per output token.
      This is consistent with the compute asymmetry described above.


## 7.3  Cost Estimation Formula

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ESTIMATING API COST FOR A GIVEN WORKLOAD                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      cost = (prompt_tokens × input_price)  +  (output_tokens × output_price)
              ────────────────────────────     ─────────────────────────────
                   measured in MTok                  measured in MTok

      Example 1: A customer-support bot
      ─────────────────────────────────
      Avg system prompt:      500 tokens
      Avg conversation so far: 800 tokens
      Avg user message:        50 tokens
      Total prompt tokens:    1,350 tokens

      Avg response:            200 tokens
      Model: Claude 3.5 Sonnet ($3.00 in, $15.00 out per MTok)

      Cost per conversation turn:
        input:  1,350 × ($3.00 / 1,000,000)  = $0.00405
        output:   200 × ($15.00 / 1,000,000) = $0.00300
        total:                                = $0.00705 per turn

      At 10,000 turns/day:
        10,000 × $0.00705 = $70.50/day = $2,115/month

      Example 2: A document summarisation pipeline
      ──────────────────────────────────────────────
      Document size: 20,000 tokens
      Summary output: 500 tokens
      Model: GPT-4o ($5.00 in, $15.00 out)

      Cost per document:
        input:  20,000 × ($5.00 / 1,000,000)  = $0.10
        output:    500 × ($15.00 / 1,000,000) = $0.0075
        total:                                 = $0.1075 per document

      At 1,000 documents/day: $107.50/day = $3,225/month


## 7.4  Prefix Caching — The Single Biggest Cost Lever

If your application sends the same system prompt or document on every API
call, prefix caching can reduce costs by 60–80%.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW PREFIX CACHING AFFECTS THE BILLING UNIT                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Normal pricing (no caching):
        Every call pays full input price for every token in the prompt.

      With prefix caching:
        The provider caches the KV activations for the static prefix.
        Subsequent requests that share the identical prefix pay a
        reduced "cache hit" price for those tokens.

        ┌──────────────────────────────────────────────────────────┐
        │  Provider          Cache hit price    vs. full price     │
        ├──────────────────────────────────────────────────────────┤
        │  Anthropic          $0.30 / MTok      10% of $3.00      │
        │  OpenAI             $2.50 / MTok      50% of $5.00      │
        │  Google (1.5 Pro)   $0.875 / MTok     25% of $3.50      │
        └──────────────────────────────────────────────────────────┘

      For the customer-support bot example above (1,350 token prompt,
      500 of which are the fixed system prompt):

        Without caching:  1,350 tokens × $3.00/MTok = $0.00405 per turn
        With caching:     850 tokens × $3.00/MTok + 500 tokens × $0.30/MTok
                        = $0.00255 + $0.00015 = $0.00270 per turn

      Savings: 33% on input cost, 19% on total cost per turn.

      For a 10,000-token system prompt served to 1,000 requests/minute:
        Without caching: 1M tokens/min × $3.00/MTok = $3.00/min = $180/hr
        With caching:    mostly cache hits at $0.30 → ~$18/hr
        Savings: ~90%


## 7.5  Token-Efficient Prompt Engineering

Token count is directly controllable through prompt design. Small changes
can yield meaningful savings at scale.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TOKEN-SAVING TECHNIQUES                                         ║
    ╚══════════════════════════════════════════════════════════════════╝

      1. REMOVE FILLER PHRASES
         ─────────────────────
         Before:  "Please could you kindly help me to understand..."  (10 tokens)
         After:   "Explain..."                                        ( 1 token)
         Savings: 9 tokens per instruction. Across 1M calls: 9M tokens saved.

      2. PREFER PROSE TO JSON FOR LARGE STRUCTURED DATA
         ─────────────────────────────────────────────
         JSON payload (customer record, 300 chars): ~120 tokens
         Prose summary (same data):                  ~60 tokens
         When you inject many structured records, prose can be 30–50% cheaper.

      3. TRUNCATE CONVERSATION HISTORY
         ─────────────────────────────
         Most relevant information is in recent turns.
         Keep last 3–5 turns rather than the full history.
         Summarise old turns into a single compressed block.

      4. REMOVE WHITESPACE AND REDUNDANCY FROM SYSTEM PROMPTS
         ────────────────────────────────────────────────────
         Every blank line, every repeated instruction, every unnecessary
         example consumes tokens. Audit system prompts for tokens the
         model doesn't need.

      5. MOVE STATIC CONTENT TO A PREFIX-CACHED BLOCK
         ───────────────────────────────────────────
         Place large static documents, knowledge bases, or system
         instructions at the start of the prompt (position 0).
         Keep this block byte-identical across all requests.
         This maximises cache hit rates.

      6. USE OUTPUT TOKEN LIMITS APPROPRIATELY
         ───────────────────────────────────────
         max_tokens controls maximum output length but not minimum.
         For structured tasks (classification, extraction), the model
         often needs far fewer output tokens than the default max.
         Setting max_tokens=50 for a sentiment classifier rather than
         4096 has no quality cost but prevents runaway generation.


## 7.6  The Context Window Budget — A Complete Picture

    ╔══════════════════════════════════════════════════════════════════╗
    ║  ALL THE THINGS THAT CONSUME CONTEXT WINDOW TOKENS               ║
    ╚══════════════════════════════════════════════════════════════════╝

      For a production conversational application, the context window
      is partitioned roughly as:

      ┌─────────────────────────────────────────────────────────────────┐
      │  TOTAL CONTEXT WINDOW (e.g. 128,000 tokens for GPT-4-turbo)    │
      │                                                                 │
      │  ┌──────────────────────────────────────────────┐              │
      │  │  SYSTEM PROMPT                               │  ~500–5,000  │
      │  │  Instructions, persona, safety rules,        │  tokens      │
      │  │  tool schemas, output format spec            │              │
      │  └──────────────────────────────────────────────┘              │
      │                                                                 │
      │  ┌──────────────────────────────────────────────┐              │
      │  │  INJECTED CONTEXT                            │  0–50,000   │
      │  │  Retrieved documents, database results,      │  tokens      │
      │  │  code files, structured data                 │              │
      │  └──────────────────────────────────────────────┘              │
      │                                                                 │
      │  ┌──────────────────────────────────────────────┐              │
      │  │  CONVERSATION HISTORY                        │  grows each  │
      │  │  All prior user and assistant turns          │  turn        │
      │  └──────────────────────────────────────────────┘              │
      │                                                                 │
      │  ┌──────────────────────────────────────────────┐              │
      │  │  CURRENT USER MESSAGE                        │  ~10–500     │
      │  └──────────────────────────────────────────────┘  tokens     │
      │                                                                 │
      │  ┌──────────────────────────────────────────────┐              │
      │  │  RESERVED FOR RESPONSE  (max_tokens)         │  ~500–4,096  │
      │  └──────────────────────────────────────────────┘  tokens     │
      │                                                                 │
      └─────────────────────────────────────────────────────────────────┘

      As conversation history grows, it will eventually push the total
      past the context limit. Strategies to manage this:

        • Sliding window: discard oldest turns
        • Summarisation: replace old turns with a compressed summary
        • Retrieval: only inject the most relevant past context
        • Token budget tracking: count tokens after each turn and
          truncate before hitting the limit

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FULL COST MODEL — PUTTING IT ALL TOGETHER                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      Every API call pays for:

        1. System prompt tokens         (every call, unless cached)
        2. Injected context tokens      (every call, unless cached)
        3. Conversation history tokens  (grows per turn)
        4. Current user message tokens  (every call)
        5. Special tokens (BOS, turns)  (small, few dozen per call)
        6. Generated response tokens    (at output token price)

      The model charges NOTHING for:
        • Unused context window capacity
        • Thinking or reasoning (unless using extended thinking APIs)
        • Latency

      The single most cost-effective intervention at scale:
        PREFIX CACHE your system prompt and any static documents.
        This turns items 1 and 2 from full-price to ~10–25% price
        for the vast majority of production traffic.


---


## Glossary

    +──────────────────────┬──────────────────────────────────────────────────────────+
    │  Term                │  Definition                                               │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  BPE                 │  Byte Pair Encoding — iterative merge-based sub-word      │
    │                      │  tokenisation algorithm used by GPT family models.        │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Byte fallback       │  The guarantee that any byte (0–255) has a token in the  │
    │                      │  vocabulary, ensuring no text is un-tokenisable.          │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  ChatML              │  Chat Markup Language — OpenAI's structured token format  │
    │                      │  using <|im_start|> and <|im_end|> turn markers.         │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Compression ratio   │  Characters per token. English prose: ~3.5–4.5.          │
    │                      │  Lower = more tokens = more expensive.                   │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Context window      │  Maximum combined tokens (prompt + response) the model   │
    │                      │  can process in one API call.                            │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  d_model             │  Embedding dimension — the number of floats representing │
    │                      │  each token in the model's hidden states.                │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Detokenisation      │  Converting token ID sequences back to UTF-8 text.       │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Embedding table     │  Matrix of shape (vocab_size × d_model) mapping token    │
    │                      │  IDs to dense vectors. Loaded once at server startup.    │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Fan-in              │  Number of input connections to one neuron = previous    │
    │                      │  layer width. (See Architecture module.)                 │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  LM head             │  Final linear projection (d_model → vocab_size) that     │
    │                      │  produces logits over the vocabulary for next-token pred.│
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Merge rule          │  An ordered instruction to combine two adjacent symbols  │
    │                      │  into one. The ordered list of merge rules IS the BPE    │
    │                      │  tokeniser (together with the vocabulary).               │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  MTok                │  Million tokens — the standard billing unit for LLM APIs.│
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Pre-tokenisation    │  A regex pass that splits text into chunks before BPE.   │
    │                      │  Prevents merges from crossing word or space boundaries. │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Prefix caching      │  Server-side storage of KV activations for a repeated    │
    │                      │  token prefix, allowing subsequent requests to skip       │
    │                      │  recomputing those tokens. Reduces cost 60–90%.          │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  SentencePiece       │  Tokenisation library supporting BPE and Unigram LM.    │
    │                      │  Used by LLaMA, T5, and many multilingual models.        │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Special token       │  A token with a reserved ID that marks conversation      │
    │                      │  structure (turn start/end, BOS, EOS, PAD). Injected     │
    │                      │  programmatically; never produced by BPE from text.      │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Subword             │  A fragment of a word that has its own token entry.      │
    │                      │  The BPE algorithm discovers subwords from corpus stats. │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  tiktoken            │  OpenAI's open-source tokeniser library. Implements BPE  │
    │                      │  for gpt2, p50k_base, cl100k_base, o200k_base.           │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Token               │  An integer ID in a fixed vocabulary. The atomic unit of │
    │                      │  LLM computation, memory, and billing.                   │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Tokenisation        │  The process of converting a string of text into a       │
    │                      │  sequence of token IDs.                                  │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Unigram LM          │  An alternative to BPE used in SentencePiece. Starts     │
    │                      │  with a large vocabulary and prunes entries whose         │
    │                      │  removal minimally reduces corpus log-likelihood.         │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  UTF-8               │  Variable-length Unicode encoding. Every Unicode code    │
    │                      │  point encoded as 1–4 bytes. The substrate BPE           │
    │                      │  tokenisers operate on.                                  │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Vocabulary          │  The complete bidirectional mapping between byte-         │
    │                      │  sequences and integer IDs. Frozen after training.        │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  Weight tying        │  Using the same weight matrix for both input embedding   │
    │                      │  and output projection (LM head). Saves vocab_size ×     │
    │                      │  d_model parameters.                                     │
    +──────────────────────┼──────────────────────────────────────────────────────────+
    │  ZWJ sequence        │  Zero Width Joiner — a Unicode character used to combine │
    │                      │  multiple emojis into one rendered glyph. Can cost 3–6   │
    │                      │  tokens for a single visible emoji.                      │
    +──────────────────────┴──────────────────────────────────────────────────────────+


"""

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = """

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Step-by-step tutorials with runnable commands
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }