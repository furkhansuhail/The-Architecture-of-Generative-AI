"""
Dataset Preparation and Preprocessing
=======================================

The quality of the training data is arguably the single most important
factor in LLM quality — more so than architecture choices or optimiser
tuning. "Garbage in, garbage out" is especially true at scale: a model
trained on 15 trillion tokens of carefully filtered text vastly outperforms
one trained on 15 trillion tokens of unfiltered web crawl. This module
covers the full pipeline from raw web data to clean, tokenised, binary
shards ready for training.

"""

import base64
import os
import textwrap
import re


TOPIC_NAME   = "Dataset Preparation and Preprocessing"
DISPLAY_NAME = "13 · Dataset Preparation"
ICON         = "🗂️"
SUBTITLE     = "Cleaning, Deduplication, and Filtering"


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

### The Data Pipeline Overview

Before a single training step runs, the raw data goes through a multi-stage
pipeline that can itself take weeks on a large cluster:

    Stage 1 — Collection:       web crawl → raw HTML
    Stage 2 — Extraction:       HTML → plain text (boilerplate removal)
    Stage 3 — Language ID:      filter to target language(s)
    Stage 4 — Deduplication:    remove near-duplicate documents
    Stage 5 — Quality filter:   remove low-quality text
    Stage 6 — Content filter:   remove harmful / toxic content
    Stage 7 — Tokenisation:     text → token ID sequences
    Stage 8 — Packing:          sequences → fixed-length chunks
    Stage 9 — Sharding:         split into binary files for parallel loading

This pipeline is not a one-time process. Each new training run typically
reruns stages 3–9 on an updated corpus, because the mixing ratios and
quality thresholds are revised based on ablation experiments.


### Stage 1–2: Web Crawl and Text Extraction

The most common starting point is **Common Crawl** — a non-profit web crawl
that archives ~3 billion web pages per monthly snapshot, totalling ~250 TB
of compressed WARC files.

**Text extraction** removes HTML tags, JavaScript, CSS, navigation menus,
headers, footers, and ads to recover the main article text. Tools:
    •   **Trafilatura**: ML-based main content extractor
    •   **Resiliparse**: fast C++ extractor used in many LLM datasets
    •   **BeautifulSoup**: flexible but slower Python parser

A typical extraction yields ~10–15% of the raw HTML size as usable text.
The ratio reflects how much of the modern web is boilerplate.

**C4 (Cleaned Common Crawl):** The dataset used to train T5. Applied
basic filtering to Common Crawl (remove pages with < 5 sentences, filter
bad words, require ending punctuation on lines). Resulted in 745 GB of text.

**FineWeb (HuggingFace, 2024):** State-of-the-art Common Crawl dataset.
15T tokens, heavy deduplication, quality filtering via classifier trained
on Wikipedia + books vs web spam. Used to train many modern open models.


### Stage 3: Language Identification

Language classifiers identify and filter to target languages.
Common tools:
    •   **fastText LID (lid.176.bin):** Facebook's 176-language classifier;
        processes millions of documents per second
    •   **CLD3 (Google Compact Language Detector):** browser-grade LID

For multilingual training, a per-language sampling ratio is applied.
A typical English-dominant LLM might use:
    •   ~70% English
    •   ~10% code
    •   ~5–10% other high-resource languages (German, French, Chinese, etc.)
    •   ~5% low-resource languages (to improve multilingual capability)


### Stage 4: Deduplication — The Most Important Step

Deduplication has the largest impact on data quality of any single step.
Training on many copies of the same document causes the model to memorise
that document verbatim (a privacy and security concern) and degrades
generalisation.

**Why it matters:** Studies show that duplicated data in the pre-training
corpus causes:
    1.  Verbatim memorisation of repeated text
    2.  Overrepresentation of certain writing styles/topics
    3.  Reduced sample diversity → worse generalisation
    4.  Potential privacy violations (PII in memorised text)

**Three levels of deduplication:**

    Level           Granularity         Algorithm
    ─────────────────────────────────────────────────────────────
    Exact URL       Whole document      Set membership (hash)
    Exact content   Whole document      MD5 / SHA hash of text
    Near-duplicate  Sub-document        MinHash + LSH
    ─────────────────────────────────────────────────────────────

**MinHash LSH (Locality-Sensitive Hashing):**
The standard algorithm for near-duplicate detection at scale.

    1.  Shingling: split each document into overlapping n-grams
        (typically character 5-grams or word 3-grams)
    2.  MinHash: compute k hash functions on the shingles, take the minimum
        of each → a k-dimensional signature vector
    3.  LSH banding: divide the signature into b bands of r rows each;
        documents that share any identical band are candidate duplicates
    4.  Candidate pairs: compute exact Jaccard similarity for candidates
        (or just mark them as duplicates directly)

**Jaccard similarity:**

    J(A, B) = |A ∩ B| / |A ∪ B|

where A and B are the shingle sets of two documents. A Jaccard ≥ 0.8 is
typically treated as a near-duplicate.

The MinHash estimate has a nice property:
    P(min(h(A)) == min(h(B))) = J(A, B)

So the fraction of matching MinHash values approximates Jaccard similarity
without ever computing the full shingle sets explicitly.


    **Diagram 1 — MinHash LSH Pipeline:**

    MINHASH LSH DEDUPLICATION PIPELINE
    ════════════════════════════════════════════════════════════════

    Document A: "The cat sat on the mat"
    Document B: "The cat sat on the mat and then left"
    Document C: "A completely different document about science"

    Step 1 — Character 5-grams of A:
        {"The c", "he ca", "e cat", " cat ", "cat s", ...}
        |shingles(A)| = 17

    Step 2 — MinHash signatures (k=4 hash functions):
        A: [h1_min, h2_min, h3_min, h4_min] = [7, 12, 3, 19]
        B: [7, 12, 3, 22]    ← similar to A
        C: [41, 87, 65, 31]  ← very different

    Step 3 — Jaccard estimate:
        A vs B: 3/4 = 0.75   (3 of 4 hashes match → likely duplicate!)
        A vs C: 0/4 = 0.00   (no matches → not a duplicate)

    Step 4 — Banding (b=2 bands, r=2 rows each):
        Band 0 of A: (7, 12)   Band 1 of A: (3, 19)
        Band 0 of B: (7, 12) ← MATCH in band 0 → candidate pair!
        Band 0 of C: (41,87)  ← no match

    Final: A and B are candidate duplicates → keep only one.


### Stage 5: Quality Filtering

Quality filtering removes text that is technically not a duplicate but is
still low-quality: spam, machine-generated text, navigation-only pages,
adult content, and gibberish.

**Rule-based heuristics (fast, cheap):**
    •   Fraction of lines ending with punctuation (< 0.6 → likely list spam)
    •   Mean word length (< 3 or > 10 → likely SEO spam or encoded data)
    •   Stop word coverage (< 0.3 → likely gibberish or code)
    •   Symbol-to-word ratio (> 0.1 → likely spam)
    •   Unique word ratio (< 0.2 → likely repetitive text)
    •   Bullet point ratio (> 0.9 → likely navigation/menu boilerplate)
    •   Average sentence length (< 10 words → fragmented text)
    •   Perplexity under a reference LM (high PPL → incoherent text)

**C4 heuristics (simple but effective):**
    •   Remove lines containing "javascript" (boilerplate)
    •   Remove lines with lorem ipsum
    •   Remove documents with < 3 sentences
    •   Remove if the fraction of words in a known word list < 0.8

**Classifier-based quality filtering (more powerful):**
    Train a small classifier (e.g., fastText or small BERT) to distinguish:
        High-quality: Wikipedia, books, curated web text
        Low-quality: spam, machine-generated text, social media noise

    This classifier can then score every document in the web crawl.
    Retain the top 20–30% by score. Used by:
        •   LLaMA-3 (classifier trained on Wikipedia + educational text)
        •   FineWeb (HuggingFace quality classifier)
        •   Dolma (Allen AI quality filters)


    **Diagram 2 — Quality Filter Decision Tree:**

    DOCUMENT QUALITY FILTER PIPELINE
    ════════════════════════════════════════════════════════════════

    Raw document
         │
         ▼
    [Length filter]  ── < 200 words? ──────────────────→ DISCARD
         │
         ▼
    [Language ID]    ── confidence < 0.8? ──────────────→ DISCARD
         │
         ▼
    [URL block list] ── known spam domain? ──────────────→ DISCARD
         │
         ▼
    [Heuristic rules]── symbol ratio > 0.1? ─────────────→ DISCARD
                     ── line count < 5? ──────────────────→ DISCARD
                     ── repeated n-gram ratio > 0.3? ──────→ DISCARD
         │
         ▼
    [Quality classifier]── score < threshold? ──────────→ DISCARD
         │
         ▼
    [PII filter]     ── contains SSN / CC / email? ──────→ REDACT or DISCARD
         │
         ▼
    KEEP ✓

    Typical retention rate: 15–30% of raw web crawl after all filters.


### Stage 6: Content and Safety Filtering

Removing harmful, toxic, and illegal content:
    •   Known toxic content: filter against blocklists (URLs, phrases)
    •   CSAM detection: perceptual hash matching against known illegal images
    •   Hate speech: classifier-based detection (Perspective API, custom)
    •   PII removal: regex-based detection of emails, phone numbers, SSNs

**The tension:** aggressive content filtering also removes legitimate
educational content (e.g., medical information, news articles about violence).
Most production pipelines apply probabilistic filtering: reduce the sampling
probability of documents with certain signals, rather than binary remove/keep.


### Stage 7–8: Tokenisation and Packing

After cleaning, documents are tokenised and packed into fixed-length
sequences for training (covered in detail in Module 14).

Key tokenisation decisions:
    •   Add a BOS (beginning-of-sequence) token at the start of each document
    •   Add an EOS (end-of-sequence) token at the end
    •   Use document boundaries to prevent cross-document attention
      (critical for packing — see Module 14)

Binary format for storage:
    •   Token IDs are stored as uint16 or uint32 arrays
    •   LLaMA-3 vocab = 128k → requires uint32 (> 65535)
    •   A 15T token dataset at uint16: 15T × 2 bytes = 30 TB
    •   Stored as numpy memmap files or HuggingFace Arrow format


### Composition of Major LLM Training Datasets

    Dataset         Size       Sources                    Used by
    ──────────────────────────────────────────────────────────────────────
    C4              745 GB     Common Crawl (filtered)    T5, mT5
    The Pile        825 GB     22 curated sources         GPT-NeoX, etc.
    RedPajama-v1    1.2 TB     Mix of 7 sources           LLaMA replicate
    Dolma           3 TB       Web, books, code, science  OLMo
    FineWeb         15 TB      Common Crawl (quality)     Many 2024 models
    LLaMA-3 data    ~15 TB*    Proprietary mix            LLaMA-3
    ──────────────────────────────────────────────────────────────────────
    * estimated from token counts

**Common source categories and their typical proportions:**
    •   Web text (Common Crawl / FineWeb):   ~60–75%
    •   Code (GitHub, Stack Overflow):        ~5–15%
    •   Books (BookCorpus, Project Gutenberg): ~5–10%
    •   Wikipedia / encyclopedias:             ~2–5%
    •   Scientific papers (ArXiv, PubMed):    ~1–3%
    •   Curated high-quality text:            ~3–10%


### Data Freshness and Cutoff

The training data has a **knowledge cutoff** — the date beyond which the
model has no information. Managing the cutoff requires:

    1.  Clearly labelling the cutoff in model documentation
    2.  Ensuring the crawl date range is documented for each data source
    3.  Potentially including more recent data to extend the cutoff
       (common crawl is updated monthly)
    4.  Being aware that web pages are often crawled months after writing
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
Deduplication Method Comparison

| Method            | Granularity   | Time complexity  | Memory   | False positive rate | Used by               |
|-------------------|---------------|------------------|----------|---------------------|-----------------------|
| Exact URL hash    | Document      | O(N)             | O(N)     | 0%                  | All pipelines         |
| Exact MD5 hash    | Document      | O(N)             | O(N)     | ~0%                 | All pipelines         |
| MinHash LSH       | Sub-document  | O(N × k)         | O(N × b) | ~1–5%               | LLaMA, C4, FineWeb    |
| SimHash           | Document      | O(N)             | O(N)     | ~5%                 | Google, some LLMs     |
| Suffix Array      | Exact n-gram  | O(N log N)       | O(N)     | 0%                  | Deduplicating Pile    |
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "MinHash LSH Deduplication — From Scratch": {
        "description": "A complete MinHash LSH near-duplicate detector implemented from scratch — shingling, signature computation, banding, and deduplication of a document corpus.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
MINHASH LSH NEAR-DUPLICATE DETECTION — FROM SCRATCH
================================================================================

Implements the full MinHash + LSH pipeline:
    1. Shingling (character n-grams)
    2. MinHash signature computation (k random hash functions)
    3. LSH banding (b bands of r rows)
    4. Candidate pair detection
    5. Deduplication (keep one from each duplicate cluster)

No external libraries. Pure Python for educational clarity.
================================================================================
"""

import hashlib
import random
import math
from collections import defaultdict


# ── Shingling ─────────────────────────────────────────────────────────────────

def char_shingles(text: str, k: int = 5) -> set[str]:
    """Extract all character k-grams from a text."""
    text = text.lower().strip()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def word_shingles(text: str, k: int = 3) -> set[str]:
    """Extract all word k-grams (k consecutive words) from a text."""
    words = text.lower().split()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


# ── MinHash ───────────────────────────────────────────────────────────────────

def make_hash_functions(n_hashes: int, seed: int = 42) -> list:
    """
    Create n_hashes independent hash functions of the form:
        h(x) = (a * x + b) mod p   where p is a large prime
    Returns list of (a, b) pairs.
    """
    rng   = random.Random(seed)
    prime = (1 << 31) - 1   # Mersenne prime 2^31 - 1
    return [(rng.randint(1, prime - 1), rng.randint(0, prime - 1))
            for _ in range(n_hashes)]


def minhash_signature(shingles: set[str], hash_fns: list) -> list[int]:
    """
    Compute the MinHash signature for a set of shingles.

    For each hash function h, the signature entry is:
        min_{s in shingles} h(hash(s))

    This gives a k-dimensional signature that approximates set similarity.
    """
    prime = (1 << 31) - 1
    sig   = [prime] * len(hash_fns)   # initialise to max

    for shingle in shingles:
        # Convert shingle to an integer via MD5
        h_int = int(hashlib.md5(shingle.encode()).hexdigest(), 16) % prime

        for i, (a, b) in enumerate(hash_fns):
            val = (a * h_int + b) % prime
            if val < sig[i]:
                sig[i] = val

    return sig


def jaccard_from_signatures(sig_a: list, sig_b: list) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ── LSH Banding ───────────────────────────────────────────────────────────────

class LSHIndex:
    """
    Locality-Sensitive Hashing index for approximate nearest-neighbour search.

    Parameters:
        n_hashes:   number of MinHash functions (= b × r)
        n_bands:    number of bands (b)
        n_rows:     rows per band (r)

    Two documents are candidate duplicates if they share at least one
    identical band. The probability of being candidates:
        P(candidate | Jaccard = J) ≈ 1 - (1 - J^r)^b
    """

    def __init__(self, n_hashes: int = 128, n_bands: int = 32):
        assert n_hashes % n_bands == 0
        self.n_hashes  = n_hashes
        self.n_bands   = n_bands
        self.n_rows    = n_hashes // n_bands
        self.hash_fns  = make_hash_functions(n_hashes)
        self.buckets: dict[tuple, list[int]] = defaultdict(list)

    def add(self, doc_id: int, signature: list[int]):
        """Add a document's signature to the LSH index."""
        for band_idx in range(self.n_bands):
            start = band_idx * self.n_rows
            end   = start + self.n_rows
            band  = tuple(signature[start:end])
            # Use (band_idx, band) as bucket key to avoid cross-band collisions
            self.buckets[(band_idx,) + band].append(doc_id)

    def candidate_pairs(self) -> set[frozenset]:
        """Return all pairs of documents that share at least one bucket."""
        pairs = set()
        for bucket in self.buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        pairs.add(frozenset({bucket[i], bucket[j]}))
        return pairs

    def threshold(self) -> float:
        """Approximate Jaccard threshold at which P(candidate) ≈ 0.5."""
        return (1 / self.n_bands) ** (1 / self.n_rows)


# ── Full deduplication pipeline ───────────────────────────────────────────────

def deduplicate(documents: list[str], jaccard_threshold: float = 0.7,
                n_hashes: int = 128, n_bands: int = 32,
                shingle_size: int = 5, verbose: bool = True) -> list[int]:
    """
    Deduplicate a list of documents using MinHash + LSH.

    Returns a list of document indices to KEEP (one per duplicate cluster).
    """
    n_rows = n_hashes // n_bands

    if verbose:
        print(f"  MinHash LSH config: n_hashes={n_hashes}, bands={n_bands}, "
              f"rows={n_rows}")
        threshold = (1 / n_bands) ** (1 / n_rows)
        print(f"  LSH threshold (P=0.5): J ≈ {threshold:.2f}")
        print(f"  Target threshold:       J ≥ {jaccard_threshold:.2f}")
        print()

    # 1. Compute signatures
    hash_fns   = make_hash_functions(n_hashes)
    signatures = []
    for doc in documents:
        shingles = char_shingles(doc, shingle_size)
        sig      = minhash_signature(shingles, hash_fns)
        signatures.append(sig)

    # 2. Build LSH index
    index = LSHIndex(n_hashes, n_bands)
    for i, sig in enumerate(signatures):
        index.add(i, sig)

    # 3. Find candidate pairs
    candidates = index.candidate_pairs()
    if verbose:
        print(f"  Candidate pairs found: {len(candidates)}")

    # 4. Verify candidates with exact Jaccard (or use signature as proxy)
    duplicate_groups: dict[int, int] = {}   # doc_id → cluster representative
    for pair in candidates:
        a, b = tuple(pair)
        j_est = jaccard_from_signatures(signatures[a], signatures[b])
        if j_est >= jaccard_threshold:
            # Merge b into a's cluster
            rep_a = duplicate_groups.get(a, a)
            rep_b = duplicate_groups.get(b, b)
            if rep_a != rep_b:
                duplicate_groups[max(rep_a, rep_b)] = min(rep_a, rep_b)

    # 5. Determine which docs to keep
    to_remove = set(duplicate_groups.keys())
    to_keep   = [i for i in range(len(documents)) if i not in to_remove]

    if verbose:
        print(f"  Documents before dedup: {len(documents)}")
        print(f"  Documents after  dedup: {len(to_keep)}")
        print(f"  Removed:                {len(to_remove)} "
              f"({len(to_remove)/len(documents)*100:.1f}%)")

    return to_keep


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    corpus = [
        # Near-duplicates (same article, minor edits)
        "The transformer architecture was introduced in 2017 by Vaswani et al. "
        "in the paper Attention Is All You Need. It replaced recurrent networks "
        "with self-attention mechanisms, enabling massive parallelism.",

        "The transformer architecture was introduced in 2017 by Vaswani et al. "
        "in the landmark paper Attention Is All You Need. It replaced recurrent "
        "neural networks with self-attention mechanisms.",

        "The transformer model was first proposed in 2017. The paper, titled "
        "Attention Is All You Need, showed that self-attention could replace "
        "recurrence entirely in sequence modelling tasks.",

        # Completely different documents
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn from data without being explicitly programmed.",

        "Python is a high-level programming language known for its simplicity "
        "and readability. It is widely used in data science and web development.",

        "The history of photography begins in the early 19th century with the "
        "invention of the daguerreotype by Louis Daguerre in 1839.",

        # Exact duplicate
        "The transformer architecture was introduced in 2017 by Vaswani et al. "
        "in the paper Attention Is All You Need. It replaced recurrent networks "
        "with self-attention mechanisms, enabling massive parallelism.",
    ]

    print("=" * 65)
    print("  MINHASH LSH DEDUPLICATION DEMO")
    print("=" * 65)
    print()
    print("  Corpus documents:")
    for i, doc in enumerate(corpus):
        print(f"  [{i}] {doc[:70]}...")
    print()

    keep_indices = deduplicate(
        corpus,
        jaccard_threshold=0.65,
        n_hashes=64,
        n_bands=16,
        verbose=True,
    )

    print()
    print("  Kept documents:")
    for i in keep_indices:
        print(f"  [{i}] {corpus[i][:70]}...")

    # Show pairwise similarities
    print()
    print("=" * 65)
    print("  PAIRWISE JACCARD ESTIMATES")
    print("=" * 65)
    hash_fns   = make_hash_functions(128)
    sigs       = [minhash_signature(char_shingles(d), hash_fns) for d in corpus]
    print(f"  {'Pair':>8}  {'Est. Jaccard':>14}  {'Verdict':>12}")
    print(f"  {'':─>8}  {'':─>14}  {'':─>12}")
    for i in range(len(corpus)):
        for j in range(i+1, len(corpus)):
            j_est = jaccard_from_signatures(sigs[i], sigs[j])
            verdict = "DUPLICATE" if j_est >= 0.65 else "unique"
            if j_est >= 0.3:
                print(f"  [{i}] vs [{j}]  {j_est:>14.3f}  {verdict:>12}")
''',
    },

    "Quality Filtering Pipeline": {
        "description": "A multi-stage document quality filter implementing the C4/FineWeb heuristics — rule-based scoring, length filters, perplexity estimation, and PII detection.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DOCUMENT QUALITY FILTERING PIPELINE
================================================================================

Implements a production-style quality filtering pipeline with:
    1. Length filters (too short, too long)
    2. Language/script heuristics
    3. Text quality heuristics (C4-style and FineWeb-style)
    4. Repetition detection
    5. PII detection (email, phone, SSN patterns)
    6. Quality scoring (aggregate score)

Each filter is explainable — returns a reason for rejection.
================================================================================
"""

import re
import math
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FilterResult:
    kept:      bool
    score:     float          # 0.0 = garbage, 1.0 = high quality
    reasons:   list[str]      # reasons for rejection
    stats:     dict           # per-metric stats for debugging


# ── Individual filters ────────────────────────────────────────────────────────

def filter_length(text: str,
                  min_words: int = 50,
                  max_words: int = 100_000) -> Optional[str]:
    """Return rejection reason or None if passes."""
    words = text.split()
    if len(words) < min_words:
        return f"too short ({len(words)} words < {min_words})"
    if len(words) > max_words:
        return f"too long ({len(words)} words > {max_words})"
    return None


def filter_line_quality(text: str,
                        min_punct_frac: float = 0.2,
                        max_bullet_frac: float = 0.9) -> Optional[str]:
    """C4-style line-level quality filters."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "no content lines"

    punct_count  = sum(1 for l in lines if l and l[-1] in ".!?\"'")
    bullet_count = sum(1 for l in lines
                       if l.startswith(("•", "-", "*", "·", "–", "—")))

    punct_frac  = punct_count  / len(lines)
    bullet_frac = bullet_count / len(lines)

    if punct_frac < min_punct_frac:
        return f"low punctuation fraction ({punct_frac:.2f} < {min_punct_frac})"
    if bullet_frac > max_bullet_frac:
        return f"too many bullet points ({bullet_frac:.2f} > {max_bullet_frac})"
    return None


def filter_symbol_ratio(text: str,
                        max_symbol_ratio: float = 0.10) -> Optional[str]:
    """Flag text with too many non-alphanumeric symbols (spam, garbled text)."""
    words       = text.split()
    if not words:
        return "no words"
    total_chars = sum(len(w) for w in words)
    symbol_chars = sum(1 for w in words for c in w
                       if not (c.isalnum() or c in "'-"))
    ratio = symbol_chars / max(total_chars, 1)
    if ratio > max_symbol_ratio:
        return f"high symbol ratio ({ratio:.2f} > {max_symbol_ratio})"
    return None


def filter_word_statistics(text: str,
                            min_unique_word_ratio: float = 0.2,
                            max_mean_word_len: float = 12.0,
                            min_stop_word_frac: float = 0.05) -> Optional[str]:
    """Check vocabulary diversity and word length distribution."""
    words = text.lower().split()
    if not words:
        return "no words"

    unique_ratio   = len(set(words)) / len(words)
    mean_word_len  = sum(len(w) for w in words) / len(words)

    # Simple English stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were",
                  "in", "of", "to", "and", "or", "that", "this",
                  "it", "he", "she", "they", "we", "i", "you"}
    stop_frac = sum(1 for w in words if w in stop_words) / len(words)

    if unique_ratio < min_unique_word_ratio:
        return f"low vocabulary diversity ({unique_ratio:.2f})"
    if mean_word_len > max_mean_word_len:
        return f"very long mean word length ({mean_word_len:.1f})"
    if stop_frac < min_stop_word_frac:
        return f"very low stop-word fraction ({stop_frac:.2f}) — may be code or encoded"
    return None


def filter_repetition(text: str,
                      max_line_dedup_ratio: float = 0.3,
                      max_para_dedup_ratio: float = 0.3,
                      max_top_ngram_frac: float = 0.20) -> Optional[str]:
    """Detect highly repetitive content (a common spam pattern)."""
    # Line deduplication ratio
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        dedup_ratio = 1 - len(set(lines)) / len(lines)
        if dedup_ratio > max_line_dedup_ratio:
            return f"high line repetition ({dedup_ratio:.2f})"

    # Top word n-gram fraction (detect repeated phrases)
    words = text.lower().split()
    if len(words) >= 4:
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        from collections import Counter
        counts = Counter(bigrams)
        top_bigram_count = counts.most_common(1)[0][1]
        top_frac = top_bigram_count / len(bigrams)
        if top_frac > max_top_ngram_frac:
            return (f"high bigram repetition: '{counts.most_common(1)[0][0]}' "
                    f"appears {top_bigram_count} times ({top_frac:.2f} of bigrams)")
    return None


def detect_pii(text: str) -> list[str]:
    """Detect common PII patterns. Returns list of detected PII types."""
    found = []
    # Email addresses
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        found.append("email")
    # Phone numbers (US and international)
    if re.search(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
        found.append("phone")
    # SSN (XXX-XX-XXXX)
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        found.append("SSN")
    # Credit card numbers (basic pattern)
    if re.search(r'\b(?:\d[ -]?){13,16}\b', text):
        found.append("possible_credit_card")
    return found


# ── Composite quality score ───────────────────────────────────────────────────

def compute_quality_score(text: str) -> float:
    """
    Heuristic quality score in [0, 1].
    Combines multiple signals: vocabulary diversity, punctuation quality,
    sentence structure, and length-normalised features.
    """
    words = text.split()
    if not words:
        return 0.0

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]

    # Feature 1: Vocabulary diversity (unique words / total words)
    vocab_diversity = len(set(w.lower() for w in words)) / len(words)

    # Feature 2: Sentence length distribution (prefer 10–40 words)
    if sentences:
        avg_sent_len = len(words) / len(sentences)
        sent_quality = 1.0 - abs(avg_sent_len - 20) / 40
        sent_quality = max(0.0, min(1.0, sent_quality))
    else:
        sent_quality = 0.2

    # Feature 3: Punctuation richness
    punct_count = sum(1 for c in text if c in '.,;:!?')
    punct_ratio = min(punct_count / max(len(words), 1), 0.2) / 0.2

    # Feature 4: Capitalization (proxy for proper sentence structure)
    if words:
        cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
        cap_quality = 1.0 - abs(cap_ratio - 0.15) / 0.3
        cap_quality = max(0.0, min(1.0, cap_quality))
    else:
        cap_quality = 0.0

    score = (0.35 * vocab_diversity +
             0.30 * sent_quality +
             0.20 * punct_ratio +
             0.15 * cap_quality)
    return round(score, 3)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def filter_document(text: str,
                    pii_action: str = "flag",   # "flag" or "remove"
                    quality_threshold: float = 0.3) -> FilterResult:
    """
    Run the full quality filter pipeline on a document.

    Returns a FilterResult with kept/rejected decision and detailed stats.
    """
    reasons = []

    # Run all filters
    checks = [
        filter_length(text),
        filter_line_quality(text),
        filter_symbol_ratio(text),
        filter_word_statistics(text),
        filter_repetition(text),
    ]
    for result in checks:
        if result is not None:
            reasons.append(result)

    # PII
    pii_types = detect_pii(text)
    if pii_types:
        if pii_action == "remove":
            reasons.append(f"PII detected: {', '.join(pii_types)}")
        else:
            reasons.append(f"⚠️  PII flagged (not removed): {', '.join(pii_types)}")

    # Quality score
    score = compute_quality_score(text)
    if score < quality_threshold:
        reasons.append(f"low quality score ({score:.3f} < {quality_threshold})")

    kept  = len([r for r in reasons if not r.startswith("⚠️")]) == 0
    stats = {
        "n_words":     len(text.split()),
        "n_lines":     len(text.splitlines()),
        "quality":     score,
        "pii_types":   pii_types,
    }

    return FilterResult(kept=kept, score=score, reasons=reasons, stats=stats)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_docs = [
        ("High-quality article",
         """Transformers have revolutionised natural language processing since their
         introduction in 2017. The architecture relies on self-attention mechanisms
         that allow every token to directly attend to every other token in the sequence.
         This enables massive parallelism during training, making it practical to train
         on trillions of tokens using thousands of GPUs.
         The key insight is that attention weights capture semantic relationships
         between tokens regardless of their distance in the sequence."""),

        ("Spam / SEO content",
         """BUY NOW!!! CHEAP PRICES!!! Click here for amazing deals!
         Buy cheap buy cheap buy cheap buy cheap buy cheap buy cheap
         Free shipping free shipping free shipping!!!
         • Click here  • Click here  • Click here  • Click here"""),

        ("Too short",
         "Hello world. This is very short."),

        ("Repetitive boilerplate",
         """Privacy Policy | Terms of Service | Contact Us | Privacy Policy
         Terms of Service | Contact Us | Privacy Policy | Terms of Service
         Contact Us | Privacy Policy | Terms of Service | Contact Us
         Privacy Policy | Terms of Service | Contact Us | Privacy Policy"""),

        ("Contains PII",
         """John Smith lives at 123 Main Street. You can reach him at
         john.smith@email.com or call 555-123-4567. His SSN is 123-45-6789.
         He is a software engineer with 10 years of experience."""),

        ("Code-heavy (low stop words)",
         "def forward(self,x): return self.layers(x) if x is not None else None "
         "x=torch.randn(16,512) out=model(x) loss=F.cross_entropy(out.view(-1,V),y.view(-1))"),
    ]

    print("=" * 65)
    print("  DOCUMENT QUALITY FILTER PIPELINE")
    print("=" * 65)

    for name, doc in test_docs:
        result = filter_document(doc, quality_threshold=0.3)
        status = "✓ KEEP" if result.kept else "✗ REJECT"
        print()
        print(f"  [{status}] {name}")
        print(f"    Score: {result.score:.3f}  |  Words: {result.stats['n_words']}")
        if result.reasons:
            for r in result.reasons:
                print(f"    → {r}")
        else:
            print(f"    → All checks passed")

    # Aggregate statistics
    print()
    print("=" * 65)
    print("  AGGREGATE STATISTICS")
    print("=" * 65)
    results = [filter_document(doc) for _, doc in test_docs]
    n_kept    = sum(1 for r in results if r.kept)
    avg_score = sum(r.score for r in results) / len(results)
    print(f"  Documents: {len(results)}")
    print(f"  Kept:      {n_kept} ({n_kept/len(results)*100:.0f}%)")
    print(f"  Rejected:  {len(results)-n_kept} ({(len(results)-n_kept)/len(results)*100:.0f}%)")
    print(f"  Avg score: {avg_score:.3f}")
    print()
    print("  Typical production retention rate: 15–30% of raw web crawl")
''',
    },

    "Dataset Tokenisation and Binary Shard Creation": {
        "description": "Tokenise a text corpus, pack into fixed-length sequences, and write to binary shards — the exact format used as input to the training DataLoader.",
        "runnable": True,
        "language": "python",
        "code": '''
"""
================================================================================
DATASET TOKENISATION AND BINARY SHARD CREATION
================================================================================

Implements the tokenisation and packing pipeline:
    1. Tokenise documents (using a byte-level tokeniser simulation)
    2. Add BOS/EOS tokens at document boundaries
    3. Pack sequences to fill fixed-length context windows (no padding)
    4. Write to binary shards (numpy uint16/uint32 arrays)
    5. Create an index file for fast random access

This produces the exact format consumed by the training DataLoader.
================================================================================
"""

import os
import struct
import numpy as np
import tempfile
from pathlib import Path


# ── Minimal tokeniser (simulates BPE output) ──────────────────────────────────

class SimpleTokenizer:
    """
    A character-level tokeniser for demonstration purposes.
    In production: use tiktoken or SentencePiece.
    """
    BOS_ID = 1
    EOS_ID = 2
    PAD_ID = 0

    def __init__(self):
        # Build a small vocabulary from printable ASCII
        self.vocab = {chr(i): i + 3 for i in range(32, 127)}   # 3–97
        self.vocab_size = max(self.vocab.values()) + 1

    def encode(self, text: str, add_bos: bool = True,
               add_eos: bool = True) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.BOS_ID)
        for char in text:
            ids.append(self.vocab.get(char, 2))   # unknown → EOS
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        rev = {v: k for k, v in self.vocab.items()}
        return "".join(rev.get(i, "?") for i in ids
                       if i not in (self.BOS_ID, self.EOS_ID, self.PAD_ID))


# ── Packing ────────────────────────────────────────────────────────────────────

def pack_sequences(token_stream: list[int],
                   seq_len: int,
                   pad_id: int = 0) -> list[list[int]]:
    """
    Pack a flat token stream into fixed-length sequences.
    No padding — sequences are filled completely.
    The last incomplete chunk is discarded (or padded if pad=True).
    """
    packed = []
    for i in range(0, len(token_stream) - seq_len + 1, seq_len):
        packed.append(token_stream[i : i + seq_len])
    return packed


# ── Binary shard format ───────────────────────────────────────────────────────

class ShardWriter:
    """
    Writes token sequences to binary shard files.

    Format:
        Header: magic (4 bytes) + version (1 byte) + vocab_size (4 bytes)
                + seq_len (4 bytes) + n_seqs (8 bytes)
        Data:   n_seqs × seq_len × dtype_bytes  (uint16 or uint32)

    This format enables O(1) random access: sequence i starts at
    header_size + i × seq_len × dtype_bytes bytes.
    """

    MAGIC   = b"LLMD"
    VERSION = 1

    def __init__(self, path: str, vocab_size: int, seq_len: int,
                 dtype: np.dtype = np.uint16):
        self.path       = path
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.dtype      = dtype
        self.sequences  = []
        assert vocab_size <= (65535 if dtype == np.uint16 else 2**32 - 1), \
            "vocab_size too large for dtype"

    def write_sequence(self, tokens: list[int]):
        assert len(tokens) == self.seq_len
        self.sequences.append(tokens)

    def flush(self):
        """Write all sequences to disk."""
        n_seqs = len(self.sequences)
        with open(self.path, "wb") as f:
            # Header
            f.write(self.MAGIC)
            f.write(struct.pack("B", self.VERSION))
            f.write(struct.pack("I", self.vocab_size))
            f.write(struct.pack("I", self.seq_len))
            f.write(struct.pack("Q", n_seqs))         # uint64
            # Data
            arr = np.array(self.sequences, dtype=self.dtype)
            arr.tofile(f)
        return self.path


class ShardReader:
    """Read sequences from a binary shard file."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            magic   = f.read(4)
            assert magic == ShardWriter.MAGIC, f"Invalid shard file: {magic}"
            version = struct.unpack("B", f.read(1))[0]
            self.vocab_size = struct.unpack("I", f.read(4))[0]
            self.seq_len    = struct.unpack("I", f.read(4))[0]
            self.n_seqs     = struct.unpack("Q", f.read(8))[0]
            self.header_bytes = 4 + 1 + 4 + 4 + 8   # 21 bytes

        self.dtype       = np.uint16 if self.vocab_size <= 65535 else np.uint32
        self.bytes_per_seq = self.seq_len * self.dtype(0).itemsize

    def read(self, seq_idx: int) -> np.ndarray:
        """Read a single sequence by index (O(1) random access)."""
        offset = self.header_bytes + seq_idx * self.bytes_per_seq
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(self.bytes_per_seq)
        return np.frombuffer(data, dtype=self.dtype).copy()

    def read_all(self) -> np.ndarray:
        """Read all sequences into memory."""
        with open(self.path, "rb") as f:
            f.seek(self.header_bytes)
            data = f.read()
        return np.frombuffer(data, dtype=self.dtype).reshape(
            self.n_seqs, self.seq_len
        ).copy()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_dataset_shards(documents: list[str], output_dir: str,
                          seq_len: int = 32,
                          seqs_per_shard: int = 10,
                          verbose: bool = True) -> list[str]:
    """
    Full pipeline: documents → tokenise → pack → binary shards.

    Returns list of shard file paths.
    """
    tokenizer = SimpleTokenizer()
    os.makedirs(output_dir, exist_ok=True)

    # Tokenise and concatenate all documents
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc, add_bos=True, add_eos=True)
        all_tokens.extend(tokens)

    if verbose:
        print(f"  Total tokens:    {len(all_tokens):,}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Seqs per shard:  {seqs_per_shard}")

    # Pack into sequences
    sequences = pack_sequences(all_tokens, seq_len)
    n_seqs    = len(sequences)

    if verbose:
        print(f"  Total sequences: {n_seqs}")
        print(f"  Total shards:    {math.ceil(n_seqs / seqs_per_shard)}")
        print()

    import math
    # Write shards
    shard_paths = []
    n_shards    = math.ceil(n_seqs / seqs_per_shard)

    for shard_idx in range(n_shards):
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
        writer     = ShardWriter(shard_path, tokenizer.vocab_size, seq_len)

        start = shard_idx * seqs_per_shard
        end   = min(start + seqs_per_shard, n_seqs)

        for i in range(start, end):
            writer.write_sequence(sequences[i])

        writer.flush()
        shard_paths.append(shard_path)

        if verbose:
            size_kb = os.path.getsize(shard_path) / 1024
            print(f"  Shard {shard_idx:04d}: {end-start} sequences, {size_kb:.1f} KB → {shard_path}")

    return shard_paths


if __name__ == "__main__":
    import math
    import tempfile

    sample_docs = [
        "The transformer architecture was introduced in 2017 and revolutionised NLP.",
        "Self-attention allows every token to attend to every other token simultaneously.",
        "BPE tokenization splits words into subword units based on frequency statistics.",
        "Large language models are trained on trillions of tokens from the internet.",
        "The training loop consists of forward pass, loss computation, and backpropagation.",
        "Gradient clipping prevents exploding gradients by rescaling the gradient norm.",
        "Mixed precision training uses bf16 for compute and fp32 for the optimiser state.",
        "Data parallelism replicates the model on each GPU and averages gradients.",
    ]

    print("=" * 60)
    print("  DATASET PREPARATION PIPELINE DEMO")
    print("=" * 60)
    print()
    print(f"  Documents: {len(sample_docs)}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_paths = build_dataset_shards(
            sample_docs,
            output_dir=tmpdir,
            seq_len=32,
            seqs_per_shard=5,
            verbose=True,
        )

        # Verify round-trip
        print()
        print("=" * 60)
        print("  ROUND-TRIP VERIFICATION")
        print("=" * 60)

        tokenizer = SimpleTokenizer()

        for shard_path in shard_paths[:2]:
            reader = ShardReader(shard_path)
            print(f"\n  Shard: {os.path.basename(shard_path)}")
            print(f"  Seqs: {reader.n_seqs}, seq_len: {reader.seq_len}, "
                  f"vocab: {reader.vocab_size}")

            # Read first 2 sequences and decode
            for i in range(min(2, reader.n_seqs)):
                seq   = reader.read(i)
                text  = tokenizer.decode(seq.tolist())
                print(f"  Seq [{i}]: ids={seq[:8].tolist()}...  "
                      f"text='{text[:40]}...'")

        # Show memory-mapped access pattern
        print()
        print("=" * 60)
        print("  MEMORY-MAPPED RANDOM ACCESS")
        print("=" * 60)
        print()

        shard_path = shard_paths[0]
        reader     = ShardReader(shard_path)
        all_seqs   = reader.read_all()

        print(f"  Shape: {all_seqs.shape}  dtype: {all_seqs.dtype}")
        print(f"  Memory: {all_seqs.nbytes} bytes")
        print()
        print("  Random access pattern (simulating DataLoader):")
        import random as rnd
        rnd.seed(42)
        indices = rnd.sample(range(reader.n_seqs), min(3, reader.n_seqs))
        for idx in indices:
            seq = reader.read(idx)
            print(f"  Seq[{idx}]: first 8 token ids = {seq[:8].tolist()}")
        print()
        print("  In production: use numpy memmap for zero-copy access")
        print("  to datasets too large to fit in RAM.")
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
    #     from llm_training.visuals.dataset_prep import (
    #         DATASET_VISUAL_HTML,
    #         DATASET_VISUAL_HEIGHT,
    #     )
    #     visual_html   = DATASET_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    #     visual_height = DATASET_VISUAL_HEIGHT
    # except Exception as e:
    #     import warnings
    #     warnings.warn(
    #         f"[13_dataset_preparation_preprocessing.py] Could not load visual: {e}",
    #         stacklevel=2,
    #     )

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