"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""
import re
import textwrap

TOPIC_NAME    = "LLM - Retrieval-Augmented Generation (RAG)"
DISPLAY_NAME  = "LLM - Retrieval-Augmented Generation (RAG)"
ICON          = "🔍"
SUBTITLE      = "How LLMs retrieve external knowledge — pipelines, memory, and Agentic RAG"
CATEGORY      = "LLM Architecture"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""

Order:

    MODULE 07 · Retrieval-Augmented Generation (RAG)
          ├── why_rag_exists             The knowledge cutoff problem
          ├── rag_pipeline_overview      Index → Retrieve → Augment → Generate
          ├── vector_embeddings          Text → numbers that encode meaning
          ├── similarity_search          Cosine similarity — how chunks are found
          ├── chunking_strategies        Fixed-size / sentence / paragraph / semantic
          ├── retrieval_quality_problems When wrong chunks cause hallucination
          ├── reranking                  A second pass to reorder by actual relevance
          ├── context_stuffing           Placing retrieved chunks into the window
          ├── rag_vs_fine_tuning         Retrieve vs. bake into weights
          ├── hybrid_search              BM25 (keyword) + vector (semantic)
          ├── memory_in_rag              Short-term, long-term, episodic memory layers
          ├── context_window_management  Budgeting tokens: chunks + history + system
          └── agentic_rag                Multi-step retrieval, planning, and memory loops


---

### Major RAG Variations:

The RAG Family Tree
    
    01. Naive RAG — The baseline. Chunk → embed → retrieve top-k → generate. Simple but brittle.
    
    02. Advanced RAG — Improves naive RAG with better chunking (sliding window, semantic), query rewriting, 
        and reranking retrieved docs before generation.
        
    03. GraphRAG — Builds a knowledge graph from documents. Retrieves via entity relationships instead of 
        pure vector similarity. Great for multi-hop reasoning.
        
    04. Modular RAG — Treats each RAG component (retriever, reranker, generator) as a swappable module. 
        You compose pipelines based on the use case.
        
    05. Agentic RAG — The LLM decides when and how to retrieve — iteratively, adaptively, and with 
        self-correction. Uses tools/agents around the retrieval loop.
        
    06. Corrective RAG (CRAG) — Adds a grading step: retrieved documents are evaluated for relevance. 
        If they're poor, it falls back to web search or paraphrases the query.
        
    07. Self-RAG — The model generates "reflection tokens" to decide if retrieval is even needed, 
        and critiques its own output after generation.
        
    08. Hybrid RAG — Combines dense (vector/semantic) and sparse (BM25/keyword) retrieval, then merges 
        results with reciprocal rank fusion (RRF).
        
    09. HyDE (Hypothetical Document Embeddings) — Generates a hypothetical ideal answer first, 
        then uses its embedding to retrieve real docs. Bridges query-document gap.
        
    10. Multimodal RAG — Retrieves across text, images, tables, and audio. Uses vision encoders 
        alongside text embeddings.
        
    11. Long-Context RAG — Leverages massive context windows (128K+) to stuff many chunks in rather 
        than aggressively pruning, reducing retrieval errors.
        
    12. Federated RAG — Retrieves from multiple distributed knowledge bases simultaneously 
        (e.g., different departments, data silos).
        


## WHY RAG EXISTS — The Knowledge Cutoff Problem

An LLM is a frozen snapshot. Every fact it knows was baked in during pre-training, which
ended at a specific date. Once the model is deployed, those weights never change.

This creates three classes of problems:

    •   Staleness   — the model cannot know about events after its training cutoff.
                      Ask it about last week's earnings call, it will either refuse
                      or hallucinate a plausible-sounding answer.

    •   Privacy     — sensitive company documents, internal wikis, customer records
                      can never be included in public pre-training data.
                      Yet users need the model to reason over them.

    •   Precision   — even for topics the model was trained on, the knowledge is
                      diffuse. It has averaged over thousands of sources. A
                      verbatim contract clause or an exact product spec number
                      is not reliably stored in weights at full precision.

Retrieval-Augmented Generation solves all three by externalising memory. Instead of
asking "what do your weights know?", RAG asks "what does the relevant document say?"
and hands the relevant text to the model at inference time, inside the context window.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE CORE INSIGHT                                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      WITHOUT RAG                          WITH RAG
      ─────────────────────                ──────────────────────────────
      User: "What is our                   User: "What is our refund
             refund policy?"                      policy?"
                 │                                     │
                 ▼                                     ▼
          LLM consults its              Vector search finds 3 chunks
          frozen weights.               from policy_2024.pdf
          Policy is not                        │
          in training data.                    ▼
                 │                    LLM reads chunks + user question
                 ▼                             │
          Hallucination or                     ▼
          "I don't know"             Grounded, precise answer ✓


**KEY CONCEPT:**
RAG does not change the model's weights. It changes what the model READS
at inference time. The LLM is still a frozen function — RAG just gives it
better inputs.


---


## RAG PIPELINE OVERVIEW — Index → Retrieve → Augment → Generate

A RAG system has two distinct phases that run at different times:

    OFFLINE PHASE  (runs once, or on a schedule)
    ─────────────────────────────────────────────────────────────────────

    Raw Documents
    (PDFs, wikis, emails, code, etc.)
         │
         │  1. LOAD & PARSE
         │     Extract text from whatever format the document is in.
         │     Preserve structure (headings, tables) where possible.
         ▼
    Cleaned Text Corpus
         │
         │  2. CHUNK
         │     Split into smaller pieces — "chunks" — that fit
         │     comfortably inside a context window with room to spare.
         ▼
    List of Text Chunks
         │
         │  3. EMBED
         │     Pass each chunk through an embedding model.
         │     Each chunk → a dense float vector (e.g. 1536 dimensions).
         ▼
    List of (chunk_text, embedding_vector) pairs
         │
         │  4. INDEX
         │     Store vectors in a vector database (Pinecone, Weaviate,
         │     pgvector, Chroma, FAISS…).
         │     Store original chunk text alongside (or separately).
         ▼
    Vector Index  ←── ready for queries


    ONLINE PHASE  (runs for every user query)
    ─────────────────────────────────────────────────────────────────────

    User Query
         │
         │  5. EMBED THE QUERY
         │     Same embedding model used in the offline phase.
         │     Query → query vector.
         ▼
    Query Vector
         │
         │  6. RETRIEVE
         │     ANN (approximate nearest neighbour) search over the index.
         │     Return the top-k most similar chunks (k = 3–10 typically).
         ▼
    Retrieved Chunks  [chunk_1, chunk_2, chunk_3, …]
         │
         │  7. (OPTIONAL) RERANK
         │     A cross-encoder re-scores the chunks against the query
         │     with higher accuracy than pure vector similarity.
         ▼
    Re-ordered Chunks
         │
         │  8. AUGMENT
         │     Insert retrieved chunks into the prompt template
         │     alongside the user's original question.
         ▼
    Augmented Prompt (fits inside the context window)
         │
         │  9. GENERATE
         │     LLM reads the augmented prompt and produces an answer
         │     grounded in the retrieved material.
         ▼
    Final Answer → User


    ╔══════════════════════════════════════════════════════════════════╗
    ║  OFFLINE vs ONLINE — WHEN EACH PHASE RUNS                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      +────────────────┬────────────────────────────────────────────+
      │  Phase         │  When it runs                              │
      +────────────────┼────────────────────────────────────────────+
      │  Indexing      │  Once at setup; re-run when docs change    │
      │  Embedding     │  At indexing time; and per query (online)  │
      │  ANN search    │  Every query — milliseconds                │
      │  LLM generate  │  Every query — hundreds of ms to seconds   │
      +────────────────┴────────────────────────────────────────────+


---


## VECTOR EMBEDDINGS — Text → Numbers That Encode Meaning

An embedding model is a neural network trained to convert arbitrary text into
a fixed-length vector of floating-point numbers such that texts with similar
meaning produce vectors that are geometrically close to each other.

    "The cat sat on the mat."   →   [0.21,  0.87, -0.14,  0.53, ...]  (1536 dims)
    "A feline rested on a rug." →   [0.19,  0.85, -0.16,  0.51, ...]  (1536 dims)
    "Stock markets fell 3%."    →   [-0.62, 0.14,  0.77, -0.33, ...]  (1536 dims)

The first two sentences are semantically similar — their vectors are close.
The third is unrelated — its vector is far away in the 1536-dimensional space.

This is fundamentally different from keyword matching. Keywords only match
when the exact characters appear. Embeddings capture that "cat" and "feline"
mean the same thing, even though they share no letters.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  HOW AN EMBEDDING IS PRODUCED                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      Input text (any length, up to model's context limit)
              │
              ▼
      Tokenise  →  [token_1, token_2, … token_N]
              │
              ▼
      Transformer encoder layers
      (self-attention, feed-forward, residuals — same architecture
       as the decoder, but no autoregressive masking)
              │
              ▼
      Per-token hidden states  shape: (N, d_model)
              │
              │  Mean-pool or CLS-pool across all token positions
              ▼
      Single vector  shape: (d_model,)  e.g. (1536,)
              │
              │  L2 normalise to unit sphere
              ▼
      Embedding vector  ← stored in the vector DB


**KEY CONCEPT:**
The embedding is not the weights of the model — it is the OUTPUT of a single
forward pass. Different texts produce different embeddings from the same
frozen model. The embedding model itself is never updated during RAG operation.


## Popular Embedding Models

    +──────────────────────────────┬────────────┬────────────────────────────+
    │  Model                       │  Dims      │  Notes                     │
    +──────────────────────────────┼────────────┼────────────────────────────+
    │  text-embedding-3-small      │  1536      │  OpenAI, cheap, good       │
    │  text-embedding-3-large      │  3072      │  OpenAI, highest quality   │
    │  text-embedding-ada-002      │  1536      │  OpenAI legacy             │
    │  all-MiniLM-L6-v2            │  384       │  Open source, very fast    │
    │  BGE-large-en-v1.5           │  1024      │  BAAI, strong on retrieval │
    │  E5-mistral-7b-instruct      │  4096      │  LLM-based, very powerful  │
    │  Cohere embed-english-v3     │  1024      │  Strong multilingual       │
    +──────────────────────────────┴────────────┴────────────────────────────+


---


## SIMILARITY SEARCH — Cosine Similarity: How Chunks Are Found

Once every chunk has been embedded into a vector, retrieval is the problem of
finding which vectors are closest to the query vector. The standard distance
metric is cosine similarity.

## Cosine Similarity Formula

    cosine_similarity(A, B) = (A · B) / (|A| × |B|)

    Where:
        A · B  = dot product of the two vectors (sum of element-wise products)
        |A|    = L2 norm (length) of vector A
        |B|    = L2 norm (length) of vector B

    Result ranges from -1.0 to +1.0:

        +1.0   identical direction — same meaning
         0.0   perpendicular — completely unrelated meaning
        -1.0   opposite direction — antonymous meaning


If vectors are pre-normalised to unit length (which all good embedding pipelines do),
the cosine similarity reduces to a simple dot product:

    cosine_similarity(A, B) = A · B    (when |A| = |B| = 1)

This makes it extremely fast to compute.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  COSINE SIMILARITY VISUALISED (2D simplification)                ║
    ╚══════════════════════════════════════════════════════════════════╝

       Dimension 2
         ▲
         │          ● "feline rested"    ← close angle to query
         │        ↗
         │      ↗  query: "cat sat"
         │    ↗ ↗
         │   ●  ←── "cat sat on mat"    ← very close angle
         │
         │
         │                      ● "stock markets"  ← far angle
         └────────────────────────────────────────▶ Dimension 1

    Small angle between vectors = high cosine similarity = semantically close
    Large angle between vectors = low cosine similarity = unrelated


## Approximate Nearest Neighbour (ANN)

An exact nearest neighbour search over 10 million 1536-dimensional vectors is
too slow for production. Vector databases use ANN algorithms that trade a tiny
fraction of recall accuracy for orders-of-magnitude speed improvement:

    +─────────────┬────────────────────────────────────────────────────────+
    │  Algorithm  │  How it works                                          │
    +─────────────┼────────────────────────────────────────────────────────+
    │  HNSW       │  Builds a hierarchical graph of vectors; traverses     │
    │             │  from coarse to fine layers. Default in most DBs.      │
    │  IVF        │  Clusters vectors into Voronoi cells; only searches    │
    │             │  nearby cells at query time.                           │
    │  FAISS      │  Facebook's library; supports both HNSW and IVF;       │
    │             │  runs entirely in-memory.                              │
    │  ScaNN      │  Google's approach; aggressive quantisation + scoring. │
    +─────────────┴────────────────────────────────────────────────────────+


---


## CHUNKING STRATEGIES — Fixed / Sentence / Paragraph / Semantic

Chunking is arguably the most impactful decision in the entire RAG pipeline.
Too large a chunk and the embedding averages over too much text — retrieval precision drops.
Too small and individual chunks lack context — retrieved snippets are meaningless in isolation.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE GOLDILOCKS PROBLEM OF CHUNKING                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      CHUNK TOO LARGE (e.g. full document)
      ─────────────────────────────────────
      Embedding represents the AVERAGE meaning of 10,000 words.
      A query about one specific paragraph gets diluted by all
      the other paragraphs. Precision suffers.

      CHUNK TOO SMALL (e.g. individual sentences)
      ─────────────────────────────────────────────
      "The fee is $50." — retrieved correctly, but the reader
      has no idea what the fee is FOR. Context is missing.
      The LLM cannot write a useful answer from fragments.

      CHUNK JUST RIGHT (~200–500 tokens, with overlap)
      ──────────────────────────────────────────────────
      Each chunk is semantically coherent and self-contained
      enough to be useful, but small enough for the embedding
      to represent it precisely.


## Strategy 1 — Fixed-Size Chunking

Split text every N tokens (e.g. 512 tokens), with an overlap of M tokens
(e.g. 64 tokens) to prevent context from being severed at chunk boundaries.

    Document tokens: [1 … 512 | 449 … 960 | 897 … 1408 | … ]
                               ↑             ↑
                           overlap 64    overlap 64

    PROS: Simple, predictable, uniform index size.
    CONS: Cuts mid-sentence, mid-paragraph. Semantic continuity is not guaranteed.
    BEST FOR: Homogeneous text (transcripts, articles) where structure is implicit.


## Strategy 2 — Sentence-Level Chunking

Split on sentence boundaries (period + capital, or using a sentence tokeniser).
Optionally group N sentences together as one chunk.

    PROS: Preserves grammatical completeness. Good for Q&A over fact-dense text.
    CONS: Sentence lengths vary wildly — chunk sizes are unpredictable.
    BEST FOR: Legal documents, medical literature, anything with dense facts.


## Strategy 3 — Paragraph / Section Chunking

Split on double newlines, Markdown headings, or HTML section tags.
Each natural paragraph or section becomes one chunk.

    PROS: Highest semantic coherence. Paragraphs are already written as units.
    CONS: Paragraphs vary from 20 to 800 tokens — very uneven.
    BEST FOR: Wikis, documentation, structured reports.


## Strategy 4 — Semantic / Agentic Chunking

Use a sentence embedding model to detect where meaning "shifts" in the document.
A new chunk begins when consecutive sentence embeddings diverge beyond a threshold.

    [sent_1] ──── [sent_2] ──── [sent_3]  ──X── [sent_4] ──── [sent_5]
                                             ↑
                              cosine similarity drops below threshold
                              → start a new chunk here

    PROS: Chunks align with actual topic transitions, not arbitrary token counts.
    CONS: Slower to compute. Requires a second embedding pass at indexing time.
    BEST FOR: Long documents with multiple distinct topics (research papers, books).


## Overlap — Why It Matters

    ┌─────────────────────────────────────────────────────────────────┐
    │  WITHOUT OVERLAP                                                │
    │                                                                 │
    │  Chunk A: "The rate is determined by the central bank.  End."  │
    │  Chunk B: "Beginning. It is reviewed quarterly and may…"       │
    │                                                                 │
    │  A query about "how often is the rate reviewed" may match      │
    │  chunk B but miss the critical "central bank" context from A.  │
    │                                                                 │
    │  WITH OVERLAP (last 64 tokens of A repeated at start of B)     │
    │                                                                 │
    │  Chunk B: "The rate is determined by the central bank.         │
    │            It is reviewed quarterly and may…"                  │
    │                                                                 │
    │  Now chunk B is self-contained. The retriever finds it and     │
    │  the LLM has the full picture.                                  │
    └─────────────────────────────────────────────────────────────────┘


---


## RETRIEVAL QUALITY PROBLEMS — When Wrong Chunks Cause Hallucination

Retrieval failure is the most common source of errors in a RAG system.
The LLM is only as good as what it is given to read. If the wrong chunks are
retrieved, the LLM will either answer from those wrong chunks (wrong answer)
or ignore them and fall back to its frozen weights (hallucination).

    ╔══════════════════════════════════════════════════════════════════╗
    ║  TAXONOMY OF RETRIEVAL FAILURES                                  ║
    ╚══════════════════════════════════════════════════════════════════╝

    1. SEMANTIC MISMATCH
       The query embedding and the correct chunk's embedding are not
       close, even though the chunk contains the answer.

       Example:
         Query: "What happens if I miss a payment?"
         Correct chunk title: "Default and Delinquency — Section 4.2"
         The words "miss a payment" nowhere appear in the chunk.
         The chunk talks about "obligation non-fulfilment."
         Embedding similarity may be low despite the chunk being the answer.

       Fix: Better query reformulation, hypothetical document embeddings (HyDE),
            or hybrid search (adding keyword matching to catch exact phrases).


    2. CHUNK BOUNDARY SLICING
       The answer spans two chunks but only one is retrieved.

       Example:
         "The interest rate is 4.5%"  — in chunk 12
         "...applicable from Jan 2024" — in chunk 13
         Query retrieves only chunk 12 → incomplete answer.

       Fix: Larger chunks, more overlap, or "parent document retrieval"
            (retrieve small chunk, return its parent section).


    3. STALE INDEX
       Documents in the vector index have been updated in the source,
       but the index has not been re-embedded.

       Example: Policy changed in March. Embedding index still from December.
       User gets the old policy. The LLM confidently answers from stale data.

       Fix: Incremental re-indexing pipelines triggered on document change.


    4. WRONG GRANULARITY
       Top-k returns k=3 chunks, but the correct answer requires
       synthesising information from 8 different sections.

       Fix: Increase k; use map-reduce summarisation; use Agentic RAG.


    5. DISTRACTOR INJECTION
       A retrieved chunk is topically adjacent but factually incorrect
       for this specific question. The LLM is confused by conflicting
       signals in the context.

       Example: Retrieving the US refund policy when the user asked about
                the EU refund policy. Both documents are about "refund policy."

       Fix: Metadata filters (filter by region, date, document_type before ANN);
            more precise reranking.


---


## RERANKING — A Second Pass to Reorder by Actual Relevance

The ANN retrieval step uses fast approximate similarity to fetch a candidate set
(typically the top-20 or top-50 chunks). The initial ranking is good enough to
find the right documents but not precise enough to determine which of those
documents is MOST relevant.

A reranker runs a second, more expensive but more accurate relevance pass over
the candidate set.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  BI-ENCODER vs CROSS-ENCODER                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      BI-ENCODER (embedding model — used for retrieval)
      ──────────────────────────────────────────────────
      Query  →  encode independently  →  query_vector
      Chunk  →  encode independently  →  chunk_vector
                                        cosine(q, c) = score

      Fast because vectors can be pre-computed for all chunks.
      But "independent encoding" means Q and C never see each other
      during encoding — interactions are missed.


      CROSS-ENCODER (reranker — used for reranking)
      ──────────────────────────────────────────────────
      [Query + Chunk]  →  encode TOGETHER  →  single relevance score

      Slow because it must re-run for every (query, chunk) pair.
      But Q and C attend to each other — captures nuanced relevance
      that bi-encoders miss entirely.


    ┌─────────────────────────────────────────────────────────────────┐
    │                  RETRIEVAL + RERANKING PIPELINE                 │
    │                                                                 │
    │  Vector DB  →  top-50 candidates (fast ANN, bi-encoder)         │
    │                       │                                         │
    │                       ▼                                         │
    │             Cross-encoder reranker                              │
    │             scores all 50 (query, chunk) pairs                  │
    │                       │                                         │
    │                       ▼                                         │
    │  Re-ordered top-5 →  injected into context window               │
    └─────────────────────────────────────────────────────────────────┘

    The ANN step handles SCALE (millions of chunks).
    The reranker handles PRECISION (which 5 of the 50 matter most).


## Popular Reranking Models

    +──────────────────────────────────┬────────────────────────────────+
    │  Model                           │  Notes                         │
    +──────────────────────────────────┼────────────────────────────────+
    │  Cohere rerank-english-v3.0      │  API-based, strong & easy      │
    │  BGE-reranker-large              │  Open source, very strong      │
    │  cross-encoder/ms-marco-*        │  Sentence Transformers library │
    │  GPT-4o (as a reranker via API)  │  Overkill but high accuracy    │
    +──────────────────────────────────┴────────────────────────────────+


---


## CONTEXT STUFFING — Placing Retrieved Chunks Into the Window

After retrieval (and optional reranking), the chunks must be physically placed
inside the prompt that is sent to the LLM. This is "context stuffing."

    ╔══════════════════════════════════════════════════════════════════╗
    ║  A TYPICAL AUGMENTED PROMPT STRUCTURE                            ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌─────────────────────────────────────────────────────────────┐
      │  SYSTEM PROMPT                                              │
      │  "You are a helpful assistant. Answer questions using       │
      │   ONLY the context provided below. If the answer is not     │
      │   in the context, say 'I don't know.'"                      │
      ├─────────────────────────────────────────────────────────────┤
      │  RETRIEVED CONTEXT                                          │
      │                                                             │
      │  [Source: policy_2024.pdf, section 3.1]                     │
      │  "Refund requests must be submitted within 30 days of       │
      │   purchase via the customer portal. Processing takes        │
      │   5–7 business days…"                                       │
      │                                                             │
      │  [Source: faq_billing.md]                                   │
      │  "For international orders, refund timelines may extend     │
      │   to 14 business days due to currency conversion…"          │
      ├─────────────────────────────────────────────────────────────┤
      │  CONVERSATION HISTORY (if multi-turn)                       │
      │  User: "I placed an order last week."                       │
      │  Assistant: "Happy to help. What's your order number?"      │
      ├─────────────────────────────────────────────────────────────┤
      │  USER QUERY                                                 │
      │  "How long will my refund take?"                            │
      └─────────────────────────────────────────────────────────────┘

The order matters. Research shows LLMs exhibit a "lost in the middle" bias:
information in the middle of a long context is attended to less reliably
than information at the very beginning or very end.

    BEST PRACTICE: Place the most relevant chunk either first or last
    in the context block, not buried in the middle.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  LOST-IN-THE-MIDDLE EFFECT                                       ║
    ╚══════════════════════════════════════════════════════════════════╝

      Attention-to-context by position (empirical finding):

      Start of context  ████████████████████  HIGH
      Middle            █████████             MEDIUM-LOW
      End of context    ████████████████████  HIGH

      Practical implication:
      If you have 5 chunks and only 1 is truly correct, put that
      chunk first or last. Never bury it at position 3 of 5.


---


## RAG vs FINE-TUNING — Retrieve vs. Bake Into Weights

Two strategies exist for giving a model specialised knowledge:

    ╔══════════════════════════════════════════════════════════════════╗
    ║  RAG vs FINE-TUNING COMPARISON                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌───────────────────┬─────────────────────┬─────────────────────┐
      │                   │  RAG                │  Fine-Tuning        │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  How it works     │  Retrieves docs at  │  Trains new weights │
      │                   │  inference time     │  on target data     │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  Knowledge        │  Updated by         │  Frozen until next  │
      │  freshness        │  updating index     │  fine-tuning run    │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  Verbatim recall  │  Excellent          │  Unreliable         │
      │                   │  (chunk text        │  (diffuse in        │
      │                   │   injected as-is)   │   weights)          │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  Tone / style     │  No change          │  Can learn new      │
      │                   │                     │  writing patterns   │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  Cost             │  Compute per query  │  Large up-front     │
      │                   │  (retrieval + LLM)  │  training cost      │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  Hallucination    │  Low when retrieval │  Can memorise and   │
      │  risk             │  is correct         │  confabulate        │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  Auditability     │  High — you can     │  Low — knowledge    │
      │                   │  trace which chunk  │  is opaque in       │
      │                   │  produced the answer│  parameters         │
      ├───────────────────┼─────────────────────┼─────────────────────┤
      │  Best for         │  Factual lookup,    │  Style/tone,        │
      │                   │  private docs,      │  task format,       │
      │                   │  changing data      │  domain jargon      │
      └───────────────────┴─────────────────────┴─────────────────────┘

**KEY CONCEPT:**
Fine-tuning teaches a model HOW to behave. RAG teaches a model WHAT to say
on a specific topic. The two are complementary — fine-tune for format and
tone, RAG for live, updatable, verifiable knowledge.


---


## HYBRID SEARCH — BM25 (Keyword) + Vector (Semantic)

Pure vector search is excellent for semantic matching but has one well-known
weakness: rare words, product codes, names, and identifiers that appear
rarely in the embedding model's training data are poorly handled.

    Example: Query = "error code E-4501-B"
    Vector search: "error" and "code" map to known concepts, but "E-4501-B"
                   is an opaque string. The embedding averages over its bytes.
                   Precision is low for exact identifier lookup.

BM25 (Best Match 25) is a classical information retrieval algorithm that scores
documents by exact term frequency and inverse document frequency. It excels
precisely where vector search struggles — rare, specific tokens.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  BM25 INTUITION                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝

      BM25 score(query Q, document D):

          score = Σ  IDF(term) × TF(term, D) × (k₁ + 1)
                       ─────────────────────────────────
                       TF(term, D) + k₁ × (1 - b + b × |D|/avgdl)

          IDF(term)  = log( (N - df + 0.5) / (df + 0.5) )
                       penalises common words ("the", "and")
                       rewards rare words ("E-4501-B")

          TF(term,D) = raw count of term in document, but SATURATED
                       — doubling word count does not double score

          k₁, b      = tuning constants (typically k₁=1.5, b=0.75)
          |D|/avgdl  = normalises for document length


## Hybrid Fusion

Both BM25 and vector search return ranked lists of chunks with scores. These
lists must be merged into a single ranking. The standard method is
Reciprocal Rank Fusion (RRF):

    RRF_score(chunk) = Σ  1 / (k + rank_in_list_i)

    Where k = 60 (a tuning constant), and the sum is over each
    retrieval system (BM25 list, vector list).

    This rewards chunks that appear high in BOTH lists without
    requiring the scores to be on the same numeric scale.


    ┌─────────────────────────────────────────────────────────────────┐
    │                  HYBRID SEARCH PIPELINE                         │
    │                                                                 │
    │  User Query                                                     │
    │       │                                                         │
    │       ├──────────────────────────────────┐                     │
    │       │                                  │                     │
    │       ▼                                  ▼                     │
    │  BM25 keyword search            Vector ANN search              │
    │  top-50 by TF-IDF               top-50 by cosine sim           │
    │       │                                  │                     │
    │       └──────────────┬───────────────────┘                     │
    │                      ▼                                         │
    │             RRF score fusion                                    │
    │                      │                                         │
    │                      ▼                                         │
    │         Top-k fused candidates → (optional) rerank             │
    │                      │                                         │
    │                      ▼                                         │
    │              Inject into context window                         │
    └─────────────────────────────────────────────────────────────────┘


---


## MEMORY IN RAG — Short-Term, Long-Term, and Episodic Memory Layers

Up to this point we have treated RAG as stateless: every query starts fresh,
retrieves chunks, and discards them. But real applications — customer support
assistants, personal AI tutors, enterprise copilots — need to remember things
across queries and across sessions. Memory in RAG systems is stratified into
three conceptually distinct layers.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THREE MEMORY LAYERS IN A RAG SYSTEM                             ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌──────────────────────────────────────────────────────────────┐
      │  LAYER 1 — IN-CONTEXT MEMORY  (short-term)                   │
      │                                                              │
      │  What it is:  the context window itself.                     │
      │  Scope:       lives only during a single LLM call.           │
      │  Contents:    system prompt, retrieved chunks, conversation  │
      │               history for the current session, user query.   │
      │  Duration:    wiped after the call completes.                │
      │                                                              │
      │  Analogy: working memory / RAM.                              │
      │  Limit: bounded strictly by the model's context window size. │
      └──────────────────────────────────────────────────────────────┘

      ┌──────────────────────────────────────────────────────────────┐
      │  LAYER 2 — EXTERNAL MEMORY  (long-term)                      │
      │                                                              │
      │  What it is:  the vector database + document store.          │
      │  Scope:       persistent across all sessions and users.      │
      │  Contents:    embedded representations of all indexed docs;  │
      │               structured metadata; raw chunk text.           │
      │  Duration:    indefinite — until explicitly deleted.         │
      │                                                              │
      │  Analogy: a library of books — vast but slow to read.        │
      │  Access: via a retrieval query, not direct inspection.       │
      └──────────────────────────────────────────────────────────────┘

      ┌──────────────────────────────────────────────────────────────┐
      │  LAYER 3 — EPISODIC / SESSION MEMORY  (medium-term)          │
      │                                                              │
      │  What it is:  a stored record of past conversation turns.    │
      │  Scope:       per-user, persisted in a database.             │
      │  Contents:    compressed summaries of prior sessions, key    │
      │               facts extracted from past conversations,       │
      │               user preferences, entity mentions.             │
      │  Duration:    configurable — days to months.                 │
      │                                                              │
      │  Analogy: a journal or notebook. Not every word, but the     │
      │           key facts you jotted down.                         │
      │  Access: retrieved like any other document — via vector      │
      │           search over session embeddings, or SQL lookup.     │
      └──────────────────────────────────────────────────────────────┘


## How the Three Layers Interact at Query Time

    ╔══════════════════════════════════════════════════════════════════╗
    ║  MEMORY RETRIEVAL FLOW (per query)                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      User sends query: "What was that pricing tier we discussed?"
               │
               ├────────────────────────────────────────────────────┐
               │                                                     │
               ▼                                                     ▼
      EPISODIC MEMORY SEARCH                           EXTERNAL MEMORY SEARCH
      Vector search over past                          Vector search over product
      session summaries for this                       documentation, pricing sheets,
      user → "User asked about                         spec docs
              Tier 3 pricing on                               │
              2024-11-03"                                     │
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                          Merge results: session context
                          + document context injected into
                          IN-CONTEXT WINDOW
                                      │
                                      ▼
                            LLM generates answer
                                      │
                                      ▼
                            New turn STORED BACK into
                            episodic memory store


## Episodic Memory — What Gets Stored and How

Storing raw transcript turns is wasteful. A 60-minute conversation may produce
40,000 tokens. Repeatedly loading that into the context window on every new
query is expensive and wastes space for retrieved chunks.

Better practice is to extract and compress:

    END-OF-SESSION MEMORY EXTRACTION PIPELINE

    Full session transcript
         │
         │  LLM summarisation pass  ("Summarise key facts, decisions,
         │                           and user preferences from this chat")
         ▼
    Session Summary (300–600 tokens)  ← stored as a vector + text
         │
         │  Entity extraction pass  ("List all named entities, products,
         │                           and specific values mentioned")
         ▼
    Structured Entities
    { user_name: "Alice", tier_discussed: "Tier 3",
      date_of_discussion: "2024-11-03", budget: "$5,000/mo" }
         │
         ▼
    Stored in episodic memory DB  (vector + JSON)
    ← retrievable by future sessions for the same user


---


## CONTEXT WINDOW MANAGEMENT IN RAG

In RAG systems the context window is a shared budget. Every token used by one
component is a token unavailable to another. Careful budgeting is non-optional.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  THE CONTEXT WINDOW TOKEN BUDGET                                 ║
    ╚══════════════════════════════════════════════════════════════════╝

      Model: Claude 3.5 Sonnet (200K context window)

      ┌─────────────────────────────────────────────────────────────┐
      │  SYSTEM PROMPT              ~  500 tokens   (fixed)         │
      │  RETRIEVED CHUNKS           ~ 4,000 tokens  (variable)      │
      │  EPISODIC MEMORY SUMMARY    ~  600 tokens   (variable)      │
      │  CONVERSATION HISTORY       ~ 2,000 tokens  (grows/turn)    │
      │  USER QUERY                 ~  100 tokens   (per query)     │
      │  GENERATION BUFFER          ~ 2,000 tokens  (for output)    │
      │  ─────────────────────────────────────────────────          │
      │  TOTAL USED                 ~ 9,200 tokens                  │
      └─────────────────────────────────────────────────────────────┘

      With a 200K window this seems trivial — but at GPT-4 pricing
      (~$10/M input tokens) or when using a 4K context model,
      every token counts enormously.


## The Growing History Problem in RAG

Exactly as with plain LLM calls (see Module 01), each turn in a RAG session
appends to the conversation history that must be re-sent:

    Turn 1:  [system] + [chunk_A] + [user_1]                        → LLM
    Turn 2:  [system] + [chunk_B] + [user_1 + asst_1 + user_2]      → LLM
    Turn 3:  [system] + [chunk_C] + [u1+a1+u2+a2+u3]                → LLM
    Turn 10: [system] + [chunk_D] + [9 turns of history + user_10]   → LLM
                                     ↑
                        History is now thousands of tokens.
                        Chunks are competing for space with history.

Three strategies to manage this:

    STRATEGY 1 — SLIDING WINDOW
    Keep only the last N turns. Drop older turns.
    Simple but loses early context that may still be relevant.

    STRATEGY 2 — SUMMARISE AND COMPRESS
    Every K turns, have the LLM summarise the conversation so far
    into a compact paragraph, replacing the raw turns.
    The summary (200 tokens) replaces K turns (2,000 tokens).

    STRATEGY 3 — RETRIEVE FROM HISTORY
    Store past turns in a separate vector index. Retrieve only
    the historically relevant turns for the current query —
    same mechanism as document retrieval, applied to conversation.


## Token Budget Allocation in Practice

    ╔══════════════════════════════════════════════════════════════════╗
    ║  BUDGET ALLOCATION RULES OF THUMB                                ║
    ╚══════════════════════════════════════════════════════════════════╝

      For a 16K context window (tight budget):
      ─────────────────────────────────────────
       System prompt         ≤  500 tokens  (10 % of input budget)
       Retrieved chunks      ≤ 6,000 tokens (priority — this is RAG's job)
       Conversation history  ≤ 3,000 tokens (compress aggressively)
       Query + misc          ≤   500 tokens
       Output reservation    ≤ 2,000 tokens
       ─────────────────────────────────────
       Total budget used     = 12,000 tokens

      For a 128K context window (comfortable):
      ─────────────────────────────────────────
       System prompt          ≤  2,000 tokens
       Retrieved chunks       ≤ 40,000 tokens (can pull 80+ chunks)
       Episodic memory        ≤  5,000 tokens
       Full conversation      ≤ 30,000 tokens
       Query + misc           ≤  1,000 tokens
       Output reservation     ≤  8,000 tokens
       ─────────────────────────────────────────
       Total budget used      = 86,000 tokens


**KEY CONCEPT:**
Even with a 200K context window, a RAG system must prioritise what goes into
context. The act of retrieval IS the act of selecting what the model will
read. Every byte of history and every chunk occupies the same window. There
is no such thing as "limitless context" — there is only the question of how
well you manage the budget you have.


---


## AGENTIC RAG — Multi-Step Retrieval, Planning, and Memory Loops

Standard RAG is reactive: one query → one retrieval → one answer. This breaks
down when questions are complex, multi-hop, or require iterative exploration.

Agentic RAG places the LLM inside a control loop. Instead of being a passive
reader of retrieved chunks, the LLM becomes an active agent that:

    •   Plans what to retrieve
    •   Decides whether retrieved information is sufficient
    •   Issues follow-up queries if it is not
    •   Synthesises findings across multiple retrieval rounds
    •   Maintains a working memory of what it has found so far


    ╔══════════════════════════════════════════════════════════════════╗
    ║  STANDARD RAG vs AGENTIC RAG                                     ║
    ╚══════════════════════════════════════════════════════════════════╝

      STANDARD RAG
      ────────────────────────────────────────────────────────────
      User query
           │
           ▼
      Single retrieval → top-k chunks → LLM → answer
           ↑
      One pass. No iteration. LLM has no choice about what to read.


      AGENTIC RAG
      ────────────────────────────────────────────────────────────
      User query
           │
           ▼
      LLM PLANS:  "To answer this I need to:
                   (1) find the Q3 revenue figures
                   (2) find the Q3 expenses
                   (3) find the prior year comparison"
           │
           ▼  (tool call)
      Retrieval 1: search("Q3 2024 revenue")
           │  result → chunk set A
           ▼
      LLM EVALUATES: "I have revenue but no expense breakdown.
                      I need to retrieve more."
           │
           ▼  (tool call)
      Retrieval 2: search("Q3 2024 operating expenses breakdown")
           │  result → chunk set B
           ▼
      LLM EVALUATES: "I still lack the prior year comparison.
                      Let me query differently."
           │
           ▼  (tool call)
      Retrieval 3: search("Q3 2023 annual report summary")
           │  result → chunk set C
           ▼
      LLM SYNTHESISES across A + B + C
           │
           ▼
      Final comprehensive answer


## The Agentic RAG Control Loop

    ╔══════════════════════════════════════════════════════════════════╗
    ║  AGENTIC RAG ARCHITECTURE                                        ║
    ╚══════════════════════════════════════════════════════════════════╝

                         ┌────────────────────────────┐
                         │      USER QUERY             │
                         └──────────────┬─────────────┘
                                        │
                                        ▼
                         ┌────────────────────────────┐
                         │       PLANNER LLM           │
                         │  "What do I need to find?" │◄──────────┐
                         │  "Is my current info        │           │
                         │   sufficient to answer?"    │           │
                         └──────────────┬─────────────┘           │
                                        │                          │
                    ┌───────────────────┼───────────────────┐      │
                    │                   │                   │      │
                    ▼                   ▼                   ▼      │
            Vector search        BM25 search       SQL / API       │
            tool call            tool call         tool call       │
                    │                   │                   │      │
                    └───────────────────┼───────────────────┘      │
                                        │                          │
                                        ▼                          │
                         ┌────────────────────────────┐            │
                         │    SCRATCHPAD / WORKING     │            │
                         │       MEMORY BUFFER         │            │
                         │  (accumulates findings      │            │
                         │   across retrieval rounds)  │            │
                         └──────────────┬─────────────┘            │
                                        │                          │
                                        ▼                          │
                         ┌────────────────────────────┐            │
                         │   SUFFICIENCY EVALUATOR     │            │
                         │  "Do I have enough to       │            │
                         │   answer confidently?"      │            │
                         └──────────────┬─────────────┘            │
                                        │                          │
                        ┌───────────────┴──────────────┐           │
                        │ NO — need more info           │ YES       │
                        ▼                               ▼           │
             Issue new retrieval query         SYNTHESISER LLM      │
             with refined search terms ────────────────────────────┘  (loop)
                                               Generates final answer
                                                        │
                                                        ▼
                                                  USER RESPONSE


## Why Context Window and Memory Are Critical in Agentic RAG

In a standard single-pass RAG call, context management is a one-time
packaging problem. In Agentic RAG it becomes an active, ongoing challenge
across the full planning loop.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  AGENTIC RAG MEMORY CHALLENGES                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

    CHALLENGE 1 — ACCUMULATING SCRATCHPAD OVERFLOW

      Each retrieval round adds new chunks to the agent's working memory.
      After 5 rounds × 4 chunks × 500 tokens = 10,000 tokens of retrieved
      material, plus the growing chain of agent thoughts.

      Without management: the window fills up and early retrieved
      material gets evicted. The agent "forgets" what it found in
      round 1 by the time it reaches round 5.

      Solution: Maintain a STRUCTURED SCRATCHPAD — a running summary
      of findings that compresses prior rounds:

          Round 1 finding: "Q3 revenue = $4.2M (from annual_report.pdf, p.12)"
          Round 2 finding: "Q3 OPEX = $3.1M (from expense_report.xlsx, sheet 3)"
          Round 3 finding: "Q3 2023 revenue = $3.8M for YoY comparison"

      These structured notes (≈50 tokens each) replace 500-token chunks.


    CHALLENGE 2 — REPLANNING WITH PARTIAL INFORMATION

      The planner LLM must hold the original goal + all prior findings
      + the current question in mind simultaneously.

      As the number of rounds grows, the portion of the context window
      available for the CURRENT retrieval result shrinks.

      Typical context allocation in a 5-round agentic loop:
      ┌──────────────────────────────────────────────────┐
      │  System prompt + agent instructions   ~  800 tok │
      │  Original user goal                   ~  200 tok │
      │  Scratchpad of prior findings         ~ 3,000 tok│  (grows each round)
      │  Current retrieved chunks             ~ 4,000 tok│  (needed this round)
      │  Agent chain-of-thought               ~ 2,000 tok│  (grows each round)
      │  ─────────────────────────────────────────────   │
      │  Total per round-5 call               ~10,000 tok│
      └──────────────────────────────────────────────────┘

      The scratchpad and CoT consume progressively more space.
      If the window is 16K, by round 7 the agent may be unable
      to fit both prior findings AND new retrieved chunks.


    CHALLENGE 3 — CROSS-SESSION GOAL PERSISTENCE

      Agentic tasks can span sessions: a research agent that works
      across multiple user conversations on a complex analysis.

      Without persistent memory, the agent starts from scratch each time.
      With episodic memory, it can:

          •  Resume from where it stopped last session
          •  Re-retrieve only what has changed
          •  Present the user with its current progress state


## Agentic RAG Memory Architecture (Full Picture)

    ╔══════════════════════════════════════════════════════════════════╗
    ║  FOUR MEMORY LAYERS IN AGENTIC RAG                               ║
    ╚══════════════════════════════════════════════════════════════════╝

      ┌────────────────────────────────────────────────────────────┐
      │  LAYER 1 — IN-CONTEXT SCRATCHPAD  (ephemeral, per call)    │
      │                                                            │
      │  Contents: current agent step, retrieved chunks for        │
      │            this round, chain-of-thought, partial findings. │
      │  Lifespan: single LLM call.                                │
      │  Access:   direct — the LLM reads it as plain text.        │
      └────────────────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────────────────┐
      │  LAYER 2 — TASK WORKING MEMORY  (ephemeral, per task)      │
      │                                                            │
      │  Contents: structured notes summarising all rounds so far, │
      │            current plan state (which sub-goals are done),  │
      │            fact table extracted from prior retrievals.     │
      │  Lifespan: full agentic task (may span multiple LLM calls).│
      │  Access:   serialised to JSON between calls, re-injected   │
      │            at the start of the next call's context.        │
      └────────────────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────────────────┐
      │  LAYER 3 — EXTERNAL DOCUMENT INDEX  (persistent)           │
      │                                                            │
      │  Contents: the full vector database of all indexed docs.   │
      │  Lifespan: indefinite.                                     │
      │  Access:   retrieval tool calls — NOT loaded into context  │
      │            until the agent explicitly fetches a chunk.     │
      └────────────────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────────────────┐
      │  LAYER 4 — EPISODIC / USER MEMORY  (persistent)            │
      │                                                            │
      │  Contents: past task outcomes, user preferences, entity    │
      │            facts learned from prior sessions.              │
      │  Lifespan: configurable (days to years).                   │
      │  Access:   retrieved at task start to bootstrap the agent. │
      └────────────────────────────────────────────────────────────┘


## ReAct — The Dominant Agentic RAG Pattern

ReAct (Reason + Act) is the most widely implemented pattern for agentic
retrieval. Each step of the agent loop follows the same structure:

    THOUGHT:  "I need to find the renewal price for Enterprise tier.
               My prior searches returned SMB pricing only.
               I should search for 'enterprise tier renewal pricing 2024'."

    ACTION:   search("enterprise tier renewal pricing 2024")

    OBSERVATION: "Enterprise Tier: $24,000/year. Renewal within 60 days
                  of expiry qualifies for 10% loyalty discount."

    THOUGHT:  "I now have the enterprise renewal price and the discount
               condition. Combined with the SMB data from earlier, I
               can answer the user's full question about pricing tiers."

    ACTION:   finish("Here is a comparison of SMB and Enterprise renewal…")

Each (Thought, Action, Observation) triplet fits in the context window.
Prior triplets are either retained as-is (expensive) or compressed into
the structured scratchpad (efficient).

    ╔══════════════════════════════════════════════════════════════════╗
    ║  REACT LOOP CONTEXT WINDOW BUDGET EXAMPLE                        ║
    ╚══════════════════════════════════════════════════════════════════╝

      Budget: 16K tokens.  Each round consumes:

      ┌─────────────────────────────────────────────────────────────┐
      │  Fixed overhead (system + task goal)         ~  1,000 tok   │
      │  Compressed scratchpad (grows +150/round)    ~  600 tok     │
      │  Latest retrieved chunks (top-3 × 400 tok)   ~ 1,200 tok   │
      │  Agent CoT for this step                     ~  400 tok     │
      │  Prior raw ReAct turns (kept for last 3)     ~ 2,400 tok   │
      │  ───────────────────────────────────────────────────────    │
      │  Per-round budget used                       ~ 5,600 tok   │
      │  Remaining headroom for output + safety      ~10,400 tok   │
      └─────────────────────────────────────────────────────────────┘

      At this burn rate, a 16K window supports ~10 retrieval rounds
      before management strategies (compression, eviction) are needed.


---


## QUICK REFERENCE — Key Numbers and Design Rules


    +──────────────────────────────────┬──────────────────────────────────────────────+
    │  Parameter / Decision            │  Typical Values / Rules of Thumb             │
    +──────────────────────────────────┼──────────────────────────────────────────────+
    │  Chunk size                      │  200–500 tokens; 100–150 tokens for Q&A       │
    │  Chunk overlap                   │  10–15% of chunk size (e.g. 50 tok for 400)   │
    │  Top-k retrieval (initial)       │  20–50 for reranking; 3–8 for direct inject   │
    │  Top-k after reranking           │  3–8 chunks in context                        │
    │  Embedding dimensions            │  384 (fast) → 1536 (balanced) → 3072 (best)   │
    │  BM25 weight in hybrid           │  30–50% of fusion score; tune per domain      │
    │  Context chunk budget            │  30–50% of total context window               │
    │  History compression trigger     │  Every 6–10 turns or when history > 3K tokens │
    │  Agentic retrieval rounds        │  2–5 typical; >10 is a planning problem        │
    │  Scratchpad compression ratio    │  10:1 (500 tok chunk → 50 tok fact note)       │
    │  Episodic memory retention       │  30–90 days; indefinite for key entities       │
    +──────────────────────────────────┴──────────────────────────────────────────────+


---


## GLOSSARY OF KEY TERMS

    +──────────────────────────┬──────────────────────────────────────────────────────+
    │  Term                    │  Definition                                          │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  ANN                     │  Approximate Nearest Neighbour — fast search         │
    │                          │  trading small recall loss for large speed gain.     │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Agentic RAG             │  RAG architecture where an LLM iteratively plans     │
    │                          │  and issues multiple retrieval calls.                │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  BM25                    │  Best Match 25 — classical TF-IDF-based keyword      │
    │                          │  retrieval algorithm.                                │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Chunk                   │  A sub-segment of a document used as the unit of     │
    │                          │  indexing and retrieval.                             │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Context stuffing        │  Injecting retrieved chunks into the LLM prompt.     │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Cross-encoder           │  Reranker that encodes query + chunk jointly for     │
    │                          │  precise relevance scoring.                          │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Embedding               │  A dense float vector representing the meaning of    │
    │                          │  text, produced by an encoder model.                 │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Episodic memory         │  Stored summaries and entity facts from prior        │
    │                          │  sessions; retrieved to bootstrap new sessions.      │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  HNSW                    │  Hierarchical Navigable Small World — the dominant   │
    │                          │  ANN graph algorithm used by most vector DBs.        │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  HyDE                    │  Hypothetical Document Embeddings — generate a       │
    │                          │  hypothetical answer, embed it, search from that     │
    │                          │  embedding to improve retrieval precision.           │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Knowledge cutoff        │  The date after which an LLM has no pre-trained      │
    │                          │  knowledge — the core motivation for RAG.            │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Lost in the middle      │  Empirical finding that LLMs attend less reliably    │
    │                          │  to context in the middle of a long window.          │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  RAG                     │  Retrieval-Augmented Generation — architecture that  │
    │                          │  retrieves external documents at inference time.     │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  ReAct                   │  Reason + Act — alternating Thought / Action /       │
    │                          │  Observation loop for agentic LLM systems.           │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Reranker                │  A cross-encoder that re-scores a candidate set of   │
    │                          │  chunks after initial ANN retrieval.                 │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  RRF                     │  Reciprocal Rank Fusion — score-independent method   │
    │                          │  to merge ranked lists from multiple retrievers.     │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Scratchpad              │  In-context working memory buffer in Agentic RAG;    │
    │                          │  compressed summary of multi-round findings.         │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Semantic chunking       │  Splitting documents at embedding-detected topic     │
    │                          │  boundaries rather than fixed token counts.          │
    +──────────────────────────┼──────────────────────────────────────────────────────+
    │  Vector database         │  Storage and retrieval system optimised for ANN      │
    │                          │  search over high-dimensional float vectors.         │
    +──────────────────────────┴──────────────────────────────────────────────────────+

---


### Rabbit Hole Topics 

The documents are never merged with the model. The model's weights are completely untouched. 
What actually happens is closer to selective, dynamic, just-in-time reading. 
The model reads a small excerpt of a relevant document for the duration of one single 
forward pass, and then that text is gone. Nothing was integrated. 
The model is identical after the call to what it was before.
The more precise mental model is that RAG separates two things that fine-tuning conflates:

What the model knows how to do — reasoning, language, instruction-following — lives in the 
weights, permanently.
What the model reads to answer a specific question — the retrieved chunks — lives in the 
context window, temporarily.

This distinction matters practically. 
If you "integrated" documents into the model (fine-tuning), updating your knowledge base means 
retraining. With RAG, you update the vector index and the model is instantly using the new 
information on the next query, with no weight changes at all.

A sharper one-sentence description would be: RAG is a retrieval system that dynamically 
selects the most relevant fragments from an external document store and places them into 
the model's context window at inference time, without modifying the model itself.

The hard part of RAG isn't the "putting text in a prompt" step — that's trivial. 
The hard part is the retrieval machinery that decides which fragments to pull from 
potentially millions of chunks, in milliseconds, with high precision. 
That's the chunking, embedding, ANN indexing, reranking, and hybrid search work — 
none of which has anything to do with "integrating documents."


### Context WIndow Curiosity 

"I want to understand how context windows are physically managed in both standard 
LLM deployments and multi-agent RAG systems:

**Context window storage in a standard LLM**
 
    Where does the active context window actually live during inference? 
    Is the entire context (prompt + conversation history + KV cache) held in GPU VRAM, 
    system RAM, or is any portion of it spilled to disk? And how does this change as 
    the context grows longer?
    
    Context management in a multi-agent RAG system — When multiple agents are operating in 
    parallel or in sequence, how is context handled across agent boundaries?
        
        * Does each agent maintain its own isolated context window?
        * Or is there a shared context store that all agents read from and write to?
        * How is relevant context selectively passed between agents — is it the full history, 
          a summary, or only retrieved chunks from the RAG pipeline?

    Context window limits per agent — Is it a common architectural decision to deliberately 
    cap each individual agent's context window to a fixed token range in order to control 
    memory usage and inference cost? What are the tradeoffs of doing this vs. allowing each 
    agent unrestricted context?
    
    Persistent vs. ephemeral context — Is context ever intentionally saved to local disk or 
    an external vector store between agent calls, and then retrieved on demand? Or is the 
    expectation that all context remains in-memory for the duration of a pipeline run?"
    
## Explanation 

**These are four distinct but deeply connected questions.**

## 1. Where Does the Context Window Actually Live During Inference?

    The context window is not a single monolithic thing stored in one place. 
    It is better understood as two separate concerns: the token sequence (the text) 
    and the KV cache (the computed representations of that text). 
    They live in different places and are managed differently.
    
## The KV cache lives in GPU VRAM — and it dominates everything.

When you send a prompt to a model, the first thing that happens is the pre-fill pass: all prompt tokens are processed 
in parallel, and for every token, at every transformer layer, the Key and Value matrices are computed and stored. 
These stored K/V pairs are the KV cache. 

Every subsequent generated token reads from this cache rather than recomputing it. 
The KV cache is the memory-intensive object, and it lives entirely in GPU VRAM because it must be read on every 
single decode step. 
Even a brief trip to system RAM would introduce a latency penalty that makes generation feel broken.

The raw token IDs themselves — the integers representing your prompt text — are tiny by comparison. 
A 10,000-token prompt is about 40KB of integers. 
The KV cache for that same prompt on a 70B-parameter model is on the order of several gigabytes. 
The numbers look like this at a concrete level:

    **KV cache memory = 2 × num_layers × num_heads × d_head × seq_len × bytes_per_element**
    
    For a 70B model (80 layers, 64 heads, d_head=128) at 10K tokens in fp16:
    = 2 × 80 × 64 × 128 × 10,000 × 2 bytes ≈ 33 GB

This is why long-context inference is primarily a VRAM problem, not a compute problem.

## **What happens when context grows and VRAM pressure increases ?**

In a well-managed production inference server (vLLM, TensorRT-LLM, SGLang), the KV cache is managed using a 
technique called PagedAttention, which treats GPU VRAM like virtual memory in an operating system — the KV 
cache is divided into fixed-size blocks (pages), and blocks are allocated and freed dynamically per sequence. 

Under moderate load, everything stays in VRAM. Under heavy concurrent load, some KV cache pages can be swapped 
to system RAM (CPU memory) when VRAM is saturated, and pulled back to VRAM when that sequence becomes active again. 
This is analogous to OS page swapping — it keeps throughput up but adds latency to the affected sequence.

Disk is almost never involved in the active context. Disk access is four to five orders of magnitude slower 
than VRAM reads. No production inference system spills active KV cache to disk during a live request. 
Disk comes into the picture only for prefix caching — storing pre-computed KV activations for a fixed system 
prompt across restarts, so the server doesn't have to recompute the same 2,000-token system prompt from 
scratch on every boot.

So the physical layering looks like this:


    GPU VRAM (HBM)      ← KV cache for all active sequences; model weights
                          Read speed ~3,000 GB/s. Everything here is "live."
    
    CPU System RAM      ← Overflow KV pages under memory pressure (swapped)
                          Read speed ~50–100 GB/s. Adds latency.
    
    NVMe SSD            ← Model weights at rest; prefix cache snapshots
                          Read speed ~5–10 GB/s. Not touched during inference.

As context grows longer, the KV cache grows linearly with sequence length. 
The practical effect is that each concurrent user session occupies more VRAM, which reduces the number of 
sessions the GPU can handle simultaneously. This is the core reason why inference providers charge more 
per token for long-context models — it isn't the compute cost that scales most aggressively, it's the VRAM cost.


## 2. Context Handling Across Agent Boundaries in a Multi-Agent RAG System

Each agent maintains its own isolated context window. There is no such thing as a shared context store that agents 
directly read and write simultaneously — because the context window is not an external data structure, it is the 
input tensor to a specific LLM call. 
You cannot have two agents "share" a context window any more than two programs can share a function's call stack. 
Each agent call is a separate forward pass with its own KV cache that lives and dies with that call.

What gets passed between agents is a message or payload, not raw context. 
The architectural question is what that payload contains, and this is where the real design choices live.

**Three common patterns for inter-agent context passing:**

Pattern A — Full history handoff. Agent A completes its work, serialises its entire conversation history as a list 
            of {role, content} messages, and passes it to Agent B, which prepends it to its own context. 
            Simple, lossless, and extremely expensive as the history grows. Used when Agent B genuinely needs to 
            reason over everything Agent A did — for example, a critic agent reviewing a writer agent's full 
            reasoning chain.
            
Pattern B — Summary handoff. Before Agent A hands off, it runs a final LLM call to compress its work into a 
            structured summary: "I retrieved X, found Y, concluded Z, and here is my output." 
            Agent B receives only the summary, not the raw history. This is the most common pattern in production 
            multi-agent systems because it keeps each agent's context predictable and bounded regardless of how 
            much work the upstream agent did.
            
Pattern C — Structured artifact handoff. Agent A produces a typed output — a JSON object, a database row, a list 
            of retrieved chunks with source metadata — and passes only that structured artifact. 
            Agent B's context window contains only its own system prompt, the structured input, and whatever 
            it retrieves from the RAG pipeline. This is the cleanest architecture and the one most amenable to 
            debugging, but it requires that inter-agent communication be well-specified upfront.
            
In a RAG-augmented multi-agent system, Pattern C is often combined with a shared vector store. 
The vector store is not a "shared context window" — it is an external database. 
Each agent independently queries it and retrieves whatever chunks are relevant to its current sub-task. 

Agent A might query the vector store and write a summary of its findings back to the store as a new document, 
which Agent B then retrieves. 

The vector store acts as a shared external memory that agents communicate through indirectly, mediated by retrieval 
queries rather than direct memory access.



    Agent A                     Shared Vector Store              Agent B
   │                               │                            │
   │── query("topic X") ──────────►│                            │
   │◄─ chunks [1,2,3] ─────────────│                            │
   │   (processes, generates       │                            │
   │    summary of findings)       │                            │
   │── write(summary_doc) ────────►│                            │
   │                               │◄─── query("topic X") ──────│
   │                               │──── [chunks + summary] ───►│
   │                               │                            │ (reads Agent A's
   │                               │                            │  summary as a doc)
   
   
This is an important architectural point: in well-designed multi-agent RAG, agents do not directly read each other's 
context windows. 
They communicate through the retrieval layer, which means their interactions are asynchronous, decoupled, and 
searchable.


## 3. Deliberately Capping Each Agent's Context Window

Yes, this is extremely common in production systems, and it is usually the right call. 
The practice is sometimes called context budgeting or window pinning.
The core argument for capping: if you allow each agent unrestricted context, you give up predictability, 
cost control, and latency guarantees simultaneously. 
An agent that is allowed to accumulate 50,000 tokens of context before making a call is an agent whose inference 
cost is unpredictable, whose VRAM footprint is unpredictable, and whose response latency is unpredictable. 
In a pipeline with five agents in sequence, that unpredictability compounds.

A hard cap — say, 8,192 tokens per agent — forces the system architect to be explicit about what each agent 
actually needs to know. That pressure produces better designs. It forces you to write tighter summaries, 
more precise RAG queries, and cleaner inter-agent handoffs. It also produces systems that are easier to test, 
because the input to each agent is bounded and you can reason about worst-case costs.

**The tradeoffs in concrete terms:**

                    CAPPED CONTEXT               UNRESTRICTED CONTEXT
                    ─────────────────────────    ─────────────────────────
Inference cost      Predictable, bounded         Scales with accumulated history
VRAM usage          Fixed per agent slot         Can saturate GPU under long tasks
Latency             Consistent                   Degrades on long inputs
Information loss    Risk — early context         None — all history available
                    may be summarised away
Design pressure     High — forces good           Low — "just pass everything"
                    handoff design                 is always an option
Debugging           Easier — bounded inputs      Harder — context is opaque
                                                 and very large



The case against capping is that summarisation loses information. 
If Agent A summarises 20,000 tokens into 500 tokens before handing off to Agent B, and Agent B needs a specific 
detail that was considered minor at summarisation time, that detail is gone. 
This is a real failure mode. The mitigation is to use the shared vector store pattern — Agent A writes its full 
working set to the store before summarising, so Agent B can always retrieve the raw material if needed.

In practice, a hybrid approach is most common: cap the context passed directly between agents, 
but maintain a full artifact store (vector DB or structured DB) that any agent can query if it 
needs to reach back into prior work.


## 4. Persistent vs. Ephemeral Context

Both patterns are used, and they serve different purposes.

Ephemeral in-memory context is the default for any single pipeline run. The KV cache for each agent call lives in 
VRAM for the duration of that call and is freed immediately after. 
The conversation state (the list of message objects) lives in the orchestration process's heap memory for the 
duration of the pipeline run and is discarded when the run completes. Nothing is written to disk. 
This is appropriate for stateless, single-session pipelines — a user asks a question, agents work, 
answer is returned, everything is discarded.

    * The task spans multiple user sessions (a research agent that runs over hours or days)
    
    * The pipeline must be resumable after failure
    
    * Multiple pipeline runs need to share knowledge
    
    * Compliance or auditability requires that reasoning steps be logged

In these cases, the pattern is to treat the external vector store or a structured database as the 
durable memory layer, and treat in-context state as a cache that is rebuilt from the durable layer 
at the start of each call.


    START OF EACH AGENT CALL
            │
            ▼
    Load task state from DB        ← persisted JSON: sub-goals, completed steps, key facts
            │
            ▼
    Query vector store             ← retrieve relevant prior findings + external docs
            │
            ▼
    Assemble context window        ← system prompt + task state + retrieved chunks + query
            │
            ▼
    LLM forward pass               ← ephemeral: KV cache lives here, dies here
            │
            ▼
    Extract new findings/facts
            │
            ▼
    Write back to DB / vector store ← persist what was learned before the call's state is lost
            │
            ▼
    KV cache freed, context gone   ← nothing is held in memory between calls
    
The key discipline is this: context window = working memory, external store = long-term memory. 
Anything that must survive beyond a single LLM call must be explicitly written to the external 
store before the call ends. 
The context window is not a safe place to store things — it exists only for the microseconds of 
the forward pass and is gone the instant the response is returned.

This is architecturally analogous to how a CPU register works versus RAM versus disk. 
The context window is the register — the fastest possible access, but volatile and tiny relative to 
what the system as a whole knows. 
The vector store is the disk — persistent, large, but requires an explicit read operation to 
bring data into the register for use.
    


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
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
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
        scripts_dir = None
    if main_script is None:
        main_script = None # _MAIN_SCRIPT

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
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────


def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 1600
    try:
        from LLM_Architecture.visuals.rag_visual import (   # ← match your exact folder casing
            RAG_VISUAL_HTML,
            RAG_VISUAL_HEIGHT,
        )
        visual_html   = RAG_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = RAG_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"[07_gmm.py] Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None, # COMPLEXITY,
        "operations":    OPERATIONS,
    }
