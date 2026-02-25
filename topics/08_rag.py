"""Module: 08 · RAG"""

DISPLAY_NAME = "08 · RAG"
ICON         = "🔍"
SUBTITLE     = "Naive to Advanced to GraphRAG — grounding LLMs in external knowledge"

THEORY = """
## 08 · RAG (Retrieval-Augmented Generation)

Retrieval-Augmented Generation is the architectural pattern that grounds a language model's
outputs in external, verifiable knowledge. Instead of relying solely on what was baked into
the weights during pretraining, the model is given relevant passages at inference time — and
those passages become the primary source of truth for the response. Understanding RAG from
first principles means understanding *why* it exists, *how* each stage works, and *where*
each design decision creates trade-offs.

---

### 1 · The Knowledge Problem — Why RAG Exists

A pretrained language model is a compressed, probabilistic representation of the text it
was trained on. That compression has three structural limitations:

**1.1 Staleness.** Weights are frozen after training. Any fact that changed after the
training cutoff is either absent or subtly wrong. A model trained in early 2024 does not
know about a product launched in late 2024.

**1.2 Hallucination under uncertainty.** When a model lacks confident knowledge of a fact,
it does not say "I don't know." It generates the most statistically plausible continuation —
which is often a confident-sounding but fabricated answer. This is an emergent property of
autoregressive training, not a fixable bug in the training data.

**1.3 Capacity constraints.** A 7B parameter model can memorise roughly an order of magnitude
less than a 70B model. Even large models cannot memorise the full content of an enterprise
knowledge base, a legal document corpus, or a product catalog that runs to millions of items.

RAG addresses all three: it retrieves fresh, specific, authoritative passages at inference
time, provides them as explicit context, and allows the model to cite its sources — making
hallucinations both less likely and more detectable.

**The RAG contract.** When RAG works well, the model's job changes from "recall a fact from
weights" to "extract and synthesise information from the provided passages." This is a
dramatically easier task that large language models perform reliably.

---

### 2 · The RAG Pipeline — End-to-End Overview

A RAG system has two major phases: **indexing** (done offline, once) and
**querying** (done online, per request).

```
INDEXING PHASE (offline)
  Raw Documents
      │
      ▼
  Document Processing (parse, clean, metadata)
      │
      ▼
  Chunking (split into retrieval units)
      │
      ▼
  Embedding Model → Dense Vectors
      │
      ▼
  Vector Store (HNSW index)
      │
      ▼  ← also stored
  Inverted Index (BM25, for sparse retrieval)

QUERYING PHASE (online, per request)
  User Query
      │
      ▼
  [Optional] Query Transformation (expansion, HyDE, decomposition)
      │
      ▼
  Retrieval (dense + sparse → hybrid fusion)
      │
      ▼
  [Optional] Reranking (cross-encoder)
      │
      ▼
  Context Assembly (format retrieved passages)
      │
      ▼
  LLM Generation (prompt = system + context + query)
      │
      ▼
  Response (with citations)
```

Each box in this pipeline is a distinct design decision with measurable impact on end-to-end
quality. Naive RAG implements only the minimum viable version of each. Advanced RAG adds
optional enrichment steps that improve retrieval precision and generation quality.

---

### 3 · Document Processing and Chunking

**Why chunking matters.** A document is rarely the right unit of retrieval. A 40-page PDF
is too coarse: retrieving it costs the entire context window and most of the content is
irrelevant to any specific query. A single sentence is too fine: it lacks the surrounding
context that makes it interpretable. The right granularity depends on the retrieval task.

**3.1 Fixed-size chunking.** Split every N tokens (typically 256–512) with an overlap of
M tokens (typically 10–20% of N). Simple, predictable, works well when document sections
are roughly uniform in content density. The overlap prevents a fact from being split across
two chunks where neither half is retrievable.

```
|← chunk 1 (256 tokens) →|
                |← chunk 2 (256 tokens) →|
                          |← chunk 3 →|
          ↑ overlap (50 tokens)
```

**3.2 Sentence / semantic chunking.** Split on sentence or paragraph boundaries, then
merge short segments until a target size is reached. Preserves semantic coherence at the
cost of variable chunk sizes. Libraries like spaCy or NLTK provide sentence segmentation.

**3.3 Recursive character splitting.** Try to split on paragraph breaks first, then
sentences, then words, then characters — stopping when the chunk is small enough.
This is the default in LangChain's RecursiveCharacterTextSplitter and works well
for mixed-format documents.

**3.4 Document-aware / structural chunking.** Use document structure: split on Markdown
headings, HTML section tags, PDF page breaks, or LaTeX \section commands. Each chunk
inherits metadata (page number, section title, document source) that can be used for
filtering during retrieval.

**3.5 Parent-child chunking (small-to-big).** Index small chunks (128 tokens) for precise
retrieval, but at query time retrieve the parent chunk (512 tokens) to give the LLM full
context. This is one of the most impactful Advanced RAG techniques — precision of small
chunks, coherence of large ones.

**Chunking rules of thumb:**
- Chunk size should match the expected span of an answer, not the span of a document.
- Always preserve metadata: source URL, page, section, timestamp, author.
- Token count (not character count) is the right measure — models care about tokens.
- Test retrieval quality at multiple chunk sizes; 256–512 tokens is a common sweet spot.

---

### 4 · Embeddings — Turning Text into Vectors

An embedding model maps a chunk of text to a fixed-length dense vector. Chunks with
similar meaning land close together in this high-dimensional space; dissimilar chunks
land far apart. This is the foundation of semantic retrieval.

**4.1 Architecture.** Most embedding models are encoder-only transformers (BERT-style).
The final [CLS] token embedding or the mean-pooled token embeddings are used as the
chunk representation. Training uses contrastive objectives: positive pairs (query,
relevant passage) are pulled together; negative pairs are pushed apart.

**4.2 Embedding training objectives:**

- **Contrastive loss (SimCSE):** minimize cos_sim(anchor, positive) distance,
  maximise cos_sim(anchor, negative) distance.
- **Multiple Negatives Ranking Loss:** in-batch negatives — other documents in the
  same batch serve as negatives. Efficient; scales to large batches.
- **Matryoshka Representation Learning (MRL):** train embeddings that are useful at
  multiple truncated dimensions (e.g., first 64, 128, 256, 512 dims of a 1536-dim
  vector are each independently trained). Allows dimension reduction with graceful
  accuracy degradation.

**4.3 Key embedding models (2024):**

| Model | Dims | Context | Benchmark (MTEB) |
|---|---|---|---|
| text-embedding-3-small | 1536 | 8,191 | 62.3 |
| text-embedding-3-large | 3072 | 8,191 | 64.6 |
| BGE-M3 (BAAI) | 1024 | 8,192 | 64.3 |
| E5-mistral-7b | 4096 | 32,768 | 66.6 |
| GTE-Qwen2-7B | 3584 | 32,768 | 67.2 |

**4.4 The embedding gap.** Embedding models are typically trained on query-passage pairs
from web search. Domain-specific text (legal, medical, financial, code) may require
fine-tuning on domain data to achieve production-quality retrieval. This is called
domain adaptation of embeddings.

**4.5 Late chunking.** A recent technique: embed the full document with a long-context
embedding model, then extract chunk-level embeddings from the token representations
at the positions corresponding to each chunk. This preserves cross-chunk context in the
chunk embeddings — something standard per-chunk embedding loses.

---

### 5 · Vector Stores and Approximate Nearest Neighbour Search

Given Q (query vector) and a corpus of N chunk vectors, exact nearest-neighbour search
requires computing Q · cᵢ for all N chunks — O(N) per query. At N = 1M chunks, this is
~1 second per query even with optimised BLAS. Production systems need sub-100ms response.

**5.1 Approximate Nearest Neighbour (ANN).** ANN algorithms trade a small amount of
recall (occasionally missing a true nearest neighbour) for orders-of-magnitude faster
search. The key metric is recall@K: what fraction of the true top-K results are returned?

**5.2 HNSW (Hierarchical Navigable Small World).** The dominant ANN algorithm for RAG.
Builds a multi-layer graph where higher layers are sparse (long-range connections) and
lower layers are dense (local connections). Query traversal starts at the top layer,
descends greedily to the bottom, and returns the K nearest nodes.

```
Layer 2 (sparse, long-range):  [A] ─────────────────── [B]
Layer 1 (medium):               [A] ──── [C] ────────── [B]
Layer 0 (dense, local):        [A]─[C]─[D]─[E]─[F]─[G]─[B]
                                 ↑ query enters here at layer 2
```

HNSW parameters: ef_construction (build quality), M (connections per node), ef_search
(search breadth). Higher values → better recall but slower build and search.

**5.3 IVF (Inverted File Index).** Partitions the vector space into K clusters via K-means.
At query time, only the nprobe nearest clusters are searched. Much lower memory than HNSW;
slightly lower recall. Implemented in FAISS (Facebook AI Similarity Search).

**5.4 Product Quantisation (PQ).** Compresses each vector by splitting it into M sub-vectors
and quantising each independently. Reduces memory 4–32× at the cost of ~5–10% recall
degradation. Enables billion-scale corpora to fit in GPU memory.

**5.5 Popular vector databases:**

| System | Index | Filtering | Scale |
|---|---|---|---|
| Pinecone | HNSW/PQ | Yes (metadata) | Managed, TB |
| Weaviate | HNSW | Yes (GraphQL) | Self-hosted |
| Qdrant | HNSW | Yes (rich) | Self-hosted |
| Milvus | IVF + HNSW | Yes | Self-hosted, TB |
| pgvector | IVF | Yes (SQL) | PostgreSQL |
| ChromaDB | HNSW | Yes | Embedded |
| FAISS | IVF/HNSW/PQ | No | Library |

---

### 6 · Retrieval — Dense, Sparse, and Hybrid

**6.1 Dense retrieval.** Embed the query with the same embedding model used for chunks.
Compute cosine similarity between query vector and all chunk vectors. Return top-K by
score. Excels at semantic matching — finds paraphrases and related concepts even when
no words overlap.

```
cos_sim(q, c) = (q · c) / (||q|| · ||c||)
```

**6.2 Sparse retrieval — BM25.** BM25 (Best Match 25) is a probabilistic TF-IDF variant
that scores a passage based on the frequency of query terms in the passage, normalised
for document length. It excels at exact keyword matching — critical for proper nouns,
product codes, version numbers, and technical terms that embedding models may conflate
with similar-sounding words.

```
BM25(q, d) = Σ_t IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1·(1-b+b·|d|/avgdl))
```

Where: tf = term frequency in document, IDF = inverse document frequency across corpus,
k1 ≈ 1.2–2.0 (term saturation), b ≈ 0.75 (length normalisation), avgdl = average
document length.

**6.3 Hybrid retrieval — the best of both worlds.** Dense retrieval misses exact matches;
sparse retrieval misses semantic similarity. Hybrid combines both by running them in
parallel and fusing their ranked lists.

**Reciprocal Rank Fusion (RRF):**
```
RRF_score(doc) = Σ_r 1 / (k + rank_r(doc))
```
Where k = 60 (smoothing constant), rank_r(doc) is the document's rank in retrieval
system r. Documents that rank well in multiple systems score highest. RRF is robust
to score scale differences — it uses ranks, not raw scores.

**6.4 Metadata filtering.** Before ANN search, apply hard filters on chunk metadata:
date range, source domain, document type, author, language. This is called
pre-filtering. It reduces the search space and prevents stale or out-of-scope
documents from contaminating results.

---

### 7 · Reranking — Cross-Encoder Precision

A bi-encoder (embedding model) encodes query and passage independently, enabling fast
ANN search. But this independence is a limitation: the model cannot see the query when
encoding the passage, so it cannot model complex query-passage interactions.

A **cross-encoder reranker** takes the concatenated (query, passage) pair as input and
produces a single relevance score. Because query and passage attend to each other through
all transformer layers, the cross-encoder captures much richer relevance signals — but
it must score each candidate individually, making it O(K) model calls.

**Two-stage retrieval:**
1. **First stage (recall stage):** retrieve top-100 candidates with fast ANN search.
   High recall, modest precision. Cost: one ANN query.
2. **Second stage (precision stage):** rerank top-100 using cross-encoder.
   Return top-K (typically 3–10). Higher precision, lower recall (but recall is already
   high from stage 1). Cost: 100 cross-encoder forward passes.

**Popular rerankers:** Cohere Rerank, BGE-Reranker-v2, Jina Reranker, MixedBread mxbai-rerank.

Reranking typically improves NDCG@10 by 10–20% over dense-only retrieval.

---

### 8 · Naive RAG — The Baseline

Naive RAG is the minimal implementation:
1. Chunk documents with fixed-size splitter.
2. Embed each chunk with an embedding model.
3. Store vectors in a vector DB.
4. At query time: embed query, retrieve top-K by cosine similarity.
5. Assemble prompt: [system] + [retrieved chunks] + [query].
6. Generate response.

**Naive RAG failure modes:**

| Failure | Cause | Frequency |
|---|---|---|
| Low retrieval recall | Query and relevant chunk don't semantically match | High |
| Retrieval of irrelevant chunks | Embedding similarity without keyword specificity | Medium |
| Lost-in-the-middle | Relevant chunk buried in long context | Medium |
| Chunk boundary artifacts | Answer split across two chunks, neither retrieved | Medium |
| Redundant context | Multiple nearly-identical chunks retrieved | Medium |
| No citation | Model doesn't know which chunk each fact came from | High |

Despite these failures, Naive RAG is a strong baseline. Many production systems run on
Naive RAG with careful prompt engineering. Advanced RAG adds targeted fixes for each
failure mode.

---

### 9 · Advanced RAG — Targeted Enhancements

Advanced RAG adds optional modules at three points in the pipeline.

**9.1 Pre-Retrieval Enhancements (improve the query)**

*Query rewriting.* Use a small LLM to reformulate the user's query into a form more
likely to match indexed chunks. Especially useful when the user asks conversationally
("What was that thing about memory limits?") and the relevant chunks use formal technical
language.

*Query expansion.* Generate K variants of the query, retrieve for each, and union the
results. Increases recall at the cost of K× more retrievals.

*HyDE (Hypothetical Document Embeddings, Gao et al. 2022).* Generate a hypothetical
answer document using the LLM (without any retrieval), embed that document, and use
its embedding as the query vector. The hypothesis is: a generated answer is more
semantically similar to a real answer passage than the original question is. Empirically
improves recall by 5–15% on knowledge-intensive tasks.

```
User query: "How does HNSW handle insertions?"
HyDE doc:   "HNSW handles insertions by assigning the new node a random level,
             then connecting it to M nearest neighbours at each level..."
             → embed this → use as query vector
```

*Step-back prompting.* For complex queries, generate a more abstract "step-back question"
and retrieve for that too. "What is the best medication for fever?" might step back to
"What are the mechanisms of fever and antipyretic drugs?" to retrieve foundational
context before specific facts.

*Query decomposition.* Break multi-hop queries into sub-queries, retrieve for each,
and synthesise. "How does the CEO of Company X compare to the CEO of Company Y?"
becomes two sub-queries retrieved and answered independently, then compared.

**9.2 Retrieval Enhancements (improve what is retrieved)**

*Parent-child chunking.* Already described in section 3. Index child chunks; serve
parent chunks to the LLM.

*Contextual retrieval (Anthropic, 2024).* Before indexing, prepend each chunk with a
short LLM-generated summary of how it fits within the document. This enriches the
chunk embedding with document-level context, substantially improving recall.

*Self-querying / metadata filtering.* Parse the user query with an LLM to extract
structured filter predicates (e.g., "date > 2023", "source = 'SEC filing'") and apply
them as pre-filters before ANN search.

*Iterative/adaptive retrieval.* After each retrieval + partial generation step, assess
whether more information is needed and issue follow-up retrieval queries. This is called
FLARE (Forward-Looking Active Retrieval, Jiang et al. 2023).

**9.3 Post-Retrieval Enhancements (improve what the LLM sees)**

*Reranking.* Already described in section 7.

*Context compression / LLMLingua.* Use a small model to compress each retrieved passage,
retaining only the sentences most relevant to the query. Reduces context length by 2–10×,
lowering cost and the lost-in-the-middle risk.

*Lost-in-the-middle mitigation.* Place the most relevant retrieved passage first and last
in the context window, with less relevant passages in the middle. The model attends most
to position 0 and the final positions.

*Deduplication.* Remove near-duplicate retrieved chunks before assembly. MinHash or
embedding cosine similarity can detect near-duplicates efficiently.

---

### 10 · Modular RAG

As RAG systems matured, practitioners found that fixed pipeline topologies were
inflexible. Modular RAG (Gao et al., 2023) treats each RAG component as a swappable
module with a defined interface. The pipeline topology itself becomes a design choice:

**Sequential topology:** Query → Retrieve → Rerank → Generate (classic RAG)
**Branching topology:** Query → [Retrieve A] + [Retrieve B] → Merge → Generate
**Iterative topology:** Query → Retrieve → Generate partial → Re-query → Generate final
**Conditional topology:** Classify query → if factual: RAG; if creative: direct LLM

**Routing.** A router LLM (or a lightweight classifier) decides whether to:
- Use RAG (the query requires external knowledge)
- Use the LLM's parametric knowledge directly (the query is general)
- Use a specific tool (calculator, code interpreter, API)
- Use multiple retrieval sources (web + internal docs + SQL database)

Routing is the key enabler of intelligent agents that use RAG as one tool among many.

---

### 11 · GraphRAG — Reasoning over Knowledge Graphs

Standard RAG retrieves text chunks. GraphRAG (Edge et al., Microsoft 2024) represents the
corpus as a knowledge graph and retrieves structured relational information.

**11.1 Knowledge graph construction.**
1. Apply entity extraction to every chunk: identify named entities (people, places,
   organisations, concepts) and their types.
2. Extract relationships: (entity_A, relation_type, entity_B) triples.
3. Build a graph where nodes are entities and edges are relationships.
4. Cluster the graph into communities (using Leiden or Louvain algorithms).
5. Generate a summary for each community using an LLM.

**11.2 Community summaries as retrieval targets.**
Instead of retrieving raw text chunks, retrieve community summaries. A community summary
represents a coherent cluster of related entities and their relationships — it provides
context that spans multiple source documents, capturing cross-document connections that
chunk-level retrieval misses.

**11.3 When GraphRAG beats chunk RAG.**
GraphRAG excels at global, synthesising questions: "What are the main themes across this
entire corpus?" or "What relationships exist between Entity A and Entity B through indirect
connections?" Standard RAG excels at local, specific questions: "What did Document X say
about topic Y?"

**GraphRAG vs RAG trade-offs:**

| Dimension | Chunk RAG | GraphRAG |
|---|---|---|
| Local fact retrieval | Excellent | Good |
| Global synthesis | Poor | Excellent |
| Multi-hop reasoning | Poor | Good |
| Index build cost | Low (embed + store) | High (LLM extraction) |
| Query latency | Low | Medium |
| Index update | Easy (add chunks) | Hard (rebuild subgraph) |

---

### 12 · RAG Evaluation — RAGAS and Metrics

**12.1 RAGAS (Shahul et al., 2023)** provides a framework for evaluating RAG systems
without human labels, using an LLM as a judge.

**Four core RAGAS metrics:**

*Context Precision.* What fraction of the retrieved context is actually relevant to the
question? Measures retrieval precision.
```
Context Precision = relevant_chunks_retrieved / total_chunks_retrieved
```

*Context Recall.* What fraction of the information needed to answer the question was
present in the retrieved context? Measures retrieval recall.
```
Context Recall = facts_in_context / total_facts_needed
```

*Faithfulness.* Is the generated answer supported by the retrieved context, or does it
contain hallucinated facts? Each claim in the answer is checked against the context.
```
Faithfulness = supported_claims / total_claims_in_answer
```

*Answer Relevance.* Is the answer relevant to the question? Measures whether the model
stayed on topic and answered what was asked.
```
Answer Relevance = cos_sim(embed(answer), embed(question))
```

**12.2 Additional evaluation dimensions:**

- *NDCG@K (Normalised Discounted Cumulative Gain):* Measures retrieval ranking quality,
  accounting for the position of relevant results. Standard IR metric.
- *MRR (Mean Reciprocal Rank):* Average of 1/rank of the first relevant result.
- *Latency:* P50/P95/P99 end-to-end query time.
- *Cost per query:* Embedding + ANN search + reranker + LLM generation.
- *Hallucination rate:* Fraction of claims in answers that are unsupported by context.

**12.3 Evaluation best practices:**
- Build a test set of (query, ground-truth answer, ground-truth relevant passages) triples.
- Measure retrieval and generation metrics separately — a retrieval failure is different
  from a generation failure.
- Use both automated metrics (RAGAS) and periodic human evaluation.
- Track metrics over time as your corpus evolves.

---

### Key Takeaways

- RAG exists to solve three structural LLM limitations: staleness, hallucination, and
  capacity. It changes the model's task from "recall" to "read and synthesise."
- Every stage of the pipeline — chunking, embedding, indexing, retrieval, reranking,
  generation — is a design decision with measurable quality impact.
- Chunk size is the most impactful single parameter: too large loses precision, too small
  loses context. Parent-child chunking is the standard fix.
- Dense (semantic) and sparse (BM25) retrieval have complementary failure modes. Hybrid
  retrieval with RRF fusion consistently beats either alone.
- Two-stage retrieval (ANN + reranker) is the production-grade pattern. Stage 1 maximises
  recall; stage 2 maximises precision.
- Naive RAG is a strong baseline. Advanced RAG adds targeted modules: HyDE improves recall,
  contextual compression reduces cost, reranking improves precision.
- GraphRAG is not a replacement for chunk RAG — it is a complement for global synthesis
  queries that span multiple documents and require multi-hop reasoning.
- RAGAS provides four complementary metrics that evaluate retrieval and generation
  independently. Track all four; optimising one in isolation can degrade others.
"""

OPERATIONS = {
    "1 · Document Chunking Strategies": {
        "description": (
            "Implement and compare fixed-size, sentence-aware, and recursive chunking. "
            "Show token distributions, overlap mechanics, and chunk quality scores."
        ),
        "language": "python",
        "code": """\
import re, math
from collections import Counter

# ── tokeniser (character-level word approximation) ──────────────────────────
def word_tokens(text):
    return re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?|[.,;:!?]", text)

# ── sample corpus ─────────────────────────────────────────────────────────────
DOCUMENT = (
    "Retrieval-Augmented Generation (RAG) is an AI framework that combines the strengths "
    "of traditional information retrieval systems with the capabilities of large language "
    "models. It was first introduced by Lewis et al. in 2020 at Facebook AI Research. "
    "The core idea is straightforward: instead of relying entirely on parametric knowledge "
    "encoded in model weights, RAG systems retrieve relevant documents from an external "
    "knowledge base and provide them as context to the language model during generation. "
    "This approach addresses several fundamental limitations of pure language models. "
    "First, it mitigates hallucinations by grounding responses in retrieved evidence. "
    "Second, it allows the knowledge base to be updated without retraining the model. "
    "Third, it enables the model to cite specific sources for its claims. "
    "The retrieval component typically uses dense vector search over document embeddings. "
    "Embeddings are computed using encoder models such as BERT or its variants. "
    "The vector similarity search is performed using approximate nearest neighbour algorithms "
    "like HNSW or IVF, enabling sub-millisecond retrieval over millions of documents. "
    "Advanced RAG systems incorporate reranking, query expansion, and iterative retrieval. "
    "GraphRAG extends the paradigm by building knowledge graphs from the corpus, enabling "
    "multi-hop reasoning and global synthesis across the entire document collection."
)

# ── Strategy 1: Fixed-size chunking with overlap ─────────────────────────────
def fixed_chunker(text, chunk_tokens=40, overlap_tokens=8):
    tokens = word_tokens(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_text = " ".join(tokens[start:end])
        chunks.append({
            "id":    len(chunks),
            "text":  chunk_text,
            "start": start,
            "end":   end,
            "size":  end - start,
        })
        if end == len(tokens):
            break
        start += chunk_tokens - overlap_tokens
    return chunks

# ── Strategy 2: Sentence-aware chunking ──────────────────────────────────────
def sentence_chunker(text, max_tokens=50):
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current_tokens = []
    for sent in sentences:
        sent_toks = word_tokens(sent)
        if current_tokens and len(current_tokens) + len(sent_toks) > max_tokens:
            chunks.append({
                "id":   len(chunks),
                "text": " ".join(current_tokens),
                "size": len(current_tokens),
            })
            current_tokens = sent_toks
        else:
            current_tokens.extend(sent_toks)
    if current_tokens:
        chunks.append({"id": len(chunks), "text": " ".join(current_tokens), "size": len(current_tokens)})
    return chunks

# ── Strategy 3: Recursive splitter (paragraph > sentence > word) ─────────────
def recursive_chunker(text, max_tokens=50):
    def split_and_merge(text, separators, max_tok):
        sep = separators[0]
        rest = separators[1:]
        parts = text.split(sep) if sep else list(text)
        chunks = []
        current = []
        for part in parts:
            toks = word_tokens(part)
            if current and len(current) + len(toks) > max_tok and rest:
                sub = recursive_chunker(" ".join(current), max_tok)
                chunks.extend(d["text"] for d in sub)
                current = toks
            elif len(toks) > max_tok and rest:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                sub = recursive_chunker(part, max_tok)
                chunks.extend(d["text"] for d in sub)
            else:
                current.extend(toks)
                if len(current) >= max_tok:
                    chunks.append(" ".join(current[:max_tok]))
                    current = current[max_tok:]
        if current:
            chunks.append(" ".join(current))
        return chunks

    raw = split_and_merge(text, [". ", ", ", " "], max_tokens)
    return [{"id": i, "text": c, "size": len(word_tokens(c))} for i, c in enumerate(raw)]

# ── Run all strategies ────────────────────────────────────────────────────────
strategies = [
    ("Fixed-size (40 tok, 8 overlap)", fixed_chunker(DOCUMENT)),
    ("Sentence-aware (max 50 tok)",    sentence_chunker(DOCUMENT)),
    ("Recursive (max 50 tok)",         recursive_chunker(DOCUMENT)),
]

print("=" * 66)
print("Chunking Strategy Comparison")
print("=" * 66)
print(f"  Document size: {len(word_tokens(DOCUMENT))} tokens")
print()
print(f"  {'Strategy':<38} {'Chunks':>7} {'Avg':>6} {'Min':>6} {'Max':>6} {'CV':>6}")
print("  " + "-" * 66)

for name, chunks in strategies:
    sizes = [c["size"] for c in chunks]
    avg   = sum(sizes) / len(sizes)
    mn    = min(sizes)
    mx    = max(sizes)
    cv    = (math.sqrt(sum((s - avg) ** 2 for s in sizes) / len(sizes)) / avg) if avg else 0
    print(f"  {name:<38} {len(chunks):>7} {avg:>6.1f} {mn:>6} {mx:>6} {cv:>6.2f}")

print()
print("  CV = Coefficient of Variation (lower = more uniform chunk sizes)")
print()

# ── Show fixed-size chunks for inspection ─────────────────────────────────────
fixed_chunks = fixed_chunker(DOCUMENT)
print("=" * 66)
print("Fixed-size chunks (first 3, showing overlap):")
print("=" * 66)
for ch in fixed_chunks[:3]:
    preview = ch["text"][:72] + ("..." if len(ch["text"]) > 72 else "")
    print(f"  Chunk {ch['id']:02d} [tokens {ch['start']:3d}-{ch['end']:3d}]  {preview}")

print()
print("  ^ Overlap tokens appear in consecutive chunk beginnings/endings.")
print("  This ensures no fact is split unrecoverably across chunk boundaries.")
""",
    },

    "2 · BM25 Sparse Retrieval": {
        "description": (
            "Implement BM25 from scratch: build an inverted index, compute TF-IDF-like "
            "scores with length normalisation, and retrieve ranked results for sample queries."
        ),
        "language": "python",
        "code": """\
import math, re
from collections import defaultdict, Counter

# ── BM25 parameters ──────────────────────────────────────────────────────────
K1 = 1.5    # term frequency saturation
B  = 0.75   # length normalisation strength

# ── corpus ───────────────────────────────────────────────────────────────────
CORPUS = [
    "HNSW is a graph-based approximate nearest neighbour algorithm used in vector databases.",
    "BM25 is a probabilistic retrieval function based on TF-IDF with length normalisation.",
    "Dense retrieval uses embedding models to encode queries and passages into vector space.",
    "Hybrid retrieval combines dense vector search with sparse BM25 using Reciprocal Rank Fusion.",
    "The chunking strategy determines the granularity of retrieval units in a RAG pipeline.",
    "Cross-encoder rerankers score query-passage pairs jointly, capturing complex interactions.",
    "FAISS provides efficient approximate nearest neighbour search at billion-scale corpora.",
    "GraphRAG builds knowledge graphs from documents to enable multi-hop reasoning queries.",
    "HyDE generates a hypothetical answer and uses its embedding as the retrieval query vector.",
    "Context compression reduces retrieved passage length while preserving relevant content.",
    "Vector databases store embeddings with metadata for filtered semantic search.",
    "RAGAS evaluates RAG systems using faithfulness, context precision, and answer relevance.",
    "Embedding models map text to dense vectors using contrastive learning objectives.",
    "Query expansion generates multiple query variants to increase retrieval recall.",
    "The inverted index maps each term to the list of documents containing that term.",
]

def tokenise(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25:
    def __init__(self, corpus):
        self.corpus = corpus
        self.N = len(corpus)
        self.tokenised = [tokenise(d) for d in corpus]
        self.avgdl = sum(len(d) for d in self.tokenised) / self.N

        # Build inverted index: term -> {doc_id: term_freq}
        self.inv_idx = defaultdict(dict)
        self.df = Counter()   # document frequency per term
        for i, tokens in enumerate(self.tokenised):
            tf = Counter(tokens)
            for term, freq in tf.items():
                self.inv_idx[term][i] = freq
                self.df[term] += 1

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_terms, doc_id):
        doc_len = len(self.tokenised[doc_id])
        score = 0.0
        for term in query_terms:
            if doc_id not in self.inv_idx.get(term, {}):
                continue
            tf  = self.inv_idx[term][doc_id]
            idf = self.idf(term)
            norm_tf = tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len / self.avgdl))
            score += idf * norm_tf
        return score

    def retrieve(self, query, top_k=5):
        q_terms = tokenise(query)
        # Only score documents that contain at least one query term
        candidate_docs = set()
        for term in q_terms:
            candidate_docs.update(self.inv_idx.get(term, {}).keys())
        scored = [(self.score(q_terms, doc_id), doc_id) for doc_id in candidate_docs]
        scored.sort(reverse=True)
        return [(score, self.corpus[doc_id], doc_id) for score, doc_id in scored[:top_k]]

bm25 = BM25(CORPUS)

queries = [
    "BM25 TF-IDF retrieval scoring",
    "graph neural network nearest neighbour",
    "how to evaluate a RAG pipeline",
    "embedding models vector space contrastive",
    "compression of retrieved passages context",
]

print("=" * 70)
print("BM25 Sparse Retrieval Results")
print("=" * 70)
print(f"  Corpus: {len(CORPUS)} documents | k1={K1} | b={B}")
print()

for query in queries:
    results = bm25.retrieve(query, top_k=3)
    print(f"  Query: '{query}'")
    for rank, (score, text, doc_id) in enumerate(results, 1):
        preview = text[:62] + ("..." if len(text) > 62 else "")
        print(f"    {rank}. [{score:5.3f}] {preview}")
    print()

# ── IDF analysis ─────────────────────────────────────────────────────────────
print("=" * 70)
print("IDF Analysis: Most Discriminative Terms")
print("=" * 70)
all_terms = set()
for doc in bm25.tokenised:
    all_terms.update(doc)

term_idfs = [(bm25.idf(t), t) for t in all_terms if len(t) > 3]
term_idfs.sort(reverse=True)

print(f"  {'Term':<22} {'IDF':>8}  {'DocFreq':>9}")
print("  " + "-" * 44)
for idf, term in term_idfs[:12]:
    df = bm25.df.get(term, 0)
    bar = "=" * int(idf * 5)
    print(f"  {term:<22} {idf:>8.3f}  {df:>9}  {bar}")
""",
    },

    "3 · Dense Retrieval with Cosine Similarity": {
        "description": (
            "Simulate dense retrieval: create deterministic mock embeddings, build a "
            "vector index, compute cosine similarity, and retrieve semantically relevant chunks."
        ),
        "language": "python",
        "code": """\
import math, hashlib

# ── Deterministic mock embedding (no external deps) ──────────────────────────
DIM = 32  # reduced from real 1536 for simulation

def text_to_embedding(text, dim=DIM):
    \"\"\"
    Deterministic pseudo-embedding: hash-based projection.
    Words with similar meaning share hash buckets because we seed on
    normalised lowercase tokens, giving consistent semantic clustering.
    \"\"\"
    tokens = text.lower().split()
    vec = [0.0] * dim
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(dim):
            bit = (h >> (i % 128)) & 1
            vec[i] += 1.0 if bit else -1.0
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b))

# ── Document store ────────────────────────────────────────────────────────────
CHUNKS = [
    ("C00", "HNSW graph index enables fast approximate nearest neighbour search for dense vectors."),
    ("C01", "BM25 keyword scoring uses inverse document frequency and term frequency saturation."),
    ("C02", "Semantic embeddings capture meaning; cosine similarity measures vector proximity."),
    ("C03", "Hybrid retrieval fuses dense vector search and sparse BM25 with rank fusion algorithms."),
    ("C04", "Cross-encoder reranking scores query and passage jointly for high-precision results."),
    ("C05", "Document chunking splits text into retrieval units; overlap prevents boundary loss."),
    ("C06", "GraphRAG constructs knowledge graphs for multi-hop reasoning across document collections."),
    ("C07", "RAGAS metrics measure faithfulness, context precision, recall, and answer relevance."),
    ("C08", "HyDE hypothetical document embeddings improve retrieval recall by 5 to 15 percent."),
    ("C09", "Vector databases like Pinecone, Qdrant, and Weaviate store embeddings with metadata."),
    ("C10", "Query expansion generates multiple paraphrases to increase retrieval coverage."),
    ("C11", "Parent-child chunking retrieves small chunks but serves large parent chunks to the LLM."),
    ("C12", "Contrastive learning trains embedding models to push relevant pairs close together."),
    ("C13", "Context compression removes irrelevant sentences from retrieved passages before generation."),
    ("C14", "Iterative retrieval issues follow-up queries based on partial answers generated so far."),
]

# Build index
print("Building dense vector index...")
index = [(cid, text, text_to_embedding(text)) for cid, text in CHUNKS]
print(f"  Indexed {len(index)} chunks at {DIM}D embeddings")
print()

def dense_retrieve(query, top_k=5):
    q_vec = text_to_embedding(query)
    scored = [(cosine_sim(q_vec, emb), cid, text) for cid, text, emb in index]
    scored.sort(reverse=True)
    return scored[:top_k]

# ── Run queries ───────────────────────────────────────────────────────────────
queries = [
    "how does approximate nearest neighbour search work",
    "combining keyword and semantic search",
    "evaluating the quality of a RAG system",
    "improving retrieval recall with query techniques",
    "compressing context before sending to the language model",
]

print("=" * 70)
print("Dense Retrieval Results (Cosine Similarity)")
print("=" * 70)

for query in queries:
    results = dense_retrieve(query, top_k=3)
    print(f"  Query: '{query}'")
    for rank, (score, cid, text) in enumerate(results, 1):
        preview = text[:60] + ("..." if len(text) > 60 else "")
        bar = "=" * int(score * 20)
        print(f"    {rank}. [{cid}] {score:.4f} {bar}")
        print(f"         {preview}")
    print()

# ── Embedding space distances ─────────────────────────────────────────────────
print("=" * 70)
print("Semantic Distance Matrix (cos_sim between selected chunks):")
print("=" * 70)

selected = [("Dense Retrieval", "C02"), ("BM25 Sparse", "C01"),
            ("Hybrid", "C03"),          ("Reranking", "C04")]

vecs = {label: text_to_embedding(dict(CHUNKS)[cid]) for label, cid in selected}
print(f"  {'':22} " + " ".join(f"{label[:10]:>12}" for label, _ in selected))
for la, _ in selected:
    row = "  " + f"{la:<22}"
    for lb, _ in selected:
        sim = cosine_sim(vecs[la], vecs[lb])
        row += f" {sim:12.4f}"
    print(row)
""",
    },

    "4 · Hybrid Retrieval with RRF Fusion": {
        "description": (
            "Combine dense and sparse retrieval using Reciprocal Rank Fusion. "
            "Show rank-level fusion, score independence, and quality vs single-method retrieval."
        ),
        "language": "python",
        "code": """\
import math, re, hashlib
from collections import defaultdict, Counter

# ── reuse BM25 and Dense from previous ops ────────────────────────────────────
K1_BM25, B_BM25 = 1.5, 0.75
DIM = 32
RRF_K = 60  # RRF smoothing constant

def tokenise(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def embed(text, dim=DIM):
    tokens = text.lower().split()
    vec = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(dim):
            bit = (h >> (i % 128)) & 1
            vec[i] += 1.0 if bit else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b))

CORPUS = [
    ("C00", "HNSW graph enables fast approximate nearest neighbour dense vector search."),
    ("C01", "BM25 keyword scoring uses IDF and term frequency with length normalisation."),
    ("C02", "Dense semantic embeddings and cosine similarity for semantic retrieval."),
    ("C03", "Hybrid search fuses dense and sparse ranked lists using reciprocal rank fusion."),
    ("C04", "Cross-encoder reranking jointly scores query and passage for high precision."),
    ("C05", "Document chunking splits text; fixed-size overlap prevents boundary loss."),
    ("C06", "GraphRAG builds knowledge graphs for multi-hop reasoning and global synthesis."),
    ("C07", "RAGAS evaluates faithfulness context precision recall and answer relevance."),
    ("C08", "HyDE generates a hypothetical passage whose embedding improves recall."),
    ("C09", "Vector databases like Pinecone Qdrant Weaviate store embeddings with metadata filters."),
    ("C10", "Query expansion paraphrases the question to increase retrieval coverage."),
    ("C11", "Parent child chunking indexes small chunks but returns larger parent chunks."),
]

# Build BM25 index
inv_idx = defaultdict(dict)
df = Counter()
tokenised = [tokenise(text) for _, text in CORPUS]
avgdl = sum(len(t) for t in tokenised) / len(tokenised)
N = len(CORPUS)
for i, tokens in enumerate(tokenised):
    tf = Counter(tokens)
    for term, freq in tf.items():
        inv_idx[term][i] = freq
        df[term] += 1

def bm25_score(q_terms, doc_id):
    dl = len(tokenised[doc_id])
    score = 0.0
    for term in q_terms:
        if doc_id not in inv_idx.get(term, {}):
            continue
        tf  = inv_idx[term][doc_id]
        idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
        norm_tf = tf * (K1_BM25 + 1) / (tf + K1_BM25 * (1 - B_BM25 + B_BM25 * dl / avgdl))
        score += idf * norm_tf
    return score

# Build dense index
dense_index = [(cid, text, embed(text)) for cid, text in CORPUS]

def sparse_retrieve(query, top_k=10):
    q_terms = tokenise(query)
    cands = set()
    for t in q_terms:
        cands.update(inv_idx.get(t, {}).keys())
    scored = [(bm25_score(q_terms, i), i) for i in cands]
    scored.sort(reverse=True)
    return [CORPUS[i][0] for _, i in scored[:top_k]]

def dense_retrieve(query, top_k=10):
    qv = embed(query)
    scored = [(cosine_sim(qv, emb), cid) for cid, _, emb in dense_index]
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:top_k]]

def rrf_fusion(ranked_lists, k=RRF_K):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieve(query, top_k=5):
    sparse_list = sparse_retrieve(query, top_k=10)
    dense_list  = dense_retrieve(query, top_k=10)
    fused       = rrf_fusion([sparse_list, dense_list])
    return [(score, cid, dict(CORPUS)[cid]) for cid, score in fused[:top_k]]

# ── Comparison ────────────────────────────────────────────────────────────────
print("=" * 70)
print("Hybrid Retrieval with Reciprocal Rank Fusion")
print(f"  RRF k={RRF_K}  |  Dense DIM={DIM}  |  BM25 k1={K1_BM25} b={B_BM25}")
print("=" * 70)

test_cases = [
    ("semantic vector nearest neighbour",  "C00"),  # dense wins
    ("BM25 keyword IDF frequency",         "C01"),  # sparse wins
    ("hybrid fusion reciprocal rank",      "C03"),  # both
    ("parent child chunking context",      "C11"),  # semantic
]

for query, expected in test_cases:
    sp_list   = sparse_retrieve(query, 5)
    dn_list   = dense_retrieve(query, 5)
    hy_list   = hybrid_retrieve(query, 5)
    hy_ids    = [cid for _, cid, _ in hy_list]

    sp_hit = expected in sp_list[:5]
    dn_hit = expected in dn_list[:5]
    hy_hit = expected in hy_ids[:5]

    print(f"  Query: '{query}'")
    print(f"    Expected: {expected}")
    print(f"    Sparse top-5: {sp_list[:5]}  hit={sp_hit}")
    print(f"    Dense  top-5: {dn_list[:5]}  hit={dn_hit}")
    print(f"    Hybrid top-5: {hy_ids[:5]}  hit={hy_hit}")
    print()

# ── RRF score mechanics ───────────────────────────────────────────────────────
print("=" * 70)
print("RRF Score Mechanics: same doc at different ranks")
print(f"  score = 1 / (k + rank)  with k={RRF_K}")
print()
print(f"  {'Rank in List 1':>18} {'Rank in List 2':>16} {'RRF Score':>12}")
print("  " + "-" * 50)
for r1, r2 in [(1, 1), (1, 3), (1, 10), (3, 3), (5, 5), (10, 10), (1, 100)]:
    score = 1 / (RRF_K + r1) + 1 / (RRF_K + r2)
    bar = "=" * int(score * 2000)
    print(f"  {r1:>18} {r2:>16} {score:>12.6f}  {bar}")
""",
    },

    "5 · HNSW Index Simulation": {
        "description": (
            "Simulate HNSW graph construction and greedy search: build multi-layer "
            "graph, traverse from coarse to fine, and measure recall vs exact search."
        ),
        "language": "python",
        "code": """\
import math, random, hashlib

random.seed(42)
DIM = 8  # low-dim for clarity

def rand_vec(seed):
    r = random.Random(seed)
    v = [r.gauss(0, 1) for _ in range(DIM)]
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

def cosine_sim(a, b):
    return sum(x*y for x, y in zip(a, b))

# ── HNSW parameters ───────────────────────────────────────────────────────────
N_NODES   = 60    # corpus size
M         = 4     # connections per node per layer
M_L0      = 8     # connections at layer 0 (denser)
ML        = 1 / math.log(M)  # level multiplier
MAX_LEVEL = 3

class HNSWIndex:
    def __init__(self, m=M, m_l0=M_L0, max_level=MAX_LEVEL):
        self.m         = m
        self.m_l0      = m_l0
        self.max_level = max_level
        self.vectors   = {}       # node_id -> vector
        self.layers    = [{} for _ in range(max_level + 1)]  # layer -> node -> [neighbours]
        self.node_levels = {}     # node_id -> max_level
        self.entry_point = None
        self.ep_level    = -1

    def _random_level(self, seed):
        r = random.Random(seed)
        level = 0
        while r.random() < (1 / self.m) and level < self.max_level:
            level += 1
        return level

    def insert(self, node_id, vector):
        self.vectors[node_id] = vector
        level = self._random_level(node_id)
        self.node_levels[node_id] = level

        for l in range(level + 1):
            if node_id not in self.layers[l]:
                self.layers[l][node_id] = []

        if self.entry_point is None:
            self.entry_point = node_id
            self.ep_level    = level
            return

        # Greedy search from entry point down to level+1
        ep = self.entry_point
        for l in range(self.ep_level, level, -1):
            # Move closer at this layer
            changed = True
            while changed:
                changed = False
                for nb in self.layers[l].get(ep, []):
                    if dist(vector, self.vectors[nb]) < dist(vector, self.vectors[ep]):
                        ep = nb
                        changed = True

        # Insert with connections at levels 0 .. level
        for l in range(min(level, self.ep_level) + 1):
            candidates = list(self.layers[l].keys())
            if not candidates:
                continue
            # Find M nearest at this layer
            scored = sorted(candidates, key=lambda n: dist(vector, self.vectors[n]))
            max_conn = self.m_l0 if l == 0 else self.m
            neighbours = scored[:max_conn]
            self.layers[l][node_id] = neighbours
            # Reciprocal connections
            for nb in neighbours:
                if node_id not in self.layers[l][nb]:
                    self.layers[l][nb].append(node_id)

        if level > self.ep_level:
            self.entry_point = node_id
            self.ep_level    = level

    def search(self, query, top_k=5):
        if self.entry_point is None:
            return []
        ep = self.entry_point
        nodes_visited = 0

        for l in range(self.ep_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for nb in self.layers[l].get(ep, []):
                    nodes_visited += 1
                    if dist(query, self.vectors[nb]) < dist(query, self.vectors[ep]):
                        ep = nb
                        changed = True

        # Layer 0: beam search
        candidates = {ep}
        visited    = {ep}
        dynamic_ef = max(top_k * 3, 20)

        for _ in range(dynamic_ef):
            if not candidates:
                break
            best = min(candidates, key=lambda n: dist(query, self.vectors[n]))
            candidates.remove(best)
            for nb in self.layers[0].get(best, []):
                if nb not in visited:
                    visited.add(nb)
                    candidates.add(nb)
                    nodes_visited += 1

        all_visited = sorted(visited, key=lambda n: dist(query, self.vectors[n]))
        results = [(cosine_sim(query, self.vectors[n]), n) for n in all_visited[:top_k]]
        results.sort(reverse=True)
        return results, nodes_visited

# ── Build index ────────────────────────────────────────────────────────────────
hnsw = HNSWIndex()
nodes = [(i, rand_vec(i)) for i in range(N_NODES)]
for node_id, vec in nodes:
    hnsw.insert(node_id, vec)

# Layer statistics
print("=" * 60)
print(f"HNSW Index: {N_NODES} nodes, M={M}, M_L0={M_L0}")
print("=" * 60)
print()
print("  Layer statistics:")
for l in range(MAX_LEVEL + 1):
    n_nodes = len(hnsw.layers[l])
    n_edges = sum(len(nb) for nb in hnsw.layers[l].values())
    avg_deg = n_edges / n_nodes if n_nodes else 0
    bar = "=" * n_nodes
    print(f"  Layer {l}: {n_nodes:3d} nodes | avg degree {avg_deg:.1f} | {bar}")

# ── Recall experiment ─────────────────────────────────────────────────────────
print()
print("  Recall@K vs Exact Search (100 queries):")
print()

n_queries   = 100
all_vecs    = [v for _, v in nodes]

def exact_search(query, top_k):
    scored = sorted(range(len(all_vecs)), key=lambda i: dist(query, all_vecs[i]))
    return set(scored[:top_k])

total_recall = {5: 0, 10: 0}
total_visited = 0

for q_seed in range(1000, 1000 + n_queries):
    q = rand_vec(q_seed * 7 + 13)
    hnsw_results, visited = hnsw.search(q, top_k=10)
    hnsw_ids = {n for _, n in hnsw_results}
    exact_5  = exact_search(q, 5)
    exact_10 = exact_search(q, 10)
    total_recall[5]  += len(hnsw_ids & exact_5) / 5
    total_recall[10] += len(hnsw_ids & exact_10) / 10
    total_visited    += visited

avg_r5  = total_recall[5]  / n_queries
avg_r10 = total_recall[10] / n_queries
avg_vis = total_visited / n_queries

print(f"  Recall@5:   {avg_r5:.3f}  (fraction of true top-5 found by HNSW)")
print(f"  Recall@10:  {avg_r10:.3f}  (fraction of true top-10 found by HNSW)")
print(f"  Avg nodes visited per query: {avg_vis:.1f} / {N_NODES} ({avg_vis/N_NODES:.1%})")
print()
print("  HNSW achieves high recall while visiting a fraction of all nodes.")
print("  Production indexes (M=16, ef=128) achieve >0.99 recall@10.")
""",
    },

    "6 · Cross-Encoder Reranking": {
        "description": (
            "Simulate two-stage retrieval: first-stage ANN recall, then cross-encoder "
            "reranking for precision. Show NDCG improvement and score distribution."
        ),
        "language": "python",
        "code": """\
import math, random, hashlib, re
from collections import defaultdict, Counter

random.seed(42)
DIM = 16

def embed(text, dim=DIM):
    tokens = text.lower().split()
    vec = [0.0] * dim
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(dim):
            bit = (h >> (i % 128)) & 1
            vec[i] += 1.0 if bit else -1.0
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/norm for v in vec]

def cosine_sim(a, b):
    return sum(x*y for x, y in zip(a, b))

# ── Corpus with relevance labels ─────────────────────────────────────────────
# (doc_id, text, relevance_grade_0-3)
QUERY = "How does HNSW approximate nearest neighbour search work?"

CORPUS_LABELLED = [
    ("D00", "HNSW constructs a multi-layer navigable small world graph for fast ANN search.", 3),
    ("D01", "Approximate nearest neighbour algorithms trade small recall loss for speed.", 2),
    ("D02", "HNSW parameters include ef_construction, M connections, and ef_search beam width.", 3),
    ("D03", "Graph-based indexes like HNSW outperform IVF at high recall thresholds.", 2),
    ("D04", "BM25 is a probabilistic keyword scoring function unrelated to graph indexes.", 0),
    ("D05", "Vector databases store high-dimensional embeddings for semantic search.", 1),
    ("D06", "HNSW greedy traversal starts at entry point and descends layer by layer.", 3),
    ("D07", "Product quantisation compresses vectors for memory efficiency.", 1),
    ("D08", "Cosine similarity between query and document embeddings drives dense retrieval.", 1),
    ("D09", "The IVF index partitions the vector space into K clusters via k-means.", 1),
    ("D10", "FAISS implements both IVF and HNSW indexes for billion-scale ANN search.", 2),
    ("D11", "Cross-encoder rerankers score query-document pairs jointly.", 0),
    ("D12", "HNSW insertion assigns random levels using a geometric distribution.", 2),
    ("D13", "RAG pipelines use vector search for first-stage retrieval.", 1),
    ("D14", "Hierarchical graphs in HNSW enable logarithmic search complexity.", 3),
]

corpus = [(did, text) for did, text, _ in CORPUS_LABELLED]
relevance = {did: grade for did, _, grade in CORPUS_LABELLED}

# ── First-stage: bi-encoder (dense) ──────────────────────────────────────────
def biencoder_retrieve(query, corpus, top_k=10):
    qv = embed(query)
    scored = [(cosine_sim(qv, embed(text)), did, text) for did, text in corpus]
    scored.sort(reverse=True)
    return scored[:top_k]

# ── Second-stage: cross-encoder (simulated) ──────────────────────────────────
def cross_encoder_score(query, text):
    \"\"\"
    Simulate cross-encoder: uses exact keyword overlap + semantic embedding,
    weighted toward exact match (cross-encoders reward term overlap more than bi-encoders).
    \"\"\"
    q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
    d_toks = re.findall(r"[a-z0-9]+", text.lower())
    d_tok_set = set(d_toks)

    # Exact overlap fraction (cross-encoder strong signal)
    overlap = len(q_toks & d_tok_set) / max(len(q_toks), 1)

    # Semantic embedding
    sem = cosine_sim(embed(query), embed(text))

    # Cross-encoder: 60% keyword overlap, 40% semantic
    base_score = 0.6 * overlap + 0.4 * sem

    # Add noise to simulate model variance
    seed = hash(query + text) % (2**31)
    r = random.Random(seed)
    return min(1.0, max(0.0, base_score + r.gauss(0, 0.05)))

# ── NDCG@K ────────────────────────────────────────────────────────────────────
def dcg(rankings, k, relevance_dict):
    score = 0.0
    for rank, did in enumerate(rankings[:k], 1):
        rel = relevance_dict.get(did, 0)
        score += (2**rel - 1) / math.log2(rank + 1)
    return score

def ndcg(rankings, k, relevance_dict):
    ideal = sorted(relevance_dict.values(), reverse=True)
    ideal_rankings = [did for did, _ in sorted(relevance_dict.items(), key=lambda x: x[1], reverse=True)]
    ideal_dcg = dcg(ideal_rankings, k, relevance_dict)
    if ideal_dcg == 0:
        return 0.0
    return dcg(rankings, k, relevance_dict) / ideal_dcg

# ── Run two-stage pipeline ────────────────────────────────────────────────────
stage1_results = biencoder_retrieve(QUERY, corpus, top_k=10)
stage1_ranked  = [did for _, did, _ in stage1_results]

# Rerank top-10 with cross-encoder
reranked = [(cross_encoder_score(QUERY, text), did, text)
            for _, did, text in stage1_results]
reranked.sort(reverse=True)
stage2_ranked = [did for _, did, _ in reranked]

print("=" * 68)
print(f"Query: '{QUERY}'")
print("=" * 68)
print()
print(f"  {'Rank':>5} {'Stage1 (Bi-Enc)':>20} {'Rel':>5} | {'Stage2 (Reranked)':>20} {'Rel':>5}")
print("  " + "-" * 60)
for i in range(min(8, len(stage1_ranked))):
    d1  = stage1_ranked[i]
    d2  = stage2_ranked[i] if i < len(stage2_ranked) else "-"
    r1  = relevance.get(d1, 0)
    r2  = relevance.get(d2, 0)
    marker = " <--" if d1 != d2 else ""
    print(f"  {i+1:>5} {d1:>20}  {'*'*r1:<5} | {d2:>20}  {'*'*r2:<5}{marker}")

print()
for k in [3, 5, 10]:
    n1 = ndcg(stage1_ranked, k, relevance)
    n2 = ndcg(stage2_ranked, k, relevance)
    gain = (n2 - n1) / n1 * 100 if n1 else 0
    bar = "+" * int(abs(gain) / 2)
    print(f"  NDCG@{k}:  Bi-encoder={n1:.4f}  Reranked={n2:.4f}  gain={gain:+.1f}%  {bar}")

print()
print("  Reranker improves NDCG by jointly scoring query+passage interactions.")
print("  Two-stage: Stage1 high recall (ANN), Stage2 high precision (reranker).")
""",
    },

    "7 · Naive RAG Pipeline End-to-End": {
        "description": (
            "Build a minimal but complete RAG pipeline: index documents, retrieve top-K, "
            "assemble a prompt, and simulate generation with grounded answers and citations."
        ),
        "language": "python",
        "code": """\
import math, re, hashlib, random
from collections import defaultdict, Counter

random.seed(0)
DIM = 24

def tokenise(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def embed(text, dim=DIM):
    vec = [0.0] * dim
    for tok in tokenise(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(dim):
            vec[i] += 1.0 if (h >> (i % 128)) & 1 else -1.0
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/norm for v in vec]

def cosine_sim(a, b):
    return sum(x*y for x, y in zip(a, b))

# ── knowledge base ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {"id": "KB-01", "source": "rag_paper_2020",
     "text": "RAG was introduced by Lewis et al. in 2020, combining parametric and non-parametric memory."},
    {"id": "KB-02", "source": "rag_paper_2020",
     "text": "The retrieval component in RAG uses Maximum Inner Product Search over dense document embeddings."},
    {"id": "KB-03", "source": "hnsw_paper_2018",
     "text": "HNSW achieves recall@1 above 0.99 on standard benchmarks with query times under 1 millisecond."},
    {"id": "KB-04", "source": "bm25_paper_1994",
     "text": "BM25 Okapi scoring uses term frequency saturation with parameter k1 and length normalisation with b."},
    {"id": "KB-05", "source": "hyde_paper_2022",
     "text": "HyDE (Hypothetical Document Embeddings) generates a fake answer and embeds it as the query vector."},
    {"id": "KB-06", "source": "advanced_rag_survey_2024",
     "text": "Advanced RAG adds pre-retrieval query transformation, retrieval fusion, and post-retrieval compression."},
    {"id": "KB-07", "source": "graphrag_paper_2024",
     "text": "GraphRAG extracts entity-relation triples to build a knowledge graph for multi-hop reasoning."},
    {"id": "KB-08", "source": "ragas_paper_2023",
     "text": "RAGAS measures faithfulness as the fraction of answer claims supported by the retrieved context."},
    {"id": "KB-09", "source": "reranking_survey_2023",
     "text": "Cross-encoder rerankers improve NDCG@10 by 10-20 percent over bi-encoder first-stage retrieval."},
    {"id": "KB-10", "source": "chunking_blog_2024",
     "text": "Parent-child chunking indexes 128-token child chunks but returns 512-token parent chunks to the LLM."},
    {"id": "KB-11", "source": "rrf_paper_2009",
     "text": "Reciprocal Rank Fusion with k=60 outperforms linear score combination across diverse retrieval tasks."},
    {"id": "KB-12", "source": "embedding_survey_2024",
     "text": "Matryoshka Representation Learning trains embeddings that remain useful at reduced vector dimensions."},
]

# ── Index ─────────────────────────────────────────────────────────────────────
print("Indexing knowledge base...")
indexed = [(doc, embed(doc["text"])) for doc in KNOWLEDGE_BASE]
print(f"  Indexed {len(indexed)} chunks")
print()

def retrieve(query, top_k=3):
    qv = embed(query)
    scored = [(cosine_sim(qv, ev), doc) for doc, ev in indexed]
    scored.sort(reverse=True)
    return scored[:top_k]

def assemble_prompt(query, retrieved_docs):
    NL = chr(10)
    context_parts = []
    for i, (score, doc) in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] Source: {doc['source']} (id={doc['id']})")
        context_parts.append(f"    {doc['text']}")
    context = NL.join(context_parts)
    prompt = (
        "You are a helpful AI assistant. Answer the question using ONLY the provided context. "
        + "Cite sources using [n] notation." + NL + NL
        + "CONTEXT:" + NL + context + NL + NL
        + "QUESTION: " + query + NL + NL
        + "ANSWER:"
    )
    return prompt

def simulate_generation(query, retrieved_docs):
    \"\"\"
    Simulate LLM generation: extract key facts from retrieved docs
    and compose an answer with citation markers.
    \"\"\"
    facts = []
    for i, (score, doc) in enumerate(retrieved_docs, 1):
        # Use first sentence of each doc as a key fact
        fact = doc["text"].rstrip(".")
        facts.append(f"{fact} [{i}].")
    answer = " ".join(facts[:2])  # use top-2 retrieved facts
    return answer

# ── Run pipeline on several queries ──────────────────────────────────────────
queries = [
    "What is HyDE and how does it improve retrieval?",
    "How does cross-encoder reranking compare to bi-encoder retrieval?",
    "What algorithm does HNSW use for fast approximate search?",
    "How does RAG evaluate answer quality with RAGAS?",
]

print("=" * 70)
print("Naive RAG Pipeline: End-to-End")
print("=" * 70)

for query in queries:
    print()
    print(f"  Q: {query}")
    print("  " + "-" * 60)
    retrieved = retrieve(query, top_k=3)

    for rank, (score, doc) in enumerate(retrieved, 1):
        preview = doc["text"][:58] + ("..." if len(doc["text"]) > 58 else "")
        print(f"  [{rank}] {doc['id']} score={score:.4f}  {preview}")

    answer = simulate_generation(query, retrieved)
    print()
    print(f"  Answer: {answer}")

# ── Prompt structure inspection ───────────────────────────────────────────────
print()
print("=" * 70)
print("Assembled Prompt Structure (first query)")
print("=" * 70)
sample_q   = queries[0]
sample_ret = retrieve(sample_q, top_k=3)
prompt     = assemble_prompt(sample_q, sample_ret)
for line in prompt.split(chr(10))[:20]:
    print(f"  {line}")

ctx_tokens = len(tokenise(prompt))
print(f"  Prompt token estimate: {ctx_tokens}")
print(f"  Retrieved context tokens: {sum(len(tokenise(doc['text'])) for _, doc in sample_ret)}")
""",
    },

    "8 · Advanced RAG — HyDE + Query Expansion": {
        "description": (
            "Implement HyDE (Hypothetical Document Embeddings) and multi-query expansion, "
            "show recall improvements over naive single-query dense retrieval."
        ),
        "language": "python",
        "code": """\
import math, random, hashlib, re
from collections import defaultdict

random.seed(42)
DIM = 32

def embed(text, dim=DIM):
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(dim):
            vec[i] += 1.0 if (h >> (i % 128)) & 1 else -1.0
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/norm for v in vec]

def cosine_sim(a, b):
    return sum(x*y for x, y in zip(a, b))

def avg_vec(vecs):
    if not vecs:
        return [0.0] * DIM
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(DIM)]

# ── Corpus with ground-truth relevance ────────────────────────────────────────
CORPUS = [
    ("P00", "HyDE generates a hypothetical answer passage and uses its embedding as the query vector."),
    ("P01", "Hypothetical document embeddings close the semantic gap between queries and answer passages."),
    ("P02", "Query expansion generates multiple paraphrased questions to increase retrieval coverage."),
    ("P03", "Multi-query retrieval unions results from several query variants reducing missed documents."),
    ("P04", "Step-back prompting abstracts the query to retrieve foundational contextual background."),
    ("P05", "Dense retrieval embeds query and passage independently using a bi-encoder architecture."),
    ("P06", "Cross-encoder reranking scores query and passage jointly for high precision second stage."),
    ("P07", "BM25 term frequency scoring complements semantic embedding for exact keyword matching."),
    ("P08", "FLARE iterative retrieval issues follow-up queries based on generated partial answers."),
    ("P09", "Query decomposition breaks multi-hop questions into sequential sub-questions."),
    ("P10", "Self-querying parsers extract structured metadata filters from natural language queries."),
    ("P11", "RAG context compression removes irrelevant sentences before sending to the LLM."),
    ("P12", "Parent-child chunking retrieves small precise chunks but returns larger context windows."),
    ("P13", "Contextual retrieval prepends chunk-level document summaries before embedding."),
    ("P14", "Retrieval quality is measured by recall at K, NDCG, and MRR across test queries."),
]

# Ground-truth relevant for each query
GROUND_TRUTH = {
    "How does HyDE improve retrieval over standard dense search?": {"P00", "P01", "P05"},
    "What techniques expand query coverage?":                       {"P02", "P03", "P04", "P09"},
    "How can we improve retrieval recall without retraining?":      {"P00", "P02", "P03", "P08", "P13"},
}

indexed = [(pid, text, embed(text)) for pid, text in CORPUS]

def retrieve(query_vec, top_k=5):
    scored = [(cosine_sim(query_vec, ev), pid) for pid, _, ev in indexed]
    scored.sort(reverse=True)
    return {pid for _, pid in scored[:top_k]}

# ── Simulated HyDE hypothetical documents ─────────────────────────────────────
HYDE_DOCS = {
    "How does HyDE improve retrieval over standard dense search?": (
        "HyDE improves retrieval by generating a hypothetical passage that looks like "
        "an answer to the query. This hypothetical passage embedding is closer in vector "
        "space to real answer passages than the original question embedding is, "
        "closing the query-document semantic gap in dense retrieval."
    ),
    "What techniques expand query coverage?": (
        "Query coverage is expanded through paraphrase generation, multi-query expansion, "
        "step-back abstraction, and decomposition of complex multi-hop questions into "
        "sequential sub-questions that are each retrieved independently."
    ),
    "How can we improve retrieval recall without retraining?": (
        "Retrieval recall can be improved without retraining by using HyDE hypothetical "
        "documents, multi-query expansion, contextual chunk enrichment, iterative FLARE "
        "retrieval, and parent-child chunking strategies."
    ),
}

# ── Query expansion variants ──────────────────────────────────────────────────
EXPANSIONS = {
    "How does HyDE improve retrieval over standard dense search?": [
        "hypothetical document embedding query vector representation",
        "closing semantic gap between questions and answer passages",
        "HyDE generated fake answer retrieval technique",
    ],
    "What techniques expand query coverage?": [
        "multiple query paraphrase retrieval union results",
        "step back prompting abstract background context retrieval",
        "query decomposition sub-question retrieval multi-hop",
    ],
    "How can we improve retrieval recall without retraining?": [
        "improve recall retrieval techniques without model training",
        "chunk contextual enrichment embedding quality recall",
        "iterative query reformulation active retrieval recall",
    ],
}

print("=" * 70)
print("Advanced RAG: HyDE + Query Expansion vs Naive Retrieval")
print("=" * 70)

for query, ground_truth in GROUND_TRUTH.items():
    print()
    print(f"  Query: {query[:60]}...")
    print(f"  Ground truth: {sorted(ground_truth)}")
    print()

    # Naive: embed query directly
    naive_vec   = embed(query)
    naive_hits  = retrieve(naive_vec, top_k=5)
    naive_prec  = len(naive_hits & ground_truth) / 5
    naive_rec   = len(naive_hits & ground_truth) / len(ground_truth)

    # HyDE: embed hypothetical document
    hyde_text   = HYDE_DOCS[query]
    hyde_vec    = embed(hyde_text)
    hyde_hits   = retrieve(hyde_vec, top_k=5)
    hyde_prec   = len(hyde_hits & ground_truth) / 5
    hyde_rec    = len(hyde_hits & ground_truth) / len(ground_truth)

    # Multi-query expansion: union of K query variants
    all_hits = set(naive_hits)
    for variant in EXPANSIONS[query]:
        all_hits |= retrieve(embed(variant), top_k=5)
    expand_hits = all_hits
    expand_prec = len(expand_hits & ground_truth) / max(len(expand_hits), 1)
    expand_rec  = len(expand_hits & ground_truth) / len(ground_truth)

    # HyDE + Expansion combined: average embedding
    all_vecs  = [hyde_vec] + [embed(v) for v in EXPANSIONS[query]]
    combo_vec = avg_vec(all_vecs)
    combo_hits = retrieve(combo_vec, top_k=5)
    combo_prec = len(combo_hits & ground_truth) / 5
    combo_rec  = len(combo_hits & ground_truth) / len(ground_truth)

    rows = [
        ("Naive (direct embed)", naive_hits,  naive_prec,  naive_rec),
        ("HyDE",                 hyde_hits,   hyde_prec,   hyde_rec),
        ("Multi-query expand",   expand_hits, expand_prec, expand_rec),
        ("HyDE + Expansion",     combo_hits,  combo_prec,  combo_rec),
    ]
    print(f"  {'Method':<25} {'Retrieved':>20} {'Prec':>7} {'Recall':>8}")
    print("  " + "-" * 64)
    for method, hits, prec, rec in rows:
        hits_str = str(sorted(hits))[:22]
        print(f"  {method:<25} {hits_str:>20} {prec:>6.2f} {rec:>7.2f}")
""",
    },

    "9 · GraphRAG — Entity Extraction and Graph Traversal": {
        "description": (
            "Simulate GraphRAG: extract entities and relationships, build a knowledge graph, "
            "cluster into communities, and answer global queries via community summaries."
        ),
        "language": "python",
        "code": """\
import re, math
from collections import defaultdict

# ── Entity-Relationship extraction (rule-based simulation) ────────────────────
DOCUMENTS = [
    ("DOC-1", "Lewis et al. introduced RAG in 2020 at Facebook AI Research, combining retrieval with T5."),
    ("DOC-2", "HNSW was created by Malkov and Yashunin and achieves high recall on ANNS benchmarks."),
    ("DOC-3", "DPR (Dense Passage Retrieval) by Karpukhin et al. uses BERT for bi-encoder retrieval."),
    ("DOC-4", "Pinecone and Weaviate are cloud-native vector databases that support HNSW indexing."),
    ("DOC-5", "RAGAS by Shahul et al. evaluates RAG systems using faithfulness and context precision."),
    ("DOC-6", "LangChain integrates with Pinecone, Weaviate, Qdrant, and Chroma for RAG pipelines."),
    ("DOC-7", "GraphRAG by Microsoft extracts entities and builds community summaries from documents."),
    ("DOC-8", "HyDE by Gao et al. generates hypothetical documents for improved dense retrieval."),
    ("DOC-9", "Cohere Rerank provides a cross-encoder reranking API compatible with RAG pipelines."),
    ("DOC-10", "FAISS from Facebook supports IVF and HNSW indexes for billion-scale vector search."),
]

# ── Entity types ──────────────────────────────────────────────────────────────
PERSON_ENTITIES = {
    "Lewis", "Malkov", "Yashunin", "Karpukhin", "Shahul", "Gao"
}
TECH_ENTITIES = {
    "RAG", "HNSW", "DPR", "BERT", "T5", "RAGAS", "HyDE", "GraphRAG",
    "FAISS", "LangChain", "Pinecone", "Weaviate", "Qdrant", "Chroma"
}
ORG_ENTITIES = {
    "Facebook", "Microsoft", "Cohere"
}

def extract_entities(text):
    found = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text):
        if word in PERSON_ENTITIES:
            found.add((word, "PERSON"))
        elif word in TECH_ENTITIES:
            found.add((word, "TECH"))
        elif word in ORG_ENTITIES:
            found.add((word, "ORG"))
    return found

RELATION_PATTERNS = [
    (r"(\w+) (?:introduced|created|developed|proposed|built) (\w+)", "INTRODUCED"),
    (r"(\w+) integrates? with (\w+)", "INTEGRATES_WITH"),
    (r"(\w+) support[s]? (\w+)", "SUPPORTS"),
    (r"(\w+) (?:uses?|using) (\w+)", "USES"),
    (r"(\w+) (?:achieves?|provides?) (\w+)", "ACHIEVES"),
    (r"(\w+) (?:evaluates?) (\w+)", "EVALUATES"),
    (r"(\w+) (?:compatible with) (\w+)", "COMPATIBLE_WITH"),
]

def extract_relations(text):
    rels = []
    all_known = PERSON_ENTITIES | TECH_ENTITIES | ORG_ENTITIES
    for pattern, rel_type in RELATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            src, tgt = match.group(1), match.group(2)
            if src in all_known and tgt in all_known:
                rels.append((src, rel_type, tgt))
    return rels

# ── Build knowledge graph ─────────────────────────────────────────────────────
nodes = {}    # entity -> {"type": ..., "docs": set()}
edges = defaultdict(set)   # (src, rel, tgt) set
doc_entities = {}

print("=" * 66)
print("GraphRAG: Knowledge Graph Construction")
print("=" * 66)
print()

for doc_id, text in DOCUMENTS:
    ents  = extract_entities(text)
    rels  = extract_relations(text)
    doc_entities[doc_id] = {e for e, _ in ents}

    for entity, etype in ents:
        if entity not in nodes:
            nodes[entity] = {"type": etype, "docs": set()}
        nodes[entity]["docs"].add(doc_id)

    for src, rel_type, tgt in rels:
        edges[(src, tgt)].add(rel_type)

print(f"  Nodes extracted: {len(nodes)}")
print(f"  Edges extracted: {len(edges)}")
print()

# ── Adjacency for graph traversal ────────────────────────────────────────────
adj = defaultdict(set)
for (src, tgt) in edges:
    adj[src].add(tgt)
    adj[tgt].add(src)

# ── Community detection (simple: connected components + size-based grouping) ──
def find_communities(adj, nodes):
    visited = set()
    communities = []
    for start in nodes:
        if start in visited:
            continue
        community = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            community.append(node)
            stack.extend(adj[node] - visited)
        communities.append(community)
    return communities

communities = find_communities(adj, nodes)
communities.sort(key=len, reverse=True)

print("  Graph Communities (connected components):")
for i, comm in enumerate(communities[:5]):
    types = [nodes.get(n, {}).get("type", "?") for n in comm]
    print(f"  Community {i}: {comm[:6]} ... ({len(comm)} nodes)")

# ── Community summaries ───────────────────────────────────────────────────────
def summarise_community(community, nodes, edges):
    techs   = [n for n in community if nodes.get(n, {}).get("type") == "TECH"]
    persons = [n for n in community if nodes.get(n, {}).get("type") == "PERSON"]
    orgs    = [n for n in community if nodes.get(n, {}).get("type") == "ORG"]
    rels    = [(s, r, t) for (s, t), rs in edges.items()
               if s in set(community) and t in set(community)
               for r in rs]
    summary = f"Technologies: {', '.join(techs[:4])}. "
    if persons:
        summary += f"Researchers: {', '.join(persons[:3])}. "
    if orgs:
        summary += f"Organisations: {', '.join(orgs[:3])}. "
    if rels:
        summary += f"Key relations: {rels[0][0]} {rels[0][1]} {rels[0][2]}."
    return summary

print()
print("  Community Summaries (for global query answering):")
for i, comm in enumerate(communities[:3]):
    summary = summarise_community(comm, nodes, edges)
    print(f"  C{i}: {summary[:80]}...")

# ── Multi-hop query via graph traversal ──────────────────────────────────────
print()
print("=" * 66)
print("Multi-Hop Graph Traversal")
print("=" * 66)

def graph_query(start_entity, hops=2):
    \"\"\"BFS from start entity, collect all entities within N hops.\"\"\"
    visited = {start_entity}
    frontier = {start_entity}
    path_docs = set(nodes.get(start_entity, {}).get("docs", set()))
    for hop in range(hops):
        next_frontier = set()
        for entity in frontier:
            for nb in adj[entity]:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
                    path_docs.update(nodes.get(nb, {}).get("docs", set()))
        frontier = next_frontier
    return visited, path_docs

query_entities = ["RAG", "HNSW", "LangChain"]
for qe in query_entities:
    if qe not in nodes:
        continue
    entity_network, relevant_docs = graph_query(qe, hops=2)
    print(f"  Query entity: {qe}")
    print(f"    2-hop network: {sorted(entity_network)[:8]}")
    print(f"    Relevant docs: {sorted(relevant_docs)}")
    print()

print("  GraphRAG advantage: multi-hop traversal finds connections that")
print("  chunk RAG misses because they span multiple documents.")
""",
    },

    "10 · RAGAS Evaluation Metrics": {
        "description": (
            "Implement all four RAGAS metrics from scratch: context precision, context recall, "
            "faithfulness, and answer relevance. Evaluate a sample RAG run."
        ),
        "language": "python",
        "code": """\
import math, re, hashlib
from collections import Counter

# ── Token utilities ───────────────────────────────────────────────────────────
def tokenise(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def embed_sim(a, b):
    \"\"\"Token Jaccard similarity as embedding proxy.\"\"\"
    ta, tb = set(tokenise(a)), set(tokenise(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def f1_overlap(pred_tokens, ref_tokens):
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(pred_tokens)
    rec  = n_common / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)

# ── RAGAS Metric Implementations ──────────────────────────────────────────────

def context_precision(retrieved_chunks, relevant_chunk_ids):
    \"\"\"
    What fraction of retrieved chunks are actually relevant?
    Precision = |retrieved AND relevant| / |retrieved|
    \"\"\"
    retrieved_ids = set(c["id"] for c in retrieved_chunks)
    relevant_ids  = set(relevant_chunk_ids)
    hits = retrieved_ids & relevant_ids
    return len(hits) / len(retrieved_ids) if retrieved_ids else 0.0

def context_recall(retrieved_chunks, ground_truth_facts):
    \"\"\"
    What fraction of ground-truth facts are covered by retrieved context?
    For each fact, check if any retrieved chunk contains it (token overlap).
    \"\"\"
    if not ground_truth_facts:
        return 1.0
    context_text = " ".join(c["text"] for c in retrieved_chunks)
    covered = 0
    for fact in ground_truth_facts:
        # A fact is "covered" if its key terms appear in context
        key_terms = set(tokenise(fact)) - {"the", "a", "is", "are", "in", "of", "to", "and"}
        context_terms = set(tokenise(context_text))
        if len(key_terms & context_terms) / max(len(key_terms), 1) >= 0.6:
            covered += 1
    return covered / len(ground_truth_facts)

def faithfulness(answer, retrieved_chunks):
    \"\"\"
    Are all claims in the answer supported by the retrieved context?
    Split answer into sentences; check each against retrieved context.
    \"\"\"
    context_text = " ".join(c["text"] for c in retrieved_chunks)
    # Split answer into claims (sentences)
    claims = [s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 10]
    if not claims:
        return 1.0
    supported = 0
    for claim in claims:
        overlap = embed_sim(claim, context_text)
        if overlap >= 0.12:   # threshold for "supported"
            supported += 1
    return supported / len(claims)

def answer_relevance(answer, question):
    \"\"\"
    Is the answer relevant to the question?
    Proxy: semantic overlap between generated reverse-questions and original.
    In real RAGAS: LLM generates N questions from answer, computes avg cos_sim.
    \"\"\"
    return embed_sim(answer, question)

# ── Sample RAG runs to evaluate ───────────────────────────────────────────────
CHUNKS = [
    {"id": "C1", "text": "RAGAS measures faithfulness as the fraction of answer claims supported by context."},
    {"id": "C2", "text": "Context precision measures what fraction of retrieved chunks are relevant to the query."},
    {"id": "C3", "text": "Context recall measures what fraction of needed facts are present in retrieved context."},
    {"id": "C4", "text": "Answer relevance measures whether the generated answer addresses the original question."},
    {"id": "C5", "text": "BM25 is a keyword retrieval function unrelated to RAGAS evaluation metrics."},
    {"id": "C6", "text": "HNSW graph indexing is used for fast approximate nearest neighbour vector search."},
    {"id": "C7", "text": "RAGAS was introduced by Shahul et al. as an evaluation framework for RAG systems."},
]

EVAL_CASES = [
    {
        "question":    "What does RAGAS measure and how?",
        "answer":      "RAGAS measures faithfulness, context precision, context recall, and answer relevance. "
                       "Faithfulness checks if answer claims are supported by retrieved context. "
                       "RAGAS was introduced by Shahul et al. as a RAG evaluation framework.",
        "retrieved":   ["C1", "C2", "C3", "C4", "C7"],
        "relevant":    ["C1", "C2", "C3", "C4", "C7"],
        "gt_facts":    [
            "RAGAS measures faithfulness",
            "context precision measures fraction of relevant retrieved chunks",
            "context recall measures fraction of needed facts in context",
            "answer relevance measures if answer addresses the question",
        ],
    },
    {
        "question":    "What does RAGAS measure and how?",
        "answer":      "RAGAS uses BM25 for retrieval and HNSW for indexing. "
                       "It evaluates whether the index is fast enough for production use.",
        "retrieved":   ["C5", "C6", "C2"],       # two irrelevant retrieved
        "relevant":    ["C1", "C2", "C3", "C4"], # correct relevant set
        "gt_facts":    [
            "RAGAS measures faithfulness",
            "context precision measures fraction of relevant retrieved chunks",
            "context recall measures fraction of needed facts in context",
            "answer relevance measures if answer addresses the question",
        ],
    },
    {
        "question":    "How does GraphRAG differ from standard chunk RAG?",
        "answer":      "GraphRAG builds knowledge graphs and community summaries enabling multi-hop "
                       "reasoning across documents, while chunk RAG retrieves text segments directly.",
        "retrieved":   ["C1", "C2"],          # irrelevant to GraphRAG
        "relevant":    [],                    # none of our chunks cover GraphRAG
        "gt_facts":    ["GraphRAG builds knowledge graphs", "multi-hop reasoning"],
    },
]

print("=" * 72)
print("RAGAS Evaluation: Four Core Metrics")
print("=" * 72)
print()
print(f"  {'Case':<8} {'Context Prec':>14} {'Context Rec':>13} {'Faithfulness':>14} {'Ans Rel':>10}")
print("  " + "-" * 64)

for i, case in enumerate(EVAL_CASES, 1):
    retrieved_chunks = [c for c in CHUNKS if c["id"] in case["retrieved"]]

    cp  = context_precision(retrieved_chunks, case["relevant"])
    cr  = context_recall(retrieved_chunks, case["gt_facts"])
    fth = faithfulness(case["answer"], retrieved_chunks)
    ar  = answer_relevance(case["answer"], case["question"])

    print(f"  Case {i}    {cp:>13.3f} {cr:>13.3f} {fth:>14.3f} {ar:>10.3f}")

print()
print("  Case 1: All metrics high (good retrieval, grounded answer)")
print("  Case 2: Low precision (bad retrieval), low recall, low faithfulness (hallucinated)")
print("  Case 3: Low precision/recall (no relevant docs indexed), lower faithfulness")
print()

# ── Metric breakdown for Case 1 ───────────────────────────────────────────────
print("=" * 72)
print("Detailed Breakdown: Case 1")
print("=" * 72)

case = EVAL_CASES[0]
retrieved_chunks = [c for c in CHUNKS if c["id"] in case["retrieved"]]
retrieved_ids = set(case["retrieved"])
relevant_ids  = set(case["relevant"])
hits = retrieved_ids & relevant_ids

print(f"  Retrieved: {sorted(retrieved_ids)}")
print(f"  Relevant:  {sorted(relevant_ids)}")
print(f"  True pos:  {sorted(hits)}")
print()
print(f"  Context Precision = {len(hits)}/{len(retrieved_ids)} = {len(hits)/len(retrieved_ids):.3f}")
print()

claims = [s.strip() for s in re.split(r"[.!?]+", case["answer"]) if len(s.strip()) > 10]
context_text = " ".join(c["text"] for c in retrieved_chunks)
print(f"  Faithfulness claim check:")
for claim in claims:
    overlap = embed_sim(claim, context_text)
    supported = overlap >= 0.12
    mark = "SUPPORTED" if supported else "UNSUPPORTED"
    print(f"    [{mark}] (overlap={overlap:.3f}) {claim[:60]}...")

print()
print("  RAGAS aggregates these per-claim checks into a 0-1 faithfulness score.")
print("  A score below 0.8 indicates the LLM is making unsupported claims.")
""",
    },
}


def get_topic_data():
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "operations":   OPERATIONS,
    }